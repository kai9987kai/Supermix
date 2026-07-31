"""MiMoMix v53 neural core: hybrid attention + sparse MoE + MTP + verified recursion.

This module is the model half of the Supermix v53 "MiMoMix" line. It fuses three
lineages into one compact, CPU-runnable PyTorch model:

* **Xiaomi MiMo (V2-Flash / V2.5-Pro) structural techniques.**
  - Hybrid attention: local sliding-window attention (SWA) interleaved with full
    global attention (GA) at a configurable ratio (MiMo-V2-Flash uses 5:1 with a
    128-token window; MiMo-V2.5-Pro uses 6:1). Only the GA layers keep an
    unbounded KV cache, which is where the ~7x KV-cache reduction comes from.
  - A learnable per-head **attention sink** bias, so a head can dump probability
    mass into a null slot instead of being forced to attend somewhere. This is
    the "off-by-one"/softmax-with-sink formulation used by StreamingLLM-style
    stabilisation and by recent open-weight models.
  - **Auxiliary-loss-free load balancing** for the sparse MoE: a per-expert
    routing bias is nudged by a sign rule so selection stays balanced without
    a gradient-carrying balance loss distorting the language objective, plus a
    router z-loss to stop router logits drifting large.
  - **Multi-Token Prediction (MTP)** modules, trained as extra causal depths and
    reusable at inference as the draft model for self-speculative decoding.
  - RoPE with a **progressive context-extension** schedule (none / NTK-aware /
    YaRN), matching the 32K-native to long-context extension pattern.

* **Supermix v51/v52 cognition.** The recurrent latent thinking core keeps
  weight-tied refinement, ACT-style halting with a ponder cost, deep supervision
  over per-cycle decodes, trainable temperature calibration, and the supervised
  quality/continue verifier that gates escalation. See
  ``source/model_variants.py::CognitiveLeapV52ExpertHead`` for the classifier-side
  ancestor whose contract this core deliberately mirrors.

* **AI-Dem-Lab instrumentation.** Every forward pass publishes a JSON-safe
  telemetry dict (router occupancy, sink mass, halting depth, calibrated
  entropy, MTP depth losses). ``mimomix_observatory.py`` turns that stream into
  novelty/stability/resonance measurements and a feedback signal.

Scope honesty: this is a small, self-contained research model. It implements the
*mechanisms* named above and tests them; it is not a reproduction of any
frontier checkpoint, and none of the numbers published by those model families
transfer to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "MiMoMixConfig",
    "MiMoMixOutput",
    "RMSNorm",
    "RotaryEmbedding",
    "HybridAttention",
    "DenseFeedForward",
    "SparseMoEFeedForward",
    "MiMoMixBlock",
    "MultiTokenPredictionModule",
    "RecursiveThinkingCore",
    "MiMoMixModel",
    "attention_layout",
    "build_mimomix",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MiMoMixConfig:
    """Every structural knob for a MiMoMix model.

    Defaults are deliberately tiny so the whole stack trains and tests on CPU in
    seconds. Ratios and window semantics follow the MiMo hybrid-attention
    description; the sizes do not.
    """

    vocab_size: int = 512
    hidden_size: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: Optional[int] = None
    intermediate_size: int = 256
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # --- hybrid attention -------------------------------------------------
    sliding_window: int = 128
    #: number of consecutive SWA layers per one global layer (MiMo uses 5 or 6)
    hybrid_ratio: int = 5
    #: force the final layer to be global so the last hidden state can see
    #: the whole prefix even when the ratio would have made it local
    final_layer_global: bool = True
    attention_sink: bool = True

    # --- rope / context extension ----------------------------------------
    #: RoPE base for the *global* layers, which are the ones that must reach
    #: the full context.
    rope_theta: float = 10000.0
    #: RoPE base for the *local* (sliding-window) layers. Gemma 3 and MiMo
    #: decouple these: a layer that can only ever see `sliding_window` tokens
    #: has no long-range dependency to encode, so a large base or a context
    #: extension applied to it is at best wasted and at worst harmful. ``None``
    #: reuses ``rope_theta``, which is the single-table behaviour.
    rope_local_theta: Optional[float] = 10000.0
    #: Apply the context-extension policy to local layers too. Off by default,
    #: for the reason above.
    rope_scale_local: bool = False
    native_context: int = 256
    max_position_embeddings: int = 1024
    #: "none" | "ntk" | "yarn"
    rope_scaling: str = "yarn"
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0

    # --- sparse MoE -------------------------------------------------------
    use_moe: bool = True
    #: dense layers at the bottom of the stack (MoE models commonly keep the
    #: first block(s) dense for stability)
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_intermediate_size: int = 64
    #: aux-loss-free load balancing update speed (gamma); 0 disables the rule.
    #: Keep it small. The sign rule is a control loop, and an oversized step
    #: overshoots the balance point every update and oscillates instead of
    #: converging -- which reads as *worse* balance, not faster balance.
    router_bias_update_speed: float = 1e-3
    #: Fire the bias update inside forward() instead of on an explicit
    #: MiMoMixModel.step_router_bias() call. Off by default: firing per-forward
    #: makes the effective gamma depend on gradient-accumulation depth and on
    #: how many times the controller probes a request. See
    #: SparseMoEFeedForward.update_router_bias.
    router_bias_auto_update: bool = False
    router_z_loss_coef: float = 1e-3
    #: small complementary sequence-level balance loss; the bias rule does the
    #: heavy lifting, this only discourages within-sequence collapse
    router_balance_loss_coef: float = 1e-3
    router_score_function: str = "softmax"  # "softmax" | "sigmoid"
    norm_topk_prob: bool = True

    # --- multi-token prediction ------------------------------------------
    n_mtp_layers: int = 2
    mtp_loss_weight: float = 0.3

    # --- recursive thinking core -----------------------------------------
    use_thinking_core: bool = True
    thinking_latent_dim: int = 64
    thinking_cycles: int = 3
    thinking_max_cycles: int = 8
    thinking_inner_steps: int = 2
    ponder_loss_weight: float = 1e-2
    consistency_loss_weight: float = 1e-2

    def __post_init__(self) -> None:
        if self.head_dim is None:
            if self.hidden_size % self.n_heads != 0:
                raise ValueError("hidden_size must divide n_heads when head_dim is unset")
            self.head_dim = self.hidden_size // self.n_heads
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if self.rope_scaling not in {"none", "ntk", "yarn"}:
            raise ValueError(f"unknown rope_scaling: {self.rope_scaling!r}")
        if self.router_score_function not in {"softmax", "sigmoid"}:
            raise ValueError(f"unknown router_score_function: {self.router_score_function!r}")
        if self.use_moe and self.moe_top_k > self.n_routed_experts:
            raise ValueError("moe_top_k cannot exceed n_routed_experts")
        if self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")
        if self.hybrid_ratio < 0:
            raise ValueError("hybrid_ratio must be >= 0")

    @property
    def context_scale(self) -> float:
        """Extension factor from the natively trained context to the target."""

        return max(1.0, float(self.max_position_embeddings) / float(self.native_context))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def attention_layout(n_layers: int, hybrid_ratio: int, final_layer_global: bool = True) -> Tuple[str, ...]:
    """Return ``("swa", "swa", ..., "global")`` for each layer index.

    With ``hybrid_ratio == r`` every ``r``-th layer is global, giving the
    ``r:1`` local:global interleave MiMo describes. ``hybrid_ratio == 0``
    means "all global" (a plain dense-attention model).
    """

    if hybrid_ratio <= 0:
        return tuple("global" for _ in range(n_layers))
    layout: List[str] = []
    for index in range(n_layers):
        is_global = ((index + 1) % (hybrid_ratio + 1)) == 0
        layout.append("global" if is_global else "swa")
    if final_layer_global and layout:
        layout[-1] = "global"
    return tuple(layout)


# ---------------------------------------------------------------------------
# Normalisation and rotary embeddings
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


def _yarn_correction_dim(num_rotations: float, dim: int, base: float, native_context: int) -> float:
    return (dim * math.log(native_context / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(
    beta_fast: float, beta_slow: float, dim: int, base: float, native_context: int
) -> Tuple[int, int]:
    low = math.floor(_yarn_correction_dim(beta_fast, dim, base, native_context))
    high = math.ceil(_yarn_correction_dim(beta_slow, dim, base, native_context))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp(low: float, high: float, dim: int) -> torch.Tensor:
    if abs(high - low) < 1e-3:
        high = low + 1e-3
    ramp = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0.0, 1.0)


class RotaryEmbedding(nn.Module):
    """RoPE with an explicit progressive context-extension policy.

    ``rope_scaling``:

    ``none``
        Plain RoPE. Positions beyond the trained context extrapolate badly.
    ``ntk``
        NTK-aware base rescaling: ``theta' = theta * s ** (d / (d - 2))``. One
        cheap knob, no retraining required, mild degradation on short context.
    ``yarn``
        Per-frequency-band interpolation: high-frequency dimensions (short
        wavelength relative to the native context) are left alone, low-frequency
        dimensions are fully interpolated by ``s``, and a linear ramp blends the
        band in between. YaRN additionally scales attention logits by
        ``0.1 * ln(s) + 1`` to compensate for the entropy change; that factor is
        exposed as :attr:`attention_scaling` and applied by the attention layer.
    """

    def __init__(self, config: MiMoMixConfig, kind: str = "global"):
        super().__init__()
        self.config = config
        self.kind = kind
        dim = int(config.head_dim)
        if kind == "swa" and config.rope_local_theta is not None:
            base = float(config.rope_local_theta)
        else:
            base = float(config.rope_theta)
        scale = config.context_scale
        if kind == "swa" and not config.rope_scale_local:
            scale = 1.0
        attention_scaling = 1.0

        if config.rope_scaling == "ntk" and scale > 1.0:
            base = base * (scale ** (dim / (dim - 2)))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        elif config.rope_scaling == "yarn" and scale > 1.0:
            pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            inv_freq_extrapolation = 1.0 / pos_freqs
            inv_freq_interpolation = 1.0 / (scale * pos_freqs)
            low, high = _yarn_correction_range(
                config.yarn_beta_fast, config.yarn_beta_slow, dim, base, config.native_context
            )
            # 1 => keep the untouched extrapolation frequency, 0 => interpolate
            extrapolation_factor = 1.0 - _yarn_linear_ramp(low, high, dim // 2)
            inv_freq = (
                inv_freq_interpolation * (1.0 - extrapolation_factor)
                + inv_freq_extrapolation * extrapolation_factor
            )
            attention_scaling = 0.1 * math.log(scale) + 1.0
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = float(attention_scaling)
        self.effective_base = float(base)

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``positions`` is ``(T,)`` -> ``(cos, sin)`` each ``(T, head_dim)``."""

        freqs = positions.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """``x`` is ``(B, H, T, D)``; ``cos``/``sin`` are ``(T, D)``."""

    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Hybrid attention
# ---------------------------------------------------------------------------


class HybridAttention(nn.Module):
    """Grouped-query attention that is either sliding-window or global.

    The ``kind`` is fixed per layer by :func:`attention_layout`. An SWA layer
    only ever needs the last ``sliding_window`` keys, so its cache is trimmed;
    a global layer keeps everything. That asymmetry is the whole point of the
    hybrid design and is asserted by the KV-cache tests.

    When ``attention_sink`` is on, one learnable per-head logit is concatenated
    to the score row before the softmax and then dropped from the weights. The
    head can therefore emit a near-zero output instead of being forced to
    normalise over real tokens -- the standard fix for the "attention sink"
    pathology that otherwise pins mass on position 0.
    """

    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str):
        super().__init__()
        if kind not in {"swa", "global"}:
            raise ValueError(f"unknown attention kind: {kind!r}")
        self.config = config
        self.layer_index = int(layer_index)
        self.kind = kind
        self.n_heads = int(config.n_heads)
        self.n_kv_heads = int(config.n_kv_heads)
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim ** -0.5
        self.window = int(config.sliding_window) if kind == "swa" else None

        # Overwritten by MiMoMixModel with the rotary policy's logit temperature.
        self._attention_scaling = 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        if config.attention_sink:
            # Starts at 0 => the sink initially holds softmax weight exp(0)=1
            # relative to the (scaled) real logits, a mild, learnable floor.
            self.sink = nn.Parameter(torch.zeros(self.n_heads))
        else:
            self.register_parameter("sink", None)

        self.register_buffer("last_sink_mass", torch.zeros(1), persistent=False)

    def cache_span(self, slack: int = 0) -> Optional[int]:
        """How many past keys this layer must retain, or ``None`` for all.

        ``slack`` keeps extra entries beyond the strict window. Speculative
        decoding needs it: rejecting ``r`` drafted tokens means rolling the
        cache back by ``r``, and a cache trimmed to exactly ``window`` would
        have already discarded the keys that rollback brings back into range.
        Holding ``window + slack`` makes any rollback of at most ``slack``
        entries exact.
        """

        if self.kind == "global":
            return None
        return int(self.window) + max(0, int(slack))

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        b, h, t, d = x.shape
        return x[:, :, None].expand(b, h, self.n_rep, t, d).reshape(b, h * self.n_rep, t, d)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Returns ``(output, present_kv)``.

        ``query_positions`` are the absolute positions of ``hidden_states``.
        ``key_positions`` are the absolute positions of the *cached* keys (empty
        tensor when there is no cache); the new keys extend it.
        ``attention_mask`` is an optional ``(B, K_total)`` bool tensor which is
        ``True`` for real (non-padding) key positions.
        """

        bsz, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q_cos, q_sin = cos, sin
        q = apply_rotary(q, q_cos, q_sin)
        k = apply_rotary(k, q_cos, q_sin)

        if past_kv is not None and past_kv[0].numel() > 0:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
            all_key_positions = torch.cat([key_positions, query_positions], dim=0)
        else:
            all_key_positions = query_positions

        present: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if use_cache:
            span = self.cache_span(cache_slack)
            if span is not None and k.shape[2] > span:
                present = (k[:, :, -span:].detach(), v[:, :, -span:].detach())
            else:
                present = (k.detach(), v.detach())

        key_repeat = self._repeat_kv(k)
        value_repeat = self._repeat_kv(v)

        scores = torch.matmul(q, key_repeat.transpose(2, 3)) * self.scaling
        # YaRN's logit temperature compensation, a no-op for other schedules.
        scores = scores * self._attention_scaling

        causal = query_positions.view(-1, 1) >= all_key_positions.view(1, -1)
        if self.window is not None:
            causal = causal & (query_positions.view(-1, 1) - all_key_positions.view(1, -1) < self.window)
        allowed = causal.view(1, 1, q_len, -1)
        if attention_mask is not None:
            allowed = allowed & attention_mask.view(bsz, 1, 1, -1)
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~allowed, neg_inf)

        if self.sink is not None:
            sink = self.sink.view(1, self.n_heads, 1, 1).expand(bsz, self.n_heads, q_len, 1)
            scores = torch.cat([sink.to(scores.dtype), scores], dim=-1)
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            sink_mass = weights[..., 0]
            weights = weights[..., 1:]
            self.last_sink_mass = sink_mass.detach().mean().reshape(1)
        else:
            # A fully masked row cannot happen under causal masking (a query can
            # always see itself), so a plain softmax is safe here.
            weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
            self.last_sink_mass = torch.zeros(1, device=q.device)

        weights = self.dropout(weights)
        out = torch.matmul(weights, value_repeat)
        out = out.transpose(1, 2).reshape(bsz, q_len, self.n_heads * self.head_dim)
        return self.o_proj(out), present


# ---------------------------------------------------------------------------
# Feed-forward: dense and sparse-MoE
# ---------------------------------------------------------------------------


class DenseFeedForward(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class SparseMoEFeedForward(nn.Module):
    """Top-k sparse MoE with auxiliary-loss-free load balancing.

    Routing follows the DeepSeek-V3 / *Auxiliary-Loss-Free Load Balancing*
    recipe:

    1. score every expert from the token, ``s_i = softmax(W x)_i`` (or sigmoid);
    2. **select** the top-k experts by ``s_i + b_i`` where ``b_i`` is a
       non-gradient per-expert bias;
    3. **weight** the selected experts by the raw ``s_i`` (the bias never enters
       the forward value, so it cannot distort the model's function);
    4. after the step, nudge ``b_i`` by ``gamma * sign(mean_load - load_i)``, so
       overloaded experts become less attractive next step.

    Two gradient-carrying regularisers remain, both small:

    * a **router z-loss** ``mean(logsumexp(logits)^2)`` that keeps router logits
      from drifting into a numerically bad regime;
    * a light **sequence-level balance loss** ``N * sum(f_i * P_i)`` that
      discourages within-sequence collapse. The bias rule does the real work.

    ``n_shared_experts`` experts are always applied. Shared experts capture the
    common transformation so the routed experts can specialise -- fine-grained
    expert segmentation in the DeepSeekMoE sense.
    """

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        self.config = config
        self.n_routed = int(config.n_routed_experts)
        self.top_k = int(config.moe_top_k)
        self.n_shared = int(config.n_shared_experts)
        self.score_function = config.router_score_function
        self.norm_topk_prob = bool(config.norm_topk_prob)
        self.update_speed = float(config.router_bias_update_speed)

        self.gate = nn.Linear(config.hidden_size, self.n_routed, bias=False)
        self.experts = nn.ModuleList(
            [
                DenseFeedForward(config.hidden_size, config.moe_intermediate_size, config.dropout)
                for _ in range(self.n_routed)
            ]
        )
        if self.n_shared > 0:
            self.shared_expert = DenseFeedForward(
                config.hidden_size,
                config.moe_intermediate_size * self.n_shared,
                config.dropout,
            )
        else:
            self.shared_expert = None

        self.auto_update_bias = bool(config.router_bias_auto_update)

        # Selection bias is state, not a parameter: it is updated by a rule, not
        # by the optimizer, and it must survive a checkpoint round trip.
        self.register_buffer("expert_bias", torch.zeros(self.n_routed), persistent=True)
        self.register_buffer("last_expert_load", torch.zeros(self.n_routed), persistent=False)
        # Load accumulated since the last bias step. See update_router_bias().
        self.register_buffer("pending_load", torch.zeros(self.n_routed), persistent=False)
        self.register_buffer("pending_batches", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_z_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_balance_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_router_entropy", torch.zeros(()), persistent=False)

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_function == "sigmoid":
            return torch.sigmoid(logits)
        return F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)

    @torch.no_grad()
    def update_router_bias(self) -> bool:
        """Apply one bias update from the load accumulated since the last call.

        **This is deliberately not done inside** :meth:`forward`. The bias rule
        is a control loop with step size ``gamma``, and firing it once per
        forward makes the effective step depend on how many forwards happen per
        optimizer step:

        * gradient accumulation over ``N`` micro-batches fires it ``N`` times,
          multiplying gamma by ``N``;
        * the thinking controller probes the same request at several budgets,
          firing it once per probe;
        * any evaluation pass left in train mode corrupts routing state.

        An oversized effective step does not merely converge faster -- the sign
        rule overshoots the balance point every update and **rings**, so the
        router looks *more* collapsed, not less. Measured on the benchmark task,
        ``gamma=1e-3`` converges while ``gamma>=5e-2`` starves most experts.

        Call this once per optimizer step, after ``optimizer.step()``. Returns
        ``False`` if there was nothing to apply.
        """

        if self.update_speed <= 0.0 or float(self.pending_batches) <= 0.0:
            return False
        load = self.pending_load / self.pending_batches
        target = load.mean()
        # sign(+) => this expert is under-loaded => raise its bias
        self.expert_bias += self.update_speed * torch.sign(target - load)
        self.pending_load.zero_()
        self.pending_batches.zero_()
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        n_tokens = flat.shape[0]

        logits = self.gate(flat)
        scores = self._scores(logits)
        selection_scores = scores + self.expert_bias.to(scores.dtype).unsqueeze(0)
        _, expert_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        gate_weights = scores.gather(-1, expert_indices)
        if self.norm_topk_prob and self.top_k > 1:
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        one_hot = F.one_hot(expert_indices, num_classes=self.n_routed).sum(dim=1)  # (N, E)
        for expert_id, expert in enumerate(self.experts):
            token_ids = torch.nonzero(one_hot[:, expert_id], as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_ids))
            weight = (gate_weights * (expert_indices == expert_id)).sum(dim=-1)
            weight = weight.index_select(0, token_ids).unsqueeze(-1)
            output.index_add_(0, token_ids, expert_out * weight.to(expert_out.dtype))

        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        # --- routing telemetry and regularisers ---------------------------
        load = one_hot.float().mean(dim=0)  # fraction of tokens per expert
        mean_prob = scores.float().mean(dim=0)
        self.last_expert_load = load.detach()
        self.last_router_entropy = (
            -(mean_prob * mean_prob.clamp_min(1e-9).log()).sum().detach()
        )
        z_loss = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        balance_loss = float(self.n_routed) * torch.sum(load * mean_prob)
        self.last_router_z_loss = z_loss.detach()
        self.last_router_balance_loss = balance_loss.detach()
        if self.training:
            self._aux_loss = (
                self.config.router_z_loss_coef * z_loss
                + self.config.router_balance_loss_coef * balance_loss
            )
            with torch.no_grad():
                self.pending_load += load.detach()
                self.pending_batches += 1.0
            if self.auto_update_bias:
                self.update_router_bias()
        else:
            self._aux_loss = logits.new_zeros(())

        return output.reshape(original_shape)

    def aux_loss(self) -> torch.Tensor:
        value = getattr(self, "_aux_loss", None)
        if value is None:
            return self.expert_bias.new_zeros(())
        return value


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------


class MiMoMixBlock(nn.Module):
    def __init__(self, config: MiMoMixConfig, layer_index: int, kind: str, force_dense: bool = False):
        super().__init__()
        self.layer_index = int(layer_index)
        self.kind = kind
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = HybridAttention(config, layer_index, kind)
        self.post_attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        is_moe = config.use_moe and layer_index >= config.n_dense_layers and not force_dense
        self.mlp: nn.Module
        if is_moe:
            self.mlp = SparseMoEFeedForward(config)
        else:
            self.mlp = DenseFeedForward(config.hidden_size, config.intermediate_size, config.dropout)
        self.is_moe = is_moe

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        past_kv=None,
        attention_mask=None,
        use_cache: bool = False,
        cache_slack: int = 0,
    ):
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_norm(hidden_states),
            cos,
            sin,
            query_positions,
            key_positions,
            past_kv=past_kv,
            attention_mask=attention_mask,
            use_cache=use_cache,
            cache_slack=cache_slack,
        )
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.post_attn_norm(hidden_states))
        return hidden_states, present


# ---------------------------------------------------------------------------
# Multi-token prediction
# ---------------------------------------------------------------------------


class MultiTokenPredictionModule(nn.Module):
    """One MTP depth, in the sequential DeepSeek-V3 style.

    Depth ``k`` predicts token ``t + k + 1`` from the hidden state of depth
    ``k - 1`` combined with the *embedding of the token it already knows*:

    ``h'_i = W [ norm(h_i^{k-1}) ; norm(emb(x_{i+k})) ]`` followed by one
    transformer block. Embedding and output head are shared with the trunk, so a
    depth costs one block, not one model.

    At inference the same modules become the draft model for self-speculative
    decoding (see :mod:`mimomix_decoding`).
    """

    def __init__(self, config: MiMoMixConfig, depth: int):
        super().__init__()
        self.depth = int(depth)
        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        # MTP depths use global attention and a *dense* FFN. MiMo describes the
        # MTP module as lightweight with dense FFNs; a routed depth would also
        # make the draft path's cost data-dependent, which defeats the point.
        self.block = MiMoMixBlock(
            config, layer_index=config.n_layers + depth, kind="global", force_dense=True
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        prev_hidden: torch.Tensor,
        shifted_embeddings: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query_positions: torch.Tensor,
        attention_mask=None,
    ) -> torch.Tensor:
        merged = torch.cat(
            [self.hidden_norm(prev_hidden), self.embed_norm(shifted_embeddings)], dim=-1
        )
        hidden = self.proj(merged)
        empty_positions = query_positions.new_empty((0,))
        hidden, _ = self.block(
            hidden,
            cos,
            sin,
            query_positions,
            empty_positions,
            past_kv=None,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return self.final_norm(hidden)


# ---------------------------------------------------------------------------
# Recursive thinking core (Supermix v51/v52 heritage)
# ---------------------------------------------------------------------------


class RecursiveThinkingCore(nn.Module):
    """Weight-tied latent refinement with ACT halting and a quality verifier.

    This is the sequence-model port of ``CognitiveLeapV52ExpertHead``. It runs
    on the trunk's final hidden state and:

    * refines a latent ``z`` for up to ``max_cycles`` weight-tied cycles;
    * emits a halting probability per cycle (ACT / PonderNet), producing a
      halting-weighted mixture of the per-cycle residuals plus a ponder cost;
    * carries a trainable temperature so the head's confidence is calibratable;
    * emits ``p(correct)`` and ``p(continue)`` from a supervised quality head.

    The verifier is *advisory*: it reports whether more compute is warranted.
    It does not silently mutate the returned hidden state beyond a bounded,
    near-zero-initialised residual, so a freshly built model reproduces the
    trunk output up to that residual.
    """

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        hidden = int(config.hidden_size)
        latent = int(config.thinking_latent_dim)
        self.config = config
        self.n_cycles = int(config.thinking_cycles)
        self.max_cycles = int(config.thinking_max_cycles)
        self.inner_steps = int(config.thinking_inner_steps)

        self.to_latent = nn.Sequential(RMSNorm(hidden, config.rms_norm_eps), nn.Linear(hidden, latent))
        self.cell = nn.GRUCell(latent, latent)
        self.refine = nn.Sequential(
            nn.Linear(latent, latent), nn.GELU(), nn.Linear(latent, latent)
        )
        self.halt_head = nn.Linear(latent, 1)
        self.to_residual = nn.Linear(latent, hidden, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.log_temperature = nn.Parameter(torch.zeros(()))

        self.quality_encoder = nn.Sequential(nn.Linear(latent + 3, latent), nn.GELU())
        self.quality_head = nn.Linear(latent, 2)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        nn.init.normal_(self.to_residual.weight, std=0.01)

        self.register_buffer("last_cycles_used", torch.zeros(()), persistent=False)
        self.register_buffer("last_ponder_cost", torch.zeros(()), persistent=False)
        self.register_buffer("last_consistency_loss", torch.zeros(()), persistent=False)
        self.register_buffer("last_quality_score", torch.full((), 0.5), persistent=False)
        self.register_buffer("last_continue_probability", torch.full((), 0.5), persistent=False)
        self.last_exit_reason = "not_run"
        self._last_quality_logits: Optional[torch.Tensor] = None

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature.clamp(min=-2.3, max=2.3))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cycles: Optional[int] = None,
        adaptive: bool = False,
        halt_threshold: float = 0.95,
        stability_tol: float = 1e-3,
        stability_patience: int = 2,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        budget = self.n_cycles if cycles is None else int(cycles)
        budget = max(1, min(budget, self.max_cycles))

        shape = hidden_states.shape
        flat = hidden_states.reshape(-1, shape[-1])
        z = self.to_latent(flat)
        state = torch.zeros_like(z)

        cumulative_halt = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
        residual_mixture = torch.zeros_like(flat)
        ponder = torch.zeros(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
        consistency = flat.new_zeros(())
        previous_residual: Optional[torch.Tensor] = None
        stable_steps = 0
        cycles_used = 0
        exit_reason = "budget_exhausted"

        for cycle in range(budget):
            cycles_used = cycle + 1
            for _ in range(self.inner_steps):
                state = self.cell(self.refine(z), state)
            z = z + state

            halt_logit = self.halt_head(state)
            halt_p = torch.sigmoid(halt_logit)
            remaining = (1.0 - cumulative_halt).clamp_min(0.0)
            # PonderNet-style: this cycle claims halt_p of the mass still in
            # flight. Whatever is still unclaimed when the loop ends is given to
            # the final residual below, so the mixture stays convex no matter
            # which exit fired.
            step_weight = remaining * halt_p
            cumulative_halt = cumulative_halt + step_weight
            step_residual = self.to_residual(z)
            residual_mixture = residual_mixture + step_weight * step_residual
            ponder = ponder + remaining

            if previous_residual is not None:
                delta = (step_residual - previous_residual).pow(2).mean()
                consistency = consistency + delta
                if adaptive and float(delta.detach()) <= stability_tol:
                    stable_steps += 1
                    if stable_steps >= stability_patience:
                        exit_reason = "prediction_stability"
                        previous_residual = step_residual
                        break
                else:
                    stable_steps = 0
            previous_residual = step_residual

            if adaptive and float(cumulative_halt.mean().detach()) >= halt_threshold:
                exit_reason = "halting_threshold"
                break

        # Any unspent halting mass falls to the last residual so the mixture is
        # a proper convex combination regardless of where the loop exited.
        leftover = (1.0 - cumulative_halt).clamp_min(0.0)
        if previous_residual is not None:
            residual_mixture = residual_mixture + leftover * previous_residual

        scale = self.residual_scale + (1e-4 if self.training else 0.0)
        refined = flat + scale * residual_mixture

        confidence = torch.sigmoid(residual_mixture.abs().mean(dim=-1, keepdim=True))
        depth_feature = torch.full_like(confidence, float(cycles_used) / float(self.max_cycles))
        halt_feature = cumulative_halt
        quality_state = self.quality_encoder(
            torch.cat([z, confidence, depth_feature, halt_feature], dim=-1)
        )
        # Temperature scaling on the verifier's own logits. This is the one
        # calibration knob the controller trusts, and it is trained by
        # verifier_loss -- deliberately not by the language-model objective, so
        # calibrating the verifier cannot drift the token distribution.
        quality_logits = self.quality_head(quality_state) / self.temperature
        quality_probs = torch.sigmoid(quality_logits)

        self.last_cycles_used = torch.tensor(float(cycles_used), device=flat.device)
        self.last_ponder_cost = ponder.mean().detach()
        self.last_consistency_loss = (consistency / max(1, cycles_used - 1)).detach()
        self.last_quality_score = quality_probs[:, 0].mean().detach()
        self.last_continue_probability = quality_probs[:, 1].mean().detach()
        self.last_exit_reason = exit_reason
        # Retained for verifier_loss(...); cleared outside training so an
        # inference pass never pins a graph.
        self._last_quality_logits = quality_logits if self.training else None

        info = {
            "quality_logits": quality_logits,
            "quality_probability": quality_probs[:, 0],
            "continue_probability": quality_probs[:, 1],
            "ponder_cost": ponder.mean(),
            "consistency_loss": consistency / max(1, cycles_used - 1),
            "cycles_used": torch.tensor(float(cycles_used), device=flat.device),
        }
        return refined.reshape(shape), info


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass
class MiMoMixOutput:
    """Structured forward result. ``telemetry`` is always JSON-safe."""

    logits: torch.Tensor
    hidden_states: torch.Tensor
    #: trunk output *before* the thinking core and final norm -- the state the
    #: MTP depths were trained to consume, and therefore the draft seed
    trunk_hidden: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    lm_loss: Optional[torch.Tensor] = None
    mtp_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    mtp_logits: List[torch.Tensor] = field(default_factory=list)
    past_key_values: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None
    telemetry: Dict[str, object] = field(default_factory=dict)


class MiMoMixModel(nn.Module):
    """Decoder-only transformer with the full MiMoMix stack."""

    def __init__(self, config: MiMoMixConfig):
        super().__init__()
        self.config = config
        self.layout = attention_layout(config.n_layers, config.hybrid_ratio, config.final_layer_global)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Two rotary tables. Global layers carry the context-extension policy;
        # local layers keep their own (usually smaller, usually unscaled) base,
        # because a sliding window cannot express a dependency longer than
        # itself and extending its frequencies only distorts what it can see.
        self.rotary = RotaryEmbedding(config, kind="global")
        self.rotary_local = RotaryEmbedding(config, kind="swa")
        self.layers = nn.ModuleList(
            [MiMoMixBlock(config, i, kind) for i, kind in enumerate(self.layout)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp_modules = nn.ModuleList(
            [MultiTokenPredictionModule(config, depth) for depth in range(config.n_mtp_layers)]
        )
        self.thinking_core = RecursiveThinkingCore(config) if config.use_thinking_core else None

        # Propagate each layer's own RoPE attention-temperature policy. MTP
        # depths are global, so they take the global scaling.
        for module in self.modules():
            if isinstance(module, HybridAttention):
                source = self.rotary_local if module.kind == "swa" else self.rotary
                module._attention_scaling = source.attention_scaling

        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # -- parameter accounting ---------------------------------------------

    def parameter_report(self) -> Dict[str, int]:
        """Total vs. per-token-active parameters -- the headline MoE ratio.

        "Active" counts, for each MoE layer, the shared expert plus ``top_k``
        routed experts rather than all of them. It excludes the MTP depths,
        which are draft-only at inference.
        """

        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for module in self.modules():
            if isinstance(module, SparseMoEFeedForward):
                per_expert = sum(p.numel() for p in module.experts[0].parameters())
                inactive += per_expert * (module.n_routed - module.top_k)
        mtp = sum(p.numel() for p in self.mtp_modules.parameters())
        return {
            "total": int(total),
            "active_per_token": int(total - inactive - mtp),
            "routed_but_idle": int(inactive),
            "mtp_draft": int(mtp),
        }

    # -- forward ------------------------------------------------------------

    def _collect_aux_loss(self, device, dtype) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=dtype)
        for module in self.modules():
            if isinstance(module, SparseMoEFeedForward):
                total = total + module.aux_loss().to(device=device, dtype=dtype)
        return total

    def step_router_bias(self) -> int:
        """Apply the aux-loss-free bias update to every MoE layer.

        Call once per optimizer step, **after** ``optimizer.step()``. Returns
        how many layers were updated. See
        :meth:`SparseMoEFeedForward.update_router_bias` for why this is not
        folded into ``forward``.
        """

        return sum(
            1
            for module in self.modules()
            if isinstance(module, SparseMoEFeedForward) and module.update_router_bias()
        )

    def telemetry(self) -> Dict[str, object]:
        """JSON-safe snapshot of the last forward pass."""

        sinks: List[float] = []
        loads: List[List[float]] = []
        router_entropy: List[float] = []
        for module in self.modules():
            if isinstance(module, HybridAttention):
                sinks.append(float(module.last_sink_mass.mean()))
            if isinstance(module, SparseMoEFeedForward):
                loads.append([float(v) for v in module.last_expert_load])
                router_entropy.append(float(module.last_router_entropy))
        snapshot: Dict[str, object] = {
            "attention_layout": list(self.layout),
            "sliding_window": int(self.config.sliding_window),
            "rope_scaling": self.config.rope_scaling,
            "rope_attention_scaling": float(self.rotary.attention_scaling),
            "rope_global_base": float(self.rotary.effective_base),
            "rope_local_base": float(self.rotary_local.effective_base),
            "mean_sink_mass": float(sum(sinks) / len(sinks)) if sinks else 0.0,
            "per_layer_sink_mass": sinks,
            "expert_load": loads,
            "router_entropy": router_entropy,
            "parameters": self.parameter_report(),
        }
        if self.thinking_core is not None:
            snapshot["thinking"] = {
                "cycles_used": float(self.thinking_core.last_cycles_used),
                "ponder_cost": float(self.thinking_core.last_ponder_cost),
                "consistency_loss": float(self.thinking_core.last_consistency_loss),
                "quality_probability": float(self.thinking_core.last_quality_score),
                "continue_probability": float(self.thinking_core.last_continue_probability),
                "exit_reason": self.thinking_core.last_exit_reason,
                "temperature": float(self.thinking_core.temperature.detach()),
            }
        return snapshot

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
        thinking_cycles: Optional[int] = None,
        adaptive_thinking: bool = False,
        return_mtp: Optional[bool] = None,
        cache_slack: int = 0,
        past_length: Optional[int] = None,
    ) -> MiMoMixOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if past_length is not None:
            # The caller knows the true absolute position. Trust it: inferring
            # it from cache lengths is only valid while some layer keeps an
            # untrimmed (global) cache.
            past_len = int(past_length)
        else:
            past_len = 0
            if past_key_values is not None and len(past_key_values) > 0:
                for entry in past_key_values:
                    if entry is not None and entry[0].numel() > 0:
                        past_len = max(past_len, int(entry[0].shape[2]))
        if attention_mask is not None and past_len > 0:
            # Under a hybrid cache each layer retains a different number of past
            # keys, so one caller-supplied mask cannot describe them all. Rather
            # than silently misalign it, refuse the combination.
            raise ValueError(
                "attention_mask is only supported for a full-sequence forward; "
                "with a KV cache the per-layer key spans differ"
            )
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=device)
            if attention_mask.shape != (bsz, seq_len):
                raise ValueError(
                    f"attention_mask must be {(bsz, seq_len)}, got {tuple(attention_mask.shape)}"
                )

        # Absolute positions: with a trimmed SWA cache the *cached* positions
        # differ per layer, so each layer receives its own key_positions below.
        query_positions = torch.arange(past_len, past_len + seq_len, device=device)
        cos, sin = self.rotary(query_positions)
        local_cos, local_sin = self.rotary_local(query_positions)

        hidden = self.embed_tokens(input_ids)
        presents: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = []
        for index, layer in enumerate(self.layers):
            past_kv = None
            if past_key_values is not None and index < len(past_key_values):
                past_kv = past_key_values[index]
            cached = 0 if past_kv is None or past_kv[0].numel() == 0 else int(past_kv[0].shape[2])
            key_positions = torch.arange(past_len - cached, past_len, device=device)
            layer_cos, layer_sin = (local_cos, local_sin) if layer.kind == "swa" else (cos, sin)
            hidden, present = layer(
                hidden,
                layer_cos,
                layer_sin,
                query_positions,
                key_positions,
                past_kv=past_kv,
                attention_mask=attention_mask,
                use_cache=use_cache,
                cache_slack=cache_slack,
            )
            presents.append(present)

        trunk_hidden = hidden
        thinking_info: Dict[str, torch.Tensor] = {}
        if self.thinking_core is not None:
            hidden, thinking_info = self.thinking_core(
                hidden, cycles=thinking_cycles, adaptive=adaptive_thinking
            )

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        want_mtp = self.config.n_mtp_layers > 0 if return_mtp is None else bool(return_mtp)
        mtp_logits: List[torch.Tensor] = []
        mtp_loss: Optional[torch.Tensor] = None
        if want_mtp and len(self.mtp_modules) > 0:
            prev_hidden = trunk_hidden
            embeddings = self.embed_tokens(input_ids)
            depth_losses: List[torch.Tensor] = []
            for depth, module in enumerate(self.mtp_modules, start=1):
                # depth k consumes the embedding of the token k positions ahead
                shifted = torch.zeros_like(embeddings)
                if seq_len > depth:
                    shifted[:, : seq_len - depth] = embeddings[:, depth:]
                prev_hidden = module(prev_hidden, shifted, cos, sin, query_positions, attention_mask)
                depth_logits = self.lm_head(self.norm(prev_hidden))
                mtp_logits.append(depth_logits)
                if labels is not None and seq_len > depth + 1:
                    # depth k at position i predicts label at i + k + 1
                    pred = depth_logits[:, : seq_len - depth - 1]
                    target = labels[:, depth + 1 :]
                    depth_losses.append(
                        F.cross_entropy(
                            pred.reshape(-1, pred.shape[-1]), target.reshape(-1), reduction="mean"
                        )
                    )
            if depth_losses:
                mtp_loss = torch.stack(depth_losses).mean()

        lm_loss: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None
        aux_loss = self._collect_aux_loss(logits.device, logits.dtype)
        if labels is not None:
            shift_logits = logits[:, :-1]
            shift_labels = labels[:, 1:]
            lm_loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1)
            )
            loss = lm_loss + aux_loss
            if mtp_loss is not None:
                loss = loss + self.config.mtp_loss_weight * mtp_loss
            if thinking_info:
                loss = (
                    loss
                    + self.config.ponder_loss_weight * thinking_info["ponder_cost"]
                    + self.config.consistency_loss_weight * thinking_info["consistency_loss"]
                )

        telemetry = self.telemetry()
        telemetry["sequence_length"] = int(seq_len)
        telemetry["past_length"] = int(past_len)
        if thinking_info:
            telemetry["thinking"]["quality_probability"] = float(
                thinking_info["quality_probability"].detach().mean()
            )
            telemetry["thinking"]["continue_probability"] = float(
                thinking_info["continue_probability"].detach().mean()
            )

        return MiMoMixOutput(
            logits=logits,
            hidden_states=hidden,
            trunk_hidden=trunk_hidden,
            loss=loss,
            lm_loss=lm_loss,
            mtp_loss=mtp_loss,
            aux_loss=aux_loss,
            mtp_logits=mtp_logits,
            past_key_values=presents if use_cache else None,
            telemetry=telemetry,
        )

    @torch.no_grad()
    def propose_draft(
        self, trunk_hidden_last: torch.Tensor, seed_token: torch.Tensor, position: int
    ) -> torch.Tensor:
        """Run the MTP chain to speculate ``n_mtp_layers`` tokens ahead.

        ``trunk_hidden_last`` is ``(B, 1, H)`` -- the trunk state at the last
        real position. ``seed_token`` is ``(B, 1)``: the token the trunk already
        committed for the next slot, which depth 1 conditions on.

        Draft quality only affects *speed*. Every speculative token is checked
        against the trunk in :mod:`mimomix_decoding`, so a bad draft costs
        throughput and never changes the emitted sequence.
        """

        if len(self.mtp_modules) == 0:
            return seed_token.new_zeros((seed_token.shape[0], 0))
        positions = torch.tensor([int(position)], device=seed_token.device)
        cos, sin = self.rotary(positions)
        prev = trunk_hidden_last
        current = seed_token
        drafted: List[torch.Tensor] = []
        for module in self.mtp_modules:
            embedded = self.embed_tokens(current)
            prev = module(prev, embedded, cos, sin, positions)
            current = self.lm_head(self.norm(prev)).argmax(dim=-1)
            drafted.append(current)
        return torch.cat(drafted, dim=1)

    def verifier_loss(self, correctness: torch.Tensor) -> torch.Tensor:
        """Supervised quality/continue objective for the thinking core.

        ``correctness`` is a float tensor in ``{0, 1}`` per flattened position.
        ``p(correct)`` is trained to match it and ``p(continue)`` to match its
        complement -- the v52 contract: keep thinking exactly when the current
        answer is wrong.
        """

        if self.thinking_core is None:
            raise RuntimeError("model was built without a thinking core")
        logits = getattr(self.thinking_core, "_last_quality_logits", None)
        if logits is None:
            raise RuntimeError("verifier_loss requires a forward pass that stored quality logits")
        target = correctness.reshape(-1).to(logits.dtype)
        return F.binary_cross_entropy_with_logits(
            logits[:, 0], target
        ) + F.binary_cross_entropy_with_logits(logits[:, 1], 1.0 - target)


def build_mimomix(**overrides) -> MiMoMixModel:
    """Convenience constructor: ``build_mimomix(n_layers=4, use_moe=False)``."""

    return MiMoMixModel(MiMoMixConfig(**overrides))
