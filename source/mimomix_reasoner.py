"""Supermix v56 -- a latent state-machine reasoner on the v53 MiMoMix trunk.

V56 is an additive research line. It reuses the v53 MiMoMix components verbatim
(`MiMoMixBlock`, hybrid SWA/global attention with learnable sinks, sparse MoE
with the auxiliary-loss-free bias rule, decoupled RoPE tables, and the
`RecursiveThinkingCore` with its ACT halting and supervised verifier) and adds
one thing v53 does not have: an explicit **latent state machine** between the
trunk and the answer.

## Why a state machine

The v51 line trains a feed-forward head on a flat 128-dimensional vector whose
label is a 4-step composition of modular operations. Its recorded held-out
accuracy on that task is 0.1710 against a 0.1430 majority-class floor, i.e. the
head learns the last step's statistics and not the composition.

V56 changes the *shape* of the computation rather than its size:

1. **Slot tokenisation.** The same input vector is read as a short sequence of
   slots. No dimension is added, removed, or re-derived from the generator --
   `LatentStateReasoner` consumes exactly the tensor the v51 head consumed.
2. **Position-equivariant encoding.** Repeated blocks share one encoder, so an
   operation means the same thing wherever it appears. Without sharing, a model
   must learn `n_blocks x n_ops` maps instead of `n_ops`.
3. **Row-stochastic transitions.** Each operator slot emits a `K x K`
   row-stochastic matrix and the state is composed in log space. The state is a
   distribution over `n_states` latent states, not a hidden vector, so the
   composition is the model's own arithmetic rather than an emergent property.
4. **Identity initialisation.** A product of near-uniform stochastic matrices
   mixes to uniform and starves the gradient reaching the first operator. Each
   operator's logits are biased toward the identity at init, so a fresh model
   passes the initial state through unchanged -- the same discipline v53 uses
   when it zero-initialises the thinking core's residual scale.
5. **A crispness prior.** The maps being composed are deterministic functions,
   so a row that is one-hot is a row that has learned a function. The optional
   operator-entropy penalty says so directly instead of hoping.

## What is deliberately absent

Multi-token prediction and speculative decoding do not apply: v56 emits one
answer per input, so there is no next token to draft and no acceptance length to
report. They stay in `mimomix_core`/`mimomix_decoding` for the generative line.

Nothing here claims the model is good. `benchmark_mimomix_reasoner.py` measures
it and `run_v56_promotion_gate.py` decides whether a measurement survives a
paired multi-seed test.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimomix_core import (
    HybridAttention,
    MiMoMixBlock,
    MiMoMixConfig,
    RecursiveThinkingCore,
    RMSNorm,
    RotaryEmbedding,
    SparseMoEFeedForward,
    attention_layout,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "LatentStateReasoner",
    "ReasonerConfig",
    "ReasonerOutput",
    "SlotTokenizer",
    "block_slots",
    "build_reasoner",
    "load_reasoner",
    "patch_slots",
    "save_reasoner",
]

CHECKPOINT_SCHEMA = "supermix-v56-reasoner-checkpoint-v1"

Slot = Tuple[int, int]


# ---------------------------------------------------------------------------
# Slot layouts
# ---------------------------------------------------------------------------


def block_slots(
    input_dim: int, block_offset: int, block_stride: int, block_count: int
) -> Tuple[Tuple[Slot, ...], Tuple[Slot, ...]]:
    """Split ``input_dim`` into (context slots, operator slots) by block layout.

    ``block_count`` operator slots of width ``block_stride`` start at
    ``block_offset``. Whatever sits before the first block and after the last one
    becomes a context slot, so **no input dimension is discarded** -- a dropped
    tail would be a silent change to what the model is given.
    """

    if block_offset < 0 or block_stride < 1 or block_count < 1:
        raise ValueError("block_offset >= 0, block_stride >= 1, block_count >= 1 required")
    end = block_offset + block_stride * block_count
    if end > input_dim:
        raise ValueError(
            f"block layout needs {end} dims but input_dim is {input_dim}"
        )
    operators = tuple(
        (block_offset + index * block_stride, block_offset + (index + 1) * block_stride)
        for index in range(block_count)
    )
    context: List[Slot] = []
    if block_offset > 0:
        context.append((0, block_offset))
    if end < input_dim:
        context.append((end, input_dim))
    if not context:
        raise ValueError("block layout leaves no context slot; the state needs an origin")
    return tuple(context), operators


def patch_slots(input_dim: int, patch_count: int) -> Tuple[Tuple[Slot, ...], Tuple[Slot, ...]]:
    """Split ``input_dim`` into equal patches, knowing nothing about the layout.

    The first patch is context and the rest are operator slots. This is the
    layout-agnostic view: it encodes no knowledge of where the generator's
    boundaries actually are, so a result under this mode cannot be explained by
    handing the model the task's structure.
    """

    if patch_count < 2:
        raise ValueError("patch_count must be >= 2 (one context patch, one operator patch)")
    if patch_count > input_dim:
        raise ValueError("patch_count cannot exceed input_dim")
    step = input_dim // patch_count
    bounds: List[Slot] = []
    for index in range(patch_count):
        start = index * step
        stop = input_dim if index == patch_count - 1 else (index + 1) * step
        bounds.append((start, stop))
    return (bounds[0],), tuple(bounds[1:])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ReasonerConfig:
    """Every structural knob for a v56 reasoner.

    Defaults target the v51 chained-modular-arithmetic input contract
    (``input_dim=128``, four 24-wide op blocks at offset 10, ten classes) because
    that is the task with a recorded baseline to beat. Nothing in the model
    depends on those numbers.
    """

    # --- input view -------------------------------------------------------
    input_dim: int = 128
    n_classes: int = 10
    #: "blocks" uses the generator's block layout; "patches" knows nothing.
    slot_layout: str = "blocks"
    block_offset: int = 10
    block_stride: int = 24
    block_count: int = 4
    patch_count: int = 8
    #: one shared encoder for every operator slot (position equivariance)
    share_block_encoder: bool = True
    #: add a learned per-position embedding to operator tokens
    positional_blocks: bool = True
    input_norm: bool = True

    # --- trunk (v53 MiMoMix components) -----------------------------------
    hidden_size: int = 96
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 2
    intermediate_size: int = 192
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6
    sliding_window: int = 4
    hybrid_ratio: int = 2
    attention_sink: bool = True
    rope_theta: float = 10000.0
    rope_local_theta: Optional[float] = 10000.0
    #: the token sequence here is a handful of slots, so context extension has
    #: nothing to extend. "none" is the honest default.
    rope_scaling: str = "none"
    use_moe: bool = True
    n_dense_layers: int = 1
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_top_k: int = 2
    moe_intermediate_size: int = 64
    router_bias_update_speed: float = 1e-3
    router_z_loss_coef: float = 1e-3
    router_balance_loss_coef: float = 1e-3

    # --- latent state machine --------------------------------------------
    n_states: int = 10
    operator_hidden: int = 192
    operator_layers: int = 2
    #: diagonal logit bias at init; a fresh chain is then ~identity
    identity_gain: float = 4.0
    operator_temperature: float = 1.0
    #: penalty on transition-row entropy -- the maps being learned are functions
    operator_entropy_weight: float = 0.0
    #: initialise the class head as a scaled identity when n_states == n_classes
    identity_class_head: bool = True
    #: What the operator head reads.
    #:
    #: "slot"  -- the pre-trunk slot embedding only. Exactly position-equivariant:
    #:            the same operation in any block produces the same operator.
    #: "trunk" -- the post-trunk hidden state only. Maximally expressive, but
    #:            attention mixes neighbouring slots into it, so the same
    #:            operation in a different block is no longer the same input and
    #:            the shared encoder's advantage is lost.
    #: "gated" -- the slot embedding plus a zero-initialised gate on the trunk
    #:            state. A fresh model is exactly equivariant and can learn to
    #:            spend context only where context pays, which is the same
    #:            discipline the thinking core uses for its residual scale.
    operator_context: str = "gated"

    # --- thinking core (v52/v53 lineage) ---------------------------------
    use_thinking_core: bool = True
    thinking_latent_dim: int = 64
    thinking_cycles: int = 3
    thinking_max_cycles: int = 8
    thinking_inner_steps: int = 2
    ponder_loss_weight: float = 1e-2
    consistency_loss_weight: float = 1e-2

    def __post_init__(self) -> None:
        if self.slot_layout not in {"blocks", "patches"}:
            raise ValueError(f"unknown slot_layout: {self.slot_layout!r}")
        if self.operator_context not in {"slot", "trunk", "gated"}:
            raise ValueError(f"unknown operator_context: {self.operator_context!r}")
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if self.n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        if self.operator_layers < 1:
            raise ValueError("operator_layers must be >= 1")
        if self.operator_temperature <= 0:
            raise ValueError("operator_temperature must be > 0")
        # surfaces the slot-layout errors at construction time, not first forward
        self.slots()

    def slots(self) -> Tuple[Tuple[Slot, ...], Tuple[Slot, ...]]:
        if self.slot_layout == "blocks":
            return block_slots(
                self.input_dim, self.block_offset, self.block_stride, self.block_count
            )
        return patch_slots(self.input_dim, self.patch_count)

    @property
    def sequence_length(self) -> int:
        context, operators = self.slots()
        return 1 + len(operators)

    def trunk_config(self) -> MiMoMixConfig:
        """The `MiMoMixConfig` used for the trunk blocks.

        `vocab_size` and the MTP/thinking fields exist on `MiMoMixConfig` because
        it also describes a language model. v56 embeds no tokens through it and
        runs no MTP depths, so `vocab_size` is set to `n_classes` purely to keep
        the dataclass valid and `n_mtp_layers` is pinned to 0.
        """

        return MiMoMixConfig(
            vocab_size=max(2, int(self.n_classes)),
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=False,
            sliding_window=max(1, int(self.sliding_window)),
            hybrid_ratio=self.hybrid_ratio,
            final_layer_global=True,
            attention_sink=self.attention_sink,
            rope_theta=self.rope_theta,
            rope_local_theta=self.rope_local_theta,
            rope_scale_local=False,
            native_context=max(8, self.sequence_length),
            max_position_embeddings=max(8, self.sequence_length),
            rope_scaling=self.rope_scaling,
            use_moe=self.use_moe,
            n_dense_layers=self.n_dense_layers,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            moe_top_k=self.moe_top_k,
            moe_intermediate_size=self.moe_intermediate_size,
            router_bias_update_speed=self.router_bias_update_speed,
            router_bias_auto_update=False,
            router_z_loss_coef=self.router_z_loss_coef,
            router_balance_loss_coef=self.router_balance_loss_coef,
            n_mtp_layers=0,
            mtp_loss_weight=0.0,
            use_thinking_core=False,  # v56 owns the thinking core explicitly
            thinking_latent_dim=self.thinking_latent_dim,
            thinking_cycles=self.thinking_cycles,
            thinking_max_cycles=self.thinking_max_cycles,
            thinking_inner_steps=self.thinking_inner_steps,
            ponder_loss_weight=self.ponder_loss_weight,
            consistency_loss_weight=self.consistency_loss_weight,
        )

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class ReasonerOutput:
    """Structured forward result. ``telemetry`` is always JSON-safe."""

    logits: torch.Tensor
    #: (B, n_states) log-probabilities of the final latent state
    state_log_probs: torch.Tensor
    #: per operator slot, (B, n_states, n_states) log transition matrices
    operator_log_probs: List[torch.Tensor] = field(default_factory=list)
    #: the state after every step, starting with the initial state. Length is
    #: ``1 + len(operator_log_probs)``. This is the model's reasoning made
    #: inspectable rather than inferred -- the web interface reads it directly.
    state_trace: List[torch.Tensor] = field(default_factory=list)
    loss: Optional[torch.Tensor] = None
    task_loss: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    operator_entropy: Optional[torch.Tensor] = None
    quality_probability: Optional[torch.Tensor] = None
    continue_probability: Optional[torch.Tensor] = None
    telemetry: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class SlotTokenizer(nn.Module):
    """Read a flat feature vector as a short sequence of slot tokens.

    The context slots are summed into a single token so the state has exactly one
    origin; operator slots become one token each. When
    ``share_block_encoder`` is set, every operator slot goes through the *same*
    projection, which requires them to have equal width.
    """

    def __init__(self, config: ReasonerConfig):
        super().__init__()
        self.config = config
        context, operators = config.slots()
        self.context_slots = context
        self.operator_slots = operators
        hidden = int(config.hidden_size)

        self.context_proj = nn.ModuleList(
            [nn.Linear(stop - start, hidden) for start, stop in context]
        )

        widths = {stop - start for start, stop in operators}
        self.shared = bool(config.share_block_encoder)
        if self.shared:
            if len(widths) != 1:
                raise ValueError(
                    "share_block_encoder requires equal-width operator slots, got "
                    f"widths {sorted(widths)}"
                )
            self.operator_proj = nn.ModuleList([nn.Linear(widths.pop(), hidden)])
        else:
            self.operator_proj = nn.ModuleList(
                [nn.Linear(stop - start, hidden) for start, stop in operators]
            )

        self.position = (
            nn.Parameter(torch.randn(len(operators), hidden) * 0.02)
            if config.positional_blocks
            else None
        )
        self.norm = RMSNorm(hidden, config.rms_norm_eps) if config.input_norm else None

    @property
    def sequence_length(self) -> int:
        return 1 + len(self.operator_slots)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() == 3 and inputs.shape[1] == 1:
            # the v51 harness hands out (B, 1, D)
            inputs = inputs.squeeze(1)
        if inputs.dim() != 2:
            raise ValueError(f"expected (B, D) or (B, 1, D) input, got {tuple(inputs.shape)}")
        if inputs.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input_dim {self.config.input_dim}, got {inputs.shape[-1]}"
            )

        context = None
        for proj, (start, stop) in zip(self.context_proj, self.context_slots):
            piece = proj(inputs[:, start:stop])
            context = piece if context is None else context + piece

        tokens = [context]
        for index, (start, stop) in enumerate(self.operator_slots):
            proj = self.operator_proj[0] if self.shared else self.operator_proj[index]
            token = proj(inputs[:, start:stop])
            if self.position is not None:
                token = token + self.position[index]
            tokens.append(token)

        stacked = torch.stack(tokens, dim=1)
        return self.norm(stacked) if self.norm is not None else stacked


class TransitionOperator(nn.Module):
    """Map a token to a row-stochastic ``K x K`` transition matrix.

    The head is nonlinear on purpose. A linear map from a concatenation of
    one-hot fields is additive in its fields, so it cannot express the
    *interaction* between an operation and its operand -- and an operator that
    cannot represent that interaction cannot represent the task.
    """

    def __init__(self, config: ReasonerConfig):
        super().__init__()
        states = int(config.n_states)
        self.states = states
        self.temperature = float(config.operator_temperature)

        layers: List[nn.Module] = []
        width = int(config.hidden_size)
        for _ in range(max(1, int(config.operator_layers))):
            layers.append(nn.Linear(width, config.operator_hidden))
            layers.append(nn.GELU())
            width = int(config.operator_hidden)
        final = nn.Linear(width, states * states)
        # small final weights + a diagonal bias => a fresh operator is ~identity
        nn.init.normal_(final.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.register_buffer(
            "identity_bias",
            float(config.identity_gain) * torch.eye(states).reshape(-1),
            persistent=False,
        )

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        raw = self.net(token) + self.identity_bias
        raw = raw.view(-1, self.states, self.states) / self.temperature
        return F.log_softmax(raw, dim=-1)


class LatentStateReasoner(nn.Module):
    """Slot tokeniser -> MiMoMix trunk -> thinking core -> latent state machine."""

    def __init__(self, config: ReasonerConfig):
        super().__init__()
        self.config = config
        trunk = config.trunk_config()
        self.trunk_config = trunk
        self.layout = attention_layout(trunk.n_layers, trunk.hybrid_ratio, trunk.final_layer_global)

        self.tokenizer = SlotTokenizer(config)
        self.rotary = RotaryEmbedding(trunk, kind="global")
        self.rotary_local = RotaryEmbedding(trunk, kind="swa")
        self.layers = nn.ModuleList(
            [MiMoMixBlock(trunk, index, kind) for index, kind in enumerate(self.layout)]
        )
        self.norm = RMSNorm(trunk.hidden_size, trunk.rms_norm_eps)
        self.thinking_core = RecursiveThinkingCore(trunk) if config.use_thinking_core else None

        states = int(config.n_states)
        self.initial_state = nn.Sequential(
            nn.Linear(trunk.hidden_size, trunk.hidden_size),
            nn.GELU(),
            nn.Linear(trunk.hidden_size, states),
        )
        self.operator = TransitionOperator(config)
        self.operator_norm = RMSNorm(trunk.hidden_size, trunk.rms_norm_eps)
        # zero-init so a fresh "gated" model is exactly position-equivariant
        self.operator_context_gate = (
            nn.Parameter(torch.zeros(())) if config.operator_context == "gated" else None
        )
        self.class_head = nn.Linear(states, int(config.n_classes))
        if config.identity_class_head and states == int(config.n_classes):
            # the state already *is* the answer distribution at init; let training
            # move away from that rather than having to discover it
            with torch.no_grad():
                self.class_head.weight.copy_(torch.eye(states))
                self.class_head.bias.zero_()

        # Each layer's rotary attention-temperature policy, exactly as
        # MiMoMixModel propagates it (mimomix_core.py:994-997). Skipping this
        # silently drops YaRN's logit temperature on every layer.
        for module in self.modules():
            if isinstance(module, HybridAttention):
                source = self.rotary_local if module.kind == "swa" else self.rotary
                module._attention_scaling = source.attention_scaling

    # -- accounting --------------------------------------------------------

    def parameter_report(self) -> Dict[str, int]:
        """Total vs. per-input-active parameters -- the ratio MoE models carry."""

        total = sum(p.numel() for p in self.parameters())
        inactive = 0
        for module in self.modules():
            if isinstance(module, SparseMoEFeedForward):
                per_expert = sum(p.numel() for p in module.experts[0].parameters())
                inactive += per_expert * (module.n_routed - module.top_k)
        return {
            "total": int(total),
            "active_per_input": int(total - inactive),
            "routed_but_idle": int(inactive),
        }

    def step_router_bias(self) -> int:
        """Apply the aux-loss-free bias update once per optimizer step.

        Firing per forward instead makes the effective step size depend on
        gradient-accumulation depth and on how many budgets the controller
        probes, which makes the sign rule oscillate rather than converge. See
        `SparseMoEFeedForward.update_router_bias`.
        """

        return sum(
            1
            for module in self.modules()
            if isinstance(module, SparseMoEFeedForward) and module.update_router_bias()
        )

    def _collect_aux_loss(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=dtype)
        for module in self.modules():
            if isinstance(module, SparseMoEFeedForward):
                total = total + module.aux_loss().to(device=device, dtype=dtype)
        return total

    def telemetry(self) -> Dict[str, object]:
        """JSON-safe snapshot of the last forward pass."""

        sinks: List[float] = []
        loads: List[List[float]] = []
        entropies: List[float] = []
        for module in self.modules():
            if isinstance(module, HybridAttention):
                sinks.append(float(module.last_sink_mass.mean()))
            if isinstance(module, SparseMoEFeedForward):
                loads.append([float(v) for v in module.last_expert_load])
                entropies.append(float(module.last_router_entropy))
        snapshot: Dict[str, object] = {
            "attention_layout": list(self.layout),
            "sliding_window": int(self.trunk_config.sliding_window),
            "sequence_length": int(self.tokenizer.sequence_length),
            "n_states": int(self.config.n_states),
            "slot_layout": self.config.slot_layout,
            "share_block_encoder": bool(self.config.share_block_encoder),
            "mean_sink_mass": float(sum(sinks) / len(sinks)) if sinks else 0.0,
            "per_layer_sink_mass": sinks,
            "expert_load": loads,
            "router_entropy": entropies,
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

    # -- forward -----------------------------------------------------------

    def _operator_input(
        self, tokens: torch.Tensor, hidden: torch.Tensor, index: int
    ) -> torch.Tensor:
        """Choose what the operator head at slot ``index`` reads.

        See `ReasonerConfig.operator_context`. The "gated" path adds the trunk
        state through a scalar that starts at zero, so equivariance is the
        starting point and context is something the model has to earn.
        """

        if self.config.operator_context == "trunk":
            return hidden[:, index]
        slot = self.operator_norm(tokens[:, index])
        if self.operator_context_gate is None:
            return slot
        return slot + self.operator_context_gate * hidden[:, index]

    def forward(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        thinking_cycles: Optional[int] = None,
        adaptive_thinking: bool = False,
        halt_threshold: float = 0.95,
        stability_tol: float = 1e-3,
        stability_patience: int = 2,
    ) -> ReasonerOutput:
        tokens = self.tokenizer(inputs)
        seq_len = tokens.shape[1]
        device = tokens.device

        positions = torch.arange(seq_len, device=device)
        cos, sin = self.rotary(positions)
        local_cos, local_sin = self.rotary_local(positions)
        empty_keys = torch.zeros(0, dtype=torch.long, device=device)

        hidden = tokens
        for layer in self.layers:
            layer_cos, layer_sin = (
                (local_cos, local_sin) if layer.kind == "swa" else (cos, sin)
            )
            hidden, _ = layer(hidden, layer_cos, layer_sin, positions, empty_keys)

        thinking_info: Dict[str, torch.Tensor] = {}
        if self.thinking_core is not None:
            hidden, thinking_info = self.thinking_core(
                hidden,
                cycles=thinking_cycles,
                adaptive=adaptive_thinking,
                halt_threshold=halt_threshold,
                stability_tol=stability_tol,
                stability_patience=stability_patience,
            )
        hidden = self.norm(hidden)

        state = F.log_softmax(self.initial_state(hidden[:, 0]), dim=-1)
        operators: List[torch.Tensor] = []
        trace: List[torch.Tensor] = [state]
        for index in range(1, seq_len):
            log_transition = self.operator(self._operator_input(tokens, hidden, index))
            operators.append(log_transition)
            # log-space composition: a plain probability product underflows and
            # loses the gradient that has to reach the first operator
            state = torch.logsumexp(state.unsqueeze(-1) + log_transition, dim=1)
            trace.append(state)

        logits = self.class_head(state)

        operator_entropy = None
        if operators:
            stacked = torch.stack(operators, dim=0)
            operator_entropy = -(stacked.exp() * stacked).sum(dim=-1).mean()

        aux = self._collect_aux_loss(logits.device, logits.dtype)
        task_loss: Optional[torch.Tensor] = None
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            task_loss = F.cross_entropy(logits, labels)
            loss = task_loss + aux
            if operator_entropy is not None and self.config.operator_entropy_weight:
                loss = loss + self.config.operator_entropy_weight * operator_entropy
            if thinking_info:
                loss = (
                    loss
                    + self.config.ponder_loss_weight * thinking_info["ponder_cost"]
                    + self.config.consistency_loss_weight * thinking_info["consistency_loss"]
                )

        telemetry = self.telemetry()
        if operator_entropy is not None:
            max_entropy = float(torch.log(torch.tensor(float(self.config.n_states))))
            telemetry["operator_entropy"] = float(operator_entropy.detach())
            telemetry["operator_normalised_entropy"] = (
                float(operator_entropy.detach()) / max_entropy if max_entropy > 0 else 0.0
            )
            telemetry["operator_mean_max_probability"] = float(
                torch.stack(operators, dim=0).exp().max(dim=-1).values.mean().detach()
            )
        telemetry["state_entropy"] = float(-(state.exp() * state).sum(dim=-1).mean().detach())

        quality = thinking_info.get("quality_probability") if thinking_info else None
        continue_p = thinking_info.get("continue_probability") if thinking_info else None
        return ReasonerOutput(
            logits=logits,
            state_log_probs=state,
            operator_log_probs=operators,
            state_trace=trace,
            loss=loss,
            task_loss=task_loss,
            aux_loss=aux,
            operator_entropy=operator_entropy,
            quality_probability=quality,
            continue_probability=continue_p,
            telemetry=telemetry,
        )

    def verifier_loss(self, correctness: torch.Tensor) -> torch.Tensor:
        """Supervised quality/continue objective for the thinking core.

        ``correctness`` is a float tensor in ``{0, 1}`` per example. It is
        broadcast across slot positions because the verifier reads every
        position's refined state. ``p(correct)`` is trained to match it and
        ``p(continue)`` to match its complement -- the v52 contract: keep
        thinking exactly when the current answer is wrong.

        The verifier's own logits carry a trainable temperature that this loss
        trains and the task loss does not, so calibrating the verifier cannot
        drift the answer distribution.
        """

        if self.thinking_core is None:
            raise RuntimeError("model was built without a thinking core")
        logits = getattr(self.thinking_core, "_last_quality_logits", None)
        if logits is None:
            raise RuntimeError(
                "verifier_loss requires a training-mode forward pass that stored quality logits"
            )
        positions = logits.shape[0] // max(1, correctness.shape[0])
        target = correctness.reshape(-1, 1).expand(-1, positions).reshape(-1)
        target = target.to(logits.dtype)
        if target.shape[0] != logits.shape[0]:
            raise ValueError(
                f"correctness broadcast to {target.shape[0]} rows but verifier has {logits.shape[0]}"
            )
        return F.binary_cross_entropy_with_logits(
            logits[:, 0], target
        ) + F.binary_cross_entropy_with_logits(logits[:, 1], 1.0 - target)


# ---------------------------------------------------------------------------
# Construction and checkpoint I/O
# ---------------------------------------------------------------------------


def build_reasoner(**overrides) -> LatentStateReasoner:
    """Convenience constructor: ``build_reasoner(n_states=16, use_moe=False)``."""

    return LatentStateReasoner(ReasonerConfig(**overrides))


def save_reasoner(model: LatentStateReasoner, path, extra: Optional[Dict[str, object]] = None) -> None:
    """Persist config *and* weights.

    v53 has no checkpoint I/O, which is why its API can only ever serve random
    weights. A state dict alone is not reloadable here: the shapes depend on the
    slot layout and `n_states`, so the config travels with the tensors.
    """

    payload: Dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "config": model.config.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }
    if extra:
        payload["extra"] = extra
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_reasoner(path, map_location: str = "cpu") -> Tuple[LatentStateReasoner, Dict[str, object]]:
    """Rebuild a reasoner from a checkpoint written by :func:`save_reasoner`.

    Returns ``(model, payload)`` with the model in ``eval()`` mode. A checkpoint
    whose schema string does not match is refused rather than coerced -- a
    silently mismatched config produces a model that loads and is wrong.
    """

    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"not a {CHECKPOINT_SCHEMA} checkpoint: got schema {payload.get('schema')!r}"
            if isinstance(payload, dict)
            else f"not a {CHECKPOINT_SCHEMA} checkpoint"
        )
    config = ReasonerConfig(**payload["config"])
    model = LatentStateReasoner(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
