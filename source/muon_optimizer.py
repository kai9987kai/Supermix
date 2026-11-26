"""Muon orthogonalized momentum + decoupled AdamW hybrid optimizer (Supermix v102).

Synthesizes Keller Jordan's Muon optimizer (Newton-Schulz quintic polar factor
orthogonalization for 2D hidden weight matrices) and the Muon formulation from
``kai9987kai/OMNIS-COGNITIVE/frontier/optimizers.py`` into a clean, pure-PyTorch
drop-in optimizer compatible with ``MiMoMixModel`` training loops.

Design contract:
- 2D hidden transformation matrices (attention projections, FFN/MoE weights,
  recurrent/connectome linear maps) receive Nesterov momentum followed by 5-step
  quintic Newton-Schulz orthogonalization with rectangular aspect-ratio scaling
  ``sqrt(max(1.0, rows / cols))``.
- Embeddings, ``lm_head``, 1D norms/biases, and scalar gates are automatically
  routed to a decoupled AdamW update so calibration parameters and token lookup
  tables retain exact AdamW conditioning.
- Strictly opt-in (``--optimizer muon_hybrid``); ``--optimizer adamw`` remains
  the default so all prior Supermix runs reproduce unchanged.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch.optim import Optimizer

__all__ = [
    "newton_schulz_orthogonalize",
    "spectral_norm_clip_",
    "is_muon_eligible_parameter",
    "build_muon_hybrid_param_groups",
    "MuonHybridOptimizer",
]

# Quintic polynomial coefficients for fast approximate polar factor U V^T
# (Keller Jordan, Muon 2024/2025).
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315

# Polar Express 5-step minimax-optimal quintic schedule (Amsel et al. 2025,
# arXiv:2505.16932) achieving ~42% lower ||O O^T - I||_F in the same 5 steps.
_POLAR_EXPRESS_COEFFS: Tuple[Tuple[float, float, float], ...] = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


@torch.no_grad()
def newton_schulz_orthogonalize(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    polar_express: bool = False,
) -> torch.Tensor:
    """Compute approximate orthogonal polar factor of a 2D tensor via Newton-Schulz.

    Uses fused ``torch.addmm`` GEMMs for reduced memory bandwidth. Set
    ``polar_express=True`` (v105) for minimax-optimal per-step Polar Express
    polynomial coefficients while keeping default ``False`` bit-compatible.
    """
    if grad.ndim != 2:
        raise ValueError(f"newton_schulz_orthogonalize expects 2D tensor, got shape {tuple(grad.shape)}")
    orig_dtype = grad.dtype
    x = grad.float()
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.t().contiguous()
        transposed = True
    norm = x.norm()
    if float(norm) < 1e-12:
        return torch.zeros_like(grad)
    x = x / (norm + float(eps))
    n_steps = max(1, int(steps))
    for step_idx in range(n_steps):
        if polar_express:
            a_k, b_k, c_k = _POLAR_EXPRESS_COEFFS[min(step_idx, len(_POLAR_EXPRESS_COEFFS) - 1)]
        else:
            a_k, b_k, c_k = _NS_A, _NS_B, _NS_C
        a_mat = torch.mm(x, x.t())
        b_mat = torch.addmm(a_mat, a_mat, a_mat, beta=b_k, alpha=c_k)
        x = torch.addmm(x, b_mat, x, beta=a_k, alpha=1.0)
    if transposed:
        x = x.t()
    return x.to(dtype=orig_dtype)


@torch.no_grad()
def spectral_norm_clip_(
    weight: torch.Tensor,
    *,
    tau: float,
    power_iters: int = 4,
) -> Tuple[bool, float]:
    """In-place spectral-norm clipping (MuonClip / Kimi-K2 QK-Clip style, v106).

    Estimates the dominant singular value of a 2D weight matrix via deterministic
    power iteration; if ``sigma_max > tau``, rescales ``weight`` in place by
    ``tau / sigma_max``. Returns ``(was_clipped, sigma_before)``.
    """
    if weight.ndim != 2:
        raise ValueError(f"spectral_norm_clip_ expects 2D tensor, got shape {tuple(weight.shape)}")
    limit = float(tau)
    if limit <= 0.0:
        return False, 0.0
    w = weight.float()
    rows, cols = int(w.shape[0]), int(w.shape[1])
    if rows == 0 or cols == 0:
        return False, 0.0
    # Deterministic initialization along the column-norm vector for fast convergence
    v = w.pow(2).sum(dim=0).sqrt()
    v_norm = float(v.norm().item())
    if v_norm <= 1e-12:
        return False, 0.0
    v = v / v_norm
    iters = max(1, int(power_iters))
    for _ in range(iters):
        u = w @ v
        un = float(u.norm().item())
        if un <= 1e-12:
            return False, 0.0
        u = u / un
        v = w.t() @ u
        vn = float(v.norm().item())
        if vn <= 1e-12:
            return False, 0.0
        v = v / vn
    sigma = float((w @ v).norm().item())
    if sigma > limit:
        weight.mul_(limit / sigma)
        return True, sigma
    return False, sigma


def is_muon_eligible_parameter(name: str, param: torch.Tensor) -> bool:
    """Return True for 2D hidden weight matrices suitable for Muon orthogonalization."""
    if param.ndim != 2:
        return False
    lower = str(name or "").lower()
    if any(
        excl in lower
        for excl in (
            "embed_tokens",
            "lm_head",
            "embedding",
            "edge_logit",
            "mhc_",
            "engram_memory.tables",
            "attn_res.",
            "value_residual.",
            "ufwm.layer_gates",
            "ufwm.layer_role_emb",
            "ufwm.bank_init",
        )
    ):
        return False
    return min(param.shape) >= 2


def build_muon_hybrid_param_groups(
    model: torch.nn.Module,
    *,
    lr: float = 3e-4,
    muon_lr_scale: float = 5.0,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    no_decay_1d: bool = True,
    cautious_mask: bool = False,
    row_normalize: bool = False,
    polar_express: bool = False,
    spectral_clip_tau: float = 0.0,
    mars_gamma: float = 0.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Split model parameters into Muon (2D hidden) and AdamW (aux/embed/1D) groups."""
    muon_params: List[torch.nn.Parameter] = []
    adam_decay_params: List[torch.nn.Parameter] = []
    adam_no_decay_params: List[torch.nn.Parameter] = []
    seen_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen_ids:
            continue
        seen_ids.add(pid)
        if is_muon_eligible_parameter(name, param):
            muon_params.append(param)
        elif no_decay_1d and (
            param.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "gate" in name.lower()
            or "scale" in name.lower()
            or "temperature" in name.lower()
        ):
            adam_no_decay_params.append(param)
        else:
            adam_decay_params.append(param)

    groups: List[Dict[str, Any]] = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": float(lr) * float(muon_lr_scale),
                "weight_decay": float(weight_decay),
                "momentum": float(momentum),
                "ns_steps": int(ns_steps),
                "nesterov": bool(nesterov),
                "cautious_mask": bool(cautious_mask),
                "row_normalize": bool(row_normalize),
                "polar_express": bool(polar_express),
                "spectral_clip_tau": float(spectral_clip_tau),
                "mars_gamma": float(mars_gamma),
            }
        )
    if adam_decay_params:
        groups.append(
            {
                "params": adam_decay_params,
                "use_muon": False,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "betas": tuple(betas),
                "eps": float(eps),
            }
        )
    if adam_no_decay_params:
        groups.append(
            {
                "params": adam_no_decay_params,
                "use_muon": False,
                "lr": float(lr),
                "weight_decay": 0.0,
                "betas": tuple(betas),
                "eps": float(eps),
            }
        )
    counts = {
        "muon_tensors": len(muon_params),
        "adam_decay_tensors": len(adam_decay_params),
        "adam_no_decay_tensors": len(adam_no_decay_params),
    }
    return groups, counts


class MuonHybridOptimizer(Optimizer):
    """Hybrid optimizer applying Muon to 2D hidden matrices and AdamW to all other tensors."""

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        cautious_mask: bool = False,
        row_normalize: bool = False,
        polar_express: bool = False,
        spectral_clip_tau: float = 0.0,
        mars_gamma: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            ns_steps=int(ns_steps),
            nesterov=bool(nesterov),
            betas=tuple(betas),
            eps=float(eps),
            cautious_mask=bool(cautious_mask),
            row_normalize=bool(row_normalize),
            polar_express=bool(polar_express),
            spectral_clip_tau=float(spectral_clip_tau),
            mars_gamma=float(mars_gamma),
            use_muon=False,
        )
        super().__init__(params, defaults)
        self.last_ortho_updates: int = 0
        self.last_adam_updates: int = 0
        self.last_cautious_active_ratio: float = 1.0
        self.last_polar_express_used: bool = False
        self.last_mars_used: bool = False
        self.last_mars_vr_ratio: float = 0.0
        self.last_spectral_clips: int = 0
        self.last_mean_update_rms: float = 0.0
        self.step_count: int = 0

    def optimizer_telemetry(self) -> Dict[str, Any]:
        return {
            "step_count": int(self.step_count),
            "last_ortho_updates": int(self.last_ortho_updates),
            "last_adam_updates": int(self.last_adam_updates),
            "last_cautious_active_ratio": round(float(self.last_cautious_active_ratio), 6),
            "polar_express_active": bool(self.last_polar_express_used),
            "mars_active": bool(self.last_mars_used),
            "last_mars_vr_ratio": round(float(self.last_mars_vr_ratio), 6),
            "last_spectral_clips": int(self.last_spectral_clips),
            "last_mean_update_rms": round(float(self.last_mean_update_rms), 6),
        }

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ortho_updates = 0
        adam_updates = 0
        spectral_clips = 0
        cautious_ratios: List[float] = []
        mars_vr_ratios: List[float] = []
        update_rms_vals: List[float] = []
        used_polar_express = False
        used_mars = False

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            use_muon = bool(group.get("use_muon", False))

            if use_muon:
                beta = float(group.get("momentum", 0.95))
                ns_steps = int(group.get("ns_steps", 5))
                nesterov = bool(group.get("nesterov", True))
                cautious = bool(group.get("cautious_mask", False))
                row_norm = bool(group.get("row_normalize", False))
                polar_express = bool(group.get("polar_express", False))
                clip_tau = float(group.get("spectral_clip_tau", 0.0))
                mars_gamma = max(0.0, float(group.get("mars_gamma", 0.0)))
                if polar_express:
                    used_polar_express = True
                if mars_gamma > 0.0:
                    used_mars = True
                mars_scale = mars_gamma * (beta / max(1e-4, 1.0 - beta))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("MuonHybridOptimizer does not support sparse gradients")
                    state = self.state[p]
                    eff_g = g
                    if mars_gamma > 0.0:
                        prev_g = state.get("prev_grad")
                        if prev_g is not None:
                            diff = g - prev_g
                            c_t = g.add(diff, alpha=mars_scale)
                            # Norm trust-region clamp (MARS-Clip): ||c_t|| <= 2 * ||g_t||
                            g_norm = float(g.float().norm().item())
                            c_norm = float(c_t.float().norm().item())
                            if g_norm > 1e-12 and c_norm > 2.0 * g_norm:
                                c_t = c_t * ((2.0 * g_norm) / c_norm)
                            vr_ratio = float((c_t - g).float().norm().item()) / max(g_norm, 1e-12)
                            mars_vr_ratios.append(vr_ratio)
                            eff_g = c_t
                            prev_g.copy_(g)
                        else:
                            state["prev_grad"] = g.detach().clone()
                            mars_vr_ratios.append(0.0)
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(eff_g)
                        state["momentum_buffer"] = buf
                    buf.mul_(beta).add_(eff_g)
                    update_g = eff_g.add(buf, alpha=beta) if nesterov else buf
                    if p.ndim == 2:
                        ortho = newton_schulz_orthogonalize(
                            update_g, steps=ns_steps, polar_express=polar_express
                        )
                        if row_norm:
                            # NorMuon-style row-wise RMS variance stabilization (single float cast)
                            ortho_f = ortho.float()
                            sq = ortho_f * ortho_f
                            row_rms = sq.mean(dim=-1, keepdim=True).add_(1e-8).sqrt_()
                            global_rms = sq.mean().add_(1e-8).sqrt_()
                            ortho = (ortho_f * (global_rms / row_rms)).to(dtype=ortho.dtype)
                        if cautious:
                            # Cautious sign-alignment masking (Liang et al.)
                            mask = (ortho * g > 0).to(dtype=ortho.dtype)
                            active_ratio = float(mask.mean().item())
                            cautious_ratios.append(active_ratio)
                            ortho = ortho * mask / max(active_ratio, 1e-3)
                        aspect = math.sqrt(max(1.0, float(p.shape[0]) / float(max(1, p.shape[1]))))
                        step_scale = lr * aspect
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        p.add_(ortho, alpha=-step_scale)
                        update_rms_vals.append(
                            float(ortho.detach().float().pow(2).mean().sqrt().item()) * step_scale
                        )
                        if clip_tau > 0.0:
                            clipped, _ = spectral_norm_clip_(p, tau=clip_tau)
                            if clipped:
                                spectral_clips += 1
                        ortho_updates += 1
                    else:
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        p.add_(update_g, alpha=-lr)
                        adam_updates += 1
            else:
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                eps = float(group.get("eps", 1e-8))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("MuonHybridOptimizer does not support sparse gradients")
                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    step_t = int(state["step"])
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
                    bc1 = 1.0 - beta1**step_t
                    bc2 = 1.0 - beta2**step_t
                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    adam_updates += 1

        self.last_ortho_updates = ortho_updates
        self.last_adam_updates = adam_updates
        self.last_spectral_clips = spectral_clips
        self.last_mean_update_rms = (
            sum(update_rms_vals) / len(update_rms_vals) if update_rms_vals else 0.0
        )
        self.last_cautious_active_ratio = (
            sum(cautious_ratios) / len(cautious_ratios) if cautious_ratios else 1.0
        )
        self.last_polar_express_used = used_polar_express
        self.last_mars_used = used_mars
        self.last_mars_vr_ratio = (
            sum(mars_vr_ratios) / len(mars_vr_ratios) if mars_vr_ratios else 0.0
        )
        self.step_count += 1
        return loss
