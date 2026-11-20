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
    "is_muon_eligible_parameter",
    "build_muon_hybrid_param_groups",
    "MuonHybridOptimizer",
]

# Quintic polynomial coefficients for fast approximate polar factor U V^T
# (Keller Jordan, Muon 2024/2025).
_NS_A = 3.4445
_NS_B = -4.7750
_NS_C = 2.0315


@torch.no_grad()
def newton_schulz_orthogonalize(
    grad: torch.Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute approximate orthogonal polar factor of a 2D tensor via Newton-Schulz."""
    if grad.ndim != 2:
        raise ValueError(f"newton_schulz_orthogonalize expects 2D tensor, got shape {tuple(grad.shape)}")
    orig_dtype = grad.dtype
    x = grad.float()
    transposed = False
    if x.shape[0] > x.shape[1]:
        x = x.t()
        transposed = True
    norm = x.norm()
    if float(norm) < 1e-12:
        return torch.zeros_like(grad)
    x = x / (norm + float(eps))
    n_steps = max(1, int(steps))
    for _ in range(n_steps):
        a_mat = x @ x.t()
        b_mat = _NS_B * a_mat + _NS_C * (a_mat @ a_mat)
        x = _NS_A * x + b_mat @ x
    if transposed:
        x = x.t()
    return x.to(dtype=orig_dtype)


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
    ) -> None:
        defaults = dict(
            lr=float(lr),
            weight_decay=float(weight_decay),
            momentum=float(momentum),
            ns_steps=int(ns_steps),
            nesterov=bool(nesterov),
            betas=tuple(betas),
            eps=float(eps),
            use_muon=False,
        )
        super().__init__(params, defaults)
        self.last_ortho_updates: int = 0
        self.last_adam_updates: int = 0

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        ortho_updates = 0
        adam_updates = 0

        for group in self.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            use_muon = bool(group.get("use_muon", False))

            if use_muon:
                beta = float(group.get("momentum", 0.95))
                ns_steps = int(group.get("ns_steps", 5))
                nesterov = bool(group.get("nesterov", True))
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.is_sparse:
                        raise RuntimeError("MuonHybridOptimizer does not support sparse gradients")
                    state = self.state[p]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(g)
                        state["momentum_buffer"] = buf
                    buf.mul_(beta).add_(g)
                    update_g = g.add(buf, alpha=beta) if nesterov else buf
                    if p.ndim == 2:
                        ortho = newton_schulz_orthogonalize(update_g, steps=ns_steps)
                        aspect = math.sqrt(max(1.0, float(p.shape[0]) / float(max(1, p.shape[1]))))
                        if wd != 0.0:
                            p.mul_(1.0 - lr * wd)
                        p.add_(ortho, alpha=-lr * aspect)
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
        return loss
