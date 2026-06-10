"""v43 Titan-Dreamer frontier expert.

Combines three 2025-26 research directions:

1. Titans neural long-term memory (Behrouz et al., arXiv 2501.00663) + MIRAS:
   a deep 2-layer memory MLP whose *fast weights* are updated at test time by
   surprise gradients with momentum and a forgetting decay, then read with a
   query projection and a MAG-style sigmoid gate. A straight-through path keeps
   the slow (outer) memory weights trainable.
2. TNT chunkwise test-time memorization (arXiv 2511.07343): the inner update is
   done over the whole flattened token chunk in a small number of steps,
   keeping the forward stable and parallel.
3. Dreamer depth-recurrent attention mixtures (arXiv 2601.21582) with
   mixture-of-recursions token routing (Looped LMs, arXiv 2510.25741): the
   latent is refined over depth steps that attend back over previous depth
   states + persistent memory tokens, and each token chooses how many
   recursions to spend via a depth router.

Keeps `weight` / `bias` head keys so warm-starting from base checkpoints is safe.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import ChampionNet


class TitanDreamerExpertHead(nn.Module):
    """Titans test-time memory + depth-recurrent attention mixture head."""

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        mem_dim: int = 512,
        mem_steps: int = 2,
        mem_lr: float = 0.05,
        mem_momentum: float = 0.85,
        mem_forget: float = 0.02,
        depth_steps: int = 3,
        n_persist: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mem_steps = mem_steps
        self.mem_lr = mem_lr
        self.mem_momentum = mem_momentum
        self.mem_forget = mem_forget
        self.depth_steps = depth_steps

        # Warm-start-safe base head
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Shared always-on expert
        self.shared_up = nn.Linear(in_dim, 1024, bias=False)
        self.shared_down = nn.Linear(1024, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(0.0))

        # --- Titans neural memory (slow weights = init of fast weights) ---
        self.titan_w1 = nn.Parameter(torch.randn(mem_dim, in_dim) * 0.02)
        self.titan_b1 = nn.Parameter(torch.zeros(mem_dim))
        self.titan_w2 = nn.Parameter(torch.randn(in_dim, mem_dim) * 0.02)
        self.mem_q = nn.Linear(in_dim, in_dim, bias=False)
        self.mem_k = nn.Linear(in_dim, in_dim, bias=False)
        self.mem_v = nn.Linear(in_dim, in_dim, bias=False)
        self.mem_gate = nn.Linear(in_dim, in_dim, bias=True)   # MAG gate
        self.mem_out = nn.Linear(in_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        # --- Dreamer depth recurrence with persistent tokens ---
        self.persist_tokens = nn.Parameter(torch.randn(n_persist, in_dim) * 0.02)
        self.depth_q = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_k = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_v = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_cells = nn.ModuleList()
        for _ in range(depth_steps):
            self.depth_cells.append(nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 768, bias=False),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(768, in_dim, bias=False),
            ))
        # Mixture-of-recursions: each token distributes credit over depths
        self.recursion_router = nn.Linear(in_dim, depth_steps, bias=True)
        self.depth_out = nn.Linear(in_dim, out_dim, bias=False)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.shared_up.weight, std=0.01)
        nn.init.zeros_(self.shared_down.weight)
        for proj in (self.mem_q, self.mem_k, self.mem_v, self.depth_q, self.depth_k, self.depth_v):
            nn.init.normal_(proj.weight, std=0.02)
        nn.init.zeros_(self.mem_gate.weight)
        nn.init.zeros_(self.mem_gate.bias)
        nn.init.normal_(self.mem_out.weight, std=0.01)
        nn.init.normal_(self.depth_out.weight, std=0.01)
        nn.init.zeros_(self.recursion_router.weight)
        nn.init.zeros_(self.recursion_router.bias)

    # ------------------------------------------------------------------
    def _titan_memory_read(self, xf: torch.Tensor) -> torch.Tensor:
        """Surprise-gradient fast-weight update (momentum + forgetting), then read."""
        q = self.mem_q(xf)
        k = self.mem_k(xf).detach()
        v = self.mem_v(xf).detach()

        fw1 = self.titan_w1.detach().clone()
        fw2 = self.titan_w2.detach().clone()
        b1 = self.titan_b1.detach()
        mom1 = torch.zeros_like(fw1)
        mom2 = torch.zeros_like(fw2)

        for _ in range(self.mem_steps):
            with torch.enable_grad():
                w1_ = fw1.requires_grad_(True)
                w2_ = fw2.requires_grad_(True)
                pred = F.linear(F.silu(F.linear(k, w1_, b1)), w2_)
                surprise = F.mse_loss(pred, v)
                g1, g2 = torch.autograd.grad(surprise, (w1_, w2_))
            mom1 = self.mem_momentum * mom1 - self.mem_lr * g1
            mom2 = self.mem_momentum * mom2 - self.mem_lr * g2
            fw1 = (1.0 - self.mem_forget) * fw1.detach() + mom1
            fw2 = (1.0 - self.mem_forget) * fw2.detach() + mom2

        # Straight-through: fast values, slow-weight gradients
        w1_eff = fw1.detach() + (self.titan_w1 - self.titan_w1.detach())
        w2_eff = fw2.detach() + (self.titan_w2 - self.titan_w2.detach())
        read = F.linear(F.silu(F.linear(q, w1_eff, self.titan_b1)), w2_eff)
        return read * torch.sigmoid(self.mem_gate(xf))

    def _depth_recurrence(self, xf: torch.Tensor) -> torch.Tensor:
        bsz = xf.shape[0]
        scale = self.in_dim ** -0.5
        persist = self.persist_tokens.unsqueeze(0).expand(bsz, -1, -1)
        h = xf
        states = [h]
        depth_feats = []
        for cell in self.depth_cells:
            ctx = torch.cat([persist, torch.stack(states, dim=1)], dim=1)  # (B, P+S, D)
            att = torch.softmax(
                (self.depth_q(h).unsqueeze(1) @ self.depth_k(ctx).transpose(1, 2)) * scale,
                dim=-1,
            )
            attended = (att @ self.depth_v(ctx)).squeeze(1)
            h = h + cell(h + attended)
            states.append(h)
            depth_feats.append(h)
        route = torch.softmax(self.recursion_router(xf), dim=-1)  # (B, depth_steps)
        mix = (torch.stack(depth_feats, dim=1) * route.unsqueeze(-1)).sum(dim=1)
        return self.depth_out(self.dropout(mix))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_logits = F.linear(x, self.weight, self.bias)
        prefix = x.shape[:-1]
        xf = x.reshape(-1, self.in_dim)

        shared = self.shared_norm(self.shared_down(F.silu(self.shared_up(xf))))
        mem_logits = self.mem_out(self._titan_memory_read(xf))
        depth_logits = self._depth_recurrence(xf)

        out = (
            base_logits.reshape(-1, self.out_dim)
            + self.shared_scale * shared
            + self.alpha * mem_logits
            + self.beta * depth_logits
        )
        return out.view(*prefix, self.out_dim)


class ChampionNetTitanDreamerExpert(nn.Module):
    """Backbone wrapper for the v43 TitanDreamerExpertHead."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(TitanDreamerExpertHead(256, 10, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
