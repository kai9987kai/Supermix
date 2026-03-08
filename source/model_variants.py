"""Model variant definitions for the Supermix_27 ChampionNet architecture.

This module defines all classifier head variants (from simple ExpandedClassifierHead
through the advanced MoE heads) and their backbone wrappers. It also provides
utility functions for model construction, weight loading with backward
compatibility, and checkpoint introspection.

Classifier Head Evolution:
    ExpandedClassifierHead        → Single adapter branch
    ExpandedClassifierHeadXL      → Dual adapters + router
    ExpandedClassifierHeadXXL     → Tri-branch + two-stage routing
    ExpandedClassifierHeadXXXL    → Quad-branch + three-stage routing
    ExpandedClassifierHeadUltra   → Five branches + domain-expert calibration
    ExpandedClassifierHeadMega    → Six branches + cross-attention fusion
    GatedExpertClassifierHead     → MoE with Noisy Top-K Gating
    HierarchicalMoEClassifierHead → Two-level hierarchical routing
    DeepExpertClassifierHead      → Aux-loss-free dynamic bias load balancing
    ExpertChoiceClassifierHead    → Expert Choice (experts pick tokens)
    SmarterExpertClassifierHead   → Sigma Gating + LoRA adapters
"""

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import ChampionNet, GatedFFN


SUPPORTED_MODEL_SIZES: Tuple[str, ...] = (
    "base",
    "large",
    "xlarge",
    "xxlarge",
    "xxxlarge",
    "ultralarge",
    "megalarge",
    "ultra_expert",
    "hierarchical_expert",
    "deep_expert",
    "expert_choice",
    "smarter_expert",
    "thought_expert",
    "recursive_expert",
    "reflexive_expert",
    "metacognitive_expert",
    "tree_of_thought_expert",
    "consensus_expert",
)
EXPANSION_DIM_MODEL_SIZES = frozenset({"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"})
EXTRA_EXPANSION_DIM_MODEL_SIZES = frozenset({"xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"})
THIRD_EXPANSION_DIM_MODEL_SIZES = frozenset({"xxlarge", "xxxlarge", "ultralarge", "megalarge"})
FOURTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"xxxlarge", "ultralarge", "megalarge"})
FIFTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"ultralarge", "megalarge"})
SIXTH_EXPANSION_DIM_MODEL_SIZES = frozenset({"megalarge"})


class ExpandedClassifierHead(nn.Module):
    """
    Larger classifier head that keeps base checkpoint compatibility:
    - retains `weight` and `bias` keys used by the original final linear
    - adds an extra adapter branch (256 -> expansion_dim -> 10)
    - starts near original behavior with alpha initialized to 0
    """

    def __init__(self, in_dim: int = 256, out_dim: int = 10, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        adapter_logits = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        return base_logits + self.alpha * adapter_logits


class ExpandedClassifierHeadXL(nn.Module):
    """
    Wider classifier head for extra capacity and richer routing:
    - keeps base-compatible keys (`weight`, `bias`, `adapter_up/down`, `alpha`)
    - adds a second wider adapter branch + learned token-wise router
    - remains warm-start compatible from base/large checkpoints
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 768,
        extra_expansion_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Reuse large-head key names so xlarge can warm-start from large checkpoints.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))

        route = torch.softmax(self.router(x), dim=-1)
        mix = route[..., :1] * a1 + route[..., 1:2] * a2

        return base_logits + self.alpha * a1 + self.beta * mix


class ExpandedClassifierHeadXXL(nn.Module):
    """
    Maximum-capacity classifier head with tri-branch routing:
    - retains all xlarge-compatible keys for warm-starting
    - adds a third branch and second-stage router for richer fusion
    - remains near-base behavior at init via alpha/beta/gamma initialized to 0
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        # New xxl-only branch.
        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_all = torch.softmax(self.router2(x), dim=-1)
        mix_all = route_all[..., :1] * a1 + route_all[..., 1:2] * a2 + route_all[..., 2:3] * a3

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_all


class ExpandedClassifierHeadXXXL(nn.Module):
    """
    Highest-capacity classifier head with quad-branch routing:
    - preserves xxlarge-compatible keys for warm-starting from xxlarge checkpoints
    - adds a fourth branch plus a third routing stage
    - initializes new scaling params at 0 to remain stable at startup
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxlarge key names to preserve compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        # New xxxlarge-only branch.
        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        return base_logits + self.alpha * a1 + self.beta * mix_ab + self.gamma * mix_abc + self.delta * mix_abcd


class ExpandedClassifierHeadUltra(nn.Module):
    """
    Maximum-capacity classifier head with five routed branches:
    - keeps xxxlarge-compatible keys for warm-starting
    - adds a fifth branch + fourth routing stage
    - adds domain-expert calibration (science/code/writing/math friendly) for smarter adaptation
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep xxxlarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        # New ultralarge-only branch.
        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Architecture upgrade: low-cost domain-expert correction + bounded calibration.
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.normal_(self.adapter_up.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.normal_(self.adapter_up_b.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_b.weight)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        nn.init.normal_(self.adapter_up_c.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_c.weight)
        nn.init.zeros_(self.router2.weight)
        nn.init.zeros_(self.router2.bias)

        nn.init.normal_(self.adapter_up_d.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_d.weight)
        nn.init.zeros_(self.router3.weight)
        nn.init.zeros_(self.router3.bias)

        nn.init.normal_(self.adapter_up_e.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.adapter_down_e.weight)
        nn.init.zeros_(self.router4.weight)
        nn.init.zeros_(self.router4.bias)

        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.zeros_(self.domain_router.weight)
        nn.init.zeros_(self.domain_router.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        # Domain-expert calibration branch improves robustness across diverse domains.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.zeta * dom_mix
            + self.theta * calib
        )


class CrossAttentionFusion(nn.Module):
    """
    Lightweight cross-attention module that lets adapter branches attend to each other.
    Input: (B, T, N_branches, out_dim) tensor of stacked branch outputs.
    Output: (B, T, out_dim) fused representation.
    """
    def __init__(self, out_dim: int = 10, n_branches: int = 6, n_heads: int = 2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(1, out_dim // n_heads)
        inner = self.n_heads * self.head_dim
        self.q_proj = nn.Linear(out_dim, inner, bias=False)
        self.k_proj = nn.Linear(out_dim, inner, bias=False)
        self.v_proj = nn.Linear(out_dim, inner, bias=False)
        self.out_proj = nn.Linear(inner, out_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        self.fusion_weight = nn.Linear(out_dim, 1, bias=True)

    def forward(self, branch_stack):
        # branch_stack: (B, T, N, D)
        B, T, N, D = branch_stack.shape
        flat = branch_stack.view(B * T, N, D)
        q = self.q_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B*T, heads, N, head_dim)
        out = out.transpose(1, 2).contiguous().view(B * T, N, self.n_heads * self.head_dim)
        out = self.out_proj(out)  # (B*T, N, D)
        # Weighted sum over branches
        w = torch.softmax(self.fusion_weight(out), dim=1)  # (B*T, N, 1)
        fused = (w * out).sum(dim=1)  # (B*T, D)
        return fused.view(B, T, D)


class ExpandedClassifierHeadMega(nn.Module):
    """
    Maximum-capacity classifier head with six routed branches + cross-attention fusion:
    - keeps ultralarge-compatible keys for warm-starting
    - adds a sixth branch (Mish activation) + fifth routing stage
    - adds CrossAttentionFusion for inter-branch communication
    - adds a reasoning_gate that amplifies logit differences for sharper routing
    - initializes new scale parameters at 0 for stable adaptation
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Keep ultralarge keys for compatibility.
        self.adapter_up = nn.Linear(in_dim, expansion_dim, bias=False)
        self.adapter_down = nn.Linear(expansion_dim, out_dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_b = nn.Linear(in_dim, extra_expansion_dim, bias=False)
        self.adapter_down_b = nn.Linear(extra_expansion_dim, out_dim, bias=False)
        self.router = nn.Linear(in_dim, 2, bias=True)
        self.beta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_c = nn.Linear(in_dim, third_expansion_dim, bias=False)
        self.adapter_down_c = nn.Linear(third_expansion_dim, out_dim, bias=False)
        self.router2 = nn.Linear(in_dim, 3, bias=True)
        self.gamma = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_d = nn.Linear(in_dim, fourth_expansion_dim, bias=False)
        self.adapter_down_d = nn.Linear(fourth_expansion_dim, out_dim, bias=False)
        self.router3 = nn.Linear(in_dim, 4, bias=True)
        self.delta = nn.Parameter(torch.tensor(0.0))

        self.adapter_up_e = nn.Linear(in_dim, fifth_expansion_dim, bias=False)
        self.adapter_down_e = nn.Linear(fifth_expansion_dim, out_dim, bias=False)
        self.router4 = nn.Linear(in_dim, 5, bias=True)
        self.epsilon = nn.Parameter(torch.tensor(0.0))

        # Domain-expert calibration (from ultralarge).
        self.pre_norm = nn.LayerNorm(in_dim)
        self.domain_router = nn.Linear(in_dim, 4, bias=True)
        self.domain_experts = nn.Linear(in_dim, out_dim * 4, bias=False)
        self.calib_gate = nn.Linear(in_dim, out_dim, bias=True)
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.theta = nn.Parameter(torch.tensor(0.0))

        # --- Megalarge-only additions ---
        # Sixth branch with Mish activation.
        self.adapter_up_f = nn.Linear(in_dim, sixth_expansion_dim, bias=False)
        self.adapter_down_f = nn.Linear(sixth_expansion_dim, out_dim, bias=False)
        self.router5 = nn.Linear(in_dim, 6, bias=True)
        self.iota = nn.Parameter(torch.tensor(0.0))

        # Cross-attention fusion lets branches attend to each other.
        self.cross_attn_fusion = CrossAttentionFusion(out_dim=out_dim, n_branches=6, n_heads=2)
        self.kappa = nn.Parameter(torch.tensor(0.0))

        # Reasoning gate amplifies logit differences for sharper routing.
        self.reasoning_gate = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(in_dim // 4, out_dim, bias=True),
        )
        self.lambda_ = nn.Parameter(torch.tensor(0.0))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        for name in ['adapter_up', 'adapter_up_b', 'adapter_up_c', 'adapter_up_d', 'adapter_up_e', 'adapter_up_f']:
            nn.init.normal_(getattr(self, name).weight, mean=0.0, std=0.02)
        for name in ['adapter_down', 'adapter_down_b', 'adapter_down_c', 'adapter_down_d', 'adapter_down_e', 'adapter_down_f']:
            nn.init.zeros_(getattr(self, name).weight)
        for name in ['router', 'router2', 'router3', 'router4', 'router5', 'domain_router']:
            nn.init.zeros_(getattr(self, name).weight)
            nn.init.zeros_(getattr(self, name).bias)
        nn.init.ones_(self.pre_norm.weight)
        nn.init.zeros_(self.pre_norm.bias)
        nn.init.normal_(self.domain_experts.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.calib_gate.weight)
        nn.init.zeros_(self.calib_gate.bias)
        # Reasoning gate init
        for m in self.reasoning_gate:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)

        a1 = self.adapter_down(self.dropout(F.silu(self.adapter_up(x))))
        a2 = self.adapter_down_b(self.dropout(F.gelu(self.adapter_up_b(x))))
        a3 = self.adapter_down_c(self.dropout(F.mish(self.adapter_up_c(x))))
        a4 = self.adapter_down_d(self.dropout(F.relu(self.adapter_up_d(x))))
        a5 = self.adapter_down_e(self.dropout(F.selu(self.adapter_up_e(x))))
        a6 = self.adapter_down_f(self.dropout(F.mish(self.adapter_up_f(x))))

        route_ab = torch.softmax(self.router(x), dim=-1)
        mix_ab = route_ab[..., :1] * a1 + route_ab[..., 1:2] * a2

        route_abc = torch.softmax(self.router2(x), dim=-1)
        mix_abc = route_abc[..., :1] * a1 + route_abc[..., 1:2] * a2 + route_abc[..., 2:3] * a3

        route_abcd = torch.softmax(self.router3(x), dim=-1)
        mix_abcd = (
            route_abcd[..., :1] * a1
            + route_abcd[..., 1:2] * a2
            + route_abcd[..., 2:3] * a3
            + route_abcd[..., 3:4] * a4
        )

        route_abcde = torch.softmax(self.router4(x), dim=-1)
        mix_abcde = (
            route_abcde[..., :1] * a1
            + route_abcde[..., 1:2] * a2
            + route_abcde[..., 2:3] * a3
            + route_abcde[..., 3:4] * a4
            + route_abcde[..., 4:5] * a5
        )

        route_all = torch.softmax(self.router5(x), dim=-1)
        mix_all = (
            route_all[..., :1] * a1
            + route_all[..., 1:2] * a2
            + route_all[..., 2:3] * a3
            + route_all[..., 3:4] * a4
            + route_all[..., 4:5] * a5
            + route_all[..., 5:6] * a6
        )

        # Domain-expert calibration branch.
        h = self.pre_norm(x)
        dom_logits = self.domain_experts(self.dropout(F.silu(h)))
        dom_logits = dom_logits.view(*dom_logits.shape[:-1], 4, base_logits.shape[-1])
        dom_w = torch.softmax(self.domain_router(h), dim=-1).unsqueeze(-1)
        dom_mix = (dom_w * dom_logits).sum(dim=-2)
        calib = torch.tanh(self.calib_gate(h))

        # Cross-attention fusion: stack all 6 branches and let them attend.
        branch_stack = torch.stack([a1, a2, a3, a4, a5, a6], dim=-2)  # (B,T,6,out_dim)
        cross_fused = self.cross_attn_fusion(branch_stack)

        # Reasoning gate: amplify logit differences for sharper decisions.
        reason = torch.tanh(self.reasoning_gate(x))

        return (
            base_logits
            + self.alpha * a1
            + self.beta * mix_ab
            + self.gamma * mix_abc
            + self.delta * mix_abcd
            + self.epsilon * mix_abcde
            + self.iota * mix_all
            + self.zeta * dom_mix
            + self.theta * calib
            + self.kappa * cross_fused
            + self.lambda_ * reason
        )


class GatedExpertClassifierHead(nn.Module):
    """
    Advanced routing head with specialized experts:
    - Top-K gating selects most relevant experts per input
    - Experts utilize diverse activation functions and expansion ratios
    - Residual calibration ensures logit stability
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 6,
        top_k: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.top_k = top_k
        self.n_experts = n_experts

        # Experts with varying architectures
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 512), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 512), out_dim, bias=False) for i in range(n_experts)
        ])
        self.activations = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        for eup in self.experts_up:
            nn.init.normal_(eup.weight, mean=0.0, std=0.01)
        for edown in self.experts_down:
            nn.init.zeros_(edown.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.noise_gate.weight)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        
        # Gating logic with Noisy Top-K
        clean_logits = self.gate(x)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x))
            noise = torch.randn_like(clean_logits) * noise_std
            gate_logits = clean_logits + noise
        else:
            gate_logits = clean_logits

        weights = torch.softmax(gate_logits, dim=-1)
        top_weights, top_idx = torch.topk(weights, k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Expert processing
        expert_outputs = []
        for i in range(self.n_experts):
            out = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x))))
            expert_outputs.append(out)
        expert_stack = torch.stack(expert_outputs, dim=-2) # (B, T, N, D)
        
        # Gather top experts
        # We need to broadcast the indices for gather
        gather_idx = top_idx.unsqueeze(-1).expand(*top_idx.shape, base_logits.shape[-1])
        selected_experts = torch.gather(expert_stack, -2, gather_idx) # (B, T, K, D)
        
        expert_logits = (selected_experts * top_weights.unsqueeze(-1)).sum(dim=-2)
        calib = self.calibration(x)
        
        return base_logits + self.alpha * expert_logits + self.theta * calib


class HierarchicalMoEClassifierHead(nn.Module):
    """
    Next-generation classifier head with hierarchical two-level MoE routing:
    - Domain-level gating selects which expert group is most relevant
    - Per-domain expert gating selects the best expert(s) within the group
    - A shared 'always-on' expert provides a stable baseline signal
    - Per-expert residual LoRA adapters add lightweight correction capacity
    - Per-expert LayerNorm on outputs improves gradient flow
    - Auxiliary load-balancing loss prevents expert collapse during training
    Inspired by DeepSeek-MoE, GShard, and ST-MoE research.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_domain_groups: int = 2,
        experts_per_group: int = 4,
        top_k_domains: int = 1,
        top_k_experts: int = 2,
        lora_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_domain_groups = n_domain_groups
        self.experts_per_group = experts_per_group
        self.n_experts = n_domain_groups * experts_per_group
        self.top_k_domains = top_k_domains
        self.top_k_experts = top_k_experts
        self._aux_loss = torch.tensor(0.0)

        # --- Shared expert (always active, not gated) ---
        shared_dim = 1024
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(0.0))

        # --- Domain-level gating ---
        self.domain_gate = nn.Linear(in_dim, n_domain_groups, bias=False)
        self.domain_noise_gate = nn.Linear(in_dim, n_domain_groups, bias=False)

        # --- Per-domain expert gating ---
        self.expert_gates = nn.ModuleList([
            nn.Linear(in_dim, experts_per_group, bias=False)
            for _ in range(n_domain_groups)
        ])
        self.expert_noise_gates = nn.ModuleList([
            nn.Linear(in_dim, experts_per_group, bias=False)
            for _ in range(n_domain_groups)
        ])

        # --- Experts with varying expansion dims & activations ---
        # 8 experts total: group 0 (analytical): SiLU, GELU, Mish, ReLU
        #                   group 1 (creative):  SELU, Tanh, Softplus, Mish-variant
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i], bias=False) for i in range(self.n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i], out_dim, bias=False) for i in range(self.n_experts)
        ])
        self.activations = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.mish]

        # --- Per-expert output LayerNorm ---
        self.expert_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(self.n_experts)
        ])

        # --- Per-expert residual LoRA adapters ---
        self.lora_down = nn.ModuleList([
            nn.Linear(in_dim, lora_rank, bias=False) for _ in range(self.n_experts)
        ])
        self.lora_up = nn.ModuleList([
            nn.Linear(lora_rank, out_dim, bias=False) for _ in range(self.n_experts)
        ])

        # --- Aggregation ---
        self.alpha = nn.Parameter(torch.tensor(0.0))  # expert mixture scale
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))  # calibration scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        # Shared expert
        nn.init.normal_(self.shared_up.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.shared_down.weight)
        nn.init.ones_(self.shared_norm.weight)
        nn.init.zeros_(self.shared_norm.bias)
        # Domain gates
        nn.init.zeros_(self.domain_gate.weight)
        nn.init.zeros_(self.domain_noise_gate.weight)
        # Per-domain expert gates
        for g in self.expert_gates:
            nn.init.zeros_(g.weight)
        for g in self.expert_noise_gates:
            nn.init.zeros_(g.weight)
        # Experts
        for eup in self.experts_up:
            nn.init.normal_(eup.weight, mean=0.0, std=0.01)
        for edown in self.experts_down:
            nn.init.zeros_(edown.weight)
        # Expert norms
        for norm in self.expert_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        # LoRA adapters
        for ld in self.lora_down:
            nn.init.normal_(ld.weight, mean=0.0, std=0.01)
        for lu in self.lora_up:
            nn.init.zeros_(lu.weight)
        # Calibration
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def _compute_aux_loss(self, gate_probs, expert_mask):
        """
        Auxiliary load-balancing loss (Switch Transformer style).
        Encourages uniform expert utilization.
        gate_probs: (B*T, N) softmax probabilities
        expert_mask: (B*T, N) binary mask of selected experts
        """
        # Fraction of tokens dispatched to each expert
        f = expert_mask.float().mean(dim=0)  # (N,)
        # Average gate probability for each expert
        p = gate_probs.mean(dim=0)  # (N,)
        n = gate_probs.shape[-1]
        return n * (f * p).sum()

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]  # (B, T) or (B,)
        x_flat = x.reshape(-1, x.shape[-1])  # (BT, in_dim)
        BT = x_flat.shape[0]

        # --- Shared expert (always active) ---
        shared_out = self.shared_down(self.dropout(F.silu(self.shared_up(x))))
        shared_out = self.shared_norm(shared_out)

        # --- Domain-level gating ---
        clean_domain = self.domain_gate(x_flat)  # (BT, n_groups)
        if self.training:
            noise_std = F.softplus(self.domain_noise_gate(x_flat))
            domain_logits = clean_domain + torch.randn_like(clean_domain) * noise_std
        else:
            domain_logits = clean_domain
        domain_probs = torch.softmax(domain_logits, dim=-1)  # (BT, n_groups)

        # Select top-k domains
        top_dom_w, top_dom_idx = torch.topk(domain_probs, k=self.top_k_domains, dim=-1)
        top_dom_w = top_dom_w / (top_dom_w.sum(dim=-1, keepdim=True) + 1e-6)

        # --- Process all experts ---
        all_expert_outs = []
        for i in range(self.n_experts):
            act = self.activations[i]
            main_out = self.experts_down[i](self.dropout(act(self.experts_up[i](x_flat))))
            lora_out = self.lora_up[i](self.dropout(F.silu(self.lora_down[i](x_flat))))
            expert_out = self.expert_norms[i](main_out + lora_out)
            all_expert_outs.append(expert_out)
        expert_stack = torch.stack(all_expert_outs, dim=1)  # (BT, n_experts, out_dim)

        # --- Per-domain expert gating & selection ---
        aggregated = x_flat.new_zeros(BT, base_logits.shape[-1])
        all_gate_probs_list = []
        all_expert_mask_list = []

        for d in range(self.n_domain_groups):
            # Expert gating within this domain
            clean_exp = self.expert_gates[d](x_flat)  # (BT, experts_per_group)
            if self.training:
                noise_std = F.softplus(self.expert_noise_gates[d](x_flat))
                exp_logits = clean_exp + torch.randn_like(clean_exp) * noise_std
            else:
                exp_logits = clean_exp
            exp_probs = torch.softmax(exp_logits, dim=-1)  # (BT, experts_per_group)

            top_exp_w, top_exp_idx = torch.topk(exp_probs, k=self.top_k_experts, dim=-1)
            top_exp_w = top_exp_w / (top_exp_w.sum(dim=-1, keepdim=True) + 1e-6)

            # Map local expert indices to global
            global_idx = top_exp_idx + d * self.experts_per_group  # (BT, top_k_experts)
            gather_idx = global_idx.unsqueeze(-1).expand(-1, -1, base_logits.shape[-1])
            selected = torch.gather(expert_stack, 1, gather_idx)  # (BT, top_k, out_dim)
            domain_expert_out = (selected * top_exp_w.unsqueeze(-1)).sum(dim=1)  # (BT, out_dim)

            # Weight by domain probability
            # Get domain d's weight for each token
            # Check if domain d is in the top-k for each token
            domain_d_mask = (top_dom_idx == d).any(dim=-1)  # (BT,)
            # Get the weight for domain d where it was selected
            domain_d_weight = x_flat.new_zeros(BT)
            for k_idx in range(self.top_k_domains):
                mask_k = (top_dom_idx[:, k_idx] == d)
                domain_d_weight = domain_d_weight + mask_k.float() * top_dom_w[:, k_idx]

            aggregated = aggregated + domain_d_weight.unsqueeze(-1) * domain_expert_out

            # Collect for aux loss
            all_gate_probs_list.append(exp_probs)
            exp_mask = torch.zeros(BT, self.experts_per_group, device=x_flat.device)
            exp_mask.scatter_(1, top_exp_idx, 1.0)
            all_expert_mask_list.append(exp_mask)

        # Reshape aggregated back to original shape
        aggregated = aggregated.view(*shape_prefix, -1)

        # --- Auxiliary load-balancing loss ---
        if self.training:
            # Domain-level balance
            dom_mask = torch.zeros(BT, self.n_domain_groups, device=x_flat.device)
            dom_mask.scatter_(1, top_dom_idx, 1.0)
            aux_domain = self._compute_aux_loss(domain_probs, dom_mask)
            # Expert-level balance (per group)
            aux_expert = sum(
                self._compute_aux_loss(gp, em)
                for gp, em in zip(all_gate_probs_list, all_expert_mask_list)
            ) / max(1, self.n_domain_groups)
            self._aux_loss = aux_domain + aux_expert
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device)

        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + self.alpha * aggregated
            + self.theta * calib
        )


class DeepExpertClassifierHead(nn.Module):
    """
    State-of-the-art MoE classifier head featuring:
    - 1 always-active shared expert for universal knowledge.
    - N routed experts using Top-K gating.
    - Auxiliary-loss-free load balancing (dynamic bias adjustment during training).
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.top_k = min(top_k, n_experts)
        self.n_experts = n_experts

        # Shared Expert (always active)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # Routed Experts
        # Using a mix of activations + varying inner dimensions for diversity
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 256), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 256), out_dim, bias=False) for i in range(n_experts)
        ])
        # Cycle activations if n_experts > 6
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Routing mechanism
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        
        # Aux-free load balancing terms
        # These biases are added to the routing logits during training.
        # They don't require gradients; we update them manually.
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.experts_down[i].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.shared_down.weight, a=math.sqrt(5))
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.noise_gate.weight, std=0.02)
        nn.init.constant_(self.calibration.weight, 0)
        nn.init.constant_(self.calibration.bias, 0)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        BT = x_flat.shape[0]

        # Shared Expert
        shared_out = self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        shared_out = self.shared_norm(shared_out)

        # Routed Experts Gating
        clean_logits = self.gate(x_flat)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x_flat))
            noise = torch.randn_like(clean_logits) * noise_std
            gate_logits = clean_logits + noise + self.expert_bias
        else:
            gate_logits = clean_logits

        weights = torch.softmax(gate_logits, dim=-1)
        top_weights, top_idx = torch.topk(weights, k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Process Routed Experts
        all_expert_outs = []
        for i in range(self.n_experts):
            act = self.activations[i]
            out = self.experts_down[i](self.dropout(act(self.experts_up[i](x_flat))))
            out = self.expert_norms[i](out)
            all_expert_outs.append(out)
        expert_stack = torch.stack(all_expert_outs, dim=1)  # (BT, n_experts, out_dim)

        # Gather \u0026 combine chosen experts
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, base_logits.shape[-1])
        selected_experts = torch.gather(expert_stack, 1, gather_idx)  # (BT, K, out_dim)
        routed_out = (selected_experts * top_weights.unsqueeze(-1)).sum(dim=1)  # (BT, out_dim)

        # Reshape combinations back
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        # Update expert bias tracking internally during training if needed (aux-free load balance)
        # Usually implemented in the loop, but we can store load stats here
        if self.training:
            # Fraction of tokens each expert received
            with torch.no_grad():
                expert_mask = torch.zeros(BT, self.n_experts, device=x.device)
                expert_mask.scatter_(1, top_idx, 1.0)
                expert_load = expert_mask.mean(dim=0)  # (n_experts,)
                target_load = self.top_k / self.n_experts
                # Adjust bias: decrease bias for overloaded experts, increase for underloaded
                alpha_bias = 0.1 # learning rate for bias
                self.expert_bias.add_(alpha_bias * (target_load - expert_load))

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class ExpertChoiceClassifierHead(nn.Module):
    """
    Expert Choice (EC) Routing Classifier Head:
    Instead of tokens selecting the top-K experts, experts select the top-K tokens.
    This guarantees 100% load balancing across experts and avoids dropped tokens for active experts.
    Tokens can be routed to a variable number of experts based on their relevance.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        capacity_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.n_experts = n_experts
        # How many tokens an expert will pick, relative to average token load per expert
        self.capacity_factor = capacity_factor
        
        # Routed Experts
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, 1024 + (i * 256), bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(1024 + (i * 256), out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # Routing mechanism: Token-to-expert affinity
        # We need a score for each (token, expert) pair. 
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.noise_gate = nn.Linear(in_dim, n_experts, bias=False)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.experts_down[i].weight, a=math.sqrt(5))
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.noise_gate.weight, std=0.02)
        nn.init.constant_(self.calibration.weight, 0)
        nn.init.constant_(self.calibration.bias, 0)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])  # (BT, D)
        BT = x_flat.shape[0]

        # Expert token capacity (how many tokens each expert will select)
        # Average load per expert is BT / n_experts. We multiply by capacity_factor.
        # e.g., if capacity_factor is 2.0, each expert processes 2 * (BT / n) tokens.
        expert_capacity = min(BT, max(1, int(math.ceil((BT * self.capacity_factor) / self.n_experts))))

        # 1. Compute affinity scores
        clean_logits = self.gate(x_flat)  # (BT, n_experts)
        if self.training:
            noise_std = F.softplus(self.noise_gate(x_flat))
            noise = torch.randn_like(clean_logits) * noise_std
            affinity_logits = clean_logits + noise
        else:
            affinity_logits = clean_logits

        # In Expert Choice, normalization is over experts first to get routing probabilities
        # then experts pick top tokens based on those probabilities.
        # This implies experts compete for tokens based on how much the token "wants" that expert.
        token_to_expert_probs = torch.softmax(affinity_logits, dim=-1)  # (BT, n_experts)

        # Transpose to (n_experts, BT) so each expert can pick top tokens
        expert_to_token_probs = token_to_expert_probs.t()  # (n_experts, BT)

        expert_outputs = torch.zeros((BT, base_logits.shape[-1]), device=x.device, dtype=x.dtype)
        
        # We need a scaling factor later. If an expert picks a token, its output is scaled by 
        # the token_to_expert_prob.
        
        # 2. Each expert processes its top tokens
        for i in range(self.n_experts):
            probs_i = expert_to_token_probs[i]  # (BT,)
            
            # Select top-k tokens for this expert
            top_probs, top_token_idx = torch.topk(probs_i, k=expert_capacity, dim=0)  # (capacity,)
            
            # Gather tokens
            selected_tokens = x_flat[top_token_idx]  # (capacity, D)
            
            # Process through expert i
            act = self.activations[i]
            out = self.experts_down[i](self.dropout(act(self.experts_up[i](selected_tokens))))
            out = self.expert_norms[i](out)  # (capacity, out_dim)
            
            # Scale by routing probability
            out = out * top_probs.unsqueeze(-1)  # (capacity, out_dim)
            
            # Add to the global output buffer using `scatter_add_` or index_add
            # out is (capacity, out_dim). top_token_idx is (capacity,). 
            # expert_outputs is (BT, out_dim).
            expert_outputs.index_add_(0, top_token_idx, out)
            
        expert_outputs = expert_outputs.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.alpha * expert_outputs
            + self.theta * calib
        )


class SmarterExpertClassifierHead(nn.Module):
    """
    Next-generation MoE classifier head featuring:
    - 1 always-active shared anchor expert.
    - 8 diversity-driven routed experts.
    - Sigma Gating (independent sigmoid scores per expert).
    - Aux-loss-free load balancing (DeepSeek-V3 style dynamic bias).
    - Lightweight residual LoRA adapters per expert.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts

        # 1. Shared Anchor Expert
        shared_dim = 2048
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        # Mix of activations
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters for experts
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Routing Mechanism (Sigma Gating)
        self.gate = nn.Linear(in_dim, n_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_down[i].weight)
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        BT = x.reshape(-1, x.shape[-1]).shape[0]
        x_flat = x.reshape(-1, x.shape[-1])

        # Shared Anchor processing
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))

        # Sigma Gating
        clean_logits = self.gate(x_flat)
        if self.training:
            gate_logits = clean_logits + self.expert_bias
        else:
            gate_logits = clean_logits
        
        # Sigma activation
        gate_scores = torch.sigmoid(gate_logits) 

        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        
        expert_stack = torch.stack(all_expert_logits, dim=1) 
        routed_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1) 

        if self.training:
            with torch.no_grad():
                target_load = 1.0 / self.n_experts
                actual_load = gate_scores.mean(dim=0)
                self.expert_bias.add_(0.01 * (target_load - actual_load))

        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class CrossAttentionFusion(nn.Module):
    """
    Multi-head cross-attention fusion for expert representations.
    Allows different expert branches to attend to each other's outputs.
    """
    def __init__(self, out_dim: int, n_branches: int, n_heads: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.n_branches = n_branches
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads

        self.q_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        
        self.fusion_weight = nn.Linear(out_dim, 1)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        x_flat = x.view(B * T, N, D)
        
        q = self.q_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * T, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        out = self.out_proj(out)
        
        # Weighted reduction to single D-dim vector per token
        scores = torch.sigmoid(self.fusion_weight(out)) # (B*T, N, 1)
        fused = (out * scores).sum(dim=1) / (scores.sum(dim=1) + 1e-6)
        
        return fused.view(B, T, D)


class ReasoningCell(nn.Module):
    """
    Sub-module for iterative reasoning steps.
    Combines a residual MLP with LayerNorm and dropout.
    """
    def __init__(self, dim: int = 256, inner_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Linear(dim, inner_dim)
        self.down = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = F.silu

    def forward(self, x):
        h = self.norm(x)
        h = self.down(self.dropout(self.act(self.up(h))))
        return x + h


class ThoughtExpertClassifierHead(nn.Module):
    """
    Ultra-advanced MoE classifier head featuring iterative reasoning:
    - 1 always-active shared anchor expert.
    - 8 diversity-driven routed experts.
    - 3-step reasoning loop that refines the feature representation.
    - Sigma Gating (independent sigmoid scores) at each step.
    - Cross-expert attention fusion integrated into the "thinking" process.
    - Lightweight residual LoRA adapters per expert.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        reasoning_steps: int = 3,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts
        self.reasoning_steps = reasoning_steps

        # 1. Shared Anchor Expert
        shared_dim = 2048
        self.shared_up = nn.Linear(in_dim, shared_dim, bias=False)
        self.shared_down = nn.Linear(shared_dim, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters for experts
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Reasoning Transformation
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout) for _ in range(reasoning_steps)
        ])
        
        # 5. Iterative Routing (Sigma Gating)
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(reasoning_steps)
        ])
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # 6. Cross-Expert Attention Fusion
        self.cross_attn = CrossAttentionFusion(out_dim=out_dim, n_branches=n_experts, n_heads=2)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.experts_down[i].weight)
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
        
        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # Shared Anchor processing
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))

        # Pre-compute expert candidates (B*T, n_experts, out_dim)
        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        expert_stack = torch.stack(all_expert_logits, dim=1) 

        # Iterative Reasoning Loop
        current_features = x_flat
        total_routed_out = 0
        
        for step in range(self.reasoning_steps):
            # 1. Refine features
            current_features = self.reasoning_cells[step](current_features)
            
            # 2. Gate experts based on refined features
            gate_logits = self.gates[step](current_features)
            if self.training:
                gate_logits = gate_logits + self.expert_bias
            gate_scores = torch.sigmoid(gate_logits) # (B*T, n_experts)
            
            # 3. Fuse experts for this step
            step_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1)
            
            # 4. Attention-based cross-expert correction (residual)
            # Re-weight experts with attention to find hidden correlations
            # Shape: expert_stack is (BT, N, D), we need (BT, 1, N, D) for CrossAttentionFusion
            attn_fused = self.cross_attn(expert_stack.unsqueeze(1)).squeeze(1)
            
            total_routed_out = total_routed_out + step_out + 0.1 * attn_fused

            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.expert_bias.add_(0.01 * (target_load - actual_load))

        routed_out = total_routed_out / self.reasoning_steps
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class RecursiveThoughtExpertHead(nn.Module):
    """
    Generation v14: RecursiveThoughtExpert (Recursive reasoning + ACE + Multi-Head Routing)
    Features:
    - Adaptive Reasoning Depth (ACE): Early exit logic based on confidence.
    - Multi-Head Sigma Gating (2 heads): Richer routing subspace.
    - Hierarchical Shared Experts: Global (2048) + Local (512) experts.
    - Cross-Expert Attention Fusion (Residual).
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 8,
        reasoning_steps: int = 3,
        n_heads: int = 2,
        lora_rank: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts
        self.reasoning_steps = reasoning_steps
        self.n_heads = n_heads

        # 1. Hierarchical Shared Experts
        # Global
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # Local (New in v14)
        self.local_up = nn.Linear(in_dim, 512, bias=False)
        self.local_down = nn.Linear(512, out_dim, bias=False)
        self.local_norm = nn.LayerNorm(out_dim)
        self.local_scale = nn.Parameter(torch.tensor(0.5))

        # 2. Routed Experts with diversity
        expansion_dims = [1024, 1536, 2048, 2560, 1024, 1536, 2048, 3072]
        self.experts_up = nn.ModuleList([
            nn.Linear(in_dim, expansion_dims[i % 8], bias=False) for i in range(n_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(expansion_dims[i % 8], out_dim, bias=False) for i in range(n_experts)
        ])
        base_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.activations = [base_acts[i % len(base_acts)] for i in range(n_experts)]
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 3. Residual LoRA Adapters
        self.lora_down = nn.ModuleList([nn.Linear(in_dim, lora_rank, bias=False) for _ in range(n_experts)])
        self.lora_up = nn.ModuleList([nn.Linear(lora_rank, out_dim, bias=False) for _ in range(n_experts)])

        # 4. Reasoning & ACE
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout) for _ in range(reasoning_steps)
        ])
        self.exit_gates = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(reasoning_steps)
        ])
        
        # 5. Multi-Head Sigma Gating (Flattened for registration)
        self.gates = nn.ModuleList([
            nn.Linear(in_dim, n_experts, bias=False) for _ in range(reasoning_steps * n_heads)
        ])
        self.register_buffer("expert_bias", torch.zeros(n_experts))

        # 6. Cross-Expert Attention Fusion
        self.cross_attn = CrossAttentionFusion(out_dim=out_dim, n_branches=n_experts, n_heads=2)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.calibration = nn.Linear(in_dim, out_dim, bias=True)
        self.theta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        nn.init.kaiming_uniform_(self.local_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.local_down.weight)
        
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.experts_down[i].weight, std=0.01) # Small initial signal
            nn.init.normal_(self.lora_down[i].weight, std=0.02)
            nn.init.zeros_(self.lora_up[i].weight)
        
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
        
        for e in self.exit_gates:
            nn.init.constant_(e.bias, -3.0) # Start with low exit prob

        nn.init.zeros_(self.calibration.weight)
        nn.init.zeros_(self.calibration.bias)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        
        # 1. Global Shared Expert
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))
        # 2. Local Shared Expert
        local_out = self.local_norm(self.local_down(self.dropout(F.gelu(self.local_up(x_flat)))))

        # Pre-compute expert candidates
        all_expert_logits = []
        for i in range(self.n_experts):
            core = self.experts_down[i](self.dropout(self.activations[i](self.experts_up[i](x_flat))))
            lora = self.lora_up[i](self.lora_down[i](x_flat))
            out = self.expert_norms[i](core + lora)
            all_expert_logits.append(out)
        expert_stack = torch.stack(all_expert_logits, dim=1) 

        # Iterative Reasoning Loop with ACE
        current_features = x_flat
        total_routed_out = torch.zeros_like(expert_stack[:, 0, :])
        cumulative_exit_prob = torch.zeros(x_flat.shape[0], 1, device=x.device)
        depth_count = 0
        
        for step in range(self.reasoning_steps):
            depth_count += 1
            # A. Refine features
            current_features = self.reasoning_cells[step](current_features)
            
            # B. Multi-Head Sigma Gating
            head_logits = []
            for h in range(self.n_heads):
                idx = step * self.n_heads + h
                head_logits.append(self.gates[idx](current_features))
            # Average head logits (Sigma Gating)
            gate_logits = torch.stack(head_logits, dim=0).mean(dim=0)
            
            if self.training:
                gate_logits = gate_logits + self.expert_bias
            gate_scores = torch.sigmoid(gate_logits) # (B*T, n_experts)
            
            # C. ACE: Early Exit check
            exit_logit = self.exit_gates[step](current_features)
            exit_prob = torch.sigmoid(exit_logit) # (B*T, 1)
            
            # D. Fuse experts for this step
            step_out = (expert_stack * gate_scores.unsqueeze(-1)).sum(dim=1)
            
            # E. Attention-based cross-expert correction
            attn_fused = self.cross_attn(expert_stack.unsqueeze(1)).squeeze(1)
            
            # Weighted contribution based on current depth
            total_routed_out = total_routed_out + (1.0 - cumulative_exit_prob) * (step_out + 0.1 * attn_fused)
            
            # Update cumulative exit probability
            cumulative_exit_prob = cumulative_exit_prob + (1.0 - cumulative_exit_prob) * exit_prob

            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.expert_bias.add_(0.01 * (target_load - actual_load))
            
            # Early exit condition
            if not self.training and cumulative_exit_prob.mean() > 0.9:
                break

        routed_out = total_routed_out / depth_count
        routed_out = routed_out.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        local_out = local_out.view(*shape_prefix, -1)
        calib = self.calibration(x)

        return (
            base_logits
            + self.shared_scale * shared_out
            + self.local_scale * local_out
            + (self.alpha + 1e-4) * routed_out
            + self.theta * calib
        )


class ChampionNetRecursiveExpert(nn.Module):
    """
    Backbone-compatible model with RecursiveThoughtExpertHead.
    """
    def __init__(self, n_experts: int = 8, reasoning_steps: int = 3, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(RecursiveThoughtExpertHead(
            256, 10, 
            n_experts=n_experts, 
            reasoning_steps=reasoning_steps, 
            lora_rank=lora_rank, 
            dropout=dropout
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




class ReflexiveThoughtExpertHead(nn.Module):
    """
    Generation v15: ReflexiveThoughtExpert (Self-Reflection MoE)
    Features:
    - Initial Pass: Generates baseline logits using MoE routing.
    - Critique Pass: Takes original features + initial logits to self-correct using specialized critique experts.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_experts = n_experts

        # 1. Initial Pass Experts
        self.initial_up = nn.ModuleList([nn.Linear(in_dim, 1024, bias=False) for _ in range(n_experts)])
        self.initial_down = nn.ModuleList([nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)])
        self.initial_gate = nn.Linear(in_dim, n_experts, bias=False)
        self.initial_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        # 2. Critique Pass Experts (Input: features + intermediate logits)
        critique_dim = in_dim + out_dim
        self.critique_up = nn.ModuleList([nn.Linear(critique_dim, 1024, bias=False) for _ in range(n_experts)])
        self.critique_down = nn.ModuleList([nn.Linear(1024, out_dim, bias=False) for _ in range(n_experts)])
        self.critique_gate = nn.Linear(critique_dim, n_experts, bias=False)
        self.critique_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.zeros_(self.initial_gate.weight)
        nn.init.zeros_(self.critique_gate.weight)
        for i in range(self.n_experts):
            nn.init.kaiming_uniform_(self.initial_up[i].weight, a=5**0.5)
            nn.init.normal_(self.initial_down[i].weight, std=0.01)
            nn.init.kaiming_uniform_(self.critique_up[i].weight, a=5**0.5)
            nn.init.normal_(self.critique_down[i].weight, std=0.01)

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        
        # --- Initial Pass ---
        gate_init = torch.softmax(self.initial_gate(x), dim=-1) # (B, T, E)
        init_outs = []
        for i in range(self.n_experts):
            h = self.dropout(F.silu(self.initial_up[i](x)))
            init_outs.append(self.initial_norms[i](self.initial_down[i](h)))
        init_stack = torch.stack(init_outs, dim=-2) # (B, T, E, D)
        init_fused = (init_stack * gate_init.unsqueeze(-1)).sum(dim=-2)
        
        # Intermediate prediction
        inter_logits = base_logits + self.alpha * init_fused
        
        # --- Critique Pass ---
        crit_in = torch.cat([x, inter_logits], dim=-1) 
        gate_crit = torch.softmax(self.critique_gate(crit_in), dim=-1)
        crit_outs = []
        for i in range(self.n_experts):
            h = self.dropout(F.gelu(self.critique_up[i](crit_in)))
            crit_outs.append(self.critique_norms[i](self.critique_down[i](h)))
        crit_stack = torch.stack(crit_outs, dim=-2)
        crit_fused = (crit_stack * gate_crit.unsqueeze(-1)).sum(dim=-2)
        
        # Final prediction
        return inter_logits + self.beta * crit_fused


class ChampionNetReflexiveExpert(nn.Module):
    """
    Backbone-compatible model with ReflexiveThoughtExpertHead.
    """
    def __init__(self, n_experts: int = 4, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ReflexiveThoughtExpertHead(256, 10, n_experts=n_experts, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MetaCognitiveExpertHead(nn.Module):
    """
    Generation v16: MetaCognitiveExpert (Iterative Reflection + PonderNet Halting)
    Combines the best ideas from v14 (recursive reasoning, shared experts, ACE)
    and v15 (self-reflection critique pass) into a unified "think → critique → refine"
    loop with learned adaptive halting.

    Features:
    - Shared Expert: Always-on global expert for stable signal.
    - ReasoningCells: Per-step feature refinement with residual MLP.
    - Proposal MoE: Generates predictions via routed experts at each step.
    - Critique MoE: Inspects [features || proposal_logits] to output corrections.
    - PonderNet Halting: Learned halt probability per step for adaptive depth.
    - Dynamic Bias Load Balancing: Buffer biases maintain expert utilization.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_proposal_experts: int = 6,
        n_critique_experts: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.n_proposal_experts = n_proposal_experts
        self.n_critique_experts = n_critique_experts
        self.reasoning_steps = reasoning_steps

        # 1. Shared Expert (always-on)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        # 2. Reasoning Cells (per step)
        self.reasoning_cells = nn.ModuleList([
            ReasoningCell(in_dim, inner_dim=512, dropout=dropout)
            for _ in range(reasoning_steps)
        ])

        # 3. Proposal MoE (per step routing)
        prop_dims = [1024, 1536, 2048, 2560, 1024, 1536]
        prop_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.proposal_up = nn.ModuleList([
            nn.Linear(in_dim, prop_dims[i % len(prop_dims)], bias=False)
            for i in range(n_proposal_experts)
        ])
        self.proposal_down = nn.ModuleList([
            nn.Linear(prop_dims[i % len(prop_dims)], out_dim, bias=False)
            for i in range(n_proposal_experts)
        ])
        self.proposal_acts = [prop_acts[i % len(prop_acts)] for i in range(n_proposal_experts)]
        self.proposal_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(n_proposal_experts)
        ])
        self.proposal_gates = nn.ModuleList([
            nn.Linear(in_dim, n_proposal_experts, bias=False)
            for _ in range(reasoning_steps)
        ])
        self.register_buffer("proposal_bias", torch.zeros(n_proposal_experts))

        # 4. Critique MoE (input: features + proposal logits)
        critique_dim = in_dim + out_dim
        self.critique_up = nn.ModuleList([
            nn.Linear(critique_dim, 1024, bias=False)
            for _ in range(n_critique_experts)
        ])
        self.critique_down = nn.ModuleList([
            nn.Linear(1024, out_dim, bias=False)
            for _ in range(n_critique_experts)
        ])
        self.critique_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(n_critique_experts)
        ])
        self.critique_gates = nn.ModuleList([
            nn.Linear(critique_dim, n_critique_experts, bias=False)
            for _ in range(reasoning_steps)
        ])

        # 5. Halting Gates (PonderNet-style)
        self.halt_gates = nn.ModuleList([
            nn.Linear(in_dim, 1) for _ in range(reasoning_steps)
        ])

        # Aggregation
        self.alpha = nn.Parameter(torch.tensor(0.0))  # proposal scale
        self.beta = nn.Parameter(torch.tensor(0.0))   # critique scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)

        for i in range(self.n_proposal_experts):
            nn.init.kaiming_uniform_(self.proposal_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.proposal_down[i].weight, std=0.01)
        for i in range(self.n_critique_experts):
            nn.init.kaiming_uniform_(self.critique_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.critique_down[i].weight, std=0.01)

        for g in self.proposal_gates:
            nn.init.normal_(g.weight, std=0.02)
        for g in self.critique_gates:
            nn.init.normal_(g.weight, std=0.02)
        for h in self.halt_gates:
            nn.init.constant_(h.bias, -3.0)  # start with low halt probability

    def forward(self, x):
        base_logits = F.linear(x, self.weight, self.bias)
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])

        # 1. Shared Expert (always-on)
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )

        # 2. Pre-compute proposal expert outputs
        prop_outs = []
        for i in range(self.n_proposal_experts):
            h = self.dropout(self.proposal_acts[i](self.proposal_up[i](x_flat)))
            prop_outs.append(self.proposal_norms[i](self.proposal_down[i](h)))
        prop_stack = torch.stack(prop_outs, dim=1)  # (BT, E_prop, D)

        # 3. Iterative Reflect Loop
        current_features = x_flat
        total_proposal = torch.zeros_like(base_logits.reshape(-1, base_logits.shape[-1]))
        total_critique = torch.zeros_like(total_proposal)
        cumulative_halt = torch.zeros(x_flat.shape[0], 1, device=x.device)
        depth_count = 0

        for step in range(self.reasoning_steps):
            depth_count += 1

            # A. Refine features
            current_features = self.reasoning_cells[step](current_features)

            # B. Proposal MoE routing (Sigma Gating)
            gate_logits = self.proposal_gates[step](current_features)
            if self.training:
                gate_logits = gate_logits + self.proposal_bias
            gate_scores = torch.sigmoid(gate_logits)  # (BT, E_prop)
            step_proposal = (prop_stack * gate_scores.unsqueeze(-1)).sum(dim=1)

            # C. Critique MoE: inspect [features || proposal]
            crit_in = torch.cat([current_features, step_proposal], dim=-1)
            crit_gate = torch.softmax(
                self.critique_gates[step](crit_in), dim=-1
            )  # (BT, E_crit)
            crit_outs = []
            for i in range(self.n_critique_experts):
                h = self.dropout(F.gelu(self.critique_up[i](crit_in)))
                crit_outs.append(self.critique_norms[i](self.critique_down[i](h)))
            crit_stack = torch.stack(crit_outs, dim=1)  # (BT, E_crit, D)
            step_critique = (crit_stack * crit_gate.unsqueeze(-1)).sum(dim=1)

            # D. Halting gate (PonderNet-style)
            halt_prob = torch.sigmoid(self.halt_gates[step](current_features))

            # E. Accumulate weighted by remaining probability mass
            remaining = 1.0 - cumulative_halt
            total_proposal = total_proposal + remaining * step_proposal
            total_critique = total_critique + remaining * step_critique
            cumulative_halt = cumulative_halt + remaining * halt_prob

            # F. Dynamic bias load balancing
            if self.training:
                with torch.no_grad():
                    target_load = 1.0 / self.n_proposal_experts
                    actual_load = gate_scores.mean(dim=0)
                    self.proposal_bias.add_(0.01 * (target_load - actual_load))

            # G. Early exit during inference
            if not self.training and cumulative_halt.mean() > 0.9:
                break

        # Normalize by effective depth
        total_proposal = total_proposal / max(1, depth_count)
        total_critique = total_critique / max(1, depth_count)

        # Reshape back
        total_proposal = total_proposal.view(*shape_prefix, -1)
        total_critique = total_critique.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)

        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * total_proposal
            + (self.beta + 1e-4) * total_critique
        )


class ChampionNetMetaCognitiveExpert(nn.Module):
    """
    Backbone-compatible model with MetaCognitiveExpertHead (v16).
    """
    def __init__(
        self,
        n_proposal_experts: int = 6,
        n_critique_experts: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(MetaCognitiveExpertHead(
            256, 10,
            n_proposal_experts=n_proposal_experts,
            n_critique_experts=n_critique_experts,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TreeOfThoughtExpertHead(nn.Module):
    """
    Generation v17: Tree-of-Thought Expert (Latent Beam Search)
    Embeds true test-time compute scaling into the forward pass.
    Instead of a single trajectory, maintains a beam of k states.
    At each step, m Action Experts propose modifications, creating k*m candidates.
    A Value Network scores them, and the top-k are kept.
    """
    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_action_experts: int = 6,
        beam_size: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_action_experts = n_action_experts
        self.beam_size = beam_size
        self.reasoning_steps = reasoning_steps
        
        # Base projection
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        # 1. Global Shared Expert (always-on)
        self.shared_up = nn.Linear(in_dim, 2048, bias=False)
        self.shared_down = nn.Linear(2048, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))
        
        # 2. Value Network (Scorer)
        # Evaluates the "goodness" of a latent state.
        self.value_net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )
        
        # 3. Action MoE (Proposer)
        # We use a set of experts that propose feature modifications
        self.action_up = nn.ModuleList([
            nn.Linear(in_dim, 1024, bias=False) for _ in range(n_action_experts)
        ])
        self.action_down = nn.ModuleList([
            nn.Linear(1024, in_dim, bias=False) for _ in range(n_action_experts)
        ])
        self.action_norms = nn.ModuleList([
            nn.LayerNorm(in_dim) for _ in range(n_action_experts)
        ])
        
        # Final projection from feature to out_dim
        self.final_up = nn.Linear(in_dim, 1024, bias=False)
        self.final_down = nn.Linear(1024, out_dim, bias=False)
        self.final_norm = nn.LayerNorm(out_dim)
        
        self.alpha = nn.Parameter(torch.tensor(0.0))  # tree scale
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.shared_down.weight, std=0.01)
        
        for p in self.value_net.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)
                
        for i in range(self.n_action_experts):
            nn.init.kaiming_uniform_(self.action_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.action_down[i].weight, std=0.01)
            
        nn.init.kaiming_uniform_(self.final_up.weight, a=math.sqrt(5))
        nn.init.normal_(self.final_down.weight, std=0.01)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)
        
        base_logits = F.linear(x_flat, self.weight, self.bias)
        
        # 1. Shared Expert
        shared_out = self.shared_norm(
            self.shared_down(self.dropout(F.silu(self.shared_up(x_flat))))
        )
        
        # 2. Tree-of-Thought Search
        # Initial beam state: K=1
        beam_states = x_flat.unsqueeze(1) # (N, 1, D)
        
        for step in range(self.reasoning_steps):
            K = beam_states.shape[1]
            
            # A. Expansion: Each action expert proposes a modification to every state in the beam
            nk_states = beam_states.reshape(N * K, self.in_dim)
            
            candidates = []
            for i in range(self.n_action_experts):
                h = self.dropout(F.gelu(self.action_up[i](nk_states)))
                delta = self.action_norms[i](self.action_down[i](h))
                new_state = nk_states + delta
                candidates.append(new_state)
                
            # Stack candidates
            cand_stack = torch.stack(candidates, dim=1)
            cand_stack = cand_stack.reshape(N, K * self.n_action_experts, self.in_dim)
            
            # B. Evaluation (Value Network)
            flat_cands = cand_stack.reshape(N * K * self.n_action_experts, self.in_dim)
            scores = self.value_net(flat_cands).squeeze(-1) # (N * K * M)
            scores = scores.reshape(N, K * self.n_action_experts) # (N, K*M)
            
            # C. Selection (Beam Search)
            K_next = min(self.beam_size, K * self.n_action_experts)
            
            topk_scores, topk_indices = torch.topk(scores, k=K_next, dim=1) # (N, K_next)
            
            expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, self.in_dim)
            beam_states = torch.gather(cand_stack, 1, expanded_indices) # (N, K_next, D)
            
        # 3. Aggregation across the final beam
        final_K = beam_states.shape[1]
        flat_final = beam_states.reshape(N * final_K, self.in_dim)
        
        h_final = self.dropout(F.silu(self.final_up(flat_final)))
        final_preds = self.final_norm(self.final_down(h_final)) # (N * final_K, out_dim)
        final_preds = final_preds.reshape(N, final_K, self.out_dim) # (N, K, out_dim)
        
        final_scores = self.value_net(flat_final).squeeze(-1).reshape(N, final_K) # (N, K)
        agg_weights = torch.softmax(final_scores, dim=1) # (N, K)
        
        tree_fused = (final_preds * agg_weights.unsqueeze(-1)).sum(dim=1)
        
        base_logits = base_logits.view(*shape_prefix, -1)
        shared_out = shared_out.view(*shape_prefix, -1)
        tree_fused = tree_fused.view(*shape_prefix, -1)
        
        return (
            base_logits
            + self.shared_scale * shared_out
            + (self.alpha + 1e-4) * tree_fused
        )


class ChampionNetTreeOfThoughtExpert(nn.Module):
    """
    Backbone-compatible model with TreeOfThoughtExpertHead (v17).
    """
    def __init__(
        self,
        n_action_experts: int = 6,
        beam_size: int = 4,
        reasoning_steps: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(TreeOfThoughtExpertHead(
            256, 10,
            n_action_experts=n_action_experts,
            beam_size=beam_size,
            reasoning_steps=reasoning_steps,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConsensusExpertHead(nn.Module):
    """
    Generation v18: Consensus Expert (Architectural Diversity + Learned Arbitration)
    Uses three fundamentally different computational paradigms as independent
    reasoning pathways, then a Consensus Network arbitrates.

    Pathways:
    1. MLP Pathway — Deep feed-forward MoE experts (fast pattern matching)
    2. Attention Pathway — Self-attention over learned expert embeddings (relational reasoning)
    3. Convolutional Pathway — 1D convolution over the feature vector (local pattern detection)

    A Consensus Network evaluates inter-pathway agreement and produces the final answer.
    """

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_mlp_experts: int = 6,
        n_attn_heads: int = 4,
        conv_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_mlp_experts = n_mlp_experts

        # Base projection
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # --- Pathway 1: MLP MoE ---
        mlp_dims = [1024, 1536, 2048, 1024, 1536, 2048]
        mlp_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh]
        self.mlp_up = nn.ModuleList([
            nn.Linear(in_dim, mlp_dims[i % len(mlp_dims)], bias=False)
            for i in range(n_mlp_experts)
        ])
        self.mlp_down = nn.ModuleList([
            nn.Linear(mlp_dims[i % len(mlp_dims)], out_dim, bias=False)
            for i in range(n_mlp_experts)
        ])
        self.mlp_acts = [mlp_acts[i % len(mlp_acts)] for i in range(n_mlp_experts)]
        self.mlp_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_mlp_experts)])
        self.mlp_gate = nn.Linear(in_dim, n_mlp_experts, bias=False)

        # --- Pathway 2: Self-Attention ---
        self.n_attn_heads = n_attn_heads
        self.attn_embed = nn.Parameter(torch.randn(8, in_dim) * 0.02)
        self.attn_q = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_k = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_v = nn.Linear(in_dim, in_dim, bias=False)
        self.attn_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_norm = nn.LayerNorm(out_dim)
        self.attn_scale = (in_dim // n_attn_heads) ** -0.5

        # --- Pathway 3: 1D Convolution ---
        self.conv_channels = conv_channels
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.conv_pool = nn.AdaptiveAvgPool1d(1)
        self.conv_proj = nn.Linear(conv_channels, out_dim, bias=False)
        self.conv_norm = nn.LayerNorm(out_dim)

        # --- Consensus Network (Arbiter) ---
        consensus_in = 3 * out_dim
        self.consensus = nn.Sequential(
            nn.Linear(consensus_in, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        for i in range(self.n_mlp_experts):
            nn.init.kaiming_uniform_(self.mlp_up[i].weight, a=math.sqrt(5))
            nn.init.normal_(self.mlp_down[i].weight, std=0.01)
        nn.init.normal_(self.mlp_gate.weight, std=0.02)

        nn.init.kaiming_uniform_(self.attn_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.attn_v.weight, a=math.sqrt(5))
        nn.init.normal_(self.attn_proj.weight, std=0.01)

        nn.init.normal_(self.conv_proj.weight, std=0.01)

        for p in self.consensus.parameters():
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.zeros_(p)

    def forward(self, x):
        N = x.shape[0] * x.shape[1] if x.dim() == 3 else x.shape[0]
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(N, self.in_dim)

        base_logits = F.linear(x_flat, self.weight, self.bias)

        # --- Pathway 1: MLP MoE ---
        gate_scores = torch.sigmoid(self.mlp_gate(x_flat))
        mlp_outs = []
        for i in range(self.n_mlp_experts):
            h = self.dropout(self.mlp_acts[i](self.mlp_up[i](x_flat)))
            mlp_outs.append(self.mlp_norms[i](self.mlp_down[i](h)))
        mlp_stack = torch.stack(mlp_outs, dim=1)
        mlp_pred = (mlp_stack * gate_scores.unsqueeze(-1)).sum(dim=1)

        # --- Pathway 2: Self-Attention ---
        embeds = self.attn_embed.unsqueeze(0).expand(N, -1, -1)
        q = self.attn_q(x_flat).unsqueeze(1)
        k = self.attn_k(embeds)
        v = self.attn_v(embeds)
        attn_weights = torch.softmax(
            (q * k).sum(dim=-1, keepdim=True) * self.attn_scale, dim=1
        )
        attn_out = (attn_weights * v).sum(dim=1)
        attn_pred = self.attn_norm(self.attn_proj(attn_out))

        # --- Pathway 3: 1D Convolution ---
        x_conv = x_flat.unsqueeze(1)
        h_conv = F.gelu(self.conv1(x_conv))
        h_conv = F.gelu(self.conv2(h_conv))
        h_conv = self.conv_pool(h_conv).squeeze(-1)
        conv_pred = self.conv_norm(self.conv_proj(h_conv))

        # --- Consensus Network ---
        pathway_preds = torch.stack([mlp_pred, attn_pred, conv_pred], dim=1)
        consensus_in = torch.cat([mlp_pred, attn_pred, conv_pred], dim=-1)
        consensus_weights = torch.softmax(self.consensus(consensus_in), dim=-1)
        consensus_fused = (pathway_preds * consensus_weights.unsqueeze(-1)).sum(dim=1)

        consensus_fused = consensus_fused.view(*shape_prefix, -1)
        base_logits = base_logits.view(*shape_prefix, -1)

        return base_logits + (self.alpha + 1e-4) * consensus_fused


class ChampionNetConsensusExpert(nn.Module):
    """
    Backbone-compatible model with ConsensusExpertHead (v18).
    """
    def __init__(
        self,
        n_mlp_experts: int = 6,
        n_attn_heads: int = 4,
        conv_channels: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ConsensusExpertHead(
            256, 10,
            n_mlp_experts=n_mlp_experts,
            n_attn_heads=n_attn_heads,
            conv_channels=conv_channels,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetHierarchicalExpert(nn.Module):
    """
    Backbone-compatible model with HierarchicalMoEClassifierHead.
    Uses the same layers 0-9 backbone from ChampionNet with the new
    hierarchical MoE head at layer 10.
    """
    def __init__(
        self,
        n_domain_groups: int = 2,
        experts_per_group: int = 4,
        top_k_domains: int = 1,
        top_k_experts: int = 2,
        lora_rank: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(HierarchicalMoEClassifierHead(
            256, 10,
            n_domain_groups=n_domain_groups,
            experts_per_group=experts_per_group,
            top_k_domains=top_k_domains,
            top_k_experts=top_k_experts,
            lora_rank=lora_rank,
            dropout=dropout,
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetDeepExpert(nn.Module):
    """
    Backbone-compatible model with DeepExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(DeepExpertClassifierHead(256, 10, n_experts=n_experts, top_k=top_k, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetExpertChoice(nn.Module):
    """
    Backbone-compatible model with ExpertChoiceClassifierHead.
    """
    def __init__(self, n_experts: int = 8, capacity_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ExpertChoiceClassifierHead(256, 10, n_experts=n_experts, capacity_factor=capacity_factor, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

        return x


class ChampionNetSmarterExpert(nn.Module):
    """
    Backbone-compatible model with SmarterExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(SmarterExpertClassifierHead(256, 10, n_experts=n_experts, lora_rank=lora_rank, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class ChampionNetThoughtExpert(nn.Module):
    """
    Backbone-compatible model with ThoughtExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 8, reasoning_steps: int = 3, lora_rank: int = 32, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ThoughtExpertClassifierHead(
            256, 10, 
            n_experts=n_experts, 
            reasoning_steps=reasoning_steps, 
            lora_rank=lora_rank, 
            dropout=dropout
        ))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetUltraExpert(nn.Module):
    """
    Backbone-compatible model with GatedExpertClassifierHead.
    """
    def __init__(self, n_experts: int = 6, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(GatedExpertClassifierHead(256, 10, n_experts=n_experts, top_k=top_k, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetLarge(nn.Module):
    """
    Backbone-compatible larger model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHead
    """

    def __init__(self, expansion_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(ExpandedClassifierHead(256, 10, expansion_dim=expansion_dim, dropout=dropout))
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXL(nn.Module):
    """
    Backbone-compatible xlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXL
    """

    def __init__(self, expansion_dim: int = 768, extra_expansion_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXL(nn.Module):
    """
    Backbone-compatible xxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetXXXL(nn.Module):
    """
    Backbone-compatible xxxlarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadXXXL
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadXXXL(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetUltra(nn.Module):
    """
    Backbone-compatible ultralarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadUltra
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            ExpandedClassifierHeadUltra(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ChampionNetMega(nn.Module):
    """
    Backbone-compatible megalarge model:
    - keeps layers 0..9 and layer 11 from ChampionNet
    - replaces layer 10 with ExpandedClassifierHeadMega
    - adds GatedFFN after BitNetLinear for extra backbone capacity
    """

    def __init__(
        self,
        expansion_dim: int = 1024,
        extra_expansion_dim: int = 2048,
        third_expansion_dim: int = 3072,
        fourth_expansion_dim: int = 4096,
        fifth_expansion_dim: int = 6144,
        sixth_expansion_dim: int = 8192,
        dropout: float = 0.1,
    ):
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        # Insert GatedFFN for extra backbone capacity (between BitNet and classifier).
        layers.append(GatedFFN(d_model=256, d_inner=512))
        layers.append(
            ExpandedClassifierHeadMega(
                256,
                10,
                expansion_dim=expansion_dim,
                extra_expansion_dim=extra_expansion_dim,
                third_expansion_dim=third_expansion_dim,
                fourth_expansion_dim=fourth_expansion_dim,
                fifth_expansion_dim=fifth_expansion_dim,
                sixth_expansion_dim=sixth_expansion_dim,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_model(
    model_size: str = "base",
    expansion_dim: int = 512,
    dropout: float = 0.1,
    extra_expansion_dim: Optional[int] = None,
    third_expansion_dim: Optional[int] = None,
    fourth_expansion_dim: Optional[int] = None,
    fifth_expansion_dim: Optional[int] = None,
    sixth_expansion_dim: Optional[int] = None,
) -> nn.Module:
    """Construct a ChampionNet model variant by name.

    Factory function that dispatches to the appropriate backbone class
    based on model_size. Expansion dims default to sensible values when
    not provided.

    Args:
        model_size: One of 'base', 'large', 'xlarge', 'xxlarge', 'xxxlarge',
            'ultralarge', 'megalarge', 'ultra_expert', 'hierarchical_expert',
            'deep_expert', 'expert_choice', 'smarter_expert', 'thought_expert',
            'recursive_expert'.
        expansion_dim: Hidden size for the primary adapter branch.
        dropout: Dropout rate for adapter layers.
        extra_expansion_dim: Hidden size for the secondary adapter branch.
        third_expansion_dim: Hidden size for the third adapter branch.
        fourth_expansion_dim: Hidden size for the fourth adapter branch.
        fifth_expansion_dim: Hidden size for the fifth adapter branch.
        sixth_expansion_dim: Hidden size for the sixth adapter branch.

    Returns:
        An nn.Module ready for training or inference.

    Raises:
        ValueError: If model_size is not recognized.
    """
    if model_size == "base":
        return ChampionNet()
    if model_size == "large":
        return ChampionNetLarge(expansion_dim=expansion_dim, dropout=dropout)
    if model_size == "xlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(1024, expansion_dim * 2))
        return ChampionNetXL(expansion_dim=expansion_dim, extra_expansion_dim=extra_dim, dropout=dropout)
    if model_size == "xxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        return ChampionNetXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            dropout=dropout,
        )
    if model_size == "xxxlarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        return ChampionNetXXXL(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            dropout=dropout,
        )
    if model_size == "ultralarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        return ChampionNetUltra(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            dropout=dropout,
        )
    if model_size == "megalarge":
        extra_dim = int(extra_expansion_dim) if extra_expansion_dim is not None else int(max(2048, expansion_dim * 2))
        third_dim = int(third_expansion_dim) if third_expansion_dim is not None else int(max(3072, extra_dim + expansion_dim))
        fourth_dim = (
            int(fourth_expansion_dim)
            if fourth_expansion_dim is not None
            else int(max(4096, third_dim + expansion_dim))
        )
        fifth_dim = (
            int(fifth_expansion_dim)
            if fifth_expansion_dim is not None
            else int(max(6144, fourth_dim + expansion_dim))
        )
        sixth_dim = (
            int(sixth_expansion_dim)
            if sixth_expansion_dim is not None
            else int(max(8192, fifth_dim + expansion_dim))
        )
        return ChampionNetMega(
            expansion_dim=expansion_dim,
            extra_expansion_dim=extra_expansion_dim,
            third_expansion_dim=third_dim,
            fourth_expansion_dim=fourth_dim,
            fifth_expansion_dim=fifth_dim,
            sixth_expansion_dim=sixth_dim,
            dropout=dropout,
        )
    if model_size == "ultra_expert":
        return ChampionNetUltraExpert(dropout=dropout)
    if model_size == "hierarchical_expert":
        return ChampionNetHierarchicalExpert(dropout=dropout)
    if model_size == "deep_expert":
        return ChampionNetDeepExpert(dropout=dropout)
    if model_size == "expert_choice":
        return ChampionNetExpertChoice(dropout=dropout)
    if model_size == "smarter_expert":
        return ChampionNetSmarterExpert(dropout=dropout)
    if model_size == "thought_expert":
        return ChampionNetThoughtExpert(dropout=dropout)
    if model_size == "recursive_expert":
        return ChampionNetRecursiveExpert(dropout=dropout)
    if model_size == "reflexive_expert":
        return ChampionNetReflexiveExpert(dropout=dropout)
    if model_size == "metacognitive_expert":
        return ChampionNetMetaCognitiveExpert(dropout=dropout)
    if model_size == "tree_of_thought_expert":
        return ChampionNetTreeOfThoughtExpert(dropout=dropout)
    if model_size == "consensus_expert":
        return ChampionNetConsensusExpert(dropout=dropout)

    raise ValueError(
        f"Unknown model_size={model_size!r}. Use one of {SUPPORTED_MODEL_SIZES}."
    )



def load_weights_for_model(model: nn.Module, state_dict: dict, model_size: str) -> Tuple[List[str], List[str]]:
    """Load checkpoint weights into a model with backward compatibility.

    Handles warm-starting from smaller checkpoints by filtering out expected
    missing keys (new head parameters) and stripping incompatible head weights
    when upgrading across variant families.

    Args:
        model: The target model to load weights into.
        state_dict: Checkpoint state dictionary.
        model_size: The variant name of the target model.

    Returns:
        Tuple of (unexpected_missing_keys, unexpected_keys). Keys that are
        expected to be missing for warm-start scenarios are filtered out.

    Raises:
        RuntimeError: If weight dimensions are incompatible.
        ValueError: If model_size is not recognized.
    """
    if model_size == "base":
        incompatible = model.load_state_dict(state_dict, strict=False)
        return list(incompatible.missing_keys), list(incompatible.unexpected_keys)

    if model_size == "large":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load large checkpoint: likely head-dimension mismatch. "
                "Use matching --expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when loading base checkpoint into large model.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xlarge checkpoint: likely expansion/aux dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xlarge from base/large checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxlarge checkpoint: likely expansion/aux/third dimension mismatch. "
                "Use matching --expansion_dim/--extra_expansion_dim/--third_expansion_dim or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxlarge from base/large/xlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "xxxlarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load xxxlarge checkpoint: likely expansion/aux/third/fourth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting xxxlarge from base/large/xlarge/xxlarge checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "ultralarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load ultralarge checkpoint: likely expansion/aux/third/fourth/fifth dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting ultralarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
        }
        missing_filtered = [k for k in missing if k and k not in allowed_missing]
        unexpected_filtered = [k for k in unexpected if k]
        return missing_filtered, unexpected_filtered

    if model_size == "megalarge":
        try:
            incompatible = model.load_state_dict(state_dict, strict=False)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load megalarge checkpoint: likely expansion dimension mismatch. "
                "Use matching dims or auto-detect from checkpoint."
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # Expected when warm-starting megalarge from smaller checkpoints.
        allowed_missing = {
            "layers.10.adapter_up.weight",
            "layers.10.adapter_down.weight",
            "layers.10.alpha",
            "layers.10.adapter_up_b.weight",
            "layers.10.adapter_down_b.weight",
            "layers.10.router.weight",
            "layers.10.router.bias",
            "layers.10.beta",
            "layers.10.adapter_up_c.weight",
            "layers.10.adapter_down_c.weight",
            "layers.10.router2.weight",
            "layers.10.router2.bias",
            "layers.10.gamma",
            "layers.10.adapter_up_d.weight",
            "layers.10.adapter_down_d.weight",
            "layers.10.router3.weight",
            "layers.10.router3.bias",
            "layers.10.delta",
            "layers.10.adapter_up_e.weight",
            "layers.10.adapter_down_e.weight",
            "layers.10.router4.weight",
            "layers.10.router4.bias",
            "layers.10.epsilon",
            "layers.10.pre_norm.weight",
            "layers.10.pre_norm.bias",
            "layers.10.domain_router.weight",
            "layers.10.domain_router.bias",
            "layers.10.domain_experts.weight",
            "layers.10.calib_gate.weight",
            "layers.10.calib_gate.bias",
            "layers.10.zeta",
            "layers.10.theta",
            # Megalarge-only keys
            "layers.10.adapter_up_f.weight",
            "layers.10.adapter_down_f.weight",
            "layers.10.router5.weight",
            "layers.10.router5.bias",
            "layers.10.iota",
            "layers.10.cross_attn_fusion.q_proj.weight",
            "layers.10.cross_attn_fusion.k_proj.weight",
            "layers.10.cross_attn_fusion.v_proj.weight",
            "layers.10.cross_attn_fusion.out_proj.weight",
            "layers.10.cross_attn_fusion.fusion_weight.weight",
            "layers.10.cross_attn_fusion.fusion_weight.bias",
            "layers.10.kappa",
            "layers.10.reasoning_gate.0.weight",
            "layers.10.reasoning_gate.2.weight",
            "layers.10.reasoning_gate.2.bias",
            "layers.10.lambda_",
            # GatedFFN backbone layer (layers.10 in mega = GatedFFN)
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        }
        # Megalarge uses layers.10=GatedFFN, layers.11=Head, layers.12=norm
        # So head keys are at layers.11.* not layers.10.*
        mega_allowed = set()
        for k in allowed_missing:
            if k.startswith("layers.10."):
                mega_key = "layers.11." + k[len("layers.10."):]
                mega_allowed.add(mega_key)
        mega_allowed.update(allowed_missing)
        # Also allow GatedFFN keys at layers.10
        mega_allowed.update({
            "layers.10.norm.weight",
            "layers.10.up_proj.weight",
            "layers.10.down_proj.weight",
            "layers.10.layer_scale.gamma",
        })
        
        # Robust loading: if checkpoint is not megalarge, strip head keys to avoid mismatch
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "megalarge":
            filtered_sd = {k: v for k, v in state_dict.items() if not k.startswith("layers.10.") and not k.startswith("layers.11.")}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in mega_allowed]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "ultra_expert":
        allowed_missing = {
            "layers.10.experts_up.0.weight", "layers.10.experts_up.1.weight",
            "layers.10.experts_up.2.weight", "layers.10.experts_up.3.weight",
            "layers.10.experts_up.4.weight", "layers.10.experts_up.5.weight",
            "layers.10.experts_down.0.weight", "layers.10.experts_down.1.weight",
            "layers.10.experts_down.2.weight", "layers.10.experts_down.3.weight",
            "layers.10.experts_down.4.weight", "layers.10.experts_down.5.weight",
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "ultra_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "hierarchical_expert":
        # Allow all new hierarchical head parameters to be missing when warm-starting
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", f"{head_pref}shared_scale",
            f"{head_pref}domain_gate.weight", f"{head_pref}domain_noise_gate.weight",
            f"{head_pref}alpha", f"{head_pref}theta",
            f"{head_pref}calibration.weight", f"{head_pref}calibration.bias",
        }
        for i in range(2): # groups
            allowed_missing.add(f"{head_pref}expert_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_noise_gates.{i}.weight")
        for i in range(8): # experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "hierarchical_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "thought_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", 
            f"{head_pref}shared_scale", f"{head_pref}expert_bias",
            f"{head_pref}alpha", f"{head_pref}calibration.weight",
            f"{head_pref}calibration.bias", f"{head_pref}theta",
            f"{head_pref}cross_attn.q_proj.weight", f"{head_pref}cross_attn.k_proj.weight",
            f"{head_pref}cross_attn.v_proj.weight", f"{head_pref}cross_attn.out_proj.weight",
            f"{head_pref}cross_attn.fusion_weight.weight", f"{head_pref}cross_attn.fusion_weight.bias",
        }
        for i in range(8): # n_experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
        for i in range(3): # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}gates.{i}.weight")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "thought_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "recursive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias", 
            f"{head_pref}shared_scale", f"{head_pref}expert_bias",
            f"{head_pref}local_up.weight", f"{head_pref}local_down.weight",
            f"{head_pref}local_norm.weight", f"{head_pref}local_norm.bias", 
            f"{head_pref}local_scale",
            f"{head_pref}alpha", f"{head_pref}calibration.weight",
            f"{head_pref}calibration.bias", f"{head_pref}theta",
            f"{head_pref}cross_attn.q_proj.weight", f"{head_pref}cross_attn.k_proj.weight",
            f"{head_pref}cross_attn.v_proj.weight", f"{head_pref}cross_attn.out_proj.weight",
            f"{head_pref}cross_attn.fusion_weight.weight", f"{head_pref}cross_attn.fusion_weight.bias",
        }
        for i in range(8): # n_experts
            allowed_missing.add(f"{head_pref}experts_up.{i}.weight")
            allowed_missing.add(f"{head_pref}experts_down.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}expert_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}lora_down.{i}.weight")
            allowed_missing.add(f"{head_pref}lora_up.{i}.weight")
        for i in range(3): # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}exit_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}exit_gates.{i}.bias")
            for h in range(2): # n_heads
                allowed_missing.add(f"{head_pref}gates.{i * 2 + h}.weight")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "recursive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "reflexive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}initial_gate.weight",
            f"{head_pref}critique_gate.weight",
            f"{head_pref}alpha",
            f"{head_pref}beta",
        }
        for i in range(4): # default n_experts for reflexive
            allowed_missing.add(f"{head_pref}initial_up.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_down.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}initial_norms.{i}.bias")
            allowed_missing.add(f"{head_pref}critique_up.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_down.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "reflexive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "metacognitive_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}proposal_bias",
            f"{head_pref}alpha", f"{head_pref}beta",
        }
        for i in range(6):  # n_proposal_experts
            allowed_missing.add(f"{head_pref}proposal_up.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_down.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}proposal_norms.{i}.bias")
        for i in range(4):  # n_critique_experts
            allowed_missing.add(f"{head_pref}critique_up.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_down.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_norms.{i}.bias")
        for i in range(3):  # reasoning_steps
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.norm.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.up.bias")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.weight")
            allowed_missing.add(f"{head_pref}reasoning_cells.{i}.down.bias")
            allowed_missing.add(f"{head_pref}proposal_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}critique_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.weight")
            allowed_missing.add(f"{head_pref}halt_gates.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "metacognitive_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "tree_of_thought_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}shared_up.weight", f"{head_pref}shared_down.weight",
            f"{head_pref}shared_norm.weight", f"{head_pref}shared_norm.bias",
            f"{head_pref}shared_scale", f"{head_pref}alpha",
            f"{head_pref}final_up.weight", f"{head_pref}final_down.weight",
            f"{head_pref}final_norm.weight", f"{head_pref}final_norm.bias",
        }
        for i in range(6):  # n_action_experts
            allowed_missing.add(f"{head_pref}action_up.{i}.weight")
            allowed_missing.add(f"{head_pref}action_down.{i}.weight")
            allowed_missing.add(f"{head_pref}action_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}action_norms.{i}.bias")
        # Value net
        allowed_missing.add(f"{head_pref}value_net.0.weight")
        allowed_missing.add(f"{head_pref}value_net.0.bias")
        allowed_missing.add(f"{head_pref}value_net.3.weight")
        allowed_missing.add(f"{head_pref}value_net.3.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "tree_of_thought_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "consensus_expert":
        head_pref = "layers.10."
        allowed_missing = {
            f"{head_pref}alpha",
            f"{head_pref}mlp_gate.weight",
            f"{head_pref}attn_embed",
            f"{head_pref}attn_q.weight", f"{head_pref}attn_k.weight",
            f"{head_pref}attn_v.weight", f"{head_pref}attn_proj.weight",
            f"{head_pref}attn_norm.weight", f"{head_pref}attn_norm.bias",
            f"{head_pref}conv1.weight", f"{head_pref}conv1.bias",
            f"{head_pref}conv2.weight", f"{head_pref}conv2.bias",
            f"{head_pref}conv_proj.weight",
            f"{head_pref}conv_norm.weight", f"{head_pref}conv_norm.bias",
            f"{head_pref}consensus.0.weight", f"{head_pref}consensus.0.bias",
            f"{head_pref}consensus.3.weight", f"{head_pref}consensus.3.bias",
            f"{head_pref}consensus.5.weight", f"{head_pref}consensus.5.bias",
        }
        for i in range(6):  # n_mlp_experts
            allowed_missing.add(f"{head_pref}mlp_up.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_down.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_norms.{i}.weight")
            allowed_missing.add(f"{head_pref}mlp_norms.{i}.bias")

        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "consensus_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)

        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "deep_expert":
        allowed_missing = {
            "layers.10.shared_up.weight", "layers.10.shared_down.weight",
            "layers.10.shared_norm.weight", "layers.10.shared_norm.bias", 
            "layers.10.shared_scale", "layers.10.expert_bias",
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8): # n_experts max default
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "deep_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "expert_choice":
        allowed_missing = {
            "layers.10.gate.weight", "layers.10.noise_gate.weight",
            "layers.10.alpha", "layers.10.calibration.weight",
            "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8): # default max experts
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "expert_choice":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered

    if model_size == "smarter_expert":
        allowed_missing = {
            "layers.10.shared_up.weight", "layers.10.shared_down.weight",
            "layers.10.shared_norm.weight", "layers.10.shared_norm.bias", 
            "layers.10.shared_scale", "layers.10.expert_bias",
            "layers.10.gate.weight", "layers.10.alpha", 
            "layers.10.calibration.weight", "layers.10.calibration.bias", "layers.10.theta",
        }
        for i in range(8):
            allowed_missing.add(f"layers.10.experts_up.{i}.weight")
            allowed_missing.add(f"layers.10.experts_down.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.weight")
            allowed_missing.add(f"layers.10.expert_norms.{i}.bias")
            allowed_missing.add(f"layers.10.lora_down.{i}.weight")
            allowed_missing.add(f"layers.10.lora_up.{i}.weight")
            
        ckpt_size = detect_model_size_from_state_dict(state_dict)
        if ckpt_size != "smarter_expert":
            filtered_sd = {k: v for k, v in state_dict.items() if not (k.startswith("layers.10.") and k not in ["layers.10.weight", "layers.10.bias"])}
            incompatible = model.load_state_dict(filtered_sd, strict=False)
        else:
            incompatible = model.load_state_dict(state_dict, strict=False)
            
        missing_filtered = [k for k in incompatible.missing_keys if k and k not in allowed_missing]
        unexpected_filtered = [k for k in incompatible.unexpected_keys if k]
        return missing_filtered, unexpected_filtered


    raise ValueError(f"Unsupported model_size={model_size!r}")


def detect_model_size_from_state_dict(state_dict: dict) -> str:
    """Infer the model variant from checkpoint keys.

    Examines the state_dict for variant-specific parameter names to
    determine which model architecture produced the checkpoint.

    Args:
        state_dict: Checkpoint state dictionary.

    Returns:
        A model size string (e.g., 'base', 'ultra_expert', 'smarter_expert').
    """
    # Megalarge: has GatedFFN at layers.10 + head at layers.11
    if "layers.11.adapter_up_f.weight" in state_dict or "layers.11.router5.weight" in state_dict:
        return "megalarge"
    if "layers.10.adapter_up_e.weight" in state_dict or "layers.10.router4.weight" in state_dict:
        return "ultralarge"
    if "layers.10.adapter_up_d.weight" in state_dict or "layers.10.router3.weight" in state_dict:
        return "xxxlarge"
    if "layers.10.adapter_up_c.weight" in state_dict or "layers.10.router2.weight" in state_dict:
        return "xxlarge"
    if "layers.10.adapter_up_b.weight" in state_dict or "layers.10.router.weight" in state_dict:
        return "xlarge"
    if "layers.10.adapter_up.weight" in state_dict or "layers.10.adapter_down.weight" in state_dict:
        return "large"
    if "layers.10.domain_gate.weight" in state_dict or "layers.10.expert_gates.0.weight" in state_dict:
        return "hierarchical_expert"
    if "layers.10.shared_up.weight" in state_dict and "layers.10.gate.weight" in state_dict:
        if "layers.10.lora_up.0.weight" in state_dict:
            return "smarter_expert"
        return "deep_expert"

    if "layers.10.experts_up.0.weight" in state_dict and "layers.10.gate.weight" in state_dict and "layers.10.shared_scale" not in state_dict:
        # ultra_expert also has gate.weight, but we distinguish EC by checking if it doesn't have EC specific but also no base keys maybe?
        # A simpler way is to check the capacity_factor or something, but it's not in state dict.
        # Actually both ultra_expert and expert_choice have gate.weight and experts_up.0.weight.
        # We can detect expert_choice by checking for expert_norms which ultra_expert lacks.
        if "layers.10.expert_norms.0.weight" in state_dict:
            return "expert_choice"
        return "ultra_expert"
    
    if "layers.10.halt_gates.0.weight" in state_dict and "layers.10.proposal_gates.0.weight" in state_dict:
        return "metacognitive_expert"
    if "layers.10.value_net.0.weight" in state_dict and "layers.10.action_up.0.weight" in state_dict:
        return "tree_of_thought_expert"
    if "layers.10.mlp_gate.weight" in state_dict and "layers.10.consensus.0.weight" in state_dict:
        return "consensus_expert"
    if "layers.10.critique_gate.weight" in state_dict:
        return "reflexive_expert"
    if "layers.10.exit_gates.0.weight" in state_dict:
        return "recursive_expert"
    if "layers.10.reasoning_cells.0.up.weight" in state_dict:
        return "thought_expert"

    if "layers.10.gate.weight" in state_dict or "layers.10.experts_up.0.weight" in state_dict:
        return "ultra_expert"
    return "base"


def detect_large_head_expansion_dim(state_dict: dict, default: int = 512) -> int:
    """Detect the primary adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xlarge_aux_expansion_dim(state_dict: dict, default: int = 1024) -> int:
    """Detect the secondary (XL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_b.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxlarge_third_expansion_dim(state_dict: dict, default: int = 3072) -> int:
    """Detect the third (XXL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_c.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_xxxlarge_fourth_expansion_dim(state_dict: dict, default: int = 4096) -> int:
    """Detect the fourth (XXXL) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_d.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_ultralarge_fifth_expansion_dim(state_dict: dict, default: int = 6144) -> int:
    """Detect the fifth (Ultra) adapter expansion dim from a checkpoint."""
    key = "layers.10.adapter_up_e.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def detect_megalarge_sixth_expansion_dim(state_dict: dict, default: int = 8192) -> int:
    """Detect the sixth (Mega) adapter expansion dim from a checkpoint."""
    key = "layers.11.adapter_up_f.weight"
    weight = state_dict.get(key)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0])
    return int(default)


def list_trainable_parameter_names(model: nn.Module) -> Sequence[str]:
    """Return the names of all parameters with requires_grad=True."""
    return [name for name, p in model.named_parameters() if p.requires_grad]
