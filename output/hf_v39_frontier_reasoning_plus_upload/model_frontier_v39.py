"""v39 larger frontier architecture with verifier-guided reasoning.

This variant keeps the v35 frontier-collective keys so it can warm-start from the
current best frontier checkpoints, then adds an internal verifier/correction path
inspired by recent work on tool-integrated self-verification and concise reasoning.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from run import ChampionNet
from model_variants import MixtureOfRouters, ReasoningCell, TemporalSSMAdapter


def _renorm_sparse_topk(probs: torch.Tensor, active_k: torch.Tensor, max_k: int) -> torch.Tensor:
    max_k = max(1, int(max_k))
    top_vals, top_idx = torch.topk(probs, k=min(max_k, probs.shape[-1]), dim=-1)
    rank_idx = torch.arange(top_vals.shape[-1], device=probs.device).unsqueeze(0)
    active_mask = rank_idx < active_k.unsqueeze(-1)
    sparse = torch.zeros_like(probs)
    sparse.scatter_(1, top_idx, top_vals * active_mask.float())
    return sparse / sparse.sum(dim=-1, keepdim=True).clamp_min(1e-6)


class FrontierVerifierExpertHead(nn.Module):
    """Larger frontier head with an internal verifier and correction branch."""

    def __init__(
        self,
        in_dim: int = 256,
        out_dim: int = 10,
        n_experts: int = 12,
        n_mem_slots: int = 32,
        reasoning_steps: int = 6,
        n_sub_routers: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_experts = n_experts
        self.n_mem_slots = n_mem_slots
        self.reasoning_steps = reasoning_steps

        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.shared_up = nn.Linear(in_dim, 3072, bias=False)
        self.shared_down = nn.Linear(3072, out_dim, bias=False)
        self.shared_norm = nn.LayerNorm(out_dim)
        self.shared_scale = nn.Parameter(torch.tensor(1.0))

        self.memory_keys = nn.Parameter(torch.randn(n_mem_slots, in_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(n_mem_slots, in_dim) * 0.02)
        self.memory_query = nn.Linear(in_dim, in_dim, bias=False)
        self.memory_out = nn.Linear(in_dim, out_dim, bias=False)
        self.memory_write_gate = nn.Linear(in_dim * 2, n_mem_slots, bias=True)
        self.memory_write_value = nn.Linear(in_dim, in_dim, bias=True)
        self.mem_scale = in_dim ** -0.5

        self.depth_query = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_key = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_value = nn.Linear(in_dim, in_dim, bias=False)
        self.depth_out = nn.Linear(in_dim, out_dim, bias=False)
        self.depth_scale = in_dim ** -0.5

        self.reasoning_cells = nn.ModuleList(
            [ReasoningCell(in_dim, inner_dim=896, dropout=dropout) for _ in range(reasoning_steps)]
        )
        self.ssm_adapter = TemporalSSMAdapter(in_dim, d_state=48)

        self.mor_router = MixtureOfRouters(in_dim, n_experts, n_sub_routers=n_sub_routers)
        expert_dims = [2048, 2560, 3072, 3584, 4096, 2304, 2816, 3328, 3840, 4608, 3072, 4096]
        expert_acts = [F.silu, F.gelu, F.mish, F.relu, F.selu, torch.tanh, F.softplus, F.elu]
        self.experts_up = nn.ModuleList(
            [nn.Linear(in_dim, expert_dims[i % len(expert_dims)], bias=False) for i in range(n_experts)]
        )
        self.experts_down = nn.ModuleList(
            [nn.Linear(expert_dims[i % len(expert_dims)], out_dim, bias=False) for i in range(n_experts)]
        )
        self.expert_norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(n_experts)])
        self.expert_acts = [expert_acts[i % len(expert_acts)] for i in range(n_experts)]
        self.router_budget = nn.Linear(in_dim, 5, bias=True)
        self.register_buffer("expert_bias", torch.zeros(n_experts), persistent=False)
        self.halt_gates = nn.ModuleList([nn.Linear(in_dim, 1, bias=True) for _ in range(reasoning_steps)])

        self.shared_aux_up = nn.Linear(in_dim, 4096, bias=False)
        self.shared_aux_down = nn.Linear(4096, out_dim, bias=False)
        self.shared_aux_norm = nn.LayerNorm(out_dim)
        self.shared_aux_gate = nn.Linear(in_dim, 2, bias=True)

        self.reflect_cells = nn.ModuleList([ReasoningCell(in_dim, inner_dim=1280, dropout=dropout) for _ in range(3)])
        self.reflect_out = nn.Linear(in_dim, out_dim, bias=False)

        self.collective_q = nn.Linear(in_dim, in_dim, bias=False)
        self.collective_k = nn.Linear(in_dim, in_dim, bias=False)
        self.collective_v = nn.Linear(in_dim, in_dim, bias=False)
        self.collective_out = nn.Linear(in_dim, out_dim, bias=False)
        self.collective_scale = in_dim ** -0.5

        self.revisit_gate = nn.Linear(in_dim * 3, 1, bias=True)
        self.verifier_cells = nn.ModuleList([ReasoningCell(in_dim, inner_dim=1536, dropout=dropout) for _ in range(2)])
        self.verifier_q = nn.Linear(in_dim, in_dim, bias=False)
        self.verifier_k = nn.Linear(in_dim, in_dim, bias=False)
        self.verifier_v = nn.Linear(in_dim, in_dim, bias=False)
        self.verifier_out = nn.Linear(in_dim, out_dim, bias=False)
        self.verifier_gate = nn.Linear(in_dim * 2, 2, bias=True)
        self.correction_out = nn.Linear(in_dim, out_dim, bias=False)
        self.verifier_scale = in_dim ** -0.5

        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.gamma = nn.Parameter(torch.tensor(0.0))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.epsilon = nn.Parameter(torch.tensor(0.0))
        self.zeta = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(dropout)
        self._aux_loss = torch.tensor(0.0)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in > 0:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        nn.init.kaiming_uniform_(self.shared_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_down.weight)
        nn.init.normal_(self.memory_keys, std=0.02)
        nn.init.normal_(self.memory_values, std=0.02)
        nn.init.normal_(self.memory_query.weight, std=0.01)
        nn.init.zeros_(self.memory_out.weight)
        nn.init.zeros_(self.memory_write_gate.weight)
        nn.init.zeros_(self.memory_write_gate.bias)
        nn.init.normal_(self.memory_write_value.weight, std=0.01)
        nn.init.zeros_(self.memory_write_value.bias)
        nn.init.normal_(self.depth_query.weight, std=0.01)
        nn.init.normal_(self.depth_key.weight, std=0.01)
        nn.init.normal_(self.depth_value.weight, std=0.01)
        nn.init.zeros_(self.depth_out.weight)
        for idx in range(self.n_experts):
            nn.init.kaiming_uniform_(self.experts_up[idx].weight, a=math.sqrt(5))
            nn.init.normal_(self.experts_down[idx].weight, std=0.01)
        nn.init.zeros_(self.router_budget.weight)
        nn.init.zeros_(self.router_budget.bias)
        for gate in self.halt_gates:
            nn.init.normal_(gate.weight, std=0.01)
            nn.init.constant_(gate.bias, -2.0)

        nn.init.kaiming_uniform_(self.shared_aux_up.weight, a=math.sqrt(5))
        nn.init.zeros_(self.shared_aux_down.weight)
        nn.init.zeros_(self.shared_aux_gate.weight)
        nn.init.zeros_(self.shared_aux_gate.bias)
        nn.init.zeros_(self.reflect_out.weight)
        nn.init.normal_(self.collective_q.weight, std=0.01)
        nn.init.normal_(self.collective_k.weight, std=0.01)
        nn.init.normal_(self.collective_v.weight, std=0.01)
        nn.init.zeros_(self.collective_out.weight)

        nn.init.zeros_(self.revisit_gate.weight)
        nn.init.zeros_(self.revisit_gate.bias)
        nn.init.normal_(self.verifier_q.weight, std=0.01)
        nn.init.normal_(self.verifier_k.weight, std=0.01)
        nn.init.normal_(self.verifier_v.weight, std=0.01)
        nn.init.zeros_(self.verifier_out.weight)
        nn.init.zeros_(self.verifier_gate.weight)
        nn.init.zeros_(self.verifier_gate.bias)
        nn.init.zeros_(self.correction_out.weight)

    def _depth_context(self, current: torch.Tensor, depth_bank: List[torch.Tensor]) -> torch.Tensor:
        if not depth_bank:
            return torch.zeros_like(current)
        depth_stack = torch.stack(depth_bank, dim=1)
        q = self.depth_query(current).unsqueeze(1)
        k = self.depth_key(depth_stack)
        v = self.depth_value(depth_stack)
        scores = (q * k).sum(dim=-1) * self.depth_scale
        attn = torch.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * v).sum(dim=1)

    def _collective_context(self, current: torch.Tensor, state_bank: List[torch.Tensor]) -> torch.Tensor:
        if not state_bank:
            return torch.zeros_like(current)
        state_stack = torch.stack(state_bank, dim=1)
        q = self.collective_q(current).unsqueeze(1)
        k = self.collective_k(state_stack)
        v = self.collective_v(state_stack)
        scores = (q * k).sum(dim=-1) * self.collective_scale
        attn = torch.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * v).sum(dim=1)

    def _verifier_context(self, current: torch.Tensor, state_bank: List[torch.Tensor]) -> torch.Tensor:
        if not state_bank:
            return torch.zeros_like(current)
        state_stack = torch.stack(state_bank, dim=1)
        q = self.verifier_q(current).unsqueeze(1)
        k = self.verifier_k(state_stack)
        v = self.verifier_v(state_stack)
        scores = (q * k).sum(dim=-1) * self.verifier_scale
        attn = torch.softmax(scores, dim=-1)
        return (attn.unsqueeze(-1) * v).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape_prefix = x.shape[:-1]
        x_flat = x.reshape(-1, self.in_dim)
        n_items = x_flat.shape[0]

        base_logits = F.linear(x_flat, self.weight, self.bias)
        shared_out = self.shared_norm(self.shared_down(self.dropout(F.silu(self.shared_up(x_flat)))))
        shared_aux_out = self.shared_aux_norm(self.shared_aux_down(self.dropout(F.gelu(self.shared_aux_up(x_flat)))))

        memory_values = self.memory_values.unsqueeze(0).expand(n_items, -1, -1).clone()
        current = x_flat
        total_logits = torch.zeros_like(base_logits)
        cumulative_halt = torch.zeros(n_items, 1, device=x.device, dtype=x.dtype)
        expert_usage: List[torch.Tensor] = []
        halt_usage: List[torch.Tensor] = []
        memory_entropy_vals: List[torch.Tensor] = []
        revisit_usage: List[torch.Tensor] = []
        depth_bank: List[torch.Tensor] = []

        max_topk = min(5, self.n_experts)
        steps_run = 0
        mem_ctx = torch.zeros_like(current)
        depth_ctx = torch.zeros_like(current)

        for step in range(self.reasoning_steps):
            steps_run += 1
            mem_query = self.memory_query(current)
            mem_scores = torch.matmul(mem_query, self.memory_keys.t()) * self.mem_scale
            mem_attn = torch.softmax(mem_scores, dim=-1)
            mem_ctx = torch.bmm(mem_attn.unsqueeze(1), memory_values).squeeze(1)
            depth_ctx = self._depth_context(current, depth_bank)
            ssm_corr, _ = self.ssm_adapter(current)

            revisit = torch.sigmoid(self.revisit_gate(torch.cat([current, mem_ctx, depth_ctx], dim=-1)))
            enriched = current + revisit * (0.36 * mem_ctx + 0.24 * depth_ctx + 0.18 * ssm_corr)
            current = self.reasoning_cells[step](enriched)
            depth_bank.append(current)

            gate_logits = self.mor_router(current) + self.expert_bias
            gate_probs = torch.softmax(gate_logits, dim=-1)
            budget_choice = 1 + torch.argmax(torch.softmax(self.router_budget(current), dim=-1), dim=-1)
            sparse_gate = _renorm_sparse_topk(gate_probs, active_k=budget_choice, max_k=max_topk)

            expert_outs = []
            for idx in range(self.n_experts):
                h = self.dropout(self.expert_acts[idx](self.experts_up[idx](current)))
                expert_outs.append(self.expert_norms[idx](self.experts_down[idx](h)))
            expert_stack = torch.stack(expert_outs, dim=1)
            expert_logits = (expert_stack * sparse_gate.unsqueeze(-1)).sum(dim=1)

            mem_logits = self.memory_out(mem_ctx)
            depth_logits = self.depth_out(depth_ctx)
            step_logits = expert_logits + 0.24 * mem_logits + 0.16 * depth_logits

            certainty = sparse_gate.max(dim=-1, keepdim=True).values
            surprise = 1.0 - certainty
            write_gate = torch.softmax(self.memory_write_gate(torch.cat([current, mem_ctx], dim=-1)), dim=-1)
            write_value = self.memory_write_value(current).unsqueeze(1)
            memory_values = memory_values + (
                surprise.unsqueeze(-1) * write_gate.unsqueeze(-1) * (write_value - memory_values)
            )

            halt_prob = torch.sigmoid(self.halt_gates[step](current + 0.20 * mem_ctx + 0.10 * depth_ctx))
            remaining = 1.0 - cumulative_halt
            total_logits = total_logits + remaining * step_logits
            cumulative_halt = cumulative_halt + remaining * halt_prob

            expert_usage.append(sparse_gate.mean(dim=0))
            halt_usage.append(halt_prob.mean())
            revisit_usage.append(revisit.mean())
            entropy = -(mem_attn * torch.log(mem_attn.clamp_min(1e-8))).sum(dim=-1)
            memory_entropy_vals.append(entropy.mean() / math.log(float(self.n_mem_slots)))

            if self.training:
                with torch.no_grad():
                    target = 1.0 / float(self.n_experts)
                    self.expert_bias.add_(0.01 * (target - sparse_gate.mean(dim=0)))
            elif cumulative_halt.mean() > 0.94:
                break

        reflect_state = current
        for cell in self.reflect_cells:
            reflect_state = cell(reflect_state + 0.18 * mem_ctx + 0.12 * depth_ctx)
        reflect_logits = self.reflect_out(reflect_state)

        collective_bank = [x_flat] + depth_bank + [reflect_state]
        collective_ctx = self._collective_context(current, collective_bank)
        collective_logits = self.collective_out(collective_ctx)

        verifier_state = current
        verifier_ctx = torch.zeros_like(current)
        verifier_bank = [x_flat] + depth_bank + [reflect_state, current]
        for cell in self.verifier_cells:
            verifier_ctx = self._verifier_context(verifier_state, verifier_bank)
            verifier_state = cell(verifier_state + 0.20 * verifier_ctx + 0.10 * reflect_state + 0.08 * mem_ctx)
        verifier_logits = self.verifier_out(verifier_ctx)
        correction_logits = self.correction_out(verifier_state)
        verify_mix = torch.softmax(self.verifier_gate(torch.cat([current, verifier_state], dim=-1)), dim=-1)
        verifier_mix = verify_mix[:, :1] * verifier_logits + verify_mix[:, 1:] * correction_logits

        mix_weights = torch.softmax(self.shared_aux_gate(current), dim=-1)
        shared_mix = mix_weights[:, :1] * shared_out + mix_weights[:, 1:] * shared_aux_out

        total_logits = total_logits / float(max(1, steps_run))
        correction_support = verify_mix[:, :1] * reflect_logits + verify_mix[:, 1:] * collective_logits
        output = (
            base_logits
            + self.shared_scale * shared_mix
            + (self.alpha + 1e-4) * total_logits
            + self.beta * self.memory_out(current)
            + self.gamma * reflect_logits
            + self.delta * collective_logits
            + self.epsilon * verifier_mix
            + self.zeta * correction_support
        )

        if expert_usage:
            mean_usage = torch.stack(expert_usage, dim=0).mean(dim=0)
            uniform = torch.full_like(mean_usage, 1.0 / float(self.n_experts))
            balance_loss = F.mse_loss(mean_usage, uniform)
            halt_target = torch.tensor(0.90, device=x.device, dtype=x.dtype)
            halt_loss = (torch.stack(halt_usage).mean() - halt_target).pow(2)
            mem_entropy = torch.stack(memory_entropy_vals).mean()
            mem_loss = (mem_entropy - 0.60).pow(2)
            mix_entropy = -(mix_weights * torch.log(mix_weights.clamp_min(1e-8))).sum(dim=-1).mean() / math.log(2.0)
            mix_loss = (mix_entropy - 0.58).pow(2)
            verify_entropy = -(verify_mix * torch.log(verify_mix.clamp_min(1e-8))).sum(dim=-1).mean() / math.log(2.0)
            verify_loss = (verify_entropy - 0.55).pow(2)
            revisit_loss = (torch.stack(revisit_usage).mean() - 0.45).pow(2)
            self._aux_loss = (
                balance_loss
                + 0.30 * halt_loss
                + 0.20 * mem_loss
                + 0.15 * mix_loss
                + 0.12 * verify_loss
                + 0.08 * revisit_loss
            )
        else:
            self._aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return output.view(*shape_prefix, self.out_dim)


class ChampionNetFrontierVerifierExpert(nn.Module):
    """Backbone wrapper for the larger v39 verifier-guided frontier head."""

    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        base = ChampionNet()
        layers = [base.layers[i] for i in range(10)]
        layers.append(
            FrontierVerifierExpertHead(
                256,
                10,
                n_experts=12,
                n_mem_slots=32,
                reasoning_steps=6,
                n_sub_routers=5,
                dropout=dropout,
            )
        )
        layers.append(base.layers[11])
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
