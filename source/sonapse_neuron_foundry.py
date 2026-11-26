"""Sonapse Synthetic Neuron Foundry & Pluggable MiMoMix Layer (Supermix v107).

Designs and implements six new additive, differentiable PyTorch synthetic neuron
architectures inspired by ``kai9987kai/SONAPSE-MORPHOGENESIS``, ``ODDITY-LAB``,
``Two-Soma-Neuron-Lab``, ``NeuroPulse-CableLab``, ``Supermix-Aether``, and
2025-2026 connectome/cellular neuroscience literature:

1. ``SolitonAxonNeuron`` (``soliton_axon``):
   Nonlinear dispersion-compensated electromechanical soliton pulse unit
   balancing steepening ``tanh(beta * drive)`` against finite-difference feature
   dispersion and an adaptive post-pulse refractory trace.
2. ``SpineApparatusNeuron`` (``spine_apparatus``):
   MICrONS EM connectome dual-regime dendritic spine switch: an intracellular
   calcium reservoir ``ca_t`` smoothly arbitrates between nascent exploratory
   Oja-Hebbian correlation and mature spine-apparatus error-modulated credit.
3. ``AstrocyticTripartiteNeuron`` (``astrocyte_tripartite``):
   Astrocyte-gated multi-timescale tripartite synapse (AGMP): pairs fast somatic
   integration with a slow astrocytic Ca2+ wave and gliotransmitter (D-serine /
   ATP) metaplasticity gate that locks stable states and unlocks plasticity
   under distribution shift.
4. ``EphapticFieldNeuron`` (``ephaptic_field``):
   Zero-lag extracellular local field potential (LFP) ephaptic coupling across
   neighboring channels ``(V_ext - V_soma)`` for fast phase synchronization
   without dense chemical synaptic wiring.
5. ``RetrogradeEndocannabinoidNeuron`` (``retrograde_messengers``):
   Post-to-pre retrograde nitric-oxide / endocannabinoid (DSI/DSE) signaling:
   postsynaptic depolarization bursts synthesize a bounded retrograde messenger
   that dynamically modulates presynaptic release probability.
6. ``FastWeightEngramNeuron`` (``fast_weight_engram``):
   Intracellular rank-1 fast-weight associative cytoskeletal trace paired with
   slow Palimpsest consolidation, bridging Universal Fast-Weight Memory down to
   single-cell organelle dynamics.

Also provides:
- ``HeterogeneousNeuronPopulation``: runs all 6 neuron families in parallel and
  fuses them via learned context-conditioned routing weights + metabolic energy
  accounting.
- ``SonapseNeuronFoundryLayer``: pluggable causal sequence layer for
  ``MiMoMixModel`` (``use_neuron_foundry=True``) with exact zero-init birth
  identity (``gate == 0``), strict causal state recurrence, rank-3
  ``(B, T', state_width)`` KV-cache state history, and bit-exact MTP
  self-speculative rollback parity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "NEURON_FOUNDRY_FAMILIES",
    "SolitonAxonNeuron",
    "SpineApparatusNeuron",
    "AstrocyticTripartiteNeuron",
    "EphapticFieldNeuron",
    "RetrogradeEndocannabinoidNeuron",
    "FastWeightEngramNeuron",
    "HeterogeneousNeuronPopulation",
    "SonapseNeuronFoundryLayer",
]

NEURON_FOUNDRY_FAMILIES: Tuple[str, ...] = (
    "soliton_axon",
    "spine_apparatus",
    "astrocyte_tripartite",
    "ephaptic_field",
    "retrograde_messengers",
    "fast_weight_engram",
)


class SolitonAxonNeuron(nn.Module):
    """Nonlinear dispersion-compensated soliton pulse neuron with refractory recovery.

    State ``s = [u, r]`` each ``(B, D)`` (total state width ``2 * D``):
        ``lap_u = roll(u, 1) - 2*u + roll(u, -1)``  (circular channel dispersion)
        ``pulse = tanh(steepen * (W_in x + u) + disp * lap_u) - refrac_gain * r``
        ``u_next = (1 - leak) * u + leak * pulse``
        ``r_next = (1 - refrac_decay) * r + refrac_decay * relu(u_next)``
    """

    state_mult: int = 2

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.in_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.steepen_logit = nn.Parameter(torch.zeros(self.dim))
        self.disp_strength = nn.Parameter(torch.full((self.dim,), 0.15))
        self.leak_logit = nn.Parameter(torch.zeros(self.dim))
        self.refrac_logit = nn.Parameter(torch.full((self.dim,), -1.2))
        self.refrac_gain = nn.Parameter(torch.full((self.dim,), 0.25))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        u = state[:, :d]
        r = state[:, d : 2 * d]
        drive = self.in_proj(x)
        steepen = 1.0 + F.softplus(self.steepen_logit).to(x.dtype)
        disp = 0.25 * torch.tanh(self.disp_strength).to(x.dtype)
        if d >= 3:
            lap_u = torch.roll(u, 1, dims=-1) - 2.0 * u + torch.roll(u, -1, dims=-1)
        else:
            lap_u = -u
        rg = 0.5 * torch.sigmoid(self.refrac_gain).to(x.dtype)
        pulse = torch.tanh(steepen * (drive + 0.5 * u) + disp * lap_u) - rg * r
        leak = torch.sigmoid(self.leak_logit).to(x.dtype)
        u_next = (1.0 - leak) * u + leak * pulse
        r_decay = torch.sigmoid(self.refrac_logit).to(x.dtype)
        r_next = (1.0 - r_decay) * r + r_decay * F.relu(u_next)
        return u_next, torch.cat([u_next, r_next], dim=-1)


class SpineApparatusNeuron(nn.Module):
    """MICrONS connectome dual-regime spine-apparatus calcium-switching neuron.

    State ``s = [soma, ca_reservoir]`` each ``(B, D)`` (width ``2 * D``):
        Nascent regime (low Ca2+): exploratory local Oja-Hebbian correlation.
        Mature spine-apparatus regime (high Ca2+): predictive error-driven credit.
        ``switch = sigmoid(sharpness * (ca - threshold))``
    """

    state_mult: int = 2

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.heb_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.pred_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.err_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.ca_decay_logit = nn.Parameter(torch.full((self.dim,), -1.5))
        self.ca_threshold = nn.Parameter(torch.full((self.dim,), 0.30))
        self.soma_leak_logit = nn.Parameter(torch.zeros(self.dim))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        soma = state[:, :d]
        ca = state[:, d : 2 * d]
        # Nascent exploratory Oja-normalized Hebbian current
        h_raw = self.heb_proj(x)
        nascent = torch.tanh(h_raw - 0.2 * (soma * soma) * h_raw)
        # Mature spine-apparatus predictive credit current
        pred = self.pred_proj(soma)
        mature = torch.tanh(self.err_proj(x - pred) + h_raw)
        ca_alpha = torch.sigmoid(self.ca_decay_logit).to(x.dtype)
        coincidence = (x.abs() * soma.abs()).clamp(0.0, 4.0)
        ca_next = (1.0 - ca_alpha) * ca + ca_alpha * coincidence
        switch = torch.sigmoid(4.0 * (ca_next - self.ca_threshold.to(x.dtype)))
        blended = (1.0 - switch) * nascent + switch * mature
        leak = torch.sigmoid(self.soma_leak_logit).to(x.dtype)
        soma_next = (1.0 - leak) * soma + leak * blended
        return soma_next, torch.cat([soma_next, ca_next], dim=-1)


class AstrocyticTripartiteNeuron(nn.Module):
    """Astrocyte-gated multi-timescale tripartite synapse neuron (AGMP).

    State ``s = [v_soma, astro_ca]`` each ``(B, D)`` (width ``2 * D``):
        Slow astrocytic Ca2+ integrates neuronal activity + surprise over long
        horizons; gliotransmitter gate suppresses drift during calm regimes and
        unlocks rapid somatic adaptation under novelty bursts.
    """

    state_mult: int = 2

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.syn_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.gliotrans_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.astro_leak_logit = nn.Parameter(torch.full((self.dim,), -2.4))
        self.base_leak_logit = nn.Parameter(torch.full((self.dim,), -0.8))
        self.target_activity = nn.Parameter(torch.full((self.dim,), 0.25))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        v_soma = state[:, :d]
        astro_ca = state[:, d : 2 * d]
        syn_in = self.syn_proj(x)
        surprise = (syn_in - v_soma).abs().clamp(0.0, 4.0)
        a_leak = torch.sigmoid(self.astro_leak_logit).to(x.dtype)
        astro_next = (1.0 - a_leak) * astro_ca + a_leak * (0.5 * v_soma.abs() + 0.5 * surprise)
        # Gliotransmitter (D-serine / ATP) modulates effective plasticity rate & gain
        glio_gate = torch.sigmoid(self.gliotrans_proj(astro_next - self.target_activity.to(x.dtype)))
        eff_leak = (torch.sigmoid(self.base_leak_logit).to(x.dtype) * (0.5 + glio_gate)).clamp(0.02, 0.98)
        v_next = (1.0 - eff_leak) * v_soma + eff_leak * torch.tanh(syn_in * (0.75 + 0.5 * glio_gate))
        return v_next, torch.cat([v_next, astro_next], dim=-1)


class EphapticFieldNeuron(nn.Module):
    """Extracellular local field potential (LFP) zero-lag ephaptic coupling neuron.

    State ``s = [v_soma]`` ``(B, D)`` (width ``D``):
        Computes population-mean & local-neighborhood extracellular potential
        ``v_ext`` and injects zero-lag transmembrane ephaptic current
        ``g_eph * (v_ext - v_soma)`` alongside chemical synaptic input.
    """

    state_mult: int = 1

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.chem_proj = nn.Linear(self.dim, self.dim, bias=True)
        self.field_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.ephaptic_conductance = nn.Parameter(torch.full((self.dim,), 0.25))
        self.leak_logit = nn.Parameter(torch.zeros(self.dim))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        chem = self.chem_proj(x)
        # Extracellular local field potential combines global LFP mean + projected local field
        lfp_global = chem.mean(dim=-1, keepdim=True)
        v_ext = 0.5 * lfp_global + 0.5 * torch.tanh(self.field_proj(state))
        g_eph = 0.5 * torch.sigmoid(self.ephaptic_conductance).to(x.dtype)
        transmembrane = chem + g_eph * (v_ext - state)
        leak = torch.sigmoid(self.leak_logit).to(x.dtype)
        v_next = (1.0 - leak) * state + leak * F.silu(transmembrane).clamp(-6.0, 6.0)
        return v_next, v_next


class RetrogradeEndocannabinoidNeuron(nn.Module):
    """Post-to-pre retrograde messenger (2-AG / NO DSI-DSE) synaptic release neuron.

    State ``s = [post_v, retro_msg]`` each ``(B, D)`` (width ``2 * D``):
        Postsynaptic depolarization synthesizes retrograde messenger ``retro_msg``
        that feeds back onto incoming presynaptic drive to modulate release
        probability ``p_rel in [0.35, 1.45]`` (preventing runaway saturation while
        amplifying coherent coincidences).
    """

    state_mult: int = 2

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.pre_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.post_bias = nn.Parameter(torch.zeros(self.dim))
        self.retro_sign = nn.Parameter(torch.full((self.dim,), -0.35))  # negative = DSI suppression default
        self.retro_decay_logit = nn.Parameter(torch.full((self.dim,), -1.1))
        self.leak_logit = nn.Parameter(torch.zeros(self.dim))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        post_v = state[:, :d]
        retro = state[:, d : 2 * d]
        p_rel = (1.0 + 0.45 * torch.tanh(self.retro_sign).to(x.dtype) * retro).clamp(0.35, 1.45)
        vesicle_drive = p_rel * self.pre_proj(x) + self.post_bias.to(x.dtype)
        leak = torch.sigmoid(self.leak_logit).to(x.dtype)
        post_next = (1.0 - leak) * post_v + leak * torch.tanh(vesicle_drive)
        r_alpha = torch.sigmoid(self.retro_decay_logit).to(x.dtype)
        retro_next = (1.0 - r_alpha) * retro + r_alpha * torch.tanh(post_next * post_next)
        return post_next, torch.cat([post_next, retro_next], dim=-1)


class FastWeightEngramNeuron(nn.Module):
    """Intracellular rank-1 fast-weight cytoskeletal engram + Palimpsest slow trace.

    State ``s = [fast_k, fast_v, slow_v]`` each ``(B, D)`` (width ``3 * D``):
        Maintains an intracellular associative key-value fast trace updated by
        local prediction error and gradually consolidated into a slow cytoskeletal
        Palimpsest trace.
    """

    state_mult: int = 3

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.k_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.q_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.write_logit = nn.Parameter(torch.full((self.dim,), -0.7))
        self.consolidate_logit = nn.Parameter(torch.full((self.dim,), -2.2))

    def step(self, x: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dim
        fast_k = state[:, :d]
        fast_v = state[:, d : 2 * d]
        slow_v = state[:, 2 * d : 3 * d]

        k_t = F.normalize(self.k_proj(x), p=2, dim=-1, eps=1e-6)
        v_t = torch.tanh(self.v_proj(x))
        q_t = F.normalize(self.q_proj(x), p=2, dim=-1, eps=1e-6)

        # Associative intracellular retrieval + delta surprise update
        match_prev = (fast_k * k_t).sum(dim=-1, keepdim=True)
        err = v_t - match_prev * fast_v
        w_gate = torch.sigmoid(self.write_logit).to(x.dtype)
        fast_k_next = F.normalize((1.0 - w_gate) * fast_k + w_gate * k_t, p=2, dim=-1, eps=1e-6)
        fast_v_next = ((1.0 - 0.5 * w_gate) * fast_v + w_gate * err).clamp(-4.0, 4.0)

        c_gate = (0.15 * torch.sigmoid(self.consolidate_logit).to(x.dtype))
        slow_v_next = (1.0 - c_gate) * slow_v + c_gate * fast_v_next

        read_match = (fast_k_next * q_t).sum(dim=-1, keepdim=True)
        out = torch.tanh(x + read_match * (0.65 * fast_v_next + 0.35 * slow_v_next))
        return out, torch.cat([fast_k_next, fast_v_next, slow_v_next], dim=-1)


class HeterogeneousNeuronPopulation(nn.Module):
    """Parallel bank of all 6 synthetic neuron families with learned context routing."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.soliton = SolitonAxonNeuron(self.dim)
        self.spine = SpineApparatusNeuron(self.dim)
        self.astrocyte = AstrocyticTripartiteNeuron(self.dim)
        self.ephaptic = EphapticFieldNeuron(self.dim)
        self.retrograde = RetrogradeEndocannabinoidNeuron(self.dim)
        self.engram_cell = FastWeightEngramNeuron(self.dim)
        self.cells: nn.ModuleList = nn.ModuleList(
            [
                self.soliton,
                self.spine,
                self.astrocyte,
                self.ephaptic,
                self.retrograde,
                self.engram_cell,
            ]
        )
        self.slices: List[Tuple[int, int]] = []
        cursor = 0
        for cell in self.cells:
            width = int(cell.state_mult) * self.dim
            self.slices.append((cursor, cursor + width))
            cursor += width
        self.total_state_dim: int = cursor
        self.router = nn.Linear(self.dim, len(self.cells), bias=True)
        # Metabolic relative cost weights mirroring SONAPSE-MORPHOGENESIS energy accounting
        self.register_buffer(
            "metabolic_weights",
            torch.tensor([1.15, 1.25, 1.10, 0.85, 1.05, 1.35], dtype=torch.float32),
            persistent=False,
        )

    def step(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = int(x.shape[0])
        if state is None or state.numel() == 0:
            state = x.new_zeros((bsz, self.total_state_dim))
        else:
            state = state.to(dtype=x.dtype, device=x.device)

        cell_outs: List[torch.Tensor] = []
        next_states: List[torch.Tensor] = []
        for idx, cell in enumerate(self.cells):
            s0, s1 = self.slices[idx]
            y_c, st_c = cell.step(x, state[:, s0:s1])
            cell_outs.append(y_c)
            next_states.append(st_c)

        weights = F.softmax(self.router(x).float(), dim=-1).to(x.dtype)  # (B, 6)
        stacked = torch.stack(cell_outs, dim=1)  # (B, 6, D)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return fused, torch.cat(next_states, dim=-1), weights


class SonapseNeuronFoundryLayer(nn.Module):
    """Causal sequence wrapper grafting ``HeterogeneousNeuronPopulation`` into ``MiMoMixModel``.

    Guarantees:
    - Exact zero-init birth identity (``gate == 0`` -> ``torch.equal`` with base model).
    - Strict left-to-right token causality.
    - Bit-exact incremental KV-cache & MTP speculative rollback parity via a
      rank-3 ``(B, T', total_state_dim)`` state tail entry in ``past_key_values``.
    """

    def __init__(
        self,
        hidden_size: int,
        foundry_dim: int = 32,
        after_layer: int = 2,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.foundry_dim = max(4, int(foundry_dim))
        self.after_layer = int(after_layer)
        self.eps = float(rms_norm_eps)
        self.norm_weight = nn.Parameter(torch.ones(self.hidden_size))
        self.in_proj = nn.Linear(self.hidden_size, self.foundry_dim, bias=False)
        self.population = HeterogeneousNeuronPopulation(self.foundry_dim)
        self.out_proj = nn.Linear(self.foundry_dim, self.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(self.hidden_size))

        self.register_buffer(
            "last_family_weights",
            torch.full((len(NEURON_FOUNDRY_FAMILIES),), 1.0 / float(len(NEURON_FOUNDRY_FAMILIES))),
            persistent=False,
        )
        self.register_buffer("last_metabolic_cost", torch.zeros(()), persistent=False)
        self.register_buffer("last_state_rms", torch.zeros(()), persistent=False)
        self.reset_special_parameters()

    @property
    def total_state_dim(self) -> int:
        return int(self.population.total_state_dim)

    def reset_special_parameters(self) -> None:
        with torch.no_grad():
            self.gate.zero_()
            self.norm_weight.fill_(1.0)
            nn.init.zeros_(self.population.router.bias)

    def _rms_norm(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float32 and self.norm_weight.dtype == torch.float32 and hasattr(F, "rms_norm"):
            return F.rms_norm(x, (x.shape[-1],), self.norm_weight, self.eps)
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (self.norm_weight * (xf * torch.rsqrt(var + self.eps))).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_state_flat: Optional[torch.Tensor] = None,
        keep_states: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        normed = self._rms_norm(hidden_states)
        x_seq = self.in_proj(normed)

        if prev_state_flat is not None and prev_state_flat.numel() > 0:
            state = prev_state_flat.to(dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            state = hidden_states.new_zeros((bsz, self.total_state_dim))

        if seq_len == 1:
            y_0, state, w_0 = self.population.step(x_seq[:, 0], state)
            y_seq = y_0.unsqueeze(1)
            hist = (
                state.unsqueeze(1)
                if (keep_states is None or int(keep_states) >= 1)
                else hidden_states.new_zeros((bsz, 0, self.total_state_dim))
            )
            mean_w = w_0.detach().float().mean(dim=0)
        else:
            hist_start = max(0, seq_len - int(keep_states)) if keep_states is not None else 0
            x_steps = x_seq.unbind(1)
            outs: List[torch.Tensor] = []
            st_hist: List[torch.Tensor] = []
            w_acc = hidden_states.new_zeros((len(NEURON_FOUNDRY_FAMILIES),), dtype=torch.float32)
            for t in range(seq_len):
                y_t, state, w_t = self.population.step(x_steps[t], state)
                outs.append(y_t)
                w_acc = w_acc + w_t.detach().float().mean(dim=0)
                if t >= hist_start:
                    st_hist.append(state)
            y_seq = torch.stack(outs, dim=1)
            hist = (
                st_hist[0].unsqueeze(1)
                if len(st_hist) == 1
                else (
                    torch.stack(st_hist, dim=1)
                    if st_hist
                    else hidden_states.new_zeros((bsz, 0, self.total_state_dim))
                )
            )
            mean_w = w_acc / float(max(1, seq_len))

        with torch.no_grad():
            self.last_family_weights.copy_(mean_w)
            met = (mean_w * self.population.metabolic_weights.to(mean_w.device)).sum()
            self.last_metabolic_cost = met * float(self.foundry_dim)
            self.last_state_rms = state.detach().float().pow(2).mean().sqrt()

        out = hidden_states + self.gate.to(dtype=hidden_states.dtype) * self.out_proj(y_seq)
        return out, hist

    def telemetry(self) -> Dict[str, object]:
        weights_list = [round(float(v), 5) for v in self.last_family_weights.detach().cpu().tolist()]
        dominant_idx = int(torch.argmax(self.last_family_weights.detach()).item())
        return {
            "enabled": True,
            "foundry_dim": int(self.foundry_dim),
            "after_layer": int(self.after_layer),
            "families": list(NEURON_FOUNDRY_FAMILIES),
            "family_routing_weights": dict(zip(NEURON_FOUNDRY_FAMILIES, weights_list)),
            "dominant_family": NEURON_FOUNDRY_FAMILIES[dominant_idx],
            "metabolic_cost_estimate": round(float(self.last_metabolic_cost.detach()), 4),
            "last_state_rms": round(float(self.last_state_rms.detach()), 5),
            "gate_mean_abs": float(self.gate.detach().abs().mean()),
        }
