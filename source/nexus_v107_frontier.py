"""Supermix v107 Universal Fast-Weight, Connectome Plasticity & Morphogenesis Engines.

Five pure-stdlib, deterministic, zero-``eval``, fail-closed analytical engines
with canonical SHA-256 audit receipts (``supermix-v107-receipt-v1``):

1. ``UniversalFastWeightProfiler``:
   Analytical capacity, parameter-sharing, signal-to-interference ratio (SIR),
   and rank-3 KV-cache footprint profiler for Universal Fast-Weight Memory
   (uTTT-MoE / uTTT-Dense, Cai et al. Oct 2026, ``arXiv:2610.05484``).
2. ``ConnectomePlasticityOrchestrator``:
   Three-Factor Behavioral Timescale Synaptic Plasticity (BTSP) + MICrONS EM
   spine-apparatus dual-regime switching + astrocytic Ca2+ metaplasticity
   simulator with strict Dale's law sign verification.
3. ``SyntheticNeuronFoundryAssay``:
   Dynamical responsiveness, memory-horizon, phase-coherence, and metabolic ATP
   budget assay across all six ``SONAPSE-MORPHOGENESIS`` synthetic neuron
   families (Soliton, Spine-Apparatus, Astrocyte, Ephaptic, Retrograde, Engram).
4. ``AntipodeFailureDistiller``:
   Inspired by ``kai9987kai/SONAPSE-MORPHOGENESIS`` (``antipode/``): clusters
   hard-negative failure trajectories and synthesizes anti-collapse Boltzmann
   curriculum weights for targeted corrective refinement.
5. ``SymbiosisContinualRetentionAuditor``:
   Inspired by ``kai9987kai/SONAPSE-MORPHOGENESIS`` (``symbiosis/``): computes
   exact Fisher-weighted EWC quadratic drift, closed-form proximal synaptic
   consolidation, backward transfer, and catastrophic-forgetting risk grades.

Honesty & Rule 9 contract:
- Every returned payload sets ``"answer_authority": False``.
- All inputs are strictly validated (fail closed via ``ValueError``).
- Zero network I/O, zero ``eval``/``exec``, zero automatic checkpoint promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "FRONTIER_V107_VERSION",
    "RECEIPT_SCHEMA_V107",
    "UniversalFastWeightProfiler",
    "ConnectomePlasticityOrchestrator",
    "SyntheticNeuronFoundryAssay",
    "AntipodeFailureDistiller",
    "SymbiosisContinualRetentionAuditor",
]

FRONTIER_V107_VERSION = "v107_universal_morphogenesis"
RECEIPT_SCHEMA_V107 = "supermix-v107-receipt-v1"


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite(value: Any, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _require_int(
    value: Any,
    name: str,
    *,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return int(value)


def _require_float_seq(
    values: Any, name: str, *, min_len: int = 1, max_len: int = 100_000
) -> List[float]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple of numbers")
    if len(values) < min_len:
        raise ValueError(f"{name} must have at least {min_len} element(s), got {len(values)}")
    if len(values) > max_len:
        raise ValueError(f"{name} exceeds maximum length {max_len}")
    return [_require_finite(v, f"{name}[{idx}]") for idx, v in enumerate(values)]


def _build_receipt(engine_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    canonical = json.dumps(
        {
            "schema": RECEIPT_SCHEMA_V107,
            "version": FRONTIER_V107_VERSION,
            "engine": engine_name,
            "body": body,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema": RECEIPT_SCHEMA_V107,
        "version": FRONTIER_V107_VERSION,
        "engine": engine_name,
        "receipt_sha256": digest,
        "answer_authority": False,
    }


# ---------------------------------------------------------------------------
# 1. Universal Fast-Weight Profiler (uTTT-MoE / uTTT-Dense)
# ---------------------------------------------------------------------------


class UniversalFastWeightProfiler:
    """Analytical profiler for Universal Fast-Weight Memory (uTTT, arXiv:2610.05484)."""

    def profile(
        self,
        *,
        hidden_size: int = 128,
        n_layers: int = 6,
        n_kv_heads: int = 2,
        head_dim: int = 32,
        fast_dim: int = 24,
        n_banks: int = 4,
        top_k: int = 2,
        seq_len: int = 512,
        mode: str = "moe",
        surprise_trace: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        h = _require_int(hidden_size, "hidden_size", min_value=8, max_value=65536)
        layers = _require_int(n_layers, "n_layers", min_value=1, max_value=512)
        kv_h = _require_int(n_kv_heads, "n_kv_heads", min_value=1, max_value=256)
        hd = _require_int(head_dim, "head_dim", min_value=2, max_value=1024)
        d = _require_int(fast_dim, "fast_dim", min_value=2, max_value=1024)
        banks = _require_int(n_banks, "n_banks", min_value=1, max_value=256)
        t_len = _require_int(seq_len, "seq_len", min_value=1, max_value=1_000_000)
        mode_s = str(mode or "").strip().lower()
        if mode_s not in {"moe", "dense"}:
            raise ValueError(f"mode must be 'moe' or 'dense', got {mode!r}")
        k_act = _require_int(top_k, "top_k", min_value=1, max_value=banks) if mode_s == "moe" else banks

        # Parameter count of shared Universal Programmer vs unshared per-layer programmers
        core_proj_params = 4 * (h * d) + 2 * (h + 1) + banks * (h + 1)
        shared_params = core_proj_params + banks * d * d + layers * h + layers * h + h + h
        unshared_equiv_params = (layers + 1) * (core_proj_params + banks * d * d + h) + h
        sharing_ratio = unshared_equiv_params / float(max(1, shared_params))

        # Constant-size rank-3 temporal state vs linear-in-T full-attention KV cache (FP32 bytes)
        ufwm_state_floats = banks * d * d
        ufwm_state_bytes = 4 * ufwm_state_floats
        full_kv_bytes = 4 * (layers * 2 * kv_h * t_len * hd)
        cache_ratio = full_kv_bytes / float(max(1, ufwm_state_bytes))

        # Associative capacity & SIR analysis
        total_rank_capacity = banks * d
        active_compute_fraction = k_act / float(banks)
        if surprise_trace is not None:
            s_vals = _require_float_seq(surprise_trace, "surprise_trace", min_len=1, max_len=100_000)
            if any(v < 0.0 for v in s_vals):
                raise ValueError("surprise_trace entries must be >= 0")
            mean_surprise = statistics.fmean(s_vals)
            high_surprise_writes = sum(1 for v in s_vals if v >= max(0.25, mean_surprise))
            effective_writes = max(1.0, float(high_surprise_writes) * (k_act / float(banks)))
        else:
            mean_surprise = 0.35
            effective_writes = max(1.0, 0.35 * float(t_len) * (k_act / float(banks)))

        sir_linear = (banks * d * d) / effective_writes
        sir_db = 10.0 * math.log10(max(1e-6, sir_linear))

        body = {
            "mode": mode_s,
            "hidden_size": h,
            "n_layers": layers,
            "fast_dim": d,
            "n_banks": banks,
            "active_banks_per_step": k_act,
            "active_bank_fraction": round(active_compute_fraction, 6),
            "shared_programmer_params": int(shared_params),
            "unshared_equivalent_params": int(unshared_equiv_params),
            "parameter_sharing_ratio": round(sharing_ratio, 4),
            "total_rank_capacity": int(total_rank_capacity),
            "ufwm_state_bytes": int(ufwm_state_bytes),
            "full_attention_kv_bytes_at_seq_len": int(full_kv_bytes),
            "kv_vs_ufwm_compression_ratio": round(cache_ratio, 4),
            "mean_surprise": round(mean_surprise, 6),
            "estimated_sir_db": round(sir_db, 4),
            "regime": (
                "high_fidelity_associative"
                if sir_db >= 6.0
                else ("moderate_interference" if sir_db >= 0.0 else "palimpsest_saturation")
            ),
            "answer_authority": False,
        }
        body["receipt"] = _build_receipt("UniversalFastWeightProfiler", body)
        return body


# ---------------------------------------------------------------------------
# 2. Connectome Plasticity Orchestrator (BTSP + MICrONS Spine-Apparatus + AGMP)
# ---------------------------------------------------------------------------


class ConnectomePlasticityOrchestrator:
    """Simulates 3-factor BTSP + MICrONS dual-regime spine switching + astrocyte gating."""

    def orchestrate(
        self,
        *,
        edge_magnitudes: Sequence[float],
        pre_rates: Sequence[float],
        post_rates: Sequence[float],
        modulatory_signal: float = 0.5,
        astrocyte_calcium: float = 0.30,
        spine_threshold: float = 0.35,
        lr: float = 0.05,
        dale_signs: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        mags = _require_float_seq(edge_magnitudes, "edge_magnitudes", min_len=1, max_len=10_000)
        pre = _require_float_seq(pre_rates, "pre_rates", min_len=len(mags), max_len=len(mags))
        post = _require_float_seq(post_rates, "post_rates", min_len=len(mags), max_len=len(mags))
        if any(m < 0.0 for m in mags):
            raise ValueError("edge_magnitudes must be non-negative")
        mod = _require_finite(modulatory_signal, "modulatory_signal")
        if not (-2.0 <= mod <= 2.0):
            raise ValueError(f"modulatory_signal must lie in [-2, 2], got {mod}")
        astro_ca = _require_finite(astrocyte_calcium, "astrocyte_calcium")
        if not (0.0 <= astro_ca <= 5.0):
            raise ValueError(f"astrocyte_calcium must lie in [0, 5], got {astro_ca}")
        thresh = _require_finite(spine_threshold, "spine_threshold")
        if thresh <= 0.0:
            raise ValueError(f"spine_threshold must be > 0, got {thresh}")
        step_lr = _require_finite(lr, "lr")
        if not (0.0 < step_lr <= 1.0):
            raise ValueError(f"lr must lie in (0, 1], got {step_lr}")

        signs: List[int]
        if dale_signs is not None:
            if not isinstance(dale_signs, (list, tuple)) or len(dale_signs) != len(mags):
                raise ValueError("dale_signs must match edge_magnitudes length")
            signs = []
            for idx, s in enumerate(dale_signs):
                si = _require_int(s, f"dale_signs[{idx}]")
                if si not in (-1, 1):
                    raise ValueError(f"dale_signs[{idx}] must be -1 or +1, got {si}")
                signs.append(si)
        else:
            signs = [1] * len(mags)

        # Astrocytic gliotransmitter gate (AGMP): unlocks plasticity when Ca2+ exceeds calm baseline 0.25
        astro_gate = 0.5 + 1.0 / (1.0 + math.exp(-4.0 * (astro_ca - 0.25)))
        eff_lr = step_lr * astro_gate

        updated_mags: List[float] = []
        signed_before: List[float] = []
        signed_after: List[float] = []
        nascent_count = 0
        mature_count = 0
        dale_violations = 0

        for m_i, r_pre, r_post, s_i in zip(mags, pre, post, signs):
            coinc = r_post * r_pre
            if m_i < thresh:
                nascent_count += 1
                elig = coinc - 0.15 * (r_post * r_post) * m_i
            else:
                mature_count += 1
                elig = 0.35 * (coinc - 0.30 * m_i)
            delta = max(-0.25, min(0.25, eff_lr * mod * elig))
            m_new = max(1e-6, min(12.0, m_i + delta))
            updated_mags.append(round(m_new, 6))
            sb = s_i * m_i
            sa = s_i * m_new
            signed_before.append(sb)
            signed_after.append(sa)
            if (sb > 0 and sa < 0) or (sb < 0 and sa > 0) or (s_i > 0 and sa < 0) or (s_i < 0 and sa > 0):
                dale_violations += 1

        n_edges = len(mags)
        mean_delta = statistics.fmean(abs(a - b) for a, b in zip(updated_mags, mags))
        rms_before = math.sqrt(statistics.fmean(m * m for m in mags))
        rms_after = math.sqrt(statistics.fmean(m * m for m in updated_mags))

        body = {
            "n_synapses": n_edges,
            "nascent_count": nascent_count,
            "mature_spine_apparatus_count": mature_count,
            "mature_fraction": round(mature_count / float(n_edges), 6),
            "astrocyte_gliotransmitter_gate": round(astro_gate, 6),
            "effective_lr": round(eff_lr, 6),
            "dale_sign_violations": dale_violations,
            "dale_law_preserved": dale_violations == 0,
            "mean_abs_magnitude_delta": round(mean_delta, 6),
            "rms_magnitude_before": round(rms_before, 6),
            "rms_magnitude_after": round(rms_after, 6),
            "updated_magnitudes": updated_mags,
            "answer_authority": False,
        }
        body["receipt"] = _build_receipt("ConnectomePlasticityOrchestrator", body)
        return body


# ---------------------------------------------------------------------------
# 3. Synthetic Neuron Foundry Assay (6 Organelle Families)
# ---------------------------------------------------------------------------


class SyntheticNeuronFoundryAssay:
    """Dynamical and metabolic benchmark across all 6 v107 synthetic neuron families."""

    _METABOLIC_COSTS: Dict[str, float] = {
        "soliton_axon": 1.15,
        "spine_apparatus": 1.25,
        "astrocyte_tripartite": 1.10,
        "ephaptic_field": 0.85,
        "retrograde_messengers": 1.05,
        "fast_weight_engram": 1.35,
    }

    def assay(
        self,
        *,
        stimulus_series: Sequence[float],
        foundry_dim: int = 32,
        metabolic_budget: float = 40.0,
    ) -> Dict[str, Any]:
        stim = _require_float_seq(stimulus_series, "stimulus_series", min_len=2, max_len=10_000)
        dim = _require_int(foundry_dim, "foundry_dim", min_value=4, max_value=4096)
        budget = _require_finite(metabolic_budget, "metabolic_budget")
        if budget <= 0.0:
            raise ValueError(f"metabolic_budget must be > 0, got {budget}")

        # Deterministic scalar organelle dynamical trajectories over stimulus_series
        u_sol, r_sol = 0.0, 0.0
        soma_sp, ca_sp = 0.0, 0.0
        v_ast, ca_ast = 0.0, 0.0
        v_eph = 0.0
        v_ret, msg_ret = 0.0, 0.0
        fk_eng, fv_eng, sv_eng = 0.0, 0.0, 0.0

        traces: Dict[str, List[float]] = {k: [] for k in self._METABOLIC_COSTS}
        for x in stim:
            # 1. Soliton axon
            pulse = math.tanh(1.5 * (x + 0.5 * u_sol) - 0.15 * u_sol) - 0.15 * r_sol
            u_sol = 0.45 * u_sol + 0.55 * pulse
            r_sol = 0.75 * r_sol + 0.25 * max(0.0, u_sol)
            traces["soliton_axon"].append(u_sol)

            # 2. Spine apparatus
            ca_sp = 0.80 * ca_sp + 0.20 * min(4.0, abs(x * soma_sp))
            sw = 1.0 / (1.0 + math.exp(-4.0 * (ca_sp - 0.30)))
            nasc = math.tanh(x - 0.2 * soma_sp * soma_sp * x)
            mat = math.tanh((x - 0.6 * soma_sp) + 0.5 * x)
            soma_sp = 0.5 * soma_sp + 0.5 * ((1.0 - sw) * nasc + sw * mat)
            traces["spine_apparatus"].append(soma_sp)

            # 3. Astrocyte tripartite
            ca_ast = 0.90 * ca_ast + 0.10 * (0.5 * abs(v_ast) + 0.5 * min(4.0, abs(x - v_ast)))
            glio = 1.0 / (1.0 + math.exp(-(ca_ast - 0.25)))
            eff_l = min(0.95, max(0.05, 0.35 * (0.5 + glio)))
            v_ast = (1.0 - eff_l) * v_ast + eff_l * math.tanh(x * (0.75 + 0.5 * glio))
            traces["astrocyte_tripartite"].append(v_ast)

            # 4. Ephaptic field
            v_ext = 0.5 * x + 0.5 * math.tanh(v_eph)
            tm = x + 0.20 * (v_ext - v_eph)
            silu_tm = tm / (1.0 + math.exp(-max(-20.0, min(20.0, tm))))
            v_eph = 0.5 * v_eph + 0.5 * max(-6.0, min(6.0, silu_tm))
            traces["ephaptic_field"].append(v_eph)

            # 5. Retrograde messengers
            p_rel = max(0.35, min(1.45, 1.0 - 0.35 * msg_ret))
            v_ret = 0.5 * v_ret + 0.5 * math.tanh(p_rel * x)
            msg_ret = 0.75 * msg_ret + 0.25 * math.tanh(v_ret * v_ret)
            traces["retrograde_messengers"].append(v_ret)

            # 6. Fast-weight engram
            k_sign = 1.0 if x >= 0.0 else -1.0
            v_t = math.tanh(x)
            err_e = v_t - (fk_eng * k_sign) * fv_eng
            fk_eng = 0.65 * fk_eng + 0.35 * k_sign
            fv_eng = max(-4.0, min(4.0, 0.80 * fv_eng + 0.35 * err_e))
            sv_eng = 0.95 * sv_eng + 0.05 * fv_eng
            out_e = math.tanh(x + (fk_eng * k_sign) * (0.65 * fv_eng + 0.35 * sv_eng))
            traces["fast_weight_engram"].append(out_e)

        family_profiles: List[Dict[str, Any]] = []
        raw_utilities: Dict[str, float] = {}
        for fam, series in traces.items():
            rms = math.sqrt(statistics.fmean(v * v for v in series))
            # Lag-1 autocorrelation measures short/medium retention horizon
            if len(series) >= 3:
                m_s = statistics.fmean(series)
                num = sum((series[i] - m_s) * (series[i - 1] - m_s) for i in range(1, len(series)))
                den = sum((v - m_s) ** 2 for v in series)
                autocorr = abs(num / den) if den > 1e-12 else 0.0
            else:
                autocorr = abs(series[-1] - series[0])
            cost = self._METABOLIC_COSTS[fam] * float(dim)
            efficiency = (0.6 * rms + 0.4 * autocorr) / max(1e-6, self._METABOLIC_COSTS[fam])
            raw_utilities[fam] = efficiency
            family_profiles.append(
                {
                    "family": fam,
                    "response_rms": round(rms, 6),
                    "temporal_persistence": round(autocorr, 6),
                    "metabolic_cost_at_dim": round(cost, 4),
                    "utility_per_atp": round(efficiency, 6),
                }
            )

        # Budget-aware routing mixture: shift toward low-cost ephaptic/retrograde if budget is tight
        min_possible_cost = min(self._METABOLIC_COSTS.values()) * float(dim)
        temp = 0.25 if budget >= 1.15 * float(dim) else 0.12
        scores: Dict[str, float] = {}
        for fam, util in raw_utilities.items():
            fam_cost = self._METABOLIC_COSTS[fam] * float(dim)
            penalty = max(0.0, (fam_cost - budget) / max(1.0, float(dim)))
            scores[fam] = (util - 0.8 * penalty) / temp
        max_sc = max(scores.values())
        exps = {f: math.exp(sc - max_sc) for f, sc in scores.items()}
        z_sum = sum(exps.values())
        mixture = {f: round(v / z_sum, 6) for f, v in exps.items()}
        expected_cost = sum(mixture[f] * self._METABOLIC_COSTS[f] * float(dim) for f in mixture)
        dominant = max(mixture, key=lambda k: mixture[k])

        body = {
            "foundry_dim": dim,
            "stimulus_length": len(stim),
            "metabolic_budget": round(budget, 4),
            "budget_feasible": budget >= min_possible_cost,
            "expected_metabolic_cost": round(expected_cost, 4),
            "dominant_family": dominant,
            "recommended_mixture": mixture,
            "family_profiles": family_profiles,
            "answer_authority": False,
        }
        body["receipt"] = _build_receipt("SyntheticNeuronFoundryAssay", body)
        return body


# ---------------------------------------------------------------------------
# 4. Antipode Failure Distiller (SONAPSE-MORPHOGENESIS antipode/)
# ---------------------------------------------------------------------------


class AntipodeFailureDistiller:
    """Clusters failure trajectories and synthesizes anti-collapse curriculum weights."""

    def distill(
        self,
        *,
        records: Sequence[Mapping[str, Any]],
        temperature: float = 0.5,
        uniform_floor: float = 0.15,
    ) -> Dict[str, Any]:
        if not isinstance(records, (list, tuple)) or len(records) == 0:
            raise ValueError("records must be a non-empty sequence of trajectory dicts")
        if len(records) > 50_000:
            raise ValueError("records exceeds maximum length 50000")
        temp = _require_finite(temperature, "temperature")
        if not (0.05 <= temp <= 10.0):
            raise ValueError(f"temperature must lie in [0.05, 10.0], got {temp}")
        floor = _require_finite(uniform_floor, "uniform_floor")
        if not (0.0 <= floor <= 0.8):
            raise ValueError(f"uniform_floor must lie in [0.0, 0.8], got {floor}")

        clusters: Dict[str, Dict[str, Any]] = {}
        for idx, rec in enumerate(records):
            if not isinstance(rec, Mapping):
                raise ValueError(f"records[{idx}] must be a mapping")
            cat = str(rec.get("category") or rec.get("prompt_category") or "").strip()
            if not cat:
                raise ValueError(f"records[{idx}] requires a non-empty 'category'")
            loss = _require_finite(rec.get("loss", 0.0), f"records[{idx}].loss")
            entropy = _require_finite(rec.get("entropy", 0.0), f"records[{idx}].entropy")
            if loss < 0.0 or entropy < 0.0:
                raise ValueError(f"records[{idx}] loss and entropy must be >= 0")
            failed = bool(rec.get("failed", loss >= 1.0))

            bucket = clusters.setdefault(
                cat,
                {"count": 0, "failures": 0, "loss_sum": 0.0, "entropy_sum": 0.0},
            )
            bucket["count"] += 1
            bucket["failures"] += 1 if failed else 0
            bucket["loss_sum"] += loss
            bucket["entropy_sum"] += entropy

        cluster_summaries: List[Dict[str, Any]] = []
        severities: Dict[str, float] = {}
        for cat in sorted(clusters):
            b = clusters[cat]
            cnt = int(b["count"])
            fail_rate = float(b["failures"]) / float(cnt)
            mean_loss = float(b["loss_sum"]) / float(cnt)
            mean_ent = float(b["entropy_sum"]) / float(cnt)
            # Antipode severity: prioritizes high failure rate + high loss + epistemic uncertainty
            severity = 0.50 * fail_rate + 0.35 * math.tanh(mean_loss / 2.0) + 0.15 * math.tanh(mean_ent / 2.0)
            severities[cat] = severity
            cluster_summaries.append(
                {
                    "category": cat,
                    "count": cnt,
                    "failures": int(b["failures"]),
                    "failure_rate": round(fail_rate, 6),
                    "mean_loss": round(mean_loss, 6),
                    "mean_entropy": round(mean_ent, 6),
                    "antipode_severity": round(severity, 6),
                }
            )

        n_cats = len(severities)
        max_sev = max(severities.values())
        boltz = {c: math.exp((s - max_sev) / temp) for c, s in severities.items()}
        b_sum = sum(boltz.values())
        curriculum_weights = {
            c: round((1.0 - floor) * (boltz[c] / b_sum) + (floor / float(n_cats)), 6)
            for c in sorted(severities)
        }
        hardest = max(severities, key=lambda c: severities[c])
        cluster_summaries.sort(key=lambda item: item["antipode_severity"], reverse=True)

        body = {
            "total_records": len(records),
            "cluster_count": n_cats,
            "primary_antipode_category": hardest,
            "curriculum_weights": curriculum_weights,
            "clusters": cluster_summaries,
            "answer_authority": False,
        }
        body["receipt"] = _build_receipt("AntipodeFailureDistiller", body)
        return body


# ---------------------------------------------------------------------------
# 5. Symbiosis Continual Retention Auditor (SONAPSE-MORPHOGENESIS symbiosis/)
# ---------------------------------------------------------------------------


class SymbiosisContinualRetentionAuditor:
    """Audits Fisher-weighted EWC synaptic consolidation and continual retention."""

    def audit(
        self,
        *,
        anchor_weights: Sequence[float],
        current_weights: Sequence[float],
        fisher_importances: Sequence[float],
        old_task_loss_before: float,
        old_task_loss_after: float,
        new_task_loss_before: float,
        new_task_loss_after: float,
        ewc_lambda: float = 0.15,
    ) -> Dict[str, Any]:
        anchor = _require_float_seq(anchor_weights, "anchor_weights", min_len=1, max_len=50_000)
        current = _require_float_seq(
            current_weights, "current_weights", min_len=len(anchor), max_len=len(anchor)
        )
        fisher = _require_float_seq(
            fisher_importances, "fisher_importances", min_len=len(anchor), max_len=len(anchor)
        )
        if any(f < 0.0 for f in fisher):
            raise ValueError("fisher_importances must be non-negative")
        old_b = _require_finite(old_task_loss_before, "old_task_loss_before")
        old_a = _require_finite(old_task_loss_after, "old_task_loss_after")
        new_b = _require_finite(new_task_loss_before, "new_task_loss_before")
        new_a = _require_finite(new_task_loss_after, "new_task_loss_after")
        if min(old_b, old_a, new_b, new_a) < 0.0:
            raise ValueError("task losses must be >= 0")
        lam = _require_finite(ewc_lambda, "ewc_lambda")
        if not (0.0 <= lam <= 100.0):
            raise ValueError(f"ewc_lambda must lie in [0, 100], got {lam}")

        f_mean = statistics.fmean(fisher)
        f_scale = f_mean if f_mean > 1e-12 else 1.0
        ewc_penalty = 0.5 * lam * sum(
            f_i * ((w_c - w_a) ** 2) for w_a, w_c, f_i in zip(anchor, current, fisher)
        )
        consolidated: List[float] = []
        for w_a, w_c, f_i in zip(anchor, current, fisher):
            f_norm = min(10.0, f_i / f_scale)
            c = lam * f_norm
            w_prox = (w_c + c * w_a) / (1.0 + c)
            consolidated.append(round(w_prox, 6))

        drift_before = statistics.fmean(abs(w_c - w_a) for w_a, w_c in zip(anchor, current))
        drift_after = statistics.fmean(abs(w_p - w_a) for w_a, w_p in zip(anchor, consolidated))
        forgetting_delta = old_a - old_b
        plasticity_gain = new_b - new_a

        if forgetting_delta <= 0.05 and plasticity_gain >= 0.0:
            grade = "stable_symbiotic"
        elif forgetting_delta <= 0.25:
            grade = "mild_drift"
        else:
            grade = "catastrophic_forgetting_risk"

        body = {
            "n_parameters": len(anchor),
            "ewc_lambda": round(lam, 6),
            "ewc_quadratic_penalty": round(ewc_penalty, 6),
            "mean_l1_drift_before": round(drift_before, 6),
            "mean_l1_drift_after_consolidation": round(drift_after, 6),
            "forgetting_loss_delta": round(forgetting_delta, 6),
            "new_task_plasticity_gain": round(plasticity_gain, 6),
            "retention_grade": grade,
            "consolidated_weights": consolidated,
            "answer_authority": False,
        }
        body["receipt"] = _build_receipt("SymbiosisContinualRetentionAuditor", body)
        return body
