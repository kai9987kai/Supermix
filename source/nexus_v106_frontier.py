"""Supermix v106 Frontier Synthesis, Diagnostics & Test-Time Compute Engines.

Five pure-stdlib, deterministic, zero-``eval``, fail-closed analytical engines
with canonical SHA-256 audit receipts (``supermix-v106-receipt-v1``):

1. ``TestTimeComputeAllocator``:
   Optimal allocation of a fixed inference token budget across parallel
   Best-of-N sampling, sequential self-refinement steps, and verification
   passes, with a non-dominated cost/accuracy Pareto frontier.
2. ``SpectralWeightDiagnostics``:
   Heavy-tailed spectral diagnostics (Hill/CSN tail exponent alpha, stable
   rank, Roy-Vetterli effective rank, condition ratio, and Marchenko-Pastur
   bulk spike count) for weight/activation singular-value spectra.
3. ``CausalCounterfactualVerifier``:
   Pearl's 3-step Twin-Network Structural Causal Model (Abduction -> Action ->
   Prediction) with Natural Direct/Indirect Effect (NDE/NIE) decomposition and
   VanderWeele-Ding E-value sensitivity bounds.
4. ``AdaptiveQuantizationPlanner``:
   Mixed-precision per-layer bit-width allocation minimizing total
   sensitivity-weighted Shannon rate-distortion ``sum(w_i * 2^(-2 b_i))``
   subject to a target average bits-per-weight budget.
5. ``TelemetryDriftDetector``:
   Online two-sided CUSUM + Page-Hinkley sequential change-point detection over
   latency/entropy/reward streams paired with Jensen-Shannon routing histogram
   divergence.

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
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "FRONTIER_V106_VERSION",
    "RECEIPT_SCHEMA_V106",
    "TestTimeComputeAllocator",
    "SpectralWeightDiagnostics",
    "CausalCounterfactualVerifier",
    "AdaptiveQuantizationPlanner",
    "TelemetryDriftDetector",
]

FRONTIER_V106_VERSION = "v106_frontier_synthesis"
RECEIPT_SCHEMA_V106 = "supermix-v106-receipt-v1"


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


def _require_int(value: Any, name: str, *, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return int(value)


def _build_receipt(engine_name: str, body: Dict[str, Any]) -> Dict[str, Any]:
    canonical = json.dumps(
        {
            "schema": RECEIPT_SCHEMA_V106,
            "version": FRONTIER_V106_VERSION,
            "engine": engine_name,
            "body": body,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "schema": RECEIPT_SCHEMA_V106,
        "version": FRONTIER_V106_VERSION,
        "engine": engine_name,
        "receipt_sha256": digest,
        "answer_authority": False,
    }


# ---------------------------------------------------------------------------
# 1. Test-Time Compute Allocator
# ---------------------------------------------------------------------------


class TestTimeComputeAllocator:
    """Optimal test-time compute allocation over Best-of-N, refinement, and verification."""

    def allocate(
        self,
        *,
        total_token_budget: int = 4096,
        base_pass1: float = 0.35,
        tokens_per_sample: int = 256,
        refinement_gain_per_step: float = 0.08,
        tokens_per_refinement: int = 192,
        verifier_accuracy: float = 0.85,
        tokens_per_verify: int = 64,
        max_parallel_n: int = 16,
        max_refinement_steps: int = 6,
    ) -> Dict[str, Any]:
        budget = _require_int(total_token_budget, "total_token_budget", min_value=64, max_value=10_000_000)
        p1 = _require_finite(base_pass1, "base_pass1")
        if not (0.0 < p1 < 1.0):
            raise ValueError(f"base_pass1 must lie in (0, 1), got {p1}")
        tok_s = _require_int(tokens_per_sample, "tokens_per_sample", min_value=1, max_value=1_000_000)
        r_gain = _require_finite(refinement_gain_per_step, "refinement_gain_per_step")
        if not (0.0 <= r_gain <= 0.5):
            raise ValueError(f"refinement_gain_per_step must lie in [0, 0.5], got {r_gain}")
        tok_r = _require_int(tokens_per_refinement, "tokens_per_refinement", min_value=1, max_value=1_000_000)
        v_acc = _require_finite(verifier_accuracy, "verifier_accuracy")
        if not (0.5 <= v_acc <= 1.0):
            raise ValueError(f"verifier_accuracy must lie in [0.5, 1.0], got {v_acc}")
        tok_v = _require_int(tokens_per_verify, "tokens_per_verify", min_value=0, max_value=1_000_000)
        max_n = _require_int(max_parallel_n, "max_parallel_n", min_value=1, max_value=128)
        max_r = _require_int(max_refinement_steps, "max_refinement_steps", min_value=0, max_value=32)

        if tok_s > budget:
            raise ValueError(
                f"total_token_budget ({budget}) is smaller than a single sample ({tok_s})"
            )

        candidates: List[Dict[str, Any]] = []
        v_lift = 2.0 * v_acc - 1.0  # 0 at random verifier (0.5), 1 at exact oracle (1.0)

        for r in range(max_r + 1):
            p_single = 1.0 - (1.0 - p1) * ((1.0 - r_gain) ** r)
            for use_verifier in (False, True):
                per_sample_cost = tok_s + r * tok_r + (tok_v if use_verifier else 0)
                if per_sample_cost <= 0 or per_sample_cost > budget:
                    continue
                max_feasible_n = min(max_n, budget // per_sample_cost)
                for n in range(1, max_feasible_n + 1):
                    if n == 1 and use_verifier:
                        # Verifying a single sample cannot choose among alternatives; skip dominated option.
                        continue
                    total_cost = n * per_sample_cost
                    p_any = 1.0 - ((1.0 - p_single) ** n)
                    if n == 1:
                        p_sel = p_single
                    elif use_verifier:
                        p_sel = p_single + v_lift * (p_any - p_single)
                    else:
                        # Majority-vote self-consistency proxy: amplifies only when p_single > 0.5
                        sc_lift = min(0.85, max(0.0, (p_single - 0.5) * 1.4))
                        p_sel = p_single + sc_lift * (p_any - p_single)
                    p_sel = max(0.0, min(1.0, p_sel))
                    candidates.append(
                        {
                            "n_samples": n,
                            "refinement_steps": r,
                            "use_verifier": use_verifier,
                            "total_tokens": total_cost,
                            "per_sample_pass_prob": round(p_single, 6),
                            "pass_at_n_oracle": round(p_any, 6),
                            "expected_pass_prob": round(p_sel, 6),
                            "gain_over_pass1": round(p_sel - p1, 6),
                        }
                    )

        if not candidates:
            raise ValueError("no feasible test-time compute strategy fits within total_token_budget")

        best = max(
            candidates,
            key=lambda c: (c["expected_pass_prob"], -c["total_tokens"], -c["refinement_steps"]),
        )

        # Build non-dominated (total_tokens ascending, expected_pass_prob strictly increasing) frontier
        by_cost = sorted(candidates, key=lambda c: (c["total_tokens"], -c["expected_pass_prob"]))
        pareto: List[Dict[str, Any]] = []
        running_best = -1.0
        for row in by_cost:
            if row["expected_pass_prob"] > running_best + 1e-9:
                pareto.append(row)
                running_best = row["expected_pass_prob"]

        gain = best["expected_pass_prob"] - p1
        eff_per_1k = (gain / (best["total_tokens"] / 1000.0)) if best["total_tokens"] > 0 else 0.0

        if best["n_samples"] == 1 and best["refinement_steps"] == 0:
            regime = "single_shot_baseline"
        elif best["n_samples"] > 1 and best["refinement_steps"] > 0:
            regime = "hybrid_parallel_refinement"
        elif best["n_samples"] > 1:
            regime = "parallel_best_of_n"
        else:
            regime = "sequential_refinement"

        body = {
            "total_token_budget": budget,
            "base_pass1": round(p1, 6),
            "optimal_strategy": best,
            "recommended_regime": regime,
            "budget_utilization": round(best["total_tokens"] / float(budget), 6),
            "absolute_pass_gain": round(gain, 6),
            "gain_per_1k_tokens": round(eff_per_1k, 6),
            "candidates_evaluated": len(candidates),
            "pareto_frontier_count": len(pareto),
            "pareto_frontier": pareto[:12],
        }
        return {
            **body,
            "answer_authority": False,
            "receipt": _build_receipt("TestTimeComputeAllocator", body),
        }


# ---------------------------------------------------------------------------
# 2. Spectral Weight Diagnostics (Heavy-Tailed / Martin-Mahoney / MP)
# ---------------------------------------------------------------------------


class SpectralWeightDiagnostics:
    """Heavy-tailed spectral alpha, stable rank, and Marchenko-Pastur diagnostics."""

    def diagnose(
        self,
        *,
        singular_values: Sequence[float],
        matrix_shape: Optional[Sequence[int]] = None,
        layer_name: str = "layer",
    ) -> Dict[str, Any]:
        if not isinstance(singular_values, (list, tuple)) or len(singular_values) < 3:
            raise ValueError("singular_values must be a sequence of at least 3 non-negative numbers")

        svs: List[float] = []
        for idx, raw in enumerate(singular_values):
            val = _require_finite(raw, f"singular_values[{idx}]")
            if val < 0.0:
                raise ValueError(f"singular_values[{idx}] must be >= 0, got {val}")
            svs.append(val)

        svs.sort(reverse=True)
        pos_svs = [s for s in svs if s > 1e-12]
        if len(pos_svs) < 2:
            raise ValueError("singular_values must contain at least 2 strictly positive values")

        eigs = [s * s for s in pos_svs]
        frob_sq = sum(eigs)
        s_max = pos_svs[0]
        s_min = pos_svs[-1]
        stable_rank = frob_sq / (s_max * s_max)
        condition_number = s_max / s_min

        # Effective spectral entropy rank (Roy & Vetterli)
        probs = [e / frob_sq for e in eigs]
        entropy = -sum(p * math.log(max(p, 1e-15)) for p in probs)
        effective_rank = math.exp(entropy)
        normalized_rank = effective_rank / float(len(pos_svs))

        # Clauset-Shalizi-Newman / Hill power-law MLE on the top tail of eigenvalues
        tail_k = max(2, min(len(eigs), max(3, (len(eigs) + 1) // 2)))
        tail = eigs[:tail_k]
        lam_min = tail[-1]
        log_sum = sum(math.log(max(e / lam_min, 1.0)) for e in tail)
        if log_sum <= 1e-12:
            alpha_hill = 10.0  # flat / uniform spectrum -> steep cutoff
        else:
            alpha_hill = 1.0 + float(tail_k) / log_sum
        alpha_hill = min(20.0, alpha_hill)

        # Optional Marchenko-Pastur bulk upper edge & spike count
        mp_edge: Optional[float] = None
        mp_spike_count: Optional[int] = None
        aspect_ratio: Optional[float] = None
        if matrix_shape is not None:
            if not isinstance(matrix_shape, (list, tuple)) or len(matrix_shape) != 2:
                raise ValueError("matrix_shape must be a 2-element [rows, cols] sequence")
            r_dim = _require_int(matrix_shape[0], "matrix_shape[0]", min_value=2)
            c_dim = _require_int(matrix_shape[1], "matrix_shape[1]", min_value=2)
            q_ratio = float(max(r_dim, c_dim)) / float(min(r_dim, c_dim))
            aspect_ratio = round(q_ratio, 6)
            med_eig = statistics.median(eigs)
            # Bulk upper edge lambda_+ = sigma^2 * (1 + 1/sqrt(Q))^2 with median bulk calibration
            mp_edge_val = med_eig * ((1.0 + 1.0 / math.sqrt(q_ratio)) ** 2)
            mp_edge = round(mp_edge_val, 6)
            mp_spike_count = sum(1 for e in eigs if e > mp_edge_val)

        if stable_rank < 1.3 or normalized_rank < 0.15:
            health_status = "rank_collapsed"
            guidance = "Severe spectral collapse detected; reduce weight decay or add orthogonalization (Muon)."
        elif alpha_hill < 2.0:
            health_status = "heavy_spike_risk"
            guidance = "Extreme heavy-tail spike (alpha < 2.0); consider spectral-norm clipping (MuonClip)."
        elif 2.0 <= alpha_hill <= 4.5:
            health_status = "healthy_heavy_tailed"
            guidance = "Well-correlated heavy-tailed spectrum in the optimal Martin-Mahoney [2.0, 4.5] band."
        elif alpha_hill > 6.0:
            health_status = "over_regularized"
            guidance = "Near-random/under-trained bulk spectrum (alpha > 6.0); increase capacity or training steps."
        else:
            health_status = "moderate_tail"
            guidance = "Moderately structured spectrum; monitor alpha convergence as training progresses."

        body = {
            "layer_name": str(layer_name)[:64],
            "num_singular_values": len(svs),
            "positive_rank": len(pos_svs),
            "spectral_norm": round(s_max, 6),
            "frobenius_norm": round(math.sqrt(frob_sq), 6),
            "stable_rank": round(stable_rank, 6),
            "effective_rank": round(effective_rank, 6),
            "normalized_effective_rank": round(normalized_rank, 6),
            "condition_number": round(condition_number, 6),
            "power_law_alpha": round(alpha_hill, 6),
            "aspect_ratio_q": aspect_ratio,
            "marchenko_pastur_edge": mp_edge,
            "mp_spike_count": mp_spike_count,
            "health_status": health_status,
            "guidance": guidance,
        }
        return {
            **body,
            "answer_authority": False,
            "receipt": _build_receipt("SpectralWeightDiagnostics", body),
        }


# ---------------------------------------------------------------------------
# 3. Causal Counterfactual Verifier (Pearl Twin-Network SCM)
# ---------------------------------------------------------------------------


class CausalCounterfactualVerifier:
    """Exact linear-additive Structural Causal Model (Abduction-Action-Prediction)."""

    def verify_counterfactual(
        self,
        *,
        weights: Dict[str, float],
        observed: Dict[str, float],
        intervention: Dict[str, float],
        outcome_var: str = "Y",
        treatment_var: str = "X",
        mediator_var: Optional[str] = None,
        treatment_to_mediator: float = 0.0,
        mediator_to_outcome: float = 0.0,
        outcome_std: float = 1.0,
    ) -> Dict[str, Any]:
        if not isinstance(weights, dict) or not weights:
            raise ValueError("weights must be a non-empty mapping of variable -> structural weight")
        if not isinstance(observed, dict) or outcome_var not in observed:
            raise ValueError(f"observed must be a mapping containing outcome_var {outcome_var!r}")
        if not isinstance(intervention, dict) or not intervention:
            raise ValueError("intervention must be a non-empty mapping of do(var=val)")
        if outcome_var in intervention:
            raise ValueError("cannot intervene directly on the outcome_var being evaluated")

        clean_w: Dict[str, float] = {}
        for k, v in weights.items():
            if not isinstance(k, str) or not k.strip():
                raise ValueError("weight keys must be non-empty strings")
            clean_w[k] = _require_finite(v, f"weights[{k!r}]")

        clean_obs: Dict[str, float] = {}
        for k, v in observed.items():
            clean_obs[str(k)] = _require_finite(v, f"observed[{k!r}]")

        clean_do: Dict[str, float] = {}
        for k, v in intervention.items():
            clean_do[str(k)] = _require_finite(v, f"intervention[{k!r}]")

        for k in clean_w:
            if k not in clean_obs:
                raise ValueError(f"observed is missing structural parent {k!r}")

        a_tm = _require_finite(treatment_to_mediator, "treatment_to_mediator")
        w_mo = _require_finite(mediator_to_outcome, "mediator_to_outcome")
        y_std = _require_finite(outcome_std, "outcome_std")
        if y_std <= 0.0:
            raise ValueError(f"outcome_std must be > 0, got {y_std}")

        has_mediator = bool(mediator_var and str(mediator_var).strip())
        med_name = str(mediator_var).strip() if has_mediator else None
        if has_mediator and med_name is not None:
            if med_name not in clean_obs:
                raise ValueError(f"observed is missing mediator_var {med_name!r}")
            if treatment_var not in clean_obs:
                raise ValueError(f"observed is missing treatment_var {treatment_var!r}")

        # Step 1: Abduction — recover exact exogenous noise U_Y (and U_M if mediator present)
        y_obs = clean_obs[outcome_var]
        obs_linear = sum(clean_w[v] * clean_obs[v] for v in clean_w)
        if has_mediator and med_name is not None:
            obs_linear += w_mo * clean_obs[med_name]
            u_m = clean_obs[med_name] - a_tm * clean_obs[treatment_var]
        else:
            u_m = None
        u_y = y_obs - obs_linear

        # Step 2: Action — construct twin-network world under do(intervention)
        cf_world = dict(clean_obs)
        for k, v in clean_do.items():
            cf_world[k] = v

        # Propagate treatment -> mediator unless mediator itself is directly intervened
        if has_mediator and med_name is not None and med_name not in clean_do and u_m is not None:
            cf_world[med_name] = a_tm * cf_world[treatment_var] + u_m

        # Step 3: Prediction — evaluate counterfactual outcome Y_{do}
        cf_linear = sum(clean_w[v] * cf_world.get(v, 0.0) for v in clean_w)
        if has_mediator and med_name is not None:
            cf_linear += w_mo * cf_world[med_name]
        y_cf = cf_linear + u_y
        delta_y = y_cf - y_obs

        # NDE / NIE mediation decomposition along treatment_var (when intervened)
        nde = 0.0
        nie = 0.0
        if treatment_var in clean_do and treatment_var in clean_obs:
            dx = clean_do[treatment_var] - clean_obs[treatment_var]
            nde = clean_w.get(treatment_var, 0.0) * dx
            if has_mediator and med_name is not None and med_name not in clean_do:
                nie = a_tm * w_mo * dx

        # VanderWeele-Ding E-value sensitivity bound against unmeasured confounding
        std_effect = abs(delta_y) / y_std
        rr_approx = math.exp(min(20.0, 0.91 * std_effect))
        e_value = rr_approx + math.sqrt(max(0.0, rr_approx * (rr_approx - 1.0))) if rr_approx > 1.0 else 1.0

        body = {
            "outcome_var": outcome_var,
            "treatment_var": treatment_var,
            "mediator_var": med_name,
            "observed_outcome": round(y_obs, 6),
            "counterfactual_outcome": round(y_cf, 6),
            "counterfactual_delta": round(delta_y, 6),
            "abduced_noise": {
                "U_outcome": round(u_y, 6),
                **({"U_mediator": round(u_m, 6)} if u_m is not None else {}),
            },
            "natural_direct_effect": round(nde, 6),
            "natural_indirect_effect": round(nie, 6),
            "mediation_decomposition_exact": (
                abs((nde + nie) - delta_y) < 1e-6
                if (len(clean_do) == 1 and treatment_var in clean_do)
                else None
            ),
            "standardized_effect_size": round(std_effect, 6),
            "e_value_confounding_bound": round(e_value, 6),
            "intervention": clean_do,
        }
        return {
            **body,
            "answer_authority": False,
            "receipt": _build_receipt("CausalCounterfactualVerifier", body),
        }


# ---------------------------------------------------------------------------
# 4. Adaptive Mixed-Precision Quantization Planner
# ---------------------------------------------------------------------------


class AdaptiveQuantizationPlanner:
    """Sensitivity-weighted per-layer mixed-precision bit-width allocation optimizer."""

    def plan(
        self,
        *,
        layers: Sequence[Dict[str, Any]],
        target_avg_bits: float = 4.5,
        allowed_bits: Sequence[int] = (2, 3, 4, 6, 8, 16),
    ) -> Dict[str, Any]:
        if not isinstance(layers, (list, tuple)) or not layers:
            raise ValueError("layers must be a non-empty list of layer specs")
        target_b = _require_finite(target_avg_bits, "target_avg_bits")
        if target_b <= 0.0 or target_b > 32.0:
            raise ValueError(f"target_avg_bits must lie in (0, 32], got {target_b}")
        if not isinstance(allowed_bits, (list, tuple)) or not allowed_bits:
            raise ValueError("allowed_bits must be a non-empty sequence of positive integers")
        grid = sorted({_require_int(b, "allowed_bits", min_value=1, max_value=32) for b in allowed_bits})

        parsed: List[Dict[str, Any]] = []
        total_params = 0
        for idx, item in enumerate(layers):
            if not isinstance(item, dict):
                raise ValueError(f"layers[{idx}] must be a dict")
            name = str(item.get("name") or f"layer_{idx}")
            params = _require_int(item.get("params"), f"layers[{idx}].params", min_value=1)
            sens = _require_finite(item.get("sensitivity"), f"layers[{idx}].sensitivity")
            if sens <= 0.0:
                raise ValueError(f"layers[{idx}].sensitivity must be > 0, got {sens}")
            min_b = item.get("min_bits", grid[0])
            max_b = item.get("max_bits", grid[-1])
            min_b_i = _require_int(min_b, f"layers[{idx}].min_bits", min_value=1, max_value=32)
            max_b_i = _require_int(max_b, f"layers[{idx}].max_bits", min_value=1, max_value=32)
            layer_grid = [b for b in grid if min_b_i <= b <= max_b_i]
            if not layer_grid:
                raise ValueError(f"layers[{idx}] ({name}) has no allowed_bits in [{min_b_i}, {max_b_i}]")
            parsed.append(
                {
                    "name": name,
                    "params": params,
                    "sensitivity": sens,
                    "grid": layer_grid,
                    "grid_idx": 0,
                }
            )
            total_params += params

        min_total_bits = sum(row["params"] * row["grid"][0] for row in parsed)
        budget_bits = target_b * float(total_params)
        if min_total_bits > budget_bits + 1e-6:
            raise ValueError(
                f"target_avg_bits ({target_b}) is below the minimum feasible average "
                f"({min_total_bits / total_params:.4f})"
            )

        used_bits = float(min_total_bits)

        def _dist(sens: float, bits: int) -> float:
            return sens * (2.0 ** (-2.0 * float(bits)))

        # Greedy marginal distortion-reduction-per-bit allocation (exact for convex rate-distortion!)
        while True:
            best_i: Optional[int] = None
            best_score = -1.0
            for idx, row in enumerate(parsed):
                gi = row["grid_idx"]
                if gi + 1 >= len(row["grid"]):
                    continue
                b_cur = row["grid"][gi]
                b_nxt = row["grid"][gi + 1]
                delta_bits = float(row["params"] * (b_nxt - b_cur))
                if used_bits + delta_bits > budget_bits + 1e-6:
                    continue
                delta_dist = _dist(row["sensitivity"], b_cur) - _dist(row["sensitivity"], b_nxt)
                score = delta_dist / max(delta_bits, 1.0)
                if score > best_score:
                    best_score = score
                    best_i = idx
            if best_i is None:
                break
            row = parsed[best_i]
            b_cur = row["grid"][row["grid_idx"]]
            b_nxt = row["grid"][row["grid_idx"] + 1]
            used_bits += float(row["params"] * (b_nxt - b_cur))
            row["grid_idx"] += 1

        assignments: List[Dict[str, Any]] = []
        bit_hist: Dict[str, int] = {}
        total_distortion = 0.0
        for row in parsed:
            bits = row["grid"][row["grid_idx"]]
            dist = _dist(row["sensitivity"], bits)
            total_distortion += dist
            bit_hist[str(bits)] = bit_hist.get(str(bits), 0) + 1
            assignments.append(
                {
                    "name": row["name"],
                    "params": row["params"],
                    "sensitivity": round(row["sensitivity"], 6),
                    "assigned_bits": bits,
                    "estimated_distortion": round(dist, 8),
                }
            )

        # Uniform baseline: highest single bitwidth <= target_avg_bits
        uniform_candidates = [b for b in grid if float(b) <= target_b + 1e-9]
        uniform_bits = uniform_candidates[-1] if uniform_candidates else grid[0]
        uniform_distortion = sum(_dist(row["sensitivity"], uniform_bits) for row in parsed)
        dist_reduction = (
            1.0 - (total_distortion / uniform_distortion)
            if uniform_distortion > 1e-15
            else 0.0
        )
        achieved_avg = used_bits / float(total_params)

        body = {
            "total_params": total_params,
            "target_avg_bits": round(target_b, 4),
            "achieved_avg_bits": round(achieved_avg, 4),
            "compression_vs_fp16": round(16.0 / max(achieved_avg, 1e-6), 4),
            "compression_vs_fp32": round(32.0 / max(achieved_avg, 1e-6), 4),
            "mixed_precision_distortion": round(total_distortion, 8),
            "uniform_baseline_bits": uniform_bits,
            "uniform_baseline_distortion": round(uniform_distortion, 8),
            "distortion_reduction_vs_uniform": round(dist_reduction, 6),
            "bit_histogram": bit_hist,
            "layer_assignments": assignments,
        }
        return {
            **body,
            "answer_authority": False,
            "receipt": _build_receipt("AdaptiveQuantizationPlanner", body),
        }


# ---------------------------------------------------------------------------
# 5. Online Telemetry Drift Detector (CUSUM + Page-Hinkley + JSD)
# ---------------------------------------------------------------------------


class TelemetryDriftDetector:
    """Sequential CUSUM/Page-Hinkley drift & Jensen-Shannon distribution shift detector."""

    @staticmethod
    def _jensen_shannon_bits(p: Sequence[float], q: Sequence[float]) -> float:
        if len(p) != len(q) or len(p) == 0:
            raise ValueError("baseline_hist and recent_hist must be non-empty sequences of equal length")
        p_vals = [_require_finite(x, f"baseline_hist[{i}]") for i, x in enumerate(p)]
        q_vals = [_require_finite(x, f"recent_hist[{i}]") for i, x in enumerate(q)]
        if any(x < 0.0 for x in p_vals) or any(x < 0.0 for x in q_vals):
            raise ValueError("histogram weights must be non-negative")
        ps, qs = sum(p_vals), sum(q_vals)
        if ps <= 0.0 or qs <= 0.0:
            raise ValueError("histogram sums must be strictly positive")
        pn = [x / ps for x in p_vals]
        qn = [x / qs for x in q_vals]
        jsd = 0.0
        for pi, qi in zip(pn, qn):
            m = 0.5 * (pi + qi)
            if pi > 0.0:
                jsd += 0.5 * pi * math.log2(pi / m)
            if qi > 0.0:
                jsd += 0.5 * qi * math.log2(qi / m)
        return max(0.0, min(1.0, jsd))

    def detect(
        self,
        *,
        stream: Sequence[float],
        baseline_hist: Optional[Sequence[float]] = None,
        recent_hist: Optional[Sequence[float]] = None,
        slack_k: float = 0.5,
        threshold: float = 4.0,
        warmup: int = 5,
    ) -> Dict[str, Any]:
        if not isinstance(stream, (list, tuple)) or len(stream) < 4:
            raise ValueError("stream must be a sequence of at least 4 finite observations")
        vals = [_require_finite(x, f"stream[{i}]") for i, x in enumerate(stream)]
        k_slack = _require_finite(slack_k, "slack_k")
        if k_slack < 0.0:
            raise ValueError(f"slack_k must be >= 0, got {k_slack}")
        thresh = _require_finite(threshold, "threshold")
        if thresh <= 0.0:
            raise ValueError(f"threshold must be > 0, got {thresh}")
        w = _require_int(warmup, "warmup", min_value=2, max_value=len(vals) - 1)

        base_slice = vals[:w]
        base_mean = statistics.mean(base_slice)
        base_std = statistics.pstdev(base_slice)
        eff_std = max(base_std, abs(base_mean) * 0.05, 1e-6)

        # Two-sided standardized CUSUM + Page-Hinkley tracking
        s_pos = 0.0
        s_neg = 0.0
        max_pos = 0.0
        max_neg = 0.0
        change_point: Optional[int] = None
        direction = "stable"

        ph_cum = 0.0
        ph_min = 0.0
        ph_max_stat = 0.0
        running_sum = sum(base_slice)
        for idx in range(w, len(vals)):
            x = vals[idx]
            z = (x - base_mean) / eff_std
            s_pos = max(0.0, s_pos + z - k_slack)
            s_neg = max(0.0, s_neg - z - k_slack)
            if s_pos > max_pos:
                max_pos = s_pos
            if s_neg > max_neg:
                max_neg = s_neg
            if change_point is None and (s_pos > thresh or s_neg > thresh):
                change_point = idx
                direction = "upward_shift" if s_pos >= s_neg else "downward_shift"

            running_sum += x
            running_mean = running_sum / float(idx + 1)
            ph_cum += (x - running_mean) / eff_std - k_slack * 0.5
            if ph_cum < ph_min:
                ph_min = ph_cum
            ph_max_stat = max(ph_max_stat, ph_cum - ph_min)

        max_cusum = max(max_pos, max_neg)
        tail_slice = vals[w:]
        tail_mean = statistics.mean(tail_slice)
        shift_z = (tail_mean - base_mean) / eff_std

        jsd_bits: Optional[float] = None
        if baseline_hist is not None or recent_hist is not None:
            if baseline_hist is None or recent_hist is None:
                raise ValueError("both baseline_hist and recent_hist must be provided together")
            jsd_bits = round(self._jensen_shannon_bits(baseline_hist, recent_hist), 6)

        hist_alert = jsd_bits is not None and jsd_bits >= 0.15
        drift_detected = (change_point is not None) or hist_alert
        if (max_cusum >= 2.0 * thresh) or (jsd_bits is not None and jsd_bits >= 0.35):
            severity = "critical_drift"
            recommendation = "Trigger shadow recalibration and verify routing/quantization telemetry immediately."
        elif drift_detected:
            severity = "drift_detected"
            recommendation = "Inspect post-change-point traces and tighten conformal/adaptive thresholds."
        elif max_cusum >= 0.65 * thresh or (jsd_bits is not None and jsd_bits >= 0.08):
            severity = "watch"
            recommendation = "Mild elevation in CUSUM/JSD; continue monitoring rolling telemetry window."
        else:
            severity = "nominal"
            recommendation = "Stream and routing distributions remain statistically stationary."

        body = {
            "observations": len(vals),
            "warmup_window": w,
            "baseline_mean": round(base_mean, 6),
            "baseline_std": round(base_std, 6),
            "tail_mean": round(tail_mean, 6),
            "tail_shift_z": round(shift_z, 6),
            "max_cusum_pos": round(max_pos, 6),
            "max_cusum_neg": round(max_neg, 6),
            "max_cusum": round(max_cusum, 6),
            "page_hinkley_stat": round(ph_max_stat, 6),
            "threshold": round(thresh, 6),
            "drift_detected": drift_detected,
            "change_point_index": change_point,
            "shift_direction": direction,
            "histogram_jsd_bits": jsd_bits,
            "severity": severity,
            "recommendation": recommendation,
        }
        return {
            **body,
            "answer_authority": False,
            "receipt": _build_receipt("TelemetryDriftDetector", body),
        }
