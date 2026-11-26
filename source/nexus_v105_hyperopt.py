"""Supermix v105 Hyper-Optimization, Systems Profiling & Statistical Guard Engines.

Five deterministic, zero-external-dependency performance-engineering and
systems-optimization engines extending the v102-v104 frontier ecosystem:

1. ``SpeculativeDraftOptimizer`` — Analytical optimal draft-depth solver for
   MTP self-speculative decoding: computes exact expected tokens/step
   ``E[T(k)] = 1 + sum_{j=1}^k prod_{i=1}^j p_i``, exact cost-adjusted net
   speedup vs greedy across ``k = 0..K``, optimal depth ``k*``, and adaptive
   throttling thresholds.
2. ``KVCacheCompressionPlanner`` — Exact layer-by-layer KV + recurrent state
   memory footprint planner across SWA/Global/MLA/Engram/DeltaNet/Titans/CNS
   up to 1M-token contexts, comparing ``fp16``, ``fp8``, ``int8_symmetric``,
   and ``int4_group`` against a strict ``memory_budget_mb`` ceiling.
3. ``RooflineComputeProfiler`` — Hardware-aware analytical Roofline model
   computing prefill & decode FLOPs, memory bytes, arithmetic intensity
   (FLOPs/byte), hardware ridge point, memory-bound vs compute-bound regime,
   and theoretical throughput ceilings (tokens/sec).
4. ``CascadeEnsembleRouter`` — Cost-quality optimal multi-tier cascade planner
   (e.g., symbolic/flash fast tier escalating to deep/pro tier) selecting the
   exact accuracy-maximizing confidence threshold under a strict average
   latency budget.
5. ``SelfHealingBenchmarkGuard`` — Paired statistical regression & speedup
   guard computing paired latency speedup, p95 delta, paired quality delta,
   95% confidence intervals, Cohen's ``d_z`` effect size, and fail-closed
   ``APPROVE_OPTIMIZATION`` / ``HOLD_*`` release gating.

Engineering contract:
- Every engine sets ``answer_authority=False`` and emits a canonical SHA-256
  receipt so diagnostics can never masquerade as closed-world truth proofs.
- Zero ``eval``/``exec``; pure Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EVOLUTION_V105_VERSION = "supermix-v105-hyperopt-v1"
RECEIPT_SCHEMA_V105 = "supermix-v105-receipt-v1"

__all__ = [
    "EVOLUTION_V105_VERSION",
    "RECEIPT_SCHEMA_V105",
    "SpeculativeDraftOptimizer",
    "KVCacheCompressionPlanner",
    "RooflineComputeProfiler",
    "CascadeEnsembleRouter",
    "SelfHealingBenchmarkGuard",
]


def _canonical_receipt(engine: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA_V105,
        "evolution_version": EVOLUTION_V105_VERSION,
        "engine": str(engine),
        "answer_authority": False,
        "payload": payload,
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "schema_version": RECEIPT_SCHEMA_V105,
        "evolution_version": EVOLUTION_V105_VERSION,
        "engine": str(engine),
        "answer_authority": False,
        "receipt_sha256": digest,
    }


# ---------------------------------------------------------------------------
# 1. SpeculativeDraftOptimizer
# ---------------------------------------------------------------------------


class SpeculativeDraftOptimizer:
    """Exact optimal draft-depth solver for MTP self-speculative decoding."""

    def optimize(
        self,
        *,
        depth_accept_probs: Sequence[float],
        draft_cost_ratio: float = 0.08,
        verify_overhead_per_token: float = 0.04,
    ) -> Dict[str, Any]:
        probs = [float(p) for p in depth_accept_probs]
        if not probs:
            raise ValueError("depth_accept_probs must be non-empty")
        if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in probs):
            raise ValueError("depth_accept_probs values must be finite probabilities in [0, 1]")
        d_cost = float(draft_cost_ratio)
        v_over = float(verify_overhead_per_token)
        if not math.isfinite(d_cost) or d_cost < 0.0:
            raise ValueError("draft_cost_ratio must be finite and >= 0")
        if not math.isfinite(v_over) or v_over < 0.0:
            raise ValueError("verify_overhead_per_token must be finite and >= 0")

        sweep: List[Dict[str, Any]] = []
        survival = 1.0
        expected_tokens = 1.0
        sweep.append(
            {
                "draft_depth": 0,
                "expected_tokens_per_step": 1.0,
                "relative_step_cost": 1.0,
                "net_speedup": 1.0,
                "marginal_acceptance_prob": 1.0,
            }
        )
        for k, p_k in enumerate(probs, start=1):
            survival *= p_k
            expected_tokens += survival
            step_cost = 1.0 + k * (d_cost + v_over)
            speedup = expected_tokens / step_cost
            sweep.append(
                {
                    "draft_depth": k,
                    "expected_tokens_per_step": round(expected_tokens, 6),
                    "relative_step_cost": round(step_cost, 6),
                    "net_speedup": round(speedup, 6),
                    "marginal_acceptance_prob": round(survival, 6),
                }
            )

        best = max(sweep, key=lambda row: (row["net_speedup"], -row["draft_depth"]))
        # Break-even minimum depth-1 acceptance rate to beat greedy:
        # (1 + p1) / (1 + d_cost + v_over) > 1  <=>  p1 > d_cost + v_over
        breakeven_p1 = min(1.0, d_cost + v_over)
        recommended_mode = "speculative_mtp" if best["draft_depth"] > 0 else "greedy_fallback"

        result: Dict[str, Any] = {
            "max_depths_evaluated": len(probs),
            "draft_cost_ratio": round(d_cost, 6),
            "verify_overhead_per_token": round(v_over, 6),
            "optimal_draft_depth": int(best["draft_depth"]),
            "optimal_expected_tokens": best["expected_tokens_per_step"],
            "optimal_net_speedup": best["net_speedup"],
            "breakeven_depth1_acceptance": round(breakeven_p1, 6),
            "recommended_mode": recommended_mode,
            "depth_sweep": sweep,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("SpeculativeDraftOptimizer", result)
        return result


# ---------------------------------------------------------------------------
# 2. KVCacheCompressionPlanner
# ---------------------------------------------------------------------------


class KVCacheCompressionPlanner:
    """Exact hybrid-attention + recurrent-state KV memory planner up to 1M context."""

    _SCHEMES: Sequence[Dict[str, Any]] = (
        {"name": "fp16", "bytes_per_elem": 2.0, "scale_overhead": 0.0, "quality_retention": 1.000},
        {"name": "fp8_e4m3", "bytes_per_elem": 1.0, "scale_overhead": 0.02, "quality_retention": 0.997},
        {"name": "int8_symmetric", "bytes_per_elem": 1.0, "scale_overhead": 0.0625, "quality_retention": 0.994},
        {"name": "int4_group32", "bytes_per_elem": 0.5, "scale_overhead": 0.0625, "quality_retention": 0.978},
    )

    def plan(
        self,
        *,
        seq_len: int = 32768,
        batch_size: int = 1,
        n_layers: int = 24,
        hybrid_ratio: int = 5,
        sliding_window: int = 128,
        n_kv_heads: int = 4,
        head_dim: int = 64,
        use_mla: bool = False,
        mla_latent_dim: int = 64,
        mla_pe_dim: int = 32,
        deltanet_dim: int = 0,
        titans_dim: int = 0,
        cns_nodes: int = 0,
        memory_budget_mb: float = 256.0,
    ) -> Dict[str, Any]:
        if int(seq_len) < 1 or int(batch_size) < 1 or int(n_layers) < 1:
            raise ValueError("seq_len, batch_size, and n_layers must be >= 1")
        if int(sliding_window) < 1 or int(n_kv_heads) < 1 or int(head_dim) < 2:
            raise ValueError("sliding_window, n_kv_heads must be >= 1 and head_dim >= 2")
        if float(memory_budget_mb) <= 0.0 or not math.isfinite(float(memory_budget_mb)):
            raise ValueError("memory_budget_mb must be finite and > 0")

        t = int(seq_len)
        b = int(batch_size)
        layers = int(n_layers)
        ratio = max(0, int(hybrid_ratio))
        window = int(sliding_window)

        n_global = 0
        n_swa = 0
        for i in range(layers):
            is_global = (ratio == 0) or (((i + 1) % (ratio + 1)) == 0) or (i == layers - 1)
            if is_global:
                n_global += 1
            else:
                n_swa += 1

        # Elements per token per layer
        if use_mla:
            global_elems_per_tok = int(mla_latent_dim) + int(mla_pe_dim)
        else:
            global_elems_per_tok = 2 * int(n_kv_heads) * int(head_dim)
        swa_elems_per_tok = 2 * int(n_kv_heads) * int(head_dim)

        global_tokens = t
        swa_tokens = min(t, window)

        hybrid_kv_elems = b * (n_global * global_tokens * global_elems_per_tok + n_swa * swa_tokens * swa_elems_per_tok)
        all_global_kv_elems = b * layers * t * (2 * int(n_kv_heads) * int(head_dim))

        # Fixed-size O(1) recurrent state bytes (fp32 for stability)
        recurrent_elems = b * (
            max(0, int(deltanet_dim)) ** 2
            + 2 * (max(0, int(titans_dim)) ** 2)
            + max(0, int(cns_nodes))
        )
        recurrent_mb = (recurrent_elems * 4.0) / (1024.0 * 1024.0)

        schemes_report: List[Dict[str, Any]] = []
        selected_scheme: Optional[str] = None
        for sch in self._SCHEMES:
            eff_bpe = sch["bytes_per_elem"] + sch["scale_overhead"]
            kv_mb = (hybrid_kv_elems * eff_bpe) / (1024.0 * 1024.0)
            total_mb = kv_mb + recurrent_mb
            dense_fp16_mb = (all_global_kv_elems * 2.0) / (1024.0 * 1024.0)
            fits = total_mb <= float(memory_budget_mb)
            if fits and selected_scheme is None:
                selected_scheme = sch["name"]
            schemes_report.append(
                {
                    "scheme": sch["name"],
                    "kv_cache_mb": round(kv_mb, 4),
                    "recurrent_state_mb": round(recurrent_mb, 4),
                    "total_memory_mb": round(total_mb, 4),
                    "compression_vs_dense_fp16": round(dense_fp16_mb / max(1e-9, total_mb), 3),
                    "quality_retention": sch["quality_retention"],
                    "fits_budget": fits,
                }
            )

        hybrid_vs_dense_factor = (
            float(all_global_kv_elems) / float(max(1, hybrid_kv_elems))
        )
        decision = "FIT_WITHIN_BUDGET" if selected_scheme is not None else "EXCEEDS_BUDGET"

        result: Dict[str, Any] = {
            "seq_len": t,
            "batch_size": b,
            "n_layers": layers,
            "n_global_layers": n_global,
            "n_swa_layers": n_swa,
            "use_mla": bool(use_mla),
            "memory_budget_mb": round(float(memory_budget_mb), 4),
            "hybrid_vs_dense_reduction_factor": round(hybrid_vs_dense_factor, 4),
            "recurrent_state_mb": round(recurrent_mb, 4),
            "recommended_scheme": selected_scheme,
            "decision": decision,
            "schemes": schemes_report,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("KVCacheCompressionPlanner", result)
        return result


# ---------------------------------------------------------------------------
# 3. RooflineComputeProfiler
# ---------------------------------------------------------------------------


class RooflineComputeProfiler:
    """Analytical hardware Roofline profiler for MiMoMix prefill & decode."""

    def profile(
        self,
        *,
        hidden_size: int = 512,
        n_layers: int = 8,
        intermediate_size: int = 1024,
        moe_intermediate_size: int = 512,
        moe_top_k: int = 2,
        n_shared_experts: int = 1,
        use_moe: bool = True,
        vocab_size: int = 9460,
        seq_len: int = 256,
        batch_size: int = 1,
        weight_bytes_per_param: float = 2.0,
        peak_gflops: float = 400.0,
        memory_bandwidth_gbs: float = 50.0,
    ) -> Dict[str, Any]:
        h = int(hidden_size)
        layers = int(n_layers)
        t = int(seq_len)
        b = int(batch_size)
        if h < 8 or layers < 1 or t < 1 or b < 1:
            raise ValueError("hidden_size >= 8 and n_layers/seq_len/batch_size >= 1 required")
        if float(peak_gflops) <= 0.0 or float(memory_bandwidth_gbs) <= 0.0 or float(weight_bytes_per_param) <= 0.0:
            raise ValueError("hardware peak_gflops, memory_bandwidth_gbs, and weight_bytes_per_param must be > 0")

        # Active parameters per token across attention + FFN/MoE + lm_head
        attn_params_per_layer = 4 * h * h
        if use_moe:
            mlp_active_per_layer = 3 * h * int(moe_intermediate_size) * (int(moe_top_k) + int(n_shared_experts))
        else:
            mlp_active_per_layer = 3 * h * int(intermediate_size)
        active_params = layers * (attn_params_per_layer + mlp_active_per_layer) + h * int(vocab_size)

        # Prefill: processes B*T tokens at once -> weights read once, 2*active_params*B*T + attention FLOPs
        prefill_flops = 2.0 * active_params * b * t + 4.0 * layers * b * (t * t) * h
        prefill_bytes = active_params * float(weight_bytes_per_param) + 2.0 * layers * b * t * h * float(weight_bytes_per_param)
        prefill_ai = prefill_flops / max(1.0, prefill_bytes)

        # Decode (T=1 step at context length t): weights read once for B tokens + KV cache read
        decode_flops = 2.0 * active_params * b + 4.0 * layers * b * t * h
        decode_bytes = active_params * float(weight_bytes_per_param) + 2.0 * layers * b * t * h * float(weight_bytes_per_param)
        decode_ai = decode_flops / max(1.0, decode_bytes)

        ridge_point = float(peak_gflops) / float(memory_bandwidth_gbs)  # FLOPs/byte

        def _ceiling_tps(flops: float, nbytes: float, tokens: int) -> Tuple[float, str, float]:
            ai = flops / max(1.0, nbytes)
            regime = "compute_bound" if ai >= ridge_point else "memory_bandwidth_bound"
            achievable_gflops = min(float(peak_gflops), ai * float(memory_bandwidth_gbs))
            time_sec = (flops / 1e9) / max(1e-9, achievable_gflops)
            return round(tokens / max(1e-9, time_sec), 3), regime, round(achievable_gflops, 3)

        prefill_tps, prefill_regime, prefill_gflops = _ceiling_tps(prefill_flops, prefill_bytes, b * t)
        decode_tps, decode_regime, decode_gflops = _ceiling_tps(decode_flops, decode_bytes, b)

        bottleneck_hint = (
            "Decode is memory-bandwidth bound: prioritize int8/int4 weights, MTP speculation, and SWA/MLA KV reduction."
            if decode_regime == "memory_bandwidth_bound"
            else "Decode is compute bound: prioritize MoE sparsity, fused GEMMs, and operator fusion."
        )

        result: Dict[str, Any] = {
            "active_params_per_token": int(active_params),
            "hardware_ridge_point_flops_per_byte": round(ridge_point, 4),
            "prefill": {
                "tokens": b * t,
                "gflops": round(prefill_flops / 1e9, 6),
                "memory_mb": round(prefill_bytes / (1024.0 * 1024.0), 4),
                "arithmetic_intensity": round(prefill_ai, 4),
                "regime": prefill_regime,
                "achievable_gflops": prefill_gflops,
                "ceiling_tokens_per_sec": prefill_tps,
            },
            "decode": {
                "tokens": b,
                "gflops": round(decode_flops / 1e9, 6),
                "memory_mb": round(decode_bytes / (1024.0 * 1024.0), 4),
                "arithmetic_intensity": round(decode_ai, 4),
                "regime": decode_regime,
                "achievable_gflops": decode_gflops,
                "ceiling_tokens_per_sec": decode_tps,
            },
            "optimization_guidance": bottleneck_hint,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("RooflineComputeProfiler", result)
        return result


# ---------------------------------------------------------------------------
# 4. CascadeEnsembleRouter
# ---------------------------------------------------------------------------


class CascadeEnsembleRouter:
    """Cost-quality optimal multi-tier model cascade threshold optimizer."""

    def optimize_cascade(
        self,
        *,
        calibration_records: Sequence[Mapping[str, Any]],
        fast_cost_ms: float = 18.0,
        deep_cost_ms: float = 180.0,
        max_avg_latency_ms: float = 75.0,
    ) -> Dict[str, Any]:
        if not calibration_records:
            raise ValueError("calibration_records must be non-empty")
        f_cost = float(fast_cost_ms)
        d_cost = float(deep_cost_ms)
        budget = float(max_avg_latency_ms)
        if f_cost <= 0.0 or d_cost <= 0.0 or budget <= 0.0:
            raise ValueError("fast_cost_ms, deep_cost_ms, and max_avg_latency_ms must be > 0")

        parsed: List[Dict[str, Any]] = []
        for rec in calibration_records:
            conf = float(rec.get("fast_confidence", rec.get("confidence", 0.5)))
            if not math.isfinite(conf):
                raise ValueError("fast_confidence must be finite")
            parsed.append(
                {
                    "conf": max(0.0, min(1.0, conf)),
                    "fast_ok": 1.0 if bool(rec.get("fast_correct", False)) else 0.0,
                    "deep_ok": 1.0 if bool(rec.get("deep_correct", False)) else 0.0,
                }
            )

        n = float(len(parsed))
        fast_only_acc = sum(r["fast_ok"] for r in parsed) / n
        deep_only_acc = sum(r["deep_ok"] for r in parsed) / n

        # Candidate escalation thresholds: escalate to deep iff conf < tau
        thresholds = sorted({0.0, 1.01, *(r["conf"] + 1e-6 for r in parsed), *(r["conf"] for r in parsed)})
        feasible: List[Dict[str, Any]] = []
        all_points: List[Dict[str, Any]] = []
        for tau in thresholds:
            escalated = 0
            correct = 0.0
            for r in parsed:
                if r["conf"] < tau:
                    escalated += 1
                    correct += r["deep_ok"]
                else:
                    correct += r["fast_ok"]
            esc_rate = escalated / n
            avg_lat = f_cost + esc_rate * d_cost
            acc = correct / n
            pt = {
                "threshold": round(min(1.01, tau), 6),
                "accuracy": round(acc, 6),
                "escalation_rate": round(esc_rate, 6),
                "avg_latency_ms": round(avg_lat, 4),
                "fits_budget": avg_lat <= budget + 1e-9,
            }
            all_points.append(pt)
            if pt["fits_budget"]:
                feasible.append(pt)

        if feasible:
            best = max(feasible, key=lambda p: (p["accuracy"], -p["avg_latency_ms"]))
            decision = "OPTIMAL_CASCADE_FOUND"
        else:
            best = min(all_points, key=lambda p: (p["avg_latency_ms"], -p["accuracy"]))
            decision = "BUDGET_BELOW_FAST_FLOOR"

        latency_saved_vs_deep = 1.0 - (best["avg_latency_ms"] / max(1e-9, f_cost + d_cost))
        result: Dict[str, Any] = {
            "calibration_count": len(parsed),
            "fast_only_accuracy": round(fast_only_acc, 6),
            "deep_only_accuracy": round(deep_only_acc, 6),
            "max_avg_latency_ms": round(budget, 4),
            "optimal_threshold": best["threshold"],
            "cascade_accuracy": best["accuracy"],
            "accuracy_gain_over_fast": round(best["accuracy"] - fast_only_acc, 6),
            "escalation_rate": best["escalation_rate"],
            "avg_latency_ms": best["avg_latency_ms"],
            "latency_saved_vs_always_deep": round(latency_saved_vs_deep, 6),
            "decision": decision,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("CascadeEnsembleRouter", result)
        return result


# ---------------------------------------------------------------------------
# 5. SelfHealingBenchmarkGuard
# ---------------------------------------------------------------------------


class SelfHealingBenchmarkGuard:
    """Paired statistical regression & speedup guard for optimization releases."""

    @staticmethod
    def _percentile(sorted_vals: Sequence[float], q: float) -> float:
        idx = int(round(q * (len(sorted_vals) - 1)))
        return float(sorted_vals[max(0, min(len(sorted_vals) - 1, idx))])

    def evaluate(
        self,
        *,
        baseline_latencies_ms: Sequence[float],
        candidate_latencies_ms: Sequence[float],
        baseline_scores: Optional[Sequence[float]] = None,
        candidate_scores: Optional[Sequence[float]] = None,
        max_quality_drop: float = 0.005,
        max_latency_regression_pct: float = 5.0,
    ) -> Dict[str, Any]:
        b_lat = [float(x) for x in baseline_latencies_ms]
        c_lat = [float(x) for x in candidate_latencies_ms]
        if len(b_lat) < 2 or len(b_lat) != len(c_lat):
            raise ValueError("baseline_latencies_ms and candidate_latencies_ms must have equal length >= 2")
        if any(not math.isfinite(x) or x <= 0.0 for x in (*b_lat, *c_lat)):
            raise ValueError("latencies must be finite and > 0")

        n = len(b_lat)
        mean_b = sum(b_lat) / float(n)
        mean_c = sum(c_lat) / float(n)
        speedup = mean_b / max(1e-9, mean_c)
        lat_change_pct = ((mean_c - mean_b) / max(1e-9, mean_b)) * 100.0

        # Paired latency savings: positive means candidate is faster
        diffs = [b - c for b, c in zip(b_lat, c_lat)]
        mean_diff = sum(diffs) / float(n)
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / float(max(1, n - 1))
        sd_diff = math.sqrt(max(0.0, var_diff))
        se_diff = sd_diff / math.sqrt(float(n))
        ci_low = mean_diff - 1.96 * se_diff
        ci_high = mean_diff + 1.96 * se_diff
        cohens_dz = mean_diff / sd_diff if sd_diff > 1e-12 else (float("inf") if mean_diff > 0 else 0.0)
        if not math.isfinite(cohens_dz):
            cohens_dz = 10.0 if mean_diff > 0 else -10.0

        p95_b = self._percentile(sorted(b_lat), 0.95)
        p95_c = self._percentile(sorted(c_lat), 0.95)

        quality_delta = 0.0
        quality_regressed = False
        if baseline_scores is not None or candidate_scores is not None:
            if baseline_scores is None or candidate_scores is None:
                raise ValueError("both baseline_scores and candidate_scores must be provided together")
            b_sc = [float(x) for x in baseline_scores]
            c_sc = [float(x) for x in candidate_scores]
            if len(b_sc) != len(c_sc) or len(b_sc) < 1:
                raise ValueError("baseline_scores and candidate_scores must have equal non-empty length")
            quality_delta = (sum(c_sc) - sum(b_sc)) / float(len(b_sc))
            quality_regressed = quality_delta < -abs(float(max_quality_drop))

        latency_regressed = lat_change_pct > float(max_latency_regression_pct)
        if quality_regressed:
            decision = "HOLD_QUALITY_REGRESSION"
        elif latency_regressed:
            decision = "HOLD_LATENCY_REGRESSION"
        else:
            decision = "APPROVE_OPTIMIZATION"

        result: Dict[str, Any] = {
            "sample_count": n,
            "baseline_mean_ms": round(mean_b, 4),
            "candidate_mean_ms": round(mean_c, 4),
            "baseline_p95_ms": round(p95_b, 4),
            "candidate_p95_ms": round(p95_c, 4),
            "speedup_factor": round(speedup, 4),
            "latency_change_pct": round(lat_change_pct, 4),
            "paired_savings_ms_ci95": [round(ci_low, 4), round(ci_high, 4)],
            "cohens_dz_speedup": round(cohens_dz, 4),
            "quality_delta": round(quality_delta, 6),
            "quality_regressed": quality_regressed,
            "latency_regressed": latency_regressed,
            "decision": decision,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("SelfHealingBenchmarkGuard", result)
        return result
