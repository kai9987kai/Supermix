"""Supermix v104 Frontier Epistemic, Mechanistic & Optimization Engines.

Five deterministic, zero-external-dependency diagnostic and synthesis engines
extending the v102/v103 frontier ecosystem:

1. ``ConformalCalibratedRouter`` — Distribution-free finite-sample split-conformal
   prediction sets over model families with exact quantile calibration, empirical
   coverage / singleton efficiency accounting, and fail-closed abstention.
2. ``MechanisticCircuitTracer`` — Causal path-patching & activation-steering
   attribution across layered transformer + memory + connectome circuits,
   measuring direct effect, mediated effect, mediation fraction, sufficiency,
   necessity, and minimal circuit sparsity.
3. ``EpisodicGraphConsolidator`` — Temporal knowledge-graph belief revision
   (AGM-style contradiction detection & supersession by evidence strength and
   recency), salience pruning, consistency scoring, and multi-hop path tracing.
4. ``VerifiableProcessRewardAuditor`` — Step-level Process Reward Model (PRM)
   auditing multi-step reasoning traces with zero-eval AST arithmetic checks,
   cross-step grounding continuity, weakest-link bottleneck detection, and
   geometric-mean chain soundness.
5. ``ParetoFrontierAutoTuner`` — Multi-objective Supermix configuration Pareto
   front extraction, normalized hypervolume computation, and deployment-profile
   recommendations (``edge_cpu``, ``balanced_studio``, ``max_rigor``).

Engineering contract:
- Every engine sets ``answer_authority=False`` and emits a canonical SHA-256
  receipt so diagnostics can never masquerade as closed-world truth proofs.
- Zero ``eval``/``exec``; pure Python standard library.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator as _op
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EVOLUTION_V104_VERSION = "supermix-v104-evolution-v1"
RECEIPT_SCHEMA_V104 = "supermix-v104-receipt-v1"

__all__ = [
    "EVOLUTION_V104_VERSION",
    "RECEIPT_SCHEMA_V104",
    "ConformalCalibratedRouter",
    "MechanisticCircuitTracer",
    "EpisodicGraphConsolidator",
    "VerifiableProcessRewardAuditor",
    "ParetoFrontierAutoTuner",
]


def _canonical_receipt(engine: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA_V104,
        "evolution_version": EVOLUTION_V104_VERSION,
        "engine": str(engine),
        "answer_authority": False,
        "payload": payload,
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return {
        "schema_version": RECEIPT_SCHEMA_V104,
        "evolution_version": EVOLUTION_V104_VERSION,
        "engine": str(engine),
        "answer_authority": False,
        "receipt_sha256": digest,
    }


# ---------------------------------------------------------------------------
# 1. ConformalCalibratedRouter
# ---------------------------------------------------------------------------


class ConformalCalibratedRouter:
    """Finite-sample distribution-free split-conformal routing engine.

    Given calibration nonconformity scores ``s_1, ..., s_n`` (higher = worse fit,
    e.g. ``1 - p(true_family | x)``) and target miscoverage ``alpha in (0, 1)``,
    computes the exact finite-sample split-conformal quantile:
        ``k = ceil((n + 1) * (1 - alpha))``
        ``q_hat = sorted_scores[k - 1]`` (or ``inf`` if ``k > n``)
    and forms prediction sets ``C(x) = {m : 1 - score(m) <= q_hat}`` with
    fail-closed abstention whenever ``|C(x)| == 0`` or ``|C(x)| > max_set_size``.
    """

    def calibrate_and_route(
        self,
        *,
        calibration_scores: Sequence[float],
        candidate_scores: Mapping[str, float],
        alpha: float = 0.10,
        max_set_size: int = 2,
    ) -> Dict[str, Any]:
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError("alpha must be strictly between 0 and 1")
        cal = sorted(float(s) for s in calibration_scores)
        if not cal:
            raise ValueError("calibration_scores must be non-empty")
        if any(not math.isfinite(s) or s < 0.0 for s in cal):
            raise ValueError("calibration_scores must be finite and non-negative")
        if not candidate_scores:
            raise ValueError("candidate_scores must be non-empty")
        if int(max_set_size) < 1:
            raise ValueError("max_set_size must be >= 1")

        n = len(cal)
        rank = math.ceil((n + 1) * (1.0 - float(alpha)))
        if rank <= 0:
            q_hat = cal[0]
            finite_q = True
        elif rank > n:
            q_hat = 1.0
            finite_q = False
        else:
            q_hat = cal[rank - 1]
            finite_q = True

        # Nonconformity of each candidate family = 1.0 - calibrated_confidence
        nonconformity: Dict[str, float] = {}
        for fam, conf in candidate_scores.items():
            c_val = float(conf)
            if not math.isfinite(c_val):
                raise ValueError(f"Non-finite confidence for candidate {fam!r}")
            nonconformity[str(fam)] = max(0.0, 1.0 - max(0.0, min(1.0, c_val)))

        admitted = sorted(
            [fam for fam, nc in nonconformity.items() if nc <= q_hat + 1e-12],
            key=lambda f: (nonconformity[f], f),
        )
        empirical_cal_coverage = sum(1 for s in cal if s <= q_hat + 1e-12) / float(n)
        singleton = len(admitted) == 1
        if len(admitted) == 0:
            decision = "ABSTAIN_EMPTY_SET"
            selected: Optional[str] = None
        elif len(admitted) > int(max_set_size):
            decision = "ABSTAIN_AMBIGUOUS_SET"
            selected = None
        else:
            decision = "ROUTE_SINGLETON" if singleton else "ROUTE_CONFORMAL_SET"
            selected = admitted[0]

        result: Dict[str, Any] = {
            "alpha": round(float(alpha), 6),
            "target_coverage": round(1.0 - float(alpha), 6),
            "calibration_n": n,
            "quantile_rank": rank,
            "q_hat": round(float(q_hat), 6),
            "finite_sample_bound_valid": finite_q,
            "empirical_cal_coverage": round(float(empirical_cal_coverage), 6),
            "candidate_nonconformity": {k: round(v, 6) for k, v in sorted(nonconformity.items())},
            "prediction_set": admitted,
            "set_size": len(admitted),
            "singleton_efficiency": singleton,
            "decision": decision,
            "selected_family": selected,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("ConformalCalibratedRouter", result)
        return result


# ---------------------------------------------------------------------------
# 2. MechanisticCircuitTracer
# ---------------------------------------------------------------------------


class MechanisticCircuitTracer:
    """Deterministic causal path-patching & activation-steering circuit tracer.

    Models a 4-stage Supermix residual computation graph:
        ``input -> engram_mlp -> attn_deltanet -> connectome_titans -> logit``
    and computes exact clean vs. corrupted logit difference, individual node
    direct/indirect patching recovery, mediation fraction, necessity (ablation
    drop), sufficiency (restoration recovery), and minimal circuit sparsity.
    """

    STAGE_NAMES: Tuple[str, ...] = (
        "engram_mlp",
        "attn_deltanet",
        "connectome_titans",
        "thinking_readout",
    )

    def trace_circuit(
        self,
        *,
        clean_activations: Mapping[str, float],
        corrupted_activations: Mapping[str, float],
        edge_weights: Optional[Mapping[str, float]] = None,
        sufficiency_threshold: float = 0.80,
    ) -> Dict[str, Any]:
        weights = {
            stage: float((edge_weights or {}).get(stage, 1.0))
            for stage in self.STAGE_NAMES
        }
        clean = {stage: float(clean_activations.get(stage, 0.0)) for stage in self.STAGE_NAMES}
        corrupt = {
            stage: float(corrupted_activations.get(stage, 0.0)) for stage in self.STAGE_NAMES
        }

        def _logit(acts: Mapping[str, float]) -> float:
            # Linear main effects + deterministic pairwise cascade interaction
            linear = sum(weights[s] * acts[s] for s in self.STAGE_NAMES)
            cascade = (
                0.25 * acts["engram_mlp"] * acts["attn_deltanet"]
                + 0.20 * acts["attn_deltanet"] * acts["connectome_titans"]
                + 0.15 * acts["connectome_titans"] * acts["thinking_readout"]
            )
            return linear + cascade

        clean_logit = _logit(clean)
        corrupt_logit = _logit(corrupt)
        total_effect = clean_logit - corrupt_logit
        denom = total_effect if abs(total_effect) > 1e-9 else 1.0

        node_reports: List[Dict[str, Any]] = []
        for stage in self.STAGE_NAMES:
            # Patch single stage from clean into corrupted (sufficiency / indirect recovery)
            patched = dict(corrupt)
            patched[stage] = clean[stage]
            restored_delta = _logit(patched) - corrupt_logit
            sufficiency = restored_delta / denom

            # Ablate single stage from clean to corrupted (necessity / ablation drop)
            ablated = dict(clean)
            ablated[stage] = corrupt[stage]
            ablation_drop = clean_logit - _logit(ablated)
            necessity = ablation_drop / denom

            node_reports.append(
                {
                    "stage": stage,
                    "clean_activation": round(clean[stage], 6),
                    "corrupted_activation": round(corrupt[stage], 6),
                    "restored_logit_delta": round(restored_delta, 6),
                    "ablation_logit_drop": round(ablation_drop, 6),
                    "sufficiency_ratio": round(sufficiency, 6),
                    "necessity_ratio": round(necessity, 6),
                }
            )

        # Greedy minimal circuit achieving >= sufficiency_threshold of total_effect
        ranked = sorted(node_reports, key=lambda r: abs(r["restored_logit_delta"]), reverse=True)
        minimal_circuit: List[str] = []
        curr = dict(corrupt)
        curr_recovery = 0.0
        for rep in ranked:
            minimal_circuit.append(rep["stage"])
            curr[rep["stage"]] = clean[rep["stage"]]
            curr_recovery = (_logit(curr) - corrupt_logit) / denom
            if curr_recovery >= float(sufficiency_threshold) - 1e-9:
                break

        dominant_stage = ranked[0]["stage"] if ranked else "none"
        sparsity = 1.0 - (len(minimal_circuit) / float(len(self.STAGE_NAMES)))

        result: Dict[str, Any] = {
            "clean_logit": round(clean_logit, 6),
            "corrupted_logit": round(corrupt_logit, 6),
            "total_causal_effect": round(total_effect, 6),
            "dominant_stage": dominant_stage,
            "minimal_circuit": minimal_circuit,
            "minimal_circuit_recovery": round(curr_recovery, 6),
            "circuit_sparsity": round(sparsity, 6),
            "sufficiency_threshold": round(float(sufficiency_threshold), 6),
            "node_attributions": node_reports,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("MechanisticCircuitTracer", result)
        return result


# ---------------------------------------------------------------------------
# 3. EpisodicGraphConsolidator
# ---------------------------------------------------------------------------


class EpisodicGraphConsolidator:
    """Temporal knowledge-graph belief-revision & multi-hop consolidation engine.

    Resolves functional-predicate contradictions ``(s, p, o1)`` vs ``(s, p, o2)``
    via AGM-inspired evidence-strength + recency scoring, prunes edges below
    ``min_salience``, computes contradiction-free consistency score, and finds
    shortest transitive inference paths between entities.
    """

    def consolidate(
        self,
        *,
        triples: Sequence[Mapping[str, Any]],
        min_salience: float = 0.25,
        query_source: Optional[str] = None,
        query_target: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not triples:
            raise ValueError("triples must be non-empty")

        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        pruned_count = 0
        for idx, raw in enumerate(triples):
            s = str(raw.get("subject", "")).strip()
            p = str(raw.get("predicate", "")).strip()
            o = str(raw.get("object", "")).strip()
            conf = float(raw.get("confidence", 0.5))
            ts = float(raw.get("timestamp", float(idx)))
            if not (s and p and o):
                continue
            if conf < float(min_salience):
                pruned_count += 1
                continue
            grouped.setdefault((s, p), []).append(
                {
                    "subject": s,
                    "predicate": p,
                    "object": o,
                    "confidence": max(0.0, min(1.0, conf)),
                    "timestamp": ts,
                    "source": str(raw.get("source", "episodic")),
                }
            )

        accepted: List[Dict[str, Any]] = []
        conflicts: List[Dict[str, Any]] = []
        for (s, p), candidates in sorted(grouped.items()):
            distinct_objs = {c["object"] for c in candidates}
            # Rank by weighted (confidence * 0.7 + normalized recency * 0.3)
            max_ts = max(c["timestamp"] for c in candidates)
            min_ts = min(c["timestamp"] for c in candidates)
            span = max(1e-6, max_ts - min_ts)
            ranked = sorted(
                candidates,
                key=lambda c: (
                    0.7 * c["confidence"] + 0.3 * ((c["timestamp"] - min_ts) / span),
                    c["timestamp"],
                    c["object"],
                ),
                reverse=True,
            )
            winner = ranked[0]
            accepted.append(winner)
            if len(distinct_objs) > 1:
                superseded = [c for c in ranked[1:] if c["object"] != winner["object"]]
                conflicts.append(
                    {
                        "subject": s,
                        "predicate": p,
                        "winner": winner["object"],
                        "superseded_objects": sorted({c["object"] for c in superseded}),
                        "resolution_rule": "agm_confidence_recency_dominance",
                    }
                )

        # Build adjacency for multi-hop path discovery
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for edge in accepted:
            adj.setdefault(edge["subject"], []).append((edge["predicate"], edge["object"]))

        path_found: Optional[List[Dict[str, str]]] = None
        if query_source and query_target:
            src = str(query_source).strip()
            dst = str(query_target).strip()
            queue: List[Tuple[str, List[Dict[str, str]]]] = [(src, [])]
            visited = {src}
            while queue:
                node, path = queue.pop(0)
                if node == dst and path:
                    path_found = path
                    break
                if len(path) >= 6:
                    continue
                for pred, nxt in adj.get(node, []):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append((nxt, path + [{"from": node, "predicate": pred, "to": nxt}]))

        total_groups = max(1, len(grouped))
        consistency_score = 1.0 - (len(conflicts) / float(total_groups))
        result: Dict[str, Any] = {
            "input_triple_count": len(triples),
            "pruned_low_salience": pruned_count,
            "consolidated_edges": accepted,
            "consolidated_count": len(accepted),
            "conflicts_resolved": conflicts,
            "conflict_count": len(conflicts),
            "pre_revision_consistency": round(consistency_score, 6),
            "post_revision_consistent": True,
            "query_path": path_found,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("EpisodicGraphConsolidator", result)
        return result


# ---------------------------------------------------------------------------
# 4. VerifiableProcessRewardAuditor
# ---------------------------------------------------------------------------


_SAFE_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
}
_SAFE_UNARY_OPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval_arith(expr: str) -> Optional[float]:
    cleaned = expr.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if not cleaned or len(cleaned) > 120:
        return None
    try:
        node = ast.parse(cleaned, mode="eval").body
    except SyntaxError:
        return None

    ops_budget = [64]

    def _walk(n: ast.AST) -> float:
        ops_budget[0] -= 1
        if ops_budget[0] < 0:
            raise ValueError("op budget exceeded")
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_UNARY_OPS:
            return float(_SAFE_UNARY_OPS[type(n.op)](_walk(n.operand)))
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_BIN_OPS:
            lhs = _walk(n.left)
            rhs = _walk(n.right)
            if isinstance(n.op, ast.Pow) and abs(rhs) > 20:
                raise ValueError("exponent too large")
            if isinstance(n.op, ast.Div) and abs(rhs) < 1e-12:
                raise ValueError("division by zero")
            return float(_SAFE_BIN_OPS[type(n.op)](lhs, rhs))
        raise ValueError("unsupported node")

    try:
        val = _walk(node)
        return val if math.isfinite(val) else None
    except Exception:
        return None


class VerifiableProcessRewardAuditor:
    """Step-level Process Reward Model (PRM) & proof-chain coherence auditor.

    For each step in a reasoning trace:
    - Extracts inline equations ``<lhs> = <rhs>`` and audits them via a bounded
      zero-``eval`` AST evaluator.
    - Measures token/number grounding continuity from the problem prompt and
      earlier verified steps.
    - Computes a per-step process reward in ``[0, 1]``, geometric-mean chain
      soundness, and pinpoints the weakest-link step.
    """

    _EQ_RE = re.compile(r"([0-9\.\+\-\*\/\^\(\)\s×÷]+?)\s*=\s*(-?\d+(?:\.\d+)?)")
    _NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def audit_trace(
        self,
        *,
        prompt: str,
        steps: Sequence[str],
    ) -> Dict[str, Any]:
        if not steps:
            raise ValueError("steps must be non-empty")

        known_tokens = set(re.findall(r"[a-zA-Z_]{3,}", str(prompt).lower()))
        known_nums = set(self._NUM_RE.findall(str(prompt)))

        step_reports: List[Dict[str, Any]] = []
        total_eqs = 0
        passed_eqs = 0
        failed_eqs = 0

        for idx, raw_step in enumerate(steps, start=1):
            text = str(raw_step or "").strip()
            step_eqs = list(self._EQ_RE.finditer(text))
            eq_checks: List[Dict[str, Any]] = []
            step_arith_ok = True
            for m in step_eqs:
                lhs_s, rhs_s = m.group(1).strip(), m.group(2).strip()
                if not any(op_c in lhs_s for op_c in "+-*/^×÷"):
                    continue
                expected = _safe_eval_arith(lhs_s)
                claimed = float(rhs_s)
                if expected is None:
                    continue
                total_eqs += 1
                ok = abs(expected - claimed) <= 1e-4 * max(1.0, abs(expected))
                if ok:
                    passed_eqs += 1
                else:
                    failed_eqs += 1
                    step_arith_ok = False
                eq_checks.append(
                    {
                        "expression": f"{lhs_s} = {rhs_s}",
                        "computed": round(expected, 6),
                        "claimed": round(claimed, 6),
                        "verified": ok,
                    }
                )

            step_words = set(re.findall(r"[a-zA-Z_]{3,}", text.lower()))
            step_nums = set(self._NUM_RE.findall(text))
            overlap_words = len(step_words & known_tokens)
            overlap_nums = len(step_nums & known_nums)
            grounding = 1.0 if (overlap_words + overlap_nums) > 0 else 0.35

            if not text:
                reward = 0.0
            elif not step_arith_ok:
                reward = 0.10
            elif eq_checks:
                reward = min(1.0, 0.75 + 0.25 * grounding)
            else:
                reward = min(0.95, 0.45 + 0.50 * grounding)

            known_tokens |= step_words
            known_nums |= step_nums
            step_reports.append(
                {
                    "step_index": idx,
                    "text": text,
                    "equations_checked": eq_checks,
                    "arithmetic_consistent": step_arith_ok,
                    "grounding_score": round(grounding, 4),
                    "step_reward": round(reward, 4),
                }
            )

        rewards = [max(1e-6, float(r["step_reward"])) for r in step_reports]
        geo_mean = math.exp(sum(math.log(r) for r in rewards) / float(len(rewards)))
        weakest = min(step_reports, key=lambda r: (r["step_reward"], r["step_index"]))

        result: Dict[str, Any] = {
            "step_count": len(step_reports),
            "equations_audited": total_eqs,
            "equations_passed": passed_eqs,
            "equations_failed": failed_eqs,
            "all_arithmetic_verified": failed_eqs == 0,
            "chain_soundness_geo_mean": round(geo_mean, 6),
            "weakest_link_step": weakest["step_index"],
            "weakest_link_reward": weakest["step_reward"],
            "step_reports": step_reports,
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("VerifiableProcessRewardAuditor", result)
        return result


# ---------------------------------------------------------------------------
# 5. ParetoFrontierAutoTuner
# ---------------------------------------------------------------------------


class ParetoFrontierAutoTuner:
    """Multi-objective Pareto front & deployment-profile recommender.

    Evaluates candidate Supermix configurations on:
    - ``quality_score`` (maximize, in ``[0, 1]``)
    - ``latency_ms`` (minimize, ``> 0``)
    - ``memory_mb`` (minimize, ``> 0``)
    - ``calibration_error`` (minimize, ``>= 0``)

    Extracts exact non-dominated Pareto points, computes a normalized 2D
    ``(quality, speed)`` hypervolume indicator, and selects optimal operating
    points for ``edge_cpu``, ``balanced_studio``, and ``max_rigor`` profiles.
    """

    @staticmethod
    def _dominates(a: Mapping[str, float], b: Mapping[str, float]) -> bool:
        better_or_eq = (
            a["quality_score"] >= b["quality_score"]
            and a["latency_ms"] <= b["latency_ms"]
            and a["memory_mb"] <= b["memory_mb"]
            and a["calibration_error"] <= b["calibration_error"]
        )
        strictly_better = (
            a["quality_score"] > b["quality_score"]
            or a["latency_ms"] < b["latency_ms"]
            or a["memory_mb"] < b["memory_mb"]
            or a["calibration_error"] < b["calibration_error"]
        )
        return better_or_eq and strictly_better

    def optimize(
        self,
        *,
        candidates: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if not candidates:
            raise ValueError("candidates must be non-empty")

        normed: List[Dict[str, Any]] = []
        for idx, raw in enumerate(candidates):
            name = str(raw.get("name") or f"config_{idx + 1}")
            q = max(0.0, min(1.0, float(raw.get("quality_score", 0.5))))
            lat = max(1e-3, float(raw.get("latency_ms", 100.0)))
            mem = max(1e-3, float(raw.get("memory_mb", 256.0)))
            ece = max(0.0, float(raw.get("calibration_error", 0.05)))
            normed.append(
                {
                    "name": name,
                    "quality_score": q,
                    "latency_ms": lat,
                    "memory_mb": mem,
                    "calibration_error": ece,
                }
            )

        pareto: List[Dict[str, Any]] = []
        dominated: List[str] = []
        for i, cand in enumerate(normed):
            is_dom = any(
                self._dominates(other, cand) for j, other in enumerate(normed) if i != j
            )
            if is_dom:
                dominated.append(cand["name"])
            else:
                pareto.append(cand)

        # 2D normalized hypervolume in (quality in [0,1], speed = 1 / (1 + lat/max_lat))
        max_lat = max(c["latency_ms"] for c in normed)
        pts = sorted(
            [(c["quality_score"], 1.0 - (c["latency_ms"] / (max_lat * 1.1))) for c in pareto],
            key=lambda xy: xy[0],
            reverse=True,
        )
        hv = 0.0
        best_y = 0.0
        for x_val, y_val in pts:
            if y_val > best_y:
                hv += x_val * (y_val - best_y)
                best_y = y_val

        # Profile recommendations from Pareto set
        edge_cpu = min(pareto, key=lambda c: (c["latency_ms"] + 0.25 * c["memory_mb"], -c["quality_score"]))
        max_rigor = max(pareto, key=lambda c: (c["quality_score"] - 0.5 * c["calibration_error"], -c["latency_ms"]))
        max_mem = max(c["memory_mb"] for c in pareto)
        balanced = max(
            pareto,
            key=lambda c: (
                0.50 * c["quality_score"]
                - 0.25 * (c["latency_ms"] / max_lat)
                - 0.15 * (c["memory_mb"] / max_mem)
                - 0.10 * c["calibration_error"]
            ),
        )

        result: Dict[str, Any] = {
            "candidate_count": len(normed),
            "pareto_front": pareto,
            "pareto_count": len(pareto),
            "dominated_configs": sorted(dominated),
            "hypervolume_2d_quality_speed": round(hv, 6),
            "recommendations": {
                "edge_cpu": edge_cpu["name"],
                "balanced_studio": balanced["name"],
                "max_rigor": max_rigor["name"],
            },
            "answer_authority": False,
        }
        result["receipt"] = _canonical_receipt("ParetoFrontierAutoTuner", result)
        return result
