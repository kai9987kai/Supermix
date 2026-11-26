"""Multi-model collaboration, Riemannian geometry, Faultline selection, and fusion."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFCollectiveOrchestrator"]

_STEP_ARITH_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*([+\-xX*/^])\s*(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)"
)
_STEP_PCT_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
    re.I,
)


class SCFCollectiveOrchestrator:
    """Generates, critiques, and fuses independent candidate solutions with geometry telemetry."""

    def __init__(self) -> None:
        self._tether_engine: Optional[Any] = None
        self._faultline_engine: Optional[Any] = None

    def gather_and_synthesize(
        self,
        prompt: str,
        *,
        specialist_families: Sequence[str],
        candidate_generators: Optional[Mapping[str, Callable[[str], Dict[str, Any]]]] = None,
        shared_context: Optional[Dict[str, Any]] = None,
        max_candidates: int = 3,
        expert_probs: Optional[Sequence[Sequence[float]]] = None,
        failure_matrix: Optional[Sequence[Sequence[float]]] = None,
    ) -> Dict[str, Any]:
        """Collect independent candidate outputs and rank by confidence, verification & consensus."""

        generators = dict(candidate_generators or {})
        candidates: List[Dict[str, Any]] = []
        ordered_families = list(
            dict.fromkeys([*list(generators.keys()), *list(specialist_families)])
        )[: max(1, int(max_candidates))]
        for family in ordered_families:
            fn = generators.get(family)
            if fn is not None:
                raw = dict(fn(prompt))
            else:
                raw = {
                    "family": family,
                    "answer": "",
                    "confidence": 0.45,
                    "verified": False,
                    "rationale": f"Advisory draft context from {family} (weights optional)",
                }
            raw.setdefault("family", family)
            raw.setdefault("confidence", 0.5)
            raw.setdefault("verified", False)
            candidates.append(raw)

        def _canonical_value(raw_ans: str) -> str:
            text = str(raw_ans or "").strip()
            if not text:
                return ""
            try:
                import eval_problem_solving as ep

                num = ep.extract_answer(text)
                if num is not None:
                    return f"num:{float(num):.8g}"
            except Exception:
                pass
            return " ".join(text.lower().split())

        def _audit_trace_arithmetic(raw_ans: str) -> Dict[str, Any]:
            """Audit inline arithmetic, power, and percentage steps inside a candidate trace."""
            text = str(raw_ans or "")
            checked = 0
            bad_steps = 0
            failures: List[str] = []

            # 1. Binary arithmetic steps: a op b = c
            for a_s, op_s, b_s, c_s in _STEP_ARITH_RE.findall(text):
                try:
                    a_v, b_v, c_v = float(a_s), float(b_s), float(c_s)
                except ValueError:
                    continue
                if op_s == "+":
                    exp_v = a_v + b_v
                elif op_s == "-":
                    exp_v = a_v - b_v
                elif op_s in ("*", "x", "X"):
                    exp_v = a_v * b_v
                elif op_s == "/":
                    if b_v == 0.0:
                        checked += 1
                        bad_steps += 1
                        failures.append(f"{a_s}/{b_s}={c_s} (div0)")
                        continue
                    exp_v = a_v / b_v
                elif op_s == "^":
                    if abs(b_v) > 32:
                        continue
                    exp_v = a_v ** b_v
                else:
                    continue
                checked += 1
                if abs(exp_v - c_v) > max(1e-4, 1e-4 * abs(exp_v)):
                    bad_steps += 1
                    failures.append(f"{a_s} {op_s} {b_s} = {c_s} (expected {exp_v:.6g})")

            # 2. Inline percentage steps: p% of b = c
            for p_s, b_s, c_s in _STEP_PCT_RE.findall(text):
                try:
                    exp_v = (float(p_s) * float(b_s)) / 100.0
                    c_v = float(c_s)
                except ValueError:
                    continue
                checked += 1
                if abs(exp_v - c_v) > max(1e-4, 1e-4 * abs(exp_v)):
                    bad_steps += 1
                    failures.append(f"{p_s}% of {b_s} = {c_s} (expected {exp_v:.6g})")

            fidelity = 1.0 if checked == 0 else max(0.0, float(checked - bad_steps) / float(checked))
            return {
                "steps_checked": checked,
                "failures": bad_steps,
                "failure_details": failures[:4],
                "process_fidelity_score": round(fidelity, 4),
            }

        answers = [
            _canonical_value(str(c.get("answer") or ""))
            for c in candidates
            if str(c.get("answer") or "").strip()
        ]
        critiqued: List[Dict[str, Any]] = []
        for cand in candidates:
            ans = str(cand.get("answer") or "").strip()
            canon = _canonical_value(ans)
            agree_count = sum(1 for a in answers if a == canon and canon != "")
            consensus_bonus = 0.15 * max(0, agree_count - 1)
            verified_bonus = 0.35 if bool(cand.get("verified")) else 0.0
            conf = max(0.0, min(1.0, float(cand.get("confidence", 0.0))))
            nonempty_bonus = 0.10 if ans else -0.20
            if cand.get("verified"):
                audit = {"steps_checked": 0, "failures": 0, "failure_details": [], "process_fidelity_score": 1.0}
            else:
                audit = _audit_trace_arithmetic(ans)
            trace_errors = int(audit["failures"])
            trace_penalty = min(0.24, 0.12 * trace_errors)
            total_score = round(conf + consensus_bonus + verified_bonus + nonempty_bonus - trace_penalty, 4)
            note = (
                "verified_exact"
                if cand.get("verified")
                else (
                    "trace_arithmetic_inconsistency"
                    if trace_errors > 0
                    else (
                        "cross_model_consensus"
                        if agree_count > 1
                        else "unverified_advisory"
                    )
                )
            )
            critiqued.append(
                {
                    **cand,
                    "canonical_answer": canon,
                    "consensus_agreement": agree_count,
                    "trace_audit_failures": trace_errors,
                    "trace_steps_checked": int(audit["steps_checked"]),
                    "process_fidelity_score": float(audit["process_fidelity_score"]),
                    "trace_failure_details": list(audit["failure_details"]),
                    "composite_score": total_score,
                    "critique_notes": note,
                }
            )

        critiqued.sort(
            key=lambda item: (-item["composite_score"], -float(item.get("confidence", 0.0)))
        )
        best = (
            critiqued[0]
            if critiqued
            else {"family": "none", "answer": "", "composite_score": 0.0}
        )

        # Torch score fusion summary using existing score_fusion helpers
        fusion_mode = "consensus_verified_priority"
        try:
            import score_fusion as sf
            import torch

            if len(critiqued) >= 2:
                signals = {
                    c["family"]: torch.tensor(
                        [float(c["composite_score"]), max(0.0, 1.0 - float(c["composite_score"]))]
                    )
                    for c in critiqued
                }
                weights = {c["family"]: 1.0 / float(len(critiqued)) for c in critiqued}
                _, diag = sf.fuse_signals(
                    signals, weights, mode="consensus", allow_experimental=True
                )
                fusion_mode = str(diag.get("mode", fusion_mode))
        except Exception:
            pass

        # Riemannian Fisher-Rao geometry (TetherForensicsEngine) & Faultline panel diversity
        ensemble_forensics: Dict[str, Any] = {"answer_authority": False}
        try:
            if self._tether_engine is None:
                import nexus_v95_ecosystem as v95

                self._tether_engine = v95.TetherForensicsEngine()
                self._faultline_engine = v95.FaultlineEnsembleEngine()

            tether = self._tether_engine
            probs = (
                expert_probs
                if expert_probs is not None
                else [
                    [
                        max(0.05, min(0.95, float(c.get("confidence", 0.5)))),
                        max(0.05, 1.0 - max(0.05, min(0.95, float(c.get("confidence", 0.5))))),
                    ]
                    for c in critiqued
                ]
            )
            if probs:
                ensemble_forensics["tether"] = tether.analyze(probs)
            if failure_matrix is not None and self._faultline_engine is not None:
                ensemble_forensics["faultline"] = self._faultline_engine.select_panel(
                    failure_matrix
                )
        except Exception as exc:
            ensemble_forensics["error"] = f"{type(exc).__name__}: {exc}"

        return {
            "best_candidate": best,
            "candidates": critiqued,
            "fusion_mode": fusion_mode,
            "ensemble_forensics": ensemble_forensics,
            "shared_context_keys": sorted((shared_context or {}).keys()),
        }
