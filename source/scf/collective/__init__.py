"""Multi-model collaboration, Riemannian geometry, Faultline selection, and fusion."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFCollectiveOrchestrator"]


class SCFCollectiveOrchestrator:
    """Generates, critiques, and fuses independent candidate solutions with geometry telemetry."""

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

        answers = [
            str(c.get("answer") or "").strip()
            for c in candidates
            if str(c.get("answer") or "").strip()
        ]
        critiqued: List[Dict[str, Any]] = []
        for cand in candidates:
            ans = str(cand.get("answer") or "").strip()
            agree_count = sum(1 for a in answers if a == ans and ans != "")
            consensus_bonus = 0.15 * max(0, agree_count - 1)
            verified_bonus = 0.35 if bool(cand.get("verified")) else 0.0
            conf = max(0.0, min(1.0, float(cand.get("confidence", 0.0))))
            nonempty_bonus = 0.10 if ans else -0.20
            total_score = round(conf + consensus_bonus + verified_bonus + nonempty_bonus, 4)
            critiqued.append(
                {
                    **cand,
                    "consensus_agreement": agree_count,
                    "composite_score": total_score,
                    "critique_notes": (
                        "verified_exact"
                        if cand.get("verified")
                        else (
                            "cross_model_consensus"
                            if agree_count > 1
                            else "unverified_advisory"
                        )
                    ),
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
            import nexus_v95_ecosystem as v95

            tether = v95.TetherForensicsEngine()
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
            if failure_matrix is not None:
                ensemble_forensics["faultline"] = v95.FaultlineEnsembleEngine().select_panel(
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
