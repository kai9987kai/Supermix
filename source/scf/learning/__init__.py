"""Controlled self-improvement and regression-gated learning pipelines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFLearningGate"]


class SCFLearningGate:
    """Wraps promotion gates, v94/v95 transactional neurogenesis, and curriculum readiness.

    Enforces Engineering Rule 9: never automatically promote a model without
    explicitly passing all regression and evaluation gates.
    """

    automatic_promotion_enabled: bool = False

    def inspect_readiness(self) -> Dict[str, Any]:
        import prepare_v94 as pv94

        v95_ready = pv94.check_v95_readiness()
        return {
            "automatic_promotion_enabled": False,
            "v95_readiness": v95_ready,
            "promotion_gate_modules": [
                "source/qwen_adapter_promotion.py",
                "source/run_v56_promotion_gate.py",
                "source/run_qwen_general_promotion_gate.py",
                "source/v94_growth_policy.py",
                "source/v95_neurogenesis_adapter.py",
            ],
        }

    def evaluate_promotion_request(
        self,
        *,
        baseline_accuracy: float,
        candidate_accuracy: float,
        regression_passed: bool,
        certified_eval_count: int,
    ) -> Dict[str, Any]:
        """Advisory evaluation check — never mutates active model pointers automatically."""

        delta = float(candidate_accuracy) - float(baseline_accuracy)
        eligible = bool(regression_passed) and int(certified_eval_count) > 0 and delta >= 0.0
        return {
            "automatic_promotion_enabled": False,
            "promoted": False,
            "eligible_for_manual_promotion": eligible,
            "accuracy_delta": round(delta, 6),
            "regression_passed": bool(regression_passed),
            "certified_eval_count": int(certified_eval_count),
            "reason": (
                "manual_approval_required_gates_passed"
                if eligible
                else "promotion_blocked_by_regression_or_insufficient_evidence"
            ),
        }

    def benchmark_checkpoint_pair(
        self,
        baseline_checkpoint: Path,
        candidate_checkpoint: Path,
        *,
        n_problems: int = 41,
        seed: int = 82,
        max_new_tokens: int = 112,
        tasks: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Paired benchmark evaluation between baseline and candidate checkpoints."""
        import eval_problem_solving as eps

        base_path = Path(baseline_checkpoint)
        cand_path = Path(candidate_checkpoint)
        if not base_path.is_file() or not cand_path.is_file():
            return {
                "completed": False,
                "reason": "checkpoint_missing",
                "baseline_exists": base_path.is_file(),
                "candidate_exists": cand_path.is_file(),
                "automatic_promotion_enabled": False,
                "promoted": False,
            }

        problems = list(eps.generate_novel(int(n_problems), seed=int(seed), tasks=tasks))
        base_tx: list = []
        cand_tx: list = []
        base_rep = eps.evaluate(
            base_path, problems, max_new_tokens=int(max_new_tokens), transcript=base_tx
        )
        cand_rep = eps.evaluate(
            cand_path, problems, max_new_tokens=int(max_new_tokens), transcript=cand_tx
        )
        base_acc = float(base_rep.get("overall", {}).get("accuracy", 0.0))
        cand_acc = float(cand_rep.get("overall", {}).get("accuracy", 0.0))
        cand_only = sum(1 for b, c in zip(base_tx, cand_tx) if not b.get("correct") and c.get("correct"))
        base_only = sum(1 for b, c in zip(base_tx, cand_tx) if b.get("correct") and not c.get("correct"))
        promo = self.evaluate_promotion_request(
            baseline_accuracy=base_acc,
            candidate_accuracy=cand_acc,
            regression_passed=cand_only >= base_only,
            certified_eval_count=len(problems),
        )
        return {
            "completed": True,
            "n_problems": len(problems),
            "baseline_checkpoint": str(base_path),
            "candidate_checkpoint": str(cand_path),
            "baseline_accuracy": base_acc,
            "candidate_accuracy": cand_acc,
            "accuracy_delta": round(cand_acc - base_acc, 6),
            "paired_discordant": {"candidate_only": cand_only, "baseline_only": base_only},
            "promotion_gate": promo,
            "automatic_promotion_enabled": False,
            "promoted": False,
        }
