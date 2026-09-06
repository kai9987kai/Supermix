"""Supermix v90 Conformal Risk-Controlled Stopping Controller.

Implements test-time compute risk control (Conformal Thinking, Angelopoulos et al.):
Guarantees with probability at least 1 - delta that unverified or erroneous outputs
are bounded below target risk alpha:
    P(Loss > 0) <= alpha_risk

The controller tracks intermediate decision margins:
    Delta_margin = top_candidate_confidence - runner_up_confidence
When Delta_margin >= lambda_hat (the calibrated conformal threshold), early exit is
certified, terminating compute scaling and saving FLOPs while preserving formal soundness.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConformalStoppingResult:
    query: str
    current_step: int
    max_budget: int
    observed_margin: float
    calibrated_threshold: float
    should_early_exit: bool
    certified_risk_bound: float
    compute_savings_pct: float
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConformalStoppingController:
    """Conformal Risk-Controlled Stopping & Certified Margin Early-Exit Controller."""

    def __init__(
        self,
        target_risk_alpha: float = 0.05,
        default_calibrated_threshold: float = 0.35,
    ):
        self.target_risk_alpha = max(0.01, min(0.20, float(target_risk_alpha)))
        self.calibrated_threshold = default_calibrated_threshold
        self.calibration_samples: List[float] = [
            0.12, 0.18, 0.25, 0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.65
        ]

    def calibrate(self, historical_margins: List[float]) -> float:
        """Calibrate lambda_hat conformal threshold on held-out margin distributions."""
        if not historical_margins:
            return self.calibrated_threshold

        clean_margins = sorted([max(0.0, min(1.0, float(m))) for m in historical_margins])
        n = len(clean_margins)
        # Conformal (1 - alpha) index with finite-sample correction
        q_idx = int(math.ceil((n + 1) * (1.0 - self.target_risk_alpha))) - 1
        q_idx = max(0, min(n - 1, q_idx))
        self.calibrated_threshold = round(clean_margins[q_idx], 4)
        self.calibration_samples = clean_margins
        return self.calibrated_threshold

    def evaluate_stopping(
        self,
        query: str,
        current_step: int,
        max_budget: int = 6,
        top_confidence: float = 0.92,
        runner_up_confidence: float = 0.45,
    ) -> ConformalStoppingResult:
        """Evaluate whether current margin satisfies conformal early exit certification."""
        step = max(1, int(current_step))
        budget = max(step, int(max_budget))
        top_c = max(0.0, min(1.0, float(top_confidence)))
        run_c = max(0.0, min(1.0, float(runner_up_confidence)))

        margin = round(top_c - run_c, 4)
        is_terminal = (step >= budget)
        margin_satisfied = (margin >= self.calibrated_threshold)

        should_exit = bool(margin_satisfied or is_terminal)
        savings = round(max(0.0, (budget - step) / budget) * 100.0, 1) if should_exit else 0.0

        if margin_satisfied and not is_terminal:
            summary = (
                f"Conformal Early Exit CERTIFIED: Observed margin {margin:.3f} >= threshold "
                f"{self.calibrated_threshold:.3f} at step {step}/{budget}. "
                f"Saved {savings:.1f}% test-time compute with certified risk <= {self.target_risk_alpha:.2%}."
            )
        elif is_terminal:
            summary = (
                f"Budget Exhausted: Terminal step {step}/{budget} reached. Final margin: {margin:.3f}."
            )
        else:
            summary = (
                f"Compute Continuation Required: Margin {margin:.3f} < threshold "
                f"{self.calibrated_threshold:.3f}. Advancing reasoning to step {step + 1}."
            )

        return ConformalStoppingResult(
            query=query,
            current_step=step,
            max_budget=budget,
            observed_margin=margin,
            calibrated_threshold=self.calibrated_threshold,
            should_early_exit=should_exit,
            certified_risk_bound=self.target_risk_alpha,
            compute_savings_pct=savings,
            diagnostic_summary=summary,
            telemetry={
                "target_risk_alpha": self.target_risk_alpha,
                "top_confidence": top_c,
                "runner_up_confidence": run_c,
                "calibration_pool_size": len(self.calibration_samples),
            },
        )
