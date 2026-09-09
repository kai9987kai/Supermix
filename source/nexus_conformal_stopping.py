"""Experimental margin-based stopping advice without a risk certificate.

Unlabeled margins do not identify error rates. This compatibility controller
provides a heuristic threshold and budget-stop advice only. Conformal risk
control would require a specified loss, suitable held-out calibration outcomes,
and validated assumptions; none are supplied by this interface.
See https://arxiv.org/abs/2208.02814 (Conformal Risk Control).
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
    certified_risk_bound: Optional[float]
    compute_savings_pct: float
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConformalStoppingController:
    """Compatibility API for uncalibrated, advisory compute stopping."""

    def __init__(
        self,
        target_risk_alpha: float = 0.05,
        default_calibrated_threshold: float = 0.35,
    ):
        self.target_risk_alpha = self.validate_risk(target_risk_alpha)
        self.calibrated_threshold = self.validate_probability(default_calibrated_threshold, "threshold")
        self.calibration_samples: List[float] = []

    @staticmethod
    def validate_probability(value: float, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")
        return numeric

    @classmethod
    def validate_risk(cls, value: float) -> float:
        risk = cls.validate_probability(value, "target_risk_alpha")
        if not 0.0 < risk < 1.0:
            raise ValueError("target_risk_alpha must be strictly between 0 and 1")
        return risk

    def calibrate(self, historical_margins: List[float]) -> float:
        """Fit an empirical margin quantile; this does not calibrate error risk."""
        if not historical_margins:
            return self.calibrated_threshold
        # Validate the entire input before replacing any prior state.
        margins = sorted(self.validate_probability(value, "historical margin") for value in historical_margins)
        index = max(0, math.ceil(len(margins) * (1.0 - self.target_risk_alpha)) - 1)
        self.calibrated_threshold = margins[index]
        self.calibration_samples = margins
        return self.calibrated_threshold

    def evaluate_stopping(
        self,
        query: str,
        current_step: int,
        max_budget: int = 6,
        top_confidence: float = 0.92,
        runner_up_confidence: float = 0.45,
        target_risk_alpha: Optional[float] = None,
    ) -> ConformalStoppingResult:
        """Return heuristic stop advice; budget exhaustion is never certification."""
        if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < 1:
            raise ValueError("current_step must be a positive integer")
        if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget < current_step:
            raise ValueError("max_budget must be an integer at least current_step")
        risk = self.target_risk_alpha if target_risk_alpha is None else self.validate_risk(target_risk_alpha)
        top = self.validate_probability(top_confidence, "top_confidence")
        runner_up = self.validate_probability(runner_up_confidence, "runner_up_confidence")
        margin = top - runner_up
        terminal = current_step >= max_budget
        margin_satisfied = margin >= self.calibrated_threshold
        should_exit = margin_satisfied or terminal
        savings = round((max_budget - current_step) / max_budget * 100.0, 1) if should_exit else 0.0
        if terminal:
            reason = "budget_exhausted"
            summary = f"Budget exhausted at step {current_step}/{max_budget}; no output correctness or risk bound established."
        elif margin_satisfied:
            reason = "heuristic_margin"
            summary = (
                f"Heuristic stop advice: margin {margin:.3f} meets threshold {self.calibrated_threshold:.3f}. "
                f"Potential step-budget savings {savings:.1f}%; error risk is uncalibrated."
            )
        else:
            reason = "continue"
            summary = (
                f"Continue compute: margin {margin:.3f} is below heuristic threshold {self.calibrated_threshold:.3f}. "
                "Error risk is uncalibrated."
            )
        return ConformalStoppingResult(
            query=query,
            current_step=current_step,
            max_budget=max_budget,
            observed_margin=margin,
            calibrated_threshold=self.calibrated_threshold,
            should_early_exit=should_exit,
            certified_risk_bound=None,
            compute_savings_pct=savings,
            diagnostic_summary=summary,
            telemetry={
                "target_risk_alpha": risk,
                "top_confidence": top,
                "runner_up_confidence": runner_up,
                "calibration_pool_size": len(self.calibration_samples),
                "calibration_status": "unlabeled_margin_quantile" if self.calibration_samples else "uncalibrated",
                "risk_certified": False,
                "answer_authority": False,
                "stop_reason": reason,
                "savings_measure": "potential_step_budget_fraction_not_measured_flops",
            },
        )
