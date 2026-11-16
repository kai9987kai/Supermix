"""Margin-based stopping advisor with optional Labeled Conformal Risk Control.

When used with unlabeled margins alone (the compatibility path), this controller
provides a heuristic margin threshold and step-budget stop advice with
``certified_risk_bound=None`` because unlabeled margins do not identify error
rates. When calibrated via ``calibrate_labeled(margins, losses)`` with bounded
held-out losses ``L_i in [0, b_max]``, it computes the finite-sample Monotone
Conformal Risk Control threshold (Angelopoulos et al., ICLR 2024,
https://arxiv.org/abs/2208.02814) and selective Risk-Coverage AURC / E-AURC
diagnostics (Geifman & El-Yaniv, NeurIPS 2017).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    """Stopping controller supporting unlabeled heuristics and labeled Conformal Risk Control."""

    def __init__(
        self,
        target_risk_alpha: float = 0.05,
        default_calibrated_threshold: float = 0.35,
    ):
        self.target_risk_alpha = self.validate_risk(target_risk_alpha)
        self.calibrated_threshold = self.validate_probability(default_calibrated_threshold, "threshold")
        self.calibration_samples: List[float] = []
        self._labeled_calibrated: bool = False
        self._labeled_pairs: List[Tuple[float, float]] = []
        self._labeled_b_max: float = 1.0
        self._labeled_alpha: float = self.target_risk_alpha
        self._empirical_crc_risk: Optional[float] = None
        self._rc_profile: Optional[Dict[str, Any]] = None

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

    def calibrate(self, historical_margins: Sequence[float]) -> float:
        """Fit an empirical margin quantile; this does not calibrate error risk."""
        if not historical_margins:
            return self.calibrated_threshold
        # Validate the entire input before replacing any prior state.
        margins = sorted(self.validate_probability(value, "historical margin") for value in historical_margins)
        index = max(0, math.ceil(len(margins) * (1.0 - self.target_risk_alpha)) - 1)
        self.calibrated_threshold = margins[index]
        self.calibration_samples = margins
        self._labeled_calibrated = False
        self._labeled_pairs = []
        self._empirical_crc_risk = None
        self._rc_profile = None
        return self.calibrated_threshold

    @staticmethod
    def _crc_threshold_for_alpha(
        pairs: Sequence[Tuple[float, float]],
        alpha: float,
        b_max: float,
    ) -> Tuple[float, float]:
        """Find smallest threshold lambda with finite-sample monotone CRC bound <= alpha."""
        n = len(pairs)
        candidates = sorted({round(m, 12) for m, _ in pairs})
        if 1.0 not in candidates:
            candidates.append(1.0)
        # Iterate from highest threshold down to preserve monotonicity over suffix losses
        best_threshold = 1.0
        best_bound = b_max / (n + 1)
        # Evaluate all candidates in ascending order and pick the smallest lambda such that
        # all lambda' >= lambda also satisfy the monotone CRC condition.
        bounds_by_cand: List[Tuple[float, float]] = []
        for lam in candidates:
            accepted_losses = [loss for margin, loss in pairs if margin >= lam]
            if not accepted_losses:
                emp_risk = 0.0
            else:
                emp_risk = sum(accepted_losses) / len(accepted_losses)
            # Finite-sample CRC upper bound weighted over acceptance rate
            crc_bound = (sum(accepted_losses) + b_max) / (n + 1)
            bounds_by_cand.append((lam, crc_bound if accepted_losses else emp_risk))

        valid_suffix = True
        for lam, crc_bound in reversed(bounds_by_cand):
            if crc_bound <= alpha and valid_suffix:
                best_threshold = lam
                best_bound = crc_bound
            else:
                valid_suffix = False
        return float(best_threshold), float(best_bound)

    def calibrate_labeled(
        self,
        margins: Sequence[float],
        losses: Sequence[float],
        *,
        b_max: float = 1.0,
        target_risk_alpha: Optional[float] = None,
    ) -> float:
        """Calibrate stopping threshold via Monotone Conformal Risk Control (Angelopoulos et al. 2024)."""
        if len(margins) != len(losses) or not margins:
            raise ValueError("margins and losses must be non-empty sequences of equal length")
        b_lim = float(b_max)
        if not math.isfinite(b_lim) or b_lim <= 0.0:
            raise ValueError("b_max must be finite and positive")
        alpha = self.target_risk_alpha if target_risk_alpha is None else self.validate_risk(target_risk_alpha)
        validated_pairs: List[Tuple[float, float]] = []
        for m, l_val in zip(margins, losses):
            m_clean = self.validate_probability(m, "calibration margin")
            l_clean = float(l_val)
            if not math.isfinite(l_clean) or not 0.0 <= l_clean <= b_lim:
                raise ValueError(f"calibration loss must be finite and in [0, {b_lim}]")
            validated_pairs.append((m_clean, l_clean))

        threshold, emp_bound = self._crc_threshold_for_alpha(validated_pairs, alpha, b_lim)
        rc_profile = self.selective_risk_coverage_profile(
            [m for m, _ in validated_pairs],
            [l_v for _, l_v in validated_pairs],
        )
        self.calibrated_threshold = threshold
        self.calibration_samples = sorted(m for m, _ in validated_pairs)
        self._labeled_pairs = validated_pairs
        self._labeled_b_max = b_lim
        self._labeled_alpha = alpha
        self._labeled_calibrated = True
        self._empirical_crc_risk = emp_bound
        self._rc_profile = rc_profile
        return threshold

    @classmethod
    def selective_risk_coverage_profile(
        cls,
        margins: Sequence[float],
        losses: Sequence[float],
    ) -> Dict[str, Any]:
        """Compute Risk-Coverage curve, AURC, Oracle AURC, and Excess AURC (Geifman & El-Yaniv 2017)."""
        if len(margins) != len(losses) or not margins:
            raise ValueError("margins and losses must be non-empty sequences of equal length")
        items = [
            (cls.validate_probability(m, "margin"), float(loss))
            for m, loss in zip(margins, losses)
        ]
        for _, loss in items:
            if not math.isfinite(loss) or loss < 0.0:
                raise ValueError("losses must be finite and non-negative")

        n = len(items)
        # Sort descending by margin (highest confidence accepted first)
        ordered = sorted(items, key=lambda pair: -pair[0])
        oracle_ordered = sorted(items, key=lambda pair: pair[1])

        curve: List[Dict[str, float]] = []
        cum_loss = 0.0
        oracle_cum = 0.0
        aurc = 0.0
        oracle_aurc = 0.0
        prev_cov = 0.0
        prev_risk = ordered[0][1]
        prev_oracle_risk = oracle_ordered[0][1]

        for idx, ((m_val, loss_val), (_, o_loss)) in enumerate(zip(ordered, oracle_ordered), start=1):
            cum_loss += loss_val
            oracle_cum += o_loss
            cov = idx / n
            sel_risk = cum_loss / idx
            o_risk = oracle_cum / idx
            dc = cov - prev_cov
            aurc += 0.5 * (prev_risk + sel_risk) * dc
            oracle_aurc += 0.5 * (prev_oracle_risk + o_risk) * dc
            prev_cov = cov
            prev_risk = sel_risk
            prev_oracle_risk = o_risk
            curve.append({
                "coverage": round(cov, 6),
                "margin_threshold": round(m_val, 6),
                "selective_risk": round(sel_risk, 6),
                "oracle_risk": round(o_risk, 6),
            })

        e_aurc = max(0.0, aurc - oracle_aurc)
        return {
            "n_samples": n,
            "mean_loss": round(cum_loss / n, 6),
            "aurc": round(aurc, 6),
            "oracle_aurc": round(oracle_aurc, 6),
            "e_aurc": round(e_aurc, 6),
            "curve": curve,
        }

    def evaluate_stopping(
        self,
        query: str,
        current_step: int,
        max_budget: int = 6,
        top_confidence: float = 0.92,
        runner_up_confidence: float = 0.45,
        target_risk_alpha: Optional[float] = None,
    ) -> ConformalStoppingResult:
        """Return stopping decision, with finite-sample CRC certificate when labeled-calibrated."""
        if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < 1:
            raise ValueError("current_step must be a positive integer")
        if isinstance(max_budget, bool) or not isinstance(max_budget, int) or max_budget < current_step:
            raise ValueError("max_budget must be an integer at least current_step")
        risk = self.target_risk_alpha if target_risk_alpha is None else self.validate_risk(target_risk_alpha)
        top = self.validate_probability(top_confidence, "top_confidence")
        runner_up = self.validate_probability(runner_up_confidence, "runner_up_confidence")
        margin = top - runner_up

        active_threshold = self.calibrated_threshold
        active_emp_bound = self._empirical_crc_risk
        if self._labeled_calibrated and abs(risk - self._labeled_alpha) > 1e-12:
            active_threshold, active_emp_bound = self._crc_threshold_for_alpha(
                self._labeled_pairs, risk, self._labeled_b_max
            )

        terminal = current_step >= max_budget
        margin_satisfied = margin >= active_threshold
        should_exit = margin_satisfied or terminal
        savings = round((max_budget - current_step) / max_budget * 100.0, 1) if should_exit else 0.0

        certified_bound: Optional[float] = risk if (self._labeled_calibrated and not terminal) else None
        if self._labeled_calibrated and margin_satisfied:
            certified_bound = risk

        if terminal and not margin_satisfied:
            reason = "budget_exhausted"
            summary = f"Budget exhausted at step {current_step}/{max_budget}; no output correctness or risk bound established."
            certified_bound = None
        elif margin_satisfied and self._labeled_calibrated:
            reason = "conformal_risk_controlled"
            summary = (
                f"CRC early-exit certified at risk alpha={risk:.3f} (empirical CRC bound={active_emp_bound:.4f}): "
                f"margin {margin:.3f} >= threshold {active_threshold:.3f}. "
                f"Potential step-budget savings {savings:.1f}%."
            )
        elif margin_satisfied:
            reason = "heuristic_margin"
            summary = (
                f"Heuristic stop advice: margin {margin:.3f} meets threshold {active_threshold:.3f}. "
                f"Potential step-budget savings {savings:.1f}%; error risk is uncalibrated."
            )
        else:
            reason = "continue"
            risk_note = f"CRC threshold {active_threshold:.3f} (alpha={risk:.3f})." if self._labeled_calibrated else "Error risk is uncalibrated."
            summary = f"Continue compute: margin {margin:.3f} is below threshold {active_threshold:.3f}. {risk_note}"

        if self._labeled_calibrated:
            cal_status = "labeled_conformal_risk_control"
        elif self.calibration_samples:
            cal_status = "unlabeled_margin_quantile"
        else:
            cal_status = "uncalibrated"

        telemetry: Dict[str, Any] = {
            "target_risk_alpha": risk,
            "top_confidence": top,
            "runner_up_confidence": runner_up,
            "calibration_pool_size": len(self.calibration_samples),
            "calibration_status": cal_status,
            "risk_certified": bool(certified_bound is not None),
            "answer_authority": False,
            "stop_reason": reason,
            "savings_measure": "potential_step_budget_fraction_not_measured_flops",
        }
        if self._labeled_calibrated and active_emp_bound is not None:
            telemetry["empirical_crc_risk"] = round(active_emp_bound, 6)
        if self._rc_profile is not None:
            telemetry["selective_aurc"] = self._rc_profile["aurc"]
            telemetry["selective_e_aurc"] = self._rc_profile["e_aurc"]

        return ConformalStoppingResult(
            query=query,
            current_step=current_step,
            max_budget=max_budget,
            observed_margin=margin,
            calibrated_threshold=active_threshold,
            should_early_exit=should_exit,
            certified_risk_bound=certified_bound,
            compute_savings_pct=savings,
            diagnostic_summary=summary,
            telemetry=telemetry,
        )
