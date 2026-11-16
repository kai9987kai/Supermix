"""Adaptive computation, monotone conformal risk control (CRC), and decode safety."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf.config import SCFFeatureFlags, resolve_decoding_policy  # noqa: E402

__all__ = ["SCFAdaptiveComputeController"]


class SCFAdaptiveComputeController:
    """Unifies Cognitive Leap budgets, monotone Conformal Risk Control, and decode policy."""

    def __init__(self, *, target_risk_alpha: float = 0.05) -> None:
        self.target_risk_alpha = float(target_risk_alpha)

    def evaluate_budget(
        self,
        *,
        prompt: str,
        flags: SCFFeatureFlags,
        current_step: int = 1,
        max_budget: int = 6,
        top_confidence: float = 0.90,
        runner_up_confidence: float = 0.25,
        adaptive_thinking: bool = False,
    ) -> Dict[str, Any]:
        import chat_app
        import nexus_conformal_stopping as ncs

        decoding_policy = resolve_decoding_policy(
            flags, adaptive_thinking=adaptive_thinking
        )
        est = chat_app.estimate_auto_reasoning_cycles(prompt)
        est_cycles = int(est.get("cycles", 1)) if isinstance(est, dict) else 1
        resolved_list = chat_app.resolve_runtime_compute_cycles([est_cycles])
        planned_cycles = int(resolved_list[0]) if resolved_list else est_cycles
        if not flags.SCF_ENABLED or not flags.SCF_ADAPTIVE_COMPUTE_ENABLED:
            return {
                "adaptive_enabled": False,
                "planned_cycles": int(planned_cycles),
                "should_stop": True,
                "decision": "FIXED_BASELINE_BUDGET",
                "decoding_policy": decoding_policy,
            }

        conformal = ncs.ConformalStoppingController(target_risk_alpha=self.target_risk_alpha)
        res = conformal.evaluate_stopping(
            query=prompt,
            current_step=max(1, int(current_step)),
            max_budget=max(max(1, int(current_step)), int(max_budget)),
            top_confidence=float(top_confidence),
            runner_up_confidence=float(runner_up_confidence),
        ).to_dict()
        return {
            "adaptive_enabled": True,
            "planned_cycles": int(planned_cycles),
            "should_stop": bool(res.get("should_stop", False)),
            "conformal_decision": res,
            "decoding_policy": decoding_policy,
        }

    def calibrate_selective_risk(
        self,
        margins: Sequence[float],
        losses: Sequence[float],
        *,
        b_max: float = 1.0,
        target_risk_alpha: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute finite-sample monotone CRC threshold & AURC/E-AURC profile."""
        import nexus_conformal_stopping as ncs

        alpha = float(target_risk_alpha if target_risk_alpha is not None else self.target_risk_alpha)
        ctrl = ncs.ConformalStoppingController(target_risk_alpha=alpha)
        threshold = float(
            ctrl.calibrate_labeled(margins, losses, b_max=b_max, target_risk_alpha=alpha)
        )
        profile = ncs.ConformalStoppingController.selective_risk_coverage_profile(
            margins, losses
        )
        return {
            "calibrated_threshold": threshold,
            "target_risk_alpha": alpha,
            "risk_coverage_profile": profile,
            "answer_authority": False,
        }
