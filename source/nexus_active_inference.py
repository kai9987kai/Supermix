"""Supermix v89 Epistemic Active Inference Engine.

Implements Karl Friston's Free Energy Principle for autoregressive and multi-step
reasoning. Actions are evaluated by their Expected Free Energy (EFE), decomposing into:
1. Pragmatic Value (Goal Realization / Risk Minimization): Alignment with prior task targets.
2. Epistemic Value (Information Gain / Active Exploration): Reduction of posterior uncertainty.

Action selection follows a precision-weighted softmax policy:
    P(a) = softmax(-beta * G(a))
where precision beta is modulated by the model's RSI volatility oscillator.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ReasoningActionType(str, Enum):
    DECOMPOSE_SUBGOAL = "DECOMPOSE_SUBGOAL"
    EXECUTE_DETERMINISTIC_STEP = "EXECUTE_DETERMINISTIC_STEP"
    CAUSAL_COUNTERFACTUAL_CHECK = "CAUSAL_COUNTERFACTUAL_CHECK"
    EXPAND_SPECULATIVE_DRAFT = "EXPAND_SPECULATIVE_DRAFT"
    BACKTRACK_PRUNE = "BACKTRACK_PRUNE"
    HALT_AND_SEAL = "HALT_AND_SEAL"


@dataclass
class ReasoningAction:
    action_type: ReasoningActionType
    description: str
    pragmatic_risk: float  # Divergence from goal prior D_KL[Q(s'|a) || P(s')]
    epistemic_gain: float  # Expected information gain I(s'; theta)
    expected_free_energy: float  # G(a) = pragmatic_risk - epistemic_gain
    selection_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["action_type"] = self.action_type.value
        return d


@dataclass
class ActiveInferenceResult:
    query: str
    current_state_summary: str
    local_entropy: float
    rsi_volatility: float
    precision_beta: float
    candidate_actions: List[ReasoningAction]
    selected_action: ReasoningAction
    epistemic_pragmatic_ratio: float
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "current_state_summary": self.current_state_summary,
            "local_entropy": self.local_entropy,
            "rsi_volatility": self.rsi_volatility,
            "precision_beta": self.precision_beta,
            "candidate_actions": [a.to_dict() for a in self.candidate_actions],
            "selected_action": self.selected_action.to_dict(),
            "epistemic_pragmatic_ratio": self.epistemic_pragmatic_ratio,
            "diagnostic_summary": self.diagnostic_summary,
            "telemetry": self.telemetry,
        }


class ActiveInferenceController:
    """Karl Friston Free Energy Principle Controller for Cognitive Planning."""

    def __init__(self, base_temperature: float = 0.70, epistemic_weight: float = 1.0):
        self.base_temperature = max(0.05, min(2.0, float(base_temperature)))
        self.epistemic_weight = max(0.1, min(5.0, float(epistemic_weight)))

    def compute_precision_beta(self, rsi: float, local_entropy: float) -> float:
        """Dynamic precision beta = (1 / T) * (1 + |RSI - 50| / 50) * exp(-0.3 * H).

        When RSI indicates extreme volatility or conviction (overbought > 70 or oversold < 30),
        precision tightens (higher beta). When entropy H is high, precision softens to allow exploration.
        """
        clamped_rsi = max(0.0, min(100.0, float(rsi)))
        clamped_ent = max(0.0, float(local_entropy))
        rsi_deviation = abs(clamped_rsi - 50.0) / 50.0
        beta = (1.0 / self.base_temperature) * (1.0 + 0.8 * rsi_deviation) * math.exp(-0.25 * min(3.0, clamped_ent))
        return round(max(0.2, min(10.0, beta)), 4)

    def evaluate_expected_free_energy(
        self,
        action_type: ReasoningActionType,
        step_index: int,
        local_entropy: float,
        verification_confidence: float,
        has_pending_subgoals: bool,
    ) -> Tuple[float, float, float]:
        """Compute (pragmatic_risk, epistemic_gain, G(a)).

        G(a) = Pragmatic_Risk - gamma * Epistemic_Gain
        Lower G(a) means a more desirable, uncertainty-resolving, goal-directed action.
        """
        conf = max(0.0, min(1.0, float(verification_confidence)))
        h = max(0.0, min(5.0, float(local_entropy)))

        if action_type == ReasoningActionType.EXECUTE_DETERMINISTIC_STEP:
            # Low pragmatic risk when confidence is high; moderate epistemic gain
            pragmatic_risk = (1.0 - conf) * 1.5
            epistemic_gain = 0.45 * (1.0 + 0.5 * h)

        elif action_type == ReasoningActionType.DECOMPOSE_SUBGOAL:
            # High epistemic gain when entropy is elevated or subgoals remain
            pragmatic_risk = 0.8 if has_pending_subgoals else 1.8
            epistemic_gain = 0.95 * (0.8 + 0.4 * h)

        elif action_type == ReasoningActionType.CAUSAL_COUNTERFACTUAL_CHECK:
            # Strong epistemic probe; evaluates register sensitivity
            pragmatic_risk = 0.6 + 0.4 * (1.0 - conf)
            epistemic_gain = 1.10 * (1.0 - conf * 0.5)

        elif action_type == ReasoningActionType.EXPAND_SPECULATIVE_DRAFT:
            # Fast forward expansion; risky if entropy is high
            pragmatic_risk = 0.4 + 0.8 * (h / 3.0)
            epistemic_gain = 0.50 * (conf + 0.2)

        elif action_type == ReasoningActionType.BACKTRACK_PRUNE:
            # High pragmatic cost unless confidence is abysmal
            pragmatic_risk = 0.3 if conf < 0.35 else 2.5
            epistemic_gain = 0.85 if conf < 0.35 else 0.15

        elif action_type == ReasoningActionType.HALT_AND_SEAL:
            # Goal state: lowest pragmatic risk if confident and finished
            pragmatic_risk = (1.0 - conf) * 3.0 + (1.5 if has_pending_subgoals else 0.1)
            epistemic_gain = 0.05  # Terminal state yields negligible new information

        else:
            pragmatic_risk = 1.0
            epistemic_gain = 0.5

        g = pragmatic_risk - (self.epistemic_weight * epistemic_gain)
        return round(pragmatic_risk, 4), round(epistemic_gain, 4), round(g, 4)

    def decide(
        self,
        query: str,
        current_trace_steps: Optional[List[str]] = None,
        local_entropy: float = 0.85,
        rsi_volatility: float = 50.0,
        verification_confidence: float = 0.80,
        has_pending_subgoals: bool = False,
    ) -> ActiveInferenceResult:
        trace = current_trace_steps or []
        step_idx = len(trace)

        beta = self.compute_precision_beta(rsi_volatility, local_entropy)

        # Build candidate actions
        candidates: List[ReasoningAction] = []
        action_descriptions = {
            ReasoningActionType.DECOMPOSE_SUBGOAL: "Deconstruct compound objective into verified atomic sub-derivations",
            ReasoningActionType.EXECUTE_DETERMINISTIC_STEP: "Apply exact symbolic or arithmetic transformation to active register",
            ReasoningActionType.CAUSAL_COUNTERFACTUAL_CHECK: "Perturb intermediate register to measure downstream counterfactual response",
            ReasoningActionType.EXPAND_SPECULATIVE_DRAFT: "Draft speculative candidate continuation with multi-token predictor",
            ReasoningActionType.BACKTRACK_PRUNE: "Ablate current unpromising reasoning branch and backtrack to stable register",
            ReasoningActionType.HALT_AND_SEAL: "Emit definitive ground-truth conclusion and commit turn to evidence ledger",
        }

        for act_type, desc in action_descriptions.items():
            # If trace is empty, HALT is not allowed
            if act_type == ReasoningActionType.HALT_AND_SEAL and step_idx == 0:
                continue
            # If no errors and high confidence, BACKTRACK is deprioritized
            p_risk, e_gain, g = self.evaluate_expected_free_energy(
                act_type,
                step_idx,
                local_entropy,
                verification_confidence,
                has_pending_subgoals,
            )
            candidates.append(
                ReasoningAction(
                    action_type=act_type,
                    description=desc,
                    pragmatic_risk=p_risk,
                    epistemic_gain=e_gain,
                    expected_free_energy=g,
                )
            )

        # Softmax probability over negative free energy: P(a) ~ exp(-beta * G(a))
        neg_beta_g = [-beta * a.expected_free_energy for a in candidates]
        max_val = max(neg_beta_g)
        exp_vals = [math.exp(v - max_val) for v in neg_beta_g]
        sum_exp = sum(exp_vals) or 1.0

        for idx, act in enumerate(candidates):
            act.selection_probability = round(exp_vals[idx] / sum_exp, 4)

        # Select highest probability action (deterministic argmax under active inference)
        selected = max(candidates, key=lambda a: a.selection_probability)

        tot_epistemic = sum(a.epistemic_gain for a in candidates)
        tot_pragmatic = sum(a.pragmatic_risk for a in candidates)
        ratio = round(tot_epistemic / max(1e-5, tot_pragmatic), 3)

        summary = (
            f"Active Inference Policy: Selected [{selected.action_type.value}] "
            f"(P={selected.selection_probability:.3f}, G={selected.expected_free_energy:.3f}, "
            f"Precision beta={beta:.2f}, RSI={rsi_volatility:.1f}). "
            f"Epistemic Gain={selected.epistemic_gain:.3f}, Pragmatic Risk={selected.pragmatic_risk:.3f}."
        )

        return ActiveInferenceResult(
            query=query,
            current_state_summary=f"Trace length {step_idx} steps. Last step: {trace[-1] if trace else 'ROOT'}",
            local_entropy=round(local_entropy, 3),
            rsi_volatility=round(rsi_volatility, 1),
            precision_beta=beta,
            candidate_actions=candidates,
            selected_action=selected,
            epistemic_pragmatic_ratio=ratio,
            diagnostic_summary=summary,
            telemetry={
                "base_temperature": self.base_temperature,
                "epistemic_weight": self.epistemic_weight,
                "step_index": step_idx,
            },
        )
