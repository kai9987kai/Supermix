"""Adaptive thinking controller for MiMoMix.

This is the runtime half of the v53 line. It answers one question per request:
*how much recursive compute should this turn get, and when is it safe to stop?*

It fuses two lineages:

* **Supermix v51/v52.1 progressive auto compute.** Plan before inference; set
  separate task-difficulty and epistemic-risk floors; evaluate an escalating
  budget ladder; require **cross-budget agreement** on the ordered top-k
  decision before accepting an early exit; and **reuse the accepted probe's
  output** rather than recomputing it. Crucially the accepted output is the
  literal output of the accepted budget -- never a blend of budgets, never a
  post-hoc patch.

* **MiMo's token-efficiency direction.** Route to a cheap mode by default and
  escalate only for long-horizon or tool-using work, treating "shortest
  trajectory that still lands" as the objective rather than "always think
  hard". The model's own quality/continue verifier acts as an advisory stop
  signal, consumed by this controller instead of starting a second nested
  escalation loop inside the model.

Two boundaries are deliberate and enforced:

1. **The controller cannot change what the model computed.** It selects among
   outputs the model actually produced at real budgets. :func:`audit_decision_fidelity`
   is the only way to claim a savings number, and it pays full price to do so.
2. **The verifier is advisory.** A high continue-probability can *prevent* an
   early exit; it can never authorise one on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mimomix_core import MiMoMixModel, MiMoMixOutput


__all__ = [
    "ThinkingMode",
    "RequestFeatures",
    "ThinkingPolicy",
    "ThinkingPlan",
    "ThinkingDecision",
    "plan_request",
    "decide",
    "audit_decision_fidelity",
]


ThinkingMode = str  # "fast" | "deep" | "agent"
VALID_MODES = ("fast", "deep", "agent")


# ---------------------------------------------------------------------------
# Request features and planning
# ---------------------------------------------------------------------------


@dataclass
class RequestFeatures:
    """Observable, prompt-derived signals. All optional, all bounded.

    These mirror the shape of the existing ``prompt_understanding`` profile:
    cheap deterministic cues, never the raw prompt, never anything that could
    grant permissions. A caller that has the real profile can map it onto these
    fields; a caller that has only token ids can leave everything at default.
    """

    prompt_tokens: int = 0
    #: number of distinct requested acts (multi-part questions are harder)
    requested_acts: int = 1
    #: caller declared tools are available for this turn
    tool_calls_available: int = 0
    #: the turn contains a hard constraint conflict or a missing reference
    has_conflict: bool = False
    #: the turn needs external evidence / freshness
    needs_evidence: bool = False
    #: the turn carries an immediate personal-safety cue
    safety_critical: bool = False
    #: caller-declared expected output length
    max_output_tokens: int = 256

    def difficulty(self) -> float:
        """Deterministic task-difficulty score in ``[0, 1]``."""

        length_term = min(1.0, math.log1p(max(0, self.prompt_tokens)) / math.log(4096.0))
        act_term = min(1.0, max(0, self.requested_acts - 1) / 4.0)
        tool_term = min(1.0, self.tool_calls_available / 8.0)
        output_term = min(1.0, math.log1p(max(0, self.max_output_tokens)) / math.log(8192.0))
        score = 0.35 * length_term + 0.25 * act_term + 0.25 * tool_term + 0.15 * output_term
        return round(min(1.0, max(0.0, score)), 6)

    def epistemic_risk(self) -> float:
        """Deterministic risk-of-being-confidently-wrong score in ``[0, 1]``."""

        score = 0.0
        if self.has_conflict:
            score += 0.45
        if self.needs_evidence:
            score += 0.35
        if self.requested_acts > 2:
            score += 0.10
        if self.tool_calls_available > 0:
            score += 0.10
        return round(min(1.0, score), 6)


@dataclass
class ThinkingPolicy:
    """Static configuration for the controller."""

    #: escalating budget ladder, in recursive cycles
    ladder: Tuple[int, ...] = (1, 2, 4)
    #: ordered top-k depth that must agree across budgets (v51 uses ordered top-3)
    agreement_rank_depth: int = 3
    #: minimum ``P(rank 1) - P(rank 2)`` gap before an early exit is allowed.
    #: This protects the emitted token, which is what the exit actually risks.
    decision_margin: float = 5e-4
    #: minimum ``P(rank k) - P(rank k+1)`` gap. Defaults to 0 (off): on a
    #: confident distribution the tail collapses, so gating on it would block
    #: every early exit on the requests that most deserve one. Raise it only
    #: when the *ordering* of the top-k list is itself a product surface.
    boundary_margin: float = 0.0
    #: confidence floor for an early exit
    confidence_target: float = 0.30
    #: entropy ceiling (nats) for an early exit
    entropy_target: float = 3.0
    #: verifier continue-probability above which an early exit is refused
    continue_threshold: float = 0.60
    #: an early exit needs two consecutive budgets that agree
    require_cross_budget_agreement: bool = True
    #: floors applied on top of the plan, per mode
    mode_floor: Dict[str, int] = field(default_factory=lambda: {"fast": 1, "deep": 2, "agent": 2})
    mode_ceiling: Dict[str, int] = field(default_factory=lambda: {"fast": 2, "deep": 4, "agent": 8})

    def normalised_ladder(self, floor: int, ceiling: int) -> Tuple[int, ...]:
        values = sorted({int(b) for b in self.ladder if floor <= int(b) <= ceiling})
        if not values:
            values = [max(1, min(floor, ceiling))]
        if values[-1] < ceiling:
            values.append(int(ceiling))
        return tuple(values)


@dataclass
class ThinkingPlan:
    """What the controller decided *before* touching the model."""

    mode: ThinkingMode
    difficulty: float
    epistemic_risk: float
    floor_cycles: int
    ceiling_cycles: int
    ladder: Tuple[int, ...]
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["ladder"] = list(self.ladder)
        return payload


def plan_request(
    features: RequestFeatures,
    policy: Optional[ThinkingPolicy] = None,
    mode: Optional[ThinkingMode] = None,
    max_model_cycles: int = 8,
) -> ThinkingPlan:
    """Choose a mode and a bounded budget ladder from observable cues only.

    Mode selection follows the routing shape sketched in ``readme task.txt``:
    tools or long horizons go to ``agent``; long / multi-part / evidence-seeking
    turns go to ``deep``; everything else stays ``fast``. A safety-critical turn
    is *not* pushed to a deep budget -- the v52.1 rule is that urgent-help
    guidance must not be delayed by forced compute.
    """

    policy = policy or ThinkingPolicy()
    difficulty = features.difficulty()
    risk = features.epistemic_risk()
    reasons: List[str] = []

    if mode is not None:
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        chosen = mode
        reasons.append("mode_explicitly_requested")
    elif features.safety_critical:
        chosen = "fast"
        reasons.append("safety_fast_path")
    elif features.tool_calls_available > 0:
        chosen = "agent"
        reasons.append("tools_available")
    elif difficulty >= 0.45 or risk >= 0.40:
        chosen = "deep"
        reasons.append("difficulty_or_risk_floor")
    else:
        chosen = "fast"
        reasons.append("default_fast")

    floor = int(policy.mode_floor.get(chosen, 1))
    ceiling = int(policy.mode_ceiling.get(chosen, 4))

    # Difficulty and epistemic risk raise the floor independently: a hard task
    # and a risky task need compute for different reasons.
    if difficulty >= 0.65:
        floor = max(floor, 2)
        reasons.append("difficulty_floor")
    if risk >= 0.60:
        floor = max(floor, 2)
        reasons.append("epistemic_risk_floor")
    if features.safety_critical:
        ceiling = min(ceiling, max(floor, 2))
        reasons.append("safety_ceiling")

    ceiling = max(1, min(ceiling, int(max_model_cycles)))
    floor = max(1, min(floor, ceiling))
    ladder = policy.normalised_ladder(floor, ceiling)

    return ThinkingPlan(
        mode=chosen,
        difficulty=difficulty,
        epistemic_risk=risk,
        floor_cycles=floor,
        ceiling_cycles=ceiling,
        ladder=ladder,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Decision probes
# ---------------------------------------------------------------------------


def _decision_signature(
    logits: torch.Tensor, rank_depth: int
) -> Tuple[Tuple[int, ...], float, float, float, float]:
    """Ordered top-k ids, confidence, entropy, and **two** margins.

    ``logits`` is ``(V,)`` for a single position. The *ordered* top-k is what
    must persist across budgets -- an unordered set can hide a swap that
    changes the emitted token.

    The two margins answer different questions and must not be conflated:

    ``decision_margin`` = ``P(rank 1) - P(rank 2)``
        How safe the *emitted token* is. This is what an early exit risks, so
        this is what the gate uses.
    ``boundary_margin`` = ``P(rank k) - P(rank k+1)``
        How safe the ordered top-k *listing* is at its outer edge. Reported as
        a diagnostic and gated only if the caller opts in.

    Gating on the boundary margin alone is a real trap: a confident model puts
    ~1.0 on rank 1 and ~1e-5 across the tail, so the rank-k/rank-k+1 gap
    collapses toward zero exactly when the decision is safest. That blocks every
    early exit on precisely the requests that deserve one.
    """

    probs = torch.softmax(logits.float(), dim=-1)
    vocab = probs.shape[-1]
    depth = max(1, min(int(rank_depth), vocab))
    top = torch.topk(probs, k=min(depth + 1, vocab))
    ordered = tuple(int(i) for i in top.indices[:depth].tolist())
    confidence = float(top.values[0])
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    decision_margin = float(top.values[0] - top.values[1]) if vocab > 1 else 1.0
    if top.values.shape[0] > depth:
        boundary_margin = float(top.values[depth - 1] - top.values[depth])
    else:
        boundary_margin = float(top.values[-1])
    return ordered, confidence, entropy, decision_margin, boundary_margin


@dataclass
class BudgetProbe:
    budget: int
    ordered_top_k: Tuple[int, ...]
    confidence: float
    entropy: float
    decision_margin: float
    boundary_margin: float
    continue_probability: float
    quality_probability: float
    cycles_used: float
    exit_reason: str

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["ordered_top_k"] = list(self.ordered_top_k)
        return payload


@dataclass
class ThinkingDecision:
    """The controller's audit record. JSON-safe and free of prompt content."""

    plan: ThinkingPlan
    accepted_budget: int
    budgets_evaluated: List[int]
    forward_evaluations: int
    exit_reason: str
    reused_accepted_probe: bool
    probes: List[BudgetProbe]

    @property
    def cycles_spent(self) -> int:
        """Recursive cycles actually executed, summed over every probe."""

        return sum(p.budget for p in self.probes)

    @property
    def cycle_reduction(self) -> float:
        """Cycles saved versus always running the ceiling budget.

        This is the honest headline metric, and it is **signed**. A ladder that
        exits early saves cycles; a ladder that runs to exhaustion spends more
        than a fixed ceiling budget would have and reports a negative number.
        Progressive compute is a bet, and this is where the bet is scored.

        Forward *count* is deliberately not reported as a saving: probing at
        budget 1 and at budget 4 are not the same unit of work, so a forward
        ratio would flatter the controller.
        """

        ceiling = self.plan.ceiling_cycles
        if ceiling <= 0:
            return 0.0
        return round(1.0 - (self.cycles_spent / ceiling), 6)

    def to_dict(self) -> Dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "accepted_budget": self.accepted_budget,
            "budgets_evaluated": list(self.budgets_evaluated),
            "forward_evaluations": self.forward_evaluations,
            "cycles_spent": self.cycles_spent,
            "exit_reason": self.exit_reason,
            "reused_accepted_probe": self.reused_accepted_probe,
            "cycle_reduction": self.cycle_reduction,
            "probes": [p.to_dict() for p in self.probes],
        }


def _probe(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    budget: int,
    rank_depth: int,
) -> Tuple[MiMoMixOutput, BudgetProbe]:
    with torch.no_grad():
        output = model(
            input_ids,
            thinking_cycles=budget,
            adaptive_thinking=False,
            return_mtp=False,
            past_length=0,
        )
    # The decision is read at the last position of the first row: this is a
    # next-token decision probe, not a claim about the whole batch.
    ordered, confidence, entropy, decision_margin, boundary_margin = _decision_signature(
        output.logits[0, -1], rank_depth
    )
    thinking = output.telemetry.get("thinking", {}) if output.telemetry else {}
    probe = BudgetProbe(
        budget=int(budget),
        ordered_top_k=ordered,
        confidence=confidence,
        entropy=entropy,
        decision_margin=decision_margin,
        boundary_margin=boundary_margin,
        continue_probability=float(thinking.get("continue_probability", 0.0)),
        quality_probability=float(thinking.get("quality_probability", 0.0)),
        cycles_used=float(thinking.get("cycles_used", budget)),
        exit_reason=str(thinking.get("exit_reason", "no_thinking_core")),
    )
    return output, probe


def decide(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    features: Optional[RequestFeatures] = None,
    policy: Optional[ThinkingPolicy] = None,
    mode: Optional[ThinkingMode] = None,
    plan: Optional[ThinkingPlan] = None,
) -> Tuple[MiMoMixOutput, ThinkingDecision]:
    """Run the progressive budget ladder and return the accepted probe's output.

    The returned :class:`MiMoMixOutput` is *the object produced at the accepted
    budget* -- literally reused, not recomputed and not merged. If no budget
    earns an early exit, the ceiling budget's output is returned exactly, which
    is the same thing a fixed-budget runtime would have produced.
    """

    policy = policy or ThinkingPolicy()
    features = features or RequestFeatures(prompt_tokens=int(input_ids.shape[1]))
    max_cycles = model.config.thinking_max_cycles if model.thinking_core is not None else 1
    plan = plan or plan_request(features, policy=policy, mode=mode, max_model_cycles=max_cycles)

    probes: List[BudgetProbe] = []
    outputs: List[MiMoMixOutput] = []
    accepted_index: Optional[int] = None
    exit_reason = "ceiling_budget"

    for index, budget in enumerate(plan.ladder):
        output, probe = _probe(model, input_ids, budget, policy.agreement_rank_depth)
        probes.append(probe)
        outputs.append(output)

        if budget >= plan.ceiling_cycles:
            accepted_index = index
            exit_reason = "ceiling_budget"
            break
        if budget < plan.floor_cycles:
            continue

        # --- gate 1: the model's own advisory stop signal ------------------
        if probe.continue_probability >= policy.continue_threshold:
            continue
        # --- gate 2: calibrated confidence / entropy targets ---------------
        if probe.confidence < policy.confidence_target:
            continue
        if probe.entropy > policy.entropy_target:
            continue
        # --- gate 3: decision-safety at the rank-1/rank-2 boundary ---------
        if probe.decision_margin < policy.decision_margin:
            continue
        # --- gate 3b: optional ordered-top-k listing stability -------------
        if policy.boundary_margin > 0.0 and probe.boundary_margin < policy.boundary_margin:
            continue
        # --- gate 4: cross-budget agreement --------------------------------
        if policy.require_cross_budget_agreement:
            if index == 0:
                continue
            if probes[index - 1].ordered_top_k != probe.ordered_top_k:
                continue
            exit_reason = "cross_budget_agreement"
        else:
            exit_reason = "single_budget_targets_met"
        accepted_index = index
        break

    if accepted_index is None:
        accepted_index = len(probes) - 1
        exit_reason = "ladder_exhausted"

    decision = ThinkingDecision(
        plan=plan,
        accepted_budget=probes[accepted_index].budget,
        budgets_evaluated=[p.budget for p in probes],
        forward_evaluations=len(probes),
        exit_reason=exit_reason,
        reused_accepted_probe=True,
        probes=probes,
    )
    return outputs[accepted_index], decision


def audit_decision_fidelity(
    model: MiMoMixModel,
    requests: Sequence[torch.Tensor],
    features: Optional[Sequence[RequestFeatures]] = None,
    policy: Optional[ThinkingPolicy] = None,
    mode: Optional[ThinkingMode] = None,
) -> Dict[str, object]:
    """Pay full price to measure what the controller's savings actually cost.

    For every request this runs the controller *and* the ceiling budget, then
    reports how often the accepted decision differs from the ceiling decision.
    This is the only honest basis for a savings claim: the controller alone
    never sees the counterfactual it is being graded against.

    Reported keys:

    ``top1_disagreements``
        accepted top-1 token differs from the ceiling budget's top-1.
    ``ordered_topk_disagreements``
        the ordered top-k differs (a strictly stricter test than top-1).
    ``mean_cycles`` / ``ceiling_cycles``
        recursive cycles actually spent versus the always-ceiling baseline.
    """

    policy = policy or ThinkingPolicy()
    total = len(requests)
    top1_bad = 0
    topk_bad = 0
    spent_cycles = 0
    ceiling_cycles = 0
    exits: Dict[str, int] = {}

    for index, ids in enumerate(requests):
        feats = features[index] if features is not None else RequestFeatures(prompt_tokens=int(ids.shape[1]))
        _, decision = decide(model, ids, features=feats, policy=policy, mode=mode)
        exits[decision.exit_reason] = exits.get(decision.exit_reason, 0) + 1
        spent_cycles += sum(p.budget for p in decision.probes)
        ceiling_cycles += decision.plan.ceiling_cycles

        _, reference = _probe(
            model, ids, decision.plan.ceiling_cycles, policy.agreement_rank_depth
        )
        accepted = next(p for p in decision.probes if p.budget == decision.accepted_budget)
        if accepted.ordered_top_k[:1] != reference.ordered_top_k[:1]:
            top1_bad += 1
        if accepted.ordered_top_k != reference.ordered_top_k:
            topk_bad += 1

    return {
        "requests": total,
        "top1_disagreements": top1_bad,
        "ordered_topk_disagreements": topk_bad,
        "top1_fidelity": round(1.0 - (top1_bad / total), 6) if total else 1.0,
        "ordered_topk_fidelity": round(1.0 - (topk_bad / total), 6) if total else 1.0,
        "cycles_spent": spent_cycles,
        "cycles_if_always_ceiling": ceiling_cycles,
        "cycle_reduction": round(1.0 - (spent_cycles / ceiling_cycles), 6) if ceiling_cycles else 0.0,
        "exit_reasons": exits,
    }
