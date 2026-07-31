"""Tests for the MiMoMix adaptive thinking controller.

The controller's whole claim is that it saves compute *without* changing the
decision. So the tests care about two things: that the plan is deterministic and
bounded, and that the accepted output is literally the output the model produced
at the accepted budget -- never a blend, never a patch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_controller as ctl  # noqa: E402
import mimomix_core as mc  # noqa: E402


def build(seed: int = 0, **overrides) -> mc.MiMoMixModel:
    torch.manual_seed(seed)
    base = dict(
        vocab_size=48,
        hidden_size=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        sliding_window=4,
        hybrid_ratio=2,
        native_context=16,
        max_position_embeddings=64,
        n_routed_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        n_mtp_layers=1,
        thinking_cycles=2,
        thinking_max_cycles=8,
    )
    base.update(overrides)
    model = mc.MiMoMixModel(mc.MiMoMixConfig(**base))
    model.eval()
    return model


def responsive(seed: int = 0, scale: float = 0.5, **overrides) -> mc.MiMoMixModel:
    """A model whose thinking core actually changes the output per budget.

    Fresh cores are near-identity by design, which would make every budget agree
    trivially. Raising the residual scale gives the controller something real to
    disagree about.
    """

    model = build(seed, **overrides)
    with torch.no_grad():
        model.thinking_core.residual_scale.fill_(scale)
    return model


# ---------------------------------------------------------------------------
# Features and planning
# ---------------------------------------------------------------------------


def test_feature_scores_are_bounded_and_deterministic():
    features = ctl.RequestFeatures(
        prompt_tokens=100000, requested_acts=99, tool_calls_available=99,
        has_conflict=True, needs_evidence=True, max_output_tokens=100000,
    )
    assert 0.0 <= features.difficulty() <= 1.0
    assert 0.0 <= features.epistemic_risk() <= 1.0
    assert features.difficulty() == features.difficulty()

    empty = ctl.RequestFeatures()
    assert empty.difficulty() >= 0.0 and empty.epistemic_risk() == 0.0


def test_mode_routing_follows_the_declared_rules():
    assert ctl.plan_request(ctl.RequestFeatures(prompt_tokens=10)).mode == "fast"
    assert ctl.plan_request(ctl.RequestFeatures(prompt_tokens=200, tool_calls_available=3)).mode == "agent"
    assert ctl.plan_request(
        ctl.RequestFeatures(prompt_tokens=4000, requested_acts=3, needs_evidence=True)
    ).mode == "deep"


def test_safety_critical_turns_are_not_forced_into_deep_compute():
    """v52.1 rule: urgent-help guidance must not wait on a big budget."""

    plan = ctl.plan_request(
        ctl.RequestFeatures(prompt_tokens=4000, requested_acts=4, safety_critical=True)
    )
    assert plan.mode == "fast"
    assert plan.ceiling_cycles <= 2
    assert "safety_ceiling" in plan.reasons


def test_explicit_mode_overrides_inference():
    plan = ctl.plan_request(ctl.RequestFeatures(prompt_tokens=5), mode="agent")
    assert plan.mode == "agent"
    assert "mode_explicitly_requested" in plan.reasons
    with pytest.raises(ValueError):
        ctl.plan_request(ctl.RequestFeatures(), mode="turbo")


def test_ladder_is_sorted_bounded_and_ends_at_the_ceiling():
    for features in (
        ctl.RequestFeatures(prompt_tokens=5),
        ctl.RequestFeatures(prompt_tokens=9000, tool_calls_available=8, has_conflict=True),
        ctl.RequestFeatures(prompt_tokens=300, needs_evidence=True),
    ):
        plan = ctl.plan_request(features, max_model_cycles=8)
        assert list(plan.ladder) == sorted(plan.ladder)
        assert plan.ladder[0] >= plan.floor_cycles
        assert plan.ladder[-1] == plan.ceiling_cycles
        assert plan.floor_cycles <= plan.ceiling_cycles


def test_ladder_never_exceeds_what_the_model_can_run():
    plan = ctl.plan_request(ctl.RequestFeatures(tool_calls_available=8), max_model_cycles=2)
    assert plan.ceiling_cycles <= 2
    assert max(plan.ladder) <= 2


def test_plan_is_json_safe():
    json.dumps(ctl.plan_request(ctl.RequestFeatures(prompt_tokens=42)).to_dict())


# ---------------------------------------------------------------------------
# Decision probes
# ---------------------------------------------------------------------------


def test_decision_signature_reports_ordered_topk_and_both_margins():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    ordered, confidence, entropy, decision, boundary = ctl._decision_signature(logits, rank_depth=3)
    assert ordered == (0, 1, 2)
    assert confidence == pytest.approx(0.5, abs=1e-6)
    assert decision == pytest.approx(0.5 - 0.3, abs=1e-6)
    assert boundary == pytest.approx(0.15 - 0.05, abs=1e-6)
    assert entropy > 0.0


def test_a_saturated_distribution_has_a_wide_decision_margin_and_a_flat_boundary():
    """The exact case that made a boundary-margin gate block confident turns."""

    probs = torch.tensor([0.9999, 3e-5, 2e-5, 1e-5, 5e-6])
    probs = probs / probs.sum()
    _, confidence, entropy, decision, boundary = ctl._decision_signature(
        torch.log(probs), rank_depth=3
    )
    assert confidence > 0.999 and entropy < 0.01
    assert decision > 0.99, "the emitted token is maximally safe"
    assert boundary < 1e-4, "yet the tail ordering is a coin flip"


def test_the_gate_uses_the_decision_margin_not_the_tail():
    """A confident model must be allowed to exit; the tail must not veto it."""

    torch.manual_seed(0)
    model = build(0, tie_word_embeddings=False)
    ids = torch.randint(0, 48, (1, 8))
    with torch.no_grad():
        # Point one output row along the actual final hidden state, so that
        # token's logit dominates regardless of the hidden state's sign. The
        # rest are zeroed, giving ~1.0 on rank 1 and a flat, arbitrary tail --
        # exactly the shape that a boundary-margin gate would wrongly veto.
        hidden = model(ids, thinking_cycles=1, return_mtp=False, past_length=0).hidden_states
        direction = hidden[0, -1]
        model.lm_head.weight.zero_()
        model.lm_head.weight[3] = 40.0 * direction / direction.norm()
        probe = model(ids, thinking_cycles=1, return_mtp=False, past_length=0)
    _, confidence, entropy, decision, boundary = ctl._decision_signature(probe.logits[0, -1], 3)
    assert confidence > 0.9 and entropy < 1.0, "test setup failed to saturate"
    assert decision > 0.9 and boundary < 1e-3, "test setup lacks the flat tail"

    permissive = dict(ladder=(1, 2, 4, 8), confidence_target=0.5, entropy_target=1.0,
                      continue_threshold=1.01)
    _, allowed = ctl.decide(model, ids, mode="agent", policy=ctl.ThinkingPolicy(**permissive))
    _, blocked = ctl.decide(
        model, ids, mode="agent", policy=ctl.ThinkingPolicy(boundary_margin=0.1, **permissive)
    )
    assert allowed.exit_reason == "cross_budget_agreement"
    assert blocked.exit_reason == "ceiling_budget"


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_accepted_output_is_the_probe_not_a_blend():
    """Decision safety: the returned tensor must equal a real budget's output."""

    model = responsive(1)
    ids = torch.randint(0, 48, (1, 8))
    output, decision = ctl.decide(model, ids)
    with torch.no_grad():
        direct = model(ids, thinking_cycles=decision.accepted_budget,
                       adaptive_thinking=False, return_mtp=False, past_length=0)
    assert torch.equal(output.logits, direct.logits)
    assert decision.reused_accepted_probe is True


def test_a_flat_distribution_never_earns_an_early_exit():
    """An untrained model has near-maximal entropy; the gates must hold."""

    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    _, decision = ctl.decide(model, ids, policy=ctl.ThinkingPolicy(entropy_target=0.1))
    assert decision.exit_reason == "ceiling_budget"
    assert decision.accepted_budget == decision.plan.ceiling_cycles


def test_relaxed_targets_allow_a_cross_budget_early_exit():
    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    policy = ctl.ThinkingPolicy(
        ladder=(1, 2, 4, 8),
        confidence_target=0.0,
        entropy_target=99.0,
        continue_threshold=1.01,
        decision_margin=0.0,
    )
    _, decision = ctl.decide(model, ids, mode="agent", policy=policy)
    assert decision.exit_reason == "cross_budget_agreement"
    assert decision.accepted_budget < decision.plan.ceiling_cycles
    assert decision.cycle_reduction > 0.0


def test_the_verifier_can_veto_but_never_authorise_an_exit():
    """continue_probability >= threshold must block, even with everything else met."""

    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    permissive = dict(
        ladder=(1, 2, 4, 8), confidence_target=0.0, entropy_target=99.0, decision_margin=0.0
    )
    _, allowed = ctl.decide(
        model, ids, mode="agent", policy=ctl.ThinkingPolicy(continue_threshold=1.01, **permissive)
    )
    _, vetoed = ctl.decide(
        model, ids, mode="agent", policy=ctl.ThinkingPolicy(continue_threshold=0.0, **permissive)
    )
    assert allowed.accepted_budget < allowed.plan.ceiling_cycles
    assert vetoed.exit_reason == "ceiling_budget"


def test_first_rung_can_never_exit_when_agreement_is_required():
    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    policy = ctl.ThinkingPolicy(
        ladder=(1, 2, 4, 8), confidence_target=0.0, entropy_target=99.0,
        continue_threshold=1.01, decision_margin=0.0, require_cross_budget_agreement=True,
    )
    _, decision = ctl.decide(model, ids, mode="agent", policy=policy)
    # the mode floor prunes the ladder, so compare against the *planned* first rung
    assert decision.accepted_budget > decision.plan.ladder[0]


def test_agreement_can_be_disabled_for_a_single_budget_exit():
    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    policy = ctl.ThinkingPolicy(
        ladder=(1, 2, 4, 8), confidence_target=0.0, entropy_target=99.0,
        continue_threshold=1.01, decision_margin=0.0, require_cross_budget_agreement=False,
    )
    _, decision = ctl.decide(model, ids, mode="agent", policy=policy)
    assert decision.exit_reason == "single_budget_targets_met"
    assert decision.accepted_budget == decision.plan.ladder[0]


def test_a_single_rung_ladder_costs_exactly_one_forward():
    model = build(0)
    ids = torch.randint(0, 48, (1, 6))
    plan = ctl.ThinkingPlan(
        mode="fast", difficulty=0.0, epistemic_risk=0.0,
        floor_cycles=2, ceiling_cycles=2, ladder=(2,),
    )
    _, decision = ctl.decide(model, ids, plan=plan)
    assert decision.forward_evaluations == 1
    assert decision.cycle_reduction == 0.0


def test_cycle_reduction_is_signed_and_honest():
    """A ladder that runs to exhaustion spent more than a fixed ceiling would."""

    model = build(0)
    ids = torch.randint(0, 48, (1, 10))
    _, decision = ctl.decide(model, ids, mode="agent", policy=ctl.ThinkingPolicy(entropy_target=0.0))
    assert decision.cycles_spent > decision.plan.ceiling_cycles
    assert decision.cycle_reduction < 0.0


def test_decision_is_deterministic():
    model = responsive(2)
    ids = torch.randint(0, 48, (1, 9))
    first = ctl.decide(model, ids)[1].to_dict()
    second = ctl.decide(model, ids)[1].to_dict()
    assert first == second


def test_decision_is_json_safe():
    model = build(0)
    _, decision = ctl.decide(model, torch.randint(0, 48, (1, 7)))
    payload = json.loads(json.dumps(decision.to_dict()))
    assert payload["probes"] and payload["plan"]["mode"]


# ---------------------------------------------------------------------------
# Fidelity audit
# ---------------------------------------------------------------------------


def test_audit_reports_fidelity_and_pays_for_the_counterfactual():
    model = responsive(3)
    requests = [torch.randint(0, 48, (1, 6 + i)) for i in range(8)]
    report = ctl.audit_decision_fidelity(model, requests)
    assert report["requests"] == 8
    assert 0.0 <= report["top1_fidelity"] <= 1.0
    assert report["ordered_topk_fidelity"] <= report["top1_fidelity"]
    assert report["cycles_if_always_ceiling"] > 0
    assert sum(report["exit_reasons"].values()) == 8


def test_ceiling_only_ladder_has_perfect_fidelity_by_construction():
    """If the accepted budget is always the ceiling, it cannot disagree with it."""

    model = responsive(4)
    requests = [torch.randint(0, 48, (1, 8)) for _ in range(5)]
    report = ctl.audit_decision_fidelity(
        model, requests, policy=ctl.ThinkingPolicy(entropy_target=0.0)
    )
    assert report["top1_disagreements"] == 0
    assert report["ordered_topk_disagreements"] == 0
