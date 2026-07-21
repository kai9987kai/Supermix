from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from source.route_policy_ledger import (
    EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION,
    build_logging_support_envelope,
)
from source.route_policy_lab import (
    AUTO_AGENT_MODE_ORDER,
    POLICY_PROFILES,
    RoutePolicyProfile,
    analyze_route_policy,
    ordered_allowed_modes,
    propensity_readiness,
    select_profile_action,
)


def _usage(
    route_id,
    *,
    score,
    selected,
    allowed=None,
    cost=1.0,
    latency=100.0,
    policy_extra=None,
    status="completed",
    success=True,
    executed=None,
    fingerprint_valid=True,
    fingerprint_reason=None,
    outcome_contract_precommitted=True,
):
    policy = {
        "score": score,
        "selected_agent_mode": selected,
    }
    if allowed is not None:
        policy["allowed_agent_modes"] = allowed
    policy.update(policy_extra or {})
    row = {
        "route_id": route_id,
        "selected_agent_mode": selected,
        "executed_agent_mode": selected if executed is None else executed,
        "status": status,
        "success": success,
        "outcome_contracts_precommitted_at_begin": outcome_contract_precommitted,
        "auto_agent_policy": policy,
        "route_economics": {
            "actual": {
                "cost_units": cost,
                "elapsed_ms": latency,
            }
        },
    }
    if fingerprint_valid is not None:
        row["decision_record_fingerprint_valid"] = bool(fingerprint_valid)
        row["decision_record_fingerprint_reason"] = (
            fingerprint_reason
            or ("verified" if fingerprint_valid else "fingerprint_mismatch")
        )
    return row


def _feedback(route_id, quality):
    return {
        "route_id": route_id,
        "feedback_axes": {"quality": quality},
        "rating": "up" if quality == 1 else "down",
    }


def _stochastic_policy(*, selected, context, version="route-v2", probabilities=None):
    probabilities = probabilities or {"off": 0.5, "collective": 0.5}
    eligible_actions = list(probabilities)
    positive_actions = [action for action, probability in probabilities.items() if probability > 0.0]
    randomized = len(positive_actions) >= 2
    logging_support = build_logging_support_envelope(
        {
            "schema_version": "route-support-v1",
            "decision_type": "randomized" if randomized else "deterministic",
            "probability_stage": "post_filter",
            "sampler": {
                "name": "test-randomized-assignment" if randomized else "test-argmax",
                "version": "1",
                "exploration_rate": 0.5 if randomized else 0.0,
                "assignment_unit": "route",
                "assignment_commitment": (
                    f"{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:"
                    + "ab" * 32
                    if randomized
                    else None
                ),
            },
            "candidates": [{"action": action} for action in eligible_actions],
            "exclusions": [],
        },
        eligible_modes=eligible_actions,
        action_probabilities=probabilities,
        chosen_mode=selected,
    )
    return {
        "score": 2,
        "selected_agent_mode": selected,
        "allowed_agent_modes": ["off", "collective"],
        "policy_id": "auto-route-v2",
        "policy_version": version,
        "feature_schema_version": "route-context-v1",
        "decision_context": context,
        "decision_type": "stochastic",
        "probability_stage": "post_filter",
        "eligible_actions": eligible_actions,
        "post_filter_action_probabilities": probabilities,
        "logging_propensity": probabilities[selected],
        "logging_support": logging_support,
        "candidate_set_hash": logging_support["candidate_set_hash"],
        "distribution_hash": logging_support["distribution_hash"],
    }


def _durable_lifecycle(route_count: int):
    return {
        "lifecycle": {
            "counts": {
                "started": route_count,
                "completed": route_count,
                "failed": 0,
                "inflight": 0,
            }
        },
        "analysis_window": {"truncated": False},
    }


def _balanced_randomized_evidence(route_count: int = 40):
    assert route_count >= 4 and route_count % 4 == 0
    context = {"action_mode": "chat", "budget": "balanced"}
    usage_rows = []
    feedback_rows = []
    midpoint = route_count // 2
    for index in range(route_count):
        target_off = index < midpoint
        score = 1 if target_off else 2
        within_group = index if target_off else index - midpoint
        target = "off" if target_off else "collective"
        selected = target if within_group % 2 == 0 else ("collective" if target == "off" else "off")
        policy = _stochastic_policy(selected=selected, context=context)
        policy["score"] = score
        route_id = f"route-{index}"
        usage_rows.append(
            _usage(
                route_id,
                score=score,
                selected=selected,
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )
        feedback_rows.append(_feedback(route_id, 1 if index % 3 else -1))
    return context, usage_rows, feedback_rows


def test_builtin_profiles_are_immutable_and_monotonic() -> None:
    assert tuple(POLICY_PROFILES) == ("efficiency", "balanced", "quality_first")
    assert POLICY_PROFILES["balanced"].thresholds == {
        "off": 0,
        "collective": 2,
        "loop": 4,
        "collective_loop": 5,
    }
    with pytest.raises(TypeError):
        POLICY_PROFILES["new"] = RoutePolicyProfile("new", 1, 2, 3)
    with pytest.raises(TypeError):
        POLICY_PROFILES["balanced"].thresholds["loop"] = 99
    with pytest.raises(FrozenInstanceError):
        POLICY_PROFILES["balanced"].loop_min_score = 99
    with pytest.raises(ValueError):
        RoutePolicyProfile("bad", 4, 3, 5)


def test_profile_selection_preserves_neighbor_order_and_allowed_modes() -> None:
    assert AUTO_AGENT_MODE_ORDER == ("off", "collective", "loop", "collective_loop")
    assert ordered_allowed_modes(["loop", "off", "collective", "loop", "invalid"]) == (
        "off",
        "collective",
        "loop",
    )
    assert select_profile_action(1, AUTO_AGENT_MODE_ORDER, "balanced") == "off"
    assert select_profile_action(2, AUTO_AGENT_MODE_ORDER, "balanced") == "collective"
    assert select_profile_action(4, AUTO_AGENT_MODE_ORDER, "balanced") == "loop"
    assert select_profile_action(5, AUTO_AGENT_MODE_ORDER, "balanced") == "collective_loop"
    assert select_profile_action(4, ["off", "collective", "collective_loop"], "balanced") == "collective"
    assert select_profile_action(6, ["off", "collective", "collective_loop"], "balanced") == "collective_loop"
    assert select_profile_action(3, AUTO_AGENT_MODE_ORDER, "efficiency") == "collective"
    assert select_profile_action(3, AUTO_AGENT_MODE_ORDER, "quality_first") == "loop"
    assert select_profile_action("not-a-score", [], "balanced") == "off"


def test_exact_join_support_candidate_agreement_and_matched_observed_metrics() -> None:
    usage_rows = [
        _usage(
            "route-1",
            score=2,
            selected="collective",
            allowed=list(AUTO_AGENT_MODE_ORDER),
            cost=2.0,
            latency=120.0,
        ),
        _usage(
            "route-2",
            score=4,
            selected="off",
            allowed=list(AUTO_AGENT_MODE_ORDER),
            cost=1.0,
            latency=60.0,
        ),
        _usage("route-3", score=5, selected="collective_loop", cost=8.0, latency=900.0),
        {"route_id": "", "selected_agent_mode": "off"},
    ]
    feedback_rows = [
        _feedback("route-1", 1),
        {
            "route_id": "route-2",
            "feedback_axes": {"quality": None, "cost_pressure": 1},
            "rating": "down",
        },
        _feedback("feedback-only", -1),
    ]

    report = analyze_route_policy(usage_rows, feedback_rows, profile="balanced")

    assert report["analysis_kind"] == "associational_matched_route_replay"
    assert report["causal_interpretation"]["causal"] is False
    assert report["support"]["usage"] == {
        "rows": 4,
        "rows_with_route_id": 3,
        "rows_without_route_id": 1,
        "unique_route_ids": 3,
        "duplicate_route_ids": 0,
    }
    assert report["support"]["exact_joined_route_ids"] == 2
    assert report["support"]["exact_usage_join_coverage"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["support"]["exact_feedback_join_coverage"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["candidate_action_agreement"] == {
        "evaluable_routes": 2,
        "matched_routes": 1,
        "changed_routes": 1,
        "agreement_rate": 0.5,
    }
    matched = report["matched_observed"]
    assert matched["quality_sample_count"] == 1
    assert matched["approval_rate"] == 1.0
    assert matched["avg_cost_units"] == 2.0
    assert matched["avg_elapsed_ms"] == 120.0
    assert matched["causal"] is False


def test_historical_deterministic_logs_are_noncausal_and_block_promotion() -> None:
    report = analyze_route_policy(
        [_usage("route-1", score=4, selected="off")],
        [_feedback("route-1", 1)],
        profile="balanced",
    )

    readiness = report["propensity_readiness"]
    assert readiness["valid_routes"] == 0
    assert readiness["off_policy_estimate_computed"] is False
    assert readiness["invalid_reason_counts"]["expected_policy_version_required"] == 1
    assert readiness["invalid_reason_counts"]["expected_context_required"] == 1
    assert readiness["invalid_reason_counts"]["post_filter_probability_vector_missing"] == 1
    gate = report["promotion_gate"]
    assert gate["status"] == "blocked"
    assert gate["deployment"] == "shadow_only"
    assert gate["reason_code"] == "no_valid_randomized_overlap"
    assert gate["automatic_promotion_allowed"] is False
    assert gate["requires_validated_external_ope"] is True
    assert gate["causal_claim"] is False
    assert report["evaluation_readiness"]["target_overlap"]["effective_sample_size"] == 0.0
    assert all("off-policy estimate" not in warning.lower() or "not" in warning.lower() for warning in report["warnings"])


def test_propensity_readiness_requires_exact_version_context_and_positive_overlap() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    policy = _stochastic_policy(selected="collective", context=context)
    ready = propensity_readiness(
        policy,
        chosen_action="collective",
        target_action="off",
        expected_policy_version="route-v2",
        expected_context=context,
    )
    assert ready["ready"] is True
    assert ready["chosen_probability"] == 0.5
    assert ready["target_probability"] == 0.5
    assert ready["off_policy_estimate"] is False

    wrong_version = propensity_readiness(
        policy,
        chosen_action="collective",
        expected_policy_version="route-v3",
        expected_context=context,
    )
    assert "policy_version_mismatch" in wrong_version["reasons"]

    wrong_context = propensity_readiness(
        policy,
        chosen_action="collective",
        expected_policy_version="route-v2",
        expected_context={"action_mode": "image", "budget": "balanced"},
    )
    assert "decision_context_mismatch" in wrong_context["reasons"]

    deterministic = _stochastic_policy(
        selected="off",
        context=context,
        probabilities={"off": 1.0, "collective": 0.0},
    )
    no_overlap = propensity_readiness(
        deterministic,
        chosen_action="off",
        target_action="collective",
        expected_policy_version="route-v2",
        expected_context=context,
    )
    assert no_overlap["ready"] is False
    assert "insufficient_randomized_support" in no_overlap["reasons"]
    assert "target_action_has_no_positive_probability" in no_overlap["reasons"]


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"decision_type": "deterministic"}, "decision_not_explicitly_stochastic"),
        ({"probability_stage": "pre_filter"}, "probabilities_not_post_filter"),
        ({"eligible_actions": ["off"]}, "eligible_actions_invalid"),
        (
            {"post_filter_action_probabilities": {"off": 0.7, "collective": 0.7}},
            "probabilities_do_not_sum_to_one",
        ),
        (
            {"post_filter_action_probabilities": {"off": 1.0}},
            "probability_vector_not_exactly_eligible_actions",
        ),
    ],
)
def test_propensity_readiness_rejects_malformed_logging(mutation, reason) -> None:
    context = "chat:balanced"
    policy = _stochastic_policy(selected="off", context=context)
    policy.update(mutation)
    result = propensity_readiness(
        policy,
        chosen_action="off",
        expected_policy_version="route-v2",
        expected_context=context,
    )
    assert result["ready"] is False
    assert reason in result["reasons"]


def test_valid_propensity_rows_only_make_external_evaluation_input_ready() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    usage_rows = []
    feedback_rows = []
    for index, selected in enumerate(("off", "collective"), start=1):
        policy = _stochastic_policy(selected=selected, context=context)
        usage_rows.append(
            _usage(
                f"route-{index}",
                score=2,
                selected=selected,
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )
        feedback_rows.append(_feedback(f"route-{index}", 1))

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        profile="balanced",
        expected_policy_version="route-v2",
        expected_context=context,
        min_overlap_routes=2,
        min_global_effective_sample_size=1,
        min_per_action_effective_sample_size=1,
        durable_lifecycle=_durable_lifecycle(2),
    )

    assert report["propensity_readiness"]["valid_routes"] == 2
    assert report["propensity_readiness"]["ready_as_external_ope_input"] is True
    assert report["propensity_readiness"]["off_policy_estimate_computed"] is False
    assert report["candidate_action_agreement"]["changed_routes"] == 1
    assert report["promotion_gate"]["status"] == "external_ope_required"
    assert report["promotion_gate"]["deployment"] == "shadow_only"
    assert report["promotion_gate"]["automatic_promotion_allowed"] is False


def test_matched_approval_interval_is_reproducible_and_descriptive_only() -> None:
    usage_rows = []
    feedback_rows = []
    for index, quality in enumerate((1, 1, 1, -1), start=1):
        usage_rows.append(_usage(f"route-{index}", score=2, selected="collective"))
        feedback_rows.append(_feedback(f"route-{index}", quality))

    first = analyze_route_policy(usage_rows, feedback_rows)
    second = analyze_route_policy(usage_rows, feedback_rows)
    interval = first["matched_observed"]["approval_interval"]

    assert first == second
    assert first["matched_observed"]["approval_rate"] == 0.75
    assert interval["method"] == "descriptive_unweighted_wilson"
    assert interval["causal"] is False
    assert 0.0 < interval["lower_bound"] < 0.75 < interval["upper_bound"] < 1.0


def test_duplicate_route_ids_use_last_row_and_are_reported() -> None:
    usage_rows = [
        _usage("route-1", score=1, selected="off", cost=99.0),
        _usage("route-1", score=2, selected="collective", cost=3.0),
    ]
    report = analyze_route_policy(usage_rows, [_feedback("route-1", 1)])

    assert report["support"]["usage"]["duplicate_route_ids"] == 1
    assert report["candidate_action_agreement"]["matched_routes"] == 1
    assert report["matched_observed"]["avg_cost_units"] == 3.0


def test_invalid_profile_and_overlap_floor_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown route policy profile"):
        analyze_route_policy([], [], profile="turbo")
    with pytest.raises(ValueError, match="at least 1"):
        analyze_route_policy([], [], min_overlap_routes=0)


def test_probability_stage_is_required() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    policy = _stochastic_policy(selected="off", context=context)
    policy.pop("probability_stage")

    result = propensity_readiness(
        policy,
        chosen_action="off",
        target_action="collective",
        expected_policy_version="route-v2",
        expected_context=context,
    )

    assert result["ready"] is False
    assert "probabilities_not_post_filter" in result["reasons"]


def test_readiness_checks_all_usage_rows_not_only_feedback_join() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    usage_rows = []
    feedback_rows = []
    for index in range(120):
        route_id = f"route-{index}"
        if index < 20:
            selected = "collective" if index % 2 == 0 else "off"
            policy = _stochastic_policy(selected=selected, context=context)
            feedback_rows.append(_feedback(route_id, 1))
        else:
            selected = "off"
            policy = {"score": 2, "selected_agent_mode": selected}
        usage_rows.append(
            _usage(
                route_id,
                score=2,
                selected=selected,
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_version="route-v2",
        expected_context=context,
        min_global_effective_sample_size=5,
        min_per_action_effective_sample_size=5,
    )
    integrity = report["evaluation_readiness"]["logging_integrity"]

    assert integrity["checked_usage_routes"] == 120
    assert integrity["valid_routes"] == 20
    assert integrity["invalid_routes"] == 100
    assert integrity["valid_rate"] == pytest.approx(1 / 6, abs=1e-6)
    assert report["evaluation_readiness"]["outcome_observation"]["quality_observed_routes"] == 20
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False
    assert "logging_integrity_incomplete" in report["promotion_gate"]["blocking_reason_codes"]


def test_zero_candidate_matches_yield_zero_overlap_ess() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    probabilities = {"off": 0.999, "collective": 0.001}
    usage_rows = []
    feedback_rows = []
    for index in range(20):
        route_id = f"route-{index}"
        policy = _stochastic_policy(
            selected="off",
            context=context,
            probabilities=probabilities,
        )
        usage_rows.append(
            _usage(
                route_id,
                score=2,
                selected="off",
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )
        feedback_rows.append(_feedback(route_id, 1))

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_version="route-v2",
        expected_context=context,
    )
    overlap = report["evaluation_readiness"]["target_overlap"]

    assert report["evaluation_readiness"]["logging_integrity"]["valid_routes"] == 20
    assert overlap["effective_sample_size"] == 0.0
    assert overlap["nonzero_weight_routes"] == 0
    assert overlap["minimum_target_probability"] == 0.001
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False
    assert "insufficient_global_overlap_ess" in report["promotion_gate"]["blocking_reason_codes"]


def test_per_action_overlap_blocks_when_global_ess_hides_unseen_action() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    usage_rows = []
    feedback_rows = []
    for index in range(20):
        target_off = index >= 10
        score = 1 if target_off else 2
        selected = "collective"
        policy = _stochastic_policy(selected=selected, context=context)
        policy["score"] = score
        route_id = f"route-{index}"
        usage_rows.append(
            _usage(
                route_id,
                score=score,
                selected=selected,
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )
        feedback_rows.append(_feedback(route_id, 1))

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_version="route-v2",
        expected_context=context,
        min_global_effective_sample_size=5,
        min_per_action_effective_sample_size=5,
    )
    overlap = report["evaluation_readiness"]["target_overlap"]

    assert overlap["effective_sample_size"] == 10.0
    assert overlap["per_action"]["collective"]["effective_sample_size"] == 10.0
    assert overlap["per_action"]["off"]["effective_sample_size"] == 0.0
    assert overlap["weakest_target_action"] == "off"
    assert "insufficient_per_action_overlap" in report["promotion_gate"]["blocking_reason_codes"]


def test_unknown_quality_missingness_blocks_external_ope_readiness() -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    usage_rows = []
    feedback_rows = []
    for index in range(40):
        route_id = f"route-{index}"
        selected = "collective" if index % 2 == 0 else "off"
        policy = _stochastic_policy(selected=selected, context=context)
        usage_rows.append(
            _usage(
                route_id,
                score=2,
                selected=selected,
                allowed=["off", "collective"],
                policy_extra=policy,
            )
        )
        if index < 10:
            feedback_rows.append(_feedback(route_id, 1))

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_version="route-v2",
        expected_context=context,
        min_global_effective_sample_size=10,
        min_per_action_effective_sample_size=10,
    )
    outcome = report["evaluation_readiness"]["outcome_observation"]

    assert outcome["quality_observed_routes"] == 10
    assert outcome["quality_missing_routes"] == 30
    assert outcome["observation_propensity_logged"] is False
    assert outcome["ready"] is False
    assert "unknown_quality_observation_process" in report["promotion_gate"]["blocking_reason_codes"]


def test_balanced_overlap_advances_only_to_external_ope_required() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    readiness = report["evaluation_readiness"]

    assert readiness["target_overlap"]["effective_sample_size"] == 20.0
    assert readiness["target_overlap"]["per_action"]["off"]["effective_sample_size"] == 10.0
    assert readiness["target_overlap"]["per_action"]["collective"]["effective_sample_size"] == 10.0
    assert readiness["ready_for_external_ope"] is True
    assert readiness["policy_value_estimated"] is False
    assert report["promotion_gate"]["status"] == "external_ope_required"
    assert report["promotion_gate"]["automatic_promotion_allowed"] is False


def test_route_specific_context_integrity_detects_mismatch() -> None:
    first_context = {"action_mode": "chat", "budget": "balanced", "score": 2}
    second_context = {"action_mode": "chat", "budget": "balanced", "score": 3}
    first = _stochastic_policy(selected="collective", context=first_context)
    second = _stochastic_policy(selected="collective", context=second_context)
    usage_rows = [
        _usage("route-1", score=2, selected="collective", policy_extra=first),
        _usage("route-2", score=2, selected="collective", policy_extra=second),
    ]

    report = analyze_route_policy(
        usage_rows,
        [_feedback("route-1", 1), _feedback("route-2", 1)],
        expected_policy_version="route-v2",
        expected_context_by_route_id={
            "route-1": first_context,
            "route-2": {"action_mode": "chat", "budget": "balanced", "score": 99},
        },
        min_global_effective_sample_size=1,
        min_per_action_effective_sample_size=1,
    )
    integrity = report["evaluation_readiness"]["logging_integrity"]

    assert integrity["valid_routes"] == 1
    assert integrity["invalid_routes"] == 1
    assert integrity["invalid_reason_counts"]["decision_context_mismatch"] == 1
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_legacy_compatibility_evidence_never_fabricates_lifecycle_readiness() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=None,
    )

    lifecycle = report["evaluation_readiness"]["lifecycle_integrity"]
    assert lifecycle["durable_lifecycle_present"] is False
    assert lifecycle["reconciled"] is False
    assert report["promotion_gate"]["checks"]["durable_lifecycle_present"] is False
    assert "durable_lifecycle_required" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_minimum_overlap_route_floor_is_enforced() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_overlap_routes=1000,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )

    assert report["promotion_gate"]["checks"]["minimum_overlap_routes_met"] is False
    assert "insufficient_overlap_routes" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("missing", "logging_support_missing"),
        ("commitment", "assignment_commitment_missing"),
        ("candidate_hash", "candidate_set_hash_mismatch"),
        ("distribution_hash", "distribution_hash_mismatch"),
    ],
)
def test_randomized_labels_require_a_verified_support_envelope(mutation, reason) -> None:
    context = {"action_mode": "chat", "budget": "balanced"}
    policy = _stochastic_policy(selected="collective", context=context)
    if mutation == "missing":
        policy.pop("logging_support")
    elif mutation == "commitment":
        policy["logging_support"] = deepcopy(policy["logging_support"])
        policy["logging_support"]["sampler"]["assignment_commitment"] = None
    elif mutation == "candidate_hash":
        policy["logging_support"] = deepcopy(policy["logging_support"])
        policy["logging_support"]["candidate_set_hash"] = "0" * 64
    else:
        policy["logging_support"] = deepcopy(policy["logging_support"])
        policy["logging_support"]["distribution_hash"] = "0" * 64

    result = propensity_readiness(
        policy,
        chosen_action="collective",
        target_action="off",
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
    )

    assert result["ready"] is False
    assert reason in result["reasons"]


def test_unversioned_observation_propensity_cannot_clear_missing_quality() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    for row in feedback_rows:
        row.pop("feedback_axes", None)
        row["rating"] = ""
        row["observation_propensity"] = 0.75

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    observation = report["evaluation_readiness"]["outcome_observation"]

    assert observation["raw_observation_propensity_routes"] == 40
    assert observation["versioned_observation_propensity_routes"] == 0
    assert observation["observation_propensity_logged"] is False
    assert observation["ready"] is False
    assert "unknown_quality_observation_process" in report["promotion_gate"]["blocking_reason_codes"]


@pytest.mark.parametrize("duplicate_population", ["usage", "feedback"])
def test_duplicate_route_ids_in_durable_population_fail_closed(duplicate_population) -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    if duplicate_population == "usage":
        usage_rows.append(deepcopy(usage_rows[0]))
    else:
        feedback_rows.append(deepcopy(feedback_rows[0]))

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    population = report["evaluation_readiness"]["population_integrity"]

    assert population[f"duplicate_{duplicate_population}_route_ids"] == 1
    assert population["complete"] is False
    assert report["promotion_gate"]["checks"]["population_integrity_complete"] is False
    assert "duplicate_route_ids_in_durable_population" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


@pytest.mark.parametrize(
    ("mutation", "diagnostic"),
    [
        ("missing_score", "missing_or_invalid_score_routes"),
        ("invalid_score", "missing_or_invalid_score_routes"),
        ("missing_mode", "missing_or_invalid_chosen_mode_routes"),
        ("invalid_mode", "missing_or_invalid_chosen_mode_routes"),
    ],
)
def test_unevaluable_route_in_durable_population_fails_closed(mutation, diagnostic) -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    row = usage_rows[0]
    if mutation == "missing_score":
        row["auto_agent_policy"].pop("score")
    elif mutation == "invalid_score":
        row["auto_agent_policy"]["score"] = "not-a-score"
    elif mutation == "missing_mode":
        row.pop("selected_agent_mode")
        row["auto_agent_policy"].pop("selected_agent_mode")
    else:
        row["selected_agent_mode"] = "not-a-mode"
        row["auto_agent_policy"]["selected_agent_mode"] = "not-a-mode"

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    population = report["evaluation_readiness"]["population_integrity"]

    assert population["unevaluable_usage_routes"] == 1
    assert population[diagnostic] == 1
    assert population["complete"] is False
    assert "unevaluable_routes_in_durable_population" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


@pytest.mark.parametrize(
    ("status", "success", "completed", "failed", "inflight"),
    [
        ("failed", False, 39, 1, 0),
        ("inflight", None, 39, 0, 1),
    ],
)
def test_quality_on_non_successful_durable_route_is_rejected(
    status, success, completed, failed, inflight
) -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    usage_rows[0]["status"] = status
    usage_rows[0]["success"] = success
    feedback_rows[0]["route_status"] = status
    lifecycle = {
        "lifecycle": {
            "counts": {
                "started": 40,
                "completed": completed,
                "failed": failed,
                "inflight": inflight,
            }
        },
        "analysis_window": {"truncated": False},
    }

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=lifecycle,
    )
    outcome = report["evaluation_readiness"]["outcome_observation"]

    assert outcome["raw_quality_outcome_routes"] == 40
    assert outcome["quality_observed_routes"] == 39
    assert outcome["invalid_quality_outcome_routes"] == 1
    assert outcome["quality_on_non_successful_routes"] == 1
    assert outcome["evidence_integrity_complete"] is False
    assert report["promotion_gate"]["checks"]["outcome_evidence_integrity"] is False
    assert "quality_outcome_on_ineligible_route" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_chosen_and_executed_mode_mismatch_fails_execution_and_outcome_integrity() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    chosen = usage_rows[0]["selected_agent_mode"]
    usage_rows[0]["executed_agent_mode"] = "collective" if chosen == "off" else "off"

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    execution = report["evaluation_readiness"]["execution_integrity"]
    outcome = report["evaluation_readiness"]["outcome_observation"]

    assert execution["chosen_executed_mode_mismatch_routes"] == 1
    assert execution["unverified_successful_execution_routes"] == 1
    assert execution["complete"] is False
    assert outcome["quality_on_unverified_execution_routes"] == 1
    assert report["promotion_gate"]["checks"]["execution_integrity_complete"] is False
    assert "chosen_executed_mode_mismatch" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_zero_quality_outcomes_never_certify_with_versioned_observation_propensities() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    for row in feedback_rows:
        row.pop("feedback_axes", None)
        row["rating"] = ""
        row["observation_propensity"] = 0.75
        row["observation_policy_id"] = "quality-observation-v1"
        row["observation_policy_version"] = "1"
        row["outcome_definition_version"] = "quality-v1"

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        min_global_effective_sample_size=20,
        min_per_action_effective_sample_size=10,
        durable_lifecycle=_durable_lifecycle(40),
    )
    observation = report["evaluation_readiness"]["outcome_observation"]

    assert observation["observation_propensity_logged"] is True
    assert observation["versioned_observation_propensity_routes"] == 40
    assert observation["quality_observed_routes"] == 0
    assert observation["has_observed_quality_outcomes"] is False
    assert observation["evidence_integrity_complete"] is True
    assert observation["ready"] is False
    assert report["promotion_gate"]["checks"]["quality_observation_ready"] is False
    assert "no_observed_quality_outcomes" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False
    assert report["evaluation_readiness"]["policy_value_estimated"] is False
    assert report["promotion_gate"]["automatic_promotion_allowed"] is False


def test_invalid_decision_record_fingerprint_blocks_durable_readiness() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    usage_rows[0]["decision_record_fingerprint_valid"] = False
    usage_rows[0]["decision_record_fingerprint_reason"] = "fingerprint_mismatch"

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        durable_lifecycle=_durable_lifecycle(len(usage_rows)),
    )

    integrity = report["evaluation_readiness"]["logging_integrity"]
    assert integrity["decision_record_fingerprint_complete"] is False
    assert integrity["decision_record_fingerprint_valid_routes"] == len(usage_rows) - 1
    assert integrity["decision_record_fingerprint_invalid_reason_counts"] == {
        "fingerprint_mismatch": 1,
    }
    assert report["promotion_gate"]["checks"]["logging_integrity_complete"] is False
    assert "decision_record_fingerprint_invalid" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_missing_precommitted_outcome_contract_blocks_durable_readiness() -> None:
    context, usage_rows, feedback_rows = _balanced_randomized_evidence()
    usage_rows[0]["outcome_contracts_precommitted_at_begin"] = False

    report = analyze_route_policy(
        usage_rows,
        feedback_rows,
        expected_policy_id="auto-route-v2",
        expected_policy_version="route-v2",
        expected_feature_schema="route-context-v1",
        expected_context=context,
        durable_lifecycle=_durable_lifecycle(len(usage_rows)),
    )

    observation = report["evaluation_readiness"]["outcome_observation"]
    assert observation["outcome_contract_required"] is True
    assert observation["precommitted_outcome_contract_routes"] == len(usage_rows) - 1
    assert observation["missing_or_invalid_outcome_contract_routes"] == 1
    assert observation["outcome_contract_integrity_complete"] is False
    assert observation["evidence_integrity_complete"] is False
    assert report["promotion_gate"]["checks"]["outcome_evidence_integrity"] is False
    assert "outcome_contract_not_precommitted" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False
    assert report["promotion_gate"]["automatic_promotion_allowed"] is False
