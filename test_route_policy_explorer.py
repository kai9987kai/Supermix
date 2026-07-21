import builtins
import copy
import json
import math

import pytest

from source.route_policy_explorer import (
    ACTIVATION_BLOCKERS,
    ASSIGNMENT_ALGORITHM,
    AUTO_AGENT_MODE_ORDER,
    DEFAULT_ASSUMED_FEEDBACK_RATE,
    DEFAULT_EXPLORATION_RATE,
    DEFAULT_TARGET_OBSERVED_LABELS,
    MIN_POSITIVE_EXPLORATION_PROBABILITY,
    REHEARSAL_RECEIPT_SCHEMA_VERSION,
    REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION,
    STUDY_ASSIGNMENT_SCHEMA_VERSION,
    STUDY_ID,
    STUDY_PLAN_SCHEMA_VERSION,
    STUDY_VERSION,
    _minimum_simultaneous_label_routes,
    _multinomial_at_least_each_probability,
    assign_adjacent_route,
    plan_adjacent_route_study,
    validate_adjacent_route_study,
)


_SOURCE_CONTRACT = {
    "policy_id": "auto-route-v2",
    "policy_version": "2.0.0",
    "feature_schema_version": "route-context-v1",
    "support_schema_version": "route-support-v1",
    "candidate_set_hash": "a" * 64,
    "distribution_hash": "b" * 64,
    "outcome_contract_schema_version": "route-outcome-contract-v1",
}


_ECONOMICS = {
    "off": (1.0, 1, 0, "low"),
    "collective": (3.0, 3, 0, "moderate"),
    "loop": (12.0, 12, 4, "frontier"),
    "collective_loop": (24.0, 24, 4, "frontier"),
}


def _candidate(action, baseline="collective"):
    cost, calls, steps, tier = _ECONOMICS[action]
    return {
        "action": action,
        "estimated_cost_units": cost,
        "estimated_model_calls": calls,
        "planned_loop_steps": steps,
        "latency_tier": tier,
        "selected": action == baseline,
    }


def _partition(*, baseline="collective", feasible=AUTO_AGENT_MODE_ORDER):
    feasible_set = set(feasible)
    candidates = [_candidate(action, baseline) for action in AUTO_AGENT_MODE_ORDER if action in feasible_set]
    exclusions = [
        {"action": action, "reasons": ["capability_or_policy_filter"]}
        for action in AUTO_AGENT_MODE_ORDER
        if action not in feasible_set
    ]
    return candidates, exclusions


def _plan(
    *,
    baseline="collective",
    feasible=AUTO_AGENT_MODE_ORDER,
    source_contract=None,
    **kwargs,
):
    candidates, exclusions = _partition(baseline=baseline, feasible=feasible)
    return plan_adjacent_route_study(
        baseline,
        candidates,
        exclusions,
        source_contract=source_contract or _SOURCE_CONTRACT,
        **kwargs,
    )


def test_default_plan_is_adjacent_canonical_prompt_free_and_shadow_only():
    plan = _plan()

    assert ACTIVATION_BLOCKERS == (
        "target_policy_class_not_precommitted",
        "outcome_observation_process_not_validated",
        "population_definition_not_precommitted",
        "session_carryover_not_addressed",
        "interference_not_addressed",
        "stopping_rules_not_precommitted",
        "preassignment_seed_commitment_not_sealed",
        "external_ope_not_validated",
    )

    assert plan["schema_version"] == STUDY_PLAN_SCHEMA_VERSION
    assert plan["study"] == {
        "study_id": STUDY_ID,
        "study_version": STUDY_VERSION,
        "label": "Bounded-Exposure Adjacent-Route Rehearsal v1",
        "design_kind": "bounded_exposure_adjacent_rehearsal",
    }
    assert len(plan["design_hash"]) == 64
    int(plan["design_hash"], 16)

    charter = plan["charter"]
    assert charter["source_contract"] == _SOURCE_CONTRACT
    assert charter["prompt_free_contract"] == {
        "prompt_free": True,
        "raw_prompt_included": False,
        "raw_session_id_included": False,
        "free_form_text_fields_allowed": False,
        "canonical_json": "sorted_keys_compact_utf8_no_nan",
    }
    assert charter["enrollment"]["eligible"] is True
    assert charter["enrollment"]["baseline_action"] == "collective"
    assert charter["enrollment"]["adjacent_feasible_actions"] == ["off", "loop"]
    design = charter["probability_design"]
    assert design["decision_type"] == "randomized"
    assert design["probability_stage"] == "post_filter"
    assert design["requested_exploration_rate"] == DEFAULT_EXPLORATION_RATE
    assert design["applied_exploration_rate"] == DEFAULT_EXPLORATION_RATE
    assert design["eligible_actions"] == ["off", "collective", "loop"]
    assert design["action_probabilities"] == {
        "off": 0.05,
        "collective": 0.9,
        "loop": 0.05,
    }
    assert design["minimum_positive_exploration_probability"] == MIN_POSITIVE_EXPLORATION_PROBABILITY
    assert sum(design["action_probabilities"].values()) == pytest.approx(1.0)
    assert [row["action"] for row in charter["candidates"]] == ["off", "collective", "loop"]
    assert charter["exclusions"] == [
        {
            "action": "collective_loop",
            "reasons": ["nonadjacent_exploration_guard"],
            "post_filter_feasible": True,
        }
    ]
    boundaries = charter["causal_boundaries"]
    assert boundaries["deployment"] == "shadow_only"
    assert boundaries["execution_enabled"] is False
    assert boundaries["execution_performed"] is False
    assert boundaries["io_performed"] is False
    assert boundaries["off_policy_estimate_computed"] is False
    assert boundaries["automatic_promotion_allowed"] is False
    assert boundaries["baseline_performance_guarantee"] is False
    assert boundaries["logging_policy_optimality_claim"] is False
    assert boundaries["ledger_eligible"] is False
    assert boundaries["preassignment_commitment_sealed"] is False
    assert boundaries["nonce_grinding_resistant"] is False
    assert boundaries["activation_blockers"] == list(ACTIVATION_BLOCKERS)
    json.dumps(plan, sort_keys=True, allow_nan=False)


def test_plan_is_order_invariant_and_does_not_mutate_inputs():
    candidates, exclusions = _partition()
    candidates.reverse()
    before_candidates = copy.deepcopy(candidates)
    before_exclusions = copy.deepcopy(exclusions)

    shuffled = plan_adjacent_route_study(
        "collective", candidates, exclusions, source_contract=_SOURCE_CONTRACT
    )
    canonical = _plan()

    assert shuffled == canonical
    assert candidates == before_candidates
    assert exclusions == before_exclusions


def test_plan_validation_survives_json_javascript_integral_float_transport():
    plan = _plan()

    def javascript_numbers(value):
        if isinstance(value, dict):
            return {key: javascript_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [javascript_numbers(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    transported = javascript_numbers(json.loads(json.dumps(plan)))
    assert isinstance(plan["charter"]["candidates"][0]["estimated_cost_units"], float)
    assert isinstance(
        transported["charter"]["candidates"][0]["estimated_cost_units"], int
    )

    validated = validate_adjacent_route_study(transported)
    assert validated["design_hash"] == plan["design_hash"]
    assert validated["plan"] == plan

    tampered = copy.deepcopy(transported)
    tampered["charter"]["candidates"][0]["estimated_model_calls"] += 1
    with pytest.raises(ValueError, match="design_hash does not match"):
        validate_adjacent_route_study(tampered)


def test_source_contract_is_strict_and_bound_into_the_design_hash():
    baseline = _plan()
    changed_contract = {**_SOURCE_CONTRACT, "policy_version": "2.0.1"}
    changed = _plan(source_contract=changed_contract)

    assert baseline["charter"]["source_contract"] == _SOURCE_CONTRACT
    assert changed["charter"]["source_contract"] == changed_contract
    assert baseline["design_hash"] != changed["design_hash"]

    missing = dict(_SOURCE_CONTRACT)
    missing.pop("candidate_set_hash")
    with pytest.raises(ValueError, match="every required provenance field"):
        _plan(source_contract=missing)
    malformed = {**_SOURCE_CONTRACT, "distribution_hash": "ABC"}
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _plan(source_contract=malformed)
    stale_schema = {**_SOURCE_CONTRACT, "support_schema_version": "route-support-v0"}
    with pytest.raises(ValueError, match="must be route-support-v1"):
        _plan(source_contract=stale_schema)

    tampered = copy.deepcopy(baseline)
    tampered["charter"]["source_contract"]["policy_version"] = "9.9.9"
    with pytest.raises(ValueError, match="design_hash"):
        assign_adjacent_route(tampered, "source-contract-tamper-nonce")


def test_resource_and_traffic_forecasts_are_expected_and_not_power_claims():
    plan = _plan(planned_routes=100, scenario_confidence=0.95)
    charter = plan["charter"]
    traffic = charter["traffic_scenario"]
    forecast = charter["resource_forecast"]

    assert traffic["analysis_type"] == "traffic_capacity_scenario_not_statistical_power"
    assert traffic["expected"] == {
        "routes_by_action": {"off": 5.0, "collective": 90.0, "loop": 5.0},
        "exploration_routes": 10.0,
    }
    assert traffic["high_probability"]["method"] == (
        "exact_binomial_marginal_upper_quantiles_with_bonferroni_union_bound"
    )
    assert traffic["high_probability"]["simultaneous_confidence_at_least"] == 0.95
    assert traffic["high_probability"]["upper_exploration_routes"] >= 10
    assert all(
        traffic["high_probability"]["upper_routes_by_action"][action]
        >= math.floor(expected)
        for action, expected in traffic["expected"]["routes_by_action"].items()
    )
    assert traffic["not_power_analysis"] is True
    assert traffic["power_analysis_performed"] is False
    assert traffic["minimum_detectable_effect"] is None
    assert traffic["sample_size_recommendation"] is None
    assert traffic["policy_value_estimate"] is None
    label_scenario = traffic["observed_label_scenario"]
    assert label_scenario["analysis_type"] == (
        "simultaneous_alternate_label_traffic_not_power_or_mnar_correction"
    )
    assert label_scenario["target_scope"] == (
        "at_least_target_observed_labels_on_every_alternate_action"
    )
    assert label_scenario["alternate_actions"] == ["off", "loop"]
    assert label_scenario["alternate_action_propensities"] == {
        "off": 0.05,
        "loop": 0.05,
    }
    assert label_scenario["weakest_alternate_propensity"] == 0.05
    assert label_scenario["assumed_feedback_rate"] == DEFAULT_ASSUMED_FEEDBACK_RATE
    assert label_scenario["per_route_observed_label_probability_by_alternate_action"] == {
        "off": 0.015,
        "loop": 0.015,
    }
    assert (
        label_scenario["target_observed_labels_per_alternate_action"]
        == DEFAULT_TARGET_OBSERVED_LABELS
    )
    assert label_scenario[
        "expected_observed_labels_at_planned_routes_by_alternate_action"
    ] == {"off": 1.5, "loop": 1.5}
    assert label_scenario["expected_routes_for_target_by_alternate_action"] == {
        "off": 1333.333333,
        "loop": 1333.333333,
    }
    exact = label_scenario["exact_simultaneous_target"]
    assert exact["method"] == "exact_joint_multinomial_tail_inversion_two_alternates"
    minimum_routes = exact[
        "minimum_routes_for_target_on_every_alternate_action"
    ]
    assert minimum_routes > 1333
    assert exact["probability_at_minimum_routes"] >= 0.95
    assert exact["probability_at_previous_route_count"] < 0.95
    assert label_scenario["missingness_identification_performed"] is False
    assert label_scenario["mnar_correction_performed"] is False
    assert label_scenario["not_power_analysis"] is True

    assert forecast["baseline_per_route"] == {
        "cost_units": 3.0,
        "model_calls": 3,
        "loop_steps": 0,
    }
    assert forecast["expected_per_route"] == {
        "cost_units": 3.35,
        "model_calls": 3.35,
        "loop_steps": 0.2,
    }
    assert forecast["expected_for_planned_routes"] == {
        "cost_units": 335.0,
        "model_calls": 335.0,
        "loop_steps": 20.0,
    }
    assert forecast["expected_increment_vs_baseline"] == {
        "cost_units": 35.0,
        "model_calls": 35.0,
        "loop_steps": 20.0,
    }
    assert forecast["forecast_only"] is True
    assert forecast["budget_guarantee"] is False
    assert forecast["high_probability_conservative_capacity_upper"]["cost_units"] >= 335.0


def test_linear_joint_recurrence_matches_exhaustive_small_multinomials():
    def exhaustive(trials, first, second, target):
        remainder = 1.0 - first - second
        total = 0.0
        for first_count in range(target, trials + 1):
            for second_count in range(target, trials - first_count + 1):
                remainder_count = trials - first_count - second_count
                coefficient = math.factorial(trials) / (
                    math.factorial(first_count)
                    * math.factorial(second_count)
                    * math.factorial(remainder_count)
                )
                total += coefficient * (first**first_count) * (second**second_count) * (
                    remainder**remainder_count
                )
        return total

    for trials in range(2, 11):
        for target in range(1, min(3, trials // 2) + 1):
            for probabilities in ((0.05, 0.05), (0.1, 0.2), (0.25, 0.15)):
                actual = _multinomial_at_least_each_probability(
                    trials, probabilities, target
                )
                assert actual == pytest.approx(
                    exhaustive(trials, *probabilities, target), abs=1e-12
                )


def test_maximum_label_target_keeps_exact_joint_inversion_practical():
    plan = _plan(
        planned_routes=100_000,
        assumed_feedback_rate=0.30,
        target_observed_labels=1_000,
    )
    exact = plan["charter"]["traffic_scenario"]["observed_label_scenario"][
        "exact_simultaneous_target"
    ]
    assert exact["method"] == "exact_joint_multinomial_tail_inversion_two_alternates"
    assert exact["probability_at_minimum_routes"] >= 0.95
    assert exact["probability_at_previous_route_count"] < 0.95


def test_tiny_label_probability_large_n_inversion_avoids_lgamma_cancellation():
    minimum_routes = _minimum_simultaneous_label_routes(
        label_probabilities=(1e-8, 1e-8),
        target_labels=10,
        confidence=0.95,
    )
    assert minimum_routes == 1_706_028_428
    assert _multinomial_at_least_each_probability(
        minimum_routes, (1e-8, 1e-8), 10
    ) >= 0.95
    assert _multinomial_at_least_each_probability(
        minimum_routes - 1, (1e-8, 1e-8), 10
    ) < 0.95


def test_one_neighbor_receives_full_alternate_mass_after_post_filtering():
    plan = _plan(
        baseline="off",
        feasible=("off", "collective"),
    )

    assert plan["charter"]["enrollment"]["adjacent_feasible_actions"] == ["collective"]
    assert plan["charter"]["probability_design"]["action_probabilities"] == {
        "off": 0.9,
        "collective": 0.1,
    }
    assert plan["charter"]["probability_design"]["minimum_positive_exploration_probability"] == 0.1
    exclusions = {row["action"]: row for row in plan["charter"]["exclusions"]}
    assert exclusions["loop"]["post_filter_feasible"] is False
    assert exclusions["collective_loop"]["post_filter_feasible"] is False
    exact = plan["charter"]["traffic_scenario"]["observed_label_scenario"][
        "exact_simultaneous_target"
    ]
    assert exact["method"] == "exact_binomial_tail_inversion_single_alternate"
    assert exact["probability_at_minimum_routes"] >= 0.95
    assert exact["probability_at_previous_route_count"] < 0.95


def test_nonadjacent_only_support_is_not_enrolled_and_cannot_be_assigned():
    plan = _plan(
        baseline="off",
        feasible=("off", "loop"),
    )

    enrollment = plan["charter"]["enrollment"]
    design = plan["charter"]["probability_design"]
    assert enrollment == {
        "eligible": False,
        "reason": "no_feasible_adjacent_action",
        "baseline_action": "off",
        "adjacent_feasible_actions": [],
        "maximum_adjacent_neighbors": 2,
    }
    assert design["decision_type"] == "deterministic_not_enrolled"
    assert design["applied_exploration_rate"] == 0.0
    assert design["action_probabilities"] == {"off": 1.0}
    assert plan["charter"]["traffic_scenario"]["expected"]["exploration_routes"] == 0.0
    label_scenario = plan["charter"]["traffic_scenario"]["observed_label_scenario"]
    assert label_scenario["weakest_alternate_propensity"] is None
    assert label_scenario["alternate_actions"] == []
    assert label_scenario["per_route_observed_label_probability_by_alternate_action"] == {}
    assert label_scenario["expected_routes_for_target_by_alternate_action"] == {}
    assert label_scenario["exact_simultaneous_target"]["method"] == (
        "not_applicable_no_alternate_actions"
    )
    assert label_scenario["exact_simultaneous_target"][
        "minimum_routes_for_target_on_every_alternate_action"
    ] is None
    with pytest.raises(ValueError, match="not eligible"):
        assign_adjacent_route(plan, "non-enrolled-assignment-nonce")


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda candidates, exclusions: candidates.append(copy.deepcopy(candidates[0])), "duplicate"),
        (lambda candidates, exclusions: candidates[0].update({"prompt": "private"}), "unsupported fields"),
        (lambda candidates, exclusions: candidates[0].update({"estimated_cost_units": float("nan")}), "finite"),
        (lambda candidates, exclusions: candidates[0].update({"estimated_model_calls": True}), "integer"),
        (lambda candidates, exclusions: candidates[0].update({"selected": True}), "selected flags"),
        (lambda candidates, exclusions: exclusions.append({"action": "off", "reasons": ["capability_or_policy_filter"]}), "disjoint"),
        (lambda candidates, exclusions: candidates.pop(), "partition all route actions"),
    ],
)
def test_strict_candidate_partition_validation(mutation, match):
    candidates, exclusions = _partition()
    mutation(candidates, exclusions)
    with pytest.raises(ValueError, match=match):
        plan_adjacent_route_study(
            "collective", candidates, exclusions, source_contract=_SOURCE_CONTRACT
        )


def test_strict_exclusion_reason_and_planning_parameter_validation():
    candidates, exclusions = _partition(baseline="off", feasible=("off", "collective"))
    bad_reasons = copy.deepcopy(exclusions)
    bad_reasons[0]["reasons"] = ["prompt said this was risky"]
    with pytest.raises(ValueError, match="versioned reason codes"):
        plan_adjacent_route_study(
            "off", candidates, bad_reasons, source_contract=_SOURCE_CONTRACT
        )
    with pytest.raises(ValueError, match="at or above 0.05"):
        plan_adjacent_route_study(
            "collective",
            *_partition(),
            source_contract=_SOURCE_CONTRACT,
            exploration_rate=0.09,
        )
    with pytest.raises(ValueError, match=r"\(0, 0.20\]"):
        plan_adjacent_route_study(
            "collective",
            *_partition(),
            source_contract=_SOURCE_CONTRACT,
            exploration_rate=0.21,
        )
    with pytest.raises(ValueError, match="planned_routes"):
        plan_adjacent_route_study(
            "collective", *_partition(), source_contract=_SOURCE_CONTRACT, planned_routes=0
        )
    with pytest.raises(ValueError, match="scenario_confidence"):
        plan_adjacent_route_study(
            "collective",
            *_partition(),
            source_contract=_SOURCE_CONTRACT,
            scenario_confidence=1.0,
        )
    with pytest.raises(ValueError, match="assumed_feedback_rate"):
        plan_adjacent_route_study(
            "collective",
            *_partition(),
            source_contract=_SOURCE_CONTRACT,
            assumed_feedback_rate=0.0,
        )
    with pytest.raises(ValueError, match="target_observed_labels"):
        plan_adjacent_route_study(
            "collective",
            *_partition(),
            source_contract=_SOURCE_CONTRACT,
            target_observed_labels=0,
        )


def test_feedback_assumption_and_target_change_label_traffic_and_design_hash():
    baseline = _plan(
        planned_routes=1000,
        assumed_feedback_rate=0.5,
        target_observed_labels=10,
    )
    alternative = _plan(
        planned_routes=1000,
        assumed_feedback_rate=0.25,
        target_observed_labels=10,
    )

    scenario = baseline["charter"]["traffic_scenario"]["observed_label_scenario"]
    assert scenario["per_route_observed_label_probability_by_alternate_action"] == {
        "off": 0.025,
        "loop": 0.025,
    }
    assert scenario["expected_observed_labels_at_planned_routes_by_alternate_action"] == {
        "off": 25.0,
        "loop": 25.0,
    }
    assert scenario["expected_routes_for_target_by_alternate_action"] == {
        "off": 400.0,
        "loop": 400.0,
    }
    assert baseline["design_hash"] != alternative["design_hash"]
    assert (
        scenario["exact_simultaneous_target"][
            "minimum_routes_for_target_on_every_alternate_action"
        ]
        < alternative["charter"]["traffic_scenario"]["observed_label_scenario"][
            "exact_simultaneous_target"
        ]["minimum_routes_for_target_on_every_alternate_action"]
    )


def test_planned_routes_accepts_browser_maximum_of_one_hundred_thousand():
    plan = _plan(
        planned_routes=100_000,
        assumed_feedback_rate=1.0,
        target_observed_labels=1,
    )
    assert plan["charter"]["traffic_scenario"]["planned_routes"] == 100_000


def test_assign_is_nonce_deterministic_and_returns_non_ledger_rehearsal_artifacts():
    plan = _plan()
    nonce = "route-assignment-nonce-0001"

    first = assign_adjacent_route(plan, nonce)
    second = assign_adjacent_route(copy.deepcopy(plan), nonce)

    assert first == second
    assert first["schema_version"] == STUDY_ASSIGNMENT_SCHEMA_VERSION
    assert first["study_id"] == STUDY_ID
    assert first["study_version"] == STUDY_VERSION
    assert first["design_hash"] == plan["design_hash"]
    receipt = first["rehearsal_assignment_receipt"]
    proposal = first["rehearsal_support_proposal"]
    assert receipt["schema_version"] == REHEARSAL_RECEIPT_SCHEMA_VERSION
    assert receipt["rehearsal_only"] is True
    assert receipt["assignment_algorithm"] == ASSIGNMENT_ALGORITHM
    assert len(receipt["assignment_nonce_hash"]) == 64
    assert len(receipt["assignment_draw_hash"]) == 64
    assert 0.0 <= float(receipt["assignment_draw_unit_interval"]) < 1.0
    assert receipt["proposed_action"] in {"off", "collective", "loop"}
    assert receipt["proposed_action_probability"] == proposal["action_probabilities"][
        receipt["proposed_action"]
    ]
    assert nonce not in json.dumps(first, sort_keys=True)
    serialized = json.dumps(first, sort_keys=True)
    assert "logging_support" not in serialized
    assert '"assignment_commitment":' not in serialized

    assert proposal["schema_version"] == REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION
    assert proposal["decision_type"] == "rehearsal_randomized_non_ledger"
    assert proposal["probability_stage"] == "post_filter_rehearsal_only"
    assert proposal["source_contract"] == _SOURCE_CONTRACT
    assert proposal["ledger_eligible"] is False
    assert proposal["preassignment_commitment_sealed"] is False
    assert proposal["nonce_grinding_resistant"] is False
    assert proposal["activation_blockers"] == list(ACTIVATION_BLOCKERS)
    assert first["ledger_eligible"] is False
    assert first["preassignment_commitment_sealed"] is False
    assert first["nonce_grinding_resistant"] is False
    assert first["activation_blockers"] == list(ACTIVATION_BLOCKERS)
    assert first["side_effects"] == {
        "io_performed": False,
        "ledger_write_performed": False,
        "execution_started": False,
        "model_inference_started": False,
    }
    assert first["causal_boundaries"]["execution_enabled"] is False
    assert first["causal_boundaries"]["off_policy_estimate_computed"] is False
    assert first["causal_boundaries"]["automatic_promotion_allowed"] is False


def test_fixed_nonce_space_reaches_baseline_and_each_adjacent_action():
    plan = _plan()
    observed = {}
    for index in range(10_000):
        nonce = f"fixed-study-nonce-{index:05d}"
        assigned = assign_adjacent_route(plan, nonce)
        receipt = assigned["rehearsal_assignment_receipt"]
        observed.setdefault(receipt["proposed_action"], receipt)
        if set(observed) == {"off", "collective", "loop"}:
            break

    assert set(observed) == {"off", "collective", "loop"}
    assert observed["off"]["is_exploration_proposal"] is True
    assert observed["collective"]["is_exploration_proposal"] is False
    assert observed["loop"]["is_exploration_proposal"] is True


def test_nonce_changes_rehearsal_receipt_and_plan_tampering_fails_closed():
    plan = _plan()
    first = assign_adjacent_route(plan, "route-assignment-nonce-alpha")
    second = assign_adjacent_route(plan, "route-assignment-nonce-bravo")
    first_receipt = first["rehearsal_assignment_receipt"]
    second_receipt = second["rehearsal_assignment_receipt"]
    assert first_receipt["assignment_draw_hash"] != second_receipt["assignment_draw_hash"]
    assert first_receipt["assignment_nonce_hash"] != second_receipt["assignment_nonce_hash"]

    tampered = copy.deepcopy(plan)
    tampered["charter"]["probability_design"]["action_probabilities"]["off"] = 0.1
    with pytest.raises(ValueError, match="design_hash"):
        assign_adjacent_route(tampered, "route-assignment-nonce-alpha")
    with pytest.raises(ValueError, match="between 16 and 240"):
        assign_adjacent_route(plan, "short")


def test_planning_and_assignment_perform_no_file_io(monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("planner attempted file I/O")

    monkeypatch.setattr(builtins, "open", forbidden_open)
    plan = _plan(planned_routes=25)
    assignment = assign_adjacent_route(plan, "file-io-free-assignment-nonce")
    assert assignment["side_effects"]["io_performed"] is False
