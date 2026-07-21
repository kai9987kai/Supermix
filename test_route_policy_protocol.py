import builtins
import copy
import json

import pytest

from source.route_policy_explorer import ACTIVATION_BLOCKERS, plan_adjacent_route_study
from source.route_policy_protocol import (
    PROTOCOL_BUILD_INPUT_SCHEMA_VERSION,
    PROTOCOL_SCHEMA_VERSION,
    REVIEW_BUNDLE_SCHEMA_VERSION,
    _PROTOCOL_HASH_DOMAIN,
    _REVIEW_BUNDLE_HASH_DOMAIN,
    _domain_hash,
    audit_route_study_review_bundle,
    audit_route_study_protocol,
    build_route_study_review_bundle,
    build_route_study_protocol,
)
from source.route_policy_study_cli import _example_payload


def _study(*, candidate_hash="a", distribution_hash="b"):
    payload = _example_payload()
    payload["source_contract"] = {
        **payload["source_contract"],
        "candidate_set_hash": candidate_hash * 64,
        "distribution_hash": distribution_hash * 64,
    }
    return plan_adjacent_route_study(
        payload["baseline_mode"],
        payload["post_filter_candidates"],
        payload["post_filter_exclusions"],
        source_contract=payload["source_contract"],
        exploration_rate=payload["exploration_rate"],
        planned_routes=payload["planned_routes"],
        scenario_confidence=payload["scenario_confidence"],
        assumed_feedback_rate=payload["assumed_feedback_rate"],
        target_observed_labels=payload["target_observed_labels"],
    )


def _rehash_protocol(protocol):
    protocol["protocol_hash"] = _domain_hash(
        _PROTOCOL_HASH_DOMAIN,
        {
            "schema_version": protocol["schema_version"],
            "protocol": protocol["protocol"],
            "charter": protocol["charter"],
        },
    )


def _rehash_bundle(bundle):
    bundle["bundle_hash"] = _domain_hash(
        _REVIEW_BUNDLE_HASH_DOMAIN,
        {
            "schema_version": bundle["schema_version"],
            "bundle": bundle["bundle"],
            "protocol_builder": bundle["protocol_builder"],
            "source_study_plans": bundle["source_study_plans"],
            "protocol": bundle["protocol"],
        },
    )


def test_default_protocol_is_prompt_free_canonical_and_fail_closed():
    protocol = build_route_study_protocol(_study())

    assert protocol["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert protocol["protocol"] == {
        "version": "1.0.0",
        "label": "Stateful Route Experiment Preflight v1",
        "state": "draft_for_independent_review",
        "activation_available": False,
    }
    assert len(protocol["protocol_hash"]) == 64
    int(protocol["protocol_hash"], 16)

    charter = protocol["charter"]
    assert charter["source_studies"]["support_stratum_count"] == 1
    assert charter["source_studies"]["source_plans_embedded"] is False
    stratum = charter["source_studies"]["support_strata"][0]
    assert stratum["executed_logging_propensities"] is False
    assert stratum["eligible_actions"] == ["off", "collective", "loop"]
    assert charter["target_policy_class"]["profile_name"] == "balanced"
    assert charter["target_policy_class"]["thresholds"] == {
        "off": 0,
        "collective": 2,
        "loop": 4,
        "collective_loop": 5,
    }
    assert charter["target_policy_class"]["optimality_claim"] is False
    assert charter["population"]["one_study_policy_per_cluster"] is True
    assert charter["population"]["population_precommitted"] is False
    assert charter["stopping_and_resources"]["outcome_dependent_stopping_allowed"] is False
    assert charter["stopping_and_resources"]["automatic_promotion_allowed"] is False
    assert charter["randomness"]["assignment_implementation_available"] is False
    assert charter["randomness"]["assignment_performed"] is False
    assert charter["randomness"]["seed_commitment"] is None
    assert charter["external_evaluation"]["causal_estimate_available"] is False
    assert charter["outcome_observation"]["observation_process_validated"] is False
    assert set(charter["outcome_observation"]["contracts"]) == {
        "route_success",
        "user_quality_rating",
        "cost",
        "latency",
    }
    assert all(
        contract["precommitted"] is True
        and contract["commitment_source"] == "protocol_draft"
        for contract in charter["outcome_observation"]["contracts"].values()
    )
    assert [row["code"] for row in charter["blocker_register"]] == list(
        ACTIVATION_BLOCKERS
    )
    assert all(row["activation_blocking"] is True for row in charter["blocker_register"])
    boundaries = charter["causal_boundaries"]
    assert boundaries["activation_blockers"] == list(ACTIVATION_BLOCKERS)
    assert boundaries["declarations_are_not_validation"] is True
    assert boundaries["execution_enabled"] is False
    assert boundaries["ledger_write_performed"] is False
    assert boundaries["model_inference_performed"] is False
    assert boundaries["off_policy_estimate_computed"] is False
    assert boundaries["automatic_promotion_allowed"] is False
    assert audit_route_study_protocol(protocol)["ok"] is True
    json.dumps(protocol, sort_keys=True, allow_nan=False)


def test_default_design_surfaces_unknown_state_and_screens_out_route_assignment():
    screen = build_route_study_protocol(_study())["charter"]["stateful_design"]

    assert screen["selected_design_mode"] == "sticky_session_cluster"
    assert screen["assignment_unit"] == "session_hash"
    assert screen["selected_design_status"] == "declaration_incomplete"
    assert screen["carryover_validated"] is False
    assert screen["interference_validated"] is False
    route = screen["candidate_screen"][0]
    assert route == {
        "design_mode": "route_randomization",
        "status": "screened_out",
        "blocking_reasons": [
            "stateful_product_campaigns_cannot_randomize_independent_routes_in_v1"
        ],
        "assignment_implementation_available": False,
    }
    assert set(screen["selected_design_blocking_reasons"]) == {
        "carryover_scope_unknown",
        "interference_scope_unknown",
        "temporal_variation_unknown",
    }


def test_switchback_declarations_remain_unvalidated_and_require_washout():
    no_washout = build_route_study_protocol(
        _study(),
        design_mode="clustered_switchback",
        carryover_scope="within_session",
        interference_scope="shared_resource",
        temporal_variation="nonstationary",
        block_length_routes=20,
        washout_routes=0,
    )
    screen = no_washout["charter"]["stateful_design"]
    assert screen["assignment_unit"] == "session_hash_x_time_block"
    assert screen["selected_design_status"] == "incompatible_or_unvalidated_assumptions"
    assert set(screen["selected_design_blocking_reasons"]) == {
        "positive_washout_required_for_declared_carryover",
        "interference_graph_and_exposure_mapping_not_validated",
        "block_balance_and_mixing_not_validated",
    }

    with_washout = build_route_study_protocol(
        _study(),
        design_mode="clustered_switchback",
        carryover_scope="within_session",
        interference_scope="none_declared",
        temporal_variation="stable_declared",
        block_length_routes=20,
        washout_routes=4,
    )
    selected = with_washout["charter"]["stateful_design"]
    assert selected["selected_design_status"] == "assumptions_declared_unvalidated"
    assert selected["selected_design_blocking_reasons"] == []
    assert selected["causal_design_certified"] is False
    assert with_washout["protocol"]["activation_available"] is False


def test_profile_population_resources_and_seed_commitment_are_bound_but_unsealed():
    base = build_route_study_protocol(_study())
    changed = build_route_study_protocol(
        _study(),
        target_policy_profile="quality_first",
        population_rule_id="local-consenting-session-clusters",
        population_rule_version="2",
        cluster_key_schema_version="study-scoped-session-hash-v2",
        planned_clusters=480,
        max_routes_per_cluster=12,
        analysis_every_clusters=60,
        seed_commitment="c" * 64,
        external_estimator_id="independent-cluster-aipw-v1",
        external_reviewer_id="review-board-v1",
    )

    assert changed["protocol_hash"] != base["protocol_hash"]
    assert changed["charter"]["target_policy_class"]["profile_name"] == "quality_first"
    assert changed["charter"]["population"]["planned_clusters"] == 480
    assert changed["charter"]["stopping_and_resources"]["planned_route_ceiling"] == 5760
    randomness = changed["charter"]["randomness"]
    assert randomness["seed_commitment"] == "c" * 64
    assert randomness["seed_material_included"] is False
    assert randomness["seed_commitment_sealed"] is False
    assert randomness["nonce_grinding_resistant"] is False
    seed_blocker = next(
        row
        for row in changed["charter"]["blocker_register"]
        if row["code"] == "preassignment_seed_commitment_not_sealed"
    )
    assert seed_blocker["status"] == "drafted_unsealed"
    assert changed["charter"]["external_evaluation"]["validated"] is False


def test_multiple_strata_are_deduplicated_order_invariant_and_share_source_cohort():
    first = _study(candidate_hash="a", distribution_hash="b")
    second = _study(candidate_hash="c", distribution_hash="d")

    forward = build_route_study_protocol([first, second])
    reverse = build_route_study_protocol([second, first])
    assert forward == reverse
    assert forward["charter"]["source_studies"]["support_stratum_count"] == 2
    hashes = [
        row["study_design_hash"]
        for row in forward["charter"]["source_studies"]["support_strata"]
    ]
    assert hashes == sorted(hashes)

    with pytest.raises(ValueError, match="duplicate design hashes"):
        build_route_study_protocol([first, first])

    incompatible = copy.deepcopy(second)
    incompatible["charter"]["source_contract"]["policy_version"] = "3.0.0"
    # Rebuilding gives it a valid explorer hash but a different source cohort.
    payload = _example_payload()
    payload["source_contract"]["policy_version"] = "3.0.0"
    incompatible = plan_adjacent_route_study(
        payload["baseline_mode"],
        payload["post_filter_candidates"],
        payload["post_filter_exclusions"],
        source_contract=payload["source_contract"],
    )
    with pytest.raises(ValueError, match="one source policy/schema cohort"):
        build_route_study_protocol([first, incompatible])


def test_audit_rejects_tampering_activation_and_schema_drift():
    protocol = build_route_study_protocol(_study())

    tampered = copy.deepcopy(protocol)
    tampered["charter"]["target_policy_class"]["thresholds"]["loop"] = 999
    with pytest.raises(ValueError, match="frozen profile"):
        audit_route_study_protocol(tampered)

    activated = copy.deepcopy(protocol)
    activated["charter"]["causal_boundaries"]["execution_enabled"] = True
    with pytest.raises(ValueError, match="fail-closed"):
        audit_route_study_protocol(activated)

    extra = copy.deepcopy(protocol)
    extra["activate"] = True
    with pytest.raises(ValueError, match="top-level"):
        audit_route_study_protocol(extra)

    rehashed_unsafe = copy.deepcopy(protocol)
    rehashed_unsafe["charter"]["randomness"]["assignment_performed"] = True
    rehashed_unsafe["protocol_hash"] = _domain_hash(
        _PROTOCOL_HASH_DOMAIN,
        {
            "schema_version": rehashed_unsafe["schema_version"],
            "protocol": rehashed_unsafe["protocol"],
            "charter": rehashed_unsafe["charter"],
        },
    )
    with pytest.raises(ValueError, match="non-assigning and unsealed"):
        audit_route_study_protocol(rehashed_unsafe)


def test_builder_rejects_unsafe_or_ambiguous_protocol_inputs():
    plan = _study()
    with pytest.raises(ValueError, match="design_mode must be one of"):
        build_route_study_protocol(plan, design_mode="route_randomization")
    with pytest.raises(ValueError, match="carryover_scope must be one of"):
        build_route_study_protocol(plan, carryover_scope="probably fine")
    with pytest.raises(ValueError, match="prompt-free identifier"):
        build_route_study_protocol(plan, population_rule_id="people who ask about health")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_route_study_protocol(plan, seed_commitment="ABC")
    with pytest.raises(ValueError, match="smaller than block_length"):
        build_route_study_protocol(plan, block_length_routes=10, washout_routes=10)


def test_builder_does_not_use_io_randomness_or_mutate_source_plan(monkeypatch):
    plan = _study()
    before = copy.deepcopy(plan)

    def fail(*_args, **_kwargs):
        raise AssertionError("protocol preflight must not perform file I/O")

    monkeypatch.setattr(builtins, "open", fail)
    protocol = build_route_study_protocol(plan)
    assert plan == before
    assert protocol["charter"]["causal_boundaries"]["io_performed"] is False
    assert protocol["charter"]["randomness"]["assignment_performed"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda draft: draft["charter"]["stopping_and_resources"].update(
                outcome_dependent_stopping_allowed=True,
                automatic_promotion_allowed=True,
            ),
            "stopping rules",
        ),
        (
            lambda draft: draft["charter"]["population"].update(
                population_precommitted=True,
                cluster_map_validated=True,
            ),
            "population",
        ),
        (
            lambda draft: draft["charter"]["outcome_observation"]["contracts"][
                "user_quality_rating"
            ].update(unit="invented_quality_proxy"),
            "outcome contracts",
        ),
        (
            lambda draft: draft["charter"]["external_evaluation"].update(
                validated=True,
                winner_available=True,
            ),
            "external evaluation",
        ),
    ],
)
def test_structural_audit_rejects_semantic_mutations_even_after_public_rehash(mutate, message):
    draft = build_route_study_protocol(_study())
    mutate(draft)
    _rehash_protocol(draft)

    with pytest.raises(ValueError, match=message):
        audit_route_study_protocol(draft)


def test_review_bundle_is_source_bound_order_invariant_and_explicitly_unsigned():
    first = _study(candidate_hash="a", distribution_hash="b")
    second = _study(candidate_hash="c", distribution_hash="d")

    forward = build_route_study_review_bundle(
        [first, second],
        target_policy_profile="quality_first",
        carryover_scope="within_session",
        interference_scope="none_declared",
        temporal_variation="stable_declared",
    )
    reverse = build_route_study_review_bundle(
        [second, first],
        target_policy_profile="quality_first",
        carryover_scope="within_session",
        interference_scope="none_declared",
        temporal_variation="stable_declared",
    )

    assert forward == reverse
    assert forward["schema_version"] == REVIEW_BUNDLE_SCHEMA_VERSION
    assert forward["protocol_builder"]["schema_version"] == (
        PROTOCOL_BUILD_INPUT_SCHEMA_VERSION
    )
    assert forward["bundle"]["verification_level"] == (
        "full_source_bound_reconstruction"
    )
    assert forward["bundle"]["authenticity_proof_available"] is False
    assert forward["bundle"]["trusted_timestamp_available"] is False
    audit = audit_route_study_review_bundle(forward)
    assert audit["ok"] is True
    assert audit["support_stratum_count"] == 2
    assert audit["source_plan_reconstruction_performed"] is True
    assert audit["activation_available"] is False


def test_review_bundle_reconstruction_rejects_rehashed_source_binding_substitution():
    bundle = build_route_study_review_bundle(
        [_study(candidate_hash="a", distribution_hash="b"), _study(candidate_hash="c", distribution_hash="d")]
    )
    protocol = bundle["protocol"]
    old_hash = protocol["charter"]["source_studies"]["support_strata"][0][
        "study_design_hash"
    ]
    replacement = "f" * 64 if old_hash != "f" * 64 else "e" * 64
    protocol["charter"]["source_studies"]["support_strata"][0][
        "study_design_hash"
    ] = replacement
    protocol["charter"]["population"]["admitted_support_strata"][0] = replacement
    # Keep structural ordering canonical so only full source reconstruction exposes it.
    paired = list(
        zip(
            protocol["charter"]["source_studies"]["support_strata"],
            protocol["charter"]["population"]["admitted_support_strata"],
        )
    )
    paired.sort(key=lambda pair: pair[0]["study_design_hash"])
    protocol["charter"]["source_studies"]["support_strata"] = [pair[0] for pair in paired]
    protocol["charter"]["population"]["admitted_support_strata"] = [
        pair[1] for pair in paired
    ]
    _rehash_protocol(protocol)
    assert audit_route_study_protocol(protocol)["verification_level"] == (
        "structural_without_source_plans"
    )
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match="does not reconstruct"):
        audit_route_study_review_bundle(bundle)


def test_review_bundle_rejects_reordered_or_noncanonical_source_plans_after_rehash():
    bundle = build_route_study_review_bundle(
        [_study(candidate_hash="a", distribution_hash="b"), _study(candidate_hash="c", distribution_hash="d")]
    )
    bundle["source_study_plans"].reverse()
    _rehash_bundle(bundle)

    with pytest.raises(ValueError, match="not canonical or canonically ordered"):
        audit_route_study_review_bundle(bundle)
