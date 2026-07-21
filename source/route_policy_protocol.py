"""Fail-closed stateful experiment protocol preflight for route rehearsals.

The adjacent-route explorer describes one prompt-specific support stratum.
This module wraps one or more of those immutable plans in a prompt-free,
reviewable protocol *draft*.  It freezes a target-policy class, population and
resource declarations, Route Outcome Contracts, and a stateful design screen.

It deliberately does not seal a protocol, generate or reveal a random seed,
assign a cluster, execute a route, write evidence, estimate policy value, or
authorize promotion.  Declarations are kept distinct from independent
validation so that state carryover, interference, and missingness cannot be
silently assumed away.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:
    from .route_policy_explorer import (
        ACTIVATION_BLOCKERS,
        STUDY_ID,
        STUDY_PLAN_SCHEMA_VERSION,
        STUDY_VERSION,
        validate_adjacent_route_study,
    )
    from .route_policy_lab import AUTO_AGENT_MODE_ORDER, get_policy_profile
    from .route_policy_ledger import (
        OUTCOME_CONTRACT_SCHEMA_VERSION,
        build_route_outcome_contracts,
    )
except ImportError:  # pragma: no cover - direct source/ execution compatibility
    from route_policy_explorer import (
        ACTIVATION_BLOCKERS,
        STUDY_ID,
        STUDY_PLAN_SCHEMA_VERSION,
        STUDY_VERSION,
        validate_adjacent_route_study,
    )
    from route_policy_lab import AUTO_AGENT_MODE_ORDER, get_policy_profile
    from route_policy_ledger import (
        OUTCOME_CONTRACT_SCHEMA_VERSION,
        build_route_outcome_contracts,
    )


PROTOCOL_SCHEMA_VERSION = "route-study-protocol-preflight-v1"
PROTOCOL_VERSION = "1.0.0"
PROTOCOL_LABEL = "Stateful Route Experiment Preflight v1"
PROTOCOL_BUILD_INPUT_SCHEMA_VERSION = "route-study-protocol-build-input-v1"
REVIEW_BUNDLE_SCHEMA_VERSION = "route-study-review-bundle-v1"
REVIEW_BUNDLE_VERSION = "1.0.0"
REVIEW_BUNDLE_LABEL = "Route Protocol Review Bundle v1"
TARGET_POLICY_CLASS_SCHEMA_VERSION = "route-target-policy-class-v1"
POPULATION_SCHEMA_VERSION = "route-study-population-v1"
STATEFUL_DESIGN_SCHEMA_VERSION = "route-stateful-design-screen-v1"
STOPPING_SCHEMA_VERSION = "route-study-stopping-v1"
RANDOMNESS_SCHEMA_VERSION = "route-study-randomness-v1"

DESIGN_MODES: Tuple[str, ...] = (
    "sticky_session_cluster",
    "clustered_switchback",
)
CARRYOVER_SCOPES: Tuple[str, ...] = (
    "unknown",
    "none_declared",
    "within_session",
    "cross_session",
)
INTERFERENCE_SCOPES: Tuple[str, ...] = (
    "unknown",
    "none_declared",
    "shared_resource",
    "cross_cluster",
)
TEMPORAL_VARIATION_SCOPES: Tuple[str, ...] = (
    "unknown",
    "stable_declared",
    "nonstationary",
)

DEFAULT_PLANNED_CLUSTERS = 200
DEFAULT_MAX_ROUTES_PER_CLUSTER = 20
DEFAULT_ANALYSIS_EVERY_CLUSTERS = 50
MAX_PLANNED_CLUSTERS = 1_000_000
MAX_ROUTES_PER_CLUSTER = 10_000

_PROTOCOL_HASH_DOMAIN = b"supermix.route-study.protocol-preflight.v1\x00"
_REVIEW_BUNDLE_HASH_DOMAIN = b"supermix.route-study.review-bundle.v1\x00"
_TARGET_CLASS_HASH_DOMAIN = b"supermix.route-study.target-class.v1\x00"
_OUTCOME_SET_HASH_DOMAIN = b"supermix.route-study.outcome-set.v1\x00"
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TOP_LEVEL_KEYS = {"schema_version", "protocol", "charter", "protocol_hash"}
_PROTOCOL_KEYS = {"version", "label", "state", "activation_available"}
_CHARTER_KEYS = {
    "source_studies",
    "target_policy_class",
    "population",
    "stateful_design",
    "outcome_observation",
    "stopping_and_resources",
    "randomness",
    "external_evaluation",
    "blocker_register",
    "prompt_free_contract",
    "causal_boundaries",
}
PROTOCOL_OPTION_KEYS = frozenset(
    {
        "target_policy_profile",
        "design_mode",
        "carryover_scope",
        "interference_scope",
        "temporal_variation",
        "population_rule_id",
        "population_rule_version",
        "cluster_key_schema_version",
        "planned_clusters",
        "max_routes_per_cluster",
        "analysis_every_clusters",
        "block_length_routes",
        "washout_routes",
        "seed_commitment",
        "external_estimator_id",
        "external_reviewer_id",
    }
)
PROTOCOL_BUILD_INPUT_KEYS = frozenset({"study_plans", *PROTOCOL_OPTION_KEYS})

_SOURCE_STUDIES_KEYS = {
    "study_schema_version",
    "study_id",
    "study_version",
    "support_stratum_count",
    "common_source_contract",
    "support_strata",
    "source_plans_embedded",
}
_COMMON_SOURCE_CONTRACT_KEYS = {
    "policy_id",
    "policy_version",
    "feature_schema_version",
    "support_schema_version",
    "outcome_contract_schema_version",
}
_SUPPORT_STRATUM_KEYS = {
    "study_design_hash",
    "candidate_set_hash",
    "distribution_hash",
    "baseline_action",
    "eligible_actions",
    "rehearsed_action_probabilities",
    "executed_logging_propensities",
}
_POPULATION_KEYS = {
    "schema_version",
    "population_rule_id",
    "population_rule_version",
    "cluster_key_schema_version",
    "cluster_identifier_exported",
    "study_scoped_pseudonym_required",
    "start_after_protocol_seal_required",
    "one_study_policy_per_cluster",
    "planned_clusters",
    "max_routes_per_cluster",
    "admitted_support_strata",
    "population_precommitted",
    "cluster_map_validated",
}
_STOPPING_KEYS = {
    "schema_version",
    "planned_cluster_ceiling",
    "planned_route_ceiling",
    "analysis_every_clusters",
    "fixed_analysis_schedule",
    "outcome_dependent_stopping_allowed",
    "automatic_promotion_allowed",
    "resource_guardrail_action",
    "rules_precommitted",
}
_RANDOMNESS_KEYS = {
    "schema_version",
    "planned_derivation",
    "seed_commitment",
    "seed_material_included",
    "seed_commitment_sealed",
    "caller_selected_nonce_allowed",
    "assignment_receipt_required_before_first_route",
    "assignment_implementation_available",
    "assignment_performed",
    "nonce_grinding_resistant",
}
_OUTCOME_KEYS = {
    "schema_version",
    "contracts",
    "observation_process_validated",
    "missingness_identified",
    "mnar_correction_available",
    "rating_nonresponse_treated_as_outcome",
    "contract_set_hash",
}
_EXTERNAL_KEYS = {
    "estimator_id",
    "reviewer_id",
    "validated",
    "causal_estimate_available",
    "winner_available",
}
_PROMPT_FREE_KEYS = {
    "prompt_free",
    "raw_prompt_included",
    "raw_session_id_included",
    "free_form_text_fields_allowed",
    "canonical_json",
}
_CAUSAL_BOUNDARY_KEYS = {
    "deployment",
    "protocol_sealed",
    "activation_available",
    "assignment_available",
    "assignment_performed",
    "execution_enabled",
    "io_performed",
    "ledger_write_performed",
    "model_inference_performed",
    "off_policy_estimate_computed",
    "causal_identification_performed",
    "power_analysis_performed",
    "automatic_promotion_allowed",
    "always_valid_inference_available",
    "declarations_are_not_validation",
    "activation_blockers",
    "interpretation",
}
_DESIGN_KEYS = {
    "schema_version",
    "selected_design_mode",
    "selected_design_status",
    "assignment_unit",
    "sticky_within_live_session",
    "carryover_scope",
    "interference_scope",
    "temporal_variation",
    "block_length_routes",
    "washout_routes",
    "candidate_screen",
    "selected_design_blocking_reasons",
    "carryover_validated",
    "interference_validated",
    "mixing_validated",
    "causal_design_certified",
}
_DESIGN_CANDIDATE_KEYS = {
    "design_mode",
    "status",
    "blocking_reasons",
    "assignment_implementation_available",
    "selected",
}
_BLOCKER_KEYS = {"code", "status", "activation_blocking"}
_BUNDLE_TOP_LEVEL_KEYS = {
    "schema_version",
    "bundle",
    "protocol_builder",
    "source_study_plans",
    "protocol",
    "bundle_hash",
}
_BUNDLE_META_KEYS = {
    "version",
    "label",
    "state",
    "verification_level",
    "authenticity_proof_available",
    "trusted_timestamp_available",
    "activation_available",
}


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("protocol must be canonical JSON without non-finite values") from exc


def _domain_hash(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _identifier(value: Any, name: str) -> str:
    cooked = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(cooked):
        raise ValueError(f"{name} must be a versioned prompt-free identifier")
    return cooked


def _enum(value: Any, allowed: Sequence[str], name: str) -> str:
    cooked = str(value or "").strip()
    if cooked not in allowed:
        raise ValueError(f"{name} must be one of: {', '.join(allowed)}")
    return cooked


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    try:
        cooked = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}") from exc
    if isinstance(value, float) and (not math.isfinite(value) or value != cooked):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    if not minimum <= cooked <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return cooked


def _seed_commitment(value: Any) -> Optional[str]:
    if value is None or str(value).strip() == "":
        return None
    cooked = str(value).strip()
    if not _SHA256_RE.fullmatch(cooked):
        raise ValueError("seed_commitment must be a lowercase SHA-256 digest or omitted")
    return cooked


def _normalize_study_plans(value: Any) -> list[Dict[str, Any]]:
    if isinstance(value, Mapping):
        raw_plans = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_plans = list(value)
    else:
        raise ValueError("study_plans must be an eligible plan or a non-empty sequence")
    if not raw_plans:
        raise ValueError("study_plans must contain at least one eligible rehearsal plan")
    if len(raw_plans) > 1_000:
        raise ValueError("study_plans supports at most 1000 prompt-free support strata")

    validated = [validate_adjacent_route_study(plan) for plan in raw_plans]
    hashes = [row["design_hash"] for row in validated]
    if len(set(hashes)) != len(hashes):
        raise ValueError("study_plans must not contain duplicate design hashes")
    validated.sort(key=lambda row: row["design_hash"])

    common_keys = (
        "policy_id",
        "policy_version",
        "feature_schema_version",
        "support_schema_version",
        "outcome_contract_schema_version",
    )
    first_contract = validated[0]["source_contract"]
    for row in validated[1:]:
        if any(row["source_contract"][key] != first_contract[key] for key in common_keys):
            raise ValueError("all study support strata must share one source policy/schema cohort")
    return validated


def _target_policy_class(profile_name: str) -> Dict[str, Any]:
    profile = get_policy_profile(profile_name)
    payload = {
        "schema_version": TARGET_POLICY_CLASS_SCHEMA_VERSION,
        "profile_name": profile.name,
        "thresholds": dict(profile.thresholds),
        "supported_actions": list(AUTO_AGENT_MODE_ORDER),
        "frozen_in_draft": True,
        "externally_validated": False,
        "optimality_claim": False,
    }
    return {**payload, "class_hash": _domain_hash(_TARGET_CLASS_HASH_DOMAIN, payload)}


def _design_reasons(
    mode: str,
    *,
    carryover_scope: str,
    interference_scope: str,
    temporal_variation: str,
    washout_routes: int,
) -> list[str]:
    reasons: list[str] = []
    if carryover_scope == "unknown":
        reasons.append("carryover_scope_unknown")
    if interference_scope == "unknown":
        reasons.append("interference_scope_unknown")
    if temporal_variation == "unknown":
        reasons.append("temporal_variation_unknown")

    if mode == "sticky_session_cluster":
        if carryover_scope == "cross_session":
            reasons.append("session_cluster_does_not_cover_cross_session_state")
        if interference_scope in {"shared_resource", "cross_cluster"}:
            reasons.append("session_cluster_does_not_block_declared_interference")
        if temporal_variation == "nonstationary":
            reasons.append("temporal_blocking_not_declared")
    elif mode == "clustered_switchback":
        if carryover_scope in {"within_session", "cross_session"} and washout_routes <= 0:
            reasons.append("positive_washout_required_for_declared_carryover")
        if carryover_scope == "cross_session":
            reasons.append("finite_cross_session_carryover_not_validated")
        if interference_scope in {"shared_resource", "cross_cluster"}:
            reasons.append("interference_graph_and_exposure_mapping_not_validated")
        if temporal_variation == "nonstationary":
            reasons.append("block_balance_and_mixing_not_validated")
    return reasons


def _design_screen(
    selected_mode: str,
    *,
    carryover_scope: str,
    interference_scope: str,
    temporal_variation: str,
    block_length_routes: int,
    washout_routes: int,
) -> Dict[str, Any]:
    candidates = [
        {
            "design_mode": "route_randomization",
            "status": "screened_out",
            "blocking_reasons": [
                "stateful_product_campaigns_cannot_randomize_independent_routes_in_v1"
            ],
            "assignment_implementation_available": False,
        }
    ]
    selected_reasons: list[str] = []
    for mode in DESIGN_MODES:
        reasons = _design_reasons(
            mode,
            carryover_scope=carryover_scope,
            interference_scope=interference_scope,
            temporal_variation=temporal_variation,
            washout_routes=washout_routes,
        )
        if mode == selected_mode:
            selected_reasons = list(reasons)
        candidates.append(
            {
                "design_mode": mode,
                "selected": mode == selected_mode,
                "status": (
                    "assumptions_declared_unvalidated" if not reasons else "blocked_for_review"
                ),
                "blocking_reasons": reasons,
                "assignment_implementation_available": False,
            }
        )

    has_unknown = any(
        value == "unknown"
        for value in (carryover_scope, interference_scope, temporal_variation)
    )
    selected_status = (
        "declaration_incomplete"
        if has_unknown
        else (
            "assumptions_declared_unvalidated"
            if not selected_reasons
            else "incompatible_or_unvalidated_assumptions"
        )
    )
    return {
        "schema_version": STATEFUL_DESIGN_SCHEMA_VERSION,
        "selected_design_mode": selected_mode,
        "selected_design_status": selected_status,
        "assignment_unit": (
            "session_hash" if selected_mode == "sticky_session_cluster" else "session_hash_x_time_block"
        ),
        "sticky_within_live_session": True,
        "carryover_scope": carryover_scope,
        "interference_scope": interference_scope,
        "temporal_variation": temporal_variation,
        "block_length_routes": block_length_routes if selected_mode == "clustered_switchback" else None,
        "washout_routes": washout_routes if selected_mode == "clustered_switchback" else None,
        "candidate_screen": candidates,
        "selected_design_blocking_reasons": selected_reasons,
        "carryover_validated": False,
        "interference_validated": False,
        "mixing_validated": False,
        "causal_design_certified": False,
    }


def _blocker_register(
    *,
    carryover_scope: str,
    interference_scope: str,
    seed_commitment: Optional[str],
) -> list[Dict[str, Any]]:
    status_by_code = {
        "target_policy_class_not_precommitted": "drafted_unsealed",
        "outcome_observation_process_not_validated": "drafted_unvalidated",
        "population_definition_not_precommitted": "drafted_unsealed",
        "session_carryover_not_addressed": (
            "unresolved" if carryover_scope == "unknown" else "declared_unvalidated"
        ),
        "interference_not_addressed": (
            "unresolved" if interference_scope == "unknown" else "declared_unvalidated"
        ),
        "stopping_rules_not_precommitted": "drafted_unsealed",
        "preassignment_seed_commitment_not_sealed": (
            "unresolved" if seed_commitment is None else "drafted_unsealed"
        ),
        "external_ope_not_validated": "unresolved",
    }
    return [
        {
            "code": code,
            "status": status_by_code[code],
            "activation_blocking": True,
        }
        for code in ACTIVATION_BLOCKERS
    ]


def _normalize_protocol_options(
    *,
    target_policy_profile: str = "balanced",
    design_mode: str = "sticky_session_cluster",
    carryover_scope: str = "unknown",
    interference_scope: str = "unknown",
    temporal_variation: str = "unknown",
    population_rule_id: str = "interactive-auto-route-opt-in",
    population_rule_version: str = "1",
    cluster_key_schema_version: str = "session-hash-v1",
    planned_clusters: int = DEFAULT_PLANNED_CLUSTERS,
    max_routes_per_cluster: int = DEFAULT_MAX_ROUTES_PER_CLUSTER,
    analysis_every_clusters: int = DEFAULT_ANALYSIS_EVERY_CLUSTERS,
    block_length_routes: int = 20,
    washout_routes: int = 0,
    seed_commitment: Optional[str] = None,
    external_estimator_id: Optional[str] = None,
    external_reviewer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the complete canonical builder option set used by review bundles."""

    profile = get_policy_profile(str(target_policy_profile or "").strip())
    cooked_mode = _enum(design_mode, DESIGN_MODES, "design_mode")
    cooked_carryover = _enum(carryover_scope, CARRYOVER_SCOPES, "carryover_scope")
    cooked_interference = _enum(
        interference_scope, INTERFERENCE_SCOPES, "interference_scope"
    )
    cooked_temporal = _enum(
        temporal_variation, TEMPORAL_VARIATION_SCOPES, "temporal_variation"
    )
    cooked_clusters = _bounded_int(
        planned_clusters, "planned_clusters", minimum=2, maximum=MAX_PLANNED_CLUSTERS
    )
    cooked_max_routes = _bounded_int(
        max_routes_per_cluster,
        "max_routes_per_cluster",
        minimum=1,
        maximum=MAX_ROUTES_PER_CLUSTER,
    )
    cooked_analysis_every = _bounded_int(
        analysis_every_clusters,
        "analysis_every_clusters",
        minimum=1,
        maximum=cooked_clusters,
    )
    cooked_block = _bounded_int(
        block_length_routes,
        "block_length_routes",
        minimum=2,
        maximum=MAX_ROUTES_PER_CLUSTER,
    )
    cooked_washout = _bounded_int(
        washout_routes,
        "washout_routes",
        minimum=0,
        maximum=MAX_ROUTES_PER_CLUSTER,
    )
    if cooked_washout >= cooked_block:
        raise ValueError("washout_routes must be smaller than block_length_routes")
    estimator_id = (
        _identifier(external_estimator_id, "external_estimator_id")
        if external_estimator_id is not None and str(external_estimator_id).strip()
        else None
    )
    reviewer_id = (
        _identifier(external_reviewer_id, "external_reviewer_id")
        if external_reviewer_id is not None and str(external_reviewer_id).strip()
        else None
    )
    return {
        "target_policy_profile": profile.name,
        "design_mode": cooked_mode,
        "carryover_scope": cooked_carryover,
        "interference_scope": cooked_interference,
        "temporal_variation": cooked_temporal,
        "population_rule_id": _identifier(population_rule_id, "population_rule_id"),
        "population_rule_version": _identifier(
            population_rule_version, "population_rule_version"
        ),
        "cluster_key_schema_version": _identifier(
            cluster_key_schema_version, "cluster_key_schema_version"
        ),
        "planned_clusters": cooked_clusters,
        "max_routes_per_cluster": cooked_max_routes,
        "analysis_every_clusters": cooked_analysis_every,
        "block_length_routes": cooked_block,
        "washout_routes": cooked_washout,
        "seed_commitment": _seed_commitment(seed_commitment),
        "external_estimator_id": estimator_id,
        "external_reviewer_id": reviewer_id,
    }


def _split_protocol_build_input(payload: Any) -> Tuple[Any, Dict[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("protocol build input must be a JSON object")
    unknown = set(payload) - set(PROTOCOL_BUILD_INPUT_KEYS)
    if unknown:
        raise ValueError(
            "protocol build input contains unsupported or non-prompt-free fields: "
            + ", ".join(sorted(map(str, unknown)))
        )
    if "study_plans" not in payload:
        raise ValueError("protocol build input is missing required field: study_plans")
    options = _normalize_protocol_options(
        **{key: payload[key] for key in PROTOCOL_OPTION_KEYS if key in payload}
    )
    return payload["study_plans"], options


def build_route_study_protocol_from_input(payload: Any) -> Dict[str, Any]:
    """Build from the one closed, prompt-free input schema shared by web and CLI."""

    study_plans, options = _split_protocol_build_input(payload)
    return build_route_study_protocol(study_plans, **options)


def build_route_study_protocol(
    study_plans: Any,
    *,
    target_policy_profile: str = "balanced",
    design_mode: str = "sticky_session_cluster",
    carryover_scope: str = "unknown",
    interference_scope: str = "unknown",
    temporal_variation: str = "unknown",
    population_rule_id: str = "interactive-auto-route-opt-in",
    population_rule_version: str = "1",
    cluster_key_schema_version: str = "session-hash-v1",
    planned_clusters: int = DEFAULT_PLANNED_CLUSTERS,
    max_routes_per_cluster: int = DEFAULT_MAX_ROUTES_PER_CLUSTER,
    analysis_every_clusters: int = DEFAULT_ANALYSIS_EVERY_CLUSTERS,
    block_length_routes: int = 20,
    washout_routes: int = 0,
    seed_commitment: Optional[str] = None,
    external_estimator_id: Optional[str] = None,
    external_reviewer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a canonical, non-sealed protocol preflight around study strata."""

    validated = _normalize_study_plans(study_plans)
    options = _normalize_protocol_options(
        target_policy_profile=target_policy_profile,
        design_mode=design_mode,
        carryover_scope=carryover_scope,
        interference_scope=interference_scope,
        temporal_variation=temporal_variation,
        population_rule_id=population_rule_id,
        population_rule_version=population_rule_version,
        cluster_key_schema_version=cluster_key_schema_version,
        planned_clusters=planned_clusters,
        max_routes_per_cluster=max_routes_per_cluster,
        analysis_every_clusters=analysis_every_clusters,
        block_length_routes=block_length_routes,
        washout_routes=washout_routes,
        seed_commitment=seed_commitment,
        external_estimator_id=external_estimator_id,
        external_reviewer_id=external_reviewer_id,
    )

    first_contract = validated[0]["source_contract"]
    common_source_contract = {
        key: first_contract[key]
        for key in (
            "policy_id",
            "policy_version",
            "feature_schema_version",
            "support_schema_version",
            "outcome_contract_schema_version",
        )
    }
    strata = [
        {
            "study_design_hash": row["design_hash"],
            "candidate_set_hash": row["source_contract"]["candidate_set_hash"],
            "distribution_hash": row["source_contract"]["distribution_hash"],
            "baseline_action": row["enrollment"]["baseline_action"],
            "eligible_actions": list(row["probability_design"]["eligible_actions"]),
            "rehearsed_action_probabilities": dict(
                row["probability_design"]["action_probabilities"]
            ),
            "executed_logging_propensities": False,
        }
        for row in validated
    ]

    target_class = _target_policy_class(options["target_policy_profile"])
    outcome_contracts = build_route_outcome_contracts(
        precommitted=True,
        commitment_source="protocol_draft",
    )
    outcome_payload = {
        "schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
        "contracts": outcome_contracts,
        "observation_process_validated": False,
        "missingness_identified": False,
        "mnar_correction_available": False,
        "rating_nonresponse_treated_as_outcome": False,
    }
    outcome_payload["contract_set_hash"] = _domain_hash(
        _OUTCOME_SET_HASH_DOMAIN, outcome_payload
    )

    design = _design_screen(
        options["design_mode"],
        carryover_scope=options["carryover_scope"],
        interference_scope=options["interference_scope"],
        temporal_variation=options["temporal_variation"],
        block_length_routes=options["block_length_routes"],
        washout_routes=options["washout_routes"],
    )
    population = {
        "schema_version": POPULATION_SCHEMA_VERSION,
        "population_rule_id": options["population_rule_id"],
        "population_rule_version": options["population_rule_version"],
        "cluster_key_schema_version": options["cluster_key_schema_version"],
        "cluster_identifier_exported": False,
        "study_scoped_pseudonym_required": True,
        "start_after_protocol_seal_required": True,
        "one_study_policy_per_cluster": True,
        "planned_clusters": options["planned_clusters"],
        "max_routes_per_cluster": options["max_routes_per_cluster"],
        "admitted_support_strata": [row["study_design_hash"] for row in strata],
        "population_precommitted": False,
        "cluster_map_validated": False,
    }
    route_ceiling = options["planned_clusters"] * options["max_routes_per_cluster"]
    stopping = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "planned_cluster_ceiling": options["planned_clusters"],
        "planned_route_ceiling": route_ceiling,
        "analysis_every_clusters": options["analysis_every_clusters"],
        "fixed_analysis_schedule": True,
        "outcome_dependent_stopping_allowed": False,
        "automatic_promotion_allowed": False,
        "resource_guardrail_action": "pause_and_independent_review",
        "rules_precommitted": False,
    }
    randomness = {
        "schema_version": RANDOMNESS_SCHEMA_VERSION,
        "planned_derivation": "hmac_sha256_seed_design_hash_cluster_hash_block_id",
        "seed_commitment": options["seed_commitment"],
        "seed_material_included": False,
        "seed_commitment_sealed": False,
        "caller_selected_nonce_allowed": False,
        "assignment_receipt_required_before_first_route": True,
        "assignment_implementation_available": False,
        "assignment_performed": False,
        "nonce_grinding_resistant": False,
    }
    blocker_register = _blocker_register(
        carryover_scope=options["carryover_scope"],
        interference_scope=options["interference_scope"],
        seed_commitment=options["seed_commitment"],
    )
    charter = {
        "source_studies": {
            "study_schema_version": STUDY_PLAN_SCHEMA_VERSION,
            "study_id": STUDY_ID,
            "study_version": STUDY_VERSION,
            "support_stratum_count": len(strata),
            "common_source_contract": common_source_contract,
            "support_strata": strata,
            "source_plans_embedded": False,
        },
        "target_policy_class": target_class,
        "population": population,
        "stateful_design": design,
        "outcome_observation": outcome_payload,
        "stopping_and_resources": stopping,
        "randomness": randomness,
        "external_evaluation": {
            "estimator_id": options["external_estimator_id"],
            "reviewer_id": options["external_reviewer_id"],
            "validated": False,
            "causal_estimate_available": False,
            "winner_available": False,
        },
        "blocker_register": blocker_register,
        "prompt_free_contract": {
            "prompt_free": True,
            "raw_prompt_included": False,
            "raw_session_id_included": False,
            "free_form_text_fields_allowed": False,
            "canonical_json": "sorted_keys_compact_utf8_no_nan",
        },
        "causal_boundaries": {
            "deployment": "shadow_only",
            "protocol_sealed": False,
            "activation_available": False,
            "assignment_available": False,
            "assignment_performed": False,
            "execution_enabled": False,
            "io_performed": False,
            "ledger_write_performed": False,
            "model_inference_performed": False,
            "off_policy_estimate_computed": False,
            "causal_identification_performed": False,
            "power_analysis_performed": False,
            "automatic_promotion_allowed": False,
            "always_valid_inference_available": False,
            "declarations_are_not_validation": True,
            "activation_blockers": list(ACTIVATION_BLOCKERS),
            "interpretation": "prompt-free protocol draft for independent review only",
        },
    }
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "label": PROTOCOL_LABEL,
            "state": "draft_for_independent_review",
            "activation_available": False,
        },
        "charter": charter,
    }
    return {**payload, "protocol_hash": _domain_hash(_PROTOCOL_HASH_DOMAIN, payload)}


def _require_exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} does not match the v1 schema")
    return value


def _sha256_digest(value: Any, name: str) -> str:
    cooked = str(value or "").strip()
    if not _SHA256_RE.fullmatch(cooked):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cooked


def _optional_identifier(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _identifier(value, name)


def _validate_source_studies(value: Any) -> Tuple[Mapping[str, Any], list[str]]:
    source = _require_exact_keys(value, _SOURCE_STUDIES_KEYS, "protocol source studies")
    if source.get("study_schema_version") != STUDY_PLAN_SCHEMA_VERSION:
        raise ValueError("protocol source study schema_version is unsupported")
    if source.get("study_id") != STUDY_ID or source.get("study_version") != STUDY_VERSION:
        raise ValueError("protocol source study identity is unsupported")
    if source.get("source_plans_embedded") is not False:
        raise ValueError("protocol draft cannot claim embedded source plans")

    common = _require_exact_keys(
        source.get("common_source_contract"),
        _COMMON_SOURCE_CONTRACT_KEYS,
        "protocol common source contract",
    )
    for key in (
        "policy_id",
        "policy_version",
        "feature_schema_version",
        "support_schema_version",
        "outcome_contract_schema_version",
    ):
        _identifier(common.get(key), f"common source contract {key}")
    if common.get("outcome_contract_schema_version") != OUTCOME_CONTRACT_SCHEMA_VERSION:
        raise ValueError("protocol source outcome contract schema is unsupported")

    raw_strata = source.get("support_strata")
    if (
        not isinstance(raw_strata, Sequence)
        or isinstance(raw_strata, (str, bytes))
        or not raw_strata
        or len(raw_strata) > 1_000
    ):
        raise ValueError("protocol must bind between 1 and 1000 support strata")
    if source.get("support_stratum_count") != len(raw_strata):
        raise ValueError("protocol support stratum count does not match its inventory")

    design_hashes: list[str] = []
    for index, raw in enumerate(raw_strata):
        row = _require_exact_keys(
            raw, _SUPPORT_STRATUM_KEYS, f"protocol support stratum {index}"
        )
        design_hash = _sha256_digest(row.get("study_design_hash"), "study_design_hash")
        _sha256_digest(row.get("candidate_set_hash"), "candidate_set_hash")
        _sha256_digest(row.get("distribution_hash"), "distribution_hash")
        eligible_raw = row.get("eligible_actions")
        if not isinstance(eligible_raw, list) or not eligible_raw:
            raise ValueError("protocol support stratum eligible_actions must be a non-empty list")
        eligible = [str(action) for action in eligible_raw]
        if len(set(eligible)) != len(eligible) or any(
            action not in AUTO_AGENT_MODE_ORDER for action in eligible
        ):
            raise ValueError("protocol support stratum eligible_actions are unsupported")
        canonical_eligible = [action for action in AUTO_AGENT_MODE_ORDER if action in eligible]
        if eligible != canonical_eligible:
            raise ValueError("protocol support stratum eligible_actions are not canonical")
        if row.get("baseline_action") not in eligible:
            raise ValueError("protocol support stratum baseline is outside eligible_actions")
        probabilities = row.get("rehearsed_action_probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(eligible):
            raise ValueError("protocol support stratum probability vector is incomplete")
        total = 0.0
        for action in eligible:
            value = probabilities.get(action)
            if isinstance(value, bool):
                raise ValueError("protocol support probabilities must be finite positive numbers")
            try:
                probability = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "protocol support probabilities must be finite positive numbers"
                ) from exc
            if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
                raise ValueError("protocol support probabilities must be finite positive numbers")
            total += probability
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("protocol support probabilities must sum to one")
        if row.get("executed_logging_propensities") is not False:
            raise ValueError("protocol strata cannot claim executed logging propensities")
        design_hashes.append(design_hash)
    if len(set(design_hashes)) != len(design_hashes) or design_hashes != sorted(design_hashes):
        raise ValueError("protocol support strata must be unique and canonically ordered")
    return source, design_hashes


def _validate_design_screen(value: Any) -> Mapping[str, Any]:
    design = _require_exact_keys(value, _DESIGN_KEYS, "protocol stateful design screen")
    if design.get("schema_version") != STATEFUL_DESIGN_SCHEMA_VERSION:
        raise ValueError("protocol stateful design schema_version is unsupported")
    selected_mode = _enum(
        design.get("selected_design_mode"), DESIGN_MODES, "selected_design_mode"
    )
    carryover = _enum(design.get("carryover_scope"), CARRYOVER_SCOPES, "carryover_scope")
    interference = _enum(
        design.get("interference_scope"), INTERFERENCE_SCOPES, "interference_scope"
    )
    temporal = _enum(
        design.get("temporal_variation"), TEMPORAL_VARIATION_SCOPES, "temporal_variation"
    )
    if selected_mode == "clustered_switchback":
        block = _bounded_int(
            design.get("block_length_routes"),
            "block_length_routes",
            minimum=2,
            maximum=MAX_ROUTES_PER_CLUSTER,
        )
        washout = _bounded_int(
            design.get("washout_routes"),
            "washout_routes",
            minimum=0,
            maximum=MAX_ROUTES_PER_CLUSTER,
        )
        if washout >= block:
            raise ValueError("protocol switchback washout must be smaller than block length")
        expected = _design_screen(
            selected_mode,
            carryover_scope=carryover,
            interference_scope=interference,
            temporal_variation=temporal,
            block_length_routes=block,
            washout_routes=washout,
        )
        if dict(design) != expected:
            raise ValueError("protocol stateful design is not a canonical builder result")
        return design

    if design.get("block_length_routes") is not None or design.get("washout_routes") is not None:
        raise ValueError("sticky-session design cannot expose switchback controls")
    candidates = design.get("candidate_screen")
    if not isinstance(candidates, list) or len(candidates) != 3:
        raise ValueError("protocol design candidate screen is incomplete")
    expected_route = {
        "design_mode": "route_randomization",
        "status": "screened_out",
        "blocking_reasons": [
            "stateful_product_campaigns_cannot_randomize_independent_routes_in_v1"
        ],
        "assignment_implementation_available": False,
    }
    if candidates[0] != expected_route:
        raise ValueError("protocol must keep route-level randomization screened out")
    for index, mode in enumerate(DESIGN_MODES, start=1):
        row = _require_exact_keys(
            candidates[index], _DESIGN_CANDIDATE_KEYS, f"protocol design candidate {mode}"
        )
        if row.get("design_mode") != mode or row.get("selected") is not (mode == selected_mode):
            raise ValueError("protocol design candidate selection is inconsistent")
        possible_reasons = [
            _design_reasons(
                mode,
                carryover_scope=carryover,
                interference_scope=interference,
                temporal_variation=temporal,
                washout_routes=washout,
            )
            for washout in (0, 1)
        ]
        if row.get("blocking_reasons") not in possible_reasons:
            raise ValueError("protocol design candidate reasons are not canonical")
        expected_status = (
            "assumptions_declared_unvalidated"
            if not row.get("blocking_reasons")
            else "blocked_for_review"
        )
        if row.get("status") != expected_status or row.get(
            "assignment_implementation_available"
        ) is not False:
            raise ValueError("protocol design candidate status is not canonical")
    selected_reasons = _design_reasons(
        selected_mode,
        carryover_scope=carryover,
        interference_scope=interference,
        temporal_variation=temporal,
        washout_routes=0,
    )
    has_unknown = "unknown" in (carryover, interference, temporal)
    selected_status = (
        "declaration_incomplete"
        if has_unknown
        else (
            "assumptions_declared_unvalidated"
            if not selected_reasons
            else "incompatible_or_unvalidated_assumptions"
        )
    )
    expected_fixed = {
        "selected_design_status": selected_status,
        "assignment_unit": "session_hash",
        "sticky_within_live_session": True,
        "selected_design_blocking_reasons": selected_reasons,
        "carryover_validated": False,
        "interference_validated": False,
        "mixing_validated": False,
        "causal_design_certified": False,
    }
    if any(design.get(key) != expected for key, expected in expected_fixed.items()):
        raise ValueError("protocol sticky-session design claims unsupported semantics")
    return design


def _validate_protocol_semantics(charter: Mapping[str, Any]) -> Tuple[Mapping[str, Any], int]:
    source, design_hashes = _validate_source_studies(charter.get("source_studies"))

    target_class = charter.get("target_policy_class")
    if not isinstance(target_class, Mapping):
        raise ValueError("protocol target-policy class is missing")
    expected_target = _target_policy_class(str(target_class.get("profile_name") or ""))
    if dict(target_class) != expected_target:
        raise ValueError("protocol target-policy class does not match a frozen profile")

    design = _validate_design_screen(charter.get("stateful_design"))
    population = _require_exact_keys(
        charter.get("population"), _POPULATION_KEYS, "protocol population"
    )
    if population.get("schema_version") != POPULATION_SCHEMA_VERSION:
        raise ValueError("protocol population schema_version is unsupported")
    _identifier(population.get("population_rule_id"), "population_rule_id")
    _identifier(population.get("population_rule_version"), "population_rule_version")
    _identifier(population.get("cluster_key_schema_version"), "cluster_key_schema_version")
    planned_clusters = _bounded_int(
        population.get("planned_clusters"),
        "planned_clusters",
        minimum=2,
        maximum=MAX_PLANNED_CLUSTERS,
    )
    max_routes = _bounded_int(
        population.get("max_routes_per_cluster"),
        "max_routes_per_cluster",
        minimum=1,
        maximum=MAX_ROUTES_PER_CLUSTER,
    )
    expected_population_flags = {
        "cluster_identifier_exported": False,
        "study_scoped_pseudonym_required": True,
        "start_after_protocol_seal_required": True,
        "one_study_policy_per_cluster": True,
        "population_precommitted": False,
        "cluster_map_validated": False,
    }
    if any(population.get(key) is not expected for key, expected in expected_population_flags.items()):
        raise ValueError("protocol population must remain draft-only and unvalidated")
    if population.get("admitted_support_strata") != design_hashes:
        raise ValueError("protocol population does not bind the canonical support strata")

    stopping = _require_exact_keys(
        charter.get("stopping_and_resources"), _STOPPING_KEYS, "protocol stopping rules"
    )
    if stopping.get("schema_version") != STOPPING_SCHEMA_VERSION:
        raise ValueError("protocol stopping schema_version is unsupported")
    analysis_every = _bounded_int(
        stopping.get("analysis_every_clusters"),
        "analysis_every_clusters",
        minimum=1,
        maximum=planned_clusters,
    )
    expected_stopping = {
        "schema_version": STOPPING_SCHEMA_VERSION,
        "planned_cluster_ceiling": planned_clusters,
        "planned_route_ceiling": planned_clusters * max_routes,
        "analysis_every_clusters": analysis_every,
        "fixed_analysis_schedule": True,
        "outcome_dependent_stopping_allowed": False,
        "automatic_promotion_allowed": False,
        "resource_guardrail_action": "pause_and_independent_review",
        "rules_precommitted": False,
    }
    if dict(stopping) != expected_stopping:
        raise ValueError("protocol stopping rules are not canonical or fail-closed")

    randomness = _require_exact_keys(
        charter.get("randomness"), _RANDOMNESS_KEYS, "protocol randomness contract"
    )
    expected_randomness = {
        "schema_version": RANDOMNESS_SCHEMA_VERSION,
        "planned_derivation": "hmac_sha256_seed_design_hash_cluster_hash_block_id",
        "seed_commitment": _seed_commitment(randomness.get("seed_commitment")),
        "seed_material_included": False,
        "seed_commitment_sealed": False,
        "caller_selected_nonce_allowed": False,
        "assignment_receipt_required_before_first_route": True,
        "assignment_implementation_available": False,
        "assignment_performed": False,
        "nonce_grinding_resistant": False,
    }
    if dict(randomness) != expected_randomness:
        raise ValueError("protocol randomness contract must remain non-assigning and unsealed")

    outcome = _require_exact_keys(
        charter.get("outcome_observation"), _OUTCOME_KEYS, "protocol outcome observation"
    )
    expected_outcome = {
        "schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
        "contracts": build_route_outcome_contracts(
            precommitted=True, commitment_source="protocol_draft"
        ),
        "observation_process_validated": False,
        "missingness_identified": False,
        "mnar_correction_available": False,
        "rating_nonresponse_treated_as_outcome": False,
    }
    expected_outcome["contract_set_hash"] = _domain_hash(
        _OUTCOME_SET_HASH_DOMAIN, expected_outcome
    )
    if dict(outcome) != expected_outcome:
        raise ValueError("protocol outcome contracts are not the frozen v1 definitions")

    external = _require_exact_keys(
        charter.get("external_evaluation"), _EXTERNAL_KEYS, "protocol external evaluation"
    )
    expected_external = {
        "estimator_id": _optional_identifier(external.get("estimator_id"), "estimator_id"),
        "reviewer_id": _optional_identifier(external.get("reviewer_id"), "reviewer_id"),
        "validated": False,
        "causal_estimate_available": False,
        "winner_available": False,
    }
    if dict(external) != expected_external:
        raise ValueError("protocol external evaluation must remain unvalidated")

    blocker_register = charter.get("blocker_register")
    if not isinstance(blocker_register, list) or any(
        not isinstance(row, Mapping) or set(row) != _BLOCKER_KEYS for row in blocker_register
    ):
        raise ValueError("protocol blocker register does not match the v1 schema")
    expected_blockers = _blocker_register(
        carryover_scope=str(design["carryover_scope"]),
        interference_scope=str(design["interference_scope"]),
        seed_commitment=randomness["seed_commitment"],
    )
    if blocker_register != expected_blockers:
        raise ValueError("protocol blocker register is not canonical")

    prompt_free = _require_exact_keys(
        charter.get("prompt_free_contract"), _PROMPT_FREE_KEYS, "protocol prompt-free contract"
    )
    expected_prompt_free = {
        "prompt_free": True,
        "raw_prompt_included": False,
        "raw_session_id_included": False,
        "free_form_text_fields_allowed": False,
        "canonical_json": "sorted_keys_compact_utf8_no_nan",
    }
    if dict(prompt_free) != expected_prompt_free:
        raise ValueError("protocol prompt-free contract is not canonical")

    boundaries = _require_exact_keys(
        charter.get("causal_boundaries"), _CAUSAL_BOUNDARY_KEYS, "protocol causal boundaries"
    )
    expected_boundaries = {
        "deployment": "shadow_only",
        "protocol_sealed": False,
        "activation_available": False,
        "assignment_available": False,
        "assignment_performed": False,
        "execution_enabled": False,
        "io_performed": False,
        "ledger_write_performed": False,
        "model_inference_performed": False,
        "off_policy_estimate_computed": False,
        "causal_identification_performed": False,
        "power_analysis_performed": False,
        "automatic_promotion_allowed": False,
        "always_valid_inference_available": False,
        "declarations_are_not_validation": True,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
        "interpretation": "prompt-free protocol draft for independent review only",
    }
    if dict(boundaries) != expected_boundaries:
        raise ValueError("protocol causal boundaries must remain exactly fail-closed")
    return design, len(design_hashes)


def audit_route_study_protocol(protocol: Any) -> Dict[str, Any]:
    """Verify the canonical hash and fail-closed boundaries of a draft."""

    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a JSON object")
    if set(protocol) != _TOP_LEVEL_KEYS:
        raise ValueError("protocol top-level fields do not match the v1 schema")
    if protocol.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unsupported route study protocol schema_version")
    protocol_meta = protocol.get("protocol")
    charter = protocol.get("charter")
    if not isinstance(protocol_meta, Mapping) or set(protocol_meta) != _PROTOCOL_KEYS:
        raise ValueError("protocol metadata does not match the v1 schema")
    if not isinstance(charter, Mapping) or set(charter) != _CHARTER_KEYS:
        raise ValueError("protocol charter does not match the v1 schema")
    expected_protocol_meta = {
        "version": PROTOCOL_VERSION,
        "label": PROTOCOL_LABEL,
        "state": "draft_for_independent_review",
        "activation_available": False,
    }
    if dict(protocol_meta) != expected_protocol_meta:
        raise ValueError("protocol metadata is not the canonical fail-closed v1 draft")
    semantic_design, semantic_stratum_count = _validate_protocol_semantics(charter)
    boundaries = charter.get("causal_boundaries")
    if not isinstance(boundaries, Mapping):
        raise ValueError("protocol causal boundaries are missing")
    forbidden_true = (
        "protocol_sealed",
        "activation_available",
        "assignment_available",
        "assignment_performed",
        "execution_enabled",
        "io_performed",
        "ledger_write_performed",
        "model_inference_performed",
        "off_policy_estimate_computed",
        "causal_identification_performed",
        "power_analysis_performed",
        "automatic_promotion_allowed",
        "always_valid_inference_available",
    )
    if any(boundaries.get(key) is not False for key in forbidden_true):
        raise ValueError("protocol causal boundaries must remain fail-closed")
    if boundaries.get("activation_blockers") != list(ACTIVATION_BLOCKERS):
        raise ValueError("protocol must retain every activation blocker")

    blocker_register = charter.get("blocker_register")
    if not isinstance(blocker_register, Sequence) or isinstance(
        blocker_register, (str, bytes)
    ):
        raise ValueError("protocol blocker register is missing")
    blocker_codes = [
        str(row.get("code") or "") if isinstance(row, Mapping) else ""
        for row in blocker_register
    ]
    if blocker_codes != list(ACTIVATION_BLOCKERS) or any(
        not isinstance(row, Mapping) or row.get("activation_blocking") is not True
        for row in blocker_register
    ):
        raise ValueError("protocol blocker register must retain every blocking gate")

    target_class = charter.get("target_policy_class")
    if not isinstance(target_class, Mapping):
        raise ValueError("protocol target-policy class is missing")
    expected_target = _target_policy_class(str(target_class.get("profile_name") or ""))
    if dict(target_class) != expected_target:
        raise ValueError("protocol target-policy class does not match a frozen profile")

    design = charter.get("stateful_design")
    if not isinstance(design, Mapping):
        raise ValueError("protocol stateful design screen is missing")
    if design.get("selected_design_mode") not in DESIGN_MODES:
        raise ValueError("protocol selected design mode is unsupported")
    if (
        design.get("assignment_implementation_available") is True
        or design.get("causal_design_certified") is not False
        or design.get("carryover_validated") is not False
        or design.get("interference_validated") is not False
        or design.get("mixing_validated") is not False
    ):
        raise ValueError("protocol design declarations cannot claim validation")

    randomness = charter.get("randomness")
    if not isinstance(randomness, Mapping) or any(
        randomness.get(key) is not False
        for key in (
            "seed_material_included",
            "seed_commitment_sealed",
            "caller_selected_nonce_allowed",
            "assignment_implementation_available",
            "assignment_performed",
            "nonce_grinding_resistant",
        )
    ):
        raise ValueError("protocol randomness contract must remain non-assigning and unsealed")
    external = charter.get("external_evaluation")
    if not isinstance(external, Mapping) or any(
        external.get(key) is not False
        for key in ("validated", "causal_estimate_available", "winner_available")
    ):
        raise ValueError("protocol external evaluation must remain unvalidated")
    outcome = charter.get("outcome_observation")
    if not isinstance(outcome, Mapping) or any(
        outcome.get(key) is not False
        for key in (
            "observation_process_validated",
            "missingness_identified",
            "mnar_correction_available",
            "rating_nonresponse_treated_as_outcome",
        )
    ):
        raise ValueError("protocol outcome observation must remain unvalidated")

    source_studies = charter.get("source_studies")
    strata = source_studies.get("support_strata") if isinstance(source_studies, Mapping) else None
    if not isinstance(strata, Sequence) or isinstance(strata, (str, bytes)) or not strata:
        raise ValueError("protocol must bind at least one support stratum")
    if any(
        not isinstance(row, Mapping) or row.get("executed_logging_propensities") is not False
        for row in strata
    ):
        raise ValueError("protocol strata cannot claim executed logging propensities")

    supplied_hash = str(protocol.get("protocol_hash") or "").strip()
    payload = {
        "schema_version": protocol["schema_version"],
        "protocol": dict(protocol_meta),
        "charter": dict(charter),
    }
    expected_hash = _domain_hash(_PROTOCOL_HASH_DOMAIN, payload)
    if supplied_hash != expected_hash:
        raise ValueError("protocol_hash does not match the canonical protocol draft")
    return {
        "ok": True,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_hash": supplied_hash,
        "state": "draft_for_independent_review",
        "support_stratum_count": semantic_stratum_count,
        "selected_design_mode": str(semantic_design["selected_design_mode"]),
        "selected_design_status": str(semantic_design["selected_design_status"]),
        "verification_level": "structural_without_source_plans",
        "source_plan_reconstruction_performed": False,
        "authenticity_proof_available": False,
        "trusted_timestamp_available": False,
        "activation_available": False,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
    }


def build_route_study_review_bundle(
    study_plans: Any,
    **protocol_options: Any,
) -> Dict[str, Any]:
    """Create a portable, prompt-free bundle that supports full reconstruction.

    The bundle is still an unsigned, untimestamped review artifact.  It binds
    canonical source plans to the exact protocol builder options and result, but
    it does not prove authorship, registration time, causal validity, or safety
    to activate.
    """

    unknown = set(protocol_options) - set(PROTOCOL_OPTION_KEYS)
    if unknown:
        raise ValueError(
            "review bundle contains unsupported protocol options: "
            + ", ".join(sorted(map(str, unknown)))
        )
    validated = _normalize_study_plans(study_plans)
    canonical_plans = [row["plan"] for row in validated]
    options = _normalize_protocol_options(**protocol_options)
    protocol = build_route_study_protocol(canonical_plans, **options)
    payload = {
        "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
        "bundle": {
            "version": REVIEW_BUNDLE_VERSION,
            "label": REVIEW_BUNDLE_LABEL,
            "state": "ready_for_semantic_review",
            "verification_level": "full_source_bound_reconstruction",
            "authenticity_proof_available": False,
            "trusted_timestamp_available": False,
            "activation_available": False,
        },
        "protocol_builder": {
            "schema_version": PROTOCOL_BUILD_INPUT_SCHEMA_VERSION,
            "options": options,
        },
        "source_study_plans": canonical_plans,
        "protocol": protocol,
    }
    return {**payload, "bundle_hash": _domain_hash(_REVIEW_BUNDLE_HASH_DOMAIN, payload)}


def build_route_study_review_bundle_from_input(payload: Any) -> Dict[str, Any]:
    """Build a review bundle from the shared closed prompt-free input schema."""

    study_plans, options = _split_protocol_build_input(payload)
    return build_route_study_review_bundle(study_plans, **options)


def audit_route_study_review_bundle(bundle: Any) -> Dict[str, Any]:
    """Reconstruct a bundled protocol from its complete canonical source plans."""

    if not isinstance(bundle, Mapping):
        raise ValueError("review bundle must be a JSON object")
    if set(bundle) != _BUNDLE_TOP_LEVEL_KEYS:
        raise ValueError("review bundle top-level fields do not match the v1 schema")
    if bundle.get("schema_version") != REVIEW_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported route study review bundle schema_version")
    meta = _require_exact_keys(bundle.get("bundle"), _BUNDLE_META_KEYS, "review bundle metadata")
    expected_meta = {
        "version": REVIEW_BUNDLE_VERSION,
        "label": REVIEW_BUNDLE_LABEL,
        "state": "ready_for_semantic_review",
        "verification_level": "full_source_bound_reconstruction",
        "authenticity_proof_available": False,
        "trusted_timestamp_available": False,
        "activation_available": False,
    }
    if dict(meta) != expected_meta:
        raise ValueError("review bundle metadata is not the canonical fail-closed v1 contract")

    builder = _require_exact_keys(
        bundle.get("protocol_builder"),
        {"schema_version", "options"},
        "review bundle protocol builder",
    )
    if builder.get("schema_version") != PROTOCOL_BUILD_INPUT_SCHEMA_VERSION:
        raise ValueError("review bundle protocol builder schema_version is unsupported")
    raw_options = builder.get("options")
    if not isinstance(raw_options, Mapping) or set(raw_options) != set(PROTOCOL_OPTION_KEYS):
        raise ValueError("review bundle protocol options are incomplete or unsupported")
    canonical_options = _normalize_protocol_options(**dict(raw_options))
    if dict(raw_options) != canonical_options:
        raise ValueError("review bundle protocol options are not canonical")

    raw_plans = bundle.get("source_study_plans")
    if not isinstance(raw_plans, list):
        raise ValueError("review bundle source_study_plans must be a canonical list")
    validated_plans = _normalize_study_plans(raw_plans)
    canonical_plans = [row["plan"] for row in validated_plans]
    if raw_plans != canonical_plans:
        raise ValueError("review bundle source plans are not canonical or canonically ordered")

    supplied_protocol = bundle.get("protocol")
    structural = audit_route_study_protocol(supplied_protocol)
    reconstructed = build_route_study_protocol(canonical_plans, **canonical_options)
    if _canonical_json(supplied_protocol) != _canonical_json(reconstructed):
        raise ValueError("review bundle protocol does not reconstruct from its source plans")

    supplied_hash = _sha256_digest(bundle.get("bundle_hash"), "bundle_hash")
    payload = {
        "schema_version": bundle["schema_version"],
        "bundle": dict(meta),
        "protocol_builder": {
            "schema_version": builder["schema_version"],
            "options": dict(raw_options),
        },
        "source_study_plans": raw_plans,
        "protocol": supplied_protocol,
    }
    expected_hash = _domain_hash(_REVIEW_BUNDLE_HASH_DOMAIN, payload)
    if supplied_hash != expected_hash:
        raise ValueError("bundle_hash does not match the canonical review bundle")
    return {
        "ok": True,
        "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
        "bundle_hash": supplied_hash,
        "protocol_hash": structural["protocol_hash"],
        "state": "ready_for_semantic_review",
        "support_stratum_count": len(canonical_plans),
        "verification_level": "full_source_bound_reconstruction",
        "source_plan_reconstruction_performed": True,
        "authenticity_proof_available": False,
        "trusted_timestamp_available": False,
        "activation_available": False,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
    }


__all__ = [
    "CARRYOVER_SCOPES",
    "DEFAULT_ANALYSIS_EVERY_CLUSTERS",
    "DEFAULT_MAX_ROUTES_PER_CLUSTER",
    "DEFAULT_PLANNED_CLUSTERS",
    "DESIGN_MODES",
    "INTERFERENCE_SCOPES",
    "PROTOCOL_BUILD_INPUT_KEYS",
    "PROTOCOL_BUILD_INPUT_SCHEMA_VERSION",
    "PROTOCOL_LABEL",
    "PROTOCOL_OPTION_KEYS",
    "PROTOCOL_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "REVIEW_BUNDLE_LABEL",
    "REVIEW_BUNDLE_SCHEMA_VERSION",
    "REVIEW_BUNDLE_VERSION",
    "TEMPORAL_VARIATION_SCOPES",
    "audit_route_study_review_bundle",
    "audit_route_study_protocol",
    "build_route_study_protocol_from_input",
    "build_route_study_review_bundle",
    "build_route_study_review_bundle_from_input",
    "build_route_study_protocol",
]
