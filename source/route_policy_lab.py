"""Honest, deterministic evidence summaries for Supermix route policies.

This module deliberately does *not* estimate counterfactual policy value.  It
can replay a threshold profile against historical route decisions and describe
outcomes for the subset where the candidate agrees with the observed action.
Those matched outcomes are associational: deterministic product logs do not
identify what would have happened under a different route.

The propensity checks are intentionally strict.  They describe whether a row
contains the minimum logging metadata a separate, validated off-policy
evaluator would need; passing them is not itself an off-policy estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

try:
    from .route_policy_ledger import SUPPORT_SCHEMA_VERSION, build_logging_support_envelope
except ImportError:  # Runtime entry points add ``source`` directly to sys.path.
    from route_policy_ledger import SUPPORT_SCHEMA_VERSION, build_logging_support_envelope


AUTO_AGENT_MODE_ORDER: Tuple[str, ...] = (
    "off",
    "collective",
    "loop",
    "collective_loop",
)


@dataclass(frozen=True)
class RoutePolicyProfile:
    """Immutable score thresholds for the three non-trivial route modes."""

    name: str
    collective_min_score: int
    loop_min_score: int
    collective_loop_min_score: int

    def __post_init__(self) -> None:
        thresholds = (
            self.collective_min_score,
            self.loop_min_score,
            self.collective_loop_min_score,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in thresholds):
            raise TypeError("profile thresholds must be integers")
        if not (0 <= thresholds[0] <= thresholds[1] <= thresholds[2]):
            raise ValueError("profile thresholds must be non-negative and monotonic")

    @property
    def thresholds(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "off": 0,
                "collective": self.collective_min_score,
                "loop": self.loop_min_score,
                "collective_loop": self.collective_loop_min_score,
            }
        )

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "thresholds": dict(self.thresholds)}


POLICY_PROFILES: Mapping[str, RoutePolicyProfile] = MappingProxyType(
    {
        "efficiency": RoutePolicyProfile("efficiency", 3, 5, 6),
        "balanced": RoutePolicyProfile("balanced", 2, 4, 5),
        "quality_first": RoutePolicyProfile("quality_first", 1, 3, 4),
    }
)


_MISSING = object()
_POSITIVE_RATINGS = frozenset(("up", "good", "approve", "approved", "positive"))
_NEGATIVE_RATINGS = frozenset(("down", "bad", "reject", "rejected", "negative"))
READINESS_SCHEMA_VERSION = "route-readiness-v2"
READINESS_MIN_TARGET_PROBABILITY = 0.05
READINESS_MIN_GLOBAL_EFFECTIVE_SAMPLE_SIZE = 20.0
READINESS_MIN_PER_ACTION_EFFECTIVE_SAMPLE_SIZE = 10.0


def get_policy_profile(profile: Union[str, RoutePolicyProfile]) -> RoutePolicyProfile:
    if isinstance(profile, RoutePolicyProfile):
        return profile
    key = str(profile or "").strip().lower()
    try:
        return POLICY_PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(POLICY_PROFILES)
        raise ValueError(f"unknown route policy profile {profile!r}; choose one of: {choices}") from exc


def ordered_allowed_modes(modes: Optional[Iterable[Any]]) -> Tuple[str, ...]:
    """Return valid modes once each, in the canonical shallow-to-deep order.

    ``off`` is always retained as the safe fallback, including for empty or
    malformed eligibility lists.
    """

    supplied = set()
    if modes is not None and not isinstance(modes, (str, bytes)):
        supplied = {str(mode).strip() for mode in modes if str(mode).strip() in AUTO_AGENT_MODE_ORDER}
    supplied.add("off")
    return tuple(mode for mode in AUTO_AGENT_MODE_ORDER if mode in supplied)


def select_profile_action(
    score: Any,
    allowed_modes: Optional[Iterable[Any]],
    profile: Union[str, RoutePolicyProfile] = "balanced",
) -> str:
    """Choose the deepest allowed mode whose immutable threshold is met."""

    selected_profile = get_policy_profile(profile)
    try:
        cooked_score = int(score)
    except (TypeError, ValueError, OverflowError):
        cooked_score = 0
    cooked_score = max(0, cooked_score)
    allowed = set(ordered_allowed_modes(allowed_modes))
    thresholds = selected_profile.thresholds
    for mode in reversed(AUTO_AGENT_MODE_ORDER):
        if mode in allowed and cooked_score >= thresholds[mode]:
            return mode
    return "off"


def profile_decision(
    score: Any,
    allowed_modes: Optional[Iterable[Any]],
    profile: Union[str, RoutePolicyProfile] = "balanced",
) -> Dict[str, Any]:
    selected_profile = get_policy_profile(profile)
    ordered = ordered_allowed_modes(allowed_modes)
    try:
        cooked_score = max(0, int(score))
    except (TypeError, ValueError, OverflowError):
        cooked_score = 0
    return {
        "profile": selected_profile.name,
        "thresholds": dict(selected_profile.thresholds),
        "score": cooked_score,
        "allowed_agent_modes": list(ordered),
        "selected_agent_mode": select_profile_action(cooked_score, ordered, selected_profile),
    }


def _finite_probability(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(cooked) or cooked < 0.0 or cooked > 1.0:
        return None
    return cooked


def propensity_readiness(
    policy: Any,
    *,
    chosen_action: str,
    target_action: Optional[str] = None,
    expected_policy_id: Optional[str] = None,
    expected_policy_version: Optional[str] = None,
    expected_feature_schema: Optional[str] = None,
    expected_context: Any = _MISSING,
) -> Dict[str, Any]:
    """Validate strict post-filter behavior-policy logging for one decision.

    A ready row must be explicitly stochastic, use the exact requested policy
    version and decision context, list at least two eligible actions, and carry
    a normalized post-filter probability vector with positive probability for
    the chosen action (and target action, when supplied).
    """

    reasons = []
    row = policy if isinstance(policy, Mapping) else {}

    expected_id = str(expected_policy_id or "").strip()
    actual_id = str(row.get("policy_id") or "").strip()
    if expected_id and actual_id != expected_id:
        reasons.append("policy_id_mismatch")

    expected_version = str(expected_policy_version or "").strip()
    actual_version = str(row.get("policy_version") or "").strip()
    if not expected_version:
        reasons.append("expected_policy_version_required")
    elif actual_version != expected_version:
        reasons.append("policy_version_mismatch")

    if expected_context is _MISSING:
        reasons.append("expected_context_required")
    elif "decision_context" not in row or row.get("decision_context") != expected_context:
        reasons.append("decision_context_mismatch")

    decision_type = str(row.get("decision_type") or "").strip().lower()
    if decision_type not in {"stochastic", "randomized"}:
        reasons.append("decision_not_explicitly_stochastic")

    expected_schema = str(expected_feature_schema or "").strip()
    actual_schema = str(row.get("feature_schema_version") or "").strip()
    if expected_schema and actual_schema != expected_schema:
        reasons.append("feature_schema_mismatch")

    if str(row.get("probability_stage") or "").strip() != "post_filter":
        reasons.append("probabilities_not_post_filter")

    eligible_raw = row.get("eligible_actions")
    eligible: Tuple[str, ...] = ()
    if not isinstance(eligible_raw, (list, tuple)) or isinstance(eligible_raw, (str, bytes)):
        reasons.append("eligible_actions_missing")
    else:
        eligible_values = [str(action).strip() for action in eligible_raw]
        if (
            len(eligible_values) < 2
            or len(set(eligible_values)) != len(eligible_values)
            or any(action not in AUTO_AGENT_MODE_ORDER for action in eligible_values)
        ):
            reasons.append("eligible_actions_invalid")
        else:
            eligible = tuple(eligible_values)

    probabilities_raw = row.get("post_filter_action_probabilities")
    probabilities: Dict[str, float] = {}
    if not isinstance(probabilities_raw, Mapping):
        reasons.append("post_filter_probability_vector_missing")
    else:
        invalid_probability = False
        for action, value in probabilities_raw.items():
            action_key = str(action).strip()
            probability = _finite_probability(value)
            if action_key not in AUTO_AGENT_MODE_ORDER or probability is None:
                invalid_probability = True
                continue
            probabilities[action_key] = probability
        if invalid_probability:
            reasons.append("post_filter_probability_vector_invalid")
        if eligible and set(probabilities) != set(eligible):
            reasons.append("probability_vector_not_exactly_eligible_actions")
        if probabilities and not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            reasons.append("probabilities_do_not_sum_to_one")

    positive_actions = tuple(
        action for action in AUTO_AGENT_MODE_ORDER if probabilities.get(action, 0.0) > 0.0
    )
    if len(positive_actions) < 2:
        reasons.append("insufficient_randomized_support")

    chosen = str(chosen_action or "").strip()
    if chosen not in eligible or probabilities.get(chosen, 0.0) <= 0.0:
        reasons.append("chosen_action_has_no_positive_probability")

    target = str(target_action or "").strip()
    if target_action is not None and (target not in eligible or probabilities.get(target, 0.0) <= 0.0):
        reasons.append("target_action_has_no_positive_probability")

    support = row.get("logging_support")
    if not isinstance(support, Mapping):
        reasons.append("logging_support_missing")
    else:
        support_schema = str(support.get("schema_version") or "").strip()
        if support_schema != SUPPORT_SCHEMA_VERSION:
            reasons.append("support_schema_mismatch")

        support_decision_type = str(support.get("decision_type") or "").strip().lower()
        if support_decision_type == "stochastic":
            support_decision_type = "randomized"
        normalized_row_decision_type = "randomized" if decision_type == "stochastic" else decision_type
        if support_decision_type != normalized_row_decision_type:
            reasons.append("support_decision_type_mismatch")

        support_stage = str(support.get("probability_stage") or "").strip()
        if support_stage != "post_filter" or support_stage != str(row.get("probability_stage") or "").strip():
            reasons.append("support_probability_stage_mismatch")

        sampler = support.get("sampler") if isinstance(support.get("sampler"), Mapping) else {}
        assignment_commitment = str(sampler.get("assignment_commitment") or "").strip()
        if support_decision_type == "randomized" and not assignment_commitment:
            reasons.append("assignment_commitment_missing")

        if eligible and set(probabilities) == set(eligible) and chosen in eligible:
            try:
                canonical_support = build_logging_support_envelope(
                    support,
                    eligible_modes=eligible,
                    action_probabilities=probabilities,
                    chosen_mode=chosen,
                )
            except (TypeError, ValueError):
                reasons.append("logging_support_invalid")
            else:
                canonical_chosen_probability = float(canonical_support["chosen_probability"])
                support_chosen_probability = _finite_probability(support.get("chosen_probability"))
                row_logging_propensity = _finite_probability(row.get("logging_propensity"))
                if (
                    support_chosen_probability is None
                    or not math.isclose(
                        support_chosen_probability,
                        canonical_chosen_probability,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    reasons.append("chosen_probability_mismatch")
                if (
                    row_logging_propensity is None
                    or not math.isclose(
                        row_logging_propensity,
                        canonical_chosen_probability,
                        rel_tol=0.0,
                        abs_tol=1e-9,
                    )
                ):
                    reasons.append("logging_propensity_mismatch")

                canonical_candidate_hash = str(canonical_support["candidate_set_hash"])
                if (
                    str(support.get("candidate_set_hash") or "") != canonical_candidate_hash
                    or str(row.get("candidate_set_hash") or "") != canonical_candidate_hash
                ):
                    reasons.append("candidate_set_hash_mismatch")

                canonical_distribution_hash = str(canonical_support["distribution_hash"])
                if (
                    str(support.get("distribution_hash") or "") != canonical_distribution_hash
                    or str(row.get("distribution_hash") or "") != canonical_distribution_hash
                ):
                    reasons.append("distribution_hash_mismatch")

    # Preserve first-occurrence order while avoiding noisy duplicate failures.
    reasons = list(dict.fromkeys(reasons))
    return {
        "ready": not reasons,
        "reasons": reasons,
        "policy_id": actual_id or None,
        "policy_version": actual_version or None,
        "feature_schema_version": actual_schema or None,
        "chosen_action": chosen or None,
        "chosen_probability": probabilities.get(chosen),
        "target_action": target or None,
        "target_probability": probabilities.get(target) if target else None,
        "eligible_actions": list(eligible),
        "positive_support_actions": list(positive_actions),
        "post_filter_action_probabilities": {
            action: round(probabilities[action], 9)
            for action in AUTO_AGENT_MODE_ORDER
            if action in probabilities
        },
        "causal_claim": False,
        "off_policy_estimate": False,
    }


def _rows(rows: Optional[Sequence[Any]]) -> Tuple[Mapping[str, Any], ...]:
    return tuple(row for row in (rows or ()) if isinstance(row, Mapping))


def _route_index(rows: Sequence[Mapping[str, Any]]) -> Tuple[Dict[str, Mapping[str, Any]], Dict[str, int]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    with_id = 0
    for row in rows:
        route_id = str(row.get("route_id") or "").strip()
        if not route_id:
            continue
        with_id += 1
        indexed[route_id] = row
    return indexed, {
        "rows": len(rows),
        "rows_with_route_id": with_id,
        "rows_without_route_id": len(rows) - with_id,
        "unique_route_ids": len(indexed),
        "duplicate_route_ids": with_id - len(indexed),
    }


def _score(row: Mapping[str, Any]) -> Optional[int]:
    policy = row.get("auto_agent_policy") if isinstance(row.get("auto_agent_policy"), Mapping) else {}
    raw = policy.get("score")
    if raw is None:
        raw = row.get("score")
    if isinstance(raw, bool):
        return None
    try:
        cooked = int(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if cooked < 0:
        return None
    if isinstance(raw, float) and (not math.isfinite(raw) or not raw.is_integer()):
        return None
    return cooked


def _observed_mode(row: Mapping[str, Any]) -> Optional[str]:
    policy = row.get("auto_agent_policy") if isinstance(row.get("auto_agent_policy"), Mapping) else {}
    mode = str(
        row.get("chosen_agent_mode")
        or row.get("chosen_mode")
        or row.get("selected_agent_mode")
        or policy.get("selected_agent_mode")
        or ""
    ).strip()
    return mode if mode in AUTO_AGENT_MODE_ORDER else None


def _durable_execution_state(row: Mapping[str, Any]) -> Dict[str, Any]:
    status = str(row.get("status") or row.get("route_status") or "").strip().lower()
    status_valid = status in {"inflight", "completed", "failed"}
    success = row.get("success")
    if status == "completed":
        status_success_consistent = success is True
    elif status == "failed":
        status_success_consistent = success is False
    elif status == "inflight":
        status_success_consistent = success is None
    else:
        status_success_consistent = False

    chosen_mode = _observed_mode(row)
    executed_text = str(
        row.get("executed_agent_mode") or row.get("executed_mode") or ""
    ).strip()
    executed_mode = executed_text if executed_text in AUTO_AGENT_MODE_ORDER else None
    successful_terminal = bool(status == "completed" and success is True)
    mode_mismatch = bool(
        chosen_mode is not None
        and executed_mode is not None
        and chosen_mode != executed_mode
    )
    execution_mode_verified = bool(
        successful_terminal
        and chosen_mode is not None
        and executed_mode is not None
        and not mode_mismatch
    )
    return {
        "status": status,
        "status_valid": status_valid,
        "status_success_consistent": status_success_consistent,
        "successful_terminal": successful_terminal,
        "chosen_mode": chosen_mode,
        "executed_mode": executed_mode,
        "executed_mode_present": bool(executed_text),
        "mode_mismatch": mode_mismatch,
        "execution_mode_verified": execution_mode_verified,
    }


def _allowed_modes(row: Mapping[str, Any]) -> Tuple[Tuple[str, ...], bool]:
    policy = row.get("auto_agent_policy") if isinstance(row.get("auto_agent_policy"), Mapping) else {}
    raw = policy.get("allowed_agent_modes")
    if raw is None:
        raw = row.get("allowed_agent_modes")
    explicit = isinstance(raw, (list, tuple)) and not isinstance(raw, (str, bytes))
    if not explicit:
        return AUTO_AGENT_MODE_ORDER, False
    return ordered_allowed_modes(raw), True


def _quality_outcome(row: Mapping[str, Any]) -> Optional[int]:
    axes = row.get("feedback_axes")
    if isinstance(axes, Mapping) and "quality" in axes:
        quality = axes.get("quality")
        if quality is None:
            return None
        try:
            cooked = float(quality)
        except (TypeError, ValueError, OverflowError):
            return None
        if cooked > 0:
            return 1
        if cooked < 0:
            return 0
        return None

    delta = row.get("score_delta")
    if delta is not None and not isinstance(delta, bool):
        try:
            cooked_delta = float(delta)
        except (TypeError, ValueError, OverflowError):
            cooked_delta = 0.0
        if cooked_delta > 0:
            return 1
        if cooked_delta < 0:
            return 0

    rating = str(row.get("rating") or "").strip().lower()
    if rating in _POSITIVE_RATINGS:
        return 1
    if rating in _NEGATIVE_RATINGS:
        return 0
    return None


def _actual_economics(
    usage: Mapping[str, Any], feedback: Mapping[str, Any]
) -> Mapping[str, Any]:
    for row in (usage, feedback):
        economics = row.get("route_economics") if isinstance(row.get("route_economics"), Mapping) else {}
        actual = economics.get("actual") if isinstance(economics.get("actual"), Mapping) else {}
        if actual:
            return actual
    return {}


def _finite_metric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(cooked) or cooked < 0.0:
        return None
    return cooked


def _ratio(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 6) if denominator else None


def _wilson_interval(approved: int, total: int, z: float = 1.959963984540054) -> Dict[str, Any]:
    if total <= 0:
        return {
            "method": "descriptive_unweighted_wilson",
            "confidence_level": 0.95,
            "lower_bound": None,
            "upper_bound": None,
            "causal": False,
        }
    probability = approved / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (probability + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt((probability * (1.0 - probability) + z2 / (4.0 * total)) / total) / denominator
    return {
        "method": "descriptive_unweighted_wilson",
        "confidence_level": 0.95,
        "lower_bound": round(max(0.0, center - radius), 6),
        "upper_bound": round(min(1.0, center + radius), 6),
        "causal": False,
    }


def _reason_counts(results: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in results:
        for reason in result.get("reasons") or ():
            key = str(reason)
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _effective_sample_size(weights: Sequence[float]) -> float:
    total = sum(float(weight) for weight in weights)
    squared = sum(float(weight) * float(weight) for weight in weights)
    return round((total * total) / squared, 6) if total > 0.0 and squared > 0.0 else 0.0


def _expected_context_for_route(
    route_id: str,
    *,
    expected_context: Any,
    expected_context_by_route_id: Optional[Mapping[str, Any]],
) -> Any:
    if expected_context_by_route_id is not None:
        return expected_context_by_route_id.get(route_id, _MISSING)
    return expected_context


def analyze_route_policy(
    usage_rows: Optional[Sequence[Any]],
    feedback_rows: Optional[Sequence[Any]],
    *,
    profile: Union[str, RoutePolicyProfile] = "balanced",
    expected_policy_id: Optional[str] = None,
    expected_policy_version: Optional[str] = None,
    expected_feature_schema: Optional[str] = None,
    expected_context: Any = _MISSING,
    expected_context_by_route_id: Optional[Mapping[str, Any]] = None,
    min_overlap_routes: int = 20,
    min_target_probability: float = READINESS_MIN_TARGET_PROBABILITY,
    min_global_effective_sample_size: float = READINESS_MIN_GLOBAL_EFFECTIVE_SAMPLE_SIZE,
    min_per_action_effective_sample_size: float = READINESS_MIN_PER_ACTION_EFFECTIVE_SAMPLE_SIZE,
    durable_lifecycle: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an associational, exact-route-id policy replay report.

    The report never imputes outcomes for changed actions.  Only candidate
    decisions that exactly match the historical action contribute to the
    matched approval/cost/latency description.
    """

    if isinstance(min_overlap_routes, bool) or int(min_overlap_routes) < 1:
        raise ValueError("min_overlap_routes must be at least 1")
    min_overlap_routes = int(min_overlap_routes)
    try:
        min_target_probability = float(min_target_probability)
        min_global_effective_sample_size = float(min_global_effective_sample_size)
        min_per_action_effective_sample_size = float(min_per_action_effective_sample_size)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("readiness thresholds must be finite numbers") from exc
    if not math.isfinite(min_target_probability) or not 0.0 < min_target_probability <= 1.0:
        raise ValueError("min_target_probability must be in (0, 1]")
    if (
        not math.isfinite(min_global_effective_sample_size)
        or min_global_effective_sample_size <= 0.0
        or not math.isfinite(min_per_action_effective_sample_size)
        or min_per_action_effective_sample_size <= 0.0
    ):
        raise ValueError("effective sample size thresholds must be positive")
    if expected_context_by_route_id is not None and not isinstance(expected_context_by_route_id, Mapping):
        raise ValueError("expected_context_by_route_id must be a mapping")
    selected_profile = get_policy_profile(profile)
    usage = _rows(usage_rows)
    feedback = _rows(feedback_rows)
    usage_index, usage_counts = _route_index(usage)
    feedback_index, feedback_counts = _route_index(feedback)
    joined_ids = tuple(sorted(set(usage_index) & set(feedback_index)))
    lifecycle_source = durable_lifecycle if isinstance(durable_lifecycle, Mapping) else None
    durable_lifecycle_present = lifecycle_source is not None

    def fingerprint_valid(row: Mapping[str, Any]) -> bool:
        if row.get("decision_record_fingerprint_valid") is True:
            return True
        policy = row.get("auto_agent_policy")
        return bool(
            isinstance(policy, Mapping)
            and policy.get("decision_record_fingerprint_valid") is True
        )

    def fingerprint_reason(row: Mapping[str, Any]) -> str:
        reason = row.get("decision_record_fingerprint_reason")
        if not reason and isinstance(row.get("auto_agent_policy"), Mapping):
            reason = row["auto_agent_policy"].get("decision_record_fingerprint_reason")
        return str(reason or "missing_unverifiable")

    def outcome_contract_precommitted(row: Mapping[str, Any]) -> bool:
        if row.get("outcome_contracts_precommitted_at_begin") is True:
            return True
        policy = row.get("auto_agent_policy")
        return bool(
            isinstance(policy, Mapping)
            and policy.get("outcome_contracts_precommitted_at_begin") is True
        )

    fingerprint_valid_routes = sum(1 for row in usage_index.values() if fingerprint_valid(row))
    fingerprint_invalid_reason_counts: Dict[str, int] = {}
    for row in usage_index.values():
        if fingerprint_valid(row):
            continue
        reason = fingerprint_reason(row)
        fingerprint_invalid_reason_counts[reason] = fingerprint_invalid_reason_counts.get(reason, 0) + 1
    decision_record_fingerprint_complete = bool(
        not durable_lifecycle_present
        or (
            usage_counts["unique_route_ids"] > 0
            and fingerprint_valid_routes == usage_counts["unique_route_ids"]
        )
    )
    precommitted_outcome_contract_routes = sum(
        1 for row in usage_index.values() if outcome_contract_precommitted(row)
    )
    outcome_contract_integrity_complete = bool(
        not durable_lifecycle_present
        or (
            usage_counts["unique_route_ids"] > 0
            and precommitted_outcome_contract_routes
            == usage_counts["unique_route_ids"]
        )
    )

    missing_or_invalid_score_ids = {
        route_id for route_id, row in usage_index.items() if _score(row) is None
    }
    missing_or_invalid_chosen_mode_ids = {
        route_id for route_id, row in usage_index.items() if _observed_mode(row) is None
    }
    unevaluable_usage_ids = missing_or_invalid_score_ids | missing_or_invalid_chosen_mode_ids
    orphan_feedback_ids = set(feedback_index) - set(usage_index)
    population_integrity_complete = bool(
        not durable_lifecycle_present
        or (
            usage_counts["rows"] > 0
            and usage_counts["rows_without_route_id"] == 0
            and usage_counts["duplicate_route_ids"] == 0
            and feedback_counts["rows_without_route_id"] == 0
            and feedback_counts["duplicate_route_ids"] == 0
            and not unevaluable_usage_ids
            and not orphan_feedback_ids
        )
    )

    execution_states = {
        route_id: _durable_execution_state(row) for route_id, row in usage_index.items()
    }
    row_status_counts = {"completed": 0, "failed": 0, "inflight": 0, "unknown": 0}
    for state in execution_states.values():
        status = str(state["status"])
        if bool(state["status_valid"]):
            row_status_counts[status] += 1
        else:
            row_status_counts["unknown"] += 1
    status_success_mismatch_routes = sum(
        1 for state in execution_states.values() if not bool(state["status_success_consistent"])
    )
    successful_terminal_routes = sum(
        1 for state in execution_states.values() if bool(state["successful_terminal"])
    )
    unverified_successful_execution_routes = sum(
        1
        for state in execution_states.values()
        if bool(state["successful_terminal"]) and not bool(state["execution_mode_verified"])
    )
    missing_executed_mode_routes = sum(
        1
        for state in execution_states.values()
        if bool(state["successful_terminal"]) and not bool(state["executed_mode_present"])
    )
    invalid_executed_mode_routes = sum(
        1
        for state in execution_states.values()
        if bool(state["successful_terminal"])
        and bool(state["executed_mode_present"])
        and state["executed_mode"] is None
    )
    chosen_executed_mode_mismatch_routes = sum(
        1 for state in execution_states.values() if bool(state["mode_mismatch"])
    )
    feedback_route_status_mismatch_routes = sum(
        1
        for route_id in joined_ids
        if str(feedback_index[route_id].get("route_status") or "").strip().lower()
        and str(feedback_index[route_id].get("route_status") or "").strip().lower()
        != str(execution_states[route_id]["status"])
    )
    execution_integrity_complete = bool(
        not durable_lifecycle_present
        or (
            row_status_counts["unknown"] == 0
            and status_success_mismatch_routes == 0
            and unverified_successful_execution_routes == 0
            and chosen_executed_mode_mismatch_routes == 0
            and feedback_route_status_mismatch_routes == 0
        )
    )

    usage_scored = sum(1 for row in usage_index.values() if _score(row) is not None)
    evaluable = 0
    matched = 0
    changed = 0
    explicit_allowed = 0
    assumed_allowed = 0
    approvals = []
    costs = []
    latencies = []
    propensity_results = []
    all_evaluable = 0
    all_changed = 0
    quality_observed = 0
    orphan_quality_outcomes = sum(
        1
        for route_id in orphan_feedback_ids
        if _quality_outcome(feedback_index[route_id]) is not None
    )
    raw_quality_outcomes = orphan_quality_outcomes
    invalid_quality_outcomes = orphan_quality_outcomes
    quality_on_non_successful_routes = 0
    quality_on_unverified_execution_routes = 0
    raw_observation_propensity_rows = 0
    versioned_observation_propensity_rows = 0
    positive_support_routes = 0
    target_probabilities: list[float] = []
    global_weights: list[float] = []
    per_action_work: Dict[str, Dict[str, Any]] = {
        mode: {
            "target_routes": 0,
            "positive_support_routes": 0,
            "observed_matches": 0,
            "target_probabilities": [],
            "weights": [],
        }
        for mode in AUTO_AGENT_MODE_ORDER
    }

    for route_id in sorted(usage_index):
        usage_row = usage_index[route_id]
        score = _score(usage_row)
        observed = _observed_mode(usage_row)
        if score is None or observed is None:
            continue
        allowed, _was_explicit = _allowed_modes(usage_row)
        candidate = select_profile_action(score, allowed, selected_profile)
        all_evaluable += 1
        if candidate != observed:
            all_changed += 1

        policy = usage_row.get("auto_agent_policy")
        result = propensity_readiness(
            policy,
            chosen_action=observed,
            target_action=candidate,
            expected_policy_id=expected_policy_id,
            expected_policy_version=expected_policy_version,
            expected_feature_schema=expected_feature_schema,
            expected_context=_expected_context_for_route(
                route_id,
                expected_context=expected_context,
                expected_context_by_route_id=expected_context_by_route_id,
            ),
        )
        propensity_results.append(result)
        action_work = per_action_work[candidate]
        action_work["target_routes"] += 1
        target_probability = result.get("target_probability")
        if isinstance(target_probability, (int, float)) and float(target_probability) > 0.0:
            cooked_target_probability = float(target_probability)
            positive_support_routes += 1
            target_probabilities.append(cooked_target_probability)
            action_work["positive_support_routes"] += 1
            action_work["target_probabilities"].append(cooked_target_probability)

        if result.get("ready") and observed == candidate:
            chosen_probability = float(result.get("chosen_probability") or 0.0)
            if chosen_probability > 0.0:
                weight = 1.0 / chosen_probability
                global_weights.append(weight)
                action_work["weights"].append(weight)
                action_work["observed_matches"] += 1

        feedback_row = feedback_index.get(route_id)
        quality = _quality_outcome(feedback_row) if feedback_row is not None else None
        if quality is not None:
            raw_quality_outcomes += 1
            execution_state = execution_states[route_id]
            feedback_status = str(feedback_row.get("route_status") or "").strip().lower()
            feedback_status_consistent = bool(
                not feedback_status or feedback_status == str(execution_state["status"])
            )
            successful_route = bool(execution_state["successful_terminal"])
            verified_execution = bool(execution_state["execution_mode_verified"])
            quality_evidence_valid = bool(
                not durable_lifecycle_present
                or (successful_route and verified_execution and feedback_status_consistent)
            )
            if quality_evidence_valid:
                quality_observed += 1
            else:
                invalid_quality_outcomes += 1
                if not successful_route:
                    quality_on_non_successful_routes += 1
                if successful_route and not verified_execution:
                    quality_on_unverified_execution_routes += 1
        if feedback_row is not None:
            observation_probability = _finite_probability(feedback_row.get("observation_propensity"))
            if observation_probability is not None and observation_probability > 0.0:
                raw_observation_propensity_rows += 1
                observation_policy_id = str(feedback_row.get("observation_policy_id") or "").strip()
                observation_policy_version = str(
                    feedback_row.get("observation_policy_version") or ""
                ).strip()
                outcome_definition_version = str(
                    feedback_row.get("outcome_definition_version") or ""
                ).strip()
                if observation_policy_id and observation_policy_version and outcome_definition_version:
                    versioned_observation_propensity_rows += 1

    for route_id in joined_ids:
        usage_row = usage_index[route_id]
        feedback_row = feedback_index[route_id]
        score = _score(usage_row)
        observed = _observed_mode(usage_row)
        if score is None or observed is None:
            continue
        allowed, was_explicit = _allowed_modes(usage_row)
        if was_explicit:
            explicit_allowed += 1
        else:
            assumed_allowed += 1
        candidate = select_profile_action(score, allowed, selected_profile)
        evaluable += 1

        if candidate != observed:
            changed += 1
            continue
        matched += 1
        quality = _quality_outcome(feedback_row)
        execution_state = execution_states[route_id]
        feedback_status = str(feedback_row.get("route_status") or "").strip().lower()
        quality_evidence_valid = bool(
            not durable_lifecycle_present
            or (
                bool(execution_state["successful_terminal"])
                and bool(execution_state["execution_mode_verified"])
                and (not feedback_status or feedback_status == str(execution_state["status"]))
            )
        )
        if quality is not None and quality_evidence_valid:
            approvals.append(quality)
        economics = _actual_economics(usage_row, feedback_row)
        cost = _finite_metric(economics.get("cost_units"))
        latency = _finite_metric(economics.get("elapsed_ms"))
        if cost is not None:
            costs.append(cost)
        if latency is not None:
            latencies.append(latency)

    valid_propensity = sum(1 for result in propensity_results if result["ready"])
    approval_count = sum(approvals)
    quality_count = len(approvals)
    matched_observed = {
        "routes": matched,
        "quality_sample_count": quality_count,
        "approved": approval_count,
        "rejected": quality_count - approval_count,
        "approval_rate": _ratio(approval_count, quality_count),
        "approval_interval": _wilson_interval(approval_count, quality_count),
        "cost_sample_count": len(costs),
        "avg_cost_units": round(sum(costs) / len(costs), 6) if costs else None,
        "total_cost_units": round(sum(costs), 6) if costs else None,
        "max_cost_units": round(max(costs), 6) if costs else None,
        "latency_sample_count": len(latencies),
        "avg_elapsed_ms": round(sum(latencies) / len(latencies), 6) if latencies else None,
        "max_elapsed_ms": round(max(latencies), 6) if latencies else None,
        "interpretation": "observed outcomes where candidate and historical actions agree",
        "causal": False,
    }

    global_effective_sample_size = _effective_sample_size(global_weights)
    per_action: Dict[str, Dict[str, Any]] = {}
    for mode in AUTO_AGENT_MODE_ORDER:
        work = per_action_work[mode]
        if not int(work["target_routes"]):
            continue
        probabilities = list(work["target_probabilities"])
        per_action[mode] = {
            "target_routes": int(work["target_routes"]),
            "positive_support_routes": int(work["positive_support_routes"]),
            "observed_matches": int(work["observed_matches"]),
            "effective_sample_size": _effective_sample_size(work["weights"]),
            "minimum_target_probability": round(min(probabilities), 9) if probabilities else None,
        }
    weakest_target_action = None
    weakest_action_effective_sample_size = None
    if per_action:
        weakest_target_action, weakest_row = min(
            per_action.items(),
            key=lambda item: (float(item[1]["effective_sample_size"]), AUTO_AGENT_MODE_ORDER.index(item[0])),
        )
        weakest_action_effective_sample_size = float(weakest_row["effective_sample_size"])

    if durable_lifecycle_present:
        lifecycle_report = (
            lifecycle_source.get("lifecycle")
            if isinstance(lifecycle_source.get("lifecycle"), Mapping)
            else lifecycle_source
        )
        lifecycle_counts = (
            lifecycle_report.get("counts") if isinstance(lifecycle_report.get("counts"), Mapping) else {}
        )
        analysis_window = (
            lifecycle_source.get("analysis_window")
            if isinstance(lifecycle_source.get("analysis_window"), Mapping)
            else {}
        )
        durable_started = int(lifecycle_counts.get("started") or 0)
        durable_completed = int(lifecycle_counts.get("completed") or 0)
        durable_failed = int(lifecycle_counts.get("failed") or 0)
        durable_inflight = int(lifecycle_counts.get("inflight") or 0)
        row_lifecycle_counts_match = bool(
            row_status_counts["unknown"] == 0
            and row_status_counts["completed"] == durable_completed
            and row_status_counts["failed"] == durable_failed
            and row_status_counts["inflight"] == durable_inflight
        )
        lifecycle_reconciled = bool(
            durable_started == usage_counts["unique_route_ids"]
            and durable_completed + durable_failed + durable_inflight == durable_started
            and durable_inflight == 0
            and not bool(analysis_window.get("truncated"))
            and row_lifecycle_counts_match
        )
    else:
        durable_started = 0
        durable_completed = 0
        durable_failed = 0
        durable_inflight = 0
        row_lifecycle_counts_match = False
        lifecycle_reconciled = False

    quality_missing = max(0, all_evaluable - quality_observed)
    observation_propensity_logged = bool(
        all_evaluable > 0 and versioned_observation_propensity_rows == all_evaluable
    )
    outcome_evidence_integrity = bool(
        not durable_lifecycle_present
        or (
            invalid_quality_outcomes == 0
            and outcome_contract_integrity_complete
        )
    )
    quality_observation_ready = bool(
        all_evaluable > 0
        and quality_observed > 0
        and outcome_evidence_integrity
        and (quality_missing == 0 or observation_propensity_logged)
    )
    minimum_observed_target_probability = min(target_probabilities) if target_probabilities else None
    logging_integrity_complete = bool(
        all_evaluable > 0
        and valid_propensity == all_evaluable
        and decision_record_fingerprint_complete
    )
    target_probability_floor_met = bool(
        logging_integrity_complete
        and minimum_observed_target_probability is not None
        and minimum_observed_target_probability >= min_target_probability
    )
    global_overlap_met = global_effective_sample_size >= min_global_effective_sample_size
    per_action_overlap_met = bool(per_action) and all(
        float(row["effective_sample_size"]) >= min_per_action_effective_sample_size
        for row in per_action.values()
    )
    minimum_overlap_routes_met = valid_propensity >= min_overlap_routes
    checks = {
        "candidate_delta_present": all_changed > 0,
        "population_integrity_complete": population_integrity_complete,
        "execution_integrity_complete": execution_integrity_complete,
        "logging_integrity_complete": logging_integrity_complete,
        "minimum_overlap_routes_met": minimum_overlap_routes_met,
        "target_probability_floor_met": target_probability_floor_met,
        "global_overlap_ess_met": global_overlap_met,
        "per_action_overlap_met": per_action_overlap_met,
        "outcome_evidence_integrity": outcome_evidence_integrity,
        "quality_observation_ready": quality_observation_ready,
        "durable_lifecycle_present": durable_lifecycle_present,
        "lifecycle_reconciled": lifecycle_reconciled,
    }
    ready_for_external_ope = all(checks.values())
    blocking_reasons: list[str] = []
    if all_evaluable == 0:
        blocking_reasons.append("no_evaluable_routes")
    elif valid_propensity == 0:
        blocking_reasons.append("no_valid_randomized_overlap")
    elif not logging_integrity_complete:
        blocking_reasons.append("logging_integrity_incomplete")
    if durable_lifecycle_present and not decision_record_fingerprint_complete:
        blocking_reasons.append("decision_record_fingerprint_invalid")
    if not checks["candidate_delta_present"]:
        blocking_reasons.append("no_candidate_policy_delta")
    if durable_lifecycle_present and not population_integrity_complete:
        blocking_reasons.append("population_integrity_incomplete")
        if usage_counts["duplicate_route_ids"] or feedback_counts["duplicate_route_ids"]:
            blocking_reasons.append("duplicate_route_ids_in_durable_population")
        if usage_counts["rows_without_route_id"] or feedback_counts["rows_without_route_id"]:
            blocking_reasons.append("missing_route_ids_in_durable_population")
        if unevaluable_usage_ids:
            blocking_reasons.append("unevaluable_routes_in_durable_population")
        if orphan_feedback_ids:
            blocking_reasons.append("orphan_feedback_in_durable_population")
    if durable_lifecycle_present and not execution_integrity_complete:
        blocking_reasons.append("execution_integrity_incomplete")
        if row_status_counts["unknown"] or status_success_mismatch_routes:
            blocking_reasons.append("invalid_durable_route_state")
        if unverified_successful_execution_routes:
            blocking_reasons.append("unverified_successful_execution_mode")
        if chosen_executed_mode_mismatch_routes:
            blocking_reasons.append("chosen_executed_mode_mismatch")
        if feedback_route_status_mismatch_routes:
            blocking_reasons.append("feedback_route_status_mismatch")
    if not minimum_overlap_routes_met:
        blocking_reasons.append("insufficient_overlap_routes")
    if not target_probability_floor_met:
        blocking_reasons.append("insufficient_target_probability")
    if not global_overlap_met:
        blocking_reasons.append("insufficient_global_overlap_ess")
    if not per_action_overlap_met:
        blocking_reasons.append("insufficient_per_action_overlap")
    if durable_lifecycle_present and not outcome_evidence_integrity:
        blocking_reasons.append("outcome_evidence_integrity_failed")
        if not outcome_contract_integrity_complete:
            blocking_reasons.append("outcome_contract_not_precommitted")
        if invalid_quality_outcomes:
            blocking_reasons.append("quality_outcome_on_ineligible_route")
    if quality_observed == 0:
        blocking_reasons.append("no_observed_quality_outcomes")
    if not quality_observation_ready:
        blocking_reasons.append("unknown_quality_observation_process")
    if not durable_lifecycle_present:
        blocking_reasons.append("durable_lifecycle_required")
    elif not lifecycle_reconciled:
        blocking_reasons.append("lifecycle_not_reconciled")
    blocking_reasons = list(dict.fromkeys(blocking_reasons))

    evaluation_readiness = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "thresholds": {
            "minimum_valid_routes": min_overlap_routes,
            "minimum_target_probability": min_target_probability,
            "minimum_global_effective_sample_size": min_global_effective_sample_size,
            "minimum_per_action_effective_sample_size": min_per_action_effective_sample_size,
            "require_all_evaluable_logging_valid": True,
            "require_complete_quality_observation_or_model": True,
            "require_at_least_one_observed_quality_outcome": True,
            "require_durable_population_and_execution_integrity": True,
            "require_precommitted_outcome_contracts": True,
        },
        "population_integrity": {
            "durable_population": durable_lifecycle_present,
            "usage_rows": usage_counts["rows"],
            "usage_rows_with_route_id": usage_counts["rows_with_route_id"],
            "usage_rows_without_route_id": usage_counts["rows_without_route_id"],
            "unique_usage_route_ids": usage_counts["unique_route_ids"],
            "duplicate_usage_route_ids": usage_counts["duplicate_route_ids"],
            "feedback_rows": feedback_counts["rows"],
            "feedback_rows_without_route_id": feedback_counts["rows_without_route_id"],
            "duplicate_feedback_route_ids": feedback_counts["duplicate_route_ids"],
            "orphan_feedback_routes": len(orphan_feedback_ids),
            "evaluable_usage_routes": all_evaluable,
            "unevaluable_usage_routes": len(unevaluable_usage_ids),
            "missing_or_invalid_score_routes": len(missing_or_invalid_score_ids),
            "missing_or_invalid_chosen_mode_routes": len(missing_or_invalid_chosen_mode_ids),
            "complete": population_integrity_complete,
        },
        "execution_integrity": {
            "durable_population": durable_lifecycle_present,
            "row_status_counts": dict(row_status_counts),
            "row_lifecycle_counts_match": row_lifecycle_counts_match,
            "successful_terminal_routes": successful_terminal_routes,
            "failed_or_nonterminal_routes": row_status_counts["failed"] + row_status_counts["inflight"],
            "status_success_mismatch_routes": status_success_mismatch_routes,
            "missing_executed_mode_routes": missing_executed_mode_routes,
            "invalid_executed_mode_routes": invalid_executed_mode_routes,
            "unverified_successful_execution_routes": unverified_successful_execution_routes,
            "chosen_executed_mode_mismatch_routes": chosen_executed_mode_mismatch_routes,
            "feedback_route_status_mismatch_routes": feedback_route_status_mismatch_routes,
            "complete": execution_integrity_complete,
        },
        "logging_integrity": {
            "evaluable_usage_routes": all_evaluable,
            "checked_usage_routes": len(propensity_results),
            "valid_routes": valid_propensity,
            "invalid_routes": len(propensity_results) - valid_propensity,
            "valid_rate": _ratio(valid_propensity, len(propensity_results)),
            "all_evaluable_routes_valid": logging_integrity_complete,
            "invalid_reason_counts": _reason_counts(propensity_results),
            "decision_record_fingerprint_required": durable_lifecycle_present,
            "decision_record_fingerprint_valid_routes": fingerprint_valid_routes,
            "decision_record_fingerprint_invalid_routes": max(
                0, usage_counts["unique_route_ids"] - fingerprint_valid_routes
            ),
            "decision_record_fingerprint_invalid_reason_counts": fingerprint_invalid_reason_counts,
            "decision_record_fingerprint_complete": decision_record_fingerprint_complete,
        },
        "target_overlap": {
            "positive_support_routes": positive_support_routes,
            "unsupported_routes": max(0, all_evaluable - positive_support_routes),
            "minimum_target_probability": (
                round(minimum_observed_target_probability, 9)
                if minimum_observed_target_probability is not None
                else None
            ),
            "nonzero_weight_routes": len(global_weights),
            "effective_sample_size": global_effective_sample_size,
            "max_importance_weight": round(max(global_weights), 6) if global_weights else None,
            "per_action": per_action,
            "weakest_target_action": weakest_target_action,
            "weakest_action_effective_sample_size": weakest_action_effective_sample_size,
        },
        "outcome_observation": {
            "evaluable_routes": all_evaluable,
            "raw_quality_outcome_routes": raw_quality_outcomes,
            "quality_observed_routes": quality_observed,
            "quality_missing_routes": quality_missing,
            "quality_coverage_rate": _ratio(quality_observed, all_evaluable),
            "has_observed_quality_outcomes": quality_observed > 0,
            "invalid_quality_outcome_routes": invalid_quality_outcomes,
            "orphan_quality_outcome_routes": orphan_quality_outcomes,
            "quality_on_non_successful_routes": quality_on_non_successful_routes,
            "quality_on_unverified_execution_routes": quality_on_unverified_execution_routes,
            "outcome_contract_required": durable_lifecycle_present,
            "precommitted_outcome_contract_routes": precommitted_outcome_contract_routes,
            "missing_or_invalid_outcome_contract_routes": max(
                0,
                usage_counts["unique_route_ids"]
                - precommitted_outcome_contract_routes,
            ),
            "outcome_contract_integrity_complete": outcome_contract_integrity_complete,
            "evidence_integrity_complete": outcome_evidence_integrity,
            "observation_propensity_logged": observation_propensity_logged,
            "raw_observation_propensity_routes": raw_observation_propensity_rows,
            "versioned_observation_propensity_routes": versioned_observation_propensity_rows,
            "required_observation_contract_fields": [
                "observation_policy_id",
                "observation_policy_version",
                "outcome_definition_version",
            ],
            "missingness_semantics": "unknown",
            "ready": quality_observation_ready,
        },
        "lifecycle_integrity": {
            "durable_lifecycle_present": durable_lifecycle_present,
            "durable_started_routes": durable_started,
            "durable_terminal_routes": durable_completed + durable_failed,
            "durable_inflight_routes": durable_inflight,
            "replay_usage_routes": usage_counts["unique_route_ids"],
            "replay_decision_coverage_rate": _ratio(usage_counts["unique_route_ids"], durable_started),
            "reconciled": lifecycle_reconciled,
        },
        "ready_for_external_ope": ready_for_external_ope,
        "policy_value_estimated": False,
        "causal_claim": False,
    }

    propensity_summary = {
        "checked_joined_evaluable_routes": evaluable,
        "checked_evaluable_usage_routes": len(propensity_results),
        "valid_routes": valid_propensity,
        "invalid_routes": len(propensity_results) - valid_propensity,
        "valid_rate": _ratio(valid_propensity, len(propensity_results)),
        "required_policy_id": str(expected_policy_id or "") or None,
        "required_policy_version": str(expected_policy_version or "") or None,
        "required_feature_schema": str(expected_feature_schema or "") or None,
        "required_decision_context": (
            "route_specific"
            if expected_context_by_route_id is not None
            else (None if expected_context is _MISSING else expected_context)
        ),
        "invalid_reason_counts": _reason_counts(propensity_results),
        "minimum_routes_for_external_ope": min_overlap_routes,
        "ready_as_external_ope_input": ready_for_external_ope,
        "off_policy_estimate_computed": False,
        "causal_claim": False,
    }

    gate_status = "external_ope_required" if ready_for_external_ope else "blocked"
    gate_reason = (
        "validated_off_policy_estimator_and_review_required"
        if ready_for_external_ope
        else (blocking_reasons[0] if blocking_reasons else "readiness_checks_failed")
    )

    warnings = [
        "Historical matched replay is associational and cannot identify outcomes for actions not taken.",
        "The Wilson interval describes observed matched approvals; it is not a policy-value confidence interval.",
        "The readiness certificate diagnoses overlap and observation integrity; it does not estimate policy value.",
        "Missing quality feedback remains unknown unless an observation propensity or validated observation model is supplied.",
        "Durable quality evidence counts only successfully completed routes whose executed and chosen modes agree.",
    ]
    if assumed_allowed:
        warnings.append(
            "Some rows lacked an explicit allowed_agent_modes set; canonical eligibility was assumed for replay only."
        )

    return {
        "analysis_kind": "associational_matched_route_replay",
        "profile": selected_profile.as_dict(),
        "mode_order": list(AUTO_AGENT_MODE_ORDER),
        "causal_interpretation": {
            "causal": False,
            "off_policy_estimate": False,
            "counterfactual_actions_scored": False,
            "label": "historical associational evidence",
        },
        "support": {
            "usage": usage_counts,
            "feedback": feedback_counts,
            "exact_joined_route_ids": len(joined_ids),
            "exact_usage_join_coverage": _ratio(len(joined_ids), usage_counts["unique_route_ids"]),
            "exact_feedback_join_coverage": _ratio(len(joined_ids), feedback_counts["unique_route_ids"]),
            "usage_scored_routes": usage_scored,
            "joined_evaluable_routes": evaluable,
            "joined_explicit_allowed_modes": explicit_allowed,
            "joined_assumed_canonical_allowed_modes": assumed_allowed,
        },
        "candidate_action_agreement": {
            "evaluable_routes": evaluable,
            "matched_routes": matched,
            "changed_routes": changed,
            "agreement_rate": _ratio(matched, evaluable),
        },
        "matched_observed": matched_observed,
        "propensity_readiness": propensity_summary,
        "evaluation_readiness": evaluation_readiness,
        "promotion_gate": {
            "status": gate_status,
            "deployment": "shadow_only",
            "gate_policy_version": READINESS_SCHEMA_VERSION,
            "reason_code": gate_reason,
            "blocking_reason_codes": blocking_reasons,
            "checks": checks,
            "passed_checks": sum(1 for passed in checks.values() if passed),
            "total_checks": len(checks),
            "automatic_promotion_allowed": False,
            "requires_validated_external_ope": True,
            "causal_claim": False,
        },
        "warnings": warnings,
    }


build_policy_evidence_report = analyze_route_policy


__all__ = [
    "AUTO_AGENT_MODE_ORDER",
    "POLICY_PROFILES",
    "READINESS_MIN_GLOBAL_EFFECTIVE_SAMPLE_SIZE",
    "READINESS_MIN_PER_ACTION_EFFECTIVE_SAMPLE_SIZE",
    "READINESS_MIN_TARGET_PROBABILITY",
    "READINESS_SCHEMA_VERSION",
    "RoutePolicyProfile",
    "analyze_route_policy",
    "build_policy_evidence_report",
    "get_policy_profile",
    "ordered_allowed_modes",
    "profile_decision",
    "propensity_readiness",
    "select_profile_action",
]
