"""Pure planning primitives for a bounded-exposure adjacent-route rehearsal.

This module deliberately stops before runtime integration.  It can describe a
versioned, prompt-free exploration charter and deterministically assign an
action from caller-supplied nonce material, but it performs no I/O, writes no
ledger rows, starts no inference, estimates no policy value, and authorizes no
promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from decimal import Decimal, localcontext
from typing import Any, Dict, List, Mapping, Sequence, Tuple

try:
    from .route_policy_ledger import OUTCOME_CONTRACT_SCHEMA_VERSION, SUPPORT_SCHEMA_VERSION
except ImportError:  # pragma: no cover - direct source/ execution compatibility
    from route_policy_ledger import OUTCOME_CONTRACT_SCHEMA_VERSION, SUPPORT_SCHEMA_VERSION


STUDY_PLAN_SCHEMA_VERSION = "route-exploration-plan-v1"
STUDY_ASSIGNMENT_SCHEMA_VERSION = "route-exploration-rehearsal-assignment-v1"
REHEARSAL_RECEIPT_SCHEMA_VERSION = "route-exploration-rehearsal-receipt-v1"
REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION = "route-exploration-rehearsal-support-proposal-v1"
STUDY_ID = "auto-route-adjacent-explorer-v1"
STUDY_VERSION = "1.0.0"
STUDY_LABEL = "Bounded-Exposure Adjacent-Route Rehearsal v1"
ASSIGNMENT_ALGORITHM = "sha256-cdf-v1"

AUTO_AGENT_MODE_ORDER: Tuple[str, ...] = (
    "off",
    "collective",
    "loop",
    "collective_loop",
)
DEFAULT_EXPLORATION_RATE = 0.10
MAX_EXPLORATION_RATE = 0.20
MIN_POSITIVE_EXPLORATION_PROBABILITY = 0.05
MAX_ADJACENT_NEIGHBORS = 2
DEFAULT_PLANNED_ROUTES = 100
DEFAULT_SCENARIO_CONFIDENCE = 0.95
DEFAULT_ASSUMED_FEEDBACK_RATE = 0.30
DEFAULT_TARGET_OBSERVED_LABELS = 20
MAX_PLANNED_ROUTES = 100_000
MAX_TARGET_OBSERVED_LABELS = 1_000
MAX_EXACT_BINOMIAL_ROUTE_FORECAST = 10_000_000_000

_DESIGN_HASH_DOMAIN = b"supermix.route-explorer.design.v1\x00"
_NONCE_HASH_DOMAIN = b"supermix.route-explorer.nonce.v1\x00"
_DRAW_HASH_DOMAIN = b"supermix.route-explorer.draw.v1\x00"
_SOURCE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")

ACTIVATION_BLOCKERS: Tuple[str, ...] = (
    "target_policy_class_not_precommitted",
    "outcome_observation_process_not_validated",
    "population_definition_not_precommitted",
    "session_carryover_not_addressed",
    "interference_not_addressed",
    "stopping_rules_not_precommitted",
    "preassignment_seed_commitment_not_sealed",
    "external_ope_not_validated",
)

_SOURCE_CONTRACT_KEYS = {
    "policy_id",
    "policy_version",
    "feature_schema_version",
    "support_schema_version",
    "candidate_set_hash",
    "distribution_hash",
    "outcome_contract_schema_version",
}

_CANDIDATE_KEYS = {
    "action",
    "estimated_cost_units",
    "estimated_model_calls",
    "planned_loop_steps",
    "latency_tier",
    "selected",
}
_REQUIRED_CANDIDATE_KEYS = {
    "action",
    "estimated_cost_units",
    "estimated_model_calls",
    "planned_loop_steps",
    "latency_tier",
}
_EXCLUSION_KEYS = {"action", "reasons"}
_ALLOWED_EXCLUSION_REASONS = {
    "action_mode_unsupported",
    "capability_or_policy_filter",
    "session_budget_post_filter",
}
_LATENCY_TIERS = {"low", "moderate", "high", "frontier", "unknown"}
_PLAN_KEYS = {"schema_version", "study", "charter", "design_hash"}
_STUDY_KEYS = {"study_id", "study_version", "label", "design_kind"}
_CHARTER_KEYS = {
    "source_contract",
    "prompt_free_contract",
    "enrollment",
    "probability_design",
    "candidates",
    "exclusions",
    "traffic_scenario",
    "resource_forecast",
    "causal_boundaries",
}
_ENROLLMENT_KEYS = {
    "eligible",
    "reason",
    "baseline_action",
    "adjacent_feasible_actions",
    "maximum_adjacent_neighbors",
}
_PROBABILITY_DESIGN_KEYS = {
    "decision_type",
    "probability_stage",
    "requested_exploration_rate",
    "applied_exploration_rate",
    "minimum_positive_exploration_probability",
    "eligible_actions",
    "action_probabilities",
    "assignment_algorithm",
    "assignment_unit",
    "assignment_performed",
}
_STUDY_CANDIDATE_KEYS = {
    "action",
    "baseline",
    "adjacent_distance",
    "estimated_cost_units",
    "estimated_model_calls",
    "planned_loop_steps",
    "latency_tier",
}
_STUDY_EXCLUSION_KEYS = {
    "action",
    "reasons",
    "post_filter_feasible",
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
        raise ValueError("study charter must be canonical JSON without non-finite values") from exc


def _domain_hash(domain: bytes, value: Any) -> str:
    return hashlib.sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


def _strict_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {', '.join(sorted(map(str, unknown)))}")


def _source_identifier(value: Any, name: str) -> str:
    cooked = str(value or "").strip()
    if not _SOURCE_IDENTIFIER_RE.fullmatch(cooked):
        raise ValueError(f"source_contract {name} must be a versioned identifier")
    return cooked


def _sha256_hex(value: Any, name: str) -> str:
    cooked = str(value or "").strip()
    if len(cooked) != 64 or any(character not in "0123456789abcdef" for character in cooked):
        raise ValueError(f"source_contract {name} must be a lowercase SHA-256 hex digest")
    return cooked


def _normalize_source_contract(value: Any) -> Dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("source_contract must be a JSON object")
    _strict_keys(value, _SOURCE_CONTRACT_KEYS, "source_contract")
    if set(value) != _SOURCE_CONTRACT_KEYS:
        raise ValueError("source_contract must contain every required provenance field")
    support_schema = _source_identifier(
        value.get("support_schema_version"), "support_schema_version"
    )
    outcome_schema = _source_identifier(
        value.get("outcome_contract_schema_version"),
        "outcome_contract_schema_version",
    )
    if support_schema != SUPPORT_SCHEMA_VERSION:
        raise ValueError(
            f"source_contract support_schema_version must be {SUPPORT_SCHEMA_VERSION}"
        )
    if outcome_schema != OUTCOME_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "source_contract outcome_contract_schema_version must be "
            f"{OUTCOME_CONTRACT_SCHEMA_VERSION}"
        )
    return {
        "policy_id": _source_identifier(value.get("policy_id"), "policy_id"),
        "policy_version": _source_identifier(
            value.get("policy_version"), "policy_version"
        ),
        "feature_schema_version": _source_identifier(
            value.get("feature_schema_version"), "feature_schema_version"
        ),
        "support_schema_version": support_schema,
        "candidate_set_hash": _sha256_hex(
            value.get("candidate_set_hash"), "candidate_set_hash"
        ),
        "distribution_hash": _sha256_hex(
            value.get("distribution_hash"), "distribution_hash"
        ),
        "outcome_contract_schema_version": outcome_schema,
    }


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite non-negative number")
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(cooked) or cooked < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return round(cooked, 6)


def _nonnegative_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _validate_rate(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("exploration_rate must be a finite number in (0, 0.20]")
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("exploration_rate must be a finite number in (0, 0.20]") from exc
    if not math.isfinite(cooked) or not 0.0 < cooked <= MAX_EXPLORATION_RATE:
        raise ValueError("exploration_rate must be a finite number in (0, 0.20]")
    return round(cooked, 12)


def _validate_planned_routes(value: Any) -> int:
    routes = _nonnegative_int(value, "planned_routes", minimum=1)
    if routes > MAX_PLANNED_ROUTES:
        raise ValueError(f"planned_routes must not exceed {MAX_PLANNED_ROUTES}")
    return routes


def _validate_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("scenario_confidence must be a finite number in (0.5, 1)")
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scenario_confidence must be a finite number in (0.5, 1)") from exc
    if not math.isfinite(cooked) or not 0.5 < cooked < 1.0:
        raise ValueError("scenario_confidence must be a finite number in (0.5, 1)")
    return round(cooked, 12)


def _validate_feedback_rate(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("assumed_feedback_rate must be a finite number in (0, 1]")
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("assumed_feedback_rate must be a finite number in (0, 1]") from exc
    if not math.isfinite(cooked) or not 0.0 < cooked <= 1.0:
        raise ValueError("assumed_feedback_rate must be a finite number in (0, 1]")
    return round(cooked, 12)


def _validate_target_labels(value: Any) -> int:
    target = _nonnegative_int(value, "target_observed_labels", minimum=1)
    if target > MAX_TARGET_OBSERVED_LABELS:
        raise ValueError(
            f"target_observed_labels must not exceed {MAX_TARGET_OBSERVED_LABELS}"
        )
    return target


def _normalize_candidates(value: Any, baseline_mode: str) -> List[Dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("post_filter_candidates must be a non-empty sequence")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("post_filter_candidates must contain JSON objects")
        _strict_keys(raw, _CANDIDATE_KEYS, "post-filter candidate")
        missing = _REQUIRED_CANDIDATE_KEYS - set(raw)
        if missing:
            raise ValueError(
                "post-filter candidate is missing required fields: "
                + ", ".join(sorted(missing))
            )
        action = str(raw.get("action") or "").strip()
        if action not in AUTO_AGENT_MODE_ORDER:
            raise ValueError(f"unsupported route action: {action or '<empty>'}")
        if action in seen:
            raise ValueError(f"duplicate post-filter candidate action: {action}")
        selected = raw.get("selected", action == baseline_mode)
        if not isinstance(selected, bool) or selected is not (action == baseline_mode):
            raise ValueError("candidate selected flags must identify exactly the baseline action")
        latency_tier = str(raw.get("latency_tier") or "").strip().lower()
        if latency_tier not in _LATENCY_TIERS:
            raise ValueError(f"unsupported latency_tier: {latency_tier or '<empty>'}")
        normalized.append(
            {
                "action": action,
                "estimated_cost_units": _finite_nonnegative(
                    raw.get("estimated_cost_units"), "estimated_cost_units"
                ),
                "estimated_model_calls": _nonnegative_int(
                    raw.get("estimated_model_calls"), "estimated_model_calls", minimum=1
                ),
                "planned_loop_steps": _nonnegative_int(
                    raw.get("planned_loop_steps"), "planned_loop_steps"
                ),
                "latency_tier": latency_tier,
                "selected": selected,
            }
        )
        seen.add(action)
    if not normalized:
        raise ValueError("post_filter_candidates must be a non-empty sequence")
    if baseline_mode not in seen:
        raise ValueError("baseline_mode must be present in post_filter_candidates")
    return sorted(normalized, key=lambda row: AUTO_AGENT_MODE_ORDER.index(row["action"]))


def _normalize_exclusions(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        value = []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("post_filter_exclusions must be a sequence")
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("post_filter_exclusions must contain JSON objects")
        _strict_keys(raw, _EXCLUSION_KEYS, "post-filter exclusion")
        action = str(raw.get("action") or "").strip()
        if action not in AUTO_AGENT_MODE_ORDER:
            raise ValueError(f"unsupported excluded route action: {action or '<empty>'}")
        if action in seen:
            raise ValueError(f"duplicate post-filter exclusion action: {action}")
        reasons_raw = raw.get("reasons")
        if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
            raise ValueError("post-filter exclusion reasons must be a non-empty sequence")
        reasons = [str(reason or "").strip() for reason in reasons_raw]
        if not reasons or any(reason not in _ALLOWED_EXCLUSION_REASONS for reason in reasons):
            raise ValueError("post-filter exclusions must use only versioned reason codes")
        if len(set(reasons)) != len(reasons):
            raise ValueError("post-filter exclusion reasons must be unique")
        normalized.append({"action": action, "reasons": reasons})
        seen.add(action)
    return sorted(normalized, key=lambda row: AUTO_AGENT_MODE_ORDER.index(row["action"]))


def _binomial_upper_quantile(trials: int, probability: float, confidence: float) -> int:
    """Return the exact smallest k with Binomial(n, p) CDF(k) >= confidence."""

    if probability <= 0.0:
        return 0
    if probability >= 1.0:
        return trials
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    log_n_factorial = math.lgamma(trials + 1)
    log_probabilities = [
        log_n_factorial
        - math.lgamma(k + 1)
        - math.lgamma(trials - k + 1)
        + (k * log_p)
        + ((trials - k) * log_q)
        for k in range(trials + 1)
    ]
    pivot = max(log_probabilities)
    scaled = [math.exp(item - pivot) for item in log_probabilities]
    total = math.fsum(scaled)
    threshold = confidence * total
    cumulative = 0.0
    for k, mass in enumerate(scaled):
        cumulative += mass
        if cumulative >= threshold:
            return k
    return trials


def _binomial_below_probability(
    trials: int,
    probability: float,
    target_successes: int,
) -> float:
    """Return exact P[Binomial(trials, probability) < target_successes]."""

    if target_successes <= 0:
        return 0.0
    if trials < target_successes or probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    log_q = math.log1p(-probability)
    log_ratio_constant = math.log(probability) - log_q
    log_probability = trials * log_q
    lower_tail_logs = []
    for successes in range(target_successes):
        lower_tail_logs.append(log_probability)
        if successes + 1 < target_successes:
            log_probability += (
                math.log(trials - successes)
                - math.log(successes + 1)
                + log_ratio_constant
            )
    pivot = max(lower_tail_logs)
    lower_tail = math.exp(pivot) * math.fsum(
        math.exp(item - pivot) for item in lower_tail_logs
    )
    return max(0.0, min(1.0, lower_tail))


def _binomial_at_least_probability(
    trials: int,
    probability: float,
    target_successes: int,
) -> float:
    """Return exact P[Binomial(trials, probability) >= target_successes]."""

    return 1.0 - _binomial_below_probability(
        trials, probability, target_successes
    )


def _binomial_log_pmf(trials: int, probability: float, successes: int) -> float:
    if successes < 0 or successes > trials or not 0.0 < probability < 1.0:
        return -math.inf
    log_q = math.log1p(-probability)
    return (
        trials * log_q
        + successes * (math.log(probability) - log_q)
        + math.fsum(
            math.log(trials - index) - math.log(index + 1)
            for index in range(successes)
        )
    )


def _logaddexp(left: float, right: float) -> float:
    if left == -math.inf:
        return right
    if right == -math.inf:
        return left
    pivot = max(left, right)
    return pivot + math.log1p(math.exp(-abs(left - right)))


def _multinomial_at_least_each_probability(
    trials: int,
    probabilities: Sequence[float],
    target_successes: int,
) -> float:
    """Exact P[X1 >= t and X2 >= t] for a three-cell multinomial."""

    if len(probabilities) != 2:
        raise ValueError("joint multinomial rehearsal requires exactly two alternates")
    first, second = (float(probabilities[0]), float(probabilities[1]))
    if trials < (2 * target_successes) or first <= 0.0 or second <= 0.0:
        return 0.0
    remainder = 1.0 - first - second
    if remainder <= 0.0:
        raise ValueError("joint multinomial label probabilities must sum to less than 1")
    first_lower = _binomial_below_probability(trials, first, target_successes)
    second_lower = _binomial_below_probability(trials, second, target_successes)

    # Condition on X1=i.  Given X1=i, X2 is Binomial(n-i, p2/(1-p1)).
    # Summing i<t is exact.  The conditional CDF and its boundary PMF can be
    # decremented from n-i to n-i-1 in O(1), reducing the rectangle from O(t^2)
    # log-gamma evaluations to O(t) stable recurrence steps.
    conditional_probability = second / (1.0 - first)
    boundary_successes = target_successes - 1
    conditional_cdf = _binomial_below_probability(
        trials, conditional_probability, target_successes
    )
    conditional_boundary_pmf_log = _binomial_log_pmf(
        trials, conditional_probability, boundary_successes
    )
    conditional_boundary_pmf = (
        math.exp(conditional_boundary_pmf_log)
        if conditional_boundary_pmf_log > -math.inf
        else 0.0
    )
    marginal_log_pmf = trials * math.log1p(-first)
    marginal_log_ratio_constant = math.log(first) - math.log1p(-first)
    rectangle_log_probability = -math.inf
    conditional_trials = trials
    for first_count in range(target_successes):
        if conditional_cdf > 0.0:
            rectangle_log_probability = _logaddexp(
                rectangle_log_probability,
                marginal_log_pmf + math.log(conditional_cdf),
            )
        if first_count + 1 >= target_successes:
            break
        next_boundary_pmf = conditional_boundary_pmf * (
            (conditional_trials - boundary_successes)
            / (conditional_trials * (1.0 - conditional_probability))
        )
        conditional_cdf = min(
            1.0,
            conditional_cdf + (conditional_probability * next_boundary_pmf),
        )
        conditional_boundary_pmf = next_boundary_pmf
        conditional_trials -= 1
        marginal_log_pmf += (
            math.log(trials - first_count)
            - math.log(first_count + 1)
            + marginal_log_ratio_constant
        )
    both_lower = (
        math.exp(rectangle_log_probability)
        if rectangle_log_probability > -math.inf
        else 0.0
    )
    joint = 1.0 - first_lower - second_lower + both_lower
    return max(0.0, min(1.0, joint))


def _simultaneous_label_probability(
    trials: int,
    label_probabilities: Sequence[float],
    target_successes: int,
) -> float:
    if len(label_probabilities) == 1:
        return _binomial_at_least_probability(
            trials, float(label_probabilities[0]), target_successes
        )
    if len(label_probabilities) == 2:
        return _multinomial_at_least_each_probability(
            trials, label_probabilities, target_successes
        )
    raise ValueError("study supports simultaneous targets for one or two alternates")


def _minimum_simultaneous_label_routes(
    *,
    label_probabilities: Sequence[float],
    target_labels: int,
    confidence: float,
) -> int:
    """Invert the exact binomial or joint multinomial tail."""

    if not label_probabilities or any(probability <= 0.0 for probability in label_probabilities):
        raise ValueError("alternate label probabilities must be positive for exact inversion")
    lower = target_labels * len(label_probabilities)
    weakest = min(float(probability) for probability in label_probabilities)
    upper = max(lower, int(math.ceil(target_labels / weakest)))
    while _simultaneous_label_probability(
        upper, label_probabilities, target_labels
    ) < confidence:
        if upper >= MAX_EXACT_BINOMIAL_ROUTE_FORECAST:
            raise ValueError("exact simultaneous target route forecast exceeds the supported limit")
        upper = min(MAX_EXACT_BINOMIAL_ROUTE_FORECAST, max(upper + 1, upper * 2))
    while lower < upper:
        midpoint = (lower + upper) // 2
        if _simultaneous_label_probability(
            midpoint, label_probabilities, target_labels
        ) >= confidence:
            upper = midpoint
        else:
            lower = midpoint + 1
    return lower


def _traffic_scenario(
    probabilities: Mapping[str, float],
    *,
    planned_routes: int,
    confidence: float,
    exploration_rate: float,
    baseline_mode: str,
    assumed_feedback_rate: float,
    target_observed_labels: int,
) -> Dict[str, Any]:
    actions = list(probabilities)
    marginal_confidence = 1.0 - ((1.0 - confidence) / max(1, len(actions)))
    expected_by_action = {
        action: round(planned_routes * float(probability), 6)
        for action, probability in probabilities.items()
    }
    upper_by_action = {
        action: _binomial_upper_quantile(
            planned_routes, float(probability), marginal_confidence
        )
        for action, probability in probabilities.items()
    }
    alternate_propensities = {
        action: float(probability)
        for action, probability in probabilities.items()
        if action != baseline_mode
    }
    weakest_alternate_propensity = (
        min(alternate_propensities.values()) if alternate_propensities else None
    )
    label_probabilities_by_action = {
        action: round(propensity * assumed_feedback_rate, 12)
        for action, propensity in alternate_propensities.items()
    }
    label_probabilities = list(label_probabilities_by_action.values())
    expected_labels_by_action = {
        action: round(planned_routes * probability, 6)
        for action, probability in label_probabilities_by_action.items()
    }
    expected_routes_by_action = {
        action: round(target_observed_labels / probability, 6)
        for action, probability in label_probabilities_by_action.items()
    }
    minimum_routes = (
        _minimum_simultaneous_label_routes(
            label_probabilities=label_probabilities,
            target_labels=target_observed_labels,
            confidence=confidence,
        )
        if label_probabilities
        else None
    )
    simultaneous_method = (
        "exact_binomial_tail_inversion_single_alternate"
        if len(label_probabilities) == 1
        else (
            "exact_joint_multinomial_tail_inversion_two_alternates"
            if len(label_probabilities) == 2
            else "not_applicable_no_alternate_actions"
        )
    )
    probability_at_minimum = (
        _simultaneous_label_probability(
            minimum_routes, label_probabilities, target_observed_labels
        )
        if minimum_routes is not None
        else None
    )
    probability_at_previous = (
        _simultaneous_label_probability(
            minimum_routes - 1, label_probabilities, target_observed_labels
        )
        if minimum_routes is not None and minimum_routes > 0
        else None
    )
    label_scenario = {
        "analysis_type": "simultaneous_alternate_label_traffic_not_power_or_mnar_correction",
        "target_scope": "at_least_target_observed_labels_on_every_alternate_action",
        "alternate_actions": list(alternate_propensities),
        "alternate_action_propensities": alternate_propensities,
        "weakest_alternate_propensity": weakest_alternate_propensity,
        "assumed_feedback_rate": assumed_feedback_rate,
        "per_route_observed_label_probability_by_alternate_action": (
            label_probabilities_by_action
        ),
        "target_observed_labels_per_alternate_action": target_observed_labels,
        "expected_observed_labels_at_planned_routes_by_alternate_action": (
            expected_labels_by_action
        ),
        "expected_routes_for_target_by_alternate_action": expected_routes_by_action,
        "exact_simultaneous_target": {
            "method": simultaneous_method,
            "confidence_level": confidence,
            "minimum_routes_for_target_on_every_alternate_action": minimum_routes,
            "probability_at_minimum_routes": (
                round(probability_at_minimum, 12)
                if probability_at_minimum is not None
                else None
            ),
            "probability_at_previous_route_count": (
                round(probability_at_previous, 12)
                if probability_at_previous is not None
                else None
            ),
        },
        "feedback_rate_source": "caller_assumption_for_capacity_planning",
        "missingness_identification_performed": False,
        "mnar_correction_performed": False,
        "not_power_analysis": True,
        "interpretation": (
            "simultaneous per-alternate traffic planning under an assumed feedback rate; "
            "not statistical power, causal identification, or correction for selective feedback"
        ),
    }
    return {
        "analysis_type": "traffic_capacity_scenario_not_statistical_power",
        "planned_routes": planned_routes,
        "assignment_model": "fixed_n_independent_route_assignments",
        "expected": {
            "routes_by_action": expected_by_action,
            "exploration_routes": round(planned_routes * exploration_rate, 6),
        },
        "high_probability": {
            "confidence_level": confidence,
            "method": "exact_binomial_marginal_upper_quantiles_with_bonferroni_union_bound",
            "marginal_confidence_level": round(marginal_confidence, 12),
            "simultaneous_confidence_at_least": confidence,
            "upper_routes_by_action": upper_by_action,
            "upper_exploration_routes": _binomial_upper_quantile(
                planned_routes, exploration_rate, confidence
            ),
            "interpretation": "capacity scenario only; not an effect-detection guarantee",
        },
        "not_power_analysis": True,
        "power_analysis_performed": False,
        "minimum_detectable_effect": None,
        "sample_size_recommendation": None,
        "policy_value_estimate": None,
        "observed_label_scenario": label_scenario,
    }


def _weighted_metric(
    candidates: Mapping[str, Mapping[str, Any]],
    probabilities: Mapping[str, float],
    metric: str,
) -> float:
    return round(
        math.fsum(
            float(probabilities[action]) * float(candidates[action][metric])
            for action in probabilities
        ),
        6,
    )


def _resource_forecast(
    candidates: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, float],
    baseline_mode: str,
    traffic: Mapping[str, Any],
) -> Dict[str, Any]:
    candidate_map = {str(row["action"]): row for row in candidates}
    planned_routes = int(traffic["planned_routes"])
    expected_per_route = {
        "cost_units": _weighted_metric(candidate_map, probabilities, "estimated_cost_units"),
        "model_calls": _weighted_metric(candidate_map, probabilities, "estimated_model_calls"),
        "loop_steps": _weighted_metric(candidate_map, probabilities, "planned_loop_steps"),
    }
    baseline = candidate_map[baseline_mode]
    baseline_per_route = {
        "cost_units": float(baseline["estimated_cost_units"]),
        "model_calls": int(baseline["estimated_model_calls"]),
        "loop_steps": int(baseline["planned_loop_steps"]),
    }
    expected_totals = {
        key: round(float(value) * planned_routes, 6)
        for key, value in expected_per_route.items()
    }
    baseline_totals = {
        key: round(float(value) * planned_routes, 6)
        for key, value in baseline_per_route.items()
    }
    upper_counts = traffic["high_probability"]["upper_routes_by_action"]
    conservative_upper = {
        "cost_units": round(
            math.fsum(
                int(upper_counts[action]) * float(candidate_map[action]["estimated_cost_units"])
                for action in probabilities
            ),
            6,
        ),
        "model_calls": round(
            math.fsum(
                int(upper_counts[action]) * int(candidate_map[action]["estimated_model_calls"])
                for action in probabilities
            ),
            6,
        ),
        "loop_steps": round(
            math.fsum(
                int(upper_counts[action]) * int(candidate_map[action]["planned_loop_steps"])
                for action in probabilities
            ),
            6,
        ),
    }
    return {
        "basis": "caller_supplied_post_filter_route_economics",
        "by_action": {
            action: {
                "probability": float(probabilities[action]),
                "estimated_cost_units": float(candidate_map[action]["estimated_cost_units"]),
                "estimated_model_calls": int(candidate_map[action]["estimated_model_calls"]),
                "planned_loop_steps": int(candidate_map[action]["planned_loop_steps"]),
                "latency_tier": str(candidate_map[action]["latency_tier"]),
            }
            for action in probabilities
        },
        "baseline_per_route": baseline_per_route,
        "expected_per_route": expected_per_route,
        "expected_for_planned_routes": expected_totals,
        "baseline_for_planned_routes": baseline_totals,
        "expected_increment_vs_baseline": {
            key: round(expected_totals[key] - baseline_totals[key], 6)
            for key in expected_totals
        },
        "high_probability_conservative_capacity_upper": {
            "confidence_level": traffic["high_probability"]["confidence_level"],
            "count_bound_method": traffic["high_probability"]["method"],
            **conservative_upper,
        },
        "forecast_only": True,
        "budget_guarantee": False,
    }


def _causal_boundaries() -> Dict[str, Any]:
    return {
        "deployment": "shadow_only",
        "execution_enabled": False,
        "execution_performed": False,
        "io_performed": False,
        "ledger_write_performed": False,
        "model_inference_performed": False,
        "off_policy_estimate_computed": False,
        "causal_identification_performed": False,
        "missingness_identification_performed": False,
        "mnar_correction_performed": False,
        "assumed_feedback_rate_is_not_an_observation_model": True,
        "power_analysis_performed": False,
        "automatic_promotion_allowed": False,
        "baseline_performance_guarantee": False,
        "logging_policy_optimality_claim": False,
        "ledger_eligible": False,
        "preassignment_commitment_sealed": False,
        "nonce_grinding_resistant": False,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
        "requires_separate_runtime_integration": True,
        "requires_validated_external_ope": True,
        "interpretation": "design, audit, and capacity-planning artifact only",
    }


def plan_adjacent_route_study(
    baseline_mode: str,
    post_filter_candidates: Sequence[Mapping[str, Any]],
    post_filter_exclusions: Sequence[Mapping[str, Any]],
    *,
    source_contract: Mapping[str, Any],
    exploration_rate: float = DEFAULT_EXPLORATION_RATE,
    planned_routes: int = DEFAULT_PLANNED_ROUTES,
    scenario_confidence: float = DEFAULT_SCENARIO_CONFIDENCE,
    assumed_feedback_rate: float = DEFAULT_ASSUMED_FEEDBACK_RATE,
    target_observed_labels: int = DEFAULT_TARGET_OBSERVED_LABELS,
) -> Dict[str, Any]:
    """Build a canonical, prompt-free, non-executing adjacent-route charter.

    The candidate/exclusion inputs must describe the complete final action
    partition after capability, policy, and budget filters.  Only immediate
    feasible neighbors of the deterministic baseline receive exploration mass.
    """

    cooked_baseline = str(baseline_mode or "").strip()
    if cooked_baseline not in AUTO_AGENT_MODE_ORDER:
        raise ValueError("baseline_mode must be a canonical route action")
    cooked_rate = _validate_rate(exploration_rate)
    cooked_routes = _validate_planned_routes(planned_routes)
    cooked_confidence = _validate_confidence(scenario_confidence)
    cooked_feedback_rate = _validate_feedback_rate(assumed_feedback_rate)
    cooked_target_labels = _validate_target_labels(target_observed_labels)
    cooked_source_contract = _normalize_source_contract(source_contract)
    candidates = _normalize_candidates(post_filter_candidates, cooked_baseline)
    exclusions = _normalize_exclusions(post_filter_exclusions)

    candidate_actions = {row["action"] for row in candidates}
    exclusion_actions = {row["action"] for row in exclusions}
    if candidate_actions & exclusion_actions:
        raise ValueError("post-filter candidates and exclusions must be disjoint")
    if candidate_actions | exclusion_actions != set(AUTO_AGENT_MODE_ORDER):
        raise ValueError("post-filter candidates and exclusions must partition all route actions")

    baseline_index = AUTO_AGENT_MODE_ORDER.index(cooked_baseline)
    neighbor_actions = [
        row["action"]
        for row in candidates
        if abs(AUTO_AGENT_MODE_ORDER.index(row["action"]) - baseline_index) == 1
    ]
    if len(neighbor_actions) > MAX_ADJACENT_NEIGHBORS:  # defensive against mode-order changes
        raise ValueError("study supports at most two adjacent feasible neighbors")
    enrolled = bool(neighbor_actions)
    if enrolled:
        per_neighbor = round(cooked_rate / len(neighbor_actions), 12)
        if per_neighbor + 1e-12 < MIN_POSITIVE_EXPLORATION_PROBABILITY:
            raise ValueError(
                "exploration_rate split across feasible neighbors must keep each at or above 0.05"
            )
        study_actions = [
            mode
            for mode in AUTO_AGENT_MODE_ORDER
            if mode == cooked_baseline or mode in neighbor_actions
        ]
        probabilities = {
            action: (
                round(1.0 - cooked_rate, 12)
                if action == cooked_baseline
                else per_neighbor
            )
            for action in study_actions
        }
        applied_rate = cooked_rate
        enrollment_reason = "eligible_adjacent_post_filter_support"
    else:
        study_actions = [cooked_baseline]
        probabilities = {cooked_baseline: 1.0}
        applied_rate = 0.0
        enrollment_reason = "no_feasible_adjacent_action"

    candidate_by_action = {row["action"]: row for row in candidates}
    study_candidates = [
        {
            "action": action,
            "baseline": action == cooked_baseline,
            "adjacent_distance": abs(AUTO_AGENT_MODE_ORDER.index(action) - baseline_index),
            "estimated_cost_units": candidate_by_action[action]["estimated_cost_units"],
            "estimated_model_calls": candidate_by_action[action]["estimated_model_calls"],
            "planned_loop_steps": candidate_by_action[action]["planned_loop_steps"],
            "latency_tier": candidate_by_action[action]["latency_tier"],
        }
        for action in study_actions
    ]
    study_exclusions = [
        {
            "action": row["action"],
            "reasons": list(row["reasons"]),
            "post_filter_feasible": False,
        }
        for row in exclusions
    ]
    for row in candidates:
        if row["action"] not in study_actions:
            study_exclusions.append(
                {
                    "action": row["action"],
                    "reasons": ["nonadjacent_exploration_guard"],
                    "post_filter_feasible": True,
                }
            )
    study_exclusions.sort(key=lambda row: AUTO_AGENT_MODE_ORDER.index(row["action"]))

    traffic = _traffic_scenario(
        probabilities,
        planned_routes=cooked_routes,
        confidence=cooked_confidence,
        exploration_rate=applied_rate,
        baseline_mode=cooked_baseline,
        assumed_feedback_rate=cooked_feedback_rate,
        target_observed_labels=cooked_target_labels,
    )
    resource_forecast = _resource_forecast(
        study_candidates,
        probabilities,
        cooked_baseline,
        traffic,
    )
    charter = {
        "source_contract": cooked_source_contract,
        "prompt_free_contract": {
            "prompt_free": True,
            "raw_prompt_included": False,
            "raw_session_id_included": False,
            "free_form_text_fields_allowed": False,
            "canonical_json": "sorted_keys_compact_utf8_no_nan",
        },
        "enrollment": {
            "eligible": enrolled,
            "reason": enrollment_reason,
            "baseline_action": cooked_baseline,
            "adjacent_feasible_actions": list(neighbor_actions),
            "maximum_adjacent_neighbors": MAX_ADJACENT_NEIGHBORS,
        },
        "probability_design": {
            "decision_type": "randomized" if enrolled else "deterministic_not_enrolled",
            "probability_stage": "post_filter",
            "requested_exploration_rate": cooked_rate,
            "applied_exploration_rate": applied_rate,
            "minimum_positive_exploration_probability": (
                min(probabilities[action] for action in neighbor_actions)
                if neighbor_actions
                else None
            ),
            "eligible_actions": list(study_actions),
            "action_probabilities": dict(probabilities),
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "assignment_unit": "route",
            "assignment_performed": False,
        },
        "candidates": study_candidates,
        "exclusions": study_exclusions,
        "traffic_scenario": traffic,
        "resource_forecast": resource_forecast,
        "causal_boundaries": _causal_boundaries(),
    }
    payload = {
        "schema_version": STUDY_PLAN_SCHEMA_VERSION,
        "study": {
            "study_id": STUDY_ID,
            "study_version": STUDY_VERSION,
            "label": STUDY_LABEL,
            "design_kind": "bounded_exposure_adjacent_rehearsal",
        },
        "charter": charter,
    }
    return {**payload, "design_hash": _domain_hash(_DESIGN_HASH_DOMAIN, payload)}


def _validated_assignment_plan(plan: Any) -> Dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be a canonical adjacent-route study object")
    _strict_keys(plan, _PLAN_KEYS, "plan")
    if set(plan) != _PLAN_KEYS:
        raise ValueError("plan is missing required top-level fields")
    if plan.get("schema_version") != STUDY_PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported study plan schema_version")
    study = plan.get("study")
    charter = plan.get("charter")
    if not isinstance(study, Mapping) or not isinstance(charter, Mapping):
        raise ValueError("plan study and charter must be JSON objects")
    _strict_keys(study, _STUDY_KEYS, "plan study")
    _strict_keys(charter, _CHARTER_KEYS, "plan charter")
    if set(study) != _STUDY_KEYS or set(charter) != _CHARTER_KEYS:
        raise ValueError("plan study or charter is missing required fields")
    if study.get("study_id") != STUDY_ID or study.get("study_version") != STUDY_VERSION:
        raise ValueError("plan does not identify Bounded-Exposure Adjacent-Route Rehearsal v1")
    supplied_hash = str(plan.get("design_hash") or "").strip()
    payload = {
        "schema_version": plan["schema_version"],
        "study": dict(study),
        "charter": dict(charter),
    }
    expected_hash = _domain_hash(_DESIGN_HASH_DOMAIN, payload)
    if len(supplied_hash) != 64 or supplied_hash != expected_hash:
        raise ValueError("plan design_hash does not match its canonical charter")
    enrollment = charter.get("enrollment")
    probability_design = charter.get("probability_design")
    source_contract = _normalize_source_contract(charter.get("source_contract"))
    candidates = charter.get("candidates")
    exclusions = charter.get("exclusions")
    boundaries = charter.get("causal_boundaries")
    if not all(
        isinstance(item, Mapping)
        for item in (enrollment, probability_design, boundaries)
    ) or not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("plan charter has invalid assignment fields")
    if not isinstance(exclusions, Sequence) or isinstance(exclusions, (str, bytes)):
        raise ValueError("plan charter exclusions must be a sequence")
    if enrollment.get("eligible") is not True:
        raise ValueError("plan is not eligible for randomized assignment")
    _strict_keys(enrollment, _ENROLLMENT_KEYS, "plan enrollment")
    _strict_keys(probability_design, _PROBABILITY_DESIGN_KEYS, "plan probability design")
    if set(enrollment) != _ENROLLMENT_KEYS or set(probability_design) != _PROBABILITY_DESIGN_KEYS:
        raise ValueError("plan enrollment or probability design is missing required fields")
    if probability_design.get("decision_type") != "randomized":
        raise ValueError("eligible plan must declare a randomized decision type")
    if probability_design.get("probability_stage") != "post_filter":
        raise ValueError("eligible plan probabilities must be post-filter")
    if probability_design.get("assignment_algorithm") != ASSIGNMENT_ALGORITHM:
        raise ValueError("plan uses an unsupported assignment algorithm")
    if probability_design.get("assignment_unit") != "route":
        raise ValueError("plan assignment_unit must be route")
    if probability_design.get("assignment_performed") is not False:
        raise ValueError("plan must not claim an assignment was already performed")
    if boundaries.get("execution_enabled") is not False:
        raise ValueError("planner assignments cannot enable execution")
    if dict(boundaries) != _causal_boundaries():
        raise ValueError("plan causal boundaries do not match the non-executing study contract")

    baseline_action = str(enrollment.get("baseline_action") or "")
    if baseline_action not in AUTO_AGENT_MODE_ORDER:
        raise ValueError("plan baseline action is invalid")
    neighbors_raw = enrollment.get("adjacent_feasible_actions")
    if not isinstance(neighbors_raw, Sequence) or isinstance(neighbors_raw, (str, bytes)):
        raise ValueError("plan adjacent feasible actions must be a sequence")
    neighbors = [str(action) for action in neighbors_raw]
    if not 1 <= len(neighbors) <= MAX_ADJACENT_NEIGHBORS or len(set(neighbors)) != len(neighbors):
        raise ValueError("eligible plan must contain one or two unique adjacent actions")
    baseline_index = AUTO_AGENT_MODE_ORDER.index(baseline_action)
    if any(
        action not in AUTO_AGENT_MODE_ORDER
        or abs(AUTO_AGENT_MODE_ORDER.index(action) - baseline_index) != 1
        for action in neighbors
    ):
        raise ValueError("plan contains a nonadjacent exploration action")

    eligible_raw = probability_design.get("eligible_actions")
    probabilities_raw = probability_design.get("action_probabilities")
    if not isinstance(eligible_raw, Sequence) or isinstance(eligible_raw, (str, bytes)):
        raise ValueError("plan eligible actions must be a sequence")
    if not isinstance(probabilities_raw, Mapping):
        raise ValueError("plan action probabilities must be a mapping")
    eligible_actions = [str(action) for action in eligible_raw]
    expected_eligible = [
        action
        for action in AUTO_AGENT_MODE_ORDER
        if action == baseline_action or action in neighbors
    ]
    if eligible_actions != expected_eligible or set(probabilities_raw) != set(expected_eligible):
        raise ValueError("plan eligible actions do not match its adjacent enrollment set")
    requested_rate = _validate_rate(probability_design.get("requested_exploration_rate"))
    applied_rate = _validate_rate(probability_design.get("applied_exploration_rate"))
    if not math.isclose(requested_rate, applied_rate, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("eligible plan must apply its requested exploration rate")
    expected_neighbor_probability = round(applied_rate / len(neighbors), 12)
    if expected_neighbor_probability + 1e-12 < MIN_POSITIVE_EXPLORATION_PROBABILITY:
        raise ValueError("plan alternate probabilities fall below the study floor")
    cooked_probabilities: Dict[str, float] = {}
    for action in eligible_actions:
        raw_probability = probabilities_raw.get(action)
        if isinstance(raw_probability, bool):
            raise ValueError("plan action probabilities must be finite numbers")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("plan action probabilities must be finite numbers") from exc
        expected_probability = (
            round(1.0 - applied_rate, 12)
            if action == baseline_action
            else expected_neighbor_probability
        )
        if not math.isfinite(probability) or not math.isclose(
            probability, expected_probability, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("plan action probabilities do not match its versioned design")
        cooked_probabilities[action] = probability
    if not math.isclose(sum(cooked_probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("plan action probabilities must sum to 1")
    minimum_probability = probability_design.get("minimum_positive_exploration_probability")
    if not isinstance(minimum_probability, (int, float)) or isinstance(minimum_probability, bool):
        raise ValueError("plan minimum exploration probability is invalid")
    if not math.isclose(
        float(minimum_probability),
        expected_neighbor_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("plan minimum exploration probability does not match its distribution")

    normalized_candidates: List[Dict[str, Any]] = []
    candidate_actions: List[str] = []
    for raw_candidate in candidates:
        if not isinstance(raw_candidate, Mapping):
            raise ValueError("plan candidates must contain JSON objects")
        _strict_keys(raw_candidate, _STUDY_CANDIDATE_KEYS, "plan candidate")
        if set(raw_candidate) != _STUDY_CANDIDATE_KEYS:
            raise ValueError("plan candidate is missing required fields")
        action = str(raw_candidate.get("action") or "")
        if action not in expected_eligible or action in candidate_actions:
            raise ValueError("plan candidates must exactly match unique eligible actions")
        baseline_flag = raw_candidate.get("baseline")
        if not isinstance(baseline_flag, bool) or baseline_flag is not (action == baseline_action):
            raise ValueError("plan candidate baseline flags are invalid")
        distance = _nonnegative_int(
            raw_candidate.get("adjacent_distance"), "adjacent_distance"
        )
        expected_distance = abs(AUTO_AGENT_MODE_ORDER.index(action) - baseline_index)
        if distance != expected_distance or distance not in {0, 1}:
            raise ValueError("plan candidate adjacent distance is invalid")
        latency_tier = str(raw_candidate.get("latency_tier") or "")
        if latency_tier not in _LATENCY_TIERS:
            raise ValueError("plan candidate latency tier is invalid")
        normalized_candidates.append(
            {
                "action": action,
                "baseline": baseline_flag,
                "adjacent_distance": distance,
                "estimated_cost_units": _finite_nonnegative(
                    raw_candidate.get("estimated_cost_units"), "estimated_cost_units"
                ),
                "estimated_model_calls": _nonnegative_int(
                    raw_candidate.get("estimated_model_calls"),
                    "estimated_model_calls",
                    minimum=1,
                ),
                "planned_loop_steps": _nonnegative_int(
                    raw_candidate.get("planned_loop_steps"), "planned_loop_steps"
                ),
                "latency_tier": latency_tier,
            }
        )
        candidate_actions.append(action)
    if candidate_actions != expected_eligible:
        raise ValueError("plan candidate order must match canonical eligible action order")

    normalized_exclusions: List[Dict[str, Any]] = []
    exclusion_actions: List[str] = []
    for raw_exclusion in exclusions:
        if not isinstance(raw_exclusion, Mapping):
            raise ValueError("plan exclusions must contain JSON objects")
        _strict_keys(raw_exclusion, _STUDY_EXCLUSION_KEYS, "plan exclusion")
        if set(raw_exclusion) != _STUDY_EXCLUSION_KEYS:
            raise ValueError("plan exclusion is missing required fields")
        action = str(raw_exclusion.get("action") or "")
        if action not in AUTO_AGENT_MODE_ORDER or action in expected_eligible or action in exclusion_actions:
            raise ValueError("plan exclusions must be unique and outside eligible actions")
        reasons_raw = raw_exclusion.get("reasons")
        if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
            raise ValueError("plan exclusion reasons must be a sequence")
        reasons = [str(reason) for reason in reasons_raw]
        allowed_reasons = _ALLOWED_EXCLUSION_REASONS | {"nonadjacent_exploration_guard"}
        if not reasons or any(reason not in allowed_reasons for reason in reasons):
            raise ValueError("plan exclusion reasons are invalid")
        feasible = raw_exclusion.get("post_filter_feasible")
        if not isinstance(feasible, bool):
            raise ValueError("plan exclusion feasibility flag must be boolean")
        if feasible is not (reasons == ["nonadjacent_exploration_guard"]):
            raise ValueError("plan exclusion feasibility contradicts its reason")
        normalized_exclusions.append(
            {
                "action": action,
                "reasons": reasons,
                "post_filter_feasible": feasible,
            }
        )
        exclusion_actions.append(action)
    if set(candidate_actions) | set(exclusion_actions) != set(AUTO_AGENT_MODE_ORDER):
        raise ValueError("plan candidates and exclusions must partition all route actions")
    if exclusion_actions != sorted(
        exclusion_actions, key=lambda action: AUTO_AGENT_MODE_ORDER.index(action)
    ):
        raise ValueError("plan exclusions must use canonical action order")
    return {
        "plan": dict(plan),
        "study": dict(study),
        "charter": dict(charter),
        "design_hash": supplied_hash,
        "enrollment": dict(enrollment),
        "probability_design": dict(probability_design),
        "source_contract": source_contract,
        "candidates": normalized_candidates,
        "exclusions": normalized_exclusions,
    }


def validate_adjacent_route_study(plan: Any) -> Dict[str, Any]:
    """Validate an eligible rehearsal plan without drawing an assignment.

    The returned projection is a defensive, normalized view for other
    prompt-free control-plane tools.  It performs no randomness, I/O, ledger
    writes, model inference, or activation.
    """

    try:
        return _validated_assignment_plan(plan)
    except ValueError as exc:
        if str(exc) != "plan design_hash does not match its canonical charter":
            raise
        # JSON/JavaScript transports erase the lexical distinction between
        # integral floats (``1.0``) and integers (``1``).  Rebuild from the
        # closed v1 fields and accept only exact semantic equality with the
        # canonical builder result; this restores transport portability without
        # accepting a changed plan or a caller-supplied replacement hash.
        if not isinstance(plan, Mapping):
            raise
        charter = plan.get("charter")
        if not isinstance(charter, Mapping):
            raise
        enrollment = charter.get("enrollment")
        probability = charter.get("probability_design")
        traffic = charter.get("traffic_scenario")
        if not all(isinstance(item, Mapping) for item in (enrollment, probability, traffic)):
            raise
        observed = traffic.get("observed_label_scenario")
        high_probability = traffic.get("high_probability")
        candidates = charter.get("candidates")
        exclusions = charter.get("exclusions")
        if (
            not isinstance(observed, Mapping)
            or not isinstance(high_probability, Mapping)
            or not isinstance(candidates, Sequence)
            or isinstance(candidates, (str, bytes))
            or not isinstance(exclusions, Sequence)
            or isinstance(exclusions, (str, bytes))
        ):
            raise
        baseline = str(enrollment.get("baseline_action") or "")
        reconstructed_candidates: List[Dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, Mapping):
                raise
            reconstructed_candidates.append(
                {
                    "action": raw.get("action"),
                    "estimated_cost_units": raw.get("estimated_cost_units"),
                    "estimated_model_calls": raw.get("estimated_model_calls"),
                    "planned_loop_steps": raw.get("planned_loop_steps"),
                    "latency_tier": raw.get("latency_tier"),
                    "selected": str(raw.get("action") or "") == baseline,
                }
            )
        reconstructed_exclusions: List[Dict[str, Any]] = []
        for raw in exclusions:
            if not isinstance(raw, Mapping):
                raise
            if raw.get("post_filter_feasible") is True:
                reconstructed_candidates.append(
                    {
                        "action": raw.get("action"),
                        "estimated_cost_units": 0.0,
                        "estimated_model_calls": 1,
                        "planned_loop_steps": 0,
                        "latency_tier": "unknown",
                        "selected": False,
                    }
                )
            else:
                reconstructed_exclusions.append(
                    {"action": raw.get("action"), "reasons": raw.get("reasons")}
                )
        rebuilt = plan_adjacent_route_study(
            baseline,
            reconstructed_candidates,
            reconstructed_exclusions,
            source_contract=charter.get("source_contract"),
            exploration_rate=probability.get("requested_exploration_rate"),
            planned_routes=traffic.get("planned_routes"),
            scenario_confidence=high_probability.get("confidence_level"),
            assumed_feedback_rate=observed.get("assumed_feedback_rate"),
            target_observed_labels=observed.get(
                "target_observed_labels_per_alternate_action"
            ),
        )
        if dict(plan) != rebuilt:
            raise
        return _validated_assignment_plan(rebuilt)


def _nonce_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        cooked = value
    elif isinstance(value, str):
        if not value.strip():
            raise ValueError("assignment_nonce must not be empty")
        cooked = value.encode("utf-8")
    else:
        raise ValueError("assignment_nonce must be bytes or a string")
    if not 16 <= len(cooked) <= 240:
        raise ValueError("assignment_nonce must contain between 16 and 240 bytes")
    return cooked


def assign_adjacent_route(plan: Mapping[str, Any], assignment_nonce: Any) -> Dict[str, Any]:
    """Produce a deterministic rehearsal receipt without ledger-ready support."""

    validated = _validated_assignment_plan(plan)
    nonce = _nonce_bytes(assignment_nonce)
    nonce_hash = hashlib.sha256(_NONCE_HASH_DOMAIN + nonce).hexdigest()
    design_bytes = bytes.fromhex(validated["design_hash"])
    nonce_hash_bytes = bytes.fromhex(nonce_hash)
    draw_digest = hashlib.sha256(
        _DRAW_HASH_DOMAIN + design_bytes + nonce_hash_bytes
    ).digest()
    draw_int = int.from_bytes(draw_digest, "big", signed=False)
    with localcontext() as context:
        context.prec = 90
        draw = Decimal(draw_int) / Decimal(1 << 256)
        cumulative = Decimal(0)
        chosen_action = ""
        probabilities = validated["probability_design"].get("action_probabilities")
        eligible_actions = validated["probability_design"].get("eligible_actions")
        if not isinstance(probabilities, Mapping) or not isinstance(eligible_actions, Sequence):
            raise ValueError("plan probability design is invalid")
        for action in eligible_actions:
            cooked_action = str(action)
            if cooked_action not in probabilities:
                raise ValueError("eligible action is missing its assignment probability")
            probability = Decimal(str(probabilities[cooked_action]))
            if not probability.is_finite() or probability <= 0:
                raise ValueError("eligible assignment probabilities must be positive and finite")
            cumulative += probability
            if not chosen_action and draw < cumulative:
                chosen_action = cooked_action
        if abs(cumulative - Decimal(1)) > Decimal("0.000001"):
            raise ValueError("plan assignment probabilities must sum to 1")
        if not chosen_action:
            chosen_action = str(eligible_actions[-1])
        draw_decimal = format(draw, ".24f")

    applied_rate = float(validated["probability_design"]["applied_exploration_rate"])
    cooked_probabilities = {
        str(action): float(validated["probability_design"]["action_probabilities"][action])
        for action in validated["probability_design"]["eligible_actions"]
    }
    baseline_action = str(validated["enrollment"]["baseline_action"])
    receipt = {
        "schema_version": REHEARSAL_RECEIPT_SCHEMA_VERSION,
        "rehearsal_only": True,
        "study_design_hash": validated["design_hash"],
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "assignment_nonce_hash": nonce_hash,
        "assignment_draw_hash": draw_digest.hex(),
        "assignment_draw_unit_interval": draw_decimal,
        "baseline_action": baseline_action,
        "proposed_action": chosen_action,
        "is_exploration_proposal": chosen_action != baseline_action,
        "proposed_action_probability": float(cooked_probabilities[chosen_action]),
    }
    support_proposal = {
        "schema_version": REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION,
        "decision_type": "rehearsal_randomized_non_ledger",
        "probability_stage": "post_filter_rehearsal_only",
        "rehearsal_only": True,
        "study_design_hash": validated["design_hash"],
        "source_contract": dict(validated["source_contract"]),
        "baseline_action": baseline_action,
        "eligible_actions": list(cooked_probabilities),
        "action_probabilities": cooked_probabilities,
        "proposed_action": chosen_action,
        "proposed_action_probability": float(cooked_probabilities[chosen_action]),
        "exploration_rate": applied_rate,
        "candidates": [dict(row) for row in validated["candidates"]],
        "exclusions": [dict(row) for row in validated["exclusions"]],
        "ledger_eligible": False,
        "preassignment_commitment_sealed": False,
        "nonce_grinding_resistant": False,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
    }
    return {
        "schema_version": STUDY_ASSIGNMENT_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_version": STUDY_VERSION,
        "design_hash": validated["design_hash"],
        "rehearsal_assignment_receipt": receipt,
        "rehearsal_support_proposal": support_proposal,
        "ledger_eligible": False,
        "preassignment_commitment_sealed": False,
        "nonce_grinding_resistant": False,
        "activation_blockers": list(ACTIVATION_BLOCKERS),
        "side_effects": {
            "io_performed": False,
            "ledger_write_performed": False,
            "execution_started": False,
            "model_inference_started": False,
        },
        "causal_boundaries": _causal_boundaries(),
    }


__all__ = [
    "ACTIVATION_BLOCKERS",
    "ASSIGNMENT_ALGORITHM",
    "AUTO_AGENT_MODE_ORDER",
    "DEFAULT_ASSUMED_FEEDBACK_RATE",
    "DEFAULT_EXPLORATION_RATE",
    "DEFAULT_PLANNED_ROUTES",
    "DEFAULT_SCENARIO_CONFIDENCE",
    "DEFAULT_TARGET_OBSERVED_LABELS",
    "MAX_ADJACENT_NEIGHBORS",
    "MIN_POSITIVE_EXPLORATION_PROBABILITY",
    "REHEARSAL_RECEIPT_SCHEMA_VERSION",
    "REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION",
    "STUDY_ASSIGNMENT_SCHEMA_VERSION",
    "STUDY_ID",
    "STUDY_LABEL",
    "STUDY_PLAN_SCHEMA_VERSION",
    "STUDY_VERSION",
    "assign_adjacent_route",
    "plan_adjacent_route_study",
    "validate_adjacent_route_study",
]
