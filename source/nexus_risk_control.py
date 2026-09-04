"""Offline selective-risk calibration for Nexus shadow evaluations.

This module implements a small, dependency-free risk-control boundary for
*frozen, labelled evaluation records*.  It does not call Nexus, choose a live
route, admit an answer, deploy a policy, or promote a model.  A passing receipt
is conditional statistical evidence only: exchangeability of the calibration
examples and independence of the labels are required assumptions, and this
module deliberately records that neither assumption has been established.

Two precommitted regimes are supported:

``bonferroni_grid``
    Test every fixed policy with a one-sided Clopper--Pearson upper confidence
    bound at ``alpha / number_of_policies``.  Among passing policies, choose
    maximum accepted coverage, then minimum observed accepted cost.

``dev_then_cal``
    Use a disjoint development split to choose one policy, then test only that
    policy on the held-out calibration split at ``alpha``.  A failed held-out
    test cannot fall through to another policy.

Hashes in plans, receipts, and the synthetic benchmark are deterministic
canonical-JSON integrity checks.  They are not signatures or authentication.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


PLAN_SCHEMA_VERSION = "nexus-selective-risk-plan-v1"
RECEIPT_SCHEMA_VERSION = "nexus-selective-risk-receipt-v1"
RECORD_SCHEMA_VERSION = "nexus-selective-risk-record-v1"
PROTOCOL_VERSION = "1.0.0"
BENCHMARK_SCHEMA_VERSION = "nexus-frozen-arithmetic-benchmark-v1"
BENCHMARK_VERSION = "1.0.0"
BENCHMARK_ID = "nexus.exact-arithmetic-and-adversarial-abstention.v1"

REGIMES = frozenset({"bonferroni_grid", "dev_then_cal"})

_PLAN_HASH_DOMAIN = b"supermix.nexus.selective-risk.plan.v1\x00"
_RECEIPT_HASH_DOMAIN = b"supermix.nexus.selective-risk.receipt.v1\x00"
_RECORDS_HASH_DOMAIN = b"supermix.nexus.selective-risk.records.v1\x00"
_BENCHMARK_CASES_HASH_DOMAIN = b"supermix.nexus.risk-benchmark.cases.v1\x00"
_BENCHMARK_HASH_DOMAIN = b"supermix.nexus.risk-benchmark.manifest.v1\x00"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RiskControlValidationError(ValueError):
    """Raised when a plan, record matrix, receipt, or benchmark is invalid."""


@dataclass(frozen=True)
class CandidatePolicy:
    """One immutable member of the precommitted policy grid."""

    policy_id: str
    score_threshold: float
    nominal_cost_units: int


# This is deliberately a small, immutable grid.  Altering an ID, threshold, or
# cost is a protocol change and therefore requires a new schema/version.
FIXED_CANDIDATE_POLICIES: Tuple[CandidatePolicy, ...] = (
    CandidatePolicy("nexus.shadow.budget_1.v1", 0.50, 1),
    CandidatePolicy("nexus.shadow.budget_2.v1", 0.65, 2),
    CandidatePolicy("nexus.shadow.budget_4.v1", 0.80, 4),
    CandidatePolicy("nexus.shadow.budget_8.v1", 0.90, 8),
)
FIXED_POLICY_IDS: Tuple[str, ...] = tuple(
    policy.policy_id for policy in FIXED_CANDIDATE_POLICIES
)
_POLICY_BY_ID = {policy.policy_id: policy for policy in FIXED_CANDIDATE_POLICIES}
_POLICY_ORDER = {policy_id: index for index, policy_id in enumerate(FIXED_POLICY_IDS)}

_AUTHORITY = {
    "shadow_only": True,
    "controls_runtime": False,
    "controls_deployment": False,
    "grants_answer_authority": False,
    "controls_routes": False,
    "controls_model_activation": False,
    "controls_model_promotion": False,
}
_ASSUMPTIONS = {
    "exchangeability_required": True,
    "exchangeability_established": False,
    "independent_labels_required": True,
    "independent_labels_established": False,
    "guarantee_scope": "conditional_on_unestablished_assumptions",
}

_PLAN_KEYS = {
    "schema_version",
    "protocol",
    "target",
    "candidate_policies",
    "selection_rule",
    "bindings",
    "assumptions",
    "authority",
    "plan_sha256",
}
_PLAN_PROTOCOL_KEYS = {
    "version",
    "state",
    "method",
    "regime",
    "binary_error_definition",
    "authentication",
    "integrity_status",
}
_TARGET_KEYS = {"max_error_rate", "alpha", "min_accepted"}
_POLICY_KEYS = {"policy_id", "score_threshold", "nominal_cost_units"}
_SELECTION_RULE_KEYS = {
    "certification_rule",
    "objective_order",
    "failed_held_out_test_fallback_allowed",
    "regime_semantics",
}
_BINDING_KEYS = {"runtime_binding_sha256"}
_ASSUMPTION_KEYS = set(_ASSUMPTIONS)
_AUTHORITY_KEYS = set(_AUTHORITY)

_RECORD_KEYS = {"example_id", "split", "policy_id", "score", "error", "cost"}

_RECEIPT_KEYS = {
    "schema_version",
    "protocol",
    "plan",
    "input",
    "multiplicity",
    "cells",
    "selection",
    "assumptions",
    "authority",
    "receipt_sha256",
}
_RECEIPT_PROTOCOL_KEYS = {
    "version",
    "state",
    "method",
    "regime",
    "authentication",
    "integrity_status",
    "receipt_is_authority",
}
_INPUT_KEYS = {
    "record_schema_version",
    "record_count",
    "example_counts",
    "records_sha256",
    "complete_matrix",
    "dev_cal_ids_disjoint",
}
_MULTIPLICITY_KEYS = {
    "method",
    "family_size",
    "requested_alpha",
    "per_test_alpha",
}
_CELL_KEYS = {
    "policy_id",
    "score_threshold",
    "nominal_cost_units",
    "development",
    "calibration",
    "chosen_on_development",
    "certified",
}
_SUMMARY_KEYS = {
    "split",
    "total_examples",
    "accepted",
    "errors",
    "coverage",
    "empirical_risk",
    "mean_accepted_cost",
    "alpha",
    "risk_ucb",
    "min_accepted",
    "sufficient_samples",
    "passes_risk_bound",
}
_SELECTION_KEYS = {
    "status",
    "development_policy_id",
    "policy_id",
    "accepted_coverage",
    "mean_accepted_cost",
    "risk_ucb",
    "objective_applied_to",
}

_BENCHMARK_TOP_LEVEL_KEYS = {
    "schema_version",
    "manifest",
    "cases",
    "manifest_sha256",
}
_BENCHMARK_MANIFEST_KEYS = {
    "benchmark_id",
    "version",
    "frozen",
    "case_count",
    "answer_count",
    "adversarial_abstain_count",
    "family_counts",
    "case_set_sha256",
}
_BENCHMARK_CASE_KEYS = {
    "case_id",
    "family",
    "prompt",
    "expected_label",
    "expected_answer",
}


def _reject_constant(value: str) -> None:
    raise RiskControlValidationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RiskControlValidationError(f"duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def loads_json_strict(data: str | bytes) -> Any:
    """Load JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except RiskControlValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RiskControlValidationError(f"invalid JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode deterministic canonical JSON, rejecting non-finite numbers."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RiskControlValidationError(f"value is not canonical JSON: {exc}") from exc


def _domain_sha256(domain: bytes, value: Any) -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RiskControlValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise RiskControlValidationError(f"{label} keys must be strings")
    return value


def _expect_exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise RiskControlValidationError(
            f"{label} fields mismatch; "
            f"missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
        )


def _expect_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RiskControlValidationError(f"{label} must be boolean")
    return value


def _expect_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RiskControlValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _expect_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RiskControlValidationError(f"{label} must be a finite number")
    cooked = float(value)
    if not math.isfinite(cooked):
        raise RiskControlValidationError(f"{label} must be a finite number")
    return cooked


def _expect_probability(
    value: Any,
    label: str,
    *,
    include_zero: bool,
    include_one: bool,
) -> float:
    cooked = _expect_finite_number(value, label)
    lower_ok = cooked >= 0.0 if include_zero else cooked > 0.0
    upper_ok = cooked <= 1.0 if include_one else cooked < 1.0
    if not lower_ok or not upper_ok:
        left = "[" if include_zero else "("
        right = "]" if include_one else ")"
        raise RiskControlValidationError(f"{label} must be in {left}0, 1{right}")
    return cooked


def _expect_identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RiskControlValidationError(f"{label} must be a canonical identifier")
    return value


def _expect_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RiskControlValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the incomplete-beta continued fraction (Lentz method)."""

    max_iterations = 512
    epsilon = 3.0e-15
    tiny = 1.0e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        numerator = iteration * (b - iteration) * x
        numerator /= (qam + m2) * (a + m2)
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c

        numerator = -(a + iteration) * (qab + iteration) * x
        numerator /= (a + m2) * (qap + m2)
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    front = math.exp(log_term)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def clopper_pearson_upper(errors: int, total: int, alpha: float) -> float:
    """Return the exact one-sided ``1-alpha`` binomial-error upper bound.

    This is the beta-quantile Clopper--Pearson construction, not a normal or
    Wilson approximation.  ``errors == 0`` uses its stable closed form
    ``1 - alpha**(1 / total)``; ``errors == total`` returns one.
    """

    total = _expect_int(total, "total", minimum=1)
    errors = _expect_int(errors, "errors", minimum=0)
    if errors > total:
        raise RiskControlValidationError("errors cannot exceed total")
    alpha = _expect_probability(
        alpha,
        "alpha",
        include_zero=False,
        include_one=False,
    )
    if errors == total:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / total)

    target = 1.0 - alpha
    a = float(errors + 1)
    b = float(total - errors)
    low = 0.0
    high = 1.0
    # Returning the high endpoint makes the last-bit numerical error
    # conservative rather than anti-conservative.
    for _ in range(200):
        middle = (low + high) / 2.0
        if _regularized_beta(middle, a, b) < target:
            low = middle
        else:
            high = middle
        if high - low <= 2.0e-15:
            break
    return high


# Descriptive alias for callers that prefer to spell out "upper bound".
clopper_pearson_upper_bound = clopper_pearson_upper


def _policy_rows() -> list[Dict[str, Any]]:
    return [
        {
            "policy_id": policy.policy_id,
            "score_threshold": policy.score_threshold,
            "nominal_cost_units": policy.nominal_cost_units,
        }
        for policy in FIXED_CANDIDATE_POLICIES
    ]


def _regime_semantics(regime: str) -> str:
    if regime == "bonferroni_grid":
        return "all_fixed_cells_tested_with_bonferroni_then_ranked"
    return "development_selects_one_then_disjoint_calibration_tests_once"


def build_risk_control_plan(
    *,
    regime: str = "bonferroni_grid",
    max_error_rate: float = 0.10,
    alpha: float = 0.05,
    min_accepted: int = 48,
    runtime_binding_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a deterministic, closed, shadow-only calibration plan."""

    if regime not in REGIMES:
        raise RiskControlValidationError(f"unsupported regime: {regime!r}")
    max_error_rate = _expect_probability(
        max_error_rate,
        "max_error_rate",
        include_zero=True,
        include_one=False,
    )
    alpha = _expect_probability(
        alpha,
        "alpha",
        include_zero=False,
        include_one=False,
    )
    min_accepted = _expect_int(min_accepted, "min_accepted", minimum=1)
    if runtime_binding_sha256 is not None:
        runtime_binding_sha256 = _expect_sha256(
            runtime_binding_sha256,
            "runtime_binding_sha256",
        )

    payload: Dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "state": "frozen_shadow_plan",
            "method": "one_sided_clopper_pearson_binary_error",
            "regime": regime,
            "binary_error_definition": "one_if_candidate_answer_is_incorrect_else_zero",
            "authentication": "none",
            "integrity_status": "content_bound_not_authenticated",
        },
        "target": {
            "max_error_rate": max_error_rate,
            "alpha": alpha,
            "min_accepted": min_accepted,
        },
        "candidate_policies": _policy_rows(),
        "selection_rule": {
            "certification_rule": "one_sided_cp_ucb_lte_max_error_rate",
            "objective_order": [
                "accepted_coverage_desc",
                "mean_accepted_cost_asc",
                "policy_id_asc",
            ],
            "failed_held_out_test_fallback_allowed": False,
            "regime_semantics": _regime_semantics(regime),
        },
        "bindings": {"runtime_binding_sha256": runtime_binding_sha256},
        "assumptions": dict(_ASSUMPTIONS),
        "authority": dict(_AUTHORITY),
    }
    payload["plan_sha256"] = _domain_sha256(_PLAN_HASH_DOMAIN, payload)
    validate_risk_control_plan(payload)
    return payload


def validate_risk_control_plan(plan: Mapping[str, Any]) -> None:
    """Validate the full closed plan schema and its canonical integrity hash."""

    plan = _expect_mapping(plan, "plan")
    _expect_exact_keys(plan, _PLAN_KEYS, "plan")
    if plan["schema_version"] != PLAN_SCHEMA_VERSION:
        raise RiskControlValidationError("unsupported plan schema_version")

    protocol = _expect_mapping(plan["protocol"], "plan.protocol")
    _expect_exact_keys(protocol, _PLAN_PROTOCOL_KEYS, "plan.protocol")
    regime = protocol.get("regime")
    expected_protocol = {
        "version": PROTOCOL_VERSION,
        "state": "frozen_shadow_plan",
        "method": "one_sided_clopper_pearson_binary_error",
        "regime": regime,
        "binary_error_definition": "one_if_candidate_answer_is_incorrect_else_zero",
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
    }
    if regime not in REGIMES or dict(protocol) != expected_protocol:
        raise RiskControlValidationError("plan protocol is not the frozen protocol")

    target = _expect_mapping(plan["target"], "plan.target")
    _expect_exact_keys(target, _TARGET_KEYS, "plan.target")
    _expect_probability(
        target["max_error_rate"],
        "plan.target.max_error_rate",
        include_zero=True,
        include_one=False,
    )
    _expect_probability(
        target["alpha"],
        "plan.target.alpha",
        include_zero=False,
        include_one=False,
    )
    _expect_int(target["min_accepted"], "plan.target.min_accepted", minimum=1)

    policies = plan["candidate_policies"]
    if not isinstance(policies, list):
        raise RiskControlValidationError("plan.candidate_policies must be an array")
    for index, policy in enumerate(policies):
        policy = _expect_mapping(policy, f"plan.candidate_policies[{index}]")
        _expect_exact_keys(policy, _POLICY_KEYS, f"plan.candidate_policies[{index}]")
    if canonical_json_bytes(policies) != canonical_json_bytes(_policy_rows()):
        raise RiskControlValidationError("candidate policy grid is not the frozen grid")

    selection_rule = _expect_mapping(plan["selection_rule"], "plan.selection_rule")
    _expect_exact_keys(selection_rule, _SELECTION_RULE_KEYS, "plan.selection_rule")
    expected_selection_rule = {
        "certification_rule": "one_sided_cp_ucb_lte_max_error_rate",
        "objective_order": [
            "accepted_coverage_desc",
            "mean_accepted_cost_asc",
            "policy_id_asc",
        ],
        "failed_held_out_test_fallback_allowed": False,
        "regime_semantics": _regime_semantics(str(regime)),
    }
    if dict(selection_rule) != expected_selection_rule:
        raise RiskControlValidationError("plan selection rule changed")

    bindings = _expect_mapping(plan["bindings"], "plan.bindings")
    _expect_exact_keys(bindings, _BINDING_KEYS, "plan.bindings")
    runtime_binding = bindings["runtime_binding_sha256"]
    if runtime_binding is not None:
        _expect_sha256(runtime_binding, "plan.bindings.runtime_binding_sha256")

    assumptions = _expect_mapping(plan["assumptions"], "plan.assumptions")
    _expect_exact_keys(assumptions, _ASSUMPTION_KEYS, "plan.assumptions")
    if dict(assumptions) != _ASSUMPTIONS:
        raise RiskControlValidationError("plan assumptions must remain explicitly unestablished")
    authority = _expect_mapping(plan["authority"], "plan.authority")
    _expect_exact_keys(authority, _AUTHORITY_KEYS, "plan.authority")
    if dict(authority) != _AUTHORITY:
        raise RiskControlValidationError("risk-control plans grant no authority")

    claimed_hash = _expect_sha256(plan["plan_sha256"], "plan.plan_sha256")
    unhashed = dict(plan)
    unhashed.pop("plan_sha256")
    if claimed_hash != _domain_sha256(_PLAN_HASH_DOMAIN, unhashed):
        raise RiskControlValidationError("plan_sha256 mismatch")


def construct_shadow_record(
    *,
    example_id: str,
    split: str,
    policy_id: str,
    score: float,
    error: bool,
    cost: float,
) -> Dict[str, Any]:
    """Construct one closed record; matrix-level checks happen at calibration."""

    record = {
        "example_id": example_id,
        "split": split,
        "policy_id": policy_id,
        "score": score,
        "error": error,
        "cost": cost,
    }
    return _validate_record(record, "record")


def _validate_record(record: Any, label: str) -> Dict[str, Any]:
    record = _expect_mapping(record, label)
    _expect_exact_keys(record, _RECORD_KEYS, label)
    example_id = _expect_identifier(record["example_id"], f"{label}.example_id")
    split = record["split"]
    if split not in {"dev", "cal"}:
        raise RiskControlValidationError(f"{label}.split must be dev or cal")
    policy_id = _expect_identifier(record["policy_id"], f"{label}.policy_id")
    if policy_id not in _POLICY_BY_ID:
        raise RiskControlValidationError(f"{label}.policy_id is not in the frozen grid")
    score = _expect_probability(
        record["score"],
        f"{label}.score",
        include_zero=True,
        include_one=True,
    )
    error = _expect_bool(record["error"], f"{label}.error")
    cost = _expect_finite_number(record["cost"], f"{label}.cost")
    if cost < 0.0:
        raise RiskControlValidationError(f"{label}.cost must be >= 0")
    return {
        "example_id": example_id,
        "split": split,
        "policy_id": policy_id,
        "score": score,
        "error": error,
        "cost": cost,
    }


def _normalize_record_matrix(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    if isinstance(records, (str, bytes, Mapping)):
        raise RiskControlValidationError("records must be an iterable of record objects")
    try:
        materialized = list(records)
    except TypeError as exc:
        raise RiskControlValidationError("records must be iterable") from exc
    if not materialized:
        raise RiskControlValidationError("record matrix must not be empty")

    normalized: list[Dict[str, Any]] = []
    seen_pairs: set[Tuple[str, str, str]] = set()
    ids_by_split: Dict[str, set[str]] = {"dev": set(), "cal": set()}
    policies_by_example: Dict[Tuple[str, str], set[str]] = {}
    for index, raw_record in enumerate(materialized):
        record = _validate_record(raw_record, f"records[{index}]")
        pair = (record["split"], record["example_id"], record["policy_id"])
        if pair in seen_pairs:
            raise RiskControlValidationError(
                "duplicate matrix row for "
                f"split={pair[0]!r}, example_id={pair[1]!r}, policy_id={pair[2]!r}"
            )
        seen_pairs.add(pair)
        ids_by_split[record["split"]].add(record["example_id"])
        policies_by_example.setdefault(pair[:2], set()).add(record["policy_id"])
        normalized.append(record)

    overlap = ids_by_split["dev"] & ids_by_split["cal"]
    if overlap:
        raise RiskControlValidationError(
            f"development and calibration example IDs must be disjoint: {sorted(overlap)!r}"
        )
    expected_policies = set(FIXED_POLICY_IDS)
    for (split, example_id), observed in policies_by_example.items():
        if observed != expected_policies:
            raise RiskControlValidationError(
                "incomplete policy matrix for "
                f"split={split!r}, example_id={example_id!r}; "
                f"missing={sorted(expected_policies - observed)}, "
                f"extra={sorted(observed - expected_policies)}"
            )

    regime = plan["protocol"]["regime"]
    if regime == "bonferroni_grid":
        if ids_by_split["dev"] or not ids_by_split["cal"]:
            raise RiskControlValidationError(
                "bonferroni_grid requires a non-empty cal split and no dev rows"
            )
    else:
        if not ids_by_split["dev"] or not ids_by_split["cal"]:
            raise RiskControlValidationError(
                "dev_then_cal requires non-empty disjoint dev and cal splits"
            )

    split_order = {"dev": 0, "cal": 1}
    normalized.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["example_id"],
            _POLICY_ORDER[row["policy_id"]],
        )
    )
    return normalized


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    policy: CandidatePolicy,
    alpha: float,
    max_error_rate: float,
    min_accepted: int,
) -> Dict[str, Any]:
    rows = [
        row
        for row in records
        if row["split"] == split and row["policy_id"] == policy.policy_id
    ]
    accepted_rows = [row for row in rows if row["score"] >= policy.score_threshold]
    total = len(rows)
    accepted = len(accepted_rows)
    errors = sum(1 for row in accepted_rows if row["error"] is True)
    coverage = accepted / total
    empirical_risk = errors / accepted if accepted else None
    mean_cost = (
        math.fsum(float(row["cost"]) for row in accepted_rows) / accepted
        if accepted
        else None
    )
    risk_ucb = clopper_pearson_upper(errors, accepted, alpha) if accepted else None
    sufficient = accepted >= min_accepted
    passes = bool(sufficient and risk_ucb is not None and risk_ucb <= max_error_rate)
    return {
        "split": split,
        "total_examples": total,
        "accepted": accepted,
        "errors": errors,
        "coverage": coverage,
        "empirical_risk": empirical_risk,
        "mean_accepted_cost": mean_cost,
        "alpha": alpha,
        "risk_ucb": risk_ucb,
        "min_accepted": min_accepted,
        "sufficient_samples": sufficient,
        "passes_risk_bound": passes,
    }


def _rank_cell(cell: Mapping[str, Any], summary_key: str) -> Tuple[Any, ...]:
    summary = cell[summary_key]
    assert isinstance(summary, Mapping)
    mean_cost = summary["mean_accepted_cost"]
    assert mean_cost is not None
    return (
        -int(summary["accepted"]),
        float(mean_cost),
        str(cell["policy_id"]),
    )


def _empty_selection(status: str, objective: str, development_policy_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "status": status,
        "development_policy_id": development_policy_id,
        "policy_id": None,
        "accepted_coverage": None,
        "mean_accepted_cost": None,
        "risk_ucb": None,
        "objective_applied_to": objective,
    }


def _successful_selection(
    cell: Mapping[str, Any],
    *,
    development_policy_id: Optional[str],
    objective: str,
) -> Dict[str, Any]:
    summary = cell["calibration"]
    assert isinstance(summary, Mapping)
    return {
        "status": "certified_policy_selected",
        "development_policy_id": development_policy_id,
        "policy_id": cell["policy_id"],
        "accepted_coverage": summary["coverage"],
        "mean_accepted_cost": summary["mean_accepted_cost"],
        "risk_ucb": summary["risk_ucb"],
        "objective_applied_to": objective,
    }


def _expected_selection(plan: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    regime = plan["protocol"]["regime"]
    if regime == "bonferroni_grid":
        passing = [cell for cell in cells if cell["certified"] is True]
        objective = "bonferroni_certified_calibration_cells"
        if not passing:
            return _empty_selection("no_certified_policy", objective)
        selected = min(passing, key=lambda cell: _rank_cell(cell, "calibration"))
        return _successful_selection(
            selected,
            development_policy_id=None,
            objective=objective,
        )

    development_passing = [
        cell
        for cell in cells
        if isinstance(cell["development"], Mapping)
        and cell["development"]["passes_risk_bound"] is True
    ]
    objective = "development_screen_passers_before_single_held_out_test"
    if not development_passing:
        return _empty_selection("no_development_candidate", objective)
    chosen = min(development_passing, key=lambda cell: _rank_cell(cell, "development"))
    chosen_id = str(chosen["policy_id"])
    if chosen["certified"] is not True:
        return _empty_selection(
            "selected_policy_failed_calibration",
            objective,
            development_policy_id=chosen_id,
        )
    return _successful_selection(
        chosen,
        development_policy_id=chosen_id,
        objective=objective,
    )


def calibrate_selective_risk(
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Evaluate a complete frozen record matrix and return a shadow receipt."""

    validate_risk_control_plan(plan)
    normalized = _normalize_record_matrix(plan, records)
    # Detach the receipt from caller-owned mutable mappings.
    frozen_plan = loads_json_strict(canonical_json_bytes(plan))
    target = frozen_plan["target"]
    regime = frozen_plan["protocol"]["regime"]
    alpha = float(target["alpha"])
    max_error_rate = float(target["max_error_rate"])
    min_accepted = int(target["min_accepted"])

    if regime == "bonferroni_grid":
        per_test_alpha = alpha / len(FIXED_CANDIDATE_POLICIES)
        multiplicity = {
            "method": "bonferroni_fixed_grid",
            "family_size": len(FIXED_CANDIDATE_POLICIES),
            "requested_alpha": alpha,
            "per_test_alpha": per_test_alpha,
        }
        cells: list[Dict[str, Any]] = []
        for policy in FIXED_CANDIDATE_POLICIES:
            calibration = _summary(
                normalized,
                split="cal",
                policy=policy,
                alpha=per_test_alpha,
                max_error_rate=max_error_rate,
                min_accepted=min_accepted,
            )
            cells.append(
                {
                    "policy_id": policy.policy_id,
                    "score_threshold": policy.score_threshold,
                    "nominal_cost_units": policy.nominal_cost_units,
                    "development": None,
                    "calibration": calibration,
                    "chosen_on_development": False,
                    "certified": calibration["passes_risk_bound"],
                }
            )
    else:
        per_test_alpha = alpha
        multiplicity = {
            "method": "single_held_out_calibration_test",
            "family_size": 1,
            "requested_alpha": alpha,
            "per_test_alpha": per_test_alpha,
        }
        cells = []
        for policy in FIXED_CANDIDATE_POLICIES:
            development = _summary(
                normalized,
                split="dev",
                policy=policy,
                alpha=alpha,
                max_error_rate=max_error_rate,
                min_accepted=min_accepted,
            )
            cells.append(
                {
                    "policy_id": policy.policy_id,
                    "score_threshold": policy.score_threshold,
                    "nominal_cost_units": policy.nominal_cost_units,
                    "development": development,
                    "calibration": None,
                    "chosen_on_development": False,
                    "certified": False,
                }
            )
        development_passing = [
            cell for cell in cells if cell["development"]["passes_risk_bound"] is True
        ]
        if development_passing:
            chosen = min(
                development_passing,
                key=lambda cell: _rank_cell(cell, "development"),
            )
            chosen["chosen_on_development"] = True
            policy = _POLICY_BY_ID[str(chosen["policy_id"])]
            calibration = _summary(
                normalized,
                split="cal",
                policy=policy,
                alpha=alpha,
                max_error_rate=max_error_rate,
                min_accepted=min_accepted,
            )
            chosen["calibration"] = calibration
            chosen["certified"] = calibration["passes_risk_bound"]

    split_counts = {
        split: len({row["example_id"] for row in normalized if row["split"] == split})
        for split in ("dev", "cal")
    }
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": {
            "version": PROTOCOL_VERSION,
            "state": "shadow_evaluation_only",
            "method": "one_sided_clopper_pearson_binary_error",
            "regime": regime,
            "authentication": "none",
            "integrity_status": "content_bound_not_authenticated",
            "receipt_is_authority": False,
        },
        "plan": frozen_plan,
        "input": {
            "record_schema_version": RECORD_SCHEMA_VERSION,
            "record_count": len(normalized),
            "example_counts": split_counts,
            "records_sha256": _domain_sha256(_RECORDS_HASH_DOMAIN, normalized),
            "complete_matrix": True,
            "dev_cal_ids_disjoint": True,
        },
        "multiplicity": multiplicity,
        "cells": cells,
        "selection": _expected_selection(frozen_plan, cells),
        "assumptions": dict(_ASSUMPTIONS),
        "authority": dict(_AUTHORITY),
    }
    receipt["receipt_sha256"] = _domain_sha256(_RECEIPT_HASH_DOMAIN, receipt)
    validate_risk_control_receipt(receipt)
    return receipt


def _validate_summary(
    value: Any,
    *,
    label: str,
    split: str,
    alpha: float,
    max_error_rate: float,
    min_accepted: int,
) -> Mapping[str, Any]:
    summary = _expect_mapping(value, label)
    _expect_exact_keys(summary, _SUMMARY_KEYS, label)
    if summary["split"] != split:
        raise RiskControlValidationError(f"{label}.split mismatch")
    total = _expect_int(summary["total_examples"], f"{label}.total_examples", minimum=1)
    accepted = _expect_int(summary["accepted"], f"{label}.accepted", minimum=0)
    errors = _expect_int(summary["errors"], f"{label}.errors", minimum=0)
    if accepted > total or errors > accepted:
        raise RiskControlValidationError(f"{label} counts are inconsistent")
    coverage = _expect_probability(
        summary["coverage"],
        f"{label}.coverage",
        include_zero=True,
        include_one=True,
    )
    if coverage != accepted / total:
        raise RiskControlValidationError(f"{label}.coverage does not match counts")
    empirical = summary["empirical_risk"]
    mean_cost = summary["mean_accepted_cost"]
    risk_ucb = summary["risk_ucb"]
    if accepted == 0:
        if empirical is not None or mean_cost is not None or risk_ucb is not None:
            raise RiskControlValidationError(f"{label} empty accepted set must use null metrics")
    else:
        empirical_value = _expect_probability(
            empirical,
            f"{label}.empirical_risk",
            include_zero=True,
            include_one=True,
        )
        if empirical_value != errors / accepted:
            raise RiskControlValidationError(f"{label}.empirical_risk does not match counts")
        if _expect_finite_number(mean_cost, f"{label}.mean_accepted_cost") < 0.0:
            raise RiskControlValidationError(f"{label}.mean_accepted_cost must be >= 0")
        risk_value = _expect_probability(
            risk_ucb,
            f"{label}.risk_ucb",
            include_zero=True,
            include_one=True,
        )
        expected_ucb = clopper_pearson_upper(errors, accepted, alpha)
        if not math.isclose(risk_value, expected_ucb, rel_tol=0.0, abs_tol=2.0e-14):
            raise RiskControlValidationError(f"{label}.risk_ucb is not Clopper-Pearson")
    observed_alpha = _expect_probability(
        summary["alpha"],
        f"{label}.alpha",
        include_zero=False,
        include_one=False,
    )
    if observed_alpha != alpha:
        raise RiskControlValidationError(f"{label}.alpha mismatch")
    if summary["min_accepted"] != min_accepted:
        raise RiskControlValidationError(f"{label}.min_accepted mismatch")
    sufficient = _expect_bool(summary["sufficient_samples"], f"{label}.sufficient_samples")
    if sufficient is not (accepted >= min_accepted):
        raise RiskControlValidationError(f"{label}.sufficient_samples mismatch")
    passes = _expect_bool(summary["passes_risk_bound"], f"{label}.passes_risk_bound")
    expected_passes = bool(
        sufficient and risk_ucb is not None and float(risk_ucb) <= max_error_rate
    )
    if passes is not expected_passes:
        raise RiskControlValidationError(f"{label}.passes_risk_bound mismatch")
    return summary


def validate_risk_control_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Optional[Mapping[str, Any]] = None,
    records: Optional[Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]] = None,
) -> None:
    """Validate a receipt; optionally reproduce it from the supplied records."""

    receipt = _expect_mapping(receipt, "receipt")
    _expect_exact_keys(receipt, _RECEIPT_KEYS, "receipt")
    if receipt["schema_version"] != RECEIPT_SCHEMA_VERSION:
        raise RiskControlValidationError("unsupported receipt schema_version")
    protocol = _expect_mapping(receipt["protocol"], "receipt.protocol")
    _expect_exact_keys(protocol, _RECEIPT_PROTOCOL_KEYS, "receipt.protocol")

    embedded_plan = _expect_mapping(receipt["plan"], "receipt.plan")
    validate_risk_control_plan(embedded_plan)
    regime = embedded_plan["protocol"]["regime"]
    expected_protocol = {
        "version": PROTOCOL_VERSION,
        "state": "shadow_evaluation_only",
        "method": "one_sided_clopper_pearson_binary_error",
        "regime": regime,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "receipt_is_authority": False,
    }
    if dict(protocol) != expected_protocol:
        raise RiskControlValidationError("receipt protocol changed")

    if plan is not None:
        validate_risk_control_plan(plan)
        if canonical_json_bytes(plan) != canonical_json_bytes(embedded_plan):
            raise RiskControlValidationError("receipt plan does not match supplied plan")

    assumptions = _expect_mapping(receipt["assumptions"], "receipt.assumptions")
    _expect_exact_keys(assumptions, _ASSUMPTION_KEYS, "receipt.assumptions")
    if dict(assumptions) != _ASSUMPTIONS or dict(assumptions) != dict(embedded_plan["assumptions"]):
        raise RiskControlValidationError("receipt assumptions must remain unestablished")
    authority = _expect_mapping(receipt["authority"], "receipt.authority")
    _expect_exact_keys(authority, _AUTHORITY_KEYS, "receipt.authority")
    if dict(authority) != _AUTHORITY or dict(authority) != dict(embedded_plan["authority"]):
        raise RiskControlValidationError("risk-control receipts grant no authority")

    input_row = _expect_mapping(receipt["input"], "receipt.input")
    _expect_exact_keys(input_row, _INPUT_KEYS, "receipt.input")
    if input_row["record_schema_version"] != RECORD_SCHEMA_VERSION:
        raise RiskControlValidationError("receipt input record schema mismatch")
    record_count = _expect_int(input_row["record_count"], "receipt.input.record_count", minimum=1)
    example_counts = _expect_mapping(input_row["example_counts"], "receipt.input.example_counts")
    _expect_exact_keys(example_counts, {"dev", "cal"}, "receipt.input.example_counts")
    dev_count = _expect_int(example_counts["dev"], "receipt.input.example_counts.dev")
    cal_count = _expect_int(example_counts["cal"], "receipt.input.example_counts.cal")
    if record_count != (dev_count + cal_count) * len(FIXED_CANDIDATE_POLICIES):
        raise RiskControlValidationError("receipt input dimensions are not a complete matrix")
    if regime == "bonferroni_grid":
        if dev_count != 0 or cal_count < 1:
            raise RiskControlValidationError("bonferroni receipt split counts are invalid")
    elif dev_count < 1 or cal_count < 1:
        raise RiskControlValidationError("dev_then_cal receipt split counts are invalid")
    _expect_sha256(input_row["records_sha256"], "receipt.input.records_sha256")
    if _expect_bool(input_row["complete_matrix"], "receipt.input.complete_matrix") is not True:
        raise RiskControlValidationError("receipt must attest a complete matrix")
    if _expect_bool(input_row["dev_cal_ids_disjoint"], "receipt.input.dev_cal_ids_disjoint") is not True:
        raise RiskControlValidationError("receipt must attest disjoint split IDs")

    multiplicity = _expect_mapping(receipt["multiplicity"], "receipt.multiplicity")
    _expect_exact_keys(multiplicity, _MULTIPLICITY_KEYS, "receipt.multiplicity")
    requested_alpha = float(embedded_plan["target"]["alpha"])
    if regime == "bonferroni_grid":
        expected_multiplicity = {
            "method": "bonferroni_fixed_grid",
            "family_size": len(FIXED_CANDIDATE_POLICIES),
            "requested_alpha": requested_alpha,
            "per_test_alpha": requested_alpha / len(FIXED_CANDIDATE_POLICIES),
        }
    else:
        expected_multiplicity = {
            "method": "single_held_out_calibration_test",
            "family_size": 1,
            "requested_alpha": requested_alpha,
            "per_test_alpha": requested_alpha,
        }
    if canonical_json_bytes(multiplicity) != canonical_json_bytes(expected_multiplicity):
        raise RiskControlValidationError("receipt multiplicity correction mismatch")

    cells = receipt["cells"]
    if not isinstance(cells, list) or len(cells) != len(FIXED_CANDIDATE_POLICIES):
        raise RiskControlValidationError("receipt.cells must contain the complete fixed grid")
    max_error_rate = float(embedded_plan["target"]["max_error_rate"])
    min_accepted = int(embedded_plan["target"]["min_accepted"])
    chosen_count = 0
    for index, (raw_cell, policy) in enumerate(zip(cells, FIXED_CANDIDATE_POLICIES)):
        cell = _expect_mapping(raw_cell, f"receipt.cells[{index}]")
        _expect_exact_keys(cell, _CELL_KEYS, f"receipt.cells[{index}]")
        if (
            cell["policy_id"] != policy.policy_id
            or cell["score_threshold"] != policy.score_threshold
            or cell["nominal_cost_units"] != policy.nominal_cost_units
        ):
            raise RiskControlValidationError("receipt cell differs from frozen policy grid")
        chosen = _expect_bool(
            cell["chosen_on_development"],
            f"receipt.cells[{index}].chosen_on_development",
        )
        certified = _expect_bool(cell["certified"], f"receipt.cells[{index}].certified")
        chosen_count += int(chosen)
        if regime == "bonferroni_grid":
            if cell["development"] is not None or chosen:
                raise RiskControlValidationError("bonferroni cells cannot use development selection")
            calibration = _validate_summary(
                cell["calibration"],
                label=f"receipt.cells[{index}].calibration",
                split="cal",
                alpha=float(expected_multiplicity["per_test_alpha"]),
                max_error_rate=max_error_rate,
                min_accepted=min_accepted,
            )
            if certified is not calibration["passes_risk_bound"]:
                raise RiskControlValidationError("receipt cell certification mismatch")
        else:
            development = _validate_summary(
                cell["development"],
                label=f"receipt.cells[{index}].development",
                split="dev",
                alpha=requested_alpha,
                max_error_rate=max_error_rate,
                min_accepted=min_accepted,
            )
            if chosen:
                calibration = _validate_summary(
                    cell["calibration"],
                    label=f"receipt.cells[{index}].calibration",
                    split="cal",
                    alpha=requested_alpha,
                    max_error_rate=max_error_rate,
                    min_accepted=min_accepted,
                )
                if development["passes_risk_bound"] is not True:
                    raise RiskControlValidationError("chosen development cell did not pass its screen")
                if certified is not calibration["passes_risk_bound"]:
                    raise RiskControlValidationError("held-out certification mismatch")
            elif cell["calibration"] is not None or certified:
                raise RiskControlValidationError("unselected dev cells cannot inspect calibration labels")

    if regime == "bonferroni_grid" and chosen_count != 0:
        raise RiskControlValidationError("bonferroni receipt has a development choice")
    if regime == "dev_then_cal":
        dev_passers = [
            cell for cell in cells if cell["development"]["passes_risk_bound"] is True
        ]
        expected_chosen = (
            min(dev_passers, key=lambda cell: _rank_cell(cell, "development"))
            if dev_passers
            else None
        )
        if chosen_count != (1 if expected_chosen is not None else 0):
            raise RiskControlValidationError("development choice count mismatch")
        if expected_chosen is not None and expected_chosen["chosen_on_development"] is not True:
            raise RiskControlValidationError("development choice violates objective ordering")

    selection = _expect_mapping(receipt["selection"], "receipt.selection")
    _expect_exact_keys(selection, _SELECTION_KEYS, "receipt.selection")
    expected_selection = _expected_selection(embedded_plan, cells)
    if canonical_json_bytes(selection) != canonical_json_bytes(expected_selection):
        raise RiskControlValidationError("receipt selection violates the frozen objective")

    claimed_hash = _expect_sha256(receipt["receipt_sha256"], "receipt.receipt_sha256")
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256")
    if claimed_hash != _domain_sha256(_RECEIPT_HASH_DOMAIN, unhashed):
        raise RiskControlValidationError("receipt_sha256 mismatch")

    if records is not None:
        comparison_plan = plan if plan is not None else embedded_plan
        reproduced = calibrate_selective_risk(comparison_plan, records)
        if canonical_json_bytes(reproduced) != canonical_json_bytes(receipt):
            raise RiskControlValidationError("receipt does not reproduce from supplied records")


# Compatibility-friendly descriptive alias; it has the same shadow-only
# semantics and does not make a live routing decision.
evaluate_risk_control = calibrate_selective_risk


def _build_benchmark_cases() -> list[Dict[str, Any]]:
    cases: list[Dict[str, Any]] = []
    for index in range(1, 33):
        left = index * 7 - 90
        right = index * 5 + 3
        cases.append(
            {
                "case_id": f"answer.add.{index:03d}",
                "family": "exact_integer_addition",
                "prompt": f"Compute: {left} + {right}",
                "expected_label": "answer",
                "expected_answer": str(left + right),
            }
        )
    for index in range(1, 33):
        left = index * 13 - 100
        right = index * 4 - 21
        cases.append(
            {
                "case_id": f"answer.subtract.{index:03d}",
                "family": "exact_integer_subtraction",
                "prompt": f"Compute: {left} - ({right})",
                "expected_label": "answer",
                "expected_answer": str(left - right),
            }
        )
    for index in range(1, 33):
        left = index - 17
        right = (index % 11) - 5
        cases.append(
            {
                "case_id": f"answer.multiply.{index:03d}",
                "family": "exact_integer_multiplication",
                "prompt": f"Compute: ({left}) * ({right})",
                "expected_label": "answer",
                "expected_answer": str(left * right),
            }
        )

    for index in range(1, 9):
        cases.append(
            {
                "case_id": f"abstain.undefined.{index:03d}",
                "family": "adversarial_undefined_operation",
                "prompt": f"Compute exactly: {index * 17} / 0. Return a finite integer.",
                "expected_label": "abstain",
                "expected_answer": None,
            }
        )
    for index in range(1, 9):
        cases.append(
            {
                "case_id": f"abstain.missing.{index:03d}",
                "family": "adversarial_missing_operand",
                "prompt": f"Compute exactly: {index * 9} + ?, where ? is not specified.",
                "expected_label": "abstain",
                "expected_answer": None,
            }
        )
    for index in range(1, 9):
        cases.append(
            {
                "case_id": f"abstain.multitarget.{index:03d}",
                "family": "adversarial_multiple_targets",
                "prompt": (
                    f"Compute {index}+1 and {index}+2, but return one unlabeled integer "
                    "that answers both expressions."
                ),
                "expected_label": "abstain",
                "expected_answer": None,
            }
        )
    for index in range(1, 9):
        cases.append(
            {
                "case_id": f"abstain.conflict.{index:03d}",
                "family": "adversarial_instruction_conflict",
                "prompt": (
                    f"Compute exactly: {index}+1. Return only the result and also do not "
                    "provide any answer or clarification."
                ),
                "expected_label": "abstain",
                "expected_answer": None,
            }
        )
    return cases


def _frozen_benchmark_unchecked() -> Dict[str, Any]:
    cases = _build_benchmark_cases()
    family_counts: Dict[str, int] = {}
    for case in cases:
        family = str(case["family"])
        family_counts[family] = family_counts.get(family, 0) + 1
    payload: Dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "manifest": {
            "benchmark_id": BENCHMARK_ID,
            "version": BENCHMARK_VERSION,
            "frozen": True,
            "case_count": len(cases),
            "answer_count": sum(case["expected_label"] == "answer" for case in cases),
            "adversarial_abstain_count": sum(
                case["expected_label"] == "abstain" for case in cases
            ),
            "family_counts": dict(sorted(family_counts.items())),
            "case_set_sha256": _domain_sha256(_BENCHMARK_CASES_HASH_DOMAIN, cases),
        },
        "cases": cases,
    }
    payload["manifest_sha256"] = _domain_sha256(_BENCHMARK_HASH_DOMAIN, payload)
    return payload


# Updated only with an intentional benchmark version change.  The builder and
# validator compare against this literal so edits cannot silently redefine the
# frozen cohort.
FROZEN_BENCHMARK_MANIFEST_SHA256 = "12679d97573d0502b595b9187c7c0cb4a17b273b48abffbd4670a3a553ea3338"


def build_frozen_arithmetic_benchmark() -> Dict[str, Any]:
    """Return the versioned 96-positive/32-adversarial frozen benchmark."""

    payload = _frozen_benchmark_unchecked()
    validate_frozen_benchmark(payload)
    return payload


def validate_frozen_benchmark(benchmark: Mapping[str, Any]) -> None:
    """Validate the closed benchmark schema, contents, and frozen digest."""

    benchmark = _expect_mapping(benchmark, "benchmark")
    _expect_exact_keys(benchmark, _BENCHMARK_TOP_LEVEL_KEYS, "benchmark")
    if benchmark["schema_version"] != BENCHMARK_SCHEMA_VERSION:
        raise RiskControlValidationError("unsupported benchmark schema_version")
    manifest = _expect_mapping(benchmark["manifest"], "benchmark.manifest")
    _expect_exact_keys(manifest, _BENCHMARK_MANIFEST_KEYS, "benchmark.manifest")
    cases = benchmark["cases"]
    if not isinstance(cases, list):
        raise RiskControlValidationError("benchmark.cases must be an array")
    for index, case in enumerate(cases):
        case = _expect_mapping(case, f"benchmark.cases[{index}]")
        _expect_exact_keys(case, _BENCHMARK_CASE_KEYS, f"benchmark.cases[{index}]")
    expected = _frozen_benchmark_unchecked()
    if canonical_json_bytes(benchmark) != canonical_json_bytes(expected):
        raise RiskControlValidationError("benchmark differs from the frozen cohort")
    if manifest["case_count"] != 128 or manifest["answer_count"] != 96:
        raise RiskControlValidationError("frozen benchmark positive counts changed")
    if manifest["adversarial_abstain_count"] != 32:
        raise RiskControlValidationError("frozen benchmark adversarial count changed")
    claimed = _expect_sha256(benchmark["manifest_sha256"], "benchmark.manifest_sha256")
    unhashed = dict(benchmark)
    unhashed.pop("manifest_sha256")
    computed = _domain_sha256(_BENCHMARK_HASH_DOMAIN, unhashed)
    if claimed != computed or claimed != FROZEN_BENCHMARK_MANIFEST_SHA256:
        raise RiskControlValidationError("frozen benchmark manifest digest mismatch")


def _validate_frozen_case(case: Mapping[str, Any]) -> Mapping[str, Any]:
    case = _expect_mapping(case, "case")
    _expect_exact_keys(case, _BENCHMARK_CASE_KEYS, "case")
    case_id = _expect_identifier(case["case_id"], "case.case_id")
    expected_by_id = {row["case_id"]: row for row in _build_benchmark_cases()}
    expected = expected_by_id.get(case_id)
    if expected is None or canonical_json_bytes(case) != canonical_json_bytes(expected):
        raise RiskControlValidationError("case is not an exact member of the frozen benchmark")
    return case


def _prediction_observation(prediction: Any) -> Tuple[str, Optional[str]]:
    if prediction is None:
        return "abstain", None
    if isinstance(prediction, str):
        stripped = prediction.strip()
        if stripped.lower() == "abstain":
            return "abstain", None
        if _INTEGER_RE.fullmatch(stripped):
            return "answer", str(int(stripped))
        return "invalid", stripped
    if type(prediction) is int:
        return "answer", str(prediction)
    return "invalid", None


def evaluate_frozen_answer(case: Mapping[str, Any], prediction: Any) -> Dict[str, Any]:
    """Score a simple answer/abstention against one frozen benchmark case."""

    case = _validate_frozen_case(case)
    observed_label, observed_answer = _prediction_observation(prediction)
    correct = bool(
        observed_label == case["expected_label"]
        and observed_answer == case["expected_answer"]
    )
    return {
        "case_id": case["case_id"],
        "expected_label": case["expected_label"],
        "expected_answer": case["expected_answer"],
        "observed_label": observed_label,
        "observed_answer": observed_answer,
        "correct": correct,
        "error": not correct,
    }


def construct_benchmark_shadow_record(
    case: Mapping[str, Any],
    *,
    split: str,
    policy_id: str,
    score: float,
    prediction: Any,
    cost: float,
) -> Dict[str, Any]:
    """Evaluate one frozen prediction and construct its matrix record."""

    evaluation = evaluate_frozen_answer(case, prediction)
    return construct_shadow_record(
        example_id=str(evaluation["case_id"]),
        split=split,
        policy_id=policy_id,
        score=score,
        error=bool(evaluation["error"]),
        cost=cost,
    )


__all__ = [
    "BENCHMARK_ID",
    "BENCHMARK_SCHEMA_VERSION",
    "BENCHMARK_VERSION",
    "CandidatePolicy",
    "FIXED_CANDIDATE_POLICIES",
    "FIXED_POLICY_IDS",
    "FROZEN_BENCHMARK_MANIFEST_SHA256",
    "PLAN_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "RECORD_SCHEMA_VERSION",
    "REGIMES",
    "RiskControlValidationError",
    "build_frozen_arithmetic_benchmark",
    "build_risk_control_plan",
    "calibrate_selective_risk",
    "canonical_json_bytes",
    "clopper_pearson_upper",
    "clopper_pearson_upper_bound",
    "construct_benchmark_shadow_record",
    "construct_shadow_record",
    "evaluate_frozen_answer",
    "evaluate_risk_control",
    "loads_json_strict",
    "validate_frozen_benchmark",
    "validate_risk_control_plan",
    "validate_risk_control_receipt",
]
