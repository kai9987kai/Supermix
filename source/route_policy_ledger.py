"""Durable two-phase route-decision and feedback ledger.

The ledger commits a decision before route execution begins, then records its
terminal outcome in a separate transaction.  A process failure therefore
leaves durable, explicitly ``inflight`` evidence instead of silently dropping
the attempted route.  SQLite WAL mode plus short per-operation connections and
``BEGIN IMMEDIATE`` transactions provide safe access from concurrent workers.

Only a domain-separated SHA-256 digest of the session identifier is stored.
Missing feedback is reported as unknown; it is never interpreted as negative.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


LEDGER_SCHEMA_VERSION = 3
DECISION_STATUSES: Tuple[str, ...] = ("inflight", "completed", "failed")
SUPPORT_SCHEMA_VERSION = "route-support-v1"
EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION = "route-execution-assignment-v1"
DECISION_FINGERPRINT_SCHEMA_VERSION = "route-decision-fingerprint-v1"
DECISION_TYPES: Tuple[str, ...] = ("deterministic", "randomized", "legacy_unknown")
OUTCOME_CONTRACT_SCHEMA_VERSION = "route-outcome-contract-v1"
OUTCOME_MATURITY_SCHEMA_VERSION = "route-outcome-maturity-v1"
OUTCOME_NAMES: Tuple[str, ...] = (
    "route_success",
    "user_quality_rating",
    "cost",
    "latency",
)
_SESSION_HASH_DOMAIN = b"supermix-route-policy-ledger-v1\x00"
_CANDIDATE_HASH_DOMAIN = b"supermix-route-candidate-set-v1\x00"
_DISTRIBUTION_HASH_DOMAIN = b"supermix-route-distribution-v1\x00"
_DECISION_FINGERPRINT_DOMAIN = b"supermix-route-decision-record-v1\x00"
_OUTCOME_CONTRACT_HASH_DOMAIN = b"supermix-route-outcome-contract-v1\x00"
_EXECUTED_ASSIGNMENT_COMMITMENT_RE = re.compile(
    rf"^{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:[0-9a-f]{{64}}$"
)
_PROMPT_BEARING_CONTEXT_KEYS = frozenset(
    {
        "conversation",
        "input",
        "input_text",
        "memory_context",
        "message",
        "messages",
        "prompt",
        "query",
        "raw_prompt",
        "raw_text",
        "text",
        "tool_context",
        "user_prompt",
    }
)

_OUTCOME_CONTRACT_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "route_success": {
        "outcome_definition_version": "route-success-v1",
        "observation_policy_id": "route-completion",
        "observation_policy_version": "1",
        "value_type": "boolean",
        "unit": "boolean",
        "maturity_delay_seconds": 0.0,
    },
    "user_quality_rating": {
        "outcome_definition_version": "user-quality-rating-v1",
        "observation_policy_id": "explicit-route-feedback",
        "observation_policy_version": "1",
        "value_type": "ordinal",
        "unit": "signed_unit_interval",
        "maturity_delay_seconds": 0.0,
    },
    "cost": {
        "outcome_definition_version": "route-cost-v1",
        "observation_policy_id": "runtime-economics",
        "observation_policy_version": "1",
        "value_type": "number",
        "unit": "cost_units",
        "maturity_delay_seconds": 0.0,
    },
    "latency": {
        "outcome_definition_version": "route-latency-v1",
        "observation_policy_id": "runtime-economics",
        "observation_policy_version": "1",
        "value_type": "number",
        "unit": "milliseconds",
        "maturity_delay_seconds": 0.0,
    },
}


class RoutePolicyLedgerError(RuntimeError):
    """Base class for ledger failures with stable caller-facing semantics."""


class DecisionNotFoundError(RoutePolicyLedgerError, KeyError):
    """Raised when a route id does not exist in the ledger."""


class LedgerConflictError(RoutePolicyLedgerError):
    """Raised when an idempotency key or completed decision is reused differently."""


def hash_session_identity(session_id: str) -> str:
    """Return the stable, domain-separated session identity stored in SQLite."""

    cooked = str(session_id or "").strip()
    if not cooked:
        raise ValueError("session_id is required")
    return hashlib.sha256(_SESSION_HASH_DOMAIN + cooked.encode("utf-8")).hexdigest()


def _text(value: Any, name: str, *, limit: int = 240) -> str:
    cooked = str(value or "").strip()
    if not cooked:
        raise ValueError(f"{name} is required")
    if len(cooked) > limit:
        raise ValueError(f"{name} must be at most {limit} characters")
    return cooked


def _route_id(value: Any) -> str:
    if value is None:
        return str(uuid.uuid4())
    cooked = str(value).strip().lower()
    if not cooked:
        raise ValueError("route_id must be a UUID or omitted")
    try:
        uuid.UUID(cooked)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError("route_id must be a valid UUID") from exc
    # Preserve a caller's valid hex-vs-hyphen representation so an already
    # propagated route id remains byte-for-byte usable by feedback clients.
    return cooked


def _existing_route_id(value: Any) -> str:
    """Normalize a lookup key, treating invalid legacy ids as absent rows."""

    try:
        return _route_id(value)
    except ValueError as exc:
        raise DecisionNotFoundError(f"unknown route_id: {str(value or '').strip()}") from exc


def _json_mapping(value: Any, name: str) -> str:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable without non-finite numbers") from exc


def _eligible_modes(value: Any) -> List[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("eligible_modes must be a non-empty sequence")
    modes = [str(mode or "").strip() for mode in value]
    if not modes or any(not mode for mode in modes):
        raise ValueError("eligible_modes must contain non-empty mode names")
    if len(set(modes)) != len(modes):
        raise ValueError("eligible_modes must not contain duplicates")
    return modes


def _probabilities(value: Any, eligible_modes: Sequence[str], chosen_mode: str) -> Dict[str, float]:
    if value is None:
        raise ValueError("action_probabilities are required for durable route decisions")
    if not isinstance(value, Mapping):
        raise ValueError("action_probabilities must be a JSON object")
    probabilities: Dict[str, float] = {}
    for raw_mode, raw_probability in value.items():
        mode = str(raw_mode or "").strip()
        if isinstance(raw_probability, bool):
            raise ValueError("action probabilities must be finite numbers between 0 and 1")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("action probabilities must be finite numbers between 0 and 1") from exc
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError("action probabilities must be finite numbers between 0 and 1")
        probabilities[mode] = probability
    if set(probabilities) != set(eligible_modes):
        raise ValueError("action_probabilities must have exactly the eligible mode keys")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("action probabilities must sum to 1")
    if probabilities.get(chosen_mode, 0.0) <= 0.0:
        raise ValueError("chosen_mode must have positive logged probability")
    return probabilities


def _legacy_probability_mapping(value: Any) -> Dict[str, float]:
    """Decode a legacy probability object without trusting its JSON shape."""

    try:
        raw = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, Mapping):
        return {}

    probabilities: Dict[str, float] = {}
    for raw_mode, raw_probability in raw.items():
        if isinstance(raw_probability, bool):
            return {}
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError, OverflowError):
            return {}
        if not math.isfinite(probability):
            return {}
        probabilities[str(raw_mode)] = probability
    return probabilities


def _canonical_json(value: Any, name: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable without non-finite numbers") from exc


def _prompt_free_context(value: Any) -> Any:
    """Remove raw-text fields from the read-only policy-evidence projection."""

    if isinstance(value, Mapping):
        sanitized: Dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = key.strip().lower()
            if (
                normalized in _PROMPT_BEARING_CONTEXT_KEYS
                or normalized.endswith("_prompt")
                or normalized.endswith("_message")
            ):
                continue
            sanitized[key] = _prompt_free_context(raw_value)
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_prompt_free_context(item) for item in value]
    return value


def _domain_hash(domain: bytes, value: Any, name: str) -> str:
    return hashlib.sha256(domain + _canonical_json(value, name).encode("utf-8")).hexdigest()


def _decision_record_fingerprint(
    *,
    policy_name: str,
    policy_version: str,
    policy_schema_version: str,
    decision_context: Mapping[str, Any],
    eligible_modes: Sequence[str],
    action_probabilities: Mapping[str, float],
    chosen_mode: str,
    candidate_set_hash: str,
    distribution_hash: str,
) -> str:
    """Hash the immutable, prompt-free route decision made before execution."""

    payload = {
        "schema_version": DECISION_FINGERPRINT_SCHEMA_VERSION,
        "policy": {
            "name": policy_name,
            "version": policy_version,
            "schema_version": policy_schema_version,
        },
        "decision_context": dict(decision_context),
        "eligible_modes": list(eligible_modes),
        "action_probabilities": dict(action_probabilities),
        "chosen_mode": chosen_mode,
        "candidate_set_hash": candidate_set_hash,
        "distribution_hash": distribution_hash,
    }
    return _domain_hash(
        _DECISION_FINGERPRINT_DOMAIN,
        payload,
        "route decision record fingerprint",
    )


def _normalize_logging_support(
    value: Optional[Mapping[str, Any]],
    *,
    eligible_modes: Sequence[str],
    probabilities: Mapping[str, float],
    chosen_mode: str,
) -> Dict[str, Any]:
    """Validate and fingerprint the immutable post-filter logging envelope."""

    positive_actions = [mode for mode in eligible_modes if float(probabilities.get(mode, 0.0)) > 0.0]
    if value is None:
        if len(positive_actions) != 1:
            raise ValueError(
                "randomized action_probabilities require logging_support with an assignment commitment"
            )
        raw: Mapping[str, Any] = {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "decision_type": "deterministic",
            "probability_stage": "post_filter",
            "sampler": {
                "name": "argmax",
                "version": "1",
                "exploration_rate": 0.0,
                "assignment_unit": "route",
                "assignment_commitment": None,
            },
            "candidates": [{"action": mode} for mode in eligible_modes],
            "exclusions": [],
        }
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise ValueError("logging_support must be a JSON object")

    schema_version = _text(
        raw.get("schema_version") or SUPPORT_SCHEMA_VERSION,
        "support schema_version",
        limit=80,
    )
    if schema_version != SUPPORT_SCHEMA_VERSION:
        raise ValueError(
            f"logging_support schema_version must be {SUPPORT_SCHEMA_VERSION}"
        )
    if raw.get("shadow_only") is True or raw.get("rehearsal_only") is True:
        raise ValueError("shadow-only or rehearsal-only support is never ledger eligible")
    if "ledger_eligible" in raw and raw.get("ledger_eligible") is not True:
        raise ValueError("logging_support explicitly marked ledger_eligible=false is rejected")
    decision_type = str(raw.get("decision_type") or "").strip().lower()
    if decision_type == "stochastic":
        decision_type = "randomized"
    if decision_type not in {"deterministic", "randomized"}:
        raise ValueError("logging_support decision_type must be deterministic or randomized")
    probability_stage = str(raw.get("probability_stage") or "").strip().lower()
    if probability_stage != "post_filter":
        raise ValueError("logging_support probability_stage must be post_filter")

    candidates_raw = raw.get("candidates")
    if not isinstance(candidates_raw, Sequence) or isinstance(candidates_raw, (str, bytes)):
        raise ValueError("logging_support candidates must be a sequence")
    candidates: List[Dict[str, Any]] = []
    candidate_actions: List[str] = []
    for candidate in candidates_raw:
        if not isinstance(candidate, Mapping):
            raise ValueError("logging_support candidates must be JSON objects")
        action = _text(candidate.get("action"), "candidate action", limit=80)
        candidate_actions.append(action)
        normalized_candidate = dict(candidate)
        normalized_candidate["action"] = action
        _canonical_json(normalized_candidate, "logging_support candidate")
        candidates.append(normalized_candidate)
    if candidate_actions != list(eligible_modes):
        raise ValueError("logging_support candidate actions must exactly match eligible_modes order")

    exclusions_raw = raw.get("exclusions") or []
    if not isinstance(exclusions_raw, Sequence) or isinstance(exclusions_raw, (str, bytes)):
        raise ValueError("logging_support exclusions must be a sequence")
    exclusions: List[Dict[str, Any]] = []
    excluded_actions: set[str] = set()
    for exclusion in exclusions_raw:
        if not isinstance(exclusion, Mapping):
            raise ValueError("logging_support exclusions must be JSON objects")
        action = _text(exclusion.get("action"), "excluded action", limit=80)
        if action in eligible_modes or action in excluded_actions:
            raise ValueError("logging_support exclusions must be unique and outside eligible_modes")
        reasons_raw = exclusion.get("reasons")
        if not isinstance(reasons_raw, Sequence) or isinstance(reasons_raw, (str, bytes)):
            raise ValueError("logging_support exclusion reasons must be a non-empty sequence")
        reasons = [_text(reason, "exclusion reason", limit=120) for reason in reasons_raw]
        if not reasons:
            raise ValueError("logging_support exclusion reasons must be a non-empty sequence")
        excluded_actions.add(action)
        normalized_exclusion = dict(exclusion)
        normalized_exclusion.update({"action": action, "reasons": reasons})
        _canonical_json(normalized_exclusion, "logging_support exclusion")
        exclusions.append(normalized_exclusion)

    sampler_raw = raw.get("sampler")
    if not isinstance(sampler_raw, Mapping):
        raise ValueError("logging_support sampler must be a JSON object")
    if sampler_raw.get("shadow_only") is True or sampler_raw.get("rehearsal_only") is True:
        raise ValueError("shadow-only or rehearsal-only sampler support is never ledger eligible")
    if "ledger_eligible" in sampler_raw and sampler_raw.get("ledger_eligible") is not True:
        raise ValueError("sampler explicitly marked ledger_eligible=false is rejected")
    sampler_name = _text(sampler_raw.get("name"), "sampler name", limit=80)
    sampler_version = _text(sampler_raw.get("version"), "sampler version", limit=80)
    assignment_unit = _text(sampler_raw.get("assignment_unit") or "route", "assignment unit", limit=80)
    exploration_raw = sampler_raw.get("exploration_rate", 0.0)
    if isinstance(exploration_raw, bool):
        raise ValueError("sampler exploration_rate must be a finite number between 0 and 1")
    try:
        exploration_rate = float(exploration_raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("sampler exploration_rate must be a finite number between 0 and 1") from exc
    if not math.isfinite(exploration_rate) or not 0.0 <= exploration_rate <= 1.0:
        raise ValueError("sampler exploration_rate must be a finite number between 0 and 1")
    commitment_raw = sampler_raw.get("assignment_commitment")
    assignment_commitment = str(commitment_raw or "").strip() or None
    if assignment_commitment is not None and len(assignment_commitment) > 240:
        raise ValueError("sampler assignment_commitment must be at most 240 characters")
    if assignment_commitment is not None and assignment_commitment.lower().startswith(
        ("shadow:", "shadow-v1:", "route-study-shadow", "route-shadow:")
    ):
        raise ValueError("shadow assignment commitments cannot support executed ledger writes")
    if assignment_commitment is not None and not _EXECUTED_ASSIGNMENT_COMMITMENT_RE.fullmatch(
        assignment_commitment
    ):
        raise ValueError(
            "sampler assignment_commitment must use the closed "
            f"{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:<sha256> namespace"
        )

    if decision_type == "deterministic":
        if len(positive_actions) != 1 or exploration_rate != 0.0:
            raise ValueError("deterministic logging_support requires one positive action and zero exploration")
    else:
        if len(positive_actions) < 2:
            raise ValueError("randomized logging_support requires at least two positive actions")
        if exploration_rate <= 0.0:
            raise ValueError("randomized logging_support requires positive exploration_rate")
        if assignment_commitment is None:
            raise ValueError("randomized logging_support requires an assignment commitment")

    sampler = {
        "name": sampler_name,
        "version": sampler_version,
        "exploration_rate": exploration_rate,
        "assignment_unit": assignment_unit,
        "assignment_commitment": assignment_commitment,
    }
    envelope = {
        "schema_version": schema_version,
        "decision_type": decision_type,
        "probability_stage": probability_stage,
        "sampler": sampler,
        "candidates": candidates,
        "exclusions": exclusions,
        "chosen_probability": float(probabilities[chosen_mode]),
    }
    candidate_payload = {
        "schema_version": schema_version,
        "candidates": candidates,
        "exclusions": exclusions,
    }
    distribution_payload = {
        "schema_version": schema_version,
        "decision_type": decision_type,
        "probability_stage": probability_stage,
        "sampler": sampler,
        "eligible_modes": list(eligible_modes),
        "action_probabilities": dict(probabilities),
    }
    envelope["candidate_set_hash"] = _domain_hash(
        _CANDIDATE_HASH_DOMAIN, candidate_payload, "candidate support envelope"
    )
    envelope["distribution_hash"] = _domain_hash(
        _DISTRIBUTION_HASH_DOMAIN, distribution_payload, "logging distribution"
    )
    return envelope


def _utc(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def build_logging_support_envelope(
    logging_support: Optional[Mapping[str, Any]],
    *,
    eligible_modes: Sequence[str],
    action_probabilities: Mapping[str, Any],
    chosen_mode: str,
) -> Dict[str, Any]:
    """Build the same canonical support envelope used by durable writes."""

    modes = _eligible_modes(eligible_modes)
    cooked_chosen = _text(chosen_mode, "chosen_mode", limit=80)
    if cooked_chosen not in modes:
        raise ValueError("chosen_mode must be present in eligible_modes")
    probabilities = _probabilities(action_probabilities, modes, cooked_chosen)
    return _normalize_logging_support(
        logging_support,
        eligible_modes=modes,
        probabilities=probabilities,
        chosen_mode=cooked_chosen,
    )


def build_route_outcome_contracts(
    outcome_contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    precommitted: bool = True,
    commitment_source: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Return the canonical, complete Route Outcome Contract v1 set.

    Callers may omit the set entirely to use conservative built-in definitions,
    or supply exactly the four supported outcome names with per-outcome
    definition, observation-policy, and maturity overrides.  The builder never
    accepts additional outcomes: changing an outcome universe requires a new
    schema version rather than silently changing an existing experiment.
    """

    if not isinstance(precommitted, bool):
        raise ValueError("precommitted must be a boolean")
    if outcome_contracts is not None and not isinstance(outcome_contracts, Mapping):
        raise ValueError("outcome_contracts must be a JSON object")
    if outcome_contracts is not None and set(outcome_contracts) != set(OUTCOME_NAMES):
        raise ValueError("outcome_contracts must contain exactly the four canonical outcome names")

    if commitment_source is None:
        supplied_sources = {
            str(contract.get("commitment_source") or "").strip()
            for contract in (outcome_contracts or {}).values()
            if isinstance(contract, Mapping) and contract.get("commitment_source")
        }
        if len(supplied_sources) == 1:
            commitment_source = supplied_sources.pop()
        elif supplied_sources:
            raise ValueError("outcome contracts must use one commitment_source")
        else:
            commitment_source = (
                "legacy_posthoc"
                if not precommitted
                else ("caller" if outcome_contracts is not None else "safe_default")
            )
    cooked_source = _text(commitment_source, "commitment_source", limit=80)
    if not precommitted and cooked_source != "legacy_posthoc":
        raise ValueError("non-precommitted outcome contracts must be legacy_posthoc")
    if precommitted and cooked_source == "legacy_posthoc":
        raise ValueError("precommitted outcome contracts cannot be legacy_posthoc")

    contracts: Dict[str, Dict[str, Any]] = {}
    for outcome_name in OUTCOME_NAMES:
        defaults = _OUTCOME_CONTRACT_DEFAULTS[outcome_name]
        raw = (outcome_contracts or {}).get(outcome_name) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"outcome contract {outcome_name} must be a JSON object")

        schema_version = str(raw.get("schema_version") or OUTCOME_CONTRACT_SCHEMA_VERSION).strip()
        if schema_version != OUTCOME_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"outcome contract {outcome_name} has an unsupported schema_version")
        raw_name = str(raw.get("outcome_name") or outcome_name).strip()
        if raw_name != outcome_name:
            raise ValueError("outcome contract outcome_name must match its mapping key")
        maturity_basis = str(raw.get("maturity_basis") or "decision_started_at").strip()
        if maturity_basis != "decision_started_at":
            raise ValueError("outcome contract maturity_basis must be decision_started_at")

        definition_version = _text(
            raw.get("outcome_definition_version")
            or defaults["outcome_definition_version"],
            f"{outcome_name} outcome_definition_version",
            limit=120,
        )
        observation_policy_id = _text(
            raw.get("observation_policy_id") or defaults["observation_policy_id"],
            f"{outcome_name} observation_policy_id",
            limit=120,
        )
        observation_policy_version = _text(
            raw.get("observation_policy_version")
            or defaults["observation_policy_version"],
            f"{outcome_name} observation_policy_version",
            limit=120,
        )
        value_type = str(raw.get("value_type") or defaults["value_type"]).strip()
        unit = str(raw.get("unit") or defaults["unit"]).strip()
        if value_type != defaults["value_type"] or unit != defaults["unit"]:
            raise ValueError(f"outcome contract {outcome_name} cannot change value_type or unit")
        if "precommitted" in raw and raw.get("precommitted") is not precommitted:
            raise ValueError(f"outcome contract {outcome_name} precommitted flag conflicts with the write phase")
        raw_source = str(raw.get("commitment_source") or cooked_source).strip()
        if raw_source != cooked_source:
            raise ValueError("outcome contracts must use one commitment_source")

        delay_raw = raw.get("maturity_delay_seconds", defaults["maturity_delay_seconds"])
        if isinstance(delay_raw, bool):
            raise ValueError("maturity_delay_seconds must be a finite non-negative number")
        try:
            maturity_delay_seconds = float(delay_raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("maturity_delay_seconds must be a finite non-negative number") from exc
        if not math.isfinite(maturity_delay_seconds) or maturity_delay_seconds < 0.0:
            raise ValueError("maturity_delay_seconds must be a finite non-negative number")

        canonical: Dict[str, Any] = {
            "schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
            "outcome_name": outcome_name,
            "outcome_definition_version": definition_version,
            "observation_policy_id": observation_policy_id,
            "observation_policy_version": observation_policy_version,
            "value_type": value_type,
            "unit": unit,
            "maturity_delay_seconds": maturity_delay_seconds,
            "maturity_basis": "decision_started_at",
            "precommitted": precommitted,
            "commitment_source": cooked_source,
        }
        contract_hash = _domain_hash(
            _OUTCOME_CONTRACT_HASH_DOMAIN,
            canonical,
            f"{outcome_name} outcome contract",
        )
        supplied_hash = raw.get("contract_hash")
        if supplied_hash is not None and str(supplied_hash).strip() != contract_hash:
            raise ValueError(f"outcome contract {outcome_name} contract_hash does not match its canonical fields")
        canonical["contract_hash"] = contract_hash
        contracts[outcome_name] = canonical
    return contracts


def _finite_actual_metric(actual_economics: Mapping[str, Any], key: str) -> Optional[float]:
    """Read current flattened economics and the older ``actual`` nesting."""

    raw = actual_economics.get(key)
    if raw is None:
        nested = actual_economics.get("actual")
        if isinstance(nested, Mapping):
            raw = nested.get(key)
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value >= 0.0 else None


def _quality_observation(
    feedback: Mapping[str, Any],
    *,
    revision: int,
) -> Tuple[str, Optional[Any], Dict[str, Any]]:
    """Project only an explicit quality signal; never infer one from absence."""

    intent = str(feedback.get("feedback_intent") or feedback.get("intent") or "").strip().lower()
    metadata: Dict[str, Any] = {
        "feedback_revision": int(revision),
        "feedback_intent": intent or None,
        "quality_signal_source": None,
        "raw_rating": None,
    }
    non_quality_intents = {
        "cost",
        "latency",
        "lower_cost",
        "faster",
        "too_costly",
        "too_slow",
    }
    if intent in non_quality_intents or str(feedback.get("observation_status") or "").strip().lower() == "not_observed":
        return "not_observed", None, metadata

    axes = feedback.get("feedback_axes")
    if isinstance(axes, Mapping):
        raw_quality = axes.get("quality")
        if not isinstance(raw_quality, bool) and raw_quality is not None:
            try:
                quality = float(raw_quality)
            except (TypeError, ValueError, OverflowError):
                quality = math.nan
            if math.isfinite(quality) and -1.0 <= quality <= 1.0:
                metadata["quality_signal_source"] = "feedback_axes.quality"
                return "observed", quality, metadata

    rating = str(feedback.get("rating") or "").strip().lower()
    scores = {"up": 1.0, "down": -1.0}
    if rating in scores:
        metadata["quality_signal_source"] = "rating"
        metadata["raw_rating"] = rating
        return "observed", scores[rating], metadata
    return "not_observed", None, metadata


def _verify_decision_record_fingerprint(
    *,
    policy_name: str,
    policy_version: str,
    policy_schema_version: str,
    decision_context: Any,
    eligible_modes: Any,
    action_probabilities: Mapping[str, float],
    chosen_mode: str,
    logging_support: Mapping[str, Any],
    support_candidate_set_hash: Optional[str],
    support_distribution_hash: Optional[str],
) -> Tuple[Optional[str], bool, str]:
    """Fail closed when a durable decision no longer matches its fingerprint."""

    raw_fingerprint = logging_support.get("decision_record_fingerprint")
    stored_fingerprint = (
        str(raw_fingerprint).strip()
        if isinstance(raw_fingerprint, str) and raw_fingerprint.strip()
        else None
    )
    if stored_fingerprint is None:
        reason = (
            "legacy_unverifiable"
            if logging_support.get("migration_source") == "ledger_schema_v1"
            else "missing_unverifiable"
        )
        return None, False, reason

    fingerprint_schema = logging_support.get(
        "decision_record_fingerprint_schema_version"
    )
    if fingerprint_schema != DECISION_FINGERPRINT_SCHEMA_VERSION:
        return stored_fingerprint, False, "unsupported_schema"
    if len(stored_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in stored_fingerprint
    ):
        return stored_fingerprint, False, "malformed_fingerprint"
    if not isinstance(decision_context, Mapping):
        return stored_fingerprint, False, "invalid_decision_context"
    if not isinstance(eligible_modes, Sequence) or isinstance(
        eligible_modes, (str, bytes)
    ):
        return stored_fingerprint, False, "invalid_eligible_modes"

    envelope_candidate_hash = logging_support.get("candidate_set_hash")
    envelope_distribution_hash = logging_support.get("distribution_hash")
    if (
        not isinstance(envelope_candidate_hash, str)
        or not isinstance(envelope_distribution_hash, str)
        or envelope_candidate_hash != support_candidate_set_hash
        or envelope_distribution_hash != support_distribution_hash
    ):
        return stored_fingerprint, False, "support_projection_mismatch"

    try:
        expected_fingerprint = _decision_record_fingerprint(
            policy_name=policy_name,
            policy_version=policy_version,
            policy_schema_version=policy_schema_version,
            decision_context=decision_context,
            eligible_modes=eligible_modes,
            action_probabilities=action_probabilities,
            chosen_mode=chosen_mode,
            candidate_set_hash=envelope_candidate_hash,
            distribution_hash=envelope_distribution_hash,
        )
    except (TypeError, ValueError):
        return stored_fingerprint, False, "invalid_canonical_record"
    if not hmac.compare_digest(stored_fingerprint, expected_fingerprint):
        return stored_fingerprint, False, "fingerprint_mismatch"
    return stored_fingerprint, True, "verified"


class RoutePolicyLedger:
    """SQLite-backed route ledger using one durable transaction per phase."""

    def __init__(self, db_path: Any, *, timeout_seconds: float = 30.0) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) == ":memory:":
            raise ValueError("RoutePolicyLedger requires a durable filesystem path")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            current_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current_version not in (0, 1, 2, LEDGER_SCHEMA_VERSION):
                raise RoutePolicyLedgerError(
                    f"unsupported route ledger schema {current_version}; expected 1, 2, or {LEDGER_SCHEMA_VERSION}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_counters (
                    session_hash TEXT PRIMARY KEY,
                    next_sequence INTEGER NOT NULL CHECK (next_sequence >= 1)
                );

                CREATE TABLE IF NOT EXISTS route_decisions (
                    route_id TEXT PRIMARY KEY,
                    session_hash TEXT NOT NULL,
                    session_sequence INTEGER NOT NULL CHECK (session_sequence >= 1),
                    ledger_schema_version INTEGER NOT NULL,
                    policy_name TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    policy_schema_version TEXT NOT NULL,
                    decision_context_json TEXT NOT NULL,
                    eligible_modes_json TEXT NOT NULL,
                    action_probabilities_json TEXT NOT NULL,
                    chosen_mode TEXT NOT NULL,
                    executed_mode TEXT,
                    estimated_economics_json TEXT NOT NULL,
                    actual_economics_json TEXT,
                    status TEXT NOT NULL CHECK (status IN ('inflight', 'completed', 'failed')),
                    success INTEGER CHECK (success IS NULL OR success IN (0, 1)),
                    error_category TEXT,
                    error_message TEXT,
                    started_at REAL NOT NULL,
                    completed_at REAL,
                    UNIQUE (session_hash, session_sequence)
                );

                CREATE INDEX IF NOT EXISTS idx_route_decisions_session_sequence
                    ON route_decisions(session_hash, session_sequence DESC);
                CREATE INDEX IF NOT EXISTS idx_route_decisions_status
                    ON route_decisions(status, started_at);

                CREATE TABLE IF NOT EXISTS route_feedback_revisions (
                    route_id TEXT NOT NULL REFERENCES route_decisions(route_id) ON DELETE CASCADE,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    idempotency_key TEXT NOT NULL,
                    feedback_json TEXT NOT NULL,
                    recorded_at REAL NOT NULL,
                    PRIMARY KEY (route_id, revision),
                    UNIQUE (route_id, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS idx_route_feedback_latest
                    ON route_feedback_revisions(route_id, revision DESC);

                CREATE TABLE IF NOT EXISTS route_decision_support (
                    route_id TEXT PRIMARY KEY
                        REFERENCES route_decisions(route_id) ON DELETE CASCADE,
                    support_schema_version TEXT NOT NULL,
                    decision_type TEXT NOT NULL
                        CHECK (decision_type IN ('deterministic', 'randomized', 'legacy_unknown')),
                    probability_stage TEXT NOT NULL,
                    sampler_name TEXT NOT NULL,
                    sampler_version TEXT NOT NULL,
                    exploration_rate REAL NOT NULL
                        CHECK (exploration_rate >= 0.0 AND exploration_rate <= 1.0),
                    chosen_probability REAL
                        CHECK (chosen_probability IS NULL OR (chosen_probability > 0.0 AND chosen_probability <= 1.0)),
                    candidate_set_hash TEXT NOT NULL,
                    distribution_hash TEXT NOT NULL,
                    logging_envelope_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_route_support_decision_type
                    ON route_decision_support(decision_type, probability_stage);

                CREATE TABLE IF NOT EXISTS route_outcome_contracts (
                    route_id TEXT NOT NULL
                        REFERENCES route_decisions(route_id) ON DELETE CASCADE,
                    outcome_name TEXT NOT NULL
                        CHECK (outcome_name IN ('route_success', 'user_quality_rating', 'cost', 'latency')),
                    contract_schema_version TEXT NOT NULL,
                    outcome_definition_version TEXT NOT NULL,
                    observation_policy_id TEXT NOT NULL,
                    observation_policy_version TEXT NOT NULL,
                    value_type TEXT NOT NULL,
                    unit TEXT NOT NULL,
                    maturity_delay_seconds REAL NOT NULL
                        CHECK (maturity_delay_seconds >= 0.0),
                    maturity_basis TEXT NOT NULL
                        CHECK (maturity_basis = 'decision_started_at'),
                    precommitted INTEGER NOT NULL
                        CHECK (precommitted IN (0, 1)),
                    commitment_source TEXT NOT NULL,
                    contract_hash TEXT NOT NULL,
                    contract_json TEXT NOT NULL,
                    committed_at REAL NOT NULL,
                    PRIMARY KEY (route_id, outcome_name)
                );

                CREATE INDEX IF NOT EXISTS idx_route_outcome_contracts_name
                    ON route_outcome_contracts(outcome_name, precommitted);

                CREATE TABLE IF NOT EXISTS route_outcome_observation_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_id TEXT NOT NULL,
                    outcome_name TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    observation_status TEXT NOT NULL
                        CHECK (observation_status IN ('observed', 'not_observed')),
                    value_json TEXT,
                    event_source TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    recorded_at REAL NOT NULL,
                    FOREIGN KEY (route_id, outcome_name)
                        REFERENCES route_outcome_contracts(route_id, outcome_name) ON DELETE CASCADE,
                    UNIQUE (route_id, outcome_name, event_key),
                    CHECK (
                        (observation_status = 'observed' AND value_json IS NOT NULL)
                        OR (observation_status = 'not_observed' AND value_json IS NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_route_outcome_events_route_time
                    ON route_outcome_observation_events(route_id, recorded_at, event_id);
                CREATE INDEX IF NOT EXISTS idx_route_outcome_events_name_status
                    ON route_outcome_observation_events(outcome_name, observation_status, recorded_at);

                CREATE TRIGGER IF NOT EXISTS route_outcome_contracts_no_update
                BEFORE UPDATE ON route_outcome_contracts
                BEGIN
                    SELECT RAISE(ABORT, 'route outcome contracts are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS route_outcome_contracts_no_delete
                BEFORE DELETE ON route_outcome_contracts
                BEGIN
                    SELECT RAISE(ABORT, 'route outcome contracts are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS route_outcome_events_no_update
                BEFORE UPDATE ON route_outcome_observation_events
                BEGIN
                    SELECT RAISE(ABORT, 'route outcome observation events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS route_outcome_events_no_delete
                BEFORE DELETE ON route_outcome_observation_events
                BEGIN
                    SELECT RAISE(ABORT, 'route outcome observation events are append-only');
                END;
                """
            )
            self._backfill_support_rows(connection)
            self._backfill_outcome_rows(connection)
            connection.execute(
                "INSERT OR REPLACE INTO ledger_metadata(key, value) VALUES ('schema_version', ?)",
                (str(LEDGER_SCHEMA_VERSION),),
            )
            connection.execute(f"PRAGMA user_version = {LEDGER_SCHEMA_VERSION}")
        finally:
            connection.close()

    @staticmethod
    def _backfill_support_rows(connection: sqlite3.Connection) -> None:
        """Idempotently migrate v1 decisions into an explicitly non-causal envelope."""

        rows = connection.execute(
            """
            SELECT d.route_id, d.eligible_modes_json, d.action_probabilities_json, d.chosen_mode
            FROM route_decisions d
            LEFT JOIN route_decision_support s ON s.route_id = d.route_id
            WHERE s.route_id IS NULL
            """
        ).fetchall()
        for row in rows:
            route_id = str(row["route_id"])
            chosen_mode = str(row["chosen_mode"])
            try:
                modes = [str(mode) for mode in json.loads(str(row["eligible_modes_json"]))]
            except (TypeError, ValueError, json.JSONDecodeError):
                modes = []

            probabilities = _legacy_probability_mapping(row["action_probabilities_json"])
            valid_vector = (
                bool(modes)
                and bool(probabilities)
                and set(probabilities) == set(modes)
                and all(0.0 <= value <= 1.0 for value in probabilities.values())
                and math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6)
                and probabilities.get(chosen_mode, 0.0) > 0.0
            )
            positive_actions = (
                [mode for mode in modes if probabilities.get(mode, 0.0) > 0.0]
                if valid_vector
                else []
            )

            deterministic = valid_vector and len(positive_actions) == 1
            decision_type = "deterministic" if deterministic else "legacy_unknown"
            chosen_probability = probabilities.get(chosen_mode) if valid_vector else None
            candidates = [{"action": mode, "legacy_migrated": True} for mode in modes]
            sampler = {
                "name": "legacy_v1",
                "version": "1",
                "exploration_rate": 0.0,
                "assignment_unit": "route",
                "assignment_commitment": None,
            }
            envelope = {
                "schema_version": SUPPORT_SCHEMA_VERSION,
                "decision_type": decision_type,
                "probability_stage": "post_filter",
                "sampler": sampler,
                "candidates": candidates,
                "exclusions": [],
                "chosen_probability": chosen_probability,
                "migration_source": "ledger_schema_v1",
                "decision_record_fingerprint_schema_version": None,
                "decision_record_fingerprint": None,
            }
            candidate_hash = _domain_hash(
                _CANDIDATE_HASH_DOMAIN,
                {
                    "schema_version": SUPPORT_SCHEMA_VERSION,
                    "candidates": candidates,
                    "exclusions": [],
                },
                "legacy candidate support envelope",
            )
            distribution_hash = _domain_hash(
                _DISTRIBUTION_HASH_DOMAIN,
                {
                    "schema_version": SUPPORT_SCHEMA_VERSION,
                    "decision_type": decision_type,
                    "probability_stage": "post_filter",
                    "sampler": sampler,
                    "eligible_modes": modes,
                    "action_probabilities": probabilities,
                },
                "legacy logging distribution",
            )
            envelope.update(
                {
                    "candidate_set_hash": candidate_hash,
                    "distribution_hash": distribution_hash,
                }
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO route_decision_support(
                    route_id, support_schema_version, decision_type, probability_stage,
                    sampler_name, sampler_version, exploration_rate, chosen_probability,
                    candidate_set_hash, distribution_hash, logging_envelope_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    route_id,
                    SUPPORT_SCHEMA_VERSION,
                    decision_type,
                    "post_filter",
                    sampler["name"],
                    sampler["version"],
                    0.0,
                    chosen_probability,
                    candidate_hash,
                    distribution_hash,
                    _canonical_json(envelope, "legacy logging envelope"),
                ),
            )

    @staticmethod
    def _insert_outcome_contract_rows(
        connection: sqlite3.Connection,
        route_id: str,
        contracts: Mapping[str, Mapping[str, Any]],
        *,
        committed_at: float,
    ) -> None:
        for outcome_name in OUTCOME_NAMES:
            contract = dict(contracts[outcome_name])
            connection.execute(
                """
                INSERT OR IGNORE INTO route_outcome_contracts(
                    route_id, outcome_name, contract_schema_version,
                    outcome_definition_version, observation_policy_id,
                    observation_policy_version, value_type, unit,
                    maturity_delay_seconds, maturity_basis, precommitted,
                    commitment_source, contract_hash, contract_json, committed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    route_id,
                    outcome_name,
                    contract["schema_version"],
                    contract["outcome_definition_version"],
                    contract["observation_policy_id"],
                    contract["observation_policy_version"],
                    contract["value_type"],
                    contract["unit"],
                    contract["maturity_delay_seconds"],
                    contract["maturity_basis"],
                    1 if contract["precommitted"] else 0,
                    contract["commitment_source"],
                    contract["contract_hash"],
                    _canonical_json(contract, f"{outcome_name} outcome contract"),
                    float(committed_at),
                ),
            )

    @staticmethod
    def _insert_outcome_event(
        connection: sqlite3.Connection,
        *,
        route_id: str,
        outcome_name: str,
        event_key: str,
        observation_status: str,
        value: Optional[Any],
        event_source: str,
        metadata: Optional[Mapping[str, Any]],
        observed_at: float,
        recorded_at: float,
    ) -> None:
        if outcome_name not in OUTCOME_NAMES:
            raise ValueError("unsupported outcome_name")
        if observation_status not in {"observed", "not_observed"}:
            raise ValueError("observation_status must be observed or not_observed")
        if observation_status == "observed" and value is None:
            raise ValueError("observed outcome events require a value")
        if observation_status == "not_observed" and value is not None:
            raise ValueError("not_observed outcome events cannot carry a value")
        value_json = (
            _canonical_json(value, f"{outcome_name} observed value")
            if value is not None
            else None
        )
        metadata_json = _json_mapping(metadata, "outcome event metadata")
        connection.execute(
            """
            INSERT OR IGNORE INTO route_outcome_observation_events(
                route_id, outcome_name, event_key, observation_status,
                value_json, event_source, metadata_json, observed_at, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                route_id,
                outcome_name,
                _text(event_key, "event_key", limit=240),
                observation_status,
                value_json,
                _text(event_source, "event_source", limit=80),
                metadata_json,
                float(observed_at),
                float(recorded_at),
            ),
        )

    @classmethod
    def _backfill_outcome_rows(cls, connection: sqlite3.Connection) -> None:
        """Idempotently mark old lifecycle data as descriptive, post-hoc evidence."""

        migrated_at = float(time.time())
        decisions = connection.execute(
            """
            SELECT d.*
            FROM route_decisions d
            LEFT JOIN route_outcome_contracts c ON c.route_id = d.route_id
            GROUP BY d.route_id
            HAVING COUNT(c.outcome_name) < 4
            """
        ).fetchall()
        for row in decisions:
            route_id = str(row["route_id"])
            contracts = build_route_outcome_contracts(
                precommitted=False,
                commitment_source="legacy_posthoc",
            )
            cls._insert_outcome_contract_rows(
                connection,
                route_id,
                contracts,
                committed_at=migrated_at,
            )

        # Events are separately idempotent so a crash between contract and
        # event backfill can be completed on the next open.
        legacy_rows = connection.execute(
            """
            SELECT d.*
            FROM route_decisions d
            JOIN route_outcome_contracts c ON c.route_id = d.route_id
            WHERE c.outcome_name = 'route_success'
              AND c.precommitted = 0
              AND d.status != 'inflight'
            """
        ).fetchall()
        for row in legacy_rows:
            route_id = str(row["route_id"])
            observed_at = float(row["completed_at"] or row["started_at"])
            legacy_version = int(row["ledger_schema_version"] or 1)
            metadata = {
                "migration_source": f"ledger_schema_v{legacy_version}",
                "descriptive_only": True,
            }
            success_value = (
                bool(row["success"])
                if row["success"] is not None
                else None
            )
            cls._insert_outcome_event(
                connection,
                route_id=route_id,
                outcome_name="route_success",
                event_key="legacy_completion:route_success",
                observation_status="observed" if success_value is not None else "not_observed",
                value=success_value,
                event_source="legacy_posthoc",
                metadata=metadata,
                observed_at=observed_at,
                recorded_at=migrated_at,
            )
            try:
                actual = json.loads(str(row["actual_economics_json"] or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                actual = {}
            if not isinstance(actual, Mapping):
                actual = {}
            for outcome_name, metric_name in (("cost", "cost_units"), ("latency", "elapsed_ms")):
                metric = _finite_actual_metric(actual, metric_name)
                cls._insert_outcome_event(
                    connection,
                    route_id=route_id,
                    outcome_name=outcome_name,
                    event_key=f"legacy_completion:{outcome_name}",
                    observation_status="observed" if metric is not None else "not_observed",
                    value=metric,
                    event_source="legacy_posthoc",
                    metadata=metadata,
                    observed_at=observed_at,
                    recorded_at=migrated_at,
                )

        feedback_rows = connection.execute(
            """
            SELECT f.route_id, f.revision, f.feedback_json, f.recorded_at,
                   d.ledger_schema_version
            FROM route_feedback_revisions f
            JOIN route_decisions d ON d.route_id = f.route_id
            JOIN route_outcome_contracts c
              ON c.route_id = f.route_id
             AND c.outcome_name = 'user_quality_rating'
             AND c.precommitted = 0
            ORDER BY f.route_id, f.revision
            """
        ).fetchall()
        for row in feedback_rows:
            try:
                feedback = json.loads(str(row["feedback_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                feedback = {}
            if not isinstance(feedback, Mapping):
                feedback = {}
            revision = int(row["revision"])
            status, value, metadata = _quality_observation(feedback, revision=revision)
            metadata.update(
                {
                    "migration_source": f"ledger_schema_v{int(row['ledger_schema_version'] or 1)}",
                    "descriptive_only": True,
                }
            )
            cls._insert_outcome_event(
                connection,
                route_id=str(row["route_id"]),
                outcome_name="user_quality_rating",
                event_key=f"legacy_feedback_revision:{revision}",
                observation_status=status,
                value=value,
                event_source="legacy_posthoc",
                metadata=metadata,
                observed_at=float(row["recorded_at"]),
                recorded_at=migrated_at,
            )

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        """Keep a multi-query projection on one SQLite snapshot."""

        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin_decision(
        self,
        *,
        session_id: str,
        policy_name: str,
        policy_version: str,
        policy_schema_version: str,
        decision_context: Mapping[str, Any],
        eligible_modes: Sequence[str],
        chosen_mode: str,
        action_probabilities: Optional[Mapping[str, Any]] = None,
        logging_support: Optional[Mapping[str, Any]] = None,
        estimated_economics: Optional[Mapping[str, Any]] = None,
        outcome_contracts: Optional[Mapping[str, Mapping[str, Any]]] = None,
        route_id: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Commit an inflight decision before executing its chosen route."""

        cooked_route_id = _route_id(route_id)
        session_hash = hash_session_identity(session_id)
        cooked_policy_name = _text(policy_name, "policy_name", limit=120)
        cooked_policy_version = _text(policy_version, "policy_version", limit=120)
        cooked_policy_schema = _text(policy_schema_version, "policy_schema_version", limit=120)
        context_json = _json_mapping(decision_context, "decision_context")
        modes = _eligible_modes(eligible_modes)
        cooked_chosen = _text(chosen_mode, "chosen_mode", limit=80)
        if cooked_chosen not in modes:
            raise ValueError("chosen_mode must be present in eligible_modes")
        probabilities = _probabilities(action_probabilities, modes, cooked_chosen)
        support = _normalize_logging_support(
            logging_support,
            eligible_modes=modes,
            probabilities=probabilities,
            chosen_mode=cooked_chosen,
        )
        support["decision_record_fingerprint_schema_version"] = (
            DECISION_FINGERPRINT_SCHEMA_VERSION
        )
        support["decision_record_fingerprint"] = _decision_record_fingerprint(
            policy_name=cooked_policy_name,
            policy_version=cooked_policy_version,
            policy_schema_version=cooked_policy_schema,
            decision_context=json.loads(context_json),
            eligible_modes=modes,
            action_probabilities=probabilities,
            chosen_mode=cooked_chosen,
            candidate_set_hash=support["candidate_set_hash"],
            distribution_hash=support["distribution_hash"],
        )
        probabilities_json = json.dumps(probabilities, sort_keys=True, separators=(",", ":"), allow_nan=False)
        support_json = _canonical_json(support, "logging_support")
        modes_json = json.dumps(modes, separators=(",", ":"), ensure_ascii=False)
        estimates_json = _json_mapping(estimated_economics, "estimated_economics")
        contracts = build_route_outcome_contracts(
            outcome_contracts,
            precommitted=True,
        )
        started_at = float(time.time())

        try:
            with self._write_transaction() as connection:
                connection.execute(
                    "INSERT OR IGNORE INTO session_counters(session_hash, next_sequence) VALUES (?, 1)",
                    (session_hash,),
                )
                sequence = int(
                    connection.execute(
                        "SELECT next_sequence FROM session_counters WHERE session_hash = ?",
                        (session_hash,),
                    ).fetchone()[0]
                )
                connection.execute(
                    "UPDATE session_counters SET next_sequence = ? WHERE session_hash = ?",
                    (sequence + 1, session_hash),
                )
                connection.execute(
                    """
                    INSERT INTO route_decisions(
                        route_id, session_hash, session_sequence, ledger_schema_version,
                        policy_name, policy_version, policy_schema_version,
                        decision_context_json, eligible_modes_json, action_probabilities_json,
                        chosen_mode, estimated_economics_json, status, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'inflight', ?)
                    """,
                    (
                        cooked_route_id,
                        session_hash,
                        sequence,
                        LEDGER_SCHEMA_VERSION,
                        cooked_policy_name,
                        cooked_policy_version,
                        cooked_policy_schema,
                        context_json,
                        modes_json,
                        probabilities_json,
                        cooked_chosen,
                        estimates_json,
                        started_at,
                    ),
                )
                sampler = support["sampler"]
                connection.execute(
                    """
                    INSERT INTO route_decision_support(
                        route_id, support_schema_version, decision_type, probability_stage,
                        sampler_name, sampler_version, exploration_rate, chosen_probability,
                        candidate_set_hash, distribution_hash, logging_envelope_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cooked_route_id,
                        support["schema_version"],
                        support["decision_type"],
                        support["probability_stage"],
                        sampler["name"],
                        sampler["version"],
                        sampler["exploration_rate"],
                        support["chosen_probability"],
                        support["candidate_set_hash"],
                        support["distribution_hash"],
                        support_json,
                    ),
                )
                self._insert_outcome_contract_rows(
                    connection,
                    cooked_route_id,
                    contracts,
                    committed_at=started_at,
                )
        except sqlite3.IntegrityError as exc:
            raise LedgerConflictError(f"route_id already exists: {cooked_route_id}") from exc
        return self.get_decision(cooked_route_id)

    def complete_decision(
        self,
        route_id: Any,
        *,
        success: bool,
        executed_mode: Optional[str] = None,
        actual_economics: Optional[Mapping[str, Any]] = None,
        error_category: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Complete an inflight decision, idempotently for an identical retry."""

        cooked_route_id = _existing_route_id(route_id)
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        cooked_executed = str(executed_mode or "").strip() or None
        if cooked_executed is not None and len(cooked_executed) > 80:
            raise ValueError("executed_mode must be at most 80 characters")
        actual_json = _json_mapping(actual_economics, "actual_economics")
        actual_mapping = json.loads(actual_json)
        cooked_error_category = str(error_category or "").strip() or None
        cooked_error_message = str(error_message or "").strip() or None
        if success and (cooked_error_category or cooked_error_message):
            raise ValueError("successful decisions cannot have error metadata")
        if not success and not cooked_error_category:
            raise ValueError("failed decisions require error_category")
        if cooked_error_category is not None and len(cooked_error_category) > 120:
            raise ValueError("error_category must be at most 120 characters")
        if cooked_error_message is not None and len(cooked_error_message) > 2000:
            raise ValueError("error_message must be at most 2000 characters")
        terminal_status = "completed" if success else "failed"
        completed_at = float(time.time())

        with self._write_transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM route_decisions WHERE route_id = ?", (cooked_route_id,)
            ).fetchone()
            if existing is None:
                raise DecisionNotFoundError(f"unknown route_id: {cooked_route_id}")
            if str(existing["status"]) != "inflight":
                same_completion = (
                    str(existing["status"]) == terminal_status
                    and bool(existing["success"]) is success
                    and existing["executed_mode"] == cooked_executed
                    and str(existing["actual_economics_json"] or "{}") == actual_json
                    and existing["error_category"] == cooked_error_category
                    and existing["error_message"] == cooked_error_message
                )
                if not same_completion:
                    raise LedgerConflictError(f"route decision already completed differently: {cooked_route_id}")
            else:
                connection.execute(
                    """
                    UPDATE route_decisions
                    SET status = ?, success = ?, executed_mode = ?, actual_economics_json = ?,
                        error_category = ?, error_message = ?, completed_at = ?
                    WHERE route_id = ? AND status = 'inflight'
                    """,
                    (
                        terminal_status,
                        1 if success else 0,
                        cooked_executed,
                        actual_json,
                        cooked_error_category,
                        cooked_error_message,
                        completed_at,
                        cooked_route_id,
                    ),
                )
                completion_metadata = {
                    "executed_mode": cooked_executed,
                    "terminal_status": terminal_status,
                    "error_category": cooked_error_category,
                }
                self._insert_outcome_event(
                    connection,
                    route_id=cooked_route_id,
                    outcome_name="route_success",
                    event_key="completion",
                    observation_status="observed",
                    value=success,
                    event_source="route_completion",
                    metadata=completion_metadata,
                    observed_at=completed_at,
                    recorded_at=completed_at,
                )
                for outcome_name, metric_name in (("cost", "cost_units"), ("latency", "elapsed_ms")):
                    metric = _finite_actual_metric(actual_mapping, metric_name)
                    self._insert_outcome_event(
                        connection,
                        route_id=cooked_route_id,
                        outcome_name=outcome_name,
                        event_key="completion",
                        observation_status="observed" if metric is not None else "not_observed",
                        value=metric,
                        event_source="route_completion",
                        metadata=completion_metadata,
                        observed_at=completed_at,
                        recorded_at=completed_at,
                    )
        return self.get_decision(cooked_route_id)

    def record_feedback(
        self,
        route_id: Any,
        feedback: Mapping[str, Any],
        *,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a feedback revision or return the identical prior revision."""

        cooked_route_id = _existing_route_id(route_id)
        feedback_json = _json_mapping(feedback, "feedback")
        explicit_key = idempotency_key is not None
        cooked_key = _text(idempotency_key, "idempotency_key", limit=240) if explicit_key else None
        recorded_at = float(time.time())
        idempotent = False

        with self._write_transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM route_decisions WHERE route_id = ?", (cooked_route_id,)
            ).fetchone()
            if exists is None:
                raise DecisionNotFoundError(f"unknown route_id: {cooked_route_id}")
            if explicit_key:
                prior = connection.execute(
                    """
                    SELECT revision, idempotency_key, feedback_json, recorded_at
                    FROM route_feedback_revisions
                    WHERE route_id = ? AND idempotency_key = ?
                    """,
                    (cooked_route_id, cooked_key),
                ).fetchone()
            else:
                # Content-derived compatibility is deliberately limited to an
                # adjacent identical retry.  Matching an older revision would
                # suppress a legitimate up -> down -> up sequence.
                prior = connection.execute(
                    """
                    SELECT revision, idempotency_key, feedback_json, recorded_at
                    FROM route_feedback_revisions
                    WHERE route_id = ? ORDER BY revision DESC LIMIT 1
                    """,
                    (cooked_route_id,),
                ).fetchone()
                if prior is not None and str(prior["feedback_json"]) != feedback_json:
                    prior = None
            if prior is not None:
                if explicit_key and str(prior["feedback_json"]) != feedback_json:
                    raise LedgerConflictError(
                        f"feedback idempotency key reused with different content for {cooked_route_id}"
                    )
                revision = int(prior["revision"])
                cooked_key = str(prior["idempotency_key"])
                recorded_at = float(prior["recorded_at"])
                idempotent = True
            else:
                prior_revision = connection.execute(
                    "SELECT COALESCE(MAX(revision), 0) FROM route_feedback_revisions WHERE route_id = ?",
                    (cooked_route_id,),
                ).fetchone()[0]
                revision = int(prior_revision) + 1
                if not explicit_key:
                    content_hash = hashlib.sha256(feedback_json.encode("utf-8")).hexdigest()
                    cooked_key = f"content-sha256:{content_hash}:revision:{revision}"
                connection.execute(
                    """
                    INSERT INTO route_feedback_revisions(
                        route_id, revision, idempotency_key, feedback_json, recorded_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (cooked_route_id, revision, cooked_key, feedback_json, recorded_at),
                )
                feedback_mapping = json.loads(feedback_json)
                observation_status, observation_value, observation_metadata = _quality_observation(
                    feedback_mapping,
                    revision=revision,
                )
                self._insert_outcome_event(
                    connection,
                    route_id=cooked_route_id,
                    outcome_name="user_quality_rating",
                    event_key=f"feedback_revision:{revision}",
                    observation_status=observation_status,
                    value=observation_value,
                    event_source="feedback_revision",
                    metadata=observation_metadata,
                    observed_at=recorded_at,
                    recorded_at=recorded_at,
                )
        return {
            "route_id": cooked_route_id,
            "revision": revision,
            "idempotency_key": cooked_key,
            "feedback": json.loads(feedback_json),
            "recorded_at": recorded_at,
            "recorded_at_utc": _utc(recorded_at),
            "idempotent": idempotent,
        }

    def feedback_history(self, route_id: Any) -> List[Dict[str, Any]]:
        cooked_route_id = _existing_route_id(route_id)
        connection = self._connect()
        try:
            exists = connection.execute(
                "SELECT 1 FROM route_decisions WHERE route_id = ?", (cooked_route_id,)
            ).fetchone()
            if exists is None:
                raise DecisionNotFoundError(f"unknown route_id: {cooked_route_id}")
            rows = connection.execute(
                """
                SELECT revision, idempotency_key, feedback_json, recorded_at
                FROM route_feedback_revisions
                WHERE route_id = ? ORDER BY revision ASC
                """,
                (cooked_route_id,),
            ).fetchall()
            return [self._feedback_row(cooked_route_id, row) for row in rows]
        finally:
            connection.close()

    @staticmethod
    def _feedback_row(route_id: str, row: sqlite3.Row) -> Dict[str, Any]:
        timestamp = float(row["recorded_at"])
        return {
            "route_id": route_id,
            "revision": int(row["revision"]),
            "idempotency_key": str(row["idempotency_key"]),
            "feedback": json.loads(str(row["feedback_json"])),
            "recorded_at": timestamp,
            "recorded_at_utc": _utc(timestamp),
        }

    @staticmethod
    def _outcome_contract_row(row: sqlite3.Row) -> Dict[str, Any]:
        canonical: Dict[str, Any] = {
            "schema_version": str(row["contract_schema_version"]),
            "outcome_name": str(row["outcome_name"]),
            "outcome_definition_version": str(row["outcome_definition_version"]),
            "observation_policy_id": str(row["observation_policy_id"]),
            "observation_policy_version": str(row["observation_policy_version"]),
            "value_type": str(row["value_type"]),
            "unit": str(row["unit"]),
            "maturity_delay_seconds": float(row["maturity_delay_seconds"]),
            "maturity_basis": str(row["maturity_basis"]),
            "precommitted": bool(row["precommitted"]),
            "commitment_source": str(row["commitment_source"]),
        }
        expected_hash = _domain_hash(
            _OUTCOME_CONTRACT_HASH_DOMAIN,
            canonical,
            f"{canonical['outcome_name']} outcome contract",
        )
        stored_hash = str(row["contract_hash"])
        if len(stored_hash) != 64 or any(character not in "0123456789abcdef" for character in stored_hash):
            hash_valid = False
            hash_reason = "malformed_hash"
        elif not hmac.compare_digest(stored_hash, expected_hash):
            hash_valid = False
            hash_reason = "hash_mismatch"
        else:
            try:
                raw = json.loads(str(row["contract_json"]))
            except (TypeError, ValueError, json.JSONDecodeError):
                raw = None
            projected = dict(canonical)
            projected["contract_hash"] = stored_hash
            try:
                projection_matches = bool(
                    isinstance(raw, Mapping)
                    and _canonical_json(dict(raw), "stored outcome contract")
                    == _canonical_json(projected, "projected outcome contract")
                )
            except ValueError:
                projection_matches = False
            if not projection_matches:
                hash_valid = False
                hash_reason = "contract_projection_mismatch"
            else:
                hash_valid = True
                hash_reason = "verified"
        contract = dict(canonical)
        contract["contract_hash"] = stored_hash
        try:
            committed_at = float(row["committed_at"])
        except (TypeError, ValueError, OverflowError):
            committed_at = math.nan
        contract.update(
            {
                "contract_hash_valid": hash_valid,
                "contract_hash_reason": hash_reason,
                "committed_at": committed_at,
                "committed_at_utc": _utc(committed_at) if math.isfinite(committed_at) else None,
            }
        )
        return contract

    @staticmethod
    def _outcome_event_row(row: sqlite3.Row) -> Dict[str, Any]:
        observed_at = float(row["observed_at"])
        recorded_at = float(row["recorded_at"])
        return {
            "event_id": int(row["event_id"]),
            "route_id": str(row["route_id"]),
            "outcome_name": str(row["outcome_name"]),
            "event_key": str(row["event_key"]),
            "observation_status": str(row["observation_status"]),
            "value": (
                json.loads(str(row["value_json"]))
                if row["value_json"] is not None
                else None
            ),
            "event_source": str(row["event_source"]),
            "metadata": json.loads(str(row["metadata_json"])),
            "observed_at": observed_at,
            "observed_at_utc": _utc(observed_at),
            "recorded_at": recorded_at,
            "recorded_at_utc": _utc(recorded_at),
        }

    def _decision_from_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        route_id = str(row["route_id"])
        support_row = connection.execute(
            "SELECT * FROM route_decision_support WHERE route_id = ?", (route_id,)
        ).fetchone()
        contract_cutoff = "AND committed_at <= ?" if as_of is not None else ""
        contract_parameters: Tuple[Any, ...] = (
            (route_id, float(as_of)) if as_of is not None else (route_id,)
        )
        contract_rows = connection.execute(
            f"""
            SELECT * FROM route_outcome_contracts
            WHERE route_id = ? {contract_cutoff} ORDER BY outcome_name
            """,
            contract_parameters,
        ).fetchall()
        outcome_contracts = {
            str(contract_row["outcome_name"]): self._outcome_contract_row(contract_row)
            for contract_row in contract_rows
        }
        event_parameters: Tuple[Any, ...] = (
            (route_id, float(as_of)) if as_of is not None else (route_id,)
        )
        event_cutoff = "AND recorded_at <= ?" if as_of is not None else ""
        event_rows = connection.execute(
            f"""
            SELECT * FROM route_outcome_observation_events
            WHERE route_id = ? {event_cutoff}
            ORDER BY recorded_at, event_id
            """,
            event_parameters,
        ).fetchall()
        outcome_events = [self._outcome_event_row(event_row) for event_row in event_rows]
        feedback_cutoff = "AND recorded_at <= ?" if as_of is not None else ""
        feedback_parameters: Tuple[Any, ...] = (
            (route_id, float(as_of), route_id)
            if as_of is not None
            else (route_id, route_id)
        )
        feedback_projection = connection.execute(
            f"""
            SELECT
                f.revision,
                f.idempotency_key,
                f.feedback_json,
                f.recorded_at,
                summary.revision_count
            FROM (
                SELECT COUNT(*) AS revision_count, MAX(revision) AS latest_revision
                FROM route_feedback_revisions
                WHERE route_id = ? {feedback_cutoff}
            ) AS summary
            LEFT JOIN route_feedback_revisions AS f
                ON f.route_id = ? AND f.revision = summary.latest_revision
            """,
            feedback_parameters,
        ).fetchone()
        revision_count = int(feedback_projection["revision_count"])
        latest_feedback = (
            self._feedback_row(route_id, feedback_projection)
            if feedback_projection["revision"] is not None
            else None
        )
        raw_logging_support = (
            json.loads(str(support_row["logging_envelope_json"]))
            if support_row is not None
            else None
        )
        logging_support = (
            dict(raw_logging_support)
            if isinstance(raw_logging_support, Mapping)
            else {
                "schema_version": None,
                "decision_type": "legacy_unknown",
                "probability_stage": None,
                "sampler": {},
                "candidates": [],
                "exclusions": [],
                "chosen_probability": None,
                "candidate_set_hash": None,
                "distribution_hash": None,
                "decision_record_fingerprint_schema_version": None,
                "decision_record_fingerprint": None,
            }
        )
        decision_context = json.loads(str(row["decision_context_json"]))
        eligible_modes = json.loads(str(row["eligible_modes_json"]))
        action_probabilities = _legacy_probability_mapping(row["action_probabilities_json"])
        support_candidate_set_hash = (
            str(support_row["candidate_set_hash"])
            if support_row is not None
            else None
        )
        support_distribution_hash = (
            str(support_row["distribution_hash"])
            if support_row is not None
            else None
        )
        (
            decision_record_fingerprint,
            decision_record_fingerprint_valid,
            decision_record_fingerprint_reason,
        ) = _verify_decision_record_fingerprint(
            policy_name=str(row["policy_name"]),
            policy_version=str(row["policy_version"]),
            policy_schema_version=str(row["policy_schema_version"]),
            decision_context=decision_context,
            eligible_modes=eligible_modes,
            action_probabilities=action_probabilities,
            chosen_mode=str(row["chosen_mode"]),
            logging_support=logging_support,
            support_candidate_set_hash=support_candidate_set_hash,
            support_distribution_hash=support_distribution_hash,
        )
        started_at = float(row["started_at"])
        completed_at = float(row["completed_at"]) if row["completed_at"] is not None else None
        historical_inflight = (
            as_of is not None
            and completed_at is not None
            and completed_at > float(as_of)
        )
        projected_completed_at = None if historical_inflight else completed_at
        projected_status = "inflight" if historical_inflight else str(row["status"])
        projected_success = (
            None
            if historical_inflight or row["success"] is None
            else bool(row["success"])
        )
        for contract in outcome_contracts.values():
            if not bool(contract.get("precommitted")):
                contract["commitment_timing_valid"] = False
                contract["commitment_timing_reason"] = "not_precommitted"
                continue
            try:
                committed_at = float(contract.get("committed_at"))
            except (TypeError, ValueError, OverflowError):
                committed_at = math.nan
            timing_valid = math.isfinite(committed_at) and committed_at <= started_at
            contract["commitment_timing_valid"] = timing_valid
            contract["commitment_timing_reason"] = (
                "verified" if timing_valid else "committed_after_decision_start"
            )
        contract_sources = {
            str(contract.get("commitment_source") or "")
            for contract in outcome_contracts.values()
        }
        complete_contract_set = set(outcome_contracts) == set(OUTCOME_NAMES)
        return {
            "route_id": route_id,
            "session_hash": str(row["session_hash"]),
            "session_sequence": int(row["session_sequence"]),
            "ledger_schema_version": int(row["ledger_schema_version"]),
            "policy_name": str(row["policy_name"]),
            "policy_version": str(row["policy_version"]),
            "policy_schema_version": str(row["policy_schema_version"]),
            "decision_context": decision_context,
            "eligible_modes": eligible_modes,
            "action_probabilities": action_probabilities,
            "logging_support": logging_support,
            "decision_type": logging_support.get("decision_type"),
            "probability_stage": logging_support.get("probability_stage"),
            "chosen_probability": logging_support.get("chosen_probability"),
            "candidate_set_hash": logging_support.get("candidate_set_hash"),
            "distribution_hash": logging_support.get("distribution_hash"),
            "decision_record_fingerprint": decision_record_fingerprint,
            "decision_record_fingerprint_valid": decision_record_fingerprint_valid,
            "decision_record_fingerprint_reason": decision_record_fingerprint_reason,
            "chosen_mode": str(row["chosen_mode"]),
            "executed_mode": (
                str(row["executed_mode"])
                if not historical_inflight and row["executed_mode"] is not None
                else None
            ),
            "estimated_economics": json.loads(str(row["estimated_economics_json"])),
            "actual_economics": (
                json.loads(str(row["actual_economics_json"] or "{}"))
                if not historical_inflight
                else {}
            ),
            "outcome_contracts": outcome_contracts,
            "outcome_events": outcome_events,
            "outcome_contracts_precommitted_at_begin": (
                complete_contract_set
                and all(
                    bool(contract.get("precommitted"))
                    and bool(contract.get("contract_hash_valid"))
                    and bool(contract.get("commitment_timing_valid"))
                    for contract in outcome_contracts.values()
                )
            ),
            "outcome_contracts_defaulted_at_begin": (
                complete_contract_set
                and contract_sources == {"safe_default"}
                and all(
                    bool(contract.get("contract_hash_valid"))
                    and bool(contract.get("commitment_timing_valid"))
                    for contract in outcome_contracts.values()
                )
            ),
            "outcome_contract_commitment_source": (
                next(iter(contract_sources)) if len(contract_sources) == 1 else "mixed"
            ),
            "status": projected_status,
            "success": projected_success,
            "error_category": (
                str(row["error_category"])
                if not historical_inflight and row["error_category"] is not None
                else None
            ),
            "error_message": (
                str(row["error_message"])
                if not historical_inflight and row["error_message"] is not None
                else None
            ),
            "started_at": started_at,
            "started_at_utc": _utc(started_at),
            "completed_at": projected_completed_at,
            "completed_at_utc": _utc(projected_completed_at),
            "duration_ms": (
                round((projected_completed_at - started_at) * 1000.0, 3)
                if projected_completed_at is not None
                else None
            ),
            "feedback_status": "known" if revision_count else "unknown",
            "feedback_revision_count": revision_count,
            "latest_feedback": latest_feedback,
        }

    def get_decision(self, route_id: Any) -> Dict[str, Any]:
        cooked_route_id = _existing_route_id(route_id)
        with self._read_transaction() as connection:
            row = connection.execute(
                "SELECT * FROM route_decisions WHERE route_id = ?", (cooked_route_id,)
            ).fetchone()
            if row is None:
                raise DecisionNotFoundError(f"unknown route_id: {cooked_route_id}")
            return self._decision_from_row(connection, row)

    def list_decisions(
        self,
        *,
        session_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        policy_version: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        cooked_limit = max(1, min(1000, int(limit)))
        session_hash = hash_session_identity(session_id) if session_id is not None else None
        clauses: List[str] = []
        parameters: List[Any] = []
        if session_hash is not None:
            clauses.append("session_hash = ?")
            parameters.append(session_hash)
        if policy_name is not None:
            clauses.append("policy_name = ?")
            parameters.append(_text(policy_name, "policy_name", limit=120))
        if policy_version is not None:
            clauses.append("policy_version = ?")
            parameters.append(_text(policy_version, "policy_version", limit=120))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._read_transaction() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM route_decisions
                {where}
                ORDER BY started_at DESC, route_id DESC LIMIT ?
                """,
                (*parameters, cooked_limit),
            ).fetchall()
            return [self._decision_from_row(connection, row) for row in rows]

    @staticmethod
    def _outcome_contract_maturity(
        decisions: Sequence[Mapping[str, Any]],
        *,
        as_of: float,
    ) -> Dict[str, Any]:
        by_outcome: Dict[str, Dict[str, Any]] = {}
        for outcome_name in OUTCOME_NAMES:
            contract_count = 0
            invalid_contract_count = 0
            late_commitment_count = 0
            precommitted_count = 0
            legacy_posthoc_count = 0
            mature_contract_count = 0
            pending_contract_count = 0
            recorded_event_count = 0
            observed_event_count = 0
            not_observed_event_count = 0
            routes_with_recorded_event: set[str] = set()
            for decision in decisions:
                route_id = str(decision.get("route_id") or "")
                contracts = decision.get("outcome_contracts")
                contract = contracts.get(outcome_name) if isinstance(contracts, Mapping) else None
                if isinstance(contract, Mapping):
                    contract_count += 1
                    if not bool(contract.get("contract_hash_valid")):
                        invalid_contract_count += 1
                        continue
                    if bool(contract.get("precommitted")) and not bool(
                        contract.get("commitment_timing_valid")
                    ):
                        late_commitment_count += 1
                        continue
                    if bool(contract.get("precommitted")):
                        precommitted_count += 1
                    if contract.get("commitment_source") == "legacy_posthoc":
                        legacy_posthoc_count += 1
                    maturity_at = float(decision.get("started_at") or 0.0) + float(
                        contract.get("maturity_delay_seconds") or 0.0
                    )
                    if maturity_at <= as_of:
                        mature_contract_count += 1
                    else:
                        pending_contract_count += 1
                events = decision.get("outcome_events")
                if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
                    continue
                for event in events:
                    if not isinstance(event, Mapping) or event.get("outcome_name") != outcome_name:
                        continue
                    recorded_event_count += 1
                    routes_with_recorded_event.add(route_id)
                    if event.get("observation_status") == "observed":
                        observed_event_count += 1
                    elif event.get("observation_status") == "not_observed":
                        not_observed_event_count += 1
            by_outcome[outcome_name] = {
                "contract_count": contract_count,
                "invalid_contract_count": invalid_contract_count,
                "late_commitment_count": late_commitment_count,
                "precommitted_count": precommitted_count,
                "legacy_posthoc_count": legacy_posthoc_count,
                "mature_contract_count": mature_contract_count,
                "pending_contract_count": pending_contract_count,
                "recorded_event_count": recorded_event_count,
                "observed_event_count": observed_event_count,
                "not_observed_event_count": not_observed_event_count,
                "routes_with_recorded_event": len(routes_with_recorded_event),
            }

        complete_sets = [
            decision
            for decision in decisions
            if isinstance(decision.get("outcome_contracts"), Mapping)
            and set(decision["outcome_contracts"]) == set(OUTCOME_NAMES)
            and all(
                bool(contract.get("contract_hash_valid"))
                for contract in decision["outcome_contracts"].values()
                if isinstance(contract, Mapping)
            )
            and all(
                isinstance(contract, Mapping)
                for contract in decision["outcome_contracts"].values()
            )
        ]
        return {
            "schema_version": OUTCOME_MATURITY_SCHEMA_VERSION,
            "as_of": as_of,
            "as_of_utc": _utc(as_of),
            "basis": "decision_started_at_plus_maturity_delay_seconds",
            "descriptive_only": True,
            "policy_value_estimate": None,
            "causal_identification": "not_performed",
            "missingness_identification": "not_performed",
            "absence_semantics": "no_event_recorded_is_not_classified",
            "included_routes": len(decisions),
            "complete_contract_sets": len(complete_sets),
            "precommitted_routes": sum(
                1
                for decision in complete_sets
                if bool(decision.get("outcome_contracts_precommitted_at_begin"))
            ),
            "legacy_posthoc_routes": sum(
                1
                for decision in complete_sets
                if decision.get("outcome_contract_commitment_source") == "legacy_posthoc"
            ),
            "by_outcome": by_outcome,
        }

    def policy_evidence_snapshot(
        self,
        *,
        session_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        policy_version: Optional[str] = None,
        limit: int = 1000,
        as_of: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return a prompt-free, ledger-native policy-evidence projection.

        The projection keeps lifecycle and missing-feedback state separate and
        exposes the exact post-filter support envelope used at decision time.
        It is intentionally read-only and contains no policy-value estimate.
        """

        if isinstance(as_of, bool):
            raise ValueError("as_of must be a finite timestamp")
        try:
            as_of_value = float(time.time() if as_of is None else as_of)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("as_of must be a finite timestamp") from exc
        if not math.isfinite(as_of_value):
            raise ValueError("as_of must be a finite timestamp")
        cooked_limit = max(1, min(1000, int(limit)))
        session_hash = hash_session_identity(session_id) if session_id is not None else None
        clauses = ["started_at <= ?"]
        parameters: List[Any] = [as_of_value]
        if session_hash is not None:
            clauses.append("session_hash = ?")
            parameters.append(session_hash)
        cooked_policy_name = (
            _text(policy_name, "policy_name", limit=120)
            if policy_name is not None
            else None
        )
        cooked_policy_version = (
            _text(policy_version, "policy_version", limit=120)
            if policy_version is not None
            else None
        )
        if cooked_policy_name is not None:
            clauses.append("policy_name = ?")
            parameters.append(cooked_policy_name)
        if cooked_policy_version is not None:
            clauses.append("policy_version = ?")
            parameters.append(cooked_policy_version)
        where = f"WHERE {' AND '.join(clauses)}"

        # Lifecycle counts, decisions, contracts, events, and latest feedback
        # are all read from one SQLite snapshot at one explicit cutoff.
        with self._read_transaction() as connection:
            status = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS started,
                    COALESCE(SUM(CASE WHEN status = 'completed' AND completed_at <= ? THEN 1 ELSE 0 END), 0) AS completed,
                    COALESCE(SUM(CASE WHEN status = 'failed' AND completed_at <= ? THEN 1 ELSE 0 END), 0) AS failed
                FROM route_decisions {where}
                """,
                (as_of_value, as_of_value, *parameters),
            ).fetchone()
            started = int(status["started"])
            completed = int(status["completed"])
            failed = int(status["failed"])
            terminal = completed + failed
            feedback_clauses = [
                clause.replace("session_hash", "d.session_hash")
                .replace("policy_name", "d.policy_name")
                .replace("policy_version", "d.policy_version")
                .replace("started_at", "d.started_at")
                for clause in clauses
            ]
            feedback_where = f"WHERE {' AND '.join(feedback_clauses)}"
            feedback = connection.execute(
                f"""
                SELECT
                    COUNT(f.revision) AS revision_count,
                    COUNT(DISTINCT f.route_id) AS known,
                    COUNT(DISTINCT CASE WHEN d.completed_at <= ? THEN f.route_id END) AS terminal_known
                FROM route_decisions d
                LEFT JOIN route_feedback_revisions f
                  ON f.route_id = d.route_id AND f.recorded_at <= ?
                {feedback_where}
                """,
                (as_of_value, as_of_value, *parameters),
            ).fetchone()
            known = int(feedback["known"])
            terminal_known = int(feedback["terminal_known"])
            report = {
                "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                "session_hash": session_hash,
                "policy_filter": {
                    "policy_name": policy_name,
                    "policy_version": policy_version,
                },
                "counts": {
                    "started": started,
                    "completed": completed,
                    "failed": failed,
                    "inflight": started - terminal,
                },
                "feedback_coverage": {
                    "known": known,
                    "unknown": started - known,
                    "coverage_rate": round(known / started, 6) if started else 0.0,
                    "revision_count": int(feedback["revision_count"]),
                    "terminal_known": terminal_known,
                    "terminal_unknown": terminal - terminal_known,
                    "terminal_coverage_rate": (
                        round(terminal_known / terminal, 6) if terminal else 0.0
                    ),
                    "missing_feedback_semantics": "unknown",
                },
            }
            rows = connection.execute(
                f"""
                SELECT * FROM route_decisions
                {where}
                ORDER BY started_at DESC, route_id DESC LIMIT ?
                """,
                (*parameters, cooked_limit),
            ).fetchall()
            decisions = [
                self._decision_from_row(connection, row, as_of=as_of_value)
                for row in rows
            ]
        decisions.sort(key=lambda row: (int(row.get("session_sequence") or 0), str(row.get("route_id") or "")))
        usage_rows: List[Dict[str, Any]] = []
        feedback_rows: List[Dict[str, Any]] = []
        expected_context_by_route_id: Dict[str, Any] = {}

        for decision in decisions:
            route_id = str(decision["route_id"])
            context = _prompt_free_context(decision.get("decision_context") or {})
            if not isinstance(context, dict):
                context = {}
            support = dict(decision.get("logging_support") or {})
            eligible_modes = list(decision.get("eligible_modes") or [])
            allowed_modes_raw = context.get("allowed_agent_modes")
            allowed_modes = (
                [str(mode) for mode in allowed_modes_raw]
                if isinstance(allowed_modes_raw, (list, tuple))
                else list(eligible_modes)
            )
            auto_policy = {
                "policy_id": decision.get("policy_name"),
                "policy_version": decision.get("policy_version"),
                "feature_schema_version": decision.get("policy_schema_version"),
                "decision_type": support.get("decision_type"),
                "probability_stage": support.get("probability_stage"),
                "decision_context": context,
                "action_mode": context.get("action_mode"),
                "budget_profile": context.get("budget_profile"),
                "score": context.get("score"),
                "selected_agent_mode": decision.get("chosen_mode"),
                "allowed_agent_modes": allowed_modes,
                "eligible_actions": eligible_modes,
                "action_probabilities": dict(decision.get("action_probabilities") or {}),
                "post_filter_action_probabilities": dict(decision.get("action_probabilities") or {}),
                "logging_propensity": support.get("chosen_probability"),
                "candidate_set_hash": support.get("candidate_set_hash"),
                "distribution_hash": support.get("distribution_hash"),
                "decision_record_fingerprint": decision.get(
                    "decision_record_fingerprint"
                ),
                "decision_record_fingerprint_valid": bool(
                    decision.get("decision_record_fingerprint_valid")
                ),
                "decision_record_fingerprint_reason": decision.get(
                    "decision_record_fingerprint_reason"
                ),
                "outcome_contracts_precommitted_at_begin": bool(
                    decision.get("outcome_contracts_precommitted_at_begin")
                ),
                "logging_support": support,
            }
            usage_rows.append(
                {
                    "route_id": route_id,
                    "session_hash": decision.get("session_hash"),
                    "session_sequence": decision.get("session_sequence"),
                    "selected_agent_mode": decision.get("chosen_mode"),
                    "executed_agent_mode": decision.get("executed_mode"),
                    "status": decision.get("status"),
                    "success": decision.get("success"),
                    "decision_record_fingerprint": decision.get(
                        "decision_record_fingerprint"
                    ),
                    "decision_record_fingerprint_valid": bool(
                        decision.get("decision_record_fingerprint_valid")
                    ),
                    "decision_record_fingerprint_reason": decision.get(
                        "decision_record_fingerprint_reason"
                    ),
                    "outcome_contracts": dict(decision.get("outcome_contracts") or {}),
                    "outcome_events": list(decision.get("outcome_events") or []),
                    "outcome_contracts_precommitted_at_begin": bool(
                        decision.get("outcome_contracts_precommitted_at_begin")
                    ),
                    "outcome_contracts_defaulted_at_begin": bool(
                        decision.get("outcome_contracts_defaulted_at_begin")
                    ),
                    "auto_agent_policy": auto_policy,
                    "route_economics": {
                        "estimate": dict(decision.get("estimated_economics") or {}),
                        "actual": dict(decision.get("actual_economics") or {}),
                    },
                }
            )
            expected_context_by_route_id[route_id] = context

            latest = decision.get("latest_feedback")
            if isinstance(latest, Mapping) and isinstance(latest.get("feedback"), Mapping):
                feedback = dict(latest["feedback"])
                feedback.update(
                    {
                        "route_id": route_id,
                        "feedback_revision": latest.get("revision"),
                        "recorded_at": latest.get("recorded_at"),
                        "route_status": decision.get("status"),
                        "route_economics": {
                            "actual": dict(decision.get("actual_economics") or {})
                        },
                    }
                )
                feedback_rows.append(feedback)

        started = int((report.get("counts") or {}).get("started") or 0)
        return {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "support_schema_version": SUPPORT_SCHEMA_VERSION,
            "outcome_contract_schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
            "as_of": as_of_value,
            "as_of_utc": _utc(as_of_value),
            "policy_filter": {
                "policy_name": policy_name,
                "policy_version": policy_version,
            },
            "analysis_window": {
                "limit": cooked_limit,
                "included_decisions": len(decisions),
                "total_decisions": started,
                "truncated": started > len(decisions),
                "order": "session_sequence_ascending",
                "as_of": as_of_value,
                "as_of_utc": _utc(as_of_value),
            },
            "usage_rows": usage_rows,
            "feedback_rows": feedback_rows,
            "expected_context_by_route_id": expected_context_by_route_id,
            "lifecycle": report,
            "outcome_contract_maturity": self._outcome_contract_maturity(
                decisions,
                as_of=as_of_value,
            ),
        }

    def report(
        self,
        *,
        session_id: Optional[str] = None,
        policy_name: Optional[str] = None,
        policy_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        session_hash = hash_session_identity(session_id) if session_id is not None else None
        clauses: List[str] = []
        parameters_list: List[Any] = []
        if session_hash is not None:
            clauses.append("session_hash = ?")
            parameters_list.append(session_hash)
        if policy_name is not None:
            clauses.append("policy_name = ?")
            parameters_list.append(_text(policy_name, "policy_name", limit=120))
        if policy_version is not None:
            clauses.append("policy_version = ?")
            parameters_list.append(_text(policy_version, "policy_version", limit=120))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        parameters: Tuple[Any, ...] = tuple(parameters_list)
        connection = self._connect()
        try:
            status = connection.execute(
                f"""
                SELECT
                    COUNT(*) AS started,
                    COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) AS completed,
                    COALESCE(SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END), 0) AS failed,
                    COALESCE(SUM(CASE WHEN status = 'inflight' THEN 1 ELSE 0 END), 0) AS inflight
                FROM route_decisions {where}
                """,
                parameters,
            ).fetchone()
            feedback_clauses = [clause.replace("session_hash", "d.session_hash").replace("policy_name", "d.policy_name").replace("policy_version", "d.policy_version") for clause in clauses]
            feedback_where = f"WHERE {' AND '.join(feedback_clauses)}" if feedback_clauses else ""
            feedback = connection.execute(
                f"""
                SELECT
                    COUNT(f.revision) AS revision_count,
                    COUNT(DISTINCT f.route_id) AS known,
                    COUNT(DISTINCT CASE WHEN d.status != 'inflight' THEN f.route_id END) AS terminal_known
                FROM route_decisions d
                LEFT JOIN route_feedback_revisions f ON f.route_id = d.route_id
                {feedback_where}
                """,
                parameters,
            ).fetchone()
        finally:
            connection.close()

        started = int(status["started"])
        completed = int(status["completed"])
        failed = int(status["failed"])
        inflight = int(status["inflight"])
        terminal = completed + failed
        known = int(feedback["known"])
        terminal_known = int(feedback["terminal_known"])
        return {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "session_hash": session_hash,
            "policy_filter": {
                "policy_name": policy_name,
                "policy_version": policy_version,
            },
            "counts": {
                "started": started,
                "completed": completed,
                "failed": failed,
                "inflight": inflight,
            },
            "feedback_coverage": {
                "known": known,
                "unknown": started - known,
                "coverage_rate": round(known / started, 6) if started else 0.0,
                "revision_count": int(feedback["revision_count"]),
                "terminal_known": terminal_known,
                "terminal_unknown": terminal - terminal_known,
                "terminal_coverage_rate": round(terminal_known / terminal, 6) if terminal else 0.0,
                "missing_feedback_semantics": "unknown",
            },
        }

    def snapshot(self, *, session_id: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        report = self.report(session_id=session_id)
        report["recent_decisions"] = self.list_decisions(session_id=session_id, limit=limit)
        return report


__all__ = [
    "DECISION_TYPES",
    "DECISION_STATUSES",
    "DECISION_FINGERPRINT_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "OUTCOME_CONTRACT_SCHEMA_VERSION",
    "OUTCOME_MATURITY_SCHEMA_VERSION",
    "OUTCOME_NAMES",
    "SUPPORT_SCHEMA_VERSION",
    "DecisionNotFoundError",
    "LedgerConflictError",
    "RoutePolicyLedger",
    "RoutePolicyLedgerError",
    "build_logging_support_envelope",
    "build_route_outcome_contracts",
    "hash_session_identity",
]
