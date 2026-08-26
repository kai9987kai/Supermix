"""Evidence-first answer admission for the experimental NexusMind surfaces.

The Nexus modules expose several useful *processes*: an exact closed-world
reasoner, deterministic ideation templates, a scaffolded swarm/GoT search and
an untrained MiMo telemetry core.  Those processes do not share the same
answer authority and their internal scores are not interchangeable with a
calibrated probability of correctness.

This module makes that boundary machine-readable.  A decision receipt records
why an output was answered, offered as analysis only, or withheld.  The hash is
only an integrity checksum over the receipt; it never creates factual, tool,
permission, safety, model-promotion, or memory authority.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SELECTIVE_ANSWER_SCHEMA_VERSION = "nexus-selective-answer-v1"
SELECTIVE_ANSWER_POLICY_VERSION = "verifier-first-admission-v1"

DECISIONS = frozenset({"answered", "analysis_only", "abstained"})
EVIDENCE_CLASSES = frozenset(
    {
        "verified_exact",
        "deterministic_heuristic",
        "template_deliberation",
        "unverified_neural",
        "no_applicable_verifier",
    }
)
ANSWER_VERIFIER_IDS = frozenset({"grounding_runtime.finalize_grounded_response"})
VERIFIED_GROUNDING_REASONS = frozenset(
    {"explicit_arithmetic_exact", "verified_reasoning_solution"}
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class EpistemicDecision:
    """A bounded decision about whether a candidate may be shown as an answer."""

    decision: str
    evidence_class: str
    reason: str
    claim_scope: str
    answer_authority: bool = False
    correctness_confidence: Optional[float] = None
    confidence_kind: str = "unavailable"
    calibrated: bool = False
    internal_score: Optional[float] = None
    internal_score_name: str = ""
    verifier: Dict[str, Any] = field(default_factory=dict)
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    protocol: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.decision not in DECISIONS:
            raise ValueError(f"unsupported selective-answer decision: {self.decision}")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"unsupported evidence class: {self.evidence_class}")
        if not str(self.reason).strip() or not str(self.claim_scope).strip():
            raise ValueError("reason and claim_scope are required")
        if not self.limitations:
            raise ValueError("at least one explicit limitation is required")
        if self.answer_authority:
            if self.decision != "answered" or self.evidence_class != "verified_exact":
                raise ValueError("only a verified exact result may have answer authority")
            if self.correctness_confidence != 1.0:
                raise ValueError("verified exact authority uses deterministic in-scope confidence 1.0")
            if self.confidence_kind != "deterministic_in_scope":
                raise ValueError("verified exact authority must identify deterministic confidence")
        elif self.correctness_confidence is not None:
            raise ValueError("non-authoritative output cannot publish correctness confidence")
        if self.decision == "answered" and not self.answer_authority:
            raise ValueError("answered decisions require verified exact answer authority")
        if self.decision == "analysis_only" and self.evidence_class not in {
            "deterministic_heuristic",
            "template_deliberation",
        }:
            raise ValueError("analysis-only decisions require a heuristic evidence class")
        if self.decision == "abstained" and self.evidence_class not in {
            "unverified_neural",
            "no_applicable_verifier",
        }:
            raise ValueError("abstentions require a non-authoritative evidence class")
        if self.calibrated:
            raise ValueError("policy v1 does not admit empirical calibration claims")
        if self.internal_score is not None:
            if not math.isfinite(float(self.internal_score)):
                raise ValueError("internal_score must be finite")
            if not self.internal_score_name:
                raise ValueError("internal_score_name is required with internal_score")
        if self.decision == "abstained" and self.internal_score is not None:
            raise ValueError("abstentions cannot publish an internal score")

    def _payload(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["limitations"] = list(self.limitations)
        payload.update(
            {
                "schema_version": SELECTIVE_ANSWER_SCHEMA_VERSION,
                "policy_version": SELECTIVE_ANSWER_POLICY_VERSION,
                "receipt_is_authority": False,
                "authority": {
                    "answers_within_claim_scope": bool(self.answer_authority),
                    "controls_tools": False,
                    "controls_permissions": False,
                    "controls_safety": False,
                    "controls_memory": False,
                    "controls_routes": False,
                    "controls_model_activation": False,
                    "controls_model_promotion": False,
                },
            }
        )
        return payload

    def to_dict(self) -> Dict[str, Any]:
        payload = self._payload()
        payload["receipt_sha256"] = hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest()
        return payload


_SHARED_LIMITS = (
    "The decision receipt is integrity metadata, not independent evidence.",
    "It grants no tool, permission, safety, memory, route, activation, or promotion authority.",
)


def verified_exact_decision(
    *,
    reason: str,
    claim_scope: str,
    verifier_id: str,
    verifier_receipt_sha256: str = "",
    protocol: Optional[Mapping[str, Any]] = None,
) -> EpistemicDecision:
    """Admit a freshly recomputed, strictly gated closed-world answer."""

    verifier: Dict[str, Any] = {
        "id": str(verifier_id),
        "passed": True,
        "independent_recompute": True,
    }
    if verifier["id"] not in ANSWER_VERIFIER_IDS:
        raise ValueError("verified answers require an allowlisted verifier")
    if _is_sha256(verifier_receipt_sha256):
        verifier["receipt_sha256"] = verifier_receipt_sha256.lower()
    return EpistemicDecision(
        decision="answered",
        evidence_class="verified_exact",
        reason=reason,
        claim_scope=claim_scope,
        answer_authority=True,
        correctness_confidence=1.0,
        confidence_kind="deterministic_in_scope",
        calibrated=False,
        verifier=verifier,
        limitations=(
            "Confidence 1.0 applies only to the accepted closed-world parse and deterministic recomputation; it is not empirical calibration.",
            *_SHARED_LIMITS,
        ),
        protocol=dict(protocol or {}),
    )


def analysis_only_decision(
    *,
    reason: str,
    claim_scope: str,
    evidence_class: str = "deterministic_heuristic",
    internal_score: Optional[float] = None,
    internal_score_name: str = "",
    limitations: Sequence[str] = (),
    protocol: Optional[Mapping[str, Any]] = None,
) -> EpistemicDecision:
    """Expose a bounded heuristic artifact without presenting it as an answer."""

    return EpistemicDecision(
        decision="analysis_only",
        evidence_class=evidence_class,
        reason=reason,
        claim_scope=claim_scope,
        answer_authority=False,
        correctness_confidence=None,
        confidence_kind="unavailable",
        calibrated=False,
        internal_score=internal_score,
        internal_score_name=internal_score_name,
        verifier={"id": "none", "passed": False, "independent_recompute": False},
        limitations=tuple(limitations) + _SHARED_LIMITS,
        protocol=dict(protocol or {}),
    )


def abstained_decision(
    *,
    reason: str,
    claim_scope: str,
    evidence_class: str = "no_applicable_verifier",
    limitations: Sequence[str] = (),
    protocol: Optional[Mapping[str, Any]] = None,
) -> EpistemicDecision:
    """Withhold an answer when no eligible verifier can support it."""

    return EpistemicDecision(
        decision="abstained",
        evidence_class=evidence_class,
        reason=reason,
        claim_scope=claim_scope,
        answer_authority=False,
        correctness_confidence=None,
        confidence_kind="unavailable",
        calibrated=False,
        verifier={"id": "none", "passed": False, "independent_recompute": False},
        limitations=tuple(limitations) + _SHARED_LIMITS,
        protocol=dict(protocol or {}),
    )


def verify_epistemic_receipt(value: Any) -> bool:
    """Validate schema, invariants, and the self-checksum of a serialized receipt."""

    if not isinstance(value, Mapping):
        return False
    payload = dict(value)
    supplied = payload.pop("receipt_sha256", None)
    if not _is_sha256(supplied):
        return False
    expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if supplied.lower() != expected:
        return False
    if payload.get("schema_version") != SELECTIVE_ANSWER_SCHEMA_VERSION:
        return False
    if payload.get("policy_version") != SELECTIVE_ANSWER_POLICY_VERSION:
        return False
    if payload.get("receipt_is_authority") is not False:
        return False
    authority = payload.get("authority")
    if not isinstance(authority, Mapping):
        return False
    forbidden = (
        "controls_tools",
        "controls_permissions",
        "controls_safety",
        "controls_memory",
        "controls_routes",
        "controls_model_activation",
        "controls_model_promotion",
    )
    if any(authority.get(key) is not False for key in forbidden):
        return False
    decision = payload.get("decision")
    evidence = payload.get("evidence_class")
    answer_authority = payload.get("answer_authority")
    confidence = payload.get("correctness_confidence")
    if not isinstance(payload.get("reason"), str) or not payload["reason"].strip():
        return False
    if not isinstance(payload.get("claim_scope"), str) or not payload["claim_scope"].strip():
        return False
    if not isinstance(payload.get("limitations"), list) or not payload["limitations"]:
        return False
    if payload.get("calibrated") is not False:
        return False
    if not isinstance(payload.get("verifier"), Mapping) or not isinstance(payload.get("protocol"), Mapping):
        return False
    internal_score = payload.get("internal_score")
    if internal_score is not None:
        try:
            if not math.isfinite(float(internal_score)):
                return False
        except (TypeError, ValueError):
            return False
        if not isinstance(payload.get("internal_score_name"), str) or not payload["internal_score_name"]:
            return False
    if answer_authority is True:
        verifier = payload["verifier"]
        return bool(
            decision == "answered"
            and evidence == "verified_exact"
            and confidence == 1.0
            and payload.get("confidence_kind") == "deterministic_in_scope"
            and authority.get("answers_within_claim_scope") is True
            and verifier.get("id") in ANSWER_VERIFIER_IDS
            and verifier.get("passed") is True
            and verifier.get("independent_recompute") is True
            and internal_score is None
        )
    verifier = payload["verifier"]
    if not (
        answer_authority is False
        and decision in {"analysis_only", "abstained"}
        and confidence is None
        and payload.get("confidence_kind") == "unavailable"
        and authority.get("answers_within_claim_scope") is False
        and set(verifier) == {"id", "passed", "independent_recompute"}
        and verifier.get("id") == "none"
        and verifier.get("passed") is False
        and verifier.get("independent_recompute") is False
    ):
        return False
    if decision == "analysis_only":
        return evidence in {"deterministic_heuristic", "template_deliberation"}
    return bool(
        evidence in {"unverified_neural", "no_applicable_verifier"}
        and internal_score is None
        and payload.get("internal_score_name") == ""
    )


def verify_grounded_answer_result(
    value: Any,
    *,
    receipt_schema_version: str,
    require_science_plan: bool = False,
) -> bool:
    """Validate the grounder result/receipt/text binding used by every surface.

    This is an invariant check, not a signature check. It prevents malformed or
    partially selected grounder records from acquiring answer authority, while
    callers still trust the locally executed grounder implementation itself.
    """

    if not isinstance(value, Mapping):
        return False
    reason = value.get("reason")
    text = value.get("text")
    receipt = value.get("answer_receipt")
    if not (
        reason in VERIFIED_GROUNDING_REASONS
        and isinstance(text, str)
        and text.strip()
        and isinstance(receipt, Mapping)
    ):
        return False

    verification = receipt.get("verification")
    authority = receipt.get("authority")
    required_authority_keys = {
        "controls_compute",
        "controls_interaction_strategy",
        "controls_permissions",
        "controls_promotion",
        "controls_routes",
        "controls_safety",
        "controls_tools",
    }
    if not (
        receipt.get("schema_version") == receipt_schema_version
        and receipt.get("decision") == "verified_selected"
        and receipt.get("selected") is True
        and receipt.get("solved") is True
        and receipt.get("diagnostic_only") is True
        and isinstance(verification, Mapping)
        and verification.get("passed") is True
        and type(verification.get("independent")) is bool
        and isinstance(authority, Mapping)
        and required_authority_keys.issubset(authority)
        and all(authority.get(key) is False for key in required_authority_keys)
    ):
        return False

    if reason == "explicit_arithmetic_exact":
        arithmetic = value.get("arithmetic")
        display = arithmetic.get("display") if isinstance(arithmetic, Mapping) else None
        if not (
            receipt.get("selection_reason") == "exact_arithmetic"
            and receipt.get("kind") == "exact_arithmetic"
            and receipt.get("method") == "bounded_exact_arithmetic"
            and receipt.get("problem_class") == "arithmetic"
            and isinstance(arithmetic, Mapping)
            and arithmetic.get("solved") is True
            and isinstance(display, str)
            and display
            and display in text
        ):
            return False
    else:
        reasoning = value.get("reasoning")
        answer = reasoning.get("answer") if isinstance(reasoning, Mapping) else None
        display = answer.get("display") if isinstance(answer, Mapping) else None
        reasoning_verification = (
            reasoning.get("verification") if isinstance(reasoning, Mapping) else None
        )
        method = reasoning.get("method") if isinstance(reasoning, Mapping) else None
        if not (
            receipt.get("selection_reason") == "verified_reasoning"
            and receipt.get("kind") == "deliberate_reasoning"
            and isinstance(reasoning, Mapping)
            and isinstance(method, str)
            and method
            and receipt.get("method") == method
            and receipt.get("problem_class") == reasoning.get("problem_class")
            and reasoning.get("solved") is True
            and isinstance(reasoning_verification, Mapping)
            and reasoning_verification.get("passed") is True
            and isinstance(display, str)
            and display
            and display in text
        ):
            return False

    if require_science_plan:
        science_plan = receipt.get("science_plan")
        science_verification = (
            science_plan.get("verification") if isinstance(science_plan, Mapping) else None
        )
        checks = science_plan.get("checks") if isinstance(science_plan, Mapping) else None
        if not (
            isinstance(science_plan, Mapping)
            and science_plan.get("present") is True
            and science_plan.get("formula_id") == receipt.get("method")
            and isinstance(science_verification, Mapping)
            and science_verification.get("passed") is True
            and isinstance(checks, Mapping)
            and checks
            and all(value is True for value in checks.values())
        ):
            return False
    return True


__all__ = [
    "EVIDENCE_CLASSES",
    "ANSWER_VERIFIER_IDS",
    "EpistemicDecision",
    "SELECTIVE_ANSWER_POLICY_VERSION",
    "SELECTIVE_ANSWER_SCHEMA_VERSION",
    "abstained_decision",
    "analysis_only_decision",
    "verified_exact_decision",
    "verify_epistemic_receipt",
    "verify_grounded_answer_result",
]
