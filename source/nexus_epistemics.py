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
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SELECTIVE_ANSWER_SCHEMA_VERSION = "nexus-selective-answer-v2"
SELECTIVE_ANSWER_POLICY_VERSION = "request-bound-verifier-first-admission-v2"

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
ANSWER_SURFACES = frozenset({"engine", "think", "solve", "scientific", "chat"})
VERIFIED_GROUNDING_REASONS = frozenset(
    {"explicit_arithmetic_exact", "verified_reasoning_solution"}
)
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?:/\d+)?(?!\w|\.\d)"
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


def _numeric_tokens(value: Any) -> Tuple[str, ...]:
    return tuple(match.group(0) for match in _NUMERIC_TOKEN_RE.finditer(str(value or "")))


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
    bindings: Dict[str, Any] = field(default_factory=dict)
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
        if not all(isinstance(item, str) and item.strip() for item in self.limitations):
            raise ValueError("limitations must be non-empty strings")
        if self.answer_authority:
            if self.decision != "answered" or self.evidence_class != "verified_exact":
                raise ValueError("only a verified exact result may have answer authority")
            if self.correctness_confidence is not None:
                raise ValueError("deterministic assurance is not numeric correctness confidence")
            if self.confidence_kind != "deterministic_assurance_not_probability":
                raise ValueError("verified exact authority must identify non-probabilistic assurance")
            required_bindings = {
                "request_sha256",
                "output_sha256",
                "verifier_receipt_sha256",
                "request_nonce_sha256",
                "surface",
            }
            if set(self.bindings) != required_bindings:
                raise ValueError("verified exact authority requires a closed request/output binding")
            if not all(
                _is_sha256(self.bindings.get(key))
                for key in ("request_sha256", "output_sha256", "verifier_receipt_sha256")
            ):
                raise ValueError("verified exact authority requires SHA-256 request/output/verifier bindings")
            nonce_digest = self.bindings.get("request_nonce_sha256")
            if not _is_sha256(nonce_digest):
                raise ValueError("verified exact authority requires a SHA-256 request nonce binding")
            if self.bindings.get("surface") not in ANSWER_SURFACES:
                raise ValueError("verified exact authority requires an allowlisted surface")
        elif self.correctness_confidence is not None:
            raise ValueError("non-authoritative output cannot publish correctness confidence")
        elif self.bindings:
            raise ValueError("non-authoritative outputs cannot publish answer bindings")
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
            raise ValueError("the selective-answer policy does not admit empirical calibration claims")
        if self.internal_score is not None:
            if not math.isfinite(float(self.internal_score)):
                raise ValueError("internal_score must be finite")
            if not self.internal_score_name:
                raise ValueError("internal_score_name is required with internal_score")
        elif self.internal_score_name:
            raise ValueError("internal_score_name requires internal_score")
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
    request_sha256: str,
    output_sha256: str,
    verifier_receipt_sha256: str,
    surface: str,
    request_nonce_sha256: str,
    protocol: Optional[Mapping[str, Any]] = None,
) -> EpistemicDecision:
    """Admit a freshly recomputed, strictly gated closed-world answer."""

    verifier: Dict[str, Any] = {
        "id": str(verifier_id),
        "passed": True,
        "fresh_recompute": True,
        "algorithmically_independent": False,
    }
    if verifier["id"] not in ANSWER_VERIFIER_IDS:
        raise ValueError("verified answers require an allowlisted verifier")
    bindings = {
        "request_sha256": str(request_sha256).lower(),
        "output_sha256": str(output_sha256).lower(),
        "verifier_receipt_sha256": str(verifier_receipt_sha256).lower(),
        "request_nonce_sha256": str(request_nonce_sha256).lower(),
        "surface": str(surface),
    }
    return EpistemicDecision(
        decision="answered",
        evidence_class="verified_exact",
        reason=reason,
        claim_scope=claim_scope,
        answer_authority=True,
        correctness_confidence=None,
        confidence_kind="deterministic_assurance_not_probability",
        calibrated=False,
        verifier=verifier,
        bindings=bindings,
        limitations=(
            "Deterministic assurance applies only to the accepted closed-world parse and recomputation; no numeric correctness probability is claimed.",
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
        verifier={
            "id": "none",
            "passed": False,
            "fresh_recompute": False,
            "algorithmically_independent": False,
        },
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
        verifier={
            "id": "none",
            "passed": False,
            "fresh_recompute": False,
            "algorithmically_independent": False,
        },
        limitations=tuple(limitations) + _SHARED_LIMITS,
        protocol=dict(protocol or {}),
    )


def verify_epistemic_receipt(value: Any) -> bool:
    """Validate schema, invariants, and the self-checksum of a serialized receipt."""

    if not isinstance(value, Mapping):
        return False
    payload = dict(value)
    expected_top_level = {
        "decision",
        "evidence_class",
        "reason",
        "claim_scope",
        "answer_authority",
        "correctness_confidence",
        "confidence_kind",
        "calibrated",
        "internal_score",
        "internal_score_name",
        "verifier",
        "bindings",
        "limitations",
        "protocol",
        "schema_version",
        "policy_version",
        "receipt_is_authority",
        "authority",
        "receipt_sha256",
    }
    if set(payload) != expected_top_level:
        return False
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
    forbidden = {
        "controls_tools",
        "controls_permissions",
        "controls_safety",
        "controls_memory",
        "controls_routes",
        "controls_model_activation",
        "controls_model_promotion",
    }
    if set(authority) != {"answers_within_claim_scope", *forbidden}:
        return False
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
    if (
        not isinstance(payload.get("limitations"), list)
        or not payload["limitations"]
        or not all(
            isinstance(item, str) and item.strip() for item in payload["limitations"]
        )
    ):
        return False
    if payload.get("calibrated") is not False:
        return False
    if not isinstance(payload.get("verifier"), Mapping) or not isinstance(payload.get("protocol"), Mapping):
        return False
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping):
        return False
    internal_score = payload.get("internal_score")
    internal_score_name = payload.get("internal_score_name")
    if not isinstance(internal_score_name, str):
        return False
    if internal_score is not None:
        try:
            if not math.isfinite(float(internal_score)):
                return False
        except (TypeError, ValueError):
            return False
        if not internal_score_name:
            return False
    elif internal_score_name:
        return False
    if answer_authority is True:
        verifier = payload["verifier"]
        return bool(
            decision == "answered"
            and evidence == "verified_exact"
            and confidence is None
            and payload.get("confidence_kind") == "deterministic_assurance_not_probability"
            and authority.get("answers_within_claim_scope") is True
            and set(verifier)
            == {"id", "passed", "fresh_recompute", "algorithmically_independent"}
            and verifier.get("id") in ANSWER_VERIFIER_IDS
            and verifier.get("passed") is True
            and verifier.get("fresh_recompute") is True
            and verifier.get("algorithmically_independent") is False
            and set(bindings)
            == {
                "request_sha256",
                "output_sha256",
                "verifier_receipt_sha256",
                "request_nonce_sha256",
                "surface",
            }
            and all(
                _is_sha256(bindings.get(key))
                for key in ("request_sha256", "output_sha256", "verifier_receipt_sha256")
            )
            and _is_sha256(bindings.get("request_nonce_sha256"))
            and bindings.get("surface") in ANSWER_SURFACES
            and internal_score is None
        )
    verifier = payload["verifier"]
    if not (
        answer_authority is False
        and decision in {"analysis_only", "abstained"}
        and confidence is None
        and payload.get("confidence_kind") == "unavailable"
        and authority.get("answers_within_claim_scope") is False
        and set(verifier)
        == {"id", "passed", "fresh_recompute", "algorithmically_independent"}
        and verifier.get("id") == "none"
        and verifier.get("passed") is False
        and verifier.get("fresh_recompute") is False
        and verifier.get("algorithmically_independent") is False
        and dict(bindings) == {}
    ):
        return False
    if decision == "analysis_only":
        return evidence in {"deterministic_heuristic", "template_deliberation"}
    return bool(
        evidence in {"unverified_neural", "no_applicable_verifier"}
        and internal_score is None
        and payload.get("internal_score_name") == ""
    )


def verify_epistemic_receipt_binding(
    value: Any,
    *,
    request_sha256: str,
    output_sha256: str,
    verifier_receipt_sha256: str,
    surface: str,
    request_nonce_sha256: str = "",
) -> bool:
    """Check an answered receipt against the caller's expected context."""

    if not verify_epistemic_receipt(value) or not isinstance(value, Mapping):
        return False
    if value.get("decision") != "answered" or value.get("answer_authority") is not True:
        return False
    bindings = value.get("bindings")
    return bool(
        isinstance(bindings, Mapping)
        and bindings.get("request_sha256") == str(request_sha256).lower()
        and bindings.get("output_sha256") == str(output_sha256).lower()
        and bindings.get("verifier_receipt_sha256")
        == str(verifier_receipt_sha256).lower()
        and bindings.get("surface") == surface
        and bindings.get("request_nonce_sha256") == str(request_nonce_sha256).lower()
    )


def verify_grounded_answer_result(
    value: Any,
    *,
    receipt_schema_version: str,
    runtime_version: str,
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
    consensus = receipt.get("consensus")
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
        and receipt.get("runtime_version") == runtime_version
        and receipt.get("decision") == "verified_selected"
        and receipt.get("selected") is True
        and receipt.get("solved") is True
        and receipt.get("diagnostic_only") is True
        and isinstance(verification, Mapping)
        and verification.get("passed") is True
        and type(verification.get("independent")) is bool
        and isinstance(authority, Mapping)
        and set(authority) == required_authority_keys
        and all(authority.get(key) is False for key in required_authority_keys)
        and isinstance(consensus, Mapping)
        and set(consensus) == {"conflicting", "paths"}
        and consensus.get("conflicting") is False
        and type(consensus.get("paths")) is int
        and 0 <= consensus.get("paths") <= 64
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
        allowed_numeric_tokens = set(
            _numeric_tokens(
                " ".join(
                    str(arithmetic.get(key) or "")
                    for key in ("expression", "exact", "display", "approximation")
                )
            )
        )
        if not allowed_numeric_tokens or any(
            token not in allowed_numeric_tokens for token in _numeric_tokens(text)
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
        answer_representations = (
            [
                str(answer.get(key) or "")
                for key in ("display", "exact", "approximation")
            ]
            if isinstance(answer, Mapping)
            else []
        )
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
            and any(value and value in text for value in answer_representations)
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
    "ANSWER_SURFACES",
    "EpistemicDecision",
    "SELECTIVE_ANSWER_POLICY_VERSION",
    "SELECTIVE_ANSWER_SCHEMA_VERSION",
    "abstained_decision",
    "analysis_only_decision",
    "verified_exact_decision",
    "verify_epistemic_receipt",
    "verify_epistemic_receipt_binding",
    "verify_grounded_answer_result",
]
