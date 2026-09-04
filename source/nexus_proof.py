"""Renderer-verifiable proof-carrying numeric claims for NexusMind.

The capsule produced here binds one submitted request, one exact public output,
and every numeric span in that output to a freshly accepted deterministic
grounder result.  It is intentionally *not* a signature or a probability of
correctness.  A renderer earns a verified mark only by sending the capsule back
through the live verifier, which recomputes the answer and compares the entire
closed-schema capsule.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any, Dict, List, Optional, Tuple

import nexus_epistemics as epistemics
import nexus_independent_checker as independent_checker


PROOF_CAPSULE_SCHEMA_VERSION = "nexus-proof-carrying-number-v2"
PROOF_CAPSULE_POLICY_VERSION = "renderer-fresh-revalidation-v2-independent-witness"

_NUMERIC_SPAN_RE = re.compile(
    r"(?<![\w.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE][-+]?\d+)?(?:/\d+)?(?!\w|\.\d)"
)
_NONCE_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_AUTHORITY_KEYS = {
    "controls_tools",
    "controls_permissions",
    "controls_safety",
    "controls_memory",
    "controls_routes",
    "controls_model_activation",
    "controls_model_promotion",
}
_METHOD_LITERAL_SOURCES = {
    "constant_acceleration.displacement": ("formula denominator 2",),
}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _nonce_digest(nonce: Any) -> Optional[str]:
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        return None
    return text_sha256(nonce)


def valid_request_nonce(nonce: Any) -> bool:
    """Return whether ``nonce`` is eligible for proof and replay binding."""

    return _nonce_digest(nonce) is not None


def _percentage_representations(value: str) -> Tuple[str, ...]:
    try:
        with localcontext() as context:
            context.prec = 50
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                decimal = Decimal(numerator) / Decimal(denominator)
            else:
                decimal = Decimal(value.replace(",", ""))
            percentage = decimal * 100
            rendered: List[str] = []
            for places in range(13):
                candidate = format(percentage, f".{places}f").rstrip("0").rstrip(".")
                candidate = candidate or "0"
                if candidate not in rendered:
                    rendered.append(candidate)
            return tuple(rendered)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return ()


def _answer_parts(
    grounded: Mapping[str, Any],
) -> Optional[Tuple[str, str, str, str, Tuple[str, ...]]]:
    receipt = grounded.get("answer_receipt")
    if not isinstance(receipt, Mapping):
        return None
    arithmetic = grounded.get("arithmetic")
    reasoning = grounded.get("reasoning")
    if grounded.get("reason") == "explicit_arithmetic_exact":
        if not isinstance(arithmetic, Mapping):
            return None
        display = arithmetic.get("display")
        unit = ""
        raw_representations = (
            arithmetic.get("display"),
            arithmetic.get("exact"),
            arithmetic.get("approximation"),
        )
    else:
        answer = reasoning.get("answer") if isinstance(reasoning, Mapping) else None
        if not isinstance(answer, Mapping):
            return None
        display = answer.get("display")
        unit = answer.get("unit") or ""
        raw_representations = (
            answer.get("display"),
            answer.get("exact"),
            answer.get("approximation"),
        )
    if not isinstance(display, str) or not display or not isinstance(unit, str):
        return None
    representations: List[str] = []
    for value in raw_representations:
        if isinstance(value, str) and value and value not in representations:
            representations.append(value)
    if str(receipt.get("problem_class") or "") == "probability":
        for value in tuple(representations):
            for percentage in _percentage_representations(value):
                if percentage not in representations:
                    representations.append(percentage)
    return (
        display,
        unit,
        str(receipt.get("problem_class") or ""),
        str(receipt.get("method") or ""),
        tuple(representations),
    )


def _numeric_claims(
    query: str,
    output_text: str,
    display_answer: str,
    verified_representations: Tuple[str, ...],
    unit: str,
    derivation_literals: Tuple[str, ...],
    answer_start: int,
    answer_end: int,
) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    # Unicode numeric confusables are not silently normalized into verified
    # claims.  The trusted render path currently emits ASCII numeric forms.
    if any(char.isnumeric() and char not in "0123456789" for char in output_text):
        return None

    verified_tokens = {
        token
        for representation in verified_representations
        for token in _NUMERIC_SPAN_RE.findall(representation)
    }
    unit_tokens = set(_NUMERIC_SPAN_RE.findall(unit))
    derivation_tokens = {
        token
        for literal in derivation_literals
        for token in _NUMERIC_SPAN_RE.findall(literal)
    }
    query_tokens = set(_NUMERIC_SPAN_RE.findall(query))
    if not verified_tokens:
        return None
    claims: List[Dict[str, Any]] = []
    derived_count = 0

    for index, match in enumerate(_NUMERIC_SPAN_RE.finditer(output_text), start=1):
        token = match.group(0)
        span = {
            "start": match.start(),
            "end": match.end(),
            "sha256": text_sha256(token),
        }
        if (
            answer_start <= match.start()
            and match.end() <= answer_end
            and token in verified_tokens
        ):
            role = "derived_answer"
            source = {
                "kind": "verified_result",
                "sha256": canonical_sha256(list(verified_representations)),
            }
            derived_count += 1
        elif token in unit_tokens:
            role = "verified_unit_literal"
            source = {
                "kind": "verified_result_unit",
                "sha256": text_sha256(unit),
            }
        elif token in query_tokens:
            query_start = query.find(token)
            role = "input_echo"
            source = {
                "kind": "submitted_request",
                "start": query_start,
                "end": query_start + len(token),
                "sha256": text_sha256(token),
            }
        elif token in derivation_tokens:
            role = "verified_derivation_literal"
            source = {
                "kind": "allowlisted_derivation",
                "sha256": canonical_sha256(list(derivation_literals)),
            }
        elif token in verified_tokens:
            role = "derived_answer"
            source = {
                "kind": "verified_result",
                "sha256": canonical_sha256(list(verified_representations)),
            }
            derived_count += 1
        else:
            return None
        claims.append(
            {
                "claim_id": f"numeric-{index}",
                "token": token,
                "role": role,
                "match_policy": "exact_utf8_span",
                "span": span,
                "source": source,
            }
        )

    if not claims or derived_count < 1:
        return None
    coverage = {
        "numeric_span_count": len(claims),
        "verified_numeric_span_count": len(claims),
        "derived_answer_span_count": derived_count,
        "complete": True,
        "unbound_numeric_span_count": 0,
    }
    return claims, coverage


def build_proof_capsule(
    *,
    query: str,
    grounded: Mapping[str, Any],
    receipt_schema_version: str,
    runtime_version: str,
    surface: str,
    request_nonce: str = "",
) -> Optional[Dict[str, Any]]:
    """Build a closed-schema capsule or fail closed when a number is unbound."""

    if not isinstance(query, str) or not query or surface not in epistemics.ANSWER_SURFACES:
        return None
    nonce_sha256 = _nonce_digest(request_nonce)
    if nonce_sha256 is None:
        return None
    if not epistemics.verify_grounded_answer_result(
        grounded,
        receipt_schema_version=receipt_schema_version,
        runtime_version=runtime_version,
    ):
        return None

    output_text = grounded.get("text")
    answer_parts = _answer_parts(grounded)
    receipt = grounded.get("answer_receipt")
    if not isinstance(output_text, str) or answer_parts is None or not isinstance(receipt, Mapping):
        return None
    display_answer, unit, problem_class, method, verified_representations = answer_parts
    reasoning = grounded.get("reasoning")
    derivation_literals = tuple(
        str(value)
        for value in (
            list(reasoning.get("steps") or ())
            if isinstance(reasoning, Mapping)
            else []
        )
        if isinstance(value, str) and value
    ) + tuple(_METHOD_LITERAL_SOURCES.get(method, ()))
    if unit and unit not in output_text:
        return None

    rendered_answer = next(
        (value for value in verified_representations if value in output_text),
        "",
    )
    if not rendered_answer:
        return None
    # The trusted grounding templates place the canonical answer at their final
    # matching occurrence. Earlier equal values can be input echoes (for
    # example, "50 is 100% of what number?") and must not acquire the answer
    # role merely because their token text is identical.
    answer_start = output_text.rfind(rendered_answer)
    answer_end = answer_start + len(rendered_answer)
    claims_and_coverage = _numeric_claims(
        query,
        output_text,
        display_answer,
        verified_representations,
        unit,
        derivation_literals,
        answer_start,
        answer_end,
    )
    if claims_and_coverage is None:
        return None
    claims, coverage = claims_and_coverage
    independent_result = independent_checker.check_certificate(
        query=query,
        display_answer=display_answer,
        problem_class=problem_class,
        method=method,
        unit=unit,
    )
    if (
        independent_result.get("status") != "passed"
        or independent_result.get("algorithmically_independent") is not True
    ):
        # Every renderer capsule requires a successful second implementation
        # witness. Unsupported deterministic families defer until a dedicated
        # checker exists instead of inheriting answer authority from grounding.
        return None
    capsule: Dict[str, Any] = {
        "schema_version": PROOF_CAPSULE_SCHEMA_VERSION,
        "policy_version": PROOF_CAPSULE_POLICY_VERSION,
        "decision": "claim_checked",
        "capsule_is_signature": False,
        "bindings": {
            "request_sha256": text_sha256(query),
            "output_sha256": text_sha256(output_text),
            "display_answer_sha256": text_sha256(display_answer),
            "verifier_receipt_sha256": canonical_sha256(dict(receipt)),
            "request_nonce_sha256": nonce_sha256,
            "surface": surface,
        },
        "result": {
            "display_answer": display_answer,
            "unit": unit,
            "problem_class": problem_class,
            "method": method,
            "verified_representations": list(verified_representations),
            "derivation_literals": list(derivation_literals),
            "answer_span": {
                "start": answer_start,
                "end": answer_end,
                "sha256": text_sha256(rendered_answer),
                "token": rendered_answer,
            },
        },
        "numeric_claims": claims,
        "coverage": coverage,
        "independent_checker": independent_result,
        "verifier": {
            "id": "grounding_runtime.finalize_grounded_response",
            "runtime_version": runtime_version,
            "receipt_schema_version": receipt_schema_version,
            "fresh_renderer_revalidation_required": True,
            "algorithmically_independent": False,
        },
        "authority": {key: False for key in sorted(_AUTHORITY_KEYS)},
        "limitations": [
            "The capsule is a self-checksummed request/result binding, not a signature.",
            "Fresh renderer revalidation reruns the same deterministic implementation; it is not algorithmic independence or empirical calibration.",
            "Only exactly bound numeric spans may receive a verified mark.",
        ],
    }
    capsule["capsule_sha256"] = canonical_sha256(capsule)
    return capsule


def verify_proof_capsule_integrity(
    capsule: Any,
    *,
    query: str,
    output_text: str,
    display_answer: str,
    surface: str,
    request_nonce: str = "",
) -> bool:
    """Validate capsule structure and its direct request/output/span bindings."""

    if not isinstance(capsule, Mapping):
        return False
    row = dict(capsule)
    expected_keys = {
        "schema_version",
        "policy_version",
        "decision",
        "capsule_is_signature",
        "bindings",
        "result",
        "numeric_claims",
        "coverage",
        "independent_checker",
        "verifier",
        "authority",
        "limitations",
        "capsule_sha256",
    }
    if set(row) != expected_keys:
        return False
    supplied_digest = row.pop("capsule_sha256", None)
    if not _is_sha256(supplied_digest) or canonical_sha256(row) != supplied_digest:
        return False
    if (
        row.get("schema_version") != PROOF_CAPSULE_SCHEMA_VERSION
        or row.get("policy_version") != PROOF_CAPSULE_POLICY_VERSION
        or row.get("decision") != "claim_checked"
        or row.get("capsule_is_signature") is not False
    ):
        return False

    nonce_sha256 = _nonce_digest(request_nonce)
    bindings = row.get("bindings")
    result = row.get("result")
    verifier = row.get("verifier")
    authority = row.get("authority")
    coverage = row.get("coverage")
    claims = row.get("numeric_claims")
    independent_result = row.get("independent_checker")
    if nonce_sha256 is None or not all(
        isinstance(value, Mapping)
        for value in (bindings, result, verifier, authority, coverage)
    ) or not isinstance(claims, list):
        return False
    if set(bindings) != {
        "request_sha256",
        "output_sha256",
        "display_answer_sha256",
        "verifier_receipt_sha256",
        "request_nonce_sha256",
        "surface",
    }:
        return False
    if not (
        bindings.get("request_sha256") == text_sha256(query)
        and bindings.get("output_sha256") == text_sha256(output_text)
        and bindings.get("display_answer_sha256") == text_sha256(display_answer)
        and _is_sha256(bindings.get("verifier_receipt_sha256"))
        and bindings.get("request_nonce_sha256") == nonce_sha256
        and surface in epistemics.ANSWER_SURFACES
        and bindings.get("surface") == surface
    ):
        return False
    if set(result) != {
        "display_answer",
        "unit",
        "problem_class",
        "method",
        "verified_representations",
        "derivation_literals",
        "answer_span",
    }:
        return False
    if result.get("display_answer") != display_answer:
        return False
    representations = result.get("verified_representations")
    derivation_literals = result.get("derivation_literals")
    if not isinstance(representations, list) or not representations or not all(
        isinstance(value, str) and value for value in representations
    ) or not isinstance(derivation_literals, list) or not all(
        isinstance(value, str) and value for value in derivation_literals
    ):
        return False
    rendered_answer = next((value for value in representations if value in output_text), "")
    answer_span = result.get("answer_span")
    answer_start = output_text.rfind(rendered_answer)
    answer_end = answer_start + len(rendered_answer)
    if not rendered_answer or not isinstance(answer_span, Mapping) or dict(answer_span) != {
        "start": answer_start,
        "end": answer_end,
        "sha256": text_sha256(rendered_answer),
        "token": rendered_answer,
    }:
        return False
    rebuilt = _numeric_claims(
        query,
        output_text,
        display_answer,
        tuple(representations),
        str(result.get("unit") or ""),
        tuple(derivation_literals),
        answer_start,
        answer_end,
    )
    if rebuilt is None:
        return False
    expected_claims, expected_coverage = rebuilt
    if claims != expected_claims or dict(coverage) != expected_coverage:
        return False
    expected_independent_result = independent_checker.check_certificate(
        query=query,
        display_answer=display_answer,
        problem_class=str(result.get("problem_class") or ""),
        method=str(result.get("method") or ""),
        unit=str(result.get("unit") or ""),
    )
    if not isinstance(independent_result, Mapping) or dict(independent_result) != expected_independent_result:
        return False
    if (
        expected_independent_result.get("status") != "passed"
        or expected_independent_result.get("algorithmically_independent") is not True
    ):
        return False
    if set(verifier) != {
        "id",
        "runtime_version",
        "receipt_schema_version",
        "fresh_renderer_revalidation_required",
        "algorithmically_independent",
    } or not (
        verifier.get("id") == "grounding_runtime.finalize_grounded_response"
        and verifier.get("fresh_renderer_revalidation_required") is True
        and verifier.get("algorithmically_independent") is False
    ):
        return False
    if set(authority) != _AUTHORITY_KEYS or any(value is not False for value in authority.values()):
        return False
    limitations = row.get("limitations")
    return isinstance(limitations, list) and len(limitations) >= 3 and all(
        isinstance(item, str) and item for item in limitations
    )


__all__ = [
    "PROOF_CAPSULE_POLICY_VERSION",
    "PROOF_CAPSULE_SCHEMA_VERSION",
    "build_proof_capsule",
    "canonical_sha256",
    "text_sha256",
    "valid_request_nonce",
    "verify_proof_capsule_integrity",
]
