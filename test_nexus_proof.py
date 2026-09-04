"""Adversarial contract tests for renderer-revalidated Nexus proof capsules."""

from __future__ import annotations

import copy
import json
import sys
import warnings
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import grounding_runtime as grounding
import nexus_api as api
import nexus_proof as proof


ARITHMETIC_QUERY = "What is 2 + 3 * 4?"
SCIENCE_QUERY = (
    "Under constant acceleration with initial velocity 0 m/s, acceleration "
    "9.8 m/s^2, and time 5 s, what is the final velocity?"
)
ARITHMETIC_NONCE = "proof-nonce-arithmetic-0001"
SCIENCE_NONCE = "proof-nonce-science-0001"


def _fresh_grounding(query: str):
    return grounding.finalize_grounded_response("", query)


def _build_capsule(query: str, grounded, nonce: str, surface: str = "solve"):
    return proof.build_proof_capsule(
        query=query,
        grounded=grounded,
        receipt_schema_version=grounding.VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION,
        runtime_version=grounding.GROUNDING_RUNTIME_VERSION,
        surface=surface,
        request_nonce=nonce,
    )


def _rehash(capsule):
    cooked = copy.deepcopy(capsule)
    cooked.pop("capsule_sha256", None)
    cooked["capsule_sha256"] = proof.canonical_sha256(cooked)
    return cooked


def _stateless_service():
    # handle_verify and handle_scientific do not access engine state. Avoid
    # constructing the experimental neural model in these contract-only tests.
    return api.NexusApiService.__new__(api.NexusApiService)


def _verify_with_service(
    *,
    query: str,
    output: str,
    display_answer: str,
    capsule,
    nonce: str,
    surface: str = "solve",
):
    return _stateless_service().handle_verify(
        api.VerifyRequest(
            query=query,
            output=output,
            display_answer=display_answer,
            surface=surface,
            proof_capsule=capsule,
            request_nonce=nonce,
        )
    )


@pytest.fixture(scope="module")
def arithmetic_bundle():
    grounded = _fresh_grounding(ARITHMETIC_QUERY)
    capsule = _build_capsule(ARITHMETIC_QUERY, grounded, ARITHMETIC_NONCE)
    assert capsule is not None
    return grounded, capsule


@pytest.fixture(scope="module")
def science_bundle():
    grounded = _fresh_grounding(SCIENCE_QUERY)
    capsule = _build_capsule(SCIENCE_QUERY, grounded, SCIENCE_NONCE, "scientific")
    assert capsule is not None
    return grounded, capsule


def test_fresh_exact_arithmetic_capsule_binds_every_numeric_claim(arithmetic_bundle):
    grounded, capsule = arithmetic_bundle
    display = capsule["result"]["display_answer"]

    assert grounded["text"] == "The exact result is 14."
    assert display == "14"
    assert capsule["capsule_is_signature"] is False
    assert capsule["verifier"]["algorithmically_independent"] is False
    assert capsule["independent_checker"]["status"] == "passed"
    assert capsule["independent_checker"]["algorithmically_independent"] is True
    assert capsule["coverage"] == {
        "numeric_span_count": 1,
        "verified_numeric_span_count": 1,
        "derived_answer_span_count": 1,
        "complete": True,
        "unbound_numeric_span_count": 0,
    }
    assert proof.verify_proof_capsule_integrity(
        capsule,
        query=ARITHMETIC_QUERY,
        output_text=grounded["text"],
        display_answer=display,
        surface="solve",
        request_nonce=ARITHMETIC_NONCE,
    )


@pytest.mark.parametrize(
    "invalid_nonce",
    [None, "", "short", "non-ascii-caf\u00e9-0001", "x" * 129, "has.periods.0001"],
)
def test_capsule_requires_a_valid_ascii_request_nonce(arithmetic_bundle, invalid_nonce):
    grounded, capsule = arithmetic_bundle

    assert proof.valid_request_nonce(invalid_nonce) is False
    assert _build_capsule(ARITHMETIC_QUERY, grounded, invalid_nonce) is None
    assert not proof.verify_proof_capsule_integrity(
        capsule,
        query=ARITHMETIC_QUERY,
        output_text=grounded["text"],
        display_answer="14",
        surface="solve",
        request_nonce=invalid_nonce,
    )


def test_capsules_reject_nonpublic_surface_identifiers(arithmetic_bundle):
    grounded, capsule = arithmetic_bundle

    assert _build_capsule(
        ARITHMETIC_QUERY,
        grounded,
        ARITHMETIC_NONCE,
        surface="fixture",
    ) is None
    assert not proof.verify_proof_capsule_integrity(
        capsule,
        query=ARITHMETIC_QUERY,
        output_text=grounded["text"],
        display_answer="14",
        surface="fixture",
        request_nonce=ARITHMETIC_NONCE,
    )


def test_fresh_science_capsule_and_renderer_revalidation_succeed(science_bundle):
    grounded, capsule = science_bundle
    display = capsule["result"]["display_answer"]

    assert grounded["reason"] == "verified_reasoning_solution"
    assert capsule["result"]["problem_class"] == "scientific_scenario"
    assert capsule["result"]["method"] == "constant_acceleration.final_velocity"
    assert display == "49"
    assert capsule["result"]["unit"] == "m/s"
    assert capsule["independent_checker"]["status"] == "passed"
    assert capsule["independent_checker"]["checker_id"] == "nexus-independent-science-checker-v1"
    assert proof.verify_proof_capsule_integrity(
        capsule,
        query=SCIENCE_QUERY,
        output_text=grounded["text"],
        display_answer=display,
        surface="scientific",
        request_nonce=SCIENCE_NONCE,
    )

    response = _stateless_service().handle_scientific(
        api.ScientificRequest(query=SCIENCE_QUERY, request_nonce=SCIENCE_NONCE)
    )
    assert response["status"] == "answered"
    assert response["confidence"] is None
    assert response["proof_capsule"] == capsule

    verdict = _verify_with_service(
        query=SCIENCE_QUERY,
        output=response["output"],
        display_answer=display,
        capsule=response["proof_capsule"],
        nonce=SCIENCE_NONCE,
        surface="scientific",
    )
    assert verdict["status"] == "verified"
    assert verdict["renderer_may_mark_numeric_claims_verified"] is True
    assert verdict["confidence"] is None


@pytest.mark.parametrize(
    "query",
    [
        "evaluate 5 / 6",
        "Assuming equally likely outcomes, given 3 favourable outcomes and 8 total outcomes, what is the probability?",
        "Calculate the kinetic energy of a 2 kg object moving at 3 m/s.",
        "Convert 5 km to miles. Show your work.",
        "What is the area of a rectangle with length 8 cm and width 5 cm?",
        "With constant acceleration, an object starts from rest, acceleration is 3 m/s^2, and time is 4 s. Calculate its displacement.",
    ],
)
def test_capsules_require_a_supported_independent_witness(query):
    grounded = _fresh_grounding(query)
    capsule = _build_capsule(query, grounded, ARITHMETIC_NONCE)

    assert grounded["reason"] in {"explicit_arithmetic_exact", "verified_reasoning_solution"}
    problem_class = grounded.get("reasoning", {}).get("problem_class", "")
    if capsule is None:
        # The grounder remains useful diagnostic evidence, but no public proof
        # capsule is admitted until a separate checker covers this family.
        assert problem_class not in {"arithmetic", "scientific_scenario"}
        assert proof.independent_checker.check_certificate(
            query=query,
            display_answer=str(grounded.get("reasoning", {}).get("answer", {}).get("display", "")),
            problem_class=problem_class,
            method=str(grounded.get("reasoning", {}).get("method", "")),
            unit=str(grounded.get("reasoning", {}).get("answer", {}).get("unit", "")),
        )["algorithmically_independent"] is False
        return
    assert capsule is not None
    assert capsule["coverage"]["complete"] is True
    assert capsule["coverage"]["unbound_numeric_span_count"] == 0
    assert proof.verify_proof_capsule_integrity(
        capsule,
        query=query,
        output_text=grounded["text"],
        display_answer=capsule["result"]["display_answer"],
        surface="solve",
        request_nonce=ARITHMETIC_NONCE,
    )


def test_unsupported_percent_reasoning_defers_before_capsule_creation():
    query = "50 is 100% of what number?"
    grounded = _fresh_grounding(query)
    capsule = _build_capsule(query, grounded, ARITHMETIC_NONCE)

    assert grounded["reason"] == "verified_reasoning_solution"
    assert grounded["reasoning"]["problem_class"] == "percent"
    assert capsule is None


def test_appended_contradictory_number_is_rejected_even_after_rehash(arithmetic_bundle):
    grounded, capsule = arithmetic_bundle
    forged_output = grounded["text"] + " However, the final answer is 999."
    forged = copy.deepcopy(capsule)
    forged["bindings"]["output_sha256"] = proof.text_sha256(forged_output)
    forged = _rehash(forged)

    assert not proof.verify_proof_capsule_integrity(
        forged,
        query=ARITHMETIC_QUERY,
        output_text=forged_output,
        display_answer="14",
        surface="solve",
        request_nonce=ARITHMETIC_NONCE,
    )
    verdict = _verify_with_service(
        query=ARITHMETIC_QUERY,
        output=forged_output,
        display_answer="14",
        capsule=forged,
        nonce=ARITHMETIC_NONCE,
    )
    assert verdict["status"] == "rejected"
    assert verdict["renderer_may_mark_numeric_claims_verified"] is False
    assert "999" not in json.dumps(verdict)

    forged_grounding = copy.deepcopy(grounded)
    forged_grounding["text"] = forged_output
    assert _build_capsule(
        ARITHMETIC_QUERY, forged_grounding, ARITHMETIC_NONCE
    ) is None


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("query", "What is 2 + 3 * 5?"),
        ("output", "The exact result is 15."),
        ("display_answer", "15"),
        ("nonce", "proof-nonce-arithmetic-9999"),
    ],
)
def test_changed_request_result_display_or_nonce_fails_renderer_binding(
    arithmetic_bundle, field, replacement
):
    grounded, capsule = arithmetic_bundle
    values = {
        "query": ARITHMETIC_QUERY,
        "output": grounded["text"],
        "display_answer": capsule["result"]["display_answer"],
        "nonce": ARITHMETIC_NONCE,
    }
    values[field] = replacement

    assert not proof.verify_proof_capsule_integrity(
        capsule,
        query=values["query"],
        output_text=values["output"],
        display_answer=values["display_answer"],
        surface="solve",
        request_nonce=values["nonce"],
    )
    verdict = _verify_with_service(
        query=values["query"],
        output=values["output"],
        display_answer=values["display_answer"],
        capsule=capsule,
        nonce=values["nonce"],
    )
    assert verdict["status"] == "rejected"
    assert verdict["fresh_verifier_calls"] == 0
    assert verdict["capsule_sha256"] == ""


def test_recomputed_self_hash_does_not_create_renderer_authority(arithmetic_bundle):
    grounded, capsule = arithmetic_bundle
    tampered = copy.deepcopy(capsule)
    tampered["limitations"][0] = "A recomputed checksum is now claimed as authority."
    tampered = _rehash(tampered)
    unsigned = dict(tampered)
    supplied = unsigned.pop("capsule_sha256")

    # The attacker can recompute this public checksum; only a fresh exact-capsule
    # comparison at /v1/verify prevents that mutation from reaching the renderer.
    assert supplied == proof.canonical_sha256(unsigned)
    verdict = _verify_with_service(
        query=ARITHMETIC_QUERY,
        output=grounded["text"],
        display_answer="14",
        capsule=tampered,
        nonce=ARITHMETIC_NONCE,
    )
    assert verdict["status"] == "rejected"
    assert verdict["renderer_may_mark_numeric_claims_verified"] is False


@pytest.mark.parametrize(
    "location", ["top", "authority", "result", "verifier", "independent_checker"]
)
def test_unknown_capsule_fields_fail_closed(arithmetic_bundle, location):
    grounded, capsule = arithmetic_bundle
    tampered = copy.deepcopy(capsule)
    target = tampered if location == "top" else tampered[location]
    target["attacker_extension"] = True
    tampered = _rehash(tampered)

    assert not proof.verify_proof_capsule_integrity(
        tampered,
        query=ARITHMETIC_QUERY,
        output_text=grounded["text"],
        display_answer="14",
        surface="solve",
        request_nonce=ARITHMETIC_NONCE,
    )
    verdict = _verify_with_service(
        query=ARITHMETIC_QUERY,
        output=grounded["text"],
        display_answer="14",
        capsule=tampered,
        nonce=ARITHMETIC_NONCE,
    )
    assert verdict["status"] == "rejected"


@pytest.mark.parametrize("confusable", ["１４", "¹⁴", "١٤"])
def test_unicode_numeric_confusables_never_receive_a_verified_mark(
    arithmetic_bundle, confusable
):
    _grounded, capsule = arithmetic_bundle
    forged_output = f"The exact result is {confusable}."
    tampered = copy.deepcopy(capsule)
    tampered["bindings"]["output_sha256"] = proof.text_sha256(forged_output)
    tampered["bindings"]["display_answer_sha256"] = proof.text_sha256(confusable)
    tampered["result"]["display_answer"] = confusable
    tampered["result"]["answer_span"] = {
        "start": forged_output.find(confusable),
        "end": forged_output.find(confusable) + len(confusable),
        "sha256": proof.text_sha256(confusable),
    }
    tampered = _rehash(tampered)

    assert not proof.verify_proof_capsule_integrity(
        tampered,
        query=ARITHMETIC_QUERY,
        output_text=forged_output,
        display_answer=confusable,
        surface="solve",
        request_nonce=ARITHMETIC_NONCE,
    )
    verdict = _verify_with_service(
        query=ARITHMETIC_QUERY,
        output=forged_output,
        display_answer=confusable,
        capsule=tampered,
        nonce=ARITHMETIC_NONCE,
    )
    assert verdict["status"] == "rejected"
    assert confusable not in json.dumps(verdict, ensure_ascii=False)


@pytest.mark.parametrize("field", ["runtime_version", "schema_version"])
def test_stale_grounder_runtime_or_receipt_schema_cannot_build_capsule(
    arithmetic_bundle, field
):
    grounded, _capsule = arithmetic_bundle
    stale = copy.deepcopy(grounded)
    stale["answer_receipt"][field] = "stale-or-untrusted-version"

    assert _build_capsule(ARITHMETIC_QUERY, stale, ARITHMETIC_NONCE) is None


@pytest.mark.parametrize("mutation", ["runtime", "receipt_digest"])
def test_fresh_api_revalidation_rejects_stale_capsule_provenance(
    arithmetic_bundle, mutation
):
    grounded, capsule = arithmetic_bundle
    stale = copy.deepcopy(capsule)
    if mutation == "runtime":
        stale["verifier"]["runtime_version"] = "stale-runtime"
    else:
        stale["bindings"]["verifier_receipt_sha256"] = "f" * 64
    stale = _rehash(stale)

    verdict = _verify_with_service(
        query=ARITHMETIC_QUERY,
        output=grounded["text"],
        display_answer="14",
        capsule=stale,
        nonce=ARITHMETIC_NONCE,
    )
    assert verdict["status"] == "rejected"
    assert verdict["renderer_may_mark_numeric_claims_verified"] is False
    assert verdict["capsule_sha256"] == ""


def test_v2_verify_accepts_only_the_fresh_exact_capsule_match(arithmetic_bundle):
    grounded, capsule = arithmetic_bundle
    app = api.create_app(_stateless_service())
    if not hasattr(app, "routes"):
        pytest.skip("FastAPI is not installed")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        client = TestClient(app)
    response = client.post(
        "/v1/verify",
        json={
            "query": ARITHMETIC_QUERY,
            "output": grounded["text"],
            "display_answer": "14",
            "surface": "solve",
            "proof_capsule": capsule,
            "request_nonce": ARITHMETIC_NONCE,
        },
    )

    assert response.status_code == 200
    verdict = response.json()
    assert verdict == {
        "status": "verified",
        "verified": True,
        "reason": "fresh_recompute_exact_capsule_match",
        "confidence": None,
        "assurance_kind": "deterministic_assurance_not_probability",
        "renderer_may_mark_numeric_claims_verified": True,
        "fresh_verifier_calls": 1,
        "capsule_sha256": capsule["capsule_sha256"],
    }
