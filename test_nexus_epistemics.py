"""Invariant tests for NexusMind evidence-first answer admission receipts."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_epistemics as epistemics


_REQUEST_SHA = "1" * 64
_OUTPUT_SHA = "2" * 64
_VERIFIER_SHA = "3" * 64
_NONCE_SHA = "4" * 64


def _verified_fixture(**overrides):
    values = {
        "reason": "verified_test_fixture",
        "claim_scope": "fixture arithmetic expression only",
        "verifier_id": "grounding_runtime.finalize_grounded_response",
        "request_sha256": _REQUEST_SHA,
        "output_sha256": _OUTPUT_SHA,
        "verifier_receipt_sha256": _VERIFIER_SHA,
        "request_nonce_sha256": _NONCE_SHA,
        "surface": "solve",
    }
    values.update(overrides)
    return epistemics.verified_exact_decision(**values)


def _rehash(receipt):
    payload = copy.deepcopy(receipt)
    payload.pop("receipt_sha256", None)
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def test_verified_exact_receipt_is_bounded_and_self_consistent():
    decision = _verified_fixture(
        protocol={"candidate_count": 1, "verifier_calls": 1},
    )
    receipt = decision.to_dict()

    assert epistemics.verify_epistemic_receipt(receipt)
    assert receipt["decision"] == "answered"
    assert receipt["correctness_confidence"] is None
    assert receipt["confidence_kind"] == "deterministic_assurance_not_probability"
    assert receipt["calibrated"] is False
    assert receipt["receipt_is_authority"] is False
    assert receipt["authority"]["answers_within_claim_scope"] is True
    assert receipt["bindings"]["request_sha256"] == _REQUEST_SHA
    assert receipt["bindings"]["output_sha256"] == _OUTPUT_SHA
    assert all(
        receipt["authority"][key] is False
        for key in (
            "controls_tools",
            "controls_permissions",
            "controls_safety",
            "controls_memory",
            "controls_routes",
            "controls_model_activation",
            "controls_model_promotion",
        )
    )


def test_verified_exact_receipt_requires_a_nonce_digest_for_public_surfaces():
    values = {
        "reason": "missing_nonce_fixture",
        "claim_scope": "fixture arithmetic expression only",
        "verifier_id": "grounding_runtime.finalize_grounded_response",
        "request_sha256": _REQUEST_SHA,
        "output_sha256": _OUTPUT_SHA,
        "verifier_receipt_sha256": _VERIFIER_SHA,
        "surface": "solve",
        "request_nonce_sha256": "",
    }

    with pytest.raises(ValueError, match="nonce binding"):
        epistemics.verified_exact_decision(**values)

    receipt = _verified_fixture().to_dict()
    receipt["bindings"]["request_nonce_sha256"] = ""
    assert epistemics.verify_epistemic_receipt(_rehash(receipt)) is False


@pytest.mark.parametrize("factory", [epistemics.analysis_only_decision, epistemics.abstained_decision])
def test_non_authoritative_decisions_never_publish_correctness_confidence(factory):
    decision = factory(
        reason="fixture_without_verification",
        claim_scope="fixture analysis only",
        limitations=("No eligible verifier ran.",),
    )
    receipt = decision.to_dict()

    assert epistemics.verify_epistemic_receipt(receipt)
    assert receipt["answer_authority"] is False
    assert receipt["correctness_confidence"] is None
    assert receipt["authority"]["answers_within_claim_scope"] is False


def test_receipt_tampering_fails_closed():
    receipt = epistemics.abstained_decision(
        reason="fixture_abstention",
        claim_scope="fixture",
        limitations=("No eligible verifier ran.",),
    ).to_dict()

    tampered = copy.deepcopy(receipt)
    tampered["decision"] = "answered"
    tampered["answer_authority"] = True
    tampered["correctness_confidence"] = 1.0

    assert epistemics.verify_epistemic_receipt(tampered) is False

    # A self-checksum is not a signature. Even if an untrusted caller recomputes
    # it, schema invariants must reject a non-boolean authority field.
    rehashed = copy.deepcopy(receipt)
    rehashed["answer_authority"] = None
    rehashed = _rehash(rehashed)
    assert epistemics.verify_epistemic_receipt(rehashed) is False


def test_rehashed_answered_receipt_requires_the_allowlisted_passing_verifier():
    receipt = _verified_fixture().to_dict()

    for verifier in (
        {
            "id": "none",
            "passed": False,
            "fresh_recompute": False,
            "algorithmically_independent": False,
        },
        {
            "id": "attacker.fake_verifier",
            "passed": True,
            "fresh_recompute": True,
            "algorithmically_independent": False,
        },
        {
            "id": "grounding_runtime.finalize_grounded_response",
            "passed": False,
            "fresh_recompute": True,
            "algorithmically_independent": False,
        },
    ):
        forged = copy.deepcopy(receipt)
        forged["verifier"] = verifier
        assert epistemics.verify_epistemic_receipt(_rehash(forged)) is False


def test_constructor_rejects_confidence_without_answer_authority():
    with pytest.raises(ValueError, match="cannot publish correctness confidence"):
        epistemics.EpistemicDecision(
            decision="analysis_only",
            evidence_class="deterministic_heuristic",
            reason="invalid_fixture",
            claim_scope="fixture",
            answer_authority=False,
            correctness_confidence=0.9,
            limitations=("Invalid by construction.",),
        )


def test_rehashed_non_authoritative_receipts_require_fail_closed_invariants():
    analysis = epistemics.analysis_only_decision(
        reason="fixture_analysis",
        claim_scope="fixture",
        internal_score=0.5,
        internal_score_name="template_priority",
        limitations=("No verifier ran.",),
    ).to_dict()
    abstained = epistemics.abstained_decision(
        reason="fixture_abstention",
        claim_scope="fixture",
        limitations=("No verifier ran.",),
    ).to_dict()

    forged_rows = []
    forged = copy.deepcopy(analysis)
    forged["verifier"] = {
        "id": "grounding_runtime.finalize_grounded_response",
        "passed": True,
        "fresh_recompute": True,
        "algorithmically_independent": False,
    }
    forged_rows.append(forged)
    forged = copy.deepcopy(analysis)
    forged["evidence_class"] = "verified_exact"
    forged_rows.append(forged)
    forged = copy.deepcopy(abstained)
    forged["evidence_class"] = "deterministic_heuristic"
    forged_rows.append(forged)
    forged = copy.deepcopy(abstained)
    forged["internal_score"] = 0.9
    forged["internal_score_name"] = "hidden_score"
    forged_rows.append(forged)
    forged = copy.deepcopy(abstained)
    forged["internal_score_name"] = "orphaned_score_label"
    forged_rows.append(forged)
    forged = copy.deepcopy(abstained)
    forged["limitations"] = ["valid", 7]
    forged_rows.append(forged)

    assert all(
        epistemics.verify_epistemic_receipt(_rehash(row)) is False
        for row in forged_rows
    )


@pytest.mark.parametrize("field", ["request_sha256", "output_sha256", "verifier_receipt_sha256"])
def test_verified_receipt_rejects_detached_or_replayed_bindings(field):
    receipt = _verified_fixture().to_dict()
    forged = copy.deepcopy(receipt)
    forged["bindings"][field] = "f" * 64

    # The self-checksum can describe a different claim, but the relying
    # consumer must match it to its own request/output context.
    assert epistemics.verify_epistemic_receipt(_rehash(forged)) is True
    assert epistemics.verify_epistemic_receipt_binding(
        _rehash(forged),
        request_sha256=_REQUEST_SHA,
        output_sha256=_OUTPUT_SHA,
        verifier_receipt_sha256=_VERIFIER_SHA,
        request_nonce_sha256=_NONCE_SHA,
        surface="solve",
    ) is False


def test_closed_schema_rejects_unknown_authority_or_verifier_fields_even_when_rehashed():
    receipt = _verified_fixture().to_dict()
    for section, key in (("authority", "controls_future"), ("verifier", "future_claim")):
        forged = copy.deepcopy(receipt)
        forged[section][key] = True
        assert epistemics.verify_epistemic_receipt(_rehash(forged)) is False
