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
    decision = epistemics.verified_exact_decision(
        reason="verified_test_fixture",
        claim_scope="fixture arithmetic expression only",
        verifier_id="grounding_runtime.finalize_grounded_response",
        protocol={"candidate_count": 1, "verifier_calls": 1},
    )
    receipt = decision.to_dict()

    assert epistemics.verify_epistemic_receipt(receipt)
    assert receipt["decision"] == "answered"
    assert receipt["correctness_confidence"] == 1.0
    assert receipt["confidence_kind"] == "deterministic_in_scope"
    assert receipt["calibrated"] is False
    assert receipt["receipt_is_authority"] is False
    assert receipt["authority"]["answers_within_claim_scope"] is True
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
    receipt = epistemics.verified_exact_decision(
        reason="verified_test_fixture",
        claim_scope="fixture arithmetic expression only",
        verifier_id="grounding_runtime.finalize_grounded_response",
    ).to_dict()

    for verifier in (
        {"id": "none", "passed": False, "independent_recompute": False},
        {"id": "attacker.fake_verifier", "passed": True, "independent_recompute": True},
        {
            "id": "grounding_runtime.finalize_grounded_response",
            "passed": False,
            "independent_recompute": True,
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
        "independent_recompute": True,
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

    assert all(
        epistemics.verify_epistemic_receipt(_rehash(row)) is False
        for row in forged_rows
    )
