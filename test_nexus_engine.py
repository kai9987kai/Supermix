"""Tests for the NexusMind Unified Engine."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_engine as engine
import nexus_epistemics as epistemics


def test_engine_initialization():
    eng = engine.build_default_engine()
    assert eng.model is not None
    assert eng.swarm_engine is not None
    assert eng.got_engine is not None
    assert eng.observatory is not None
    assert eng.q_learner is not None


def test_engine_flash_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="What is sliding window attention?", mode="fast")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "fast"
    assert len(res.thought_steps) >= 1
    assert res.confidence is None
    assert res.speculative_acceptance_rate is None
    assert res.epistemics["decision"] == "abstained"
    assert res.epistemics["answer_authority"] is False
    assert epistemics.verify_epistemic_receipt(res.epistemics)
    assert "Fast Flash response" not in res.final_output
    assert res.telemetry["synthetic_observability_probe"]["is_live_quality_evidence"] is False


def test_engine_deep_mode_recursive_ponder():
    eng = engine.build_default_engine()
    res = eng.process(query="Deduce hierarchical routing policy", mode="deep")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "deep"
    assert any(s.stage == "ponder" for s in res.thought_steps)
    assert res.confidence is None
    assert res.epistemics["decision"] == "abstained"
    assert "verified inference" not in res.final_output.lower()


def test_engine_swarm_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="Debate memory authority boundaries", mode="swarm")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "swarm"
    assert "swarm_receipt" in res.audit_receipts
    assert any(s.stage == "swarm_debate" for s in res.thought_steps)
    assert res.confidence is None
    assert res.epistemics["decision"] == "analysis_only"
    assert "not a verified answer" in res.final_output.lower()


def test_engine_got_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="Explore decision tree for routing", mode="got")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "got"
    assert "got_receipt" in res.audit_receipts
    assert res.confidence is None
    assert res.epistemics["decision"] == "analysis_only"


def test_engine_scientific_mode_deterministic_execution():
    eng = engine.build_default_engine()
    q = "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and temperature is 300 K. What is its pressure?"
    res = eng.process(query=q, mode="auto")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected in ("scientific", "solve")
    assert res.confidence == 1.0
    assert res.epistemics["decision"] == "answered"
    assert res.epistemics["answer_authority"] is True
    assert "verified_answer_receipt" in res.audit_receipts
    assert "Pa" in str(res.audit_receipts) or "Pa" in res.final_output


@pytest.mark.parametrize(
    "query",
    [
        "Do not calculate the force when mass is 3 kg and acceleration is 5 m/s^2.",
        "A mass is 3 kg and acceleration is 5 m/s^2. What is the force, and predict Tesla stock tomorrow?",
        "A mass is either 5 kg or 7 kg and acceleration is 3 m/s^2. What force?",
        'The documentation says "calculate force for mass 3 kg and acceleration 5 m/s^2". Explain the sentence without calculating.',
        "A 5 kg solid sphere rolls at 4 m/s without slipping. What is its total kinetic energy?",
        "A 10 N force moves an object 5 m at an angle of 60 degrees. What work is done?",
        "On the Moon, what is the potential energy of a 2 kg object at height 5 m?",
    ],
)
def test_strict_solver_withholds_legacy_false_positive_families(query):
    eng = engine.build_default_engine()
    res = eng.process(query=query, mode="solve")

    assert res.epistemics["decision"] == "abstained"
    assert res.epistemics["answer_authority"] is False
    assert res.confidence is None
    assert res.tool_calls_used == 0
    assert epistemics.verify_epistemic_receipt(res.epistemics)
    legacy_audit = res.audit_receipts["legacy_nexus_solver_audit"]
    assert legacy_audit["answer_withheld"] is True
    assert legacy_audit["full_receipt_withheld"] is True
    assert "receipt" not in legacy_audit


def test_negated_legacy_candidate_is_not_leaked_through_engine_audit_metadata():
    eng = engine.build_default_engine()
    res = eng.process(
        query="Do not calculate the force when mass is 3 kg and acceleration is 5 m/s^2.",
        mode="solve",
    )

    legacy_audit = res.audit_receipts["legacy_nexus_solver_audit"]
    assert legacy_audit["matched"] is True
    assert "15" not in json.dumps(legacy_audit, sort_keys=True)
    assert "raw_result_fraction" not in legacy_audit
    assert "display_result" not in legacy_audit


@pytest.mark.parametrize("mutation", ["not_selected", "verification_failed"])
def test_engine_rejects_malformed_grounder_receipt_without_leaking_candidate(monkeypatch, mutation):
    eng = engine.build_default_engine()
    query = "What is 2 + 3 * 4?"
    forged = json.loads(json.dumps(engine.grounding.finalize_grounded_response("", query)))
    sentinel = "ENGINE_GROUNDER_LEAK_858585"
    forged["text"] = sentinel
    if mutation == "not_selected":
        forged["answer_receipt"]["selected"] = False
    else:
        forged["answer_receipt"]["verification"]["passed"] = False
    monkeypatch.setattr(
        engine.grounding,
        "finalize_grounded_response",
        lambda *_args, **_kwargs: forged,
    )

    res = eng.process(query=query, mode="solve")
    encoded = json.dumps(res.to_dict(), sort_keys=True)

    assert res.epistemics["decision"] == "abstained"
    assert res.confidence is None
    assert "verified_answer_receipt" not in res.audit_receipts
    assert sentinel not in encoded


@pytest.mark.parametrize("failure_kind", ["none", "list", "exception"])
def test_engine_grounder_failures_abstain_without_traceback(monkeypatch, failure_kind):
    eng = engine.build_default_engine()

    if failure_kind == "exception":
        def fail(*_args, **_kwargs):
            raise RuntimeError("GROUNDER_EXCEPTION_SENTINEL")

        replacement = fail
    else:
        replacement = lambda *_args, **_kwargs: None if failure_kind == "none" else []
    monkeypatch.setattr(engine.grounding, "finalize_grounded_response", replacement)

    res = eng.process(query="What is 2 + 3 * 4?", mode="solve")
    encoded = json.dumps(res.to_dict(), sort_keys=True)

    assert res.epistemics["decision"] == "abstained"
    assert res.confidence is None
    assert "GROUNDER_EXCEPTION_SENTINEL" not in encoded


def test_tool_declarations_are_not_counted_as_executions_or_evidence():
    eng = engine.build_default_engine()
    res = eng.process(
        query="Use the weather tool and answer with today's forecast.",
        mode="agent",
        tools=[{"name": "weather", "description": "Weather lookup"}],
    )

    assert res.tool_calls_used == 0
    assert res.telemetry["declared_tool_count"] == 1
    assert res.telemetry["external_tool_calls_executed"] == 0
    assert res.epistemics["decision"] == "abstained"


def test_unverified_runtime_output_does_not_self_train_q_policy():
    eng = engine.build_default_engine()
    before = eng.q_learner.to_dict()

    res = eng.process(query="Explain an open-world topic.", mode="fast")

    assert eng.q_learner.to_dict() == before
    assert res.telemetry["q_learning_update"] == "skipped_requires_external_verified_feedback"


@pytest.mark.parametrize("mode", ["fast", "deep"])
def test_untrained_probe_does_not_publish_decision_head_probabilities(mode):
    eng = engine.build_default_engine()
    res = eng.process(query="Inspect the experimental latent probe.", mode=mode)
    encoded = json.dumps(res.to_dict(), sort_keys=True)

    assert "quality_probability" not in encoded
    assert "continue_probability" not in encoded
    assert res.confidence is None


@pytest.mark.parametrize("mode", ["innovate", "chat", "swarm", "got"])
def test_analysis_only_modes_never_publish_step_correctness_confidence(mode):
    eng = engine.build_default_engine()
    res = eng.process(query="Map evidence-boundary design options.", mode=mode)

    assert res.epistemics["decision"] == "analysis_only"
    assert res.confidence is None
    assert res.thought_steps
    assert all(step.confidence is None for step in res.thought_steps)
