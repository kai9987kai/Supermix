"""Comprehensive test suite for Supermix v89 Frontier Epistemic Hybrid.

Verifies:
1. Epistemic Active Inference & Karl Friston Expected Free Energy (EFE) minimization
2. Neuro-Symbolic Proof Verification & First-Error Localization (FEL) with symbolic repair
3. Bidirectional Speculative Decoding & Inverse Equation Verification
4. Epistemic Monte Carlo Tree Search (MCTS)
5. NexusEngine v89 convenience interfaces
6. REST API v89 endpoints & capability telemetry
"""

import sys
from pathlib import Path

# Ensure source is on python search path
repo_root = Path(__file__).resolve().parent
source_dir = repo_root / "source"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

import pytest
from starlette.testclient import TestClient

from nexus_active_inference import (
    ActiveInferenceController,
    ActiveInferenceResult,
    ReasoningActionType,
)
from nexus_api import create_app
from nexus_engine import (
    EpistemicTreeSearchResult,
    NexusEngine,
    build_default_engine,
)
from nexus_proof_verification import (
    FirstErrorLocalizer,
    FirstErrorResult,
    ProofErrorCategory,
)
from nexus_speculative_bidirectional import (
    BidirectionalSpeculationResult,
    BidirectionalSpeculativeDraftEngine,
)


# ---------------------------------------------------------------------------
# Component 1: Active Inference & Expected Free Energy
# ---------------------------------------------------------------------------


def test_active_inference_controller_efe_calculation():
    controller = ActiveInferenceController(base_temperature=0.7, epistemic_weight=1.0)
    beta = controller.compute_precision_beta(rsi=50.0, local_entropy=0.85)
    assert 0.5 <= beta <= 5.0

    res = controller.decide(
        query="Solve 15 * 8 then add 40",
        current_trace_steps=["15 * 8 = 120"],
        local_entropy=0.8,
        rsi_volatility=55.0,
        verification_confidence=0.90,
    )
    assert isinstance(res, ActiveInferenceResult)
    assert len(res.candidate_actions) >= 4
    # Selection probabilities sum to ~1.0
    prob_sum = sum(a.selection_probability for a in res.candidate_actions)
    assert pytest.approx(prob_sum, abs=0.02) == 1.0

    # Selected action should match the candidate with maximum probability
    best_cand = max(res.candidate_actions, key=lambda a: a.selection_probability)
    assert res.selected_action.action_type == best_cand.action_type
    assert res.precision_beta > 0.0
    assert res.epistemic_pragmatic_ratio > 0.0


def test_active_inference_low_confidence_backtrack_preference():
    controller = ActiveInferenceController(base_temperature=0.7)
    # When verification confidence is critically low (e.g. 0.1), BACKTRACK_PRUNE pragmatic risk is lowest
    p_risk, e_gain, g = controller.evaluate_expected_free_energy(
        ReasoningActionType.BACKTRACK_PRUNE,
        step_index=2,
        local_entropy=1.2,
        verification_confidence=0.1,
        has_pending_subgoals=False,
    )
    assert p_risk < 1.0
    assert g < 0.0  # Favorable negative free energy for pruning broken branches


# ---------------------------------------------------------------------------
# Component 2: Neuro-Symbolic Proof Verification & First-Error Localization
# ---------------------------------------------------------------------------


def test_first_error_localization_valid_proof():
    localizer = FirstErrorLocalizer()
    problem = "Calculate 10 + 20 + 12"
    trace = [
        "10 + 20 = 30",
        "30 + 12 = 42",
        "total 42",
    ]
    res = localizer.verify_and_localize(problem, trace)
    assert isinstance(res, FirstErrorResult)
    assert res.has_error is False
    assert res.first_error_index == -1
    assert res.error_category == ProofErrorCategory.NONE
    assert res.proof_fidelity_score == 1.0
    assert res.verified_final_answer == "42.0"


def test_first_error_localization_arithmetic_deviation():
    localizer = FirstErrorLocalizer()
    problem = "Find 25 + 35 then add 10"
    trace = [
        "25 + 35 = 60",
        "60 + 10 = 80",  # Error: 60 + 10 = 70, not 80
        "total 80",
    ]
    res = localizer.verify_and_localize(problem, trace)
    assert res.has_error is True
    assert res.first_error_index == 1
    assert res.error_category == ProofErrorCategory.ARITHMETIC_ERROR
    assert "80" in res.error_step_text
    # Check that symbolic repair corrected the step
    repaired_step = res.repaired_trace[1]
    assert "60 + 10 = 70" in repaired_step
    assert res.proof_fidelity_score < 1.0


def test_first_error_localization_phantom_register():
    localizer = FirstErrorLocalizer()
    problem = "Calculate 10 + 20"
    trace = [
        "10 + 777 = 787",  # 777 is not in problem premises or prior registers
    ]
    res = localizer.verify_and_localize(problem, trace)
    assert res.has_error is True
    assert res.first_error_index == 0
    assert res.error_category == ProofErrorCategory.PHANTOM_REGISTER
    assert "phantom" in res.diagnostic_explanation.lower()


# ---------------------------------------------------------------------------
# Component 3: Bidirectional Speculative Inversion
# ---------------------------------------------------------------------------


def test_bidirectional_speculation_physics_accepted():
    engine = BidirectionalSpeculativeDraftEngine(acceptance_threshold=0.90)
    problem = "mass 12 kg velocity 5 m/s find kinetic energy"
    # KE = 0.5 * 12 * 25 = 150 J
    res = engine.speculate_and_verify(problem, candidate_answer="150")
    assert isinstance(res, BidirectionalSpeculationResult)
    assert res.is_accepted is True
    assert res.consistency_score >= 0.95
    assert "sqrt(2 * KE / mass)" in res.reverse_draft
    assert "velocity = 5" in res.reverse_inferred_premise


def test_bidirectional_speculation_arithmetic_accepted():
    engine = BidirectionalSpeculativeDraftEngine(acceptance_threshold=0.90)
    problem = "What is 45 * 6?"
    # 45 * 6 = 270
    res = engine.speculate_and_verify(problem, candidate_answer="270")
    assert res.is_accepted is True
    assert res.consistency_score >= 0.99
    assert "Reverse: 270 / 6" in res.reverse_draft


def test_bidirectional_speculation_hallucination_rejected():
    engine = BidirectionalSpeculativeDraftEngine(acceptance_threshold=0.90)
    problem = "mass 10 kg acceleration 3 m/s^2 find force"
    # True force is 30 N. Hallucinated answer is 90 N.
    res = engine.speculate_and_verify(problem, candidate_answer="90")
    assert res.is_accepted is False
    assert res.consistency_score < 0.50
    assert res.rejection_reason is not None


# ---------------------------------------------------------------------------
# Component 4: Epistemic Monte Carlo Tree Search
# ---------------------------------------------------------------------------


def test_epistemic_tree_search_execution():
    engine = build_default_engine()
    res = engine.run_epistemic_tree_search(
        query="Calculate 20 + 40 then add 30",
        max_depth=3,
        beam_width=2,
    )
    assert isinstance(res, EpistemicTreeSearchResult)
    assert res.total_nodes_evaluated >= 3
    assert len(res.all_nodes) >= 3
    assert len(res.optimal_trace) >= 1
    assert "valid_nodes_count" in res.telemetry


# ---------------------------------------------------------------------------
# Component 5: NexusEngine Convenience Methods
# ---------------------------------------------------------------------------


def test_nexus_engine_v89_convenience_methods():
    engine = NexusEngine()

    # Active Inference
    ai_res = engine.evaluate_active_inference(
        query="Test query",
        current_trace_steps=["step 1"],
        local_entropy=0.7,
        rsi_volatility=52.0,
    )
    assert isinstance(ai_res, ActiveInferenceResult)

    # First Error Localization
    fel_res = engine.locate_first_error(
        problem="Compute 12 + 18",
        trace_steps=["12 + 18 = 30", "total 30"],
    )
    assert isinstance(fel_res, FirstErrorResult)
    assert fel_res.has_error is False

    # Bidirectional Speculation
    bidi_res = engine.verify_bidirectional_speculation(
        problem="What is 20 + 30?",
        candidate_answer="50",
    )
    assert isinstance(bidi_res, BidirectionalSpeculationResult)
    assert bidi_res.is_accepted is True

    # Epistemic Tree Search
    mcts_res = engine.run_epistemic_tree_search("Compute 5 + 5", max_depth=2, beam_width=2)
    assert isinstance(mcts_res, EpistemicTreeSearchResult)


# ---------------------------------------------------------------------------
# Component 6: REST API v89 Integration
# ---------------------------------------------------------------------------


def test_api_v89_endpoints_integration():
    app = create_app()
    client = TestClient(app)

    # 1. Health Endpoint
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert "v89" in resp_health.json()["service"]

    # 2. Signals Telemetry
    resp_signals = client.get("/v1/signals")
    assert resp_signals.status_code == 200
    sig_data = resp_signals.json()
    assert "v89_frontier_epistemic" in sig_data
    assert sig_data["v89_frontier_epistemic"]["active_inference_controller"] is True
    assert sig_data["v89_frontier_epistemic"]["proof_first_error_localizer"] is True

    # 3. Active Inference Endpoint
    resp_ai = client.post(
        "/v1/active_inference/decide",
        json={"query": "Solve 10 + 20", "current_trace_steps": []},
    )
    assert resp_ai.status_code == 200
    ai_data = resp_ai.json()
    assert "selected_action" in ai_data
    assert "precision_beta" in ai_data

    # 4. Proof Verify Endpoint
    resp_pv = client.post(
        "/v1/proof/verify_steps",
        json={
            "problem": "Calculate 15 + 25",
            "trace_steps": ["15 + 25 = 40", "total 40"],
        },
    )
    assert resp_pv.status_code == 200
    pv_data = resp_pv.json()
    assert pv_data["has_error"] is False
    assert pv_data["proof_fidelity_score"] == 1.0

    # 5. Bidirectional Speculation Endpoint
    resp_bidi = client.post(
        "/v1/speculative/bidirectional",
        json={"problem": "What is 10 * 20?", "candidate_answer": "200"},
    )
    assert resp_bidi.status_code == 200
    bidi_data = resp_bidi.json()
    assert bidi_data["is_accepted"] is True
    assert bidi_data["consistency_score"] >= 0.95

    # 6. Epistemic MCTS Endpoint
    resp_mcts = client.post(
        "/v1/mcts/epistemic_search",
        json={"query": "Compute 10 + 20", "max_depth": 2, "beam_width": 2},
    )
    assert resp_mcts.status_code == 200
    mcts_data = resp_mcts.json()
    assert mcts_data["total_nodes_evaluated"] >= 1
    assert "optimal_trace" in mcts_data
