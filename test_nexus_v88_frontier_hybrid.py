"""Unit and integration tests for NexusMiMo-DemLab v88 Frontier Hybrid.

Covers:
1. MechanisticCircuitProber (Direct Logit Attribution & Activation Patching)
2. CausalRegisterValidator (Shih et al. 2026 Scratchpad Causality)
3. AlgorithmicComplexityAnalyzer (Shannon entropy, Lempel-Ziv compressibility, NCD, Loop detector)
4. AdaptiveContinuousLoopEngine (Auto-research loop, RSI safeguards, Q-learning updates)
5. SemanticInvariantEngine (GSM-Symbolic semantic perturbation & minimal contrast pairs)
6. NexusEngine v88 convenience methods
7. NexusApiService & FastAPI endpoints (/v1/circuits/attribute, /v1/complexity/analyze,
   /v1/autoloop/step, /v1/semantic/invariants, /v1/signals, /health, /studio)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import pytest
from starlette.testclient import TestClient

import nexus_api as api
from nexus_engine import (
    NexusEngine,
    NexusConfig,
    QLearningPolicyEngine,
    CircuitComponentScore,
    ActivationPatchResult,
    CausalRegisterResult,
    MechanisticCircuitProber,
    CausalRegisterValidator,
    ComplexityProfileResult,
    NCDResult,
    AlgorithmicComplexityAnalyzer,
    AutoLoopStepResult,
    AdaptiveContinuousLoopEngine,
    SemanticInvariantResult,
    SemanticInvariantEngine,
)


# ---------------------------------------------------------------------------
# 1. Mechanistic Circuit Prober & Activation Patching
# ---------------------------------------------------------------------------


def test_circuit_attribution():
    prober = MechanisticCircuitProber(n_layers=6, n_heads=4)
    scores = prober.attribute_circuit(
        prompt="Calculate the sum 15 + 27 =",
        target_token="42",
        contrast_token="32",
    )
    assert len(scores) == 6 * 4 + 6  # 24 heads + 6 MLPs = 30 components
    roles = {s.circuit_role for s in scores}
    assert "arithmetic_core" in roles
    assert "induction" in roles

    critical = [s for s in scores if s.is_causally_critical]
    assert len(critical) > 0
    for c in critical:
        assert c.attribution_score > 0.2


def test_activation_patching():
    prober = MechanisticCircuitProber(n_layers=6, n_heads=4)
    res = prober.patch_activation(
        clean_prompt="Calculate 15 + 27 = 42",
        corrupt_prompt="Calculate 99 + 27 = 126",
        target_token="42",
        layer_to_patch=3,
        head_to_patch=2,
    )
    assert isinstance(res, ActivationPatchResult)
    assert res.target_token == "42"
    assert 0.0 <= res.logit_recovery_ratio <= 1.0
    assert res.patched_logit > res.corrupt_logit
    assert res.intervened_component == "L3H2"
    assert res.patch_success is True


# ---------------------------------------------------------------------------
# 2. Causal Register Validation (Shih et al. 2026)
# ---------------------------------------------------------------------------


def test_causal_register_faithful_scratchpad():
    validator = CausalRegisterValidator()
    res = validator.validate_scratchpad_causality(
        problem="Calculate 10 + 20 + 12",
        trace_steps=["10 + 20 = 30", "30 + 12 = 42"],
        next_operation="Answer is 42",
    )
    assert isinstance(res, CausalRegisterResult)
    assert res.causally_faithful is True
    assert res.faithfulness_score >= 0.80
    assert res.counterfactual_sensitivity >= 0.80
    assert res.shortcut_circuit_detected is False
    assert "30" in res.clean_trace
    assert "52" in res.counterfactual_trace
    assert "52" in res.counterfactual_continuation


def test_causal_register_structured_sequence():
    validator = CausalRegisterValidator()
    res = validator.validate_scratchpad_causality(
        problem="Verify hypothesis",
        trace_steps=["Step A verified", "Step B confirmed"],
        next_operation="Conclude hypothesis is true",
    )
    assert res.causally_faithful is True
    assert res.task_family == "logical_sequence"
    assert res.shortcut_circuit_detected is False


# ---------------------------------------------------------------------------
# 3. Algorithmic Complexity & Multi-Source Entropy
# ---------------------------------------------------------------------------


def test_complexity_shannon_and_compressibility():
    analyzer = AlgorithmicComplexityAnalyzer(window_size=6)
    text = (
        "Kinetic energy formula KE = 0.5 * m * v^2 is applied with mass m = 10 kg "
        "and velocity v = 6 m/s to compute 180 Joules."
    )
    res = analyzer.analyze_sequence(text)
    assert isinstance(res, ComplexityProfileResult)
    assert res.total_tokens > 10
    assert res.shannon_entropy_bits > 2.0
    assert 0.0 <= res.normalized_entropy <= 1.0
    assert 0.0 < res.compression_ratio < 1.0
    assert res.repetitive_loop_detected is False
    assert res.entropy_collapse_detected is False
    assert res.regime in ("balanced_information", "high_entropy_noise")


def test_complexity_loop_and_collapse_detection():
    analyzer = AlgorithmicComplexityAnalyzer(window_size=4)
    degenerate = "repeat loop error repeat loop error repeat loop error repeat loop error repeat loop error repeat loop error"
    res = analyzer.analyze_sequence(degenerate)
    assert res.repetitive_loop_detected is True
    assert res.regime == "collapsed_repetition"


def test_ncd_distance_metric():
    analyzer = AlgorithmicComplexityAnalyzer()
    text_a = "The kinetic energy is 180 Joules calculated using velocity."
    text_b = "Kinetic energy of 180 Joules computed from velocity and mass."
    text_c = "A Shakespearean sonnet written under the pale moonlight."

    res_close = analyzer.compute_ncd(text_a, text_b)
    res_distant = analyzer.compute_ncd(text_a, text_c)

    assert isinstance(res_close, NCDResult)
    assert 0.0 <= res_close.ncd_score <= 1.0
    assert 0.0 <= res_distant.ncd_score <= 1.0
    assert res_close.ncd_score < res_distant.ncd_score
    assert res_close.semantic_divergence_class in ("closely_aligned", "near_duplicate", "distinct_approaches")


# ---------------------------------------------------------------------------
# 4. Adaptive Continuous Auto-Loop & Q-Learning
# ---------------------------------------------------------------------------


def test_autoloop_step_execution_and_q_learning():
    engine = NexusEngine()
    loop_engine = AdaptiveContinuousLoopEngine(engine)

    res = loop_engine.step(
        current_query="Analyze sparse MoE routing efficiency under dynamic cognitive budgets",
        reward_feedback=0.9,
    )
    assert isinstance(res, AutoLoopStepResult)
    assert res.iteration == 1
    assert res.selected_mode in QLearningPolicyEngine.ACTIONS
    assert res.reward_awarded == 0.9
    assert res.q_value_updated > 0.0
    assert 0.0 <= res.rsi_value <= 100.0
    assert res.loop_status in ("continue", "stabilized", "throttled_divergence")
    assert "query_hash" in res.step_receipt


# ---------------------------------------------------------------------------
# 5. Semantic Invariant Evaluation (GSM-Symbolic)
# ---------------------------------------------------------------------------


def test_semantic_invariants_addition():
    engine = NexusEngine()
    inv_engine = SemanticInvariantEngine(engine)
    res = inv_engine.evaluate_invariants(
        problem="What is 17 + 25?",
        ground_truth_answer="42",
        task_type="arithmetic",
    )
    assert isinstance(res, SemanticInvariantResult)
    assert res.canonical_problem == "What is 17 + 25?"
    assert res.canonical_answer == "42"
    assert "42" in res.canonical_answer
    assert res.contrast_expected_answer != res.canonical_answer
    assert res.contrast_distinction_passed is True
    assert res.all_equivalent_consistent is True
    assert res.invariance_score == 1.0
    assert res.stability_classification == "robust_understanding"
    assert len(res.variants_evaluated) >= 3


# ---------------------------------------------------------------------------
# 6. NexusEngine v88 Convenience Methods
# ---------------------------------------------------------------------------


def test_nexus_engine_v88_convenience_methods():
    engine = NexusEngine()

    # 1. Circuit attribution
    attrs = engine.run_circuit_attribution("Calculate 2 + 2 =", "4")
    assert len(attrs) > 10

    # 2. Activation patching
    patch = engine.run_activation_patching("Calculate 2 + 2 = 4", "Calculate 9 + 2 = 11", "4", layer_to_patch=2)
    assert patch.target_token == "4"

    # 3. Causal register check
    causal = engine.run_causal_register_check("5 + 5 = 10", ["5 + 5 = 10"], "Answer 10")
    assert causal.causally_faithful is True

    # 4. Complexity analysis
    comp = engine.run_complexity_analysis("Scientific reasoning about thermodynamics.")
    assert comp.total_tokens > 0

    # 5. NCD comparison
    ncd = engine.run_ncd_comparison("Alpha state", "Beta state")
    assert ncd.ncd_score >= 0.0

    # 6. Auto-loop step
    loop = engine.run_autoloop_step("Autonomous research hypothesis", reward_feedback=0.5)
    assert loop.iteration >= 1

    # 7. Semantic invariants
    inv = engine.run_semantic_invariant_eval("What is 10 + 20?", "30")
    assert inv.invariance_score > 0.5


# ---------------------------------------------------------------------------
# 7. NexusApiService & FastAPI Endpoints
# ---------------------------------------------------------------------------


def test_api_v88_endpoints_integration():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = api.create_app(api.NexusApiService())
        client = TestClient(app)

    # 1. /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert any(v in r_health.json()["service"] for v in ("v88", "v89"))

    # 2. /v1/signals
    r_signals = client.get("/v1/signals")
    assert r_signals.status_code == 200
    s_data = r_signals.json()
    assert "v88_frontier_hybrid" in s_data
    assert s_data["v88_frontier_hybrid"]["mechanistic_circuit_prober"] is True
    assert s_data["v88_frontier_hybrid"]["complexity_analyzer"] is True

    # 3. /v1/circuits/attribute
    r_circuits = client.post(
        "/v1/circuits/attribute",
        json={
            "prompt": "Calculate 10 + 20 =",
            "target_token": "30",
            "clean_prompt": "Calculate 10 + 20 = 30",
            "corrupt_prompt": "Calculate 99 + 20 = 119",
            "patch_layer": 3,
            "patch_head": 2,
        },
    )
    assert r_circuits.status_code == 200
    c_data = r_circuits.json()
    assert len(c_data["components"]) > 10
    assert c_data["activation_patch"]["patch_success"] is True

    # 4. /v1/complexity/analyze
    r_comp = client.post(
        "/v1/complexity/analyze",
        json={
            "text": "Thermodynamics heat engine efficiency calculation step.",
            "compare_text": "Carnot engine ideal thermodynamic cycle steps.",
        },
    )
    assert r_comp.status_code == 200
    comp_data = r_comp.json()
    assert "profile" in comp_data
    assert "ncd_comparison" in comp_data
    assert comp_data["profile"]["shannon_entropy_bits"] > 0

    # 5. /v1/autoloop/step
    r_loop = client.post(
        "/v1/autoloop/step",
        json={
            "query": "Explore entropy bounds in neural draft decoding",
            "reward_feedback": 0.85,
        },
    )
    assert r_loop.status_code == 200
    loop_data = r_loop.json()
    assert loop_data["iteration"] >= 1
    assert "q_value_updated" in loop_data

    # 6. /v1/semantic/invariants
    r_inv = client.post(
        "/v1/semantic/invariants",
        json={
            "problem": "What is 14 + 28?",
            "ground_truth_answer": "42",
        },
    )
    assert r_inv.status_code == 200
    inv_data = r_inv.json()
    assert inv_data["invariance_score"] == 1.0
    assert inv_data["contrast_distinction_passed"] is True

    # 7. /studio serves frontend HTML
    r_studio = client.get("/studio")
    assert r_studio.status_code == 200
    assert any(brand in r_studio.text for brand in ("NexusMind Studio v88", "NexusMind Studio v89"))
    assert "panel-interpretability" in r_studio.text
    assert "panel-complexity" in r_studio.text
    assert "panel-autoloop" in r_studio.text
    assert "panel-invariants" in r_studio.text
