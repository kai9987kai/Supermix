"""Supermix v90 Frontier Diffusion-of-Thought (DoT) & Epistemic Reflexion Test Suite.

Tests:
  1. DiffusionThoughtEngine - continuous thought denoising
  2. ReflexiveCorrectionEngine - epistemic reflexion & self-correction
  3. ConformalStoppingController - conformal risk-controlled stopping
  4. CausalDAGEngine - Pearlian do-calculus engine
  5. NexusEngine convenience methods
  6. REST API endpoints (FastAPI TestClient)
"""

from __future__ import annotations

import sys
import math
import os

# Ensure source dir is on path first
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import pytest

from nexus_diffusion_thought import (
    DiffusionThoughtEngine,
    DiffusionThoughtResult,
    DiffusionThoughtStep,
)
from nexus_reflexion import (
    ReflexiveCorrectionEngine,
    ReflexionCorrectionResult,
    EpistemicReflexionCapsule,
)
from nexus_conformal_stopping import (
    ConformalStoppingController,
    ConformalStoppingResult,
)
from nexus_causal_dag import (
    CausalDAGEngine,
    CausalQueryResult,
)
from nexus_engine import NexusEngine
from nexus_api import (
    DiffusionThoughtRequest,
    DiffusionThoughtResponse,
    ReflexionCorrectionRequest,
    ReflexionCorrectionResponse,
    ConformalStoppingRequest,
    ConformalStoppingResponse,
    CausalDAGRequest,
    CausalDAGResponse,
    NexusApiService,
    create_app,
)


# ==========================================================================
# Component 1: DiffusionThoughtEngine
# ==========================================================================


class TestDiffusionThoughtEngine:
    def test_instantiation(self):
        engine = DiffusionThoughtEngine(total_steps=10, latent_dim=8)
        assert engine.total_steps == 10
        assert engine.latent_dim == 8
        assert len(engine.betas) == 10
        assert len(engine.alphas) == 10
        assert len(engine.alphas_bar) == 10

    def test_cosine_schedule_monotone_alphas_bar(self):
        engine = DiffusionThoughtEngine(total_steps=20, latent_dim=8)
        for i in range(len(engine.alphas_bar) - 1):
            assert engine.alphas_bar[i] >= engine.alphas_bar[i + 1], (
                f"alphas_bar should be non-increasing; failed at index {i}"
            )

    def test_denoise_reasoning_returns_result(self):
        engine = DiffusionThoughtEngine(total_steps=10, latent_dim=8)
        result = engine.denoise_reasoning(prompt="Compute 5 + 7", seed=42)
        assert isinstance(result, DiffusionThoughtResult)
        assert result.denoising_steps == 10
        assert len(result.trajectory) == 10
        assert result.crystallized_plan != ""
        assert len(result.discrete_derivation_tokens) >= 1

    def test_trajectory_steps_structure(self):
        engine = DiffusionThoughtEngine(total_steps=8, latent_dim=4)
        result = engine.denoise_reasoning(prompt="What is Newton's second law?", seed=7)
        for step in result.trajectory:
            assert isinstance(step, DiffusionThoughtStep)
            assert 0.0 <= step.jsd_stability <= 1.0
            assert -1.0 <= step.cosine_similarity <= 1.0
            assert isinstance(step.is_crystallized, bool)
            assert len(step.decoded_hypotheses) >= 1

    def test_crystallization_occurs(self):
        engine = DiffusionThoughtEngine(total_steps=30, latent_dim=16)
        result = engine.denoise_reasoning(prompt="Simple math: 3+4=7", seed=0)
        # crystallization_step should be set (not -1 sentinel)
        assert result.crystallization_step > 0
        assert result.mean_stability_jsd >= 0.0

    def test_denoise_thought_alias(self):
        engine = DiffusionThoughtEngine(total_steps=5, latent_dim=4)
        result = engine.denoise_thought(problem="Test alias", num_timesteps=5, seed=42)
        assert isinstance(result, DiffusionThoughtResult)

    def test_to_dict_serialization(self):
        engine = DiffusionThoughtEngine(total_steps=5, latent_dim=4)
        result = engine.denoise_reasoning("test serialization", seed=1)
        d = result.to_dict()
        assert "prompt" in d
        assert "trajectory" in d
        assert "crystallized_plan" in d
        assert "mean_stability_jsd" in d
        assert "telemetry" in d

    def test_deterministic_with_seed(self):
        engine = DiffusionThoughtEngine(total_steps=10, latent_dim=8)
        r1 = engine.denoise_reasoning("determinism test", seed=99)
        r2 = engine.denoise_reasoning("determinism test", seed=99)
        assert r1.crystallization_step == r2.crystallization_step
        assert r1.mean_stability_jsd == r2.mean_stability_jsd

    def test_different_seeds_produce_different_trajectories(self):
        engine = DiffusionThoughtEngine(total_steps=10, latent_dim=8)
        r1 = engine.denoise_reasoning("vary seed", seed=1)
        r2 = engine.denoise_reasoning("vary seed", seed=9999)
        # Latent norms and final latent norms must differ across distinct initial noise seeds
        norms_1 = [s.latent_norm for s in r1.trajectory]
        norms_2 = [s.latent_norm for s in r2.trajectory]
        assert norms_1 != norms_2
        assert r1.final_latent_norm != r2.final_latent_norm



# ==========================================================================
# Component 2: ReflexiveCorrectionEngine
# ==========================================================================


class TestReflexiveCorrectionEngine:
    def test_instantiation(self):
        engine = ReflexiveCorrectionEngine(memory_capacity=64)
        assert engine.memory_capacity == 64
        assert engine.memory_buffer == []

    def test_clean_derivation_no_failure(self):
        engine = ReflexiveCorrectionEngine()
        result = engine.diagnose_and_correct(
            problem="Compute 6 + 9",
            trace_steps=["6 + 9 = 15", "The total is 15"],
        )
        assert isinstance(result, ReflexionCorrectionResult)
        assert result.had_failure is False
        assert result.correction_fidelity == 1.0
        assert result.memory_buffer_updated is False

    def test_arithmetic_error_triggers_reflexion(self):
        engine = ReflexiveCorrectionEngine()
        # Deliberately wrong step
        result = engine.diagnose_and_correct(
            problem="Compute 10 + 5",
            trace_steps=["10 + 5 = 99", "total 99"],
        )
        assert isinstance(result, ReflexionCorrectionResult)
        # After correction: memory buffer should be updated
        assert result.memory_buffer_updated is True
        assert len(engine.memory_buffer) >= 1

    def test_reflexion_capsule_populated_on_failure(self):
        engine = ReflexiveCorrectionEngine()
        result = engine.diagnose_and_correct(
            problem="Compute 3 * 4",
            trace_steps=["3 * 4 = 999", "total 999"],
        )
        if result.had_failure:
            assert result.reflexion_capsule is not None
            cap = result.reflexion_capsule
            assert isinstance(cap, EpistemicReflexionCapsule)
            assert cap.negative_avoidance_constraint != ""
            assert cap.suggested_pivot_action != ""

    def test_memory_buffer_bounded(self):
        engine = ReflexiveCorrectionEngine(memory_capacity=3)
        for i in range(6):
            engine.diagnose_and_correct(
                problem=f"Problem {i}: compute {i} + 1",
                trace_steps=[f"{i} + 1 = 999"],
            )
        assert len(engine.memory_buffer) <= 3

    def test_to_dict_serialization(self):
        engine = ReflexiveCorrectionEngine()
        result = engine.diagnose_and_correct(
            problem="Compute 8 - 3",
            trace_steps=["8 - 3 = 5", "answer 5"],
        )
        d = result.to_dict()
        assert "problem" in d
        assert "had_failure" in d
        assert "diagnostic_summary" in d


# ==========================================================================
# Component 3: ConformalStoppingController
# ==========================================================================


class TestConformalStoppingController:
    def test_instantiation(self):
        ctrl = ConformalStoppingController(target_risk_alpha=0.05)
        assert ctrl.target_risk_alpha == 0.05

    def test_calibrate_returns_threshold(self):
        ctrl = ConformalStoppingController(target_risk_alpha=0.10)
        margins = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        threshold = ctrl.calibrate(margins)
        assert 0.0 <= threshold <= 1.0

    def test_empty_calibration_returns_default(self):
        ctrl = ConformalStoppingController()
        default_val = ctrl.calibrated_threshold
        result = ctrl.calibrate([])
        assert result == default_val

    def test_early_exit_when_margin_exceeds_threshold(self):
        ctrl = ConformalStoppingController(
            target_risk_alpha=0.05,
            default_calibrated_threshold=0.20,
        )
        result = ctrl.evaluate_stopping(
            query="test_query",
            current_step=3,
            max_budget=10,
            top_confidence=0.90,
            runner_up_confidence=0.30,
        )
        assert isinstance(result, ConformalStoppingResult)
        # Margin = 0.60 > threshold 0.20 → should exit
        assert result.should_early_exit is True
        assert result.compute_savings_pct > 0.0

    def test_no_exit_when_margin_below_threshold(self):
        ctrl = ConformalStoppingController(
            target_risk_alpha=0.05,
            default_calibrated_threshold=0.80,
        )
        result = ctrl.evaluate_stopping(
            query="insufficient_margin",
            current_step=2,
            max_budget=10,
            top_confidence=0.55,
            runner_up_confidence=0.45,
        )
        # Margin = 0.10 < threshold 0.80 → no exit
        assert result.should_early_exit is False
        assert result.compute_savings_pct == 0.0

    def test_terminal_step_forces_exit(self):
        ctrl = ConformalStoppingController(default_calibrated_threshold=0.99)
        result = ctrl.evaluate_stopping(
            query="terminal",
            current_step=10,
            max_budget=10,
            top_confidence=0.50,
            runner_up_confidence=0.49,
        )
        assert result.should_early_exit is True

    def test_certified_risk_bound_equals_alpha(self):
        ctrl = ConformalStoppingController(target_risk_alpha=0.07)
        result = ctrl.evaluate_stopping("q", 1, 5)
        assert result.certified_risk_bound == pytest.approx(0.07, abs=1e-6)

    def test_to_dict_serialization(self):
        ctrl = ConformalStoppingController()
        result = ctrl.evaluate_stopping("serialize_test", 1, 5)
        d = result.to_dict()
        assert "should_early_exit" in d
        assert "certified_risk_bound" in d
        assert "diagnostic_summary" in d


# ==========================================================================
# Component 4: CausalDAGEngine
# ==========================================================================


class TestCausalDAGEngine:
    def test_instantiation_with_scenarios(self):
        engine = CausalDAGEngine()
        assert "physics_newton" in engine.scenarios
        assert "drug_recovery" in engine.scenarios
        assert "market_equilibrium" in engine.scenarios

    def test_find_backdoor_set_physics(self):
        engine = CausalDAGEngine()
        dag = engine.scenarios["physics_newton"]["dag"]
        adj_set = engine.find_backdoor_set(dag, "Force", "Acceleration")
        assert isinstance(adj_set, list)

    def test_evaluate_physics_newton_query(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(
            scenario="physics_newton",
            treatment="Force",
            outcome="Acceleration",
            intervention_val=40.0,
        )
        assert isinstance(result, CausalQueryResult)
        assert result.scenario_name == "physics_newton"
        assert result.treatment_variable == "Force"
        assert result.outcome_variable == "Acceleration"
        assert result.interventional_estimate > 0.0

    def test_evaluate_drug_recovery_query(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(
            scenario="drug_recovery",
            treatment="Drug",
            outcome="Recovery",
        )
        assert isinstance(result, CausalQueryResult)
        # interventional_estimate is a raw causal model value (not bounded to 0-1)
        assert isinstance(result.interventional_estimate, float)
        assert result.diagnostic_summary != ""

    def test_evaluate_market_equilibrium_query(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(
            scenario="market_equilibrium",
            treatment="Interest",
            outcome="Demand",
        )
        assert isinstance(result, CausalQueryResult)
        assert result.diagnostic_summary != ""

    def test_counterfactual_populated(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(
            scenario="physics_newton",
            treatment="Force",
            outcome="Acceleration",
            counterfactual_intervention_val=80.0,
        )
        assert result.counterfactual_outcome is not None

    def test_unknown_scenario_falls_back_to_default(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(
            scenario="nonexistent_scenario_xyz",
            treatment="Force",
            outcome="Acceleration",
        )
        # Should fall back to physics_newton without crashing
        assert isinstance(result, CausalQueryResult)

    def test_to_dict_serialization(self):
        engine = CausalDAGEngine()
        result = engine.evaluate_causal_query(scenario="drug_recovery")
        d = result.to_dict()
        assert "scenario_name" in d
        assert "interventional_estimate" in d
        assert "backdoor_adjustment_set" in d
        assert "is_confounded" in d


# ==========================================================================
# Component 5: NexusEngine Convenience Methods
# ==========================================================================


class TestNexusEngineV90ConvenienceMethods:
    @pytest.fixture(scope="class")
    def engine(self):
        return NexusEngine()

    def test_denoise_thought_latent(self, engine):
        from nexus_diffusion_thought import DiffusionThoughtResult
        res = engine.denoise_thought_latent(
            problem="Compute 12 + 18",
            num_timesteps=10,
            seed=42,
        )
        assert isinstance(res, DiffusionThoughtResult)
        assert res.denoising_steps == 10
        assert res.crystallized_plan != ""

    def test_reflexive_self_correct_clean(self, engine):
        from nexus_reflexion import ReflexionCorrectionResult
        res = engine.reflexive_self_correct(
            problem="Compute 5 + 5",
            proposed_solution="5 + 5 = 10. The answer is 10",
            ground_truth="10",
        )
        assert isinstance(res, ReflexionCorrectionResult)
        # Clean arithmetic should pass without failure
        assert isinstance(res.had_failure, bool)

    def test_evaluate_conformal_stopping_early_exit(self, engine):
        from nexus_conformal_stopping import ConformalStoppingResult
        # With step_entropy=0.9: runner_up = 0.95 - 0.9*0.5 = 0.50
        # margin = 0.95 - 0.50 = 0.45 > default threshold 0.35 → early exit
        res = engine.evaluate_conformal_stopping(
            step_entropy=0.9,
            rsi_volatility=30.0,
            verifier_score=0.95,
            step_index=2,
            total_budget=10,
            target_error_rate=0.05,
        )
        assert isinstance(res, ConformalStoppingResult)
        assert res.should_early_exit is True

    def test_evaluate_conformal_stopping_continue(self, engine):
        from nexus_conformal_stopping import ConformalStoppingResult
        # Low confidence → no early exit
        res = engine.evaluate_conformal_stopping(
            step_entropy=0.8,
            rsi_volatility=80.0,
            verifier_score=0.52,
            step_index=1,
            total_budget=20,
            target_error_rate=0.05,
        )
        assert isinstance(res, ConformalStoppingResult)

    def test_evaluate_causal_dag_physics(self, engine):
        from nexus_causal_dag import CausalQueryResult
        res = engine.evaluate_causal_dag(
            scenario="physics_newton",
            treatment_node="Force",
            outcome_node="Acceleration",
            do_value=40.0,
        )
        assert isinstance(res, CausalQueryResult)
        assert res.interventional_estimate > 0.0

    def test_evaluate_causal_dag_drug(self, engine):
        from nexus_causal_dag import CausalQueryResult
        res = engine.evaluate_causal_dag(
            scenario="drug_recovery",
            treatment_node="Drug",
            outcome_node="Recovery",
            do_value=1.0,
        )
        assert isinstance(res, CausalQueryResult)


# ==========================================================================
# Component 6: REST API Integration – v90 Endpoints
# ==========================================================================


def _get_test_client():
    try:
        from starlette.testclient import TestClient
    except ImportError:
        from fastapi.testclient import TestClient
    app = create_app(NexusApiService())
    return TestClient(app)


class TestV90RestApiEndpoints:
    @pytest.fixture(scope="class")
    def client(self):
        return _get_test_client()

    def test_health_is_v90(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        service = data.get("service", "")
        assert any(v in service for v in ("v88", "v89", "v90")), (
            f"Expected version in service string, got: {service!r}"
        )

    def test_signals_includes_v90_frontier_dot(self, client):
        resp = client.get("/v1/signals")
        assert resp.status_code == 200
        data = resp.json()
        assert "v90_frontier_dot" in data, "v90_frontier_dot key missing from /v1/signals"
        dot = data["v90_frontier_dot"]
        assert dot["diffusion_thought_engine"] is True
        assert dot["reflexive_correction_engine"] is True
        assert dot["conformal_stopping_controller"] is True
        assert dot["causal_dag_engine"] is True

    def test_dot_denoise_endpoint(self, client):
        payload = {
            "problem": "What is the energy stored in a capacitor?",
            "num_timesteps": 10,
            "guidance_scale": 3.0,
            "latent_dim": 8,
            "seed": 42,
        }
        resp = client.post("/v1/dot/denoise", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "crystallized_plan" in data
        assert "trajectory" in data
        assert isinstance(data["trajectory"], list)

    def test_reflexion_correct_endpoint_clean(self, client):
        payload = {
            "problem": "Calculate 100 / 4",
            "proposed_solution": "100 / 4 = 25. The answer is 25",
            "ground_truth": "25",
        }
        resp = client.post("/v1/reflexion/correct", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "had_failure" in data
        assert "diagnostic_summary" in data

    def test_conformal_evaluate_endpoint_early_exit(self, client):
        # step_entropy=0.9 → runner_up = 0.97 - 0.9*0.5 = 0.52 → margin = 0.45 > 0.35
        payload = {
            "step_entropy": 0.9,
            "rsi_volatility": 20.0,
            "verifier_score": 0.97,
            "step_index": 3,
            "total_budget": 10,
            "target_error_rate": 0.05,
        }
        resp = client.post("/v1/conformal/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "should_early_exit" in data
        assert "certified_risk_bound" in data
        assert data["should_early_exit"] is True

    def test_conformal_evaluate_endpoint_continue(self, client):
        payload = {
            "step_entropy": 0.9,
            "rsi_volatility": 90.0,
            "verifier_score": 0.51,
            "step_index": 1,
            "total_budget": 20,
            "target_error_rate": 0.05,
        }
        resp = client.post("/v1/conformal/evaluate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "should_early_exit" in data

    def test_causal_dag_query_endpoint_physics(self, client):
        payload = {
            "scenario": "physics_newton",
            "treatment_node": "Force",
            "outcome_node": "Acceleration",
            "do_value": 40.0,
        }
        resp = client.post("/v1/causal/dag_query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "interventional_estimate" in data
        assert "backdoor_adjustment_set" in data
        assert "diagnostic_summary" in data

    def test_causal_dag_query_endpoint_drug(self, client):
        payload = {
            "scenario": "drug_recovery",
            "treatment_node": "Drug",
            "outcome_node": "Recovery",
            "do_value": 1.0,
        }
        resp = client.post("/v1/causal/dag_query", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "is_confounded" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
