import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "source"))

import nexus_api as api
import nexus_engine as ne
import mimomix_observatory as observatory


def test_adaptive_policy_budget_planning():
    cfg = ne.NexusConfig(max_thinking_budget=8)
    q_engine = observatory.BudgetPolicyLearner()
    rsi = ne.RSIMomentumOscillator()
    policy = ne.AdaptiveThinkingPolicy(cfg, q_engine, rsi)

    plan = policy.plan_compute_budget(
        query="Calculate the differential cross section of quantum scattering",
        difficulty=0.75,
        risk=0.4,
        entropy_val=0.62,
    )

    assert plan["mode"] == "adaptive"
    assert 1 <= plan["shadow_recommended_cycles"] <= cfg.max_thinking_budget
    assert plan["applied_max_cycles"] == plan["allocated_cycles"] == 1
    assert plan["shadow_recommendation_applied"] is False
    assert 0.3 <= plan["requested_mod_capacity_ratio"] <= 1.0
    assert plan["requested_differential_attention"] is True
    assert "rsi_momentum" in plan
    assert plan["estimated_difficulty"] == 0.75
    assert plan["epistemic_risk"] == 0.4
    assert plan["policy_evidence"] == "authored_shadow_heuristic_not_calibrated"
    assert plan["execution_authorized"] is False
    assert plan["answer_authority"] is False


def test_adaptive_policy_rsi_hysteresis():
    cfg = ne.NexusConfig(max_thinking_budget=8)
    q_engine = observatory.BudgetPolicyLearner()
    
    # High RSI oscillator (momentum of novelty/instability)
    rsi_high = ne.RSIMomentumOscillator()
    for _ in range(15):
        rsi_high.update(1.0)  # Upward impulse -> high RSI

    policy_high = ne.AdaptiveThinkingPolicy(cfg, q_engine, rsi_high)
    plan_high = policy_high.plan_compute_budget("test query", 0.5, 0.5)

    # Low RSI oscillator (stable sequence)
    rsi_low = ne.RSIMomentumOscillator()
    for _ in range(15):
        rsi_low.update(0.0)  # Downward impulse -> low RSI

    policy_low = ne.AdaptiveThinkingPolicy(cfg, q_engine, rsi_low)
    plan_low = policy_low.plan_compute_budget("test query", 0.5, 0.5)

    assert plan_high["shadow_recommended_cycles"] >= plan_low["shadow_recommended_cycles"]
    assert plan_high["applied_max_cycles"] == plan_low["applied_max_cycles"]


def test_nexus_engine_adaptive_mode_execution():
    engine = ne.build_default_engine()
    res = engine.process("Derive and calculate quantum harmonic oscillator energy levels", mode="adaptive")

    assert res.mode_selected == "adaptive"
    assert res.thought_steps
    adaptive_steps = [s for s in res.thought_steps if s.stage == "adaptive_compute"]
    assert len(adaptive_steps) >= 1
    assert "compute_budget_report" in res.telemetry
    budget_rep = res.telemetry["compute_budget_report"]
    assert budget_rep["mode"] == "adaptive"
    assert "allocated_cycles" in budget_rep
    assert budget_rep["executed_mechanisms"]["applied_max_cycles"] == budget_rep["allocated_cycles"]
    assert budget_rep["executed_mechanisms"]["observed_cycles"] is not None
    assert budget_rep["executed_mechanisms"]["adaptive_thinking"] is True
    assert budget_rep["executed_mechanisms"]["differential_attention"] is False
    assert budget_rep["executed_mechanisms"]["mixture_of_depths"] is False
    assert budget_rep["executed_mechanisms"]["multi_latent_attention"] is False
    assert budget_rep["executed_mechanisms"]["mod_capacity_ratio"] is None
    assert budget_rep["optional_mechanism_request_applied"] is False
    assert budget_rep["execution_authorized"] is False
    assert all(
        row == {
            "available": True,
            "configured": False,
            "executed": False,
            "efficiency_validated": False,
        }
        for row in budget_rep["module_census"].values()
    )


def test_nexus_engine_adaptive_entropy_sources():
    engine = ne.build_default_engine()
    for source in ["crypto", "seeded", "chaotic", "os_csprng_transform"]:
        res = engine.process("Explore the entropy landscape of neural representations", mode="adaptive", entropy_source=source)
        assert res.mode_selected == "adaptive"
        assert res.telemetry["compute_budget_report"]["entropy_estimate"] is not None


def test_public_api_preserves_adaptive_mode_and_sanitizes_execution_telemetry():
    svc = api.NexusApiService()
    response = svc.handle_think(
        api.ThinkRequest(
            prompt="Explore a difficult proof strategy",
            mode="adaptive",
            thinking_budget=3,
            entropy_source="seeded",
        )
    )

    assert response.mode_selected == "adaptive"
    assert response.epistemics["decision"] == "abstained"
    assert response.confidence is None
    report = response.telemetry["compute_budget_report"]
    assert report["allocated_cycles"] == 3
    assert report["executed_mechanisms"]["applied_max_cycles"] == 3
    assert report["executed_mechanisms"]["observed_cycles"] is not None
    assert report["executed_mechanisms"]["differential_attention"] is False
    assert report["executed_mechanisms"]["mixture_of_depths"] is False
    assert report["policy_evidence"] == "authored_shadow_heuristic_not_calibrated"
    assert report["module_census"]["multi_latent_attention"]["executed"] is False

    neural = next(
        item for item in svc.handle_models()["models"]
        if item["id"] == "nexus-experimental-neural-telemetry"
    )
    assert "adaptive" in neural["modes"]
