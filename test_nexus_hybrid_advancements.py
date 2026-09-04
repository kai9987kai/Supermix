"""Comprehensive automated tests for the unified MiMoMix-Nexus v80 hybrid architecture.

Tests cover:
1. MultiSourceEntropyEngine (crypto, seeded, software transform, chaotic, and cellular automata)
2. RSIMomentumOscillator (descriptive numeric-sequence momentum diagnostics)
3. QLearningPolicyEngine (discretization, epsilon-greedy action selection, Bellman updates)
4. NexusEngine hybrid processing with multi-source entropy and dynamic thinking budgets
5. NexusApiService endpoints: /v1/think, /v1/entropy, /v1/signals, /v1/telemetry
"""

from __future__ import annotations

import math
import pytest
from starlette.testclient import TestClient

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))

from nexus_engine import (
    MultiSourceEntropyEngine,
    RSIMomentumOscillator,
    QLearningPolicyEngine,
    NexusEngine,
    NexusConfig,
)
from nexus_api import (
    NexusApiService,
    ThinkRequest,
    EntropyRequest,
    create_app,
)


# ---------------------------------------------------------------------------
# 1. MultiSourceEntropyEngine Tests
# ---------------------------------------------------------------------------


def test_entropy_engine_crypto_sampling():
    engine = MultiSourceEntropyEngine()
    samples = engine.sample(source="crypto", count=20)
    assert len(samples) == 20
    for s in samples:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0


def test_entropy_engine_seeded_reproducibility():
    engine = MultiSourceEntropyEngine()
    samples1 = engine.sample(source="seeded", count=10, seed=12345)
    samples2 = engine.sample(source="seeded", count=10, seed=12345)
    samples3 = engine.sample(source="seeded", count=10, seed=99999)

    assert samples1 == samples2
    assert samples1 != samples3
    assert len(samples1) == 10


def test_entropy_engine_os_csprng_transform_has_truthful_provenance():
    engine = MultiSourceEntropyEngine()
    samples = engine.sample(source="os_csprng_transform", count=15)
    assert len(samples) == 15
    for s in samples:
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0
    provenance = engine.source_provenance("os_csprng_transform")
    assert provenance["quantum_hardware_used"] is False
    assert provenance["security_claim"] == "none_for_the_transformed_stream"
    assert engine.normalize_source("qrng") == "os_csprng_transform"


def test_entropy_engine_chaotic_logistic_map():
    engine = MultiSourceEntropyEngine()
    samples1 = engine.sample(source="chaotic", count=25, seed=500)
    assert len(samples1) == 25
    for s in samples1:
        assert isinstance(s, float)
        assert 0.0 < s < 1.0


def test_cellular_automata_rule30_and_rule110():
    engine = MultiSourceEntropyEngine()
    grid30 = engine.cellular_automata_step(rule=30, steps=10, width=15)
    assert len(grid30) == 10
    assert len(grid30[0]) == 15
    # Central cell should be 1 in initial state
    assert grid30[0][7] == 1
    assert sum(grid30[0]) == 1

    grid110 = engine.cellular_automata_step(rule=110, steps=8, width=21)
    assert len(grid110) == 8
    assert len(grid110[0]) == 21


# ---------------------------------------------------------------------------
# 2. RSIMomentumOscillator Tests
# ---------------------------------------------------------------------------


def test_rsi_oscillator_initialization_and_bounds():
    osc = RSIMomentumOscillator(window=14)
    res = osc.update(0.5)
    assert res["rsi"] == 50.0
    assert res["regime"] == "flat_or_unresolved_probe"
    assert res["extreme_momentum_flag"] is False


def test_rsi_oscillator_rising_trend_overbought():
    osc = RSIMomentumOscillator(window=5)
    # Feed an escalating sequence of novelty/entropy values
    values = [0.1, 0.2, 0.35, 0.55, 0.75, 0.90, 0.98]
    res = {}
    for v in values:
        res = osc.update(v)

    assert res["rsi"] >= 70.0
    assert res["regime"] == "high_positive_probe_momentum"
    assert res["volatility"] > 0.0


def test_rsi_oscillator_falling_trend_oversold():
    osc = RSIMomentumOscillator(window=5)
    # Feed a declining sequence
    values = [0.95, 0.85, 0.70, 0.50, 0.30, 0.15, 0.05]
    res = {}
    for v in values:
        res = osc.update(v)

    assert res["rsi"] <= 30.0
    assert res["regime"] == "high_negative_probe_momentum"


# ---------------------------------------------------------------------------
# 3. QLearningPolicyEngine Tests
# ---------------------------------------------------------------------------


def test_q_learning_policy_discretization_and_action_selection():
    q_engine = QLearningPolicyEngine(alpha=0.2, gamma=0.9, epsilon=0.0)
    action_easy_low_risk = q_engine.select_action(difficulty=0.1, risk=0.1)
    assert action_easy_low_risk in QLearningPolicyEngine.ACTIONS

    action_hard_high_risk = q_engine.select_action(difficulty=0.8, risk=0.8)
    assert action_hard_high_risk in ["deep", "swarm", "agent", "got", "solve"]


def test_q_learning_policy_bellman_update():
    q_engine = QLearningPolicyEngine(alpha=0.3, gamma=0.9, epsilon=0.0)
    state = q_engine.discretize_state(0.5, 0.5)
    initial_q = q_engine.q_table[state]["deep"]

    # Reward of 1.0 should increase Q-value
    new_q = q_engine.update(
        difficulty=0.5,
        risk=0.5,
        action="deep",
        reward=1.0,
        next_difficulty=0.5,
        next_risk=0.5,
    )
    assert new_q > initial_q
    assert q_engine.q_table[state]["deep"] == new_q


def test_q_learning_policy_summary():
    q_engine = QLearningPolicyEngine()
    summary = q_engine.get_policy_summary()
    assert "total_states" in summary
    assert "state_matrix" in summary
    assert summary["total_states"] == 16  # 4x4 grid


# ---------------------------------------------------------------------------
# 4. NexusEngine End-to-End Hybrid Processing
# ---------------------------------------------------------------------------


def test_nexus_engine_process_with_entropy_source():
    engine = NexusEngine()
    res = engine.process(
        query="What is the difference between entropy and randomness in neural sampling?",
        mode="fast",
        entropy_source="os_csprng_transform",
    )
    assert res.mode_selected == "fast"
    assert "entropy_telemetry" in res.telemetry
    assert res.telemetry["entropy_telemetry"]["active_source"] == "os_csprng_transform"
    assert res.telemetry["entropy_telemetry"]["provenance"]["quantum_hardware_used"] is False
    assert "rsi_diagnostic" in res.telemetry
    assert res.telemetry["rsi_diagnostic"]["rsi"] >= 0.0
    assert res.telemetry["rsi_diagnostic"]["is_live_reasoning_signal"] is False


def test_nexus_engine_process_solve_exact_remains_authoritative():
    engine = NexusEngine()
    q = "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and temperature is 300 K. What is its pressure?"
    res = engine.process(
        query=q,
        mode="solve",
    )
    assert res.mode_selected in ("solve", "scientific")
    assert res.confidence is None
    assert res.epistemics["confidence_kind"] == "deterministic_assurance_not_probability"
    assert res.epistemics["decision"] == "answered"
    assert res.epistemics["answer_authority"] is True
    assert "verified_answer_receipt" in res.audit_receipts


def test_nexus_engine_process_swarm_deliberation():
    engine = NexusEngine()
    res = engine.process(
        query="Debate the optimal KV cache eviction policy for long context models",
        mode="swarm",
    )
    assert res.mode_selected == "swarm"
    assert "Analysis-only swarm scaffold" in res.final_output
    assert len(res.thought_steps) >= 1
    assert "swarm_receipt" in res.audit_receipts


# ---------------------------------------------------------------------------
# 5. NexusApiService Endpoints Tests
# ---------------------------------------------------------------------------


def test_api_service_handle_entropy():
    svc = NexusApiService()
    req = EntropyRequest(source="chaotic", count=12, rule=90, ca_steps=8, ca_width=15)
    resp = svc.handle_entropy(req)
    assert resp["source"] == "chaotic"
    assert resp["count"] == 12
    assert len(resp["samples"]) == 12
    assert resp["rule"] == 90
    assert len(resp["cellular_automata_grid"]) == 8


def test_api_service_handle_signals():
    svc = NexusApiService()
    resp = svc.handle_signals()
    assert "q_policy" in resp
    assert "rsi_diagnostic" in resp
    assert "entropy_sources_available" in resp
    assert "crypto" in resp["entropy_sources_available"]
    assert "qrng" not in resp["entropy_sources_available"]
    assert resp["q_policy"]["connected_to_nexus_process"] is False
    assert "hybrid_attention" in resp


def test_api_fastapi_endpoints():
    app = create_app()
    client = TestClient(app)

    # 1. /health
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    # 2. /v1/signals
    r_signals = client.get("/v1/signals")
    assert r_signals.status_code == 200
    assert "q_policy" in r_signals.json()

    # 3. /v1/entropy
    r_entropy = client.post(
        "/v1/entropy",
        json={"source": "qrng", "count": 8, "rule": 30, "ca_steps": 4, "ca_width": 11},
    )
    assert r_entropy.status_code == 200
    assert r_entropy.json()["source"] == "os_csprng_transform"
    assert r_entropy.json()["requested_source"] == "qrng"
    assert r_entropy.json()["provenance"]["quantum_hardware_used"] is False

    # 4. /v1/think with entropy source
    r_think = client.post(
        "/v1/think",
        json={
            "prompt": "Evaluate sparse MoE scaling with auxiliary loss free balancing",
            "mode": "fast",
            "thinking_budget": 2,
            "entropy_source": "chaotic",
        },
    )
    assert r_think.status_code == 200
    data = r_think.json()
    assert "mode_selected" in data
    assert "telemetry" in data
