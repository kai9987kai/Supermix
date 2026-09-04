"""Unit tests for NexusMind v84 innovations.

Covers:
- Quantum Density Matrix, Von Neumann Entropy & Decoherence Channels
- Wolfram Rule 110 Glider & Soliton Logic Engine
- Dynamic 5D Cognitive Trajectory Tracking
- Speculative Tree Search with Step-Level PRM and Backtracking
- v84 REST API Endpoints
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import pytest
import nexus_api as api
from nexus_engine import (
    NexusEngine,
    QuantumStateEngine,
    QuantumDensityResult,
    WolframGliderEngine,
    GliderCollisionResult,
    CognitiveTrajectoryTracker,
    CognitiveTrajectoryResult,
    SpeculativeTreeSearchEngine,
    SpeculativeTreeResult,
)


def test_quantum_state_pure_bell_state():
    engine = QuantumStateEngine()
    res = engine.analyze_state(parameter_p=1.0, noise_rate=0.0, channel_type="unitary")
    assert isinstance(res, QuantumDensityResult)
    assert res.von_neumann_entropy == pytest.approx(0.0, abs=1e-3)
    assert res.purity == pytest.approx(1.0, abs=1e-3)
    assert res.concurrence == pytest.approx(1.0, abs=1e-3)
    assert res.is_entangled is True


def test_quantum_state_maximally_mixed_state():
    engine = QuantumStateEngine()
    res = engine.analyze_state(parameter_p=0.0, noise_rate=0.0)
    assert res.von_neumann_entropy == pytest.approx(2.0, abs=1e-3)
    assert res.purity == pytest.approx(0.25, abs=1e-3)
    assert res.concurrence == pytest.approx(0.0, abs=1e-3)
    assert res.is_entangled is False


def test_quantum_state_depolarizing_noise():
    engine = QuantumStateEngine()
    pure = engine.analyze_state(parameter_p=1.0, noise_rate=0.0)
    noisy = engine.analyze_state(parameter_p=1.0, noise_rate=0.5, channel_type="depolarizing")
    assert noisy.von_neumann_entropy > pure.von_neumann_entropy
    assert noisy.purity < pure.purity
    assert noisy.concurrence < pure.concurrence


def test_wolfram_glider_simulation():
    engine = WolframGliderEngine()
    res = engine.simulate_collision(
        glider_type_left="glider_A",
        glider_type_right="glider_C",
        separation=8,
        steps=16,
        width=30,
    )
    assert isinstance(res, GliderCollisionResult)
    assert res.rule == 110
    assert res.ether_period == 14
    assert len(res.grid) == 16
    assert len(res.gliders_identified) == 2
    assert "logic_operation_analog" in res.to_dict()


def test_cognitive_trajectory_tracker():
    tracker = CognitiveTrajectoryTracker()
    steps = [
        "Calculate the mathematical derivative and solve the formula",
        "Imagine an artistic metaphor and write poetic stories",
        "Verify the evidence receipt and inspect the security audit ledger",
        "Plan the agent mission tasks and execute the target goals",
    ]
    res = tracker.trace_trajectory(steps)
    assert isinstance(res, CognitiveTrajectoryResult)
    assert len(res.coordinates_2d) == 4
    assert res.step_archetypes[0] == "logos"
    assert res.step_archetypes[1] == "mythos"
    assert res.step_archetypes[2] == "ethos"
    assert res.step_archetypes[3] == "telos"
    assert res.total_path_length > 0.0
    assert res.net_cognitive_drift > 0.0
    assert len(res.velocities) == 4
    assert len(res.curvatures) == 4


def test_speculative_tree_search_backtracking():
    engine = SpeculativeTreeSearchEngine(branching_factor=2, max_depth=3)
    res = engine.search("Prove that the square root of 2 is irrational")
    assert isinstance(res, SpeculativeTreeResult)
    assert res.total_nodes_evaluated > 3
    assert len(res.optimal_path_node_ids) >= 2
    assert 0.0 <= res.prm_mean_score <= 1.0
    assert len(res.final_output) > 10
    assert "schema_version" in res.receipt


def test_nexus_engine_v84_convenience_methods():
    engine = NexusEngine()
    q_res = engine.run_quantum_state_analysis(parameter_p=0.8, noise_rate=0.1)
    assert q_res.purity > 0.5

    g_res = engine.run_glider_simulation(steps=12, width=25)
    assert g_res.rule == 110

    t_res = engine.run_cognitive_trajectory(["Logic reasoning step", "Creative ideation"])
    assert len(t_res.coordinates_2d) == 2

    s_res = engine.run_speculative_tree_search("What is 10 * 10?")
    assert len(s_res.nodes) > 1


def test_v84_api_endpoints():
    import warnings
    from starlette.testclient import TestClient

    app = api.create_app(api.NexusApiService())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(app)

    # 1. /v1/quantum/state
    q_resp = client.post("/v1/quantum/state", json={"parameter_p": 0.9, "noise_rate": 0.1, "channel_type": "depolarizing"})
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "von_neumann_entropy" in q_data
    assert "purity" in q_data
    assert "concurrence" in q_data

    # 2. /v1/wolfram/gliders
    g_resp = client.post("/v1/wolfram/gliders", json={"steps": 15, "width": 30})
    assert g_resp.status_code == 200
    g_data = g_resp.json()
    assert g_data["rule"] == 110
    assert "logic_operation_analog" in g_data

    # 3. /v1/resonance/trajectory
    t_resp = client.post("/v1/resonance/trajectory", json={"steps": ["Math equation", "Poetic metaphor"]})
    assert t_resp.status_code == 200
    t_data = t_resp.json()
    assert len(t_data["coordinates_2d"]) == 2
    assert "total_path_length" in t_data

    # 4. /v1/speculative-tree
    st_resp = client.post("/v1/speculative-tree", json={"query": "Solve algorithmic problem", "max_depth": 3})
    assert st_resp.status_code == 200
    st_data = st_resp.json()
    assert "optimal_path_node_ids" in st_data
    assert "prm_mean_score" in st_data
