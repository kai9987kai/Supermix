"""Tests for the NexusMind 5-Agent Cognitive Swarm engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_swarm as swarm


def test_agent_roles_and_initialization():
    agents = [
        swarm.CognitiveAgent("gen", swarm.AgentRole.GENERATOR, 0.25),
        swarm.CognitiveAgent("crit", swarm.AgentRole.CRITIC, 0.25),
        swarm.CognitiveAgent("skep", swarm.AgentRole.SKEPTIC, 0.20),
        swarm.CognitiveAgent("arch", swarm.AgentRole.ARCHIVIST, 0.15),
        swarm.CognitiveAgent("anom", swarm.AgentRole.ANOMALY_HUNTER, 0.15),
    ]
    engine = swarm.SwarmEngine(agents=agents, max_rounds=2)
    assert len(engine.agents) == 5
    assert engine.max_rounds == 2


def test_replicator_dynamics_weight_update():
    weights = {"gen": 0.2, "crit": 0.2, "skep": 0.2, "arch": 0.2, "anom": 0.2}
    fitness = {"gen": 1.2, "crit": 1.4, "skep": 0.8, "arch": 0.9, "anom": 1.0}
    updated = swarm.replicator_weight_update(weights, fitness, learning_rate=0.2)

    assert len(updated) == 5
    assert updated["crit"] > updated["gen"] > updated["skep"]
    assert pytest.approx(sum(updated.values()), 0.01) == 1.0


def test_swarm_deliberation_execution():
    engine = swarm.SwarmEngine(max_rounds=3)
    res = engine.deliberate(query="Verify optimal cache eviction strategy under sliding window attention.")

    assert isinstance(res, swarm.SwarmDeliberationResult)
    assert len(res.rounds) >= 1
    assert res.final_confidence > 0.0
    assert len(res.consensus_output) > 0
    assert res.receipt.schema_version == "nexus-swarm-receipt-v1"
    assert len(res.receipt.query_digest) == 64
    assert len(res.receipt.consensus_digest) == 64
    assert res.receipt.authority_bits["has_open_world_authority"] is False
    assert "verified consensus" not in res.consensus_output.lower()
    assert "not verification" in res.consensus_output.lower()
    assert res.telemetry["answer_verified"] is False
    assert res.to_dict()["answer_authority"] is False
    assert res.to_dict()["score_semantics"] == "template_agent_agreement_not_correctness"


def test_swarm_deterministic_reproducibility():
    engine1 = swarm.SwarmEngine(max_rounds=2)
    engine2 = swarm.SwarmEngine(max_rounds=2)
    query = "Analyze consistency of quantum random generator"

    res1 = engine1.deliberate(query)
    res2 = engine2.deliberate(query)

    assert res1.consensus_output == res2.consensus_output
    assert res1.final_confidence == res2.final_confidence
    assert res1.receipt.query_digest == res2.receipt.query_digest
