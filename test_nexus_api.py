"""Tests for the NexusMind API service endpoints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_api as api


def test_api_service_think_fast():
    svc = api.NexusApiService()
    req = api.ThinkRequest(prompt="Explain attention sinks", mode="fast")
    resp = svc.handle_think(req)

    assert isinstance(resp, api.ThinkResponse)
    assert resp.model == "nexus-v78-flash"
    assert resp.mode_selected == "fast"
    assert len(resp.thought_steps) >= 1


def test_api_service_think_deep():
    svc = api.NexusApiService()
    req = api.ThinkRequest(prompt="Analyze recursive latent refinement", mode="deep")
    resp = svc.handle_think(req)

    assert isinstance(resp, api.ThinkResponse)
    assert resp.model == "nexus-v78-pro"
    assert resp.mode_selected == "deep"


def test_api_service_swarm_endpoint():
    svc = api.NexusApiService()
    req = api.SwarmRequest(query="Deliberate on MoE load balancing", max_rounds=2)
    resp = svc.handle_swarm(req)

    assert "consensus_output" in resp
    assert "rounds" in resp
    assert "receipt" in resp


def test_api_service_got_endpoint():
    svc = api.NexusApiService()
    req = api.GoTRequest(query="Synthesize optimal search path", max_depth=2)
    resp = svc.handle_got(req)

    assert "final_output" in resp
    assert "best_path_nodes" in resp
    assert "receipt" in resp


def test_api_service_scientific_endpoint():
    svc = api.NexusApiService()
    req = api.ScientificRequest(
        query="Under constant acceleration with initial velocity 0 m/s, acceleration 9.8 m/s^2, and time 5 s, what is the final velocity?"
    )
    resp = svc.handle_scientific(req)

    assert resp.get("status") == "success"
    assert "receipt" in resp
    answer_display = resp.get("result", {}).get("answer", {}).get("display", "")
    assert "49" in answer_display


def test_api_service_telemetry_and_feedback():
    svc = api.NexusApiService()
    telem = svc.handle_telemetry()
    assert "chsh_bell_value" in telem
    assert "moe_experts" in telem

    fb_req = api.FeedbackRequest(
        difficulty=0.5,
        epistemic_risk=0.2,
        budget_used=4,
        reward=1.0,
    )
    fb_resp = svc.handle_feedback(fb_req)
    assert fb_resp["status"] == "ok"
