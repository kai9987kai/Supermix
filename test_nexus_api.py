"""Tests for the NexusMind API service endpoints."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_api as api
import nexus_epistemics as epistemics


class _AdversarialEngine:
    """Minimal injected engine used to probe the public API trust boundary."""

    def __init__(self, result):
        self.result = result

    def process(self, **_kwargs):
        return self.result


def test_api_service_think_fast():
    svc = api.NexusApiService()
    req = api.ThinkRequest(prompt="Explain attention sinks", mode="fast")
    resp = svc.handle_think(req)

    assert isinstance(resp, api.ThinkResponse)
    assert resp.model == "nexus-experimental-neural-telemetry"
    assert resp.mode_selected == "fast"
    assert len(resp.thought_steps) >= 1
    assert resp.confidence is None
    assert resp.epistemics["decision"] == "abstained"
    assert epistemics.verify_epistemic_receipt(resp.epistemics)


def test_api_service_think_deep():
    svc = api.NexusApiService()
    req = api.ThinkRequest(prompt="Analyze recursive latent refinement", mode="deep")
    resp = svc.handle_think(req)

    assert isinstance(resp, api.ThinkResponse)
    assert resp.model == "nexus-experimental-neural-telemetry"
    assert resp.mode_selected == "deep"
    assert resp.confidence is None
    assert resp.epistemics["decision"] == "abstained"


def test_think_withholds_every_candidate_field_when_engine_receipt_is_invalid():
    leaked = api.NexusResult(
        query="private",
        mode_selected="fast",
        final_output="The secret candidate is 15.",
        confidence=0.99,
        thought_steps=[],
        audit_receipts={"solver": {"display_result": "15"}},
        telemetry={
            "synthetic_observability_probe": {
                "is_live_quality_evidence": False,
                "candidate": "15",
            },
            "entropy": {"candidate": "15"},
            "quality_probability": 0.99,
        },
        epistemics={"decision": "answered", "receipt_sha256": "0" * 64},
    )
    svc = api.NexusApiService(_AdversarialEngine(leaked))

    resp = svc.handle_think(api.ThinkRequest(prompt="Do not calculate 3 times 5", mode="fast"))
    encoded = json.dumps(resp.to_dict(), sort_keys=True)

    assert resp.epistemics["decision"] == "abstained"
    assert resp.confidence is None
    assert resp.audit_receipts == {}
    assert "secret candidate" not in encoded
    assert '"15"' not in encoded
    assert "quality_probability" not in encoded


def test_think_recomputes_instead_of_trusting_valid_self_hashed_answer_receipt():
    claimed = epistemics.verified_exact_decision(
        reason="engine_claim",
        claim_scope="engine supplied scope",
        verifier_id="grounding_runtime.finalize_grounded_response",
    ).to_dict()
    fake = api.NexusResult(
        query="ignored",
        mode_selected="fast",
        final_output="999",
        confidence=1.0,
        audit_receipts={"claimed": {"display_result": "999"}},
        epistemics=claimed,
    )
    svc = api.NexusApiService(_AdversarialEngine(fake))

    exact = svc.handle_think(api.ThinkRequest(prompt="What is 2 + 3 * 4?", mode="fast"))
    open_world = svc.handle_think(
        api.ThinkRequest(prompt="Predict tomorrow's stock-market winner.", mode="fast")
    )

    assert exact.epistemics["decision"] == "answered"
    assert "14" in exact.output
    assert exact.confidence == 1.0
    assert "999" not in json.dumps(exact.to_dict(), sort_keys=True)
    assert open_world.epistemics["decision"] == "abstained"
    assert open_world.confidence is None
    assert open_world.audit_receipts == {}
    assert "999" not in json.dumps(open_world.to_dict(), sort_keys=True)


def test_think_latency_includes_fresh_api_verifier(monkeypatch):
    query = "What is 2 + 3 * 4?"
    grounded = json.loads(json.dumps(api.grounding.finalize_grounded_response("", query)))
    clock = {"seconds": 10.0}

    claimed = epistemics.verified_exact_decision(
        reason="engine_claim",
        claim_scope="engine supplied scope",
        verifier_id="grounding_runtime.finalize_grounded_response",
    ).to_dict()
    fake = api.NexusResult(
        query=query,
        mode_selected="fast",
        final_output="untrusted",
        confidence=1.0,
        epistemics=claimed,
    )

    def delayed_grounder(*_args, **_kwargs):
        clock["seconds"] += 0.25
        return grounded

    monkeypatch.setattr(api.time, "perf_counter", lambda: clock["seconds"])
    monkeypatch.setattr(api.grounding, "finalize_grounded_response", delayed_grounder)
    resp = api.NexusApiService(_AdversarialEngine(fake)).handle_think(
        api.ThinkRequest(prompt=query, mode="fast")
    )

    assert resp.epistemics["decision"] == "answered"
    assert resp.latency_ms == 250.0


def test_think_rebuilds_valid_abstention_metadata_and_canonicalizes_engine_mode():
    sentinel = "ENGINE_LEAK_151515"
    receipt = epistemics.abstained_decision(
        reason=sentinel,
        claim_scope=sentinel,
        limitations=(sentinel,),
        protocol={"candidate": sentinel},
    ).to_dict()
    fake = api.NexusResult(
        query=sentinel,
        mode_selected=sentinel,
        final_output=sentinel,
        latency_ms=151515.0,
        audit_receipts={"candidate": sentinel},
        telemetry={"entropy": {"candidate": sentinel, "confidence": 0.99}},
        epistemics=receipt,
    )
    svc = api.NexusApiService(_AdversarialEngine(fake))

    resp = svc.handle_think(api.ThinkRequest(prompt="Unsupported request", mode="fast"))
    encoded = json.dumps(resp.to_dict(), sort_keys=True)

    assert resp.mode_selected == "fast"
    assert resp.epistemics["decision"] == "abstained"
    assert resp.latency_ms != 151515.0
    assert sentinel not in encoded


def test_think_does_not_forward_forgeable_analysis_only_engine_payloads():
    sentinel = "ANALYSIS_LEAK_424242"
    receipt = epistemics.analysis_only_decision(
        reason="forged_but_schema_valid_analysis",
        claim_scope="untrusted injected analysis",
        evidence_class="template_deliberation",
        internal_score=0.99,
        internal_score_name="template_score",
        protocol={"candidate": sentinel},
    ).to_dict()
    fake = api.NexusResult(
        query="ignored",
        mode_selected="swarm",
        final_output=sentinel,
        confidence=None,
        audit_receipts={"candidate": sentinel},
        telemetry={"synthetic_observability_probe": {"candidate": sentinel}},
        epistemics=receipt,
    )
    svc = api.NexusApiService(_AdversarialEngine(fake))

    resp = svc.handle_think(api.ThinkRequest(prompt="Analyze options", mode="swarm"))
    encoded = json.dumps(resp.to_dict(), sort_keys=True)

    assert resp.epistemics["decision"] == "abstained"
    assert resp.audit_receipts == {}
    assert sentinel not in encoded


def test_api_service_swarm_endpoint():
    svc = api.NexusApiService()
    req = api.SwarmRequest(query="Deliberate on MoE load balancing", max_rounds=2)
    resp = svc.handle_swarm(req)

    assert "consensus_output" in resp
    assert "rounds" in resp
    assert "receipt" in resp
    assert len(resp["rounds"]) <= 2
    assert resp["status"] == "analysis_only"
    assert resp["answer_authority"] is False
    assert resp["confidence"] is None
    assert resp["internal_consensus_score"] > 0.0
    assert resp["epistemics"]["decision"] == "analysis_only"

    def assert_no_numeric_confidence(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if "confidence" in key.lower():
                    assert item is None or not isinstance(item, (int, float))
                assert_no_numeric_confidence(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_numeric_confidence(item)

    assert_no_numeric_confidence(resp)


@pytest.mark.parametrize("mutation", ["missing_receipt", "not_selected", "verification_failed"])
@pytest.mark.parametrize("surface", ["think", "solve", "scientific"])
def test_public_answer_surfaces_reject_malformed_grounder_receipts(monkeypatch, mutation, surface):
    sentinel = "MALFORMED_GROUNDER_737373"
    if surface == "scientific":
        query = (
            "Under constant acceleration with initial velocity 0 m/s, acceleration "
            "9.8 m/s^2, and time 5 s, what is the final velocity?"
        )
    else:
        query = "What is 2 + 3 * 4?"
    forged = json.loads(json.dumps(api.grounding.finalize_grounded_response("", query)))
    forged["text"] = sentinel
    if mutation == "missing_receipt":
        forged.pop("answer_receipt", None)
    elif mutation == "not_selected":
        forged["answer_receipt"]["selected"] = False
    else:
        forged["answer_receipt"]["verification"]["passed"] = False

    monkeypatch.setattr(
        api.grounding,
        "finalize_grounded_response",
        lambda *_args, **_kwargs: forged,
    )
    svc = api.NexusApiService()
    if surface == "think":
        payload = svc.handle_think(api.ThinkRequest(prompt=query, mode="solve")).to_dict()
    elif surface == "solve":
        payload = svc.handle_solve(api.SolveRequest(query=query))
    else:
        payload = svc.handle_scientific(api.ScientificRequest(query=query))

    assert payload["epistemics"]["decision"] == "abstained"
    assert payload["confidence"] is None
    assert sentinel not in json.dumps(payload, sort_keys=True)


def test_api_service_got_endpoint():
    svc = api.NexusApiService()
    req = api.GoTRequest(query="Synthesize optimal search path", max_depth=2)
    resp = svc.handle_got(req)

    assert "final_output" in resp
    assert "best_path_nodes" in resp
    assert "receipt" in resp
    assert resp["receipt"]["max_search_depth"] <= 2
    assert resp["status"] == "analysis_only"
    assert resp["confidence"] is None
    assert resp["epistemics"]["decision"] == "analysis_only"


def test_api_service_scientific_endpoint():
    svc = api.NexusApiService()
    req = api.ScientificRequest(
        query="Under constant acceleration with initial velocity 0 m/s, acceleration 9.8 m/s^2, and time 5 s, what is the final velocity?"
    )
    resp = svc.handle_scientific(req)

    assert resp.get("status") == "answered"
    assert resp["answer_authority"] is True
    assert resp["confidence"] == 1.0
    assert "receipt" in resp
    answer_display = resp.get("result", {}).get("answer", {}).get("display", "")
    assert "49" in answer_display


def test_api_solve_is_verifier_first_and_withholds_legacy_match():
    svc = api.NexusApiService()

    exact = svc.handle_solve(api.SolveRequest(query="What is 2 + 3 * 4?"))
    rejected = svc.handle_solve(
        api.SolveRequest(
            query="Do not calculate force when mass is 3 kg and acceleration is 5 m/s^2."
        )
    )

    assert exact["status"] == "answered"
    assert exact["display_answer"] == "14"
    assert exact["epistemics"]["decision"] == "answered"
    assert exact["receipt"]["receipt_is_authority"] is False
    receipt_payload = dict(exact["receipt"])
    supplied_digest = receipt_payload.pop("receipt_sha256")
    assert api._canonical_sha256(receipt_payload) == supplied_digest
    assert rejected["status"] == "abstained"
    assert rejected["solved"] is False
    assert rejected["confidence"] is None
    assert "display_answer" not in rejected
    legacy_audit = rejected["audit"]["legacy_nexus_solver"]
    assert legacy_audit["candidate_withheld_unless_strict_gate_passes"] is True
    assert legacy_audit["full_receipt_withheld"] is True
    assert "receipt" not in legacy_audit
    assert "15" not in json.dumps(legacy_audit, sort_keys=True)


def test_api_service_telemetry_and_feedback():
    svc = api.NexusApiService()
    telem = svc.handle_telemetry()
    assert "chsh_bell_value" in telem["synthetic_metric_probe"]
    assert telem["synthetic_metric_probe"]["input_is_live_model_output"] is False
    assert "moe_experts" in telem

    policy_before = svc.engine.q_learner.to_dict()

    fb_req = api.FeedbackRequest(
        difficulty=0.5,
        epistemic_risk=0.2,
        budget_used=4,
        reward=1.0,
    )
    fb_resp = svc.handle_feedback(fb_req)
    assert fb_resp["status"] == "rejected"
    assert fb_resp["policy_updated"] is False
    assert svc.engine.q_learner.to_dict() == policy_before


def test_model_catalog_reports_observed_capabilities_without_fake_context_windows():
    svc = api.NexusApiService()
    payload = svc.handle_models()
    models = {row["id"]: row for row in payload["models"]}

    assert set(models) == {
        "nexus-exact-solver",
        "nexus-heuristic-suite",
        "nexus-experimental-neural-telemetry",
    }
    assert models["nexus-experimental-neural-telemetry"]["generator_ready"] is False
    assert models["nexus-experimental-neural-telemetry"]["input_limit_characters"] == 64
    assert models["nexus-experimental-neural-telemetry"]["configured_sliding_window_tokens"] == 128
    assert all("context_window" not in row for row in models.values())


def test_fastapi_serves_studio_on_same_origin_when_available():
    app = api.create_app(api.NexusApiService())
    if not hasattr(app, "routes"):
        pytest.skip("FastAPI is not installed")

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/studio" in paths
    assert api._STUDIO_PATH.is_file()

    # Exercise actual JSON-body resolution. With postponed annotations and
    # function-local Pydantic classes, these routes previously misclassified
    # `req` as a query parameter and every browser POST returned HTTP 422.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        client = TestClient(app)
    solve = client.post("/v1/solve", json={"query": "What is 2 + 3 * 4?"})
    innovate = client.post(
        "/v1/innovate",
        json={"topic": "evidence-first routing", "count": 2},
    )

    assert solve.status_code == 200
    assert solve.json()["status"] == "answered"
    assert solve.json()["display_answer"] == "14"
    assert innovate.status_code == 200
    assert innovate.json()["status"] == "analysis_only"
    assert len(innovate.json()["concepts"]) == 2
