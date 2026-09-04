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


VALID_NONCE = "test-request-nonce-0001"


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
        request_sha256="1" * 64,
        output_sha256="2" * 64,
        verifier_receipt_sha256="3" * 64,
        request_nonce_sha256="4" * 64,
        surface="think",
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

    exact = svc.handle_think(
        api.ThinkRequest(
            prompt="What is 2 + 3 * 4?",
            mode="fast",
            request_nonce=VALID_NONCE,
        )
    )
    open_world = svc.handle_think(
        api.ThinkRequest(prompt="Predict tomorrow's stock-market winner.", mode="fast")
    )

    assert exact.epistemics["decision"] == "answered"
    assert "14" in exact.output
    assert exact.confidence is None
    assert exact.epistemics["confidence_kind"] == "deterministic_assurance_not_probability"
    assert exact.proof_capsule["coverage"]["complete"] is True
    assert exact.output == "The exact result is 14."
    assert "claimed" not in exact.audit_receipts
    assert "display_result" not in json.dumps(exact.audit_receipts, sort_keys=True)
    assert exact.telemetry["api_fresh_recompute"] is True
    assert open_world.epistemics["decision"] == "abstained"
    assert open_world.confidence is None
    assert open_world.audit_receipts == {}
    assert "999" not in open_world.output
    assert open_world.audit_receipts == {}


def test_think_latency_includes_fresh_api_verifier(monkeypatch):
    query = "What is 2 + 3 * 4?"
    grounded = json.loads(json.dumps(api.grounding.finalize_grounded_response("", query)))
    clock = {"seconds": 10.0}

    claimed = epistemics.verified_exact_decision(
        reason="engine_claim",
        claim_scope="engine supplied scope",
        verifier_id="grounding_runtime.finalize_grounded_response",
        request_sha256="1" * 64,
        output_sha256="2" * 64,
        verifier_receipt_sha256="3" * 64,
        request_nonce_sha256="4" * 64,
        surface="think",
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
        api.ThinkRequest(prompt=query, mode="fast", request_nonce=VALID_NONCE)
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
        payload = svc.handle_think(
            api.ThinkRequest(prompt=query, mode="solve", request_nonce=VALID_NONCE)
        ).to_dict()
    elif surface == "solve":
        payload = svc.handle_solve(
            api.SolveRequest(query=query, request_nonce=VALID_NONCE)
        )
    else:
        payload = svc.handle_scientific(
            api.ScientificRequest(query=query, request_nonce=VALID_NONCE)
        )

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
        query="Under constant acceleration with initial velocity 0 m/s, acceleration 9.8 m/s^2, and time 5 s, what is the final velocity?",
        request_nonce=VALID_NONCE,
    )
    resp = svc.handle_scientific(req)

    assert resp.get("status") == "answered"
    assert resp["answer_authority"] is True
    assert resp["confidence"] is None
    assert resp["assurance_kind"] == "deterministic_assurance_not_probability"
    assert resp["proof_capsule"]["coverage"]["complete"] is True
    assert "receipt" in resp
    answer_display = resp.get("result", {}).get("answer", {}).get("display", "")
    assert "49" in answer_display


def test_exact_public_surfaces_require_a_valid_request_nonce():
    svc = api.NexusApiService()
    arithmetic = "What is 2 + 3 * 4?"
    science = (
        "Under constant acceleration with initial velocity 0 m/s, acceleration "
        "9.8 m/s^2, and time 5 s, what is the final velocity?"
    )

    solve = svc.handle_solve(api.SolveRequest(query=arithmetic))
    scientific = svc.handle_scientific(api.ScientificRequest(query=science))
    chat = svc.handle_chat(
        api.ChatTurnRequest(session_id="missing-nonce", message=arithmetic)
    )
    think = svc.handle_think(api.ThinkRequest(prompt=arithmetic, mode="solve"))
    streamed = list(
        svc.handle_think_stream(
            api.ThinkRequest(prompt=arithmetic, mode="solve", stream=True)
        )
    )

    for payload in (solve, scientific, chat, think.to_dict()):
        assert payload["epistemics"]["decision"] == "abstained"
        assert payload["epistemics"]["reason"] == "valid_request_nonce_required"
        assert payload["epistemics"]["answer_authority"] is False
        assert payload.get("answer_authority", False) is False
        assert not payload.get("proof_capsule")
    assert chat["conversation_state_updated"] is False
    assert streamed[-1]["event"] == "done"
    assert streamed[-1]["status"] == "abstained"
    assert streamed[-1]["proof_capsule_sha256"] == ""
    telemetry = next(event for event in streamed if event["event"] == "telemetry")
    assert telemetry["proof_capsule"] == {}


def test_grounded_class_without_independent_checker_is_deferred():
    svc = api.NexusApiService()
    query = "What is the area of a rectangle with length 8 cm and width 5 cm?"
    grounded = api._fresh_verified_grounding(query)

    assert grounded is not None
    assert grounded["answer_receipt"]["problem_class"] == "geometry"
    assert api._proof_capsule(query, grounded, "solve", VALID_NONCE) is None

    result = svc.handle_solve(api.SolveRequest(query=query, request_nonce=VALID_NONCE))
    assert result["status"] == "abstained"
    assert result["answer_authority"] is False
    assert "proof_capsule" not in result


def test_chat_routes_supported_math_to_proof_carrying_answer_only():
    svc = api.NexusApiService()
    nonce = "0123456789abcdef0123456789abcdef"

    exact = svc.handle_chat(
        api.ChatTurnRequest(
            session_id="proof-chat",
            message="What is 2 + 3 * 4?",
            request_nonce=nonce,
        )
    )
    open_world = svc.handle_chat(
        api.ChatTurnRequest(
            session_id="proof-chat",
            message="What will tomorrow's best stock be?",
            request_nonce="fedcba9876543210fedcba9876543210",
        )
    )

    assert exact["status"] == "answered"
    assert exact["reply"] == "The exact result is 14."
    assert exact["output"] == exact["reply"]
    assert exact["confidence"] is None
    assert exact["conversation_state_updated"] is False
    assert exact["proof_capsule"]["bindings"]["request_nonce_sha256"] == api.proof.text_sha256(nonce)
    assert open_world["status"] == "analysis_only"
    assert open_world["answer_authority"] is False
    assert open_world["confidence"] is None
    assert "proof_capsule" not in open_world


def test_renderer_verify_requires_exact_query_output_display_capsule_and_nonce():
    svc = api.NexusApiService()
    query = "What is 2 + 3 * 4?"
    nonce = "0123456789abcdef0123456789abcdef"
    solved = svc.handle_solve(api.SolveRequest(query=query, request_nonce=nonce))

    accepted = svc.handle_verify(
        api.VerifyRequest(
            query=query,
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="solve",
            proof_capsule=solved["proof_capsule"],
            request_nonce=nonce,
        )
    )
    exact_replay = svc.handle_verify(
        api.VerifyRequest(
            query=query,
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="solve",
            proof_capsule=solved["proof_capsule"],
            request_nonce=nonce,
        )
    )
    replayed = svc.handle_verify(
        api.VerifyRequest(
            query="What is 8 + 9?",
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="solve",
            proof_capsule=solved["proof_capsule"],
            request_nonce=nonce,
        )
    )
    cross_surface_replay = svc.handle_verify(
        api.VerifyRequest(
            query=query,
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="chat",
            proof_capsule=solved["proof_capsule"],
            request_nonce=nonce,
        )
    )

    assert accepted["verified"] is True
    assert accepted["confidence"] is None
    assert exact_replay["verified"] is False
    assert exact_replay["reason"] == "request_nonce_replayed"
    assert exact_replay["fresh_verifier_calls"] == 0
    assert replayed["verified"] is False
    assert replayed["capsule_sha256"] == ""
    assert cross_surface_replay["verified"] is False
    assert cross_surface_replay["capsule_sha256"] == ""


def test_renderer_verify_enforces_the_scientific_surface_policy():
    svc = api.NexusApiService()
    query = "What is 2 + 3 * 4?"
    nonce = "0123456789abcdef0123456789abcdef"
    grounded = api._fresh_verified_grounding(query)
    assert grounded is not None
    capsule = api._proof_capsule(query, grounded, "scientific", nonce)
    assert capsule is not None
    verdict = svc.handle_verify(
        api.VerifyRequest(
            query=query,
            output=grounded["text"],
            display_answer=capsule["result"]["display_answer"],
            surface="scientific",
            proof_capsule=capsule,
            request_nonce=nonce,
        )
    )

    assert verdict["verified"] is False
    assert verdict["fresh_verifier_calls"] == 1
    assert verdict["capsule_sha256"] == ""


def test_renderer_verify_rejects_empty_nonce_without_running_grounding():
    svc = api.NexusApiService()
    query = "What is 2 + 3 * 4?"
    solved = svc.handle_solve(api.SolveRequest(query=query, request_nonce=VALID_NONCE))
    request = api.VerifyRequest(
        query=query,
        output=solved["output"],
        display_answer=solved["display_answer"],
        surface="solve",
        proof_capsule=solved["proof_capsule"],
        request_nonce="",
    )

    first = svc.handle_verify(request)
    second = svc.handle_verify(request)

    assert first["verified"] is False
    assert second["verified"] is False
    assert first["reason"] == second["reason"] == "valid_request_nonce_required"
    assert first["fresh_verifier_calls"] == second["fresh_verifier_calls"] == 0


@pytest.mark.parametrize(
    "invalid_nonce",
    ["short", "non-ascii-caf\u00e9-0001", "x" * 129, "contains.periods.0001"],
)
def test_renderer_verify_rejects_invalid_nonce_before_grounding(invalid_nonce):
    svc = api.NexusApiService()
    query = "What is 2 + 3 * 4?"
    solved = svc.handle_solve(api.SolveRequest(query=query, request_nonce=VALID_NONCE))

    verdict = svc.handle_verify(
        api.VerifyRequest(
            query=query,
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="solve",
            proof_capsule=solved["proof_capsule"],
            request_nonce=invalid_nonce,
        )
    )

    assert verdict["verified"] is False
    assert verdict["reason"] == "valid_request_nonce_required"
    assert verdict["fresh_verifier_calls"] == 0
    assert verdict["capsule_sha256"] == ""


def test_renderer_verify_can_share_durable_nonce_ledger_across_service_instances(tmp_path):
    db = tmp_path / "verification-nonces.sqlite"
    query = "What is 2 + 3 * 4?"
    nonce = "durable-api-nonce"
    first_service = api.NexusApiService(verification_nonce_db=db)
    solved = first_service.handle_solve(api.SolveRequest(query=query, request_nonce=nonce))
    request = api.VerifyRequest(
        query=query,
        output=solved["output"],
        display_answer=solved["display_answer"],
        surface="solve",
        proof_capsule=solved["proof_capsule"],
        request_nonce=nonce,
    )

    accepted = first_service.handle_verify(request)
    second_service = api.NexusApiService(verification_nonce_db=db)
    replayed = second_service.handle_verify(request)

    assert accepted["verified"] is True
    assert replayed["verified"] is False
    assert replayed["reason"] == "request_nonce_replayed"
    assert replayed["fresh_verifier_calls"] == 0


def test_injected_nonce_ledger_identity_and_capacity_fail_closed():
    store = api.nonce_ledger.InMemoryNonceLedger(ttl_seconds=60, max_entries=1)
    svc = api.NexusApiService(verification_nonce_store=store)
    query = "What is 2 + 3 * 4?"
    first_nonce = "capacity-nonce-first-0001"
    second_nonce = "capacity-nonce-second-0002"

    assert svc._verification_nonce_store is store

    def verification_request(nonce):
        solved = svc.handle_solve(api.SolveRequest(query=query, request_nonce=nonce))
        return api.VerifyRequest(
            query=query,
            output=solved["output"],
            display_answer=solved["display_answer"],
            surface="solve",
            proof_capsule=solved["proof_capsule"],
            request_nonce=nonce,
        )

    first_request = verification_request(first_nonce)
    second_request = verification_request(second_nonce)
    accepted = svc.handle_verify(first_request)
    at_capacity = svc.handle_verify(second_request)
    replayed = svc.handle_verify(first_request)

    assert accepted["verified"] is True
    assert at_capacity["verified"] is False
    assert at_capacity["reason"] == "nonce_ledger_capacity_exhausted"
    assert at_capacity["renderer_may_mark_numeric_claims_verified"] is False
    assert replayed["verified"] is False
    assert replayed["reason"] == "request_nonce_replayed"
    assert replayed["fresh_verifier_calls"] == 0


def test_think_stream_preserves_order_and_carries_proof_capsule():
    svc = api.NexusApiService()
    req = api.ThinkRequest(
        prompt="What is 2 + 3 * 4?",
        mode="solve",
        request_nonce="stream-proof-nonce",
        stream=True,
    )

    events = list(svc.handle_think_stream(req))
    assert events[0]["event"] == "start"
    assert events[0]["stream_contract"] == "nexus-sse-proof-carrying-v1"
    tokens = [event for event in events if event["event"] == "token"]
    assert tokens
    assert [event["chunk_index"] for event in tokens] == list(range(len(tokens)))
    assert all(event["chunk_count"] == len(tokens) for event in tokens)

    telemetry = next(event for event in events if event["event"] == "telemetry")
    response = svc.handle_think(req)
    assert "".join(event["delta"] for event in tokens) == response.output
    assert telemetry["proof_capsule"] == response.proof_capsule
    done = events[-1]
    assert done["event"] == "done"
    assert done["status"] == response.epistemics["decision"]
    assert done["proof_capsule_sha256"] == response.proof_capsule["capsule_sha256"]


def test_api_solve_is_verifier_first_and_withholds_legacy_match():
    svc = api.NexusApiService()

    exact = svc.handle_solve(
        api.SolveRequest(query="What is 2 + 3 * 4?", request_nonce=VALID_NONCE)
    )
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
    assert telem["service"] == "NexusMind Experimental Evidence API v82"
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


def test_fastapi_serves_studio_on_same_origin_when_available(tmp_path):
    app = api.create_app(api.NexusApiService())
    if not hasattr(app, "routes"):
        pytest.skip("FastAPI is not installed")

    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/studio" in paths
    assert "/v1/verify" in paths
    assert api._STUDIO_PATH.is_file()

    # Exercise actual JSON-body resolution. With postponed annotations and
    # function-local Pydantic classes, these routes previously misclassified
    # `req` as a query parameter and every browser POST returned HTTP 422.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from fastapi.testclient import TestClient

        client = TestClient(app)
    solve = client.post(
        "/v1/solve",
        json={"query": "What is 2 + 3 * 4?", "request_nonce": VALID_NONCE},
    )
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

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["verification_nonce_backend"] == "in_memory_process_local"
    assert health.json()["verification_nonce_ttl_seconds"] == 15 * 60
    assert health.json()["verification_nonce_required"] is True
    assert health.json()["independent_witness_required"] is True

    streamed = client.post(
        "/v1/think",
        json={
            "prompt": "What is 2 + 3 * 4?",
            "mode": "solve",
            "request_nonce": "api-stream-proof-nonce",
            "stream": True,
        },
    )
    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    assert streamed.headers["x-nexus-stream-contract"] == "nexus-sse-proof-carrying-v1"
    assert "event: start" in streamed.text
    assert '"event":"telemetry"' in streamed.text
    assert '"proof_capsule"' in streamed.text
    assert "event: done" in streamed.text

    durable_app = api.create_app(
        api.NexusApiService(verification_nonce_db=tmp_path / "health-nonces.sqlite")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        durable_client = TestClient(durable_app)
    durable_health = durable_client.get("/health")
    assert durable_health.status_code == 200
    assert durable_health.json()["verification_nonce_backend"] == "sqlite_durable"


def test_endpoints_bell_resonance_compare(tmp_path):
    import warnings
    from starlette.testclient import TestClient

    app = api.create_app(api.NexusApiService())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(app)

    # Test /v1/quantum/bell
    b_resp = client.post("/v1/quantum/bell", json={"shots": 300})
    assert b_resp.status_code == 200
    b_data = b_resp.json()
    assert "chsh_s_quantum" in b_data
    assert b_data["violates_classical_bound"] is True

    # Test /v1/resonance
    r_resp = client.post("/v1/resonance", json={"query": "solve mathematical equation"})
    assert r_resp.status_code == 200
    r_data = r_resp.json()
    assert r_data["dominant_archetype"] == "logos"

    # Test /v1/compare
    c_resp = client.post(
        "/v1/compare",
        json={"query_a": "What is 2+2?", "query_b": "What is 3+3?", "mode_a": "auto", "mode_b": "auto"}
    )
    assert c_resp.status_code == 200
    c_data = c_resp.json()
    assert "jensen_shannon_divergence" in c_data
    assert "summary_verdict" in c_data

    # Test enhanced /v1/entropy with complexity
    e_resp = client.post("/v1/entropy", json={"rule": 30, "count": 10})
    assert e_resp.status_code == 200
    e_data = e_resp.json()
    assert e_data["complexity_class"] == "Class 3 (Chaotic)"
    assert "spatial_entropy" in e_data
