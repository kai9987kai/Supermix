"""Verification tests for Supermix v101 math, problem solving, ideation, conversation memory, and server UI."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def test_extended_mathematics_and_zero_eval_ast_solver() -> None:
    from nexus_solver import NexusSolver
    from scf.verification import SCFVerificationGate

    gate = SCFVerificationGate()
    solver = NexusSolver()

    cases = [
        ("Calculate (18 * 14 + 36) / 6", "48"),
        ("What is 2^10 - 24?", "1000"),
        ("Solve x + y = 10 and x - y = 2", "x = 6, y = 4"),
        ("Solve x^2 - 5x + 6 = 0", "x1 = 3, x2 = 2"),
        ("What comes next in the sequence: 2, 6, 12, 20, 30", "42"),
        ("What comes next in the sequence: 3, 9, 27, 81", "243"),
        ("What comes next in the sequence: 1, 1, 2, 3, 5, 8", "13"),
        ("Find the derivative of 3x^3 - 4x + 5 at x = 2", "32"),
        ("Integrate 6x^2 + 2x from x = 0 to x = 2", "20"),
        ("Find the median of 4, 8, 6, 5, 3", "5"),
        ("Find the gcd of 84, 120", "12"),
    ]
    for query, expected in cases:
        res = gate.verify(query=query)
        assert res["verified"] is True, query
        assert res["verified_answer"] == expected, (query, res)

    # Negation & unsafe code injection remain fail-closed
    assert gate.verify(query="Do not calculate (18 * 14 + 36) / 6")["verified"] is False
    assert gate.verify(query="Evaluate __import__('os').system('echo bad')")["verified"] is False

    # Athenaeum-inspired algebraic equivalence verification
    eq_ok = solver.verify_algebraic_equivalence("(a + b)^2", "a^2 + 2*a*b + b^2")
    assert eq_ok["equivalent"] is True and eq_ok["answer_authority"] is True
    eq_bad = solver.verify_algebraic_equivalence("(a + b)^2", "a^2 + b^2")
    assert eq_bad["equivalent"] is False and eq_bad["answer_authority"] is False


def test_map_elites_ideation_and_crosswire_problem_solving() -> None:
    from scf import CognitiveFabricRuntime, SCFFeatureFlags
    from scf.collective import SCFCollectiveOrchestrator

    # Collective orchestrator penalizes intermediate arithmetic mistakes (e.g. 72 - 78 = -16)
    orch = SCFCollectiveOrchestrator()
    coll = orch.gather_and_synthesize(
        "Solve this basic math problem: 472 - 378",
        specialist_families=("MiMoMix", "CognitiveLeap"),
        candidate_generators={
            "MiMoMix": lambda _q: {
                "family": "MiMoMix",
                "answer": "400 - 300 = 100, 72 - 78 = -16, total 84",
                "confidence": 0.65,
                "verified": False,
            },
            "CognitiveLeap": lambda _q: {
                "family": "CognitiveLeap",
                "answer": "400 - 300 = 100, 72 - 78 = -6, total 94",
                "confidence": 0.65,
                "verified": False,
            },
        },
        max_candidates=2,
    )
    by_fam = {c["family"]: c for c in coll["candidates"]}
    assert by_fam["MiMoMix"]["trace_audit_failures"] == 1
    assert by_fam["CognitiveLeap"]["trace_audit_failures"] == 0
    assert coll["best_candidate"]["family"] == "CognitiveLeap"

    rt = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()

    # Ideation uses MAP-Elites diversity + honors session constraints without false authority
    idea_out = rt.run_task(
        "Brainstorm ways to speed up Python CPU inference without losing accuracy",
        flags_override=flags,
        recent_turns=[("Remember I prefer Python and concise structured steps.", "Got it.")],
    )
    assert idea_out["routing"]["assessment"]["submode"] == "ideation"
    assert idea_out["assistant_synthesis"]["mode"] == "ideation"
    assert idea_out["assistant_synthesis"]["answer_authority"] is False
    assert len(idea_out["ideation"]["concepts"]) >= 3
    cells = {tuple(c["metadata"]["map_elites_cell"]) for c in idea_out["ideation"]["concepts"]}
    assert len(cells) >= 2

    # Problem solving triggers decomposition + Crosswire clarifying probes even on prose debug queries
    prob_out = rt.run_task(
        "How should I debug an intermittent async race condition step by step?",
        flags_override=flags,
        recent_turns=[("Remember I prefer Python and concise structured steps.", "Got it.")],
    )
    syn = prob_out["assistant_synthesis"]
    assert syn["mode"] == "problem_solving"
    assert syn["answer_authority"] is False
    assert len(syn["bullets"]) >= 4
    assert any("Memory alignment:" in b for b in syn["bullets"])

    vague_prob = rt.run_task(
        "Why is it failing and how should I optimize this?",
        flags_override=flags,
    )
    assert vague_prob["assistant_synthesis"]["mode"] == "problem_solving"
    assert len(vague_prob["assistant_synthesis"]["clarifying_questions"]) >= 1

    # Conversation continuity honors standing constraints without authority escalation
    conv_out = rt.run_task(
        "Okay what should our next focus be?",
        flags_override=flags,
        recent_turns=[
            ("Remember I prefer Python and short concise answers.", "Got it."),
            ("Also we use pytest for all checks.", "Noted."),
        ],
    )
    assert conv_out["assistant_synthesis"]["mode"] == "conversation_continuity"
    assert conv_out["memory"]["working_state"]["style_preference"] == "concise"
    assert conv_out["memory"]["working_state"]["turn_count"] == 2
    assert conv_out["memory"]["memory_firewall_enforced"] is True


def test_chat_server_cognitive_synthesis_ui_and_multi_turn_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    import supermix_chat_server as scs

    class _StubV101:
        def telemetry(self):
            return {
                "cns_core": {
                    "alive_nodes": 810,
                    "edges_installed": 19980,
                    "hebbian_trace": True,
                    "hebbian_gain_mean": 1.002,
                    "predictive_error": True,
                    "pred_error_norm": 0.0005,
                    "divisive_norm": True,
                    "div_norm_factor_mean": 1.003,
                    "spectral_radius_est": 0.90,
                },
                "thinking": {"cycles_used": 2, "exit_reason": "entropy_halt", "cycle_entropies": [0.40, 0.14]},
            }

    loaded = scs.LoadedModel(
        name="v101_scf_cognitive",
        checkpoint="stub_v101.pt",
        model=_StubV101(),
        tokenizer=object(),
        extra={"run_name": "v101_scf_cognitive"},
        lock=threading.Lock(),
    )
    registry = scs.ModelRegistry({"v101_scf_cognitive": "stub_v101.pt"}, max_resident=1)
    monkeypatch.setattr(registry, "acquire", lambda _name: loaded)
    monkeypatch.setattr(scs, "stream_speculative_tokens", lambda *args, **kwargs: iter(["Start with three steps."]))

    # Keep neural history_turns=0 while verifying Cognitive Fabric receives bounded session turns
    app = scs.build_app(registry, max_concurrency=2, history_turns=0)
    client = app.test_client()

    html = client.get("/").get_data(as_text=True)
    assert "[v101 Cognitive]" in html
    assert "assistant_synthesis" in html
    assert "buf.indexOf('\n\n')" not in html

    # Turn 1 sets user preference
    r1 = client.post(
        "/api/chat",
        json={
            "message": "Remember I prefer Python and concise structured steps.",
            "model": "v101_scf_cognitive",
            "mode": "speculative",
            "session_id": "s-v101",
            "scf_verify": True,
            "scf_runtime": True,
        },
    )
    assert r1.status_code == 200
    r1_raw = r1.get_data(as_text=True)
    assert "event: done" in r1_raw

    # Turn 2 triggers MAP-Elites ideation backed by session memory
    r2 = client.post(
        "/api/chat",
        json={
            "message": "Brainstorm ways to debug slow tests faster",
            "model": "v101_scf_cognitive",
            "mode": "speculative",
            "session_id": "s-v101",
            "scf_verify": True,
            "scf_runtime": True,
        },
    )
    assert r2.status_code == 200
    done_line = [ln for ln in r2.get_data(as_text=True).splitlines() if ln.startswith("data: ")][-1]
    done = json.loads(done_line[len("data: "):])
    rt = done["scf_runtime"]
    assert rt["submode"] == "ideation"
    assert rt["memory_summary"]["turn_count"] == 1
    assert rt["memory_summary"]["style_preference"] == "concise"
    assert rt["assistant_synthesis"]["mode"] == "ideation"
    assert len(rt["assistant_synthesis"]["bullets"]) >= 3


def test_v101_corpus_builder_and_gate_disjoint_integrity(tmp_path: Path) -> None:
    from build_v101_cognitive_mix import build_v101_cognitive_mix

    out_jsonl = tmp_path / "v101.jsonl"
    out_manifest = tmp_path / "v101_manifest.json"
    manifest = build_v101_cognitive_mix(out_jsonl, out_manifest, seed=101)
    assert manifest["schema"] == "supermix-v101-cognitive-mix-v1"
    assert manifest["total_rows"] >= 9000
    assert manifest["generator_seed"] == 10101
    assert "math_chat" in manifest["task_counts"]
    assert "subtraction" in manifest["task_counts"]
    assert "problem_solving" in manifest["task_counts"]
    assert "ideation" in manifest["task_counts"]
