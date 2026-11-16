"""Verification suite for Supermix v97 neural architecture and SCF v91.5 upgrades."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import mimomix_core as mc  # noqa: E402
from nexus_api import NexusApiService, SCFRunRequest, create_app  # noqa: E402
from scf import CognitiveFabricRuntime, SCFConfig, SCFFeatureFlags  # noqa: E402
from scf.adaptive import SCFAdaptiveComputeController  # noqa: E402
from scf.collective import SCFCollectiveOrchestrator  # noqa: E402
from scf.learning import SCFLearningGate  # noqa: E402
from scf.memory import SCFLayeredMemory  # noqa: E402
from scf.models import MiMoMixAdapter  # noqa: E402
from scf.reasoning import SCFReasoningCoordinator  # noqa: E402
from scf.verification import SCFVerificationGate  # noqa: E402


def _synthetic_hemi(path: Path, n: int = 8) -> str:
    rng = np.random.default_rng(7)
    roles = np.array(["sensory", "central", "central", "descending"] * 2, dtype=object)
    sign = np.array([1, -1, 1, 1, 1, -1, 1, 1], dtype=np.int8)
    mat = rng.random((n, n)) * (rng.random((n, n)) < 0.35)
    np.fill_diagonal(mat, 0.0)
    for post in range(n):
        if mat[post].sum() == 0:
            mat[post, (post + 1) % n] = 0.4
    post, pre = np.nonzero(mat)
    frac = mat[post, pre] / mat.sum(1)[post]
    np.savez_compressed(
        path,
        module_role=roles,
        module_sign=sign,
        edge_post=post.astype(np.int64),
        edge_pre=pre.astype(np.int64),
        edge_fraction=frac.astype(np.float32),
        rewired_post=post.astype(np.int64),
        rewired_pre=pre.astype(np.int64),
        module_side=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
    )
    return str(path)


def test_v97_connectome_hebbian_and_thinking_entropy_halt(tmp_path: Path):
    hemi_path = _synthetic_hemi(tmp_path / "hemi.npz")
    base_kwargs = dict(
        vocab_size=64,
        hidden_size=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=36,
        moe_intermediate_size=16,
        n_routed_experts=4,
        n_mtp_layers=1,
        sliding_window=8,
        native_context=32,
        max_position_embeddings=32,
        thinking_latent_dim=8,
        thinking_cycles=3,
        thinking_max_cycles=4,
        use_cns_core=True,
        cns_nodes=8,
        cns_after_layer=1,
        cns_steps=2,
        cns_temporal=True,
        cns_dual_soma=True,
        cns_commissural_gate=True,
        cns_glial_homeostasis=True,
    )
    torch.manual_seed(0)
    m_off = mc.MiMoMixModel(mc.MiMoMixConfig(**base_kwargs)).eval()
    m_off.cns_core.load_graph(hemi_path)

    torch.manual_seed(0)
    m_on = mc.MiMoMixModel(
        mc.MiMoMixConfig(
            **base_kwargs, cns_hebbian_trace=True, thinking_entropy_halt=True
        )
    ).eval()
    m_on.cns_core.load_graph(hemi_path)

    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        logits_off = m_off(x, return_mtp=False).logits
        logits_on = m_on(x, return_mtp=False).logits
        # Zero-init hebbian_strength preserves initial logits identically
        assert torch.allclose(logits_off, logits_on, atol=1e-6)

    tele_off = m_off.telemetry()
    tele_on = m_on.telemetry()
    assert "hebbian_trace" not in tele_off["cns_core"]
    assert "cycle_entropies" not in tele_off["thinking"]
    assert tele_on["cns_core"]["hebbian_trace"] is True
    assert tele_on["cns_core"]["hebbian_gain_mean"] == pytest.approx(1.0)
    assert len(tele_on["thinking"]["cycle_entropies"]) == 3

    # Active hebbian trace modulates connectome excitability cleanly
    with torch.no_grad():
        m_on.cns_core.gate.fill_(0.5)
        m_on.cns_core.hebbian_strength.fill_(1.0)
        _ = m_on(x, return_mtp=False)
    assert 0.75 <= m_on.cns_core.telemetry()["hebbian_gain_mean"] <= 1.25


def test_scf_connectome_atlas_and_extended_math_verification():
    gate = SCFVerificationGate()
    v_pct = gate.verify(query="What is 25% of 232?")
    assert v_pct["verified"] is True and v_pct["verified_answer"] == "58"

    v_math = gate.verify(query="Solve this basic math problem: 472 - 378")
    assert v_math["verified"] is True and v_math["verified_answer"] == "94"

    v_neg = gate.verify(query="What is NOT 25% of 232?")
    assert v_neg["verified"] is False

    atlas_path = ROOT / "datasets" / "v93_malecns" / "malecns_sided_types.npz"
    if atlas_path.is_file():
        v_type = gate.verify(query="How many Li20 neurons does the male CNS contain?")
        assert v_type["verified"] is True and v_type["verified_answer"] == "40"

        v_side = gate.verify(query="type ORN_VA2 right side neuron count in the male CNS")
        assert v_side["verified"] is True and v_side["verified_answer"] == "41"

        v_pair = gate.verify(query="In the male CNS, how many synapses does type Tm2 make onto type Dm3c?")
        assert v_pair["verified"] is True and v_pair["verified_answer"] == "12210"


def test_scf_v95_research_integrations_and_flag_toggles():
    orchestrator = SCFCollectiveOrchestrator()
    coll = orchestrator.gather_and_synthesize(
        "compare candidates",
        specialist_families=["NexusMind", "MiMoMix"],
        candidate_generators={
            "NexusMind": lambda _q: {"family": "NexusMind", "answer": "48", "confidence": 0.92, "verified": True},
            "MiMoMix": lambda _q: {"family": "MiMoMix", "answer": "48", "confidence": 0.65, "verified": False},
        },
        failure_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert coll["ensemble_forensics"]["answer_authority"] is False
    assert "tether" in coll["ensemble_forensics"]
    assert "faultline" in coll["ensemble_forensics"]

    crc = SCFAdaptiveComputeController().calibrate_selective_risk(
        [0.15, 0.28, 0.42, 0.55, 0.68, 0.80, 0.91],
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        target_risk_alpha=0.25,
    )
    assert crc["answer_authority"] is False
    assert "aurc" in crc["risk_coverage_profile"]

    shield = SCFReasoningCoordinator().shield_plan_action(candidate_action="EXEC")
    assert shield["answer_authority"] is False and "shield_verdict" in shield

    assay = SCFLayeredMemory().run_memory_assay(recovery_trajectory=[0.34, 0.52, 0.74, 0.88])
    assert assay["answer_authority"] is False and "map_mechanism" in assay

    # Turning SCF_VERIFICATION_ENABLED off disables deterministic override
    rt = CognitiveFabricRuntime(
        SCFConfig(flags=SCFFeatureFlags(SCF_ENABLED=True, SCF_VERIFICATION_ENABLED=False))
    )
    no_ver = rt.run_task("What is 2 + 2?")
    assert no_ver["verified"] is False
    assert no_ver["verification"]["verifier_source"] == "disabled_by_flag"


def test_scf_live_neural_adapter_and_learning_gate(tmp_path: Path):
    gate = SCFLearningGate()
    missing = gate.benchmark_checkpoint_pair(tmp_path / "a.pt", tmp_path / "b.pt")
    assert missing["completed"] is False and missing["promoted"] is False
    assert gate.automatic_promotion_enabled is False

    adapter = MiMoMixAdapter()
    ckpts = adapter.discover_talk_checkpoints()
    if ckpts:
        gen = adapter.infer_text(
            "can you help me with tests",
            flags=SCFFeatureFlags.all_enabled(speculative_decoding=True),
            max_new_tokens=32,
        )
        assert gen["generated"] is True
        assert gen["verified"] is False
        assert gen["answer_authority"] is False
        assert gen["model_conditional"] is True
        assert len(gen["answer"]) > 0


def test_nexus_api_scf_endpoints():
    from starlette.testclient import TestClient

    svc = NexusApiService()
    status = svc.handle_scf_status()
    assert status["automatic_promotion_enabled"] is False
    assert status["default_flags"]["SCF_ENABLED"] is False

    fallback = svc.handle_scf_run(SCFRunRequest(query="What is 2 + 2?", enable_scf=False))
    assert fallback["fallback_to_legacy"] is True
    assert fallback["answer_authority"] is False

    app = create_app(svc)
    client = TestClient(app)
    r_status = client.get("/v1/scf/status")
    assert r_status.status_code == 200
    assert r_status.json()["internal_name"] == "Cognitive Fabric"

    r_run = client.post(
        "/v1/scf/run",
        json={"query": "What is 25% of 232?", "enable_scf": True, "enable_verification": True},
    )
    assert r_run.status_code == 200
    payload = r_run.json()
    assert payload["verified"] is True
    assert payload["final_answer"] == "58"
    assert payload["answer_authority"] is True


def test_v97_evolution_manifest_exists_if_built():
    manifest = ROOT / "datasets" / "v97" / "v97_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["total_rows"] == 4800
        assert "percent" in data["task_counts"]
        assert "cns_type_count" in data["task_counts"]


def test_chat_interfaces_integrate_v97_speculative_and_scf_verification():
    import supermix_chat_server as chat_srv
    from starlette.testclient import TestClient

    discovered = chat_srv.discover_default_models(ROOT)
    v97_rel = "output/v97_scf_connectome/v97_scf_connectome.selected.pt"
    if (ROOT / v97_rel).is_file():
        assert "v97_scf_connectome" in discovered
        # Test Studio /v1/chat neural integration with v97 checkpoint
        client = TestClient(create_app())
        r_chat = client.post(
            "/v1/chat",
            json={
                "session_id": "test_v97_neural",
                "message": "can you help me with tests",
                "use_neural_model": True,
                "model_checkpoint": v97_rel,
                "speculative": True,
                "max_new_tokens": 32,
            },
        )
        assert r_chat.status_code == 200
        d = r_chat.json()
        assert d["status"] == "analysis_only"
        assert d["answer_authority"] is False
        assert d["neural_telemetry"]["run_name"] == "v97_scf_connectome"
        arch = d["neural_telemetry"]["architecture_telemetry"]
        assert arch["cns_core"]["hebbian_trace"] is True
        assert len(arch["thinking"]["cycle_entropies"]) >= 1

        # Reject unknown checkpoint path outside discovered allowlist
        r_bad = client.post(
            "/v1/chat",
            json={
                "session_id": "test_v97_bad",
                "message": "hello",
                "use_neural_model": True,
                "model_checkpoint": "../../nonexistent.pt",
            },
        )
        assert r_bad.status_code == 422

        # Test supermix_chat_server speculative mode + scf_verify on v97 checkpoint
        reg = chat_srv.ModelRegistry({"v97": str(ROOT / v97_rel)}, max_resident=1)
        flask_client = chat_srv.build_app(reg).test_client()
        bad_scf = flask_client.post("/api/chat", json={"message": "hello", "scf_verify": "yes"})
        assert bad_scf.status_code == 400

        sse = flask_client.post(
            "/api/chat",
            json={
                "message": "What is 25% of 232?",
                "model": "v97",
                "mode": "speculative",
                "scf_verify": True,
                "max_new_tokens": 48,
            },
        )
        assert sse.status_code == 200
        lines = sse.get_data(as_text=True).splitlines()
        done = json.loads(next(lines[i + 1][6:] for i, ln in enumerate(lines) if ln == "event: done"))
        assert done["scf_check"]["verified"] is True
        assert done["scf_check"]["verified_answer"] == "58"
        assert done["speculative_stats"]["acceptance_length"] >= 1.0
        assert done["architecture_telemetry"]["cns_core"]["hebbian_trace"] is True
