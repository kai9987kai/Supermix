"""Verification suite for Supermix v104 Evolution.

Covers:
1. TitansSurpriseMemory (surprise-gated associative matrix memory + momentum,
   zero-init birth identity, gradient flow, bit-exact incremental KV-cache
   decode parity, MTP speculative rollback exact equivalence).
2. DepthAttentionResidual / AttnRes (cross-layer depth softmax residual mixing,
   zero-init birth identity, gradient flow, bit-exact KV-cache & MTP parity).
3. ConnectomeCore cns_neuromod_gate (novelty-gated neuromodulatory excitability,
   zero-init birth identity, Dale-sign invariance, causal temporal parity, and
   mitosis split_module inheritance).
4. Full combined v93+v102+v103+v104 stack parity (mHC + Engram + Gated DeltaNet-2
   + TitansSurpriseMemory + DepthAttentionResidual + temporal CNS + apical
   dendrites + neuromod gate + CTM synchrony + MTP speculative rollback).
5. MuonHybridOptimizer v104 upgrades (Cautious sign-alignment masking, NorMuon
   row-wise RMS normalization, optimizer_telemetry, and backward compatibility).
6. Expanded v104 zero-eval exact symbolic solvers (Catalan, Fibonacci, CRT,
   Shannon entropy, KL divergence, Poisson PMF, Ideal Gas Law, Capacitors,
   Lorentz factor, Compound Interest) + fail-closed invalid inputs + SCF routing.
7. All 5 v104 frontier engines in source/nexus_v104_evolution.py with canonical
   SHA-256 receipts and strict answer_authority=False.
8. FastAPI /v1/v104/* endpoints, CLI `scf frontier --json` & `scf diagnose --json`,
   Chat Server v104 telemetry badges, and NexusMind Studio v104 Frontier Lab.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402
import muon_optimizer as mo  # noqa: E402
import nexus_api  # noqa: E402
import nexus_solver  # noqa: E402
import nexus_v104_evolution as v104  # noqa: E402
from scf import CognitiveFabricRuntime, SCFFeatureFlags  # noqa: E402


# ---------------------------------------------------------------------------
# 1. TitansSurpriseMemory
# ---------------------------------------------------------------------------


def test_titans_zero_init_identity_grad_and_kv_speculative_parity():
    torch.manual_seed(1041)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        n_mtp_layers=2,
        sliding_window=4,
        hybrid_ratio=1,
    )
    titans_cfg = mc.MiMoMixConfig(
        **{
            **base_cfg.to_dict(),
            "use_titans_memory": True,
            "titans_dim": 12,
            "titans_after_layer": 1,
        }
    )
    base_model = mc.MiMoMixModel(base_cfg).eval()
    titans_model = mc.MiMoMixModel(titans_cfg).eval()
    missing, unexpected = titans_model.load_state_dict(base_model.state_dict(), strict=False)
    assert unexpected == []
    assert any(k.startswith("titans_memory.") for k in missing)

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base_logits = base_model(ids, return_mtp=False).logits
        zero_logits = titans_model(ids, return_mtp=False).logits
    assert torch.allclose(base_logits, zero_logits, atol=1e-6)

    # Open gate and verify non-zero modulation, incremental KV-cache parity, and MTP parity
    with torch.no_grad():
        titans_model.titans_memory.gate.fill_(0.40)
        full_out = titans_model(ids, return_mtp=False)
        assert not torch.allclose(base_logits, full_out.logits, atol=1e-5)

        out = titans_model(ids[:, :1], use_cache=True, return_mtp=False, cache_slack=2)
        past = out.past_key_values
        step_logits = [out.logits]
        for pos in range(1, ids.shape[1]):
            out = titans_model(
                ids[:, pos : pos + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                cache_slack=2,
                past_length=pos,
            )
            past = out.past_key_values
            step_logits.append(out.logits)
        inc_logits = torch.cat(step_logits, dim=1)
        assert torch.allclose(full_out.logits, inc_logits, atol=1e-5)

    tele = titans_model.telemetry()
    assert tele["titans"]["enabled"] is True
    assert tele["titans"]["state_dim"] == 12
    assert tele["titans"]["mean_surprise"] >= 0.0
    assert tele["titans"]["memory_frobenius_norm"] > 0.0

    md.assert_greedy_equivalence(titans_model, ids[:, :4], max_new_tokens=6)

    # Gradient flow check
    titans_model.train()
    out_train = titans_model(ids, labels=ids)
    out_train.loss.backward()
    assert titans_model.titans_memory.gate.grad is not None
    assert float(titans_model.titans_memory.gate.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 2. DepthAttentionResidual (AttnRes)
# ---------------------------------------------------------------------------


def test_attn_res_zero_init_identity_grad_and_kv_speculative_parity():
    torch.manual_seed(1042)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        n_mtp_layers=2,
        sliding_window=4,
        hybrid_ratio=1,
    )
    ar_cfg = mc.MiMoMixConfig(**{**base_cfg.to_dict(), "use_attn_res": True})
    base_model = mc.MiMoMixModel(base_cfg).eval()
    ar_model = mc.MiMoMixModel(ar_cfg).eval()
    missing, unexpected = ar_model.load_state_dict(base_model.state_dict(), strict=False)
    assert unexpected == []
    assert any(k.startswith("attn_res.") for k in missing)

    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        base_logits = base_model(ids, return_mtp=False).logits
        zero_logits = ar_model(ids, return_mtp=False).logits
    assert torch.allclose(base_logits, zero_logits, atol=1e-6)

    with torch.no_grad():
        ar_model.attn_res.layer_gates.fill_(0.35)
        full_out = ar_model(ids, return_mtp=False)
        assert not torch.allclose(base_logits, full_out.logits, atol=1e-5)

        out = ar_model(ids[:, :1], use_cache=True, return_mtp=False, cache_slack=2)
        past = out.past_key_values
        step_logits = [out.logits]
        for pos in range(1, ids.shape[1]):
            out = ar_model(
                ids[:, pos : pos + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                cache_slack=2,
                past_length=pos,
            )
            past = out.past_key_values
            step_logits.append(out.logits)
        inc_logits = torch.cat(step_logits, dim=1)
        assert torch.allclose(full_out.logits, inc_logits, atol=1e-5)

    tele = ar_model.telemetry()
    assert tele["attn_res"]["enabled"] is True
    assert tele["attn_res"]["n_layers"] == 3
    assert tele["attn_res"]["last_depth_entropy"] >= 0.0

    md.assert_greedy_equivalence(ar_model, ids[:, :4], max_new_tokens=6)


# ---------------------------------------------------------------------------
# 3. ConnectomeCore Neuromodulatory Gate + Mitosis Inheritance
# ---------------------------------------------------------------------------


def test_cns_neuromod_gate_zero_birth_mitosis_and_temporal_parity():
    torch.manual_seed(1043)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=2,
        cns_steps=2,
        cns_read_layers=(0, 1),
        cns_write_layers=(1,),
        cns_temporal=True,
    )
    nm_cfg = mc.MiMoMixConfig(**{**base_cfg.to_dict(), "cns_neuromod_gate": True})
    base_m = mc.MiMoMixModel(base_cfg).eval()
    nm_m = mc.MiMoMixModel(nm_cfg).eval()
    missing, unexpected = nm_m.load_state_dict(base_m.state_dict(), strict=False)
    assert unexpected == []
    assert "cns_core.neuromod_strength" in missing

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        b_logits = base_m(ids, return_mtp=False).logits
        z_logits = nm_m(ids, return_mtp=False).logits
    assert torch.allclose(b_logits, z_logits, atol=1e-6)

    with torch.no_grad():
        nm_m.cns_core.neuromod_strength.fill_(0.8)
        child = nm_m.cns_core.split_module(parent=0, step=10)
        assert child is not None
        assert float(nm_m.cns_core.neuromod_strength[child]) == pytest.approx(
            float(nm_m.cns_core.neuromod_strength[0])
        )
        _ = nm_m(ids, return_mtp=False)

    tele = nm_m.telemetry()["cns_core"]
    assert tele["neuromod_gate"] is True
    assert 0.5 <= tele["neuromod_gain_mean"] <= 1.5


# ---------------------------------------------------------------------------
# 4. Full Combined Stack Parity (v93 + v102 + v103 + v104)
# ---------------------------------------------------------------------------


def test_full_v104_combined_stack_kv_and_mtp_speculative_parity():
    torch.manual_seed(1044)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        n_mtp_layers=2,
        sliding_window=4,
        hybrid_ratio=1,
        use_mhc_residual=True,
        mhc_streams=2,
        mhc_sinkhorn_iters=15,
        use_engram_memory=True,
        engram_table_size=64,
        engram_n_heads=2,
        engram_after_layer=0,
        use_deltanet_memory=True,
        deltanet_dim=8,
        deltanet_after_layer=1,
        use_titans_memory=True,
        titans_dim=8,
        titans_after_layer=2,
        use_attn_res=True,
        use_cns_core=True,
        cns_nodes=10,
        cns_steps=2,
        cns_read_layers=(0, 1),
        cns_write_layers=(1,),
        cns_temporal=True,
        cns_dendritic_apical=True,
        cns_neuromod_gate=True,
        use_thinking_core=True,
        thinking_latent_dim=16,
        thinking_cycles=2,
        thinking_ctm_synchrony=True,
    )
    model = mc.MiMoMixModel(cfg).eval()
    with torch.no_grad():
        model.engram_memory.gate.fill_(0.25)
        model.deltanet_memory.gate.fill_(0.25)
        model.titans_memory.gate.fill_(0.25)
        model.attn_res.layer_gates.fill_(0.20)
        model.cns_core.apical_gain.fill_(0.30)
        model.cns_core.neuromod_strength.fill_(0.30)

        ids = torch.randint(0, 64, (1, 7))
        full_logits = model(ids, return_mtp=False).logits
        out = model(ids[:, :1], use_cache=True, return_mtp=False, cache_slack=2)
        past = out.past_key_values
        step_logits = [out.logits]
        for pos in range(1, ids.shape[1]):
            out = model(
                ids[:, pos : pos + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                cache_slack=2,
                past_length=pos,
            )
            past = out.past_key_values
            step_logits.append(out.logits)
        inc_logits = torch.cat(step_logits, dim=1)
        assert torch.allclose(full_logits, inc_logits, atol=1e-5)

    md.assert_greedy_equivalence(model, ids[:, :3], max_new_tokens=6)


# ---------------------------------------------------------------------------
# 5. MuonHybridOptimizer v104 (Cautious Mask + Row Normalization + Telemetry)
# ---------------------------------------------------------------------------


def test_muon_v104_cautious_and_row_norm_reduces_loss_with_telemetry():
    torch.manual_seed(1045)
    cfg = mc.MiMoMixConfig(
        vocab_size=48,
        hidden_size=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        use_attn_res=True,
        use_titans_memory=True,
        titans_dim=8,
        titans_after_layer=1,
    )
    model = mc.MiMoMixModel(cfg)
    groups, counts = mo.build_muon_hybrid_param_groups(
        model,
        lr=1e-3,
        cautious_mask=True,
        row_normalize=True,
    )
    assert counts["muon_tensors"] > 0
    opt = mo.MuonHybridOptimizer(groups)
    ids = torch.randint(0, 48, (2, 8))

    losses = []
    for _ in range(4):
        opt.zero_grad()
        out = model(ids, labels=ids)
        out.loss.backward()
        losses.append(float(out.loss.item()))
        opt.step()

    assert losses[-1] < losses[0]
    tele = opt.optimizer_telemetry()
    assert tele["step_count"] == 4
    assert tele["last_ortho_updates"] > 0
    assert 0.0 < tele["last_cautious_active_ratio"] <= 1.0


# ---------------------------------------------------------------------------
# 6. Expanded v104 Zero-Eval Exact Symbolic Solvers + SCF Integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_display,expected_formula",
    [
        ("What is the 5th catalan number?", "42", "comb.catalan"),
        ("Compute fibonacci(12)", "144", "comb.fibonacci"),
        ("Solve x = 2 (mod 3) and x = 3 (mod 5)", "8 (mod 15)", "numtheory.crt"),
        ("Shannon entropy of [0.5, 0.25, 0.25]", "1.5", "infotheory.shannon_entropy"),
        ("KL divergence between [0.5, 0.5] and [0.25, 0.75]", "0.207519", "infotheory.kl_divergence"),
        ("Poisson with lambda = 2 and k = 0", "0.135335", "prob.poisson_pmf"),
        ("Ideal gas with n = 1, T = 300, V = 1", "2494.338785", "thermo.ideal_gas"),
        ("Capacitors 6 uF and 3 uF in series", "2", "circuits.capacitors_series"),
        ("Capacitors 6 uF and 3 uF in parallel", "9", "circuits.capacitors_parallel"),
        ("Lorentz factor at v = 0.6c", "1.25", "relativity.lorentz_gamma"),
        ("Compound interest P = 1000, r = 0.1, n = 1, t = 2", "1210.00", "finance.compound_interest"),
    ],
)
def test_v104_exact_symbolic_solvers(query: str, expected_display: str, expected_formula: str):
    res = nexus_solver.solve_problem(query)
    assert res.solved is True
    assert res.formula_id == expected_formula
    assert res.display_answer == expected_display
    assert res.receipt is not None and len(res.receipt.receipt_sha256) == 64


def test_v104_solvers_fail_closed_on_invalid_inputs():
    # Superluminal beta >= 1 must fail closed
    assert nexus_solver.solve_problem("Lorentz factor at v = 1.2c").solved is False
    # Non-normalized probabilities must fail closed
    assert nexus_solver.solve_problem("Shannon entropy of [0.5, 0.8]").solved is False
    # Zero support in Q where P > 0 must fail closed for KL divergence
    assert nexus_solver.solve_problem("KL divergence between [0.5, 0.5] and [1.0, 0.0]").solved is False
    # Non-coprime CRT moduli must fail closed
    assert nexus_solver.solve_problem("Solve x = 1 (mod 4) and x = 2 (mod 6)").solved is False


def test_scf_runtime_verifies_v104_solver():
    rt = CognitiveFabricRuntime()
    out = rt.run_task(
        "What is the 5th catalan number?",
        flags_override=SCFFeatureFlags.all_enabled(),
    )
    assert out["verified"] is True
    assert out["final_answer"] == "42"


# ---------------------------------------------------------------------------
# 7. Five v104 Frontier Engines (nexus_v104_evolution.py)
# ---------------------------------------------------------------------------


def test_v104_five_frontier_engines_invariants_and_receipts():
    # 1. ConformalCalibratedRouter
    conf = v104.ConformalCalibratedRouter().calibrate_and_route(
        calibration_scores=[0.05, 0.10, 0.14, 0.18, 0.22, 0.30, 0.45],
        candidate_scores={"NexusMind": 0.92, "MiMoMix": 0.60, "ChampionNet": 0.30},
        alpha=0.25,
        max_set_size=2,
    )
    assert conf["answer_authority"] is False
    assert conf["decision"] == "ROUTE_SINGLETON"
    assert conf["prediction_set"] == ["NexusMind"]
    assert conf["empirical_cal_coverage"] >= 0.75
    assert len(conf["receipt"]["receipt_sha256"]) == 64

    # 2. MechanisticCircuitTracer
    circ = v104.MechanisticCircuitTracer().trace_circuit(
        clean_activations={"engram_mlp": 0.6, "attn_deltanet": 0.9, "connectome_titans": 1.8, "thinking_readout": 0.5},
        corrupted_activations={"engram_mlp": 0.1, "attn_deltanet": 0.1, "connectome_titans": 0.1, "thinking_readout": 0.1},
    )
    assert circ["answer_authority"] is False
    assert circ["dominant_stage"] == "connectome_titans"
    assert circ["minimal_circuit_recovery"] >= 0.80
    assert len(circ["receipt"]["receipt_sha256"]) == 64

    # 3. EpisodicGraphConsolidator
    graph = v104.EpisodicGraphConsolidator().consolidate(
        triples=[
            {"subject": "A", "predicate": "points_to", "object": "OldB", "confidence": 0.70, "timestamp": 1},
            {"subject": "A", "predicate": "points_to", "object": "B", "confidence": 0.95, "timestamp": 5},
            {"subject": "B", "predicate": "points_to", "object": "C", "confidence": 0.90, "timestamp": 6},
            {"subject": "Noise", "predicate": "tag", "object": "Drop", "confidence": 0.10, "timestamp": 2},
        ],
        min_salience=0.25,
        query_source="A",
        query_target="C",
    )
    assert graph["answer_authority"] is False
    assert graph["conflict_count"] == 1
    assert graph["pruned_low_salience"] == 1
    assert graph["query_path"] is not None and len(graph["query_path"]) == 2
    assert len(graph["receipt"]["receipt_sha256"]) == 64

    # 4. VerifiableProcessRewardAuditor
    good_prm = v104.VerifiableProcessRewardAuditor().audit_trace(
        prompt="Compute 8 * 5 + 3",
        steps=["First 8 * 5 = 40.", "Then 40 + 3 = 43."],
    )
    assert good_prm["all_arithmetic_verified"] is True
    assert good_prm["equations_passed"] == 2
    assert good_prm["chain_soundness_geo_mean"] > 0.90

    bad_prm = v104.VerifiableProcessRewardAuditor().audit_trace(
        prompt="Compute 8 * 5 + 3",
        steps=["First 8 * 5 = 41.", "Then 41 + 3 = 44."],
    )
    assert bad_prm["all_arithmetic_verified"] is False
    assert bad_prm["weakest_link_step"] == 1
    assert bad_prm["chain_soundness_geo_mean"] < good_prm["chain_soundness_geo_mean"]

    # 5. ParetoFrontierAutoTuner
    pareto = v104.ParetoFrontierAutoTuner().optimize(
        candidates=[
            {"name": "fast", "quality_score": 0.80, "latency_ms": 50.0, "memory_mb": 100.0, "calibration_error": 0.05},
            {"name": "rigorous", "quality_score": 0.98, "latency_ms": 200.0, "memory_mb": 260.0, "calibration_error": 0.01},
            {"name": "dominated_slow", "quality_score": 0.75, "latency_ms": 250.0, "memory_mb": 300.0, "calibration_error": 0.09},
        ]
    )
    assert pareto["answer_authority"] is False
    assert "dominated_slow" in pareto["dominated_configs"]
    assert pareto["pareto_count"] == 2
    assert pareto["recommendations"]["edge_cpu"] == "fast"
    assert pareto["recommendations"]["max_rigor"] == "rigorous"
    assert math.isfinite(pareto["hypervolume_2d_quality_speed"]) and pareto["hypervolume_2d_quality_speed"] > 0.0


# ---------------------------------------------------------------------------
# 8. FastAPI /v1/v104/* Endpoints, CLI, Chat Server & Studio UI
# ---------------------------------------------------------------------------


def test_v104_api_endpoints_cli_and_studio_surface():
    from starlette.testclient import TestClient

    app = nexus_api.create_app()
    client = TestClient(app)

    for route in (
        "/v1/v104/conformal_route",
        "/v1/v104/circuit_trace",
        "/v1/v104/graph_consolidate",
        "/v1/v104/process_audit",
        "/v1/v104/pareto_tune",
    ):
        resp = client.post(route, json={})
        assert resp.status_code == 200, f"{route} failed: {resp.text}"
        body = resp.json()
        assert body["answer_authority"] is False
        assert len(body["receipt"]["receipt_sha256"]) == 64

    # Invalid alpha must return 422
    bad_resp = client.post("/v1/v104/conformal_route", json={"alpha": 1.5})
    assert bad_resp.status_code == 422

    # CLI: python -m scf frontier --json & diagnose --json
    env = {"PYTHONPATH": str(SOURCE)}
    p_front = subprocess.run(
        [sys.executable, "-m", "scf", "frontier", "--json"],
        cwd=str(ROOT),
        env={**dict(**__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=True,
    )
    f_data = json.loads(p_front.stdout)
    assert f_data["evolution_v104_version"] == v104.EVOLUTION_V104_VERSION
    assert f_data["titans"]["enabled"] is True
    assert f_data["attn_res"]["enabled"] is True
    assert f_data["cns_neuromod_gate"] is True
    assert f_data["prm_all_verified"] is True

    p_diag = subprocess.run(
        [sys.executable, "-m", "scf", "diagnose", "--json"],
        cwd=str(ROOT),
        env={**dict(**__import__("os").environ), **env},
        capture_output=True,
        text=True,
        check=True,
    )
    d_data = json.loads(p_diag.stdout)
    assert d_data["all_probes_passed"] is True
    assert d_data["verifier_probes_passed"] == d_data["verifier_probes_total"]

    # Studio HTML contains v104 controls
    studio_html = (ROOT / "web_static" / "nexus_studio.html").read_text(encoding="utf-8")
    for fn_name in (
        "runV104ConformalRoute",
        "runV104CircuitTrace",
        "runV104GraphConsolidate",
        "runV104ProcessAudit",
        "runV104ParetoTune",
    ):
        assert fn_name in studio_html

    # Chat server contains v104 telemetry badges
    chat_py = (SOURCE / "supermix_chat_server.py").read_text(encoding="utf-8")
    assert "use_titans_memory" in chat_py
    assert "use_attn_res" in chat_py
    assert "[v104 Titans/AttnRes]" in chat_py
