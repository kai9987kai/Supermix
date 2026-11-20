"""Verification suite for Supermix v103 Frontier Synthesis.

Covers:
1. DeepSeek-style EngramConditionalMemory (O(1) hashed N-gram lookup, zero-init
   identity, exact incremental KV-cache decode parity, MTP speculative parity).
2. GatedDeltaNetMemory (2026 decoupled channel-wise decay/erase/write fast-weight
   matrix state, zero-init identity, exact KV-cache & MTP speculative parity).
3. Combined v102+v103+v93 stack parity (mHC + Engram + DeltaNet-2 + temporal CNS
   + apical dendrites + CTM synchrony + MTP speculative rollback).
4. Evo-OMEGA SHY synaptic homeostasis sleep cycle on ConnectomeCore.
5. All 5 v103 stdlib+NumPy synthesis engines in source/nexus_v103_synthesis.py
   (Continuum/BTSP/Hopfield + MNEMOSYNE 5-mode lesion assay, Top-K SAE causal
   ablation probe, FractalConnectomeAnalyzer, CausewaySensitivityWorkbench,
   NeuroHiveResourcePlanner).
6. Exact rational v103 linear-algebra & series solvers in NexusSolver + SCF routing.
7. FastAPI /v1/v103/* endpoints, /v1/signals, CLI `scf frontier --json`,
   Chat Server v103 UI badges, and NexusMind Studio fail-closed v102/v103 lab.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402
import nexus_api  # noqa: E402
import nexus_solver  # noqa: E402
import nexus_v103_synthesis as v103  # noqa: E402
from scf import CognitiveFabricRuntime, SCFFeatureFlags  # noqa: E402


# ---------------------------------------------------------------------------
# 1. EngramConditionalMemory
# ---------------------------------------------------------------------------


def test_engram_zero_init_identity_and_kv_speculative_parity():
    torch.manual_seed(1031)
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
    eng_cfg = mc.MiMoMixConfig(
        **{
            **base_cfg.to_dict(),
            "use_engram_memory": True,
            "engram_table_size": 128,
            "engram_n_heads": 2,
            "engram_after_layer": 0,
        }
    )
    base_model = mc.MiMoMixModel(base_cfg).eval()
    eng_model = mc.MiMoMixModel(eng_cfg).eval()
    missing, unexpected = eng_model.load_state_dict(base_model.state_dict(), strict=False)
    assert unexpected == []
    assert any(k.startswith("engram_memory.") for k in missing)

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        base_logits = base_model(ids, return_mtp=False).logits
        eng_zero_logits = eng_model(ids, return_mtp=False).logits
    assert torch.allclose(base_logits, eng_zero_logits, atol=1e-6)

    # Open the residual gate so Engram actively modulates representations
    with torch.no_grad():
        eng_model.engram_memory.gate.fill_(0.35)
        full_out = eng_model(ids, return_mtp=False)
        assert not torch.allclose(base_logits, full_out.logits, atol=1e-5)

        # Incremental token-by-token parity starting from length-1 prefill
        step_logits = []
        past = None
        for t in range(ids.shape[1]):
            step = eng_model(
                ids[:, t : t + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                past_length=t,
            )
            step_logits.append(step.logits)
            past = step.past_key_values
        inc_logits = torch.cat(step_logits, dim=1)
        assert float((full_out.logits - inc_logits).abs().max()) < 1e-5

    # Exact MTP speculative rollback equivalence (both short and normal prompts)
    md.assert_greedy_equivalence(eng_model, ids[:, :1], max_new_tokens=6)
    md.assert_greedy_equivalence(eng_model, ids[:, :4], max_new_tokens=6)
    telem = full_out.telemetry
    assert telem["engram"]["enabled"] is True
    assert telem["engram"]["n_heads"] == 2
    assert telem["engram"]["gate_mean_abs"] == pytest.approx(0.35, abs=1e-5)


# ---------------------------------------------------------------------------
# 2. GatedDeltaNetMemory
# ---------------------------------------------------------------------------


def test_deltanet_zero_init_identity_and_kv_speculative_parity():
    torch.manual_seed(1032)
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
    dn_cfg = mc.MiMoMixConfig(
        **{
            **base_cfg.to_dict(),
            "use_deltanet_memory": True,
            "deltanet_dim": 16,
            "deltanet_after_layer": 1,
        }
    )
    base_model = mc.MiMoMixModel(base_cfg).eval()
    dn_model = mc.MiMoMixModel(dn_cfg).eval()
    missing, unexpected = dn_model.load_state_dict(base_model.state_dict(), strict=False)
    assert unexpected == []
    assert any(k.startswith("deltanet_memory.") for k in missing)

    ids = torch.randint(0, 64, (1, 9))
    with torch.no_grad():
        base_logits = base_model(ids, return_mtp=False).logits
        dn_zero_logits = dn_model(ids, return_mtp=False).logits
    assert torch.allclose(base_logits, dn_zero_logits, atol=1e-6)

    with torch.no_grad():
        dn_model.deltanet_memory.gate.fill_(0.30)
        full_out = dn_model(ids, return_mtp=False)
        assert not torch.allclose(base_logits, full_out.logits, atol=1e-5)

        step_logits = []
        past = None
        for t in range(ids.shape[1]):
            step = dn_model(
                ids[:, t : t + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                past_length=t,
            )
            step_logits.append(step.logits)
            past = step.past_key_values
        inc_logits = torch.cat(step_logits, dim=1)
        assert float((full_out.logits - inc_logits).abs().max()) < 1e-5

    md.assert_greedy_equivalence(dn_model, ids[:, :4], max_new_tokens=6)
    telem = full_out.telemetry
    assert telem["deltanet"]["enabled"] is True
    assert telem["deltanet"]["state_dim"] == 16
    assert 0.0 < telem["deltanet"]["mean_decay_gate"] < 1.0
    assert telem["deltanet"]["state_frobenius_norm"] > 0.0


# ---------------------------------------------------------------------------
# 3. Combined v102 + v103 + v93 Full Stack Parity
# ---------------------------------------------------------------------------


def test_combined_mhc_engram_deltanet_temporal_cns_speculative_parity():
    torch.manual_seed(1033)
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
        use_thinking_core=True,
        thinking_cycles=2,
        thinking_ctm_synchrony=True,
        use_mhc_residual=True,
        mhc_streams=2,
        mhc_sinkhorn_iters=20,
        use_cns_core=True,
        cns_nodes=8,
        cns_steps=2,
        cns_after_layer=1,
        cns_temporal=True,
        cns_dendritic_apical=True,
        use_engram_memory=True,
        engram_table_size=128,
        engram_n_heads=2,
        engram_after_layer=0,
        use_deltanet_memory=True,
        deltanet_dim=16,
        deltanet_after_layer=1,
    )
    model = mc.MiMoMixModel(cfg).eval()
    with torch.no_grad():
        # Install a simple active connectome graph and open every gate
        model.cns_core.mask.fill_(0.0)
        for i in range(8):
            model.cns_core.mask[i, (i + 1) % 8] = 1.0
            model.cns_core.mask[i, (i + 2) % 8] = 1.0
        model.cns_core.edge_logit.fill_(-1.0)
        model.cns_core.gate.fill_(0.20)
        model.cns_core.apical_gain.fill_(0.25)
        model.engram_memory.gate.fill_(0.25)
        model.deltanet_memory.gate.fill_(0.25)
        model.thinking_core.ctm_out.weight.normal_(0.0, 0.05)

    ids = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        full_out = model(ids, return_mtp=False)
        step_logits = []
        past = None
        for t in range(ids.shape[1]):
            step = model(
                ids[:, t : t + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                past_length=t,
            )
            step_logits.append(step.logits)
            past = step.past_key_values
        inc_logits = torch.cat(step_logits, dim=1)
        assert float((full_out.logits - inc_logits).abs().max()) < 1e-5

    md.assert_greedy_equivalence(model, ids[:, :3], max_new_tokens=5)
    assert full_out.telemetry["mhc"]["enabled"] is True
    assert full_out.telemetry["engram"]["enabled"] is True
    assert full_out.telemetry["deltanet"]["enabled"] is True
    assert full_out.telemetry["cns_core"]["dendritic_apical"] is True
    assert full_out.telemetry["thinking"]["ctm_synchrony"] is True


# ---------------------------------------------------------------------------
# 4. Evo-OMEGA SHY Synaptic Homeostasis Sleep Cycle
# ---------------------------------------------------------------------------


def test_connectome_shy_sleep_cycle_prunes_weak_preserves_dale_and_criticality():
    cfg = mc.MiMoMixConfig(
        vocab_size=32,
        hidden_size=16,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        intermediate_size=32,
        use_moe=False,
        use_cns_core=True,
        cns_nodes=6,
        cns_steps=2,
        cns_after_layer=0,
    )
    core = mc.ConnectomeCore(cfg)
    with torch.no_grad():
        core.mask.zero_()
        core.sign.copy_(torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]))
        core.edge_logit.fill_(-10.0)
        for i in range(6):
            # Strong primary synapse
            j_strong = (i + 1) % 6
            core.mask[i, j_strong] = 1.0
            core.edge_logit[i, j_strong] = 1.2
            # Ultra-weak noise synapse
            j_weak = (i + 2) % 6
            core.mask[i, j_weak] = 1.0
            core.edge_logit[i, j_weak] = -4.5  # softplus(-4.5) ~ 0.011

    signs_before = core.sign.clone()
    report = core.synaptic_homeostasis_sleep_cycle(
        downscale_factor=0.90,
        prune_floor=0.05,
        target_radius=0.85,
        power_iters=10,
    )
    assert report["edges_before"] == 12.0
    assert report["edges_pruned"] == 6.0
    assert report["edges_after"] == 6.0
    assert torch.equal(core.sign, signs_before)
    # Every postsynaptic neuron retained its strongest synapse
    assert int((core.mask.sum(dim=1) >= 1.0).sum()) == 6
    # Spectral radius moved smoothly toward target 0.85
    assert abs(report["rho_after"] - 0.85) < abs(report["rho_before"] - 0.85)
    # Dale's law still holds on effective weights
    w_eff = core.weight()
    for col in range(6):
        if signs_before[col] > 0:
            assert bool((w_eff[:, col] >= -1e-7).all())
        else:
            assert bool((w_eff[:, col] <= 1e-7).all())


# ---------------------------------------------------------------------------
# 5. v103 Synthesis Engines (source/nexus_v103_synthesis.py)
# ---------------------------------------------------------------------------


def test_continuum_btsp_hopfield_mnemosyne_lesion_assay():
    mem_a = v103.ContinuumBTSPHopfieldMemory(dim=12, out_dim=4, btsp_bits=32, seed=103)
    res_a = mem_a.run_benchmark(n_patterns=8, steps=24)
    mem_b = v103.ContinuumBTSPHopfieldMemory(dim=12, out_dim=4, btsp_bits=32, seed=103)
    res_b = mem_b.run_benchmark(n_patterns=8, steps=24)

    assert res_a == res_b
    assert res_a["answer_authority"] is False
    assert len(res_a["receipt_sha256"]) == 64
    assert res_a["continuum_final_mse"] < res_a["continuum_initial_mse"]
    assert res_a["btsp_one_shot_overlap"] == pytest.approx(1.0)
    les = res_a["mnemosyne_lesion_cosine"]
    assert set(les.keys()) == set(v103.MNEMOSYNE_LESION_MODES)
    assert les["erased"] == 0.0
    assert les["memory_free"] == 0.0
    assert les["intact"] > 0.85
    assert les["intact"] > les["shifted"]
    assert les["intact"] > les["random_retrieval"]


def test_topk_sae_causal_ablation_probe():
    rng = np.random.default_rng(103)
    latent = rng.exponential(1.2, size=(48, 3))
    basis = rng.normal(0.0, 1.0, size=(3, 8))
    samples = latent @ basis + rng.normal(0.0, 0.03, size=(48, 8))

    probe = v103.TopKSparseAutoencoderProbe(input_dim=8, dict_size=24, k_active=3, seed=103)
    res = probe.fit_and_probe(samples, epochs=40, lr=0.08)
    assert res["answer_authority"] is False
    assert len(res["receipt_sha256"]) == 64
    assert res["final_mse"] < res["initial_mse"]
    assert res["reconstruction_r2"] > 0.5
    assert res["mean_l0_active"] <= 3.0 + 1e-6
    assert res["causal_ablation_shift"] > res["control_ablation_shift"]
    assert res["causal_specificity_ratio"] > 1.0


def test_fractal_connectome_analyzer():
    res = v103.FractalConnectomeAnalyzer.compare_topologies(n_nodes=27, seed=103)
    assert res["answer_authority"] is False
    assert len(res["receipt_sha256"]) == 64
    topos = res["topologies"]
    assert set(topos.keys()) == {"sierpinski", "hilbert", "small_world", "random_control"}
    assert topos["sierpinski"]["clustering_coefficient"] > topos["hilbert"]["clustering_coefficient"]
    assert topos["sierpinski"]["clustering_coefficient"] > topos["random_control"]["clustering_coefficient"]
    assert topos["sierpinski"]["spectral_radius"] > 2.0
    with pytest.raises(ValueError):
        v103.FractalConnectomeAnalyzer.build_topology("nonexistent_graph")


def test_causeway_sensitivity_workbench_detects_reversal():
    policies = {
        "aggressive_fast": {"accuracy_weight": 0.90, "cost_penalty": 0.25, "safety_margin": 0.15},
        "balanced_verified": {"accuracy_weight": 0.84, "cost_penalty": 0.30, "safety_margin": 0.85},
    }
    res = v103.CausewaySensitivityWorkbench.evaluate_policies(
        policies=policies,
        assumptions={"stress_amplitude": 0.25, "risk_aversion": 0.40, "cost_sensitivity": 0.20},
        sensitivity_flips={"stress_amplitude": 0.95, "risk_aversion": 1.80, "cost_sensitivity": 0.60},
        seeds=(11, 29, 47, 73),
    )
    assert res["answer_authority"] is False
    assert len(res["receipt_sha256"]) == 64
    assert res["baseline_winner"] == "aggressive_fast"
    assert "stress_amplitude" in res["winner_reversing_assumptions"]
    assert res["decision_robust_across_lenses"] is False


def test_neurohive_resource_planner_conservation_and_repair():
    tasks = [
        {"id": "t1", "token_cost": 50, "thermal_delta": 20.0},
        {"id": "t2", "token_cost": 80, "thermal_delta": 25.0},
        {"id": "t3", "token_cost": 120, "thermal_delta": 30.0},
        {"id": "t4", "token_cost": 60, "thermal_delta": 20.0},
    ]
    res = v103.NeuroHiveResourcePlanner().simulate_dispatch(
        tasks=tasks,
        token_budget=500,
        thermal_limit=70.0,
        fault_rate=1.0,
        enable_mnt_repair=True,
        seed=103,
    )
    assert res["answer_authority"] is False
    assert len(res["receipt_sha256"]) == 64
    assert res["budget_conserved"] is True
    assert res["spent_tokens"] + res["remaining_tokens"] == 500
    assert res["faults_injected"] == 4
    assert res["faults_repaired"] == 4
    assert res["thermal_throttles"] >= 1
    assert len(res["completed_tasks"]) == 4
    assert res["swarm_intelligence_index"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. Exact Symbolic v103 Solvers & SCF Routing
# ---------------------------------------------------------------------------


def test_v103_exact_solvers_and_refusal_guards():
    solver = nexus_solver.NexusSolver()

    r_inv = solver.solve("inverse of [[4, 7], [2, 6]]")
    assert r_inv.solved is True
    assert r_inv.display_answer == "[[3/5, -7/10], [-1/5, 2/5]]"

    r_sing = solver.solve("inverse of [[1, 2], [2, 4]]")
    assert r_sing.solved is False

    r_eig = solver.solve("eigenvalues of [[4, 1], [2, 3]]")
    assert r_eig.solved is True
    assert r_eig.display_answer == "lambda1 = 5, lambda2 = 2"

    r_complex_eig = solver.solve("eigenvalues of [[0, -1], [1, 0]]")
    assert r_complex_eig.solved is False

    r_dot = solver.solve("dot product of [1, 2, 3] and [4, -5, 6]")
    assert r_dot.solved is True
    assert r_dot.display_answer == "12"

    r_cross = solver.solve("cross product of [1, 0, 0] and [0, 1, 0]")
    assert r_cross.solved is True
    assert r_cross.display_answer == "[0, 0, 1]"

    r_gfin = solver.solve("sum of geometric series a = 3, r = 2, n = 4")
    assert r_gfin.solved is True
    assert r_gfin.display_answer == "45"

    r_ginf = solver.solve("sum of infinite geometric series a = 6, r = 1/3")
    assert r_ginf.solved is True
    assert r_ginf.display_answer == "9"

    r_gdiv = solver.solve("sum of infinite geometric series a = 6, r = 2")
    assert r_gdiv.solved is False

    r_poly = solver.solve("evaluate 2x^3 - 3x^2 + 5 at x = 4")
    assert r_poly.solved is True
    assert r_poly.display_answer == "85"

    fabric = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()
    scf_inv = fabric.run_task("inverse of [[4, 7], [2, 6]]", flags_override=flags)
    assert scf_inv["verified"] is True
    assert scf_inv["assistant_synthesis"]["answer_authority"] is True
    assert "[[3/5, -7/10], [-1/5, 2/5]]" in scf_inv["final_answer"]

    scf_eig = fabric.run_task("eigenvalues of [[4, 1], [2, 3]]", flags_override=flags)
    assert scf_eig["verified"] is True
    assert "5" in scf_eig["final_answer"] and "2" in scf_eig["final_answer"]


# ---------------------------------------------------------------------------
# 7. FastAPI Endpoints, CLI, Chat Server & Studio Integration
# ---------------------------------------------------------------------------


def test_v103_api_endpoints_and_signals():
    from fastapi.testclient import TestClient

    svc = nexus_api.NexusApiService()
    signals = svc.handle_signals()
    assert "v103_frontier_synthesis" in signals
    assert all(signals["v103_frontier_synthesis"].values())

    client = TestClient(nexus_api.create_app(svc))
    for path, body in (
        ("/v1/v103/continuum_memory", {"n_patterns": 6, "steps": 16}),
        ("/v1/v103/sae_probe", {"dict_size": 16, "k_active": 2, "epochs": 20}),
        ("/v1/v103/fractal_topology", {"n_nodes": 18}),
        ("/v1/v103/causeway_sensitivity", {}),
        ("/v1/v103/neurohive_plan", {"token_budget": 600, "fault_rate": 0.2}),
    ):
        resp = client.post(path, json=body)
        assert resp.status_code == 200, (path, resp.text)
        data = resp.json()
        assert data["answer_authority"] is False
        assert len(data["receipt_sha256"]) == 64


def test_scf_frontier_cli_includes_v103_synthesis_checks():
    proc = subprocess.run(
        [sys.executable, "-m", "scf", "frontier", "--json"],
        cwd=str(SOURCE),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["synthesis_version"] == v103.FRONTIER_V103_VERSION
    assert payload["answer_authority"] is False
    assert payload["engram"]["enabled"] is True
    assert payload["deltanet"]["enabled"] is True
    assert payload["btsp_one_shot_overlap"] == pytest.approx(1.0)
    assert payload["mnemosyne_intact_vs_erased"] > 0.8
    assert payload["fractal_highest_clustering"] == "sierpinski"
    assert payload["neurohive_budget_conserved"] is True


def test_studio_and_chat_server_expose_v103_surfaces():
    studio_html = (ROOT / "web_static" / "nexus_studio.html").read_text(encoding="utf-8")
    for token in (
        'id="panel-v103-frontier"',
        "/v1/v103/continuum_memory",
        "/v1/v103/sae_probe",
        "/v1/v103/fractal_topology",
        "/v1/v103/causeway_sensitivity",
        "/v1/v103/neurohive_plan",
        "/v1/frontier/cvar_audit",
        "/v1/frontier/carrier_atlas",
        'for="v103MemPatterns"',
    ):
        assert token in studio_html

    chat_src = (SOURCE / "supermix_chat_server.py").read_text(encoding="utf-8")
    assert "Engram" in chat_src
    assert "DeltaNet2" in chat_src
    assert "Matrix 2x2 Inv" in chat_src

    # Full inline <script> syntax check on NexusMind Studio (catches latent brace bugs)
    js_check = subprocess.run(
        [
            "node",
            "-e",
            "const fs=require('fs'),vm=require('vm');"
            "const h=fs.readFileSync(process.argv[1],'utf8');"
            "const s=h.split('<script>')[1].split('</script>')[0];"
            "new vm.Script(s);",
            str(ROOT / "web_static" / "nexus_studio.html"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert js_check.returncode == 0, js_check.stderr


def test_warm_start_grafting_supports_v102_v103_zero_init_modules(tmp_path):
    import mimomix_text as text_utils
    import train_mimomix_generalisation as gen

    tok = text_utils.WordTokenizer.build(["hello world", "test math 2 + 2 = 4"], max_vocab=64)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        n_mtp_layers=1,
        use_thinking_core=True,
        thinking_cycles=2,
    )
    base_model = mc.MiMoMixModel(base_cfg).eval()
    ckpt_path = tmp_path / "base.pt"
    torch.save(
        {
            "config": base_cfg.to_dict(),
            "state_dict": base_model.state_dict(),
            "tokenizer": tok.to_dict(),
            "extra": {"run_name": "base", "steps": 10},
        },
        ckpt_path,
    )

    graft_cfg = mc.MiMoMixConfig(
        **{
            **base_cfg.to_dict(),
            "use_mhc_residual": True,
            "mhc_streams": 2,
            "thinking_ctm_synchrony": True,
            "use_engram_memory": True,
            "engram_table_size": 64,
            "engram_n_heads": 2,
            "engram_after_layer": 0,
            "use_deltanet_memory": True,
            "deltanet_dim": 16,
            "deltanet_after_layer": 1,
        }
    )
    graft_model = mc.MiMoMixModel(graft_cfg).eval()
    prov = gen.load_initial_weights(graft_model, tok, str(ckpt_path))
    assert set(prov["allowed_missing"].keys()) == {
        "mhc_layers",
        "ctm_synchrony",
        "engram_memory",
        "deltanet_memory",
    }
    prefixes = gen.new_parameter_prefixes(graft_cfg)
    for expected in ("mhc_layers.", "thinking_core.ctm_", "engram_memory.", "deltanet_memory."):
        assert expected in prefixes
    ids = torch.randint(0, tok.vocab_size, (1, 5))
    with torch.no_grad():
        assert torch.allclose(
            base_model(ids, return_mtp=False).logits,
            graft_model(ids, return_mtp=False).logits,
            atol=1e-5,
        )
