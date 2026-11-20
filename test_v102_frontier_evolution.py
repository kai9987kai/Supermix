"""Verification suite for Supermix v102 Frontier Architecture, Neuro-Symbolic & Cognitive Fabric."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def test_mhc_birkhoff_polytope_identity_decode_and_speculative_parity() -> None:
    import mimomix_core as mc
    import mimomix_decoding as md

    torch.manual_seed(102)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        n_mtp_layers=1,
        use_thinking_core=False,
        hybrid_ratio=1,
        sliding_window=4,
        use_mhc_residual=False,
    )
    torch.manual_seed(19)
    base_model = mc.MiMoMixModel(base_cfg)
    base_model.eval()

    mhc_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        n_mtp_layers=1,
        use_thinking_core=False,
        hybrid_ratio=1,
        sliding_window=4,
        use_mhc_residual=True,
        mhc_streams=3,
        mhc_sinkhorn_iters=20,
    )
    torch.manual_seed(19)
    mhc_model = mc.MiMoMixModel(mhc_cfg)
    mhc_model.load_state_dict(base_model.state_dict(), strict=False)
    mhc_model.eval()

    x = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        y_base = base_model(x).logits
        y_mhc = mhc_model(x).logits
    # Identity mapping at birth: identical streams + doubly stochastic I_n preserves exact output.
    assert torch.allclose(y_base, y_mhc, atol=1e-5)

    # Birkhoff polytope invariants (doubly stochastic + spectral norm <= 1) after random mixing logits.
    with torch.no_grad():
        for mhc_layer in mhc_model.mhc_layers:
            mhc_layer.res_logits.normal_(0.0, 1.2)
            mhc_layer.pre_logits.normal_(0.0, 0.5)
            mhc_layer.post_logits.normal_(0.0, 0.5)
            h_res = mhc_layer.doubly_stochastic_matrix()
            assert torch.all(h_res >= 0.0)
            assert torch.allclose(h_res.sum(dim=-1), torch.ones(3), atol=1e-3)
            assert torch.allclose(h_res.sum(dim=-2), torch.ones(3), atol=1e-3)
            sv = torch.linalg.svdvals(h_res)
            assert float(sv.max()) <= 1.0 + 1e-3
            # Higher Sinkhorn iteration count converges monotonically tighter on the Birkhoff polytope.
            tight = mc.ManifoldHyperConnection(n_streams=3, sinkhorn_iters=40)
            tight.res_logits.copy_(mhc_layer.res_logits)
            h_tight = tight.doubly_stochastic_matrix()
            assert torch.allclose(h_tight.sum(dim=-1), torch.ones(3), atol=1e-5)
            assert torch.allclose(h_tight.sum(dim=-2), torch.ones(3), atol=1e-5)

        y_mixed = mhc_model(x).logits
        assert not torch.allclose(y_base, y_mixed, atol=1e-5)

        # Incremental KV-cache decode parity
        past = None
        steps = []
        for t in range(x.shape[1]):
            out = mhc_model(x[:, t : t + 1], past_key_values=past, use_cache=True, past_length=t)
            past = out.past_key_values
            steps.append(out.logits)
        assert float((y_mixed - torch.cat(steps, dim=1)).abs().max()) < 1e-5

    # Exact MTP speculative == greedy parity under mHC
    spec_report = md.assert_greedy_equivalence(mhc_model, x, max_new_tokens=6)
    assert spec_report["tokens"] == 6

    tele = mhc_model.telemetry()
    assert tele["mhc"]["enabled"] is True
    assert tele["mhc"]["streams"] == 3
    assert tele["mhc"]["max_doubly_stochastic_error"] < 1e-3


def test_cns_dendritic_apical_identity_causal_temporal_and_mitosis() -> None:
    import mimomix_core as mc

    torch.manual_seed(31)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=4,
        cns_steps=2,
        cns_after_layer=0,
        cns_temporal=True,
        cns_dendritic_apical=False,
    )
    torch.manual_seed(37)
    base_model = mc.MiMoMixModel(base_cfg)
    base_model.eval()

    apical_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=4,
        cns_steps=2,
        cns_after_layer=0,
        cns_temporal=True,
        cns_dendritic_apical=True,
    )
    torch.manual_seed(37)
    apical_model = mc.MiMoMixModel(apical_cfg)
    apical_model.load_state_dict(base_model.state_dict(), strict=False)
    apical_model.eval()

    x = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        for core in (base_model.cns_core, apical_model.cns_core):
            core.in_mask.fill_(1.0)
            core.out_mask.fill_(1.0)
            core.mask.fill_(1.0)
            core.edge_logit.fill_(0.0)
            core.gate.fill_(0.5)
        y_base = base_model(x).logits
        y_init = apical_model(x).logits
    assert torch.allclose(y_base, y_init, atol=1e-6)

    # Non-zero apical gain modulates rates while preserving exact causal incremental decode parity.
    with torch.no_grad():
        apical_model.cns_core.apical_gain.fill_(1.15)
        apical_model.cns_core.apical_leak_logit.fill_(0.4)
        y_active = apical_model(x).logits
        assert not torch.allclose(y_base, y_active, atol=1e-5)

        past = None
        inc = []
        for t in range(x.shape[1]):
            out = apical_model(x[:, t : t + 1], past_key_values=past, use_cache=True, past_length=t)
            past = out.past_key_values
            inc.append(out.logits)
        assert float((y_active - torch.cat(inc, dim=1)).abs().max()) < 1e-5

    tel = apical_model.telemetry()["cns_core"]
    assert tel["dendritic_apical"] is True
    assert tel["apical_gain_mean_abs"] > 1.0

    # Neurogenesis mitosis cleanly copies apical compartment parameters.
    child = apical_model.cns_core.split_module(parent=0, step=1)
    assert child is not None
    assert float(apical_model.cns_core.apical_gain.detach()[child]) == pytest.approx(1.15, abs=1e-6)
    assert float(apical_model.cns_core.apical_leak_logit.detach()[child]) == pytest.approx(0.4, abs=1e-6)


def test_thinking_ctm_synchrony_identity_and_coherence() -> None:
    import mimomix_core as mc

    torch.manual_seed(41)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        use_thinking_core=True,
        thinking_latent_dim=16,
        thinking_cycles=3,
        thinking_ctm_synchrony=False,
    )
    torch.manual_seed(43)
    base_model = mc.MiMoMixModel(base_cfg)
    base_model.eval()

    ctm_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        use_thinking_core=True,
        thinking_latent_dim=16,
        thinking_cycles=3,
        thinking_ctm_synchrony=True,
    )
    torch.manual_seed(43)
    ctm_model = mc.MiMoMixModel(ctm_cfg)
    ctm_model.load_state_dict(base_model.state_dict(), strict=False)
    ctm_model.eval()

    x = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        y_base = base_model(x).logits
        y_init = ctm_model(x).logits
    assert torch.allclose(y_base, y_init, atol=1e-6)
    tel_init = ctm_model.telemetry()["thinking"]
    assert tel_init["ctm_synchrony"] is True
    assert tel_init["ctm_synchrony_norm"] == pytest.approx(0.0, abs=1e-7)
    assert 0.0 <= tel_init["ctm_synchrony_coherence"] <= 1.01

    with torch.no_grad():
        ctm_model.thinking_core.ctm_out.weight.normal_(0.0, 0.25)
        y_active = ctm_model(x).logits
    assert not torch.allclose(y_base, y_active, atol=1e-5)
    assert ctm_model.telemetry()["thinking"]["ctm_synchrony_norm"] > 0.0


def test_muon_hybrid_optimizer_orthogonality_and_convergence() -> None:
    import mimomix_core as mc
    import muon_optimizer as mo

    torch.manual_seed(102)
    g = torch.randn(24, 24)
    ortho = mo.newton_schulz_orthogonalize(g, steps=5)
    gram = ortho @ ortho.t()
    off_diag = (gram - torch.diag(gram.diagonal())).abs().mean()
    assert float(off_diag) < 0.15
    assert 0.6 < float(gram.diagonal().mean()) < 1.4

    cfg = mc.MiMoMixConfig(
        vocab_size=32,
        hidden_size=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=32,
        use_moe=False,
        n_mtp_layers=0,
        use_thinking_core=False,
    )
    model = mc.MiMoMixModel(cfg)
    groups, counts = mo.build_muon_hybrid_param_groups(model, lr=2e-3, muon_lr_scale=4.0)
    assert counts["muon_tensors"] > 0
    assert counts["adam_decay_tensors"] + counts["adam_no_decay_tensors"] > 0
    opt = mo.MuonHybridOptimizer(groups, lr=2e-3)

    x = torch.randint(0, 32, (2, 10))
    model.train()
    losses = []
    for _ in range(8):
        opt.zero_grad()
        out = model(x, labels=x)
        loss = out.loss
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert opt.last_ortho_updates == counts["muon_tensors"]


def test_v102_extended_zero_eval_symbolic_solvers_and_refusals() -> None:
    from scf.verification import SCFVerificationGate

    gate = SCFVerificationGate()
    cases = [
        ("Solve x + y + z = 6, 2x - y + z = 3, x + 2y - z = 2", "x = 1, y = 2, z = 3"),
        ("Find the determinant of [[1, 2], [3, 4]]", "-2"),
        ("Find the determinant of [[2, 1, 0], [1, 3, 1], [0, 2, 4]]", "16"),
        ("Find the trace of [[2, 1, 0], [1, 3, 1], [0, 2, 4]]", "9"),
        ("What is 7^13 mod 11?", "2"),
        ("Find the modular inverse of 3 mod 11", "4"),
        ("Prime factorization of 360", "2^3 * 3^2 * 5"),
        ("Find Euler totient of 36", "12"),
        ("What is the binomial probability of k=2 successes in n=5 trials with p=0.4?", "0.3456"),
        ("Bayes: prior 0.01, sensitivity 0.95, false positive rate 0.05", "0.161017"),
        ("Parallel resistors 6 ohm and 3 ohm", "2"),
        ("Thin lens with f = 10 cm and u = 30 cm", "15"),
    ]
    for query, expected in cases:
        res = gate.verify(query=query)
        assert res["verified"] is True, (query, res)
        assert res["verified_answer"] == expected, (query, res)

    # Strict refusal guards remain fail-closed
    for blocked in (
        "Do not compute the determinant of [[1, 2], [3, 4]]",
        "Verify proof: 7^13 mod 11",
        "Check proof: Bayes prior 0.01, sensitivity 0.95, false positive rate 0.05",
    ):
        assert gate.verify(query=blocked)["verified"] is False


def test_v102_frontier_ecosystem_engines_and_receipts() -> None:
    import mimomix_core as mc
    import nexus_v102_frontier as v102

    # 1. NexusSearchHybridRRF + Titans surprise + 7-class taxonomy
    rrf = v102.NexusSearchHybridRRF()
    docs = [
        {"id": "d1", "text": "Remember I prefer Python and concise structured steps."},
        {"id": "d2", "text": "Male CNS connectome neuron count for Li20 is 40."},
        {"id": "d3", "text": "Run ablation benchmark seed 82 with spectral criticality."},
    ]
    res_rrf = rrf.rank_documents("Which Python style preference did I set?", docs, top_k=3)
    assert res_rrf["hits"][0]["id"] == "d1"
    assert res_rrf["hits"][0]["memory_class"] == "procedural"
    assert res_rrf["hits"][0]["claims_can_authorize_tools"] is False
    assert 0.0 <= res_rrf["query_surprise"] <= 1.0
    assert res_rrf["answer_authority"] is False

    # 2. TesseraRatifiedLemmaCommons: independent 2-lineage ratification + epoch expiry
    commons = v102.TesseraRatifiedLemmaCommons(min_independent_lineages=2, expiry_epochs=1)
    s1 = commons.submit_discovery(expression="a^2 - b^2 = (a-b)(a+b)", lineage="NexusMind", task_family="algebra")
    assert s1["ratified"] is False
    s2 = commons.submit_discovery(expression="a^2 - b^2 = (a-b)(a+b)", lineage="MiMoMix", task_family="quadratics")
    assert s2["ratified"] is True and s2["newly_ratified"] is True
    assert len(commons.active_lemmas()) == 1
    exp1 = commons.advance_epoch(1)
    assert exp1["active_ratified_count"] == 1
    exp2 = commons.advance_epoch(1)
    assert len(exp2["expired_now"]) == 1
    assert exp2["active_ratified_count"] == 0

    # 3. CounterformFalsificationAuditor: CVaR preference + fail-closed ABSTAIN
    auditor = v102.CounterformFalsificationAuditor()
    rec = auditor.audit_candidates(
        candidate_losses={"steady": [0.12, 0.14, 0.16, 0.18], "tail_trap": [0.05, 0.05, 0.05, 0.92]},
        consistency_residuals={"steady": 0.05, "tail_trap": 0.05},
        projection_residuals={"steady": 0.05, "tail_trap": 0.05},
    )
    assert rec["decision"] == "RECOMMEND"
    assert rec["selected_candidate"] == "steady"
    assert rec["answer_authority"] is False

    abstain = auditor.audit_candidates(
        candidate_losses={"bad": [0.7, 0.8, 0.9, 0.95]},
        consistency_residuals={"bad": 0.4},
        projection_residuals={"bad": 0.5},
    )
    assert abstain["decision"] == "ABSTAIN_UNSUPPORTED"
    assert abstain["selected_candidate"] is None

    # 4. CarrierInteractionAtlas: exact 2^4 = 16 Mobius reconstruction + Ghost R^2 + Aether FLOPs
    carriers = ["trunk", "cns", "thinking", "memory"]
    outcomes = {
        "": 0.95,
        "trunk": 0.45,
        "cns": 0.88,
        "thinking": 0.90,
        "memory": 0.91,
        "cns,trunk": 0.32,
        "thinking,trunk": 0.38,
        "memory,trunk": 0.40,
        "cns,thinking": 0.80,
        "cns,memory": 0.83,
        "memory,thinking": 0.85,
        "cns,thinking,trunk": 0.25,
        "cns,memory,trunk": 0.28,
        "memory,thinking,trunk": 0.34,
        "cns,memory,thinking": 0.74,
        "cns,memory,thinking,trunk": 0.15,
    }
    atlas = v102.CarrierInteractionAtlas.decompose_subset_factorial(carriers, outcomes)
    assert atlas["subset_count"] == 16
    assert atlas["exact_reconstruction_verified"] is True
    assert atlas["max_reconstruction_error"] < 1e-9
    assert len(atlas["interactions_by_order"]["order_1"]) == 4
    assert len(atlas["interactions_by_order"]["order_2"]) == 6
    assert len(atlas["interactions_by_order"]["order_3"]) == 4
    assert len(atlas["interactions_by_order"]["order_4"]) == 1

    # GhostInTheMachine v7 state-predicted error R^2
    states = [[float(i), float(i % 3), 0.5 * float(i)] for i in range(1, 25)]
    authored_err = [0.2 * s[0] - 0.5 * s[1] + 0.01 for s in states]
    exog_err = [((-1) ** i) * (0.3 if i % 5 == 0 else 0.02) for i in range(1, 25)]
    auth_diag = v102.CarrierInteractionAtlas.authorship_error_predictability(states, authored_err)
    exog_diag = v102.CarrierInteractionAtlas.authorship_error_predictability(states, exog_err)
    assert auth_diag["state_predicted_error_r2"] > 0.95
    assert auth_diag["stabilization_regime"] == "self_authored_stabilized"
    assert exog_diag["state_predicted_error_r2"] < auth_diag["state_predicted_error_r2"]

    # Supermix-Aether analytical FLOPs + KV cache compression
    cfg = mc.MiMoMixConfig(use_moe=True, use_mhc_residual=True, use_cns_core=True, use_thinking_core=True)
    flops = v102.CarrierInteractionAtlas.profile_mimomix_compute(cfg, seq_len=128)
    assert flops["active_flops_per_token"] < flops["dense_equivalent_flops_per_token"]
    assert flops["kv_cache_compression_ratio"] >= 1.0

    # 5. SonapseDiscoveryMonitor Mode A / Mode B / Pareto
    mon = v102.SonapseDiscoveryMonitor()
    rep = mon.evaluate_step(train_loss=0.05, dev_loss=0.08, spectral_radius=1.35, target_spectral_radius=0.90)
    assert rep["mode"] == "MODE_A_REPAIR"
    disc = mon.evaluate_step(
        train_loss=0.02, dev_loss=0.09, prev_dev_loss=0.08, active_module_fraction=0.85, spare_slots_free=4
    )
    assert disc["mode"] == "MODE_B_DISCOVERY"
    assert mon.pareto_accept(baseline_loss=0.10, candidate_loss=0.09, param_delta_fraction=0.03, spectral_radius_after=0.92)["accepted"] is True
    assert mon.pareto_accept(baseline_loss=0.10, candidate_loss=0.09, param_delta_fraction=0.25, spectral_radius_after=0.92)["accepted"] is False

    # 6. HyperionComputeScheduler
    sched = v102.HyperionComputeScheduler().schedule(
        domain="math_science", submode="problem_solving", complexity="high", has_memory_turns=True
    )
    assert sched["wave_count"] == 3
    assert sched["parallelism_speedup_est"] >= 1.0
    assert sched["answer_authority"] is False


def test_scf_runtime_rrf_memory_process_audit_cli_api_and_chat_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    from scf import CognitiveFabricRuntime, SCFFeatureFlags
    from scf.__main__ import main as scf_main
    from scf.collective import SCFCollectiveOrchestrator
    import supermix_chat_server as scs

    # 1. Process-fidelity trace audit catches percentage & power mistakes too
    orch = SCFCollectiveOrchestrator()
    coll = orch.gather_and_synthesize(
        "Compute 20% of 250 then add 2^4",
        specialist_families=("MiMoMix", "CognitiveLeap"),
        candidate_generators={
            "MiMoMix": lambda _q: {
                "family": "MiMoMix",
                "answer": "20% of 250 = 60, 2^4 = 12, total 72",
                "confidence": 0.70,
                "verified": False,
            },
            "CognitiveLeap": lambda _q: {
                "family": "CognitiveLeap",
                "answer": "20% of 250 = 50, 2^4 = 16, total 66",
                "confidence": 0.70,
                "verified": False,
            },
        },
        max_candidates=2,
    )
    by_fam = {c["family"]: c for c in coll["candidates"]}
    assert by_fam["MiMoMix"]["trace_audit_failures"] == 2
    assert by_fam["MiMoMix"]["process_fidelity_score"] == 0.0
    assert by_fam["CognitiveLeap"]["trace_audit_failures"] == 0
    assert by_fam["CognitiveLeap"]["process_fidelity_score"] == 1.0
    assert coll["best_candidate"]["family"] == "CognitiveLeap"

    # 2. SCF Runtime surfaces Hyperion schedule + RRF memory while keeping firewall intact
    rt = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()
    out = rt.run_task(
        "What is 7^13 mod 11?",
        flags_override=flags,
        recent_turns=[("Remember I prefer concise Python explanations.", "Got it.")],
    )
    assert out["verified"] is True
    assert out["final_answer"] == "2"
    assert out["routing"]["hyperion_schedule"]["wave_count"] == 3
    assert len(out["memory"]["working_state"]["rrf_memory_hits"]) == 1
    assert out["memory"]["memory_firewall_enforced"] is True

    # 3. Unified CLI `python -m scf frontier --json`
    buf = StringIO()
    with redirect_stdout(buf):
        rc = scf_main(["frontier", "--json"])
    assert rc == 0
    cli_data = json.loads(buf.getvalue())
    assert cli_data["frontier_version"] == "102.0.0-frontier"
    assert cli_data["mhc"]["enabled"] is True
    assert cli_data["carrier_atlas_exact"] is True
    assert cli_data["tessera_ratified"] is True

    # 4. Nexus API v102 endpoints
    from starlette.testclient import TestClient
    import nexus_api

    api_client = TestClient(nexus_api.create_app())
    sig_resp = api_client.get("/v1/signals")
    assert sig_resp.status_code == 200
    assert sig_resp.json()["v102_frontier_synthesis"]["mhc_birkhoff_residual"] is True

    cvar_resp = api_client.post(
        "/v1/frontier/cvar_audit",
        json={"candidate_losses": {"a": [0.1, 0.2, 0.2, 0.2], "b": [0.05, 0.05, 0.05, 0.9]}},
    )
    assert cvar_resp.status_code == 200
    assert cvar_resp.json()["selected_candidate"] == "a"

    atlas_resp = api_client.post(
        "/v1/frontier/carrier_atlas",
        json={"carriers": ["a", "b"], "subset_outcomes": {"": 1.0, "a": 0.6, "b": 0.7, "a,b": 0.2}},
    )
    assert atlas_resp.status_code == 200
    assert atlas_resp.json()["exact_reconstruction_verified"] is True

    tess_resp = api_client.post(
        "/v1/frontier/tessera_commons",
        json={
            "submissions": [
                {"expression": "sin^2 + cos^2 = 1", "lineage": "L1", "task_family": "trig"},
                {"expression": "sin^2 + cos^2 = 1", "lineage": "L2", "task_family": "geom"},
            ]
        },
    )
    assert tess_resp.status_code == 200
    assert len(tess_resp.json()["snapshot"]["active_ratified"]) == 1

    hyp_resp = api_client.post(
        "/v1/frontier/hyperion_schedule",
        json={"domain": "code", "submode": "problem_solving", "complexity": "high", "has_memory_turns": True},
    )
    assert hyp_resp.status_code == 200
    assert hyp_resp.json()["wave_count"] == 3

    # 5. Chat server v102 telemetry badges & valid inline JS
    class _StubV102:
        def telemetry(self):
            return {
                "cns_core": {
                    "alive_nodes": 810,
                    "edges_installed": 19980,
                    "dendritic_apical": True,
                    "apical_mod_mean": 1.04,
                    "spectral_radius_est": 0.90,
                },
                "mhc": {"enabled": True, "streams": 2, "max_doubly_stochastic_error": 1e-6},
                "thinking": {
                    "cycles_used": 2,
                    "exit_reason": "entropy_stability",
                    "ctm_synchrony": True,
                    "ctm_synchrony_coherence": 0.612,
                },
            }

    loaded = scs.LoadedModel(
        name="v102_frontier",
        checkpoint="stub_v102.pt",
        model=_StubV102(),
        tokenizer=object(),
        extra={"run_name": "v102_frontier"},
        lock=threading.Lock(),
    )
    registry = scs.ModelRegistry({"v102_frontier": "stub_v102.pt"}, max_resident=1)
    monkeypatch.setattr(registry, "acquire", lambda _name: loaded)
    monkeypatch.setattr(scs, "stream_speculative_tokens", lambda *args, **kwargs: iter(["2"]))

    chat_app = scs.build_app(registry, max_concurrency=2, history_turns=0)
    chat_client = chat_app.test_client()
    html = chat_client.get("/").get_data(as_text=True)
    assert "[v102 mHC]" in html
    assert "CTM-Sync=" in html
    assert "Apical=" in html

    m_js = re.search(r"<script>(.*)</script>", html, flags=re.DOTALL)
    assert m_js is not None
    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
            tf.write(m_js.group(1))
            js_path = Path(tf.name)
        try:
            res = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True)
            assert res.returncode == 0, res.stderr
        finally:
            js_path.unlink(missing_ok=True)
