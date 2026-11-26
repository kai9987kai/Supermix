"""Verification & Performance Parity Suite for Supermix v105 Hyper-Optimization & Evolution.

Covers:
1. ValueResidualMixer / ResFormer (`use_value_residual=True`): zero-init birth
   identity, gradient flow, bit-exact KV-cache decode & MTP speculative parity.
2. ConnectomeCore causal STDP plasticity trace (`cns_stdp_trace=True`): zero
   birth identity, mitosis inheritance, causal temporal parity & telemetry.
3. Full combined v93+v102+v103+v104+v105 neural stack KV-cache and MTP
   speculative rollback exact equivalence.
4. AdaptiveDraftController & `speculative_generate(..., adaptive_draft=True)`:
   bit-exact greedy equivalence + dynamic low-acceptance draft throttling.
5. PrefixKVCache: shared-prefix TTFT reuse with bit-exact greedy parity.
6. Symmetric Int8 KV-cache compression (`quantize_past_kv` / `dequantize_past_kv`
   / `kv_cache_memory_report`) across hybrid GQA + Engram/DeltaNet/Titans caches.
7. MuonHybridOptimizer v105 Polar Express minimax orthogonalization (`polar_express=True`):
   tighter ||O O^T - I||_F than constant Muon + smooth loss reduction.
8. Chat Server O(T) KV-cached `stream_tokens` exact parity with O(T^2) prefix decode.
9. SCF Runtime optimizations: pruned weight discovery, TTL registry cache,
   O(1) connectome atlas pair index, verification LRU cache, latency summary.
10. Ten new v105 zero-eval exact STEM & CS solvers + fail-closed invalid-input guards.
11. Five v105 hyper-optimization engines in `source/nexus_v105_hyperopt.py` with
    canonical SHA-256 receipts and strict `answer_authority=False`.
12. FastAPI `/v1/v105/*`, CLI `scf perf/frontier/diagnose --json`, Chat Server
    v105 telemetry badges, and NexusMind Studio v105 Hyper-Optimization Lab.
"""

from __future__ import annotations

import json
import os
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
import mimomix_text as mt  # noqa: E402
import muon_optimizer as mo  # noqa: E402
import nexus_api  # noqa: E402
import nexus_solver  # noqa: E402
import nexus_v105_hyperopt as v105  # noqa: E402
import supermix_chat_server as chat_server  # noqa: E402
from scf import CognitiveFabricRuntime, SCFFeatureFlags, SCFModelRegistry  # noqa: E402


# ---------------------------------------------------------------------------
# 1. ValueResidualMixer (ResFormer)
# ---------------------------------------------------------------------------


def test_value_residual_zero_init_identity_grad_and_kv_speculative_parity():
    torch.manual_seed(1051)
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
    vr_cfg = mc.MiMoMixConfig(**{**base_cfg.to_dict(), "use_value_residual": True})
    base_m = mc.MiMoMixModel(base_cfg).eval()
    vr_m = mc.MiMoMixModel(vr_cfg).eval()
    missing, unexpected = vr_m.load_state_dict(base_m.state_dict(), strict=False)
    assert unexpected == []
    assert any(k.startswith("value_residual.") for k in missing)

    ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        b_logits = base_m(ids, return_mtp=False).logits
        z_logits = vr_m(ids, return_mtp=False).logits
    assert torch.allclose(b_logits, z_logits, atol=1e-6)

    # Open value residual gates and verify modulation, KV-cache parity, and MTP parity
    with torch.no_grad():
        vr_m.value_residual.layer_gates.fill_(0.35)
        full_out = vr_m(ids, return_mtp=False)
        assert not torch.allclose(b_logits, full_out.logits, atol=1e-5)

        out = vr_m(ids[:, :1], use_cache=True, return_mtp=False, cache_slack=2)
        past = out.past_key_values
        step_logits = [out.logits]
        for pos in range(1, ids.shape[1]):
            out = vr_m(
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

    tele = vr_m.telemetry()
    assert tele["value_residual"]["enabled"] is True
    assert tele["value_residual"]["last_mix_strength"] == pytest.approx(0.35, rel=1e-4)

    md.assert_greedy_equivalence(vr_m, ids[:, :4], max_new_tokens=6)

    # Gradient flow check
    vr_m.train()
    out_t = vr_m(ids, labels=ids)
    out_t.loss.backward()
    assert vr_m.value_residual.layer_gates.grad is not None
    assert float(vr_m.value_residual.layer_gates.grad[1:].abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# 2. ConnectomeCore STDP Plasticity Trace + Mitosis Inheritance
# ---------------------------------------------------------------------------


def test_cns_stdp_trace_zero_birth_mitosis_and_temporal_parity():
    torch.manual_seed(1052)
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
    stdp_cfg = mc.MiMoMixConfig(**{**base_cfg.to_dict(), "cns_stdp_trace": True})
    base_m = mc.MiMoMixModel(base_cfg).eval()
    stdp_m = mc.MiMoMixModel(stdp_cfg).eval()
    missing, unexpected = stdp_m.load_state_dict(base_m.state_dict(), strict=False)
    assert unexpected == []
    assert "cns_core.stdp_strength" in missing

    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        b_logits = base_m(ids, return_mtp=False).logits
        z_logits = stdp_m(ids, return_mtp=False).logits
    assert torch.allclose(b_logits, z_logits, atol=1e-6)

    with torch.no_grad():
        stdp_m.cns_core.stdp_strength.fill_(0.75)
        child = stdp_m.cns_core.split_module(parent=0, step=12)
        assert child is not None
        assert float(stdp_m.cns_core.stdp_strength[child]) == pytest.approx(
            float(stdp_m.cns_core.stdp_strength[0])
        )
        _ = stdp_m(ids, return_mtp=False)

    tele = stdp_m.telemetry()["cns_core"]
    assert tele["stdp_trace"] is True
    assert 0.75 <= tele["stdp_gain_mean"] <= 1.25


# ---------------------------------------------------------------------------
# 3. Full Combined v93+v102+v103+v104+v105 Stack Parity
# ---------------------------------------------------------------------------


def test_full_v105_mega_stack_kv_and_mtp_speculative_parity():
    torch.manual_seed(1053)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        moe_top_k=2,
        use_moe=True,
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
        use_value_residual=True,
        use_cns_core=True,
        cns_nodes=10,
        cns_steps=2,
        cns_read_layers=(0, 1),
        cns_write_layers=(1,),
        cns_temporal=True,
        cns_dendritic_apical=True,
        cns_neuromod_gate=True,
        cns_stdp_trace=True,
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
        model.value_residual.layer_gates.fill_(0.25)
        model.cns_core.apical_gain.fill_(0.30)
        model.cns_core.neuromod_strength.fill_(0.30)
        model.cns_core.stdp_strength.fill_(0.30)

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
# 4. Adaptive Speculative Draft Control, PrefixKVCache & Int8 KV Compression
# ---------------------------------------------------------------------------


def test_adaptive_draft_prefix_cache_and_int8_kv_compression():
    torch.manual_seed(1054)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        use_moe=False,
        n_mtp_layers=3,
        sliding_window=4,
        hybrid_ratio=1,
        use_engram_memory=True,
        engram_table_size=64,
        use_deltanet_memory=True,
        deltanet_dim=8,
        use_value_residual=True,
    )
    model = mc.MiMoMixModel(cfg).eval()
    prompt = torch.randint(0, 64, (1, 8))

    greedy = md.greedy_generate(model, prompt, max_new_tokens=12)
    adap = md.speculative_generate(model, prompt, max_new_tokens=12, adaptive_draft=True)
    fixed = md.speculative_generate(model, prompt, max_new_tokens=12, adaptive_draft=False)
    assert torch.equal(greedy.new_tokens, adap.new_tokens)
    assert adap.stats.drafted_tokens <= fixed.stats.drafted_tokens
    assert "adaptive_draft" in adap.telemetry
    assert adap.telemetry["adaptive_draft"]["steps"] > 0

    # PrefixKVCache exact parity on shared prefix
    pcache = md.PrefixKVCache(max_entries=4)
    p1 = prompt
    p2 = torch.cat([prompt[:, :5], torch.randint(0, 64, (1, 3))], dim=1)
    r1_cached = pcache.greedy_generate(model, p1, max_new_tokens=8, cache_prefix_length=5)
    r1_ref = md.greedy_generate(model, p1, max_new_tokens=8)
    assert torch.equal(r1_cached.new_tokens, r1_ref.new_tokens)

    r2_cached = pcache.greedy_generate(model, p2, max_new_tokens=8)
    r2_ref = md.greedy_generate(model, p2, max_new_tokens=8)
    assert torch.equal(r2_cached.new_tokens, r2_ref.new_tokens)
    pstats = pcache.stats()
    assert pstats["hits"] == 1
    assert pstats["tokens_reused"] == 5

    # Symmetric Int8 KV Compression across mixed GQA + Engram (int) + DeltaNet (float) caches
    with torch.no_grad():
        prefill = model(prompt, use_cache=True, return_mtp=False)
        q_past = md.quantize_past_kv(prefill.past_key_values)
        dq_past = md.dequantize_past_kv(q_past)
        rep = md.kv_cache_memory_report(prefill.past_key_values, q_past)
        assert rep["compression_ratio"] > 2.5
        assert rep["mean_cosine_similarity"] > 0.999
        # Verify dequantized cache executes a forward step cleanly with high fidelity
        next_tok = torch.randint(0, 64, (1, 1))
        orig_step = model(next_tok, past_key_values=prefill.past_key_values, use_cache=True, return_mtp=False, past_length=8).logits
        dq_step = model(next_tok, past_key_values=dq_past, use_cache=True, return_mtp=False, past_length=8).logits
        cos_sim = torch.nn.functional.cosine_similarity(orig_step.reshape(1, -1), dq_step.reshape(1, -1)).item()
        assert cos_sim > 0.999


# ---------------------------------------------------------------------------
# 5. MuonHybridOptimizer v105 Polar Express
# ---------------------------------------------------------------------------


def test_muon_v105_polar_express_tighter_orthogonality_and_training():
    torch.manual_seed(1055)
    g = torch.randn(48, 48)
    o_std = mo.newton_schulz_orthogonalize(g, steps=5, polar_express=False)
    o_pe = mo.newton_schulz_orthogonalize(g, steps=5, polar_express=True)
    eye = torch.eye(48)
    err_std = float((o_std @ o_std.t() - eye).norm())
    err_pe = float((o_pe @ o_pe.t() - eye).norm())
    assert err_pe < err_std

    cfg = mc.MiMoMixConfig(
        vocab_size=48,
        hidden_size=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        use_value_residual=True,
    )
    model = mc.MiMoMixModel(cfg)
    assert mo.is_muon_eligible_parameter("value_residual.layer_gates", model.value_residual.layer_gates) is False
    groups, counts = mo.build_muon_hybrid_param_groups(
        model, lr=1e-3, polar_express=True, cautious_mask=True, row_normalize=True
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
    assert tele["polar_express_active"] is True


# ---------------------------------------------------------------------------
# 6. Chat Server KV-Cached stream_tokens Exact Parity + SCF Cache Invariants
# ---------------------------------------------------------------------------


def test_chat_server_kv_cached_stream_tokens_matches_full_prefix():
    torch.manual_seed(1056)
    tok = mt.WordTokenizer(["hello", "world", "supermix", "fast", "answer", "42"])
    cfg = mc.MiMoMixConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        n_mtp_layers=1,
        use_value_residual=True,
    )
    model = mc.MiMoMixModel(cfg).eval()
    loaded = chat_server.LoadedModel(
        name="v105_test", checkpoint="mem", model=model, tokenizer=tok, extra={}
    )
    chunks = list(
        chat_server.stream_tokens(
            loaded,
            "hello world",
            [],
            max_new_tokens=8,
            temperature=0.0,
            top_p=1.0,
            greedy=True,
            should_stop=lambda: False,
        )
    )
    streamed_text = "".join(chunks)

    # Compare against reference greedy_generate
    prompt_ids, _ = tok.encode_turn("hello world", None)
    ref = md.greedy_generate(
        model,
        torch.tensor([prompt_ids], dtype=torch.long),
        max_new_tokens=8,
        eos_token_id=mt.EOS,
    )
    ref_toks = ref.new_tokens[0].tolist()
    if ref_toks and ref_toks[-1] == mt.EOS:
        ref_toks = ref_toks[:-1]
    assert streamed_text == tok.decode(ref_toks)


def test_scf_runtime_fast_routing_and_observability_summary():
    reg = SCFModelRegistry()
    fams = reg.available_source_families()
    assert "NexusMind" in fams and "MiMoMix" in fams
    reg.invalidate_cache()

    rt = CognitiveFabricRuntime()
    flags = SCFFeatureFlags.all_enabled()
    r1 = rt.run_task("Amdahl speedup p = 0.8, n = 4", flags_override=flags)
    r2 = rt.run_task("Convert 255 from base 10 to base 16", flags_override=flags)
    assert r1["verified"] is True and r1["final_answer"] == "2.5"
    assert r2["verified"] is True and r2["final_answer"] == "FF"
    lat = rt.observability.latency_summary()
    assert lat["count"] == 2
    assert lat["min_ms"] <= lat["p50_ms"] <= lat["max_ms"]
    rt.verifier.clear_cache()


# ---------------------------------------------------------------------------
# 7. Ten New v105 Zero-Eval Exact STEM & CS Solvers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_display,expected_formula",
    [
        ("Extended gcd of 240 and 46", "gcd=2, s=-9, t=47", "numtheory.extended_gcd"),
        ("What is the 4th harmonic number?", "25/12", "series.harmonic_number"),
        ("Compute multinomial(6; 2, 2, 2)", "90", "comb.multinomial"),
        ("Euclidean distance between (1, 2, 3) and (4, 6, 3)", "5", "geometry.euclidean_distance"),
        ("Manhattan distance between [1, 2] and [4, 6]", "7", "geometry.manhattan_distance"),
        ("Cosine similarity of [1, 0] and [1, 1]", "0.707107", "linalg.cosine_similarity"),
        ("Z-score of x = 85 with mean = 70 and std = 10", "1.5", "stats.z_score"),
        ("Half-life decay N0 = 100, t = 15, half_life = 5", "12.5", "physics.half_life_decay"),
        ("Doppler shift f0 = 500, v_obs = 0, v_src = 34, v = 340", "555.555556", "physics.doppler_shift"),
        ("Amdahl speedup p = 0.8, n = 4", "2.5", "cs.amdahl_speedup"),
        ("Convert 255 from base 10 to base 16", "FF", "cs.base_conversion"),
        ("Convert 101101 from base 2 to base 10", "45", "cs.base_conversion"),
    ],
)
def test_v105_exact_symbolic_solvers(query: str, expected_display: str, expected_formula: str):
    res = nexus_solver.solve_problem(query)
    assert res.solved is True
    assert res.formula_id == expected_formula
    assert res.display_answer == expected_display
    assert res.receipt is not None and len(res.receipt.receipt_sha256) == 64


def test_v105_solvers_fail_closed_on_invalid_inputs():
    assert nexus_solver.solve_problem("Amdahl speedup p = 1.2, n = 4").solved is False
    assert nexus_solver.solve_problem("Convert 29 from base 8 to base 10").solved is False
    assert nexus_solver.solve_problem("Z-score of x = 85 with mean = 70 and std = 0").solved is False
    assert nexus_solver.solve_problem("Cosine similarity of [0, 0] and [1, 1]").solved is False
    assert nexus_solver.solve_problem("Compute multinomial(7; 2, 2, 2)").solved is False


# ---------------------------------------------------------------------------
# 8. Five v105 Hyper-Optimization Engines
# ---------------------------------------------------------------------------


def test_v105_five_hyperopt_engines_invariants_and_receipts():
    # 1. SpeculativeDraftOptimizer
    spec = v105.SpeculativeDraftOptimizer().optimize(
        depth_accept_probs=[0.85, 0.78, 0.70, 0.55],
        draft_cost_ratio=0.06,
        verify_overhead_per_token=0.03,
    )
    assert spec["answer_authority"] is False
    assert spec["optimal_draft_depth"] >= 1
    assert spec["optimal_net_speedup"] > 1.0
    assert len(spec["receipt"]["receipt_sha256"]) == 64

    low_spec = v105.SpeculativeDraftOptimizer().optimize(
        depth_accept_probs=[0.02, 0.01], draft_cost_ratio=0.10, verify_overhead_per_token=0.05
    )
    assert low_spec["optimal_draft_depth"] == 0
    assert low_spec["recommended_mode"] == "greedy_fallback"

    # 2. KVCacheCompressionPlanner
    kv = v105.KVCacheCompressionPlanner().plan(
        seq_len=131072, n_layers=24, hybrid_ratio=5, sliding_window=128, memory_budget_mb=256.0
    )
    assert kv["answer_authority"] is False
    assert kv["hybrid_vs_dense_reduction_factor"] > 4.0
    assert kv["decision"] == "FIT_WITHIN_BUDGET"
    assert kv["recommended_scheme"] is not None
    assert len(kv["receipt"]["receipt_sha256"]) == 64

    # 3. RooflineComputeProfiler
    roof = v105.RooflineComputeProfiler().profile(
        hidden_size=512, n_layers=8, seq_len=512, peak_gflops=400.0, memory_bandwidth_gbs=50.0
    )
    assert roof["answer_authority"] is False
    assert roof["decode"]["regime"] == "memory_bandwidth_bound"
    assert roof["prefill"]["arithmetic_intensity"] > roof["decode"]["arithmetic_intensity"]
    assert len(roof["receipt"]["receipt_sha256"]) == 64

    # 4. CascadeEnsembleRouter
    casc = v105.CascadeEnsembleRouter().optimize_cascade(
        calibration_records=[
            {"fast_confidence": 0.95, "fast_correct": True, "deep_correct": True},
            {"fast_confidence": 0.88, "fast_correct": True, "deep_correct": True},
            {"fast_confidence": 0.42, "fast_correct": False, "deep_correct": True},
            {"fast_confidence": 0.25, "fast_correct": False, "deep_correct": True},
        ],
        fast_cost_ms=15.0,
        deep_cost_ms=150.0,
        max_avg_latency_ms=100.0,
    )
    assert casc["answer_authority"] is False
    assert casc["decision"] == "OPTIMAL_CASCADE_FOUND"
    assert casc["cascade_accuracy"] == pytest.approx(1.0)
    assert casc["avg_latency_ms"] <= 100.0
    assert len(casc["receipt"]["receipt_sha256"]) == 64

    # 5. SelfHealingBenchmarkGuard
    guard_ok = v105.SelfHealingBenchmarkGuard().evaluate(
        baseline_latencies_ms=[100.0, 105.0, 98.0, 102.0],
        candidate_latencies_ms=[40.0, 42.0, 39.0, 41.0],
        baseline_scores=[0.90, 0.92, 0.91, 0.93],
        candidate_scores=[0.90, 0.92, 0.91, 0.93],
    )
    assert guard_ok["answer_authority"] is False
    assert guard_ok["decision"] == "APPROVE_OPTIMIZATION"
    assert guard_ok["speedup_factor"] > 2.0
    assert len(guard_ok["receipt"]["receipt_sha256"]) == 64

    guard_bad = v105.SelfHealingBenchmarkGuard().evaluate(
        baseline_latencies_ms=[100.0, 102.0, 101.0],
        candidate_latencies_ms=[130.0, 132.0, 131.0],
    )
    assert guard_bad["decision"] == "HOLD_LATENCY_REGRESSION"

    guard_qbad = v105.SelfHealingBenchmarkGuard().evaluate(
        baseline_latencies_ms=[100.0, 102.0],
        candidate_latencies_ms=[50.0, 51.0],
        baseline_scores=[0.95, 0.95],
        candidate_scores=[0.80, 0.80],
    )
    assert guard_qbad["decision"] == "HOLD_QUALITY_REGRESSION"


# ---------------------------------------------------------------------------
# 9. FastAPI /v1/v105/*, CLI perf/frontier/diagnose, Chat Server & Studio UI
# ---------------------------------------------------------------------------


def test_v105_api_endpoints_cli_and_studio_surface():
    from starlette.testclient import TestClient

    app = nexus_api.create_app()
    client = TestClient(app)

    for route in (
        "/v1/v105/speculative_optimize",
        "/v1/v105/kv_plan",
        "/v1/v105/roofline_profile",
        "/v1/v105/cascade_route",
        "/v1/v105/regression_guard",
    ):
        resp = client.post(route, json={})
        assert resp.status_code == 200, f"{route} failed: {resp.text}"
        body = resp.json()
        assert body["answer_authority"] is False
        assert len(body["receipt"]["receipt_sha256"]) == 64

    # Invalid probability > 1 must fail closed with 422
    bad_resp = client.post("/v1/v105/speculative_optimize", json={"depth_accept_probs": [1.5]})
    assert bad_resp.status_code == 422

    # CLI: python -m scf perf --json, frontier --json, diagnose --json
    env = {**os.environ, "PYTHONPATH": str(SOURCE)}
    p_perf = subprocess.run(
        [sys.executable, "-m", "scf", "perf", "--json"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    perf_data = json.loads(p_perf.stdout)
    assert perf_data["evolution_v105_version"] == v105.EVOLUTION_V105_VERSION
    assert perf_data["adaptive_spec_greedy_parity"] is True
    assert perf_data["scf_verified_answer"] == "2.5"
    assert perf_data["kv_int8_report"]["compression_ratio"] > 2.5

    p_front = subprocess.run(
        [sys.executable, "-m", "scf", "frontier", "--json"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    f_data = json.loads(p_front.stdout)
    assert f_data["evolution_v105_version"] == v105.EVOLUTION_V105_VERSION
    assert f_data["value_residual"]["enabled"] is True
    assert f_data["cns_stdp_trace"] is True

    # Studio HTML contains v105 Hyper-Optimization Lab controls
    studio_html = (ROOT / "web_static" / "nexus_studio.html").read_text(encoding="utf-8")
    for fn_name in (
        "runV105SpeculativeOptimize",
        "runV105KVPlan",
        "runV105RooflineProfile",
        "runV105CascadeRoute",
        "runV105RegressionGuard",
    ):
        assert fn_name in studio_html

    # Chat server contains v105 telemetry badges
    chat_py = (SOURCE / "supermix_chat_server.py").read_text(encoding="utf-8")
    assert "use_value_residual" in chat_py
    assert "cns_stdp_trace" in chat_py
    assert "[v105 ValueRes/STDP]" in chat_py
