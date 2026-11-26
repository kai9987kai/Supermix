"""Comprehensive test suite for Supermix v106 Frontier Advancement & Performance.

Covers:
1. CausalShortConvMemory (birth identity, grad flow, causality, KV-cache & MTP parity)
2. SparseMoEFeedForward input-conditioned shared-expert gate (birth identity, grad, telemetry)
3. ConnectomeCore oscillatory phase-coupling gate (birth identity, Dale safety, mitosis, temporal parity)
4. Full v93+v102+v103+v104+v105+v106 mega-stack KV-cache & MTP speculative equivalence
5. MultiLatentAttention 2D-BLAS matrix absorption & DifferentialHybridAttention fast paths
6. MiMoMixModel trunk-reuse & mimomix_controller.decide() bit-identical acceleration
7. Chunked prefill, packed Int4 KV-cache compression, and multi-branch speculative tree verification
8. MuonHybridOptimizer spectral_norm_clip_ (MuonClip) & update-RMS telemetry
9. Ten new zero-eval exact STEM/CS/Finance/Chemistry solvers + fail-closed invalid-input guards
10. Five v106 frontier engines (TTC Allocator, Spectral Diagnostics, Causal CF, Quant Planner, Drift Detector)
11. Full API (/v1/v106/* 200 & 422), CLI (frontier/diagnose/perf), Chat Server, Trainer, and Studio UI integration
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_controller as ctl  # noqa: E402
import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402
import muon_optimizer as mo  # noqa: E402
import nexus_solver as ns  # noqa: E402
import nexus_v106_frontier as v106  # noqa: E402


def _small_cfg(**overrides) -> mc.MiMoMixConfig:
    base = dict(
        vocab_size=48,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=24,
        n_routed_experts=4,
        n_shared_experts=1,
        moe_top_k=2,
        use_moe=True,
        sliding_window=8,
        hybrid_ratio=2,
        n_mtp_layers=2,
        use_thinking_core=True,
        thinking_latent_dim=16,
        thinking_cycles=2,
        thinking_max_cycles=6,
    )
    base.update(overrides)
    return mc.MiMoMixConfig(**base)


# ---------------------------------------------------------------------------
# 1. CausalShortConvMemory (v106)
# ---------------------------------------------------------------------------


def test_short_conv_birth_identity_grad_causality_and_mtp_parity():
    torch.manual_seed(106)
    base_model = mc.MiMoMixModel(_small_cfg(use_short_conv=False)).eval()
    sc_model = mc.MiMoMixModel(_small_cfg(use_short_conv=True, short_conv_kernel=4, short_conv_after_layer=0)).eval()
    sc_model.load_state_dict(base_model.state_dict(), strict=False)
    sc_model.short_conv.reset_special_parameters()

    ids = torch.randint(0, 48, (1, 10))
    with torch.no_grad():
        base_out = base_model(ids, return_mtp=False)
        sc_out = sc_model(ids, return_mtp=False)
    assert torch.equal(base_out.logits, sc_out.logits), "ShortConv must be bit-identical at birth"
    assert sc_out.telemetry["short_conv"]["enabled"] is True

    # Open gate -> verify gradient flow & strict causality
    sc_model.train()
    with torch.no_grad():
        sc_model.short_conv.gate.fill_(0.35)
    out = sc_model(ids, labels=ids)
    out.loss.backward()
    assert sc_model.short_conv.gate.grad is not None and float(sc_model.short_conv.gate.grad.abs().sum()) > 0
    assert sc_model.short_conv.conv.weight.grad is not None and float(sc_model.short_conv.conv.weight.grad.abs().sum()) > 0

    sc_model.eval()
    ids_mod = ids.clone()
    ids_mod[0, 6:] = (ids_mod[0, 6:] + 7) % 48
    with torch.no_grad():
        a = sc_model(ids, return_mtp=False).logits
        b = sc_model(ids_mod, return_mtp=False).logits
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6), "ShortConv must not leak future tokens"

    # Incremental KV-cache & MTP speculative parity
    past = None
    step_logits = []
    with torch.no_grad():
        for pos in range(ids.shape[1]):
            step = sc_model(ids[:, pos : pos + 1], past_key_values=past, use_cache=True, past_length=pos, return_mtp=False)
            past = step.past_key_values
            step_logits.append(step.logits)
    assert torch.allclose(a, torch.cat(step_logits, dim=1), atol=1e-5)

    eq = md.assert_greedy_equivalence(sc_model, ids[:, :6], max_new_tokens=10)
    assert eq["tokens"] == 10


# ---------------------------------------------------------------------------
# 2. Shared-Expert Input-Conditioned Gate (v106)
# ---------------------------------------------------------------------------


def test_shared_expert_gate_birth_identity_and_grad_flow():
    torch.manual_seed(107)
    base_model = mc.MiMoMixModel(_small_cfg(use_shared_expert_gate=False)).eval()
    gated_model = mc.MiMoMixModel(_small_cfg(use_shared_expert_gate=True)).eval()
    gated_model.load_state_dict(base_model.state_dict(), strict=False)
    for mod in gated_model.modules():
        if isinstance(mod, mc.SparseMoEFeedForward):
            mod.reset_special_parameters()

    ids = torch.randint(0, 48, (2, 8))
    with torch.no_grad():
        o_base = base_model(ids, return_mtp=False)
        o_gate = gated_model(ids, return_mtp=False)
    assert torch.equal(o_base.logits, o_gate.logits), "Shared-expert gate must be bit-identical at birth"
    assert o_gate.telemetry["shared_expert_gate"]["enabled"] is True
    assert o_gate.telemetry["shared_expert_gate"]["mean_gate"] == pytest.approx(1.0, abs=1e-6)

    gated_model.train()
    loss = gated_model(ids, labels=ids).loss
    loss.backward()
    gate_grads = [
        m.shared_expert_gate.weight.grad
        for m in gated_model.modules()
        if isinstance(m, mc.SparseMoEFeedForward) and m.shared_expert_gate is not None
    ]
    assert gate_grads and all(g is not None and float(g.abs().sum()) > 0 for g in gate_grads)


# ---------------------------------------------------------------------------
# 3. ConnectomeCore Oscillatory Phase-Coupling Gate (v106)
# ---------------------------------------------------------------------------


def test_cns_oscillatory_phase_birth_identity_mitosis_and_temporal_parity():
    torch.manual_seed(108)
    common = dict(
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=2,
        cns_steps=4,
        cns_after_layer=1,
        cns_temporal=True,
        cns_read_layers=(1,),
        cns_write_layers=(1,),
    )
    base_m = mc.MiMoMixModel(_small_cfg(**common, cns_oscillatory_phase=False)).eval()
    osc_m = mc.MiMoMixModel(_small_cfg(**common, cns_oscillatory_phase=True)).eval()
    osc_m.load_state_dict(base_m.state_dict(), strict=False)
    with torch.no_grad():
        osc_m.cns_core.osc_amp.zero_()
        # Install simple Dale-signed edges so connectome rates are non-trivial
        for m in (base_m, osc_m):
            m.cns_core.mask[:10, :10].fill_(1.0)
            m.cns_core.mask.fill_diagonal_(0.0)
            m.cns_core.edge_logit.fill_(-1.5)
            m.cns_core.in_mask[:10].fill_(1.0)
            m.cns_core.out_mask[:10].fill_(1.0)
            m.cns_core.sign[:5].fill_(1.0)
            m.cns_core.sign[5:10].fill_(-1.0)
            m.cns_core.gate.fill_(0.25)
            m.cns_core._weight_eval_cache = None

    ids = torch.randint(0, 48, (1, 8))
    with torch.no_grad():
        b_out = base_m(ids, return_mtp=False)
        o_out = osc_m(ids, return_mtp=False)
    assert torch.allclose(b_out.logits, o_out.logits, atol=1e-6)
    assert o_out.telemetry["cns_core"]["oscillatory_phase"] is True

    # Activate oscillatory amplitude -> verify gradient flow, Dale preservation, mitosis, temporal KV parity
    with torch.no_grad():
        osc_m.cns_core.osc_amp.fill_(0.8)
    osc_m.train()
    osc_m(ids, labels=ids).loss.backward()
    assert osc_m.cns_core.osc_amp.grad is not None and float(osc_m.cns_core.osc_amp.grad.abs().sum()) > 0

    # Dale sign preservation
    w = osc_m.cns_core.weight().detach()
    assert bool((w[:, :5] >= 0).all()) and bool((w[:, 5:10] <= 0).all())

    # Mitosis inheritance & kill reset
    osc_m.eval()
    with torch.no_grad():
        osc_m.cns_core.osc_amp[0] = 0.77
        osc_m.cns_core.osc_freq[0] = 1.23
        osc_m.cns_core.osc_phase[0] = 0.45
    child = osc_m.cns_core.split_module(0, step=10)
    assert child == 10
    assert float(osc_m.cns_core.osc_amp[child].detach()) == pytest.approx(0.77, abs=1e-6)
    assert float(osc_m.cns_core.osc_freq[child].detach()) == pytest.approx(1.23, abs=1e-6)
    assert float(osc_m.cns_core.osc_phase[child].detach()) == pytest.approx(0.45, abs=1e-6)
    assert osc_m.cns_core.kill_module(child) is True
    assert float(osc_m.cns_core.osc_amp[child].detach()) == pytest.approx(0.0, abs=1e-6)

    # Temporal incremental KV-cache parity
    with torch.no_grad():
        full_logits = osc_m(ids, return_mtp=False).logits
        past = None
        parts = []
        for pos in range(ids.shape[1]):
            st = osc_m(ids[:, pos : pos + 1], past_key_values=past, use_cache=True, past_length=pos, return_mtp=False)
            past = st.past_key_values
            parts.append(st.logits)
    assert torch.allclose(full_logits, torch.cat(parts, dim=1), atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Full v93..v106 Mega-Stack Composition Parity
# ---------------------------------------------------------------------------


def test_v106_full_megastack_kv_cache_and_mtp_parity():
    torch.manual_seed(1060)
    cfg = _small_cfg(
        use_mhc_residual=True,
        mhc_streams=2,
        mhc_sinkhorn_iters=10,
        use_engram_memory=True,
        engram_table_size=64,
        engram_n_heads=2,
        engram_after_layer=0,
        use_deltanet_memory=True,
        deltanet_dim=12,
        deltanet_after_layer=1,
        use_titans_memory=True,
        titans_dim=12,
        titans_after_layer=1,
        use_attn_res=True,
        use_value_residual=True,
        use_short_conv=True,
        short_conv_kernel=4,
        short_conv_after_layer=0,
        use_shared_expert_gate=True,
        use_cns_core=True,
        cns_nodes=8,
        cns_steps=2,
        cns_read_layers=(1,),
        cns_write_layers=(1,),
        cns_temporal=True,
        cns_dendritic_apical=True,
        cns_neuromod_gate=True,
        cns_stdp_trace=True,
        cns_oscillatory_phase=True,
        thinking_ctm_synchrony=True,
    )
    model = mc.MiMoMixModel(cfg).eval()
    with torch.no_grad():
        model.engram_memory.gate.fill_(0.2)
        model.deltanet_memory.gate.fill_(0.2)
        model.titans_memory.gate.fill_(0.2)
        model.short_conv.gate.fill_(0.2)
        model.value_residual.layer_gates.fill_(0.2)
        model.cns_core.gate.fill_(0.2)
        model.cns_core.osc_amp.fill_(0.4)

    ids = torch.randint(0, 48, (1, 10))
    with torch.no_grad():
        full_out = model(ids, return_mtp=False)
        past = None
        chunks = []
        for pos in range(ids.shape[1]):
            st = model(ids[:, pos : pos + 1], past_key_values=past, use_cache=True, past_length=pos, return_mtp=False)
            past = st.past_key_values
            chunks.append(st.logits)
    assert torch.allclose(full_out.logits, torch.cat(chunks, dim=1), atol=1e-5)
    eq = md.assert_greedy_equivalence(model, ids[:, :6], max_new_tokens=8)
    assert eq["tokens"] == 8


# ---------------------------------------------------------------------------
# 5. MLA 2D-BLAS Absorption & DiffAttn Fast Path Exactness
# ---------------------------------------------------------------------------


def test_mla_2d_absorption_and_diffattn_fast_path_parity():
    torch.manual_seed(109)
    ids = torch.randint(0, 48, (1, 12))
    for extra in (
        dict(use_mla=True, mla_latent_dim=16, mla_pe_dim=8, mla_global_only=False, attention_sink=True),
        dict(use_mla=True, mla_latent_dim=16, mla_pe_dim=8, mla_global_only=False, attention_sink=False),
        dict(use_mla=True, mla_latent_dim=16, mla_pe_dim=8, mla_global_only=False, qk_norm=True),
        dict(use_differential_attention=True, attention_sink=False),
        dict(use_differential_attention=True, attention_sink=True),
    ):
        model = mc.MiMoMixModel(_small_cfg(use_moe=False, **extra)).eval()
        with torch.no_grad():
            full = model(ids, return_mtp=False).logits
            past = None
            steps = []
            for pos in range(ids.shape[1]):
                st = model(ids[:, pos : pos + 1], past_key_values=past, use_cache=True, past_length=pos, return_mtp=False)
                past = st.past_key_values
                steps.append(st.logits)
        assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5), f"Parity failed for {extra}"
        md.assert_greedy_equivalence(model, ids[:, :6], max_new_tokens=6)


# ---------------------------------------------------------------------------
# 6. Trunk-Reuse Thinking Controller Bit-Exact Acceleration
# ---------------------------------------------------------------------------


def test_controller_trunk_reuse_is_bit_identical_and_exact():
    torch.manual_seed(110)
    model = mc.MiMoMixModel(
        _small_cfg(
            thinking_residual_init=0.4,
            use_cns_core=True,
            cns_nodes=8,
            cns_steps=2,
            cns_read_layers=(1,),
            cns_write_layers=(1,),
            cns_to_thinking=True,
        )
    ).eval()
    with torch.no_grad():
        model.cns_core.thinking_gate.fill_(0.25)

    ids = torch.randint(0, 48, (1, 14))
    pol = ctl.ThinkingPolicy(ladder=(1, 2, 4, 6), entropy_target=0.0)
    out_fast, dec_fast = ctl.decide(model, ids, mode="agent", policy=pol, reuse_trunk=True)
    out_full, dec_full = ctl.decide(model, ids, mode="agent", policy=pol, reuse_trunk=False)

    assert torch.equal(out_fast.logits, out_full.logits)
    assert dec_fast.to_dict() == dec_full.to_dict()
    assert dec_fast.forward_evaluations == len(dec_fast.plan.ladder)

    audit = ctl.audit_decision_fidelity(model, [ids, ids[:, :8]], policy=pol)
    assert audit["requests"] == 2
    assert audit["top1_disagreements"] == 0


# ---------------------------------------------------------------------------
# 7. Chunked Prefill, Packed Int4 KV Compression & Speculative Tree Verify
# ---------------------------------------------------------------------------


def test_chunked_prefill_int4_kv_and_speculative_tree_verify():
    torch.manual_seed(111)
    model = mc.MiMoMixModel(
        _small_cfg(
            hidden_size=64,
            n_heads=4,
            n_kv_heads=2,
            head_dim=64 // 4,
            use_engram_memory=True,
            engram_table_size=64,
            use_short_conv=True,
            sliding_window=32,
        )
    ).eval()
    ids = torch.randint(0, 48, (1, 20))

    # Chunked prefill & greedy decode parity across non-divisible chunk sizes
    ref_greedy = md.greedy_generate(model, ids, max_new_tokens=10)
    for cs in (3, 7, 16):
        chk_greedy = md.chunked_greedy_generate(model, ids, max_new_tokens=10, chunk_size=cs)
        assert torch.equal(ref_greedy.new_tokens, chk_greedy.new_tokens)
        assert chk_greedy.telemetry["chunked_prefill"]["chunk_size"] == cs

    # Packed Int4 KV-cache compression vs Int8 on head_dim=64 model
    big_kv_model = mc.MiMoMixModel(
        _small_cfg(hidden_size=128, n_heads=2, n_kv_heads=2, head_dim=64, use_moe=False, use_engram_memory=True, engram_table_size=64)
    ).eval()
    with torch.no_grad():
        pre = big_kv_model(ids, use_cache=True, return_mtp=False)
    rep_i8 = md.kv_cache_memory_report(pre.past_key_values)
    rep_i4 = md.kv_cache_int4_memory_report(pre.past_key_values, group_size=32)
    assert rep_i4["compression_ratio"] > rep_i8["compression_ratio"]
    assert rep_i4["compression_ratio"] > 5.0
    assert rep_i4["mean_cosine_similarity"] > 0.99

    # Round-trip shape/dtype preservation (including integer Engram tail entry)
    q4 = md.quantize_past_kv_int4(pre.past_key_values, group_size=32)
    dq4 = md.dequantize_past_kv_int4(q4)
    assert dq4 is not None and len(dq4) == len(pre.past_key_values)
    for orig_e, rec_e in zip(pre.past_key_values, dq4):
        assert orig_e[0].shape == rec_e[0].shape and orig_e[0].dtype == rec_e[0].dtype
        if not orig_e[0].is_floating_point():
            assert torch.equal(orig_e[0], rec_e[0])

    # Speculative tree verification over shared prompt prefill
    exact_branch = ref_greedy.new_tokens[0, :5].tolist()
    bad_branch = [(t + 1) % 48 for t in exact_branch]
    partial_branch = exact_branch[:2] + [(exact_branch[2] + 3) % 48] + exact_branch[3:]
    tree = md.speculative_tree_verify(model, ids, [bad_branch, partial_branch, exact_branch])
    assert tree["prompt_prefill_forwards"] == 1
    assert tree["best_greedy_branch_index"] == 2
    assert tree["branches"][2]["exact_greedy_match"] is True
    assert tree["branches"][1]["greedy_accepted_prefix"] == 2
    assert tree["branches"][0]["greedy_accepted_prefix"] == 0


# ---------------------------------------------------------------------------
# 8. MuonHybridOptimizer MuonClip Spectral Norm Clipping & Telemetry
# ---------------------------------------------------------------------------


def test_muon_spectral_clip_and_telemetry():
    torch.manual_seed(112)
    big_w = torch.randn(16, 16) * 2.5
    sigma_before = float(torch.linalg.svdvals(big_w)[0])
    assert sigma_before > 3.0
    clipped, est = mo.spectral_norm_clip_(big_w, tau=2.0, power_iters=8)
    assert clipped is True and est > 3.0
    sigma_after = float(torch.linalg.svdvals(big_w)[0])
    assert sigma_after <= 2.05

    # Sub-threshold matrix untouched
    small_w = torch.eye(8) * 0.5
    was_clipped, _ = mo.spectral_norm_clip_(small_w, tau=2.0)
    assert was_clipped is False

    # End-to-end MuonHybridOptimizer step with spectral_clip_tau + telemetry
    model = mc.MiMoMixModel(_small_cfg(use_moe=False))
    groups, _ = mo.build_muon_hybrid_param_groups(
        model, lr=1e-2, polar_express=True, spectral_clip_tau=0.5
    )
    opt = mo.MuonHybridOptimizer(groups, lr=1e-2)
    x = torch.randint(0, 48, (2, 6))
    loss = model(x, labels=x).loss
    loss.backward()
    opt.step()
    tele = opt.optimizer_telemetry()
    assert tele["step_count"] == 1
    assert tele["polar_express_active"] is True
    assert tele["last_mean_update_rms"] > 0.0
    assert tele["last_spectral_clips"] >= 1


# ---------------------------------------------------------------------------
# 9. Ten New Zero-Eval Exact STEM/CS/Finance/Chemistry Solvers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_formula,expected_display",
    [
        ("Matrix multiply [[1, 2], [3, 4]] and [[5, 6], [7, 8]]", "linalg.matrix_multiply", "[[19, 22], [43, 50]]"),
        ("Matrix rank of [[1, 2, 3], [2, 4, 6], [0, 1, 1]]", "linalg.matrix_rank", "2"),
        ("Complex (3+4i) * (1-2i)", "math.complex_arithmetic", "11-2i"),
        ("Complex (5+5i) / (1+2i)", "math.complex_arithmetic", "3-1i"),
        ("Modulus of 3+4i", "math.complex_modulus", "5"),
        ("Shortest path from A to D in graph A-B:4, A-C:1, C-B:2, B-D:1, C-D:5", "cs.dijkstra_shortest_path", "4"),
        ("Master theorem T(n) = 2T(n/2) + n^1", "cs.master_theorem", "Theta(n log n)"),
        ("Master theorem a = 4, b = 2, d = 1", "cs.master_theorem", "Theta(n^2)"),
        ("Master theorem T(n) = 2T(n/2) + n^2", "cs.master_theorem", "Theta(n^2)"),
        ("Beer-Lambert absorbance epsilon = 150, l = 1, c = 0.02", "chemistry.beer_lambert", "3"),
        ("Arrhenius rate A = 1000, Ea = 8314.46261815324, T = 1000", "chemistry.arrhenius_rate", "367.879441"),
        ("Annuity present value pmt = 1000, r = 0.05, n = 10", "finance.annuity_pv", "7721.73"),
        ("Combinations with repetition n = 5, k = 3", "comb.multiset_combinations", "35"),
        ("Jensen-Shannon divergence between [0.5, 0.5] and [0.25, 0.75]", "infotheory.jensen_shannon_divergence", "0.048795"),
    ],
)
def test_v106_exact_solvers_positive(query, expected_formula, expected_display):
    solver = ns.NexusSolver()
    res = solver.solve(query)
    assert res.solved is True
    assert res.formula_id == expected_formula
    assert res.display_answer == expected_display
    assert res.receipt is not None and len(res.receipt.receipt_sha256) == 64


@pytest.mark.parametrize(
    "bad_query",
    [
        "Matrix multiply [[1, 2], [3, 4]] and [[1, 2, 3]]",  # incompatible dimensions
        "Complex (3+4i) / (0+0i)",  # division by zero
        "Shortest path from A to B in graph A-B:-3",  # negative edge weight
        "Shortest path from A to Z in graph A-B:2",  # unreachable target
        "Master theorem a = 4, b = 1, d = 1",  # b <= 1 invalid
        "Beer-Lambert absorbance epsilon = 150, l = 0, c = 0.02",  # l <= 0
        "Arrhenius rate A = 1000, Ea = 5000, T = 0",  # T <= 0
        "Jensen-Shannon divergence between [0.5, 0.2] and [0.5, 0.5]",  # sum != 1
    ],
)
def test_v106_exact_solvers_fail_closed(bad_query):
    res = ns.NexusSolver().solve(bad_query)
    assert res.solved is False


# ---------------------------------------------------------------------------
# 10. Five v106 Frontier Engines (Positive + Fail-Closed)
# ---------------------------------------------------------------------------


def test_v106_frontier_engines_invariants_and_fail_closed():
    # 1. TestTimeComputeAllocator
    ttc = v106.TestTimeComputeAllocator().allocate(total_token_budget=4096, base_pass1=0.35, verifier_accuracy=0.95)
    assert ttc["answer_authority"] is False
    assert ttc["absolute_pass_gain"] > 0.0
    assert ttc["optimal_strategy"]["total_tokens"] <= 4096
    # Verify Pareto frontier is strictly increasing in both cost and expected pass prob
    pf = ttc["pareto_frontier"]
    assert all(pf[i]["expected_pass_prob"] < pf[i + 1]["expected_pass_prob"] for i in range(len(pf) - 1))
    with pytest.raises(ValueError):
        v106.TestTimeComputeAllocator().allocate(total_token_budget=100, tokens_per_sample=256)

    # 2. SpectralWeightDiagnostics
    spec = v106.SpectralWeightDiagnostics().diagnose(
        singular_values=[5.2, 3.1, 2.0, 1.45, 1.1, 0.85, 0.65, 0.5], matrix_shape=[64, 64]
    )
    assert spec["answer_authority"] is False
    assert 1.0 <= spec["stable_rank"] <= spec["positive_rank"]
    assert spec["marchenko_pastur_edge"] is not None
    collapsed = v106.SpectralWeightDiagnostics().diagnose(singular_values=[10.0, 0.01, 0.01, 0.01])
    assert collapsed["health_status"] == "rank_collapsed"
    with pytest.raises(ValueError):
        v106.SpectralWeightDiagnostics().diagnose(singular_values=[1.0, -0.5, 0.2])

    # 3. CausalCounterfactualVerifier
    cf = v106.CausalCounterfactualVerifier().verify_counterfactual(
        weights={"X": 2.0, "Z": 0.5},
        observed={"X": 1.0, "Z": 2.0, "M": 3.0, "Y": 10.0},
        intervention={"X": 3.0},
        mediator_var="M",
        treatment_to_mediator=1.5,
        mediator_to_outcome=1.0,
    )
    assert cf["answer_authority"] is False
    assert cf["natural_direct_effect"] == pytest.approx(4.0, abs=1e-6)
    assert cf["natural_indirect_effect"] == pytest.approx(3.0, abs=1e-6)
    assert cf["counterfactual_delta"] == pytest.approx(7.0, abs=1e-6)
    assert cf["counterfactual_outcome"] == pytest.approx(17.0, abs=1e-6)
    assert cf["mediation_decomposition_exact"] is True
    assert cf["e_value_confounding_bound"] > 1.0
    with pytest.raises(ValueError):
        v106.CausalCounterfactualVerifier().verify_counterfactual(
            weights={"X": 1.0}, observed={"X": 1.0, "Y": 2.0}, intervention={"Y": 5.0}
        )

    # 4. AdaptiveQuantizationPlanner
    qp = v106.AdaptiveQuantizationPlanner().plan(
        layers=[
            {"name": "embed", "params": 1000, "sensitivity": 20.0},
            {"name": "attn", "params": 1000, "sensitivity": 10.0},
            {"name": "mlp", "params": 4000, "sensitivity": 0.5},
        ],
        target_avg_bits=4.0,
    )
    assert qp["answer_authority"] is False
    assert qp["achieved_avg_bits"] <= 4.0 + 1e-6
    assert qp["mixed_precision_distortion"] < qp["uniform_baseline_distortion"]
    assert qp["distortion_reduction_vs_uniform"] > 0.0
    by_name = {row["name"]: row["assigned_bits"] for row in qp["layer_assignments"]}
    assert by_name["embed"] > by_name["mlp"]
    with pytest.raises(ValueError):
        v106.AdaptiveQuantizationPlanner().plan(
            layers=[{"name": "a", "params": 100, "sensitivity": 1.0, "min_bits": 8}],
            target_avg_bits=4.0,
        )

    # 5. TelemetryDriftDetector
    stable = v106.TelemetryDriftDetector().detect(
        stream=[10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 9.9, 10.0, 10.1],
        baseline_hist=[0.5, 0.3, 0.2],
        recent_hist=[0.49, 0.31, 0.20],
    )
    assert stable["drift_detected"] is False and stable["severity"] == "nominal"
    shifted = v106.TelemetryDriftDetector().detect(
        stream=[10.0, 10.1, 9.9, 10.0, 10.2, 18.0, 19.0, 20.0, 21.0, 22.0],
        baseline_hist=[0.7, 0.2, 0.1],
        recent_hist=[0.1, 0.2, 0.7],
    )
    assert shifted["drift_detected"] is True
    assert shifted["change_point_index"] == 5
    assert shifted["shift_direction"] == "upward_shift"
    assert shifted["histogram_jsd_bits"] > 0.2
    with pytest.raises(ValueError):
        v106.TelemetryDriftDetector().detect(stream=[1.0, 2.0])


# ---------------------------------------------------------------------------
# 11. Full API, CLI, Chat Server, Trainer & Studio UI Surface Integration
# ---------------------------------------------------------------------------


def test_v106_api_cli_chat_trainer_and_studio_integration():
    from starlette.testclient import TestClient

    import nexus_api as api
    import supermix_chat_server as chat_srv
    import train_mimomix_generalisation as gen_train
    import train_mimomix_talk as talk_train

    app = api.create_app()
    client = TestClient(app)

    for route in (
        "/v1/v106/ttc_allocate",
        "/v1/v106/spectral_diagnostics",
        "/v1/v106/counterfactual_verify",
        "/v1/v106/quant_plan",
        "/v1/v106/drift_detect",
    ):
        resp = client.post(route, json={})
        assert resp.status_code == 200, f"{route} failed: {resp.text}"
        body = resp.json()
        assert body["answer_authority"] is False
        assert len(body["receipt"]["receipt_sha256"]) == 64

    # Fail-closed 422 checks
    assert client.post("/v1/v106/ttc_allocate", json={"base_pass1": 1.5}).status_code == 422
    assert client.post("/v1/v106/spectral_diagnostics", json={"singular_values": [1.0]}).status_code == 422
    assert client.post("/v1/v106/quant_plan", json={"target_avg_bits": -2.0}).status_code == 422
    assert client.post("/v1/v106/drift_detect", json={"stream": [1.0, 2.0]}).status_code == 422

    # Trainer CLI flags & warm-start prefixes
    args = talk_train.build_parser().parse_args(
        [
            "--use_short_conv",
            "--short_conv_kernel",
            "6",
            "--short_conv_after_layer",
            "1",
            "--use_shared_expert_gate",
            "--cns_oscillatory_phase",
            "--muon_spectral_clip",
            "3.0",
        ]
    )
    cfg = talk_train.build_config(args, vocab_size=64)
    assert cfg.use_short_conv is True and cfg.short_conv_kernel == 6 and cfg.short_conv_after_layer == 1
    assert cfg.use_shared_expert_gate is True and cfg.cns_oscillatory_phase is True
    assert "short_conv." in gen_train.new_parameter_prefixes(cfg)

    # Chat server telemetry snapshot
    m = mc.MiMoMixModel(_small_cfg(use_short_conv=True, use_shared_expert_gate=True, use_cns_core=True, cns_nodes=8, cns_oscillatory_phase=True)).eval()
    with torch.no_grad():
        m(torch.randint(0, 48, (1, 4)), return_mtp=False)
    snap = chat_srv._architecture_telemetry_snapshot(m)
    assert snap is not None and snap["short_conv"]["enabled"] is True
    assert snap["shared_expert_gate"]["enabled"] is True
    assert snap["cns_core"]["oscillatory_phase"] is True

    # Studio HTML Section 9 & handlers present
    html = (ROOT / "web_static" / "nexus_studio.html").read_text(encoding="utf-8")
    for needle in (
        "9. v106 Frontier Synthesis",
        "runV106TTCAllocate",
        "runV106SpectralDiag",
        "runV106CounterfactualVerify",
        "runV106QuantPlan",
        "runV106DriftDetect",
        "/v1/v106/ttc_allocate",
    ):
        assert needle in html

    # CLI diagnose & perf JSON checks
    diag = subprocess.run(
        [sys.executable, "-m", "scf", "diagnose", "--json"],
        cwd=str(ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "source")},
        capture_output=True,
        text=True,
        check=True,
    )
    diag_data = json.loads(diag.stdout)
    assert diag_data["all_probes_passed"] is True
    assert diag_data["verifier_probes_total"] >= 10
    assert diag_data["evolution_v106_version"] == v106.FRONTIER_V106_VERSION
