"""Comprehensive test suite for Supermix v107: Universal Fast-Weight Memory,
Sonapse Synthetic Neuron Foundry, Connectome Plasticity, MARS-Muon, Outlier-Mixed
Int4 KV Compression, 10 Exact Solvers, and 5 v107 Morphogenesis Engines.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
for _p in (ROOT, SOURCE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402
import model_variants as mv  # noqa: E402
import muon_optimizer as mo  # noqa: E402
import nexus_api as napi  # noqa: E402
import nexus_solver as ns  # noqa: E402
import nexus_v107_frontier as v107  # noqa: E402
import sonapse_neuron_foundry as snf  # noqa: E402
import supermix_chat_server as chat_srv  # noqa: E402
import train_mimomix_generalisation as train_gen  # noqa: E402
import train_mimomix_talk as train_talk  # noqa: E402
from scf import __main__ as scf_cli  # noqa: E402
from scf.models import ChampionNetAdapter  # noqa: E402
from scf.verification import SCFVerificationGate  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Universal Fast-Weight Memory (uTTT-MoE & uTTT-Dense)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["moe", "dense"])
def test_ufwm_birth_identity_causality_grads_and_kv_speculative_parity(mode: str):
    base_kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=24,
        n_routed_experts=4,
        moe_top_k=2,
        n_mtp_layers=2,
        sliding_window=8,
    )
    torch.manual_seed(107)
    base_model = mc.MiMoMixModel(mc.MiMoMixConfig(**base_kwargs)).eval()

    torch.manual_seed(107)
    ufwm_cfg = mc.MiMoMixConfig(
        **base_kwargs,
        use_ufwm=True,
        ufwm_dim=12,
        ufwm_banks=4,
        ufwm_top_k=2,
        ufwm_mode=mode,
    )
    ufwm_model = mc.MiMoMixModel(ufwm_cfg)
    # Copy shared trunk weights so we isolate UFWM zero-gate birth identity
    ufwm_model.load_state_dict(base_model.state_dict(), strict=False)
    ufwm_model.eval()

    tokens = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        out_base = base_model(tokens, return_mtp=False)
        out_ufwm = ufwm_model(tokens, return_mtp=False)
    assert torch.equal(out_base.logits, out_ufwm.logits)
    assert out_ufwm.telemetry["ufwm"]["enabled"] is True
    assert out_ufwm.telemetry["ufwm"]["mode"] == mode

    # First backward pass must deliver non-zero gradients to UFWM gates
    ufwm_model.train()
    labels = torch.randint(0, 64, (2, 10))
    loss_out = ufwm_model(tokens, labels=labels)
    assert loss_out.loss is not None
    loss_out.loss.backward()
    assert ufwm_model.ufwm is not None
    assert float(ufwm_model.ufwm.layer_gates.grad.abs().sum()) > 0.0
    assert float(ufwm_model.ufwm.temporal_gate.grad.abs().sum()) > 0.0

    # Open UFWM depth and temporal gates, verify parameter gradients, causality, KV & MTP parity
    with torch.no_grad():
        ufwm_model.ufwm.layer_gates.fill_(0.20)
        ufwm_model.ufwm.temporal_gate.fill_(0.25)
    ufwm_model.zero_grad(set_to_none=True)
    loss2 = ufwm_model(tokens, labels=labels)
    assert loss2.loss is not None
    loss2.loss.backward()
    for p_name in ("bank_init", "layer_role_emb", "k_proj", "v_proj", "q_proj", "out_proj"):
        sub = getattr(ufwm_model.ufwm, p_name)
        tensor = sub if isinstance(sub, torch.Tensor) else sub.weight
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0.0, p_name

    # Strict left-to-right causality check
    ufwm_model.eval()
    tokens_pert = tokens.clone()
    tokens_pert[:, 5:] = (tokens_pert[:, 5:] + 7) % 64
    with torch.no_grad():
        logits_a = ufwm_model(tokens, return_mtp=False).logits
        logits_b = ufwm_model(tokens_pert, return_mtp=False).logits
    assert torch.allclose(logits_a[:, :5], logits_b[:, :5], atol=1e-6)
    assert not torch.allclose(logits_a[:, 5:], logits_b[:, 5:], atol=1e-4)

    # Batched (B, T) vs incremental T x (B, 1) KV-cache parity
    with torch.no_grad():
        past = None
        step_logits = []
        for t in range(tokens.shape[1]):
            step_out = ufwm_model(
                tokens[:, t : t + 1],
                past_key_values=past,
                use_cache=True,
                return_mtp=False,
                past_length=t,
            )
            past = step_out.past_key_values
            step_logits.append(step_out.logits)
        cached_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(logits_a, cached_logits, atol=1e-5)

    # Exact MTP speculative decoding == greedy decoding
    prompt = tokens[:1, :6]
    g_res = md.greedy_generate(ufwm_model, prompt, max_new_tokens=8)
    s_res = md.speculative_generate(ufwm_model, prompt, max_new_tokens=8)
    assert torch.equal(g_res.sequences, s_res.sequences)


def test_ufwm_and_foundry_config_validation_fails_closed():
    with pytest.raises(ValueError, match="ufwm_dim"):
        mc.MiMoMixConfig(ufwm_dim=1)
    with pytest.raises(ValueError, match="ufwm_mode"):
        mc.MiMoMixConfig(ufwm_mode="invalid")
    with pytest.raises(ValueError, match="ufwm_top_k"):
        mc.MiMoMixConfig(ufwm_banks=3, ufwm_top_k=5, ufwm_mode="moe")
    with pytest.raises(ValueError, match="neuron_foundry_dim"):
        mc.MiMoMixConfig(neuron_foundry_dim=2)
    with pytest.raises(ValueError, match="neuron_foundry_after_layer"):
        mc.MiMoMixConfig(n_layers=3, use_neuron_foundry=True, neuron_foundry_after_layer=3)


# ---------------------------------------------------------------------------
# 2. Sonapse Synthetic Neuron Foundry & ChampionNet Expert Head
# ---------------------------------------------------------------------------


def test_all_six_synthetic_neuron_families_and_foundry_layer_parity():
    assert len(snf.NEURON_FOUNDRY_FAMILIES) == 6
    cells = [
        snf.SolitonAxonNeuron(12),
        snf.SpineApparatusNeuron(12),
        snf.AstrocyticTripartiteNeuron(12),
        snf.EphapticFieldNeuron(12),
        snf.RetrogradeEndocannabinoidNeuron(12),
        snf.FastWeightEngramNeuron(12),
    ]
    for cell in cells:
        x = torch.randn(3, 12, requires_grad=True)
        st0 = torch.zeros(3, cell.state_mult * 12)
        y1, st1 = cell.step(x, st0)
        y2, st2 = cell.step(x * 0.5, st1)
        assert y2.shape == (3, 12)
        assert st2.shape == (3, cell.state_mult * 12)
        assert torch.isfinite(y2).all() and torch.isfinite(st2).all()
        (y2.pow(2).mean() + st2.pow(2).mean()).backward()
        assert x.grad is not None and float(x.grad.abs().sum()) > 0.0

    # SonapseNeuronFoundryLayer inside MiMoMixModel: birth identity + KV/MTP parity
    base_kw = dict(vocab_size=48, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2, n_mtp_layers=2)
    torch.manual_seed(207)
    m_base = mc.MiMoMixModel(mc.MiMoMixConfig(**base_kw)).eval()
    torch.manual_seed(207)
    m_nf = mc.MiMoMixModel(
        mc.MiMoMixConfig(**base_kw, use_neuron_foundry=True, neuron_foundry_dim=12, neuron_foundry_after_layer=1)
    )
    m_nf.load_state_dict(m_base.state_dict(), strict=False)
    m_nf.eval()

    ids = torch.randint(0, 48, (2, 9))
    with torch.no_grad():
        assert torch.equal(m_base(ids, return_mtp=False).logits, m_nf(ids, return_mtp=False).logits)
        assert m_nf.neuron_foundry is not None
        m_nf.neuron_foundry.gate.fill_(0.30)
        full_logits = m_nf(ids, return_mtp=False).logits

        past = None
        parts = []
        for t in range(ids.shape[1]):
            st = m_nf(ids[:, t : t + 1], past_key_values=past, use_cache=True, return_mtp=False, past_length=t)
            past = st.past_key_values
            parts.append(st.logits)
        assert torch.allclose(full_logits, torch.cat(parts, dim=1), atol=1e-5)

    g = md.greedy_generate(m_nf, ids[:1, :5], max_new_tokens=7)
    s = md.speculative_generate(m_nf, ids[:1, :5], max_new_tokens=7)
    assert torch.equal(g.sequences, s.sequences)


def test_champion_net_universal_fast_weight_expert_and_scf_isolation():
    assert "universal_fast_weight_expert" in mv.SUPPORTED_MODEL_SIZES
    model = mv.build_model("universal_fast_weight_expert").eval()
    sd = model.state_dict()
    assert mv.detect_model_size_from_state_dict(sd) == "universal_fast_weight_expert"

    x3 = torch.randn(2, 1, 128)
    x2_head = torch.randn(2, 256)
    head = model.layers[10]
    with torch.no_grad():
        out3 = model(x3)
        out2_head = head(x2_head)
    assert out3.shape == (2, 1, 10)
    assert out2_head.shape == (2, 10)
    assert head.last_active_banks == 2
    assert head.last_metabolic_cost > 0.0

    # Warm-start from base ChampionNet checkpoint preserves exact logits at alpha=beta=0
    base = mv.build_model("base").eval()
    missing, unexpected = mv.load_weights_for_model(model, base.state_dict(), "universal_fast_weight_expert")
    assert missing == [] and unexpected == []
    with torch.no_grad():
        assert torch.allclose(base(x3), model(x3), atol=1e-6)

    # ChampionNetAdapter experimental isolation contract
    adapter = ChampionNetAdapter()
    blocked = adapter.build(model_size="universal_fast_weight", allow_experimental=False)
    assert blocked["built"] is False and blocked["experimental_isolated"] is True
    built = adapter.build(model_size="universal_fast_weight", allow_experimental=True)
    assert built["built"] is True and built["production_ready"] is False


# ---------------------------------------------------------------------------
# 3. Connectome Plasticity, Astrocyte/Retrograde Physiology & Mega-Stack
# ---------------------------------------------------------------------------


def test_connectome_astrocyte_retrograde_btsp_and_ewc_consolidation():
    cfg = mc.MiMoMixConfig(
        vocab_size=48,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=2,
        cns_steps=3,
        cns_after_layer=1,
        cns_temporal=True,
        cns_astrocyte_gate=True,
        cns_retrograde_ephaptic=True,
    )
    model = mc.MiMoMixModel(cfg)
    core = model.cns_core
    assert core is not None

    # Install synthetic Dale-signed edges on live slots 0..9
    with torch.no_grad():
        core.alive.zero_()
        core.alive[:10] = 1
        core.sign.fill_(1.0)
        core.sign[7:10] = -1.0  # inhibitory modules
        core.in_mask[:5] = 1.0
        core.out_mask[5:10] = 1.0
        for post in range(10):
            for pre in range(10):
                if post != pre and ((post + pre) % 2 == 0):
                    core.mask[post, pre] = 1.0
                    core.edge_logit[post, pre] = -1.2 if (post + pre) % 4 == 0 else -0.4
        core.gate.fill_(0.35)

    # At birth, astro_strength == 0 and ephaptic/retro_strength == 0 -> exact baseline match
    ids = torch.randint(0, 48, (1, 7))
    model.eval()
    with torch.no_grad():
        logits_zero = model(ids, return_mtp=False).logits
        core.astro_strength.fill_(0.8)
        core.ephaptic_strength.fill_(0.6)
        core.retro_strength.fill_(0.5)
        core._step_params_eval_cache = None
        logits_active = model(ids, return_mtp=False).logits
    assert not torch.allclose(logits_zero, logits_active, atol=1e-5)

    # Temporal KV-cache parity with active astrocyte + retrograde/ephaptic gates
    with torch.no_grad():
        past = None
        step_outs = []
        for t in range(ids.shape[1]):
            r = model(ids[:, t : t + 1], past_key_values=past, use_cache=True, return_mtp=False, past_length=t)
            past = r.past_key_values
            step_outs.append(r.logits)
        assert torch.allclose(logits_active, torch.cat(step_outs, dim=1), atol=1e-5)

    # Mitosis & apoptosis copy/reset v107 attributes cleanly
    child = core.split_module(0, step=10)
    assert child == 10
    assert float(core.astro_strength[child].detach()) == pytest.approx(0.8)
    assert float(core.ephaptic_strength[child].detach()) == pytest.approx(0.6)
    assert core.kill_module(child) is True
    assert float(core.astro_strength[child].detach()) == 0.0

    # Three-Factor BTSP + MICrONS Spine-Apparatus update preserves Dale signs strictly
    w_before = core.weight().detach().clone()
    pre_act = torch.rand(16, 12)
    post_act = torch.rand(16, 12)
    btsp_rep = core.three_factor_btsp_update(
        pre_act, post_act, modulatory_signal=0.8, lr=0.05, spine_threshold=0.35, target_radius=0.90
    )
    assert btsp_rep["updated_edges"] > 0
    assert 0.0 <= btsp_rep["mature_fraction"] <= 1.0
    w_after = core.weight().detach()
    conn = core.mask > 0
    # Excitatory columns stay >= 0, inhibitory columns stay <= 0
    assert bool((w_after[:, :7][conn[:, :7]] > 0).all())
    assert bool((w_after[:, 7:10][conn[:, 7:10]] < 0).all())
    assert not torch.allclose(w_before, w_after)
    with pytest.raises(ValueError, match="lr"):
        core.three_factor_btsp_update(pre_act, post_act, 0.5, lr=0.0)

    # SONAPSE-SYMBIOSIS EWC synaptic consolidation pulls toward anchor while keeping Dale signs
    anchor_logits = core.edge_logit.detach().clone()
    with torch.no_grad():
        core.edge_logit.add_(0.4 * core.mask)
    fisher = torch.rand_like(core.edge_logit) + 0.5
    ewc_rep = core.ewc_synaptic_consolidate(anchor_logits, fisher, pull_strength=0.5)
    assert ewc_rep["consolidated_edges"] > 0
    assert ewc_rep["mean_mag_drift_after"] < ewc_rep["mean_mag_drift_before"]


def test_full_v93_to_v107_megastack_kv_and_speculative_parity():
    torch.manual_seed(307)
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=24,
        n_routed_experts=4,
        moe_top_k=2,
        n_mtp_layers=2,
        use_mhc_residual=True,
        use_engram_memory=True,
        engram_table_size=128,
        use_deltanet_memory=True,
        deltanet_dim=12,
        use_titans_memory=True,
        titans_dim=12,
        use_attn_res=True,
        use_value_residual=True,
        use_short_conv=True,
        use_shared_expert_gate=True,
        use_neuron_foundry=True,
        neuron_foundry_dim=12,
        neuron_foundry_after_layer=1,
        use_ufwm=True,
        ufwm_dim=12,
        ufwm_banks=4,
        ufwm_top_k=2,
        use_cns_core=True,
        cns_nodes=10,
        cns_steps=2,
        cns_temporal=True,
        cns_stdp_trace=True,
        cns_oscillatory_phase=True,
        cns_astrocyte_gate=True,
        cns_retrograde_ephaptic=True,
    )
    model = mc.MiMoMixModel(cfg)
    with torch.no_grad():
        assert model.ufwm is not None and model.neuron_foundry is not None
        model.ufwm.layer_gates.fill_(0.15)
        model.ufwm.temporal_gate.fill_(0.20)
        model.neuron_foundry.gate.fill_(0.18)
        assert model.deltanet_memory is not None and model.titans_memory is not None
        model.deltanet_memory.gate.fill_(0.12)
        model.titans_memory.gate.fill_(0.12)
        assert model.short_conv is not None and model.engram_memory is not None
        model.short_conv.gate.fill_(0.10)
        model.engram_memory.gate.fill_(0.10)
    model.eval()

    prompt = torch.randint(0, 64, (1, 8))
    g_res = md.greedy_generate(model, prompt, max_new_tokens=8)
    s_res = md.speculative_generate(model, prompt, max_new_tokens=8)
    assert torch.equal(g_res.sequences, s_res.sequences)


# ---------------------------------------------------------------------------
# 4. Performance Optimizations, MARS-Muon & Outlier-Mixed Int4 KV Cache
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sink_on", [True, False])
def test_rotary_position_cache_and_zero_copy_gqa_exactness(sink_on: bool):
    cfg = mc.MiMoMixConfig(
        vocab_size=32, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, attention_sink=sink_on
    )
    rot = mc.RotaryEmbedding(cfg)
    pos = rot.position_range(5, 11, torch.device("cpu"))
    assert torch.equal(pos, torch.arange(5, 16, dtype=torch.long))

    attn = mc.HybridAttention(cfg, layer_index=0, kind="global").eval()
    h = torch.randn(2, 6, 32)
    q_pos = rot.position_range(0, 6, torch.device("cpu"))
    k_pos = rot.position_range(0, 0, torch.device("cpu"))
    cos, sin = rot.forward_range(0, 6, torch.device("cpu"))
    with torch.no_grad():
        full_out, present = attn(h, cos, sin, q_pos, k_pos, use_cache=True, contiguous_causal=True)
        # Single-token decode step against cached K/V
        h_next = torch.randn(2, 1, 32)
        q_pos1 = rot.position_range(6, 1, torch.device("cpu"))
        k_pos1 = rot.position_range(0, 6, torch.device("cpu"))
        cos1, sin1 = rot.forward_range(6, 1, torch.device("cpu"))
        step_out, _ = attn(
            h_next, cos1, sin1, q_pos1, k_pos1, past_kv=present, use_cache=True, contiguous_causal=True
        )
        # Compare against full 7-token sequence
        h_all = torch.cat([h, h_next], dim=1)
        q_pos_all = rot.position_range(0, 7, torch.device("cpu"))
        cos_all, sin_all = rot.forward_range(0, 7, torch.device("cpu"))
        all_out, _ = attn(h_all, cos_all, sin_all, q_pos_all, k_pos, use_cache=False)
    assert torch.allclose(full_out, all_out[:, :6], atol=1e-5)
    assert torch.allclose(step_out, all_out[:, 6:7], atol=1e-5)


def test_mars_muon_optimizer_and_outlier_mixed_int4_kv_superior_fidelity():
    cfg = mc.MiMoMixConfig(vocab_size=32, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, use_ufwm=True)
    model = mc.MiMoMixModel(cfg)
    groups, counts = mo.build_muon_hybrid_param_groups(
        model, lr=1e-3, polar_express=True, mars_gamma=0.25
    )
    assert counts["muon_tensors"] > 0
    assert model.ufwm is not None
    assert not mo.is_muon_eligible_parameter("ufwm.layer_gates", model.ufwm.layer_gates)
    assert not mo.is_muon_eligible_parameter("ufwm.layer_role_emb", model.ufwm.layer_role_emb)
    opt = mo.MuonHybridOptimizer(groups, lr=1e-3, polar_express=True, mars_gamma=0.25)

    ids = torch.randint(0, 32, (2, 8))
    labels = torch.randint(0, 32, (2, 8))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        out = model(ids, labels=labels)
        assert out.loss is not None
        out.loss.backward()
        opt.step()
    tele = opt.optimizer_telemetry()
    assert tele["mars_active"] is True
    assert tele["last_mars_vr_ratio"] > 0.0
    assert tele["polar_express_active"] is True

    # Heavy-tailed KV cache with massive RoPE/sink channel outliers + Engram int64 tail
    torch.manual_seed(407)
    k_raw = torch.randn(1, 2, 24, 32) * 0.25
    v_raw = torch.randn(1, 2, 24, 32) * 0.25
    # Inject 2 large outlier channels (15x scale)
    k_raw[..., 3] *= 15.0
    k_raw[..., 19] *= 12.0
    v_raw[..., 7] *= 14.0
    engram_tail = (torch.randint(0, 100, (1, 4, 1), dtype=torch.long), torch.zeros((1, 4, 0), dtype=torch.long))
    past = [(k_raw, v_raw), engram_tail]

    rep_uniform = md.kv_cache_int4_memory_report(past, group_size=16)
    rep_outlier = md.kv_cache_outlier_memory_report(past, group_size=16, outlier_channels=2)

    assert rep_outlier["compression_ratio"] > 3.5
    assert rep_outlier["mean_cosine_similarity"] > rep_uniform["mean_cosine_similarity"]
    assert rep_outlier["max_abs_error"] < rep_uniform["max_abs_error"] * 0.35

    restored = md.dequantize_past_kv_mixed_outlier(
        md.quantize_past_kv_mixed_outlier(past, group_size=16, outlier_channels=2)
    )
    assert restored is not None
    assert torch.equal(restored[1][0], engram_tail[0])


# ---------------------------------------------------------------------------
# 5. Ten New Zero-Eval Exact Solvers (Positive + Fail-Closed)
# ---------------------------------------------------------------------------


def test_ten_new_v107_exact_solvers_and_scf_verification_gate():
    gate = SCFVerificationGate()
    cases = [
        (
            "Nernst potential z = 1, T = 310, out = 5, in = 140",
            "neuroscience.nernst_potential",
            "-89.0156",
        ),
        (
            "GHK potential T = 310, pK = 1, K_out = 5, K_in = 140, pNa = 0.04, Na_out = 145, Na_in = 12, pCl = 0.45, Cl_out = 110, Cl_in = 4",
            "neuroscience.ghk_potential",
            "-72.4803",
        ),
        (
            "Cable constants rm = 4000, ri = 1000, cm = 0.001",
            "neuroscience.cable_constants",
            "lambda=2, tau=4",
        ),
        (
            "Michaelis-Menten Vmax = 120, Km = 4, S = 12",
            "biochemistry.michaelis_menten",
            "90",
        ),
        (
            "Henderson-Hasselbalch pKa = 4.76, base = 0.2, acid = 0.02",
            "chemistry.henderson_hasselbalch",
            "5.76",
        ),
        (
            "Shannon-Hartley bandwidth = 1000, snr = 15",
            "infotheory.shannon_hartley",
            "4000",
        ),
        (
            "M/M/1 queue lambda = 3, mu = 5",
            "cs.mm1_queue",
            "1.5",
        ),
        (
            "Markov stationary distribution of [[0.7, 0.3], [0.4, 0.6]]",
            "math.markov_stationary",
            "[0.5714, 0.4286]",
        ),
        (
            "Quadratic form x = [1, 2], A = [[3, 1], [1, 2]]",
            "linalg.quadratic_form",
            "15",
        ),
        (
            "Put-call parity S = 100, K = 100, r = 0, T = 1, P = 6.5",
            "finance.put_call_parity",
            "6.5",
        ),
    ]
    for query, formula_id, expected in cases:
        s_res = ns.solve_problem(query)
        assert s_res.solved is True, query
        assert s_res.formula_id == formula_id
        assert s_res.display_answer == expected
        assert s_res.receipt is not None and len(s_res.receipt.receipt_sha256) == 64

        v_res = gate.verify(query=query)
        assert v_res["verified"] is True, query
        assert v_res["verified_answer"] == s_res.display_answer

    # Fail-closed domain guards on invalid/unstable inputs
    invalid_queries = [
        "Nernst potential z = 0, T = 310, out = 5, in = 140",
        "Nernst potential z = 1, T = 310, out = -5, in = 140",
        "GHK potential T = 310, pK = -1, K_out = 5, K_in = 140, pNa = 0.04, Na_out = 145, Na_in = 12",
        "Cable constants rm = -4000, ri = 1000, cm = 0.001",
        "Michaelis-Menten Vmax = 120, Km = 0, S = 12",
        "Henderson-Hasselbalch pKa = 4.76, base = 0, acid = 0.02",
        "Shannon-Hartley bandwidth = 1000, snr = -2",
        "M/M/1 queue lambda = 6, mu = 5",  # unstable queue lambda >= mu!
        "Markov stationary distribution of [[0.8, 0.3], [0.4, 0.6]]",  # row sum != 1
        "Quadratic form x = [1, 2, 3], A = [[3, 1], [1, 2]]",  # dim mismatch
    ]
    for bad_q in invalid_queries:
        assert ns.solve_problem(bad_q).solved is False, bad_q


# ---------------------------------------------------------------------------
# 6. Five v107 Analytical Engines, API Routes, CLI, Trainers & Studio UI
# ---------------------------------------------------------------------------


def test_five_v107_engines_receipts_and_fail_closed_validation():
    prof = v107.UniversalFastWeightProfiler().profile(
        hidden_size=128, n_layers=6, fast_dim=24, n_banks=4, top_k=2, seq_len=512
    )
    assert prof["answer_authority"] is False
    assert prof["parameter_sharing_ratio"] > 3.0
    assert len(prof["receipt"]["receipt_sha256"]) == 64
    with pytest.raises(ValueError):
        v107.UniversalFastWeightProfiler().profile(mode="bad_mode")

    plast = v107.ConnectomePlasticityOrchestrator().orchestrate(
        edge_magnitudes=[0.1, 0.2, 0.5, 0.7],
        pre_rates=[0.8, 0.4, 0.9, 0.6],
        post_rates=[0.7, 0.5, 0.8, 0.5],
        dale_signs=[1, -1, 1, -1],
    )
    assert plast["answer_authority"] is False
    assert plast["dale_law_preserved"] is True
    assert plast["nascent_count"] == 2 and plast["mature_spine_apparatus_count"] == 2
    with pytest.raises(ValueError):
        v107.ConnectomePlasticityOrchestrator().orchestrate(
            edge_magnitudes=[-0.1], pre_rates=[0.5], post_rates=[0.5]
        )

    assay = v107.SyntheticNeuronFoundryAssay().assay(
        stimulus_series=[0.1, 0.8, -0.4, 1.1, 0.2, -0.5, 0.9], foundry_dim=32, metabolic_budget=40.0
    )
    assert assay["answer_authority"] is False
    assert len(assay["family_profiles"]) == 6
    assert sum(assay["recommended_mixture"].values()) == pytest.approx(1.0, abs=1e-4)
    with pytest.raises(ValueError):
        v107.SyntheticNeuronFoundryAssay().assay(stimulus_series=[1.0], foundry_dim=32)

    anti = v107.AntipodeFailureDistiller().distill(
        records=[
            {"category": "easy", "loss": 0.1, "entropy": 0.1, "failed": False},
            {"category": "hard", "loss": 2.2, "entropy": 1.4, "failed": True},
        ]
    )
    assert anti["answer_authority"] is False
    assert anti["primary_antipode_category"] == "hard"
    assert anti["curriculum_weights"]["hard"] > anti["curriculum_weights"]["easy"]
    with pytest.raises(ValueError):
        v107.AntipodeFailureDistiller().distill(records=[])

    sym = v107.SymbiosisContinualRetentionAuditor().audit(
        anchor_weights=[0.5, -0.4, 0.9],
        current_weights=[0.7, -0.6, 0.95],
        fisher_importances=[5.0, 2.0, 8.0],
        old_task_loss_before=0.40,
        old_task_loss_after=0.42,
        new_task_loss_before=1.20,
        new_task_loss_after=0.65,
    )
    assert sym["answer_authority"] is False
    assert sym["retention_grade"] == "stable_symbiotic"
    assert sym["mean_l1_drift_after_consolidation"] < sym["mean_l1_drift_before"]
    with pytest.raises(ValueError):
        v107.SymbiosisContinualRetentionAuditor().audit(
            anchor_weights=[0.5],
            current_weights=[0.5],
            fisher_importances=[-1.0],
            old_task_loss_before=0.4,
            old_task_loss_after=0.4,
            new_task_loss_before=1.0,
            new_task_loss_after=0.5,
        )


def test_v107_api_endpoints_cli_trainers_chat_and_studio(capsys):
    from fastapi.testclient import TestClient

    app = napi.create_app()
    client = TestClient(app)

    sig = client.get("/v1/signals")
    assert sig.status_code == 200
    assert sig.json()["v107_universal_morphogenesis"]["universal_fast_weight_memory_uttt"] is True

    for path in (
        "/v1/v107/ufwm_profile",
        "/v1/v107/plasticity_orchestrate",
        "/v1/v107/neuron_assay",
        "/v1/v107/antipode_distill",
        "/v1/v107/symbiosis_retention",
    ):
        resp = client.post(path, json={})
        assert resp.status_code == 200, path
        body = resp.json()
        assert body["answer_authority"] is False
        assert len(body["receipt"]["receipt_sha256"]) == 64

    # 422 fail-closed validation on invalid inputs
    assert client.post("/v1/v107/ufwm_profile", json={"mode": "invalid"}).status_code == 422
    assert client.post("/v1/v107/neuron_assay", json={"metabolic_budget": -5.0}).status_code == 422

    # SCF CLI commands: frontier, diagnose (14/14), perf
    assert scf_cli.main(["frontier", "--json"]) == 0
    f_json = json.loads(capsys.readouterr().out)
    assert f_json["evolution_v107_version"] == v107.FRONTIER_V107_VERSION
    assert f_json["ufwm"]["enabled"] is True
    assert f_json["neuron_foundry"]["enabled"] is True

    assert scf_cli.main(["diagnose", "--json"]) == 0
    d_json = json.loads(capsys.readouterr().out)
    assert d_json["all_probes_passed"] is True
    assert d_json["verifier_probes_passed"] == 14

    assert scf_cli.main(["perf", "--json"]) == 0
    p_json = json.loads(capsys.readouterr().out)
    assert p_json["evolution_v107_version"] == v107.FRONTIER_V107_VERSION
    assert p_json["kv_outlier_report"]["compression_ratio"] > 1.0

    # Trainer CLI flag -> MiMoMixConfig roundtrip + warm-start prefix registration
    args = train_talk.build_parser().parse_args(
        [
            "--use_ufwm",
            "--ufwm_dim",
            "16",
            "--ufwm_banks",
            "4",
            "--ufwm_top_k",
            "2",
            "--use_neuron_foundry",
            "--neuron_foundry_dim",
            "16",
            "--cns_astrocyte_gate",
            "--cns_retrograde_ephaptic",
            "--muon_mars_gamma",
            "0.2",
        ]
    )
    built_cfg = train_talk.build_config(args, vocab_size=128)
    assert built_cfg.use_ufwm is True and built_cfg.ufwm_dim == 16
    assert built_cfg.use_neuron_foundry is True and built_cfg.neuron_foundry_dim == 16
    assert built_cfg.cns_astrocyte_gate is True and built_cfg.cns_retrograde_ephaptic is True
    prefixes = train_gen.new_parameter_prefixes(built_cfg)
    assert "ufwm." in prefixes and "neuron_foundry." in prefixes

    # Chat server telemetry snapshot & Studio UI section 10 check
    m_chat = mc.MiMoMixModel(built_cfg).eval()
    with torch.no_grad():
        m_chat(torch.randint(0, 128, (1, 4)))
    snap = chat_srv._architecture_telemetry_snapshot(m_chat)
    assert snap is not None
    assert snap["ufwm"]["enabled"] is True
    assert snap["neuron_foundry"]["enabled"] is True

    studio_html = (ROOT / "web_static" / "nexus_studio.html").read_text(encoding="utf-8")
    for token in (
        "10. v107 Universal Fast-Weight",
        "runV107UfwmProfile",
        "runV107PlasticityOrchestrate",
        "runV107NeuronAssay",
        "runV107AntipodeDistill",
        "runV107SymbiosisRetention",
    ):
        assert token in studio_html, token
