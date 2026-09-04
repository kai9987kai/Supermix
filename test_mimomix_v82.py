"""v82 core-model tests: four confirmed bug fixes, config hygiene, cited features.

Every test here either pins a bug that was measured on the pre-v82 code, or
pins a *default-off* research feature to the behaviour its flag promises. None
of the research features has been shown to improve this model -- these tests
check that the mechanism does what it says, not that it helps. See
docs/V79_OMNI_FRONTIER.md and V81_WHAT_THE_MODEL_CAN_LEARN.md for the house
rule about not claiming unmeasured gains.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent / "source"))

import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def tiny(**overrides):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        n_routed_experts=4,
        moe_intermediate_size=16,
        n_dense_layers=1,
        n_mtp_layers=0,
        use_thinking_core=False,
        rope_scaling="none",
        hybrid_ratio=0,
    )
    base.update(overrides)
    return mc.MiMoMixConfig(**base)


def decode_parity(config, seq_len: int = 8, seed: int = 5) -> float:
    """max |dlogit| between one full forward and its incremental replay."""

    torch.manual_seed(seed)
    model = mc.MiMoMixModel(config)
    model.eval()
    ids = torch.randint(0, config.vocab_size, (1, seq_len))
    with torch.no_grad():
        full = model(ids)
        past = None
        steps = []
        for i in range(seq_len):
            out = model(ids[:, i : i + 1], past_key_values=past, use_cache=True, past_length=i)
            past = out.past_key_values
            steps.append(out.logits)
    return float((full.logits - torch.cat(steps, dim=1)).abs().max())


# ---------------------------------------------------------------------------
# BUG A -- MLA partial rope was not a rotation
# ---------------------------------------------------------------------------


def test_apply_rotary_preserves_norm_for_a_narrow_table():
    """A rotation cannot change a vector's norm. The pre-v82 MLA path did.

    Measured on the pre-v82 code at the v80-ish shape: norm drift 2.2866.
    """

    cfg = tiny(head_dim=16, hidden_size=64, n_heads=4)
    rot = mc.RotaryEmbedding(cfg, kind="global", rotary_dim=8)
    x = torch.randn(1, 1, 6, 8)
    cos, sin = rot(torch.arange(6))
    y = mc.apply_rotary(x, cos, sin)
    drift = (y.norm(dim=-1) - x.norm(dim=-1)).abs().max()
    assert float(drift) < 1e-5, float(drift)


def test_slicing_a_wide_table_is_still_not_a_rotation():
    """Pin the *old* failure so the fix cannot silently regress to it."""

    cfg = tiny(head_dim=16, hidden_size=64, n_heads=4)
    trunk = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = trunk(torch.arange(6))
    x = torch.randn(1, 1, 6, 8)
    y = mc.apply_rotary(x, cos[:, :8], sin[:, :8])
    drift = float((y.norm(dim=-1) - x.norm(dim=-1)).abs().max())
    assert drift > 1e-3, "slicing a wider table used to break the norm; it still should"


def test_mla_pe_rotation_is_relative():
    """The same relative offset must score the same at any absolute position.

    Measured before the fix: +3.9137 at (5, 2) but +7.5882 at (15, 12).
    """

    cfg = tiny(hidden_size=64, n_heads=4, head_dim=16, use_mla=True,
               mla_latent_dim=24, mla_pe_dim=8)
    attn = mc.MultiLatentAttention(cfg, 0, "global")
    torch.manual_seed(3)
    q = torch.randn(1, 1, 1, cfg.mla_pe_dim)
    k = torch.randn(1, 1, 1, cfg.mla_pe_dim)

    def score(a: int, b: int) -> float:
        cos, sin = attn.pe_rotary(torch.tensor([a, b]))
        qa = mc.apply_rotary(q, cos[0:1], sin[0:1])
        kb = mc.apply_rotary(k, cos[1:2], sin[1:2])
        return float((qa * kb).sum())

    near, far = score(5, 2), score(15, 12)
    assert abs(near - far) < 1e-4, (near, far)


def test_mla_owns_a_pe_sized_rotary_table():
    cfg = tiny(hidden_size=64, n_heads=4, head_dim=16, use_mla=True,
               mla_latent_dim=24, mla_pe_dim=8)
    attn = mc.MultiLatentAttention(cfg, 0, "global")
    assert attn.pe_rotary.rotary_dim == cfg.mla_pe_dim
    cos, _ = attn.pe_rotary(torch.arange(4))
    assert cos.shape == (4, cfg.mla_pe_dim)


def test_full_rotary_dim_is_unchanged():
    """rotary_dim=None must reproduce the old full rotation exactly."""

    cfg = tiny(head_dim=8)
    rot = mc.RotaryEmbedding(cfg, kind="global")
    assert rot.rotary_dim == cfg.head_dim
    x = torch.randn(1, 2, 5, cfg.head_dim)
    cos, sin = rot(torch.arange(5))
    manual = x * cos.unsqueeze(0).unsqueeze(0) + mc._rotate_half(x) * sin.unsqueeze(0).unsqueeze(0)
    assert torch.equal(mc.apply_rotary(x, cos, sin), manual)


def test_partial_rotary_passes_the_tail_through():
    cfg = tiny(head_dim=8, rotary_dim=4)
    rot = mc.RotaryEmbedding(cfg, kind="global")
    assert rot.rotary_dim == 4
    x = torch.randn(1, 1, 3, 8)
    cos, sin = rot(torch.arange(3))
    y = mc.apply_rotary(x, cos, sin)
    assert torch.equal(y[..., 4:], x[..., 4:])
    assert not torch.equal(y[..., :4], x[..., :4])


def test_apply_rotary_rejects_an_oversized_table():
    x = torch.randn(1, 1, 2, 4)
    cos = torch.ones(2, 8)
    with pytest.raises(ValueError, match="exceeds the head width"):
        mc.apply_rotary(x, cos, cos)


# ---------------------------------------------------------------------------
# BUG B -- MLA broke speculative decoding
# ---------------------------------------------------------------------------


def test_sequence_axis_is_chosen_by_rank():
    assert md._sequence_axis(torch.zeros(1, 2, 3, 4)) == 2
    assert md._sequence_axis(torch.zeros(1, 3, 4)) == 1
    with pytest.raises(ValueError):
        md._sequence_axis(torch.zeros(3, 4))


def test_trim_past_trims_a_three_dimensional_latent_cache_on_time():
    c_kv = torch.randn(1, 6, 16)      # (B, T, latent) -- MLA
    k_pe = torch.randn(1, 1, 6, 4)    # (B, 1, T, pe)
    trimmed = md.trim_past([(c_kv, k_pe)], 2)
    assert trimmed[0][0].shape == (1, 4, 16), "latent width must survive"
    assert trimmed[0][1].shape == (1, 1, 4, 4)


def test_trim_past_still_trims_a_four_dimensional_cache():
    keys = torch.randn(2, 3, 6, 8)
    values = torch.randn(2, 3, 6, 8)
    trimmed = md.trim_past([(keys, values)], 2)
    assert trimmed[0][0].shape == (2, 3, 4, 8)


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(dict(hybrid_ratio=0), id="all-global"),
        pytest.param(dict(hybrid_ratio=1, sliding_window=4), id="hybrid"),
        pytest.param(
            dict(hybrid_ratio=1, sliding_window=4, mla_global_only=False), id="mla-everywhere"
        ),
    ],
)
def test_speculative_decoding_matches_greedy_under_mla(layout):
    torch.manual_seed(4)
    cfg = tiny(
        n_layers=4,
        head_dim=8,
        use_mla=True,
        mla_latent_dim=16,
        mla_pe_dim=4,
        n_mtp_layers=2,
        **layout,
    )
    model = mc.MiMoMixModel(cfg)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    report = md.assert_greedy_equivalence(model, ids, max_new_tokens=10)
    assert report["tokens"] == 10


# ---------------------------------------------------------------------------
# BUG C -- Mixture of Depths
# ---------------------------------------------------------------------------


def test_mod_decode_parity_matches_the_non_mod_baseline():
    """Measured pre-v82: 8.1e-2 for MoD against a 1.2e-7 baseline."""

    baseline = decode_parity(tiny())
    with_mod = decode_parity(tiny(use_mod=True, mod_capacity_ratio=0.5))
    assert baseline < 1e-5
    assert with_mod < max(1e-5, baseline * 10.0), (baseline, with_mod)


def test_disabling_the_causal_predictor_restores_the_old_mismatch():
    broken = decode_parity(tiny(use_mod=True, mod_capacity_ratio=0.5, mod_causal_predictor=False))
    assert broken > 1e-3, broken


def test_mod_router_selection_is_causal_in_eval():
    """Changing a later token must not change an earlier token's selection.

    Measured pre-v82 on the top-k router: changing token 7 of 8 moved
    positions 0-3 of the block output by 0.519.
    """

    torch.manual_seed(0)
    router = mc.MixtureOfDepthsRouter(16, 0.5, causal_predictor=True)
    router.eval()
    x = torch.randn(1, 8, 16)
    mask_a, _ = router.gate_mask(x)
    x2 = x.clone()
    x2[0, 7] = torch.randn(16) * 5.0
    mask_b, _ = router.gate_mask(x2)
    assert torch.equal(mask_a[:, :7], mask_b[:, :7])
    assert router.last_selection_mode == "predictor"


def test_mod_training_uses_topk_and_trains_the_predictor():
    torch.manual_seed(0)
    router = mc.MixtureOfDepthsRouter(16, 0.5, causal_predictor=True, predictor_loss_coef=0.5)
    router.train()
    x = torch.randn(2, 8, 16)
    mask, _ = router.gate_mask(x)
    assert router.last_selection_mode == "topk"
    assert int(mask.sum(dim=-1)[0]) == 4
    loss = router.aux_loss()
    assert loss.requires_grad and float(loss.detach()) > 0.0


def test_mod_predictor_loss_reaches_the_model_aux_loss():
    torch.manual_seed(0)
    cfg = tiny(use_mod=True, mod_capacity_ratio=0.5, mod_predictor_loss_coef=1.0)
    model = mc.MiMoMixModel(cfg)
    model.train()
    ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(ids, labels=ids)
    assert out.aux_loss is not None and float(out.aux_loss.detach()) > 0.0

    zero = tiny(use_mod=True, mod_capacity_ratio=0.5, mod_predictor_loss_coef=0.0,
                router_z_loss_coef=0.0, router_balance_loss_coef=0.0)
    torch.manual_seed(0)
    quiet = mc.MiMoMixModel(zero)
    quiet.train()
    assert float(quiet(ids, labels=ids).aux_loss.detach()) == 0.0


def test_mod_excludes_gated_out_tokens_from_expert_load():
    torch.manual_seed(0)
    cfg = tiny(use_mod=True, mod_capacity_ratio=0.5)
    block = mc.MiMoMixBlock(cfg, layer_index=1, kind="global")
    block.train()
    assert block.is_moe
    x = torch.randn(2, 8, cfg.hidden_size)
    positions = torch.arange(8)
    cos, sin = mc.RotaryEmbedding(cfg, kind="global")(positions)
    block(x, cos, sin, positions, positions.new_empty((0,)))
    load = block.mlp.last_expert_load
    # top_k experts are picked for each of the 4 selected tokens of each row;
    # the load must sum to top_k, not to top_k scaled by all 8 tokens.
    assert float(load.sum()) == pytest.approx(float(cfg.moe_top_k), abs=1e-5)


def test_mod_telemetry_does_not_claim_a_compute_saving():
    cfg = tiny(use_mod=True, mod_capacity_ratio=0.5)
    model = mc.MiMoMixModel(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (1, 8)))
    assert out.telemetry["mod_compute_saved"] is False
    assert "no FLOPs are skipped" in out.telemetry["mod_note"]
    assert "mod_selection_mode" in out.telemetry


def test_no_docstring_claims_mod_saves_compute():
    text = (Path(__file__).parent / "source" / "mimomix_core.py").read_text(encoding="utf-8")
    lowered = text.lower()
    for phrase in ("skip the block", "saves compute", "compute saving for"):
        if phrase in lowered:
            raise AssertionError(f"unsupported MoD compute claim in mimomix_core.py: {phrase!r}")


# ---------------------------------------------------------------------------
# BUG D -- _init_weights clobbered the thinking core
# ---------------------------------------------------------------------------


def test_thinking_core_keeps_its_deliberate_init_after_construction():
    """Measured pre-v82: quality_head.abs().sum() 1.2718, to_residual std 0.0200."""

    torch.manual_seed(0)
    model = mc.build_mimomix(vocab_size=64, hidden_size=32, n_layers=2, use_thinking_core=True)
    core = model.thinking_core
    assert float(core.quality_head.weight.abs().sum()) == 0.0
    assert float(core.quality_head.bias.abs().sum()) == 0.0
    std = float(core.to_residual.weight.detach().std())
    assert 0.005 < std < 0.015, std


def test_restoring_the_init_does_not_wake_the_core():
    """The gate still starts closed. See docs/V59_MECHANISM_CAUSALITY.md."""

    model = mc.build_mimomix(vocab_size=64, hidden_size=32, n_layers=2, use_thinking_core=True)
    assert float(model.thinking_core.residual_scale.detach()) == 0.0
    warm = mc.build_mimomix(
        vocab_size=64, hidden_size=32, n_layers=2, thinking_residual_init=0.1
    )
    assert float(warm.thinking_core.residual_scale.detach()) == pytest.approx(0.1)


def test_restore_hook_is_counted():
    model = mc.build_mimomix(vocab_size=64, hidden_size=32, n_layers=2, use_thinking_core=True)
    assert model.restored_special_inits >= 1


# ---------------------------------------------------------------------------
# PART 2 -- config hygiene
# ---------------------------------------------------------------------------


def test_mla_global_only_is_a_real_field():
    assert "mla_global_only" in mc.MiMoMixConfig.field_names()
    assert mc.MiMoMixConfig().mla_global_only is True
    cfg = tiny(use_mla=True, head_dim=8, mla_pe_dim=4, mla_global_only=False,
               hybrid_ratio=1, sliding_window=4)
    model = mc.MiMoMixModel(cfg)
    kinds = {type(layer.self_attn).__name__ for layer in model.layers}
    assert kinds == {"MultiLatentAttention"}


def test_mla_global_only_true_leaves_swa_layers_on_gqa():
    cfg = tiny(use_mla=True, head_dim=8, mla_pe_dim=4, hybrid_ratio=1, sliding_window=4,
               n_layers=4)
    model = mc.MiMoMixModel(cfg)
    kinds = [type(layer.self_attn).__name__ for layer in model.layers]
    assert "HybridAttention" in kinds and "MultiLatentAttention" in kinds


def test_mla_pe_dim_must_fit_in_the_head():
    with pytest.raises(ValueError, match="mla_pe_dim"):
        mc.MiMoMixConfig(hidden_size=64, n_heads=4, head_dim=16, use_mla=True, mla_pe_dim=32)
    with pytest.raises(ValueError, match="mla_pe_dim"):
        mc.MiMoMixConfig(hidden_size=64, n_heads=4, head_dim=16, use_mla=True, mla_pe_dim=7)


def test_differential_and_mla_together_are_refused():
    with pytest.raises(ValueError, match="mutually exclusive"):
        mc.MiMoMixConfig(use_mla=True, use_differential_attention=True)


def test_rotary_dim_validation():
    with pytest.raises(ValueError, match="rotary_dim"):
        mc.MiMoMixConfig(hidden_size=64, n_heads=4, head_dim=16, rotary_dim=32)
    with pytest.raises(ValueError, match="rotary_dim"):
        mc.MiMoMixConfig(hidden_size=64, n_heads=4, head_dim=16, rotary_dim=5)


def test_from_dict_reports_unknown_keys_instead_of_raising():
    payload = dict(mc.MiMoMixConfig(hidden_size=64, n_heads=4).to_dict())
    payload["a_knob_from_the_future"] = 3
    with pytest.raises(TypeError):
        mc.MiMoMixConfig(**payload)
    with pytest.warns(UserWarning, match="a_knob_from_the_future"):
        cfg = mc.MiMoMixConfig.from_dict(payload)
    assert cfg.unknown_keys == ("a_knob_from_the_future",)
    assert cfg.hidden_size == 64


def test_from_dict_round_trips_to_dict():
    cfg = mc.MiMoMixConfig(hidden_size=64, n_heads=4, global_layers=(1, 3), n_layers=4)
    again = mc.MiMoMixConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()
    assert again.unknown_keys == ()


def test_global_layers_validation():
    with pytest.raises(ValueError, match="global_layers"):
        mc.MiMoMixConfig(n_layers=4, global_layers=(9,))


# ---------------------------------------------------------------------------
# PART 3 -- cited research features, all default-off
# ---------------------------------------------------------------------------


def test_every_new_flag_defaults_to_todays_behaviour():
    cfg = mc.MiMoMixConfig()
    assert cfg.qk_norm is False
    assert cfg.attention_output_gate is False
    assert cfg.attention_sink_kinds == "all"
    assert cfg.rotary_dim is None
    assert cfg.global_layers is None
    assert cfg.moe_balance_scope == "batch"
    assert cfg.differential_noise_ratio == 1
    assert cfg.mod_causal_predictor is True
    assert cfg.mla_global_only is True
    assert cfg.differential_output_norm is True


@pytest.mark.parametrize("cls_flag", ["plain", "differential", "mla"])
def test_qk_norm_adds_norms_to_every_attention_class(cls_flag):
    over = {"plain": {}, "differential": dict(use_differential_attention=True),
            "mla": dict(use_mla=True, head_dim=8, mla_pe_dim=4)}[cls_flag]
    off = mc.MiMoMixModel(tiny(**over))
    on = mc.MiMoMixModel(tiny(qk_norm=True, **over))
    assert all(layer.self_attn.q_norm is None for layer in off.layers)
    assert all(layer.self_attn.q_norm is not None for layer in on.layers)


def test_qk_norm_changes_the_output_and_keeps_decode_parity():
    torch.manual_seed(0)
    cfg_off, cfg_on = tiny(), tiny(qk_norm=True)
    torch.manual_seed(0)
    off = mc.MiMoMixModel(cfg_off)
    torch.manual_seed(0)
    on = mc.MiMoMixModel(cfg_on)
    ids = torch.randint(0, 64, (1, 8))
    off.eval()
    on.eval()
    with torch.no_grad():
        assert not torch.allclose(off(ids).logits, on(ids).logits)
    assert decode_parity(cfg_on) < 1e-5


def test_output_gate_starts_near_identity():
    torch.manual_seed(0)
    cfg = tiny(attention_output_gate=True)
    model = mc.MiMoMixModel(cfg)
    gate = model.layers[0].self_attn.out_gate
    # weight zeroed + bias +4 => constant sigmoid(4) = 0.982, restored *after*
    # the model's blanket _init_weights sweep.
    assert float(gate.proj.weight.abs().sum()) == 0.0
    assert float(gate.proj.bias.min()) == pytest.approx(4.0)
    x = torch.randn(1, 5, cfg.hidden_size)
    out = gate(torch.ones(1, 5, cfg.n_heads * cfg.head_dim), x)
    assert float(out.min()) == pytest.approx(0.98201, abs=1e-4)


def test_output_gate_keeps_decode_parity():
    assert decode_parity(tiny(attention_output_gate=True)) < 1e-5


def test_attention_sink_kinds_swa_drops_the_global_sink():
    cfg = tiny(n_layers=4, hybrid_ratio=1, sliding_window=4, attention_sink_kinds="swa")
    model = mc.MiMoMixModel(cfg)
    for layer, kind in zip(model.layers, model.layout):
        has_sink = layer.self_attn.sink is not None
        assert has_sink == (kind == "swa"), (kind, has_sink)


def test_attention_sink_kinds_all_is_unchanged():
    cfg = tiny(n_layers=4, hybrid_ratio=1, sliding_window=4)
    model = mc.MiMoMixModel(cfg)
    assert all(layer.self_attn.sink is not None for layer in model.layers)


def test_global_layers_overrides_the_ratio():
    assert mc.attention_layout(6, 5, False, (0, 2)) == (
        "global", "swa", "global", "swa", "swa", "swa"
    )
    assert mc.attention_layout(6, 5, True, (0, 2))[-1] == "global"
    # None keeps the old behaviour byte for byte
    assert mc.attention_layout(12, 5, False) == mc.attention_layout(12, 5, False, None)


def test_global_layers_reaches_the_model_layout():
    cfg = tiny(n_layers=4, hybrid_ratio=5, sliding_window=4, global_layers=(1,),
               final_layer_global=False)
    model = mc.MiMoMixModel(cfg)
    assert model.layout == ("swa", "global", "swa", "swa")


def test_differential_output_norm_changes_the_result_and_can_be_switched_off():
    torch.manual_seed(0)
    on = mc.MiMoMixModel(tiny(use_differential_attention=True))
    torch.manual_seed(0)
    off = mc.MiMoMixModel(tiny(use_differential_attention=True, differential_output_norm=False))
    assert on.layers[0].self_attn.output_norm is not None
    assert off.layers[0].self_attn.output_norm is None
    ids = torch.randint(0, 64, (1, 8))
    on.eval()
    off.eval()
    with torch.no_grad():
        assert not torch.allclose(on(ids).logits, off(ids).logits)


def test_differential_noise_ratio_shares_noise_heads():
    cfg = tiny(use_differential_attention=True, n_heads=4, differential_noise_ratio=2)
    attn = mc.DifferentialHybridAttention(cfg, 0, "global")
    assert attn.n_noise_heads == 2
    assert attn.sink2.shape == (2,)
    x = torch.randn(1, 6, cfg.hidden_size)
    positions = torch.arange(6)
    cos, sin = mc.RotaryEmbedding(cfg, kind="global")(positions)
    out, _ = attn(x, cos, sin, positions, positions.new_empty((0,)))
    assert out.shape == (1, 6, cfg.hidden_size)


def test_differential_noise_ratio_one_keeps_the_old_shapes():
    cfg = tiny(use_differential_attention=True, n_heads=4)
    attn = mc.DifferentialHybridAttention(cfg, 0, "global")
    assert attn.n_noise_heads == cfg.n_heads
    assert attn.sink2.shape == (cfg.n_heads,)


def test_differential_noise_ratio_must_divide_heads():
    with pytest.raises(ValueError, match="differential_noise_ratio"):
        mc.MiMoMixConfig(n_heads=4, use_differential_attention=True, differential_noise_ratio=3)


def test_moe_balance_scope_batch_is_the_default_and_unchanged():
    torch.manual_seed(0)
    cfg = tiny()
    assert cfg.moe_balance_scope == "batch"
    model = mc.MiMoMixModel(cfg)
    model.train()
    ids = torch.randint(0, 64, (2, 8))
    model(ids, labels=ids)
    moe = [m for m in model.modules() if isinstance(m, mc.SparseMoEFeedForward)][0]
    assert moe.balance_scope == "batch"
    assert float(moe.last_expert_load.sum()) == pytest.approx(float(cfg.moe_top_k), abs=1e-5)


def test_moe_balance_scope_sequence_differs_from_batch():
    """Batch-wise and sequence-wise balance agree only if every sequence routes
    the same way; on random data they must not."""

    torch.manual_seed(0)
    batch_cfg = tiny(moe_balance_scope="batch")
    torch.manual_seed(0)
    batch_model = mc.MiMoMixModel(batch_cfg)
    torch.manual_seed(0)
    seq_model = mc.MiMoMixModel(tiny(moe_balance_scope="sequence"))
    ids = torch.randint(0, 64, (4, 8))
    batch_model.train()
    seq_model.train()
    batch_model(ids, labels=ids)
    seq_model(ids, labels=ids)
    a = [m.last_router_balance_loss for m in batch_model.modules()
         if isinstance(m, mc.SparseMoEFeedForward)][0]
    b = [m.last_router_balance_loss for m in seq_model.modules()
         if isinstance(m, mc.SparseMoEFeedForward)][0]
    assert not torch.isclose(a, b), (float(a), float(b))


def test_moe_token_mask_excludes_tokens_from_the_balance_loss():
    torch.manual_seed(0)
    cfg = tiny()
    moe = mc.SparseMoEFeedForward(cfg)
    moe.train()
    x = torch.randn(2, 8, cfg.hidden_size)
    moe(x)
    unmasked = moe.last_router_balance_loss.clone()
    mask = torch.zeros(2, 8, dtype=torch.bool)
    mask[:, :4] = True
    moe(x, token_mask=mask)
    assert not torch.isclose(unmasked, moe.last_router_balance_loss)


# ---------------------------------------------------------------------------
# PART 4 -- regression guards
# ---------------------------------------------------------------------------


def test_default_model_decode_parity_is_unchanged():
    assert decode_parity(tiny()) < 1e-5


@pytest.mark.parametrize(
    "name,over",
    [
        ("baseline", {}),
        ("differential", dict(use_differential_attention=True)),
        ("mla", dict(use_mla=True, head_dim=8, mla_latent_dim=16, mla_pe_dim=4)),
        ("mod", dict(use_mod=True, mod_capacity_ratio=0.5)),
        ("qk_norm", dict(qk_norm=True)),
        ("gated", dict(attention_output_gate=True)),
        ("partial_rope", dict(rotary_dim=4, head_dim=8)),
        ("sink_swa", dict(attention_sink_kinds="swa", hybrid_ratio=1, sliding_window=4)),
    ],
)
def test_full_forward_matches_incremental_decode(name, over):
    delta = decode_parity(tiny(**over))
    assert delta < 1e-5, f"{name}: {delta}"


def test_v80_checkpoint_still_loads_strictly():
    path = Path(__file__).parent / "output" / "v80_omni" / "v80_omni.pt"
    if not path.exists():
        pytest.skip("v80 checkpoint not present")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = mc.MiMoMixConfig.from_dict(payload["config"])
    assert config.unknown_keys == ()
    model = mc.MiMoMixModel(config)
    report = model.load_state_dict(payload["state_dict"], strict=True)
    assert list(report.missing_keys) == []
    assert list(report.unexpected_keys) == []


def test_new_flags_do_not_change_the_v80_state_dict_shape():
    """Every default-off flag must leave the parameter set alone."""

    base = tiny(use_thinking_core=True, n_mtp_layers=1)
    reference = set(mc.MiMoMixModel(base).state_dict().keys())
    assert reference == set(mc.MiMoMixModel(tiny(use_thinking_core=True, n_mtp_layers=1)).state_dict().keys())
    for flag in ("qk_norm", "attention_output_gate"):
        changed = set(
            mc.MiMoMixModel(
                tiny(use_thinking_core=True, n_mtp_layers=1, **{flag: True})
            ).state_dict().keys()
        )
        assert changed > reference, flag
