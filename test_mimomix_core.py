"""Tests for the MiMoMix v53 neural core.

These assert *mechanisms*, not quality: that the hybrid cache really is smaller,
that incremental decoding really matches a full forward, that the aux-loss-free
balancer really balances, and that the bias really cannot touch the forward
value. A passing suite proves integration and gradient flow. It does not prove
the model is good at anything.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_core as mc  # noqa: E402


def tiny_config(**overrides) -> mc.MiMoMixConfig:
    base = dict(
        vocab_size=48,
        hidden_size=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        sliding_window=4,
        hybrid_ratio=2,
        native_context=16,
        max_position_embeddings=64,
        n_routed_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        n_mtp_layers=2,
        thinking_cycles=2,
        thinking_max_cycles=6,
    )
    base.update(overrides)
    return mc.MiMoMixConfig(**base)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_rejects_incoherent_shapes():
    with pytest.raises(ValueError):
        mc.MiMoMixConfig(hidden_size=30, n_heads=4)
    with pytest.raises(ValueError):
        mc.MiMoMixConfig(n_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        mc.MiMoMixConfig(rope_scaling="magic")
    with pytest.raises(ValueError):
        mc.MiMoMixConfig(n_routed_experts=2, moe_top_k=4)
    with pytest.raises(ValueError):
        mc.MiMoMixConfig(sliding_window=0)


def test_config_is_json_safe():
    json.dumps(tiny_config().to_dict())


def test_context_scale():
    assert tiny_config(native_context=32, max_position_embeddings=256).context_scale == 8.0
    # never below 1: a target shorter than native does not "compress" rope
    assert tiny_config(native_context=256, max_position_embeddings=32).context_scale == 1.0


# ---------------------------------------------------------------------------
# Hybrid attention layout and cache
# ---------------------------------------------------------------------------


def test_attention_layout_matches_the_declared_ratio():
    # MiMo-V2-Flash: 5:1 local:global
    layout = mc.attention_layout(12, 5, final_layer_global=False)
    assert layout.count("global") == 2
    assert layout[5] == "global" and layout[11] == "global"
    assert all(k == "swa" for i, k in enumerate(layout) if (i + 1) % 6)

    # MiMo-V2.5-Pro: 6:1
    assert mc.attention_layout(14, 6, final_layer_global=False).count("global") == 2

    # ratio 0 means dense attention everywhere
    assert set(mc.attention_layout(4, 0)) == {"global"}


def test_final_layer_is_forced_global_when_requested():
    layout = mc.attention_layout(5, 5, final_layer_global=True)
    assert layout[-1] == "global"


def test_sliding_window_layers_hold_a_smaller_cache():
    model = mc.MiMoMixModel(tiny_config())
    model.eval()
    ids = torch.randint(0, 48, (1, 20))
    with torch.no_grad():
        out = model(ids, use_cache=True, past_length=0)
    spans = [entry[0].shape[2] for entry in out.past_key_values]
    for kind, span in zip(model.layout, spans):
        if kind == "swa":
            assert span == model.config.sliding_window
        else:
            assert span == 20
    # the whole point: hybrid holds strictly fewer entries than all-global
    assert sum(spans) < 20 * len(model.layout)


def test_cache_slack_is_honoured():
    model = mc.MiMoMixModel(tiny_config())
    model.eval()
    ids = torch.randint(0, 48, (1, 20))
    with torch.no_grad():
        out = model(ids, use_cache=True, cache_slack=3, past_length=0)
    for kind, entry in zip(model.layout, out.past_key_values):
        if kind == "swa":
            assert entry[0].shape[2] == model.config.sliding_window + 3


@pytest.mark.parametrize("rope_scaling", ["none", "ntk", "yarn"])
def test_incremental_decoding_matches_a_full_forward(rope_scaling):
    torch.manual_seed(0)
    model = mc.MiMoMixModel(tiny_config(rope_scaling=rope_scaling))
    model.eval()
    ids = torch.randint(0, 48, (2, 14))
    with torch.no_grad():
        full = model(ids, past_length=0)
        prefix = model(ids[:, :9], use_cache=True, past_length=0)
        rest = model(ids[:, 9:], past_key_values=prefix.past_key_values, use_cache=True, past_length=9)
    assert torch.allclose(rest.logits, full.logits[:, 9:], atol=1e-5)


def test_attention_mask_is_refused_alongside_a_cache():
    model = mc.MiMoMixModel(tiny_config())
    ids = torch.randint(0, 48, (1, 6))
    with torch.no_grad():
        prefix = model(ids, use_cache=True, past_length=0)
        with pytest.raises(ValueError, match="full-sequence"):
            model(
                ids,
                attention_mask=torch.ones(1, 6, dtype=torch.bool),
                past_key_values=prefix.past_key_values,
                past_length=6,
            )


def test_attention_sink_absorbs_probability_and_can_be_disabled():
    with_sink = mc.MiMoMixModel(tiny_config(attention_sink=True))
    without = mc.MiMoMixModel(tiny_config(attention_sink=False))
    ids = torch.randint(0, 48, (1, 8))
    with torch.no_grad():
        with_sink(ids, past_length=0)
        without(ids, past_length=0)
    assert with_sink.telemetry()["mean_sink_mass"] > 0.0
    assert without.telemetry()["mean_sink_mass"] == 0.0

    # A raised sink logit must take mass away from the real tokens.
    layer = with_sink.layers[0].self_attn
    with torch.no_grad():
        layer.sink.fill_(6.0)
        with_sink(ids, past_length=0)
    assert float(layer.last_sink_mass) > 0.9


def test_local_and_global_layers_use_decoupled_rope():
    """A sliding window cannot express a dependency longer than itself.

    Extending its frequencies is at best wasted, so local layers keep their own
    base and skip the context-extension policy unless explicitly told not to.
    """

    config = tiny_config(
        rope_scaling="yarn",
        rope_theta=640000.0,
        rope_local_theta=10000.0,
        native_context=16,
        max_position_embeddings=256,
    )
    model = mc.MiMoMixModel(config)

    assert model.rotary.kind == "global" and model.rotary_local.kind == "swa"
    assert model.rotary_local.effective_base == 10000.0
    assert model.rotary.effective_base == 640000.0
    # global carries the YaRN logit temperature; local does not
    assert model.rotary.attention_scaling > 1.0
    assert model.rotary_local.attention_scaling == 1.0
    assert not torch.allclose(model.rotary.inv_freq, model.rotary_local.inv_freq)

    for layer in model.layers:
        source = model.rotary_local if layer.kind == "swa" else model.rotary
        assert layer.self_attn._attention_scaling == source.attention_scaling
    # MTP depths are global layers and take the global policy
    for module in model.mtp_modules:
        assert module.block.self_attn._attention_scaling == model.rotary.attention_scaling


def test_local_rope_scaling_can_be_opted_into():
    scaled = mc.MiMoMixModel(
        tiny_config(rope_scaling="yarn", rope_scale_local=True,
                    native_context=16, max_position_embeddings=256)
    )
    assert scaled.rotary_local.attention_scaling > 1.0


def test_a_shared_rope_table_is_still_available():
    shared = mc.MiMoMixModel(tiny_config(rope_local_theta=None, rope_scale_local=True))
    assert torch.allclose(shared.rotary.inv_freq, shared.rotary_local.inv_freq)


def test_yarn_scaling_changes_frequencies_and_logit_temperature():
    plain = mc.RotaryEmbedding(tiny_config(rope_scaling="none"))
    yarn = mc.RotaryEmbedding(tiny_config(rope_scaling="yarn", native_context=16, max_position_embeddings=256))
    ntk = mc.RotaryEmbedding(tiny_config(rope_scaling="ntk", native_context=16, max_position_embeddings=256))

    assert plain.attention_scaling == 1.0
    assert yarn.attention_scaling == pytest.approx(0.1 * math.log(16.0) + 1.0)
    assert ntk.attention_scaling == 1.0
    assert ntk.effective_base > plain.effective_base

    # YaRN leaves the highest frequency alone and interpolates the lowest.
    assert yarn.inv_freq[0] == pytest.approx(float(plain.inv_freq[0]), rel=1e-6)
    assert float(yarn.inv_freq[-1]) < float(plain.inv_freq[-1])


# ---------------------------------------------------------------------------
# Sparse MoE
# ---------------------------------------------------------------------------


def test_router_bias_selects_but_never_weights():
    """The aux-loss-free bias must not appear in the forward value."""

    torch.manual_seed(0)
    moe = mc.SparseMoEFeedForward(tiny_config())
    moe.eval()
    x = torch.randn(6, 32)

    with torch.no_grad():
        baseline = moe(x)
        scores = torch.softmax(moe.gate(x), dim=-1)
        chosen = torch.topk(scores, moe.top_k, dim=-1).indices
        # Raise the bias only on already-selected experts: selection is
        # unchanged, so if the bias leaked into the gate weights the output
        # would move.
        bump = torch.zeros(moe.n_routed)
        for expert in chosen.flatten().unique():
            bump[expert] = 0.5
        moe.expert_bias.copy_(bump)
        after = moe(x)
        again_chosen = torch.topk(scores + moe.expert_bias, moe.top_k, dim=-1).indices

    assert torch.equal(chosen, again_chosen), "test setup changed the selection"
    assert torch.allclose(baseline, after, atol=0.0), "expert_bias leaked into the forward value"


def test_bias_rule_rebalances_a_collapsed_router():
    """Identical tokens collapse onto one expert pair; the bias must spread them."""

    torch.manual_seed(0)
    moe = mc.SparseMoEFeedForward(tiny_config(router_bias_update_speed=0.2))
    moe.train()
    x = torch.randn(1, 32).repeat(8, 1)  # every token routes identically

    first = None
    used = set()
    for step in range(60):
        moe(x)
        moe.update_router_bias()
        active = {i for i, v in enumerate(moe.last_expert_load.tolist()) if v > 0}
        if step == 0:
            first = set(active)
        used |= active

    assert len(first) == moe.top_k, "collapsed setup did not collapse"
    assert used == set(range(moe.n_routed)), (
        f"balancer never reached every expert: {sorted(used)}"
    )


def test_bias_update_is_not_fired_by_forward():
    """One optimizer step must mean one bias update, whatever the forward count."""

    moe = mc.SparseMoEFeedForward(tiny_config(router_bias_update_speed=0.1))
    moe.train()
    x = torch.randn(4, 32)

    for _ in range(5):  # e.g. gradient accumulation, or controller probes
        moe(x)
    assert torch.equal(moe.expert_bias, torch.zeros(moe.n_routed)), (
        "forward() moved the bias; effective gamma would scale with forward count"
    )
    assert float(moe.pending_batches) == 5.0

    assert moe.update_router_bias() is True
    assert moe.expert_bias.abs().sum() > 0
    assert float(moe.pending_batches) == 0.0
    # nothing accumulated since, so a second call is a no-op
    assert moe.update_router_bias() is False


def test_bias_update_magnitude_is_one_gamma_step():
    moe = mc.SparseMoEFeedForward(tiny_config(router_bias_update_speed=0.1))
    moe.train()
    for _ in range(7):
        moe(torch.randn(4, 32))
    moe.update_router_bias()
    non_zero = moe.expert_bias[moe.expert_bias != 0]
    assert torch.allclose(non_zero.abs(), torch.full_like(non_zero, 0.1))


def test_auto_update_mode_is_opt_in():
    auto = mc.SparseMoEFeedForward(tiny_config(router_bias_update_speed=0.1, router_bias_auto_update=True))
    auto.train()
    auto(torch.randn(4, 32))
    assert auto.expert_bias.abs().sum() > 0


def test_eval_forwards_never_touch_routing_state():
    moe = mc.SparseMoEFeedForward(tiny_config(router_bias_update_speed=0.1))
    moe.eval()
    moe(torch.randn(4, 32))
    assert float(moe.pending_batches) == 0.0
    assert moe.update_router_bias() is False


def test_model_level_bias_step_covers_every_moe_layer():
    model = mc.MiMoMixModel(tiny_config(router_bias_update_speed=0.1))
    model.train()
    model(torch.randint(0, 48, (2, 8)))
    expected = sum(1 for m in model.modules() if isinstance(m, mc.SparseMoEFeedForward))
    assert model.step_router_bias() == expected
    assert model.step_router_bias() == 0


def test_router_bias_survives_a_state_dict_round_trip():
    moe = mc.SparseMoEFeedForward(tiny_config())
    with torch.no_grad():
        moe.expert_bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))
    restored = mc.SparseMoEFeedForward(tiny_config())
    restored.load_state_dict(moe.state_dict())
    assert torch.allclose(restored.expert_bias, moe.expert_bias)


def test_router_regularisers_are_training_only():
    model = mc.MiMoMixModel(tiny_config())
    ids = torch.randint(0, 48, (2, 8))

    model.eval()
    with torch.no_grad():
        assert float(model(ids, past_length=0).aux_loss) == 0.0

    model.train()
    assert float(model(ids, past_length=0).aux_loss.detach()) > 0.0


def test_shared_expert_is_always_applied():
    config = tiny_config(n_shared_experts=1)
    moe = mc.SparseMoEFeedForward(config)
    assert moe.shared_expert is not None
    moe_off = mc.SparseMoEFeedForward(tiny_config(n_shared_experts=0))
    assert moe_off.shared_expert is None


def test_first_layers_stay_dense():
    model = mc.MiMoMixModel(tiny_config(n_dense_layers=2))
    kinds = [isinstance(layer.mlp, mc.SparseMoEFeedForward) for layer in model.layers]
    assert kinds[:2] == [False, False]
    assert any(kinds[2:])


# ---------------------------------------------------------------------------
# Multi-token prediction
# ---------------------------------------------------------------------------


def test_mtp_depths_produce_one_logit_tensor_each():
    model = mc.MiMoMixModel(tiny_config(n_mtp_layers=3))
    ids = torch.randint(0, 48, (2, 10))
    out = model(ids, labels=ids)
    assert len(out.mtp_logits) == 3
    for depth_logits in out.mtp_logits:
        assert depth_logits.shape == (2, 10, 48)
    assert out.mtp_loss is not None and float(out.mtp_loss.detach()) > 0.0


def test_mtp_can_be_switched_off_at_call_time():
    model = mc.MiMoMixModel(tiny_config())
    ids = torch.randint(0, 48, (1, 8))
    out = model(ids, return_mtp=False)
    assert out.mtp_logits == [] and out.mtp_loss is None


def test_mtp_modules_use_dense_feed_forward():
    model = mc.MiMoMixModel(tiny_config(use_moe=True))
    for module in model.mtp_modules:
        assert isinstance(module.block.mlp, mc.DenseFeedForward)


def test_propose_draft_returns_one_token_per_depth():
    model = mc.MiMoMixModel(tiny_config(n_mtp_layers=3))
    model.eval()
    ids = torch.randint(0, 48, (2, 7))
    with torch.no_grad():
        out = model(ids, past_length=0)
        seed = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        draft = model.propose_draft(out.trunk_hidden[:, -1:], seed, position=6)
    assert draft.shape == (2, 3)
    assert draft.dtype == torch.long


# ---------------------------------------------------------------------------
# Recursive thinking core
# ---------------------------------------------------------------------------


def test_thinking_core_halting_mass_is_convex():
    core = mc.RecursiveThinkingCore(tiny_config())
    core.eval()
    hidden = torch.randn(3, 5, 32)
    for budget in (1, 2, 4):
        _, info = core(hidden, cycles=budget)
        assert 1 <= float(info["cycles_used"]) <= budget
        assert float(info["ponder_cost"].detach()) >= 0.0


def test_thinking_core_respects_the_max_cycle_cap():
    core = mc.RecursiveThinkingCore(tiny_config(thinking_max_cycles=3))
    _, info = core(torch.randn(2, 4, 32), cycles=99)
    assert float(info["cycles_used"]) <= 3


def test_adaptive_thinking_can_exit_before_the_budget():
    core = mc.RecursiveThinkingCore(tiny_config(thinking_max_cycles=8))
    core.eval()
    hidden = torch.randn(2, 3, 32)
    _, fixed = core(hidden, cycles=8, adaptive=False)
    _, adaptive = core(hidden, cycles=8, adaptive=True, stability_tol=1e9, stability_patience=1)
    assert float(fixed["cycles_used"]) == 8
    assert float(adaptive["cycles_used"]) < 8
    assert core.last_exit_reason == "prediction_stability"


def test_fresh_thinking_core_is_a_near_identity():
    """Near-zero residual init keeps a freshly built model's behaviour."""

    core = mc.RecursiveThinkingCore(tiny_config())
    core.eval()
    hidden = torch.randn(2, 4, 32)
    refined, _ = core(hidden, cycles=3)
    assert torch.allclose(refined, hidden, atol=1e-6)


def test_verifier_loss_trains_the_quality_head_and_temperature():
    model = mc.MiMoMixModel(tiny_config())
    model.train()
    ids = torch.randint(0, 48, (2, 6))
    model(ids)
    correctness = torch.randint(0, 2, (2 * 6,)).float()
    loss = model.verifier_loss(correctness)
    loss.backward()

    core = model.thinking_core
    assert core.quality_head.weight.grad is not None
    assert core.quality_head.weight.grad.abs().sum() > 0
    assert core.log_temperature.grad is not None


def test_verifier_loss_requires_a_training_forward():
    model = mc.MiMoMixModel(tiny_config())
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, 48, (1, 5)))
    with pytest.raises(RuntimeError, match="forward pass"):
        model.verifier_loss(torch.ones(5))


# ---------------------------------------------------------------------------
# Whole-model contracts
# ---------------------------------------------------------------------------


def test_every_trained_parameter_receives_gradient():
    """The LM objective must reach everything except the verifier-only path."""

    torch.manual_seed(0)
    model = mc.MiMoMixModel(tiny_config())
    model.train()
    ids = torch.randint(0, 48, (2, 10))
    model(ids, labels=ids).loss.backward()

    verifier_only = {
        "thinking_core.quality_encoder.0.weight",
        "thinking_core.quality_encoder.0.bias",
        "thinking_core.quality_head.weight",
        "thinking_core.quality_head.bias",
        "thinking_core.log_temperature",
    }
    missing = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or parameter.grad.abs().sum() == 0
    }
    assert missing <= verifier_only, f"unexpectedly unreachable parameters: {sorted(missing - verifier_only)}"


def test_loss_composition_includes_every_declared_term():
    model = mc.MiMoMixModel(tiny_config())
    model.train()
    ids = torch.randint(0, 48, (2, 10))
    out = model(ids, labels=ids)
    assert out.lm_loss is not None and out.mtp_loss is not None and out.aux_loss is not None
    assert float(out.loss.detach()) > float(out.lm_loss.detach())


def test_telemetry_is_json_safe_and_complete():
    model = mc.MiMoMixModel(tiny_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.randint(0, 48, (1, 9)), past_length=0)
    payload = json.dumps(out.telemetry)
    restored = json.loads(payload)
    for key in ("attention_layout", "expert_load", "parameters", "thinking", "mean_sink_mass"):
        assert key in restored
    assert restored["thinking"]["exit_reason"]


def test_parameter_report_separates_active_from_idle():
    model = mc.MiMoMixModel(tiny_config())
    report = model.parameter_report()
    assert report["total"] == sum(p.numel() for p in model.parameters())
    assert report["routed_but_idle"] > 0
    assert report["active_per_token"] < report["total"]
    assert (
        report["active_per_token"] + report["routed_but_idle"] + report["mtp_draft"]
        == report["total"]
    )


def test_a_dense_model_reports_no_idle_parameters():
    model = mc.MiMoMixModel(tiny_config(use_moe=False, n_mtp_layers=0))
    report = model.parameter_report()
    assert report["routed_but_idle"] == 0
    assert report["active_per_token"] == report["total"]


def test_weight_tying_is_real():
    tied = mc.MiMoMixModel(tiny_config(tie_word_embeddings=True))
    assert tied.lm_head.weight.data_ptr() == tied.embed_tokens.weight.data_ptr()
    untied = mc.MiMoMixModel(tiny_config(tie_word_embeddings=False))
    assert untied.lm_head.weight.data_ptr() != untied.embed_tokens.weight.data_ptr()


def test_build_helper_accepts_overrides():
    model = mc.build_mimomix(n_layers=3, use_moe=False, vocab_size=32, hidden_size=16, n_heads=2, n_kv_heads=1)
    assert model.config.n_layers == 3 and model.config.use_moe is False
