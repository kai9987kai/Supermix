"""Tests for MTP self-speculative decoding.

The load-bearing property is exact greedy equivalence: speculation may change
how many forwards run, never which tokens come out. Everything else here exists
to make sure that property is not accidentally satisfied (e.g. by the draft
never being accepted).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_core as mc  # noqa: E402
import mimomix_decoding as md  # noqa: E402


def build(seed: int = 0, **overrides) -> mc.MiMoMixModel:
    torch.manual_seed(seed)
    base = dict(
        vocab_size=48,
        hidden_size=32,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        sliding_window=3,
        hybrid_ratio=2,
        native_context=16,
        max_position_embeddings=96,
        n_routed_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        n_mtp_layers=3,
        thinking_cycles=2,
    )
    base.update(overrides)
    model = mc.MiMoMixModel(mc.MiMoMixConfig(**base))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# The guarantee
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_speculative_decoding_is_bit_identical_to_greedy(seed):
    model = build(seed)
    ids = torch.randint(0, 48, (1, 7))
    report = md.assert_greedy_equivalence(model, ids, max_new_tokens=16)
    assert report["tokens"] == 16


def test_equivalence_holds_for_a_batch():
    model = build(9, n_kv_heads=1, sliding_window=2, hybrid_ratio=3)
    ids = torch.randint(0, 48, (4, 6))
    md.assert_greedy_equivalence(model, ids, max_new_tokens=12)


def test_equivalence_holds_without_a_thinking_core_or_moe():
    model = build(5, use_thinking_core=False, use_moe=False, rope_scaling="none")
    ids = torch.randint(0, 48, (2, 5))
    md.assert_greedy_equivalence(model, ids, max_new_tokens=10)


def test_equivalence_holds_when_every_layer_is_sliding_window():
    """No global layer means past_length cannot be inferred from the cache."""

    model = build(7, hybrid_ratio=99, final_layer_global=False, sliding_window=2)
    assert set(model.layout) == {"swa"}
    ids = torch.randint(0, 48, (1, 6))
    md.assert_greedy_equivalence(model, ids, max_new_tokens=12)


@pytest.mark.parametrize("cycles", [1, 3])
def test_equivalence_holds_at_each_fixed_thinking_budget(cycles):
    model = build(2)
    ids = torch.randint(0, 48, (1, 6))
    md.assert_greedy_equivalence(model, ids, max_new_tokens=10, thinking_cycles=cycles)


def test_speculation_refuses_adaptive_thinking():
    model = build(0)
    with pytest.raises(ValueError, match="block-independent"):
        md.speculative_generate(model, torch.randint(0, 48, (1, 4)), adaptive_thinking=True)


# ---------------------------------------------------------------------------
# Drafts are actually accepted sometimes
# ---------------------------------------------------------------------------


def test_a_learnable_pattern_produces_real_acceptance():
    """Train on a repeating cycle so the MTP depths can predict ahead.

    Without this the equivalence tests could pass trivially -- a draft that is
    never accepted is also never wrong.
    """

    torch.manual_seed(0)
    model = mc.MiMoMixModel(
        mc.MiMoMixConfig(
            vocab_size=8,
            hidden_size=64,
            n_layers=4,
            n_heads=4,
            n_kv_heads=2,
            intermediate_size=128,
            sliding_window=16,
            hybrid_ratio=0,
            native_context=64,
            max_position_embeddings=64,
            use_moe=False,
            n_mtp_layers=3,
            mtp_loss_weight=1.0,
            use_thinking_core=False,
            rope_scaling="none",
        )
    )
    pattern = torch.tensor([[0, 1, 2, 3] * 8])
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(220):
        optimiser.zero_grad()
        model(pattern, labels=pattern).loss.backward()
        optimiser.step()

    model.eval()
    prompt = pattern[:, :8]
    fast = md.speculative_generate(model, prompt, max_new_tokens=20)
    slow = md.greedy_generate(model, prompt, max_new_tokens=20)

    assert torch.equal(fast.new_tokens, slow.new_tokens)
    assert fast.stats.acceptance_rate > 0.5, f"draft never accepted: {fast.stats.to_dict()}"
    assert fast.stats.acceptance_length > 1.5
    assert fast.stats.verify_forwards < slow.stats.verify_forwards


def test_draft_length_can_be_capped():
    model = build(3)
    ids = torch.randint(0, 48, (1, 5))
    capped = md.speculative_generate(model, ids, max_new_tokens=8, draft_length=1)
    reference = md.greedy_generate(model, ids, max_new_tokens=8)
    assert torch.equal(capped.new_tokens, reference.new_tokens)


def test_zero_mtp_layers_degrades_to_greedy():
    model = build(4, n_mtp_layers=0)
    ids = torch.randint(0, 48, (1, 5))
    result = md.speculative_generate(model, ids, max_new_tokens=8)
    reference = md.greedy_generate(model, ids, max_new_tokens=8)
    assert torch.equal(result.new_tokens, reference.new_tokens)
    assert result.stats.drafted_tokens == 0
    assert result.stats.acceptance_rate == 0.0


# ---------------------------------------------------------------------------
# Statistics and cache bookkeeping
# ---------------------------------------------------------------------------


def test_greedy_scores_exactly_one_token_per_forward():
    model = build(0)
    ids = torch.randint(0, 48, (1, 5))
    stats = md.greedy_generate(model, ids, max_new_tokens=12).stats
    # one prefill token plus one per verify forward
    assert stats.new_tokens == stats.verify_forwards + 1
    assert stats.acceptance_length == pytest.approx(stats.new_tokens / stats.verify_forwards)


def test_stats_are_json_safe():
    import json

    model = build(0)
    result = md.speculative_generate(model, torch.randint(0, 48, (1, 5)), max_new_tokens=6)
    payload = json.loads(json.dumps(result.stats.to_dict()))
    assert payload["mode"] == "speculative"
    assert "acceptance_length" in payload


def test_generation_stops_at_eos():
    model = build(0)
    ids = torch.randint(0, 48, (1, 5))
    long = md.greedy_generate(model, ids, max_new_tokens=20)
    target = int(long.new_tokens[0, 3])
    stopped = md.greedy_generate(model, ids, max_new_tokens=20, eos_token_id=target)
    assert stopped.stats.new_tokens <= 4


def test_trim_past_removes_exactly_the_requested_tail():
    keys = torch.arange(24).float().reshape(1, 2, 6, 2)
    values = keys.clone()
    trimmed = md.trim_past([(keys, values)], 2)
    assert trimmed[0][0].shape[2] == 4
    assert torch.equal(trimmed[0][0], keys[:, :, :4])
    assert md.trim_past([(keys, values)], 0)[0][0].shape[2] == 6
    assert md.trim_past(None, 3) is None


def test_hybrid_cache_footprint_matches_the_layout():
    model = build(0, sliding_window=128, hybrid_ratio=5, n_layers=12)
    footprint = md.hybrid_cache_footprint(model, 4096)
    n_global = model.layout.count("global")
    n_swa = model.layout.count("swa")
    assert footprint["hybrid_entries"] == n_global * 4096 + n_swa * 128
    assert footprint["all_global_entries"] == 12 * 4096
    assert footprint["reduction_factor"] > 3.0

    # A short sequence cannot save anything: the window is never reached.
    short = md.hybrid_cache_footprint(model, 64)
    assert short["reduction_factor"] == 1.0
    assert short["saved_fraction"] == 0.0
