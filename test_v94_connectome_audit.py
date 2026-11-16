"""Offline v94 checks, on tiny models; no training checkpoint is loaded."""
import os
import sys
from dataclasses import replace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from mimomix_core import MiMoMixConfig, MiMoMixModel
from v94_connectome_audit import MODES, RowLoss, compare_losses, connectome_mode


@pytest.fixture
def model():
    # Keep RNG local and avoid competing with any training workload.
    with torch.random.fork_rng():
        torch.manual_seed(94)
        net = MiMoMixModel(MiMoMixConfig(
            vocab_size=31, hidden_size=16, n_layers=3, n_heads=2, n_kv_heads=1,
            intermediate_size=24, moe_intermediate_size=8, n_routed_experts=4,
            moe_top_k=2, n_mtp_layers=1, native_context=16,
            max_position_embeddings=16, sliding_window=8, thinking_latent_dim=8,
            use_cns_core=True, cns_nodes=6, cns_spare_nodes=2,
            cns_after_layer=1, cns_steps=2, cns_read_layers=(0, 1),
            cns_write_layers=(1, 2), cns_to_thinking=True,
        )).eval()
        core = net.cns_core
        with torch.no_grad():
            core.hemisphere[:4].copy_(torch.tensor([0, 0, 1, 1], dtype=torch.int8))
            core.mask[:4, :4].fill_(1)
            core.in_mask[:4].fill_(1)
            core.out_mask[:4].fill_(1)
            core.read_in.weight[:4].normal_(0, 0.5)
            core.read_out.weight[:, :4].normal_(0, 0.5)
            core.extra_read_out[0].weight[:, :4].normal_(0, 0.5)
            core.to_thinking.weight[:, :4].normal_(0, 0.5)
            core.gate.fill_(0.5)
            core.extra_gates[0].fill_(0.5)
            core.thinking_gate.fill_(0.5)
    return net


def logits(model):
    with torch.no_grad():
        return model(torch.tensor([[1, 4, 9, 3]]), return_mtp=False).logits


def test_all_off_closes_paths_the_old_single_gate_check_misses(model):
    core = model.cns_core
    baseline = logits(model)
    with torch.no_grad():
        core.gate.zero_()
    old_partial = logits(model)
    with connectome_mode(model, "all_off"):
        actual = logits(model)
        assert all(torch.count_nonzero(g).item() == 0 for g in
                   [core.gate, *core.extra_gates, core.thinking_gate])
    assert not torch.equal(old_partial, actual)
    with torch.no_grad():
        core.extra_gates[0].zero_()
        core.thinking_gate.zero_()
    assert torch.equal(logits(model), actual)
    assert not torch.equal(baseline, actual)


@pytest.mark.parametrize("mode", MODES)
def test_modes_restore_parameters_buffers_and_flags_even_on_error(model, mode):
    core = model.cns_core
    core.collect_stats = True
    core.ablate_cross, core.ablate_grown, core.ablate_side = True, True, 1
    telemetry = model.telemetry()
    parameters = {k: v.detach().clone() for k, v in model.named_parameters()}
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    identities = {k: id(v) for k, v in model.named_parameters()}
    with pytest.raises(RuntimeError, match="test failure"):
        with connectome_mode(model, mode):
            assert not core.collect_stats
            assert core.ablate_cross == (mode == "commissure_off")
            logits(model)
            raise RuntimeError("test failure")
    assert core.collect_stats and core.ablate_cross and core.ablate_grown
    assert core.ablate_side == 1
    assert model.telemetry() == telemetry
    assert identities == {k: id(v) for k, v in model.named_parameters()}
    assert all(torch.equal(parameters[k], v) for k, v in model.named_parameters())
    assert all(torch.equal(buffers[k], v) for k, v in model.named_buffers())


def test_successful_audit_preserves_plain_telemetry(model):
    before = model.telemetry()
    with connectome_mode(model, "baseline"):
        logits(model)
    assert model.telemetry() == before


def test_rejects_training_model_and_one_sided_graph(model):
    model.train()
    with pytest.raises(ValueError, match="eval"):
        with connectome_mode(model, "all_off"):
            pass
    model.eval()
    model.cns_core.hemisphere.zero_()
    with pytest.raises(ValueError, match="both sides"):
        with connectome_mode(model, "left_off"):
            pass


def test_paired_comparison_is_order_independent_and_token_weighted():
    before = [RowLoss("a", "old", 2, 2), RowLoss("b", "old", 8, 4)]
    after = [RowLoss("b", "old", 12, 4), RowLoss("a", "old", 3, 2)]
    result = compare_losses(before, after, bootstrap_samples=100)
    assert result == compare_losses(before[::-1], after[::-1], bootstrap_samples=100)
    assert result["overall"]["delta_nats"] == pytest.approx(5 / 6)
    assert result["overall"]["ci95"] == pytest.approx([0.5, 1])


def test_repeated_semantic_group_has_no_false_independent_interval():
    before = [RowLoss(str(i), "new", 2, 2, "one_problem") for i in range(5)]
    result = compare_losses(before, before, bootstrap_samples=100)
    assert result["overall"]["independent_groups"] == 1
    assert result["overall"]["ci95"] is None


@pytest.mark.parametrize("field,value", [("row_id", "other"), ("family", "new"),
    ("token_count", 3), ("token_count", 0), ("loss_sum", float("nan")),
    ("loss_sum", -1), ("group_id", "other")])
def test_comparison_rejects_bad_pairing_or_losses(field, value):
    row = RowLoss("a", "old", 2, 2)
    with pytest.raises(ValueError):
        compare_losses([row], [replace(row, **{field: value})], bootstrap_samples=100)


def test_comparison_rejects_duplicate_ids():
    row = RowLoss("a", "old", 2, 2)
    with pytest.raises(ValueError, match="duplicate"):
        compare_losses([row, row], [row, row], bootstrap_samples=100)
