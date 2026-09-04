import math
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import mimomix_core as mc


def test_mod_config_validation():
    cfg = mc.MiMoMixConfig(hidden_size=128, use_mod=True, mod_capacity_ratio=0.6)
    assert cfg.use_mod is True
    assert cfg.mod_capacity_ratio == 0.6

    with pytest.raises(ValueError, match="mod_capacity_ratio must be in"):
        mc.MiMoMixConfig(use_mod=True, mod_capacity_ratio=0.0)

    with pytest.raises(ValueError, match="mod_capacity_ratio must be in"):
        mc.MiMoMixConfig(use_mod=True, mod_capacity_ratio=1.5)


def test_mod_router_forward():
    router = mc.MixtureOfDepthsRouter(hidden_size=64, capacity_ratio=0.5)
    bsz, seq_len = 2, 8
    x = torch.randn(bsz, seq_len, 64)

    selected_indices, weights, skip_ratio = router(x)
    assert selected_indices.shape == (bsz, 4)  # 50% of 8 = 4
    assert weights.shape == (bsz, seq_len)
    assert (weights >= 0.0).all() and (weights <= 1.0).all()
    assert torch.isclose(skip_ratio, torch.tensor([0.5]))


def test_mod_block_execution():
    cfg = mc.MiMoMixConfig(
        hidden_size=64,
        n_layers=4,
        n_dense_layers=1,
        use_mod=True,
        mod_capacity_ratio=0.5,
    )
    block = mc.MiMoMixBlock(cfg, layer_index=2, kind="global")
    assert block.use_mod is True
    assert block.mod_router is not None

    bsz, seq_len = 2, 6
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    positions = torch.arange(seq_len)
    rotary = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = rotary(positions)
    empty_pos = torch.empty((0,), dtype=torch.long)

    out, present = block(x, cos, sin, positions, empty_pos)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)


def test_mod_gradient_flow():
    cfg = mc.MiMoMixConfig(
        hidden_size=64,
        n_layers=4,
        n_dense_layers=1,
        use_mod=True,
        mod_capacity_ratio=0.5,
    )
    block = mc.MiMoMixBlock(cfg, layer_index=2, kind="global")
    x = torch.randn(2, 6, cfg.hidden_size, requires_grad=True)
    positions = torch.arange(6)
    rotary = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = rotary(positions)
    empty_pos = torch.empty((0,), dtype=torch.long)

    out, _ = block(x, cos, sin, positions, empty_pos)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert block.mod_router.router_proj.weight.grad is not None


def test_mod_full_model_telemetry():
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=4,
        n_dense_layers=1,
        use_mod=True,
        mod_capacity_ratio=0.5,
        use_thinking_core=False,
    )
    model = mc.MiMoMixModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "mod_skip_ratios" in out.telemetry
    assert "mod_mean_skip" in out.telemetry
    assert out.telemetry["mod_mean_skip"] > 0.0


def test_mod_and_moe_joint_execution():
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=4,
        n_dense_layers=1,
        use_moe=True,
        n_routed_experts=4,
        moe_top_k=2,
        use_mod=True,
        mod_capacity_ratio=0.5,
        use_thinking_core=False,
    )
    model = mc.MiMoMixModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert "mod_skip_ratios" in out.telemetry
    assert "expert_load" in out.telemetry
