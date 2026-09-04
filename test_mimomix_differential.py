import math
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import mimomix_core as mc


def test_differential_config_validation():
    cfg = mc.MiMoMixConfig(hidden_size=128, n_heads=4, head_dim=32, use_differential_attention=True)
    assert cfg.use_differential_attention is True
    assert cfg.differential_lambda_init == 0.8

    with pytest.raises(ValueError, match="head_dim must be"):
        mc.MiMoMixConfig(hidden_size=60, n_heads=4, head_dim=15, use_differential_attention=True)


def test_differential_hybrid_attention_forward():
    cfg = mc.MiMoMixConfig(hidden_size=64, n_heads=4, n_kv_heads=2, head_dim=16, use_differential_attention=True)
    attn = mc.DifferentialHybridAttention(cfg, layer_index=0, kind="global")
    
    bsz, seq_len = 2, 8
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    positions = torch.arange(seq_len)
    rotary = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = rotary(positions)
    empty_pos = torch.empty((0,), dtype=torch.long)

    out, present = attn(x, cos, sin, positions, empty_pos, use_cache=True)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)
    assert present is not None
    assert present[0].shape == (bsz, cfg.n_kv_heads, seq_len, cfg.head_dim)
    assert present[1].shape == (bsz, cfg.n_kv_heads, seq_len, cfg.head_dim)
    assert attn.last_lambda.shape == (cfg.n_heads,)
    assert (attn.last_lambda > 0.0).all() and (attn.last_lambda < 1.0).all()


def test_differential_lambda_learning():
    cfg = mc.MiMoMixConfig(hidden_size=64, n_heads=4, n_kv_heads=2, head_dim=16, use_differential_attention=True)
    attn = mc.DifferentialHybridAttention(cfg, layer_index=0, kind="global")

    x = torch.randn(2, 4, cfg.hidden_size, requires_grad=True)
    positions = torch.arange(4)
    rotary = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = rotary(positions)
    empty_pos = torch.empty((0,), dtype=torch.long)

    out, _ = attn(x, cos, sin, positions, empty_pos)
    loss = out.sum()
    loss.backward()

    assert attn.lambda_param.grad is not None
    assert torch.isfinite(attn.lambda_param.grad).all()


def test_differential_kv_caching_step_by_step():
    cfg = mc.MiMoMixConfig(vocab_size=64, hidden_size=32, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=16, use_differential_attention=True)
    model = mc.MiMoMixModel(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        full_out = model(input_ids)

    past_kv = None
    step_logits = []
    with torch.no_grad():
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i : i + 1]
            out = model(token, past_key_values=past_kv, use_cache=True, past_length=i)
            past_kv = out.past_key_values
            step_logits.append(out.logits)

    combined_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_out.logits, combined_logits, atol=1e-5)


def test_differential_model_full_forward_and_telemetry():
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=2,
        head_dim=16,
        use_differential_attention=True,
        use_thinking_core=False,
    )
    model = mc.MiMoMixModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "differential_attention" in out.telemetry
    assert out.telemetry["differential_attention"] is True
    assert "differential_lambdas" in out.telemetry
    assert len(out.telemetry["differential_lambdas"]) >= cfg.n_layers
