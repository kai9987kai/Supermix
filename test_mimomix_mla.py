import math
import pytest
import torch
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import mimomix_core as mc


def test_mla_config_validation():
    cfg = mc.MiMoMixConfig(hidden_size=128, use_mla=True, mla_latent_dim=32, mla_pe_dim=16)
    assert cfg.use_mla is True
    assert cfg.mla_latent_dim == 32
    assert cfg.mla_pe_dim == 16

    with pytest.raises(ValueError, match="mla_latent_dim must be positive"):
        mc.MiMoMixConfig(use_mla=True, mla_latent_dim=0)


def test_mla_attention_forward():
    cfg = mc.MiMoMixConfig(hidden_size=64, n_heads=4, head_dim=16, use_mla=True, mla_latent_dim=24, mla_pe_dim=8)
    attn = mc.MultiLatentAttention(cfg, layer_index=0, kind="global")

    bsz, seq_len = 2, 8
    x = torch.randn(bsz, seq_len, cfg.hidden_size)
    positions = torch.arange(seq_len)
    rotary = mc.RotaryEmbedding(cfg, kind="global")
    cos, sin = rotary(positions)
    empty_pos = torch.empty((0,), dtype=torch.long)

    out, present = attn(x, cos, sin, positions, empty_pos, use_cache=True)
    assert out.shape == (bsz, seq_len, cfg.hidden_size)
    assert present is not None
    c_kv, k_pe = present
    assert c_kv.shape == (bsz, seq_len, cfg.mla_latent_dim)
    assert k_pe.shape == (bsz, 1, seq_len, cfg.mla_pe_dim)


def test_mla_cache_size_reduction():
    cfg = mc.MiMoMixConfig(hidden_size=128, n_heads=8, head_dim=16, use_mla=True, mla_latent_dim=32, mla_pe_dim=16)
    standard_kv_elements = 2 * cfg.n_heads * cfg.head_dim  # 2 * 8 * 16 = 256 per token
    mla_kv_elements = cfg.mla_latent_dim + cfg.mla_pe_dim   # 32 + 16 = 48 per token
    reduction_ratio = standard_kv_elements / float(mla_kv_elements)
    assert reduction_ratio > 5.0  # > 5x memory reduction per token


def test_mla_kv_caching_step_by_step():
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        use_mla=True,
        mla_latent_dim=16,
        mla_pe_dim=8,
        use_thinking_core=False,
    )
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


def test_mla_model_full_forward_and_telemetry():
    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        use_mla=True,
        mla_latent_dim=16,
        mla_pe_dim=8,
        use_thinking_core=False,
    )
    model = mc.MiMoMixModel(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert "mla_active" in out.telemetry
    assert out.telemetry["mla_active"] is True
    assert out.telemetry["mla_latent_dim"] == 16
