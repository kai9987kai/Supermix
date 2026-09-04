"""Tests for multimodal token projection in MiMoMix (Xiaomi MiMo-V2.5 lineage)."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

from mimomix_core import (
    MiMoMixConfig,
    MiMoMixModel,
    MultimodalProjectionHead,
)


def test_multimodal_projection_head_shapes():
    proj = MultimodalProjectionHead(input_dim=64, hidden_size=128, modality="vision")
    features = torch.randn(2, 8, 64)
    out = proj(features)
    assert out.shape == (2, 8, 128)
    assert torch.isfinite(out).all()


def test_mimomix_model_with_multimodal_enabled():
    config = MiMoMixConfig(
        n_layers=2,
        hidden_size=64,
        n_heads=2,
        n_kv_heads=1,
        use_multimodal=True,
        multimodal_input_dim=32,
    )
    model = MiMoMixModel(config)
    assert model.multimodal_projector is not None

    raw_features = torch.randn(1, 4, 32)
    projected = model.encode_multimodal_tokens(raw_features, modality="vision")
    assert projected.shape == (1, 4, 64)
    assert torch.isfinite(projected).all()


def test_mimomix_model_multimodal_disabled_raises():
    config = MiMoMixConfig(
        n_layers=2,
        hidden_size=64,
        n_heads=2,
        n_kv_heads=1,
        use_multimodal=False,
    )
    model = MiMoMixModel(config)
    assert model.multimodal_projector is None

    raw_features = torch.randn(1, 4, 32)
    with pytest.raises(RuntimeError, match="multimodal_projector is not enabled"):
        model.encode_multimodal_tokens(raw_features)
