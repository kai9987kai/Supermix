from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.join(os.getcwd(), "source"))

from multimodel_catalog import ModelRecord
from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2
from omni_collective_v7_model import OmniCollectiveNetV7, _prompt_variants_v7
from train_omni_collective_v2 import OmniRow
import train_omni_collective_v7 as train_v7


def test_prompt_variants_v7_adds_grounding_math_and_protein_paths():
    variants = _prompt_variants_v7(
        "Solve 3*x + 7 = 19 and explain why alpha helices matter in protein folding.",
        "x = 4",
        response_confidence=0.18,
        domain_confidence=0.22,
    )
    joined = "\n".join(variants).lower()
    assert "ground the answer" in joined
    assert "hallucination guard" in joined
    assert "may involve math" in joined
    assert "protein folding" in joined
    assert len(variants) >= 6


def test_specialist_profile_rows_v7_cover_chat_and_non_chat_models(monkeypatch, tmp_path):
    fake_zip = tmp_path / "fake.zip"
    fake_zip.write_text("x", encoding="utf-8")
    records = [
        ModelRecord(
            key="omni_collective_v6",
            label="Omni Collective V6 Frontier",
            family="fusion",
            kind="omni_collective_v6",
            capabilities=("chat", "vision"),
            zip_path=fake_zip,
            common_row_key="omni_collective_v6",
            common_overall_exact=0.0733,
            note="Chat-capable multimodal frontier.",
            benchmark_hint="All-model distilled multimodal frontier.",
        ),
        ModelRecord(
            key="dcgan_mnist_model",
            label="DCGAN MNIST",
            family="gan",
            kind="dcgan_image",
            capabilities=("image",),
            zip_path=fake_zip,
            common_row_key=None,
            common_overall_exact=None,
            note="Unconditional GAN trained on digits.",
            benchmark_hint="MNIST digit-grid generator.",
        ),
    ]
    monkeypatch.setattr(train_v7, "discover_model_records", lambda models_dir: list(records))
    rows, summary = train_v7._specialist_profile_rows_v7(Path.cwd(), tmp_path, seed=7)
    prompts = "\n".join(row.prompt for row in rows).lower()
    responses = "\n".join(row.response_text for row in rows).lower()
    assert summary["record_count"] == 2
    assert "dcgan mnist" in prompts
    assert "omni collective v6 frontier" in prompts
    assert "not a normal chat consultant" in responses
    assert "fusion" in responses or "gan" in responses


def test_benchmax_rows_v7_build_from_mocked_builders(monkeypatch):
    sample_v33 = [{"user": "Solve 2+2", "assistant": "The answer is 4.", "metadata": {"benchmark": "gsm8k"}}]
    sample_v39 = [{"user": "What is BoolQ asking for?", "assistant": "A yes or no answer.", "metadata": {"benchmark": "boolq"}}]
    monkeypatch.setattr(train_v7, "build_v33_style_rows", lambda repo_root, seed, sample_size: (list(sample_v33), {"count": 1}))
    monkeypatch.setattr(train_v7, "build_v39_style_rows", lambda repo_root, seed, sample_size: (list(sample_v39), {"count": 1}))
    rows, counts = train_v7._benchmax_rows_v7(Path.cwd(), seed=19)
    assert rows
    assert counts["benchmax_v33_v7"] == 1
    assert counts["benchmax_v39_v7"] == 1
    assert any(row.domain in {"knowledge", "math", "planning", "coding"} for row in rows)


def test_dedupe_rows_v7_keeps_unique_prompt_response_pairs():
    rows = [
        OmniRow(prompt="A", intent="general", response_text="B", domain="general", source="x"),
        OmniRow(prompt="A", intent="general", response_text="B", domain="general", source="x"),
        OmniRow(prompt="A", intent="general", response_text="C", domain="general", source="x"),
    ]
    deduped = train_v7._dedupe_rows(rows)
    assert len(deduped) == 2


def test_teacher_resume_state_roundtrip(tmp_path):
    resume_dir = train_v7._teacher_resume_dir_v7(tmp_path, seed=11, limit=8, teacher_keys=["v40_benchmax", "qwen_v28"])
    sample_path = train_v7._teacher_sample_path_v7(resume_dir)
    state_path = train_v7._teacher_state_path_v7(resume_dir, "v40_benchmax")
    rows = [
        OmniRow(prompt="P1", intent="general", response_text="R1", domain="general", source="s1"),
        OmniRow(prompt="P2", intent="coding", response_text="R2", domain="coding", source="s2"),
    ]
    train_v7._save_teacher_sample_v7(sample_path, rows)
    loaded_rows = train_v7._load_teacher_sample_v7(sample_path)
    assert [row.prompt for row in loaded_rows] == ["P1", "P2"]

    train_v7._write_json_atomic(
        state_path,
        {
            "teacher": "v40_benchmax",
            "status": "partial",
            "sample_total": 2,
            "processed_rows": 2,
            "empty_row_indices": [2],
            "rows": [{"row_index": 1, "score": 0.42, "candidate": "Candidate A"}],
        },
    )
    state = train_v7._load_teacher_state_v7(state_path)
    assert state["status"] == "partial"
    assert state["processed_rows"] == 2
    assert state["rows"][1][1] == "Candidate A"
    assert 2 in state["empty_row_indices"]


def test_v7_network_builds_and_runs():
    model = OmniCollectiveNetV7(
        vocab_size=48,
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=13,
        num_vision_classes=5,
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=24,
        text_hidden=40,
        image_channels=12,
        word_buckets=256,
        word_embed_dim=16,
        deep_text_channels=48,
        deep_image_channels=20,
        fusion_hidden=80,
        memory_slots=4,
        depth_steps=2,
        expert_count=2,
        expert_hidden=96,
        context_top_k=2,
        expert_top_k=1,
    )
    outputs = model(
        torch.randint(0, 48, (2, 40)),
        torch.randn(2, 3, 64, 64),
        torch.ones(2),
        torch.randint(0, 256, (2, 14)),
        torch.randn(2, 12),
    )
    assert outputs["intent"].shape == (2, len(OMNI_INTENTS_V2))
    assert outputs["response"].shape == (2, 13)
    assert outputs["domain"].shape == (2, len(OMNI_DOMAIN_LABELS_V2))
    assert float(outputs["balance_loss"].item()) >= 0.0
