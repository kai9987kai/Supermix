from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(os.path.join(os.getcwd(), "source"))

from multimodel_catalog import ModelRecord
from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2
from omni_collective_v8_model import OmniCollectiveNetV8, _prompt_variants_v8
from train_omni_collective_v2 import OmniRow
import train_omni_collective_v8 as train_v8


def test_prompt_variants_v8_adds_grounding_math_protein_and_materials_paths():
    variants = _prompt_variants_v8(
        "Solve 3*x + 7 = 19, explain alpha helices, and suggest a perovskite CIF seed.",
        "x = 4",
        response_confidence=0.18,
        domain_confidence=0.22,
    )
    joined = "\n".join(variants).lower()
    assert "ground the answer" in joined
    assert "hallucination guard" in joined
    assert "may involve math" in joined
    assert "protein folding" in joined
    assert "materials or crystal generation" in joined
    assert len(variants) >= 7


def test_specialist_profile_rows_v8_cover_chat_and_non_chat_models(monkeypatch, tmp_path):
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
    monkeypatch.setattr(train_v8, "discover_model_records", lambda models_dir: list(records))
    rows, summary = train_v8._specialist_profile_rows_v7(Path.cwd(), tmp_path, seed=7)
    prompts = "\n".join(row.prompt for row in rows).lower()
    responses = "\n".join(row.response_text for row in rows).lower()
    assert summary["record_count"] == 2
    assert "dcgan mnist" in prompts
    assert "omni collective v6 frontier" in prompts
    assert "not a normal chat consultant" in responses
    assert "fusion" in responses or "gan" in responses


def test_benchmax_rows_v8_build_from_mocked_builders(monkeypatch):
    sample_v33 = [{"user": "Solve 2+2", "assistant": "The answer is 4.", "metadata": {"benchmark": "gsm8k"}}]
    sample_v39 = [{"user": "What is BoolQ asking for?", "assistant": "A yes or no answer.", "metadata": {"benchmark": "boolq"}}]
    monkeypatch.setattr(train_v8, "build_v33_style_rows", lambda repo_root, seed, sample_size: (list(sample_v33), {"count": 1}))
    monkeypatch.setattr(train_v8, "build_v39_style_rows", lambda repo_root, seed, sample_size: (list(sample_v39), {"count": 1}))
    rows, counts = train_v8._benchmax_rows_v7(Path.cwd(), seed=19)
    assert rows
    assert counts["benchmax_v33_v8"] == 1
    assert counts["benchmax_v39_v8"] == 1
    assert any(row.domain in {"knowledge", "math", "planning", "coding"} for row in rows)


def test_dedupe_rows_v8_keeps_unique_prompt_response_pairs():
    rows = [
        OmniRow(prompt="A", intent="general", response_text="B", domain="general", source="x"),
        OmniRow(prompt="A", intent="general", response_text="B", domain="general", source="x"),
        OmniRow(prompt="A", intent="general", response_text="C", domain="general", source="x"),
    ]
    deduped = train_v8._dedupe_rows(rows)
    assert len(deduped) == 2


def test_teacher_resume_state_roundtrip(tmp_path):
    resume_dir = train_v8._teacher_resume_dir_v8(tmp_path, seed=11, limit=8, teacher_keys=["v40_benchmax", "qwen_v28"])
    sample_path = train_v8._teacher_sample_path_v8(resume_dir)
    state_path = train_v8._teacher_state_path_v8(resume_dir, "v40_benchmax")
    rows = [
        OmniRow(prompt="P1", intent="general", response_text="R1", domain="general", source="s1"),
        OmniRow(prompt="P2", intent="coding", response_text="R2", domain="coding", source="s2"),
    ]
    train_v8._save_teacher_sample_v8(sample_path, rows)
    loaded_rows = train_v8._load_teacher_sample_v8(sample_path)
    assert [row.prompt for row in loaded_rows] == ["P1", "P2"]

    train_v8._write_json_atomic(
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
    state = train_v8._load_teacher_state_v8(state_path)
    assert state["status"] == "partial"
    assert state["processed_rows"] == 2
    assert state["rows"][1][1] == "Candidate A"
    assert 2 in state["empty_row_indices"]


def test_v8_network_builds_and_runs():
    model = OmniCollectiveNetV8(
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


def test_materials_and_gemma_helpers_v8(tmp_path):
    rows = train_v8._materials_rows_v8()
    assert rows
    assert any("cif" in row.response_text.lower() or "space group" in row.response_text.lower() for row in rows)
    gemma_rows, summary = train_v8._gemma_four_slice_rows_v8(rows, seed=9, limit=4)
    assert gemma_rows == []
    assert summary["status"] == "skipped_no_local_gemma4"


def test_resolve_training_runtime_cpu_defaults(tmp_path):
    runtime = train_v8.resolve_training_runtime(
        repo_root=tmp_path,
        requested_device="cpu",
        amp_mode="auto",
        amp_dtype="auto",
        compile_requested=False,
        compile_mode="reduce-overhead",
        grad_accum_steps=4,
        ema_decay=0.95,
        warmup_steps=0,
        warmup_ratio=0.1,
        min_lr_scale=0.2,
        batch_size=8,
    )
    assert runtime.resolved_device == "cpu"
    assert runtime.device_type == "cpu"
    assert runtime.amp_enabled is False
    assert runtime.scaler_enabled is False
    assert runtime.effective_batch_size == 32
    assert runtime.ema_decay == pytest.approx(0.95)


def test_warmup_cosine_scheduler_changes_lr():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = train_v8.create_warmup_cosine_scheduler(
        optimizer,
        total_steps=6,
        warmup_steps=2,
        warmup_ratio=0.0,
        min_lr_scale=0.2,
    )
    lrs = []
    for _ in range(6):
        optimizer.step()
        scheduler.step()
        lrs.append(float(optimizer.param_groups[0]["lr"]))
    assert lrs[0] > 0.0
    assert lrs[1] >= lrs[0]
    assert lrs[-1] < lrs[1]
    assert lrs[-1] >= 0.2


def test_train_stage_resumable_v8_recovers_from_saved_progress(tmp_path):
    train_rows = [
        OmniRow(prompt="Explain why tests matter.", intent="general", response_text="Tests catch regressions early.", domain="general", source="t"),
        OmniRow(prompt="Solve 2 + 2.", intent="general", response_text="4", domain="math", source="t"),
        OmniRow(prompt="Name one protein folding force.", intent="knowledge", response_text="Hydrophobic collapse is one major force.", domain="knowledge", source="t"),
        OmniRow(prompt="Write one OpenSCAD primitive.", intent="coding", response_text="cube([1,1,1]);", domain="coding", source="t"),
    ]
    val_rows = [
        OmniRow(prompt="Why avoid hallucinations?", intent="general", response_text="Because grounded answers are more trustworthy.", domain="knowledge", source="v"),
        OmniRow(prompt="What is 3 + 4?", intent="general", response_text="7", domain="math", source="v"),
    ]
    vocab = train_v8.build_char_vocab([row.prompt for row in (train_rows + val_rows)], min_frequency=1)
    response_bank = sorted({row.response_text for row in (train_rows + val_rows)})
    stage_dir = tmp_path / "resume"
    run_state_path = tmp_path / "run_state.json"
    runtime = train_v8.resolve_training_runtime(
        repo_root=tmp_path,
        requested_device="cpu",
        amp_mode="off",
        amp_dtype="auto",
        compile_requested=False,
        compile_mode="reduce-overhead",
        grad_accum_steps=2,
        ema_decay=0.9,
        warmup_steps=0,
        warmup_ratio=0.2,
        min_lr_scale=0.1,
        batch_size=2,
    )

    model = OmniCollectiveNetV8(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
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
    with pytest.raises(RuntimeError, match="Injected stage interruption"):
        train_v8._train_stage_resumable_v8(
            model=model,
            forward_model=model,
            train_rows=train_rows,
            val_rows=val_rows,
            vocab=vocab,
            response_bank=response_bank,
            image_size=32,
            max_len=64,
            max_words=24,
            word_buckets=256,
            batch_size=2,
            learning_rate=0.001,
            epochs=1,
            seed=17,
            runtime=runtime,
            loss_weights={"intent": 0.58, "response": 1.0, "domain": 0.82, "vision": 0.78},
            balance_weight=0.01,
            stage_name="stage1",
            stage_dir=stage_dir,
            run_state_path=run_state_path,
            progress_every_batches=1,
            checkpoint_every_batches=1,
            grad_accum_steps=2,
            inject_interrupt_after_batch=1,
        )
    assert train_v8._stage_progress_path_v8(stage_dir, "stage1").exists()
    interrupted_state = __import__("json").loads(run_state_path.read_text(encoding="utf-8"))
    assert interrupted_state["stage"] == "stage1"

    resumed_model = OmniCollectiveNetV8(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
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
    result = train_v8._train_stage_resumable_v8(
        model=resumed_model,
        forward_model=resumed_model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=32,
        max_len=64,
        max_words=24,
        word_buckets=256,
        batch_size=2,
        learning_rate=0.001,
        epochs=1,
        seed=17,
        runtime=runtime,
        loss_weights={"intent": 0.58, "response": 1.0, "domain": 0.82, "vision": 0.78},
        balance_weight=0.01,
        stage_name="stage1",
        stage_dir=stage_dir,
        run_state_path=run_state_path,
        progress_every_batches=1,
        checkpoint_every_batches=1,
        grad_accum_steps=2,
    )
    assert result["train_rows"] == len(train_rows)
    assert result["val_rows"] == len(val_rows)
    assert result["runtime"]["resolved_device"] == "cpu"
    assert result["effective_batch_size"] == 4
    assert train_v8._stage_complete_weights_path_v8(stage_dir, "stage1").exists()
    assert train_v8._stage_complete_meta_path_v8(stage_dir, "stage1").exists()
    assert not train_v8._stage_progress_path_v8(stage_dir, "stage1").exists()
