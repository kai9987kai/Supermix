#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    import train_omni_collective_v8 as train_v8
    from omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8
except ImportError:  # pragma: no cover
    from . import train_omni_collective_v8 as train_v8
    from .omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8


def _resolve_checkpoint_from_state(state_path: Path) -> Tuple[Dict[str, object], Path]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(str(state["checkpoint_path"])).resolve()
    return state, checkpoint_path


def _copy_snapshot(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = destination_dir / source_path.name
    last_error: Optional[Exception] = None
    for _ in range(8):
        try:
            shutil.copy2(source_path, snapshot_path)
            return snapshot_path
        except Exception as exc:  # pragma: no cover - windows file races are timing-dependent
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Failed to snapshot checkpoint {source_path}: {last_error}")


def _select_state_dict(checkpoint: Dict[str, object]) -> Dict[str, torch.Tensor]:
    for key in ("ema_state", "best_state", "model_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value:
            return value
    raise RuntimeError("Checkpoint did not contain a usable model state")


def export_preview(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    images_dir: Path,
    state_path: Path,
    checkpoint_path: Optional[Path],
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
    device: str,
) -> Dict[str, object]:
    if checkpoint_path is None:
        state_payload, checkpoint_path = _resolve_checkpoint_from_state(state_path)
    else:
        state_payload = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        checkpoint_path = checkpoint_path.resolve()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = (output_dir / f"supermix_omni_collective_v8_preview_{stamp}").resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_snapshot = _copy_snapshot(checkpoint_path, artifact_dir)
    checkpoint = torch.load(checkpoint_snapshot, map_location="cpu")
    state_dict = _select_state_dict(checkpoint)

    _, full_rows, dataset_summary = train_v8.build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=int(seed),
        distill_limit=int(distill_limit),
        teacher_model_limit=int(teacher_model_limit),
    )
    train_rows, _ = train_v8.split_rows(full_rows, seed=int(seed) + 17)
    vocab = train_v8.build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})

    model = OmniCollectiveNetV8(
        vocab_size=max(len(vocab), 2),
        num_intents=len(train_v8.OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(train_v8.SCIENCE_IMAGE_CLASSES),
        num_domains=len(train_v8.OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=132,
        text_hidden=272,
        image_channels=56,
        word_buckets=16384,
        word_embed_dim=120,
        deep_text_channels=384,
        deep_image_channels=128,
        fusion_hidden=1088,
        memory_slots=28,
        depth_steps=11,
        expert_count=10,
        expert_hidden=1792,
        context_top_k=4,
        expert_top_k=2,
    )
    model.load_state_dict(state_dict)
    model.eval()

    weights_path = artifact_dir / "omni_collective_v8_preview.pth"
    meta_path = artifact_dir / "omni_collective_v8_preview_meta.json"
    summary_path = artifact_dir / "omni_collective_v8_preview_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v8_preview_{stamp}.zip"

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = train_v8.ensure_base_images(images_dir)

    stage_resume_dir = checkpoint_path.parent
    stage1_meta_path = stage_resume_dir / "stage1_complete.json"
    stage1_meta = json.loads(stage1_meta_path.read_text(encoding="utf-8")) if stage1_meta_path.exists() else None
    preview_state = {
        "stage_name": str(checkpoint.get("stage_name") or ""),
        "epoch": int(checkpoint.get("epoch") or 0),
        "next_batch_index": int(checkpoint.get("next_batch_index") or 0),
        "optimizer_steps_done": int(checkpoint.get("optimizer_steps_done") or 0),
        "saved_at": str(checkpoint.get("saved_at") or ""),
    }
    meta = {
        "architecture_version": 8,
        "preview": True,
        "preview_source": {
            "state_path": str(state_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_snapshot": str(checkpoint_snapshot),
            "state_payload": state_payload,
            "checkpoint_preview_state": preview_state,
        },
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(train_v8.OMNI_INTENTS_V2),
        "domain_labels": list(train_v8.OMNI_DOMAIN_LABELS_V2),
        "max_len": 384,
        "image_size": 128,
        "embed_dim": 132,
        "text_hidden": 272,
        "image_channels": 56,
        "word_buckets": 16384,
        "max_words": 84,
        "word_embed_dim": 120,
        "deep_text_channels": 384,
        "deep_image_channels": 128,
        "fusion_hidden": 1088,
        "memory_slots": 28,
        "depth_steps": 11,
        "expert_count": 10,
        "expert_hidden": 1792,
        "context_top_k": 4,
        "expert_top_k": 2,
        "parameter_count": parameter_count,
        "seed": int(seed),
        "stage1": stage1_meta,
        "stage2_preview": preview_state,
        "dataset_summary": dataset_summary,
        "deliberation_passes": 10,
        "minimum_passes": 5,
        "grounding_threshold": 0.54,
        "prompt_understanding_mode": "all_model_multitype_grounded_consensus_math_protein_materials_three_d_conversation",
        "notes": [
            "Preview export cut from the latest resumable stage2 checkpoint while the main training run continues.",
            "This artifact is for benchmarking and inspection only; the full run remains the source of truth.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    engine = OmniCollectiveEngineV8(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))
    sample_prompts = [
        "Reply in exactly two sentences explaining why regression tests matter.",
        "Solve 3*x + 7 = 19.",
        "Which local model is best for benchmark-focused reasoning prompts?",
        "Write a tiny OpenSCAD snippet for a centered cylinder with a hole.",
    ]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "preview": True,
        "dataset_summary": dataset_summary,
        "preview_source": meta["preview_source"],
        "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
        archive.write(checkpoint_snapshot, arcname=checkpoint_snapshot.name)

    return {
        "artifact_dir": str(artifact_dir),
        "zip_path": str(zip_path),
        "checkpoint_snapshot": str(checkpoint_snapshot),
        "preview_batch_index": int(checkpoint.get("next_batch_index") or 0),
        "preview_saved_at": str(checkpoint.get("saved_at") or ""),
        "parameter_count": parameter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a benchmarkable omni_collective_v8 preview from the latest resumable checkpoint.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(train_v8.DEFAULT_MODELS_DIR))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--state_path", default="output/omni_collective_v8_train_state.json")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--distill_limit", type=int, default=160)
    parser.add_argument("--teacher_model_limit", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = export_preview(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        state_path=Path(args.state_path).resolve(),
        checkpoint_path=Path(args.checkpoint_path).resolve() if str(args.checkpoint_path).strip() else None,
        seed=int(args.seed),
        distill_limit=int(args.distill_limit),
        teacher_model_limit=int(args.teacher_model_limit),
        device=str(args.device),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
