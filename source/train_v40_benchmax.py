#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v5_model import OmniCollectiveEngineV5, OmniCollectiveNetV5
    from train_image_recognition_model import ensure_base_images
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, split_rows
    from train_omni_collective_v4 import _load_expanded_state_from_zip, _train_stage
    from v40_benchmax_common import build_ablation_pack, load_manifest
except ImportError:  # pragma: no cover
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v5_model import OmniCollectiveEngineV5, OmniCollectiveNetV5
    from .train_image_recognition_model import ensure_base_images
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, split_rows
    from .train_omni_collective_v4 import _load_expanded_state_from_zip, _train_stage
    from .v40_benchmax_common import build_ablation_pack, load_manifest


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v5_frontier_20260330.zip")
DEFAULT_QWEN_ADAPTER_ZIP = DEFAULT_MODELS_DIR / "qwen_supermix_enhanced_v28_cloud_plus_runpod_budget_final_adapter.zip"


def _convert_rows(rows: Sequence[Dict[str, Any]], *, limit: int = 0) -> List[OmniRow]:
    out: List[OmniRow] = []
    for row in rows:
        out.append(
            OmniRow(
                prompt=str(row.get("prompt") or ""),
                intent=str(row.get("intent") or "general"),
                response_text=str(row.get("response_text") or ""),
                domain=str(row.get("domain") or "general"),
                source=str(row.get("source") or "v40_benchmax"),
            )
        )
        if limit > 0 and len(out) >= limit:
            break
    return out


def _load_research_pack_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if not cooked:
                continue
            payload = json.loads(cooked)
            if isinstance(payload, dict):
                rows.append(dict(payload))
    return rows


def _model_dims(head_recipe: Dict[str, Any]) -> Dict[str, int]:
    keys = ("base_embed_dim", "text_hidden", "image_channels", "word_buckets", "word_embed_dim", "deep_text_channels", "deep_image_channels", "fusion_hidden", "memory_slots", "depth_steps", "expert_count", "expert_hidden", "context_top_k", "expert_top_k")
    return {key: int(head_recipe[key]) for key in keys}


def _loss_weights(head_recipe: Dict[str, Any], stage: str) -> Dict[str, float]:
    key = f"{stage}_loss_weights"
    return {name: float(value) for name, value in dict(head_recipe[key]).items()}


def _data_recipe_key(name: str) -> str:
    lowered = str(name or "").strip().lower()
    if "v39" in lowered:
        return "v39"
    return "v33"


def train_ablation(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    base_zip: Path,
    adapter_zip: Path,
    ablation_id: str,
    seed: int,
    dry_run: bool,
    research_pack_jsonl: Path | None = None,
) -> Dict[str, Any]:
    manifest = load_manifest()
    ablation_map = {item["ablation_id"]: item for item in manifest["ablation_matrix"]}
    if ablation_id not in ablation_map:
        raise KeyError(f"Unknown ablation_id: {ablation_id}")
    ablation = ablation_map[ablation_id]
    head_recipe = dict(manifest["head_recipes"][ablation["head_recipe"]])

    data_recipe_key = _data_recipe_key(str(ablation.get("data_recipe") or ablation.get("data_family") or "v33"))
    rows_dict, data_summary = build_ablation_pack(
        repo_root,
        ablation_id,
        seed=seed,
        sample_size=int(manifest["data_recipes"][data_recipe_key]["sample_size"]),
    )
    research_rows_dict: List[Dict[str, Any]] = []
    if research_pack_jsonl and research_pack_jsonl.exists():
        research_rows_dict = _load_research_pack_rows(research_pack_jsonl)
        rows_dict = list(rows_dict) + list(research_rows_dict)
        data_summary = dict(data_summary)
        data_summary["research_pack_jsonl"] = str(research_pack_jsonl)
        data_summary["research_pack_rows"] = len(research_rows_dict)
    rows = _convert_rows(rows_dict)
    train_rows, val_rows = split_rows(rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in rows if row.response_text})
    max_len = 320
    max_words = 72
    word_buckets = int(head_recipe["word_buckets"])
    image_size = 128
    if dry_run:
        return {
            "ablation_id": ablation_id,
            "data_summary": data_summary,
            "head_recipe": head_recipe,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "stage1_rows": len(train_stage1),
            "vocab_size": len(vocab),
            "response_bank": len(response_bank),
            "research_pack_rows": len(research_rows_dict),
            "dry_run": True,
        }

    device = torch.device("cpu")
    model = OmniCollectiveNetV5(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        **_model_dims(head_recipe),
    ).to(device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    stage1 = _train_stage(
        model=model,
        train_rows=train_stage1,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=32,
        learning_rate=0.00042,
        epochs=1,
        seed=seed + 101,
        device=device,
        loss_weights=_loss_weights(head_recipe, "stage1"),
        balance_weight=float(head_recipe["balance_weight"]),
    )
    stage2 = _train_stage(
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=16,
        learning_rate=0.00020,
        epochs=1,
        seed=seed + 151,
        device=device,
        loss_weights=_loss_weights(head_recipe, "stage2"),
        balance_weight=float(head_recipe["balance_weight"]) * 1.1,
    )
    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_v40_benchmax_{ablation_id}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v40_benchmax.pth"
    meta_path = artifact_dir / "omni_collective_v40_benchmax_meta.json"
    summary_path = artifact_dir / "omni_collective_v40_benchmax_summary.json"
    zip_path = output_dir / f"supermix_v40_benchmax_{ablation_id}_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(repo_root / "output" / "science_vision_dataset" / "images")
    meta = {
        "architecture_version": 40,
        "family": "v40_benchmax",
        "ablation_id": ablation_id,
        "data_recipe": ablation["data_recipe"],
        "head_recipe": ablation["head_recipe"],
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": image_size,
        **_model_dims(head_recipe),
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "research_pack_rows": len(research_rows_dict),
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "deliberation_passes": int(head_recipe["deliberation_passes"]),
        "notes": [
            "v40_benchmax is an isolated benchmark-maximization line with explicit 2x2 data/head ablations.",
            "The head recipe controls routed depth and inference deliberation, while the data recipe controls curated v33-style or benchmark-heavy v39-style rows.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV5(weights_path=weights_path, meta_path=meta_path, device=device)
    summary = {
        "artifact": zip_path.name,
        "ablation_id": ablation_id,
        "parameter_count": parameter_count,
        "data_summary": data_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "sample_outputs": [
            {"prompt": prompt, "answer": engine.answer(prompt)}
            for prompt in (
                "Debug this Python traceback and explain the likely root cause.",
                "Write a simple OpenSCAD example for a hollow box with 2 mm walls.",
                "Answer the multiple-choice science question and end with a final answer letter.",
            )
        ],
        "head_recipe": head_recipe,
        "notes": [
            "The benchmark-maximization line is intended to beat current common-benchmark leaders through explicit ablations, hard-example replay, collective distillation, and checkpoint soup.",
            "The v33-style recipe emphasizes curated paper-note and local prompt rows; the v39-style recipe emphasizes reasoning and benchmark verification rows.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "data_summary": data_summary,
        "head_recipe": head_recipe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a v40_benchmax ablation candidate.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--qwen_adapter_zip", default=str(DEFAULT_QWEN_ADAPTER_ZIP))
    parser.add_argument("--ablation_id", required=True)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--research_pack_jsonl", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    result = train_ablation(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        base_zip=Path(args.base_zip).resolve(),
        adapter_zip=Path(args.qwen_adapter_zip).resolve(),
        ablation_id=str(args.ablation_id),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        research_pack_jsonl=Path(args.research_pack_jsonl).resolve() if str(args.research_pack_jsonl).strip() else None,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
