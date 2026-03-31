from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v4_model import OmniCollectiveEngineV4, OmniCollectiveNetV4
    from train_image_recognition_model import ensure_base_images
    from train_omni_collective_v2 import (
        DEFAULT_MODELS_DIR,
        OmniDatasetV2,
        OmniRow,
        _curated_rows,
        _image_prompt_rows,
        _language_rows,
        _math_rows,
        _model_selection_rows,
        _normalize,
        _rows_from_jsonl,
        _science_rows,
        _weighted_score,
        evaluate,
        split_rows,
    )
    from train_omni_collective_v3 import (
        LocalQwenGenerator,
        _extract_qwen_adapter_dir,
        _resolve_local_qwen_base_model,
        _score_candidate,
        _three_d_rows,
        _video_rows_with_contact_sheets,
    )
except ImportError:  # pragma: no cover
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v4_model import OmniCollectiveEngineV4, OmniCollectiveNetV4
    from .train_image_recognition_model import ensure_base_images
    from .train_omni_collective_v2 import (
        DEFAULT_MODELS_DIR,
        OmniDatasetV2,
        OmniRow,
        _curated_rows,
        _image_prompt_rows,
        _language_rows,
        _math_rows,
        _model_selection_rows,
        _normalize,
        _rows_from_jsonl,
        _science_rows,
        _weighted_score,
        evaluate,
        split_rows,
    )
    from .train_omni_collective_v3 import (
        LocalQwenGenerator,
        _extract_qwen_adapter_dir,
        _resolve_local_qwen_base_model,
        _score_candidate,
        _three_d_rows,
        _video_rows_with_contact_sheets,
    )


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v3_frontier_20260329.zip")
DEFAULT_QWEN_ADAPTER_ZIP = Path(r"C:\Users\kai99\Desktop\models\qwen_supermix_enhanced_v28_cloud_plus_runpod_budget_final_adapter.zip")


def _image_prompt_rows_v4(seed: int, limit: int) -> List[OmniRow]:
    rows = list(_image_prompt_rows(seed=seed + 7, limit=max(0, int(limit) // 2)))
    rng = random.Random(int(seed))
    subjects = [
        "storm-battered lighthouse on black volcanic cliffs",
        "neon arcade seen through rain-specked glass",
        "moonlit greenhouse with rare orchids",
        "documentary portrait of a watchmaker",
        "forest research outpost after rainfall",
        "sunlit observatory packed with brass instruments",
        "ultra-clean robotics lab with reflective floors",
        "snowbound alpine rescue cabin at dusk",
    ]
    styles = ["photorealistic", "cinematic still", "documentary photo", "editorial photograph", "architectural photography"]
    cameras = ["35mm lens", "50mm lens", "85mm portrait lens", "wide establishing shot", "shallow depth of field"]
    lights = ["soft overcast light", "dramatic side lighting", "misty ambient glow", "high-contrast twilight", "warm sunrise rim light"]
    details = ["realistic textures", "natural color grading", "believable materials", "high dynamic range detail", "polished composition"]
    while len(rows) < int(limit):
        subject = rng.choice(subjects)
        rows.append(
            OmniRow(
                prompt=f"Create a strong photo-generation prompt for {subject}.",
                intent="image_prompt",
                response_text=f"{rng.choice(styles)} {subject}, {rng.choice(cameras)}, {rng.choice(lights)}, {rng.choice(details)}, vivid but realistic atmosphere.",
                domain="image_prompt",
                source="image_prompt_v4",
            )
        )
    return rows[: int(limit)]


def _routing_repair_rows() -> List[OmniRow]:
    items = [
        ("Make a photorealistic image prompt for a storm-battered lighthouse.", "photorealistic storm-battered lighthouse on jagged sea cliffs, heavy spray, dark storm clouds, cinematic lighting, wet stone texture, realistic ocean detail"),
        ("Create a clean product-photo prompt for a silver mechanical keyboard.", "luxury product photo of a silver mechanical keyboard, studio lighting, matte desk surface, sharp keycap detail, realistic reflections, premium packaging aesthetic"),
        ("Describe the likely cause of this Python import error in plain language.", "The import is probably failing because the package is missing, the module path is wrong, or the code is running in the wrong environment."),
        ("Explain the phrase break the ice in simple language.", "It means helping people feel less awkward so a conversation can start more easily."),
    ]
    return [
        OmniRow(
            prompt=prompt,
            intent="image_prompt" if "prompt" in prompt.lower() else "language",
            response_text=response,
            domain="image_prompt" if "prompt" in prompt.lower() else "language",
            source="routing_repair",
        )
        for prompt, response in items
    ]


def _build_large_rows_v4(repo_root: Path, models_dir: Path, images_dir: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    counts: Dict[str, int] = defaultdict(int)
    rows.extend(_curated_rows())
    rows.extend(_routing_repair_rows())
    rows.extend(_model_selection_rows(models_dir))
    rows.extend(_language_rows())
    rows.extend(_image_prompt_rows_v4(seed=seed + 11, limit=340))
    rows.extend(_math_rows(limit=56))

    dataset_specs = [
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 520, "coding", "coding"),
        ("conversation_data.world_events_2026_02_19.jsonl", 140, "knowledge", "world_events"),
        ("conversation_data.hybrid_v6_live_knowledge.jsonl", 860, "general", "hybrid"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 620, "language", "books"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 940, "creative", "creative"),
        ("conversation_data.mega_creative_100k.jsonl", 280, "creative", "creative_100k"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 1400, "knowledge", "reasoning"),
        ("conversation_data.supermix_plus_v27_500k.jsonl", 2200, "knowledge", "supermix_plus"),
        ("conversation_data.quality_anchor_v2.jsonl", 60, "general", "quality_anchor"),
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 220, "knowledge", "delta_anchor"),
        ("conversation_data.delta_official_refresh_2026_03_26.jsonl", 60, "knowledge", "delta_official"),
        ("conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 180, "language", "dictionary"),
        ("conversation_data.bible_kjv_public_domain_smoke.jsonl", 180, "language", "bible"),
        ("conversation_data.finnegans_wake_study_noninfringing_smoke.jsonl", 120, "creative", "finnegans"),
        ("conversation_data.science_essentials_smoke.jsonl", 160, "knowledge", "science"),
        ("conversation_data.science_novel_examples_smoke.jsonl", 120, "knowledge", "science_novel"),
    ]
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(rows), domain=domain, source_tag=source_tag)
        rows.extend(sampled)
        counts[source_tag] += len(sampled)

    science_rows = _science_rows(images_dir, repeats=6, seed=seed + 101)
    rows.extend(science_rows)
    counts["science_image"] += len(science_rows)

    three_d_rows = _three_d_rows(repo_root / "output" / "omni_v4_generated", repeats=10, seed=seed + 131)
    rows.extend(three_d_rows)
    counts["3d"] += len(three_d_rows)

    video_rows = _video_rows_with_contact_sheets(repo_root / "output" / "omni_v4_generated", repeats=16, seed=seed + 151)
    rows.extend(video_rows)
    counts["video_contact"] += len(video_rows)
    return rows, dict(sorted(counts.items()))


def _teacher_league_distill_rows(rows: Sequence[OmniRow], *, adapter_zip: Path, seed: int, limit: int) -> Tuple[List[OmniRow], Dict[str, object]]:
    eligible = [row for row in rows if row.domain in {"coding", "knowledge", "language", "creative", "planning", "general", "image_prompt", "model_selection"}]
    grouped: Dict[str, List[OmniRow]] = defaultdict(list)
    for row in eligible:
        grouped[row.domain].append(row)
    rng = random.Random(int(seed))
    sample: List[OmniRow] = []
    per_domain = max(6, int(limit) // max(1, len(grouped)))
    for domain, items in sorted(grouped.items()):
        cooked = list(items)
        rng.shuffle(cooked)
        sample.extend(cooked[:per_domain])
    rng.shuffle(sample)
    sample = sample[: int(limit)]

    qwen_base = _resolve_local_qwen_base_model()
    accepted: List[OmniRow] = []
    direct_counts: Dict[str, int] = defaultdict(int)
    repair_counts: Dict[str, int] = defaultdict(int)
    discarded = 0
    with tempfile.TemporaryDirectory(prefix="omni_v4_qwen_") as tmpdir:
        tmp_root = Path(tmpdir)
        adapter_dir = _extract_qwen_adapter_dir(adapter_zip, tmp_root)
        engines = {
            "qwen_base": LocalQwenGenerator(base_model=qwen_base, adapter_dir=tmp_root / "__base__", device="cpu"),
            "qwen_v28": LocalQwenGenerator(base_model=qwen_base, adapter_dir=adapter_dir, device="cpu"),
        }
        try:
            for index, row in enumerate(sample, start=1):
                best: Tuple[float, str, str] | None = None
                for teacher_name, engine in engines.items():
                    candidate = str(engine.generate(row.prompt, max_new_tokens=72) or "").strip()
                    if not candidate:
                        continue
                    score = float(_score_candidate(row.response_text, candidate)["composite"])
                    if best is None or score > best[0]:
                        best = (score, teacher_name, candidate)
                if best is None:
                    discarded += 1
                elif best[0] >= 0.21:
                    accepted.append(OmniRow(prompt=row.prompt, intent=row.intent, response_text=_normalize(best[2], 420), domain=row.domain, image_path=row.image_path, vision_label=row.vision_label, source=f"{best[1]}_distill"))
                    direct_counts[best[1]] += 1
                elif best[0] >= 0.08:
                    accepted.append(
                        OmniRow(
                            prompt=_normalize(f"Fix and improve this draft answer so it is concise and correct.\nRequest: {row.prompt}\nDraft: {best[2]}", 320),
                            intent=row.intent,
                            response_text=row.response_text,
                            domain=row.domain,
                            image_path=row.image_path,
                            vision_label=row.vision_label,
                            source=f"{best[1]}_repair",
                        )
                    )
                    repair_counts[best[1]] += 1
                else:
                    discarded += 1
                if index % 8 == 0 or index == len(sample):
                    print(json.dumps({"event": "teacher_league_progress", "completed": index, "total": len(sample), "accepted": len(accepted), "discarded": discarded}, ensure_ascii=True), flush=True)
        finally:
            del engines
            gc.collect()
    return accepted, {
        "requested": int(limit),
        "sampled": len(sample),
        "accepted_total": len(accepted),
        "accepted_direct": dict(sorted(direct_counts.items())),
        "accepted_repair": dict(sorted(repair_counts.items())),
        "discarded": discarded,
        "adapter_zip": str(adapter_zip),
        "base_model": str(qwen_base),
    }


def build_training_rows(*, repo_root: Path, models_dir: Path, images_dir: Path, adapter_zip: Path, seed: int, qwen_distill_limit: int) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    full_rows, source_counts = _build_large_rows_v4(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    qwen_rows, qwen_summary = _teacher_league_distill_rows(full_rows, adapter_zip=adapter_zip, seed=seed + 211, limit=qwen_distill_limit)
    full_rows.extend(qwen_rows)
    source_counts["qwen_teacher_league_total"] = len(qwen_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    return stage1_rows, list(full_rows), {"stage1_rows": len(stage1_rows), "stage2_rows": len(full_rows), "source_counts": dict(sorted(source_counts.items())), "teacher_league": qwen_summary}


def _inflate_tensor(source: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    if source.ndim != target.ndim:
        return None
    result = target.detach().cpu().clone()
    slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(source.shape, target.shape))
    result[slices] = source.detach().cpu()[slices]
    return result


def _load_expanded_state_from_zip(model: OmniCollectiveNetV4, base_zip: Path) -> Dict[str, object]:
    if not base_zip.exists():
        return {"loaded": False, "reason": f"Missing base zip: {base_zip}"}
    with tempfile.TemporaryDirectory(prefix="omni_v4_base_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(tmp_path)
        weight_candidates = sorted(tmp_path.glob("**/*.pth"))
        if not weight_candidates:
            return {"loaded": False, "reason": f"No .pth weights found inside {base_zip.name}"}
        state = torch.load(weight_candidates[0], map_location="cpu", weights_only=False)
        model_state = model.state_dict()
        compatible: Dict[str, torch.Tensor] = {}
        exact_keys = 0
        partial_keys = 0
        for key, target in model_state.items():
            source = state.get(key)
            if source is None:
                continue
            if tuple(source.shape) == tuple(target.shape):
                compatible[key] = source.detach().cpu()
                exact_keys += 1
            else:
                inflated = _inflate_tensor(source, target)
                if inflated is not None:
                    compatible[key] = inflated
                    partial_keys += 1
        info = model.load_state_dict(compatible, strict=False)
        return {"loaded": True, "source_weights": weight_candidates[0].name, "exact_match_keys": exact_keys, "partial_resize_keys": partial_keys, "missing_key_count": len(info.missing_keys), "unexpected_key_count": len(info.unexpected_keys)}


def _train_stage(*, model: OmniCollectiveNetV4, train_rows: Sequence[OmniRow], val_rows: Sequence[OmniRow], vocab: Dict[str, int], response_bank: Sequence[str], image_size: int, max_len: int, max_words: int, word_buckets: int, batch_size: int, learning_rate: float, epochs: int, seed: int, device: torch.device, loss_weights: Dict[str, float], balance_weight: float) -> Dict[str, object]:
    response_to_index = {text: idx for idx, text in enumerate(response_bank)}
    train_ds = OmniDatasetV2(train_rows, vocab=vocab, max_len=max_len, image_size=image_size, max_words=max_words, word_buckets=word_buckets, response_to_index=response_to_index)
    val_ds = OmniDatasetV2(val_rows, vocab=vocab, max_len=max_len, image_size=image_size, max_words=max_words, word_buckets=word_buckets, response_to_index=response_to_index)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.022)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs) * max(1, len(train_loader))))
    intent_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    response_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    vision_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.01)
    domain_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_state = None
    best_score = -1.0
    history: List[Dict[str, float]] = []
    print(json.dumps({"event": "stage_start", "train_rows": len(train_rows), "val_rows": len(val_rows), "response_bank": len(response_bank), "batch_size": int(batch_size), "learning_rate": float(learning_rate), "epochs": int(epochs), "loss_weights": loss_weights, "balance_weight": float(balance_weight)}, ensure_ascii=True), flush=True)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch["token_ids"].to(device), batch["image_tensor"].to(device), batch["has_image"].to(device), batch["word_ids"].to(device), batch["prompt_features"].to(device))
            loss = float(loss_weights["intent"]) * intent_loss_fn(outputs["intent"], batch["intent"].to(device)) + float(loss_weights["response"]) * response_loss_fn(outputs["response"], batch["response"].to(device)) + float(loss_weights["domain"]) * domain_loss_fn(outputs["domain"], batch["domain"].to(device))
            if bool(batch["vision"].ge(0).any()):
                loss = loss + float(loss_weights["vision"]) * vision_loss_fn(outputs["vision"], batch["vision"].to(device))
            if "balance_loss" in outputs:
                loss = loss + float(balance_weight) * outputs["balance_loss"]
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item()) * int(batch["intent"].size(0))
            total_items += int(batch["intent"].size(0))
        val_metrics = evaluate(model, val_loader, device)
        score = _weighted_score(val_metrics)
        history.append({"epoch": float(epoch), "train_loss": float(total_loss / max(total_items, 1)), "val_intent_accuracy": val_metrics["intent_accuracy"], "val_response_accuracy": val_metrics["response_accuracy"], "val_vision_accuracy": val_metrics["vision_accuracy"], "val_domain_accuracy": val_metrics["domain_accuracy"], "score": score, "lr": float(scheduler.get_last_lr()[0])})
        print(json.dumps({"event": "epoch_end", **history[-1]}, ensure_ascii=True), flush=True)
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model state captured during training stage.")
    model.load_state_dict(best_state)
    return {"history": history, "best_score": best_score, "val_metrics": evaluate(model, val_loader, device), "train_rows": len(train_rows), "val_rows": len(val_rows)}


def train_model(*, repo_root: Path, output_dir: Path, models_dir: Path, base_zip: Path, adapter_zip: Path, images_dir: Path, image_size: int, batch_size: int, stage1_epochs: int, stage2_epochs: int, stage1_lr: float, stage2_lr: float, seed: int, qwen_distill_limit: int) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    _stage1_rows, full_rows, dataset_summary = build_training_rows(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, adapter_zip=adapter_zip, seed=seed, qwen_distill_limit=qwen_distill_limit)
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})
    print(json.dumps({"event": "label_space", "train_rows": len(train_rows), "val_rows": len(val_rows), "response_bank": len(response_bank), "vocab_size": len(vocab)}, ensure_ascii=True), flush=True)
    max_len = 288
    max_words = 64
    word_buckets = 12288
    device = torch.device("cpu")
    model = OmniCollectiveNetV4(vocab_size=max(len(vocab), 2), num_intents=len(OMNI_INTENTS_V2), num_responses=max(len(response_bank), 1), num_vision_classes=len(SCIENCE_IMAGE_CLASSES), num_domains=len(OMNI_DOMAIN_LABELS_V2)).to(device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)
    stage1 = _train_stage(model=model, train_rows=train_stage1, val_rows=val_rows, vocab=vocab, response_bank=response_bank, image_size=image_size, max_len=max_len, max_words=max_words, word_buckets=word_buckets, batch_size=batch_size, learning_rate=stage1_lr, epochs=stage1_epochs, seed=seed + 101, device=device, loss_weights={"intent": 0.58, "response": 1.00, "domain": 0.72, "vision": 0.72}, balance_weight=0.025)
    stage2 = _train_stage(model=model, train_rows=train_rows, val_rows=val_rows, vocab=vocab, response_bank=response_bank, image_size=image_size, max_len=max_len, max_words=max_words, word_buckets=word_buckets, batch_size=max(24, int(batch_size) // 2), learning_rate=stage2_lr, epochs=stage2_epochs, seed=seed + 151, device=device, loss_weights={"intent": 0.50, "response": 1.00, "domain": 0.68, "vision": 1.12}, balance_weight=0.035)
    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v4_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v4_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v4_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v4_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v4_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name
    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {"architecture_version": 4, "vocab": vocab, "response_bank": response_bank, "class_info": class_info, "intent_labels": list(OMNI_INTENTS_V2), "domain_labels": list(OMNI_DOMAIN_LABELS_V2), "max_len": max_len, "image_size": int(image_size), "embed_dim": 84, "text_hidden": 160, "image_channels": 36, "word_buckets": word_buckets, "max_words": max_words, "word_embed_dim": 72, "deep_text_channels": 192, "deep_image_channels": 64, "fusion_hidden": 576, "memory_slots": 12, "depth_steps": 5, "expert_count": 4, "expert_hidden": 896, "context_top_k": 3, "expert_top_k": 2, "parameter_count": parameter_count, "warm_start": warm_start, "stage1": stage1, "stage2": stage2, "seed": int(seed)}
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV4(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = ["Debug this Python stack trace and explain the root cause.", "Describe the key motion shown across these video frames.", "Make a photorealistic image prompt for a storm-battered lighthouse.", "Explain the phrase break the ice in simple language."]
    summary = {"artifact": zip_path.name, "parameter_count": parameter_count, "dataset_summary": dataset_summary, "warm_start": warm_start, "stage1": stage1, "stage2": stage2, "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts], "notes": ["v4 adds shared sparse experts, per-depth modality-context routing, balance regularization, and partial tensor warm-start from v3.", "The dataset is wider than v3 across coding, knowledge, image prompts, science images, 3D rows, video contact sheets, and Qwen teacher-league repair rows."]}
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {"zip_path": str(zip_path), "desktop_zip_path": str(desktop_zip_path), "artifact_dir": str(artifact_dir), "parameter_count": parameter_count, "stage1_val": stage1["val_metrics"], "stage2_val": stage2["val_metrics"], "warm_start": warm_start, "dataset_summary": dataset_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the upgraded omni_collective_v4 frontier model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--qwen_adapter_zip", default=str(DEFAULT_QWEN_ADAPTER_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=0.00072)
    parser.add_argument("--stage2_lr", type=float, default=0.00034)
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--qwen_distill_limit", type=int, default=36)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(repo_root=Path(args.repo_root).resolve(), output_dir=Path(args.output_dir).resolve(), models_dir=Path(args.models_dir).resolve(), base_zip=Path(args.base_zip).resolve(), adapter_zip=Path(args.qwen_adapter_zip).resolve(), images_dir=Path(args.images_dir).resolve(), image_size=int(args.image_size), batch_size=int(args.batch_size), stage1_epochs=int(args.stage1_epochs), stage2_epochs=int(args.stage2_epochs), stage1_lr=float(args.stage1_lr), stage2_lr=float(args.stage2_lr), seed=int(args.seed), qwen_distill_limit=int(args.qwen_distill_limit))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
