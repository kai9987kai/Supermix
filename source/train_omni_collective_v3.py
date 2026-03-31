from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageSequence
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from build_3d_video_test_dataset import MODEL_BUILDERS, VIDEO_BUILDERS, _rows_for_model, _rows_for_video
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v3_model import OmniCollectiveEngineV3, OmniCollectiveNetV3
    from train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
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
except ImportError:  # pragma: no cover
    from .build_3d_video_test_dataset import MODEL_BUILDERS, VIDEO_BUILDERS, _rows_for_model, _rows_for_video
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v3_model import OmniCollectiveEngineV3, OmniCollectiveNetV3
    from .train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
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


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v2_frontier_20260328.zip")
DEFAULT_QWEN_ADAPTER_ZIP = Path(r"C:\Users\kai99\Desktop\models\qwen_supermix_enhanced_v28_cloud_plus_runpod_budget_final_adapter.zip")


def _normalize_response(text: str) -> str:
    cooked = " ".join(str(text or "").replace("<|assistant|>", "").replace("</s>", "").split())
    lowered = cooked.lower()
    for prefix in ("assistant:", "bot:", "answer:"):
        if lowered.startswith(prefix):
            cooked = " ".join(cooked[len(prefix) :].split())
            lowered = cooked.lower()
    return cooked


def _token_f1(reference: str, candidate: str) -> float:
    ref_tokens = _normalize_response(reference).lower().split()
    cand_tokens = _normalize_response(candidate).lower().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    ref_counts: Dict[str, int] = defaultdict(int)
    cand_counts: Dict[str, int] = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1
    for token in cand_tokens:
        cand_counts[token] += 1
    overlap = sum(min(ref_counts[token], cand_counts[token]) for token in ref_counts)
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(cand_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)
    return 2.0 * precision * recall / max(precision + recall, 1e-8)


def _score_candidate(reference: str, candidate: str) -> Dict[str, float]:
    from difflib import SequenceMatcher
    import re

    symbol_only_re = re.compile(r"^[-=+*/%0-9.,:;(){}\[\]\s]+$")
    word_re = re.compile(r"[A-Za-z0-9']+")
    candidate = _normalize_response(candidate)
    reference = _normalize_response(reference)
    if not candidate:
        return {"token_f1": 0.0, "char_similarity": 0.0, "penalty": 1.0, "composite": -1.0}
    f1 = float(_token_f1(reference, candidate))
    similarity = float(SequenceMatcher(None, reference.lower(), candidate.lower()).ratio())
    words = candidate.split()
    visible = sum(1 for ch in candidate if not ch.isspace())
    alpha = sum(1 for ch in candidate if ch.isalpha())
    alpha_ratio = (alpha / visible) if visible else 0.0
    tokens = [tok.lower() for tok in word_re.findall(candidate)]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0
    penalty = 0.0
    if len(words) < 5:
        penalty += 0.14
    if len(words) > 120:
        penalty += 0.03
    if symbol_only_re.fullmatch(candidate):
        penalty += 0.45
    if alpha_ratio < 0.42:
        penalty += 0.10
    if tokens and unique_ratio < 0.42:
        penalty += 0.08
    if any(fragment in candidate for fragment in ("=0", "-3", "-)", "-.")):
        penalty += 0.12
    length_bonus = 0.02 if 8 <= len(words) <= 80 else 0.0
    composite = 0.60 * f1 + 0.34 * similarity + 0.055 + length_bonus - penalty
    return {
        "token_f1": f1,
        "char_similarity": similarity,
        "penalty": penalty,
        "composite": composite,
    }


class LocalQwenGenerator:
    def __init__(self, *, base_model: Path, adapter_dir: Path, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from peft import PeftModel
        except ImportError:  # pragma: no cover
            PeftModel = None

        self.device = torch.device(device)
        self.base_model = str(base_model)
        self.adapter_dir = adapter_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, local_files_only=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
        ).to(self.device)
        if adapter_dir.exists() and PeftModel is not None:
            model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
            model = model.merge_and_unload()
            model = model.to(self.device)
        self.model = model.eval()

    def _build_prompt(self, user_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are Supermix Qwen v3 distillation teacher. Answer directly, clearly, and without hidden-reasoning narration.",
            },
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = [f"{message['role'].upper()}: {message['content']}" for message in messages]
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(self, user_text: str, max_new_tokens: int) -> str:
        prompt = self._build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max(24, int(max_new_tokens)),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            use_cache=True,
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return _normalize_response(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def _resolve_local_qwen_base_model() -> Path:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots"
    if not hub_dir.exists():
        raise FileNotFoundError(f"Missing local Qwen base model cache under {hub_dir}")
    candidates = sorted([path for path in hub_dir.iterdir() if path.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No Qwen snapshot directories found under {hub_dir}")
    return candidates[-1]


def _extract_qwen_adapter_dir(zip_path: Path, temp_root: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_root)
    adapter_dirs = sorted(temp_root.glob("**/adapter"))
    if not adapter_dirs:
        raise FileNotFoundError(f"No adapter directory found inside {zip_path}")
    return adapter_dirs[0]


def _render_video_contact_sheet(video_path: Path, output_path: Path, *, tile_size: int = 96) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(video_path) as image:
        frames = [frame.convert("RGB").copy() for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"No frames found in {video_path}")
    while len(frames) < 4:
        frames.append(frames[-1].copy())
    picks = [frames[0], frames[len(frames) // 3], frames[(2 * len(frames)) // 3], frames[-1]]
    thumbs = [frame.resize((tile_size, tile_size), Image.Resampling.BICUBIC) for frame in picks]

    motion = Image.new("RGB", picks[0].size, (0, 0, 0))
    for left, right in zip(frames, frames[1:]):
        diff = ImageChops.difference(left.convert("RGB"), right.convert("RGB"))
        motion = ImageChops.add(motion, diff, scale=1.7)
    motion = motion.resize((tile_size, tile_size), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (tile_size * 2, tile_size * 2), (18, 18, 24))
    canvas.paste(thumbs[0], (0, 0))
    canvas.paste(thumbs[1], (tile_size, 0))
    canvas.paste(thumbs[2], (0, tile_size))
    canvas.paste(motion, (tile_size, tile_size))
    canvas.save(output_path)
    return output_path


def _three_d_rows(work_dir: Path, *, repeats: int, seed: int) -> List[OmniRow]:
    rng = random.Random(int(seed))
    models_dir = work_dir / "generated_3d_models"
    rows: List[OmniRow] = []
    for stem, builder in MODEL_BUILDERS:
        path = models_dir / f"{stem}.obj"
        meta = builder(path)
        for _ in range(max(1, int(repeats))):
            for item in _rows_for_model(str(path), meta, rng):
                prompt = _normalize(str(item.get("user") or ""), 220)
                response = _normalize(str(item.get("assistant") or ""), 320)
                if prompt and response:
                    rows.append(
                        OmniRow(
                            prompt=prompt,
                            intent="spatial_3d",
                            response_text=response,
                            domain="spatial_3d",
                            source="3d",
                        )
                    )
    return rows


def _video_rows_with_contact_sheets(work_dir: Path, *, repeats: int, seed: int) -> List[OmniRow]:
    rng = random.Random(int(seed))
    videos_dir = work_dir / "generated_videos"
    sheet_dir = work_dir / "video_contact_sheets"
    rows: List[OmniRow] = []
    for filename, builder in VIDEO_BUILDERS:
        path = videos_dir / filename
        meta = builder(path)
        sheet_path = _render_video_contact_sheet(path, sheet_dir / f"{Path(filename).stem}_contact.png")
        for _ in range(max(1, int(repeats))):
            for item in _rows_for_video(str(path), meta, rng):
                prompt = _normalize(str(item.get("user") or ""), 220)
                response = _normalize(str(item.get("assistant") or ""), 300)
                if prompt and response:
                    rows.append(
                        OmniRow(
                            prompt=prompt,
                            intent="video",
                            response_text=response,
                            domain="video",
                            image_path=str(sheet_path),
                            source="video_contact",
                        )
                    )
        concept = str(meta.get("concept") or Path(filename).stem.replace("_", " "))
        caption = str(meta.get("caption") or concept)
        rows.extend(
            [
                OmniRow(
                    prompt=f"Describe the key motion shown across these video frames for {concept}.",
                    intent="video",
                    response_text=f"The frame sheet shows {caption}. Focus on the change across time rather than a single still frame.",
                    domain="video",
                    image_path=str(sheet_path),
                    source="video_contact",
                ),
                OmniRow(
                    prompt=f"What changes over time in this contact sheet for {concept}?",
                    intent="video",
                    response_text=f"Across time, the visual state shifts in a way that matches {concept}. The motion-summary tile highlights those changes.",
                    domain="video",
                    image_path=str(sheet_path),
                    source="video_contact",
                ),
            ]
        )
    return rows


def _build_large_rows(repo_root: Path, models_dir: Path, images_dir: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    source_counts: Dict[str, int] = defaultdict(int)

    rows.extend(_curated_rows())
    rows.extend(_model_selection_rows(models_dir))
    rows.extend(_language_rows())
    rows.extend(_image_prompt_rows(seed=seed + 11, limit=260))
    rows.extend(_math_rows(limit=52))

    dataset_specs = [
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 560, "coding", "coding"),
        ("conversation_data.world_events_2026_02_19.jsonl", 180, "knowledge", "world_events"),
        ("conversation_data.hybrid_v6_live_knowledge.jsonl", 600, "general", "hybrid"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 420, "language", "books"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 600, "creative", "creative"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 900, "knowledge", "reasoning"),
        ("conversation_data.supermix_plus_v27_500k.jsonl", 1500, "knowledge", "supermix_plus"),
        ("conversation_data.quality_anchor_v2.jsonl", 80, "general", "quality_anchor"),
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 220, "knowledge", "delta_anchor"),
        ("conversation_data.delta_official_refresh_2026_03_26.jsonl", 80, "knowledge", "delta_official"),
        ("conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 160, "language", "dictionary"),
        ("conversation_data.bible_kjv_public_domain_smoke.jsonl", 140, "language", "bible"),
        ("conversation_data.science_essentials_smoke.jsonl", 120, "knowledge", "science"),
    ]
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(rows), domain=domain, source_tag=source_tag)
        rows.extend(sampled)
        source_counts[source_tag] += len(sampled)

    science_rows = _science_rows(images_dir, repeats=5, seed=seed + 101)
    rows.extend(science_rows)
    source_counts["science_image"] += len(science_rows)

    three_d_rows = _three_d_rows(repo_root / "output" / "omni_v3_generated", repeats=8, seed=seed + 131)
    rows.extend(three_d_rows)
    source_counts["3d"] += len(three_d_rows)

    video_rows = _video_rows_with_contact_sheets(repo_root / "output" / "omni_v3_generated", repeats=12, seed=seed + 151)
    rows.extend(video_rows)
    source_counts["video_contact"] += len(video_rows)

    return rows, dict(sorted(source_counts.items()))


def _distill_qwen_rows(rows: Sequence[OmniRow], *, adapter_zip: Path, seed: int, limit: int) -> Tuple[List[OmniRow], Dict[str, object]]:
    chosen = [row for row in rows if row.domain in {"coding", "knowledge", "language", "planning", "general", "creative"}]
    grouped: Dict[str, List[OmniRow]] = defaultdict(list)
    for row in chosen:
        grouped[row.domain].append(row)
    rng = random.Random(int(seed))
    sample: List[OmniRow] = []
    domains = sorted(grouped)
    per_domain = max(4, int(limit) // max(1, len(domains)))
    for domain in domains:
        items = list(grouped[domain])
        rng.shuffle(items)
        sample.extend(items[:per_domain])
    rng.shuffle(sample)
    sample = sample[: int(limit)]

    qwen_base = _resolve_local_qwen_base_model()
    accepted: List[OmniRow] = []
    accepted_count = 0
    repair_count = 0
    discarded = 0

    with tempfile.TemporaryDirectory(prefix="omni_v3_qwen_") as tmpdir:
        tmp_root = Path(tmpdir)
        adapter_dir = _extract_qwen_adapter_dir(adapter_zip, tmp_root)
        engine = LocalQwenGenerator(base_model=qwen_base, adapter_dir=adapter_dir, device="cpu")
        try:
            for index, row in enumerate(sample, start=1):
                candidate = _normalize_response(engine.generate(row.prompt, max_new_tokens=64))
                if not candidate:
                    discarded += 1
                    if index % 8 == 0 or index == len(sample):
                        print(json.dumps({"event": "qwen_distill_progress", "completed": index, "total": len(sample), "accepted": accepted_count, "repair": repair_count, "discarded": discarded}, ensure_ascii=True), flush=True)
                    continue
                metrics = _score_candidate(row.response_text, candidate)
                if metrics["composite"] >= 0.18:
                    accepted.append(
                        OmniRow(
                            prompt=row.prompt,
                            intent=row.intent,
                            response_text=candidate,
                            domain=row.domain,
                            image_path=row.image_path,
                            vision_label=row.vision_label,
                            source="qwen_distill",
                        )
                    )
                    accepted_count += 1
                elif metrics["composite"] >= 0.05:
                    accepted.append(
                        OmniRow(
                            prompt=_normalize(
                                f"Improve and verify this draft answer.\nRequest: {row.prompt}\nDraft: {candidate}",
                                260,
                            ),
                            intent=row.intent,
                            response_text=row.response_text,
                            domain=row.domain,
                            image_path=row.image_path,
                            vision_label=row.vision_label,
                            source="qwen_repair",
                        )
                    )
                    repair_count += 1
                else:
                    discarded += 1
                if index % 8 == 0 or index == len(sample):
                    print(json.dumps({"event": "qwen_distill_progress", "completed": index, "total": len(sample), "accepted": accepted_count, "repair": repair_count, "discarded": discarded}, ensure_ascii=True), flush=True)
        finally:
            del engine
            gc.collect()

    summary = {
        "requested": int(limit),
        "sampled": len(sample),
        "accepted_teacher_rows": accepted_count,
        "accepted_repair_rows": repair_count,
        "discarded": discarded,
        "adapter_zip": str(adapter_zip),
        "base_model": str(qwen_base),
    }
    return accepted, summary


def build_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    adapter_zip: Path,
    seed: int,
    qwen_distill_limit: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    full_rows, source_counts = _build_large_rows(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    qwen_rows, qwen_summary = _distill_qwen_rows(full_rows, adapter_zip=adapter_zip, seed=seed + 211, limit=qwen_distill_limit)
    full_rows.extend(qwen_rows)
    source_counts["qwen_distill_total"] = len(qwen_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    stage2_rows = list(full_rows)
    summary = {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "qwen_distill": qwen_summary,
    }
    return stage1_rows, stage2_rows, summary


def _load_base_state_from_zip(model: OmniCollectiveNetV3, base_zip: Path) -> Dict[str, object]:
    if not base_zip.exists():
        return {"loaded": False, "reason": f"Missing base zip: {base_zip}"}
    with tempfile.TemporaryDirectory(prefix="omni_v3_base_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(tmp_path)
        weight_candidates = sorted(tmp_path.glob("**/*.pth"))
        if not weight_candidates:
            return {"loaded": False, "reason": f"No .pth weights found inside {base_zip.name}"}
        weights_path = weight_candidates[0]
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
        }
        info = model.load_state_dict(compatible, strict=False)
        return {
            "loaded": True,
            "source_weights": weights_path.name,
            "matched_keys": len(compatible),
            "missing_keys": list(info.missing_keys),
            "unexpected_keys": list(info.unexpected_keys),
        }


def _train_stage(
    *,
    model: OmniCollectiveNetV3,
    train_rows: Sequence[OmniRow],
    val_rows: Sequence[OmniRow],
    vocab: Dict[str, int],
    response_bank: Sequence[str],
    image_size: int,
    max_len: int,
    max_words: int,
    word_buckets: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    response_to_index = {text: idx for idx, text in enumerate(response_bank)}
    train_ds = OmniDatasetV2(
        train_rows,
        vocab=vocab,
        max_len=max_len,
        image_size=image_size,
        max_words=max_words,
        word_buckets=word_buckets,
        response_to_index=response_to_index,
    )
    val_ds = OmniDatasetV2(
        val_rows,
        vocab=vocab,
        max_len=max_len,
        image_size=image_size,
        max_words=max_words,
        word_buckets=word_buckets,
        response_to_index=response_to_index,
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.025)
    intent_loss_fn = nn.CrossEntropyLoss()
    response_loss_fn = nn.CrossEntropyLoss()
    vision_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    domain_loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_score = -1.0
    history: List[Dict[str, float]] = []
    print(
        json.dumps(
            {
                "event": "stage_start",
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                batch["token_ids"].to(device),
                batch["image_tensor"].to(device),
                batch["has_image"].to(device),
                batch["word_ids"].to(device),
                batch["prompt_features"].to(device),
            )
            loss = (
                0.55 * intent_loss_fn(outputs["intent"], batch["intent"].to(device))
                + 1.00 * response_loss_fn(outputs["response"], batch["response"].to(device))
                + 0.72 * domain_loss_fn(outputs["domain"], batch["domain"].to(device))
            )
            if bool(batch["vision"].ge(0).any()):
                loss = loss + 0.90 * vision_loss_fn(outputs["vision"], batch["vision"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.15)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["intent"].size(0))
            total_items += int(batch["intent"].size(0))
        val_metrics = evaluate(model, val_loader, device)
        score = _weighted_score(val_metrics)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(total_loss / max(total_items, 1)),
                "val_intent_accuracy": val_metrics["intent_accuracy"],
                "val_response_accuracy": val_metrics["response_accuracy"],
                "val_vision_accuracy": val_metrics["vision_accuracy"],
                "val_domain_accuracy": val_metrics["domain_accuracy"],
                "score": score,
            }
        )
        print(json.dumps({"event": "epoch_end", **history[-1]}, ensure_ascii=True), flush=True)
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model state captured during training stage.")
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_score": best_score,
        "val_metrics": evaluate(model, val_loader, device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }


def train_model(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    base_zip: Path,
    adapter_zip: Path,
    images_dir: Path,
    image_size: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    seed: int,
    qwen_distill_limit: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    _stage1_rows, full_rows, dataset_summary = build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        adapter_zip=adapter_zip,
        seed=seed,
        qwen_distill_limit=qwen_distill_limit,
    )
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)

    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    train_stage2 = list(train_rows)
    val_stage = list(val_rows)

    vocab = build_char_vocab([row.prompt for row in train_stage2], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})
    print(
        json.dumps(
            {
                "event": "label_space",
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "vocab_size": len(vocab),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    max_len = 256
    max_words = 56
    word_buckets = 8192
    device = torch.device("cpu")
    model = OmniCollectiveNetV3(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
    ).to(device)

    warm_start = _load_base_state_from_zip(model, base_zip)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)

    stage1 = _train_stage(
        model=model,
        train_rows=train_stage1,
        val_rows=val_stage,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=batch_size,
        learning_rate=stage1_lr,
        epochs=stage1_epochs,
        seed=seed + 101,
        device=device,
    )
    stage2 = _train_stage(
        model=model,
        train_rows=train_stage2,
        val_rows=val_stage,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=max(16, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        device=device,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v3_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v3_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v3_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v3_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v3_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 3,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 72,
        "text_hidden": 128,
        "image_channels": 32,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 64,
        "deep_text_channels": 144,
        "deep_image_channels": 56,
        "fusion_hidden": 448,
        "memory_slots": 8,
        "depth_blocks": 4,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    engine = OmniCollectiveEngineV3(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = [
        "Debug this Python stack trace and explain the root cause.",
        "Describe the key motion shown across these video frames.",
        "Make a photorealistic image prompt for a storm-battered lighthouse.",
        "Explain the phrase break the ice in simple language.",
    ]
    sample_outputs = [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "sample_outputs": sample_outputs,
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
        "dataset_summary": dataset_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an expanded omni_collective v3 frontier model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--qwen_adapter_zip", default=str(DEFAULT_QWEN_ADAPTER_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=112)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=0.00085)
    parser.add_argument("--stage2_lr", type=float, default=0.00045)
    parser.add_argument("--seed", type=int, default=93)
    parser.add_argument("--qwen_distill_limit", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        base_zip=Path(args.base_zip).resolve(),
        adapter_zip=Path(args.qwen_adapter_zip).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
        qwen_distill_limit=int(args.qwen_distill_limit),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
