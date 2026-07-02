from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from build_3d_video_test_dataset import (
        MODEL_BUILDERS,
        VIDEO_BUILDERS,
        _rows_for_model,
        _rows_for_video,
    )
    from build_science_image_test_dataset import DRAWERS, _rows_for_image
    from image_recognition_model import SCIENCE_IMAGE_CLASSES, load_image_tensor
    from multimodel_catalog import discover_model_records
    from omni_collective_model import (
        DOMAIN_TO_INDEX_V2,
        INTENT_TO_INDEX_V2,
        OMNI_DOMAIN_LABELS_V2,
        OMNI_INTENTS_V2,
        OmniCollectiveEngine,
        OmniCollectiveNetV2,
        build_char_vocab,
        encode_text,
        encode_word_hashes,
        prompt_feature_vector,
    )
    from train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
    from train_math_equation_model import build_rows as build_math_rows
except ImportError:  # pragma: no cover
    from .build_3d_video_test_dataset import (
        MODEL_BUILDERS,
        VIDEO_BUILDERS,
        _rows_for_model,
        _rows_for_video,
    )
    from .build_science_image_test_dataset import DRAWERS, _rows_for_image
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .multimodel_catalog import discover_model_records
    from .omni_collective_model import (
        DOMAIN_TO_INDEX_V2,
        INTENT_TO_INDEX_V2,
        OMNI_DOMAIN_LABELS_V2,
        OMNI_INTENTS_V2,
        OmniCollectiveEngine,
        OmniCollectiveNetV2,
        build_char_vocab,
        encode_text,
        encode_word_hashes,
        prompt_feature_vector,
    )
    from .train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
    from .train_math_equation_model import build_rows as build_math_rows


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")
DEFAULT_BASE_ZIP = DEFAULT_MODELS_DIR / "supermix_omni_collective_v1_20260327.zip"


@dataclass
class OmniRow:
    prompt: str
    intent: str
    response_text: str
    domain: str
    image_path: str = ""
    vision_label: Optional[str] = None
    source: str = ""


def _normalize(text: str, limit: int) -> str:
    cooked = " ".join(str(text or "").strip().split())
    return cooked[: int(limit)]


def _weighted_score(metrics: Dict[str, float]) -> float:
    return (
        0.20 * float(metrics.get("intent_accuracy", 0.0))
        + 0.45 * float(metrics.get("response_accuracy", 0.0))
        + 0.20 * float(metrics.get("domain_accuracy", 0.0))
        + 0.15 * float(metrics.get("vision_accuracy", 0.0))
    )


def _infer_intent(prompt: str, domain: str, image_path: str = "") -> str:
    lowered = str(prompt or "").lower()
    if image_path:
        return "vision"
    if domain == "coding":
        return "coding"
    if domain == "math":
        return "math"
    if domain == "language":
        return "language"
    if domain == "image_prompt":
        return "image_prompt"
    if domain == "spatial_3d":
        return "spatial_3d"
    if domain == "video":
        return "video"
    if domain == "model_selection":
        return "model_selection"
    if domain == "tool":
        if "command prompt" in lowered or "cmd" in lowered:
            return "command"
        return "current_info"
    if "plan" in lowered or "steps" in lowered or "roadmap" in lowered:
        return "planning"
    if "compare" in lowered or "tradeoff" in lowered:
        return "comparison"
    if "story" in lowered or "rewrite" in lowered or "creative" in lowered or "poem" in lowered:
        return "creative"
    if "latest" in lowered or "current" in lowered or "recent" in lowered or "news" in lowered:
        return "current_info"
    if domain == "knowledge":
        return "knowledge"
    return "general"


def _sample_jsonl(path: Path, *, limit: int, seed: int) -> List[dict]:
    rng = random.Random(int(seed))
    chosen: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except Exception:
                continue
            if len(chosen) < int(limit):
                chosen.append(row)
                continue
            swap_index = rng.randint(1, idx)
            if swap_index <= int(limit):
                chosen[swap_index - 1] = row
    return chosen


def _rows_from_jsonl(
    path: Path,
    *,
    limit: int,
    seed: int,
    domain: str,
    source_tag: str,
) -> List[OmniRow]:
    rows: List[OmniRow] = []
    for item in _sample_jsonl(path, limit=limit, seed=seed):
        prompt = _normalize(str(item.get("user") or item.get("prompt") or ""), 240)
        response = _normalize(str(item.get("assistant") or item.get("response") or ""), 380)
        if len(prompt) < 6 or len(response) < 8:
            continue
        rows.append(
            OmniRow(
                prompt=prompt,
                intent=_infer_intent(prompt, domain),
                response_text=response,
                domain=domain,
                source=source_tag,
            )
        )
    return rows


def _curated_rows() -> List[OmniRow]:
    return [
        OmniRow(
            prompt="Debug this Python stack trace and tell me the likeliest root cause.",
            intent="coding",
            response_text="Start with the first failing frame, isolate the bad input or missing dependency, and confirm the root cause before patching.",
            domain="coding",
            source="curated",
        ),
        OmniRow(
            prompt="Give me a compact step-by-step plan to finish this task safely.",
            intent="planning",
            response_text="Break the task into the next concrete steps, verify after each step, and keep the work in a state you can test quickly.",
            domain="planning",
            source="curated",
        ),
        OmniRow(
            prompt="Compare two possible solutions and tell me the tradeoffs.",
            intent="comparison",
            response_text="Compare the options on complexity, reliability, cost, and reversibility, then choose the one that preserves the safest rollback path.",
            domain="general",
            source="curated",
        ),
        OmniRow(
            prompt="I need the latest information about this topic.",
            intent="current_info",
            response_text="Use web search for current facts, then answer with the verified result instead of relying on stale local memory.",
            domain="tool",
            source="curated",
        ),
        OmniRow(
            prompt="Open Command Prompt for me.",
            intent="command",
            response_text="When you explicitly ask, I can trigger the runtime to open Command Prompt without running extra shell commands inside it.",
            domain="tool",
            source="curated",
        ),
        OmniRow(
            prompt="Explain this sentence in simpler language.",
            intent="language",
            response_text="Restate the meaning in shorter, clearer language while preserving the original point.",
            domain="language",
            source="curated",
        ),
        OmniRow(
            prompt="Make a photorealistic image prompt for a rainy Tokyo alley at night.",
            intent="image_prompt",
            response_text="Photorealistic rainy Tokyo alley at night, neon reflections, wet pavement, shallow depth of field, cinematic lighting, fine detail, realistic textures, moody atmosphere.",
            domain="image_prompt",
            source="curated",
        ),
        OmniRow(
            prompt="Help me choose the best local model for a coding task.",
            intent="model_selection",
            response_text="For stronger coding or reasoning prompts, prefer the highest benchmarked text model unless you need a faster lightweight answer.",
            domain="model_selection",
            source="curated",
        ),
    ]


def _model_selection_rows(models_dir: Path, allowed_model_keys: Optional[Sequence[str]] = None) -> List[OmniRow]:
    rows: List[OmniRow] = []
    allowed = {str(key) for key in allowed_model_keys} if allowed_model_keys else None
    for record in discover_model_records(models_dir=models_dir):
        if allowed is not None and record.key not in allowed:
            continue
        score = record.display_score
        score_text = f"{score:.4f}" if score is not None else "specialist"
        response = (
            f"{record.label} is best used for {record.benchmark_hint or record.note or record.family}. "
            f"Capabilities: {', '.join(record.capabilities)}. Score hint: {score_text}."
        )
        rows.extend(
            [
                OmniRow(
                    prompt=f"What is {record.label} best for?",
                    intent="model_selection",
                    response_text=response,
                    domain="model_selection",
                    source="catalog",
                ),
                OmniRow(
                    prompt=f"When should I use {record.label}?",
                    intent="model_selection",
                    response_text=response,
                    domain="model_selection",
                    source="catalog",
                ),
            ]
        )
    return rows


def _science_rows(images_dir: Path, *, repeats: int, seed: int) -> List[OmniRow]:
    class_info = ensure_base_images(images_dir)
    rng = random.Random(int(seed))
    rows: List[OmniRow] = []
    for raw_name, drawer in DRAWERS:
        image_path = images_dir / f"{raw_name}.png"
        if not image_path.exists():
            drawer(image_path)
        meta = class_info[CLASS_KEY_MAP[raw_name]]
        for _ in range(max(1, int(repeats))):
            for item in _rows_for_image(str(image_path), meta, rng):
                prompt = _normalize(str(item.get("user") or ""), 220)
                response = _normalize(str(item.get("assistant") or ""), 320)
                if not prompt or not response:
                    continue
                rows.append(
                    OmniRow(
                        prompt=prompt,
                        intent="vision",
                        response_text=response,
                        domain="vision",
                        image_path=str(image_path),
                        vision_label=CLASS_KEY_MAP[raw_name],
                        source="science_image",
                    )
                )
    return rows


def _three_d_and_video_rows(work_dir: Path, *, repeats: int, seed: int) -> List[OmniRow]:
    rng = random.Random(int(seed))
    models_dir = work_dir / "generated_3d_models"
    videos_dir = work_dir / "generated_videos"
    rows: List[OmniRow] = []
    for stem, builder in MODEL_BUILDERS:
        path = models_dir / f"{stem}.obj"
        meta = builder(path)
        for _ in range(max(1, int(repeats))):
            for item in _rows_for_model(str(path), meta, rng):
                prompt = _normalize(str(item.get("user") or ""), 220)
                response = _normalize(str(item.get("assistant") or ""), 280)
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
    for filename, builder in VIDEO_BUILDERS:
        path = videos_dir / filename
        meta = builder(path)
        for _ in range(max(1, int(repeats))):
            for item in _rows_for_video(str(path), meta, rng):
                prompt = _normalize(str(item.get("user") or ""), 220)
                response = _normalize(str(item.get("assistant") or ""), 280)
                if prompt and response:
                    rows.append(
                        OmniRow(
                            prompt=prompt,
                            intent="video",
                            response_text=response,
                            domain="video",
                            source="video",
                        )
                    )
    return rows


def _math_rows(limit: int) -> List[OmniRow]:
    rows: List[OmniRow] = []
    generic = "This is a symbolic math prompt. Use the solver path to compute the exact answer."
    for prompt, _label in build_math_rows(samples_per_intent=max(18, int(limit)), seed=73)[: int(limit) * 8]:
        rows.append(
            OmniRow(
                prompt=_normalize(prompt, 220),
                intent="math",
                response_text=generic,
                domain="math",
                source="math",
            )
        )
    return rows


def _image_prompt_rows(seed: int, limit: int) -> List[OmniRow]:
    rng = random.Random(int(seed))
    subjects = [
        "rainy Tokyo alley at night",
        "glass greenhouse in the Arctic",
        "retro robot in a repair shop",
        "cathedral library lit by sunrise",
        "underwater research station",
        "storm-battered lighthouse",
        "ancient observatory on a mountain ridge",
        "bioluminescent forest path",
        "spaceship cockpit over Mars",
        "vintage camera on a wooden desk",
    ]
    styles = ["photorealistic", "cinematic", "editorial", "documentary photo", "high-detail studio photo"]
    moods = ["moody atmosphere", "soft ambient light", "dramatic contrast", "foggy depth", "warm sunrise glow"]
    lenses = ["35mm lens", "50mm lens", "wide shot", "close-up framing", "shallow depth of field"]
    rows: List[OmniRow] = []
    for _ in range(int(limit)):
        subject = rng.choice(subjects)
        style = rng.choice(styles)
        mood = rng.choice(moods)
        lens = rng.choice(lenses)
        prompt = f"Create a strong photo prompt for {subject}."
        response = f"{style} {subject}, {lens}, {mood}, realistic textures, natural color grading, high detail, polished composition."
        rows.append(
            OmniRow(
                prompt=prompt,
                intent="image_prompt",
                response_text=response,
                domain="image_prompt",
                source="image_prompt",
            )
        )
    return rows


def _language_rows() -> List[OmniRow]:
    return [
        OmniRow(
            prompt="Explain the phrase 'break the ice' in simple language.",
            intent="language",
            response_text="It means making people feel more comfortable and starting a conversation in an easier way.",
            domain="language",
            source="language",
        ),
        OmniRow(
            prompt="Translate a short greeting into Spanish: Good morning, how are you?",
            intent="language",
            response_text="Buenos dias, como estas?",
            domain="language",
            source="language",
        ),
        OmniRow(
            prompt="Explain what a metaphor is for a beginner.",
            intent="language",
            response_text="A metaphor describes one thing as if it were another thing to make an idea clearer or more vivid.",
            domain="language",
            source="language",
        ),
        OmniRow(
            prompt="Rewrite this message to sound more direct and professional.",
            intent="language",
            response_text="Tighten the wording, remove filler, and keep the meaning intact while sounding clear and professional.",
            domain="language",
            source="language",
        ),
    ]


def build_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    seed: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    datasets_dir = repo_root / "datasets"
    full_rows: List[OmniRow] = []
    full_rows.extend(_curated_rows())
    full_rows.extend(_model_selection_rows(models_dir))
    full_rows.extend(_language_rows())
    full_rows.extend(_image_prompt_rows(seed=seed + 11, limit=180))
    full_rows.extend(_math_rows(limit=36))

    dataset_specs = [
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 280, "coding", "coding"),
        ("conversation_data.world_events_2026_02_19.jsonl", 80, "knowledge", "world_events"),
        ("conversation_data.hybrid_v6_live_knowledge.jsonl", 360, "general", "hybrid"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 260, "language", "books"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 420, "creative", "creative"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 620, "knowledge", "reasoning"),
        ("conversation_data.supermix_plus_v27_500k.jsonl", 900, "knowledge", "supermix_plus"),
        ("conversation_data.quality_anchor_v2.jsonl", 80, "general", "quality_anchor"),
    ]
    source_counts: Dict[str, int] = defaultdict(int)
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if path.exists():
            sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(full_rows), domain=domain, source_tag=source_tag)
            full_rows.extend(sampled)
            source_counts[source_tag] += len(sampled)

    science_rows = _science_rows(images_dir, repeats=3, seed=seed + 101)
    three_d_rows = _three_d_and_video_rows(repo_root / "output" / "omni_v2_generated", repeats=6, seed=seed + 131)
    full_rows.extend(science_rows)
    full_rows.extend(three_d_rows)
    source_counts["science_image"] += len(science_rows)
    source_counts["3d_video"] += len(three_d_rows)

    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    stage2_rows = list(full_rows)

    summary = {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(stage2_rows),
        "source_counts": dict(sorted(source_counts.items())),
    }
    return stage1_rows, stage2_rows, summary


def split_rows(rows: Sequence[OmniRow], seed: int) -> Tuple[List[OmniRow], List[OmniRow]]:
    grouped: Dict[str, List[OmniRow]] = defaultdict(list)
    for row in rows:
        grouped[row.domain].append(row)
    rng = random.Random(int(seed))
    train_rows: List[OmniRow] = []
    val_rows: List[OmniRow] = []
    for _key, items in grouped.items():
        cooked = list(items)
        rng.shuffle(cooked)
        pivot = max(1, int(len(cooked) * 0.88))
        train_rows.extend(cooked[:pivot])
        val_rows.extend(cooked[pivot:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


class OmniDatasetV2(Dataset):
    def __init__(
        self,
        rows: Sequence[OmniRow],
        *,
        vocab: Dict[str, int],
        max_len: int,
        image_size: int,
        max_words: int,
        word_buckets: int,
        response_to_index: Dict[str, int],
    ) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.image_size = int(image_size)
        self.max_words = int(max_words)
        self.word_buckets = int(word_buckets)
        self.response_to_index = response_to_index
        self.vision_to_index = {name: idx for idx, name in enumerate(SCIENCE_IMAGE_CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        token_ids = torch.tensor(encode_text(row.prompt, self.vocab, self.max_len), dtype=torch.long)
        word_ids = torch.tensor(
            encode_word_hashes(row.prompt, buckets=self.word_buckets, max_words=self.max_words),
            dtype=torch.long,
        )
        prompt_features = torch.tensor(prompt_feature_vector(row.prompt), dtype=torch.float32)
        if row.image_path:
            image_tensor = load_image_tensor(row.image_path, self.image_size)
            has_image = torch.tensor(1.0, dtype=torch.float32)
        else:
            image_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            has_image = torch.tensor(0.0, dtype=torch.float32)
        response_index = self.response_to_index[row.response_text]
        vision_index = self.vision_to_index.get(str(row.vision_label or ""), -1)
        return {
            "token_ids": token_ids,
            "word_ids": word_ids,
            "prompt_features": prompt_features,
            "image_tensor": image_tensor,
            "has_image": has_image,
            "intent": torch.tensor(INTENT_TO_INDEX_V2[row.intent], dtype=torch.long),
            "response": torch.tensor(response_index, dtype=torch.long),
            "vision": torch.tensor(vision_index, dtype=torch.long),
            "domain": torch.tensor(DOMAIN_TO_INDEX_V2[row.domain], dtype=torch.long),
        }


def evaluate(model: OmniCollectiveNetV2, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    intent_correct = 0
    response_correct = 0
    response_total = 0
    vision_correct = 0
    vision_total = 0
    domain_correct = 0
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch["token_ids"].to(device),
                batch["image_tensor"].to(device),
                batch["has_image"].to(device),
                batch["word_ids"].to(device),
                batch["prompt_features"].to(device),
            )
            intent_pred = outputs["intent"].argmax(dim=1).cpu()
            response_pred = outputs["response"].argmax(dim=1).cpu()
            vision_pred = outputs["vision"].argmax(dim=1).cpu()
            domain_pred = outputs["domain"].argmax(dim=1).cpu()
            intent = batch["intent"]
            response = batch["response"]
            vision = batch["vision"]
            domain = batch["domain"]
            total += int(intent.numel())
            intent_correct += int((intent_pred == intent).sum().item())
            response_correct += int((response_pred == response).sum().item())
            response_total += int(response.numel())
            vision_mask = vision.ge(0)
            if bool(vision_mask.any()):
                vision_correct += int((vision_pred[vision_mask] == vision[vision_mask]).sum().item())
                vision_total += int(vision_mask.sum().item())
            domain_correct += int((domain_pred == domain).sum().item())
    return {
        "intent_accuracy": float(intent_correct / max(total, 1)),
        "response_accuracy": float(response_correct / max(response_total, 1)),
        "vision_accuracy": float(vision_correct / max(vision_total, 1)),
        "domain_accuracy": float(domain_correct / max(total, 1)),
    }


def _load_base_state(model: OmniCollectiveNetV2, base_zip: Path) -> Dict[str, object]:
    if not base_zip.exists():
        return {"loaded": False, "reason": f"Missing base zip: {base_zip}"}
    with tempfile.TemporaryDirectory(prefix="omni_v2_base_") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(base_zip) as archive:
            archive.extractall(tmp_path)
        weights_path = tmp_path / "omni_collective_v1.pth"
        if not weights_path.exists():
            return {"loaded": False, "reason": "Base zip did not contain omni_collective_v1.pth"}
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
            "matched_keys": len(compatible),
            "missing_keys": list(info.missing_keys),
            "unexpected_keys": list(info.unexpected_keys),
        }


def _train_stage(
    *,
    model: OmniCollectiveNetV2,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.02)
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
                + 0.70 * domain_loss_fn(outputs["domain"], batch["domain"].to(device))
            )
            if bool(batch["vision"].ge(0).any()):
                loss = loss + 0.95 * vision_loss_fn(outputs["vision"], batch["vision"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.2)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["intent"].size(0))
            total_items += int(batch["intent"].size(0))
        val_metrics = evaluate(model, val_loader, device)
        score = _weighted_score(val_metrics)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(total_loss / max(total_items, 1)),
                "train_intent_accuracy": -1.0,
                "train_response_accuracy": -1.0,
                "train_vision_accuracy": -1.0,
                "train_domain_accuracy": -1.0,
                "val_intent_accuracy": val_metrics["intent_accuracy"],
                "val_response_accuracy": val_metrics["response_accuracy"],
                "val_vision_accuracy": val_metrics["vision_accuracy"],
                "val_domain_accuracy": val_metrics["domain_accuracy"],
                "score": score,
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch_end",
                    "epoch": epoch,
                    "train_loss": history[-1]["train_loss"],
                    "val_intent_accuracy": history[-1]["val_intent_accuracy"],
                    "val_response_accuracy": history[-1]["val_response_accuracy"],
                    "val_vision_accuracy": history[-1]["val_vision_accuracy"],
                    "val_domain_accuracy": history[-1]["val_domain_accuracy"],
                    "score": score,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No model state captured during training stage.")
    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_score": best_score,
        "train_metrics": {},
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
    images_dir: Path,
    image_size: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    _stage1_rows, full_rows, dataset_summary = build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
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
    max_len = 240
    max_words = 48
    word_buckets = 4096

    device = torch.device("cpu")
    model = OmniCollectiveNetV2(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
    ).to(device)
    warm_start = _load_base_state(model, base_zip)
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
    artifact_dir = output_dir / f"supermix_omni_collective_v2_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v2_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v2_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v2_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v2_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 2,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 56,
        "text_hidden": 96,
        "image_channels": 24,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 48,
        "deep_text_channels": 96,
        "deep_image_channels": 40,
        "fusion_hidden": 320,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    engine = OmniCollectiveEngine(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = [
        "Debug this Python stack trace and explain the root cause.",
        "Make a photorealistic image prompt for a storm-battered lighthouse.",
        "Explain the phrase break the ice in simple language.",
        "Describe this 3D model in simple terms.",
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
    parser = argparse.ArgumentParser(description="Train a larger omni_collective continuation model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--stage1_epochs", type=int, default=2)
    parser.add_argument("--stage2_epochs", type=int, default=2)
    parser.add_argument("--stage1_lr", type=float, default=0.0010)
    parser.add_argument("--stage2_lr", type=float, default=0.00055)
    parser.add_argument("--seed", type=int, default=77)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        base_zip=Path(args.base_zip).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
