from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from build_science_image_test_dataset import DRAWERS, _rows_for_image
    from image_recognition_model import SCIENCE_IMAGE_CLASSES, load_image_tensor
    from multimodel_catalog import discover_model_records
    from omni_collective_model import INTENT_TO_INDEX, OMNI_INTENTS, OmniCollectiveNet, build_char_vocab, encode_text
    from train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
    from train_math_equation_model import build_rows as build_math_rows
except ImportError:  # pragma: no cover
    from .build_science_image_test_dataset import DRAWERS, _rows_for_image
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES, load_image_tensor
    from .multimodel_catalog import discover_model_records
    from .omni_collective_model import INTENT_TO_INDEX, OMNI_INTENTS, OmniCollectiveNet, build_char_vocab, encode_text
    from .train_image_recognition_model import CLASS_KEY_MAP, ensure_base_images
    from .train_math_equation_model import build_rows as build_math_rows


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


@dataclass
class OmniRow:
    prompt: str
    intent: str
    response_text: str
    image_path: str = ""
    vision_label: Optional[str] = None


class OmniDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[OmniRow],
        *,
        vocab: Dict[str, int],
        max_len: int,
        image_size: int,
        response_to_index: Dict[str, int],
    ) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.image_size = int(image_size)
        self.response_to_index = response_to_index
        self.vision_to_index = {name: idx for idx, name in enumerate(SCIENCE_IMAGE_CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.rows[index]
        token_ids = torch.tensor(encode_text(row.prompt, self.vocab, self.max_len), dtype=torch.long)
        if row.image_path:
            image_tensor = load_image_tensor(row.image_path, self.image_size)
            has_image = torch.tensor(1.0, dtype=torch.float32)
        else:
            image_tensor = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            has_image = torch.tensor(0.0, dtype=torch.float32)
        response_index = self.response_to_index.get(row.response_text, -1)
        vision_index = self.vision_to_index.get(str(row.vision_label or ""), -1)
        return {
            "token_ids": token_ids,
            "image_tensor": image_tensor,
            "has_image": has_image,
            "intent": torch.tensor(INTENT_TO_INDEX[row.intent], dtype=torch.long),
            "response": torch.tensor(response_index, dtype=torch.long),
            "vision": torch.tensor(vision_index, dtype=torch.long),
        }


def _curated_general_rows() -> List[OmniRow]:
    return [
        OmniRow(
            prompt="Debug this Python stack trace and give the most likely root cause.",
            intent="coding",
            response_text="Start with the first failing frame, identify the bad input or missing dependency, then confirm the root cause before patching.",
        ),
        OmniRow(
            prompt="Compare two ways to solve this problem and tell me the tradeoffs.",
            intent="comparison",
            response_text="Compare the options on complexity, reliability, cost, and reversibility, then choose the one that preserves the safest rollback path.",
        ),
        OmniRow(
            prompt="Give me a compact step-by-step plan to finish this task.",
            intent="planning",
            response_text="Break the task into the next concrete steps, verify after each step, and keep the work in a state you can test quickly.",
        ),
        OmniRow(
            prompt="Rewrite this draft so it sounds more direct and professional.",
            intent="creative",
            response_text="Tighten the wording, remove filler, keep the tone direct, and preserve the core meaning.",
        ),
        OmniRow(
            prompt="I need the latest version information from the web.",
            intent="current_info",
            response_text="Use web search for current facts, then answer with the verified result instead of relying on stale local memory.",
        ),
        OmniRow(
            prompt="Open Command Prompt for me.",
            intent="command",
            response_text="When you explicitly ask, I can trigger the runtime to open Command Prompt without running extra shell commands inside it.",
        ),
        OmniRow(
            prompt="Write a short image prompt for a storm-battered lighthouse at night.",
            intent="image_prompt",
            response_text="Cinematic storm-battered lighthouse at night, crashing waves, cold moonlight, wet stone, dramatic clouds, high contrast, atmospheric detail.",
        ),
        OmniRow(
            prompt="Help me choose the best local model for a coding task.",
            intent="model_selection",
            response_text="For stronger coding or reasoning prompts, prefer the highest benchmarked text model unless you need a faster lightweight answer.",
        ),
        OmniRow(
            prompt="Summarize the problem directly.",
            intent="general",
            response_text="State the core issue first, then give the shortest useful next step.",
        ),
    ]


def _distill_model_rows(models_dir: Path) -> List[OmniRow]:
    rows: List[OmniRow] = []
    for record in discover_model_records(models_dir=models_dir):
        cap_text = ", ".join(record.capabilities) or "chat"
        summary = record.note or record.benchmark_hint or f"{record.label} is part of the {record.family} model family."
        response = f"{record.label} works best for {summary} Capabilities: {cap_text}."
        rows.extend(
            [
                OmniRow(prompt=f"What is {record.label} best for?", intent="model_selection", response_text=response),
                OmniRow(prompt=f"When should I use {record.label}?", intent="model_selection", response_text=response),
                OmniRow(prompt=f"Quick strengths of {record.label}.", intent="model_selection", response_text=response),
            ]
        )
    return rows


def _science_rows(images_dir: Path) -> List[OmniRow]:
    class_info = ensure_base_images(images_dir)
    rows: List[OmniRow] = []
    rng = random.Random(27)
    for raw_name, drawer in DRAWERS:
        image_path = images_dir / f"{raw_name}.png"
        if not image_path.exists():
            drawer(image_path)
        meta = class_info[CLASS_KEY_MAP[raw_name]]
        for item in _rows_for_image(str(image_path), meta, rng):
            rows.append(
                OmniRow(
                    prompt=str(item.get("user") or "").strip(),
                    intent="vision",
                    response_text=str(item.get("assistant") or "").strip(),
                    image_path=str(image_path),
                    vision_label=CLASS_KEY_MAP[raw_name],
                )
            )
    return rows


def _math_rows(limit: int) -> List[OmniRow]:
    prompts = build_math_rows(samples_per_intent=max(24, int(limit)), seed=31)
    rows: List[OmniRow] = []
    generic = "This is a symbolic math prompt. Use the solver path to compute the exact answer."
    for prompt, _label in prompts[: int(limit) * 8]:
        rows.append(OmniRow(prompt=prompt, intent="math", response_text=generic))
    return rows


def build_training_rows(models_dir: Path, images_dir: Path) -> List[OmniRow]:
    rows = []
    rows.extend(_curated_general_rows())
    rows.extend(_distill_model_rows(models_dir))
    rows.extend(_science_rows(images_dir))
    rows.extend(_math_rows(limit=40))
    return rows


def split_rows(rows: Sequence[OmniRow], seed: int) -> tuple[List[OmniRow], List[OmniRow]]:
    cooked = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(cooked)
    pivot = int(len(cooked) * 0.88)
    return cooked[:pivot], cooked[pivot:]


def evaluate(model: OmniCollectiveNet, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total = 0
    intent_correct = 0
    response_correct = 0
    response_total = 0
    vision_correct = 0
    vision_total = 0
    with torch.inference_mode():
        for batch in loader:
            outputs = model(
                batch["token_ids"].to(device),
                batch["image_tensor"].to(device),
                batch["has_image"].to(device),
            )
            intent_pred = outputs["intent"].argmax(dim=1).cpu()
            response_pred = outputs["response"].argmax(dim=1).cpu()
            vision_pred = outputs["vision"].argmax(dim=1).cpu()
            intent = batch["intent"]
            response = batch["response"]
            vision = batch["vision"]
            total += int(intent.numel())
            intent_correct += int((intent_pred == intent).sum().item())
            response_mask = response.ge(0)
            if bool(response_mask.any()):
                response_correct += int((response_pred[response_mask] == response[response_mask]).sum().item())
                response_total += int(response_mask.sum().item())
            vision_mask = vision.ge(0)
            if bool(vision_mask.any()):
                vision_correct += int((vision_pred[vision_mask] == vision[vision_mask]).sum().item())
                vision_total += int(vision_mask.sum().item())
    return {
        "intent_accuracy": float(intent_correct / max(total, 1)),
        "response_accuracy": float(response_correct / max(response_total, 1)),
        "vision_accuracy": float(vision_correct / max(vision_total, 1)),
    }


def train_model(
    *,
    output_dir: Path,
    models_dir: Path,
    images_dir: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    rows = build_training_rows(models_dir=models_dir, images_dir=images_dir)
    train_rows, val_rows = split_rows(rows, seed=seed)
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in rows if row.response_text})
    response_to_index = {text: idx for idx, text in enumerate(response_bank)}
    max_len = 220

    train_ds = OmniDataset(train_rows, vocab=vocab, max_len=max_len, image_size=image_size, response_to_index=response_to_index)
    val_ds = OmniDataset(val_rows, vocab=vocab, max_len=max_len, image_size=image_size, response_to_index=response_to_index)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OmniCollectiveNet(
        vocab_size=len(vocab),
        num_intents=len(OMNI_INTENTS),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.02)
    intent_loss_fn = nn.CrossEntropyLoss()
    response_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    vision_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    best_state = None
    best_score = -1.0
    history: List[Dict[str, float]] = []
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
            )
            loss_terms = [0.7 * intent_loss_fn(outputs["intent"], batch["intent"].to(device))]
            if bool(batch["response"].ge(0).any()):
                loss_terms.append(1.0 * response_loss_fn(outputs["response"], batch["response"].to(device)))
            if bool(batch["vision"].ge(0).any()):
                loss_terms.append(0.9 * vision_loss_fn(outputs["vision"], batch["vision"].to(device)))
            loss = loss_terms[0]
            for term in loss_terms[1:]:
                loss = loss + term
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["intent"].size(0))
            total_items += int(batch["intent"].size(0))
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        score = val_metrics["intent_accuracy"] + val_metrics["response_accuracy"] + val_metrics["vision_accuracy"]
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(total_loss / max(total_items, 1)),
                "train_intent_accuracy": train_metrics["intent_accuracy"],
                "train_response_accuracy": train_metrics["response_accuracy"],
                "train_vision_accuracy": train_metrics["vision_accuracy"],
                "val_intent_accuracy": val_metrics["intent_accuracy"],
                "val_response_accuracy": val_metrics["response_accuracy"],
                "val_vision_accuracy": val_metrics["vision_accuracy"],
            }
        )
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not capture a model state.")
    model.load_state_dict(best_state)
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v1.pth"
    meta_path = artifact_dir / "omni_collective_v1_meta.json"
    summary_path = artifact_dir / "omni_collective_v1_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 56,
        "text_hidden": 96,
        "image_channels": 24,
        "fusion_hidden": 176,
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "catalog_models_distilled": [record.key for record in discover_model_records(models_dir=models_dir)],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    summary = {
        "artifact": zip_path.name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parameter_count": parameter_count,
        "history": history,
        "meta": meta,
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
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parameter_count": parameter_count,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fused local omnibus model for Supermix Studio.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=0.0011)
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
