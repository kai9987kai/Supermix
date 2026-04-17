from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

try:
    from build_science_image_test_dataset import DRAWERS
    from image_recognition_model import SCIENCE_IMAGE_CLASSES, ScienceImageRecognitionNet, load_image_tensor
except ImportError:  # pragma: no cover
    from .build_science_image_test_dataset import DRAWERS
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES, ScienceImageRecognitionNet, load_image_tensor


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")

CLASS_KEY_MAP = {
    "cell_diagram": "cell",
    "magnet_poles": "magnet",
    "simple_circuit": "circuit",
    "water_cycle_icon": "water_cycle",
    "moon_phases_strip": "moon_phases",
    "plant_photosynthesis": "photosynthesis",
    "solar_system_sketch": "solar_system",
    "dna_ladder": "dna",
    "thermometer_icon": "temperature",
    "states_of_matter_particles": "states_of_matter",
    "prism_dispersion": "light_dispersion",
}


@dataclass
class ImageSample:
    image_path: Path
    class_key: str
    seed: int


class AugmentedScienceImageDataset(Dataset):
    def __init__(self, rows: Sequence[ImageSample], image_size: int, augment: bool) -> None:
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.class_to_index = {name: idx for idx, name in enumerate(SCIENCE_IMAGE_CLASSES)}

    def __len__(self) -> int:
        return len(self.rows)

    def _augment_image(self, image: Image.Image, seed: int) -> Image.Image:
        rng = random.Random(int(seed))
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        rotated = image.rotate(rng.uniform(-16.0, 16.0), resample=Image.Resampling.BILINEAR, fillcolor=(255, 255, 255))
        scale = rng.uniform(0.82, 1.14)
        new_w = max(48, min(int(rotated.width * scale), 192))
        new_h = max(48, min(int(rotated.height * scale), 192))
        resized = rotated.resize((new_w, new_h), Image.Resampling.BILINEAR)
        offset_x = rng.randint(-10, 10) + (canvas.width - new_w) // 2
        offset_y = rng.randint(-10, 10) + (canvas.height - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        if rng.random() < 0.55:
            canvas = ImageEnhance.Brightness(canvas).enhance(rng.uniform(0.86, 1.16))
        if rng.random() < 0.55:
            canvas = ImageEnhance.Contrast(canvas).enhance(rng.uniform(0.82, 1.22))
        if rng.random() < 0.30:
            canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.8)))
        if rng.random() < 0.30:
            noise = torch.randn(3, canvas.height, canvas.width) * rng.uniform(0.01, 0.035)
            base = load_image_tensor_from_image(canvas)
            mixed = (base + noise).clamp(0.0, 1.0)
            canvas = tensor_to_pil(mixed)
        return canvas

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(row.image_path) as image:
            image = image.convert("RGB")
            if self.augment:
                image = self._augment_image(image, row.seed)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            tensor = load_image_tensor_from_image(image)
        target = torch.tensor(self.class_to_index[row.class_key], dtype=torch.long)
        return tensor, target


def load_image_tensor_from_image(image: Image.Image) -> torch.Tensor:
    buffer = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
    return buffer.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    data = tensor.detach().clamp(0.0, 1.0).permute(1, 2, 0).mul(255.0).round().byte().cpu().numpy()
    return Image.fromarray(data, mode="RGB")


def ensure_base_images(images_dir: Path) -> Dict[str, Dict[str, str]]:
    images_dir.mkdir(parents=True, exist_ok=True)
    class_info: Dict[str, Dict[str, str]] = {}
    for raw_name, drawer in DRAWERS:
        meta = drawer(images_dir / f"{raw_name}.png")
        class_key = CLASS_KEY_MAP[raw_name]
        class_info[class_key] = {
            "caption": str(meta.get("caption") or "").strip(),
            "tags": str(meta.get("tags") or "").strip(),
            "concept": str(meta.get("concept") or class_key).strip(),
            "source_image": str((images_dir / f"{raw_name}.png").resolve()),
        }
    return class_info


def build_rows(images_dir: Path, *, samples_per_class: int, seed: int) -> tuple[List[ImageSample], List[ImageSample], Dict[str, Dict[str, str]]]:
    class_info = ensure_base_images(images_dir)
    rng = random.Random(int(seed))
    train_rows: List[ImageSample] = []
    val_rows: List[ImageSample] = []
    for raw_name, _drawer in DRAWERS:
        class_key = CLASS_KEY_MAP[raw_name]
        base_image = images_dir / f"{raw_name}.png"
        for idx in range(int(samples_per_class)):
            sample = ImageSample(
                image_path=base_image,
                class_key=class_key,
                seed=rng.randint(0, 10_000_000),
            )
            if idx < max(8, int(samples_per_class * 0.16)):
                val_rows.append(sample)
            else:
                train_rows.append(sample)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows, class_info


def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    total = 0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=1)
            total += int(yb.numel())
            correct += int((pred == yb).sum().item())
    return float(correct / max(total, 1))


def train_model(
    *,
    output_dir: Path,
    models_dir: Path,
    images_dir: Path,
    image_size: int,
    samples_per_class: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    train_rows, val_rows, class_info = build_rows(images_dir, samples_per_class=samples_per_class, seed=seed)
    train_ds = AugmentedScienceImageDataset(train_rows, image_size=image_size, augment=True)
    val_ds = AugmentedScienceImageDataset(val_rows, image_size=image_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScienceImageRecognitionNet(num_classes=len(SCIENCE_IMAGE_CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.02)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(2, int(epochs)))

    best_state = None
    best_val = -math.inf
    history: List[Dict[str, float]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(yb.size(0))
            total_examples += int(yb.size(0))
        scheduler.step()
        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(total_loss / max(total_examples, 1)),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not capture a model state.")
    model.load_state_dict(best_state)
    train_acc = accuracy(model, train_loader, device)
    val_acc = accuracy(model, val_loader, device)

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_science_image_recognition_micro_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "science_image_recognition_micro_v1.pth"
    meta_path = artifact_dir / "science_image_recognition_micro_v1_meta.json"
    summary_path = artifact_dir / "science_image_recognition_micro_v1_summary.json"
    zip_path = output_dir / f"supermix_science_image_recognition_micro_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    meta = {
        "classes": list(SCIENCE_IMAGE_CLASSES),
        "class_info": class_info,
        "image_size": int(image_size),
        "base_channels": 24,
        "hidden_dim": 96,
        "parameter_count": parameter_count,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "samples_per_class": int(samples_per_class),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
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
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "parameter_count": parameter_count,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local science image recognition model and package it for Supermix Studio.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--samples_per_class", type=int, default=140)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=0.0012)
    parser.add_argument("--seed", type=int, default=27)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        samples_per_class=int(args.samples_per_class),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
