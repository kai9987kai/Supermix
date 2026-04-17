#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F 
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chat_pipeline import text_to_model_input
from model_frontier_native_image_v36 import ChampionNetFrontierCollectiveNativeImage
from model_native_image_lite_v37 import ChampionNetUltraExpertNativeImageLite, save_prompt_image
from run import safe_load_state_dict

RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


@dataclass
class PairRow:
    prompt: str
    image_path: str
    concept: str


class PromptImageDataset(Dataset):
    def __init__(self, rows: Sequence[PairRow], *, feature_mode: str, image_size: int) -> None:
        self.rows = list(rows)
        self.feature_mode = feature_mode
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        features = text_to_model_input(row.prompt, feature_mode=self.feature_mode).squeeze(0)
        image = Image.open(row.image_path).convert("RGB").resize((self.image_size, self.image_size), RESAMPLE_LANCZOS)
        image_bytes = torch.tensor(bytearray(image.tobytes()), dtype=torch.uint8)
        image_tensor = image_bytes.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        return features, image_tensor, row.prompt


def _load_rows(path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            rows.append(
                PairRow(
                    prompt=str(obj.get("prompt", "")),
                    image_path=str(obj.get("image_path", "")),
                    concept=str(obj.get("concept", "")),
                )
            )
    if not rows:
        raise ValueError(f"No prompt-image pairs found in {path}")
    return rows


def _split_rows(rows: Sequence[PairRow], *, seed: int, val_fraction: float) -> tuple[List[PairRow], List[PairRow]]:
    items = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(items)
    val_n = max(1, int(round(len(items) * float(val_fraction))))
    val_n = min(val_n, max(1, len(items) - 1))
    return items[val_n:], items[:val_n]


def _resolve_feature_mode(teacher_meta: Path | None, student_meta: Path | None, explicit_mode: str) -> str:
    if explicit_mode:
        return explicit_mode
    for meta_path in (teacher_meta, student_meta):
        if meta_path is None or not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        value = str(meta.get("feature_mode", "")).strip()
        if value:
            return value
    return "context_mix_v4"


def _gradient_map(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = image[..., :, 1:] - image[..., :, :-1]
    grad_y = image[..., 1:, :] - image[..., :-1, :]
    return grad_x, grad_y


def _image_loss(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, Dict[str, float]]:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    pred_dx, pred_dy = _gradient_map(pred)
    tgt_dx, tgt_dy = _gradient_map(target)
    edge = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)
    loss = l1 + 0.25 * mse + 0.10 * edge
    return loss, {"l1": float(l1.item()), "mse": float(mse.item()), "edge": float(edge.item())}


def _teacher_image_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target) + 0.25 * F.mse_loss(pred, target)


def _teacher_text_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(student_logits, teacher_logits)


def _load_student_text_weights(model: ChampionNetUltraExpertNativeImageLite, weights_path: Path) -> Dict[str, int]:
    state_dict = safe_load_state_dict(str(weights_path))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("image_")]
    unexpected = list(incompatible.unexpected_keys)
    return {"missing_non_image": len(missing), "unexpected": len(unexpected)}


def _load_teacher_model(weights_path: Path, device: torch.device) -> ChampionNetFrontierCollectiveNativeImage:
    model = ChampionNetFrontierCollectiveNativeImage(image_size=64).to(device)
    state_dict = safe_load_state_dict(str(weights_path))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def _set_trainable_layers(model: ChampionNetUltraExpertNativeImageLite, *, freeze_text_backbone: bool, unfreeze_text_layers: int) -> None:
    for param in model.parameters():
        param.requires_grad = True
    if not freeze_text_backbone:
        return
    for param in model.layers.parameters():
        param.requires_grad = False
    keep = max(0, min(int(unfreeze_text_layers), len(model.layers)))
    if keep > 0:
        for layer in model.layers[-keep:]:
            for param in layer.parameters():
                param.requires_grad = True


def _count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(int(param.numel()) for param in model.parameters())
    trainable = sum(int(param.numel()) for param in model.parameters() if param.requires_grad)
    return {"total_parameters": total, "trainable_parameters": trainable}


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a lite native-image single-checkpoint model from v36.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--teacher_weights", required=True)
    parser.add_argument("--teacher_meta", default="")
    parser.add_argument("--student_base_weights", required=True)
    parser.add_argument("--student_base_meta", default="")
    parser.add_argument("--output_weights", required=True)
    parser.add_argument("--output_meta", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--sample_prompt", default="simple water cycle icon with blue water, white cloud, rain, and a yellow sun")
    parser.add_argument("--sample_output", default="")
    parser.add_argument("--feature_mode", default="")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=56)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze_text_backbone", action="store_true")
    parser.add_argument("--unfreeze_text_layers", type=int, default=4)
    parser.add_argument("--gt_image_weight", type=float, default=1.0)
    parser.add_argument("--teacher_image_weight", type=float, default=0.45)
    parser.add_argument("--teacher_text_weight", type=float, default=0.08)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    pairs_path = Path(args.pairs)
    teacher_weights = Path(args.teacher_weights)
    teacher_meta = Path(args.teacher_meta) if args.teacher_meta else None
    student_base_weights = Path(args.student_base_weights)
    student_base_meta = Path(args.student_base_meta) if args.student_base_meta else None
    output_weights = Path(args.output_weights)
    output_meta = Path(args.output_meta)
    summary_out = Path(args.summary_out)
    sample_output = Path(args.sample_output) if args.sample_output else None
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    if sample_output is not None:
        sample_output.parent.mkdir(parents=True, exist_ok=True)

    feature_mode = _resolve_feature_mode(teacher_meta, student_base_meta, str(args.feature_mode or ""))
    rows = _load_rows(pairs_path)
    train_rows, val_rows = _split_rows(rows, seed=int(args.seed), val_fraction=float(args.val_fraction))
    train_ds = PromptImageDataset(train_rows, feature_mode=feature_mode, image_size=int(args.image_size))
    val_ds = PromptImageDataset(val_rows, feature_mode=feature_mode, image_size=int(args.image_size))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    device = torch.device(str(args.device))
    teacher = _load_teacher_model(teacher_weights, device)
    student = ChampionNetUltraExpertNativeImageLite(image_size=int(args.image_size)).to(device)
    student_load_report = _load_student_text_weights(student, student_base_weights)
    _set_trainable_layers(
        student,
        freeze_text_backbone=bool(args.freeze_text_backbone),
        unfreeze_text_layers=int(args.unfreeze_text_layers),
    )
    param_counts = _count_parameters(student)

    trainable_params = [param for param in student.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected for v37 lite native image distillation.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_val_total = float("inf")
    best_val_parts: Dict[str, float] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        train_stats = {"total": 0.0, "gt_image": 0.0, "teacher_image": 0.0, "teacher_text": 0.0, "count": 0}
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                teacher_logits = teacher(xb)
                teacher_image = teacher.forward_image(xb)

            student_logits = student(xb)
            student_image = student.forward_image(xb)

            gt_image_loss, _ = _image_loss(student_image, yb)
            teacher_image_loss = _teacher_image_loss(student_image, teacher_image)
            teacher_text_loss = _teacher_text_loss(student_logits, teacher_logits)
            total_loss = (
                float(args.gt_image_weight) * gt_image_loss
                + float(args.teacher_image_weight) * teacher_image_loss
                + float(args.teacher_text_weight) * teacher_text_loss
            )

            total_loss.backward()
            optimizer.step()

            batch_n = int(xb.shape[0])
            train_stats["total"] += float(total_loss.item()) * batch_n
            train_stats["gt_image"] += float(gt_image_loss.item()) * batch_n
            train_stats["teacher_image"] += float(teacher_image_loss.item()) * batch_n
            train_stats["teacher_text"] += float(teacher_text_loss.item()) * batch_n
            train_stats["count"] += batch_n

        student.eval()
        val_stats = {"total": 0.0, "gt_image": 0.0, "teacher_image": 0.0, "teacher_text": 0.0, "count": 0}
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                teacher_logits = teacher(xb)
                teacher_image = teacher.forward_image(xb)
                student_logits = student(xb)
                student_image = student.forward_image(xb)

                gt_image_loss, _ = _image_loss(student_image, yb)
                teacher_image_loss = _teacher_image_loss(student_image, teacher_image)
                teacher_text_loss = _teacher_text_loss(student_logits, teacher_logits)
                total_loss = (
                    float(args.gt_image_weight) * gt_image_loss
                    + float(args.teacher_image_weight) * teacher_image_loss
                    + float(args.teacher_text_weight) * teacher_text_loss
                )

                batch_n = int(xb.shape[0])
                val_stats["total"] += float(total_loss.item()) * batch_n
                val_stats["gt_image"] += float(gt_image_loss.item()) * batch_n
                val_stats["teacher_image"] += float(teacher_image_loss.item()) * batch_n
                val_stats["teacher_text"] += float(teacher_text_loss.item()) * batch_n
                val_stats["count"] += batch_n

        epoch_record = {"epoch": float(epoch)}
        for key in ("total", "gt_image", "teacher_image", "teacher_text"):
            epoch_record[f"train_{key}"] = train_stats[key] / max(1, train_stats["count"])
            epoch_record[f"val_{key}"] = val_stats[key] / max(1, val_stats["count"])
        history.append(epoch_record)
        print(
            "[v37-lite-distill] "
            f"epoch={epoch} "
            f"train_total={epoch_record['train_total']:.4f} "
            f"val_total={epoch_record['val_total']:.4f} "
            f"val_gt_image={epoch_record['val_gt_image']:.4f} "
            f"val_teacher_image={epoch_record['val_teacher_image']:.4f} "
            f"val_teacher_text={epoch_record['val_teacher_text']:.4f}"
        )

        if epoch_record["val_total"] < best_val_total:
            best_val_total = epoch_record["val_total"]
            best_val_parts = {
                "val_gt_image": epoch_record["val_gt_image"],
                "val_teacher_image": epoch_record["val_teacher_image"],
                "val_teacher_text": epoch_record["val_teacher_text"],
            }
            best_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Distillation finished without producing a best student state.")

    student.load_state_dict(best_state, strict=True)
    torch.save(best_state, output_weights)

    meta_payload = {
        "variant": "v37_native_image_lite_single_checkpoint",
        "teacher_weights": str(teacher_weights),
        "teacher_meta": str(teacher_meta) if teacher_meta else "",
        "student_base_weights": str(student_base_weights),
        "student_base_meta": str(student_base_meta) if student_base_meta else "",
        "feature_mode": feature_mode,
        "image_size": int(args.image_size),
        "pairs_path": str(pairs_path),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "freeze_text_backbone": bool(args.freeze_text_backbone),
        "unfreeze_text_layers": int(args.unfreeze_text_layers),
        "best_val_total_loss": best_val_total,
        "best_val_parts": best_val_parts,
        "sample_prompt": str(args.sample_prompt),
        "student_load_report": student_load_report,
        "student_param_counts": param_counts,
        "loss_weights": {
            "gt_image_weight": float(args.gt_image_weight),
            "teacher_image_weight": float(args.teacher_image_weight),
            "teacher_text_weight": float(args.teacher_text_weight),
        },
    }
    output_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "meta": meta_payload,
        "history": history,
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if sample_output is not None:
        student.eval()
        save_prompt_image(student, str(args.sample_prompt), sample_output, feature_mode=feature_mode, device=device)
        print(f"[v37-lite-distill] sample={sample_output}")

    print(f"[v37-lite-distill] weights={output_weights}")
    print(f"[v37-lite-distill] meta={output_meta}")
    print(f"[v37-lite-distill] summary={summary_out}")


if __name__ == "__main__":
    main()
