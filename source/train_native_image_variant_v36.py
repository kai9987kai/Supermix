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
from model_frontier_native_image_v36 import ChampionNetFrontierCollectiveNativeImage, save_prompt_image
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


def _resolve_feature_mode(base_meta: Path | None, explicit_mode: str) -> str:
    if explicit_mode:
        return explicit_mode
    if base_meta is not None and base_meta.exists():
        try:
            meta = json.loads(base_meta.read_text(encoding="utf-8"))
            value = str(meta.get("feature_mode", "")).strip()
            if value:
                return value
        except Exception:
            pass
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


def _load_base_weights(model: ChampionNetFrontierCollectiveNativeImage, base_weights: Path) -> Dict[str, int]:
    state_dict = safe_load_state_dict(str(base_weights))
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if not key.startswith("image_")]
    unexpected = list(incompatible.unexpected_keys)
    return {"missing_non_image": len(missing), "unexpected": len(unexpected)}


def _set_trainable_layers(model: ChampionNetFrontierCollectiveNativeImage, *, freeze_text_backbone: bool, unfreeze_text_layers: int) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the native-image v36 single-checkpoint variant.")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--base_weights", required=True)
    parser.add_argument("--base_meta", default="")
    parser.add_argument("--output_weights", required=True)
    parser.add_argument("--output_meta", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--sample_prompt", default="simple water cycle icon with cloud, rain, sun, and blue water")
    parser.add_argument("--sample_output", default="")
    parser.add_argument("--feature_mode", default="")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze_text_backbone", action="store_true")
    parser.add_argument("--unfreeze_text_layers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    random.seed(int(args.seed))

    pairs_path = Path(args.pairs)
    base_weights = Path(args.base_weights)
    base_meta = Path(args.base_meta) if args.base_meta else None
    output_weights = Path(args.output_weights)
    output_meta = Path(args.output_meta)
    summary_out = Path(args.summary_out)
    sample_output = Path(args.sample_output) if args.sample_output else None
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    if sample_output is not None:
        sample_output.parent.mkdir(parents=True, exist_ok=True)

    feature_mode = _resolve_feature_mode(base_meta, str(args.feature_mode or ""))
    rows = _load_rows(pairs_path)
    train_rows, val_rows = _split_rows(rows, seed=int(args.seed), val_fraction=float(args.val_fraction))
    train_ds = PromptImageDataset(train_rows, feature_mode=feature_mode, image_size=int(args.image_size))
    val_ds = PromptImageDataset(val_rows, feature_mode=feature_mode, image_size=int(args.image_size))
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False)

    device = torch.device(str(args.device))
    model = ChampionNetFrontierCollectiveNativeImage(image_size=int(args.image_size)).to(device)
    load_report = _load_base_weights(model, base_weights)
    _set_trainable_layers(
        model,
        freeze_text_backbone=bool(args.freeze_text_backbone),
        unfreeze_text_layers=int(args.unfreeze_text_layers),
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected for v36 native image training.")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    best_state = None
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for xb, yb, _ in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_image(xb)
            loss, parts = _image_loss(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.shape[0]
            train_count += int(xb.shape[0])

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model.forward_image(xb)
                loss, parts = _image_loss(pred, yb)
                val_loss += float(loss.item()) * xb.shape[0]
                val_count += int(xb.shape[0])

        train_loss = train_loss / max(1, train_count)
        val_loss = val_loss / max(1, val_count)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss})
        print(f"[v36-native-train] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training finished without producing a best model state.")

    model.load_state_dict(best_state, strict=True)
    torch.save(best_state, output_weights)

    meta_payload = {
        "variant": "v36_native_image_single_checkpoint",
        "base_weights": str(base_weights),
        "base_meta": str(base_meta) if base_meta else "",
        "feature_mode": feature_mode,
        "image_size": int(args.image_size),
        "pairs_path": str(pairs_path),
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "freeze_text_backbone": bool(args.freeze_text_backbone),
        "unfreeze_text_layers": int(args.unfreeze_text_layers),
        "best_val_loss": best_val,
        "sample_prompt": str(args.sample_prompt),
        "load_report": load_report,
    }
    output_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    summary_payload = {
        "meta": meta_payload,
        "history": history,
    }
    summary_out.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    if sample_output is not None:
        model.eval()
        save_prompt_image(model, str(args.sample_prompt), sample_output, feature_mode=feature_mode, device=device)
        print(f"[v36-native-train] sample={sample_output}")

    print(f"[v36-native-train] weights={output_weights}")
    print(f"[v36-native-train] meta={output_meta}")
    print(f"[v36-native-train] summary={summary_out}")


if __name__ == "__main__":
    main()
