from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from three_d_generation_model import (
        THREE_D_CONCEPT_LABELS,
        THREE_D_DISPLAY_NAMES,
        THREE_D_GENERATION_SPECS,
        THREE_D_CONCEPT_TO_INDEX,
        ThreeDGenerationMiniNet,
        build_vocab,
        encode_text,
        encode_words,
    )
except ImportError:  # pragma: no cover
    from .three_d_generation_model import (
        THREE_D_CONCEPT_LABELS,
        THREE_D_DISPLAY_NAMES,
        THREE_D_GENERATION_SPECS,
        THREE_D_CONCEPT_TO_INDEX,
        ThreeDGenerationMiniNet,
        build_vocab,
        encode_text,
        encode_words,
    )


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


@dataclass
class ThreeDRow:
    prompt: str
    concept: str
    variant: str
    answer: str


class PromptDataset(Dataset):
    def __init__(self, rows: Sequence[ThreeDRow], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.word_buckets = int(word_buckets)
        self.max_words = int(max_words)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        char_ids = torch.tensor(encode_text(row.prompt, self.vocab, self.max_len), dtype=torch.long)
        word_ids = torch.tensor(
            encode_words(row.prompt, word_buckets=self.word_buckets, max_words=self.max_words),
            dtype=torch.long,
        )
        target = torch.tensor(THREE_D_CONCEPT_TO_INDEX[row.concept], dtype=torch.long)
        return char_ids, word_ids, target


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _concept_answers() -> Dict[str, Dict[str, str]]:
    return {
        str(item["concept"]): {str(key): str(value) for key, value in dict(item["answers"]).items()}
        for item in THREE_D_GENERATION_SPECS
    }


def _prompt_templates_for(concept: str, display: str) -> List[Tuple[str, str]]:
    if concept == "hollow_box":
        return [
            ("Write a simple OpenSCAD example for a hollow box with 2 mm walls.", "code"),
            ("How do I make a parametric hollow box in OpenSCAD?", "explain_then_code"),
            ("Generate code only for a shell box in OpenSCAD.", "code"),
            ("Give me a concise OpenSCAD snippet for a rectangular enclosure.", "concise_answer"),
            ("Show an editable OpenSCAD hollow box example.", "explain_then_code"),
        ]
    if concept == "rounded_plate_holes":
        return [
            ("How do I model a rounded plate with four mounting holes in OpenSCAD?", "explain_then_code"),
            ("Write code for a rounded mounting plate with four holes.", "code"),
            ("Give a concise OpenSCAD example for a rounded rectangle plate with corner holes.", "code"),
            ("Best parametric approach for a rounded plate with mounting holes?", "explain_then_code"),
            ("Generate an OpenSCAD mount plate with four bolt holes.", "code"),
        ]
    if concept == "phone_stand":
        return [
            ("Give me an OpenSCAD example for a parametric phone stand.", "code"),
            ("How would you model a simple desk phone stand in OpenSCAD?", "explain_then_code"),
            ("Generate code only for a small phone stand.", "code"),
            ("Make a concise OpenSCAD snippet for a phone stand.", "concise_answer"),
            ("Show an editable parametric phone stand model.", "explain_then_code"),
        ]
    if concept == "cylinder_center_hole":
        return [
            ("Show a concise OpenSCAD snippet for a cylinder with a centered hole.", "code"),
            ("How do I model a tube or spacer with a centered hole in OpenSCAD?", "explain_then_code"),
            ("Write code only for a cylinder with a through-hole.", "code"),
            ("Generate an OpenSCAD example for a ring spacer.", "code"),
            ("What is the shortest OpenSCAD way to cut a centered hole through a cylinder?", "concise_answer"),
        ]
    if concept == "radial_spokes":
        return [
            ("How do I repeat spokes evenly around a circle in OpenSCAD?", "explain_then_code"),
            ("Generate OpenSCAD code for radial spokes around a hub.", "code"),
            ("Show a concise snippet for evenly spaced spokes.", "concise_answer"),
            ("How would I place twelve spokes around a circle in OpenSCAD?", "explain_then_code"),
            ("Write code only for wheel-like radial spokes.", "code"),
        ]
    if concept == "countersunk_hole":
        return [
            ("Write a tiny OpenSCAD module for a countersunk screw hole.", "code"),
            ("How do I model a countersunk hole in OpenSCAD?", "explain_then_code"),
            ("Generate code only for a countersunk fastener hole.", "code"),
            ("Give a concise OpenSCAD countersunk hole snippet.", "concise_answer"),
            ("Show an editable countersunk-hole module.", "explain_then_code"),
        ]
    if concept == "mirrored_feature":
        return [
            ("How do I mirror a feature onto both sides of a part in OpenSCAD?", "explain_then_code"),
            ("Write code for a mirrored feature on both sides of a part.", "code"),
            ("Show a concise example using mirror() in OpenSCAD.", "concise_answer"),
            ("Generate code only for a symmetric mirrored feature.", "code"),
            ("Best way to keep mirrored geometry editable in OpenSCAD?", "explain_then_code"),
        ]
    if concept == "decorative_star":
        return [
            ("Give an OpenSCAD example for a gear-like decorative star without importing libraries.", "code"),
            ("How do I make a star polygon in OpenSCAD and extrude it?", "explain_then_code"),
            ("Generate code only for a decorative star shape.", "code"),
            ("Show a concise OpenSCAD star snippet.", "concise_answer"),
            ("Create a parametric extruded star in OpenSCAD.", "code"),
        ]
    if concept == "boolean_ops":
        return [
            ("What is the difference between union, difference, and intersection in OpenSCAD?", "explain_then_code"),
            ("Explain OpenSCAD boolean operations in plain language.", "explain_then_code"),
            ("Give a concise answer about union(), difference(), and intersection().", "concise_answer"),
            ("When should I use difference() instead of union() in OpenSCAD?", "explain_then_code"),
            ("Explain boolean ops in OpenSCAD, no code block.", "concise_answer"),
        ]
    if concept == "editable_models":
        return [
            ("Best practice for keeping OpenSCAD models editable?", "explain_then_code"),
            ("How do I keep a parametric OpenSCAD design maintainable?", "explain_then_code"),
            ("Give a concise answer for editable OpenSCAD models.", "concise_answer"),
            ("What is the best parametric modeling pattern in OpenSCAD?", "explain_then_code"),
            ("Explain how to keep OpenSCAD parts easy to edit later.", "explain_then_code"),
        ]
    if concept == "pyramid":
        return [
            ("Generate an OpenSCAD square pyramid using polyhedron().", "code"),
            ("How do I build a square pyramid in OpenSCAD?", "explain_then_code"),
            ("Write code only for a pyramid mesh in OpenSCAD.", "code"),
            ("Show a concise pyramid polyhedron snippet.", "concise_answer"),
            ("Create a simple 3D pyramid model for OpenSCAD.", "code"),
        ]
    if concept == "triangular_prism":
        return [
            ("Generate an OpenSCAD triangular prism using polyhedron().", "code"),
            ("How do I model a triangular prism in OpenSCAD?", "explain_then_code"),
            ("Write code only for a prism mesh.", "code"),
            ("Show a concise triangular prism snippet.", "concise_answer"),
            ("Create a 3D triangular prism model in OpenSCAD.", "code"),
        ]
    if concept == "tetrahedron":
        return [
            ("Generate an OpenSCAD tetrahedron with polyhedron().", "code"),
            ("How do I make a tetrahedron in OpenSCAD?", "explain_then_code"),
            ("Write code only for a tetrahedron mesh.", "code"),
            ("Show a concise tetrahedron snippet.", "concise_answer"),
            ("Create a simple 3D tetrahedron model in OpenSCAD.", "code"),
        ]
    if concept == "grid_plane":
        return [
            ("Generate an OpenSCAD flat grid plane.", "code"),
            ("How would I model a simple floor grid in OpenSCAD?", "explain_then_code"),
            ("Write code only for a grid plane.", "code"),
            ("Show a concise OpenSCAD grid-plane example.", "concise_answer"),
            ("Create a thin grid mesh or base plane in OpenSCAD.", "code"),
        ]
    return [
        (f"Generate a small OpenSCAD example for {display}.", "code"),
        (f"How do I model {display} in OpenSCAD?", "explain_then_code"),
        (f"Give a concise OpenSCAD snippet for {display}.", "concise_answer"),
    ]


def build_rows(seed: int) -> Tuple[List[ThreeDRow], Dict[str, object], Dict[str, Dict[str, str]]]:
    rng = _rng(seed)
    answer_bank = _concept_answers()
    rows: List[ThreeDRow] = []
    source_counts: Counter[str] = Counter()

    for concept in THREE_D_CONCEPT_LABELS:
        display = THREE_D_DISPLAY_NAMES.get(concept, concept.replace("_", " "))
        variants = answer_bank.get(concept, {})
        for prompt, variant in _prompt_templates_for(concept, display):
            answer = (
                variants.get(variant)
                or variants.get("code")
                or variants.get("explain_then_code")
                or next(iter(variants.values()))
            )
            rows.append(ThreeDRow(prompt=prompt, concept=concept, variant=variant, answer=answer))
            source_counts["curated_3d_templates"] += 1
            prefixed = f"3D modeling request: {prompt}"
            rows.append(ThreeDRow(prompt=prefixed, concept=concept, variant=variant, answer=answer))
            source_counts["prefixed_3d_templates"] += 1
        if concept in {"pyramid", "triangular_prism", "tetrahedron", "grid_plane"}:
            mesh_prompt = f"Write OpenSCAD for a {display} using polyhedron or simple primitives."
            mesh_answer = variants.get("code") or variants.get("explain_then_code") or next(iter(variants.values()))
            rows.append(ThreeDRow(prompt=mesh_prompt, concept=concept, variant="code", answer=mesh_answer))
            source_counts["shape_mesh_prompts"] += 1

    rng.shuffle(rows)
    summary: Dict[str, object] = {
        "source_rows": len(rows),
        "source_counts": dict(sorted(source_counts.items())),
        "concepts": len(THREE_D_CONCEPT_LABELS),
    }
    return rows, summary, answer_bank


def split_rows(rows: Sequence[ThreeDRow], seed: int) -> Tuple[List[ThreeDRow], List[ThreeDRow]]:
    rng = _rng(seed)
    by_concept: Dict[str, List[ThreeDRow]] = defaultdict(list)
    for row in rows:
        by_concept[row.concept].append(row)
    train_rows: List[ThreeDRow] = []
    val_rows: List[ThreeDRow] = []
    for concept in sorted(by_concept):
        bucket = list(by_concept[concept])
        rng.shuffle(bucket)
        pivot = max(1, int(len(bucket) * 0.18))
        val_rows.extend(bucket[:pivot])
        train_rows.extend(bucket[pivot:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def accuracy_for_model(model: ThreeDGenerationMiniNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for char_ids, word_ids, target in loader:
            char_ids = char_ids.to(device)
            word_ids = word_ids.to(device)
            target = target.to(device)
            pred = model(char_ids, word_ids).argmax(dim=1)
            total += int(target.numel())
            correct += int((pred == target).sum().item())
    return float(correct / max(total, 1))


def sample_predictions(
    model: ThreeDGenerationMiniNet,
    rows: Sequence[ThreeDRow],
    vocab: Dict[str, int],
    *,
    max_len: int,
    word_buckets: int,
    max_words: int,
    device: torch.device,
    count: int = 5,
) -> List[Dict[str, object]]:
    selected = list(rows[:count])
    if not selected:
        return []
    char_ids = torch.tensor([encode_text(row.prompt, vocab, max_len) for row in selected], dtype=torch.long, device=device)
    word_ids = torch.tensor(
        [encode_words(row.prompt, word_buckets=word_buckets, max_words=max_words) for row in selected],
        dtype=torch.long,
        device=device,
    )
    with torch.inference_mode():
        logits = model(char_ids, word_ids)
        probs = torch.softmax(logits, dim=1).detach().cpu()
    samples: List[Dict[str, object]] = []
    for row, prob in zip(selected, probs):
        idx = int(prob.argmax().item())
        predicted = THREE_D_CONCEPT_LABELS[idx]
        samples.append(
            {
                "prompt": row.prompt,
                "expected_concept": row.concept,
                "predicted_concept": predicted,
                "predicted_label": THREE_D_DISPLAY_NAMES.get(predicted, predicted.replace("_", " ")),
                "confidence": round(float(prob[idx].item()), 4),
            }
        )
    return samples


def train_model(
    *,
    output_dir: Path,
    models_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rows, row_summary, answer_bank = build_rows(seed=seed)
    train_rows, val_rows = split_rows(rows, seed=seed + 101)
    vocab = build_vocab([row.prompt for row in train_rows], min_frequency=1)
    max_len = 224
    max_words = 28
    word_buckets = 512
    train_dataset = PromptDataset(train_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    val_dataset = PromptDataset(val_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = ThreeDGenerationMiniNet(
        vocab_size=len(vocab),
        num_concepts=len(THREE_D_CONCEPT_LABELS),
        char_embed_dim=28,
        conv_channels=56,
        word_buckets=word_buckets,
        word_embed_dim=20,
        hidden_dim=112,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for char_ids, word_ids, target in train_loader:
            char_ids = char_ids.to(device)
            word_ids = word_ids.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(char_ids, word_ids)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.size(0))
            total_examples += int(target.size(0))
        train_acc = accuracy_for_model(model, train_loader, device)
        val_acc = accuracy_for_model(model, val_loader, device)
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
        raise RuntimeError("No model state was captured during training.")
    model.load_state_dict(best_state)
    train_acc = accuracy_for_model(model, train_loader, device)
    val_acc = accuracy_for_model(model, val_loader, device)

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_3d_generation_micro_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "three_d_generation_micro_v1.pth"
    meta_path = artifact_dir / "three_d_generation_micro_v1_meta.json"
    summary_path = artifact_dir / "three_d_generation_micro_v1_summary.json"
    zip_path = output_dir / f"supermix_3d_generation_micro_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    concept_distribution = Counter(row.concept for row in rows)
    meta = {
        "labels": list(THREE_D_CONCEPT_LABELS),
        "display_names": dict(THREE_D_DISPLAY_NAMES),
        "vocab": vocab,
        "max_len": max_len,
        "max_words": max_words,
        "word_buckets": word_buckets,
        "char_embed_dim": 28,
        "conv_channels": 56,
        "word_embed_dim": 20,
        "hidden_dim": 112,
        "answer_bank": answer_bank,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "parameter_count": parameter_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "concept_distribution": dict(sorted(concept_distribution.items())),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "row_summary": {
            **row_summary,
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "history": history,
        "sample_predictions": sample_predictions(
            model,
            val_rows,
            vocab,
            max_len=max_len,
            word_buckets=word_buckets,
            max_words=max_words,
            device=device,
            count=6,
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    source_files = [
        weights_path,
        meta_path,
        summary_path,
        SOURCE_DIR / "three_d_generation_model.py",
        Path(__file__).resolve(),
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_files:
            archive.write(file_path, arcname=file_path.name)

    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)

    return {
        "artifact_dir": str(artifact_dir),
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "meta_path": str(meta_path),
        "summary_path": str(summary_path),
        "parameter_count": parameter_count,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "row_summary": summary["row_summary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small 3D-generation specialist model.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=9e-4)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
