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
    from protein_folding_model import (
        CONCEPT_DISPLAY_NAMES,
        CONCEPT_TO_INDEX,
        PROTEIN_CONCEPT_LABELS,
        ProteinFoldingMiniNet,
        build_vocab,
        encode_text,
        encode_words,
    )
    from v40_benchmax_common import PROTEIN_FOLDING_CONCEPTS, build_protein_folding_rows
except ImportError:  # pragma: no cover
    from .protein_folding_model import (
        CONCEPT_DISPLAY_NAMES,
        CONCEPT_TO_INDEX,
        PROTEIN_CONCEPT_LABELS,
        ProteinFoldingMiniNet,
        build_vocab,
        encode_text,
        encode_words,
    )
    from .v40_benchmax_common import PROTEIN_FOLDING_CONCEPTS, build_protein_folding_rows


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


@dataclass
class ProteinRow:
    prompt: str
    concept: str
    variant: str
    answer: str


class PromptDataset(Dataset):
    def __init__(self, rows: Sequence[ProteinRow], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int) -> None:
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
        target = torch.tensor(CONCEPT_TO_INDEX[row.concept], dtype=torch.long)
        return char_ids, word_ids, target


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _base_answer_bank(seed: int) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
    rows, summary = build_protein_folding_rows(seed=seed, max_rows=0)
    bank: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        meta = dict(row.get("metadata") or {})
        concept = str(meta.get("concept") or "")
        variant = str(meta.get("variant") or "concept_explain")
        answer = str(row.get("response_text") or row.get("assistant") or "").strip()
        if concept and answer:
            bank[concept][variant] = answer
    return {key: dict(value) for key, value in bank.items()}, dict(summary)


def _concept_display(concept: str, prompt: str) -> str:
    display = CONCEPT_DISPLAY_NAMES.get(concept)
    if display:
        return display
    base = str(prompt or "")
    if "?" in base:
        base = base.split("?", 1)[0]
    return base.strip() or concept.replace("_", " ")


def build_rows(seed: int) -> Tuple[List[ProteinRow], Dict[str, object], Dict[str, Dict[str, str]]]:
    answer_bank, base_summary = _base_answer_bank(seed)
    rows: List[ProteinRow] = []
    source_counts: Counter[str] = Counter()

    for concept in PROTEIN_CONCEPT_LABELS:
        variants = answer_bank.get(concept, {})
        concept_item = next(item for item in PROTEIN_FOLDING_CONCEPTS if item["concept"] == concept)
        display = _concept_display(concept, concept_item["prompt"])
        prompt_specs = [
            ("concept_explain", f"What is {display} in protein folding?"),
            ("concept_explain", f"Define {display} for a protein-folding discussion."),
            ("concise_answer", f"In one sentence, what does {display} mean in protein folding?"),
            ("student_paragraph", f"Teach a beginner the idea of {display} in protein folding."),
            ("structure_prediction_link", f"Why does {display} matter for protein structure prediction?"),
            ("model_debugging", f"You are debugging a protein model. Why should you check {display}?"),
            ("failure_mode", f"What can go wrong if a model ignores {display}?"),
            ("ranking_and_confidence", f"How does {display} affect model ranking or confidence?"),
            ("spatial_reasoning", f"How is {display} connected to 3D structure reasoning?"),
            ("error_correction", f"Correct this misconception about {display} in protein folding."),
        ]
        for variant, prompt in prompt_specs:
            answer = variants.get(variant) or variants.get("concept_explain")
            if not answer:
                continue
            rows.append(ProteinRow(prompt=prompt, concept=concept, variant=variant, answer=answer))
            source_counts["synthetic_prompt_templates"] += 1

    base_rows, _base_rows_summary = build_protein_folding_rows(seed=seed + 17, max_rows=0)
    for row in base_rows:
        meta = dict(row.get("metadata") or {})
        concept = str(meta.get("concept") or "")
        variant = str(meta.get("variant") or "concept_explain")
        prompt = str(row.get("prompt") or "").strip()
        answer = str(row.get("response_text") or row.get("assistant") or "").strip()
        if concept and prompt and answer:
            rows.append(ProteinRow(prompt=prompt, concept=concept, variant=variant, answer=answer))
            source_counts["v40_protein_pack"] += 1

    _rng(seed + 29).shuffle(rows)
    summary: Dict[str, object] = {
        "source_rows": len(rows),
        "base_summary": base_summary,
        "source_counts": dict(sorted(source_counts.items())),
        "concepts": len(PROTEIN_CONCEPT_LABELS),
    }
    return rows, summary, answer_bank


def split_rows(rows: Sequence[ProteinRow], seed: int) -> Tuple[List[ProteinRow], List[ProteinRow]]:
    rng = _rng(seed)
    by_concept: Dict[str, List[ProteinRow]] = defaultdict(list)
    for row in rows:
        by_concept[row.concept].append(row)
    train_rows: List[ProteinRow] = []
    val_rows: List[ProteinRow] = []
    for concept in sorted(by_concept):
        bucket = list(by_concept[concept])
        rng.shuffle(bucket)
        pivot = max(1, int(len(bucket) * 0.18))
        val_rows.extend(bucket[:pivot])
        train_rows.extend(bucket[pivot:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def accuracy_for_model(model: ProteinFoldingMiniNet, loader: DataLoader, device: torch.device) -> float:
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


def sample_predictions(model: ProteinFoldingMiniNet, rows: Sequence[ProteinRow], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int, device: torch.device, count: int = 5) -> List[Dict[str, object]]:
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
        predicted = PROTEIN_CONCEPT_LABELS[idx]
        samples.append(
            {
                "prompt": row.prompt,
                "expected_concept": row.concept,
                "predicted_concept": predicted,
                "predicted_label": CONCEPT_DISPLAY_NAMES.get(predicted, predicted.replace("_", " ")),
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
    word_buckets = 384
    train_dataset = PromptDataset(train_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    val_dataset = PromptDataset(val_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = ProteinFoldingMiniNet(
        vocab_size=len(vocab),
        num_concepts=len(PROTEIN_CONCEPT_LABELS),
        char_embed_dim=28,
        conv_channels=56,
        word_buckets=word_buckets,
        word_embed_dim=18,
        hidden_dim=96,
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
    artifact_dir = output_dir / f"supermix_protein_folding_micro_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "protein_folding_micro_v1.pth"
    meta_path = artifact_dir / "protein_folding_micro_v1_meta.json"
    summary_path = artifact_dir / "protein_folding_micro_v1_summary.json"
    zip_path = output_dir / f"supermix_protein_folding_micro_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    concept_distribution = Counter(row.concept for row in rows)
    meta = {
        "labels": list(PROTEIN_CONCEPT_LABELS),
        "display_names": dict(CONCEPT_DISPLAY_NAMES),
        "vocab": vocab,
        "max_len": max_len,
        "max_words": max_words,
        "word_buckets": word_buckets,
        "char_embed_dim": 28,
        "conv_channels": 56,
        "word_embed_dim": 18,
        "hidden_dim": 96,
        "answer_bank": answer_bank,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "parameter_count": parameter_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "artifact": zip_path.name,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "parameter_count": parameter_count,
        "history": history[-12:],
        "row_summary": row_summary,
        "concept_distribution": dict(sorted(concept_distribution.items())),
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
    parser = argparse.ArgumentParser(description="Train a mini protein-folding specialist model and package it as a Supermix zip artifact.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--learning_rate", type=float, default=0.0024)
    parser.add_argument("--seed", type=int, default=1337)
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
