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
    from mattergen_generation_model import (
        MATTERGEN_CONCEPT_LABELS,
        MATTERGEN_CONCEPT_SPECS,
        MATTERGEN_CONCEPT_TO_INDEX,
        MATTERGEN_DISPLAY_NAMES,
        PAPER_INSPIRATIONS,
        MatterGenMicroNet,
        build_vocab,
        encode_text,
        encode_words,
        vectorize_batch,
    )
except ImportError:  # pragma: no cover
    from .mattergen_generation_model import (
        MATTERGEN_CONCEPT_LABELS,
        MATTERGEN_CONCEPT_SPECS,
        MATTERGEN_CONCEPT_TO_INDEX,
        MATTERGEN_DISPLAY_NAMES,
        PAPER_INSPIRATIONS,
        MatterGenMicroNet,
        build_vocab,
        encode_text,
        encode_words,
        vectorize_batch,
    )


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


@dataclass
class MatterGenRow:
    prompt: str
    concept: str
    variant: str
    answer: str


class PromptDataset(Dataset):
    def __init__(self, rows: Sequence[MatterGenRow], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.word_buckets = int(word_buckets)
        self.max_words = int(max_words)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        char_ids = torch.tensor(encode_text(row.prompt, self.vocab, self.max_len), dtype=torch.long)
        word_ids = torch.tensor(encode_words(row.prompt, word_buckets=self.word_buckets, max_words=self.max_words), dtype=torch.long)
        _chars, _words, cond = vectorize_batch([row.prompt], self.vocab, max_len=self.max_len, word_buckets=self.word_buckets, max_words=self.max_words)
        target = torch.tensor(MATTERGEN_CONCEPT_TO_INDEX[row.concept], dtype=torch.long)
        return char_ids, word_ids, cond[0], target


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _cif_answer(spec: Dict[str, object]) -> str:
    prototype = dict(spec["prototype"])
    lattice = dict(prototype["lattice"])
    lines = [
        f"Candidate family: {spec['display_name']}",
        f"Prototype seed: {prototype['name']}",
        f"Formula: {prototype['formula']}",
        f"Space group: {prototype['space_group']}",
        f"Lattice: a={lattice['a']:.2f} A, b={lattice['b']:.2f} A, c={lattice['c']:.2f} A",
        f"Grounding note: validate {prototype['screening_focus']}.",
    ]
    return "\n".join(lines)


def build_rows(seed: int) -> Tuple[List[MatterGenRow], Dict[str, object]]:
    rows: List[MatterGenRow] = []
    source_counts: Counter[str] = Counter()
    for spec in MATTERGEN_CONCEPT_SPECS:
        concept = str(spec["concept"])
        display = str(spec["display_name"])
        tags = ", ".join(spec["property_tags"])
        answer = _cif_answer(spec)
        prompts = [
            ("candidate_card", f"Generate a {display} candidate for materials discovery."),
            ("candidate_card", f"Design a crystalline {display} with grounded property conditioning."),
            ("cif_seed", f"Give me a CIF-style seed for a {display}."),
            ("design_rationale", f"Why would you choose a {display} for targets around {tags}?"),
            ("validation_plan", f"How should I validate a generated {display} candidate before trusting it?"),
            ("concise_answer", f"Briefly suggest a {display} candidate."),
            ("candidate_card", f"Suggest a space-group-aware {display} seed with lattice parameters."),
            ("candidate_card", f"Generate a candidate material in the {display} family with stability checks."),
            ("candidate_card", f"Design a {display} with a realistic energy-above-hull target."),
            ("candidate_card", f"Generate a mattergen-style {display} seed for screening."),
        ]
        keyword_prompts = [
            f"Generate a candidate using the keyword {keyword} in a {display} prompt."
            for keyword in tuple(spec["keywords"])[:4]
        ]
        for variant, prompt in prompts:
            rows.append(MatterGenRow(prompt=prompt, concept=concept, variant=variant, answer=answer))
            source_counts["prompt_templates"] += 1
        for prompt in keyword_prompts:
            rows.append(MatterGenRow(prompt=prompt, concept=concept, variant="candidate_card", answer=answer))
            source_counts["keyword_templates"] += 1
    _rng(seed).shuffle(rows)
    return rows, {"source_rows": len(rows), "source_counts": dict(sorted(source_counts.items())), "concepts": len(MATTERGEN_CONCEPT_LABELS)}


def split_rows(rows: Sequence[MatterGenRow], seed: int) -> Tuple[List[MatterGenRow], List[MatterGenRow]]:
    rng = _rng(seed)
    by_concept: Dict[str, List[MatterGenRow]] = defaultdict(list)
    for row in rows:
        by_concept[row.concept].append(row)
    train_rows: List[MatterGenRow] = []
    val_rows: List[MatterGenRow] = []
    for concept in sorted(by_concept):
        bucket = list(by_concept[concept])
        rng.shuffle(bucket)
        pivot = max(1, int(len(bucket) * 0.2))
        val_rows.extend(bucket[:pivot])
        train_rows.extend(bucket[pivot:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def accuracy_for_model(model: MatterGenMicroNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.inference_mode():
        for char_ids, word_ids, cond, target in loader:
            pred = model(char_ids.to(device), word_ids.to(device), cond.to(device)).argmax(dim=1)
            target = target.to(device)
            total += int(target.numel())
            correct += int((pred == target).sum().item())
    return float(correct / max(total, 1))


def sample_predictions(model: MatterGenMicroNet, rows: Sequence[MatterGenRow], vocab: Dict[str, int], *, max_len: int, word_buckets: int, max_words: int, device: torch.device, count: int = 6) -> List[Dict[str, object]]:
    selected = list(rows[:count])
    if not selected:
        return []
    char_ids, word_ids, cond = vectorize_batch([row.prompt for row in selected], vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    with torch.inference_mode():
        logits = model(char_ids.to(device), word_ids.to(device), cond.to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu()
    out: List[Dict[str, object]] = []
    for row, prob in zip(selected, probs):
        idx = int(prob.argmax().item())
        predicted = MATTERGEN_CONCEPT_LABELS[idx]
        out.append({"prompt": row.prompt, "expected_concept": row.concept, "predicted_concept": predicted, "predicted_label": MATTERGEN_DISPLAY_NAMES.get(predicted, predicted), "confidence": round(float(prob[idx].item()), 4)})
    return out


def train_model(*, output_dir: Path, models_dir: Path, epochs: int, batch_size: int, learning_rate: float, seed: int) -> Dict[str, object]:
    torch.manual_seed(seed)
    rows, row_summary = build_rows(seed=seed)
    train_rows, val_rows = split_rows(rows, seed=seed + 101)
    vocab = build_vocab([row.prompt for row in train_rows], min_frequency=1)
    max_len = 256
    max_words = 32
    word_buckets = 512
    train_dataset = PromptDataset(train_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    val_dataset = PromptDataset(val_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = MatterGenMicroNet(
        vocab_size=len(vocab),
        num_concepts=len(MATTERGEN_CONCEPT_LABELS),
        char_embed_dim=30,
        conv_channels=64,
        word_buckets=word_buckets,
        word_embed_dim=18,
        condition_dim=8,
        hidden_dim=120,
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
        for char_ids, word_ids, cond, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(char_ids.to(device), word_ids.to(device), cond.to(device))
            loss = criterion(logits, target.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(target.size(0))
            total_examples += int(target.size(0))
        train_acc = accuracy_for_model(model, train_loader, device)
        val_acc = accuracy_for_model(model, val_loader, device)
        history.append({"epoch": float(epoch), "train_loss": float(total_loss / max(total_examples, 1)), "train_accuracy": train_acc, "val_accuracy": val_acc})
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No model state was captured during training.")
    model.load_state_dict(best_state)
    train_acc = accuracy_for_model(model, train_loader, device)
    val_acc = accuracy_for_model(model, val_loader, device)

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_mattergen_micro_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "mattergen_micro_v1.pth"
    meta_path = artifact_dir / "mattergen_micro_v1_meta.json"
    summary_path = artifact_dir / "mattergen_micro_v1_summary.json"
    zip_path = output_dir / f"supermix_mattergen_micro_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    meta = {
        "labels": list(MATTERGEN_CONCEPT_LABELS),
        "display_names": dict(MATTERGEN_DISPLAY_NAMES),
        "vocab": vocab,
        "max_len": max_len,
        "max_words": max_words,
        "word_buckets": word_buckets,
        "char_embed_dim": 30,
        "conv_channels": 64,
        "word_embed_dim": 18,
        "condition_dim": 8,
        "hidden_dim": 120,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "parameter_count": parameter_count,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "seed": seed,
        "concept_specs": list(MATTERGEN_CONCEPT_SPECS),
        "paper_inspirations": list(PAPER_INSPIRATIONS),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")

    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "row_summary": {**row_summary, "train_rows": len(train_rows), "val_rows": len(val_rows)},
        "history": history,
        "sample_predictions": sample_predictions(model, val_rows, vocab, max_len=max_len, word_buckets=word_buckets, max_words=max_words, device=device, count=6),
        "paper_inspirations": list(PAPER_INSPIRATIONS),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    source_files = [weights_path, meta_path, summary_path, SOURCE_DIR / "mattergen_generation_model.py", Path(__file__).resolve()]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in source_files:
            archive.write(file_path, arcname=file_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {"artifact_dir": str(artifact_dir), "zip_path": str(zip_path), "desktop_zip_path": str(desktop_zip_path), "parameter_count": parameter_count, "train_accuracy": round(train_acc, 4), "val_accuracy": round(val_acc, 4), "row_summary": summary["row_summary"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the small MatterGen-inspired materials generator.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--learning_rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=61)
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
