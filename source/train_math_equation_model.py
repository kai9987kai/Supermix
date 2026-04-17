from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import sympy as sp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from math_equation_model import (
    INTENT_LABELS,
    LABEL_TO_INDEX,
    TinyMathIntentNet,
    build_vocab,
    encode_text,
)


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


class PromptDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[str, str]], vocab: Dict[str, int], max_len: int) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label = self.rows[index]
        token_ids = torch.tensor(encode_text(text, self.vocab, self.max_len), dtype=torch.long)
        target = torch.tensor(LABEL_TO_INDEX[label], dtype=torch.long)
        return token_ids, target


def _rng(seed: int) -> random.Random:
    return random.Random(int(seed))


def _symbol(name: str = "x") -> sp.Symbol:
    return sp.symbols(name)


def _rand_coeff(rng: random.Random, low: int = -9, high: int = 9, *, non_zero: bool = False) -> int:
    value = 0
    while value == 0 if non_zero else False:
        value = rng.randint(low, high)
    if not non_zero:
        value = rng.randint(low, high)
    return value


def _rand_linear_expr(rng: random.Random, variable: sp.Symbol) -> sp.Expr:
    a = _rand_coeff(rng, -12, 12, non_zero=True)
    b = _rand_coeff(rng, -18, 18)
    return a * variable + b


def _rand_poly_expr(rng: random.Random, variable: sp.Symbol) -> sp.Expr:
    degree = rng.choice([2, 2, 3])
    expr = 0
    for power in range(degree + 1):
        coeff = _rand_coeff(rng, -6, 6, non_zero=(power == degree))
        expr += coeff * variable**power
    return sp.expand(expr)


def _prompt_variants(intent: str, payload: str, variable: str = "x") -> List[str]:
    if intent == "arithmetic":
        return [
            f"Calculate {payload}",
            f"Evaluate {payload}",
            f"What is {payload}?",
            f"Compute {payload}",
        ]
    if intent == "solve_equation":
        return [
            f"Solve {payload}",
            f"Solve the equation {payload}",
            f"Find {variable} when {payload}",
            f"What are the roots of {payload}?",
        ]
    if intent == "solve_system":
        return [
            f"Solve the system {payload}",
            f"System of equations: {payload}",
            f"Find the values that satisfy {payload}",
            f"Solve system: {payload}",
        ]
    if intent == "simplify":
        return [
            f"Simplify {payload}",
            f"Please simplify {payload}",
            f"Reduce {payload}",
            f"What does {payload} simplify to?",
        ]
    if intent == "factor":
        return [
            f"Factor {payload}",
            f"Factor the polynomial {payload}",
            f"Write {payload} in factored form",
            f"Can you factor {payload}?",
        ]
    if intent == "expand":
        return [
            f"Expand {payload}",
            f"Multiply out {payload}",
            f"Expand the expression {payload}",
            f"What is the expanded form of {payload}?",
        ]
    if intent == "differentiate":
        return [
            f"Differentiate {payload}",
            f"Find the derivative of {payload}",
            f"Compute d/d{variable} of {payload}",
            f"Derivative of {payload} with respect to {variable}",
        ]
    if intent == "integrate":
        return [
            f"Integrate {payload}",
            f"Find the integral of {payload}",
            f"Compute the antiderivative of {payload}",
            f"Integrate {payload} with respect to {variable}",
        ]
    raise ValueError(f"Unsupported intent: {intent}")


def build_rows(samples_per_intent: int, seed: int) -> List[Tuple[str, str]]:
    rng = _rng(seed)
    x, y = sp.symbols("x y")
    rows: List[Tuple[str, str]] = []

    for _ in range(samples_per_intent):
        left = rng.randint(-200, 200)
        right = rng.randint(-200, 200)
        arithmetic_expr = f"({left} + {right}) * {rng.randint(1, 9)} / {rng.randint(1, 9)}"
        rows.extend((prompt, "arithmetic") for prompt in _prompt_variants("arithmetic", arithmetic_expr))

        eq_left = _rand_linear_expr(rng, x)
        eq_right = _rand_linear_expr(rng, x) + rng.randint(-9, 9)
        equation = f"{sp.expand(eq_left)} = {sp.expand(eq_right)}"
        rows.extend((prompt, "solve_equation") for prompt in _prompt_variants("solve_equation", equation))

        system_eq1 = f"{rng.randint(1, 7)}*x + {rng.randint(1, 7)}*y = {rng.randint(-20, 20)}"
        system_eq2 = f"{rng.randint(1, 7)}*x - {rng.randint(1, 7)}*y = {rng.randint(-20, 20)}"
        system_payload = f"{system_eq1}; {system_eq2}"
        rows.extend((prompt, "solve_system") for prompt in _prompt_variants("solve_system", system_payload))

        expr = (x**2 - rng.randint(1, 7)) / (x - 1)
        rows.extend((prompt, "simplify") for prompt in _prompt_variants("simplify", str(expr)))

        factored = (x + rng.randint(1, 7)) * (x - rng.randint(1, 7))
        rows.extend((prompt, "factor") for prompt in _prompt_variants("factor", str(sp.expand(factored))))

        expandable = (x + rng.randint(1, 5)) * (x + rng.randint(1, 5)) * (x - rng.randint(1, 4))
        rows.extend((prompt, "expand") for prompt in _prompt_variants("expand", str(expandable)))

        derivative_expr = _rand_poly_expr(rng, x) + rng.randint(1, 4) * sp.sin(x)
        rows.extend((prompt, "differentiate") for prompt in _prompt_variants("differentiate", str(derivative_expr)))

        integral_expr = rng.randint(1, 5) * x**2 + rng.randint(1, 6) * x + rng.randint(-5, 5) + rng.randint(1, 4) * sp.cos(x)
        rows.extend((prompt, "integrate") for prompt in _prompt_variants("integrate", str(integral_expr)))

    rng.shuffle(rows)
    return rows


def split_rows(rows: Sequence[Tuple[str, str]], seed: int) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    cooked = list(rows)
    _rng(seed).shuffle(cooked)
    pivot = int(len(cooked) * 0.88)
    return cooked[:pivot], cooked[pivot:]


def accuracy_for_model(model: TinyMathIntentNet, loader: DataLoader, device: torch.device) -> float:
    total = 0
    correct = 0
    model.eval()
    with torch.inference_mode():
        for token_ids, target in loader:
            token_ids = token_ids.to(device)
            target = target.to(device)
            pred = model(token_ids).argmax(dim=1)
            total += int(target.numel())
            correct += int((pred == target).sum().item())
    return float(correct / max(total, 1))


def train_model(
    *,
    output_dir: Path,
    models_dir: Path,
    samples_per_intent: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> Dict[str, object]:
    torch.manual_seed(seed)
    rows = build_rows(samples_per_intent=samples_per_intent, seed=seed)
    train_rows, val_rows = split_rows(rows, seed=seed + 17)
    vocab = build_vocab([text for text, _label in train_rows], min_frequency=1)
    max_len = 192
    train_dataset = PromptDataset(train_rows, vocab, max_len)
    val_dataset = PromptDataset(val_rows, vocab, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = TinyMathIntentNet(vocab_size=len(vocab), num_labels=len(INTENT_LABELS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_val = -1.0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for token_ids, target in train_loader:
            token_ids = token_ids.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids)
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
    artifact_dir = output_dir / f"supermix_math_equation_micro_v1_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "math_equation_micro_v1.pth"
    meta_path = artifact_dir / "math_equation_micro_v1_meta.json"
    summary_path = artifact_dir / "math_equation_micro_v1_summary.json"
    zip_path = output_dir / f"supermix_math_equation_micro_v1_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    meta = {
        "labels": list(INTENT_LABELS),
        "vocab": vocab,
        "max_len": max_len,
        "embed_dim": 24,
        "hidden_dim": 48,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "parameter_count": parameter_count,
        "samples_per_intent": samples_per_intent,
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
        "history": history[-10:],
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
    parser = argparse.ArgumentParser(description="Train a tiny equation-intent model and package it as a Supermix zip artifact.")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--samples_per_intent", type=int, default=260)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        samples_per_intent=int(args.samples_per_intent),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
