"""Train v51 CognitiveLeapUltraExpert on a bounded synthetic reasoning task.

This is a controlled, local training run for the new ultra recursive head. It
does not require a base chat checkpoint; it trains from scratch on a chained
modular-arithmetic task and writes a reusable checkpoint plus metrics.

Usage:
    python source/benchmark_cognitive_leap_ultra_v51.py --epochs 4 --train_size 3000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from device_utils import resolve_device
from model_variants import ChampionNetCognitiveLeapUltraExpert


N_CLASSES = 10
N_OPS = 4
IN_DIM = 128
# Calibrated for this v51 checkpoint/workload, not a universal safety threshold.
DEFAULT_PREDICTION_STABILITY_MARGIN = 5e-4
DEFAULT_PREDICTION_STABILITY_RANK_DEPTH = 3


def make_chained_task(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, IN_DIM)
    y = torch.zeros(n, dtype=torch.long)
    starts = torch.randint(0, 10, (n,), generator=gen)
    op_types = torch.randint(0, 3, (n, N_OPS), generator=gen)  # add/mul/sub
    operands = torch.randint(1, 10, (n, N_OPS), generator=gen)
    for i in range(n):
        x[i, starts[i]] = 1.0
        acc = int(starts[i])
        for k in range(N_OPS):
            base = 10 + k * 24
            x[i, base + int(op_types[i, k])] = 1.0
            x[i, base + 3 + int(operands[i, k])] = 1.0
            operand = int(operands[i, k])
            op = int(op_types[i, k])
            if op == 0:
                acc = (acc + operand) % 10
            elif op == 1:
                acc = (acc * operand) % 10
            else:
                acc = (acc - operand) % 10
        y[i] = acc
    x = x + 0.01 * torch.randn(n, IN_DIM, generator=gen)
    return x.unsqueeze(1), y


def force_alpha(model: nn.Module, value: float = 1.0) -> None:
    for name, param in model.named_parameters():
        if name == "layers.10.alpha":
            param.data.fill_(value)


def aux_loss(model: ChampionNetCognitiveLeapUltraExpert, targets: torch.Tensor) -> torch.Tensor:
    head = model.layers[10]
    return (
        0.08 * head.deep_supervision_loss(targets)
        + 0.04 * head.last_consistency_loss
        + 0.005 * head.last_ponder_cost
        - 0.002 * head.last_gating_entropy
    )


def train_model(
    model: ChampionNetCognitiveLeapUltraExpert,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n = x_train.shape[0]
    history = []
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        perm = torch.randperm(n)
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb).squeeze(1)
            loss = F.cross_entropy(logits, yb) + aux_loss(model, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            batch = int(yb.shape[0])
            total_loss += float(loss.item()) * batch
            total_correct += int((logits.argmax(dim=-1) == yb).sum().item())
            total_seen += batch
        row = {
            "epoch": epoch,
            "train_loss": round(total_loss / max(1, total_seen), 4),
            "train_accuracy": round(total_correct / max(1, total_seen), 4),
            "seconds": round(time.perf_counter() - epoch_start, 2),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.4f} "
            f"time={row['seconds']:.2f}s"
        )
    return {"history": history, "train_seconds": round(time.perf_counter() - start, 2)}


@torch.no_grad()
def evaluate(
    model: ChampionNetCognitiveLeapUltraExpert,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    **forward_kwargs: Any,
) -> Dict[str, Any]:
    model.eval()
    x = x_test.to(device)
    y = y_test.to(device)
    start = time.perf_counter()
    logits = model(x, **forward_kwargs).squeeze(1)
    latency_ms = (time.perf_counter() - start) * 1000.0
    loss = F.cross_entropy(logits, y)
    acc = (logits.argmax(dim=-1) == y).float().mean()
    head = model.layers[10]
    observed_rank_depth = int(head.last_prediction_rank_depth.item())
    prediction_verifier_active = bool(forward_kwargs.get("adaptive_compute", False)) and (
        int(forward_kwargs.get("prediction_stability_patience", 2)) > 0
        and bool(head.last_prediction_class_selection_valid.item())
        and observed_rank_depth > 0
    )
    return {
        "loss": round(float(loss.item()), 4),
        "accuracy": round(float(acc.item()), 4),
        "latency_ms": round(latency_ms, 2),
        "cycles_used": round(float(head.last_cycles_used.item()), 2),
        "gating_entropy": round(float(head.last_gating_entropy.item()), 4),
        "prediction_verifier_active": prediction_verifier_active,
        "prediction_margin": (
            round(float(head.last_prediction_margin.item()), 8)
            if prediction_verifier_active
            else None
        ),
        "prediction_decision_margin": (
            round(float(head.last_prediction_decision_margin.item()), 8)
            if prediction_verifier_active
            else None
        ),
        "prediction_rank_depth": (
            observed_rank_depth
            if prediction_verifier_active
            else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--train_size", type=int, default=3000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    ap.add_argument(
        "--output_dir",
        default=str(SOURCE_DIR.parent / "output" / "benchmark_v51_cognitive_leap_ultra"),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    x_train, y_train = make_chained_task(args.train_size, seed=args.seed)
    x_test, y_test = make_chained_task(args.test_size, seed=args.seed + 1)

    model = ChampionNetCognitiveLeapUltraExpert()
    force_alpha(model)
    model.to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"Task: chained modular arithmetic ({N_OPS} ops, mod {N_CLASSES})")
    print(f"Train {args.train_size} / test {args.test_size}, epochs={args.epochs}, batch={args.batch_size}")
    print(f"Device: {device_info.get('resolved', device)}")
    print(f"Parameters: {params:,}")

    train = train_model(
        model,
        x_train,
        y_train,
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=float(args.lr),
        device=device,
    )

    eval_default = evaluate(model, x_test, y_test, device)
    eval_cycles = {
        str(cycles): evaluate(model, x_test, y_test, device, reasoning_cycles=cycles)
        for cycles in (1, 3, 8)
    }
    eval_adaptive = evaluate(
        model,
        x_test,
        y_test,
        device,
        reasoning_cycles=8,
        adaptive_compute=True,
        exit_tol=1e-4,
        exit_entropy_threshold=0.0,
        prediction_stability_margin=DEFAULT_PREDICTION_STABILITY_MARGIN,
        prediction_stability_rank_depth=DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "cognitive_leap_ultra_v51_trained.pth"
    metrics_path = out_dir / "benchmark_results.json"
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, checkpoint_path)

    results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "chained_modular_arithmetic",
        "model_size": "cognitive_leap_ultra_expert",
        "params": params,
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "device": str(device_info.get("resolved", device)),
        "checkpoint_path": str(checkpoint_path),
        **train,
        "eval_default": eval_default,
        "test_time_scaling": eval_cycles,
        "adaptive_compute": eval_adaptive,
        "adaptive_compute_configuration": {
            "prediction_stability_margin": DEFAULT_PREDICTION_STABILITY_MARGIN,
            "prediction_stability_rank_depth": DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
            "margin_scope": "checkpoint_calibrated_not_universal",
        },
    }
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nFinal evaluation:")
    print(f"default: acc={eval_default['accuracy']:.4f} loss={eval_default['loss']:.4f} lat={eval_default['latency_ms']:.2f}ms")
    for cycles, row in eval_cycles.items():
        print(f"cycles={cycles}: acc={row['accuracy']:.4f} loss={row['loss']:.4f} lat={row['latency_ms']:.2f}ms")
    print(
        "adaptive(8): "
        f"acc={eval_adaptive['accuracy']:.4f} loss={eval_adaptive['loss']:.4f} "
        f"cycles={eval_adaptive['cycles_used']:.0f} lat={eval_adaptive['latency_ms']:.2f}ms"
    )
    print(f"\nCheckpoint written to {checkpoint_path}")
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
