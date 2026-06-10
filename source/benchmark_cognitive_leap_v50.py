"""Benchmark: v50 CognitiveLeapExpert vs v22 CognitiveSingularityExpert.

Controlled experiment on a synthetic chained-modular-arithmetic task where
the label is the result of sequentially applying K operations to a start
digit (mod 10). Sequential composition is the kind of computation recursive
latent reasoning is supposed to buy, so it is a fair probe for whether the
v50 head's recursion earns its keep against the v22 head's much larger
hypernetwork/tree-search machinery.

Both models get identical data, optimizer, batch size, and step budget.
Reports parameters, training time, test accuracy, inference latency, and
v50 test-time compute scaling (1/3/8 cycles + adaptive early-exit).

Usage:
    python source/benchmark_cognitive_leap_v50.py --epochs 6 --train_size 4000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from model_variants import ChampionNetCognitiveExpert, ChampionNetCognitiveLeapExpert


N_CLASSES = 10
N_OPS = 3
IN_DIM = 128


def make_chained_task(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs encode a start digit and K chained ops; label is the result mod 10."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.zeros(n, IN_DIM)
    y = torch.zeros(n, dtype=torch.long)
    starts = torch.randint(0, 10, (n,), generator=gen)
    op_types = torch.randint(0, 2, (n, N_OPS), generator=gen)  # 0=add, 1=mul
    operands = torch.randint(1, 10, (n, N_OPS), generator=gen)
    for i in range(n):
        x[i, starts[i]] = 1.0
        acc = int(starts[i])
        for k in range(N_OPS):
            base = 10 + k * 22
            x[i, base + int(op_types[i, k])] = 1.0
            x[i, base + 2 + int(operands[i, k])] = 1.0
            if int(op_types[i, k]) == 0:
                acc = (acc + int(operands[i, k])) % 10
            else:
                acc = (acc * int(operands[i, k])) % 10
        y[i] = acc
    x = x + 0.01 * torch.randn(n, IN_DIM, generator=gen)
    return x.unsqueeze(1), y  # (n, 1, 128), (n,)


def force_alpha(model: nn.Module, value: float = 1.0) -> None:
    for name, param in model.named_parameters():
        if name == "layers.10.alpha":
            param.data.fill_(value)


def train_model(model, x_train, y_train, epochs, batch_size, lr, aux_fn):
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = x_train.shape[0]
    start = time.perf_counter()
    last_loss = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = x_train[idx], y_train[idx]
            logits = model(xb).squeeze(1)
            loss = F.cross_entropy(logits, yb) + aux_fn(model, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            last_loss = float(loss.item())
    return time.perf_counter() - start, last_loss


@torch.no_grad()
def accuracy(model, x, y, **fwd_kwargs) -> float:
    model.eval()
    logits = model(x, **fwd_kwargs) if fwd_kwargs else model(x)
    pred = logits.squeeze(1).argmax(dim=-1)
    return float((pred == y).float().mean().item())


@torch.no_grad()
def latency_ms(model, x, repeats: int = 5, **fwd_kwargs) -> float:
    model.eval()
    model(x, **fwd_kwargs) if fwd_kwargs else model(x)  # warmup
    start = time.perf_counter()
    for _ in range(repeats):
        model(x, **fwd_kwargs) if fwd_kwargs else model(x)
    return (time.perf_counter() - start) / repeats * 1000.0


def v22_aux(model, yb):
    head = model.layers[10]
    return 0.1 * head.last_ortho_loss


def v50_aux(model, yb):
    head = model.layers[10]
    return (
        0.1 * head.deep_supervision_loss(yb)
        + 0.05 * head.last_consistency_loss
        + 0.01 * head.last_ponder_cost
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--train_size", type=int, default=4000)
    ap.add_argument("--test_size", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        default=str(SOURCE_DIR.parent / "output" / "benchmark_v50_cognitive_leap" / "benchmark_results.json"),
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    x_train, y_train = make_chained_task(args.train_size, seed=args.seed)
    x_test, y_test = make_chained_task(args.test_size, seed=args.seed + 1)
    print(f"Task: chained modular arithmetic ({N_OPS} ops, mod {N_CLASSES})")
    print(f"Train {args.train_size} / test {args.test_size}, {args.epochs} epochs, batch {args.batch_size}\n")

    results = {"task": "chained_modular_arithmetic", "epochs": args.epochs,
               "train_size": args.train_size, "test_size": args.test_size,
               "seed": args.seed, "models": {}}

    # --- v22 Cognitive Singularity Expert ---
    torch.manual_seed(args.seed)
    v22 = ChampionNetCognitiveExpert()
    force_alpha(v22)
    params22 = sum(p.numel() for p in v22.parameters())
    t22, loss22 = train_model(v22, x_train, y_train, args.epochs, args.batch_size, args.lr, v22_aux)
    acc22 = accuracy(v22, x_test, y_test)
    lat22 = latency_ms(v22, x_test[:256])
    results["models"]["v22_cognitive_expert"] = {
        "params": params22, "train_seconds": round(t22, 2), "final_loss": round(loss22, 4),
        "test_accuracy": round(acc22, 4), "latency_ms_b256": round(lat22, 2),
    }
    print(f"v22 cognitive_expert      params={params22:>10,}  train={t22:6.1f}s  acc={acc22:.4f}  lat={lat22:7.2f}ms")

    # --- v50 Cognitive Leap Expert ---
    torch.manual_seed(args.seed)
    v50 = ChampionNetCognitiveLeapExpert()
    force_alpha(v50)
    params50 = sum(p.numel() for p in v50.parameters())
    t50, loss50 = train_model(v50, x_train, y_train, args.epochs, args.batch_size, args.lr, v50_aux)
    acc50 = accuracy(v50, x_test, y_test)
    head50 = v50.layers[10]
    scaling = {}
    for cycles in (1, 3, 8):
        scaling[str(cycles)] = {
            "test_accuracy": round(accuracy(v50, x_test, y_test, reasoning_cycles=cycles), 4),
            "latency_ms_b256": round(latency_ms(v50, x_test[:256], reasoning_cycles=cycles), 2),
        }
    acc_adapt = accuracy(v50, x_test, y_test, reasoning_cycles=8, adaptive_compute=True, exit_tol=1e-4)
    lat_adapt = latency_ms(v50, x_test[:256], reasoning_cycles=8, adaptive_compute=True, exit_tol=1e-4)
    cycles_used = float(head50.last_cycles_used.item())
    results["models"]["v50_cognitive_leap_expert"] = {
        "params": params50, "train_seconds": round(t50, 2), "final_loss": round(loss50, 4),
        "test_accuracy": round(acc50, 4),
        "test_time_scaling": scaling,
        "adaptive_compute": {
            "test_accuracy": round(acc_adapt, 4),
            "latency_ms_b256": round(lat_adapt, 2),
            "cycles_used_of_8": cycles_used,
        },
    }
    print(f"v50 cognitive_leap_expert params={params50:>10,}  train={t50:6.1f}s  acc={acc50:.4f}")
    for cycles, m in scaling.items():
        print(f"    cycles={cycles}: acc={m['test_accuracy']:.4f}  lat={m['latency_ms_b256']:7.2f}ms")
    print(f"    adaptive(8, tol=1e-4): acc={acc_adapt:.4f}  lat={lat_adapt:7.2f}ms  cycles_used={cycles_used:.0f}")

    results["summary"] = {
        "param_ratio_v22_over_v50": round(params22 / params50, 2),
        "accuracy_delta_v50_minus_v22": round(acc50 - acc22, 4),
        "train_speedup_v50": round(t22 / t50, 2) if t50 > 0 else None,
    }
    print(f"\nSummary: v50 uses {results['summary']['param_ratio_v22_over_v50']}x fewer params, "
          f"acc delta {results['summary']['accuracy_delta_v50_minus_v22']:+.4f}, "
          f"train speedup {results['summary']['train_speedup_v50']}x")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
