"""Benchmark v51 prediction-stability exits under request-sized inference.

This intentionally disables the older latent/entropy exits so the comparison
isolates the new full-output stability verifier against a fixed-cycle baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import torch

from benchmark_cognitive_leap_ultra_v51 import make_chained_task
from device_utils import configure_torch_runtime, resolve_device
from model_variants import ChampionNetCognitiveLeapUltraExpert


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "output" / "benchmark_v51_cognitive_leap_ultra_latest"
DEFAULT_WEIGHTS = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "prediction_stability_results.json"


def _load_model(weights: Path, device: torch.device) -> ChampionNetCognitiveLeapUltraExpert:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model = ChampionNetCognitiveLeapUltraExpert().to(device).eval()
    model.load_state_dict(state)
    return model


def _latency_summary(values: List[float]) -> Dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "total_ms": round(float(tensor.sum().item()), 6),
        "mean_ms": round(float(tensor.mean().item()), 3),
        "p50_ms": round(float(torch.quantile(tensor, 0.50).item()), 3),
        "p95_ms": round(float(torch.quantile(tensor, 0.95).item()), 3),
    }


def _scalar_summary(values: List[float]) -> Dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": round(float(tensor.mean().item()), 8),
        "p50": round(float(torch.quantile(tensor, 0.50).item()), 8),
        "p95": round(float(torch.quantile(tensor, 0.95).item()), 8),
        "max": round(float(tensor.max().item()), 8),
    }


def _synchronize_for_timing(device: torch.device) -> None:
    """Finish queued accelerator work so request latency is measured honestly."""

    device_type = str(device).split(":", 1)[0].lower()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()  # type: ignore[attr-defined]
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


@torch.no_grad()
def benchmark_serving_requests(
    model: ChampionNetCognitiveLeapUltraExpert,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    fixed_cycles: int,
    max_cycles: int,
    stability_patience: int,
    stability_tol: float,
    distribution_top_k: int,
    order_offset: int = 0,
) -> Dict[str, Any]:
    head = model.layers[10]
    fixed_predictions: List[int] = []
    adaptive_predictions: List[int] = []
    fixed_latencies: List[float] = []
    adaptive_latencies: List[float] = []
    adaptive_cycles: List[float] = []
    adaptive_latest_js_drift: List[float] = []
    adaptive_max_js_drift: List[float] = []
    exit_reasons: Counter[str] = Counter()
    measurement_orders: Counter[str] = Counter()

    # Keep initialization and one-time kernel setup outside measured requests.
    # End the warmup on the opposite mode from the first measured request so
    # both seed offsets receive the same cross-mode transition.
    def warm_adaptive() -> None:
        model(
            x[:1],
            reasoning_cycles=max_cycles,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=stability_patience,
            prediction_stability_tol=stability_tol,
            prediction_stability_top_k=distribution_top_k,
        )

    if int(order_offset) % 2 == 0:
        model(x[:1], reasoning_cycles=fixed_cycles)
        warm_adaptive()
    else:
        warm_adaptive()
        model(x[:1], reasoning_cycles=fixed_cycles)

    def run_fixed(sample: torch.Tensor) -> torch.Tensor:
        _synchronize_for_timing(sample.device)
        started = time.perf_counter()
        output = model(sample, reasoning_cycles=fixed_cycles)
        _synchronize_for_timing(sample.device)
        fixed_latencies.append((time.perf_counter() - started) * 1000.0)
        return output

    def run_adaptive(sample: torch.Tensor) -> torch.Tensor:
        _synchronize_for_timing(sample.device)
        started = time.perf_counter()
        output = model(
            sample,
            reasoning_cycles=max_cycles,
            adaptive_compute=True,
            exit_tol=0.0,
            exit_entropy_threshold=0.0,
            prediction_stability_patience=stability_patience,
            prediction_stability_tol=stability_tol,
            prediction_stability_top_k=distribution_top_k,
        )
        _synchronize_for_timing(sample.device)
        adaptive_latencies.append((time.perf_counter() - started) * 1000.0)
        adaptive_cycles.append(float(head.last_cycles_used.item()))
        adaptive_latest_js_drift.append(
            float(head.last_prediction_topk_js_divergence.item())
        )
        adaptive_max_js_drift.append(
            float(head.last_prediction_topk_js_divergence_max.item())
        )
        exit_reasons[str(head.last_exit_reason)] += 1
        return output

    for index in range(len(y)):
        sample = x[index : index + 1]
        fixed_first = (index + int(order_offset)) % 2 == 0
        if fixed_first:
            measurement_orders["fixed_then_adaptive"] += 1
            fixed_output = run_fixed(sample)
            adaptive_output = run_adaptive(sample)
        else:
            measurement_orders["adaptive_then_fixed"] += 1
            adaptive_output = run_adaptive(sample)
            fixed_output = run_fixed(sample)

        fixed_predictions.append(int(fixed_output.argmax(dim=-1).item()))
        adaptive_predictions.append(int(adaptive_output.argmax(dim=-1).item()))

    truth = y.detach().cpu()
    fixed_tensor = torch.tensor(fixed_predictions)
    adaptive_tensor = torch.tensor(adaptive_predictions)
    fixed_correct = int((fixed_tensor == truth).sum().item())
    adaptive_correct = int((adaptive_tensor == truth).sum().item())
    disagreement_count = int((fixed_tensor != adaptive_tensor).sum().item())
    request_count = int(len(y))
    fixed_accuracy = fixed_correct / max(1, request_count)
    adaptive_accuracy = adaptive_correct / max(1, request_count)
    agreement = 1.0 - (disagreement_count / max(1, request_count))
    fixed_latency = _latency_summary(fixed_latencies)
    adaptive_latency = _latency_summary(adaptive_latencies)
    mean_cycles = sum(adaptive_cycles) / max(1, len(adaptive_cycles))

    return {
        "fixed": {
            "cycles": int(fixed_cycles),
            "accuracy": round(fixed_accuracy, 6),
            "correct_predictions": fixed_correct,
            "latency": fixed_latency,
        },
        "prediction_stability": {
            "max_cycles": int(max_cycles),
            "patience": int(stability_patience),
            "confidence_tolerance": float(stability_tol),
            "accuracy": round(adaptive_accuracy, 6),
            "correct_predictions": adaptive_correct,
            "mean_cycles_used": round(mean_cycles, 4),
            "total_cycles_used": round(sum(adaptive_cycles), 6),
            "cycle_counts": {
                str(cycle): count
                for cycle, count in sorted(Counter(adaptive_cycles).items())
            },
            "exit_reasons": dict(sorted(exit_reasons.items())),
            "distribution_drift": {
                "metric": "top_k_jensen_shannon_divergence_nats",
                "top_k": int(distribution_top_k),
                "role": "shadow_diagnostic_only",
                "latest_cycle_pair": _scalar_summary(adaptive_latest_js_drift),
                "max_cycle_pair": _scalar_summary(adaptive_max_js_drift),
            },
            "latency": adaptive_latency,
        },
        "comparison": {
            "accuracy_delta": round(adaptive_accuracy - fixed_accuracy, 6),
            "prediction_agreement": round(agreement, 6),
            "exact_disagreement_count": disagreement_count,
            "request_count": request_count,
            "measurement_order": {
                "offset": int(order_offset),
                "fixed_then_adaptive": measurement_orders["fixed_then_adaptive"],
                "adaptive_then_fixed": measurement_orders["adaptive_then_fixed"],
            },
            "cycle_reduction_percent": round(
                100.0 * (float(fixed_cycles) - mean_cycles) / max(1.0, float(fixed_cycles)),
                3,
            ),
            "mean_latency_reduction_percent": round(
                100.0
                * (fixed_latency["mean_ms"] - adaptive_latency["mean_ms"])
                / max(1e-9, fixed_latency["mean_ms"]),
                3,
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--fixed_cycles", type=int, default=3)
    parser.add_argument("--max_cycles", type=int, default=8)
    parser.add_argument("--stability_patience", type=int, default=2)
    parser.add_argument("--stability_tol", type=float, default=0.005)
    parser.add_argument("--distribution_top_k", type=int, default=5)
    parser.add_argument("--order_offset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_num_threads", type=int, default=0)
    args = parser.parse_args()

    configure_torch_runtime(torch_num_threads=int(args.torch_num_threads))
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing v51 checkpoint: {weights}")

    sample_count = max(1, int(args.samples))
    x, y = make_chained_task(sample_count, seed=int(args.seed))
    x = x.to(device)
    y = y.to(device)
    model = _load_model(weights, device)
    result = benchmark_serving_requests(
        model,
        x,
        y,
        fixed_cycles=max(1, int(args.fixed_cycles)),
        max_cycles=max(1, int(args.max_cycles)),
        stability_patience=max(0, int(args.stability_patience)),
        stability_tol=max(0.0, float(args.stability_tol)),
        distribution_top_k=max(1, int(args.distribution_top_k)),
        order_offset=int(args.order_offset),
    )
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "device": str(device_info.get("resolved", device)),
        "samples": sample_count,
        "seed": int(args.seed),
        **result,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
