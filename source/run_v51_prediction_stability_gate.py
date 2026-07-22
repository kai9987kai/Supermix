"""Run the release gate for v51 full-output prediction stability.

The default evaluation is 8 native seeds x 512 requests.  Each seed alternates
the fixed/adaptive measurement order, and the starting order alternates across
seeds, to reduce request-order latency bias without pooling the two timings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import torch

from benchmark_cognitive_leap_ultra_v51 import make_chained_task
from benchmark_v51_prediction_stability import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_WEIGHTS,
    _load_model,
    benchmark_serving_requests,
)
from device_utils import configure_torch_runtime, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "prediction_stability_gate.json"
DEFAULT_SEEDS = (641, 643, 647, 653, 659, 661, 673, 677)
DEFAULT_SAMPLES_PER_SEED = 512
GATE_SCHEMA_VERSION = 1
MINIMUM_WEIGHTED_CYCLE_REDUCTION_PERCENT = 20.0
MINIMUM_MEDIAN_LATENCY_REDUCTION_PERCENT = 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def collect_provenance(
    *,
    device: Any,
    device_info: Mapping[str, Any],
) -> Dict[str, Any]:
    """Collect enough immutable context to reproduce or reject a gate run."""

    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "source" / "benchmark_v51_prediction_stability.py",
        PROJECT_ROOT / "source" / "benchmark_cognitive_leap_ultra_v51.py",
        PROJECT_ROOT / "source" / "model_variants.py",
    )
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    worktree_status = _git_text("status", "--porcelain", "--untracked-files=all")
    resolved_device = str(device_info.get("resolved", device))
    device_metadata: Dict[str, Any] = {
        "requested": str(device_info.get("requested", "unknown")),
        "resolved": resolved_device,
        "torch_device": str(device),
    }
    if resolved_device.startswith("cuda") and torch.cuda.is_available():
        device_metadata["name"] = torch.cuda.get_device_name(device)
        device_metadata["capability"] = list(torch.cuda.get_device_capability(device))

    return {
        "git": {
            "commit": _git_text("rev-parse", "HEAD"),
            "branch": _git_text("rev-parse", "--abbrev-ref", "HEAD"),
            "worktree_dirty": bool(worktree_status),
        },
        "source_sha256": source_hashes,
        "runtime": {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "torch": {
                "version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
                "default_dtype": str(torch.get_default_dtype()),
                "deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
            },
            "device": device_metadata,
            "threads": {
                "torch_num_threads": int(torch.get_num_threads()),
                "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                "logical_cpu_count": int(os.cpu_count() or 1),
            },
        },
    }


def _finite_float(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def aggregate_gate_results(
    seed_results: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate exact counts and evaluate every release-gate invariant."""

    if not seed_results:
        raise ValueError("At least one seed result is required")
    minimum_cycle_reduction = _finite_float(
        MINIMUM_WEIGHTED_CYCLE_REDUCTION_PERCENT,
        label="minimum cycle reduction percent",
    )
    minimum_median_latency_reduction = _finite_float(
        MINIMUM_MEDIAN_LATENCY_REDUCTION_PERCENT,
        label="minimum median latency reduction percent",
    )

    per_seed: List[Dict[str, Any]] = []
    total_requests = 0
    total_disagreements = 0
    total_fixed_cycles = 0.0
    total_adaptive_cycles = 0.0
    latency_reductions: List[float] = []
    total_fixed_latency_ms = 0.0
    total_adaptive_latency_ms = 0.0
    seen_seeds: set[int] = set()

    for row in seed_results:
        seed = int(row["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed result: {seed}")
        seen_seeds.add(seed)
        metrics = row["metrics"]
        comparison = metrics["comparison"]
        fixed = metrics["fixed"]
        adaptive = metrics["prediction_stability"]
        request_count = int(comparison["request_count"])
        if request_count <= 0:
            raise ValueError(f"Seed {seed} has no requests")

        fixed_correct = int(fixed["correct_predictions"])
        adaptive_correct = int(adaptive["correct_predictions"])
        disagreement_count = int(comparison["exact_disagreement_count"])
        fixed_cycles = _finite_float(fixed["cycles"], label="fixed cycles")
        adaptive_cycles = _finite_float(
            adaptive["total_cycles_used"], label="adaptive cycles"
        )
        if not 0 <= fixed_correct <= request_count:
            raise ValueError(f"Seed {seed} fixed correct count is invalid")
        if not 0 <= adaptive_correct <= request_count:
            raise ValueError(f"Seed {seed} adaptive correct count is invalid")
        if not 0 <= disagreement_count <= request_count:
            raise ValueError(f"Seed {seed} disagreement count is invalid")
        if fixed_cycles <= 0.0 or adaptive_cycles < 0.0:
            raise ValueError(f"Seed {seed} cycle totals are invalid")
        order = comparison["measurement_order"]
        fixed_first = int(order["fixed_then_adaptive"])
        adaptive_first = int(order["adaptive_then_fixed"])
        offset = int(order["offset"])
        expected_fixed_first = (
            (request_count + 1) // 2 if offset % 2 == 0 else request_count // 2
        )
        if (
            fixed_first != expected_fixed_first
            or adaptive_first != request_count - expected_fixed_first
        ):
            raise ValueError(f"Seed {seed} measurement order is not counterbalanced")
        fixed_mean_latency = _finite_float(
            fixed["latency"]["mean_ms"], label="fixed latency"
        )
        adaptive_mean_latency = _finite_float(
            adaptive["latency"]["mean_ms"], label="adaptive latency"
        )
        fixed_total_latency = _finite_float(
            fixed["latency"].get("total_ms", fixed_mean_latency * request_count),
            label="fixed total latency",
        )
        adaptive_total_latency = _finite_float(
            adaptive["latency"].get(
                "total_ms", adaptive_mean_latency * request_count
            ),
            label="adaptive total latency",
        )
        if fixed_total_latency <= 0.0:
            raise ValueError(f"Seed {seed} fixed latency must be positive")
        if adaptive_total_latency < 0.0:
            raise ValueError(f"Seed {seed} adaptive latency cannot be negative")
        latency_reduction = (
            100.0
            * (fixed_total_latency - adaptive_total_latency)
            / fixed_total_latency
        )
        seed_fixed_cycle_budget = fixed_cycles * request_count
        cycle_reduction = (
            100.0
            * (seed_fixed_cycle_budget - adaptive_cycles)
            / seed_fixed_cycle_budget
        )
        accuracy_delta = (adaptive_correct - fixed_correct) / request_count

        total_requests += request_count
        total_disagreements += disagreement_count
        total_fixed_cycles += fixed_cycles * request_count
        total_adaptive_cycles += adaptive_cycles
        total_fixed_latency_ms += fixed_total_latency
        total_adaptive_latency_ms += adaptive_total_latency
        latency_reductions.append(latency_reduction)
        per_seed.append(
            {
                "seed": seed,
                "samples": request_count,
                "fixed_correct": fixed_correct,
                "adaptive_correct": adaptive_correct,
                "accuracy_delta": round(accuracy_delta, 12),
                "exact_disagreement_count": disagreement_count,
                "cycle_reduction_percent": round(cycle_reduction, 6),
                "fixed_mean_latency_ms": round(
                    fixed_total_latency / request_count, 6
                ),
                "adaptive_mean_latency_ms": round(
                    adaptive_total_latency / request_count, 6
                ),
                "mean_latency_reduction_percent": round(
                    latency_reduction, 6
                ),
                "measurement_order": comparison["measurement_order"],
            }
        )

    if total_fixed_cycles <= 0.0:
        raise ValueError("Total fixed-cycle budget must be positive")
    if total_fixed_latency_ms <= 0.0:
        raise ValueError("Total fixed latency must be positive")

    weighted_cycle_reduction = (
        100.0 * (total_fixed_cycles - total_adaptive_cycles) / total_fixed_cycles
    )
    weighted_latency_reduction = (
        100.0
        * (total_fixed_latency_ms - total_adaptive_latency_ms)
        / total_fixed_latency_ms
    )
    weighted_fixed_mean_latency = total_fixed_latency_ms / total_requests
    weighted_adaptive_mean_latency = total_adaptive_latency_ms / total_requests
    median_seed_latency_reduction = float(statistics.median(latency_reductions))
    negative_accuracy_seeds = [
        row["seed"] for row in per_seed if row["adaptive_correct"] < row["fixed_correct"]
    ]

    checks = {
        "zero_prediction_disagreements": {
            "passed": total_disagreements == 0,
            "observed": total_disagreements,
            "required": 0,
        },
        "no_negative_per_seed_accuracy_delta": {
            "passed": not negative_accuracy_seeds,
            "negative_seed_ids": negative_accuracy_seeds,
            "required": "adaptive_correct >= fixed_correct for every seed",
        },
        "minimum_weighted_cycle_reduction": {
            "passed": weighted_cycle_reduction
            >= minimum_cycle_reduction,
            "observed_percent": round(weighted_cycle_reduction, 6),
            "required_minimum_percent": minimum_cycle_reduction,
        },
        "positive_median_per_seed_latency_reduction": {
            "passed": median_seed_latency_reduction
            > minimum_median_latency_reduction,
            "observed_percent": round(median_seed_latency_reduction, 6),
            "required_greater_than_percent": float(
                minimum_median_latency_reduction
            ),
        },
    }

    return {
        "summary": {
            "seed_count": len(seed_results),
            "total_samples": total_requests,
            "exact_disagreement_count": total_disagreements,
            "weighted_cycle_reduction_percent": round(
                weighted_cycle_reduction, 6
            ),
            "median_per_seed_latency_reduction_percent": round(
                median_seed_latency_reduction, 6
            ),
            "weighted_mean_latency_reduction_percent": round(
                weighted_latency_reduction, 6
            ),
            "weighted_fixed_mean_latency_ms": round(
                weighted_fixed_mean_latency, 6
            ),
            "weighted_adaptive_mean_latency_ms": round(
                weighted_adaptive_mean_latency, 6
            ),
            "total_fixed_cycle_budget": round(total_fixed_cycles, 6),
            "total_adaptive_cycles_used": round(total_adaptive_cycles, 6),
            "per_seed_accuracy_deltas": [
                {"seed": row["seed"], "accuracy_delta": row["accuracy_delta"]}
                for row in per_seed
            ],
        },
        "per_seed_gate_metrics": per_seed,
        "gates": {
            "passed": all(bool(check["passed"]) for check in checks.values()),
            "checks": checks,
        },
    }


def run_gate(
    *,
    weights: Path,
    seeds: Sequence[int],
    samples_per_seed: int,
    device: Any,
    device_info: Mapping[str, Any],
    fixed_cycles: int = 3,
    max_cycles: int = 8,
    stability_patience: int = 2,
    stability_tol: float = 0.005,
    distribution_top_k: int = 5,
    model_loader: Callable[[Path, Any], Any] | None = None,
    task_factory: Callable[..., Any] | None = None,
    benchmark_fn: Callable[..., Dict[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load one checkpoint, run every native seed, and evaluate the gate."""

    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing v51 checkpoint: {weights}")
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("At least one seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Seeds must be unique")
    if int(samples_per_seed) <= 0:
        raise ValueError("samples_per_seed must be positive")
    normalized_fixed_cycles = int(fixed_cycles)
    normalized_max_cycles = int(max_cycles)
    normalized_patience = int(stability_patience)
    normalized_tolerance = _finite_float(
        stability_tol, label="stability tolerance"
    )
    normalized_top_k = int(distribution_top_k)
    if normalized_fixed_cycles <= 0 or normalized_max_cycles <= 0:
        raise ValueError("fixed_cycles and max_cycles must be positive")
    if normalized_patience < 0:
        raise ValueError("stability_patience must be nonnegative")
    if normalized_tolerance < 0.0:
        raise ValueError("stability_tol must be nonnegative")
    if normalized_top_k <= 0:
        raise ValueError("distribution_top_k must be positive")

    loader = model_loader or _load_model
    make_task = task_factory or make_chained_task
    benchmark = benchmark_fn or benchmark_serving_requests
    checkpoint_sha256 = _sha256_file(weights)
    model = loader(weights, device)
    seed_results: List[Dict[str, Any]] = []

    for seed_index, seed in enumerate(normalized_seeds):
        x, y = make_task(int(samples_per_seed), seed=seed)
        x = x.to(device)
        y = y.to(device)
        metrics = benchmark(
            model,
            x,
            y,
            fixed_cycles=normalized_fixed_cycles,
            max_cycles=normalized_max_cycles,
            stability_patience=normalized_patience,
            stability_tol=normalized_tolerance,
            distribution_top_k=normalized_top_k,
            order_offset=seed_index % 2,
        )
        seed_results.append({"seed": seed, "metrics": metrics})

    aggregate = aggregate_gate_results(seed_results)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(weights),
            "sha256": checkpoint_sha256,
        },
        "configuration": {
            "seeds": normalized_seeds,
            "samples_per_seed": int(samples_per_seed),
            "total_samples": int(samples_per_seed) * len(normalized_seeds),
            "fixed_cycles": normalized_fixed_cycles,
            "max_cycles": normalized_max_cycles,
            "stability_patience": normalized_patience,
            "stability_tolerance": normalized_tolerance,
            "distribution_top_k": normalized_top_k,
            "counterbalance": "alternate_per_request_with_alternating_seed_offset",
        },
        "provenance": dict(provenance)
        if provenance is not None
        else collect_provenance(device=device, device_info=device_info),
        **aggregate,
        "seed_results": seed_results,
    }


def _strict_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v51 multi-seed prediction-stability release gate."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--samples-per-seed",
        "--samples_per_seed",
        dest="samples_per_seed",
        type=int,
        default=DEFAULT_SAMPLES_PER_SEED,
    )
    parser.add_argument("--fixed-cycles", "--fixed_cycles", type=int, default=3)
    parser.add_argument("--max-cycles", "--max_cycles", type=int, default=8)
    parser.add_argument(
        "--stability-patience", "--stability_patience", type=int, default=2
    )
    parser.add_argument(
        "--stability-tol", "--stability_tol", type=float, default=0.005
    )
    parser.add_argument(
        "--distribution-top-k", "--distribution_top_k", type=int, default=5
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device-preference",
        "--device_preference",
        default="cuda,npu,xpu,dml,mps,cpu",
    )
    parser.add_argument(
        "--torch-num-threads", "--torch_num_threads", type=int, default=0
    )
    parser.add_argument(
        "--torch-interop-threads", "--torch_interop_threads", type=int, default=0
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Ask torch to use deterministic algorithms where available.",
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Exit with status 2 when any release gate fails.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        strict_determinism=bool(args.strict_determinism),
    )
    device, device_info = resolve_device(
        args.device, preference=args.device_preference
    )
    payload = run_gate(
        weights=Path(args.weights),
        seeds=args.seeds,
        samples_per_seed=int(args.samples_per_seed),
        device=device,
        device_info=device_info,
        fixed_cycles=int(args.fixed_cycles),
        max_cycles=int(args.max_cycles),
        stability_patience=int(args.stability_patience),
        stability_tol=float(args.stability_tol),
        distribution_top_k=int(args.distribution_top_k),
    )
    encoded = _strict_json(payload)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded + "\n", encoding="utf-8")
    sys.stdout.write(encoded + "\n")
    if bool(args.enforce_gates) and not bool(payload["gates"]["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
