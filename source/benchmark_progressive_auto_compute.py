"""Benchmark progressive accepted-probe auto compute against legacy v1.

The legacy controller evaluates every candidate budget, selects with the v1
confidence/entropy policy, and reruns the selected budget.  The progressive
controller applies the same policy but returns the accepted probe directly.
This benchmark requires exact selected-budget, prediction, and output-tensor
agreement while measuring the avoided forward evaluations and counterbalanced
wall-clock latency on deterministic held-out v51 requests.

Cross-budget top-k JSD remains shadow telemetry.  It is audited here but never
participates in selection or any output-agreement gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
import time
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

from benchmark_cognitive_leap_ultra_v51 import make_chained_task
from benchmark_v51_prediction_stability import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_WEIGHTS,
    _load_model,
)
from chat_app import (
    DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    DEFAULT_ADAPTIVE_EXIT_TOL,
    DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
    DEFAULT_AUTO_COMPUTE_ENTROPY,
    DEFAULT_PREDICTION_STABILITY_MARGIN,
    DEFAULT_PREDICTION_STABILITY_PATIENCE,
    DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    DEFAULT_PREDICTION_STABILITY_TOL,
    evaluate_runtime_compute_budgets,
    forward_with_runtime_compute,
    progressive_auto_compute_forward,
    resolve_runtime_compute_cycles,
    select_auto_runtime_compute_budget,
)
from device_utils import configure_torch_runtime, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "progressive_auto_compute_benchmark.json"
DEFAULT_SEEDS = (719, 727, 733, 739)
DEFAULT_SAMPLES_PER_SEED = 64
DEFAULT_CYCLES = (1, 3, 8)
TRAINING_AND_ORIGINAL_TEST_SEEDS = frozenset((51, 52))
BENCHMARK_SCHEMA_VERSION = "progressive-auto-compute-benchmark-v1"
MAX_MISMATCH_DETAILS = 32


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
    """Capture the controller, model, data, runtime, and repository identity."""

    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "source" / "chat_app.py",
        PROJECT_ROOT / "source" / "benchmark_cognitive_leap_ultra_v51.py",
        PROJECT_ROOT / "source" / "model_variants.py",
    )
    source_hashes = {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
        for path in source_paths
    }
    status = _git_text("status", "--porcelain", "--untracked-files=all")
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
            "worktree_dirty": bool(status),
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


def _nonnegative_finite_float(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative, got {value!r}")
    return number


def _observed_prediction_margin_summary(
    values: Sequence[float],
) -> Dict[str, Any]:
    """Summarize final-call margins; the minimum is the safety-facing value."""

    if not values:
        return {
            "metric": "top_1_minus_top_2_probability",
            "observation_count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
        }
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "metric": "top_1_minus_top_2_probability",
        "observation_count": len(values),
        "minimum": round(float(tensor.min().item()), 8),
        "mean": round(float(tensor.mean().item()), 8),
        "maximum": round(float(tensor.max().item()), 8),
    }


def _observed_prediction_decision_margin_summary(
    values: Sequence[float],
) -> Dict[str, Any]:
    """Summarize the rank-boundary margin used by the active verifier."""

    if not values:
        return {
            "metric": "minimum_adjacent_probability_gap_through_rank_depth",
            "observation_count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
        }
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "metric": "minimum_adjacent_probability_gap_through_rank_depth",
        "observation_count": len(values),
        "minimum": round(float(tensor.min().item()), 8),
        "mean": round(float(tensor.mean().item()), 8),
        "maximum": round(float(tensor.max().item()), 8),
    }


def _prediction_margin_from_compute(compute: Mapping[str, Any]) -> float | None:
    if compute.get("prediction_verifier_active") is not True:
        return None
    value = compute.get("prediction_margin")
    if value is None:
        return None
    return _nonnegative_finite_float(
        value,
        label="observed prediction stability margin",
    )


def _prediction_decision_margin_from_compute(
    compute: Mapping[str, Any],
) -> float | None:
    if compute.get("prediction_verifier_active") is not True:
        return None
    value = compute.get("prediction_decision_margin")
    if value is None:
        return None
    return _nonnegative_finite_float(
        value,
        label="observed prediction decision margin",
    )


def _synchronize_for_timing(device: torch.device) -> None:
    device_type = str(device).split(":", 1)[0].lower()
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()  # type: ignore[attr-defined]
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _latency_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("At least one latency observation is required")
    tensor = torch.tensor(list(values), dtype=torch.float64)
    return {
        "total_ms": round(float(tensor.sum().item()), 6),
        "mean_ms": round(float(tensor.mean().item()), 6),
        "p50_ms": round(float(torch.quantile(tensor, 0.50).item()), 6),
        "p95_ms": round(float(torch.quantile(tensor, 0.95).item()), 6),
    }


@torch.no_grad()
def legacy_v1_auto_compute_forward(
    model: Any,
    x: torch.Tensor,
    available_labels: List[int],
    *,
    cycles: Any = None,
    confidence_target: Any = DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    entropy_target: Any = DEFAULT_AUTO_COMPUTE_ENTROPY,
    adaptive_compute: Any = False,
    exit_tol: Any = DEFAULT_ADAPTIVE_EXIT_TOL,
    exit_entropy_threshold: Any = DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    prediction_stability_patience: Any = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    prediction_stability_tol: Any = DEFAULT_PREDICTION_STABILITY_TOL,
    prediction_stability_margin: Any = DEFAULT_PREDICTION_STABILITY_MARGIN,
    prediction_stability_rank_depth: Any = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    auto_reasoning_context: str = "",
    distribution_top_k: int = DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """Execute the retired v1 probe-all/select/rerun controller literally."""

    del distribution_top_k  # The v1 selector had no distribution-stability input.
    stability_margin = _nonnegative_finite_float(
        prediction_stability_margin,
        label="prediction stability margin",
    )
    rows = evaluate_runtime_compute_budgets(
        model,
        x,
        available_labels,
        cycles=cycles,
        adaptive_compute=bool(adaptive_compute),
        exit_tol=exit_tol,
        exit_entropy_threshold=exit_entropy_threshold,
        prediction_stability_patience=prediction_stability_patience,
        prediction_stability_tol=prediction_stability_tol,
        prediction_stability_margin=stability_margin,
        prediction_stability_rank_depth=prediction_stability_rank_depth,
    )
    selection = select_auto_runtime_compute_budget(
        rows,
        confidence_target=float(confidence_target),
        entropy_target=float(entropy_target),
    )
    selected_cycles = int(selection["selected_reasoning_cycles"])
    output, compute = forward_with_runtime_compute(
        model,
        x,
        reasoning_cycles=selected_cycles,
        adaptive_compute=adaptive_compute,
        exit_tol=exit_tol,
        exit_entropy_threshold=exit_entropy_threshold,
        prediction_stability_patience=prediction_stability_patience,
        prediction_stability_tol=prediction_stability_tol,
        prediction_stability_margin=stability_margin,
        prediction_stability_rank_depth=prediction_stability_rank_depth,
        prediction_class_indices=available_labels,
        auto_reasoning_context=auto_reasoning_context,
    )
    plan = {
        **selection,
        "schema_version": "runtime-auto-compute-plan-v1-benchmark-reference",
        "strategy": "probe_all_select_and_rerun",
        "candidate_cycles": [
            int(value) for value in resolve_runtime_compute_cycles(cycles)
        ],
        "evaluated_cycles": [int(row["requested_cycles"]) for row in rows],
        "forward_evaluations": len(rows) + 1,
        "reused_probe_output": False,
        "mutual_stability_role": "not_present_in_v1",
    }
    compute = dict(compute)
    compute["auto_compute_plan"] = plan
    compute["inference_reused"] = False
    return output, compute, plan


def _prediction(output: torch.Tensor, labels: Sequence[int]) -> int:
    logits = output[0, 0]
    indices = torch.tensor(list(labels), dtype=torch.long, device=logits.device)
    position = int(logits.index_select(0, indices).argmax(dim=0).item())
    return int(labels[position])


def _shadow_selection_is_disabled(plan: Mapping[str, Any]) -> bool:
    if plan.get("mutual_stability_role") != "shadow_diagnostic_only":
        return False
    for row in plan.get("rows", []):
        if not isinstance(row, Mapping):
            return False
        shadow = row.get("mutual_stability_shadow")
        if not isinstance(shadow, Mapping):
            return False
        if shadow.get("role") != "shadow_diagnostic_only":
            return False
        if shadow.get("selection_enabled") is not False:
            return False
    return True


def _max_abs_difference(
    left: torch.Tensor, right: torch.Tensor
) -> float | None:
    if tuple(left.shape) != tuple(right.shape):
        return None
    return float(
        (left.detach().to(dtype=torch.float64, device="cpu")
         - right.detach().to(dtype=torch.float64, device="cpu"))
        .abs()
        .max()
        .item()
    )


@torch.no_grad()
def benchmark_requests(
    model: Any,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    available_labels: Sequence[int] = tuple(range(10)),
    cycles: Any = DEFAULT_CYCLES,
    confidence_target: float = DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    entropy_target: float = DEFAULT_AUTO_COMPUTE_ENTROPY,
    adaptive_compute: bool = False,
    exit_tol: float = DEFAULT_ADAPTIVE_EXIT_TOL,
    exit_entropy_threshold: float = DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    prediction_stability_patience: int = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    prediction_stability_tol: float = DEFAULT_PREDICTION_STABILITY_TOL,
    prediction_stability_margin: float = DEFAULT_PREDICTION_STABILITY_MARGIN,
    prediction_stability_rank_depth: int = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    distribution_top_k: int = DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
    order_offset: int = 0,
    legacy_fn: Callable[..., Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]]
    | None = None,
    progressive_fn: Callable[..., Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]]
    | None = None,
) -> Dict[str, Any]:
    """Run a paired, request-level, counterbalanced controller comparison."""

    if bool(getattr(model, "training", False)):
        raise RuntimeError("controller benchmark requires model.eval()")
    request_count = int(len(y))
    if request_count <= 0 or int(len(x)) != request_count:
        raise ValueError("x and y must contain the same positive request count")
    labels = [int(label) for label in available_labels]
    if not labels:
        raise ValueError("available_labels must not be empty")
    resolved_cycles = resolve_runtime_compute_cycles(cycles)
    settings = {
        "cycles": resolved_cycles,
        "confidence_target": _finite_float(
            confidence_target, label="confidence target"
        ),
        "entropy_target": _finite_float(entropy_target, label="entropy target"),
        "adaptive_compute": bool(adaptive_compute),
        "exit_tol": _finite_float(exit_tol, label="exit tolerance"),
        "exit_entropy_threshold": _finite_float(
            exit_entropy_threshold, label="exit entropy threshold"
        ),
        "prediction_stability_patience": int(prediction_stability_patience),
        "prediction_stability_tol": _finite_float(
            prediction_stability_tol, label="prediction stability tolerance"
        ),
        "prediction_stability_margin": _nonnegative_finite_float(
            prediction_stability_margin,
            label="prediction stability margin",
        ),
        "prediction_stability_rank_depth": int(prediction_stability_rank_depth),
        "distribution_top_k": int(distribution_top_k),
    }
    if settings["prediction_stability_patience"] < 0:
        raise ValueError("prediction_stability_patience must be nonnegative")
    if settings["prediction_stability_rank_depth"] < 0:
        raise ValueError("prediction_stability_rank_depth must be nonnegative")
    if settings["distribution_top_k"] <= 0:
        raise ValueError("distribution_top_k must be positive")

    legacy = legacy_fn or legacy_v1_auto_compute_forward
    progressive = progressive_fn or progressive_auto_compute_forward
    first_sample = x[:1]

    # Match the transition immediately before each seed's first measured mode.
    if int(order_offset) % 2 == 0:
        legacy(model, first_sample, labels, **settings)
        progressive(model, first_sample, labels, **settings)
    else:
        progressive(model, first_sample, labels, **settings)
        legacy(model, first_sample, labels, **settings)

    legacy_latencies: List[float] = []
    progressive_latencies: List[float] = []
    legacy_prediction_margins: List[float] = []
    progressive_prediction_margins: List[float] = []
    legacy_prediction_decision_margins: List[float] = []
    progressive_prediction_decision_margins: List[float] = []
    measurement_orders: Counter[str] = Counter()
    selected_cycle_pairs: Counter[str] = Counter()
    selected_cycle_disagreements = 0
    prediction_disagreements = 0
    output_disagreements = 0
    legacy_correct = 0
    progressive_correct = 0
    legacy_forward_evaluations = 0
    progressive_forward_evaluations = 0
    progressive_not_better_count = 0
    shadow_selection_violations = 0
    maximum_abs_logit_difference = 0.0
    mismatch_details: List[Dict[str, Any]] = []

    def timed_call(
        fn: Callable[..., Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]],
        sample: torch.Tensor,
        values: List[float],
        observed_margins: List[float],
        observed_decision_margins: List[float],
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        _synchronize_for_timing(sample.device)
        started = time.perf_counter()
        result = fn(model, sample, labels, **settings)
        _synchronize_for_timing(sample.device)
        values.append((time.perf_counter() - started) * 1000.0)
        observed_margin = _prediction_margin_from_compute(result[1])
        if observed_margin is not None:
            observed_margins.append(observed_margin)
        observed_decision_margin = _prediction_decision_margin_from_compute(
            result[1]
        )
        if observed_decision_margin is not None:
            observed_decision_margins.append(observed_decision_margin)
        return result

    for index in range(request_count):
        sample = x[index : index + 1]
        legacy_first = (index + int(order_offset)) % 2 == 0
        if legacy_first:
            measurement_orders["legacy_then_progressive"] += 1
            legacy_result = timed_call(
                legacy,
                sample,
                legacy_latencies,
                legacy_prediction_margins,
                legacy_prediction_decision_margins,
            )
            progressive_result = timed_call(
                progressive,
                sample,
                progressive_latencies,
                progressive_prediction_margins,
                progressive_prediction_decision_margins,
            )
        else:
            measurement_orders["progressive_then_legacy"] += 1
            progressive_result = timed_call(
                progressive,
                sample,
                progressive_latencies,
                progressive_prediction_margins,
                progressive_prediction_decision_margins,
            )
            legacy_result = timed_call(
                legacy,
                sample,
                legacy_latencies,
                legacy_prediction_margins,
                legacy_prediction_decision_margins,
            )

        legacy_output, _legacy_compute, legacy_plan = legacy_result
        progressive_output, _progressive_compute, progressive_plan = (
            progressive_result
        )
        legacy_cycles = int(legacy_plan["selected_reasoning_cycles"])
        progressive_cycles = int(progressive_plan["selected_reasoning_cycles"])
        legacy_prediction = _prediction(legacy_output, labels)
        progressive_prediction = _prediction(progressive_output, labels)
        truth = int(y[index].item())
        shapes_match = tuple(legacy_output.shape) == tuple(progressive_output.shape)
        outputs_equal = bool(
            shapes_match and torch.equal(legacy_output, progressive_output)
        )
        max_abs_difference = _max_abs_difference(
            legacy_output, progressive_output
        )
        if max_abs_difference is not None:
            maximum_abs_logit_difference = max(
                maximum_abs_logit_difference, max_abs_difference
            )

        legacy_evaluations = int(legacy_plan["forward_evaluations"])
        progressive_evaluations = int(progressive_plan["forward_evaluations"])
        legacy_forward_evaluations += legacy_evaluations
        progressive_forward_evaluations += progressive_evaluations
        progressive_not_better_count += int(
            progressive_evaluations >= legacy_evaluations
        )
        shadow_ok = _shadow_selection_is_disabled(progressive_plan)
        shadow_selection_violations += int(not shadow_ok)
        selected_cycle_pairs[f"{legacy_cycles}->{progressive_cycles}"] += 1
        selected_cycle_disagreements += int(legacy_cycles != progressive_cycles)
        prediction_disagreements += int(
            legacy_prediction != progressive_prediction
        )
        output_disagreements += int(not outputs_equal)
        legacy_correct += int(legacy_prediction == truth)
        progressive_correct += int(progressive_prediction == truth)

        if (
            legacy_cycles != progressive_cycles
            or legacy_prediction != progressive_prediction
            or not outputs_equal
            or not shadow_ok
        ) and len(mismatch_details) < MAX_MISMATCH_DETAILS:
            mismatch_details.append(
                {
                    "request_index": index,
                    "truth": truth,
                    "legacy_selected_cycles": legacy_cycles,
                    "progressive_selected_cycles": progressive_cycles,
                    "legacy_prediction": legacy_prediction,
                    "progressive_prediction": progressive_prediction,
                    "output_shapes_match": shapes_match,
                    "outputs_exactly_equal": outputs_equal,
                    "max_abs_logit_difference": max_abs_difference,
                    "shadow_selection_disabled": shadow_ok,
                }
            )

    legacy_latency = _latency_summary(legacy_latencies)
    progressive_latency = _latency_summary(progressive_latencies)
    forward_reduction = (
        100.0
        * (legacy_forward_evaluations - progressive_forward_evaluations)
        / max(1, legacy_forward_evaluations)
    )
    latency_reduction = (
        100.0
        * (legacy_latency["total_ms"] - progressive_latency["total_ms"])
        / max(1e-12, legacy_latency["total_ms"])
    )
    checks = {
        "exact_selected_cycle_agreement": {
            "passed": selected_cycle_disagreements == 0,
            "observed_disagreements": selected_cycle_disagreements,
            "required": 0,
        },
        "exact_prediction_agreement": {
            "passed": prediction_disagreements == 0,
            "observed_disagreements": prediction_disagreements,
            "required": 0,
        },
        "exact_output_tensor_agreement": {
            "passed": output_disagreements == 0,
            "observed_disagreements": output_disagreements,
            "required": 0,
        },
        "positive_forward_evaluation_reduction": {
            "passed": progressive_forward_evaluations < legacy_forward_evaluations,
            "legacy_forward_evaluations": legacy_forward_evaluations,
            "progressive_forward_evaluations": progressive_forward_evaluations,
        },
        "per_request_forward_evaluation_reduction": {
            "passed": progressive_not_better_count == 0,
            "non_reducing_request_count": progressive_not_better_count,
            "required": 0,
        },
        "mutual_stability_is_shadow_only": {
            "passed": shadow_selection_violations == 0,
            "selection_policy_violation_count": shadow_selection_violations,
            "required": 0,
        },
    }
    return {
        "request_count": request_count,
        "measurement_order": {
            "offset": int(order_offset),
            "legacy_then_progressive": measurement_orders[
                "legacy_then_progressive"
            ],
            "progressive_then_legacy": measurement_orders[
                "progressive_then_legacy"
            ],
        },
        "legacy_v1": {
            "strategy": "probe_all_select_and_rerun",
            "correct_predictions": legacy_correct,
            "forward_evaluations": legacy_forward_evaluations,
            "observed_prediction_margin": _observed_prediction_margin_summary(
                legacy_prediction_margins
            ),
            "observed_prediction_decision_margin": (
                _observed_prediction_decision_margin_summary(
                    legacy_prediction_decision_margins
                )
            ),
            "latency": legacy_latency,
        },
        "progressive": {
            "strategy": "progressive_accepted_probe",
            "correct_predictions": progressive_correct,
            "forward_evaluations": progressive_forward_evaluations,
            "observed_prediction_margin": _observed_prediction_margin_summary(
                progressive_prediction_margins
            ),
            "observed_prediction_decision_margin": (
                _observed_prediction_decision_margin_summary(
                    progressive_prediction_decision_margins
                )
            ),
            "latency": progressive_latency,
        },
        "comparison": {
            "selected_cycle_pairs": dict(sorted(selected_cycle_pairs.items())),
            "exact_selected_cycle_disagreement_count": (
                selected_cycle_disagreements
            ),
            "exact_prediction_disagreement_count": prediction_disagreements,
            "exact_output_tensor_disagreement_count": output_disagreements,
            "maximum_absolute_logit_difference": round(
                maximum_abs_logit_difference, 12
            ),
            "forward_evaluation_reduction_percent": round(
                forward_reduction, 6
            ),
            "mean_latency_reduction_percent": round(latency_reduction, 6),
            "mismatch_details": mismatch_details,
        },
        "gates": {
            "passed": all(bool(check["passed"]) for check in checks.values()),
            "checks": checks,
        },
    }


def aggregate_results(seed_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not seed_results:
        raise ValueError("At least one seed result is required")
    total_requests = 0
    total_selected_cycle_disagreements = 0
    total_prediction_disagreements = 0
    total_output_disagreements = 0
    total_legacy_correct = 0
    total_progressive_correct = 0
    total_legacy_forward_evaluations = 0
    total_progressive_forward_evaluations = 0
    total_legacy_latency_ms = 0.0
    total_progressive_latency_ms = 0.0
    maximum_abs_logit_difference = 0.0
    shadow_selection_violations = 0
    non_reducing_requests = 0
    selected_cycle_pairs: Counter[str] = Counter()
    per_seed_latency_reductions: List[float] = []
    per_seed: List[Dict[str, Any]] = []
    seen_seeds: set[int] = set()
    legacy_margin_observations = 0
    progressive_margin_observations = 0
    legacy_decision_margin_observations = 0
    progressive_decision_margin_observations = 0
    minimum_legacy_prediction_margin = math.inf
    minimum_progressive_prediction_margin = math.inf
    minimum_legacy_prediction_decision_margin = math.inf
    minimum_progressive_prediction_decision_margin = math.inf

    for seed_result in seed_results:
        seed = int(seed_result["seed"])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed result: {seed}")
        seen_seeds.add(seed)
        metrics = seed_result["metrics"]
        request_count = int(metrics["request_count"])
        if request_count <= 0:
            raise ValueError(f"Seed {seed} has no requests")
        order = metrics["measurement_order"]
        legacy_first = int(order["legacy_then_progressive"])
        progressive_first = int(order["progressive_then_legacy"])
        offset = int(order["offset"])
        expected_legacy_first = (
            (request_count + 1) // 2 if offset % 2 == 0 else request_count // 2
        )
        if (
            legacy_first != expected_legacy_first
            or progressive_first != request_count - expected_legacy_first
        ):
            raise ValueError(f"Seed {seed} measurement order is not counterbalanced")

        legacy = metrics["legacy_v1"]
        progressive = metrics["progressive"]
        comparison = metrics["comparison"]
        legacy_margin = legacy.get("observed_prediction_margin", {})
        progressive_margin = progressive.get("observed_prediction_margin", {})
        legacy_decision_margin = legacy.get(
            "observed_prediction_decision_margin", {}
        )
        progressive_decision_margin = progressive.get(
            "observed_prediction_decision_margin", {}
        )
        legacy_margin_count = int(legacy_margin.get("observation_count", 0))
        progressive_margin_count = int(
            progressive_margin.get("observation_count", 0)
        )
        legacy_margin_minimum = legacy_margin.get("minimum")
        progressive_margin_minimum = progressive_margin.get("minimum")
        legacy_decision_margin_count = int(
            legacy_decision_margin.get("observation_count", 0)
        )
        progressive_decision_margin_count = int(
            progressive_decision_margin.get("observation_count", 0)
        )
        legacy_decision_margin_minimum = legacy_decision_margin.get("minimum")
        progressive_decision_margin_minimum = progressive_decision_margin.get(
            "minimum"
        )
        if legacy_margin_count < 0 or progressive_margin_count < 0:
            raise ValueError(f"Seed {seed} margin observation count is invalid")
        if legacy_margin_count:
            legacy_margin_minimum = _nonnegative_finite_float(
                legacy_margin_minimum,
                label="minimum legacy prediction stability margin",
            )
            minimum_legacy_prediction_margin = min(
                minimum_legacy_prediction_margin,
                legacy_margin_minimum,
            )
            legacy_margin_observations += legacy_margin_count
        if progressive_margin_count:
            progressive_margin_minimum = _nonnegative_finite_float(
                progressive_margin_minimum,
                label="minimum progressive prediction stability margin",
            )
            minimum_progressive_prediction_margin = min(
                minimum_progressive_prediction_margin,
                progressive_margin_minimum,
            )
            progressive_margin_observations += progressive_margin_count
        if (
            legacy_decision_margin_count < 0
            or progressive_decision_margin_count < 0
        ):
            raise ValueError(
                f"Seed {seed} decision-margin observation count is invalid"
            )
        if legacy_decision_margin_count:
            legacy_decision_margin_minimum = _nonnegative_finite_float(
                legacy_decision_margin_minimum,
                label="minimum legacy prediction decision margin",
            )
            minimum_legacy_prediction_decision_margin = min(
                minimum_legacy_prediction_decision_margin,
                legacy_decision_margin_minimum,
            )
            legacy_decision_margin_observations += legacy_decision_margin_count
        if progressive_decision_margin_count:
            progressive_decision_margin_minimum = _nonnegative_finite_float(
                progressive_decision_margin_minimum,
                label="minimum progressive prediction decision margin",
            )
            minimum_progressive_prediction_decision_margin = min(
                minimum_progressive_prediction_decision_margin,
                progressive_decision_margin_minimum,
            )
            progressive_decision_margin_observations += (
                progressive_decision_margin_count
            )
        legacy_latency = _finite_float(
            legacy["latency"]["total_ms"], label="legacy latency"
        )
        progressive_latency = _finite_float(
            progressive["latency"]["total_ms"], label="progressive latency"
        )
        if legacy_latency <= 0.0 or progressive_latency < 0.0:
            raise ValueError(f"Seed {seed} latency totals are invalid")
        seed_latency_reduction = (
            100.0 * (legacy_latency - progressive_latency) / legacy_latency
        )
        per_seed_latency_reductions.append(seed_latency_reduction)
        total_requests += request_count
        total_selected_cycle_disagreements += int(
            comparison["exact_selected_cycle_disagreement_count"]
        )
        total_prediction_disagreements += int(
            comparison["exact_prediction_disagreement_count"]
        )
        total_output_disagreements += int(
            comparison["exact_output_tensor_disagreement_count"]
        )
        total_legacy_correct += int(legacy["correct_predictions"])
        total_progressive_correct += int(progressive["correct_predictions"])
        total_legacy_forward_evaluations += int(legacy["forward_evaluations"])
        total_progressive_forward_evaluations += int(
            progressive["forward_evaluations"]
        )
        total_legacy_latency_ms += legacy_latency
        total_progressive_latency_ms += progressive_latency
        maximum_abs_logit_difference = max(
            maximum_abs_logit_difference,
            _finite_float(
                comparison["maximum_absolute_logit_difference"],
                label="maximum absolute logit difference",
            ),
        )
        for pair, count in comparison["selected_cycle_pairs"].items():
            selected_cycle_pairs[str(pair)] += int(count)
        checks = metrics["gates"]["checks"]
        shadow_selection_violations += int(
            checks["mutual_stability_is_shadow_only"][
                "selection_policy_violation_count"
            ]
        )
        non_reducing_requests += int(
            checks["per_request_forward_evaluation_reduction"][
                "non_reducing_request_count"
            ]
        )
        per_seed.append(
            {
                "seed": seed,
                "samples": request_count,
                "selected_cycle_disagreements": int(
                    comparison["exact_selected_cycle_disagreement_count"]
                ),
                "prediction_disagreements": int(
                    comparison["exact_prediction_disagreement_count"]
                ),
                "output_tensor_disagreements": int(
                    comparison["exact_output_tensor_disagreement_count"]
                ),
                "forward_evaluation_reduction_percent": float(
                    comparison["forward_evaluation_reduction_percent"]
                ),
                "mean_latency_reduction_percent": round(
                    seed_latency_reduction, 6
                ),
                "measurement_order": dict(order),
                "minimum_observed_prediction_margin": {
                    "legacy_v1": legacy_margin_minimum,
                    "progressive": progressive_margin_minimum,
                },
                "minimum_observed_prediction_decision_margin": {
                    "legacy_v1": legacy_decision_margin_minimum,
                    "progressive": progressive_decision_margin_minimum,
                },
            }
        )

    forward_reduction = (
        100.0
        * (total_legacy_forward_evaluations - total_progressive_forward_evaluations)
        / max(1, total_legacy_forward_evaluations)
    )
    weighted_latency_reduction = (
        100.0
        * (total_legacy_latency_ms - total_progressive_latency_ms)
        / max(1e-12, total_legacy_latency_ms)
    )
    checks = {
        "exact_selected_cycle_agreement": {
            "passed": total_selected_cycle_disagreements == 0,
            "observed_disagreements": total_selected_cycle_disagreements,
            "required": 0,
        },
        "exact_prediction_agreement": {
            "passed": total_prediction_disagreements == 0,
            "observed_disagreements": total_prediction_disagreements,
            "required": 0,
        },
        "exact_output_tensor_agreement": {
            "passed": total_output_disagreements == 0,
            "observed_disagreements": total_output_disagreements,
            "required": 0,
        },
        "positive_forward_evaluation_reduction": {
            "passed": (
                total_progressive_forward_evaluations
                < total_legacy_forward_evaluations
            ),
            "legacy_forward_evaluations": total_legacy_forward_evaluations,
            "progressive_forward_evaluations": total_progressive_forward_evaluations,
        },
        "per_request_forward_evaluation_reduction": {
            "passed": non_reducing_requests == 0,
            "non_reducing_request_count": non_reducing_requests,
            "required": 0,
        },
        "mutual_stability_is_shadow_only": {
            "passed": shadow_selection_violations == 0,
            "selection_policy_violation_count": shadow_selection_violations,
            "required": 0,
        },
    }
    return {
        "summary": {
            "seed_count": len(seed_results),
            "total_samples": total_requests,
            "exact_selected_cycle_disagreement_count": (
                total_selected_cycle_disagreements
            ),
            "exact_prediction_disagreement_count": total_prediction_disagreements,
            "exact_output_tensor_disagreement_count": total_output_disagreements,
            "maximum_absolute_logit_difference": round(
                maximum_abs_logit_difference, 12
            ),
            "legacy_correct_predictions": total_legacy_correct,
            "progressive_correct_predictions": total_progressive_correct,
            "legacy_forward_evaluations": total_legacy_forward_evaluations,
            "progressive_forward_evaluations": total_progressive_forward_evaluations,
            "forward_evaluation_reduction_percent": round(forward_reduction, 6),
            "weighted_mean_latency_reduction_percent": round(
                weighted_latency_reduction, 6
            ),
            "median_per_seed_latency_reduction_percent": round(
                float(statistics.median(per_seed_latency_reductions)), 6
            ),
            "weighted_legacy_mean_latency_ms": round(
                total_legacy_latency_ms / total_requests, 6
            ),
            "weighted_progressive_mean_latency_ms": round(
                total_progressive_latency_ms / total_requests, 6
            ),
            "selected_cycle_pairs": dict(sorted(selected_cycle_pairs.items())),
            "observed_prediction_margin": {
                "metric": "top_1_minus_top_2_probability",
                "legacy_v1": {
                    "observation_count": legacy_margin_observations,
                    "minimum": (
                        round(minimum_legacy_prediction_margin, 8)
                        if legacy_margin_observations
                        else None
                    ),
                },
                "progressive": {
                    "observation_count": progressive_margin_observations,
                    "minimum": (
                        round(minimum_progressive_prediction_margin, 8)
                        if progressive_margin_observations
                        else None
                    ),
                },
            },
            "observed_prediction_decision_margin": {
                "metric": (
                    "minimum_adjacent_probability_gap_through_rank_depth"
                ),
                "legacy_v1": {
                    "observation_count": legacy_decision_margin_observations,
                    "minimum": (
                        round(minimum_legacy_prediction_decision_margin, 8)
                        if legacy_decision_margin_observations
                        else None
                    ),
                },
                "progressive": {
                    "observation_count": (
                        progressive_decision_margin_observations
                    ),
                    "minimum": (
                        round(minimum_progressive_prediction_decision_margin, 8)
                        if progressive_decision_margin_observations
                        else None
                    ),
                },
            },
        },
        "per_seed_summary": per_seed,
        "gates": {
            "passed": all(bool(check["passed"]) for check in checks.values()),
            "checks": checks,
        },
    }


def run_benchmark(
    *,
    weights: Path,
    seeds: Sequence[int],
    samples_per_seed: int,
    device: Any,
    device_info: Mapping[str, Any],
    cycles: Sequence[int] = DEFAULT_CYCLES,
    confidence_target: float = DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    entropy_target: float = DEFAULT_AUTO_COMPUTE_ENTROPY,
    adaptive_compute: bool = False,
    exit_tol: float = DEFAULT_ADAPTIVE_EXIT_TOL,
    exit_entropy_threshold: float = DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    prediction_stability_patience: int = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    prediction_stability_tol: float = DEFAULT_PREDICTION_STABILITY_TOL,
    prediction_stability_margin: float = DEFAULT_PREDICTION_STABILITY_MARGIN,
    prediction_stability_rank_depth: int = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    distribution_top_k: int = DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
    model_loader: Callable[[Path, Any], Any] | None = None,
    task_factory: Callable[..., Tuple[torch.Tensor, torch.Tensor]] | None = None,
    benchmark_fn: Callable[..., Dict[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    weights = Path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Missing v51 checkpoint: {weights}")
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds:
        raise ValueError("At least one held-out seed is required")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Seeds must be unique")
    overlapping_seeds = sorted(TRAINING_AND_ORIGINAL_TEST_SEEDS.intersection(normalized_seeds))
    if overlapping_seeds:
        raise ValueError(
            "Held-out seeds overlap v51 training/original-test seeds: "
            + ", ".join(str(seed) for seed in overlapping_seeds)
        )
    if int(samples_per_seed) <= 0:
        raise ValueError("samples_per_seed must be positive")
    resolved_cycles = resolve_runtime_compute_cycles(cycles)
    if len(resolved_cycles) < 2:
        raise ValueError("At least two distinct compute-cycle candidates are required")
    normalized_confidence_target = _finite_float(
        confidence_target, label="confidence target"
    )
    normalized_entropy_target = _finite_float(
        entropy_target, label="entropy target"
    )
    if not 0.0 <= normalized_confidence_target <= 1.0:
        raise ValueError("confidence_target must be between 0 and 1")
    if normalized_entropy_target < 0.0:
        raise ValueError("entropy_target must be nonnegative")
    normalized_prediction_stability_margin = _nonnegative_finite_float(
        prediction_stability_margin,
        label="prediction stability margin",
    )
    normalized_prediction_stability_rank_depth = int(
        prediction_stability_rank_depth
    )
    if normalized_prediction_stability_rank_depth < 0:
        raise ValueError("prediction_stability_rank_depth must be nonnegative")

    loader = model_loader or _load_model
    task = task_factory or make_chained_task
    benchmark = benchmark_fn or benchmark_requests
    checkpoint_sha256 = _sha256_file(weights)
    model = loader(weights, device)
    if hasattr(model, "eval"):
        model.eval()
    seed_results: List[Dict[str, Any]] = []
    for seed_index, seed in enumerate(normalized_seeds):
        x, y = task(int(samples_per_seed), seed=seed)
        metrics = benchmark(
            model,
            x.to(device),
            y.to(device),
            cycles=resolved_cycles,
            confidence_target=normalized_confidence_target,
            entropy_target=normalized_entropy_target,
            adaptive_compute=bool(adaptive_compute),
            exit_tol=float(exit_tol),
            exit_entropy_threshold=float(exit_entropy_threshold),
            prediction_stability_patience=int(prediction_stability_patience),
            prediction_stability_tol=float(prediction_stability_tol),
            prediction_stability_margin=(
                normalized_prediction_stability_margin
            ),
            prediction_stability_rank_depth=(
                normalized_prediction_stability_rank_depth
            ),
            distribution_top_k=int(distribution_top_k),
            order_offset=seed_index % 2,
        )
        seed_results.append({"seed": seed, "metrics": metrics})

    aggregate = aggregate_results(seed_results)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(weights),
            "sha256": checkpoint_sha256,
        },
        "configuration": {
            "seeds": normalized_seeds,
            "samples_per_seed": int(samples_per_seed),
            "total_samples": int(samples_per_seed) * len(normalized_seeds),
            "data": {
                "generator": "make_chained_task",
                "kind": "deterministic_synthetic_chained_arithmetic",
                "split": "held_out_rng_seeds",
                "excluded_training_and_original_test_seeds": sorted(
                    TRAINING_AND_ORIGINAL_TEST_SEEDS
                ),
            },
            "cycles": resolved_cycles,
            "confidence_target": normalized_confidence_target,
            "entropy_target": normalized_entropy_target,
            "adaptive_compute": bool(adaptive_compute),
            "exit_tol": float(exit_tol),
            "exit_entropy_threshold": float(exit_entropy_threshold),
            "prediction_stability_patience": int(
                prediction_stability_patience
            ),
            "prediction_stability_tol": float(prediction_stability_tol),
            "prediction_stability_margin": (
                normalized_prediction_stability_margin
            ),
            "prediction_stability_rank_depth": (
                normalized_prediction_stability_rank_depth
            ),
            "distribution_top_k": int(distribution_top_k),
            "counterbalance": (
                "alternate_per_request_with_alternating_seed_offset"
            ),
            "latency_is_release_gate": False,
            "mutual_stability_role": "shadow_diagnostic_only",
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
        description=(
            "Compare progressive accepted-probe auto compute with legacy v1 "
            "on deterministic held-out v51 requests."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--samples-per-seed",
        "--samples_per_seed",
        dest="samples_per_seed",
        type=int,
        default=DEFAULT_SAMPLES_PER_SEED,
    )
    parser.add_argument("--cycles", nargs="+", type=int, default=list(DEFAULT_CYCLES))
    parser.add_argument(
        "--confidence-target",
        "--confidence_target",
        type=float,
        default=DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    )
    parser.add_argument(
        "--entropy-target",
        "--entropy_target",
        type=float,
        default=DEFAULT_AUTO_COMPUTE_ENTROPY,
    )
    parser.add_argument("--adaptive-compute", action="store_true")
    parser.add_argument(
        "--exit-tol", type=float, default=DEFAULT_ADAPTIVE_EXIT_TOL
    )
    parser.add_argument(
        "--exit-entropy-threshold",
        type=float,
        default=DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    )
    parser.add_argument(
        "--prediction-stability-patience",
        type=int,
        default=DEFAULT_PREDICTION_STABILITY_PATIENCE,
    )
    parser.add_argument(
        "--prediction-stability-tol",
        type=float,
        default=DEFAULT_PREDICTION_STABILITY_TOL,
    )
    parser.add_argument(
        "--prediction-stability-margin",
        "--prediction_stability_margin",
        dest="prediction_stability_margin",
        type=float,
        default=DEFAULT_PREDICTION_STABILITY_MARGIN,
    )
    parser.add_argument(
        "--prediction-stability-rank-depth",
        "--prediction_stability_rank_depth",
        dest="prediction_stability_rank_depth",
        type=int,
        default=DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    )
    parser.add_argument(
        "--distribution-top-k",
        type=int,
        default=DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device-preference", default="cuda,npu,xpu,dml,mps,cpu"
    )
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-interop-threads", type=int, default=0)
    parser.add_argument("--strict-determinism", action="store_true")
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Exit with status 2 unless every exact-agreement gate passes.",
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
    payload = run_benchmark(
        weights=Path(args.weights),
        seeds=args.seeds,
        samples_per_seed=int(args.samples_per_seed),
        device=device,
        device_info=device_info,
        cycles=args.cycles,
        confidence_target=float(args.confidence_target),
        entropy_target=float(args.entropy_target),
        adaptive_compute=bool(args.adaptive_compute),
        exit_tol=float(args.exit_tol),
        exit_entropy_threshold=float(args.exit_entropy_threshold),
        prediction_stability_patience=int(args.prediction_stability_patience),
        prediction_stability_tol=float(args.prediction_stability_tol),
        prediction_stability_margin=float(args.prediction_stability_margin),
        prediction_stability_rank_depth=int(
            args.prediction_stability_rank_depth
        ),
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
