"""Benchmark v51 prediction-stability exits under request-sized inference.

The standalone defaults disable the older latent/entropy exits to isolate the
full-output verifier.  The multi-seed release gate also calls this benchmark
with the authoritative deployed runtime controls and metadata class scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import numbers
from pathlib import Path
import time
from typing import Any, Dict, List, Sequence

import torch

from benchmark_cognitive_leap_ultra_v51 import make_chained_task
import chat_app
from device_utils import configure_torch_runtime, resolve_device
from model_variants import ChampionNetCognitiveLeapUltraExpert


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "output" / "benchmark_v51_cognitive_leap_ultra_latest"
DEFAULT_WEIGHTS = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "prediction_stability_results.json"
DEFAULT_ADAPTIVE_EXIT_TOL = chat_app.DEFAULT_ADAPTIVE_EXIT_TOL
DEFAULT_ADAPTIVE_EXIT_ENTROPY = chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY
DEFAULT_PREDICTION_STABILITY_PATIENCE = (
    chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE
)
DEFAULT_PREDICTION_STABILITY_TOL = chat_app.DEFAULT_PREDICTION_STABILITY_TOL
DEFAULT_PREDICTION_STABILITY_MARGIN = chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
DEFAULT_PREDICTION_STABILITY_RANK_DEPTH = (
    chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
)
DEFAULT_DECISION_TOP_K = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
KNOWN_ADAPTIVE_EXIT_REASONS = frozenset(
    {
        "prediction_stable",
        "latent_converged",
        "low_entropy",
        "halt_mass",
        "max_cycles",
    }
)


def _load_model(weights: Path, device: torch.device) -> ChampionNetCognitiveLeapUltraExpert:
    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model = ChampionNetCognitiveLeapUltraExpert().to(device).eval()
    model.load_state_dict(state)
    return model


def _finite_values(values: Sequence[float], *, label: str) -> torch.Tensor:
    if not values:
        raise ValueError(f"{label} requires at least one value")
    tensor = torch.tensor(values, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{label} must contain only finite values")
    return tensor


def _latency_summary(values: List[float]) -> Dict[str, float]:
    tensor = _finite_values(values, label="latency summary")
    if bool((tensor <= 0.0).any().item()):
        raise ValueError("latency values must be positive")
    return {
        "total_ms": round(float(tensor.sum().item()), 6),
        "mean_ms": round(float(tensor.mean().item()), 3),
        "p50_ms": round(float(torch.quantile(tensor, 0.50).item()), 3),
        "p95_ms": round(float(torch.quantile(tensor, 0.95).item()), 3),
    }


def _scalar_summary(values: List[float], *, digits: int = 8) -> Dict[str, float]:
    tensor = _finite_values(values, label="scalar summary")
    return {
        "min": round(float(tensor.min().item()), digits),
        "mean": round(float(tensor.mean().item()), digits),
        "p50": round(float(torch.quantile(tensor, 0.50).item()), digits),
        "p95": round(float(torch.quantile(tensor, 0.95).item()), digits),
        "max": round(float(tensor.max().item()), digits),
    }


def _nonnegative_finite_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{label} must be finite and nonnegative, got {value!r}"
        ) from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value!r}")
    return number


def _strict_bool_scalar(value: Any, *, label: str) -> bool:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{label} must be a scalar boolean")
        value = value.item()
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _positive_integral_scalar(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise ValueError(f"{label} must be a scalar integer")
        value = value.item()
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    number = _nonnegative_finite_float(value, label=label)
    integer = int(number)
    if integer <= 0 or number != float(integer):
        raise ValueError(f"{label} must be a positive integer, got {value!r}")
    return integer


def _normalize_prediction_class_indices_input(
    prediction_class_indices: Any,
) -> List[int] | List[bool] | None:
    """Normalize one optional verifier scope without guessing output width."""

    if prediction_class_indices is None:
        return None
    if torch.is_tensor(prediction_class_indices):
        raw = prediction_class_indices.detach().cpu()
        if raw.ndim != 1:
            raise ValueError("prediction_class_indices must be one-dimensional")
        if raw.dtype == torch.bool:
            values: List[int] | List[bool] = [bool(value) for value in raw.tolist()]
        elif raw.dtype.is_floating_point or raw.dtype.is_complex:
            raise ValueError("prediction_class_indices must contain integers or be a bool mask")
        else:
            values = [int(value) for value in raw.tolist()]
    else:
        if isinstance(prediction_class_indices, (str, bytes)):
            raise ValueError("prediction_class_indices must be an integer sequence")
        try:
            raw_values = list(prediction_class_indices)
        except TypeError as exc:
            raise ValueError(
                "prediction_class_indices must be an integer sequence"
            ) from exc
        if raw_values and all(isinstance(value, bool) for value in raw_values):
            values = [bool(value) for value in raw_values]
        else:
            if any(isinstance(value, bool) for value in raw_values) or any(
                not isinstance(value, numbers.Integral) for value in raw_values
            ):
                raise ValueError(
                    "prediction_class_indices must contain only integers or be a bool mask"
                )
            values = [int(value) for value in raw_values]

    if not values:
        raise ValueError("prediction_class_indices cannot be empty")
    if all(isinstance(value, bool) for value in values):
        if not any(values):
            raise ValueError("prediction_class_indices bool mask must select a class")
        return values
    integer_values = [int(value) for value in values]
    if any(value < 0 for value in integer_values):
        raise ValueError("prediction_class_indices cannot contain negative indices")
    if len(set(integer_values)) != len(integer_values):
        raise ValueError("prediction_class_indices cannot contain duplicates")
    return integer_values


def _request_logits(output: Any, *, label: str) -> torch.Tensor:
    if not torch.is_tensor(output):
        raise ValueError(f"{label} output must be a tensor")
    detached = output.detach()
    if detached.ndim < 1 or int(detached.shape[-1]) < 1:
        raise ValueError(f"{label} output must have a non-empty class dimension")
    if int(detached.numel()) != int(detached.shape[-1]):
        raise ValueError(f"{label} output must contain exactly one serving request")
    logits = detached.reshape(-1).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError(f"{label} logits must be finite")
    return logits


def _resolve_prediction_scope(
    normalized_scope: List[int] | List[bool] | None,
    *,
    output_class_count: int,
    decision_top_k: int,
) -> List[int]:
    if normalized_scope is None:
        resolved = list(range(output_class_count))
    elif all(isinstance(value, bool) for value in normalized_scope):
        if len(normalized_scope) != output_class_count:
            raise ValueError(
                "prediction_class_indices bool mask length must match output classes"
            )
        resolved = [index for index, selected in enumerate(normalized_scope) if selected]
    else:
        resolved = [int(value) for value in normalized_scope]
        if max(resolved) >= output_class_count:
            raise ValueError("prediction_class_indices contains an out-of-range class")
    if len(resolved) < decision_top_k:
        raise ValueError(
            f"decision_top_k={decision_top_k} requires at least {decision_top_k} scoped classes"
        )
    return resolved


def _distribution_distances(
    fixed_logits: torch.Tensor,
    adaptive_logits: torch.Tensor,
) -> tuple[float, float]:
    fixed_distribution = torch.softmax(fixed_logits, dim=-1)
    adaptive_distribution = torch.softmax(adaptive_logits, dim=-1)
    midpoint = 0.5 * (fixed_distribution + adaptive_distribution)
    fixed_kl = torch.sum(
        torch.special.xlogy(fixed_distribution, fixed_distribution)
        - torch.special.xlogy(fixed_distribution, midpoint)
    )
    adaptive_kl = torch.sum(
        torch.special.xlogy(adaptive_distribution, adaptive_distribution)
        - torch.special.xlogy(adaptive_distribution, midpoint)
    )
    js_divergence = float((0.5 * (fixed_kl + adaptive_kl)).item())
    total_variation = float(
        (0.5 * torch.sum(torch.abs(fixed_distribution - adaptive_distribution))).item()
    )
    # JSD is nonnegative by definition. Float32 cancellation can produce a
    # small negative value for nearly identical distributions; retain a strict
    # bound so materially invalid telemetry still fails closed.
    if -1e-6 < js_divergence < 0.0:
        js_divergence = 0.0
    return (
        _nonnegative_finite_float(js_divergence, label="distribution JSD"),
        _nonnegative_finite_float(total_variation, label="total variation distance"),
    )


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
    prediction_stability_margin: float = DEFAULT_PREDICTION_STABILITY_MARGIN,
    prediction_stability_rank_depth: int = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    decision_top_k: int = DEFAULT_DECISION_TOP_K,
    prediction_class_indices: Any = None,
    exit_tol: float = 0.0,
    exit_entropy_threshold: float = 0.0,
    order_offset: int = 0,
) -> Dict[str, Any]:
    request_count = int(len(y))
    if request_count <= 0:
        raise ValueError("At least one serving request is required")
    if int(len(x)) != request_count:
        raise ValueError("x and y must contain the same number of requests")
    normalized_fixed_cycles = int(fixed_cycles)
    normalized_max_cycles = int(max_cycles)
    normalized_patience = int(stability_patience)
    normalized_distribution_top_k = int(distribution_top_k)
    normalized_decision_top_k = int(decision_top_k)
    normalized_rank_depth = int(prediction_stability_rank_depth)
    normalized_stability_tol = _nonnegative_finite_float(
        stability_tol,
        label="stability tolerance",
    )
    normalized_exit_tol = _nonnegative_finite_float(
        exit_tol,
        label="adaptive exit tolerance",
    )
    normalized_exit_entropy_threshold = _nonnegative_finite_float(
        exit_entropy_threshold,
        label="adaptive exit entropy threshold",
    )
    stability_margin = _nonnegative_finite_float(
        prediction_stability_margin,
        label="prediction stability margin",
    )
    if normalized_fixed_cycles <= 0 or normalized_max_cycles <= 0:
        raise ValueError("fixed_cycles and max_cycles must be positive")
    if normalized_patience < 0:
        raise ValueError("stability_patience must be nonnegative")
    if normalized_distribution_top_k <= 0:
        raise ValueError("distribution_top_k must be positive")
    if normalized_decision_top_k <= 0:
        raise ValueError("decision_top_k must be positive")
    if normalized_rank_depth <= 0:
        raise ValueError("prediction_stability_rank_depth must be positive")
    normalized_scope = _normalize_prediction_class_indices_input(
        prediction_class_indices
    )
    head = model.layers[10]
    fixed_predictions: List[int] = []
    adaptive_predictions: List[int] = []
    fixed_latencies: List[float] = []
    adaptive_latencies: List[float] = []
    adaptive_cycles: List[float] = []
    adaptive_latest_js_drift: List[float] = []
    adaptive_max_js_drift: List[float] = []
    adaptive_prediction_margins: List[float] = []
    adaptive_decision_margins: List[float] = []
    prediction_stable_decision_margins: List[float] = []
    adaptive_rank_depths: List[int] = []
    adaptive_prediction_class_counts: List[int] = []
    adaptive_requested_class_indices: List[List[int] | None] = []
    absolute_logit_deltas: List[float] = []
    distribution_js_divergences: List[float] = []
    distribution_total_variations: List[float] = []
    top_k_order_disagreement_count = 0
    top_k_set_disagreement_count = 0
    resolved_scope: List[int] | None = None
    output_class_count: int | None = None
    exit_reasons: Counter[str] = Counter()
    measurement_orders: Counter[str] = Counter()

    normalized_requested_class_indices = (
        None
        if normalized_scope is None
        else [
            index
            for index, selected in enumerate(normalized_scope)
            if selected
        ]
        if all(isinstance(value, bool) for value in normalized_scope)
        else [int(value) for value in normalized_scope]
    )

    def adaptive_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "reasoning_cycles": normalized_max_cycles,
            "adaptive_compute": True,
            "exit_tol": normalized_exit_tol,
            "exit_entropy_threshold": normalized_exit_entropy_threshold,
            "prediction_stability_patience": normalized_patience,
            "prediction_stability_tol": normalized_stability_tol,
            "prediction_stability_top_k": normalized_distribution_top_k,
            "prediction_stability_margin": stability_margin,
            "prediction_stability_rank_depth": normalized_rank_depth,
        }
        if normalized_scope is not None:
            kwargs["prediction_class_indices"] = normalized_scope
        return kwargs

    def record_requested_scope(kwargs: Dict[str, Any]) -> None:
        raw_scope = kwargs.get("prediction_class_indices")
        if raw_scope is None:
            adaptive_requested_class_indices.append(None)
            return
        if all(isinstance(value, bool) for value in raw_scope):
            adaptive_requested_class_indices.append(
                [index for index, selected in enumerate(raw_scope) if selected]
            )
            return
        adaptive_requested_class_indices.append([int(value) for value in raw_scope])

    def read_adaptive_scope_telemetry(*, label: str) -> int:
        selection_valid = _strict_bool_scalar(
            getattr(head, "last_prediction_class_selection_valid", None),
            label=f"{label} prediction class selection validity",
        )
        if not selection_valid:
            raise ValueError(f"{label} prediction class selection is invalid")
        return _positive_integral_scalar(
            getattr(head, "last_prediction_class_count", None),
            label=f"{label} prediction class count",
        )

    # Keep initialization and one-time kernel setup outside measured requests.
    # End the warmup on the opposite mode from the first measured request so
    # both seed offsets receive the same cross-mode transition.
    def warm_adaptive() -> None:
        kwargs = adaptive_kwargs()
        record_requested_scope(kwargs)
        model(x[:1], **kwargs)
        adaptive_prediction_class_counts.append(
            read_adaptive_scope_telemetry(label="adaptive warmup")
        )

    if int(order_offset) % 2 == 0:
        model(x[:1], reasoning_cycles=normalized_fixed_cycles)
        warm_adaptive()
    else:
        warm_adaptive()
        model(x[:1], reasoning_cycles=normalized_fixed_cycles)

    def run_fixed(sample: torch.Tensor) -> torch.Tensor:
        _synchronize_for_timing(sample.device)
        started = time.perf_counter()
        output = model(sample, reasoning_cycles=normalized_fixed_cycles)
        _synchronize_for_timing(sample.device)
        fixed_latencies.append(
            _nonnegative_finite_float(
                (time.perf_counter() - started) * 1000.0,
                label="fixed request latency",
            )
        )
        return output

    def run_adaptive(sample: torch.Tensor) -> torch.Tensor:
        _synchronize_for_timing(sample.device)
        started = time.perf_counter()
        kwargs = adaptive_kwargs()
        record_requested_scope(kwargs)
        output = model(sample, **kwargs)
        _synchronize_for_timing(sample.device)
        adaptive_latencies.append(
            _nonnegative_finite_float(
                (time.perf_counter() - started) * 1000.0,
                label="adaptive request latency",
            )
        )
        adaptive_cycles.append(
            _nonnegative_finite_float(
                head.last_cycles_used.item(), label="adaptive cycles used"
            )
        )
        adaptive_latest_js_drift.append(
            _nonnegative_finite_float(
                head.last_prediction_topk_js_divergence.item(),
                label="latest prediction distribution drift",
            )
        )
        adaptive_max_js_drift.append(
            _nonnegative_finite_float(
                head.last_prediction_topk_js_divergence_max.item(),
                label="maximum prediction distribution drift",
            )
        )
        adaptive_prediction_margins.append(
            _nonnegative_finite_float(
                head.last_prediction_margin.item(),
                label="observed prediction stability margin",
            )
        )
        observed_decision_margin = _nonnegative_finite_float(
            head.last_prediction_decision_margin.item(),
            label="observed prediction decision margin",
        )
        adaptive_decision_margins.append(observed_decision_margin)
        observed_rank_depth = int(head.last_prediction_rank_depth.item())
        if observed_rank_depth <= 0:
            raise ValueError("observed prediction rank depth must be positive")
        adaptive_rank_depths.append(observed_rank_depth)
        adaptive_prediction_class_counts.append(
            read_adaptive_scope_telemetry(label="adaptive request")
        )
        raw_exit_reason = getattr(head, "last_exit_reason", None)
        if not isinstance(raw_exit_reason, str) or not raw_exit_reason.strip():
            raise ValueError("adaptive exit reason must be a non-empty string")
        exit_reason = raw_exit_reason.strip()
        if exit_reason not in KNOWN_ADAPTIVE_EXIT_REASONS:
            raise ValueError(f"unknown adaptive exit reason: {exit_reason}")
        exit_reasons[exit_reason] += 1
        if exit_reason == "prediction_stable":
            prediction_stable_decision_margins.append(observed_decision_margin)
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

        fixed_logits = _request_logits(fixed_output, label="fixed")
        adaptive_logits = _request_logits(adaptive_output, label="adaptive")
        if fixed_logits.shape != adaptive_logits.shape:
            raise ValueError("fixed and adaptive logits must have the same shape")
        current_output_class_count = int(fixed_logits.numel())
        if output_class_count is None:
            output_class_count = current_output_class_count
            resolved_scope = _resolve_prediction_scope(
                normalized_scope,
                output_class_count=output_class_count,
                decision_top_k=normalized_decision_top_k,
            )
        elif current_output_class_count != output_class_count:
            raise ValueError("model output class count changed between requests")
        assert resolved_scope is not None
        scope_tensor = torch.tensor(resolved_scope, dtype=torch.long)
        fixed_scoped = fixed_logits.index_select(0, scope_tensor)
        adaptive_scoped = adaptive_logits.index_select(0, scope_tensor)
        fixed_order_local = torch.argsort(
            fixed_scoped, descending=True, stable=True
        )[:normalized_decision_top_k]
        adaptive_order_local = torch.argsort(
            adaptive_scoped, descending=True, stable=True
        )[:normalized_decision_top_k]
        fixed_order = [
            resolved_scope[int(index.item())] for index in fixed_order_local
        ]
        adaptive_order = [
            resolved_scope[int(index.item())] for index in adaptive_order_local
        ]
        fixed_predictions.append(fixed_order[0])
        adaptive_predictions.append(adaptive_order[0])
        top_k_order_disagreement_count += int(fixed_order != adaptive_order)
        top_k_set_disagreement_count += int(set(fixed_order) != set(adaptive_order))
        absolute_logit_deltas.extend(
            float(value)
            for value in torch.abs(fixed_scoped - adaptive_scoped).tolist()
        )
        js_divergence, total_variation = _distribution_distances(
            fixed_scoped, adaptive_scoped
        )
        distribution_js_divergences.append(js_divergence)
        distribution_total_variations.append(total_variation)

    assert output_class_count is not None and resolved_scope is not None
    truth = y.detach().cpu().reshape(-1)
    if truth.numel() != request_count:
        raise ValueError("y must contain one target per serving request")
    if truth.dtype.is_floating_point or truth.dtype.is_complex:
        raise ValueError("y must contain integer class indices")
    if bool(((truth < 0) | (truth >= output_class_count)).any().item()):
        raise ValueError("y contains an out-of-range class index")
    fixed_tensor = torch.tensor(fixed_predictions)
    adaptive_tensor = torch.tensor(adaptive_predictions)
    fixed_correct = int((fixed_tensor == truth).sum().item())
    adaptive_correct = int((adaptive_tensor == truth).sum().item())
    top1_disagreement_count = int((fixed_tensor != adaptive_tensor).sum().item())
    fixed_accuracy = fixed_correct / max(1, request_count)
    adaptive_accuracy = adaptive_correct / max(1, request_count)
    agreement = 1.0 - (top1_disagreement_count / max(1, request_count))
    fixed_latency = _latency_summary(fixed_latencies)
    adaptive_latency = _latency_summary(adaptive_latencies)
    mean_cycles = float(
        _finite_values(adaptive_cycles, label="adaptive cycles").mean().item()
    )
    expected_effective_rank_depth = min(
        normalized_rank_depth, max(1, len(resolved_scope) - 1)
    )
    if set(adaptive_rank_depths) != {expected_effective_rank_depth}:
        raise ValueError(
            "adaptive verifier reported a rank depth inconsistent with the verified scope"
        )
    if set(adaptive_prediction_class_counts) != {len(resolved_scope)}:
        raise ValueError(
            "adaptive verifier reported a class count inconsistent with the verified scope"
        )
    if set(
        None if value is None else tuple(value)
        for value in adaptive_requested_class_indices
    ) != {
        None
        if normalized_requested_class_indices is None
        else tuple(normalized_requested_class_indices)
    }:
        raise ValueError(
            "adaptive verifier calls did not preserve the exact requested class indices"
        )
    scoped_truth_count = int(
        torch.isin(truth, torch.tensor(resolved_scope, dtype=truth.dtype)).sum().item()
    )
    verified_scope = {
        "verified": True,
        "mode": (
            "all_output_classes"
            if normalized_scope is None
            else "prediction_class_indices"
        ),
        "output_class_count": output_class_count,
        "class_count": len(resolved_scope),
        "class_indices": resolved_scope,
        "requested_normalized_class_indices": normalized_requested_class_indices,
        "adaptive_requested_class_indices_verified": True,
        "truth_labels_in_scope": scoped_truth_count,
        "truth_labels_outside_scope": request_count - scoped_truth_count,
        "adaptive_verifier_rank_depth_verified": True,
        "adaptive_verifier_class_scope_verified": True,
        "observed_prediction_class_counts": sorted(
            set(adaptive_prediction_class_counts)
        ),
        "effective_prediction_rank_depth": expected_effective_rank_depth,
    }
    logit_delta_tensor = _finite_values(
        absolute_logit_deltas, label="absolute logit deltas"
    )

    return {
        "fixed": {
            "adaptive_compute": False,
            "cycles": normalized_fixed_cycles,
            "accuracy": round(fixed_accuracy, 6),
            "correct_predictions": fixed_correct,
            "latency": fixed_latency,
        },
        "prediction_stability": {
            "adaptive_compute": True,
            "max_cycles": normalized_max_cycles,
            "exit_tolerance": normalized_exit_tol,
            "exit_entropy_threshold": normalized_exit_entropy_threshold,
            "patience": normalized_patience,
            "confidence_tolerance": normalized_stability_tol,
            "prediction_margin": {
                "metric": "top_1_minus_top_2_probability",
                "configured_minimum": stability_margin,
                "observed": _scalar_summary(adaptive_prediction_margins),
            },
            "decision_margin": {
                "metric": "minimum_adjacent_probability_gap_through_rank_depth",
                "configured_minimum": stability_margin,
                "configured_rank_depth": normalized_rank_depth,
                "observed_rank_depths": sorted(set(adaptive_rank_depths)),
                "observed": _scalar_summary(adaptive_decision_margins),
                "prediction_stable_observed": {
                    "observation_count": len(prediction_stable_decision_margins),
                    "minimum": (
                        min(prediction_stable_decision_margins)
                        if prediction_stable_decision_margins
                        else None
                    ),
                    "summary": (
                        _scalar_summary(prediction_stable_decision_margins)
                        if prediction_stable_decision_margins
                        else None
                    ),
                },
            },
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
                "top_k": normalized_distribution_top_k,
                "role": "shadow_diagnostic_only",
                "latest_cycle_pair": _scalar_summary(adaptive_latest_js_drift),
                "max_cycle_pair": _scalar_summary(adaptive_max_js_drift),
            },
            "latency": adaptive_latency,
        },
        "comparison": {
            "accuracy_delta": round(adaptive_accuracy - fixed_accuracy, 6),
            "prediction_agreement": round(agreement, 6),
            "top1_disagreement_count": top1_disagreement_count,
            # Compatibility alias retained for existing artifact consumers.
            "exact_disagreement_count": top1_disagreement_count,
            "request_count": request_count,
            "decision_fidelity": {
                "verified_scope": verified_scope,
                "top_k": normalized_decision_top_k,
                "top_k_order_disagreement_count": top_k_order_disagreement_count,
                "top_k_set_disagreement_count": top_k_set_disagreement_count,
                "absolute_logit_delta": {
                    "mean": round(float(logit_delta_tensor.mean().item()), 12),
                    "max": round(float(logit_delta_tensor.max().item()), 12),
                    "role": "diagnostic_only_not_gated",
                },
                "distribution_distance": {
                    "jensen_shannon_divergence_nats": _scalar_summary(
                        distribution_js_divergences, digits=12
                    ),
                    "total_variation_distance": _scalar_summary(
                        distribution_total_variations, digits=12
                    ),
                    "role": "diagnostic_only_not_gated",
                },
                "tensor_equality_required": False,
            },
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
    parser.add_argument(
        "--stability_patience",
        type=int,
        default=DEFAULT_PREDICTION_STABILITY_PATIENCE,
    )
    parser.add_argument(
        "--stability_tol", type=float, default=DEFAULT_PREDICTION_STABILITY_TOL
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
        "--decision-top-k",
        "--decision_top_k",
        dest="decision_top_k",
        type=int,
        default=DEFAULT_DECISION_TOP_K,
    )
    parser.add_argument(
        "--prediction-class-indices",
        "--prediction_class_indices",
        dest="prediction_class_indices",
        type=int,
        nargs="+",
        default=None,
        help="Evaluate and verify one explicit class scope (single scope only).",
    )
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
        prediction_stability_margin=_nonnegative_finite_float(
            args.prediction_stability_margin,
            label="prediction stability margin",
        ),
        prediction_stability_rank_depth=int(args.prediction_stability_rank_depth),
        decision_top_k=int(args.decision_top_k),
        prediction_class_indices=args.prediction_class_indices,
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
