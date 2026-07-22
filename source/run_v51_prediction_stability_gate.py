"""Run the release gate for v51 full-output decision fidelity.

The default evaluation is 8 native seeds x 512 requests.  Each seed alternates
the fixed/adaptive measurement order, and the starting order alternates across
seeds, to reduce request-order latency bias without pooling the two timings.
Every seed is evaluated both with isolated verifier controls and with the exact
deployed runtime defaults plus the metadata-derived allowed-class scope.
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
    DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    DEFAULT_ADAPTIVE_EXIT_TOL,
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DECISION_TOP_K,
    DEFAULT_PREDICTION_STABILITY_PATIENCE,
    DEFAULT_PREDICTION_STABILITY_MARGIN,
    DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    DEFAULT_PREDICTION_STABILITY_TOL,
    DEFAULT_WEIGHTS,
    KNOWN_ADAPTIVE_EXIT_REASONS,
    _load_model,
    _normalize_prediction_class_indices_input,
    _resolve_prediction_scope,
    benchmark_serving_requests,
)
import chat_app
from device_utils import configure_torch_runtime, resolve_device


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "prediction_stability_gate.json"
DEFAULT_META = DEFAULT_ARTIFACT_DIR / "chat_demo_meta.json"
DEFAULT_SEEDS = (641, 643, 647, 653, 659, 661, 673, 677)
DEFAULT_SAMPLES_PER_SEED = 512
GATE_SCHEMA_VERSION = 4
RELEASE_FIXED_CYCLES = 3
RELEASE_MAX_CYCLES = 8
ISOLATED_VERIFIER_EXIT_TOL = 0.0
ISOLATED_VERIFIER_EXIT_ENTROPY = 0.0
MINIMUM_WEIGHTED_CYCLE_REDUCTION_PERCENT = 20.0
MINIMUM_MEDIAN_LATENCY_REDUCTION_PERCENT = 0.0
EXPECTED_RELEASE_MODEL_SIZE = "cognitive_leap_ultra_expert"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_prediction_scope(metadata: Path) -> tuple[List[int], Dict[str, Any]]:
    """Resolve the exact allowed-class scope used by the source chat runtime."""

    metadata = Path(metadata).expanduser().resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"Missing v51 chat metadata: {metadata}")
    try:
        raw = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid v51 chat metadata: {metadata}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("v51 chat metadata must be a JSON object")
    observed_model_size = raw.get("model_size")
    if observed_model_size is not None:
        if (
            not isinstance(observed_model_size, str)
            or observed_model_size.strip() != EXPECTED_RELEASE_MODEL_SIZE
        ):
            raise ValueError(
                "v51 chat metadata model_size must be "
                f"{EXPECTED_RELEASE_MODEL_SIZE!r}"
            )
        observed_model_size = observed_model_size.strip()
    observed_num_classes = raw.get("num_classes")
    if observed_num_classes is not None:
        if type(observed_num_classes) is not int:
            raise ValueError("v51 chat metadata num_classes must be an integer")
        observed_num_classes = _bounded_count(
            observed_num_classes,
            maximum=int(chat_app.MODEL_CLASSES),
            label="v51 chat metadata num_classes",
        )
        if observed_num_classes != int(chat_app.MODEL_CLASSES):
            raise ValueError(
                "v51 chat metadata num_classes does not match the runtime model"
            )
    buckets = chat_app._parse_metadata_buckets(raw.get("buckets", {}))
    if buckets:
        class_indices = sorted(int(label) for label in buckets)
        source = "metadata_nonempty_buckets"
    else:
        class_indices = list(range(int(chat_app.MODEL_CLASSES)))
        source = "runtime_all_class_fallback"
    normalized = _normalize_prediction_class_indices_input(class_indices)
    if normalized is None or any(isinstance(value, bool) for value in normalized):
        raise ValueError("Metadata allowed-class scope must resolve to integer indices")
    return [int(value) for value in normalized], {
        "path": str(metadata),
        "sha256": _sha256_file(metadata),
        "scope_source": source,
        "allowed_class_indices": [int(value) for value in normalized],
        "allowed_class_count": len(normalized),
        "model_identity": {
            "field_present": observed_model_size is not None,
            "observed_model_size": observed_model_size,
            "expected_model_size": EXPECTED_RELEASE_MODEL_SIZE,
            "verified_when_present": True,
        },
        "class_identity": {
            "field_present": observed_num_classes is not None,
            "observed_num_classes": observed_num_classes,
            "expected_num_classes": int(chat_app.MODEL_CLASSES),
            "verified_when_present": True,
        },
    }


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
        PROJECT_ROOT / "source" / "chat_app.py",
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
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _nonnegative_finite_float(value: Any, *, label: str) -> float:
    number = _finite_float(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative, got {value!r}")
    return number


def _strict_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _bounded_count(value: Any, *, maximum: int, label: str) -> int:
    numeric_value = _finite_float(value, label=label)
    count = int(numeric_value)
    if numeric_value != count:
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if count < 0 or count > maximum:
        raise ValueError(f"{label} must be between 0 and {maximum}, got {value!r}")
    return count


def _validated_scalar_summary(
    summary: Mapping[str, Any],
    *,
    label: str,
    upper_bound: float | None = None,
) -> Dict[str, float]:
    values = {
        key: _nonnegative_finite_float(summary[key], label=f"{label} {key}")
        for key in ("min", "mean", "p50", "p95", "max")
    }
    if not (
        values["min"] <= values["p50"] <= values["p95"] <= values["max"]
        and values["min"] <= values["mean"] <= values["max"]
    ):
        raise ValueError(f"{label} summary ordering is invalid")
    if upper_bound is not None and values["max"] > upper_bound + 1e-9:
        raise ValueError(f"{label} exceeds its mathematical upper bound")
    return values


def aggregate_gate_results(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    fixed_cycles: int,
    max_cycles: int,
    prediction_class_indices: Any,
    decision_top_k: int,
    prediction_stability_rank_depth: int,
    prediction_stability_margin: float,
    prediction_stability_patience: int,
    prediction_stability_tol: float,
    exit_tol: float,
    exit_entropy_threshold: float,
    distribution_top_k: int,
) -> Dict[str, Any]:
    """Aggregate counts, bounded diagnostics, and release-gate invariants."""

    if not seed_results:
        raise ValueError("At least one seed result is required")
    expected_fixed_cycles = int(fixed_cycles)
    expected_max_cycles = int(max_cycles)
    expected_patience = int(prediction_stability_patience)
    expected_stability_tolerance = _nonnegative_finite_float(
        prediction_stability_tol,
        label="configured prediction stability tolerance",
    )
    expected_margin = _nonnegative_finite_float(
        prediction_stability_margin,
        label="configured prediction stability margin",
    )
    expected_exit_tolerance = _nonnegative_finite_float(
        exit_tol,
        label="configured adaptive exit tolerance",
    )
    expected_exit_entropy = _nonnegative_finite_float(
        exit_entropy_threshold,
        label="configured adaptive exit entropy threshold",
    )
    expected_rank_depth = int(prediction_stability_rank_depth)
    expected_decision_top_k = int(decision_top_k)
    expected_distribution_top_k = int(distribution_top_k)
    expected_scope_input = _normalize_prediction_class_indices_input(
        prediction_class_indices
    )
    if expected_fixed_cycles <= 0 or expected_max_cycles <= 0:
        raise ValueError("configured fixed_cycles and max_cycles must be positive")
    if expected_patience < 0:
        raise ValueError("configured prediction stability patience must be nonnegative")
    if expected_rank_depth <= 0 or expected_decision_top_k <= 0:
        raise ValueError("configured rank depth and decision top-k must be positive")
    if expected_rank_depth < expected_decision_top_k:
        raise ValueError("configured rank depth must cover decision top-k")
    if expected_distribution_top_k <= 0:
        raise ValueError("configured distribution top-k must be positive")
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
    total_top1_disagreements = 0
    total_top_k_order_disagreements = 0
    total_top_k_set_disagreements = 0
    total_fixed_cycles = 0.0
    total_adaptive_cycles = 0.0
    latency_reductions: List[float] = []
    total_fixed_latency_ms = 0.0
    total_adaptive_latency_ms = 0.0
    seen_seeds: set[int] = set()
    minimum_observed_prediction_margin = math.inf
    minimum_observed_decision_margin = math.inf
    decision_top_k: int | None = None
    prediction_rank_depth: int | None = None
    configured_prediction_stability_margin: float | None = None
    canonical_adaptive_configuration: Dict[str, Any] | None = None
    canonical_scope: Dict[str, Any] | None = None
    total_truth_in_scope = 0
    total_truth_outside_scope = 0
    logit_delta_weighted_sum = 0.0
    logit_delta_value_count = 0
    maximum_absolute_logit_delta = 0.0
    js_weighted_sum = 0.0
    js_minimum = math.inf
    js_maximum = 0.0
    tv_weighted_sum = 0.0
    tv_minimum = math.inf
    tv_maximum = 0.0

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
        if _strict_bool(
            fixed.get("adaptive_compute"),
            label=f"seed {seed} fixed adaptive-compute flag",
        ):
            raise ValueError(f"Seed {seed} fixed mode enabled adaptive compute")
        if not _strict_bool(
            adaptive.get("adaptive_compute"),
            label=f"seed {seed} adaptive-compute flag",
        ):
            raise ValueError(f"Seed {seed} adaptive mode did not enable adaptive compute")

        fixed_correct = _bounded_count(
            fixed["correct_predictions"],
            maximum=request_count,
            label=f"seed {seed} fixed correct count",
        )
        adaptive_correct = _bounded_count(
            adaptive["correct_predictions"],
            maximum=request_count,
            label=f"seed {seed} adaptive correct count",
        )
        top1_disagreement_count = _bounded_count(
            comparison.get(
                "top1_disagreement_count",
                comparison.get("exact_disagreement_count"),
            ),
            maximum=request_count,
            label=f"seed {seed} top1 disagreement count",
        )
        legacy_disagreement_count = _bounded_count(
            comparison.get("exact_disagreement_count", top1_disagreement_count),
            maximum=request_count,
            label=f"seed {seed} exact disagreement count",
        )
        if legacy_disagreement_count != top1_disagreement_count:
            raise ValueError(f"Seed {seed} top1 disagreement aliases do not match")

        adaptive_patience = _bounded_count(
            adaptive["patience"],
            maximum=chat_app.MAX_RUNTIME_REASONING_CYCLES,
            label=f"seed {seed} prediction stability patience",
        )
        adaptive_max_cycles = _bounded_count(
            adaptive["max_cycles"],
            maximum=chat_app.MAX_RUNTIME_REASONING_CYCLES,
            label=f"seed {seed} adaptive max cycles",
        )
        seed_adaptive_configuration = {
            "max_cycles": adaptive_max_cycles,
            "exit_tolerance": _nonnegative_finite_float(
                adaptive["exit_tolerance"], label="adaptive exit tolerance"
            ),
            "exit_entropy_threshold": _nonnegative_finite_float(
                adaptive["exit_entropy_threshold"],
                label="adaptive exit entropy threshold",
            ),
            "prediction_stability_patience": adaptive_patience,
            "prediction_stability_tolerance": _nonnegative_finite_float(
                adaptive["confidence_tolerance"],
                label="prediction stability tolerance",
            ),
        }
        expected_adaptive_controls = {
            "max_cycles": expected_max_cycles,
            "exit_tolerance": expected_exit_tolerance,
            "exit_entropy_threshold": expected_exit_entropy,
            "prediction_stability_patience": expected_patience,
            "prediction_stability_tolerance": expected_stability_tolerance,
        }
        if seed_adaptive_configuration != expected_adaptive_controls:
            raise ValueError(
                f"Seed {seed} adaptive controls do not match the configured mode"
            )
        if canonical_adaptive_configuration is None:
            canonical_adaptive_configuration = seed_adaptive_configuration
        elif seed_adaptive_configuration != canonical_adaptive_configuration:
            raise ValueError("All seed results must use the same adaptive controls")

        decision_fidelity = comparison["decision_fidelity"]
        seed_top_k = int(decision_fidelity["top_k"])
        if seed_top_k <= 0:
            raise ValueError(f"Seed {seed} decision top-k must be positive")
        if seed_top_k != expected_decision_top_k:
            raise ValueError(
                f"Seed {seed} decision top-k does not match the configured mode"
            )
        if decision_top_k is None:
            decision_top_k = seed_top_k
        elif seed_top_k != decision_top_k:
            raise ValueError("All seed results must use the same decision top-k")
        top_k_order_disagreements = _bounded_count(
            decision_fidelity["top_k_order_disagreement_count"],
            maximum=request_count,
            label=f"seed {seed} top-k order disagreement count",
        )
        top_k_set_disagreements = _bounded_count(
            decision_fidelity["top_k_set_disagreement_count"],
            maximum=request_count,
            label=f"seed {seed} top-k set disagreement count",
        )
        if top_k_set_disagreements > top_k_order_disagreements:
            raise ValueError(f"Seed {seed} top-k disagreement counts are inconsistent")
        if top1_disagreement_count > top_k_order_disagreements:
            raise ValueError(f"Seed {seed} top1/top-k disagreement counts are inconsistent")
        tensor_equality_required = _strict_bool(
            decision_fidelity.get("tensor_equality_required"),
            label=f"seed {seed} tensor equality flag",
        )
        if tensor_equality_required:
            raise ValueError("Tensor equality must remain explicitly non-gating")

        verified_scope = decision_fidelity["verified_scope"]
        if not _strict_bool(
            verified_scope.get("verified"),
            label=f"seed {seed} verified scope flag",
        ):
            raise ValueError(f"Seed {seed} decision scope is not verified")
        output_class_count = int(verified_scope["output_class_count"])
        scope_class_count = int(verified_scope["class_count"])
        scope_indices = [int(value) for value in verified_scope["class_indices"]]
        expected_scope = _resolve_prediction_scope(
            expected_scope_input,
            output_class_count=output_class_count,
            decision_top_k=expected_decision_top_k,
        )
        expected_scope_mode = (
            "all_output_classes"
            if expected_scope_input is None
            else "prediction_class_indices"
        )
        expected_requested_scope = (
            None if expected_scope_input is None else expected_scope
        )
        if (
            output_class_count <= 0
            or len(scope_indices) != scope_class_count
            or len(set(scope_indices)) != scope_class_count
            or any(value < 0 or value >= output_class_count for value in scope_indices)
            or scope_class_count < seed_top_k
        ):
            raise ValueError(f"Seed {seed} verified scope is invalid")
        if (
            scope_indices != expected_scope
            or verified_scope.get("mode") != expected_scope_mode
        ):
            raise ValueError(
                f"Seed {seed} verified scope does not match the configured class indices"
            )
        requested_scope = verified_scope.get("requested_normalized_class_indices")
        if requested_scope is not None:
            if isinstance(requested_scope, (str, bytes)) or not isinstance(
                requested_scope, Sequence
            ):
                raise ValueError(
                    f"Seed {seed} requested normalized class indices are invalid"
                )
            requested_scope = [int(value) for value in requested_scope]
        if requested_scope != expected_requested_scope:
            raise ValueError(
                f"Seed {seed} requested class indices do not match the configured mode"
            )
        if not _strict_bool(
            verified_scope.get("adaptive_requested_class_indices_verified"),
            label=f"seed {seed} requested class-index verification flag",
        ):
            raise ValueError(
                f"Seed {seed} adaptive requested class indices were not verified"
            )
        truth_in_scope = _bounded_count(
            verified_scope["truth_labels_in_scope"],
            maximum=request_count,
            label=f"seed {seed} truth labels in scope",
        )
        truth_outside_scope = _bounded_count(
            verified_scope["truth_labels_outside_scope"],
            maximum=request_count,
            label=f"seed {seed} truth labels outside scope",
        )
        if truth_in_scope + truth_outside_scope != request_count:
            raise ValueError(f"Seed {seed} scope coverage does not match requests")
        static_scope = {
            "verified": True,
            "mode": str(verified_scope["mode"]),
            "output_class_count": output_class_count,
            "class_count": scope_class_count,
            "class_indices": scope_indices,
            "requested_normalized_class_indices": requested_scope,
            "adaptive_requested_class_indices_verified": True,
            "adaptive_verifier_rank_depth_verified": _strict_bool(
                verified_scope.get("adaptive_verifier_rank_depth_verified"),
                label=f"seed {seed} rank-depth verification flag",
            ),
            "adaptive_verifier_class_scope_verified": _strict_bool(
                verified_scope.get("adaptive_verifier_class_scope_verified"),
                label=f"seed {seed} class-scope verification flag",
            ),
            "observed_prediction_class_counts": [
                int(value)
                for value in verified_scope.get(
                    "observed_prediction_class_counts", []
                )
            ],
            "effective_prediction_rank_depth": int(
                verified_scope["effective_prediction_rank_depth"]
            ),
        }
        if static_scope["mode"] not in {
            "all_output_classes",
            "prediction_class_indices",
        }:
            raise ValueError(f"Seed {seed} verified scope mode is invalid")
        if static_scope["mode"] == "all_output_classes" and scope_indices != list(
            range(output_class_count)
        ):
            raise ValueError(f"Seed {seed} all-class scope is incomplete")
        if not static_scope["adaptive_verifier_rank_depth_verified"]:
            raise ValueError(f"Seed {seed} adaptive verifier scope was not verified")
        if (
            not static_scope["adaptive_verifier_class_scope_verified"]
            or static_scope["observed_prediction_class_counts"]
            != [scope_class_count]
        ):
            raise ValueError(
                f"Seed {seed} adaptive verifier class scope was not verified"
            )
        if canonical_scope is None:
            canonical_scope = static_scope
        elif static_scope != canonical_scope:
            raise ValueError("All seed results must use the same verified scope")

        absolute_logit_delta = decision_fidelity["absolute_logit_delta"]
        logit_delta_mean = _nonnegative_finite_float(
            absolute_logit_delta["mean"], label="mean absolute logit delta"
        )
        logit_delta_max = _nonnegative_finite_float(
            absolute_logit_delta["max"], label="maximum absolute logit delta"
        )
        if logit_delta_mean > logit_delta_max:
            raise ValueError(f"Seed {seed} logit delta summary is inconsistent")
        distribution_distance = decision_fidelity["distribution_distance"]
        js_summary = _validated_scalar_summary(
            distribution_distance["jensen_shannon_divergence_nats"],
            label="Jensen-Shannon divergence",
            upper_bound=math.log(2.0),
        )
        tv_summary = _validated_scalar_summary(
            distribution_distance["total_variation_distance"],
            label="total variation distance",
            upper_bound=1.0,
        )
        distribution_drift = adaptive.get("distribution_drift")
        if not isinstance(distribution_drift, Mapping):
            raise ValueError(f"Seed {seed} distribution-drift evidence is missing")
        if (
            int(distribution_drift.get("top_k", 0))
            != expected_distribution_top_k
            or distribution_drift.get("role") != "shadow_diagnostic_only"
        ):
            raise ValueError(
                f"Seed {seed} distribution controls do not match the configured mode"
            )

        fixed_cycles = _finite_float(fixed["cycles"], label="fixed cycles")
        adaptive_cycles = _nonnegative_finite_float(
            adaptive["total_cycles_used"], label="adaptive cycles"
        )
        if fixed_cycles != float(expected_fixed_cycles):
            raise ValueError(
                f"Seed {seed} fixed cycles do not match the configured fixed budget"
            )
        if not 0.0 < adaptive_cycles <= float(expected_max_cycles * request_count):
            raise ValueError(
                f"Seed {seed} adaptive total cycles are outside the configured budget"
            )
        adaptive_mean_cycles = _nonnegative_finite_float(
            adaptive["mean_cycles_used"], label="adaptive mean cycles"
        )
        if adaptive_mean_cycles <= 0.0 or not math.isclose(
            adaptive_mean_cycles,
            adaptive_cycles / request_count,
            rel_tol=0.0,
            abs_tol=5e-4,
        ):
            raise ValueError(f"Seed {seed} adaptive mean-cycle telemetry is invalid")
        raw_cycle_counts = adaptive.get("cycle_counts")
        if not isinstance(raw_cycle_counts, Mapping) or not raw_cycle_counts:
            raise ValueError(f"Seed {seed} adaptive cycle-count evidence is missing")
        cycle_count_total = 0
        cycle_weighted_total = 0
        for raw_cycle, raw_count in raw_cycle_counts.items():
            cycle_number = _finite_float(
                raw_cycle, label=f"seed {seed} adaptive cycle-count key"
            )
            cycle = int(cycle_number)
            if (
                cycle_number != float(cycle)
                or cycle <= 0
                or cycle > expected_max_cycles
            ):
                raise ValueError(
                    f"Seed {seed} adaptive cycle count exceeds the configured budget"
                )
            count = _bounded_count(
                raw_count,
                maximum=request_count,
                label=f"seed {seed} adaptive cycle {cycle} count",
            )
            if count <= 0:
                raise ValueError(f"Seed {seed} adaptive cycle counts must be positive")
            cycle_count_total += count
            cycle_weighted_total += cycle * count
        if (
            cycle_count_total != request_count
            or float(cycle_weighted_total) != adaptive_cycles
        ):
            raise ValueError(f"Seed {seed} adaptive cycle-count evidence is inconsistent")
        prediction_margin_summary = _validated_scalar_summary(
            adaptive["prediction_margin"]["observed"],
            label="prediction margin",
        )
        decision_margin = adaptive["decision_margin"]
        decision_margin_minimum = _nonnegative_finite_float(
            decision_margin["configured_minimum"],
            label="configured decision margin minimum",
        )
        prediction_margin_minimum = _nonnegative_finite_float(
            adaptive["prediction_margin"]["configured_minimum"],
            label="configured prediction margin minimum",
        )
        if decision_margin_minimum != prediction_margin_minimum:
            raise ValueError(f"Seed {seed} configured margin telemetry is inconsistent")
        if prediction_margin_minimum != expected_margin:
            raise ValueError(
                f"Seed {seed} prediction margin does not match the configured mode"
            )
        if configured_prediction_stability_margin is None:
            configured_prediction_stability_margin = prediction_margin_minimum
        elif prediction_margin_minimum != configured_prediction_stability_margin:
            raise ValueError("All seed results must use the same configured margin")
        seed_rank_depth = int(decision_margin["configured_rank_depth"])
        observed_rank_depths = [int(value) for value in decision_margin["observed_rank_depths"]]
        expected_effective_rank_depth = min(
            seed_rank_depth, max(1, scope_class_count - 1)
        )
        if (
            seed_rank_depth <= 0
            or seed_rank_depth != expected_rank_depth
            or observed_rank_depths != [expected_effective_rank_depth]
            or static_scope["effective_prediction_rank_depth"]
            != expected_effective_rank_depth
        ):
            raise ValueError(f"Seed {seed} prediction rank-depth telemetry is invalid")
        if prediction_rank_depth is None:
            prediction_rank_depth = seed_rank_depth
        elif seed_rank_depth != prediction_rank_depth:
            raise ValueError("All seed results must use the same prediction rank depth")
        decision_margin_summary = _validated_scalar_summary(
            decision_margin["observed"],
            label="decision margin",
        )

        raw_exit_reasons = adaptive.get("exit_reasons")
        if not isinstance(raw_exit_reasons, Mapping) or not raw_exit_reasons:
            raise ValueError(f"Seed {seed} adaptive exit-reason evidence is missing")
        exit_reason_counts: Dict[str, int] = {}
        for raw_reason, raw_count in raw_exit_reasons.items():
            if not isinstance(raw_reason, str) or not raw_reason.strip():
                raise ValueError(f"Seed {seed} adaptive exit reason is invalid")
            reason = raw_reason.strip()
            if reason not in KNOWN_ADAPTIVE_EXIT_REASONS:
                raise ValueError(f"Seed {seed} adaptive exit reason is unknown: {reason}")
            count = _bounded_count(
                raw_count,
                maximum=request_count,
                label=f"seed {seed} adaptive exit reason {reason} count",
            )
            if count <= 0:
                raise ValueError(
                    f"Seed {seed} adaptive exit-reason counts must be positive"
                )
            exit_reason_counts[reason] = count
        if sum(exit_reason_counts.values()) != request_count:
            raise ValueError(
                f"Seed {seed} adaptive exit-reason evidence does not cover every request"
            )
        stable_count = exit_reason_counts.get("prediction_stable", 0)
        stable_margin_evidence = decision_margin.get("prediction_stable_observed")
        if not isinstance(stable_margin_evidence, Mapping):
            raise ValueError(
                f"Seed {seed} prediction-stable margin evidence is missing"
            )
        reported_stable_count = _bounded_count(
            stable_margin_evidence.get("observation_count"),
            maximum=request_count,
            label=f"seed {seed} prediction-stable margin observation count",
        )
        if reported_stable_count != stable_count:
            raise ValueError(
                f"Seed {seed} prediction-stable margin coverage is inconsistent"
            )
        stable_margin_summary = stable_margin_evidence.get("summary")
        if stable_count:
            stable_margin_minimum = _nonnegative_finite_float(
                stable_margin_evidence.get("minimum"),
                label=f"seed {seed} prediction-stable decision margin minimum",
            )
            if not isinstance(stable_margin_summary, Mapping):
                raise ValueError(
                    f"Seed {seed} prediction-stable margin summary is missing"
                )
            validated_stable_margin = _validated_scalar_summary(
                stable_margin_summary,
                label="prediction-stable decision margin",
            )
            if (
                stable_margin_minimum < expected_margin
                or validated_stable_margin["min"] < expected_margin
            ):
                raise ValueError(
                    f"Seed {seed} prediction-stable decision margin is below the configured floor"
                )
        elif (
            stable_margin_evidence.get("minimum") is not None
            or stable_margin_summary is not None
        ):
            raise ValueError(
                f"Seed {seed} prediction-stable margin summary has no matching exits"
            )

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
        if fixed_mean_latency <= 0.0 or fixed_total_latency <= 0.0:
            raise ValueError(f"Seed {seed} fixed latency must be positive")
        if adaptive_mean_latency <= 0.0 or adaptive_total_latency <= 0.0:
            raise ValueError(f"Seed {seed} adaptive latency must be positive")
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
        total_top1_disagreements += top1_disagreement_count
        total_top_k_order_disagreements += top_k_order_disagreements
        total_top_k_set_disagreements += top_k_set_disagreements
        total_fixed_cycles += fixed_cycles * request_count
        total_adaptive_cycles += adaptive_cycles
        total_fixed_latency_ms += fixed_total_latency
        total_adaptive_latency_ms += adaptive_total_latency
        latency_reductions.append(latency_reduction)
        minimum_observed_prediction_margin = min(
            minimum_observed_prediction_margin, prediction_margin_summary["min"]
        )
        minimum_observed_decision_margin = min(
            minimum_observed_decision_margin, decision_margin_summary["min"]
        )
        total_truth_in_scope += truth_in_scope
        total_truth_outside_scope += truth_outside_scope
        seed_logit_value_count = request_count * scope_class_count
        logit_delta_weighted_sum += logit_delta_mean * seed_logit_value_count
        logit_delta_value_count += seed_logit_value_count
        maximum_absolute_logit_delta = max(maximum_absolute_logit_delta, logit_delta_max)
        js_weighted_sum += js_summary["mean"] * request_count
        js_minimum = min(js_minimum, js_summary["min"])
        js_maximum = max(js_maximum, js_summary["max"])
        tv_weighted_sum += tv_summary["mean"] * request_count
        tv_minimum = min(tv_minimum, tv_summary["min"])
        tv_maximum = max(tv_maximum, tv_summary["max"])
        per_seed.append(
            {
                "seed": seed,
                "samples": request_count,
                "fixed_correct": fixed_correct,
                "adaptive_correct": adaptive_correct,
                "accuracy_delta": round(accuracy_delta, 12),
                "top1_disagreement_count": top1_disagreement_count,
                "exact_disagreement_count": top1_disagreement_count,
                "top_k_order_disagreement_count": top_k_order_disagreements,
                "top_k_set_disagreement_count": top_k_set_disagreements,
                "minimum_observed_prediction_margin": round(
                    prediction_margin_summary["min"], 8
                ),
                "minimum_observed_decision_margin": round(
                    decision_margin_summary["min"], 8
                ),
                "maximum_absolute_logit_delta": round(logit_delta_max, 12),
                "mean_absolute_logit_delta": round(logit_delta_mean, 12),
                "mean_jensen_shannon_divergence_nats": round(js_summary["mean"], 12),
                "mean_total_variation_distance": round(tv_summary["mean"], 12),
                "cycle_reduction_percent": round(cycle_reduction, 6),
                "fixed_mean_latency_ms": round(
                    fixed_total_latency / request_count, 6
                ),
                "adaptive_mean_latency_ms": round(
                    adaptive_total_latency / request_count, 6
                ),
                "mean_latency_reduction_percent": round(latency_reduction, 6),
                "measurement_order": comparison["measurement_order"],
            }
        )

    assert decision_top_k is not None
    assert prediction_rank_depth is not None
    assert configured_prediction_stability_margin is not None
    assert canonical_adaptive_configuration is not None
    assert canonical_scope is not None
    if prediction_rank_depth < decision_top_k:
        raise ValueError(
            "prediction rank depth must cover the gated decision top-k boundary"
        )
    if canonical_scope["effective_prediction_rank_depth"] < decision_top_k:
        raise ValueError(
            "verified scope needs at least decision_top_k + 1 classes for the boundary gate"
        )
    if total_fixed_cycles <= 0.0:
        raise ValueError("Total fixed-cycle budget must be positive")
    if total_fixed_latency_ms <= 0.0:
        raise ValueError("Total fixed latency must be positive")
    if logit_delta_value_count <= 0:
        raise ValueError("Decision-fidelity logit coverage must be positive")

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
        "zero_top1_disagreements": {
            "passed": total_top1_disagreements == 0,
            "observed": total_top1_disagreements,
            "required": 0,
        },
        "zero_top_k_order_disagreements": {
            "passed": total_top_k_order_disagreements == 0,
            "top_k": decision_top_k,
            "observed": total_top_k_order_disagreements,
            "required": 0,
        },
        "zero_top_k_set_disagreements": {
            "passed": total_top_k_set_disagreements == 0,
            "top_k": decision_top_k,
            "observed": total_top_k_set_disagreements,
            "required": 0,
        },
        # Compatibility alias retained for consumers of the v1 gate artifact.
        "zero_prediction_disagreements": {
            "passed": total_top1_disagreements == 0,
            "observed": total_top1_disagreements,
            "required": 0,
            "alias_of": "zero_top1_disagreements",
        },
        "no_negative_per_seed_accuracy_delta": {
            "passed": not negative_accuracy_seeds,
            "negative_seed_ids": negative_accuracy_seeds,
            "required": "adaptive_correct >= fixed_correct for every seed",
        },
        "minimum_weighted_cycle_reduction": {
            "passed": weighted_cycle_reduction >= minimum_cycle_reduction,
            "observed_percent": round(weighted_cycle_reduction, 6),
            "required_minimum_percent": minimum_cycle_reduction,
        },
        "positive_median_per_seed_latency_reduction": {
            "passed": median_seed_latency_reduction > minimum_median_latency_reduction,
            "observed_percent": round(median_seed_latency_reduction, 6),
            "required_greater_than_percent": float(minimum_median_latency_reduction),
        },
    }

    verified_scope_summary = {
        **canonical_scope,
        "truth_labels_in_scope": total_truth_in_scope,
        "truth_labels_outside_scope": total_truth_outside_scope,
    }
    return {
        "summary": {
            "seed_count": len(seed_results),
            "total_samples": total_requests,
            "top1_disagreement_count": total_top1_disagreements,
            "exact_disagreement_count": total_top1_disagreements,
            "decision_fidelity": {
                "top_k": decision_top_k,
                "top_k_order_disagreement_count": total_top_k_order_disagreements,
                "top_k_set_disagreement_count": total_top_k_set_disagreements,
                "verified_scope": verified_scope_summary,
                "absolute_logit_delta": {
                    "mean": round(
                        logit_delta_weighted_sum / logit_delta_value_count, 12
                    ),
                    "max": round(maximum_absolute_logit_delta, 12),
                    "role": "diagnostic_only_not_gated",
                },
                "distribution_distance": {
                    "jensen_shannon_divergence_nats": {
                        "min": round(js_minimum, 12),
                        "mean": round(js_weighted_sum / total_requests, 12),
                        "max": round(js_maximum, 12),
                    },
                    "total_variation_distance": {
                        "min": round(tv_minimum, 12),
                        "mean": round(tv_weighted_sum / total_requests, 12),
                        "max": round(tv_maximum, 12),
                    },
                    "role": "diagnostic_only_not_gated",
                },
                "tensor_equality_required": False,
            },
            "prediction_stability_rank_depth": prediction_rank_depth,
            "configured_fixed_cycles": expected_fixed_cycles,
            "configured_max_cycles": expected_max_cycles,
            "distribution_top_k": expected_distribution_top_k,
            "configured_prediction_stability_margin": (
                configured_prediction_stability_margin
            ),
            "adaptive_configuration": {
                **canonical_adaptive_configuration,
                "prediction_stability_margin": (
                    configured_prediction_stability_margin
                ),
                "prediction_stability_rank_depth": prediction_rank_depth,
                "distribution_top_k": expected_distribution_top_k,
            },
            "weighted_cycle_reduction_percent": round(weighted_cycle_reduction, 6),
            "median_per_seed_latency_reduction_percent": round(
                median_seed_latency_reduction, 6
            ),
            "weighted_mean_latency_reduction_percent": round(
                weighted_latency_reduction, 6
            ),
            "weighted_fixed_mean_latency_ms": round(weighted_fixed_mean_latency, 6),
            "weighted_adaptive_mean_latency_ms": round(
                weighted_adaptive_mean_latency, 6
            ),
            "total_fixed_cycle_budget": round(total_fixed_cycles, 6),
            "total_adaptive_cycles_used": round(total_adaptive_cycles, 6),
            "minimum_observed_prediction_margin": round(
                minimum_observed_prediction_margin, 8
            ),
            "minimum_observed_decision_margin": round(
                minimum_observed_decision_margin, 8
            ),
            "per_seed_accuracy_deltas": [
                {"seed": row["seed"], "accuracy_delta": row["accuracy_delta"]}
                for row in per_seed
            ],
        },
        "per_seed_gate_metrics": per_seed,
        "gates": {
            "passed": all(bool(check["passed"]) for check in checks.values()),
            "tensor_equality_required": False,
            "checks": checks,
        },
    }


def _validate_mode_aggregate(
    aggregate: Mapping[str, Any],
    *,
    fixed_cycles: int,
    max_cycles: int,
    prediction_class_indices: List[int] | List[bool] | None,
    decision_top_k: int,
    prediction_stability_rank_depth: int,
    prediction_stability_margin: float,
    prediction_stability_patience: int,
    prediction_stability_tol: float,
    exit_tol: float,
    exit_entropy_threshold: float,
    distribution_top_k: int,
) -> None:
    summary = aggregate["summary"]
    fidelity = summary["decision_fidelity"]
    verified_scope = fidelity["verified_scope"]
    expected_scope = _resolve_prediction_scope(
        prediction_class_indices,
        output_class_count=int(verified_scope["output_class_count"]),
        decision_top_k=decision_top_k,
    )
    expected_scope_mode = (
        "all_output_classes"
        if prediction_class_indices is None
        else "prediction_class_indices"
    )
    expected_adaptive_configuration = {
        "max_cycles": max_cycles,
        "exit_tolerance": exit_tol,
        "exit_entropy_threshold": exit_entropy_threshold,
        "prediction_stability_patience": prediction_stability_patience,
        "prediction_stability_tolerance": prediction_stability_tol,
        "prediction_stability_margin": prediction_stability_margin,
        "prediction_stability_rank_depth": prediction_stability_rank_depth,
        "distribution_top_k": distribution_top_k,
    }
    if (
        verified_scope["class_indices"] != expected_scope
        or verified_scope["mode"] != expected_scope_mode
        or fidelity["top_k"] != decision_top_k
        or summary["configured_fixed_cycles"] != fixed_cycles
        or summary["configured_max_cycles"] != max_cycles
        or summary["distribution_top_k"] != distribution_top_k
        or summary["prediction_stability_rank_depth"]
        != prediction_stability_rank_depth
        or summary["configured_prediction_stability_margin"]
        != prediction_stability_margin
        or summary["adaptive_configuration"] != expected_adaptive_configuration
    ):
        raise ValueError("Benchmark results do not match the configured decision mode")


def run_gate(
    *,
    weights: Path,
    seeds: Sequence[int],
    samples_per_seed: int,
    device: Any,
    device_info: Mapping[str, Any],
    metadata: Path = DEFAULT_META,
    fixed_cycles: int = RELEASE_FIXED_CYCLES,
    max_cycles: int = RELEASE_MAX_CYCLES,
    stability_patience: int = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    stability_tol: float = DEFAULT_PREDICTION_STABILITY_TOL,
    prediction_stability_margin: float = DEFAULT_PREDICTION_STABILITY_MARGIN,
    prediction_stability_rank_depth: int = DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    decision_top_k: int = DEFAULT_DECISION_TOP_K,
    prediction_class_indices: Any = None,
    distribution_top_k: int = 5,
    model_loader: Callable[[Path, Any], Any] | None = None,
    task_factory: Callable[..., Any] | None = None,
    benchmark_fn: Callable[..., Dict[str, Any]] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate isolated-verifier and deployed-runtime decision fidelity."""

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
    normalized_margin = _finite_float(
        prediction_stability_margin,
        label="prediction stability margin",
    )
    normalized_distribution_top_k = int(distribution_top_k)
    normalized_decision_top_k = int(decision_top_k)
    normalized_rank_depth = int(prediction_stability_rank_depth)
    normalized_scope = _normalize_prediction_class_indices_input(
        prediction_class_indices
    )
    if normalized_fixed_cycles <= 0 or normalized_max_cycles <= 0:
        raise ValueError("fixed_cycles and max_cycles must be positive")
    if normalized_patience < 0:
        raise ValueError("stability_patience must be nonnegative")
    if normalized_tolerance < 0.0:
        raise ValueError("stability_tol must be nonnegative")
    if normalized_margin < 0.0:
        raise ValueError("prediction_stability_margin must be nonnegative")
    if normalized_distribution_top_k <= 0:
        raise ValueError("distribution_top_k must be positive")
    if normalized_decision_top_k <= 0:
        raise ValueError("decision_top_k must be positive")
    if normalized_rank_depth <= 0:
        raise ValueError("prediction_stability_rank_depth must be positive")
    if normalized_rank_depth < normalized_decision_top_k:
        raise ValueError(
            "prediction_stability_rank_depth must be at least decision_top_k"
        )

    release_scope, metadata_record = load_release_prediction_scope(metadata)
    release_fixed_cycles = int(RELEASE_FIXED_CYCLES)
    release_max_cycles = int(RELEASE_MAX_CYCLES)
    release_patience = int(DEFAULT_PREDICTION_STABILITY_PATIENCE)
    release_tolerance = _nonnegative_finite_float(
        DEFAULT_PREDICTION_STABILITY_TOL,
        label="release prediction stability tolerance",
    )
    release_margin = _nonnegative_finite_float(
        DEFAULT_PREDICTION_STABILITY_MARGIN,
        label="release prediction stability margin",
    )
    release_rank_depth = int(DEFAULT_PREDICTION_STABILITY_RANK_DEPTH)
    release_decision_top_k = int(DEFAULT_DECISION_TOP_K)
    release_exit_tol = _nonnegative_finite_float(
        DEFAULT_ADAPTIVE_EXIT_TOL,
        label="release adaptive exit tolerance",
    )
    release_exit_entropy = _nonnegative_finite_float(
        DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        label="release adaptive exit entropy threshold",
    )
    release_distribution_top_k = int(
        chat_app.DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K
    )
    if release_rank_depth < release_decision_top_k:
        raise ValueError("Release rank depth does not cover the decision top-k")
    if len(release_scope) <= release_decision_top_k:
        raise ValueError(
            "Release metadata scope needs at least decision_top_k + 1 allowed classes"
        )

    loader = model_loader or _load_model
    make_task = task_factory or make_chained_task
    benchmark = benchmark_fn or benchmark_serving_requests
    checkpoint_sha256 = _sha256_file(weights)
    model = loader(weights, device)
    isolated_seed_results: List[Dict[str, Any]] = []
    release_seed_results: List[Dict[str, Any]] = []
    seed_results: List[Dict[str, Any]] = []

    for seed_index, seed in enumerate(normalized_seeds):
        x, y = make_task(int(samples_per_seed), seed=seed)
        x = x.to(device)
        y = y.to(device)
        isolated_metrics = benchmark(
            model,
            x,
            y,
            fixed_cycles=normalized_fixed_cycles,
            max_cycles=normalized_max_cycles,
            stability_patience=normalized_patience,
            stability_tol=normalized_tolerance,
            prediction_stability_margin=normalized_margin,
            prediction_stability_rank_depth=normalized_rank_depth,
            decision_top_k=normalized_decision_top_k,
            prediction_class_indices=normalized_scope,
            distribution_top_k=normalized_distribution_top_k,
            exit_tol=ISOLATED_VERIFIER_EXIT_TOL,
            exit_entropy_threshold=ISOLATED_VERIFIER_EXIT_ENTROPY,
            order_offset=seed_index % 2,
        )
        release_metrics = benchmark(
            model,
            x,
            y,
            fixed_cycles=release_fixed_cycles,
            max_cycles=release_max_cycles,
            stability_patience=release_patience,
            stability_tol=release_tolerance,
            prediction_stability_margin=release_margin,
            prediction_stability_rank_depth=release_rank_depth,
            decision_top_k=release_decision_top_k,
            prediction_class_indices=release_scope,
            distribution_top_k=release_distribution_top_k,
            exit_tol=release_exit_tol,
            exit_entropy_threshold=release_exit_entropy,
            order_offset=seed_index % 2,
        )
        isolated_row = {"seed": seed, "metrics": isolated_metrics}
        release_row = {"seed": seed, "metrics": release_metrics}
        isolated_seed_results.append(isolated_row)
        release_seed_results.append(release_row)
        seed_results.append(
            {
                "seed": seed,
                # Compatibility alias now points at deployed-runtime evidence.
                "metrics": release_metrics,
                "isolated_verifier": isolated_metrics,
                "release_runtime": release_metrics,
            }
        )

    isolated_aggregate = aggregate_gate_results(
        isolated_seed_results,
        fixed_cycles=normalized_fixed_cycles,
        max_cycles=normalized_max_cycles,
        prediction_class_indices=normalized_scope,
        decision_top_k=normalized_decision_top_k,
        prediction_stability_rank_depth=normalized_rank_depth,
        prediction_stability_margin=normalized_margin,
        prediction_stability_patience=normalized_patience,
        prediction_stability_tol=normalized_tolerance,
        exit_tol=ISOLATED_VERIFIER_EXIT_TOL,
        exit_entropy_threshold=ISOLATED_VERIFIER_EXIT_ENTROPY,
        distribution_top_k=normalized_distribution_top_k,
    )
    release_aggregate = aggregate_gate_results(
        release_seed_results,
        fixed_cycles=release_fixed_cycles,
        max_cycles=release_max_cycles,
        prediction_class_indices=release_scope,
        decision_top_k=release_decision_top_k,
        prediction_stability_rank_depth=release_rank_depth,
        prediction_stability_margin=release_margin,
        prediction_stability_patience=release_patience,
        prediction_stability_tol=release_tolerance,
        exit_tol=release_exit_tol,
        exit_entropy_threshold=release_exit_entropy,
        distribution_top_k=release_distribution_top_k,
    )
    _validate_mode_aggregate(
        isolated_aggregate,
        fixed_cycles=normalized_fixed_cycles,
        max_cycles=normalized_max_cycles,
        prediction_class_indices=normalized_scope,
        decision_top_k=normalized_decision_top_k,
        prediction_stability_rank_depth=normalized_rank_depth,
        prediction_stability_margin=normalized_margin,
        prediction_stability_patience=normalized_patience,
        prediction_stability_tol=normalized_tolerance,
        exit_tol=ISOLATED_VERIFIER_EXIT_TOL,
        exit_entropy_threshold=ISOLATED_VERIFIER_EXIT_ENTROPY,
        distribution_top_k=normalized_distribution_top_k,
    )
    _validate_mode_aggregate(
        release_aggregate,
        fixed_cycles=release_fixed_cycles,
        max_cycles=release_max_cycles,
        prediction_class_indices=release_scope,
        decision_top_k=release_decision_top_k,
        prediction_stability_rank_depth=release_rank_depth,
        prediction_stability_margin=release_margin,
        prediction_stability_patience=release_patience,
        prediction_stability_tol=release_tolerance,
        exit_tol=release_exit_tol,
        exit_entropy_threshold=release_exit_entropy,
        distribution_top_k=release_distribution_top_k,
    )

    combined_checks = {
        key: {**dict(value), "mode": "release_runtime"}
        for key, value in release_aggregate["gates"]["checks"].items()
    }
    combined_checks.update(
        {
            f"isolated_verifier__{key}": {
                **dict(value),
                "mode": "isolated_verifier",
            }
            for key, value in isolated_aggregate["gates"]["checks"].items()
        }
    )
    combined_passed = bool(release_aggregate["gates"]["passed"]) and bool(
        isolated_aggregate["gates"]["passed"]
    )
    summary = {
        **release_aggregate["summary"],
        "mode_summaries": {
            "isolated_verifier": isolated_aggregate["summary"],
            "release_runtime": release_aggregate["summary"],
        },
    }
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(weights),
            "sha256": checkpoint_sha256,
        },
        "metadata": metadata_record,
        "configuration": {
            "seeds": normalized_seeds,
            "samples_per_seed": int(samples_per_seed),
            "total_samples": int(samples_per_seed) * len(normalized_seeds),
            "fixed_cycles": release_fixed_cycles,
            "max_cycles": release_max_cycles,
            "stability_patience": release_patience,
            "stability_tolerance": release_tolerance,
            "prediction_stability_margin": release_margin,
            "prediction_stability_rank_depth": release_rank_depth,
            "decision_top_k": release_decision_top_k,
            "prediction_class_indices": release_scope,
            "prediction_scope_orchestration": "single_scope",
            "distribution_top_k": release_distribution_top_k,
            "counterbalance": "alternate_per_request_with_alternating_seed_offset",
            "modes": {
                "isolated_verifier": {
                    "role": "component_isolation",
                    "fixed_cycles": normalized_fixed_cycles,
                    "max_cycles": normalized_max_cycles,
                    "exit_tolerance": ISOLATED_VERIFIER_EXIT_TOL,
                    "exit_entropy_threshold": ISOLATED_VERIFIER_EXIT_ENTROPY,
                    "prediction_stability_patience": normalized_patience,
                    "prediction_stability_tolerance": normalized_tolerance,
                    "prediction_stability_margin": normalized_margin,
                    "prediction_stability_rank_depth": normalized_rank_depth,
                    "decision_top_k": normalized_decision_top_k,
                    "prediction_class_indices": normalized_scope,
                    "distribution_top_k": normalized_distribution_top_k,
                },
                "release_runtime": {
                    "role": "enforced_deployed_runtime_configuration",
                    "fixed_cycles": release_fixed_cycles,
                    "max_cycles": release_max_cycles,
                    "exit_tolerance": release_exit_tol,
                    "exit_entropy_threshold": release_exit_entropy,
                    "prediction_stability_patience": release_patience,
                    "prediction_stability_tolerance": release_tolerance,
                    "prediction_stability_margin": release_margin,
                    "prediction_stability_rank_depth": release_rank_depth,
                    "decision_top_k": release_decision_top_k,
                    "prediction_class_indices": release_scope,
                    "prediction_class_scope_source": metadata_record[
                        "scope_source"
                    ],
                    "distribution_top_k": release_distribution_top_k,
                },
            },
        },
        "provenance": dict(provenance)
        if provenance is not None
        else collect_provenance(device=device, device_info=device_info),
        "summary": summary,
        "per_seed_gate_metrics": release_aggregate["per_seed_gate_metrics"],
        "per_mode_gate_metrics": {
            "isolated_verifier": isolated_aggregate["per_seed_gate_metrics"],
            "release_runtime": release_aggregate["per_seed_gate_metrics"],
        },
        "gates": {
            "passed": combined_passed,
            "tensor_equality_required": False,
            "mode_passed": {
                "isolated_verifier": bool(isolated_aggregate["gates"]["passed"]),
                "release_runtime": bool(release_aggregate["gates"]["passed"]),
            },
            "checks": combined_checks,
        },
        "mode_results": {
            "isolated_verifier": isolated_aggregate,
            "release_runtime": release_aggregate,
        },
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
    parser.add_argument("--meta", "--metadata", dest="metadata", default=str(DEFAULT_META))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--samples-per-seed",
        "--samples_per_seed",
        dest="samples_per_seed",
        type=int,
        default=DEFAULT_SAMPLES_PER_SEED,
    )
    parser.add_argument(
        "--fixed-cycles", "--fixed_cycles", type=int, default=RELEASE_FIXED_CYCLES
    )
    parser.add_argument(
        "--max-cycles", "--max_cycles", type=int, default=RELEASE_MAX_CYCLES
    )
    parser.add_argument(
        "--stability-patience",
        "--stability_patience",
        type=int,
        default=DEFAULT_PREDICTION_STABILITY_PATIENCE,
    )
    parser.add_argument(
        "--stability-tol",
        "--stability_tol",
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
        help="Use one verified class scope across all seeds (no multi-scope orchestration).",
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
        metadata=Path(args.metadata),
        seeds=args.seeds,
        samples_per_seed=int(args.samples_per_seed),
        device=device,
        device_info=device_info,
        fixed_cycles=int(args.fixed_cycles),
        max_cycles=int(args.max_cycles),
        stability_patience=int(args.stability_patience),
        stability_tol=float(args.stability_tol),
        prediction_stability_margin=float(args.prediction_stability_margin),
        prediction_stability_rank_depth=int(args.prediction_stability_rank_depth),
        decision_top_k=int(args.decision_top_k),
        prediction_class_indices=args.prediction_class_indices,
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
