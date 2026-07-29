"""Gate v51 adaptive compute against fixed-cycle chat response fidelity.

This is deliberately a small, frozen release prompt matrix exercised through
the real ``source/chat_web_app.py`` Engine.  It is a release-regression check,
not a universal quality evaluation and not a claim about held-out prompts.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
PROJECT_ROOT = SOURCE_DIR.parent

import chat_app
from chat_web_app import Engine
from device_utils import configure_torch_runtime, resolve_device


DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "output" / "benchmark_v51_cognitive_leap_ultra_latest"
)
DEFAULT_WEIGHTS = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
DEFAULT_META = DEFAULT_ARTIFACT_DIR / "chat_demo_meta.json"
DEFAULT_OUTPUT = DEFAULT_ARTIFACT_DIR / "chat_response_fidelity_gate.json"

SCHEMA = "v51-chat-response-fidelity-gate-v1"
FIXED_CYCLES = 3
ADAPTIVE_MAX_CYCLES = 8
REQUIRED_RELEASE_DECISION_REFERENCE_CYCLES = 3
PREDICTION_STABILITY_MARGIN = chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
# Immutable checkpoint/workload-calibrated release literal; not a universal margin.
REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN = 5e-4
AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH = int(
    chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
)
REQUIRED_RELEASE_PREDICTION_STABILITY_RANK_DEPTH = 3
# Immutable release literals: these deliberately do not derive from chat_app so
# a coordinated library/default drift cannot redefine what "v51 release" means.
AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS = MappingProxyType(
    {
        "adaptive_exit_tol": 0.001,
        "adaptive_exit_entropy": 0.2,
        "prediction_stability_patience": 2.0,
        "prediction_stability_tol": 0.005,
    }
)
CANONICAL_RELEASE_MODEL_SIZE = "cognitive_leap_ultra_expert"
CANONICAL_RELEASE_FEATURE_MODE = "context_mix_v4"
CANONICAL_RELEASE_CLASS_COUNT = 10
CANONICAL_RELEASE_WEIGHTS_SHA256 = (
    "664b1779452fe1482389413004d8bce3369f6d8ee15ab8c2c891dc5e382ebae4"
)
CANONICAL_RELEASE_METADATA_SHA256 = (
    "7134c82c96204a9aa8b255642b9b4b1fb84e7e44dbab1c69327fb66838c47f50"
)
TOP_CANDIDATE_COUNT = 5
KNOWN_ADAPTIVE_EXIT_REASONS = frozenset(
    {
        "prediction_stable",
        "decision_reference_budget",
        "latent_converged",
        "low_entropy",
        "halt_mass",
        "max_cycles",
    }
)

if PREDICTION_STABILITY_MARGIN != REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN:
    raise RuntimeError(
        "The authoritative runtime prediction-stability margin is not the "
        "v51 checkpoint/workload-calibrated release value 0.0005"
    )
if FIXED_CYCLES != REQUIRED_RELEASE_DECISION_REFERENCE_CYCLES:
    raise RuntimeError(
        "The fixed release budget does not match the trained decision reference"
    )
if (
    AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH
    != REQUIRED_RELEASE_PREDICTION_STABILITY_RANK_DEPTH
):
    raise RuntimeError(
        "The authoritative runtime prediction-stability rank depth is not the "
        "v51 release value 3"
    )

CANONICAL_RELEASE_PROMPT_MATRIX_SHA256 = (
    "652d352a6f7be0481b422925ce50f80e4c793711a1597e92f8b48eb8548f17ba"
)
CANONICAL_RELEASE_PROMPT_MATRIX_COUNT = 16
CANONICAL_RELEASE_PROMPT_MATRIX_CATEGORIES = {
    "artifacts": 1,
    "comparison": 1,
    "debugging": 1,
    "planning": 1,
    "procedure": 1,
    "robustness": 2,
    "runtime": 2,
    "scope": 2,
    "style": 2,
    "training": 1,
    "uncertainty": 2,
}

DEFAULT_SOURCE_PACKAGE_PARITY_PAIRS = (
    (
        "chat_app",
        SOURCE_DIR / "chat_app.py",
        PROJECT_ROOT / "runtime_python" / "chat_app.py",
    ),
    (
        "interaction_planner",
        SOURCE_DIR / "interaction_planner.py",
        PROJECT_ROOT / "runtime_python" / "interaction_planner.py",
    ),
    (
        "model_variants",
        SOURCE_DIR / "model_variants.py",
        PROJECT_ROOT / "runtime_python" / "model_variants.py",
    ),
)

SURFACE_SPECIFIC_RUNTIME_HASH_PAIRS = (
    (
        "chat_web_app",
        SOURCE_DIR / "chat_web_app.py",
        PROJECT_ROOT / "runtime_python" / "chat_web_app.py",
    ),
    (
        "chat_pipeline",
        SOURCE_DIR / "chat_pipeline.py",
        PROJECT_ROOT / "runtime_python" / "chat_pipeline.py",
    ),
)


# These cases are intentionally checked in and hashed.  They cover the demo's
# supported product surface without pretending to be a statistical sample of
# general chat traffic.
FROZEN_RELEASE_PROMPT_MATRIX: tuple[Mapping[str, str], ...] = (
    {
        "id": "checkpoint-scope",
        "category": "scope",
        "prompt": "What is this v51 checkpoint designed to demonstrate?",
    },
    {
        "id": "limitations",
        "category": "scope",
        "prompt": "State the most important limitation of this chat demo.",
    },
    {
        "id": "training-task",
        "category": "training",
        "prompt": "Explain the synthetic chained modular arithmetic task.",
    },
    {
        "id": "runtime-controls",
        "category": "runtime",
        "prompt": "Which runtime compute controls can I compare?",
    },
    {
        "id": "adaptive-compute",
        "category": "runtime",
        "prompt": "How should I interpret adaptive cycles used and exit reason?",
    },
    {
        "id": "load-test",
        "category": "procedure",
        "prompt": "Give me a short checklist for testing the model in the web chat.",
    },
    {
        "id": "candidate-debug",
        "category": "debugging",
        "prompt": "A response looks wrong. What candidate and timing diagnostics should I inspect?",
    },
    {
        "id": "reload-artifacts",
        "category": "artifacts",
        "prompt": "Which saved artifacts show that training and checkpoint reload completed?",
    },
    {
        "id": "uncertainty",
        "category": "uncertainty",
        "prompt": "How confident should I be in answers to broad chat questions?",
    },
    {
        "id": "next-step",
        "category": "planning",
        "prompt": "What is the next training step toward a real chat model?",
    },
    {
        "id": "concise-mode",
        "category": "style",
        "prompt": "In one sentence, summarize the v51 demo status.",
    },
    {
        "id": "technical-mode",
        "category": "style",
        "prompt": "Give a technical analyst readout of the v51 ultra architecture.",
    },
    {
        "id": "punctuation-case",
        "category": "robustness",
        "prompt": "RUNTIME COMPUTE: what do cycles, entropy, and stability mean?",
    },
    {
        "id": "direct-question",
        "category": "robustness",
        "prompt": "Is this a polished general-purpose assistant? Why or why not?",
    },
    {
        "id": "comparison-request",
        "category": "comparison",
        "prompt": "Compare fixed-cycle inference with adaptive early exit for this demo.",
    },
    {
        "id": "safe-interpretation",
        "category": "uncertainty",
        "prompt": "How can I avoid over-interpreting a successful demo response?",
    },
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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


def _finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got {value!r}")
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _optional_finite_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, label=label)


def _strict_bool(value: Any, *, label: str) -> bool:
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if type(value) is not bool:
        raise ValueError(f"{label} must be a boolean, got {value!r}")
    return value


def _integer_sequence(value: Any, *, label: str) -> List[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an integer array")
    normalized: List[int] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(f"{label}[{index}] must be an integer")
        normalized.append(int(raw))
    return normalized


def normalize_prompt_matrix(rows: Sequence[Any]) -> List[Dict[str, str]]:
    """Normalize and validate a frozen prompt matrix before hashing or running."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("Prompt matrix must be a JSON array")
    if not rows:
        raise ValueError("Prompt matrix must contain at least one prompt")
    if len(rows) > 1024:
        raise ValueError("Prompt matrix cannot contain more than 1024 prompts")

    normalized: List[Dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    for index, row in enumerate(rows):
        if isinstance(row, str):
            prompt_id = f"custom-{index + 1:03d}"
            category = "custom"
            prompt = row.strip()
        elif isinstance(row, Mapping):
            prompt_id = str(row.get("id") or f"custom-{index + 1:03d}").strip()
            category = str(row.get("category") or "custom").strip()
            prompt = str(row.get("prompt", row.get("text", ""))).strip()
        else:
            raise ValueError(f"Prompt row {index} must be a string or object")
        if not prompt_id:
            raise ValueError(f"Prompt row {index} has an empty id")
        if not category:
            raise ValueError(f"Prompt row {index} has an empty category")
        if not prompt:
            raise ValueError(f"Prompt row {index} has empty text")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        if prompt in seen_prompts:
            raise ValueError(f"Duplicate prompt text: {prompt!r}")
        seen_ids.add(prompt_id)
        seen_prompts.add(prompt)
        normalized.append(
            {"id": prompt_id, "category": category, "prompt": prompt}
        )
    return normalized


def load_prompt_matrix(path: Path | None) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    if path is None:
        prompts = normalize_prompt_matrix(FROZEN_RELEASE_PROMPT_MATRIX)
        return prompts, {
            "origin": "builtin",
            "path": None,
            "source_file_sha256": None,
        }

    prompt_path = Path(path).expanduser().resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt JSON not found: {prompt_path}")
    raw = json.loads(prompt_path.read_text(encoding="utf-8"))
    rows = raw.get("prompts") if isinstance(raw, Mapping) else raw
    prompts = normalize_prompt_matrix(rows)
    return prompts, {
        "origin": "json_file",
        "path": str(prompt_path),
        "source_file_sha256": _sha256_file(prompt_path),
    }


def prompt_matrix_sha256(prompts: Sequence[Mapping[str, str]]) -> str:
    canonical = json.dumps(
        list(prompts),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _prompt_matrix_release_observation(
    prompts: Sequence[Mapping[str, str]],
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_hash = prompt_matrix_sha256(prompts)
    observed_count = len(prompts)
    observed_categories = dict(
        sorted(Counter(row["category"] for row in prompts).items())
    )
    origin = source.get("origin")
    eligible = (
        origin == "builtin"
        and observed_hash == CANONICAL_RELEASE_PROMPT_MATRIX_SHA256
        and observed_count == CANONICAL_RELEASE_PROMPT_MATRIX_COUNT
        and observed_categories == CANONICAL_RELEASE_PROMPT_MATRIX_CATEGORIES
    )
    return {
        "release_eligible": eligible,
        "diagnostic_only": not eligible,
        "origin": origin,
        "observed": {
            "sha256": observed_hash,
            "count": observed_count,
            "categories": observed_categories,
        },
        "required": {
            "origin": "builtin",
            "sha256": CANONICAL_RELEASE_PROMPT_MATRIX_SHA256,
            "count": CANONICAL_RELEASE_PROMPT_MATRIX_COUNT,
            "categories": dict(CANONICAL_RELEASE_PROMPT_MATRIX_CATEGORIES),
        },
    }


def _collect_source_package_parity(
    pairs: Sequence[tuple[str, Path, Path]],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for name, source_path, package_path in pairs:
        source_path = Path(source_path).resolve()
        package_path = Path(package_path).resolve()
        source_exists = source_path.is_file()
        package_exists = package_path.is_file()
        source_hash = _sha256_file(source_path) if source_exists else None
        package_hash = _sha256_file(package_path) if package_exists else None
        exact_bytes = bool(
            source_exists
            and package_exists
            and source_path.read_bytes() == package_path.read_bytes()
        )
        rows.append(
            {
                "name": str(name),
                "source": str(source_path),
                "package": str(package_path),
                "source_exists": source_exists,
                "package_exists": package_exists,
                "source_sha256": source_hash,
                "package_sha256": package_hash,
                "exact_bytes": exact_bytes,
            }
        )
    return {
        "method": "sha256_and_exact_bytes",
        "required_for_release": True,
        "passed": bool(rows) and all(row["exact_bytes"] for row in rows),
        "pairs": rows,
    }


def _timing_snapshot(value: Any, *, label: str) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} timing_ms must be an object")
    snapshot: Dict[str, float] = {}
    for key, raw in value.items():
        if raw is None:
            continue
        snapshot[str(key)] = _finite_float(raw, label=f"{label} timing {key}")
    for required in ("infer", "rank_pick", "total"):
        if required not in snapshot:
            raise ValueError(f"{label} timing_ms is missing {required}")
        if snapshot[required] < 0.0:
            raise ValueError(f"{label} timing {required} cannot be negative")
    return snapshot


def _compute_snapshot(value: Any, *, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} compute diagnostics must be an object")
    numeric_keys = (
        "requested_reasoning_cycles",
        "cycles_used",
        "exit_tol",
        "exit_tolerance",
        "exit_entropy_threshold",
        "prediction_stability_patience",
        "prediction_stability_tol",
        "prediction_stability_margin",
        "prediction_stability_rank_depth",
        "prediction_rank_depth",
        "decision_reference_cycles",
        "prediction_streak",
        "prediction_confidence_delta",
        "prediction_margin",
        "prediction_decision_margin",
        "prediction_class_count",
        "ponder_cost",
        "consistency_loss",
        "gating_entropy",
    )
    snapshot: Dict[str, Any] = {}
    for key in numeric_keys:
        if key in value and value[key] is not None:
            snapshot[key] = _finite_float(value[key], label=f"{label} compute {key}")
    for key in (
        "applied",
        "adaptive_compute",
        "prediction_class_selection_valid",
        "prediction_verifier_active",
    ):
        if key in value and value[key] is not None:
            snapshot[key] = _strict_bool(
                value[key], label=f"{label} compute {key}"
            )
    for key in ("exit_reason", "reasoning_budget_mode"):
        if key in value and value[key] is not None:
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{label} compute {key} must be a non-empty string")
            snapshot[key] = value[key]
    if value.get("prediction_class_indices") is not None:
        snapshot["prediction_class_indices"] = _integer_sequence(
            value["prediction_class_indices"],
            label=f"{label} compute prediction_class_indices",
        )
    return snapshot


def _candidate_snapshot(value: Any, *, label: str) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} top_candidates must be an array")
    if len(value) != TOP_CANDIDATE_COUNT:
        raise ValueError(
            f"{label} top_candidates must contain exactly "
            f"{TOP_CANDIDATE_COUNT} rows, got {len(value)}"
        )
    candidates: List[Dict[str, Any]] = []
    seen_texts: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} candidate {index} must be an object")
        raw_text = row.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"{label} candidate {index} has empty text")
        text = raw_text.strip()
        if text in seen_texts:
            raise ValueError(f"{label} candidate texts must be unique")
        seen_texts.add(text)
        candidate: Dict[str, Any] = {"rank": index + 1, "text": text}
        if row.get("score") is not None:
            candidate["score"] = _finite_float(
                row["score"], label=f"{label} candidate {index} score"
            )
        candidates.append(candidate)
    return candidates


def _response_snapshot(
    value: Any,
    *,
    label: str,
    wall_ms: float,
) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} response must be an object")
    if value.get("ok") is not True:
        raise ValueError(f"{label} runtime response was not successful")
    response = value.get("response")
    if not isinstance(response, str) or not response:
        raise ValueError(f"{label} response text must be non-empty")
    return {
        "response": response,
        "style_mode": str(value.get("style_mode", "")),
        "top_candidates": _candidate_snapshot(
            value.get("top_candidates"), label=label
        ),
        "compute": _compute_snapshot(value.get("compute"), label=label),
        "timing_ms": _timing_snapshot(value.get("timing_ms"), label=label),
        "outer_wall_ms": _finite_float(wall_ms, label=f"{label} outer wall time"),
        "auto_compute_plan": value.get("auto_compute_plan"),
    }


def _timing_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize empty timing values")
    checked = [_finite_float(value, label="timing summary value") for value in values]
    ordered = sorted(checked)
    return {
        "count": float(len(checked)),
        "total_ms": round(sum(checked), 6),
        "mean_ms": round(sum(checked) / len(checked), 6),
        "min_ms": round(ordered[0], 6),
        "max_ms": round(ordered[-1], 6),
    }


def _runtime_release_rank_depth(engine: Any) -> Any:
    defaults = getattr(engine, "defaults", {})
    if isinstance(defaults, Mapping) and "prediction_stability_rank_depth" in defaults:
        return defaults["prediction_stability_rank_depth"]
    return getattr(chat_app, "DEFAULT_PREDICTION_STABILITY_RANK_DEPTH", None)


def _effective_adaptive_defaults(engine: Any) -> Dict[str, Any]:
    defaults = getattr(engine, "defaults", {})
    if not isinstance(defaults, Mapping):
        defaults = {}
    fallback_by_key = {
        "adaptive_exit_tol": chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
        "adaptive_exit_entropy": chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        "prediction_stability_patience": (
            chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE
        ),
        "prediction_stability_tol": chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
        "prediction_stability_margin": getattr(
            chat_app, "DEFAULT_PREDICTION_STABILITY_MARGIN", None
        ),
        "prediction_stability_rank_depth": getattr(
            chat_app, "DEFAULT_PREDICTION_STABILITY_RANK_DEPTH", None
        ),
    }
    return {
        key: _optional_finite_float(
            defaults.get(key, fallback), label=f"effective runtime default {key}"
        )
        for key, fallback in fallback_by_key.items()
    }


def _chat_app_adaptive_defaults() -> Dict[str, float]:
    return {
        "adaptive_exit_tol": _finite_float(
            chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
            label="chat_app default adaptive_exit_tol",
        ),
        "adaptive_exit_entropy": _finite_float(
            chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
            label="chat_app default adaptive_exit_entropy",
        ),
        "prediction_stability_patience": _finite_float(
            chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE,
            label="chat_app default prediction_stability_patience",
        ),
        "prediction_stability_tol": _finite_float(
            chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
            label="chat_app default prediction_stability_tol",
        ),
    }


def _release_adaptive_default_mismatches(
    observed: Mapping[str, Any],
) -> Dict[str, Dict[str, float | None]]:
    mismatches: Dict[str, Dict[str, float | None]] = {}
    for key, required in AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS.items():
        actual = observed.get(key)
        if actual != required:
            mismatches[key] = {"actual": actual, "required": required}
    return mismatches


def _canonical_artifact_identity(
    weights: Path,
    metadata: Path,
) -> Dict[str, Any]:
    canonical_weights = DEFAULT_WEIGHTS.resolve()
    canonical_metadata = DEFAULT_META.resolve()
    canonical_weights_hash = (
        _sha256_file(canonical_weights) if canonical_weights.is_file() else None
    )
    canonical_metadata_hash = (
        _sha256_file(canonical_metadata) if canonical_metadata.is_file() else None
    )
    observed_weights_hash = _sha256_file(weights)
    observed_metadata_hash = _sha256_file(metadata)
    weights_path_exact = weights == canonical_weights
    metadata_path_exact = metadata == canonical_metadata
    weights_hash_exact = observed_weights_hash == CANONICAL_RELEASE_WEIGHTS_SHA256
    metadata_hash_exact = observed_metadata_hash == CANONICAL_RELEASE_METADATA_SHA256
    canonical_files_pinned = (
        canonical_weights_hash == CANONICAL_RELEASE_WEIGHTS_SHA256
        and canonical_metadata_hash == CANONICAL_RELEASE_METADATA_SHA256
    )
    return {
        "passed": bool(
            canonical_files_pinned
            and weights_path_exact
            and metadata_path_exact
            and weights_hash_exact
            and metadata_hash_exact
        ),
        "diagnostic_only": not (
            weights_path_exact
            and metadata_path_exact
            and weights_hash_exact
            and metadata_hash_exact
            and canonical_files_pinned
        ),
        "observed": {
            "weights": {
                "path": str(weights),
                "sha256": observed_weights_hash,
                "canonical_path_exact": weights_path_exact,
                "canonical_hash_exact": weights_hash_exact,
            },
            "metadata": {
                "path": str(metadata),
                "sha256": observed_metadata_hash,
                "canonical_path_exact": metadata_path_exact,
                "canonical_hash_exact": metadata_hash_exact,
            },
        },
        "required": {
            "weights": {
                "path": str(canonical_weights),
                "sha256": CANONICAL_RELEASE_WEIGHTS_SHA256,
                "current_file_sha256": canonical_weights_hash,
            },
            "metadata": {
                "path": str(canonical_metadata),
                "sha256": CANONICAL_RELEASE_METADATA_SHA256,
                "current_file_sha256": canonical_metadata_hash,
            },
        },
        "canonical_default_files_match_pinned_hashes": canonical_files_pinned,
    }


def _record_engine_surface(
    engine_factory: Callable[..., Any],
    *,
    surface: str,
    device: Any,
    device_info: Mapping[str, Any],
    weights: Path,
    metadata: Path,
    prompts: Sequence[Mapping[str, str]],
    module_provenance: Mapping[str, str],
) -> Dict[str, Any]:
    constructor_defaults = {
        "model_size": "auto",
        "pool_mode": "all",
        "reasoning_cycles": FIXED_CYCLES,
        "adaptive_compute": False,
        "auto_compute": False,
        "response_temperature": 0.0,
        "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
    }
    common_kwargs = {
        "style_mode": "auto",
        "response_temperature": 0.0,
        "show_top_responses": TOP_CANDIDATE_COUNT,
        "auto_compute": False,
        "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
        "grounding_enabled": False,
    }
    engine = engine_factory(device, dict(device_info), dict(constructor_defaults))
    load_status = engine.load(str(weights), str(metadata))
    if not isinstance(load_status, Mapping) or load_status.get("ok") is not True:
        raise ValueError(f"{surface} Engine did not report a successful load")
    labels = _engine_label_scope(engine, load_status)
    status_fn = getattr(engine, "status", None)
    status = status_fn() if callable(status_fn) else None
    calls: List[Dict[str, Any]] = []
    for index, case in enumerate(prompts):
        mode_order = (
            ("fixed", "adaptive") if index % 2 == 0 else ("adaptive", "fixed")
        )
        for mode in mode_order:
            adaptive = mode == "adaptive"
            started = time.perf_counter()
            raw = engine.chat(
                session_id=(
                    f"v51-response-fidelity-{surface}-{index + 1:03d}-{mode}"
                ),
                user_text=case["prompt"],
                reasoning_cycles=(ADAPTIVE_MAX_CYCLES if adaptive else FIXED_CYCLES),
                adaptive_compute=adaptive,
                **common_kwargs,
            )
            snapshot = _response_snapshot(
                raw,
                label=f"prompt {case['id']} {surface} {mode}",
                wall_ms=(time.perf_counter() - started) * 1000.0,
            )
            calls.append(
                {
                    "prompt_id": case["id"],
                    "prompt": case["prompt"],
                    "mode": mode,
                    "reasoning_cycles": (
                        ADAPTIVE_MAX_CYCLES if adaptive else FIXED_CYCLES
                    ),
                    "snapshot": snapshot,
                }
            )
    return {
        "surface": surface,
        "module_provenance": dict(module_provenance),
        "device_info": dict(device_info),
        "load_status": {
            "ok": True,
            "load_ms": _optional_finite_float(
                load_status.get("load_ms"), label=f"{surface} load_ms"
            ),
            "model_size": str(load_status.get("model_size", "")),
            "feature_mode": str(load_status.get("feature_mode", "")),
            "available_labels": _optional_finite_float(
                load_status.get("available_labels"),
                label=f"{surface} available_labels",
            ),
        },
        "status": status,
        "available_labels": labels,
        "defaults": _effective_adaptive_defaults(engine),
        "rank_depth": _finite_float(
            _runtime_release_rank_depth(engine),
            label=f"{surface} runtime release prediction stability rank depth",
        ),
        "calls": calls,
    }


class _PackagedSurfaceReplayEngine:
    def __init__(
        self,
        recording: Mapping[str, Any],
        device: Any,
        device_info: Mapping[str, Any],
        defaults: Mapping[str, Any],
    ) -> None:
        self.recording = dict(recording)
        self.device = device
        self.device_info = dict(device_info)
        self.constructor_defaults = dict(defaults)
        self.defaults = dict(recording["defaults"])
        self.available_labels = list(recording["available_labels"])
        self._calls = list(recording["calls"])
        self._next_call = 0

    def load(self, weights: str, metadata: str) -> Dict[str, Any]:
        return dict(self.recording["load_status"])

    def status(self) -> Any:
        return self.recording.get("status")

    def chat(self, **kwargs: Any) -> Dict[str, Any]:
        if self._next_call >= len(self._calls):
            raise ValueError("Packaged replay received more calls than recorded")
        row = self._calls[self._next_call]
        self._next_call += 1
        mode = "adaptive" if kwargs.get("adaptive_compute") is True else "fixed"
        if (
            kwargs.get("user_text") != row["prompt"]
            or mode != row["mode"]
            or int(kwargs.get("reasoning_cycles")) != int(row["reasoning_cycles"])
        ):
            raise ValueError("Packaged replay call order or request contract mismatch")
        snapshot = row["snapshot"]
        return {
            "ok": True,
            "response": snapshot["response"],
            "style_mode": snapshot["style_mode"],
            "top_candidates": snapshot["top_candidates"],
            "compute": snapshot["compute"],
            "timing_ms": snapshot["timing_ms"],
            "auto_compute_plan": snapshot["auto_compute_plan"],
        }

    @property
    def remaining_calls(self) -> int:
        return len(self._calls) - self._next_call


def _run_packaged_surface_subprocess(
    *,
    weights: Path,
    metadata: Path,
    prompts: Sequence[Mapping[str, str]],
    device: Any,
    device_info: Mapping[str, Any],
) -> Dict[str, Any]:
    bootstrap = r"""
import importlib.util
import json
from pathlib import Path
import sys

project_root = Path(sys.argv[1]).resolve()
config_path = Path(sys.argv[2]).resolve()
output_path = Path(sys.argv[3]).resolve()
runtime_dir = (project_root / "runtime_python").resolve()
sys.path.insert(0, str(runtime_dir))

import chat_app
import chat_pipeline
import chat_web_app
import device_utils
import interaction_planner
import model_variants

modules = {
    "chat_app": chat_app,
    "chat_pipeline": chat_pipeline,
    "chat_web_app": chat_web_app,
    "device_utils": device_utils,
    "interaction_planner": interaction_planner,
    "model_variants": model_variants,
}
module_paths = {}
for name, module in modules.items():
    path = Path(module.__file__).resolve()
    if runtime_dir not in path.parents:
        raise RuntimeError(f"{name} resolved outside packaged runtime: {path}")
    module_paths[name] = str(path)

gate_path = project_root / "source" / "run_v51_chat_response_fidelity_gate.py"
spec = importlib.util.spec_from_file_location("_v51_packaged_gate_worker", gate_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load gate worker: {gate_path}")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
config = json.loads(config_path.read_text(encoding="utf-8"))
device, resolved = device_utils.resolve_device(config["device_request"])
recording = gate._record_engine_surface(
    gate.Engine,
    surface="packaged",
    device=device,
    device_info=resolved,
    weights=Path(config["weights"]),
    metadata=Path(config["metadata"]),
    prompts=config["prompts"],
    module_provenance=module_paths,
)
output_path.write_text(gate._strict_json(recording) + "\n", encoding="utf-8")
"""
    config = {
        "weights": str(weights),
        "metadata": str(metadata),
        "prompts": list(prompts),
        "device_request": str(device_info.get("resolved", device)),
    }
    with tempfile.TemporaryDirectory(prefix="v51-packaged-fidelity-") as temp_dir:
        temp_root = Path(temp_dir)
        config_path = temp_root / "config.json"
        output_path = temp_root / "recording.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, allow_nan=False), encoding="utf-8"
        )
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                bootstrap,
                str(PROJECT_ROOT),
                str(config_path),
                str(output_path),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Packaged fidelity worker failed: "
                + (completed.stderr.strip() or f"exit {completed.returncode}")
            )
        if completed.stdout.strip():
            raise RuntimeError("Packaged fidelity worker emitted unexpected stdout")
        if not output_path.is_file():
            raise RuntimeError("Packaged fidelity worker produced no recording")
        try:
            recording = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Packaged fidelity worker returned invalid JSON") from exc
    if not isinstance(recording, dict) or recording.get("surface") != "packaged":
        raise RuntimeError("Packaged fidelity worker returned an invalid recording")
    return recording


def _deterministic_response_evidence(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    """Exclude wall-clock timing while retaining exact output and compute evidence."""

    return {
        "response": snapshot.get("response"),
        "style_mode": snapshot.get("style_mode"),
        "top_candidates": snapshot.get("top_candidates"),
        "compute": snapshot.get("compute"),
        "auto_compute_plan": snapshot.get("auto_compute_plan"),
    }


def _behavior_parity_mismatches(
    source_snapshot: Mapping[str, Any],
    packaged_snapshot: Mapping[str, Any],
) -> List[str]:
    source_evidence = _deterministic_response_evidence(source_snapshot)
    packaged_evidence = _deterministic_response_evidence(packaged_snapshot)
    return [
        key
        for key in source_evidence
        if source_evidence[key] != packaged_evidence.get(key)
    ]


def _engine_label_scope(engine: Any, load_status: Mapping[str, Any]) -> List[int]:
    raw_labels = getattr(engine, "available_labels", None)
    if raw_labels is None:
        raw_count = load_status.get("available_labels")
        count = int(_finite_float(raw_count, label="loaded available label count"))
        raw_labels = list(range(count))
    labels = _integer_sequence(raw_labels, label="engine available_labels")
    if len(labels) < AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH + 1:
        raise ValueError(
            "Engine available-label scope is too small for the release rank depth"
        )
    if len(set(labels)) != len(labels) or any(label < 0 for label in labels):
        raise ValueError("Engine available_labels must be unique nonnegative integers")
    reported_count = int(
        _finite_float(
            load_status.get("available_labels"), label="loaded available label count"
        )
    )
    if reported_count != len(labels):
        raise ValueError("Engine label scope does not match the loaded status count")
    return labels


def _canonical_model_identity(
    load_status_by_surface: Mapping[str, Mapping[str, Any]],
    labels_by_surface: Mapping[str, Sequence[int]],
    status_by_surface: Mapping[str, Mapping[str, Any]],
    runtime_defaults_by_surface: Mapping[str, Mapping[str, Any]],
    rank_depth_by_surface: Mapping[str, float],
    *,
    weights: Path,
    metadata: Path,
) -> Dict[str, Any]:
    required = {
        "model_size": CANONICAL_RELEASE_MODEL_SIZE,
        "feature_mode": CANONICAL_RELEASE_FEATURE_MODE,
        "class_count": CANONICAL_RELEASE_CLASS_COUNT,
        "class_indices": list(range(CANONICAL_RELEASE_CLASS_COUNT)),
    }
    surfaces: Dict[str, Dict[str, Any]] = {}
    for surface, status in load_status_by_surface.items():
        labels = list(labels_by_surface[surface])
        engine_status = status_by_surface.get(surface, {})
        runtime_defaults = runtime_defaults_by_surface[surface]
        actual = {
            "model_size": str(status.get("model_size", "")),
            "feature_mode": str(status.get("feature_mode", "")),
            "class_count": len(labels),
            "class_indices": labels,
        }
        status_actual = {
            "loaded": engine_status.get("loaded"),
            "weights_path_exact": (
                Path(str(engine_status.get("weights", ""))).resolve() == weights
            ),
            "metadata_path_exact": (
                Path(str(engine_status.get("meta", ""))).resolve() == metadata
            ),
            "runtime_compute_supported": engine_status.get(
                "runtime_compute_supported"
            ),
            "sessions": engine_status.get("sessions"),
            "reasoning_cycles": engine_status.get("reasoning_cycles"),
            "adaptive_compute": engine_status.get("adaptive_compute"),
            "auto_compute": engine_status.get("auto_compute"),
        }
        required_status = {
            "loaded": True,
            "weights_path_exact": True,
            "metadata_path_exact": True,
            "runtime_compute_supported": True,
            "sessions": 0,
            "reasoning_cycles": FIXED_CYCLES,
            "adaptive_compute": False,
            "auto_compute": False,
        }
        verifier_defaults_exact = bool(
            not _release_adaptive_default_mismatches(runtime_defaults)
            and runtime_defaults.get("prediction_stability_margin")
            == REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN
            and rank_depth_by_surface[surface]
            == float(REQUIRED_RELEASE_PREDICTION_STABILITY_RANK_DEPTH)
        )
        surfaces[surface] = {
            "passed": bool(
                actual == required
                and status_actual == required_status
                and verifier_defaults_exact
            ),
            "actual": actual,
            "status_actual": status_actual,
            "status_required": required_status,
            "verifier_defaults_exact": verifier_defaults_exact,
        }
    return {
        "passed": bool(surfaces) and all(row["passed"] for row in surfaces.values()),
        "required": required,
        "surfaces": surfaces,
    }


def _mode_contract_violations(
    snapshot: Mapping[str, Any],
    *,
    mode: str,
    expected_labels: Sequence[int],
    engine_release_rank_depth: float,
    engine_release_defaults: Mapping[str, Any],
) -> List[str]:
    adaptive = mode == "adaptive"
    expected_budget = ADAPTIVE_MAX_CYCLES if adaptive else FIXED_CYCLES
    violations: List[str] = []

    def require(condition: bool, name: str) -> None:
        if not condition:
            violations.append(name)

    require(snapshot.get("applied") is True, "runtime_compute_not_applied")
    require(
        snapshot.get("adaptive_compute") is adaptive,
        "adaptive_compute_flag_mismatch",
    )
    require(
        snapshot.get("requested_reasoning_cycles") == float(expected_budget),
        "requested_reasoning_budget_mismatch",
    )
    telemetry_keys = {
        "adaptive_exit_tol": "exit_tol",
        "adaptive_exit_entropy": "exit_entropy_threshold",
        "prediction_stability_patience": "prediction_stability_patience",
        "prediction_stability_tol": "prediction_stability_tol",
    }
    for runtime_key, telemetry_key in telemetry_keys.items():
        required = AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS[runtime_key]
        require(
            snapshot.get(telemetry_key) == required,
            f"configured_{telemetry_key}_mismatch",
        )
        require(
            engine_release_defaults.get(runtime_key) == required,
            f"engine_release_{runtime_key}_mismatch",
        )
    cycles_used = snapshot.get("cycles_used")
    if adaptive:
        require(
            cycles_used is not None
            and 0.0 < float(cycles_used) <= float(ADAPTIVE_MAX_CYCLES),
            "adaptive_cycles_out_of_bounds",
        )
    else:
        require(cycles_used == float(FIXED_CYCLES), "fixed_cycles_not_exact")
    require(
        snapshot.get("prediction_stability_margin")
        == float(PREDICTION_STABILITY_MARGIN),
        "configured_prediction_margin_mismatch",
    )
    require(
        snapshot.get("prediction_stability_rank_depth")
        == float(AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH),
        "configured_prediction_rank_depth_mismatch",
    )
    require(
        engine_release_rank_depth
        == float(AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH),
        "engine_release_rank_depth_mismatch",
    )
    exit_reason = snapshot.get("exit_reason")
    require(isinstance(exit_reason, str) and bool(exit_reason), "exit_reason_missing")

    if adaptive:
        scope = snapshot.get("prediction_class_indices")
        require(scope == list(expected_labels), "prediction_class_scope_mismatch")
        require(
            isinstance(scope, list)
            and len(scope) == len(set(scope))
            and len(scope)
            >= AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH + 1,
            "prediction_class_scope_invalid",
        )
        require(
            snapshot.get("prediction_verifier_active") is True,
            "prediction_verifier_not_active",
        )
        require(
            snapshot.get("prediction_class_selection_valid") is True,
            "prediction_class_selection_not_valid",
        )
        require(
            snapshot.get("prediction_class_count") == float(len(expected_labels)),
            "prediction_class_count_mismatch",
        )
        require(
            snapshot.get("prediction_rank_depth")
            == float(AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH),
            "observed_prediction_rank_depth_mismatch",
        )
        decision_reference_cycles = snapshot.get("decision_reference_cycles")
        require(
            decision_reference_cycles
            == float(REQUIRED_RELEASE_DECISION_REFERENCE_CYCLES),
            "decision_reference_cycles_mismatch",
        )
        decision_margin = snapshot.get("prediction_decision_margin")
        require(decision_margin is not None, "prediction_decision_margin_missing")
        require(
            exit_reason in KNOWN_ADAPTIVE_EXIT_REASONS,
            "adaptive_exit_reason_unknown",
        )
        if exit_reason == "prediction_stable":
            require(
                decision_margin is not None
                and float(decision_margin) >= float(PREDICTION_STABILITY_MARGIN),
                "prediction_stable_decision_margin_below_floor",
            )
            require(
                decision_reference_cycles is not None
                and cycles_used is not None
                and float(cycles_used) < float(decision_reference_cycles),
                "prediction_stable_not_before_reference_budget",
            )
        elif exit_reason == "decision_reference_budget":
            require(
                decision_reference_cycles is not None
                and cycles_used == decision_reference_cycles,
                "decision_reference_budget_cycle_mismatch",
            )
    else:
        require(
            snapshot.get("prediction_verifier_active") is False,
            "fixed_prediction_verifier_active",
        )
        require(exit_reason == "max_cycles", "fixed_exit_reason_not_max_cycles")

    return violations


def _source_hashes() -> Dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        SOURCE_DIR / "chat_web_app.py",
        SOURCE_DIR / "chat_app.py",
        SOURCE_DIR / "chat_pipeline.py",
        SOURCE_DIR / "interaction_planner.py",
        SOURCE_DIR / "model_variants.py",
    )
    return {
        path.relative_to(PROJECT_ROOT).as_posix(): _sha256_file(path)
        for path in paths
        if path.is_file()
    }


def run_gate(
    *,
    weights: Path,
    metadata: Path,
    prompts: Sequence[Any] | None = None,
    prompt_source: Mapping[str, Any] | None = None,
    device: Any = None,
    device_info: Mapping[str, Any] | None = None,
    engine_factory: Callable[[Any, Dict[str, Any], Dict[str, Any]], Any] | None = None,
    packaged_engine_factory: Callable[
        [Any, Dict[str, Any], Dict[str, Any]], Any
    ]
    | None = None,
    clock: Callable[[], float] | None = None,
    created_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    source_package_parity_pairs: Sequence[tuple[str, Path, Path]] | None = None,
) -> Dict[str, Any]:
    """Run isolated fixed/adaptive comparisons on source and packaged engines.

    Factories and ``clock`` are injectable so the protocol can be unit tested
    without constructing models. Production callers execute both real Engine
    implementations. When only ``engine_factory`` is injected, tests use it for
    both surfaces; production never takes that shortcut.
    """

    weights = Path(weights).expanduser().resolve()
    metadata = Path(metadata).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not metadata.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata}")
    artifact_identity = _canonical_artifact_identity(weights, metadata)
    normalized_prompts = normalize_prompt_matrix(
        FROZEN_RELEASE_PROMPT_MATRIX if prompts is None else prompts
    )
    source = dict(
        prompt_source
        or {"origin": "builtin", "path": None, "source_file_sha256": None}
    )
    matrix_observation = _prompt_matrix_release_observation(
        normalized_prompts, source
    )
    source_package_parity = _collect_source_package_parity(
        DEFAULT_SOURCE_PACKAGE_PARITY_PAIRS
        if source_package_parity_pairs is None
        else source_package_parity_pairs
    )
    surface_specific_runtime_hashes = _collect_source_package_parity(
        SURFACE_SPECIFIC_RUNTIME_HASH_PAIRS
    )
    surface_specific_runtime_hashes.update(
        {
            "required_for_release": False,
            "role": (
                "diagnostic_hash_provenance_for_intentionally_surface_specific_files"
            ),
        }
    )
    resolved_device_info = dict(
        device_info or {"requested": "injected", "resolved": str(device)}
    )
    source_factory = engine_factory or Engine
    package_factory = packaged_engine_factory
    packaged_recording: Dict[str, Any] | None = None
    if package_factory is None:
        if engine_factory is not None:
            package_factory = source_factory
        else:
            packaged_recording = _run_packaged_surface_subprocess(
                weights=weights,
                metadata=metadata,
                prompts=normalized_prompts,
                device=device,
                device_info=resolved_device_info,
            )

            def package_factory(
                package_device: Any,
                package_device_info: Dict[str, Any],
                package_defaults: Dict[str, Any],
            ) -> _PackagedSurfaceReplayEngine:
                assert packaged_recording is not None
                return _PackagedSurfaceReplayEngine(
                    packaged_recording,
                    package_device,
                    package_device_info,
                    package_defaults,
                )
    timer = clock or time.perf_counter
    constructor_defaults = {
        "model_size": "auto",
        "pool_mode": "all",
        "reasoning_cycles": FIXED_CYCLES,
        "adaptive_compute": False,
        "auto_compute": False,
        "response_temperature": 0.0,
        "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
    }
    engines = {
        "source": source_factory(
            device, dict(resolved_device_info), dict(constructor_defaults)
        ),
        "packaged": package_factory(
            device, dict(resolved_device_info), dict(constructor_defaults)
        ),
    }
    injected_surface_factories = engine_factory is not None
    if injected_surface_factories:
        surface_module_provenance = {
            "mode": "injected_factories",
            "source_factory": getattr(
                source_factory, "__v51_verified_surface__", None
            ),
            "packaged_factory": getattr(
                package_factory, "__v51_verified_surface__", None
            ),
        }
        surface_module_provenance_passed = bool(
            source_factory is not package_factory
            and surface_module_provenance["source_factory"] == "source"
            and surface_module_provenance["packaged_factory"] == "packaged"
        )
    else:
        assert packaged_recording is not None
        runtime_root = (PROJECT_ROOT / "runtime_python").resolve()
        packaged_module_paths = dict(
            packaged_recording.get("module_provenance", {})
        )
        required_packaged_modules = {
            "chat_app",
            "chat_pipeline",
            "chat_web_app",
            "device_utils",
            "interaction_planner",
            "model_variants",
        }
        surface_module_provenance_passed = bool(
            set(packaged_module_paths) == required_packaged_modules
            and all(
                runtime_root in Path(path).resolve().parents
                for path in packaged_module_paths.values()
            )
        )
        surface_module_provenance = {
            "mode": "isolated_python_child",
            "python_isolated_flag": True,
            "packaged_modules": packaged_module_paths,
        }
    load_status_by_surface: Dict[str, Mapping[str, Any]] = {}
    status_by_surface: Dict[str, Mapping[str, Any]] = {}
    expected_labels_by_surface: Dict[str, List[int]] = {}
    adaptive_runtime_defaults_by_surface: Dict[str, Dict[str, Any]] = {}
    adaptive_runtime_default_mismatches_by_surface: Dict[
        str, Dict[str, Dict[str, float | None]]
    ] = {}
    engine_release_rank_depth_by_surface: Dict[str, float] = {}
    for surface, engine in engines.items():
        load_status = engine.load(str(weights), str(metadata))
        if not isinstance(load_status, Mapping) or load_status.get("ok") is not True:
            raise ValueError(f"{surface} Engine did not report a successful load")
        load_status_by_surface[surface] = load_status
        status_fn = getattr(engine, "status", None)
        raw_status = status_fn() if callable(status_fn) else None
        status_by_surface[surface] = (
            dict(raw_status) if isinstance(raw_status, Mapping) else {}
        )
        expected_labels_by_surface[surface] = _engine_label_scope(engine, load_status)
        runtime_defaults = _effective_adaptive_defaults(engine)
        adaptive_runtime_defaults_by_surface[surface] = runtime_defaults
        adaptive_runtime_default_mismatches_by_surface[surface] = (
            _release_adaptive_default_mismatches(runtime_defaults)
        )
        engine_release_rank_depth_by_surface[surface] = _finite_float(
            _runtime_release_rank_depth(engine),
            label=f"{surface} runtime release prediction stability rank depth",
        )

    model_identity = _canonical_model_identity(
        load_status_by_surface,
        expected_labels_by_surface,
        status_by_surface,
        adaptive_runtime_defaults_by_surface,
        engine_release_rank_depth_by_surface,
        weights=weights,
        metadata=metadata,
    )
    chat_app_runtime_defaults = _chat_app_adaptive_defaults()
    chat_app_runtime_default_mismatches = _release_adaptive_default_mismatches(
        chat_app_runtime_defaults
    )

    common_kwargs = {
        "style_mode": "auto",
        "response_temperature": 0.0,
        "show_top_responses": TOP_CANDIDATE_COUNT,
        "auto_compute": False,
        "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
        "grounding_enabled": False,
    }
    prompt_results: List[Dict[str, Any]] = []
    response_mismatches = 0
    candidate_mismatches = 0
    any_mismatches = 0
    fixed_cycle_violations = 0
    adaptive_cycle_violations = 0
    auto_compute_plan_violations = 0
    contract_violations: List[Dict[str, Any]] = []
    behavior_parity_violations: List[Dict[str, Any]] = []

    for index, case in enumerate(normalized_prompts):
        mode_order = ("fixed", "adaptive") if index % 2 == 0 else ("adaptive", "fixed")
        surface_results: Dict[str, Dict[str, Dict[str, Any]]] = {
            "source": {},
            "packaged": {},
        }
        for mode in mode_order:
            adaptive = mode == "adaptive"
            for surface, engine in engines.items():
                session_id = (
                    f"v51-response-fidelity-{surface}-{index + 1:03d}-{mode}"
                )
                started = timer()
                raw = engine.chat(
                    session_id=session_id,
                    user_text=case["prompt"],
                    reasoning_cycles=(
                        ADAPTIVE_MAX_CYCLES if adaptive else FIXED_CYCLES
                    ),
                    adaptive_compute=adaptive,
                    **common_kwargs,
                )
                wall_ms = (timer() - started) * 1000.0
                surface_results[surface][mode] = _response_snapshot(
                    raw,
                    label=f"prompt {case['id']} {surface} {mode}",
                    wall_ms=wall_ms,
                )

        per_surface_comparison: Dict[str, Dict[str, bool]] = {}
        response_match = True
        candidates_match = True
        mismatch_kinds: List[str] = []
        for surface, mode_results in surface_results.items():
            fixed = mode_results["fixed"]
            adaptive_result = mode_results["adaptive"]
            fixed_texts = [row["text"] for row in fixed["top_candidates"]]
            adaptive_texts = [
                row["text"] for row in adaptive_result["top_candidates"]
            ]
            surface_response_match = fixed["response"] == adaptive_result["response"]
            surface_candidates_match = fixed_texts == adaptive_texts
            response_match = response_match and surface_response_match
            candidates_match = candidates_match and surface_candidates_match
            per_surface_comparison[surface] = {
                "response_exact_match": surface_response_match,
                "top_candidate_text_order_exact_match": surface_candidates_match,
            }

            fixed_cycles = fixed["compute"].get("cycles_used")
            adaptive_cycles = adaptive_result["compute"].get("cycles_used")
            if fixed_cycles is None or fixed_cycles != float(FIXED_CYCLES):
                fixed_cycle_violations += 1
            if (
                adaptive_cycles is None
                or adaptive_cycles <= 0.0
                or adaptive_cycles > float(ADAPTIVE_MAX_CYCLES)
            ):
                adaptive_cycle_violations += 1
            auto_compute_plan_violations += int(
                fixed["auto_compute_plan"] is not None
            )
            auto_compute_plan_violations += int(
                adaptive_result["auto_compute_plan"] is not None
            )
            for mode, mode_snapshot in (
                ("fixed", fixed),
                ("adaptive", adaptive_result),
            ):
                mode_violations = _mode_contract_violations(
                    mode_snapshot["compute"],
                    mode=mode,
                    expected_labels=expected_labels_by_surface[surface],
                    engine_release_rank_depth=(
                        engine_release_rank_depth_by_surface[surface]
                    ),
                    engine_release_defaults=(
                        adaptive_runtime_defaults_by_surface[surface]
                    ),
                )
                if mode_snapshot["auto_compute_plan"] is not None:
                    mode_violations.append("auto_compute_plan_present")
                if mode_violations:
                    contract_violations.append(
                        {
                            "prompt_id": case["id"],
                            "surface": surface,
                            "mode": mode,
                            "violations": mode_violations,
                        }
                    )

        if not response_match:
            response_mismatches += 1
            mismatch_kinds.append("response_text")
        if not candidates_match:
            candidate_mismatches += 1
            mismatch_kinds.append("top_candidate_text_order")
        if mismatch_kinds:
            any_mismatches += 1

        behavior_parity_by_mode: Dict[str, Dict[str, Any]] = {}
        for mode in ("fixed", "adaptive"):
            parity_mismatches = _behavior_parity_mismatches(
                surface_results["source"][mode],
                surface_results["packaged"][mode],
            )
            behavior_parity_by_mode[mode] = {
                "passed": not parity_mismatches,
                "mismatched_fields": parity_mismatches,
            }
            if parity_mismatches:
                behavior_parity_violations.append(
                    {
                        "prompt_id": case["id"],
                        "mode": mode,
                        "mismatched_fields": parity_mismatches,
                    }
                )

        prompt_results.append(
            {
                **case,
                "measurement_order": list(mode_order),
                # Source aliases preserve the original artifact shape.
                "fixed": surface_results["source"]["fixed"],
                "adaptive": surface_results["source"]["adaptive"],
                "surfaces": surface_results,
                "comparison": {
                    "response_exact_match": response_match,
                    "top_candidate_text_order_exact_match": candidates_match,
                    "mismatch_kinds": mismatch_kinds,
                    "by_surface": per_surface_comparison,
                    "source_package_behavior_parity": behavior_parity_by_mode,
                },
            }
        )

    engine_defaults_passed = not any(
        adaptive_runtime_default_mismatches_by_surface.values()
    )
    packaged_engine = engines["packaged"]
    packaged_recording_consumed = bool(
        not isinstance(packaged_engine, _PackagedSurfaceReplayEngine)
        or packaged_engine.remaining_calls == 0
    )
    checks = {
        "zero_response_text_mismatches": {
            "passed": response_mismatches == 0,
            "actual": response_mismatches,
            "required": 0,
        },
        "zero_top_candidate_text_order_mismatches": {
            "passed": candidate_mismatches == 0,
            "actual": candidate_mismatches,
            "required": 0,
        },
        "canonical_builtin_release_prompt_matrix": {
            "passed": matrix_observation["release_eligible"],
            "diagnostic_only": matrix_observation["diagnostic_only"],
            "origin": matrix_observation["origin"],
            "observed": matrix_observation["observed"],
            "required": matrix_observation["required"],
        },
        "canonical_default_checkpoint_and_metadata_identity": {
            "passed": artifact_identity["passed"],
            "diagnostic_only": artifact_identity["diagnostic_only"],
            "observed": artifact_identity["observed"],
            "required": artifact_identity["required"],
            "canonical_default_files_match_pinned_hashes": artifact_identity[
                "canonical_default_files_match_pinned_hashes"
            ],
        },
        "canonical_model_runtime_identity": model_identity,
        "isolated_source_packaged_module_provenance": {
            "passed": surface_module_provenance_passed,
            "evidence": surface_module_provenance,
        },
        "source_package_runtime_exact_parity": {
            "passed": source_package_parity["passed"],
            "method": source_package_parity["method"],
            "pair_count": len(source_package_parity["pairs"]),
        },
        "source_package_engine_exact_behavior_parity": {
            "passed": (
                not behavior_parity_violations and packaged_recording_consumed
            ),
            "timing_excluded_as_nondeterministic": True,
            "compared_fields": list(
                _deterministic_response_evidence(
                    prompt_results[0]["surfaces"]["source"]["fixed"]
                )
            ),
            "violation_count": len(behavior_parity_violations),
            "violations": behavior_parity_violations,
            "packaged_recording_consumed": packaged_recording_consumed,
        },
        "fixed_runs_used_exactly_3_cycles": {
            "passed": fixed_cycle_violations == 0,
            "violations": fixed_cycle_violations,
        },
        "adaptive_runs_stayed_within_8_cycles": {
            "passed": adaptive_cycle_violations == 0,
            "violations": adaptive_cycle_violations,
        },
        "all_runs_exposed_exact_unique_candidate_count": {
            "passed": True,
            "required_per_run": TOP_CANDIDATE_COUNT,
        },
        "auto_compute_remained_disabled": {
            "passed": auto_compute_plan_violations == 0,
            "violations": auto_compute_plan_violations,
        },
        "canonical_chat_app_adaptive_defaults_unchanged": {
            "passed": not chat_app_runtime_default_mismatches,
            "actual": chat_app_runtime_defaults,
            "required": dict(AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS),
            "mismatches": chat_app_runtime_default_mismatches,
        },
        "authoritative_chat_app_adaptive_runtime_defaults": {
            "passed": engine_defaults_passed,
            "actual_by_surface": adaptive_runtime_defaults_by_surface,
            "required": dict(AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS),
            "mismatches_by_surface": (
                adaptive_runtime_default_mismatches_by_surface
            ),
        },
        "runtime_release_verifier_contract_observed": {
            "passed": not contract_violations,
            "violation_count": len(contract_violations),
            "violations": contract_violations,
            "required_rank_depth": AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH,
            "required_margin": PREDICTION_STABILITY_MARGIN,
            "required_decision_reference_cycles": (
                REQUIRED_RELEASE_DECISION_REFERENCE_CYCLES
            ),
            "required_adaptive_runtime_defaults": dict(
                AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS
            ),
            "required_class_indices": list(range(CANONICAL_RELEASE_CLASS_COUNT)),
        },
    }
    all_checks_passed = all(bool(row["passed"]) for row in checks.values())

    surface_summaries: Dict[str, Dict[str, Any]] = {}
    for surface in engines:
        fixed_rows = [row["surfaces"][surface]["fixed"] for row in prompt_results]
        adaptive_rows = [
            row["surfaces"][surface]["adaptive"] for row in prompt_results
        ]
        fixed_exits = Counter(
            str(row["compute"].get("exit_reason", "unknown")) for row in fixed_rows
        )
        adaptive_exits = Counter(
            str(row["compute"].get("exit_reason", "unknown"))
            for row in adaptive_rows
        )
        surface_summaries[surface] = {
            "fixed_total_timing_ms": _timing_summary(
                [row["timing_ms"]["total"] for row in fixed_rows]
            ),
            "adaptive_total_timing_ms": _timing_summary(
                [row["timing_ms"]["total"] for row in adaptive_rows]
            ),
            "fixed_cycles": [row["compute"].get("cycles_used") for row in fixed_rows],
            "adaptive_cycles": [
                row["compute"].get("cycles_used") for row in adaptive_rows
            ],
            "fixed_exit_reasons": dict(sorted(fixed_exits.items())),
            "adaptive_exit_reasons": dict(sorted(adaptive_exits.items())),
        }

    source_summary = surface_summaries["source"]
    release_eligible = bool(
        matrix_observation["release_eligible"]
        and artifact_identity["passed"]
        and model_identity["passed"]
    )

    worktree_status = _git_text("status", "--porcelain", "--untracked-files=all")
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "claim_scope": {
            "matrix_kind": (
                "frozen_release_prompt_matrix"
                if matrix_observation["release_eligible"]
                else "diagnostic_prompt_matrix"
            ),
            "artifact_kind": (
                "canonical_default_v51_artifacts"
                if artifact_identity["passed"]
                else "diagnostic_custom_artifacts"
            ),
            "statement": (
                "Deterministic regression evidence for the exact canonical v51 "
                "prompt matrix, checkpoint, metadata, and runtime identity only."
            ),
            "held_out_claim": False,
            "universal_chat_fidelity_claim": False,
            "release_eligible": release_eligible,
        },
        "checkpoint": artifact_identity["observed"]["weights"],
        "metadata": artifact_identity["observed"]["metadata"],
        "artifact_identity": artifact_identity,
        "model_identity": model_identity,
        "prompt_matrix": {
            **source,
            "sha256": matrix_observation["observed"]["sha256"],
            "count": matrix_observation["observed"]["count"],
            "categories": matrix_observation["observed"]["categories"],
            "frozen_for_run": True,
            "release_eligible": matrix_observation["release_eligible"],
            "diagnostic_only": matrix_observation["diagnostic_only"],
            "canonical_release_contract": matrix_observation["required"],
        },
        "source_package_parity": source_package_parity,
        "surface_specific_runtime_hashes": surface_specific_runtime_hashes,
        "surface_module_provenance": surface_module_provenance,
        "settings": {
            "engines": {
                "source": "source.chat_web_app.Engine",
                "packaged": "runtime_python.chat_web_app.Engine",
            },
            "fixed_cycles": FIXED_CYCLES,
            "adaptive_max_cycles": ADAPTIVE_MAX_CYCLES,
            "response_temperature": 0.0,
            "auto_compute": False,
            "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
            "adaptive_runtime_defaults": adaptive_runtime_defaults_by_surface[
                "source"
            ],
            "adaptive_runtime_defaults_by_surface": (
                adaptive_runtime_defaults_by_surface
            ),
            "authoritative_adaptive_runtime_defaults": dict(
                AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS
            ),
            "prediction_stability_rank_depth": {
                "selection": "runtime_release_default",
                "authoritative": AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH,
                "resolved": engine_release_rank_depth_by_surface["source"],
                "resolved_by_surface": engine_release_rank_depth_by_surface,
                "explicitly_overridden_by_gate": False,
            },
            "available_label_scope": expected_labels_by_surface["source"],
            "available_label_scope_by_surface": expected_labels_by_surface,
            "show_top_responses": TOP_CANDIDATE_COUNT,
            "session_isolation": "unique_empty_session_per_prompt_and_mode",
            "measurement_order": "alternate_fixed_first_and_adaptive_first_by_prompt",
        },
        "load": {
            surface: {
                "ok": True,
                "load_ms": _optional_finite_float(
                    status.get("load_ms"), label=f"{surface} load_ms"
                ),
                "model_size": str(status.get("model_size", "")),
                "feature_mode": str(status.get("feature_mode", "")),
                "available_labels": _optional_finite_float(
                    status.get("available_labels"),
                    label=f"{surface} available_labels",
                ),
            }
            for surface, status in load_status_by_surface.items()
        },
        "summary": {
            "prompt_count": len(normalized_prompts),
            "response_text_mismatch_count": response_mismatches,
            "top_candidate_text_order_mismatch_count": candidate_mismatches,
            "any_fidelity_mismatch_count": any_mismatches,
            "source_package_behavior_parity_violation_count": len(
                behavior_parity_violations
            ),
            "fixed_total_timing_ms": source_summary["fixed_total_timing_ms"],
            "adaptive_total_timing_ms": source_summary[
                "adaptive_total_timing_ms"
            ],
            "fixed_cycles": [
                _optional_finite_float(value, label="fixed cycles summary")
                for value in source_summary["fixed_cycles"]
            ],
            "adaptive_cycles": [
                _optional_finite_float(value, label="adaptive cycles summary")
                for value in source_summary["adaptive_cycles"]
            ],
            "fixed_exit_reasons": source_summary["fixed_exit_reasons"],
            "adaptive_exit_reasons": source_summary["adaptive_exit_reasons"],
            "by_surface": surface_summaries,
        },
        "gates": {"passed": all_checks_passed, "checks": checks},
        "provenance": dict(provenance) if provenance is not None else {
            "git": {
                "commit": _git_text("rev-parse", "HEAD"),
                "branch": _git_text("rev-parse", "--abbrev-ref", "HEAD"),
                "worktree_dirty": bool(worktree_status),
            },
            "source_sha256": _source_hashes(),
            "runtime": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "device": resolved_device_info,
            },
        },
        "prompt_results": prompt_results,
    }
    # Validate the complete payload here, not only when the CLI writes it.
    _strict_json(payload)
    return payload


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
            "Run the v51 frozen release-prompt response-fidelity gate through "
            "source chat_web_app.Engine. This is not a universal or held-out claim."
        )
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--meta", "--metadata", dest="metadata", default=str(DEFAULT_META))
    parser.add_argument("--prompts-json", "--prompts_json", type=Path, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--device-preference",
        "--device_preference",
        default="cuda,npu,xpu,dml,mps,cpu",
    )
    parser.add_argument("--torch-num-threads", "--torch_num_threads", type=int, default=0)
    parser.add_argument(
        "--torch-interop-threads", "--torch_num_interop_threads", type=int, default=0
    )
    parser.add_argument(
        "--strict-determinism",
        action="store_true",
        help="Ask torch to use deterministic algorithms where available.",
    )
    parser.add_argument(
        "--enforce-gates",
        action="store_true",
        help="Exit with status 2 when any fidelity or protocol check fails.",
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
    prompts, prompt_source = load_prompt_matrix(args.prompts_json)
    payload = run_gate(
        weights=Path(args.weights),
        metadata=Path(args.metadata),
        prompts=prompts,
        prompt_source=prompt_source,
        device=device,
        device_info=device_info,
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
