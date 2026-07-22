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
import time
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
PREDICTION_STABILITY_MARGIN = chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN = 1e-4
AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH = int(
    chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
)
REQUIRED_RELEASE_PREDICTION_STABILITY_RANK_DEPTH = 3
AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS = {
    "adaptive_exit_tol": float(chat_app.DEFAULT_ADAPTIVE_EXIT_TOL),
    "adaptive_exit_entropy": float(chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY),
    "prediction_stability_patience": float(
        chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE
    ),
    "prediction_stability_tol": float(chat_app.DEFAULT_PREDICTION_STABILITY_TOL),
}
TOP_CANDIDATE_COUNT = 5

if PREDICTION_STABILITY_MARGIN != REQUIRED_RELEASE_PREDICTION_STABILITY_MARGIN:
    raise RuntimeError(
        "The authoritative runtime prediction-stability margin is not the "
        "v51 release value 0.0001"
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
        "model_variants",
        SOURCE_DIR / "model_variants.py",
        PROJECT_ROOT / "runtime_python" / "model_variants.py",
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


def _release_adaptive_default_mismatches(
    observed: Mapping[str, Any],
) -> Dict[str, Dict[str, float | None]]:
    mismatches: Dict[str, Dict[str, float | None]] = {}
    for key, required in AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS.items():
        actual = observed.get(key)
        if actual != required:
            mismatches[key] = {"actual": actual, "required": required}
    return mismatches


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
        decision_margin = snapshot.get("prediction_decision_margin")
        require(decision_margin is not None, "prediction_decision_margin_missing")
        require(
            exit_reason
            in {
                "prediction_stable",
                "latent_converged",
                "low_entropy",
                "halt_mass",
                "max_cycles",
            },
            "adaptive_exit_reason_unknown",
        )
        if exit_reason == "prediction_stable":
            require(
                decision_margin is not None
                and float(decision_margin) >= float(PREDICTION_STABILITY_MARGIN),
                "prediction_stable_decision_margin_below_floor",
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
    clock: Callable[[], float] | None = None,
    created_at: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    source_package_parity_pairs: Sequence[tuple[str, Path, Path]] | None = None,
) -> Dict[str, Any]:
    """Run the isolated fixed/adaptive response comparison.

    ``engine_factory`` and ``clock`` are injectable so the protocol can be unit
    tested without constructing a model.  Production callers use the real
    source ``chat_web_app.Engine``.
    """

    weights = Path(weights).expanduser().resolve()
    metadata = Path(metadata).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not metadata.is_file():
        raise FileNotFoundError(f"Metadata not found: {metadata}")
    normalized_prompts = normalize_prompt_matrix(
        FROZEN_RELEASE_PROMPT_MATRIX if prompts is None else prompts
    )
    source = dict(prompt_source or {"origin": "builtin", "path": None, "source_file_sha256": None})
    matrix_observation = _prompt_matrix_release_observation(
        normalized_prompts, source
    )
    source_package_parity = _collect_source_package_parity(
        DEFAULT_SOURCE_PACKAGE_PARITY_PAIRS
        if source_package_parity_pairs is None
        else source_package_parity_pairs
    )
    resolved_device_info = dict(device_info or {"requested": "injected", "resolved": str(device)})
    factory = engine_factory or Engine
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
    engine = factory(device, resolved_device_info, constructor_defaults)
    load_status = engine.load(str(weights), str(metadata))
    if not isinstance(load_status, Mapping) or load_status.get("ok") is not True:
        raise ValueError("Engine did not report a successful load")
    expected_labels = _engine_label_scope(engine, load_status)

    common_kwargs = {
        "style_mode": "auto",
        "response_temperature": 0.0,
        "show_top_responses": TOP_CANDIDATE_COUNT,
        "auto_compute": False,
        "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
    }
    prompt_results: List[Dict[str, Any]] = []
    response_mismatches = 0
    candidate_mismatches = 0
    any_mismatches = 0
    fixed_cycle_violations = 0
    adaptive_cycle_violations = 0
    auto_compute_plan_violations = 0
    contract_violations: List[Dict[str, Any]] = []
    adaptive_runtime_defaults = _effective_adaptive_defaults(engine)
    adaptive_runtime_default_mismatches = _release_adaptive_default_mismatches(
        adaptive_runtime_defaults
    )
    engine_release_rank_depth = _finite_float(
        _runtime_release_rank_depth(engine),
        label="runtime release prediction stability rank depth",
    )

    for index, case in enumerate(normalized_prompts):
        mode_order = ("fixed", "adaptive") if index % 2 == 0 else ("adaptive", "fixed")
        mode_results: Dict[str, Dict[str, Any]] = {}
        for mode in mode_order:
            adaptive = mode == "adaptive"
            session_id = f"v51-response-fidelity-{index + 1:03d}-{mode}"
            started = timer()
            raw = engine.chat(
                session_id=session_id,
                user_text=case["prompt"],
                reasoning_cycles=ADAPTIVE_MAX_CYCLES if adaptive else FIXED_CYCLES,
                adaptive_compute=adaptive,
                **common_kwargs,
            )
            wall_ms = (timer() - started) * 1000.0
            mode_results[mode] = _response_snapshot(
                raw,
                label=f"prompt {case['id']} {mode}",
                wall_ms=wall_ms,
            )

        fixed = mode_results["fixed"]
        adaptive = mode_results["adaptive"]
        fixed_texts = [row["text"] for row in fixed["top_candidates"]]
        adaptive_texts = [row["text"] for row in adaptive["top_candidates"]]
        response_match = fixed["response"] == adaptive["response"]
        candidates_match = fixed_texts == adaptive_texts
        mismatch_kinds: List[str] = []
        if not response_match:
            response_mismatches += 1
            mismatch_kinds.append("response_text")
        if not candidates_match:
            candidate_mismatches += 1
            mismatch_kinds.append("top_candidate_text_order")
        if mismatch_kinds:
            any_mismatches += 1

        fixed_cycles = fixed["compute"].get("cycles_used")
        adaptive_cycles = adaptive["compute"].get("cycles_used")
        if fixed_cycles is None or fixed_cycles != float(FIXED_CYCLES):
            fixed_cycle_violations += 1
        if (
            adaptive_cycles is None
            or adaptive_cycles <= 0.0
            or adaptive_cycles > float(ADAPTIVE_MAX_CYCLES)
        ):
            adaptive_cycle_violations += 1
        auto_compute_plan_violations += int(fixed["auto_compute_plan"] is not None)
        auto_compute_plan_violations += int(adaptive["auto_compute_plan"] is not None)
        for mode, mode_snapshot in (("fixed", fixed), ("adaptive", adaptive)):
            mode_violations = _mode_contract_violations(
                mode_snapshot["compute"],
                mode=mode,
                expected_labels=expected_labels,
                engine_release_rank_depth=engine_release_rank_depth,
                engine_release_defaults=adaptive_runtime_defaults,
            )
            if mode_snapshot["auto_compute_plan"] is not None:
                mode_violations.append("auto_compute_plan_present")
            if mode_violations:
                contract_violations.append(
                    {
                        "prompt_id": case["id"],
                        "mode": mode,
                        "violations": mode_violations,
                    }
                )

        prompt_results.append(
            {
                **case,
                "measurement_order": list(mode_order),
                "fixed": fixed,
                "adaptive": adaptive,
                "comparison": {
                    "response_exact_match": response_match,
                    "top_candidate_text_order_exact_match": candidates_match,
                    "mismatch_kinds": mismatch_kinds,
                },
            }
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
        "source_package_runtime_exact_parity": {
            "passed": source_package_parity["passed"],
            "method": source_package_parity["method"],
            "pair_count": len(source_package_parity["pairs"]),
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
        "authoritative_chat_app_adaptive_runtime_defaults": {
            "passed": not adaptive_runtime_default_mismatches,
            "actual": {
                key: adaptive_runtime_defaults.get(key)
                for key in AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS
            },
            "required": dict(AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS),
            "mismatches": adaptive_runtime_default_mismatches,
        },
        "runtime_release_verifier_contract_observed": {
            "passed": not contract_violations,
            "violation_count": len(contract_violations),
            "violations": contract_violations,
            "required_rank_depth": AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH,
            "required_margin": PREDICTION_STABILITY_MARGIN,
            "required_adaptive_runtime_defaults": dict(
                AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS
            ),
            "required_class_indices": expected_labels,
        },
    }
    all_checks_passed = all(bool(row["passed"]) for row in checks.values())

    fixed_total_timings = [row["fixed"]["timing_ms"]["total"] for row in prompt_results]
    adaptive_total_timings = [row["adaptive"]["timing_ms"]["total"] for row in prompt_results]
    fixed_cycles_values = [row["fixed"]["compute"].get("cycles_used") for row in prompt_results]
    adaptive_cycles_values = [row["adaptive"]["compute"].get("cycles_used") for row in prompt_results]
    fixed_exits = Counter(str(row["fixed"]["compute"].get("exit_reason", "unknown")) for row in prompt_results)
    adaptive_exits = Counter(str(row["adaptive"]["compute"].get("exit_reason", "unknown")) for row in prompt_results)

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
            "statement": "Deterministic regression evidence for this exact hashed prompt matrix and checkpoint only.",
            "held_out_claim": False,
            "universal_chat_fidelity_claim": False,
            "release_eligible": matrix_observation["release_eligible"],
        },
        "checkpoint": {"path": str(weights), "sha256": _sha256_file(weights)},
        "metadata": {"path": str(metadata), "sha256": _sha256_file(metadata)},
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
        "settings": {
            "engine": "source.chat_web_app.Engine",
            "fixed_cycles": FIXED_CYCLES,
            "adaptive_max_cycles": ADAPTIVE_MAX_CYCLES,
            "response_temperature": 0.0,
            "auto_compute": False,
            "prediction_stability_margin": PREDICTION_STABILITY_MARGIN,
            "adaptive_runtime_defaults": adaptive_runtime_defaults,
            "authoritative_adaptive_runtime_defaults": dict(
                AUTHORITATIVE_RELEASE_ADAPTIVE_DEFAULTS
            ),
            "prediction_stability_rank_depth": {
                "selection": "runtime_release_default",
                "authoritative": AUTHORITATIVE_RELEASE_PREDICTION_STABILITY_RANK_DEPTH,
                "resolved": engine_release_rank_depth,
                "explicitly_overridden_by_gate": False,
            },
            "available_label_scope": expected_labels,
            "show_top_responses": TOP_CANDIDATE_COUNT,
            "session_isolation": "unique_empty_session_per_prompt_and_mode",
            "measurement_order": "alternate_fixed_first_and_adaptive_first_by_prompt",
        },
        "load": {
            "ok": True,
            "load_ms": _optional_finite_float(load_status.get("load_ms"), label="load_ms"),
            "model_size": str(load_status.get("model_size", "")),
            "feature_mode": str(load_status.get("feature_mode", "")),
            "available_labels": _optional_finite_float(load_status.get("available_labels"), label="available_labels"),
        },
        "summary": {
            "prompt_count": len(normalized_prompts),
            "response_text_mismatch_count": response_mismatches,
            "top_candidate_text_order_mismatch_count": candidate_mismatches,
            "any_fidelity_mismatch_count": any_mismatches,
            "fixed_total_timing_ms": _timing_summary(fixed_total_timings),
            "adaptive_total_timing_ms": _timing_summary(adaptive_total_timings),
            "fixed_cycles": [
                _optional_finite_float(value, label="fixed cycles summary")
                for value in fixed_cycles_values
            ],
            "adaptive_cycles": [
                _optional_finite_float(value, label="adaptive cycles summary")
                for value in adaptive_cycles_values
            ],
            "fixed_exit_reasons": dict(sorted(fixed_exits.items())),
            "adaptive_exit_reasons": dict(sorted(adaptive_exits.items())),
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
