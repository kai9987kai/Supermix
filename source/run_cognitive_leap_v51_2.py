"""Train and gate a variance-reduced Cognitive Leap v51.2 successor.

The workflow is intentionally split into bounded preparation, training,
development replay, and a separately sealed final phase:

1. ``prepare`` freezes code hashes, train/development/final seeds, search space,
   and bounded evaluation criteria before any candidate is trained.
2. ``train`` fine-tunes independent descendants of canonical v51, builds
   zero-cost checkpoint soups, and selects exactly one candidate on development
   cohorts. The final cohorts are not evaluated in this phase.
3. ``verify-development`` reloads every bound member and independently replays
   the complete dual-comparator candidate matrix, including rejected screens.
4. ``finalize`` consumes the frozen final cohort once and emits a provenance-
   complete bounded receipt. A started sentinel prevents accidental reruns.

This is a synthetic modulo-10 research benchmark. Passing it never grants a
general-chat, general-reasoning, Auto-routing, or default-model claim.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from benchmark_cognitive_leap_ultra_v51 import (  # noqa: E402
    FAMILY_TAG_SCHEMA as BENCHMARK_FAMILY_TAG_SCHEMA,
    GENERATOR_SCHEMA as BENCHMARK_GENERATOR_SCHEMA,
    aux_loss,
    make_chained_task_with_metadata,
    operation_family_tags,
)
from model_variants import ChampionNetCognitiveLeapUltraExpert  # noqa: E402


PROTOCOL_SCHEMA = "supermix-cognitive-leap-bounded-protocol-v2"
SELECTION_SCHEMA = "supermix-cognitive-leap-development-selection-v2"
EVALUATION_SCHEMA = "supermix-cognitive-leap-three-way-evaluation-v1"
THREE_WAY_PREDICTION_ARTIFACT_SCHEMA = (
    "supermix-cognitive-leap-three-way-logits-jsonl-v1"
)
GENERATOR_SCHEMA = BENCHMARK_GENERATOR_SCHEMA
FAMILY_TAG_SCHEMA = BENCHMARK_FAMILY_TAG_SCHEMA
COHORT_SCHEMA = "supermix-cognitive-leap-cohort-v1"
AUTHORITY = {
    "activation": False,
    "auto_route": False,
    "default_model": False,
    "fallback": False,
    "consultant": False,
    "tools": False,
    "permissions": False,
    "safety": False,
    "promotion": False,
    "store_publication": False,
    "release": False,
}
BASELINE_RELATIVE = Path(
    "output/benchmark_v51_cognitive_leap_ultra_latest/"
    "cognitive_leap_ultra_v51_trained.pth"
)
PRIOR_CANDIDATE_RELATIVE = Path(
    "output/training_candidates/"
    "cognitive_leap_ultra_v51_1_balanced_blend30_seed151/"
    "cognitive_leap_ultra_v51_1_balanced_blend30.pth"
)
CANONICAL_BASELINE_FILE_SHA256 = (
    "664b1779452fe1482389413004d8bce3369f6d8ee15ab8c2c891dc5e382ebae4"
)
CANONICAL_BASELINE_STATE_SHA256 = (
    "bed39f133c710e718aab7d7de387b42890ee0767fbbe70e8cc626b2d0d56ede5"
)
PRIOR_CANDIDATE_FILE_SHA256 = (
    "c627d905951fbfefa8155a9aae064d04fcc574cb8464f08fc716947422de06cb"
)
PRIOR_CANDIDATE_STATE_SHA256 = (
    "9850e8b7595795667642294049fc2394771a52997390b961c2624c78e41bf1a0"
)
DEFAULT_OUTPUT_RELATIVE = Path(
    "output/training_candidates/cognitive_leap_ultra_v51_2_variance_soup"
)
DEV_SEEDS = tuple(range(21_052, 61_052, 1_000))
FINAL_SEEDS = tuple(range(101_052, 121_052, 1_000))
CRITERIA: dict[str, Any] = {
    "minimum_accuracy_gain": 0.002,
    "maximum_p_value": 0.05,
    "minimum_nonregressing_seed_fraction": 0.80,
    "minimum_worst_seed_delta": -0.01,
    "minimum_nonregressing_operation_families": 8,
    "minimum_worst_operation_family_delta": -0.005,
    "minimum_nonregressing_classes": 7,
    "minimum_worst_class_delta": -0.03,
    "require_mean_loss_nonregression": True,
}
DEVELOPMENT_CRITERIA: dict[str, Any] = {
    **CRITERIA,
    "minimum_accuracy_gain": 0.003,
    "minimum_worst_seed_delta": -0.005,
    "minimum_worst_operation_family_delta": 0.0,
    "minimum_nonregressing_classes": 8,
    "minimum_worst_class_delta": -0.02,
}
PRIOR_CANDIDATE_CRITERIA: dict[str, Any] = {
    **CRITERIA,
    # Ten net-correct examples on the fixed 40,000-example final cohort.
    "minimum_accuracy_gain": 0.00025,
}
MEMBER_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "tempered_251",
        "train_seed": 251,
        "shuffle_seed": 5_251,
        "dropout_seed": 6_251,
        "lr": 7.5e-5,
        "balance_exponent": 0.25,
    },
    {
        "name": "tempered_351",
        "train_seed": 351,
        "shuffle_seed": 5_351,
        "dropout_seed": 6_351,
        "lr": 7.5e-5,
        "balance_exponent": 0.25,
    },
)
SOUP_GROUPS: tuple[tuple[str, ...], ...] = (
    ("tempered_251", "tempered_351"),
)
BASE_BLEND_ALPHAS = (0.20, 0.25, 0.30)
EVALUATION_PROFILE_SCHEMA = "supermix-cognitive-leap-evaluation-profile-v1"
SELECTION_ORDER = (
    "both_gates_pass",
    "total_check_count",
    "prior_candidate_accuracy_delta",
    "release_accuracy_delta",
    "combined_nonregression_counts",
    "negative_candidate_loss",
)
CLAIM_SCOPE = {
    "task": "four-operation chained modular arithmetic modulo 10",
    "general_chat_claim": False,
    "general_reasoning_claim": False,
    "production_default_claim": False,
    "auto_route_claim": False,
    "manual_activation_claim": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_evaluation_profile() -> dict[str, Any]:
    """Return the immutable three-way development/final evaluation policy."""

    return {
        "schema": EVALUATION_PROFILE_SCHEMA,
        "task_schemas": {
            "generator": GENERATOR_SCHEMA,
            "family_tags": FAMILY_TAG_SCHEMA,
            "cohort": COHORT_SCHEMA,
        },
        "evaluator": {
            "model_class": "ChampionNetCognitiveLeapUltraExpert",
            "reasoning_cycles": 3,
            "logits_dtype": "torch.float32",
            "argmax_tie_rule": "lowest_class_index",
        },
        "release_baseline": {
            "path": BASELINE_RELATIVE.as_posix(),
            "size_bytes": 9_016_017,
            "sha256": CANONICAL_BASELINE_FILE_SHA256,
            "canonical_state_sha256": CANONICAL_BASELINE_STATE_SHA256,
            "tensor_count": 100,
            "element_count": 2_245_719,
        },
        "prior_candidate": {
            "status": "unpromoted_prior_candidate",
            "path": PRIOR_CANDIDATE_RELATIVE.as_posix(),
            "size_bytes": 9_017_183,
            "sha256": PRIOR_CANDIDATE_FILE_SHA256,
            "canonical_state_sha256": PRIOR_CANDIDATE_STATE_SHA256,
            "tensor_count": 100,
            "element_count": 2_245_719,
        },
        "training": {
            "train_size_per_member": 12_000,
            "epochs": 1,
            "batch_size": 128,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "members": [dict(config) for config in MEMBER_CONFIGS],
        },
        "development": {
            "seeds": list(DEV_SEEDS),
            "samples_per_seed": 2_000,
            "soup_groups": [list(group) for group in SOUP_GROUPS],
            "baseline_blend_alphas": list(BASE_BLEND_ALPHAS),
            "selection_order": list(SELECTION_ORDER),
            "release_continuity_criteria": dict(DEVELOPMENT_CRITERIA),
            "prior_candidate_superiority_criteria": dict(
                PRIOR_CANDIDATE_CRITERIA
            ),
        },
        "final": {
            "seeds": list(FINAL_SEEDS),
            "samples_per_seed": 2_000,
            "single_use": True,
            "release_continuity_criteria": dict(CRITERIA),
            "prior_candidate_superiority_criteria": dict(
                PRIOR_CANDIDATE_CRITERIA
            ),
            "overall_gate": "logical_and",
        },
        "claim_scope": dict(CLAIM_SCOPE),
        "authority": dict(AUTHORITY),
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "trusted_timestamp": False,
    }


def canonical_evaluation_profile_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(canonical_evaluation_profile()))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_json_exclusive(path: Path, value: Any) -> None:
    """Create a JSON sentinel atomically and fail if another process won."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key is forbidden: {key}")
        value[key] = item
    return value


def load_json_strict(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def resolve_repo_relative(value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"Repository artifact path must be relative: {value}")
    resolved = (REPO_ROOT / relative).resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Repository artifact path escapes the tree: {value}") from error
    return resolved


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _run_git_optional(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_binding() -> dict[str, Any]:
    status = _run_git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
        "status_line_count": len(status.splitlines()) if status else 0,
        "timestamps_trusted": False,
    }


def current_code_bindings() -> dict[str, dict[str, Any]]:
    paths = (
        (Path(__file__).resolve(), ("main", "finalize_once")),
        (
            SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py",
            ("make_chained_task_with_metadata", "aux_loss"),
        ),
        (SOURCE_DIR / "model_variants.py", ("ChampionNetCognitiveLeapUltraExpert",)),
        (SOURCE_DIR / "run.py", ("ChampionNet", "GatedFFN")),
        (SOURCE_DIR / "device_utils.py", ("resolve_device",)),
        (
            SOURCE_DIR / "cognitive_leap_three_way_receipt.py",
            ("validate_receipt", "try_validate_receipt"),
        ),
    )
    records: dict[str, dict[str, Any]] = {}
    for path, symbols in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required evaluator source is missing: {path}")
        relative_name = relative_path(path).replace("\\", "/")
        records[relative_name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "symbols": list(symbols),
            "worktree_git_blob_sha1": _run_git("hash-object", relative_name),
            "head_git_blob_sha1": _run_git_optional(
                "rev-parse",
                f"HEAD:{relative_name}",
            ),
        }
    return records


def snapshot_bound_sources(
    output_dir: Path,
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    snapshot_root = output_dir / "source_snapshot"
    records: dict[str, Any] = {}
    for relative_name, binding in bindings.items():
        expected_sha256 = str(binding["sha256"])
        source_path = REPO_ROOT / relative_name
        target_path = snapshot_root / relative_name
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(source_path.read_bytes())
        observed = sha256_file(target_path)
        if observed != expected_sha256:
            raise ValueError(f"Source snapshot hash mismatch for {relative_name}")
        records[relative_name] = {
            "path": relative_path(target_path),
            "sha256": observed,
            "size_bytes": target_path.stat().st_size,
        }
    return records


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def selection_digest(selection: Mapping[str, Any]) -> str:
    payload = dict(selection)
    payload.pop("selection_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def tensor_digest_update(digest: Any, label: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    if array.dtype.itemsize > 1:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(canonical_json_bytes(list(value.shape)) + b"\0")
    digest.update(array.tobytes(order="C"))


def state_dict_summary(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    finite = True
    elements = 0
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        tensor_digest_update(digest, name, tensor)
        elements += int(tensor.numel())
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            finite = False
    return {
        "tensor_count": len(state),
        "element_count": elements,
        "all_finite": finite,
        "tensor_byte_order": "little_endian",
        "canonical_state_sha256": digest.hexdigest(),
    }


def state_dict_inventory(state: Mapping[str, torch.Tensor]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest = hashlib.sha256()
        tensor_digest_update(digest, name, tensor)
        inventory.append(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "element_count": int(tensor.numel()),
                "canonical_tensor_sha256": digest.hexdigest(),
            }
        )
    return inventory


def load_state(path: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError(f"Checkpoint is not a nonempty state dict: {path}")
    if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        loaded = loaded["state_dict"]
    state = {str(name): tensor.detach().cpu() for name, tensor in loaded.items()}
    model = ChampionNetCognitiveLeapUltraExpert()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Strict-load mismatch for {path}: {incompatible}")
    summary = state_dict_summary(state)
    if not summary["all_finite"]:
        raise ValueError(f"Checkpoint contains nonfinite values: {path}")
    return state


def save_state(path: Path, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({key: value.detach().cpu() for key, value in state.items()}, temporary)
    temporary.replace(path)
    strict = load_state(path)
    return {
        "path": relative_path(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        **state_dict_summary(strict),
        "tensor_inventory": state_dict_inventory(strict),
        "strict_load": True,
    }


def average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("At least one state dict is required")
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ValueError("State dict key/order mismatch")
    source_weights = (1.0,) * len(states) if weights is None else weights
    raw_weights = tuple(float(value) for value in source_weights)
    if (
        len(raw_weights) != len(states)
        or any(not math.isfinite(value) or value < 0 for value in raw_weights)
    ):
        raise ValueError("Invalid state weights")
    total = sum(raw_weights)
    if total <= 0:
        raise ValueError("State weights must have positive mass")
    normalized = tuple(value / total for value in raw_weights)
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key].detach().cpu() for state in states]
        first = tensors[0]
        if any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in tensors[1:]
        ):
            raise ValueError(f"State dtype/shape differs for {key}")
        if any(
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all().item())
            for tensor in tensors
        ):
            raise ValueError(f"State contains nonfinite values for {key}")
        if first.is_floating_point() or first.is_complex():
            mixed = torch.zeros_like(first)
            for weight, tensor in zip(normalized, tensors):
                mixed.add_(tensor.to(dtype=first.dtype), alpha=weight)
            result[key] = mixed
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating state differs for {key}")
            result[key] = first.clone()
    return result


def blend_with_baseline(
    baseline: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    alpha: float,
) -> dict[str, torch.Tensor]:
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("Blend alpha must be in [0, 1]")
    return average_states((baseline, candidate), (1.0 - value, value))


def _criterion_fraction(value: Any) -> Fraction:
    """Interpret a JSON numeric criterion as its exact decimal rational value."""

    if isinstance(value, bool):
        raise ValueError("Boolean values are not numeric criteria")
    try:
        result = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid numeric criterion: {value!r}") from exc
    return result


def _ratio_at_least(numerator: int, denominator: int, threshold: Any) -> bool:
    """Compare a count-derived ratio without first rounding it to binary float."""

    if int(denominator) <= 0:
        raise ValueError("Ratio denominator must be positive")
    bound = _criterion_fraction(threshold)
    return int(numerator) * bound.denominator >= bound.numerator * int(denominator)


def _ratio_at_most(numerator: int, denominator: int, threshold: Any) -> bool:
    if int(denominator) <= 0:
        raise ValueError("Ratio denominator must be positive")
    bound = _criterion_fraction(threshold)
    return int(numerator) * bound.denominator <= bound.numerator * int(denominator)


def _ceil_scaled_fraction(count: int, fraction: Any) -> int:
    bound = _criterion_fraction(fraction)
    numerator = int(count) * bound.numerator
    return -(-numerator // bound.denominator)


def _exact_mcnemar_terms(wins: int, regressions: int) -> tuple[int, int]:
    discordant = int(wins) + int(regressions)
    if discordant <= 0:
        return 1, 1
    tail = min(int(wins), int(regressions))
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    denominator = 1 << discordant
    return (denominator, denominator) if numerator >= denominator else (numerator, denominator)


def exact_mcnemar_two_sided(wins: int, regressions: int) -> float:
    numerator, denominator = _exact_mcnemar_terms(wins, regressions)
    return numerator / denominator


def cohort_specification(
    seeds: Sequence[int],
    samples_per_seed: int,
    *,
    cohort_role: str,
) -> dict[str, Any]:
    ordered_seeds = tuple(int(seed) for seed in seeds)
    if not ordered_seeds or len(set(ordered_seeds)) != len(ordered_seeds):
        raise ValueError("Cohort seeds must be nonempty and unique")
    if int(samples_per_seed) <= 0:
        raise ValueError("Cohort samples_per_seed must be positive")
    if cohort_role not in {"development", "final"}:
        raise ValueError("Unsupported cohort role")
    return {
        "schema": COHORT_SCHEMA,
        "generator_schema": GENERATOR_SCHEMA,
        "family_tag_schema": FAMILY_TAG_SCHEMA,
        "cohort_role": cohort_role,
        "seeds": list(ordered_seeds),
        "samples_per_seed": int(samples_per_seed),
        "generator_source_sha256": sha256_file(
            SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
        ),
    }


def build_cohort(
    seeds: Sequence[int],
    samples_per_seed: int,
    *,
    cohort_role: str,
) -> dict[str, Any]:
    specification = cohort_specification(
        seeds,
        samples_per_seed,
        cohort_role=cohort_role,
    )
    ordered_seeds = tuple(int(seed) for seed in specification["seeds"])
    specification_sha256 = sha256_bytes(canonical_json_bytes(specification))
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for seed in ordered_seeds:
        x, y, metadata = make_chained_task_with_metadata(samples_per_seed, int(seed))
        digest.update(struct.pack("<q", int(seed)))
        tensor_digest_update(digest, "x", x)
        tensor_digest_update(digest, "y", y)
        for name in ("starts", "op_types", "operands"):
            tensor_digest_update(digest, name, metadata[name])
        rows.append({"seed": int(seed), "x": x, "y": y, **metadata})
    return {
        "schema": COHORT_SCHEMA,
        "generator_schema": GENERATOR_SCHEMA,
        "family_tag_schema": FAMILY_TAG_SCHEMA,
        "cohort_role": cohort_role,
        "seeds": list(ordered_seeds),
        "samples_per_seed": int(samples_per_seed),
        "n": len(ordered_seeds) * int(samples_per_seed),
        "specification": specification,
        "specification_sha256": specification_sha256,
        "dataset_sha256": digest.hexdigest(),
        "dataset_id": sha256_bytes(
            canonical_json_bytes(
                {
                    "specification_sha256": specification_sha256,
                    "dataset_sha256": digest.hexdigest(),
                }
            )
        ),
        "rows": rows,
    }


@torch.no_grad()
def predict_cohort(
    model: ChampionNetCognitiveLeapUltraExpert,
    cohort: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    prediction_digest = hashlib.sha256()
    logits_digest = hashlib.sha256()
    per_example_digest = hashlib.sha256()
    seed_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total = 0
    for row in cohort["rows"]:
        x = row["x"].to(device)
        y = row["y"].to(device)
        logits = model(x, reasoning_cycles=3).squeeze(1)
        logits_cpu = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
        if logits_cpu.shape != (int(y.numel()), 10):
            raise ValueError(f"Unexpected evaluator logit shape: {tuple(logits_cpu.shape)}")
        if not bool(torch.isfinite(logits_cpu).all().item()):
            raise ValueError("Evaluator produced nonfinite logits")
        predictions = logits_cpu.argmax(dim=-1)
        loss_sum = float(F.cross_entropy(logits, y, reduction="sum").item())
        prediction_digest.update(struct.pack("<q", int(row["seed"])))
        tensor_digest_update(prediction_digest, "targets", row["y"])
        tensor_digest_update(prediction_digest, "predictions", predictions)
        logits_digest.update(struct.pack("<q", int(row["seed"])))
        tensor_digest_update(logits_digest, "logits_f32", logits_cpu)
        for index in range(int(row["y"].numel())):
            target = int(row["y"][index].item())
            prediction = int(predictions[index].item())
            per_example_digest.update(
                canonical_json_bytes(
                    {
                        "seed": int(row["seed"]),
                        "index": index,
                        "target": target,
                        "prediction": prediction,
                        "correct": prediction == target,
                        "logits_f32le_hex": struct.pack(
                            "<10f",
                            *[float(value) for value in logits_cpu[index].tolist()],
                        ).hex(),
                    }
                )
                + b"\n"
            )
        seed_rows.append(
            {
                "seed": int(row["seed"]),
                "targets": row["y"].clone(),
                "predictions": predictions,
                "logits": logits_cpu,
                "loss_sum": loss_sum,
            }
        )
        total_loss += loss_sum
        total += int(y.numel())
    return {
        "mean_loss": total_loss / max(1, total),
        "prediction_sha256": prediction_digest.hexdigest(),
        "logits_sha256": logits_digest.hexdigest(),
        "per_example_sha256": per_example_digest.hexdigest(),
        "seed_rows": seed_rows,
    }


def _group_correct_count(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, int]:
    count = int(mask.sum().item())
    if count <= 0:
        return 0, 0
    correct = int(predictions[mask].eq(targets[mask]).sum().item())
    return correct, count


def compare_predictions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    criteria: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_by_seed = {row["seed"]: row for row in baseline["seed_rows"]}
    candidate_by_seed = {row["seed"]: row for row in candidate["seed_rows"]}
    seed_rows: list[dict[str, Any]] = []
    all_targets: list[torch.Tensor] = []
    all_baseline: list[torch.Tensor] = []
    all_candidate: list[torch.Tensor] = []
    all_ops: list[torch.Tensor] = []
    paired_digest = hashlib.sha256()
    wins = regressions = ties = 0
    for cohort_row in cohort["rows"]:
        seed = int(cohort_row["seed"])
        reference = baseline_by_seed[seed]
        tuned = candidate_by_seed[seed]
        targets = reference["targets"]
        reference_predictions = reference["predictions"]
        tuned_predictions = tuned["predictions"]
        reference_correct = reference_predictions.eq(targets)
        tuned_correct = tuned_predictions.eq(targets)
        row_wins = int((tuned_correct & ~reference_correct).sum().item())
        row_regressions = int((reference_correct & ~tuned_correct).sum().item())
        row_ties = int(targets.numel()) - row_wins - row_regressions
        row_n = int(targets.numel())
        reference_correct_count = int(reference_correct.sum().item())
        tuned_correct_count = int(tuned_correct.sum().item())
        reference_accuracy = reference_correct_count / max(1, row_n)
        tuned_accuracy = tuned_correct_count / max(1, row_n)
        seed_rows.append(
            {
                "seed": seed,
                "n": int(targets.numel()),
                "baseline_accuracy": reference_accuracy,
                "candidate_accuracy": tuned_accuracy,
                "accuracy_delta": (tuned_correct_count - reference_correct_count)
                / max(1, row_n),
                "baseline_loss": reference["loss_sum"] / max(1, int(targets.numel())),
                "candidate_loss": tuned["loss_sum"] / max(1, int(targets.numel())),
                "wins": row_wins,
                "regressions": row_regressions,
                "ties": row_ties,
            }
        )
        outcome = tuned_correct.to(torch.int8) - reference_correct.to(torch.int8)
        paired_digest.update(struct.pack("<q", seed))
        tensor_digest_update(paired_digest, "paired_outcome", outcome)
        wins += row_wins
        regressions += row_regressions
        ties += row_ties
        all_targets.append(targets)
        all_baseline.append(reference_predictions)
        all_candidate.append(tuned_predictions)
        all_ops.append(cohort_row["op_types"])

    targets = torch.cat(all_targets)
    reference_predictions = torch.cat(all_baseline)
    tuned_predictions = torch.cat(all_candidate)
    op_types = torch.cat(all_ops)
    reference_correct = reference_predictions.eq(targets)
    tuned_correct = tuned_predictions.eq(targets)
    total_n = int(targets.numel())
    baseline_correct_count = int(reference_correct.sum().item())
    candidate_correct_count = int(tuned_correct.sum().item())
    baseline_accuracy = baseline_correct_count / max(1, total_n)
    candidate_accuracy = candidate_correct_count / max(1, total_n)

    operation_family_rows: list[dict[str, Any]] = []
    operation_family_deltas: list[tuple[int, int]] = []
    op_names = ("add", "mul", "sub")
    for op_index, op_name in enumerate(op_names):
        mask = op_types[:, 0].eq(op_index)
        base_correct, count = _group_correct_count(targets, reference_predictions, mask)
        tuned_correct_count, tuned_count = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if tuned_count != count:
            raise ValueError("Operation-family comparison count mismatch")
        delta_count = tuned_correct_count - base_correct
        base_value = base_correct / count if count else 0.0
        tuned_value = tuned_correct_count / count if count else 0.0
        operation_family_rows.append(
            {
                "family": f"first_{op_name}",
                "n": count,
                "baseline_accuracy": base_value,
                "candidate_accuracy": tuned_value,
                "delta": delta_count / count if count else 0.0,
            }
        )
        operation_family_deltas.append((delta_count, count))
    multiplication_counts = op_types.eq(1).sum(dim=1)
    for count in range(5):
        mask = multiplication_counts.eq(count)
        base_correct, group_n = _group_correct_count(
            targets, reference_predictions, mask
        )
        tuned_correct_count, tuned_n = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if tuned_n != group_n:
            raise ValueError("Operation-family comparison count mismatch")
        delta_count = tuned_correct_count - base_correct
        base_value = base_correct / group_n if group_n else 0.0
        tuned_value = tuned_correct_count / group_n if group_n else 0.0
        operation_family_rows.append(
            {
                "family": f"mul_count_{count}",
                "n": group_n,
                "baseline_accuracy": base_value,
                "candidate_accuracy": tuned_value,
                "delta": delta_count / group_n if group_n else 0.0,
            }
        )
        operation_family_deltas.append((delta_count, group_n))

    class_rows: list[dict[str, Any]] = []
    class_deltas: list[tuple[int, int]] = []
    for class_index in range(10):
        mask = targets.eq(class_index)
        base_correct, count = _group_correct_count(targets, reference_predictions, mask)
        tuned_correct_count, tuned_count = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if tuned_count != count:
            raise ValueError("Class comparison count mismatch")
        delta_count = tuned_correct_count - base_correct
        base_value = base_correct / count if count else 0.0
        tuned_value = tuned_correct_count / count if count else 0.0
        class_rows.append(
            {
                "class": str(class_index),
                "n": count,
                "baseline_accuracy": base_value,
                "candidate_accuracy": tuned_value,
                "delta": delta_count / count if count else 0.0,
            }
        )
        class_deltas.append((delta_count, count))

    seed_deltas = [
        (int(row["wins"]) - int(row["regressions"]), int(row["n"]))
        for row in seed_rows
    ]
    nonregressing_seed_count = sum(delta >= 0 for delta, _count in seed_deltas)
    required_seed_count = _ceil_scaled_fraction(
        len(seed_rows), criteria["minimum_nonregressing_seed_fraction"]
    )
    eligible_family_pairs = [
        (row, exact)
        for row, exact in zip(operation_family_rows, operation_family_deltas)
        if exact[1] > 0
    ]
    eligible_class_pairs = [
        (row, exact)
        for row, exact in zip(class_rows, class_deltas)
        if exact[1] > 0
    ]
    eligible_family_rows = [row for row, _exact in eligible_family_pairs]
    eligible_class_rows = [row for row, _exact in eligible_class_pairs]
    eligible_family_deltas = [exact for _row, exact in eligible_family_pairs]
    eligible_class_deltas = [exact for _row, exact in eligible_class_pairs]
    nonregressing_family_count = sum(
        delta >= 0 for delta, _count in eligible_family_deltas
    )
    nonregressing_class_count = sum(
        delta >= 0 for delta, _count in eligible_class_deltas
    )
    mcnemar_numerator, mcnemar_denominator = _exact_mcnemar_terms(wins, regressions)
    summary = {
        "seed_count": len(seed_rows),
        "n": int(targets.numel()),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": (candidate_correct_count - baseline_correct_count)
        / max(1, total_n),
        "wins": wins,
        "regressions": regressions,
        "ties": ties,
        "exact_mcnemar_p_two_sided": exact_mcnemar_two_sided(wins, regressions),
        "mean_baseline_loss": float(baseline["mean_loss"]),
        "mean_candidate_loss": float(candidate["mean_loss"]),
        "required_nonregressing_seed_count": required_seed_count,
        "nonregressing_seed_count": nonregressing_seed_count,
        "worst_seed_delta": min(row["accuracy_delta"] for row in seed_rows),
        "nonregressing_family_count": nonregressing_family_count,
        "eligible_family_count": len(eligible_family_rows),
        "worst_family_delta": min(row["delta"] for row in eligible_family_rows),
        "nonregressing_class_count": nonregressing_class_count,
        "eligible_class_count": len(eligible_class_rows),
        "worst_class_delta": min(row["delta"] for row in eligible_class_rows),
    }
    checks = {
        "accuracy_gain": _ratio_at_least(
            candidate_correct_count - baseline_correct_count,
            total_n,
            criteria["minimum_accuracy_gain"],
        ),
        "paired_significance": _ratio_at_most(
            mcnemar_numerator,
            mcnemar_denominator,
            criteria["maximum_p_value"],
        ),
        "mean_loss_nonregression": (
            not bool(criteria["require_mean_loss_nonregression"])
            or summary["mean_candidate_loss"] <= summary["mean_baseline_loss"]
        ),
        "seed_nonregression": (
            summary["nonregressing_seed_count"] >= required_seed_count
            and all(
                _ratio_at_least(delta, count, criteria["minimum_worst_seed_delta"])
                for delta, count in seed_deltas
            )
        ),
        "operation_family_nonregression": (
            summary["nonregressing_family_count"]
            >= int(criteria["minimum_nonregressing_operation_families"])
            and all(
                _ratio_at_least(
                    delta,
                    count,
                    criteria["minimum_worst_operation_family_delta"],
                )
                for delta, count in eligible_family_deltas
            )
        ),
        "class_bounded_nonregression": (
            summary["nonregressing_class_count"]
            >= int(criteria["minimum_nonregressing_classes"])
            and all(
                _ratio_at_least(delta, count, criteria["minimum_worst_class_delta"])
                for delta, count in eligible_class_deltas
            )
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "seed_rows": seed_rows,
        "operation_family_rows": operation_family_rows,
        "class_rows": class_rows,
        "evidence": {
            "cohort_schema": cohort["schema"],
            "generator_schema": cohort["generator_schema"],
            "family_tag_schema": cohort["family_tag_schema"],
            "cohort_role": cohort["cohort_role"],
            "dataset_id": cohort["dataset_id"],
            "dataset_specification_sha256": cohort["specification_sha256"],
            "dataset_sha256": cohort["dataset_sha256"],
            "baseline_prediction_sha256": baseline["prediction_sha256"],
            "candidate_prediction_sha256": candidate["prediction_sha256"],
            "baseline_logits_sha256": baseline["logits_sha256"],
            "candidate_logits_sha256": candidate["logits_sha256"],
            "baseline_per_example_sha256": baseline["per_example_sha256"],
            "candidate_per_example_sha256": candidate["per_example_sha256"],
            "paired_outcome_sha256": paired_digest.hexdigest(),
        },
    }


def environment_binding(device: torch.device) -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip().lower().replace("_", "-")
        if name:
            dependencies[name] = str(distribution.version)
    dependency_rows = [
        {"name": name, "version": dependencies[name]}
        for name in sorted(dependencies)
    ]
    executable = Path(sys.executable).resolve()
    torch_build = torch.__config__.show()
    critical_records: dict[str, Any] = {}
    for distribution_name in ("torch", "numpy"):
        try:
            distribution = importlib.metadata.distribution(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            critical_records[distribution_name] = {"status": "missing"}
            continue
        record_paths = [
            distribution.locate_file(entry)
            for entry in (distribution.files or ())
            if str(entry).replace("\\", "/").endswith(".dist-info/RECORD")
        ]
        if len(record_paths) != 1 or not Path(record_paths[0]).is_file():
            critical_records[distribution_name] = {
                "version": distribution.version,
                "record_status": "missing_or_ambiguous",
            }
        else:
            record_path = Path(record_paths[0])
            critical_records[distribution_name] = {
                "version": distribution.version,
                "record_sha256": sha256_file(record_path),
                "record_size_bytes": record_path.stat().st_size,
            }
    platform_binding = {
        "description": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "byteorder": sys.byteorder,
    }
    return {
        "authentication": "none",
        "timestamps_trusted": False,
        "host_identity_trusted": False,
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "executable_name": executable.name,
            "executable_sha256": sha256_file(executable),
        },
        "dependencies": dependency_rows,
        "dependency_lock_sha256": sha256_bytes(canonical_json_bytes(dependency_rows)),
        "critical_distribution_records": critical_records,
        "platform": platform_binding,
        "host_binding_sha256": sha256_bytes(canonical_json_bytes(platform_binding)),
        "torch": {
            "version": torch.__version__,
            "build_config_sha256": sha256_bytes(torch_build.encode("utf-8")),
            "device": str(device),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "initial_seed": torch.initial_seed(),
            "mkldnn_enabled": torch.backends.mkldnn.enabled,
            "cudnn_enabled": torch.backends.cudnn.enabled,
        },
        "rng": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "member_seeds_are_protocol_bound": True,
            "development_and_final_generators_are_seed_local": True,
            "environment_controls": {
                name: os.environ.get(name)
                for name in (
                    "PYTHONHASHSEED",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "CUBLAS_WORKSPACE_CONFIG",
                )
            },
        },
        "invocation": {
            "argv": [executable.name, *sys.argv],
            "working_tree": ".",
        },
    }


def validate_environment_compatibility(
    frozen: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    for key in (
        "authentication",
        "python",
        "dependencies",
        "dependency_lock_sha256",
        "critical_distribution_records",
        "platform",
        "host_binding_sha256",
        "rng",
    ):
        if frozen.get(key) != current.get(key):
            raise ValueError(f"Execution environment changed after protocol freeze: {key}")
    frozen_torch = dict(frozen.get("torch", {}))
    current_torch = dict(current.get("torch", {}))
    frozen_torch.pop("initial_seed", None)
    current_torch.pop("initial_seed", None)
    if frozen_torch != current_torch:
        raise ValueError("Torch execution environment changed after protocol freeze")


def prepare_protocol(output_dir: Path, args: argparse.Namespace) -> Path:
    baseline_path = (REPO_ROOT / BASELINE_RELATIVE).resolve()
    prior_candidate_path = (REPO_ROOT / PRIOR_CANDIDATE_RELATIVE).resolve()
    if not baseline_path.is_file():
        raise FileNotFoundError(f"Canonical baseline is missing: {baseline_path}")
    if not prior_candidate_path.is_file():
        raise FileNotFoundError(
            f"Exact prior v51.1 candidate is missing: {prior_candidate_path}"
        )
    if (
        int(args.train_size) != 12_000
        or int(args.epochs) != 1
        or int(args.batch_size) != 128
        or int(args.samples_per_seed) != 2_000
    ):
        raise ValueError(
            "v51.2 uses the immutable evaluation profile: train-size=12000, "
            "epochs=1, batch-size=128, samples-per-seed=2000"
        )
    baseline_state = load_state(baseline_path)
    prior_candidate_state = load_state(prior_candidate_path)
    baseline_summary = state_dict_summary(baseline_state)
    prior_candidate_summary = state_dict_summary(prior_candidate_state)
    if (
        baseline_path.stat().st_size != 9_016_017
        or sha256_file(baseline_path) != CANONICAL_BASELINE_FILE_SHA256
        or baseline_summary["canonical_state_sha256"]
        != CANONICAL_BASELINE_STATE_SHA256
    ):
        raise ValueError("Canonical v51 release baseline identity mismatch")
    if (
        prior_candidate_path.stat().st_size != 9_017_183
        or sha256_file(prior_candidate_path) != PRIOR_CANDIDATE_FILE_SHA256
        or prior_candidate_summary["canonical_state_sha256"]
        != PRIOR_CANDIDATE_STATE_SHA256
    ):
        raise ValueError("Exact prior v51.1 candidate identity mismatch")
    protocol_path = output_dir / "protocol.json"
    if protocol_path.exists():
        raise FileExistsError(f"Protocol already exists: {protocol_path}")
    git = git_binding()
    if git["dirty"] and not bool(args.allow_dirty_development):
        raise RuntimeError(
            "Protocol preparation requires a clean Git worktree; use "
            "--allow-dirty-development only for a development run that can never finalize"
        )
    execution_mode = "dirty_development" if git["dirty"] else "clean_final_eligible"
    code_bindings = current_code_bindings()
    source_snapshot = snapshot_bound_sources(output_dir, code_bindings)
    training_seeds = {int(config["train_seed"]) for config in MEMBER_CONFIGS}
    if (
        set(DEV_SEEDS) & set(FINAL_SEEDS)
        or training_seeds & (set(DEV_SEEDS) | set(FINAL_SEEDS))
    ):
        raise ValueError("Training, development, and final cohorts must be disjoint")
    development_specification = cohort_specification(
        DEV_SEEDS,
        int(args.samples_per_seed),
        cohort_role="development",
    )
    final_specification = cohort_specification(
        FINAL_SEEDS,
        int(args.samples_per_seed),
        cohort_role="final",
    )
    evaluation_profile = canonical_evaluation_profile()
    evaluation_profile_sha256 = canonical_evaluation_profile_sha256()
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "created_at": utc_now(),
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "evaluation_profile": evaluation_profile,
        "evaluation_profile_sha256": evaluation_profile_sha256,
        "task_schemas": {
            "generator": GENERATOR_SCHEMA,
            "family_tags": FAMILY_TAG_SCHEMA,
            "cohort": COHORT_SCHEMA,
        },
        "execution_mode": execution_mode,
        "finalization_allowed": execution_mode == "clean_final_eligible",
        "claim_scope": dict(CLAIM_SCOPE),
        "baseline": {
            "path": relative_path(baseline_path),
            "size_bytes": baseline_path.stat().st_size,
            "sha256": sha256_file(baseline_path),
            **state_dict_summary(baseline_state),
            "tensor_inventory": state_dict_inventory(baseline_state),
            "strict_load": True,
        },
        "prior_candidate": {
            "status": "unpromoted_prior_candidate",
            "path": relative_path(prior_candidate_path),
            "size_bytes": prior_candidate_path.stat().st_size,
            "sha256": sha256_file(prior_candidate_path),
            **prior_candidate_summary,
            "tensor_inventory": state_dict_inventory(prior_candidate_state),
            "strict_load": True,
        },
        "training": {
            "train_size_per_member": int(args.train_size),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "members": [dict(config) for config in MEMBER_CONFIGS],
        },
        "development": {
            "seeds": list(DEV_SEEDS),
            "samples_per_seed": int(args.samples_per_seed),
            "soup_groups": [list(group) for group in SOUP_GROUPS],
            "baseline_blend_alphas": list(BASE_BLEND_ALPHAS),
            "selection_order": list(SELECTION_ORDER),
            "criteria": dict(DEVELOPMENT_CRITERIA),
            "prior_candidate_criteria": dict(PRIOR_CANDIDATE_CRITERIA),
            "cohort_specification": development_specification,
            "cohort_specification_sha256": sha256_bytes(
                canonical_json_bytes(development_specification)
            ),
        },
        "final": {
            "seeds": list(FINAL_SEEDS),
            "samples_per_seed": int(args.samples_per_seed),
            "single_use": True,
            "cohort_specification": final_specification,
            "cohort_specification_sha256": sha256_bytes(
                canonical_json_bytes(final_specification)
            ),
        },
        "criteria": dict(CRITERIA),
        "prior_candidate_criteria": dict(PRIOR_CANDIDATE_CRITERIA),
        "code_bindings": code_bindings,
        "source_snapshot": source_snapshot,
        "git": git,
        "environment_at_freeze": environment_binding(torch.device("cpu")),
        "prepare_invocation": {
            "argv": [Path(sys.executable).name, *sys.argv],
            "working_tree": ".",
        },
    }
    protocol["protocol_sha256"] = protocol_digest(protocol)
    write_json_atomic(protocol_path, protocol)
    return protocol_path


def validate_canonical_evaluation_profile(protocol: Mapping[str, Any]) -> None:
    expected = canonical_evaluation_profile()
    expected_sha256 = canonical_evaluation_profile_sha256()
    if (
        protocol.get("evaluation_profile") != expected
        or protocol.get("evaluation_profile_sha256") != expected_sha256
        or sha256_bytes(canonical_json_bytes(protocol.get("evaluation_profile")))
        != expected_sha256
    ):
        raise ValueError("Protocol does not match the immutable v51.2 evaluation profile")
    if protocol.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("Protocol claim scope exceeds the bounded evaluation profile")
    if protocol.get("authority") != AUTHORITY:
        raise ValueError("Protocol authority contract changed")
    if (
        protocol.get("authentication") != "none"
        or protocol.get("integrity_status") != "content_bound_not_authenticated"
        or protocol.get("trusted_timestamp") is not False
    ):
        raise ValueError("Protocol evidence integrity contract changed")
    if protocol.get("criteria") != CRITERIA or protocol.get(
        "prior_candidate_criteria"
    ) != PRIOR_CANDIDATE_CRITERIA:
        raise ValueError("Protocol final comparison criteria changed")
    development = protocol.get("development", {})
    if (
        development.get("seeds") != list(DEV_SEEDS)
        or development.get("samples_per_seed") != 2_000
        or development.get("soup_groups")
        != [list(group) for group in SOUP_GROUPS]
        or development.get("baseline_blend_alphas") != list(BASE_BLEND_ALPHAS)
        or development.get("selection_order") != list(SELECTION_ORDER)
        or development.get("criteria") != DEVELOPMENT_CRITERIA
        or development.get("prior_candidate_criteria")
        != PRIOR_CANDIDATE_CRITERIA
    ):
        raise ValueError("Protocol development search differs from the frozen profile")
    final = protocol.get("final", {})
    if (
        final.get("seeds") != list(FINAL_SEEDS)
        or final.get("samples_per_seed") != 2_000
        or final.get("single_use") is not True
    ):
        raise ValueError("Protocol final holdout differs from the frozen profile")
    training = protocol.get("training", {})
    if training != expected["training"]:
        raise ValueError("Protocol training recipe differs from the frozen profile")
    for field, profile_field in (
        ("baseline", "release_baseline"),
        ("prior_candidate", "prior_candidate"),
    ):
        artifact = protocol.get(field, {})
        profile_artifact = expected[profile_field]
        for key in (
            "path",
            "size_bytes",
            "sha256",
            "canonical_state_sha256",
            "tensor_count",
            "element_count",
        ):
            observed = artifact.get(key)
            expected_value = profile_artifact.get(key)
            if key == "path" and isinstance(observed, str):
                observed = Path(observed).as_posix()
            if observed != expected_value:
                raise ValueError(f"Protocol {field} identity differs from the profile")


def load_and_validate_protocol(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "protocol.json"
    protocol = load_json_strict(path)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("Unsupported or missing protocol schema")
    if protocol.get("protocol_sha256") != protocol_digest(protocol):
        raise ValueError("Protocol digest mismatch")
    validate_canonical_evaluation_profile(protocol)
    if set(protocol.get("development", {}).get("seeds", ())) & set(
        protocol.get("final", {}).get("seeds", ())
    ):
        raise ValueError("Development and final cohorts overlap")
    mode = protocol.get("execution_mode")
    if mode not in {"dirty_development", "clean_final_eligible"} or bool(
        protocol.get("finalization_allowed")
    ) != (mode == "clean_final_eligible"):
        raise ValueError("Invalid protocol execution mode")
    development_seeds = [int(seed) for seed in protocol["development"]["seeds"]]
    final_seeds = [int(seed) for seed in protocol["final"]["seeds"]]
    members = protocol["training"]["members"]
    member_names = [str(member["name"]) for member in members]
    training_seeds = [int(member["train_seed"]) for member in members]
    if (
        not development_seeds
        or not final_seeds
        or not member_names
        or len(set(development_seeds)) != len(development_seeds)
        or len(set(final_seeds)) != len(final_seeds)
        or len(set(member_names)) != len(member_names)
        or len(set(training_seeds)) != len(training_seeds)
        or set(training_seeds) & (set(development_seeds) | set(final_seeds))
    ):
        raise ValueError("Protocol cohort/member identities are not unique and disjoint")
    soup_groups = protocol["development"]["soup_groups"]
    if not soup_groups or any(
        not group
        or len(set(group)) != len(group)
        or any(str(name) not in member_names for name in group)
        for group in soup_groups
    ):
        raise ValueError("Protocol soup groups are invalid")
    alphas = [float(value) for value in protocol["development"]["baseline_blend_alphas"]]
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in alphas)
    ):
        raise ValueError("Protocol baseline blend alphas are invalid")
    if (
        int(protocol["training"]["train_size_per_member"]) <= 0
        or int(protocol["training"]["epochs"]) <= 0
        or int(protocol["training"]["batch_size"]) <= 0
        or int(protocol["development"]["samples_per_seed"]) <= 0
        or int(protocol["final"]["samples_per_seed"]) <= 0
    ):
        raise ValueError("Protocol dimensions must be positive")
    if protocol.get("task_schemas") != {
        "generator": GENERATOR_SCHEMA,
        "family_tags": FAMILY_TAG_SCHEMA,
        "cohort": COHORT_SCHEMA,
    }:
        raise ValueError("Protocol task schemas changed")
    for role in ("development", "final"):
        expected_specification = cohort_specification(
            protocol[role]["seeds"],
            int(protocol[role]["samples_per_seed"]),
            cohort_role=role,
        )
        if (
            protocol[role].get("cohort_specification") != expected_specification
            or protocol[role].get("cohort_specification_sha256")
            != sha256_bytes(canonical_json_bytes(expected_specification))
        ):
            raise ValueError(f"Protocol {role} cohort commitment changed")
    if protocol.get("code_bindings") != current_code_bindings():
        raise ValueError("Bound source code changed after protocol preparation")
    validate_environment_compatibility(
        protocol.get("environment_at_freeze", {}),
        environment_binding(torch.device("cpu")),
    )
    if set(protocol.get("source_snapshot", {})) != set(protocol["code_bindings"]):
        raise ValueError("Source snapshot does not cover every bound source")
    for relative_name, record in protocol["source_snapshot"].items():
        snapshot_path = resolve_repo_relative(str(record["path"]))
        try:
            snapshot_path.relative_to((output_dir / "source_snapshot").resolve())
        except ValueError as error:
            raise ValueError("Bound source snapshot is outside the protocol output") from error
        if (
            record.get("sha256")
            != protocol["code_bindings"].get(relative_name, {}).get("sha256")
            or not snapshot_path.is_file()
            or snapshot_path.stat().st_size != int(record["size_bytes"])
            or sha256_file(snapshot_path) != record["sha256"]
        ):
            raise ValueError(f"Bound source snapshot changed: {relative_name}")
    baseline_path = resolve_repo_relative(str(protocol["baseline"]["path"]))
    if (
        not baseline_path.is_file()
        or baseline_path.stat().st_size != int(protocol["baseline"]["size_bytes"])
        or sha256_file(baseline_path) != protocol["baseline"]["sha256"]
    ):
        raise ValueError("Canonical baseline changed after protocol preparation")
    prior_candidate_path = resolve_repo_relative(
        str(protocol["prior_candidate"]["path"])
    )
    if (
        not prior_candidate_path.is_file()
        or prior_candidate_path.stat().st_size
        != int(protocol["prior_candidate"]["size_bytes"])
        or sha256_file(prior_candidate_path)
        != protocol["prior_candidate"]["sha256"]
    ):
        raise ValueError("Exact prior v51.1 candidate changed after protocol preparation")
    if mode == "clean_final_eligible":
        frozen_git = protocol["git"]
        current_git = git_binding()
        if (
            bool(frozen_git["dirty"])
            or current_git["dirty"]
            or current_git["commit"] != frozen_git["commit"]
            or any(
                binding.get("head_git_blob_sha1") is None
                or binding.get("head_git_blob_sha1")
                != binding.get("worktree_git_blob_sha1")
                for binding in protocol["code_bindings"].values()
            )
        ):
            raise ValueError("Clean final protocol is no longer bound to its committed tree")
    return protocol


def _class_weights(targets: torch.Tensor, exponent: float) -> torch.Tensor | None:
    value = float(exponent)
    if value <= 0:
        return None
    counts = torch.bincount(targets, minlength=10).float().clamp_min(1.0)
    weights = (counts.mean() / counts).pow(value)
    sample_weight_mean = (weights * counts).sum() / counts.sum()
    return weights / sample_weight_mean.clamp_min(1e-12)


def train_member(
    baseline_state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    training: Mapping[str, Any],
    protocol: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    x_train, y_train, metadata = make_chained_task_with_metadata(
        int(training["train_size_per_member"]),
        int(config["train_seed"]),
    )
    dataset_digest = hashlib.sha256()
    tensor_digest_update(dataset_digest, "x", x_train)
    tensor_digest_update(dataset_digest, "y", y_train)
    for name in ("starts", "op_types", "operands"):
        tensor_digest_update(dataset_digest, name, metadata[name])
    dataset_specification = {
        "schema": COHORT_SCHEMA,
        "generator_schema": GENERATOR_SCHEMA,
        "family_tag_schema": FAMILY_TAG_SCHEMA,
        "cohort_role": "training",
        "member": str(config["name"]),
        "seed": int(config["train_seed"]),
        "n": int(training["train_size_per_member"]),
        "generator_source_sha256": sha256_file(
            SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
        ),
    }
    dataset_specification_sha256 = sha256_bytes(
        canonical_json_bytes(dataset_specification)
    )
    dataset_sha256 = dataset_digest.hexdigest()
    dataset_id = sha256_bytes(
        canonical_json_bytes(
            {
                "specification_sha256": dataset_specification_sha256,
                "dataset_sha256": dataset_sha256,
            }
        )
    )
    model = ChampionNetCognitiveLeapUltraExpert()
    model.load_state_dict(dict(baseline_state), strict=True)
    model.to(device)
    torch.manual_seed(int(config["dropout_seed"]))
    shuffle_generator = torch.Generator().manual_seed(int(config["shuffle_seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    class_weights = _class_weights(y_train, float(config["balance_exponent"]))
    if class_weights is not None:
        class_weights = class_weights.to(device)
    history: list[dict[str, Any]] = []
    rng_before_sha256 = sha256_bytes(torch.get_rng_state().numpy().tobytes())
    shuffle_before_sha256 = sha256_bytes(shuffle_generator.get_state().numpy().tobytes())
    permutation_digest = hashlib.sha256()
    started = time.perf_counter()
    batch_size = int(training["batch_size"])
    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        epoch_started = time.perf_counter()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator)
        tensor_digest_update(permutation_digest, f"epoch_{epoch}_permutation", permutation)
        for offset in range(0, int(x_train.shape[0]), batch_size):
            indices = permutation[offset : offset + batch_size]
            xb = x_train[indices].to(device)
            yb = y_train[indices].to(device)
            logits = model(xb).squeeze(1)
            loss = F.cross_entropy(logits, yb, weight=class_weights) + aux_loss(model, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(training["gradient_clip_norm"]),
            )
            optimizer.step()
            count = int(yb.numel())
            total_loss += float(loss.item()) * count
            total_correct += int(logits.argmax(dim=-1).eq(yb).sum().item())
            total_seen += count
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, total_seen),
            "accuracy": total_correct / max(1, total_seen),
            "seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(
            f"member={config['name']} epoch={epoch}/{training['epochs']} "
            f"loss={row['loss']:.5f} acc={row['accuracy']:.5f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint_path = output_dir / "members" / str(config["name"]) / "weights.pth"
    artifact = save_state(checkpoint_path, state)
    receipt = {
        "schema": "supermix-cognitive-leap-training-receipt-v2",
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": protocol["evaluation_profile_sha256"],
        "parent_baseline": protocol["baseline"],
        "config": dict(config),
        "dataset": {
            "specification": dataset_specification,
            "specification_sha256": dataset_specification_sha256,
            "dataset_sha256": dataset_sha256,
            "dataset_id": dataset_id,
        },
        "class_counts": torch.bincount(y_train, minlength=10).tolist(),
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "optimizer": {
            "name": "AdamW",
            "lr": float(config["lr"]),
            "weight_decay": float(training["weight_decay"]),
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
        },
        "rng": {
            "dropout_seed": int(config["dropout_seed"]),
            "shuffle_seed": int(config["shuffle_seed"]),
            "cpu_state_before_sha256": rng_before_sha256,
            "cpu_state_after_sha256": sha256_bytes(
                torch.get_rng_state().numpy().tobytes()
            ),
            "shuffle_state_before_sha256": shuffle_before_sha256,
            "shuffle_state_after_sha256": sha256_bytes(
                shuffle_generator.get_state().numpy().tobytes()
            ),
            "permutation_sha256": permutation_digest.hexdigest(),
        },
        "history": history,
        "train_seconds": time.perf_counter() - started,
        "artifact": artifact,
    }
    receipt["receipt_id"] = sha256_bytes(canonical_json_bytes(receipt))
    write_json_atomic(checkpoint_path.with_name("training_receipt.json"), receipt)
    return state, receipt


def selection_score(result: Mapping[str, Any]) -> tuple[Any, ...]:
    checks = result["checks"]
    summary = result["summary"]
    return (
        int(result["passed"]),
        sum(bool(value) for value in checks.values()),
        int(summary["nonregressing_seed_count"]),
        int(summary["nonregressing_family_count"]),
        int(summary["nonregressing_class_count"]),
        float(summary["accuracy_delta"]),
        -float(summary["mean_candidate_loss"]),
    )


def dual_selection_score(
    release_comparison: Mapping[str, Any],
    prior_comparison: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Rank only dual-gate candidates using the predeclared deterministic order."""

    release_summary = release_comparison["summary"]
    prior_summary = prior_comparison["summary"]
    both_passed = bool(release_comparison["passed"] and prior_comparison["passed"])
    total_checks = sum(bool(value) for value in release_comparison["checks"].values())
    total_checks += sum(bool(value) for value in prior_comparison["checks"].values())
    combined_nonregression = sum(
        int(summary[key])
        for summary in (release_summary, prior_summary)
        for key in (
            "nonregressing_seed_count",
            "nonregressing_family_count",
            "nonregressing_class_count",
        )
    )
    return (
        int(both_passed),
        total_checks,
        float(prior_summary["accuracy_delta"]),
        float(release_summary["accuracy_delta"]),
        combined_nonregression,
        -float(release_summary["mean_candidate_loss"]),
    )


def dual_candidate_row(
    *,
    name: str,
    group: Sequence[str],
    alpha: float,
    release_comparison: Mapping[str, Any],
    prior_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    score = dual_selection_score(release_comparison, prior_comparison)
    return {
        "name": name,
        "members": list(group),
        "member_weights": [1.0 / len(group)] * len(group),
        "baseline_blend_alpha": float(alpha),
        "passed": bool(score[0]),
        "comparisons": {
            "release_continuity": {
                key: release_comparison[key]
                for key in ("passed", "checks", "summary", "evidence")
            },
            "prior_candidate_superiority": {
                key: prior_comparison[key]
                for key in ("passed", "checks", "summary", "evidence")
            },
        },
        "selection_score": list(score),
    }


def write_three_way_final_prediction_artifact(
    path: Path,
    release_baseline: Mapping[str, Any],
    prior_candidate: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    *,
    evaluation_profile_sha256: str,
) -> dict[str, Any]:
    """Write one deterministic artifact containing all three exact logit streams."""

    prediction_sets = {
        "release_baseline": {
            row["seed"]: row for row in release_baseline["seed_rows"]
        },
        "prior_candidate": {
            row["seed"]: row for row in prior_candidate["seed_rows"]
        },
        "candidate": {row["seed"]: row for row in candidate["seed_rows"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    row_count = 0
    content_digest = hashlib.sha256()
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle,
            mode="wb",
            filename="",
            mtime=0,
        ) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="\n") as handle:
                for cohort_row in cohort["rows"]:
                    seed = int(cohort_row["seed"])
                    targets = cohort_row["y"]
                    for index in range(int(targets.numel())):
                        target = int(targets[index].item())
                        op_types = [
                            int(value)
                            for value in cohort_row["op_types"][index].tolist()
                        ]
                        row: dict[str, Any] = {
                            "example_id": f"{cohort['dataset_id']}:{seed}:{index}",
                            "dataset_id": cohort["dataset_id"],
                            "cohort_role": cohort["cohort_role"],
                            "seed": seed,
                            "index": index,
                            "target": target,
                            "start": int(cohort_row["starts"][index].item()),
                            "op_types": op_types,
                            "operands": [
                                int(value)
                                for value in cohort_row["operands"][index].tolist()
                            ],
                            "operation_family_tags": list(
                                operation_family_tags(
                                    cohort_row["op_types"][index]
                                )
                            ),
                        }
                        for model_name in (
                            "release_baseline",
                            "prior_candidate",
                            "candidate",
                        ):
                            result = prediction_sets[model_name][seed]
                            prediction = int(result["predictions"][index].item())
                            logits = result["logits"][index]
                            row[f"{model_name}_prediction"] = prediction
                            row[f"{model_name}_correct"] = prediction == target
                            row[f"{model_name}_logits_f32le_hex"] = struct.pack(
                                "<10f",
                                *[float(value) for value in logits.tolist()],
                            ).hex()
                        payload = canonical_json_bytes(row) + b"\n"
                        handle.write(payload.decode("ascii"))
                        content_digest.update(payload)
                        row_count += 1
    temporary.replace(path)
    return {
        "schema": THREE_WAY_PREDICTION_ARTIFACT_SCHEMA,
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "uncompressed_sha256": content_digest.hexdigest(),
        "row_count": row_count,
        "evaluation_profile_sha256": evaluation_profile_sha256,
        "cohort_schema": cohort["schema"],
        "generator_schema": cohort["generator_schema"],
        "family_tag_schema": cohort["family_tag_schema"],
        "cohort_role": cohort["cohort_role"],
        "dataset_id": cohort["dataset_id"],
        "dataset_specification_sha256": cohort["specification_sha256"],
        "dataset_sha256": cohort["dataset_sha256"],
        "format": "deterministic_gzip_jsonl",
        "logits_encoding": "hex_little_endian_float32",
        "class_count": 10,
        "class_order": list(range(10)),
        "logit_shape": [10],
        "argmax_tie_rule": "lowest_class_index",
        "loss_formula": "torch_cross_entropy_float32_sum_per_seed_then_float64_total",
        "gzip_mtime": 0,
    }


def three_way_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    criteria: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    evaluation_profile_sha256: str,
) -> dict[str, Any]:
    comparison = compare_predictions(baseline, candidate, cohort, criteria)
    comparison["criteria"] = dict(criteria)
    comparison["evidence"] = {
        "evaluation_profile_sha256": evaluation_profile_sha256,
        **comparison["evidence"],
        "per_example_compressed_sha256": artifact["sha256"],
        "per_example_uncompressed_sha256": artifact["uncompressed_sha256"],
    }
    return comparison


def write_lineage_manifest(
    output_dir: Path,
    protocol: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor],
    member_states: Mapping[str, Mapping[str, torch.Tensor]],
    member_receipts: Mapping[str, Mapping[str, Any]],
    selected_row: Mapping[str, Any],
    selected_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    member_names = [str(name) for name in selected_row["members"]]
    member_weights = [float(value) for value in selected_row["member_weights"]]
    soup = average_states(
        [member_states[name] for name in member_names],
        member_weights,
    )
    reconstructed = blend_with_baseline(
        baseline_state,
        soup,
        float(selected_row["baseline_blend_alpha"]),
    )
    selected_state = load_state(REPO_ROOT / str(selected_artifact["path"]))
    exact = True
    max_abs_error = 0.0
    for name in selected_state:
        expected = selected_state[name]
        observed = reconstructed[name]
        if not torch.equal(expected, observed):
            exact = False
        if expected.is_floating_point() or expected.is_complex():
            difference = float((expected - observed).abs().max().item())
            max_abs_error = max(max_abs_error, difference)
    reconstructed_summary = state_dict_summary(reconstructed)
    selected_summary = state_dict_summary(selected_state)
    if not exact or reconstructed_summary != selected_summary:
        raise ValueError("Selected checkpoint is not an exact reconstruction of its lineage")

    members: list[dict[str, Any]] = []
    for name in member_names:
        receipt_path = (
            REPO_ROOT / str(member_receipts[name]["artifact"]["path"])
        ).with_name("training_receipt.json")
        members.append(
            {
                "name": name,
                "config": member_receipts[name]["config"],
                "artifact": member_receipts[name]["artifact"],
                "training_receipt": {
                    "path": relative_path(receipt_path),
                    "sha256": sha256_file(receipt_path),
                    "size_bytes": receipt_path.stat().st_size,
                },
            }
        )
    baseline_node_id = sha256_bytes(
        canonical_json_bytes(
            {"kind": "checkpoint", "artifact_sha256": protocol["baseline"]["sha256"]}
        )
    )
    member_node_ids = {
        member["name"]: sha256_bytes(
            canonical_json_bytes(
                {
                    "kind": "continuation",
                    "parent": baseline_node_id,
                    "artifact_sha256": member["artifact"]["sha256"],
                    "config": member["config"],
                }
            )
        )
        for member in members
    }
    soup_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "ordered_weighted_soup",
                "parents": [member_node_ids[name] for name in member_names],
                "weights": member_weights,
            }
        )
    )
    blend_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "baseline_soup_blend",
                "parents": [baseline_node_id, soup_node_id],
                "weights": [
                    1.0 - float(selected_row["baseline_blend_alpha"]),
                    float(selected_row["baseline_blend_alpha"]),
                ],
            }
        )
    )
    candidate_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "materialized_checkpoint",
                "parent": blend_node_id,
                "artifact_sha256": selected_artifact["sha256"],
            }
        )
    )
    nodes = [
        {
            "node_id": baseline_node_id,
            "kind": "checkpoint",
            "parents": [],
            "artifact": protocol["baseline"],
        },
        *[
            {
                "node_id": member_node_ids[member["name"]],
                "kind": "continuation",
                "parents": [baseline_node_id],
                "config": member["config"],
                "artifact": member["artifact"],
                "training_receipt": member["training_receipt"],
            }
            for member in members
        ],
        {
            "node_id": soup_node_id,
            "kind": "ordered_weighted_soup",
            "parents": [member_node_ids[name] for name in member_names],
            "weights": member_weights,
        },
        {
            "node_id": blend_node_id,
            "kind": "baseline_soup_blend",
            "parents": [baseline_node_id, soup_node_id],
            "weights": [
                1.0 - float(selected_row["baseline_blend_alpha"]),
                float(selected_row["baseline_blend_alpha"]),
            ],
        },
        {
            "node_id": candidate_node_id,
            "kind": "materialized_checkpoint",
            "parents": [blend_node_id],
            "artifact": dict(selected_artifact),
        },
    ]
    manifest = {
        "schema": "supermix-cognitive-leap-lineage-v2",
        "authentication": "none",
        "timestamps_trusted": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": protocol["evaluation_profile_sha256"],
        "baseline": protocol["baseline"],
        "selected_recipe": {
            "name": selected_row["name"],
            "members": member_names,
            "member_weights": member_weights,
            "baseline_blend_alpha": float(selected_row["baseline_blend_alpha"]),
        },
        "selected_development_evidence_sha256": sha256_bytes(
            canonical_json_bytes(selected_row)
        ),
        "members": members,
        "nodes": nodes,
        "root_node_id": baseline_node_id,
        "selected_node_id": candidate_node_id,
        "soup": {
            "algorithm": "ordered_float_tensor_weighted_mean_v1",
            "members": member_names,
            "weights": member_weights,
        },
        "baseline_blend": {
            "algorithm": "ordered_float_tensor_weighted_mean_v1",
            "baseline_weight": 1.0 - float(selected_row["baseline_blend_alpha"]),
            "soup_weight": float(selected_row["baseline_blend_alpha"]),
        },
        "selected_artifact": dict(selected_artifact),
        "reconstruction": {
            "exact_tensor_equality": exact,
            "max_absolute_error": max_abs_error,
            "reconstructed_canonical_state_sha256": reconstructed_summary[
                "canonical_state_sha256"
            ],
            "selected_canonical_state_sha256": selected_summary[
                "canonical_state_sha256"
            ],
            "strict_load": True,
        },
    }
    path = output_dir / "lineage_manifest.json"
    write_json_atomic(path, manifest)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": manifest["schema"],
    }


def validate_lineage_manifest(
    record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    selected_artifact: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    path = resolve_repo_relative(str(record.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("size_bytes", -1))
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError("Selection lineage manifest is missing or changed")
    manifest = load_json_strict(path)
    if (
        manifest.get("schema") != "supermix-cognitive-leap-lineage-v2"
        or manifest.get("authentication") != "none"
        or manifest.get("integrity_status")
        != "content_bound_not_authenticated"
        or manifest.get("timestamps_trusted") is not False
        or manifest.get("authority") != AUTHORITY
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("evaluation_profile_sha256")
        != protocol["evaluation_profile_sha256"]
        or manifest.get("baseline") != protocol["baseline"]
        or manifest.get("selected_artifact") != selected_artifact
    ):
        raise ValueError("Selection lineage contract mismatch")

    protocol_configs = {
        str(config["name"]): config for config in protocol["training"]["members"]
    }
    member_states: dict[str, dict[str, torch.Tensor]] = {}
    for member in manifest.get("members", []):
        name = str(member["name"])
        if name in member_states or member.get("config") != protocol_configs.get(name):
            raise ValueError(f"Lineage member configuration mismatch: {name}")
        artifact = member["artifact"]
        artifact_path = resolve_repo_relative(str(artifact["path"]))
        training_receipt = member["training_receipt"]
        training_receipt_path = resolve_repo_relative(str(training_receipt["path"]))
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(artifact["size_bytes"])
            or sha256_file(artifact_path) != artifact["sha256"]
            or not training_receipt_path.is_file()
            or training_receipt_path.stat().st_size
            != int(training_receipt["size_bytes"])
            or sha256_file(training_receipt_path) != training_receipt["sha256"]
        ):
            raise ValueError(f"Lineage member changed: {name}")
        training = load_json_strict(training_receipt_path)
        training_payload = dict(training)
        training_id = training_payload.pop("receipt_id", None)
        if (
            training_id != sha256_bytes(canonical_json_bytes(training_payload))
            or training.get("schema")
            != "supermix-cognitive-leap-training-receipt-v2"
            or training.get("authentication") != "none"
            or training.get("trusted_timestamp") is not False
            or training.get("integrity_status")
            != "content_bound_not_authenticated"
            or training.get("authority") != AUTHORITY
            or training.get("protocol_sha256") != protocol["protocol_sha256"]
            or training.get("evaluation_profile_sha256")
            != protocol["evaluation_profile_sha256"]
            or training.get("parent_baseline") != protocol["baseline"]
            or training.get("config") != protocol_configs[name]
            or training.get("artifact") != artifact
        ):
            raise ValueError(f"Lineage member training receipt mismatch: {name}")
        training_count = int(protocol["training"]["train_size_per_member"])
        training_seed = int(protocol_configs[name]["train_seed"])
        x_train, y_train, training_metadata = make_chained_task_with_metadata(
            training_count,
            training_seed,
        )
        training_digest = hashlib.sha256()
        tensor_digest_update(training_digest, "x", x_train)
        tensor_digest_update(training_digest, "y", y_train)
        for metadata_name in ("starts", "op_types", "operands"):
            tensor_digest_update(
                training_digest,
                metadata_name,
                training_metadata[metadata_name],
            )
        dataset = training.get("dataset", {})
        expected_training_specification = {
            "schema": COHORT_SCHEMA,
            "generator_schema": GENERATOR_SCHEMA,
            "family_tag_schema": FAMILY_TAG_SCHEMA,
            "cohort_role": "training",
            "member": name,
            "seed": training_seed,
            "n": training_count,
            "generator_source_sha256": sha256_file(
                SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
            ),
        }
        if (
            dataset.get("dataset_sha256") != training_digest.hexdigest()
            or dataset.get("specification") != expected_training_specification
            or dataset.get("specification_sha256")
            != sha256_bytes(canonical_json_bytes(expected_training_specification))
            or dataset.get("dataset_id")
            != sha256_bytes(
                canonical_json_bytes(
                    {
                        "specification_sha256": dataset.get(
                            "specification_sha256"
                        ),
                        "dataset_sha256": dataset.get("dataset_sha256"),
                    }
                )
            )
        ):
            raise ValueError(f"Lineage member training dataset mismatch: {name}")
        member_states[name] = load_state(artifact_path)

    soup_recipe = manifest["soup"]
    member_names = [str(name) for name in soup_recipe["members"]]
    member_weights = [float(value) for value in soup_recipe["weights"]]
    if set(member_names) != set(member_states):
        raise ValueError("Lineage soup does not bind every selected member")
    selected_recipe = manifest.get("selected_recipe", {})
    if (
        selected_recipe.get("members") != member_names
        or selected_recipe.get("member_weights") != member_weights
        or selected_recipe.get("baseline_blend_alpha")
        != manifest.get("baseline_blend", {}).get("soup_weight")
    ):
        raise ValueError("Lineage selected recipe cross-link mismatch")
    soup = average_states(
        [member_states[name] for name in member_names],
        member_weights,
    )
    baseline_state = load_state(resolve_repo_relative(str(protocol["baseline"]["path"])))
    blend_recipe = manifest["baseline_blend"]
    alpha = float(blend_recipe["soup_weight"])
    if not math.isclose(
        float(blend_recipe["baseline_weight"]),
        1.0 - alpha,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Lineage blend weights do not sum to one")
    baseline_node_id = sha256_bytes(
        canonical_json_bytes(
            {"kind": "checkpoint", "artifact_sha256": protocol["baseline"]["sha256"]}
        )
    )
    manifest_members = manifest["members"]
    member_node_ids = {
        member["name"]: sha256_bytes(
            canonical_json_bytes(
                {
                    "kind": "continuation",
                    "parent": baseline_node_id,
                    "artifact_sha256": member["artifact"]["sha256"],
                    "config": member["config"],
                }
            )
        )
        for member in manifest_members
    }
    soup_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "ordered_weighted_soup",
                "parents": [member_node_ids[name] for name in member_names],
                "weights": member_weights,
            }
        )
    )
    blend_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "baseline_soup_blend",
                "parents": [baseline_node_id, soup_node_id],
                "weights": [1.0 - alpha, alpha],
            }
        )
    )
    candidate_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "materialized_checkpoint",
                "parent": blend_node_id,
                "artifact_sha256": selected_artifact["sha256"],
            }
        )
    )
    expected_nodes = [
        {
            "node_id": baseline_node_id,
            "kind": "checkpoint",
            "parents": [],
            "artifact": protocol["baseline"],
        },
        *[
            {
                "node_id": member_node_ids[member["name"]],
                "kind": "continuation",
                "parents": [baseline_node_id],
                "config": member["config"],
                "artifact": member["artifact"],
                "training_receipt": member["training_receipt"],
            }
            for member in manifest_members
        ],
        {
            "node_id": soup_node_id,
            "kind": "ordered_weighted_soup",
            "parents": [member_node_ids[name] for name in member_names],
            "weights": member_weights,
        },
        {
            "node_id": blend_node_id,
            "kind": "baseline_soup_blend",
            "parents": [baseline_node_id, soup_node_id],
            "weights": [1.0 - alpha, alpha],
        },
        {
            "node_id": candidate_node_id,
            "kind": "materialized_checkpoint",
            "parents": [blend_node_id],
            "artifact": dict(selected_artifact),
        },
    ]
    if (
        manifest.get("root_node_id") != baseline_node_id
        or manifest.get("selected_node_id") != candidate_node_id
        or canonical_json_bytes(manifest.get("nodes"))
        != canonical_json_bytes(expected_nodes)
    ):
        raise ValueError("Lineage graph identity mismatch")
    reconstructed = blend_with_baseline(baseline_state, soup, alpha)
    selected_path = resolve_repo_relative(str(selected_artifact["path"]))
    if (
        not selected_path.is_file()
        or selected_path.stat().st_size
        != int(selected_artifact.get("size_bytes", -1))
        or sha256_file(selected_path) != selected_artifact.get("sha256")
    ):
        raise ValueError("Selected checkpoint artifact changed")
    selected_state = load_state(selected_path)
    if state_dict_summary(reconstructed) != state_dict_summary(selected_state) or any(
        not torch.equal(reconstructed[name], selected_state[name])
        for name in selected_state
    ):
        raise ValueError("Selected checkpoint failed exact lineage reconstruction")
    reconstruction = manifest["reconstruction"]
    if (
        reconstruction.get("exact_tensor_equality") is not True
        or float(reconstruction.get("max_absolute_error", math.inf)) != 0.0
        or reconstruction.get("reconstructed_canonical_state_sha256")
        != state_dict_summary(reconstructed)["canonical_state_sha256"]
    ):
        raise ValueError("Lineage reconstruction receipt mismatch")
    return baseline_state, member_states


def write_lineage_verification(
    output_dir: Path,
    lineage_record: Mapping[str, Any],
    protocol: Mapping[str, Any],
    selected_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_state, member_states = validate_lineage_manifest(
        lineage_record,
        protocol,
        selected_artifact,
    )
    lineage_manifest = load_json_strict(
        resolve_repo_relative(str(lineage_record["path"]))
    )
    verification = {
        "schema": "supermix-cognitive-leap-lineage-verification-v1",
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "valid": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": protocol["evaluation_profile_sha256"],
        "lineage_manifest": dict(lineage_record),
        "root_node_id": lineage_manifest["root_node_id"],
        "selected_node_id": lineage_manifest["selected_node_id"],
        "baseline_canonical_state_sha256": state_dict_summary(baseline_state)[
            "canonical_state_sha256"
        ],
        "member_canonical_state_sha256": {
            name: state_dict_summary(state)["canonical_state_sha256"]
            for name, state in sorted(member_states.items())
        },
        "selected_canonical_state_sha256": selected_artifact[
            "canonical_state_sha256"
        ],
        "exact_tensor_reconstruction": True,
    }
    verification["verification_id"] = sha256_bytes(canonical_json_bytes(verification))
    path = output_dir / "lineage_verification.json"
    write_json_atomic(path, verification)
    return {
        "path": relative_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema": verification["schema"],
    }


def replay_development_selection(
    protocol: Mapping[str, Any],
    selection: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor],
    member_states: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    development = build_cohort(
        protocol["development"]["seeds"],
        int(protocol["development"]["samples_per_seed"]),
        cohort_role="development",
    )
    if development["dataset_sha256"] != selection["development_dataset_sha256"]:
        raise ValueError("Development cohort identity changed")
    evaluator = ChampionNetCognitiveLeapUltraExpert().to(device)
    evaluator.load_state_dict(dict(baseline_state), strict=True)
    baseline_predictions = predict_cohort(evaluator, development, device)
    prior_candidate_state = load_state(
        resolve_repo_relative(str(protocol["prior_candidate"]["path"]))
    )
    evaluator.load_state_dict(prior_candidate_state, strict=True)
    prior_candidate_predictions = predict_cohort(evaluator, development, device)
    candidate_rows: list[dict[str, Any]] = []
    for group in protocol["development"]["soup_groups"]:
        soup = average_states([member_states[str(name)] for name in group])
        for alpha in protocol["development"]["baseline_blend_alphas"]:
            state = blend_with_baseline(baseline_state, soup, float(alpha))
            evaluator.load_state_dict(state, strict=True)
            candidate_predictions = predict_cohort(evaluator, development, device)
            release_comparison = compare_predictions(
                baseline_predictions,
                candidate_predictions,
                development,
                protocol["development"]["criteria"],
            )
            prior_comparison = compare_predictions(
                prior_candidate_predictions,
                candidate_predictions,
                development,
                protocol["development"]["prior_candidate_criteria"],
            )
            name = f"{'-'.join(group)}__alpha_{float(alpha):.2f}"
            candidate_rows.append(
                dual_candidate_row(
                    name=name,
                    group=[str(value) for value in group],
                    alpha=float(alpha),
                    release_comparison=release_comparison,
                    prior_comparison=prior_comparison,
                )
            )
    if canonical_json_bytes(candidate_rows) != canonical_json_bytes(selection["candidates"]):
        raise ValueError("Development candidate evidence did not replay exactly")
    replayed = max(candidate_rows, key=lambda row: tuple(row["selection_score"]))
    stored_selected = {
        key: value
        for key, value in selection["selected"].items()
        if key != "artifact"
    }
    if bool(selection.get("passed")) != bool(replayed["passed"]):
        raise ValueError("Frozen development pass/reject decision did not replay")
    if canonical_json_bytes(replayed) != canonical_json_bytes(stored_selected):
        raise ValueError("Frozen development selection is not the replayed winner")
    return replayed


def verify_development_selection(output_dir: Path, device: torch.device) -> Path:
    """Independently reconstruct a persisted development-only selection."""

    protocol = load_and_validate_protocol(output_dir)
    selection_path = output_dir / "selection_receipt.json"
    selection = load_json_strict(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Unsupported development selection schema")
    if selection.get("selection_sha256") != selection_digest(selection):
        raise ValueError("Development selection digest mismatch")
    if selection.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Development selection belongs to another protocol")
    if (
        selection.get("authentication") != "none"
        or selection.get("integrity_status")
        != "content_bound_not_authenticated"
        or selection.get("trusted_timestamp") is not False
        or selection.get("authority") != AUTHORITY
    ):
        raise ValueError("Development selection evidence contract mismatch")
    selection_passed = bool(selection.get("passed"))
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("Development selection winner is missing")
    if selection_passed:
        if (
            selection.get("decision") != "selected_and_frozen_for_single_final"
            or not isinstance(selected.get("artifact"), Mapping)
            or not isinstance(selection.get("lineage_manifest"), Mapping)
            or not isinstance(selection.get("lineage_verification"), Mapping)
        ):
            raise ValueError("Passing development selection shape is invalid")
    elif (
        selection.get("decision") != "no_development_candidate_passed"
        or "artifact" in selected
        or "lineage_manifest" in selection
        or "lineage_verification" in selection
    ):
        raise ValueError("Rejected development selection shape is invalid")

    baseline_state = load_state(
        resolve_repo_relative(str(protocol["baseline"]["path"]))
    )
    stored_receipts = selection.get("member_receipts")
    if not isinstance(stored_receipts, Mapping):
        raise ValueError("Development selection member receipts are missing")
    expected_member_names = {
        str(config["name"]) for config in protocol["training"]["members"]
    }
    if set(stored_receipts) != expected_member_names:
        raise ValueError("Development selection member receipt set mismatch")
    member_states: dict[str, dict[str, torch.Tensor]] = {}
    for config in protocol["training"]["members"]:
        name = str(config["name"])
        stored = stored_receipts.get(name)
        if not isinstance(stored, Mapping):
            raise ValueError(f"Development member receipt is missing: {name}")
        expected_receipt_keys = {
            "schema",
            "authentication",
            "trusted_timestamp",
            "integrity_status",
            "authority",
            "protocol_sha256",
            "evaluation_profile_sha256",
            "parent_baseline",
            "config",
            "dataset",
            "class_counts",
            "class_weights",
            "optimizer",
            "rng",
            "history",
            "train_seconds",
            "artifact",
            "receipt_id",
        }
        if set(stored) != expected_receipt_keys:
            raise ValueError(f"Development member receipt fields mismatch: {name}")
        receipt_payload = dict(stored)
        receipt_id = receipt_payload.pop("receipt_id", None)
        if receipt_id != sha256_bytes(canonical_json_bytes(receipt_payload)):
            raise ValueError(f"Development member receipt digest mismatch: {name}")
        if (
            stored.get("schema")
            != "supermix-cognitive-leap-training-receipt-v2"
            or stored.get("protocol_sha256") != protocol["protocol_sha256"]
            or stored.get("evaluation_profile_sha256")
            != protocol["evaluation_profile_sha256"]
            or stored.get("parent_baseline") != protocol["baseline"]
            or stored.get("config") != config
            or stored.get("authentication") != "none"
            or stored.get("integrity_status")
            != "content_bound_not_authenticated"
            or stored.get("trusted_timestamp") is not False
            or stored.get("authority") != AUTHORITY
        ):
            raise ValueError(f"Development member receipt contract mismatch: {name}")
        artifact = stored.get("artifact")
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Development member artifact is missing: {name}")
        artifact_path = resolve_repo_relative(str(artifact["path"]))
        expected_member_root = (output_dir / "members" / name).resolve()
        if artifact_path != expected_member_root / "weights.pth":
            raise ValueError(f"Development member artifact path mismatch: {name}")
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(artifact.get("size_bytes", -1))
            or sha256_file(artifact_path) != artifact.get("sha256")
        ):
            raise ValueError(f"Development member artifact changed: {name}")
        receipt_path = artifact_path.with_name("training_receipt.json")
        if receipt_path != expected_member_root / "training_receipt.json":
            raise ValueError(f"Development member receipt path mismatch: {name}")
        if (
            not receipt_path.is_file()
            or canonical_json_bytes(load_json_strict(receipt_path))
            != canonical_json_bytes(stored)
        ):
            raise ValueError(f"Development member receipt file changed: {name}")
        member_state = load_state(artifact_path)
        expected_summary = {
            key: artifact.get(key)
            for key in (
                "tensor_count",
                "element_count",
                "all_finite",
                "tensor_byte_order",
                "canonical_state_sha256",
            )
        }
        if (
            state_dict_summary(member_state) != expected_summary
            or state_dict_inventory(member_state) != artifact.get("tensor_inventory")
        ):
            raise ValueError(f"Development member state binding mismatch: {name}")
        training_count = int(protocol["training"]["train_size_per_member"])
        training_seed = int(config["train_seed"])
        x_train, y_train, metadata = make_chained_task_with_metadata(
            training_count,
            training_seed,
        )
        dataset_digest = hashlib.sha256()
        tensor_digest_update(dataset_digest, "x", x_train)
        tensor_digest_update(dataset_digest, "y", y_train)
        for metadata_name in ("starts", "op_types", "operands"):
            tensor_digest_update(
                dataset_digest,
                metadata_name,
                metadata[metadata_name],
            )
        expected_specification = {
            "schema": COHORT_SCHEMA,
            "generator_schema": GENERATOR_SCHEMA,
            "family_tag_schema": FAMILY_TAG_SCHEMA,
            "cohort_role": "training",
            "member": name,
            "seed": training_seed,
            "n": training_count,
            "generator_source_sha256": sha256_file(
                SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
            ),
        }
        expected_specification_sha256 = sha256_bytes(
            canonical_json_bytes(expected_specification)
        )
        expected_dataset_sha256 = dataset_digest.hexdigest()
        expected_dataset_id = sha256_bytes(
            canonical_json_bytes(
                {
                    "specification_sha256": expected_specification_sha256,
                    "dataset_sha256": expected_dataset_sha256,
                }
            )
        )
        if stored.get("dataset") != {
            "specification": expected_specification,
            "specification_sha256": expected_specification_sha256,
            "dataset_sha256": expected_dataset_sha256,
            "dataset_id": expected_dataset_id,
        }:
            raise ValueError(f"Development member dataset receipt mismatch: {name}")
        member_states[name] = member_state

    replayed = replay_development_selection(
        protocol,
        selection,
        baseline_state,
        member_states,
        device,
    )
    if selection_passed:
        selected_path = resolve_repo_relative(str(selected["artifact"]["path"]))
        if selected_path != (output_dir / "selected" / "cognitive_leap_ultra_v51_2.pth").resolve():
            raise ValueError("Passing development selected artifact path mismatch")
        lineage_baseline, lineage_members = validate_lineage_manifest(
            selection["lineage_manifest"],
            protocol,
            selected["artifact"],
        )
        if state_dict_summary(lineage_baseline) != state_dict_summary(baseline_state):
            raise ValueError("Development lineage baseline differs from replay baseline")
        if set(lineage_members) != set(member_states) or any(
            state_dict_summary(lineage_members[name])
            != state_dict_summary(member_states[name])
            for name in member_states
        ):
            raise ValueError("Development lineage members differ from replay members")
        lineage_verification_record = selection["lineage_verification"]
        lineage_verification_path = resolve_repo_relative(
            str(lineage_verification_record.get("path", ""))
        )
        if (
            not lineage_verification_path.is_file()
            or lineage_verification_path.stat().st_size
            != int(lineage_verification_record.get("size_bytes", -1))
            or sha256_file(lineage_verification_path)
            != lineage_verification_record.get("sha256")
        ):
            raise ValueError("Development lineage verification changed")
        lineage_verification = load_json_strict(lineage_verification_path)
        verification_payload = dict(lineage_verification)
        verification_id = verification_payload.pop("verification_id", None)
        if (
            lineage_verification.get("schema")
            != "supermix-cognitive-leap-lineage-verification-v1"
            or lineage_verification.get("authentication") != "none"
            or lineage_verification.get("trusted_timestamp") is not False
            or lineage_verification.get("integrity_status")
            != "content_bound_not_authenticated"
            or lineage_verification.get("authority") != AUTHORITY
            or lineage_verification.get("valid") is not True
            or lineage_verification.get("protocol_sha256")
            != protocol["protocol_sha256"]
            or lineage_verification.get("evaluation_profile_sha256")
            != protocol["evaluation_profile_sha256"]
            or lineage_verification.get("lineage_manifest")
            != selection["lineage_manifest"]
            or lineage_verification.get("selected_canonical_state_sha256")
            != selected["artifact"].get("canonical_state_sha256")
            or lineage_verification.get("exact_tensor_reconstruction") is not True
            or verification_id
            != sha256_bytes(canonical_json_bytes(verification_payload))
        ):
            raise ValueError("Development lineage verification contract mismatch")
    verification_path = output_dir / "development_replay_verification.json"
    verification = {
        "schema": "supermix-cognitive-leap-development-replay-v1",
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "selection_sha256": selection["selection_sha256"],
        "selection_file_sha256": sha256_file(selection_path),
        "passed": bool(selection["passed"]),
        "selected_name": replayed["name"],
        "selected_score": replayed["selection_score"],
        "candidate_matrix_sha256": sha256_bytes(
            canonical_json_bytes(selection["candidates"])
        ),
    }
    verification["verification_id"] = sha256_bytes(
        canonical_json_bytes(verification)
    )
    write_json_atomic(verification_path, verification)
    return verification_path


def train_and_select(output_dir: Path, device: torch.device) -> Path:
    protocol = load_and_validate_protocol(output_dir)
    selection_path = output_dir / "selection_receipt.json"
    if selection_path.exists():
        raise FileExistsError(f"Selection already exists: {selection_path}")
    baseline_path = REPO_ROOT / protocol["baseline"]["path"]
    baseline_state = load_state(baseline_path)
    member_states: dict[str, dict[str, torch.Tensor]] = {}
    member_receipts: dict[str, Any] = {}
    for config in protocol["training"]["members"]:
        state, receipt = train_member(
            baseline_state,
            config,
            protocol["training"],
            protocol,
            output_dir,
            device,
        )
        member_states[str(config["name"])] = state
        member_receipts[str(config["name"])] = receipt

    development = build_cohort(
        protocol["development"]["seeds"],
        int(protocol["development"]["samples_per_seed"]),
        cohort_role="development",
    )
    evaluator = ChampionNetCognitiveLeapUltraExpert().to(device)
    evaluator.load_state_dict(baseline_state, strict=True)
    baseline_predictions = predict_cohort(evaluator, development, device)
    prior_candidate_state = load_state(
        resolve_repo_relative(str(protocol["prior_candidate"]["path"]))
    )
    evaluator.load_state_dict(prior_candidate_state, strict=True)
    prior_candidate_predictions = predict_cohort(evaluator, development, device)
    candidate_rows: list[dict[str, Any]] = []
    selected_state: dict[str, torch.Tensor] | None = None
    selected_row: dict[str, Any] | None = None
    for group in protocol["development"]["soup_groups"]:
        soup = average_states([member_states[name] for name in group])
        for alpha in protocol["development"]["baseline_blend_alphas"]:
            state = blend_with_baseline(baseline_state, soup, float(alpha))
            evaluator.load_state_dict(state, strict=True)
            predictions = predict_cohort(evaluator, development, device)
            release_comparison = compare_predictions(
                baseline_predictions,
                predictions,
                development,
                protocol["development"]["criteria"],
            )
            prior_comparison = compare_predictions(
                prior_candidate_predictions,
                predictions,
                development,
                protocol["development"]["prior_candidate_criteria"],
            )
            name = f"{'-'.join(group)}__alpha_{float(alpha):.2f}"
            row = dual_candidate_row(
                name=name,
                group=[str(value) for value in group],
                alpha=float(alpha),
                release_comparison=release_comparison,
                prior_comparison=prior_comparison,
            )
            candidate_rows.append(row)
            release_summary = row["comparisons"]["release_continuity"]["summary"]
            prior_summary = row["comparisons"]["prior_candidate_superiority"][
                "summary"
            ]
            release_checks = row["comparisons"]["release_continuity"]["checks"]
            prior_checks = row["comparisons"]["prior_candidate_superiority"][
                "checks"
            ]
            print(
                f"dev={name} pass={row['passed']} "
                f"vs_v51={release_summary['accuracy_delta']:+.5f} "
                f"vs_v51.1={prior_summary['accuracy_delta']:+.5f} "
                f"seeds={release_summary['nonregressing_seed_count']}/"
                f"{release_summary['seed_count']}+"
                f"{prior_summary['nonregressing_seed_count']}/"
                f"{prior_summary['seed_count']} checks="
                f"{sum(release_checks.values()) + sum(prior_checks.values())}/12",
                flush=True,
            )
            if selected_row is None or tuple(row["selection_score"]) > tuple(
                selected_row["selection_score"]
            ):
                selected_row = row
                selected_state = state

    assert selected_row is not None and selected_state is not None
    if not selected_row["passed"]:
        receipt = {
            "schema": SELECTION_SCHEMA,
            "created_at": utc_now(),
            "trusted_timestamp": False,
            "authentication": "none",
            "integrity_status": "content_bound_not_authenticated",
            "authority": dict(AUTHORITY),
            "protocol_sha256": protocol["protocol_sha256"],
            "decision": "no_development_candidate_passed",
            "passed": False,
            "development_dataset_sha256": development["dataset_sha256"],
            "member_receipts": member_receipts,
            "selected": selected_row,
            "candidates": candidate_rows,
            "environment": environment_binding(device),
        }
        receipt["selection_sha256"] = selection_digest(receipt)
        write_json_atomic(selection_path, receipt)
        raise RuntimeError("No development candidate passed; final cohort remains untouched")

    candidate_path = output_dir / "selected" / "cognitive_leap_ultra_v51_2.pth"
    selected_artifact = save_state(candidate_path, selected_state)
    lineage_manifest = write_lineage_manifest(
        output_dir,
        protocol,
        baseline_state,
        member_states,
        member_receipts,
        selected_row,
        selected_artifact,
    )
    lineage_verification = write_lineage_verification(
        output_dir,
        lineage_manifest,
        protocol,
        selected_artifact,
    )
    receipt = {
        "schema": SELECTION_SCHEMA,
        "created_at": utc_now(),
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "decision": "selected_and_frozen_for_single_final",
        "passed": True,
        "development_dataset_sha256": development["dataset_sha256"],
        "member_receipts": member_receipts,
        "selected": {**selected_row, "artifact": selected_artifact},
        "lineage_manifest": lineage_manifest,
        "lineage_verification": lineage_verification,
        "candidates": candidate_rows,
        "environment": environment_binding(device),
    }
    receipt["selection_sha256"] = selection_digest(receipt)
    write_json_atomic(selection_path, receipt)
    return selection_path


def finalize_once(output_dir: Path, device: torch.device) -> Path:
    protocol = load_and_validate_protocol(output_dir)
    if not bool(protocol.get("finalization_allowed")):
        raise RuntimeError(
            "This protocol was prepared in dirty development mode; the untouched "
            "final cohort cannot be consumed"
        )
    current_git = git_binding()
    frozen_git = protocol["git"]
    if (
        current_git["dirty"]
        or current_git["commit"] != frozen_git["commit"]
        or bool(frozen_git["dirty"])
    ):
        raise RuntimeError(
            "Finalization requires the same clean Git commit used at protocol freeze"
        )
    selection_path = output_dir / "selection_receipt.json"
    selection = load_json_strict(selection_path)
    if selection.get("schema") != SELECTION_SCHEMA or not selection.get("passed"):
        raise ValueError("A passing, frozen development selection is required")
    if selection.get("selection_sha256") != selection_digest(selection):
        raise ValueError("Development selection digest mismatch")
    if selection.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("Selection receipt does not belong to this protocol")
    lineage_record = selection.get("lineage_manifest", {})
    lineage_verification_record = selection.get("lineage_verification", {})
    lineage_verification_path = resolve_repo_relative(
        str(lineage_verification_record.get("path", ""))
    )
    if (
        not lineage_verification_path.is_file()
        or lineage_verification_path.stat().st_size
        != int(lineage_verification_record.get("size_bytes", -1))
        or sha256_file(lineage_verification_path)
        != lineage_verification_record.get("sha256")
    ):
        raise ValueError("Selection lineage verification is missing or changed")
    lineage_verification = load_json_strict(lineage_verification_path)
    if (
        lineage_verification.get("schema")
        != "supermix-cognitive-leap-lineage-verification-v1"
        or lineage_verification.get("valid") is not True
        or lineage_verification.get("authentication") != "none"
        or lineage_verification.get("integrity_status")
        != "content_bound_not_authenticated"
        or lineage_verification.get("trusted_timestamp") is not False
        or lineage_verification.get("authority") != AUTHORITY
        or lineage_verification.get("protocol_sha256")
        != protocol["protocol_sha256"]
        or lineage_verification.get("evaluation_profile_sha256")
        != protocol["evaluation_profile_sha256"]
        or lineage_verification.get("lineage_manifest") != lineage_record
    ):
        raise ValueError("Selection lineage verification contract mismatch")
    selected_artifact = selection["selected"]["artifact"]
    candidate_path = resolve_repo_relative(str(selected_artifact["path"]))
    if (
        not candidate_path.is_file()
        or candidate_path.stat().st_size != int(selected_artifact["size_bytes"])
        or sha256_file(candidate_path) != selected_artifact["sha256"]
    ):
        raise ValueError("Frozen candidate changed before finalization")
    baseline_state, member_states = validate_lineage_manifest(
        lineage_record,
        protocol,
        selected_artifact,
    )
    candidate_state = load_state(candidate_path)
    prior_candidate_state = load_state(
        resolve_repo_relative(str(protocol["prior_candidate"]["path"]))
    )
    if state_dict_summary(candidate_state) != {
        key: selected_artifact[key]
        for key in (
            "tensor_count",
            "element_count",
            "all_finite",
            "tensor_byte_order",
            "canonical_state_sha256",
        )
    }:
        raise ValueError("Frozen candidate state receipt changed")
    replay_development_selection(
        protocol,
        selection,
        baseline_state,
        member_states,
        device,
    )
    evaluator = ChampionNetCognitiveLeapUltraExpert().to(device)
    evaluator.load_state_dict(baseline_state, strict=True)
    started_path = output_dir / "finalization.started.json"
    complete_path = output_dir / "finalization.complete.json"
    prediction_path = output_dir / "final_three_way_predictions.jsonl.gz"
    receipt_path = output_dir / "three_way_evaluation_receipt.json"
    if any(
        path.exists()
        for path in (started_path, complete_path, prediction_path, receipt_path)
    ):
        raise FileExistsError("Final cohort has already been started or consumed")
    final_environment = environment_binding(device)
    final_invocation_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "selection_sha256": selection["selection_sha256"],
                "invocation": final_environment["invocation"],
                "torch": final_environment["torch"],
            }
        )
    )
    write_json_exclusive(
        started_path,
        {
            "schema": "supermix-cognitive-leap-finalization-started-v2",
            "started_at": utc_now(),
            "trusted_timestamp": False,
            "authentication": "none",
            "integrity_status": "content_bound_not_authenticated",
            "authority": dict(AUTHORITY),
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluation_profile_sha256": protocol[
                "evaluation_profile_sha256"
            ],
            "selection_receipt_sha256": sha256_file(selection_path),
            "selection_content_sha256": selection["selection_sha256"],
            "baseline_sha256": protocol["baseline"]["sha256"],
            "prior_candidate_sha256": protocol["prior_candidate"]["sha256"],
            "candidate_sha256": selected_artifact["sha256"],
            "lineage_manifest_sha256": lineage_record["sha256"],
            "lineage_verification_sha256": lineage_verification_record["sha256"],
            "code_bindings_sha256": sha256_bytes(
                canonical_json_bytes(protocol["code_bindings"])
            ),
            "dependency_lock_sha256": final_environment["dependency_lock_sha256"],
            "final_invocation_sha256": final_invocation_sha256,
            "scope": "single_use_for_this_local_output_directory_only",
        },
    )

    evaluation_rng_before_sha256 = sha256_bytes(
        torch.get_rng_state().numpy().tobytes()
    )
    final_cohort = build_cohort(
        protocol["final"]["seeds"],
        int(protocol["final"]["samples_per_seed"]),
        cohort_role="final",
    )
    release_baseline_predictions = predict_cohort(evaluator, final_cohort, device)
    evaluator.load_state_dict(prior_candidate_state, strict=True)
    prior_candidate_predictions = predict_cohort(evaluator, final_cohort, device)
    evaluator.load_state_dict(candidate_state, strict=True)
    candidate_predictions = predict_cohort(evaluator, final_cohort, device)
    evaluation_rng_after_sha256 = sha256_bytes(
        torch.get_rng_state().numpy().tobytes()
    )
    if evaluation_rng_after_sha256 != evaluation_rng_before_sha256:
        raise RuntimeError("Final evaluation unexpectedly changed the global CPU RNG state")
    prediction_artifact = write_three_way_final_prediction_artifact(
        prediction_path,
        release_baseline_predictions,
        prior_candidate_predictions,
        candidate_predictions,
        final_cohort,
        evaluation_profile_sha256=protocol["evaluation_profile_sha256"],
    )
    release_comparison = three_way_comparison(
        release_baseline_predictions,
        candidate_predictions,
        final_cohort,
        protocol["criteria"],
        prediction_artifact,
        evaluation_profile_sha256=protocol["evaluation_profile_sha256"],
    )
    prior_comparison = three_way_comparison(
        prior_candidate_predictions,
        candidate_predictions,
        final_cohort,
        protocol["prior_candidate_criteria"],
        prediction_artifact,
        evaluation_profile_sha256=protocol["evaluation_profile_sha256"],
    )
    passed = bool(release_comparison["passed"] and prior_comparison["passed"])
    protocol_path = output_dir / "protocol.json"
    receipt = {
        "schema": EVALUATION_SCHEMA,
        "created_at": utc_now(),
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "gate_outcome": "pass" if passed else "reject",
        "claim_scope": protocol["claim_scope"],
        "evaluation_profile": protocol["evaluation_profile"],
        "evaluation_profile_sha256": protocol["evaluation_profile_sha256"],
        "protocol": {
            "path": relative_path(protocol_path),
            "file_sha256": sha256_file(protocol_path),
            "size_bytes": protocol_path.stat().st_size,
            "content_sha256": protocol["protocol_sha256"],
        },
        "selection": {
            "path": relative_path(selection_path),
            "file_sha256": sha256_file(selection_path),
            "size_bytes": selection_path.stat().st_size,
            "content_sha256": selection["selection_sha256"],
        },
        "artifacts": {
            "release_baseline": protocol["baseline"],
            "prior_candidate": protocol["prior_candidate"],
            "candidate": selected_artifact,
        },
        "comparisons": {
            "release_continuity": release_comparison,
            "prior_candidate_superiority": prior_comparison,
        },
        "per_example_artifact": prediction_artifact,
        "code_bindings": protocol["code_bindings"],
        "source_snapshot": protocol["source_snapshot"],
        "git_at_protocol_freeze": protocol["git"],
        "git_at_finalization": current_git,
        "environment": final_environment,
        "evaluation_rng": {
            "cpu_state_before_sha256": evaluation_rng_before_sha256,
            "cpu_state_after_sha256": evaluation_rng_after_sha256,
            "unchanged": True,
        },
        "final_invocation_sha256": final_invocation_sha256,
        "single_use_scope": "this_local_output_directory_only",
    }
    receipt["receipt_id"] = sha256_bytes(canonical_json_bytes(receipt))
    write_json_atomic(receipt_path, receipt)
    from cognitive_leap_three_way_receipt import validate_receipt  # noqa: PLC0415

    validation = validate_receipt(receipt_path, root=REPO_ROOT)
    if not validation.get("valid"):
        raise ValueError("Written three-way evaluation receipt failed semantic validation")
    write_json_atomic(
        complete_path,
        {
            "schema": "supermix-cognitive-leap-finalization-complete-v2",
            "completed_at": utc_now(),
            "trusted_timestamp": False,
            "authentication": "none",
            "integrity_status": "content_bound_not_authenticated",
            "authority": dict(AUTHORITY),
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluation_profile_sha256": protocol[
                "evaluation_profile_sha256"
            ],
            "receipt_path": relative_path(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "gate_outcome": receipt["gate_outcome"],
            "release_continuity_passed": release_comparison["passed"],
            "prior_candidate_superiority_passed": prior_comparison["passed"],
        },
    )
    return receipt_path


def configure_runtime(threads: int) -> torch.device:
    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    return torch.device("cpu")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "train", "verify-development", "finalize"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / DEFAULT_OUTPUT_RELATIVE,
    )
    parser.add_argument("--train-size", type=int, default=12_000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--samples-per-seed", type=int, default=2_000)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--allow-dirty-development",
        action="store_true",
        help=(
            "Permit protocol preparation in a dirty tree for development screening; "
            "such a protocol is permanently forbidden from consuming final seeds"
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("Training dimensions must be positive")
    if args.samples_per_seed <= 0:
        raise ValueError("samples-per-seed must be positive")
    output_dir = args.output_dir.resolve()
    device = configure_runtime(args.torch_threads)
    if args.phase == "prepare":
        path = prepare_protocol(output_dir, args)
    elif args.phase == "train":
        path = train_and_select(output_dir, device)
    elif args.phase == "verify-development":
        path = verify_development_selection(output_dir, device)
    else:
        path = finalize_once(output_dir, device)
    print(path)


if __name__ == "__main__":
    main()
