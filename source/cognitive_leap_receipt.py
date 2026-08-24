"""Fail-closed validation for bounded Cognitive Leap evaluation receipts.

The receipt validated here is evidence, not authority.  A successful bounded
gate cannot activate, route, publish, or release a model.  Validation replays
the paired final-cohort evidence from exact little-endian float32 logits instead
of trusting the aggregate numbers written by the benchmark runner.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torch.nn.functional as F


RECEIPT_SCHEMA = "supermix-cognitive-leap-bounded-evaluation-v2"
PROTOCOL_SCHEMA = "supermix-cognitive-leap-bounded-protocol-v2"
SELECTION_SCHEMA = "supermix-cognitive-leap-development-selection-v2"
PREDICTION_ARTIFACT_SCHEMA = "supermix-cognitive-leap-paired-logits-jsonl-v1"
LINEAGE_SCHEMA = "supermix-cognitive-leap-lineage-v2"

AUTHORITY_KEYS = (
    "activation",
    "auto_route",
    "default_model",
    "fallback",
    "consultant",
    "tools",
    "permissions",
    "safety",
    "promotion",
    "store_publication",
    "release",
)
NO_AUTHORITY = {name: False for name in AUTHORITY_KEYS}

CHECK_KEYS = (
    "accuracy_gain",
    "paired_significance",
    "mean_loss_nonregression",
    "seed_nonregression",
    "operation_family_nonregression",
    "class_bounded_nonregression",
)
CRITERIA_KEYS = (
    "minimum_accuracy_gain",
    "maximum_p_value",
    "minimum_nonregressing_seed_fraction",
    "minimum_worst_seed_delta",
    "minimum_nonregressing_operation_families",
    "minimum_worst_operation_family_delta",
    "minimum_nonregressing_classes",
    "minimum_worst_class_delta",
    "require_mean_loss_nonregression",
)
SUMMARY_KEYS = (
    "seed_count",
    "n",
    "baseline_accuracy",
    "candidate_accuracy",
    "accuracy_delta",
    "wins",
    "regressions",
    "ties",
    "exact_mcnemar_p_two_sided",
    "mean_baseline_loss",
    "mean_candidate_loss",
    "required_nonregressing_seed_count",
    "nonregressing_seed_count",
    "worst_seed_delta",
    "eligible_family_count",
    "nonregressing_family_count",
    "worst_family_delta",
    "eligible_class_count",
    "nonregressing_class_count",
    "worst_class_delta",
)
ROW_KEYS = (
    "dataset_id",
    "cohort_role",
    "example_id",
    "seed",
    "index",
    "target",
    "start",
    "op_types",
    "operands",
    "operation_family_tags",
    "baseline_logits_f32le_hex",
    "candidate_logits_f32le_hex",
    "baseline_prediction",
    "candidate_prediction",
    "baseline_correct",
    "candidate_correct",
)
ARTIFACT_KEYS = (
    "schema",
    "path",
    "sha256",
    "size_bytes",
    "row_count",
    "format",
    "uncompressed_sha256",
    "dataset_id",
    "cohort_role",
    "class_order",
    "class_count",
    "logit_shape",
    "logits_encoding",
    "argmax_tie_rule",
    "loss_formula",
    "validation_absolute_tolerance",
    "gzip_mtime",
)

_OP_NAMES = ("add", "mul", "sub")
_FAMILY_NAMES = tuple(
    [f"first_{name}" for name in _OP_NAMES]
    + [f"mul_count_{count}" for count in range(5)]
)
_TOP_LEVEL_REQUIRED = {
    "schema",
    "receipt_id",
    "gate_outcome",
    "authority",
    "authentication",
    "integrity_status",
    "trusted_timestamp",
    "protocol",
    "selection",
    "artifacts",
    "criteria",
    "checks",
    "summary",
    "seed_rows",
    "operation_family_rows",
    "class_rows",
    "evidence",
    "per_example_artifact",
}
_TOP_LEVEL_OPTIONAL = {
    "created_at",
    "claim_scope",
    "code_bindings",
    "source_snapshot",
    "git_at_protocol_freeze",
    "git_at_finalization",
    "environment",
    "evaluation_rng",
    "final_invocation_sha256",
    "single_use_scope",
}


class ReceiptValidationError(ValueError):
    """Raised when a bounded receipt cannot be independently reproduced."""


def _reject_constant(value: str) -> None:
    raise ReceiptValidationError(f"Non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReceiptValidationError(f"Duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def loads_json_strict(data: str | bytes) -> Any:
    """Parse JSON while rejecting duplicate keys and NaN/Infinity."""

    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except ReceiptValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReceiptValidationError(f"Invalid JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical UTF-8 JSON and reject non-finite numeric values."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError(f"Value is not canonical JSON: {exc}") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptValidationError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ReceiptValidationError(f"{label} keys must be strings")
    return value


def _expect_exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ReceiptValidationError(
            f"{label} fields mismatch; missing={missing}, extra={extra}"
        )


def _expect_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptValidationError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ReceiptValidationError(f"{label} must be >= {minimum}")
    return value


def _expect_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReceiptValidationError(f"{label} must be finite")
    return result


def _expect_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ReceiptValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    if value != value.lower() or any(character not in "0123456789abcdef" for character in value):
        raise ReceiptValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _resolve_bound_path(root: Path, raw_path: Any, label: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ReceiptValidationError(f"{label} must be a nonempty relative path")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ReceiptValidationError(f"{label} must be relative to the validation root")
    root_resolved = root.resolve()
    resolved = (root_resolved / relative).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ReceiptValidationError(f"{label} escapes the validation root") from exc
    return resolved


def _verify_file_reference(
    reference: Mapping[str, Any],
    root: Path,
    label: str,
) -> Path:
    path = _resolve_bound_path(root, reference.get("path"), f"{label}.path")
    if not path.is_file():
        raise ReceiptValidationError(f"{label} is missing: {path}")
    expected_size = _expect_int(reference.get("size_bytes"), f"{label}.size_bytes", minimum=0)
    if path.stat().st_size != expected_size:
        raise ReceiptValidationError(f"{label} size mismatch")
    expected_hash = _expect_sha256(reference.get("sha256"), f"{label}.sha256")
    if sha256_file(path) != expected_hash:
        raise ReceiptValidationError(f"{label} SHA-256 mismatch")
    return path


def _little_endian_tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    array = value.numpy()
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and struct.pack("=H", 1) != b"\x01\x00"):
        array = array.byteswap().newbyteorder("<")
    return array.tobytes(order="C")


def tensor_digest_update(digest: Any, label: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(canonical_json_bytes(list(value.shape)) + b"\0")
    digest.update(_little_endian_tensor_bytes(value))


def state_dict_summary(state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Hash a tensor state dict independently of checkpoint container bytes."""

    if not isinstance(state, Mapping) or not state:
        raise ReceiptValidationError("State dict must be a nonempty mapping")
    digest = hashlib.sha256()
    element_count = 0
    all_finite = True
    for name in sorted(state):
        if not isinstance(name, str) or not isinstance(state[name], torch.Tensor):
            raise ReceiptValidationError("State dict must map string names to tensors")
        tensor = state[name].detach().cpu().contiguous()
        tensor_digest_update(digest, name, tensor)
        element_count += int(tensor.numel())
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all().item()
        ):
            all_finite = False
    return {
        "tensor_count": len(state),
        "element_count": element_count,
        "all_finite": all_finite,
        "canonical_state_sha256": digest.hexdigest(),
        "tensor_byte_order": "little_endian",
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


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - torch supplies many exception classes
        raise ReceiptValidationError(f"Could not load checkpoint {path}: {exc}") from exc
    if isinstance(loaded, Mapping) and isinstance(loaded.get("state_dict"), Mapping):
        loaded = loaded["state_dict"]
    if not isinstance(loaded, Mapping) or not loaded:
        raise ReceiptValidationError(f"Checkpoint is not a nonempty state dict: {path}")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ReceiptValidationError(f"Checkpoint contains a non-tensor entry: {name!r}")
        state[name] = tensor.detach().cpu()
    if not state_dict_summary(state)["all_finite"]:
        raise ReceiptValidationError(f"Checkpoint contains non-finite tensors: {path}")
    return state


def average_state_dicts(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Reconstruct a weighted soup, rejecting incompatible non-float tensors."""

    if not states:
        raise ReceiptValidationError("At least one state dict is required")
    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states[1:]):
        raise ReceiptValidationError("State dict key/order mismatch")
    raw_weights = tuple(float(value) for value in (weights or (1.0,) * len(states)))
    if len(raw_weights) != len(states):
        raise ReceiptValidationError("State and weight counts differ")
    if any(not math.isfinite(value) or value < 0.0 for value in raw_weights):
        raise ReceiptValidationError("State weights must be finite and nonnegative")
    total = sum(raw_weights)
    if total <= 0.0:
        raise ReceiptValidationError("State weights must have positive mass")
    normalized = tuple(value / total for value in raw_weights)
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key].detach().cpu() for state in states]
        first = tensors[0]
        if any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in tensors[1:]
        ):
            raise ReceiptValidationError(f"State tensor metadata differs for {key}")
        if first.is_floating_point() or first.is_complex():
            mixed = torch.zeros_like(first)
            for weight, tensor in zip(normalized, tensors):
                mixed.add_(tensor, alpha=weight)
            result[key] = mixed
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ReceiptValidationError(f"Non-floating state differs for {key}")
            result[key] = first.clone()
    return result


def validate_state_reconstruction(
    parents: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
    expected: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    reconstructed = average_state_dicts(parents, weights)
    reconstructed_summary = state_dict_summary(reconstructed)
    expected_summary = state_dict_summary(expected)
    if reconstructed_summary != expected_summary:
        raise ReceiptValidationError("Reconstructed state hash does not match expected state")
    for name in reconstructed:
        if not torch.equal(reconstructed[name], expected[name]):
            raise ReceiptValidationError(f"Reconstructed tensor differs for {name}")
    return reconstructed_summary


def encode_logits_f32le_hex(logits: Sequence[float]) -> str:
    if len(logits) != 10:
        raise ReceiptValidationError("Exactly 10 logits are required")
    values = tuple(float(value) for value in logits)
    if any(not math.isfinite(value) for value in values):
        raise ReceiptValidationError("Logits must be finite")
    return struct.pack("<10f", *values).hex()


def decode_logits_f32le_hex(value: Any, label: str = "logits") -> tuple[float, ...]:
    if not isinstance(value, str) or value != value.lower() or len(value) != 80:
        raise ReceiptValidationError(f"{label} must encode 10 little-endian float32 values")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ReceiptValidationError(f"{label} is not hexadecimal") from exc
    if raw.hex() != value:
        raise ReceiptValidationError(f"{label} is not canonical lowercase hexadecimal")
    values = struct.unpack("<10f", raw)
    if any(not math.isfinite(item) for item in values):
        raise ReceiptValidationError(f"{label} contains NaN or infinity")
    return values


def _criterion_fraction(value: Any) -> Fraction:
    """Interpret a JSON numeric criterion as its exact decimal rational value."""

    if isinstance(value, bool):
        raise ReceiptValidationError("Boolean values are not numeric criteria")
    try:
        return Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ReceiptValidationError(f"Invalid numeric criterion: {value!r}") from exc


def _ratio_at_least(numerator: int, denominator: int, threshold: Any) -> bool:
    if int(denominator) <= 0:
        raise ReceiptValidationError("Ratio denominator must be positive")
    bound = _criterion_fraction(threshold)
    return int(numerator) * bound.denominator >= bound.numerator * int(denominator)


def _ratio_at_most(numerator: int, denominator: int, threshold: Any) -> bool:
    if int(denominator) <= 0:
        raise ReceiptValidationError("Ratio denominator must be positive")
    bound = _criterion_fraction(threshold)
    return int(numerator) * bound.denominator <= bound.numerator * int(denominator)


def _ceil_scaled_fraction(count: int, fraction: Any) -> int:
    bound = _criterion_fraction(fraction)
    numerator = int(count) * bound.numerator
    return -(-numerator // bound.denominator)


def _exact_mcnemar_terms(wins: int, regressions: int) -> tuple[int, int]:
    wins = _expect_int(wins, "wins", minimum=0)
    regressions = _expect_int(regressions, "regressions", minimum=0)
    discordant = wins + regressions
    if discordant == 0:
        return 1, 1
    tail = min(wins, regressions)
    numerator = 2 * sum(math.comb(discordant, index) for index in range(tail + 1))
    denominator = 1 << discordant
    return (denominator, denominator) if numerator >= denominator else (numerator, denominator)


def exact_mcnemar_two_sided(wins: int, regressions: int) -> float:
    numerator, denominator = _exact_mcnemar_terms(wins, regressions)
    return numerator / denominator


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    payload = dict(protocol)
    payload.pop("protocol_sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _canonical_task(
    samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Independent replay of the fixed v51 chained-modulo-10 generator."""

    generator = torch.Generator().manual_seed(seed)
    x = torch.zeros(samples, 128)
    y = torch.zeros(samples, dtype=torch.long)
    starts = torch.randint(0, 10, (samples,), generator=generator)
    op_types = torch.randint(0, 3, (samples, 4), generator=generator)
    operands = torch.randint(1, 10, (samples, 4), generator=generator)
    for index in range(samples):
        x[index, starts[index]] = 1.0
        accumulator = int(starts[index])
        for operation_index in range(4):
            base = 10 + operation_index * 24
            operation = int(op_types[index, operation_index])
            operand = int(operands[index, operation_index])
            x[index, base + operation] = 1.0
            x[index, base + 3 + operand] = 1.0
            if operation == 0:
                accumulator = (accumulator + operand) % 10
            elif operation == 1:
                accumulator = (accumulator * operand) % 10
            else:
                accumulator = (accumulator - operand) % 10
        y[index] = accumulator
    x = x + 0.01 * torch.randn(samples, 128, generator=generator)
    return x.unsqueeze(1), y, {
        "starts": starts,
        "op_types": op_types,
        "operands": operands,
    }


def dataset_sha256(seeds: Sequence[int], samples_per_seed: int) -> str:
    digest = hashlib.sha256()
    for seed in seeds:
        x, y, metadata = _canonical_task(samples_per_seed, int(seed))
        digest.update(struct.pack("<q", int(seed)))
        tensor_digest_update(digest, "x", x)
        tensor_digest_update(digest, "y", y)
        for name in ("starts", "op_types", "operands"):
            tensor_digest_update(digest, name, metadata[name])
    return digest.hexdigest()


def dataset_id_for(specification_sha256: str, dataset_content_sha256: str) -> str:
    """Bind a cohort specification and its generated tensor content."""

    return sha256_bytes(
        canonical_json_bytes(
            {
                "specification_sha256": _expect_sha256(
                    specification_sha256,
                    "dataset specification SHA-256",
                ),
                "dataset_sha256": _expect_sha256(
                    dataset_content_sha256,
                    "dataset content SHA-256",
                ),
            }
        )
    )


def _validate_cohort_specification(
    specification: Any,
    *,
    seeds: Sequence[int],
    samples_per_seed: int,
) -> tuple[Mapping[str, Any], str]:
    value = _expect_mapping(specification, "cohort specification")
    _expect_exact_keys(
        value,
        (
            "schema",
            "generator_schema",
            "family_tag_schema",
            "cohort_role",
            "seeds",
            "samples_per_seed",
            "generator_source_sha256",
        ),
        "cohort specification",
    )
    if (
        value["schema"] != "supermix-cognitive-leap-cohort-v1"
        or value["generator_schema"] != "supermix-cognitive-leap-generator-v1"
        or value["family_tag_schema"]
        != "supermix-cognitive-leap-family-tags-v1"
        or value["cohort_role"] != "final"
        or value["seeds"] != list(seeds)
        or value["samples_per_seed"] != samples_per_seed
    ):
        raise ReceiptValidationError("Final cohort specification mismatch")
    _expect_sha256(
        value["generator_source_sha256"],
        "cohort specification generator_source_sha256",
    )
    return value, sha256_bytes(canonical_json_bytes(value))


def _operation_families(op_types: Sequence[int]) -> list[str]:
    return [
        f"first_{_OP_NAMES[op_types[0]]}",
        f"mul_count_{sum(value == 1 for value in op_types)}",
    ]


def _argmax_lowest(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _digest_integer_tensor(
    digest: Any,
    label: str,
    values: Sequence[int],
    *,
    dtype: str,
    format_code: str,
) -> None:
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(dtype.encode("ascii") + b"\0")
    digest.update(canonical_json_bytes([len(values)]) + b"\0")
    digest.update(struct.pack(f"<{len(values)}{format_code}", *values))


def _validate_criteria(criteria: Any) -> dict[str, Any]:
    value = _expect_mapping(criteria, "criteria")
    _expect_exact_keys(value, CRITERIA_KEYS, "criteria")
    result = dict(value)
    for key in CRITERIA_KEYS[:-1]:
        if key in {
            "minimum_nonregressing_operation_families",
            "minimum_nonregressing_classes",
        }:
            _expect_int(result[key], f"criteria.{key}", minimum=0)
        else:
            _expect_number(result[key], f"criteria.{key}")
    if not isinstance(result["require_mean_loss_nonregression"], bool):
        raise ReceiptValidationError(
            "criteria.require_mean_loss_nonregression must be boolean"
        )
    fraction = _criterion_fraction(result["minimum_nonregressing_seed_fraction"])
    p_value = _criterion_fraction(result["maximum_p_value"])
    if not Fraction(0) <= fraction <= Fraction(1) or not Fraction(0) <= p_value <= Fraction(1):
        raise ReceiptValidationError("Criteria fraction and p-value must be in [0, 1]")
    return result


def _group_row(name_key: str, name: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    n = int(stats["n"])
    baseline_correct = int(stats["baseline_correct"])
    candidate_correct = int(stats["candidate_correct"])
    baseline = baseline_correct / n if n else 0.0
    candidate = candidate_correct / n if n else 0.0
    return {
        name_key: name,
        "n": n,
        "baseline_accuracy": baseline,
        "candidate_accuracy": candidate,
        "delta": (candidate_correct - baseline_correct) / n if n else 0.0,
    }


def _new_group_stats() -> dict[str, int]:
    return {"n": 0, "baseline_correct": 0, "candidate_correct": 0}


def _validate_artifact_contract(artifact: Any, expected_rows: int) -> Mapping[str, Any]:
    value = _expect_mapping(artifact, "per_example_artifact")
    _expect_exact_keys(value, ARTIFACT_KEYS, "per_example_artifact")
    if value["schema"] != PREDICTION_ARTIFACT_SCHEMA:
        raise ReceiptValidationError("Unsupported per-example artifact schema")
    if value["format"] != "gzip_jsonl":
        raise ReceiptValidationError("Per-example artifact format must be gzip_jsonl")
    if value["cohort_role"] != "final":
        raise ReceiptValidationError("Per-example artifact cohort_role must be final")
    if not isinstance(value["dataset_id"], str) or not value["dataset_id"]:
        raise ReceiptValidationError("Per-example artifact dataset_id is required")
    if (
        value["class_count"] != 10
        or value["class_order"] != list(range(10))
        or value["logit_shape"] != [10]
    ):
        raise ReceiptValidationError("Per-example logits must use canonical classes 0..9")
    if value["logits_encoding"] != "hex_little_endian_float32":
        raise ReceiptValidationError("Unsupported per-example logits encoding")
    if value["argmax_tie_rule"] != "lowest_class_index":
        raise ReceiptValidationError("Unsupported argmax tie rule")
    if (
        value["loss_formula"]
        != "torch_cross_entropy_float32_sum_per_seed_then_float64_total"
    ):
        raise ReceiptValidationError("Unsupported loss formula")
    if _expect_number(
        value["validation_absolute_tolerance"],
        "per_example_artifact.validation_absolute_tolerance",
    ) != 1e-6:
        raise ReceiptValidationError("Validation tolerance must be exactly 1e-6")
    if _expect_int(value["gzip_mtime"], "per_example_artifact.gzip_mtime") != 0:
        raise ReceiptValidationError("gzip mtime must be zero")
    if _expect_int(value["row_count"], "per_example_artifact.row_count", minimum=0) != expected_rows:
        raise ReceiptValidationError("Per-example artifact row count mismatch")
    _expect_sha256(value["sha256"], "per_example_artifact.sha256")
    _expect_sha256(
        value["uncompressed_sha256"],
        "per_example_artifact.uncompressed_sha256",
    )
    return value


def _read_required_line(handle: BinaryIO, label: str) -> bytes:
    try:
        line = handle.readline()
    except (OSError, EOFError) as exc:
        raise ReceiptValidationError(f"Could not decompress {label}: {exc}") from exc
    if not line:
        raise ReceiptValidationError(f"{label} is truncated")
    if not line.endswith(b"\n"):
        raise ReceiptValidationError(f"{label} row is missing its canonical newline")
    return line


def validate_prediction_artifact(
    artifact: Mapping[str, Any],
    *,
    root: Path,
    seeds: Sequence[int],
    samples_per_seed: int,
    criteria: Mapping[str, Any],
    cohort_specification: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay a paired gzip JSONL artifact and return independently computed metrics."""

    seed_values = [_expect_int(seed, "final seed") for seed in seeds]
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ReceiptValidationError("Final seeds must be nonempty and unique")
    sample_count = _expect_int(samples_per_seed, "samples_per_seed", minimum=1)
    criteria_value = _validate_criteria(criteria)
    _specification, specification_sha256 = _validate_cohort_specification(
        cohort_specification,
        seeds=seed_values,
        samples_per_seed=sample_count,
    )
    artifact_value = _validate_artifact_contract(
        artifact,
        len(seed_values) * sample_count,
    )
    artifact_path = _verify_file_reference(artifact_value, root, "per_example_artifact")
    compressed = artifact_path.read_bytes()
    if len(compressed) < 10 or compressed[:2] != b"\x1f\x8b":
        raise ReceiptValidationError("Per-example artifact is not gzip data")
    if int.from_bytes(compressed[4:8], "little") != 0:
        raise ReceiptValidationError("Per-example gzip header mtime is not zero")

    uncompressed_digest = hashlib.sha256()
    baseline_logits_digest = hashlib.sha256()
    candidate_logits_digest = hashlib.sha256()
    baseline_per_example_digest = hashlib.sha256()
    candidate_per_example_digest = hashlib.sha256()
    baseline_prediction_digest = hashlib.sha256()
    candidate_prediction_digest = hashlib.sha256()
    paired_outcome_digest = hashlib.sha256()
    cohort_digest = hashlib.sha256()
    seed_rows: list[dict[str, Any]] = []
    family_stats = {name: _new_group_stats() for name in _FAMILY_NAMES}
    class_stats = {str(index): _new_group_stats() for index in range(10)}
    total_baseline_loss = 0.0
    total_candidate_loss = 0.0
    total_baseline_correct = 0
    total_candidate_correct = 0
    total_wins = 0
    total_regressions = 0
    total_ties = 0
    row_count = 0
    dataset_id = str(artifact_value["dataset_id"])

    try:
        with gzip.open(artifact_path, "rb") as handle:
            for seed in seed_values:
                x, y, metadata = _canonical_task(sample_count, seed)
                cohort_digest.update(struct.pack("<q", seed))
                tensor_digest_update(cohort_digest, "x", x)
                tensor_digest_update(cohort_digest, "y", y)
                for metadata_name in ("starts", "op_types", "operands"):
                    tensor_digest_update(
                        cohort_digest,
                        metadata_name,
                        metadata[metadata_name],
                    )

                seed_targets: list[int] = []
                seed_baseline_predictions: list[int] = []
                seed_candidate_predictions: list[int] = []
                seed_outcomes: list[int] = []
                seed_baseline_logits: list[tuple[float, ...]] = []
                seed_candidate_logits: list[tuple[float, ...]] = []
                seed_baseline_correct = 0
                seed_candidate_correct = 0
                seed_wins = 0
                seed_regressions = 0

                for index in range(sample_count):
                    line = _read_required_line(handle, "per-example artifact")
                    uncompressed_digest.update(line)
                    row_count += 1
                    row = loads_json_strict(line[:-1])
                    row_value = _expect_mapping(row, f"row {row_count}")
                    _expect_exact_keys(row_value, ROW_KEYS, f"row {row_count}")
                    if canonical_json_bytes(row_value) + b"\n" != line:
                        raise ReceiptValidationError(
                            f"Row {row_count} is not canonical JSON"
                        )
                    expected_id = f"{dataset_id}:{seed}:{index}"
                    if (
                        row_value["dataset_id"] != dataset_id
                        or row_value["cohort_role"] != "final"
                        or row_value["example_id"] != expected_id
                        or row_value["seed"] != seed
                        or row_value["index"] != index
                    ):
                        raise ReceiptValidationError(
                            f"Row {row_count} order or content-bound identity mismatch"
                        )

                    target = _expect_int(row_value["target"], f"row {row_count}.target")
                    start = _expect_int(row_value["start"], f"row {row_count}.start")
                    op_types = row_value["op_types"]
                    operands = row_value["operands"]
                    if (
                        not isinstance(op_types, list)
                        or len(op_types) != 4
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or not 0 <= value <= 2
                            for value in op_types
                        )
                    ):
                        raise ReceiptValidationError(f"Row {row_count} has invalid op_types")
                    if (
                        not isinstance(operands, list)
                        or len(operands) != 4
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            or not 1 <= value <= 9
                            for value in operands
                        )
                    ):
                        raise ReceiptValidationError(f"Row {row_count} has invalid operands")
                    expected_target = int(y[index].item())
                    expected_start = int(metadata["starts"][index].item())
                    expected_ops = [int(value) for value in metadata["op_types"][index]]
                    expected_operands = [
                        int(value) for value in metadata["operands"][index]
                    ]
                    if (
                        target != expected_target
                        or start != expected_start
                        or op_types != expected_ops
                        or operands != expected_operands
                    ):
                        raise ReceiptValidationError(
                            f"Row {row_count} does not match the frozen dataset generator"
                        )
                    expected_families = _operation_families(op_types)
                    if row_value["operation_family_tags"] != expected_families:
                        raise ReceiptValidationError(
                            f"Row {row_count} operation family tags mismatch"
                        )

                    baseline_logits = decode_logits_f32le_hex(
                        row_value["baseline_logits_f32le_hex"],
                        f"row {row_count}.baseline_logits",
                    )
                    candidate_logits = decode_logits_f32le_hex(
                        row_value["candidate_logits_f32le_hex"],
                        f"row {row_count}.candidate_logits",
                    )
                    baseline_prediction = _argmax_lowest(baseline_logits)
                    candidate_prediction = _argmax_lowest(candidate_logits)
                    baseline_correct = baseline_prediction == target
                    candidate_correct = candidate_prediction == target
                    if (
                        row_value["baseline_prediction"] != baseline_prediction
                        or row_value["candidate_prediction"] != candidate_prediction
                        or type(row_value["baseline_correct"]) is not bool
                        or type(row_value["candidate_correct"]) is not bool
                        or row_value["baseline_correct"] is not baseline_correct
                        or row_value["candidate_correct"] is not candidate_correct
                    ):
                        raise ReceiptValidationError(
                            f"Row {row_count} prediction or correctness mismatch"
                        )
                    baseline_per_example_digest.update(
                        canonical_json_bytes(
                            {
                                "seed": seed,
                                "index": index,
                                "target": target,
                                "prediction": baseline_prediction,
                                "correct": baseline_correct,
                                "logits_f32le_hex": row_value[
                                    "baseline_logits_f32le_hex"
                                ],
                            }
                        )
                        + b"\n"
                    )
                    candidate_per_example_digest.update(
                        canonical_json_bytes(
                            {
                                "seed": seed,
                                "index": index,
                                "target": target,
                                "prediction": candidate_prediction,
                                "correct": candidate_correct,
                                "logits_f32le_hex": row_value[
                                    "candidate_logits_f32le_hex"
                                ],
                            }
                        )
                        + b"\n"
                    )

                    outcome = int(candidate_correct) - int(baseline_correct)
                    seed_targets.append(target)
                    seed_baseline_predictions.append(baseline_prediction)
                    seed_candidate_predictions.append(candidate_prediction)
                    seed_outcomes.append(outcome)
                    seed_baseline_logits.append(baseline_logits)
                    seed_candidate_logits.append(candidate_logits)
                    seed_baseline_correct += int(baseline_correct)
                    seed_candidate_correct += int(candidate_correct)
                    seed_wins += int(outcome == 1)
                    seed_regressions += int(outcome == -1)
                    for family in expected_families:
                        family_stats[family]["n"] += 1
                        family_stats[family]["baseline_correct"] += int(baseline_correct)
                        family_stats[family]["candidate_correct"] += int(candidate_correct)
                    class_row = class_stats[str(target)]
                    class_row["n"] += 1
                    class_row["baseline_correct"] += int(baseline_correct)
                    class_row["candidate_correct"] += int(candidate_correct)

                baseline_tensor = torch.tensor(seed_baseline_logits, dtype=torch.float32)
                candidate_tensor = torch.tensor(seed_candidate_logits, dtype=torch.float32)
                target_tensor = torch.tensor(seed_targets, dtype=torch.long)
                baseline_logits_digest.update(struct.pack("<q", seed))
                tensor_digest_update(
                    baseline_logits_digest,
                    "logits_f32",
                    baseline_tensor,
                )
                candidate_logits_digest.update(struct.pack("<q", seed))
                tensor_digest_update(
                    candidate_logits_digest,
                    "logits_f32",
                    candidate_tensor,
                )
                baseline_loss_sum = float(
                    F.cross_entropy(baseline_tensor, target_tensor, reduction="sum").item()
                )
                candidate_loss_sum = float(
                    F.cross_entropy(candidate_tensor, target_tensor, reduction="sum").item()
                )
                baseline_prediction_digest.update(struct.pack("<q", seed))
                _digest_integer_tensor(
                    baseline_prediction_digest,
                    "targets",
                    seed_targets,
                    dtype="torch.int64",
                    format_code="q",
                )
                _digest_integer_tensor(
                    baseline_prediction_digest,
                    "predictions",
                    seed_baseline_predictions,
                    dtype="torch.int64",
                    format_code="q",
                )
                candidate_prediction_digest.update(struct.pack("<q", seed))
                _digest_integer_tensor(
                    candidate_prediction_digest,
                    "targets",
                    seed_targets,
                    dtype="torch.int64",
                    format_code="q",
                )
                _digest_integer_tensor(
                    candidate_prediction_digest,
                    "predictions",
                    seed_candidate_predictions,
                    dtype="torch.int64",
                    format_code="q",
                )
                paired_outcome_digest.update(struct.pack("<q", seed))
                _digest_integer_tensor(
                    paired_outcome_digest,
                    "paired_outcome",
                    seed_outcomes,
                    dtype="torch.int8",
                    format_code="b",
                )
                seed_n = len(seed_targets)
                seed_ties = seed_n - seed_wins - seed_regressions
                seed_rows.append(
                    {
                        "seed": seed,
                        "n": seed_n,
                        "baseline_accuracy": seed_baseline_correct / seed_n,
                        "candidate_accuracy": seed_candidate_correct / seed_n,
                        "accuracy_delta": (
                            seed_candidate_correct - seed_baseline_correct
                        )
                        / seed_n,
                        "baseline_loss": baseline_loss_sum / seed_n,
                        "candidate_loss": candidate_loss_sum / seed_n,
                        "wins": seed_wins,
                        "regressions": seed_regressions,
                        "ties": seed_ties,
                    }
                )
                total_baseline_loss += baseline_loss_sum
                total_candidate_loss += candidate_loss_sum
                total_baseline_correct += seed_baseline_correct
                total_candidate_correct += seed_candidate_correct
                total_wins += seed_wins
                total_regressions += seed_regressions
                total_ties += seed_ties

            try:
                trailing = handle.read(1)
            except (OSError, EOFError) as exc:
                raise ReceiptValidationError(
                    f"Could not finish decompressing per-example artifact: {exc}"
                ) from exc
            if trailing:
                raise ReceiptValidationError("Per-example artifact has extra rows or bytes")
    except ReceiptValidationError:
        raise
    except (OSError, EOFError) as exc:
        raise ReceiptValidationError(f"Invalid gzip evidence artifact: {exc}") from exc

    if row_count != int(artifact_value["row_count"]):
        raise ReceiptValidationError("Per-example artifact row count mismatch")
    uncompressed_sha256 = uncompressed_digest.hexdigest()
    if uncompressed_sha256 != artifact_value["uncompressed_sha256"]:
        raise ReceiptValidationError("Per-example uncompressed SHA-256 mismatch")

    cohort_sha256 = cohort_digest.hexdigest()
    expected_dataset_id = dataset_id_for(specification_sha256, cohort_sha256)
    if dataset_id != expected_dataset_id:
        raise ReceiptValidationError("Per-example artifact dataset_id mismatch")

    total_n = row_count
    family_rows = [
        _group_row("family", name, family_stats[name]) for name in _FAMILY_NAMES
    ]
    class_rows = [
        _group_row("class", str(index), class_stats[str(index)])
        for index in range(10)
    ]
    family_deltas = [
        (
            int(family_stats[name]["candidate_correct"])
            - int(family_stats[name]["baseline_correct"]),
            int(family_stats[name]["n"]),
        )
        for name in _FAMILY_NAMES
    ]
    class_deltas = [
        (
            int(class_stats[str(index)]["candidate_correct"])
            - int(class_stats[str(index)]["baseline_correct"]),
            int(class_stats[str(index)]["n"]),
        )
        for index in range(10)
    ]
    eligible_family_pairs = [
        (row, exact)
        for row, exact in zip(family_rows, family_deltas)
        if exact[1] > 0
    ]
    eligible_class_pairs = [
        (row, exact)
        for row, exact in zip(class_rows, class_deltas)
        if exact[1] > 0
    ]
    eligible_families = [row for row, _exact in eligible_family_pairs]
    eligible_classes = [row for row, _exact in eligible_class_pairs]
    eligible_family_deltas = [exact for _row, exact in eligible_family_pairs]
    eligible_class_deltas = [exact for _row, exact in eligible_class_pairs]
    if not eligible_families or not eligible_classes:
        raise ReceiptValidationError("Final evidence has no eligible family or class groups")
    seed_deltas = [
        (int(row["wins"]) - int(row["regressions"]), int(row["n"]))
        for row in seed_rows
    ]
    nonregressing_seed_count = sum(delta >= 0 for delta, _count in seed_deltas)
    required_seed_count = _ceil_scaled_fraction(
        len(seed_rows), criteria_value["minimum_nonregressing_seed_fraction"]
    )
    nonregressing_family_count = sum(
        delta >= 0 for delta, _count in eligible_family_deltas
    )
    nonregressing_class_count = sum(
        delta >= 0 for delta, _count in eligible_class_deltas
    )
    mcnemar_numerator, mcnemar_denominator = _exact_mcnemar_terms(
        total_wins, total_regressions
    )
    summary = {
        "seed_count": len(seed_rows),
        "n": total_n,
        "baseline_accuracy": total_baseline_correct / total_n,
        "candidate_accuracy": total_candidate_correct / total_n,
        "accuracy_delta": (total_candidate_correct - total_baseline_correct) / total_n,
        "wins": total_wins,
        "regressions": total_regressions,
        "ties": total_ties,
        "exact_mcnemar_p_two_sided": exact_mcnemar_two_sided(
            total_wins,
            total_regressions,
        ),
        "mean_baseline_loss": total_baseline_loss / total_n,
        "mean_candidate_loss": total_candidate_loss / total_n,
        "required_nonregressing_seed_count": required_seed_count,
        "nonregressing_seed_count": nonregressing_seed_count,
        "worst_seed_delta": min(row["accuracy_delta"] for row in seed_rows),
        "eligible_family_count": len(eligible_families),
        "nonregressing_family_count": nonregressing_family_count,
        "worst_family_delta": min(row["delta"] for row in eligible_families),
        "eligible_class_count": len(eligible_classes),
        "nonregressing_class_count": nonregressing_class_count,
        "worst_class_delta": min(row["delta"] for row in eligible_classes),
    }
    checks = {
        "accuracy_gain": _ratio_at_least(
            total_candidate_correct - total_baseline_correct,
            total_n,
            criteria_value["minimum_accuracy_gain"],
        ),
        "paired_significance": _ratio_at_most(
            mcnemar_numerator,
            mcnemar_denominator,
            criteria_value["maximum_p_value"],
        ),
        "mean_loss_nonregression": (
            not bool(criteria_value["require_mean_loss_nonregression"])
            or summary["mean_candidate_loss"] <= summary["mean_baseline_loss"]
        ),
        "seed_nonregression": (
            summary["nonregressing_seed_count"] >= required_seed_count
            and all(
                _ratio_at_least(
                    delta,
                    count,
                    criteria_value["minimum_worst_seed_delta"],
                )
                for delta, count in seed_deltas
            )
        ),
        "operation_family_nonregression": (
            summary["nonregressing_family_count"]
            >= int(criteria_value["minimum_nonregressing_operation_families"])
            and all(
                _ratio_at_least(
                    delta,
                    count,
                    criteria_value["minimum_worst_operation_family_delta"],
                )
                for delta, count in eligible_family_deltas
            )
        ),
        "class_bounded_nonregression": (
            summary["nonregressing_class_count"]
            >= int(criteria_value["minimum_nonregressing_classes"])
            and all(
                _ratio_at_least(
                    delta,
                    count,
                    criteria_value["minimum_worst_class_delta"],
                )
                for delta, count in eligible_class_deltas
            )
        ),
    }
    return {
        "gate_outcome": "pass" if all(checks.values()) else "reject",
        "checks": checks,
        "summary": summary,
        "seed_rows": seed_rows,
        "operation_family_rows": family_rows,
        "class_rows": class_rows,
        "evidence": {
            "cohort_schema": "supermix-cognitive-leap-cohort-v1",
            "generator_schema": "supermix-cognitive-leap-generator-v1",
            "family_tag_schema": "supermix-cognitive-leap-family-tags-v1",
            "cohort_role": "final",
            "dataset_id": dataset_id,
            "dataset_specification_sha256": specification_sha256,
            "dataset_sha256": cohort_sha256,
            "baseline_prediction_sha256": baseline_prediction_digest.hexdigest(),
            "candidate_prediction_sha256": candidate_prediction_digest.hexdigest(),
            "baseline_logits_sha256": baseline_logits_digest.hexdigest(),
            "candidate_logits_sha256": candidate_logits_digest.hexdigest(),
            "baseline_per_example_sha256": baseline_per_example_digest.hexdigest(),
            "candidate_per_example_sha256": candidate_per_example_digest.hexdigest(),
            "paired_outcome_sha256": paired_outcome_digest.hexdigest(),
            "per_example_compressed_sha256": artifact_value["sha256"],
            "per_example_uncompressed_sha256": uncompressed_sha256,
        },
    }


def _assert_reproduced(
    claimed: Any,
    reproduced: Any,
    label: str,
    *,
    tolerance: float,
) -> None:
    if isinstance(reproduced, Mapping):
        claimed_mapping = _expect_mapping(claimed, label)
        _expect_exact_keys(claimed_mapping, reproduced.keys(), label)
        for key in reproduced:
            _assert_reproduced(
                claimed_mapping[key],
                reproduced[key],
                f"{label}.{key}",
                tolerance=tolerance,
            )
        return
    if isinstance(reproduced, list):
        if not isinstance(claimed, list) or len(claimed) != len(reproduced):
            raise ReceiptValidationError(f"{label} list length mismatch")
        for index, (claimed_item, reproduced_item) in enumerate(
            zip(claimed, reproduced)
        ):
            _assert_reproduced(
                claimed_item,
                reproduced_item,
                f"{label}[{index}]",
                tolerance=tolerance,
            )
        return
    if isinstance(reproduced, bool):
        if type(claimed) is not bool or claimed is not reproduced:
            raise ReceiptValidationError(f"{label} boolean mismatch")
        return
    if isinstance(reproduced, float):
        claimed_number = _expect_number(claimed, label)
        if not math.isclose(
            claimed_number,
            reproduced,
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            raise ReceiptValidationError(
                f"{label} mismatch: claimed={claimed_number}, reproduced={reproduced}"
            )
        return
    if claimed != reproduced or type(claimed) is not type(reproduced):
        raise ReceiptValidationError(
            f"{label} mismatch: claimed={claimed!r}, reproduced={reproduced!r}"
        )


def receipt_id_for(receipt: Mapping[str, Any]) -> str:
    """Hash the complete canonical receipt body, excluding only its own ID."""

    payload = dict(receipt)
    payload.pop("receipt_id", None)
    return sha256_bytes(canonical_json_bytes(payload))


def _validate_checkpoint_reference(
    reference: Any,
    *,
    root: Path,
    label: str,
    verify_state: bool,
) -> dict[str, torch.Tensor] | None:
    value = _expect_mapping(reference, label)
    required = {"path", "sha256", "size_bytes"}
    state_fields = {
        "tensor_count",
        "element_count",
        "all_finite",
        "canonical_state_sha256",
        "tensor_byte_order",
        "strict_load",
        "tensor_inventory",
    }
    if not required.issubset(value) or not set(value).issubset(required | state_fields):
        raise ReceiptValidationError(f"{label} checkpoint reference fields mismatch")
    path = _verify_file_reference(value, root, label)
    if not verify_state:
        return None
    if not state_fields.issubset(value):
        raise ReceiptValidationError(f"{label} lacks canonical state summary")
    if value["all_finite"] is not True or value["strict_load"] is not True:
        raise ReceiptValidationError(f"{label} must claim finite strict-load state")
    if value["tensor_byte_order"] != "little_endian":
        raise ReceiptValidationError(f"{label} tensor byte order mismatch")
    state = load_state_dict(path)
    summary = state_dict_summary(state)
    expected_summary = {
        key: value[key]
        for key in (
            "tensor_count",
            "element_count",
            "all_finite",
            "canonical_state_sha256",
            "tensor_byte_order",
        )
    }
    if summary != expected_summary:
        raise ReceiptValidationError(f"{label} canonical state summary mismatch")
    if value["tensor_inventory"] != state_dict_inventory(state):
        raise ReceiptValidationError(f"{label} tensor inventory mismatch")
    return state


def _load_json_file(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = loads_json_strict(path.read_bytes())
    except OSError as exc:
        raise ReceiptValidationError(f"Could not read {label}: {exc}") from exc
    return _expect_mapping(value, label)


def _validate_protocol_reference(
    reference: Any,
    *,
    root: Path,
) -> Mapping[str, Any]:
    value = _expect_mapping(reference, "protocol reference")
    required = {
        "path",
        "sha256",
        "final_eval_seeds",
        "samples_per_seed",
        "cohort_specification",
        "cohort_specification_sha256",
        "single_use",
    }
    allowed = required | {"file_sha256", "size_bytes"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise ReceiptValidationError("Protocol reference fields mismatch")
    path = _resolve_bound_path(root, value["path"], "protocol.path")
    if not path.is_file():
        raise ReceiptValidationError("Protocol file is missing")
    if "size_bytes" in value and path.stat().st_size != _expect_int(
        value["size_bytes"], "protocol.size_bytes", minimum=0
    ):
        raise ReceiptValidationError("Protocol file size mismatch")
    if "file_sha256" in value and sha256_file(path) != _expect_sha256(
        value["file_sha256"], "protocol.file_sha256"
    ):
        raise ReceiptValidationError("Protocol file SHA-256 mismatch")
    protocol = _load_json_file(path, "protocol")
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ReceiptValidationError("Unsupported protocol schema")
    logical_digest = protocol_digest(protocol)
    if protocol.get("protocol_sha256") != logical_digest:
        raise ReceiptValidationError("Protocol self-digest mismatch")
    if value["sha256"] != logical_digest:
        raise ReceiptValidationError("Receipt protocol digest cross-link mismatch")
    final = _expect_mapping(protocol.get("final"), "protocol.final")
    _specification, specification_sha256 = _validate_cohort_specification(
        value["cohort_specification"],
        seeds=value["final_eval_seeds"],
        samples_per_seed=value["samples_per_seed"],
    )
    if (
        value["final_eval_seeds"] != final.get("seeds")
        or value["samples_per_seed"] != final.get("samples_per_seed")
        or value["cohort_specification"] != final.get("cohort_specification")
        or value["cohort_specification_sha256"] != specification_sha256
        or final.get("cohort_specification_sha256") != specification_sha256
        or value["single_use"] is not True
        or final.get("single_use") is not True
    ):
        raise ReceiptValidationError("Receipt final cohort does not match protocol")
    development = _expect_mapping(protocol.get("development"), "protocol.development")
    if set(final["seeds"]) & set(development.get("seeds", [])):
        raise ReceiptValidationError("Development and final cohorts overlap")
    git = _expect_mapping(protocol.get("git"), "protocol.git")
    if git.get("dirty") is not False:
        raise ReceiptValidationError("Final-evaluation protocol must be frozen from a clean tree")
    code_bindings = _expect_mapping(protocol.get("code_bindings"), "protocol.code_bindings")
    generator_binding = next(
        (
            bound_record.get("sha256")
            for bound_path, bound_record in code_bindings.items()
            if Path(str(bound_path)).as_posix()
            == "source/benchmark_cognitive_leap_ultra_v51.py"
            and isinstance(bound_record, Mapping)
        ),
        None,
    )
    if value["cohort_specification"]["generator_source_sha256"] != generator_binding:
        raise ReceiptValidationError("Cohort generator source binding mismatch")
    return protocol


def _validate_selection_reference(
    reference: Any,
    *,
    root: Path,
    protocol_sha256: str,
    candidate_sha256: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    value = _expect_mapping(reference, "selection reference")
    required = {
        "path",
        "sha256",
        "name",
        "members",
        "member_weights",
        "baseline_blend_alpha",
        "lineage_manifest",
        "lineage_verification",
    }
    allowed = required | {"size_bytes"}
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise ReceiptValidationError("Selection reference fields mismatch")
    path = _resolve_bound_path(root, value["path"], "selection.path")
    if not path.is_file() or sha256_file(path) != _expect_sha256(
        value["sha256"], "selection.sha256"
    ):
        raise ReceiptValidationError("Selection file content mismatch")
    if "size_bytes" in value and path.stat().st_size != value["size_bytes"]:
        raise ReceiptValidationError("Selection file size mismatch")
    selection = _load_json_file(path, "selection")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("passed") is not True:
        raise ReceiptValidationError("A passing v2 development selection is required")
    selection_payload = dict(selection)
    selection_payload.pop("selection_sha256", None)
    if selection.get("selection_sha256") != sha256_bytes(
        canonical_json_bytes(selection_payload)
    ):
        raise ReceiptValidationError("Selection self-digest mismatch")
    if (
        selection.get("authentication") != "none"
        or selection.get("trusted_timestamp") is not False
        or selection.get("integrity_status") != "content_bound_not_authenticated"
        or selection.get("authority") != NO_AUTHORITY
    ):
        raise ReceiptValidationError("Selection evidence authority contract mismatch")
    if selection.get("protocol_sha256") != protocol_sha256:
        raise ReceiptValidationError("Selection protocol cross-link mismatch")
    selected = _expect_mapping(selection.get("selected"), "selection.selected")
    selected_artifact = _expect_mapping(
        selected.get("artifact"),
        "selection.selected.artifact",
    )
    if selected_artifact.get("sha256") != candidate_sha256:
        raise ReceiptValidationError("Selection candidate cross-link mismatch")
    for field in ("name", "members", "member_weights", "baseline_blend_alpha"):
        if value[field] != selected.get(field):
            raise ReceiptValidationError(f"Selection {field} cross-link mismatch")
    lineage_record = _expect_mapping(value["lineage_manifest"], "lineage_manifest")
    verification_record = _expect_mapping(
        value["lineage_verification"],
        "lineage_verification",
    )
    for record, label, schema in (
        (lineage_record, "lineage_manifest", LINEAGE_SCHEMA),
        (
            verification_record,
            "lineage_verification",
            "supermix-cognitive-leap-lineage-verification-v1",
        ),
    ):
        _expect_exact_keys(record, ("path", "sha256", "size_bytes", "schema"), label)
        if record["schema"] != schema:
            raise ReceiptValidationError(f"Unsupported {label} schema")
        _verify_file_reference(record, root, label)
    if selection.get("lineage_manifest") != lineage_record:
        raise ReceiptValidationError("Selection lineage manifest cross-link mismatch")
    if selection.get("lineage_verification") != verification_record:
        raise ReceiptValidationError("Selection lineage verification cross-link mismatch")
    return selection, lineage_record, verification_record


def _validate_lineage_records(
    *,
    lineage_record: Mapping[str, Any],
    verification_record: Mapping[str, Any],
    root: Path,
    protocol: Mapping[str, Any],
    baseline_reference: Mapping[str, Any],
    candidate_reference: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor] | None,
    candidate_state: Mapping[str, torch.Tensor] | None,
    verify_checkpoints: bool,
) -> None:
    lineage_path = _verify_file_reference(lineage_record, root, "lineage_manifest")
    manifest = _load_json_file(lineage_path, "lineage manifest")
    if (
        manifest.get("schema") != LINEAGE_SCHEMA
        or manifest.get("authentication") != "none"
        or manifest.get("timestamps_trusted") is not False
        or manifest.get("authority") != NO_AUTHORITY
        or manifest.get("protocol_sha256") != protocol.get("protocol_sha256")
        or manifest.get("baseline") != baseline_reference
        or manifest.get("selected_artifact") != candidate_reference
    ):
        raise ReceiptValidationError("Lineage manifest contract mismatch")

    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise ReceiptValidationError("Lineage manifest must include continuation members")
    member_states: dict[str, dict[str, torch.Tensor]] = {}
    for index, member in enumerate(members):
        member_value = _expect_mapping(member, f"lineage.members[{index}]")
        _expect_exact_keys(
            member_value,
            ("name", "config", "artifact", "training_receipt"),
            f"lineage.members[{index}]",
        )
        name = member_value["name"]
        if not isinstance(name, str) or not name or name in member_states:
            raise ReceiptValidationError("Lineage member names must be unique strings")
        state = _validate_checkpoint_reference(
            member_value["artifact"],
            root=root,
            label=f"lineage member {name}",
            verify_state=verify_checkpoints,
        )
        training_reference = _expect_mapping(
            member_value["training_receipt"],
            f"lineage member {name} training receipt",
        )
        _expect_exact_keys(
            training_reference,
            ("path", "sha256", "size_bytes"),
            f"lineage member {name} training receipt",
        )
        training_path = _verify_file_reference(
            training_reference,
            root,
            f"lineage member {name} training receipt",
        )
        training = _load_json_file(training_path, f"lineage member {name} training receipt")
        if (
            training.get("schema") != "supermix-cognitive-leap-training-receipt-v2"
            or training.get("authentication") != "none"
            or training.get("trusted_timestamp") is not False
            or training.get("integrity_status") != "content_bound_not_authenticated"
            or training.get("authority") != NO_AUTHORITY
            or training.get("config") != member_value["config"]
            or training.get("artifact") != member_value["artifact"]
        ):
            raise ReceiptValidationError(f"Lineage member {name} receipt mismatch")
        if state is not None:
            member_states[name] = state

    soup = _expect_mapping(manifest.get("soup"), "lineage.soup")
    blend = _expect_mapping(manifest.get("baseline_blend"), "lineage.baseline_blend")
    if soup.get("algorithm") != "ordered_float_tensor_weighted_mean_v1":
        raise ReceiptValidationError("Unsupported lineage soup algorithm")
    member_names = soup.get("members")
    member_weights = soup.get("weights")
    if (
        member_names != [member["name"] for member in members]
        or not isinstance(member_weights, list)
        or len(member_weights) != len(member_names)
    ):
        raise ReceiptValidationError("Lineage soup members or weights mismatch")
    if blend.get("algorithm") != "ordered_float_tensor_weighted_mean_v1":
        raise ReceiptValidationError("Unsupported lineage blend algorithm")
    alpha = _expect_number(blend.get("soup_weight"), "lineage baseline blend alpha")
    if (
        not 0.0 <= alpha <= 1.0
        or not math.isclose(
            _expect_number(blend.get("baseline_weight"), "lineage baseline weight"),
            1.0 - alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ReceiptValidationError("Lineage blend weights mismatch")

    reconstruction = _expect_mapping(
        manifest.get("reconstruction"),
        "lineage.reconstruction",
    )
    if (
        reconstruction.get("exact_tensor_equality") is not True
        or _expect_number(
            reconstruction.get("max_absolute_error"),
            "lineage.reconstruction.max_absolute_error",
        )
        != 0.0
        or reconstruction.get("selected_canonical_state_sha256")
        != candidate_reference.get("canonical_state_sha256")
    ):
        raise ReceiptValidationError("Lineage reconstruction claim mismatch")
    if verify_checkpoints:
        if baseline_state is None or candidate_state is None:
            raise ReceiptValidationError("Lineage reconstruction states are unavailable")
        if set(member_states) != set(member_names):
            raise ReceiptValidationError("Lineage member state set mismatch")
        soup_state = average_state_dicts(
            [member_states[name] for name in member_names],
            [float(weight) for weight in member_weights],
        )
        reconstructed = average_state_dicts(
            [baseline_state, soup_state],
            [1.0 - alpha, alpha],
        )
        validate_state_reconstruction(
            [baseline_state, soup_state],
            [1.0 - alpha, alpha],
            candidate_state,
        )
        if reconstruction.get("reconstructed_canonical_state_sha256") != state_dict_summary(
            reconstructed
        )["canonical_state_sha256"]:
            raise ReceiptValidationError("Lineage reconstructed state digest mismatch")

    verification_path = _verify_file_reference(
        verification_record,
        root,
        "lineage_verification",
    )
    verification = _load_json_file(verification_path, "lineage verification")
    verification_payload = dict(verification)
    verification_payload.pop("verification_id", None)
    if (
        verification.get("schema")
        != "supermix-cognitive-leap-lineage-verification-v1"
        or verification.get("authentication") != "none"
        or verification.get("trusted_timestamp") is not False
        or verification.get("integrity_status")
        != "content_bound_not_authenticated"
        or verification.get("authority") != NO_AUTHORITY
        or verification.get("valid") is not True
        or verification.get("protocol_sha256") != protocol.get("protocol_sha256")
        or verification.get("lineage_manifest") != lineage_record
        or verification.get("selected_canonical_state_sha256")
        != candidate_reference.get("canonical_state_sha256")
        or verification.get("exact_tensor_reconstruction") is not True
        or verification.get("verification_id")
        != sha256_bytes(canonical_json_bytes(verification_payload))
    ):
        raise ReceiptValidationError("Lineage verification receipt mismatch")


def _validate_source_bindings(receipt: Mapping[str, Any], protocol: Mapping[str, Any], root: Path) -> None:
    if "code_bindings" in receipt and receipt["code_bindings"] != protocol.get("code_bindings"):
        raise ReceiptValidationError("Receipt code bindings differ from protocol")
    if "source_snapshot" in receipt and receipt["source_snapshot"] != protocol.get("source_snapshot"):
        raise ReceiptValidationError("Receipt source snapshot differs from protocol")
    snapshots = protocol.get("source_snapshot", {})
    bindings = protocol.get("code_bindings", {})
    if not isinstance(snapshots, Mapping) or not isinstance(bindings, Mapping):
        raise ReceiptValidationError("Protocol source bindings are malformed")
    if set(snapshots) != set(bindings):
        raise ReceiptValidationError("Protocol source snapshot set is incomplete")
    for source_name, record in snapshots.items():
        record_value = _expect_mapping(record, f"source_snapshot.{source_name}")
        path = _verify_file_reference(record_value, root, f"source_snapshot.{source_name}")
        binding = _expect_mapping(
            bindings.get(source_name),
            f"code_bindings.{source_name}",
        )
        if (
            record_value.get("sha256") != binding.get("sha256")
            or record_value.get("size_bytes") != binding.get("size_bytes")
        ):
            raise ReceiptValidationError(f"Source snapshot binding mismatch: {source_name}")
        if not path.is_file():  # _verify_file_reference already checks; explicit for clarity.
            raise ReceiptValidationError(f"Source snapshot is missing: {source_name}")


def _load_receipt(receipt: Path | Mapping[str, Any]) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(receipt, Path):
        return _load_json_file(receipt, "receipt"), receipt
    return _expect_mapping(receipt, "receipt"), None


def validate_receipt(
    receipt: Path | Mapping[str, Any],
    *,
    root: Path,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    """Validate every bounded-evaluation claim and return reproduced evidence."""

    value, receipt_path = _load_receipt(receipt)
    missing = _TOP_LEVEL_REQUIRED - set(value)
    extra = set(value) - (_TOP_LEVEL_REQUIRED | _TOP_LEVEL_OPTIONAL)
    if missing or extra:
        raise ReceiptValidationError(
            f"Receipt fields mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if value["schema"] != RECEIPT_SCHEMA:
        raise ReceiptValidationError("Unsupported bounded receipt schema")
    if value["gate_outcome"] not in {"pass", "reject"}:
        raise ReceiptValidationError("gate_outcome must be pass or reject")
    authority = _expect_mapping(value["authority"], "authority")
    _expect_exact_keys(authority, AUTHORITY_KEYS, "authority")
    if dict(authority) != NO_AUTHORITY:
        raise ReceiptValidationError("A bounded evaluation receipt grants no authority")
    if value["authentication"] != "none":
        raise ReceiptValidationError("Bounded receipt authentication must be none")
    if value["integrity_status"] != "content_bound_not_authenticated":
        raise ReceiptValidationError("Bounded receipt integrity status mismatch")
    if value["trusted_timestamp"] is not False:
        raise ReceiptValidationError("Bounded receipt timestamps are untrusted")

    artifacts = _expect_mapping(value["artifacts"], "artifacts")
    _expect_exact_keys(artifacts, ("baseline", "candidate"), "artifacts")
    baseline_state = _validate_checkpoint_reference(
        artifacts["baseline"],
        root=root,
        label="artifacts.baseline",
        verify_state=verify_checkpoints,
    )
    candidate_state = _validate_checkpoint_reference(
        artifacts["candidate"],
        root=root,
        label="artifacts.candidate",
        verify_state=verify_checkpoints,
    )
    if baseline_state is not None and candidate_state is not None:
        if tuple(baseline_state) != tuple(candidate_state):
            raise ReceiptValidationError("Baseline and candidate state keys differ")
        for key in baseline_state:
            if (
                baseline_state[key].shape != candidate_state[key].shape
                or baseline_state[key].dtype != candidate_state[key].dtype
            ):
                raise ReceiptValidationError(
                    f"Baseline and candidate tensor metadata differ for {key}"
                )

    protocol = _validate_protocol_reference(value["protocol"], root=root)
    if protocol.get("baseline") != artifacts["baseline"]:
        raise ReceiptValidationError("Receipt baseline differs from frozen protocol")
    _selection, lineage_record, lineage_verification_record = _validate_selection_reference(
        value["selection"],
        root=root,
        protocol_sha256=value["protocol"]["sha256"],
        candidate_sha256=artifacts["candidate"]["sha256"],
    )
    _validate_lineage_records(
        lineage_record=lineage_record,
        verification_record=lineage_verification_record,
        root=root,
        protocol=protocol,
        baseline_reference=artifacts["baseline"],
        candidate_reference=artifacts["candidate"],
        baseline_state=baseline_state,
        candidate_state=candidate_state,
        verify_checkpoints=verify_checkpoints,
    )
    _validate_source_bindings(value, protocol, root)
    if "git_at_protocol_freeze" in value and value["git_at_protocol_freeze"] != protocol.get("git"):
        raise ReceiptValidationError("Protocol Git binding cross-link mismatch")
    if "claim_scope" in value and value["claim_scope"] != protocol.get("claim_scope"):
        raise ReceiptValidationError("Receipt claim scope differs from protocol")
    if "git_at_finalization" in value:
        final_git = _expect_mapping(value["git_at_finalization"], "git_at_finalization")
        protocol_git = _expect_mapping(protocol.get("git"), "protocol.git")
        if final_git.get("dirty") is not False or final_git.get("commit") != protocol_git.get("commit"):
            raise ReceiptValidationError("Finalization must use the clean frozen Git commit")
    if "evaluation_rng" in value:
        evaluation_rng = _expect_mapping(value["evaluation_rng"], "evaluation_rng")
        _expect_exact_keys(
            evaluation_rng,
            ("cpu_state_before_sha256", "cpu_state_after_sha256", "unchanged"),
            "evaluation_rng",
        )
        before = _expect_sha256(
            evaluation_rng["cpu_state_before_sha256"],
            "evaluation_rng.cpu_state_before_sha256",
        )
        if evaluation_rng["cpu_state_after_sha256"] != before or evaluation_rng["unchanged"] is not True:
            raise ReceiptValidationError("Final evaluation changed the CPU RNG state")
    if "final_invocation_sha256" in value:
        _expect_sha256(value["final_invocation_sha256"], "final_invocation_sha256")
    if "single_use_scope" in value and value["single_use_scope"] != "this_local_output_directory_only":
        raise ReceiptValidationError("Unsupported final cohort single-use scope")

    criteria = _validate_criteria(value["criteria"])
    if criteria != protocol.get("criteria"):
        raise ReceiptValidationError("Receipt criteria differ from frozen protocol")
    seeds = value["protocol"]["final_eval_seeds"]
    samples_per_seed = value["protocol"]["samples_per_seed"]
    reproduced = validate_prediction_artifact(
        value["per_example_artifact"],
        root=root,
        seeds=seeds,
        samples_per_seed=samples_per_seed,
        criteria=criteria,
        cohort_specification=value["protocol"]["cohort_specification"],
    )
    if value["per_example_artifact"]["dataset_id"] != value["evidence"].get("dataset_id"):
        raise ReceiptValidationError("Artifact and receipt dataset_id cross-link mismatch")
    tolerance = float(value["per_example_artifact"]["validation_absolute_tolerance"])
    for field in (
        "gate_outcome",
        "checks",
        "summary",
        "seed_rows",
        "operation_family_rows",
        "class_rows",
        "evidence",
    ):
        _assert_reproduced(
            value[field],
            reproduced[field],
            field,
            tolerance=tolerance,
        )
    expected_receipt_id = receipt_id_for(value)
    if value["receipt_id"] != expected_receipt_id:
        raise ReceiptValidationError("Receipt ID content binding mismatch")
    if receipt_path is not None and not receipt_path.is_file():
        raise ReceiptValidationError("Receipt file disappeared during validation")
    result = dict(reproduced)
    result.update(
        {
            "valid": True,
            "receipt_id": expected_receipt_id,
            "receipt_file_sha256": (
                sha256_file(receipt_path)
                if receipt_path is not None
                else sha256_bytes(canonical_json_bytes(value))
            ),
            "protocol_sha256": value["protocol"]["sha256"],
            "baseline_sha256": artifacts["baseline"]["sha256"],
            "candidate_sha256": artifacts["candidate"]["sha256"],
            "lineage_sha256": lineage_record["sha256"],
            "lineage_verification_sha256": lineage_verification_record["sha256"],
            "per_example_sha256": value["per_example_artifact"]["sha256"],
            "per_example_uncompressed_sha256": value["per_example_artifact"][
                "uncompressed_sha256"
            ],
        }
    )
    return result


def try_validate_receipt(
    receipt: Path | Mapping[str, Any],
    *,
    root: Path,
    verify_checkpoints: bool = True,
) -> dict[str, Any]:
    """Non-raising validation wrapper for Store/package admission gates."""

    try:
        return validate_receipt(
            receipt,
            root=root,
            verify_checkpoints=verify_checkpoints,
        )
    except (OSError, ReceiptValidationError) as exc:
        return {"valid": False, "error": str(exc)}


__all__ = [
    "ARTIFACT_KEYS",
    "AUTHORITY_KEYS",
    "CHECK_KEYS",
    "CRITERIA_KEYS",
    "LINEAGE_SCHEMA",
    "NO_AUTHORITY",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PROTOCOL_SCHEMA",
    "RECEIPT_SCHEMA",
    "ROW_KEYS",
    "ReceiptValidationError",
    "average_state_dicts",
    "canonical_json_bytes",
    "dataset_sha256",
    "dataset_id_for",
    "decode_logits_f32le_hex",
    "encode_logits_f32le_hex",
    "exact_mcnemar_two_sided",
    "load_state_dict",
    "loads_json_strict",
    "protocol_digest",
    "receipt_id_for",
    "sha256_bytes",
    "sha256_file",
    "state_dict_summary",
    "state_dict_inventory",
    "tensor_digest_update",
    "validate_prediction_artifact",
    "validate_receipt",
    "validate_state_reconstruction",
    "try_validate_receipt",
]
