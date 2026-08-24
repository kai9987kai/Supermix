"""Independent fail-closed validation for Cognitive Leap v51.2 receipts.

The v51.2 gate is deliberately three-way: a candidate must preserve the
canonical v51 release checkpoint *and* improve on the exact, unpromoted v51.1
candidate.  This module validates a single deterministic JSONL artifact that
contains logits for all three checkpoints and independently recomputes both
paired gates.  A valid receipt is evidence only; it grants no activation,
routing, publication, promotion, or release authority.

``verify_inference=False`` exists only so small synthetic unit fixtures can
exercise the receipt machinery.  Production callers must use the default,
which strict-loads every checkpoint and exactly replays all frozen-cohort
logits with ``ChampionNetCognitiveLeapUltraExpert``.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import struct
import sys
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, BinaryIO

import torch
import torch.nn.functional as F


SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from benchmark_cognitive_leap_ultra_v51 import (  # noqa: E402
    FAMILY_TAG_SCHEMA,
    GENERATOR_SCHEMA,
    make_chained_task_with_metadata,
    operation_family_tags,
)
from model_variants import ChampionNetCognitiveLeapUltraExpert  # noqa: E402


RECEIPT_SCHEMA = "supermix-cognitive-leap-three-way-evaluation-v1"
PREDICTION_ARTIFACT_SCHEMA = (
    "supermix-cognitive-leap-three-way-logits-jsonl-v1"
)
PROFILE_SCHEMA = "supermix-cognitive-leap-evaluation-profile-v1"
PROTOCOL_SCHEMA = "supermix-cognitive-leap-bounded-protocol-v2"
SELECTION_SCHEMA = "supermix-cognitive-leap-development-selection-v2"
LINEAGE_SCHEMA = "supermix-cognitive-leap-lineage-v2"
LINEAGE_VERIFICATION_SCHEMA = (
    "supermix-cognitive-leap-lineage-verification-v1"
)
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
CLAIM_SCOPE = {
    "task": "four-operation chained modular arithmetic modulo 10",
    "general_chat_claim": False,
    "general_reasoning_claim": False,
    "production_default_claim": False,
    "auto_route_claim": False,
    "manual_activation_claim": False,
}

RELEASE_CRITERIA: dict[str, Any] = {
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
DEVELOPMENT_RELEASE_CRITERIA: dict[str, Any] = {
    **RELEASE_CRITERIA,
    "minimum_accuracy_gain": 0.003,
    "minimum_worst_seed_delta": -0.005,
    "minimum_worst_operation_family_delta": 0.0,
    "minimum_nonregressing_classes": 8,
    "minimum_worst_class_delta": -0.02,
}
PRIOR_CANDIDATE_CRITERIA: dict[str, Any] = {
    **RELEASE_CRITERIA,
    "minimum_accuracy_gain": 0.00025,
}

DEV_SEEDS = tuple(range(21_052, 61_052, 1_000))
FINAL_SEEDS = tuple(range(101_052, 121_052, 1_000))
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
SELECTION_ORDER = (
    "both_gates_pass",
    "total_check_count",
    "prior_candidate_accuracy_delta",
    "release_accuracy_delta",
    "combined_nonregression_counts",
    "negative_candidate_loss",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical ASCII JSON, rejecting non-finite values."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ReceiptValidationError(f"Value is not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_canonical_evaluation_profile() -> dict[str, Any]:
    """Build the independently pinned v51.2 three-way policy."""

    return {
        "schema": PROFILE_SCHEMA,
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
            "path": (
                "output/benchmark_v51_cognitive_leap_ultra_latest/"
                "cognitive_leap_ultra_v51_trained.pth"
            ),
            "size_bytes": 9_016_017,
            "sha256": (
                "664b1779452fe1482389413004d8bce3369f6d8ee15ab8c2c891dc5e382ebae4"
            ),
            "canonical_state_sha256": (
                "bed39f133c710e718aab7d7de387b42890ee0767fbbe70e8cc626b2d0d56ede5"
            ),
            "tensor_count": 100,
            "element_count": 2_245_719,
        },
        "prior_candidate": {
            "status": "unpromoted_prior_candidate",
            "path": (
                "output/training_candidates/"
                "cognitive_leap_ultra_v51_1_balanced_blend30_seed151/"
                "cognitive_leap_ultra_v51_1_balanced_blend30.pth"
            ),
            "size_bytes": 9_017_183,
            "sha256": (
                "c627d905951fbfefa8155a9aae064d04fcc574cb8464f08fc716947422de06cb"
            ),
            "canonical_state_sha256": (
                "9850e8b7595795667642294049fc2394771a52997390b961c2624c78e41bf1a0"
            ),
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
            "soup_groups": [["tempered_251", "tempered_351"]],
            "baseline_blend_alphas": [0.20, 0.25, 0.30],
            "selection_order": list(SELECTION_ORDER),
            "release_continuity_criteria": dict(
                DEVELOPMENT_RELEASE_CRITERIA
            ),
            "prior_candidate_superiority_criteria": dict(
                PRIOR_CANDIDATE_CRITERIA
            ),
        },
        "final": {
            "seeds": list(FINAL_SEEDS),
            "samples_per_seed": 2_000,
            "single_use": True,
            "release_continuity_criteria": dict(RELEASE_CRITERIA),
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


_IMMUTABLE_PROFILE = _build_canonical_evaluation_profile()
CANONICAL_EVALUATION_PROFILE_SHA256 = (
    "3a018d1b9cde5d59c0431f0323a46993d71806604753e459200649a024332bbd"
)
PROFILE_HASH_ALLOWLIST = frozenset({CANONICAL_EVALUATION_PROFILE_SHA256})


def canonical_evaluation_profile() -> dict[str, Any]:
    """Return a defensive copy of the one accepted v51.2 profile."""

    return copy.deepcopy(_IMMUTABLE_PROFILE)


class ReceiptValidationError(ValueError):
    """Raised when a three-way receipt is malformed or fails replay."""


_computed_profile_hash = sha256_bytes(canonical_json_bytes(_IMMUTABLE_PROFILE))
if _computed_profile_hash != CANONICAL_EVALUATION_PROFILE_SHA256:
    raise RuntimeError(
        "Pinned v51.2 evaluation profile payload/hash mismatch: "
        f"{_computed_profile_hash}"
    )


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


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ReceiptValidationError(f"{label} must be an object with string keys")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    label: str,
) -> None:
    expected_set = set(expected)
    actual_set = set(value)
    if actual_set != expected_set:
        raise ReceiptValidationError(
            f"{label} fields mismatch; missing={sorted(expected_set - actual_set)}, "
            f"extra={sorted(actual_set - expected_set)}"
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
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReceiptValidationError(f"{label} must be lowercase SHA-256 hex")
    return value


def _expect_sha1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReceiptValidationError(f"{label} must be lowercase SHA-1 hex")
    return value


def _expect_equal(actual: Any, expected: Any, label: str) -> None:
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ReceiptValidationError(f"{label} changed from the immutable contract")


def _resolve_under_root(root: Path, relative_value: Any, label: str) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise ReceiptValidationError(f"{label} must be a nonempty relative path")
    relative = Path(relative_value)
    if relative.is_absolute():
        raise ReceiptValidationError(f"{label} must be repository-relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ReceiptValidationError(f"{label} escapes the validation root") from exc
    return resolved


def _validate_file_record(
    record_value: Any,
    *,
    root: Path,
    label: str,
    content_sha_key: str = "sha256",
) -> tuple[Mapping[str, Any], Path]:
    record = _expect_mapping(record_value, label)
    path = _resolve_under_root(root, record.get("path"), f"{label}.path")
    expected_size = _expect_int(
        record.get("size_bytes"), f"{label}.size_bytes", minimum=0
    )
    expected_hash = _expect_sha256(
        record.get(content_sha_key), f"{label}.{content_sha_key}"
    )
    if not path.is_file():
        raise ReceiptValidationError(f"{label} file is missing: {path}")
    if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
        raise ReceiptValidationError(f"{label} file size/hash mismatch")
    return record, path


def _digest_without(value: Mapping[str, Any], key: str) -> str:
    payload = dict(value)
    payload.pop(key, None)
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
    digest = hashlib.sha256()
    finite = True
    element_count = 0
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        tensor_digest_update(digest, name, tensor)
        element_count += int(tensor.numel())
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            finite = False
    return {
        "tensor_count": len(state),
        "element_count": element_count,
        "all_finite": finite,
        "tensor_byte_order": "little_endian",
        "canonical_state_sha256": digest.hexdigest(),
    }


def _load_checkpoint(path: Path, label: str) -> dict[str, torch.Tensor]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - torch supplies many error types
        raise ReceiptValidationError(f"{label} checkpoint cannot be loaded: {exc}") from exc
    if not isinstance(loaded, Mapping) or not loaded:
        raise ReceiptValidationError(f"{label} checkpoint is not a state dict")
    if "state_dict" in loaded and isinstance(loaded["state_dict"], Mapping):
        loaded = loaded["state_dict"]
    state: dict[str, torch.Tensor] = {}
    for name, tensor in loaded.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ReceiptValidationError(
                f"{label} checkpoint contains a non-tensor state entry"
            )
        state[name] = tensor.detach().cpu()
    if not state:
        raise ReceiptValidationError(f"{label} checkpoint state is empty")
    summary = state_dict_summary(state)
    if not summary["all_finite"]:
        raise ReceiptValidationError(f"{label} checkpoint contains non-finite tensors")
    return state


def _strict_model_from_state(
    state: Mapping[str, torch.Tensor],
    label: str,
) -> ChampionNetCognitiveLeapUltraExpert:
    # Constructors initialize parameters, so isolate their RNG use from callers.
    with torch.random.fork_rng(devices=[]):
        model = ChampionNetCognitiveLeapUltraExpert().cpu()
    try:
        incompatible = model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ReceiptValidationError(
            f"{label} failed strict ChampionNetCognitiveLeapUltraExpert load: {exc}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ReceiptValidationError(f"{label} strict-load keys mismatch")
    model.eval()
    return model


_CHECKPOINT_REQUIRED_KEYS = {
    "path",
    "size_bytes",
    "sha256",
    "tensor_count",
    "element_count",
    "all_finite",
    "tensor_byte_order",
    "canonical_state_sha256",
    "strict_load",
}
_CHECKPOINT_OPTIONAL_KEYS = {"tensor_inventory", "status"}
_REQUIRED_SOURCE_BINDINGS = {
    "source/run_cognitive_leap_v51_2.py",
    "source/benchmark_cognitive_leap_ultra_v51.py",
    "source/model_variants.py",
    "source/run.py",
    "source/device_utils.py",
    "source/cognitive_leap_three_way_receipt.py",
}


def _validate_checkpoint_record(
    record_value: Any,
    *,
    root: Path,
    label: str,
    verify_checkpoint: bool,
) -> tuple[Mapping[str, Any], Path, dict[str, torch.Tensor] | None]:
    record = _expect_mapping(record_value, label)
    actual_keys = set(record)
    if not _CHECKPOINT_REQUIRED_KEYS <= actual_keys or not actual_keys <= (
        _CHECKPOINT_REQUIRED_KEYS | _CHECKPOINT_OPTIONAL_KEYS
    ):
        raise ReceiptValidationError(f"{label} checkpoint fields mismatch")
    if record.get("all_finite") is not True:
        raise ReceiptValidationError(f"{label}.all_finite must be true")
    if record.get("tensor_byte_order") != "little_endian":
        raise ReceiptValidationError(f"{label} tensor byte order changed")
    if record.get("strict_load") is not True:
        raise ReceiptValidationError(f"{label}.strict_load must be true")
    _expect_int(record.get("tensor_count"), f"{label}.tensor_count", minimum=1)
    _expect_int(record.get("element_count"), f"{label}.element_count", minimum=1)
    _expect_sha256(
        record.get("canonical_state_sha256"),
        f"{label}.canonical_state_sha256",
    )
    checked_record, path = _validate_file_record(
        record,
        root=root,
        label=label,
    )
    state: dict[str, torch.Tensor] | None = None
    if verify_checkpoint:
        state = _load_checkpoint(path, label)
        summary = state_dict_summary(state)
        expected_summary = {
            key: checked_record[key]
            for key in (
                "tensor_count",
                "element_count",
                "all_finite",
                "tensor_byte_order",
                "canonical_state_sha256",
            )
        }
        _expect_equal(summary, expected_summary, f"{label} state summary")
        _strict_model_from_state(state, label)
    return checked_record, path, state


def _profile_artifact_projection(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "path",
            "size_bytes",
            "sha256",
            "canonical_state_sha256",
            "tensor_count",
            "element_count",
        )
    }


def _validate_profile(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    profile = _expect_mapping(receipt.get("evaluation_profile"), "evaluation_profile")
    profile_hash = _expect_sha256(
        receipt.get("evaluation_profile_sha256"),
        "evaluation_profile_sha256",
    )
    if profile_hash != sha256_bytes(canonical_json_bytes(profile)):
        raise ReceiptValidationError("Evaluation profile digest mismatch")
    if profile_hash not in PROFILE_HASH_ALLOWLIST:
        raise ReceiptValidationError("Evaluation profile hash is not allowlisted")
    _expect_equal(profile, _IMMUTABLE_PROFILE, "evaluation_profile")
    return profile


def _validate_source_bindings(
    protocol: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    bindings = _expect_mapping(protocol.get("code_bindings"), "protocol.code_bindings")
    snapshots = _expect_mapping(protocol.get("source_snapshot"), "protocol.source_snapshot")
    if not bindings or set(bindings) != set(snapshots):
        raise ReceiptValidationError(
            "Protocol code bindings and source snapshots must have identical keys"
        )
    if not _REQUIRED_SOURCE_BINDINGS <= set(bindings):
        raise ReceiptValidationError(
            "Protocol omits required evaluator source bindings: "
            f"{sorted(_REQUIRED_SOURCE_BINDINGS - set(bindings))}"
        )
    for relative_name in sorted(bindings):
        if Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
            raise ReceiptValidationError("Protocol code binding path is unsafe")
        binding = _expect_mapping(
            bindings[relative_name], f"protocol.code_bindings[{relative_name}]"
        )
        snapshot = _expect_mapping(
            snapshots[relative_name], f"protocol.source_snapshot[{relative_name}]"
        )
        binding_sha = _expect_sha256(
            binding.get("sha256"), f"code binding {relative_name}.sha256"
        )
        binding_size = _expect_int(
            binding.get("size_bytes"),
            f"code binding {relative_name}.size_bytes",
            minimum=1,
        )
        worktree_blob = _expect_sha1(
            binding.get("worktree_git_blob_sha1"),
            f"code binding {relative_name}.worktree_git_blob_sha1",
        )
        head_blob = _expect_sha1(
            binding.get("head_git_blob_sha1"),
            f"code binding {relative_name}.head_git_blob_sha1",
        )
        if worktree_blob != head_blob:
            raise ReceiptValidationError(
                f"Clean protocol source blob mismatch for {relative_name}"
            )
        if not isinstance(binding.get("symbols"), list) or not binding["symbols"]:
            raise ReceiptValidationError(
                f"Code binding symbols are missing for {relative_name}"
            )
        snapshot_record, _path = _validate_file_record(
            snapshot,
            root=root,
            label=f"source snapshot {relative_name}",
        )
        if (
            snapshot_record.get("sha256") != binding_sha
            or snapshot_record.get("size_bytes") != binding_size
        ):
            raise ReceiptValidationError(
                f"Source snapshot does not match binding for {relative_name}"
            )
        executing_source = SOURCE_DIR.parent / relative_name
        if (
            not executing_source.is_file()
            or executing_source.stat().st_size != binding_size
            or sha256_file(executing_source) != binding_sha
        ):
            raise ReceiptValidationError(
                f"Executing evaluator source differs from binding for {relative_name}"
            )


def _environment_compatible(
    frozen_value: Any,
    observed_value: Any,
    label: str,
) -> None:
    frozen = _expect_mapping(frozen_value, f"{label}.frozen")
    observed = _expect_mapping(observed_value, f"{label}.observed")
    for environment, environment_label in (
        (frozen, "frozen"),
        (observed, "observed"),
    ):
        if (
            environment.get("authentication") != "none"
            or environment.get("timestamps_trusted") is not False
            or environment.get("host_identity_trusted") is not False
        ):
            raise ReceiptValidationError(
                f"{label} {environment_label} trust contract changed"
            )
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
        if frozen.get(key) != observed.get(key):
            raise ReceiptValidationError(f"{label} changed at {key}")
    frozen_torch = dict(_expect_mapping(frozen.get("torch"), f"{label}.frozen.torch"))
    observed_torch = dict(
        _expect_mapping(observed.get("torch"), f"{label}.observed.torch")
    )
    frozen_torch.pop("initial_seed", None)
    observed_torch.pop("initial_seed", None)
    if frozen_torch != observed_torch:
        raise ReceiptValidationError(f"{label} torch configuration changed")
    if (
        frozen_torch.get("device") != "cpu"
        or frozen_torch.get("deterministic_algorithms") is not True
        or frozen_torch.get("deterministic_warn_only") is not False
        or frozen_torch.get("default_dtype") != "torch.float32"
        or frozen_torch.get("float32_matmul_precision") != "highest"
    ):
        raise ReceiptValidationError(
            f"{label} is not the frozen deterministic CPU evaluator environment"
        )
    platform_binding = _expect_mapping(
        frozen.get("platform"), f"{label}.frozen.platform"
    )
    if platform_binding.get("byteorder") != "little":
        raise ReceiptValidationError(f"{label} platform byte order changed")


def _validate_protocol(
    record_value: Any,
    *,
    root: Path,
    profile: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    record = _expect_mapping(record_value, "protocol record")
    _expect_exact_keys(
        record,
        {"path", "file_sha256", "size_bytes", "content_sha256"},
        "protocol record",
    )
    path = _resolve_under_root(root, record.get("path"), "protocol.path")
    if not path.is_file():
        raise ReceiptValidationError("Protocol file is missing")
    if path.stat().st_size != _expect_int(
        record.get("size_bytes"), "protocol.size_bytes", minimum=1
    ) or sha256_file(path) != _expect_sha256(
        record.get("file_sha256"), "protocol.file_sha256"
    ):
        raise ReceiptValidationError("Protocol file size/hash mismatch")
    protocol = _expect_mapping(
        loads_json_strict(path.read_bytes()), "protocol payload"
    )
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ReceiptValidationError("Unsupported protocol schema")
    content_sha = _expect_sha256(
        record.get("content_sha256"), "protocol.content_sha256"
    )
    if (
        protocol.get("protocol_sha256") != content_sha
        or _digest_without(protocol, "protocol_sha256") != content_sha
    ):
        raise ReceiptValidationError("Protocol content digest mismatch")
    if (
        protocol.get("authentication") != "none"
        or protocol.get("integrity_status")
        != "content_bound_not_authenticated"
        or protocol.get("trusted_timestamp") is not False
        or protocol.get("authority") != AUTHORITY
        or protocol.get("claim_scope") != CLAIM_SCOPE
    ):
        raise ReceiptValidationError("Protocol trust/authority contract changed")
    if (
        protocol.get("execution_mode") != "clean_final_eligible"
        or protocol.get("finalization_allowed") is not True
    ):
        raise ReceiptValidationError("Three-way final evidence requires clean mode")
    if (
        protocol.get("evaluation_profile_sha256")
        != sha256_bytes(canonical_json_bytes(profile))
        or protocol.get("evaluation_profile") != profile
    ):
        raise ReceiptValidationError("Protocol evaluation profile mismatch")
    if protocol.get("task_schemas") != profile["task_schemas"]:
        raise ReceiptValidationError("Protocol task schemas mismatch")
    _expect_equal(protocol.get("training"), profile["training"], "protocol.training")
    development = _expect_mapping(protocol.get("development"), "protocol.development")
    final = _expect_mapping(protocol.get("final"), "protocol.final")
    for key in ("seeds", "samples_per_seed", "soup_groups", "baseline_blend_alphas"):
        _expect_equal(
            development.get(key), profile["development"][key], f"protocol.development.{key}"
        )
    _expect_equal(
        development.get("selection_order"),
        profile["development"]["selection_order"],
        "protocol.development.selection_order",
    )
    _expect_equal(
        development.get("criteria"),
        profile["development"]["release_continuity_criteria"],
        "protocol.development.criteria",
    )
    _expect_equal(
        development.get("prior_candidate_criteria"),
        profile["development"]["prior_candidate_superiority_criteria"],
        "protocol.development.prior_candidate_criteria",
    )
    _expect_equal(final.get("seeds"), profile["final"]["seeds"], "protocol.final.seeds")
    _expect_equal(
        final.get("samples_per_seed"),
        profile["final"]["samples_per_seed"],
        "protocol.final.samples_per_seed",
    )
    if final.get("single_use") is not True:
        raise ReceiptValidationError("Protocol final cohort is not single-use")
    _expect_equal(
        protocol.get("criteria"),
        profile["final"]["release_continuity_criteria"],
        "protocol.criteria",
    )
    _expect_equal(
        protocol.get("prior_candidate_criteria"),
        profile["final"]["prior_candidate_superiority_criteria"],
        "protocol.prior_candidate_criteria",
    )
    baseline = _expect_mapping(protocol.get("baseline"), "protocol.baseline")
    prior = _expect_mapping(protocol.get("prior_candidate"), "protocol.prior_candidate")
    _expect_equal(
        _profile_artifact_projection(baseline),
        profile["release_baseline"],
        "protocol release baseline",
    )
    prior_projection = _profile_artifact_projection(prior)
    prior_projection["status"] = prior.get("status")
    _expect_equal(
        prior_projection,
        profile["prior_candidate"],
        "protocol prior candidate",
    )
    git = _expect_mapping(protocol.get("git"), "protocol.git")
    if git.get("dirty") is not False:
        raise ReceiptValidationError("Protocol Git binding is dirty")
    _expect_sha1(git.get("commit"), "protocol.git.commit")
    _validate_source_bindings(protocol, root=root)
    return record, protocol, path


_CHECK_KEYS = {
    "accuracy_gain",
    "paired_significance",
    "mean_loss_nonregression",
    "seed_nonregression",
    "operation_family_nonregression",
    "class_bounded_nonregression",
}
_COMPARISON_EVIDENCE_KEYS = {
    "cohort_schema",
    "generator_schema",
    "family_tag_schema",
    "cohort_role",
    "dataset_id",
    "dataset_specification_sha256",
    "dataset_sha256",
    "baseline_prediction_sha256",
    "candidate_prediction_sha256",
    "baseline_logits_sha256",
    "candidate_logits_sha256",
    "baseline_per_example_sha256",
    "candidate_per_example_sha256",
    "paired_outcome_sha256",
}


def _validate_compact_comparison(value: Any, label: str) -> Mapping[str, Any]:
    comparison = _expect_mapping(value, label)
    _expect_exact_keys(
        comparison,
        {"passed", "checks", "summary", "evidence"},
        label,
    )
    checks = _expect_mapping(comparison.get("checks"), f"{label}.checks")
    _expect_exact_keys(checks, _CHECK_KEYS, f"{label}.checks")
    if any(not isinstance(check, bool) for check in checks.values()):
        raise ReceiptValidationError(f"{label}.checks must contain booleans")
    if comparison.get("passed") is not all(checks.values()):
        raise ReceiptValidationError(f"{label}.passed does not equal all checks")
    summary = _expect_mapping(comparison.get("summary"), f"{label}.summary")
    for key in (
        "accuracy_delta",
        "mean_candidate_loss",
        "nonregressing_seed_count",
        "nonregressing_family_count",
        "nonregressing_class_count",
    ):
        _expect_number(summary.get(key), f"{label}.summary.{key}")
    evidence = _expect_mapping(comparison.get("evidence"), f"{label}.evidence")
    _expect_exact_keys(evidence, _COMPARISON_EVIDENCE_KEYS, f"{label}.evidence")
    if (
        evidence.get("cohort_schema") != COHORT_SCHEMA
        or evidence.get("generator_schema") != GENERATOR_SCHEMA
        or evidence.get("family_tag_schema") != FAMILY_TAG_SCHEMA
        or evidence.get("cohort_role") != "development"
    ):
        raise ReceiptValidationError(f"{label} development evidence schema mismatch")
    for key in _COMPARISON_EVIDENCE_KEYS - {
        "cohort_schema",
        "generator_schema",
        "family_tag_schema",
        "cohort_role",
    }:
        _expect_sha256(evidence.get(key), f"{label}.evidence.{key}")
    return comparison


def _dual_selection_score(
    release_comparison: Mapping[str, Any],
    prior_comparison: Mapping[str, Any],
) -> list[Any]:
    release_summary = release_comparison["summary"]
    prior_summary = prior_comparison["summary"]
    return [
        int(bool(release_comparison["passed"] and prior_comparison["passed"])),
        sum(bool(value) for value in release_comparison["checks"].values())
        + sum(bool(value) for value in prior_comparison["checks"].values()),
        float(prior_summary["accuracy_delta"]),
        float(release_summary["accuracy_delta"]),
        sum(
            int(summary[key])
            for summary in (release_summary, prior_summary)
            for key in (
                "nonregressing_seed_count",
                "nonregressing_family_count",
                "nonregressing_class_count",
            )
        ),
        -float(release_summary["mean_candidate_loss"]),
    ]


def _validate_selection_candidate(
    value: Any,
    *,
    expected_group: Sequence[str],
    expected_alpha: float,
    label: str,
) -> Mapping[str, Any]:
    row = _expect_mapping(value, label)
    _expect_exact_keys(
        row,
        {
            "name",
            "members",
            "member_weights",
            "baseline_blend_alpha",
            "passed",
            "comparisons",
            "selection_score",
        },
        label,
    )
    expected_name = f"{'-'.join(expected_group)}__alpha_{expected_alpha:.2f}"
    if row.get("name") != expected_name or row.get("members") != list(expected_group):
        raise ReceiptValidationError(f"{label} identity does not match search matrix")
    expected_weights = [1.0 / len(expected_group)] * len(expected_group)
    _expect_equal(row.get("member_weights"), expected_weights, f"{label}.member_weights")
    if _expect_number(
        row.get("baseline_blend_alpha"), f"{label}.baseline_blend_alpha"
    ) != float(expected_alpha):
        raise ReceiptValidationError(f"{label} baseline blend alpha changed")
    comparisons = _expect_mapping(row.get("comparisons"), f"{label}.comparisons")
    _expect_exact_keys(
        comparisons,
        {"release_continuity", "prior_candidate_superiority"},
        f"{label}.comparisons",
    )
    release = _validate_compact_comparison(
        comparisons["release_continuity"], f"{label}.release_continuity"
    )
    prior = _validate_compact_comparison(
        comparisons["prior_candidate_superiority"],
        f"{label}.prior_candidate_superiority",
    )
    expected_passed = bool(release["passed"] and prior["passed"])
    if row.get("passed") is not expected_passed:
        raise ReceiptValidationError(f"{label}.passed is not the dual logical AND")
    expected_score = _dual_selection_score(release, prior)
    _expect_equal(row.get("selection_score"), expected_score, f"{label}.selection_score")
    return row


def _average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float],
) -> dict[str, torch.Tensor]:
    if not states or len(states) != len(weights):
        raise ReceiptValidationError("Lineage weighted soup is empty or mis-sized")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise ReceiptValidationError("Lineage state key/order mismatch")
    raw_weights = tuple(float(value) for value in weights)
    if any(not math.isfinite(value) or value < 0 for value in raw_weights):
        raise ReceiptValidationError("Lineage weights are invalid")
    total = sum(raw_weights)
    if total <= 0:
        raise ReceiptValidationError("Lineage weights have no positive mass")
    normalized = tuple(value / total for value in raw_weights)
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [state[key].detach().cpu() for state in states]
        first = tensors[0]
        if any(
            tensor.shape != first.shape or tensor.dtype != first.dtype
            for tensor in tensors[1:]
        ):
            raise ReceiptValidationError(f"Lineage tensor mismatch for {key}")
        if first.is_floating_point() or first.is_complex():
            mixed = torch.zeros_like(first)
            for weight, tensor in zip(normalized, tensors, strict=True):
                mixed.add_(tensor, alpha=weight)
            if not bool(torch.isfinite(mixed).all()):
                raise ReceiptValidationError(f"Lineage produced non-finite tensor {key}")
            result[key] = mixed
        else:
            if any(not torch.equal(first, tensor) for tensor in tensors[1:]):
                raise ReceiptValidationError(f"Lineage non-floating tensor differs: {key}")
            result[key] = first.clone()
    return result


def _validate_lineage(
    selection: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    release_state: Mapping[str, torch.Tensor] | None,
    candidate_state: Mapping[str, torch.Tensor] | None,
    root: Path,
    verify_checkpoint: bool,
) -> None:
    manifest_record, manifest_path = _validate_file_record(
        selection.get("lineage_manifest"),
        root=root,
        label="selection.lineage_manifest",
    )
    verification_record, verification_path = _validate_file_record(
        selection.get("lineage_verification"),
        root=root,
        label="selection.lineage_verification",
    )
    if manifest_record.get("schema") != LINEAGE_SCHEMA:
        raise ReceiptValidationError("Lineage manifest record schema mismatch")
    if verification_record.get("schema") != LINEAGE_VERIFICATION_SCHEMA:
        raise ReceiptValidationError("Lineage verification record schema mismatch")
    manifest = _expect_mapping(
        loads_json_strict(manifest_path.read_bytes()), "lineage manifest"
    )
    verification = _expect_mapping(
        loads_json_strict(verification_path.read_bytes()), "lineage verification"
    )
    if (
        manifest.get("schema") != LINEAGE_SCHEMA
        or manifest.get("authentication") != "none"
        or manifest.get("timestamps_trusted") is not False
        or manifest.get("integrity_status")
        != "content_bound_not_authenticated"
        or manifest.get("authority") != AUTHORITY
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("evaluation_profile_sha256")
        != protocol["evaluation_profile_sha256"]
        or manifest.get("baseline") != protocol["baseline"]
        or manifest.get("selected_artifact") != candidate_artifact
    ):
        raise ReceiptValidationError("Lineage manifest contract mismatch")
    if (
        verification.get("schema") != LINEAGE_VERIFICATION_SCHEMA
        or verification.get("authentication") != "none"
        or verification.get("integrity_status")
        != "content_bound_not_authenticated"
        or verification.get("trusted_timestamp") is not False
        or verification.get("authority") != AUTHORITY
        or verification.get("valid") is not True
        or verification.get("protocol_sha256") != protocol["protocol_sha256"]
        or verification.get("evaluation_profile_sha256")
        != protocol["evaluation_profile_sha256"]
        or verification.get("lineage_manifest") != manifest_record
        or verification.get("selected_canonical_state_sha256")
        != candidate_artifact["canonical_state_sha256"]
        or verification.get("exact_tensor_reconstruction") is not True
    ):
        raise ReceiptValidationError("Lineage verification contract mismatch")

    selected = _expect_mapping(selection.get("selected"), "selection.selected")
    selected_names = list(selected.get("members", []))
    selected_weights = list(selected.get("member_weights", []))
    selected_without_artifact = {
        key: value for key, value in selected.items() if key != "artifact"
    }
    expected_recipe = {
        "name": selected.get("name"),
        "members": selected_names,
        "member_weights": selected_weights,
        "baseline_blend_alpha": selected.get("baseline_blend_alpha"),
    }
    if (
        manifest.get("selected_recipe") != expected_recipe
        or manifest.get("selected_development_evidence_sha256")
        != sha256_bytes(canonical_json_bytes(selected_without_artifact))
    ):
        raise ReceiptValidationError("Lineage selected development recipe mismatch")
    members = list(manifest.get("members", []))
    if [member.get("name") for member in members] != selected_names:
        raise ReceiptValidationError("Lineage member order does not match selection")
    protocol_configs = {
        config["name"]: config for config in protocol["training"]["members"]
    }
    member_states: list[dict[str, torch.Tensor]] = []
    normalized_members: list[Mapping[str, Any]] = []
    for index, member_value in enumerate(members):
        member = _expect_mapping(member_value, f"lineage.members[{index}]")
        name = member.get("name")
        if name not in protocol_configs or member.get("config") != protocol_configs[name]:
            raise ReceiptValidationError("Lineage member config changed")
        artifact = _expect_mapping(member.get("artifact"), f"lineage member {name}.artifact")
        training_receipt_record, training_path = _validate_file_record(
            member.get("training_receipt"),
            root=root,
            label=f"lineage member {name}.training_receipt",
        )
        if training_receipt_record.get("path") == manifest_record.get("path"):
            raise ReceiptValidationError("Lineage training receipt aliases manifest")
        training_receipt = _expect_mapping(
            loads_json_strict(training_path.read_bytes()),
            f"lineage member {name}.training receipt payload",
        )
        if (
            training_receipt.get("schema")
            != "supermix-cognitive-leap-training-receipt-v2"
            or training_receipt.get("authentication") != "none"
            or training_receipt.get("trusted_timestamp") is not False
            or training_receipt.get("integrity_status")
            != "content_bound_not_authenticated"
            or training_receipt.get("authority") != AUTHORITY
            or training_receipt.get("protocol_sha256")
            != protocol["protocol_sha256"]
            or training_receipt.get("evaluation_profile_sha256")
            != protocol["evaluation_profile_sha256"]
            or training_receipt.get("parent_baseline") != protocol["baseline"]
            or training_receipt.get("config") != protocol_configs[name]
            or training_receipt.get("artifact") != artifact
            or training_receipt.get("receipt_id")
            != _digest_without(training_receipt, "receipt_id")
        ):
            raise ReceiptValidationError(
                f"Lineage member {name} training receipt contract mismatch"
            )
        embedded_member_receipts = _expect_mapping(
            selection.get("member_receipts"), "selection.member_receipts"
        )
        if embedded_member_receipts.get(name) != training_receipt:
            raise ReceiptValidationError(
                f"Selection/lineage training receipt mismatch for {name}"
            )
        _record, _path, state = _validate_checkpoint_record(
            artifact,
            root=root,
            label=f"lineage member {name}.artifact",
            verify_checkpoint=verify_checkpoint,
        )
        normalized_members.append(member)
        if verify_checkpoint:
            assert state is not None
            member_states.append(state)

    soup = _expect_mapping(manifest.get("soup"), "lineage.soup")
    blend = _expect_mapping(manifest.get("baseline_blend"), "lineage.baseline_blend")
    if (
        soup.get("algorithm") != "ordered_float_tensor_weighted_mean_v1"
        or soup.get("members") != selected_names
        or soup.get("weights") != selected_weights
        or blend.get("algorithm") != "ordered_float_tensor_weighted_mean_v1"
    ):
        raise ReceiptValidationError("Lineage soup recipe mismatch")
    alpha = _expect_number(
        selected.get("baseline_blend_alpha"),
        "selection.selected.baseline_blend_alpha",
    )
    if (
        _expect_number(blend.get("soup_weight"), "lineage.soup_weight") != alpha
        or not math.isclose(
            _expect_number(blend.get("baseline_weight"), "lineage.baseline_weight"),
            1.0 - alpha,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ReceiptValidationError("Lineage baseline blend mismatch")
    baseline_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "checkpoint",
                "artifact_sha256": protocol["baseline"]["sha256"],
            }
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
        for member in normalized_members
    }
    soup_node_id = sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "ordered_weighted_soup",
                "parents": [member_node_ids[name] for name in selected_names],
                "weights": selected_weights,
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
                "artifact_sha256": candidate_artifact["sha256"],
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
            for member in normalized_members
        ],
        {
            "node_id": soup_node_id,
            "kind": "ordered_weighted_soup",
            "parents": [member_node_ids[name] for name in selected_names],
            "weights": selected_weights,
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
            "artifact": candidate_artifact,
        },
    ]
    if (
        manifest.get("root_node_id") != baseline_node_id
        or manifest.get("selected_node_id") != candidate_node_id
        or manifest.get("nodes") != expected_nodes
    ):
        raise ReceiptValidationError("Lineage graph identity mismatch")
    reconstruction = _expect_mapping(
        manifest.get("reconstruction"), "lineage.reconstruction"
    )
    if (
        reconstruction.get("exact_tensor_equality") is not True
        or _expect_number(
            reconstruction.get("max_absolute_error"),
            "lineage.reconstruction.max_absolute_error",
        )
        != 0.0
        or reconstruction.get("selected_canonical_state_sha256")
        != candidate_artifact["canonical_state_sha256"]
    ):
        raise ReceiptValidationError("Lineage reconstruction claim mismatch")
    if (
        verification.get("root_node_id") != baseline_node_id
        or verification.get("selected_node_id") != candidate_node_id
    ):
        raise ReceiptValidationError("Lineage verification node IDs mismatch")
    if verify_checkpoint:
        assert release_state is not None and candidate_state is not None
        soup_state = _average_states(member_states, selected_weights)
        reconstructed = _average_states(
            [release_state, soup_state], [1.0 - alpha, alpha]
        )
        if state_dict_summary(reconstructed) != state_dict_summary(candidate_state):
            raise ReceiptValidationError("Candidate state summary is not reconstructed")
        if set(reconstructed) != set(candidate_state) or any(
            not torch.equal(reconstructed[name], candidate_state[name])
            for name in reconstructed
        ):
            raise ReceiptValidationError("Candidate is not exact lineage reconstruction")


def _validate_selection(
    record_value: Any,
    *,
    root: Path,
    protocol: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidate_artifact: Mapping[str, Any],
    release_state: Mapping[str, torch.Tensor] | None,
    candidate_state: Mapping[str, torch.Tensor] | None,
    verify_checkpoint: bool,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Path]:
    record = _expect_mapping(record_value, "selection record")
    _expect_exact_keys(
        record,
        {"path", "file_sha256", "size_bytes", "content_sha256"},
        "selection record",
    )
    path = _resolve_under_root(root, record.get("path"), "selection.path")
    if not path.is_file():
        raise ReceiptValidationError("Selection file is missing")
    if path.stat().st_size != _expect_int(
        record.get("size_bytes"), "selection.size_bytes", minimum=1
    ) or sha256_file(path) != _expect_sha256(
        record.get("file_sha256"), "selection.file_sha256"
    ):
        raise ReceiptValidationError("Selection file size/hash mismatch")
    selection = _expect_mapping(
        loads_json_strict(path.read_bytes()), "selection payload"
    )
    content_sha = _expect_sha256(
        record.get("content_sha256"), "selection.content_sha256"
    )
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("selection_sha256") != content_sha
        or _digest_without(selection, "selection_sha256") != content_sha
    ):
        raise ReceiptValidationError("Selection schema/content digest mismatch")
    if (
        selection.get("authentication") != "none"
        or selection.get("integrity_status")
        != "content_bound_not_authenticated"
        or selection.get("trusted_timestamp") is not False
        or selection.get("authority") != AUTHORITY
        or selection.get("protocol_sha256") != protocol["protocol_sha256"]
        or selection.get("decision")
        != "selected_and_frozen_for_single_final"
        or selection.get("passed") is not True
    ):
        raise ReceiptValidationError("Selection trust/decision contract mismatch")
    groups = profile["development"]["soup_groups"]
    alphas = profile["development"]["baseline_blend_alphas"]
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(groups) * len(alphas):
        raise ReceiptValidationError("Selection candidate matrix is incomplete")
    validated: list[Mapping[str, Any]] = []
    offset = 0
    for group in groups:
        for alpha in alphas:
            validated.append(
                _validate_selection_candidate(
                    candidates[offset],
                    expected_group=group,
                    expected_alpha=float(alpha),
                    label=f"selection.candidates[{offset}]",
                )
            )
            offset += 1
    development_dataset_sha = _expect_sha256(
        selection.get("development_dataset_sha256"),
        "selection.development_dataset_sha256",
    )
    release_baseline_identity: tuple[str, str, str] | None = None
    prior_baseline_identity: tuple[str, str, str] | None = None
    shared_dataset_identity: tuple[str, str, str] | None = None
    for index, row in enumerate(validated):
        release_evidence = row["comparisons"]["release_continuity"]["evidence"]
        prior_evidence = row["comparisons"]["prior_candidate_superiority"][
            "evidence"
        ]
        release_dataset = (
            release_evidence["dataset_id"],
            release_evidence["dataset_specification_sha256"],
            release_evidence["dataset_sha256"],
        )
        prior_dataset = (
            prior_evidence["dataset_id"],
            prior_evidence["dataset_specification_sha256"],
            prior_evidence["dataset_sha256"],
        )
        if (
            release_dataset != prior_dataset
            or release_evidence["dataset_sha256"] != development_dataset_sha
        ):
            raise ReceiptValidationError(
                f"Selection candidate {index} dual comparisons use different cohorts"
            )
        if shared_dataset_identity is None:
            shared_dataset_identity = release_dataset
        elif release_dataset != shared_dataset_identity:
            raise ReceiptValidationError("Selection candidates use different cohorts")
        release_identity = (
            release_evidence["baseline_prediction_sha256"],
            release_evidence["baseline_logits_sha256"],
            release_evidence["baseline_per_example_sha256"],
        )
        prior_identity = (
            prior_evidence["baseline_prediction_sha256"],
            prior_evidence["baseline_logits_sha256"],
            prior_evidence["baseline_per_example_sha256"],
        )
        if release_baseline_identity is None:
            release_baseline_identity = release_identity
            prior_baseline_identity = prior_identity
        elif (
            release_identity != release_baseline_identity
            or prior_identity != prior_baseline_identity
        ):
            raise ReceiptValidationError(
                "Selection baseline identities changed across candidates"
            )
        for suffix in (
            "prediction_sha256",
            "logits_sha256",
            "per_example_sha256",
        ):
            if release_evidence[f"candidate_{suffix}"] != prior_evidence[
                f"candidate_{suffix}"
            ]:
                raise ReceiptValidationError(
                    f"Selection candidate {index} dual evidence candidate mismatch"
                )
    winner = max(validated, key=lambda row: tuple(row["selection_score"]))
    selected = _expect_mapping(selection.get("selected"), "selection.selected")
    if selected.get("artifact") != candidate_artifact:
        raise ReceiptValidationError("Selection candidate artifact crosslink mismatch")
    selected_without_artifact = {
        key: value for key, value in selected.items() if key != "artifact"
    }
    if winner["passed"] is not True or selected_without_artifact != winner:
        raise ReceiptValidationError("Selection winner is not deterministic/passing")
    if set(selection.get("member_receipts", {})) != {
        config["name"] for config in profile["training"]["members"]
    }:
        raise ReceiptValidationError("Selection member receipt set is incomplete")
    _environment_compatible(
        protocol.get("environment_at_freeze"),
        selection.get("environment"),
        "selection environment",
    )
    _validate_lineage(
        selection,
        protocol=protocol,
        candidate_artifact=candidate_artifact,
        release_state=release_state,
        candidate_state=candidate_state,
        root=root,
        verify_checkpoint=verify_checkpoint,
    )
    return record, selection, path


_ROW_KEYS = {
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
    "release_baseline_logits_f32le_hex",
    "prior_candidate_logits_f32le_hex",
    "candidate_logits_f32le_hex",
    "release_baseline_prediction",
    "prior_candidate_prediction",
    "candidate_prediction",
    "release_baseline_correct",
    "prior_candidate_correct",
    "candidate_correct",
}
_ARTIFACT_KEYS = {
    "schema",
    "path",
    "sha256",
    "size_bytes",
    "uncompressed_sha256",
    "row_count",
    "evaluation_profile_sha256",
    "cohort_schema",
    "generator_schema",
    "family_tag_schema",
    "cohort_role",
    "dataset_id",
    "dataset_specification_sha256",
    "dataset_sha256",
    "format",
    "logits_encoding",
    "class_count",
    "class_order",
    "logit_shape",
    "argmax_tie_rule",
    "loss_formula",
    "gzip_mtime",
}
_MODEL_NAMES = ("release_baseline", "prior_candidate", "candidate")


def _decode_logits(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, str) or len(value) != 80:
        raise ReceiptValidationError(f"{label} must encode exactly ten float32 logits")
    try:
        raw = bytes.fromhex(value)
        logits = struct.unpack("<10f", raw)
    except (ValueError, struct.error) as exc:
        raise ReceiptValidationError(f"{label} is invalid float32 hex") from exc
    if not all(math.isfinite(logit) for logit in logits):
        raise ReceiptValidationError(f"{label} contains non-finite logits")
    return logits


def _argmax_lowest(logits: Sequence[float]) -> int:
    return max(range(len(logits)), key=lambda index: logits[index])


def _canonical_cohort_specification(
    protocol: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> Mapping[str, Any]:
    final = _expect_mapping(protocol.get("final"), "protocol.final")
    specification = _expect_mapping(
        final.get("cohort_specification"), "protocol.final.cohort_specification"
    )
    _expect_exact_keys(
        specification,
        {
            "schema",
            "generator_schema",
            "family_tag_schema",
            "cohort_role",
            "seeds",
            "samples_per_seed",
            "generator_source_sha256",
        },
        "protocol.final.cohort_specification",
    )
    if (
        specification.get("schema") != COHORT_SCHEMA
        or specification.get("generator_schema") != GENERATOR_SCHEMA
        or specification.get("family_tag_schema") != FAMILY_TAG_SCHEMA
        or specification.get("cohort_role") != "final"
        or specification.get("seeds") != profile["final"]["seeds"]
        or specification.get("samples_per_seed")
        != profile["final"]["samples_per_seed"]
    ):
        raise ReceiptValidationError("Final cohort specification changed")
    generator_binding = _expect_mapping(
        protocol["code_bindings"].get(
            "source/benchmark_cognitive_leap_ultra_v51.py"
        ),
        "canonical generator code binding",
    )
    if specification.get("generator_source_sha256") != generator_binding.get("sha256"):
        raise ReceiptValidationError("Cohort generator source binding mismatch")
    specification_hash = sha256_bytes(canonical_json_bytes(specification))
    if final.get("cohort_specification_sha256") != specification_hash:
        raise ReceiptValidationError("Final cohort specification digest mismatch")
    return specification


def _predict_exact(
    model: ChampionNetCognitiveLeapUltraExpert,
    x: torch.Tensor,
    label: str,
) -> torch.Tensor:
    with torch.inference_mode():
        logits = model(x.cpu(), reasoning_cycles=3).squeeze(1)
    logits = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if logits.shape != (int(x.shape[0]), 10):
        raise ReceiptValidationError(
            f"{label} replay produced unexpected logits shape {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all()):
        raise ReceiptValidationError(f"{label} replay produced non-finite logits")
    return logits


def _validate_prediction_artifact(
    record_value: Any,
    *,
    root: Path,
    protocol: Mapping[str, Any],
    profile: Mapping[str, Any],
    models: Mapping[str, ChampionNetCognitiveLeapUltraExpert] | None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    record = _expect_mapping(record_value, "per_example_artifact")
    _expect_exact_keys(record, _ARTIFACT_KEYS, "per_example_artifact")
    _record, path = _validate_file_record(
        record,
        root=root,
        label="per_example_artifact",
    )
    profile_hash = sha256_bytes(canonical_json_bytes(profile))
    expected_row_count = (
        len(profile["final"]["seeds"])
        * int(profile["final"]["samples_per_seed"])
    )
    if (
        record.get("schema") != PREDICTION_ARTIFACT_SCHEMA
        or record.get("evaluation_profile_sha256") != profile_hash
        or record.get("cohort_schema") != COHORT_SCHEMA
        or record.get("generator_schema") != GENERATOR_SCHEMA
        or record.get("family_tag_schema") != FAMILY_TAG_SCHEMA
        or record.get("cohort_role") != "final"
        or record.get("format") != "deterministic_gzip_jsonl"
        or record.get("logits_encoding") != "hex_little_endian_float32"
        or record.get("class_count") != 10
        or record.get("class_order") != list(range(10))
        or record.get("logit_shape") != [10]
        or record.get("argmax_tie_rule") != "lowest_class_index"
        or record.get("loss_formula")
        != "torch_cross_entropy_float32_sum_per_seed_then_float64_total"
        or record.get("gzip_mtime") != 0
        or record.get("row_count") != expected_row_count
    ):
        raise ReceiptValidationError("Three-way artifact format contract mismatch")
    specification = _canonical_cohort_specification(protocol, profile)
    specification_hash = sha256_bytes(canonical_json_bytes(specification))
    if record.get("dataset_specification_sha256") != specification_hash:
        raise ReceiptValidationError("Artifact cohort specification mismatch")

    with path.open("rb") as header_handle:
        header = header_handle.read(10)
    if (
        len(header) != 10
        or header[:3] != b"\x1f\x8b\x08"
        or header[3] != 0
        or struct.unpack("<I", header[4:8])[0] != 0
    ):
        raise ReceiptValidationError(
            "Artifact gzip header is not deterministic (flags/mtime)"
        )

    uncompressed_digest = hashlib.sha256()
    cohort_digest = hashlib.sha256()
    model_results: dict[str, dict[str, Any]] = {
        name: {
            "prediction_digest": hashlib.sha256(),
            "logits_digest": hashlib.sha256(),
            "per_example_digest": hashlib.sha256(),
            "seed_rows": [],
            "total_loss": 0.0,
            "total": 0,
        }
        for name in _MODEL_NAMES
    }
    cohort_rows: list[dict[str, Any]] = []
    row_count = 0
    raw_handle: BinaryIO
    with path.open("rb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="rb") as gzip_handle:
            for seed in profile["final"]["seeds"]:
                n = int(profile["final"]["samples_per_seed"])
                x, targets, metadata = make_chained_task_with_metadata(n, int(seed))
                cohort_digest.update(struct.pack("<q", int(seed)))
                tensor_digest_update(cohort_digest, "x", x)
                tensor_digest_update(cohort_digest, "y", targets)
                for metadata_name in ("starts", "op_types", "operands"):
                    tensor_digest_update(
                        cohort_digest, metadata_name, metadata[metadata_name]
                    )
                replayed_logits = (
                    {
                        name: _predict_exact(model, x, name)
                        for name, model in models.items()
                    }
                    if models is not None
                    else None
                )
                parsed_logits: dict[str, list[tuple[float, ...]]] = {
                    name: [] for name in _MODEL_NAMES
                }
                parsed_predictions: dict[str, list[int]] = {
                    name: [] for name in _MODEL_NAMES
                }
                for index in range(n):
                    payload = gzip_handle.readline()
                    if not payload:
                        raise ReceiptValidationError(
                            "Three-way artifact ended before the frozen cohort"
                        )
                    uncompressed_digest.update(payload)
                    if not payload.endswith(b"\n") or payload == b"\n":
                        raise ReceiptValidationError("Artifact row is not one JSON line")
                    row = _expect_mapping(
                        loads_json_strict(payload[:-1]),
                        f"artifact row {row_count}",
                    )
                    if payload != canonical_json_bytes(row) + b"\n":
                        raise ReceiptValidationError(
                            f"Artifact row {row_count} is not canonical JSON"
                        )
                    _expect_exact_keys(row, _ROW_KEYS, f"artifact row {row_count}")
                    expected_target = int(targets[index])
                    expected_op_types = [
                        int(value) for value in metadata["op_types"][index].tolist()
                    ]
                    expected_operands = [
                        int(value) for value in metadata["operands"][index].tolist()
                    ]
                    expected_families = list(
                        operation_family_tags(metadata["op_types"][index])
                    )
                    if (
                        row.get("dataset_id") != record.get("dataset_id")
                        or row.get("cohort_role") != "final"
                        or row.get("example_id")
                        != f"{record.get('dataset_id')}:{seed}:{index}"
                        or row.get("seed") != seed
                        or row.get("index") != index
                        or row.get("target") != expected_target
                        or row.get("start") != int(metadata["starts"][index])
                        or row.get("op_types") != expected_op_types
                        or row.get("operands") != expected_operands
                        or row.get("operation_family_tags") != expected_families
                    ):
                        raise ReceiptValidationError(
                            f"Artifact row {row_count} task metadata mismatch"
                        )
                    for model_name in _MODEL_NAMES:
                        logits_key = f"{model_name}_logits_f32le_hex"
                        prediction_key = f"{model_name}_prediction"
                        correct_key = f"{model_name}_correct"
                        logits = _decode_logits(
                            row.get(logits_key), f"row {row_count}.{logits_key}"
                        )
                        prediction = _argmax_lowest(logits)
                        if (
                            row.get(prediction_key) != prediction
                            or row.get(correct_key)
                            is not (prediction == expected_target)
                        ):
                            raise ReceiptValidationError(
                                f"Artifact row {row_count} {model_name} outcome mismatch"
                            )
                        if replayed_logits is not None:
                            expected_hex = struct.pack(
                                "<10f",
                                *[
                                    float(value)
                                    for value in replayed_logits[model_name][index].tolist()
                                ],
                            ).hex()
                            if row[logits_key] != expected_hex:
                                raise ReceiptValidationError(
                                    f"Artifact row {row_count} {model_name} logits do "
                                    "not come from the bound checkpoint"
                                )
                        parsed_logits[model_name].append(logits)
                        parsed_predictions[model_name].append(prediction)
                        per_example_payload = canonical_json_bytes(
                            {
                                "seed": int(seed),
                                "index": index,
                                "target": expected_target,
                                "prediction": prediction,
                                "correct": prediction == expected_target,
                                "logits_f32le_hex": row[logits_key],
                            }
                        ) + b"\n"
                        model_results[model_name]["per_example_digest"].update(
                            per_example_payload
                        )
                    row_count += 1
                cohort_rows.append(
                    {
                        "seed": int(seed),
                        "targets": targets.clone(),
                        "op_types": metadata["op_types"].clone(),
                    }
                )
                for model_name in _MODEL_NAMES:
                    logits_tensor = torch.tensor(
                        parsed_logits[model_name], dtype=torch.float32
                    )
                    predictions_tensor = torch.tensor(
                        parsed_predictions[model_name], dtype=torch.long
                    )
                    loss_sum = float(
                        F.cross_entropy(
                            logits_tensor,
                            targets,
                            reduction="sum",
                        ).item()
                    )
                    result = model_results[model_name]
                    result["prediction_digest"].update(struct.pack("<q", int(seed)))
                    tensor_digest_update(result["prediction_digest"], "targets", targets)
                    tensor_digest_update(
                        result["prediction_digest"],
                        "predictions",
                        predictions_tensor,
                    )
                    result["logits_digest"].update(struct.pack("<q", int(seed)))
                    tensor_digest_update(
                        result["logits_digest"], "logits_f32", logits_tensor
                    )
                    result["seed_rows"].append(
                        {
                            "seed": int(seed),
                            "targets": targets.clone(),
                            "predictions": predictions_tensor,
                            "logits": logits_tensor,
                            "loss_sum": loss_sum,
                        }
                    )
                    result["total_loss"] += loss_sum
                    result["total"] += n
            trailing = gzip_handle.read(1)
            if trailing:
                raise ReceiptValidationError(
                    "Three-way artifact contains rows beyond the frozen cohort"
                )
    if row_count != expected_row_count:
        raise ReceiptValidationError("Three-way artifact row count mismatch")
    if uncompressed_digest.hexdigest() != _expect_sha256(
        record.get("uncompressed_sha256"),
        "per_example_artifact.uncompressed_sha256",
    ):
        raise ReceiptValidationError("Artifact uncompressed digest mismatch")

    dataset_sha = cohort_digest.hexdigest()
    dataset_id = sha256_bytes(
        canonical_json_bytes(
            {
                "specification_sha256": specification_hash,
                "dataset_sha256": dataset_sha,
            }
        )
    )
    if (
        record.get("dataset_sha256") != dataset_sha
        or record.get("dataset_id") != dataset_id
    ):
        raise ReceiptValidationError("Artifact dataset identity mismatch")
    finalized_models: dict[str, Any] = {}
    for model_name, result in model_results.items():
        finalized_models[model_name] = {
            "mean_loss": result["total_loss"] / max(1, result["total"]),
            "prediction_sha256": result["prediction_digest"].hexdigest(),
            "logits_sha256": result["logits_digest"].hexdigest(),
            "per_example_sha256": result["per_example_digest"].hexdigest(),
            "seed_rows": result["seed_rows"],
        }
    return record, {
        "schema": COHORT_SCHEMA,
        "generator_schema": GENERATOR_SCHEMA,
        "family_tag_schema": FAMILY_TAG_SCHEMA,
        "cohort_role": "final",
        "dataset_id": dataset_id,
        "specification_sha256": specification_hash,
        "dataset_sha256": dataset_sha,
        "rows": cohort_rows,
        "models": finalized_models,
    }


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
    discordant = int(wins) + int(regressions)
    if discordant <= 0:
        return 1, 1
    tail = min(int(wins), int(regressions))
    numerator = 2 * sum(
        math.comb(discordant, index) for index in range(tail + 1)
    )
    denominator = 1 << discordant
    return (denominator, denominator) if numerator >= denominator else (numerator, denominator)


def _exact_mcnemar_two_sided(wins: int, regressions: int) -> float:
    numerator, denominator = _exact_mcnemar_terms(wins, regressions)
    return numerator / denominator


def _group_correct_count(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[int, int]:
    count = int(mask.sum())
    if count <= 0:
        return 0, 0
    return int(predictions[mask].eq(targets[mask]).sum()), count


def _compare_models(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    cohort: Mapping[str, Any],
    criteria: Mapping[str, Any],
    artifact: Mapping[str, Any],
    profile_hash: str,
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
        seed = cohort_row["seed"]
        reference = baseline_by_seed[seed]
        tuned = candidate_by_seed[seed]
        targets = reference["targets"]
        reference_predictions = reference["predictions"]
        tuned_predictions = tuned["predictions"]
        reference_correct = reference_predictions.eq(targets)
        tuned_correct = tuned_predictions.eq(targets)
        row_wins = int((tuned_correct & ~reference_correct).sum())
        row_regressions = int((reference_correct & ~tuned_correct).sum())
        row_n = int(targets.numel())
        row_ties = row_n - row_wins - row_regressions
        reference_correct_count = int(reference_correct.sum())
        tuned_correct_count = int(tuned_correct.sum())
        reference_accuracy = reference_correct_count / row_n
        tuned_accuracy = tuned_correct_count / row_n
        seed_rows.append(
            {
                "seed": seed,
                "n": row_n,
                "baseline_accuracy": reference_accuracy,
                "candidate_accuracy": tuned_accuracy,
                "accuracy_delta": (tuned_correct_count - reference_correct_count)
                / row_n,
                "baseline_loss": reference["loss_sum"] / row_n,
                "candidate_loss": tuned["loss_sum"] / row_n,
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
    total_baseline_correct_count = int(reference_correct.sum())
    total_candidate_correct_count = int(tuned_correct.sum())
    baseline_accuracy = total_baseline_correct_count / total_n
    candidate_accuracy = total_candidate_correct_count / total_n
    operation_family_rows: list[dict[str, Any]] = []
    operation_family_deltas: list[tuple[int, int]] = []
    for operation_index, operation_name in enumerate(("add", "mul", "sub")):
        mask = op_types[:, 0].eq(operation_index)
        baseline_correct_count, count = _group_correct_count(
            targets, reference_predictions, mask
        )
        candidate_correct_count, candidate_count = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if candidate_count != count:
            raise ReceiptValidationError("Operation-family comparison count mismatch")
        delta_count = candidate_correct_count - baseline_correct_count
        base_value = baseline_correct_count / count if count else 0.0
        candidate_value = candidate_correct_count / count if count else 0.0
        operation_family_rows.append(
            {
                "family": f"first_{operation_name}",
                "n": count,
                "baseline_accuracy": base_value,
                "candidate_accuracy": candidate_value,
                "delta": delta_count / count if count else 0.0,
            }
        )
        operation_family_deltas.append((delta_count, count))
    multiplication_counts = op_types.eq(1).sum(dim=1)
    for count in range(5):
        mask = multiplication_counts.eq(count)
        baseline_correct_count, group_n = _group_correct_count(
            targets, reference_predictions, mask
        )
        candidate_correct_count, candidate_n = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if candidate_n != group_n:
            raise ReceiptValidationError("Operation-family comparison count mismatch")
        delta_count = candidate_correct_count - baseline_correct_count
        base_value = baseline_correct_count / group_n if group_n else 0.0
        candidate_value = candidate_correct_count / group_n if group_n else 0.0
        operation_family_rows.append(
            {
                "family": f"mul_count_{count}",
                "n": group_n,
                "baseline_accuracy": base_value,
                "candidate_accuracy": candidate_value,
                "delta": delta_count / group_n if group_n else 0.0,
            }
        )
        operation_family_deltas.append((delta_count, group_n))
    class_rows: list[dict[str, Any]] = []
    class_deltas: list[tuple[int, int]] = []
    for class_index in range(10):
        mask = targets.eq(class_index)
        baseline_correct_count, count = _group_correct_count(
            targets, reference_predictions, mask
        )
        candidate_correct_count, candidate_count = _group_correct_count(
            targets, tuned_predictions, mask
        )
        if candidate_count != count:
            raise ReceiptValidationError("Class comparison count mismatch")
        delta_count = candidate_correct_count - baseline_correct_count
        base_value = baseline_correct_count / count if count else 0.0
        candidate_value = candidate_correct_count / count if count else 0.0
        class_rows.append(
            {
                "class": str(class_index),
                "n": count,
                "baseline_accuracy": base_value,
                "candidate_accuracy": candidate_value,
                "delta": delta_count / count if count else 0.0,
            }
        )
        class_deltas.append((delta_count, count))
    required_seed_count = _ceil_scaled_fraction(
        len(seed_rows), criteria["minimum_nonregressing_seed_fraction"]
    )
    seed_deltas = [
        (int(row["wins"]) - int(row["regressions"]), int(row["n"]))
        for row in seed_rows
    ]
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
    eligible_families = [row for row, _exact in eligible_family_pairs]
    eligible_classes = [row for row, _exact in eligible_class_pairs]
    eligible_family_deltas = [exact for _row, exact in eligible_family_pairs]
    eligible_class_deltas = [exact for _row, exact in eligible_class_pairs]
    mcnemar_numerator, mcnemar_denominator = _exact_mcnemar_terms(wins, regressions)
    summary = {
        "seed_count": len(seed_rows),
        "n": total_n,
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": (
            total_candidate_correct_count - total_baseline_correct_count
        )
        / total_n,
        "wins": wins,
        "regressions": regressions,
        "ties": ties,
        "exact_mcnemar_p_two_sided": _exact_mcnemar_two_sided(wins, regressions),
        "mean_baseline_loss": float(baseline["mean_loss"]),
        "mean_candidate_loss": float(candidate["mean_loss"]),
        "required_nonregressing_seed_count": required_seed_count,
        "nonregressing_seed_count": sum(
            delta >= 0 for delta, _count in seed_deltas
        ),
        "worst_seed_delta": min(row["accuracy_delta"] for row in seed_rows),
        "eligible_family_count": len(eligible_families),
        "nonregressing_family_count": sum(
            delta >= 0 for delta, _count in eligible_family_deltas
        ),
        "worst_family_delta": min(row["delta"] for row in eligible_families),
        "eligible_class_count": len(eligible_classes),
        "nonregressing_class_count": sum(
            delta >= 0 for delta, _count in eligible_class_deltas
        ),
        "worst_class_delta": min(row["delta"] for row in eligible_classes),
    }
    checks = {
        "accuracy_gain": _ratio_at_least(
            total_candidate_correct_count - total_baseline_correct_count,
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
        "criteria": dict(criteria),
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "seed_rows": seed_rows,
        "operation_family_rows": operation_family_rows,
        "class_rows": class_rows,
        "evidence": {
            "evaluation_profile_sha256": profile_hash,
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
            "per_example_compressed_sha256": artifact["sha256"],
            "per_example_uncompressed_sha256": artifact["uncompressed_sha256"],
        },
    }


_TOP_LEVEL_REQUIRED = {
    "schema",
    "receipt_id",
    "gate_outcome",
    "authority",
    "authentication",
    "integrity_status",
    "trusted_timestamp",
    "claim_scope",
    "evaluation_profile",
    "evaluation_profile_sha256",
    "protocol",
    "selection",
    "artifacts",
    "comparisons",
    "per_example_artifact",
    "code_bindings",
    "source_snapshot",
    "git_at_protocol_freeze",
    "git_at_finalization",
    "environment",
    "evaluation_rng",
    "final_invocation_sha256",
    "single_use_scope",
}
_TOP_LEVEL_OPTIONAL = {"created_at"}


def _load_receipt(
    receipt_or_path: str | Path | Mapping[str, Any],
) -> tuple[Mapping[str, Any], Path | None]:
    if isinstance(receipt_or_path, Mapping):
        return _expect_mapping(receipt_or_path, "receipt"), None
    path = Path(receipt_or_path)
    if not path.is_file():
        raise ReceiptValidationError(f"Receipt file is missing: {path}")
    return _expect_mapping(loads_json_strict(path.read_bytes()), "receipt"), path


def validate_receipt(
    receipt_or_path: str | Path | Mapping[str, Any],
    *,
    root: str | Path | None = None,
    verify_inference: bool = True,
) -> dict[str, Any]:
    """Validate and replay one immutable three-way v51.2 receipt.

    ``verify_inference=False`` is intentionally limited to synthetic unit
    fixtures.  It still validates all JSONL logits and recomputes both paired
    comparisons, but it does not establish that the checkpoint files produced
    those logits.
    """

    if not isinstance(verify_inference, bool):
        raise ReceiptValidationError("verify_inference must be a boolean")
    receipt, receipt_path = _load_receipt(receipt_or_path)
    actual_keys = set(receipt)
    if not _TOP_LEVEL_REQUIRED <= actual_keys or not actual_keys <= (
        _TOP_LEVEL_REQUIRED | _TOP_LEVEL_OPTIONAL
    ):
        raise ReceiptValidationError(
            "Receipt top-level fields mismatch; "
            f"missing={sorted(_TOP_LEVEL_REQUIRED - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_REQUIRED - _TOP_LEVEL_OPTIONAL)}"
        )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReceiptValidationError("Unsupported three-way receipt schema")
    if (
        receipt.get("authority") != AUTHORITY
        or receipt.get("claim_scope") != CLAIM_SCOPE
        or receipt.get("authentication") != "none"
        or receipt.get("integrity_status")
        != "content_bound_not_authenticated"
        or receipt.get("trusted_timestamp") is not False
    ):
        raise ReceiptValidationError("Receipt trust/claim/authority contract changed")
    receipt_id = _expect_sha256(receipt.get("receipt_id"), "receipt_id")
    if receipt_id != _digest_without(receipt, "receipt_id"):
        raise ReceiptValidationError("Receipt content digest mismatch")
    profile = _validate_profile(receipt)
    profile_hash = sha256_bytes(canonical_json_bytes(profile))
    resolved_root = (
        Path(root).resolve()
        if root is not None
        else (
            receipt_path.resolve().parent
            if receipt_path is not None
            else Path.cwd().resolve()
        )
    )
    protocol_record, protocol, _protocol_path = _validate_protocol(
        receipt.get("protocol"), root=resolved_root, profile=profile
    )
    if receipt.get("code_bindings") != protocol.get("code_bindings"):
        raise ReceiptValidationError("Receipt code bindings differ from protocol")
    if receipt.get("source_snapshot") != protocol.get("source_snapshot"):
        raise ReceiptValidationError("Receipt source snapshot differs from protocol")
    if receipt.get("git_at_protocol_freeze") != protocol.get("git"):
        raise ReceiptValidationError("Receipt protocol Git binding mismatch")
    git_final = _expect_mapping(
        receipt.get("git_at_finalization"), "git_at_finalization"
    )
    if (
        git_final.get("dirty") is not False
        or git_final.get("commit") != protocol["git"]["commit"]
    ):
        raise ReceiptValidationError("Finalization did not use the frozen clean commit")
    _environment_compatible(
        protocol.get("environment_at_freeze"),
        receipt.get("environment"),
        "final environment",
    )

    artifacts = _expect_mapping(receipt.get("artifacts"), "artifacts")
    _expect_exact_keys(
        artifacts,
        {"release_baseline", "prior_candidate", "candidate"},
        "artifacts",
    )
    release_artifact, _release_path, release_state = _validate_checkpoint_record(
        artifacts["release_baseline"],
        root=resolved_root,
        label="artifacts.release_baseline",
        verify_checkpoint=verify_inference,
    )
    prior_artifact, _prior_path, prior_state = _validate_checkpoint_record(
        artifacts["prior_candidate"],
        root=resolved_root,
        label="artifacts.prior_candidate",
        verify_checkpoint=verify_inference,
    )
    candidate_artifact, _candidate_path, candidate_state = (
        _validate_checkpoint_record(
            artifacts["candidate"],
            root=resolved_root,
            label="artifacts.candidate",
            verify_checkpoint=verify_inference,
        )
    )
    if release_artifact != protocol.get("baseline"):
        raise ReceiptValidationError("Release baseline artifact/protocol mismatch")
    if prior_artifact != protocol.get("prior_candidate"):
        raise ReceiptValidationError("Prior candidate artifact/protocol mismatch")
    selection_record, selection, _selection_path = _validate_selection(
        receipt.get("selection"),
        root=resolved_root,
        protocol=protocol,
        profile=profile,
        candidate_artifact=candidate_artifact,
        release_state=release_state,
        candidate_state=candidate_state,
        verify_checkpoint=verify_inference,
    )

    models: dict[str, ChampionNetCognitiveLeapUltraExpert] | None = None
    if verify_inference:
        assert release_state is not None
        assert prior_state is not None
        assert candidate_state is not None
        models = {
            "release_baseline": _strict_model_from_state(
                release_state, "release_baseline"
            ),
            "prior_candidate": _strict_model_from_state(
                prior_state, "prior_candidate"
            ),
            "candidate": _strict_model_from_state(candidate_state, "candidate"),
        }
    artifact_record, replay = _validate_prediction_artifact(
        receipt.get("per_example_artifact"),
        root=resolved_root,
        protocol=protocol,
        profile=profile,
        models=models,
    )
    release_comparison = _compare_models(
        replay["models"]["release_baseline"],
        replay["models"]["candidate"],
        replay,
        profile["final"]["release_continuity_criteria"],
        artifact_record,
        profile_hash,
    )
    prior_comparison = _compare_models(
        replay["models"]["prior_candidate"],
        replay["models"]["candidate"],
        replay,
        profile["final"]["prior_candidate_superiority_criteria"],
        artifact_record,
        profile_hash,
    )
    comparisons = _expect_mapping(receipt.get("comparisons"), "comparisons")
    _expect_exact_keys(
        comparisons,
        {"release_continuity", "prior_candidate_superiority"},
        "comparisons",
    )
    _expect_equal(
        comparisons["release_continuity"],
        release_comparison,
        "release_continuity comparison",
    )
    _expect_equal(
        comparisons["prior_candidate_superiority"],
        prior_comparison,
        "prior_candidate_superiority comparison",
    )
    passed = bool(release_comparison["passed"] and prior_comparison["passed"])
    expected_outcome = "pass" if passed else "reject"
    if receipt.get("gate_outcome") != expected_outcome:
        raise ReceiptValidationError("Top-level gate is not the logical AND")
    if receipt.get("single_use_scope") != "this_local_output_directory_only":
        raise ReceiptValidationError("Receipt single-use scope changed")
    evaluation_rng = _expect_mapping(receipt.get("evaluation_rng"), "evaluation_rng")
    _expect_exact_keys(
        evaluation_rng,
        {"cpu_state_before_sha256", "cpu_state_after_sha256", "unchanged"},
        "evaluation_rng",
    )
    rng_before = _expect_sha256(
        evaluation_rng.get("cpu_state_before_sha256"),
        "evaluation_rng.cpu_state_before_sha256",
    )
    if (
        evaluation_rng.get("cpu_state_after_sha256") != rng_before
        or evaluation_rng.get("unchanged") is not True
    ):
        raise ReceiptValidationError("Final evaluation RNG state changed")
    final_invocation_sha = _expect_sha256(
        receipt.get("final_invocation_sha256"), "final_invocation_sha256"
    )
    environment = _expect_mapping(receipt.get("environment"), "environment")
    expected_invocation_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "protocol_sha256": protocol_record["content_sha256"],
                "selection_sha256": selection_record["content_sha256"],
                "invocation": environment.get("invocation"),
                "torch": environment.get("torch"),
            }
        )
    )
    if final_invocation_sha != expected_invocation_sha:
        raise ReceiptValidationError("Final invocation binding mismatch")

    return {
        "valid": True,
        "schema": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "gate_outcome": expected_outcome,
        "evaluation_profile_schema": profile["schema"],
        "evaluation_profile_sha256": profile_hash,
        "protocol_sha256": protocol_record["content_sha256"],
        "selection_sha256": selection_record["content_sha256"],
        "release_baseline_sha256": release_artifact["sha256"],
        "prior_candidate_sha256": prior_artifact["sha256"],
        "candidate_sha256": candidate_artifact["sha256"],
        "per_example_artifact_sha256": artifact_record["sha256"],
        "release_continuity_passed": release_comparison["passed"],
        "prior_candidate_superiority_passed": prior_comparison["passed"],
        "checkpoint_inference_replayed": verify_inference,
        "authority": dict(AUTHORITY),
    }


def try_validate_receipt(
    receipt_or_path: str | Path | Mapping[str, Any],
    *,
    root: str | Path | None = None,
    verify_inference: bool = True,
) -> dict[str, Any]:
    """Return a non-raising validation result for UI/store call sites."""

    try:
        return validate_receipt(
            receipt_or_path,
            root=root,
            verify_inference=verify_inference,
        )
    except Exception as exc:  # fail closed across malformed torch/filesystem input
        return {
            "valid": False,
            "schema": RECEIPT_SCHEMA,
            "error": str(exc),
            "checkpoint_inference_replayed": False,
            "authority": dict(AUTHORITY),
        }


__all__ = [
    "AUTHORITY",
    "CANONICAL_EVALUATION_PROFILE_SHA256",
    "CLAIM_SCOPE",
    "PREDICTION_ARTIFACT_SCHEMA",
    "PROFILE_HASH_ALLOWLIST",
    "PROFILE_SCHEMA",
    "RECEIPT_SCHEMA",
    "ReceiptValidationError",
    "canonical_evaluation_profile",
    "canonical_json_bytes",
    "loads_json_strict",
    "sha256_bytes",
    "sha256_file",
    "state_dict_summary",
    "tensor_digest_update",
    "try_validate_receipt",
    "validate_receipt",
]
