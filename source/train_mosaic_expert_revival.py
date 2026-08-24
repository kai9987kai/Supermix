"""Fail-closed Mosaic Expert Revival fallback for the frozen v70 checkpoint.

This utility is intentionally not a general fine-tuner.  It will run only when
all of the following are supplied and hash-matched:

* the preregistered v70 parent checkpoint and its embedded tokenizer;
* a deterministic Mosaic bundle manifest;
* a frozen v71 clean-gate receipt whose decision is ``rejected``.

Only four least-used routed experts per MoE layer, their four gate rows, and
their four non-gradient selection-bias entries may change.  Everything else is
snapshotted and checked byte-for-byte before any recoverable checkpoint is
written.  No activation pointer is read or written by this module.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

import build_mosaic_expert_revival_dataset as mosaic_data
import mimomix_core
import mimomix_text
import train_mimomix_talk
import v72_model_promotion
from mimomix_core import MiMoMixModel, SparseMoEFeedForward
from train_mimomix_talk import CHECKPOINT_SCHEMA as TALK_CHECKPOINT_SCHEMA
from train_mimomix_talk import load_talk_checkpoint


V71_FAILURE_SCHEMA = "supermix-v72-collision-free-promotion-receipt-v1"
V71_PROMOTION_MANIFEST_SCHEMA = "supermix-v72-collision-free-promotion-manifest-v1"
V71_PROMOTION_POLICY_ID = "supermix-v72-collision-free-promotion-policy-v1"
CHECKPOINT_SCHEMA = "supermix-mosaic-expert-revival-checkpoint-v1"
PROVENANCE_SCHEMA = "supermix-mosaic-expert-revival-provenance-v1"
SELECTION_SCHEMA = "supermix-mosaic-expert-revival-metrics-audit-v1"
REVIVED_PER_LAYER = 4
BATCH_PATTERN = ("mosaic", "mosaic", "mosaic", "math", "dialogue")

PREREGISTRATION: dict[str, Any] = {
    "name": "mosaic_expert_revival",
    "parent": {
        "run_name": "v70_moe",
        "n_layers": 4,
        "n_dense_layers": 1,
        "n_routed_experts": 32,
        "revived_experts_per_moe_layer": REVIVED_PER_LAYER,
    },
    "calibration": {
        "split": "train",
        "domains": ["dialogue", "math"],
        "subset_per_domain": 256,
        "subset_rule": "lowest component ids after deterministic dataset construction",
        "uses_targets": False,
        "selection": "four lowest combined normalized prompt-token loads; ties by expert id",
        "donors": "two dialogue-affine clones and two math-affine clones per layer",
    },
    "training": {
        "steps": 2400,
        "batch_size": 16,
        "sequence_length": 128,
        "optimizer": "AdamW",
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "expert_weight_decay": 0.01,
        "gate_weight_decay": 0.0,
        "peak_learning_rate": 4e-4,
        "minimum_learning_rate": 4e-5,
        "warmup_steps": 192,
        "gradient_clip": 1.0,
        "batch_pattern": list(BATCH_PATTERN),
        "routing_boost": {"initial": 1.5, "last_step": 600, "applies_to": "mosaic_only"},
        "selected_bias_update_speed": 1e-3,
        "checkpoint_every": 400,
        "seed": 710_413,
        "deterministic_algorithms": True,
        "torch_num_threads": 4,
        "torch_num_interop_threads": 1,
    },
    "selection": {
        "split": "dev",
        "minimum_composition_gain": 0.10,
        "minimum_math_chain_gain": 0.10,
        "original_math_family_regression_tolerance": 0.0,
        "legacy_chat_item_regression_tolerance": 0.0,
        "legacy_chat_aggregate_regression_tolerance": 0.0,
        "dialogue_metric_semantics": "seen-target retention only; not semantic quality or novel conversation",
        "rule": "earliest checkpoint satisfying every gate; no weighted averaging",
        "authority": "metrics-only checks are audit-only until a content-bound Mosaic evaluator receipt exists",
        "holdout_role": "one final audit after selection; never ranks checkpoints",
    },
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_sha256(tokenizer: mimomix_text.WordTokenizer) -> str:
    return sha256_bytes(canonical_json(tokenizer.to_dict()).encode("utf-8"))


def preregistration_sha256() -> str:
    return sha256_bytes(canonical_json(PREREGISTRATION).encode("utf-8"))


def dependency_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        Path(mosaic_data.__file__).resolve(),
        Path(mimomix_core.__file__).resolve(),
        Path(mimomix_text.__file__).resolve(),
        Path(train_mimomix_talk.__file__).resolve(),
    )
    return {path.name: sha256_file(path) for path in paths}


def configure_deterministic_runtime() -> None:
    training = PREREGISTRATION["training"]
    torch.set_num_threads(int(training["torch_num_threads"]))
    if torch.get_num_interop_threads() != int(training["torch_num_interop_threads"]):
        torch.set_num_interop_threads(int(training["torch_num_interop_threads"]))
    torch.use_deterministic_algorithms(bool(training["deterministic_algorithms"]))
    torch.manual_seed(int(training["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training["seed"]))


def runtime_provenance(device_name: str) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_build": sys.version,
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "system": platform.system(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device_name,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read valid JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def verify_v71_failure_receipt(
    path: Path,
    expected_sha256: str,
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    candidate_checkpoint_path: Path,
    expected_candidate_checkpoint_sha256: str,
    expected_parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Verify the authoritative frozen v72 gate rejected the v71 candidate.

    The frozen evaluator predates this fallback and intentionally contains no
    fallback-specific authorization fields.  Authorization comes from an exact
    content-bound rejection: receipt, manifest, evaluator, v70 baseline, and
    v71 candidate bytes must all agree.
    """

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"v71 failure-receipt hash mismatch: expected {expected_sha256}, got {actual}")
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise ValueError(
            f"v71 promotion-manifest hash mismatch: expected {expected_manifest_sha256}, "
            f"got {actual_manifest_hash}"
        )
    actual_candidate_hash = sha256_file(candidate_checkpoint_path)
    if actual_candidate_hash != expected_candidate_checkpoint_sha256:
        raise ValueError(
            f"v71 candidate hash mismatch: expected {expected_candidate_checkpoint_sha256}, "
            f"got {actual_candidate_hash}"
        )
    receipt = _read_json_object(path)
    manifest = _read_json_object(manifest_path)
    try:
        v72_model_promotion._validate_manifest(manifest)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"v71 promotion manifest fails the authoritative validator: {exc}") from exc
    if receipt.get("schema") != V71_FAILURE_SCHEMA:
        raise ValueError("v71 gate receipt has the wrong authoritative schema")
    if receipt.get("policy_id") != V71_PROMOTION_POLICY_ID:
        raise ValueError("v71 gate receipt has the wrong frozen policy id")
    if receipt.get("mode") != "review_only_no_write_pointer":
        raise ValueError("v71 gate receipt is not review-only")
    if receipt.get("passed") is not False:
        raise ValueError("v71 gate receipt does not record passed=false")
    decision = receipt.get("decision")
    if not isinstance(decision, Mapping) or decision.get("passed") is not False:
        raise ValueError("v71 gate receipt has no nested failed decision")
    blockers = decision.get("blockers")
    if not isinstance(blockers, list) or not blockers or not all(isinstance(value, str) and value for value in blockers):
        raise ValueError("v71 gate rejection has no concrete blockers")
    pointer = receipt.get("pointer")
    if not isinstance(pointer, Mapping) or any(
        pointer.get(field) is not False for field in ("write_requested", "write_supported", "pointer_written")
    ):
        raise ValueError("v71 gate receipt is not a no-pointer review receipt")

    receipt_manifest = receipt.get("manifest")
    if not isinstance(receipt_manifest, Mapping):
        raise ValueError("v71 gate receipt omits its manifest binding")
    if receipt_manifest.get("sha256") != actual_manifest_hash:
        raise ValueError("v71 gate receipt manifest hash differs from supplied manifest")
    if Path(str(receipt_manifest.get("path", ""))).expanduser().resolve() != manifest_path.resolve():
        raise ValueError("v71 gate receipt manifest path differs from supplied manifest")
    if manifest.get("schema") != V71_PROMOTION_MANIFEST_SCHEMA:
        raise ValueError("v71 promotion manifest has the wrong schema")
    baseline_binding = manifest.get("baseline")
    candidate_binding = manifest.get("candidate")
    if not isinstance(baseline_binding, Mapping) or not isinstance(candidate_binding, Mapping):
        raise ValueError("v71 promotion manifest model bindings are invalid")
    if baseline_binding.get("checkpoint_sha256") != expected_parent_checkpoint_sha256:
        raise ValueError("v71 promotion manifest baseline is not the supplied v70 parent")
    expected_candidate_path = Path(str(candidate_binding.get("checkpoint_expected", ""))).expanduser().resolve()
    if expected_candidate_path != candidate_checkpoint_path.resolve():
        raise ValueError("v71 promotion manifest expected a different candidate path")

    artifacts = receipt.get("artifact_binding")
    if not isinstance(artifacts, Mapping):
        raise ValueError("v71 gate receipt omits artifact bindings")
    required_hashes = {
        "baseline_checkpoint_sha256": expected_parent_checkpoint_sha256,
        "candidate_checkpoint_sha256": actual_candidate_hash,
    }
    for key, expected in required_hashes.items():
        if artifacts.get(key) != expected:
            raise ValueError(f"v71 gate receipt artifact mismatch for {key}")
    if Path(str(artifacts.get("candidate_checkpoint", ""))).expanduser().resolve() != candidate_checkpoint_path.resolve():
        raise ValueError("v71 gate receipt candidate path differs from supplied candidate")
    if artifacts.get("changed_during_evaluation") != []:
        raise ValueError("v71 gate artifacts changed during evaluation")
    for side, binding in (("baseline", baseline_binding), ("candidate", candidate_binding)):
        corpus_path = Path(str(binding.get("corpus", ""))).expanduser().resolve()
        corpus_hash = binding.get("corpus_sha256")
        if not corpus_path.is_file() or not _is_sha256(corpus_hash) or sha256_file(corpus_path) != corpus_hash:
            raise ValueError(f"v71 promotion manifest has a stale {side} corpus binding")
        if Path(str(artifacts.get(f"{side}_corpus", ""))).expanduser().resolve() != corpus_path:
            raise ValueError(f"v71 gate receipt {side} corpus path differs from manifest")
        if artifacts.get(f"{side}_corpus_sha256") != corpus_hash:
            raise ValueError(f"v71 gate receipt {side} corpus hash differs from manifest")
    evaluator_path = Path(str(artifacts.get("evaluator_path", ""))).expanduser().resolve()
    evaluator_hash = artifacts.get("evaluator_sha256")
    if evaluator_path != Path(v72_model_promotion.__file__).resolve():
        raise ValueError("v71 gate receipt names a different evaluator implementation")
    if not evaluator_path.is_file() or not _is_sha256(evaluator_hash) or sha256_file(evaluator_path) != evaluator_hash:
        raise ValueError("v71 gate evaluator binding is missing or stale")
    for receipt_key, manifest_key in (("prompt_set_sha256", "prompt_set"), ("chat_set_sha256", "chat_set")):
        manifest_set = manifest.get(manifest_key)
        if not isinstance(manifest_set, Mapping) or artifacts.get(receipt_key) != manifest_set.get("sha256"):
            raise ValueError(f"v71 gate receipt differs from manifest {manifest_key} binding")
    return receipt


def _verified_file_bytes(bundle_dir: Path, manifest: Mapping[str, Any], relative: str) -> bytes:
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not isinstance(files.get(relative), Mapping):
        raise ValueError(f"dataset manifest omits {relative}")
    expected = files[relative].get("sha256")
    if not _is_sha256(expected):
        raise ValueError(f"dataset manifest has invalid hash for {relative}")
    path = bundle_dir / relative
    value = path.read_bytes()
    actual = sha256_bytes(value)
    if actual != expected:
        raise ValueError(f"dataset file hash mismatch for {relative}: expected {expected}, got {actual}")
    return value


def _decode_jsonl(value: bytes, relative: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(value.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {relative}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object row in {relative}:{line_number}")
        rows.append(row)
    if not rows:
        raise ValueError(f"dataset file is empty: {relative}")
    return rows


def load_training_bundle(bundle_dir: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    """Load train examples only; dev and holdout are integrity-hashed but unopened."""

    manifest_path = bundle_dir / "manifest.json"
    actual_manifest_hash = sha256_file(manifest_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise ValueError(
            f"dataset manifest hash mismatch: expected {expected_manifest_sha256}, got {actual_manifest_hash}"
        )
    manifest = _read_json_object(manifest_path)
    if manifest.get("schema") != mosaic_data.BUNDLE_SCHEMA:
        raise ValueError("unsupported Mosaic dataset manifest schema")
    if manifest.get("cross_split_collision_count") != 0:
        raise ValueError("dataset manifest reports cross-split collisions")
    tokenizer_binding = manifest.get("tokenizer_binding")
    if (
        not isinstance(tokenizer_binding, Mapping)
        or tokenizer_binding.get("sequence_length") != PREREGISTRATION["training"]["sequence_length"]
        or tokenizer_binding.get("unknown_token_policy") != "reject_before_split_or_write"
    ):
        raise ValueError("dataset is not bound to the frozen tokenizer/sequence policy")
    token_validation = manifest.get("prewrite_token_validation")
    if (
        not isinstance(token_validation, Mapping)
        or token_validation.get("unknown_rows") != 0
        or token_validation.get("overlength_rows") != 0
        or int(token_validation.get("maximum_observed_turn_tokens", 10**9))
        > PREREGISTRATION["training"]["sequence_length"]
    ):
        raise ValueError("dataset failed pre-write tokenizer validation")
    split_policy = manifest.get("split_policy")
    if not isinstance(split_policy, Mapping) or "never ranks" not in str(split_policy.get("holdout_role", "")):
        raise ValueError("dataset manifest does not seal holdout from model selection")
    if "train atomics only" not in str(split_policy.get("calibration", "")):
        raise ValueError("dataset manifest does not bind calibration to train atomics")

    external_scans = manifest.get("external_corpus_scans")
    if not isinstance(external_scans, list) or len(external_scans) < 2:
        raise ValueError("training requires forbidden-corpus scans for both v70 and v71")
    if any(not isinstance(scan, Mapping) or not _is_sha256(scan.get("sha256")) for scan in external_scans):
        raise ValueError("dataset has invalid external-corpus scan provenance")
    if len({str(scan["sha256"]) for scan in external_scans}) < 2:
        raise ValueError("v70 and v71 forbidden-corpus scans must bind distinct corpus bytes")
    if manifest.get("external_corpus_collision_count") != 0:
        raise ValueError("dataset reports a protected external-corpus collision")
    replay_exemption = manifest.get("dialogue_replay_exemption")
    if not isinstance(replay_exemption, Mapping) or replay_exemption.get("scope") != (
        "atomic dialogue prompts sourced from the approved replay corpus only"
    ):
        raise ValueError("dataset lacks the narrow dialogue replay exemption")

    # Integrity check the opaque ID ledger plus dev/holdout artifacts without
    # parsing any evaluation examples into the training process.
    id_file = manifest.get("id_file")
    if not isinstance(id_file, str):
        raise ValueError("dataset manifest omits id_file")
    id_payload = _verified_file_bytes(bundle_dir, manifest, id_file)
    if sha256_bytes(id_payload) != manifest.get("id_set_sha256"):
        raise ValueError("dataset id-set hash mismatch")
    if len([line for line in id_payload.splitlines() if line]) != manifest.get("id_count"):
        raise ValueError("dataset id-set count mismatch")
    for relative in (
        "dev.jsonl",
        "dev_dialogue.jsonl",
        "dev_math.jsonl",
        "holdout.jsonl",
        "holdout_dialogue.jsonl",
        "holdout_math.jsonl",
    ):
        _verified_file_bytes(bundle_dir, manifest, relative)

    result: dict[str, Any] = {
        "manifest": manifest,
        "manifest_sha256": actual_manifest_hash,
    }
    seen_components: dict[str, str] = {}
    seen_rows: set[str] = set()
    for split in ("train",):
        mosaic_relative = f"{split}.jsonl"
        mosaic_rows = _decode_jsonl(_verified_file_bytes(bundle_dir, manifest, mosaic_relative), mosaic_relative)
        for row in mosaic_rows:
            if row.get("split") != split or not mosaic_data.verify_mosaic_row(row):
                raise ValueError(f"invalid or mislabeled row in {mosaic_relative}")
            row_id = str(row["row_id"])
            if row_id in seen_rows:
                raise ValueError(f"duplicate mosaic row id: {row_id}")
            seen_rows.add(row_id)
            for component in row.get("components", []):
                component_id = str(component["component_id"])
                previous = seen_components.setdefault(component_id, split)
                if previous != split:
                    raise ValueError(f"component split collision: {component_id}")
        result[f"{split}_mosaic"] = mosaic_rows
        for domain in ("dialogue", "math"):
            relative = f"{split}_{domain}.jsonl"
            atomic_rows = _decode_jsonl(_verified_file_bytes(bundle_dir, manifest, relative), relative)
            for row in atomic_rows:
                if row.get("split") != split or row.get("domain") != domain or not mosaic_data.verify_atomic_row(row):
                    raise ValueError(f"invalid or mislabeled row in {relative}")
                component_id = str(row["row_id"])
                previous = seen_components.setdefault(component_id, split)
                if previous != split:
                    raise ValueError(f"component split collision: {component_id}")
            result[f"{split}_{domain}"] = atomic_rows
    return result


def verify_dataset_external_corpus_binding(
    dataset_manifest: Mapping[str, Any],
    promotion_manifest: Mapping[str, Any],
) -> None:
    scans = dataset_manifest.get("external_corpus_scans")
    baseline = promotion_manifest.get("baseline")
    candidate = promotion_manifest.get("candidate")
    if not isinstance(scans, list) or not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
        raise ValueError("cannot bind dataset scans to the promotion corpora")
    actual = {str(scan.get("sha256", "")) for scan in scans if isinstance(scan, Mapping)}
    expected = {str(baseline.get("corpus_sha256", "")), str(candidate.get("corpus_sha256", ""))}
    if len(expected) != 2 or not all(_is_sha256(value) for value in expected) or actual != expected:
        raise ValueError("dataset forbidden-corpus scans are not exactly the frozen v70/v71 corpora")


def fixed_train_calibration_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_domain: str,
    limit: int = 256,
) -> list[Mapping[str, Any]]:
    if limit <= 0:
        raise ValueError("calibration subset limit must be positive")
    for row in rows:
        if (
            row.get("schema") != mosaic_data.ATOMIC_ROW_SCHEMA
            or row.get("split") != "train"
            or row.get("domain") != expected_domain
        ):
            raise ValueError("calibration subset may contain only train atomic rows from one domain")
    return sorted(rows, key=lambda row: str(row.get("row_id", "")))[:limit]


def _pair(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("schema") == mosaic_data.ATOMIC_ROW_SCHEMA:
        component = row["component"]
        return str(component["user"]), str(component["assistant"])
    return str(row["user"]), str(row["assistant"])


def validate_tokenizer_coverage(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: mimomix_text.WordTokenizer,
) -> dict[str, int]:
    """Reject even one unknown token; the checkpoint tokenizer is immutable."""

    checked = 0
    unknown_rows: list[str] = []
    for row in rows:
        checked += 1
        user, assistant = _pair(row)
        if tokenizer.unknown_rate(user) != 0.0 or tokenizer.unknown_rate(assistant) != 0.0:
            unknown_rows.append(str(row.get("row_id", f"row-{checked}")))
    if unknown_rows:
        raise ValueError(
            f"v70 tokenizer cannot encode {len(unknown_rows)} rows without <unk>: {unknown_rows[:5]}"
        )
    return {"checked_rows": checked, "unknown_rows": 0}


def load_verified_v70(
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
    *,
    map_location: str = "cpu",
) -> tuple[MiMoMixModel, mimomix_text.WordTokenizer, dict[str, Any]]:
    actual = sha256_file(checkpoint_path)
    if actual != expected_checkpoint_sha256:
        raise ValueError(f"v70 checkpoint hash mismatch: expected {expected_checkpoint_sha256}, got {actual}")
    model, tokenizer, payload = load_talk_checkpoint(checkpoint_path, map_location=map_location)
    extra = payload.get("extra")
    if not isinstance(extra, Mapping) or extra.get("run_name") != PREREGISTRATION["parent"]["run_name"]:
        raise ValueError("parent checkpoint is not the preregistered v70_moe run")
    config = model.config
    expected_architecture = PREREGISTRATION["parent"]
    for field in ("n_layers", "n_dense_layers", "n_routed_experts"):
        if int(getattr(config, field)) != int(expected_architecture[field]):
            raise ValueError(
                f"v70 architecture mismatch for {field}: expected {expected_architecture[field]}, "
                f"got {getattr(config, field)}"
            )
    actual_tokenizer_hash = tokenizer_sha256(tokenizer)
    if actual_tokenizer_hash != expected_tokenizer_sha256:
        raise ValueError(
            f"embedded v70 tokenizer hash mismatch: expected {expected_tokenizer_sha256}, "
            f"got {actual_tokenizer_hash}"
        )
    return model, tokenizer, payload


def moe_layers(model: MiMoMixModel) -> list[tuple[str, SparseMoEFeedForward]]:
    result = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, SparseMoEFeedForward)
    ]
    if not result:
        raise ValueError("model has no routed MoE layers")
    for name, module in result:
        if module.n_routed < REVIVED_PER_LAYER + 2:
            raise ValueError(f"{name} has too few experts for four revived rows and two distinct donors")
    return result


@torch.no_grad()
def calibrate_expert_usage(
    model: MiMoMixModel,
    tokenizer: mimomix_text.WordTokenizer,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_domain: str,
    maximum_length: int = 128,
) -> dict[str, list[float]]:
    """Measure prompt-only routing on the fixed train subset, one unpadded turn at a time."""

    if expected_domain not in {"dialogue", "math"}:
        raise ValueError("calibration domain must be dialogue or math")
    if not rows:
        raise ValueError("calibration requires train rows")
    layers = moe_layers(model)
    totals = {name: torch.zeros(module.n_routed, dtype=torch.float64) for name, module in layers}
    tokens = {name: 0 for name, _ in layers}
    biases = {name: module.expert_bias.detach().cpu().clone() for name, module in layers}
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    for row in rows:
        if row.get("schema") != mosaic_data.ATOMIC_ROW_SCHEMA or row.get("split") != "train":
            raise ValueError("calibration may use only atomic train rows")
        if row.get("domain") != expected_domain:
            raise ValueError("calibration row domain mismatch")
        user, _ = _pair(row)
        encoded, _ = tokenizer.encode_turn(user, None)
        if len(encoded) > maximum_length:
            raise ValueError(f"calibration prompt exceeds {maximum_length} tokens: {row.get('row_id')}")
        inputs = torch.tensor([encoded], dtype=torch.long, device=device)
        model(inputs, return_mtp=False, thinking_cycles=1)
        token_count = len(encoded)
        for name, module in layers:
            totals[name] += module.last_expert_load.detach().cpu().double() * token_count
            tokens[name] += token_count
    if was_training:
        model.train()
    for name, module in layers:
        if not torch.equal(module.expert_bias.detach().cpu(), biases[name]):
            raise RuntimeError(f"calibration mutated router bias in {name}")
        module.pending_load.zero_()
        module.pending_batches.zero_()
    return {name: [float(value) for value in totals[name] / max(1, tokens[name])] for name, _ in layers}


def build_expert_plan(
    model: MiMoMixModel,
    dialogue_usage: Mapping[str, Sequence[float]],
    math_usage: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    plan: dict[str, Any] = {"revived_per_layer": REVIVED_PER_LAYER, "layers": {}}
    for name, module in moe_layers(model):
        dialogue = [float(value) for value in dialogue_usage.get(name, ())]
        maths = [float(value) for value in math_usage.get(name, ())]
        if len(dialogue) != module.n_routed or len(maths) != module.n_routed:
            raise ValueError(f"calibration width mismatch in {name}")
        if any(not math.isfinite(value) or value < 0.0 for value in (*dialogue, *maths)):
            raise ValueError(f"invalid calibration load in {name}")
        dialogue_total = sum(dialogue)
        math_total = sum(maths)
        if dialogue_total <= 0.0 or math_total <= 0.0:
            raise ValueError(f"empty calibration routing in {name}")
        combined = [dialogue[index] / dialogue_total + maths[index] / math_total for index in range(module.n_routed)]
        revived = sorted(range(module.n_routed), key=lambda index: (combined[index], index))[:REVIVED_PER_LAYER]
        available = [index for index in range(module.n_routed) if index not in revived]
        affinity = [dialogue[index] / dialogue_total - maths[index] / math_total for index in range(module.n_routed)]
        dialogue_donor = max(available, key=lambda index: (affinity[index], -index))
        math_candidates = [index for index in available if index != dialogue_donor]
        math_donor = min(math_candidates, key=lambda index: (affinity[index], index))
        assignments = {
            str(expert_id): dialogue_donor if position < 2 else math_donor
            for position, expert_id in enumerate(revived)
        }
        plan["layers"][name] = {
            "revived": revived,
            "dialogue_donor": dialogue_donor,
            "math_donor": math_donor,
            "assignments": assignments,
            "combined_load": [round(value, 12) for value in combined],
            "dialogue_affinity": [round(value, 12) for value in affinity],
        }
    return plan


@torch.no_grad()
def revive_from_donors(model: MiMoMixModel, plan: Mapping[str, Any]) -> None:
    layer_map = dict(moe_layers(model))
    plan_layers = plan.get("layers")
    if not isinstance(plan_layers, Mapping) or set(plan_layers) != set(layer_map):
        raise ValueError("expert plan does not exactly cover the model's MoE layers")
    for name, module in layer_map.items():
        entry = plan_layers[name]
        revived = [int(value) for value in entry.get("revived", ())]
        assignments = entry.get("assignments")
        if len(revived) != REVIVED_PER_LAYER or len(set(revived)) != REVIVED_PER_LAYER:
            raise ValueError(f"{name} does not select exactly four unique experts")
        if not isinstance(assignments, Mapping) or set(assignments) != {str(value) for value in revived}:
            raise ValueError(f"{name} donor assignments are incomplete")
        for destination in revived:
            donor = int(assignments[str(destination)])
            if donor in revived or donor < 0 or donor >= module.n_routed:
                raise ValueError(f"{name} has invalid donor {donor} for expert {destination}")
            module.experts[destination].load_state_dict(copy.deepcopy(module.experts[donor].state_dict()))
            module.gate.weight[destination].copy_(module.gate.weight[donor])
            module.expert_bias[destination].copy_(module.expert_bias[donor])


def _plan_layer(plan: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    layers = plan.get("layers")
    if not isinstance(layers, Mapping) or not isinstance(layers.get(name), Mapping):
        raise ValueError(f"expert plan has no layer {name}")
    return layers[name]


def configure_isolated_optimizer(
    model: MiMoMixModel,
    plan: Mapping[str, Any],
) -> tuple[torch.optim.AdamW, list[Any]]:
    """Freeze the model and mask the monolithic gate tensor to selected rows."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    expert_parameters: list[torch.nn.Parameter] = []
    gate_parameters: list[torch.nn.Parameter] = []
    hooks: list[Any] = []
    for name, module in moe_layers(model):
        revived = [int(value) for value in _plan_layer(plan, name)["revived"]]
        for expert_id in revived:
            for parameter in module.experts[expert_id].parameters():
                parameter.requires_grad_(True)
                expert_parameters.append(parameter)
        module.gate.weight.requires_grad_(True)
        gate_parameters.append(module.gate.weight)
        row_mask = torch.zeros_like(module.gate.weight)
        row_mask[revived] = 1
        hooks.append(module.gate.weight.register_hook(lambda gradient, mask=row_mask: gradient * mask))
        # Never allow the shared model controller to mutate every bias entry.
        module.auto_update_bias = False
        module.update_speed = 0.0
        module.pending_load.zero_()
        module.pending_batches.zero_()
    if not expert_parameters or not gate_parameters:
        raise ValueError("isolated optimizer would have no parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": expert_parameters, "weight_decay": PREREGISTRATION["training"]["expert_weight_decay"]},
            # AdamW decay on the monolithic tensor would mutate masked rows even
            # with zero gradient.  It is therefore exactly zero for gate tensors.
            {"params": gate_parameters, "weight_decay": 0.0},
        ],
        lr=PREREGISTRATION["training"]["peak_learning_rate"],
        betas=tuple(PREREGISTRATION["training"]["betas"]),
        eps=PREREGISTRATION["training"]["epsilon"],
    )
    return optimizer, hooks


def snapshot_persistent_state(model: MiMoMixModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def assert_frozen_state_unchanged(
    model: MiMoMixModel,
    baseline: Mapping[str, torch.Tensor],
    plan: Mapping[str, Any],
) -> None:
    current = model.state_dict()
    layer_map = dict(moe_layers(model))
    allowed_expert_prefixes: list[str] = []
    special: dict[str, tuple[str, list[int]]] = {}
    for name in layer_map:
        revived = [int(value) for value in _plan_layer(plan, name)["revived"]]
        allowed_expert_prefixes.extend(f"{name}.experts.{expert_id}." for expert_id in revived)
        special[f"{name}.gate.weight"] = ("rows", revived)
        special[f"{name}.expert_bias"] = ("entries", revived)
    if set(current) != set(baseline):
        raise RuntimeError("model state keys changed during isolated training")
    for name, tensor in current.items():
        previous = baseline[name].to(device=tensor.device, dtype=tensor.dtype)
        if any(name.startswith(prefix) for prefix in allowed_expert_prefixes):
            continue
        if name in special:
            _, selected = special[name]
            mask = torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device)
            mask[selected] = False
            if not torch.equal(tensor[mask], previous[mask]):
                raise RuntimeError(f"frozen router entries changed in {name}")
            continue
        if not torch.equal(tensor, previous):
            raise RuntimeError(f"frozen state changed in {name}")


def assert_gradient_isolation(model: MiMoMixModel, plan: Mapping[str, Any]) -> None:
    layer_map = dict(moe_layers(model))
    allowed_parameter_ids: set[int] = set()
    for name, module in layer_map.items():
        revived = [int(value) for value in _plan_layer(plan, name)["revived"]]
        allowed_parameter_ids.add(id(module.gate.weight))
        for expert_id in revived:
            allowed_parameter_ids.update(id(parameter) for parameter in module.experts[expert_id].parameters())
        if module.gate.weight.grad is not None:
            mask = torch.ones(module.n_routed, dtype=torch.bool, device=module.gate.weight.grad.device)
            mask[revived] = False
            if torch.count_nonzero(module.gate.weight.grad[mask]).item() != 0:
                raise RuntimeError(f"gradient escaped selected gate rows in {name}")
    for name, parameter in model.named_parameters():
        allowed = id(parameter) in allowed_parameter_ids
        if allowed != parameter.requires_grad:
            raise RuntimeError(f"requires_grad isolation mismatch for {name}")
        if not allowed and parameter.grad is not None:
            raise RuntimeError(f"frozen parameter received a gradient: {name}")


@contextlib.contextmanager
def temporary_selected_routing_boost(
    model: MiMoMixModel,
    plan: Mapping[str, Any],
    amount: float,
) -> Iterator[None]:
    saved: list[tuple[SparseMoEFeedForward, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for name, module in moe_layers(model):
                original = module.expert_bias.detach().clone()
                selected = [int(value) for value in _plan_layer(plan, name)["revived"]]
                module.expert_bias[selected] += float(amount)
                saved.append((module, original))
        yield
    finally:
        with torch.no_grad():
            for module, original in saved:
                module.expert_bias.copy_(original)


@torch.no_grad()
def update_selected_router_biases(
    model: MiMoMixModel,
    plan: Mapping[str, Any],
    speed: float = 1e-3,
) -> None:
    """Balance only revived entries; assert every other buffer entry is exact."""

    for name, module in moe_layers(model):
        selected = [int(value) for value in _plan_layer(plan, name)["revived"]]
        before = module.expert_bias.detach().clone()
        selected_load = module.last_expert_load[selected]
        target = selected_load.mean()
        module.expert_bias[selected] += float(speed) * torch.sign(target - selected_load)
        mask = torch.ones(module.n_routed, dtype=torch.bool, device=module.expert_bias.device)
        mask[selected] = False
        if not torch.equal(module.expert_bias[mask], before[mask]):
            raise RuntimeError(f"router-bias update escaped selected entries in {name}")
        module.pending_load.zero_()
        module.pending_batches.zero_()


def learning_rate_at_step(step: int) -> float:
    training = PREREGISTRATION["training"]
    total = int(training["steps"])
    warmup = int(training["warmup_steps"])
    peak = float(training["peak_learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    if step < 1 or step > total:
        raise ValueError(f"step must be in [1, {total}]")
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total - warmup)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def batch_kind_at_step(step: int) -> str:
    if step < 1:
        raise ValueError("step must be positive")
    return BATCH_PATTERN[(step - 1) % len(BATCH_PATTERN)]


def routing_boost_at_step(step: int) -> float:
    if step < 1:
        raise ValueError("step must be positive")
    initial = float(PREREGISTRATION["training"]["routing_boost"]["initial"])
    last = int(PREREGISTRATION["training"]["routing_boost"]["last_step"])
    if step >= last:
        return 0.0
    return initial * (last - step) / (last - 1)


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _build_exact_tensors(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: mimomix_text.WordTokenizer,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [_pair(row) for row in rows]
    too_long = [
        str(rows[index].get("row_id"))
        for index, (user, assistant) in enumerate(pairs)
        if len(tokenizer.encode_turn(user, assistant)[0]) > sequence_length
    ]
    if too_long:
        raise ValueError(f"{len(too_long)} turns exceed the frozen sequence length: {too_long[:5]}")
    inputs, labels = mimomix_text.build_training_tensors(
        pairs,
        tokenizer,
        sequence_length=sequence_length,
        mask_prompt=True,
        turn_aligned=True,
    )
    if inputs.shape[0] != len(rows):
        raise RuntimeError("turn-aligned packing silently dropped a verified row")
    return inputs, labels


def _sample_batch(
    tensors: tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = tensors
    indices = torch.randint(0, inputs.shape[0], (batch_size,), generator=generator)
    return inputs.index_select(0, indices).to(device), labels.index_select(0, indices).to(device)


def _tensor_digest(name: str, tensor: torch.Tensor, digest: Any) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(canonical_json(list(value.shape)).encode("ascii") + b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())


def persistent_state_sha256(model: MiMoMixModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        _tensor_digest(name, tensor, digest)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


IMMUTABLE_PROVENANCE_KEYS = (
    "parent_checkpoint_sha256",
    "tokenizer_sha256",
    "dataset_manifest_sha256",
    "v71_failure_receipt_sha256",
    "v71_promotion_manifest_sha256",
    "v71_candidate_checkpoint_sha256",
    "preregistration_sha256",
    "source_sha256",
    "runtime",
    "post_revival_pretraining_state_sha256",
)


def validate_immutable_provenance(provenance: Mapping[str, Any], label: str) -> None:
    missing = [key for key in IMMUTABLE_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"{label} provenance omits immutable keys: {missing}")
    hash_keys = (
        "parent_checkpoint_sha256",
        "tokenizer_sha256",
        "dataset_manifest_sha256",
        "v71_failure_receipt_sha256",
        "v71_promotion_manifest_sha256",
        "v71_candidate_checkpoint_sha256",
        "preregistration_sha256",
        "post_revival_pretraining_state_sha256",
    )
    if any(not _is_sha256(provenance.get(key)) for key in hash_keys):
        raise ValueError(f"{label} provenance has an invalid immutable hash")
    sources = provenance.get("source_sha256")
    expected_sources = {
        "train_mosaic_expert_revival.py",
        "build_mosaic_expert_revival_dataset.py",
        "mimomix_core.py",
        "mimomix_text.py",
        "train_mimomix_talk.py",
    }
    if (
        not isinstance(sources, Mapping)
        or set(sources) != expected_sources
        or any(not _is_sha256(value) for value in sources.values())
    ):
        raise ValueError(f"{label} provenance has invalid dependency-source bindings")
    runtime = provenance.get("runtime")
    required_runtime = {
        "python_version",
        "python_build",
        "python_implementation",
        "machine",
        "system",
        "torch_version",
        "torch_cuda_version",
        "cuda_available",
        "requested_device",
        "deterministic_algorithms",
        "torch_num_threads",
        "torch_num_interop_threads",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != required_runtime:
        raise ValueError(f"{label} provenance has an invalid runtime binding")


def validate_recovery_optimizer_state(
    state_dict: Mapping[str, Any],
    *,
    model: MiMoMixModel,
    expert_plan: Mapping[str, Any],
    step: int,
) -> None:
    groups = state_dict.get("param_groups")
    states = state_dict.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(states, Mapping):
        raise ValueError("recovery optimizer must have exactly expert and gate groups")
    training = PREREGISTRATION["training"]
    expected_weight_decay = (float(training["expert_weight_decay"]), 0.0)
    expected_lr = learning_rate_at_step(step)
    expected_betas = tuple(float(value) for value in training["betas"])
    expected_modes = {
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    expected_counts = (
        sum(
            len(list(module.experts[expert_id].parameters()))
            for name, module in moe_layers(model)
            for expert_id in _plan_layer(expert_plan, name)["revived"]
        ),
        len(moe_layers(model)),
    )
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("recovery optimizer parameter group is malformed")
        if len(group["params"]) != expected_counts[index]:
            raise ValueError("recovery optimizer parameter count differs from isolation plan")
        if float(group.get("weight_decay", math.nan)) != expected_weight_decay[index]:
            raise ValueError("recovery optimizer weight decay violates gate-row isolation")
        if tuple(group.get("betas", ())) != expected_betas:
            raise ValueError("recovery optimizer betas differ from preregistration")
        if float(group.get("eps", math.nan)) != float(training["epsilon"]):
            raise ValueError("recovery optimizer epsilon differs from preregistration")
        if float(group.get("lr", math.nan)) != expected_lr:
            raise ValueError("recovery optimizer learning rate differs from saved step schedule")
        for mode_name, expected_mode in expected_modes.items():
            if group.get(mode_name) != expected_mode:
                raise ValueError(
                    f"recovery optimizer {mode_name} differs from the registered AdamW mode"
                )

    gate_parameter_ids = groups[1]["params"]
    for parameter_id, (name, module) in zip(gate_parameter_ids, moe_layers(model)):
        parameter_state = states.get(parameter_id)
        if not isinstance(parameter_state, Mapping):
            raise ValueError(f"recovery optimizer omits gate moments for {name}")
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"recovery optimizer has unexpected gate state for {name}")
        state_step = parameter_state["step"]
        if isinstance(state_step, torch.Tensor):
            if state_step.numel() != 1:
                raise ValueError(f"recovery optimizer has malformed step state for {name}")
            state_step = state_step.detach().cpu().item()
        if isinstance(state_step, bool) or not isinstance(state_step, (int, float)):
            raise ValueError(f"recovery optimizer has malformed step state for {name}")
        if float(state_step) != float(step):
            raise ValueError(f"recovery optimizer step state differs from checkpoint step for {name}")
        selected = [int(value) for value in _plan_layer(expert_plan, name)["revived"]]
        frozen_mask = torch.ones(module.n_routed, dtype=torch.bool)
        frozen_mask[selected] = False
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if not isinstance(moment, torch.Tensor) or tuple(moment.shape) != tuple(module.gate.weight.shape):
                raise ValueError(f"recovery optimizer has malformed {moment_name} for {name}")
            if torch.count_nonzero(moment.detach().cpu()[frozen_mask]).item() != 0:
                raise ValueError(f"recovery optimizer has nonzero frozen-row {moment_name} for {name}")


def save_recoverable_checkpoint(
    path: Path,
    *,
    model: MiMoMixModel,
    tokenizer: mimomix_text.WordTokenizer,
    optimizer: torch.optim.Optimizer,
    step: int,
    generator: torch.Generator,
    expert_plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite recoverable checkpoint: {path}")
    assert_gradient_isolation(model, expert_plan)
    validate_immutable_provenance(provenance, "checkpoint")
    validate_recovery_optimizer_state(
        optimizer.state_dict(), model=model, expert_plan=expert_plan, step=step
    )
    payload = {
        # Keep the established top-level schema so existing evaluation and
        # generation tools can load every recovery point without conversion.
        "schema": TALK_CHECKPOINT_SCHEMA,
        "fallback_schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "preregistration": PREREGISTRATION,
        "preregistration_sha256": preregistration_sha256(),
        "expert_plan": expert_plan,
        "provenance": dict(provenance),
        "config": dict(model.config.__dict__),
        "state_dict": model.state_dict(),
        "tokenizer": tokenizer.to_dict(),
        "extra": {
            "run_name": "mosaic_expert_revival",
            "candidate_only": True,
            "step": int(step),
            "parent_checkpoint_sha256": provenance.get("parent_checkpoint_sha256"),
            "preregistration_sha256": preregistration_sha256(),
            "activation_pointer_written": False,
        },
        "optimiser_state": optimizer.state_dict(),
        "sampler_generator_state": generator.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "model_state_sha256": persistent_state_sha256(model),
        "activation_pointer_written": False,
    }
    atomic_torch_save(path, payload)


def load_recoverable_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    model: MiMoMixModel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    expert_plan: Mapping[str, Any],
    expected_provenance: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Restore model, optimizer, and RNG state after validating every binding."""

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"resume checkpoint hash mismatch: expected {expected_sha256}, got {actual}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != TALK_CHECKPOINT_SCHEMA
        or payload.get("fallback_schema") != CHECKPOINT_SCHEMA
    ):
        raise ValueError("unsupported Mosaic revival checkpoint")
    if payload.get("preregistration_sha256") != preregistration_sha256():
        raise ValueError("resume checkpoint preregistration mismatch")
    if canonical_json(payload.get("expert_plan")) != canonical_json(expert_plan):
        raise ValueError("resume checkpoint expert plan differs from fixed train-only calibration")
    stored_provenance = payload.get("provenance")
    if not isinstance(stored_provenance, Mapping):
        raise ValueError("resume checkpoint omits provenance")
    validate_immutable_provenance(stored_provenance, "stored checkpoint")
    validate_immutable_provenance(expected_provenance, "current run")
    for key in IMMUTABLE_PROVENANCE_KEYS:
        if canonical_json(stored_provenance[key]) != canonical_json(expected_provenance[key]):
            raise ValueError(f"resume checkpoint provenance mismatch for {key}")
    step = payload.get("step")
    checkpoint_every = int(PREREGISTRATION["training"]["checkpoint_every"])
    total_steps = int(PREREGISTRATION["training"]["steps"])
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0 or step > total_steps:
        raise ValueError("resume checkpoint has an invalid step")
    if step % checkpoint_every != 0:
        raise ValueError("resume checkpoint is not on a recoverable checkpoint boundary")
    optimiser_state = payload.get("optimiser_state")
    if not isinstance(optimiser_state, Mapping):
        raise ValueError("resume checkpoint omits optimizer state")
    validate_recovery_optimizer_state(
        optimiser_state, model=model, expert_plan=expert_plan, step=step
    )
    model.load_state_dict(payload["state_dict"])
    if persistent_state_sha256(model) != payload.get("model_state_sha256"):
        raise ValueError("resume checkpoint model-state digest mismatch")
    optimizer.load_state_dict(optimiser_state)
    validate_recovery_optimizer_state(
        optimizer.state_dict(), model=model, expert_plan=expert_plan, step=step
    )
    generator.set_state(payload["sampler_generator_state"])
    torch.random.set_rng_state(payload["torch_rng_state"])
    cuda_state = payload.get("cuda_rng_state_all")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("resume checkpoint requires CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)
    assert_gradient_isolation(model, expert_plan)
    return step, payload


def recoverable_checkpoint_inventory(
    *,
    resume_checkpoint: Path | None,
    expected_resume_checkpoint_sha256: str | None,
    resume_step: int,
    written_checkpoints: Sequence[tuple[int, Path]],
) -> list[dict[str, Any]]:
    """Inventory only the hash-bound input and files created by this call."""

    records: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    seen_paths: set[Path] = set()
    if resume_checkpoint is not None:
        if not _is_sha256(expected_resume_checkpoint_sha256):
            raise ValueError("recoverable inventory requires a bound resume hash")
        resolved = resume_checkpoint.resolve()
        if sha256_file(resolved) != expected_resume_checkpoint_sha256:
            raise ValueError("resume checkpoint changed while fallback training was running")
        records.append(
            {
                "role": "hash_verified_resume_source",
                "path": str(resolved),
                "step": int(resume_step),
                "sha256": expected_resume_checkpoint_sha256,
            }
        )
        seen_steps.add(int(resume_step))
        seen_paths.add(resolved)
    elif expected_resume_checkpoint_sha256 is not None or resume_step != 0:
        raise ValueError("recoverable inventory has resume metadata without a resume checkpoint")
    checkpoint_every = int(PREREGISTRATION["training"]["checkpoint_every"])
    total_steps = int(PREREGISTRATION["training"]["steps"])
    for step, path in written_checkpoints:
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or step > total_steps
            or step % checkpoint_every != 0
        ):
            raise ValueError("written recovery point is not on a registered checkpoint boundary")
        resolved = path.resolve()
        if step in seen_steps or resolved in seen_paths:
            raise ValueError("duplicate recoverable checkpoint lineage entry")
        if not resolved.is_file():
            raise FileNotFoundError(f"written recovery point is missing: {resolved}")
        records.append(
            {
                "role": "written_this_invocation",
                "path": path.name,
                "step": step,
                "sha256": sha256_file(resolved),
            }
        )
        seen_steps.add(step)
        seen_paths.add(resolved)
    return records


def _finite_score(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    numeric = float(value)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return numeric


def _score_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty object")
    return {str(key): _finite_score(score, f"{label}.{key}") for key, score in value.items()}


def _gate_candidate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    minimum_composition_gain: float,
    minimum_math_chain_gain: float,
) -> list[str]:
    reasons: list[str] = []
    if candidate.get("split") != "dev":
        reasons.append("checkpoint selection requires split=dev")
    if candidate.get("evaluation_manifest_sha256") != baseline.get("evaluation_manifest_sha256"):
        reasons.append("evaluation manifest differs from baseline")
    baseline_families = _score_map(baseline.get("original_math_families"), "baseline.original_math_families")
    candidate_families = _score_map(candidate.get("original_math_families"), "candidate.original_math_families")
    if set(candidate_families) != set(baseline_families):
        reasons.append("original math family keys differ from baseline")
    else:
        for family, baseline_score in baseline_families.items():
            if candidate_families[family] < baseline_score:
                reasons.append(f"original math regression: {family}")
    baseline_items = _score_map(baseline.get("legacy_chat_items"), "baseline.legacy_chat_items")
    candidate_items = _score_map(candidate.get("legacy_chat_items"), "candidate.legacy_chat_items")
    if set(candidate_items) != set(baseline_items):
        reasons.append("legacy chat item keys differ from baseline")
    else:
        for item, baseline_score in baseline_items.items():
            if candidate_items[item] < baseline_score:
                reasons.append(f"legacy chat regression: {item}")
    if _finite_score(candidate.get("legacy_chat_score"), "candidate.legacy_chat_score") < _finite_score(
        baseline.get("legacy_chat_score"), "baseline.legacy_chat_score"
    ):
        reasons.append("legacy chat aggregate regression")
    if _finite_score(candidate.get("composition_score"), "candidate.composition_score") < _finite_score(
        baseline.get("composition_score"), "baseline.composition_score"
    ) + minimum_composition_gain:
        reasons.append("composition gain below preregistered minimum")
    if _finite_score(candidate.get("math_chain_score"), "candidate.math_chain_score") < _finite_score(
        baseline.get("math_chain_score"), "baseline.math_chain_score"
    ) + minimum_math_chain_gain:
        reasons.append("math-chain gain below preregistered minimum")
    if not _is_sha256(candidate.get("checkpoint_sha256")):
        reasons.append("candidate lacks a lowercase checkpoint_sha256")
    step = candidate.get("step")
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step <= 0
        or step > int(PREREGISTRATION["training"]["steps"])
        or step % 400 != 0
    ):
        reasons.append("candidate step is not a positive 400-step checkpoint")
    return reasons


def select_fail_closed_checkpoint(
    baseline: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_composition_gain: float = 0.10,
    minimum_math_chain_gain: float = 0.10,
) -> dict[str, Any]:
    """Audit metric claims, but never select from unbound JSON metrics.

    The input mappings are not content-bound evaluator receipts, so even an
    all-gates pass has no selection authority.  The earliest passing step is
    reported only as an audit lead for a future frozen evaluator.
    """

    if baseline.get("split") != "dev":
        raise ValueError("baseline selection metrics must be from dev, never holdout")
    if not _is_sha256(baseline.get("evaluation_manifest_sha256")):
        raise ValueError("baseline lacks a lowercase evaluation_manifest_sha256")
    if minimum_composition_gain != PREREGISTRATION["selection"]["minimum_composition_gain"]:
        raise ValueError("composition gain differs from preregistration")
    if minimum_math_chain_gain != PREREGISTRATION["selection"]["minimum_math_chain_gain"]:
        raise ValueError("math-chain gain differs from preregistration")
    audits: list[dict[str, Any]] = []
    eligible: list[Mapping[str, Any]] = []
    def audit_order(value: Mapping[str, Any]) -> int:
        step = value.get("step")
        return step if isinstance(step, int) and not isinstance(step, bool) else 10**18

    for candidate in sorted(candidates, key=audit_order):
        try:
            reasons = _gate_candidate(
                baseline,
                candidate,
                minimum_composition_gain=minimum_composition_gain,
                minimum_math_chain_gain=minimum_math_chain_gain,
            )
        except (TypeError, ValueError) as exc:
            reasons = [f"invalid metrics: {exc}"]
        audits.append({"step": candidate.get("step"), "eligible": not reasons, "reasons": reasons})
        if not reasons:
            eligible.append(candidate)
    audit_eligible = eligible[0] if eligible else None
    return {
        "schema": SELECTION_SCHEMA,
        "mode": "audit_only_unbound_metrics",
        "decision": "audit_gate_passed_no_selection_authority" if audit_eligible is not None else "no_candidate_passed",
        "audit_eligible_step": audit_eligible.get("step") if audit_eligible is not None else None,
        "audit_eligible_checkpoint_sha256": (
            audit_eligible.get("checkpoint_sha256") if audit_eligible is not None else None
        ),
        "selected_step": None,
        "selected_checkpoint_sha256": None,
        "selection_authorized": False,
        "selection_blocker": "requires a frozen content-bound Mosaic evaluator receipt",
        "dialogue_metric_semantics": "seen-target retention only; not semantic quality or novel conversation",
        "candidate_audits": audits,
        "preregistration_sha256": preregistration_sha256(),
        "activation_pointer_written": False,
        "holdout_used_for_selection": False,
    }


def train_fallback(
    *,
    parent_checkpoint: Path,
    expected_parent_sha256: str,
    expected_tokenizer_sha256: str,
    dataset_dir: Path,
    expected_dataset_manifest_sha256: str,
    v71_failure_receipt: Path,
    expected_v71_failure_receipt_sha256: str,
    v71_promotion_manifest: Path,
    expected_v71_promotion_manifest_sha256: str,
    v71_candidate_checkpoint: Path,
    expected_v71_candidate_checkpoint_sha256: str,
    output_dir: Path,
    device_name: str = "cpu",
    resume_checkpoint: Path | None = None,
    expected_resume_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Run the preregistered fallback.  Callers must opt in through the CLI."""

    resume_path_supplied = resume_checkpoint is not None
    resume_hash_supplied = expected_resume_checkpoint_sha256 is not None
    if resume_path_supplied != resume_hash_supplied:
        raise ValueError("resume checkpoint path and expected SHA-256 must be supplied together")
    if resume_hash_supplied and not _is_sha256(expected_resume_checkpoint_sha256):
        raise ValueError("expected resume checkpoint SHA-256 must be lowercase hexadecimal")

    # The failed frozen gate is checked before loading the model or dataset, so
    # an active/incomplete v71 run cannot accidentally trigger fallback work.
    verify_v71_failure_receipt(
        v71_failure_receipt,
        expected_v71_failure_receipt_sha256,
        manifest_path=v71_promotion_manifest,
        expected_manifest_sha256=expected_v71_promotion_manifest_sha256,
        candidate_checkpoint_path=v71_candidate_checkpoint,
        expected_candidate_checkpoint_sha256=expected_v71_candidate_checkpoint_sha256,
        expected_parent_checkpoint_sha256=expected_parent_sha256,
    )
    configure_deterministic_runtime()
    bundle = load_training_bundle(dataset_dir, expected_dataset_manifest_sha256)
    verify_dataset_external_corpus_binding(
        bundle["manifest"], _read_json_object(v71_promotion_manifest)
    )
    tokenizer_binding = bundle["manifest"]["tokenizer_binding"]
    if tokenizer_binding.get("parent_checkpoint_sha256") != expected_parent_sha256:
        raise ValueError("dataset tokenizer binding uses a different parent checkpoint")
    if tokenizer_binding.get("tokenizer_sha256") != expected_tokenizer_sha256:
        raise ValueError("dataset tokenizer binding uses a different embedded tokenizer")
    model, tokenizer, _ = load_verified_v70(
        parent_checkpoint,
        expected_parent_sha256,
        expected_tokenizer_sha256,
        map_location="cpu",
    )
    all_rows: list[Mapping[str, Any]] = []
    for key in (
        "train_mosaic",
        "train_math",
        "train_dialogue",
    ):
        all_rows.extend(bundle[key])
    coverage = validate_tokenizer_coverage(all_rows, tokenizer)
    device = torch.device(device_name)
    model.to(device)
    dialogue_usage = calibrate_expert_usage(
        model,
        tokenizer,
        fixed_train_calibration_subset(
            bundle["train_dialogue"],
            expected_domain="dialogue",
            limit=PREREGISTRATION["calibration"]["subset_per_domain"],
        ),
        expected_domain="dialogue",
        maximum_length=PREREGISTRATION["training"]["sequence_length"],
    )
    math_usage = calibrate_expert_usage(
        model,
        tokenizer,
        fixed_train_calibration_subset(
            bundle["train_math"],
            expected_domain="math",
            limit=PREREGISTRATION["calibration"]["subset_per_domain"],
        ),
        expected_domain="math",
        maximum_length=PREREGISTRATION["training"]["sequence_length"],
    )
    expert_plan = build_expert_plan(model, dialogue_usage, math_usage)
    revive_from_donors(model, expert_plan)
    baseline_state = snapshot_persistent_state(model)
    baseline_state_hash = persistent_state_sha256(model)
    optimizer, hooks = configure_isolated_optimizer(model, expert_plan)
    training = PREREGISTRATION["training"]
    tensors = {
        "mosaic": _build_exact_tensors(bundle["train_mosaic"], tokenizer, training["sequence_length"]),
        "math": _build_exact_tensors(bundle["train_math"], tokenizer, training["sequence_length"]),
        "dialogue": _build_exact_tensors(bundle["train_dialogue"], tokenizer, training["sequence_length"]),
    }
    generator = torch.Generator().manual_seed(training["seed"])
    provenance: dict[str, Any] = {
        "schema": PROVENANCE_SCHEMA,
        "status": "training_in_progress",
        "parent_checkpoint_sha256": expected_parent_sha256,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "dataset_manifest_sha256": expected_dataset_manifest_sha256,
        "v71_failure_receipt_sha256": expected_v71_failure_receipt_sha256,
        "v71_promotion_manifest_sha256": expected_v71_promotion_manifest_sha256,
        "v71_candidate_checkpoint_sha256": expected_v71_candidate_checkpoint_sha256,
        "preregistration_sha256": preregistration_sha256(),
        "source_sha256": dependency_source_hashes(),
        "runtime": runtime_provenance(device_name),
        "coverage": coverage,
        "calibration": {"dialogue_usage": dialogue_usage, "math_usage": math_usage},
        "expert_plan": expert_plan,
        "post_revival_pretraining_state_sha256": baseline_state_hash,
        "activation_pointer_written": False,
        "selection_status": "not_evaluated",
    }
    start_step = 0
    if resume_checkpoint is not None:
        start_step, _ = load_recoverable_checkpoint(
            resume_checkpoint,
            expected_sha256=expected_resume_checkpoint_sha256,
            model=model,
            optimizer=optimizer,
            generator=generator,
            expert_plan=expert_plan,
            expected_provenance=provenance,
        )
        assert_frozen_state_unchanged(model, baseline_state, expert_plan)
        provenance["resumed_from"] = {
            "path": str(resume_checkpoint.resolve()),
            "sha256": expected_resume_checkpoint_sha256,
            "step": start_step,
        }
    checkpoint_every = int(training["checkpoint_every"])
    planned_checkpoint_paths = [
        output_dir / f"checkpoint_step_{step:05d}.pt"
        for step in range(
            ((start_step // checkpoint_every) + 1) * checkpoint_every,
            int(training["steps"]) + 1,
            checkpoint_every,
        )
    ]
    preexisting_checkpoints = [path for path in planned_checkpoint_paths if path.exists()]
    if preexisting_checkpoints:
        names = [str(path.resolve()) for path in preexisting_checkpoints]
        raise FileExistsError(f"refusing dirty checkpoint targets: {names}")
    written_checkpoints: list[tuple[int, Path]] = []
    _atomic_json(output_dir / "training_provenance.json", provenance)
    model.train()
    try:
        for step in range(start_step + 1, training["steps"] + 1):
            kind = batch_kind_at_step(step)
            inputs, labels = _sample_batch(tensors[kind], training["batch_size"], generator, device)
            _set_optimizer_lr(optimizer, learning_rate_at_step(step))
            optimizer.zero_grad(set_to_none=True)
            boost = routing_boost_at_step(step) if kind == "mosaic" else 0.0
            with temporary_selected_routing_boost(model, expert_plan, boost):
                output = model(inputs, labels=labels, return_mtp=False)
            if output.loss is None or not torch.isfinite(output.loss):
                raise RuntimeError(f"non-finite or missing loss at step {step}")
            output.loss.backward()
            assert_gradient_isolation(model, expert_plan)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                training["gradient_clip"],
            )
            optimizer.step()
            update_selected_router_biases(model, expert_plan, training["selected_bias_update_speed"])
            if step % training["checkpoint_every"] == 0:
                assert_frozen_state_unchanged(model, baseline_state, expert_plan)
                checkpoint_path = output_dir / f"checkpoint_step_{step:05d}.pt"
                save_recoverable_checkpoint(
                    checkpoint_path,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    step=step,
                    generator=generator,
                    expert_plan=expert_plan,
                    provenance=provenance,
                )
                written_checkpoints.append((step, checkpoint_path))
    finally:
        for hook in hooks:
            hook.remove()
    assert_frozen_state_unchanged(model, baseline_state, expert_plan)
    provenance["status"] = "training_complete_awaiting_frozen_dev_receipts"
    provenance["final_model_state_sha256"] = persistent_state_sha256(model)
    provenance["recoverable_checkpoints"] = recoverable_checkpoint_inventory(
        resume_checkpoint=resume_checkpoint,
        expected_resume_checkpoint_sha256=expected_resume_checkpoint_sha256,
        resume_step=start_step,
        written_checkpoints=written_checkpoints,
    )
    _atomic_json(output_dir / "training_provenance.json", provenance)
    return provenance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="run only after a frozen v71 gate rejection")
    train.add_argument("--authorize-preregistered-fallback", action="store_true", required=True)
    train.add_argument("--parent-checkpoint", type=Path, required=True)
    train.add_argument("--expected-parent-sha256", required=True)
    train.add_argument("--expected-tokenizer-sha256", required=True)
    train.add_argument("--dataset-dir", type=Path, required=True)
    train.add_argument("--expected-dataset-manifest-sha256", required=True)
    train.add_argument("--v71-failure-receipt", type=Path, required=True)
    train.add_argument("--expected-v71-failure-receipt-sha256", required=True)
    train.add_argument("--v71-promotion-manifest", type=Path, required=True)
    train.add_argument("--expected-v71-promotion-manifest-sha256", required=True)
    train.add_argument("--v71-candidate-checkpoint", type=Path, required=True)
    train.add_argument("--expected-v71-candidate-checkpoint-sha256", required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cpu")
    train.add_argument("--resume-checkpoint", type=Path)
    train.add_argument("--expected-resume-checkpoint-sha256")
    audit = subparsers.add_parser("audit", help="audit unbound dev metrics without selecting")
    audit.add_argument("--baseline-metrics", type=Path, required=True)
    audit.add_argument("--candidate-metrics", type=Path, action="append", required=True)
    audit.add_argument("--output-receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "train":
        provenance = train_fallback(
            parent_checkpoint=args.parent_checkpoint,
            expected_parent_sha256=args.expected_parent_sha256,
            expected_tokenizer_sha256=args.expected_tokenizer_sha256,
            dataset_dir=args.dataset_dir,
            expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
            v71_failure_receipt=args.v71_failure_receipt,
            expected_v71_failure_receipt_sha256=args.expected_v71_failure_receipt_sha256,
            v71_promotion_manifest=args.v71_promotion_manifest,
            expected_v71_promotion_manifest_sha256=args.expected_v71_promotion_manifest_sha256,
            v71_candidate_checkpoint=args.v71_candidate_checkpoint,
            expected_v71_candidate_checkpoint_sha256=args.expected_v71_candidate_checkpoint_sha256,
            output_dir=args.output_dir,
            device_name=args.device,
            resume_checkpoint=args.resume_checkpoint,
            expected_resume_checkpoint_sha256=args.expected_resume_checkpoint_sha256,
        )
        print(canonical_json({"status": provenance["status"], "output_dir": str(args.output_dir)}))
        return 0
    baseline = _read_json_object(args.baseline_metrics)
    candidates = [_read_json_object(path) for path in args.candidate_metrics]
    receipt = select_fail_closed_checkpoint(baseline, candidates)
    _atomic_json(args.output_receipt, receipt)
    print(canonical_json(receipt))
    return 0 if receipt["decision"] == "audit_gate_passed_no_selection_authority" else 2


if __name__ == "__main__":
    raise SystemExit(main())
