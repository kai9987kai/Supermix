from __future__ import annotations

import copy
import gzip
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import cognitive_leap_three_way_receipt as receipt  # noqa: E402
import run_cognitive_leap_v51_2 as runner  # noqa: E402
from benchmark_cognitive_leap_ultra_v51 import (  # noqa: E402
    make_chained_task_with_metadata,
    operation_family_tags,
)


PASSING_CRITERIA = {
    "minimum_accuracy_gain": 0.0,
    "maximum_p_value": 1.0,
    "minimum_nonregressing_seed_fraction": 0.0,
    "minimum_worst_seed_delta": -1.0,
    "minimum_nonregressing_operation_families": 0,
    "minimum_worst_operation_family_delta": -1.0,
    "minimum_nonregressing_classes": 0,
    "minimum_worst_class_delta": -1.0,
    "require_mean_loss_nonregression": False,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_record(path: Path, root: Path, **extra: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": receipt.sha256_file(path),
        "size_bytes": path.stat().st_size,
        **extra,
    }


def _checkpoint_record(
    path: Path,
    root: Path,
    *,
    state_hash: str,
    status: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state_summary = summary or {
        "tensor_count": 1,
        "element_count": 1,
        "all_finite": True,
        "tensor_byte_order": "little_endian",
        "canonical_state_sha256": state_hash,
    }
    value = {
        **_file_record(path, root),
        **state_summary,
        "strict_load": True,
    }
    if status is not None:
        value["status"] = status
    return value


def _compact_comparison(dataset_hash: str) -> dict[str, Any]:
    checks = {key: True for key in receipt._CHECK_KEYS}
    evidence = {
        "cohort_schema": receipt.COHORT_SCHEMA,
        "generator_schema": receipt.GENERATOR_SCHEMA,
        "family_tag_schema": receipt.FAMILY_TAG_SCHEMA,
        "cohort_role": "development",
        "dataset_id": "0" * 64,
        "dataset_specification_sha256": "1" * 64,
        "dataset_sha256": dataset_hash,
        "baseline_prediction_sha256": "2" * 64,
        "candidate_prediction_sha256": "3" * 64,
        "baseline_logits_sha256": "4" * 64,
        "candidate_logits_sha256": "5" * 64,
        "baseline_per_example_sha256": "6" * 64,
        "candidate_per_example_sha256": "7" * 64,
        "paired_outcome_sha256": "8" * 64,
    }
    return {
        "passed": True,
        "checks": checks,
        "summary": {
            "accuracy_delta": 0.0,
            "mean_candidate_loss": 0.1,
            "nonregressing_seed_count": 1,
            "nonregressing_family_count": 8,
            "nonregressing_class_count": 10,
        },
        "evidence": evidence,
    }


def _test_environment() -> dict[str, Any]:
    return {
        "authentication": "none",
        "timestamps_trusted": False,
        "host_identity_trusted": False,
        "python": {"version": "test"},
        "dependencies": [{"name": "torch", "version": "test"}],
        "dependency_lock_sha256": "1" * 64,
        "critical_distribution_records": {},
        "platform": {"system": "test", "byteorder": "little"},
        "host_binding_sha256": "2" * 64,
        "rng": {"cpu": "bound"},
        "torch": {
            "version": "test",
            "initial_seed": 1,
            "num_threads": 1,
            "device": "cpu",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "default_dtype": "torch.float32",
            "float32_matmul_precision": "highest",
        },
        "invocation": {"argv": ["python", "test"], "working_tree": "."},
    }


def _write_three_way_artifact(
    root: Path,
    *,
    profile: dict[str, Any],
    protocol: dict[str, Any],
    models: dict[str, receipt.ChampionNetCognitiveLeapUltraExpert] | None = None,
) -> dict[str, Any]:
    specification = protocol["final"]["cohort_specification"]
    specification_hash = receipt.sha256_bytes(
        receipt.canonical_json_bytes(specification)
    )
    cohort_digest = hashlib.sha256()
    generated: list[tuple[int, Any, Any, dict[str, torch.Tensor] | None]] = []
    for seed in profile["final"]["seeds"]:
        x, targets, metadata = make_chained_task_with_metadata(
            profile["final"]["samples_per_seed"], seed
        )
        cohort_digest.update(struct.pack("<q", seed))
        receipt.tensor_digest_update(cohort_digest, "x", x)
        receipt.tensor_digest_update(cohort_digest, "y", targets)
        for name in ("starts", "op_types", "operands"):
            receipt.tensor_digest_update(cohort_digest, name, metadata[name])
        model_logits = (
            {
                name: receipt._predict_exact(model, x, name)
                for name, model in models.items()
            }
            if models is not None
            else None
        )
        generated.append((seed, targets, metadata, model_logits))
    dataset_hash = cohort_digest.hexdigest()
    dataset_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "specification_sha256": specification_hash,
                "dataset_sha256": dataset_hash,
            }
        )
    )
    rows: list[dict[str, Any]] = []
    for seed, targets, metadata, model_logits in generated:
        for index in range(int(targets.numel())):
            target = int(targets[index])
            row = {
                "dataset_id": dataset_id,
                "cohort_role": "final",
                "example_id": f"{dataset_id}:{seed}:{index}",
                "seed": seed,
                "index": index,
                "target": target,
                "start": int(metadata["starts"][index]),
                "op_types": [int(value) for value in metadata["op_types"][index]],
                "operands": [int(value) for value in metadata["operands"][index]],
                "operation_family_tags": list(
                    operation_family_tags(metadata["op_types"][index])
                ),
            }
            for model_name in receipt._MODEL_NAMES:
                if model_logits is None:
                    logits = [-2.0] * 10
                    logits[target] = 4.0
                else:
                    logits = [
                        float(value) for value in model_logits[model_name][index]
                    ]
                logits_hex = struct.pack("<10f", *logits).hex()
                prediction = max(range(10), key=lambda offset: logits[offset])
                row[f"{model_name}_logits_f32le_hex"] = logits_hex
                row[f"{model_name}_prediction"] = prediction
                row[f"{model_name}_correct"] = prediction == target
            rows.append(row)
    payload = b"".join(receipt.canonical_json_bytes(row) + b"\n" for row in rows)
    path = root / "three_way_predictions.jsonl.gz"
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle,
            mode="wb",
            filename="",
            mtime=0,
        ) as gzip_handle:
            gzip_handle.write(payload)
    return {
        "schema": receipt.PREDICTION_ARTIFACT_SCHEMA,
        "path": path.relative_to(root).as_posix(),
        "sha256": receipt.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(rows),
        "evaluation_profile_sha256": receipt.sha256_bytes(
            receipt.canonical_json_bytes(profile)
        ),
        "cohort_schema": receipt.COHORT_SCHEMA,
        "generator_schema": receipt.GENERATOR_SCHEMA,
        "family_tag_schema": receipt.FAMILY_TAG_SCHEMA,
        "cohort_role": "final",
        "dataset_id": dataset_id,
        "dataset_specification_sha256": specification_hash,
        "dataset_sha256": dataset_hash,
        "format": "deterministic_gzip_jsonl",
        "logits_encoding": "hex_little_endian_float32",
        "class_count": 10,
        "class_order": list(range(10)),
        "logit_shape": [10],
        "argmax_tie_rule": "lowest_class_index",
        "loss_formula": (
            "torch_cross_entropy_float32_sum_per_seed_then_float64_total"
        ),
        "gzip_mtime": 0,
    }


def _build_synthetic_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    real_checkpoints: bool = False,
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path
    state: dict[str, torch.Tensor] | None = None
    state_summary: dict[str, Any] | None = None
    models: dict[str, receipt.ChampionNetCognitiveLeapUltraExpert] | None = None
    if real_checkpoints:
        with torch.random.fork_rng(devices=[]):
            source_model = receipt.ChampionNetCognitiveLeapUltraExpert().cpu().eval()
        state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in source_model.state_dict().items()
        }
        state_summary = receipt.state_dict_summary(state)
        for name in ("release.pth", "prior.pth", "candidate.pth", "member.pth"):
            torch.save(state, root / name)
        models = {
            name: receipt._strict_model_from_state(state, name)
            for name in receipt._MODEL_NAMES
        }
    else:
        for name, payload in (
            ("release.pth", b"synthetic release"),
            ("prior.pth", b"synthetic prior"),
            ("candidate.pth", b"synthetic candidate"),
            ("member.pth", b"synthetic member"),
        ):
            (root / name).write_bytes(payload)
    release_artifact = _checkpoint_record(
        root / "release.pth",
        root,
        state_hash="3" * 64,
        summary=state_summary,
    )
    prior_artifact = _checkpoint_record(
        root / "prior.pth",
        root,
        state_hash="4" * 64,
        status="unpromoted_prior_candidate",
        summary=state_summary,
    )
    candidate_artifact = _checkpoint_record(
        root / "candidate.pth",
        root,
        state_hash="5" * 64,
        summary=state_summary,
    )
    member_artifact = _checkpoint_record(
        root / "member.pth",
        root,
        state_hash="6" * 64,
        summary=state_summary,
    )

    profile = receipt.canonical_evaluation_profile()
    member_config = {
        "name": "synthetic_member",
        "train_seed": 7,
        "shuffle_seed": 8,
        "dropout_seed": 9,
        "lr": 0.001,
        "balance_exponent": 0.25,
    }
    profile["release_baseline"] = receipt._profile_artifact_projection(
        release_artifact
    )
    profile["prior_candidate"] = receipt._profile_artifact_projection(prior_artifact)
    profile["prior_candidate"]["status"] = "unpromoted_prior_candidate"
    profile["training"] = {
        "train_size_per_member": 4,
        "epochs": 1,
        "batch_size": 2,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "members": [member_config],
    }
    profile["development"] = {
        "seeds": [7_001],
        "samples_per_seed": 2,
        "soup_groups": [["synthetic_member"]],
        "baseline_blend_alphas": [0.5],
        "selection_order": list(receipt.SELECTION_ORDER),
        "release_continuity_criteria": dict(PASSING_CRITERIA),
        "prior_candidate_superiority_criteria": dict(PASSING_CRITERIA),
    }
    profile["final"] = {
        "seeds": [9_001],
        "samples_per_seed": 6,
        "single_use": True,
        "release_continuity_criteria": dict(PASSING_CRITERIA),
        "prior_candidate_superiority_criteria": dict(PASSING_CRITERIA),
        "overall_gate": "logical_and",
    }
    profile_hash = receipt.sha256_bytes(receipt.canonical_json_bytes(profile))
    monkeypatch.setattr(receipt, "_IMMUTABLE_PROFILE", profile)
    monkeypatch.setattr(receipt, "PROFILE_HASH_ALLOWLIST", frozenset({profile_hash}))

    code_bindings: dict[str, Any] = {}
    source_snapshot: dict[str, Any] = {}
    generator_hash = ""
    for relative_name in sorted(receipt._REQUIRED_SOURCE_BINDINGS):
        source_path = Path(__file__).resolve().parent / relative_name
        source_bytes = source_path.read_bytes()
        snapshot_path = root / "snapshot" / relative_name
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_bytes(source_bytes)
        source_hash = hashlib.sha256(source_bytes).hexdigest()
        code_bindings[relative_name] = {
            "sha256": source_hash,
            "size_bytes": len(source_bytes),
            "symbols": ["synthetic_bound_symbol"],
            "worktree_git_blob_sha1": "a" * 40,
            "head_git_blob_sha1": "a" * 40,
        }
        source_snapshot[relative_name] = _file_record(snapshot_path, root)
        if relative_name == "source/benchmark_cognitive_leap_ultra_v51.py":
            generator_hash = source_hash
    environment = _test_environment()
    cohort_specification = {
        "schema": receipt.COHORT_SCHEMA,
        "generator_schema": receipt.GENERATOR_SCHEMA,
        "family_tag_schema": receipt.FAMILY_TAG_SCHEMA,
        "cohort_role": "final",
        "seeds": profile["final"]["seeds"],
        "samples_per_seed": profile["final"]["samples_per_seed"],
        "generator_source_sha256": generator_hash,
    }
    protocol = {
        "schema": receipt.PROTOCOL_SCHEMA,
        "created_at": "untrusted",
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(receipt.AUTHORITY),
        "evaluation_profile": profile,
        "evaluation_profile_sha256": profile_hash,
        "task_schemas": profile["task_schemas"],
        "execution_mode": "clean_final_eligible",
        "finalization_allowed": True,
        "claim_scope": dict(receipt.CLAIM_SCOPE),
        "baseline": release_artifact,
        "prior_candidate": prior_artifact,
        "training": profile["training"],
        "development": {
            **profile["development"],
            "criteria": profile["development"].pop(
                "release_continuity_criteria"
            ),
            "prior_candidate_criteria": profile["development"].pop(
                "prior_candidate_superiority_criteria"
            ),
        },
        "final": {
            "seeds": profile["final"]["seeds"],
            "samples_per_seed": profile["final"]["samples_per_seed"],
            "single_use": True,
            "cohort_specification": cohort_specification,
            "cohort_specification_sha256": receipt.sha256_bytes(
                receipt.canonical_json_bytes(cohort_specification)
            ),
        },
        "criteria": profile["final"]["release_continuity_criteria"],
        "prior_candidate_criteria": profile["final"][
            "prior_candidate_superiority_criteria"
        ],
        "code_bindings": code_bindings,
        "source_snapshot": source_snapshot,
        "git": {"commit": "b" * 40, "dirty": False},
        "environment_at_freeze": environment,
    }
    # The protocol development projection above must not mutate the pinned profile.
    profile["development"]["release_continuity_criteria"] = dict(PASSING_CRITERIA)
    profile["development"]["prior_candidate_superiority_criteria"] = dict(
        PASSING_CRITERIA
    )
    protocol["protocol_sha256"] = receipt._digest_without(
        protocol, "protocol_sha256"
    )
    protocol_path = root / "protocol.json"
    _write_json(protocol_path, protocol)
    protocol_record = {
        "path": protocol_path.relative_to(root).as_posix(),
        "file_sha256": receipt.sha256_file(protocol_path),
        "size_bytes": protocol_path.stat().st_size,
        "content_sha256": protocol["protocol_sha256"],
    }

    compact = _compact_comparison("7" * 64)
    selection_candidate = {
        "name": "synthetic_member__alpha_0.50",
        "members": ["synthetic_member"],
        "member_weights": [1.0],
        "baseline_blend_alpha": 0.5,
        "passed": True,
        "comparisons": {
            "release_continuity": copy.deepcopy(compact),
            "prior_candidate_superiority": copy.deepcopy(compact),
        },
    }
    selection_candidate["selection_score"] = receipt._dual_selection_score(
        selection_candidate["comparisons"]["release_continuity"],
        selection_candidate["comparisons"]["prior_candidate_superiority"],
    )

    training_receipt = {
        "schema": "supermix-cognitive-leap-training-receipt-v2",
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(receipt.AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": profile_hash,
        "parent_baseline": release_artifact,
        "config": member_config,
        "artifact": member_artifact,
    }
    training_receipt["receipt_id"] = receipt._digest_without(
        training_receipt, "receipt_id"
    )
    training_path = root / "training_receipt.json"
    _write_json(training_path, training_receipt)
    training_record = _file_record(training_path, root)
    root_node_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "kind": "checkpoint",
                "artifact_sha256": release_artifact["sha256"],
            }
        )
    )
    member_node_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "kind": "continuation",
                "parent": root_node_id,
                "artifact_sha256": member_artifact["sha256"],
                "config": member_config,
            }
        )
    )
    soup_node_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "kind": "ordered_weighted_soup",
                "parents": [member_node_id],
                "weights": [1.0],
            }
        )
    )
    blend_node_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "kind": "baseline_soup_blend",
                "parents": [root_node_id, soup_node_id],
                "weights": [0.5, 0.5],
            }
        )
    )
    selected_node_id = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "kind": "materialized_checkpoint",
                "parent": blend_node_id,
                "artifact_sha256": candidate_artifact["sha256"],
            }
        )
    )
    nodes = [
        {
            "node_id": root_node_id,
            "kind": "checkpoint",
            "parents": [],
            "artifact": release_artifact,
        },
        {
            "node_id": member_node_id,
            "kind": "continuation",
            "parents": [root_node_id],
            "config": member_config,
            "artifact": member_artifact,
            "training_receipt": training_record,
        },
        {
            "node_id": soup_node_id,
            "kind": "ordered_weighted_soup",
            "parents": [member_node_id],
            "weights": [1.0],
        },
        {
            "node_id": blend_node_id,
            "kind": "baseline_soup_blend",
            "parents": [root_node_id, soup_node_id],
            "weights": [0.5, 0.5],
        },
        {
            "node_id": selected_node_id,
            "kind": "materialized_checkpoint",
            "parents": [blend_node_id],
            "artifact": candidate_artifact,
        },
    ]
    lineage = {
        "schema": receipt.LINEAGE_SCHEMA,
        "authentication": "none",
        "timestamps_trusted": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(receipt.AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": profile_hash,
        "baseline": release_artifact,
        "selected_recipe": {
            "name": selection_candidate["name"],
            "members": selection_candidate["members"],
            "member_weights": selection_candidate["member_weights"],
            "baseline_blend_alpha": selection_candidate[
                "baseline_blend_alpha"
            ],
        },
        "selected_development_evidence_sha256": receipt.sha256_bytes(
            receipt.canonical_json_bytes(selection_candidate)
        ),
        "members": [
            {
                "name": "synthetic_member",
                "config": member_config,
                "artifact": member_artifact,
                "training_receipt": training_record,
            }
        ],
        "nodes": nodes,
        "root_node_id": root_node_id,
        "selected_node_id": selected_node_id,
        "soup": {
            "algorithm": "ordered_float_tensor_weighted_mean_v1",
            "members": ["synthetic_member"],
            "weights": [1.0],
        },
        "baseline_blend": {
            "algorithm": "ordered_float_tensor_weighted_mean_v1",
            "baseline_weight": 0.5,
            "soup_weight": 0.5,
        },
        "selected_artifact": candidate_artifact,
        "reconstruction": {
            "exact_tensor_equality": True,
            "max_absolute_error": 0.0,
            "reconstructed_canonical_state_sha256": candidate_artifact[
                "canonical_state_sha256"
            ],
            "selected_canonical_state_sha256": candidate_artifact[
                "canonical_state_sha256"
            ],
            "strict_load": True,
        },
    }
    lineage_path = root / "lineage_manifest.json"
    _write_json(lineage_path, lineage)
    lineage_record = _file_record(
        lineage_path, root, schema=receipt.LINEAGE_SCHEMA
    )
    lineage_verification = {
        "schema": receipt.LINEAGE_VERIFICATION_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(receipt.AUTHORITY),
        "valid": True,
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_profile_sha256": profile_hash,
        "lineage_manifest": lineage_record,
        "root_node_id": root_node_id,
        "selected_node_id": selected_node_id,
        "selected_canonical_state_sha256": candidate_artifact[
            "canonical_state_sha256"
        ],
        "exact_tensor_reconstruction": True,
    }
    lineage_verification_path = root / "lineage_verification.json"
    _write_json(lineage_verification_path, lineage_verification)
    lineage_verification_record = _file_record(
        lineage_verification_path,
        root,
        schema=receipt.LINEAGE_VERIFICATION_SCHEMA,
    )
    selection = {
        "schema": receipt.SELECTION_SCHEMA,
        "created_at": "untrusted",
        "trusted_timestamp": False,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(receipt.AUTHORITY),
        "protocol_sha256": protocol["protocol_sha256"],
        "decision": "selected_and_frozen_for_single_final",
        "passed": True,
        "development_dataset_sha256": "7" * 64,
        "member_receipts": {"synthetic_member": training_receipt},
        "selected": {**selection_candidate, "artifact": candidate_artifact},
        "lineage_manifest": lineage_record,
        "lineage_verification": lineage_verification_record,
        "candidates": [selection_candidate],
        "environment": environment,
    }
    selection["selection_sha256"] = receipt._digest_without(
        selection, "selection_sha256"
    )
    selection_path = root / "selection.json"
    _write_json(selection_path, selection)
    selection_record = {
        "path": selection_path.relative_to(root).as_posix(),
        "file_sha256": receipt.sha256_file(selection_path),
        "size_bytes": selection_path.stat().st_size,
        "content_sha256": selection["selection_sha256"],
    }

    artifact = _write_three_way_artifact(
        root,
        profile=profile,
        protocol=protocol,
        models=models,
    )
    _artifact_record, replay = receipt._validate_prediction_artifact(
        artifact,
        root=root,
        protocol=protocol,
        profile=profile,
        models=None,
    )
    release_comparison = receipt._compare_models(
        replay["models"]["release_baseline"],
        replay["models"]["candidate"],
        replay,
        profile["final"]["release_continuity_criteria"],
        artifact,
        profile_hash,
    )
    prior_comparison = receipt._compare_models(
        replay["models"]["prior_candidate"],
        replay["models"]["candidate"],
        replay,
        profile["final"]["prior_candidate_superiority_criteria"],
        artifact,
        profile_hash,
    )
    final_invocation_hash = receipt.sha256_bytes(
        receipt.canonical_json_bytes(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "selection_sha256": selection["selection_sha256"],
                "invocation": environment["invocation"],
                "torch": environment["torch"],
            }
        )
    )
    value = {
        "schema": receipt.RECEIPT_SCHEMA,
        "created_at": "untrusted",
        "gate_outcome": "pass",
        "authority": dict(receipt.AUTHORITY),
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "trusted_timestamp": False,
        "claim_scope": dict(receipt.CLAIM_SCOPE),
        "evaluation_profile": profile,
        "evaluation_profile_sha256": profile_hash,
        "protocol": protocol_record,
        "selection": selection_record,
        "artifacts": {
            "release_baseline": release_artifact,
            "prior_candidate": prior_artifact,
            "candidate": candidate_artifact,
        },
        "comparisons": {
            "release_continuity": release_comparison,
            "prior_candidate_superiority": prior_comparison,
        },
        "per_example_artifact": artifact,
        "code_bindings": code_bindings,
        "source_snapshot": source_snapshot,
        "git_at_protocol_freeze": protocol["git"],
        "git_at_finalization": {"commit": "b" * 40, "dirty": False},
        "environment": environment,
        "evaluation_rng": {
            "cpu_state_before_sha256": "c" * 64,
            "cpu_state_after_sha256": "c" * 64,
            "unchanged": True,
        },
        "final_invocation_sha256": final_invocation_hash,
        "single_use_scope": "this_local_output_directory_only",
    }
    value["receipt_id"] = receipt._digest_without(value, "receipt_id")
    receipt_path = root / "receipt.json"
    _write_json(receipt_path, value)
    return receipt_path, value


def test_profile_is_independently_identical_to_runner() -> None:
    assert receipt.canonical_evaluation_profile() == runner.canonical_evaluation_profile()
    assert (
        receipt.CANONICAL_EVALUATION_PROFILE_SHA256
        == runner.canonical_evaluation_profile_sha256()
        == "3a018d1b9cde5d59c0431f0323a46993d71806604753e459200649a024332bbd"
    )
    assert receipt.PROFILE_HASH_ALLOWLIST == frozenset(
        {receipt.CANONICAL_EVALUATION_PROFILE_SHA256}
    )


def test_three_way_protocol_does_not_require_legacy_paired_validator() -> None:
    assert "source/cognitive_leap_three_way_receipt.py" in (
        receipt._REQUIRED_SOURCE_BINDINGS
    )
    assert "source/cognitive_leap_receipt.py" not in receipt._REQUIRED_SOURCE_BINDINGS


def test_strict_json_rejects_duplicate_and_nonfinite_values() -> None:
    with pytest.raises(receipt.ReceiptValidationError, match="Duplicate"):
        receipt.loads_json_strict('{"value":1,"value":2}')
    with pytest.raises(receipt.ReceiptValidationError, match="Non-finite"):
        receipt.loads_json_strict('{"value":NaN}')
    with pytest.raises(receipt.ReceiptValidationError, match="canonical JSON"):
        receipt.canonical_json_bytes({"value": float("inf")})


def test_profile_allowlist_rejects_self_consistent_policy_mutation() -> None:
    profile = receipt.canonical_evaluation_profile()
    profile["final"]["samples_per_seed"] = 1
    mutated_hash = receipt.sha256_bytes(receipt.canonical_json_bytes(profile))
    with pytest.raises(receipt.ReceiptValidationError, match="not allowlisted"):
        receipt._validate_profile(
            {
                "evaluation_profile": profile,
                "evaluation_profile_sha256": mutated_hash,
            }
        )


def test_synthetic_three_way_receipt_recomputes_both_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _value = _build_synthetic_receipt(tmp_path, monkeypatch)
    result = receipt.validate_receipt(
        path,
        root=tmp_path,
        verify_inference=False,
    )
    assert result["valid"] is True
    assert result["gate_outcome"] == "pass"
    assert result["release_continuity_passed"] is True
    assert result["prior_candidate_superiority_passed"] is True
    assert result["checkpoint_inference_replayed"] is False


def test_default_validation_strict_loads_bound_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _value = _build_synthetic_receipt(tmp_path, monkeypatch)
    with pytest.raises(receipt.ReceiptValidationError, match="checkpoint cannot be loaded"):
        receipt.validate_receipt(path, root=tmp_path)


def test_default_validation_replays_exact_checkpoint_logits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _value = _build_synthetic_receipt(
        tmp_path,
        monkeypatch,
        real_checkpoints=True,
    )
    result = receipt.validate_receipt(path, root=tmp_path)
    assert result["valid"] is True
    assert result["checkpoint_inference_replayed"] is True
    assert result["gate_outcome"] == "pass"


def test_top_gate_cannot_claim_pass_if_either_comparison_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, value = _build_synthetic_receipt(tmp_path, monkeypatch)
    value["gate_outcome"] = "reject"
    value["receipt_id"] = receipt._digest_without(value, "receipt_id")
    result = receipt.try_validate_receipt(
        value,
        root=tmp_path,
        verify_inference=False,
    )
    assert result["valid"] is False
    assert "logical AND" in result["error"]


def test_artifact_rejects_three_way_logit_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _path, value = _build_synthetic_receipt(tmp_path, monkeypatch)
    artifact_path = tmp_path / value["per_example_artifact"]["path"]
    compressed = bytearray(artifact_path.read_bytes())
    compressed[-5] ^= 1
    artifact_path.write_bytes(compressed)
    result = receipt.try_validate_receipt(
        value,
        root=tmp_path,
        verify_inference=False,
    )
    assert result["valid"] is False
    assert "size/hash mismatch" in result["error"]


def test_three_way_gate_uses_exact_inclusive_count_thresholds() -> None:
    sample_count = 2_000
    targets = torch.zeros(sample_count, dtype=torch.long)
    baseline_predictions = torch.ones(sample_count, dtype=torch.long)
    baseline_predictions[:1_000] = 0
    candidate_at_boundary = baseline_predictions.clone()
    candidate_at_boundary[990:1_000] = 1
    candidate_just_below = candidate_at_boundary.clone()
    candidate_just_below[989] = 1

    def evaluation(predictions: torch.Tensor, digest: str) -> dict[str, Any]:
        return {
            "mean_loss": 1.0,
            "prediction_sha256": digest * 64,
            "logits_sha256": "a" * 64,
            "per_example_sha256": "b" * 64,
            "seed_rows": [
                {
                    "seed": 7,
                    "targets": targets,
                    "predictions": predictions,
                    "loss_sum": float(sample_count),
                }
            ],
        }

    cohort = {
        "dataset_sha256": "c" * 64,
        "dataset_id": "d" * 64,
        "specification_sha256": "e" * 64,
        "schema": receipt.COHORT_SCHEMA,
        "generator_schema": receipt.GENERATOR_SCHEMA,
        "family_tag_schema": receipt.FAMILY_TAG_SCHEMA,
        "cohort_role": "final",
        "rows": [
            {
                "seed": 7,
                "y": targets,
                "op_types": torch.zeros((sample_count, 4), dtype=torch.long),
            }
        ],
    }
    criteria = {
        **receipt.RELEASE_CRITERIA,
        "minimum_accuracy_gain": -0.005,
        "maximum_p_value": 1.0,
        "minimum_nonregressing_seed_fraction": 0.0,
        "minimum_worst_seed_delta": -0.005,
        "minimum_nonregressing_operation_families": 0,
        "minimum_worst_operation_family_delta": -0.005,
        "minimum_nonregressing_classes": 0,
        "minimum_worst_class_delta": -0.005,
        "require_mean_loss_nonregression": False,
    }
    artifact = {"sha256": "3" * 64, "uncompressed_sha256": "4" * 64}
    baseline = evaluation(baseline_predictions, "f")
    boundary = receipt._compare_models(
        baseline,
        evaluation(candidate_at_boundary, "1"),
        cohort,
        criteria,
        artifact,
        "5" * 64,
    )
    below = receipt._compare_models(
        baseline,
        evaluation(candidate_just_below, "2"),
        cohort,
        criteria,
        artifact,
        "5" * 64,
    )

    assert boundary["summary"]["accuracy_delta"] == -0.005
    assert boundary["checks"]["accuracy_gain"] is True
    assert boundary["checks"]["seed_nonregression"] is True
    assert boundary["checks"]["operation_family_nonregression"] is True
    assert boundary["checks"]["class_bounded_nonregression"] is True
    assert below["summary"]["accuracy_delta"] == -0.0055
    assert below["checks"]["accuracy_gain"] is False
    assert below["checks"]["seed_nonregression"] is False
    assert below["checks"]["operation_family_nonregression"] is False
    assert below["checks"]["class_bounded_nonregression"] is False
