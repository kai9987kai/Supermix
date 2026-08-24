"""Run the predeclared v51.2 round-three head-calibration experiment.

This module is development-only.  It starts from the exact rejected v51.1
candidate, trains two deterministic output-calibration members while every
other parameter remains frozen, and evaluates a fixed soup/blend grid on a
fresh development cohort.  It deliberately has no finalization, packaging,
publication, activation, or routing entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

try:  # Package import in tests; direct import when executed as a script.
    from . import run_cognitive_leap_v51_2 as runner
except ImportError:  # pragma: no cover - exercised by the command-line entry point
    import run_cognitive_leap_v51_2 as runner


SCHEMA = "supermix-cognitive-leap-v51.2-round3-headcal-development-v1"
TRAINING_RECEIPT_SCHEMA = "supermix-cognitive-leap-v51.2-round3-headcal-training-v1"
SELECTION_SCHEMA = "supermix-cognitive-leap-v51.2-round3-headcal-selection-v1"
LINEAGE_SCHEMA = "supermix-cognitive-leap-v51.2-round3-headcal-lineage-v1"
REPLAY_SCHEMA = "supermix-cognitive-leap-v51.2-round3-headcal-replay-v1"

DEV3_SEEDS = tuple(range(61_052, 101_052, 1_000))
SAMPLES_PER_SEED = 2_000
MEMBER_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "name": "headcal_451",
        "train_seed": 451,
        "shuffle_seed": 5_451,
        "rng_seed": 6_451,
    },
    {
        "name": "headcal_551",
        "train_seed": 551,
        "shuffle_seed": 5_551,
        "rng_seed": 6_551,
    },
)
TRAIN_SIZE_PER_MEMBER = 24_000
EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 2.5e-5
GRADIENT_CLIP_NORM = 1.0
BLEND_ALPHAS = (0.25, 0.50, 0.75, 1.00)
TRAINABLE_PARAMETER_NAMES = (
    "layers.10.bias",
    "layers.10.shared_norm.bias",
    "layers.10.decode_head.weight",
    "layers.11.weight",
)
TRAINABLE_PARAMETER_COUNT = 1_310
MODEL_PARAMETER_COUNT = 2_245_715
ROUND_ONE_SELECTED_ALPHA = 0.30
ROUND_TWO_BETAS = (0.25, 0.50, 0.75)
ROUND_TWO_SELECTED_BETA = 0.25
AUTHORITY = dict(runner.AUTHORITY)
CLAIM_SCOPE = dict(runner.CLAIM_SCOPE)


def _content_digest(value: Mapping[str, Any], digest_key: str) -> str:
    payload = dict(value)
    payload.pop(digest_key, None)
    return runner.sha256_bytes(runner.canonical_json_bytes(payload))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_file(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": runner.relative_path(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": runner.sha256_file(resolved),
    }


def _same_bound_file(record: Mapping[str, Any], path: Path) -> bool:
    return dict(record) == _bound_file(path)


def _validate_checkpoint_record(record: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "path",
        "size_bytes",
        "sha256",
        "canonical_state_sha256",
        "tensor_count",
        "element_count",
    }
    if not required.issubset(record):
        raise ValueError("Checkpoint record is incomplete")
    path = runner.resolve_repo_relative(str(record["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size_bytes"])
        or runner.sha256_file(path) != record["sha256"]
    ):
        raise ValueError(f"Checkpoint artifact changed: {path}")
    state = runner.load_state(path)
    summary = runner.state_dict_summary(state)
    for key in ("canonical_state_sha256", "tensor_count", "element_count"):
        if summary[key] != record[key]:
            raise ValueError(f"Checkpoint state binding changed: {key}")
    return {
        "path": str(record["path"]),
        "size_bytes": int(record["size_bytes"]),
        "sha256": str(record["sha256"]),
        "canonical_state_sha256": str(record["canonical_state_sha256"]),
        "tensor_count": int(record["tensor_count"]),
        "element_count": int(record["element_count"]),
    }


def _validate_parents(
    round1_dir: Path,
    round2_dir: Path,
) -> dict[str, Any]:
    round1_dir = round1_dir.resolve()
    round2_dir = round2_dir.resolve()
    protocol_path = round1_dir / "protocol.json"
    selection_path = round1_dir / "selection_receipt.json"
    replay_path = round1_dir / "development_replay_verification.json"
    round2_spec_path = round2_dir / "search_specification.json"
    round2_receipt_path = round2_dir / "round2_development_receipt.json"

    protocol = runner.load_json_strict(protocol_path)
    selection = runner.load_json_strict(selection_path)
    replay = runner.load_json_strict(replay_path)
    round2_spec = runner.load_json_strict(round2_spec_path)
    round2_receipt = runner.load_json_strict(round2_receipt_path)
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, selection, replay, round2_spec, round2_receipt)
    ):
        raise ValueError("Round-three parents must be JSON objects")

    if (
        protocol.get("protocol_sha256") != runner.protocol_digest(protocol)
        or protocol.get("evaluation_profile_sha256")
        != runner.canonical_evaluation_profile_sha256()
        or selection.get("selection_sha256") != runner.selection_digest(selection)
        or selection.get("protocol_sha256") != protocol["protocol_sha256"]
        or selection.get("passed") is not False
        or selection.get("decision") != "no_development_candidate_passed"
        or selection.get("selected", {}).get("baseline_blend_alpha")
        != ROUND_ONE_SELECTED_ALPHA
    ):
        raise ValueError("Round-one rejection evidence is missing or changed")

    replay_payload = dict(replay)
    replay_id = replay_payload.pop("verification_id", None)
    if (
        replay_id != runner.sha256_bytes(runner.canonical_json_bytes(replay_payload))
        or replay.get("passed") is not False
        or replay.get("protocol_sha256") != protocol["protocol_sha256"]
        or replay.get("selection_sha256") != selection["selection_sha256"]
        or replay.get("selection_file_sha256") != runner.sha256_file(selection_path)
    ):
        raise ValueError("Round-one independent replay evidence changed")

    round2_spec_digest = _content_digest(round2_spec, "specification_sha256")
    if (
        round2_spec.get("schema")
        != "supermix-cognitive-leap-v51.2-round2-development-v1"
        or round2_spec.get("specification_sha256") != round2_spec_digest
        or tuple(round2_spec.get("betas", ())) != ROUND_TWO_BETAS
        or tuple(round2_spec.get("seeds", ())) != runner.DEV_SEEDS
        or tuple(round2_spec.get("seeds", ())) == DEV3_SEEDS
        or round2_spec.get("release_continuity_criteria") != runner.DEVELOPMENT_CRITERIA
        or round2_spec.get("prior_candidate_superiority_criteria")
        != runner.PRIOR_CANDIDATE_CRITERIA
    ):
        raise ValueError("Round-two specification changed")
    round2_parent = round2_spec.get("parent", {})
    if (
        not _same_bound_file(round2_parent.get("protocol", {}), protocol_path)
        or round2_parent.get("protocol_content_sha256") != protocol["protocol_sha256"]
        or not _same_bound_file(round2_parent.get("selection", {}), selection_path)
        or round2_parent.get("selection_content_sha256")
        != selection["selection_sha256"]
        or not _same_bound_file(
            round2_parent.get("development_replay", {}), replay_path
        )
    ):
        raise ValueError("Round-two parent bindings changed")

    round2_receipt_payload = dict(round2_receipt)
    round2_receipt_id = round2_receipt_payload.pop("receipt_id", None)
    if (
        round2_receipt_id
        != runner.sha256_bytes(runner.canonical_json_bytes(round2_receipt_payload))
        or round2_receipt.get("passed") is not False
        or round2_receipt.get("decision") != "no_development_candidate_passed"
        or round2_receipt.get("selected", {}).get("beta") != ROUND_TWO_SELECTED_BETA
        or round2_receipt.get("specification_content_sha256")
        != round2_spec["specification_sha256"]
        or not _same_bound_file(
            round2_receipt.get("specification", {}), round2_spec_path
        )
    ):
        raise ValueError("Round-two rejection receipt changed")

    baseline = _validate_checkpoint_record(protocol.get("baseline", {}))
    prior = _validate_checkpoint_record(protocol.get("prior_candidate", {}))
    if (
        baseline["sha256"] != runner.CANONICAL_BASELINE_FILE_SHA256
        or baseline["canonical_state_sha256"] != runner.CANONICAL_BASELINE_STATE_SHA256
        or prior["sha256"] != runner.PRIOR_CANDIDATE_FILE_SHA256
        or prior["canonical_state_sha256"] != runner.PRIOR_CANDIDATE_STATE_SHA256
    ):
        raise ValueError("Canonical comparator identity changed")

    member_artifacts: dict[str, Any] = {}
    receipts = selection.get("member_receipts", {})
    expected_members = {str(value["name"]) for value in runner.MEMBER_CONFIGS}
    if not isinstance(receipts, Mapping) or set(receipts) != expected_members:
        raise ValueError("Round-one member receipt set changed")
    for name in sorted(receipts):
        receipt = receipts[name]
        if not isinstance(receipt, Mapping):
            raise ValueError("Round-one member receipt is not an object")
        member_artifacts[name] = _validate_checkpoint_record(
            receipt.get("artifact", {})
        )

    return {
        "round1": {
            "protocol": _bound_file(protocol_path),
            "protocol_content_sha256": protocol["protocol_sha256"],
            "selection": _bound_file(selection_path),
            "selection_content_sha256": selection["selection_sha256"],
            "development_replay": _bound_file(replay_path),
            "development_replay_id": replay_id,
        },
        "round2": {
            "specification": _bound_file(round2_spec_path),
            "specification_content_sha256": round2_spec["specification_sha256"],
            "receipt": _bound_file(round2_receipt_path),
            "receipt_id": round2_receipt_id,
        },
        "release_baseline": baseline,
        "prior_candidate": prior,
        "round1_member_artifacts": member_artifacts,
    }


def _specification_digest(specification: Mapping[str, Any]) -> str:
    return _content_digest(specification, "specification_sha256")


def build_specification(
    round1_dir: Path,
    round2_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    if device.type != "cpu":
        raise ValueError("Round-three head calibration is CPU-only")
    if len(DEV3_SEEDS) != 40 or set(DEV3_SEEDS) & set(runner.FINAL_SEEDS):
        raise RuntimeError("Round-three development seeds overlap final seeds")
    if set(DEV3_SEEDS) & set(runner.DEV_SEEDS):
        raise RuntimeError("Round-three development seeds are not fresh")
    if any(AUTHORITY.values()) or any(
        value is not False for key, value in CLAIM_SCOPE.items() if key != "task"
    ):
        raise RuntimeError("Development authority or claims are enabled")
    parents = _validate_parents(round1_dir, round2_dir)
    specification: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": runner.utc_now(),
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "claim_scope": dict(CLAIM_SCOPE),
        "execution_mode": "development_only_no_finalization",
        "final_cohort_access": False,
        "parents": parents,
        "training": {
            "initial_checkpoint_role": "prior_candidate",
            "train_size_per_member": TRAIN_SIZE_PER_MEMBER,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "model_mode": "eval",
            "reasoning_cycles": 3,
            "objective": {
                "name": "unweighted_cross_entropy",
                "class_weights": None,
                "auxiliary_loss": False,
                "distillation": False,
            },
            "optimizer": {
                "name": "AdamW",
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
            },
            "trainable_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "model_parameter_count": MODEL_PARAMETER_COUNT,
            "members": [dict(value) for value in MEMBER_CONFIGS],
            "member_soup_weights": [0.5, 0.5],
        },
        "development": {
            "cohort_round": 3,
            "fresh_after_rounds": [1, 2],
            "seeds": list(DEV3_SEEDS),
            "samples_per_seed": SAMPLES_PER_SEED,
            "blend_alphas": list(BLEND_ALPHAS),
            "selection_order": list(runner.SELECTION_ORDER),
            "release_continuity_criteria": dict(runner.DEVELOPMENT_CRITERIA),
            "prior_candidate_superiority_criteria": dict(
                runner.PRIOR_CANDIDATE_CRITERIA
            ),
            "overall_gate": "logical_and",
        },
        "code": {
            "round3_sha256": runner.sha256_file(Path(__file__).resolve()),
            "runner_sha256": runner.sha256_file(Path(runner.__file__).resolve()),
            "generator_sha256": runner.sha256_file(
                runner.SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
            ),
            "model_variants_sha256": runner.sha256_file(
                runner.SOURCE_DIR / "model_variants.py"
            ),
        },
        "environment_at_freeze": runner.environment_binding(device),
    }
    specification["specification_sha256"] = _specification_digest(specification)
    return specification


def _validate_specification(
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    path = output_dir / "search_specification.json"
    specification = runner.load_json_strict(path)
    if not isinstance(specification, Mapping):
        raise ValueError("Round-three specification is not an object")
    if set(specification) != {
        "schema",
        "created_at",
        "authentication",
        "trusted_timestamp",
        "integrity_status",
        "authority",
        "claim_scope",
        "execution_mode",
        "final_cohort_access",
        "parents",
        "training",
        "development",
        "code",
        "environment_at_freeze",
        "specification_sha256",
    }:
        raise ValueError("Round-three specification fields changed")
    if (
        specification.get("schema") != SCHEMA
        or specification.get("specification_sha256")
        != _specification_digest(specification)
        or specification.get("authentication") != "none"
        or specification.get("trusted_timestamp") is not False
        or specification.get("integrity_status") != "content_bound_not_authenticated"
        or specification.get("authority") != AUTHORITY
        or specification.get("claim_scope") != CLAIM_SCOPE
        or specification.get("execution_mode") != "development_only_no_finalization"
        or specification.get("final_cohort_access") is not False
    ):
        raise ValueError("Round-three specification contract changed")
    training = specification.get("training", {})
    development = specification.get("development", {})
    if (
        training.get("train_size_per_member") != TRAIN_SIZE_PER_MEMBER
        or training.get("epochs") != EPOCHS
        or training.get("batch_size") != BATCH_SIZE
        or training.get("model_mode") != "eval"
        or training.get("reasoning_cycles") != 3
        or training.get("objective")
        != {
            "name": "unweighted_cross_entropy",
            "class_weights": None,
            "auxiliary_loss": False,
            "distillation": False,
        }
        or training.get("trainable_parameter_names") != list(TRAINABLE_PARAMETER_NAMES)
        or training.get("trainable_parameter_count") != TRAINABLE_PARAMETER_COUNT
        or training.get("model_parameter_count") != MODEL_PARAMETER_COUNT
        or training.get("members") != [dict(value) for value in MEMBER_CONFIGS]
        or training.get("member_soup_weights") != [0.5, 0.5]
        or development.get("cohort_round") != 3
        or development.get("fresh_after_rounds") != [1, 2]
        or tuple(development.get("seeds", ())) != DEV3_SEEDS
        or set(development.get("seeds", ())) & set(runner.FINAL_SEEDS)
        or development.get("samples_per_seed") != SAMPLES_PER_SEED
        or tuple(development.get("blend_alphas", ())) != BLEND_ALPHAS
        or development.get("selection_order") != list(runner.SELECTION_ORDER)
        or development.get("release_continuity_criteria") != runner.DEVELOPMENT_CRITERIA
        or development.get("prior_candidate_superiority_criteria")
        != runner.PRIOR_CANDIDATE_CRITERIA
        or development.get("overall_gate") != "logical_and"
    ):
        raise ValueError("Round-three frozen search changed")
    optimizer = training.get("optimizer", {})
    if optimizer != {
        "name": "AdamW",
        "lr": LEARNING_RATE,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "gradient_clip_norm": GRADIENT_CLIP_NORM,
    }:
        raise ValueError("Round-three optimizer changed")
    parents = specification.get("parents", {})
    try:
        round1_dir = runner.resolve_repo_relative(
            str(parents["round1"]["protocol"]["path"])
        ).parent
        round2_dir = runner.resolve_repo_relative(
            str(parents["round2"]["specification"]["path"])
        ).parent
    except (KeyError, TypeError) as exc:
        raise ValueError("Round-three parent paths are missing") from exc
    if _validate_parents(round1_dir, round2_dir) != parents:
        raise ValueError("Round-three exact parent closure changed")
    code = specification.get("code", {})
    if code != {
        "round3_sha256": runner.sha256_file(Path(__file__).resolve()),
        "runner_sha256": runner.sha256_file(Path(runner.__file__).resolve()),
        "generator_sha256": runner.sha256_file(
            runner.SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
        ),
        "model_variants_sha256": runner.sha256_file(
            runner.SOURCE_DIR / "model_variants.py"
        ),
    }:
        raise ValueError("Round-three bound source changed")
    if device.type != "cpu":
        raise ValueError("Round-three verification is CPU-only")
    runner.validate_environment_compatibility(
        specification["environment_at_freeze"], runner.environment_binding(device)
    )
    return dict(specification)


def _training_dataset(
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    seed = int(config["train_seed"])
    forbidden = set(DEV3_SEEDS) | set(runner.DEV_SEEDS) | set(runner.FINAL_SEEDS)
    if seed in forbidden:
        raise RuntimeError("Training seed overlaps an evaluation seed")
    x_train, y_train, metadata = runner.make_chained_task_with_metadata(
        TRAIN_SIZE_PER_MEMBER, seed
    )
    digest = hashlib.sha256()
    runner.tensor_digest_update(digest, "x", x_train)
    runner.tensor_digest_update(digest, "y", y_train)
    for name in ("starts", "op_types", "operands"):
        runner.tensor_digest_update(digest, name, metadata[name])
    dataset_specification = {
        "schema": runner.COHORT_SCHEMA,
        "generator_schema": runner.GENERATOR_SCHEMA,
        "family_tag_schema": runner.FAMILY_TAG_SCHEMA,
        "cohort_role": "round3_head_calibration_training",
        "member": str(config["name"]),
        "seed": seed,
        "n": TRAIN_SIZE_PER_MEMBER,
        "generator_source_sha256": runner.sha256_file(
            runner.SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
        ),
    }
    specification_sha256 = runner.sha256_bytes(
        runner.canonical_json_bytes(dataset_specification)
    )
    dataset_sha256 = digest.hexdigest()
    return (
        x_train,
        y_train,
        {
            "specification": dataset_specification,
            "specification_sha256": specification_sha256,
            "dataset_sha256": dataset_sha256,
            "dataset_id": runner.sha256_bytes(
                runner.canonical_json_bytes(
                    {
                        "specification_sha256": specification_sha256,
                        "dataset_sha256": dataset_sha256,
                    }
                )
            ),
        },
    )


def _configure_head_only_model(
    prior_state: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[
    runner.ChampionNetCognitiveLeapUltraExpert,
    list[torch.nn.Parameter],
]:
    if device.type != "cpu":
        raise ValueError("Head calibration is CPU-only")
    model = runner.ChampionNetCognitiveLeapUltraExpert().to(device)
    model.load_state_dict(dict(prior_state), strict=True)
    named = dict(model.named_parameters())
    if set(TRAINABLE_PARAMETER_NAMES) - set(named):
        raise ValueError("A frozen head-calibration parameter is missing")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable: list[torch.nn.Parameter] = []
    for name in TRAINABLE_PARAMETER_NAMES:
        named[name].requires_grad_(True)
        trainable.append(named[name])
    actual_names = tuple(
        name for name, value in model.named_parameters() if value.requires_grad
    )
    actual_count = sum(value.numel() for value in trainable)
    total_count = sum(value.numel() for value in model.parameters())
    if (
        actual_names != TRAINABLE_PARAMETER_NAMES
        or actual_count != TRAINABLE_PARAMETER_COUNT
        or total_count != MODEL_PARAMETER_COUNT
    ):
        raise ValueError("Head-only trainable parameter contract changed")
    model.eval()
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("Head calibration must run with dropout disabled")
    return model, trainable


def _frozen_head_only_state(
    model: runner.ChampionNetCognitiveLeapUltraExpert,
    prior_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    trained = model.state_dict()
    parameter_names = {name for name, _value in model.named_parameters()}
    for name in parameter_names - set(TRAINABLE_PARAMETER_NAMES):
        if not torch.equal(trained[name].detach().cpu(), prior_state[name]):
            raise ValueError(f"Frozen parameter changed: {name}")
    result = {name: value.detach().cpu().clone() for name, value in prior_state.items()}
    for name in TRAINABLE_PARAMETER_NAMES:
        result[name] = trained[name].detach().cpu().clone()
    for name in result:
        if name not in TRAINABLE_PARAMETER_NAMES and not torch.equal(
            result[name], prior_state[name]
        ):
            raise ValueError(f"Frozen state changed: {name}")
    return result


def train_headcal_member(
    prior_state: Mapping[str, torch.Tensor],
    prior_binding: Mapping[str, Any],
    config: Mapping[str, Any],
    specification: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    specification_path = output_dir / "search_specification.json"
    if not specification_path.is_file() or runner.canonical_json_bytes(
        runner.load_json_strict(specification_path)
    ) != runner.canonical_json_bytes(specification):
        raise ValueError("Frozen round-three specification is not persisted")
    x_train, y_train, dataset = _training_dataset(config)
    model, trainable = _configure_head_only_model(prior_state, device)
    torch.manual_seed(int(config["rng_seed"]))
    shuffle_generator = torch.Generator(device="cpu").manual_seed(
        int(config["shuffle_seed"])
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )
    rng_before = runner.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    shuffle_before = runner.sha256_bytes(
        shuffle_generator.get_state().numpy().tobytes()
    )
    permutation_digest = hashlib.sha256()
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        epoch_started = time.perf_counter()
        permutation = torch.randperm(x_train.shape[0], generator=shuffle_generator)
        runner.tensor_digest_update(
            permutation_digest, f"epoch_{epoch}_permutation", permutation
        )
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for offset in range(0, int(x_train.shape[0]), BATCH_SIZE):
            indices = permutation[offset : offset + BATCH_SIZE]
            xb = x_train[indices].to(device)
            yb = y_train[indices].to(device)
            logits = model(xb, reasoning_cycles=3).squeeze(1)
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP_NORM)
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
            f"round3 member={config['name']} epoch={epoch}/{EPOCHS} "
            f"loss={row['loss']:.5f} accuracy={row['accuracy']:.5f} "
            f"seconds={row['seconds']:.1f}",
            flush=True,
        )
    rng_after = runner.sha256_bytes(torch.get_rng_state().numpy().tobytes())
    shuffle_after = runner.sha256_bytes(shuffle_generator.get_state().numpy().tobytes())
    train_seconds = time.perf_counter() - started
    state = _frozen_head_only_state(model, prior_state)
    member_dir = output_dir / "members" / str(config["name"])
    artifact = runner.save_state(member_dir / "weights.pth", state)
    receipt: dict[str, Any] = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "specification_sha256": specification["specification_sha256"],
        "parent_prior_candidate": dict(prior_binding),
        "config": dict(config),
        "dataset": dataset,
        "model_mode": "eval",
        "reasoning_cycles": 3,
        "objective": dict(specification["training"]["objective"]),
        "trainable_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
        "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
        "model_parameter_count": MODEL_PARAMETER_COUNT,
        "optimizer": dict(specification["training"]["optimizer"]),
        "rng": {
            "torch_before_sha256": rng_before,
            "torch_after_sha256": rng_after,
            "shuffle_before_sha256": shuffle_before,
            "shuffle_after_sha256": shuffle_after,
            "permutation_sha256": permutation_digest.hexdigest(),
        },
        "history": history,
        "train_seconds": train_seconds,
        "frozen_state_exact": True,
        "artifact": artifact,
    }
    receipt["receipt_id"] = _content_digest(receipt, "receipt_id")
    runner.write_json_atomic(member_dir / "training_receipt.json", receipt)
    return state, receipt


def build_dev3_cohort() -> dict[str, Any]:
    if set(DEV3_SEEDS) & set(runner.FINAL_SEEDS):
        raise RuntimeError("Final seeds are prohibited in round three")
    return runner.build_cohort(
        DEV3_SEEDS,
        SAMPLES_PER_SEED,
        cohort_role="development",
    )


def _candidate_states(
    prior_state: Mapping[str, torch.Tensor],
    member_states: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[float, dict[str, torch.Tensor]]:
    expected_names = tuple(str(value["name"]) for value in MEMBER_CONFIGS)
    if set(member_states) != set(expected_names):
        raise ValueError("Round-three member state set changed")
    soup = runner.average_states(
        [member_states[name] for name in expected_names],
        [0.5, 0.5],
    )
    return {
        alpha: runner.average_states(
            [prior_state, soup],
            [1.0 - alpha, alpha],
        )
        for alpha in BLEND_ALPHAS
    }


def _evaluate_candidates(
    specification: Mapping[str, Any],
    baseline_state: Mapping[str, torch.Tensor],
    prior_state: Mapping[str, torch.Tensor],
    candidate_states: Mapping[float, Mapping[str, torch.Tensor]],
    cohort: Mapping[str, Any],
    device: torch.device,
) -> list[dict[str, Any]]:
    evaluator = runner.ChampionNetCognitiveLeapUltraExpert().to(device)
    evaluator.load_state_dict(dict(baseline_state), strict=True)
    baseline_predictions = runner.predict_cohort(evaluator, cohort, device)
    evaluator.load_state_dict(dict(prior_state), strict=True)
    prior_predictions = runner.predict_cohort(evaluator, cohort, device)
    member_names = [str(value["name"]) for value in MEMBER_CONFIGS]
    rows: list[dict[str, Any]] = []
    for alpha in BLEND_ALPHAS:
        state = candidate_states[alpha]
        evaluator.load_state_dict(dict(state), strict=True)
        candidate_predictions = runner.predict_cohort(evaluator, cohort, device)
        release = runner.compare_predictions(
            baseline_predictions,
            candidate_predictions,
            cohort,
            specification["development"]["release_continuity_criteria"],
        )
        prior = runner.compare_predictions(
            prior_predictions,
            candidate_predictions,
            cohort,
            specification["development"]["prior_candidate_superiority_criteria"],
        )
        row = runner.dual_candidate_row(
            name=f"{'-'.join(member_names)}__alpha_{alpha:.2f}",
            group=member_names,
            alpha=alpha,
            release_comparison=release,
            prior_comparison=prior,
        )
        row["canonical_state_sha256"] = runner.state_dict_summary(state)[
            "canonical_state_sha256"
        ]
        rows.append(row)
        print(
            f"round3 alpha={alpha:.2f} pass={row['passed']} "
            f"vs_v51={release['summary']['accuracy_delta']:+.5f} "
            f"vs_v51.1={prior['summary']['accuracy_delta']:+.5f} "
            f"checks={sum(release['checks'].values()) + sum(prior['checks'].values())}/12",
            flush=True,
        )
    return rows


def _write_lineage(
    output_dir: Path,
    specification: Mapping[str, Any],
    member_receipts: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Any],
    selected_artifact: Mapping[str, Any],
    reconstructed_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "specification_sha256": specification["specification_sha256"],
        "method": "prior_started_equal_member_soup_then_linear_interpolation",
        "root_prior_candidate": dict(specification["parents"]["prior_candidate"]),
        "members": {
            name: {
                "receipt_id": receipt["receipt_id"],
                "artifact": dict(receipt["artifact"]),
            }
            for name, receipt in sorted(member_receipts.items())
        },
        "member_weights": [0.5, 0.5],
        "selected_alpha": float(selected["baseline_blend_alpha"]),
        "selected_canonical_state_sha256": runner.state_dict_summary(
            reconstructed_state
        )["canonical_state_sha256"],
        "selected_artifact": dict(selected_artifact),
        "exact_tensor_reconstruction": True,
    }
    manifest["lineage_id"] = _content_digest(manifest, "lineage_id")
    path = output_dir / "selected" / "lineage_manifest.json"
    runner.write_json_exclusive(path, manifest)
    return {
        **_bound_file(path),
        "schema": LINEAGE_SCHEMA,
        "lineage_id": manifest["lineage_id"],
    }


def run_search(
    round1_dir: Path,
    round2_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Round-three output already exists: {output_dir}")
    specification = build_specification(round1_dir, round2_dir, device)
    output_dir.mkdir(parents=True)
    specification_path = output_dir / "search_specification.json"
    runner.write_json_exclusive(specification_path, specification)

    # No training or development generation may occur before the exclusive
    # specification write above.
    specification = _validate_specification(output_dir, device)
    baseline_state = runner.load_state(
        runner.resolve_repo_relative(
            specification["parents"]["release_baseline"]["path"]
        )
    )
    prior_state = runner.load_state(
        runner.resolve_repo_relative(
            specification["parents"]["prior_candidate"]["path"]
        )
    )
    member_states: dict[str, dict[str, torch.Tensor]] = {}
    member_receipts: dict[str, dict[str, Any]] = {}
    for config in MEMBER_CONFIGS:
        state, receipt = train_headcal_member(
            prior_state,
            specification["parents"]["prior_candidate"],
            config,
            specification,
            output_dir,
            device,
        )
        member_states[str(config["name"])] = state
        member_receipts[str(config["name"])] = receipt

    candidate_states = _candidate_states(prior_state, member_states)
    cohort = build_dev3_cohort()
    rows = _evaluate_candidates(
        specification,
        baseline_state,
        prior_state,
        candidate_states,
        cohort,
        device,
    )
    selected = max(rows, key=lambda row: tuple(row["selection_score"]))
    receipt: dict[str, Any] = {
        "schema": SELECTION_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "claim_scope": dict(CLAIM_SCOPE),
        "specification": _bound_file(specification_path),
        "specification_content_sha256": specification["specification_sha256"],
        "development_dataset_sha256": cohort["dataset_sha256"],
        "member_receipts": member_receipts,
        "candidates": rows,
        "selected": selected,
        "passed": bool(selected["passed"]),
        "decision": (
            "development_candidate_found"
            if selected["passed"]
            else "no_development_candidate_passed"
        ),
        "environment": runner.environment_binding(device),
    }
    if selected["passed"]:
        alpha = float(selected["baseline_blend_alpha"])
        selected_state = candidate_states[alpha]
        selected_path = output_dir / "selected" / "cognitive_leap_ultra_v51_2.pth"
        selected_artifact = runner.save_state(selected_path, selected_state)
        strict_selected = runner.load_state(selected_path)
        if any(
            not torch.equal(strict_selected[name], selected_state[name])
            for name in strict_selected
        ):
            raise ValueError("Selected head-calibration checkpoint changed on save")
        lineage = _write_lineage(
            output_dir,
            specification,
            member_receipts,
            selected,
            selected_artifact,
            selected_state,
        )
        receipt["selected"] = {**selected, "artifact": selected_artifact}
        receipt["lineage"] = lineage
    receipt["receipt_id"] = _content_digest(receipt, "receipt_id")
    receipt_path = output_dir / "round3_development_receipt.json"
    runner.write_json_atomic(receipt_path, receipt)
    return receipt_path


def _load_member_evidence(
    output_dir: Path,
    specification: Mapping[str, Any],
    selection: Mapping[str, Any],
    prior_state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, torch.Tensor]]:
    receipts = selection.get("member_receipts")
    expected = {str(value["name"]): value for value in MEMBER_CONFIGS}
    if not isinstance(receipts, Mapping) or set(receipts) != set(expected):
        raise ValueError("Round-three member receipt set changed")
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, config in expected.items():
        receipt = receipts[name]
        if not isinstance(receipt, Mapping):
            raise ValueError("Round-three member receipt is not an object")
        if set(receipt) != {
            "schema",
            "authentication",
            "trusted_timestamp",
            "integrity_status",
            "authority",
            "specification_sha256",
            "parent_prior_candidate",
            "config",
            "dataset",
            "model_mode",
            "reasoning_cycles",
            "objective",
            "trainable_parameter_names",
            "trainable_parameter_count",
            "model_parameter_count",
            "optimizer",
            "rng",
            "history",
            "train_seconds",
            "frozen_state_exact",
            "artifact",
            "receipt_id",
        }:
            raise ValueError(f"Round-three member receipt fields changed: {name}")
        if (
            receipt.get("schema") != TRAINING_RECEIPT_SCHEMA
            or receipt.get("receipt_id") != _content_digest(receipt, "receipt_id")
            or receipt.get("authentication") != "none"
            or receipt.get("trusted_timestamp") is not False
            or receipt.get("integrity_status") != "content_bound_not_authenticated"
            or receipt.get("authority") != AUTHORITY
            or receipt.get("specification_sha256")
            != specification["specification_sha256"]
            or receipt.get("parent_prior_candidate")
            != specification["parents"]["prior_candidate"]
            or receipt.get("config") != config
            or receipt.get("model_mode") != "eval"
            or receipt.get("reasoning_cycles") != 3
            or receipt.get("objective") != specification["training"]["objective"]
            or receipt.get("trainable_parameter_names")
            != list(TRAINABLE_PARAMETER_NAMES)
            or receipt.get("trainable_parameter_count") != TRAINABLE_PARAMETER_COUNT
            or receipt.get("model_parameter_count") != MODEL_PARAMETER_COUNT
            or receipt.get("optimizer") != specification["training"]["optimizer"]
            or receipt.get("frozen_state_exact") is not True
        ):
            raise ValueError(f"Round-three member receipt changed: {name}")
        dataset = receipt.get("dataset", {})
        dataset_specification = dataset.get("specification", {})
        expected_dataset_specification = {
            "schema": runner.COHORT_SCHEMA,
            "generator_schema": runner.GENERATOR_SCHEMA,
            "family_tag_schema": runner.FAMILY_TAG_SCHEMA,
            "cohort_role": "round3_head_calibration_training",
            "member": name,
            "seed": int(config["train_seed"]),
            "n": TRAIN_SIZE_PER_MEMBER,
            "generator_source_sha256": runner.sha256_file(
                runner.SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
            ),
        }
        if (
            set(dataset)
            != {
                "specification",
                "specification_sha256",
                "dataset_sha256",
                "dataset_id",
            }
            or dataset_specification != expected_dataset_specification
            or dataset.get("specification_sha256")
            != runner.sha256_bytes(runner.canonical_json_bytes(dataset_specification))
            or dataset.get("dataset_id")
            != runner.sha256_bytes(
                runner.canonical_json_bytes(
                    {
                        "specification_sha256": dataset.get("specification_sha256"),
                        "dataset_sha256": dataset.get("dataset_sha256"),
                    }
                )
            )
            or not _is_sha256(dataset.get("dataset_sha256"))
        ):
            raise ValueError(f"Round-three member dataset binding changed: {name}")
        rng = receipt.get("rng", {})
        history = receipt.get("history")
        if (
            set(rng)
            != {
                "torch_before_sha256",
                "torch_after_sha256",
                "shuffle_before_sha256",
                "shuffle_after_sha256",
                "permutation_sha256",
            }
            or not all(_is_sha256(value) for value in rng.values())
            or not isinstance(history, list)
            or len(history) != EPOCHS
            or any(
                set(row) != {"epoch", "loss", "accuracy", "seconds"}
                or row.get("epoch") != index
                or not all(
                    isinstance(row.get(field), (int, float))
                    and math.isfinite(float(row[field]))
                    for field in ("loss", "accuracy", "seconds")
                )
                or not 0.0 <= float(row["accuracy"]) <= 1.0
                or float(row["seconds"]) < 0.0
                for index, row in enumerate(history, start=1)
            )
            or not isinstance(receipt.get("train_seconds"), (int, float))
            or not math.isfinite(float(receipt["train_seconds"]))
            or float(receipt["train_seconds"]) < 0.0
        ):
            raise ValueError(f"Round-three member execution evidence changed: {name}")
        expected_receipt_path = output_dir / "members" / name / "training_receipt.json"
        if not expected_receipt_path.is_file():
            raise ValueError(f"Round-three member receipt is missing: {name}")
        disk_receipt = runner.load_json_strict(expected_receipt_path)
        if runner.canonical_json_bytes(disk_receipt) != runner.canonical_json_bytes(
            receipt
        ):
            raise ValueError(f"Round-three embedded member receipt differs: {name}")
        artifact = receipt.get("artifact", {})
        expected_artifact_path = output_dir / "members" / name / "weights.pth"
        if (
            runner.resolve_repo_relative(str(artifact.get("path", "")))
            != expected_artifact_path.resolve()
        ):
            raise ValueError(f"Round-three member artifact path changed: {name}")
        _validate_checkpoint_record(artifact)
        state = runner.load_state(expected_artifact_path)
        for key in prior_state:
            if key not in TRAINABLE_PARAMETER_NAMES and not torch.equal(
                state[key], prior_state[key]
            ):
                raise ValueError(f"Frozen member state changed: {name}:{key}")
        states[name] = state
    return states


def _validate_selected_lineage(
    output_dir: Path,
    specification: Mapping[str, Any],
    selection: Mapping[str, Any],
    reconstructed: Mapping[str, torch.Tensor],
) -> None:
    selected = selection["selected"]
    artifact = selected.get("artifact")
    lineage_record = selection.get("lineage")
    if not isinstance(artifact, Mapping) or not isinstance(lineage_record, Mapping):
        raise ValueError("Passing round-three evidence lacks artifact lineage")
    if (
        set(lineage_record) != {"path", "size_bytes", "sha256", "schema", "lineage_id"}
        or lineage_record.get("schema") != LINEAGE_SCHEMA
    ):
        raise ValueError("Round-three lineage reference changed")
    selected_path = output_dir / "selected" / "cognitive_leap_ultra_v51_2.pth"
    if (
        runner.resolve_repo_relative(str(artifact.get("path", "")))
        != selected_path.resolve()
    ):
        raise ValueError("Selected round-three artifact path changed")
    _validate_checkpoint_record(artifact)
    stored = runner.load_state(selected_path)
    if any(not torch.equal(stored[name], reconstructed[name]) for name in stored):
        raise ValueError("Selected checkpoint failed exact reconstruction")
    lineage_path = output_dir / "selected" / "lineage_manifest.json"
    if not _same_bound_file(
        {key: lineage_record[key] for key in ("path", "size_bytes", "sha256")},
        lineage_path,
    ):
        raise ValueError("Round-three lineage file binding changed")
    lineage = runner.load_json_strict(lineage_path)
    expected_members = {
        name: {
            "receipt_id": receipt["receipt_id"],
            "artifact": dict(receipt["artifact"]),
        }
        for name, receipt in sorted(selection["member_receipts"].items())
    }
    if (
        set(lineage)
        != {
            "schema",
            "authentication",
            "trusted_timestamp",
            "integrity_status",
            "authority",
            "specification_sha256",
            "method",
            "root_prior_candidate",
            "members",
            "member_weights",
            "selected_alpha",
            "selected_canonical_state_sha256",
            "selected_artifact",
            "exact_tensor_reconstruction",
            "lineage_id",
        }
        or lineage.get("schema") != LINEAGE_SCHEMA
        or lineage.get("lineage_id") != _content_digest(lineage, "lineage_id")
        or lineage_record.get("lineage_id") != lineage["lineage_id"]
        or lineage.get("authentication") != "none"
        or lineage.get("trusted_timestamp") is not False
        or lineage.get("integrity_status") != "content_bound_not_authenticated"
        or lineage.get("authority") != AUTHORITY
        or lineage.get("specification_sha256") != specification["specification_sha256"]
        or lineage.get("method")
        != "prior_started_equal_member_soup_then_linear_interpolation"
        or lineage.get("root_prior_candidate")
        != specification["parents"]["prior_candidate"]
        or lineage.get("members") != expected_members
        or lineage.get("member_weights") != [0.5, 0.5]
        or lineage.get("selected_alpha") != selected["baseline_blend_alpha"]
        or lineage.get("selected_canonical_state_sha256")
        != runner.state_dict_summary(reconstructed)["canonical_state_sha256"]
        or lineage.get("selected_artifact") != artifact
        or lineage.get("exact_tensor_reconstruction") is not True
    ):
        raise ValueError("Round-three lineage contract changed")


def verify_development(output_dir: Path, device: torch.device) -> Path:
    output_dir = output_dir.resolve()
    specification = _validate_specification(output_dir, device)
    parents = specification["parents"]

    selection_path = output_dir / "round3_development_receipt.json"
    selection = runner.load_json_strict(selection_path)
    if not isinstance(selection, Mapping):
        raise ValueError("Round-three selection is not an object")
    expected_selection_fields = {
        "schema",
        "authentication",
        "trusted_timestamp",
        "integrity_status",
        "authority",
        "claim_scope",
        "specification",
        "specification_content_sha256",
        "development_dataset_sha256",
        "member_receipts",
        "candidates",
        "selected",
        "passed",
        "decision",
        "environment",
        "receipt_id",
    }
    if bool(selection.get("passed")):
        expected_selection_fields.add("lineage")
    if set(selection) != expected_selection_fields:
        raise ValueError("Round-three selection fields changed")
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("receipt_id") != _content_digest(selection, "receipt_id")
        or selection.get("authentication") != "none"
        or selection.get("trusted_timestamp") is not False
        or selection.get("integrity_status") != "content_bound_not_authenticated"
        or selection.get("authority") != AUTHORITY
        or selection.get("claim_scope") != CLAIM_SCOPE
        or selection.get("specification_content_sha256")
        != specification["specification_sha256"]
        or not _same_bound_file(
            selection.get("specification", {}),
            output_dir / "search_specification.json",
        )
    ):
        raise ValueError("Round-three selection contract changed")
    if not isinstance(selection.get("environment"), Mapping):
        raise ValueError("Round-three selection environment is missing")
    runner.validate_environment_compatibility(
        selection["environment"], runner.environment_binding(device)
    )
    passed = bool(selection.get("passed"))
    if passed:
        if selection.get("decision") != "development_candidate_found":
            raise ValueError("Passing round-three decision changed")
    elif (
        selection.get("decision") != "no_development_candidate_passed"
        or "artifact" in selection.get("selected", {})
        or "lineage" in selection
        or (output_dir / "selected").exists()
    ):
        raise ValueError("Rejected round-three evidence has release artifacts")

    baseline_state = runner.load_state(
        runner.resolve_repo_relative(parents["release_baseline"]["path"])
    )
    prior_state = runner.load_state(
        runner.resolve_repo_relative(parents["prior_candidate"]["path"])
    )
    member_states = _load_member_evidence(
        output_dir, specification, selection, prior_state
    )
    candidate_states = _candidate_states(prior_state, member_states)
    candidates = selection.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != len(BLEND_ALPHAS)
        or tuple(row.get("baseline_blend_alpha") for row in candidates) != BLEND_ALPHAS
    ):
        raise ValueError("Round-three frozen candidate grid changed")
    for row in candidates:
        alpha = float(row["baseline_blend_alpha"])
        if (
            row.get("canonical_state_sha256")
            != runner.state_dict_summary(candidate_states[alpha])[
                "canonical_state_sha256"
            ]
        ):
            raise ValueError("Round-three candidate state binding changed")
    selected_alpha = float(selection.get("selected", {}).get("baseline_blend_alpha"))
    if selected_alpha not in candidate_states:
        raise ValueError("Round-three selected alpha is outside the frozen grid")
    if passed:
        _validate_selected_lineage(
            output_dir,
            specification,
            selection,
            candidate_states[selected_alpha],
        )

    # Only after every static artifact, receipt, and reconstruction check passes
    # may verification regenerate the fresh development cohort.
    cohort = build_dev3_cohort()
    if cohort["dataset_sha256"] != selection["development_dataset_sha256"]:
        raise ValueError("Round-three development cohort identity changed")
    replayed_rows = _evaluate_candidates(
        specification,
        baseline_state,
        prior_state,
        candidate_states,
        cohort,
        device,
    )
    if runner.canonical_json_bytes(replayed_rows) != runner.canonical_json_bytes(
        selection.get("candidates")
    ):
        raise ValueError("Round-three candidate matrix did not replay exactly")
    replayed_selected = max(
        replayed_rows, key=lambda row: tuple(row["selection_score"])
    )
    stored_selected = {
        key: value for key, value in selection["selected"].items() if key != "artifact"
    }
    if (
        runner.canonical_json_bytes(replayed_selected)
        != runner.canonical_json_bytes(stored_selected)
        or bool(replayed_selected["passed"]) != passed
    ):
        raise ValueError("Round-three frozen selection did not replay")
    replay: dict[str, Any] = {
        "schema": REPLAY_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(AUTHORITY),
        "specification_sha256": specification["specification_sha256"],
        "selection_receipt_id": selection["receipt_id"],
        "selection_file_sha256": runner.sha256_file(selection_path),
        "development_dataset_sha256": cohort["dataset_sha256"],
        "development_seeds": list(DEV3_SEEDS),
        "passed": passed,
        "selected_name": replayed_selected["name"],
        "selected_score": replayed_selected["selection_score"],
        "candidate_matrix_sha256": runner.sha256_bytes(
            runner.canonical_json_bytes(replayed_rows)
        ),
    }
    replay["verification_id"] = _content_digest(replay, "verification_id")
    path = output_dir / "development_replay_verification.json"
    runner.write_json_exclusive(path, replay)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        default="run",
        choices=("run", "verify-development"),
    )
    parser.add_argument("--round1", type=Path)
    parser.add_argument("--round2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    device = runner.configure_runtime(args.torch_threads)
    if args.phase == "run":
        if args.round1 is None or args.round2 is None:
            raise ValueError("run requires --round1 and --round2")
        path = run_search(
            args.round1.resolve(),
            args.round2.resolve(),
            args.output_dir.resolve(),
            device,
        )
    else:
        if args.round1 is not None or args.round2 is not None:
            raise ValueError(
                "verify-development reads parent paths from the specification"
            )
        path = verify_development(args.output_dir.resolve(), device)
    print(path)


if __name__ == "__main__":
    main()
