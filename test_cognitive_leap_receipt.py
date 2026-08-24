from __future__ import annotations

import gzip
import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest
import torch


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import cognitive_leap_receipt as receipt  # noqa: E402
from benchmark_cognitive_leap_ultra_v51 import (  # noqa: E402
    make_chained_task_with_metadata,
)


PASSING_CRITERIA = {
    "minimum_accuracy_gain": -1.0,
    "maximum_p_value": 1.0,
    "minimum_nonregressing_seed_fraction": 0.0,
    "minimum_worst_seed_delta": -1.0,
    "minimum_nonregressing_operation_families": 0,
    "minimum_worst_operation_family_delta": -1.0,
    "minimum_nonregressing_classes": 0,
    "minimum_worst_class_delta": -1.0,
    "require_mean_loss_nonregression": False,
}


def _family_tags(op_types: list[int]) -> list[str]:
    return [
        f"first_{('add', 'mul', 'sub')[op_types[0]]}",
        f"mul_count_{sum(value == 1 for value in op_types)}",
    ]


def _logits(prediction: int, strength: float) -> list[float]:
    values = [-0.25] * 10
    values[prediction] = strength
    return values


def _write_gzip(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str, int]:
    payload = b"".join(receipt.canonical_json_bytes(row) + b"\n" for row in rows)
    with path.open("wb") as raw_handle:
        with gzip.GzipFile(
            fileobj=raw_handle,
            mode="wb",
            filename="",
            mtime=0,
        ) as handle:
            handle.write(payload)
    return receipt.sha256_file(path), hashlib.sha256(payload).hexdigest(), path.stat().st_size


def _artifact_fixture(
    tmp_path: Path,
    *,
    seeds: tuple[int, ...] = (1_052, 2_052),
    samples_per_seed: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    generator_path = SOURCE_DIR / "benchmark_cognitive_leap_ultra_v51.py"
    specification = {
        "schema": "supermix-cognitive-leap-cohort-v1",
        "generator_schema": "supermix-cognitive-leap-generator-v1",
        "family_tag_schema": "supermix-cognitive-leap-family-tags-v1",
        "cohort_role": "final",
        "seeds": list(seeds),
        "samples_per_seed": samples_per_seed,
        "generator_source_sha256": receipt.sha256_file(generator_path),
    }
    specification_sha256 = receipt.sha256_bytes(
        receipt.canonical_json_bytes(specification)
    )
    content_sha256 = receipt.dataset_sha256(seeds, samples_per_seed)
    dataset_id = receipt.dataset_id_for(specification_sha256, content_sha256)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        _x, targets, metadata = make_chained_task_with_metadata(samples_per_seed, seed)
        for index in range(samples_per_seed):
            target = int(targets[index].item())
            baseline_prediction = target if index % 2 else (target + 1) % 10
            candidate_prediction = target
            op_types = [int(value) for value in metadata["op_types"][index]]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "cohort_role": "final",
                    "example_id": f"{dataset_id}:{seed}:{index}",
                    "seed": seed,
                    "index": index,
                    "target": target,
                    "start": int(metadata["starts"][index].item()),
                    "op_types": op_types,
                    "operands": [
                        int(value) for value in metadata["operands"][index]
                    ],
                    "operation_family_tags": _family_tags(op_types),
                    "baseline_logits_f32le_hex": receipt.encode_logits_f32le_hex(
                        _logits(baseline_prediction, 3.0)
                    ),
                    "candidate_logits_f32le_hex": receipt.encode_logits_f32le_hex(
                        _logits(candidate_prediction, 4.0)
                    ),
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                    "baseline_correct": baseline_prediction == target,
                    "candidate_correct": True,
                }
            )

    path = tmp_path / "final_predictions.jsonl.gz"
    compressed_sha256, uncompressed_sha256, size_bytes = _write_gzip(path, rows)
    artifact = {
        "schema": receipt.PREDICTION_ARTIFACT_SCHEMA,
        "path": path.name,
        "sha256": compressed_sha256,
        "size_bytes": size_bytes,
        "row_count": len(rows),
        "format": "gzip_jsonl",
        "uncompressed_sha256": uncompressed_sha256,
        "dataset_id": dataset_id,
        "cohort_role": "final",
        "class_order": list(range(10)),
        "class_count": 10,
        "logit_shape": [10],
        "logits_encoding": "hex_little_endian_float32",
        "argmax_tie_rule": "lowest_class_index",
        "loss_formula": "torch_cross_entropy_float32_sum_per_seed_then_float64_total",
        "validation_absolute_tolerance": 1e-6,
        "gzip_mtime": 0,
    }
    return artifact, rows, specification


def _rewrite_artifact(
    tmp_path: Path,
    artifact: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    path = tmp_path / artifact["path"]
    compressed_sha256, uncompressed_sha256, size_bytes = _write_gzip(path, rows)
    artifact.update(
        {
            "sha256": compressed_sha256,
            "uncompressed_sha256": uncompressed_sha256,
            "size_bytes": size_bytes,
        }
    )


def _validate_artifact(
    tmp_path: Path,
    artifact: dict[str, Any],
    specification: dict[str, Any],
) -> dict[str, Any]:
    return receipt.validate_prediction_artifact(
        artifact,
        root=tmp_path,
        seeds=specification["seeds"],
        samples_per_seed=specification["samples_per_seed"],
        criteria=PASSING_CRITERIA,
        cohort_specification=specification,
    )


def test_canonical_json_rejects_nonfinite_and_duplicate_keys() -> None:
    with pytest.raises(receipt.ReceiptValidationError, match="canonical JSON"):
        receipt.canonical_json_bytes({"bad": float("nan")})
    with pytest.raises(receipt.ReceiptValidationError, match="Non-finite"):
        receipt.loads_json_strict('{"bad":NaN}')
    with pytest.raises(receipt.ReceiptValidationError, match="Duplicate"):
        receipt.loads_json_strict('{"same":1,"same":2}')


def test_exact_little_endian_logits_round_trip_and_nonfinite_rejection() -> None:
    values = [index / 10 for index in range(10)]
    encoded = receipt.encode_logits_f32le_hex(values)
    decoded = receipt.decode_logits_f32le_hex(encoded)
    assert decoded == pytest.approx(values, abs=1e-7)
    nan_payload = (b"\x00\x00\xc0\x7f" + b"\x00\x00\x00\x00" * 9).hex()
    with pytest.raises(receipt.ReceiptValidationError, match="NaN"):
        receipt.decode_logits_f32le_hex(nan_payload)


def test_dataset_replay_matches_bound_benchmark_generator() -> None:
    seed = 7_052
    samples = 6
    expected_x, expected_y, expected_metadata = make_chained_task_with_metadata(
        samples,
        seed,
    )
    replay_x, replay_y, replay_metadata = receipt._canonical_task(samples, seed)
    assert torch.equal(expected_x, replay_x)
    assert torch.equal(expected_y, replay_y)
    for name in ("starts", "op_types", "operands"):
        assert torch.equal(expected_metadata[name], replay_metadata[name])


def test_prediction_artifact_recomputes_complete_gate_evidence(tmp_path: Path) -> None:
    artifact, _rows, specification = _artifact_fixture(tmp_path)
    result = _validate_artifact(tmp_path, artifact, specification)
    assert result["gate_outcome"] == "pass"
    assert result["summary"]["n"] == artifact["row_count"]
    assert result["summary"]["wins"] > 0
    assert result["summary"]["regressions"] == 0
    assert result["summary"]["ties"] + result["summary"]["wins"] == artifact["row_count"]
    assert result["summary"]["eligible_family_count"] <= 8
    assert result["summary"]["eligible_class_count"] <= 10
    assert result["evidence"]["dataset_id"] == artifact["dataset_id"]
    assert result["evidence"]["dataset_sha256"] == receipt.dataset_sha256(
        specification["seeds"],
        specification["samples_per_seed"],
    )
    assert result["evidence"]["per_example_compressed_sha256"] == artifact["sha256"]


def test_legacy_validator_uses_exact_inclusive_count_thresholds(
    tmp_path: Path,
) -> None:
    artifact, rows, specification = _artifact_fixture(
        tmp_path,
        seeds=(1_052,),
        samples_per_seed=200,
    )

    def set_predictions(regression_count: int) -> None:
        for index, row in enumerate(rows):
            target = int(row["target"])
            candidate_prediction = (
                (target + 1) % 10 if index < regression_count else target
            )
            row.update(
                {
                    "baseline_logits_f32le_hex": receipt.encode_logits_f32le_hex(
                        _logits(target, 3.0)
                    ),
                    "candidate_logits_f32le_hex": receipt.encode_logits_f32le_hex(
                        _logits(candidate_prediction, 4.0)
                    ),
                    "baseline_prediction": target,
                    "candidate_prediction": candidate_prediction,
                    "baseline_correct": True,
                    "candidate_correct": candidate_prediction == target,
                }
            )
        _rewrite_artifact(tmp_path, artifact, rows)

    criteria = {
        **PASSING_CRITERIA,
        "minimum_accuracy_gain": -0.005,
        "minimum_worst_seed_delta": -0.005,
    }
    set_predictions(1)
    boundary = receipt.validate_prediction_artifact(
        artifact,
        root=tmp_path,
        seeds=specification["seeds"],
        samples_per_seed=specification["samples_per_seed"],
        criteria=criteria,
        cohort_specification=specification,
    )
    set_predictions(2)
    below = receipt.validate_prediction_artifact(
        artifact,
        root=tmp_path,
        seeds=specification["seeds"],
        samples_per_seed=specification["samples_per_seed"],
        criteria=criteria,
        cohort_specification=specification,
    )

    assert boundary["summary"]["accuracy_delta"] == -0.005
    assert boundary["checks"]["accuracy_gain"] is True
    assert boundary["checks"]["seed_nonregression"] is True
    assert below["summary"]["accuracy_delta"] == -0.01
    assert below["checks"]["accuracy_gain"] is False
    assert below["checks"]["seed_nonregression"] is False


def test_prediction_artifact_rejects_reordered_rows_even_with_new_hashes(
    tmp_path: Path,
) -> None:
    artifact, rows, specification = _artifact_fixture(tmp_path)
    rows[0], rows[1] = rows[1], rows[0]
    _rewrite_artifact(tmp_path, artifact, rows)
    with pytest.raises(receipt.ReceiptValidationError, match="order"):
        _validate_artifact(tmp_path, artifact, specification)


def test_prediction_artifact_rejects_truncation_even_with_new_hashes(
    tmp_path: Path,
) -> None:
    artifact, rows, specification = _artifact_fixture(tmp_path)
    _rewrite_artifact(tmp_path, artifact, rows[:-1])
    with pytest.raises(receipt.ReceiptValidationError, match="truncated"):
        _validate_artifact(tmp_path, artifact, specification)


def test_prediction_artifact_rejects_semantic_tamper_even_with_new_hashes(
    tmp_path: Path,
) -> None:
    artifact, rows, specification = _artifact_fixture(tmp_path)
    rows[0]["target"] = (rows[0]["target"] + 1) % 10
    _rewrite_artifact(tmp_path, artifact, rows)
    with pytest.raises(receipt.ReceiptValidationError, match="generator"):
        _validate_artifact(tmp_path, artifact, specification)


def test_prediction_artifact_rejects_logit_corruption(tmp_path: Path) -> None:
    artifact, rows, specification = _artifact_fixture(tmp_path)
    path = tmp_path / artifact["path"]
    compressed = bytearray(path.read_bytes())
    compressed[-5] ^= 0x01
    path.write_bytes(compressed)
    with pytest.raises(receipt.ReceiptValidationError, match="SHA-256"):
        _validate_artifact(tmp_path, artifact, specification)


def test_prediction_artifact_rejects_prediction_not_derived_from_logits(
    tmp_path: Path,
) -> None:
    artifact, rows, specification = _artifact_fixture(tmp_path)
    rows[0]["baseline_prediction"] = (rows[0]["baseline_prediction"] + 1) % 10
    rows[0]["baseline_correct"] = rows[0]["baseline_prediction"] == rows[0]["target"]
    _rewrite_artifact(tmp_path, artifact, rows)
    with pytest.raises(receipt.ReceiptValidationError, match="prediction"):
        _validate_artifact(tmp_path, artifact, specification)


def test_state_hash_average_and_exact_reconstruction() -> None:
    left = {
        "weight": torch.tensor([1.0, 3.0], dtype=torch.float32),
        "counter": torch.tensor([2], dtype=torch.int64),
    }
    right = {
        "weight": torch.tensor([3.0, 5.0], dtype=torch.float32),
        "counter": torch.tensor([2], dtype=torch.int64),
    }
    expected = {
        "weight": torch.tensor([1.5, 3.5], dtype=torch.float32),
        "counter": torch.tensor([2], dtype=torch.int64),
    }
    averaged = receipt.average_state_dicts([left, right], [0.75, 0.25])
    assert torch.equal(averaged["weight"], expected["weight"])
    summary = receipt.validate_state_reconstruction(
        [left, right],
        [0.75, 0.25],
        expected,
    )
    assert summary["all_finite"] is True
    assert summary["tensor_byte_order"] == "little_endian"


def test_state_average_rejects_nonfloating_mismatch() -> None:
    left = {"counter": torch.tensor([1], dtype=torch.int64)}
    right = {"counter": torch.tensor([2], dtype=torch.int64)}
    with pytest.raises(receipt.ReceiptValidationError, match="Non-floating"):
        receipt.average_state_dicts([left, right])


def _invalid_receipt(authority: dict[str, bool]) -> dict[str, Any]:
    return {
        "schema": receipt.RECEIPT_SCHEMA,
        "receipt_id": "0" * 64,
        "gate_outcome": "reject",
        "authority": authority,
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "trusted_timestamp": False,
        "protocol": {},
        "selection": {},
        "artifacts": {},
        "criteria": {},
        "checks": {},
        "summary": {},
        "seed_rows": [],
        "operation_family_rows": [],
        "class_rows": [],
        "evidence": {},
        "per_example_artifact": {},
    }


def test_receipt_rejects_any_authority_before_artifact_access(tmp_path: Path) -> None:
    authority = dict(receipt.NO_AUTHORITY)
    authority["store_publication"] = True
    with pytest.raises(receipt.ReceiptValidationError, match="no authority"):
        receipt.validate_receipt(_invalid_receipt(authority), root=tmp_path)


def test_receipt_rejects_unknown_gate_outcome_and_nonraising_wrapper(
    tmp_path: Path,
) -> None:
    value = _invalid_receipt(dict(receipt.NO_AUTHORITY))
    value["gate_outcome"] = "promote"
    with pytest.raises(receipt.ReceiptValidationError, match="pass or reject"):
        receipt.validate_receipt(value, root=tmp_path)
    result = receipt.try_validate_receipt(value, root=tmp_path)
    assert result["valid"] is False
    assert "pass or reject" in result["error"]


def test_exact_mcnemar_known_values() -> None:
    assert receipt.exact_mcnemar_two_sided(0, 0) == 1.0
    assert receipt.exact_mcnemar_two_sided(4, 0) == pytest.approx(0.125)
    assert receipt.exact_mcnemar_two_sided(5, 1) == pytest.approx(0.21875)
