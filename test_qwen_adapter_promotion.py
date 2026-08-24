from __future__ import annotations

import json
import copy
import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pytest

import source.qwen_adapter_promotion as promotion_module
import source.run_qwen_general_promotion_gate as promotion_gate
from source.build_general_intelligence_curriculum import build_curriculum
from source.qwen_adapter_promotion import (
    BENCHMARK_SCHEMA_VERSION,
    GATE_FILENAME,
    PRODUCTION_POLICY_ID,
    PRODUCTION_THRESHOLD_FLOORS,
    PROMOTION_FILENAME,
    adapter_activation_kind,
    attest_adapter_for_runtime,
    evaluation_code_hashes,
    sha256_file,
    validate_promoted_adapter,
)
from source.qwen_paired_evidence import (
    PAIRED_EVIDENCE_SCHEMA_VERSION,
    build_paired_evidence,
    derive_sample_comparison,
    deterministic_eval_selection,
    paired_evidence_sha256,
)
from source.run_qwen_general_promotion_gate import (
    PRODUCTION_PROTOCOL,
    evaluate_promotion,
    run_gate,
)
from source.verifiable_reasoning import VERIFIER_SCHEMA_VERSION


SOURCE_DIR = Path(__file__).resolve().parent / "source"


def _metrics(*, accuracy: float, loss: float, token_f1: float) -> dict[str, object]:
    return {
        "eval_samples": 30.0,
        "verified_samples": 30.0,
        "verified_accuracy": accuracy,
        "eval_loss": loss,
        "token_f1": token_f1,
        "generation_cap": 64.0,
        "generation_cap_hits": 0.0,
        "generation_cap_rate": 0.0,
        "verified_samples_family_math": 15.0,
        "verified_accuracy_family_math": accuracy,
        "verified_samples_family_science": 15.0,
        "verified_accuracy_family_science": accuracy,
    }


def _benchmark(*, reference_accuracy: float, tuned_accuracy: float) -> dict[str, object]:
    sample_count = 30
    reference_correct = round(reference_accuracy * sample_count)
    tuned_correct = round(tuned_accuracy * sample_count)
    wins = max(0, tuned_correct - reference_correct)
    regressions = max(0, reference_correct - tuned_correct)
    discordant = wins + regressions
    if discordant:
        p_value = sum(math.comb(discordant, count) for count in range(wins, discordant + 1)) / (
            2**discordant
        )
    else:
        p_value = 1.0
    evidence = {
        "schema": PAIRED_EVIDENCE_SCHEMA_VERSION,
        "identity": {"template_cluster_count": 6},
        "transitions": {
            "samples": sample_count,
            "base_correct": reference_correct,
            "tuned_correct": tuned_correct,
            "wins": wins,
            "regressions": regressions,
            "discordant_pairs": discordant,
            "base_accuracy": reference_accuracy,
            "tuned_accuracy": tuned_accuracy,
            "accuracy_delta": tuned_accuracy - reference_accuracy,
        },
        "per_family": {
            family: {
                "samples": 15,
                "base_accuracy": reference_accuracy,
                "tuned_accuracy": tuned_accuracy,
                "accuracy_delta": tuned_accuracy - reference_accuracy,
            }
            for family in ("math", "science")
        },
        "mcnemar_exact_one_sided": {"p_value": p_value},
        "template_cluster_bootstrap": {
            "lower_95": (tuned_accuracy - reference_accuracy) / 2.0,
            "upper_95": tuned_accuracy - reference_accuracy,
        },
    }
    return {
        "base": _metrics(accuracy=reference_accuracy, loss=2.0, token_f1=0.20),
        "tuned": _metrics(accuracy=tuned_accuracy, loss=1.7, token_f1=0.24),
        "paired_evidence": evidence,
    }


def test_promotion_gate_accepts_measured_gain_without_family_regression() -> None:
    decision = evaluate_promotion(_benchmark(reference_accuracy=0.20, tuned_accuracy=0.40))
    assert decision["passed"] is True
    assert decision["blockers"] == []


def test_promotion_gate_rejects_promising_but_underpowered_paired_gain() -> None:
    benchmark = _benchmark(reference_accuracy=0.20, tuned_accuracy=11 / 30)
    tuned = benchmark["tuned"]
    evidence = benchmark["paired_evidence"]
    assert isinstance(tuned, dict) and isinstance(evidence, dict)
    tuned["verified_accuracy"] = 11 / 30
    tuned["verified_accuracy_family_math"] = 6 / 15
    tuned["verified_accuracy_family_science"] = 5 / 15
    transitions = evidence["transitions"]
    assert isinstance(transitions, dict)
    transitions.update(
        {
            "base_correct": 6,
            "tuned_correct": 11,
            "wins": 6,
            "regressions": 1,
            "discordant_pairs": 7,
            "base_accuracy": 0.20,
            "tuned_accuracy": 11 / 30,
            "accuracy_delta": 5 / 30,
        }
    )
    evidence["mcnemar_exact_one_sided"] = {"p_value": 0.0625}
    evidence["template_cluster_bootstrap"] = {
        "lower_95": -0.0281,
        "upper_95": 0.3615,
    }
    evidence["per_family"] = {
        "math": {
            "samples": 15,
            "base_accuracy": 0.20,
            "tuned_accuracy": 6 / 15,
            "accuracy_delta": 3 / 15,
        },
        "science": {
            "samples": 15,
            "base_accuracy": 0.20,
            "tuned_accuracy": 5 / 15,
            "accuracy_delta": 2 / 15,
        },
    }

    decision = evaluate_promotion(benchmark)
    assert decision["passed"] is False
    assert "paired_exact_p_value_above_threshold" in decision["blockers"]
    assert "paired_regression_rate_above_threshold" in decision["blockers"]
    assert "paired_cluster_lower_bound_not_above_threshold" in decision["blockers"]


def test_promotion_gate_rejects_tie_loss_and_family_regression() -> None:
    benchmark = _benchmark(reference_accuracy=0.40, tuned_accuracy=0.40)
    tuned = benchmark["tuned"]
    assert isinstance(tuned, dict)
    tuned["eval_loss"] = 2.2
    tuned["verified_accuracy_family_math"] = 0.0
    decision = evaluate_promotion(benchmark)
    assert decision["passed"] is False
    assert "verified_accuracy_gain_below_threshold" in decision["blockers"]
    assert "eval_loss_regression" in decision["blockers"]
    assert "family_regression:math" in decision["blockers"]


@pytest.mark.parametrize(
    ("side", "metric", "value"),
    (
        ("tuned", "verified_accuracy", float("nan")),
        ("base", "eval_loss", float("inf")),
        ("tuned", "token_f1", -0.01),
        ("base", "generation_cap_rate", 1.01),
        ("tuned", "verified_samples", 30.5),
        ("base", "perplexity", float("nan")),
        ("tuned", "generated_tokens_per_sec", float("inf")),
    ),
)
def test_promotion_gate_rejects_nonfinite_or_out_of_range_metrics(
    side: str,
    metric: str,
    value: float,
) -> None:
    benchmark = _benchmark(reference_accuracy=0.20, tuned_accuracy=0.40)
    metrics = benchmark[side]
    assert isinstance(metrics, dict)
    metrics[metric] = value
    assert evaluate_promotion(benchmark)["passed"] is False


def test_promotion_gate_requires_exact_sample_and_family_parity() -> None:
    benchmark = _benchmark(reference_accuracy=0.20, tuned_accuracy=0.40)
    tuned = benchmark["tuned"]
    assert isinstance(tuned, dict)
    tuned["eval_samples"] = 29.0
    tuned.pop("verified_samples_family_science")
    decision = evaluate_promotion(benchmark)
    assert decision["passed"] is False
    assert "base_tuned_sample_count_mismatch" in decision["blockers"]
    assert "family_metric_set_mismatch" in decision["blockers"]


def test_promotion_gate_enforces_family_coverage_and_generation_cap_rate() -> None:
    benchmark = _benchmark(reference_accuracy=0.20, tuned_accuracy=0.40)
    tuned = benchmark["tuned"]
    assert isinstance(tuned, dict)
    tuned["generation_cap_hits"] = 3.0
    tuned["generation_cap_rate"] = 0.10
    decision = evaluate_promotion(
        benchmark,
        min_family_verified_samples=16,
        max_generation_cap_rate=0.05,
    )
    assert decision["passed"] is False
    assert "insufficient_family_samples:math" in decision["blockers"]
    assert "insufficient_family_samples:science" in decision["blockers"]
    assert "tuned_generation_cap_rate_above_threshold" in decision["blockers"]


def test_promotion_gate_rejects_unrealizable_accuracy_counts() -> None:
    benchmark = _benchmark(reference_accuracy=0.20, tuned_accuracy=0.41)
    decision = evaluate_promotion(benchmark)
    assert decision["passed"] is False
    assert "tuned_verified_accuracy_not_realizable" in decision["blockers"]
    assert "tuned_family_accuracy_not_realizable:math" in decision["blockers"]


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def _make_adapter(root: Path, weights: bytes) -> Path:
    adapter = root / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path":"Qwen/test","r":8}',
        encoding="utf-8",
    )
    return adapter


@lru_cache(maxsize=1)
def _production_curriculum() -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    bundle = build_curriculum(seed=6201, train_rows=1_200, eval_rows=150)
    return bundle.eval_rows, bundle.manifest


def _make_bound_gate_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    adapter = _make_adapter(tmp_path / "candidate", b"weights-v1")
    resolved_base_model = (
        tmp_path
        / "model-cache"
        / "models--Qwen--test"
        / "snapshots"
        / "revision-abc"
    )
    resolved_base_model.mkdir(parents=True)
    curriculum_rows, cached_manifest = _production_curriculum()
    rows = deterministic_eval_selection(
        curriculum_rows,
        seed=int(PRODUCTION_PROTOCOL["selection_seed"]),
        samples_per_family=int(PRODUCTION_PROTOCOL["samples_per_family"]),
        max_eval_samples=int(PRODUCTION_PROTOCOL["max_eval_samples"]),
    )
    curriculum_eval_bytes = _jsonl_bytes(list(curriculum_rows))
    assert hashlib.sha256(curriculum_eval_bytes).hexdigest() == str(
        PRODUCTION_PROTOCOL["curriculum_eval_sha256"]
    )

    curriculum_dir = tmp_path / "curriculum"
    curriculum_dir.mkdir()
    curriculum_eval = curriculum_dir / "general_intelligence_eval.jsonl"
    curriculum_eval.write_bytes(curriculum_eval_bytes)
    curriculum_manifest = curriculum_dir / "general_intelligence_manifest.json"
    curriculum_manifest.write_text(
        json.dumps(cached_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    selected_eval = benchmark_dir / "eval_pairs.jsonl"
    selected_eval.write_bytes(_jsonl_bytes(rows))
    base_correct_indices: set[int] = set()
    tuned_correct_indices = set(range(len(rows)))

    def sample_rows(correct_indices: set[int], *, loss: float) -> list[dict[str, object]]:
        detailed: list[dict[str, object]] = []
        for sample_index, row in enumerate(rows):
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            reference = str(row["assistant"])
            detailed.append(
                {
                    "sample_index": sample_index,
                    "source": str(row["source"]),
                    "user": str(row["user"]),
                    "reference": reference,
                    "prediction": (
                        reference if sample_index in correct_indices else ""
                    ),
                    "prompt_signature": f"prompt-signature-{sample_index}",
                    "example_id": str(metadata["example_id"]),
                    "template_id": str(metadata["template_id"]),
                    "split_group": str(metadata["split_group"]),
                    "problem_family": str(metadata["problem_family"]),
                    "loss": loss,
                    "token_f1": 0.0,
                    "char_similarity": 0.0,
                    "prompt_tokens": 8,
                    "generated_tokens": 1,
                    "generation_cap": 64,
                    "generation_cap_hit": False,
                    "verified_correct": sample_index in correct_indices,
                }
            )
        return detailed

    base_samples = sample_rows(base_correct_indices, loss=2.0)
    tuned_samples = sample_rows(tuned_correct_indices, loss=1.7)
    base_samples_path = benchmark_dir / "base_samples.jsonl"
    tuned_samples_path = benchmark_dir / "tuned_samples.jsonl"
    comparison_path = benchmark_dir / "sample_comparison.jsonl"
    base_samples_path.write_bytes(_jsonl_bytes(base_samples))
    tuned_samples_path.write_bytes(_jsonl_bytes(tuned_samples))
    comparison_path.write_bytes(_jsonl_bytes(derive_sample_comparison(base_samples, tuned_samples)))
    artifact_hashes = {
        "base_samples_sha256": sha256_file(base_samples_path),
        "tuned_samples_sha256": sha256_file(tuned_samples_path),
        "sample_comparison_sha256": sha256_file(comparison_path),
    }
    paired_evidence = build_paired_evidence(
        base_samples,
        tuned_samples,
        rows,
        artifact_hashes=artifact_hashes,
        bootstrap_seed=int(PRODUCTION_PROTOCOL["paired_bootstrap_seed"]),
        bootstrap_resamples=int(PRODUCTION_PROTOCOL["paired_bootstrap_resamples"]),
    )
    recomputed = paired_evidence["recomputed_metrics"]
    assert isinstance(recomputed, dict)
    benchmark_path = benchmark_dir / "benchmark_results.json"
    benchmark = {
            "schema": BENCHMARK_SCHEMA_VERSION,
            "config": {
                "base_model": "Qwen/test",
                "base_model_revision": "revision-abc",
                "resolved_base_model_path": str(resolved_base_model.resolve()),
                "adapter_dir": str(adapter.resolve()),
                "reference_adapter_dir": "",
                "eval_source": str(curriculum_eval.resolve()),
                "curriculum_manifest": str(curriculum_manifest.resolve()),
                "eval_samples": len(rows),
                "seed": int(PRODUCTION_PROTOCOL["selection_seed"]),
                "samples_per_family": int(PRODUCTION_PROTOCOL["samples_per_family"]),
                "max_eval_samples": int(PRODUCTION_PROTOCOL["max_eval_samples"]),
                "max_length": int(PRODUCTION_PROTOCOL["max_length"]),
                "max_new_tokens": int(PRODUCTION_PROTOCOL["max_new_tokens"]),
                "paired_bootstrap_seed": int(PRODUCTION_PROTOCOL["paired_bootstrap_seed"]),
                "paired_bootstrap_resamples": int(
                    PRODUCTION_PROTOCOL["paired_bootstrap_resamples"]
                ),
            },
            "provenance": {
                "base_model": "Qwen/test",
                "base_model_revision": "revision-abc",
                "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
                "selected_eval_sha256": sha256_file(selected_eval),
                "curriculum_manifest_sha256": sha256_file(curriculum_manifest),
                "curriculum_eval_sha256": sha256_file(curriculum_eval),
                "verifier_schema": VERIFIER_SCHEMA_VERSION,
                "code_hashes": evaluation_code_hashes(SOURCE_DIR),
                **artifact_hashes,
                "paired_evidence_sha256": paired_evidence_sha256(paired_evidence),
                "paired_evidence_schema": PAIRED_EVIDENCE_SCHEMA_VERSION,
            },
            "artifact_hashes": artifact_hashes,
            "artifacts": {
                "base_samples_jsonl": str(base_samples_path),
                "tuned_samples_jsonl": str(tuned_samples_path),
                "sample_comparison_jsonl": str(comparison_path),
            },
            "paired_evidence": paired_evidence,
            "base": recomputed["base"],
            "tuned": recomputed["tuned"],
            "delta_tuned_minus_base": recomputed["delta_tuned_minus_base"],
        }
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    return benchmark_path, adapter, curriculum_manifest


def _run_bound_gate(
    *,
    benchmark_path: Path,
    adapter: Path,
    curriculum_manifest: Path,
    pointer: Path,
    write_pointer: bool = True,
    threshold_overrides: Mapping[str, int | float] | None = None,
) -> dict[str, object]:
    thresholds: dict[str, int | float] = {
        "min_verified_samples": 20,
        "min_verified_gain": 0.05,
        "min_tuned_accuracy": 0.20,
        "max_family_regression": 0.0,
        "max_loss_ratio": 1.05,
        "min_token_f1_delta": -0.02,
        "min_family_verified_samples": 1,
        "max_generation_cap_rate": 0.05,
        "max_paired_p_value": 0.05,
        "max_paired_regression_rate": 0.02,
        "min_paired_cluster_lower_bound": 0.0,
        "min_template_clusters": 5,
    }
    thresholds.update(dict(threshold_overrides or {}))
    return run_gate(
        benchmark_path=benchmark_path,
        adapter_dir=adapter,
        curriculum_manifest_path=curriculum_manifest,
        pointer_path=pointer,
        base_model="Qwen/test",
        base_model_revision="revision-abc",
        min_verified_samples=int(thresholds["min_verified_samples"]),
        min_verified_gain=float(thresholds["min_verified_gain"]),
        min_tuned_accuracy=float(thresholds["min_tuned_accuracy"]),
        max_family_regression=float(thresholds["max_family_regression"]),
        max_loss_ratio=float(thresholds["max_loss_ratio"]),
        min_token_f1_delta=float(thresholds["min_token_f1_delta"]),
        min_family_verified_samples=int(thresholds["min_family_verified_samples"]),
        max_generation_cap_rate=float(thresholds["max_generation_cap_rate"]),
        max_paired_p_value=float(thresholds["max_paired_p_value"]),
        max_paired_regression_rate=float(thresholds["max_paired_regression_rate"]),
        min_paired_cluster_lower_bound=float(
            thresholds["min_paired_cluster_lower_bound"]
        ),
        min_template_clusters=int(thresholds["min_template_clusters"]),
        write_pointer=write_pointer,
    )


def _load_jsonl_records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _rebind_paired_benchmark(
    benchmark_path: Path,
    *,
    bootstrap_seed: int | None = None,
    bootstrap_resamples: int | None = None,
) -> None:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    base_path = benchmark_path.parent / "base_samples.jsonl"
    tuned_path = benchmark_path.parent / "tuned_samples.jsonl"
    comparison_path = benchmark_path.parent / "sample_comparison.jsonl"
    selected_path = benchmark_path.parent / "eval_pairs.jsonl"
    artifact_hashes = {
        "base_samples_sha256": sha256_file(base_path),
        "tuned_samples_sha256": sha256_file(tuned_path),
        "sample_comparison_sha256": sha256_file(comparison_path),
    }
    seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else benchmark["config"]["paired_bootstrap_seed"]
    )
    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else benchmark["config"]["paired_bootstrap_resamples"]
    )
    evidence = build_paired_evidence(
        _load_jsonl_records(base_path),
        _load_jsonl_records(tuned_path),
        _load_jsonl_records(selected_path),
        artifact_hashes=artifact_hashes,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )
    benchmark["config"]["paired_bootstrap_seed"] = seed
    benchmark["config"]["paired_bootstrap_resamples"] = resamples
    benchmark["artifact_hashes"] = artifact_hashes
    benchmark["paired_evidence"] = evidence
    benchmark["provenance"].update(artifact_hashes)
    benchmark["provenance"]["paired_evidence_sha256"] = paired_evidence_sha256(evidence)
    recomputed = evidence["recomputed_metrics"]
    benchmark["base"] = recomputed["base"]
    benchmark["tuned"] = recomputed["tuned"]
    benchmark["delta_tuned_minus_base"] = recomputed["delta_tuned_minus_base"]
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")


def _rebind_curriculum_files(
    benchmark_path: Path,
    curriculum_manifest: Path,
    rows: list[dict[str, object]],
) -> None:
    manifest = json.loads(curriculum_manifest.read_text(encoding="utf-8"))
    curriculum_eval = curriculum_manifest.parent / manifest["eval"]["file"]
    curriculum_eval.write_bytes(_jsonl_bytes(rows))
    manifest["eval"]["sha256"] = sha256_file(curriculum_eval)
    curriculum_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["provenance"]["curriculum_eval_sha256"] = sha256_file(curriculum_eval)
    benchmark["provenance"]["curriculum_manifest_sha256"] = sha256_file(
        curriculum_manifest
    )
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")


def test_gate_receipt_is_content_bound_and_detects_tampering(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is True
    manifest = validate_promoted_adapter(adapter)
    assert manifest is not None
    assert manifest["benchmark_schema"] == BENCHMARK_SCHEMA_VERSION
    assert manifest["base_model_revision"] == "revision-abc"
    gate = json.loads((adapter.parent / GATE_FILENAME).read_text(encoding="utf-8"))
    decision = gate["decision"]
    binding = decision["binding"]
    for payload in (manifest, gate, decision):
        assert payload["policy_id"] == PRODUCTION_POLICY_ID
        assert payload["policy_mode"] == "production"
        assert payload["production_eligible"] is True
        assert payload["production_threshold_floors"] == dict(PRODUCTION_THRESHOLD_FLOORS)
        assert payload["production_protocol"] == dict(PRODUCTION_PROTOCOL)
    assert binding["policy_id"] == PRODUCTION_POLICY_ID
    assert binding["policy_mode"] == "production"
    assert binding["production_eligible"] is True

    (adapter / "adapter_model.safetensors").write_bytes(b"tampered")
    assert validate_promoted_adapter(adapter) is None


@pytest.mark.parametrize(
    ("field", "bogus"),
    (
        ("verifier_schema", "bogus-verifier-v999"),
        ("paired_evidence_schema", "bogus-paired-evidence-v999"),
    ),
)
def test_runtime_validator_rejects_internally_consistent_unknown_subschemas(
    tmp_path: Path,
    field: str,
    bogus: str,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is True

    gate_path = adapter.parent / GATE_FILENAME
    manifest_path = adapter.parent / PROMOTION_FILENAME
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate[field] = bogus
    gate["decision"]["binding"][field] = bogus
    manifest[field] = bogus
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    manifest["gate_sha256"] = sha256_file(gate_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_promoted_adapter(adapter) is None


def test_runtime_validator_requires_the_complete_oracle_aware_code_hash_set(
    tmp_path: Path,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is True

    gate_path = adapter.parent / GATE_FILENAME
    manifest_path = adapter.parent / PROMOTION_FILENAME
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate["code_hashes"]["verifier"].pop("logical_entailment.py")
    manifest["code_hashes"] = copy.deepcopy(gate["code_hashes"])
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    manifest["gate_sha256"] = sha256_file(gate_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_promoted_adapter(adapter) is None


@pytest.mark.parametrize("bad_threshold", (0.0, float("nan")))
def test_runtime_validator_rejects_rebound_weak_policy_threshold(
    tmp_path: Path,
    bad_threshold: float,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is True

    gate_path = adapter.parent / GATE_FILENAME
    manifest_path = adapter.parent / PROMOTION_FILENAME
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate["decision"]["thresholds"]["min_verified_gain"] = bad_threshold
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    manifest["gate_sha256"] = sha256_file(gate_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert validate_promoted_adapter(adapter) is None


def test_gate_rejects_sample_artifact_tampering_and_revokes_receipt(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    first = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert first["passed"] is True
    assert validate_promoted_adapter(adapter) is not None

    base_samples = benchmark_path.parent / "base_samples.jsonl"
    base_samples.write_text(base_samples.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert second["passed"] is False
    assert "invalid_paired_evidence_or_artifacts" in second["blockers"]
    assert validate_promoted_adapter(adapter) is None


def test_gate_rejects_rebound_junk_comparison_artifact(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    comparison_path = benchmark_path.parent / "sample_comparison.jsonl"
    comparison = _load_jsonl_records(comparison_path)
    comparison.append({**comparison[0], "junk": True})
    comparison_path.write_bytes(_jsonl_bytes(comparison))
    # Simulate an attacker rebinding every self-reported hash and evidence digest.
    _rebind_paired_benchmark(benchmark_path)

    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is False
    assert "invalid_paired_evidence_or_artifacts" in result["binding_blockers"]
    assert not (adapter.parent / PROMOTION_FILENAME).exists()


def test_gate_rejects_changed_curriculum_even_when_hashes_are_rebound(
    tmp_path: Path,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    manifest = json.loads(curriculum_manifest.read_text(encoding="utf-8"))
    curriculum_eval = curriculum_manifest.parent / manifest["eval"]["file"]
    curriculum_rows = _load_jsonl_records(curriculum_eval)
    selected_rows = _load_jsonl_records(benchmark_path.parent / "eval_pairs.jsonl")
    selected_metadata = selected_rows[0]["metadata"]
    assert isinstance(selected_metadata, dict)
    selected_id = selected_metadata["example_id"]
    changed = False
    for row in curriculum_rows:
        metadata = row.get("metadata")
        if isinstance(metadata, dict) and metadata.get("example_id") == selected_id:
            row["source"] = f"{row['source']}-rebound"
            changed = True
            break
    assert changed
    _rebind_curriculum_files(benchmark_path, curriculum_manifest, curriculum_rows)

    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is False
    assert "invalid_selected_eval_artifact" in result["binding_blockers"]
    assert (
        "production_protocol_violation:curriculum_eval_sha256"
        in result["policy_blockers"]
    )


def test_gate_validates_unique_identity_across_full_curriculum(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    manifest = json.loads(curriculum_manifest.read_text(encoding="utf-8"))
    curriculum_eval = curriculum_manifest.parent / manifest["eval"]["file"]
    curriculum_rows = _load_jsonl_records(curriculum_eval)
    first_metadata = curriculum_rows[0]["metadata"]
    second_metadata = curriculum_rows[1]["metadata"]
    assert isinstance(first_metadata, dict) and isinstance(second_metadata, dict)
    second_metadata["example_id"] = first_metadata["example_id"]
    _rebind_curriculum_files(benchmark_path, curriculum_manifest, curriculum_rows)

    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is False
    assert "invalid_curriculum_manifest_or_eval" in result["binding_blockers"]


def test_adapter_config_accepts_exact_snapshot_and_rejects_rebound_model(
    tmp_path: Path,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    config_path = adapter / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {"base_model_name_or_path": benchmark["config"]["resolved_base_model_path"], "r": 8}
        ),
        encoding="utf-8",
    )
    benchmark["provenance"]["adapter_config_sha256"] = sha256_file(config_path)
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    accepted = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert accepted["passed"] is True
    assert validate_promoted_adapter(adapter) is not None

    config_path.write_text(
        '{"base_model_name_or_path":"Other/model","r":8}',
        encoding="utf-8",
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["provenance"]["adapter_config_sha256"] = sha256_file(config_path)
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    rejected = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert rejected["passed"] is False
    assert "adapter_config_base_model_mismatch" in rejected["binding_blockers"]
    assert validate_promoted_adapter(adapter) is None


def test_no_write_pointer_is_review_only_and_never_writes_manifest(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    pointer = tmp_path / "pointer.txt"
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=pointer,
        write_pointer=False,
    )
    assert result["passed"] is True
    assert result["policy_mode"] == "review"
    assert result["production_eligible"] is True
    assert result["promotion_manifest_path"] == ""
    assert result["pointer_written"] is False
    assert not pointer.exists()
    assert not (adapter.parent / PROMOTION_FILENAME).exists()


@pytest.mark.parametrize("write_pointer", (True, False))
def test_looser_thresholds_are_nonactivating_research_only(
    tmp_path: Path,
    write_pointer: bool,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    pointer = tmp_path / "pointer.txt"
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=pointer,
        write_pointer=write_pointer,
        threshold_overrides={"min_verified_gain": 0.0},
    )
    assert result["policy_mode"] == "research"
    assert result["production_eligible"] is False
    assert result["pointer_written"] is False
    assert result["promotion_manifest_path"] == ""
    assert not pointer.exists()
    assert not (adapter.parent / PROMOTION_FILENAME).exists()
    if write_pointer:
        assert result["passed"] is False
        assert "nonproduction_policy_requires_no_write_pointer" in result["policy_blockers"]
    else:
        assert result["passed"] is True


def test_custom_bootstrap_is_recomputable_but_review_only(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    _rebind_paired_benchmark(benchmark_path, bootstrap_resamples=100)
    pointer = tmp_path / "pointer.txt"
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=pointer,
        write_pointer=False,
    )
    assert result["passed"] is True
    assert result["policy_mode"] == "research"
    assert result["production_eligible"] is False
    assert "paired_bootstrap_resamples" in json.loads(
        Path(result["gate_path"]).read_text(encoding="utf-8")
    )["protocol_violations"]
    assert not pointer.exists()
    assert not (adapter.parent / PROMOTION_FILENAME).exists()


def test_nonproduction_protocol_cannot_write_pointer_or_manifest(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark["config"]["max_length"] = 128
    benchmark_path.write_text(json.dumps(benchmark), encoding="utf-8")
    pointer = tmp_path / "pointer.txt"
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=pointer,
    )
    assert result["passed"] is False
    assert result["policy_mode"] == "research"
    assert "production_protocol_violation:max_length" in result["policy_blockers"]
    assert not pointer.exists()
    assert not (adapter.parent / PROMOTION_FILENAME).exists()


def test_stricter_production_thresholds_can_still_promote(tmp_path: Path) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
        threshold_overrides={"min_verified_gain": 0.50},
    )
    assert result["passed"] is True
    assert result["policy_mode"] == "production"
    assert result["production_eligible"] is True
    assert validate_promoted_adapter(adapter) is not None


def test_runtime_attestation_is_shared_fail_closed_and_revision_bound(tmp_path: Path) -> None:
    candidate = _make_adapter(
        tmp_path / "artifacts" / "general_intelligence_candidate_unpromoted",
        b"candidate",
    )
    assert adapter_activation_kind(candidate) is None
    with pytest.raises(ValueError, match="not eligible"):
        attest_adapter_for_runtime(candidate)

    benchmark_path, promoted, curriculum_manifest = _make_bound_gate_fixture(
        tmp_path / "promoted"
    )
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=promoted,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "promoted-pointer.txt",
    )
    assert result["passed"] is True
    matching_snapshot = (
        tmp_path
        / "promoted"
        / "model-cache"
        / "models--Qwen--test"
        / "snapshots"
        / "revision-abc"
    )
    attestation = attest_adapter_for_runtime(
        promoted,
        resolved_base_model=matching_snapshot,
    )
    assert attestation["activation_kind"] == "promoted"
    assert attestation["base_revision_status"] == "verified_snapshot"
    assert attestation["promotion_schema"].endswith("-v4")

    wrong_snapshot = tmp_path / "model-cache" / "snapshots" / "wrong-revision"
    wrong_snapshot.mkdir(parents=True)
    with pytest.raises(ValueError, match="revision"):
        attest_adapter_for_runtime(promoted, resolved_base_model=wrong_snapshot)

    unrelated_local_copy = tmp_path / "packaged-base-without-provenance"
    unrelated_local_copy.mkdir()
    with pytest.raises(ValueError, match="identity"):
        attest_adapter_for_runtime(promoted, resolved_base_model=unrelated_local_copy)

    legacy = _make_adapter(
        tmp_path / "artifacts" / "qwen_supermix_enhanced_v28_legacy",
        b"legacy",
    )
    assert adapter_activation_kind(legacy) is None
    with pytest.raises(ValueError, match="not eligible"):
        attest_adapter_for_runtime(legacy)


def test_attestation_keeps_receipt_hash_when_files_swap_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "pointer.txt",
    )
    assert result["passed"] is True
    original_sha256 = sha256_file(adapter / "adapter_model.safetensors")
    original_validate = promotion_module.validate_promoted_adapter

    def validate_then_swap(candidate: Path | str) -> dict[str, object] | None:
        manifest = original_validate(candidate)
        (adapter / "adapter_model.safetensors").write_bytes(b"swapped-after-validation")
        return manifest

    monkeypatch.setattr(promotion_module, "validate_promoted_adapter", validate_then_swap)
    attestation = promotion_module.attest_adapter_for_runtime(adapter)

    assert attestation["adapter_sha256"] == original_sha256
    assert sha256_file(adapter / "adapter_model.safetensors") != original_sha256


def test_gate_rejects_benchmark_for_arbitrary_adapter(tmp_path: Path) -> None:
    benchmark_path, _bound_adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    arbitrary_adapter = _make_adapter(tmp_path / "arbitrary", b"different-weights")
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=arbitrary_adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "arbitrary-pointer.txt",
    )
    assert result["passed"] is False
    assert "benchmark_adapter_path_mismatch" in result["blockers"]
    assert "benchmark_adapter_weights_mismatch" in result["blockers"]
    assert not (arbitrary_adapter.parent / PROMOTION_FILENAME).exists()


def test_gate_rejects_adapter_aba_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path, adapter, curriculum_manifest = _make_bound_gate_fixture(tmp_path)
    weights_path = adapter / "adapter_model.safetensors"
    benchmark_weights = b"weights-v1"
    arbitrary_weights = b"arbitrary-weights-aba"
    weights_path.write_bytes(arbitrary_weights)

    original_sha256_file = promotion_gate.sha256_file
    weights_hash_calls = 0

    def aba_sha256_file(path: Path | str) -> str:
        nonlocal weights_hash_calls
        candidate = Path(path).resolve()
        digest = original_sha256_file(candidate)
        if candidate == weights_path.resolve():
            weights_hash_calls += 1
            if weights_hash_calls == 1:
                weights_path.write_bytes(benchmark_weights)
            elif weights_hash_calls == 2:
                weights_path.write_bytes(arbitrary_weights)
        return digest

    monkeypatch.setattr(promotion_gate, "sha256_file", aba_sha256_file)
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "aba-pointer.txt",
    )

    assert result["passed"] is False
    assert "benchmark_adapter_weights_mismatch" in result["blockers"]
    assert "adapter_changed_during_gate" in result["blockers"]
    assert weights_path.read_bytes() == arbitrary_weights
    assert not (adapter.parent / PROMOTION_FILENAME).exists()


def test_old_weak_receipt_schema_cannot_validate(tmp_path: Path) -> None:
    artifact = tmp_path / "candidate"
    adapter = _make_adapter(artifact, b"weights-v1")
    gate = {
        "schema": "supermix-qwen-general-promotion-gate-v1",
        "passed": True,
        "base_model": "Qwen/test",
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
    }
    gate_path = artifact / GATE_FILENAME
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    manifest = {
        "schema": "supermix-qwen-adapter-promotion-v1",
        "passed": True,
        "base_model": "Qwen/test",
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
        "gate_file": GATE_FILENAME,
        "gate_sha256": sha256_file(gate_path),
    }
    (artifact / PROMOTION_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    assert validate_promoted_adapter(adapter) is None
