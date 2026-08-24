from __future__ import annotations

# ruff: noqa: E402

import copy
import sys
import json
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

import qwen_supermix_pipeline as qp
import run_research_baseline as baseline
from qwen_adapter_promotion import BENCHMARK_SCHEMA_VERSION, sha256_file
from qwen_paired_evidence import (
    PAIRED_EVIDENCE_SCHEMA_VERSION,
    build_paired_evidence,
    paired_evidence_sha256,
)
from run_research_baseline import (
    _load_reusable_reference,
    _resolve_base_model_for_evaluation,
    _stratified_eval_sample,
)


BASE_MODEL = "Qwen/test"
BASE_REVISION = "revision-abc"
CURRICULUM_PROVENANCE = {
    "curriculum_manifest_sha256": "a" * 64,
    "curriculum_eval_sha256": "b" * 64,
}
CODE_HASHES = {
    "evaluator": {"evaluator.py": "c" * 64},
    "verifier": {"verifier.py": "d" * 64},
    "policy": {"policy.py": "e" * 64},
}


def _pair(family: str, index: int) -> qp.ChatPair:
    return qp.ChatPair(
        user=f"prompt-{family}-{index}",
        assistant="answer",
        source="test",
        metadata={
            "example_id": f"eval-{family}-{index}",
            "template_id": f"eval.{family}.v1",
            "split_group": f"eval:{family}",
            "problem_family": family,
        },
    )


def _reference_payload(
    *,
    selected_eval: Path,
    samples_per_family: int | None,
    max_eval_samples: int,
    max_new_tokens: int,
) -> dict[str, object]:
    resolved_base_model = selected_eval.parent / "model-cache" / "snapshots" / BASE_REVISION
    resolved_base_model.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_REVISION,
        "resolved_base_model_path": str(resolved_base_model.resolve()),
        "adapter_dir": "",
        "reference_adapter_dir": "",
        "seed": 52,
        "max_eval_samples": max_eval_samples,
        "max_length": 256,
        "max_new_tokens": max_new_tokens,
    }
    if samples_per_family is not None:
        config["samples_per_family"] = samples_per_family
    return {
        "schema": BENCHMARK_SCHEMA_VERSION,
        "config": config,
        "provenance": {
            "base_model": BASE_MODEL,
            "base_model_revision": BASE_REVISION,
            "selected_eval_sha256": sha256_file(selected_eval),
            **CURRICULUM_PROVENANCE,
            "adapter_sha256": "",
            "adapter_config_sha256": "",
            "code_hashes": CODE_HASHES,
            "verifier_schema": qp._VERIFIER_SCHEMA,
        },
        "base": {"verified_accuracy": 0.25, "eval_loss": 2.0},
        "artifacts": {},
    }


def test_stratified_eval_sampling_is_balanced_and_deterministic() -> None:
    pairs = [*(_pair("math", i) for i in range(5)), *(_pair("science", i) for i in range(4))]
    first = _stratified_eval_sample(pairs, samples_per_family=2, seed=52)
    second = _stratified_eval_sample(pairs, samples_per_family=2, seed=52)
    assert first == second
    assert len(first) == 4
    assert sum(pair.metadata["problem_family"] == "math" for pair in first) == 2
    assert sum(pair.metadata["problem_family"] == "science" for pair in first) == 2


def test_stratified_eval_sampling_keeps_small_families() -> None:
    pairs = [_pair("rare", 0), *(_pair("common", i) for i in range(6))]
    sampled = _stratified_eval_sample(pairs, samples_per_family=3, seed=7)
    assert len(sampled) == 4
    assert any(pair.metadata["problem_family"] == "rare" for pair in sampled)


def test_sample_comparison_preserves_trusted_eval_identity() -> None:
    identity = {
        "example_id": "eval-1",
        "template_id": "eval.math.v1",
        "split_group": "eval:math",
        "problem_family": "math",
    }
    shared = {
        "sample_index": 0,
        "source": "held-out",
        "prompt_signature": "signature",
        "prompt_complexity": 0.4,
        "user": "What is two plus two?",
        "reference": "4",
        "loss": 1.0,
        "token_f1": 0.0,
        "char_similarity": 0.0,
        "gen_seconds": 0.1,
        "generated_tokens": 1,
        **identity,
    }
    base = {**shared, "prediction": "5"}
    tuned = {**shared, "prediction": "4", "token_f1": 1.0, "char_similarity": 1.0}
    comparison = qp.build_benchmark_sample_comparison([base], [tuned])
    assert len(comparison) == 1
    assert {key: comparison[0][key] for key in identity} == identity


def test_base_model_resolution_is_bound_to_local_snapshot_revision(tmp_path: Path) -> None:
    snapshot = tmp_path / "models--Qwen--test" / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    resolved, revision = _resolve_base_model_for_evaluation(str(snapshot), BASE_REVISION)
    assert Path(resolved).samefile(snapshot)
    assert revision == BASE_REVISION

    with pytest.raises(ValueError, match="does not match"):
        _resolve_base_model_for_evaluation(str(snapshot), "different-revision")

    ordinary_directory = tmp_path / "mutable-model"
    ordinary_directory.mkdir()
    with pytest.raises(ValueError, match="snapshots"):
        _resolve_base_model_for_evaluation(str(ordinary_directory), BASE_REVISION)


def test_peft_loader_preserves_base_mutating_initializers() -> None:
    class Config:
        init_lora_weights = "pissa_niter_4"

    config = Config()
    assert qp._disable_peft_init_for_weight_load(config) == ""
    assert config.init_lora_weights == "pissa_niter_4"

    config.init_lora_weights = "corda"
    assert qp._disable_peft_init_for_weight_load(config) == ""
    assert config.init_lora_weights == "corda"


def test_peft_loader_disables_ordinary_initialization_before_weight_load() -> None:
    class Config:
        init_lora_weights = True

    config = Config()
    assert qp._disable_peft_init_for_weight_load(config) == "True"
    assert config.init_lora_weights is False


def test_reusable_reference_requires_identical_eval_and_decode_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "base"
    run_dir.mkdir()
    selected_eval = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(selected_eval, [_pair("math", 1), _pair("science", 2)])
    benchmark_path = run_dir / "benchmark_results.json"
    benchmark_path.write_text(
        json.dumps(
            _reference_payload(
                selected_eval=selected_eval,
                samples_per_family=1,
                max_eval_samples=0,
                max_new_tokens=32,
            )
        ),
        encoding="utf-8",
    )
    metrics, samples, digest = _load_reusable_reference(
        benchmark_path,
        selected_eval_path=selected_eval,
        seed=52,
        samples_per_family=1,
        max_eval_samples=0,
        max_length=256,
        max_new_tokens=32,
        base_model=BASE_MODEL,
        base_model_revision=BASE_REVISION,
        curriculum_provenance=CURRICULUM_PROVENANCE,
        code_hashes=CODE_HASHES,
    )
    assert metrics["verified_accuracy"] == 0.25
    assert samples == []
    assert len(digest) == 64

    with pytest.raises(ValueError, match="max_new_tokens"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=1,
            max_eval_samples=0,
            max_length=256,
            max_new_tokens=33,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )


def test_reusable_reference_binds_detailed_base_samples(tmp_path: Path) -> None:
    run_dir = tmp_path / "sample-bound-base"
    run_dir.mkdir()
    selected_eval = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(selected_eval, [_pair("math", 1)])
    base_samples_path = run_dir / "base_samples.jsonl"
    qp.save_jsonl_records(base_samples_path, [{"sample_index": 0, "prediction": "answer"}])
    sample_sha = sha256_file(base_samples_path)
    payload = _reference_payload(
        selected_eval=selected_eval,
        samples_per_family=0,
        max_eval_samples=1,
        max_new_tokens=48,
    )
    payload["artifacts"] = {"base_samples_jsonl": str(base_samples_path)}
    payload["artifact_hashes"] = {"base_samples_sha256": sample_sha}
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    provenance["base_samples_sha256"] = sample_sha
    benchmark_path = run_dir / "benchmark_results.json"
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    _, samples, _ = _load_reusable_reference(
        benchmark_path,
        selected_eval_path=selected_eval,
        seed=52,
        samples_per_family=0,
        max_eval_samples=1,
        max_length=256,
        max_new_tokens=48,
        base_model=BASE_MODEL,
        base_model_revision=BASE_REVISION,
        curriculum_provenance=CURRICULUM_PROVENANCE,
        code_hashes=CODE_HASHES,
    )
    assert samples == [{"sample_index": 0, "prediction": "answer"}]

    base_samples_path.write_text('{"sample_index": 0, "prediction": "tampered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=0,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )


def test_reusable_reference_accepts_legacy_disabled_family_sampling(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy-base"
    run_dir.mkdir()
    selected_eval = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(selected_eval, [_pair("math", 1)])
    benchmark_path = run_dir / "benchmark_results.json"
    payload = _reference_payload(
        selected_eval=selected_eval,
        samples_per_family=None,
        max_eval_samples=1,
        max_new_tokens=48,
    )
    base_metrics = payload["base"]
    assert isinstance(base_metrics, dict)
    base_metrics["verified_accuracy"] = 0.0
    benchmark_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    metrics, _, _ = _load_reusable_reference(
        benchmark_path,
        selected_eval_path=selected_eval,
        seed=52,
        samples_per_family=0,
        max_eval_samples=1,
        max_length=256,
        max_new_tokens=48,
        base_model=BASE_MODEL,
        base_model_revision=BASE_REVISION,
        curriculum_provenance=CURRICULUM_PROVENANCE,
        code_hashes=CODE_HASHES,
    )
    assert metrics["verified_accuracy"] == 0.0

    with pytest.raises(ValueError, match="samples_per_family"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=1,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )


def test_reusable_reference_rejects_base_revision_or_adapter_binding(tmp_path: Path) -> None:
    run_dir = tmp_path / "bound-base"
    run_dir.mkdir()
    selected_eval = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(selected_eval, [_pair("math", 1)])
    benchmark_path = run_dir / "benchmark_results.json"
    payload = _reference_payload(
        selected_eval=selected_eval,
        samples_per_family=0,
        max_eval_samples=1,
        max_new_tokens=48,
    )
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="base_model_revision"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=0,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision="different-revision",
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )

    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    provenance["adapter_sha256"] = "e" * 64
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpectedly binds an adapter"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=0,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )


def test_reusable_reference_requires_claimed_immutable_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "snapshot-bound-base"
    run_dir.mkdir()
    selected_eval = run_dir / "eval_pairs.jsonl"
    qp.save_jsonl(selected_eval, [_pair("math", 1)])
    benchmark_path = run_dir / "benchmark_results.json"
    payload = _reference_payload(
        selected_eval=selected_eval,
        samples_per_family=0,
        max_eval_samples=1,
        max_new_tokens=48,
    )
    config = payload["config"]
    assert isinstance(config, dict)

    config.pop("resolved_base_model_path")
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="resolved_base_model_path"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=0,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )

    wrong_snapshot = run_dir / "other-cache" / "snapshots" / "different-revision"
    wrong_snapshot.mkdir(parents=True)
    config["resolved_base_model_path"] = str(wrong_snapshot.resolve())
    benchmark_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="claimed immutable snapshot"):
        _load_reusable_reference(
            benchmark_path,
            selected_eval_path=selected_eval,
            seed=52,
            samples_per_family=0,
            max_eval_samples=1,
            max_length=256,
            max_new_tokens=48,
            base_model=BASE_MODEL,
            base_model_revision=BASE_REVISION,
            curriculum_provenance=CURRICULUM_PROVENANCE,
            code_hashes=CODE_HASHES,
        )


def test_baseline_embeds_content_bound_paired_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "models--Qwen--test" / "snapshots" / BASE_REVISION
    snapshot.mkdir(parents=True)
    adapter = tmp_path / "candidate" / "adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"candidate")
    eval_pairs = [
        qp.ChatPair(
            user=f"Compute case {index}.",
            assistant=str(index + 2),
            source="held-out",
            metadata={
                "verifier_schema": qp._VERIFIER_SCHEMA,
                "verifier_type": "integer",
                "expected_answer": str(index + 2),
                "absolute_tolerance": "0",
                "example_id": f"eval-{index}",
                "template_id": f"eval.math.{index}",
                "split_group": f"eval:math:{index}",
                "problem_family": "math",
            },
        )
        for index in range(2)
    ]
    eval_source = tmp_path / "held_out.jsonl"
    qp.save_jsonl(eval_source, eval_pairs)

    def samples(predictions: tuple[str, ...], loss: float) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for index, pair in enumerate(eval_pairs):
            rows.append(
                {
                    "sample_index": index,
                    "source": pair.source,
                    "user": pair.user,
                    "reference": pair.assistant,
                    "prediction": predictions[index],
                    "loss": loss,
                    "generated_tokens": 1,
                    "generation_cap": 16,
                    "generation_cap_hit": False,
                    **{
                        key: str(pair.metadata[key])
                        for key in (
                            "example_id",
                            "template_id",
                            "split_group",
                            "problem_family",
                        )
                    },
                }
            )
        return rows

    base_samples = samples(("wrong", "wrong"), 1.2)
    tuned_samples = samples(("2", "3"), 0.8)
    provisional = build_paired_evidence(
        base_samples,
        tuned_samples,
        eval_pairs,
        artifact_hashes={
            "base_samples_sha256": "a" * 64,
            "tuned_samples_sha256": "b" * 64,
            "sample_comparison_sha256": "c" * 64,
        },
        bootstrap_seed=9,
        bootstrap_resamples=100,
    )
    recomputed = provisional["recomputed_metrics"]
    assert isinstance(recomputed, dict)

    def fake_evaluate_model_detailed(*, adapter_dir=None, **_kwargs):
        if adapter_dir is None:
            return dict(recomputed["base"]), copy.deepcopy(base_samples)
        return dict(recomputed["tuned"]), copy.deepcopy(tuned_samples)

    monkeypatch.setattr(qp, "evaluate_model_detailed", fake_evaluate_model_detailed)
    monkeypatch.setattr(qp, "plot_benchmark", lambda *_args, **_kwargs: None)
    output_root = tmp_path / "benchmarks"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_research_baseline.py",
            "--base_model",
            str(snapshot),
            "--adapter_dir",
            str(adapter),
            "--eval_jsonl",
            str(eval_source),
            "--output_root",
            str(output_root),
            "--run_name",
            "paired",
            "--max_new_tokens",
            "16",
            "--paired_bootstrap_seed",
            "9",
            "--paired_bootstrap_resamples",
            "100",
        ],
    )
    baseline.main()

    benchmark_path = output_root / "paired" / "benchmark_results.json"
    result = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert result["schema"] == BENCHMARK_SCHEMA_VERSION
    assert result["paired_evidence"]["schema"] == PAIRED_EVIDENCE_SCHEMA_VERSION
    assert result["provenance"]["paired_evidence_schema"] == PAIRED_EVIDENCE_SCHEMA_VERSION
    assert result["provenance"]["paired_evidence_sha256"] == paired_evidence_sha256(
        result["paired_evidence"]
    )
    for artifact_key, hash_key in (
        ("base_samples_jsonl", "base_samples_sha256"),
        ("tuned_samples_jsonl", "tuned_samples_sha256"),
        ("sample_comparison_jsonl", "sample_comparison_sha256"),
    ):
        artifact_path = Path(result["artifacts"][artifact_key])
        assert result["artifact_hashes"][hash_key] == sha256_file(artifact_path)
        assert result["provenance"][hash_key] == sha256_file(artifact_path)
