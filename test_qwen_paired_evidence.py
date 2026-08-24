from __future__ import annotations

# ruff: noqa: E402

import copy
import sys
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

from qwen_paired_evidence import (
    PAIRED_EVIDENCE_SCHEMA_VERSION,
    build_paired_evidence,
    derive_sample_comparison,
    deterministic_eval_selection,
    paired_evidence_sha256,
    recompute_and_validate_paired_evidence,
    validate_sample_comparison,
    validate_reported_metrics,
)
from qwen_supermix_pipeline import ChatPair, build_benchmark_sample_comparison
from run_research_baseline import _stratified_eval_sample
from verifiable_reasoning import VERIFIER_SCHEMA_VERSION


ARTIFACT_HASHES = {
    "base_samples_sha256": "a" * 64,
    "tuned_samples_sha256": "b" * 64,
    "sample_comparison_sha256": "c" * 64,
}


def _eval_row(index: int, *, expected: int, family: str, template_id: str) -> dict[str, object]:
    return {
        "user": f"Compute held-out case {index}.",
        "assistant": str(expected),
        "source": "paired-eval",
        "metadata": {
            "verifier_schema": VERIFIER_SCHEMA_VERSION,
            "verifier_type": "integer",
            "expected_answer": str(expected),
            "absolute_tolerance": "0",
            "example_id": f"eval-{index}",
            "template_id": template_id,
            "split_group": f"eval:{template_id}",
            "problem_family": family,
        },
    }


def _sample(
    eval_row: dict[str, object],
    index: int,
    prediction: str,
    *,
    stored_correct: bool,
) -> dict[str, object]:
    metadata = eval_row["metadata"]
    assert isinstance(metadata, dict)
    return {
        "sample_index": index,
        "source": eval_row["source"],
        "user": eval_row["user"],
        "reference": eval_row["assistant"],
        "prediction": prediction,
        "example_id": metadata["example_id"],
        "template_id": metadata["template_id"],
        "split_group": metadata["split_group"],
        "problem_family": metadata["problem_family"],
        "loss": 1.0 + index / 10.0,
        "token_f1": 99.0,
        "char_similarity": 99.0,
        "generated_tokens": 4,
        "generation_cap": 16,
        "generation_cap_hit": False,
        "verified_correct": stored_correct,
    }


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    eval_rows = [
        _eval_row(0, expected=4, family="math", template_id="template-a"),
        _eval_row(1, expected=6, family="math", template_id="template-a"),
        _eval_row(2, expected=8, family="science", template_id="template-b"),
        _eval_row(3, expected=10, family="science", template_id="template-c"),
    ]
    base_predictions = ("5", "6", "0", "10")
    tuned_predictions = ("4", "7", "1", "10")
    base = [
        _sample(row, index, base_predictions[index], stored_correct=index not in {1, 3})
        for index, row in enumerate(eval_rows)
    ]
    tuned = [
        _sample(row, index, tuned_predictions[index], stored_correct=index not in {0, 3})
        for index, row in enumerate(eval_rows)
    ]
    return eval_rows, base, tuned


def test_paired_evidence_reverifies_predictions_and_records_exact_statistics() -> None:
    eval_rows, base, tuned = _fixture()
    evidence = build_paired_evidence(
        base,
        tuned,
        eval_rows,
        artifact_hashes=ARTIFACT_HASHES,
        bootstrap_seed=17,
        bootstrap_resamples=300,
    )

    assert evidence["schema"] == PAIRED_EVIDENCE_SCHEMA_VERSION
    assert evidence["artifact_hashes"] == ARTIFACT_HASHES
    assert evidence["transitions"] == {
        "samples": 4,
        "base_correct": 2,
        "tuned_correct": 2,
        "wins": 1,
        "regressions": 1,
        "both_correct": 1,
        "both_incorrect": 1,
        "ties": 2,
        "discordant_pairs": 2,
        "base_accuracy": 0.5,
        "tuned_accuracy": 0.5,
        "accuracy_delta": 0.0,
    }
    mcnemar = evidence["mcnemar_exact_one_sided"]
    assert isinstance(mcnemar, dict)
    assert mcnemar["p_value_numerator"] == "3"
    assert mcnemar["p_value_denominator"] == "4"
    assert mcnemar["p_value"] == 0.75
    identity = evidence["identity"]
    assert isinstance(identity, dict)
    assert identity["template_cluster_count"] == 3
    metrics = evidence["recomputed_metrics"]
    assert isinstance(metrics, dict)
    assert metrics["base"]["verified_accuracy"] == 0.5
    assert metrics["tuned"]["verified_accuracy"] == 0.5
    assert metrics["base"]["token_f1"] != 99.0

    repeated = build_paired_evidence(
        base,
        tuned,
        eval_rows,
        artifact_hashes=ARTIFACT_HASHES,
        bootstrap_seed=17,
        bootstrap_resamples=300,
    )
    assert repeated == evidence
    assert len(paired_evidence_sha256(evidence)) == 64
    assert (
        recompute_and_validate_paired_evidence(
            evidence,
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
        )
        == evidence
    )


def test_exact_one_sided_mcnemar_and_cluster_bounds_for_all_wins() -> None:
    eval_rows = [
        _eval_row(index, expected=index + 2, family="math", template_id=f"template-{index}")
        for index in range(3)
    ]
    base = [
        _sample(row, index, "999", stored_correct=True)
        for index, row in enumerate(eval_rows)
    ]
    tuned = [
        _sample(row, index, str(index + 2), stored_correct=False)
        for index, row in enumerate(eval_rows)
    ]
    evidence = build_paired_evidence(
        base,
        tuned,
        eval_rows,
        artifact_hashes=ARTIFACT_HASHES,
        bootstrap_seed=3,
        bootstrap_resamples=100,
    )
    mcnemar = evidence["mcnemar_exact_one_sided"]
    assert mcnemar["p_value_numerator"] == "1"
    assert mcnemar["p_value_denominator"] == "8"
    assert mcnemar["p_value"] == 0.125
    bootstrap = evidence["template_cluster_bootstrap"]
    assert bootstrap["lower_95"] == 1.0
    assert bootstrap["upper_95"] == 1.0


def test_paired_evidence_fails_closed_on_missing_cluster_or_sample_alignment() -> None:
    eval_rows, base, tuned = _fixture()
    metadata = eval_rows[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata.pop("template_id")
    with pytest.raises(ValueError, match="template_id"):
        build_paired_evidence(
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )

    eval_rows, base, tuned = _fixture()
    tuned[1]["sample_index"] = 0
    with pytest.raises(ValueError, match="duplicate sample_index"):
        build_paired_evidence(
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    "identity_field",
    ("example_id", "template_id", "split_group", "problem_family"),
)
def test_paired_evidence_requires_complete_eval_identity(identity_field: str) -> None:
    eval_rows, base, tuned = _fixture()
    metadata = eval_rows[0]["metadata"]
    assert isinstance(metadata, dict)
    metadata[identity_field] = "  "
    with pytest.raises(ValueError, match=identity_field):
        build_paired_evidence(
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )


def test_paired_evidence_requires_unique_example_ids() -> None:
    eval_rows, base, tuned = _fixture()
    first_metadata = eval_rows[0]["metadata"]
    second_metadata = eval_rows[1]["metadata"]
    assert isinstance(first_metadata, dict) and isinstance(second_metadata, dict)
    second_metadata["example_id"] = first_metadata["example_id"]
    with pytest.raises(ValueError, match="duplicate example_id"):
        build_paired_evidence(
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )


def test_deterministic_selection_matches_baseline_algorithm_and_order() -> None:
    eval_rows = [
        _eval_row(
            index,
            expected=index + 10,
            family=("math" if index % 2 == 0 else "science"),
            template_id=f"template-{index}",
        )
        for index in range(10)
    ]
    chat_pairs = [
        ChatPair(
            user=str(row["user"]),
            assistant=str(row["assistant"]),
            source=str(row["source"]),
            metadata=dict(row["metadata"]),
        )
        for row in eval_rows
    ]
    baseline = _stratified_eval_sample(chat_pairs, samples_per_family=2, seed=77)
    selected = deterministic_eval_selection(
        eval_rows,
        seed=77,
        samples_per_family=2,
        max_eval_samples=0,
    )
    assert [row["metadata"]["example_id"] for row in selected] == [
        pair.metadata["example_id"] for pair in baseline
    ]


def test_comparison_artifact_must_be_exact_complete_derivation() -> None:
    _eval_rows, base, tuned = _fixture()
    expected = derive_sample_comparison(base, tuned)
    assert expected == build_benchmark_sample_comparison(base, tuned)
    validate_sample_comparison(expected, base, tuned)

    for tampered in (
        expected[:-1],
        [*expected, dict(expected[0])],
        [{**expected[0], "junk": True}, *expected[1:]],
    ):
        with pytest.raises(ValueError, match="exact complete"):
            validate_sample_comparison(tampered, base, tuned)

def test_paired_evidence_rejects_identity_or_payload_tampering() -> None:
    eval_rows, base, tuned = _fixture()
    tuned[0]["user"] = "different prompt"
    with pytest.raises(ValueError, match="trusted user identity"):
        build_paired_evidence(
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
            bootstrap_seed=1,
            bootstrap_resamples=100,
        )

    eval_rows, base, tuned = _fixture()
    evidence = build_paired_evidence(
        base,
        tuned,
        eval_rows,
        artifact_hashes=ARTIFACT_HASHES,
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )
    tampered = copy.deepcopy(evidence)
    tampered["transitions"]["wins"] = 4
    with pytest.raises(ValueError, match="does not match recomputed"):
        recompute_and_validate_paired_evidence(
            tampered,
            base,
            tuned,
            eval_rows,
            artifact_hashes=ARTIFACT_HASHES,
        )


def test_reported_metrics_must_match_recomputed_samples() -> None:
    eval_rows, base, tuned = _fixture()
    evidence = build_paired_evidence(
        base,
        tuned,
        eval_rows,
        artifact_hashes=ARTIFACT_HASHES,
        bootstrap_seed=1,
        bootstrap_resamples=100,
    )
    metrics = evidence["recomputed_metrics"]
    reported = dict(metrics["base"])
    validate_reported_metrics(reported, metrics["base"], side="base")
    reported["verified_accuracy"] = 0.75
    with pytest.raises(ValueError, match="verified_accuracy"):
        validate_reported_metrics(reported, metrics["base"], side="base")
