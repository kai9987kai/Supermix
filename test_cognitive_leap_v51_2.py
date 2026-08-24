from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest
import torch

from source import run_cognitive_leap_v51_2 as runner
from source.benchmark_cognitive_leap_ultra_v51 import (
    derive_chained_targets,
    make_chained_task,
    make_chained_task_with_metadata,
    operation_family_tags,
)


def test_generator_metadata_preserves_golden_canonical_cohort() -> None:
    x, y, metadata = make_chained_task_with_metadata(32, 52)
    wrapper_x, wrapper_y = make_chained_task(32, 52)
    assert torch.equal(x, wrapper_x)
    assert torch.equal(y, wrapper_y)
    assert metadata["starts"].shape == (32,)
    assert metadata["op_types"].shape == (32, 4)
    assert metadata["operands"].shape == (32, 4)

    reconstructed = metadata["starts"].clone()
    for offset in range(4):
        operands = metadata["operands"][:, offset]
        operations = metadata["op_types"][:, offset]
        reconstructed = torch.where(
            operations.eq(0),
            (reconstructed + operands) % 10,
            torch.where(
                operations.eq(1),
                (reconstructed * operands) % 10,
                (reconstructed - operands) % 10,
            ),
        )
    assert torch.equal(y, reconstructed)
    assert torch.equal(
        y,
        derive_chained_targets(
            metadata["starts"],
            metadata["op_types"],
            metadata["operands"],
        ),
    )
    assert operation_family_tags(metadata["op_types"][0])[0].startswith("first_")

    cohort = runner.build_cohort([52], 32, cohort_role="development")
    assert cohort["dataset_sha256"] == (
        "b5884fdbbcc5600a6b98b7701d91e3a877ba686f9fafdf1dcefb339f56e0ea2f"
    )


@pytest.mark.parametrize(
    "seeds,n,role",
    [([], 1, "development"), ([1, 1], 1, "development"), ([1], 0, "final"), ([1], 1, "training")],
)
def test_cohort_specification_rejects_ambiguous_inputs(
    seeds: list[int],
    n: int,
    role: str,
) -> None:
    with pytest.raises(ValueError):
        runner.cohort_specification(seeds, n, cohort_role=role)


def test_canonical_json_and_state_mixing_fail_closed() -> None:
    with pytest.raises(ValueError):
        runner.canonical_json_bytes({"bad": math.nan})
    one = {"float": torch.tensor([1.0]), "count": torch.tensor([1])}
    two = {"float": torch.tensor([3.0]), "count": torch.tensor([1])}
    mixed = runner.average_states((one, two), (0.25, 0.75))
    assert torch.equal(mixed["float"], torch.tensor([2.5]))
    with pytest.raises(ValueError):
        runner.average_states((one, two), [])
    with pytest.raises(ValueError):
        runner.average_states((one, two), (math.nan, 1.0))
    with pytest.raises(ValueError):
        runner.average_states((one, {**two, "count": torch.tensor([2])}))
    with pytest.raises(ValueError):
        runner.average_states((one, {**two, "float": torch.tensor([math.inf])}))


def test_strict_json_file_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"value":1,"value":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        runner.load_json_strict(path)


def test_evaluation_profile_is_fixed_to_two_exact_comparators() -> None:
    profile = runner.canonical_evaluation_profile()
    assert profile["release_baseline"]["sha256"] == (
        "664b1779452fe1482389413004d8bce3369f6d8ee15ab8c2c891dc5e382ebae4"
    )
    assert profile["prior_candidate"]["sha256"] == (
        "c627d905951fbfefa8155a9aae064d04fcc574cb8464f08fc716947422de06cb"
    )
    assert profile["prior_candidate"]["status"] == "unpromoted_prior_candidate"
    assert profile["final"]["seeds"] == list(runner.FINAL_SEEDS)
    assert profile["final"]["samples_per_seed"] == 2_000
    assert profile["final"]["overall_gate"] == "logical_and"
    assert profile["final"]["release_continuity_criteria"] == runner.CRITERIA
    assert profile["final"]["prior_candidate_superiority_criteria"] == (
        runner.PRIOR_CANDIDATE_CRITERIA
    )
    assert runner.canonical_evaluation_profile_sha256() == runner.sha256_bytes(
        runner.canonical_json_bytes(profile)
    )
    assert runner.canonical_evaluation_profile_sha256() == (
        "3a018d1b9cde5d59c0431f0323a46993d71806604753e459200649a024332bbd"
    )


def test_v51_2_code_bindings_use_only_three_way_receipt_validator() -> None:
    bindings = runner.current_code_bindings()

    assert "source/cognitive_leap_three_way_receipt.py" in bindings
    assert "source/cognitive_leap_receipt.py" not in bindings
    assert not hasattr(runner, "PREDICTION_ARTIFACT_SCHEMA")
    assert not hasattr(runner, "write_final_prediction_artifact")


def test_protocol_profile_rejects_relaxed_gate_or_shortened_holdout() -> None:
    profile = runner.canonical_evaluation_profile()
    protocol = {
        "evaluation_profile": profile,
        "evaluation_profile_sha256": runner.canonical_evaluation_profile_sha256(),
        "claim_scope": dict(runner.CLAIM_SCOPE),
        "authority": dict(runner.AUTHORITY),
        "authentication": "none",
        "integrity_status": "content_bound_not_authenticated",
        "trusted_timestamp": False,
        "criteria": dict(runner.CRITERIA),
        "prior_candidate_criteria": dict(runner.PRIOR_CANDIDATE_CRITERIA),
        "training": dict(profile["training"]),
        "development": {
            "seeds": list(runner.DEV_SEEDS),
            "samples_per_seed": 2_000,
            "soup_groups": [list(group) for group in runner.SOUP_GROUPS],
            "baseline_blend_alphas": list(runner.BASE_BLEND_ALPHAS),
            "selection_order": list(runner.SELECTION_ORDER),
            "criteria": dict(runner.DEVELOPMENT_CRITERIA),
            "prior_candidate_criteria": dict(runner.PRIOR_CANDIDATE_CRITERIA),
        },
        "final": {
            "seeds": list(runner.FINAL_SEEDS),
            "samples_per_seed": 2_000,
            "single_use": True,
        },
        "baseline": dict(profile["release_baseline"]),
        "prior_candidate": dict(profile["prior_candidate"]),
    }
    runner.validate_canonical_evaluation_profile(protocol)

    relaxed = json.loads(json.dumps(protocol))
    relaxed["criteria"]["minimum_accuracy_gain"] = -1.0
    with pytest.raises(ValueError, match="criteria"):
        runner.validate_canonical_evaluation_profile(relaxed)

    shortened = json.loads(json.dumps(protocol))
    shortened["final"]["seeds"] = shortened["final"]["seeds"][:-1]
    with pytest.raises(ValueError, match="holdout"):
        runner.validate_canonical_evaluation_profile(shortened)


def test_dual_selection_requires_both_gates_and_prefers_prior_gain() -> None:
    cohort = runner.build_cohort([9], 4, cohort_role="development")
    predictions = _prediction_rows(cohort)
    release = runner.compare_predictions(
        predictions,
        predictions,
        cohort,
        runner.DEVELOPMENT_CRITERIA,
    )
    prior = runner.compare_predictions(
        predictions,
        predictions,
        cohort,
        runner.PRIOR_CANDIDATE_CRITERIA,
    )
    row = runner.dual_candidate_row(
        name="candidate",
        group=["left", "right"],
        alpha=0.25,
        release_comparison=release,
        prior_comparison=prior,
    )
    assert row["passed"] is False
    assert row["selection_score"][0] == 0
    assert set(row["comparisons"]) == {
        "release_continuity",
        "prior_candidate_superiority",
    }

    def comparison(passed: bool, delta: float) -> dict[str, object]:
        return {
            "passed": passed,
            "checks": {name: passed for name in range(6)},
            "summary": {
                "accuracy_delta": delta,
                "nonregressing_seed_count": 20,
                "nonregressing_family_count": 8,
                "nonregressing_class_count": 10,
                "mean_candidate_loss": 1.0,
            },
        }

    release_pass = comparison(True, 0.01)
    prior_pass = comparison(True, 0.001)
    release_fail = comparison(False, 0.01)
    prior_fail = comparison(False, 0.001)
    assert runner.dual_selection_score(release_pass, prior_fail)[0] == 0
    assert runner.dual_selection_score(release_fail, prior_pass)[0] == 0
    assert runner.dual_selection_score(release_pass, prior_pass)[0] == 1
    better_prior = comparison(True, 0.002)
    assert runner.dual_selection_score(release_pass, better_prior) > (
        runner.dual_selection_score(release_pass, prior_pass)
    )
    tied_rows = [
        {"name": "first", "selection_score": list(runner.dual_selection_score(release_pass, prior_pass))},
        {"name": "second", "selection_score": list(runner.dual_selection_score(release_pass, prior_pass))},
    ]
    assert max(tied_rows, key=lambda item: tuple(item["selection_score"]))[
        "name"
    ] == "first"


def test_strict_json_reader_rejects_nonfinite_constants(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite"):
        runner.load_json_strict(path)


def test_empty_groups_never_count_as_nonregressing() -> None:
    targets = torch.tensor([0, 1])
    predictions = torch.tensor([0, 1])
    evaluation = {
        "mean_loss": 0.5,
        "prediction_sha256": "a" * 64,
        "logits_sha256": "b" * 64,
        "per_example_sha256": "c" * 64,
        "seed_rows": [
            {
                "seed": 7,
                "targets": targets,
                "predictions": predictions,
                "loss_sum": 1.0,
            }
        ],
    }
    cohort = {
        "dataset_sha256": "b" * 64,
        "dataset_id": "c" * 64,
        "specification_sha256": "d" * 64,
        "schema": runner.COHORT_SCHEMA,
        "generator_schema": runner.GENERATOR_SCHEMA,
        "family_tag_schema": runner.FAMILY_TAG_SCHEMA,
        "cohort_role": "development",
        "rows": [
            {
                "seed": 7,
                "y": targets,
                "op_types": torch.zeros((2, 4), dtype=torch.long),
            }
        ],
    }
    result = runner.compare_predictions(evaluation, evaluation, cohort, runner.CRITERIA)
    assert result["summary"]["eligible_family_count"] == 2
    assert result["summary"]["nonregressing_family_count"] == 2
    assert result["checks"]["operation_family_nonregression"] is False


def test_count_derived_thresholds_are_inclusive_without_float_epsilon() -> None:
    sample_count = 2_000
    targets = torch.zeros(sample_count, dtype=torch.long)
    baseline_predictions = torch.ones(sample_count, dtype=torch.long)
    baseline_predictions[:1_000] = 0
    candidate_at_boundary = baseline_predictions.clone()
    candidate_at_boundary[990:1_000] = 1
    candidate_just_below = candidate_at_boundary.clone()
    candidate_just_below[989] = 1

    def evaluation(predictions: torch.Tensor, digest: str) -> dict[str, object]:
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
        "schema": runner.COHORT_SCHEMA,
        "generator_schema": runner.GENERATOR_SCHEMA,
        "family_tag_schema": runner.FAMILY_TAG_SCHEMA,
        "cohort_role": "development",
        "rows": [
            {
                "seed": 7,
                "y": targets,
                "op_types": torch.zeros((sample_count, 4), dtype=torch.long),
            }
        ],
    }
    criteria = {
        **runner.CRITERIA,
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
    baseline = evaluation(baseline_predictions, "f")
    boundary = runner.compare_predictions(
        baseline,
        evaluation(candidate_at_boundary, "1"),
        cohort,
        criteria,
    )
    below = runner.compare_predictions(
        baseline,
        evaluation(candidate_just_below, "2"),
        cohort,
        criteria,
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


def _prediction_rows(cohort: dict[str, object]) -> dict[str, object]:
    seed_row = cohort["rows"][0]  # type: ignore[index]
    targets = seed_row["y"]  # type: ignore[index]
    logits = torch.full((int(targets.numel()), 10), -1.0)
    logits[torch.arange(targets.numel()), targets] = 1.0
    return {
        "mean_loss": 0.0,
        "prediction_sha256": "e" * 64,
        "logits_sha256": "f" * 64,
        "per_example_sha256": "a" * 64,
        "seed_rows": [
            {
                "seed": seed_row["seed"],  # type: ignore[index]
                "targets": targets,
                "predictions": targets.clone(),
                "logits": logits,
                "loss_sum": 0.0,
            }
        ],
    }


def test_three_way_final_artifact_matches_validator_wire_contract(
    tmp_path: Path,
) -> None:
    cohort = runner.build_cohort([9], 2, cohort_role="final")
    predictions = _prediction_rows(cohort)
    profile_sha256 = runner.canonical_evaluation_profile_sha256()
    first = runner.write_three_way_final_prediction_artifact(
        tmp_path / "first-three-way.jsonl.gz",
        predictions,
        predictions,
        predictions,
        cohort,
        evaluation_profile_sha256=profile_sha256,
    )
    second = runner.write_three_way_final_prediction_artifact(
        tmp_path / "second-three-way.jsonl.gz",
        predictions,
        predictions,
        predictions,
        cohort,
        evaluation_profile_sha256=profile_sha256,
    )

    assert first["schema"] == runner.THREE_WAY_PREDICTION_ARTIFACT_SCHEMA
    assert first["format"] == "deterministic_gzip_jsonl"
    assert first["evaluation_profile_sha256"] == profile_sha256
    assert first["sha256"] == second["sha256"]
    assert first["uncompressed_sha256"] == second["uncompressed_sha256"]
    assert first["gzip_mtime"] == 0
    with gzip.open(
        tmp_path / "first-three-way.jsonl.gz", "rt", encoding="utf-8"
    ) as handle:
        rows = [json.loads(line) for line in handle]
    assert len(rows) == 2
    assert set(rows[0]) == {
        "example_id",
        "dataset_id",
        "cohort_role",
        "seed",
        "index",
        "target",
        "start",
        "op_types",
        "operands",
        "operation_family_tags",
        "release_baseline_prediction",
        "release_baseline_correct",
        "release_baseline_logits_f32le_hex",
        "prior_candidate_prediction",
        "prior_candidate_correct",
        "prior_candidate_logits_f32le_hex",
        "candidate_prediction",
        "candidate_correct",
        "candidate_logits_f32le_hex",
    }
    for model_name in ("release_baseline", "prior_candidate", "candidate"):
        assert len(bytes.fromhex(rows[0][f"{model_name}_logits_f32le_hex"])) == 40


def test_constructed_evaluator_and_local_cohort_leave_global_rng_unchanged() -> None:
    evaluator = runner.ChampionNetCognitiveLeapUltraExpert()
    before = torch.get_rng_state().clone()
    cohort = runner.build_cohort([123], 2, cohort_role="development")
    runner.predict_cohort(evaluator, cohort, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), before)


def test_dirty_development_protocol_cannot_touch_final_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "load_and_validate_protocol",
        lambda _output: {"finalization_allowed": False},
    )
    touched = False

    def forbidden(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal touched
        touched = True
        raise AssertionError("final cohort was touched")

    monkeypatch.setattr(runner, "build_cohort", forbidden)
    with pytest.raises(RuntimeError, match="dirty development mode"):
        runner.finalize_once(tmp_path, torch.device("cpu"))
    assert touched is False
    assert not (tmp_path / "finalization.started.json").exists()


def test_train_selection_requests_only_development_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {
        "protocol_sha256": "1" * 64,
        "evaluation_profile_sha256": runner.canonical_evaluation_profile_sha256(),
        "baseline": {"path": "baseline.pth"},
        "prior_candidate": {"path": "prior.pth"},
        "training": {"members": [{"name": "member"}]},
        "development": {
            "seeds": [9],
            "samples_per_seed": 1,
            "soup_groups": [["member"]],
            "baseline_blend_alphas": [0.25],
            "criteria": runner.DEVELOPMENT_CRITERIA,
            "prior_candidate_criteria": runner.PRIOR_CANDIDATE_CRITERIA,
        },
    }
    monkeypatch.setattr(runner, "load_and_validate_protocol", lambda _path: protocol)
    state = {"weight": torch.tensor([1.0])}
    monkeypatch.setattr(runner, "load_state", lambda _path: state)
    monkeypatch.setattr(
        runner,
        "train_member",
        lambda *_args, **_kwargs: (state, {"artifact": {}, "config": {}}),
    )
    observed_roles: list[str] = []

    def build_cohort(
        _seeds: object,
        _samples: int,
        *,
        cohort_role: str,
    ) -> dict[str, object]:
        observed_roles.append(cohort_role)
        if cohort_role == "final":
            raise AssertionError("training touched the final cohort")
        return {"dataset_sha256": "2" * 64}

    class Evaluator:
        def to(self, _device: object) -> "Evaluator":
            return self

        def load_state_dict(self, _state: object, strict: bool = True) -> None:
            assert strict is True

    monkeypatch.setattr(runner, "build_cohort", build_cohort)
    monkeypatch.setattr(runner, "ChampionNetCognitiveLeapUltraExpert", Evaluator)
    monkeypatch.setattr(runner, "predict_cohort", lambda *_args: {"predictions": True})
    monkeypatch.setattr(runner, "average_states", lambda *_args: state)
    monkeypatch.setattr(runner, "blend_with_baseline", lambda *_args: state)
    comparison = {
        "passed": False,
        "checks": {f"check_{index}": False for index in range(6)},
        "summary": {
            "accuracy_delta": 0.0,
            "nonregressing_seed_count": 0,
            "seed_count": 1,
            "nonregressing_family_count": 0,
            "nonregressing_class_count": 0,
            "mean_candidate_loss": 1.0,
        },
        "evidence": {},
    }
    monkeypatch.setattr(runner, "compare_predictions", lambda *_args: comparison)
    monkeypatch.setattr(runner, "environment_binding", lambda _device: {})

    with pytest.raises(RuntimeError, match="final cohort remains untouched"):
        runner.train_and_select(tmp_path, torch.device("cpu"))
    assert observed_roles == ["development"]


def test_single_use_sentinel_is_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "sentinel.json"
    runner.write_json_exclusive(path, {"value": 1})
    with pytest.raises(FileExistsError):
        runner.write_json_exclusive(path, {"value": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_exact_mcnemar_and_selection_digest() -> None:
    assert runner.exact_mcnemar_two_sided(5, 0) == pytest.approx(0.0625)
    selection = {"schema": "test", "passed": True}
    digest = runner.selection_digest(selection)
    selection["selection_sha256"] = digest
    assert runner.selection_digest(selection) == digest
    selection["passed"] = False
    assert runner.selection_digest(selection) != digest
