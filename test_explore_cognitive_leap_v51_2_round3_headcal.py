from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from source import explore_cognitive_leap_v51_2_round3_headcal as round3


def _comparison(passed: bool) -> dict[str, Any]:
    checks = {
        "accuracy_gain": passed,
        "paired_significance": passed,
        "mean_loss_nonregression": passed,
        "seed_nonregression": passed,
        "operation_family_nonregression": passed,
        "class_bounded_nonregression": passed,
    }
    return {
        "passed": passed,
        "checks": checks,
        "summary": {
            "accuracy_delta": 0.001 if passed else 0.0,
            "mean_candidate_loss": 1.0,
            "nonregressing_seed_count": 40 if passed else 0,
            "nonregressing_family_count": 8 if passed else 0,
            "nonregressing_class_count": 10 if passed else 0,
        },
        "evidence": {"dataset_sha256": "dev3"},
    }


def _candidate(alpha: float, passed: bool) -> dict[str, Any]:
    release = _comparison(passed)
    prior = _comparison(passed)
    row = round3.runner.dual_candidate_row(
        name=f"headcal_451-headcal_551__alpha_{alpha:.2f}",
        group=["headcal_451", "headcal_551"],
        alpha=alpha,
        release_comparison=release,
        prior_comparison=prior,
    )
    row["canonical_state_sha256"] = f"state-{alpha:.2f}"
    return row


def _fake_specification() -> dict[str, Any]:
    return {
        "specification_sha256": "spec-id",
        "parents": {
            "round1": {"protocol": {"path": "round1/protocol.json"}},
            "round2": {"specification": {"path": "round2/search_specification.json"}},
            "release_baseline": {"path": "baseline.pth"},
            "prior_candidate": {"path": "prior.pth"},
        },
        "training": {
            "objective": {
                "name": "unweighted_cross_entropy",
                "class_weights": None,
                "auxiliary_loss": False,
                "distillation": False,
            },
            "optimizer": {},
        },
        "development": {
            "release_continuity_criteria": dict(round3.runner.DEVELOPMENT_CRITERIA),
            "prior_candidate_superiority_criteria": dict(
                round3.runner.PRIOR_CANDIDATE_CRITERIA
            ),
        },
    }


def test_round3_constants_are_fresh_bounded_and_non_authoritative() -> None:
    assert round3.DEV3_SEEDS == tuple(range(61_052, 101_052, 1_000))
    assert len(round3.DEV3_SEEDS) == 40
    assert set(round3.DEV3_SEEDS).isdisjoint(round3.runner.DEV_SEEDS)
    assert set(round3.DEV3_SEEDS).isdisjoint(round3.runner.FINAL_SEEDS)
    assert round3.SAMPLES_PER_SEED == 2_000
    assert round3.BLEND_ALPHAS == (0.25, 0.50, 0.75, 1.00)
    assert round3.TRAINABLE_PARAMETER_NAMES == (
        "layers.10.bias",
        "layers.10.shared_norm.bias",
        "layers.10.decode_head.weight",
        "layers.11.weight",
    )
    assert round3.TRAINABLE_PARAMETER_COUNT == 1_310
    assert not any(round3.AUTHORITY.values())
    assert all(
        value is False for key, value in round3.CLAIM_SCOPE.items() if key != "task"
    )
    parser = round3.build_arg_parser()
    assert set(parser._option_string_actions) == {
        "-h",
        "--help",
        "--round1",
        "--round2",
        "--output-dir",
        "--torch-threads",
    }
    phase = next(action for action in parser._actions if action.dest == "phase")
    assert tuple(phase.choices) == ("run", "verify-development")


def test_build_specification_freezes_exact_headcal_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(round3, "_validate_parents", lambda *_args: {"bound": True})
    monkeypatch.setattr(round3.runner, "sha256_file", lambda _path: "code-sha")
    monkeypatch.setattr(
        round3.runner,
        "environment_binding",
        lambda _device: {"environment": "bound"},
    )
    specification = round3.build_specification(
        Path("round1"), Path("round2"), torch.device("cpu")
    )
    assert specification["execution_mode"] == "development_only_no_finalization"
    assert specification["final_cohort_access"] is False
    assert specification["training"] == {
        "initial_checkpoint_role": "prior_candidate",
        "train_size_per_member": 24_000,
        "epochs": 1,
        "batch_size": 128,
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
            "lr": 2.5e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": False,
            "capturable": False,
            "differentiable": False,
            "fused": False,
            "gradient_clip_norm": 1.0,
        },
        "trainable_parameter_names": list(round3.TRAINABLE_PARAMETER_NAMES),
        "trainable_parameter_count": 1_310,
        "model_parameter_count": 2_245_715,
        "members": [dict(value) for value in round3.MEMBER_CONFIGS],
        "member_soup_weights": [0.5, 0.5],
    }
    assert specification["development"]["seeds"] == list(round3.DEV3_SEEDS)
    assert specification["development"]["release_continuity_criteria"] == (
        round3.runner.DEVELOPMENT_CRITERIA
    )
    assert specification["development"]["prior_candidate_superiority_criteria"] == (
        round3.runner.PRIOR_CANDIDATE_CRITERIA
    )
    assert specification["specification_sha256"] == round3._specification_digest(
        specification
    )


def test_head_only_model_exposes_exact_four_trainables() -> None:
    seed_model = round3.runner.ChampionNetCognitiveLeapUltraExpert()
    prior = {
        name: value.detach().clone() for name, value in seed_model.state_dict().items()
    }
    model, trainable = round3._configure_head_only_model(prior, torch.device("cpu"))
    assert not model.training
    assert all(not module.training for module in model.modules())
    assert [
        name for name, value in model.named_parameters() if value.requires_grad
    ] == list(round3.TRAINABLE_PARAMETER_NAMES)
    assert sum(value.numel() for value in trainable) == 1_310
    assert sum(value.numel() for value in model.parameters()) == 2_245_715
    frozen_name, frozen_parameter = next(
        (name, value)
        for name, value in model.named_parameters()
        if name not in round3.TRAINABLE_PARAMETER_NAMES
    )
    with torch.no_grad():
        frozen_parameter.add_(1.0)
    with pytest.raises(ValueError, match=f"Frozen parameter changed: {frozen_name}"):
        round3._frozen_head_only_state(model, prior)


def test_candidate_grid_changes_only_frozen_headcal_tensors() -> None:
    prior = {
        "trained": torch.tensor([0.0]),
        "frozen": torch.tensor([7.0]),
        "counter": torch.tensor([3]),
    }
    one = {
        "trained": torch.tensor([2.0]),
        "frozen": torch.tensor([7.0]),
        "counter": torch.tensor([3]),
    }
    two = {
        "trained": torch.tensor([4.0]),
        "frozen": torch.tensor([7.0]),
        "counter": torch.tensor([3]),
    }
    states = round3._candidate_states(
        prior,
        {"headcal_451": one, "headcal_551": two},
    )
    assert tuple(states) == round3.BLEND_ALPHAS
    assert torch.equal(states[0.25]["trained"], torch.tensor([0.75]))
    assert torch.equal(states[1.0]["trained"], torch.tensor([3.0]))
    assert all(
        torch.equal(value["frozen"], prior["frozen"]) for value in states.values()
    )
    assert all(
        torch.equal(value["counter"], prior["counter"]) for value in states.values()
    )


def test_build_dev3_cohort_never_requests_final_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[int, ...], int, str]] = []

    def spy(seeds: Any, samples: int, *, cohort_role: str) -> dict[str, Any]:
        values = tuple(int(value) for value in seeds)
        calls.append((values, samples, cohort_role))
        assert set(values).isdisjoint(round3.runner.FINAL_SEEDS)
        return {"seeds": list(values)}

    monkeypatch.setattr(round3.runner, "build_cohort", spy)
    cohort = round3.build_dev3_cohort()
    assert cohort["seeds"] == list(round3.DEV3_SEEDS)
    assert calls == [(round3.DEV3_SEEDS, 2_000, "development")]


def test_member_training_refuses_to_generate_data_before_spec_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = False

    def forbidden(_config: Any) -> Any:
        nonlocal generated
        generated = True
        raise AssertionError("training data must not be generated")

    monkeypatch.setattr(round3, "_training_dataset", forbidden)
    with pytest.raises(ValueError, match="specification is not persisted"):
        round3.train_headcal_member(
            {},
            {},
            round3.MEMBER_CONFIGS[0],
            _fake_specification(),
            tmp_path,
            torch.device("cpu"),
        )
    assert generated is False


def _stub_run_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    passed: bool,
) -> tuple[Path, list[str]]:
    output_dir = tmp_path / ("passing" if passed else "rejected")
    specification = _fake_specification()
    events: list[str] = []
    monkeypatch.setattr(
        round3,
        "build_specification",
        lambda *_args: dict(specification),
    )

    def validate(path: Path, _device: torch.device) -> dict[str, Any]:
        assert (path / "search_specification.json").is_file()
        events.append("validated_after_spec")
        return dict(specification)

    monkeypatch.setattr(round3, "_validate_specification", validate)
    monkeypatch.setattr(round3.runner, "load_state", lambda _path: {})

    def train(
        _prior: Any,
        _binding: Any,
        config: Any,
        _spec: Any,
        member_output: Path,
        _device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        assert (member_output / "search_specification.json").is_file()
        events.append(f"trained:{config['name']}")
        return {}, {"receipt_id": config["name"], "artifact": {}}

    monkeypatch.setattr(round3, "train_headcal_member", train)
    monkeypatch.setattr(
        round3,
        "_candidate_states",
        lambda *_args: {alpha: {} for alpha in round3.BLEND_ALPHAS},
    )

    def dev3() -> dict[str, Any]:
        assert (output_dir / "search_specification.json").is_file()
        events.append("dev3_after_training")
        return {"dataset_sha256": "dev3"}

    monkeypatch.setattr(round3, "build_dev3_cohort", dev3)
    rows = [_candidate(alpha, passed) for alpha in round3.BLEND_ALPHAS]
    monkeypatch.setattr(round3, "_evaluate_candidates", lambda *_args: rows)
    monkeypatch.setattr(round3.runner, "environment_binding", lambda _device: {})
    monkeypatch.setattr(
        round3,
        "_bound_file",
        lambda path: {"path": str(path), "size_bytes": 1, "sha256": "bound"},
    )
    saved: list[str] = []

    def save(path: Path, _state: Any) -> dict[str, Any]:
        saved.append(str(path))
        return {
            "path": str(path),
            "size_bytes": 1,
            "sha256": "saved",
            "canonical_state_sha256": "state-0.25",
            "tensor_count": 0,
            "element_count": 0,
        }

    monkeypatch.setattr(round3.runner, "save_state", save)
    monkeypatch.setattr(
        round3,
        "_write_lineage",
        lambda *_args: {"path": "lineage", "sha256": "lineage"},
    )
    path = round3.run_search(
        Path("round1"),
        Path("round2"),
        output_dir,
        torch.device("cpu"),
    )
    return path, saved


@pytest.mark.parametrize("passed", [False, True])
def test_specification_precedes_work_and_candidate_is_saved_only_on_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
) -> None:
    path, saved = _stub_run_search(tmp_path, monkeypatch, passed=passed)
    receipt = round3.runner.load_json_strict(path)
    assert path.parent.joinpath("search_specification.json").is_file()
    assert receipt["passed"] is passed
    assert bool(saved) is passed
    if passed:
        assert "artifact" in receipt["selected"]
        assert "lineage" in receipt
    else:
        assert "artifact" not in receipt["selected"]
        assert "lineage" not in receipt


def _write_rejected_selection(output_dir: Path, candidate: dict[str, Any]) -> Path:
    specification_path = output_dir / "search_specification.json"
    round3.runner.write_json_atomic(specification_path, {"placeholder": True})
    selection: dict[str, Any] = {
        "schema": round3.SELECTION_SCHEMA,
        "authentication": "none",
        "trusted_timestamp": False,
        "integrity_status": "content_bound_not_authenticated",
        "authority": dict(round3.AUTHORITY),
        "claim_scope": dict(round3.CLAIM_SCOPE),
        "specification": {
            "path": str(specification_path),
            "size_bytes": specification_path.stat().st_size,
            "sha256": round3.runner.sha256_file(specification_path),
        },
        "specification_content_sha256": "spec-id",
        "development_dataset_sha256": "dev3",
        "member_receipts": {},
        "candidates": [_candidate(alpha, False) for alpha in round3.BLEND_ALPHAS],
        "selected": candidate,
        "passed": False,
        "decision": "no_development_candidate_passed",
        "environment": {},
    }
    selection["receipt_id"] = round3._content_digest(selection, "receipt_id")
    path = output_dir / "round3_development_receipt.json"
    round3.runner.write_json_atomic(path, selection)
    return path


def test_verify_development_replays_only_dev3_after_static_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "verify"
    output_dir.mkdir()
    candidate = _candidate(0.25, False)
    _write_rejected_selection(output_dir, candidate)
    specification = _fake_specification()
    monkeypatch.setattr(
        round3,
        "_validate_specification",
        lambda *_args: dict(specification),
    )
    monkeypatch.setattr(
        round3,
        "_validate_parents",
        lambda *_args: specification["parents"],
    )
    monkeypatch.setattr(round3, "_same_bound_file", lambda *_args: True)
    monkeypatch.setattr(round3.runner, "load_state", lambda _path: {})
    monkeypatch.setattr(
        round3.runner,
        "validate_environment_compatibility",
        lambda *_args: None,
    )
    monkeypatch.setattr(round3.runner, "environment_binding", lambda _device: {})
    monkeypatch.setattr(
        round3.runner,
        "state_dict_summary",
        lambda state: {"canonical_state_sha256": f"state-{float(state['alpha']):.2f}"},
    )
    monkeypatch.setattr(round3, "_load_member_evidence", lambda *_args: {})
    monkeypatch.setattr(
        round3,
        "_candidate_states",
        lambda *_args: {
            alpha: {"alpha": torch.tensor(alpha)} for alpha in round3.BLEND_ALPHAS
        },
    )
    candidates = [_candidate(alpha, False) for alpha in round3.BLEND_ALPHAS]
    monkeypatch.setattr(
        round3,
        "_evaluate_candidates",
        lambda *_args: candidates,
    )
    cohort_calls: list[tuple[tuple[int, ...], str]] = []

    def cohort_spy(seeds: Any, _samples: int, *, cohort_role: str) -> dict[str, Any]:
        values = tuple(int(value) for value in seeds)
        cohort_calls.append((values, cohort_role))
        assert set(values).isdisjoint(round3.runner.FINAL_SEEDS)
        return {"dataset_sha256": "dev3"}

    monkeypatch.setattr(round3.runner, "build_cohort", cohort_spy)
    replay_path = round3.verify_development(output_dir, torch.device("cpu"))
    replay = round3.runner.load_json_strict(replay_path)
    assert cohort_calls == [(round3.DEV3_SEEDS, "development")]
    assert replay["development_seeds"] == list(round3.DEV3_SEEDS)
    assert set(replay["development_seeds"]).isdisjoint(round3.runner.FINAL_SEEDS)
    assert replay["passed"] is False


def test_verify_rejects_bad_receipt_before_generating_any_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "bad"
    output_dir.mkdir()
    candidate = _candidate(0.25, False)
    selection_path = _write_rejected_selection(output_dir, candidate)
    selection = round3.runner.load_json_strict(selection_path)
    selection["decision"] = "tampered"
    round3.runner.write_json_atomic(selection_path, selection)
    specification = _fake_specification()
    monkeypatch.setattr(
        round3,
        "_validate_specification",
        lambda *_args: dict(specification),
    )
    monkeypatch.setattr(
        round3,
        "_validate_parents",
        lambda *_args: specification["parents"],
    )
    called = False

    def forbidden(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("cohort generation must not occur")

    monkeypatch.setattr(round3.runner, "build_cohort", forbidden)
    with pytest.raises(ValueError, match="selection contract"):
        round3.verify_development(output_dir, torch.device("cpu"))
    assert called is False
