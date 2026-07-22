import hashlib
import json
import os
import sys

import pytest
import torch


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import benchmark_progressive_auto_compute as benchmark


class _CycleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.prediction_stability_margins = []

    def forward(
        self,
        x,
        *,
        reasoning_cycles=3,
        adaptive_compute=False,
        exit_tol=0.001,
        exit_entropy_threshold=0.2,
        prediction_stability_patience=2,
        prediction_stability_tol=0.005,
        prediction_stability_margin=0.0,
    ):
        del (
            adaptive_compute,
            exit_tol,
            exit_entropy_threshold,
            prediction_stability_patience,
            prediction_stability_tol,
        )
        self.prediction_stability_margins.append(
            float(prediction_stability_margin)
        )
        cycle = int(reasoning_cycles)
        self.calls.append(cycle)
        logits = torch.zeros(len(x), 1, 10, dtype=x.dtype, device=x.device)
        logits[:, :, 0] = {1: 0.0, 3: 2.0, 8: 3.0}[cycle]
        return logits


class _BareModel:
    training = False

    def eval(self):
        return self


def _shadow_plan(*, cycles=3, forward_evaluations=2):
    return {
        "selected_reasoning_cycles": cycles,
        "forward_evaluations": forward_evaluations,
        "mutual_stability_role": "shadow_diagnostic_only",
        "rows": [
            {
                "mutual_stability_shadow": {
                    "role": "shadow_diagnostic_only",
                    "selection_enabled": False,
                }
            }
        ],
    }


def _legacy_result(
    output,
    *,
    cycles=3,
    forward_evaluations=4,
    margin=0.25,
    decision_margin=0.125,
):
    return (
        output,
        {
            "prediction_margin": margin,
            "prediction_decision_margin": decision_margin,
            "prediction_verifier_active": True,
        },
        {
            "selected_reasoning_cycles": cycles,
            "forward_evaluations": forward_evaluations,
        },
    )


def _progressive_result(
    output,
    *,
    cycles=3,
    forward_evaluations=2,
    margin=0.25,
    decision_margin=0.125,
):
    return (
        output,
        {
            "prediction_margin": margin,
            "prediction_decision_margin": decision_margin,
            "prediction_verifier_active": True,
        },
        _shadow_plan(cycles=cycles, forward_evaluations=forward_evaluations),
    )


def test_cli_defaults_use_four_held_out_seeds_and_256_requests():
    args = benchmark.build_parser().parse_args([])

    assert args.seeds == [719, 727, 733, 739]
    assert args.samples_per_seed == 64
    assert len(args.seeds) * args.samples_per_seed == 256
    assert not set(args.seeds).intersection(
        benchmark.TRAINING_AND_ORIGINAL_TEST_SEEDS
    )
    assert args.cycles == [1, 3, 8]
    assert (
        args.prediction_stability_margin
        == benchmark.DEFAULT_PREDICTION_STABILITY_MARGIN
    )
    assert (
        args.prediction_stability_rank_depth
        == benchmark.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
    )


def test_literal_legacy_v1_and_progressive_accept_same_probe_and_output():
    model = _CycleModel().eval()
    x = torch.zeros(1, 1, 4)
    settings = {
        "cycles": [1, 3, 8],
        "confidence_target": 0.8,
        "entropy_target": 0.0,
        "adaptive_compute": True,
        "prediction_stability_margin": 0.125,
    }

    legacy_output, _legacy_compute, legacy_plan = (
        benchmark.legacy_v1_auto_compute_forward(model, x, [0, 1], **settings)
    )
    progressive_output, _progressive_compute, progressive_plan = (
        benchmark.progressive_auto_compute_forward(
            model, x, [0, 1], **settings
        )
    )

    assert model.calls == [1, 3, 8, 3, 1, 3]
    # Both controllers receive identical verifier settings; only their probe
    # scheduling and accepted-output reuse differ.
    assert model.prediction_stability_margins == [0.125] * 6
    assert legacy_plan["selected_reasoning_cycles"] == 3
    assert progressive_plan["selected_reasoning_cycles"] == 3
    assert legacy_plan["forward_evaluations"] == 4
    assert progressive_plan["forward_evaluations"] == 2
    assert torch.equal(legacy_output, progressive_output)
    assert benchmark._shadow_selection_is_disabled(progressive_plan) is True


@pytest.mark.parametrize(
    ("offset", "expected_measured_calls"),
    [
        (0, ["legacy", "progressive", "progressive", "legacy"]),
        (1, ["progressive", "legacy", "legacy", "progressive"]),
    ],
)
def test_request_benchmark_counterbalances_and_counts_forward_reduction(
    monkeypatch, offset, expected_measured_calls
):
    calls = []

    def legacy_fn(_model, sample, _labels, **_kwargs):
        calls.append("legacy")
        return _legacy_result(
            torch.tensor([[[2.0, 0.0]]], device=sample.device)
        )

    def progressive_fn(_model, sample, _labels, **_kwargs):
        calls.append("progressive")
        return _progressive_result(
            torch.tensor([[[2.0, 0.0]]], device=sample.device)
        )

    ticks = iter(float(index) for index in range(20))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result = benchmark.benchmark_requests(
        _BareModel(),
        torch.zeros(2, 1, 4),
        torch.zeros(2, dtype=torch.long),
        available_labels=[0, 1],
        order_offset=offset,
        legacy_fn=legacy_fn,
        progressive_fn=progressive_fn,
    )

    assert calls[2:] == expected_measured_calls
    assert result["measurement_order"] == {
        "offset": offset,
        "legacy_then_progressive": 1,
        "progressive_then_legacy": 1,
    }
    assert result["legacy_v1"]["forward_evaluations"] == 8
    assert result["progressive"]["forward_evaluations"] == 4
    assert result["comparison"]["forward_evaluation_reduction_percent"] == 50
    assert result["comparison"]["mean_latency_reduction_percent"] == 0
    assert result["comparison"]["exact_output_tensor_disagreement_count"] == 0
    assert result["legacy_v1"]["observed_prediction_margin"]["minimum"] == 0.25
    assert result["progressive"]["observed_prediction_margin"]["minimum"] == 0.25
    assert (
        result["legacy_v1"]["observed_prediction_decision_margin"]["minimum"]
        == 0.125
    )
    assert (
        result["progressive"]["observed_prediction_decision_margin"]["minimum"]
        == 0.125
    )
    assert result["gates"]["passed"] is True


def test_request_benchmark_fails_every_exact_equivalence_gate(monkeypatch):
    def legacy_fn(_model, sample, _labels, **_kwargs):
        return _legacy_result(
            torch.tensor([[[2.0, 0.0]]], device=sample.device), cycles=3
        )

    def progressive_fn(_model, sample, _labels, **_kwargs):
        return _progressive_result(
            torch.tensor([[[0.0, 2.0]]], device=sample.device), cycles=8
        )

    ticks = iter(float(index) for index in range(20))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result = benchmark.benchmark_requests(
        _BareModel(),
        torch.zeros(1, 1, 4),
        torch.zeros(1, dtype=torch.long),
        available_labels=[0, 1],
        legacy_fn=legacy_fn,
        progressive_fn=progressive_fn,
    )
    checks = result["gates"]["checks"]

    assert result["gates"]["passed"] is False
    assert checks["exact_selected_cycle_agreement"]["passed"] is False
    assert checks["exact_prediction_agreement"]["passed"] is False
    assert checks["exact_output_tensor_agreement"]["passed"] is False
    assert result["comparison"]["maximum_absolute_logit_difference"] == 2
    assert len(result["comparison"]["mismatch_details"]) == 1


def test_shadow_diagnostic_cannot_silently_become_a_selector(monkeypatch):
    def legacy_fn(_model, sample, _labels, **_kwargs):
        return _legacy_result(torch.zeros(1, 1, 2, device=sample.device))

    def progressive_fn(_model, sample, _labels, **_kwargs):
        output, compute, plan = _progressive_result(
            torch.zeros(1, 1, 2, device=sample.device)
        )
        plan["rows"][0]["mutual_stability_shadow"]["selection_enabled"] = True
        return output, compute, plan

    ticks = iter(float(index) for index in range(20))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result = benchmark.benchmark_requests(
        _BareModel(),
        torch.zeros(1, 1, 4),
        torch.zeros(1, dtype=torch.long),
        available_labels=[0, 1],
        legacy_fn=legacy_fn,
        progressive_fn=progressive_fn,
    )

    check = result["gates"]["checks"]["mutual_stability_is_shadow_only"]
    assert check["passed"] is False
    assert check["selection_policy_violation_count"] == 1


def _fixture_metrics(*, samples, offset, legacy_ms=20.0, progressive_ms=12.0):
    legacy_first = (samples + 1) // 2 if offset == 0 else samples // 2
    return {
        "request_count": samples,
        "measurement_order": {
            "offset": offset,
            "legacy_then_progressive": legacy_first,
            "progressive_then_legacy": samples - legacy_first,
        },
        "legacy_v1": {
            "correct_predictions": samples,
            "forward_evaluations": samples * 4,
            "observed_prediction_margin": {
                "observation_count": samples,
                "minimum": 0.25,
            },
            "observed_prediction_decision_margin": {
                "observation_count": samples,
                "minimum": 0.125,
            },
            "latency": {"total_ms": legacy_ms, "mean_ms": legacy_ms / samples},
        },
        "progressive": {
            "correct_predictions": samples,
            "forward_evaluations": samples * 2,
            "observed_prediction_margin": {
                "observation_count": samples,
                "minimum": 0.2,
            },
            "observed_prediction_decision_margin": {
                "observation_count": samples,
                "minimum": 0.1,
            },
            "latency": {
                "total_ms": progressive_ms,
                "mean_ms": progressive_ms / samples,
            },
        },
        "comparison": {
            "selected_cycle_pairs": {"3->3": samples},
            "exact_selected_cycle_disagreement_count": 0,
            "exact_prediction_disagreement_count": 0,
            "exact_output_tensor_disagreement_count": 0,
            "maximum_absolute_logit_difference": 0.0,
            "forward_evaluation_reduction_percent": 50.0,
        },
        "gates": {
            "passed": True,
            "checks": {
                "per_request_forward_evaluation_reduction": {
                    "non_reducing_request_count": 0
                },
                "mutual_stability_is_shadow_only": {
                    "selection_policy_violation_count": 0
                },
            },
        },
    }


def test_aggregate_uses_exact_counts_weighted_work_and_median_seed_latency():
    aggregate = benchmark.aggregate_results(
        [
            {"seed": 719, "metrics": _fixture_metrics(samples=2, offset=0)},
            {
                "seed": 727,
                "metrics": _fixture_metrics(
                    samples=4, offset=1, legacy_ms=80, progressive_ms=60
                ),
            },
        ]
    )
    summary = aggregate["summary"]

    assert aggregate["gates"]["passed"] is True
    assert summary["total_samples"] == 6
    assert summary["legacy_forward_evaluations"] == 24
    assert summary["progressive_forward_evaluations"] == 12
    assert summary["forward_evaluation_reduction_percent"] == 50
    assert summary["weighted_mean_latency_reduction_percent"] == 28
    assert summary["median_per_seed_latency_reduction_percent"] == 32.5
    assert summary["selected_cycle_pairs"] == {"3->3": 6}
    assert summary["observed_prediction_margin"] == {
        "metric": "top_1_minus_top_2_probability",
        "legacy_v1": {"observation_count": 6, "minimum": 0.25},
        "progressive": {"observation_count": 6, "minimum": 0.2},
    }
    assert summary["observed_prediction_decision_margin"] == {
        "metric": "minimum_adjacent_probability_gap_through_rank_depth",
        "legacy_v1": {"observation_count": 6, "minimum": 0.125},
        "progressive": {"observation_count": 6, "minimum": 0.1},
    }


def test_run_benchmark_loads_checkpoint_once_and_alternates_seed_offsets(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"real checkpoint identity")
    loads = []
    tasks = []
    offsets = []
    margins = []
    rank_depths = []

    def loader(path, device):
        loads.append((path, device))
        return _BareModel()

    def task_factory(samples, *, seed):
        tasks.append((samples, seed))
        return torch.zeros(samples, 1, 4), torch.zeros(samples, dtype=torch.long)

    def benchmark_fn(_model, _x, y, **kwargs):
        offsets.append(kwargs["order_offset"])
        margins.append(kwargs["prediction_stability_margin"])
        rank_depths.append(kwargs["prediction_stability_rank_depth"])
        return _fixture_metrics(samples=len(y), offset=kwargs["order_offset"])

    result = benchmark.run_benchmark(
        weights=weights,
        seeds=[719, 727, 733],
        samples_per_seed=3,
        device=torch.device("cpu"),
        device_info={"requested": "cpu", "resolved": "cpu"},
        model_loader=loader,
        task_factory=task_factory,
        benchmark_fn=benchmark_fn,
        provenance={"fixture": True},
    )

    assert len(loads) == 1
    assert tasks == [(3, 719), (3, 727), (3, 733)]
    assert offsets == [0, 1, 0]
    assert margins == [benchmark.DEFAULT_PREDICTION_STABILITY_MARGIN] * 3
    assert rank_depths == [benchmark.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH] * 3
    assert result["checkpoint"]["sha256"] == hashlib.sha256(
        b"real checkpoint identity"
    ).hexdigest()
    assert result["configuration"]["total_samples"] == 9
    assert (
        result["configuration"]["prediction_stability_margin"]
        == benchmark.DEFAULT_PREDICTION_STABILITY_MARGIN
    )
    assert (
        result["configuration"]["prediction_stability_rank_depth"]
        == benchmark.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
    )
    assert result["configuration"]["mutual_stability_role"] == (
        "shadow_diagnostic_only"
    )
    assert result["gates"]["passed"] is True


def test_inactive_verifier_margins_are_not_reported_as_observed():
    assert benchmark._prediction_margin_from_compute(
        {"prediction_verifier_active": False, "prediction_margin": 0.0}
    ) is None
    assert benchmark._prediction_margin_from_compute(
        {"prediction_margin": 0.25}
    ) is None
    assert benchmark._prediction_decision_margin_from_compute(
        {
            "prediction_verifier_active": False,
            "prediction_decision_margin": 0.125,
        }
    ) is None
    assert benchmark._prediction_decision_margin_from_compute(
        {"prediction_decision_margin": 0.125}
    ) is None
    assert benchmark._prediction_decision_margin_from_compute(
        {
            "prediction_verifier_active": True,
            "prediction_decision_margin": 0.125,
        }
    ) == 0.125


@pytest.mark.parametrize("seed", [51, 52])
def test_run_benchmark_rejects_training_or_original_test_seed(tmp_path, seed):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="overlap"):
        benchmark.run_benchmark(
            weights=weights,
            seeds=[seed],
            samples_per_seed=1,
            device=torch.device("cpu"),
            device_info={"resolved": "cpu"},
            model_loader=lambda *_args: _BareModel(),
            provenance={"fixture": True},
        )


def test_provenance_captures_controller_model_data_runtime_and_git(monkeypatch):
    def fake_git(*args):
        if args == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "agent/benchmark"
        return ""

    monkeypatch.setattr(benchmark, "_git_text", fake_git)
    provenance = benchmark.collect_provenance(
        device=torch.device("cpu"),
        device_info={"requested": "auto", "resolved": "cpu"},
    )

    assert provenance["git"] == {
        "commit": "abc123",
        "branch": "agent/benchmark",
        "worktree_dirty": False,
    }
    assert set(provenance["source_sha256"]) == {
        "source/benchmark_progressive_auto_compute.py",
        "source/chat_app.py",
        "source/benchmark_cognitive_leap_ultra_v51.py",
        "source/model_variants.py",
    }
    assert all(len(value) == 64 for value in provenance["source_sha256"].values())
    assert provenance["runtime"]["torch"]["version"] == str(torch.__version__)
    assert provenance["runtime"]["device"]["resolved"] == "cpu"


def test_main_enforces_gates_and_emits_strict_json(monkeypatch, tmp_path, capsys):
    payload = {"gates": {"passed": False}, "finite": 1.0}
    monkeypatch.setattr(benchmark, "configure_torch_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        benchmark,
        "resolve_device",
        lambda *_args, **_kwargs: (torch.device("cpu"), {"resolved": "cpu"}),
    )
    monkeypatch.setattr(benchmark, "run_benchmark", lambda **_kwargs: payload)
    output = tmp_path / "benchmark.json"

    exit_code = benchmark.main(
        ["--output", str(output), "--enforce-gates"]
    )

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_strict_json_rejects_non_finite_values():
    with pytest.raises(ValueError):
        benchmark._strict_json({"bad": float("nan")})


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_run_benchmark_rejects_invalid_prediction_stability_margin(
    tmp_path, margin
):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="prediction stability margin"):
        benchmark.run_benchmark(
            weights=weights,
            seeds=[719],
            samples_per_seed=1,
            device=torch.device("cpu"),
            device_info={"resolved": "cpu"},
            prediction_stability_margin=margin,
            model_loader=lambda *_args: _BareModel(),
            provenance={"fixture": True},
        )
