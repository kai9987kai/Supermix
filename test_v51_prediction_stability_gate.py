import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import benchmark_v51_prediction_stability as single_seed
import run_v51_prediction_stability_gate as gate


class _StubHead:
    def __init__(self):
        self.last_cycles_used = torch.tensor(2.0)
        self.last_prediction_topk_js_divergence = torch.tensor(0.001)
        self.last_prediction_topk_js_divergence_max = torch.tensor(0.002)
        self.last_exit_reason = "prediction_stable"


class _StubModel:
    def __init__(self):
        self.layers = [None] * 11
        self.layers[10] = _StubHead()
        self.calls = []

    def __call__(self, _sample, *, reasoning_cycles, adaptive_compute=False, **_kwargs):
        self.calls.append("adaptive" if adaptive_compute else "fixed")
        logits = torch.tensor([[0.0, 1.0]])
        return logits


def test_gate_cli_defaults_to_native_4096_request_matrix():
    args = gate.build_parser().parse_args([])

    assert args.seeds == [641, 643, 647, 653, 659, 661, 673, 677]
    assert args.samples_per_seed == 512
    assert len(args.seeds) * args.samples_per_seed == 4096


def _fixture_metrics(
    *,
    samples,
    fixed_correct,
    adaptive_correct,
    disagreements,
    fixed_cycles,
    adaptive_total_cycles,
    fixed_latency,
    adaptive_latency,
    latency_reduction,
    cycle_reduction,
    offset=0,
):
    return {
        "fixed": {
            "cycles": fixed_cycles,
            "correct_predictions": fixed_correct,
            "latency": {"mean_ms": fixed_latency},
        },
        "prediction_stability": {
            "correct_predictions": adaptive_correct,
            "total_cycles_used": adaptive_total_cycles,
            "latency": {"mean_ms": adaptive_latency},
        },
        "comparison": {
            "request_count": samples,
            "exact_disagreement_count": disagreements,
            "cycle_reduction_percent": cycle_reduction,
            "mean_latency_reduction_percent": latency_reduction,
            "measurement_order": {
                "offset": offset,
                "fixed_then_adaptive": (samples + (1 - offset)) // 2,
                "adaptive_then_fixed": (samples + offset) // 2,
            },
        },
    }


@pytest.mark.parametrize(
    ("offset", "expected_calls"),
    [
        (0, ["fixed", "adaptive", "adaptive", "fixed"]),
        (1, ["adaptive", "fixed", "fixed", "adaptive"]),
    ],
)
def test_single_seed_benchmark_counterbalances_each_request(
    monkeypatch, offset, expected_calls
):
    model = _StubModel()
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))
    x = torch.zeros(2, 1, 4)
    y = torch.ones(2, dtype=torch.long)

    result = single_seed.benchmark_serving_requests(
        model,
        x,
        y,
        fixed_cycles=3,
        max_cycles=8,
        stability_patience=2,
        stability_tol=0.005,
        distribution_top_k=5,
        order_offset=offset,
    )

    # Two unmeasured warmups precede the request calls.
    assert model.calls[2:] == expected_calls
    assert result["comparison"]["measurement_order"] == {
        "offset": offset,
        "fixed_then_adaptive": 1,
        "adaptive_then_fixed": 1,
    }
    assert result["fixed"]["latency"]["mean_ms"] == 1000.0
    assert result["prediction_stability"]["latency"]["mean_ms"] == 1000.0
    assert result["comparison"]["exact_disagreement_count"] == 0
    assert result["prediction_stability"]["total_cycles_used"] == 4.0


def test_aggregate_gate_uses_exact_counts_and_weighted_metrics():
    seed_results = [
        {
            "seed": 641,
            "metrics": _fixture_metrics(
                samples=2,
                fixed_correct=1,
                adaptive_correct=1,
                disagreements=0,
                fixed_cycles=3,
                adaptive_total_cycles=4,
                fixed_latency=10,
                adaptive_latency=9,
                latency_reduction=10,
                cycle_reduction=33.333,
            ),
        },
        {
            "seed": 643,
            "metrics": _fixture_metrics(
                samples=4,
                fixed_correct=3,
                adaptive_correct=4,
                disagreements=0,
                fixed_cycles=3,
                adaptive_total_cycles=6,
                fixed_latency=20,
                adaptive_latency=14,
                latency_reduction=30,
                cycle_reduction=50,
                offset=1,
            ),
        },
    ]

    result = gate.aggregate_gate_results(seed_results)

    assert result["gates"]["passed"] is True
    assert result["summary"]["exact_disagreement_count"] == 0
    assert result["summary"]["weighted_cycle_reduction_percent"] == pytest.approx(
        44.444444
    )
    assert result["summary"]["median_per_seed_latency_reduction_percent"] == 20
    assert result["summary"]["weighted_fixed_mean_latency_ms"] == pytest.approx(
        16.666667
    )
    assert result["summary"]["weighted_adaptive_mean_latency_ms"] == pytest.approx(
        12.333333
    )
    assert result["summary"]["per_seed_accuracy_deltas"] == [
        {"seed": 641, "accuracy_delta": 0.0},
        {"seed": 643, "accuracy_delta": 0.25},
    ]


def test_aggregate_gate_reports_every_failed_invariant():
    seed_results = [
        {
            "seed": 641,
            "metrics": _fixture_metrics(
                samples=4,
                fixed_correct=4,
                adaptive_correct=3,
                disagreements=1,
                fixed_cycles=3,
                adaptive_total_cycles=10,
                fixed_latency=10,
                adaptive_latency=11,
                latency_reduction=-10,
                cycle_reduction=16.667,
            ),
        }
    ]

    result = gate.aggregate_gate_results(seed_results)
    checks = result["gates"]["checks"]

    assert result["gates"]["passed"] is False
    assert checks["zero_prediction_disagreements"]["passed"] is False
    assert checks["no_negative_per_seed_accuracy_delta"]["negative_seed_ids"] == [641]
    assert checks["minimum_weighted_cycle_reduction"]["passed"] is False
    assert checks["positive_median_per_seed_latency_reduction"]["passed"] is False


def test_run_gate_loads_checkpoint_once_and_alternates_seed_offsets(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"one checkpoint")
    loads = []
    tasks = []
    offsets = []

    def model_loader(path, device):
        loads.append((path, device))
        return object()

    def task_factory(samples, *, seed):
        tasks.append((samples, seed))
        return torch.zeros(samples, 1), torch.zeros(samples, dtype=torch.long)

    def benchmark_fn(_model, _x, y, **kwargs):
        offsets.append(kwargs["order_offset"])
        samples = len(y)
        return _fixture_metrics(
            samples=samples,
            fixed_correct=samples,
            adaptive_correct=samples,
            disagreements=0,
            fixed_cycles=3,
            adaptive_total_cycles=2 * samples,
            fixed_latency=10,
            adaptive_latency=8,
            latency_reduction=20,
            cycle_reduction=33.333,
            offset=kwargs["order_offset"],
        )

    result = gate.run_gate(
        weights=weights,
        seeds=[641, 643, 647],
        samples_per_seed=3,
        device=torch.device("cpu"),
        device_info={"requested": "cpu", "resolved": "cpu"},
        model_loader=model_loader,
        task_factory=task_factory,
        benchmark_fn=benchmark_fn,
        provenance={"fixture": True},
    )

    assert len(loads) == 1
    assert tasks == [(3, 641), (3, 643), (3, 647)]
    assert offsets == [0, 1, 0]
    assert result["checkpoint"]["sha256"] == hashlib.sha256(
        b"one checkpoint"
    ).hexdigest()
    assert result["configuration"]["total_samples"] == 9
    assert result["gates"]["passed"] is True


def test_provenance_captures_git_sources_runtime_device_and_threads(monkeypatch):
    def fake_git(*args):
        if args == ("rev-parse", "HEAD"):
            return "abc123"
        if args == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "main"
        return ""

    monkeypatch.setattr(gate, "_git_text", fake_git)

    provenance = gate.collect_provenance(
        device=torch.device("cpu"),
        device_info={"requested": "auto", "resolved": "cpu"},
    )

    assert provenance["git"] == {
        "commit": "abc123",
        "branch": "main",
        "worktree_dirty": False,
    }
    assert set(provenance["source_sha256"]) == {
        "source/run_v51_prediction_stability_gate.py",
        "source/benchmark_v51_prediction_stability.py",
        "source/benchmark_cognitive_leap_ultra_v51.py",
        "source/model_variants.py",
    }
    assert all(len(value) == 64 for value in provenance["source_sha256"].values())
    assert provenance["runtime"]["torch"]["version"] == str(torch.__version__)
    assert provenance["runtime"]["device"]["resolved"] == "cpu"
    assert provenance["runtime"]["threads"]["torch_num_threads"] >= 1


def test_enforce_gates_returns_two_and_stdout_remains_strict_json(
    monkeypatch, tmp_path, capsys
):
    payload = {"gates": {"passed": False}, "value": 1}
    monkeypatch.setattr(gate, "configure_torch_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        gate,
        "resolve_device",
        lambda *_args, **_kwargs: (torch.device("cpu"), {"resolved": "cpu"}),
    )
    monkeypatch.setattr(gate, "run_gate", lambda **_kwargs: payload)
    output = tmp_path / "gate.json"

    exit_code = gate.main(["--output", str(output), "--enforce-gates"])

    assert exit_code == 2
    assert json.loads(capsys.readouterr().out) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_strict_json_rejects_non_finite_values():
    with pytest.raises(ValueError):
        gate._strict_json({"bad": float("nan")})
