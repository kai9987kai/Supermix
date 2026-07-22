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
        self.last_prediction_margin = torch.tensor(0.25)
        self.last_prediction_decision_margin = torch.tensor(0.125)
        self.last_prediction_rank_depth = torch.tensor(3.0)
        self.last_prediction_class_count = torch.tensor(5.0)
        self.last_prediction_class_selection_valid = torch.tensor(True)
        self.last_exit_reason = "prediction_stable"


class _StubModel:
    def __init__(
        self,
        *,
        fixed_logits=None,
        adaptive_logits=None,
        class_selection_valid=True,
        class_count_override=None,
    ):
        self.layers = [None] * 11
        self.layers[10] = _StubHead()
        self.calls = []
        self.adaptive_margins = []
        self.adaptive_rank_depths = []
        self.adaptive_scopes = []
        self.fixed_logits = fixed_logits or [0.0, 4.0, 3.0, 2.0, 1.0]
        self.adaptive_logits = adaptive_logits or list(self.fixed_logits)
        self.class_selection_valid = class_selection_valid
        self.class_count_override = class_count_override

    def __call__(self, _sample, *, reasoning_cycles, adaptive_compute=False, **_kwargs):
        self.calls.append("adaptive" if adaptive_compute else "fixed")
        if adaptive_compute:
            self.adaptive_margins.append(
                _kwargs["prediction_stability_margin"]
            )
            requested_rank_depth = int(_kwargs["prediction_stability_rank_depth"])
            scope = _kwargs.get("prediction_class_indices")
            if scope is None:
                scope_count = len(self.adaptive_logits)
            elif scope and all(isinstance(value, bool) for value in scope):
                scope_count = sum(scope)
            else:
                scope_count = len(scope)
            effective_rank_depth = min(requested_rank_depth, max(1, scope_count - 1))
            self.layers[10].last_prediction_rank_depth = torch.tensor(
                float(effective_rank_depth)
            )
            self.layers[10].last_prediction_class_count = torch.tensor(
                float(
                    scope_count
                    if self.class_count_override is None
                    else self.class_count_override
                )
            )
            self.layers[10].last_prediction_class_selection_valid = torch.tensor(
                self.class_selection_valid
            )
            self.adaptive_rank_depths.append(requested_rank_depth)
            self.adaptive_scopes.append(scope)
        logits = torch.tensor(
            [self.adaptive_logits if adaptive_compute else self.fixed_logits]
        )
        return logits


def test_gate_cli_defaults_to_native_4096_request_matrix():
    args = gate.build_parser().parse_args([])

    assert args.seeds == [641, 643, 647, 653, 659, 661, 673, 677]
    assert args.samples_per_seed == 512
    assert len(args.seeds) * args.samples_per_seed == 4096
    assert (
        args.prediction_stability_margin
        == single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN
    )
    assert args.prediction_stability_rank_depth == 3
    assert args.decision_top_k == 3
    assert args.prediction_class_indices is None
    assert Path(args.metadata) == gate.DEFAULT_META
    assert single_seed.DEFAULT_ADAPTIVE_EXIT_TOL == single_seed.chat_app.DEFAULT_ADAPTIVE_EXIT_TOL
    assert (
        single_seed.DEFAULT_ADAPTIVE_EXIT_ENTROPY
        == single_seed.chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY
    )
    assert (
        single_seed.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
        == single_seed.chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH
    )


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
    configured_margin=single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN,
    exit_tolerance=0.0,
    exit_entropy_threshold=0.0,
    stability_patience=2,
    stability_tolerance=0.005,
    observed_margin=0.25,
    observed_decision_margin=0.125,
    top_k_order_disagreements=None,
    top_k_set_disagreements=None,
    decision_top_k=3,
    rank_depth=3,
    logit_delta_mean=0.0,
    logit_delta_max=0.0,
    js_mean=0.0,
    js_max=0.0,
    tv_mean=0.0,
    tv_max=0.0,
    scope_mode="all_output_classes",
    scope_indices=None,
    offset=0,
    max_cycles=8,
    distribution_top_k=5,
    exit_reasons=None,
    fixed_total_latency=None,
    adaptive_total_latency=None,
):
    if top_k_order_disagreements is None:
        top_k_order_disagreements = disagreements
    if top_k_set_disagreements is None:
        top_k_set_disagreements = disagreements
    if scope_indices is None:
        scope_indices = [0, 1, 2, 3, 4]
    if exit_reasons is None:
        exit_reasons = {"prediction_stable": samples}
    if fixed_total_latency is None:
        fixed_total_latency = fixed_latency * samples
    if adaptive_total_latency is None:
        adaptive_total_latency = adaptive_latency * samples
    base_cycles, extra_cycles = divmod(int(adaptive_total_cycles), int(samples))
    cycle_counts = {}
    if samples - extra_cycles:
        cycle_counts[str(base_cycles)] = samples - extra_cycles
    if extra_cycles:
        cycle_counts[str(base_cycles + 1)] = extra_cycles
    effective_rank_depth = min(rank_depth, max(1, len(scope_indices) - 1))
    def scalar(mean, maximum):
        return {
            "min": 0.0,
            "mean": mean,
            "p50": mean,
            "p95": maximum,
            "max": maximum,
        }
    return {
        "fixed": {
            "adaptive_compute": False,
            "cycles": fixed_cycles,
            "correct_predictions": fixed_correct,
            "latency": {
                "mean_ms": fixed_latency,
                "total_ms": fixed_total_latency,
            },
        },
        "prediction_stability": {
            "adaptive_compute": True,
            "max_cycles": max_cycles,
            "patience": stability_patience,
            "confidence_tolerance": stability_tolerance,
            "exit_tolerance": exit_tolerance,
            "exit_entropy_threshold": exit_entropy_threshold,
            "correct_predictions": adaptive_correct,
            "total_cycles_used": adaptive_total_cycles,
            "mean_cycles_used": round(adaptive_total_cycles / samples, 4),
            "cycle_counts": cycle_counts,
            "exit_reasons": exit_reasons,
            "prediction_margin": {
                "configured_minimum": configured_margin,
                "observed": {
                    "min": observed_margin,
                    "mean": observed_margin,
                    "p50": observed_margin,
                    "p95": observed_margin,
                    "max": observed_margin,
                },
            },
            "decision_margin": {
                "configured_minimum": configured_margin,
                "configured_rank_depth": rank_depth,
                "observed_rank_depths": [effective_rank_depth],
                "observed": {
                    "min": observed_decision_margin,
                    "mean": observed_decision_margin,
                    "p50": observed_decision_margin,
                    "p95": observed_decision_margin,
                    "max": observed_decision_margin,
                },
                "prediction_stable_observed": {
                    "observation_count": exit_reasons.get(
                        "prediction_stable", 0
                    ),
                    "minimum": (
                        observed_decision_margin
                        if exit_reasons.get("prediction_stable", 0)
                        else None
                    ),
                    "summary": (
                        {
                            "min": observed_decision_margin,
                            "mean": observed_decision_margin,
                            "p50": observed_decision_margin,
                            "p95": observed_decision_margin,
                            "max": observed_decision_margin,
                        }
                        if exit_reasons.get("prediction_stable", 0)
                        else None
                    ),
                },
            },
            "distribution_drift": {
                "top_k": distribution_top_k,
                "role": "shadow_diagnostic_only",
            },
            "latency": {
                "mean_ms": adaptive_latency,
                "total_ms": adaptive_total_latency,
            },
        },
        "comparison": {
            "request_count": samples,
            "top1_disagreement_count": disagreements,
            "exact_disagreement_count": disagreements,
            "decision_fidelity": {
                "verified_scope": {
                    "verified": True,
                    "mode": scope_mode,
                    "output_class_count": 5,
                    "class_count": len(scope_indices),
                    "class_indices": scope_indices,
                    "requested_normalized_class_indices": (
                        None
                        if scope_mode == "all_output_classes"
                        else scope_indices
                    ),
                    "adaptive_requested_class_indices_verified": True,
                    "truth_labels_in_scope": samples,
                    "truth_labels_outside_scope": 0,
                    "adaptive_verifier_rank_depth_verified": True,
                    "adaptive_verifier_class_scope_verified": True,
                    "observed_prediction_class_counts": [len(scope_indices)],
                    "effective_prediction_rank_depth": effective_rank_depth,
                },
                "top_k": decision_top_k,
                "top_k_order_disagreement_count": top_k_order_disagreements,
                "top_k_set_disagreement_count": top_k_set_disagreements,
                "absolute_logit_delta": {
                    "mean": logit_delta_mean,
                    "max": logit_delta_max,
                },
                "distribution_distance": {
                    "jensen_shannon_divergence_nats": scalar(js_mean, js_max),
                    "total_variation_distance": scalar(tv_mean, tv_max),
                },
                "tensor_equality_required": False,
            },
            "cycle_reduction_percent": cycle_reduction,
            "mean_latency_reduction_percent": latency_reduction,
            "measurement_order": {
                "offset": offset,
                "fixed_then_adaptive": (samples + (1 - offset)) // 2,
                "adaptive_then_fixed": (samples + offset) // 2,
            },
        },
    }


def _aggregate(seed_results, **overrides):
    configuration = {
        "fixed_cycles": 3,
        "max_cycles": 8,
        "prediction_class_indices": None,
        "decision_top_k": 3,
        "prediction_stability_rank_depth": 3,
        "prediction_stability_margin": (
            single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN
        ),
        "prediction_stability_patience": 2,
        "prediction_stability_tol": 0.005,
        "exit_tol": 0.0,
        "exit_entropy_threshold": 0.0,
        "distribution_top_k": 5,
    }
    configuration.update(overrides)
    return gate.aggregate_gate_results(seed_results, **configuration)


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
    assert result["comparison"]["top1_disagreement_count"] == 0
    assert result["comparison"]["exact_disagreement_count"] == 0
    assert result["comparison"]["decision_fidelity"] == {
        "verified_scope": {
            "verified": True,
            "mode": "all_output_classes",
            "output_class_count": 5,
            "class_count": 5,
            "class_indices": [0, 1, 2, 3, 4],
            "requested_normalized_class_indices": None,
            "adaptive_requested_class_indices_verified": True,
            "truth_labels_in_scope": 2,
            "truth_labels_outside_scope": 0,
            "adaptive_verifier_rank_depth_verified": True,
            "adaptive_verifier_class_scope_verified": True,
            "observed_prediction_class_counts": [5],
            "effective_prediction_rank_depth": 3,
        },
        "top_k": 3,
        "top_k_order_disagreement_count": 0,
        "top_k_set_disagreement_count": 0,
        "absolute_logit_delta": {
            "mean": 0.0,
            "max": 0.0,
            "role": "diagnostic_only_not_gated",
        },
        "distribution_distance": {
            "jensen_shannon_divergence_nats": {
                "min": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "max": 0.0,
            },
            "total_variation_distance": {
                "min": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "max": 0.0,
            },
            "role": "diagnostic_only_not_gated",
        },
        "tensor_equality_required": False,
    }
    assert result["prediction_stability"]["total_cycles_used"] == 4.0
    assert model.adaptive_margins == [
        single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN,
    ] * 3
    assert result["prediction_stability"]["prediction_margin"] == {
        "metric": "top_1_minus_top_2_probability",
        "configured_minimum": single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN,
        "observed": {
            "min": 0.25,
            "mean": 0.25,
            "p50": 0.25,
            "p95": 0.25,
            "max": 0.25,
        },
    }
    assert result["prediction_stability"]["decision_margin"] == {
        "metric": "minimum_adjacent_probability_gap_through_rank_depth",
        "configured_minimum": single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN,
        "configured_rank_depth": 3,
        "observed_rank_depths": [3],
        "observed": {
            "min": 0.125,
            "mean": 0.125,
            "p50": 0.125,
            "p95": 0.125,
            "max": 0.125,
        },
        "prediction_stable_observed": {
            "observation_count": 2,
            "minimum": 0.125,
            "summary": {
                "min": 0.125,
                "mean": 0.125,
                "p50": 0.125,
                "p95": 0.125,
                "max": 0.125,
            },
        },
    }
    assert model.adaptive_rank_depths == [3, 3, 3]


def test_single_seed_decision_fidelity_detects_top3_order_set_and_distribution_drift(
    monkeypatch,
):
    model = _StubModel(
        fixed_logits=[5.0, 4.0, 3.0, 2.0],
        adaptive_logits=[4.0, 5.0, 2.0, 3.0],
    )
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    result = single_seed.benchmark_serving_requests(
        model,
        torch.zeros(1, 1, 4),
        torch.zeros(1, dtype=torch.long),
        fixed_cycles=3,
        max_cycles=8,
        stability_patience=2,
        stability_tol=0.005,
        distribution_top_k=5,
    )

    comparison = result["comparison"]
    fidelity = comparison["decision_fidelity"]
    assert comparison["top1_disagreement_count"] == 1
    assert comparison["exact_disagreement_count"] == 1
    assert fidelity["top_k_order_disagreement_count"] == 1
    assert fidelity["top_k_set_disagreement_count"] == 1
    assert fidelity["absolute_logit_delta"]["mean"] == 1.0
    assert fidelity["absolute_logit_delta"]["max"] == 1.0
    assert (
        fidelity["distribution_distance"]["jensen_shannon_divergence_nats"][
            "mean"
        ]
        > 0.0
    )
    assert (
        fidelity["distribution_distance"]["total_variation_distance"]["mean"]
        > 0.0
    )


def test_single_seed_rejects_unknown_exit_reason_evidence(monkeypatch):
    model = _StubModel()
    model.layers[10].last_exit_reason = "unknown_exit"
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(ValueError, match="unknown adaptive exit reason"):
        single_seed.benchmark_serving_requests(
            model,
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.long),
            fixed_cycles=3,
            max_cycles=8,
            stability_patience=2,
            stability_tol=0.005,
            distribution_top_k=5,
        )


def test_single_seed_scoped_two_class_verifier_clamps_observed_rank_depth(
    monkeypatch,
):
    model = _StubModel()
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    result = single_seed.benchmark_serving_requests(
        model,
        torch.zeros(1, 1, 4),
        torch.tensor([1]),
        fixed_cycles=3,
        max_cycles=8,
        stability_patience=2,
        stability_tol=0.005,
        distribution_top_k=5,
        prediction_stability_rank_depth=3,
        decision_top_k=2,
        prediction_class_indices=[1, 3],
    )

    assert model.adaptive_scopes == [[1, 3], [1, 3]]
    assert result["prediction_stability"]["decision_margin"][
        "observed_rank_depths"
    ] == [1]
    assert result["comparison"]["decision_fidelity"]["verified_scope"] == {
        "verified": True,
        "mode": "prediction_class_indices",
        "output_class_count": 5,
        "class_count": 2,
        "class_indices": [1, 3],
        "requested_normalized_class_indices": [1, 3],
        "adaptive_requested_class_indices_verified": True,
        "truth_labels_in_scope": 1,
        "truth_labels_outside_scope": 0,
        "adaptive_verifier_rank_depth_verified": True,
        "adaptive_verifier_class_scope_verified": True,
        "observed_prediction_class_counts": [2],
        "effective_prediction_rank_depth": 1,
    }


def test_single_seed_records_bool_mask_as_exact_normalized_requested_indices(
    monkeypatch,
):
    model = _StubModel()
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    result = single_seed.benchmark_serving_requests(
        model,
        torch.zeros(1, 1, 4),
        torch.tensor([1]),
        fixed_cycles=3,
        max_cycles=8,
        stability_patience=2,
        stability_tol=0.005,
        distribution_top_k=5,
        prediction_stability_rank_depth=3,
        decision_top_k=2,
        prediction_class_indices=[False, True, False, True, False],
    )

    scope = result["comparison"]["decision_fidelity"]["verified_scope"]
    assert scope["class_indices"] == [1, 3]
    assert scope["requested_normalized_class_indices"] == [1, 3]
    assert scope["adaptive_requested_class_indices_verified"] is True


def test_single_seed_fails_closed_when_model_rejects_full_class_scope(monkeypatch):
    model = _StubModel(class_selection_valid=False)
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(ValueError, match="prediction class selection is invalid"):
        single_seed.benchmark_serving_requests(
            model,
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.long),
            fixed_cycles=3,
            max_cycles=8,
            stability_patience=2,
            stability_tol=0.005,
            distribution_top_k=5,
        )


@pytest.mark.parametrize(
    ("scope", "reported_count"),
    [
        (None, 4),
        ([0, 1, 2, 3], 5),
    ],
)
def test_single_seed_fails_closed_on_full_or_scoped_class_count_mismatch(
    monkeypatch, scope, reported_count
):
    model = _StubModel(class_count_override=reported_count)
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(ValueError, match="class count inconsistent"):
        single_seed.benchmark_serving_requests(
            model,
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.long),
            fixed_cycles=3,
            max_cycles=8,
            stability_patience=2,
            stability_tol=0.005,
            distribution_top_k=5,
            prediction_class_indices=scope,
        )


def test_single_seed_rejects_nonfinite_logits(monkeypatch):
    model = _StubModel(adaptive_logits=[float("nan"), 1.0, 0.0, -1.0])
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(ValueError, match="adaptive logits must be finite"):
        single_seed.benchmark_serving_requests(
            model,
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.long),
            fixed_cycles=3,
            max_cycles=8,
            stability_patience=2,
            stability_tol=0.005,
            distribution_top_k=5,
        )


@pytest.mark.parametrize(
    ("scope", "match"),
    [
        ([0, 0, 1], "duplicates"),
        ([0, 1], "requires at least 3"),
        ([0, 1, 99], "out-of-range"),
        ([0.0, 1.0, 2.0], "only integers"),
    ],
)
def test_single_seed_rejects_unverifiable_prediction_scopes(
    monkeypatch, scope, match
):
    model = _StubModel()
    ticks = iter(float(index) for index in range(100))
    monkeypatch.setattr(single_seed.time, "perf_counter", lambda: next(ticks))

    with pytest.raises(ValueError, match=match):
        single_seed.benchmark_serving_requests(
            model,
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.long),
            fixed_cycles=3,
            max_cycles=8,
            stability_patience=2,
            stability_tol=0.005,
            distribution_top_k=5,
            prediction_class_indices=scope,
        )


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

    result = _aggregate(seed_results)

    assert result["gates"]["passed"] is True
    assert result["summary"]["top1_disagreement_count"] == 0
    assert result["summary"]["exact_disagreement_count"] == 0
    assert result["summary"]["decision_fidelity"][
        "top_k_order_disagreement_count"
    ] == 0
    assert result["summary"]["decision_fidelity"][
        "top_k_set_disagreement_count"
    ] == 0
    assert result["summary"]["decision_fidelity"][
        "tensor_equality_required"
    ] is False
    assert result["gates"]["tensor_equality_required"] is False
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
    assert result["summary"]["minimum_observed_prediction_margin"] == 0.25
    assert result["summary"]["minimum_observed_decision_margin"] == 0.125


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

    result = _aggregate(seed_results)
    checks = result["gates"]["checks"]

    assert result["gates"]["passed"] is False
    assert checks["zero_top1_disagreements"]["passed"] is False
    assert checks["zero_top_k_order_disagreements"]["passed"] is False
    assert checks["zero_top_k_set_disagreements"]["passed"] is False
    assert checks["zero_prediction_disagreements"]["passed"] is False
    assert checks["no_negative_per_seed_accuracy_delta"]["negative_seed_ids"] == [641]
    assert checks["minimum_weighted_cycle_reduction"]["passed"] is False
    assert checks["positive_median_per_seed_latency_reduction"]["passed"] is False


def test_run_gate_loads_checkpoint_once_and_alternates_seed_offsets(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"one checkpoint")
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "buckets": {
                    str(label): [{"text": f"response {label}"}]
                    for label in range(5)
                }
            }
        ),
        encoding="utf-8",
    )
    loads = []
    tasks = []
    offsets = []
    margins = []
    rank_depths = []
    decision_top_ks = []
    scopes = []
    exit_controls = []

    def model_loader(path, device):
        loads.append((path, device))
        return object()

    def task_factory(samples, *, seed):
        tasks.append((samples, seed))
        return torch.zeros(samples, 1), torch.zeros(samples, dtype=torch.long)

    def benchmark_fn(_model, _x, y, **kwargs):
        offsets.append(kwargs["order_offset"])
        margins.append(kwargs["prediction_stability_margin"])
        rank_depths.append(kwargs["prediction_stability_rank_depth"])
        decision_top_ks.append(kwargs["decision_top_k"])
        scopes.append(kwargs["prediction_class_indices"])
        exit_controls.append(
            (kwargs["exit_tol"], kwargs["exit_entropy_threshold"])
        )
        samples = len(y)
        return _fixture_metrics(
            samples=samples,
            fixed_correct=samples,
            adaptive_correct=samples,
            disagreements=0,
            fixed_cycles=kwargs["fixed_cycles"],
            adaptive_total_cycles=2 * samples,
            fixed_latency=10,
            adaptive_latency=8,
            latency_reduction=20,
            cycle_reduction=33.333,
            configured_margin=kwargs["prediction_stability_margin"],
            exit_tolerance=kwargs["exit_tol"],
            exit_entropy_threshold=kwargs["exit_entropy_threshold"],
            stability_patience=kwargs["stability_patience"],
            stability_tolerance=kwargs["stability_tol"],
            decision_top_k=kwargs["decision_top_k"],
            rank_depth=kwargs["prediction_stability_rank_depth"],
            scope_mode="prediction_class_indices",
            scope_indices=kwargs["prediction_class_indices"],
            offset=kwargs["order_offset"],
        )

    result = gate.run_gate(
        weights=weights,
        metadata=metadata,
        seeds=[641, 643, 647],
        samples_per_seed=3,
        device=torch.device("cpu"),
        device_info={"requested": "cpu", "resolved": "cpu"},
        model_loader=model_loader,
        task_factory=task_factory,
        benchmark_fn=benchmark_fn,
        prediction_class_indices=[0, 1, 2, 3, 4],
        provenance={"fixture": True},
    )

    assert len(loads) == 1
    assert tasks == [(3, 641), (3, 643), (3, 647)]
    assert offsets == [0, 0, 1, 1, 0, 0]
    assert margins == [gate.DEFAULT_PREDICTION_STABILITY_MARGIN] * 6
    assert rank_depths == [3] * 6
    assert decision_top_ks == [3] * 6
    assert scopes == [[0, 1, 2, 3, 4]] * 6
    assert exit_controls == [
        (0.0, 0.0),
        (
            single_seed.chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
            single_seed.chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        ),
    ] * 3
    assert result["checkpoint"]["sha256"] == hashlib.sha256(
        b"one checkpoint"
    ).hexdigest()
    assert result["configuration"]["total_samples"] == 9
    assert (
        result["configuration"]["prediction_stability_margin"]
        == gate.DEFAULT_PREDICTION_STABILITY_MARGIN
    )
    assert result["configuration"]["prediction_stability_rank_depth"] == 3
    assert result["configuration"]["decision_top_k"] == 3
    assert result["configuration"]["prediction_class_indices"] == [0, 1, 2, 3, 4]
    assert result["configuration"]["prediction_scope_orchestration"] == "single_scope"
    assert result["configuration"]["modes"]["isolated_verifier"][
        "exit_tolerance"
    ] == 0.0
    assert result["configuration"]["modes"]["release_runtime"][
        "exit_tolerance"
    ] == single_seed.chat_app.DEFAULT_ADAPTIVE_EXIT_TOL
    assert result["configuration"]["modes"]["release_runtime"][
        "prediction_class_indices"
    ] == [0, 1, 2, 3, 4]
    assert result["metadata"]["scope_source"] == "metadata_nonempty_buckets"
    assert result["gates"]["mode_passed"] == {
        "isolated_verifier": True,
        "release_runtime": True,
    }
    assert result["summary"]["mode_summaries"]["release_runtime"] == result[
        "mode_results"
    ]["release_runtime"]["summary"]
    assert result["gates"]["passed"] is True


def test_release_runtime_top3_disagreement_fails_combined_gate(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "buckets": {
                    str(label): [{"text": f"response {label}"}]
                    for label in range(5)
                }
            }
        ),
        encoding="utf-8",
    )

    def benchmark_fn(_model, _x, y, **kwargs):
        release_mode = kwargs["exit_tol"] == single_seed.DEFAULT_ADAPTIVE_EXIT_TOL
        scope = kwargs["prediction_class_indices"]
        resolved_scope = list(range(5)) if scope is None else scope
        return _fixture_metrics(
            samples=len(y),
            fixed_correct=len(y),
            adaptive_correct=len(y),
            disagreements=0,
            top_k_order_disagreements=1 if release_mode else 0,
            top_k_set_disagreements=1 if release_mode else 0,
            fixed_cycles=kwargs["fixed_cycles"],
            adaptive_total_cycles=2 * len(y),
            fixed_latency=10,
            adaptive_latency=8,
            latency_reduction=20,
            cycle_reduction=33.333,
            configured_margin=kwargs["prediction_stability_margin"],
            exit_tolerance=kwargs["exit_tol"],
            exit_entropy_threshold=kwargs["exit_entropy_threshold"],
            stability_patience=kwargs["stability_patience"],
            stability_tolerance=kwargs["stability_tol"],
            decision_top_k=kwargs["decision_top_k"],
            rank_depth=kwargs["prediction_stability_rank_depth"],
            scope_mode=(
                "all_output_classes"
                if scope is None
                else "prediction_class_indices"
            ),
            scope_indices=resolved_scope,
            offset=kwargs["order_offset"],
        )

    result = gate.run_gate(
        weights=weights,
        metadata=metadata,
        seeds=[641],
        samples_per_seed=2,
        device=torch.device("cpu"),
        device_info={"resolved": "cpu"},
        model_loader=lambda *_args: object(),
        task_factory=lambda samples, *, seed: (
            torch.zeros(samples, 1),
            torch.zeros(samples, dtype=torch.long),
        ),
        benchmark_fn=benchmark_fn,
        provenance={"fixture": True},
    )

    assert result["gates"]["passed"] is False
    assert result["gates"]["mode_passed"] == {
        "isolated_verifier": True,
        "release_runtime": False,
    }
    assert result["gates"]["checks"]["zero_top_k_order_disagreements"][
        "passed"
    ] is False
    assert result["gates"]["checks"][
        "isolated_verifier__zero_top_k_order_disagreements"
    ]["passed"] is True


def test_metadata_scope_matches_runtime_nonempty_bucket_policy(tmp_path):
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "buckets": {
                    "0": [{"text": "zero"}],
                    "2": [],
                    "3": [{"text": "three"}],
                    "99": [{"text": "out of range"}],
                }
            }
        ),
        encoding="utf-8",
    )

    scope, record = gate.load_release_prediction_scope(metadata)

    assert scope == [0, 3]
    assert record["scope_source"] == "metadata_nonempty_buckets"
    assert record["allowed_class_indices"] == [0, 3]

    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match=r"decision_top_k \+ 1 allowed classes"):
        gate.run_gate(
            weights=weights,
            metadata=metadata,
            seeds=[641],
            samples_per_seed=1,
            device=torch.device("cpu"),
            device_info={"resolved": "cpu"},
            model_loader=lambda *_args: pytest.fail("model must not load"),
            provenance={"fixture": True},
        )


def test_metadata_binds_model_size_and_num_classes_when_present(tmp_path):
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                "model_size": "cognitive_leap_ultra_expert",
                "num_classes": 10,
                "buckets": {
                    str(label): [{"text": f"response {label}"}]
                    for label in range(5)
                },
            }
        ),
        encoding="utf-8",
    )

    scope, record = gate.load_release_prediction_scope(metadata)

    assert scope == [0, 1, 2, 3, 4]
    assert record["model_identity"] == {
        "field_present": True,
        "observed_model_size": "cognitive_leap_ultra_expert",
        "expected_model_size": "cognitive_leap_ultra_expert",
        "verified_when_present": True,
    }
    assert record["class_identity"] == {
        "field_present": True,
        "observed_num_classes": 10,
        "expected_num_classes": 10,
        "verified_when_present": True,
    }


@pytest.mark.parametrize(
    ("field", "bad_value", "match"),
    [
        ("model_size", "base", "model_size"),
        ("num_classes", 9, "num_classes"),
        ("num_classes", "10", "integer"),
    ],
)
def test_metadata_rejects_mismatched_model_identity(
    tmp_path, field, bad_value, match
):
    metadata = tmp_path / "meta.json"
    metadata.write_text(
        json.dumps(
            {
                field: bad_value,
                "buckets": {
                    str(label): [{"text": f"response {label}"}]
                    for label in range(5)
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=match):
        gate.load_release_prediction_scope(metadata)


@pytest.mark.parametrize(
    ("field_path", "bad_value"),
    [
        (("comparison", "decision_fidelity", "absolute_logit_delta", "mean"), float("nan")),
        (
            (
                "comparison",
                "decision_fidelity",
                "distribution_distance",
                "total_variation_distance",
                "max",
            ),
            float("inf"),
        ),
        (("prediction_stability", "decision_margin", "observed", "min"), -0.1),
    ],
)
def test_aggregate_gate_rejects_nonfinite_or_negative_decision_metrics(
    field_path, bad_value
):
    metrics = _fixture_metrics(
        samples=2,
        fixed_correct=2,
        adaptive_correct=2,
        disagreements=0,
        fixed_cycles=3,
        adaptive_total_cycles=4,
        fixed_latency=10,
        adaptive_latency=8,
        latency_reduction=20,
        cycle_reduction=33.333,
    )
    target = metrics
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = bad_value

    with pytest.raises(ValueError, match="finite|nonnegative"):
        _aggregate([{"seed": 641, "metrics": metrics}])


def _valid_aggregate_metrics():
    return _fixture_metrics(
        samples=2,
        fixed_correct=2,
        adaptive_correct=2,
        disagreements=0,
        fixed_cycles=3,
        adaptive_total_cycles=4,
        fixed_latency=10,
        adaptive_latency=8,
        latency_reduction=20,
        cycle_reduction=33.333,
    )


def _mutate_path(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("field_path", "bad_value", "match"),
    [
        (("fixed", "cycles"), 4, "fixed cycles"),
        (("prediction_stability", "max_cycles"), 7, "adaptive controls"),
        (("prediction_stability", "total_cycles_used"), 0, "total cycles"),
        (("prediction_stability", "total_cycles_used"), 17, "total cycles"),
        (("fixed", "latency", "mean_ms"), 0, "fixed latency"),
        (("prediction_stability", "latency", "mean_ms"), 0, "adaptive latency"),
        (("fixed", "latency", "total_ms"), 0, "fixed latency"),
        (("prediction_stability", "latency", "total_ms"), 0, "adaptive latency"),
        (("fixed", "latency", "total_ms"), 20.01, "inconsistent"),
        (
            ("prediction_stability", "latency", "total_ms"),
            16.01,
            "inconsistent",
        ),
        (("fixed", "latency", "mean_ms"), float("nan"), "finite"),
        (
            ("prediction_stability", "latency", "mean_ms"),
            float("inf"),
            "finite",
        ),
        (("prediction_stability", "exit_tolerance"), 0.5, "adaptive controls"),
        (
            ("prediction_stability", "exit_entropy_threshold"),
            0.5,
            "adaptive controls",
        ),
        (("prediction_stability", "patience"), 1, "adaptive controls"),
        (
            ("prediction_stability", "confidence_tolerance"),
            0.5,
            "adaptive controls",
        ),
        (
            ("comparison", "decision_fidelity", "top_k"),
            2,
            "decision top-k",
        ),
        (
            ("prediction_stability", "distribution_drift", "top_k"),
            4,
            "distribution controls",
        ),
    ],
)
def test_aggregate_gate_binds_budgets_latencies_and_release_controls(
    field_path, bad_value, match
):
    metrics = _valid_aggregate_metrics()
    _mutate_path(metrics, field_path, bad_value)

    with pytest.raises(ValueError, match=match):
        _aggregate([{"seed": 641, "metrics": metrics}])


@pytest.mark.parametrize("mode", ["fixed", "prediction_stability"])
def test_aggregate_gate_requires_total_latency_evidence(mode):
    metrics = _valid_aggregate_metrics()
    metrics[mode]["latency"].pop("total_ms")

    with pytest.raises(ValueError, match="missing total_ms"):
        _aggregate([{"seed": 641, "metrics": metrics}])


def test_aggregate_gate_allows_normal_latency_rounding_envelope():
    metrics = _fixture_metrics(
        samples=3,
        fixed_correct=3,
        adaptive_correct=3,
        disagreements=0,
        fixed_cycles=3,
        adaptive_total_cycles=6,
        fixed_latency=10.123,
        adaptive_latency=8.456,
        fixed_total_latency=30.3704,
        adaptive_total_latency=25.3694,
        latency_reduction=16.47,
        cycle_reduction=33.333,
    )

    result = _aggregate([{"seed": 641, "metrics": metrics}])

    assert result["gates"]["passed"] is True
    latency_evidence = result["per_seed_gate_metrics"][0]["latency_consistency"]
    assert latency_evidence["fixed"][
        "expected_total_from_rounded_mean_ms"
    ] == pytest.approx(30.369)
    assert latency_evidence["adaptive"][
        "expected_total_from_rounded_mean_ms"
    ] == pytest.approx(25.368)
    assert latency_evidence["fixed"][
        "absolute_rounding_tolerance_ms"
    ] == pytest.approx(0.001501)


@pytest.mark.parametrize(
    "field_path",
    [
        ("fixed", "adaptive_compute"),
        ("prediction_stability", "adaptive_compute"),
        ("comparison", "decision_fidelity", "verified_scope", "verified"),
        (
            "comparison",
            "decision_fidelity",
            "verified_scope",
            "adaptive_requested_class_indices_verified",
        ),
        (
            "comparison",
            "decision_fidelity",
            "verified_scope",
            "adaptive_verifier_rank_depth_verified",
        ),
        (
            "comparison",
            "decision_fidelity",
            "verified_scope",
            "adaptive_verifier_class_scope_verified",
        ),
    ],
)
def test_aggregate_gate_rejects_string_boolean_control_or_scope_evidence(field_path):
    metrics = _valid_aggregate_metrics()
    _mutate_path(metrics, field_path, "false")

    with pytest.raises(ValueError, match="must be a boolean"):
        _aggregate([{"seed": 641, "metrics": metrics}])


@pytest.mark.parametrize(
    ("exit_reasons", "match"),
    [
        (None, "missing"),
        ({}, "missing"),
        ({"not_a_runtime_exit": 2}, "unknown"),
        ({"max_cycles": 1}, "cover every request"),
    ],
)
def test_aggregate_gate_requires_complete_known_exit_reason_evidence(
    exit_reasons, match
):
    metrics = _valid_aggregate_metrics()
    metrics["prediction_stability"]["exit_reasons"] = exit_reasons

    with pytest.raises(ValueError, match=match):
        _aggregate([{"seed": 641, "metrics": metrics}])


def test_aggregate_gate_rejects_prediction_stable_margin_below_floor():
    metrics = _valid_aggregate_metrics()
    stable = metrics["prediction_stability"]["decision_margin"][
        "prediction_stable_observed"
    ]
    stable["minimum"] = single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN / 2
    stable["summary"] = {
        key: single_seed.DEFAULT_PREDICTION_STABILITY_MARGIN / 2
        for key in ("min", "mean", "p50", "p95", "max")
    }

    with pytest.raises(ValueError, match="below the configured floor"):
        _aggregate([{"seed": 641, "metrics": metrics}])


def test_aggregate_gate_binds_margin_rank_and_exact_requested_scope_identity():
    margin_metrics = _valid_aggregate_metrics()
    margin_metrics["prediction_stability"]["prediction_margin"][
        "configured_minimum"
    ] = 0.2
    margin_metrics["prediction_stability"]["decision_margin"][
        "configured_minimum"
    ] = 0.2
    with pytest.raises(ValueError, match="configured mode"):
        _aggregate([{"seed": 641, "metrics": margin_metrics}])

    rank_metrics = _valid_aggregate_metrics()
    rank_metrics["prediction_stability"]["decision_margin"][
        "configured_rank_depth"
    ] = 4
    with pytest.raises(ValueError, match="rank-depth telemetry"):
        _aggregate([{"seed": 641, "metrics": rank_metrics}])

    scope_metrics = _fixture_metrics(
        samples=2,
        fixed_correct=2,
        adaptive_correct=2,
        disagreements=0,
        fixed_cycles=3,
        adaptive_total_cycles=4,
        fixed_latency=10,
        adaptive_latency=8,
        latency_reduction=20,
        cycle_reduction=33.333,
        scope_mode="prediction_class_indices",
        scope_indices=[0, 1, 3, 4],
    )
    scope_metrics["comparison"]["decision_fidelity"]["verified_scope"][
        "requested_normalized_class_indices"
    ] = [0, 1, 2, 4]
    with pytest.raises(ValueError, match="requested class indices"):
        _aggregate(
            [{"seed": 641, "metrics": scope_metrics}],
            prediction_class_indices=[0, 1, 3, 4],
        )


def test_aggregate_gate_requires_zero_top3_order_and_set_disagreements():
    metrics = _fixture_metrics(
        samples=2,
        fixed_correct=2,
        adaptive_correct=2,
        disagreements=0,
        top_k_order_disagreements=1,
        top_k_set_disagreements=1,
        fixed_cycles=3,
        adaptive_total_cycles=4,
        fixed_latency=10,
        adaptive_latency=8,
        latency_reduction=20,
        cycle_reduction=33.333,
    )

    result = _aggregate([{"seed": 641, "metrics": metrics}])

    assert result["gates"]["passed"] is False
    assert result["gates"]["checks"]["zero_top1_disagreements"]["passed"] is True
    assert result["gates"]["checks"]["zero_top_k_order_disagreements"] == {
        "passed": False,
        "top_k": 3,
        "observed": 1,
        "required": 0,
    }
    assert result["gates"]["checks"]["zero_top_k_set_disagreements"][
        "passed"
    ] is False


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
        "source/chat_app.py",
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


@pytest.mark.parametrize("margin", [-0.1, float("nan"), float("inf")])
def test_run_gate_rejects_invalid_prediction_stability_margin(tmp_path, margin):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="prediction_stability_margin|finite"):
        gate.run_gate(
            weights=weights,
            seeds=[641],
            samples_per_seed=1,
            device=torch.device("cpu"),
            device_info={"resolved": "cpu"},
            prediction_stability_margin=margin,
            model_loader=lambda *_args: object(),
            provenance={"fixture": True},
        )


def test_run_gate_requires_rank_depth_to_cover_decision_top_k(tmp_path):
    weights = tmp_path / "checkpoint.pth"
    weights.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="rank_depth must be at least decision_top_k"):
        gate.run_gate(
            weights=weights,
            seeds=[641],
            samples_per_seed=1,
            device=torch.device("cpu"),
            device_info={"resolved": "cpu"},
            prediction_stability_rank_depth=2,
            decision_top_k=3,
            model_loader=lambda *_args: object(),
            provenance={"fixture": True},
        )
