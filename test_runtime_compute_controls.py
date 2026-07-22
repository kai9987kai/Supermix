import importlib.util
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import torch


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import chat_app
from chat_web_app import Engine, build_app


class _RuntimeComputeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.register_buffer("last_cycles_used", torch.tensor(0.0))
        self.register_buffer("last_ponder_cost", torch.tensor(0.0))
        self.register_buffer("last_consistency_loss", torch.tensor(0.0))
        self.register_buffer("last_gating_entropy", torch.tensor(0.0))
        self.register_buffer("last_prediction_streak", torch.tensor(0.0))
        self.register_buffer("last_prediction_confidence_delta", torch.tensor(0.0))
        self.register_buffer("last_prediction_margin", torch.tensor(0.0))
        self.register_buffer("last_prediction_decision_margin", torch.tensor(0.0))
        self.register_buffer("last_decision_reference_cycles", torch.tensor(3.0))
        self.register_buffer("last_prediction_rank_depth", torch.tensor(1.0))
        self.register_buffer("last_prediction_class_count", torch.tensor(10.0))
        self.register_buffer("last_prediction_class_selection_valid", torch.tensor(1.0))
        self.last_exit_reason = "not_run"

    def forward(
        self,
        x,
        reasoning_cycles=None,
        adaptive_compute=False,
        exit_tol=1e-3,
        exit_entropy_threshold=0.2,
        prediction_stability_patience=2,
        prediction_stability_tol=5e-3,
        prediction_stability_margin=chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN,
        prediction_stability_rank_depth=chat_app.DEFAULT_PREDICTION_STABILITY_RANK_DEPTH,
    ):
        self.calls.append(
            {
                "reasoning_cycles": reasoning_cycles,
                "adaptive_compute": adaptive_compute,
                "exit_tol": exit_tol,
                "exit_entropy_threshold": exit_entropy_threshold,
                "prediction_stability_patience": prediction_stability_patience,
                "prediction_stability_tol": prediction_stability_tol,
                "prediction_stability_margin": prediction_stability_margin,
            }
        )
        self.last_rank_depth_arg = prediction_stability_rank_depth
        self.last_cycles_used = torch.tensor(float(reasoning_cycles or 3))
        self.last_ponder_cost = torch.tensor(float(reasoning_cycles or 3) - 0.25)
        self.last_consistency_loss = torch.tensor(0.125)
        self.last_gating_entropy = torch.tensor(0.75)
        self.last_prediction_streak = torch.tensor(float(prediction_stability_patience))
        self.last_prediction_confidence_delta = torch.tensor(float(prediction_stability_tol))
        self.last_prediction_margin = torch.tensor(0.25)
        self.last_prediction_decision_margin = torch.tensor(0.125)
        self.last_prediction_rank_depth = torch.tensor(float(prediction_stability_rank_depth))
        self.last_exit_reason = "prediction_stable" if adaptive_compute and prediction_stability_patience else "max_cycles"
        logits = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
        logits[..., 2] = 4.0
        return logits


class _LegacyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)


class _ClassAwareRuntimeComputeModel(_RuntimeComputeModel):
    def forward(self, x, prediction_class_indices=None, **kwargs):
        output = super().forward(x, **kwargs)
        self.calls[-1]["prediction_class_indices"] = prediction_class_indices
        return output


def _candidate(text: str):
    vec = chat_app.text_to_model_input(text, feature_mode="legacy")[0, 0].tolist()
    return {
        "text": text,
        "vec": vec,
        "ctx_vec": vec,
        "bucket_score": 1.0,
        "count": 1,
    }


def test_metadata_bucket_labels_are_bounded_and_invalid_only_metadata_falls_back():
    zero_rows = [_candidate("zero")]
    last_rows = [_candidate("last")]
    raw = {
        "-1": [_candidate("negative")],
        "0": zero_rows,
        str(chat_app.MODEL_CLASSES - 1): last_rows,
        str(chat_app.MODEL_CLASSES): [_candidate("upper-bound")],
        "999": [_candidate("far-out-of-range")],
        "not-a-label": [_candidate("malformed")],
        "2.5": [_candidate("fractional")],
        2.5: [_candidate("numeric-fractional")],
        None: [_candidate("none")],
    }

    assert chat_app._parse_metadata_buckets(raw) == {
        0: zero_rows,
        chat_app.MODEL_CLASSES - 1: last_rows,
    }

    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    engine._parse_buckets({"buckets": raw})
    assert engine.buckets == {0: zero_rows, chat_app.MODEL_CLASSES - 1: last_rows}
    assert engine.available_labels == [0, chat_app.MODEL_CLASSES - 1]

    engine._parse_buckets(
        {
            "buckets": {
                "-1": [_candidate("negative")],
                str(chat_app.MODEL_CLASSES): [_candidate("upper-bound")],
                "bad": [_candidate("malformed")],
            }
        }
    )
    assert engine.buckets == {}
    assert engine.available_labels == list(range(chat_app.MODEL_CLASSES))


def _stub_engine_load(monkeypatch, metadata_by_name):
    monkeypatch.setattr(
        chat_app,
        "load_metadata",
        lambda path: metadata_by_name[Path(path).name],
    )
    monkeypatch.setattr(chat_app, "safe_load_state_dict", lambda _path: {})
    monkeypatch.setattr(
        chat_app,
        "detect_model_size_from_state_dict",
        lambda _state: "cognitive_leap_expert",
    )
    monkeypatch.setattr(
        chat_app,
        "build_model",
        lambda **_kwargs: _RuntimeComputeModel(),
    )
    monkeypatch.setattr(
        chat_app,
        "load_weights_for_model",
        lambda _model, _state, model_size: ([], []),
    )


def _write_load_files(tmp_path: Path, *meta_names: str):
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"stub")
    metas = []
    for name in meta_names:
        path = tmp_path / name
        path.write_text("{}", encoding="utf-8")
        metas.append(path)
    return weights, metas


def smoke_test_runtime_compute_helper_applies_only_supported_kwargs():
    x = torch.zeros(1, 1, 128)
    recursive = _RuntimeComputeModel()

    _, diag = chat_app.forward_with_runtime_compute(
        recursive,
        x,
        reasoning_cycles="128",
        adaptive_compute="on",
        exit_tol="0.25",
        exit_entropy_threshold="0.75",
        prediction_stability_patience="4",
        prediction_stability_tol="0.02",
        prediction_stability_margin="0.15",
    )

    assert recursive.calls[-1] == {
        "reasoning_cycles": chat_app.MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": True,
        "exit_tol": 0.25,
        "exit_entropy_threshold": 0.75,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.02,
        "prediction_stability_margin": 0.15,
    }
    assert diag["applied"] is True
    assert diag["requested_reasoning_cycles"] == chat_app.MAX_RUNTIME_REASONING_CYCLES
    assert diag["cycles_used"] == float(chat_app.MAX_RUNTIME_REASONING_CYCLES)
    assert diag["ponder_cost"] == float(chat_app.MAX_RUNTIME_REASONING_CYCLES) - 0.25
    assert diag["consistency_loss"] == 0.125
    assert diag["gating_entropy"] == 0.75
    assert diag["exit_entropy_threshold"] == 0.75
    assert diag["prediction_streak"] == 4.0
    assert abs(diag["prediction_confidence_delta"] - 0.02) < 1e-6
    assert diag["prediction_stability_margin"] == 0.15
    assert diag["prediction_stability_rank_depth"] == 3
    assert diag["prediction_verifier_active"] is True
    assert diag["prediction_margin"] == 0.25
    assert diag["prediction_decision_margin"] == 0.125
    assert diag["decision_reference_cycles"] == 3.0
    assert diag["prediction_rank_depth"] == 3.0
    assert recursive.last_rank_depth_arg == 3
    assert diag["exit_reason"] == "prediction_stable"

    legacy = _LegacyModel()
    _, legacy_diag = chat_app.forward_with_runtime_compute(
        legacy,
        x,
        reasoning_cycles=8,
        adaptive_compute=True,
        exit_tol=0.5,
    )

    assert legacy.calls == 1
    assert legacy_diag["applied"] is False
    assert legacy_diag["cycles_used"] is None
    assert legacy_diag["prediction_verifier_active"] is False
    assert legacy_diag["prediction_margin"] is None


def test_prediction_rank_depth_coercion_and_inactive_telemetry_suppression():
    assert chat_app._coerce_prediction_stability_rank_depth("5") == 5
    assert chat_app._coerce_prediction_stability_rank_depth(0) == 0
    for invalid in (-1, "bad", float("inf")):
        assert chat_app._coerce_prediction_stability_rank_depth(invalid) == 3

    x = torch.zeros(1, 1, 128)
    for controls in (
        {"adaptive_compute": False},
        {"adaptive_compute": True, "prediction_stability_patience": 0},
        {"adaptive_compute": True, "prediction_stability_rank_depth": 0},
    ):
        _, diag = chat_app.forward_with_runtime_compute(
            _RuntimeComputeModel(), x, reasoning_cycles=3, **controls
        )
        assert diag["prediction_verifier_active"] is False
        for key in (
            "prediction_streak",
            "prediction_confidence_delta",
            "prediction_margin",
            "prediction_decision_margin",
            "decision_reference_cycles",
            "prediction_rank_depth",
            "prediction_class_count",
            "prediction_class_selection_valid",
        ):
            assert diag[key] is None


def test_budget_evaluator_forwards_complete_adaptive_verifier_controls():
    model = _RuntimeComputeModel()
    rows = chat_app.evaluate_runtime_compute_budgets(
        model,
        torch.zeros(1, 1, 128),
        [2],
        cycles=[2],
        adaptive_compute=True,
        exit_tol=0.04,
        exit_entropy_threshold=0.6,
        prediction_stability_patience=4,
        prediction_stability_tol=0.03,
        prediction_stability_margin=0.02,
        prediction_stability_rank_depth=4,
    )
    assert model.calls[-1] == {
        "reasoning_cycles": 2,
        "adaptive_compute": True,
        "exit_tol": 0.04,
        "exit_entropy_threshold": 0.6,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.03,
        "prediction_stability_margin": 0.02,
    }
    assert model.last_rank_depth_arg == 4
    assert rows[0]["compute"]["prediction_stability_rank_depth"] == 4
    assert rows[0]["compute"]["prediction_verifier_active"] is True


def smoke_test_auto_reasoning_budget_selects_cycles_from_context():
    x = torch.zeros(1, 1, 128)
    recursive = _RuntimeComputeModel()

    _, diag = chat_app.forward_with_runtime_compute(
        recursive,
        x,
        reasoning_cycles="auto",
        adaptive_compute=False,
        auto_reasoning_context=(
            "Debug this traceback, inspect the runtime integration, compare benchmarks, "
            "and verify the fix with tests."
        ),
    )

    assert diag["reasoning_budget_mode"] == "auto"
    assert diag["auto_reasoning_policy"]["cycles"] == 16
    assert "code_or_debug" in diag["auto_reasoning_policy"]["reasons"]
    assert recursive.calls[-1]["reasoning_cycles"] == 16


def smoke_test_web_engine_forwards_runtime_compute_controls():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("Runtime compute response.")]}
        engine.available_labels = [2]

    result = engine.chat(
        session_id="s",
        user_text="use more reasoning",
        reasoning_cycles=5,
        adaptive_compute=True,
        adaptive_exit_tol=0.125,
        adaptive_exit_entropy=0.5,
        prediction_stability_patience=3,
        prediction_stability_tol=0.01,
        prediction_stability_margin=0.12,
    )

    assert result["ok"] is True
    assert result["compute"]["applied"] is True
    assert result["compute"]["cycles_used"] == 5.0
    assert model.calls[-1] == {
        "reasoning_cycles": 5,
        "adaptive_compute": True,
        "exit_tol": 0.125,
        "exit_entropy_threshold": 0.5,
        "prediction_stability_patience": 3,
        "prediction_stability_tol": 0.01,
        "prediction_stability_margin": 0.12,
    }


def test_web_engine_scopes_prediction_verifier_to_available_labels():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _ClassAwareRuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_ultra_expert"
        engine.buckets = {
            2: [_candidate("Allowed two.")],
            9: [_candidate("Allowed nine.")],
        }
        engine.available_labels = [2, 9]

    result = engine.chat(
        session_id="partial-label-verifier",
        user_text="scope the verifier",
        adaptive_compute=True,
    )

    assert result["ok"] is True
    assert model.calls[-1]["prediction_class_indices"] == [2, 9]
    assert result["compute"]["prediction_class_indices"] == [2, 9]


def smoke_test_web_engine_auto_runtime_compute_controls():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("Auto compute response.")]}
        engine.available_labels = [2]

    result = engine.chat(
        session_id="auto",
        user_text="debug this runtime error and verify the benchmark integration with tests",
        reasoning_cycles="auto",
    )

    assert result["ok"] is True
    assert result["compute"]["reasoning_budget_mode"] == "auto"
    assert result["compute"]["selected_reasoning_cycles"] == 16
    assert model.calls[-1]["reasoning_cycles"] == 16


def smoke_test_web_api_accepts_runtime_compute_payload():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("API compute response.")]}
        engine.available_labels = [2]

    app = build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "session_id": "api",
            "message": "think harder",
            "reasoning_cycles": 7,
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.05,
            "adaptive_exit_entropy": 0.4,
            "prediction_stability_patience": 5,
            "prediction_stability_tol": 0.03,
            "prediction_stability_margin": 0.11,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["compute"]["applied"] is True
    assert payload["compute"]["cycles_used"] == 7.0
    assert model.calls[-1] == {
        "reasoning_cycles": 7,
        "adaptive_compute": True,
        "exit_tol": 0.05,
        "exit_entropy_threshold": 0.4,
        "prediction_stability_patience": 5,
        "prediction_stability_tol": 0.03,
        "prediction_stability_margin": 0.11,
    }

    for requested_margin, expected_margin in (
        (-1.0, chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN),
        (0.0, 0.0),
    ):
        response = client.post(
            "/api/chat",
            json={
                "session_id": f"api-margin-{requested_margin}",
                "message": "validate stability margin",
                "adaptive_compute": True,
                "prediction_stability_margin": requested_margin,
            },
        )
        assert response.status_code == 200, response.get_data(as_text=True)
        assert response.get_json()["compute"]["prediction_stability_margin"] == expected_margin
        assert model.calls[-1]["prediction_stability_margin"] == expected_margin


def smoke_test_web_api_accepts_auto_runtime_compute_payload():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("API auto response.")]}
        engine.available_labels = [2]

    app = build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "session_id": "api-auto",
            "message": "analyze this traceback, compare benchmarks, verify tests, and fix runtime integration",
            "reasoning_cycles": "auto",
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["compute"]["reasoning_budget_mode"] == "auto"
    assert payload["compute"]["selected_reasoning_cycles"] == 16
    assert model.calls[-1]["reasoning_cycles"] == 16


def smoke_test_compute_sweep_reports_budget_rows_without_mutating_session():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("Sweep response.")]}
        engine.available_labels = [2]
        engine.sessions["sweep"] = [("prior", "answer")]

    before_history = list(engine.sessions["sweep"])
    result = engine.compute_sweep(
        session_id="sweep",
        user_text="compare compute budgets",
        cycles=[1, "3", 999],
        adaptive_compute=True,
        adaptive_exit_tol=0.2,
        adaptive_exit_entropy=0.6,
        prediction_stability_patience=6,
        prediction_stability_tol=0.04,
        prediction_stability_margin=0.09,
    )

    assert result["ok"] is True
    assert [row["requested_cycles"] for row in result["rows"]] == [
        1,
        3,
        chat_app.MAX_RUNTIME_REASONING_CYCLES,
    ]
    assert [row["cycles_used"] for row in result["rows"]] == [
        1.0,
        3.0,
        float(chat_app.MAX_RUNTIME_REASONING_CYCLES),
    ]
    assert all(row["predicted_label"] == 2 for row in result["rows"])
    assert engine.sessions["sweep"] == before_history
    assert model.calls[-1] == {
        "reasoning_cycles": chat_app.MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": True,
        "exit_tol": 0.2,
        "exit_entropy_threshold": 0.6,
        "prediction_stability_patience": 6,
        "prediction_stability_tol": 0.04,
        "prediction_stability_margin": 0.09,
    }


def smoke_test_web_api_accepts_compute_sweep_payload():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    model = _RuntimeComputeModel()
    with engine.lock:
        engine.model = model
        engine.feature_mode = "legacy"
        engine.model_size = "cognitive_leap_expert"
        engine.buckets = {2: [_candidate("API sweep response.")]}
        engine.available_labels = [2]

    app = build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/compute_sweep",
        json={
            "session_id": "api-sweep",
            "message": "sweep this prompt",
            "cycles": [1, 2],
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.15,
            "adaptive_exit_entropy": 0.55,
            "prediction_stability_patience": 4,
            "prediction_stability_tol": 0.025,
            "prediction_stability_margin": 0.08,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["ok"] is True
    assert [row["requested_cycles"] for row in payload["rows"]] == [1, 2]
    assert payload["rows"][0]["compute"]["applied"] is True
    assert payload["rows"][1]["compute"]["exit_entropy_threshold"] == 0.55
    assert payload["rows"][1]["compute"]["prediction_stability_patience"] == 4
    assert abs(payload["rows"][1]["compute"]["prediction_stability_tol"] - 0.025) < 1e-6
    assert payload["rows"][1]["compute"]["prediction_stability_margin"] == 0.08
    assert "api-sweep" not in engine.sessions


def test_engine_load_applies_whitelisted_metadata_compute_defaults_and_request_overrides(
    tmp_path: Path,
    monkeypatch,
):
    metadata = {
        "feature_mode": "legacy",
        "model_size": "cognitive_leap_expert",
        "runtime_defaults": {
            "reasoning_cycles": "7",
            "adaptive_compute": "on",
            "adaptive_exit_tol": "0.25",
            "adaptive_exit_entropy": "0.75",
            "prediction_stability_patience": "4",
            "prediction_stability_tol": "0.02",
            "prediction_stability_margin": "0.13",
            "pool_mode": "topk",
            "untrusted_extra": "ignored",
        },
        "buckets": {"2": [_candidate("Metadata compute response.")]},
    }
    _stub_engine_load(monkeypatch, {"meta.json": metadata})
    weights, (meta_path,) = _write_load_files(tmp_path, "meta.json")
    engine = Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {"model_size": "auto", "pool_mode": "all", "max_turns": 2},
    )

    loaded = engine.load(str(weights), str(meta_path))

    assert loaded["reasoning_cycles"] == 7
    assert loaded["adaptive_compute"] is True
    assert loaded["adaptive_exit_tol"] == 0.25
    assert loaded["adaptive_exit_entropy"] == 0.75
    assert loaded["prediction_stability_patience"] == 4
    assert loaded["prediction_stability_tol"] == 0.02
    assert loaded["prediction_stability_margin"] == 0.13
    assert engine.defaults["pool_mode"] == "all"
    assert "untrusted_extra" not in engine.defaults

    engine.chat(session_id="meta-defaults", user_text="use metadata defaults")
    assert engine.model.calls[-1] == {
        "reasoning_cycles": 7,
        "adaptive_compute": True,
        "exit_tol": 0.25,
        "exit_entropy_threshold": 0.75,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.02,
        "prediction_stability_margin": 0.13,
    }

    engine.chat(
        session_id="request-overrides",
        user_text="override every compute default",
        reasoning_cycles=5,
        adaptive_compute=True,
        adaptive_exit_tol=0.125,
        adaptive_exit_entropy=0.5,
        prediction_stability_patience=3,
        prediction_stability_tol=0.01,
        prediction_stability_margin=0.06,
    )
    assert engine.model.calls[-1] == {
        "reasoning_cycles": 5,
        "adaptive_compute": True,
        "exit_tol": 0.125,
        "exit_entropy_threshold": 0.5,
        "prediction_stability_patience": 3,
        "prediction_stability_tol": 0.01,
        "prediction_stability_margin": 0.06,
    }

    disabled = engine.chat(
        session_id="request-disables-adaptive",
        user_text="disable adaptive compute for this request",
        adaptive_compute=False,
    )
    assert disabled["compute"]["adaptive_compute"] is False
    assert engine.model.calls[-1]["adaptive_compute"] is False


def test_engine_load_rebuilds_defaults_and_constructor_values_win(
    tmp_path: Path,
    monkeypatch,
):
    first_meta = {
        "feature_mode": "legacy",
        "model_size": "cognitive_leap_expert",
        "runtime_defaults": {
            "reasoning_cycles": 8,
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.4,
            "adaptive_exit_entropy": 0.8,
            "prediction_stability_patience": 6,
            "prediction_stability_tol": 0.04,
            "prediction_stability_margin": 0.07,
        },
    }
    second_meta = {
        "feature_mode": "legacy",
        "model_size": "cognitive_leap_expert",
        "runtime_defaults": {"adaptive_compute": False},
    }
    metadata_by_name = {"first.json": first_meta, "second.json": second_meta}
    _stub_engine_load(monkeypatch, metadata_by_name)
    weights, (first_path, second_path) = _write_load_files(
        tmp_path,
        "first.json",
        "second.json",
    )

    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    engine.load(str(weights), str(first_path))
    assert engine.status()["reasoning_cycles"] == 8
    assert engine.status()["prediction_stability_patience"] == 6
    assert engine.status()["prediction_stability_margin"] == 0.07

    reloaded = engine.load(str(weights), str(second_path))
    assert reloaded["reasoning_cycles"] == "default"
    assert reloaded["adaptive_compute"] is False
    assert reloaded["adaptive_exit_tol"] == 1e-3
    assert reloaded["adaptive_exit_entropy"] == chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY
    assert reloaded["prediction_stability_patience"] == chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE
    assert reloaded["prediction_stability_tol"] == chat_app.DEFAULT_PREDICTION_STABILITY_TOL
    assert reloaded["prediction_stability_margin"] == chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN

    explicit = {
        "pool_mode": "all",
        "reasoning_cycles": 3,
        "adaptive_compute": False,
        "adaptive_exit_tol": 0.1,
        "adaptive_exit_entropy": 0.3,
        "prediction_stability_patience": 2,
        "prediction_stability_tol": 0.005,
        "prediction_stability_margin": 0.02,
    }
    explicit_engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, explicit)
    explicit_status = explicit_engine.load(str(weights), str(first_path))
    assert explicit_status["reasoning_cycles"] == 3
    assert explicit_status["adaptive_compute"] is False
    assert explicit_status["adaptive_exit_tol"] == 0.1
    assert explicit_status["adaptive_exit_entropy"] == 0.3
    assert explicit_status["prediction_stability_patience"] == 2
    assert explicit_status["prediction_stability_tol"] == 0.005
    assert explicit_status["prediction_stability_margin"] == 0.02


def test_cli_omits_unspecified_compute_defaults_so_metadata_can_win():
    import chat_web_app

    assert chat_web_app._normalize_runtime_compute_defaults(
        {"prediction_stability_margin": -1}
    )["prediction_stability_margin"] == chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
    assert chat_web_app._normalize_runtime_compute_defaults(
        {"prediction_stability_margin": 0}
    )["prediction_stability_margin"] == 0.0

    unspecified = Namespace(
        reasoning_cycles=None,
        adaptive_compute=None,
        adaptive_exit_tol=None,
        adaptive_exit_entropy=None,
        prediction_stability_patience=None,
        prediction_stability_tol=None,
        prediction_stability_margin=None,
    )
    assert chat_web_app._runtime_compute_cli_overrides(unspecified) == {}

    explicit = Namespace(
        reasoning_cycles="auto",
        adaptive_compute=True,
        adaptive_exit_tol=0.02,
        adaptive_exit_entropy=0.4,
        prediction_stability_patience=4,
        prediction_stability_tol=0.01,
        prediction_stability_margin=0.03,
    )
    assert chat_web_app._runtime_compute_cli_overrides(explicit) == {
        "reasoning_cycles": "auto",
        "adaptive_compute": True,
        "adaptive_exit_tol": 0.02,
        "adaptive_exit_entropy": 0.4,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.01,
        "prediction_stability_margin": 0.03,
    }

    explicit_disabled = Namespace(adaptive_compute=False)
    assert chat_web_app._runtime_compute_cli_overrides(explicit_disabled) == {
        "adaptive_compute": False,
    }

    explicit_auto_disabled = Namespace(auto_compute=False)
    assert chat_web_app._runtime_compute_cli_overrides(explicit_auto_disabled) == {
        "auto_compute": False,
    }

    explicit_auto_enabled = Namespace(auto_compute=True)
    assert chat_web_app._runtime_compute_cli_overrides(explicit_auto_enabled) == {
        "auto_compute": True,
    }


def test_root_chat_web_app_compatibility_forwards_compute_helpers():
    root_entrypoint = Path(__file__).resolve().parent / "chat_web_app.py"
    spec = importlib.util.spec_from_file_location(
        "_root_chat_web_app_compatibility_test",
        root_entrypoint,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._normalize_runtime_compute_defaults(
        {"prediction_stability_margin": -1}
    )["prediction_stability_margin"] == chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
    assert module._runtime_compute_cli_overrides(Namespace(adaptive_compute=False)) == {
        "adaptive_compute": False,
    }


def test_runtime_compute_smoke_suite():
    smoke_test_runtime_compute_helper_applies_only_supported_kwargs()
    smoke_test_auto_reasoning_budget_selects_cycles_from_context()
    smoke_test_web_engine_forwards_runtime_compute_controls()
    smoke_test_web_engine_auto_runtime_compute_controls()
    smoke_test_web_api_accepts_runtime_compute_payload()
    smoke_test_web_api_accepts_auto_runtime_compute_payload()
    smoke_test_compute_sweep_reports_budget_rows_without_mutating_session()
    smoke_test_web_api_accepts_compute_sweep_payload()


def test_packaged_context_mix_v4_matches_source_runtime():
    runtime_path = Path("runtime_python/chat_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("supermix_runtime_chat_pipeline", runtime_path)
    assert spec is not None and spec.loader is not None
    runtime_pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime_pipeline)

    context = (
        "system: reasoning_budget=deep\n"
        "system: knowledge_recency=latest\n"
        "system: source_quality=research\n"
        "user: continue and improve the newest arxiv research implementation"
    )
    source_vector = chat_app.text_to_model_input(context, feature_mode="context_mix_v4")
    runtime_vector = runtime_pipeline.text_to_model_input(context, feature_mode="context_mix_v4")

    assert runtime_pipeline.resolve_feature_mode("context_mix_v3", smarter_auto=True) == "context_mix_v4"
    assert torch.allclose(source_vector, runtime_vector)


def test_packaged_runtime_rejects_out_of_range_metadata_bucket_labels():
    runtime_dir = Path("runtime_python").resolve()
    script = f"""
import json
import sys
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_app
import chat_web_app

raw = {{
    "-1": [{{"text": "negative"}}],
    "0": [{{"text": "valid"}}],
    str(chat_app.MODEL_CLASSES): [{{"text": "upper-bound"}}],
    "not-a-label": [{{"text": "malformed"}}],
}}
parsed = chat_app._parse_metadata_buckets(raw)
engine = chat_web_app.Engine(torch.device("cpu"), {{"resolved": "cpu"}}, {{"pool_mode": "all"}})
engine._parse_buckets({{"buckets": raw}})
valid = {{"parsed": sorted(parsed), "engine": sorted(engine.buckets), "labels": engine.available_labels}}
engine._parse_buckets({{"buckets": {{"-1": [{{"text": "bad"}}], str(chat_app.MODEL_CLASSES): [{{"text": "bad"}}], "bad": [{{"text": "bad"}}]}}}})
print(json.dumps({{"valid": valid, "fallback_buckets": engine.buckets, "fallback_labels": engine.available_labels}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["valid"] == {"parsed": [0], "engine": [0], "labels": [0]}
    assert payload["fallback_buckets"] == {}
    assert payload["fallback_labels"] == list(range(chat_app.MODEL_CLASSES))


def test_packaged_web_runtime_rebuilds_metadata_compute_defaults():
    runtime_dir = Path("runtime_python").resolve()
    script = f"""
import json
import sys
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_web_app

metadata = {{
    "runtime_defaults": {{
        "reasoning_cycles": 8,
        "adaptive_compute": True,
        "adaptive_exit_tol": 0.4,
        "prediction_stability_patience": 6,
        "prediction_stability_margin": 0.05,
    }}
}}
plain = chat_web_app.Engine(torch.device("cpu"), {{"resolved": "cpu"}}, {{"pool_mode": "all"}})
first = plain._build_effective_defaults(metadata)
clean = plain._build_effective_defaults({{"runtime_defaults": {{"adaptive_compute": False}}}})
explicit = chat_web_app.Engine(
    torch.device("cpu"),
    {{"resolved": "cpu"}},
    {{"pool_mode": "all", "reasoning_cycles": 3, "adaptive_compute": False}},
)
overridden = explicit._build_effective_defaults(metadata)
print(json.dumps({{"first": first, "clean": clean, "overridden": overridden}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["first"]["reasoning_cycles"] == 8
    assert payload["first"]["adaptive_compute"] is True
    assert payload["first"]["adaptive_exit_tol"] == 0.4
    assert payload["first"]["prediction_stability_patience"] == 6
    assert payload["first"]["prediction_stability_margin"] == 0.05
    assert payload["clean"]["reasoning_cycles"] is None
    assert payload["clean"]["adaptive_compute"] is False
    assert payload["clean"]["adaptive_exit_tol"] == 1e-3
    assert payload["clean"]["prediction_stability_margin"] == chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
    assert payload["overridden"]["reasoning_cycles"] == 3
    assert payload["overridden"]["adaptive_compute"] is False


def test_source_and_packaged_web_uis_expose_prediction_stability_margin():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    source_html = build_app(engine, "weights.pth", "meta.json").test_client().get("/").get_data(as_text=True)
    packaged_source = Path("runtime_python/chat_web_app.py").read_text(encoding="utf-8")

    for text in (source_html, packaged_source):
        assert "stabilityMargin" in text
        assert "stabilityRankDepth" in text
        assert "prediction_stability_margin" in text
        assert "prediction_stability_rank_depth" in text
        assert "prediction_margin" in text
        assert "prediction_decision_margin" in text
        assert "decision_reference_cycles" in text
        assert "prediction_verifier_active" in text
        assert "prediction_class_count" in text
        assert "fmtNum(compute.prediction_stability_margin,6)" in text
        assert "margin floor" in text
        assert "step='0.0001' value='0.0005'" in text
        assert "checkpoint/workload-calibrated default" in text


def test_source_and_packaged_web_uis_omit_blank_optional_verifier_inputs_but_preserve_zero():
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    source_html = build_app(engine, "weights.pth", "meta.json").test_client().get("/").get_data(as_text=True)
    packaged_source = Path("runtime_python/chat_web_app.py").read_text(encoding="utf-8")

    for text in (source_html, packaged_source):
        assert (
            "function addOptionalFiniteNumber(payload,key,input)"
            "{const raw=input.value.trim();if(raw==='')return;"
            "const value=Number(raw);if(Number.isFinite(value))payload[key]=value;}"
        ) in text
        for key in (
            "prediction_stability_margin",
            "prediction_stability_rank_depth",
        ):
            assert text.count(f"addOptionalFiniteNumber(payload,'{key}'") == 2
            assert f"{key}:Number(" not in text
        assert "jpost('/api/chat',payload)" in text
        assert "jpost('/api/compute_sweep',payload)" in text


def test_packaged_web_api_forwards_prediction_stability_margin():
    runtime_dir = Path("runtime_python").resolve()
    script = f"""
import json
import sys
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_web_app

class Model(torch.nn.Module):
    def forward(
        self,
        x,
        reasoning_cycles=None,
        adaptive_compute=False,
        exit_tol=1e-3,
        exit_entropy_threshold=0.2,
        prediction_stability_patience=2,
        prediction_stability_tol=5e-3,
        prediction_stability_margin=5e-4,
        prediction_stability_rank_depth=3,
    ):
        logits = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
        logits[..., 2] = 4.0
        return logits

engine = chat_web_app.Engine(torch.device("cpu"), {{"resolved": "cpu"}}, {{"pool_mode": "all"}})
engine.model = Model().eval()
engine.available_labels = [2]
client = chat_web_app.build_app(engine, "weights.pth", "meta.json").test_client()
results = {{}}
for name, margin in (("positive", 0.17), ("negative", -1.0), ("zero", 0.0)):
    response = client.post(
        "/api/compute_sweep",
        json={{
            "session_id": "packaged-margin-" + name,
            "message": "compare compute",
            "cycles": [1],
            "adaptive_compute": True,
            "prediction_stability_margin": margin,
            "prediction_stability_rank_depth": 4,
        }},
    )
    results[name] = {{"status": response.status_code, "payload": response.get_json()}}
print(json.dumps(results, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert result["positive"]["status"] == 200
    assert result["positive"]["payload"]["rows"][0]["compute"]["prediction_stability_margin"] == 0.17
    assert result["positive"]["payload"]["rows"][0]["compute"]["prediction_stability_rank_depth"] == 4
    assert result["positive"]["payload"]["rows"][0]["compute"]["prediction_verifier_active"] is True
    assert result["negative"]["payload"]["rows"][0]["compute"]["prediction_stability_margin"] == chat_app.DEFAULT_PREDICTION_STABILITY_MARGIN
    assert result["zero"]["payload"]["rows"][0]["compute"]["prediction_stability_margin"] == 0.0


def test_packaged_web_runtime_escapes_default_paths():
    runtime_dir = Path("runtime_python").resolve()
    script = f"""
import json
import sys
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_web_app

weights = "C:/models/o'hare\\\" onfocus=\\\"alert(1)"
meta = "C:/models/<meta>&config.json"
engine = chat_web_app.Engine(torch.device("cpu"), {{"resolved": "cpu"}}, {{"pool_mode": "all"}})
html = chat_web_app.build_app(engine, weights, meta).test_client().get("/").get_data(as_text=True)
print(json.dumps({{
    "raw_weights_present": weights in html,
    "raw_meta_present": meta in html,
    "escaped_apostrophe": "o&#x27;hare" in html,
    "escaped_quote": "&quot; onfocus=&quot;" in html,
    "escaped_meta": "&lt;meta&gt;&amp;config.json" in html,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload == {
        "raw_weights_present": False,
        "raw_meta_present": False,
        "escaped_apostrophe": True,
        "escaped_quote": True,
        "escaped_meta": True,
    }


def test_packaged_web_runtime_exposes_get_only_shadow_registry_status(tmp_path):
    runtime_dir = Path("runtime_python").resolve()
    missing_registry = (tmp_path / "missing-shadow.sqlite3").resolve()
    script = f"""
import json
import sys
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_web_app

engine = chat_web_app.Engine(
    torch.device("cpu"),
    {{"resolved": "cpu"}},
    {{"pool_mode": "all", "route_shadow_registry_path": {str(missing_registry)!r}}},
)
app = chat_web_app.build_app(engine, "", "")
client = app.test_client()
html = client.get("/").get_data(as_text=True)
status = client.get("/api/route_shadow_registry/status")
payload = status.get_json()
print(json.dumps({{
    "status_code": status.status_code,
    "available": payload["route_shadow_registry"]["available"],
    "read_only": payload["route_shadow_registry"]["read_only"],
    "execution_enabled": payload["route_shadow_registry"]["execution_enabled"],
    "mutation_method": client.post("/api/route_shadow_registry/status", json={{}}).status_code,
    "unknown_seal": client.post("/api/route_shadow_registry/seal", json={{}}).status_code,
    "ui": "routeShadowStatus" in html and "refreshRouteShadowRegistry" in html,
}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload == {
        "status_code": 200,
        "available": False,
        "read_only": True,
        "execution_enabled": False,
        "mutation_method": 405,
        "unknown_seal": 404,
        "ui": True,
    }
    assert not missing_registry.exists()


if __name__ == "__main__":
    test_runtime_compute_smoke_suite()
    print("Runtime compute control smoke tests PASSED!")
