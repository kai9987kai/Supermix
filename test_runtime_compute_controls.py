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
    ):
        self.calls.append(
            {
                "reasoning_cycles": reasoning_cycles,
                "adaptive_compute": adaptive_compute,
                "exit_tol": exit_tol,
                "exit_entropy_threshold": exit_entropy_threshold,
                "prediction_stability_patience": prediction_stability_patience,
                "prediction_stability_tol": prediction_stability_tol,
            }
        )
        self.last_cycles_used = torch.tensor(float(reasoning_cycles or 3))
        self.last_ponder_cost = torch.tensor(float(reasoning_cycles or 3) - 0.25)
        self.last_consistency_loss = torch.tensor(0.125)
        self.last_gating_entropy = torch.tensor(0.75)
        self.last_prediction_streak = torch.tensor(float(prediction_stability_patience))
        self.last_prediction_confidence_delta = torch.tensor(float(prediction_stability_tol))
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


def _candidate(text: str):
    vec = chat_app.text_to_model_input(text, feature_mode="legacy")[0, 0].tolist()
    return {
        "text": text,
        "vec": vec,
        "ctx_vec": vec,
        "bucket_score": 1.0,
        "count": 1,
    }


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
    )

    assert recursive.calls[-1] == {
        "reasoning_cycles": chat_app.MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": True,
        "exit_tol": 0.25,
        "exit_entropy_threshold": 0.75,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.02,
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
    }


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
    }


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
    )
    assert engine.model.calls[-1] == {
        "reasoning_cycles": 5,
        "adaptive_compute": True,
        "exit_tol": 0.125,
        "exit_entropy_threshold": 0.5,
        "prediction_stability_patience": 3,
        "prediction_stability_tol": 0.01,
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

    reloaded = engine.load(str(weights), str(second_path))
    assert reloaded["reasoning_cycles"] == "default"
    assert reloaded["adaptive_compute"] is False
    assert reloaded["adaptive_exit_tol"] == 1e-3
    assert reloaded["adaptive_exit_entropy"] == chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY
    assert reloaded["prediction_stability_patience"] == chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE
    assert reloaded["prediction_stability_tol"] == chat_app.DEFAULT_PREDICTION_STABILITY_TOL

    explicit = {
        "pool_mode": "all",
        "reasoning_cycles": 3,
        "adaptive_compute": False,
        "adaptive_exit_tol": 0.1,
        "adaptive_exit_entropy": 0.3,
        "prediction_stability_patience": 2,
        "prediction_stability_tol": 0.005,
    }
    explicit_engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, explicit)
    explicit_status = explicit_engine.load(str(weights), str(first_path))
    assert explicit_status["reasoning_cycles"] == 3
    assert explicit_status["adaptive_compute"] is False
    assert explicit_status["adaptive_exit_tol"] == 0.1
    assert explicit_status["adaptive_exit_entropy"] == 0.3
    assert explicit_status["prediction_stability_patience"] == 2
    assert explicit_status["prediction_stability_tol"] == 0.005


def test_cli_omits_unspecified_compute_defaults_so_metadata_can_win():
    import chat_web_app

    unspecified = Namespace(
        reasoning_cycles=None,
        adaptive_compute=None,
        adaptive_exit_tol=None,
        adaptive_exit_entropy=None,
        prediction_stability_patience=None,
        prediction_stability_tol=None,
    )
    assert chat_web_app._runtime_compute_cli_overrides(unspecified) == {}

    explicit = Namespace(
        reasoning_cycles="auto",
        adaptive_compute=True,
        adaptive_exit_tol=0.02,
        adaptive_exit_entropy=0.4,
        prediction_stability_patience=4,
        prediction_stability_tol=0.01,
    )
    assert chat_web_app._runtime_compute_cli_overrides(explicit) == {
        "reasoning_cycles": "auto",
        "adaptive_compute": True,
        "adaptive_exit_tol": 0.02,
        "adaptive_exit_entropy": 0.4,
        "prediction_stability_patience": 4,
        "prediction_stability_tol": 0.01,
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
    assert payload["clean"]["reasoning_cycles"] is None
    assert payload["clean"]["adaptive_compute"] is False
    assert payload["clean"]["adaptive_exit_tol"] == 1e-3
    assert payload["overridden"]["reasoning_cycles"] == 3
    assert payload["overridden"]["adaptive_compute"] is False


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
