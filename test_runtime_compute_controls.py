import os
import sys

import torch
import torch.nn as nn


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import chat_app
import chat_web_app


class RuntimeAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.last_cycles_used = torch.tensor(0.0)
        self.last_ponder_cost = torch.tensor(0.0)
        self.last_consistency_loss = torch.tensor(0.0)

    def forward(
        self,
        x,
        reasoning_cycles=None,
        adaptive_compute=False,
        exit_tol=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
    ):
        self.calls.append(
            {
                "reasoning_cycles": reasoning_cycles,
                "adaptive_compute": adaptive_compute,
                "exit_tol": exit_tol,
            }
        )
        cycles = int(reasoning_cycles or 3)
        used = min(cycles, 2) if adaptive_compute else cycles
        self.last_cycles_used = torch.tensor(float(used))
        self.last_ponder_cost = torch.tensor(float(used) + 0.25)
        self.last_consistency_loss = torch.tensor(float(exit_tol))
        logits = torch.zeros(x.shape[0], x.shape[1], chat_app.MODEL_CLASSES, device=x.device)
        logits[..., 0] = 3.0
        return logits


class LegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return torch.zeros(x.shape[0], x.shape[1], chat_app.MODEL_CLASSES, device=x.device)


def _bucket_row(text):
    vec = [0.0] * 128
    return {"text": text, "count": 1, "vec": vec, "ctx_vec": vec}


def test_forward_with_runtime_compute_applies_supported_kwargs_only():
    model = RuntimeAwareModel()
    x = torch.zeros(1, 1, 128)

    _, diagnostics = chat_app.forward_with_runtime_compute(
        model,
        x,
        reasoning_cycles=999,
        adaptive_compute=True,
        exit_tol=0.123,
        return_diagnostics=True,
    )

    assert model.calls[-1] == {
        "reasoning_cycles": chat_app.MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": True,
        "exit_tol": 0.123,
    }
    assert diagnostics["supported"] is True
    assert diagnostics["requested_reasoning_cycles"] == chat_app.MAX_RUNTIME_REASONING_CYCLES
    assert diagnostics["cycles_used"] == 2.0
    assert diagnostics["ponder_cost"] == 2.25
    assert diagnostics["consistency_loss"] == 0.123

    legacy = LegacyModel()
    chat_app.forward_with_runtime_compute(
        legacy,
        x,
        reasoning_cycles=8,
        adaptive_compute=True,
        exit_tol=0.5,
    )
    assert legacy.calls == 1
    assert chat_app.model_supports_runtime_compute(legacy) is False


def test_web_engine_forwards_runtime_compute_controls_without_mutating_contract():
    engine = chat_web_app.Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {
            "max_turns": 1,
            "top_labels": 1,
            "pool_mode": "topk",
            "response_temperature": 0.0,
            "temperature": 0.0,
            "style_mode": "balanced",
            "creativity": 0.0,
        },
    )
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("controlled compute answer")]}
    engine.available_labels = [0]

    result = engine.chat(
        session_id="s1",
        user_text="use more compute",
        reasoning_cycles=7,
        adaptive_compute=True,
        adaptive_exit_tol=0.01,
    )

    assert result["ok"] is True
    assert result["response"]
    assert result["compute"]["requested_reasoning_cycles"] == 7
    assert result["compute"]["cycles_used"] == 2.0
    assert result["timing_ms"]["cycles_used"] == 2.0
    assert model.calls[-1]["adaptive_compute"] is True


def test_api_chat_accepts_runtime_compute_payload():
    engine = chat_web_app.Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "topk"})
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("api compute answer")]}
    engine.available_labels = [0]

    app = chat_web_app.build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "session_id": "api-session",
            "message": "turn on adaptive compute",
            "reasoning_cycles": 5,
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.02,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["compute"]["requested_reasoning_cycles"] == 5
    assert payload["compute"]["adaptive_compute"] is True
    assert payload["compute"]["exit_tol"] == 0.02
    assert model.calls[-1]["reasoning_cycles"] == 5


def test_compute_sweep_reports_budget_rows_without_mutating_session():
    engine = chat_web_app.Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "topk", "max_turns": 2})
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("sweep answer")]}
    engine.available_labels = [0]
    engine.sessions["sweep-session"] = [("previous", "answer")]

    result = engine.compute_sweep(
        session_id="sweep-session",
        user_text="compare compute budgets",
        cycles=[1, 3, 8],
        adaptive_compute=True,
        adaptive_exit_tol=0.01,
    )

    assert result["ok"] is True
    assert result["history_turns"] == 1
    assert [row["requested_cycles"] for row in result["rows"]] == [1, 3, 8]
    assert [row["cycles_used"] for row in result["rows"]] == [1.0, 2.0, 2.0]
    assert all(row["predicted_label"] == 0 for row in result["rows"])
    assert engine.sessions["sweep-session"] == [("previous", "answer")]


def test_api_compute_sweep_accepts_payload_without_mutating_session():
    engine = chat_web_app.Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "topk"})
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("api sweep answer")]}
    engine.available_labels = [0]
    engine.sessions["api-sweep"] = [("prior", "reply")]

    app = chat_web_app.build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/compute_sweep",
        json={
            "session_id": "api-sweep",
            "message": "try sweep",
            "cycles": [2, 4],
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.05,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert [row["requested_cycles"] for row in payload["rows"]] == [2, 4]
    assert payload["rows"][0]["compute"]["adaptive_compute"] is True
    assert model.calls[-1]["reasoning_cycles"] == 4
    assert engine.sessions["api-sweep"] == [("prior", "reply")]


if __name__ == "__main__":
    test_forward_with_runtime_compute_applies_supported_kwargs_only()
    test_web_engine_forwards_runtime_compute_controls_without_mutating_contract()
    test_api_chat_accepts_runtime_compute_payload()
    test_compute_sweep_reports_budget_rows_without_mutating_session()
    test_api_compute_sweep_accepts_payload_without_mutating_session()
    print("runtime compute control tests passed")
