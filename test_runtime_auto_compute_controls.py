import json
import os
from pathlib import Path
import subprocess
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
        self.eval()

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
        logits[..., 0] = float(cycles)
        return logits


class ProbabilityScheduleModel(nn.Module):
    def __init__(self, probabilities):
        super().__init__()
        self.probabilities = dict(probabilities)
        self.calls = []
        self.last_cycles_used = torch.tensor(0.0)
        self.eval()

    def forward(
        self,
        x,
        reasoning_cycles=None,
        adaptive_compute=False,
        exit_tol=chat_app.DEFAULT_ADAPTIVE_EXIT_TOL,
        exit_entropy_threshold=chat_app.DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        prediction_stability_patience=chat_app.DEFAULT_PREDICTION_STABILITY_PATIENCE,
        prediction_stability_tol=chat_app.DEFAULT_PREDICTION_STABILITY_TOL,
    ):
        cycles = int(reasoning_cycles or 1)
        self.calls.append(
            {
                "reasoning_cycles": cycles,
                "adaptive_compute": adaptive_compute,
                "exit_tol": exit_tol,
                "exit_entropy_threshold": exit_entropy_threshold,
                "prediction_stability_patience": prediction_stability_patience,
                "prediction_stability_tol": prediction_stability_tol,
            }
        )
        self.last_cycles_used = torch.tensor(float(cycles))
        probability = float(self.probabilities[cycles])
        tail = (1.0 - probability) / (chat_app.MODEL_CLASSES - 1)
        probs = torch.full(
            (chat_app.MODEL_CLASSES,),
            tail,
            dtype=torch.float64,
            device=x.device,
        )
        probs[0] = probability
        return torch.log(probs).reshape(1, 1, -1).expand(x.shape[0], x.shape[1], -1)


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


def test_runtime_compute_budget_helpers_select_earliest_confident_budget():
    model = RuntimeAwareModel()
    x = torch.zeros(1, 1, 128)

    rows = chat_app.evaluate_runtime_compute_budgets(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3, 8],
        adaptive_compute=False,
        exit_tol=0.01,
    )
    plan = chat_app.select_auto_runtime_compute_budget(rows)

    assert [row["requested_cycles"] for row in rows] == [1, 3, 8]
    assert rows[0]["confidence"] < chat_app.DEFAULT_AUTO_COMPUTE_CONFIDENCE
    assert rows[1]["confidence"] >= chat_app.DEFAULT_AUTO_COMPUTE_CONFIDENCE
    assert plan["selected_reasoning_cycles"] == 3
    assert plan["reason"] == "confidence_target"


def test_progressive_auto_compute_reuses_first_qualifying_probe():
    model = RuntimeAwareModel()
    x = torch.zeros(1, 1, 128)

    output, compute, plan = chat_app.progressive_auto_compute_forward(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3, 8],
    )

    assert [call["reasoning_cycles"] for call in model.calls] == [1, 3]
    assert int(output[0, 0].argmax().item()) == 0
    assert float(output[0, 0, 0].item()) == 3.0
    assert plan["schema_version"] == "runtime-auto-compute-plan-v2"
    assert plan["strategy"] == "progressive_accepted_probe"
    assert plan["selected_reasoning_cycles"] == 3
    assert plan["reason"] == "confidence_target"
    assert plan["evaluated_cycles"] == [1, 3]
    assert plan["skipped_cycles"] == [8]
    assert plan["forward_evaluations"] == 2
    assert plan["legacy_forward_evaluations"] == 4
    assert plan["forward_reduction_percent"] == 50.0
    assert plan["reused_probe_output"] is True
    assert plan["selection_semantics"] == "legacy_v1_selection_policy"
    assert plan["probe_control_scope"] == "full_runtime_controls_v2"
    assert compute["auto_compute_plan"] == plan
    assert compute["inference_reused"] is True


def test_progressive_auto_compute_matches_v1_rounded_confidence_boundary():
    x = torch.zeros(1, 1, 128)
    legacy_model = ProbabilityScheduleModel({1: 0.5499996, 3: 0.8})
    rows = chat_app.evaluate_runtime_compute_budgets(
        legacy_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3],
    )
    legacy_plan = chat_app.select_auto_runtime_compute_budget(
        rows,
        confidence_target=0.55,
        entropy_target=0.0,
    )
    progressive_model = ProbabilityScheduleModel({1: 0.5499996, 3: 0.8})
    _, _, progressive_plan = chat_app.progressive_auto_compute_forward(
        progressive_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3],
        confidence_target=0.55,
        entropy_target=0.0,
    )

    assert rows[0]["confidence"] == 0.55
    assert legacy_plan["selected_reasoning_cycles"] == 1
    assert progressive_plan["selected_reasoning_cycles"] == 1
    assert [call["reasoning_cycles"] for call in progressive_model.calls] == [1]


def test_progressive_auto_compute_matches_v1_rounded_entropy_boundary():
    x = torch.zeros(1, 1, 128)
    probability = 0.4728206703246546
    legacy_model = ProbabilityScheduleModel({1: probability, 3: 0.8})
    rows = chat_app.evaluate_runtime_compute_budgets(
        legacy_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3],
    )
    legacy_plan = chat_app.select_auto_runtime_compute_budget(
        rows,
        confidence_target=1.0,
        entropy_target=1.85,
    )
    progressive_model = ProbabilityScheduleModel({1: probability, 3: 0.8})
    _, _, progressive_plan = chat_app.progressive_auto_compute_forward(
        progressive_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3],
        confidence_target=1.0,
        entropy_target=1.85,
    )

    assert rows[0]["entropy"] == 1.85
    assert legacy_plan["selected_reasoning_cycles"] == 1
    assert progressive_plan["selected_reasoning_cycles"] == 1
    assert progressive_plan["reason"] == "entropy_target"


def test_progressive_auto_compute_shadow_cost_cannot_change_fallback(monkeypatch):
    model = ProbabilityScheduleModel({1: 0.1, 3: 0.1})
    x = torch.zeros(1, 1, 128)
    ticks = iter([0.0, 0.030, 0.040, 0.041, 0.091])
    monkeypatch.setattr(chat_app.time, "perf_counter", lambda: next(ticks))
    js_calls = []

    def expensive_shadow(*_args, **_kwargs):
        js_calls.append(chat_app.time.perf_counter())
        return 0.0

    monkeypatch.setattr(chat_app, "_topk_distribution_js_divergence", expensive_shadow)
    _, _, plan = chat_app.progressive_auto_compute_forward(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3],
        confidence_target=1.0,
        entropy_target=0.0,
    )

    assert [row["latency_ms"] for row in plan["rows"]] == [30.0, 1.0]
    assert js_calls == [0.091]
    assert plan["selected_reasoning_cycles"] == 3


def test_progressive_auto_compute_can_fallback_to_nonfinal_probe():
    model = ProbabilityScheduleModel({1: 0.6, 3: 0.2, 8: 0.4})
    x = torch.zeros(1, 1, 128)
    _, _, plan = chat_app.progressive_auto_compute_forward(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3, 8],
        confidence_target=1.0,
        entropy_target=0.0,
    )

    assert plan["selected_index"] == 0
    assert plan["selected_reasoning_cycles"] == 1
    assert plan["rows"][-1]["requested_cycles"] == 8
    terminal_metrics = chat_app.compact_auto_compute_plan_metrics(plan)
    assert terminal_metrics["auto_selected"] == 1
    assert terminal_metrics["auto_forwards"] == "3/4"
    assert terminal_metrics["auto_reused"] is True
    assert "auto_shadow_js" not in terminal_metrics


def test_topk_distribution_js_divergence_is_symmetric_and_bounded():
    left = torch.tensor([0.7, 0.2, 0.1])
    right = torch.tensor([0.1, 0.3, 0.6])

    left_right = chat_app._topk_distribution_js_divergence(left, right, top_k=2)
    right_left = chat_app._topk_distribution_js_divergence(right, left, top_k=2)

    assert chat_app._topk_distribution_js_divergence(left, left, top_k=2) == 0.0
    assert left_right == right_left
    assert 0.0 <= left_right <= 0.6931471806


def test_source_web_diagnostics_resolve_the_selected_probe_row():
    assert "rows[selectedIndex]" in chat_web_app.HTML
    assert "rows[rows.length-1].mutual_stability_shadow" not in chat_web_app.HTML


def test_packaged_runtime_progressive_helper_and_api_smoke():
    root = Path(__file__).resolve().parent
    script = r'''
import json
import sys
from pathlib import Path
import torch

root = Path.cwd()
sys.path.insert(0, str(root / "runtime_python"))
import chat_app
import chat_web_app

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.last_cycles_used = torch.tensor(0.0)

    def forward(self, x, reasoning_cycles=None, **_kwargs):
        cycles = int(reasoning_cycles or 1)
        self.calls.append(cycles)
        self.last_cycles_used = torch.tensor(float(cycles))
        logits = torch.zeros(x.shape[0], x.shape[1], chat_app.MODEL_CLASSES)
        logits[..., 0] = float(cycles)
        return logits

model = Model().eval()
engine = chat_web_app.Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "topk"})
engine.model = model
engine.feature_mode = "legacy"
vec = [0.0] * 128
engine.buckets = {0: [{"text": "packaged progressive answer", "count": 1, "vec": vec, "ctx_vec": vec}]}
engine.available_labels = [0]
client = chat_web_app.build_app(engine, "weights.pth", "meta.json").test_client()
response = client.post("/api/chat", json={"session_id": "packaged", "message": "test", "auto_compute": True})
payload = response.get_json()
assert response.status_code == 200, payload
assert payload["auto_compute_plan"]["reused_probe_output"] is True
assert "rows[selectedIndex]" in chat_web_app.HTML
print(json.dumps({"cycles": model.calls, "strategy": payload["auto_compute_plan"]["strategy"]}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(completed.stdout)

    assert payload["strategy"] == "progressive_accepted_probe"
    assert payload["cycles"] == [1]


def test_progressive_auto_compute_applies_full_v2_probe_controls():
    model = ProbabilityScheduleModel({1: 0.8})
    x = torch.zeros(1, 1, 128)
    _, _, plan = chat_app.progressive_auto_compute_forward(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1],
        adaptive_compute=True,
        exit_tol=0.02,
        exit_entropy_threshold=0.7,
        prediction_stability_patience=4,
        prediction_stability_tol=0.123,
    )

    assert model.calls == [
        {
            "reasoning_cycles": 1,
            "adaptive_compute": True,
            "exit_tol": 0.02,
            "exit_entropy_threshold": 0.7,
            "prediction_stability_patience": 4,
            "prediction_stability_tol": 0.123,
        }
    ]
    assert plan["probe_control_scope"] == "full_runtime_controls_v2"


def test_progressive_auto_compute_early_first_probe_and_no_target_fallback():
    x = torch.zeros(1, 1, 128)
    early_model = RuntimeAwareModel()
    early_output, _, early_plan = chat_app.progressive_auto_compute_forward(
        early_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3, 8],
        confidence_target=0.2,
    )

    assert [call["reasoning_cycles"] for call in early_model.calls] == [1]
    assert float(early_output[0, 0, 0].item()) == 1.0
    assert early_plan["forward_reduction_percent"] == 75.0

    fallback_model = RuntimeAwareModel()
    fallback_output, _, fallback_plan = chat_app.progressive_auto_compute_forward(
        fallback_model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        cycles=[1, 3, 8],
        confidence_target=1.0,
        entropy_target=0.0,
    )

    assert [call["reasoning_cycles"] for call in fallback_model.calls] == [1, 3, 8]
    assert float(fallback_output[0, 0, 0].item()) == 8.0
    assert fallback_plan["reason"] == "best_confidence"
    assert fallback_plan["selected_reasoning_cycles"] == 8
    assert fallback_plan["forward_evaluations"] == 3
    shadow = fallback_plan["rows"][-1]["mutual_stability_shadow"]
    assert shadow["role"] == "shadow_diagnostic_only"
    assert shadow["selection_enabled"] is False
    assert shadow["top1_persistent"] is True
    assert shadow["js_divergence"] is not None


def test_progressive_auto_compute_rejects_training_mode():
    model = RuntimeAwareModel().train()
    x = torch.zeros(1, 1, 128)

    try:
        chat_app.progressive_auto_compute_forward(
            model,
            x,
            list(range(chat_app.MODEL_CLASSES)),
        )
    except RuntimeError as exc:
        assert "model.eval" in str(exc)
    else:
        raise AssertionError("training-mode progressive inference must fail closed")


def test_progressive_auto_compute_sanitizes_non_finite_thresholds():
    model = RuntimeAwareModel()
    x = torch.zeros(1, 1, 128)

    _, _, plan = chat_app.progressive_auto_compute_forward(
        model,
        x,
        list(range(chat_app.MODEL_CLASSES)),
        confidence_target=float("nan"),
        entropy_target=float("inf"),
    )

    assert plan["confidence_target"] == chat_app.DEFAULT_AUTO_COMPUTE_CONFIDENCE
    assert plan["entropy_target"] == chat_app.DEFAULT_AUTO_COMPUTE_ENTROPY


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


def test_chat_auto_compute_selects_confident_budget():
    engine = chat_web_app.Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {"pool_mode": "topk", "auto_compute": True, "adaptive_compute": False},
    )
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("auto compute answer")]}
    engine.available_labels = list(range(chat_app.MODEL_CLASSES))

    result = engine.chat(
        session_id="auto-session",
        user_text="choose the right compute depth",
        auto_compute=True,
    )

    plan = result["auto_compute_plan"]
    assert plan["enabled"] is True
    assert plan["selected_reasoning_cycles"] == 3
    assert plan["reason"] == "confidence_target"
    assert result["compute"]["requested_reasoning_cycles"] == 3
    assert [call["reasoning_cycles"] for call in model.calls] == [1, 3]
    assert plan["reused_probe_output"] is True
    assert engine.sessions["auto-session"], "Auto-compute chat should still append the final turn"


def test_api_chat_accepts_auto_compute_payload():
    engine = chat_web_app.Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "topk"})
    model = RuntimeAwareModel()
    engine.model = model
    engine.feature_mode = "legacy"
    engine.buckets = {0: [_bucket_row("api auto compute answer")]}
    engine.available_labels = list(range(chat_app.MODEL_CLASSES))

    app = chat_web_app.build_app(engine, "weights.pth", "meta.json")
    client = app.test_client()
    response = client.post(
        "/api/chat",
        json={
            "session_id": "api-auto",
            "message": "auto compute please",
            "auto_compute": True,
        },
    )

    assert response.status_code == 200, response.get_data(as_text=True)
    payload = response.get_json()
    assert payload["auto_compute_plan"]["selected_reasoning_cycles"] == 3
    assert payload["compute"]["auto_compute_plan"]["selected_reasoning_cycles"] == 3
    assert payload["auto_compute_plan"]["forward_evaluations"] == 2
    assert payload["auto_compute_plan"]["reused_probe_output"] is True


if __name__ == "__main__":
    test_forward_with_runtime_compute_applies_supported_kwargs_only()
    test_runtime_compute_budget_helpers_select_earliest_confident_budget()
    test_progressive_auto_compute_reuses_first_qualifying_probe()
    test_progressive_auto_compute_early_first_probe_and_no_target_fallback()
    test_progressive_auto_compute_rejects_training_mode()
    test_progressive_auto_compute_sanitizes_non_finite_thresholds()
    test_web_engine_forwards_runtime_compute_controls_without_mutating_contract()
    test_api_chat_accepts_runtime_compute_payload()
    test_compute_sweep_reports_budget_rows_without_mutating_session()
    test_api_compute_sweep_accepts_payload_without_mutating_session()
    test_chat_auto_compute_selects_confident_budget()
    test_api_chat_accepts_auto_compute_payload()
    print("runtime compute control tests passed")
