import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import materialize_v51_chat_demo
from materialize_v51_chat_demo import DEFAULT_ARTIFACT_DIR, build_demo_metadata, check_load


def test_v51_demo_metadata_uses_portable_project_paths(monkeypatch):
    weights = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
    metrics = DEFAULT_ARTIFACT_DIR / "benchmark_results.json"
    monkeypatch.setattr(materialize_v51_chat_demo, "_load_metrics", lambda _path: {})

    payload = build_demo_metadata(weights=weights, metrics_path=metrics)

    assert payload["checkpoint_path"] == "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth"
    assert payload["benchmark_metrics"] == "output/benchmark_v51_cognitive_leap_ultra/benchmark_results.json"
    assert payload["prediction_stability_metrics"] == (
        "output/benchmark_v51_cognitive_leap_ultra/prediction_stability_results.json"
    )
    assert payload["benchmark_summary"]["train_seconds"] is None
    assert payload["benchmark_summary"]["prediction_stability"] is None
    assert payload["runtime_defaults"]["prediction_stability_patience"] == 2
    assert payload["runtime_defaults"]["prediction_stability_tol"] == 0.005
    assert payload["runtime_defaults"]["prediction_stability_margin"] == 0.0005
    assert payload["runtime_defaults"]["prediction_stability_rank_depth"] == 3
    assert "New folder (4)" not in payload["checkpoint_path"]
    assert "New folder (4)" not in payload["benchmark_metrics"]


def test_v51_demo_check_forwards_released_prediction_stability_margin(monkeypatch):
    calls = {}

    class StubEngine:
        def __init__(self, device, device_info, defaults):
            calls["defaults"] = defaults

        def load(self, weights, meta):
            return {"loaded": True}

        def chat(self, **kwargs):
            calls["chat"] = kwargs
            return {"response": "ok", "compute": {}, "top_candidates": []}

    monkeypatch.setattr(materialize_v51_chat_demo, "Engine", StubEngine)

    result = check_load(Path("weights.pth"), Path("meta.json"))

    assert result["chat_response"] == "ok"
    assert calls["defaults"]["prediction_stability_margin"] == 0.0005
    assert calls["chat"]["prediction_stability_margin"] == 0.0005
    assert calls["defaults"]["prediction_stability_rank_depth"] == 3
    assert calls["chat"]["prediction_stability_rank_depth"] == 3
