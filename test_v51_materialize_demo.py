import os
import sys
from pathlib import Path


sys.path.insert(0, os.path.join(os.getcwd(), "source"))

from materialize_v51_chat_demo import DEFAULT_ARTIFACT_DIR, build_demo_metadata


def test_v51_demo_metadata_uses_portable_project_paths():
    weights = DEFAULT_ARTIFACT_DIR / "cognitive_leap_ultra_v51_trained.pth"
    metrics = DEFAULT_ARTIFACT_DIR / "benchmark_results.json"

    payload = build_demo_metadata(weights=weights, metrics_path=metrics)

    assert payload["checkpoint_path"] == "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth"
    assert payload["benchmark_metrics"] == "output/benchmark_v51_cognitive_leap_ultra/benchmark_results.json"
    assert payload["runtime_defaults"]["prediction_stability_patience"] == 2
    assert payload["runtime_defaults"]["prediction_stability_tol"] == 0.005
    assert "New folder (4)" not in payload["checkpoint_path"]
    assert "New folder (4)" not in payload["benchmark_metrics"]
