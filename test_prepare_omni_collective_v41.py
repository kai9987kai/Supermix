import json
import tempfile
from pathlib import Path

from source.prepare_omni_collective_v41 import build_v41_blueprint, latest_v8_summary_path


def _sample_summary():
    return {
        "artifact": "supermix_omni_collective_v8_frontier_20260408.zip",
        "parameter_count": 112025979,
        "dataset_summary": {
            "stage1_rows": 34124,
            "stage2_rows": 34334,
            "teacher_league": {
                "teacher_keys": [
                    "v40_benchmax",
                    "omni_collective_v7",
                    "qwen_v28",
                    "qwen_v30",
                    "omni_collective_v6",
                ]
            },
        },
        "stage1": {
            "best_score": 0.3722,
            "val_metrics": {
                "intent_accuracy": 0.6962,
                "response_accuracy": 0.0605,
                "vision_accuracy": 0.5384,
                "domain_accuracy": 0.625,
            },
        },
        "stage2": {
            "best_score": 0.4183,
            "val_metrics": {
                "intent_accuracy": 0.6860,
                "response_accuracy": 0.0935,
                "vision_accuracy": 0.6923,
                "domain_accuracy": 0.6761,
            },
        },
    }


def test_build_v41_blueprint_raises_targets_from_v8():
    blueprint = build_v41_blueprint(_sample_summary())

    assert blueprint["family"] == "omni_collective_v41"
    assert blueprint["architecture"]["parameter_count_target"] > 112025979
    assert blueprint["architecture"]["parameter_count_target"] <= 132000000
    assert blueprint["architecture"]["expert_count"]["to"] > blueprint["architecture"]["expert_count"]["from"]
    assert blueprint["architecture"]["deliberation_passes"]["to"] > blueprint["architecture"]["deliberation_passes"]["from"]
    assert blueprint["success_gates"]["response_accuracy_min"] > 0.0935
    assert blueprint["teacher_strategy"]["primary_teachers"][0] == "omni_collective_v8"


def test_build_v41_blueprint_keeps_recent_method_references():
    blueprint = build_v41_blueprint(_sample_summary())
    names = {entry["name"] for entry in blueprint["recent_method_references"]}

    assert "DeepSeek-V3 Technical Report" in names
    assert "s1: Simple test-time scaling" in names
    assert "LIMO: Less Is More for Reasoning" in names
    assert "RefineCoder" in names
    assert "Self-Play Fine-Tuning (SPIN)" in names


def test_latest_v8_summary_path_prefers_newest_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        older = root / "output" / "supermix_omni_collective_v8_frontier_20260407"
        newer = root / "output" / "supermix_omni_collective_v8_frontier_20260408"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        older_path = older / "omni_collective_v8_frontier_summary.json"
        newer_path = newer / "omni_collective_v8_frontier_summary.json"
        older_path.write_text(json.dumps({"artifact": "older"}), encoding="utf-8")
        newer_path.write_text(json.dumps({"artifact": "newer"}), encoding="utf-8")

        assert latest_v8_summary_path(root) == newer_path.resolve()
