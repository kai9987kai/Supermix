import json
import tempfile
from pathlib import Path

from source.prepare_omni_collective_v42 import build_v42_blueprint, latest_v41_summary_path


def _sample_summary():
    return {
        "artifact": "supermix_omni_collective_v41_frontier_20260410.zip",
        "parameter_count": 136200770,
        "dataset_summary": {
            "stage1_rows": 34086,
            "stage2_rows": 34303,
        },
        "stage1": {
            "best_score": 0.3501,
            "val_metrics": {
                "intent_accuracy": 0.7400,
                "response_accuracy": 0.0890,
                "vision_accuracy": 0.1538,
                "domain_accuracy": 0.6946,
            },
        },
        "stage2": {
            "best_score": 0.3575,
            "val_metrics": {
                "intent_accuracy": 0.5782,
                "response_accuracy": 0.1038,
                "vision_accuracy": 0.3846,
                "domain_accuracy": 0.6874,
            },
        },
    }


def test_build_v42_blueprint_raises_targets_from_v41():
    blueprint = build_v42_blueprint(_sample_summary())

    assert blueprint["family"] == "omni_collective_v42"
    assert blueprint["architecture"]["parameter_count_target"] > 136200770
    assert blueprint["architecture"]["expert_count"]["to"] > blueprint["architecture"]["expert_count"]["from"]
    assert blueprint["architecture"]["deliberation_passes"]["to"] > blueprint["architecture"]["deliberation_passes"]["from"]
    assert blueprint["success_gates"]["response_accuracy_min"] > 0.1038


def test_build_v42_blueprint_keeps_new_research_and_teacher_refs():
    blueprint = build_v42_blueprint(_sample_summary())
    names = {entry["name"] for entry in blueprint["recent_method_references"]}
    teachers = {entry["model"] for entry in blueprint["teacher_strategy"]["teacher_roles"]}

    assert "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" in names
    assert "Small Language Models Need Strong Verifiers to Self-Correct Reasoning" in names
    assert "Learning from Partial Chain-of-Thought via Truncated-Reasoning Self-Distillation" in names
    assert "google/gemma-4-31B-it" in names
    assert "Qwen/Qwen3.5-397B-A17B" in names
    assert "Qwen/Qwen3-Coder-Next" in teachers
    assert "Qwen/Qwen3-Omni-30B-A3B-Instruct" in teachers


def test_build_v42_blueprint_adds_v40_bridge_context():
    blueprint = build_v42_blueprint(_sample_summary())

    assert "baseline_findings" in blueprint
    assert "v40_strengths_to_import" in blueprint["baseline_findings"]
    assert "v40_manifest_snapshot" in blueprint
    assert blueprint["teacher_strategy"]["turboquant_runtime_note"]["enabled"] is True


def test_latest_v41_summary_path_prefers_newest_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        older = root / "output" / "supermix_omni_collective_v41_frontier_20260409_010000"
        newer = root / "output" / "supermix_omni_collective_v41_frontier_20260410_020000"
        older.mkdir(parents=True)
        newer.mkdir(parents=True)
        older_path = older / "omni_collective_v41_frontier_summary.json"
        newer_path = newer / "omni_collective_v41_frontier_summary.json"
        older_path.write_text(json.dumps({"artifact": "older"}), encoding="utf-8")
        newer_path.write_text(json.dumps({"artifact": "newer"}), encoding="utf-8")

        assert latest_v41_summary_path(root) == newer_path.resolve()
