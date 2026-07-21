import json
import sys
import types
from pathlib import Path


def _unexpected_training_call(*args, **kwargs):
    raise AssertionError("The focused v42 unit suite must not enter the v8 training runtime.")


_training_runtime_stub = types.ModuleType("train_omni_collective_v8")
_training_runtime_stub._train_stage_resumable_v8 = _unexpected_training_call
_training_runtime_stub.build_training_rows = _unexpected_training_call
sys.modules.setdefault("train_omni_collective_v8", _training_runtime_stub)


from source.train_omni_collective_v42 import (
    _benchmark_bridge_rows_v42,
    _cleanup_completed_smoke_stage,
    _cleanup_smoke_checkpoint_temps,
    _diversity_rows_v42,
    _promotion_eval_pack_v42,
    _row,
    _smoke_training_rows_v42,
    _stage_resume_dir_v42,
    _teacher_role_rows_v42,
    _turboquant_budget_rows_v42,
    _verifier_repair_rows_v42,
    assemble_v42_training_rows,
    build_training_rows_v42,
    build_training_rows_v42_dry_run,
    build_v42_prep_pack,
)


def _sample_summary():
    return {
        "artifact": "supermix_omni_collective_v41_frontier_20260410.zip",
        "parameter_count": 136200770,
        "dataset_summary": {
            "stage1_rows": 34086,
            "stage2_rows": 34303,
            "source_counts": {"base_a": 20, "base_b": 10},
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


def _write_summary(root: Path) -> Path:
    path = root / "v41_summary.json"
    path.write_text(json.dumps(_sample_summary()), encoding="utf-8")
    return path


def test_v42_curriculum_builders_cover_benchmark_teacher_and_verifier_roles():
    benchmark_rows, benchmark_counts = _benchmark_bridge_rows_v42(seed=42)
    teacher_rows, teacher_counts = _teacher_role_rows_v42(seed=42)
    verifier_rows, verifier_counts = _verifier_repair_rows_v42(seed=42)

    assert len(benchmark_rows) == 9
    assert any("v40_benchmax" in row.response_text for row in benchmark_rows)
    assert any(row.source.endswith("gsm8k") for row in benchmark_rows)
    assert sum(benchmark_counts.values()) == len(benchmark_rows)

    teacher_answers = "\n".join(row.response_text for row in teacher_rows)
    assert len(teacher_rows) == 8
    assert "google/gemma-4-31B-it" in teacher_answers
    assert "Qwen/Qwen3.5-397B-A17B" in teacher_answers
    assert "Qwen/Qwen3-Coder-Next" in teacher_answers
    assert "Qwen/Qwen3-Omni-30B-A3B-Instruct" in teacher_answers
    assert sum(teacher_counts.values()) == len(teacher_rows)

    assert len(verifier_rows) == 8
    assert any("Verifier note:" in row.prompt for row in verifier_rows)
    assert any(row.source.endswith("coding_repair") for row in verifier_rows)
    assert sum(verifier_counts.values()) == len(verifier_rows)


def test_v42_budget_and_diversity_rows_are_bounded_deterministic_and_distinct():
    budget_rows, budget_counts = _turboquant_budget_rows_v42(seed=42)
    diversity_rows, diversity_counts = _diversity_rows_v42(seed=42, limit=7)
    repeated_rows, repeated_counts = _diversity_rows_v42(seed=42, limit=7)

    assert len(budget_rows) == 8
    assert any("not how to quantize weights" in row.response_text for row in budget_rows)
    assert any("compressed evidence" in row.response_text for row in budget_rows)
    assert sum(budget_counts.values()) == len(budget_rows)

    assert len(diversity_rows) == 7
    assert [(row.prompt, row.response_text) for row in diversity_rows] == [
        (row.prompt, row.response_text) for row in repeated_rows
    ]
    assert diversity_counts == repeated_counts
    assert len({row.prompt for row in diversity_rows}) == 7


def test_promotion_eval_pack_v42_adds_all_new_gate_types():
    rows = _promotion_eval_pack_v42()
    v42_rows = [row for row in rows if str(row.get("source", "")).startswith("promotion_eval_v42::")]

    assert len(v42_rows) == 4
    assert {row["focus"] for row in v42_rows} == {
        "benchmark_bridge",
        "budget_reasoning",
        "teacher_roles",
    }
    assert any(row["expected"] == "v40_benchmax" for row in v42_rows)
    assert any(row["expected"] == "Qwen/Qwen3-Coder-Next" for row in v42_rows)


def test_build_v42_prep_pack_writes_each_curriculum_artifact(tmp_path):
    payload = build_v42_prep_pack(
        summary_path=_write_summary(tmp_path),
        output_root=tmp_path / "v42_prep",
        seed=42,
        benchmark_limit=5,
        teacher_route_limit=5,
        verifier_limit=5,
        budget_limit=5,
        diversity_limit=6,
    )

    assert payload["family"] == "omni_collective_v42"
    assert payload["total_new_rows"] == 26
    assert set(payload["row_groups"]) == {
        "benchmark_bridge",
        "teacher_roles",
        "verifier_repair",
        "turboquant_budget",
        "diversity_mix",
    }
    assert all(Path(group["path"]).exists() for group in payload["row_groups"].values())
    assert Path(payload["blueprint_path"]).exists()
    assert Path(payload["promotion_eval_pack_path"]).exists()
    assert Path(payload["summary_path"]).exists()


def test_assemble_v42_training_rows_merges_groups_and_filters_stage1(tmp_path):
    prep = build_v42_prep_pack(
        summary_path=_write_summary(tmp_path),
        output_root=tmp_path / "v42_prep",
        seed=42,
        benchmark_limit=3,
        teacher_route_limit=3,
        verifier_limit=3,
        budget_limit=3,
        diversity_limit=4,
    )
    base_stage1 = [
        _row("Base text", "Base answer", source="base_stage1", intent="general", domain="general")
    ]
    base_full = list(base_stage1) + [
        _row("Base image", "Image answer", source="base_stage2", intent="vision", domain="vision")
    ]

    stage1_rows, full_rows, summary = assemble_v42_training_rows(
        base_stage1_rows=base_stage1,
        base_full_rows=base_full,
        base_summary={"source_counts": {"base_stage1": 1, "base_stage2": 1}},
        prep_root=Path(prep["output_root"]),
    )

    assert len(full_rows) == len(base_full) + prep["total_new_rows"]
    assert all(row.domain not in {"vision", "spatial_3d", "video"} for row in stage1_rows)
    assert any(row.domain == "vision" for row in full_rows)
    assert summary["v42_added_rows"] == prep["total_new_rows"]
    assert set(summary["v42_row_groups"]) == set(prep["row_groups"])
    assert summary["source_counts"]["base_stage1"] == 1


def test_build_training_rows_v42_uses_v41_base_builder(monkeypatch, tmp_path):
    from source import train_omni_collective_v42 as module

    captured = {}

    def _fake_v41_builder(**kwargs):
        captured.update(kwargs)
        base = [_row("Base prompt", "Base answer", source="base_v41", intent="general", domain="general")]
        return base, list(base), {"source_counts": {"base_v41": 1}}

    monkeypatch.setattr(module, "build_training_rows_v41", _fake_v41_builder)
    summary_path = _write_summary(tmp_path)
    stage1_rows, full_rows, summary = build_training_rows_v42(
        repo_root=tmp_path,
        models_dir=tmp_path / "models",
        images_dir=tmp_path / "images",
        summary_path=summary_path,
        output_root=tmp_path / "v42_out",
        seed=42,
        benchmark_limit=2,
        teacher_route_limit=2,
        verifier_limit=2,
        budget_limit=2,
        diversity_limit=2,
        base_communication_limit=7,
        base_disagreement_limit=6,
    )

    assert len(stage1_rows) > 1
    assert len(full_rows) > 1
    assert captured["communication_limit"] == 7
    assert captured["disagreement_limit"] == 6
    assert Path(captured["output_root"]).name == "v41_base"
    assert summary["prep_payload"]["family"] == "omni_collective_v42"
    assert summary["v42_added_rows"] == 10


def test_build_training_rows_v42_dry_run_layers_counts_over_v41(monkeypatch, tmp_path):
    from source import train_omni_collective_v42 as module

    def _fake_v41_dry_run(**kwargs):
        return {
            "estimated_stage1_rows": 100,
            "estimated_stage2_rows": 120,
            "source_counts": {"base_v41": 120},
        }

    monkeypatch.setattr(module, "build_training_rows_v41_dry_run", _fake_v41_dry_run)
    payload = build_training_rows_v42_dry_run(
        summary_path=_write_summary(tmp_path),
        output_root=tmp_path / "v42_out",
        seed=42,
        benchmark_limit=2,
        teacher_route_limit=2,
        verifier_limit=2,
        budget_limit=2,
        diversity_limit=2,
    )

    assert payload["dry_run"] is True
    assert payload["base_mode"] == "frozen_v41_plus_v42_rows"
    assert payload["added_stage2_rows"] == 10
    assert payload["estimated_stage2_rows"] == 130
    assert payload["estimated_stage1_rows"] == 100 + payload["added_stage1_rows"]
    assert payload["source_counts"]["base_v41"] == 120
    assert Path(payload["summary_path"]).exists()


def test_smoke_training_rows_v42_prioritize_v42_and_keep_multimodal_rows():
    stage1_rows = [
        _row("Benchmark route", "Use v40.", source="benchmark_bridge_v42::route", intent="general", domain="reasoning"),
        _row("Repair draft", "Use verifier.", source="teacher_verifier_v42::repair", intent="comparison", domain="coding"),
        _row("Base text", "Base answer.", source="base_v41", intent="general", domain="general"),
    ]
    full_rows = list(stage1_rows) + [
        _row("Base image", "Grounded image.", source="base_vision", intent="vision", domain="vision"),
        _row("3D request", "Use geometry.", source="diversity_mix_v42::persona", intent="general", domain="spatial_3d"),
    ]

    smoke_stage1, smoke_full, summary = _smoke_training_rows_v42(
        stage1_rows=stage1_rows,
        full_rows=full_rows,
        seed=42,
    )

    assert smoke_stage1
    assert smoke_full
    assert all(row.domain not in {"vision", "spatial_3d", "video"} for row in smoke_stage1)
    assert any("_v42" in row.source for row in smoke_full)
    assert any(row.domain in {"vision", "spatial_3d"} for row in smoke_full)
    assert summary["v42_priority_rows"] >= 2
    assert summary["multimodal_rows"] == 2


def test_v42_resume_paths_and_smoke_checkpoint_cleanup_are_mode_safe(tmp_path):
    smoke_dir = _stage_resume_dir_v42(
        tmp_path,
        seed=42,
        distill_limit=8,
        teacher_model_limit=3,
        smoke_train=True,
    )
    frontier_dir = _stage_resume_dir_v42(
        tmp_path,
        seed=42,
        distill_limit=8,
        teacher_model_limit=3,
        smoke_train=False,
    )

    assert smoke_dir != frontier_dir
    assert "smoke_" in smoke_dir.name
    assert "frontier_" in frontier_dir.name
    assert "teacherlimit_3" in smoke_dir.name

    smoke_dir.mkdir(parents=True)
    progress = smoke_dir / "stage1_progress.pt"
    stage1_temp = smoke_dir / "stage1_progress.pt.tmp"
    stage2_temp = smoke_dir / "stage2_progress.pt.tmp"
    progress.write_bytes(b"checkpoint")
    stage1_temp.write_bytes(b"partial")
    stage2_temp.write_bytes(b"partial")

    _cleanup_smoke_checkpoint_temps(smoke_dir)
    assert progress.exists()
    assert not stage1_temp.exists()
    assert not stage2_temp.exists()

    _cleanup_completed_smoke_stage(smoke_dir, "stage1")
    assert not progress.exists()
