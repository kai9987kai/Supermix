import json
import json
import tempfile
from pathlib import Path

from source.train_omni_collective_v41 import (
    _communication_polish_rows_v41,
    _code_critique_repair_rows_v41,
    _latent_plan_rows_v41,
    _reasoning_budget_rows_v41,
    _stage_resume_dir_v41,
    _smoke_training_rows_v41,
    _teacher_disagreement_rows_v41,
    _row,
    assemble_v41_training_rows,
    build_training_rows_v41,
    build_training_rows_v41_dry_run,
    build_v41_prep_pack,
)


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
        "sample_outputs": [
            {
                "prompt": "Which local model is best for benchmark-focused reasoning prompts?",
                "answer": "Upload an image first, then I can identify the science concept and explain the visual clues.",
            }
        ],
    }


def test_communication_polish_rows_v41_covers_multiple_sources():
    rows, counts = _communication_polish_rows_v41(seed=41, limit=40)

    assert rows
    assert any(row.source.endswith("clarity_rewrite") for row in rows)
    assert any(row.source.endswith("grounded_uncertainty") for row in rows)
    assert any(row.source.endswith("coding_explain") for row in rows)
    assert any(row.source.endswith("creative_grounded") for row in rows)
    assert counts


def test_teacher_disagreement_rows_v41_uses_summary_teachers_and_repair_sources():
    rows, summary = _teacher_disagreement_rows_v41(summary=_sample_summary(), seed=41, limit=24)

    assert rows
    assert summary["teacher_pair"][0] == "v40_benchmax"
    assert any("Two strong local teachers disagree" in row.prompt for row in rows)
    assert any(row.source.endswith("route_then_answer") for row in rows)
    assert any("v40_benchmax" in row.prompt for row in rows)


def test_new_v41_training_slices_cover_plans_repairs_and_budgets():
    latent_rows, latent_summary = _latent_plan_rows_v41(seed=41, limit=16)
    repair_rows, repair_summary = _code_critique_repair_rows_v41(seed=41, limit=16)
    budget_rows, budget_summary = _reasoning_budget_rows_v41(seed=41, limit=24)

    assert latent_rows
    assert any("Hidden plan target" in row.prompt for row in latent_rows)
    assert latent_summary

    assert repair_rows
    assert any("Access is denied" in row.prompt or "No space left on device" in row.prompt for row in repair_rows)
    assert repair_summary

    assert budget_rows
    assert any("Reasoning budget: short" in row.prompt for row in budget_rows)
    assert any("Reasoning budget: deep" in row.prompt for row in budget_rows)
    assert budget_summary


def test_build_v41_prep_pack_writes_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_path = root / "v8_summary.json"
        summary_path.write_text(json.dumps(_sample_summary()), encoding="utf-8")

        payload = build_v41_prep_pack(
            summary_path=summary_path,
            output_root=root / "v41_prep",
            seed=41,
            communication_limit=16,
            disagreement_limit=12,
        )

        assert payload["communication_polish"]["row_count"] > 0
        assert payload["latent_plan"]["row_count"] > 0
        assert payload["code_critique_repair"]["row_count"] > 0
        assert payload["reasoning_budget"]["row_count"] > 0
        assert payload["teacher_disagreement"]["row_count"] > 0
        assert payload["promotion_eval_pack"]["item_count"] > 0
        assert Path(payload["blueprint_path"]).exists()
        assert Path(payload["communication_polish"]["path"]).exists()
        assert Path(payload["latent_plan"]["path"]).exists()
        assert Path(payload["code_critique_repair"]["path"]).exists()
        assert Path(payload["reasoning_budget"]["path"]).exists()
        assert Path(payload["teacher_disagreement"]["path"]).exists()
        assert Path(payload["promotion_eval_pack"]["path"]).exists()


def test_assemble_v41_training_rows_merges_base_and_v41_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_path = root / "v8_summary.json"
        summary_path.write_text(json.dumps(_sample_summary()), encoding="utf-8")
        prep_payload = build_v41_prep_pack(
            summary_path=summary_path,
            output_root=root / "v41_prep",
            seed=41,
            communication_limit=16,
            disagreement_limit=12,
        )

        base_stage1 = [_row("Base stage1 prompt", "Base stage1 answer", source="base_stage1", intent="general", domain="general")]
        base_full = list(base_stage1) + [
            _row("Base vision prompt", "Base vision answer", source="base_stage2", intent="vision", domain="vision")
        ]
        stage1_rows, full_rows, merged = assemble_v41_training_rows(
            base_stage1_rows=base_stage1,
            base_full_rows=base_full,
            base_summary={"source_counts": {"base_stage1": 1, "base_stage2": 1}},
            prep_root=Path(prep_payload["output_root"]),
        )

        assert len(full_rows) > len(base_full)
        assert len(stage1_rows) >= len(base_stage1)
        assert merged["v41_added_rows"] > 0
        assert "communication_polish" in merged["v41_row_groups"]
        assert merged["source_counts"]["base_stage1"] == 1


def test_build_training_rows_v41_uses_base_builder_and_writes_dry_run_summary(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_path = root / "v8_summary.json"
        summary_path.write_text(json.dumps(_sample_summary()), encoding="utf-8")

        from source import train_omni_collective_v41 as module

        def _fake_base_builder(**kwargs):
            base_stage1 = [_row("Base prompt", "Base answer", source="base_stage1", intent="general", domain="general")]
            base_full = list(base_stage1)
            return base_stage1, base_full, {"source_counts": {"base_stage1": 1}, "teacher_league": {"teacher_keys": []}}

        monkeypatch.setattr(module, "build_training_rows_v8", _fake_base_builder)

        stage1_rows, full_rows, merged = build_training_rows_v41(
            repo_root=root,
            models_dir=root,
            images_dir=root,
            summary_path=summary_path,
            output_root=root / "v41_out",
            seed=41,
            communication_limit=12,
            disagreement_limit=12,
        )

        assert len(stage1_rows) > 1
        assert len(full_rows) > 1
        assert merged["prep_payload"]["family"] == "omni_collective_v41"
        assert merged["v41_row_groups"]["teacher_disagreement"]["row_count"] > 0


def test_build_training_rows_v41_dry_run_uses_frozen_counts_and_writes_summary():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary_path = root / "v8_summary.json"
        summary = _sample_summary()
        summary["dataset_summary"]["source_counts"] = {"base_a": 10, "base_b": 5}
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        payload = build_training_rows_v41_dry_run(
            summary_path=summary_path,
            output_root=root / "v41_out",
            seed=41,
            communication_limit=12,
            disagreement_limit=12,
        )

        assert payload["dry_run"] is True
        assert payload["base_mode"] == "frozen_counts_plus_v41_rows"
        assert payload["estimated_stage1_rows"] > summary["dataset_summary"]["stage1_rows"]
        assert payload["estimated_stage2_rows"] > summary["dataset_summary"]["stage2_rows"]
        assert Path(payload["summary_path"]).exists()


def test_smoke_training_rows_v41_keeps_v41_mix_and_multimodal_rows():
    stage1_rows = [
        _row("Explain regression tests.", "They catch unintended changes.", source="communication_polish_v41::clarity_rewrite", intent="general", domain="general"),
        _row("Fix this bug.", "Split the logic and add a regression test.", source="teacher_disagreement_v41::repair", intent="coding", domain="coding"),
        _row("Choose a model.", "Use v40_benchmax for benchmark reasoning.", source="base_stage1", intent="model_selection", domain="model_selection"),
        _row("Plan the work.", "List the steps in order.", source="base_stage1", intent="planning", domain="planning"),
    ]
    full_rows = list(stage1_rows) + [
        _row("Analyze this image.", "Looks like lab equipment.", source="base_stage2", intent="vision", domain="vision"),
        _row("Make a 3D part.", "Use OpenSCAD for a cylinder.", source="base_stage2", intent="general", domain="spatial_3d"),
    ]

    smoke_stage1, smoke_full, summary = _smoke_training_rows_v41(
        stage1_rows=stage1_rows,
        full_rows=full_rows,
        seed=41,
    )

    assert smoke_stage1
    assert smoke_full
    assert any("_v41" in row.source for row in smoke_full)
    assert any(row.domain in {"vision", "spatial_3d"} for row in smoke_full)
    assert summary["v41_priority_rows"] >= 1


def test_stage_resume_dir_v41_separates_smoke_and_frontier(tmp_path):
    smoke_dir = _stage_resume_dir_v41(
        tmp_path,
        seed=41,
        distill_limit=0,
        teacher_model_limit=0,
        smoke_train=True,
    )
    frontier_dir = _stage_resume_dir_v41(
        tmp_path,
        seed=41,
        distill_limit=0,
        teacher_model_limit=0,
        smoke_train=False,
    )

    assert smoke_dir != frontier_dir
    assert "smoke_" in smoke_dir.name
    assert "frontier_" in frontier_dir.name
