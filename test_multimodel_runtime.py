import sys
from pathlib import Path

from source.multimodel_catalog import ModelRecord

SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.multimodel_runtime import UnifiedModelManager


def _record(key: str, kind: str, capabilities: tuple[str, ...], score: float | None = None) -> ModelRecord:
    return ModelRecord(
        key=key,
        label=key,
        family="test",
        kind=kind,
        capabilities=capabilities,
        zip_path=Path(f"{key}.zip"),
        common_row_key=key,
        common_overall_exact=score,
    )


def test_collective_panel_includes_omni_collective_v2_v3_v4_v5_v6_v7_v8_v8_preview_v40_and_domain_specialists(tmp_path: Path) -> None:
    records = (
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("protein_folding_micro_v1", "protein_folding", ("chat",), None),
        _record("mattergen_micro_v1", "mattergen_generation", ("chat",), None),
        _record("three_d_generation_micro_v1", "three_d_generation", ("chat",), None),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v8_preview", "omni_collective_v8", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("science_vision_micro_v1", "image_recognition", ("chat", "vision"), None),
        _record("v38_native_xlite_fp16", "native_image", ("image",), 0.01),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    consultants = manager._collective_consultants()
    keys = [record.key for record in consultants]
    assert "protein_folding_micro_v1" in keys
    assert "mattergen_micro_v1" in keys
    assert "three_d_generation_micro_v1" in keys
    assert "omni_collective_v2" in keys
    assert "omni_collective_v3" in keys
    assert "omni_collective_v4" in keys
    assert "omni_collective_v5" in keys
    assert "omni_collective_v6" in keys
    assert "omni_collective_v7" in keys
    assert "omni_collective_v8" in keys
    assert "omni_collective_v8_preview" in keys
    assert "v40_benchmax" in keys
    assert "v38_native_xlite_fp16" not in keys


def test_default_text_record_prefers_v40_benchmax(tmp_path: Path) -> None:
    records = (
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    chosen = manager._default_text_record()
    assert chosen.key == "v40_benchmax"


def test_model_store_catalog_marks_installed_and_selectable_records(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    installed = models_dir / "dcgan_v2_in_progress.zip"
    installed.write_bytes(b"zip")
    records = (
        _record("dcgan_v2_in_progress", "dcgan_image", ("image",), None),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
        models_dir=models_dir,
        common_summary_path=tmp_path / "missing_summary.json",
    )

    manager._fetch_model_store_manifest_locked = lambda force_refresh=False: {  # type: ignore[method-assign]
        "models": [
            {"file_name": "dcgan_v2_in_progress.zip", "size_bytes": 3, "size_mb": 0.0, "family": "gan"},
            {"file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip", "size_bytes": 10, "size_mb": 0.0, "family": "fusion"},
        ]
    }

    payload = manager.model_store_catalog(force_refresh=True)
    by_name = {row["file_name"]: row for row in payload["models"]}
    assert by_name["dcgan_v2_in_progress.zip"]["installed"] is True
    assert by_name["dcgan_v2_in_progress.zip"]["selectable"] is True
    assert by_name["supermix_omni_collective_v8_preview_20260407_001155.zip"]["known"] is True
    assert by_name["supermix_omni_collective_v8_preview_20260407_001155.zip"]["installed"] is False
