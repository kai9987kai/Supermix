from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import qwen_chat_desktop_app as desktop_app  # noqa: E402
import qwen_chat_web_app as web_app  # noqa: E402
import qwen_adapter_promotion as promotion  # noqa: E402
from qwen_adapter_promotion import (  # noqa: E402
    GATE_FILENAME,
    PROMOTION_FILENAME,
    PROMOTION_SCHEMA_VERSION,
)
from test_qwen_adapter_promotion import (  # noqa: E402
    _make_bound_gate_fixture,
    _run_bound_gate,
)


SELECTORS = (web_app, desktop_app)


def _write_adapter(adapter: Path, *, weights: bytes = b"adapter-weights") -> Path:
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text('{"r":8}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    return adapter.resolve()


def _allow_legacy(
    monkeypatch: pytest.MonkeyPatch,
    adapter: Path,
    artifact_name: str,
) -> None:
    allowlist = dict(promotion.LEGACY_ADAPTER_ALLOWLIST)
    allowlist[promotion._adapter_namespace_token(artifact_name)] = {
        "adapter_config_sha256": promotion.sha256_file(adapter / "adapter_config.json"),
        "adapter_sha256": promotion.sha256_file(adapter / "adapter_model.safetensors"),
    }
    monkeypatch.setattr(promotion, "LEGACY_ADAPTER_ALLOWLIST", allowlist)


@pytest.mark.parametrize("selector", SELECTORS, ids=("web", "desktop"))
def test_candidate_and_failed_receipt_never_enter_legacy_fallback(
    tmp_path,
    selector,
    monkeypatch,
) -> None:
    legacy = _write_adapter(
        tmp_path / "artifacts" / "qwen_supermix_auto_verify_20260320" / "adapter"
    )
    _allow_legacy(monkeypatch, legacy, "qwen_supermix_auto_verify_20260320")
    unpromoted = _write_adapter(
        tmp_path / "artifacts" / "general_intelligence_v1" / "candidate_one" / "adapter"
    )
    revoked = _write_adapter(
        tmp_path / "artifacts" / "candidate_mixed_15" / "adapter",
        weights=b"revoked",
    )
    (revoked.parent / PROMOTION_FILENAME).write_text(
        json.dumps({"schema": PROMOTION_SCHEMA_VERSION, "passed": False}),
        encoding="utf-8",
    )
    for artifact in (unpromoted.parent, revoked.parent):
        (artifact / "benchmark_results.json").write_text("{}", encoding="utf-8")

    assert selector.adapter_activation_kind(unpromoted) is None
    assert selector.adapter_activation_kind(revoked) is None
    assert selector.find_latest_adapter_dir(tmp_path) == legacy


@pytest.mark.parametrize("selector", SELECTORS, ids=("web", "desktop"))
def test_candidate_only_auto_selection_fails_closed_but_explicit_path_still_works(
    tmp_path,
    selector,
) -> None:
    candidate = _write_adapter(
        tmp_path
        / "artifacts"
        / "training_candidates"
        / "qwen_general_intelligence_v1"
        / "adapter"
    )

    with pytest.raises(FileNotFoundError, match="validated promoted"):
        selector.find_latest_adapter_dir(tmp_path)
    assert selector.resolve_adapter_dir(tmp_path, str(candidate)) == candidate


@pytest.mark.parametrize("selector", SELECTORS, ids=("web", "desktop"))
def test_valid_promoted_general_intelligence_candidate_wins_auto_selection(
    tmp_path,
    selector,
) -> None:
    _write_adapter(tmp_path / "artifacts" / "qwen_supermix_enhanced_v25" / "adapter")
    benchmark_path, promoted, curriculum_manifest = _make_bound_gate_fixture(
        tmp_path / "artifacts" / "general_intelligence_v1"
    )
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=promoted,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "promotion-pointer.txt",
    )
    assert result["passed"] is True

    assert selector.adapter_activation_kind(promoted) == "promoted"
    assert selector.find_latest_adapter_dir(tmp_path) == promoted


def _write_bundle(
    root: Path,
    *,
    artifact_name: str,
    declared_kind: str,
    weights: bytes = b"adapter-weights",
) -> Path:
    artifact = root / "bundled_latest_artifact"
    adapter = _write_adapter(artifact / "adapter", weights=weights)
    (artifact / "release_manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": artifact_name,
                "adapter_relative_path": "adapter",
                "adapter_activation": declared_kind,
                "promotion_manifest_relative_path": (
                    PROMOTION_FILENAME if declared_kind == "promoted" else None
                ),
                "promotion_gate_relative_path": (
                    GATE_FILENAME if declared_kind == "promoted" else None
                ),
            }
        ),
        encoding="utf-8",
    )
    return adapter


def test_bundled_promoted_adapter_requires_valid_staged_receipts(tmp_path, monkeypatch) -> None:
    benchmark_path, receipt_adapter, curriculum_manifest = _make_bound_gate_fixture(
        tmp_path / "receipt_source"
    )
    result = _run_bound_gate(
        benchmark_path=benchmark_path,
        adapter=receipt_adapter,
        curriculum_manifest=curriculum_manifest,
        pointer=tmp_path / "receipt-pointer.txt",
    )
    assert result["passed"] is True
    adapter = _write_bundle(
        tmp_path,
        artifact_name="general_intelligence_candidate_v1",
        declared_kind="promoted",
        weights=b"weights-v1",
    )
    shutil.copy2(receipt_adapter / "adapter_config.json", adapter / "adapter_config.json")
    for filename in (PROMOTION_FILENAME, GATE_FILENAME):
        shutil.copy2(receipt_adapter.parent / filename, adapter.parent / filename)
    monkeypatch.setattr(desktop_app.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert desktop_app.find_bundled_adapter_dir() == adapter

    (adapter.parent / GATE_FILENAME).write_text('{"passed":false}', encoding="utf-8")
    assert desktop_app.find_bundled_adapter_dir() is None


def test_bundled_unpromoted_candidate_is_rejected_but_named_legacy_is_preserved(
    tmp_path,
    monkeypatch,
) -> None:
    candidate_root = tmp_path / "candidate_bundle"
    _write_bundle(
        candidate_root,
        artifact_name="general_intelligence_candidate_v1",
        declared_kind="promoted",
    )
    monkeypatch.setattr(desktop_app.sys, "_MEIPASS", str(candidate_root), raising=False)
    assert desktop_app.find_bundled_adapter_dir() is None

    legacy_root = tmp_path / "legacy_bundle"
    legacy = _write_bundle(
        legacy_root,
        artifact_name="qwen_supermix_enhanced_v25",
        declared_kind="legacy",
    )
    _allow_legacy(monkeypatch, legacy, "qwen_supermix_enhanced_v25")
    monkeypatch.setattr(desktop_app.sys, "_MEIPASS", str(legacy_root), raising=False)
    assert desktop_app.find_bundled_adapter_dir() == legacy


@pytest.mark.parametrize("selector", SELECTORS, ids=("web", "desktop"))
def test_legacy_prefix_without_allowlisted_content_is_rejected(tmp_path, selector) -> None:
    spoofed = _write_adapter(
        tmp_path / "artifacts" / "qwen_supermix_enhanced_v999" / "adapter",
        weights=b"arbitrary-unevaluated-adapter",
    )

    assert selector.adapter_activation_kind(spoofed) is None
    with pytest.raises(FileNotFoundError, match="validated promoted"):
        selector.find_latest_adapter_dir(tmp_path)


def test_runtime_attestation_uses_one_promotion_validation_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    adapter = _write_adapter(tmp_path / "artifacts" / "candidate_race" / "adapter")
    calls = 0

    def changing_validation(_adapter: Path) -> dict[str, object] | None:
        nonlocal calls
        calls += 1
        return None if calls == 1 else {"schema": PROMOTION_SCHEMA_VERSION, "passed": True}

    monkeypatch.setattr(promotion, "validate_promoted_adapter", changing_validation)

    with pytest.raises(ValueError, match="not eligible"):
        promotion.attest_adapter_for_runtime(adapter)
    assert calls == 1
