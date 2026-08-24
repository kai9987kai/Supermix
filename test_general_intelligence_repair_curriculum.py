from __future__ import annotations

import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

from source.build_general_intelligence_curriculum import (  # noqa: E402
    build_curriculum,
    write_curriculum,
)
from source.build_general_intelligence_repair_curriculum import (  # noqa: E402
    FOCUS_FAMILIES,
    PRIORITY_REPLAY_FAMILIES,
    REQUIRED_FAMILIES,
    REPAIR_FILENAME_PREFIX,
    RepairCurriculumBundle,
    build_repair_curriculum,
    write_repair_curriculum,
)
from source.run_qwen_general_promotion_gate import _curriculum_evidence  # noqa: E402
from source.run_research_baseline import _curriculum_provenance  # noqa: E402
from source.verifiable_reasoning import verify_candidate  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _jsonl_bytes(rows: list[dict[str, object]]) -> bytes:
    return b"\n".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        for row in rows
    ) + b"\n"


def _write_parent(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent_dir = tmp_path / "parent"
    paths = write_curriculum(
        build_curriculum(seed=7021, train_rows=150, eval_rows=75),
        parent_dir,
    )
    return (
        Path(paths["train_jsonl"]),
        Path(paths["manifest_json"]),
        Path(paths["eval_jsonl"]),
    )


def _metadata(row: dict[str, object]) -> dict[str, object]:
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def test_repair_curriculum_is_deterministic_focused_and_verified(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    first = build_repair_curriculum(
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        manifest_path=manifest_path,
        seed=912,
        target_rows=120,
        focus_fraction="0.55",
    )
    second = build_repair_curriculum(
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        manifest_path=manifest_path,
        seed=912,
        target_rows=120,
        focus_fraction="0.55",
    )
    changed_seed = build_repair_curriculum(
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        manifest_path=manifest_path,
        seed=913,
        target_rows=120,
        focus_fraction="0.55",
    )

    assert first == second
    assert first != changed_seed
    assert len(first.train_rows) == 120
    family_counts = Counter(
        str(_metadata(row)["problem_family"]) for row in first.train_rows
    )
    assert set(family_counts) == set(REQUIRED_FAMILIES)
    assert sum(family_counts[family] for family in FOCUS_FAMILIES) == 66
    assert family_counts["ratios_probability"] > family_counts["calibrated_prediction"]
    ordinary_replay = set(REQUIRED_FAMILIES) - set(FOCUS_FAMILIES) - set(
        PRIORITY_REPLAY_FAMILIES
    )
    assert min(family_counts[family] for family in PRIORITY_REPLAY_FAMILIES) > max(
        family_counts[family] for family in ordinary_replay
    )

    prompts: set[str] = set()
    example_ids: set[str] = set()
    for row in first.train_rows:
        metadata = _metadata(row)
        assert metadata["curriculum_split"] == "train"
        prompt = " ".join(str(row["user"]).split()).casefold()
        example_id = str(metadata["example_id"]).casefold()
        assert prompt not in prompts
        assert example_id not in example_ids
        prompts.add(prompt)
        example_ids.add(example_id)
        result = verify_candidate(row["user"], row["assistant"], metadata)
        assert result.valid_spec
        assert result.passed

    parent_eval = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_template_ids = {
        str(_metadata(row)["template_id"]) for row in parent_eval
    }
    repair_template_ids = {
        str(_metadata(row)["template_id"]) for row in first.train_rows
    }
    assert repair_template_ids.isdisjoint(eval_template_ids)
    generated = [
        row
        for row in first.train_rows
        if "repair_variant_index" in _metadata(row)
    ]
    assert generated
    assert all(
        str(_metadata(row)["template_id"]).startswith("repair.train.")
        for row in generated
    )
    ratio_variants = [
        row
        for row in generated
        if _metadata(row)["problem_family"] == "ratios_probability"
    ]
    variant_kinds = {
        str(_metadata(row).get("repair_variant_kind") or "") for row in ratio_variants
    }
    assert variant_kinds == {"leading_zero_minimal_pair", "compact_fraction"}
    leading_zero_targets = [
        str(_metadata(row)["expected_answer"])
        for row in ratio_variants
        if _metadata(row).get("repair_variant_kind") == "leading_zero_minimal_pair"
    ]
    assert leading_zero_targets
    assert all(target.startswith("0.0") for target in leading_zero_targets)
    leading_zero_rows = [
        row
        for row in ratio_variants
        if _metadata(row).get("repair_variant_kind") == "leading_zero_minimal_pair"
    ]
    for row in leading_zero_rows:
        metadata = _metadata(row)
        expected = str(metadata["expected_answer"])
        assert metadata["verifier_type"] == "normalized_exact"
        assert row["assistant"] == expected
        assert len(expected) == 5 and expected[0] == "0" and expected[1] == "."
        assert not verify_candidate(row["user"], expected[1:], metadata).passed
        assert not verify_candidate(row["user"], f"{expected}0", metadata).passed
        numeric_metadata = dict(metadata)
        numeric_metadata["verifier_type"] = "decimal"
        numeric_result = verify_candidate(row["user"], row["assistant"], numeric_metadata)
        assert numeric_result.valid_spec
        assert numeric_result.passed
    ratio_rows = [
        row
        for row in first.train_rows
        if _metadata(row)["problem_family"] == "ratios_probability"
    ]
    assert all(
        str(row["assistant"]).startswith("Answer: ")
        for row in ratio_rows
        if _metadata(row).get("repair_variant_kind") != "leading_zero_minimal_pair"
    )


def test_repair_curriculum_writes_content_addressed_gate_evidence(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    bundle = build_repair_curriculum(
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        manifest_path=manifest_path,
    )
    output_dir = tmp_path / "repair"
    paths = write_repair_curriculum(bundle, output_dir)

    assert len(bundle.train_rows) == 480
    repair = bundle.manifest["repair"]
    assert isinstance(repair, dict)
    assert repair["actual_focus_rows"] == 264
    assert repair["actual_focus_fraction"] == pytest.approx(0.55)
    assert set(repair["family_targets"]) == set(REQUIRED_FAMILIES)
    assert repair["family_targets"]["ratios_probability"] == 198
    assert repair["family_targets"]["calibrated_prediction"] == 66
    assert repair["generated_variant_counts"]["leading_zero_minimal_pair"] > 0
    assert repair["generated_variant_counts"]["compact_fraction"] > 0

    content_id = str(bundle.manifest["content_id"])
    content_prefix = content_id.removeprefix("sha256:")[:16]
    expected_stem = f"{REPAIR_FILENAME_PREFIX}_{content_prefix}"
    assert Path(paths["train_jsonl"]).name == f"{expected_stem}.train.jsonl"
    assert Path(paths["eval_jsonl"]).name == f"{expected_stem}.eval.jsonl"
    assert Path(paths["manifest_json"]).name == f"{expected_stem}.manifest.json"
    assert Path(paths["eval_jsonl"]).read_bytes() == eval_path.read_bytes()

    written_manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))
    eval_summary = written_manifest["eval"]
    assert eval_summary["file"] == Path(paths["eval_jsonl"]).name
    assert eval_summary["sha256"] == _sha256(eval_path.read_bytes())
    assert eval_summary["rows"] == 75
    assert eval_summary["byte_identical_to_parent"] is True
    assert set(eval_summary["family_counts"]) == set(REQUIRED_FAMILIES)
    parent = written_manifest["parent"]
    assert parent["manifest_sha256"] == _sha256(manifest_path.read_bytes())
    assert parent["train_sha256"] == _sha256(train_path.read_bytes())
    assert parent["eval_sha256"] == _sha256(eval_path.read_bytes())
    assert written_manifest["all_targets_verified"] is True

    gate_evidence = _curriculum_evidence(Path(paths["manifest_json"]))
    assert gate_evidence["curriculum_eval"] == str(Path(paths["eval_jsonl"]).resolve())
    assert gate_evidence["curriculum_eval_sha256"] == _sha256(eval_path.read_bytes())
    evaluator_evidence = _curriculum_provenance(
        Path(paths["manifest_json"]),
        eval_source=Path(paths["eval_jsonl"]),
    )
    assert evaluator_evidence["curriculum_eval_sha256"] == _sha256(
        eval_path.read_bytes()
    )

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_repair_curriculum(bundle, output_dir)


def test_repair_curriculum_rejects_parent_train_hash_mismatch(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    train_path.write_bytes(train_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="train hash does not match"):
        build_repair_curriculum(
            train_jsonl_path=train_path,
            eval_jsonl_path=eval_path,
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("duplicate", "duplicate prompts"),
        ("wrong_split", "curriculum_split must remain train"),
    ),
)
def test_repair_curriculum_rejects_invalid_parent_rows(
    tmp_path: Path,
    corruption: str,
    message: str,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if corruption == "duplicate":
        rows.append(rows[0])
    else:
        _metadata(rows[0])["curriculum_split"] = "eval"
    train_path.write_bytes(_jsonl_bytes(rows))

    with pytest.raises(ValueError, match=message):
        build_repair_curriculum(
            train_jsonl_path=train_path,
            eval_jsonl_path=eval_path,
            manifest_path=manifest_path,
        )


def test_repair_curriculum_rejects_manifest_eval_template_overlap(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    train_rows = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _metadata(eval_rows[0])["template_id"] = _metadata(train_rows[0])["template_id"]
    eval_payload = _jsonl_bytes(eval_rows)
    eval_path.write_bytes(eval_payload)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["eval"]["sha256"] = _sha256(eval_payload)
    manifest["eval"]["template_ids"] = sorted(
        {str(_metadata(row)["template_id"]) for row in eval_rows}
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="train template IDs overlap"):
        build_repair_curriculum(
            train_jsonl_path=train_path,
            eval_jsonl_path=eval_path,
            manifest_path=manifest_path,
        )


def test_repair_curriculum_rejects_eval_not_declared_by_manifest(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    copied_eval = tmp_path / "different-eval.jsonl"
    copied_eval.write_bytes(eval_path.read_bytes())

    with pytest.raises(ValueError, match="eval JSONL is not the artifact declared"):
        build_repair_curriculum(
            train_jsonl_path=train_path,
            eval_jsonl_path=copied_eval,
            manifest_path=manifest_path,
        )


def test_repair_curriculum_requires_eval_path_at_python_api(tmp_path: Path) -> None:
    train_path, manifest_path, _eval_path = _write_parent(tmp_path)

    with pytest.raises(TypeError, match="eval_jsonl_path"):
        build_repair_curriculum(  # type: ignore[call-arg]
            train_jsonl_path=train_path,
            manifest_path=manifest_path,
        )


def test_repair_curriculum_rejects_self_consistent_eval_missing_family(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    eval_rows = [
        json.loads(line)
        for line in eval_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_rows = [
        row
        for row in eval_rows
        if _metadata(row)["problem_family"] != "typo_noise_robustness"
    ]
    eval_payload = _jsonl_bytes(eval_rows)
    eval_path.write_bytes(eval_payload)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eval_summary = manifest["eval"]
    eval_summary["sha256"] = _sha256(eval_payload)
    eval_summary["rows"] = len(eval_rows)
    eval_summary["family_counts"] = dict(
        sorted(Counter(str(_metadata(row)["problem_family"]) for row in eval_rows).items())
    )
    eval_summary["source_counts"] = dict(
        sorted(Counter(str(row["source"]) for row in eval_rows).items())
    )
    eval_summary["verifier_type_counts"] = dict(
        sorted(Counter(str(_metadata(row)["verifier_type"]) for row in eval_rows).items())
    )
    eval_summary["template_ids"] = sorted(
        {str(_metadata(row)["template_id"]) for row in eval_rows}
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="eval family set"):
        build_repair_curriculum(
            train_jsonl_path=train_path,
            eval_jsonl_path=eval_path,
            manifest_path=manifest_path,
        )


def test_repair_writer_rejects_noncanonical_or_broken_identity(
    tmp_path: Path,
) -> None:
    train_path, manifest_path, eval_path = _write_parent(tmp_path)
    bundle = build_repair_curriculum(
        train_jsonl_path=train_path,
        eval_jsonl_path=eval_path,
        manifest_path=manifest_path,
        target_rows=120,
    )

    corruptions: list[tuple[str, dict[str, object], str]] = []
    bad_content_id = copy.deepcopy(bundle.manifest)
    bad_content_id["content_id"] = f"sha256:{'0' * 64}"
    corruptions.append(("content", bad_content_id, "content_id is not canonical"))

    bad_filename = copy.deepcopy(bundle.manifest)
    bad_filename["train"]["file"] = "not-content-addressed.train.jsonl"  # type: ignore[index]
    corruptions.append(("filename", bad_filename, "filenames are not canonical"))

    bad_eval_chain = copy.deepcopy(bundle.manifest)
    bad_eval_chain["eval"]["parent_sha256"] = "0" * 64  # type: ignore[index]
    corruptions.append(("eval", bad_eval_chain, "eval hash chain"))

    for name, manifest, message in corruptions:
        tampered = RepairCurriculumBundle(
            train_rows=bundle.train_rows,
            eval_bytes=bundle.eval_bytes,
            manifest=manifest,
        )
        with pytest.raises(ValueError, match=message):
            write_repair_curriculum(tampered, tmp_path / name)
