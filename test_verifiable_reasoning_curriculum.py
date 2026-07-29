from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from source.build_verifiable_reasoning_curriculum import (
    CURRICULUM_SCHEMA_VERSION,
    CURRICULUM_SOURCE,
    EVAL_FILENAME,
    MANIFEST_FILENAME,
    PROBLEM_FAMILIES,
    TRAIN_FILENAME,
    build_curriculum,
    main,
    write_curriculum,
)
from source.verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate


def _metadata(row: dict[str, object]) -> dict[str, object]:
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def test_curriculum_is_deterministic_for_the_same_seed() -> None:
    first = build_curriculum(seed=73, train_rows=75, eval_rows=35)
    second = build_curriculum(seed=73, train_rows=75, eval_rows=35)
    assert first.train_rows == second.train_rows
    assert first.eval_rows == second.eval_rows
    assert first.manifest == second.manifest


def test_curriculum_changes_with_seed_but_keeps_the_contract() -> None:
    first = build_curriculum(seed=11, train_rows=30, eval_rows=15)
    second = build_curriculum(seed=12, train_rows=30, eval_rows=15)
    assert first.train_rows != second.train_rows
    assert first.eval_rows != second.eval_rows
    assert first.manifest["curriculum_schema"] == CURRICULUM_SCHEMA_VERSION
    assert second.manifest["verifier_schema"] == VERIFIER_SCHEMA_VERSION


def test_train_and_eval_templates_and_prompts_are_disjoint() -> None:
    bundle = build_curriculum(seed=51, train_rows=100, eval_rows=50)
    train_templates = {_metadata(row)["template_id"] for row in bundle.train_rows}
    eval_templates = {_metadata(row)["template_id"] for row in bundle.eval_rows}
    train_prompts = {str(row["user"]).strip().casefold() for row in bundle.train_rows}
    eval_prompts = {str(row["user"]).strip().casefold() for row in bundle.eval_rows}
    assert train_templates.isdisjoint(eval_templates)
    assert train_prompts.isdisjoint(eval_prompts)
    assert bundle.manifest["template_ids_disjoint"] is True
    assert bundle.manifest["prompt_text_disjoint"] is True


def test_curriculum_covers_every_required_family_and_verifier_type() -> None:
    bundle = build_curriculum(seed=9, train_rows=150, eval_rows=75)
    rows = [*bundle.train_rows, *bundle.eval_rows]
    families = {_metadata(row)["problem_family"] for row in rows}
    verifier_types = {_metadata(row)["verifier_type"] for row in rows}
    assert families == set(PROBLEM_FAMILIES)
    assert verifier_types == {
        "integer",
        "decimal",
        "fraction",
        "normalized_exact",
        "multiple_choice",
        "json_field",
    }


def test_every_generated_assistant_passes_its_own_verifier() -> None:
    bundle = build_curriculum(seed=103, train_rows=125, eval_rows=65)
    for split, rows in (("train", bundle.train_rows), ("eval", bundle.eval_rows)):
        for row in rows:
            metadata = _metadata(row)
            assert metadata["curriculum_split"] == split
            result = verify_candidate(row["user"], row["assistant"], metadata)
            assert result.valid_spec, metadata["example_id"]
            assert result.passed, (metadata["example_id"], result.reason)
            assert result.reward == 1.0


def test_rows_are_chatpair_compatible_and_metadata_is_scalar() -> None:
    bundle = build_curriculum(seed=5, train_rows=50, eval_rows=25)
    for row in (*bundle.train_rows, *bundle.eval_rows):
        assert set(row) == {"user", "assistant", "source", "metadata"}
        assert row["source"] == CURRICULUM_SOURCE
        assert isinstance(row["user"], str) and row["user"]
        assert isinstance(row["assistant"], str) and row["assistant"]
        metadata = _metadata(row)
        assert metadata["verifier_schema"] == VERIFIER_SCHEMA_VERSION
        assert metadata["verified_correct"] is True
        assert metadata["rule_reward"] == 1.0
        assert all(isinstance(key, str) for key in metadata)
        assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())
        aliases = json.loads(str(metadata["aliases_json"]))
        assert isinstance(aliases, list)


def test_family_allocation_is_balanced() -> None:
    bundle = build_curriculum(seed=41, train_rows=53, eval_rows=27)
    for rows in (bundle.train_rows, bundle.eval_rows):
        counts = Counter(str(_metadata(row)["problem_family"]) for row in rows)
        assert set(counts) == set(PROBLEM_FAMILIES)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_manifest_hashes_match_exact_written_jsonl(tmp_path: Path) -> None:
    bundle = build_curriculum(seed=82, train_rows=40, eval_rows=20)
    paths = write_curriculum(bundle, tmp_path)
    train_path = Path(paths["train_jsonl"])
    eval_path = Path(paths["eval_jsonl"])
    manifest_path = Path(paths["manifest_json"])
    assert train_path.name == TRAIN_FILENAME
    assert eval_path.name == EVAL_FILENAME
    assert manifest_path.name == MANIFEST_FILENAME

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == bundle.manifest
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["train"]["sha256"]
    assert hashlib.sha256(eval_path.read_bytes()).hexdigest() == manifest["eval"]["sha256"]

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
    assert len(train_rows) == 40
    assert len(eval_rows) == 20
    assert train_rows[0] == bundle.train_rows[0]
    assert eval_rows[0] == bundle.eval_rows[0]


def test_writer_refuses_accidental_overwrite(tmp_path: Path) -> None:
    bundle = build_curriculum(seed=3, train_rows=10, eval_rows=5)
    write_curriculum(bundle, tmp_path)
    with pytest.raises(FileExistsError):
        write_curriculum(bundle, tmp_path)
    write_curriculum(bundle, tmp_path, overwrite=True)


def test_cli_builds_artifacts_and_supports_explicit_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "cli-output"
    args = [
        "--output-dir",
        str(output_dir),
        "--seed",
        "19",
        "--train-rows",
        "25",
        "--eval-rows",
        "10",
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["train_rows"] == 25
    assert payload["eval_rows"] == 10
    assert (output_dir / TRAIN_FILENAME).exists()
    assert (output_dir / EVAL_FILENAME).exists()
    assert (output_dir / MANIFEST_FILENAME).exists()

    with pytest.raises(FileExistsError):
        main(args)
    assert main([*args, "--overwrite"]) == 0


def test_family_subset_and_invalid_configuration() -> None:
    bundle = build_curriculum(
        seed=7,
        train_rows=12,
        eval_rows=6,
        families=("sequences", "evidence_in_prompt_qa"),
    )
    assert {
        _metadata(row)["problem_family"]
        for row in (*bundle.train_rows, *bundle.eval_rows)
    } == {"sequences", "evidence_in_prompt_qa"}

    with pytest.raises(ValueError, match="Unknown problem family"):
        build_curriculum(
            seed=7,
            train_rows=2,
            eval_rows=2,
            families=("unsafe_code_execution",),
        )
    with pytest.raises(ValueError, match="must both be positive"):
        build_curriculum(seed=7, train_rows=0, eval_rows=2)
