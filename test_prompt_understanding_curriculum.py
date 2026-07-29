from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from source.build_prompt_understanding_curriculum import (
    BENCHMARK_FILENAME,
    BENCHMARK_SCHEMA_VERSION,
    CURRICULUM_SCHEMA_VERSION,
    CURRICULUM_SOURCE,
    EVAL_FILENAME,
    EVAL_TEMPLATE_IDS,
    MANIFEST_FILENAME,
    PROMPT_FAMILIES,
    PROMPT_SPEC_FIELDS,
    PROMPT_SPEC_SCHEMA_VERSION,
    PROMPT_VERIFIER_SCHEMA_VERSION,
    TRAIN_FILENAME,
    TRAIN_TEMPLATE_IDS,
    build_behavioral_benchmark_report,
    build_curriculum,
    expected_prompt_spec,
    expected_user_prompt,
    main,
    verify_prompt_spec,
    write_curriculum,
)
from source.verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate


def _metadata(row: dict[str, object]) -> dict[str, object]:
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    return metadata


def _assistant_spec(row: dict[str, object]) -> dict[str, object]:
    spec = json.loads(str(row["assistant"]))
    assert isinstance(spec, dict)
    return spec


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def test_curriculum_is_deterministic_and_seed_sensitive() -> None:
    first = build_curriculum(seed=73, train_rows=50, eval_rows=25)
    second = build_curriculum(seed=73, train_rows=50, eval_rows=25)
    changed = build_curriculum(seed=74, train_rows=50, eval_rows=25)

    assert first == second
    assert first.train_rows != changed.train_rows
    assert first.eval_rows != changed.eval_rows
    assert first.manifest["curriculum_schema"] == CURRICULUM_SCHEMA_VERSION
    assert first.manifest["prompt_spec_schema"] == PROMPT_SPEC_SCHEMA_VERSION
    assert first.manifest["shared_verifier_schema"] == VERIFIER_SCHEMA_VERSION


def test_five_families_are_balanced_and_both_decisions_are_covered() -> None:
    bundle = build_curriculum(seed=21, train_rows=55, eval_rows=35)
    for rows in (bundle.train_rows, bundle.eval_rows):
        counts = Counter(str(_metadata(row)["problem_family"]) for row in rows)
        decisions = Counter(str(_assistant_spec(row)["decision"]) for row in rows)
        assert set(counts) == set(PROMPT_FAMILIES)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert set(decisions) == {"act", "ask"}


def test_train_eval_templates_and_prompt_text_are_disjoint() -> None:
    bundle = build_curriculum(seed=11, train_rows=80, eval_rows=40)
    train_templates = {_metadata(row)["template_id"] for row in bundle.train_rows}
    eval_templates = {_metadata(row)["template_id"] for row in bundle.eval_rows}
    train_prompts = {str(row["user"]).strip().casefold() for row in bundle.train_rows}
    eval_prompts = {str(row["user"]).strip().casefold() for row in bundle.eval_rows}

    assert TRAIN_TEMPLATE_IDS.isdisjoint(EVAL_TEMPLATE_IDS)
    assert train_templates.isdisjoint(eval_templates)
    assert train_prompts.isdisjoint(eval_prompts)
    assert bundle.manifest["template_ids_disjoint"] is True
    assert bundle.manifest["prompt_text_disjoint"] is True


def test_every_target_is_canonical_json_and_passes_both_verifiers() -> None:
    bundle = build_curriculum(seed=103, train_rows=50, eval_rows=25)
    for split, rows in (("train", bundle.train_rows), ("eval", bundle.eval_rows)):
        for row in rows:
            metadata = _metadata(row)
            spec = _assistant_spec(row)

            assert set(row) == {"user", "assistant", "source", "metadata"}
            assert row["source"] == CURRICULUM_SOURCE
            assert metadata["curriculum_split"] == split
            assert metadata["prompt_verifier_schema"] == PROMPT_VERIFIER_SCHEMA_VERSION
            assert metadata["prompt_spec_schema"] == PROMPT_SPEC_SCHEMA_VERSION
            assert set(spec) == set(PROMPT_SPEC_FIELDS)
            assert row["assistant"] == _compact_json(spec)
            assert spec == expected_prompt_spec(metadata)
            assert row["user"] == expected_user_prompt(metadata)
            assert all(isinstance(value, (str, int, float, bool)) for value in metadata.values())

            exact = verify_prompt_spec(row["assistant"], metadata)
            shared = verify_candidate(row["user"], row["assistant"], metadata)
            assert exact.valid_spec and exact.passed, exact.to_payload()
            assert shared.valid_spec and shared.passed, shared.to_payload()


def test_full_verifier_rejects_every_single_field_mutation() -> None:
    bundle = build_curriculum(seed=7, train_rows=10, eval_rows=10)
    for row in bundle.eval_rows:
        metadata = _metadata(row)
        expected = expected_prompt_spec(metadata)
        for field in PROMPT_SPEC_FIELDS:
            mutated = dict(expected)
            if field == "schema":
                mutated[field] = "unknown-schema"
            elif field == "decision":
                mutated[field] = "ask" if expected[field] == "act" else "act"
            elif field in {"goal", "reference"}:
                mutated[field] = f"{expected[field]} changed"
            elif field == "turn_relation":
                mutated[field] = (
                    "follow_up" if expected[field] == "single_turn" else "single_turn"
                )
            else:
                mapping = expected[field]
                assert isinstance(mapping, dict)
                mutated[field] = {**mapping, "unexpected": "true"}

            result = verify_prompt_spec(_compact_json(mutated), metadata)
            assert result.valid_spec
            assert not result.passed
            if field != "schema":
                assert field in result.mismatched_fields


def test_full_verifier_is_not_fooled_by_a_mutated_shared_self_label() -> None:
    bundle = build_curriculum(seed=3, train_rows=20, eval_rows=10)
    row = next(row for row in bundle.train_rows if _assistant_spec(row)["decision"] == "ask")
    metadata = dict(_metadata(row))
    candidate = _assistant_spec(row)
    candidate["decision"] = "act"
    assistant = _compact_json(candidate)
    metadata["expected_answer"] = _compact_json("act")

    shared = verify_candidate(row["user"], assistant, metadata)
    exact = verify_prompt_spec(assistant, metadata)
    assert shared.passed, "The setup should demonstrate why decision-only labels are insufficient."
    assert exact.valid_spec
    assert not exact.passed
    assert exact.mismatched_fields == ("decision",)


def test_strict_json_rejects_duplicate_keys_unknown_fields_and_nonstandard_constants() -> None:
    row = build_curriculum(seed=5, train_rows=5, eval_rows=5).train_rows[0]
    metadata = _metadata(row)
    spec = _assistant_spec(row)
    decision = str(spec["decision"])
    duplicate = str(row["assistant"]).replace(
        '{"constraints"',
        f'{{"decision":"{decision}","constraints"',
        1,
    )
    with_extra = {**spec, "extra": "not allowed"}
    with_nan = dict(spec)
    with_nan["goal"] = float("nan")

    assert not verify_prompt_spec(duplicate, metadata).passed
    assert not verify_prompt_spec(_compact_json(with_extra), metadata).passed
    assert not verify_prompt_spec(_compact_json(with_nan), metadata).passed
    assert not verify_prompt_spec("```json\n{}\n```", metadata).passed


def test_behavioral_benchmark_has_per_family_per_field_rejection() -> None:
    bundle = build_curriculum(seed=37, train_rows=25, eval_rows=25)
    report = bundle.benchmark_report
    rebuilt = build_behavioral_benchmark_report(bundle.eval_rows)

    assert report == rebuilt
    assert report["benchmark_schema"] == BENCHMARK_SCHEMA_VERSION
    assert report["status"] == "pass"
    assert report["gold_accuracy"] == 1.0
    assert report["mutation_rejection_rate"] == 1.0
    assert report["mutations"] == len(bundle.eval_rows) * len(PROMPT_SPEC_FIELDS)
    assert set(report["families"]) == set(PROMPT_FAMILIES)
    for family, payload in report["families"].items():
        assert family in PROMPT_FAMILIES
        assert payload["gold_passes"] == payload["rows"]
        assert payload["mutations"] == payload["rows"] * len(PROMPT_SPEC_FIELDS)
        assert payload["mutation_rejections"] == payload["mutations"]


def test_manifest_hashes_match_exact_artifacts(tmp_path: Path) -> None:
    bundle = build_curriculum(seed=82, train_rows=30, eval_rows=15)
    paths = write_curriculum(bundle, tmp_path)
    train_path = Path(paths["train_jsonl"])
    eval_path = Path(paths["eval_jsonl"])
    manifest_path = Path(paths["manifest_json"])
    benchmark_path = Path(paths["benchmark_json"])

    assert train_path.name == TRAIN_FILENAME
    assert eval_path.name == EVAL_FILENAME
    assert manifest_path.name == MANIFEST_FILENAME
    assert benchmark_path.name == BENCHMARK_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert manifest == bundle.manifest
    assert benchmark == bundle.benchmark_report
    assert hashlib.sha256(train_path.read_bytes()).hexdigest() == manifest["train"]["sha256"]
    assert hashlib.sha256(eval_path.read_bytes()).hexdigest() == manifest["eval"]["sha256"]
    assert (
        hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
        == manifest["benchmark"]["sha256"]
    )

    manifest_without_hash = dict(manifest)
    expected_hash = manifest_without_hash.pop("manifest_sha256")
    actual_hash = hashlib.sha256(
        _compact_json(manifest_without_hash).encode("utf-8")
    ).hexdigest()
    assert expected_hash == actual_hash


def test_writer_refuses_overwrite_and_cli_emits_all_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "curriculum"
    args = [
        "--output-dir",
        str(output_dir),
        "--train-rows",
        "15",
        "--eval-rows",
        "10",
        "--seed",
        "19",
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["benchmark_status"] == "pass"
    assert payload["gold_accuracy"] == 1.0
    assert payload["mutation_rejection_rate"] == 1.0
    for filename in (TRAIN_FILENAME, EVAL_FILENAME, MANIFEST_FILENAME, BENCHMARK_FILENAME):
        assert (output_dir / filename).exists()

    with pytest.raises(FileExistsError):
        main(args)
    assert main([*args, "--overwrite"]) == 0


def test_existing_qwen_loader_keeps_every_generated_row(tmp_path: Path) -> None:
    bundle = build_curriculum(seed=97, train_rows=15, eval_rows=10)
    paths = write_curriculum(bundle, tmp_path)
    source_dir = Path(__file__).resolve().parent / "source"
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    pipeline = importlib.import_module("qwen_supermix_pipeline")

    pairs = pipeline.load_jsonl_pairs(
        [paths["train_jsonl"], paths["eval_jsonl"]],
        max_records=100,
        min_chars=1,
        max_source_fraction=0.0,
        max_synthetic_fraction=0.0,
        max_prompt_signature_count=0,
        log_every_records=0,
    )
    assert len(pairs) == len(bundle.train_rows) + len(bundle.eval_rows)
    assert all(pair.metadata["prompt_verifier_schema"] == PROMPT_VERIFIER_SCHEMA_VERSION for pair in pairs)
    assert all(pair.metadata["prompt_spec_schema"] == PROMPT_SPEC_SCHEMA_VERSION for pair in pairs)
    assert all(verify_candidate(pair.user, pair.assistant, pair.metadata).passed for pair in pairs)


def test_family_subset_and_invalid_configuration() -> None:
    bundle = build_curriculum(
        seed=7,
        train_rows=8,
        eval_rows=4,
        families=("typo_noise_robustness", "hard_conflict_ask_vs_act"),
    )
    assert {
        _metadata(row)["problem_family"]
        for row in (*bundle.train_rows, *bundle.eval_rows)
    } == {"typo_noise_robustness", "hard_conflict_ask_vs_act"}

    with pytest.raises(ValueError, match="Unknown prompt family"):
        build_curriculum(
            seed=7,
            train_rows=5,
            eval_rows=5,
            families=("untrusted_dynamic_execution",),
        )
    with pytest.raises(ValueError, match="cover every selected"):
        build_curriculum(seed=7, train_rows=4, eval_rows=5)
    with pytest.raises(ValueError, match="seed must be an integer"):
        build_curriculum(seed=True, train_rows=5, eval_rows=5)


def test_invalid_template_metadata_fails_closed() -> None:
    row = build_curriculum(seed=13, train_rows=5, eval_rows=5).train_rows[0]
    metadata = dict(_metadata(row))
    metadata["template_id"] = "eval.unknown.self_labeled.v1"
    result = verify_prompt_spec(row["assistant"], metadata)
    assert not result.valid_spec
    assert not result.passed
    assert result.reason.startswith("invalid_verifier_metadata:")
