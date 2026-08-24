"""Focused tests for the preregistered v72 MiMoMix promotion gate."""

from __future__ import annotations

import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import v72_model_promotion as promotion  # noqa: E402


def _write_corpus(path: Path, prompts=()) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"user": str(prompt), "assistant": "A bounded synthetic answer.", "task": "test"}
        for prompt in prompts
    ]
    if not rows:
        rows = [{"user": "ordinary unrelated prompt", "assistant": "ordinary reply"}]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _freeze_fixture(tmp_path: Path, *, baseline_prompts=(), candidate_prompts=()):
    baseline_checkpoint = tmp_path / "v70.pt"
    candidate_checkpoint = tmp_path / "candidate.pt"
    baseline_corpus = tmp_path / "v70.jsonl"
    candidate_corpus = tmp_path / "candidate.jsonl"
    manifest_path = tmp_path / "frozen.json"
    baseline_checkpoint.write_bytes(b"immutable-v70-checkpoint")
    _write_corpus(baseline_corpus, baseline_prompts)
    _write_corpus(candidate_corpus, candidate_prompts)
    # Production has no comparator override.  Tests replace the constants only
    # inside this Python process so tiny fixtures exercise the same binding.
    promotion.V70_CHECKPOINT_SHA256 = promotion.sha256_file(baseline_checkpoint)
    promotion.V70_CORPUS_SHA256 = promotion.sha256_file(baseline_corpus)
    manifest = promotion.freeze_manifest(
        baseline_checkpoint=baseline_checkpoint,
        baseline_corpus=baseline_corpus,
        candidate_corpus=candidate_corpus,
        candidate_checkpoint=candidate_checkpoint,
        output=manifest_path,
    )
    return {
        "baseline_checkpoint": baseline_checkpoint,
        "candidate_checkpoint": candidate_checkpoint,
        "baseline_corpus": baseline_corpus,
        "candidate_corpus": candidate_corpus,
        "manifest_path": manifest_path,
        "manifest": manifest,
    }


def _answer(record, *, correct: bool) -> str:
    expected = record["expected"]
    value = Fraction(int(expected["numerator"]), int(expected["denominator"]))
    if not correct:
        value += 1
    return f"bounded working, total {value.numerator}/{value.denominator}"


def _passing_runner(checkpoint, math_records, chat_records, protocol):
    candidate = checkpoint.name == "candidate.pt"
    math = []
    per_family_seen = {family: 0 for family in promotion.FAMILIES}
    for record in math_records:
        family = str(record["family"])
        per_family_seen[family] += 1
        # The baseline misses two per family; the candidate repairs all ten.
        correct = candidate or per_family_seen[family] > 2
        math.append(
            {
                "id": record["id"],
                "reply": _answer(record, correct=correct),
                "tokens": 12,
                "prompt_unknown_rate": 0.0,
                "error": "",
            }
        )
    chat = [
        {
            "id": record["id"],
            "reply": "Check the traceback and isolate the failing function first.",
            "tokens": 10,
            "prompt_unknown_rate": 0.0,
            "error": "",
        }
        for record in chat_records
    ]
    return {
        "checkpoint_metadata": {"synthetic": True, "candidate": candidate},
        "math": math,
        "chat": chat,
    }


def _make_candidate_after_freeze(fixture) -> None:
    fixture["candidate_checkpoint"].write_bytes(b"new-candidate-checkpoint")
    # Bind the ordering even on filesystems whose default timestamp granularity
    # is coarser than the gap between these two tiny writes.
    manifest_time = fixture["manifest_path"].stat().st_mtime_ns
    os.utime(
        fixture["candidate_checkpoint"],
        ns=(manifest_time + 1_000_000, manifest_time + 1_000_000),
    )


def test_prompt_pool_is_deterministic_and_balanced():
    first = promotion._build_prompt_pool()
    second = promotion._build_prompt_pool()

    assert first == second
    assert set(first) == {
        (family, seed)
        for family in promotion.FAMILIES
        for seed in promotion.EVALUATION_SEEDS
    }
    assert all(
        len(rows)
        == promotion.SAMPLES_PER_FAMILY_PER_SEED * promotion.POOL_MULTIPLIER
        for rows in first.values()
    )


def test_freeze_binds_inputs_balances_cells_and_records_candidate_absence(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    manifest = fixture["manifest"]
    records = manifest["prompt_set"]["records"]

    assert manifest["status"] == "frozen_unscored"
    assert manifest["candidate"]["checkpoint_present_at_freeze"] is False
    assert manifest["candidate"]["checkpoint_sha256_at_freeze"] is None
    assert manifest["pointer_policy"]["write_supported"] is False
    assert manifest["prompt_set"]["sha256"] == promotion._records_sha256(records)
    assert len(records) == 160
    cells = {
        (row["family"], row["generation_seed"]): 0
        for row in records
    }
    for row in records:
        cells[(row["family"], row["generation_seed"])] += 1
    assert set(cells.values()) == {promotion.SAMPLES_PER_FAMILY_PER_SEED}
    assert promotion.sha256_file(fixture["baseline_checkpoint"]) == manifest[
        "baseline"
    ]["checkpoint_sha256"]


def test_freeze_rejects_a_candidate_checkpoint_that_already_exists(tmp_path):
    baseline = tmp_path / "base.pt"
    candidate = tmp_path / "candidate.pt"
    base_corpus = tmp_path / "base.jsonl"
    candidate_corpus = tmp_path / "candidate.jsonl"
    baseline.write_bytes(b"base")
    candidate.write_bytes(b"already scored or populated")
    _write_corpus(base_corpus)
    _write_corpus(candidate_corpus)

    with pytest.raises(FileExistsError, match="must be frozen before candidate scoring"):
        promotion.freeze_manifest(
            baseline_checkpoint=baseline,
            baseline_corpus=base_corpus,
            candidate_corpus=candidate_corpus,
            candidate_checkpoint=candidate,
            output=tmp_path / "manifest.json",
        )


def test_exact_corpus_collisions_are_rejected_from_the_frozen_set(tmp_path):
    pool = promotion._build_prompt_pool()
    first = pool[(promotion.FAMILIES[0], promotion.EVALUATION_SEEDS[0])][0]["prompt"]
    second = pool[(promotion.FAMILIES[1], promotion.EVALUATION_SEEDS[1])][0]["prompt"]
    fixture = _freeze_fixture(
        tmp_path, baseline_prompts=(first,), candidate_prompts=(second,)
    )
    manifest = fixture["manifest"]
    selected = {record["prompt"] for record in manifest["prompt_set"]["records"]}

    assert first not in selected
    assert second not in selected
    assert manifest["prompt_set"]["rejected_exact_collision_prompts"] == 2
    assert manifest["baseline"]["corpus_scan"]["promotion_pool_collision_count"] == 1
    assert manifest["candidate"]["corpus_scan"]["promotion_pool_collision_count"] == 1


def test_legacy_seed65_contamination_is_mandatory_and_family_counted(tmp_path):
    legacy = list(promotion._legacy_seed65_prompts().items())
    fixture = _freeze_fixture(
        tmp_path,
        baseline_prompts=[prompt for prompt, _family in legacy[:2]],
        candidate_prompts=[prompt for prompt, _family in legacy[2:5]],
    )
    disclosure = fixture["manifest"]["legacy_seed65_development_contamination"]

    assert disclosure["classification"] == "adaptive_contaminated_development_benchmark"
    assert disclosure["promotion_authority"] is False
    assert disclosure["baseline_unique_prompt_hits"] == 2
    assert disclosure["candidate_unique_prompt_hits"] == 3
    assert sum(disclosure["baseline_family_hits"].values()) == 2
    assert sum(disclosure["candidate_family_hits"].values()) == 3


def test_passing_review_receipt_binds_checkpoints_and_never_touches_pointer(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    _make_candidate_after_freeze(fixture)
    sentinel_pointer = tmp_path / "active-pointer.txt"
    sentinel_pointer.write_text("keep-me", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"

    receipt = promotion.evaluate_manifest(
        manifest_path=fixture["manifest_path"],
        candidate_checkpoint=fixture["candidate_checkpoint"],
        output=receipt_path,
        no_write_pointer=True,
        runner=_passing_runner,
    )

    assert receipt["passed"] is True
    assert receipt["decision"]["blockers"] == []
    assert receipt["paired_evidence"]["wins"] == 10
    assert receipt["paired_evidence"]["regressions"] == 0
    assert receipt["decision"]["overall_accuracy_gain"] == pytest.approx(10 / 160)
    assert receipt["pointer"] == {
        "write_requested": False,
        "write_supported": False,
        "pointer_path": None,
        "pointer_written": False,
    }
    assert sentinel_pointer.read_text(encoding="utf-8") == "keep-me"
    assert receipt["artifact_binding"]["candidate_checkpoint_sha256"] == promotion.sha256_file(
        fixture["candidate_checkpoint"]
    )
    assert receipt_path.is_file()


def test_review_mode_must_be_explicit(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    _make_candidate_after_freeze(fixture)

    with pytest.raises(ValueError, match="requires explicit --no-write-pointer"):
        promotion.evaluate_manifest(
            manifest_path=fixture["manifest_path"],
            candidate_checkpoint=fixture["candidate_checkpoint"],
            output=tmp_path / "receipt.json",
            no_write_pointer=False,
            runner=_passing_runner,
        )


def test_evaluate_rejects_evaluator_hash_tampering_before_runner(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    _make_candidate_after_freeze(fixture)
    payload = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
    payload["evaluator"]["sha256"] = "0" * 64
    fixture["manifest_path"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="evaluator source changed"):
        promotion.evaluate_manifest(
            manifest_path=fixture["manifest_path"],
            candidate_checkpoint=fixture["candidate_checkpoint"],
            output=tmp_path / "receipt.json",
            no_write_pointer=True,
            runner=lambda *_args: pytest.fail("runner must not be called"),
        )


def test_evaluate_rejects_corpus_change_before_runner(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    _make_candidate_after_freeze(fixture)
    with fixture["candidate_corpus"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"user": "changed", "assistant": "changed"}) + "\n")

    with pytest.raises(ValueError, match="candidate corpus changed"):
        promotion.evaluate_manifest(
            manifest_path=fixture["manifest_path"],
            candidate_checkpoint=fixture["candidate_checkpoint"],
            output=tmp_path / "receipt.json",
            no_write_pointer=True,
            runner=lambda *_args: pytest.fail("runner must not be called"),
        )


def test_gate_fails_on_family_regression_caps_unparsed_and_chat_loss(tmp_path):
    fixture = _freeze_fixture(tmp_path)
    _make_candidate_after_freeze(fixture)

    def failing_runner(checkpoint, math_records, chat_records, protocol):
        candidate = checkpoint.name == "candidate.pt"
        math = []
        for index, record in enumerate(math_records):
            candidate_regression = candidate and record["family"] == "arithmetic"
            capped = candidate and index < 10
            math.append(
                {
                    "id": record["id"],
                    "reply": "" if capped else _answer(record, correct=not candidate_regression),
                    "tokens": promotion.MATH_MAX_NEW_TOKENS if capped else 12,
                    "prompt_unknown_rate": 0.0,
                    "error": "",
                }
            )
        chat = [
            {
                "id": record["id"],
                "reply": "" if candidate else "Check the traceback and isolate the function.",
                "tokens": 0 if candidate else 9,
                "prompt_unknown_rate": 0.0,
                "error": "",
            }
            for record in chat_records
        ]
        return {"checkpoint_metadata": {}, "math": math, "chat": chat}

    receipt = promotion.evaluate_manifest(
        manifest_path=fixture["manifest_path"],
        candidate_checkpoint=fixture["candidate_checkpoint"],
        output=tmp_path / "failed-receipt.json",
        no_write_pointer=True,
        runner=failing_runner,
    )

    blockers = set(receipt["decision"]["blockers"])
    assert receipt["passed"] is False
    assert "family_regression:arithmetic" in blockers
    assert "candidate_generation_cap_rate_above_threshold" in blockers
    assert "candidate_unparsed_rate_above_threshold" in blockers
    assert "candidate_chat_operational_below_threshold" in blockers
    assert "chat_similarity_below_threshold" in blockers
    assert receipt["pointer"]["pointer_written"] is False


def test_frozen_manifest_is_write_once(tmp_path):
    fixture = _freeze_fixture(tmp_path)

    with pytest.raises(FileExistsError, match="already exists"):
        promotion.freeze_manifest(
            baseline_checkpoint=fixture["baseline_checkpoint"],
            baseline_corpus=fixture["baseline_corpus"],
            candidate_corpus=fixture["candidate_corpus"],
            candidate_checkpoint=fixture["candidate_checkpoint"],
            output=fixture["manifest_path"],
        )
