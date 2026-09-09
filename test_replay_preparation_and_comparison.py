"""Regression checks for cohort integrity and sign-resolved training targets."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from compare_problem_transcripts import compare, mcnemar_exact
from prepare_mimomix_replay import group_id, repair_algebra
from eval_problem_solving import extract_answer
from step_audit import audit


def write_rows(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def row(prompt="2 + 3", reply="total 5", expected=5, correct=True):
    return dict(source="novel", task="arithmetic", prompt=prompt, reply=reply,
                expected=expected, correct=correct, tokens=3, truncated=False)


def test_partial_cohort_requires_explicit_opt_in(tmp_path):
    a = write_rows(tmp_path, "a.jsonl", [row(), row("3 + 3", "total 6", 6)])
    b = write_rows(tmp_path, "b.jsonl", [row()])
    with pytest.raises(ValueError, match="different input cohorts"):
        compare(a, b)
    result = compare(a, b, allow_partial=True)
    assert result["overall"]["n"] == 1
    assert result["excluded_baseline_by_task"] == {"arithmetic": 1}
    assert result["promotion_authorized"] is False


def test_duplicate_draws_do_not_inflate_evidence(tmp_path):
    a = write_rows(tmp_path, "a.jsonl", [row(), row()])
    result = compare(a, a)
    assert result["overall"]["n"] == 1
    assert result["baseline"]["duplicate_draws"] == 1
    b = write_rows(tmp_path, "b.jsonl", [row(), row(reply="total 7", correct=False)])
    with pytest.raises(ValueError, match="conflicting duplicate"):
        compare(a, b)


def test_tampered_scores_and_targets_rejected(tmp_path):
    a = write_rows(tmp_path, "a.jsonl", [row()])
    b = write_rows(tmp_path, "b.jsonl", [row(reply="total 7")])
    with pytest.raises(ValueError, match="fresh scoring"):
        compare(a, b)
    c = write_rows(tmp_path, "c.jsonl", [row(reply="total 7", expected=7)])
    with pytest.raises(ValueError, match="expected answers differ"):
        compare(a, c)


def test_exact_paired_tail():
    assert mcnemar_exact(0, 0) == 1
    assert mcnemar_exact(0, 5) == .0625
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_operand_groups_survive_rewording_and_permutation():
    assert group_id("average", "Average 10, 20 and 30") == group_id("average", "Mean of 30, 10, 20")
    assert group_id("algebra", "x + -12 = 3") != group_id("algebra", "x + 12 = 3")


def test_all_algebra_sign_targets_preserve_answer():
    for x in range(-30, 31):
        for constant in range(-30, 31):
            prompt = f"Solve for x: x + {constant} = {x + constant}"
            reply = repair_algebra(prompt)
            assert extract_answer(reply) == x
            # The independent written-step checker catches a changed sign or
            # incorrect place-value split even if the final total is correct.
            result = audit(reply)
            assert not any(not step.ok for step in result.steps)


def test_algebra_repair_rejects_compound_prompt():
    with pytest.raises(ValueError, match="unsupported algebra"):
        repair_algebra("Solve for x: x + 3 = 6; then double x")
