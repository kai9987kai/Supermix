"""The execution-verified code corpus.

v86 has no coding ability because there is no code in the corpus. This family
adds it, and it fits the project's method exactly: a code row is checked by
**running the snippet**, so the interpreter is the exact oracle in the same way
`nexus_solver` is for the science rows.

These tests pin the four properties that make a corpus family safe to train on.
Each has cost this project a run at some point:

* every intermediate inside the learnability envelope (v79 scored 0.03 on rows
  with an undecomposed three-digit product; v80 scored 0.77-0.87 on the same
  tasks with the working shown),
* every reply ending ``total <number>`` (answers are extracted as the last
  number, so ``total 5 m/s^2`` extracts as 2),
* every turn inside the 128-token budget (``_build_turn_aligned_tensors``
  drops a longer turn *silently*),
* every row verified before it ships.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import build_code_corpus as code  # noqa: E402
import eval_problem_solving as solving  # noqa: E402

TERMINAL = re.compile(r"total -?\d+(?:\.\d+)?$")
ADDITION = re.compile(r"(\d+) \+ (\d+) = (\d+)")
SUBTRACTION = re.compile(r"(\d+) - (\d+) = (-?\d+)")


def sample(per_task: int = 25, seed: int = 87):
    rows, report = code.build(per_task, seed, None)
    return rows, report


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


def test_every_row_is_verified_by_executing_it():
    """A shipped row is one the interpreter agreed with. Nothing else ships."""
    rows, report = sample()
    assert rows
    assert report["drop_rate"] == 0.0, (
        f"rows were dropped: {report.get('drop_reasons')}. A non-zero rate is "
        "not a failure in itself, but it must be reported, not silent."
    )
    assert "exec" in report["verified_by"]


def test_execution_reads_the_value_the_snippet_actually_produces():
    """The oracle must actually compute, not restate the claim it was given."""
    result = code.run_snippet("x = 0\nfor i in range(3): x = x + 4", "x")
    assert result.ok, result
    assert result.value == 12


def test_a_derivation_that_disagrees_with_execution_is_rejected():
    """A checker that never says no is decoration.

    `verify` compares the interpreter against the worked answer, so a problem
    whose derivation is wrong must fail even though the snippet itself runs.
    """
    rng = random.Random(3)
    problem = code.TASKS["code_loop_add"](rng)
    assert code.verify(problem).ok

    tampered = code.CodeProblem(
        problem.task, problem.domain, problem.prompt, problem.response,
        problem.answer + 1, problem.unit, problem.canonical, problem.target,
        dict(problem.params),
    )
    verdict = code.verify(tampered)
    assert not verdict.ok
    assert "derivation says" in (verdict.reason or "")


def test_execution_refuses_anything_outside_the_allowlist():
    """Snippets are built from templates, but the executor is the last line."""
    for hostile in ("import os", "open('x')", "__import__('os')",
                    "eval('1')", "x = [].__class__"):
        result = code.run_snippet(hostile, "x")
        assert not result.ok, f"the allowlist admitted: {hostile!r}"


# ---------------------------------------------------------------------------
# The learnability envelope
# ---------------------------------------------------------------------------


def test_every_reply_ends_in_a_bare_number():
    rows, _ = sample()
    bad = [r for r in rows if not TERMINAL.search(r["assistant"].strip())]
    assert not bad, (
        f"{len(bad)} replies do not end 'total <number>'. Answers are "
        f"extracted as the last number, so these score wrong: "
        f"{[r['assistant'][-40:] for r in bad[:3]]}"
    )


def test_the_extractor_recovers_the_intended_answer():
    """The end-to-end property: what the corpus teaches is what scoring reads."""
    rng = random.Random(4)
    for name, generator in sorted(code.TASKS.items()):
        for _ in range(8):
            problem = generator(rng)
            got = solving.extract_answer(problem.response)
            assert got is not None, f"{name}: no number extracted"
            assert abs(got - problem.answer) < 1e-6, (
                f"{name}: extractor read {got}, generator meant {problem.answer}"
            )


def test_no_arithmetic_step_leaves_the_two_digit_envelope():
    """Rule 1: two-digit operands, or the model cannot follow the step."""
    rows, _ = sample(per_task=40)
    worst = 0
    example = ""
    for row in rows:
        for pattern in (ADDITION, SUBTRACTION):
            for match in pattern.finditer(row["assistant"]):
                for operand in (match.group(1), match.group(2)):
                    if int(operand) > worst:
                        worst, example = int(operand), match.group(0)
    assert worst <= 99, (
        f"an operand reached {worst} in '{example}'. v79 emitted "
        "'167 x 11 = 1837' in one jump and scored 0.03 on that task."
    )


def test_every_turn_fits_the_sequence_budget():
    """Rule 3: a turn over 128 tokens is dropped by packing without a word."""
    text_utils = pytest.importorskip("mimomix_text")
    rows, _ = sample(per_task=40)
    tokenizer = text_utils.WordTokenizer.build(
        (field for row in rows for field in (row["user"], row["assistant"])),
        max_vocab=16384, digit_tokens=True,
    )
    lengths = [
        len(tokenizer.encode(row["user"])) + len(tokenizer.encode(row["assistant"]))
        for row in rows
    ]
    over = [n for n in lengths if n >= code.DEFAULT_SEQUENCE_LENGTH]
    assert not over, (
        f"{len(over)} of {len(rows)} turns reach "
        f"{code.DEFAULT_SEQUENCE_LENGTH} tokens (max {max(lengths)}). "
        "Turn-aligned packing discards these and reports nothing."
    )


def test_the_working_is_shown_rather_than_asserted():
    """Rule 4: a scratchpad helps only where it decomposes the operation.

    v86's `average` emits running totals without operands and its individual
    additions are correct 1.5% of the time. A loop trace must not repeat that.
    """
    rng = random.Random(11)
    for _ in range(10):
        problem = code.TASKS["code_loop_add"](rng)
        assert ADDITION.search(problem.response), (
            f"no addition written as an equation in: {problem.response}"
        )


# ---------------------------------------------------------------------------
# Benchmark wiring
# ---------------------------------------------------------------------------


def test_the_code_tasks_are_scored_by_the_same_benchmark():
    assert solving.CODE_TASKS, "no code tasks registered into GENERATORS"
    for name in solving.CODE_TASKS:
        assert name in solving.GENERATORS
        assert name.startswith("code_")


def test_registering_code_tasks_does_not_disturb_the_existing_benchmark():
    """The property that lets the corpus grow without invalidating history.

    Before v82 one RNG was shared across tasks in turn, so adding a task
    shifted every later task's problems for the same seed -- which is why v74
    and v80 were never comparable. The per-task RNG means nine new tasks leave
    the published fingerprint intact.
    """
    original = [t for t in solving.GENERATORS if t not in solving.CODE_TASKS]
    assert solving.generator_fingerprint(original) == (
        "4077062251bc762c9716a730f3818ad2"
    ), "the v80/v86 baseline fingerprint moved; published scores are no longer paired"

    with_only_original = [
        p.prompt for p in solving.generate_novel(630, seed=65, tasks=original)
        if p.task == "force"
    ]
    with_everything = [
        p.prompt for p in solving.generate_novel(900, seed=65) if p.task == "force"
    ]
    shared = min(len(with_only_original), len(with_everything))
    assert with_only_original[:shared] == with_everything[:shared]
