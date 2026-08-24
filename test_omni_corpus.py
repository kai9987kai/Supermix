"""Tests for the solver-verified scientific corpus.

The claim this corpus makes is unusual for this repository: **every row is
independently verified**. A generator computes an answer, and `nexus_solver`
recomputes it exactly from a parsed query; disagreement drops the row. These
tests exist to make sure that claim stays true, and that two specific
previously-observed failures cannot recur.

1. v74 was format-brittle -- one template per task, and 0 of 5 naturally-typed
   questions answered correctly. Each task here must carry several phrasings.
2. Answers are extracted as the last number in the reply, so a response ending
   `total 5 m/s^2` extracts as **2**. No response may end in a unit.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import build_omni_corpus as omni  # noqa: E402


ALL_TASKS = sorted(omni.TASKS)


def _sample(task: str, seed: int = 0):
    return omni.TASKS[task](random.Random(seed))


# -- the verification claim -------------------------------------------------


@pytest.mark.parametrize("task", ALL_TASKS)
def test_every_task_verifies_against_the_solver(task):
    """The whole premise: an independent exact solver agrees."""

    for seed in range(5):
        problem = _sample(task, seed)
        assert omni.verify(problem), f"{task} seed {seed} disagreed with the solver"


@pytest.mark.parametrize("task", ALL_TASKS)
def test_the_response_parses_to_the_stated_answer(task):
    """A right answer the benchmark cannot read scores as wrong."""

    for seed in range(5):
        problem = _sample(task, seed)
        assert omni.extract_answer(problem.response) == problem.answer


def test_verification_rejects_a_wrong_answer():
    """A verifier that passed everything would be decoration."""

    problem = _sample("force")
    broken = omni.OmniProblem(
        problem.task, problem.domain, problem.prompt, problem.response,
        problem.answer + 1, problem.unit, problem.canonical,
    )

    assert omni.verify(broken) is False


def test_verification_rejects_an_unparseable_query():
    problem = _sample("force")
    broken = omni.OmniProblem(
        problem.task, problem.domain, problem.prompt, problem.response,
        problem.answer, problem.unit, "this is not a solvable query",
    )

    assert omni.verify(broken) is False


# -- the v76 lesson: phrasing variety ---------------------------------------


@pytest.mark.parametrize("task", ALL_TASKS)
def test_each_task_has_more_than_one_phrasing(task):
    """v74 had one template per task and learned the template, not the task."""

    prompts = {_sample(task, seed).prompt for seed in range(40)}

    assert len(prompts) > 1, f"{task} produced only one phrasing"


def test_phrasings_differ_in_wording_not_only_in_numbers():
    """Different numbers in the same sentence is not variety."""

    shapes = set()
    for seed in range(60):
        prompt = _sample("force", seed).prompt
        shapes.add("".join(c for c in prompt if not c.isdigit()))

    assert len(shapes) >= 3


def test_the_canonical_query_is_independent_of_the_phrasing():
    """The model sees variety; the oracle always sees a form it parses."""

    for seed in range(20):
        problem = _sample("force", seed)
        assert "find the force" in problem.canonical
        assert omni.verify(problem)


# -- the extraction trap ----------------------------------------------------


@pytest.mark.parametrize("task", ALL_TASKS)
def test_no_response_ends_in_a_unit(task):
    """`total 5 m/s^2` extracts as 2. Every reply must end `total <number>`."""

    import re

    for seed in range(5):
        response = _sample(task, seed).response
        assert re.search(r"total -?\d+(?:\.\d+)?$", response), response


def test_the_trap_is_real():
    """Pinned so the rule above is not mistaken for style."""

    assert omni.extract_answer("the answer is 5 m/s^2") == 2.0
    assert omni.extract_answer("total 5") == 5.0


# -- number rendering -------------------------------------------------------


def test_integers_render_without_a_trailing_decimal():
    assert omni._number(60) == "60"
    assert omni._number(60.0) == "60"


def test_non_integers_are_kept():
    assert omni._number(0.5) == "0.5"


# -- corpus construction ----------------------------------------------------


def test_build_produces_the_requested_count():
    rows, report = omni.build(per_task=3, seed=5)

    assert report["rows"] == len(rows)
    assert all(count == 3 for count in report["per_task"].values())


def test_build_rows_carry_the_training_fields():
    rows, _ = omni.build(per_task=2, seed=6)

    for row in rows:
        assert set(row) == {"user", "assistant", "domain", "task"}
        assert row["user"] and row["assistant"]


def test_build_can_be_restricted_to_named_tasks():
    rows, report = omni.build(per_task=2, seed=7, tasks=["force", "momentum"])

    assert set(report["per_task"]) == {"force", "momentum"}
    assert {row["task"] for row in rows} == {"force", "momentum"}


def test_build_is_reproducible_for_a_seed():
    first, _ = omni.build(per_task=3, seed=11, tasks=["work"])
    second, _ = omni.build(per_task=3, seed=11, tasks=["work"])

    assert first == second


def test_different_seeds_give_different_problems():
    first, _ = omni.build(per_task=3, seed=11, tasks=["work"])
    second, _ = omni.build(per_task=3, seed=12, tasks=["work"])

    assert first != second


def test_the_report_records_how_verification_went():
    _, report = omni.build(per_task=2, seed=8)

    assert report["verified_by"] == "nexus_solver.solve_problem"
    assert "dropped_failing_verification" in report
    assert report["tolerance"] == omni.TOLERANCE


def test_the_corpus_spans_more_than_one_domain():
    """The point is knowledge beyond arithmetic."""

    _, report = omni.build(per_task=1, seed=9)
    domains = {omni.TASKS[t](random.Random(0)).domain for t in report["per_task"]}

    assert len(domains) >= 3
    assert "physics" in domains and "chemistry" in domains


def test_an_exhausted_task_is_reported_not_padded():
    """`combination` over n<=40, k<=8 holds only about a thousand questions.

    Asking for more must not ship the same question hundreds of times --
    duplicated rows are precisely what a recitation-proof benchmark exists to
    punish.
    """

    rows, report = omni.build(per_task=5000, seed=13, tasks=["combination"])

    assert "combination" in report["short_of_requested"]
    assert report["short_of_requested"]["combination"]["produced"] == len(rows)
    assert report["short_of_requested"]["combination"]["asked"] == 5000


def test_no_duplicate_prompts_within_a_task():
    rows, _ = omni.build(per_task=5000, seed=14, tasks=["combination"])
    prompts = [row["user"] for row in rows]

    assert len(prompts) == len(set(prompts))


def test_a_task_with_room_is_not_reported_short():
    _, report = omni.build(per_task=200, seed=15, tasks=["work"])

    assert report["short_of_requested"] == {}


def test_every_generated_row_in_a_build_verifies():
    """End to end: nothing unverified reaches the corpus."""

    rng = random.Random(3)
    for task in ALL_TASKS:
        for _ in range(3):
            assert omni.verify(omni.TASKS[task](rng))
