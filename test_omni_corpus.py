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

    rows, report = omni.build(per_task=5000, seed=13, tasks=["combination"],
                              repeat=False)

    assert "combination" in report["short_of_requested"]
    assert report["short_of_requested"]["combination"]["produced"] == len(rows)
    assert report["short_of_requested"]["combination"]["asked"] == 5000


def test_no_duplicate_prompts_within_a_task():
    rows, _ = omni.build(per_task=5000, seed=14, tasks=["combination"],
                         repeat=False)
    prompts = [row["user"] for row in rows]

    assert len(prompts) == len(set(prompts))


def test_a_task_with_room_is_not_reported_short():
    _, report = omni.build(per_task=200, seed=15, tasks=["work"], repeat=False)

    assert report["short_of_requested"] == {}


def test_repetition_is_the_default_and_is_reported():
    """v74's winning task repeats 712 pairs 56x; the factor must be visible."""

    rows, report = omni.build(per_task=600, seed=16, tasks=["force"])

    assert len(rows) == 600
    assert report["distinct_prompts"]["force"] <= 600
    assert report["repetition"]["force"] >= 1.0


def test_unique_mode_still_available():
    rows, _ = omni.build(per_task=300, seed=17, tasks=["force"], repeat=False)
    prompts = [row["user"] for row in rows]

    assert len(prompts) == len(set(prompts))


# -- showing the working (v80) ----------------------------------------------
#
# v79 learned the physics and failed the arithmetic. Measured on the finished
# model, asking for force from mass x acceleration:
#
#     single-digit operands   12/12 correct
#     two-digit                9/12
#     three-digit              1/12
#
# The first version of this module wrote `167 x 11 = 1837` in one step. v66
# established the model cannot do that, and v74's arithmetic corpus never asks
# it to -- it splits every product by place value and never uses an operand
# above 99.

MULTIPLICATIVE = ["force", "momentum", "work", "voltage",
                  "electrical_power", "wave_speed"]


def test_decomposition_matches_v74s_proven_format():
    """Byte-identical to the corpus that scored 0.93 on multiplication.

    v74 writes two partial products and goes straight to the total; it does
    not write the addition out, and the model learned to do it. Departing
    from the only format measured to work would be a guess.
    """

    assert omni.decompose_product(80, 3) == "80 x 3 = 240, 0 x 3 = 0"
    assert omni.decompose_product(25, 7) == "20 x 7 = 140, 5 x 7 = 35"
    assert omni.decompose_product(96, 2) == "90 x 2 = 180, 6 x 2 = 12"


def test_decomposition_splits_by_place_value():
    working = omni.decompose_product(167, 11)

    assert "100 x 11 = 1100" in working
    assert "60 x 11 = 660" in working
    assert "7 x 11 = 77" in working


def test_decomposition_shows_running_sums():
    """Without these the model must hold partial products in its head."""

    working = omni.decompose_product(167, 11)

    assert "1100 + 660 = 1760" in working
    assert "1760 + 77 = 1837" in working


def test_a_single_digit_operand_needs_no_running_sum():
    working = omni.decompose_product(7, 6)

    assert working == "7 x 6 = 42"


def test_zero_places_are_kept_for_a_uniform_shape():
    """v74 keeps them, so every problem has the same number of steps.

    An earlier version dropped zero terms as noise. v74 -- the corpus that
    worked -- writes `0 x 3 = 0`, giving a two-digit problem exactly two
    partial products every time.
    """

    assert omni.decompose_product(70, 7) == "70 x 7 = 490, 0 x 7 = 0"
    assert "0 x 3 = 0" in omni.decompose_product(105, 3)


@pytest.mark.parametrize("a,b", [(167, 11), (400, 60), (1234, 7)])
def test_multi_part_decompositions_carry_a_running_sum(a, b):
    """Three or more parts need the addition written out; two do not."""

    assert omni.decompose_product(a, b).endswith(f"= {a * b}")


@pytest.mark.parametrize("a,b", [(11, 2), (99, 9), (25, 7), (80, 3)])
def test_two_digit_products_use_v74s_two_part_form(a, b):
    working = omni.decompose_product(a, b)
    terms = [piece.strip() for piece in working.split(",")]

    assert len(terms) == 2, working
    assert " + " not in working  # v74 writes no explicit addition step
    tens, units = divmod(a, 10)
    assert terms[0] == f"{tens * 10} x {b} = {tens * 10 * b}"
    assert terms[1] == f"{units} x {b} = {units * b}"


@pytest.mark.parametrize("task", MULTIPLICATIVE)
def test_multiplicative_tasks_show_their_working(task):
    """The v79 failure, pinned: no product may appear as a single step."""

    for seed in range(8):
        problem = _sample(task, seed)
        operands = [v for v in problem.params.values() if isinstance(v, int)]
        if not operands or max(operands) < 10:
            continue  # single-digit products legitimately need no split
        # A response that jumped straight to the answer would contain the
        # answer exactly once before "total"; a decomposed one shows parts.
        assert " x " in problem.response
        assert problem.response.count("=") >= 2, problem.response


@pytest.mark.parametrize("task", MULTIPLICATIVE)
def test_decomposed_responses_still_verify(task):
    """Showing the working must not change the answer."""

    for seed in range(8):
        problem = _sample(task, seed)
        assert omni.verify(problem)
        assert omni.extract_answer(problem.response) == problem.answer


def test_every_generated_row_in_a_build_verifies():
    """End to end: nothing unverified reaches the corpus."""

    rng = random.Random(3)
    for task in ALL_TASKS:
        for _ in range(3):
            assert omni.verify(omni.TASKS[task](rng))


# -- the v82 options must not disturb this corpus ---------------------------
#
# v80 is the only trained artifact this module has produced, and every option
# added in v82 is an untested hypothesis. A default that moved would make the
# next run incomparable with the one measurement in hand. `test_corpus_v82.py`
# exercises the options themselves; these two keep the *absence* of them
# pinned here, next to the tests that describe the shipped corpus.


def test_the_shipped_row_still_has_exactly_four_fields():
    """`--keep_canonical` adds a fifth; it must stay opt-in."""

    rows, _ = omni.build(per_task=2, seed=6)

    assert all(set(row) == {"user", "assistant", "domain", "task"} for row in rows)


def test_naming_every_v82_option_at_its_default_changes_nothing():
    baseline, _ = omni.build(per_task=20, seed=21, tasks=["combination", "force"])
    explicit, _ = omni.build(per_task=20, seed=21, tasks=["combination", "force"],
                             retry_rate=0.0, balanced_operands=False,
                             priming_fraction=0.0, keep_canonical=False)

    assert baseline == explicit
    assert omni.COMBINATION_IN_ENVELOPE is False
