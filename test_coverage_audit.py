"""The train/eval coverage audit, and the hole it was written to find.

`percent` scored 0.533 on v86 and was read as a hard task. It is two tasks
averaged: 16/17 on the percentages the corpus teaches and 0/13 on the two it
does not. `_scratchpad_percent` drew from [5, 10, 20, 25, 50] and
`eval_problem_solving._percent` draws from [5, 10, 12, 15, 20, 25].

Nothing reported that for eight versions. The generators are in different
files, neither imports the other, and their disagreement shows up as a
plausible-looking score rather than an error. These tests pin the audit that
finds such a gap and the coverage it now has to keep.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import build_scratchpad_math as scratch  # noqa: E402
import coverage_audit as coverage  # noqa: E402
import eval_problem_solving as solving  # noqa: E402


# ---------------------------------------------------------------------------
# The audit
# ---------------------------------------------------------------------------


def test_it_finds_a_value_the_benchmark_asks_for_and_the_corpus_never_teaches():
    """The exact shape of the percent defect, in miniature.

    A checker that never says no is decoration, so this is the test that the
    audit can fail at all.
    """

    # The bases start above every percentage on purpose. Overlapping them
    # would let `What is 5% of 50?` count as teaching the value 50, which is
    # true of the audit as written -- it compares values, not roles -- and
    # would make this test pass for the wrong reason.
    report = coverage.compare(
        corpus_prompts=[f"What is {p}% of {n}?"
                        for p in (5, 10, 20, 25, 50) for n in range(200, 300)],
        benchmark_prompts=[f"What is {p}% of {n}?"
                           for p in (5, 10, 12, 15, 20, 25) for n in range(200, 300)],
    )
    missing = {hole["value"] for hole in report["asked_but_never_taught"]}
    assert missing == {12.0, 15.0}
    assert {entry["value"] for entry in report["taught_but_never_asked"]} == {50.0}


def test_a_rare_draw_is_not_reported_as_a_hole():
    """A value in one prompt of a thousand is a tail, not a defect.

    Without the threshold every continuous range would report dozens of
    "holes" -- any operand the corpus happened not to draw -- and the audit
    would be noise that nobody reads.
    """

    corpus = [f"add {n}" for n in range(1, 500)]
    benchmark = [f"add {n}" for n in range(1, 500)] + ["add 9999"]
    report = coverage.compare(corpus, benchmark, threshold=0.02)
    assert report["asked_but_never_taught"] == []

    loud = coverage.compare(corpus, benchmark + ["add 9999"] * 200, threshold=0.02)
    assert [hole["value"] for hole in loud["asked_but_never_taught"]] == [9999.0]


def test_the_audit_reports_how_often_the_benchmark_asks():
    """A share, so a reader can tell a defect from a tail without rerunning."""

    report = coverage.compare(["x is 1"] * 10, ["x is 1"] * 5 + ["x is 7"] * 5)
    hole = report["asked_but_never_taught"][0]
    assert hole["value"] == 7.0
    assert hole["share_of_benchmark"] == 0.5
    assert hole["benchmark_prompts"] == 5


# ---------------------------------------------------------------------------
# The coverage it now has to keep
# ---------------------------------------------------------------------------


def test_the_corpus_teaches_every_percentage_the_benchmark_asks_for():
    """The regression this whole file exists to prevent.

    Adding a percentage to either generator without the other reopens a hole
    worth eleven benchmark problems in thirty.
    """

    corpus = set()
    rng = random.Random(87)
    for _ in range(3000):
        corpus.update(coverage.numbers_in(
            scratch._scratchpad_percent(rng)["expression"]))

    benchmark = set()
    bench_rng = solving.task_rng("percent", 87)
    for _ in range(3000):
        benchmark.add(float(
            solving.GENERATORS["percent"](bench_rng).prompt.split("%")[0].split()[-1]))

    assert benchmark <= corpus, (
        f"the benchmark asks for {sorted(benchmark - corpus)} and the corpus "
        "never teaches it"
    )


def test_no_task_asks_about_a_value_its_corpus_never_teaches():
    """The audit run over every task, as a gate rather than a report."""

    report = coverage.audit(samples=1500, seed=87)
    offenders = {
        name: entry["asked_but_never_taught"]
        for name, entry in report["tasks"].items()
        if entry["asked_but_never_taught"]
    }
    assert not offenders, offenders


def test_every_benchmark_task_is_either_compared_or_named_as_uncompared():
    """Silence about a task would make an empty report look like a pass."""

    report = coverage.audit(samples=200, seed=87)
    covered = set(report["tasks"]) | set(report["not_compared"])
    assert covered == set(solving.GENERATORS)


# ---------------------------------------------------------------------------
# The percent format
# ---------------------------------------------------------------------------


def test_a_split_percentage_writes_the_sum_of_its_parts():
    """v86 wrote `times 10 = 5.6, times 5 = 2.8, total 1.8`.

    Both parts right, and then 5.6 + 2.8 produced 1.8. Ten of fourteen wrong
    percent replies have every written step true, so the silent addition is
    where they go.
    """

    scratch.DECOMPOSE_INNER = True
    try:
        rng = random.Random(4)
        split = [row for row in (scratch._scratchpad_percent(rng) for _ in range(600))
                 if row["working"].count("times ") == 2]
        assert split, "no two-part percentage was generated"
        for row in split:
            assert " + " in row["working"], row["working"]
            stated = float(row["working"].rsplit("total ", 1)[1])
            assert abs(stated - row["answer"]) < 1e-4
    finally:
        scratch.DECOMPOSE_INNER = False


def test_a_single_part_percentage_has_nothing_to_add():
    """`times 50` alone: writing `x + 0 = x` would teach a step that is noise."""

    scratch.DECOMPOSE_INNER = True
    try:
        rng = random.Random(4)
        single = [row for row in (scratch._scratchpad_percent(rng) for _ in range(600))
                  if row["working"].count("times ") == 1]
        assert single
        for row in single:
            assert " + " not in row["working"], row["working"]
    finally:
        scratch.DECOMPOSE_INNER = False
