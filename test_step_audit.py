"""The step auditor.

Accuracy tells you a reply was wrong. It does not tell you which of its five
operations was wrong, whether the answer followed from the working, or whether
the working was there at all. `step_audit` answers those, and this pins the
properties that make its answers worth acting on.

The one that matters most is the last section. v86 answers ``420 / 7`` with
``320 / 7 = 60, total 60``: the answer is right and the working that appears to
produce it is false. A tool that reported such a reply as sound would say the
scratchpad is doing work when it is decoration, and every conclusion drawn from
it afterwards would be wrong in the same direction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import step_audit as audit  # noqa: E402


# ---------------------------------------------------------------------------
# Carry load
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("left, right, expected", [
    # The shape a place-value decomposition produces: the partials occupy
    # different columns, so nothing propagates and the "addition" is assembly.
    (350, 45, 0),
    (1100, 660, 0),
    (60, 6, 0),
    # Real addition: the units column reaches twelve.
    (54, 72, 1),
    (58, 14, 1),
    # Every column carries.
    (999, 999, 3),
])
def test_carry_load_counts_columns_that_interact(left, right, expected):
    assert audit.carry_load(left, right, "+") == expected


def test_a_borrow_is_counted_and_not_confused_with_a_carry():
    # `300 + -2` is a subtraction however it is written, and subtraction
    # borrows where addition carries. Counting it as a carry-free addition
    # would call the hardest step in `arithmetic` easy.
    assert audit.carry_load(300, -2, "+") == 2
    assert audit.carry_load(300, 2, "-") == 2
    assert audit.carry_load(370, 20, "-") == 0


def test_decimal_columns_align_on_the_point():
    # `60.3 + 30.15` is the step `percent` performs without writing it.
    assert audit.carry_load(60.3, 30.15, "+") == 0
    # 603 + 598 once scaled: hundredths reach 11, tenths 10, ones 12.
    assert audit.carry_load(6.03, 5.98, "+") == 3


def test_an_undecidable_step_is_not_reported_as_easy():
    """Silence, not zero. A `None` shows up as unknown; a 0 would be a claim."""

    assert audit.carry_load(12, 4, "x") is None
    assert audit.carry_load(12, 4, "/") is None


# ---------------------------------------------------------------------------
# Reading a reply
# ---------------------------------------------------------------------------


def test_a_written_step_is_checked_against_arithmetic():
    report = audit.audit("70 x 5 = 350, 9 x 5 = 45, total 395")
    assert [s.ok for s in report.written] == [True, True]
    assert report.first_bad is None


def test_a_false_step_is_found_and_located():
    """A checker that never says no is decoration.

    This is v86's actual reply to ``What is 15% of 603?``. The plan is right --
    15% is 10% plus 5% -- and the third step is false.
    """

    report = audit.audit(
        "1 percent of 603 = 6.03, times 10 = 60.3, times 5 = 30.65, total 153.25")
    assert report.verdict == "arithmetic_slip"
    bad = report.first_bad
    assert bad is not None
    assert bad.text == "times 5 = 30.65"
    assert bad.expected == pytest.approx(30.15)
    # The two steps before it were sound, so the reply is not noise: it is a
    # correct decomposition with an arithmetic error at a known position.
    assert bad.position == 2


def test_a_relative_step_is_read_against_the_anchor_not_its_neighbour():
    """``times 10`` and ``times 5`` both scale the one-percent value.

    Reading the second as ``60.3 x 5`` would call a true step false, and this
    module's whole output is which steps are false.
    """

    report = audit.audit(
        "1 percent of 603 = 6.03, times 10 = 60.3, times 5 = 30.15, total 90.45")
    assert all(s.ok for s in report.written)


def test_the_unwritten_step_a_decomposition_performs_is_recovered():
    """`multiplication` never writes ``350 + 45 = 395``. It still happened."""

    report = audit.audit("70 x 5 = 350, 9 x 5 = 45, total 395")
    assert report.verdict == "unwritten_step"
    assert len(report.unwritten) == 1
    assert report.unwritten[0].carries == 0
    assert report.unwritten_note == "total 395 = 350 + 45"


def test_a_running_sum_exposes_every_addition_it_hides():
    """``sum: 54 then 126 then 174`` performs two additions and writes none."""

    report = audit.audit("sum: 54 then 126 then 174, total 174, "
                         "divide by 3, total 58")
    hidden = [s for s in report.unwritten if s.kind == "running"]
    assert len(hidden) == 2
    assert hidden[0].left == 54 and hidden[0].right == pytest.approx(72)


def test_the_divisor_step_after_a_running_sum_is_still_checked():
    """The anchor has to survive a shape that writes no operands at all."""

    report = audit.audit("sum: 58 then 60 then 104 then 135 then 212 then 228, "
                         "total 228, divide by 6, total 34.3333")
    bad = report.first_bad
    assert bad is not None and bad.expected == pytest.approx(38.0)


def test_a_chained_reply_is_clean_rather_than_summed():
    """``66 + 20 = 86, 86 - 29 = 57`` chains; its total is the last result.

    Reading this as a sum of partials would invent an addition the reply never
    performed and report a carry load for it.
    """

    report = audit.audit("66 + 20 = 86, 86 - 29 = 57, total 57")
    assert report.verdict == "clean"
    assert report.unwritten == []


def test_an_answer_with_nothing_behind_it_is_named_as_such():
    assert audit.audit("total 42").verdict == "unsupported"
    assert audit.audit("the force is large").verdict == "unreadable"


def test_the_auditor_reads_only_the_reply():
    """No generator, no expected answer, no execution: characters only.

    The audit has to apply equally to a corpus row and to model output, and it
    has to be impossible for it to score a reply using information the reply
    does not contain.
    """

    report = audit.audit("70 x 5 = 350, 9 x 5 = 45, total 395")
    assert report.reply == "70 x 5 = 350, 9 x 5 = 45, total 395"
    assert audit.audit.__code__.co_argcount == 1


# ---------------------------------------------------------------------------
# Working that does not support the answer
# ---------------------------------------------------------------------------


def test_a_right_answer_above_false_working_is_counted_separately():
    """v86's reply to ``Solve this basic math problem: 420 / 7``.

    ``320 / 7`` is not 60, and 60 is the right answer to the question asked.
    The answer therefore did not come from the working, and a summary that
    folded this into "correct" would be reporting a scratchpad that works.
    """

    table = audit.summarise([
        {"task": "division", "reply": "320 / 7 = 60, 0 / 7 = 0, total 60",
         "correct": True},
        {"task": "division", "reply": "150 / 3 = 50, 21 / 3 = 7, total 57",
         "correct": True},
    ])
    entry = table["division"]
    assert entry["correct"] == 2
    assert entry["correct_with_false_step"] == 1
    assert entry["decorative_working_rate"] == 0.5


def test_a_wrong_answer_above_sound_working_is_a_different_failure():
    """Every step true and the conclusion still missed is a plan error.

    Charging that to arithmetic would send the next corpus change to the wrong
    place -- decomposing steps that were already right.
    """

    table = audit.summarise([
        {"task": "word_problem", "reply": "66 + 20 = 86, 86 - 29 = 57, total 57",
         "correct": False},
    ])
    assert table["word_problem"]["wrong_with_sound_steps"] == 1
    assert table["word_problem"]["false_steps"] == 0


def test_summarise_reports_no_rate_where_it_has_no_denominator():
    """A task with nothing correct gets `None`, not a a misleading zero."""

    table = audit.summarise([
        {"task": "average", "reply": "sum: 1 then 3, total 3, divide by 2, total 9",
         "correct": False},
    ])
    assert table["average"]["decorative_working_rate"] is None
