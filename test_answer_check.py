"""Tests for live answer verification.

The interface renders three states: CORRECT, WRONG, and NOT CHECKED. The
dangerous failure is a false CORRECT -- it would present wrong arithmetic as
verified, which is worse than showing nothing. The second-worst is a confident
verdict on a question the parser only half-recognised, so the parsers are
deliberately narrow and both directions are tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import answer_check as check  # noqa: E402


# -- the five supported shapes ---------------------------------------------


@pytest.mark.parametrize(
    "question,expected,task",
    [
        ("Solve this basic math problem: 617 + 288", 905.0, "arithmetic"),
        ("Quick question: 524 - 305", 219.0, "arithmetic"),
        ("What is 25% of 840?", 210.0, "percent"),
        ("Solve for x: x + 14 = 39", 25.0, "algebra_one_step"),
        ("Solve for x: x - 10 = 26", 36.0, "algebra_one_step"),
        (
            "A student has 45 marbles. They get 38 more and then give away 27. "
            "How many marbles do they have now?",
            56.0,
            "word_problem",
        ),
        ("Find the average (mean) of these numbers: 40, 60, 20, 80", 50.0, "average"),
    ],
)
def test_ground_truth_is_recomputed_from_the_question(question, expected, task):
    parsed = check.parse_question(question)

    assert parsed is not None
    assert parsed[0] == task
    assert parsed[1] == pytest.approx(expected)


def test_correct_reply_is_marked_correct():
    verdict = check.check(
        "Solve this basic math problem: 617 + 288",
        "600 + 200 = 800, 17 + 88 = 105, total 905",
    )

    assert verdict is not None and verdict.correct


def test_wrong_reply_is_marked_wrong():
    """The v70 failure mode: correct working, slipped digit."""

    verdict = check.check(
        "Solve this basic math problem: 617 + 288",
        "600 + 200 = 800, 17 + 88 = 107, total 907",
    )

    assert verdict is not None and not verdict.correct
    assert verdict.predicted == 907.0 and verdict.expected == 905.0


def test_scratchpad_is_scored_on_its_final_number():
    """Taking the first number would score the working, not the answer."""

    verdict = check.check("Quick question: 524 - 305", "500 - 300 = 200, 24 - 5 = 19, total 219")

    assert verdict.correct


# -- refusing to judge ------------------------------------------------------


def test_a_chat_question_is_not_checked():
    """NOT CHECKED must not be reachable as a pass."""

    assert check.check("why is my script failing", "Check the traceback first.") is None


@pytest.mark.parametrize(
    "question",
    ["hello", "what is your name", "tell me a story about the sea", ""],
)
def test_non_maths_questions_return_none(question):
    assert check.parse_question(question) is None


def test_a_reply_with_no_number_is_wrong_not_unchecked():
    """The question was checkable; the reply simply failed to answer it."""

    verdict = check.check("What is 25% of 840?", "Synonym: glad.")

    assert verdict is not None
    assert verdict.predicted is None and not verdict.correct


# -- parser precedence ------------------------------------------------------


def test_word_problem_is_not_read_as_bare_arithmetic():
    """It contains three numbers; a naive 'a + b' search would seize on them."""

    parsed = check.parse_question(
        "A student has 45 marbles. They get 38 more and then give away 27. "
        "How many marbles do they have now?"
    )

    assert parsed[0] == "word_problem"
    assert parsed[1] == 56.0


def test_average_is_not_read_as_bare_arithmetic():
    parsed = check.parse_question("Find the average (mean) of these numbers: 40, 60, 20, 80")

    assert parsed[0] == "average"


def test_algebra_is_not_read_as_bare_arithmetic():
    """`x + 14 = 39` contains '+' but the answer is 25, not 53."""

    parsed = check.parse_question("Solve for x: x + 14 = 39")

    assert parsed[0] == "algebra_one_step"
    assert parsed[1] == 25.0


# -- agreement with the offline benchmark -----------------------------------


def test_extract_answer_matches_the_benchmark():
    """The live check and the benchmark must not disagree about a reply."""

    import eval_problem_solving as offline

    for reply in ("79", "376 - 17 = 359", "Calculate directly. Answer: 10.68.", "9/14"):
        assert check.extract_answer(reply) == offline.extract_answer(reply)


def test_tolerance_matches_the_benchmark():
    import eval_problem_solving as offline

    assert check.TOLERANCE == offline.TOLERANCE


def test_supported_shapes_are_all_parseable():
    """The interface advertises these; every one must actually verify."""

    for example in check.supported_shapes():
        assert check.parse_question(example) is not None


# -- the v74 task types (v76) -----------------------------------------------
#
# v74 added multiplication, division, sequence and two-step, and scores
# 1.00/1.00/0.98/0.98 on them. Until these parsers existed the live checker
# reported NOT CHECKED for the model's four strongest tasks.


def test_multiplication_is_checked():
    result = check.parse_question("What is 25 x 7?")

    assert result == ("multiplication", 175.0)


def test_multiplication_accepts_an_asterisk():
    assert check.parse_question("What is 25 * 7?")[1] == 175.0


def test_division_is_checked():
    assert check.parse_question("Quick question: 70 / 5") == ("division", 14.0)


def test_division_by_zero_is_not_checkable_rather_than_an_error():
    assert check.parse_question("Quick question: 70 / 0") is None


def test_sequence_is_checked():
    result = check.parse_question("What comes next in the sequence: 7, 17, 27, 37?")

    assert result == ("sequence", 47.0)


def test_sequence_with_uneven_steps_is_not_checkable():
    """Guessing a rule would invent a right answer where none was derived."""

    assert check.parse_question("What comes next in the sequence: 1, 2, 4, 8?") is None


def test_sequence_needs_at_least_three_terms():
    assert check.parse_question("What comes next in the sequence: 5, 10?") is None


def test_two_step_is_checked():
    result = check.parse_question("What is 50% of 698, then add 28?")

    assert result[0] == "two_step"
    assert result[1] == pytest.approx(377.0)


def test_two_step_subtract():
    result = check.parse_question("What is 50% of 200, then subtract 30?")

    assert result[1] == pytest.approx(70.0)


def test_two_step_beats_percent_in_parser_order():
    """A two-step question contains a percent question; percent must not win."""

    assert check.parse_question("What is 20% of 150, then add 12?")[0] == "two_step"


def test_algebra_still_wins_over_multiplication():
    """`x` is both the multiplication sign and the unknown in this corpus."""

    assert check.parse_question("Solve for x: x + 14 = 39")[0] == "algebra_one_step"


def test_sequence_wins_over_average():
    """Both are comma-separated number lists."""

    assert check.parse_question(
        "What comes next in the sequence: 7, 17, 27, 37?"
    )[0] == "sequence"


def test_average_still_parses_as_average():
    assert check.parse_question(
        "Find the average (mean) of these numbers: 40, 60, 20, 80"
    ) == ("average", 50.0)


def test_addition_still_parses_as_arithmetic():
    assert check.parse_question("Solve this basic math problem: 617 + 288")[0] == "arithmetic"


def test_every_advertised_shape_actually_parses():
    """A shape the interface advertises but cannot verify would be a lie."""

    for shape in check.supported_shapes():
        assert check.parse_question(shape) is not None, shape


def test_conversation_is_still_not_checkable():
    for text in ("hello", "why is my script failing", "what is your name"):
        assert check.check(text, "anything") is None
