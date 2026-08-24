"""Tests for rewriting a typed question into v74's training format.

The measured problem: v74 scores 0.894 on the benchmark, which generates
prompts in the corpus format, and answered every naturally-phrased chat
question wrongly. Probing isolated the cause to the operator token and the
presence of a lead-in.

The risk in a fix like this is that it quietly does more than presentation --
changes an operand, reorders an equation, or mangles ordinary conversation on
its way past. Most of these tests exist to pin that it does not.
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

import prompt_normaliser as pn  # noqa: E402


# -- the measured failures --------------------------------------------------


@pytest.mark.parametrize("text", [
    "what is 47 times 6",
    "what is 47 * 6",
    "47 multiplied by 6",
    "What is 47 x 6?",
])
def test_multiplication_reaches_the_trained_form(text):
    result = pn.normalise(text)

    assert result.prompt == "What is 47 x 6?"
    assert result.rule == "multiplication"


@pytest.mark.parametrize("text", [
    "what is 128 divided by 8",
    "128 / 8",
    "what is 128 over 8",
])
def test_division_reaches_the_trained_form(text):
    result = pn.normalise(text)

    assert result.prompt == "Quick question: 128 / 8"
    assert result.rule == "division"


def test_addition_reaches_the_trained_form():
    assert pn.normalise("what is 721 plus 513").prompt == "Please help with this. 721 + 513"


def test_subtraction_reaches_the_trained_form():
    assert pn.normalise("832 minus 630").prompt == "Solve this basic math problem: 832 - 630"


def test_subtract_from_reverses_the_operands():
    """'subtract 3 from 10' is 10 - 3, not 3 - 10."""

    assert pn.normalise("subtract 3 from 10").prompt.endswith("10 - 3")


# -- operand integrity ------------------------------------------------------


def test_numbers_are_never_altered():
    result = pn.normalise("what is 1234 times 5678")

    assert "1234" in result.prompt and "5678" in result.prompt


def test_decimals_survive():
    assert "2.5" in pn.normalise("what is 2.5 times 4").prompt


def test_operand_order_is_preserved():
    """A - B is not B - A; a normaliser that sorted them would be silently wrong."""

    assert pn.normalise("100 minus 7").prompt.endswith("100 - 7")


def test_nothing_is_computed():
    """The answer must never appear in the rewritten prompt."""

    assert "282" not in pn.normalise("what is 47 times 6").prompt


# -- the other task shapes --------------------------------------------------


def test_percent():
    result = pn.normalise("what is 15% of 240")

    assert result.prompt == "What is 15% of 240?"
    assert result.rule == "percent"


def test_percent_spelled_out():
    assert pn.normalise("what is 15 percent of 240").prompt == "What is 15% of 240?"


def test_two_step_beats_percent():
    """A two-step question contains a percent question; order must favour it."""

    result = pn.normalise("what is 20% of 150 then add 12")

    assert result.rule == "two_step"
    assert result.prompt == "What is 20% of 150, then add 12?"


def test_two_step_subtract():
    assert pn.normalise("what is 20% of 150 then subtract 12").prompt.endswith(
        "then subtract 12?"
    )


def test_average():
    result = pn.normalise("what is the average of 12, 18 and 30")

    assert result.prompt == "Find the average (mean) of these numbers: 12, 18, 30"
    assert result.rule == "average"


def test_mean_is_a_synonym():
    assert pn.normalise("mean of 4 and 8").rule == "average"


def test_sequence():
    result = pn.normalise("what comes next: 5, 12, 19, 26")

    assert result.prompt == "What comes next in the sequence: 5, 12, 19, 26?"
    assert result.rule == "sequence"


def test_sequence_needs_enough_terms():
    """Two numbers do not establish a sequence; it must not guess."""

    assert pn.normalise("what comes next 5, 12").rule != "sequence"


def test_algebra_keeps_both_sides_in_order():
    result = pn.normalise("solve for x: x + 5 = 12")

    assert result.prompt == "Solve for x: x + 5 = 12"
    assert result.rule == "algebra_one_step"


def test_algebra_negative_result_is_not_read_as_subtraction():
    assert pn.normalise("solve for x: x + 0 = -12").prompt.endswith("= -12")


# -- everything it must leave alone -----------------------------------------


@pytest.mark.parametrize("text", [
    "hello",
    "can you help me with tests",
    "why is my script failing",
    "what is your name",
])
def test_conversation_passes_through_untouched(text):
    result = pn.normalise(text)

    assert result.prompt == text
    assert result.rule is None
    assert result.changed is False


def test_word_problems_pass_through_untouched():
    """The corpus writes these in prose already."""

    text = ("A student has 68 cookies. They get 32 more and then give away 60. "
            "How many cookies do they have now?")

    assert pn.normalise(text).prompt == text
    assert pn.normalise(text).rule is None


def test_empty_input_is_safe():
    assert pn.normalise("").prompt == ""
    assert pn.normalise("   ").rule is None


def test_changed_is_false_when_already_in_the_trained_form():
    """Recognised but identical: the caller should not report a rewrite."""

    result = pn.normalise("What is 47 x 6?")

    assert result.changed is False


def test_changed_is_true_when_actually_rewritten():
    assert pn.normalise("what is 47 times 6").changed is True


def test_the_original_is_always_retained_for_display():
    result = pn.normalise("what is 47 times 6")

    assert result.original == "what is 47 times 6"
