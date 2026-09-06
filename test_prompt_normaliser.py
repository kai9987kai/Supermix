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

import re
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


# ---------------------------------------------------------------------------
# v85: science rules
#
# v80 answers 5 of 10 naturally-typed questions. Three of the five failures are
# physics questions the normaliser had no rule for, because it only ever covered
# arithmetic. These map the way a person writes a physics question onto the terse
# labelled form `build_omni_corpus` generates.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, rule, expected",
    [
        (
            "If something weighs 25 kg and speeds up at 4 metres per second "
            "squared, what force is that?",
            "force",
            "Given mass 25 kg and acceleration 4 m/s^2, compute the force.",
        ),
        (
            "A 30 kg mass is pushed with 90 N. How fast does it accelerate?",
            "acceleration",
            "force 90 N mass 30 kg find acceleration",
        ),
        (
            "how much momentum does a 14 kg trolley moving at 5 m/s have?",
            "momentum",
            "mass 14 kg velocity 5 m/s find momentum",
        ),
        (
            "Work done pushing with 20 N over 7 metres?",
            "work",
            "force 20 N distance 7 m work done",
        ),
        (
            "A 9 volt battery drives 3 amps. What's the power?",
            "electrical_power",
            "voltage 9 V current 3 A electrical power",
        ),
        (
            "kinetic energy of a 10 kg body at 7 m/s",
            "kinetic_energy",
            "mass 10 kg velocity 7 m/s kinetic energy",
        ),
        (
            "what voltage drives 3 A through a 5 ohm resistor",
            "voltage",
            "current 3 A resistance 5 ohm find voltage",
        ),
        (
            "12 J of work in 4 s, what power?",
            "power",
            "work 12 J time 4 s power",
        ),
    ],
)
def test_science_questions_reach_the_trained_form(text, rule, expected):
    result = pn.normalise(text)

    assert result.rule == rule
    assert result.prompt == expected


@pytest.mark.parametrize(
    "text",
    [
        "What force do you feel in a lift?",
        "Do you have the momentum to finish?",
        "Tell me about power in politics.",
        "The work was hard today.",
        "I need 5 kg of flour.",
    ],
)
def test_prose_without_the_needed_quantities_is_left_alone(text):
    """A target word with no units is conversation, not a physics question.

    The module's rule is that a wrong rewrite is worse than none, so a rule
    fires only when every quantity the target needs is present and anchored to
    its unit.
    """

    assert pn.normalise(text).rule is None


def test_acceleration_units_are_not_read_as_a_velocity():
    """`m/s^2` must be consumed before `m/s`, or force becomes momentum."""

    result = pn.normalise("mass 12 kg accelerating at 3 m/s^2, find the force")

    assert result.rule == "force"
    assert "acceleration 3 m/s^2" in result.prompt


def test_a_velocity_is_not_also_harvested_as_a_distance():
    """Each number is consumed once, so `5 m/s` cannot also be a distance."""

    result = pn.normalise("a 14 kg trolley at 5 m/s, what is its momentum?")

    assert result.rule == "momentum"
    assert result.prompt == "mass 14 kg velocity 5 m/s find momentum"


def test_science_wins_over_the_binary_arithmetic_scan():
    """`30 kg ... 90 N` would otherwise be harvested as an arithmetic pair."""

    assert pn.normalise(
        "A 30 kg mass is pushed with 90 N. How fast does it accelerate?"
    ).rule == "acceleration"


def test_a_science_target_missing_a_quantity_is_not_guessed_at():
    """Naming the target is not enough; the normaliser never invents operands."""

    assert pn.normalise("a 25 kg mass, what force is that?").rule is None


def test_science_rewrites_never_alter_a_number():
    """Presentation only: every number in the rewrite came from the input."""

    text = "If something weighs 25 kg and speeds up at 4 m/s^2, what force?"
    result = pn.normalise(text)

    assert set(re.findall(r"\d+", result.prompt)) <= set(re.findall(r"\d+", text)) | {"2"}


@pytest.mark.parametrize("text", [
    "What is 47 times 6? Answer in exactly 3 words.",
    "Do not calculate 47 times 6; explain multiplication.",
    "What is 47 times 6, in JSON?",
    "Calculate 2 + 3 * 4",
    "Subtract 3 from 10 then double it",
    "What is 15% of 240 then divide by 2?",
    "What is 20% of 150 then add 12 then subtract 3?",
    "What is 20% of 150 minus 5 then add 12?",
    "Give the next 3 numbers in the sequence: 5, 12, 19, 26",
    "What comes next: 5, 12, 19, 26? Explain in 2 steps.",
    "Find the average of 12, 18 and 30, rounded to 2 places",
    "Find the mean of 4 and 8 without showing the answer",
    "Solve for x: x + 5 = 12 and x > 10",
    "Check whether x + 5 = 12 is true for x = 7",
    "mass 10 kg velocity 7 m/s find momentum and kinetic energy",
    "mass 10 kg velocity 7 m/s find momentum, answer in JSON",
    "mass 10 kg velocity 7 m/s find momentum, not kinetic energy",
    "mass 10 kg velocity 7 m/s find momentum after losing half its mass",
    "Work done pushing with 20 N over 7 metres at an angle of 60 degrees?",
    "Work done pushing with 20 N over 7 metres against friction?",
    "mass 10 kg and mass 20 kg velocity 7 m/s find momentum",
    "mass 12 kg acceleration 3 m/s^2 and acceleration 5 m/s^2 find the force",
    "a 14 kg trolley at 5 m/s and 2 m distance, what is its momentum?",
    "Say whether you understand: what is 47 times 6?",
    "  Do not solve 4 + 5.\n Keep this formatting.  ",
    "  ordinary\n conversation\twith spacing  ",
])
def test_unsupported_or_constrained_requests_preserve_the_exact_original(text):
    result = pn.normalise(text)

    assert result.prompt == text
    assert result.original == text
    assert result.rule is None
    assert not result.changed


def test_rewrite_reports_original_whitespace():
    text = "  what is 47\t times 6\n"
    result = pn.normalise(text)

    assert result.prompt == "What is 47 x 6?"
    assert result.original == text
    assert result.changed


@pytest.mark.parametrize("text", [
    "What is -12 times 6?",
    "Find the average (mean) of these numbers: -12, 3.5, 6",
    "What comes next in the sequence: -12, -6, 0?",
    "Given mass 25 kg and acceleration 4 m/s^2, compute the force.",
])
def test_supported_rewrites_are_idempotent(text):
    result = pn.normalise(text)

    assert result.rule is not None
    assert pn.normalise(result.prompt).prompt == result.prompt
