"""Long division, and the property `decompose_quotient` did not have.

v87 split the three division tasks' quotients by place value on the strength of
a real measurement -- a written step producing one significant place scores
0.825 against 0.075 for three -- and made every task using it much worse:
`power` 0.333 -> 0.048. The steps were easier and unreachable, because each
partial dividend was back-computed from the answer:

    model:  6400 / 64 = 100,  420 / 64 = 5,  122 / 64 = 2
    truth:  6400 / 64 = 100, 1920 / 64 = 30, 192 / 64 = 3

To write `1920` the model had to already know the quotient digit was 30.

So the rule that failure established, and that these tests enforce:

> A decomposition helps only when every step it adds is **derivable forward**
> from what is already written, and **inside the model's arithmetic**.

`test_every_operand_is_already_on_the_page` is the first half, mechanically
checked. The envelope and budget tests are the second.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import build_omni_corpus as omni  # noqa: E402

DIVISION_TASKS = ("power", "acceleration", "molarity")
STEP = re.compile(r"(\d+) ([-x]) (\d+) = (-?\d+)")
INTO = re.compile(r"(\d+) into (\d+) = (\d+)")


@pytest.fixture(autouse=True)
def _restore():
    was_long, was_phrasing = omni.LONG_DIVISION, omni.NATURAL_PHRASINGS
    yield
    omni.LONG_DIVISION, omni.NATURAL_PHRASINGS = was_long, was_phrasing


def test_the_flag_is_off_by_default():
    """It is a measured proposal, not a result. Nothing has trained on it."""

    assert omni.LONG_DIVISION is False


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dividend, divisor", [
    (5712, 48), (255, 15), (8178, 47), (690, 30), (260, 20),
    (23800, 100), (144, 12), (1000, 8), (9999, 99),
])
def test_the_trace_reaches_the_right_quotient(dividend, divisor):
    trace = omni.long_division(dividend, divisor)
    assert trace.endswith(f"total {dividend // divisor}")


def test_every_written_step_is_arithmetically_true():
    rng = random.Random(5)
    for _ in range(400):
        divisor = rng.randint(2, 199)
        dividend = divisor * rng.randint(2, 300)
        trace = omni.long_division(dividend, divisor)
        for left, operator, right, stated in STEP.findall(trace):
            truth = (int(left) - int(right) if operator == "-"
                     else int(left) * int(right))
            assert int(stated) == truth, f"{left} {operator} {right} in {trace}"
        for divides, into, digit in INTO.findall(trace):
            assert int(into) // int(divides) == int(digit), trace


# ---------------------------------------------------------------------------
# Derivable forward -- the property decompose_quotient lacked
# ---------------------------------------------------------------------------


def test_every_operand_is_already_on_the_page():
    """No step may name a number the model could not have produced yet.

    Legal sources for an operand, in order: the divisor, a digit-extension of
    the dividend's leading digits, a result stated by an earlier step, or that
    result with the next dividend digit appended (the bring-down). Anything else
    has to come from knowing the answer, which is what killed the last attempt.
    """

    rng = random.Random(9)
    for _ in range(300):
        divisor = rng.randint(2, 199)
        quotient = rng.randint(2, 300)
        dividend = divisor * quotient
        trace = omni.long_division(dividend, divisor)

        known = {divisor, dividend}
        # Every prefix of the dividend is readable off the question.
        text = str(dividend)
        for end in range(1, len(text) + 1):
            known.add(int(text[:end]))

        def bring_down(value: int) -> None:
            """A remainder, and that remainder with any next digit appended."""

            known.add(value)
            for digit in range(10):
                known.add(value * 10 + digit)

        # A step's remainder is written only when it is non-zero and the digit
        # is non-zero. In every other case a reader derives it from the line
        # above -- `carry - digit x divisor` -- so `pending` carries it forward
        # until either a subtraction states it or the next step needs it.
        pending = None

        def settle() -> None:
            nonlocal pending
            if pending is not None:
                carry, digit = pending
                bring_down(carry - digit * divisor)
                pending = None

        for piece in trace.split(", "):
            into = INTO.fullmatch(piece)
            if into:
                divides, target, digit = (int(x) for x in into.groups())
                settle()
                assert divides in known, f"{divides} unavailable in {trace}"
                assert target in known, f"{target} unavailable in {trace}"
                known.add(digit)
                pending = (target, digit)
                continue
            step = STEP.fullmatch(piece)
            if not step:
                continue                      # the closing `total N`
            left, operator, right, stated = step.groups()
            assert int(left) in known, f"{left} unavailable in {trace}"
            assert int(right) in known, f"{right} unavailable in {trace}"
            known.add(int(stated))
            if operator == "-":
                bring_down(int(stated))
                pending = None


def test_a_redundant_zero_remainder_is_not_written():
    """`432 - 432 = 0` restates what the line above already shows.

    Dropping it is worth about eight tokens, and it is what brings the longest
    turn back under the sequence budget: three of 18,000 sampled rows reached
    130 tokens with it, and turn-aligned packing discards those silently.
    """

    trace = omni.long_division(5712, 48)
    assert "432 - 432" not in trace
    assert trace.endswith("9 x 48 = 432, total 119")


# ---------------------------------------------------------------------------
# The envelope and the budget
# ---------------------------------------------------------------------------


def test_no_operand_leaves_the_learned_width():
    """Why `subtract multiples` was rejected and this was not.

    That format asks for `16992 - 11800`; `subtraction` trains on three digits.
    Long division's operands are bounded by ten times the divisor, so they stay
    within a digit of what the model has seen.
    """

    omni.LONG_DIVISION = True
    rng = random.Random(11)
    worst = 0
    for task in DIVISION_TASKS:
        for _ in range(300):
            problem = omni.TASKS[task](rng)
            for match in STEP.finditer(problem.response):
                worst = max(worst, len(match.group(1)), len(match.group(3)))
    assert worst <= 4, f"an operand reached {worst} digits"


def test_no_turn_leaves_the_sequence_budget():
    text_utils = pytest.importorskip("mimomix_text")
    omni.LONG_DIVISION = True
    omni.NATURAL_PHRASINGS = True
    rng = random.Random(3)
    rows = [omni.TASKS[t](rng) for t in DIVISION_TASKS for _ in range(700)]
    tokenizer = text_utils.WordTokenizer.build(
        (f for p in rows for f in (p.prompt, p.response)),
        max_vocab=16384, digit_tokens=True)
    lengths = [len(tokenizer.encode(p.prompt)) + len(tokenizer.encode(p.response))
               for p in rows]
    assert max(lengths) < omni.DEFAULT_SEQUENCE_LENGTH, (
        f"longest turn is {max(lengths)} tokens; packing drops these"
    )


def test_the_solver_still_verifies_every_row():
    """A longer working is still a checked working."""

    omni.LONG_DIVISION = True
    rng = random.Random(7)
    for task in DIVISION_TASKS:
        for _ in range(60):
            assert omni.verify(omni.TASKS[task](rng)), task


def test_the_answer_is_still_the_last_number():
    omni.LONG_DIVISION = True
    rng = random.Random(13)
    for task in DIVISION_TASKS:
        for _ in range(60):
            problem = omni.TASKS[task](rng)
            assert problem.response.rstrip().split()[-1] == str(
                int(problem.answer)), problem.response
