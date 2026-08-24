"""Independently check a maths reply against the question that was asked.

The session's central finding is that only a *checkable* answer resists
recitation: a model reproducing a remembered reply to a novel problem is simply
wrong, and no amount of fluency hides it. `eval_problem_solving.py` applies that
offline, over problems it generates itself.

This module applies it live, to whatever the user typed. It re-derives the truth
from the question and compares, so the interface can say "wrong, the answer is
905" instead of presenting confident arithmetic and leaving the reader to check.

It recognises only the shapes the models were trained on -- nine of them, one
per task in the v74 corpus. Anything else returns `None`, which the interface
must render as *not checked* rather than as correct: an unrecognised question
is not a passed one.

Multiplication, division, sequence and two-step were added for v74, which
introduced those tasks. Before that a chat reply to "What is 47 x 6?" showed
NOT CHECKED -- the model's strongest tasks were the ones nothing verified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

#: Matches the tolerance `eval_problem_solving.is_correct` uses, so the live
#: check and the benchmark cannot disagree about the same answer.
TOLERANCE = 1e-6

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass
class Check:
    """The verdict on one reply."""

    task: str
    expected: float
    predicted: Optional[float]
    correct: bool

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "expected": self.expected,
            "predicted": self.predicted,
            "correct": self.correct,
        }


def extract_answer(text: str) -> Optional[float]:
    """The reply's answer is the last number it produces.

    Identical rule to `eval_problem_solving.extract_answer`; the scratchpad
    formats all end with the result, so "600 + 200 = 800, 17 + 88 = 105, total
    905" reads as 905 rather than 600.
    """

    matches = _NUMBER.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1].rstrip(".")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            return float(numerator) / float(denominator)
        return float(raw)
    except (ValueError, ZeroDivisionError):
        return None


# -- question parsers -------------------------------------------------------
#
# Each returns (task, expected) or None. They are deliberately narrow: a loose
# pattern that half-matched an unrelated question would produce a confident
# wrong verdict, which is worse than no verdict at all.


def _binary(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"(-?\d+)\s*([+-])\s*(-?\d+)", question)
    if not match or "=" in question:
        return None
    left, op, right = int(match.group(1)), match.group(2), int(match.group(3))
    return ("arithmetic", float(left + right if op == "+" else left - right))


def _percent(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", question, re.I)
    if not match:
        return None
    return ("percent", float(match.group(1)) * float(match.group(2)) / 100.0)


def _algebra(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"x\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)", question, re.I)
    if not match:
        return None
    op, constant, right = match.group(1), int(match.group(2)), int(match.group(3))
    return ("algebra_one_step", float(right - constant if op == "+" else right + constant))


def _average(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"\b(average|mean)\b", question, re.I):
        return None
    tail = question.split(":", 1)[-1]
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", tail)]
    if len(values) < 2:
        return None
    return ("average", sum(values) / len(values))


def _multiplication(question: str) -> Optional[Tuple[str, float]]:
    """`A x B`, the corpus's multiplication form.

    Must run after `_algebra`: `x` is this corpus's multiplication sign *and*
    its unknown, and only the digit on the left tells them apart.
    """

    match = re.search(r"(-?\d+(?:\.\d+)?)\s*[x*]\s*(-?\d+(?:\.\d+)?)", question, re.I)
    if not match or "=" in question:
        return None
    return ("multiplication", float(match.group(1)) * float(match.group(2)))


def _division(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(r"(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)", question)
    if not match or "=" in question:
        return None
    divisor = float(match.group(2))
    if divisor == 0:
        # Not checkable rather than an exception; the question has no answer.
        return None
    return ("division", float(match.group(1)) / divisor)


def _two_step(question: str) -> Optional[Tuple[str, float]]:
    """`P% of N, then add/subtract M`. Must precede `_percent`, which it contains."""

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)\s*,?\s*then\s*(add|subtract)\s*(-?\d+(?:\.\d+)?)",
        question,
        re.I,
    )
    if not match:
        return None
    percent, whole, operation, operand = match.groups()
    base = float(percent) * float(whole) / 100.0
    delta = float(operand)
    return ("two_step", base + delta if operation.lower() == "add" else base - delta)


def _sequence(question: str) -> Optional[Tuple[str, float]]:
    """The next term of an arithmetic progression.

    Returns None when the differences are not constant. The corpus only
    contains arithmetic progressions, so anything else is a question this
    cannot verify -- and reporting "not checked" is correct where guessing a
    rule would silently invent a right answer.
    """

    if not re.search(r"\b(next|sequence)\b", question, re.I):
        return None
    tail = question.split(":", 1)[-1]
    values = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", tail)]
    if len(values) < 3:
        return None
    steps = {round(b - a, 9) for a, b in zip(values, values[1:])}
    if len(steps) != 1:
        return None
    return ("sequence", values[-1] + steps.pop())


def _word_problem(question: str) -> Optional[Tuple[str, float]]:
    match = re.search(
        r"has\s+(\d+).*?get\s+(\d+)\s+more.*?give\s+away\s+(\d+)", question, re.I | re.S
    )
    if not match:
        return None
    start, gain, lose = (int(match.group(i)) for i in (1, 2, 3))
    return ("word_problem", float(start + gain - lose))


#: Order matters, and every entry below is placed against a specific ambiguity:
#:
#: * `_word_problem` and `_average` precede everything numeric, because both
#:   contain bare numbers a naive "a + b" search would seize on.
#: * `_sequence` precedes them too -- "7, 17, 27, 37" is a comma-separated list
#:   of numbers, which is exactly what an average looks like.
#: * `_two_step` precedes `_percent` because it *contains* a percent question.
#: * `_algebra` precedes `_multiplication` because `x` is both this corpus's
#:   multiplication sign and its unknown.
PARSERS: Tuple[Callable[[str], Optional[Tuple[str, float]]], ...] = (
    _word_problem,
    _sequence,
    _average,
    _two_step,
    _algebra,
    _percent,
    _division,
    _multiplication,
    _binary,
)


def parse_question(question: str) -> Optional[Tuple[str, float]]:
    for parser in PARSERS:
        result = parser(question)
        if result is not None:
            return result
    return None


def check(question: str, reply: str) -> Optional[Check]:
    """Verify a reply, or return ``None`` when the question is not checkable.

    ``None`` means *not checked*. The caller must not render it as correct: the
    whole value of this is that a wrong answer is visibly wrong, and quietly
    passing anything unrecognised would destroy that.
    """

    parsed = parse_question(question)
    if parsed is None:
        return None
    task, expected = parsed
    predicted = extract_answer(reply)
    correct = (
        predicted is not None
        and abs(predicted - expected) <= max(TOLERANCE, abs(expected) * 1e-6)
    )
    return Check(task=task, expected=expected, predicted=predicted, correct=correct)


def supported_shapes() -> List[str]:
    """The question forms this can verify, for the interface to advertise."""

    return [
        "Solve this basic math problem: 617 + 288",
        "What is 25% of 840?",
        "Solve for x: x + 14 = 39",
        "A student has 45 marbles. They get 38 more and then give away 27. How many marbles do they have now?",
        "Find the average (mean) of these numbers: 40, 60, 20, 80",
        "What is 25 x 7?",
        "Quick question: 70 / 5",
        "What comes next in the sequence: 7, 17, 27, 37?",
        "What is 50% of 698, then add 28?",
    ]
