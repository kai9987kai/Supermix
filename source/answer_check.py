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


# -- science shapes (v81) ---------------------------------------------------
#
# v80 answers physics correctly and the interface said NOT CHECKED for every
# one of them, because these shapes were never taught to the checker. A model
# whose strongest new capability cannot be verified live is the same gap v76
# closed for multiplication, one domain over.
#
# Each reads the quantities by name and unit, so it matches the corpus's
# phrasings without depending on any single one of them.

def _quantity(question: str, names: str, unit: str) -> Optional[float]:
    """Read `<name> <number> <unit>` in either order, as the corpus writes it."""

    number = r"(-?\d+(?:\.\d+)?)"
    # Both names and unit are alternations, so both must be grouped. Left
    # bare, `(\d+)\s*m|metres?` parses as `(\d+)\s*m` OR `metres?` -- the
    # second branch has no capture group, and match.group(1) is then None.
    for pattern in (rf"(?:{names})\D{{0,24}}?{number}\s*(?:{unit})\b",
                    rf"{number}\s*(?:{unit})\b\D{{0,24}}?(?:{names})"):
        match = re.search(pattern, question, re.I)
        if match:
            return float(match.group(1))
    return None


#: (task, words identifying the target, quantity A, quantity B)
#: Division tasks are named in `_DIVISION_LAWS`; everything else multiplies.
#: Units carry their spelled-out forms. The corpus writes "57 volts and 5
#: amps" as readily as "57 V, 5 A", and a checker that only knew the symbols
#: reported NOT CHECKED for a third of the questions the model answers.
_MASS = r"kg|kilograms?"
_FORCE = r"N|newtons?"
_ANY = r"[a-z/^]*"

_PRODUCT_LAWS = (
    ("force", r"force",
     (r"mass|body|block|object", _MASS), (r"accelerat\w*", r"m/s\^?2")),
    ("momentum", r"momentum",
     (r"mass|object|body", _MASS), (r"velocity|speed|moves|travelling|at", r"m/s")),
    ("work", r"work",
     (r"force", _FORCE), (r"distance|moves|through|over|acts", r"m|metres?")),
    ("voltage", r"voltage|potential difference",
     (r"current|flows|carrying|drives", r"A|amps?|amperes?"),
     (r"resistance|ohm|through|across|resistor", r"ohms?")),
    ("electrical_power", r"electrical power|power dissipated|power|used at",
     (r"voltage|volts?|runs at|at", r"V|volts?"),
     (r"current|drawing|amps?|and", r"A|amps?|amperes?")),
    ("wave_speed", r"wave speed|speed of|its speed|speed at",
     (r"frequency|at", r"Hz|hertz"), (r"wavelength|with", r"m|metres?")),
    ("acceleration", r"acceleration|accelerat\w*",
     (r"force|results from|from", _FORCE), (r"mass|body|object|on", _MASS)),
    ("power", r"power",
     (r"work|delivered|done", r"J|joules?"), (r"time|in|over", r"s|seconds?")),
    ("molarity", r"molarity|concentration|molar",
     (r"mol|moles|solute|of", r"mol|moles"),
     (r"volume|litres?|liters?|dissolved|in", r"L|litres?|liters?")),
)


_DIVISION_LAWS = frozenset({"acceleration", "power", "molarity"})


def _science(question: str) -> Optional[Tuple[str, float]]:
    for task, target, (a_names, a_unit), (b_names, b_unit) in _PRODUCT_LAWS:
        if not re.search(target, question, re.I):
            continue
        a = _quantity(question, a_names, a_unit)
        b = _quantity(question, b_names, b_unit)
        if a is None or b is None:
            continue
        if task in _DIVISION_LAWS:
            if b == 0:
                return None   # not checkable rather than an exception
            return (task, a / b)
        return (task, a * b)
    return None


def _kinetic_energy(question: str) -> Optional[Tuple[str, float]]:
    if not re.search(r"kinetic energy", question, re.I):
        return None
    mass = _quantity(question, r"mass|body", r"kg")
    velocity = _quantity(question, r"velocity|speed|moves|at", r"m/s")
    if mass is None or velocity is None:
        return None
    return ("kinetic_energy", 0.5 * mass * velocity * velocity)


def _combination_choose(question: str) -> Optional[Tuple[str, float]]:
    """`n choose k`, however the corpus words it.

    The corpus fixes k at 2 so the working can be shown, but this reads
    whatever k is stated rather than assuming it -- an assumption here would
    produce a confident wrong verdict on any other k.
    """

    if not re.search(r"combination|choose|chosen|taken", question, re.I):
        return None
    number = r"(\d+)"
    for pattern in (rf"{number}\s*choose\s*{number}",
                    rf"n\s*=\s*{number}\s*k\s*=\s*{number}",
                    rf"of\s*{number}\s*things taken\s*{number}",
                    rf"can\s*{number}\s*items? be chosen from\s*{number}"):
        match = re.search(pattern, question, re.I)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        n, k = (b, a) if a < b else (a, b)   # "2 chosen from 30" reverses them
        if k > n:
            return None
        import math as _math

        return ("combination", float(_math.comb(n, k)))
    return None


def _arithmetic_series(question: str) -> Optional[Tuple[str, float]]:
    """Sum of the first n terms of an arithmetic progression."""

    if not re.search(r"arithmetic (?:series|progression)", question, re.I):
        return None
    first = re.search(r"first term\s*(?:is\s*)?(-?\d+)", question, re.I)
    difference = re.search(r"(?:common )?difference\s*(?:of\s*)?(-?\d+)", question, re.I)
    terms = re.search(r"(?:sum of|first)\s*(\d+)\s*terms|(?:\bn\s*(\d+))", question, re.I)
    if not (first and difference and terms):
        return None
    count = int(terms.group(1) or terms.group(2))
    a, d = int(first.group(1)), int(difference.group(1))
    last = a + (count - 1) * d
    return ("arithmetic_series", float(count * (a + last) / 2))


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
#: * `_kinetic_energy` and `_science` precede the bare-number parsers, because
#:   a physics question carries two numbers and a naive `a x b` search would
#:   seize on them without knowing which law applies. `_kinetic_energy` runs
#:   first of the two: it names a mass and a velocity, which is also what
#:   `momentum` matches on.
PARSERS: Tuple[Callable[[str], Optional[Tuple[str, float]]], ...] = (
    _word_problem,
    _combination_choose,
    _arithmetic_series,
    _kinetic_energy,
    _science,
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
