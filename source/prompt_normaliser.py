"""Rewrite a naturally-typed question into the shape v74 was trained on.

v74 scores **0.894** on the n=500 benchmark and produced nonsense for every
naturally-phrased question typed into the chat interface. Both are true, and
the reason is that the benchmark generates prompts in the corpus's own format
while a person does not:

    "What is 47 x 6?"      -> 40 x 6 = 240, 7 x 6 = 42, total 282   correct
    "what is 47 times 6"   -> 400 x 6 = 200, 7 x 6 = 42, total 242  wrong

Probing which features actually matter, rather than assuming:

| feature                          | matters |
|----------------------------------|---------|
| operator token (`x` vs `times`)  | **yes** |
| a lead-in phrase being present   | **yes** |
| capitalisation                   | no      |
| trailing question mark           | no      |

`"47 x 6"` with no lead-in was read as algebra ("subtract 6 from both sides"),
so the lead-in is doing real work: it selects the task, not just the register.

## What this is and is not

It is a **presentation** fix. It maps how a person writes an operation onto the
token the model was trained on. It does not compute anything, it never alters a
number, and it never invents operands -- if it cannot recognise the shape it
returns the text untouched so ordinary conversation still reaches the model.

It does **not** make the model more capable, and it must not be described as
doing so. A question the model gets wrong in the training format stays wrong
here: `"What is 15% of 240?"` returns 26.0 (should be 36) both before and
after normalisation, because `percent` genuinely scores 0.75.

The rewrite is reported to the caller so the interface can show what was
actually asked. Silently changing someone's question and presenting the answer
as a reply to what they typed would misrepresent the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

#: Lead-ins drawn verbatim from the corpus, one per arithmetic task. The model
#: accepts any of them for any task, but using each task's own lead-in keeps the
#: rewritten prompt inside the distribution it was trained on.
LEAD_IN = {
    "multiplication": "What is {a} x {b}?",
    "division": "Quick question: {a} / {b}",
    "addition": "Please help with this. {a} + {b}",
    "subtraction": "Solve this basic math problem: {a} - {b}",
}

NUMBER = r"-?\d+(?:\.\d+)?"

#: Terse labelled forms taken verbatim from `build_omni_corpus`, one per science
#: task. Each corpus task carries four or five phrasings; these are the ones that
#: name every quantity explicitly, so a rewrite cannot be misread as a different
#: task once the units are stripped.
SCIENCE_LEAD_IN = {
    "force": "Given mass {m} kg and acceleration {a} m/s^2, compute the force.",
    "acceleration": "force {f} N mass {m} kg find acceleration",
    "momentum": "mass {m} kg velocity {v} m/s find momentum",
    "kinetic_energy": "mass {m} kg velocity {v} m/s kinetic energy",
    "work": "force {f} N distance {d} m work done",
    "electrical_power": "voltage {u} V current {i} A electrical power",
    "voltage": "current {i} A resistance {r} ohm find voltage",
    "power": "work {w} J time {t} s power",
}

#: Quantity patterns, ordered so the more specific unit wins. `m/s^2` must be
#: tried before `m/s`, and both before a bare `m`, or an acceleration is read as
#: a velocity and a velocity as a distance.
QUANTITY_PATTERNS = (
    ("a", rf"({NUMBER})\s*(?:m\s*/\s*s\s*(?:\^|\*\*)?\s*2|m/s²|"
          rf"met(?:re|er)s?\s+per\s+second\s+squared)"),
    ("v", rf"({NUMBER})\s*(?:m\s*/\s*s(?![\^²0-9])|met(?:re|er)s?\s+per\s+second(?!\s+squared))"),
    ("m", rf"({NUMBER})\s*(?:kg\b|kilogram(?:me)?s?\b)"),
    ("f", rf"({NUMBER})\s*(?:N\b|newtons?\b)"),
    ("u", rf"({NUMBER})\s*(?:V\b|volts?\b)"),
    ("i", rf"({NUMBER})\s*(?:A\b|amp(?:ere)?s?\b)"),
    ("r", rf"({NUMBER})\s*(?:ohms?\b|Ω)"),
    ("w", rf"({NUMBER})\s*(?:J\b|joules?\b)"),
    ("t", rf"({NUMBER})\s*(?:s\b|seconds?\b)"),
    ("d", rf"({NUMBER})\s*(?:m\b|met(?:re|er)s?\b)"),
)

#: What each task asks for, and what it needs to be answerable. A target whose
#: quantities are not all present is left alone rather than guessed at.
SCIENCE_TARGETS = (
    ("kinetic_energy", r"kinetic\s+energ", ("m", "v")),
    ("electrical_power", r"(?:electrical\s+power|power)", ("u", "i")),
    ("power", r"power", ("w", "t")),
    ("voltage", r"(?:voltage|potential\s+difference)", ("i", "r")),
    ("momentum", r"momentum", ("m", "v")),
    ("acceleration", r"accelerat", ("f", "m")),
    ("work", r"work", ("f", "d")),
    ("force", r"force", ("m", "a")),
)


@dataclass(frozen=True)
class Normalised:
    """The prompt to send, and an honest record of what was done to it."""

    prompt: str
    rule: Optional[str] = None
    original: Optional[str] = None

    @property
    def changed(self) -> bool:
        return self.rule is not None and self.prompt != self.original


def _numbers(text: str) -> List[str]:
    return re.findall(NUMBER, text)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _binary(text: str) -> Optional[Normalised]:
    """`A <op> B` in any of the ways people write it."""

    # "subtract A from B" names its operands in the opposite order to "B - A",
    # so it cannot be handled by the symmetric `A op B` scan below.
    reversed_subtraction = re.search(
        rf"subtract(?:ing)?\s+({NUMBER})\s+from\s+({NUMBER})", text, flags=re.IGNORECASE
    )
    if reversed_subtraction:
        return Normalised(
            LEAD_IN["subtraction"].format(
                a=reversed_subtraction.group(2), b=reversed_subtraction.group(1)
            ),
            "subtraction",
        )

    operators = [
        ("multiplication", r"(?:x|\*|times|multiplied\s+by)"),
        ("division", r"(?:/|÷|divided\s+by|over)"),
        ("addition", r"(?:\+|plus|added\s+to)"),
        ("subtraction", r"(?:-|minus|take\s+away|less)"),
    ]
    for task, pattern in operators:
        # Require the operator to be delimited so "6 - 2" matches but the
        # minus sign inside "-12" does not.
        match = re.search(
            rf"({NUMBER})\s*(?:{pattern})\s*({NUMBER})",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        a, b = match.group(1), match.group(2)
        return Normalised(LEAD_IN[task].format(a=a, b=b), task)
    return None


def _quantities(text: str) -> dict:
    """Every quantity the text names by its unit, keyed by symbol.

    A unit is consumed once matched, so a single number cannot be read as two
    different quantities. `"7 m/s"` is a velocity and is then unavailable as a
    distance, which is what stops `m/s` being harvested twice.
    """

    found: dict = {}
    remaining = text
    for symbol, pattern in QUANTITY_PATTERNS:
        match = re.search(pattern, remaining, flags=re.IGNORECASE)
        if match:
            found[symbol] = match.group(1)
            remaining = remaining[: match.start()] + " " + remaining[match.end():]
    return found


def _science(text: str) -> Optional[Normalised]:
    """Rewrite a physics question into the terse labelled corpus form.

    Deliberately conservative, for the reason the module docstring gives: a
    wrong rewrite is worse than none. A rule fires only when the text names the
    target *and* every quantity that target needs, each anchored to its unit.
    "What force do you feel in a lift?" names a target and no quantities, so it
    goes through untouched to ordinary conversation.
    """

    lowered = text.lower()
    quantities = _quantities(text)
    if not quantities:
        return None
    for task, target_pattern, required in SCIENCE_TARGETS:
        if not re.search(target_pattern, lowered):
            continue
        if not all(symbol in quantities for symbol in required):
            continue
        values = {symbol: quantities[symbol] for symbol in required}
        return Normalised(SCIENCE_LEAD_IN[task].format(**values), task)
    return None


def normalise(text: str) -> Normalised:
    """Rewrite `text` into the corpus format, or return it unchanged.

    Rules are ordered most-specific first: a two-step question contains a
    percent question, and a percent question contains numbers that would
    otherwise look like an average.
    """

    if not text or not text.strip():
        return Normalised(text, None, text)
    source = _clean(text)
    lowered = source.lower()

    # Two-step: a percentage followed by a further operation.
    two_step = re.search(
        rf"({NUMBER})\s*(?:%|percent)\s*(?:of)?\s*({NUMBER}).*?"
        rf"then\s*(add|subtract|plus|minus)\s*({NUMBER})",
        lowered,
    )
    if two_step:
        percent, whole, operation, operand = two_step.groups()
        word = "add" if operation in ("add", "plus") else "subtract"
        return Normalised(
            f"What is {percent}% of {whole}, then {word} {operand}?",
            "two_step",
            source,
        )

    percent = re.search(
        rf"({NUMBER})\s*(?:%|percent)\s*(?:of)\s*({NUMBER})", lowered
    )
    if percent:
        return Normalised(
            f"What is {percent.group(1)}% of {percent.group(2)}?", "percent", source
        )

    if re.search(r"\b(?:average|mean)\b", lowered):
        values = _numbers(source)
        if len(values) >= 2:
            joined = ", ".join(values)
            return Normalised(
                f"Find the average (mean) of these numbers: {joined}",
                "average",
                source,
            )

    if re.search(r"\b(?:next|sequence|comes\s+after|continue)\b", lowered):
        values = _numbers(source)
        if len(values) >= 3:
            joined = ", ".join(values)
            return Normalised(
                f"What comes next in the sequence: {joined}?", "sequence", source
            )

    # Algebra is already written the way the corpus writes it, and rewriting an
    # equation risks reordering its sides. Only the lead-in is normalised.
    algebra = re.search(
        rf"x\s*([+\-*/])\s*({NUMBER})\s*=\s*({NUMBER})", lowered
    )
    if algebra:
        operator, operand, result = algebra.groups()
        return Normalised(
            f"Solve for x: x {operator} {operand} = {result}", "algebra_one_step", source
        )

    # Science before the binary scan. "A 30 kg mass is pushed with 90 N" would
    # otherwise be harvested as an arithmetic pair by the `A op B` search.
    science = _science(source)
    if science is not None:
        return Normalised(science.prompt, science.rule, source)

    binary = _binary(source)
    if binary is not None:
        return Normalised(binary.prompt, binary.rule, source)

    # A word problem, or ordinary conversation. Both go through untouched: the
    # corpus's word problems are written in plain prose already, and rewriting
    # conversation would be pure damage.
    return Normalised(source, None, source)
