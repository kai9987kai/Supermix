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

    binary = _binary(source)
    if binary is not None:
        return Normalised(binary.prompt, binary.rule, source)

    # A word problem, or ordinary conversation. Both go through untouched: the
    # corpus's word problems are written in plain prose already, and rewriting
    # conversation would be pure damage.
    return Normalised(source, None, source)
