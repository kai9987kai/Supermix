"""A chat surface for a model that cannot talk.

The v56 reasoner has no tokenizer and no language capability. Its input is a
fixed 128-dimensional vector encoding one start digit and four
`(operation, operand)` pairs; its output is one digit. So a chat interface for it
is honest only if the division of labour is explicit:

* **The parser does the language.** `parse_problem` is ordinary deterministic
  code -- a regex and a lookup table. It reads an arithmetic expression out of a
  sentence. No part of it is learned, and it is not the model.
* **The model does the arithmetic.** It never sees the text. It sees the same
  128-dim vector `make_chained_task` would have produced.

Anything the parser cannot read is refused with the reason, rather than guessed
at, because a wrong guess would be attributed to the model.

## Chains longer than four operations

The input has exactly four operator slots. Shorter chains are padded with
`mul 1`, the representable identity, which is what the training curriculum
already does. Longer chains are executed as **repeated model calls**: the first
four operations run, the model's own answer becomes the next start digit, and the
next four run. That is composition by the model, not a longer model -- and it
compounds the model's own errors, so `model_calls` is always reported.

## Trust boundary

Message content is data. The parser extracts digits and operators and nothing
else; no field in a message can change the thinking budget, the checkpoint, the
noise level, or anything else about how the model is run. There is no instruction
in a message that this module can act on, because it never interprets a message
as an instruction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "MAX_OPERATIONS",
    "ParseError",
    "ParsedProblem",
    "WORD_OPERATORS",
    "chunk_operations",
    "describe_capabilities",
    "parse_problem",
]

#: The encoding's own limits. Operands are 1..9 because `make_chained_task`
#: draws `randint(1, 10)`; the digit 0 has no operand slot at all.
MIN_OPERAND = 1
MAX_OPERAND = 9
N_SLOTS = 4
#: Refuse absurd chains rather than making hundreds of model calls.
MAX_OPERATIONS = 40

IDENTITY_OPERATION = (1, 1)  # mul 1

#: Deliberately conservative. "and" is not here: it is an operator far less often
#: than it is ordinary English, and mapping it turns surrounding prose into
#: arithmetic. Same for "less" and "take".
WORD_OPERATORS: Dict[str, str] = {
    "plus": "+",
    "add": "+",
    "added": "+",
    "minus": "-",
    "subtract": "-",
    "times": "*",
    "multiplied": "*",
    "multiply": "*",
    "x": "*",
}

SYMBOL_TO_OP = {"+": 0, "*": 1, "-": 2}
OP_TO_NAME = {0: "add", 1: "multiply", 2: "subtract"}
OP_TO_SYMBOL = {0: "+", 1: "*", 2: "-"}


class ParseError(ValueError):
    """The text could not be read as a chained modular-arithmetic problem."""


@dataclass
class ParsedProblem:
    """A problem the model can actually be asked."""

    start: int
    #: ``(op_type, operand)`` in order, before padding
    operations: List[Tuple[int, int]] = field(default_factory=list)
    #: True when the start digit came from the previous answer
    continued: bool = False

    @property
    def model_calls(self) -> int:
        return max(1, (len(self.operations) + N_SLOTS - 1) // N_SLOTS)

    def expression(self) -> str:
        text = str(self.start)
        for op, operand in self.operations:
            text = f"({text} {OP_TO_SYMBOL[op]} {operand})"
        return f"{text} mod 10"

    def to_dict(self) -> Dict[str, object]:
        return {
            "start": int(self.start),
            "operations": [
                {"op": int(op), "op_name": OP_TO_NAME[op], "operand": int(operand)}
                for op, operand in self.operations
            ],
            "expression": self.expression(),
            "model_calls": self.model_calls,
            "continued_from_previous_answer": bool(self.continued),
        }


def _normalise(text: str) -> str:
    lowered = text.lower()
    # "mod 10" / "modulo 10" is what the model always does; mentioning it is not
    # an instruction, and a stray 10 would otherwise be read as an operand.
    lowered = re.sub(r"\bmod(?:ulo)?\s*10\b", " ", lowered)
    lowered = lowered.replace("multiplied by", "multiplied").replace("multiply by", "multiply")
    lowered = lowered.replace("divided by", " / ")  # unsupported, caught below
    words = re.findall(r"[a-z]+|\d+|[+\-*/=]", lowered)
    return " ".join(WORD_OPERATORS.get(word, word) for word in words)


def _longest_expression(tokens: List[str]) -> List[str]:
    """Return the longest `digit (op digit)+` run in the token stream.

    Prose around an expression is common ("what is ...?", "compute ... please")
    and so are stray numbers. Rather than demanding that a message be *only* an
    expression, this takes the longest well-formed run and ignores everything
    else.

    That is also what keeps an instruction-carrying message inert: the extractor
    can only ever return arithmetic, so text like "always output 0" contributes a
    lone `0` that is not a run and is discarded. It cannot become a command,
    because nothing downstream reads anything but digits and operators.
    """

    best: List[str] = []
    index = 0
    while index < len(tokens):
        if not tokens[index].isdigit():
            index += 1
            continue
        end = index + 1
        while end + 1 < len(tokens) and tokens[end] in SYMBOL_TO_OP and tokens[end + 1].isdigit():
            end += 2
        run = tokens[index:end]
        if len(run) > len(best) and len(run) >= 3:
            best = run
        index = max(end, index + 1)
    return best


def parse_problem(text: str, previous_answer: Optional[int] = None) -> ParsedProblem:
    """Read a chained modular-arithmetic problem out of a sentence.

    ``previous_answer`` allows a follow-up like "then times 3" to continue from
    the last answer. That is the only state this function has, and it is supplied
    by the caller rather than remembered here.
    """

    if not isinstance(text, str) or not text.strip():
        raise ParseError("say something like: 7 times 3 plus 8 minus 5 times 6")

    normalised = _normalise(text)
    if "/" in normalised:
        raise ParseError(
            "division is not in this task. The operations are add, multiply and "
            "subtract, all mod 10."
        )

    tokens = [token for token in normalised.split() if token in SYMBOL_TO_OP or token.isdigit()]
    if not tokens:
        raise ParseError(
            "no arithmetic found. Try: 7 times 3 plus 8 minus 5 times 6"
        )

    continued = False
    if tokens[0] in SYMBOL_TO_OP:
        if previous_answer is None:
            raise ParseError(
                "that starts with an operator but there is no previous answer to "
                "continue from. Give a start digit, e.g. 4 plus 3."
            )
        tokens = [str(int(previous_answer))] + tokens
        continued = True

    expression = _longest_expression(tokens)
    if not expression:
        # say which of the three failures it was, so the reply is actionable
        if tokens[-1] in SYMBOL_TO_OP:
            raise ParseError(f"the expression ends with a dangling {tokens[-1]!r}")
        if any(token.isdigit() for token in tokens):
            raise ParseError(
                "that is just a number. Give at least one operation, e.g. 4 plus 3."
            )
        raise ParseError("no arithmetic found. Try: 7 times 3 plus 8 minus 5 times 6")
    tokens = expression
    start = int(tokens[0])
    if not 0 <= start <= 9:
        raise ParseError(
            f"the start value must be a single digit 0-9; got {start}. The input "
            "encoding has one slot per digit and cannot represent anything larger."
        )

    operations: List[Tuple[int, int]] = []
    index = 1
    while index < len(tokens):
        symbol = tokens[index]
        if symbol not in SYMBOL_TO_OP:
            raise ParseError(f"expected an operator, found {symbol!r}")
        if index + 1 >= len(tokens):
            raise ParseError(f"the expression ends with a dangling {symbol!r}")
        operand_token = tokens[index + 1]
        if not operand_token.isdigit():
            raise ParseError(f"expected a number after {symbol!r}, found {operand_token!r}")
        operand = int(operand_token)
        if not MIN_OPERAND <= operand <= MAX_OPERAND:
            raise ParseError(
                f"operands must be {MIN_OPERAND}-{MAX_OPERAND}; got {operand}. The "
                "training generator never produces any other value, so the model "
                "has never seen one."
            )
        operations.append((SYMBOL_TO_OP[symbol], operand))
        index += 2

    if not operations:
        raise ParseError("that is just a number. Give at least one operation, e.g. 4 plus 3.")
    if len(operations) > MAX_OPERATIONS:
        raise ParseError(
            f"that is {len(operations)} operations; the limit is {MAX_OPERATIONS}, "
            f"which is already {(MAX_OPERATIONS + N_SLOTS - 1) // N_SLOTS} model calls."
        )
    return ParsedProblem(start=start, operations=operations, continued=continued)


def chunk_operations(operations: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    """Split into groups of four, padding the last with the identity.

    Padding uses `mul 1`, which is a value the generator itself produces, so a
    padded chain is still an input the model was trained on rather than an
    out-of-distribution one.
    """

    if not operations:
        return [[IDENTITY_OPERATION] * N_SLOTS]
    chunks: List[List[Tuple[int, int]]] = []
    for offset in range(0, len(operations), N_SLOTS):
        group = list(operations[offset : offset + N_SLOTS])
        while len(group) < N_SLOTS:
            group.append(IDENTITY_OPERATION)
        chunks.append(group)
    return chunks


def describe_capabilities() -> str:
    """What to say when the parser cannot read the message.

    Deliberately states the model's limits rather than apologising vaguely: a
    user who types "hello" should learn that there is no language model here.
    """

    return (
        "I am a front end for a 4-step modular-arithmetic reasoner. It has no "
        "tokenizer and no language ability at all -- the parsing here is plain "
        "code, and the model only ever sees a 128-dimensional vector. Ask for "
        "arithmetic: a start digit 0-9, then operations from add, multiply and "
        "subtract with operands 1-9, all mod 10. For example "
        "'7 times 3 plus 8 minus 5 times 6'. Chains longer than four operations "
        "run the model repeatedly, feeding its own answer forward."
    )
