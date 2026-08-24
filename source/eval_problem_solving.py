"""Measure problem solving by whether the answer is *right*.

Every metric in this repo so far scores how probable the training text was. V64
showed why that is not enough: verbatim reproduction of training data is the
lowest-loss behaviour available, so perplexity actively prefers a model that
recites over one that reasons.

Arithmetic does not have that problem. The answer to "what is 617 + 288" is
checkable, and a model that recites a remembered answer to a *novel* problem is
simply wrong. Exact-match accuracy on freshly generated problems is therefore a
recitation-proof measure of problem solving, which is what this module provides.

The decisive comparison is the pair:

* **seen** -- problems drawn verbatim from the training corpus.
* **novel** -- problems in identical phrasing with operands generated here.

A model that has learned arithmetic scores similarly on both. A model that has
memorised the corpus scores well on *seen* and near zero on *novel*, and the gap
between them is the measurement. Aggregate accuracy alone cannot tell those two
models apart, which is the whole point.

    python source/eval_problem_solving.py --checkpoint output/v64_meaning/v64_meaning.partial.pt
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

RECEIPT_SCHEMA = "supermix-v65-problem-solving-accuracy-v1"

#: Answers are floats; comparison needs a tolerance rather than equality.
TOLERANCE = 1e-6

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")


@dataclass
class Problem:
    task: str
    prompt: str
    answer: float
    source: str  # "novel" or "seen"


# -- generators ------------------------------------------------------------
#
# Phrasing is copied from `build_english_math_dataset.py` so the *format* is
# in-distribution and only the operands are new. A model that failed here purely
# because the wording was unfamiliar would be measuring the wrong thing.


def _arithmetic(rng: random.Random) -> Problem:
    a, b = rng.randint(100, 999), rng.randint(10, 999)
    op = rng.choice(["+", "-"])
    answer = a + b if op == "+" else a - b
    return Problem(
        "arithmetic", f"Solve this basic math problem: {a} {op} {b}", float(answer), "novel"
    )


def _percent(rng: random.Random) -> Problem:
    pct, base = rng.choice([5, 10, 12, 15, 20, 25]), rng.randint(20, 2000)
    return Problem("percent", f"What is {pct}% of {base}?", pct * base / 100.0, "novel")


def _average(rng: random.Random) -> Problem:
    values = [rng.randint(5, 99) for _ in range(rng.choice([4, 5, 6]))]
    joined = ", ".join(str(v) for v in values)
    return Problem(
        "average",
        f"Find the average (mean) of these numbers: {joined}",
        sum(values) / len(values),
        "novel",
    )


def _algebra(rng: random.Random) -> Problem:
    x, b = rng.randint(-30, 30), rng.randint(-30, 30)
    return Problem("algebra_one_step", f"Solve for x: x + {b} = {x + b}", float(x), "novel")


def _word_problem(rng: random.Random) -> Problem:
    start, gain, lose = rng.randint(20, 99), rng.randint(5, 60), rng.randint(5, 60)
    item = rng.choice(["notebooks", "cookies", "marbles", "stickers"])
    return Problem(
        "word_problem",
        f"A student has {start} {item}. They get {gain} more and then give away "
        f"{lose}. How many {item} do they have now?",
        float(start + gain - lose),
        "novel",
    )



# -- v74 task types ---------------------------------------------------------
#
# Added alongside the corpus generators, not after the fact. A capability the
# benchmark cannot see is a capability nobody can claim.


def _multiplication(rng: random.Random) -> Problem:
    a, b = rng.randint(11, 99), rng.randint(2, 9)
    return Problem("multiplication", f"Solve this basic math problem: {a} x {b}",
                   float(a * b), "novel")


def _division(rng: random.Random) -> Problem:
    b, quotient = rng.randint(2, 9), rng.randint(11, 60)
    return Problem("division", f"Solve this basic math problem: {b * quotient} / {b}",
                   float(quotient), "novel")


def _sequence(rng: random.Random) -> Problem:
    start, step = rng.randint(3, 40), rng.randint(2, 12)
    terms = [start + step * i for i in range(4)]
    joined = ", ".join(str(t) for t in terms)
    return Problem("sequence", f"What comes next in the sequence: {joined}?",
                   float(terms[-1] + step), "novel")


def _two_step(rng: random.Random) -> Problem:
    pct = rng.choice([10, 20, 25, 50])
    base = rng.choice([x for x in range(40, 900) if (x * pct) % 100 == 0])
    first = base * pct // 100
    delta = rng.randint(5, 60)
    add = rng.random() < 0.5
    answer = first + delta if add else first - delta
    word = "add" if add else "subtract"
    return Problem("two_step", f"What is {pct}% of {base}, then {word} {delta}?",
                   float(answer), "novel")


GENERATORS: Dict[str, Callable[[random.Random], Problem]] = {
    "arithmetic": _arithmetic,
    "percent": _percent,
    "average": _average,
    "algebra_one_step": _algebra,
    "word_problem": _word_problem,
    "multiplication": _multiplication,
    "division": _division,
    "sequence": _sequence,
    "two_step": _two_step,
}


def generate_novel(
    count: int, seed: int = 65, tasks: Optional[Sequence[str]] = None
) -> List[Problem]:
    rng = random.Random(seed)
    names = list(tasks or GENERATORS)
    return [GENERATORS[names[i % len(names)]](rng) for i in range(count)]


# -- seen problems ---------------------------------------------------------


def load_seen(
    path: Path, count: int, seed: int = 65, tasks: Optional[Sequence[str]] = None
) -> List[Problem]:
    """Problems lifted verbatim from the corpus, with their stated answers.

    These are the memorisation control. Rows whose answer cannot be parsed are
    skipped rather than guessed at -- a wrong ground truth would understate the
    model and make the comparison meaningless in the flattering direction.
    """

    wanted = set(tasks or GENERATORS)
    rows: List[Problem] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("task") not in wanted:
                continue
            answer = extract_answer(str(record.get("assistant", "")))
            if answer is None:
                continue
            rows.append(Problem(record["task"], str(record["user"]), answer, "seen"))
    random.Random(seed).shuffle(rows)
    return rows[:count]


# -- scoring ---------------------------------------------------------------


def extract_answer(text: str) -> Optional[float]:
    """The model's answer is the last number it produces.

    The corpus answers in several shapes -- "79", "376 - 17 = 359",
    "Calculate directly. Answer: 10.68." -- and in every one the final number is
    the answer. Taking the last rather than the first is what makes
    "376 - 17 = 359" score as 359 instead of 376.
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


def is_correct(predicted: Optional[float], expected: float) -> bool:
    if predicted is None:
        return False
    return abs(predicted - expected) <= max(TOLERANCE, abs(expected) * 1e-6)


def evaluate(
    checkpoint: Path, problems: Sequence[Problem], max_new_tokens: int = 40
) -> Dict[str, Any]:
    model, tokenizer, payload = load_talk_checkpoint(checkpoint)
    model.eval()

    per_task: Dict[Tuple[str, str], Dict[str, int]] = {}
    unparsed = 0
    examples: List[Dict[str, Any]] = []
    for problem in problems:
        reply = generate_reply(model, tokenizer, problem.prompt, max_new_tokens=max_new_tokens)
        text = reply["reply"] if isinstance(reply, dict) else str(reply)
        predicted = extract_answer(text)
        if predicted is None:
            unparsed += 1
        correct = is_correct(predicted, problem.answer)

        bucket = per_task.setdefault(
            (problem.source, problem.task), {"n": 0, "correct": 0, "errors": []}
        )
        bucket["n"] += 1
        bucket["correct"] += int(correct)
        # Exact match alone cannot tell "computed approximately" from "produced
        # noise", and the difference is the whole question for a model this
        # size. Relative error separates them: v65 answers 51.5 where the truth
        # is 51.333, which is a different failure from answering "Synonym: glad".
        if predicted is not None:
            bucket["errors"].append(
                abs(predicted - problem.answer) / max(1.0, abs(problem.answer))
            )
        if len(examples) < 12:
            examples.append(
                {
                    "source": problem.source,
                    "task": problem.task,
                    "prompt": problem.prompt[:90],
                    "reply": text[:90],
                    "expected": problem.answer,
                    "predicted": predicted,
                    "correct": correct,
                }
            )

    by_source: Dict[str, Dict[str, Any]] = {}
    for (source, task), counts in sorted(per_task.items()):
        entry = by_source.setdefault(source, {"n": 0, "correct": 0, "tasks": {}})
        entry["n"] += counts["n"]
        entry["correct"] += counts["correct"]
        errors = sorted(counts["errors"])
        entry.setdefault("errors", []).extend(errors)
        entry["tasks"][task] = {
            "n": counts["n"],
            "correct": counts["correct"],
            "accuracy": round(counts["correct"] / max(1, counts["n"]), 4),
            "median_relative_error": (
                round(errors[len(errors) // 2], 4) if errors else None
            ),
            "within_10_percent": round(
                sum(1 for e in errors if e <= 0.10) / max(1, counts["n"]), 4
            ),
        }
    for entry in by_source.values():
        entry["accuracy"] = round(entry["correct"] / max(1, entry["n"]), 4)
        errors = sorted(entry.pop("errors", []))
        entry["median_relative_error"] = (
            round(errors[len(errors) // 2], 4) if errors else None
        )
        entry["within_10_percent"] = round(
            sum(1 for e in errors if e <= 0.10) / max(1, entry["n"]), 4
        )

    seen = by_source.get("seen", {}).get("accuracy")
    novel = by_source.get("novel", {}).get("accuracy")
    return {
        "schema": RECEIPT_SCHEMA,
        "checkpoint": str(checkpoint),
        "dev_loss": (payload.get("extra") or {}).get("best_dev_loss"),
        "problems": len(problems),
        "unparsed_replies": unparsed,
        "by_source": by_source,
        "memorisation_gap": (
            round(seen - novel, 4) if seen is not None and novel is not None else None
        ),
        "examples": examples,
        "non_claims": [
            "Accuracy on generated arithmetic is not general problem solving. It "
            "covers five task types with small operands and nothing else.",
            "A high 'seen' score with a low 'novel' score is recall, not skill; "
            "the gap is the finding, not either number alone.",
            "Answer extraction takes the last number in the reply. A reply that "
            "reasons correctly and then trails off into another number scores "
            "wrong, so this is a lower bound on the model.",
        ],
    }


def print_summary(report: Dict[str, Any]) -> None:
    print(f"checkpoint  {report['checkpoint']}")
    print(f"dev loss    {report['dev_loss']}")
    print(f"problems    {report['problems']}   unparsed replies {report['unparsed_replies']}")
    print()
    print(f"{'source':8s} {'task':18s} {'n':>5s} {'correct':>8s} {'accuracy':>9s}")
    print("-" * 54)
    for source in ("seen", "novel"):
        entry = report["by_source"].get(source)
        if not entry:
            continue
        for task, counts in sorted(entry["tasks"].items()):
            print(
                f"{source:8s} {task:18s} {counts['n']:5d} {counts['correct']:8d} "
                f"{counts['accuracy']:9.3f}"
            )
        mre = entry.get("median_relative_error")
        print(
            f"{source:8s} {'ALL':18s} {entry['n']:5d} {entry['correct']:8d} "
            f"{entry['accuracy']:9.3f}   median rel.err "
            f"{('n/a' if mre is None else f'{mre:.3f}')}  within10% "
            f"{entry['within_10_percent']:.3f}"
        )
        print()
    gap = report.get("memorisation_gap")
    if gap is not None:
        print(f"memorisation gap (seen - novel): {gap:+.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--corpus",
        default="datasets/v62/english_math_40k.jsonl",
        help="corpus to draw 'seen' problems from",
    )
    parser.add_argument("--novel", type=int, default=100)
    parser.add_argument("--seen", type=int, default=100)
    parser.add_argument("--seed", type=int, default=65)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    problems = list(generate_novel(args.novel, seed=args.seed))
    corpus = Path(args.corpus)
    if args.seen and corpus.is_file():
        problems.extend(load_seen(corpus, args.seen, seed=args.seed))
    report = evaluate(Path(args.checkpoint), problems, args.max_new_tokens)
    print_summary(report)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
