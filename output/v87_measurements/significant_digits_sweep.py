"""Is it the width of the quotient, or the number of digits that had to be found?

The dose-response measured `power` at 0.125 when the quotient has three digits
and 0.725 when it has one. The obvious fix is `division`'s: split the quotient
by place value so each written step returns one place. But that fix only helps
if a quotient like ``200`` is easy while ``174`` is hard -- both are three
digits wide, and only one of them requires three separate decisions.

If round and arbitrary three-digit quotients score the same, the width is what
matters, place-value splitting leaves the hard step exactly as hard, and the
rewrite would cost a training run to learn nothing. This is the check that
decides whether the rewrite is worth making, and it costs four minutes.

Run:  python output/v87_measurements/significant_digits_sweep.py
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from eval_problem_solving import (  # noqa: E402
    extract_answer, generate_reply, is_correct, load_talk_checkpoint, wilson_interval,
)
import step_audit  # noqa: E402

CHECKPOINT = ROOT / "output" / "v86_corpus" / "v86_corpus.pt"
OUT = Path(__file__).with_suffix(".json")

TEMPLATES = [
    "{w} J of work is done in {t} s. What is the power?",
    "work {w} J time {t} s power",
    "Find the power when {w} J is delivered over {t} s.",
    "What power corresponds to {w} joules in {t} seconds?",
]
PER_CELL = 40


def quotients(kind: str, rng: random.Random, count: int):
    """Quotients of a fixed width and a fixed number of significant digits.

    `_power` draws its answer from [2, 300], so every value here is one the
    corpus contains. What varies is only how much of it had to be worked out.
    """

    values = []
    while len(values) < count:
        if kind == "3-digit round":            # 100, 200, 300: one decision
            value = rng.choice([100, 200, 300])
        elif kind == "3-digit two-place":      # 120, 250: two decisions
            value = rng.choice([1, 2]) * 100 + rng.randrange(1, 10) * 10
        elif kind == "3-digit full":           # 174, 252: three decisions
            value = rng.randrange(101, 300)
            if value % 10 == 0:
                continue
        elif kind == "2-digit round":          # 20, 50: one decision
            value = rng.randrange(1, 10) * 10
        else:                                  # "1-digit"
            value = rng.randrange(2, 10)
        values.append(value)
    return values


def main() -> int:
    model, tokenizer, _ = load_talk_checkpoint(str(CHECKPOINT))
    model.eval()

    started = time.time()
    cells = {}
    for kind in ("1-digit", "2-digit round", "3-digit round",
                 "3-digit two-place", "3-digit full"):
        rng = random.Random(f"significant/{kind}")
        correct = step_true = 0
        for index, power in enumerate(quotients(kind, rng, PER_CELL)):
            # The divisor is held in one place: the dose-response showed its
            # width barely matters, so varying it here would only add noise.
            time_s = rng.randint(10, 99)
            work = power * time_s
            prompt = TEMPLATES[index % len(TEMPLATES)].format(w=work, t=time_s)
            reply = generate_reply(model, tokenizer, prompt, max_new_tokens=96)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            correct += int(is_correct(extract_answer(text), float(power)))
            report = step_audit.audit(text)
            divisions = [s for s in report.written if s.operator == "/"]
            step_true += int(bool(divisions) and all(s.ok for s in divisions))
        low, high = wilson_interval(correct, PER_CELL)
        cells[kind] = {
            "n": PER_CELL, "correct": correct,
            "accuracy": round(correct / PER_CELL, 4),
            "accuracy_95ci": [round(low, 4), round(high, 4)],
            "division_step_true": round(step_true / PER_CELL, 4),
        }
        print(f"{kind:18s} acc {correct / PER_CELL:.3f}  "
              f"step {step_true / PER_CELL:.3f}", flush=True)

    report = {
        "schema": "supermix-v87-significant-digits-v1",
        "checkpoint": str(CHECKPOINT),
        "question": (
            "Holding quotient width at three digits, does accuracy depend on "
            "how many of those digits had to be determined?"
        ),
        "decides": (
            "Whether splitting the science divisions by place value can help. "
            "It can only help if a one-significant-digit quotient is easier "
            "than a three-significant-digit one of the same width."
        ),
        "seconds": round(time.time() - started, 1),
        "cells": cells,
        "non_claims": [
            "One task, one checkpoint. A gradient here says the rewrite is "
            "worth trying, not that it will succeed.",
            "The divisor is fixed to two digits, so this says nothing about "
            "divisor width -- the dose-response covers that.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
