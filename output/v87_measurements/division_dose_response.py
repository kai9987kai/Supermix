"""Does operand magnitude cause the science-division failures, or accompany them?

`power` scores 0.400 and `voltage` scores 1.000 on v86. Both come out of
`build_omni_corpus`, both are one physical formula applied to two integers, and
both are scored by the same benchmark. The difference visible in the corpus is
that `voltage` decomposes its product -- ``60 x 6 = 360, 1 x 6 = 6`` -- while
`power` writes its division whole: ``19152 / 76 = 252``.

Across tasks that is a correlation, and a weak one, because the tasks differ in
a dozen other ways at the same time. This script holds the task, the format,
the prompt wording and the model fixed and moves only the size of the operands,
which is the one thing the hypothesis says should matter.

The grid is `power` problems built from the generator's own template, with the
divisor and the quotient each pinned to a digit width. If magnitude is causal
the accuracy surface falls as either width grows. If the 0.400 comes from
something else about the task, the surface is flat and the hypothesis is wrong.

Run:  python output/v87_measurements/division_dose_response.py
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

# The generator's four wordings, used verbatim so the prompts stay in
# distribution. A magnitude effect measured on a wording the model never saw
# would be a wording effect wearing a magnitude costume.
TEMPLATES = [
    "{w} J of work is done in {t} s. What is the power?",
    "work {w} J time {t} s power",
    "Find the power when {w} J is delivered over {t} s.",
    "What power corresponds to {w} joules in {t} seconds?",
]

# `_power` draws time from [2, 100] and power from [2, 300], so every cell here
# is inside the range the corpus actually contains. Nothing is extrapolated.
WIDTHS = {
    "1-digit": (2, 9),
    "2-digit": (10, 99),
    "3-digit": (100, 300),
}
DIVISOR_WIDTHS = ["1-digit", "2-digit"]
QUOTIENT_WIDTHS = ["1-digit", "2-digit", "3-digit"]
PER_CELL = 40


def cell_problems(divisor_width: str, quotient_width: str, rng: random.Random):
    lo_t, hi_t = WIDTHS[divisor_width]
    lo_p, hi_p = WIDTHS[quotient_width]
    problems = []
    for index in range(PER_CELL):
        time_s = rng.randint(lo_t, hi_t)
        power = rng.randint(lo_p, hi_p)
        work = power * time_s
        prompt = TEMPLATES[index % len(TEMPLATES)].format(w=work, t=time_s)
        problems.append((prompt, float(power), work, time_s))
    return problems


def main() -> int:
    model, tokenizer, _ = load_talk_checkpoint(str(CHECKPOINT))
    model.eval()

    started = time.time()
    cells = {}
    transcript = []
    for divisor_width in DIVISOR_WIDTHS:
        for quotient_width in QUOTIENT_WIDTHS:
            # One RNG per cell, derived from the cell's name, so a cell can be
            # rerun on its own and re-draws the same problems.
            rng = random.Random(f"{divisor_width}/{quotient_width}")
            correct = 0
            step_true = 0
            problems = cell_problems(divisor_width, quotient_width, rng)
            for prompt, answer, work, time_s in problems:
                reply = generate_reply(model, tokenizer, prompt, max_new_tokens=96)
                text = reply["reply"] if isinstance(reply, dict) else str(reply)
                predicted = extract_answer(text)
                ok = is_correct(predicted, answer)
                correct += int(ok)
                # The division is the only written step. Auditing it separately
                # from the final answer distinguishes "divided wrongly" from
                # "divided rightly and then lost the number".
                report = step_audit.audit(text)
                divisions = [s for s in report.written if s.operator == "/"]
                step_true += int(bool(divisions) and all(s.ok for s in divisions))
                transcript.append({
                    "cell": f"{divisor_width}/{quotient_width}",
                    "prompt": prompt, "reply": text,
                    "expected": answer, "predicted": predicted, "correct": ok,
                    "work": work, "time_s": time_s,
                })
            low, high = wilson_interval(correct, len(problems))
            cells[f"divisor {divisor_width} / quotient {quotient_width}"] = {
                "n": len(problems),
                "correct": correct,
                "accuracy": round(correct / len(problems), 4),
                "accuracy_95ci": [round(low, 4), round(high, 4)],
                "division_step_true": round(step_true / len(problems), 4),
            }
            print(f"{divisor_width:>8s} / {quotient_width:<8s} "
                  f"acc {correct / len(problems):.3f}  "
                  f"step {step_true / len(problems):.3f}", flush=True)

    report = {
        "schema": "supermix-v87-division-dose-response-v1",
        "checkpoint": str(CHECKPOINT),
        "question": (
            "Within a single task, format and wording held fixed, does accuracy "
            "fall as the operands of the one undecomposed step grow?"
        ),
        "per_cell": PER_CELL,
        "seconds": round(time.time() - started, 1),
        "cells": cells,
        "non_claims": [
            "This measures one task on one checkpoint. It does not establish "
            "that decomposing the step would raise accuracy; that requires "
            "training a corpus that decomposes it.",
            "The cells are not a random sample of the generator's own "
            "distribution, so the cell accuracies do not average to the "
            "task's benchmark score.",
        ],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT.parent / "division_dose_response_replies.json").write_text(
        json.dumps(transcript, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
