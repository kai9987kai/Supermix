"""Does a borrowing remainder step predict a wrong subtraction?

## The first version of this file was invalid, and the reason is worth keeping

It asked the model two-digit subtractions -- `61 - 56` -- to isolate the borrow.
Everything scored 0/100, which looked like a devastating result and was an
artefact. `subtraction` trains only on **three-digit** operands, so a two-digit
question is out of distribution and the model answers it by inventing a hundreds
column:

    asked 61 - 56  ->  600 - 0 = 600, 11 - 56 = -45, total 555

This is the same error `subdivision_probe.py` made in v87, which its own README
warns about at length. Probing outside the training distribution measures the
distribution, not the mechanism.

## What this asks instead

The `subtraction` format already splits every problem into a hundreds step and a
remainder step:

    561 - 356  ->  500 - 300 = 200, 61 - 56 = 5, total 205

The remainder step is a two-digit subtraction that the model performs **inside
its own distribution**, and it either borrows or it does not. So the question can
be asked without leaving the corpus at all: generate three-digit subtractions
whose remainder borrows, and matched ones whose remainder does not, and compare.

    borrows       561 - 356   remainder 61 - 56
    carry-free    568 - 356   remainder 68 - 56

Same task, same shape, same operand widths, one variable. If borrowing is what
breaks these steps, the first column scores worse. If the two are equal,
carrying is not the mechanism and splitting the remainder by column -- which
would cost tokens on every arithmetic row -- buys nothing.

`--report_step` additionally reads the model's own written remainder step out of
the reply and scores that in isolation, which separates "the remainder step was
wrong" from "the remainder step was right and something later was not".

    python output/v87_measurements/carry_probe.py --checkpoint output/v87_corpus/v87_corpus.pt
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from eval_problem_solving import extract_answer, wilson_interval  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

REMAINDER = re.compile(r"(\d+) - (\d+) = (-?\d+)")


def build_cases(n: int, seed: int) -> list[dict]:
    """Matched three-digit subtractions whose remainder step does or does not borrow.

    Both members of a pair share a subtrahend and the same hundreds step, so the
    only difference is whether the units column of the remainder borrows.
    """

    rng = random.Random(seed)
    cases = []
    while len(cases) < n:
        left_h = rng.randint(2, 9) * 100
        right_h = rng.randint(1, left_h // 100 - 1) * 100
        right_r = rng.randint(11, 89)
        units = right_r % 10
        if units == 0:
            continue                      # nothing can borrow against zero
        borrow_units = rng.randint(0, units - 1)
        free_units = rng.randint(units, 9)
        # Both members of the pair must have a POSITIVE remainder. Without this
        # the borrowing case can come out negative while its control does not,
        # and the contrast is then borrow-plus-sign rather than borrow alone.
        floor = (right_r // 10) + 1
        if floor > 9:
            continue
        tens = rng.randint(floor, 9) * 10
        borrow_r, free_r = tens + borrow_units, tens + free_units
        cases.append({
            "borrows_left": left_h + borrow_r, "free_left": left_h + free_r,
            "right": right_h + right_r,
            "borrows_remainder": (borrow_r, right_r),
            "free_remainder": (free_r, right_r),
        })
    return cases


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def written_remainder_ok(reply: str, expected: tuple[int, int]) -> bool | None:
    """Did the model's own written remainder step state the truth?

    None when no step of that shape was written at all, which is not the same
    as writing a wrong one and must not be scored as if it were.
    """

    left, right = expected
    for match in REMAINDER.finditer(reply):
        if int(match.group(1)) == left and int(match.group(2)) == right:
            return int(match.group(3)) == left - right
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="does a borrowing remainder step fail more")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=88)
    ap.add_argument("--cap", type=int, default=48)
    ap.add_argument("--output")
    args = ap.parse_args(argv)

    cases = build_cases(args.n, args.seed)
    model, tokenizer, _ = load_talk_checkpoint(args.checkpoint)
    model.eval()

    def ask(left: int, right: int, remainder: tuple[int, int]) -> tuple[bool, bool | None, str]:
        prompt = f"Solve this basic math problem: {left} - {right}"
        out = generate_reply(model, tokenizer, prompt, max_new_tokens=args.cap)
        text = out["reply"] if isinstance(out, dict) else str(out)
        got = extract_answer(text)
        return (got is not None and abs(got - (left - right)) < 1e-6,
                written_remainder_ok(text, remainder), text)

    for index, case in enumerate(cases, 1):
        for label in ("borrows", "free"):
            ok, step, text = ask(case[f"{label}_left"], case["right"],
                                 case[f"{label}_remainder"])
            case[f"{label}_ok"] = ok
            case[f"{label}_step_ok"] = step
            case[f"{label}_reply"] = text
        if index % 20 == 0:
            print(f"  {index}/{len(cases)}", flush=True)

    n = len(cases)
    print()
    for label, title in (("borrows", "remainder borrows"),
                         ("free", "remainder carry-free")):
        correct = sum(c[f"{label}_ok"] for c in cases)
        lo, hi = wilson_interval(correct, n)
        print(f"  {title:22s} {correct:4d}/{n} = {correct / n:.3f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]")

    print("\n  the written remainder step itself, where the model wrote one:")
    step_scores = {}
    for label, title in (("borrows", "remainder borrows"),
                         ("free", "remainder carry-free")):
        scored = [c for c in cases if c[f"{label}_step_ok"] is not None]
        correct = sum(c[f"{label}_step_ok"] for c in scored)
        step_scores[label] = (correct, len(scored))
        if scored:
            lo, hi = wilson_interval(correct, len(scored))
            print(f"    {title:20s} {correct:4d}/{len(scored)} = "
                  f"{correct / len(scored):.3f} 95% CI [{lo:.3f}, {hi:.3f}]")
        else:
            print(f"    {title:20s} no step of that shape was ever written")

    borrow_only = sum(1 for c in cases if c["borrows_ok"] and not c["free_ok"])
    free_only = sum(1 for c in cases if c["free_ok"] and not c["borrows_ok"])
    p_value = mcnemar(borrow_only, free_only)
    print(f"\n  carry-free-only wins {free_only}, borrowing-only wins "
          f"{borrow_only}, McNemar exact two-sided p = {p_value:.4f}")

    borrows_correct = sum(c["borrows_ok"] for c in cases)
    free_correct = sum(c["free_ok"] for c in cases)
    print()
    if p_value > 0.05:
        print("  READ: borrowing is NOT the mechanism -- a carry-free remainder "
              "scores no\n  better. Splitting the remainder by column would cost "
              "tokens on every\n  arithmetic row and buy nothing. Do not build it.")
    elif free_correct > borrows_correct:
        print("  READ: a borrowing remainder step is measurably worse. Splitting "
              "it so no\n  written step crosses a column is worth an arm -- the "
              "tens digit is read off\n  the subtrahend, so both halves are "
              "derivable forward.")
    else:
        print("  READ: the difference is significant in the wrong direction. "
              "Investigate\n  before building anything.")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "schema": "supermix-v88-carry-probe-v2",
            "checkpoint": args.checkpoint, "n": n,
            "note": "in-distribution: three-digit subtractions whose written "
                    "remainder step does or does not borrow",
            "borrows": {"correct": borrows_correct, "accuracy": borrows_correct / n,
                        "wilson95": wilson_interval(borrows_correct, n)},
            "carry_free": {"correct": free_correct, "accuracy": free_correct / n,
                           "wilson95": wilson_interval(free_correct, n)},
            "written_step": {k: {"correct": v[0], "scored": v[1]}
                             for k, v in step_scores.items()},
            "mcnemar": {"borrowing_only": borrow_only, "carry_free_only": free_only,
                        "p_value": p_value},
            "cases": cases,
        }, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
