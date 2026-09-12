"""Is `average` failing because the chain is long, or for some other reason?

`average` is this line's oldest failure: v74 scored 0.700 on it (paired), v80
scores 0.033. Every format arm proposed for it -- binary running sums, a
quotient-remainder division tail, place-value accumulators -- assumes the
accumulation CHAIN is what breaks, and that a longer chain breaks harder.

`eval_problem_solving._average` samples 4, 5 or 6 values and the receipt does
not record which. So the assumption has never been tested. If accuracy is flat
across arities, chain length is not the mechanism and every one of those arms is
aimed at the wrong target -- which is worth knowing before spending eleven hours.

Also reports where the answer goes wrong: the running sum, or the division.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
sys.path.insert(0, str(REPO / "source"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "6")))

import eval_problem_solving as solving  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

CHECKPOINT = REPO / "output" / "v80_omni" / "v80_omni.pt"
NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_arity", type=int, default=40)
    ap.add_argument("--cap", type=int, default=96)
    ap.add_argument("--seed", type=int, default=8600)
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "average_by_arity.json"))
    args = ap.parse_args()

    model, tokenizer, _ = load_talk_checkpoint(CHECKPOINT)
    model.eval()

    # Draw until each arity bucket is full. The generator picks 4, 5 or 6
    # internally, so the arity is read back off the prompt rather than chosen.
    rng = random.Random(args.seed)
    buckets: dict = defaultdict(list)
    guard = 0
    while any(len(buckets[k]) < args.per_arity for k in (4, 5, 6)) and guard < 20000:
        guard += 1
        problem = solving.GENERATORS["average"](rng)
        values = [float(v) for v in NUMBERS.findall(problem.prompt)]
        arity = len(values)
        if arity in (4, 5, 6) and len(buckets[arity]) < args.per_arity:
            buckets[arity].append((problem, values))

    print(f"checkpoint {CHECKPOINT.name}  cap {args.cap}  "
          f"{args.per_arity} problems per arity\n")
    print(f"{'values':>7s} {'n':>4s} {'correct':>8s} {'accuracy':>9s} "
          f"{'95% CI':>16s} {'sum ok':>8s} {'div ok':>8s}")
    print("-" * 66)

    out_rows = []
    summary = {}
    for arity in (4, 5, 6):
        items = buckets[arity]
        correct = sum_right = div_right = 0
        for problem, values in items:
            reply = generate_reply(model, tokenizer, problem.prompt,
                                   max_new_tokens=args.cap)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            predicted = solving.extract_answer(text)
            hit = solving.is_correct(predicted, problem.answer)
            correct += bool(hit)

            # Did it get the running total right, whatever it did after?
            true_sum = sum(values)
            numbers = [float(v) for v in NUMBERS.findall(text)]
            sum_ok = any(abs(v - true_sum) < 1e-6 for v in numbers)
            sum_right += bool(sum_ok)
            # Given whatever total it wrote, was the division consistent?
            div_ok = bool(predicted is not None and sum_ok
                          and abs(predicted - true_sum / arity) < 1e-6)
            div_right += bool(div_ok)

            out_rows.append({"arity": arity, "prompt": problem.prompt[:80],
                             "reply": text[:110], "expected": problem.answer,
                             "predicted": predicted, "correct": bool(hit),
                             "reached_correct_sum": bool(sum_ok),
                             "division_consistent": bool(div_ok)})

        n = len(items)
        lo, hi = solving.wilson_interval(correct, n)
        summary[arity] = {"n": n, "correct": correct,
                          "accuracy": round(correct / n, 4),
                          "wilson95": [round(lo, 4), round(hi, 4)],
                          "reached_correct_sum": round(sum_right / n, 4),
                          "division_consistent": round(div_right / n, 4)}
        print(f"{arity:7d} {n:4d} {correct:8d} {correct / n:9.3f} "
              f"[{lo:.3f}, {hi:.3f}] {sum_right / n:8.3f} {div_right / n:8.3f}")

    accs = [summary[a]["accuracy"] for a in (4, 5, 6)]
    spread = max(accs) - min(accs)
    widest = max(summary[a]["wilson95"][1] - summary[a]["wilson95"][0]
                 for a in (4, 5, 6))
    flat = spread < widest / 2
    print()
    print(f"accuracy spread across arities : {spread:.3f}")
    print(f"widest single-cell interval    : {widest:.3f}")
    print()
    if flat:
        print("READING: FLAT across chain length. Chain length is NOT the")
        print("mechanism, and the accumulation-format arms are aimed at the")
        print("wrong target. Look at the division step or the format instead.")
    else:
        print("READING: accuracy FALLS with chain length, which is what the")
        print("accumulation-format arms assume. They are aimed correctly.")

    payload = {
        "schema": "supermix-v86-average-by-arity-v1",
        "question": ("Does `average` accuracy fall with the number of values? "
                     "Every proposed format arm assumes it does."),
        "checkpoint": str(CHECKPOINT),
        "settings": {"per_arity": args.per_arity, "cap": args.cap,
                     "seed": args.seed},
        "by_arity": summary,
        "accuracy_spread": round(spread, 4),
        "widest_interval": round(widest, 4),
        "flat_across_chain_length": bool(flat),
        "rows": out_rows,
        "non_claims": [
            f"n={args.per_arity} per arity; the 95% half-width at that size is "
            "in each row's wilson95 field. A spread smaller than one interval "
            "decides nothing.",
            "`reached_correct_sum` looks for the true total anywhere in the "
            "reply, so it is an upper bound: a reply could contain the right "
            "number by coincidence.",
            "This measures one checkpoint on one task. It says where to aim, "
            "not what any arm would score.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
