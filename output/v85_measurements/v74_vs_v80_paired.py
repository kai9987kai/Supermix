"""The first paired comparison of v74 and v80.

`docs/V85_MEASURABLE_ARCHITECTURE.md` lists v80's regression against v74 on their
nine shared tasks (0.894 -> 0.622) as unexplained. Part of it is not a regression
at all: the two were never scored on the same questions.

Until v85 the benchmark drew every task's problems from **one RNG shared across
tasks in turn**. v74 was scored over 10 registered tasks and v80 over 21, so the
same seed produced different problems for every shared task. Verified:

    per-task RNG (v85):  `average` problems identical whether drawn from 9 or 21 tasks
    shared RNG (before): not identical

v85's per-task RNG is what makes a paired run possible, so this runs one. Same
nine tasks, same seed, same generation cap, same problems, both checkpoints.

Neither checkpoint is retrained or modified. This only re-scores them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
sys.path.insert(0, str(REPO / "source"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "4")))

import eval_problem_solving as solving  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

SHARED = ["algebra_one_step", "arithmetic", "average", "division",
          "multiplication", "percent", "sequence", "two_step", "word_problem"]

CHECKPOINTS = {
    "v74": REPO / "output" / "v74_broad" / "v74_broad.pt",
    "v80": REPO / "output" / "v80_omni" / "v80_omni.pt",
}

#: What each was published as, for context only. These are NOT comparable with
#: the numbers this script produces, and that is the entire point.
PUBLISHED = {
    "v74": {"algebra_one_step": 0.893, "arithmetic": 0.893, "average": 0.589,
            "division": 1.000, "multiplication": 1.000, "percent": 0.750,
            "sequence": 0.982, "two_step": 0.982, "word_problem": 0.964},
    "v80": {"algebra_one_step": 0.300, "arithmetic": 0.633, "average": 0.033,
            "division": 0.600, "multiplication": 1.000, "percent": 0.600,
            "sequence": 0.833, "two_step": 0.733, "word_problem": 0.867},
}


def score(name, path, problems, cap):
    model, tokenizer, _ = load_talk_checkpoint(path)
    model.eval()
    per_task = {}
    truncated = 0
    for p in problems:
        out = generate_reply(model, tokenizer, p.prompt, max_new_tokens=cap)
        text = out["reply"] if isinstance(out, dict) else str(out)
        ok = solving.is_correct(solving.extract_answer(text), p.answer)
        if not solving.looks_terminated(text):
            truncated += 1
        row = per_task.setdefault(p.task, {"n": 0, "correct": 0})
        row["n"] += 1
        row["correct"] += bool(ok)
    total = sum(v["n"] for v in per_task.values())
    correct = sum(v["correct"] for v in per_task.values())
    print(f"  {name}: {correct}/{total} = {correct / total:.4f}  "
          f"(truncated {truncated})", flush=True)
    return {"correct": correct, "n": total, "accuracy": correct / total,
            "truncated": truncated, "per_task": per_task}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_task", type=int, default=30)
    ap.add_argument("--cap", type=int, default=96)
    ap.add_argument("--seed", type=int, default=74)
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "v74_vs_v80_paired.json"))
    args = ap.parse_args()

    missing = [k for k, v in CHECKPOINTS.items() if not v.exists()]
    if missing:
        print(f"missing checkpoints: {missing}")
        return 1

    count = args.per_task * len(SHARED)
    problems = solving.generate_novel(count, seed=args.seed, tasks=SHARED)
    got = {}
    for p in problems:
        got[p.task] = got.get(p.task, 0) + 1
    print(f"{len(problems)} problems over {len(SHARED)} shared tasks, "
          f"cap {args.cap}, seed {args.seed}")
    print(f"per task: {sorted(got.items())}")
    print("both checkpoints see the identical problem list\n")

    results = {name: score(name, path, problems, args.cap)
               for name, path in CHECKPOINTS.items()}

    print()
    print(f"{'task':20s} {'v74':>7s} {'v80':>7s} {'delta':>8s}   "
          f"{'published v74':>13s} {'published v80':>13s}")
    print("-" * 76)
    rows = {}
    for task in SHARED:
        a = results["v74"]["per_task"][task]
        b = results["v80"]["per_task"][task]
        pa, pb = a["correct"] / a["n"], b["correct"] / b["n"]
        rows[task] = {"v74": round(pa, 4), "v80": round(pb, 4),
                      "delta": round(pb - pa, 4), "n": a["n"],
                      "published_v74": PUBLISHED["v74"][task],
                      "published_v80": PUBLISHED["v80"][task]}
        print(f"{task:20s} {pa:7.3f} {pb:7.3f} {pb - pa:+8.3f}   "
              f"{PUBLISHED['v74'][task]:13.3f} {PUBLISHED['v80'][task]:13.3f}")
    print("-" * 76)
    ov74 = results["v74"]["accuracy"]
    ov80 = results["v80"]["accuracy"]
    print(f"{'OVERALL':20s} {ov74:7.3f} {ov80:7.3f} {ov80 - ov74:+8.3f}   "
          f"{0.894:13.3f} {0.622:13.3f}")

    lo74, hi74 = solving.wilson_interval(results["v74"]["correct"], results["v74"]["n"])
    lo80, hi80 = solving.wilson_interval(results["v80"]["correct"], results["v80"]["n"])
    payload = {
        "schema": "supermix-v85-v74-vs-v80-paired-v1",
        "question": ("Is v80's regression against v74 real, or partly an artifact of "
                     "the two never having been scored on the same problems?"),
        "method": ("v85's per-task RNG makes the draw independent of how many tasks "
                   "are registered, so both checkpoints see one identical problem "
                   "list. Before v85 one RNG was shared across tasks in turn, and "
                   "v74 was scored over 10 registered tasks while v80 was scored "
                   "over 21 -- so no shared-task comparison between them was paired."),
        "settings": {"tasks": SHARED, "per_task": args.per_task, "cap": args.cap,
                     "seed": args.seed,
                     "generator_fingerprint": solving.generator_fingerprint(SHARED)},
        "v74": {"correct": results["v74"]["correct"], "n": results["v74"]["n"],
                "accuracy": round(ov74, 4), "wilson95": [round(lo74, 4), round(hi74, 4)],
                "truncated": results["v74"]["truncated"]},
        "v80": {"correct": results["v80"]["correct"], "n": results["v80"]["n"],
                "accuracy": round(ov80, 4), "wilson95": [round(lo80, 4), round(hi80, 4)],
                "truncated": results["v80"]["truncated"]},
        "per_task": rows,
        "published_for_context_only": PUBLISHED,
        "non_claims": [
            f"n={args.per_task} per task: a per-task 95% half-width is about "
            f"{int(100 * 1.96 * (0.25 / args.per_task) ** 0.5)} points, so only the "
            "overall row and large per-task moves are readable.",
            "The published columns were measured on DIFFERENT problems at an "
            "unrecorded generation cap. They are shown for context and must not be "
            "differenced against these.",
            "This explains how much of the gap is the ruler. It does not explain "
            "whatever gap remains.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
