"""Score a checkpoint against an earlier one on the same problems, paired.

`evaluate_v87.py` did this for one pair of checkpoints with both paths hard
coded. That is how `eval_problem_solving.py` ended up with `--corpus` pinned to
a v62 dataset for eight versions, silently reporting a memorisation gap for the
wrong corpus, so this takes both sides as arguments.

The comparison is paired and tested with McNemar's exact test on the discordant
pairs, not two independent intervals: the same problems are put to both models,
so only the problems they disagree on carry information about which is better.

Three groups are reported separately, because averaging them hides the thing
worth seeing:

* **the tasks both models were trained on** -- the only honest headline;
* **the tasks only the newer model has** -- folding these into the average
  flatters it, since the older model could not attempt them at all;
* **the untouched control tasks** -- a fall here is exposure dilution or
  interference, not any format that was deliberately changed.

    python output/v87_measurements/evaluate_against.py \
        --new output/v88_corpus/v88_corpus.pt --new_name v88 \
        --baseline_replies output/v87_measurements/v87_replies.jsonl \
        --baseline_name v87
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

# The 21 tasks scored since v80. Anything outside this list is reported on its
# own rather than averaged in.
SHARED_TASKS = [
    "arithmetic", "percent", "average", "algebra_one_step", "word_problem",
    "multiplication", "division", "sequence", "two_step", "force",
    "acceleration", "momentum", "kinetic_energy", "work", "power", "voltage",
    "electrical_power", "wave_speed", "molarity", "combination",
    "arithmetic_series",
]
CONTROL = ["multiplication", "division", "sequence", "force", "momentum",
           "work", "voltage", "electrical_power", "wave_speed",
           "kinetic_energy", "arithmetic_series"]


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def wilson(correct: int, n: int) -> tuple[float, float]:
    from eval_problem_solving import wilson_interval
    return wilson_interval(correct, n)


def score(checkpoint: Path, replies: Path, problems: int, seed: int) -> None:
    if replies.exists():
        print(f"reusing {replies}")
        return
    print(f"scoring {checkpoint.name} on {problems} problems...", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "source" / "eval_problem_solving.py"),
        "--checkpoint", str(checkpoint), "--novel", str(problems), "--seen", "0",
        "--seed", str(seed), "--dump_replies", str(replies),
        "--output", str(replies.with_suffix(".run.json")),
    ], cwd=ROOT, check=True)


def load(path: Path) -> Dict[tuple, bool]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            rows[(r["task"], r["prompt"])] = bool(r["correct"])
    return rows


def compare(old: Dict[tuple, bool], new: Dict[tuple, bool],
            keys: Sequence[tuple], label: str,
            old_name: str, new_name: str) -> Optional[float]:
    if not keys:
        print(f"\n{label}: no shared problems")
        return None
    a = sum(old[k] for k in keys)
    b = sum(new[k] for k in keys)
    only_old = sum(1 for k in keys if old[k] and not new[k])
    only_new = sum(1 for k in keys if new[k] and not old[k])
    p = mcnemar(only_old, only_new)
    print(f"\n{label}  (n={len(keys)})")
    print(f"  {old_name:8s} {a:4d}/{len(keys)} = {a / len(keys):.4f}")
    print(f"  {new_name:8s} {b:4d}/{len(keys)} = {b / len(keys):.4f}")
    print(f"  {new_name} wins {only_new}, {old_name} wins {only_old}, "
          f"McNemar exact two-sided p = {p:.4f}")
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="paired comparison of two checkpoints")
    ap.add_argument("--new", required=True, help="checkpoint to score")
    ap.add_argument("--new_name", default="new")
    ap.add_argument("--baseline_replies", required=True,
                    help="dumped replies from the earlier checkpoint")
    ap.add_argument("--baseline_name", default="baseline")
    ap.add_argument("--problems", type=int, default=630)
    ap.add_argument("--seed", type=int, default=65,
                    help="must match the seed the baseline replies were dumped at")
    ap.add_argument("--output")
    args = ap.parse_args(argv)

    new_ckpt = Path(args.new)
    if not new_ckpt.exists():
        print(f"no checkpoint at {new_ckpt}")
        return 1
    baseline_replies = Path(args.baseline_replies)
    if not baseline_replies.exists():
        print(f"no baseline replies at {baseline_replies}")
        return 1

    new_replies = HERE / f"{args.new_name}_replies.jsonl"
    score(new_ckpt, new_replies, args.problems, args.seed)

    old, new = load(baseline_replies), load(new_replies)
    shared = sorted(set(old) & set(new))
    print(f"\n{len(shared)} problems scored by both "
          f"({args.baseline_name} has {len(old)}, {args.new_name} has {len(new)})")
    if not shared:
        print("nothing to compare -- were both dumped at the same seed?")
        return 1

    results = {}
    results["all_shared"] = compare(
        old, new, [k for k in shared if k[0] in SHARED_TASKS],
        "ALL SHARED TASKS", args.baseline_name, args.new_name)
    results["control"] = compare(
        old, new, [k for k in shared if k[0] in CONTROL],
        "UNTOUCHED CONTROL TASKS", args.baseline_name, args.new_name)

    print("\nper task:")
    print(f"  {'task':20s} {args.baseline_name:>8s} {args.new_name:>8s} "
          f"{'delta':>8s}  95% CI on {args.new_name}")
    per_task = {}
    for task in SHARED_TASKS:
        keys = [k for k in shared if k[0] == task]
        if not keys:
            continue
        a = sum(old[k] for k in keys) / len(keys)
        b_correct = sum(new[k] for k in keys)
        b = b_correct / len(keys)
        lo, hi = wilson(b_correct, len(keys))
        per_task[task] = {"baseline": a, "new": b, "delta": b - a, "n": len(keys)}
        print(f"  {task:20s} {a:8.3f} {b:8.3f} {b - a:+8.3f}  [{lo:.3f}, {hi:.3f}]")

    only_new_tasks = sorted({k[0] for k in new} - {k[0] for k in old})
    if only_new_tasks:
        keys = [k for k in new if k[0] in only_new_tasks]
        correct = sum(new[k] for k in keys)
        lo, hi = wilson(correct, len(keys))
        print(f"\ntasks only {args.new_name} has ({len(only_new_tasks)}): "
              f"{correct}/{len(keys)} = {correct / len(keys):.4f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  {args.baseline_name} could not attempt these, so they are "
              "reported apart from the headline")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "schema": "supermix-paired-comparison-v1",
            "baseline": args.baseline_name, "new": args.new_name,
            "checkpoint": str(new_ckpt), "problems": len(shared),
            "mcnemar_p": results, "per_task": per_task,
        }, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
