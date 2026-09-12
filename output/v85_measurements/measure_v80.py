"""Read-only measurements on the shipped v80 checkpoint.

Four questions this repository has never answered, none of which needs a retrain:

 1. What does the generation cap cost?  The offline benchmark defaults to 40 new
    tokens while arithmetic_series replies are 81 tokens.  Re-score the same
    problems at several caps.  This can move the published 0.575 headline.

 2. Does the recursive thinking core do anything at inference?  V59 measured it
    inert on v58 (0 of 12,192 decisions changed) and said explicitly that
    inertness is a property of one run, not of the mechanism.  It has never been
    re-measured on a problem-solving checkpoint.

 3. Does the phrasing decoupling built into the v79 corpus survive natural
    typing?  v74 scored 0.894 on its own format and 0 of 5 typed naturally.  The
    same measurement has never been made for the science tasks.

 4. Do greedy and MTP speculative decoding agree on a trained checkpoint?

Point SUPERMIX_SOURCE at a frozen copy of source/ so a concurrent edit cannot
change the answer mid-run.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
SOURCE = Path(os.environ.get("SUPERMIX_SOURCE", str(REPO / "source")))
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "6")))

import eval_problem_solving as solving  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

CHECKPOINT = REPO / "output" / "v80_omni" / "v80_omni.pt"


def reply_text(model, tokenizer, prompt, cap, **kw):
    out = generate_reply(model, tokenizer, prompt, max_new_tokens=cap, **kw)
    return out["reply"] if isinstance(out, dict) else str(out)


def score(model, tokenizer, problems, cap, **kw):
    correct = 0
    truncated = 0
    per_task = {}
    for p in problems:
        text = reply_text(model, tokenizer, p.prompt, cap, **kw)
        predicted = solving.extract_answer(text)
        hit = solving.is_correct(predicted, p.answer)
        # A reply that never reaches its terminal "total <n>" was cut off by the
        # cap. Scoring that as wrong measures the cap, not the model.
        if "total" not in text:
            truncated += 1
        row = per_task.setdefault(p.task, [0, 0])
        row[1] += 1
        if hit:
            row[0] += 1
            correct += 1
    return {
        "correct": correct,
        "n": len(problems),
        "accuracy": round(correct / max(1, len(problems)), 4),
        "truncated_replies": truncated,
        "per_task": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
                     for k, v in sorted(per_task.items())},
    }


def q2_generation_cap(model, tokenizer, problems, caps, out):
    print("\n=== Q2: what does the generation cap cost? ===", flush=True)
    results = {}
    for cap in caps:
        t0 = time.perf_counter()
        r = score(model, tokenizer, problems, cap)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[f"cap_{cap}"] = r
        print(f"  max_new_tokens={cap:3d}: {r['correct']:3d}/{r['n']} = {r['accuracy']:.4f}  "
              f"truncated={r['truncated_replies']:3d}  ({r['seconds']}s)", flush=True)
    out["q2_generation_cap"] = results
    lo = results[f"cap_{caps[0]}"]
    hi = max(results.values(), key=lambda v: v["accuracy"])
    out["q2_verdict"] = {
        "at_lowest_cap": lo["accuracy"],
        "best": hi["accuracy"],
        "delta": round(hi["accuracy"] - lo["accuracy"], 4),
    }
    print(f"  the cap alone moves the headline {lo['accuracy']} -> {hi['accuracy']}")
    # which tasks moved
    base = lo["per_task"]
    best = hi["per_task"]
    moved = {k: (base[k]["accuracy"], best[k]["accuracy"])
             for k in base if k in best and base[k]["accuracy"] != best[k]["accuracy"]}
    out["q2_tasks_moved"] = moved
    for k, (a, b) in sorted(moved.items(), key=lambda kv: kv[1][0] - kv[1][1]):
        print(f"    {k:20s} {a:.3f} -> {b:.3f}")


def q1_thinking_cycles(model, tokenizer, problems, cap, out):
    print("\n=== Q1: does the recursive thinking core change anything? ===", flush=True)
    results = {}
    for cycles in (1, 2, 3, 6):
        t0 = time.perf_counter()
        r = score(model, tokenizer, problems, cap, thinking_cycles=cycles)
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results[f"cycles_{cycles}"] = r
        print(f"  thinking_cycles={cycles}: {r['correct']:3d}/{r['n']} = {r['accuracy']:.4f}  "
              f"({r['seconds']}s)", flush=True)
    out["q1_thinking_cycles"] = results
    accs = {k: v["accuracy"] for k, v in results.items()}
    identical = len(set(accs.values())) == 1
    out["q1_verdict"] = {
        "accuracies": accs,
        "identical_across_cycle_counts": identical,
        "reading": ("Cycle count changes nothing, which is what an inert core looks like. "
                    "v80 kept thinking_residual_init=0.0, so the core's own gradient path "
                    "is gated to zero; see docs/V59_MECHANISM_CAUSALITY.md."
                    if identical else
                    "Cycle count changes the answer, so the core is not inert on this checkpoint."),
    }
    print(f"  identical across cycle counts: {identical}")


def q3_natural_phrasing(model, tokenizer, cap, out):
    print("\n=== Q3: does phrasing decoupling survive natural typing? ===", flush=True)
    natural = [
        ("If something weighs 25 kg and speeds up at 4 metres per second squared, what force is that?", 100.0, "force"),
        ("what's the force on a 12 kg object accelerating at 3 m/s^2", 36.0, "force"),
        ("A 30 kg mass is pushed with 90 N. How fast does it accelerate?", 3.0, "acceleration"),
        ("how much momentum does a 14 kg trolley moving at 5 m/s have?", 70.0, "momentum"),
        ("Work done pushing with 20 N over 7 metres?", 140.0, "work"),
        ("A 9 volt battery drives 3 amps. What's the power?", 27.0, "electrical_power"),
        ("what is 47 times 6", 282.0, "multiplication"),
        ("What is 47 x 6?", 282.0, "multiplication"),
        ("Find the average of 61, 63, 72 and 61.", 64.25, "average"),
        ("Solve for x: x + 29 = 34", 5.0, "algebra_one_step"),
    ]
    rows = []
    hits = 0
    for prompt, expected, task in natural:
        text = reply_text(model, tokenizer, prompt, cap)
        predicted = solving.extract_answer(text)
        hit = solving.is_correct(predicted, expected)
        hits += bool(hit)
        rows.append({"task": task, "prompt": prompt, "reply": text[:140],
                     "expected": expected, "predicted": predicted, "correct": bool(hit)})
        print(f"  [{'OK ' if hit else 'BAD'}] {prompt[:56]:56s} -> {predicted} (want {expected})", flush=True)
    out["q3_natural_phrasing"] = {"correct": hits, "n": len(natural), "rows": rows}
    print(f"  natural phrasing: {hits}/{len(natural)}")


def q4_speculative_parity(model, tokenizer, problems, cap, out):
    print("\n=== Q4: greedy vs MTP speculative decoding parity ===", flush=True)
    n = min(30, len(problems))
    mismatches = []
    for p in problems[:n]:
        greedy = reply_text(model, tokenizer, p.prompt, cap, speculative=False)
        spec = reply_text(model, tokenizer, p.prompt, cap, speculative=True)
        if greedy != spec:
            mismatches.append({"task": p.task, "prompt": p.prompt[:80],
                               "greedy": greedy[:90], "speculative": spec[:90]})
    out["q4_speculative_parity"] = {
        "n": n, "mismatches": len(mismatches), "examples": mismatches[:5],
    }
    print(f"  mismatches: {len(mismatches)}/{n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_task", type=int, default=6)
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--caps", default="40,64,96,128")
    ap.add_argument("--seed", type=int, default=8202)
    ap.add_argument("--out", default=str(REPO / "output" / "v82_measurements" / "v80_measurements.json"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    print(f"source   : {SOURCE}")
    print(f"threads  : {torch.get_num_threads()}")
    model, tokenizer, payload = load_talk_checkpoint(CHECKPOINT)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    print(f"checkpoint: {CHECKPOINT.name}  params {total:,}  vocab {tokenizer.vocab_size}", flush=True)

    rng = random.Random(args.seed)
    tasks = sorted(solving.GENERATORS)
    problems = []
    for name in tasks:
        for _ in range(args.per_task):
            problems.append(solving.GENERATORS[name](rng))
    print(f"problems : {len(problems)} over {len(tasks)} tasks ({args.per_task} each)", flush=True)

    out = {
        "schema": "supermix-v82-checkpoint-measurements-v1",
        "checkpoint": str(CHECKPOINT),
        "parameters": total,
        "tasks": tasks,
        "per_task": args.per_task,
        "seed": args.seed,
        "source_tree": str(SOURCE),
        "non_claims": [
            "These measure one checkpoint on generated problems in the corpus prompt format.",
            "Nothing here is a natural-language capability measurement except Q3, which is n=10.",
            "Answer extraction takes the last number in a reply, so every score is a lower bound.",
            f"n={args.per_task} per task: a per-task difference of less than roughly "
            f"{int(100 * 1.96 * (0.25 / max(1, args.per_task)) ** 0.5)} points is noise.",
        ],
    }
    want = set(args.only.split(",")) if args.only else {"q1", "q2", "q3", "q4"}
    caps = [int(c) for c in args.caps.split(",")]
    if "q2" in want:
        q2_generation_cap(model, tokenizer, problems, caps, out)
    if "q1" in want:
        q1_thinking_cycles(model, tokenizer, problems, args.cap, out)
    if "q3" in want:
        q3_natural_phrasing(model, tokenizer, args.cap, out)
    if "q4" in want:
        q4_speculative_parity(model, tokenizer, problems, args.cap, out)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
