"""Is solver-verified rejection sampling worth twenty hours?

The repository owns an exact solver (`nexus_solver.solve_problem`) that already
verifies every generated corpus row. That makes a rejection-sampling loop
possible: sample k answers, keep the ones the solver confirms, fine-tune on
those (STaR / RFT / ReST-EM). Nobody has run it, and it costs a full training
run to find out.

There is a cheap experiment that decides it first. If the model can already
produce a correct answer *somewhere* in k samples far more often than it does
greedily, rejection sampling has raw material to work with. If pass@k is barely
above pass@1, there is nothing to harvest and the run should not be spent.

    pass@1  = greedy accuracy, what the benchmark reports
    pass@k  = fraction of problems where at least one of k samples is correct
    headroom = pass@k - pass@1

This measures it. Nothing is trained.

Sampling had to be written here: every decode path in this repository is argmax.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
SOURCE = Path(os.environ.get("SUPERMIX_SOURCE", str(REPO / "source")))
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "6")))

import eval_problem_solving as solving  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

CHECKPOINT = REPO / "output" / "v80_omni" / "v80_omni.pt"


@torch.no_grad()
def sample_reply(model, tokenizer, prompt, max_new_tokens, temperature, top_k, generator):
    """Temperature + top-k sampling. Mirrors greedy_generate's cache handling."""
    model.eval()
    ids, _ = tokenizer.encode_turn(prompt, None)
    input_ids = torch.tensor([ids], dtype=torch.long)

    out = model(input_ids, use_cache=True, return_mtp=False, past_length=0)
    past = out.past_key_values
    position = int(input_ids.shape[1])

    emitted = []
    logits = out.logits[:, -1]
    for _ in range(max_new_tokens):
        scaled = logits / max(1e-6, temperature)
        if top_k:
            kth = torch.topk(scaled, min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probs = torch.softmax(scaled, dim=-1)
        token = torch.multinomial(probs, num_samples=1, generator=generator)
        tok = int(token)
        if tok == text_utils.EOS:
            break
        emitted.append(tok)
        step = model(token, past_key_values=past, use_cache=True,
                     return_mtp=False, past_length=position)
        past = step.past_key_values
        position += 1
        logits = step.logits[:, -1]
    return tokenizer.decode(emitted).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_task", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--cap", type=int, default=112)
    ap.add_argument("--seed", type=int, default=860)
    ap.add_argument("--tasks", default="")
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "passk_headroom.json"))
    args = ap.parse_args()

    model, tokenizer, _ = load_talk_checkpoint(CHECKPOINT)
    model.eval()

    all_tasks = sorted(solving.GENERATORS)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] or all_tasks
    rng = random.Random(args.seed)
    problems = []
    for name in tasks:
        for _ in range(args.per_task):
            problems.append(solving.GENERATORS[name](rng))

    print(f"checkpoint {CHECKPOINT.name}   problems {len(problems)} over {len(tasks)} tasks")
    print(f"k={args.k}  temperature={args.temperature}  top_k={args.top_k}  cap={args.cap}")
    print()

    gen = torch.Generator().manual_seed(args.seed)
    per_task = defaultdict(lambda: {"n": 0, "greedy": 0, "samples_only": 0,
                                    "union": 0, "sample_hits": 0, "samples": 0})
    rows = []
    started = time.perf_counter()
    for index, p in enumerate(problems):
        greedy_out = generate_reply(model, tokenizer, p.prompt, max_new_tokens=args.cap)
        greedy_text = greedy_out["reply"] if isinstance(greedy_out, dict) else str(greedy_out)
        greedy_ok = solving.is_correct(solving.extract_answer(greedy_text), p.answer)

        hits = 0
        found = False
        for _ in range(args.k):
            text = sample_reply(model, tokenizer, p.prompt, args.cap,
                                args.temperature, args.top_k, gen)
            ok = solving.is_correct(solving.extract_answer(text), p.answer)
            hits += bool(ok)
            found = found or ok

        bucket = per_task[p.task]
        bucket["n"] += 1
        bucket["greedy"] += bool(greedy_ok)
        # Two different quantities, and conflating them inflates the headline.
        # `samples_only` is pass@k as the term is normally defined and as this
        # module's docstring defines it: at least one of the k SAMPLES correct.
        # `union` folds in the greedy decode, which is the thing a harvest is
        # meant to improve on, not part of the harvest. The `power` task is the
        # proof: zero of its samples are ever correct, so its union cell is
        # entirely greedy.
        bucket["samples_only"] += bool(found)
        bucket["union"] += bool(found or greedy_ok)
        bucket["sample_hits"] += hits
        bucket["samples"] += args.k
        rows.append({"task": p.task, "greedy_correct": bool(greedy_ok),
                     "any_sample_correct": bool(found),
                     "union_correct": bool(found or greedy_ok),
                     "hits_of_k": hits})
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(problems)}  "
                  f"({time.perf_counter() - started:.0f}s)", flush=True)

    n = sum(v["n"] for v in per_task.values())
    greedy = sum(v["greedy"] for v in per_task.values())
    anyk = sum(v["samples_only"] for v in per_task.values())
    union = sum(v["union"] for v in per_task.values())
    hits = sum(v["sample_hits"] for v in per_task.values())
    samples = sum(v["samples"] for v in per_task.values())

    print()
    print(f"{'task':22s} {'n':>3s} {'pass@1':>7s} {f'pass@{args.k}':>8s} "
          f"{'union':>7s} {'headroom':>9s} {'sample rate':>12s}")
    print("-" * 78)
    harvestable = []
    for task in sorted(per_task):
        v = per_task[task]
        p1 = v["greedy"] / v["n"]
        pk = v["samples_only"] / v["n"]
        un = v["union"] / v["n"]
        rate = v["sample_hits"] / v["samples"]
        print(f"{task:22s} {v['n']:3d} {p1:7.3f} {pk:8.3f} {un:7.3f} "
              f"{un - p1:+9.3f} {rate:12.3f}")
        if un - p1 >= 0.15:
            harvestable.append(task)
    print("-" * 78)
    print(f"{'OVERALL':22s} {n:3d} {greedy / n:7.3f} {anyk / n:8.3f} "
          f"{union / n:7.3f} {(union - greedy) / n:+9.3f} {hits / samples:12.3f}")
    print()
    print(f"pass@{args.k} counts the {args.k} SAMPLES only. 'union' additionally "
          "counts problems only the greedy decode solved; headroom is union "
          "minus greedy, which is the set a harvest would gain.")

    payload = {
        "schema": "supermix-v85-passk-headroom-v1",
        "question": "Does solver-verified rejection sampling have raw material to harvest?",
        "checkpoint": str(CHECKPOINT),
        "settings": {"k": args.k, "temperature": args.temperature, "top_k": args.top_k,
                     "cap": args.cap, "per_task": args.per_task, "seed": args.seed},
        "overall": {"n": n, "pass_at_1": round(greedy / n, 4),
                    f"pass_at_{args.k}": round(anyk / n, 4),
                    "union_greedy_or_sample": round(union / n, 4),
                    "headroom": round((union - greedy) / n, 4),
                    "per_sample_correct_rate": round(hits / samples, 4)},
        "per_task": {k: {"n": v["n"], "pass_at_1": round(v["greedy"] / v["n"], 4),
                         f"pass_at_{args.k}": round(v["samples_only"] / v["n"], 4),
                         "union_greedy_or_sample": round(v["union"] / v["n"], 4),
                         "headroom": round((v["union"] - v["greedy"]) / v["n"], 4),
                         "per_sample_correct_rate": round(v["sample_hits"] / v["samples"], 4)}
                     for k, v in sorted(per_task.items())},
        "tasks_with_harvestable_headroom": harvestable,
        "rows": rows,
        "reading": (
            "Headroom is the fraction of problems the model can already solve "
            "somewhere in k samples but does not solve greedily. That is exactly "
            "the set a rejection-sampling round would harvest and train on. Near "
            "zero headroom means the run is not worth spending."
        ),
        "non_claims": [
            f"n={args.per_task} per task; per-task headroom below roughly "
            f"{int(100 * 1.96 * (0.25 / max(1, args.per_task)) ** 0.5)} points is noise.",
            "pass@k here counts the k SAMPLES only. The separate union column adds "
            "problems solved only by the greedy decode; an earlier version of this "
            "script reported the union as pass@k, which inflated it (0.708 against "
            "the true 0.667 overall). Headroom is union minus greedy either way.",
            "pass@k is an upper bound on what rejection sampling can harvest, not a "
            "prediction of what a fine-tune would reach.",
            "Sampling was added here; every shipped decode path is argmax. These "
            "temperature and top-k settings are a first guess, not a tuned choice.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    print(f"\ntasks with harvestable headroom (>= 0.15): {harvestable or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
