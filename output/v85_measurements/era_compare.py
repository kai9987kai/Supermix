"""Separate "the model never learned this" from "the format changed underneath it".

v80's corpus was built 2026-08-25 23:37, so it trained on the generators as of
commit a5bd5bf2. Two later commits rewrote them:

    74642029  2026-08-26 10:50  v81 repetition/format/operand corrections
    c7041897  2026-08-26 22:43  rewrote kinetic_energy, combination, arithmetic_series

The benchmark reads its problems from whatever generators are checked out. So
the published v80 scores of 0.00 on those three tasks were measured against one
format, and any rerun today measures a different one. Asking v80 both questions
tells us which zeros are the model and which are the ruler.

Read-only: generator sources come from `git show`, nothing in the tree is touched.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
SP = Path(r"C:\Users\kai99\AppData\Local\Temp\claude\C--Users-kai99-Desktop-New-folder--9-\159b930d-3bbf-417f-b0cf-b59d28111fbe\scratchpad")
SOURCE = Path(os.environ.get("SUPERMIX_SOURCE", str(SP / "source_frozen")))
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "4")))

import eval_problem_solving as solving  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

CHECKPOINT = REPO / "output" / "v80_omni" / "v80_omni.pt"
ERAS = {
    "a5bd5bf2_what_v80_trained_on": SP / "gen_eras" / "boc_a5bd5bf2.py",
    "74642029_v81_corrections": SP / "gen_eras" / "boc_74642029.py",
    "c7041897_current": SP / "gen_eras" / "boc_c7041897.py",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # dataclass needs this before exec
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="arithmetic_series,combination,kinetic_energy,average,algebra_one_step,force,multiplication")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--cap", type=int, default=128)
    ap.add_argument("--seed", type=int, default=8202)
    ap.add_argument("--out", default=str(SP / "era_compare.json"))
    args = ap.parse_args()

    wanted = [t.strip() for t in args.tasks.split(",") if t.strip()]
    model, tokenizer, _ = load_talk_checkpoint(CHECKPOINT)
    model.eval()
    print(f"checkpoint {CHECKPOINT.name}  params {sum(p.numel() for p in model.parameters()):,}")
    print(f"tasks {wanted}  n={args.n} each  cap={args.cap}\n")

    out = {
        "schema": "supermix-v82-generator-era-comparison-v1",
        "checkpoint": str(CHECKPOINT),
        "question": "How much of v80's per-task score is the model, and how much is the "
                    "generator version the benchmark happened to be run against?",
        "v80_corpus_built": "2026-08-25T23:37:59+01:00",
        "eras": {},
        "n_per_task": args.n,
        "cap": args.cap,
        "seed": args.seed,
    }

    table = {}
    for era, path in ERAS.items():
        if not path.exists():
            print(f"skip {era}: {path} missing")
            continue
        mod = load_module(f"boc_{era}", path)
        rng = random.Random(args.seed)
        per_task = {}
        # A generator absent from an era simply cannot be scored there.
        for task in wanted:
            gen = mod.TASKS.get(task) if hasattr(mod, "TASKS") else None
            if gen is None:
                per_task[task] = None
                continue
            correct = 0
            sample = None
            for i in range(args.n):
                p = gen(rng)
                text = generate_reply(model, tokenizer, p.prompt, max_new_tokens=args.cap)
                text = text["reply"] if isinstance(text, dict) else str(text)
                predicted = solving.extract_answer(text)
                if solving.is_correct(predicted, p.answer):
                    correct += 1
                if i == 0:
                    sample = {"prompt": p.prompt[:90], "target_response": p.response[:110],
                              "model_reply": text[:110], "expected": p.answer,
                              "predicted": predicted}
            per_task[task] = {"correct": correct, "n": args.n,
                              "accuracy": round(correct / args.n, 4), "sample": sample}
            print(f"  {era:34s} {task:18s} {correct:2d}/{args.n} = {correct/args.n:.3f}", flush=True)
        out["eras"][era] = per_task
        table[era] = {k: (v["accuracy"] if v else None) for k, v in per_task.items()}

    print("\n=== accuracy by generator era ===")
    eras = list(table)
    print(f"{'task':20s} " + "  ".join(f"{e.split('_')[0]:>10s}" for e in eras))
    for task in wanted:
        cells = []
        for e in eras:
            v = table[e].get(task)
            cells.append(f"{v:10.3f}" if v is not None else f"{'absent':>10s}")
        print(f"{task:20s} " + "  ".join(cells))
    out["table"] = table
    out["non_claims"] = [
        f"n={args.n} per task per era. The 95% interval on a single cell is roughly "
        f"+-{int(100 * 1.96 * (0.25 / args.n) ** 0.5)} points, so only large moves are readable.",
        "Answer extraction takes the last number in a reply, so every cell is a lower bound.",
        "This compares one checkpoint across three question formats. It says nothing about "
        "what a model trained on any of these formats would score.",
    ]
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
