"""What does the prompt normaliser buy on naturally-typed questions?

v74 scored 0.894 on its own benchmark and 0 of 5 on questions typed the way a
person types them. v80, measured here, answers 5 of 10 -- the phrasing variety
built into the v79 corpus partly worked. The remaining failures are three
physics questions and two arithmetic ones.

`prompt_normaliser.py` exists to bridge exactly that gap and the chat server
applies it, but the benchmark does not, so its value has never been measured.
v85 added eight science rules to it. This scores the same questions twice, raw
and normalised, on the same checkpoint.

Nothing is trained and no number is computed by the normaliser: it only rewrites
the question into the shape the corpus uses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(r"C:\Users\kai99\Desktop\New folder (9)\Supermix")
SOURCE = Path(os.environ.get("SUPERMIX_SOURCE", str(REPO / "source")))
sys.path.insert(0, str(SOURCE))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("SUPERMIX_THREADS", "6")))

import eval_problem_solving as solving  # noqa: E402
import prompt_normaliser as pn  # noqa: E402
from train_mimomix_talk import generate_reply, load_talk_checkpoint  # noqa: E402

CHECKPOINT = REPO / "output" / "v80_omni" / "v80_omni.pt"

#: Questions written the way a person actually types them, not the way the
#: corpus generates them. Expanded from the ten used in the v85 sweep so the
#: result is less of a coin toss, but still small: this is an indication.
QUESTIONS = [
    # physics -- the three the normaliser had no rule for, plus more of the same
    ("If something weighs 25 kg and speeds up at 4 metres per second squared, what force is that?", 100.0),
    ("what's the force on a 12 kg object accelerating at 3 m/s^2", 36.0),
    ("A 30 kg mass is pushed with 90 N. How fast does it accelerate?", 3.0),
    ("A force of 84 N acts on 21 kg. How fast does it accelerate?", 4.0),
    ("how much momentum does a 14 kg trolley moving at 5 m/s have?", 70.0),
    ("What's the momentum of a 32 kg cart rolling at 4 m/s?", 128.0),
    ("Work done pushing with 20 N over 7 metres?", 140.0),
    ("How much work is done by 15 N across 6 m?", 90.0),
    ("A 9 volt battery drives 3 amps. What's the power?", 27.0),
    ("Electrical power at 12 volts and 5 amps?", 60.0),
    ("kinetic energy of a 10 kg body at 7 m/s", 245.0),
    ("what voltage drives 3 A through a 5 ohm resistor", 15.0),
    # arithmetic -- the operator and lead-in brittleness
    ("what is 47 times 6", 282.0),
    ("What is 47 x 6?", 282.0),
    ("whats 36 multiplied by 4", 144.0),
    ("Find the average of 61, 63, 72 and 61.", 64.25),
    ("what's the mean of 10, 20, 30 and 40", 25.0),
    ("Solve for x: x + 29 = 34", 5.0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=112)
    ap.add_argument("--out", default=str(REPO / "output" / "v85_measurements" / "natural_phrasing.json"))
    args = ap.parse_args()

    model, tokenizer, _ = load_talk_checkpoint(CHECKPOINT)
    model.eval()

    def answer(prompt):
        out = generate_reply(model, tokenizer, prompt, max_new_tokens=args.cap)
        text = out["reply"] if isinstance(out, dict) else str(out)
        return text, solving.extract_answer(text)

    rows = []
    raw_hits = norm_hits = rewritten = 0
    print(f"{'':5s} {'':5s}  question")
    print("-" * 78)
    for question, truth in QUESTIONS:
        raw_text, raw_pred = answer(question)
        raw_ok = solving.is_correct(raw_pred, truth)

        norm = pn.normalise(question)
        if norm.changed:
            rewritten += 1
            norm_text, norm_pred = answer(norm.prompt)
        else:
            norm_text, norm_pred = raw_text, raw_pred
        norm_ok = solving.is_correct(norm_pred, truth)

        raw_hits += bool(raw_ok)
        norm_hits += bool(norm_ok)
        flag = ("FIXED" if norm_ok and not raw_ok else
                "BROKE" if raw_ok and not norm_ok else
                "  ok " if raw_ok else "  -- ")
        print(f"{'Y' if raw_ok else 'n':^5s} {'Y' if norm_ok else 'n':^5s} "
              f"{flag} {question[:52]}")
        if norm.changed:
            print(f"{'':17s}asked as: {norm.prompt[:58]}")
        rows.append({
            "question": question, "truth": truth,
            "raw_prompt_answer": raw_pred, "raw_correct": bool(raw_ok),
            "normalised_prompt": norm.prompt if norm.changed else None,
            "rule": norm.rule,
            "normalised_answer": norm_pred, "normalised_correct": bool(norm_ok),
        })

    n = len(QUESTIONS)
    print("-" * 78)
    print(f"raw          {raw_hits}/{n} = {raw_hits / n:.3f}")
    print(f"normalised   {norm_hits}/{n} = {norm_hits / n:.3f}")
    print(f"rewritten    {rewritten}/{n} questions")

    lo_r, hi_r = solving.wilson_interval(raw_hits, n)
    lo_n, hi_n = solving.wilson_interval(norm_hits, n)
    payload = {
        "schema": "supermix-v85-natural-phrasing-v1",
        "checkpoint": str(CHECKPOINT),
        "n": n,
        "raw": {"correct": raw_hits, "accuracy": round(raw_hits / n, 4),
                "wilson95": [round(lo_r, 4), round(hi_r, 4)]},
        "normalised": {"correct": norm_hits, "accuracy": round(norm_hits / n, 4),
                       "wilson95": [round(lo_n, 4), round(hi_n, 4)]},
        "questions_rewritten": rewritten,
        "rows": rows,
        "non_claims": [
            f"n={n}. The Wilson intervals overlap unless the change is large; "
            "this is an indication, not a rate.",
            "The normaliser computes nothing. It rewrites the question into the "
            "shape the corpus uses, so any gain is the model answering a question "
            "it was trained on the form of.",
            "A question the model gets wrong in the trained format stays wrong.",
            "These questions were written by hand to probe known failure modes, "
            "so they are not a random sample of what a user would ask.",
        ],
    }
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
