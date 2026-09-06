"""Score v87 against v86 on the same 630 problems, paired.

The training probe is 100 problems over 30 tasks -- three or four per task -- so
nothing it says about an individual task can be trusted. This runs the real
benchmark at n=30 per task, on the identical problems v86 was scored on, and
tests the difference the way a paired comparison should be tested.

Three questions it answers, in order of importance:

1. **Did the overall score move?** McNemar's test on the paired outcomes, not
   two independent intervals. v80 and v86 were compared this way and the same
   seed and generator fingerprint make v86 and v87 comparable.

2. **Did the six changed tasks improve?** `power`, `molarity`, `acceleration`,
   `percent`, `average`, `algebra_one_step` each had a measured defect and a
   specific fix. Each is 30 problems here rather than three.

3. **Did the eleven untouched tasks hold?** They scored 1.000 on v86. A fall is
   exposure dilution or format interference, not a format change -- and the
   step-15,000 probe put `division` at 0/4 twice running, with the hypothesis
   that `decompose_quotient`'s three-part division collides with
   `_scratchpad_division`'s two-part one.

Run after training finishes:
    python output/v87_measurements/evaluate_v87.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

HERE = Path(__file__).parent
V87 = ROOT / "output" / "v87_corpus" / "v87_corpus.pt"
V86_REPLIES = HERE / "v86_replies.jsonl"
V87_REPLIES = HERE / "v87_replies.jsonl"
RECEIPT = HERE / "v87_paired_n630.json"

# The 21 tasks v86 was scored on. The nine code tasks are scored separately:
# v86 never trained them, so folding them into a comparison would flatter v87.
V86_TASKS = [
    "arithmetic", "percent", "average", "algebra_one_step", "word_problem",
    "multiplication", "division", "sequence", "two_step", "force",
    "acceleration", "momentum", "kinetic_energy", "work", "power", "voltage",
    "electrical_power", "wave_speed", "molarity", "combination",
    "arithmetic_series",
]
CHANGED = ["power", "molarity", "acceleration", "percent", "average",
           "algebra_one_step"]
CONTROL = ["multiplication", "division", "sequence", "force", "momentum",
           "work", "voltage", "electrical_power", "wave_speed",
           "kinetic_energy", "arithmetic_series"]


def mcnemar(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar on the discordant pairs.

    Only problems the two models disagree on carry information about which is
    better; the concordant ones cancel. With `n` discordant pairs the null is
    a fair coin, so this is an exact binomial tail.
    """

    from math import comb
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def run_eval() -> None:
    if V87_REPLIES.exists():
        print(f"reusing {V87_REPLIES}")
        return
    print("scoring v87 on 630 problems (this takes ~10 minutes)...", flush=True)
    subprocess.run([
        sys.executable, str(ROOT / "source" / "eval_problem_solving.py"),
        "--checkpoint", str(V87), "--novel", "630", "--seen", "0",
        "--seed", "65",
        "--dump_replies", str(V87_REPLIES),
        "--output", str(HERE / "v87_eval_run.json"),
    ], cwd=ROOT, check=True)


def load(path: Path) -> dict:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            rows[(r["task"], r["prompt"])] = bool(r["correct"])
    return rows


def main() -> int:
    if not V87.exists():
        print(f"no selected checkpoint yet at {V87}")
        return 1
    run_eval()

    import step_audit
    from eval_problem_solving import wilson_interval

    v86, v87 = load(V86_REPLIES), load(V87_REPLIES)
    shared = sorted(set(v86) & set(v87))
    print(f"\n{len(shared)} problems scored by both checkpoints "
          f"(v86 had {len(v86)}, v87 has {len(v87)})")

    def compare(names, label):
        keys = [k for k in shared if k[0] in names]
        a = sum(v86[k] for k in keys)
        b = sum(v87[k] for k in keys)
        only86 = sum(1 for k in keys if v86[k] and not v87[k])
        only87 = sum(1 for k in keys if v87[k] and not v86[k])
        p = mcnemar(only86, only87)
        print(f"\n{label}  (n={len(keys)})")
        print(f"  v86 {a}/{len(keys)} = {a / len(keys):.4f}")
        print(f"  v87 {b}/{len(keys)} = {b / len(keys):.4f}")
        print(f"  v87 wins {only87}, v86 wins {only86}, "
              f"McNemar exact two-sided p = {p:.4f}")
        return p

    compare(V86_TASKS, "ALL 21 SHARED TASKS")
    compare(CHANGED, "THE SIX CHANGED TASKS")
    compare(CONTROL, "THE ELEVEN UNTOUCHED CONTROL TASKS")

    print("\nper task:")
    print(f"  {'task':20s} {'v86':>6s} {'v87':>6s} {'delta':>7s}  95% CI on v87")
    for task in V86_TASKS:
        keys = [k for k in shared if k[0] == task]
        if not keys:
            continue
        a = sum(v86[k] for k in keys) / len(keys)
        b = sum(v87[k] for k in keys) / len(keys)
        lo, hi = wilson_interval(sum(v87[k] for k in keys), len(keys))
        mark = "  <-- changed" if task in CHANGED else ""
        print(f"  {task:20s} {a:6.3f} {b:6.3f} {b - a:+7.3f}  "
              f"[{lo:.3f}, {hi:.3f}]{mark}")

    code_keys = [k for k in v87 if k[0].startswith("code_")]
    if code_keys:
        c = sum(v87[k] for k in code_keys)
        lo, hi = wilson_interval(c, len(code_keys))
        print(f"\nthe nine code tasks (v86 could not attempt these): "
              f"{c}/{len(code_keys)} = {c / len(code_keys):.4f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]")

    print("\nwhere v87's wrong replies first go wrong:")
    with V87_REPLIES.open(encoding="utf-8") as handle:
        table = step_audit.summarise([json.loads(l) for l in handle])
    print(f"  {'task':20s} {'wrong':>5s} {'written step false':>18s} "
          f"{'every step true':>16s}")
    for name, entry in sorted(table.items(), key=lambda kv: kv[1]["accuracy"]):
        if entry["wrong"]:
            print(f"  {name:20s} {entry['wrong']:5d} "
                  f"{entry['wrong_at_a_written_step']:18d} "
                  f"{entry['wrong_with_sound_steps']:16d}")

    RECEIPT.write_text(json.dumps(
        {"schema": "supermix-v87-paired-v1", "problems": len(shared),
         "step_audit": table}, indent=2), encoding="utf-8")
    print(f"\nreceipt -> {RECEIPT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
