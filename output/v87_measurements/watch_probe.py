"""Report each accuracy probe of the running v87 job, comparable to v86.

The probe covers 30 tasks; v86's covered 21. Averaging the new nine code tasks
into the headline makes the two runs look further apart than they are, so this
prints the original-21 subset alongside, which is the number that pairs with
v86's recorded curve:

    step   3000   6000   9000  12000  15000  18000
    v86     0.22   0.40   0.47   0.66   0.78   0.78

Reads the recovery checkpoint by memory-map, so it costs the training run
nothing beyond a file read.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from eval_problem_solving import wilson_interval  # noqa: E402

CHECKPOINT = ROOT / "output" / "v87_corpus" / "v87_corpus.partial.pt"
V86_CURVE = {3000: 0.22, 6000: 0.40, 9000: 0.47, 12000: 0.66, 15000: 0.78, 18000: 0.78}


def report(entry: dict) -> None:
    step = entry["step"]
    by_task = entry.get("probe_by_task") or {}
    original = {k: v for k, v in by_task.items() if not k.startswith("code_")}
    code = {k: v for k, v in by_task.items() if k.startswith("code_")}

    def rate(bucket):
        c = sum(v["correct"] for v in bucket.values())
        n = sum(v["total"] for v in bucket.values())
        return c, n

    oc, on = rate(original)
    cc, cn = rate(code)
    lo, hi = wilson_interval(oc, on) if on else (0.0, 0.0)
    baseline = V86_CURVE.get(step)

    print(f"\n=== step {step} | dev_loss {entry.get('dev_loss')} "
          f"| elapsed {entry.get('elapsed_seconds', 0) / 3600:.1f}h ===")
    print(f"  all 30 tasks     {entry.get('probe_accuracy')}")
    print(f"  original 21      {oc}/{on} = {oc / max(1, on):.3f} "
          f"95% CI [{lo:.3f}, {hi:.3f}]"
          + (f"   v86 here: {baseline}" if baseline is not None else ""))
    print(f"  the nine code    {cc}/{cn} = {cc / max(1, cn):.3f}"
          + "   (v86 never trained these)")
    # The tasks v87 changed, which is where a difference should show first.
    changed = ("power", "molarity", "acceleration", "percent", "average",
               "algebra_one_step")
    line = "  ".join(
        f"{name}={by_task[name]['correct']}/{by_task[name]['total']}"
        for name in changed if name in by_task)
    print(f"  changed tasks    {line}")


def main() -> int:
    seen = set()
    deadline = time.time() + float(sys.argv[1]) if len(sys.argv) > 1 else None
    while True:
        try:
            extra = torch.load(CHECKPOINT, map_location="cpu", mmap=True,
                               weights_only=False).get("extra", {})
            for entry in extra.get("history") or []:
                if "probe_accuracy" in entry and entry["step"] not in seen:
                    seen.add(entry["step"])
                    report(entry)
                    sys.stdout.flush()
        except Exception:
            pass          # a checkpoint mid-write reads as corrupt; try later
        if deadline and time.time() > deadline:
            return 0
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
