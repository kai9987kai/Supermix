"""How many written steps does each task's format ever use?

A sample of 4,619 rows from `addition`, `subtraction`, `multiplication` and
`division` found exactly two equations in every one of them. If that holds over
the whole corpus then the number of steps is not something the model derives
from the problem -- it is a constant, and the cheapest way to fit the training
loss is to emit two equations always and put numbers in them.

There is direct evidence it does exactly that. Asked ``100 / 2`` -- a problem
whose answer needs one step -- v86 replies ``100 / 2 = 50, 0 / 2 = 0, total 50``.
The second equation is padding to reach the arity it learned, and it invented a
partial to fill it.

This scans every row rather than a sample, because "always two" and "almost
always two" call for different fixes: a constant is a defect in the format, and
a near-constant is a distribution to rebalance.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

import step_audit  # noqa: E402

CORPUS = ROOT / "datasets" / "v86" / "v86_combined.jsonl"
OUT = Path(__file__).with_suffix(".json")

#: A partial that contributes nothing: the corpus drops zero place values on
#: purpose, so any of these in model output is invented.
ZERO_PARTIAL = re.compile(r"\b0 [-+x*/] \d+(?:\.\d+)? = 0\b|\b\d+(?:\.\d+)? [x*/] 0 = 0\b")


def main() -> int:
    arity = collections.defaultdict(collections.Counter)
    zero_partials = collections.Counter()
    rows = collections.Counter()
    with CORPUS.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            task = record.get("task")
            if not task:
                continue
            reply = record.get("assistant", "")
            rows[task] += 1
            arity[task][len(step_audit.audit(reply).written)] += 1
            if ZERO_PARTIAL.search(reply):
                zero_partials[task] += 1

    report = {"schema": "supermix-v87-arity-scan-v1", "corpus": str(CORPUS), "tasks": {}}
    for task in sorted(rows):
        counts = arity[task]
        widths = sorted(counts)
        total = rows[task]
        report["tasks"][task] = {
            "rows": total,
            "step_counts": {str(k): counts[k] for k in widths},
            "distinct_step_counts": len(widths),
            "modal_share": round(max(counts.values()) / total, 4),
            "zero_partial_rows": zero_partials[task],
            "zero_partial_share": round(zero_partials[task] / total, 4),
        }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'task':20s} {'rows':>7s} {'distinct':>8s} {'modal':>6s} {'step counts'}")
    for task, entry in report["tasks"].items():
        counts = ", ".join(f"{k}x{v}" for k, v in entry["step_counts"].items())
        print(f"{task:20s} {entry['rows']:7d} {entry['distinct_step_counts']:8d} "
              f"{entry['modal_share']:6.3f} {counts[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
