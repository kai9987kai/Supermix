"""Weight the v93 corpus for a warm start: new-family rows are duplicated.

The trainer draws 16 rows per step uniformly from the packed corpus and has
no per-task weighting (train_mimomix_generalisation.py, the batch pick), so
the only way to give a family more exposure is to give it more rows. v93
warm-starts from v89, whose 1,060,000 task rows are already learned to
0.935 on the novel benchmark (v91 arm C); the 390,000 new rows are the ones
that have to be learned from scratch. Left at their natural share (25.2% of
rows) an 8,000-step run would show each new science task ~3,300 of its
40,000 rows -- a quarter of the ~13,400 draws per task that v89 needed to
reach 1.000 from scratch. Duplicating every new row ``--factor`` times (3 by
default) raises the new share to about half of every batch, which is the
cheapest exposure the budget allows without dropping v89 rows (which would
risk forgetting what the warm start is for).

Exact duplicates are safe for the split: the v58 splitter keeps identical
rows together (``duplicate_rows_kept_together`` in every receipt since v63),
so a duplicated row never lands in dev while its twin trains.

The mix is a training artefact only; ``v93_combined.jsonl`` stays the
canonical corpus, and the vocabulary extension is unaffected because
duplicates add no token types.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--corpus", default="datasets/v93/v93_combined.jsonl")
    parser.add_argument("--manifest", default="datasets/v93/v93_manifest.json")
    parser.add_argument("--output", default="datasets/v93/v93_train_mix.jsonl")
    parser.add_argument("--factor", type=int, default=3,
                        help="how many copies of every new-family row (1 = no weighting)")
    parser.add_argument("--steps", type=int, default=8000, help="planned steps, for the exposure table")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args(argv)

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    v89_tasks = set(manifest.get("v89_tasks") or [])
    if not v89_tasks:
        v89_tasks = {
            "force", "acceleration", "momentum", "kinetic_energy", "work", "power", "voltage",
            "electrical_power", "wave_speed", "molarity", "combination", "arithmetic_series",
            "addition", "subtraction", "average", "percent", "algebra_one_step", "word_problem",
            "multiplication", "division", "sequence", "two_step", "code_loop_add",
            "code_loop_subtract", "code_list_sum", "code_list_extreme", "code_index",
            "code_conditional", "code_divmod", "code_nested_loop", "code_while_accumulate",
        }

    def is_new(row: dict) -> bool:
        task = row.get("task")
        if task is not None:
            return task not in v89_tasks
        # language rows: v89's dialogue rows are inherited, everything else is new
        return row.get("domain") not in (None, "dialogue")

    started = time.time()
    counts: collections.Counter = collections.Counter()
    weighted: collections.Counter = collections.Counter()
    total_in = total_out = 0
    tmp = args.output + ".tmp"
    with open(args.corpus, encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("task") or f"lang:{row.get('domain', 'dialogue')}"
            copies = args.factor if is_new(row) else 1
            counts[key] += 1
            weighted[key] += copies
            total_in += 1
            for _ in range(copies):
                dst.write(line if line.endswith("\n") else line + "\n")
                total_out += 1
    os.replace(tmp, args.output)

    draws = args.steps * args.batch_size
    exposure = {
        key: {
            "rows": counts[key],
            "weighted_rows": weighted[key],
            "share": round(weighted[key] / total_out, 5),
            "expected_draws": round(draws * weighted[key] / total_out),
            "expected_epochs_of_its_rows": round(draws * weighted[key] / total_out / counts[key], 4),
        }
        for key in sorted(counts)
    }
    new_share = sum(v["share"] for k, v in exposure.items() if weighted[k] > counts[k])
    receipt = {
        "schema": "supermix-v93-train-mix-v1",
        "source": args.corpus,
        "output": args.output,
        "factor": args.factor,
        "rows_in": total_in,
        "rows_out": total_out,
        "new_family_share_of_rows": round(new_share, 4),
        "planned": {"steps": args.steps, "batch_size": args.batch_size, "row_draws": draws},
        "reference": "v89 from scratch: 24,250 steps x 16 = 388,000 draws over 1,156,108 rows, "
                     "about 13,400 draws per 40,000-row task",
        "exposure": exposure,
        "seconds": round(time.time() - started, 1),
    }
    with open(args.output.rsplit(".", 1)[0] + ".manifest.json", "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({k: v for k, v in receipt.items() if k != "exposure"}, indent=1))
    for key, entry in exposure.items():
        print(f"  {key:28s} rows {entry['rows']:>8,}  x{entry['weighted_rows'] // max(entry['rows'], 1)}  "
              f"share {entry['share']:.4f}  draws {entry['expected_draws']:>7,}  "
              f"epochs {entry['expected_epochs_of_its_rows']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
