"""Audit paired model replies without trusting saved correctness flags.

Exact cohorts are required by default. --allow-partial explicitly reports the
intersection and every excluded task count; partial comparisons cannot support
promotion. These diagnostic receipts never authorize promotion in either mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from eval_problem_solving import extract_answer, is_correct


def load_rows(path: Path) -> tuple[dict, dict]:
    payload = path.read_bytes()
    rows, duplicates = {}, 0
    for line in payload.decode("utf-8").splitlines():
        row = json.loads(line)
        key = tuple(row[name] for name in ("source", "task", "prompt"))
        if any(not isinstance(value, str) or not value for value in key):
            raise ValueError("invalid input identity")
        if (isinstance(row["expected"], bool) or not isinstance(row["expected"], (float, int))
                or not math.isfinite(row["expected"]) or not isinstance(row["reply"], str)):
            raise ValueError("invalid response or expected answer")
        score = is_correct(extract_answer(row["reply"]), row["expected"])
        if type(row["correct"]) is not bool or row["correct"] != score:
            raise ValueError("saved correctness disagrees with fresh scoring")
        # Repeated identical draws carry no extra independent evidence.
        if key in rows:
            if any(rows[key].get(k) != row.get(k) for k in
                   ("reply", "expected", "correct", "tokens", "truncated")):
                raise ValueError("conflicting duplicate input")
            duplicates += 1
        rows[key] = row
    if not rows:
        raise ValueError("empty transcript")
    return rows, {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                  "unique_inputs": len(rows), "duplicate_draws": duplicates}


def mcnemar_exact(baseline_only: int, candidate_only: int) -> float:
    n = baseline_only + candidate_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(baseline_only, candidate_only) + 1))
    return min(1.0, 2 * (tail / (2 ** n)))


def compare(baseline: Path, candidate: Path, *, allow_partial: bool = False) -> dict:
    left, left_info = load_rows(baseline)
    right, right_info = load_rows(candidate)
    shared = sorted(left.keys() & right.keys())
    if not shared:
        raise ValueError("no shared inputs")
    if left.keys() != right.keys() and not allow_partial:
        raise ValueError("different input cohorts; use --allow-partial for an intersection diagnostic")
    for key in shared:
        if left[key]["expected"] != right[key]["expected"]:
            raise ValueError("paired expected answers differ")

    def score(keys):
        a = sum(left[k]["correct"] for k in keys)
        b = sum(right[k]["correct"] for k in keys)
        wins = sum(right[k]["correct"] and not left[k]["correct"] for k in keys)
        losses = sum(left[k]["correct"] and not right[k]["correct"] for k in keys)
        return {"n": len(keys), "baseline_correct": a, "candidate_correct": b,
                "accuracy_delta": (b - a) / len(keys), "candidate_only_correct": wins,
                "baseline_only_correct": losses, "mcnemar_exact_two_sided_p": mcnemar_exact(losses, wins)}

    return {"schema": "supermix-paired-transcript-audit-v1", "baseline": left_info,
            "candidate": right_info, "complete_cohort_match": left.keys() == right.keys(),
            "excluded_baseline_by_task": dict(Counter(k[1] for k in left.keys() - right.keys())),
            "excluded_candidate_by_task": dict(Counter(k[1] for k in right.keys() - left.keys())),
            "paired_inputs_sha256": hashlib.sha256(json.dumps(shared).encode()).hexdigest(),
            "overall": score(shared), "by_task": {task: score([k for k in shared if k[1] == task])
                                                    for task in sorted({k[1] for k in shared})},
            "promotion_authorized": False,
            "limitations": "Descriptive paired outcomes; no checkpoint authentication, decoding-budget comparability, semantic independence, multiplicity correction, or unused-holdout claim follows from legacy transcripts alone."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, allow_partial=args.allow_partial)
    if args.output.exists():
        raise ValueError("comparison output already exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
