"""Compare generalisation runs that differ in architecture, not in arm.

`train_mimomix_generalisation.compare` deliberately refuses anything but one
`full` arm against one `ablation` arm, because that comparison is only meaningful
when the runs differ in exactly one field. Comparing two *architectures* is a
different question with a different precondition, so it gets its own tool rather
than a loosened version of that one.

The precondition here is that the runs saw the same data: identical withheld
sentences, identical tier row counts, and an identical tokenizer. When those
hold, tier losses are directly comparable, because the models are scoring the
same tokens of the same rows under the same vocabulary. When they do not, the
losses are not comparable at all -- a different vocabulary means perplexity is
measured over a different unit -- and this tool raises instead of printing a
table that invites the comparison anyway.

    python source/compare_generalisation_runs.py output/v60_diverse output/v61_scaled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TIER_ORDER = (
    "tier1_seen_response",
    "tier2_unseen_response",
    "tier3_unseen_sentence",
)


def load_receipt(directory: str) -> Dict[str, Any]:
    path = Path(directory) / "generalisation_results.json"
    if not path.exists():
        raise FileNotFoundError(f"no receipt at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_comparable(reports: Sequence[Dict[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """Raise unless the runs scored the same tokens under the same vocabulary."""

    reference, reference_name = reports[0], names[0]
    reference_sentences = sorted(reference["held_out_sentences"])
    reference_vocab = reference["tokenizer"]["vocab_size"]
    reference_rows = {name: reference["tiers"][name]["pairs"] for name in TIER_ORDER}

    for report, name in zip(reports[1:], names[1:]):
        if sorted(report["held_out_sentences"]) != reference_sentences:
            raise ValueError(
                f"{name} withheld different sentences from {reference_name}; the "
                "tiers measure different things and the losses are not comparable"
            )
        vocab = report["tokenizer"]["vocab_size"]
        if vocab != reference_vocab:
            raise ValueError(
                f"{name} has vocabulary {vocab} against {reference_name}'s "
                f"{reference_vocab}. Perplexity would be measured over a "
                "different unit, so these numbers cannot be compared."
            )
        rows = {tier: report["tiers"][tier]["pairs"] for tier in TIER_ORDER}
        if rows != reference_rows:
            raise ValueError(
                f"{name} scored different tier row counts {rows} against "
                f"{reference_name}'s {reference_rows}"
            )

    return {
        "held_out_sentences": len(reference_sentences),
        "vocab_size": reference_vocab,
        "tier_rows": reference_rows,
    }


def compare_runs(directories: Sequence[str]) -> Dict[str, Any]:
    if len(directories) < 2:
        raise ValueError("need at least two run directories")

    names = [Path(d).name for d in directories]
    reports = [load_receipt(d) for d in directories]
    shared = assert_comparable(reports, names)

    rows: List[Dict[str, Any]] = []
    baseline = reports[0]
    for tier in TIER_ORDER:
        entry: Dict[str, Any] = {"tier": tier, "rows": shared["tier_rows"][tier]}
        for report, name in zip(reports, names):
            entry[name] = report["tiers"][tier]["loss"]
        entry["delta_vs_first"] = round(
            reports[-1]["tiers"][tier]["loss"] - baseline["tiers"][tier]["loss"], 6
        )
        rows.append(entry)

    return {
        "schema": "supermix-v61-architecture-comparison-v1",
        "runs": [
            {
                "name": name,
                "parameters": report["parameters"]["total"],
                "active_per_token": report["parameters"]["active_per_token"],
                "steps": report["hyperparameters"]["steps"],
                "dev_loss": report["selection"]["best_dev_loss"],
                "n_routed_experts": report["config"]["n_routed_experts"],
                "n_layers": report["config"]["n_layers"],
            }
            for report, name in zip(reports, names)
        ],
        "shared_split": shared,
        "tiers": rows,
        "note": (
            "tier losses are comparable because the runs share a split and a "
            "tokenizer; a difference here is a difference in the model, not in "
            "what was measured"
        ),
    }


def print_comparison(result: Dict[str, Any]) -> None:
    print(f"{'run':22s} {'total params':>13s} {'active':>10s} {'experts':>8s} "
          f"{'layers':>7s} {'steps':>7s} {'dev':>8s}")
    print("-" * 82)
    for run in result["runs"]:
        print(f"{run['name']:22s} {run['parameters']:13,d} {run['active_per_token']:10,d} "
              f"{run['n_routed_experts']:8d} {run['n_layers']:7d} {run['steps']:7d} "
              f"{run['dev_loss']:8.4f}")

    names = [run["name"] for run in result["runs"]]
    print()
    header = f"{'tier':24s} {'rows':>6s}"
    for name in names:
        header += f" {name[:14]:>14s}"
    header += f" {'delta':>10s}"
    print(header)
    print("-" * len(header))
    for row in result["tiers"]:
        line = f"{row['tier']:24s} {row['rows']:6d}"
        for name in names:
            line += f" {row[name]:14.4f}"
        line += f" {row['delta_vs_first']:+10.4f}"
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directories", nargs="+", help="finished run directories")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_runs(args.directories)
    print_comparison(result)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
