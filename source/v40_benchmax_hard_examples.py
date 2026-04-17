from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from v40_benchmax_core import export_hard_example_pack, load_manifest
except ImportError:  # pragma: no cover
    from .v40_benchmax_core import export_hard_example_pack, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v40_benchmax hard examples from benchmark detail rows.")
    parser.add_argument("--details_jsonl", required=True)
    parser.add_argument("--benchmark_summary", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "v40_benchmax_manifest.json"))
    parser.add_argument("--max_examples_per_bucket", type=int, default=0)
    parser.add_argument("--max_examples_total", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_dir = Path(args.output_dir).resolve() if args.output_dir.strip() else (Path(__file__).resolve().parent.parent / "output" / "v40_benchmax" / "hard_examples").resolve()
    summary = export_hard_example_pack(
        details_jsonl=Path(args.details_jsonl),
        benchmark_summary=Path(args.benchmark_summary) if args.benchmark_summary.strip() else None,
        output_dir=output_dir,
        manifest=manifest,
        max_examples_per_bucket=args.max_examples_per_bucket or None,
        max_examples_total=args.max_examples_total or None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
