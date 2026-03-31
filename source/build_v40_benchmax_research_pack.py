#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from v40_benchmax_common import load_manifest
from v40_benchmax_experimental import build_research_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a research-inspired replay pack for v40_benchmax from hard benchmark examples.")
    parser.add_argument("--hard_examples_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--distillation_jsonl", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args()

    result = build_research_pack(
        hard_examples_jsonl=Path(args.hard_examples_jsonl).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        manifest=load_manifest(),
        distillation_jsonl=Path(args.distillation_jsonl).resolve() if str(args.distillation_jsonl).strip() else None,
        max_rows=int(args.max_rows) or None,
    )
    print(f"[v40-benchmax] wrote research pack to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
