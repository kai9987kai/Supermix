#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from v40_benchmax_common import build_hard_example_pack, csv_write, hard_example_signature, json_dump, jsonl_write, load_manifest


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except Exception:
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract v40_benchmax hard examples from benchmark details.")
    parser.add_argument("--details_jsonl", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_examples", type=int, default=480)
    parser.add_argument("--max_examples_per_group", type=int, default=48)
    args = parser.parse_args()

    details_path = Path(args.details_jsonl).resolve()
    summary_json = Path(args.summary_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    details_rows = _load_jsonl(details_path)
    manifest = load_manifest()
    hard_rows, summary = build_hard_example_pack(
        details_rows,
        max_examples=int(args.max_examples),
        max_examples_per_group=int(args.max_examples_per_group),
        leader_models=manifest["hard_example_policy"]["leader_models"],
    )
    for row in hard_rows:
        row.setdefault("failure_signature", hard_example_signature(row))
        row["source_details_jsonl"] = str(details_path)
        row["source_summary_json"] = str(summary_json)
        row["v40_family"] = manifest["family"]

    jsonl_write(output_dir / "hard_examples.jsonl", hard_rows)
    json_dump(output_dir / "hard_examples_summary.json", summary)
    csv_write(
        output_dir / "hard_examples.csv",
        hard_rows,
        ["model", "benchmark", "failure_signature", "exact", "token_f1", "char_similarity", "hardness_score", "prompt", "reference_text", "prediction"],
    )
    md_lines = [
        "# v40_benchmax hard examples",
        "",
        f"- input details: `{details_path}`",
        f"- input summary: `{summary_json}`",
        f"- selected examples: `{len(hard_rows)}`",
        "",
        "## Failure signatures",
    ]
    top_groups = sorted(summary["group_counts"].items(), key=lambda item: item[1], reverse=True)
    for signature, count in top_groups[:24]:
        md_lines.append(f"- `{signature}`: `{count}`")
    (output_dir / "hard_examples.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[v40-benchmax] wrote hard examples to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
