#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from v40_benchmax_common import ablation_root, build_ablation_pack, csv_write, json_dump, jsonl_write, load_manifest, manifest_ablation_map


def _write_pack(repo_root: Path, ablation_id: str, rows: List[Dict[str, object]], summary: Dict[str, object]) -> None:
    pack_dir = ablation_root(repo_root, ablation_id)
    pack_dir.mkdir(parents=True, exist_ok=True)
    rows_path = pack_dir / "rows.jsonl"
    summary_path = pack_dir / "summary.json"
    csv_path = pack_dir / "summary.csv"
    jsonl_write(rows_path, rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    csv_write(
        csv_path,
        [
            {
                "ablation_id": ablation_id,
                "data_recipe": summary.get("recipe", ""),
                "head_recipe": summary.get("head_recipe", ""),
                "rows": summary.get("selected_rows", 0),
                "source_rows": summary.get("source_rows", 0),
            }
        ],
        ["ablation_id", "data_recipe", "head_recipe", "rows", "source_rows"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v40_benchmax controlled 2x2 ablation packs.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--ablation_id", action="append", default=[], help="Build only the named ablation. Repeat to select multiple.")
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    manifest_out = output_dir / "v40_benchmax_manifest.json"
    json_dump(manifest_out, manifest)

    ablations = manifest_ablation_map(manifest)
    wanted = [item.strip() for item in args.ablation_id if item.strip()] or sorted(ablations)
    rows_written: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    for ablation_id in wanted:
        rows, summary = build_ablation_pack(repo_root, ablation_id, seed=int(args.seed), sample_size=int(args.sample_size) or None)
        summary_rows.append(summary)
        if not args.dry_run:
            _write_pack(repo_root, ablation_id, rows, summary)
            rows_written.append({"ablation_id": ablation_id, "rows": len(rows)})

    table_path = output_dir / "v40_benchmax_ablation_table.csv"
    csv_write(
        table_path,
        summary_rows,
        ["ablation_id", "recipe", "head_recipe", "selected_rows", "source_rows"],
    )
    summary_path = output_dir / "v40_benchmax_ablation_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "family": manifest["family"],
                "output_dir": str(output_dir),
                "ablation_ids": wanted,
                "dry_run": bool(args.dry_run),
                "summary_rows": summary_rows,
                "rows_written": rows_written,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"[v40-benchmax] wrote ablation summary to {summary_path}")
    print(f"[v40-benchmax] wrote ablation table to {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
