#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v40_benchmax_common import json_dump, load_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the v40_benchmax source-of-truth manifest and a compact ablation table.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest_out", default="")
    parser.add_argument("--table_out", default="")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    manifest_out = Path(args.manifest_out).resolve() if args.manifest_out else output_dir / "v40_benchmax_manifest.json"
    table_out = Path(args.table_out).resolve() if args.table_out else output_dir / "v40_benchmax_ablation_table.json"

    json_dump(manifest_out, manifest)
    table = {
        "family": manifest["family"],
        "objective": manifest["objective"],
        "baselines": manifest["baselines"],
        "ablation_matrix": manifest["ablation_matrix"],
        "data_recipes": manifest["data_recipes"],
        "head_recipes": manifest["head_recipes"],
        "distillation": manifest["distillation"],
        "research_pack": manifest.get("research_pack", {}),
        "soup": manifest["soup"],
        "report": manifest["report"],
    }
    table_out.write_text(json.dumps(table, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[v40-benchmax] wrote manifest to {manifest_out}")
    print(f"[v40-benchmax] wrote table to {table_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
