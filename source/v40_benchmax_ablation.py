from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

try:
    from v40_benchmax_core import ablation_rows, build_ablation_table, default_output_root, load_json, load_manifest, write_csv
except ImportError:  # pragma: no cover
    from .v40_benchmax_core import ablation_rows, build_ablation_table, default_output_root, load_json, load_manifest, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v40_benchmax 2x2 ablation manifest and optional comparison table.")
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "v40_benchmax_manifest.json"))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--result_summary", action="append", default=[], help="Optional label=summary.json path for observed ablation results.")
    return parser.parse_args()


def _parse_labeled_paths(items: List[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected label=path, got {item!r}")
        label, path = item.split("=", 1)
        result[label.strip()] = Path(path.strip()).resolve()
    return result


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_root = Path(args.output_dir).resolve() if args.output_dir.strip() else default_output_root(manifest)
    ablation_dir = output_root / "ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)

    planned = ablation_rows(manifest)
    results = _parse_labeled_paths(list(args.result_summary))
    observed = []
    for entry in planned:
        row = dict(entry)
        label = row["id"]
        summary_path = results.get(label)
        if summary_path and summary_path.exists():
            summary = load_json(summary_path)
            table = build_ablation_table(manifest=manifest, result_rows=summary.get("summary_rows") or [])
            match = next((item for item in table if item.get("id") == label or item.get("model") == label), None)
            if match:
                row.update(match)
        observed.append(row)

    json_path = ablation_dir / "v40_benchmax_ablation_matrix.json"
    csv_path = ablation_dir / "v40_benchmax_ablation_matrix.csv"
    md_path = ablation_dir / "v40_benchmax_ablation_matrix.md"
    json_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "manifest": manifest,
                "output_dir": str(output_root),
                "ablation_rows": observed,
                "result_summaries": {key: str(path) for key, path in results.items()},
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    write_csv(
        csv_path,
        observed,
        ["id", "data_family", "recipe_family", "status", "overall_exact", "avg_token_f1", "avg_char_similarity", "avg_gen_seconds", "model_seconds"],
    )
    lines = ["# v40_benchmax 2x2 Ablation", "", "| ID | Data | Recipe | Status | Overall exact |", "|---|---|---|---|---|"]
    for row in observed:
        lines.append(
            f"| {row.get('id','')} | {row.get('data_family','')} | {row.get('recipe_family','')} | {row.get('status','planned')} | "
            f"{float(row.get('overall_exact') or 0.0):.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(str(json_path))
    print(str(csv_path))
    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
