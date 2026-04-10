#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from v40_benchmax_common import csv_write, json_dump, load_manifest
from v40_benchmax_core import build_promotion_report, extract_model_row


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_candidate_model(summary: Mapping[str, Any], explicit_name: str, manifest: Mapping[str, Any]) -> str:
    if explicit_name.strip():
        return explicit_name.strip()
    summary_rows = summary.get("summary_rows")
    if isinstance(summary_rows, list) and len(summary_rows) == 1:
        model = str((summary_rows[0] or {}).get("model") or "").strip()
        if model:
            return model
    preferred = [
        str(manifest.get("family") or "").strip(),
        str(manifest.get("experiment_family") or "").strip(),
        str((manifest.get("report") or {}).get("candidate_model") or "").strip(),
    ]
    if isinstance(summary_rows, list):
        available = {str((row or {}).get("model") or "").strip() for row in summary_rows if isinstance(row, Mapping)}
        for name in preferred:
            if name and name in available:
                return name
    if preferred[0]:
        return preferred[0]
    raise ValueError("Could not resolve candidate model name. Pass --candidate_model explicitly.")


def _single_model_summary(summary: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    row = extract_model_row(summary, model_name)
    return {
        "model": model_name,
        "summary_rows": [row],
    }


def _best_ablation(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    def _rank_value(row: Mapping[str, Any]) -> float:
        for key in ("overall_exact", "candidate_overall_exact", "selected_rows", "source_rows"):
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    return max(rows, key=_rank_value)


def _ablation_signal(best_ablation: Mapping[str, Any]) -> str:
    if not best_ablation:
        return "No ablation signal is available yet."
    ablation_id = str(best_ablation.get("ablation_id") or best_ablation.get("id") or "unknown")
    data_recipe = str(best_ablation.get("data_recipe") or best_ablation.get("data_family") or "unknown")
    head_recipe = str(best_ablation.get("head_recipe") or best_ablation.get("recipe_family") or "unknown")
    return f"Best ablation so far: `{ablation_id}` using data `{data_recipe}` and recipe/head `{head_recipe}`."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare a v40_benchmax candidate against v33_final and v39_final and emit a promotion gate report."
    )
    parser.add_argument("--candidate_summary", required=True)
    parser.add_argument("--v33_summary", required=True)
    parser.add_argument("--v39_summary", required=True)
    parser.add_argument("--candidate_model", default="")
    parser.add_argument("--ablation_summary", default="")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    candidate_summary = _load_json(Path(args.candidate_summary).resolve())
    candidate_model = _resolve_candidate_model(candidate_summary, str(args.candidate_model), manifest)
    v33_summary = _load_json(Path(args.v33_summary).resolve())
    v39_summary = _load_json(Path(args.v39_summary).resolve())

    comparison = build_promotion_report(
        benchmark_summary={
            "summary_rows": [
                extract_model_row(v33_summary, "v33_final"),
                extract_model_row(v39_summary, "v39_final"),
                extract_model_row(candidate_summary, candidate_model),
            ]
        },
        candidate_model=candidate_model,
        leader_models=["v33_final", "v39_final"],
        manifest=manifest,
    )

    candidate_row = comparison["candidate"]
    v33_row = comparison["leaders"]["v33_final"]
    v39_row = comparison["leaders"]["v39_final"]

    ablation_rows: List[Dict[str, Any]] = []
    ablation_path = Path(args.ablation_summary).resolve() if args.ablation_summary else None
    if ablation_path and ablation_path.exists():
        payload = _load_json(ablation_path)
        for row in payload.get("summary_rows") or []:
            if isinstance(row, Mapping):
                ablation_rows.append(dict(row))
    best_ablation = _best_ablation(ablation_rows)

    delta_vs_v33 = float(comparison["candidate_vs_leaders"]["v33_final"]["overall_exact_delta"])
    delta_vs_v39 = float(comparison["candidate_vs_leaders"]["v39_final"]["overall_exact_delta"])
    promoted = bool(comparison["promotion_gate"]["promote"])
    winner = candidate_model if promoted else str(comparison["best_leader"])
    ablation_signal = _ablation_signal(best_ablation)

    md_lines = [
        "# v40_benchmax report",
        "",
        f"- winning experiment: `{winner}`",
        f"- candidate: `{candidate_model}`",
        f"- promoted: `{promoted}`",
        f"- delta vs v33_final: `{delta_vs_v33:.6f}`",
        f"- delta vs v39_final: `{delta_vs_v39:.6f}`",
        "",
        "## Baselines",
        f"- v33_final: `{float(v33_row.get('overall_exact') or 0.0):.6f}`",
        f"- v39_final: `{float(v39_row.get('overall_exact') or 0.0):.6f}`",
        "",
        "## Attribution",
    ]
    for item in comparison.get("attribution", []):
        md_lines.append(f"- {item}")
    md_lines.extend(
        [
            "",
            "## Ablation Signal",
            f"- {ablation_signal}",
            "",
            "## Recommendation",
            f"- {comparison.get('recommended_next_run', '')}",
        ]
    )

    summary = {
        "family": manifest["family"],
        "winner": winner,
        "candidate_model": candidate_model,
        "candidate": _single_model_summary(candidate_summary, candidate_model),
        "baselines": {
            "v33_final": _single_model_summary(v33_summary, "v33_final"),
            "v39_final": _single_model_summary(v39_summary, "v39_final"),
        },
        "comparison": comparison,
        "delta_vs_v33_final": round(delta_vs_v33, 6),
        "delta_vs_v39_final": round(delta_vs_v39, 6),
        "ablation_rows": ablation_rows,
        "best_ablation": best_ablation,
        "ablation_signal": ablation_signal,
        "recommendation": comparison.get("recommended_next_run", ""),
    }
    json_dump(output_dir / "v40_benchmax_report.json", summary)
    csv_write(
        output_dir / "v40_benchmax_report.csv",
        [
            {
                "winner": winner,
                "candidate": candidate_model,
                "candidate_overall_exact": candidate_row.get("overall_exact"),
                "v33_final": v33_row.get("overall_exact"),
                "v39_final": v39_row.get("overall_exact"),
                "delta_vs_v33_final": delta_vs_v33,
                "delta_vs_v39_final": delta_vs_v39,
                "best_leader": comparison.get("best_leader"),
                "promoted": promoted,
                "attribution": "; ".join(str(item) for item in comparison.get("attribution", [])),
                "best_ablation": best_ablation.get("ablation_id") or best_ablation.get("id") or "",
            }
        ],
        [
            "winner",
            "candidate",
            "candidate_overall_exact",
            "v33_final",
            "v39_final",
            "delta_vs_v33_final",
            "delta_vs_v39_final",
            "best_leader",
            "promoted",
            "attribution",
            "best_ablation",
        ],
    )
    (output_dir / "v40_benchmax_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[v40-benchmax] wrote report to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
