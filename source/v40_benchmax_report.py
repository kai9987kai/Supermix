from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

try:
    from v40_benchmax_core import build_promotion_report, load_json, load_manifest, write_csv
except ImportError:  # pragma: no cover
    from .v40_benchmax_core import build_promotion_report, load_json, load_manifest, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the v40_benchmax comparison report and promotion gate.")
    parser.add_argument("--benchmark_summary", required=True)
    parser.add_argument("--candidate_model", default="v40_benchmax")
    parser.add_argument("--leader_model", action="append", default=["v33_final", "v39_final"])
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "v40_benchmax_manifest.json"))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    output_root = Path(args.output_dir).resolve() if args.output_dir.strip() else (Path(__file__).resolve().parent.parent / "output" / "v40_benchmax" / "reports").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    benchmark_summary = load_json(args.benchmark_summary)
    report = build_promotion_report(
        benchmark_summary=benchmark_summary,
        candidate_model=args.candidate_model,
        leader_models=args.leader_model,
        manifest=manifest,
    )

    json_path = output_root / "v40_benchmax_report.json"
    md_path = output_root / "v40_benchmax_report.md"
    csv_path = output_root / "v40_benchmax_report.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    lines: List[str] = [
        "# v40_benchmax Promotion Report",
        "",
        f"- Candidate: `{report['candidate_model']}`",
        f"- Winner: `{report['candidate_model'] if report['promotion_gate']['promote'] else report['best_leader']}`",
        f"- Promote: `{report['promotion_gate']['promote']}`",
        f"- Recommended next run: {report['recommended_next_run']}",
        "",
        "## Deltas",
    ]
    for name, delta in report["candidate_vs_leaders"].items():
        lines.append(f"- vs `{name}`: overall_exact {delta['overall_exact_delta']:+.4f}")
    lines.extend(["", "## Attribution"])
    for item in report["attribution"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Benchmarks"])
    candidate = report["candidate"]
    lines.append("| benchmark | candidate |")
    lines.append("|---|---|")
    for bench, score in sorted(dict(candidate.get("benchmarks") or {}).items()):
        lines.append(f"| {bench} | {float(score):.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = [
        {
            "candidate_model": report["candidate_model"],
            "best_leader": report["best_leader"],
            "promote": report["promotion_gate"]["promote"],
            "candidate_score": float(report["promotion_gate"]["candidate_score"]),
            "best_leader_score": float(report["promotion_gate"]["best_leader_score"]),
            "overall_delta": float(report["promotion_gate"]["candidate_score"]) - float(report["promotion_gate"]["best_leader_score"]),
            "recommended_next_run": report["recommended_next_run"],
        }
    ]
    write_csv(csv_path, rows, ["candidate_model", "best_leader", "promote", "candidate_score", "best_leader_score", "overall_delta", "recommended_next_run"])

    print(str(json_path))
    print(str(md_path))
    print(str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
