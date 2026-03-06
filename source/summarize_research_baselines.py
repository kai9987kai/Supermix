import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float))


def _collect_numeric_metrics(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = set()
    for row in rows:
        for key, value in row.items():
            if _is_number(value):
                keys.add(key)

    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(keys):
        vals = [float(row[key]) for row in rows if _is_number(row.get(key))]
        if not vals:
            continue
        out[key] = {
            "mean": float(mean(vals)),
            "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
            "min": float(min(vals)),
            "max": float(max(vals)),
        }
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate multiple baseline benchmark_results.json files.")
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or benchmark_results.json files.",
    )
    ap.add_argument("--output", required=True, help="Output summary JSON path.")
    return ap.parse_args()


def _load_benchmark_json(path_like: str) -> Dict[str, object]:
    p = Path(path_like).expanduser().resolve()
    if p.is_dir():
        p = p / "benchmark_results.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing benchmark JSON: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()

    loaded = [_load_benchmark_json(path_like) for path_like in args.runs]
    base_rows = [row.get("base", {}) for row in loaded if isinstance(row.get("base"), dict)]
    tuned_rows = [row.get("tuned", {}) for row in loaded if isinstance(row.get("tuned"), dict)]
    delta_rows = [row.get("delta_tuned_minus_base", {}) for row in loaded if isinstance(row.get("delta_tuned_minus_base"), dict)]

    summary = {
        "run_count": int(len(loaded)),
        "run_inputs": [str(Path(p).expanduser().resolve()) for p in args.runs],
        "base": _collect_numeric_metrics(base_rows),
        "tuned": _collect_numeric_metrics(tuned_rows),
        "delta_tuned_minus_base": _collect_numeric_metrics(delta_rows),
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
