import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_browser_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    buckets = meta.get("buckets") or {}
    label_priors = meta.get("label_priors") or {}

    candidates: List[Dict[str, Any]] = []
    for label, rows in buckets.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            candidates.append(
                {
                    "label": int(label) if str(label).isdigit() else str(label),
                    "text": text,
                    "count": int(row.get("count", 1) or 1),
                }
            )

    return {
        "format": "champion-browser-chat-meta-v1",
        "source_meta": str(meta.get("fine_tuned_weights") or meta.get("base_weights") or ""),
        "source_meta_file": str(meta.get("fine_tuned_weights") or ""),
        "created_at": meta.get("created_at"),
        "feature_mode": "browser_lexical_retrieval",
        "note": "Static browser mode uses metadata retrieval only; PyTorch .pth weights are not executed in-browser.",
        "model_size": meta.get("model_size", "unknown"),
        "num_classes": int(meta.get("num_classes", len(buckets) or 0)),
        "label_priors": label_priors,
        "candidates": candidates,
        "stats": {
            "candidate_count": len(candidates),
            "bucket_count": len(buckets),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export a lightweight browser-friendly chat metadata file.")
    ap.add_argument("--meta", required=True, help="Input chat_model_meta*.json")
    ap.add_argument("--out", required=True, help="Output browser JSON file")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_path = Path(args.out)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    browser_meta = build_browser_meta(meta)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(browser_meta, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Saved browser meta: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Candidates: {browser_meta['stats']['candidate_count']}")


if __name__ == "__main__":
    main()

