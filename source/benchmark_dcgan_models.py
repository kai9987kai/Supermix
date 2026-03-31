from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from dcgan_image_model import DCGANImageEngine, DCGAN_SPECS, _find_generator_weights


DEFAULT_MODELS_DIR = Path(r"C:\Users\kai99\Desktop\models")


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cleaned.split("-") if part)
    return cooked[:72] or "artifact"


def _extract_zip_once(zip_path: Path, extraction_root: Path) -> Path:
    target = extraction_root / _safe_slug(zip_path.stem)
    marker = target / ".extract_complete"
    if marker.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
    marker.write_text(zip_path.name, encoding="utf-8")
    return target


def benchmark_model(*, key: str, zip_path: Path, output_root: Path, extraction_root: Path, sample_count: int, seed: int) -> Dict[str, object]:
    spec = DCGAN_SPECS[key]
    extracted_dir = _extract_zip_once(zip_path, extraction_root)
    weights_path = _find_generator_weights(extracted_dir, spec.preferred_weights)
    engine = DCGANImageEngine(key=key, weights_path=weights_path)
    metrics = engine.benchmark(sample_count=sample_count, seed=seed)

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_root / f"{key}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sample_path = artifact_dir / f"{key}_benchmark_samples.png"
    summary_path = artifact_dir / f"{key}_benchmark_summary.json"
    engine.render_to_path(prompt=f"benchmark-grid::{key}", output_path=sample_path, sample_count=min(sample_count, 16), seed=seed + 17)
    payload = {
        "model_key": key,
        "label": spec.label,
        "zip_path": str(zip_path),
        "weights_path": str(weights_path),
        "sample_count": int(sample_count),
        "seed": int(seed),
        "specialist_score": metrics["specialist_score"],
        "metrics": metrics,
        "artifacts": {
            "sample_grid_png": str(sample_path),
        },
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "sample_path": str(sample_path),
        **payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local DCGAN specialist model zips with a small image-generation quality heuristic.")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_key", action="append", default=[])
    parser.add_argument("--sample_count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    extraction_root = output_dir / "_dcgan_extract"
    requested = {item.strip() for item in args.model_key if item.strip()}

    zip_map: Dict[str, Path] = {
        "dcgan_mnist_model": models_dir / "dcgan_mnist_model.zip",
        "dcgan_v2_in_progress": models_dir / "dcgan_v2_in_progress.zip",
    }
    results: List[Dict[str, object]] = []
    for key, zip_path in zip_map.items():
        if requested and key not in requested:
            continue
        if not zip_path.exists():
            continue
        results.append(
            benchmark_model(
                key=key,
                zip_path=zip_path,
                output_root=output_dir,
                extraction_root=extraction_root,
                sample_count=int(args.sample_count),
                seed=int(args.seed),
            )
        )
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
