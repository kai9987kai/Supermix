#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    family: str
    filename_tokens: Sequence[str]
    common_row_key: Optional[str]
    virtual: bool = False
    note: str = ""
    recipe_eval_accuracy: Optional[float] = None
    score_source: str = "common"
    specialist_summary_path: Optional[str] = None
    specialist_metric_key: Optional[str] = None
    specialist_metric_label: str = ""


ARTIFACT_SPECS: Sequence[ArtifactSpec] = (
    ArtifactSpec(
        key="qwen_v28",
        label="qwen_v28",
        family="qwen",
        filename_tokens=("qwen_supermix_enhanced_v28_cloud_plus_runpod_budget_final_adapter",),
        common_row_key="qwen_v28",
        note="LoRA adapter benchmarked with the Qwen base model.",
    ),
    ArtifactSpec(
        key="qwen_v30",
        label="qwen_v30_experimental",
        family="qwen",
        filename_tokens=(
            "qwen_supermix_enhanced_v30_anchor_refresh_20260326_experimental_adapter",
            "qwen_supermix_enhanced_v30_anchor_refresh_20260326_experimental_bundle",
        ),
        common_row_key="qwen_v30",
        note="Experimental adapter benchmarked with the Qwen base model.",
    ),
    ArtifactSpec(
        key="v30_lite",
        label="v30_lite_fp16",
        family="champion",
        filename_tokens=("champion_v30_lite_student_fp16_bundle_20260326",),
        common_row_key="v30_lite",
        note="Mapped to the existing v30_lite common-benchmark row.",
    ),
    ArtifactSpec(
        key="v31_final",
        label="v31_final",
        family="champion",
        filename_tokens=(
            "champion_v31_hybrid_plus_refresh_final_model_20260326",
            "champion_v31_hybrid_plus_refresh_bundle_20260326",
        ),
        common_row_key="v31_final",
    ),
    ArtifactSpec(
        key="v31_image_variant",
        label="v31_image_variant",
        family="wrapper",
        filename_tokens=("champion_v31_image_variant_bundle_20260326",),
        common_row_key="v31_final",
        note="Wrapper artifact reusing the v31_final text checkpoint.",
    ),
    ArtifactSpec(
        key="v32_final",
        label="v32_final",
        family="champion",
        filename_tokens=(
            "champion_v32_omnifuse_final_model_20260326",
            "champion_v32_omnifuse_bundle_20260326",
        ),
        common_row_key="v32_final",
    ),
    ArtifactSpec(
        key="v33_final",
        label="v33_final",
        family="champion",
        filename_tokens=(
            "champion_v33_frontier_full_model_20260326",
            "champion_v33_frontier_full_bundle_20260326",
        ),
        common_row_key="v33_final",
    ),
    ArtifactSpec(
        key="v34_final",
        label="v34_final",
        family="champion",
        filename_tokens=("champion_v34_frontier_plus_full_model_20260326",),
        common_row_key="v34_stage2",
        note="Official v34 artifact chosen from stage2.",
    ),
    ArtifactSpec(
        key="v35_final",
        label="v35_final",
        family="champion",
        filename_tokens=("champion_v35_collective_allteachers_full_model_20260326",),
        common_row_key="v35_stage2",
        note="Mapped to the stronger v35_stage2 common-benchmark row.",
    ),
    ArtifactSpec(
        key="v36_native",
        label="v36_native",
        family="native_image",
        filename_tokens=("champion_v36_native_image_single_checkpoint_model_20260327",),
        common_row_key="v36_native",
    ),
    ArtifactSpec(
        key="v37_native_lite",
        label="v37_native_lite",
        family="native_image",
        filename_tokens=("champion_v37_native_image_lite_single_checkpoint_model_20260327",),
        common_row_key="v37_native_lite",
    ),
    ArtifactSpec(
        key="v38_native_xlite",
        label="v38_native_xlite",
        family="native_image",
        filename_tokens=(
            "champion_v38_native_image_xlite_single_checkpoint_model_20260327",
            "champion_v38_native_image_xlite_single_checkpoint_model_fp16_20260327",
        ),
        common_row_key="v38_native_xlite",
        note="Represents the v38 native-image line; fp16 zip is the same model family.",
    ),
    ArtifactSpec(
        key="v38_native_xlite_fp16",
        label="v38_native_xlite_fp16",
        family="native_image",
        filename_tokens=("champion_v38_native_image_xlite_single_checkpoint_model_fp16_20260327",),
        common_row_key="v38_native_xlite",
        note="Half-precision package of the same v38 XLite checkpoint.",
    ),
    ArtifactSpec(
        key="v39_final",
        label="v39_final",
        family="champion",
        filename_tokens=("champion_v39_frontier_reasoning_plus_full_model_20260327",),
        common_row_key="v39_final",
        recipe_eval_accuracy=0.0549,
        score_source="recipe_eval_only",
        note="Finished v39 artifact. Recipe holdout score is preserved alongside the later common-benchmark score.",
    ),
    ArtifactSpec(
        key="science_vision_micro_v1",
        label="science_vision_micro_v1",
        family="vision",
        filename_tokens=("supermix_science_image_recognition_micro_v1_20260327",),
        common_row_key=None,
        score_source="specialist_only",
        note="Specialist upload-image recognition model. Common text benchmarks are not applicable.",
        specialist_summary_path="output/supermix_science_image_recognition_micro_v1_20260327/science_image_recognition_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="vision val",
    ),
    ArtifactSpec(
        key="dcgan_mnist_model",
        label="dcgan_mnist_model",
        family="gan",
        filename_tokens=("dcgan_mnist_model.zip", "dcgan_mnist_model"),
        common_row_key=None,
        score_source="specialist_only",
        note="Unconditional grayscale DCGAN trained on MNIST digits. Specialist score comes from the local GAN generation benchmark.",
        specialist_summary_path="output/dcgan_mnist_model_20260331/dcgan_mnist_model_benchmark_summary.json",
        specialist_metric_key="specialist_score",
        specialist_metric_label="gan score",
    ),
    ArtifactSpec(
        key="dcgan_v2_in_progress",
        label="dcgan_v2_in_progress",
        family="gan",
        filename_tokens=("dcgan_v2_in_progress.zip", "dcgan_v2_in_progress"),
        common_row_key=None,
        score_source="specialist_only",
        note="Unconditional RGB DCGAN v2 trained on CIFAR-style images. Specialist score comes from the local GAN generation benchmark.",
        specialist_summary_path="output/dcgan_v2_in_progress_20260331/dcgan_v2_in_progress_benchmark_summary.json",
        specialist_metric_key="specialist_score",
        specialist_metric_label="gan score",
    ),
    ArtifactSpec(
        key="omni_collective_v1",
        label="omni_collective_v1",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v1_20260327",),
        common_row_key="omni_collective_v1",
        score_source="specialist_only",
        note="Local fused assistant model. Common benchmark row comes from the local add-on sweep.",
        specialist_summary_path="output/supermix_omni_collective_v1_20260327/omni_collective_v1_summary.json",
        specialist_metric_key="val_response_accuracy",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v2",
        label="omni_collective_v2",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v2_frontier_20260328",),
        common_row_key="omni_collective_v2",
        score_source="specialist_only",
        note="Frontier omni v2 model with common-benchmark add-on results.",
        specialist_summary_path="output/supermix_omni_collective_v2_frontier_20260328/omni_collective_v2_frontier_summary.json",
        specialist_metric_key="best_stage2.score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v3",
        label="omni_collective_v3",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v3_frontier_20260329",),
        common_row_key="omni_collective_v3",
        score_source="specialist_only",
        note="Frontier omni v3 model with common-benchmark add-on results.",
        specialist_summary_path="output/supermix_omni_collective_v3_frontier_20260329/omni_collective_v3_frontier_summary.json",
        specialist_metric_key="best_stage2.score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v4",
        label="omni_collective_v4",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v4_frontier_20260329",),
        common_row_key="omni_collective_v4",
        score_source="specialist_only",
        note="Frontier omni v4 model with expanded sparse-routing and common-benchmark add-on results.",
        specialist_summary_path="output/supermix_omni_collective_v4_frontier_20260329/omni_collective_v4_frontier_summary.json",
        specialist_metric_key="stage2_val.score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v5",
        label="omni_collective_v5",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v5_frontier_20260330",),
        common_row_key="omni_collective_v5",
        score_source="specialist_only",
        note="Frontier omni v5 continuation with coding, OpenSCAD, prompt-understanding deltas, and common-benchmark add-on results.",
        specialist_summary_path="output/supermix_omni_collective_v5_frontier_20260330/omni_collective_v5_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v6",
        label="omni_collective_v6",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v6_frontier_20260331",),
        common_row_key="omni_collective_v6",
        score_source="specialist_only",
        note="All-model distilled omni v6 frontier with forced small-Qwen teachers, heavier conversation/math/protein grounding, and longer deliberation.",
        specialist_summary_path="output/supermix_omni_collective_v6_frontier_20260331/omni_collective_v6_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v7",
        label="omni_collective_v7",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v7_frontier_20260403",),
        common_row_key="omni_collective_v7",
        score_source="specialist_only",
        note="All-model distilled omni v7 frontier with a larger teacher league, broader conversation/math/protein mix, and longer deliberation.",
        specialist_summary_path="output/supermix_omni_collective_v7_frontier_20260403/omni_collective_v7_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v8",
        label="omni_collective_v8",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v8_frontier_20260408",),
        common_row_key="omni_collective_v8",
        score_source="specialist_only",
        note="Final omni v8 frontier with all-model distillation, broader multimodal grounding, denser conversation data, and longer deliberation.",
        specialist_summary_path="output/supermix_omni_collective_v8_frontier_20260408/omni_collective_v8_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v42",
        label="omni_collective_v42",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v42_frontier_", "supermix_omni_collective_v42_smoke_"),
        common_row_key="omni_collective_v42",
        score_source="specialist_only",
        note="V42 continuation with benchmark-bridge replay, verifier-repair supervision, and budget-aware route control.",
        specialist_summary_path="output/supermix_omni_collective_v42_smoke_20260410_173006/omni_collective_v42_smoke_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="omni_collective_v41",
        label="omni_collective_v41",
        family="fusion",
        filename_tokens=("supermix_omni_collective_v41_frontier_",),
        common_row_key="omni_collective_v41",
        score_source="specialist_only",
        note="Frontier omni v41 continuation with hidden planning, communication-polish, uncertainty, and code-repair upgrades.",
        specialist_summary_path="output/supermix_omni_collective_v41_frontier_20260410_015540/omni_collective_v41_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="v40_benchmax",
        label="v40_benchmax",
        family="fusion",
        filename_tokens=("supermix_v40_benchmax_v40_v39data_v39recipe_20260330",),
        common_row_key="v40_benchmax",
        score_source="specialist_only",
        note="Benchmark-maximization v40 continuation built from the v39-style benchmix recipe plus support and protein-folding rows.",
        specialist_summary_path="output/supermix_v40_benchmax_v40_v39data_v39recipe_20260330/omni_collective_v40_benchmax_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    ArtifactSpec(
        key="math_equation_micro_v1",
        label="math_equation_micro_v1",
        family="math",
        filename_tokens=("supermix_math_equation_micro_v1_20260327",),
        common_row_key="math_equation_micro_v1",
        score_source="specialist_only",
        note="Math specialist with exact symbolic routing plus a local add-on common-benchmark run.",
        specialist_summary_path="output/supermix_math_equation_micro_v1_20260327/math_equation_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="math val",
    ),
    ArtifactSpec(
        key="protein_folding_micro_v1",
        label="protein_folding_micro_v1",
        family="protein",
        filename_tokens=("supermix_protein_folding_micro_v1_20260331",),
        common_row_key="protein_folding_micro_v1",
        score_source="specialist_only",
        note="Protein-folding specialist with structure-prediction concept routing plus a local add-on common-benchmark run.",
        specialist_summary_path="output/supermix_protein_folding_micro_v1_20260331/protein_folding_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="protein val",
    ),
    ArtifactSpec(
        key="three_d_generation_micro_v1",
        label="three_d_generation_micro_v1",
        family="3d",
        filename_tokens=("supermix_3d_generation_micro_v1_20260403",),
        common_row_key="three_d_generation_micro_v1",
        score_source="specialist_only",
        note="Small OpenSCAD / CAD generation specialist with a local add-on common-benchmark run.",
        specialist_summary_path="output/supermix_3d_generation_micro_v1_20260403/three_d_generation_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="3d val",
    ),
)

VIRTUAL_ARTIFACT_SPECS: Sequence[ArtifactSpec] = (
    ArtifactSpec(
        key="auto_collective_loop",
        label="auto_collective_loop_s5",
        family="router",
        filename_tokens=(),
        common_row_key="auto_collective_loop",
        virtual=True,
        score_source="runtime",
        note="Prompt-aware auto router benchmarked on a reduced 5-per-benchmark sampled suite with collective loop mode enabled, a two-step loop budget, and a benchmark-focused consultant subset.",
    ),
)


FAMILY_COLORS: Dict[str, str] = {
    "qwen": "#d97706",
    "champion": "#2563eb",
    "native_image": "#15803d",
    "wrapper": "#6b7280",
    "router": "#475569",
    "vision": "#7c3aed",
    "gan": "#b91c1c",
    "fusion": "#db2777",
    "math": "#0f766e",
    "protein": "#6d28d9",
    "3d": "#0891b2",
}

BENCHMARK_ORDER: Sequence[str] = (
    "arc_challenge",
    "boolq",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "piqa",
)

BENCHMARK_LABELS: Dict[str, str] = {
    "arc_challenge": "ARC",
    "boolq": "BoolQ",
    "gsm8k": "GSM8K",
    "hellaswag": "Hella",
    "mmlu": "MMLU",
    "piqa": "PIQA",
}

FAMILY_DESCRIPTIONS: Dict[str, str] = {
    "qwen": "Qwen adapter scored on the common benchmark sweep",
    "champion": "Champion-family text model",
    "native_image": "Native image-capable checkpoint",
    "wrapper": "Wrapper or alias artifact",
    "router": "Prompt-routed multimodel runtime benchmark",
    "vision": "Vision specialist artifact",
    "gan": "GAN image-generation specialist",
    "fusion": "Omni fused multimodal model",
    "math": "Math specialist model",
    "protein": "Protein-folding specialist model",
}


def _resolve_default_common_summary() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "output"
    candidates = sorted(output_dir.glob("benchmark_all_models_common_plus_summary_*.json"))
    if candidates:
        return candidates[-1]
    return output_dir / "benchmark_all_models_common_plus_summary_20260329.json"


def _score_for_sort(common_score: Optional[float], recipe_score: Optional[float]) -> float:
    if common_score is not None:
        return common_score
    if recipe_score is not None:
        return recipe_score
    return -1.0


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _extract_nested(payload: Dict[str, object], dotted_key: str) -> Optional[float]:
    current: object = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return _safe_float(current)


def _load_specialist_metric(repo_root: Path, spec: ArtifactSpec) -> Tuple[Optional[float], str]:
    if not spec.specialist_summary_path or not spec.specialist_metric_key:
        return None, spec.specialist_metric_label
    summary_path = (repo_root / spec.specialist_summary_path).resolve()
    if not summary_path.exists():
        return None, spec.specialist_metric_label
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    metric = _extract_nested(payload, spec.specialist_metric_key)
    if metric is None and isinstance(payload.get("meta"), dict):
        metric = _extract_nested(payload["meta"], spec.specialist_metric_key)
    if metric is None and isinstance(payload.get("history"), list) and payload["history"]:
        last_history = payload["history"][-1]
        if isinstance(last_history, dict):
            metric = _extract_nested(last_history, spec.specialist_metric_key)
    return metric, spec.specialist_metric_label


def _load_common_rows(summary_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("summary_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"summary_rows missing in {summary_path}")
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("model"):
            out[str(row["model"])] = row
    return out


def _match_token_index(spec: ArtifactSpec, path: Path) -> Optional[int]:
    for idx, token in enumerate(spec.filename_tokens):
        if token in path.name:
            return idx
    return None


def _candidate_rank(spec: ArtifactSpec, path: Path) -> Optional[Tuple[int, int, int, float]]:
    token_index = _match_token_index(spec, path)
    if token_index is None:
        return None

    name = path.name.lower()
    is_duplicate_copy = 1 if " (1)" in path.name else 0
    is_bundle = 1 if "bundle" in name else 0
    # Prefer the intended token order first, then real model archives over bundles,
    # then the non-duplicate download, and finally the most recent file.
    return (token_index, is_bundle, is_duplicate_copy, -path.stat().st_mtime)


def discover_artifacts(models_dir: Path) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    ranks: Dict[str, Tuple[int, int, int, float]] = {}
    files = [p for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    for spec in ARTIFACT_SPECS:
        for path in files:
            rank = _candidate_rank(spec, path)
            if rank is None:
                continue
            current_rank = ranks.get(spec.key)
            if current_rank is None or rank < current_rank:
                found[spec.key] = path
                ranks[spec.key] = rank
    return found


def build_zip_inventory(models_dir: Path, artifacts: Dict[str, Path]) -> Dict[str, List[str]]:
    all_zip_names = sorted(p.name for p in models_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip")
    selected = {path.name for path in artifacts.values()}
    alternate_matches: List[str] = []
    unmatched: List[str] = []

    for name in all_zip_names:
        if name in selected:
            continue
        matched = any(any(token in name for token in spec.filename_tokens) for spec in ARTIFACT_SPECS)
        if matched:
            alternate_matches.append(name)
        else:
            unmatched.append(name)

    return {
        "selected_zip_files": sorted(selected),
        "alternate_package_zip_files": alternate_matches,
        "unmatched_zip_files": unmatched,
    }


def build_rows(models_dir: Path, common_summary_path: Path, repo_root: Path) -> List[Dict[str, object]]:
    common_rows = _load_common_rows(common_summary_path)
    artifacts = discover_artifacts(models_dir)
    rows: List[Dict[str, object]] = []
    for spec in ARTIFACT_SPECS:
        path = artifacts.get(spec.key)
        if path is None:
            continue
        common_row = common_rows.get(spec.common_row_key) if spec.common_row_key else None
        common_score = _safe_float(common_row.get("overall_exact")) if common_row else None
        recipe_score = _safe_float(spec.recipe_eval_accuracy)
        specialist_score, specialist_label = _load_specialist_metric(repo_root, spec)
        row = {
            "model_key": spec.key,
            "label": spec.label,
            "family": spec.family,
            "zip_path": str(path),
            "zip_name": path.name,
            "zip_size_bytes": path.stat().st_size,
            "common_benchmark_model": spec.common_row_key,
            "common_overall_exact": common_score,
            "recipe_eval_accuracy": recipe_score,
            "specialist_metric_value": specialist_score,
            "specialist_metric_label": specialist_label,
            "score_source": spec.score_source if common_score is None else ("common_alias" if spec.common_row_key and spec.common_row_key != spec.key else "common"),
            "note": spec.note,
        }
        if common_row and isinstance(common_row.get("benchmarks"), dict):
            row["per_benchmark"] = common_row["benchmarks"]
        rows.append(row)
    for spec in VIRTUAL_ARTIFACT_SPECS:
        common_row = common_rows.get(spec.common_row_key) if spec.common_row_key else None
        if common_row is None:
            continue
        common_score = _safe_float(common_row.get("overall_exact"))
        recipe_score = _safe_float(spec.recipe_eval_accuracy)
        specialist_score, specialist_label = _load_specialist_metric(repo_root, spec)
        row = {
            "model_key": spec.key,
            "label": spec.label,
            "family": spec.family,
            "zip_path": "",
            "zip_name": "",
            "zip_size_bytes": 0,
            "common_benchmark_model": spec.common_row_key,
            "common_overall_exact": common_score,
            "recipe_eval_accuracy": recipe_score,
            "specialist_metric_value": specialist_score,
            "specialist_metric_label": specialist_label,
            "score_source": spec.score_source,
            "note": spec.note,
        }
        if isinstance(common_row.get("benchmarks"), dict):
            row["per_benchmark"] = common_row["benchmarks"]
        rows.append(row)
    rows.sort(
        key=lambda item: (
            _score_for_sort(_safe_float(item.get("common_overall_exact")), _safe_float(item.get("recipe_eval_accuracy"))),
            str(item["label"]).lower(),
        ),
        reverse=True,
    )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "family",
        "zip_name",
        "zip_size_bytes",
        "common_benchmark_model",
        "common_overall_exact",
        "recipe_eval_accuracy",
        "specialist_metric_label",
        "specialist_metric_value",
        "score_source",
        "note",
    ] + [f"benchmark_{name}" for name in BENCHMARK_ORDER]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {
                "label": row["label"],
                "family": row["family"],
                "zip_name": row["zip_name"],
                "zip_size_bytes": row["zip_size_bytes"],
                "common_benchmark_model": row.get("common_benchmark_model") or "",
                "common_overall_exact": "" if row.get("common_overall_exact") is None else f"{float(row['common_overall_exact']):.6f}",
                "recipe_eval_accuracy": "" if row.get("recipe_eval_accuracy") is None else f"{float(row['recipe_eval_accuracy']):.6f}",
                "specialist_metric_label": row.get("specialist_metric_label") or "",
                "specialist_metric_value": "" if row.get("specialist_metric_value") is None else f"{float(row['specialist_metric_value']):.6f}",
                "score_source": row["score_source"],
                "note": row["note"],
            }
            per_benchmark = row.get("per_benchmark") if isinstance(row.get("per_benchmark"), dict) else {}
            for name in BENCHMARK_ORDER:
                value = _safe_float(per_benchmark.get(name) if isinstance(per_benchmark, dict) else None)
                csv_row[f"benchmark_{name}"] = "" if value is None else f"{value:.6f}"
            writer.writerow(csv_row)


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(path: Path, rows: Sequence[Dict[str, object]], models_dir: Path, common_summary_label: str) -> None:
    row_height = 34
    bottom_pad = 110
    left_pad = 320
    right_pad = 120
    plot_width = 760
    present_families: List[str] = []
    for row in rows:
        family = str(row["family"])
        if family not in present_families:
            present_families.append(family)
    legend_items = [
        (family, FAMILY_COLORS[family], FAMILY_DESCRIPTIONS.get(family, family))
        for family in present_families
        if family in FAMILY_COLORS
    ]
    legend_x = left_pad
    legend_y = 96
    legend_line_height = 18
    marker_y = legend_y + max(len(legend_items), 1) * legend_line_height + 8
    top_pad = marker_y + 28
    width = left_pad + plot_width + right_pad
    height = top_pad + bottom_pad + row_height * len(rows)

    numeric_scores = [
        value
        for row in rows
        for value in (
            _safe_float(row.get("common_overall_exact")),
            _safe_float(row.get("recipe_eval_accuracy")),
        )
        if value is not None
    ]
    max_score = max(numeric_scores) if numeric_scores else 0.2
    max_score = max(0.2, math.ceil(max_score / 0.05) * 0.05)
    ticks = [round(step * 0.05, 2) for step in range(int(max_score / 0.05) + 1)]

    def x_for(score: float) -> float:
        return left_pad + (score / max_score) * plot_width

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: "Segoe UI", Arial, sans-serif; fill: #111827; }',
        '.small { font-size: 12px; fill: #4b5563; }',
        '.label { font-size: 14px; }',
        '.title { font-size: 24px; font-weight: 700; }',
        '.subtitle { font-size: 14px; fill: #374151; }',
        '.tick { font-size: 12px; fill: #6b7280; }',
        '.grid { stroke: #e5e7eb; stroke-width: 1; }',
        '.axis { stroke: #9ca3af; stroke-width: 1.5; }',
        '</style>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<text class="title" x="{left_pad}" y="36">Local Model Benchmark Graph</text>',
        f'<text class="subtitle" x="{left_pad}" y="60">Built from local zips in {_svg_escape(str(models_dir))} plus any saved virtual runtime rows. Common-benchmark scores come from the saved expanded sweep. v39 is recipe-eval only.</text>',
        f'<text class="small" x="{left_pad}" y="82">Duplicate downloads and alternate packaging were collapsed so each distinct local model family appears once, and runtime-only rows are shown without zip files.</text>',
    ]

    for tick in ticks:
        x = x_for(tick)
        lines.append(f'<line class="grid" x1="{x:.2f}" y1="{top_pad - 10}" x2="{x:.2f}" y2="{height - bottom_pad + 8}" />')
        lines.append(f'<text class="tick" x="{x:.2f}" y="{height - bottom_pad + 30}" text-anchor="middle">{tick:.2f}</text>')

    lines.append(f'<line class="axis" x1="{left_pad}" y1="{height - bottom_pad + 6}" x2="{left_pad + plot_width}" y2="{height - bottom_pad + 6}" />')

    for index, row in enumerate(rows):
        y = top_pad + index * row_height
        bar_y = y - 11
        label = str(row["label"])
        family = str(row["family"])
        common_score = _safe_float(row.get("common_overall_exact"))
        recipe_score = _safe_float(row.get("recipe_eval_accuracy"))
        score_source = str(row["score_source"])
        color = FAMILY_COLORS.get(family, "#2563eb")

        lines.append(f'<text class="label" x="{left_pad - 12}" y="{y + 5}" text-anchor="end">{_svg_escape(label)}</text>')

        if common_score is not None:
            bar_w = max(1.0, (common_score / max_score) * plot_width)
            lines.append(
                f'<rect x="{left_pad}" y="{bar_y}" width="{bar_w:.2f}" height="18" rx="4" fill="{color}" opacity="0.92" />'
            )
            score_text = f"{common_score:.3f}"
            lines.append(
                f'<text class="small" x="{left_pad + bar_w + 8:.2f}" y="{y + 4}">{score_text}</text>'
            )
        else:
            lines.append(
                f'<rect x="{left_pad}" y="{bar_y}" width="{plot_width}" height="18" rx="4" fill="#f9fafb" stroke="#d1d5db" stroke-dasharray="4 4" />'
            )
            lines.append(
                f'<text class="small" x="{left_pad + 8}" y="{y + 4}">no common-benchmark score</text>'
            )

        if recipe_score is not None:
            cx = x_for(recipe_score)
            lines.append(f'<line x1="{cx:.2f}" y1="{bar_y - 4}" x2="{cx:.2f}" y2="{bar_y + 22}" stroke="#b91c1c" stroke-width="2" />')
            lines.append(f'<circle cx="{cx:.2f}" cy="{y - 2}" r="5" fill="#b91c1c" />')
            lines.append(
                f'<text class="small" x="{min(cx + 10, left_pad + plot_width - 120):.2f}" y="{y - 10}">recipe {recipe_score:.3f}</text>'
            )

        source_note = {
            "common": "common",
            "common_alias": "common alias",
            "recipe_eval_only": "recipe only",
            "runtime": "runtime",
        }.get(score_source, score_source)
        lines.append(
            f'<text class="small" x="{left_pad + plot_width + 12}" y="{y + 4}">{_svg_escape(source_note)}</text>'
        )

    for idx, (_, color, text) in enumerate(legend_items):
        yy = legend_y + idx * legend_line_height
        lines.append(f'<rect x="{legend_x}" y="{yy - 10}" width="12" height="12" fill="{color}" />')
        lines.append(f'<text class="small" x="{legend_x + 18}" y="{yy}">{_svg_escape(text)}</text>')

    lines.append(f'<circle cx="{legend_x + 5}" cy="{marker_y - 4}" r="5" fill="#b91c1c" />')
    lines.append(f'<text class="small" x="{legend_x + 18}" y="{marker_y}">Recipe holdout marker when no common-benchmark run exists</text>')
    lines.append(
        f'<text class="small" x="{left_pad}" y="{height - 26}">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} from {_svg_escape(common_summary_label)} and the local models directory.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_pdf(path: Path, rows: Sequence[Dict[str, object]], models_dir: Path, generated_at: datetime, common_summary_label: str) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.utils import simpleSplit
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("reportlab is required to render the PDF graph") from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    page_width, base_page_height = landscape(letter)
    row_height = 24
    left_pad = 215
    right_pad = 100
    bottom_pad = 58
    present_families: List[str] = []
    for row in rows:
        family = str(row["family"])
        if family not in present_families:
            present_families.append(family)
    legend_items = [
        (family, FAMILY_COLORS[family], FAMILY_DESCRIPTIONS.get(family, family))
        for family in present_families
        if family in FAMILY_COLORS
    ]
    legend_line_height = 14
    top_pad = 150 + max(len(legend_items), 1) * legend_line_height
    page_height = max(base_page_height, top_pad + bottom_pad + row_height * len(rows) + 48)
    c = canvas.Canvas(str(path), pagesize=(page_width, page_height))
    c.setTitle("Local Model Benchmark Graph")

    plot_width = page_width - left_pad - right_pad
    axis_y = bottom_pad

    numeric_scores = [
        value
        for row in rows
        for value in (
            _safe_float(row.get("common_overall_exact")),
            _safe_float(row.get("recipe_eval_accuracy")),
        )
        if value is not None
    ]
    max_score = max(numeric_scores) if numeric_scores else 0.2
    max_score = max(0.2, math.ceil(max_score / 0.05) * 0.05)
    ticks = [round(step * 0.05, 2) for step in range(int(max_score / 0.05) + 1)]

    def x_for(score: float) -> float:
        return left_pad + (score / max_score) * plot_width

    def draw_wrapped(text: str, x: float, y: float, width: float, font_name: str, font_size: int, fill_color) -> float:
        c.setFillColor(fill_color)
        c.setFont(font_name, font_size)
        lines = simpleSplit(text, font_name, font_size, width)
        current_y = y
        for line in lines:
            c.drawString(x, current_y, line)
            current_y -= font_size + 2
        return current_y

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(left_pad, page_height - 32, "Local Model Benchmark Graph")
    after_subtitle_y = draw_wrapped(
        f"Built from local zips in {models_dir} plus any saved virtual runtime rows. Common-benchmark scores come from the saved expanded sweep plus local add-on runs for newly benchmarked models.",
        left_pad,
        page_height - 50,
        plot_width + right_pad - 10,
        "Helvetica",
        10,
        colors.HexColor("#374151"),
    )
    draw_wrapped(
        "Duplicate downloads and alternate packaging were collapsed so each distinct local model family appears once, and runtime-only rows are shown without zip files.",
        left_pad,
        after_subtitle_y - 2,
        plot_width + right_pad - 10,
        "Helvetica",
        9,
        colors.HexColor("#4b5563"),
    )

    legend_start_y = page_height - 88
    for idx, (_, color, text) in enumerate(legend_items):
        y = legend_start_y - idx * legend_line_height
        c.setFillColor(colors.HexColor(color))
        c.rect(left_pad, y - 8, 9, 9, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#4b5563"))
        c.setFont("Helvetica", 8)
        c.drawString(left_pad + 14, y - 1, text)

    marker_y = legend_start_y - max(len(legend_items), 1) * legend_line_height - 2
    c.setFillColor(colors.HexColor("#b91c1c"))
    c.circle(left_pad + 4, marker_y - 1, 3.2, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#4b5563"))
    c.setFont("Helvetica", 8)
    c.drawString(left_pad + 14, marker_y - 3, "Recipe holdout marker when no common-benchmark run exists")

    c.setStrokeColor(colors.HexColor("#9ca3af"))
    c.line(left_pad, axis_y, left_pad + plot_width, axis_y)
    for tick in ticks:
        x = x_for(tick)
        c.setStrokeColor(colors.HexColor("#e5e7eb"))
        c.line(x, axis_y, x, page_height - top_pad + 5)
        c.setFillColor(colors.HexColor("#6b7280"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(x, axis_y - 14, f"{tick:.2f}")

    for index, row in enumerate(rows):
        y = page_height - top_pad - index * row_height
        bar_y = y - 7
        label = str(row["label"])
        family = str(row["family"])
        common_score = _safe_float(row.get("common_overall_exact"))
        recipe_score = _safe_float(row.get("recipe_eval_accuracy"))
        score_source = str(row["score_source"])
        color = colors.HexColor(FAMILY_COLORS.get(family, "#2563eb"))

        c.setFillColor(colors.HexColor("#111827"))
        c.setFont("Helvetica", 10)
        c.drawRightString(left_pad - 10, y - 1, label)

        if common_score is not None:
            bar_width = max(1.0, (common_score / max_score) * plot_width)
            c.setFillColor(color)
            c.roundRect(left_pad, bar_y, bar_width, 12, 3, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#4b5563"))
            c.setFont("Helvetica", 8)
            c.drawString(min(left_pad + bar_width + 6, left_pad + plot_width - 34), y - 1, f"{common_score:.3f}")
        else:
            c.setStrokeColor(colors.HexColor("#d1d5db"))
            c.setFillColor(colors.HexColor("#f9fafb"))
            c.setDash(3, 3)
            c.roundRect(left_pad, bar_y, plot_width, 12, 3, stroke=1, fill=1)
            c.setDash()
            c.setFillColor(colors.HexColor("#6b7280"))
            c.setFont("Helvetica", 8)
            c.drawString(left_pad + 6, y - 1, "no common-benchmark score")

        if recipe_score is not None:
            cx = x_for(recipe_score)
            c.setStrokeColor(colors.HexColor("#b91c1c"))
            c.setLineWidth(1.2)
            c.line(cx, bar_y - 4, cx, bar_y + 16)
            c.setFillColor(colors.HexColor("#b91c1c"))
            c.circle(cx, y - 1, 3.2, stroke=0, fill=1)
            c.setFillColor(colors.HexColor("#7f1d1d"))
            c.setFont("Helvetica", 8)
            c.drawString(min(cx + 6, left_pad + plot_width - 56), y + 8, f"recipe {recipe_score:.3f}")
            c.setLineWidth(1)

        source_note = {
            "common": "common",
            "common_alias": "common alias",
            "recipe_eval_only": "recipe only",
            "runtime": "runtime",
        }.get(score_source, score_source)
        c.setFillColor(colors.HexColor("#4b5563"))
        c.setFont("Helvetica", 8)
        c.drawString(left_pad + plot_width + 10, y - 1, source_note)
    c.drawString(
        left_pad,
        20,
        f"Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} from {common_summary_label} and the local models directory.",
    )

    c.showPage()

    # Per-benchmark heatmap page for all local packaged models.
    page_width, page_height = landscape(letter)
    c.setPageSize((page_width, page_height))
    c.setTitle("Local Model Benchmark Matrix")
    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, page_height - 28, "Local Model Benchmark Matrix")
    draw_wrapped(
        "Rows cover every unique model represented by the local zip set. Common benchmark cells show saved exact-match scores; specialist-only models are marked N/A and carry their local specialist metric in the last column.",
        36,
        page_height - 46,
        page_width - 72,
        "Helvetica",
        9,
        colors.HexColor("#374151"),
    )

    table_left = 26
    table_top = page_height - 90
    row_h = 22
    model_col_w = 168
    family_col_w = 54
    metric_col_w = 56
    score_col_w = 50
    headers = ["Model", "Fam", "Overall"] + [BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER] + ["Spec"]
    col_widths = [model_col_w, family_col_w, score_col_w] + [score_col_w] * len(BENCHMARK_ORDER) + [metric_col_w]
    total_width = sum(col_widths)

    def draw_cell(x: float, y: float, w: float, h: float, text: str, fill_color, text_color=colors.black, align: str = "center", font_size: int = 7, font_name: str = "Helvetica") -> None:
        c.setFillColor(fill_color)
        c.rect(x, y, w, h, stroke=1, fill=1)
        c.setFillColor(text_color)
        c.setFont(font_name, font_size)
        if align == "left":
            c.drawString(x + 3, y + h / 2 - 2, text)
        elif align == "right":
            c.drawRightString(x + w - 3, y + h / 2 - 2, text)
        else:
            c.drawCentredString(x + w / 2, y + h / 2 - 2, text)

    x = table_left
    y = table_top
    for header, width in zip(headers, col_widths):
        draw_cell(x, y, width, row_h, header, colors.HexColor("#e5e7eb"), colors.HexColor("#111827"), font_size=8, font_name="Helvetica-Bold")
        x += width

    def heat_fill(value: Optional[float]):
        if value is None:
            return colors.HexColor("#f3f4f6")
        clamped = max(0.0, min(1.0, value))
        red = int(246 - (clamped * 110))
        green = int(244 - (clamped * 20))
        blue = int(250 - (clamped * 170))
        return colors.Color(red / 255.0, green / 255.0, blue / 255.0)

    for idx, row in enumerate(rows):
        y = table_top - (idx + 1) * row_h
        x = table_left
        fill = colors.HexColor("#ffffff" if idx % 2 == 0 else "#fafafa")
        draw_cell(x, y, model_col_w, row_h, str(row["label"]), fill, colors.HexColor("#111827"), align="left")
        x += model_col_w
        draw_cell(x, y, family_col_w, row_h, str(row["family"])[:8], colors.HexColor(FAMILY_COLORS.get(str(row["family"]), "#9ca3af")), colors.white, font_size=6)
        x += family_col_w

        overall = _safe_float(row.get("common_overall_exact"))
        draw_cell(x, y, score_col_w, row_h, "N/A" if overall is None else f"{overall:.2f}", heat_fill(overall), colors.HexColor("#111827"))
        x += score_col_w

        per_benchmark = row.get("per_benchmark") if isinstance(row.get("per_benchmark"), dict) else {}
        for benchmark_name in BENCHMARK_ORDER:
            value = _safe_float(per_benchmark.get(benchmark_name) if isinstance(per_benchmark, dict) else None)
            draw_cell(x, y, score_col_w, row_h, "N/A" if value is None else f"{value:.2f}", heat_fill(value), colors.HexColor("#111827"))
            x += score_col_w

        specialist_metric = _safe_float(row.get("specialist_metric_value"))
        specialist_text = "N/A"
        if specialist_metric is not None:
            specialist_text = f"{specialist_metric:.2f}"
        elif _safe_float(row.get("recipe_eval_accuracy")) is not None:
            specialist_text = f"r {float(row['recipe_eval_accuracy']):.2f}"
        draw_cell(x, y, metric_col_w, row_h, specialist_text, heat_fill(specialist_metric), colors.HexColor("#111827"))

    c.setFillColor(colors.HexColor("#4b5563"))
    c.setFont("Helvetica", 8)
    c.drawString(36, 34, f"Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} from saved benchmark outputs plus specialist summaries where common scores do not exist.")
    c.drawString(36, 20, "Spec column: local specialist metric when available, otherwise recipe holdout marker prefixed with 'r'.")
    c.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a local final-model benchmark graph from saved benchmark outputs and local zips.")
    parser.add_argument("--models_dir", default=r"C:\Users\kai99\Desktop\models")
    parser.add_argument("--common_summary", default=str(_resolve_default_common_summary()))
    parser.add_argument("--output_prefix", default="output/benchmark_local_all_models_multibench_20260330")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    common_summary = Path(args.common_summary).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    rows = build_rows(models_dir=models_dir, common_summary_path=common_summary, repo_root=repo_root)
    if not rows:
        raise RuntimeError(f"No matching model zips found in {models_dir}")

    artifacts = discover_artifacts(models_dir)
    inventory = build_zip_inventory(models_dir, artifacts)
    generated_at = datetime.now(timezone.utc)
    try:
        common_summary_label = common_summary.relative_to(repo_root).as_posix()
    except ValueError:
        common_summary_label = str(common_summary)

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    svg_path = output_prefix.with_suffix(".svg")
    pdf_path = (output_prefix.parent / "pdf" / output_prefix.name).with_suffix(".pdf")

    payload = {
        "created_at": generated_at.isoformat(),
        "models_dir": str(models_dir),
        "common_summary": str(common_summary),
        "row_count": len(rows),
        "rows": rows,
        "zip_inventory": inventory,
        "notes": [
            "Scores in common_overall_exact come from the existing expanded common-benchmark sweep plus local add-on benchmark runs for newly scored models.",
            "Rows marked common_alias map a final artifact to the chosen or equivalent scored checkpoint.",
            "recipe_eval_accuracy is retained when available so the graph can still show the local recipe holdout marker alongside a later common score.",
            "Specialist-only models that still lack a common score expose their local validation metric in specialist_metric_value and render as N/A in the common benchmark matrix.",
            "Runtime-only rows can also appear without a backing zip file when the saved common-benchmark summary includes benchmarked router or agent configurations.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    render_svg(svg_path, rows, models_dir, common_summary_label)
    render_pdf(pdf_path, rows, models_dir, generated_at, common_summary_label)

    print(json_path)
    print(csv_path)
    print(svg_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
