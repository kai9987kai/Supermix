#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RowSpec:
    key: str
    label: str
    family: str
    common_row_key: Optional[str]
    zip_patterns: Sequence[str] = ()
    runtime_only: bool = False
    note: str = ""
    specialist_summary_glob: Optional[str] = None
    specialist_metric_key: Optional[str] = None
    specialist_metric_label: str = ""
    dynamic_label: bool = False


BENCHMARK_ORDER: Sequence[str] = (
    "arc_challenge",
    "anli_r1",
    "bbh",
    "boolq",
    "commonsenseqa",
    "copa",
    "drop",
    "gsm8k",
    "hellaswag",
    "mmlu",
    "multirc",
    "openbookqa",
    "piqa",
    "qasc",
    "race_high",
    "sciq",
    "social_iqa",
    "strategyqa",
    "truthfulqa_mc1",
    "winogrande",
    "user_intent",
    "instruction_following",
    "context_tracking",
    "ambiguity_resolution",
    "chat_relevance",
)

BENCHMARK_LABELS: Dict[str, str] = {
    "arc_challenge": "ARC",
    "anli_r1": "ANLI",
    "bbh": "BBH",
    "boolq": "BoolQ",
    "commonsenseqa": "CSQA",
    "copa": "COPA",
    "drop": "DROP",
    "gsm8k": "GSM8K",
    "hellaswag": "Hella",
    "mmlu": "MMLU",
    "multirc": "MultiRC",
    "openbookqa": "OBQA",
    "piqa": "PIQA",
    "qasc": "QASC",
    "race_high": "RACE",
    "sciq": "SciQ",
    "social_iqa": "Social",
    "strategyqa": "Strategy",
    "truthfulqa_mc1": "TruthQA",
    "winogrande": "Wino",
    "user_intent": "Intent",
    "instruction_following": "Instr",
    "context_tracking": "Context",
    "ambiguity_resolution": "Ambig",
    "chat_relevance": "ChatRel",
}

FAMILY_COLORS: Dict[str, str] = {
    "fusion": "#db2777",
    "router": "#475569",
    "vision": "#7c3aed",
    "gan": "#b91c1c",
    "math": "#0f766e",
    "protein": "#6d28d9",
    "3d": "#0891b2",
}

DISPLAY_SPECS: Sequence[RowSpec] = (
    RowSpec(
        key="omni_collective_v8",
        label="omni_collective_v8",
        family="fusion",
        common_row_key="omni_collective_v8",
        zip_patterns=("supermix_omni_collective_v8_frontier_*.zip",),
        note="Final omni v8 frontier with all-model distillation, broader multimodal grounding, denser conversation data, and longer deliberation.",
        specialist_summary_glob="output/**/omni_collective_v8_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    RowSpec(
        key="omni_collective_v41",
        label="omni_collective_v41",
        family="fusion",
        common_row_key="omni_collective_v41",
        zip_patterns=("supermix_omni_collective_v41_frontier_*.zip",),
        note="Frontier omni v41 continuation with hidden planning, communication-polish, uncertainty, and code-repair upgrades.",
        specialist_summary_glob="output/**/omni_collective_v41_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
    ),
    RowSpec(
        key="three_d_generation_micro_v1",
        label="three_d_generation_micro_v1",
        family="3d",
        common_row_key="three_d_generation_micro_v1",
        zip_patterns=("supermix_3d_generation_micro_v1_*.zip",),
        note="Small OpenSCAD / CAD generation specialist with a local add-on common-benchmark run.",
        specialist_summary_glob="output/**/three_d_generation_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="3d val",
    ),
    RowSpec(
        key="protein_folding_micro_v1",
        label="protein_folding_micro_v1",
        family="protein",
        common_row_key="protein_folding_micro_v1",
        zip_patterns=("supermix_protein_folding_micro_v1_*.zip",),
        note="Protein-folding specialist with structure-prediction concept routing plus a local add-on common-benchmark run.",
        specialist_summary_glob="output/**/protein_folding_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="protein val",
    ),
    RowSpec(
        key="omni_collective_v46",
        label="omni_collective_v46",
        family="fusion",
        common_row_key="omni_collective_v46",
        zip_patterns=("supermix_omni_collective_v46*_frontier_*.zip",),
        note="Promoted v46-family branch selected from the champion manifest after benchmark gating.",
        specialist_summary_glob="output/**/omni_collective_v46*_frontier_summary.json",
        specialist_metric_key="stage2.best_score",
        specialist_metric_label="omni val",
        dynamic_label=True,
    ),
    RowSpec(
        key="math_equation_micro_v1",
        label="math_equation_micro_v1",
        family="math",
        common_row_key="math_equation_micro_v1",
        zip_patterns=("supermix_math_equation_micro_v1_*.zip",),
        note="Math specialist with exact symbolic routing plus a local add-on common-benchmark run.",
        specialist_summary_glob="output/**/math_equation_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="math val",
    ),
    RowSpec(
        key="auto_collective_loop",
        label="auto_collective_loop_s5",
        family="router",
        common_row_key="auto_collective_loop",
        runtime_only=True,
        note="Prompt-aware auto router benchmarked on a reduced 5-per-benchmark sampled suite with collective loop mode enabled.",
    ),
    RowSpec(
        key="science_vision_micro_v1",
        label="science_vision_micro_v1",
        family="vision",
        common_row_key=None,
        zip_patterns=("supermix_science_image_recognition_micro_v1_*.zip",),
        note="Specialist upload-image recognition model. Common text benchmarks are not applicable.",
        specialist_summary_glob="output/**/science_image_recognition_micro_v1_summary.json",
        specialist_metric_key="val_accuracy",
        specialist_metric_label="vision val",
    ),
    RowSpec(
        key="dcgan_v2_in_progress",
        label="dcgan_v2_in_progress",
        family="gan",
        common_row_key=None,
        zip_patterns=("dcgan_v2_in_progress.zip",),
        note="Unconditional RGB DCGAN v2 trained on CIFAR-style images. Specialist score comes from the local GAN generation benchmark.",
        specialist_summary_glob="output/**/dcgan_v2_in_progress_benchmark_summary.json",
        specialist_metric_key="specialist_score",
        specialist_metric_label="gan score",
    ),
    RowSpec(
        key="dcgan_mnist_model",
        label="dcgan_mnist_model",
        family="gan",
        common_row_key=None,
        zip_patterns=("dcgan_mnist_model.zip",),
        note="Unconditional grayscale DCGAN trained on MNIST digits. Specialist score comes from the local GAN generation benchmark.",
        specialist_summary_glob="output/**/dcgan_mnist_model_benchmark_summary.json",
        specialist_metric_key="specialist_score",
        specialist_metric_label="gan score",
    ),
)


def _resolve_default_common_summary() -> Path:
    output_dir = Path(__file__).resolve().parents[1] / "output"
    candidates = sorted(output_dir.glob("benchmark_all_models_common_plus_summary_*.json"))
    if candidates:
        return candidates[-1]
    return output_dir / "benchmark_all_models_common_plus_summary_20260429_v46_hybrid_evo120.json"


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


def _load_common_rows(summary_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    rows = payload.get("summary_rows")
    if not isinstance(rows, list):
        raise RuntimeError(f"summary_rows missing in {summary_path}")
    out: Dict[str, Dict[str, object]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("model"):
            out[str(row["model"])] = row
    return out


def _latest_matches(root: Path, patterns: Sequence[str]) -> List[Path]:
    matches: List[Path] = []
    seen: set[str] = set()
    for pattern in patterns:
        for candidate in root.glob(pattern):
            resolved = str(candidate.resolve())
            if resolved in seen or not candidate.is_file():
                continue
            seen.add(resolved)
            matches.append(candidate)
    matches.sort(key=lambda item: (item.stat().st_mtime, item.name))
    return matches


def _v46_champion_zip(repo_root: Path) -> Optional[Path]:
    manifest_path = repo_root / "output" / "omni_collective_v46_champion.json"
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    for key in ("desktop_zip_path", "zip_path"):
        raw_path = str(payload.get(key) or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.exists():
            return path.resolve()
    return None


def _load_specialist_metric(repo_root: Path, spec: RowSpec) -> Tuple[Optional[float], str]:
    if not spec.specialist_summary_glob or not spec.specialist_metric_key:
        return None, spec.specialist_metric_label
    matches = _latest_matches(repo_root, (spec.specialist_summary_glob,))
    if not matches:
        return None, spec.specialist_metric_label
    payload = json.loads(matches[-1].read_text(encoding="utf-8-sig"))
    metric = _extract_nested(payload, spec.specialist_metric_key)
    if metric is None and isinstance(payload.get("meta"), dict):
        metric = _extract_nested(payload["meta"], spec.specialist_metric_key)
    if metric is None and isinstance(payload.get("history"), list) and payload["history"]:
        last_history = payload["history"][-1]
        if isinstance(last_history, dict):
            metric = _extract_nested(last_history, spec.specialist_metric_key)
    return metric, spec.specialist_metric_label


def _derive_dynamic_label(zip_name: str) -> str:
    label = Path(zip_name).stem
    if label.startswith("supermix_"):
        label = label[len("supermix_") :]
    label = re.sub(r"_\d{8}_\d{6}$", "", label)
    label = re.sub(r"_\d{8}$", "", label)
    for suffix in ("_frontier", "_bundle", "_model"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
    return label


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _build_row(
    spec: RowSpec,
    *,
    repo_root: Path,
    models_dir: Path,
    common_rows: Dict[str, Dict[str, object]],
) -> Tuple[Optional[Dict[str, object]], List[Path]]:
    matched_zips = _latest_matches(models_dir, spec.zip_patterns)
    selected_zip = matched_zips[-1] if matched_zips else None
    champion_zip = _v46_champion_zip(repo_root) if spec.key == "omni_collective_v46" else None
    if champion_zip is not None:
        if all(path.resolve() != champion_zip for path in matched_zips):
            matched_zips.append(champion_zip)
        selected_zip = champion_zip

    common_row = common_rows.get(spec.common_row_key or "")
    if not selected_zip and not spec.runtime_only and common_row is None:
        return None, matched_zips

    label = spec.label
    if spec.dynamic_label and selected_zip is not None:
        label = _derive_dynamic_label(selected_zip.name)

    specialist_metric_value, specialist_metric_label = _load_specialist_metric(repo_root, spec)
    zip_name = selected_zip.name if selected_zip is not None else ""
    zip_path = str(selected_zip.resolve()) if selected_zip is not None else ""
    zip_size = int(selected_zip.stat().st_size) if selected_zip is not None else 0
    zip_mtime = _mtime_iso(selected_zip) if selected_zip is not None else ""

    common_overall = None
    per_benchmark: Optional[Dict[str, float]] = None
    if isinstance(common_row, dict):
        common_overall = _safe_float(common_row.get("overall_exact"))
        raw_benchmarks = common_row.get("benchmarks")
        if isinstance(raw_benchmarks, dict):
            per_benchmark = {
                name: float(raw_benchmarks.get(name, 0.0))
                for name in BENCHMARK_ORDER
                if raw_benchmarks.get(name) is not None
            }

    if spec.runtime_only:
        score_source = "runtime"
    elif common_overall is not None and spec.common_row_key and label != spec.common_row_key:
        score_source = "common_alias"
    elif common_overall is not None:
        score_source = "common"
    else:
        score_source = "specialist_only"

    row: Dict[str, object] = {
        "model_key": spec.key,
        "label": label,
        "family": spec.family,
        "zip_path": zip_path,
        "zip_name": zip_name,
        "zip_size_bytes": zip_size,
        "zip_mtime": zip_mtime,
        "common_benchmark_model": spec.common_row_key,
        "common_overall_exact": common_overall,
        "recipe_eval_accuracy": None,
        "specialist_metric_value": specialist_metric_value,
        "specialist_metric_label": specialist_metric_label,
        "score_source": score_source,
        "note": spec.note,
    }
    if champion_zip is not None and spec.key == "omni_collective_v46":
        row["selection_policy"] = "champion_manifest"
    if per_benchmark:
        row["per_benchmark"] = per_benchmark
    return row, matched_zips


def build_rows(
    *,
    models_dir: Path,
    common_summary_path: Path,
    repo_root: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, List[str]]]:
    common_rows = _load_common_rows(common_summary_path)
    common_summary_mtime = common_summary_path.stat().st_mtime if common_summary_path.exists() else 0.0
    rows: List[Dict[str, object]] = []
    selected_zip_names: List[str] = []
    matched_zip_names: set[str] = set()

    for spec in DISPLAY_SPECS:
        row, matched_zips = _build_row(
            spec,
            repo_root=repo_root,
            models_dir=models_dir,
            common_rows=common_rows,
        )
        matched_zip_names.update(path.name for path in matched_zips)
        if row is None:
            continue
        zip_path_value = str(row.get("zip_path") or "")
        benchmark_freshness = "no_common_score" if row.get("common_overall_exact") is None else "current"
        if zip_path_value and row.get("common_overall_exact") is not None:
            zip_mtime = Path(zip_path_value).stat().st_mtime
            if zip_mtime > common_summary_mtime + 60:
                benchmark_freshness = "package_newer_than_common_summary"
        row["benchmark_freshness"] = benchmark_freshness
        if row.get("zip_name"):
            selected_zip_names.append(str(row["zip_name"]))
        rows.append(row)

    rows.sort(
        key=lambda row: (
            _safe_float(row.get("common_overall_exact")) or -1.0,
            _safe_float(row.get("specialist_metric_value")) or -1.0,
            str(row.get("label", "")),
        ),
        reverse=True,
    )

    all_zips = sorted(path.name for path in models_dir.glob("*.zip"))
    selected_set = set(selected_zip_names)
    inventory = {
        "selected_zip_files": selected_zip_names,
        "alternate_package_zip_files": sorted(matched_zip_names - selected_set),
        "unmatched_zip_files": sorted(set(all_zips) - matched_zip_names),
    }
    return rows, inventory


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "label",
        "family",
        "common_benchmark_model",
        "common_overall_exact",
        "specialist_metric_value",
        "specialist_metric_label",
        "score_source",
        "benchmark_freshness",
        "zip_name",
        "zip_mtime",
    ] + list(BENCHMARK_ORDER)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            per_benchmark = row.get("per_benchmark") if isinstance(row.get("per_benchmark"), dict) else {}
            payload = {
                "label": row.get("label", ""),
                "family": row.get("family", ""),
                "common_benchmark_model": row.get("common_benchmark_model", ""),
                "common_overall_exact": row.get("common_overall_exact", ""),
                "specialist_metric_value": row.get("specialist_metric_value", ""),
                "specialist_metric_label": row.get("specialist_metric_label", ""),
                "score_source": row.get("score_source", ""),
                "benchmark_freshness": row.get("benchmark_freshness", ""),
                "zip_name": row.get("zip_name", ""),
                "zip_mtime": row.get("zip_mtime", ""),
            }
            for name in BENCHMARK_ORDER:
                payload[name] = per_benchmark.get(name, "")
            writer.writerow(payload)


def _heatmap_matrix(rows: Sequence[Dict[str, object]]) -> np.ndarray:
    matrix: List[List[float]] = []
    for row in rows:
        per_benchmark = row.get("per_benchmark") if isinstance(row.get("per_benchmark"), dict) else {}
        matrix.append([float(per_benchmark.get(name, np.nan)) for name in BENCHMARK_ORDER])
    return np.array(matrix, dtype=float)


def _bar_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#2563eb")


def _bar_annotation(row: Dict[str, object]) -> str:
    common = _safe_float(row.get("common_overall_exact"))
    specialist = _safe_float(row.get("specialist_metric_value"))
    stale_marker = "*" if row.get("benchmark_freshness") == "package_newer_than_common_summary" else ""
    if common is not None:
        return f"{common:.3f}{stale_marker}"
    if specialist is not None:
        label = str(row.get("specialist_metric_label") or "spec")
        return f"N/A | {label} {specialist:.3f}"
    return "N/A"


def render_graph(path: Path, rows: Sequence[Dict[str, object]], common_summary_label: str) -> None:
    labels = [str(row["label"]) for row in rows]
    families = [str(row["family"]) for row in rows]
    common_scores = [_safe_float(row.get("common_overall_exact")) or 0.0 for row in rows]
    annotations = [_bar_annotation(row) for row in rows]
    matrix = _heatmap_matrix(rows)
    masked = np.ma.masked_invalid(matrix)

    fig_height = max(7.5, 0.6 * len(rows) + 2.8)
    fig, (ax_heatmap, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(17, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
        constrained_layout=True,
    )

    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#f3f4f6")
    image = ax_heatmap.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax_heatmap.set_title("Per-Benchmark Exact")
    ax_heatmap.set_xticks(range(len(BENCHMARK_ORDER)))
    ax_heatmap.set_xticklabels([BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER], rotation=20, ha="right")
    ax_heatmap.set_yticks(range(len(labels)))
    ax_heatmap.set_yticklabels(labels)
    cbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")

    y_pos = np.arange(len(labels))
    colors = [_bar_color(family) for family in families]
    ax_bar.barh(y_pos, common_scores, color=colors)
    ax_bar.set_title("Overall Common-Benchmark Exact")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, max(0.25, max(common_scores) * 1.25 if common_scores else 0.25))
    ax_bar.set_xlabel("Score")
    ax_bar.grid(axis="x", color="#e5e7eb", linewidth=0.8)

    for yi, score, text in zip(y_pos, common_scores, annotations):
        x = score + 0.006 if score > 0 else 0.006
        ax_bar.text(x, yi, text, va="center", fontsize=8)

    legend_handles = []
    present_families: List[str] = []
    for family in families:
        if family not in present_families:
            present_families.append(family)
    for family in present_families:
        legend_handles.append(plt.Line2D([0], [0], color=_bar_color(family), lw=8, label=family))
    if legend_handles:
        ax_bar.legend(handles=legend_handles, loc="lower right")

    fig.suptitle("Local Model Benchmark Graph", fontsize=15)
    fig.text(
        0.02,
        0.01,
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from {common_summary_label}. "
        "A * marks a package newer than the common-benchmark summary; specialist-only rows show N/A on common text benchmarks.",
        fontsize=8,
        color="#4b5563",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a local benchmark graph from saved common-benchmark summaries and local model zips.")
    parser.add_argument("--models_dir", default=r"C:\Users\kai99\Desktop\models")
    parser.add_argument("--common_summary", default=str(_resolve_default_common_summary()))
    parser.add_argument("--output_prefix", default="output/benchmark_local_all_models_multibench_latest")
    parser.add_argument("--write_pdf", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    models_dir = Path(args.models_dir).resolve()
    common_summary = Path(args.common_summary).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows, inventory = build_rows(
        models_dir=models_dir,
        common_summary_path=common_summary,
        repo_root=repo_root,
    )
    if not rows:
        raise RuntimeError(f"No graph rows could be built from {models_dir}")

    try:
        common_summary_label = common_summary.relative_to(repo_root).as_posix()
    except ValueError:
        common_summary_label = str(common_summary)

    notes = [
        "Scores in common_overall_exact come from the saved expanded common-benchmark summary plus local add-on runs.",
        "Rows marked common_alias map a dynamically selected local package to the saved common-benchmark row for that model family.",
        "The v46 row selects the newest completed v46 zip on disk. Incomplete runs are ignored until they produce a finished artifact.",
        "benchmark_freshness marks rows whose selected package is newer than the common benchmark summary, so they should be re-benchmarked before comparing strictly.",
        "Specialist-only models expose their local specialist metric and render as N/A on the common text benchmarks.",
    ]

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models_dir": str(models_dir),
        "common_summary": str(common_summary),
        "row_count": len(rows),
        "rows": rows,
        "zip_inventory": inventory,
        "notes": notes,
    }

    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")

    write_csv(csv_path, rows)
    render_graph(png_path, rows, common_summary_label)
    render_graph(svg_path, rows, common_summary_label)

    pdf_message = "PDF skipped"
    if args.write_pdf:
        pdf_dir = output_prefix.parent / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = (pdf_dir / output_prefix.name).with_suffix(".pdf")
        try:
            render_graph(pdf_path, rows, common_summary_label)
            pdf_message = str(pdf_path)
        except Exception as exc:
            notes.append(f"PDF graph write was skipped: {exc}.")
            if pdf_path.exists():
                pdf_path.unlink(missing_ok=True)
            pdf_message = f"PDF skipped: {exc}"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json_path)
    print(csv_path)
    print(png_path)
    print(svg_path)
    print(pdf_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
