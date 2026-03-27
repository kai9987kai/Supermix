#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ArtifactSpec:
    key: str
    label: str
    family: str
    filename_tokens: Sequence[str]
    common_row_key: Optional[str]
    note: str = ""
    recipe_eval_accuracy: Optional[float] = None
    score_source: str = "common"


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
        key="v39_final",
        label="v39_final",
        family="champion",
        filename_tokens=("champion_v39_frontier_reasoning_plus_full_model_20260327",),
        common_row_key=None,
        recipe_eval_accuracy=0.0549,
        score_source="recipe_eval_only",
        note="No common-benchmark run was completed after the pod lost credit. This marker is the finished v39 recipe holdout score (29/528).",
    ),
)


FAMILY_COLORS: Dict[str, str] = {
    "qwen": "#d97706",
    "champion": "#2563eb",
    "native_image": "#15803d",
    "wrapper": "#6b7280",
}


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


def build_rows(models_dir: Path, common_summary_path: Path) -> List[Dict[str, object]]:
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
            "score_source": spec.score_source if common_score is None else ("common_alias" if spec.common_row_key and spec.common_row_key != spec.key else "common"),
            "note": spec.note,
        }
        if common_row and isinstance(common_row.get("benchmarks"), dict):
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
        "score_source",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "family": row["family"],
                    "zip_name": row["zip_name"],
                    "zip_size_bytes": row["zip_size_bytes"],
                    "common_benchmark_model": row.get("common_benchmark_model") or "",
                    "common_overall_exact": "" if row.get("common_overall_exact") is None else f"{float(row['common_overall_exact']):.6f}",
                    "recipe_eval_accuracy": "" if row.get("recipe_eval_accuracy") is None else f"{float(row['recipe_eval_accuracy']):.6f}",
                    "score_source": row["score_source"],
                    "note": row["note"],
                }
            )


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(path: Path, rows: Sequence[Dict[str, object]], models_dir: Path) -> None:
    row_height = 34
    top_pad = 120
    bottom_pad = 110
    left_pad = 320
    right_pad = 120
    plot_width = 760
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
        f'<text class="subtitle" x="{left_pad}" y="60">Built from local zips in {_svg_escape(str(models_dir))}. Common-benchmark scores come from the saved expanded sweep. v39 is recipe-eval only.</text>',
        f'<text class="small" x="{left_pad}" y="82">Duplicate downloads and alternate packaging were collapsed so each distinct local model family appears once.</text>',
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
        }.get(score_source, score_source)
        lines.append(
            f'<text class="small" x="{left_pad + plot_width + 12}" y="{y + 4}">{_svg_escape(source_note)}</text>'
        )

    legend_x = left_pad
    legend_y = 96
    legend_items = [
        ("qwen", FAMILY_COLORS["qwen"], "Qwen adapter scored on saved common benchmark"),
        ("champion", FAMILY_COLORS["champion"], "Champion-family final model"),
        ("native_image", FAMILY_COLORS["native_image"], "Native-image text+image checkpoint"),
        ("wrapper", FAMILY_COLORS["wrapper"], "Wrapper or alias artifact"),
    ]
    for idx, (name, color, text) in enumerate(legend_items):
        yy = legend_y + idx * 18
        lines.append(f'<rect x="{legend_x}" y="{yy - 10}" width="12" height="12" fill="{color}" />')
        lines.append(f'<text class="small" x="{legend_x + 18}" y="{yy}">{_svg_escape(text)}</text>')

    lines.append(f'<circle cx="{legend_x + 515}" cy="{legend_y - 4}" r="5" fill="#b91c1c" />')
    lines.append(f'<text class="small" x="{legend_x + 528}" y="{legend_y}">Recipe holdout marker when no common-benchmark run exists</text>')
    lines.append(
        f'<text class="small" x="{left_pad}" y="{height - 26}">Generated {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} from output/benchmark_all_models_common_plus_summary_20260327.json and the local models directory.</text>'
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def render_pdf(path: Path, rows: Sequence[Dict[str, object]], models_dir: Path, generated_at: datetime) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.utils import simpleSplit
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise RuntimeError("reportlab is required to render the PDF graph") from exc

    path.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = landscape(letter)
    c = canvas.Canvas(str(path), pagesize=(page_width, page_height))
    c.setTitle("Local Model Benchmark Graph")

    left_pad = 215
    right_pad = 100
    top_pad = 110
    bottom_pad = 75
    row_height = 24
    plot_width = page_width - left_pad - right_pad
    axis_y = page_height - bottom_pad

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
        f"Built from local zips in {models_dir}. Common-benchmark scores come from the saved expanded sweep. v39 is recipe-eval only.",
        left_pad,
        page_height - 50,
        plot_width + right_pad - 10,
        "Helvetica",
        10,
        colors.HexColor("#374151"),
    )
    draw_wrapped(
        "Duplicate downloads and alternate packaging were collapsed so each distinct local model family appears once.",
        left_pad,
        after_subtitle_y - 2,
        plot_width + right_pad - 10,
        "Helvetica",
        9,
        colors.HexColor("#4b5563"),
    )

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
        }.get(score_source, score_source)
        c.setFillColor(colors.HexColor("#4b5563"))
        c.setFont("Helvetica", 8)
        c.drawString(left_pad + plot_width + 10, y - 1, source_note)

    legend_y = 72
    legend_items = [
        ("qwen", FAMILY_COLORS["qwen"], "Qwen adapter scored on saved common benchmark"),
        ("champion", FAMILY_COLORS["champion"], "Champion-family final model"),
        ("native_image", FAMILY_COLORS["native_image"], "Native-image text+image checkpoint"),
        ("wrapper", FAMILY_COLORS["wrapper"], "Wrapper or alias artifact"),
    ]
    legend_x = left_pad
    for idx, (_, color, text) in enumerate(legend_items):
        y = legend_y - idx * 14
        c.setFillColor(colors.HexColor(color))
        c.rect(legend_x, y - 8, 9, 9, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#4b5563"))
        c.setFont("Helvetica", 8)
        c.drawString(legend_x + 14, y - 1, text)

    c.setFillColor(colors.HexColor("#b91c1c"))
    c.circle(left_pad + 360, legend_y - 1, 3.2, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#4b5563"))
    c.setFont("Helvetica", 8)
    c.drawString(left_pad + 370, legend_y - 3, "Recipe holdout marker when no common-benchmark run exists")
    c.drawString(
        left_pad,
        20,
        f"Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')} from output/benchmark_all_models_common_plus_summary_20260327.json and the local models directory.",
    )

    c.showPage()
    c.save()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a local final-model benchmark graph from saved benchmark outputs and local zips.")
    parser.add_argument("--models_dir", default=r"C:\Users\kai99\Desktop\models")
    parser.add_argument("--common_summary", default="output/benchmark_all_models_common_plus_summary_20260327.json")
    parser.add_argument("--output_prefix", default="output/benchmark_local_final_models_20260327")
    args = parser.parse_args()

    models_dir = Path(args.models_dir).resolve()
    common_summary = Path(args.common_summary).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(models_dir=models_dir, common_summary_path=common_summary)
    if not rows:
        raise RuntimeError(f"No matching model zips found in {models_dir}")

    artifacts = discover_artifacts(models_dir)
    inventory = build_zip_inventory(models_dir, artifacts)
    generated_at = datetime.now(timezone.utc)

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
            "Scores in common_overall_exact come from the existing expanded common-benchmark sweep in the repo output directory.",
            "Rows marked common_alias map a final artifact to the chosen or equivalent scored checkpoint.",
            "v39_final only has recipe_eval_accuracy because its common-benchmark run never completed after the pod lost credit.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    render_svg(svg_path, rows, models_dir)
    render_pdf(pdf_path, rows, models_dir, generated_at)

    print(json_path)
    print(csv_path)
    print(svg_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
