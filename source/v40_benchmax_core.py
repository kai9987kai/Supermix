from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_PATH = SOURCE_DIR / "v40_benchmax_manifest.json"

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CODE_RE = re.compile(
    r"\b(python|javascript|typescript|traceback|stack trace|exception|regex|sql|bash|powershell|git|code|debug|compile|openscad)\b|"
    r"(\bdef\s+\w+\b|\bclass\s+\w+\b|```)",
    re.IGNORECASE,
)
MATH_RE = re.compile(
    r"\b(solve|simplify|factor|expand|integrate|differentiate|derivative|equation|average|mean|algebra|arithmetic)\b|[0-9]+\s*[%+\-*/=^]",
    re.IGNORECASE,
)
FORMAT_RE = re.compile(r"\b(exactly|json|bullet|bullets|three bullets|numbered list|strictly)\b", re.IGNORECASE)
IMAGE_RE = re.compile(r"\b(image|photo|picture|diagram|visual|uploaded image|recognize|identify|caption)\b", re.IGNORECASE)
REFUSAL_RE = re.compile(r"\b(i cannot|i can't|no idea|don't know|not sure|unable)\b", re.IGNORECASE)

DEFAULT_V33_PROMPT_SOURCES = (
    "conversation_data.supermix_plus_v27_500k.jsonl",
    "conversation_data.delta_anchor_mix_2026_03_26.jsonl",
    "conversation_data.delta_official_refresh_2026_03_26.jsonl",
    "conversation_data.coding_knowledge_2026_02_19.jsonl",
    "conversation_data.hybrid_v6_live_knowledge.jsonl",
    "conversation_data.mega_reasoning_creative_v25_75582.jsonl",
    "conversation_data.quality_anchor_v2.jsonl",
)
DEFAULT_V39_COUNTS = {
    "arc_count": 1119,
    "boolq_count": 3000,
    "gsm8k_count": 2500,
    "hellaswag_count": 4000,
    "mmlu_count": 5000,
    "piqa_count": 4000,
}
DEFAULT_V33_SAMPLE_SIZE = 960
DEFAULT_V39_SAMPLE_SIZE = 960


def normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_manifest(path: Optional[Path | str] = None) -> Dict[str, Any]:
    manifest_path = Path(path or DEFAULT_MANIFEST_PATH).resolve()
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def default_output_root(manifest: Optional[Mapping[str, Any]] = None) -> Path:
    manifest = dict(manifest or load_manifest())
    return (SOURCE_DIR.parent / normalize_text(manifest.get("artifact_root") or "output/v40_benchmax")).resolve()


def ablation_rows(manifest: Optional[Mapping[str, Any]] = None) -> List[Dict[str, Any]]:
    payload = dict(manifest or load_manifest())
    rows = []
    for entry in payload.get("ablation_matrix", []) or []:
        if isinstance(entry, Mapping):
            rows.append(dict(entry))
    return rows


def _stable_fingerprint(payload: Mapping[str, Any]) -> str:
    cooked = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha1(cooked.encode("utf-8")).hexdigest()


def classify_hard_example(row: Mapping[str, Any]) -> Dict[str, Any]:
    benchmark = normalize_text(row.get("benchmark") or "unknown").lower()
    prompt = normalize_text(row.get("prompt"))
    prediction = normalize_text(row.get("prediction"))
    reference = normalize_text(row.get("reference_text"))
    extracted = normalize_text(row.get("prediction_extracted") or row.get("reference_extracted"))
    exact = float(row.get("exact") or 0.0)
    token_f1 = float(row.get("token_f1") or 0.0)
    char_similarity = float(row.get("char_similarity") or 0.0)

    labels: List[str] = []
    if exact >= 0.999:
        labels.append("correct")
    else:
        if benchmark == "gsm8k":
            labels.append("math_numeric_miss")
        elif benchmark in {"arc_challenge", "mmlu", "hellaswag", "piqa"}:
            labels.append("reasoning_or_knowledge_gap")
        elif benchmark == "boolq":
            labels.append("boolean_confusion")
        else:
            labels.append("general_miss")

        if MATH_RE.search(prompt):
            labels.append("math_or_arithmetic")
        if CODE_RE.search(prompt):
            labels.append("coding_or_scaffolding")
        if IMAGE_RE.search(prompt):
            labels.append("vision_or_description")
        if FORMAT_RE.search(prompt) or (len(prediction) < max(24, len(reference) // 3)):
            labels.append("formatting_or_truncation")
        if REFUSAL_RE.search(prediction):
            labels.append("refusal_or_uncertainty")
        if token_f1 < 0.25 and char_similarity < 0.30:
            labels.append("low_overlap")
        elif token_f1 < 0.55:
            labels.append("partial_overlap")

    if not labels:
        labels.append("general_hard_example")

    rationale = []
    if exact < 1.0:
        rationale.append(f"exact={exact:.3f}")
    rationale.append(f"token_f1={token_f1:.3f}")
    rationale.append(f"char_similarity={char_similarity:.3f}")
    if benchmark:
        rationale.append(f"benchmark={benchmark}")
    if extracted and extracted != prediction:
        rationale.append(f"prediction_extracted={extracted}")
    return {
        "failure_type": labels[0],
        "failure_tags": labels,
        "rationale": "; ".join(rationale),
    }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows = summary.get("summary_rows")
    if isinstance(rows, list):
        return [dict(row) for row in rows if isinstance(row, Mapping)]
    return []


def extract_model_row(summary: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    rows = _summary_rows(summary)
    target = normalize_text(model_name).lower()
    if rows:
        for row in rows:
            if normalize_text(row.get("model")).lower() == target:
                return dict(row)
        if len(rows) == 1 and not target:
            return dict(rows[0])
    if normalize_text(summary.get("model")).lower() == target and "overall_exact" in summary:
        return dict(summary)
    stage2 = summary.get("stage2")
    if isinstance(stage2, Mapping) and "best_score" in stage2:
        return {
            "model": model_name or normalize_text(summary.get("artifact") or summary.get("name") or "candidate"),
            "family": normalize_text(summary.get("family") or "v40"),
            "overall_exact": float(stage2.get("best_score") or 0.0),
            "avg_token_f1": float(stage2.get("val_metrics", {}).get("response_accuracy") or 0.0),
            "avg_char_similarity": float(stage2.get("val_metrics", {}).get("vision_accuracy") or 0.0),
            "avg_gen_seconds": 0.0,
            "model_seconds": 0.0,
            "benchmarks": {
                key: float(value)
                for key, value in dict(stage2.get("val_metrics") or {}).items()
                if isinstance(value, (int, float))
            },
        }
    raise KeyError(f"Could not find model row {model_name!r}")


def _benchmark_row_from_summary(summary: Mapping[str, Any], model_name: str) -> Dict[str, Any]:
    row = extract_model_row(summary, model_name)
    row.setdefault("overall_exact", float(row.get("overall_exact") or 0.0))
    row.setdefault("avg_token_f1", float(row.get("avg_token_f1") or 0.0))
    row.setdefault("avg_char_similarity", float(row.get("avg_char_similarity") or 0.0))
    row.setdefault("avg_gen_seconds", float(row.get("avg_gen_seconds") or 0.0))
    row.setdefault("model_seconds", float(row.get("model_seconds") or 0.0))
    row["benchmarks"] = dict(row.get("benchmarks") or {})
    return row


def build_ablation_table(
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    result_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    payload = dict(manifest or load_manifest())
    rows = []
    result_map = {}
    for row in result_rows or []:
        key = normalize_text(row.get("id") or row.get("name") or row.get("model")).lower()
        if key:
            result_map[key] = dict(row)
            result_map[normalize_text(row.get("model") or "").lower()] = dict(row)
    for entry in ablation_rows(payload):
        key = normalize_text(entry.get("ablation_id") or entry.get("id")).lower()
        result = result_map.get(key) or result_map.get(normalize_text(entry.get("model") or "").lower())
        cooked = dict(entry)
        cooked.setdefault("ablation_id", cooked.get("id") or cooked.get("ablation_id"))
        if result:
            cooked.update(
                {
                    "overall_exact": float(result.get("overall_exact") or 0.0),
                    "avg_token_f1": float(result.get("avg_token_f1") or 0.0),
                    "avg_char_similarity": float(result.get("avg_char_similarity") or 0.0),
                    "avg_gen_seconds": float(result.get("avg_gen_seconds") or 0.0),
                    "model_seconds": float(result.get("model_seconds") or 0.0),
                    "benchmarks": dict(result.get("benchmarks") or {}),
                    "status": "completed",
                }
            )
        rows.append(cooked)
    rows.sort(key=lambda row: row.get("id", ""))
    return rows


def inspect_checkpoint_compatibility(checkpoint_paths: Sequence[Path | str]) -> Dict[str, Any]:
    paths = [Path(path).resolve() for path in checkpoint_paths]
    if len(paths) < 2:
        raise ValueError("At least two checkpoints are required for compatibility inspection.")
    states: List[Dict[str, Any]] = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and any(torch.is_tensor(value) for value in payload.values()):
            state = payload
        elif isinstance(payload, dict):
            state = payload.get("state_dict") if isinstance(payload.get("state_dict"), dict) else payload
        else:
            raise TypeError(f"Unsupported checkpoint payload at {path}")
        states.append(dict(state))
    key_sets = [set(state.keys()) for state in states]
    first_keys = key_sets[0]
    shared_keys = set.intersection(*key_sets)
    missing = {str(path): sorted(first_keys - keys) for path, keys in zip(paths[1:], key_sets[1:]) if keys != first_keys}
    extra = {str(path): sorted(keys - first_keys) for path, keys in zip(paths[1:], key_sets[1:]) if keys != first_keys}
    shape_mismatches: Dict[str, Dict[str, List[str]]] = {}
    dtype_mismatches: Dict[str, Dict[str, List[str]]] = {}
    for key in sorted(shared_keys):
        shapes = [tuple(state[key].shape) if torch.is_tensor(state[key]) else None for state in states]
        dtypes = [str(state[key].dtype) if torch.is_tensor(state[key]) else type(state[key]).__name__ for state in states]
        if len({shape for shape in shapes}) > 1:
            shape_mismatches[key] = {str(path): [str(shape)] for path, shape in zip(paths, shapes)}
        if len({dtype for dtype in dtypes}) > 1:
            dtype_mismatches[key] = {str(path): [dtype] for path, dtype in zip(paths, dtypes)}
    compatible = not missing and not extra and not shape_mismatches and not dtype_mismatches and all(keys == first_keys for keys in key_sets)
    return {
        "compatible": compatible,
        "checkpoint_count": len(paths),
        "key_count": len(first_keys),
        "shared_key_count": len(shared_keys),
        "missing_keys": missing,
        "extra_keys": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "checkpoints": [str(path) for path in paths],
    }


def average_checkpoints(
    checkpoint_paths: Sequence[Path | str],
    output_path: Path | str,
    *,
    dry_run: bool = True,
    metadata_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    report = inspect_checkpoint_compatibility(checkpoint_paths)
    if not report["compatible"]:
        raise ValueError(
            "Incompatible checkpoints cannot be averaged: "
            + json.dumps(report, indent=2, ensure_ascii=True, default=_json_default)
        )
    paths = [Path(path).resolve() for path in checkpoint_paths]
    states = [dict(torch.load(path, map_location="cpu", weights_only=False)) for path in paths]
    merged: Dict[str, Any] = {}
    tensor_keys = 0
    preserved_keys = 0
    for key in states[0].keys():
        values = [state[key] for state in states]
        if torch.is_tensor(values[0]):
            tensor_keys += 1
            if values[0].dtype.is_floating_point or values[0].dtype.is_complex:
                stacked = torch.stack([value.float() for value in values], dim=0)
                merged[key] = stacked.mean(dim=0).to(dtype=values[0].dtype)
            else:
                if not all(torch.equal(values[0], value) for value in values[1:]):
                    raise ValueError(f"Non-floating tensor key {key!r} differs across checkpoints.")
                merged[key] = values[0].clone()
                preserved_keys += 1
        else:
            if not all(value == values[0] for value in values[1:]):
                raise ValueError(f"Non-tensor key {key!r} differs across checkpoints.")
            merged[key] = values[0]
            preserved_keys += 1
    report.update(
        {
            "tensor_key_count": tensor_keys,
            "preserved_key_count": preserved_keys,
            "output_path": str(Path(output_path).resolve()),
            "dry_run": bool(dry_run),
            "method": "compatibility_average",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    if not dry_run:
        output = Path(output_path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged, output)
        if metadata_path is not None:
            meta = dict(report)
            meta["output_path"] = str(output)
            meta_path = Path(metadata_path).resolve()
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def build_promotion_report(
    *,
    benchmark_summary: Mapping[str, Any],
    candidate_model: str,
    leader_models: Sequence[str],
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    rows = _summary_rows(benchmark_summary)
    if not rows:
        raise ValueError("benchmark_summary must contain summary_rows.")
    candidate = _benchmark_row_from_summary(benchmark_summary, candidate_model)
    leaders = {name: _benchmark_row_from_summary(benchmark_summary, name) for name in leader_models}
    best_leader_name, best_leader = max(leaders.items(), key=lambda item: float(item[1].get("overall_exact") or 0.0))
    candidate_score = float(candidate.get("overall_exact") or 0.0)
    leader_score = float(best_leader.get("overall_exact") or 0.0)
    candidate_benchmarks = dict(candidate.get("benchmarks") or {})
    delta_by_leader = {}
    for name, row in leaders.items():
        delta_by_leader[name] = {
            "overall_exact_delta": round(candidate_score - float(row.get("overall_exact") or 0.0), 6),
            "benchmark_deltas": {
                bench: round(candidate_benchmarks.get(bench, 0.0) - float(dict(row.get("benchmarks") or {}).get(bench, 0.0)), 6)
                for bench in sorted(set(candidate_benchmarks) | set(row.get("benchmarks") or {}))
            },
        }
    promotion_cfg = dict(payload.get("promotion_gate") or {})
    max_regression = float(promotion_cfg.get("max_benchmark_regression") or 0.01)
    min_gain_v33 = float(promotion_cfg.get("min_overall_gain_vs_v33_final") or 0.0)
    min_gain_v39 = float(promotion_cfg.get("min_overall_gain_vs_v39_final") or 0.0)
    regressions: Dict[str, float] = {}
    for bench in sorted(candidate_benchmarks):
        best_leader_bench = max(float(dict(row.get("benchmarks") or {}).get(bench, 0.0)) for row in leaders.values())
        regressions[bench] = round(float(candidate_benchmarks.get(bench, 0.0)) - best_leader_bench, 6)
    promote = candidate_score >= leader_score and all(delta >= -max_regression for delta in regressions.values())
    if "v33_final" in leaders:
        promote = promote and float(candidate_score - float(leaders["v33_final"].get("overall_exact") or 0.0)) >= min_gain_v33
    if "v39_final" in leaders:
        promote = promote and float(candidate_score - float(leaders["v39_final"].get("overall_exact") or 0.0)) >= min_gain_v39

    attribution: List[str] = []
    dataset_summary = candidate.get("dataset_summary") if isinstance(candidate.get("dataset_summary"), Mapping) else {}
    source_counts = dict(dataset_summary.get("source_counts") or {}) if isinstance(dataset_summary, Mapping) else {}
    if any(str(key).startswith("coding_delta") or "repair" in str(key) for key in source_counts):
        attribution.append("distillation / repair data")
    if any("hard" in str(key) or "benchmark" in str(key) for key in source_counts):
        attribution.append("hard-example mining")
    warm_start = candidate.get("warm_start") if isinstance(candidate.get("warm_start"), Mapping) else {}
    if warm_start:
        attribution.append("checkpoint soup / warm-start transfer")
    notes = candidate.get("notes")
    if isinstance(notes, list) and notes:
        attribution.append("recipe / head changes")
    if not attribution:
        attribution.append("data")

    recommended_next_run = (
        "Promote this candidate and rerun with a wider hard-example replay set."
        if promote
        else f"Keep iterating on the {best_leader_name} baseline and add more hard examples before another soup."
    )

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment_family": payload.get("experiment_family", "v40_benchmax"),
        "candidate_model": candidate_model,
        "candidate": candidate,
        "leaders": leaders,
        "best_leader": best_leader_name,
        "candidate_vs_leaders": delta_by_leader,
        "promotion_gate": {
            "promote": promote,
            "candidate_score": candidate_score,
            "best_leader_score": leader_score,
            "regressions": regressions,
            "max_benchmark_regression": max_regression,
        },
        "attribution": attribution,
        "recommended_next_run": recommended_next_run,
    }


def export_hard_example_pack(
    *,
    details_jsonl: Path | str,
    benchmark_summary: Optional[Mapping[str, Any]] = None,
    output_dir: Path | str,
    manifest: Optional[Mapping[str, Any]] = None,
    max_examples_per_bucket: Optional[int] = None,
    max_examples_total: Optional[int] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    policy = dict(payload.get("hard_example_policy") or {})
    per_bucket_limit = int(max_examples_per_bucket or policy.get("max_examples_per_bucket") or 48)
    total_limit = int(max_examples_total or policy.get("max_examples_total") or 256)
    min_exact = float(policy.get("min_exact") or 1.0)
    min_token_f1 = float(policy.get("min_token_f1") or 0.45)
    min_char_similarity = float(policy.get("min_char_similarity") or 0.55)

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    with Path(details_jsonl).resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if not cooked:
                continue
            row = json.loads(cooked)
            row = dict(row)
            exact = float(row.get("exact") or 0.0)
            token_f1 = float(row.get("token_f1") or 0.0)
            char_similarity = float(row.get("char_similarity") or 0.0)
            if exact >= min_exact and token_f1 >= min_token_f1 and char_similarity >= min_char_similarity:
                continue
            classification = classify_hard_example(row)
            row.update(classification)
            row["prompt_fingerprint"] = _stable_fingerprint(
                {
                    "prompt": normalize_text(row.get("prompt") or ""),
                    "benchmark": normalize_text(row.get("benchmark") or ""),
                    "reference": normalize_text(row.get("reference_text") or ""),
                }
            )
            rows.append(row)

    rows.sort(
        key=lambda item: (
            float(item.get("exact") or 0.0),
            float(item.get("token_f1") or 0.0),
            float(item.get("char_similarity") or 0.0),
        )
    )
    deduped: List[Dict[str, Any]] = []
    seen: set[str] = set()
    bucket_counts: Dict[str, int] = {}
    for row in rows:
        bucket = str(row.get("failure_type") or "general_hard_example")
        if bucket_counts.get(bucket, 0) >= per_bucket_limit:
            continue
        fingerprint = str(row.get("prompt_fingerprint"))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        deduped.append(row)
        if len(deduped) >= total_limit:
            break

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in deduped:
        groups.setdefault(str(row.get("failure_type") or "general_hard_example"), []).append(row)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "details_jsonl": str(Path(details_jsonl).resolve()),
        "output_dir": str(output_root),
        "total_examples": len(deduped),
        "group_counts": {key: len(value) for key, value in sorted(groups.items())},
        "policy": {
            "max_examples_per_bucket": per_bucket_limit,
            "max_examples_total": total_limit,
            "min_exact": min_exact,
            "min_token_f1": min_token_f1,
            "min_char_similarity": min_char_similarity,
        },
        "provenance": {
            "benchmark_summary": str(Path(benchmark_summary).resolve()) if isinstance(benchmark_summary, (str, Path)) else None,
            "manifest": str(DEFAULT_MANIFEST_PATH.resolve()),
        },
    }

    jsonl_path = output_root / "v40_benchmax_hard_examples.jsonl"
    json_path = output_root / "v40_benchmax_hard_examples.json"
    md_path = output_root / "v40_benchmax_hard_examples_summary.md"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in deduped:
            handle.write(json.dumps(row, ensure_ascii=True, default=_json_default) + "\n")
    json_path.write_text(json.dumps(summary | {"examples": deduped}, indent=2, ensure_ascii=True, default=_json_default), encoding="utf-8")

    lines = [
        "# v40_benchmax Hard Examples",
        "",
        f"- Total examples: {len(deduped)}",
        f"- Buckets: {len(groups)}",
        "",
        "## Buckets",
    ]
    for key, value in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(f"- {key}: {len(value)}")
    lines.extend(["", "## Sample Records"])
    for row in deduped[: min(12, len(deduped))]:
        lines.append(f"- [{row.get('benchmark')}] {row.get('failure_type')}: {row.get('prompt')}")
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return summary | {"jsonl_path": str(jsonl_path), "json_path": str(json_path), "md_path": str(md_path)}


def write_csv(path: Path | str, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            payload = {}
            for name in fieldnames:
                value = row.get(name)
                if isinstance(value, float):
                    payload[name] = f"{value:.6f}"
                else:
                    payload[name] = "" if value is None else value
            writer.writerow(payload)


def summary_rows_from_summary(summary_path: Path | str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    summary = _load_json(Path(summary_path).resolve())
    rows = _summary_rows(summary)
    return summary, rows


def load_json(path: Path | str) -> Dict[str, Any]:
    return _load_json(Path(path).resolve())


__all__ = [
    "ablation_rows",
    "average_checkpoints",
    "build_ablation_table",
    "build_promotion_report",
    "classify_hard_example",
    "default_output_root",
    "export_hard_example_pack",
    "inspect_checkpoint_compatibility",
    "load_json",
    "load_manifest",
    "normalize_text",
    "summary_rows_from_summary",
    "write_csv",
]
