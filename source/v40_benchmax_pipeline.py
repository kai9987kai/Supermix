#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from multimodel_runtime import UnifiedModelManager
from v40_benchmax_common import (
    ablation_root,
    build_ablation_pack,
    csv_write,
    json_dump,
    jsonl_write,
    load_manifest,
    manifest_ablation_map,
    output_root,
    stable_hash,
)
from v40_benchmax_core import (
    average_checkpoints,
    build_promotion_report,
    export_hard_example_pack,
    inspect_checkpoint_compatibility,
    load_json,
    normalize_text,
)
from v40_benchmax_experimental import build_research_pack


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if not cooked:
                continue
            yield json.loads(cooked)


def _load_prompt_entries(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for raw in _jsonl_rows(path):
            row = dict(raw)
            prompt = _prompt_text_from_row(row)
            if not prompt:
                continue
            row["prompt"] = prompt
            row.setdefault("source", str(path.name))
            rows.append(row)
    return rows


def _prompt_text_from_row(row: Mapping[str, Any]) -> str:
    if "prompt" in row and normalize_text(row.get("prompt")):
        return normalize_text(row.get("prompt"))
    if "user" in row and normalize_text(row.get("user")):
        return normalize_text(row.get("user"))
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for message in reversed(list(messages)):
            if not isinstance(message, Mapping):
                continue
            if str(message.get("role", "")).lower() == "user":
                text = normalize_text(message.get("content"))
                if text:
                    return text
    return ""


def write_ablation_bundle(
    *,
    repo_root: Path,
    output_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    seed: int = 20260327,
    sample_size: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)

    ablations = manifest_ablation_map(payload)
    ablation_summaries: List[Dict[str, Any]] = []
    for ablation_id in sorted(ablations):
        run_dir = ablation_root(repo_root, ablation_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        rows_jsonl = run_dir / "v40_benchmax_ablation_rows.jsonl"
        rows_csv = run_dir / "v40_benchmax_ablation_rows.csv"
        summary_json = run_dir / "v40_benchmax_ablation_summary.json"
        if dry_run:
            summary = dict(ablations[ablation_id])
            summary.update(
                {
                    "ablation_id": ablation_id,
                    "head_recipe": normalize_text(summary.get("head_recipe") or summary.get("recipe_family") or summary.get("data_family")),
                    "data_family": normalize_text(summary.get("data_family") or summary.get("data_recipe")),
                    "recipe_family": normalize_text(summary.get("recipe_family") or summary.get("head_recipe") or summary.get("data_family")),
                    "status": "planned",
                    "dry_run": True,
                }
            )
            jsonl_write(rows_jsonl, [])
            csv_write(rows_csv, [], fieldnames=["prompt", "assistant", "source", "intent", "domain"])
        else:
            rows, summary = build_ablation_pack(repo_root, ablation_id, seed=seed, sample_size=sample_size)
            jsonl_write(rows_jsonl, rows)
            if rows:
                csv_write(
                    rows_csv,
                    rows,
                    fieldnames=sorted({key for row in rows for key in row.keys()}),
                )
            else:
                csv_write(rows_csv, [], fieldnames=["prompt", "assistant", "source", "intent", "domain"])
        json_dump(summary_json, summary | {"rows_jsonl": str(rows_jsonl), "rows_csv": str(rows_csv)})
        ablation_summaries.append(summary | {"rows_jsonl": str(rows_jsonl), "rows_csv": str(rows_csv), "summary_json": str(summary_json)})

    matrix_rows = []
    for entry in payload.get("ablation_matrix", []) or []:
        cooked = dict(entry)
        cooked.setdefault("ablation_id", cooked.get("id"))
        matrix_rows.append(cooked)

    matrix_csv = root / "v40_benchmax_ablation_matrix.csv"
    matrix_json = root / "v40_benchmax_ablation_matrix.json"
    csv_write(
        matrix_csv,
        matrix_rows,
        fieldnames=[
            "ablation_id",
            "id",
            "data_family",
            "recipe_family",
            "data_recipe",
            "head_recipe",
            "data_source",
            "recipe_source",
            "status",
        ],
    )
    json_dump(matrix_json, {"ablation_matrix": matrix_rows})

    manifest_snapshot = root / "v40_benchmax_manifest_snapshot.json"
    json_dump(
        manifest_snapshot,
        {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_path": str((SOURCE_DIR / "v40_benchmax_manifest.json").resolve()),
            "artifact_root": str(output_root(repo_root)),
            "ablation_count": len(ablation_summaries),
            "ablation_summaries": ablation_summaries,
            "manifest": payload,
        },
    )
    return {
        "ablation_count": len(ablation_summaries),
        "manifest_snapshot": str(manifest_snapshot),
        "matrix_csv": str(matrix_csv),
        "matrix_json": str(matrix_json),
        "ablation_summaries": ablation_summaries,
        "dry_run": bool(dry_run),
    }


def build_hard_examples_bundle(
    *,
    details_jsonl: Path,
    benchmark_summary: Optional[Mapping[str, Any]],
    output_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    max_examples_per_bucket: Optional[int] = None,
    max_examples_total: Optional[int] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    export = export_hard_example_pack(
        details_jsonl=details_jsonl,
        benchmark_summary=benchmark_summary,
        output_dir=output_dir,
        manifest=payload,
        max_examples_per_bucket=max_examples_per_bucket,
        max_examples_total=max_examples_total,
    )
    bundle_summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str((SOURCE_DIR / "v40_benchmax_manifest.json").resolve()),
        "details_jsonl": str(Path(details_jsonl).resolve()),
        "benchmark_summary": str(Path(benchmark_summary).resolve()) if isinstance(benchmark_summary, (str, Path)) else None,
        "export": export,
    }
    return bundle_summary | export


def build_research_pack_bundle(
    *,
    hard_examples_jsonl: Path,
    output_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    distillation_jsonl: Optional[Path] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    result = build_research_pack(
        hard_examples_jsonl=hard_examples_jsonl,
        output_dir=output_dir,
        manifest=payload,
        distillation_jsonl=distillation_jsonl,
        max_rows=max_rows,
    )
    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str((SOURCE_DIR / "v40_benchmax_manifest.json").resolve()),
        "hard_examples_jsonl": str(hard_examples_jsonl.resolve()),
        "distillation_jsonl": str(distillation_jsonl.resolve()) if distillation_jsonl else None,
        "research_pack": result,
    }


def build_promotion_bundle(
    *,
    benchmark_summary_path: Path,
    candidate_model: str,
    output_dir: Path,
    leader_models: Optional[Sequence[str]] = None,
    manifest: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    summary = load_json(benchmark_summary_path)
    leaders = list(leader_models or payload.get("baseline_leaders") or [])
    if payload.get("nearest_comparator"):
        comparator = str(payload["nearest_comparator"])
        if comparator not in leaders:
            leaders.append(comparator)
    report = build_promotion_report(
        benchmark_summary=summary,
        candidate_model=candidate_model,
        leader_models=leaders,
        manifest=payload,
    )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "v40_benchmax_promotion_report.json"
    md_path = root / "v40_benchmax_promotion_report.md"
    csv_path = root / "v40_benchmax_promotion_report.csv"
    json_dump(json_path, report)
    lines = [
        "# v40_benchmax Promotion Report",
        "",
        f"- Candidate: {report['candidate_model']}",
        f"- Best leader: {report['best_leader']}",
        f"- Promote: {bool(report['promotion_gate']['promote'])}",
        f"- Candidate score: {float(report['promotion_gate']['candidate_score']):.6f}",
        f"- Best leader score: {float(report['promotion_gate']['best_leader_score']):.6f}",
        "",
        "## Candidate vs Leaders",
    ]
    for name, data in report.get("candidate_vs_leaders", {}).items():
        lines.append(
            f"- {name}: overall_exact_delta={float(data['overall_exact_delta']):.6f}"
        )
    lines.extend(["", "## Attribution"])
    for item in report.get("attribution", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Next Run", f"- {report.get('recommended_next_run', '')}"])
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    csv_rows = []
    for name, data in report.get("candidate_vs_leaders", {}).items():
        csv_rows.append(
            {
                "leader": name,
                "overall_exact_delta": float(data.get("overall_exact_delta") or 0.0),
                "benchmark_deltas": json.dumps(data.get("benchmark_deltas") or {}, ensure_ascii=True, sort_keys=True),
            }
        )
    csv_write(csv_path, csv_rows, fieldnames=["leader", "overall_exact_delta", "benchmark_deltas"])
    return report | {"json_path": str(json_path), "md_path": str(md_path), "csv_path": str(csv_path)}


def build_soup_bundle(
    *,
    checkpoint_paths: Sequence[Path],
    output_path: Path,
    metadata_path: Optional[Path] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    report = inspect_checkpoint_compatibility(checkpoint_paths)
    if not report.get("compatible"):
        raise ValueError(f"Checkpoint compatibility failed: {report}")
    if dry_run:
        return report | {"output_path": str(output_path.resolve()), "dry_run": True}
    merged_report = average_checkpoints(checkpoint_paths, output_path, dry_run=False, metadata_path=metadata_path)
    return merged_report


def _read_existing_hashes(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for row in _jsonl_rows(path):
        fingerprint = normalize_text(row.get("prompt_hash") or row.get("prompt") or "").lower()
        if fingerprint:
            seen.add(fingerprint)
    return seen


def build_collective_distillation_bundle(
    *,
    prompts_jsonl: Path,
    output_dir: Path,
    manifest: Optional[Mapping[str, Any]] = None,
    model_key: Optional[str] = None,
    teacher_model_keys: Optional[Sequence[str]] = None,
    action_mode: Optional[str] = None,
    agent_mode: Optional[str] = None,
    allow_web_search: Optional[bool] = None,
    allow_cmd_open: Optional[bool] = None,
    resume: bool = True,
    max_rows: Optional[int] = None,
    manager_factory: Callable[[], Any] = UnifiedModelManager,
) -> Dict[str, Any]:
    payload = dict(manifest or load_manifest())
    distill_cfg = dict(payload.get("distillation") or {})
    model_key = model_key or str(distill_cfg.get("default_model_key") or "auto")
    teacher_model_keys = [str(item).strip() for item in (teacher_model_keys or distill_cfg.get("teacher_model_keys") or [model_key]) if str(item).strip()]
    if not teacher_model_keys:
        teacher_model_keys = [model_key]
    action_mode = action_mode or str(distill_cfg.get("default_action_mode") or "auto")
    agent_mode = agent_mode or str(distill_cfg.get("default_agent_mode") or "collective")
    allow_web_search = bool(distill_cfg.get("allow_web_search") if allow_web_search is None else allow_web_search)
    allow_cmd_open = bool(distill_cfg.get("allow_cmd_open") if allow_cmd_open is None else allow_cmd_open)
    consensus_selection = str(distill_cfg.get("consensus_selection") or "majority_then_primary")
    min_consensus_votes = int(distill_cfg.get("min_consensus_votes") or 2)

    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    jsonl_path = root / "v40_benchmax_collective_distill.jsonl"
    json_path = root / "v40_benchmax_collective_distill_summary.json"
    md_path = root / "v40_benchmax_collective_distill_summary.md"
    if not resume and jsonl_path.exists():
        jsonl_path.unlink()

    prompt_rows = _load_prompt_entries([Path(prompts_jsonl).resolve()])
    seen = _read_existing_hashes(jsonl_path) if resume else set()
    runtime_error = ""
    manager = None
    try:
        manager = manager_factory()
    except Exception as exc:
        runtime_error = str(exc)
    appended: List[Dict[str, Any]] = []
    total_seen = len(seen)
    consensus_rows = 0
    disagreement_rows = 0
    agreement_sum = 0.0

    def _payload_value(payload: Any, key: str, default: Any = "") -> Any:
        if isinstance(payload, Mapping):
            return payload.get(key, default)
        return getattr(payload, key, default)

    def _fallback_teacher_result(prompt: str, row: Mapping[str, Any], teacher_key: str, *, reason: str) -> Dict[str, Any]:
        reference_text = normalize_text(row.get("reference_text"))
        reference_extracted = normalize_text(row.get("reference_extracted"))
        answer = reference_text or reference_extracted or prompt[:240]
        return {
            "active_model_key": f"{teacher_key}_stub",
            "active_model_label": "Stub Collective Teacher",
            "route_reason": reason,
            "response": answer,
            "agent_trace": {
                "mode": "stub_fallback",
                "reason": "runtime_unavailable_or_failed",
                "prompt_hash": stable_hash(prompt),
            },
        }

    with jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(prompt_rows):
            if max_rows is not None and len(appended) >= int(max_rows):
                break
            prompt = normalize_text(row.get("prompt"))
            if not prompt:
                continue
            prompt_hash = stable_hash(prompt)
            if prompt_hash in seen:
                continue
            settings = {
                "memory_enabled": True,
                "agent_mode": agent_mode,
                "web_search_enabled": allow_web_search,
                "cmd_open_enabled": allow_cmd_open,
                "web_search_budget": 3,
                "web_search_results": 5,
            }
            teacher_candidates: List[Dict[str, Any]] = []
            for teacher_key in teacher_model_keys:
                if manager is None:
                    route_reason = "stub_fallback_unavailable_runtime"
                    if runtime_error:
                        route_reason = f"{route_reason}: {runtime_error}"
                    result = _fallback_teacher_result(prompt, row, teacher_key, reason=route_reason)
                else:
                    try:
                        if hasattr(manager, "handle_prompt"):
                            result = manager.handle_prompt(
                                session_id=f"v40_benchmax::{prompt_hash[:12]}::{teacher_key}",
                                prompt=prompt,
                                model_key=teacher_key,
                                action_mode=action_mode,
                                settings=settings,
                            )
                        else:
                            result = manager.chat(  # type: ignore[attr-defined]
                                session_id=f"v40_benchmax::{prompt_hash[:12]}::{teacher_key}",
                                prompt=prompt,
                                settings=settings,
                            )
                    except Exception as exc:
                        result = _fallback_teacher_result(prompt, row, teacher_key, reason=f"stub_fallback_runtime_error: {exc}")

                answer = normalize_text(_payload_value(result, "response", _payload_value(result, "refined_prompt", _payload_value(result, "prompt_used", ""))))
                teacher_candidates.append(
                    {
                        "teacher_model_key": _payload_value(result, "active_model_key", _payload_value(result, "model_key", teacher_key)),
                        "teacher_model_label": _payload_value(result, "active_model_label", _payload_value(result, "model_label", teacher_key)),
                        "route_reason": _payload_value(result, "route_reason", ""),
                        "teacher_answer": answer,
                        "teacher_image_url": _payload_value(result, "image_url", ""),
                        "output_path": _payload_value(result, "output_path", ""),
                        "agent_trace": _payload_value(result, "agent_trace", {}) or {},
                    }
                )

            valid_candidates = [item for item in teacher_candidates if normalize_text(item.get("teacher_answer"))]
            if not valid_candidates:
                valid_candidates = list(teacher_candidates)
            answer_votes = Counter(normalize_text(item.get("teacher_answer")) for item in valid_candidates if normalize_text(item.get("teacher_answer")))
            selected = valid_candidates[0]
            if answer_votes:
                winner_answer, winner_votes = answer_votes.most_common(1)[0]
                matching = [item for item in valid_candidates if normalize_text(item.get("teacher_answer")) == winner_answer]
                if consensus_selection == "majority_then_primary" and winner_votes >= min_consensus_votes:
                    selected = matching[0]
                    consensus_rows += 1
                elif consensus_selection == "majority_then_primary":
                    selected = valid_candidates[0]
                    disagreement_rows += 1
                else:
                    selected = matching[0]
                agreement_ratio = float(winner_votes) / float(max(len(valid_candidates), 1))
            else:
                agreement_ratio = 0.0
                disagreement_rows += 1

            agreement_sum += agreement_ratio
            payload_row = {
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "source": row.get("source") or "",
                "metadata": dict(row.get("metadata") or {}),
                "teacher_model_key": selected.get("teacher_model_key", ""),
                "teacher_model_label": selected.get("teacher_model_label", ""),
                "route_reason": selected.get("route_reason", ""),
                "teacher_answer": selected.get("teacher_answer", ""),
                "teacher_image_url": selected.get("teacher_image_url", ""),
                "output_path": selected.get("output_path", ""),
                "agent_trace": selected.get("agent_trace", {}) or {},
                "teacher_candidates": teacher_candidates,
                "teacher_agreement_ratio": round(agreement_ratio, 6),
                "teacher_consensus_selection": consensus_selection,
                "teacher_consensus_votes": max(answer_votes.values()) if answer_votes else 0,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            handle.write(json.dumps(payload_row, ensure_ascii=True) + "\n")
            seen.add(prompt_hash)
            appended.append(payload_row)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts_jsonl": str(Path(prompts_jsonl).resolve()),
        "output_dir": str(root),
        "jsonl_path": str(jsonl_path),
        "prompt_count": len(prompt_rows),
        "appended_count": len(appended),
        "resume_enabled": bool(resume),
        "resume_seen_count": total_seen,
        "model_key": model_key,
        "teacher_model_keys": teacher_model_keys,
        "action_mode": action_mode,
        "agent_mode": agent_mode,
        "allow_web_search": allow_web_search,
        "allow_cmd_open": allow_cmd_open,
        "runtime_available": manager is not None,
        "runtime_error": runtime_error,
        "consensus_selection": consensus_selection,
        "min_consensus_votes": min_consensus_votes,
        "consensus_rows": consensus_rows,
        "disagreement_rows": disagreement_rows,
        "avg_teacher_agreement_ratio": round(agreement_sum / float(max(len(appended), 1)), 6) if appended else 0.0,
        "fallback_used": any(str(row.get("teacher_model_key") or "").endswith("_stub") for row in appended),
        "records": appended,
    }
    json_dump(json_path, summary)
    lines = [
        "# v40_benchmax Collective Distillation",
        "",
        f"- Prompts loaded: {len(prompt_rows)}",
        f"- New records: {len(appended)}",
        f"- Resume enabled: {bool(resume)}",
        f"- Model key: {model_key}",
        f"- Action mode: {action_mode}",
        f"- Agent mode: {agent_mode}",
        "",
        "## Sample Records",
    ]
    for row in appended[: min(12, len(appended))]:
        lines.append(f"- {row['prompt_hash']}: {row['teacher_model_label']} -> {normalize_text(row['teacher_answer'])[:140]}")
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    summary.update({"md_path": str(md_path)})
    return summary


def _parse_bool(value: str) -> bool:
    cooked = normalize_text(value).lower()
    return cooked in {"1", "true", "yes", "y", "on"}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo_root", default=str(SOURCE_DIR.parent))
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output_dir", default="")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="v40_benchmax benchmark and hard-example pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ablation = subparsers.add_parser("ablation", help="Build the 2x2 ablation packs and matrix outputs.")
    _add_common_args(ablation)
    ablation.add_argument("--seed", type=int, default=20260327)
    ablation.add_argument("--sample_size", type=int, default=0)
    ablation.add_argument("--dry_run", default="false")

    hard = subparsers.add_parser("hard-examples", help="Export hard examples from benchmark misses.")
    _add_common_args(hard)
    hard.add_argument("--details_jsonl", required=True)
    hard.add_argument("--benchmark_summary", default="")
    hard.add_argument("--max_examples_per_bucket", type=int, default=0)
    hard.add_argument("--max_examples_total", type=int, default=0)

    research = subparsers.add_parser("research-pack", help="Build research-inspired curriculum and verifier replay rows.")
    _add_common_args(research)
    research.add_argument("--hard_examples_jsonl", required=True)
    research.add_argument("--distillation_jsonl", default="")
    research.add_argument("--max_rows", type=int, default=0)

    distill = subparsers.add_parser("distill", help="Collect teacher answers using the collective runtime.")
    _add_common_args(distill)
    distill.add_argument("--prompts_jsonl", required=True)
    distill.add_argument("--model_key", default="")
    distill.add_argument("--teacher_model_key", action="append", default=[])
    distill.add_argument("--action_mode", default="")
    distill.add_argument("--agent_mode", default="")
    distill.add_argument("--allow_web_search", default="")
    distill.add_argument("--allow_cmd_open", default="")
    distill.add_argument("--resume", default="true")
    distill.add_argument("--max_rows", type=int, default=0)

    soup = subparsers.add_parser("soup", help="Run checkpoint compatibility checks or a dry-run average.")
    _add_common_args(soup)
    soup.add_argument("--checkpoint", action="append", default=[], required=True)
    soup.add_argument("--output_path", required=True)
    soup.add_argument("--metadata_path", default="")
    soup.add_argument("--dry_run", default="true")

    report = subparsers.add_parser("report", help="Build a benchmark comparison and promotion report.")
    _add_common_args(report)
    report.add_argument("--benchmark_summary", required=True)
    report.add_argument("--candidate_model", required=True)
    report.add_argument("--leader_models", action="append", default=[])

    args = parser.parse_args(argv)
    manifest = load_json(Path(args.manifest).resolve()) if str(args.manifest).strip() else load_manifest()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else output_root(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "ablation":
        result = write_ablation_bundle(
            repo_root=repo_root,
            output_dir=output_dir,
            manifest=manifest,
            seed=int(args.seed),
            sample_size=int(args.sample_size) or None,
            dry_run=_parse_bool(args.dry_run),
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.command == "hard-examples":
        details = Path(args.details_jsonl).resolve()
        benchmark_summary = load_json(Path(args.benchmark_summary).resolve()) if str(args.benchmark_summary).strip() else None
        hard_dir = output_dir / "hard_examples" / _now_stamp()
        result = build_hard_examples_bundle(
            details_jsonl=details,
            benchmark_summary=benchmark_summary,
            output_dir=hard_dir,
            manifest=manifest,
            max_examples_per_bucket=int(args.max_examples_per_bucket) or None,
            max_examples_total=int(args.max_examples_total) or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.command == "research-pack":
        research_dir = output_dir / "research_pack" / _now_stamp()
        result = build_research_pack_bundle(
            hard_examples_jsonl=Path(args.hard_examples_jsonl).resolve(),
            output_dir=research_dir,
            manifest=manifest,
            distillation_jsonl=Path(args.distillation_jsonl).resolve() if str(args.distillation_jsonl).strip() else None,
            max_rows=int(args.max_rows) or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.command == "distill":
        distill_dir = output_dir / "distillation" / _now_stamp()
        result = build_collective_distillation_bundle(
            prompts_jsonl=Path(args.prompts_jsonl).resolve(),
            output_dir=distill_dir,
            manifest=manifest,
            model_key=str(args.model_key).strip() or None,
            teacher_model_keys=[str(item).strip() for item in args.teacher_model_key if str(item).strip()] or None,
            action_mode=str(args.action_mode).strip() or None,
            agent_mode=str(args.agent_mode).strip() or None,
            allow_web_search=_parse_bool(args.allow_web_search) if str(args.allow_web_search).strip() else None,
            allow_cmd_open=_parse_bool(args.allow_cmd_open) if str(args.allow_cmd_open).strip() else None,
            resume=_parse_bool(args.resume),
            max_rows=int(args.max_rows) or None,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.command == "soup":
        soup_dir = output_dir / "soup" / _now_stamp()
        soup_dir.mkdir(parents=True, exist_ok=True)
        out_path = Path(args.output_path).resolve()
        meta_path = Path(args.metadata_path).resolve() if str(args.metadata_path).strip() else soup_dir / "v40_benchmax_soup_metadata.json"
        result = build_soup_bundle(
            checkpoint_paths=[Path(path).resolve() for path in args.checkpoint],
            output_path=out_path,
            metadata_path=meta_path,
            dry_run=_parse_bool(args.dry_run),
        )
        json_dump(soup_dir / "v40_benchmax_soup_report.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.command == "report":
        report_dir = output_dir / "reports" / _now_stamp()
        report_dir.mkdir(parents=True, exist_ok=True)
        result = build_promotion_bundle(
            benchmark_summary_path=Path(args.benchmark_summary).resolve(),
            candidate_model=str(args.candidate_model).strip(),
            output_dir=report_dir,
            leader_models=[str(item).strip() for item in args.leader_models if str(item).strip()],
            manifest=manifest,
        )
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
