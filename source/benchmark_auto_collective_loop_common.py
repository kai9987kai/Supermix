#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Sequence

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from benchmark_all_models_common import (
    _extract_answer,
    _overall_exact_score,
    _stable_hash,
    _write_csv,
    _write_jsonl,
    build_benchmark_items,
    draw_graph,
)
from multimodel_catalog import DEFAULT_COMMON_SUMMARY, DEFAULT_MODELS_DIR, discover_model_records
from multimodel_runtime import UnifiedModelManager
from qwen_supermix_pipeline import token_f1


AUTO_COLLECTIVE_LOOP_KEY = "auto_collective_loop"


def _normalize_response_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _sorted_counts(counter: Counter[str]) -> Dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0].lower())))


def benchmark_auto_collective_loop(
    *,
    manager: UnifiedModelManager,
    items,
    log_every: int,
    loop_max_steps: int,
    consultant_keys: Sequence[str],
    consultant_limit: int,
    backend_cache_size: int,
) -> tuple[list[dict], list[dict]]:
    benchmark_names = sorted({item.benchmark for item in items})
    details: List[dict] = []
    per_benchmark_exact: Dict[str, List[float]] = {name: [] for name in benchmark_names}
    token_scores: List[float] = []
    char_scores: List[float] = []
    gen_seconds: List[float] = []
    route_counts: Counter[str] = Counter()
    consulted_counts: Counter[str] = Counter()
    loop_steps_used: List[int] = []
    start_time = time.time()

    settings = {
        "agent_mode": "collective_loop",
        "loop_max_steps": int(loop_max_steps),
        "memory_enabled": False,
        "web_search_enabled": False,
        "cmd_open_enabled": False,
    }
    if consultant_keys:
        settings["collective_consultant_keys"] = list(consultant_keys)
    if consultant_limit > 0:
        settings["collective_consultant_limit"] = int(consultant_limit)

    print(f"[bench] loading {AUTO_COLLECTIVE_LOOP_KEY} (1/1)")
    for item_index, item in enumerate(items, start=1):
        item_id = f"{item.benchmark}:{item_index:04d}:{_stable_hash(item.prompt)[:12]}"
        t0 = time.time()
        payload = manager.handle_prompt(
            session_id=f"benchmark::{AUTO_COLLECTIVE_LOOP_KEY}::{item_id}",
            prompt=item.prompt,
            model_key="auto",
            action_mode="auto",
            settings=settings,
        )
        elapsed = time.time() - t0

        prediction = _normalize_response_text(payload.get("response"))
        extracted = _extract_answer(item, prediction)
        exact = 1.0 if _normalize_response_text(extracted).lower() == _normalize_response_text(item.reference_extracted).lower() else 0.0
        token = float(token_f1(item.reference_text, prediction))
        char = float(SequenceMatcher(None, item.reference_text.lower(), prediction.lower()).ratio())

        agent_trace = payload.get("agent_trace") if isinstance(payload.get("agent_trace"), dict) else {}
        consultation_rows = agent_trace.get("consultation_rows") if isinstance(agent_trace.get("consultation_rows"), list) else []
        consultation_keys: List[str] = []
        for row in consultation_rows:
            if not isinstance(row, dict):
                continue
            consultant_key = str(row.get("model_key") or row.get("model_label") or "").strip()
            if consultant_key:
                consultation_keys.append(consultant_key)
                consulted_counts[consultant_key] += 1
        loop_steps = agent_trace.get("loop_steps") if isinstance(agent_trace.get("loop_steps"), list) else []
        loop_step_count = int(payload.get("timing", {}).get("loop_steps") or len(loop_steps) or 0)
        loop_steps_used.append(loop_step_count)

        active_model_key = str(payload.get("active_model_key") or payload.get("model_key") or "").strip()
        if active_model_key:
            route_counts[active_model_key] += 1

        per_benchmark_exact[item.benchmark].append(exact)
        token_scores.append(token)
        char_scores.append(char)
        gen_seconds.append(elapsed)

        details.append(
            {
                "model": AUTO_COLLECTIVE_LOOP_KEY,
                "family": "router",
                "benchmark": item.benchmark,
                "item_id": item_id,
                "item_index": item_index,
                "prompt": item.prompt,
                "prompt_hash": _stable_hash(item.prompt),
                "reference_text": item.reference_text,
                "reference_hash": _stable_hash(item.reference_text),
                "reference_extracted": item.reference_extracted,
                "prediction": prediction,
                "prediction_hash": _stable_hash(prediction),
                "prediction_extracted": extracted,
                "exact": exact,
                "is_exact": bool(exact >= 1.0),
                "token_f1": token,
                "char_similarity": char,
                "gen_seconds": elapsed,
                "active_model_key": active_model_key,
                "active_model_label": str(payload.get("active_model_label") or ""),
                "route_reason": str(payload.get("route_reason") or ""),
                "agent_mode": str(agent_trace.get("agent_mode") or ""),
                "loop_steps": loop_step_count,
                "consultation_model_keys": consultation_keys,
                "consultation_count": len(consultation_keys),
            }
        )
        if log_every > 0 and item_index % log_every == 0:
            print(f"[bench] {AUTO_COLLECTIVE_LOOP_KEY} {item_index}/{len(items)} done")

    benchmark_scores = {
        name: float(sum(values) / max(1, len(values)))
        for name, values in per_benchmark_exact.items()
    }
    summary_rows = [
        {
            "model": AUTO_COLLECTIVE_LOOP_KEY,
            "family": "router",
            "overall_exact": _overall_exact_score(benchmark_scores),
            "avg_token_f1": float(sum(token_scores) / max(1, len(token_scores))),
            "avg_char_similarity": float(sum(char_scores) / max(1, len(char_scores))),
            "avg_gen_seconds": float(sum(gen_seconds) / max(1, len(gen_seconds))),
            "model_seconds": float(time.time() - start_time),
            "benchmarks": benchmark_scores,
            "runtime_settings": {
                "model_key": "auto",
                "action_mode": "auto",
                "agent_mode": "collective_loop",
                "loop_max_steps": int(loop_max_steps),
                "collective_consultant_keys": list(consultant_keys),
                "collective_consultant_limit": int(consultant_limit),
                "backend_cache_size": int(backend_cache_size),
                "memory_enabled": False,
                "web_search_enabled": False,
                "cmd_open_enabled": False,
            },
            "routed_model_counts": _sorted_counts(route_counts),
            "consultation_model_counts": _sorted_counts(consulted_counts),
            "avg_loop_steps": float(sum(loop_steps_used) / max(1, len(loop_steps_used))),
        }
    ]
    print(f"[bench] finished {AUTO_COLLECTIVE_LOOP_KEY} overall_exact={summary_rows[0]['overall_exact']:.4f}")
    return summary_rows, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the multimodel auto router with collective loop mode on sampled common benchmarks.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--common_summary", default=str(DEFAULT_COMMON_SUMMARY))
    parser.add_argument("--device_preference", default="cpu")
    parser.add_argument("--backend_cache_size", type=int, default=3)
    parser.add_argument("--sample_per_benchmark", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--loop_max_steps", type=int, default=3)
    parser.add_argument(
        "--collective_consultant_keys",
        default="omni_collective_v41,v40_benchmax,omni_collective_v8,v33_final,qwen_v28,omni_collective_v7,math_equation_micro_v1",
        help="Comma-separated chat model keys to prioritize in the collective worker panel.",
    )
    parser.add_argument("--collective_consultant_limit", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir).resolve()
    common_summary = Path(args.common_summary).resolve()
    consultant_keys = [part.strip() for part in str(args.collective_consultant_keys or "").split(",") if part.strip()]

    records = discover_model_records(models_dir=models_dir, common_summary_path=common_summary)
    if not records:
        raise RuntimeError(f"No local model records discovered under {models_dir}")

    manager = UnifiedModelManager(
        records=tuple(records),
        extraction_root=output_dir / "runtime_extract",
        generated_dir=output_dir / "generated",
        device_preference=str(args.device_preference),
        models_dir=models_dir,
        common_summary_path=common_summary,
        backend_cache_size=int(args.backend_cache_size),
    )

    items = build_benchmark_items(sample_per_benchmark=int(args.sample_per_benchmark), seed=int(args.seed))
    benchmark_names = sorted({item.benchmark for item in items})

    prompts_jsonl = output_dir / "benchmark_items.jsonl"
    _write_jsonl(
        prompts_jsonl,
        (
            {
                "item_id": f"{item.benchmark}:{index:04d}:{_stable_hash(item.prompt)[:12]}",
                "item_index": index,
                "benchmark": item.benchmark,
                "prompt": item.prompt,
                "prompt_hash": _stable_hash(item.prompt),
                "reference_text": item.reference_text,
                "reference_hash": _stable_hash(item.reference_text),
                "reference_extracted": item.reference_extracted,
                "max_new_tokens": item.max_new_tokens,
            }
            for index, item in enumerate(items, start=1)
        ),
    )

    summary_rows, details = benchmark_auto_collective_loop(
        manager=manager,
        items=items,
        log_every=int(args.log_every),
        loop_max_steps=int(args.loop_max_steps),
        consultant_keys=consultant_keys,
        consultant_limit=int(args.collective_consultant_limit),
        backend_cache_size=int(args.backend_cache_size),
    )

    graph_path = output_dir / "benchmark_all_models_common_graph.png"
    draw_graph(graph_path, summary_rows, benchmark_names)

    summary_path = output_dir / "benchmark_all_models_common_summary.json"
    details_path = output_dir / "benchmark_all_models_common_details.jsonl"
    csv_path = output_dir / "benchmark_all_models_common_table.csv"

    _write_jsonl(details_path, details)
    _write_csv(csv_path, summary_rows, benchmark_names)
    summary_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "models_dir": str(models_dir),
                "common_summary": str(common_summary),
                "output_dir": str(output_dir),
                "device_preference": str(args.device_preference),
                "backend_cache_size": int(args.backend_cache_size),
                "sample_per_benchmark": int(args.sample_per_benchmark),
                "benchmarks": benchmark_names,
                "models_benchmarked": [AUTO_COLLECTIVE_LOOP_KEY],
                "summary_rows": summary_rows,
                "artifacts": {
                    "prompts_jsonl": str(prompts_jsonl),
                    "details_jsonl": str(details_path),
                    "table_csv": str(csv_path),
                    "graph_png": str(graph_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[bench] wrote summary to {summary_path}")
    print(f"[bench] wrote graph to {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
