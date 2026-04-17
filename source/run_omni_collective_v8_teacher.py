from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import train_omni_collective_v8 as train_v8
from multimodel_catalog import discover_model_records
from multimodel_runtime import UnifiedModelManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one v8 teacher model in isolation and persist resumable state.")
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--teacher_key", required=True)
    parser.add_argument("--sample_jsonl", required=True)
    parser.add_argument("--state_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    models_dir = Path(args.models_dir).resolve()
    teacher_key = str(args.teacher_key)
    sample_path = Path(args.sample_jsonl).resolve()
    state_path = Path(args.state_json).resolve()

    if not str(os.environ.get("SUPERMIX_QWEN_BASE_MODEL_DIR") or "").strip():
        try:
            os.environ["SUPERMIX_QWEN_BASE_MODEL_DIR"] = str(train_v8._resolve_local_qwen_base_model())
        except Exception:
            pass

    sample = train_v8._load_teacher_sample_v8(sample_path)
    if not sample:
        raise RuntimeError(f"No teacher sample rows found at {sample_path}")

    records = {record.key: record for record in discover_model_records(models_dir=models_dir)}
    if teacher_key not in records:
        raise KeyError(f"Teacher key not found in local catalog: {teacher_key}")
    record = records[teacher_key]

    state = train_v8._load_teacher_state_v8(state_path)
    cached_rows = dict(state.get("rows") or {})
    empty_row_indices = set(state.get("empty_row_indices") or set())
    processed_rows = int(state.get("processed_rows") or 0)

    extraction_root = (repo_root / "output" / "omni_v8_teacher_cache").resolve()
    generated_dir = (repo_root / "output" / "omni_v8_teacher_generated").resolve()
    manager = UnifiedModelManager((record,), extraction_root=extraction_root, generated_dir=generated_dir, device_preference="cpu")
    _record, backend = manager.ensure_backend(record.key)

    def save_state(status: str) -> None:
        payload = {
            "teacher": teacher_key,
            "status": status,
            "sample_total": len(sample),
            "processed_rows": processed_rows,
            "empty_row_indices": sorted(empty_row_indices),
            "rows": [
                {
                    "row_index": int(row_index),
                    "score": float(score),
                    "candidate": str(candidate),
                }
                for row_index, (score, candidate) in sorted(cached_rows.items())
            ],
        }
        train_v8._write_json_atomic(state_path, payload)

    save_state("partial")
    try:
        for row_index, row in enumerate(sample, start=1):
            if row_index in cached_rows or row_index in empty_row_indices:
                continue
            session_id = f"v8distill::{teacher_key}::{row_index}"
            try:
                result = backend.chat(
                    session_id=session_id,
                    prompt=row.prompt,
                    settings={
                        "style_mode": train_v8._model_style_for_row_v8(row),
                        "response_temperature": 0.0,
                        "temperature": 0.0,
                        "max_new_tokens": 160,
                        "top_p": 0.92,
                    },
                )
                candidate = str(result.response or "").strip()
            except Exception as exc:
                candidate = ""
                print(
                    json.dumps(
                        {
                            "event": "teacher_row_error",
                            "teacher": teacher_key,
                            "row_index": row_index,
                            "reason": f"{type(exc).__name__}: {exc}",
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            finally:
                backend.clear(session_id)
            processed_rows += 1
            if not candidate:
                empty_row_indices.add(row_index)
            else:
                score = float(train_v8._score_candidate(row.response_text, candidate)["composite"])
                cached_rows[row_index] = (score, candidate)
            if row_index % 8 == 0 or row_index == len(sample):
                save_state("partial")
                print(
                    json.dumps(
                        {
                            "event": "teacher_row_progress",
                            "teacher": teacher_key,
                            "completed": row_index,
                            "sample_total": len(sample),
                            "non_empty": len(cached_rows),
                            "empty": len(empty_row_indices),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        save_state("complete")
        print(
            json.dumps(
                {
                    "event": "teacher_model_done",
                    "teacher": teacher_key,
                    "sample_rows": len(sample),
                    "non_empty": len(cached_rows),
                    "empty": len(empty_row_indices),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    finally:
        if manager._backend is not None:
            manager._backend.unload()
        gc.collect()


if __name__ == "__main__":
    main()
