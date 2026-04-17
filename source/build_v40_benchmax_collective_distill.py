#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from v40_benchmax_common import load_manifest
from v40_benchmax_pipeline import build_collective_distillation_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate additive collective-teacher distillation rows for v40_benchmax.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--model_key", default="")
    parser.add_argument("--teacher_model_key", action="append", default=[])
    parser.add_argument("--agent_mode", default="")
    parser.add_argument("--action_mode", default="")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    target_jsonl = Path(args.output_jsonl).resolve()
    bundle = build_collective_distillation_bundle(
        prompts_jsonl=Path(args.input_jsonl).resolve(),
        output_dir=target_jsonl.parent,
        manifest=manifest,
        model_key=str(args.model_key).strip() or None,
        teacher_model_keys=[str(item).strip() for item in args.teacher_model_key if str(item).strip()] or None,
        action_mode=str(args.action_mode).strip() or None,
        agent_mode=str(args.agent_mode).strip() or None,
        resume=bool(args.resume),
        max_rows=int(args.max_examples) or None,
    )

    generated_jsonl = Path(bundle["jsonl_path"]).resolve()
    if generated_jsonl != target_jsonl:
        target_jsonl.parent.mkdir(parents=True, exist_ok=True)
        target_jsonl.write_text(generated_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
        bundle["jsonl_path"] = str(target_jsonl)
    summary_out = Path(args.summary_out).resolve()
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[v40-benchmax] wrote distillation rows to {target_jsonl}")
    print(f"[v40-benchmax] wrote summary to {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
