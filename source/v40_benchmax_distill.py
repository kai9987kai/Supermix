from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    from v40_benchmax_core import load_manifest, normalize_text
except ImportError:  # pragma: no cover
    from .v40_benchmax_core import load_manifest, normalize_text


def _fingerprint(prompt: str, image_path: str = "") -> str:
    cooked = json.dumps({"prompt": normalize_text(prompt), "image_path": normalize_text(image_path)}, sort_keys=True)
    return hashlib.sha1(cooked.encode("utf-8")).hexdigest()


def distill_examples(
    rows: Sequence[Mapping[str, Any]],
    teacher_fn: Callable[[str, Optional[str]], Dict[str, Any]],
    *,
    resume_hashes: Optional[set[str]] = None,
    max_examples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen = set(resume_hashes or set())
    for row in rows:
        prompt = normalize_text(row.get("prompt") or row.get("user") or "")
        image_path = normalize_text(row.get("image_path") or "")
        if not prompt:
            continue
        fp = _fingerprint(prompt, image_path)
        if fp in seen:
            continue
        payload = teacher_fn(prompt, image_path or None)
        record = {
            "prompt": prompt,
            "image_path": image_path,
            "fingerprint": fp,
            "teacher_answer": normalize_text(payload.get("teacher_answer") or payload.get("response") or ""),
            "route_reason": normalize_text(payload.get("route_reason") or ""),
            "model_key": normalize_text(payload.get("model_key") or payload.get("active_model_key") or ""),
            "model_label": normalize_text(payload.get("model_label") or payload.get("active_model_label") or ""),
            "agent_trace": payload.get("agent_trace") or {},
            "metadata": dict(row.get("metadata") or {}),
        }
        results.append(record)
        seen.add(fp)
        if max_examples is not None and len(results) >= max_examples:
            break
    return results


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if cooked:
                rows.append(json.loads(cooked))
    return rows


def _load_existing_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cooked = line.strip()
            if cooked:
                payload = json.loads(cooked)
                fp = normalize_text(payload.get("fingerprint") or "")
                if fp:
                    hashes.add(fp)
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect teacher answers for v40_benchmax distillation.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "v40_benchmax_manifest.json"))
    parser.add_argument("--model_key", default="")
    parser.add_argument("--action_mode", default="auto")
    parser.add_argument("--agent_mode", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--web_search_enabled", action="store_true")
    parser.add_argument("--cmd_open_enabled", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    input_path = Path(args.input_jsonl).resolve()
    output_path = Path(args.output_jsonl).resolve()
    rows = _load_jsonl(input_path)
    if args.resume:
        existing_hashes = _load_existing_hashes(output_path)
    else:
        existing_hashes = set()

    if args.dry_run:
        payload = {
            "ok": True,
            "input_rows": len(rows),
            "would_skip": len(existing_hashes),
            "output_jsonl": str(output_path),
            "agent_mode": args.agent_mode or manifest.get("distillation", {}).get("default_agent_mode", "collective"),
            "action_mode": args.action_mode,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=True))
        return 0

    from multimodel_catalog import discover_model_records
    from multimodel_runtime import UnifiedModelManager

    records = discover_model_records()
    manager = UnifiedModelManager(
        records,
        extraction_root=Path("output") / "v40_benchmax" / "distill_extract",
        generated_dir=Path("output") / "v40_benchmax" / "distill_generated",
    )

    def teacher_fn(prompt: str, image_path: Optional[str]) -> Dict[str, Any]:
        settings = {
            "agent_mode": args.agent_mode or manifest.get("distillation", {}).get("default_agent_mode", "collective"),
            "web_search_enabled": bool(args.web_search_enabled or manifest.get("distillation", {}).get("allow_web_search", True)),
            "cmd_open_enabled": bool(args.cmd_open_enabled or manifest.get("distillation", {}).get("allow_cmd_open", True)),
            "memory_enabled": False,
            "uploaded_image_path": image_path or "",
        }
        payload = manager.handle_prompt(
            session_id="v40_benchmax_distill",
            prompt=prompt,
            model_key=args.model_key or manifest.get("distillation", {}).get("default_model_key", "auto"),
            action_mode=args.action_mode or manifest.get("distillation", {}).get("default_action_mode", "auto"),
            settings=settings,
        )
        return {
            "response": payload.get("response") or "",
            "route_reason": payload.get("route_reason") or "",
            "model_key": payload.get("active_model_key") or payload.get("model_key") or "",
            "model_label": payload.get("active_model_label") or payload.get("model_label") or "",
            "agent_trace": payload.get("agent_trace") or {},
        }

    distilled = distill_examples(
        rows,
        teacher_fn,
        resume_hashes=existing_hashes,
        max_examples=args.max_examples or None,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in distilled:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"ok": True, "written": len(distilled), "output_jsonl": str(output_path)}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
