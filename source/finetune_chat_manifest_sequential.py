import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    shards = payload.get("shards", payload if isinstance(payload, list) else [])
    out: List[Dict[str, Any]] = []
    for item in shards:
        if isinstance(item, str):
            out.append({"path": item, "rows": None})
        elif isinstance(item, dict) and item.get("path"):
            out.append(dict(item))
    if not out:
        raise ValueError("Manifest has no usable shard entries.")
    return out


def _replace_arg(args: List[str], key: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(args):
        if args[i] == key:
            i += 2
            continue
        out.append(args[i])
        i += 1
    out.extend([key, value])
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Out-of-core sequential shard trainer (manifest wrapper for finetune_chat.py).")
    ap.add_argument("--manifest", required=True, help="JSON manifest with shard paths.")
    ap.add_argument("--weights", required=True, help="Initial checkpoint.")
    ap.add_argument("--output", required=True, help="Final checkpoint output.")
    ap.add_argument("--meta", required=True, help="Final metadata JSON output.")
    ap.add_argument("--stage_dir", default="sequential_manifest_stages", help="Directory for intermediate checkpoints.")
    ap.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable.")
    ap.add_argument("--script", default="finetune_chat.py", help="Training script to call for each shard.")
    ap.add_argument("--max_shards", type=int, default=0, help="Optional limit for testing.")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume by skipping stages whose stage checkpoint+meta already exist.",
    )
    ap.add_argument(
        "--keep_stage_artifacts",
        action="store_true",
        help="Keep intermediate stage checkpoints/metadata (default deletes prior stages to save disk).",
    )
    ap.add_argument(
        "--child_unbuffered",
        action="store_true",
        help="Run child finetune processes with PYTHONUNBUFFERED=1 for live logs when wrapper stdout is redirected.",
    )
    ap.add_argument("--carry_best_only", action="store_true", help="Copy only final stage outputs to target paths (default behavior).")
    args, passthrough = ap.parse_known_args()

    if not passthrough:
        raise SystemExit("Pass finetune args after wrapper args (e.g. --model_size ultralarge ...).")
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    shards = _load_manifest(args.manifest)
    if args.max_shards and args.max_shards > 0:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise SystemExit("No shards to train.")

    stage_dir = Path(args.stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    current_weights = str(args.weights)
    stage_summaries: List[Dict[str, Any]] = []
    prev_stage_weights: str = ""
    prev_stage_meta: str = ""
    for idx, shard in enumerate(shards, start=1):
        shard_path = str(shard["path"])
        stage_name = f"stage_{idx:02d}_{Path(shard_path).stem}"
        stage_weights = str(stage_dir / f"{stage_name}.pth")
        stage_meta = str(stage_dir / f"{stage_name}.json")

        if args.resume and Path(stage_weights).exists() and Path(stage_meta).exists():
            print(f"[{idx}/{len(shards)}] Skipping completed stage: {shard_path}", flush=True)
            current_weights = stage_weights
            stage_summaries.append(
                {
                    "stage": idx,
                    "shard": shard_path,
                    "rows": shard.get("rows"),
                    "weights": stage_weights,
                    "meta": stage_meta,
                    "skipped": True,
                }
            )
            prev_stage_weights = stage_weights
            prev_stage_meta = stage_meta
            continue

        cmd = [args.python_exe]
        if args.child_unbuffered:
            cmd.append("-u")
        cmd += [args.script] + list(passthrough)
        cmd = _replace_arg(cmd, "--data", shard_path)
        cmd = _replace_arg(cmd, "--weights", current_weights)
        cmd = _replace_arg(cmd, "--output", stage_weights)
        cmd = _replace_arg(cmd, "--meta", stage_meta)
        # Ensure manifest isn't passed through to per-shard runs unless explicitly desired.
        filtered: List[str] = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "--data_manifest":
                i += 2
                continue
            filtered.append(cmd[i])
            i += 1
        cmd = filtered

        t0 = time.time()
        print(f"[{idx}/{len(shards)}] Training on shard: {shard_path}", flush=True)
        env = os.environ.copy()
        if args.child_unbuffered:
            env["PYTHONUNBUFFERED"] = "1"
        completed = subprocess.run(cmd, check=False, env=env)
        if completed.returncode != 0:
            raise SystemExit(f"Shard training failed (stage {idx}) with return code {completed.returncode}")
        elapsed_s = max(0.0, time.time() - t0)
        print(f"[{idx}/{len(shards)}] Completed in {elapsed_s/60.0:.1f} min", flush=True)

        current_weights = stage_weights
        stage_summaries.append(
            {
                "stage": idx,
                "shard": shard_path,
                "rows": shard.get("rows"),
                "weights": stage_weights,
                "meta": stage_meta,
            }
        )

        # Save disk on long sequential runs by deleting the previous stage once the next stage finishes.
        if not args.keep_stage_artifacts and prev_stage_weights:
            for old_path in (prev_stage_weights, prev_stage_meta):
                try:
                    if old_path and Path(old_path).exists():
                        Path(old_path).unlink()
                except Exception:
                    pass
        try:
            free_gb = shutil.disk_usage(str(stage_dir.resolve().anchor or stage_dir)).free / (1024 ** 3)
            print(f"[{idx}/{len(shards)}] Free disk: {free_gb:.2f} GB", flush=True)
        except Exception:
            pass
        prev_stage_weights = stage_weights
        prev_stage_meta = stage_meta

    out_weights = Path(args.output)
    out_meta = Path(args.meta)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(current_weights, out_weights)
    shutil.copyfile(stage_summaries[-1]["meta"], out_meta)

    wrapper_meta_path = out_meta.with_suffix(out_meta.suffix + ".sequential_manifest.json")
    wrapper_meta = {
        "manifest": str(args.manifest),
        "num_shards_trained": len(shards),
        "initial_weights": str(args.weights),
        "final_weights": str(out_weights),
        "final_meta": str(out_meta),
        "stages": stage_summaries,
    }
    wrapper_meta_path.write_text(json.dumps(wrapper_meta, indent=2), encoding="utf-8")

    print(f"Final checkpoint: {out_weights}", flush=True)
    print(f"Final metadata: {out_meta}", flush=True)
    print(f"Wrapper stage metadata: {wrapper_meta_path}", flush=True)


if __name__ == "__main__":
    main()
