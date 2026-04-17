#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

from v40_benchmax_common import average_state_dicts, compatibility_report, json_dump


def _load_state(path: Path) -> Dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint did not contain a state_dict mapping: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Average compatible v40_benchmax checkpoints with a strict compatibility gate.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path to include. Repeat for multiple.")
    parser.add_argument("--checkpoint_dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--method", default="equal")
    args = parser.parse_args()

    checkpoints: List[Path] = [Path(path).resolve() for path in args.checkpoint]
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        checkpoints.extend(sorted(checkpoint_dir.glob("*.pth")))
    if len(checkpoints) < 2:
        raise SystemExit("Need at least two checkpoints for soup.")

    state_dicts = [_load_state(path) for path in checkpoints]
    report = compatibility_report(state_dicts)
    if not report["ok"]:
        raise RuntimeError(f"Incompatible checkpoints: {report['incompatible']}")

    output_path = Path(args.output).resolve()
    summary_out = Path(args.summary_out).resolve()
    if args.dry_run:
        json_dump(
            summary_out,
            {
                "method": args.method,
                "dry_run": True,
                "checkpoints": [str(path) for path in checkpoints],
                "compatibility": report,
            },
        )
        print(json.dumps({"dry_run": True, "compatibility": report}, indent=2))
        return 0

    merged = average_state_dicts(state_dicts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    json_dump(
        summary_out,
        {
            "method": args.method,
            "dry_run": False,
            "output": str(output_path),
            "checkpoints": [str(path) for path in checkpoints],
            "compatibility": report,
        },
    )
    print(f"[v40-benchmax] wrote merged checkpoint to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
