from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from v40_benchmax_core import average_checkpoints, inspect_checkpoint_compatibility, load_manifest
except ImportError:  # pragma: no cover
    from .v40_benchmax_core import average_checkpoints, inspect_checkpoint_compatibility, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average compatible checkpoints for the v40_benchmax model soup.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Checkpoint path. Repeat for multiple inputs.")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--metadata_path", default="")
    parser.add_argument("--manifest", default=str(Path(__file__).resolve().parent / "v40_benchmax_manifest.json"))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_manifest(args.manifest)
    report = inspect_checkpoint_compatibility([Path(item) for item in args.checkpoint])
    output_path = Path(args.output_path).resolve()
    metadata_path = Path(args.metadata_path).resolve() if args.metadata_path.strip() else output_path.with_suffix(output_path.suffix + ".json")
    if args.dry_run:
        report["dry_run"] = True
        report["output_path"] = str(output_path)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return 0
    result = average_checkpoints([Path(item) for item in args.checkpoint], output_path, dry_run=False, metadata_path=metadata_path)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
