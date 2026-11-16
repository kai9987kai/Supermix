"""Build the deterministic v97 Cognitive Fabric stratified evolution JSONL corpus."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from build_v96_repair_mix import _sample_tasks_from_jsonl, _sha256  # noqa: E402

ROOT = SOURCE_DIR.parent
DEFAULT_OUTPUT = ROOT / "datasets" / "v97" / "v97_evolution_mix.jsonl"
DEFAULT_MANIFEST = ROOT / "datasets" / "v97" / "v97_manifest.json"


def build_v97_evolution_mix(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    seed: int = 97,
) -> Dict[str, Any]:
    """Assemble the 4,800-row stratified v97 continuation/evolution mix."""

    sources = [
        (
            ROOT / "datasets" / "v93" / "v93_connectome.jsonl",
            {"cns_type_count": 400, "cns_side_count": 400, "cns_pair_synapses": 400},
            19,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_omni.jsonl",
            {
                "impulse": 150,
                "ohms_current": 150,
                "spring_energy": 150,
                "permutations": 150,
                "final_velocity": 150,
            },
            23,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_code.jsonl",
            {"code_range_sum": 150, "code_list_count": 150, "code_neg_index": 150},
            17,
        ),
        (
            ROOT / "datasets" / "v89" / "v89_combined.jsonl",
            {
                "addition": 400,
                "subtraction": 400,
                "percent": 400,
                "acceleration": 400,
                "power": 200,
                "two_step": 200,
                "average": 200,
                "algebra_one_step": 200,
            },
            29,
        ),
    ]

    rows: List[Dict[str, Any]] = []
    source_records: List[Dict[str, Any]] = []
    for src_path, quotas, stride in sources:
        picked = _sample_tasks_from_jsonl(src_path, quotas, stride=stride)
        rows.extend(picked)
        source_records.append(
            {
                "path": src_path.relative_to(ROOT).as_posix(),
                "quotas": dict(quotas),
                "rows_selected": len(picked),
                "sha256": _sha256(src_path),
            }
        )

    rng = random.Random(int(seed))
    rng.shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter(str(r.get("task", "")) for r in rows)
    domain_counts = Counter(str(r.get("domain", "")) for r in rows)
    manifest = {
        "schema": "supermix-v97-evolution-mix-v1",
        "seed": int(seed),
        "total_rows": len(rows),
        "output_path": output_path.relative_to(ROOT).as_posix(),
        "output_sha256": _sha256(output_path),
        "task_counts": dict(sorted(task_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "sources": source_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v97 Cognitive Fabric evolution corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=97)
    args = parser.parse_args(argv)
    manifest = build_v97_evolution_mix(args.output, args.manifest, seed=args.seed)
    print(f"Built {manifest['total_rows']} rows -> {manifest['output_path']} (sha256={manifest['output_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
