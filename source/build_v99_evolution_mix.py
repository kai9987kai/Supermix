"""Build the balanced 41-task + conversational v99 Cognitive Fabric evolution corpus."""

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
DEFAULT_OUTPUT = ROOT / "datasets" / "v99" / "v99_evolution_mix.jsonl"
DEFAULT_MANIFEST = ROOT / "datasets" / "v99" / "v99_manifest.json"


def build_v99_evolution_mix(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    seed: int = 99,
) -> Dict[str, Any]:
    """Assemble the stratified v99 evolution corpus covering all 41 task families plus chat."""

    sources = [
        (
            ROOT / "datasets" / "v93" / "v93_connectome.jsonl",
            {"cns_type_count": 180, "cns_side_count": 180, "cns_pair_synapses": 180},
            29,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_omni.jsonl",
            {
                "impulse": 120,
                "ohms_current": 120,
                "spring_energy": 120,
                "permutations": 120,
                "final_velocity": 120,
            },
            23,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_code.jsonl",
            {"code_range_sum": 120, "code_list_count": 120, "code_neg_index": 120},
            19,
        ),
        (
            ROOT / "datasets" / "v89" / "v89_combined.jsonl",
            {
                "subtraction": 450,
                "percent": 450,
                "two_step": 450,
                "acceleration": 450,
                "average": 350,
                "electrical_power": 300,
                "addition": 250,
                "code_conditional": 250,
                "code_divmod": 250,
                "algebra_one_step": 180,
                "division": 180,
                "multiplication": 180,
                "power": 180,
                "word_problem": 180,
                "sequence": 140,
                "arithmetic_series": 140,
                "combination": 140,
                "force": 100,
                "kinetic_energy": 100,
                "molarity": 100,
                "momentum": 100,
                "voltage": 100,
                "wave_speed": 100,
                "work": 100,
                "code_index": 100,
                "code_list_extreme": 100,
                "code_list_sum": 100,
                "code_loop_add": 100,
                "code_loop_subtract": 100,
                "code_nested_loop": 100,
                "code_while_accumulate": 100,
                "": 120,
            },
            17,
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
        "schema": "supermix-v99-evolution-mix-v1",
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
    parser = argparse.ArgumentParser(description="Build the v99 Cognitive Fabric evolution corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args(argv)
    manifest = build_v99_evolution_mix(args.output, args.manifest, seed=args.seed)
    print(f"Built {manifest['total_rows']} rows -> {manifest['output_path']} (sha256={manifest['output_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
