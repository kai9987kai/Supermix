"""Build the deterministic v96 repair/continuation JSONL corpus and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "datasets" / "v96" / "v96_repair_mix.jsonl"
DEFAULT_MANIFEST = ROOT / "datasets" / "v96" / "v96_manifest.json"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _sample_tasks_from_jsonl(
    path: Path,
    quotas: Mapping[str, int],
    *,
    stride: int = 1,
) -> List[Dict[str, Any]]:
    """Stream ``path`` and collect exact per-task deterministic quotas."""

    if not path.is_file():
        raise FileNotFoundError(f"missing source corpus: {path}")
    remaining = {task: int(count) for task, count in quotas.items()}
    seen = {task: 0 for task in quotas}
    collected: List[Dict[str, Any]] = []
    stride_n = max(1, int(stride))
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task", ""))
            if task not in remaining or remaining[task] <= 0:
                continue
            seen[task] += 1
            if (seen[task] - 1) % stride_n != 0:
                continue
            collected.append(row)
            remaining[task] -= 1
            if all(left <= 0 for left in remaining.values()):
                break
    missing = {task: left for task, left in remaining.items() if left > 0}
    if missing:
        raise RuntimeError(f"insufficient rows in {path} for task quotas: {missing}")
    return collected


def build_v96_repair_mix(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    seed: int = 96,
) -> Dict[str, Any]:
    """Assemble the 3,200-row stratified v96 repair/continuation mix."""

    sources = [
        (
            ROOT / "datasets" / "v93" / "v93_connectome.jsonl",
            {"cns_type_count": 400, "cns_side_count": 400, "cns_pair_synapses": 400},
            17,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_omni.jsonl",
            {
                "impulse": 160,
                "ohms_current": 160,
                "spring_energy": 160,
                "permutations": 160,
                "final_velocity": 160,
            },
            19,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_code.jsonl",
            {"code_range_sum": 134, "code_list_count": 133, "code_neg_index": 133},
            13,
        ),
        (
            ROOT / "datasets" / "v89" / "v89_combined.jsonl",
            {"acceleration": 200, "power": 200, "two_step": 200, "average": 200},
            23,
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

    task_counts = dict(sorted(Counter(str(r.get("task", "")) for r in rows).items()))
    domain_counts = dict(sorted(Counter(str(r.get("domain", "")) for r in rows).items()))
    output_resolved = output_path.resolve()
    root_resolved = ROOT.resolve()
    rel_output = (
        output_resolved.relative_to(root_resolved).as_posix()
        if output_resolved.is_relative_to(root_resolved)
        else str(output_resolved)
    )
    manifest = {
        "schema": "supermix-v96-repair-mix-v1",
        "seed": int(seed),
        "total_rows": len(rows),
        "output_path": rel_output,
        "output_sha256": _sha256(output_path),
        "task_counts": task_counts,
        "domain_counts": domain_counts,
        "sources": source_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=96)
    args = parser.parse_args(argv)
    manifest = build_v96_repair_mix(args.output, args.manifest, seed=args.seed)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "total_rows": manifest["total_rows"],
                "output_path": manifest["output_path"],
                "output_sha256": manifest["output_sha256"],
                "task_counts": manifest["task_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
