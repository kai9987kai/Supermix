"""Assemble the v87 training corpus.

v87 changes six formats and adds one family. Every change has a measurement
behind it taken on the v86 checkpoint this week, and every one of them is
confined to tasks that scored below 0.75, so the eleven tasks already at 1.000
act as a control group: if they fall, the cause is exposure dilution rather
than any format.

    power, molarity, acceleration   one-jump division -> place-value quotient
    percent                         teaches 12% and 15%, which the benchmark
                                    asks for in a third of its problems and
                                    the corpus never contained
    percent                         writes the sum of its two parts
    average                         writes its running sum as equations
    algebra_one_step                resolves the sign in words, splits by place
    code_*                          nine execution-verified tracing tasks,
                                    built but never trained

The composition mirrors v86 exactly except for the code family: 22 tasks at
40,000 rows, plus the same 96,108 language rows, which are copied out of the
v86 file rather than regenerated so that the one part of the corpus nothing
changed is bit-identical.

Run:  python output/v87_measurements/build_v87_corpus.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source"
sys.path.insert(0, str(SOURCE))

V86 = ROOT / "datasets" / "v86" / "v86_combined.jsonl"
OUT_DIR = ROOT / "datasets" / "v87"
COMBINED = OUT_DIR / "v87_combined.jsonl"

PER_TASK = 40000
CODE_PER_TASK = 20000
SEED = 87


def run(label: str, argv: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    print(f"    {time.time() - started:.0f}s", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    omni = OUT_DIR / "v87_omni.jsonl"
    scratch = OUT_DIR / "v87_scratchpad.jsonl"
    code = OUT_DIR / "v87_code.jsonl"

    run("omni (12 science tasks, solver-verified)", [
        str(SOURCE / "build_omni_corpus.py"),
        "--per_task", str(PER_TASK), "--seed", str(SEED),
        "--output", str(omni), "--report", str(OUT_DIR / "v87_omni.report.json"),
        "--token_budget_report",
    ])
    run("scratchpad (10 arithmetic tasks)", [
        str(SOURCE / "build_scratchpad_math.py"),
        "--target", str(PER_TASK * 10), "--seed", str(SEED),
        "--output", str(scratch),
        # v86 was built with this on; the percent replies show the model
        # producing the tens-and-units split it only teaches.
        "--decompose-inner",
        # Both new in v87. Their motivation is in the flag help and their
        # effect is what this run measures.
        "--average_binary_steps",
        "--algebra_word_sign",
    ])
    run("code (9 execution-verified tracing tasks)", [
        str(SOURCE / "build_code_corpus.py"),
        "--per_task", str(CODE_PER_TASK), "--seed", str(SEED),
        "--output", str(code), "--report", str(OUT_DIR / "v87_code.report.json"),
    ])

    print("\n=== combining ===", flush=True)
    started = time.time()
    counts: dict[str, int] = {}
    language = 0
    with COMBINED.open("w", encoding="utf-8") as out:
        for part in (omni, scratch, code):
            with part.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    counts[record.get("task", "?")] = counts.get(
                        record.get("task", "?"), 0) + 1
                    out.write(line if line.endswith("\n") else line + "\n")
        # The language rows carry no `task` field. Copying them from v86 keeps
        # the one component nothing changed byte-identical, so a difference
        # between the runs cannot come from here.
        with V86.open(encoding="utf-8") as handle:
            for line in handle:
                if '"task"' in line:
                    continue
                language += 1
                out.write(line)
    print(f"    {time.time() - started:.0f}s", flush=True)

    total = sum(counts.values()) + language
    manifest = {
        "schema": "supermix-v87-corpus-v1",
        "output": str(COMBINED),
        "seed": SEED,
        "rows": total,
        "task_rows": sum(counts.values()),
        "language_rows": language,
        "language_source": str(V86),
        "per_task": counts,
        "changes_against_v86": [
            "power/molarity/acceleration: decompose_quotient by place value",
            "percent: percentages cover the benchmark's [5,10,12,15,20,25]",
            "percent: the sum of the two parts is written",
            "average: --average_binary_steps",
            "algebra_one_step: --algebra_word_sign",
            "code_*: nine tasks added at 20,000 rows each",
        ],
        "control_group": (
            "The eleven tasks at 1.000 on v86 have unchanged formats. A fall "
            "in them is exposure dilution from the larger corpus, not a format."
        ),
    }
    (OUT_DIR / "v87_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nrows          {total:,}")
    print(f"  task rows   {sum(counts.values()):,} over {len(counts)} tasks")
    print(f"  language    {language:,} (copied from v86)")
    print(f"\nwrote {COMBINED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
