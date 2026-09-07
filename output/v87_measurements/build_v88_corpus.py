"""Assemble the v88 training corpus.

v88 does two things: it undoes what v87 measured as harmful, and it turns on a
capability that has been written, flagged and disconnected since v87.

**Reverted, because v87 measured them and they cost accuracy.**

    power, molarity, acceleration   decompose_quotient OFF. It split the
                                    quotient by place value, so each partial
                                    dividend could only be obtained by already
                                    knowing the answer, and the model duly
                                    invented them: power 0.333 -> 0.048.
    percent                         the written sum OFF. Making the final
                                    addition explicit did not make it doable --
                                    both parts come out right and the sum it was
                                    forced to state does not: 0.476 -> 0.286.
                                    The COVERAGE fix stays; that closed a real
                                    hole and is orthogonal.

**Kept, because v87 measured them and they worked.**

    algebra_one_step                0.476 -> 0.952
    average                         0.048 -> 0.286
    percent                         teaches 12% and 15%, which the benchmark
                                    asks for in a third of its problems
    code_*                          nine execution-verified tracing tasks, 0.894

**New: the prompt bank the model was never trained on.**

    12 science tasks                --natural_phrasings widens four or five
                                    textbook templates to fifteen, including the
                                    register a question actually arrives in
    average, two_step               --prompt_paraphrases, the same idea for the
                                    two scratchpad tasks that have one template

`natural_phrasings.py` has held 112 hand-written forms since v87 and not one
generator passed the `_task` argument that reaches them, so every corpus ever
built used the narrow templates. Wiring it is a single-variable change: over
3,996 rows every answer and worked response is bit-identical and only the prompt
moves, the solver verifies all of them, and the median prompt actually gets one
token shorter.

The last three phrasings of every task are withheld by
`natural_phrasings.HELD_OUT_PER_TASK` and never reach this corpus, so
`eval_natural_phrasing.py` can ask whether the model learned the task or the
template. On v87 that gap is 0.7933 against 0.5800, McNemar p = 0.0000.

Run:  python output/v87_measurements/build_v88_corpus.py
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
OUT_DIR = ROOT / "datasets" / "v88"
COMBINED = OUT_DIR / "v88_combined.jsonl"

PER_TASK = 40000
CODE_PER_TASK = 20000
SEED = 88


def run(label: str, argv: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    print(f"    {time.time() - started:.0f}s", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    omni = OUT_DIR / "v88_omni.jsonl"
    scratch = OUT_DIR / "v88_scratchpad.jsonl"
    code = OUT_DIR / "v88_code.jsonl"

    run("omni (12 science tasks, solver-verified, wide phrasings)", [
        str(SOURCE / "build_omni_corpus.py"),
        "--per_task", str(PER_TASK), "--seed", str(SEED),
        "--output", str(omni), "--report", str(OUT_DIR / "v88_omni.report.json"),
        "--token_budget_report",
        # The whole point of v88. `--decompose_quotient` is deliberately absent;
        # off is its default again now that v87 measured it.
        "--natural_phrasings",
    ])
    run("scratchpad (10 arithmetic tasks)", [
        str(SOURCE / "build_scratchpad_math.py"),
        "--target", str(PER_TASK * 10), "--seed", str(SEED),
        "--output", str(scratch),
        # v86 and v87 were both built with this on.
        "--decompose-inner",
        # v87 measured both of these and both worked; kept unchanged.
        "--average_binary_steps",
        "--algebra_word_sign",
        # New in v88: `average` and `two_step` each carry a single prompt
        # template, and they are two of the three weakest arithmetic tasks.
        "--prompt_paraphrases",
    ])
    run("code (9 execution-verified tracing tasks)", [
        str(SOURCE / "build_code_corpus.py"),
        "--per_task", str(CODE_PER_TASK), "--seed", str(SEED),
        "--output", str(code), "--report", str(OUT_DIR / "v88_code.report.json"),
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
        "schema": "supermix-v88-corpus-v1",
        "output": str(COMBINED),
        "seed": SEED,
        "rows": total,
        "task_rows": sum(counts.values()),
        "language_rows": language,
        "language_source": str(V86),
        "per_task": counts,
        "changes_against_v87": [
            "power/molarity/acceleration: decompose_quotient REVERTED (0.048)",
            "percent: the written sum REVERTED (0.286); coverage fix kept",
            "12 science tasks: --natural_phrasings, 5 -> 15 prompt forms",
            "average, two_step: --prompt_paraphrases",
        ],
        "unchanged_from_v87": [
            "algebra_one_step: --algebra_word_sign (0.952)",
            "average: --average_binary_steps (0.286, up from 0.048)",
            "percent: teaches 12% and 15%",
            "code_*: nine execution-verified tracing tasks (0.894)",
        ],
        "held_out_from_training": (
            "natural_phrasings.HELD_OUT_PER_TASK withholds the last three forms "
            "of each task -- 36 phrasings over 12 tasks -- so "
            "eval_natural_phrasing.py measures generalisation and not recall. "
            "test_natural_phrasings.py fails if one reaches a corpus."
        ),
        "control_group": (
            "The eleven tasks at 1.000 on v86 have unchanged formats. A fall in "
            "them is exposure dilution, not a format. The phrasing arm touches "
            "their prompts but provably not their answers."
        ),
    }
    (OUT_DIR / "v88_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nrows          {total:,}")
    print(f"  task rows   {sum(counts.values()):,} over {len(counts)} tasks")
    print(f"  language    {language:,} (copied from v86)")
    print(f"\nwrote {COMBINED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
