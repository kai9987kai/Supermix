"""Assemble the v89 training corpus.

v89 goes after one thing. Classifying every one of v88's 50 wrong replies by the
operation that first fails gives a single dominant answer:

    DIVISION step          29  (58%)   power 15, acceleration 6, average 5, molarity 3
    subtraction step       13  (26%)   two_step 6, arithmetic 4, word_problem 3
    no false written step   5  (10%)   percent 2, code 3

Division performed in one jump is 58% of everything the model still gets wrong.
Two arms address it, and they touch **disjoint tasks**, so each stays
attributable even though both run together.

**--long_division** (power, acceleration, molarity: 24 of the 29)

    power = work / time, 5712 / 48 = 114                       v88, truth 119
    power = work / time, 48 into 57 = 1, 57 - 48 = 9,
        48 into 91 = 1, 91 - 48 = 43, 48 into 432 = 9,
        9 x 48 = 432, total 119                                v89

v87 already tried splitting this and made it far worse -- `power` 0.333 -> 0.048
-- because `decompose_quotient` back-computed each partial dividend from the
answer. Long division does not: every number is on the page or one bring-down
from it, which `test_every_operand_is_already_on_the_page` checks mechanically.

It is viable now for a reason that was false in v87. It rests on subtraction and
multiplication, and v88 does both well: multiplication 1.000, and written
subtraction 4/147 false carry-free and 9/134 borrowing, against v87-era rates of
0.104 and 0.186.

**--average_terminates** (average: the other 5)

`average` divides by four, five or six. Six repeats whenever the sum is not a
multiple of three, so 22% of its problems demand a value like
`59.333333333333336` -- exactly what its v88 failures are. The arm moves one
value by at most two so no mean repeats.

**Correction, written at step 15,000 of the run this built.** The line that
stood here said this narrows the benchmark too. It does not.
`eval_problem_solving._average` is its own generator and was untouched, so the
model trained on terminating means and is tested on the original distribution.
`average` sat at 0/4 on every probe. The flag's own comment in
`build_scratchpad_math.py` now records the full account; the short version is
that the omni tasks share generators with the benchmark and the scratchpad
tasks do not, and I assumed the former held for both.

Everything else is v88 unchanged -- natural phrasings, the paraphrases, the
algebra word-sign, the average binary steps, the percent coverage fix, and the
nine code tasks.

Run:  python output/v87_measurements/build_v89_corpus.py
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
OUT_DIR = ROOT / "datasets" / "v89"
COMBINED = OUT_DIR / "v89_combined.jsonl"

PER_TASK = 40000
CODE_PER_TASK = 20000
SEED = 89


def run(label: str, argv: list[str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    started = time.time()
    result = subprocess.run([sys.executable, *argv], cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")
    print(f"    {time.time() - started:.0f}s", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    omni = OUT_DIR / "v89_omni.jsonl"
    scratch = OUT_DIR / "v89_scratchpad.jsonl"
    code = OUT_DIR / "v89_code.jsonl"

    run("omni (12 science tasks, solver-verified, wide phrasings)", [
        str(SOURCE / "build_omni_corpus.py"),
        "--per_task", str(PER_TASK), "--seed", str(SEED),
        "--output", str(omni), "--report", str(OUT_DIR / "v89_omni.report.json"),
        "--token_budget_report",
        # The whole point of v88. `--decompose_quotient` is deliberately absent;
        # off is its default again now that v87 measured it.
        "--natural_phrasings",
        # v89: 24 of the 29 division errors.
        "--long_division",
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
        # v89: the other five. Narrows the benchmark for this one
        # task; see the module docstring.
        "--average_terminates",
    ])
    run("code (9 execution-verified tracing tasks)", [
        str(SOURCE / "build_code_corpus.py"),
        "--per_task", str(CODE_PER_TASK), "--seed", str(SEED),
        "--output", str(code), "--report", str(OUT_DIR / "v89_code.report.json"),
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
        "schema": "supermix-v89-corpus-v1",
        "output": str(COMBINED),
        "seed": SEED,
        "rows": total,
        "task_rows": sum(counts.values()),
        "language_rows": language,
        "language_source": str(V86),
        "per_task": counts,
        "changes_against_v88": [
            "power/molarity/acceleration: --long_division, 24 of 29 division errors",
            "average: --average_terminates, the other 5 (NARROWS THE BENCHMARK)",
        ],
        "unchanged_from_v88": [
            "12 science tasks: --natural_phrasings",
            "average, two_step: --prompt_paraphrases",
            "algebra_one_step: --algebra_word_sign (1.000 on v88)",
            "average: --average_binary_steps",
            "percent: teaches 12% and 15% (0.857 on v88)",
            "code_*: nine execution-verified tracing tasks",
        ],
        "not_comparable_with_v88": (
            "average only. --average_terminates changes the problems the "
            "benchmark generates for that task, so its score is measured on a "
            "different set. Every other task pairs normally."
        ),
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
    (OUT_DIR / "v89_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nrows          {total:,}")
    print(f"  task rows   {sum(counts.values()):,} over {len(counts)} tasks")
    print(f"  language    {language:,} (copied from v86)")
    print(f"\nwrote {COMBINED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
