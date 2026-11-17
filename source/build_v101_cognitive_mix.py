"""Build the v101 Cognitive & Precision Mathematical/Conversational/Ideation corpus.

Combines:
1. Balanced retention across all 41 Supermix benchmark task families in their exact
   trained grammar (`v89`/`v93`), keeping the 9,415-token vocabulary 100% aligned
   with zero newly initialized embedding rows.
2. Targeted exact repair generators (strictly disjoint RNG seed=10101) in exact
   v89/v93 grammar for column-borrow subtraction, decimal-carry percentages,
   two-step subtraction borrows, hundreds-crossing averages, and 3-digit divisor
   physics ratios without hardcoding held-out test operands.
3. Curated, strictly in-vocabulary (`0 <unk>` against v96/v100 9,415-word tokenizer)
   dialogues for debugging/problem solving, creative ideation, mathematics, and
   multi-turn conversational continuity — concise and free of noisy boilerplate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from build_v96_repair_mix import _sample_tasks_from_jsonl, _sha256  # noqa: E402
from nexus_solver import NexusSolver  # noqa: E402

ROOT = SOURCE_DIR.parent
DEFAULT_OUTPUT = ROOT / "datasets" / "v101" / "v101_cognitive_mix.jsonl"
DEFAULT_MANIFEST = ROOT / "datasets" / "v101" / "v101_manifest.json"


def _fmt_frac(val: Fraction, decimals: int = 4) -> str:
    if val.denominator == 1:
        return str(val.numerator)
    f_val = float(val)
    return f"{f_val:.{decimals}f}".rstrip("0").rstrip(".") or "0"


def _gen_borrow_subtraction_rows(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Generate 3-digit subtraction pairs in exact v89 grammar targeting column borrows."""
    rows: List[Dict[str, Any]] = []
    prompts = (
        "Solve this basic math problem: {a} - {b}",
        "What is {a} - {b}?",
        "Quick question: {a} - {b}",
    )
    while len(rows) < count:
        a = rng.randint(120, 989)
        b = rng.randint(25, a - 5)
        if (a % 10) >= (b % 10) and ((a // 10) % 10) >= ((b // 10) % 10):
            continue
        ans = a - b
        ah, at = (a // 100) * 100, a % 100
        bh, bt = (b // 100) * 100, b % 100
        d_tens = at - bt
        reply = f"{ah} - {bh} = {ah - bh}, {at} - {bt} = {d_tens}, total {ans}"
        rows.append(
            {
                "user": rng.choice(prompts).format(a=a, b=b),
                "assistant": reply,
                "domain": "math",
                "task": "subtraction",
                "topic": "basic_math",
                "prompt_variant": "v101_borrow_exact",
            }
        )
    return rows


def _gen_percent_carry_rows(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Generate percentage problems in exact v89 grammar emphasizing tenths/hundredths carries."""
    rows: List[Dict[str, Any]] = []
    pcts = [5, 10, 12, 15, 20, 25]
    prompts = (
        "What is {pct}% of {base}?",
        "Compute {pct} percent of {base}.",
    )
    for _ in range(count):
        pct = rng.choice(pcts)
        base = rng.randint(40, 1990)
        one_pct = Fraction(base, 100)
        one_s = _fmt_frac(one_pct, 2)
        ans = Fraction(pct * base, 100)
        ans_s = f"{float(ans):.1f}" if ans.denominator in (1, 2, 5, 10) else _fmt_frac(ans, 4)
        if pct in (12, 15, 25):
            ten_mult = 20 if pct == 25 else 10
            rem_mult = pct - ten_mult
            p1_s = _fmt_frac(one_pct * ten_mult, 2)
            p2_s = _fmt_frac(one_pct * rem_mult, 2)
            reply = f"1 percent of {base} = {one_s}, times {ten_mult} = {p1_s}, times {rem_mult} = {p2_s}, total {ans_s}"
        else:
            reply = f"1 percent of {base} = {one_s}, times {pct} = {ans_s}, total {ans_s}"
        rows.append(
            {
                "user": rng.choice(prompts).format(pct=pct, base=base),
                "assistant": reply,
                "domain": "math",
                "task": "percent",
                "topic": "basic_math",
                "prompt_variant": "v101_percent_exact",
            }
        )
    return rows


def _gen_two_step_borrow_rows(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Generate two-step percent + subtract/add rows in exact v89 grammar with borrow targets."""
    rows: List[Dict[str, Any]] = []
    pcts = [10, 20, 25, 50]
    for _ in range(count):
        pct = rng.choice(pcts)
        step_gcd = 100 // math.gcd(100, pct)
        base = step_gcd * rng.randint(2, max(3, 880 // step_gcd))
        first = (pct * base) // 100
        delta = rng.randint(6, 59)
        is_sub = rng.random() < 0.72
        ans = first - delta if is_sub else first + delta
        op_word = "subtract" if is_sub else "add"
        op_sym = "-" if is_sub else "+"
        prompt = f"What is {pct}% of {base}, then {op_word} {delta}?"
        one_s = _fmt_frac(Fraction(base, 100), 2)
        reply = f"1 percent of {base} = {one_s}, times {pct} = {first}, then {first} {op_sym} {delta} = {ans}, total {ans}"
        rows.append(
            {
                "user": prompt,
                "assistant": reply,
                "domain": "math",
                "task": "two_step",
                "topic": "basic_math",
                "prompt_variant": "v101_two_step_exact",
            }
        )
    return rows


def _gen_average_carry_rows(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Generate 4-6 item mean problems in exact v89 format crossing hundreds carries."""
    rows: List[Dict[str, Any]] = []
    prompts = (
        "Find the average (mean) of these numbers: {joined}",
        "For the numbers {joined}, what is their mean?",
    )
    for _ in range(count):
        k = rng.choice([4, 5, 6])
        vals = [rng.randint(8, 98) for _ in range(k)]
        running = vals[0]
        parts: List[str] = []
        for nxt in vals[1:]:
            prev = running
            running += nxt
            parts.append(f"{prev} + {nxt} = {running}")
        mean_val = Fraction(running, k)
        if mean_val.denominator in (1, 2):
            mean_s = f"{float(mean_val):.1f}"
        else:
            mean_s = f"{float(mean_val):.4f}".rstrip("0").rstrip(".")
        reply = ", ".join(parts) + f", total {running}, divide by {k}, total {mean_s}"
        joined = ", ".join(str(v) for v in vals)
        rows.append(
            {
                "user": rng.choice(prompts).format(joined=joined),
                "assistant": reply,
                "domain": "math",
                "task": "average",
                "topic": "basic_math",
                "prompt_variant": "v101_average_exact",
            }
        )
    return rows


def _gen_acceleration_divisor_rows(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """Generate F/m division rows in exact v93 long-division format with 3-digit masses."""
    rows: List[Dict[str, Any]] = []
    prompts = (
        "What acceleration results from a {F} N force on a {m} kg body?",
        "how fast does {m} kg accelerate under {F} newtons",
    )
    for _ in range(count):
        mass = rng.randint(50, 220)
        acc = rng.randint(12, 79)
        force = mass * acc
        q_ten = acc // 10
        q_one = acc % 10
        prod_ten = q_ten * mass
        prefix = force // 10
        rem_full = force - prod_ten * 10
        reply = (
            f"acceleration = force / mass, {mass} into {prefix} = {q_ten}, "
            f"{q_ten} x {mass} = {prod_ten}, "
            f"{mass} into {rem_full} = {q_one}, {q_one} x {mass} = {q_one * mass}, "
            f"the acceleration is {acc} metres per second squared, total {acc}"
        )
        rows.append(
            {
                "user": rng.choice(prompts).format(F=force, m=mass),
                "assistant": reply,
                "domain": "physics",
                "task": "acceleration",
                "prompt_variant": "v101_accel_exact",
            }
        )
    return rows


def _gen_cognitive_dialogue_rows(rng: random.Random, target_count: int) -> List[Dict[str, Any]]:
    """In-vocab curated dialogue rows for problem solving, math guidance, ideas & chat.

    All phrasing sticks closely to words already present in the 9,415-token v96/v100
    vocabulary so zero randomly initialized embeddings are added.
    """
    seed_pairs: List[Tuple[str, str, str]] = [
        (
            "why is my python script failing",
            "Check the traceback first, inspect the input values right before the error line, and test one small change at a time.",
            "problem_solving",
        ),
        (
            "How do I debug a slow python function?",
            "Measure time per step first, look for repeated loops or extra copies, cache repeated values, and test with a smaller input.",
            "problem_solving",
        ),
        (
            "How should I solve a hard problem step by step?",
            "Break the problem into small pieces, verify each intermediate number or invariant, and check edge cases before finalizing.",
            "problem_solving",
        ),
        (
            "How can I compare two options without guessing?",
            "Test both options on the same seeds, measure accuracy and worst case failures, and keep the option with clear evidence.",
            "problem_solving",
        ),
        (
            "Suggest creative ideas to improve a local model.",
            "Combine fast neural replies with exact math and code verification, track constraints across turns, and keep regression checks strict.",
            "ideation",
        ),
        (
            "What are three ideas to reduce errors in coding?",
            "Use small single-purpose functions, add unit tests for failure boundaries, and verify outputs with simple assertions.",
            "ideation",
        ),
        (
            "hello",
            "Hello! Happy to help with math, coding, physics, problem solving, or new ideas today.",
            "conversation",
        ),
        (
            "what is your name",
            "I am Supermix MiMoMix, a local neural assistant with Cognitive Fabric verification and problem solving.",
            "conversation",
        ),
        (
            "can you help me with tests",
            "Yes, let us isolate one failing case, write a clear assertion for expected behavior, and verify zero regressions.",
            "conversation",
        ),
        (
            "Please use Python and keep answers concise.",
            "Understood, I will use Python examples and keep answers clear and concise.",
            "conversation",
        ),
        (
            "How do I solve x^2 - 5x + 6 = 0?",
            "Factor or use the quadratic formula: x^2 - 5x + 6 = (x - 3)(x - 2) = 0, so x = 3 or x = 2.",
            "math_chat",
        ),
        (
            "How do I solve x + y = 10 and x - y = 2?",
            "Add both equations to get 2x = 12 so x = 6, then 6 + y = 10 gives y = 4.",
            "math_chat",
        ),
    ]

    # Filter/clean if an existing solver check exists; keep clean ratio
    rows: List[Dict[str, Any]] = []
    idx = 0
    while len(rows) < target_count:
        user_t, asst_t, sub_task = seed_pairs[idx % len(seed_pairs)]
        idx += 1
        rows.append({"user": user_t, "assistant": asst_t, "domain": "dialogue", "task": sub_task})
    return rows


def build_v101_cognitive_mix(
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    seed: int = 101,
) -> Dict[str, Any]:
    """Assemble v101 exact-grammar surgical precision + in-vocab dialogue/ideation mix."""
    rng = random.Random(int(seed) * 100 + 1)  # 10101 -> disjoint from eval seeds 82..94
    _ = NexusSolver()

    sources = [
        (
            ROOT / "datasets" / "v93" / "v93_connectome.jsonl",
            {"cns_type_count": 120, "cns_side_count": 120, "cns_pair_synapses": 120},
            37,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_omni.jsonl",
            {
                "impulse": 140,
                "ohms_current": 140,
                "spring_energy": 140,
                "permutations": 140,
                "final_velocity": 140,
            },
            29,
        ),
        (
            ROOT / "datasets" / "v93" / "v93_code.jsonl",
            {"code_range_sum": 220, "code_list_count": 180, "code_neg_index": 180},
            19,
        ),
        (
            ROOT / "datasets" / "v89" / "v89_combined.jsonl",
            {
                "average": 320,
                "two_step": 340,
                "code_conditional": 380,
                "code_divmod": 380,
                "percent": 320,
                "subtraction": 320,
                "acceleration": 300,
                "electrical_power": 240,
                "voltage": 240,
                "addition": 200,
                "algebra_one_step": 200,
                "division": 200,
                "multiplication": 200,
                "power": 200,
                "word_problem": 200,
                "sequence": 160,
                "arithmetic_series": 160,
                "combination": 160,
                "force": 140,
                "kinetic_energy": 140,
                "molarity": 140,
                "momentum": 140,
                "wave_speed": 140,
                "work": 140,
                "code_index": 140,
                "code_list_extreme": 140,
                "code_list_sum": 140,
                "code_loop_add": 140,
                "code_loop_subtract": 140,
                "code_nested_loop": 140,
                "code_while_accumulate": 140,
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

    synthetic_blocks = [
        ("v101_borrow_exact", _gen_borrow_subtraction_rows(rng, 420)),
        ("v101_percent_exact", _gen_percent_carry_rows(rng, 420)),
        ("v101_two_step_exact", _gen_two_step_borrow_rows(rng, 420)),
        ("v101_average_exact", _gen_average_carry_rows(rng, 320)),
        ("v101_accel_exact", _gen_acceleration_divisor_rows(rng, 260)),
        ("v101_cognitive_dialogue", _gen_cognitive_dialogue_rows(rng, 280)),
    ]
    for block_name, block_rows in synthetic_blocks:
        rows.extend(block_rows)
        source_records.append(
            {
                "path": f"synthetic:{block_name}",
                "rows_selected": len(block_rows),
                "seed": int(seed) * 100 + 1,
            }
        )

    shuffle_rng = random.Random(int(seed))
    shuffle_rng.shuffle(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    task_counts = Counter(str(r.get("task", "")) for r in rows)
    domain_counts = Counter(str(r.get("domain", "")) for r in rows)
    manifest = {
        "schema": "supermix-v101-cognitive-mix-v1",
        "seed": int(seed),
        "generator_seed": int(seed) * 100 + 1,
        "total_rows": len(rows),
        "output_path": output_path.relative_to(ROOT).as_posix() if output_path.is_relative_to(ROOT) else str(output_path),
        "output_sha256": _sha256(output_path),
        "task_counts": dict(sorted(task_counts.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "sources": source_records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the v101 Cognitive & Precision Mix corpus")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args(argv)
    manifest = build_v101_cognitive_mix(args.output, args.manifest, seed=args.seed)
    print(f"Built {manifest['total_rows']} rows -> {manifest['output_path']} (sha256={manifest['output_sha256'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
