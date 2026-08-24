"""Inspect the deliberate reasoning engine.

This is a prompt-free, non-executing design tool. It does not assign a route,
write evidence, estimate policy value, or enable promotion. It only shows what
the engine would compute for a request and whether that result would be allowed
to replace a retrieved response.

Examples::

    python source/reasoning_cli.py --example
    python source/reasoning_cli.py --query "Convert 5 km to miles" --steps
    python source/reasoning_cli.py --query "Assuming 5 independent Bernoulli trials with fixed success probability of 1/2, what is the probability of exactly 3 successes?" --steps
    python source/reasoning_cli.py --query "Facts: robin. Rules: robin -> bird; bird -> animal. Query: animal." --steps
    python source/reasoning_cli.py --example --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning_engine import (  # noqa: E402
    REASONING_ENGINE_VERSION,
    frame_problem,
    reasoning_diagnostics,
    render_reasoning_answer,
    solve_problem,
)


# A fixed, disjoint inspection set: solvable requests that must be answered, and
# number-bearing requests that must be left alone.
EXAMPLE_PROBLEMS: Sequence[str] = (
    "What is 15% of 240?",
    "12 is what percent of 48?",
    "What is the percent increase from 80 to 100?",
    "Convert 5 km to miles",
    "Convert 100 celsius to fahrenheit",
    "Solve 3x + 5 = 20",
    "A train travels 120 km in 2 hours. What is its speed?",
    "Alice can paint a room in 4 hours and Bob can paint it in 6 hours. "
    "How long will it take them working together?",
    "What is the next number in the sequence 2, 4, 8, 16?",
    "What is the mean of 2, 4, 6, 8?",
    "What is the gcd of 48 and 18?",
    "Is 91 prime?",
    "How many ways can you choose 3 from 10?",
    "How many days between 2026-01-01 and 2026-03-01?",
    "What is the compound interest on $1000 at 5% for 3 years compounded annually?",
    "Two numbers sum to 30 and differ by 6. What are they?",
    "A shirt costs $80 with 25% off, then 8% tax is added. What is the final price?",
    "What is the area of a rectangle with length 8 cm and width 5 cm?",
    "Given 3 favourable outcomes among 8 equally likely total outcomes, what is the probability?",
    "Assuming 5 independent Bernoulli trials with fixed success probability of 1/2, "
    "what is the probability of exactly 3 successes?",
    "Using Newton's second law, what is the net force on a 5 kg object accelerating at 3 m/s^2?",
    "Using Ohm's law for one resistor, what is the voltage for 2 A through 10 ohms?",
    "Facts: robin. Rules: robin -> bird; bird -> animal. Query: animal.",
)

EXAMPLE_ESTIMATES: Sequence[str] = (
    "Assuming trials are independent with the same success probability, "
    "we observed 7 successes in 10 trials. What is the predicted probability "
    "for the next trial?",
)

EXAMPLE_NON_PROBLEMS: Sequence[str] = (
    "Tell me about the history of Rome",
    "Our revenue went from 80 to 100 last quarter, write a summary email.",
    "Version 3.5 of the library is out, 2 of my services broke.",
    "How do I convert my app to TypeScript?",
    "I have 3 cats and 2 dogs, they are lovely",
)


def _describe(query: str, *, show_steps: bool) -> Dict[str, Any]:
    result = solve_problem(query)
    return {
        "query": query,
        "solved": bool(result["solved"]),
        "override_allowed": bool(result["override_allowed"]),
        "problem_class": str(result["problem_class"]),
        "method": str(result["method"]),
        "verified": bool(result["verification"]["passed"]),
        "independent_check": bool(result["verification"]["independent"]),
        "check": str(result["verification"]["method"]),
        "conflicting": bool(result["consensus"]["conflicting"]),
        "tier": str(result["budget"]["tier"]),
        "solvers_run": int(result["budget"]["solvers_run"]),
        "answer": render_reasoning_answer(result, include_steps=show_steps),
        "diagnostics": reasoning_diagnostics(result),
    }


def _print_row(row: Dict[str, Any]) -> None:
    if row["override_allowed"]:
        badge = "answer"
    elif row["solved"] and row["problem_class"] == "prediction":
        badge = "estimate"
    elif row["solved"]:
        badge = "unverified"
    else:
        badge = "abstain"
    print(f"[{badge:>10}] {row['query']}")
    if row["solved"]:
        detail = (
            f"{row['problem_class']}/{row['method']} "
            f"tier={row['tier']} solvers={row['solvers_run']} "
            f"check={row['check']}{'' if row['independent_check'] else ' (not independent)'}"
        )
        print(f"             {detail}")
        for line in str(row["answer"]).splitlines():
            print(f"             {line}")
    print()


def _run_example(show_steps: bool) -> List[Dict[str, Any]]:
    print(f"Deliberate reasoning engine: {REASONING_ENGINE_VERSION}")
    print("This tool computes and audits only. It has no routing or compute authority.\n")

    print("== requests that should be solved ==\n")
    solvable = [_describe(query, show_steps=show_steps) for query in EXAMPLE_PROBLEMS]
    for row in solvable:
        _print_row(row)

    print("== model-conditional estimates that must not replace an answer ==\n")
    estimates = [_describe(query, show_steps=show_steps) for query in EXAMPLE_ESTIMATES]
    for row in estimates:
        _print_row(row)

    print("== requests that should be left alone ==\n")
    non_problems = [_describe(query, show_steps=False) for query in EXAMPLE_NON_PROBLEMS]
    for row in non_problems:
        _print_row(row)

    answered = sum(1 for row in solvable if row["override_allowed"])
    abstained = sum(1 for row in non_problems if not row["override_allowed"])
    print("== summary ==")
    print(f"solved and verified : {answered}/{len(solvable)}")
    print(f"correctly abstained : {abstained}/{len(non_problems)}")
    return solvable + estimates + non_problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", help="inspect one request")
    parser.add_argument("--example", action="store_true", help="run the built-in inspection set")
    parser.add_argument("--steps", action="store_true", help="include the recorded solution steps")
    parser.add_argument("--frame", action="store_true", help="show the problem frame instead of solving")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args(argv)

    if not args.query and not args.example:
        parser.print_help()
        return 2

    if args.query:
        if args.frame:
            payload: Any = frame_problem(args.query)
        else:
            payload = _describe(args.query, show_steps=args.steps)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        elif args.frame:
            for key, value in sorted(payload.items()):
                print(f"{key}: {value}")
        else:
            _print_row(payload)
        return 0

    rows = _run_example(args.steps)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
