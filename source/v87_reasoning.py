"""Bounded problem identities and process checks for the v87 training arms.

Only the existing average and percentage-then-operation grammars are accepted.
No generated text is executed and unsupported working is never a process pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction
from typing import Any

NUMBER = r"-?\d+(?:\.\d+)?"
FRACTIONS = {10: ("one tenth", 10), 20: ("one fifth", 5),
             25: ("one quarter", 4), 50: ("one half", 2)}


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def parse_problem(prompt: str, task: str) -> dict:
    if task == "average":
        match = re.fullmatch(r"Find the average \(mean\) of these numbers: (\d+(?:, \d+){3,5})", prompt)
        if match:
            values = [int(v) for v in match[1].split(", ")]
            if all(5 <= v <= 99 for v in values):
                return {"task": task, "values": values}
    elif task == "two_step":
        match = re.fullmatch(r"What is (10|20|25|50)% of (\d+), then (add|subtract) (\d+)\?", prompt)
        if match:
            pct, base, op, delta = match.groups()
            pct, base, delta = int(pct), int(base), int(delta)
            if 40 <= base < 900 and 5 <= delta <= 60 and base * pct % 100 == 0:
                return {"task": task, "pct": pct, "base": base, "op": op, "delta": delta}
    raise ValueError(f"unsupported {task} training prompt: {prompt[:160]!r}")


def group_id(case: dict) -> str:
    # All average permutations and both two-step operation contrasts share a
    # group, preventing a reworded or sign-flipped exam item from entering train.
    identity = ({"task": "average", "values": sorted(case["values"])}
                if case["task"] == "average" else
                {k: case[k] for k in ("task", "pct", "base", "delta")})
    return digest_json(identity)


def expected_answer(case: dict) -> Fraction:
    if case["task"] == "average":
        return Fraction(sum(case["values"]), len(case["values"]))
    first = Fraction(case["pct"] * case["base"], 100)
    return first + case["delta"] if case["op"] == "add" else first - case["delta"]


def canonical_prompt(case: dict) -> str:
    if case["task"] == "average":
        return "Find the average (mean) of these numbers: " + ", ".join(map(str, case["values"]))
    return f"What is {case['pct']}% of {case['base']}, then {case['op']} {case['delta']}?"


def render_working(case: dict) -> str:
    if case["task"] == "average":
        values = case["values"]
        running, steps = values[0], []
        for value in values[1:]:
            steps.append(f"{running} + {value} = {running + value}")
            running += value
        return (", ".join(steps) + f", total {running}, divide by {len(values)}, "
                f"total {round(float(expected_answer(case)), 6)}")
    fraction, divisor = FRACTIONS[case["pct"]]
    first = case["base"] // divisor
    operator = "+" if case["op"] == "add" else "-"
    answer = int(expected_answer(case))
    return (f"{case['pct']} percent is {fraction}, {case['base']} / {divisor} = {first}, "
            f"then {first} {operator} {case['delta']} = {answer}, total {answer}")


def answer_correct(case: dict, reply: str) -> bool:
    match = re.search(rf"\btotal ({NUMBER})[.]?\s*$", reply)
    if not match:
        return False
    expected = expected_answer(case)
    return abs(Fraction(match[1]) - expected) <= max(Fraction(1, 10**6), abs(expected) / 10**6)


def verify_working(case: dict, reply: str) -> dict:
    """Check every declared step and its link to the original operands.

    Covers the legacy v86 working and the compact v87 alternatives. The full
    reply must match a supported trace, so correct fragments cannot hide an
    unsupported extra assertion. This verifies text, not internal computation.
    """
    text = reply.strip().removesuffix(".")
    checks: list[bool] = []
    trace_format = None
    if case["task"] == "average":
        match = re.fullmatch(rf"(.+), total (\d+), divide by (\d+), total ({NUMBER})", text)
        if match:
            prefix, stated_sum, count, answer = match.groups()
            values = case["values"]
            if prefix.startswith("sum: ") and re.fullmatch(r"\d+(?: then \d+)*", prefix[5:]):
                totals = [int(v) for v in prefix[5:].split(" then ")]
                if len(totals) == len(values):
                    trace_format = "legacy_running_totals"
                    checks.extend(value == sum(values[:i+1]) for i, value in enumerate(totals))
            else:
                terms = prefix.split(", ")
                parsed = [re.fullmatch(r"(\d+) \+ (\d+) = (\d+)", term) for term in terms]
                if len(terms) == len(values)-1 and all(parsed):
                    trace_format = "binary_additions"
                    previous = values[0]
                    for i, step in enumerate(parsed):
                        left, right, result = map(int, step.groups())
                        checks.append(left == previous and right == values[i+1] and left+right == result)
                        previous = result
            if trace_format:
                checks.extend((int(stated_sum) == sum(values), int(count) == len(values),
                               abs(Fraction(answer) - Fraction(int(stated_sum), int(count) or 1)) <= Fraction(1, 10**6)))
    else:
        match = re.fullmatch(rf"(.+), then ({NUMBER}) ([+-]) ({NUMBER}) = ({NUMBER}), total ({NUMBER})", text)
        if match:
            prefix, left, op, delta, result, answer = match.groups()
            first = Fraction(case["base"] * case["pct"], 100)
            legacy = re.fullmatch(rf"1 percent of (\d+) = ({NUMBER}), times (\d+) = ({NUMBER})", prefix)
            division = re.fullmatch(rf"(\d+) percent is (one tenth|one fifth|one quarter|one half), (\d+) / (\d+) = ({NUMBER})", prefix)
            if legacy:
                base, hundredth, pct, stated_first = legacy.groups()
                trace_format = "legacy_percent"
                checks.extend((int(base) == case["base"] and Fraction(hundredth) == Fraction(int(base), 100),
                               int(pct) == case["pct"] and Fraction(hundredth)*int(pct) == Fraction(stated_first)))
            elif division:
                pct, name, base, divisor, stated_first = division.groups()
                trace_format = "fraction_division"
                checks.extend((int(pct) == case["pct"] and (name, int(divisor)) == FRACTIONS[case["pct"]],
                               int(base) == case["base"] and int(divisor) > 0 and
                               Fraction(int(base), int(divisor)) == Fraction(stated_first)))
            if trace_format:
                wanted_op = "+" if case["op"] == "add" else "-"
                calculated = Fraction(left) + Fraction(delta) if op == "+" else Fraction(left) - Fraction(delta)
                checks.extend((Fraction(stated_first) == first,
                               Fraction(left) == Fraction(stated_first) and op == wanted_op and int(Fraction(delta)) == case["delta"] and Fraction(delta).denominator == 1,
                               calculated == Fraction(result), Fraction(result) == Fraction(answer)))
    final_ok = answer_correct(case, text)
    return {"supported": trace_format is not None, "format": trace_format,
            "checked_steps": len(checks), "correct_steps": sum(checks),
            "first_error": next((i for i, ok in enumerate(checks) if not ok), None),
            "answer_correct": final_ok,
            "process_correct": bool(checks) and all(checks) and final_ok}
