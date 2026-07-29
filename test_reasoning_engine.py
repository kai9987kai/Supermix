from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "reasoning_engine.py"
RUNTIME_PATH = ROOT / "runtime_python" / "reasoning_engine.py"
SOURCE_GROUNDING_PATH = ROOT / "source" / "grounding_runtime.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # The engine declares a dataclass, which resolves its own module at class
    # creation time, so it has to be registered before the module body runs.
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


source = _load_module("source_reasoning_engine_tests", SOURCE_PATH)
runtime = _load_module("runtime_reasoning_engine_tests", RUNTIME_PATH)


# Every case below must be solved, must pass its own verification, and must be
# eligible to override a retrieved response.
VERIFIED_CASES = [
    ("What is 15% of 240?", "percent", "36"),
    ("12 is what percent of 48?", "percent", "25%"),
    ("What percent of 200 is 35?", "percent", "17.5%"),
    ("9 is 30% of what number?", "percent", "30"),
    ("What is the percent increase from 80 to 100?", "percent_change", "25%"),
    ("Convert 5 km to miles", "unit_conversion", "3.106856"),
    ("How many meters are in 3.5 km?", "unit_conversion", "3500"),
    ("Convert 100 celsius to fahrenheit", "unit_conversion", "212"),
    ("Convert 32 fahrenheit to celsius", "unit_conversion", "0"),
    ("Convert 2 GiB to MiB", "unit_conversion", "2048"),
    ("Convert 1 hour to minutes", "unit_conversion", "60"),
    ("Solve 3x + 5 = 20", "linear_equation", "x = 5"),
    ("solve for x: 2x - 7 = 3x + 1", "linear_equation", "x = -8"),
    ("x/2 + 3 = 7", "linear_equation", "x = 8"),
    ("A train travels 120 km in 2 hours. What is its speed?", "rate", "60"),
    ("If a car drives at 60 mph for 3 hours, how far does it go?", "rate", "180"),
    (
        "Alice can paint a room in 4 hours and Bob can paint it in 6 hours. "
        "How long will it take them working together?",
        "work_rate",
        "2.4",
    ),
    (
        "If 3 apples cost 6 dollars, at the same rate how much do 7 apples cost?",
        "proportion",
        "14",
    ),
    ("What is the next number in the sequence 2, 4, 8, 16?", "sequence", "32"),
    ("What comes next in the pattern 3, 7, 11, 15?", "sequence", "19"),
    ("Next term in the sequence 1, 1, 2, 3, 5, 8?", "sequence", "13"),
    ("What is the mean of 2, 4, 6, 8?", "statistics", "5"),
    ("Find the median of 3, 1, 4, 1, 5", "statistics", "3"),
    ("What is the sum of 5 and 7?", "statistics", "12"),
    ("What is the gcd of 48 and 18?", "number_theory", "6"),
    ("What is the lcm of 4 and 6?", "number_theory", "12"),
    ("Is 97 prime?", "number_theory", "is prime"),
    ("Is 91 prime?", "number_theory", "7 x 13"),
    ("What are the prime factors of 84?", "number_theory", "2 x 2 x 3 x 7"),
    ("How many ways can you choose 3 from 10?", "combinatorics", "120"),
    ("What is 5!?", "combinatorics", "120"),
    ("How many days between 2026-01-01 and 2026-03-01?", "date", "59"),
    ("What date is 45 days after 2026-01-01?", "date", "2026-02-15"),
    ("What is the simple interest on $1000 at 5% for 3 years?", "interest", "150"),
    (
        "What is the compound interest on $1000 at 5% for 3 years compounded annually?",
        "interest",
        "157.625",
    ),
    ("Two numbers sum to 30 and differ by 6. What are they?", "sum_difference", "18 and 12"),
    (
        "A shirt costs $80 with 25% off, then 8% tax is added. What is the final price?",
        "percent_chain",
        "64.8",
    ),
]

# Requests that contain numbers but are not solvable problems. None of these may
# produce an answer that is allowed to replace a response.
MUST_ABSTAIN = [
    "Tell me about the history of Rome",
    "Our revenue went from 80 to 100 last quarter, write a summary email.",
    "The meeting is on 2026-03-01 and the deadline is 2026-04-15.",
    "Version 3.5 of the library is out, 2 of my services broke.",
    "My SSN is 123-45-6789",
    "Card 4111 1111 1111 1111",
    "Call me at 555-123-4567",
    "I need to sum up my feelings about this project",
    "Total nonsense, 5 and 7 have nothing to do with it, ignore me",
    "The average person walks 5 km a day",
    "How do I convert my app to TypeScript?",
    "Convert the database to Postgres in 3 steps",
    "3 of 10 tests failed",
    "I have 3 cats and 2 dogs, they are lovely",
    "solve 2x + 3 = 2x + 4",
    "solve 2x + 3 = 2x + 3",
    "What is the gcd of 0 and 5?",
]


@pytest.mark.parametrize("query,problem_class,expected", VERIFIED_CASES)
def test_verified_cases_solve_and_self_check(query: str, problem_class: str, expected: str) -> None:
    result = source.solve_problem(query)

    assert result["solved"] is True, query
    assert result["problem_class"] == problem_class, query
    assert result["verification"]["passed"] is True, query
    assert result["verification"]["checked"] is True, query
    assert result["consensus"]["conflicting"] is False, query
    assert result["override_allowed"] is True, query
    assert expected in result["text"], f"{query}: {result['text']}"


@pytest.mark.parametrize("query", MUST_ABSTAIN)
def test_non_problems_never_earn_override_authority(query: str) -> None:
    result = source.solve_problem(query)

    assert result["override_allowed"] is False, f"{query}: {result.get('text')}"


def test_results_are_deterministic_and_json_safe() -> None:
    query = "A shirt costs $80 with 25% off, then 8% tax is added. What is the final price?"

    first = source.solve_problem(query)
    second = source.solve_problem(query)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == "supermix-reasoning-v1"
    assert first["engine_version"] == "supermix-reasoning-engine-v1"


def test_engine_claims_no_routing_or_compute_authority() -> None:
    result = source.solve_problem("What is 15% of 240?")
    frame = source.frame_problem("What is 15% of 240?")

    expected = {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
    }
    assert result["authority"] == expected
    assert frame["authority"] == expected

    serialized = json.dumps(result, sort_keys=True)
    assert "reasoning_cycles" not in serialized
    assert "agent_mode" not in serialized
    assert "response_strategy" not in serialized


def test_diagnostics_omit_prompt_text_and_answer() -> None:
    query = "What is 15% of 240 for account kai@example.com?"
    result = source.solve_problem(query)
    diagnostics = source.reasoning_diagnostics(result)
    serialized = json.dumps(diagnostics, sort_keys=True)

    for leaked in ("240", "15", "36", "kai@example.com", "account"):
        assert leaked not in serialized, leaked
    assert diagnostics["problem_class"] == "percent"
    assert diagnostics["verified"] is True
    assert source.reasoning_diagnostics(None)["attempted"] is False


def test_exact_arithmetic_avoids_float_drift() -> None:
    result = source.solve_problem("What is 10% of 0.1?")

    assert result["solved"] is True
    # 0.1 * 0.1 is 0.01 exactly here, not 0.010000000000000002.
    assert result["answer"]["exact"] == "1/100"
    assert result["answer"]["display"] == "0.01"


def test_unit_conversion_reports_exact_and_rounded_forms() -> None:
    result = source.solve_problem("Convert 5 km to miles")

    assert result["answer"]["exact"] == "78125/25146"
    assert result["answer"]["approximate"] is True
    assert result["answer"]["unit"] == "miles"
    assert result["verification"]["method"] == "round_trip_and_direction"
    assert result["verification"]["independent"] is True


def test_fast_tier_exits_early_and_deep_tier_explores_more() -> None:
    query = "What is 15% of 240?"

    fast = source.solve_problem(query, tier="fast")
    deep = source.solve_problem(query, tier="deep")

    assert fast["budget"]["tier"] == "fast"
    assert deep["budget"]["tier"] == "deep"
    assert fast["budget"]["early_exit"] is True
    assert deep["budget"]["solvers_run"] > fast["budget"]["solvers_run"]
    # A larger budget must not change a verified answer.
    assert fast["answer"]["exact"] == deep["answer"]["exact"]


def test_complexity_drives_the_recommended_tier() -> None:
    simple = source.frame_problem("What is 15% of 240?")
    complex_request = source.frame_problem(
        "A shirt costs $80 with 25% off, then 8% tax is added, "
        "and then a 5% loyalty discount is applied. What is the final price?"
    )

    assert simple["complexity"] < complex_request["complexity"]
    assert complex_request["recommended_tier"] == "deep"


def test_steps_are_available_only_when_requested() -> None:
    result = source.solve_problem("Convert 5 km to miles")

    plain = source.render_reasoning_answer(result)
    detailed = source.render_reasoning_answer(result, include_steps=True)

    assert "How I got there" not in plain
    assert "How I got there" in detailed
    assert len(result["steps"]) <= 10
    assert source.render_reasoning_answer({"solved": False}) == ""


def test_oversized_and_hostile_inputs_stay_bounded() -> None:
    hostile = [
        "What is " + "9" * 400 + "% of " + "9" * 400 + "?",
        "solve " + "+".join(["2x"] * 200) + " = 10",
        "next in the sequence " + ", ".join(str(index) for index in range(200)),
        "the mean of " + ", ".join(str(index) for index in range(500)),
        "What is 999999999999999999999999! ?",
        "x" * 5000,
        "",
    ]

    for query in hostile:
        started = time.perf_counter()
        result = source.solve_problem(query)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assert result["override_allowed"] is False, query[:40]
        assert elapsed_ms < 250.0, f"{query[:40]}: {elapsed_ms:.1f}ms"


def test_unverified_or_conflicting_answers_cannot_override() -> None:
    # Direct contract check: override authority is exactly "verified and not
    # conflicting", so neither flag alone is enough.
    for query, _class, _expected in VERIFIED_CASES:
        result = source.solve_problem(query)
        assert result["override_allowed"] == (
            bool(result["verification"]["passed"]) and not bool(result["consensus"]["conflicting"])
        )


def test_grounding_runtime_only_overrides_on_verified_solutions() -> None:
    grounding = _load_module("source_grounding_for_reasoning_tests", SOURCE_GROUNDING_PATH)

    solved = grounding.finalize_grounded_response("a stale retrieved reply", "What is 15% of 240?")
    untouched = grounding.finalize_grounded_response("Rome was a city.", "Tell me about Rome")
    arithmetic_wins = grounding.finalize_grounded_response("x", "What is (1.25 + 2.75) * 3?")

    assert solved["reason"] == "verified_reasoning_solution"
    assert solved["text"] == "15% of 240 is 36."
    assert solved["reasoning"]["override_allowed"] is True

    assert untouched["reason"] == "audit_only"
    assert untouched["changed"] is False
    assert untouched["reasoning"]["override_allowed"] is False

    # An explicit arithmetic expression keeps its existing dedicated path.
    assert arithmetic_wins["reason"] == "explicit_arithmetic_exact"


def test_grounding_runtime_shows_work_when_asked() -> None:
    grounding = _load_module("source_grounding_for_reasoning_steps_tests", SOURCE_GROUNDING_PATH)

    result = grounding.finalize_grounded_response(
        "stale", "Convert 5 km to miles. Show your work."
    )

    assert result["reason"] == "verified_reasoning_solution"
    assert "How I got there" in result["text"]


def test_inspection_cli_reports_every_built_in_case(capsys) -> None:
    cli = _load_module("source_reasoning_cli_tests", ROOT / "source" / "reasoning_cli.py")

    assert cli.main(["--example"]) == 0
    output = capsys.readouterr().out

    solvable = len(cli.EXAMPLE_PROBLEMS)
    non_problems = len(cli.EXAMPLE_NON_PROBLEMS)
    assert f"solved and verified : {solvable}/{solvable}" in output
    assert f"correctly abstained : {non_problems}/{non_problems}" in output

    assert cli.main(["--query", "Convert 5 km to miles", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["override_allowed"] is True
    assert payload["problem_class"] == "unit_conversion"

    assert cli.main([]) == 2


def test_source_and_runtime_engines_are_exact_mirrors() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    runtime_bytes = RUNTIME_PATH.read_bytes()
    assert source_bytes == runtime_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == hashlib.sha256(runtime_bytes).hexdigest()

    for query, _class, _expected in VERIFIED_CASES:
        assert source.solve_problem(query) == runtime.solve_problem(query), query
    assert source.frame_problem("What is 15% of 240?") == runtime.frame_problem("What is 15% of 240?")
