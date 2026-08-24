from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from fractions import Fraction
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
    (
        "Facts: robin. Rules: robin -> bird; bird -> animal. Query: animal.",
        "logical_entailment",
        "animal follows",
    ),
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
    (
        "A tank starts with 120 liters. Then 25% is removed. "
        "Then 15000 milliliters are added. What is the final volume?",
        "quantity_transition",
        "105 L",
    ),
    (
        "A queue starts with 100 items. Then 20% are removed. "
        "Then 15 items are added. What is the final quantity?",
        "quantity_transition",
        "95 items",
    ),
    ("What is the area of a rectangle with length 8 cm and width 5 cm?", "geometry", "40 cm^2"),
    ("Calculate the perimeter of a rectangle with length 8 cm and width 5 cm.", "geometry", "26 cm"),
    ("Find the area of a triangle with base 10 cm and height 6 cm.", "geometry", "30 cm^2"),
    ("What is the area of a circle with radius 3 cm?", "geometry", "9*pi cm^2"),
    (
        "A right triangle has legs of 3 cm and 4 cm. What is the hypotenuse?",
        "geometry",
        "5 cm",
    ),
    (
        "A right triangle has hypotenuse 13 cm and known leg 5 cm. What is the missing leg?",
        "geometry",
        "12 cm",
    ),
    (
        "Assuming equally likely outcomes, given 3 favourable outcomes and 8 total outcomes, "
        "what is the probability?",
        "probability",
        "3/8",
    ),
    ("What is the probability of heads on a fair coin toss?", "probability", "1/2"),
    ("What is the probability of rolling a 4 on a fair 6-sided die?", "probability", "1/6"),
    (
        "Using Newton's second law, what is the net force on a 5 kg object accelerating at 3 m/s^2?",
        "physics",
        "15 N",
    ),
    ("What is the density of an object with mass 10 g and volume 2 cm3?", "physics", "5000 kg/m^3"),
    ("Calculate the kinetic energy of a 2 kg object moving at 3 m/s.", "physics", "9 J"),
    ("Using Ohm's law for one resistor, what is the voltage for 2 A through 10 ohms?", "physics", "20 V"),
    ("Using Ohm's law for one resistor, what is the current for 12 V across 4 ohms?", "physics", "3 A"),
    ("Using Ohm's law for one resistor, what is the resistance for 12 V and 3 A?", "physics", "4 ohm"),
]

SCIENCE_VERIFIED_CASES = [
    (
        "Assuming constant acceleration, an object has initial velocity 36 km/h, "
        "acceleration 2 m/s^2, and time 5 s. What is its final velocity?",
        "constant_acceleration.final_velocity",
        "20",
        "m/s",
    ),
    (
        "With constant acceleration, an object starts from rest, acceleration is "
        "3 m/s², and time is 4 s. Calculate its displacement.",
        "constant_acceleration.displacement",
        "24",
        "m",
    ),
    (
        "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and "
        "temperature is 300 K. What is its pressure?",
        "ideal_gas.pressure",
        "623584696361493/6250000000",
        "Pa",
    ),
    (
        "Using the ideal gas law, a sample has pressure 101.325 kPa, contains "
        "1 mol, and temperature is 300 K. What is its volume?",
        "ideal_gas.volume",
        "207861565453831/8443750000000000",
        "m^3",
    ),
    (
        "Under the ideal gas model, a sample has pressure 1 atm, volume is "
        "22.414 L, and contains 1 mol. What is its temperature?",
        "ideal_gas.temperature",
        "56777463750000000/207861565453831",
        "K",
    ),
    (
        "Assuming an ideal gas, a sample has pressure 100 kPa, volume is "
        "24.94338785445972 L, and temperature is 300 K. Determine its amount "
        "of substance.",
        "ideal_gas.amount",
        "1",
        "mol",
    ),
]

SCIENCE_FORMULA_TEXT = {
    "constant_acceleration.final_velocity": "v = u + a*t",
    "constant_acceleration.displacement": "s = u*t + (a*t^2)/2",
    "ideal_gas.pressure": "P*V = n*R*T",
    "ideal_gas.volume": "P*V = n*R*T",
    "ideal_gas.temperature": "P*V = n*R*T",
    "ideal_gas.amount": "P*V = n*R*T",
}

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
    "A rectangle is 8 cm by 5 cm; describe it without calculating anything.",
    "What is the area of a rectangle with length 8 cm and width 5 m?",
    "What is the area of a circle with radius 3 cm and radius 4 cm?",
    "What is the probability of rain tomorrow being 70%?",
    "What is the probability of heads in two tosses of a fair coin?",
    "What is the probability of heads when a fair coin is flipped 2 times?",
    "What is the probability of heads when a fair coin is flipped 12 times?",
    "What is the probability of heads when a fair coin is flipped twice?",
    "What is the probability of rolling a 4 or 5 on a fair 6-sided die?",
    "What is the probability of rolling a 4 when a fair 6-sided die is rolled twice?",
    "What is the probability of rolling a 4 when a fair 6-sided die is rolled 2 times?",
    "What is the probability of rolling a 4 when a fair 6-sided die is rolled 12 times?",
    "Given 3 favourable outcomes and 8 total outcomes, what is the probability?",
    (
        "The outcomes are weighted and not equally likely: 3 favourable outcomes and "
        "8 total outcomes. What is the probability?"
    ),
    "Given 9 favourable outcomes and 8 total outcomes, what is the probability?",
    "We observed 7 successes in 10 trials. What is the probability for the next trial?",
    (
        "Assuming trials are independent, we observed 7 successes in 10 trials. "
        "What is the probability for the next trial?"
    ),
    (
        "Assuming trials are not independent with the same success probability, "
        "we observed 7 successes in 10 trials. What is the probability for the next trial?"
    ),
    (
        "Assuming trials are independent. The success probability is constant. "
        "We observed 7 successes in 10 trials. What is the probability for the next trial?"
    ),
    (
        "Assuming trials have the same success probability. The trials are independent. "
        "We observed 7 successes in 10 trials. What is the probability for the next trial?"
    ),
    (
        "Assuming trials are independent with the same success probability, "
        "we observed 7 successes in 10 trials. Is the next trial guaranteed to succeed?"
    ),
    "What is the force on an object with mass 5 and acceleration 3?",
    "What is the force on a 5 kg object accelerating at 3 m/s^2?",
    "Using Newton's second law, what is the applied force on a 5 kg object accelerating at 3 m/s^2 with friction?",
    "What is the density of 10 kg in 0 m3?",
    "What is the density of a layered composite with mass 10 kg and volume 2 m3?",
    "Calculate the kinetic energy of a rolling 2 kg object moving at 3 m/s.",
    "Using Ohm's law, what is the voltage for current 2 A and resistance 10 ohms?",
    "Using Ohm's law for one resistor in a parallel branch, what is the voltage for 2 A through 10 ohms?",
    "What are the force and kinetic energy of a 2 kg object moving at 3 m/s and accelerating at 4 m/s^2?",
    "The documentation says 'find the area'; what does that phrase mean for a 3 cm circle?",
    "Find the word area in this sentence about a rectangle with length 8 cm and width 5 cm.",
    "Do not calculate the area of a rectangle with length 8 cm and width 5 cm.",
    "Find the perimeter. Area of a rectangle with length 8 cm and width 5 cm.",
    "What is the area of a rectangle with length 8 in and width 5 in?",
    (
        "A tank starts with 120 liters or 100 liters. Then 25% is removed. "
        "Then 15 liters are added. What is the final volume?"
    ),
    (
        "A tank starts with 120 liters. Then 25% is added and removed. "
        "Then 15 liters are added. What is the final volume?"
    ),
    (
        "A tank starts with 120 liters. Then 25% is removed. "
        "Then 15 kilograms are added. What is the final volume?"
    ),
    (
        "The initial mass is 120 liters. Then 25% is removed. "
        "Then 15 liters are added. What is the final volume?"
    ),
    (
        "A tank starts with 120 liters. Then 25% is removed. "
        "Then 15 liters are added. What is the final mass?"
    ),
    (
        "A tank starts with 120 liters. Then 10 liters and 5 liters are added. "
        "Then 25% is removed. What is the final volume?"
    ),
    (
        "A tank starts with 10 liters. Then 15 liters are removed. "
        "Then 1 liter is added. What is the final volume?"
    ),
    "A queue starts with 100 items. Then 20 items are removed. What is the final quantity?",
    (
        "A tank starts with 120 liters. Then 25% is removed. Then 15 liters are added. "
        "Do not calculate the final volume."
    ),
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


@pytest.mark.parametrize(
    "query,method,exact,unit",
    SCIENCE_VERIFIED_CASES,
)
def test_science_scenarios_dispatch_through_both_public_engines(
    query: str,
    method: str,
    exact: str,
    unit: str,
) -> None:
    for module in (source, runtime):
        result = module.solve_problem(query)

        assert result["solved"] is True, query
        assert result["override_allowed"] is True, query
        assert result["problem_class"] == "scientific_scenario", query
        assert result["method"] == method, query
        assert result["answer"]["exact"] == exact, query
        assert result["answer"]["unit"] == unit, query
        assert SCIENCE_FORMULA_TEXT[method] in result["text"], query
        assert "verified" in result["text"].lower(), query
        assert result["verification"] == {
            "checked": True,
            "passed": True,
            "method": "registry_dimension_domain_and_substitution",
            "independent": False,
        }
        assert result["epistemics"] == {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        }
        assert result["consensus"]["conflicting"] is False
        assert result["consensus"]["paths"] == 1


def test_science_metadata_is_bounded_prompt_free_and_authority_free() -> None:
    query = SCIENCE_VERIFIED_CASES[0][0]

    for module in (source, runtime):
        result = module.solve_problem(query)
        plan = result["science_plan"]
        receipt = result["science_plan_receipt"]
        diagnostics = module.reasoning_diagnostics(result)

        assert plan["formula_id"] == "constant_acceleration.final_velocity"
        assert plan["quantities"] == 3
        assert plan["steps"] == 1
        assert plan["verification_passed"] is True
        assert plan["verification_independent"] is False
        assert set(plan["authority"].values()) == {False}

        assert receipt["formula_ids"] == ["constant_acceleration.final_velocity"]
        assert [span["symbol"] for span in receipt["input_spans"]] == ["u", "a", "t"]
        assert all(len(span["sha256"]) == 64 for span in receipt["input_spans"])
        assert all(receipt["checks"].values())
        assert receipt["diagnostic_only"] is True
        assert set(receipt["authority"].values()) == {False}
        serialized_receipt = json.dumps(receipt, sort_keys=True)
        assert query not in serialized_receipt
        assert "initial velocity 36 km/h" not in serialized_receipt
        assert "20 m/s" not in serialized_receipt

        assert diagnostics["problem_class"] == "scientific_scenario"
        assert diagnostics["method"] == "constant_acceleration.final_velocity"
        assert diagnostics["model_conditional"] is True
        assert diagnostics["assumptions_explicit"] is True
        serialized_diagnostics = json.dumps(diagnostics, sort_keys=True)
        assert "science_plan" not in diagnostics
        assert "query_sha256" not in serialized_diagnostics
        assert "plan_sha256" not in serialized_diagnostics
        assert query not in serialized_diagnostics

        assert result["authority"] == {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        }


def test_science_exponent_display_remains_bound_to_the_exact_answer() -> None:
    query = (
        "Assuming an ideal gas, a sample contains 0.000000000001 mol, has volume "
        "1000 L, and temperature is 1 K. What is its pressure?"
    )

    for module in (source, runtime):
        result = module.solve_problem(query)

        assert result["method"] == "ideal_gas.pressure"
        assert result["answer"] == {
            "exact": "207861565453831/25000000000000000000000000",
            "display": "8.31446261815e-12",
            "approximation": "8.31446261815e-12",
            "approximate": True,
            "unit": "Pa",
        }
        assert result["verification"]["passed"] is True
        assert result["override_allowed"] is True


def test_multiline_science_is_rejected_at_the_same_strict_boundary() -> None:
    query = (
        "Assuming an ideal gas, a sample contains 1 mol,\n"
        "has volume 1 L, and temperature is 300 K. What is its pressure?"
    )

    for module in (source, runtime):
        science = module._load_science_plan_module()
        direct = science.solve_science_scenario(query)
        result = module.solve_problem(query)

        assert direct["solved"] is False
        assert direct["reason"] == "invalid_query_text"
        assert result["solved"] is False
        assert result["override_allowed"] is False
        assert result["problem_class"] == ""
        assert result["science_plan"] == {}
        assert result["science_plan_receipt"] == {}


@pytest.mark.parametrize(
    "query",
    [
        (
            "An object has initial velocity 4 m/s, acceleration 3 m/s^2, and time "
            "2 s. Find its displacement."
        ),
        (
            "Assuming an ideal gas, a medical ventilator sample contains 1 mol, has "
            "volume 1 L, and temperature is 300 K. What is its pressure?"
        ),
    ],
)
def test_implicit_or_high_stakes_science_never_gains_override(query: str) -> None:
    for module in (source, runtime):
        result = module.solve_problem(query)

        assert result["solved"] is False
        assert result["override_allowed"] is False
        assert result["problem_class"] == ""
        assert result["method"] == ""
        assert result["science_plan"] == {}
        assert result["science_plan_receipt"] == {}


def test_non_science_and_abstained_results_do_not_publish_science_metadata() -> None:
    for module in (source, runtime):
        for query in ("What is 15% of 240?", "Tell me about Rome"):
            result = module.solve_problem(query)

            assert result["science_plan"] == {}
            assert result["science_plan_receipt"] == {}


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
    assert first["schema_version"] == "supermix-reasoning-v2"
    assert first["engine_version"] == "supermix-reasoning-engine-v5"


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


def test_v2_formula_families_are_exact_deterministic_and_algebraically_checked() -> None:
    queries = [
        "What is the area of a rectangle with length 8 cm and width 5 cm?",
        "A right triangle has legs of 3 cm and 4 cm. What is the hypotenuse?",
        "Assuming equally likely outcomes, given 3 favourable outcomes and 8 total outcomes, "
        "what is the probability?",
        "Using Newton's second law, what is the net force on a 5 kg object accelerating at 3 m/s^2?",
        "What is the density of an object with mass 10 g and volume 2 cm3?",
        "Calculate the kinetic energy of a 2 kg object moving at 3 m/s.",
        "Using Ohm's law for one resistor, what is the voltage for 2 A through 10 ohms?",
    ]

    for query in queries:
        first = source.solve_problem(query, tier="deep")
        second = source.solve_problem(query, tier="deep")

        assert first == second, query
        assert first["override_allowed"] is True, query
        assert first["verification"]["passed"] is True, query
        assert first["verification"]["independent"] is False, query
        assert first["consensus"]["conflicting"] is False, query
        assert first["budget"]["solver_limit"] >= first["budget"]["solvers_considered"], query


def test_geometry_and_physics_metamorphic_number_changes() -> None:
    area_8 = source.solve_problem(
        "Problem 99: what is the area of a rectangle with length 8 cm and width 5 cm?"
    )
    area_9 = source.solve_problem(
        "Problem 99: what is the area of a rectangle with length 9 cm and width 5 cm?"
    )
    force_5 = source.solve_problem(
        "Experiment batch 42: using Newton's second law, what is the net force on a "
        "5 kg object accelerating at 3 m/s^2?"
    )
    force_7 = source.solve_problem(
        "Experiment batch 42: using Newton's second law, what is the net force on a "
        "7 kg object accelerating at 3 m/s^2?"
    )

    # Irrelevant identifiers are ignored, while changes to labelled formula
    # inputs produce the exact expected change.
    assert area_8["answer"]["exact"] == "40"
    assert area_9["answer"]["exact"] == "45"
    assert force_5["answer"]["exact"] == "15"
    assert force_7["answer"]["exact"] == "21"
    assert all(item["verification"]["passed"] for item in (area_8, area_9, force_5, force_7))


def test_circle_keeps_pi_symbolic_and_checks_radius_diameter_identity() -> None:
    area = source.solve_problem("What is the area of a circle with radius 3 cm?")
    circumference = source.solve_problem("What is the circumference of a circle with diameter 10 cm?")

    assert "9*pi cm^2" in area["text"]
    assert "approximately 28.274334 cm^2" in area["text"]
    assert "10*pi cm" in circumference["text"]
    assert area["answer"]["exact"] == "9*pi"
    assert area["answer"]["unit"] == "cm^2"
    assert area["verification"] == {
        "checked": True,
        "passed": True,
        "method": "radius_diameter_identity",
        "independent": False,
    }


def test_probability_counts_and_fair_experiments_use_complement_checks() -> None:
    explicit = source.solve_problem(
        "Assuming equally likely outcomes, case 99 has 4 favourable outcomes and "
        "8 total outcomes. What is the probability?"
    )
    coin = source.solve_problem("What is the probability of tails on a fair coin toss?")
    die = source.solve_problem("What is the probability of an even number on a fair six-sided die?")

    assert explicit["answer"]["exact"] == "1/2"
    assert coin["answer"]["exact"] == "1/2"
    assert die["answer"]["exact"] == "1/2"
    for result in (explicit, coin, die):
        assert result["verification"]["method"] == "complement_and_count_reconstruction"
        assert result["verification"]["independent"] is False


@pytest.mark.parametrize(
    "sides,parity,expected",
    [(5, "odd", "3/5"), (5, "even", "2/5"), (7, "odd", "4/7"), (7, "even", "3/7")],
)
def test_odd_sided_fair_die_parity_counts(sides: int, parity: str, expected: str) -> None:
    result = source.solve_problem(
        f"What is the probability of an {parity} number on a fair {sides}-sided die?"
    )

    assert result["override_allowed"] is True
    assert result["answer"]["exact"] == expected
    assert result["verification"]["passed"] is True


@pytest.mark.parametrize(
    (
        "query",
        "expected_model",
        "expected_trials",
        "expected_event",
        "expected_count",
        "expected_outcome",
        "expected_probability",
        "expected_answer",
    ),
    [
        (
            "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of exactly 6 successes?",
            "explicit_probability",
            10,
            "exactly",
            6,
            "success",
            (1, 2),
            "105/512",
        ),
        (
            "Assuming 5 independent Bernoulli trials with a constant success probability of 1/2, "
            "what is the probability of at least 3 successes?",
            "explicit_probability",
            5,
            "at_least",
            3,
            "success",
            (1, 2),
            "1/2",
        ),
        (
            "Under the assumption that 4 independent Bernoulli trials with the same success "
            "probability is 1/4, compute the probability that at most 1 success occurs?",
            "explicit_probability",
            4,
            "at_most",
            1,
            "success",
            (1, 4),
            "189/256",
        ),
        (
            "Assume 6 independent fair coin flips; find the chance of exactly 2 heads?",
            "fair_coin",
            6,
            "exactly",
            2,
            "head",
            (1, 2),
            "15/64",
        ),
    ],
)
def test_finite_bernoulli_strict_grammar_builds_canonical_ir_and_verified_exact_mass(
    query: str,
    expected_model: str,
    expected_trials: int,
    expected_event: str,
    expected_count: int,
    expected_outcome: str,
    expected_probability: tuple[int, int],
    expected_answer: str,
) -> None:
    scenario = source.parse_finite_bernoulli_scenario(query)
    result = source.solve_problem(query)

    assert scenario == {
        "schema": "supermix-finite-bernoulli-scenario-v1",
        "model": expected_model,
        "trials": expected_trials,
        "event": expected_event,
        "count": expected_count,
        "outcome": expected_outcome,
        "probability_numerator": expected_probability[0],
        "probability_denominator": expected_probability[1],
        "full_query_consumed": True,
    }
    assert result["solved"] is True
    assert result["override_allowed"] is True
    assert result["problem_class"] == "probability"
    assert result["method"] == "finite_binomial_event_probability"
    assert result["answer"]["exact"] == expected_answer
    assert result["verification"] == {
        "checked": True,
        "passed": True,
        "method": "bernoulli_convolution_and_mass_check",
        "independent": True,
    }
    assert result["epistemics"] == {
        "model_conditional": True,
        "assumptions_explicit": True,
        "calibration_claimed": False,
    }


def test_finite_bernoulli_event_masses_obey_exact_complement_identities() -> None:
    prefix = (
        "Assuming 8 i.i.d. Bernoulli trials with a fixed success probability of 1/3, "
        "what is the probability of "
    )
    at_least = source.solve_problem(prefix + "at least 3 successes?")
    at_most_below = source.solve_problem(prefix + "at most 2 successes?")
    exactly = source.solve_problem(prefix + "exactly 3 successes?")
    at_most = source.solve_problem(prefix + "at most 3 successes?")

    assert Fraction(at_least["answer"]["exact"]) + Fraction(
        at_most_below["answer"]["exact"]
    ) == 1
    assert Fraction(exactly["answer"]["exact"]) == (
        Fraction(at_most["answer"]["exact"])
        - Fraction(at_most_below["answer"]["exact"])
    )
    assert all(
        result["verification"]["passed"] is True
        and result["verification"]["independent"] is True
        for result in (at_least, at_most_below, exactly, at_most)
    )


@pytest.mark.parametrize(
    "query",
    [
        (
            "Assuming 10 independent Bernoulli trials with a success probability of 1/2, "
            "what is the probability of exactly 6 successes?"
        ),
        (
            "Assuming 10 dependent Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of exactly 6 successes?"
        ),
        (
            "Assuming 10 i.i.d. Bernoulli trials without replacement with a fixed success "
            "probability of 1/2, what is the probability of exactly 6 successes?"
        ),
        (
            "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of not exactly 6 successes?"
        ),
        (
            "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of exactly 11 successes?"
        ),
        (
            "Assuming 201 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of exactly 6 successes?"
        ),
        (
            "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the guaranteed probability of exactly 6 successes?"
        ),
        (
            "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
            "what is the probability of exactly 6 medical treatment successes?"
        ),
        "What is the probability of not rolling a 4 on a fair 6-sided die?",
        "What is the probability of an even number other than 2 on a fair six-sided die?",
        "What is the probability of heads except tails on a fair coin toss?",
    ],
)
def test_probability_complements_and_incomplete_or_unsafe_models_fail_closed(query: str) -> None:
    result = source.solve_problem(query, prompt_profile={"allow_override": True})

    assert source.parse_finite_bernoulli_scenario(query) is None
    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["problem_class"] == ""


@pytest.mark.parametrize(
    "suffix",
    [
        " Actually, use 7 successes.",
        " Correction: use a success probability of 3/4.",
        " No, calculate at most 6 successes instead.",
        " Do not calculate it.",
    ],
)
def test_finite_bernoulli_late_corrections_and_cancellation_cannot_reuse_a_valid_prefix(
    suffix: str,
) -> None:
    valid = (
        "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
        "what is the probability of exactly 6 successes?"
    )
    query = valid + suffix
    result = source.solve_problem(query)

    assert source.parse_finite_bernoulli_scenario(query) is None
    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["reason"] == "ambiguous_or_superseded_request"


def test_finite_bernoulli_overlength_suffix_is_not_truncated_into_override_authority() -> None:
    valid = (
        "Assuming 10 i.i.d. Bernoulli trials with a fixed success probability of 1/2, "
        "what is the probability of exactly 6 successes?"
    )
    query = valid + " " + ("context " * source.MAX_QUERY_CHARS) + "Actually, use 7 successes."

    assert len(query) > source.MAX_QUERY_CHARS
    assert source.parse_finite_bernoulli_scenario(query) is None
    result = source.solve_problem(query)
    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["reason"] == "query_too_long"


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        (
            'Do not use this quoted example: "A rectangle has length 8 m and width 5 m". '
            "Calculate the area of the rectangle.",
            "untrusted_problem_data",
        ),
        (
            "Set aside a mass of 5 kg and acceleration of 3 m/s^2. Using Newton's "
            "second law, calculate the net force.",
            "untrusted_problem_data",
        ),
        (
            "A rectangle has neither length 8 m nor width 5 m. Calculate its area.",
            "untrusted_problem_data",
        ),
        (
            "What is 50% of 8? What is the gcd of 8 and 12?",
            "multiple_calculation_requests",
        ),
        (
            "What is the gcd of 8 and 12? Give me the gcd of 9 and 12.",
            "multiple_calculation_requests",
        ),
        (
            "What is 50% of 8? Translate hello to French.",
            "multiple_calculation_requests",
        ),
        (
            "What is the area of a rectangle with length 8 m and width 5 m, then "
            "describe a real-world use?",
            "multiple_calculation_requests",
        ),
        (
            "Calculate 50% of 8 and draft an email.",
            "mixed_or_unconsumed_request",
        ),
        (
            "What is 50% of 8 and build a table?",
            "mixed_or_unconsumed_request",
        ),
        (
            "Quoted example: > A rectangle has length 8 m and width 5 m. "
            "Calculate the area of the rectangle.",
            "untrusted_problem_data",
        ),
        (
            "\u300cA rectangle has length 8 m and width 5 m\u300d. "
            "Calculate the area of the rectangle.",
            "untrusted_problem_data",
        ),
    ],
)
def test_hard_override_rejects_untrusted_multi_task_and_partially_consumed_prompts(
    query: str,
    reason: str,
) -> None:
    result = source.solve_problem(query)

    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["reason"] == reason


def test_explicit_authoritative_quoted_problem_can_be_consumed_as_one_task() -> None:
    query = (
        'Use this quoted problem as authoritative input: "A rectangle has length 8 m '
        'and width 5 m". Calculate the area of the rectangle.'
    )
    result = source.solve_problem(query)

    assert result["override_allowed"] is True
    assert result["method"] == "rectangle_area"
    assert result["answer"]["exact"] == "40"


@pytest.mark.parametrize(
    "query",
    (
        "Do not rely on 10% of 100. What is 20% of 100?",
        "The fake example is 10% of 100. What is 20% of 100?",
        "Do not rely on converting 5 m to cm. Convert 2 m to cm.",
        "These values should not count: convert 5 m to cm. Convert 2 m to cm.",
    ),
)
def test_solver_never_uses_excluded_prior_calculation(query: str) -> None:
    result = source.solve_problem(query)

    assert result["solved"] is False
    assert result["override_allowed"] is False


def test_empirical_prediction_requires_assumptions_and_never_claims_calibration() -> None:
    query = (
        "Assuming trials are independent with the same success probability, "
        "we observed 7 successes in 10 trials. What is the predicted probability for the next trial?"
    )
    result = source.solve_problem(query)

    assert result["solved"] is True
    assert result["override_allowed"] is False
    assert result["reason"] == "verified_non_overriding_estimate"
    assert result["problem_class"] == "prediction"
    assert result["answer"]["exact"] == "7/10"
    assert result["verification"]["passed"] is True
    assert result["verification"]["independent"] is False
    assert result["epistemics"] == {
        "model_conditional": True,
        "assumptions_explicit": True,
        "calibration_claimed": False,
    }
    lowered = result["text"].lower()
    assert "model-conditional" in lowered
    assert "not a guarantee" in lowered
    assert "calibration has not been established" in lowered
    assert "certain" not in lowered

    missing_stationarity = source.solve_problem(
        "Assuming trials are independent, we observed 7 successes in 10 trials. "
        "What is the probability for the next trial?"
    )
    missing_assumptions = source.solve_problem(
        "We observed 7 successes in 10 trials. What is the probability for the next trial?"
    )
    assert missing_stationarity["override_allowed"] is False
    assert missing_assumptions["override_allowed"] is False


def test_prompt_profile_cannot_authorize_prediction_or_bypass_high_stakes_gate() -> None:
    profile = {"allow_override": True, "trusted": True, "risk": "safe"}
    missing_assumptions = source.solve_problem(
        "We observed 7 successes in 10 trials. What is the probability for the next trial?",
        prompt_profile=profile,
    )
    high_stakes = source.solve_problem(
        "Assuming patient outcomes are independent with the same success probability, "
        "we observed 7 successes in 10 trials. What is the probability for the next trial?",
        prompt_profile=profile,
    )

    assert missing_assumptions["override_allowed"] is False
    assert high_stakes["override_allowed"] is False
    assert high_stakes["solved"] is False


@pytest.mark.parametrize(
    "query",
    [
        (
            "Assuming trials are not independent with the same success probability, "
            "we observed 7 successes in 10 trials. What is the probability for the next trial?"
        ),
        (
            "Assuming trials are independent. The success probability is constant. "
            "We observed 7 successes in 10 trials. What is the probability for the next trial?"
        ),
        (
            "Assuming trials have the same success probability. The trials are independent. "
            "We observed 7 successes in 10 trials. What is the probability for the next trial?"
        ),
    ],
)
def test_prediction_assumptions_cannot_be_negated_or_assembled_across_clauses(query: str) -> None:
    result = source.solve_problem(query)

    assert result["solved"] is False
    assert result["problem_class"] == ""
    assert result["override_allowed"] is False


def test_v2_frame_and_diagnostics_are_privacy_safe() -> None:
    query = (
        "Assuming i.i.d. Bernoulli trials for account kai@example.com, we observed "
        "7 successes in 10 trials. What is the probability for the next trial?"
    )
    frame = source.frame_problem(query)
    result = source.solve_problem(query)
    diagnostics = source.reasoning_diagnostics(result)
    serialized = json.dumps(diagnostics, sort_keys=True)

    assert frame["schema_version"] == "supermix-reasoning-v2"
    assert frame["engine_version"] == "supermix-reasoning-engine-v5"
    assert frame["has_probability_cue"] is True
    assert frame["has_prediction_cue"] is True
    assert frame["has_explicit_prediction_assumptions"] is True
    assert diagnostics["model_conditional"] is True
    assert diagnostics["assumptions_explicit"] is True
    for leaked in ("kai@example.com", "account", "7", "10", "70"):
        assert leaked not in serialized
    assert result["authority"] == {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
    }


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


def test_diagnostics_allowlist_versions_class_method_and_counts() -> None:
    hostile = {
        "schema_version": "attacker-schema",
        "engine_version": "attacker-engine",
        "attempted": True,
        "solved": True,
        "override_allowed": True,
        "problem_class": "<script>secret-class</script>",
        "method": "secret@example.com",
        "verification": {"passed": True, "independent": True},
        "consensus": {"paths": "999 secret", "conflicting": False},
        "budget": {"tier": "attacker-tier", "solvers_run": 999_999},
    }

    diagnostics = source.reasoning_diagnostics(hostile)
    serialized = json.dumps(diagnostics, sort_keys=True)

    assert diagnostics["schema_version"] == "supermix-reasoning-v2"
    assert diagnostics["engine_version"] == "supermix-reasoning-engine-v5"
    assert diagnostics["problem_class"] == ""
    assert diagnostics["method"] == ""
    assert diagnostics["tier"] == ""
    assert diagnostics["paths"] == 0
    assert diagnostics["solvers_run"] == source.MAX_SOLVER_INVOCATIONS
    for leaked in ("attacker", "secret", "example.com", "script"):
        assert leaked not in serialized

    for query, expected_class, _expected_text in VERIFIED_CASES:
        solved = source.solve_problem(query)
        safe = source.reasoning_diagnostics(solved)
        assert safe["problem_class"] == expected_class, query
        assert safe["method"] == solved["method"], query


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


def test_fast_and_deep_tiers_both_exhaust_bounded_consensus() -> None:
    query = "What is 15% of 240?"

    fast = source.solve_problem(query, tier="fast")
    deep = source.solve_problem(query, tier="deep")

    assert fast["budget"]["tier"] == "fast"
    assert deep["budget"]["tier"] == "deep"
    assert fast["budget"]["early_exit"] is False
    assert deep["budget"]["early_exit"] is False
    assert fast["budget"]["all_solvers_exhausted"] is True
    assert deep["budget"]["all_solvers_exhausted"] is True
    assert fast["budget"]["solvers_run"] == fast["budget"]["solver_limit"]
    assert deep["budget"]["solvers_run"] == deep["budget"]["solver_limit"]
    assert fast["consensus"] == deep["consensus"]
    assert fast["answer"]["exact"] == deep["answer"]["exact"]


def test_ordered_quantity_plan_uses_all_evidence_and_reverse_countercheck() -> None:
    query = (
        "A tank starts with 120 liters. Then 25% is removed. "
        "Then 15000 milliliters are added. What is the final volume?"
    )

    result = source.solve_problem(query, tier="deep")

    assert result["override_allowed"] is True
    assert result["problem_class"] == "quantity_transition"
    assert result["method"] == "ordered_quantity_transitions"
    assert result["answer"] == {
        "exact": "105",
        "display": "105",
        "approximation": "",
        "approximate": False,
        "unit": "L",
    }
    assert result["verification"] == {
        "checked": True,
        "passed": True,
        "method": "reverse_state_reconstruction",
        "independent": True,
    }
    assert result["steps"] == [
        "Start with 120 L from the explicit initial-state evidence.",
        "Step 1: 25% decrease -> 90 L.",
        "Step 2: add 15000 mL -> 105 L.",
        "Countercheck: reverse every transition and recover the exact initial state.",
    ]
    assert result["budget"]["tier"] == "deep"


def test_ordered_quantity_plan_generalizes_across_values_and_unit_forms() -> None:
    milliliters = source.solve_problem(
        "A tank starts with 160 liters. Then 25% is removed. "
        "Then 15000 milliliters are added. What is the final volume?"
    )
    liters = source.solve_problem(
        "A tank starts with 160 liters. Then 25% is removed. "
        "Then 15 liters are added. What is the final volume?"
    )
    inventory = source.solve_problem(
        "A queue starts with 80 items. Then 25% are removed. "
        "Then 10 items are added. What is the final quantity?"
    )

    assert milliliters["answer"]["exact"] == "135"
    assert liters["answer"]["exact"] == "135"
    assert inventory["answer"]["exact"] == "70"
    assert all(
        result["verification"]["method"] == "reverse_state_reconstruction"
        for result in (milliliters, liters, inventory)
    )


@pytest.mark.parametrize(
    "query",
    [
        (
            "A tank starts with 120 liters or 100 liters. Then 25% is removed. "
            "Then 15 liters are added. What is the final volume?"
        ),
        (
            "A tank starts with 120 liters. Then 25% is added and removed. "
            "Then 15 liters are added. What is the final volume?"
        ),
        (
            "A tank starts with 120 liters. Then 25% is removed. "
            "Then 15 kilograms are added. What is the final volume?"
        ),
        (
            "The initial mass is 120 liters. Then 25% is removed. "
            "Then 15 liters are added. What is the final volume?"
        ),
        (
            "A tank starts with 120 liters. Then 25% is removed. "
            "Then 15 liters are added. What is the final mass?"
        ),
        (
            "A tank starts with 120 liters. Then 10 liters and 5 liters are added. "
            "Then 25% is removed. What is the final volume?"
        ),
        (
            "A tank starts with 10 liters. Then 15 liters are removed. "
            "Then 1 liter is added. What is the final volume?"
        ),
        "A queue starts with 100 items. Then 20 items are removed. What is the final quantity?",
    ],
)
def test_ordered_quantity_plan_fails_closed_on_incomplete_or_conflicting_evidence(query: str) -> None:
    result = source.solve_problem(query, tier="deep", prompt_profile={"allow_override": True})

    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["problem_class"] == ""
    assert result["reason"] == "no_applicable_solver"


def test_bounded_horn_entailment_composes_conjunction_and_multiple_hops() -> None:
    query = (
        "Facts: warm, raining. Rules: warm & raining -> humid; humid -> slippery. "
        "Query: slippery."
    )

    result = source.solve_problem(query, tier="auto")

    assert result["override_allowed"] is True
    assert result["problem_class"] == "logical_entailment"
    assert result["method"] == "bounded_horn_entailment"
    assert result["reason"] == "verified_solution"
    assert result["answer"] == {
        "exact": "entailed:slippery",
        "display": "Entailed: slippery follows from the supplied facts and rules.",
        "approximation": "",
        "approximate": False,
        "unit": "",
    }
    assert result["verification"] == {
        "checked": True,
        "passed": True,
        "method": "finite_model_entailment_check",
        "independent": True,
    }
    assert result["epistemics"] == {
        "model_conditional": True,
        "assumptions_explicit": True,
        "calibration_claimed": False,
    }
    assert result["budget"]["tier"] == "deep"
    assert result["steps"] == [
        "Start from the supplied facts: raining, warm.",
        "Apply raining and warm -> humid; infer humid.",
        "Apply humid -> slippery; infer slippery.",
        "Countercheck: all 1 satisfying finite models make slippery true.",
    ]


def test_bounded_horn_not_entailed_is_not_a_real_world_falsehood() -> None:
    result = source.solve_problem(
        "Facts: robin. Rules: robin -> bird. Query: aquatic.",
        tier="deep",
    )

    assert result["override_allowed"] is True
    assert result["answer"]["exact"] == "not_entailed:aquatic"
    assert result["text"] == (
        "Not entailed: aquatic does not follow from the supplied facts and rules."
    )
    assert "aquatic is false" not in result["text"].lower()
    assert result["verification"]["passed"] is True
    assert result["verification"]["independent"] is True
    assert "at least one keeps aquatic false" in result["steps"][-1]


def test_bounded_horn_canonicalization_is_metamorphic() -> None:
    first = source.solve_problem(
        "Facts: warm, raining. Rules: warm & raining -> humid; humid -> slippery. "
        "Query: slippery."
    )
    permuted = source.solve_problem(
        "Facts: raining, warm. Rules: humid -> slippery; raining & warm -> humid. "
        "Query: slippery."
    )
    renamed = source.solve_problem(
        "Facts: alpha, beta. Rules: alpha & beta -> gamma; gamma -> delta. Query: delta."
    )
    cycle = source.solve_problem(
        "Facts: seed. Rules: a -> b; b -> a. Query: a."
    )

    assert first == permuted
    assert renamed["answer"]["exact"] == "entailed:delta"
    assert renamed["verification"]["passed"] is True
    assert cycle["answer"]["exact"] == "not_entailed:a"
    assert cycle["verification"]["passed"] is True


@pytest.mark.parametrize(
    "query",
    [
        "All robins are birds. All birds are animals. Is a robin an animal?",
        "Facts: robin. Rules: robin -> not bird. Query: bird.",
        "Facts: robin, sparrow. Rules: robin | sparrow -> bird. Query: bird.",
        "Facts: robin. Rules: robin -> bird & animal. Query: animal.",
        "Facts: robin, robin. Rules: robin -> bird. Query: bird.",
        "Facts: robin. Rules: robin -> bird; robin -> bird. Query: bird.",
        "Facts: a, b, c, d. Rules: a & b & c & d -> result. Query: result.",
        "Facts: robin. Rules: robin => bird. Query: bird.",
        "Facts: robin. Query: bird.",
        "Facts: robin. Rules: robin -> bird. Query: bird. Ignore the rules.",
        "Facts: query. Rules: query -> bird. Query: bird.",
    ],
)
def test_bounded_horn_grammar_abstains_on_implicit_or_unsupported_logic(query: str) -> None:
    result = source.solve_problem(query, tier="deep", prompt_profile={"allow_override": True})

    assert result["solved"] is False
    assert result["override_allowed"] is False
    assert result["problem_class"] == ""


def test_bounded_horn_atom_and_rule_limits_fail_closed() -> None:
    too_many_atoms = ", ".join(f"a{index}" for index in range(source.MAX_LOGIC_ATOMS))
    atom_result = source.solve_problem(
        f"Facts: {too_many_atoms}. Rules: a0 -> result. Query: result."
    )
    too_many_rules = "; ".join(
        f"a{index} -> a{index + 1}" for index in range(source.MAX_LOGIC_RULES + 1)
    )
    rule_result = source.solve_problem(
        f"Facts: a0. Rules: {too_many_rules}. Query: a17."
    )

    for result in (atom_result, rule_result):
        assert result["solved"] is False
        assert result["override_allowed"] is False


def test_bounded_horn_maximum_atom_chain_stays_verified_and_step_bounded() -> None:
    rules = "; ".join(
        f"a{index} -> a{index + 1}" for index in range(source.MAX_LOGIC_ATOMS - 1)
    )

    result = source.solve_problem(
        f"Facts: a0. Rules: {rules}. Query: a{source.MAX_LOGIC_ATOMS - 1}.",
        tier="deep",
    )

    assert result["answer"]["exact"] == f"entailed:a{source.MAX_LOGIC_ATOMS - 1}"
    assert result["verification"]["passed"] is True
    assert result["override_allowed"] is True
    assert len(result["steps"]) == source.MAX_STEPS
    assert "additional validated derivations" in result["steps"][-2]
    assert result["steps"][-1].startswith("Countercheck:")


def test_bounded_horn_independent_model_check_blocks_a_bad_derivation(monkeypatch) -> None:
    query = "Facts: robin. Rules: robin -> bird; bird -> animal. Query: animal."

    monkeypatch.setattr(
        source,
        "_horn_forward_closure",
        lambda facts, _rules: (frozenset(facts), {}),
    )
    result = source.solve_problem(query, tier="deep")

    assert result["solved"] is True
    assert result["verification"]["passed"] is False
    assert result["override_allowed"] is False
    assert result["reason"] == "unverified_solution"


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
    for query, _method, _exact, _unit in SCIENCE_VERIFIED_CASES:
        assert source.solve_problem(query) == runtime.solve_problem(query), query
    assert source.frame_problem("What is 15% of 240?") == runtime.frame_problem("What is 15% of 240?")
