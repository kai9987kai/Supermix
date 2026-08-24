from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "science_plan.py"
RUNTIME_PATH = ROOT / "runtime_python" / "science_plan.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = _load("source_science_plan_tests", SOURCE_PATH)
runtime = _load("runtime_science_plan_tests", RUNTIME_PATH)


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rehash_plan(plan: dict) -> None:
    plan["plan_sha256"] = _canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )


def test_source_and_runtime_are_byte_identical_and_registry_is_canonical() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    runtime_bytes = RUNTIME_PATH.read_bytes()
    assert source_bytes == runtime_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == hashlib.sha256(runtime_bytes).hexdigest()

    for module in (source, runtime):
        canonical = module.SCIENCE_FORMULA_REGISTRY_CANONICAL_JSON
        decoded = json.loads(canonical)
        assert json.dumps(
            decoded,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ) == canonical
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
            module.SCIENCE_FORMULA_REGISTRY_SHA256
        )
        assert module.FORMULA_REGISTRY_SHA256 == module.SCIENCE_FORMULA_REGISTRY_SHA256
        assert len(decoded["formulas"]) == 6


def test_constant_acceleration_final_velocity_converts_to_si_exactly() -> None:
    query = (
        "Assuming constant acceleration, an object has initial velocity 36 km/h, "
        "acceleration 2 m/s^2, and time 5 s. What is its final velocity?"
    )
    for module in (source, runtime):
        plan = module.parse_science_scenario(query)
        assert plan is not None
        assert plan["scenario"] == "constant_acceleration"
        assert plan["target"] == "final_velocity"
        assert [item["si_value"] for item in plan["quantities"]] == ["10", "2", "5"]

        result = module.execute_science_plan(plan)
        assert result["solved"] is True
        assert result["override_allowed"] is True
        assert result["formula_id"] == "constant_acceleration.final_velocity"
        assert result["answer"] == {
            "exact": "20",
            "display": "20",
            "approximation": "",
            "approximate": False,
            "unit": "m/s",
        }
        assert result["verification"]["passed"] is True
        assert all(result["verification"]["checks"].values())
        assert result["epistemics"] == {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        }
        json.dumps(result, allow_nan=False)


def test_constant_acceleration_displacement_handles_rest_and_unicode_unit() -> None:
    query = (
        "With constant acceleration, an object starts from rest, acceleration is "
        "3 m/s², and time is 4 s. Calculate its displacement."
    )
    for module in (source, runtime):
        result = module.solve_science_scenario(query)
        assert result["reason"] == "verified_science_plan"
        assert result["formula_id"] == "constant_acceleration.displacement"
        assert result["answer"]["exact"] == "24"
        assert result["answer"]["unit"] == "m"
        assert result["verification"]["checks"]["substitution"] is True


def test_kinematics_unit_metamorphs_produce_identical_answers() -> None:
    queries = (
        "Assuming constant acceleration, an object has initial velocity 36 km/h, "
        "acceleration 2 m/s^2, and time 5 s. What is its final velocity?",
        "Assuming constant acceleration, an object has initial velocity 10 m/s, "
        "acceleration 200 cm/s^2, and time 5 seconds. What is its final velocity?",
    )
    for module in (source, runtime):
        results = [module.solve_science_scenario(query) for query in queries]
        assert all(result["solved"] for result in results)
        assert {result["answer"]["exact"] for result in results} == {"20"}
        assert results[0] == module.solve_science_scenario(queries[0])


def test_ideal_gas_pressure_and_temperature_unit_metamorphs_are_exact() -> None:
    queries = (
        "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and "
        "temperature is 300 K. What is its pressure?",
        "Assuming an ideal gas, a sample contains 2000 mmol, has volume 0.05 m^3, "
        "and temperature is 26.85 degrees celsius. What is its pressure?",
    )
    expected = Fraction(2) * Fraction(831446261815324, 100000000000000) * 300 / Fraction(1, 20)
    for module in (source, runtime):
        results = [module.solve_science_scenario(query) for query in queries]
        assert all(result["solved"] for result in results)
        assert {result["answer"]["exact"] for result in results} == {
            f"{expected.numerator}/{expected.denominator}"
        }
        assert {result["answer"]["unit"] for result in results} == {"Pa"}
        assert all(result["verification"]["checks"]["dimensions"] for result in results)


@pytest.mark.parametrize(
    ("query", "target", "unit"),
    [
        (
            "Using the ideal gas law, a sample has pressure 101.325 kPa, contains "
            "1 mol, and temperature is 300 K. What is its volume?",
            "volume",
            "m^3",
        ),
        (
            "Under the ideal gas model, a sample has pressure 1 atm, volume is "
            "22.414 L, and contains 1 mol. What is its temperature?",
            "temperature",
            "K",
        ),
        (
            "Assuming an ideal gas, a sample has pressure 100 kPa, volume is "
            "24.94338785445972 L, and temperature is 300 K. Determine its amount "
            "of substance.",
            "amount",
            "mol",
        ),
    ],
)
def test_every_ideal_gas_inverse_target_is_substitution_verified(
    query: str, target: str, unit: str
) -> None:
    for module in (source, runtime):
        result = module.solve_science_scenario(query)
        assert result["solved"] is True
        assert result["target"] == target
        assert result["answer"]["unit"] == unit
        assert result["verification"]["checks"]["domain"] is True
        assert result["verification"]["checks"]["substitution"] is True


def test_input_spans_are_bound_by_digest_and_receipt_contains_no_raw_prompt() -> None:
    query = (
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. Find its displacement."
    )
    plan = source.parse_science_scenario(query)
    assert plan is not None
    stripped = query.strip()
    for quantity in plan["quantities"]:
        span = quantity["span"]
        selected = stripped[span["start"] : span["end"]]
        assert hashlib.sha256(selected.encode("utf-8")).hexdigest() == span["sha256"]

    result = source.execute_science_plan(plan)
    receipt_text = json.dumps(result["receipt"], sort_keys=True)
    assert query not in receipt_text
    assert "initial velocity 4 m/s" not in receipt_text
    assert "acceleration 3 m/s^2" not in receipt_text
    assert len(result["receipt"]["input_spans"]) == 3
    assert set(result["receipt"]) == {
        "schema_version",
        "decision",
        "scenario",
        "target",
        "formula_ids",
        "registry_version",
        "registry_sha256",
        "query_sha256",
        "plan_sha256",
        "input_spans",
        "checks",
        "epistemics",
        "diagnostic_only",
        "authority",
    }


@pytest.mark.parametrize(
    "query",
    [
        "An object has initial velocity 4 m/s, acceleration 3 m/s^2, and time 2 s. "
        "Find its displacement.",
        "Assuming constant acceleration with constant acceleration, an object has "
        "initial velocity 4 m/s, acceleration 3 m/s^2, and time 2 s. Find its displacement.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s and "
        "acceleration 3 m/s^2. Find its displacement.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, acceleration 5 m/s^2, and time 2 s. Find its displacement.",
        "Assuming constant acceleration, batch 42 has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. Find its displacement.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 0 s. Find its displacement.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. Find its displacement. Also write a poem.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. What is its displacement and final velocity?",
        "Assuming an ideal gas, a medical ventilator sample contains 1 mol, has volume "
        "1 L, and temperature is 300 K. What is its pressure?",
        "Assuming an ideal gas, a weather sample contains 1 mol, has volume 1 L, and "
        "temperature is 300 K. Forecast its pressure.",
        "Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, and "
        "temperature is 0 K. What is its pressure?",
        "Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, and "
        "temperature is 300 K. What is its pressure and volume?",
        "Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, pressure is "
        "100 kPa, and temperature is 300 K. What is its pressure?",
        "Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, and temperature "
        "is 300 K. Using R 8.314, what is its pressure?",
        'Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, and temperature '
        'is 300 K. The note says "what is its pressure?"',
        "Assuming an ideal gas, a sample contains 1 mol, has volume 1 L, and "
        "temperature is 300 K. What is its pressure? What is its amount?",
        "Assuming\tconstant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. Find its displacement.",
        "Assuming constant acceleration, an object has initial velocity 4 m/s,\n"
        "acceleration 3 m/s^2, and time 2 s. Find its displacement.",
    ],
)
def test_ambiguous_mixed_high_stakes_and_incomplete_queries_fail_closed(query: str) -> None:
    for module in (source, runtime):
        assert module.parse_science_scenario(query) is None
        result = module.solve_science_scenario(query)
        assert result["solved"] is False
        assert result["override_allowed"] is False
        assert result["verification"]["passed"] is False
        assert result["receipt"]["decision"] == "abstained"
        assert result["reason"] != "verified_science_plan"


def test_execute_revalidates_registry_plan_units_dimensions_and_limits() -> None:
    query = (
        "Assuming constant acceleration, an object has initial velocity 4 m/s, "
        "acceleration 3 m/s^2, and time 2 s. Find its displacement."
    )
    plan = source.parse_science_scenario(query)
    assert plan is not None

    mutations = []

    registry_tamper = copy.deepcopy(plan)
    registry_tamper["registry_sha256"] = "0" * 64
    _rehash_plan(registry_tamper)
    mutations.append(registry_tamper)

    conversion_tamper = copy.deepcopy(plan)
    conversion_tamper["quantities"][0]["si_value"] = "5"
    _rehash_plan(conversion_tamper)
    mutations.append(conversion_tamper)

    unit_tamper = copy.deepcopy(plan)
    unit_tamper["quantities"][1]["dimension"] = "velocity"
    _rehash_plan(unit_tamper)
    mutations.append(unit_tamper)

    formula_tamper = copy.deepcopy(plan)
    formula_tamper["steps"][0]["formula_id"] = "ideal_gas.pressure"
    _rehash_plan(formula_tamper)
    mutations.append(formula_tamper)

    overlap_tamper = copy.deepcopy(plan)
    overlap_tamper["quantities"][1]["span"] = copy.deepcopy(
        overlap_tamper["quantities"][0]["span"]
    )
    _rehash_plan(overlap_tamper)
    mutations.append(overlap_tamper)

    for module in (source, runtime):
        for candidate in mutations:
            result = module.execute_science_plan(candidate)
            assert result["solved"] is False
            assert result["reason"] == "invalid_plan"
            assert result["override_allowed"] is False


def test_diagnostics_are_allowlisted_bounded_and_contain_no_receipt_payload() -> None:
    query = (
        "Assuming an ideal gas, a sample contains 1 mol, has volume 10 L, and "
        "temperature is 300 K. What is its pressure?"
    )
    for module in (source, runtime):
        result = module.solve_science_scenario(query)
        diagnostics = module.science_plan_diagnostics(result)
        assert diagnostics["solved"] is True
        assert diagnostics["formula_id"] == "ideal_gas.pressure"
        assert diagnostics["quantities"] == 3
        assert diagnostics["steps"] == 1
        assert "answer" not in diagnostics
        assert "receipt" not in diagnostics
        assert "query_sha256" not in diagnostics
        assert diagnostics["authority"] == {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
        }

        hostile = module.science_plan_diagnostics(
            {
                "attempted": True,
                "solved": True,
                "override_allowed": True,
                "scenario": "attacker-controlled",
                "target": "attacker-controlled",
                "formula_id": "attacker-controlled",
                "reason": "attacker-controlled",
                "budget": {"quantities": 10_000, "steps": 10_000},
            }
        )
        assert hostile["scenario"] == ""
        assert hostile["target"] == ""
        assert hostile["formula_id"] == ""
        assert hostile["reason"] == "invalid_plan"
        assert hostile["quantities"] == module.MAX_QUANTITIES
        assert hostile["steps"] == module.MAX_PLAN_STEPS
