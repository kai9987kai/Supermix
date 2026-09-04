"""Adversarial tests for the bounded second arithmetic implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))

import nexus_independent_checker as checker  # noqa: E402


def test_exact_integer_expression_passes_with_independent_witness():
    result = checker.check_arithmetic_certificate(
        query="What is 2 + 3 * 4?",
        display_answer="14",
        problem_class="arithmetic",
    )

    assert result["status"] == "passed"
    assert result["algorithmically_independent"] is True
    assert result["checker_id"] == checker.CHECKER_ID
    assert result["expected_display"] == "14"


def test_fraction_expression_is_checked_exactly():
    result = checker.check_arithmetic_certificate(
        query="evaluate 5 / 6",
        display_answer="5/6",
        problem_class="arithmetic",
    )

    assert result["status"] == "passed"
    assert result["expected_display"] == "5/6"
    assert result["operations"] == 1


def test_mismatched_display_fails_closed():
    result = checker.check_arithmetic_certificate(
        query="calculate 2 + 3",
        display_answer="6",
        problem_class="arithmetic",
    )

    assert result["status"] == "failed"
    assert result["algorithmically_independent"] is False
    assert result["reason"] == "display_mismatch"


def test_unsupported_prose_and_confusable_numeric_text_are_not_admitted():
    prose = checker.check_arithmetic_certificate(
        query="What is two plus three?",
        display_answer="5",
        problem_class="arithmetic",
    )
    confusable = checker.check_arithmetic_certificate(
        query="What is 2 + 3?",
        display_answer="５",
        problem_class="arithmetic",
    )

    assert prose["status"] == "failed"
    assert prose["algorithmically_independent"] is False
    assert confusable["status"] == "failed"
    assert confusable["algorithmically_independent"] is False


def test_non_arithmetic_scope_is_explicitly_not_applicable():
    result = checker.check_arithmetic_certificate(
        query="What is the area of a rectangle with length 8 cm and width 5 cm?",
        display_answer="40",
        problem_class="geometry",
    )

    assert result == {
        "schema_version": checker.CHECKER_SCHEMA_VERSION,
        "checker_id": checker.CHECKER_ID,
        "status": "not_applicable",
        "algorithmically_independent": False,
        "reason": "non_arithmetic_claim_scope",
    }


@pytest.mark.parametrize(
    ("query", "method", "display", "unit"),
    [
        (
            "Assuming constant acceleration, an object has initial velocity 36 km/h, acceleration 2 m/s^2, and time 5 s. What is its final velocity?",
            "constant_acceleration.final_velocity",
            "20",
            "m/s",
        ),
        (
            "With constant acceleration, an object starts from rest, acceleration is 3 m/s², and time is 4 s. Calculate its displacement.",
            "constant_acceleration.displacement",
            "24",
            "m",
        ),
        (
            "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and temperature is 300 K. What is its pressure?",
            "ideal_gas.pressure",
            "99773.55141783888",
            "Pa",
        ),
        (
            "Using the ideal gas law, a sample has pressure 101.325 kPa, contains 1 mol, and temperature is 300 K. What is its volume?",
            "ideal_gas.volume",
            "0.0246172098243",
            "m^3",
        ),
        (
            "Under the ideal gas model, a sample has pressure 1 atm, volume is 22.414 L, and contains 1 mol. What is its temperature?",
            "ideal_gas.temperature",
            "273.150371143",
            "K",
        ),
        (
            "Assuming an ideal gas, a sample has pressure 100 kPa, volume is 24.94338785445972 L, and temperature is 300 K. Determine its amount of substance.",
            "ideal_gas.amount",
            "1",
            "mol",
        ),
    ],
)
def test_all_allowlisted_science_formulas_receive_independent_witness(query, method, display, unit):
    result = checker.check_science_certificate(
        query=query,
        display_answer=display,
        method=method,
        unit=unit,
    )

    assert result["status"] == "passed"
    assert result["algorithmically_independent"] is True
    assert result["checker_id"] == checker.SCIENCE_CHECKER_ID


def test_science_witness_rejects_wrong_unit_or_display():
    result = checker.check_science_certificate(
        query="Assuming constant acceleration, an object starts from rest, acceleration 3 m/s^2, and time 4 s. What is its displacement?",
        display_answer="25",
        method="constant_acceleration.displacement",
        unit="m/s",
    )

    assert result["status"] == "failed"
    assert result["algorithmically_independent"] is False
    assert result["reason"] == "science_display_or_unit_mismatch"
