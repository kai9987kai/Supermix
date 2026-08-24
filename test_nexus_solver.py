"""Tests for the NexusSolver — exact multi-paradigm math & science problem solver."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
for p in (ROOT, SOURCE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nexus_solver as ns


# ---------------------------------------------------------------------------
# Basic API surface
# ---------------------------------------------------------------------------

def test_solver_returns_result_object():
    res = ns.solve_problem("What is the kinetic energy of a 5 kg mass moving at 4 m/s?")
    assert isinstance(res, ns.SolverResult)
    assert hasattr(res, "solved")
    assert hasattr(res, "display_answer")
    assert hasattr(res, "steps")
    assert hasattr(res, "receipt")


def test_unsolved_query_is_safe():
    res = ns.solve_problem("What colour is the sky?")
    assert res.solved is False
    assert res.domain == "unresolved"
    assert res.answer_value is None


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------

def test_kinetic_energy():
    res = ns.solve_problem("Find the kinetic energy of a mass m = 10 kg moving at velocity v = 6 m/s")
    assert res.solved is True
    assert "kinetic" in res.target.lower()
    assert abs(res.answer_value - 180.0) < 0.1    # 0.5 * 10 * 36 = 180 J
    assert res.unit == "J"


def test_kinetic_energy_receipt_present():
    res = ns.solve_problem("kinetic energy mass m = 2 kg velocity v = 10 m/s")
    assert res.solved is True
    assert res.receipt is not None
    assert len(res.receipt.receipt_sha256) == 64


def test_torricelli_final_velocity():
    res = ns.solve_problem(
        "final velocity with initial velocity u = 0 m/s, acceleration a = 10 m/s^2, displacement s = 20 m"
    )
    assert res.solved is True
    assert abs(res.answer_value - 20.0) < 0.05


# ---------------------------------------------------------------------------
# Dynamics – Newton's Laws
# ---------------------------------------------------------------------------

def test_newton_second_law_force():
    res = ns.solve_problem("net force with mass m = 5 kg and acceleration a = 3 m/s^2")
    assert res.solved is True
    assert abs(res.answer_value - 15.0) < 0.01
    assert res.unit == "N"


def test_momentum():
    res = ns.solve_problem("linear momentum with mass m = 70 kg velocity v = 5 m/s")
    assert res.solved is True
    assert abs(res.answer_value - 350.0) < 0.01
    assert "kg" in res.unit


def test_impulse():
    res = ns.solve_problem("impulse with force F = 100 N time t = 0.5 s")
    assert res.solved is True
    assert abs(res.answer_value - 50.0) < 0.01


def test_friction_force():
    res = ns.solve_problem("frictional force with friction coefficient mu = 0.3 and normal force N = 200 N")
    assert res.solved is True
    assert abs(res.answer_value - 60.0) < 0.01


# ---------------------------------------------------------------------------
# Work, Energy & Power
# ---------------------------------------------------------------------------

def test_potential_energy():
    res = ns.solve_problem("gravitational potential energy mass m = 2 kg height d = 5 m")
    assert res.solved is True
    # PE = m*g*h = 2 * 9.80665 * 5 = 98.0665 J
    assert abs(res.answer_value - 98.0665) < 0.01
    assert res.unit == "J"


def test_work_done():
    res = ns.solve_problem("work done with force F = 200 N distance d = 8 m")
    assert res.solved is True
    assert abs(res.answer_value - 1600.0) < 0.01


def test_power():
    res = ns.solve_problem("power with work W = 3000 J time t = 10 s")
    assert res.solved is True
    assert abs(res.answer_value - 300.0) < 0.01
    assert res.unit == "W"


# ---------------------------------------------------------------------------
# Thermodynamics & Heat
# ---------------------------------------------------------------------------

def test_carnot_efficiency():
    res = ns.solve_problem("carnot efficiency with hot temperature T_H = 600 K cold temperature T_C = 300 K")
    assert res.solved is True
    # eta = 1 - 300/600 = 0.5 => 50%
    assert "50" in res.display_answer


# ---------------------------------------------------------------------------
# Hydrostatics & Fluids
# ---------------------------------------------------------------------------

def test_hydrostatic_pressure():
    res = ns.solve_problem("hydrostatic pressure at depth h = 10 m with density rho = 1000 kg/m^3")
    assert res.solved is True
    # P = 1000 * 9.80665 * 10 = 98066.5 Pa
    assert abs(res.answer_value - 98066.5) < 10.0
    assert res.unit == "Pa"


def test_buoyant_force():
    res = ns.solve_problem("archimedes buoyant force density rho = 1000 kg/m^3 volume V = 0.5 m^3")
    assert res.solved is True
    # Fb = 1000 * 0.5 * 9.80665 = 4903.325 N
    assert abs(res.answer_value - 4903.325) < 1.0


# ---------------------------------------------------------------------------
# Electromagnetism & Circuits
# ---------------------------------------------------------------------------

def test_ohms_law_voltage():
    res = ns.solve_problem("voltage with current I = 2 A resistance R = 47 ohm")
    assert res.solved is True
    assert abs(res.answer_value - 94.0) < 0.01
    assert res.unit == "V"


def test_ohms_law_current():
    res = ns.solve_problem("current with voltage V = 12 V resistance R = 4 ohm")
    assert res.solved is True
    assert abs(res.answer_value - 3.0) < 0.01
    assert res.unit == "A"


def test_electrical_power():
    res = ns.solve_problem("electrical power with voltage V = 230 V current I = 5 A")
    assert res.solved is True
    assert abs(res.answer_value - 1150.0) < 0.01
    assert res.unit == "W"


def test_parallel_resistance():
    res = ns.solve_problem("equivalent parallel resistance R1 = 6 ohm R2 = 3 ohm")
    assert res.solved is True
    # Req = (6*3)/(6+3) = 18/9 = 2 Ohm
    assert abs(res.answer_value - 2.0) < 0.01


# ---------------------------------------------------------------------------
# Waves & Optics
# ---------------------------------------------------------------------------

def test_wave_speed():
    res = ns.solve_problem("wave speed with frequency f = 440 Hz wavelength lambda = 0.77 m")
    assert res.solved is True
    assert abs(res.answer_value - 440 * 0.77) < 0.1


def test_wave_period():
    res = ns.solve_problem("wave period with frequency f = 50 Hz")
    assert res.solved is True
    assert abs(res.answer_value - 0.02) < 0.001
    assert res.unit == "s"


# ---------------------------------------------------------------------------
# Chemistry & Solutions
# ---------------------------------------------------------------------------

def test_molarity():
    res = ns.solve_problem("molarity concentration with moles n = 2 mol volume V = 0.5 L")
    assert res.solved is True
    assert abs(res.answer_value - 4.0) < 0.01
    assert res.unit == "M"


def test_dilution():
    res = ns.solve_problem(
        "dilution with initial concentration M1 = 2 mol/L initial volume V1 = 100 mL "
        "target concentration M2 = 0.5 mol/L"
    )
    assert res.solved is True
    # V2 = (2 * 0.1) / 0.5 = 0.4 L = 400 mL
    assert abs(res.answer_value - 400.0) < 0.1


# ---------------------------------------------------------------------------
# Pure Algebra
# ---------------------------------------------------------------------------

def test_quadratic_roots_real():
    # x^2 - 5x + 6 = 0 => x=3, x=2
    res = ns.solve_problem("solve quadratic a = 1, b = -5, c = 6")
    assert res.solved is True
    assert "3" in res.display_answer and "2" in res.display_answer


def test_linear_system_2x2():
    # 2x + 3y = 8 and 4x - y = 2  => x=1, y=2
    res = ns.solve_problem("solve system 2x +3y = 8 and 4x -1y = 2")
    assert res.solved is True
    assert "1" in res.display_answer


# ---------------------------------------------------------------------------
# Compound Interest
# ---------------------------------------------------------------------------

def test_compound_interest():
    # P=1000, r=5%, t=2 years, annual => 1000*(1.05)^2 = 1102.5
    res = ns.solve_problem(
        "compound interest principal P = 1000, rate r = 5%, time t = 2 years"
    )
    assert res.solved is True
    assert "1102" in res.display_answer


# ---------------------------------------------------------------------------
# Series & Progressions
# ---------------------------------------------------------------------------

def test_arithmetic_series_sum():
    # a=1, d=1, n=10 => Sn = 10/2 * (2 + 9) = 55
    res = ns.solve_problem("sum of arithmetic series first term a = 1 common difference d = 1 number of terms n = 10")
    assert res.solved is True
    assert abs(res.answer_value - 55.0) < 0.01


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_pythagorean_hypotenuse():
    # a=3m, b=4m => c=5m
    res = ns.solve_problem("hypotenuse of pythagorean triangle with side a = 3 m side b = 4 m")
    assert res.solved is True
    assert abs(res.answer_value - 5.0) < 0.01
    assert res.unit == "m"


# ---------------------------------------------------------------------------
# Combinatorics
# ---------------------------------------------------------------------------

def test_combinations():
    # C(5,2) = 10
    res = ns.solve_problem("combination n choose k n = 5 k = 2")
    assert res.solved is True
    assert abs(res.answer_value - 10.0) < 0.01


def test_permutations():
    # P(5,2) = 20
    res = ns.solve_problem("permutation P(n,k) n = 5 k = 2")
    assert res.solved is True
    assert abs(res.answer_value - 20.0) < 0.01


# ---------------------------------------------------------------------------
# Derivation Steps Integrity
# ---------------------------------------------------------------------------

def test_derivation_steps_are_non_empty():
    res = ns.solve_problem("kinetic energy mass m = 3 kg velocity v = 4 m/s")
    assert res.solved is True
    assert len(res.steps) >= 1
    for step in res.steps:
        assert isinstance(step, ns.DerivationStep)
        assert step.description
        assert step.formula_latex


def test_receipt_sha256_is_64_hex_chars():
    res = ns.solve_problem("momentum with mass m = 100 kg velocity v = 10 m/s")
    assert res.solved is True
    assert res.receipt is not None
    sha = res.receipt.receipt_sha256
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)
