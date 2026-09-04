"""Unit tests for AI-Dem-Lab Quantum Randomness & Bell Locality Sandbox."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import pytest
from nexus_engine import (
    QuantumBellEngine,
    BellExperimentResult,
    WolframComplexityAnalyzer,
    WolframComplexityResult,
    NexusEngine,
)


def test_quantum_bell_engine_analytical_violation():
    engine = QuantumBellEngine()
    result = engine.simulate_chsh(
        theta_a=0.0,
        theta_a_prime=45.0,
        theta_b=22.5,
        theta_b_prime=67.5,
        shots=500,
        seed=1234,
    )
    assert isinstance(result, BellExperimentResult)
    # Standard optimal angles yield S = 2 * sqrt(2) ~ 2.8284
    assert result.chsh_s_quantum > 2.80
    assert result.violates_classical_bound is True
    assert result.classical_bound == 2.0
    assert 0.99 <= result.tsirelson_ratio <= 1.01

    # Classical LHV bound check
    assert result.chsh_s_classical <= 2.05


def test_quantum_bell_collinear_angles():
    engine = QuantumBellEngine()
    # If all angles are 0, E = 1 for all, S = |1 - 1 + 1 + 1| = 2 <= 2
    result = engine.simulate_chsh(
        theta_a=0.0,
        theta_a_prime=0.0,
        theta_b=0.0,
        theta_b_prime=0.0,
        shots=200,
    )
    assert result.chsh_s_quantum == pytest.approx(2.0, abs=1e-3)
    assert result.violates_classical_bound is False


def test_nexus_engine_run_bell_experiment():
    engine = NexusEngine()
    res = engine.run_bell_experiment(shots=200)
    data = res.to_dict()
    assert "chsh_s_quantum" in data
    assert "violates_classical_bound" in data
    assert "quantum_correlations" in data


def test_wolfram_complexity_analyzer():
    analyzer = WolframComplexityAnalyzer()
    res30 = analyzer.analyze(rule=30, steps=8, width=15)
    assert isinstance(res30, WolframComplexityResult)
    assert res30.complexity_class == "Class 3 (Chaotic)"
    assert res30.spatial_entropy > 0.0
    assert len(res30.grid) == 8

    res110 = analyzer.analyze(rule=110, steps=8, width=15)
    assert res110.complexity_class == "Class 4 (Complex/Universal)"

    res0 = analyzer.analyze(rule=0, steps=8, width=15)
    assert res0.complexity_class == "Class 1 (Uniform)"

