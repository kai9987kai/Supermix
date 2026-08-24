"""Statistics and cohort-hygiene invariants for the v56 promotion gate.

The gate decides whether a measured difference is real, so its arithmetic is
checked against values computed by hand rather than against itself.

Pinned here:

1. McNemar's exact binomial matches closed-form values, and the continuity-
   corrected approximation takes over only above the declared threshold.
2. A symmetric disagreement is never significant, however large.
3. Wilson intervals match published values and stay inside [0, 1] at the ends.
4. Cohorts already used by the v51 train/test split or an existing gate can never
   be scored -- a pass has to be measured, not selected.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import run_v56_promotion_gate as gate  # noqa: E402


# ---------------------------------------------------------------------------
# McNemar
# ---------------------------------------------------------------------------


def test_mcnemar_with_no_disagreement_is_never_significant() -> None:
    result = gate.mcnemar(0, 0)
    assert result["p_value"] == 1.0
    assert result["method"] == "none"
    assert result["discordant_pairs"] == 0


def test_mcnemar_exact_matches_the_closed_form() -> None:
    """10 discordant pairs, all favouring the candidate: p = 2 / 2^10."""

    result = gate.mcnemar(baseline_only=0, candidate_only=10)
    assert result["method"] == "exact_binomial"
    # receipts store six significant digits, so compare at that precision
    assert result["p_value"] == pytest.approx(2.0 / 1024.0, rel=1e-5)


def test_mcnemar_is_symmetric_in_its_arguments() -> None:
    forward = gate.mcnemar(3, 17)
    backward = gate.mcnemar(17, 3)
    assert forward["p_value"] == backward["p_value"]
    assert forward["candidate_only_wins"] == backward["baseline_only_wins"]


def test_an_even_split_is_not_significant_at_any_size() -> None:
    for discordant in (10, 500, 4000):
        half = discordant // 2
        assert gate.mcnemar(half, half)["p_value"] > 0.5


def test_large_samples_switch_to_the_corrected_chi_square() -> None:
    small = gate.mcnemar(400, 500)
    large = gate.mcnemar(600, 700)
    assert small["method"] == "exact_binomial"
    assert large["method"] == "chi_square_continuity_corrected"
    # (|700-600| - 1)^2 / 1300
    assert large["statistic"] == pytest.approx((99.0**2) / 1300.0, rel=1e-6)
    assert large["p_value"] == pytest.approx(math.erfc(math.sqrt(large["statistic"] / 2.0)), rel=1e-5)


def test_a_decisive_difference_clears_the_default_alpha() -> None:
    assert gate.mcnemar(baseline_only=20, candidate_only=400)["p_value"] < 0.01


# ---------------------------------------------------------------------------
# Sign test
# ---------------------------------------------------------------------------


def test_sign_test_on_a_clean_sweep() -> None:
    result = gate.sign_test(wins=20, losses=0)
    assert result["p_value"] == pytest.approx(2.0 / (2.0**20), rel=1e-5)
    assert result["trials"] == 20


def test_sign_test_with_no_trials_is_inconclusive() -> None:
    assert gate.sign_test(0, 0)["p_value"] == 1.0


def test_sign_test_on_an_even_split_is_inconclusive() -> None:
    assert gate.sign_test(10, 10)["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_the_published_value() -> None:
    low, high = gate.wilson_interval(50, 100)
    assert low == pytest.approx(0.40383, abs=1e-4)
    assert high == pytest.approx(0.59617, abs=1e-4)


def test_wilson_interval_stays_inside_the_unit_range() -> None:
    for successes, total in ((0, 40), (40, 40), (1, 10_000)):
        low, high = gate.wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_interval_tightens_with_more_samples() -> None:
    narrow = gate.wilson_interval(1_000, 4_000)
    wide = gate.wilson_interval(250, 1_000)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_on_an_empty_cohort_is_degenerate() -> None:
    assert gate.wilson_interval(0, 0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Cohort hygiene
# ---------------------------------------------------------------------------


def test_the_v51_train_and_test_seeds_are_reserved() -> None:
    assert 51 in gate.RESERVED_SEEDS
    assert 52 in gate.RESERVED_SEEDS


def test_every_existing_v51_gate_cohort_is_reserved() -> None:
    for seed in (641, 643, 647, 653, 659, 661, 673, 677, 719, 727, 733, 739):
        assert seed in gate.RESERVED_SEEDS, seed
    # the two 40,000-example promotion-receipt ranges
    assert 21052 in gate.RESERVED_SEEDS and 40052 in gate.RESERVED_SEEDS
    assert 41052 in gate.RESERVED_SEEDS and 60052 in gate.RESERVED_SEEDS


def test_default_seeds_are_fresh_distinct_and_deterministic() -> None:
    seeds = gate.default_seeds(20)
    assert len(seeds) == 20
    assert len(set(seeds)) == 20
    assert not set(seeds) & gate.RESERVED_SEEDS
    assert seeds == gate.default_seeds(20)


def test_default_seeds_skip_a_reserved_range() -> None:
    """Starting inside a reserved range must step over it, not through it."""

    seeds = gate.default_seeds(5, start=21050, stride=1)
    assert not set(seeds) & gate.RESERVED_SEEDS
    assert len(set(seeds)) == 5
