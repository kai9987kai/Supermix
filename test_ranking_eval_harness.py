"""Contracts for the retrieval-quality measurement harness.

The harness exists to decide whether a ranking change helped. That makes its own
correctness load-bearing: a benchmark that scores an equivalent answer as wrong,
or that lets a measured-harmful mode reach production, is worse than no benchmark
because it produces confident wrong conclusions.

These tests pin the four properties the measurements depend on:

1. experimental fusion modes cannot be selected at runtime;
2. equivalence-aware scoring credits a response that says what the gold says;
3. the fixture mirrors production's vector semantics; and
4. the harness is deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
FIXTURE = SOURCE / "ranking_eval_set.json"


def _load(name: str, path: Path):
    sys.path.insert(0, str(SOURCE))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SOURCE))


@pytest.fixture(scope="module")
def fusion():
    return _load("harness_score_fusion", SOURCE / "score_fusion.py")


@pytest.fixture(scope="module")
def bench():
    return _load("harness_benchmark_ranking", SOURCE / "benchmark_ranking_quality.py")


# --------------------------------------------------------------------------
# 1. A measured-harmful mode must not be reachable from a config string
# --------------------------------------------------------------------------

def test_experimental_fusion_modes_are_not_runtime_selectable(fusion) -> None:
    """`calibrated` measured 51 points of top-1 below legacy. It is not an option.

    Rank calibration destroys the margin carried by sim_ctx, so no weight
    re-tune recovers it; the information is gone before the weights apply.
    """

    for mode in fusion.EXPERIMENTAL_FUSION_MODES:
        assert mode in fusion.FUSION_MODES
        assert mode not in fusion.RUNTIME_FUSION_MODES
        assert fusion.resolve_fusion_mode(mode) == fusion.DEFAULT_FUSION_MODE
        assert fusion.resolve_fusion_mode(mode, allow_experimental=True) == mode

    for mode in fusion.RUNTIME_FUSION_MODES:
        assert fusion.resolve_fusion_mode(mode) == mode


def test_unknown_or_broken_modes_degrade_to_the_measured_best_path(fusion) -> None:
    for value in ("", None, "   ", "typo", "CALIBRATED ", 0, [], {"mode": "x"}):
        assert fusion.resolve_fusion_mode(value) == fusion.DEFAULT_FUSION_MODE


# --------------------------------------------------------------------------
# 2. Equivalence-aware scoring
# --------------------------------------------------------------------------

def test_boilerplate_variants_of_one_answer_count_as_equivalent(bench) -> None:
    """58% of raw top-1 "failures" were this: same answer, different framing."""

    gold = "Hello. Tell me what you need and I will do my best to help."
    variants = [
        "Okay. Hello. Tell me what you need and I will do my best to help.",
        "Hello. Tell me what you need and I will do my best to help. "
        "Let me know if you want a deeper walkthrough.",
        "Short answer: Hello. Tell me what you need and I will do my best to help.",
    ]
    for variant in variants:
        assert bench.responses_are_equivalent(gold, variant), variant


def test_genuinely_different_answers_are_not_equivalent(bench) -> None:
    """The credit must not be so loose that a wrong answer passes."""

    pairs = [
        (
            "Vectorize where possible or batch operations to reduce Python overhead.",
            "Okay. Yes. I will give a short plan with steps, risks, and verification.",
        ),
        (
            "Yes. I can add unit tests and integration tests based on your code.",
            "Hi there. What can I help you with today?",
        ),
        (
            "Run the script from the project directory and confirm the weights path.",
            "Install the missing package in the same environment and rerun.",
        ),
    ]
    for gold, other in pairs:
        assert not bench.responses_are_equivalent(gold, other), (gold, other)


def test_equivalence_is_symmetric_and_degenerate_safe(bench) -> None:
    left = "Okay. Run the script from the project directory."
    right = "Run the script from the project directory. Let me know if you want more."
    assert bench.responses_are_equivalent(left, right)
    assert bench.responses_are_equivalent(right, left)

    for empty in ("", "   ", None, "Okay.", "Short answer:"):
        assert not bench.responses_are_equivalent(empty, left)
        assert not bench.responses_are_equivalent(left, empty)


def test_equivalence_aware_scoring_never_reports_worse_than_strict(bench) -> None:
    """Crediting an equivalent answer can only move a rank earlier, never later."""

    if not FIXTURE.exists():
        pytest.skip("corpus fixture not built")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = {**payload, "cases": payload["cases"][:25]}

    strict = bench._corpus_gold_ranks(payload, "legacy", strict=True)
    lenient = bench._corpus_gold_ranks(payload, "legacy", strict=False)

    assert len(strict) == len(lenient) == 25
    assert all(soft <= hard for soft, hard in zip(lenient, strict))
    assert all(rank >= 1 for rank in lenient)


# --------------------------------------------------------------------------
# 3. Fixture must mirror production, or it measures the wrong thing
# --------------------------------------------------------------------------

def test_fixture_carries_the_query_each_response_was_stored_under() -> None:
    """Production's dominant signal is sim_ctx: the live query against a
    candidate's stored context. A fixture without those queries measured
    legacy at 1% top-1 instead of 87%.
    """

    if not FIXTURE.exists():
        pytest.skip("corpus fixture not built")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert payload["negatives"] == "hard", "random negatives score 100% and measure nothing"
    assert payload["probe_count"] >= 100
    for case in payload["cases"][:50]:
        assert case["query"].strip()
        assert case["gold"].strip()
        assert case["distractors"]
        for distractor in case["distractors"]:
            assert distractor.get("query", "").strip(), "distractor lost its source query"


def test_gold_is_not_trivially_identifiable_by_position(bench) -> None:
    """The gold is index 0 in every case; ranking must not be able to exploit that."""

    if not FIXTURE.exists():
        pytest.skip("corpus fixture not built")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = {**payload, "cases": payload["cases"][:40]}

    ranks = bench._corpus_gold_ranks(payload, "legacy", strict=True)
    # A positional leak would show up as a perfect score on a hard-negative set.
    assert not all(rank == 1 for rank in ranks)


# --------------------------------------------------------------------------
# 4. Determinism
# --------------------------------------------------------------------------

def test_harness_is_deterministic(bench) -> None:
    if not FIXTURE.exists():
        pytest.skip("corpus fixture not built")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = {**payload, "cases": payload["cases"][:20]}

    first = bench._corpus_gold_ranks(payload, "legacy")
    second = bench._corpus_gold_ranks(payload, "legacy")
    assert first == second


def test_gated_mode_is_rank_identical_to_legacy(bench) -> None:
    """`gated` is a safety net, not a behaviour change. It must stay a no-op."""

    if not FIXTURE.exists():
        pytest.skip("corpus fixture not built")
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload = {**payload, "cases": payload["cases"][:30]}

    assert bench._corpus_gold_ranks(payload, "legacy") == bench._corpus_gold_ranks(
        payload, "gated"
    )
