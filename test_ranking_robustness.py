"""A malformed candidate score must not be able to hijack the ranking.

`bucket_score` is supplied by the caller, not derived from the candidate text.
It carries probability semantics -- a softmax probability scaled by
`--db_score_scale` / `--memory_score_scale` -- but nothing bounded it.

Measured on a real 60-candidate pool drawn from the shipped bucket metadata,
before this guard:

* `bucket_score=50` scored +9.346 against a pool maximum of +0.510 and moved
  from rank 16 to rank 1, so one caller-supplied number silently overrode every
  text signal; and
* `bucket_score=NaN` propagated into the final scores and sorted to the FRONT,
  so a malformed row won outright.

These tests pin both, and pin that ordinary values are untouched.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RUNTIME = ROOT / "runtime_python"


def _load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))


@pytest.fixture(scope="module")
def pipeline():
    return _load("robustness_chat_pipeline", SOURCE / "chat_pipeline.py")


QUERY = "how do i make a slow postgres query faster"


def _pool(pipeline, size: int = 24, bad=None):
    candidates = []
    for index in range(size):
        text = (
            f"Candidate {index}: tune the query plan, add an index on the filtered "
            f"column, and check the statistics for skew."
        )
        vec = pipeline.featurize_text(text).tolist()
        candidates.append(
            {"text": text, "vec": vec, "ctx_vec": vec, "count": 1, "bucket_score": 0.5}
        )
    if bad is not None:
        candidates[-1]["bucket_score"] = bad
    return candidates


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), float("-inf"), 1e30, -1e30, 50.0, 10_000.0],
)
def test_malformed_bucket_score_cannot_produce_a_non_finite_ranking(pipeline, bad) -> None:
    order, scores = pipeline.rank_response_candidates(_pool(pipeline, bad=bad), QUERY, [])

    assert torch.isfinite(scores).all(), f"bucket_score={bad} produced a non-finite score"
    assert not any(math.isnan(float(value)) for value in scores)
    assert sorted(order) == list(range(len(order))), "ranking lost or duplicated candidates"


def test_a_huge_bucket_score_is_bounded_rather_than_dominating(pipeline) -> None:
    _, baseline = pipeline.rank_response_candidates(_pool(pipeline), QUERY, [])
    _, hijacked = pipeline.rank_response_candidates(
        _pool(pipeline, bad=10_000.0), QUERY, []
    )

    baseline_spread = float(baseline.max() - baseline.min())
    worst = float(hijacked.max())

    # Before the guard this was +9.35 against a pool max of +0.51.
    assert worst <= float(baseline.max()) + pipeline.MAX_BUCKET_BONUS * 0.18 + 1e-6
    assert worst < 10 * max(baseline_spread, 1e-3), (
        "one caller-supplied score still dwarfs the whole pool"
    )


def test_nan_score_does_not_sort_to_the_front(pipeline) -> None:
    """NaN comparisons are false, which previously floated the bad row to rank 1."""

    order, _ = pipeline.rank_response_candidates(
        _pool(pipeline, bad=float("nan")), QUERY, []
    )
    poisoned = len(order) - 1
    assert order[0] != poisoned, "a NaN bucket_score won the ranking"


def test_ordinary_scores_are_completely_unaffected_by_the_guard(pipeline) -> None:
    """Every default configuration produces values <= 1.0 and must not move."""

    for value in (0.0, 0.25, 0.5, 0.99, 1.0):
        pool = _pool(pipeline)
        for row in pool:
            row["bucket_score"] = value
        order, scores = pipeline.rank_response_candidates(pool, QUERY, [])
        assert torch.isfinite(scores).all()
        assert len(order) == len(pool)

    # A within-range spread must still rank exactly by that spread. Every
    # candidate carries identical text here, so `bucket_score` is the only
    # thing that differs. `_pool` gives each candidate a distinct "Candidate N"
    # prefix, which makes the text signals differ too and would leave this
    # assertion testing the sum of ~20 signals rather than the guard.
    text = (
        "Tune the query plan, add an index on the filtered column, and check "
        "statistics for skew."
    )
    vec = pipeline.featurize_text(text).tolist()
    pool = [
        {"text": text, "vec": vec, "ctx_vec": vec, "count": 1, "bucket_score": index / 24}
        for index in range(24)
    ]
    order, _ = pipeline.rank_response_candidates(pool, QUERY, [])
    assert order == list(range(23, -1, -1)), (
        "with text held constant, ranking must follow bucket_score exactly"
    )


def test_source_and_runtime_agree_on_the_bound() -> None:
    runtime = _load("robustness_runtime_chat_pipeline", RUNTIME / "chat_pipeline.py")
    source = _load("robustness_source_chat_pipeline", SOURCE / "chat_pipeline.py")
    assert source.MAX_BUCKET_BONUS == runtime.MAX_BUCKET_BONUS

    pool = [
        {
            "text": f"candidate {i}",
            "vec": source.featurize_text(f"candidate {i}").tolist(),
            "ctx_vec": source.featurize_text(f"candidate {i}").tolist(),
            "count": 1,
            "bucket_score": 0.5,
        }
        for i in range(8)
    ]
    pool[-1]["bucket_score"] = float("inf")
    source_order, source_scores = source.rank_response_candidates(pool, QUERY, [])
    runtime_order, runtime_scores = runtime.rank_response_candidates(pool, QUERY, [])
    assert source_order == runtime_order
    assert torch.equal(source_scores, runtime_scores)
