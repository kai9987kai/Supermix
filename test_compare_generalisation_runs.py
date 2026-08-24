"""Tests for the v61 architecture comparison tool.

The tool's only real job is refusing comparisons that look reasonable and are
not. Most of these tests therefore build receipt pairs that differ in one way
that invalidates the comparison, and require a raise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import compare_generalisation_runs as compare  # noqa: E402


def _receipt(
    vocab: int = 10538,
    sentences=("alpha.", "beta."),
    rows=(1007, 422, 2394),
    losses=(0.2260, 0.2180, 0.3135),
    experts: int = 8,
    layers: int = 4,
    steps: int = 1000,
    total: int = 4_988_073,
):
    return {
        "held_out_sentences": list(sentences),
        "tokenizer": {"vocab_size": vocab},
        "parameters": {"total": total, "active_per_token": 3_204_381},
        "hyperparameters": {"steps": steps},
        "selection": {"best_dev_loss": 0.2852},
        "config": {"n_routed_experts": experts, "n_layers": layers},
        "tiers": {
            "tier1_seen_response": {"pairs": rows[0], "loss": losses[0]},
            "tier2_unseen_response": {"pairs": rows[1], "loss": losses[1]},
            "tier3_unseen_sentence": {"pairs": rows[2], "loss": losses[2]},
        },
    }


def _write(tmp_path: Path, name: str, payload) -> str:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "generalisation_results.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return str(directory)


def test_comparable_runs_produce_a_table(tmp_path):
    a = _write(tmp_path, "a", _receipt())
    b = _write(tmp_path, "b", _receipt(losses=(0.1856, 0.2016, 0.2873), experts=32,
                                       layers=6, steps=2000, total=15_883_701))

    result = compare.compare_runs([a, b])

    assert len(result["runs"]) == 2
    tier1 = next(r for r in result["tiers"] if r["tier"] == "tier1_seen_response")
    assert tier1["delta_vs_first"] == pytest.approx(-0.0404, abs=1e-4)


def test_refuses_runs_with_different_vocabularies(tmp_path):
    """Different vocabulary means perplexity is a different unit."""

    a = _write(tmp_path, "a", _receipt(vocab=582))
    b = _write(tmp_path, "b", _receipt(vocab=10538))

    with pytest.raises(ValueError, match="different unit"):
        compare.compare_runs([a, b])


def test_refuses_runs_with_different_withheld_sentences(tmp_path):
    a = _write(tmp_path, "a", _receipt(sentences=("alpha.", "beta.")))
    b = _write(tmp_path, "b", _receipt(sentences=("alpha.", "gamma.")))

    with pytest.raises(ValueError, match="different sentences"):
        compare.compare_runs([a, b])


def test_refuses_runs_with_different_tier_row_counts(tmp_path):
    a = _write(tmp_path, "a", _receipt(rows=(1007, 422, 2394)))
    b = _write(tmp_path, "b", _receipt(rows=(1007, 422, 2000)))

    with pytest.raises(ValueError, match="different tier row counts"):
        compare.compare_runs([a, b])


def test_sentence_order_does_not_matter(tmp_path):
    """The withheld set is a set; only its contents decide comparability."""

    a = _write(tmp_path, "a", _receipt(sentences=("alpha.", "beta.")))
    b = _write(tmp_path, "b", _receipt(sentences=("beta.", "alpha.")))

    compare.compare_runs([a, b])  # must not raise


def test_requires_two_runs(tmp_path):
    a = _write(tmp_path, "a", _receipt())

    with pytest.raises(ValueError, match="at least two"):
        compare.compare_runs([a])


def test_missing_receipt_is_reported_clearly(tmp_path):
    a = _write(tmp_path, "a", _receipt())
    missing = str(tmp_path / "nope")

    with pytest.raises(FileNotFoundError, match="no receipt"):
        compare.compare_runs([a, missing])
