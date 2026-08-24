"""Tests for the recall meter.

The meter exists to stop the interface presenting recitation as writing. Its
failure modes are symmetrical and both are bad: calling recalled text novel
flatters the model, and calling novel text recalled slanders it. Both directions
are tested.
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

import recall_index  # noqa: E402

MEMORISED = (
    "The moment hung in the air like a held breath and nobody moved at all today"
)
NOVEL = (
    "A quiet librarian repaired a bicycle while thunderstorms argued about tax "
    "policy in the marmalade factory"
)


def _index(texts=None):
    return recall_index.RecallIndex.from_texts(texts or [MEMORISED], n=8)


# -- the two directions -----------------------------------------------------


def test_memorised_text_is_reported_as_recalled():
    report = _index().score(MEMORISED)

    assert report.verbatim_rate == 1.0
    assert report.verdict == "largely_recalled"
    assert report.longest_verbatim_words >= 8


def test_novel_text_is_reported_as_novel():
    """The complementary failure: an index must not flag everything."""

    report = _index().score(NOVEL)

    assert report.verbatim_rate == 0.0
    assert report.verdict == "mostly_novel"
    assert report.longest_verbatim_words == 0


def test_partial_reuse_lands_between_the_two():
    corpus = ["the first eight words of this sentence are indexed here for testing"]
    reply = "the first eight words of this sentence are indexed here " + NOVEL

    report = _index(corpus).score(reply)

    assert 0.0 < report.verbatim_rate < 1.0


# -- properties the numbers must have --------------------------------------


def test_longest_run_counts_words_not_windows():
    """k consecutive matching windows cover k + n - 1 words."""

    report = _index().score(MEMORISED)
    words = len(recall_index.normalise(MEMORISED))

    assert report.longest_verbatim_words == words


def test_case_and_punctuation_do_not_hide_recall():
    report = _index().score(MEMORISED.upper() + " !!! ")

    assert report.verdict == "largely_recalled"


def test_short_replies_are_not_called_novel():
    """A three-word answer has no window; claiming originality would be wrong."""

    report = _index().score("yes it is")

    assert report.verdict == "too_short_to_judge"
    assert report.verbatim_rate == 0.0


def test_empty_index_reports_nothing_matched_rather_than_everything():
    empty = recall_index.RecallIndex.from_texts([], n=8)

    report = empty.score(MEMORISED)

    assert report.matched == 0
    assert report.verdict == "mostly_novel"


def test_index_deduplicates_windows():
    repeated = recall_index.RecallIndex.from_texts([MEMORISED] * 50, n=8)
    once = _index()

    assert repeated.hashes.size == once.hashes.size
    assert repeated.rows == 50


# -- corpus loading ---------------------------------------------------------


def test_jsonl_indexes_replies_not_prompts(tmp_path):
    """Indexing prompts would score a reply as recalled for echoing the user."""

    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"user": MEMORISED, "assistant": NOVEL}) + "\n", encoding="utf-8"
    )

    index = recall_index.RecallIndex.from_jsonl(path)

    assert index.score(NOVEL).verdict == "largely_recalled"
    assert index.score(MEMORISED).verdict == "mostly_novel"


def test_jsonl_survives_malformed_lines(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"assistant": MEMORISED}) + "\n{ broken\n\n", encoding="utf-8"
    )

    index = recall_index.RecallIndex.from_jsonl(path)

    assert index.score(MEMORISED).verdict == "largely_recalled"


def test_jsonl_respects_limit(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        "\n".join(json.dumps({"assistant": f"row number {i} of the corpus goes here now"})
                  for i in range(20)),
        encoding="utf-8",
    )

    index = recall_index.RecallIndex.from_jsonl(path, limit=5)

    assert index.rows == 5


def test_report_serialises_for_the_api():
    payload = _index().score(MEMORISED).to_dict()

    assert set(payload) == {
        "windows", "matched", "verbatim_rate", "longest_verbatim_words", "verdict"
    }
    assert isinstance(payload["verbatim_rate"], float)
