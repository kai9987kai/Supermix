"""Tests for the v62 domain-balanced corpus and its per-domain evaluation.

The corpus builder's job is to keep domains separable and to stop a prose-shaped
filter from deleting a domain whose answers are values rather than sentences.
That second failure was real: an 8-character minimum silently removed 73.5% of
the arithmetic rows, so it is pinned here.
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

import build_v62_corpus as builder  # noqa: E402
import eval_v62_domains as domains  # noqa: E402


# -- the short-answer regression -------------------------------------------


def test_maths_keeps_short_numeric_answers():
    """'79' is a complete answer to '498 - 419', not a truncation."""

    record = {"user": "Solve this basic math problem: 498 - 419", "assistant": "79"}

    assert builder._clean(record, "maths") == (record["user"], "79")


def test_prose_domains_still_drop_truncated_answers():
    """The filter must keep working where it was right."""

    record = {"user": "tell me a story", "assistant": "ok"}

    assert builder._clean(record, "creativity") is None


def test_short_answer_domains_are_declared_not_incidental():
    for domain in ("maths", "language"):
        assert domain in builder.SHORT_ANSWER_DOMAINS
        assert builder.SHORT_ANSWER_DOMAINS[domain] < builder.MIN_RESPONSE_CHARACTERS


def test_rows_without_a_prompt_are_dropped_in_every_domain():
    assert builder._clean({"user": "", "assistant": "42"}, "maths") is None


def test_alternative_field_names_are_accepted():
    record = {"prompt": "what is 2+2", "response": "4"}

    assert builder._clean(record, "maths") == ("what is 2+2", "4")


# -- category routing -------------------------------------------------------


def test_category_filter_selects_only_named_categories():
    keep = builder._category_in("chain_of_thought", "socratic")

    assert keep({"category": "chain_of_thought"})
    assert keep({"category": "socratic"})
    assert not keep({"category": "storytelling"})
    assert not keep({})


def test_logic_and_creativity_draw_from_disjoint_categories():
    """A row must not be able to land in both domains from the same file."""

    logic = {s.member: s for s in builder.BUNDLE_SOURCES if s.domain == "logic"}
    creative = [s for s in builder.BUNDLE_SOURCES if s.domain == "creativity"]

    for source in creative:
        if source.member in logic and source.keep and logic[source.member].keep:
            for category in ("chain_of_thought", "socratic", "debate"):
                assert not source.keep({"category": category}), (
                    f"{category} is claimed by both logic and creativity"
                )


def test_measure_counts_word_types_per_domain():
    rows = [
        {"user": "a", "assistant": "alpha beta", "domain": "x"},
        {"user": "b", "assistant": "gamma", "domain": "y"},
    ]

    measured = builder.measure(rows)

    assert measured["word_types"] == 5  # a, alpha, beta, b, gamma
    assert measured["word_types_by_domain"]["y"] == 2  # b, gamma


# -- domain attribution -----------------------------------------------------


def test_domain_map_attributes_held_out_rows(tmp_path):
    blend = tmp_path / "blend.jsonl"
    blend.write_text(
        "\n".join(
            json.dumps({"user": f"q{i}", "assistant": f"answer {i}", "domain": "maths" if i % 2 else "writing"})
            for i in range(10)
        ),
        encoding="utf-8",
    )

    mapping = domains.load_domain_map(blend)
    grouped = domains.group_by_domain([("q1", "answer 1"), ("q2", "answer 2")], mapping)

    assert grouped["maths"] == [("q1", "answer 1")]
    assert grouped["writing"] == [("q2", "answer 2")]


def test_unattributable_rows_are_labelled_not_dropped(tmp_path):
    """A row the map does not know must still be counted, under 'unknown'."""

    blend = tmp_path / "blend.jsonl"
    blend.write_text(json.dumps({"user": "q", "assistant": "a", "domain": "maths"}), encoding="utf-8")

    mapping = domains.load_domain_map(blend)
    grouped = domains.group_by_domain([("mystery", "row")], mapping)

    assert grouped["unknown"] == [("mystery", "row")]


def test_domain_map_survives_malformed_lines(tmp_path):
    blend = tmp_path / "blend.jsonl"
    blend.write_text(
        json.dumps({"user": "q", "assistant": "a", "domain": "maths"}) + "\n{ broken\n",
        encoding="utf-8",
    )

    mapping = domains.load_domain_map(blend)

    assert mapping[("q", "a")] == "maths"
