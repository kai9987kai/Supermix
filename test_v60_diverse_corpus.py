"""Tests for the v60 diverse-corpus path: JSONL loading and tier routing.

V58's ladder could only be built on a corpus whose sentences repeat hundreds of
times. On anything with real linguistic diversity `verify_split` raised, because
a row with a novel response usually also contains a sentence training never saw
-- which is tier 3's definition, not tier 2's.

The routing fix is only correct if it does two things at once, so both are
tested here: it must make a diverse corpus verifiable, **and** it must be a
no-op on a templated one, or it would silently rewrite v58's published split.
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

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402


# -- corpora ---------------------------------------------------------------


def _templated_corpus(rows: int = 4000):
    """A v58-shaped corpus: a small sentence inventory with a skewed tail.

    Mirrors the real structure rather than a caricature of it. Most sentences
    recur hundreds of times, but a handful appear in only a few rows -- which is
    what `choose_held_out_sentences` needs, since it refuses to withhold any
    sentence common enough that removing it would materially shrink training.
    """

    common = [f"Common sentence {i} explains a thing." for i in range(40)]
    rare = [f"Rare sentence {i} appears seldom." for i in range(20)]

    pairs = []
    for i in range(rows):
        a = common[i % len(common)]
        b = common[(i * 7 + 3) % len(common)]
        pairs.append((f"question {i % 50}", f"{a} {b}"))

    # Each rare sentence in exactly four rows: under the 0.002 row cap, so it is
    # eligible to be withheld, and enough of them to fill tier 3.
    for index, sentence in enumerate(rare):
        for repeat in range(4):
            partner = common[(index + repeat) % len(common)]
            pairs.append((f"rare question {index}-{repeat}", f"{sentence} {partner}"))
    return pairs


def _diverse_corpus(rows: int = 4000):
    """A corpus where almost every sentence is unique, as real text is."""

    return [
        (f"question about topic {i}", f"Unique statement {i} about subject {i * 3}. "
                                      f"Follow up detail {i * 5} clarifies it.")
        for i in range(rows)
    ]


# -- JSONL loader ----------------------------------------------------------


def _write_jsonl(path: Path, records) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    return path


def test_jsonl_loader_reads_user_assistant_pairs(tmp_path):
    path = _write_jsonl(
        tmp_path / "c.jsonl",
        [{"user": f"q{i}", "assistant": f"a long enough answer {i}"} for i in range(50)],
    )

    corpus = text_utils.load_chat_pairs_jsonl(str(path))

    assert len(corpus.train) + len(corpus.validation) == 50
    assert corpus.source == str(path)


def test_jsonl_loader_skips_malformed_lines_without_failing(tmp_path):
    """Pipeline output; one bad row must not cost a training run."""

    path = tmp_path / "c.jsonl"
    path.write_text(
        json.dumps({"user": "q", "assistant": "a good long answer"})
        + "\n{ not json at all\n"
        + json.dumps({"user": "q2", "assistant": "another good long answer"})
        + "\n\n"
        + json.dumps(["not", "a", "dict"])
        + "\n",
        encoding="utf-8",
    )

    corpus = text_utils.load_chat_pairs_jsonl(str(path), validation_fraction=0.0)

    assert len(corpus.train) + len(corpus.validation) == 2


def test_jsonl_loader_drops_short_and_empty_rows(tmp_path):
    path = _write_jsonl(
        tmp_path / "c.jsonl",
        [
            {"user": "q", "assistant": "short"},          # below min_response_characters
            {"user": "", "assistant": "a good long answer"},  # no prompt
            {"user": "q", "assistant": "a good long answer"},
        ],
    )

    corpus = text_utils.load_chat_pairs_jsonl(str(path), validation_fraction=0.0)

    assert len(corpus.train) + len(corpus.validation) == 1


def test_jsonl_loader_respects_limit(tmp_path):
    path = _write_jsonl(
        tmp_path / "c.jsonl",
        [{"user": f"q{i}", "assistant": f"a long enough answer {i}"} for i in range(100)],
    )

    corpus = text_utils.load_chat_pairs_jsonl(str(path), limit=20)

    assert len(corpus.train) + len(corpus.validation) == 20


def test_jsonl_loader_raises_on_a_corpus_with_nothing_usable(tmp_path):
    path = _write_jsonl(tmp_path / "c.jsonl", [{"user": "", "assistant": ""}])

    with pytest.raises(ValueError, match="no usable"):
        text_utils.load_chat_pairs_jsonl(str(path))


# -- tier routing ----------------------------------------------------------


def test_diverse_corpus_now_verifies():
    """The blocker: before routing, verify_split raised on this shape."""

    split = splits.build_generalisation_split(_diverse_corpus(), seed=60)

    splits.verify_split(split)  # must not raise
    assert split.settings["tier2_rerouted_to_tier3"] > 0


def test_templated_corpus_reroutes_nothing():
    """The safety property: v58's published split must not move."""

    split = splits.build_generalisation_split(_templated_corpus(), seed=58)

    splits.verify_split(split)
    assert split.settings["tier2_rerouted_to_tier3"] == 0


def test_every_tier2_row_has_only_seen_sentences_after_routing():
    """Tier 2's name promises this; routing is what makes it literally true."""

    split = splits.build_generalisation_split(_diverse_corpus(), seed=60)
    training_sentences = set(splits.sentence_inventory(split.train))

    for _, response in split.tier2_unseen_response:
        assert set(splits.split_sentences(response)).issubset(training_sentences)


def test_rerouted_rows_land_in_tier3_and_are_not_lost():
    """Routing must move rows, not drop them."""

    pairs = _diverse_corpus()
    split = splits.build_generalisation_split(pairs, seed=60)

    total = (
        len(split.train)
        + len(split.dev)
        + len(split.tier1_seen_response)
        + len(split.tier2_unseen_response)
        + len(split.tier3_unseen_sentence)
    )
    assert total == len(pairs)


def test_rerouted_rows_satisfy_tier3s_definition():
    split = splits.build_generalisation_split(_diverse_corpus(), seed=60)
    training_sentences = set(splits.sentence_inventory(split.train))

    assert split.settings["tier2_rerouted_to_tier3"] > 0
    for _, response in split.tier3_unseen_sentence:
        assert not set(splits.split_sentences(response)).issubset(training_sentences)


def test_diverse_corpus_has_the_diversity_the_tiers_need():
    """Guards the premise: this corpus is not templated.

    If a future edit made `_diverse_corpus` repetitive, the routing tests above
    would pass vacuously by rerouting nothing.
    """

    pairs = _diverse_corpus()
    inventory = splits.sentence_inventory(pairs)
    mean_repeats = sum(inventory.values()) / max(1, len(inventory))

    assert mean_repeats < 3.0, "corpus is too repetitive to exercise tier routing"
