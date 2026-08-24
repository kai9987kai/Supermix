"""Tests for the v58 generalisation ladder.

The module under test exists to stop a held-out set from quietly measuring
recall, so these tests are written to fail if a tier stops meaning what its name
says. Two of them deliberately corrupt a split and require `verify_split` to
raise: a verifier that never rejects anything is the same failure as no verifier,
and it would not otherwise be caught, because every honestly-built split passes.
"""

from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402

SENTENCES = [
    "Sure.",
    "Yes.",
    "Got it.",
    "Understood.",
    "Check the traceback first.",
    "Run the script from the project directory.",
    "Verify your virtual environment.",
    "Share expected behavior and edge cases.",
    "What can I help you with today?",
    "Send the next issue when ready.",
    "Use parameterized queries.",
    "Profile request time first.",
]

USERS = [
    "can you help me with tests",
    "why is my script failing",
    "how do I deploy this",
    "what is your name",
    "my import is broken",
    "the query is slow",
]


def synthetic_corpus(rows: int = 900, seed: int = 3) -> list:
    """A corpus with the shape that motivates the module: many responses, few sentences.

    Every response is a composition of two or three sentences drawn from a small
    inventory, which is what the real corpus is -- 37,543 distinct responses over
    192 distinct sentences.

    Sentences are drawn with **skewed** weights rather than uniformly, because
    the real inventory is skewed: its most common sentence appears 27,695 times
    and its rarest 2. A uniform inventory would make every sentence equally
    eligible for withholding and would silently stop
    `max_row_fraction_per_sentence` from ever binding, so the test guarding that
    cap would pass without exercising it.
    """

    rng = random.Random(seed)
    weights = [2 ** (len(SENTENCES) - index) for index in range(len(SENTENCES))]
    pairs = []
    for _ in range(rows):
        count = rng.choice((2, 2, 3))
        chosen: list = []
        while len(chosen) < count:
            pick = rng.choices(SENTENCES, weights=weights, k=1)[0]
            if pick not in chosen:
                chosen.append(pick)
        pairs.append((rng.choice(USERS), " ".join(chosen)))
    return pairs


@pytest.fixture(scope="module")
def corpus():
    return synthetic_corpus()


@pytest.fixture(scope="module")
def split(corpus):
    return splits.build_generalisation_split(
        corpus,
        dev_fraction=0.08,
        test_fraction=0.12,
        target_row_fraction=0.10,
        max_row_fraction_per_sentence=0.30,
        seed=11,
        source="synthetic",
    )


# -- sentence splitting -----------------------------------------------------


def test_sentence_splitting_uses_every_terminal_punctuation():
    """`?` and `!` end sentences too.

    Splitting on ``". "`` alone merges a question with the clause after it. On
    the real corpus that inflates the inventory from 192 distinct sentences to
    362, most of the extras being duplicates carrying a question as a prefix --
    which would make the held-out set look far more varied than it is.
    """

    text = "What can I help you with today? Check the traceback first. Go!"
    assert splits.split_sentences(text) == [
        "What can I help you with today?",
        "Check the traceback first.",
        "Go!",
    ]


def test_sentence_splitting_ignores_empty_pieces():
    assert splits.split_sentences("   ") == []
    assert splits.split_sentences("Yes.    Sure.") == ["Yes.", "Sure."]


def test_sentence_inventory_counts_repeats_across_rows():
    pairs = [("u", "Yes. Sure."), ("u", "Yes. Got it.")]
    inventory = splits.sentence_inventory(pairs)
    assert inventory["Yes."] == 2
    assert inventory["Sure."] == 1


# -- the tier properties ----------------------------------------------------


def test_held_out_sentences_never_appear_in_training(split):
    training = set(splits.sentence_inventory(split.train))
    assert split.held_out_sentences
    for sentence in split.held_out_sentences:
        assert sentence not in training


def test_tier1_responses_all_appear_in_training(split):
    training_responses = {response for _, response in split.train}
    assert split.tier1_seen_response
    for _, response in split.tier1_seen_response:
        assert response in training_responses


def test_tier2_responses_are_novel_but_their_sentences_are_not(split):
    """Tier 2's whole point: a response string the model never saw, built only
    from sentences it saw. If either half fails, the tier is measuring
    something other than recombination."""

    training_responses = {response for _, response in split.train}
    training_sentences = set(splits.sentence_inventory(split.train))
    assert split.tier2_unseen_response
    for _, response in split.tier2_unseen_response:
        assert response not in training_responses
        assert set(splits.split_sentences(response)).issubset(training_sentences)


def test_tier3_rows_all_carry_a_sentence_absent_from_training(split):
    training_sentences = set(splits.sentence_inventory(split.train))
    assert split.tier3_unseen_sentence
    for _, response in split.tier3_unseen_sentence:
        assert not set(splits.split_sentences(response)).issubset(training_sentences)


def test_no_dev_or_test_row_is_also_a_training_row(split):
    training = set(split.train)
    for name, rows in [("dev", split.dev)] + split.tiers():
        assert not training.intersection(rows), name


def test_the_three_tiers_do_not_overlap(split):
    for (name_a, rows_a), (name_b, rows_b) in itertools.combinations(split.tiers(), 2):
        assert not set(rows_a).intersection(rows_b), f"{name_a} overlaps {name_b}"


def test_every_row_is_accounted_for(corpus, split):
    placed = len(split.train) + len(split.dev) + sum(len(rows) for _, rows in split.tiers())
    assert placed == len(corpus)


# -- the verifier must actually reject --------------------------------------


def test_verify_split_accepts_an_honestly_built_split(split):
    evidence = splits.verify_split(split)
    assert evidence["tier3_rows_all_carry_an_unseen_sentence"] is True


def test_verify_split_rejects_a_held_out_sentence_that_leaked_into_training(split):
    """The failure this whole module exists to prevent."""

    leaked = splits.GeneralisationSplit(
        train=split.train + [("u", split.held_out_sentences[0])],
        dev=split.dev,
        tier1_seen_response=split.tier1_seen_response,
        tier2_unseen_response=split.tier2_unseen_response,
        tier3_unseen_sentence=split.tier3_unseen_sentence,
        held_out_sentences=split.held_out_sentences,
    )
    with pytest.raises(AssertionError, match="survived in training"):
        splits.verify_split(leaked)


def test_verify_split_rejects_a_tier2_row_whose_response_is_in_training(split):
    mislabelled = splits.GeneralisationSplit(
        train=split.train,
        dev=split.dev,
        tier1_seen_response=split.tier1_seen_response,
        tier2_unseen_response=split.tier2_unseen_response + [split.train[0]],
        tier3_unseen_sentence=split.tier3_unseen_sentence,
        held_out_sentences=split.held_out_sentences,
    )
    with pytest.raises(AssertionError):
        splits.verify_split(mislabelled)


def test_verify_split_rejects_a_tier3_row_with_no_unseen_sentence(split):
    """A tier-3 set that is really a tier-2 set is the silent version of the
    v57 problem: a number reported under a harder name than it earned."""

    mislabelled = splits.GeneralisationSplit(
        train=split.train,
        dev=split.dev,
        tier1_seen_response=split.tier1_seen_response,
        tier2_unseen_response=split.tier2_unseen_response,
        tier3_unseen_sentence=split.tier3_unseen_sentence + [("u", split.train[0][1])],
        held_out_sentences=split.held_out_sentences,
    )
    with pytest.raises(AssertionError, match="only sentences seen in training"):
        splits.verify_split(mislabelled)


# -- selection ---------------------------------------------------------------


def test_choose_held_out_sentences_skips_sentences_that_are_too_common(corpus):
    """Withholding a very common sentence would shrink training enough to
    confound the measurement with a data-volume change."""

    cap = 0.25
    chosen = splits.choose_held_out_sentences(
        corpus, target_row_fraction=0.05, max_row_fraction_per_sentence=cap, seed=11
    )
    assert chosen
    rows_containing = splits.sentence_inventory(corpus)
    for sentence in chosen:
        assert rows_containing[sentence] <= len(corpus) * cap

    # The cap has to bind on something, or the test proves nothing.
    assert any(count > len(corpus) * cap for count in rows_containing.values())
    assert not any(rows_containing[sentence] > len(corpus) * cap for sentence in chosen)


def test_choose_held_out_sentences_raises_when_nothing_is_small_enough(corpus):
    with pytest.raises(ValueError, match="no sentence was small enough"):
        splits.choose_held_out_sentences(
            corpus, target_row_fraction=0.05, max_row_fraction_per_sentence=1e-6, seed=11
        )


def test_choose_held_out_sentences_rejects_out_of_range_fractions(corpus):
    with pytest.raises(ValueError):
        splits.choose_held_out_sentences(corpus, target_row_fraction=0.0)
    with pytest.raises(ValueError):
        splits.choose_held_out_sentences(corpus, max_row_fraction_per_sentence=1.5)


def test_the_split_is_deterministic_for_a_seed(corpus):
    a = splits.build_generalisation_split(corpus, seed=7, max_row_fraction_per_sentence=0.30)
    b = splits.build_generalisation_split(corpus, seed=7, max_row_fraction_per_sentence=0.30)
    assert a.held_out_sentences == b.held_out_sentences
    assert a.train == b.train
    assert a.tier3_unseen_sentence == b.tier3_unseen_sentence


def test_a_different_seed_chooses_different_sentences(corpus):
    a = splits.build_generalisation_split(corpus, seed=7, max_row_fraction_per_sentence=0.30)
    b = splits.build_generalisation_split(corpus, seed=99, max_row_fraction_per_sentence=0.30)
    assert a.held_out_sentences != b.held_out_sentences


def test_build_rejects_a_split_that_leaves_no_training_rows(corpus):
    with pytest.raises(ValueError, match="leave rows for training"):
        splits.build_generalisation_split(corpus, dev_fraction=0.6, test_fraction=0.5)


# -- the receipt -------------------------------------------------------------


def test_report_names_what_each_tier_measures(split):
    report = split.report()
    assert set(report["tiers"]) == set(splits.GeneralisationSplit.TIER_MEANINGS)
    assert "template recall" in report["tiers"]["tier1_seen_response"]["measures"]
    assert "recombination" in report["tiers"]["tier2_unseen_response"]["measures"]
    assert "unseen-sentence" in report["tiers"]["tier3_unseen_sentence"]["measures"]


def test_report_records_vocabulary_coverage_per_tier(split):
    """If the held-out sentences used words the training vocabulary lacks, tier 3
    would measure the tokenizer's ceiling rather than the model's composition, so
    the receipt has to carry the number that distinguishes the two.

    The value itself is a property of the corpus, not of the module: this
    synthetic inventory puts some words only in withheld sentences, so coverage
    here is below 1. On the real corpus it is exactly 1, which
    `test_the_real_corpus_gives_tier3_full_vocabulary_coverage` pins.
    """

    tokenizer = text_utils.WordTokenizer.build(
        (field for pair in split.train for field in pair)
    )
    report = split.report(tokenizer)
    for name, _ in split.tiers():
        coverage = report["tiers"][name]["response_vocabulary_coverage"]
        assert 0.0 <= coverage <= 1.0


def test_report_is_json_serialisable(split):
    json.dumps(split.report())


# -- against the real corpus, when it is present -----------------------------

DATABASE = ROOT / "databases" / "llm_chat.db"


@pytest.mark.skipif(not DATABASE.exists(), reason="local chat database not present")
def test_the_real_corpus_produces_a_verified_split():
    corpus = text_utils.load_chat_pairs(str(DATABASE), validation_fraction=0.02, seed=57)
    pairs = list(corpus.train) + list(corpus.validation)
    split = splits.build_generalisation_split(pairs, source=str(DATABASE))
    evidence = splits.verify_split(split)
    assert evidence["tier3_rows_all_carry_an_unseen_sentence"] is True
    for _, rows in split.tiers():
        assert rows, "every tier must be non-empty on the real corpus"


@pytest.mark.skipif(not DATABASE.exists(), reason="local chat database not present")
def test_the_real_corpus_has_the_shape_the_module_documents():
    """The design rests on counted properties. If the corpus is replaced with one
    that does not have them, the tier names stop being meaningful and this fails
    rather than silently reporting a number under the wrong name."""

    corpus = text_utils.load_chat_pairs(str(DATABASE), validation_fraction=0.02, seed=57)
    pairs = list(corpus.train) + list(corpus.validation)
    responses = {response for _, response in pairs}
    inventory = splits.sentence_inventory(pairs)
    assert len(responses) > 10 * len(inventory), (
        "responses must vastly outnumber sentences for the recombination tier to exist; "
        f"got {len(responses)} responses over {len(inventory)} sentences"
    )


@pytest.mark.skipif(not DATABASE.exists(), reason="local chat database not present")
def test_the_real_corpus_gives_tier3_full_vocabulary_coverage():
    """Tier 3 must be a composition test, not a vocabulary test.

    Every word of every withheld sentence has to remain expressible from the
    training vocabulary; otherwise a high tier-3 loss would only be saying the
    model cannot emit a token it never had.
    """

    corpus = text_utils.load_chat_pairs(str(DATABASE), validation_fraction=0.02, seed=57)
    pairs = list(corpus.train) + list(corpus.validation)
    split = splits.build_generalisation_split(pairs, source=str(DATABASE))
    tokenizer = text_utils.WordTokenizer.build(
        (field for pair in split.train for field in pair)
    )
    report = split.report(tokenizer)
    assert report["tiers"]["tier3_unseen_sentence"]["response_vocabulary_coverage"] == 1.0


@pytest.mark.skipif(not DATABASE.exists(), reason="local chat database not present")
def test_the_v57_row_split_leaks_most_of_its_responses():
    """The measurement that motivates the module, pinned as a regression test.

    If a future change to `load_chat_pairs` made the row split honest, this
    would fail and the v58 documents would need rewriting -- which is the
    correct outcome, not a nuisance.
    """

    corpus = text_utils.load_chat_pairs(str(DATABASE), validation_fraction=0.02, seed=57)
    training_responses = {response for _, response in corpus.train}
    leaked = sum(1 for _, response in corpus.validation if response in training_responses)
    assert leaked / len(corpus.validation) > 0.7
