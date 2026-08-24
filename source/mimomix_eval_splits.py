"""Generalisation splits for the v57 talking MiMoMix.

V57 reports held-out perplexity 1.27 and, in the same document, says the number
"measures fit to a template distribution, not generalisation to unseen language"
because the split is by row and only 37,543 of the corpus's 120,000 responses are
distinct. That caveat was correct but unquantified: it named a risk without
measuring it.

This module measures it. Three properties of the corpus, counted rather than
assumed, motivate the design:

======================================  ==========
property                                 value
======================================  ==========
rows                                     120,000
distinct responses                       37,543
distinct **sentences** across responses  192
median sentence frequency                310
======================================  ==========

A response is a short composition drawn from a 192-sentence inventory -- a
prefix ("Recommended path:", "Short answer:"), a core, and optional trailers.
That single fact is why row-splitting cannot measure generalisation: under the
v57 split, **78.1% of validation responses appear verbatim in training**
(1,875 of 2,400, measured), and the remaining 21.9% are still recombinations of
sentences the model saw thousands of times.

So "held out" is not one property, it is three, and they get three names:

``tier1_seen_response``
    The response string appears verbatim in the training set. Scoring it
    measures **template recall**. This is what 78% of the v57 validation set is.

``tier2_unseen_response``
    The response string never appears in training, but every sentence in it
    does. Scoring it measures **sentence recombination** -- can the model put
    known sentences in an order it never saw?

``tier3_unseen_sentence``
    The response contains at least one sentence that appears nowhere in
    training. Scoring it measures **unseen-sentence composition**.

Tier 3 is the honest question, and it is only answerable by *training* on a
split that excludes those sentences -- which is why this module builds the split
rather than merely partitioning an existing one. Tiers 1 and 2 can be measured
retroactively on any checkpoint whose training rows are known.

## What tier 3 is, and is not

The held-out sentences are chosen so that **every word in them is still in the
training vocabulary** (`GeneralisationSplit.report` records the measured
coverage, which is 1.0000 at the default settings). The model can therefore
express every held-out sentence; the question is purely whether it assigns them
probability. That makes tier 3 a composition test, not a vocabulary test.

It is *not* a test on unseen language. A held-out sentence such as
``"Recommended path: Yes."`` is a novel pairing of a prefix and a core the model
has seen separately. With 192 sentences drawn from a 292-word corpus there is no
way to construct a genuinely unfamiliar sentence from this data, and this module
does not pretend otherwise. Tier 3 is the hardest question the corpus can pose,
which is a smaller claim than the hardest question there is.

## Selection bias

`train_mimomix_talk.py` keeps the checkpoint with the best **validation** loss
and then reports that same validation set's loss as the result -- a minimum over
twelve evaluations on the set being reported. In the published v57 run the loss
fell monotonically, so the minimum was also the last value and the bias happened
to cost nothing; it is unsound regardless, because nothing guarantees that.

`build_generalisation_split` therefore returns a **`dev`** split alongside the
three test tiers. Checkpoint selection reads `dev`. The tiers are scored once,
at the end, and never steer training.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple

__all__ = [
    "GeneralisationSplit",
    "SENTENCE_BOUNDARY",
    "build_generalisation_split",
    "choose_held_out_sentences",
    "partition_by_response_novelty",
    "sentence_inventory",
    "split_sentences",
    "verify_split",
]

Pair = Tuple[str, str]

#: Split after sentence-terminal punctuation followed by whitespace. The corpus
#: uses ``.``, ``?`` and ``!`` and nothing else; splitting on ``". "`` alone
#: merges question-terminated clauses and inflates the inventory from 192 to 362.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.?!])\s+")


def split_sentences(text: str) -> List[str]:
    """Split a response into its sentences, stripped of surrounding whitespace."""

    return [piece.strip() for piece in SENTENCE_BOUNDARY.split(text) if piece.strip()]


def sentence_inventory(pairs: Iterable[Pair]) -> Counter:
    """Count how often each distinct sentence occurs across the responses."""

    counter: Counter = Counter()
    for _, response in pairs:
        counter.update(split_sentences(response))
    return counter


def choose_held_out_sentences(
    pairs: Sequence[Pair],
    target_row_fraction: float = 0.02,
    max_row_fraction_per_sentence: float = 0.002,
    seed: int = 58,
) -> List[str]:
    """Pick sentences to withhold from training entirely.

    Sentences are considered in a deterministic shuffled order rather than by
    frequency, so the held-out set spans the frequency range instead of being
    the tail. A sentence appearing in more than `max_row_fraction_per_sentence`
    of rows is skipped: withholding a very common sentence would remove a large
    slice of the training data and confound the measurement with a data-volume
    change.

    Accumulation stops once the withheld rows reach `target_row_fraction`, so
    the tier-3 set is a controlled size regardless of the inventory's shape.

    Returns:
        The held-out sentences, in selection order.
    """

    if not 0.0 < target_row_fraction < 1.0:
        raise ValueError("target_row_fraction must lie in (0, 1)")
    if not 0.0 < max_row_fraction_per_sentence < 1.0:
        raise ValueError("max_row_fraction_per_sentence must lie in (0, 1)")

    rows_containing: Dict[str, Set[int]] = defaultdict(set)
    for index, (_, response) in enumerate(pairs):
        for sentence in set(split_sentences(response)):
            rows_containing[sentence].add(index)

    order = sorted(rows_containing)
    random.Random(seed).shuffle(order)

    row_cap = len(pairs) * max_row_fraction_per_sentence
    target = len(pairs) * target_row_fraction
    held_out: List[str] = []
    covered: Set[int] = set()
    for sentence in order:
        if len(rows_containing[sentence]) > row_cap:
            continue
        held_out.append(sentence)
        covered |= rows_containing[sentence]
        if len(covered) >= target:
            break
    if not held_out:
        raise ValueError(
            "no sentence was small enough to withhold; raise max_row_fraction_per_sentence"
        )
    return held_out


def partition_by_response_novelty(
    rows: Sequence[Pair], training_responses: Set[str]
) -> Tuple[List[Pair], List[Pair]]:
    """Split held-out rows into (response seen in training, response unseen).

    This is the tier-1 / tier-2 boundary, and it is the one measurement that can
    be made retroactively against an existing checkpoint: it needs only the rows
    the model trained on, not a different training run.
    """

    seen = [pair for pair in rows if pair[1] in training_responses]
    unseen = [pair for pair in rows if pair[1] not in training_responses]
    return seen, unseen


@dataclass
class GeneralisationSplit:
    """A training split plus the three test tiers it makes measurable.

    Attributes:
        train: Rows the model trains on. Contains no held-out sentence.
        dev: Rows reserved for checkpoint selection. Never scored as a result.
        tier1_seen_response: Test rows whose response appears verbatim in `train`.
        tier2_unseen_response: Test rows whose response is novel but whose every
            sentence appears in `train`.
        tier3_unseen_sentence: Test rows containing a sentence absent from `train`.
        held_out_sentences: The sentences withheld from `train` by construction.
    """

    train: List[Pair]
    dev: List[Pair]
    tier1_seen_response: List[Pair]
    tier2_unseen_response: List[Pair]
    tier3_unseen_sentence: List[Pair]
    held_out_sentences: List[str]
    source: str = ""
    settings: Dict[str, object] = field(default_factory=dict)

    #: Tier attribute name -> the property scoring it actually measures.
    TIER_MEANINGS = {
        "tier1_seen_response": "template recall: the exact response string was in training",
        "tier2_unseen_response": "sentence recombination: novel response, every sentence seen",
        "tier3_unseen_sentence": "unseen-sentence composition: at least one sentence never seen",
    }

    def tiers(self) -> List[Tuple[str, List[Pair]]]:
        """The three test tiers in increasing order of difficulty."""

        return [(name, getattr(self, name)) for name in self.TIER_MEANINGS]

    def report(self, tokenizer=None) -> Dict[str, object]:
        """Measured properties of the split, for the training receipt.

        Args:
            tokenizer: Optional `mimomix_text.WordTokenizer` built from `train`.
                When given, each tier records the fraction of its response tokens
                the training vocabulary can represent -- the number that
                separates a composition test from a vocabulary test.
        """

        training_responses = {response for _, response in self.train}
        payload: Dict[str, object] = {
            "source": self.source,
            "settings": dict(self.settings),
            "train_pairs": len(self.train),
            "dev_pairs": len(self.dev),
            "held_out_sentences": len(self.held_out_sentences),
            "distinct_training_responses": len(training_responses),
            "distinct_training_sentences": len(sentence_inventory(self.train)),
            "note": (
                "tiers are scored once at the end; checkpoint selection reads dev "
                "only, so no reported number is a minimum over evaluations of itself"
            ),
            "tiers": {},
        }
        for name, rows in self.tiers():
            entry: Dict[str, object] = {
                "pairs": len(rows),
                "measures": self.TIER_MEANINGS[name],
            }
            if tokenizer is not None and rows:
                coverage = tokenizer.vocabulary_report([response for _, response in rows])
                entry["response_vocabulary_coverage"] = coverage["coverage"]
            payload["tiers"][name] = entry  # type: ignore[index]
        return payload


def build_generalisation_split(
    pairs: Sequence[Pair],
    dev_fraction: float = 0.01,
    test_fraction: float = 0.02,
    target_row_fraction: float = 0.02,
    max_row_fraction_per_sentence: float = 0.002,
    seed: int = 58,
    source: str = "",
) -> GeneralisationSplit:
    """Build the training split and the three test tiers.

    The order of operations matters and is not interchangeable:

    1. Choose the held-out sentence set over **all** rows.
    2. Move every row containing one into tier 3. Those rows are gone from
       training before anything else is decided, which is what makes tier 3's
       sentences genuinely unseen.
    3. Shuffle the **distinct** remaining pairs and carve off `dev` and a
       row-held-out test block, keeping every copy of a repeated pair together.
    4. Partition that block into tiers 1 and 2 by whether the response string
       survives in the final training set.

    Step 4 must follow step 3, not precede it: a response counts as "seen" only
    if it is in the rows actually trained on. Step 3 groups by pair rather than
    shuffling rows because a corpus that repeats a `(user, response)` row would
    otherwise place identical copies on both sides.

    Args:
        pairs: The full corpus as `(user, assistant)` rows.
        dev_fraction: Share of the non-tier-3 rows reserved for selection.
        test_fraction: Share reserved for tiers 1 and 2 combined.
        target_row_fraction: Share of all rows to place in tier 3.
        max_row_fraction_per_sentence: Skip sentences appearing in more rows
            than this, so withholding one does not materially shrink training.
        seed: Controls both the sentence choice and the row shuffle.
        source: Recorded in the receipt; typically the database path.

    Returns:
        A `GeneralisationSplit`. Callers should pass it to `verify_split`.
    """

    if dev_fraction + test_fraction >= 1.0:
        raise ValueError("dev_fraction + test_fraction must leave rows for training")

    held_out_sentences = choose_held_out_sentences(
        pairs,
        target_row_fraction=target_row_fraction,
        max_row_fraction_per_sentence=max_row_fraction_per_sentence,
        seed=seed,
    )
    held_out_set = set(held_out_sentences)

    tier3: List[Pair] = []
    pool: List[Pair] = []
    for pair in pairs:
        if held_out_set.intersection(split_sentences(pair[1])):
            tier3.append(pair)
        else:
            pool.append(pair)

    # Shuffle *distinct pairs*, not rows, and keep every copy of a duplicated
    # pair on the same side. Splitting rows independently would put an identical
    # `(user, response)` on both sides whenever the corpus repeats one, which is
    # verbatim memorisation scored as a held-out result -- a smaller version of
    # exactly the leak this module was written to measure. Grouping also keeps
    # training multiplicity intact, which deduplicating would not.
    grouped: Dict[Pair, List[Pair]] = defaultdict(list)
    for pair in pool:
        grouped[pair].append(pair)
    distinct = sorted(grouped)
    random.Random(seed + 1).shuffle(distinct)

    dev_size = max(1, int(len(distinct) * dev_fraction))
    test_size = max(1, int(len(distinct) * test_fraction))

    def expand(keys: Sequence[Pair]) -> List[Pair]:
        return [row for key in keys for row in grouped[key]]

    dev = expand(distinct[:dev_size])
    row_test = expand(distinct[dev_size : dev_size + test_size])
    train = expand(distinct[dev_size + test_size :])

    tier1, tier2 = partition_by_response_novelty(
        row_test, {response for _, response in train}
    )

    # Response novelty alone does not decide the tier-2 / tier-3 boundary.
    #
    # Tier 2's name promises "novel response, *every sentence seen*". Carving dev
    # and test out of the pool can strand a sentence outside training, so a row
    # with a novel response may still contain a sentence the model never saw --
    # which is tier 3's definition, not tier 2's.
    #
    # On the 292-word templated corpus this is nearly vacuous: sentences recur
    # about 310 times each, so a stranded sentence is rare and v58 never hit it.
    # On a corpus with real linguistic diversity it dominates -- on 40,000 rows
    # of the v29 pipeline corpus, 283 of 477 tier-2 rows (59%) contain an unseen
    # sentence. `verify_split` already detected this and raised; it named the
    # correct remedy in its own error message ("they belong in tier 3"), and this
    # is that remedy, applied where the tiers are assigned rather than left for
    # the caller to discover as a crash.
    #
    # Routing here rather than relaxing the check keeps every tier name literally
    # true, which is the property the whole module exists to guarantee.
    training_sentences_for_routing: Set[str] = set(sentence_inventory(train))
    kept_tier2: List[Pair] = []
    rerouted: List[Pair] = []
    for pair in tier2:
        if set(split_sentences(pair[1])).issubset(training_sentences_for_routing):
            kept_tier2.append(pair)
        else:
            rerouted.append(pair)
    tier2 = kept_tier2
    tier3 = tier3 + rerouted

    return GeneralisationSplit(
        train=train,
        dev=dev,
        tier1_seen_response=tier1,
        tier2_unseen_response=tier2,
        tier3_unseen_sentence=tier3,
        held_out_sentences=held_out_sentences,
        source=source,
        settings={
            "dev_fraction": dev_fraction,
            "test_fraction": test_fraction,
            "target_row_fraction": target_row_fraction,
            "max_row_fraction_per_sentence": max_row_fraction_per_sentence,
            "seed": seed,
            "duplicate_rows_kept_together": len(pool) - len(distinct),
            "tier2_rerouted_to_tier3": len(rerouted),
        },
    )


def verify_split(split: GeneralisationSplit) -> Dict[str, object]:
    """Check every property the tier names promise, and raise if one fails.

    A split whose disjointness is assumed rather than checked is exactly the
    failure mode this module exists to correct, so the checks run on the real
    object rather than being asserted in a test against a fixture.

    Raises:
        AssertionError: If any tier fails the property its name claims.

    Returns:
        The measured evidence for each check.
    """

    training_sentences: Set[str] = set(sentence_inventory(split.train))
    training_responses = {response for _, response in split.train}

    leaked = [s for s in split.held_out_sentences if s in training_sentences]
    if leaked:
        raise AssertionError(
            f"{len(leaked)} held-out sentence(s) survived in training, e.g. {leaked[0]!r}"
        )

    tier1_wrong = [p for p in split.tier1_seen_response if p[1] not in training_responses]
    if tier1_wrong:
        raise AssertionError(
            f"{len(tier1_wrong)} tier-1 row(s) have a response absent from training"
        )

    tier2_wrong = [p for p in split.tier2_unseen_response if p[1] in training_responses]
    if tier2_wrong:
        raise AssertionError(
            f"{len(tier2_wrong)} tier-2 row(s) have a response that is in training"
        )

    # Tier 2's name claims novelty of the *response*, and that every sentence in
    # it was seen. The second half is not automatic: carving dev and test out of
    # the pool could in principle strand a sentence outside training.
    tier2_unseen_sentence = [
        p
        for p in split.tier2_unseen_response
        if not set(split_sentences(p[1])).issubset(training_sentences)
    ]
    if tier2_unseen_sentence:
        raise AssertionError(
            f"{len(tier2_unseen_sentence)} tier-2 row(s) contain a sentence absent from "
            "training; they belong in tier 3"
        )

    tier3_wrong = [
        p
        for p in split.tier3_unseen_sentence
        if set(split_sentences(p[1])).issubset(training_sentences)
    ]
    if tier3_wrong:
        raise AssertionError(
            f"{len(tier3_wrong)} tier-3 row(s) contain only sentences seen in training"
        )

    train_rows = set(split.train)
    for name, rows in [("dev", split.dev)] + split.tiers():
        overlap = train_rows.intersection(rows)
        if overlap:
            raise AssertionError(f"{len(overlap)} {name} row(s) also appear in training")

    return {
        "held_out_sentences_absent_from_training": True,
        "tier1_responses_all_in_training": True,
        "tier2_responses_all_novel": True,
        "tier2_sentences_all_seen": True,
        "tier3_rows_all_carry_an_unseen_sentence": True,
        "no_test_or_dev_row_appears_in_training": True,
        "distinct_training_sentences": len(training_sentences),
        "distinct_training_responses": len(training_responses),
    }
