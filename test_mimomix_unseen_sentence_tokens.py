"""Tests for the token-level unseen-sentence measurement.

The claim this module makes is a *controlled* one: the withheld-sentence tokens
and the seen-sentence tokens come from the same rows, so a difference between
them is a property of the sentences and not of the rows. That control is only
real if the two sets genuinely partition the response -- disjoint, and together
covering every response token. If they overlapped, tokens would be scored twice
and the comparison would be between two overlapping populations; if they left a
gap, the "seen" set would silently exclude hard tokens.

So these tests pin the partition itself rather than any number. They need no
model and no database.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import eval_mimomix_unseen_sentences as unseen  # noqa: E402
import mimomix_text as text_utils  # noqa: E402

RESPONSE = "Sure. Recommended path: Yes. Send the next issue when ready."
HELD_OUT = {"Recommended path: Yes."}


@pytest.fixture(scope="module")
def tokenizer():
    return text_utils.WordTokenizer.build(
        [RESPONSE, "hello there", "Check the traceback first. Sure."]
    )


def scored_text(pair, tokenizer, held_out, keep) -> str:
    ids, labels = unseen.masked_turn(pair, tokenizer, held_out, keep)
    return "".join(
        tokenizer.tokens[i] for i, label in zip(ids, labels) if label != -100
    )


# -- locating the sentences --------------------------------------------------


def test_spans_cover_exactly_the_withheld_sentence():
    spans = unseen.withheld_character_spans(RESPONSE, HELD_OUT)
    assert [RESPONSE[a:b] for a, b in spans] == ["Recommended path: Yes."]


def test_no_span_when_nothing_is_withheld():
    assert unseen.withheld_character_spans(RESPONSE, set()) == []


def test_a_repeated_withheld_sentence_is_found_every_time():
    response = "Yes. Sure. Yes."
    spans = unseen.withheld_character_spans(response, {"Yes."})
    assert len(spans) == 2
    assert [response[a:b] for a, b in spans] == ["Yes.", "Yes."]


def test_a_withheld_sentence_does_not_claim_a_longer_sentences_characters():
    """`"Sure."` is a prefix of `"Sure. Really."` only as a substring, never as a
    sentence. Scanning the response's own sentence split rather than doing a
    substring search is what keeps them apart."""

    response = "Sure. Really sure."
    spans = unseen.withheld_character_spans(response, {"Sure."})
    assert [response[a:b] for a, b in spans] == ["Sure."]


# -- the partition ------------------------------------------------------------


def test_the_two_keeps_partition_the_response(tokenizer):
    pair = ("hi", RESPONSE)
    ids, unseen_labels = unseen.masked_turn(pair, tokenizer, HELD_OUT, "unseen")
    _, seen_labels = unseen.masked_turn(pair, tokenizer, HELD_OUT, "seen")

    scored_unseen = {i for i, label in enumerate(unseen_labels) if label != -100}
    scored_seen = {i for i, label in enumerate(seen_labels) if label != -100}

    assert scored_unseen and scored_seen
    assert not scored_unseen & scored_seen, "a token was scored in both sets"

    _, prompt_length = tokenizer.encode_turn(*pair)
    response_pieces = len(text_utils.TOKEN_PATTERN.findall(RESPONSE))
    every_response_token = set(range(prompt_length, prompt_length + response_pieces))
    assert scored_unseen | scored_seen == every_response_token, "a token was in neither set"


def test_the_withheld_sentences_text_is_what_gets_scored(tokenizer):
    assert scored_text(("hi", RESPONSE), tokenizer, HELD_OUT, "unseen").strip() == (
        "Recommended path: Yes."
    )


def test_the_control_set_is_the_rest_of_the_same_response(tokenizer):
    assert scored_text(("hi", RESPONSE), tokenizer, HELD_OUT, "seen").strip() == (
        "Sure. Send the next issue when ready."
    )


def test_a_token_following_a_span_is_not_captured_by_it(tokenizer):
    """Tokens carry their own leading whitespace, so the token after a withheld
    sentence *starts* inside that sentence's span. Assigning by the token's
    body rather than its raw start is what stops it being misfiled."""

    scored = scored_text(("hi", RESPONSE), tokenizer, HELD_OUT, "unseen")
    assert "Send" not in scored


def test_prompt_tokens_are_never_scored_in_either_set(tokenizer):
    pair = ("why is my script failing", RESPONSE)
    _, prompt_length = tokenizer.encode_turn(*pair)
    for keep in ("unseen", "seen"):
        _, labels = unseen.masked_turn(pair, tokenizer, HELD_OUT, keep)
        assert all(label == -100 for label in labels[:prompt_length])


def test_a_response_with_no_withheld_sentence_scores_nothing_as_unseen(tokenizer):
    pair = ("hi", "Sure. Send the next issue when ready.")
    _, labels = unseen.masked_turn(pair, tokenizer, HELD_OUT, "unseen")
    assert all(label == -100 for label in labels)


# -- packing ------------------------------------------------------------------


def test_build_tensors_labels_only_the_requested_tokens(tokenizer):
    pairs = [("hi", RESPONSE)] * 40
    inputs, labels, counted = unseen.build_tensors(
        pairs, tokenizer, HELD_OUT, "unseen", sequence_length=32
    )
    assert inputs.shape == labels.shape
    assert counted > 0
    assert int((labels != -100).sum()) == counted
    # Every scored label must equal its own input, exactly as in training.
    mask = labels != -100
    assert bool((labels[mask] == inputs[mask]).all())


def test_the_two_token_sets_stay_disjoint_after_packing(tokenizer):
    pairs = [("hi", RESPONSE)] * 40
    _, unseen_labels, _ = unseen.build_tensors(
        pairs, tokenizer, HELD_OUT, "unseen", sequence_length=32
    )
    _, seen_labels, _ = unseen.build_tensors(
        pairs, tokenizer, HELD_OUT, "seen", sequence_length=32
    )
    both = (unseen_labels != -100) & (seen_labels != -100)
    assert not bool(both.any())


def test_build_tensors_refuses_a_corpus_too_small_for_one_block(tokenizer):
    with pytest.raises(ValueError, match="not enough tokens"):
        unseen.build_tensors(
            [("hi", "Sure.")], tokenizer, HELD_OUT, "unseen", sequence_length=512
        )
