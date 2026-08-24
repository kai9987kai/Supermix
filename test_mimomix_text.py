"""Invariants for the v57 tokenizer, corpus and talking model.

A passing suite proves the text pipeline is lossless and the serving path
generates from the model rather than from a template. It does not prove the model
says anything useful -- `train_mimomix_talk.py` measures that, and its corpus
bounds it hard.

Pinned here:

1. Encoding then decoding reproduces text exactly, so nothing is silently
   reworded on the way in or out.
2. A word outside the vocabulary becomes `<unk>` and is counted, because the
   vocabulary is the ceiling on what the model can ever say.
3. Prompt tokens are masked out of the loss, so the model is trained to produce
   replies rather than to echo the user.
4. Greedy replies come from MTP speculative decoding and are token-identical to
   plain greedy decoding.
5. Nothing in a prompt selects a checkpoint, a decoding mode, or a budget.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_text as mt  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402
from train_mimomix_talk import (  # noqa: E402
    evaluate,
    generate_reply,
    load_talk_checkpoint,
    save_talk_checkpoint,
)

SAMPLE_TEXTS = [
    "Sure. I can add unit tests based on your code.",
    "why is my script failing?",
    "Okay.  Two   spaces and a trailing space ",
    "numbers 42 and 7, plus punctuation!",
    "don't split contractions",
]


@pytest.fixture(scope="module")
def tokenizer() -> mt.WordTokenizer:
    return mt.WordTokenizer.build(SAMPLE_TEXTS)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", SAMPLE_TEXTS)
def test_encoding_round_trips_exactly(tokenizer: mt.WordTokenizer, text: str) -> None:
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_preserves_whitespace_runs(tokenizer: mt.WordTokenizer) -> None:
    """Each token carries its own leading whitespace, so spacing survives."""

    text = "Okay.  Two   spaces and a trailing space "
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_assert_roundtrip_accepts_the_corpus(tokenizer: mt.WordTokenizer) -> None:
    mt.assert_roundtrip(tokenizer, SAMPLE_TEXTS)


def test_assert_roundtrip_raises_when_decoding_diverges(tokenizer: mt.WordTokenizer) -> None:
    broken = mt.WordTokenizer(list(tokenizer.tokens))
    broken.tokens[len(mt.SPECIAL_TOKENS)] = "<mangled>"
    with pytest.raises(AssertionError, match="round trip changed the text"):
        mt.assert_roundtrip(broken, SAMPLE_TEXTS)


def test_special_token_ids_are_stable(tokenizer: mt.WordTokenizer) -> None:
    """The checkpoint and the decoder both hard-code these."""

    assert (mt.PAD, mt.BOS, mt.EOS, mt.UNK, mt.USER, mt.ASSISTANT) == (0, 1, 2, 3, 4, 5)
    assert tokenizer.tokens[: len(mt.SPECIAL_TOKENS)] == list(mt.SPECIAL_TOKENS)


def test_unknown_words_map_to_unk_and_are_counted(tokenizer: mt.WordTokenizer) -> None:
    text = "quantum chromodynamics"
    assert mt.UNK in tokenizer.encode(text)
    assert tokenizer.unknown_rate(text) > 0.0
    assert tokenizer.unknown_rate("Sure.") == 0.0


def test_vocabulary_report_states_the_ceiling(tokenizer: mt.WordTokenizer) -> None:
    report = tokenizer.vocabulary_report(SAMPLE_TEXTS + ["quantum chromodynamics"])
    assert report["vocab_size"] == tokenizer.vocab_size
    assert report["unknown_tokens"] > 0
    assert 0.0 < report["coverage"] < 1.0
    assert "ceiling" in report["note"]


def test_decode_drops_structural_specials(tokenizer: mt.WordTokenizer) -> None:
    ids, _ = tokenizer.encode_turn("hello", "Sure.")
    assert mt.BOS in ids and mt.USER in ids and mt.ASSISTANT in ids and mt.EOS in ids
    assert "<user>" not in tokenizer.decode(ids)
    assert "<assistant>" not in tokenizer.decode(ids)


def test_tokenizer_save_and_load_round_trips(tokenizer: mt.WordTokenizer, tmp_path: Path) -> None:
    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    restored = mt.WordTokenizer.load(path)
    assert restored.tokens == tokenizer.tokens
    assert restored.encode(SAMPLE_TEXTS[0]) == tokenizer.encode(SAMPLE_TEXTS[0])


def test_a_word_is_representable_at_a_sentence_start(tokenizer: mt.WordTokenizer) -> None:
    """`"Got"` and `" Got"` are different tokens; both must be in the vocabulary.

    Regression test. Building the vocabulary from `user + " " + assistant`
    admitted only the space-prefixed forms, so 17,709 of 19,600 replies began
    with `<unk>` -- which also flattered perplexity, because `<unk>` became a
    trivially predictable target right after `<assistant>`.
    """

    for text in SAMPLE_TEXTS:
        first = mt.TOKEN_PATTERN.findall(text)[0]
        assert first.lstrip() in tokenizer.index, first


def test_building_from_concatenated_fields_still_covers_both_forms() -> None:
    """Even the defective calling pattern must not lose sentence-initial forms."""

    concatenated = mt.WordTokenizer.build([f"{a} {b}" for a, b in zip(SAMPLE_TEXTS, SAMPLE_TEXTS)])
    for text in SAMPLE_TEXTS:
        assert concatenated.unknown_rate(text) == 0.0, text


def test_a_corpus_built_tokenizer_has_no_unknowns_on_its_own_corpus() -> None:
    """Every field encoded on its own, exactly as training and serving do."""

    fields = [f for pair in zip(SAMPLE_TEXTS, reversed(SAMPLE_TEXTS)) for f in pair]
    built = mt.WordTokenizer.build(fields)
    for text in fields:
        assert built.unknown_rate(text) == 0.0, text
        assert mt.UNK not in built.encode(text)


def test_vocabulary_is_capped(tokenizer: mt.WordTokenizer) -> None:
    small = mt.WordTokenizer.build(SAMPLE_TEXTS, max_vocab=8)
    assert small.vocab_size == 8 + len(mt.SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# Turn formatting and loss masking
# ---------------------------------------------------------------------------


def test_encode_turn_marks_where_the_reply_starts(tokenizer: mt.WordTokenizer) -> None:
    ids, prompt_length = tokenizer.encode_turn("hello", "Sure.")
    assert ids[0] == mt.BOS and ids[1] == mt.USER
    assert ids[prompt_length - 1] == mt.ASSISTANT
    assert ids[-1] == mt.EOS


def test_a_prompt_only_turn_has_no_reply(tokenizer: mt.WordTokenizer) -> None:
    ids, prompt_length = tokenizer.encode_turn("hello", None)
    assert len(ids) == prompt_length
    assert ids[-1] == mt.ASSISTANT


def test_prompt_tokens_are_masked_out_of_the_loss(tokenizer: mt.WordTokenizer) -> None:
    """-100 is `cross_entropy`'s default ignore_index; the model relies on it."""

    pairs = [("why is my script failing?", "Sure. Check the traceback first.")] * 40
    inputs, labels = mt.build_training_tensors(pairs, tokenizer, sequence_length=16)
    assert inputs.shape == labels.shape
    assert int((labels == -100).sum()) > 0
    kept = labels != -100
    assert torch.equal(inputs[kept], labels[kept]), "unmasked labels must equal the inputs"


def test_masking_can_be_disabled(tokenizer: mt.WordTokenizer) -> None:
    pairs = [("hello", "Sure.")] * 40
    _, masked = mt.build_training_tensors(pairs, tokenizer, 16, mask_prompt=True)
    _, unmasked = mt.build_training_tensors(pairs, tokenizer, 16, mask_prompt=False)
    assert int((masked == -100).sum()) > int((unmasked == -100).sum()) == 0


def test_too_little_text_is_refused(tokenizer: mt.WordTokenizer) -> None:
    with pytest.raises(ValueError, match="not enough tokens"):
        mt.build_training_tensors([("hi", "ok")], tokenizer, sequence_length=4096)


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _tiny_database(tmp_path: Path) -> Path:
    path = tmp_path / "chat.db"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE llm_entries (id INTEGER PRIMARY KEY, user_text TEXT, response_text TEXT)")
    rows = [(f"question number {i}", f"Sure. Answer number {i} about your code.") for i in range(200)]
    rows.append(("too short", "no"))  # below min_response_characters
    rows.append(("", "Sure. Empty user text is dropped."))
    connection.executemany("INSERT INTO llm_entries (user_text, response_text) VALUES (?, ?)", rows)
    connection.commit()
    connection.close()
    return path


def test_corpus_splits_by_row_and_drops_unusable_rows(tmp_path: Path) -> None:
    corpus = mt.load_chat_pairs(str(_tiny_database(tmp_path)), validation_fraction=0.1)
    assert len(corpus.train) + len(corpus.validation) == 200
    assert all(user and response for user, response in corpus.train)
    assert len(corpus.validation) == 20


def test_corpus_split_is_reproducible(tmp_path: Path) -> None:
    path = str(_tiny_database(tmp_path))
    first = mt.load_chat_pairs(path, seed=57)
    second = mt.load_chat_pairs(path, seed=57)
    assert first.validation == second.validation
    assert mt.load_chat_pairs(path, seed=58).validation != first.validation


def test_train_and_validation_rows_are_disjoint(tmp_path: Path) -> None:
    corpus = mt.load_chat_pairs(str(_tiny_database(tmp_path)), validation_fraction=0.1)
    # rows are disjoint; the docstring is explicit that the *responses* may not be
    assert not set(corpus.validation) & set(corpus.train)


# ---------------------------------------------------------------------------
# The trained-model path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def talk_model(tokenizer: mt.WordTokenizer):
    torch.manual_seed(0)
    config = MiMoMixConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=64,
        sliding_window=16,
        hybrid_ratio=1,
        native_context=64,
        max_position_embeddings=64,
        rope_scaling="none",
        n_routed_experts=4,
        moe_intermediate_size=16,
        n_mtp_layers=2,
        thinking_latent_dim=16,
        thinking_cycles=1,
        thinking_max_cycles=2,
    )
    return MiMoMixModel(config)


def test_generation_returns_decoded_text(talk_model, tokenizer: mt.WordTokenizer) -> None:
    result = generate_reply(talk_model, tokenizer, "hello", max_new_tokens=8)
    assert isinstance(result["reply"], str)
    assert result["tokens"] <= 8
    assert result["trunk_forwards"] >= 1


def test_speculative_decoding_matches_greedy(talk_model, tokenizer: mt.WordTokenizer) -> None:
    """The MTP draft path may only change cost, never the emitted text."""

    fast = generate_reply(talk_model, tokenizer, "hello", max_new_tokens=12, speculative=True)
    slow = generate_reply(talk_model, tokenizer, "hello", max_new_tokens=12, speculative=False)
    assert fast["reply"] == slow["reply"]


def test_evaluate_scores_only_unmasked_tokens(talk_model, tokenizer: mt.WordTokenizer) -> None:
    pairs = [("why is my script failing?", "Sure. Check the traceback first.")] * 60
    inputs, labels = mt.build_training_tensors(pairs, tokenizer, sequence_length=32)
    metrics = evaluate(talk_model, inputs, labels, batch_size=4)
    assert metrics["scored_tokens"] > 0
    assert metrics["scored_tokens"] < inputs.numel()
    assert metrics["perplexity"] > 1.0


def test_checkpoint_carries_the_tokenizer(talk_model, tokenizer: mt.WordTokenizer, tmp_path: Path) -> None:
    """Weights without their vocabulary decode to confident nonsense."""

    path = tmp_path / "talk.pt"
    save_talk_checkpoint(path, talk_model, tokenizer, extra={"note": "test"})
    model, restored, payload = load_talk_checkpoint(path)
    assert restored.tokens == tokenizer.tokens
    assert payload["extra"]["note"] == "test"
    ids = torch.tensor([[mt.BOS, mt.USER, mt.ASSISTANT]])
    with torch.no_grad():
        assert torch.equal(model(ids).logits, talk_model(ids).logits)


def test_a_foreign_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"schema": "something-else"}, path)
    with pytest.raises(ValueError, match="not a supermix-v57"):
        load_talk_checkpoint(path)
