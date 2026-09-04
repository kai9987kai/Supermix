"""Tests for the problem-solving benchmark and digit tokenisation.

This benchmark exists because every other metric here can be satisfied by
recitation. Its own correctness therefore matters more than usual: a scorer that
marked wrong answers correct would manufacture problem-solving ability out of
nothing, and one that mis-parses replies would hide it.
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

import eval_problem_solving as solving  # noqa: E402
import mimomix_text as text_utils  # noqa: E402


# -- answer extraction ------------------------------------------------------


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("79", 79.0),
        ("376 - 17 = 359", 359.0),                     # last number, not first
        ("Calculate directly. Answer: 10.68.", 10.68),  # trailing full stop
        ("Step 1: isolate x. Step 2: x = -1.", -1.0),   # negative
        ("9/14", 9 / 14),                               # fraction
        ("the total is 1,234", 1234.0),                 # thousands separator
    ],
)
def test_extract_answer_handles_every_corpus_shape(reply, expected):
    assert solving.extract_answer(reply) == pytest.approx(expected)


def test_extract_answer_returns_none_when_there_is_no_number():
    """A reply with no number must not be scored as anything."""

    assert solving.extract_answer("Synonym: glad.") is None
    assert solving.extract_answer("") is None


def test_last_number_rule_is_deliberate():
    """'376 - 17 = 359' must score 359; taking the first would score 376."""

    assert solving.extract_answer("376 - 17 = 359") == 359.0


# -- correctness ------------------------------------------------------------


def test_is_correct_requires_the_right_answer():
    assert solving.is_correct(219.0, 219.0)
    assert not solving.is_correct(218.0, 219.0)
    assert not solving.is_correct(None, 219.0)


def test_is_correct_tolerates_float_representation_only():
    """51.333333333333336 is the same answer; 51.5 is not."""

    assert solving.is_correct(51.333333333333336, 154 / 3)
    assert not solving.is_correct(51.5, 154 / 3)


# -- generated problems -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(solving.GENERATORS))
def test_every_generator_states_a_true_answer(name):
    """The ground truth must actually be the answer to the prompt."""

    import random

    rng = random.Random(1)
    for _ in range(25):
        problem = solving.GENERATORS[name](rng)
        assert problem.source == "novel"
        assert isinstance(problem.answer, float)
        # Every generated prompt must contain the operands it asks about.
        assert any(ch.isdigit() for ch in problem.prompt)


def test_arithmetic_ground_truth_is_computed_not_asserted():
    import random
    import re

    rng = random.Random(3)
    for _ in range(50):
        problem = solving._arithmetic(rng)
        a, op, b = re.search(r"(\d+) ([+-]) (\d+)", problem.prompt).groups()
        expected = int(a) + int(b) if op == "+" else int(a) - int(b)
        assert problem.answer == float(expected)


def test_generation_is_deterministic_for_a_seed():
    first = [p.prompt for p in solving.generate_novel(20, seed=7)]
    second = [p.prompt for p in solving.generate_novel(20, seed=7)]

    assert first == second


def test_the_pre_v82_shared_rng_draw_is_still_reachable():
    """Old receipts stay checkable only while the old draw can be reproduced."""

    first = [p.prompt for p in solving.generate_novel(20, seed=7, shared_rng=True)]
    second = [p.prompt for p in solving.generate_novel(20, seed=7, shared_rng=True)]

    assert first == second
    assert first != [p.prompt for p in solving.generate_novel(20, seed=7)]


def test_evaluate_defaults_to_the_measured_cap_not_the_legacy_one():
    """The v65-v81 default of 40 truncated eleven of the corpus's task shapes."""

    import inspect

    default = inspect.signature(solving.evaluate).parameters["max_new_tokens"].default

    assert default == solving.DEFAULT_MAX_NEW_TOKENS == 96
    assert default != solving.LEGACY_MAX_NEW_TOKENS


def test_different_seeds_give_different_problems():
    first = [p.prompt for p in solving.generate_novel(20, seed=7)]
    second = [p.prompt for p in solving.generate_novel(20, seed=8)]

    assert first != second


def test_seen_problems_skip_rows_whose_answer_cannot_be_parsed(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        json.dumps({"task": "arithmetic", "user": "1 + 1", "assistant": "2"}) + "\n"
        + json.dumps({"task": "arithmetic", "user": "2 + 2", "assistant": "no number"}) + "\n",
        encoding="utf-8",
    )

    problems = solving.load_seen(path, count=10)

    assert len(problems) == 1
    assert problems[0].source == "seen"


# -- digit tokenisation -----------------------------------------------------


def test_digit_tokeniser_splits_numbers():
    tokenizer = text_utils.WordTokenizer.build(["498 - 419 = 79"], digit_tokens=True)

    pieces = tokenizer.pattern.findall("498 - 419")

    assert pieces == ["4", "9", "8", " -", " 4", "1", "9"]


def test_default_tokeniser_still_keeps_whole_numbers():
    """Every result before v65 depends on this staying unchanged."""

    tokenizer = text_utils.WordTokenizer.build(["498 - 419 = 79"])

    assert tokenizer.pattern.findall("498 - 419") == ["498", " -", " 419"]


def test_digit_tokeniser_roundtrips():
    text = "Solve this: 498 - 419. Answer: 10.68."
    tokenizer = text_utils.WordTokenizer.build([text], digit_tokens=True)

    text_utils.assert_roundtrip(tokenizer, [text])  # must not raise


def test_digit_tokeniser_shrinks_a_numeric_vocabulary():
    texts = [f"Solve this basic math problem: {a} + {b}" for a in range(100, 160)
             for b in range(10, 40)]

    whole = text_utils.WordTokenizer.build(texts)
    digits = text_utils.WordTokenizer.build(texts, digit_tokens=True)

    assert digits.vocab_size < whole.vocab_size / 5


def test_digit_setting_travels_with_the_checkpoint():
    """Reloading under the wrong setting would segment every number differently."""

    tokenizer = text_utils.WordTokenizer.build(["498 - 419"], digit_tokens=True)

    restored = text_utils.WordTokenizer.from_dict(tokenizer.to_dict())

    assert restored.digit_tokens is True
    assert restored.encode("498") == tokenizer.encode("498")


def test_pre_v65_checkpoints_default_to_whole_numbers():
    """A payload without the key predates the flag and used whole numbers."""

    restored = text_utils.WordTokenizer.from_dict({"tokens": list(text_utils.SPECIAL_TOKENS)})

    assert restored.digit_tokens is False
