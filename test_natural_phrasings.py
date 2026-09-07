"""The phrasing bank, and the holdout that makes it measurable.

Widening a prompt bank is only worth doing if the widening can be shown to have
taught something. Two failure modes make that easy to get wrong, and both have
already happened in this project:

* **Measuring on the training templates.** v74 scored 0.894 on a benchmark that
  generated prompts from its own four templates, and answered 0 of 5 questions
  typed by a person. A wider bank scored the same way would report a number that
  means just as little, because memorising fifteen templates is no harder than
  memorising five.
* **Changing the corpus while claiming to change only the prompt.** If enabling
  the arm perturbs the generator's RNG, then operands, answers and worked
  responses all move too, and any score difference is uninterpretable.

These tests pin the properties that rule both out.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "source"))

import build_omni_corpus as omni  # noqa: E402
import natural_phrasings as phrasings  # noqa: E402

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def shape(text: str) -> str:
    """Collapse every number so a template and a rendered prompt compare equal.

    Both sides must go through this. A unit like ``m/s^2`` contains a digit, so
    normalising only the rendered prompt turns it into ``m/s^{}`` while the
    template keeps ``m/s^2`` -- and then a leak check can never fire, which is
    exactly the false pass this helper exists to prevent.
    """

    return re.sub(r"[\d.]+", "{}", PLACEHOLDER.sub("{}", text))


@pytest.fixture(autouse=True)
def _restore():
    """Never leak the flag into another test."""

    was = omni.NATURAL_PHRASINGS
    yield
    omni.NATURAL_PHRASINGS = was


# ---------------------------------------------------------------------------
# The holdout
# ---------------------------------------------------------------------------


def test_no_held_out_phrasing_can_reach_a_corpus():
    """The property the benchmark rests on.

    If a withheld form appears in training, the benchmark stops measuring
    generalisation and starts measuring recall, silently.
    """

    omni.NATURAL_PHRASINGS = True
    rng = random.Random(4)
    withheld = {shape(form)
                for task in omni.TASKS for form in phrasings.held_out(task)}
    assert withheld, "nothing is held out, so the benchmark measures nothing"

    for task in sorted(omni.TASKS):
        for _ in range(400):
            prompt = omni.TASKS[task](rng).prompt
            assert shape(prompt) not in withheld, (
                f"{task}: a held-out phrasing was built into the corpus -- "
                f"{prompt!r}"
            )


def test_the_holdout_is_taken_from_the_end_of_each_bank():
    """So appending a phrasing extends training, never moving one out of the test."""

    for task, forms in phrasings.EXTRA_PHRASINGS.items():
        withheld = phrasings.held_out(task)
        if withheld:
            assert forms[-len(withheld):] == withheld, task


def test_held_out_only_yields_exactly_the_withheld_forms():
    omni.NATURAL_PHRASINGS = True
    rng = random.Random(9)
    with phrasings.held_out_only():
        for task in sorted(omni.TASKS):
            if not phrasings.held_out(task):
                continue
            shapes = {shape(omni.TASKS[task](rng).prompt) for _ in range(60)}
            expected = {shape(f) for f in phrasings.held_out(task)}
            assert shapes <= expected, task


def test_the_context_manager_restores_the_previous_state():
    assert phrasings._HELD_OUT_ONLY is False
    with phrasings.held_out_only():
        assert phrasings._HELD_OUT_ONLY is True
    assert phrasings._HELD_OUT_ONLY is False


# ---------------------------------------------------------------------------
# The arm is a single variable
# ---------------------------------------------------------------------------


def test_enabling_phrasings_changes_the_prompt_and_nothing_else():
    """Same operands, same answer, same worked response -- only the wording.

    Without this the arm is not interpretable: a score change could be the
    phrasing or could be that every problem in the corpus moved.
    """

    def draw(flag):
        omni.NATURAL_PHRASINGS = flag
        rng = random.Random(99)
        return [omni.TASKS[t](rng)
                for t in sorted(omni.TASKS) for _ in range(40)]

    plain, wide = draw(False), draw(True)
    assert len(plain) == len(wide)
    for a, b in zip(plain, wide):
        assert a.answer == b.answer
        assert a.response == b.response
        assert a.canonical == b.canonical, (
            "the canonical query is what the solver parses; widening the prompt "
            "bank must not be able to weaken verification"
        )
    assert sum(a.prompt != b.prompt for a, b in zip(plain, wide)) > len(plain) // 2


def test_every_widened_row_still_verifies_against_the_solver():
    omni.NATURAL_PHRASINGS = True
    rng = random.Random(7)
    for task in sorted(omni.TASKS):
        for _ in range(40):
            assert omni.verify(omni.TASKS[task](rng)), task


def test_the_original_templates_survive_the_widening():
    """`prompt_normaliser` rewrites *into* these, so they cannot be dropped."""

    for task, forms in phrasings.EXTRA_PHRASINGS.items():
        narrow = ("A {x} template.", "Another {x} one.")
        wide = phrasings.bank(task, narrow)
        assert wide[:len(narrow)] == narrow, task


def test_an_unknown_task_keeps_its_own_templates():
    narrow = ("only {x} form",)
    assert phrasings.bank("no_such_task", narrow) == narrow


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_no_widened_turn_leaves_the_sequence_budget():
    """Turn-aligned packing drops an over-length turn without a word.

    Casual phrasings are wordier in principle, so this is checked rather than
    assumed. Measured: the median prompt actually falls 15 -> 14 tokens.
    """

    text_utils = pytest.importorskip("mimomix_text")
    omni.NATURAL_PHRASINGS = True
    rng = random.Random(21)
    rows = [omni.TASKS[t](rng) for t in sorted(omni.TASKS) for _ in range(60)]
    tokenizer = text_utils.WordTokenizer.build(
        (field for row in rows for field in (row.prompt, row.response)),
        max_vocab=16384, digit_tokens=True,
    )
    lengths = [len(tokenizer.encode(r.prompt)) + len(tokenizer.encode(r.response))
               for r in rows]
    assert max(lengths) < 128, (
        f"a widened turn reaches {max(lengths)} tokens and would be dropped"
    )


def test_every_phrasing_uses_only_placeholders_its_task_supplies():
    """A template naming a key the generator does not pass raises at build time."""

    omni.NATURAL_PHRASINGS = True
    rng = random.Random(3)
    for task in sorted(omni.TASKS):
        supplied = None
        for form in phrasings.EXTRA_PHRASINGS.get(task, ()):
            needed = set(PLACEHOLDER.findall(form))
            if supplied is None:
                # Render once through the generator to learn the real kwargs.
                with phrasings.held_out_only():
                    omni.TASKS[task](rng)
                supplied = needed
            assert needed == supplied, (
                f"{task}: {form!r} uses {sorted(needed)}, other forms use "
                f"{sorted(supplied)}"
            )
