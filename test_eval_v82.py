"""Tests for the v82 benchmark corrections.

Four things changed about how `eval_problem_solving` measures, and each one is
a way the previous benchmark could mislead rather than a way the model changed:

1. Tasks shared one RNG, so adding or reordering a task silently changed every
   later task's problems for the same seed.
2. The generation cap was 40 tokens where the corpus writes replies up to 99,
   so eleven task shapes were being scored on a truncated reply.
3. Only exact match was reported, and a binary grade pays a model to guess.
4. Per-task n is 30, where the 95% interval is +-17 points, and nothing in the
   receipt said so.

These tests pin the corrections. They deliberately do not load a checkpoint --
the scoring paths are exercised against a stub generator, so the suite runs in
seconds and a failure points at the scorer rather than at the model.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import answer_check as check  # noqa: E402
import eval_problem_solving as solving  # noqa: E402


# -- part 1: benchmark determinism -----------------------------------------


def test_adding_a_task_does_not_change_any_existing_task_problems():
    """The bug this replaces: a new task shifted every later task's draw.

    `combination` lost an rng draw in the commit before v82, and because one
    Random fed every generator in turn, that changed the problems of every task
    after it for the same seed. Pre- and post-commit receipts were therefore
    not paired even though both said seed 65.
    """

    names = sorted(solving.GENERATORS)
    before = solving.generate_novel(len(names) * 4, seed=65, tasks=names)

    inserted = names[:3] + ["_injected_task_"] + names[3:]
    solving.GENERATORS["_injected_task_"] = lambda rng: solving.Problem(
        "_injected_task_", f"noise {rng.randint(0, 10**6)}", 0.0, "novel"
    )
    try:
        after = solving.generate_novel(len(inserted) * 4, seed=65, tasks=inserted)
    finally:
        del solving.GENERATORS["_injected_task_"]

    def by_task(problems):
        grouped = {}
        for problem in problems:
            grouped.setdefault(problem.task, []).append(problem.prompt)
        return grouped

    original, extended = by_task(before), by_task(after)
    for task in names:
        assert original[task] == extended[task], task


def test_reordering_the_task_list_does_not_change_a_tasks_problems():
    names = sorted(solving.GENERATORS)
    forward = solving.generate_novel(len(names) * 3, seed=65, tasks=names)
    backward = solving.generate_novel(len(names) * 3, seed=65, tasks=list(reversed(names)))

    def prompts(problems, task):
        return [p.prompt for p in problems if p.task == task]

    for task in names:
        assert prompts(forward, task) == prompts(backward, task), task


def test_shared_rng_is_the_thing_that_was_broken():
    """A guard on the guard: under the old scheme reordering *does* change it."""

    names = sorted(solving.GENERATORS)
    forward = solving.generate_novel(len(names) * 3, seed=65, tasks=names, shared_rng=True)
    backward = solving.generate_novel(
        len(names) * 3, seed=65, tasks=list(reversed(names)), shared_rng=True
    )

    changed = [
        task
        for task in names
        if [p.prompt for p in forward if p.task == task]
        != [p.prompt for p in backward if p.task == task]
    ]
    assert changed, "the shared-RNG path is supposed to be order-dependent"


def test_task_rng_depends_on_the_name_not_the_position():
    first = solving.task_rng("force", 65).random()
    again = solving.task_rng("force", 65).random()
    other = solving.task_rng("work", 65).random()
    other_seed = solving.task_rng("force", 66).random()

    assert first == again
    assert first != other
    assert first != other_seed


def test_generation_is_still_deterministic_for_a_seed():
    first = [p.prompt for p in solving.generate_novel(30, seed=7)]
    second = [p.prompt for p in solving.generate_novel(30, seed=7)]

    assert first == second


def test_round_robin_order_is_unchanged():
    """Only the randomness moved; problem i is still task i % len(tasks)."""

    names = sorted(solving.GENERATORS)
    problems = solving.generate_novel(len(names) * 2, seed=65, tasks=names)

    assert [p.task for p in problems] == [names[i % len(names)] for i in range(len(problems))]


def test_legacy_shared_rng_reproduces_the_v80_receipt():
    """The v80 receipt is only re-checkable because the old draw is preserved."""

    receipt = REPO_ROOT / "output" / "v80_omni" / "problem_solving_n630.json"
    if not receipt.is_file():
        pytest.skip("v80 receipt not present")
    recorded = json.loads(receipt.read_text(encoding="utf-8"))["examples"]

    problems = solving.generate_novel(630, seed=65, shared_rng=True)

    for example, problem in zip(recorded, problems):
        assert example["prompt"] == problem.prompt[: len(example["prompt"])]


def test_generator_fingerprint_is_stable_and_task_scoped():
    assert solving.generator_fingerprint() == solving.generator_fingerprint()

    subset = sorted(solving.GENERATORS)[:5]
    assert solving.generator_fingerprint(subset) != solving.generator_fingerprint()


def test_generator_fingerprint_moves_when_a_generator_changes():
    names = sorted(solving.GENERATORS)[:3]
    before = solving.generator_fingerprint(names)

    original = solving.GENERATORS[names[0]]
    solving.GENERATORS[names[0]] = lambda rng: solving.Problem(
        names[0], "changed", 1.0, "novel"
    )
    try:
        after = solving.generator_fingerprint(names)
    finally:
        solving.GENERATORS[names[0]] = original

    assert before != after


# -- part 2: the generation cap --------------------------------------------


def test_default_cap_is_no_longer_forty():
    assert solving.LEGACY_MAX_NEW_TOKENS == 40
    assert solving.DEFAULT_MAX_NEW_TOKENS == 96


def test_default_cap_fits_inside_the_v80_context_window():
    """96 was chosen against a measurement, not for roundness.

    The longest prompt any generator produces, encoded with the v80 tokenizer,
    is 30 tokens; `max_position_embeddings` is 128. This asserts the prompt
    half of that so a wider prompt template cannot quietly push generation past
    the trained context.
    """

    torch = pytest.importorskip("torch")
    checkpoint = REPO_ROOT / "output" / "v80_omni" / "v80_omni.pt"
    if not checkpoint.is_file():
        pytest.skip("v80 checkpoint not present")
    import mimomix_text as text_utils

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    context = int(payload["config"]["max_position_embeddings"])

    longest = max(
        len(tokenizer.encode_turn(p.prompt, None)[0])
        for p in solving.generate_novel(210, seed=65)
    )

    assert longest + solving.DEFAULT_MAX_NEW_TOKENS <= context


def test_truncation_needs_both_the_cap_and_a_missing_total():
    assert solving.is_truncated("600 + 200 = 800, 17 + 88", 40, 40)
    assert not solving.is_truncated("600 + 200 = 800, total 905", 40, 40)
    assert not solving.is_truncated("gave up early", 11, 40)
    assert not solving.is_truncated("no token count", None, 40)


def test_looks_terminated_reads_the_corpus_ending():
    assert solving.looks_terminated("power = voltage x current, total 250")
    assert solving.looks_terminated("Answer: total 10.68.")
    assert not solving.looks_terminated("total")
    assert not solving.looks_terminated("600 + 200 = 800,")


# -- part 3: abstention-aware scoring and intervals -------------------------


def test_abstention_score_prices_a_guess():
    """Guessing is free under exact match and costs under this score."""

    assert solving.abstention_score(10, 0, 0) == pytest.approx(1.0)
    assert solving.abstention_score(0, 10, 0) == pytest.approx(-1.0)
    assert solving.abstention_score(0, 0, 10) == pytest.approx(0.0)
    assert solving.abstention_score(5, 5, 0) == pytest.approx(0.0)
    assert solving.abstention_score(0, 0, 0) is None


def test_abstention_score_does_not_replace_exact_match():
    """Both must be reported; the headline number stays exact match."""

    report = _stub_report()

    for entry in report["by_source"].values():
        assert "accuracy" in entry and "abstention_score" in entry


def test_wilson_interval_is_not_zero_width_at_the_extremes():
    low, high = solving.wilson_interval(0, 30)
    assert low == 0.0 and high == pytest.approx(0.1135, abs=1e-3)

    low, high = solving.wilson_interval(30, 30)
    assert low == pytest.approx(0.8865, abs=1e-3) and high == 1.0


def test_wilson_interval_width_at_n30_is_the_documented_seventeen_points():
    low, high = solving.wilson_interval(15, 30)

    assert (high - low) / 2 == pytest.approx(0.168, abs=2e-3)


def test_wilson_interval_narrows_with_n():
    narrow = solving.wilson_interval(300, 600)
    wide = solving.wilson_interval(15, 30)

    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_interval_of_an_empty_task_is_not_a_claim():
    assert solving.wilson_interval(0, 0) == (0.0, 0.0)


# -- part 4: the non-claims -------------------------------------------------


def test_non_claims_no_longer_describe_five_task_types():
    joined = " ".join(solving.NON_CLAIMS).lower()

    assert "five task types" not in joined
    assert "21 tasks" in joined


def test_the_honest_non_claims_survived():
    joined = " ".join(solving.NON_CLAIMS).lower()

    assert "lower bound" in joined            # last-number extraction
    assert "recall, not skill" in joined      # seen vs novel


def test_non_claims_state_the_interval_and_the_comparability_rule():
    joined = " ".join(solving.NON_CLAIMS).lower()

    assert "generator_fingerprint" in joined
    assert "min_selection_problems" in joined
    assert "truncated_replies" in joined


# -- receipt shape ----------------------------------------------------------


class _StubReply:
    """A generate_reply stand-in, so the scorer is tested without a model."""

    def __init__(self, replies):
        self._replies = list(replies)
        self._index = 0

    def __call__(self, model, tokenizer, prompt, max_new_tokens=40, **kwargs):
        text, tokens = self._replies[self._index % len(self._replies)]
        self._index += 1
        return {"reply": text, "tokens": tokens}


def _stub_report(monkeypatch=None, replies=None, **kwargs):
    """Run `evaluate` against a stub model and a stub generator."""

    import unittest.mock as mock

    problems = [
        solving.Problem("arithmetic", "1 + 1", 2.0, "novel"),
        solving.Problem("arithmetic", "2 + 2", 4.0, "novel"),
        solving.Problem("arithmetic", "3 + 3", 6.0, "novel"),
        solving.Problem("arithmetic", "4 + 4", 8.0, "novel"),
    ]
    replies = replies or [
        ("total 2", 5),          # correct
        ("total 5", 5),          # wrong
        ("no answer here", 5),   # abstained
        ("1 + 1 = 2, 3 + 3", 8),  # truncated at the cap
    ]
    with mock.patch.object(solving, "generate_reply", _StubReply(replies)), mock.patch.object(
        solving, "load_talk_checkpoint", lambda path: (mock.MagicMock(), None, {"extra": {}})
    ):
        return solving.evaluate(Path("stub.pt"), problems, max_new_tokens=8, **kwargs)


def test_receipt_reports_truncated_alongside_unparsed():
    report = _stub_report()

    assert report["unparsed_replies"] == 1
    assert report["truncated_replies"] == 1
    assert report["by_source"]["novel"]["truncated"] == 1


def test_a_truncated_reply_is_not_scored_as_a_confident_wrong_answer():
    """It counts as an abstention, because the harness ended the reply."""

    report = _stub_report()
    novel = report["by_source"]["novel"]

    assert novel["correct"] == 1
    assert novel["abstained"] == 2      # the no-number reply and the truncated one
    assert novel["wrong"] == 1
    assert novel["abstention_score"] == pytest.approx(0.0)


def test_accuracy_still_counts_truncated_replies_as_wrong():
    """`accuracy` stays a lower bound; the upper bound is reported separately."""

    report = _stub_report()
    novel = report["by_source"]["novel"]

    assert novel["accuracy"] == pytest.approx(0.25)
    assert novel["accuracy_untruncated"] == pytest.approx(1 / 3, abs=1e-4)


def test_receipt_records_the_settings_needed_to_compare_two_runs():
    report = _stub_report(
        settings={"seed": 65, "tasks": ["arithmetic"], "shared_rng": False,
                  "generator_fingerprint": solving.generator_fingerprint(["arithmetic"])}
    )

    settings = report["settings"]
    assert settings["max_new_tokens"] == 8
    assert settings["seed"] == 65
    assert settings["tasks"] == ["arithmetic"]
    assert settings["shared_rng"] is False
    assert settings["generator_fingerprint"]


def test_receipt_schema_moved_so_v1_receipts_are_not_mistaken_for_v2():
    report = _stub_report()

    assert report["schema"] == solving.RECEIPT_SCHEMA
    assert report["schema"] != solving.RECEIPT_SCHEMA_V1


def test_every_task_carries_an_interval():
    report = _stub_report()

    for counts in report["by_source"]["novel"]["tasks"].values():
        low, high = counts["accuracy_95ci"]
        assert 0.0 <= low <= counts["accuracy"] <= high <= 1.0


def test_print_summary_runs_on_a_full_receipt(capsys):
    solving.print_summary(_stub_report())

    out = capsys.readouterr().out
    assert "abstention score" in out
    assert "fingerprint" in out


# -- part 5: answer_check ---------------------------------------------------


def test_answer_check_covers_every_registered_benchmark_task():
    """A task the live checker cannot parse is a task nobody can verify live."""

    rng = random.Random(4242)
    unparsed = []
    for name in sorted(solving.GENERATORS):
        for _ in range(10):
            problem = solving.GENERATORS[name](rng)
            if check.parse_question(problem.prompt) is None:
                unparsed.append((name, problem.prompt))
                break

    assert unparsed == []


def test_answer_check_never_produces_a_confident_wrong_verdict():
    """The one failure mode that matters: a verdict that is wrong, not absent."""

    rng = random.Random(4242)
    disagreements = []
    for name in sorted(solving.GENERATORS):
        for _ in range(20):
            problem = solving.GENERATORS[name](rng)
            parsed = check.parse_question(problem.prompt)
            if parsed is None:
                continue
            _, expected = parsed
            if not solving.is_correct(expected, problem.answer):
                disagreements.append((name, problem.prompt, expected, problem.answer))

    assert disagreements == []


def test_the_live_checker_and_the_benchmark_agree_on_tolerance():
    assert check.TOLERANCE == solving.TOLERANCE
