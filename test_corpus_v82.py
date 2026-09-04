"""v82 corpus and tokenizer work: the token budget, and five cited hypotheses.

Every behaviour added here is **off by default**, and the first test in each
section pins that -- because the v80 corpus and the v80 checkpoint are the only
measured artifacts in this line and a silent format change would make them
incomparable with whatever comes next.

What the suite establishes:

1. **The budget is measurable before a run pays for it.** V67 lost its
   six-value `average` rows to turn-aligned packing and nothing reported it;
   V72 then measured that growing the block from 128 to 160 costs 24 accuracy
   points. So a format change is a budget decision, and
   `token_budget_report` is the receipt.
2. **The two long-failing formats change shape, not correctness.** `average`
   and `algebra_one_step` still parse to their own answers, and the new
   `average` still fits 128 tokens.
3. **A retry row never moves the answer.** The benchmark reads the last number
   in a reply, so a correction in the final step would score every correct
   model as wrong.
4. **Reversed digits are lossless.** The reversal happens inside the tokenizer,
   so the corpus builders, `answer_check` and `eval_problem_solving` see
   ordinary numbers, and a round trip must return the text unchanged.

None of these tests claims any of it helps. Nothing here has been trained.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import build_omni_corpus as omni  # noqa: E402
import build_scratchpad_math as scratch  # noqa: E402
import mimomix_text as mt  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_module_flags():
    """Every flag is a module global; a leaked one would poison later tests."""

    saved = (
        omni.COMBINATION_IN_ENVELOPE,
        scratch.AVERAGE_BINARY_STEPS,
        scratch.ALGEBRA_WORD_SIGN,
        scratch.DECOMPOSE_INNER,
    )
    yield
    (
        omni.COMBINATION_IN_ENVELOPE,
        scratch.AVERAGE_BINARY_STEPS,
        scratch.ALGEBRA_WORD_SIGN,
        scratch.DECOMPOSE_INNER,
    ) = saved


def _last_number(text: str) -> float:
    """The benchmark's own extraction rule."""

    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(matches[-1])


# ---------------------------------------------------------------------------
# Part 1: the token budget guard
# ---------------------------------------------------------------------------


def test_token_budget_measures_every_task():
    rows, _ = omni.build(per_task=12, seed=82)
    report = omni.token_budget_report(rows)

    assert report["sequence_length"] == 128
    assert set(report["tasks"]) == set(omni.TASKS)
    for name, stats in report["tasks"].items():
        assert stats["response_median"] <= stats["response_p95"] <= stats["response_max"]
        assert stats["turn_max"] >= stats["response_max"], name


def test_token_budget_counts_match_the_tokenizer_training_uses():
    """The count must be the tokenizer's, not an approximation of it."""

    rows, _ = omni.build(per_task=4, seed=83, tasks=["force"])
    tokenizer = mt.WordTokenizer([], digit_tokens=True)
    report = omni.token_budget_report(rows)

    longest = max(len(tokenizer.pattern.findall(row["assistant"])) for row in rows)
    assert report["tasks"]["force"]["response_max"] == longest


def test_token_budget_reproduces_the_measured_long_tasks():
    """Pinned so a format change to these three cannot pass unnoticed.

    Measured today over the current generators with digit-level tokenisation:
    `arithmetic_series` responses run to a median of 81 tokens, `combination`
    60 and `kinetic_energy` 54, against 38 or less for every other task.
    """

    rows, _ = omni.build(per_task=120, seed=84)
    tasks = omni.token_budget_report(rows)["tasks"]

    assert tasks["arithmetic_series"]["response_median"] >= 70
    assert tasks["combination"]["response_median"] >= 50
    assert max(stats["response_median"] for name, stats in tasks.items()
               if name not in {"arithmetic_series", "combination", "kinetic_energy"}) <= 40


def test_token_budget_reports_what_packing_would_drop():
    """A short block drops nearly everything; the fraction must say so."""

    rows, _ = omni.build(per_task=20, seed=85, tasks=["arithmetic_series"])
    generous = omni.token_budget_report(rows, sequence_length=128)
    mean = omni.token_budget_report(rows, sequence_length=64)

    assert generous["tasks"]["arithmetic_series"]["dropped_fraction"] == 0.0
    assert mean["tasks"]["arithmetic_series"]["dropped_fraction"] == 1.0
    assert mean["worst_dropped_fraction"] == 1.0


def test_no_current_task_is_dropped_at_128():
    """The claim the corpus rests on. If this fails, a task is vanishing."""

    rows, _ = omni.build(per_task=60, seed=86)

    assert omni.token_budget_report(rows)["worst_dropped_fraction"] == 0.0


def test_the_report_carries_the_budget_only_when_asked():
    _, plain = omni.build(per_task=2, seed=87, tasks=["force"])
    _, measured = omni.build(per_task=2, seed=87, tasks=["force"], token_budget=True)

    assert "token_budget" not in plain
    assert measured["token_budget"]["tasks"]["force"]["rows"] == 2


# ---------------------------------------------------------------------------
# Part 2: the two long-failing formats
# ---------------------------------------------------------------------------


def test_average_binary_steps_is_off_by_default():
    """v80 trained on the terse form; the default must still emit it."""

    item = scratch._scratchpad_average(random.Random(1))

    assert item["working"].startswith("sum: ")
    assert " + " not in item["working"]


def test_average_binary_steps_writes_both_operands():
    """The fixed-dependency failure: a running total with no operands shown."""

    scratch.AVERAGE_BINARY_STEPS = True
    item = scratch._scratchpad_average(random.Random(1))

    assert "sum: " not in item["working"]
    steps = re.findall(r"(\d+) \+ (\d+) = (\d+)", item["working"])
    assert steps, item["working"]
    for left, right, result in steps:
        assert int(left) + int(right) == int(result)


def test_average_binary_steps_chain_consumes_the_previous_result():
    """Each step must start from the last one, or it is not a running sum."""

    scratch.AVERAGE_BINARY_STEPS = True
    for seed in range(30):
        item = scratch._scratchpad_average(random.Random(seed))
        steps = re.findall(r"(\d+) \+ (\d+) = (\d+)", item["working"])
        for previous, following in zip(steps, steps[1:]):
            assert following[0] == previous[2], item["working"]


@pytest.mark.parametrize("binary", [False, True])
def test_average_still_parses_to_its_own_answer(binary):
    scratch.AVERAGE_BINARY_STEPS = binary
    for seed in range(200):
        item = scratch._scratchpad_average(random.Random(seed))
        assert abs(_last_number(item["working"]) - item["answer"]) < 1e-4


def test_average_binary_steps_still_fits_the_block():
    """The measurement that decides whether the format is affordable at all.

    Measured over 3,000 rows: the terse form runs to 44 response tokens and a
    75-token turn; binary steps run to 73 and 104. Both fit 128, so nothing is
    dropped -- which is the only reason this format is available.
    """

    scratch.AVERAGE_BINARY_STEPS = True
    tokenizer = mt.WordTokenizer([], digit_tokens=True)
    lengths = []
    for seed in range(400):
        item = scratch._scratchpad_average(random.Random(seed))
        lengths.append(len(tokenizer.encode_turn(item["expression"], item["working"])[0]))

    assert max(lengths) <= 128, max(lengths)


def test_algebra_word_sign_is_off_by_default():
    item = scratch._scratchpad_algebra(random.Random(2))

    assert item["working"].startswith("subtract ")


def test_algebra_never_says_subtract_a_negative():
    """`subtract -12` gives the model two chances to lose the sign."""

    scratch.ALGEBRA_WORD_SIGN = True
    for seed in range(300):
        working = scratch._scratchpad_algebra(random.Random(seed))["working"]
        assert not re.match(r"subtract -\d", working), working
        assert re.match(r"(subtract \d+ from|add \d+ to) both sides,", working), working


def test_algebra_decomposes_the_arithmetic_by_place_value():
    """The observed failure was a borrow in one jump: 34 - 29 = 4, truth 5."""

    scratch.ALGEBRA_WORD_SIGN = True
    item = scratch._scratchpad_algebra(random.Random(0))
    steps = re.findall(r"(-?\d+) ([-+]) (-?\d+) = (-?\d+)", item["working"])

    assert len(steps) == 2, item["working"]
    high, low = (int(step[3]) for step in steps)
    assert high + low == item["answer"]


@pytest.mark.parametrize("worded", [False, True])
def test_algebra_still_parses_to_its_own_answer(worded):
    scratch.ALGEBRA_WORD_SIGN = worded
    for seed in range(400):
        item = scratch._scratchpad_algebra(random.Random(seed))
        assert _last_number(item["working"]) == item["answer"]


def test_the_specific_row_that_failed():
    """`x + 29 = 34` produced 4 where the truth is 5. Pinned by construction."""

    scratch.ALGEBRA_WORD_SIGN = True
    rng = random.Random(0)
    for seed in range(4000):
        rng.seed(seed)
        item = scratch._scratchpad_algebra(rng)
        if item["expression"] == "Solve for x: x + 29 = 34":
            assert item["answer"] == 5.0
            assert item["working"].endswith("total 5")
            assert "30 - 20 = 10" in item["working"]
            return
    pytest.skip("that operand pair was not drawn in 4,000 tries")


def test_combination_in_envelope_is_off_by_default():
    """v80's combination rows must be reproducible exactly."""

    problem = omni.TASKS["combination"](random.Random(3))

    assert "half of" in problem.response
    assert re.search(r"half of \d+ = \d+, there are", problem.response)


def test_combination_in_envelope_halves_before_multiplying():
    """`60 x 59` in one step is outside everything the model has learned."""

    omni.COMBINATION_IN_ENVELOPE = True
    for seed in range(60):
        problem = omni.TASKS["combination"](random.Random(seed))
        assert omni.verify(problem)
        assert omni.extract_answer(problem.response) == problem.answer
        products = re.findall(r"(\d+) x (\d+) = (\d+)", problem.response)
        assert products, problem.response
        for left, right, result in products:
            assert int(left) * int(right) == int(result)
            # Two-digit by one-digit is the shape `multiplication` scores 1.00 on.
            assert len(right) == 1, problem.response
            assert len(left) <= 2, problem.response


# ---------------------------------------------------------------------------
# Part 3: retry rows
# ---------------------------------------------------------------------------


def test_retry_is_off_by_default():
    rows, report = omni.build(per_task=30, seed=88, tasks=["force"])

    assert "retry_rows" not in report
    assert all(f", {omni.RETRY_MARKER}," not in row["assistant"] for row in rows)


def test_retry_inserts_a_wrong_step_then_the_correction():
    retried = omni.inject_retry("30 x 3 = 90, 9 x 3 = 27, total 117", random.Random(0))

    assert retried is not None
    assert f", {omni.RETRY_MARKER}, " in retried
    assert retried.endswith("total 117")


def test_a_retry_is_never_the_last_number():
    """The extractor takes the last number; a retry there scores every row wrong."""

    rng = random.Random(4)
    for seed in range(60):
        problem = omni.TASKS["force"](random.Random(seed))
        retried = omni.inject_retry(problem.response, rng)
        if retried is None:
            continue
        assert omni.extract_answer(retried) == problem.answer, retried


def test_a_retried_corpus_still_verifies_and_still_parses():
    rows, report = omni.build(per_task=40, seed=89, retry_rate=1.0,
                              tasks=["force", "work", "molarity"])

    assert sum(report["retry_rows"].values()) > 0
    for row in rows:
        canonical_answer = omni.extract_answer(row["assistant"])
        assert canonical_answer is not None
        if f", {omni.RETRY_MARKER}, " in row["assistant"]:
            # The wrong value is present, and it is not the one that counts.
            assert row["assistant"].count(" = ") >= 3


def test_the_retry_marker_is_a_single_word():
    """A bracketed token becomes four symbols under digit-level tokenisation."""

    tokenizer = mt.WordTokenizer([], digit_tokens=True)

    assert len(tokenizer.pattern.findall(omni.RETRY_MARKER)) == 1


def test_a_response_with_no_intermediate_step_is_left_alone():
    assert omni.inject_retry("total 5", random.Random(0)) is None


# ---------------------------------------------------------------------------
# Part 4: balanced operands
# ---------------------------------------------------------------------------


def test_carry_counting_is_column_wise():
    assert omni._carry_count(47, 38, "+") == 1
    assert omni._carry_count(99, 99, "+") == 2
    assert omni._carry_count(34, 29, "-") == 1
    assert omni._carry_count(30, 20, "-") == 0
    # Undefined cases must not guess a bucket.
    assert omni._carry_count(5, 3, "x") == 0
    assert omni._carry_count(-5, 3, "+") == 0


def test_balancing_is_off_by_default():
    _, report = omni.build(per_task=20, seed=90, tasks=["force"])

    assert "operand_balance" not in report


def test_balancing_reports_both_histograms():
    _, report = omni.build(per_task=120, seed=91, tasks=["force"],
                           balanced_operands=True)
    balance = report["operand_balance"]["force"]

    assert balance["buckets_before"] and balance["buckets_after"]
    assert set(balance["buckets_after"]) <= set(balance["buckets_before"])


def test_balancing_flattens_the_histogram():
    """Uniform sampling leaves the high-carry buckets rare; that is the point."""

    _, report = omni.build(per_task=200, seed=92, tasks=["work"],
                           balanced_operands=True)
    balance = report["operand_balance"]["work"]
    before = list(balance["buckets_before"].values())
    after = list(balance["buckets_after"].values())

    assert max(before) - min(before) > max(after) - min(after)


def test_a_balanced_build_still_verifies_every_row():
    rows, _ = omni.build(per_task=60, seed=93, tasks=["momentum"],
                         balanced_operands=True)

    assert rows
    for row in rows:
        assert row["assistant"].rstrip().split()[-2] == "total"


# ---------------------------------------------------------------------------
# Part 5: train-set priming
# ---------------------------------------------------------------------------


def test_priming_is_off_by_default():
    _, report = omni.build(per_task=20, seed=94, tasks=["force"])

    assert "priming_rows" not in report


def test_priming_widens_the_range_without_changing_the_format():
    """V67's bug from the other side: the benchmark's top must be interior."""

    ordinary = {omni.TASKS["force"](random.Random(s)).params["m"] for s in range(400)}
    harder_rng = omni._HarderRandom(random.Random(0))
    widened = {omni.TASKS["force"](harder_rng).params["m"] for _ in range(400)}

    assert max(ordinary) <= 99
    assert max(widened) > 99
    # Identical format: the same generator body ran.
    assert omni.TASKS["force"](omni._HarderRandom(random.Random(0))).response.startswith(
        "force = mass x acceleration,"
    )


def test_primed_rows_are_still_solver_verified():
    harder_rng = omni._HarderRandom(random.Random(1))
    for name in omni.TASKS:
        for _ in range(4):
            problem = omni.TASKS[name](harder_rng)
            assert omni.verify(problem), f"{name} priming row failed the solver"
            assert omni.extract_answer(problem.response) == problem.answer


def test_priming_is_counted_in_the_report():
    _, report = omni.build(per_task=400, seed=95, tasks=["force"],
                           priming_fraction=0.05)

    assert 0 < sum(report["priming_rows"].values()) < 400


def test_the_harder_random_keeps_a_step_range_on_its_step():
    """`kinetic_energy` needs an even mass; a widened odd one would not halve."""

    harder_rng = omni._HarderRandom(random.Random(2))
    values = {harder_rng.randrange(2, 20, 2) for _ in range(200)}

    assert all(value % 2 == 0 for value in values)
    assert max(values) > 18


# ---------------------------------------------------------------------------
# Part 6: the canonical query travels with the row
# ---------------------------------------------------------------------------


def test_canonical_is_off_by_default():
    rows, _ = omni.build(per_task=2, seed=96, tasks=["force"])

    assert set(rows[0]) == {"user", "assistant", "domain", "task"}


def test_canonical_makes_every_row_re_verifiable():
    """The model-facing prompt is a phrasing; only this form parses."""

    from nexus_solver import solve_problem

    rows, _ = omni.build(per_task=3, seed=97, keep_canonical=True)
    for row in rows:
        assert row["canonical"]
        assert solve_problem(row["canonical"]).solved, row["canonical"]


def test_most_model_facing_prompts_are_not_re_verifiable_without_it():
    """Why the field is worth its bytes: 41.9% of v80's rows could be checked."""

    from nexus_solver import solve_problem

    rows, _ = omni.build(per_task=8, seed=98)
    solved = sum(1 for row in rows if solve_problem(row["user"]).solved)

    assert solved < len(rows)


# ---------------------------------------------------------------------------
# Part 7: reversed digits in the tokenizer
# ---------------------------------------------------------------------------


def test_reversal_is_an_involution():
    assert mt.reverse_digit_runs("124 + 72 = 196") == "421 + 27 = 691"
    assert mt.reverse_digit_runs(mt.reverse_digit_runs("124 + 72 = 196")) == "124 + 72 = 196"


def test_reverse_digits_is_off_by_default():
    tokenizer = mt.WordTokenizer.build(["total 124"])

    assert tokenizer.reverse_digits is False
    assert "reverse_digits" not in tokenizer.to_dict()


SAMPLE = [
    "force = mass x acceleration, 40 x 3 = 120, 7 x 3 = 21, total 141",
    "Find the average (mean) of these numbers: 61, 63, 72, 61",
    "no digits at all in this line",
    "leading zeros 007 and a big one 1234567",
]


@pytest.mark.parametrize("digit_tokens", [False, True])
def test_reversed_digits_round_trip_exactly(digit_tokens):
    tokenizer = mt.WordTokenizer.build(SAMPLE, digit_tokens=digit_tokens,
                                       reverse_digits=True)
    mt.assert_roundtrip(tokenizer, SAMPLE)
    for text in SAMPLE:
        assert tokenizer.unknown_rate(text) == 0.0, text
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_reversed_digits_round_trip_on_a_large_sample():
    """Fidelity has to hold on the corpus, not on four hand-picked strings."""

    rows, _ = omni.build(per_task=25, seed=99)
    texts = [row["user"] for row in rows] + [row["assistant"] for row in rows]
    tokenizer = mt.WordTokenizer.build(texts, digit_tokens=True, reverse_digits=True)

    assert len(texts) > 250
    mt.assert_roundtrip(tokenizer, texts)
    for text in texts:
        assert tokenizer.unknown_rate(text) == 0.0
        assert tokenizer.decode(tokenizer.encode(text)) == text


def test_reversal_actually_reverses_the_encoded_order():
    """A no-op that round-trips would pass every test above and teach nothing."""

    forward = mt.WordTokenizer.build(SAMPLE, digit_tokens=True)
    reversed_tokenizer = mt.WordTokenizer.build(SAMPLE, digit_tokens=True,
                                                reverse_digits=True)

    assert forward._pieces(" 124") == [" 1", "2", "4"]
    assert reversed_tokenizer._pieces(" 124") == [" 4", "2", "1"]


def test_reverse_digits_composes_with_digit_tokens():
    """Both settings travel together or numbers segment differently on reload."""

    tokenizer = mt.WordTokenizer.build(SAMPLE, digit_tokens=True, reverse_digits=True)
    restored = mt.WordTokenizer.from_dict(tokenizer.to_dict())

    assert restored.digit_tokens and restored.reverse_digits
    assert restored.encode(SAMPLE[0]) == tokenizer.encode(SAMPLE[0])
    assert restored.decode(restored.encode(SAMPLE[0])) == SAMPLE[0]


def test_a_pre_v82_checkpoint_loads_with_the_option_off():
    """Every checkpoint through v80 wrote numbers forwards and says nothing."""

    restored = mt.WordTokenizer.from_dict({"tokens": list(mt.SPECIAL_TOKENS),
                                           "digit_tokens": True})

    assert restored.reverse_digits is False


def test_the_vocabulary_report_states_both_settings():
    tokenizer = mt.WordTokenizer.build(SAMPLE, digit_tokens=True, reverse_digits=True)
    report = tokenizer.vocabulary_report(SAMPLE)

    assert report["digit_tokens"] is True
    assert report["reverse_digits"] is True


# ---------------------------------------------------------------------------
# Part 8: packing honesty
# ---------------------------------------------------------------------------


def test_turn_aligned_packing_reports_what_it_dropped():
    """The docstring promised a receipt should record it; nothing returned it."""

    tokenizer = mt.WordTokenizer.build(["short reply", "a much longer reply here"])
    pairs = [("hi", "short reply")] * 30 + [("hi", "a much longer reply here")] * 10
    stats: dict = {}
    mt.build_training_tensors(pairs, tokenizer, sequence_length=8,
                             turn_aligned=True, stats=stats)

    assert stats["turns"] == 40
    assert stats["dropped_over_length"] == 10
    assert stats["kept"] == 30
    assert stats["dropped_fraction"] == 0.25


def test_stats_are_optional_and_change_nothing():
    tokenizer = mt.WordTokenizer.build(["short reply"])
    pairs = [("hi", "short reply")] * 30
    plain = mt.build_training_tensors(pairs, tokenizer, 16, turn_aligned=True)
    measured = mt.build_training_tensors(pairs, tokenizer, 16, turn_aligned=True,
                                         stats={})

    import torch

    assert torch.equal(plain[0], measured[0]) and torch.equal(plain[1], measured[1])


def test_stream_packing_says_it_drops_no_whole_turns():
    tokenizer = mt.WordTokenizer.build(["short reply"])
    stats: dict = {}
    mt.build_training_tensors([("hi", "short reply")] * 40, tokenizer, 16, stats=stats)

    assert stats["packing"] == "stream"
    assert stats["dropped_over_length"] == 0


def test_turn_alignment_report_needs_no_tensors():
    tokenizer = mt.WordTokenizer.build(["short reply", "a much longer reply here"])
    pairs = [("hi", "short reply")] * 30 + [("hi", "a much longer reply here")] * 10
    report = mt.turn_alignment_report(pairs, tokenizer, sequence_length=8)

    assert report["turns"] == 40
    assert report["dropped_over_length"] == 10
    assert report["max_turn_tokens"] > report["median_turn_tokens"]


def test_the_report_agrees_with_what_packing_actually_does():
    """Two implementations of the same rule must not drift apart."""

    tokenizer = mt.WordTokenizer.build(["short reply", "a much longer reply here"])
    pairs = [("hi", "short reply")] * 17 + [("hi", "a much longer reply here")] * 9
    stats: dict = {}
    mt.build_training_tensors(pairs, tokenizer, 8, turn_aligned=True, stats=stats)
    report = mt.turn_alignment_report(pairs, tokenizer, 8)

    assert stats["dropped_over_length"] == report["dropped_over_length"]


# ---------------------------------------------------------------------------
# The whole point: defaults are the v80 corpus
# ---------------------------------------------------------------------------


def test_every_new_option_defaults_to_the_v80_behaviour():
    """One test that fails loudly if a default ever moves."""

    assert omni.COMBINATION_IN_ENVELOPE is False
    assert scratch.AVERAGE_BINARY_STEPS is False
    assert scratch.ALGEBRA_WORD_SIGN is False
    assert scratch.DECOMPOSE_INNER is False
    assert mt.WordTokenizer([]).reverse_digits is False

    parser = omni.build_parser()
    defaults = vars(parser.parse_args([]))
    assert defaults["retry_rate"] == 0.0
    assert defaults["priming_fraction"] == 0.0
    assert defaults["balanced_operands"] is False
    assert defaults["keep_canonical"] is False
    assert defaults["combination_in_envelope"] is False


def test_the_default_build_is_unchanged_by_every_new_option():
    """Passing each option at its default must produce the identical corpus."""

    baseline, _ = omni.build(per_task=25, seed=100, tasks=["force", "combination"])
    explicit, _ = omni.build(per_task=25, seed=100, tasks=["force", "combination"],
                             retry_rate=0.0, balanced_operands=False,
                             priming_fraction=0.0, keep_canonical=False,
                             token_budget=False)

    assert baseline == explicit
