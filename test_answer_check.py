"""Tests for live answer verification.

The interface renders three states: CORRECT, WRONG, and NOT CHECKED. The
dangerous failure is a false CORRECT -- it would present wrong arithmetic as
verified, which is worse than showing nothing. The second-worst is a confident
verdict on a question the parser only half-recognised, so the parsers are
deliberately narrow and both directions are tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import answer_check as check  # noqa: E402


# -- the five supported shapes ---------------------------------------------


@pytest.mark.parametrize(
    "question,expected,task",
    [
        ("Solve this basic math problem: 617 + 288", 905.0, "arithmetic"),
        ("Quick question: 524 - 305", 219.0, "arithmetic"),
        ("What is 25% of 840?", 210.0, "percent"),
        ("Solve for x: x + 14 = 39", 25.0, "algebra_one_step"),
        ("Solve for x: x - 10 = 26", 36.0, "algebra_one_step"),
        (
            "A student has 45 marbles. They get 38 more and then give away 27. "
            "How many marbles do they have now?",
            56.0,
            "word_problem",
        ),
        ("Find the average (mean) of these numbers: 40, 60, 20, 80", 50.0, "average"),
    ],
)
def test_ground_truth_is_recomputed_from_the_question(question, expected, task):
    parsed = check.parse_question(question)

    assert parsed is not None
    assert parsed[0] == task
    assert parsed[1] == pytest.approx(expected)


def test_correct_reply_is_marked_correct():
    verdict = check.check(
        "Solve this basic math problem: 617 + 288",
        "600 + 200 = 800, 17 + 88 = 105, total 905",
    )

    assert verdict is not None and verdict.correct


def test_wrong_reply_is_marked_wrong():
    """The v70 failure mode: correct working, slipped digit."""

    verdict = check.check(
        "Solve this basic math problem: 617 + 288",
        "600 + 200 = 800, 17 + 88 = 107, total 907",
    )

    assert verdict is not None and not verdict.correct
    assert verdict.predicted == 907.0 and verdict.expected == 905.0


def test_scratchpad_is_scored_on_its_final_number():
    """Taking the first number would score the working, not the answer."""

    verdict = check.check("Quick question: 524 - 305", "500 - 300 = 200, 24 - 5 = 19, total 219")

    assert verdict.correct


# -- refusing to judge ------------------------------------------------------


def test_a_chat_question_is_not_checked():
    """NOT CHECKED must not be reachable as a pass."""

    assert check.check("why is my script failing", "Check the traceback first.") is None


@pytest.mark.parametrize(
    "question",
    ["hello", "what is your name", "tell me a story about the sea", ""],
)
def test_non_maths_questions_return_none(question):
    assert check.parse_question(question) is None


def test_a_reply_with_no_number_is_wrong_not_unchecked():
    """The question was checkable; the reply simply failed to answer it."""

    verdict = check.check("What is 25% of 840?", "Synonym: glad.")

    assert verdict is not None
    assert verdict.predicted is None and not verdict.correct


# -- parser precedence ------------------------------------------------------


def test_word_problem_is_not_read_as_bare_arithmetic():
    """It contains three numbers; a naive 'a + b' search would seize on them."""

    parsed = check.parse_question(
        "A student has 45 marbles. They get 38 more and then give away 27. "
        "How many marbles do they have now?"
    )

    assert parsed[0] == "word_problem"
    assert parsed[1] == 56.0


def test_average_is_not_read_as_bare_arithmetic():
    parsed = check.parse_question("Find the average (mean) of these numbers: 40, 60, 20, 80")

    assert parsed[0] == "average"


def test_algebra_is_not_read_as_bare_arithmetic():
    """`x + 14 = 39` contains '+' but the answer is 25, not 53."""

    parsed = check.parse_question("Solve for x: x + 14 = 39")

    assert parsed[0] == "algebra_one_step"
    assert parsed[1] == 25.0


# -- agreement with the offline benchmark -----------------------------------


def test_extract_answer_matches_the_benchmark():
    """The live check and the benchmark must not disagree about a reply."""

    import eval_problem_solving as offline

    for reply in ("79", "376 - 17 = 359", "Calculate directly. Answer: 10.68.", "9/14"):
        assert check.extract_answer(reply) == offline.extract_answer(reply)


def test_tolerance_matches_the_benchmark():
    import eval_problem_solving as offline

    assert check.TOLERANCE == offline.TOLERANCE


def test_supported_shapes_are_all_parseable():
    """The interface advertises these; every one must actually verify."""

    for example in check.supported_shapes():
        assert check.parse_question(example) is not None


# -- the v74 task types (v76) -----------------------------------------------
#
# v74 added multiplication, division, sequence and two-step, and scores
# 1.00/1.00/0.98/0.98 on them. Until these parsers existed the live checker
# reported NOT CHECKED for the model's four strongest tasks.


def test_multiplication_is_checked():
    result = check.parse_question("What is 25 x 7?")

    assert result == ("multiplication", 175.0)


def test_multiplication_accepts_an_asterisk():
    assert check.parse_question("What is 25 * 7?")[1] == 175.0


def test_division_is_checked():
    assert check.parse_question("Quick question: 70 / 5") == ("division", 14.0)


def test_division_by_zero_is_not_checkable_rather_than_an_error():
    assert check.parse_question("Quick question: 70 / 0") is None


def test_sequence_is_checked():
    result = check.parse_question("What comes next in the sequence: 7, 17, 27, 37?")

    assert result == ("sequence", 47.0)


def test_sequence_with_uneven_steps_is_not_checkable():
    """Guessing a rule would invent a right answer where none was derived."""

    assert check.parse_question("What comes next in the sequence: 1, 2, 4, 8?") is None


def test_sequence_needs_at_least_three_terms():
    assert check.parse_question("What comes next in the sequence: 5, 10?") is None


def test_two_step_is_checked():
    result = check.parse_question("What is 50% of 698, then add 28?")

    assert result[0] == "two_step"
    assert result[1] == pytest.approx(377.0)


def test_two_step_subtract():
    result = check.parse_question("What is 50% of 200, then subtract 30?")

    assert result[1] == pytest.approx(70.0)


def test_two_step_beats_percent_in_parser_order():
    """A two-step question contains a percent question; percent must not win."""

    assert check.parse_question("What is 20% of 150, then add 12?")[0] == "two_step"


def test_algebra_still_wins_over_multiplication():
    """`x` is both the multiplication sign and the unknown in this corpus."""

    assert check.parse_question("Solve for x: x + 14 = 39")[0] == "algebra_one_step"


def test_sequence_wins_over_average():
    """Both are comma-separated number lists."""

    assert check.parse_question(
        "What comes next in the sequence: 7, 17, 27, 37?"
    )[0] == "sequence"


def test_average_still_parses_as_average():
    assert check.parse_question(
        "Find the average (mean) of these numbers: 40, 60, 20, 80"
    ) == ("average", 50.0)


def test_addition_still_parses_as_arithmetic():
    assert check.parse_question("Solve this basic math problem: 617 + 288")[0] == "arithmetic"


def test_every_advertised_shape_actually_parses():
    """A shape the interface advertises but cannot verify would be a lie."""

    for shape in check.supported_shapes():
        assert check.parse_question(shape) is not None, shape


def test_conversation_is_still_not_checkable():
    for text in ("hello", "why is my script failing", "what is your name"):
        assert check.check(text, "anything") is None


# -- the v79/v80 science shapes (v82) ---------------------------------------
#
# The science parsers shipped in v81 with no tests at all: `PARSERS` handled
# twelve science and mathematics shapes while `supported_shapes()` still
# advertised nine arithmetic ones, so nothing checked that the science laws
# were even the right way up. `momentum = mass / velocity` would have passed
# the whole suite.


@pytest.mark.parametrize(
    "question,task,expected",
    [
        # physics -- each is a phrasing `build_omni_corpus` actually emits
        ("Given mass 25 kg and acceleration 4 m/s^2, compute the force.", "force", 100.0),
        ("A 12 kg mass accelerates at 3 m/s^2. What is the force?", "force", 36.0),
        ("force 10640 N mass 190 kg find acceleration", "acceleration", 56.0),
        ("A force of 580 N acts on a mass of 116 kg. What is the acceleration?",
         "acceleration", 5.0),
        ("Find the acceleration produced by 580 N on 116 kg.", "acceleration", 5.0),
        ("mass 98 kg velocity 3 m/s find momentum", "momentum", 294.0),
        ("What is the kinetic energy of a 12 kg mass moving at 5 m/s?",
         "kinetic_energy", 150.0),
        ("Find the work done by 98 N acting over 2 m.", "work", 196.0),
        ("force 30 N distance 4 m work done", "work", 120.0),
        ("work 1860 J time 20 s power", "power", 93.0),
        ("What power corresponds to 1860 joules in 20 seconds?", "power", 93.0),
        ("A current of 5 A flows through a resistance of 57 ohms. What is the voltage?",
         "voltage", 285.0),
        ("A device runs at 12 V drawing 3 A. What is the electrical power?",
         "electrical_power", 36.0),
        ("A wave with frequency 40 Hz has wavelength 6 m. What is its speed?",
         "wave_speed", 240.0),
        # chemistry
        ("What is the molarity of 4 mol of solute dissolved in 2 L?", "molarity", 2.0),
        # mathematics
        ("In how many ways can 2 items be chosen from 30?", "combination", 435.0),
        ("An arithmetic series starts at 15 with common difference 4. "
         "What is the sum of the first 8 terms?", "arithmetic_series", 232.0),
        ("sum of arithmetic series first term 15 common difference 4 n 8",
         "arithmetic_series", 232.0),
    ],
)
def test_science_ground_truth_is_recomputed_from_the_question(question, task, expected):
    parsed = check.parse_question(question)

    assert parsed is not None, question
    assert parsed[0] == task
    assert parsed[1] == pytest.approx(expected)


def test_kinetic_energy_is_not_read_as_momentum():
    """Both name a mass and a velocity; only the wording separates them."""

    assert check.parse_question(
        "What is the kinetic energy of a 12 kg mass moving at 5 m/s?"
    )[0] == "kinetic_energy"


def test_acceleration_divides_rather_than_multiplies():
    """A law the wrong way up would have passed every pre-v82 test."""

    assert check.parse_question("force 100 N mass 4 kg find acceleration")[1] == 25.0


def test_power_divides_rather_than_multiplies():
    assert check.parse_question("work 100 J time 4 s power")[1] == 25.0


def test_a_science_question_missing_a_quantity_is_not_checkable():
    """Half a physics question must be NOT CHECKED, never a guessed verdict."""

    assert check.parse_question("A force acts on a mass. What is the acceleration?") is None


def test_a_wrong_science_reply_is_marked_wrong():
    verdict = check.check("mass 98 kg velocity 3 m/s find momentum",
                          "momentum = mass x velocity, total 291")

    assert verdict is not None and not verdict.correct
    assert verdict.expected == 294.0 and verdict.predicted == 291.0


def test_a_correct_science_reply_is_marked_correct():
    verdict = check.check(
        "mass 98 kg velocity 3 m/s find momentum",
        "momentum = mass x velocity, 90 x 3 = 270, 8 x 3 = 24, 270 + 24 = 294, total 294",
    )

    assert verdict is not None and verdict.correct


def test_combination_reads_k_rather_than_assuming_two():
    """The corpus fixes k=2; assuming it would be a confident wrong verdict."""

    assert check.parse_question("10 choose 3")[1] == 120.0


def test_combination_with_k_greater_than_n_is_not_checkable():
    assert check.parse_question("30 choose 40") is None


# -- the compound-expression trap (v82) -------------------------------------
#
# `answer_check` recomputes an answer from a *pattern*, not from a parse of the
# question's meaning. "What is 2 + 3 * 4?" once matched the bare `A * B` search
# and returned multiplication with expected 12.0, where precedence makes the
# truth 14. That is the exact failure this module must never produce, and it is
# also the reason it is not in `nexus_epistemics.ANSWER_VERIFIER_IDS`.


def test_a_compound_expression_is_refused_rather_than_answered_wrongly():
    assert check.parse_question("What is 2 + 3 * 4?") is None
    assert check.check("What is 2 + 3 * 4?", "14") is None


@pytest.mark.parametrize(
    "question",
    ["What is 10 - 2 - 3?", "Compute 6 / 2 + 1", "8 x 2 x 3"],
)
def test_every_chained_expression_is_refused(question):
    assert check.parse_question(question) is None


def test_a_single_operator_expression_is_still_checked():
    """The guard must not cost the nine shapes it was added to protect."""

    assert check.parse_question("Solve this basic math problem: 617 + 288")[1] == 905.0
    assert check.parse_question("Quick question: 70 / 5")[1] == 14.0
    assert check.parse_question("What is 25 x 7?")[1] == 175.0


def test_units_containing_a_slash_are_not_read_as_a_second_operator():
    """`4 m/s^2` would break the guard if it counted non-numeric operators."""

    assert check.parse_question(
        "Given mass 25 kg and acceleration 4 m/s^2, compute the force."
    ) == ("force", 100.0)


def test_answer_check_is_not_on_the_verifier_allowlist():
    """Documented, and pinned: this module may not certify an answer.

    `nexus_epistemics` gates certification on an explicit allowlist. If
    `answer_check` were ever added to it, the compound-expression trap above
    would become a false CORRECT on a live reply.
    """

    epistemics = pytest.importorskip("nexus_epistemics")

    assert not any("answer_check" in entry for entry in epistemics.ANSWER_VERIFIER_IDS)


# -- the advertised shapes (v82) --------------------------------------------


def test_supported_shapes_advertises_every_task_family():
    """It listed nine of twenty-one until v82, under-selling the checker.

    v87 adds three code-tracing shapes. They all parse as one task name,
    `code_trace`, because a single parser covers all nine code tasks: it does
    not re-implement each one, it runs the snippet the question contains. So
    the shape count grows by three while the task set grows by one.
    """

    shapes = check.supported_shapes()
    tasks = {check.parse_question(shape)[0] for shape in shapes}

    assert len(shapes) == 24
    assert tasks == {
        "arithmetic", "percent", "algebra_one_step", "word_problem", "average",
        "multiplication", "division", "sequence", "two_step",
        "force", "acceleration", "momentum", "kinetic_energy", "work", "power",
        "voltage", "electrical_power", "wave_speed", "molarity",
        "combination", "arithmetic_series",
        "code_trace",
    }


def test_advertised_shapes_match_the_benchmark_task_list():
    """A shape family the benchmark scores but the interface cannot check.

    The nine `code_*` benchmark tasks are all covered by the single
    `code_trace` parser, so they are checked without appearing under their own
    names. That is verified directly below rather than assumed from the naming.
    """

    import eval_problem_solving as offline

    advertised = {check.parse_question(shape)[0] for shape in check.supported_shapes()}
    code_tasks = set(getattr(offline, "CODE_TASKS", []))

    assert set(offline.GENERATORS) - advertised - code_tasks == set()


def test_every_code_task_is_actually_checkable():
    """The nine code tasks must be judged, not merely excused above.

    Measured 108 checked / 0 wrong / 0 unchecked at twelve samples each when
    this was written. A code question the checker cannot parse would show
    NOT CHECKED in the interface, which is safe but useless.
    """

    import random

    import eval_problem_solving as offline

    if not getattr(offline, "CODE_TASKS", None):
        pytest.skip("code corpus not present")

    rng = random.Random(3)
    for name in offline.CODE_TASKS:
        for _ in range(6):
            problem = offline.GENERATORS[name](rng)
            verdict = check.check(problem.prompt, f"... total {problem.answer:g}")
            assert verdict is not None, f"{name}: NOT CHECKED for {problem.prompt!r}"
            assert verdict.correct, (
                f"{name}: the checker disagreed with the generator's own answer "
                f"({verdict.expected} vs {problem.answer}) for {problem.prompt!r}"
            )
