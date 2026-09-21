"""The v93 corpus families and the benchmark wiring that scores them.

docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D7, adds eleven tasks to a corpus
whose existing thirty must not move: v93 warm-starts from v89 and is scored
against v89 by exact McNemar, so any change to a v89 generator or to the
problems the benchmark draws for it unpairs every published receipt. These
tests pin what has to stay fixed and check what is new the way its own
verifier does:

* the five new omni tasks re-derive through `nexus_solver` (a canonical query
  the solver cannot parse drops the whole task silently at build time),
* the three new code tasks re-derive by executing their snippet,
* the three connectome tasks equal the CC-BY arrays they were read from --
  the side table in particular, whose first reader compared an int8 with the
  string "L" and returned every side count as 0,
* every prompt fits the 32-token half of the 128-position context beside the
  benchmark's 96-token generation cap, measured with v89's own tokenizer,
* `--task_set v89` reproduces fingerprint 3b99a446cd533be9bc5f8ae57d1310b4
  and the 21-task fingerprint 4077062251bc762c9716a730f3818ad2, and
  registering the new families leaves every existing task's problems intact,
* the live checker parses every new shape and never disagrees with a
  generator, and the coverage audit names the lookup tasks as uncompared
  with a reason rather than in silence.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import answer_check as check  # noqa: E402
import build_code_corpus as code  # noqa: E402
import build_connectome_corpus as cns  # noqa: E402
import build_omni_corpus as omni  # noqa: E402
import build_v93_corpus as v93  # noqa: E402
import coverage_audit as coverage  # noqa: E402
import eval_problem_solving as solving  # noqa: E402
import natural_phrasings as phrasings  # noqa: E402

V89_FINGERPRINT = "3b99a446cd533be9bc5f8ae57d1310b4"
V89_21_TASK_FINGERPRINT = "4077062251bc762c9716a730f3818ad2"

#: The benchmark prompt ceiling: 32 + DEFAULT_MAX_NEW_TOKENS 96 = the 128
#: positions the model was trained to (test_eval_v82.py pins the sum).
PROMPT_CEILING = 32
REPLY_CEILING = solving.DEFAULT_MAX_NEW_TOKENS

V89_CHECKPOINT = ROOT / "output" / "v89_corpus" / "v89_corpus.pt"
DATA_PRESENT = cns.data_available()


@pytest.fixture(autouse=True)
def _restore_flags():
    was = (omni.NATURAL_PHRASINGS, omni.LONG_DIVISION)
    yield
    omni.NATURAL_PHRASINGS, omni.LONG_DIVISION = was


@pytest.fixture(scope="module")
def v89_tokenizer():
    if not V89_CHECKPOINT.is_file():
        pytest.skip("v89 checkpoint not present")
    return v93.load_v89_tokenizer(V89_CHECKPOINT)


def _needs_data():
    if not DATA_PRESENT:
        pytest.skip("connectome tables not present")


# ---------------------------------------------------------------------------
# The generators verify
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(omni.V93_TASKS))
def test_every_new_omni_task_verifies_against_the_solver(name):
    """0 of 200 is what `final_velocity` scored before the bridge fix.

    An unparseable canonical query does not raise: `build` counts the row
    under `dropped_failing_verification` and moves on, so a task can be
    silently absent from a corpus that reports success. 200 of 200 here, on
    both the plain and the `--long_division` written forms.
    """

    for long_division in (False, True):
        omni.LONG_DIVISION = long_division
        rng = random.Random(93)
        for _ in range(200):
            problem = omni.V93_TASKS[name](rng)
            assert omni.verify(problem), (name, problem.canonical, problem.answer)
            assert omni.extract_answer(problem.response) == problem.answer, problem.response


@pytest.mark.parametrize("name", sorted(code.V93_TASKS))
def test_every_new_code_task_verifies_by_execution(name):
    rng = random.Random(93)
    for _ in range(200):
        problem = code.V93_TASKS[name](rng)
        verdict = code.verify(problem, code.DEFAULT_TIMEOUT_SECONDS)
        assert verdict.ok, (name, problem.canonical, verdict.reason)
        assert code.extract_answer(problem.response) == problem.answer, problem.response


def test_the_new_code_syntax_is_admitted_and_nothing_wider():
    """`.count` came in through `SAFE_METHODS`; the escape shapes stay out."""

    assert code.run_snippet("nums = [2, 3, 2]\nr = nums.count(2)", "r").value == 2
    assert code.run_snippet("r = sum(range(3, 9))", "r").value == 33
    assert code.run_snippet("nums = [13, 8, 10]\nr = nums[-2]", "r").value == 8
    for hostile in ("r = [].__class__", "r = (1).__class__", "nums = [1]\nr = nums.pop()",
                    "r = ''.join(['a'])"):
        assert not code.run_snippet(hostile, "r").ok, hostile


def test_connectome_tasks_verify_by_lookup_and_equal_the_arrays():
    """The answer in the row is the number in the npz, read here independently."""

    _needs_data()
    import numpy as np

    types = np.load(cns.DEFAULT_TYPES, allow_pickle=True)
    names = [str(n) for n in types["type_names"]]
    index = {n: i for i, n in enumerate(names)}
    n_neurons = types["n_neurons"].astype(int)
    pair_weight = {
        (names[a], names[b]): int(w)
        for a, b, w in zip(types["pre"].astype(int), types["post"].astype(int),
                           types["weight"].astype(int))
    }
    sided = np.load(cns.DEFAULT_SIDED, allow_pickle=True)
    sided_names = [str(n) for n in sided["type_names"]]
    side_count = {}
    for t, s, c in zip(sided["node_type"].astype(int), sided["node_side"].astype(int),
                       sided["node_n_neurons"].astype(int)):
        key = (sided_names[t], "left" if s == 0 else "right")
        side_count[key] = side_count.get(key, 0) + int(c)

    rng = random.Random(5)
    for _ in range(300):
        problem = cns.TASKS["cns_type_count"](rng)
        assert cns.verify(problem)
        assert problem.answer == float(n_neurons[index[problem.params["type"]]])

        problem = cns.TASKS["cns_side_count"](rng)
        assert cns.verify(problem)
        key = (problem.params["type"], problem.params["side"])
        assert problem.answer == float(side_count.get(key, 0))

        problem = cns.TASKS["cns_pair_synapses"](rng)
        assert cns.verify(problem)
        assert problem.answer == float(pair_weight[(problem.params["pre"], problem.params["post"])])


def test_the_sided_table_reader_decodes_the_d1_layout():
    """node_type is an index and node_side an int8; strings match nothing.

    The first reader returned 0 for every type on every side and nothing
    downstream objected. The receipt's totals are the design document's
    L 80,786 / R 82,932, and a table that yields no sided neurons is refused.
    """

    _needs_data()
    left, right, receipt = cns.side_counts_from_sided_table(cns.DEFAULT_SIDED)
    assert receipt["neurons_left"] == 80786
    assert receipt["neurons_right"] == 82932
    population = cns.population()
    assert population.receipt["sides"]["neurons_left"] == 80786
    zero_sided = population.receipt["types_with_zero_on_one_side"]
    assert zero_sided < len(population.names) // 10, zero_sided

    import numpy as np

    with np.load(cns.DEFAULT_SIDED, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["node_side"] = np.full_like(arrays["node_side"], 7)   # no L, no R
    broken = ROOT / "datasets" / "v93" / ".broken_sided_for_test.npz"
    broken.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez(broken, **arrays)
        with pytest.raises(ValueError, match="no neurons on one side"):
            cns.side_counts_from_sided_table(broken)
    finally:
        if broken.exists():
            broken.unlink()


def test_connectome_answers_are_integers_ending_the_reply():
    _needs_data()
    rng = random.Random(11)
    for name, generator in cns.TASKS.items():
        for _ in range(50):
            problem = generator(rng)
            assert problem.response.endswith(f"total {int(problem.answer)}"), problem.response
            assert problem.answer == int(problem.answer)


# ---------------------------------------------------------------------------
# Budgets, with v89's tokenizer
# ---------------------------------------------------------------------------


def _new_problems(draws: int, natural: bool):
    omni.NATURAL_PHRASINGS = natural
    omni.LONG_DIVISION = True
    rng = random.Random(23)
    problems = []
    for name, generator in sorted(omni.V93_TASKS.items()):
        problems += [(name, generator(rng)) for _ in range(draws)]
    for name, generator in sorted(code.V93_TASKS.items()):
        problems += [(name, generator(rng)) for _ in range(draws)]
    if DATA_PRESENT:
        for name, generator in sorted(cns.TASKS.items()):
            problems += [(name, generator(rng)) for _ in range(draws)]
    return problems


def test_every_new_prompt_fits_beside_the_generation_cap(v89_tokenizer):
    """Prompt + 96 generated <= 128 positions: the geometry test_eval_v82 pins."""

    phrasings_were = phrasings.HELD_OUT_PER_TASK
    try:
        # Held-out forms included: eval_natural_phrasing asks them live.
        phrasings.HELD_OUT_PER_TASK = 0
        worst = {}
        for natural in (False, True):
            for name, problem in _new_problems(150, natural):
                length = len(v89_tokenizer.encode_turn(problem.prompt, None)[0])
                if length > worst.get(name, (0, ""))[0]:
                    worst[name] = (length, problem.prompt)
    finally:
        phrasings.HELD_OUT_PER_TASK = phrasings_were
    over = {k: v for k, v in worst.items() if v[0] > PROMPT_CEILING}
    assert not over, over


def test_every_new_reply_fits_the_generation_cap(v89_tokenizer):
    """Reply tokens plus the EOS the model must emit, as the trainer counts."""

    worst = {}
    for name, problem in _new_problems(200, False):
        length = len(v89_tokenizer.encode(problem.response)) + 1
        if length > worst.get(name, (0, ""))[0]:
            worst[name] = (length, problem.response)
    over = {k: v for k, v in worst.items() if v[0] > REPLY_CEILING}
    assert not over, over
    # The one task built against the cap: its bound is the cap, not below it.
    assert worst["code_range_sum"][0] <= REPLY_CEILING


def test_every_new_turn_fits_the_sequence_budget(v89_tokenizer):
    """Turn-aligned packing drops a longer turn silently; language rows too."""

    lengths = []
    for _, problem in _new_problems(150, True):
        lengths.append(len(v89_tokenizer.encode_turn(problem.prompt, problem.response)[0]))
    if DATA_PRESENT:
        rng = random.Random(3)
        for _ in range(400):
            row = cns.language_row(rng)
            lengths.append(len(v89_tokenizer.encode_turn(row["user"], row["assistant"])[0]))
    english, _ = v93.english_rows()
    for row in english[::10]:
        lengths.append(len(v89_tokenizer.encode_turn(row["user"], row["assistant"])[0]))
    assert max(lengths) <= omni.DEFAULT_SEQUENCE_LENGTH, max(lengths)


# ---------------------------------------------------------------------------
# Benchmark wiring
# ---------------------------------------------------------------------------


def test_the_v89_task_set_reproduces_the_published_fingerprint():
    assert solving.generator_fingerprint(solving.V89_BENCHMARK_TASKS) == V89_FINGERPRINT
    assert solving.expand_task_set("v89") == list(solving.V89_BENCHMARK_TASKS)
    assert len(solving.V89_BENCHMARK_TASKS) == 30


def test_the_21_task_fingerprint_pinned_by_test_code_corpus_still_holds():
    """The v80/v86 baseline: the thirty minus the nine v89 code tasks."""

    original = [t for t in solving.V89_BENCHMARK_TASKS if not t.startswith("code_")]
    assert len(original) == 21
    assert solving.generator_fingerprint(original) == V89_21_TASK_FINGERPRINT


def test_the_v89_names_are_the_registry_prefix_and_new_tasks_follow():
    registered = list(solving.GENERATORS)
    assert tuple(registered[:30]) == solving.V89_BENCHMARK_TASKS
    assert registered == solving.expand_task_set("all")[:len(registered)]
    expected_new = [n for n in solving.V93_NEW_TASKS
                    if DATA_PRESENT or not n.startswith("cns_")]
    assert registered[30:] == expected_new
    assert solving.V93_REGISTERED_TASKS == expected_new
    assert set(solving.V93_NEW_TASKS) == (
        set(omni.V93_TASKS) | set(code.V93_TASKS) | set(cns.TASKS))


def test_registration_does_not_move_any_existing_tasks_problems():
    """Per-task RNGs: 630 over the v89 thirty is the same 21 per task as ever."""

    v89_only = solving.generate_novel(630, seed=65, tasks=list(solving.V89_BENCHMARK_TASKS))
    everything = solving.generate_novel(630 * 2, seed=65)
    by_task_v89 = {}
    by_task_all = {}
    for problem in v89_only:
        by_task_v89.setdefault(problem.task, []).append((problem.prompt, problem.answer))
    for problem in everything:
        by_task_all.setdefault(problem.task, []).append((problem.prompt, problem.answer))
    for task in solving.V89_BENCHMARK_TASKS:
        shared = min(len(by_task_v89[task]), len(by_task_all[task]))
        assert shared >= 21, task
        assert by_task_v89[task][:shared] == by_task_all[task][:shared], task


def test_new_task_fingerprints_are_stable_and_distinct():
    new = [n for n in solving.V93_NEW_TASKS if n in solving.GENERATORS]
    assert solving.generator_fingerprint(new) == solving.generator_fingerprint(new)
    assert solving.generator_fingerprint(new) != V89_FINGERPRINT
    assert solving.generator_fingerprint() != V89_FINGERPRINT, (
        "the default registry is no longer the v89 thirty; --task_set v89 is how to pair"
    )


def test_task_set_and_tasks_are_mutually_exclusive():
    parser = solving.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "x.pt", "--task_set", "v89", "--tasks", "force"])
    args = parser.parse_args(["--checkpoint", "x.pt", "--task_set", "new"])
    assert args.task_set == "new" and args.tasks is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "x.pt", "--task_set", "v93"])
    with pytest.raises(KeyError):
        solving.expand_task_set("nope")


def test_the_new_tasks_draw_from_their_own_name_derived_rng():
    for name in solving.V93_REGISTERED_TASKS:
        first = solving.GENERATORS[name](solving.task_rng(name, 65))
        again = solving.GENERATORS[name](solving.task_rng(name, 65))
        assert (first.prompt, first.answer) == (again.prompt, again.answer)
        assert isinstance(first.answer, float)
        assert first.source == "novel"
        assert first.task == name


def test_non_claims_state_that_the_connectome_tasks_are_recall():
    joined = " ".join(solving.NON_CLAIMS).lower()
    assert "recall" in joined and "cns_type_count" in joined
    assert "same population" in joined
    assert "--task_set v89" in joined


# ---------------------------------------------------------------------------
# The default builds and the frozen registries
# ---------------------------------------------------------------------------


def test_the_v89_generator_tables_are_untouched():
    assert list(omni.TASKS) == list(solving.V89_BENCHMARK_TASKS[9:21])
    assert list(code.TASKS) == list(solving.V89_BENCHMARK_TASKS[21:])
    assert not set(omni.V93_TASKS) & set(omni.TASKS)
    assert not set(code.V93_TASKS) & set(code.TASKS)
    assert set(omni.ALL_TASKS) == set(omni.TASKS) | set(omni.V93_TASKS)
    assert set(code.ALL_TASKS) == set(code.TASKS) | set(code.V93_TASKS)


def test_a_default_omni_build_does_not_contain_a_v93_task():
    rows, report = omni.build(3, 93)
    assert set(report["per_task"]) == set(omni.TASKS)
    rows, report = omni.build(3, 93, tasks=["impulse"])
    assert set(report["per_task"]) == {"impulse"}
    assert report["dropped_failing_verification"] == {}
    with pytest.raises(ValueError, match="unknown task"):
        omni.build(3, 93, tasks=["no_such_task"])


def test_a_default_code_build_does_not_contain_a_v93_task():
    rows, report = code.build(3, 93, None)
    assert set(report["per_task"]) == set(code.TASKS)
    rows, report = code.build(3, 93, ["code_neg_index"])
    assert set(report["per_task"]) == {"code_neg_index"}
    assert report["drop_rate"] == 0.0
    for name in code.V93_TASKS:
        assert name in code.distinct_capacity()


# ---------------------------------------------------------------------------
# Natural phrasings for the new omni tasks
# ---------------------------------------------------------------------------


def test_every_new_omni_task_has_held_out_phrasings_whose_placeholders_match():
    rng = random.Random(3)
    placeholder = __import__("re").compile(r"\{(\w+)\}")
    omni.NATURAL_PHRASINGS = True
    for name, generator in omni.V93_TASKS.items():
        assert phrasings.held_out(name), f"{name} has no held-out phrasing"
        with phrasings.held_out_only():
            generator(rng)   # raises KeyError if a placeholder is not supplied
        needed = {frozenset(placeholder.findall(form))
                  for form in phrasings.EXTRA_PHRASINGS[name]}
        assert len(needed) == 1, (name, needed)


def test_every_widened_new_omni_row_still_verifies():
    omni.NATURAL_PHRASINGS = True
    rng = random.Random(7)
    for name, generator in sorted(omni.V93_TASKS.items()):
        for _ in range(60):
            problem = generator(rng)
            assert omni.verify(problem), (name, problem.prompt)


# ---------------------------------------------------------------------------
# The live checker and the coverage audit
# ---------------------------------------------------------------------------


def test_the_live_checker_parses_every_new_shape_and_never_disagrees():
    """0 unparsed and 0 confident-wrong over narrow and natural forms."""

    phrasings_were = phrasings.HELD_OUT_PER_TASK
    try:
        phrasings.HELD_OUT_PER_TASK = 0
        for natural in (False, True):
            for name, problem in _new_problems(60, natural):
                parsed = check.parse_question(problem.prompt)
                assert parsed is not None, (name, problem.prompt)
                task, expected = parsed
                if name.startswith("code_"):
                    assert task == "code_trace", (name, problem.prompt, task)
                else:
                    assert task == name, (name, problem.prompt, task)
                assert solving.is_correct(expected, problem.answer), (
                    name, problem.prompt, expected, problem.answer)
    finally:
        phrasings.HELD_OUT_PER_TASK = phrasings_were


def test_permutations_are_not_read_as_combinations():
    """`taken` is in both vocabularies; 72 is the answer, 36 the trap."""

    assert check.parse_question(
        "Find the number of permutations of 9 things taken 2 at a time.") == ("permutations", 72.0)
    assert check.parse_question(
        "Find the number of combinations of 9 things taken 2 at a time.") == ("combination", 36.0)


def test_the_checker_refuses_a_type_outside_the_population():
    _needs_data()
    assert check.parse_question(
        "How many neurons of type NOSUCHTYPE9 are in the male CNS?") is None
    assert check.parse_question("How many neurons does a fly have?") is None


def test_the_coverage_audit_names_the_lookup_tasks_with_a_reason():
    # 1,500 samples, the same draw the repo-wide gate in test_coverage_audit
    # uses: final_velocity answers span ~128 values (u 10-60 + a 2-9 x t 2-9),
    # so 50 draws of the corpus generator miss values the benchmark's 50
    # draws happen to ask, and the audit reports a hole that is sampling
    # noise, not a train/eval gap (the two sides are the same function).
    report = coverage.audit(samples=1500, seed=87, tasks=list(solving.V93_REGISTERED_TASKS))
    compared = set(report["tasks"])
    assert compared == {n for n in solving.V93_REGISTERED_TASKS if not n.startswith("cns_")}
    for name in solving.V93_REGISTERED_TASKS:
        if name.startswith("cns_"):
            assert name in report["not_compared"]
            assert "lookup" in report["not_compared_reason"][name]
    for name, entry in report["tasks"].items():
        assert entry["asked_but_never_taught"] == [], (name, entry["asked_but_never_taught"])


# ---------------------------------------------------------------------------
# The assembler
# ---------------------------------------------------------------------------


def test_english_rows_are_relabelled_as_language():
    if not v93.V62_ENGLISH_MATH.is_file():
        pytest.skip("datasets/v62 not present")
    rows, receipt = v93.english_rows()
    assert receipt["rows"] == 9983
    assert receipt["excluded_by_topic"] == {"basic_math": 30017}
    assert all("task" not in row for row in rows)
    assert all(row["domain"] == "english_foundations" for row in rows)
    assert set(receipt["kinds"]) == {
        "punctuation_capitalization", "spelling_fix", "tone_rewrite", "proofread",
        "definition", "sentence_construction", "message_writing", "grammar_fix",
        "synonym_example",
    }


def test_combine_copies_rows_byte_for_byte_and_counts_them(tmp_path):
    part_a = tmp_path / "a.jsonl"
    part_b = tmp_path / "b.jsonl"
    part_a.write_bytes(b'{"user": "u1", "assistant": "a1", "task": "force"}\r\n'
                       b'{"user": "u2", "assistant": "a2", "domain": "dialogue"}\r\n')
    part_b.write_bytes(b'{"user": "u3", "assistant": "a3", "task": "impulse"}\r\n')
    english = [{"user": "u4", "assistant": "a4", "domain": "english_foundations", "kind": "x"}]
    destination = tmp_path / "combined.jsonl"
    counts = v93.combine([("a", part_a), ("b", part_b)], english, destination)
    raw = destination.read_bytes()
    assert raw.startswith(part_a.read_bytes() + part_b.read_bytes())
    assert raw.endswith(b"\r\n")
    assert counts["per_task"] == {"force": 1, "impulse": 1}
    assert counts["language_rows"] == 2
    assert counts["per_source"]["a"] == {"rows": 2, "task_rows": 1, "language_rows": 1}
    assert counts["per_source"]["english_foundations"]["rows"] == 1


def test_new_token_types_follow_the_extend_rule(v89_tokenizer):
    """A piece the base lacks adds its raw and lstripped forms, as `build` does."""

    new_ids, seen, missing = v93.new_token_types(
        v89_tokenizer, ["type KCg-m has 12 neurons", "KCg-m again"])
    assert missing >= 1
    assert " KCg" in new_ids and "KCg" in new_ids
    # A piece the base has does not count, whatever its spacing.
    assert " has" not in new_ids and "has" not in new_ids
    report = v93.vocabulary_report({"x": [{"user": "type KCg-m", "assistant": "12"}]},
                                   v89_tokenizer, cap=10)
    assert report["families"]["x"]["new_token_ids"] == report["union_new_token_ids"]
    assert report["within_cap"] is True


def test_the_built_corpus_manifest_is_consistent():
    """Checked only where the build has run; the build itself is CPU minutes."""

    if not v93.MANIFEST.is_file():
        pytest.skip("datasets/v93 not built")
    manifest = json.loads(v93.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema"] == "supermix-v93-corpus-v1"
    assert manifest["rows"] == manifest["task_rows"] + manifest["language_rows"]
    assert manifest["rows"] == sum(s["rows"] for s in manifest["per_source"].values())
    for name in (*v93.OMNI_V93,):
        assert manifest["per_task"][name] == v93.PER_TASK, name
    for name in v93.CODE_V93:
        assert manifest["per_task"][name] == v93.CODE_PER_TASK, name
    for name in v93.CNS_V93:
        assert manifest["per_task"][name] == v93.CNS_PER_TASK, name
    v89 = json.loads((ROOT / "datasets" / "v89" / "v89_manifest.json").read_text(encoding="utf-8"))
    for name, count in v89["per_task"].items():
        assert manifest["per_task"][name] == count, name
    assert manifest["per_source"]["v89_combined"]["rows"] == v89["rows"]
    assert manifest["language_by_domain"]["dialogue"] == v89["language_rows"]
    assert manifest["language_by_domain"]["english_foundations"] == 9983
    assert manifest["vocabulary"]["union_new_token_ids"] <= manifest["vocabulary"]["cap"]
    for family, entries in manifest["token_budget"].items():
        if not isinstance(entries, dict):
            continue
        for name, entry in entries.items():
            if not isinstance(entry, dict) or "response_p95" not in entry:
                continue
            assert entry["response_p95"] + 1 <= REPLY_CEILING, (name, entry)
            assert entry["dropped_fraction"] == 0.0, (name, entry)
