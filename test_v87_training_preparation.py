"""Controlled curricula, frozen semantic groups and honest process evidence."""
import copy
import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent / "source"))
import build_scratchpad_math as scratch
import eval_prompt_robustness as evaluation
import prepare_v87_training as prep
import v87_reasoning as reasoning


def source_rows():
    rng = random.Random(86)
    rows = []
    for _ in range(12):
        for generator in (scratch._scratchpad_average, scratch._scratchpad_two_step):
            item = generator(rng)
            rows.append({"task": item["task"], "user": item["expression"], "assistant": item["working"]})
    rows.append({"task": "other", "user": "Hello", "assistant": "Hello there."})
    return rows


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(scratch, "AVERAGE_BINARY_STEPS", False)
    monkeypatch.setattr(scratch, "TWO_STEP_DIVISION_TRACE", False)
    monkeypatch.setattr(scratch, "PROMPT_PARAPHRASES", False)
    path = tmp_path / "source.jsonl"
    rows = source_rows()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


@pytest.mark.parametrize("arm", prep.ARMS)
def test_preparation_changes_only_selected_surface_and_keeps_exam_frozen(corpus, tmp_path, arm):
    source, original = corpus
    output = tmp_path / arm
    manifest = prep.prepare(source, output, expected_source_sha256=prep.sha256_file(source), arm=arm, per_family=3)
    written = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    assert len(written) == len(original)
    assert written[-1] == original[-1]
    for before, after in zip(original[:-1], written[:-1]):
        case = prep.row_case(before)
        assert after["task"] == before["task"]
        assert reasoning.verify_working(case, after["assistant"])["process_correct"]
        if arm not in ("combined", before["task"]):
            assert after["assistant"] == before["assistant"]
        if arm not in ("combined", "paraphrases"):
            assert after["user"] == before["user"]
    if arm == "control":
        assert source.read_bytes() == (output / "train.jsonl").read_bytes()
    assert not manifest["training_started"] and not manifest["promotion_authorized"]
    manifest_again, rows = evaluation.load_evaluation(output)
    assert manifest_again == manifest
    occupied = {reasoning.group_id(prep.row_case(row)) for row in original[:-1]}
    assert all(row["group_id"] not in occupied for row in rows)
    assert rows == prep.freeze_cases(occupied, 3, 10087)
    assert manifest["trainer_args"].count("--accuracy_task") == 21


def test_preparation_rejects_changed_source_and_output_overwrite(corpus, tmp_path):
    source, _ = corpus
    target = tmp_path / "bundle"
    with pytest.raises(ValueError, match="SHA256"):
        prep.prepare(source, target, expected_source_sha256="0" * 64)
    assert not target.exists()
    target.mkdir()
    with pytest.raises(FileExistsError):
        prep.prepare(source, target, expected_source_sha256=prep.sha256_file(source))


def test_rehearsal_excludes_groups_from_even_unsampled_rows(corpus, tmp_path):
    source, original = corpus
    target = tmp_path / "bundle"
    manifest = prep.prepare(source, target, expected_source_sha256=prep.sha256_file(source), limit_per_task=1, per_family=4)
    assert manifest["rehearsal"]
    assert sum(manifest["rows_by_task"].values()) == 3
    assert manifest["source_semantic_groups"] == len({reasoning.group_id(prep.row_case(row)) for row in original[:-1]})


def test_semantic_group_identity_ignores_permutation_and_operation_contrast():
    a = {"task": "average", "values": [5, 10, 15, 20]}
    assert reasoning.group_id(a) == reasoning.group_id({**a, "values": [20, 5, 10, 15]})
    b = {"task": "two_step", "pct": 25, "base": 320, "delta": 17, "op": "add"}
    assert reasoning.group_id(b) == reasoning.group_id({**b, "op": "subtract"})
    assert reasoning.expected_answer(b) != reasoning.expected_answer({**b, "op": "subtract"})


@pytest.mark.parametrize("mutation", ["prompt", "duplicate", "missing_variant", "missing_contrast", "variant", "case"])
def test_frozen_validation_rejects_tampering_and_incomplete_groups(mutation):
    rows = prep.freeze_cases(set(), 2, 92)
    if mutation == "prompt":
        rows[0]["prompt"] += " and add ten"
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "missing_variant":
        rows.pop(0)
    elif mutation == "missing_contrast":
        rows = [r for r in rows if r["case"].get("op") != "subtract"]
    elif mutation == "variant":
        rows[1]["variant"] = "two_step.eval.1"
    else:
        rows[0]["case"]["values"][0] += 1
    with pytest.raises(ValueError):
        prep.validate_frozen(rows)


def test_correct_final_answer_cannot_hide_incorrect_or_unrelated_working():
    case = {"task": "average", "values": [10, 20, 30, 40]}
    correct = reasoning.render_working(case)
    assert reasoning.verify_working(case, correct)["process_correct"]
    wrong_step = correct.replace("10 + 20 = 30", "10 + 20 = 31")
    report = reasoning.verify_working(case, wrong_step)
    assert report["answer_correct"] and not report["process_correct"]
    assert report["first_error"] == 0
    wrong_operands = correct.replace("10 + 20", "11 + 19")
    assert not reasoning.verify_working(case, wrong_operands)["process_correct"]
    assert not reasoning.verify_working(case, "2 + 2 = 4. " + correct)["process_correct"]
    assert not reasoning.verify_working(case, "total 25")["process_correct"]


@pytest.mark.parametrize("pct", [10, 20, 25, 50])
@pytest.mark.parametrize("op", ["add", "subtract"])
def test_fraction_trace_checks_quantity_operation_and_dependency(pct, op):
    case = {"task": "two_step", "pct": pct, "base": 320, "delta": 17, "op": op}
    trace = reasoning.render_working(case)
    assert reasoning.verify_working(case, trace)["process_correct"]
    assert not reasoning.verify_working({**case, "op": "subtract" if op == "add" else "add"}, trace)["process_correct"]
    assert not reasoning.verify_working(case, trace.replace("320 /", "321 /"))["process_correct"]


def scored_report(correct=True, groups=3):
    rows = prep.freeze_cases(set(), groups, 91)
    for row in rows:
        row["reply"] = reasoning.render_working(row["case"]) if correct else "total -10000"
        row["correct"] = correct
        row["process"] = reasoning.verify_working(row["case"], row["reply"])
        row["hit_token_cap"] = False
    return {"schema": evaluation.REPORT_SCHEMA, "settings": {"cap": 96}, "evaluation_sha256": "test",
            "scoring_sha256": "test", "complete": True, "results": rows}


def test_group_metrics_do_not_treat_correlated_paraphrases_as_independent():
    good, bad = scored_report(), scored_report(False)
    summary = evaluation.summarize(good["results"])
    assert summary["two_step"]["items"] == 24
    assert len(summary["two_step"]["groups"]) == 3
    comparison = evaluation.compare_reports(good, bad)
    assert not comparison["promotion_authorized"]
    for row in comparison["by_task"].values():
        assert row["paired_groups"] == 3
        assert row["all_variant_group_accuracy_delta"] == 1
        assert row["group_bootstrap_95_interval"] is None
    good["results"][0]["correct"] = False
    with pytest.raises(ValueError, match="altered scoring"):
        evaluation.compare_reports(good, bad)


def test_paired_comparison_rejects_setting_drift_and_order_changes():
    first, second = scored_report(), scored_report()
    second["settings"]["cap"] = 64
    with pytest.raises(ValueError, match="settings"):
        evaluation.compare_reports(first, second)
    second = scored_report()
    second["results"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        evaluation.compare_reports(first, second)


def test_group_limit_preserves_all_paraphrases_and_both_operations():
    rows = evaluation.select_groups(prep.freeze_cases(set(), 4, 92), 1)
    prep.validate_frozen(rows)
    assert len(rows) == 12


def test_preflight_validates_the_complete_recipe_and_rehearsal_scope(corpus, tmp_path):
    import run_v87_training as launcher

    source, _ = corpus
    bundle = tmp_path / "bundle"
    prep.prepare(source, bundle, expected_source_sha256=prep.sha256_file(source), per_family=3, limit_per_task=4)
    report, command = launcher.preflight(bundle)
    assert report["checked"] and report["rehearsal"]
    assert not report["training_launch_ready"] and not report["training_started"]
    assert "--frozen_split" in command
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["trainer_args"].extend(["--seed", "999"])
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="arguments changed"):
        launcher.preflight(bundle)


def test_full_arms_have_identical_source_partition_membership(corpus, tmp_path):
    from v87_frozen_split import load_frozen_split

    source, _ = corpus
    manifests, proofs = [], []
    for arm in ("control", "combined"):
        output = tmp_path / arm
        manifests.append(prep.prepare(source, output, expected_source_sha256=prep.sha256_file(source), arm=arm, per_family=3))
        _, proof = load_frozen_split(output / "train.jsonl", output / "frozen_split.json")
        proofs.append(proof)
    assert manifests[0]["partition_sha256"] == manifests[1]["partition_sha256"]
    assert manifests[0]["evaluation_sha256"] == manifests[1]["evaluation_sha256"]
    for key in ("actual_train_rows", "actual_dev_rows", "actual_test_rows", "source_group_sequence_sha256"):
        assert proofs[0][key] == proofs[1][key]


def test_frozen_gold_working_fits_the_training_context():
    from mimomix_text import WordTokenizer

    tokenizer = WordTokenizer([], digit_tokens=True)
    rows = prep.freeze_cases(set(), 100, 99)
    for row in rows:
        working = reasoning.render_working(row["case"])
        assert len(tokenizer.encode_turn(row["prompt"], working)[0]) <= 128
        assert len(tokenizer.pattern.findall(working)) <= 96
