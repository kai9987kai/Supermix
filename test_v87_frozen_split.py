"""Frozen source groups keep format arms matched after text changes."""

import json
from pathlib import Path
import sys

import pytest

SOURCE = Path(__file__).resolve().parent / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import mimomix_eval_splits as splits
from v87_frozen_split import load_frozen_split, source_group, write_frozen_split
from v87_reasoning import canonical_prompt, render_working


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows():
    rows = []
    for index in range(80):
        case = {"task": "two_step", "pct": 10, "base": 100 + 10 * (index // 2),
                "delta": 5, "op": "add" if index % 2 else "subtract"}
        rows.append({"task": case["task"], "user": canonical_prompt(case),
                     "assistant": render_working(case)})
    for index in range(20):
        case = {"task": "average", "values": [index + 5, 10, 20, 30]}
        rows.append({"task": case["task"], "user": canonical_prompt(case),
                     "assistant": render_working(case)})
    rows.append(dict(rows[4]))
    return rows


def test_transformed_arms_keep_source_groups_and_order(tmp_path):
    rows = _rows()
    source = tmp_path / "original.jsonl"
    changed = tmp_path / "changed.jsonl"
    _write(source, rows)
    transformed = [{**row, "user": "Please solve: " + row["user"]} for row in rows]
    _write(changed, transformed)
    control_path, changed_path = tmp_path / "control.json", tmp_path / "changed.json"
    control_receipt = write_frozen_split(source, source, control_path, seed=58)
    changed_receipt = write_frozen_split(source, changed, changed_path, seed=58)
    assert control_receipt["partition_sha256"] == changed_receipt["partition_sha256"]
    assert control_receipt["source_group_sequence_sha256"] == changed_receipt["source_group_sequence_sha256"]
    control, control_provenance = load_frozen_split(source, control_path)
    treatment, treatment_provenance = load_frozen_split(changed, changed_path)
    assert control_provenance["actual_train_rows"] == treatment_provenance["actual_train_rows"]
    for name in ("train", "dev", "tier1_seen_response", "tier2_unseen_response", "tier3_unseen_sentence"):
        assert [("Please solve: " + user, answer) for user, answer in getattr(control, name)] == getattr(treatment, name)
    positions = {tuple((row["user"], row["assistant"])): i for i, row in reversed(list(enumerate(rows)))}
    unique_positions = [positions[pair] for pair in control.train if pair != (rows[4]["user"], rows[4]["assistant"])]
    assert unique_positions == sorted(unique_positions)
    owners = {}
    pairs_to_groups = {(row["user"], row["assistant"]): source_group(row) for row in rows}
    for partition, selected in [("train", control.train), ("dev", control.dev)] + control.tiers():
        for pair in selected:
            group = pairs_to_groups[pair]
            role = "test" if partition.startswith("tier") else partition
            assert owners.setdefault(group, role) == role


def test_average_permutations_and_operation_contrasts_share_group():
    case = {"task": "average", "values": [5, 10, 20, 30]}
    reverse = {**case, "values": list(reversed(case["values"]))}
    def row(value):
        return {"task": value["task"], "user": canonical_prompt(value), "assistant": render_working(value)}
    assert source_group(row(case)) == source_group(row(reverse))
    case = {"task": "two_step", "pct": 10, "base": 200, "delta": 5, "op": "add"}
    assert source_group(row(case)) == source_group(row({**case, "op": "subtract"}))


@pytest.mark.parametrize("damage", ["corpus", "source", "partition", "filter", "limit"])
def test_hash_and_population_changes_fail_closed(tmp_path, damage):
    rows = _rows()
    source, corpus, recipe = (tmp_path / name for name in ("source.jsonl", "corpus.jsonl", "split.json"))
    _write(source, rows)
    _write(corpus, rows)
    receipt = write_frozen_split(source, corpus, recipe)
    if damage == "corpus":
        _write(corpus, list(reversed(rows)))
    elif damage == "source":
        _write(source, rows[:-1])
    elif damage == "partition":
        receipt["partition_sha256"] = "0" * 64
        recipe.write_text(json.dumps(receipt), encoding="utf-8")
    kwargs = {"min_response_characters": 10000} if damage == "filter" else {"limit": 10} if damage == "limit" else {}
    with pytest.raises(ValueError):
        load_frozen_split(corpus, recipe, **kwargs)


def test_rehearsal_indices_are_original_and_validated(tmp_path):
    rows = _rows()
    source, corpus, recipe = (tmp_path / name for name in ("source.jsonl", "corpus.jsonl", "split.json"))
    selected = list(range(0, 80, 4))
    _write(source, rows)
    _write(corpus, [rows[index] for index in selected])
    receipt = write_frozen_split(source, corpus, recipe, source_row_indices=selected)
    split, provenance = load_frozen_split(corpus, recipe)
    assert receipt["corpus_rows"] == len(selected)
    assert len(split.train) + len(split.dev) + sum(len(part) for _, part in split.tiers()) == len(selected)
    assert provenance["verified"]
    with pytest.raises(ValueError, match="unique, increasing"):
        write_frozen_split(source, corpus, tmp_path / "bad.json", source_row_indices=[4, 0])
    with pytest.raises(ValueError, match="row count"):
        write_frozen_split(source, corpus, tmp_path / "wrong.json")


def test_tiers_are_reclassified_using_actual_transformed_answers(tmp_path):
    rows = [{"user": f"Question {index}?", "assistant": f"Original answer {index}."} for index in range(100)]
    source, corpus, recipe = (tmp_path / name for name in ("source.jsonl", "corpus.jsonl", "split.json"))
    _write(source, rows)
    _write(corpus, [{**row, "assistant": "Shared transformed answer."} for row in rows])
    write_frozen_split(source, corpus, recipe)
    split, _ = load_frozen_split(corpus, recipe)
    assert split.tier1_seen_response
    assert not split.tier2_unseen_response and not split.tier3_unseen_sentence
    assert not split.held_out_sentences
    assert splits.verify_split(split)["tier1_responses_all_in_training"]


def test_trainer_exposes_frozen_split_parser_without_changing_default():
    from train_mimomix_generalisation import build_parser
    assert build_parser().parse_args([]).frozen_split is None
    assert build_parser().parse_args(["--frozen_split", "split.json"]).frozen_split == "split.json"
