import copy
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from prepare_v94 import freeze_cohorts, prepare


def rows(prefix="train"):
    return [{"row_id": f"{prefix}-{family}-{i}-{variant}", "semantic_id": f"{prefix}-{family}-{i}",
             "family": family, "prompt": f"{prefix} {family} problem {i} wording {variant}",
             "response": str(i)} for family in ("old", "new") for i in range(10) for variant in range(2)]


def test_frozen_cohorts_are_order_independent_group_disjoint_and_bound_to_content():
    train, evaluation = rows(), rows("eval")
    manifest = freeze_cohorts(train, evaluation)
    assert manifest == freeze_cohorts(train[::-1], evaluation[::-1])
    group_roles = {}
    for role, cohort in manifest["cohorts"].items():
        assert len(cohort["membership_sha256"]) == 64
        assert {r["family"] for r in cohort["rows"]} == {"old", "new"}
        for row in cohort["rows"]:
            group_roles.setdefault(row["semantic_id"], set()).add(role)
    assert all(len(roles) == 1 for roles in group_roles.values())
    train[0]["response"] = "changed answer"
    assert manifest["manifest_sha256"] != freeze_cohorts(train, evaluation)["manifest_sha256"]


@pytest.mark.parametrize("field", ["row_id", "semantic_id", "prompt"])
def test_rejects_training_evaluation_contamination(field):
    train, evaluation = rows(), rows("eval")
    # Edit both variants so the corpus is internally consistent.
    for i in (0, 1): evaluation[i][field] = train[i][field]
    with pytest.raises(ValueError, match="overlap"):
        freeze_cohorts(train, evaluation)


def test_rejects_normalized_prompt_aliases_under_different_groups():
    train = rows()
    train[2]["prompt"] = "  " + train[0]["prompt"].upper() + "  "
    with pytest.raises(ValueError, match="normalized"):
        freeze_cohorts(train, rows("eval"))


def test_requires_semantic_ids_and_enough_groups():
    train = rows()
    del train[0]["semantic_id"]
    with pytest.raises(ValueError, match="semantic_id"):
        freeze_cohorts(train, rows("eval"))
    with pytest.raises(ValueError, match="more independent"):
        freeze_cohorts(rows()[:4], rows("eval"))


def test_readiness_never_launches_and_refuses_overwrite(tmp_path):
    output = tmp_path / "bundle"
    report = prepare(output)
    assert report["status"] == "research_and_tools_prepared"
    assert report["parent_checkpoint"] is None
    assert not report["launch_enabled"] and not report["training_started"]
    plan = json.loads((output / "experiment_plan.json").read_text())
    assert not plan["launch_enabled"] and len(plan["arms"]) == 3
    with pytest.raises(FileExistsError): prepare(output)


def test_cli_bundle_freezes_explicit_inputs_without_claiming_training_ready(tmp_path):
    train, final = tmp_path / "train.jsonl", tmp_path / "final.jsonl"
    for path, data in ((train, rows()), (final, rows("eval"))):
        path.write_text("\n".join(json.dumps(row) for row in data), encoding="utf-8")
    parent = tmp_path / "checkpoint.bin"
    parent.write_bytes(b"fixture; never loaded as a model")
    output = tmp_path / "bundle"
    report = prepare(output, corpus=train, evaluation=final, parent_checkpoint=parent)
    assert report["parent_checkpoint"]["bytes"] == parent.stat().st_size
    assert report["cohort_manifest_sha256"]
    assert (output / "cohort_manifest.json").is_file()
    assert not report["launch_enabled"] and report["blockers"]


def test_missing_evaluation_does_not_create_partial_bundle(tmp_path):
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="both"):
        prepare(output, corpus="anything")
    assert not output.exists()
