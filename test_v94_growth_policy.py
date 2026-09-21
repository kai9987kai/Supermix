import copy
from dataclasses import replace
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
from v94_growth_policy import GrowthConfig, GrowthPolicy, UnitObservation, witness_acceptance

SHA = "a" * 64


def population():
    return [UnitObservation(0, True, 0, 1, True),
            UnitObservation(1, True, 0, 1, True),
            UnitObservation(2, True, 5, 0), UnitObservation(3, False, 0, 0)]


def policy(**overrides):
    config = dict(capacity=4, max_alive=4, maturity_steps=10, ema_decay=0)
    config.update(overrides)
    return GrowthPolicy(GrowthConfig(**config), SHA)


def propose(p, step=20, total_steps=100, rows=None, **kwargs):
    return p.propose(population() if rows is None else rows, step=step, total_steps=total_steps,
                     split_role=kwargs.get("role", "growth_calibration"),
                     membership_sha256=kwargs.get("sha", SHA))


def test_budget_stable_ties_and_repeated_low_utility():
    p = policy()
    first = propose(p)
    assert first["births"] == [{"parent": 0, "child": 3}]
    assert first["prune_slots"] == []
    second = propose(p, step=30)
    assert second["prune_slots"] == [2]
    assert second["active_if_all_applied"] == 3
    assert second["advisory_only"]


def test_protects_newborn_originals_and_stops_late_growth():
    p = policy()
    rows = [replace(u, utility=0) for u in population()]
    assert propose(p, step=10, rows=rows)["prune_slots"] == []
    assert propose(p, step=14, rows=rows)["prune_slots"] == []
    assert propose(p, step=15, rows=rows)["prune_slots"] == []
    assert propose(p, step=16, rows=rows)["prune_slots"] == [2]
    assert propose(policy(), step=95)["births"] == []
    assert propose(policy(), step=90)["births"]


def test_resume_reproduces_decisions_and_state_exactly():
    p = policy(ema_decay=0.5)
    propose(p)
    resumed = GrowthPolicy.from_state_dict(json.loads(json.dumps(p.state_dict())))
    assert propose(p, step=30) == propose(resumed, step=30)
    assert p.state_dict() == resumed.state_dict()


def test_capacity_limit_and_minimum_population():
    p = policy(max_alive=3, min_alive=3)
    assert propose(p)["births"] == []
    assert propose(p, step=30)["prune_slots"] == []


def test_slot_reuse_resets_history():
    p = policy()
    propose(p)
    rows = population()
    rows[2] = replace(rows[2], alive=False, born_step=0)
    propose(p, step=25, rows=rows)
    rows[2] = replace(rows[2], alive=True, born_step=26)
    assert propose(p, step=30, rows=rows)["prune_slots"] == []
    assert p.history["2"]["low_count"] == 0


def test_original_cannot_be_reclassified_and_newborn_cannot_be_backdated():
    p = policy()
    propose(p)
    rows = population()
    rows[0] = replace(rows[0], original=False, born_step=21)
    with pytest.raises(ValueError, match="original"):
        propose(p, step=30, rows=rows)
    rows = population()
    rows[3] = replace(rows[3], alive=True, born_step=0)
    with pytest.raises(ValueError, match="backdated"):
        propose(p, step=30, rows=rows)


def test_training_horizon_is_pinned_across_resume():
    p = policy()
    assert propose(p, step=95)["births"] == []
    resumed = GrowthPolicy.from_state_dict(p.state_dict())
    with pytest.raises(ValueError, match="horizon"):
        propose(resumed, step=96, total_steps=1000)


def test_proposed_birth_at_previous_event_survives_resume():
    p = policy()
    event = propose(p, step=20)
    assert event["births"] == [{"parent": 0, "child": 3}]
    restored = GrowthPolicy.from_state_dict(p.state_dict())
    rows = population()
    rows[3] = replace(rows[3], alive=True, born_step=20)
    assert propose(p, step=25, rows=rows) == propose(restored, step=25, rows=rows)
    assert restored.history["3"]["low_count"] == 0


def test_dead_original_is_rejected_before_state_commit():
    p = policy()
    rows = population()
    rows[3] = replace(rows[3], original=True)
    with pytest.raises(ValueError, match="provenance"):
        propose(p, rows=rows)
    assert p.state_dict() == policy().state_dict()


@pytest.mark.parametrize("role", ["dev", "test", "holdout", "benchmark", "train"])
def test_requires_explicit_calibration_role(role):
    with pytest.raises(ValueError, match="growth_calibration"):
        propose(policy(), role=role)


@pytest.mark.parametrize("mutation", ["nan", "negative", "duplicate", "future", "original_changed"])
def test_invalid_inputs_do_not_change_policy(mutation):
    p = policy()
    propose(p)
    saved = p.state_dict()
    rows = population()
    if mutation == "nan": rows[0] = replace(rows[0], utility=float("nan"))
    if mutation == "negative": rows[0] = replace(rows[0], utility=-1)
    if mutation == "duplicate": rows[3] = rows[0]
    if mutation == "future": rows[2] = replace(rows[2], born_step=200)
    if mutation == "original_changed": rows[0] = replace(rows[0], original=False)
    with pytest.raises(ValueError):
        propose(p, step=30, rows=rows)
    assert p.state_dict() == saved


def test_rejects_changed_membership_and_repeated_steps():
    p = policy()
    propose(p)
    with pytest.raises(ValueError): propose(p)
    with pytest.raises(ValueError): propose(p, step=19)
    with pytest.raises(ValueError): propose(p, step=30, sha="b" * 64)


@pytest.mark.parametrize("field,value", [("last_step", -2), ("schema", "other"),
    ("membership_sha256", "not-a-hash"), ("history", {"00": {}})])
def test_rejects_malformed_resume(field, value):
    state = policy().state_dict()
    state[field] = value
    with pytest.raises(ValueError): GrowthPolicy.from_state_dict(state)


def witness():
    return [{"row_id": "old", "family": "retention", "loss_sum": 1., "token_count": 1},
            {"row_id": "new", "family": "new", "loss_sum": 100., "token_count": 100}]


def test_witness_rejects_hidden_family_regression_despite_overall_improvement():
    before = witness()
    after = copy.deepcopy(before)
    after[0]["loss_sum"] = 1.01
    after[1]["loss_sum"] = 90
    result = witness_acceptance(before, after[::-1], split_role="growth_calibration", membership_sha256=SHA)
    assert not result["accepted"]
    assert result["regressed_families"] == ["retention"]
    assert not result["held_out_evidence"]


def test_witness_identity_accepts_and_nonfinite_or_mismatch_fails():
    rows = witness()
    assert witness_acceptance(rows, rows, split_role="growth_calibration", membership_sha256=SHA)["accepted"]
    for field, value in (("loss_sum", float("inf")), ("token_count", 2), ("row_id", "other")):
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            witness_acceptance(rows, bad, split_role="growth_calibration", membership_sha256=SHA)
