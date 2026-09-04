import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "source"))

import nexus_api as api
import nexus_risk_control as risk


def _complete_records(plan):
    benchmark = risk.build_frozen_arithmetic_benchmark()
    rows = []
    for case in benchmark["cases"]:
        prediction = case["expected_answer"] if case["expected_label"] == "answer" else "abstain"
        for policy in risk.FIXED_CANDIDATE_POLICIES:
            rows.append(
                risk.construct_benchmark_shadow_record(
                    case,
                    split="cal",
                    policy_id=policy.policy_id,
                    score=1.0 if prediction != "abstain" else 0.0,
                    prediction=prediction,
                    cost=float(policy.nominal_cost_units),
                )
            )
    return rows


def test_frozen_benchmark_and_plan_are_content_bound():
    benchmark = risk.build_frozen_arithmetic_benchmark()
    assert benchmark["manifest"]["case_count"] == 128
    assert benchmark["manifest_sha256"] == risk.FROZEN_BENCHMARK_MANIFEST_SHA256
    plan = risk.build_risk_control_plan(min_accepted=48)
    assert plan["authority"]["controls_runtime"] is False
    assert plan["assumptions"]["exchangeability_established"] is False


def test_calibration_receipt_is_reproducible_and_non_authoritative():
    plan = risk.build_risk_control_plan(min_accepted=48)
    rows = _complete_records(plan)
    receipt = risk.calibrate_selective_risk(plan, rows)
    risk.validate_risk_control_receipt(receipt, plan=plan, records=rows)
    assert receipt["selection"]["status"] == "certified_policy_selected"
    assert receipt["authority"]["grants_answer_authority"] is False
    assert receipt["selection"]["policy_id"] == "nexus.shadow.budget_1.v1"


def test_plan_rejects_forged_runtime_binding():
    plan = risk.build_risk_control_plan()
    plan["bindings"]["runtime_binding_sha256"] = "0" * 64
    with pytest.raises(risk.RiskControlValidationError):
        risk.validate_risk_control_plan(plan)


def test_service_audit_does_not_change_authority():
    result = api.NexusApiService().handle_risk_audit()
    assert result["status"] == "shadow_audit_complete"
    assert result["selection_authorized"] is False
    assert result["policy_applied"] is False
    assert result["answer_authority"] is False
    assert result["receipt"]["selection"]["policy_id"]
