import json

from source.route_policy_study_cli import main


def test_cli_example_exports_same_read_only_charter_to_stdout(capsys):
    assert main(["--example", "--compact"]) == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert captured.err == ""
    assert plan["study"]["study_id"] == "auto-route-adjacent-explorer-v1"
    assert plan["charter"]["causal_boundaries"]["execution_enabled"] is False
    assert plan["charter"]["causal_boundaries"]["ledger_write_performed"] is False
    assert plan["charter"]["causal_boundaries"]["automatic_promotion_allowed"] is False
    assert plan["charter"]["probability_design"]["assignment_performed"] is False
    assert plan["charter"]["source_contract"] == {
        "policy_id": "auto-route-v2",
        "policy_version": "2.0.0",
        "feature_schema_version": "route-context-v1",
        "support_schema_version": "route-support-v1",
        "candidate_set_hash": "a" * 64,
        "distribution_hash": "b" * 64,
        "outcome_contract_schema_version": "route-outcome-contract-v1",
    }
    exact = plan["charter"]["traffic_scenario"]["observed_label_scenario"][
        "exact_simultaneous_target"
    ]
    assert exact["method"] == "exact_joint_multinomial_tail_inversion_two_alternates"
    assert exact["minimum_routes_for_target_on_every_alternate_action"] == 1971


def test_cli_rejects_prompt_bearing_or_unknown_input_fields(tmp_path, capsys):
    input_path = tmp_path / "support.json"
    input_path.write_text(
        json.dumps(
            {
                "baseline_mode": "off",
                "post_filter_candidates": [],
                "post_filter_exclusions": [],
                "prompt": "must not enter a prompt-free charter",
            }
        ),
        encoding="utf-8",
    )
    assert main(["--input", str(input_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported or non-prompt-free fields: prompt" in captured.err


def test_cli_writes_explicit_output_without_assigning(tmp_path, capsys):
    output_path = tmp_path / "study.json"
    assert main(["--example", "--output", str(output_path)]) == 0
    assert capsys.readouterr().out == ""
    plan = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(plan["design_hash"]) == 64
    assert plan["charter"]["probability_design"]["assignment_performed"] is False
    assert plan["charter"]["traffic_scenario"]["observed_label_scenario"][
        "not_power_analysis"
    ] is True


def test_cli_requires_bound_source_contract(tmp_path, capsys):
    input_path = tmp_path / "support.json"
    input_path.write_text(
        json.dumps(
            {
                "baseline_mode": "off",
                "post_filter_candidates": [],
                "post_filter_exclusions": [],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--input", str(input_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "input is missing required fields: source_contract" in captured.err


def test_cli_accepts_windows_utf8_bom_inputs(tmp_path, capsys):
    from source.route_policy_study_cli import _example_payload

    input_path = tmp_path / "support.json"
    input_path.write_text(json.dumps(_example_payload()), encoding="utf-8-sig")
    assert main(["--input", str(input_path), "--compact"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["study"]["study_id"] == "auto-route-adjacent-explorer-v1"
    assert captured.err == ""
