import json

from source.route_policy_protocol_cli import main


def test_cli_example_exports_fail_closed_protocol(capsys):
    assert main(["--example", "--compact"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    protocol = json.loads(captured.out)
    assert protocol["protocol"]["label"] == "Stateful Route Experiment Preflight v1"
    assert protocol["protocol"]["activation_available"] is False
    assert protocol["charter"]["stateful_design"]["selected_design_status"] == (
        "declaration_incomplete"
    )
    assert protocol["charter"]["randomness"]["assignment_performed"] is False
    exact_contract = protocol["charter"]["source_studies"]["common_source_contract"]
    assert exact_contract["outcome_contract_schema_version"] == "route-outcome-contract-v1"


def test_cli_build_and_audit_round_trip(tmp_path, capsys):
    protocol_path = tmp_path / "protocol.json"
    assert main(["--example", "--output", str(protocol_path)]) == 0
    assert capsys.readouterr().out == ""

    assert main(["--audit", str(protocol_path), "--compact"]) == 0
    captured = capsys.readouterr()
    audit = json.loads(captured.out)
    assert captured.err == ""
    assert audit["ok"] is True
    assert audit["state"] == "draft_for_independent_review"
    assert audit["activation_available"] is False
    assert len(audit["activation_blockers"]) == 8


def test_cli_rejects_prompt_bearing_input(tmp_path, capsys):
    input_path = tmp_path / "unsafe.json"
    input_path.write_text(json.dumps({"study_plans": [], "prompt": "private text"}), encoding="utf-8")
    assert main(["--input", str(input_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unsupported or non-prompt-free fields: prompt" in captured.err


def test_cli_rejects_tampered_protocol_audit(tmp_path, capsys):
    protocol_path = tmp_path / "protocol.json"
    assert main(["--example", "--output", str(protocol_path)]) == 0
    capsys.readouterr()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["charter"]["causal_boundaries"]["automatic_promotion_allowed"] = True
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    assert main(["--audit", str(protocol_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "fail-closed" in captured.err


def test_cli_accepts_windows_utf8_bom_protocol_files(tmp_path, capsys):
    protocol_path = tmp_path / "protocol.json"
    assert main(["--example", "--output", str(protocol_path)]) == 0
    capsys.readouterr()
    raw = protocol_path.read_text(encoding="utf-8")
    protocol_path.write_text(raw, encoding="utf-8-sig")

    assert main(["--audit", str(protocol_path), "--compact"]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True
    assert captured.err == ""


def test_cli_example_bundle_round_trips_with_full_source_reconstruction(tmp_path, capsys):
    bundle_path = tmp_path / "review-bundle.json"
    assert main(["--example-bundle", "--output", str(bundle_path)]) == 0
    assert capsys.readouterr().out == ""
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "route-study-review-bundle-v1"
    assert len(bundle["source_study_plans"]) == 2
    assert bundle["bundle"]["authenticity_proof_available"] is False

    bundle_path.write_text(
        bundle_path.read_text(encoding="utf-8"), encoding="utf-8-sig"
    )
    assert main(["--audit-bundle", str(bundle_path), "--compact"]) == 0
    captured = capsys.readouterr()
    audit = json.loads(captured.out)
    assert captured.err == ""
    assert audit["ok"] is True
    assert audit["support_stratum_count"] == 2
    assert audit["verification_level"] == "full_source_bound_reconstruction"
    assert audit["source_plan_reconstruction_performed"] is True


def test_cli_bundle_audit_rejects_source_plan_substitution(tmp_path, capsys):
    bundle_path = tmp_path / "review-bundle.json"
    assert main(["--example-bundle", "--output", str(bundle_path)]) == 0
    capsys.readouterr()
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["source_study_plans"][0]["design_hash"] = "f" * 64
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")

    assert main(["--audit-bundle", str(bundle_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "design_hash does not match" in captured.err
