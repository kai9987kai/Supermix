import json
import os
from pathlib import Path

import pytest

from source import route_policy_shadow_cli as shadow_cli
from source.route_policy_ledger import hash_session_identity
from source.route_policy_protocol import build_route_study_review_bundle_from_input
from source.route_policy_protocol_cli import _example_bundle_input
from source.route_policy_shadow_cli import main
from source.route_policy_shadow_registry import (
    create_shadow_campaign_artifacts,
    generate_shadow_seed,
)


def _write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _bundle_file(tmp_path: Path) -> Path:
    path = tmp_path / "review-bundle.json"
    _write_json(path, build_route_study_review_bundle_from_input(_example_bundle_input()))
    return path


def _run_success(capsys, argv):
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    return json.loads(captured.out), captured.out


def test_cli_runs_sealed_commit_close_reveal_verify_flow_without_secret_or_raw_id_leaks(
    tmp_path, capsys
):
    bundle_path = _bundle_file(tmp_path)
    registry_path = tmp_path / "shadow.sqlite3"
    capsule_path = tmp_path / "private-seed.json"

    sealed, seal_stdout = _run_success(
        capsys,
        [
            "seal",
            "--registry",
            str(registry_path),
            "--bundle",
            str(bundle_path),
            "--seed-output",
            str(capsule_path),
            "--compact",
        ],
    )
    capsule = json.loads(capsule_path.read_text(encoding="utf-8"))
    private_seed = capsule["seed_material_base64url"]
    campaign_id = sealed["public_package"]["campaign_seal"]["seal"]["campaign_id"]
    assert sealed["created"] is True
    assert sealed["private_seed_capsule_written"] is True
    assert sealed["private_seed_material_returned"] is False
    assert '"private_seed_capsule":{' not in seal_stdout
    assert private_seed not in seal_stdout

    raw_session_identifier = "tenant-raw-id-that-must-not-be-exported"
    cluster_identifier = hash_session_identity(raw_session_identifier)
    cluster_path = tmp_path / "cluster.json"
    _write_json(cluster_path, {"cluster_identifier": cluster_identifier})
    committed, commit_stdout = _run_success(
        capsys,
        [
            "commit",
            "--registry",
            str(registry_path),
            "--campaign",
            campaign_id,
            "--seed-input",
            str(capsule_path),
            "--cluster-input",
            str(cluster_path),
            "--compact",
        ],
    )
    assert committed["created"] is True
    assert committed["chosen_arm_revealed"] is False
    assert raw_session_identifier not in commit_stdout
    assert cluster_identifier not in commit_stdout
    assert private_seed not in commit_stdout
    assert "arm_id" not in commit_stdout

    # The separate registry, WAL, and shared-memory files must not contain the
    # private seed before the explicit post-closure reveal.
    for path in tmp_path.glob("shadow.sqlite3*"):
        assert private_seed.encode("ascii") not in path.read_bytes()

    closed, _ = _run_success(
        capsys,
        ["close", "--registry", str(registry_path), "--campaign", campaign_id, "--compact"],
    )
    assert closed["closure"]["frozen_commitment_count"] == 1

    revealed, reveal_stdout = _run_success(
        capsys,
        [
            "reveal",
            "--registry",
            str(registry_path),
            "--campaign",
            campaign_id,
            "--seed-input",
            str(capsule_path),
            "--compact",
        ],
    )
    assert revealed["seed_material_revealed_in_registry"] is True
    assert revealed["seed_material_returned"] is False
    assert private_seed not in reveal_stdout

    verified, _ = _run_success(
        capsys,
        ["verify", "--registry", str(registry_path), "--campaign", campaign_id, "--compact"],
    )
    assert verified == {
        "activation_available": False,
        "campaign_id": campaign_id,
        "campaign_audit_ok": True,
        "campaign_audit_performed": True,
        "complete": True,
        "execution_enabled": False,
        "invalid_commitment_artifacts": 0,
        "matched": 1,
        "mismatched": 0,
        "total_mismatched": 0,
        "ok": True,
        "processed": 1,
        "processing_complete": True,
        "remaining": 0,
        "verification_complete": True,
    }

    status, _ = _run_success(
        capsys,
        ["status", "--registry", str(registry_path), "--campaign", campaign_id, "--compact"],
    )
    assert status["campaign_count"] == 1
    assert status["campaigns"][0]["state"] == "reveal_verification_complete"
    assert status["campaigns"][0]["matched_assignment_count"] == 1
    assert status["read_only"] is True
    assert status["execution_enabled"] is False
    assert status["automatic_promotion_allowed"] is False

    # Status is a verifier, not a mutation command.  Durable database bytes and
    # logical evidence must therefore remain unchanged across repeated reads.
    before_database = registry_path.read_bytes()
    before_rows = status
    status_again, _ = _run_success(
        capsys,
        ["status", "--registry", str(registry_path), "--campaign", campaign_id, "--compact"],
    )
    assert status_again == before_rows
    assert registry_path.read_bytes() == before_database


def test_seal_recovers_exclusive_capsule_and_is_idempotent(tmp_path, capsys):
    bundle_path = _bundle_file(tmp_path)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    capsule_path = tmp_path / "private-seed.json"
    registry_path = tmp_path / "shadow.sqlite3"
    artifacts = create_shadow_campaign_artifacts(bundle, generate_shadow_seed())
    shadow_cli._write_private_capsule(capsule_path, artifacts["private_seed_capsule"])

    first, first_stdout = _run_success(
        capsys,
        [
            "seal",
            "--registry",
            str(registry_path),
            "--bundle",
            str(bundle_path),
            "--seed-output",
            str(capsule_path),
            "--compact",
        ],
    )
    assert first["created"] is True
    assert first["private_seed_capsule_written"] is False
    assert first["private_seed_capsule_recovered"] is True
    assert artifacts["private_seed_capsule"]["seed_material_base64url"] not in first_stdout

    second, _ = _run_success(
        capsys,
        [
            "seal",
            "--registry",
            str(registry_path),
            "--bundle",
            str(bundle_path),
            "--seed-output",
            str(capsule_path),
            "--compact",
        ],
    )
    assert second["created"] is False
    assert second["private_seed_capsule_recovered"] is True

    capsule_path.unlink()
    assert (
        main(
            [
                "seal",
                "--registry",
                str(registry_path),
                "--bundle",
                str(bundle_path),
                "--seed-output",
                str(capsule_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "recover its original seed capsule" in captured.err
    assert not capsule_path.exists()


def test_cli_strict_json_rejects_duplicate_keys_and_non_finite_values(tmp_path, capsys):
    registry_path = tmp_path / "shadow.sqlite3"
    capsule_path = tmp_path / "private-seed.json"
    duplicate_bundle = tmp_path / "duplicate.json"
    duplicate_bundle.write_text('{"schema_version":"one","schema_version":"two"}', encoding="utf-8")

    assert (
        main(
            [
                "seal",
                "--registry",
                str(registry_path),
                "--bundle",
                str(duplicate_bundle),
                "--seed-output",
                str(capsule_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate object key: schema_version" in captured.err
    assert not registry_path.exists()
    assert not capsule_path.exists()

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value":NaN}', encoding="utf-8")
    assert (
        main(
            [
                "seal",
                "--registry",
                str(registry_path),
                "--bundle",
                str(non_finite),
                "--seed-output",
                str(capsule_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "non-finite number: NaN" in captured.err


def test_commit_rejects_ambiguous_cluster_input_without_echoing_identifier(tmp_path, capsys):
    bundle_path = _bundle_file(tmp_path)
    registry_path = tmp_path / "shadow.sqlite3"
    capsule_path = tmp_path / "private-seed.json"
    sealed, _ = _run_success(
        capsys,
        [
            "seal",
            "--registry",
            str(registry_path),
            "--bundle",
            str(bundle_path),
            "--seed-output",
            str(capsule_path),
            "--compact",
        ],
    )
    campaign_id = sealed["public_package"]["campaign_seal"]["seal"]["campaign_id"]
    cluster_path = tmp_path / "cluster.json"
    cluster_path.write_text(
        '{"cluster_identifier":"secret-a","cluster_identifier":"secret-b"}',
        encoding="utf-8",
    )

    assert (
        main(
            [
                "commit",
                "--registry",
                str(registry_path),
                "--campaign",
                campaign_id,
                "--seed-input",
                str(capsule_path),
                "--cluster-input",
                str(cluster_path),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "duplicate object key: cluster_identifier" in captured.err
    assert "secret-a" not in captured.err
    assert "secret-b" not in captured.err


def test_windows_capsule_acl_is_verified_before_and_after_seed_write(
    tmp_path, monkeypatch
):
    capsule_path = tmp_path / "private-seed.json"
    events = []

    def apply_acl(path):
        events.append(("apply", path.stat().st_size))

    def verify_acl(descriptor):
        events.append(("verify", os.fstat(descriptor).st_size))

    monkeypatch.setattr(shadow_cli, "_is_windows", lambda: True)
    monkeypatch.setattr(shadow_cli, "_apply_windows_private_capsule_acl", apply_acl)
    monkeypatch.setattr(shadow_cli, "_verify_windows_private_capsule_acl", verify_acl)

    shadow_cli._write_private_capsule(capsule_path, {"secret": "seed-material"})

    assert events[0] == ("apply", 0)
    assert events[1] == ("verify", 0)
    assert events[2][0] == "verify"
    assert events[2][1] > 0
    assert json.loads(capsule_path.read_text(encoding="utf-8")) == {
        "secret": "seed-material"
    }


@pytest.mark.parametrize("failure_stage", ["apply", "prewrite_verify", "postwrite_verify"])
def test_windows_capsule_acl_failure_removes_capsule(
    tmp_path, monkeypatch, failure_stage
):
    capsule_path = tmp_path / "private-seed.json"
    verify_calls = 0

    def apply_acl(_path):
        if failure_stage == "apply":
            raise OSError("mock Windows ACL apply failure")

    def verify_acl(_descriptor):
        nonlocal verify_calls
        verify_calls += 1
        if failure_stage == "prewrite_verify" and verify_calls == 1:
            raise OSError("mock Windows ACL prewrite verification failure")
        if failure_stage == "postwrite_verify" and verify_calls == 2:
            raise OSError("mock Windows ACL postwrite verification failure")

    monkeypatch.setattr(shadow_cli, "_is_windows", lambda: True)
    monkeypatch.setattr(shadow_cli, "_apply_windows_private_capsule_acl", apply_acl)
    monkeypatch.setattr(shadow_cli, "_verify_windows_private_capsule_acl", verify_acl)

    with pytest.raises(OSError, match="mock Windows ACL"):
        shadow_cli._write_private_capsule(capsule_path, {"secret": "seed-material"})

    assert not capsule_path.exists()


def test_windows_capsule_read_fails_closed_before_reading_broad_acl(
    tmp_path, monkeypatch
):
    capsule_path = tmp_path / "private-seed.json"
    capsule_path.write_text('{"secret":"seed-material"}', encoding="utf-8")

    monkeypatch.setattr(shadow_cli, "_is_windows", lambda: True)

    def reject_acl(_descriptor):
        raise OSError("mock broad Windows ACL")

    monkeypatch.setattr(shadow_cli, "_verify_windows_private_capsule_acl", reject_acl)

    with pytest.raises(OSError, match="mock broad Windows ACL"):
        shadow_cli._read_private_capsule(capsule_path)

    assert capsule_path.exists()
