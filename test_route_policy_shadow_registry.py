import base64
import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from source.route_policy_explorer import plan_adjacent_route_study
from source.route_policy_ledger import hash_session_identity
from source.route_policy_protocol import build_route_study_review_bundle
from source.route_policy_shadow_registry import (
    SHADOW_ASSIGNMENT_ALGORITHM,
    RouteShadowAssignmentRegistry,
    RouteShadowRegistryError,
    ShadowRegistryConflictError,
    _PACKAGE_HASH_DOMAIN,
    _canonical_json,
    _domain_hash,
    audit_shadow_campaign_artifacts,
    audit_shadow_seed_capsule,
    prepare_shadow_assignment_commitment,
)
from source.route_policy_study_cli import _example_payload


SEED = bytes.fromhex(
    "04b12fc8e9714e36aa4c0f39bfc81460e959f6f99b7264e614c76d612b835f58"
)
OTHER_SEED = bytes.fromhex(
    "e8134bf1a09987438cc8f0d7dfe8c6d507d635f8a10b049e4f33d744a0e4d90a"
)


def _session_hash(label: str) -> str:
    return hash_session_identity(label)


def _review_bundle(
    *,
    candidate="a",
    distribution="b",
    planned_clusters=8,
    cluster_key_schema_version="session-hash-v1",
):
    payload = _example_payload()
    payload["source_contract"] = {
        **payload["source_contract"],
        "candidate_set_hash": candidate * 64,
        "distribution_hash": distribution * 64,
    }
    study = plan_adjacent_route_study(
        payload["baseline_mode"],
        payload["post_filter_candidates"],
        payload["post_filter_exclusions"],
        source_contract=payload["source_contract"],
        exploration_rate=payload["exploration_rate"],
        planned_routes=payload["planned_routes"],
        scenario_confidence=payload["scenario_confidence"],
        assumed_feedback_rate=payload["assumed_feedback_rate"],
        target_observed_labels=payload["target_observed_labels"],
    )
    return build_route_study_review_bundle(
        study,
        planned_clusters=planned_clusters,
        analysis_every_clusters=max(1, planned_clusters // 2),
        cluster_key_schema_version=cluster_key_schema_version,
    )


def _sealed(tmp_path: Path, *, seed=SEED, bundle=None):
    db_path = tmp_path / "route-policy-shadow-registry.sqlite3"
    registry = RouteShadowAssignmentRegistry(db_path)
    result = registry.seal_campaign(bundle or _review_bundle(), seed)
    package = result["public_package"]
    return {
        "db_path": db_path,
        "registry": registry,
        "seal_result": result,
        "package": package,
        "capsule": result["private_seed_capsule"],
        "campaign_id": package["campaign_seal"]["seal"]["campaign_id"],
    }


def _assert_no_floats(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_floats(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_floats(item)
    else:
        assert not isinstance(value, float)


def _rehash_public_package(package):
    payload = {key: value for key, value in package.items() if key != "public_package_hash"}
    package["public_package_hash"] = _domain_hash(
        _PACKAGE_HASH_DOMAIN,
        payload,
        "shadow public package",
    )


def _database_files(db_path: Path):
    return [path for path in db_path.parent.glob(f"{db_path.name}*") if path.is_file()]


def test_valid_lifecycle_is_batch_reconstructable_and_stays_shadow_only(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    package = sealed["package"]
    capsule = sealed["capsule"]
    campaign_id = sealed["campaign_id"]

    assert sealed["seal_result"]["created"] is True
    assert sealed["seal_result"]["private_seed_material_persisted"] is False
    assert audit_shadow_campaign_artifacts(package)["verification_level"] == (
        "full_source_bound_shadow_seal_reconstruction"
    )
    assert audit_shadow_seed_capsule(package, capsule)["seed_material_returned"] is False

    first = registry.append_assignment_commitment(
        campaign_id=campaign_id,
        seed_capsule=capsule,
        cluster_identifier=_session_hash("raw-session-alpha"),
    )
    second = registry.append_assignment_commitment(
        campaign_id=campaign_id,
        seed_capsule=capsule,
        cluster_identifier=_session_hash("raw-session-beta"),
    )
    for result in (first, second):
        assert result["created"] is True
        assert result["chosen_arm_revealed"] is False
        assert result["private_reveal_persisted"] is False
        assert "arm_id" not in json.dumps(result["commitment"], sort_keys=True)

    before = registry.snapshot(campaign_id)
    assert before["campaigns"][0]["state"] == "accepting_commitments"
    assert before["campaigns"][0]["commitment_count"] == 2
    assert before["campaigns"][0]["whole_policy_arm_counts"] == {}

    closure = registry.close_campaign(campaign_id)
    assert closure["closure"]["frozen_commitment_count"] == 2
    seed_reveal = registry.reveal_seed(campaign_id=campaign_id, seed_capsule=capsule)
    assert seed_reveal["reveal"]["seed_material_base64url"] == capsule[
        "seed_material_base64url"
    ]

    first_batch = registry.verify_assignment_reveals(campaign_id, batch_size=1)
    second_batch = registry.verify_assignment_reveals(campaign_id, batch_size=1)
    idempotent_batch = registry.verify_assignment_reveals(campaign_id, batch_size=1)
    assert first_batch == {
        **first_batch,
        "ok": True,
        "processed": 1,
        "matched": 1,
        "mismatched": 0,
        "remaining": 1,
        "complete": False,
        "execution_enabled": False,
        "activation_available": False,
    }
    assert second_batch["processed"] == second_batch["matched"] == 1
    assert second_batch["remaining"] == 0
    assert second_batch["complete"] is True
    assert idempotent_batch["processed"] == 0
    assert idempotent_batch["complete"] is True

    after = registry.snapshot(campaign_id)
    campaign = after["campaigns"][0]
    assert campaign["state"] == "reveal_verification_complete"
    assert campaign["matched_assignment_count"] == 2
    assert campaign["mismatched_assignment_count"] == 0
    assert sum(campaign["whole_policy_arm_counts"].values()) == 2
    assert after["event_chain"]["ok"] is True
    assert after["event_chain"]["verified_events"] == 7
    assert after["schema_integrity"]["ok"] is True
    assert after["verification_level"] == "local_append_only_chain_without_external_anchor"
    assert after["authenticity_proof_available"] is False
    assert after["trusted_timestamp_available"] is False
    assert after["execution_enabled"] is False
    assert after["activation_available"] is False
    assert after["automatic_promotion_allowed"] is False

    manifest_arms = {
        row["arm_id"]: row
        for row in package["assignment_manifest"]["manifest"]["whole_policy_arms"]
    }
    with sqlite3.connect(sealed["db_path"]) as connection:
        rows = connection.execute(
            """
            SELECT c.assignment_reveal_commitment, r.assignment_reveal_hash, r.reveal_json
            FROM shadow_assignment_commitments c
            JOIN shadow_assignment_reveals r USING (commitment_hash)
            ORDER BY c.commitment_hash
            """
        ).fetchall()
    assert len(rows) == 2
    for committed_reveal_hash, actual_reveal_hash, reveal_json in rows:
        reveal = json.loads(reveal_json)
        assignment = reveal["assignment"]
        assert committed_reveal_hash == actual_reveal_hash == reveal["assignment_reveal_hash"]
        assert assignment["arm_id"] in manifest_arms
        assert assignment["arm_hash"] == manifest_arms[assignment["arm_id"]]["arm_hash"]
        assert assignment["arm_probability_bps"] == 5000
        assert assignment["assignment_algorithm"] == SHADOW_ASSIGNMENT_ALGORITHM
        assert assignment["boundaries"]["shadow_only"] is True
        assert assignment["boundaries"]["ledger_eligible"] is False

    for artifact in (package, capsule, first, second, closure, seed_reveal, after):
        _assert_no_floats(artifact)
        json.dumps(artifact, sort_keys=True, allow_nan=False)


def test_read_only_snapshot_does_not_mutate_registry_and_rejects_writes(tmp_path):
    sealed = _sealed(tmp_path)
    sealed["registry"].append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("read-only-cluster"),
    )
    before_database = sealed["db_path"].read_bytes()

    reader = RouteShadowAssignmentRegistry(sealed["db_path"], read_only=True)
    snapshot = reader.snapshot(sealed["campaign_id"])
    assert snapshot["ok"] is True
    assert snapshot["campaigns"][0]["commitment_count"] == 1
    with pytest.raises(RouteShadowRegistryError, match="opened read-only"):
        reader.close_campaign(sealed["campaign_id"])

    # SQLite may create or retire WAL coordination sidecars while servicing a
    # read-only connection, but the durable registry file and logical rows are
    # unchanged and the URI mode makes SQL writes impossible.
    assert sealed["db_path"].read_bytes() == before_database
    with sqlite3.connect(sealed["db_path"]) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM shadow_campaign_closures"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM shadow_assignment_commitments"
        ).fetchone()[0] == 1


def test_manifest_has_two_explicit_whole_policy_arms_not_route_actions(tmp_path):
    package = _sealed(tmp_path)["package"]
    manifest = package["assignment_manifest"]["manifest"]
    design = package["design_binding"]
    arms = manifest["whole_policy_arms"]

    assert manifest["assignment_algorithm"] == SHADOW_ASSIGNMENT_ALGORITHM
    assert [row["arm_id"] for row in arms] == [
        "incumbent_source_policy",
        "candidate_target_policy",
    ]
    assert [row["allocation_bps"] for row in arms] == [5000, 5000]
    assert sum(row["allocation_bps"] for row in arms) == 10_000
    assert [row["policy_binding"]["binding_type"] for row in arms] == [
        "frozen_source_policy_class",
        "frozen_target_policy_class",
    ]
    assert all(
        len(row["policy_binding"]["policy_class_hash"]) == 64 for row in arms
    )
    assert arms[0]["policy_binding"]["policy_class_manifest"]["support_strata"]
    assert arms[1]["policy_binding"]["policy_class_manifest"]["thresholds"]
    assert arms == design["whole_policy_arms"]
    assert all(len(row["arm_hash"]) == 64 for row in arms)
    arm_json = json.dumps(arms, sort_keys=True)
    assert "eligible_actions" not in arm_json
    assert "rehearsed_action_probabilities" not in arm_json


def test_seed_raw_session_and_session_hash_are_absent_from_db_and_wal_before_reveal(
    tmp_path,
):
    sealed = _sealed(tmp_path)
    raw_session = "private-raw-session-identity-never-persisted"
    session_hash = _session_hash(raw_session)
    sealed["registry"].append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=session_hash,
    )

    with sqlite3.connect(sealed["db_path"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM shadow_seed_reveals").fetchone()[0] == 0
        stored = connection.execute(
            "SELECT commitment_json, cluster_pseudonym FROM shadow_assignment_commitments"
        ).fetchone()
        assert raw_session not in stored[0]
        assert session_hash not in stored[0]
        assert session_hash != stored[1]

    encoded_seed = base64.urlsafe_b64encode(SEED).decode("ascii").rstrip("=").encode("ascii")
    assert encoded_seed.decode("ascii") not in json.dumps(sealed["package"], sort_keys=True)
    for path in _database_files(sealed["db_path"]):
        contents = path.read_bytes()
        assert SEED not in contents
        assert encoded_seed not in contents
        assert raw_session.encode("utf-8") not in contents
        assert session_hash.encode("ascii") not in contents


def test_wrong_seed_or_campaign_capsule_is_rejected(tmp_path):
    first = _sealed(tmp_path / "first", bundle=_review_bundle(candidate="a"))
    second = _sealed(
        tmp_path / "second",
        seed=OTHER_SEED,
        bundle=_review_bundle(candidate="c", distribution="d"),
    )

    with pytest.raises(ValueError, match="campaign_id does not match"):
        first["registry"].append_assignment_commitment(
            campaign_id=first["campaign_id"],
            seed_capsule=second["capsule"],
            cluster_identifier=_session_hash("cluster-one"),
        )

    wrong_seed = copy.deepcopy(first["capsule"])
    wrong_seed["seed_material_base64url"] = base64.urlsafe_b64encode(OTHER_SEED).decode(
        "ascii"
    ).rstrip("=")
    with pytest.raises(ValueError, match="does not open"):
        prepare_shadow_assignment_commitment(
            first["package"],
            wrong_seed,
            _session_hash("cluster-one"),
        )

    noncanonical = copy.deepcopy(first["capsule"])
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    canonical = noncanonical["seed_material_base64url"]
    last_index = alphabet.index(canonical[-1])
    noncanonical["seed_material_base64url"] = canonical[:-1] + alphabet[last_index | 1]
    assert base64.urlsafe_b64decode(
        noncanonical["seed_material_base64url"] + "="
    ) == SEED
    with pytest.raises(ValueError, match="not canonical"):
        audit_shadow_seed_capsule(first["package"], noncanonical)


def test_assignment_pseudonym_is_derived_internally_and_commitment_is_deterministic(tmp_path):
    sealed = _sealed(tmp_path)
    caller_supplied_hash_shaped_identifier = _session_hash("cluster-one")
    first = prepare_shadow_assignment_commitment(
        sealed["package"],
        sealed["capsule"],
        caller_supplied_hash_shaped_identifier,
    )
    second = prepare_shadow_assignment_commitment(
        sealed["package"],
        sealed["capsule"],
        caller_supplied_hash_shaped_identifier,
    )

    assert first == second
    commitment = first["commitment"]["commitment"]
    reveal = first["private_reveal"]["assignment"]
    assert commitment["cluster_pseudonym"] == reveal["cluster_pseudonym"]
    assert commitment["cluster_pseudonym"] != caller_supplied_hash_shaped_identifier
    assert commitment["chosen_arm_withheld_until_reveal"] is True
    assert "arm_id" not in commitment
    arms = {
        row["arm_id"]: row
        for row in sealed["package"]["assignment_manifest"]["manifest"][
            "whole_policy_arms"
        ]
    }
    assert reveal["arm_hash"] == arms[reveal["arm_id"]]["arm_hash"]


def test_cluster_identity_accepts_only_the_exact_canonical_session_hash(tmp_path):
    sealed = _sealed(tmp_path)
    canonical = _session_hash("cluster-alias-regression")
    assert len(canonical) == 64
    assert canonical == canonical.lower()

    prepared = prepare_shadow_assignment_commitment(
        sealed["package"],
        sealed["capsule"],
        canonical,
    )
    assert prepared["commitment"]["commitment"]["cluster_pseudonym"] != canonical

    aliases = [
        "cluster-alias-regression",
        canonical.upper(),
        f" {canonical}",
        f"{canonical}\n",
        f"sha256:{canonical}",
        canonical[:-1],
        f"{canonical}0",
        "ａ" * 64,
    ]
    assert canonical.upper() != canonical
    before = sealed["registry"].snapshot(sealed["campaign_id"])
    for alias in aliases:
        with pytest.raises(ValueError, match="canonical session_hash"):
            sealed["registry"].append_assignment_commitment(
                campaign_id=sealed["campaign_id"],
                seed_capsule=sealed["capsule"],
                cluster_identifier=alias,
            )
    after = sealed["registry"].snapshot(sealed["campaign_id"])
    assert after["campaigns"][0]["commitment_count"] == 0
    assert after["event_chain"] == before["event_chain"]


def test_shadow_registry_rejects_alternate_cluster_key_schema(tmp_path):
    with pytest.raises(
        ValueError,
        match="requires the session-hash-v1 cluster key schema",
    ):
        _sealed(
            tmp_path,
            bundle=_review_bundle(
                cluster_key_schema_version="study-scoped-session-hash-v2"
            ),
        )


def test_illegal_state_transitions_fail_closed(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    campaign_id = sealed["campaign_id"]

    with pytest.raises(RouteShadowRegistryError, match="before closure"):
        registry.reveal_seed(campaign_id=campaign_id, seed_capsule=sealed["capsule"])
    with pytest.raises(RouteShadowRegistryError, match="before assignment verification"):
        registry.verify_assignment_reveals(campaign_id)
    closure = registry.close_campaign(campaign_id)
    assert closure["closure"]["frozen_commitment_count"] == 0
    with pytest.raises(RouteShadowRegistryError, match="commitments are closed"):
        registry.append_assignment_commitment(
            campaign_id=campaign_id,
            seed_capsule=sealed["capsule"],
            cluster_identifier=_session_hash("cluster-two"),
        )
    with pytest.raises(RouteShadowRegistryError, match="before assignment verification"):
        registry.verify_assignment_reveals(campaign_id)


def test_seal_commit_close_reveal_and_verify_are_idempotent(tmp_path):
    bundle = _review_bundle()
    sealed = _sealed(tmp_path, bundle=bundle)
    registry = sealed["registry"]
    campaign_id = sealed["campaign_id"]

    same_seal = registry.seal_campaign(bundle, SEED)
    assert same_seal["created"] is False
    assert same_seal["private_seed_capsule"] is None
    assert same_seal["seed_capsule_returned_once"] is False
    with pytest.raises(ShadowRegistryConflictError, match="different seed commitment"):
        registry.seal_campaign(bundle, OTHER_SEED)

    first = registry.append_assignment_commitment(
        campaign_id=campaign_id,
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("cluster-one"),
    )
    same = registry.append_assignment_commitment(
        campaign_id=campaign_id,
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("cluster-one"),
    )
    assert first["created"] is True
    assert same["created"] is False
    assert same["commitment"] == first["commitment"]

    closure = registry.close_campaign(campaign_id)
    assert registry.close_campaign(campaign_id) == closure
    reveal = registry.reveal_seed(campaign_id=campaign_id, seed_capsule=sealed["capsule"])
    assert registry.reveal_seed(campaign_id=campaign_id, seed_capsule=sealed["capsule"]) == reveal
    verified = registry.verify_assignment_reveals(campaign_id)
    assert verified["processed"] == verified["matched"] == 1
    assert registry.verify_assignment_reveals(campaign_id)["processed"] == 0
    assert registry.snapshot(campaign_id)["event_chain"]["verified_events"] == 5


def test_planned_cluster_ceiling_is_enforced_before_any_extra_event(tmp_path):
    sealed = _sealed(tmp_path, bundle=_review_bundle(planned_clusters=2))
    registry = sealed["registry"]
    for cluster in (_session_hash("cluster-one"), _session_hash("cluster-two")):
        registry.append_assignment_commitment(
            campaign_id=sealed["campaign_id"],
            seed_capsule=sealed["capsule"],
            cluster_identifier=cluster,
        )
    before = registry.snapshot(sealed["campaign_id"])
    with pytest.raises(RouteShadowRegistryError, match="planned cluster ceiling reached"):
        registry.append_assignment_commitment(
            campaign_id=sealed["campaign_id"],
            seed_capsule=sealed["capsule"],
            cluster_identifier=_session_hash("cluster-three"),
        )
    after = registry.snapshot(sealed["campaign_id"])
    assert after["campaigns"][0]["commitment_count"] == 2
    assert after["event_chain"] == before["event_chain"]


def test_rehashed_internal_package_tampering_and_duplicate_keys_are_rejected(tmp_path):
    package = _sealed(tmp_path)["package"]
    tampered = copy.deepcopy(package)
    tampered["assignment_manifest"]["manifest"]["whole_policy_arms"][0][
        "allocation_bps"
    ] = 4999
    _rehash_public_package(tampered)
    with pytest.raises(ValueError, match="manifest does not reconstruct"):
        audit_shadow_campaign_artifacts(tampered)

    duplicate = copy.deepcopy(package)
    encoded = duplicate["origin_review_bundle"]
    duplicate["origin_review_bundle"] = encoded.replace(
        '{"bundle":',
        '{"bundle":null,"bundle":',
        1,
    )
    _rehash_public_package(duplicate)
    with pytest.raises(ValueError, match="strict canonical JSON string"):
        audit_shadow_campaign_artifacts(duplicate)


def test_shadow_native_canonicalization_rejects_floats_and_noncanonical_inputs(tmp_path):
    with pytest.raises(ValueError, match="floating-point"):
        _canonical_json({"probability": 0.5})

    sealed = _sealed(tmp_path)
    with pytest.raises(ValueError, match="canonical session_hash string"):
        sealed["registry"].append_assignment_commitment(
            campaign_id=sealed["campaign_id"],
            seed_capsule=sealed["capsule"],
            cluster_identifier={"caller": "supplied-hash"},
        )
    with pytest.raises(ValueError, match="batch_size must be an integer"):
        sealed["registry"].verify_assignment_reveals(
            sealed["campaign_id"],
            batch_size=1.0,
        )


def test_artifact_tables_and_event_log_are_append_only(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    registry.append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("cluster-one"),
    )
    registry.close_campaign(sealed["campaign_id"])
    registry.reveal_seed(campaign_id=sealed["campaign_id"], seed_capsule=sealed["capsule"])
    registry.verify_assignment_reveals(sealed["campaign_id"])

    tables = [
        "shadow_registry_metadata",
        "shadow_campaign_seals",
        "shadow_assignment_commitments",
        "shadow_campaign_closures",
        "shadow_seed_reveals",
        "shadow_assignment_reveals",
        "shadow_registry_events",
    ]
    for table in tables:
        with sqlite3.connect(sealed["db_path"]) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"UPDATE {table} SET rowid = rowid")
        with sqlite3.connect(sealed["db_path"]) as connection:
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                connection.execute(f"DELETE FROM {table}")


def test_schema_indexes_and_cross_campaign_reveal_guard_are_enforced(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    registry.append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("first-campaign-cluster"),
    )
    registry.close_campaign(sealed["campaign_id"])
    registry.reveal_seed(
        campaign_id=sealed["campaign_id"], seed_capsule=sealed["capsule"]
    )

    second_seed = bytes(range(32, 64))
    second = registry.seal_campaign(_review_bundle(planned_clusters=6), second_seed)
    second_package = second["public_package"]
    second_campaign = second_package["campaign_seal"]["seal"]["campaign_id"]
    second_capsule = second["private_seed_capsule"]
    registry.close_campaign(second_campaign)
    registry.reveal_seed(campaign_id=second_campaign, seed_capsule=second_capsule)

    with sqlite3.connect(sealed["db_path"]) as connection:
        index_names = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list('shadow_assignment_commitments')"
            ).fetchall()
        }
        reveal_index_names = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list('shadow_assignment_reveals')"
            ).fetchall()
        }
        assert "shadow_commitments_campaign_order_idx" in index_names
        assert "shadow_reveals_campaign_order_idx" in reveal_index_names
        commitment_hash = connection.execute(
            "SELECT commitment_hash FROM shadow_assignment_commitments WHERE campaign_id = ?",
            (sealed["campaign_id"],),
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError, match="campaign does not match commitment"):
            connection.execute(
                """
                INSERT INTO shadow_assignment_reveals(
                    commitment_hash, campaign_id, verification_status,
                    assignment_reveal_hash, reveal_json, verified_at_us
                ) VALUES (?, ?, 'matched', ?, '{}', 1)
                """,
                (commitment_hash, second_campaign, "a" * 64),
            )


def test_event_chain_detects_local_tampering_if_append_only_trigger_is_removed(tmp_path):
    sealed = _sealed(tmp_path)
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_events_no_update")
        connection.execute(
            "UPDATE shadow_registry_events SET event_hash = ? WHERE event_sequence = 1",
            ("f" * 64,),
        )

    snapshot = sealed["registry"].snapshot(sealed["campaign_id"])
    assert snapshot["ok"] is False
    assert snapshot["schema_integrity"]["ok"] is False
    assert snapshot["schema_integrity"]["missing_triggers"] == [
        "shadow_events_no_update"
    ]
    assert snapshot["event_chain"]["ok"] is False
    assert snapshot["event_chain"]["reason"] == "event_hash_mismatch"
    assert snapshot["authenticity_proof_available"] is False
    assert snapshot["external_transparency_anchor_available"] is False


def test_schema_audit_detects_same_name_noop_trigger_replacement(tmp_path):
    sealed = _sealed(tmp_path)
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_commitments_no_update")
        connection.execute(
            """
            CREATE TRIGGER shadow_commitments_no_update
            AFTER INSERT ON shadow_registry_metadata
            BEGIN
                SELECT 1;
            END
            """
        )

    snapshot = sealed["registry"].snapshot(sealed["campaign_id"])
    assert snapshot["ok"] is False
    assert snapshot["schema_integrity"]["missing_triggers"] == []
    assert snapshot["schema_integrity"]["definitions_ok"] is False
    assert snapshot["schema_integrity"]["definition_fingerprint"] != snapshot[
        "schema_integrity"
    ]["expected_definition_fingerprint"]


def test_mismatched_reveal_is_never_reported_as_verified_or_complete(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    registry.append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("mismatch-cluster"),
    )
    registry.close_campaign(sealed["campaign_id"])
    registry.reveal_seed(
        campaign_id=sealed["campaign_id"], seed_capsule=sealed["capsule"]
    )
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_commitments_no_update")
        connection.execute(
            """
            UPDATE shadow_assignment_commitments
            SET assignment_reveal_commitment = ?
            WHERE campaign_id = ?
            """,
            ("0" * 64, sealed["campaign_id"]),
        )

    first = registry.verify_assignment_reveals(sealed["campaign_id"])
    assert first["ok"] is False
    assert first["processing_complete"] is True
    assert first["verification_complete"] is False
    assert first["complete"] is False
    assert first["total_mismatched"] == 1

    # An idempotent later batch must not forget the persisted mismatch.
    second = registry.verify_assignment_reveals(sealed["campaign_id"])
    assert second["processed"] == 0
    assert second["ok"] is False
    assert second["complete"] is False
    assert second["total_mismatched"] == 1

    campaign = registry.snapshot(sealed["campaign_id"])["campaigns"][0]
    assert campaign["state"] == "reveal_verification_failed"
    assert campaign["lifecycle_state"] == "reveal_verification_complete"
    assert campaign["processed_reveal_count"] == 1
    assert campaign["verified_assignment_count"] == 0
    assert campaign["matched_assignment_count"] == 0
    assert campaign["mismatched_assignment_count"] == 1


def test_verify_marks_isolated_commitment_json_corruption_as_mismatch(tmp_path):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    registry.append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("corrupt-json-cluster"),
    )
    registry.close_campaign(sealed["campaign_id"])
    registry.reveal_seed(
        campaign_id=sealed["campaign_id"], seed_capsule=sealed["capsule"]
    )
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_commitments_no_update")
        connection.execute(
            "UPDATE shadow_assignment_commitments SET commitment_json = '{}'"
        )

    result = registry.verify_assignment_reveals(sealed["campaign_id"])
    assert result["ok"] is False
    assert result["complete"] is False
    assert result["invalid_commitment_artifacts"] == 1
    assert result["mismatched"] == 1
    assert result["campaign_audit_performed"] is True
    assert result["campaign_audit_ok"] is False


@pytest.mark.parametrize(
    ("table", "trigger", "column", "message"),
    [
        (
            "shadow_campaign_closures",
            "shadow_closures_no_update",
            "closure_json",
            "invalid campaign closure",
        ),
        (
            "shadow_seed_reveals",
            "shadow_seed_reveals_no_update",
            "reveal_json",
            "invalid seed reveal",
        ),
    ],
)
def test_verify_fails_closed_on_closure_or_seed_artifact_corruption(
    tmp_path, table, trigger, column, message
):
    sealed = _sealed(tmp_path)
    registry = sealed["registry"]
    registry.append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("preflight-cluster"),
    )
    registry.close_campaign(sealed["campaign_id"])
    registry.reveal_seed(
        campaign_id=sealed["campaign_id"], seed_capsule=sealed["capsule"]
    )
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute(f"DROP TRIGGER {trigger}")
        connection.execute(f"UPDATE {table} SET {column} = '{{}}'")

    with pytest.raises(RouteShadowRegistryError, match=message):
        registry.verify_assignment_reveals(sealed["campaign_id"])
    with sqlite3.connect(sealed["db_path"]) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM shadow_assignment_reveals"
        ).fetchone()[0] == 0


def test_snapshot_reconstructs_artifacts_and_matches_events_to_evidence_rows(tmp_path):
    sealed = _sealed(tmp_path / "package")
    with sqlite3.connect(sealed["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_seals_no_update")
        connection.execute(
            "UPDATE shadow_campaign_seals SET public_package_json = '{}'"
        )

    package_snapshot = sealed["registry"].snapshot(sealed["campaign_id"])
    assert package_snapshot["event_chain"]["ok"] is True
    assert package_snapshot["campaigns"][0]["artifact_audit_ok"] is False
    assert package_snapshot["ok"] is False

    second = _sealed(tmp_path / "orphan-event")
    second["registry"].append_assignment_commitment(
        campaign_id=second["campaign_id"],
        seed_capsule=second["capsule"],
        cluster_identifier=_session_hash("cluster-to-remove"),
    )
    with sqlite3.connect(second["db_path"]) as connection:
        connection.execute("DROP TRIGGER shadow_commitments_no_delete")
        connection.execute("DELETE FROM shadow_assignment_commitments")

    orphan_snapshot = second["registry"].snapshot(second["campaign_id"])
    assert orphan_snapshot["event_chain"]["ok"] is True
    assert orphan_snapshot["event_artifact_consistency"] == {
        "ok": False,
        "registered_event_artifacts": 2,
        "evidence_artifacts": 1,
        "reason": "event_evidence_artifact_mismatch",
    }
    assert orphan_snapshot["ok"] is False


def test_duplicate_cluster_commit_is_linearizable_under_concurrency(tmp_path):
    sealed = _sealed(tmp_path)
    workers = 8
    barrier = Barrier(workers)

    def append_once(_):
        registry = RouteShadowAssignmentRegistry(sealed["db_path"])
        barrier.wait()
        return registry.append_assignment_commitment(
            campaign_id=sealed["campaign_id"],
            seed_capsule=sealed["capsule"],
            cluster_identifier=_session_hash("same-cluster"),
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(append_once, range(workers)))

    assert sum(result["created"] is True for result in results) == 1
    assert len({result["commitment"]["commitment_hash"] for result in results}) == 1
    snapshot = sealed["registry"].snapshot(sealed["campaign_id"])
    assert snapshot["campaigns"][0]["commitment_count"] == 1
    assert snapshot["event_chain"]["verified_events"] == 2


def test_close_vs_commit_race_freezes_exactly_the_linearized_count(tmp_path):
    sealed = _sealed(tmp_path)
    sealed["registry"].append_assignment_commitment(
        campaign_id=sealed["campaign_id"],
        seed_capsule=sealed["capsule"],
        cluster_identifier=_session_hash("existing-cluster"),
    )
    barrier = Barrier(2)

    def append_second():
        registry = RouteShadowAssignmentRegistry(sealed["db_path"])
        barrier.wait()
        try:
            return ("append", registry.append_assignment_commitment(
                campaign_id=sealed["campaign_id"],
                seed_capsule=sealed["capsule"],
                cluster_identifier=_session_hash("racing-cluster"),
            ))
        except RouteShadowRegistryError as exc:
            return ("append_error", str(exc))

    def close():
        registry = RouteShadowAssignmentRegistry(sealed["db_path"])
        barrier.wait()
        return ("close", registry.close_campaign(sealed["campaign_id"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(append_second), pool.submit(close)]
        outcomes = [future.result() for future in results]

    snapshot = sealed["registry"].snapshot(sealed["campaign_id"])
    campaign = snapshot["campaigns"][0]
    assert campaign["state"] == "commitments_closed"
    assert campaign["frozen_commitment_count"] == campaign["commitment_count"]
    assert campaign["commitment_count"] in (1, 2)
    append_outcome = next(value for kind, value in outcomes if kind.startswith("append"))
    if campaign["commitment_count"] == 1:
        assert isinstance(append_outcome, str)
        assert "commitments are closed" in append_outcome
    else:
        assert append_outcome["created"] is True


def test_unknown_schema_version_is_rejected_without_migration(tmp_path):
    db_path = tmp_path / "future.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA user_version = 999")
    before = db_path.read_bytes()
    with pytest.raises(RouteShadowRegistryError, match="unsupported shadow registry schema 999"):
        RouteShadowAssignmentRegistry(db_path)
    assert db_path.read_bytes() == before


def test_unrelated_unversioned_database_is_rejected_without_pollution(tmp_path):
    db_path = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE unrelated_records(value TEXT NOT NULL)")
        connection.execute("INSERT INTO unrelated_records(value) VALUES ('keep-me')")
    before = db_path.read_bytes()

    with pytest.raises(
        RouteShadowRegistryError,
        match="refusing to initialize a non-empty unversioned SQLite database",
    ):
        RouteShadowAssignmentRegistry(db_path)

    assert db_path.read_bytes() == before
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT value FROM unrelated_records").fetchall() == [
            ("keep-me",)
        ]
        assert connection.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'shadow_%'"
        ).fetchone()[0] == 0
