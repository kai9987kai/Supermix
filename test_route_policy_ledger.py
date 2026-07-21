import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

import source.route_policy_ledger as route_policy_ledger
from source.route_policy_ledger import (
    DECISION_FINGERPRINT_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
    OUTCOME_CONTRACT_SCHEMA_VERSION,
    OUTCOME_MATURITY_SCHEMA_VERSION,
    OUTCOME_NAMES,
    DecisionNotFoundError,
    EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION,
    EXECUTED_ASSIGNMENT_RECORD_SCHEMA_VERSION,
    LedgerConflictError,
    RoutePolicyLedger,
    build_route_outcome_contracts,
    hash_session_identity,
)


EXECUTED_ASSIGNMENT_COMMITMENT = (
    f"{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:" + "1" * 64
)


def _begin(ledger, session_id="session-secret", **overrides):
    force_issue = overrides.pop("_issue_assignment", None)
    values = {
        "session_id": session_id,
        "policy_name": "auto-route",
        "policy_version": "auto-route-v2",
        "policy_schema_version": "2",
        "decision_context": {"action_mode": "chat", "budget": "balanced"},
        "eligible_modes": ["off", "collective"],
        "chosen_mode": "collective",
        "action_probabilities": {"off": 0.2, "collective": 0.8},
        "logging_support": {
            "schema_version": "route-support-v1",
            "decision_type": "randomized",
            "probability_stage": "post_filter",
            "sampler": {
                "name": "test_rng",
                "version": "1",
                "exploration_rate": 0.2,
                "assignment_unit": "route",
                "assignment_commitment": None,
            },
            "candidates": [{"action": "off"}, {"action": "collective"}],
            "exclusions": [],
        },
        "estimated_economics": {"estimated_model_calls": 3, "estimated_cost_units": 3.0},
    }
    logging_support_overridden = "logging_support" in overrides
    values.update(overrides)
    should_issue = not logging_support_overridden if force_issue is None else bool(force_issue)
    if should_issue:
        issued = ledger.issue_execution_assignment(
            session_id=values["session_id"],
            policy_name=values["policy_name"],
            policy_version=values["policy_version"],
            policy_schema_version=values["policy_schema_version"],
            decision_context=values["decision_context"],
            eligible_modes=values["eligible_modes"],
            chosen_mode=values["chosen_mode"],
            action_probabilities=values["action_probabilities"],
            logging_support=values["logging_support"],
            route_id=values.get("route_id"),
        )
        values["route_id"] = issued["route_id"]
        values["logging_support"] = issued["logging_support"]
    return ledger.begin_decision(**values)


def _create_v1_database(path, *, probabilities_json='{"off":1.0,"collective":0.0}'):
    route_id = str(uuid.uuid4())
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ledger_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE session_counters (
            session_hash TEXT PRIMARY KEY,
            next_sequence INTEGER NOT NULL
        );
        CREATE TABLE route_decisions (
            route_id TEXT PRIMARY KEY,
            session_hash TEXT NOT NULL,
            session_sequence INTEGER NOT NULL,
            ledger_schema_version INTEGER NOT NULL,
            policy_name TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            policy_schema_version TEXT NOT NULL,
            decision_context_json TEXT NOT NULL,
            eligible_modes_json TEXT NOT NULL,
            action_probabilities_json TEXT NOT NULL,
            chosen_mode TEXT NOT NULL,
            executed_mode TEXT,
            estimated_economics_json TEXT NOT NULL,
            actual_economics_json TEXT,
            status TEXT NOT NULL,
            success INTEGER,
            error_category TEXT,
            error_message TEXT,
            started_at REAL NOT NULL,
            completed_at REAL,
            UNIQUE (session_hash, session_sequence)
        );
        CREATE TABLE route_feedback_revisions (
            route_id TEXT NOT NULL REFERENCES route_decisions(route_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            idempotency_key TEXT NOT NULL,
            feedback_json TEXT NOT NULL,
            recorded_at REAL NOT NULL,
            PRIMARY KEY (route_id, revision),
            UNIQUE (route_id, idempotency_key)
        );
        PRAGMA user_version = 1;
        """
    )
    connection.execute(
        """
        INSERT INTO route_decisions(
            route_id, session_hash, session_sequence, ledger_schema_version,
            policy_name, policy_version, policy_schema_version,
            decision_context_json, eligible_modes_json, action_probabilities_json,
            chosen_mode, executed_mode, estimated_economics_json, actual_economics_json,
            status, success, started_at, completed_at
        ) VALUES (?, ?, 1, 1, 'auto-route', 'v1', 'context-v1', ?, ?, ?,
                  'off', 'off', '{}', '{"cost_units":1.0}', 'completed', 1, 1.0, 2.0)
        """,
        (
            route_id,
            hash_session_identity("legacy-session"),
            '{"action_mode":"text","score":1,"allowed_agent_modes":["off","collective"]}',
            '["off","collective"]',
            probabilities_json,
        ),
    )
    connection.execute(
        """
        INSERT INTO route_feedback_revisions(
            route_id, revision, idempotency_key, feedback_json, recorded_at
        ) VALUES (?, 1, 'legacy-feedback', '{"rating":"up"}', 3.0)
        """,
        (route_id,),
    )
    connection.commit()
    connection.close()
    return route_id


def _create_v2_database(path):
    route_id = _create_v1_database(path)
    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE route_decisions SET ledger_schema_version = 2 WHERE route_id = ?",
        (route_id,),
    )
    connection.execute("PRAGMA user_version = 2")
    connection.commit()
    connection.close()
    return route_id


def test_begin_decision_persists_provenance_json_and_hashed_session(tmp_path) -> None:
    database = tmp_path / "route-policy.sqlite3"
    ledger = RoutePolicyLedger(database)

    row = _begin(ledger)

    assert uuid.UUID(row["route_id"])
    assert row["session_hash"] == hash_session_identity("session-secret")
    assert row["session_hash"] != "session-secret"
    assert row["session_sequence"] == 1
    assert row["ledger_schema_version"] == LEDGER_SCHEMA_VERSION
    assert row["policy_name"] == "auto-route"
    assert row["policy_version"] == "auto-route-v2"
    assert row["policy_schema_version"] == "2"
    assert row["decision_context"] == {"action_mode": "chat", "budget": "balanced"}
    assert row["eligible_modes"] == ["off", "collective"]
    assert row["action_probabilities"] == {"collective": 0.8, "off": 0.2}
    assert row["decision_type"] == "randomized"
    assert row["probability_stage"] == "post_filter"
    assert row["chosen_probability"] == 0.8
    assert len(row["candidate_set_hash"]) == 64
    assert len(row["distribution_hash"]) == 64
    assert len(row["decision_record_fingerprint"]) == 64
    assert row["decision_record_fingerprint_valid"] is True
    assert row["decision_record_fingerprint_reason"] == "verified"
    assert row["execution_assignment_provenance_valid"] is True
    assert row["execution_assignment_provenance_reason"] == "verified"
    provenance = row["execution_assignment_provenance"]
    assert provenance["required"] is True
    assert provenance["record"]["schema_version"] == EXECUTED_ASSIGNMENT_RECORD_SCHEMA_VERSION
    assert provenance["record"]["route_id"] == row["route_id"]
    assert provenance["record"]["session_hash"] == row["session_hash"]
    assert (
        row["logging_support"]["decision_record_fingerprint_schema_version"]
        == DECISION_FINGERPRINT_SCHEMA_VERSION
    )
    assert (
        row["logging_support"]["decision_record_fingerprint"]
        == row["decision_record_fingerprint"]
    )
    assert row["chosen_mode"] == "collective"
    assert row["estimated_economics"]["estimated_model_calls"] == 3
    assert row["status"] == "inflight"
    assert row["success"] is None
    assert row["feedback_status"] == "unknown"
    assert row["started_at_utc"].endswith("Z")

    connection = sqlite3.connect(database)
    stored = connection.execute("SELECT session_hash FROM route_decisions").fetchone()[0]
    columns = [column[1] for column in connection.execute("PRAGMA table_info(route_decisions)")]
    journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    connection.close()
    assert stored == row["session_hash"]
    assert "session_id" not in columns
    assert journal_mode.lower() == "wal"


def test_two_phase_success_failure_and_inflight_survive_restart(tmp_path) -> None:
    database = tmp_path / "restart.sqlite3"
    ledger = RoutePolicyLedger(database)
    successful = _begin(ledger, route_id=str(uuid.uuid4()))
    failed = _begin(ledger, route_id=str(uuid.uuid4()))
    inflight = _begin(ledger, route_id=str(uuid.uuid4()))

    complete = ledger.complete_decision(
        successful["route_id"],
        success=True,
        executed_mode="collective",
        actual_economics={"model_calls": 2, "cost_units": 2.0, "elapsed_ms": 40.0},
    )
    failure = ledger.complete_decision(
        failed["route_id"],
        success=False,
        executed_mode="collective",
        actual_economics={"model_calls": 1, "cost_units": 1.0, "elapsed_ms": 20.0},
        error_category="provider_error",
        error_message="upstream request failed",
    )

    assert complete["status"] == "completed"
    assert complete["success"] is True
    assert complete["actual_economics"]["cost_units"] == 2.0
    assert complete["completed_at_utc"].endswith("Z")
    assert failure["status"] == "failed"
    assert failure["success"] is False
    assert failure["error_category"] == "provider_error"

    reopened = RoutePolicyLedger(database)
    assert reopened.get_decision(inflight["route_id"])["status"] == "inflight"
    assert reopened.get_decision(failed["route_id"])["error_message"] == "upstream request failed"
    assert reopened.report()["counts"] == {"started": 3, "completed": 1, "failed": 1, "inflight": 1}


def test_complete_decision_is_idempotent_but_conflicts_fail_closed(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "complete.sqlite3")
    decision = _begin(ledger)
    arguments = {
        "success": True,
        "executed_mode": "collective",
        "actual_economics": {"model_calls": 2},
    }

    first = ledger.complete_decision(decision["route_id"], **arguments)
    second = ledger.complete_decision(decision["route_id"], **arguments)

    assert second["completed_at"] == first["completed_at"]
    with pytest.raises(LedgerConflictError):
        ledger.complete_decision(
            decision["route_id"],
            success=True,
            executed_mode="off",
            actual_economics={"model_calls": 1},
        )
    with pytest.raises(ValueError, match="error_category"):
        ledger.complete_decision(str(uuid.uuid4()), success=False)
    with pytest.raises(DecisionNotFoundError):
        ledger.complete_decision(str(uuid.uuid4()), success=True)


def test_session_sequences_are_monotonic_and_independent(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "sequences.sqlite3")

    a1 = _begin(ledger, session_id="session-a")
    b1 = _begin(ledger, session_id="session-b")
    a2 = _begin(ledger, session_id="session-a")

    assert [a1["session_sequence"], a2["session_sequence"]] == [1, 2]
    assert b1["session_sequence"] == 1
    assert ledger.report(session_id="session-a")["counts"]["started"] == 2
    assert ledger.report(session_id="session-b")["counts"]["started"] == 1


def test_feedback_revisions_are_idempotent_and_missing_is_unknown(tmp_path) -> None:
    database = tmp_path / "feedback.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    assert ledger.get_decision(decision["route_id"])["feedback_status"] == "unknown"
    first = ledger.record_feedback(
        decision["route_id"], {"rating": "up", "intent": "good"}, idempotency_key="ui-submit-1"
    )
    duplicate = ledger.record_feedback(
        decision["route_id"], {"intent": "good", "rating": "up"}, idempotency_key="ui-submit-1"
    )
    second = ledger.record_feedback(
        decision["route_id"], {"rating": "down", "intent": "too_slow"}, idempotency_key="ui-submit-2"
    )

    assert first["revision"] == 1
    assert first["idempotent"] is False
    assert duplicate["revision"] == 1
    assert duplicate["recorded_at"] == first["recorded_at"]
    assert duplicate["idempotent"] is True
    assert second["revision"] == 2
    with pytest.raises(LedgerConflictError):
        ledger.record_feedback(
            decision["route_id"], {"rating": "down"}, idempotency_key="ui-submit-1"
        )

    reopened = RoutePolicyLedger(database)
    row = reopened.get_decision(decision["route_id"])
    assert row["feedback_status"] == "known"
    assert row["feedback_revision_count"] == 2
    assert row["latest_feedback"]["feedback"]["intent"] == "too_slow"
    assert [item["revision"] for item in reopened.feedback_history(decision["route_id"])] == [1, 2]


def test_content_derived_retry_does_not_suppress_a_later_deliberate_revision(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "feedback-content-retry.sqlite3")
    decision = _begin(ledger)

    first_up = ledger.record_feedback(decision["route_id"], {"rating": "up"})
    retry_up = ledger.record_feedback(decision["route_id"], {"rating": "up"})
    down = ledger.record_feedback(decision["route_id"], {"rating": "down"})
    second_up = ledger.record_feedback(decision["route_id"], {"rating": "up"})

    assert first_up["revision"] == retry_up["revision"] == 1
    assert retry_up["idempotent"] is True
    assert down["revision"] == 2
    assert second_up["revision"] == 3
    assert second_up["idempotent"] is False
    assert [row["feedback"]["rating"] for row in ledger.feedback_history(decision["route_id"])] == [
        "up",
        "down",
        "up",
    ]


def test_explicit_feedback_request_id_remains_retry_idempotent_across_revisions(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "feedback-explicit-retry.sqlite3")
    decision = _begin(ledger)

    first = ledger.record_feedback(
        decision["route_id"], {"rating": "up"}, idempotency_key="request-1"
    )
    ledger.record_feedback(decision["route_id"], {"rating": "down"})
    retry = ledger.record_feedback(
        decision["route_id"], {"rating": "up"}, idempotency_key="request-1"
    )

    assert retry["revision"] == first["revision"] == 1
    assert retry["recorded_at"] == first["recorded_at"]
    assert retry["idempotent"] is True
    assert len(ledger.feedback_history(decision["route_id"])) == 2


def test_decision_feedback_projection_uses_one_snapshot_and_one_aggregate_query(
    tmp_path, monkeypatch
) -> None:
    ledger = RoutePolicyLedger(tmp_path / "feedback-projection.sqlite3")
    decision = _begin(ledger)
    ledger.record_feedback(decision["route_id"], {"rating": "up"})
    ledger.record_feedback(decision["route_id"], {"rating": "down"})

    statements = []
    original_connect = ledger._connect

    def traced_connect():
        connection = original_connect()
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(ledger, "_connect", traced_connect)
    projected = ledger.get_decision(decision["route_id"])

    feedback_queries = [
        statement
        for statement in statements
        if "FROM route_feedback_revisions" in statement
    ]
    assert projected["feedback_revision_count"] == 2
    assert projected["latest_feedback"]["revision"] == 2
    assert projected["latest_feedback"]["feedback"]["rating"] == "down"
    assert statements[0] == "BEGIN"
    assert len(feedback_queries) == 1
    assert "COUNT(*) AS revision_count" in feedback_queries[0]
    assert "MAX(revision) AS latest_revision" in feedback_queries[0]


def test_report_and_snapshot_feedback_coverage_use_unknown_semantics(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "report.sqlite3")
    completed = _begin(ledger)
    failed = _begin(ledger)
    inflight = _begin(ledger)
    ledger.complete_decision(completed["route_id"], success=True)
    ledger.complete_decision(failed["route_id"], success=False, error_category="runtime_error")
    ledger.record_feedback(completed["route_id"], {"rating": "up"})
    ledger.record_feedback(completed["route_id"], {"rating": "down"}, idempotency_key="revision-2")
    ledger.record_feedback(inflight["route_id"], {"rating": "up"})

    report = ledger.report()
    assert report["counts"] == {"started": 3, "completed": 1, "failed": 1, "inflight": 1}
    assert report["feedback_coverage"] == {
        "known": 2,
        "unknown": 1,
        "coverage_rate": pytest.approx(2 / 3, abs=1e-6),
        "revision_count": 3,
        "terminal_known": 1,
        "terminal_unknown": 1,
        "terminal_coverage_rate": 0.5,
        "missing_feedback_semantics": "unknown",
    }

    snapshot = ledger.snapshot(limit=2)
    assert snapshot["counts"] == report["counts"]
    assert len(snapshot["recent_decisions"]) == 2
    assert snapshot["recent_decisions"][0]["session_sequence"] > snapshot["recent_decisions"][1]["session_sequence"]


def test_concurrent_begins_allocate_unique_monotonic_session_sequences(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "concurrent.sqlite3", timeout_seconds=10)

    def begin(index):
        return _begin(
            ledger,
            session_id="shared-session",
            route_id=str(uuid.uuid4()),
            decision_context={"index": index},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        decisions = list(executor.map(begin, range(24)))

    assert sorted(row["session_sequence"] for row in decisions) == list(range(1, 25))
    assert len({row["route_id"] for row in decisions}) == 24
    assert ledger.report(session_id="shared-session")["counts"] == {
        "started": 24,
        "completed": 0,
        "failed": 0,
        "inflight": 24,
    }


def test_concurrent_feedback_revisions_are_unique_and_complete(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "concurrent-feedback.sqlite3", timeout_seconds=10)
    decision = _begin(ledger)

    def revise(index):
        return ledger.record_feedback(
            decision["route_id"],
            {"rating": "up", "index": index},
            idempotency_key=f"revision-{index}",
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(revise, range(12)))

    assert sorted(row["revision"] for row in results) == list(range(1, 13))
    assert len(ledger.feedback_history(decision["route_id"])) == 12


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"route_id": "not-a-uuid"}, "valid UUID"),
        ({"eligible_modes": ["off", "off"]}, "duplicates"),
        ({"chosen_mode": "loop"}, "present in eligible_modes"),
        ({"action_probabilities": {"off": 1.0}}, "exactly the eligible"),
        ({"action_probabilities": {"off": 0.7, "collective": 0.7}}, "sum to 1"),
        ({"action_probabilities": {"off": 1.0, "collective": 0.0}}, "positive logged probability"),
        ({"decision_context": {"score": float("nan")}}, "JSON serializable"),
    ],
)
def test_begin_validation_fails_before_writing(tmp_path, overrides, message) -> None:
    ledger = RoutePolicyLedger(tmp_path / "validation.sqlite3")
    with pytest.raises(ValueError, match=message):
        _begin(ledger, **overrides)
    assert ledger.report()["counts"]["started"] == 0


def test_supplied_uuid_and_duplicate_route_conflict(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "route-id.sqlite3")
    supplied = uuid.uuid4().hex
    first = _begin(ledger, route_id=supplied)
    assert first["route_id"] == supplied
    with pytest.raises(LedgerConflictError):
        _begin(ledger, route_id=supplied)
    assert ledger.report()["counts"]["started"] == 1


def test_unknown_route_feedback_and_durable_path_validation(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "unknown.sqlite3")
    with pytest.raises(DecisionNotFoundError):
        ledger.record_feedback(str(uuid.uuid4()), {"rating": "up"})
    with pytest.raises(DecisionNotFoundError):
        ledger.get_decision(str(uuid.uuid4()))
    with pytest.raises(ValueError, match="durable filesystem path"):
        RoutePolicyLedger(":memory:")


def test_v1_migration_preserves_lifecycle_feedback_and_backfills_support(tmp_path) -> None:
    database = tmp_path / "legacy-v1.sqlite3"
    route_id = _create_v1_database(database)

    ledger = RoutePolicyLedger(database)
    row = ledger.get_decision(route_id)

    assert row["status"] == "completed"
    assert row["success"] is True
    assert row["feedback_revision_count"] == 1
    assert row["latest_feedback"]["feedback"]["rating"] == "up"
    assert row["decision_type"] == "deterministic"
    assert row["probability_stage"] == "post_filter"
    assert row["chosen_probability"] == 1.0
    assert row["logging_support"]["migration_source"] == "ledger_schema_v1"
    assert row["logging_support"]["decision_record_fingerprint"] is None
    assert row["decision_record_fingerprint"] is None
    assert row["decision_record_fingerprint_valid"] is False
    assert row["decision_record_fingerprint_reason"] == "legacy_unverifiable"

    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION
    assert connection.execute("SELECT COUNT(*) FROM route_decision_support").fetchone()[0] == 1
    connection.close()


@pytest.mark.parametrize(
    "probabilities_json",
    [
        "{}",
        "[]",
        "null",
        "42",
        '"scalar"',
        "{malformed",
        '{"off":NaN,"collective":0.0}',
        '{"off":Infinity,"collective":0.0}',
        '{"off":1e999,"collective":0.0}',
    ],
)
def test_v1_malformed_probability_vector_is_never_upgraded_to_randomized(
    tmp_path, probabilities_json
) -> None:
    database = tmp_path / "legacy-malformed.sqlite3"
    route_id = _create_v1_database(database, probabilities_json=probabilities_json)

    row = RoutePolicyLedger(database).get_decision(route_id)

    assert row["action_probabilities"] == {}
    assert row["decision_type"] == "legacy_unknown"
    assert row["chosen_probability"] is None
    assert row["logging_support"]["migration_source"] == "ledger_schema_v1"


def test_support_hashes_are_canonical_and_mapping_order_independent(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "hashes.sqlite3")
    first = _begin(ledger)
    second_support = {
        "exclusions": [],
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "sampler": {
            "assignment_commitment": None,
            "assignment_unit": "route",
            "exploration_rate": 0.2,
            "version": "1",
            "name": "test_rng",
        },
        "probability_stage": "post_filter",
        "decision_type": "randomized",
        "schema_version": "route-support-v1",
    }
    second = _begin(
        ledger,
        decision_context={"budget": "balanced", "action_mode": "chat"},
        action_probabilities={"collective": 0.8, "off": 0.2},
        logging_support=second_support,
        _issue_assignment=True,
    )

    assert second["candidate_set_hash"] == first["candidate_set_hash"]
    assert (
        second["execution_assignment_provenance"]["record"][
            "logging_support_without_commitment"
        ]["distribution_hash"]
        == first["execution_assignment_provenance"]["record"][
            "logging_support_without_commitment"
        ]["distribution_hash"]
    )
    assert second["distribution_hash"] != first["distribution_hash"]


@pytest.mark.parametrize(
    ("statement", "replacement"),
    [
        (
            "UPDATE route_decisions SET decision_context_json = ? WHERE route_id = ?",
            '{"action_mode":"chat","budget":"efficiency"}',
        ),
        (
            "UPDATE route_decisions SET policy_name = ? WHERE route_id = ?",
            "tampered-policy",
        ),
        (
            "UPDATE route_decisions SET policy_version = ? WHERE route_id = ?",
            "tampered-version",
        ),
        (
            "UPDATE route_decisions SET policy_schema_version = ? WHERE route_id = ?",
            "tampered-schema",
        ),
        (
            "UPDATE route_decisions SET eligible_modes_json = ? WHERE route_id = ?",
            '["collective","off"]',
        ),
        (
            "UPDATE route_decisions SET action_probabilities_json = ? WHERE route_id = ?",
            '{"collective":0.7,"off":0.3}',
        ),
        (
            "UPDATE route_decisions SET chosen_mode = ? WHERE route_id = ?",
            "off",
        ),
    ],
)
def test_decision_record_fingerprint_detects_sqlite_record_tampering(
    tmp_path, statement, replacement
) -> None:
    database = tmp_path / "fingerprint-tamper.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    connection.execute(statement, (replacement, decision["route_id"]))
    connection.commit()
    connection.close()

    tampered = ledger.get_decision(decision["route_id"])
    assert tampered["decision_record_fingerprint"] == decision["decision_record_fingerprint"]
    assert tampered["decision_record_fingerprint_valid"] is False
    assert tampered["decision_record_fingerprint_reason"] == "fingerprint_mismatch"


def test_decision_record_fingerprint_detects_support_projection_tampering(tmp_path) -> None:
    database = tmp_path / "fingerprint-support-tamper.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE route_decision_support SET candidate_set_hash = ? WHERE route_id = ?",
        ("0" * 64, decision["route_id"]),
    )
    connection.commit()
    connection.close()

    tampered = ledger.get_decision(decision["route_id"])
    assert tampered["decision_record_fingerprint_valid"] is False
    assert tampered["decision_record_fingerprint_reason"] == "support_projection_mismatch"


def test_existing_v2_row_without_fingerprint_is_never_retroactively_fabricated(tmp_path) -> None:
    database = tmp_path / "pre-fingerprint-v2.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    envelope = json.loads(
        connection.execute(
            "SELECT logging_envelope_json FROM route_decision_support WHERE route_id = ?",
            (decision["route_id"],),
        ).fetchone()[0]
    )
    envelope.pop("decision_record_fingerprint", None)
    envelope.pop("decision_record_fingerprint_schema_version", None)
    connection.execute(
        "UPDATE route_decision_support SET logging_envelope_json = ? WHERE route_id = ?",
        (json.dumps(envelope, sort_keys=True, separators=(",", ":")), decision["route_id"]),
    )
    connection.commit()
    connection.close()

    reopened = RoutePolicyLedger(database)
    unverifiable = reopened.get_decision(decision["route_id"])
    assert unverifiable["decision_record_fingerprint"] is None
    assert unverifiable["decision_record_fingerprint_valid"] is False
    assert unverifiable["decision_record_fingerprint_reason"] == "missing_unverifiable"


def test_randomized_label_requires_assignment_commitment_and_v2_requires_vector(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "support-validation.sqlite3")
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.1,
            "assignment_unit": "route",
            "assignment_commitment": None,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    with pytest.raises(ValueError, match="assignment commitment"):
        _begin(ledger, logging_support=support)
    with pytest.raises(ValueError, match="action_probabilities are required"):
        _begin(ledger, action_probabilities=None)
    assert ledger.report()["counts"]["started"] == 0


def test_randomized_commitment_must_be_issued_and_bound_by_same_ledger(tmp_path) -> None:
    first = RoutePolicyLedger(tmp_path / "issuer.sqlite3")
    second = RoutePolicyLedger(tmp_path / "other-ledger.sqlite3")
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": None,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    issued = first.issue_execution_assignment(
        session_id="session-secret",
        policy_name="auto-route",
        policy_version="auto-route-v2",
        policy_schema_version="2",
        decision_context={"action_mode": "chat", "budget": "balanced"},
        eligible_modes=["off", "collective"],
        chosen_mode="collective",
        action_probabilities={"off": 0.2, "collective": 0.8},
        logging_support=support,
    )

    assert issued["provenance"] == "ledger_issued"
    assert issued["assignment_commitment"].startswith(
        f"{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:"
    )
    with pytest.raises(ValueError, match="not issued by this ledger"):
        _begin(
            second,
            route_id=issued["route_id"],
            logging_support=issued["logging_support"],
        )
    assert second.report()["counts"]["started"] == 0


def test_execution_assignment_records_and_bindings_are_append_only(tmp_path) -> None:
    database = tmp_path / "append-only-assignments.sqlite3"
    decision = _begin(RoutePolicyLedger(database))

    connection = sqlite3.connect(database)
    assert connection.execute(
        "SELECT COUNT(*) FROM route_execution_assignment_records"
    ).fetchone()[0] == 1
    assert connection.execute(
        "SELECT COUNT(*) FROM route_execution_assignment_bindings"
    ).fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "UPDATE route_execution_assignment_records SET nonce_hex = ? WHERE route_id = ?",
            ("00" * 32, decision["route_id"]),
        )
    connection.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "DELETE FROM route_execution_assignment_bindings WHERE route_id = ?",
            (decision["route_id"],),
        )
    connection.close()


@pytest.mark.parametrize(
    ("location", "key", "value", "message"),
    [
        ("root", "shadow_only", True, "shadow-only"),
        ("root", "rehearsal_only", True, "rehearsal-only"),
        ("root", "ledger_eligible", False, "ledger_eligible=false"),
        ("sampler", "shadow_only", True, "shadow-only"),
        ("sampler", "rehearsal_only", True, "rehearsal-only"),
        ("sampler", "ledger_eligible", False, "ledger_eligible=false"),
    ],
)
def test_executed_ledger_rejects_shadow_or_rehearsal_support_flags(
    tmp_path, location, key, value, message
) -> None:
    ledger = RoutePolicyLedger(tmp_path / f"shadow-flag-{location}-{key}.sqlite3")
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": EXECUTED_ASSIGNMENT_COMMITMENT,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    target = support if location == "root" else support["sampler"]
    target[key] = value
    with pytest.raises(ValueError, match=message):
        _begin(ledger, logging_support=support)
    assert ledger.report()["counts"]["started"] == 0


@pytest.mark.parametrize(
    "commitment",
    [
        "shadow:campaign:assignment",
        "shadow-v1:opaque-receipt",
        "route-study-shadow-assignment-commitment-v1:abc",
        "route-shadow:abc",
    ],
)
def test_executed_ledger_rejects_reserved_shadow_commitment_namespaces(
    tmp_path, commitment
) -> None:
    ledger = RoutePolicyLedger(tmp_path / "shadow-commitment.sqlite3")
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": commitment,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    with pytest.raises(ValueError, match="shadow assignment commitments"):
        _begin(ledger, logging_support=support)
    assert ledger.report()["counts"]["started"] == 0


def test_executed_ledger_rejects_real_bare_shadow_commitment_hash(tmp_path) -> None:
    from source.route_policy_protocol import build_route_study_review_bundle_from_input
    from source.route_policy_protocol_cli import _example_bundle_input
    from source.route_policy_shadow_registry import (
        create_shadow_campaign_artifacts,
        prepare_shadow_assignment_commitment,
    )

    bundle = build_route_study_review_bundle_from_input(_example_bundle_input())
    artifacts = create_shadow_campaign_artifacts(bundle, bytes(range(32)))
    shadow_commitment = prepare_shadow_assignment_commitment(
        artifacts["public_package"],
        artifacts["private_seed_capsule"],
        hash_session_identity("ledger-boundary-cluster"),
    )["commitment"]["commitment_hash"]
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": shadow_commitment,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    with pytest.raises(ValueError, match="route-execution-assignment-v1"):
        _begin(RoutePolicyLedger(tmp_path / "bare-shadow-hash.sqlite3"), logging_support=support)


def test_executed_ledger_rejects_real_shadow_hash_wrapped_as_execution_commitment(
    tmp_path,
) -> None:
    from source.route_policy_protocol import build_route_study_review_bundle_from_input
    from source.route_policy_protocol_cli import _example_bundle_input
    from source.route_policy_shadow_registry import (
        create_shadow_campaign_artifacts,
        prepare_shadow_assignment_commitment,
    )

    bundle = build_route_study_review_bundle_from_input(_example_bundle_input())
    artifacts = create_shadow_campaign_artifacts(bundle, bytes(range(32)))
    shadow_hash = prepare_shadow_assignment_commitment(
        artifacts["public_package"],
        artifacts["private_seed_capsule"],
        hash_session_identity("ledger-boundary-cluster"),
    )["commitment"]["commitment_hash"]
    wrapped_shadow_hash = (
        f"{EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION}:{shadow_hash}"
    )
    support = {
        "schema_version": "route-support-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": wrapped_shadow_hash,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    ledger = RoutePolicyLedger(tmp_path / "wrapped-shadow-hash.sqlite3")

    with pytest.raises(ValueError, match="not issued by this ledger"):
        _begin(ledger, logging_support=support)
    assert ledger.report()["counts"]["started"] == 0


def test_executed_ledger_rejects_unknown_support_schema(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "support-schema.sqlite3")
    support = {
        "schema_version": "route-study-shadow-assignment-commitment-v1",
        "decision_type": "randomized",
        "probability_stage": "post_filter",
        "sampler": {
            "name": "rng",
            "version": "1",
            "exploration_rate": 0.2,
            "assignment_unit": "route",
            "assignment_commitment": EXECUTED_ASSIGNMENT_COMMITMENT,
        },
        "candidates": [{"action": "off"}, {"action": "collective"}],
        "exclusions": [],
    }
    with pytest.raises(ValueError, match="schema_version must be route-support-v1"):
        _begin(ledger, logging_support=support)
    assert ledger.report()["counts"]["started"] == 0


def test_policy_evidence_snapshot_is_prompt_free_and_separates_missing_feedback(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "evidence.sqlite3")
    decision = ledger.begin_decision(
        session_id="private-session",
        policy_name="auto-route-v2",
        policy_version="2.0.0",
        policy_schema_version="route-context-v1",
        decision_context={
            "action_mode": "text",
            "budget_profile": "balanced",
            "score": 2,
            "allowed_agent_modes": ["off", "collective"],
            "prompt": "private raw prompt",
            "nested": {
                "user_prompt": "private nested prompt",
                "prompt_length": 18,
            },
        },
        eligible_modes=["off", "collective"],
        chosen_mode="collective",
        action_probabilities={"off": 0.0, "collective": 1.0},
    )
    ledger.complete_decision(decision["route_id"], success=True, executed_mode="collective")
    ledger.record_feedback(
        decision["route_id"],
        {"rating": "up", "feedback_axes": {"quality": 1}, "observation_status": "observed"},
    )

    snapshot = ledger.policy_evidence_snapshot(
        session_id="private-session",
        policy_name="auto-route-v2",
        policy_version="2.0.0",
    )

    assert snapshot["analysis_window"]["included_decisions"] == 1
    assert snapshot["analysis_window"]["truncated"] is False
    assert snapshot["usage_rows"][0]["auto_agent_policy"]["decision_type"] == "deterministic"
    assert snapshot["usage_rows"][0]["decision_record_fingerprint_valid"] is True
    assert (
        snapshot["usage_rows"][0]["auto_agent_policy"][
            "decision_record_fingerprint_reason"
        ]
        == "verified"
    )
    assert (
        snapshot["usage_rows"][0]["auto_agent_policy"][
            "decision_record_fingerprint"
        ]
        == decision["decision_record_fingerprint"]
    )
    assert snapshot["feedback_rows"][0]["feedback_axes"]["quality"] == 1
    assert snapshot["expected_context_by_route_id"][decision["route_id"]]["score"] == 2
    assert snapshot["expected_context_by_route_id"][decision["route_id"]]["nested"] == {
        "prompt_length": 18
    }
    assert "private raw prompt" not in str(snapshot)
    assert "private nested prompt" not in str(snapshot)
    assert "private-session" not in str(snapshot)
    assert "prompt" not in snapshot["usage_rows"][0]


def test_route_outcome_contract_builder_is_complete_canonical_and_strict() -> None:
    contracts = build_route_outcome_contracts()

    assert tuple(contracts) == OUTCOME_NAMES
    assert {contract["schema_version"] for contract in contracts.values()} == {
        OUTCOME_CONTRACT_SCHEMA_VERSION
    }
    assert all(contract["precommitted"] is True for contract in contracts.values())
    assert all(contract["commitment_source"] == "safe_default" for contract in contracts.values())
    assert all(len(contract["contract_hash"]) == 64 for contract in contracts.values())
    assert contracts["user_quality_rating"]["value_type"] == "ordinal"
    assert contracts["user_quality_rating"]["unit"] == "signed_unit_interval"

    custom = {name: {} for name in OUTCOME_NAMES}
    custom["user_quality_rating"] = {
        "outcome_definition_version": "quality-v2",
        "observation_policy_id": "quality-dialog",
        "observation_policy_version": "2",
        "maturity_delay_seconds": 30,
    }
    built = build_route_outcome_contracts(custom)
    assert built["user_quality_rating"]["outcome_definition_version"] == "quality-v2"
    assert built["user_quality_rating"]["maturity_delay_seconds"] == 30.0
    assert all(contract["commitment_source"] == "caller" for contract in built.values())

    with pytest.raises(ValueError, match="exactly the four"):
        build_route_outcome_contracts({"route_success": {}})
    custom["cost"] = {"unit": "dollars"}
    with pytest.raises(ValueError, match="value_type or unit"):
        build_route_outcome_contracts(custom)


def test_begin_atomically_precommits_four_append_only_outcome_contracts(tmp_path) -> None:
    database = tmp_path / "contracts.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    assert set(decision["outcome_contracts"]) == set(OUTCOME_NAMES)
    assert decision["outcome_contracts_precommitted_at_begin"] is True
    assert decision["outcome_contracts_defaulted_at_begin"] is True
    assert decision["outcome_contract_commitment_source"] == "safe_default"
    assert all(
        contract["contract_hash_valid"] is True
        and contract["contract_hash_reason"] == "verified"
        and contract["commitment_timing_valid"] is True
        and contract["commitment_timing_reason"] == "verified"
        for contract in decision["outcome_contracts"].values()
    )

    connection = sqlite3.connect(database)
    assert connection.execute("SELECT COUNT(*) FROM route_outcome_contracts").fetchone()[0] == 4
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        connection.execute(
            "UPDATE route_outcome_contracts SET unit = 'changed' WHERE route_id = ?",
            (decision["route_id"],),
        )
    connection.close()

    bad = {name: {} for name in OUTCOME_NAMES}
    bad["latency"] = {"maturity_delay_seconds": float("inf")}
    with pytest.raises(ValueError, match="finite non-negative"):
        _begin(ledger, route_id=str(uuid.uuid4()), outcome_contracts=bad)
    assert ledger.report()["counts"]["started"] == 1


def test_completion_emits_three_events_with_nested_economics_compatibility(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "completion-events.sqlite3")
    decision = _begin(ledger)
    completed = ledger.complete_decision(
        decision["route_id"],
        success=True,
        executed_mode="collective",
        actual_economics={"actual": {"cost_units": 2.5, "elapsed_ms": 41.0}},
    )

    events = {event["outcome_name"]: event for event in completed["outcome_events"]}
    assert set(events) == {"route_success", "cost", "latency"}
    assert events["route_success"]["observation_status"] == "observed"
    assert events["route_success"]["value"] is True
    assert events["cost"]["value"] == 2.5
    assert events["latency"]["value"] == 41.0
    assert {event["event_source"] for event in events.values()} == {"route_completion"}
    assert completed["actual_economics"]["actual"]["cost_units"] == 2.5

    second = _begin(ledger, route_id=str(uuid.uuid4()))
    invalid = ledger.complete_decision(
        second["route_id"],
        success=False,
        error_category="provider_error",
        actual_economics={"cost_units": -1, "elapsed_ms": -20},
    )
    invalid_events = {event["outcome_name"]: event for event in invalid["outcome_events"]}
    assert invalid_events["route_success"]["value"] is False
    assert invalid_events["cost"]["observation_status"] == "not_observed"
    assert invalid_events["cost"]["value"] is None
    assert invalid_events["latency"]["observation_status"] == "not_observed"


def test_feedback_emits_exactly_one_quality_event_per_revision(tmp_path) -> None:
    ledger = RoutePolicyLedger(tmp_path / "quality-events.sqlite3")
    decision = _begin(ledger)

    first = ledger.record_feedback(
        decision["route_id"],
        {"rating": "up", "feedback_intent": "good"},
        idempotency_key="quality-1",
    )
    retry = ledger.record_feedback(
        decision["route_id"],
        {"feedback_intent": "good", "rating": "up"},
        idempotency_key="quality-1",
    )
    ledger.record_feedback(
        decision["route_id"],
        {"rating": "down", "feedback_intent": "too_slow"},
        idempotency_key="quality-2",
    )
    ledger.record_feedback(
        decision["route_id"],
        {"feedback_axes": {"quality": -0.25}},
        idempotency_key="quality-3",
    )
    ledger.record_feedback(
        decision["route_id"],
        {"feedback_axes": {"quality": 2.0}},
        idempotency_key="quality-4",
    )

    assert retry["revision"] == first["revision"] == 1
    assert retry["idempotent"] is True
    events = ledger.get_decision(decision["route_id"])["outcome_events"]
    assert len(events) == 4
    assert [event["event_key"] for event in events] == [
        "feedback_revision:1",
        "feedback_revision:2",
        "feedback_revision:3",
        "feedback_revision:4",
    ]
    assert events[0]["value"] == 1.0
    assert events[0]["metadata"]["raw_rating"] == "up"
    assert events[1]["observation_status"] == "not_observed"
    assert events[1]["value"] is None
    assert events[2]["value"] == -0.25
    assert events[3]["observation_status"] == "not_observed"


@pytest.mark.parametrize("creator", [_create_v1_database, _create_v2_database])
def test_v1_v2_migration_backfills_posthoc_contracts_and_descriptive_events(
    tmp_path, creator
) -> None:
    database = tmp_path / f"{creator.__name__}.sqlite3"
    route_id = creator(database)

    ledger = RoutePolicyLedger(database)
    decision = ledger.get_decision(route_id)

    assert set(decision["outcome_contracts"]) == set(OUTCOME_NAMES)
    assert decision["outcome_contracts_precommitted_at_begin"] is False
    assert decision["outcome_contract_commitment_source"] == "legacy_posthoc"
    assert all(
        contract["precommitted"] is False
        and contract["commitment_source"] == "legacy_posthoc"
        and contract["contract_hash_valid"] is True
        for contract in decision["outcome_contracts"].values()
    )
    events = decision["outcome_events"]
    assert len(events) == 4
    assert {event["event_source"] for event in events} == {"legacy_posthoc"}
    assert {event["outcome_name"] for event in events} == set(OUTCOME_NAMES)
    assert next(event for event in events if event["outcome_name"] == "cost")["value"] == 1.0
    assert (
        next(event for event in events if event["outcome_name"] == "latency")[
            "observation_status"
        ]
        == "not_observed"
    )

    # Reopening repeats neither contracts nor descriptive observations.
    reopened = RoutePolicyLedger(database)
    assert len(reopened.get_decision(route_id)["outcome_events"]) == 4
    connection = sqlite3.connect(database)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == LEDGER_SCHEMA_VERSION
    assert connection.execute("SELECT COUNT(*) FROM route_outcome_contracts").fetchone()[0] == 4
    assert connection.execute("SELECT COUNT(*) FROM route_outcome_observation_events").fetchone()[0] == 4
    connection.close()


def test_policy_evidence_snapshot_uses_one_finite_as_of_for_maturity(monkeypatch, tmp_path) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(route_policy_ledger.time, "time", lambda: clock["now"])
    ledger = RoutePolicyLedger(tmp_path / "fixed-as-of.sqlite3")
    contracts = {name: {} for name in OUTCOME_NAMES}
    contracts["user_quality_rating"] = {"maturity_delay_seconds": 30}
    decision = _begin(ledger, outcome_contracts=contracts)
    clock["now"] = 110.0
    ledger.complete_decision(
        decision["route_id"],
        success=True,
        actual_economics={"cost_units": 1.0, "elapsed_ms": 5.0},
    )
    clock["now"] = 120.0
    ledger.record_feedback(decision["route_id"], {"rating": "up"})

    early = ledger.policy_evidence_snapshot(as_of=105.0)
    assert early["as_of"] == 105.0
    assert early["analysis_window"]["as_of"] == 105.0
    assert early["lifecycle"]["counts"] == {
        "started": 1,
        "completed": 0,
        "failed": 0,
        "inflight": 1,
    }
    assert early["usage_rows"][0]["outcome_events"] == []
    maturity = early["outcome_contract_maturity"]
    assert maturity["schema_version"] == OUTCOME_MATURITY_SCHEMA_VERSION
    assert maturity["policy_value_estimate"] is None
    assert maturity["causal_identification"] == "not_performed"
    assert maturity["missingness_identification"] == "not_performed"
    assert maturity["by_outcome"]["route_success"]["mature_contract_count"] == 1
    assert maturity["by_outcome"]["user_quality_rating"]["pending_contract_count"] == 1

    completed = ledger.policy_evidence_snapshot(as_of=115.0)
    assert completed["lifecycle"]["counts"]["completed"] == 1
    assert len(completed["usage_rows"][0]["outcome_events"]) == 3
    assert completed["lifecycle"]["feedback_coverage"]["known"] == 0

    with_feedback = ledger.policy_evidence_snapshot(as_of=125.0)
    assert len(with_feedback["usage_rows"][0]["outcome_events"]) == 4
    assert with_feedback["lifecycle"]["feedback_coverage"]["known"] == 1
    with pytest.raises(ValueError, match="finite timestamp"):
        ledger.policy_evidence_snapshot(as_of=float("nan"))


def test_contract_hash_tampering_fails_closed_in_maturity(tmp_path) -> None:
    database = tmp_path / "contract-tamper.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER route_outcome_contracts_no_update")
    connection.execute(
        "UPDATE route_outcome_contracts SET contract_json = '{}' WHERE route_id = ? AND outcome_name = 'cost'",
        (decision["route_id"],),
    )
    connection.commit()
    connection.close()

    tampered = ledger.get_decision(decision["route_id"])
    assert tampered["outcome_contracts"]["cost"]["contract_hash_valid"] is False
    assert (
        tampered["outcome_contracts"]["cost"]["contract_hash_reason"]
        == "contract_projection_mismatch"
    )
    assert tampered["outcome_contracts_precommitted_at_begin"] is False
    maturity = ledger.policy_evidence_snapshot(
        as_of=decision["started_at"] + 2.0
    )["outcome_contract_maturity"]
    assert maturity["precommitted_routes"] == 0
    assert maturity["complete_contract_sets"] == 0
    assert maturity["by_outcome"]["cost"]["invalid_contract_count"] == 1
    assert maturity["by_outcome"]["cost"]["precommitted_count"] == 0
    assert maturity["by_outcome"]["cost"]["mature_contract_count"] == 0


def test_contract_projection_type_tampering_fails_closed(tmp_path) -> None:
    database = tmp_path / "contract-type-tamper.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER route_outcome_contracts_no_update")
    raw = json.loads(
        connection.execute(
            "SELECT contract_json FROM route_outcome_contracts "
            "WHERE route_id = ? AND outcome_name = 'cost'",
            (decision["route_id"],),
        ).fetchone()[0]
    )
    raw["precommitted"] = 1
    connection.execute(
        "UPDATE route_outcome_contracts SET contract_json = ? "
        "WHERE route_id = ? AND outcome_name = 'cost'",
        (json.dumps(raw, sort_keys=True, separators=(",", ":")), decision["route_id"]),
    )
    connection.commit()
    connection.close()

    tampered = ledger.get_decision(decision["route_id"])
    assert tampered["outcome_contracts"]["cost"]["contract_hash_valid"] is False
    assert (
        tampered["outcome_contracts"]["cost"]["contract_hash_reason"]
        == "contract_projection_mismatch"
    )
    assert tampered["outcome_contracts_precommitted_at_begin"] is False


def test_late_outcome_contract_commitment_fails_closed(tmp_path) -> None:
    database = tmp_path / "contract-timing-tamper.sqlite3"
    ledger = RoutePolicyLedger(database)
    decision = _begin(ledger)

    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER route_outcome_contracts_no_update")
    connection.execute(
        "UPDATE route_outcome_contracts SET committed_at = ? "
        "WHERE route_id = ? AND outcome_name = 'cost'",
        (decision["started_at"] + 1.0, decision["route_id"]),
    )
    connection.commit()
    connection.close()

    tampered = ledger.get_decision(decision["route_id"])
    cost_contract = tampered["outcome_contracts"]["cost"]
    assert cost_contract["contract_hash_valid"] is True
    assert cost_contract["commitment_timing_valid"] is False
    assert cost_contract["commitment_timing_reason"] == "committed_after_decision_start"
    assert tampered["outcome_contracts_precommitted_at_begin"] is False
    maturity = ledger.policy_evidence_snapshot(
        as_of=decision["started_at"] + 2.0
    )["outcome_contract_maturity"]
    assert maturity["precommitted_routes"] == 0
    assert maturity["by_outcome"]["cost"]["late_commitment_count"] == 1
    assert maturity["by_outcome"]["cost"]["precommitted_count"] == 0
    assert maturity["by_outcome"]["cost"]["mature_contract_count"] == 0
