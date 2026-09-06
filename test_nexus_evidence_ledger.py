"""Adversarial tests for the shadow-only source-locked evidence ledger."""

from __future__ import annotations

import itertools
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from nexus_evidence_ledger import (  # noqa: E402
    EvidenceLedgerError,
    EvidenceLedgerFreshnessError,
    EvidenceLedgerValidationError,
    SQLiteEvidenceLedger,
    canonical_uri,
    untrusted_ephemeral_evidence,
)


CHECKER_ID = "test-independent-checker-v1"
CHECKER_SOURCE_SHA256 = "a" * 64


def _checker_registry(*, status="passed"):
    def check(**_payload):
        passed = status == "passed"
        return {
            "status": status,
            "algorithmically_independent": passed,
            "reason": "test_match" if passed else "test_mismatch",
        }

    return {
        CHECKER_ID: {
            "version": "1.0.0",
            "source_sha256": CHECKER_SOURCE_SHA256,
            "check": check,
        }
    }


def _ledger(tmp_path, now=100.0, **kwargs):
    clock = [now]
    return (
        SQLiteEvidenceLedger(
            tmp_path / "evidence.sqlite",
            clock=lambda: clock[0],
            **kwargs,
        ),
        clock,
    )


def _snapshot(ledger, *, content="Alpha fact.\nBeta fact.", **overrides):
    values = {
        "provider": "Research API",
        "uri": "HTTPS://Example.COM/articles/42#fragment",
        "content": content,
        "fetched_at": 90.0,
        "published_at": 80.0,
        "valid_from": 0.0,
        "valid_until": 200.0,
        "extractor_version": "extractor-v1",
        "spans": [
            {"span_id": "s1", "start": 0, "end": 11},
            {"span_id": "s2", "start": 12, "end": len(content)},
        ],
    }
    values.update(overrides)
    values["fetch_capability"] = ledger._issue_server_fetch_capability(
        provider=values["provider"],
        uri=values["uri"],
        content=values["content"],
        fetched_at=values["fetched_at"],
        published_at=values["published_at"],
        event_time=values.get("event_time"),
        mention_time=values.get("mention_time"),
        valid_from=values["valid_from"],
        valid_until=values["valid_until"],
        extractor_version=values["extractor_version"],
        supersedes_snapshot_id=values.get("supersedes_snapshot_id"),
        spans=values["spans"],
    )
    return ledger.record_snapshot(**values)


def _bind_output(ledger, turn_id, sentences):
    bounded = list(sentences)
    capability = ledger._issue_generated_output_capability(turn_id, bounded)
    return ledger.bind_output(turn_id, bounded, output_capability=capability)


def test_canonical_uri_and_caller_evidence_are_bounded():
    assert canonical_uri(" HTTPS://Example.COM/article#section ") == "https://example.com/article"
    assert canonical_uri("https://example.com/a?b=2#old") == "https://example.com/a?b=2"
    ephemeral = untrusted_ephemeral_evidence("caller supplied passage", source="prompt")
    assert ephemeral == {
        "evidence_class": "untrusted_ephemeral",
        "authority": False,
        "persisted": False,
        "source": "prompt",
        "text": "caller supplied passage",
    }
    assert canonical_uri("https://[::1]/v1") == "https://[::1]/v1"
    for invalid in (
        "https://./x",
        "https://example .com/x",
        "https://_example.com/x",
        "https://example.com/a\nnext",
    ):
        with pytest.raises(EvidenceLedgerValidationError):
            canonical_uri(invalid)


def test_only_server_produced_snapshots_are_persisted(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    with pytest.raises(EvidenceLedgerValidationError, match="untrusted_ephemeral"):
        ledger.record_snapshot(
            provider="prompt",
            uri="https://example.com/caller",
            content="caller text",
            fetched_at=100.0,
            origin="caller_supplied",
        )
    assert len(ledger) == 0
    with pytest.raises(EvidenceLedgerValidationError, match="fetch capability"):
        ledger.record_snapshot(
            provider="prompt",
            uri="https://example.com/caller",
            content="caller text",
            fetched_at=100.0,
        )


def test_fetch_capability_is_ledger_bound_immutable_and_metadata_bound(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    capability = ledger._issue_server_fetch_capability(
        provider="api",
        uri="https://example.com/fact",
        content="fact",
        fetched_at=100.0,
        valid_from=0.0,
        valid_until=200.0,
    )
    with pytest.raises(AttributeError, match="immutable"):
        capability.content_sha256 = "0" * 64
    with pytest.raises(EvidenceLedgerValidationError, match="capability"):
        ledger.record_snapshot(
            provider="api",
            uri="https://example.com/fact",
            content="fact",
            fetched_at=100.0,
            valid_from=1.0,
            valid_until=200.0,
            fetch_capability=capability,
        )
    other, _clock = _ledger(tmp_path / "other")
    with pytest.raises(EvidenceLedgerValidationError, match="capability"):
        other.record_snapshot(
            provider="api",
            uri="https://example.com/fact",
            content="fact",
            fetched_at=100.0,
            valid_from=0.0,
            valid_until=200.0,
            fetch_capability=capability,
        )


def test_bounded_iterables_and_utf8_inputs_fail_closed(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    with pytest.raises(EvidenceLedgerValidationError, match="spans"):
        _snapshot(ledger, spans=itertools.repeat({"span_id": "s1", "start": 0, "end": 1}))
    with pytest.raises(EvidenceLedgerValidationError, match="1-128"):
        ledger.seal_turn("too-many", itertools.repeat(("snap-missing", "s1")))
    with pytest.raises(EvidenceLedgerValidationError, match="UTF-8"):
        untrusted_ephemeral_evidence("bad\ud800")
    with pytest.raises(EvidenceLedgerValidationError, match="UTF-8"):
        ledger._issue_server_fetch_capability(
            provider="api", uri="https://example.com/x", content="bad\ud800", fetched_at=1.0
        )
    with pytest.raises(ValueError, match="positive integer"):
        SQLiteEvidenceLedger(tmp_path / "fractional.sqlite", busy_timeout_ms=0.5)
    with pytest.raises(EvidenceLedgerValidationError, match="finite"):
        ledger._issue_server_fetch_capability(
            provider="api", uri="https://example.com/x", content="ok", fetched_at=10**10000
        )


def test_incompatible_existing_schema_fails_fast(tmp_path):
    path = tmp_path / "incompatible.sqlite"
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE nexus_evidence_schema (version TEXT PRIMARY KEY NOT NULL, applied_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO nexus_evidence_schema(version, applied_at) VALUES (?, ?)",
            ("nexus-source-locked-evidence-ledger-v1", 1.0),
        )
        conn.execute(
            "CREATE TABLE nexus_evidence_snapshots (snapshot_id TEXT PRIMARY KEY, content TEXT NOT NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(EvidenceLedgerError, match="explicit reviewed migration"):
        SQLiteEvidenceLedger(path)


def test_concurrent_writers_preserve_every_snapshot(tmp_path):
    path = tmp_path / "concurrent.sqlite"
    ledgers = [SQLiteEvidenceLedger(path) for _ in range(4)]

    def write(index):
        content = f"Concurrent fact {index}."
        return _snapshot(
            ledgers[index % len(ledgers)],
            uri=f"https://example.com/concurrent/{index}",
            content=content,
            fetched_at=100.0 + index,
            spans=[{"span_id": "whole", "start": 0, "end": len(content)}],
        )["snapshot_id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshot_ids = list(pool.map(write, range(24)))
    assert len(set(snapshot_ids)) == 24
    assert len(ledgers[0]) == 24
    health = ledgers[0].health()
    assert health["status"] == "ok"
    assert health["counts"]["snapshots"] == 24
    assert health["counts"]["spans"] == 24


def test_snapshot_is_immutable_hash_bound_and_persists_across_instances(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    assert snapshot["snapshot_id"].startswith("snap-")
    assert snapshot["origin"] == "server_fetch"
    assert snapshot["content_sha256"]
    assert snapshot["spans"][0]["text"] == "Alpha fact."
    assert snapshot["spans"][0]["byte_start"] == 0
    assert snapshot["spans"][0]["byte_end"] == len("Alpha fact.".encode())
    assert snapshot["authority_granted"] is False

    replay = _snapshot(ledger)
    assert replay["snapshot_id"] == snapshot["snapshot_id"]
    with pytest.raises(EvidenceLedgerValidationError, match="immutable snapshot spans"):
        _snapshot(ledger, spans=[{"span_id": "s1", "start": 0, "end": 5}])

    reopened = SQLiteEvidenceLedger(tmp_path / "evidence.sqlite")
    assert reopened.get_snapshot(snapshot["snapshot_id"])["content"] == snapshot["content"]
    assert len(reopened) == 1
    health = reopened.health()
    assert health["journal_mode"] == "wal"
    assert health["append_only"] is True
    assert health["stores_caller_evidence"] is False
    assert health["authority_granted"] is False


def test_append_only_triggers_block_direct_sql_mutation(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "UPDATE nexus_evidence_snapshots SET content = ? WHERE snapshot_id = ?",
                ("tampered", snapshot["snapshot_id"]),
            )
        conn.rollback()
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute(
                "DELETE FROM nexus_evidence_snapshots WHERE snapshot_id = ?",
                (snapshot["snapshot_id"],),
            )
        conn.rollback()
    finally:
        conn.close()
    assert ledger.health()["append_only"] is True


def test_reads_detect_tampering_if_a_privileged_process_disables_trigger(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        conn.execute(
            "DROP TRIGGER nexus_evidence_snapshots_append_only_update"
        )
        conn.commit()
    finally:
        conn.close()
    assert ledger.health()["append_only"] is False

    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        conn.execute(
            "UPDATE nexus_evidence_snapshots SET content = ? WHERE snapshot_id = ?",
            ("tampered", snapshot["snapshot_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(EvidenceLedgerError, match="integrity"):
        ledger.get_snapshot(snapshot["snapshot_id"])


def test_health_detects_noop_trigger_replacement(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    _snapshot(ledger)
    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        conn.execute("DROP TRIGGER nexus_evidence_snapshots_append_only_update")
        conn.execute(
            """
            CREATE TRIGGER nexus_evidence_snapshots_append_only_update
            BEFORE UPDATE ON nexus_evidence_snapshots
            BEGIN SELECT 1; END
            """
        )
        conn.commit()
    finally:
        conn.close()
    assert ledger.health()["append_only"] is False
    assert ledger.health()["status"] == "degraded"


def test_turn_seals_ordered_spans_and_claim_relations_are_auditable(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    sealed = ledger.seal_turn(
        "turn-1",
        [(snapshot["snapshot_id"], "s1"), (snapshot["snapshot_id"], "s2")],
        freshness_required=True,
        now=100.0,
    )
    assert sealed["status"] == "sealed"
    assert sealed["opened_spans"][0]["sequence"] == 0
    assert sealed["opened_spans"][1]["sequence"] == 1
    assert sealed["authority_granted"] is False
    assert ledger.seal_turn(
        "turn-1",
        [(snapshot["snapshot_id"], "s1"), (snapshot["snapshot_id"], "s2")],
        freshness_required=True,
        now=100.0,
    )["evidence_set_sha256"] == sealed["evidence_set_sha256"]
    _bind_output(
        ledger,
        "turn-1",
        [
            "Alpha fact.",
            "The source contains two short facts.",
            "The facts may be related.",
        ],
    )

    quote = ledger.record_claim(
        "turn-1",
        0,
        "Alpha fact.",
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="quotation",
    )
    compression = ledger.record_claim(
        "turn-1",
        1,
        "The source contains two short facts.",
        snapshot_id=snapshot["snapshot_id"],
        span_id="s2",
        relation="compression",
    )
    inference = ledger.record_claim(
        "turn-1",
        2,
        "The facts may be related.",
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="inference",
    )
    assert quote["status"] == "verified_quotation"
    assert quote["quote_verified"] is True
    assert compression["status"] == "auditable_compression"
    assert compression["mechanically_verified"] is False
    assert inference["status"] == "defer_inference"
    assert all(row["authority_granted"] is False for row in (quote, compression, inference))
    evaluated = ledger.evaluate_turn("turn-1", now=100.0)
    assert evaluated["status"] == "coverage_defer"
    assert evaluated["claim_count"] == 3
    assert evaluated["authority_granted"] is False


def test_claims_require_opened_spans_and_exact_quotation(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    ledger.seal_turn("turn-2", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "turn-2", ["A forged quotation.", "Beta fact."])
    with pytest.raises(EvidenceLedgerValidationError, match="quotation"):
        ledger.record_claim(
            "turn-2",
            0,
            "A forged quotation.",
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            relation="quotation",
        )
    with pytest.raises(EvidenceLedgerValidationError, match="opened evidence span"):
        ledger.record_claim(
            "turn-2",
            1,
            "Beta fact.",
            snapshot_id=snapshot["snapshot_id"],
            span_id="s2",
            relation="compression",
        )


def test_claims_cannot_verify_after_source_span_tampering(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    ledger.seal_turn("tamper-turn", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "tamper-turn", ["FORGED"])
    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        conn.execute("DROP TRIGGER nexus_evidence_spans_append_only_update")
        conn.execute(
            "UPDATE nexus_evidence_spans SET text = ? WHERE snapshot_id = ? AND span_id = ?",
            ("FORGED", snapshot["snapshot_id"], "s1"),
        )
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(EvidenceLedgerError, match="integrity"):
        ledger.record_claim(
            "tamper-turn",
            0,
            "FORGED",
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            relation="quotation",
        )


def test_checker_verification_rejects_legacy_assertions_and_forged_receipts(tmp_path):
    ledger, _clock = _ledger(
        tmp_path,
        deterministic_checkers=_checker_registry(),
    )
    snapshot = _snapshot(ledger)
    ledger.seal_turn("checker-turn", [(snapshot["snapshot_id"], "s1")])
    claim_text = "A derived conclusion."
    _bind_output(ledger, "checker-turn", [claim_text])

    with pytest.raises((TypeError, EvidenceLedgerValidationError)):
        ledger.record_claim(
            "checker-turn",
            0,
            claim_text,
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            relation="inference",
            deterministic_checker_id=CHECKER_ID,
            deterministic_verified=True,
        )
    with pytest.raises(EvidenceLedgerValidationError, match="receipt"):
        ledger.record_claim(
            "checker-turn",
            0,
            claim_text,
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            relation="inference",
            checker_receipt={
                "checker_id": CHECKER_ID,
                "status": "passed",
                "algorithmically_independent": True,
            },
        )


def test_checker_receipt_is_ledger_bound_and_exact_claim_bound(tmp_path):
    registry = _checker_registry()
    left, _clock = _ledger(tmp_path / "left", deterministic_checkers=registry)
    right, _clock = _ledger(tmp_path / "right", deterministic_checkers=registry)
    claim_text = "A derived conclusion."
    snapshots = []
    for ledger in (left, right):
        snapshot = _snapshot(ledger)
        snapshots.append(snapshot)
        ledger.seal_turn("checker-turn", [(snapshot["snapshot_id"], "s1"), (snapshot["snapshot_id"], "s2")])
        _bind_output(ledger, "checker-turn", [claim_text])

    receipt = left._run_deterministic_checker(
        "checker-turn",
        0,
        claim_text,
        snapshot_id=snapshots[0]["snapshot_id"],
        span_id="s1",
        checker_id=CHECKER_ID,
        run_nonce="checker-run-nonce-0001",
    )
    with pytest.raises(EvidenceLedgerValidationError, match="receipt"):
        right.record_claim(
            "checker-turn",
            0,
            claim_text,
            snapshot_id=snapshots[1]["snapshot_id"],
            span_id="s1",
            relation="inference",
            checker_receipt=receipt,
        )
    with pytest.raises(EvidenceLedgerValidationError, match="receipt"):
        left.record_claim(
            "checker-turn",
            0,
            claim_text,
            snapshot_id=snapshots[0]["snapshot_id"],
            span_id="s2",
            relation="inference",
            checker_receipt=receipt,
        )


def test_failed_checker_receipt_cannot_mechanically_verify_claim(tmp_path):
    ledger, _clock = _ledger(
        tmp_path,
        deterministic_checkers=_checker_registry(status="failed"),
    )
    snapshot = _snapshot(ledger)
    claim_text = "A derived conclusion."
    ledger.seal_turn("failed-check", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "failed-check", [claim_text])
    receipt = ledger._run_deterministic_checker(
        "failed-check",
        0,
        claim_text,
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        checker_id=CHECKER_ID,
        run_nonce="checker-run-nonce-0002",
    )
    claim = ledger.record_claim(
        "failed-check",
        0,
        claim_text,
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="inference",
        checker_receipt=receipt,
    )
    assert claim["mechanically_verified"] is False
    assert claim["status"] == "defer_inference"
    assert ledger.evaluate_turn("failed-check")["status"] == "coverage_defer"


def test_passed_checker_receipt_survives_restart_and_nonce_replay_is_rejected(tmp_path):
    ledger, _clock = _ledger(
        tmp_path,
        deterministic_checkers=_checker_registry(),
    )
    snapshot = _snapshot(ledger)
    claim_text = "A derived conclusion."
    ledger.seal_turn("passed-check", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "passed-check", [claim_text])
    receipt = ledger._run_deterministic_checker(
        "passed-check",
        0,
        claim_text,
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        checker_id=CHECKER_ID,
        run_nonce="checker-run-nonce-persisted-0001",
    )
    claim = ledger.record_claim(
        "passed-check",
        0,
        claim_text,
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="inference",
        checker_receipt=receipt,
    )
    assert claim["mechanically_verified"] is True
    assert claim["status"] == "checked_inference_no_authority"
    assert ledger.evaluate_turn("passed-check")["status"] == "shadow_recorded"
    assert ledger.health()["counts"]["checker_receipts"] == 1

    reopened = SQLiteEvidenceLedger(tmp_path / "evidence.sqlite")
    assert reopened.health()["status"] == "ok"
    assert reopened.evaluate_turn("passed-check")["status"] == "shadow_recorded"

    ledger.seal_turn("nonce-replay", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "nonce-replay", [claim_text])
    with pytest.raises(EvidenceLedgerValidationError, match="nonce"):
        ledger._run_deterministic_checker(
            "nonce-replay",
            0,
            claim_text,
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            checker_id=CHECKER_ID,
            run_nonce="checker-run-nonce-persisted-0001",
        )


def test_output_binding_is_exact_immutable_and_partial_coverage_defers(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    ledger.seal_turn(
        "coverage-turn",
        [(snapshot["snapshot_id"], "s1"), (snapshot["snapshot_id"], "s2")],
    )
    with pytest.raises(EvidenceLedgerValidationError, match="server-generation"):
        ledger.bind_output("coverage-turn", ["Alpha fact.", "Beta fact."])
    wrong_turn_capability = ledger._issue_generated_output_capability(
        "different-turn", ["Alpha fact.", "Beta fact."]
    )
    with pytest.raises(EvidenceLedgerValidationError, match="does not match"):
        ledger.bind_output(
            "coverage-turn",
            ["Alpha fact.", "Beta fact."],
            output_capability=wrong_turn_capability,
        )
    other, _clock = _ledger(tmp_path / "other-output")
    foreign_capability = other._issue_generated_output_capability(
        "coverage-turn", ["Alpha fact.", "Beta fact."]
    )
    with pytest.raises(EvidenceLedgerValidationError, match="another ledger"):
        ledger.bind_output(
            "coverage-turn",
            ["Alpha fact.", "Beta fact."],
            output_capability=foreign_capability,
        )
    first = _bind_output(ledger, "coverage-turn", ["Alpha fact.", "Beta fact."])
    replay = _bind_output(ledger, "coverage-turn", ["Alpha fact.", "Beta fact."])
    assert replay == first
    with pytest.raises(EvidenceLedgerValidationError, match="output"):
        _bind_output(ledger, "coverage-turn", ["Alpha fact.", "Changed fact."])
    with pytest.raises(EvidenceLedgerValidationError, match="output"):
        ledger.record_claim(
            "coverage-turn",
            0,
            "Alpha fact!",
            snapshot_id=snapshot["snapshot_id"],
            span_id="s1",
            relation="compression",
        )

    ledger.record_claim(
        "coverage-turn",
        0,
        "Alpha fact.",
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="quotation",
    )
    assert ledger.evaluate_turn("coverage-turn")["status"] == "coverage_defer"


def test_full_multi_sentence_output_coverage_is_shadow_recorded(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    ledger.seal_turn(
        "complete-turn",
        [(snapshot["snapshot_id"], "s1"), (snapshot["snapshot_id"], "s2")],
    )
    _bind_output(ledger, "complete-turn", ["Alpha fact.", "Beta fact."])
    for index, (claim_text, span_id) in enumerate(
        (("Alpha fact.", "s1"), ("Beta fact.", "s2"))
    ):
        claim = ledger.record_claim(
            "complete-turn",
            index,
            claim_text,
            snapshot_id=snapshot["snapshot_id"],
            span_id=span_id,
            relation="quotation",
        )
        assert claim["mechanically_verified"] is True
    assert ledger.evaluate_turn("complete-turn")["status"] == "shadow_recorded"


def test_claim_tamper_is_detected_after_append_only_trigger_is_restored(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    snapshot = _snapshot(ledger)
    ledger.seal_turn("claim-tamper", [(snapshot["snapshot_id"], "s1")])
    _bind_output(ledger, "claim-tamper", ["Alpha fact."])
    ledger.record_claim(
        "claim-tamper",
        0,
        "Alpha fact.",
        snapshot_id=snapshot["snapshot_id"],
        span_id="s1",
        relation="quotation",
    )

    conn = sqlite3.connect(str(tmp_path / "evidence.sqlite"))
    try:
        conn.execute("DROP TRIGGER nexus_evidence_claims_append_only_update")
        conn.execute(
            "UPDATE nexus_evidence_claims SET claim_text = ?, claim_sha256 = ?",
            ("FORGED", "0" * 64),
        )
        conn.execute(
            """
            CREATE TRIGGER nexus_evidence_claims_append_only_update
            BEFORE UPDATE ON nexus_evidence_claims
            BEGIN
                SELECT RAISE(ABORT, 'evidence ledger is append-only');
            END
            """
        )
        conn.commit()
    finally:
        conn.close()

    health = ledger.health()
    assert health["append_only"] is True
    assert health["integrity_valid"] is False
    assert health["status"] == "degraded"
    with pytest.raises(EvidenceLedgerError, match="integrity"):
        ledger.evaluate_turn("claim-tamper")


def test_freshness_sensitive_turns_defer_unknown_or_expired_snapshots(tmp_path):
    ledger, clock = _ledger(tmp_path)
    unknown = _snapshot(ledger, valid_from=None, valid_until=None)
    with pytest.raises(EvidenceLedgerFreshnessError, match="not current"):
        ledger.seal_turn("unknown-turn", [(unknown["snapshot_id"], "s1")], freshness_required=True)

    partial = _snapshot(
        ledger,
        content="Partial window.",
        spans=[{"span_id": "s1", "start": 0, "end": len("Partial window.")}],
        valid_from=0.0,
        valid_until=None,
    )
    with pytest.raises(EvidenceLedgerFreshnessError, match="not current"):
        ledger.seal_turn("partial-turn", [(partial["snapshot_id"], "s1")], freshness_required=True)

    expired = _snapshot(
        ledger,
        content="Expired fact.",
        spans=[{"span_id": "s1", "start": 0, "end": len("Expired fact.")}],
        valid_from=0.0,
        valid_until=99.0,
    )
    with pytest.raises(EvidenceLedgerFreshnessError, match="not current"):
        ledger.seal_turn("expired-turn", [(expired["snapshot_id"], "s1")], freshness_required=True)

    current = _snapshot(
        ledger,
        content="Current fact.",
        spans=[{"span_id": "s1", "start": 0, "end": len("Current fact.")}],
        valid_from=99.0,
        valid_until=101.0,
    )
    assert ledger.seal_turn(
        "current-turn", [(current["snapshot_id"], "s1")], freshness_required=True
    )["status"] == "sealed"
    clock[0] = 250.0
    assert ledger.seal_turn(
        "current-turn", [(current["snapshot_id"], "s1")], freshness_required=True
    )["status"] == "sealed"
    assert ledger.evaluate_turn("current-turn")["status"] == "freshness_defer"


def test_conflict_is_retained_and_forces_defer_without_deleting_sources(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    left = _snapshot(ledger, content="The value is 3.", spans=[{"span_id": "s1", "start": 0, "end": 14}])
    right = _snapshot(
        ledger,
        content="The value is 4.",
        spans=[{"span_id": "s1", "start": 0, "end": 14}],
        fetched_at=91.0,
    )
    conflict = ledger.declare_conflict(
        left["snapshot_id"], right["snapshot_id"], reason="independent sources disagree"
    )
    assert conflict["authority_granted"] is False
    sealed = ledger.seal_turn(
        "conflict-turn",
        [(left["snapshot_id"], "s1"), (right["snapshot_id"], "s1")],
    )
    assert sealed["status"] == "conflict_defer"
    assert ledger.seal_turn(
        "conflict-turn",
        [(left["snapshot_id"], "s1"), (right["snapshot_id"], "s1")],
    )["status"] == "conflict_defer"
    evaluated = ledger.evaluate_turn("conflict-turn")
    assert evaluated["status"] == "conflict_defer"
    assert len(ledger) == 2


def test_revisions_are_explicit_and_append_only(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    original = _snapshot(ledger, content="Version one.", spans=[{"span_id": "s1", "start": 0, "end": 12}])
    revision = _snapshot(
        ledger,
        content="Version two.",
        spans=[{"span_id": "s1", "start": 0, "end": 12}],
        fetched_at=91.0,
        supersedes_snapshot_id=original["snapshot_id"],
    )
    assert revision["supersedes_snapshot_id"] == original["snapshot_id"]
    assert revision["snapshot_id"] != original["snapshot_id"]
    assert ledger.health()["counts"]["revisions"] == 1
    assert ledger.get_snapshot(original["snapshot_id"]) is not None


def test_invalid_snapshot_metadata_fails_closed(tmp_path):
    ledger, _clock = _ledger(tmp_path)
    with pytest.raises(EvidenceLedgerValidationError):
        ledger.record_snapshot(
            provider="api",
            uri="https://example.com/x",
            content="text",
            fetched_at=100.0,
            valid_from=5.0,
            valid_until=5.0,
        )
    with pytest.raises(EvidenceLedgerValidationError):
        ledger.record_snapshot(
            provider="api",
            uri="https://user:password@example.com/x",
            content="text",
            fetched_at=100.0,
        )
