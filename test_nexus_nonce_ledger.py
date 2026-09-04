"""Tests for process-local and optional durable Nexus nonce ledgers."""

from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

from nexus_nonce_ledger import (
    InMemoryNonceLedger,
    NonceLedgerCapacityError,
    SQLiteNonceLedger,
)


def test_in_memory_ledger_is_bounded_and_expires_with_injected_clock():
    now = [100.0]
    ledger = InMemoryNonceLedger(
        ttl_seconds=5,
        max_entries=2,
        clock=lambda: now[0],
    )

    assert ledger.consume("a") is True
    assert ledger.consume("a") is False
    assert ledger.seen("a") is True
    assert ledger.consume("b") is True
    with pytest.raises(NonceLedgerCapacityError):
        ledger.consume("c")
    assert len(ledger) == 2
    assert ledger.seen("a") is True
    assert ledger.seen("b") is True

    now[0] = 106.0
    assert ledger.seen("b") is False
    assert len(ledger) == 0
    assert ledger.consume("c") is True


def test_sqlite_ledger_survives_a_new_store_instance_and_expires(tmp_path):
    now = [100.0]
    db = tmp_path / "verification-nonces.sqlite"
    first = SQLiteNonceLedger(db, ttl_seconds=5, max_entries=4, clock=lambda: now[0])
    assert first.consume("a" * 64) is True

    second = SQLiteNonceLedger(db, ttl_seconds=5, max_entries=4, clock=lambda: now[0])
    assert second.seen("a" * 64) is True
    assert second.consume("a" * 64) is False

    now[0] = 106.0
    assert second.seen("a" * 64) is False
    assert len(second) == 0


def test_sqlite_ledger_consume_is_atomic_across_workers(tmp_path):
    db = tmp_path / "concurrent-nonces.sqlite"

    def consume_once(_: int) -> bool:
        ledger = SQLiteNonceLedger(db, ttl_seconds=60, max_entries=32)
        return ledger.consume("b" * 64)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(consume_once, range(8)))

    assert sum(results) == 1


def test_sqlite_ledger_fails_closed_at_capacity_without_evicting_live_nonces(tmp_path):
    now = [100.0]
    ledger = SQLiteNonceLedger(
        tmp_path / "bounded-nonces.sqlite",
        ttl_seconds=5,
        max_entries=2,
        clock=lambda: now[0],
    )
    first, second, third = (character * 64 for character in "abc")

    assert ledger.consume(first) is True
    assert ledger.consume(second) is True
    with pytest.raises(NonceLedgerCapacityError):
        ledger.consume(third)
    assert ledger.seen(first) is True
    assert ledger.seen(second) is True
    assert ledger.seen(third) is False

    now[0] = 106.0
    assert ledger.consume(third) is True
