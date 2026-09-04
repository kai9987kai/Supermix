"""Bounded nonce ledgers for Nexus renderer freshness.

The default API path uses :class:`InMemoryNonceLedger`. Deployments that need
replay protection across worker processes or restarts may opt into
:class:`SQLiteNonceLedger`; it stores only SHA-256 nonce digests and timestamps,
never the raw browser nonce, query, output, or proof capsule.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Protocol


_SCHEMA_INIT_LOCK = threading.Lock()


class NonceLedger(Protocol):
    """Minimal store contract used by the API freshness boundary."""

    def seen(self, key: str) -> bool:
        """Return whether ``key`` was accepted within the freshness window."""

    def consume(self, key: str) -> bool:
        """Atomically accept ``key`` once, returning false for a replay."""


class NonceLedgerCapacityError(RuntimeError):
    """Raised when every bounded ledger slot contains an unexpired nonce."""


class InMemoryNonceLedger:
    """Thread-safe bounded process-local nonce ledger."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 15 * 60,
        max_entries: int = 4096,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, float] = OrderedDict()

    def _purge(self, now: float) -> None:
        expired = [
            key
            for key, seen_at in self._entries.items()
            if now - seen_at >= self.ttl_seconds
        ]
        for key in expired:
            self._entries.pop(key, None)

    def seen(self, key: str) -> bool:
        if not key:
            return False
        now = float(self._clock())
        with self._lock:
            self._purge(now)
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

    def consume(self, key: str) -> bool:
        if not key:
            return True
        now = float(self._clock())
        with self._lock:
            self._purge(now)
            if key in self._entries:
                self._entries.move_to_end(key)
                return False
            if len(self._entries) >= self.max_entries:
                raise NonceLedgerCapacityError(
                    "nonce ledger capacity is occupied by unexpired entries"
                )
            self._entries[key] = now
            return True

    def __len__(self) -> int:
        with self._lock:
            self._purge(float(self._clock()))
            return len(self._entries)


class SQLiteNonceLedger:
    """Optional durable nonce ledger shared by local API workers.

    Each operation opens a short-lived connection, enables WAL and a bounded
    busy timeout, and serializes the check/insert path with ``BEGIN IMMEDIATE``.
    The timestamp is wall-clock time because entries must survive process
    restarts; callers should still treat this as freshness protection rather
    than authentication.
    """

    _TABLE = "nexus_verification_nonces"

    def __init__(
        self,
        path: str | Path,
        *,
        ttl_seconds: float = 15 * 60,
        max_entries: int = 4096,
        busy_timeout_ms: int = 5000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0 or busy_timeout_ms <= 0:
            raise ValueError("ttl_seconds, max_entries, and busy_timeout_ms must be positive")
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError(f"nonce ledger path is a directory: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._clock = clock
        with _SCHEMA_INIT_LOCK:
            self._init_schema()

    def _connect(self, *, configure_wal: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        if configure_wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        last_error: Exception | None = None
        for attempt in range(8):
            conn: sqlite3.Connection | None = None
            try:
                conn = self._connect(configure_wal=True)
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        nonce_sha256 TEXT PRIMARY KEY,
                        seen_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_seen_at_idx "
                    f"ON {self._TABLE}(seen_at)"
                )
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == 7:
                    raise
                time.sleep(0.05 * (2**attempt))
            finally:
                if conn is not None:
                    conn.close()
        if last_error is not None:
            raise last_error

    def _purge(self, conn: sqlite3.Connection, now: float) -> None:
        conn.execute(
            f"DELETE FROM {self._TABLE} WHERE seen_at <= ?",
            (now - self.ttl_seconds,),
        )

    def seen(self, key: str) -> bool:
        if not key:
            return False
        now = float(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._purge(conn, now)
            row = conn.execute(
                f"SELECT 1 FROM {self._TABLE} WHERE nonce_sha256 = ?",
                (key,),
            ).fetchone()
            conn.commit()
            return row is not None
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def consume(self, key: str) -> bool:
        if not key:
            return True
        now = float(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._purge(conn, now)
            row = conn.execute(
                f"SELECT 1 FROM {self._TABLE} WHERE nonce_sha256 = ?",
                (key,),
            ).fetchone()
            if row is not None:
                conn.commit()
                return False
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM {self._TABLE}"
            ).fetchone()
            if count_row is not None and int(count_row[0]) >= self.max_entries:
                raise NonceLedgerCapacityError(
                    "nonce ledger capacity is occupied by unexpired entries"
                )
            conn.execute(
                f"INSERT INTO {self._TABLE}(nonce_sha256, seen_at) VALUES (?, ?)",
                (key, now),
            )
            conn.commit()
            return True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def __len__(self) -> int:
        now = float(self._clock())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._purge(conn, now)
            row = conn.execute(f"SELECT COUNT(*) FROM {self._TABLE}").fetchone()
            conn.commit()
            return int(row[0]) if row else 0
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


__all__ = [
    "NonceLedger",
    "NonceLedgerCapacityError",
    "InMemoryNonceLedger",
    "SQLiteNonceLedger",
]
