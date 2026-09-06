"""Shadow-only source-locked temporal evidence ledger.

This module is deliberately separate from conversation memory and from the
answer-authority path.  It records provenance that a future retrieval runtime
could inspect without allowing retrieved text to grant tool, permission,
safety, routing, activation, promotion, or answer authority.

Only server-produced fetch/tool results may become immutable snapshots.  Caller
supplied evidence is represented by :func:`untrusted_ephemeral_evidence` and is
never persisted.  A turn seals an ordered set of exact snapshot spans before a
caller binds exact output-sentence hashes and records quotation, compression, or
inference claims. Quotations are mechanically checked; inference verification
requires an executed, allowlisted, ledger-bound checker receipt. Compression
remains auditable but unproved. All hashes are local integrity metadata, not
signatures, authentication, trusted timestamps, non-equivocation, or proof of
source truth.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import itertools
import json
import math
import re
import secrets
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit


LEDGER_SCHEMA_VERSION = "nexus-source-locked-evidence-ledger-v2"
LEGACY_LEDGER_SCHEMA_VERSIONS = frozenset({"nexus-source-locked-evidence-ledger-v1"})
SNAPSHOT_SCHEMA_VERSION = "nexus-evidence-snapshot-v1"
PROVENANCE_RELATIONS = frozenset({"quotation", "compression", "inference"})
CHECKER_STATUSES = frozenset({"passed", "failed", "not_applicable"})
MAX_PROVIDER_CHARS = 120
MAX_URI_CHARS = 2048
MAX_EXTRACTOR_CHARS = 120
MAX_CONTENT_CHARS = 200_000
MAX_SPANS_PER_SNAPSHOT = 256
MAX_OPENED_SPANS_PER_TURN = 128
MAX_TURN_ID_CHARS = 160
MAX_CLAIM_CHARS = 12_000
MAX_OUTPUT_SENTENCES = 4096
MAX_OUTPUT_CHARS = 200_000
MAX_CHECKER_REASON_CHARS = 1000
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_INIT_LOCK = threading.Lock()


class EvidenceLedgerError(RuntimeError):
    """Base error for the append-only evidence ledger."""


class EvidenceLedgerValidationError(ValueError, EvidenceLedgerError):
    """Raised when untrusted or malformed evidence crosses a ledger boundary."""


class EvidenceLedgerFreshnessError(EvidenceLedgerValidationError):
    """Raised when a freshness-sensitive turn has stale or unknown evidence."""


def _canonical_json(value: Any) -> str:
    """Return deterministic RFC-8785-style JSON for local integrity hashes."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _domain_hash(domain: str, value: Any) -> str:
    payload = ("nexus-evidence-ledger/v1/" + domain + "\0" + _canonical_json(value)).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _capability_mac(key: bytes, domain: str, value: Any) -> str:
    payload = ("nexus-evidence-ledger/v2/capability/" + domain + "\0").encode(
        "utf-8"
    ) + _canonical_json(value).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _text_hash(text: str) -> str:
    payload = b"nexus-evidence-ledger/v1/source-bytes\0" + text.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise EvidenceLedgerValidationError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _clean_text(value: Any, *, field: str, limit: int, required: bool = True) -> str:
    if not isinstance(value, str):
        raise EvidenceLedgerValidationError(f"{field} must be a string")
    text = value.strip()
    if required and not text:
        raise EvidenceLedgerValidationError(f"{field} is required")
    if len(text) > limit:
        raise EvidenceLedgerValidationError(f"{field} exceeds {limit} characters")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        raise EvidenceLedgerValidationError(f"{field} contains a control character")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EvidenceLedgerValidationError(f"{field} is not valid UTF-8") from exc
    return text


def _source_content(value: Any) -> str:
    """Validate source text without trimming bytes that the snapshot hashes."""

    if not isinstance(value, str):
        raise EvidenceLedgerValidationError("content must be a string")
    if not value:
        raise EvidenceLedgerValidationError("content is required")
    if len(value) > MAX_CONTENT_CHARS:
        raise EvidenceLedgerValidationError(f"content exceeds {MAX_CONTENT_CHARS} characters")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise EvidenceLedgerValidationError("content contains a control character")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EvidenceLedgerValidationError("content is not valid UTF-8") from exc
    return value


def _identifier(value: Any, *, field: str, prefix: str = "") -> str:
    text = _clean_text(value, field=field, limit=MAX_TURN_ID_CHARS)
    if not _ID_RE.fullmatch(text):
        raise EvidenceLedgerValidationError(f"{field} has an invalid identifier")
    if prefix and not text.startswith(prefix):
        raise EvidenceLedgerValidationError(f"{field} must start with {prefix!r}")
    return text


def canonical_uri(value: Any) -> str:
    """Canonicalize an HTTP(S) URI while removing fragment-only variation."""

    raw = _clean_text(value, field="canonical_uri", limit=MAX_URI_CHARS)
    if any(ord(char) < 32 for char in raw):
        raise EvidenceLedgerValidationError("canonical_uri contains a control character")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise EvidenceLedgerValidationError("canonical_uri is malformed") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise EvidenceLedgerValidationError("canonical_uri must be an absolute HTTP(S) URI")
    if parsed.username is not None or parsed.password is not None:
        raise EvidenceLedgerValidationError("canonical_uri cannot contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise EvidenceLedgerValidationError("canonical_uri has an invalid port") from exc
    hostname_raw = parsed.hostname
    if not hostname_raw or not hostname_raw.strip(".") or any(char.isspace() for char in hostname_raw):
        raise EvidenceLedgerValidationError("canonical_uri has an invalid host")
    if ":" in hostname_raw:
        try:
            hostname = f"[{ipaddress.ip_address(hostname_raw)}]"
        except ValueError as exc:
            raise EvidenceLedgerValidationError("canonical_uri has an invalid IP host") from exc
    else:
        try:
            hostname = hostname_raw.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError as exc:
            raise EvidenceLedgerValidationError("canonical_uri has an invalid host") from exc
        labels = hostname.split(".")
        if any(
            not label
            or len(label) > 63
            or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label)
            for label in labels
        ):
            raise EvidenceLedgerValidationError("canonical_uri has an invalid host label")
    netloc = hostname if port is None else f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


def _timestamp(value: Any, *, field: str, optional: bool = False) -> Optional[float]:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceLedgerValidationError(f"{field} must be a finite timestamp")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise EvidenceLedgerValidationError(f"{field} must be a finite timestamp") from exc
    if not math.isfinite(result) or result < 0:
        raise EvidenceLedgerValidationError(f"{field} must be a finite non-negative timestamp")
    return 0.0 if result == 0.0 else result


def _normalise_quote(text: str) -> str:
    return " ".join(str(text).split())


class _ServerFetchCapability:
    """In-process capability issued by the server-fetch adapter."""

    __slots__ = (
        "_ledger_token",
        "_provider",
        "_canonical_uri",
        "_fetched_at",
        "_content_sha256",
        "_metadata_sha256",
        "_span_manifest_sha256",
        "_capability_mac",
        "_sealed",
    )

    def __init__(
        self,
        ledger_token: object,
        *,
        provider: str,
        canonical_uri: str,
        fetched_at: float,
        content_sha256: str,
        metadata_sha256: str,
        span_manifest_sha256: str,
        capability_mac: str,
    ) -> None:
        object.__setattr__(self, "_ledger_token", ledger_token)
        object.__setattr__(self, "_provider", provider)
        object.__setattr__(self, "_canonical_uri", canonical_uri)
        object.__setattr__(self, "_fetched_at", fetched_at)
        object.__setattr__(self, "_content_sha256", content_sha256)
        object.__setattr__(self, "_metadata_sha256", metadata_sha256)
        object.__setattr__(self, "_span_manifest_sha256", span_manifest_sha256)
        object.__setattr__(self, "_capability_mac", capability_mac)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("server fetch capabilities are immutable")

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def canonical_uri(self) -> str:
        return self._canonical_uri

    @property
    def fetched_at(self) -> float:
        return self._fetched_at

    @property
    def content_sha256(self) -> str:
        return self._content_sha256

    @property
    def metadata_sha256(self) -> str:
        return self._metadata_sha256

    @property
    def span_manifest_sha256(self) -> str:
        return self._span_manifest_sha256


class _DeterministicCheckerReceipt:
    """Immutable result minted only after a configured checker is executed."""

    __slots__ = (
        "_ledger_token",
        "_turn_id",
        "_sentence_index",
        "_claim_binding_sha256",
        "_snapshot_id",
        "_span_id",
        "_span_sha256",
        "_checker_id",
        "_checker_version",
        "_checker_source_sha256",
        "_status",
        "_algorithmically_independent",
        "_reason_sha256",
        "_run_nonce_sha256",
        "_receipt_sha256",
        "_capability_mac",
        "_sealed",
    )

    def __init__(self, ledger_token: object, **values: Any) -> None:
        object.__setattr__(self, "_ledger_token", ledger_token)
        for name in self.__slots__[1:-1]:
            object.__setattr__(self, name, values[name[1:]])
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("deterministic checker receipts are immutable")

    def _value(self, name: str) -> Any:
        return getattr(self, "_" + name)


class _GeneratedOutputCapability:
    """Immutable output manifest issued inside the trusted generation boundary."""

    __slots__ = (
        "_ledger_token",
        "_turn_id",
        "_sentence_count",
        "_sentence_hashes_json",
        "_output_sha256",
        "_manifest_sha256",
        "_capability_mac",
        "_sealed",
    )

    def __init__(self, ledger_token: object, **values: Any) -> None:
        object.__setattr__(self, "_ledger_token", ledger_token)
        for name in self.__slots__[1:-1]:
            object.__setattr__(self, name, values[name[1:]])
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError("generated output capabilities are immutable")

    def _value(self, name: str) -> Any:
        return getattr(self, "_" + name)


def _snapshot_identity(
    *,
    provider: str,
    canonical_uri_value: str,
    fetched_at: float,
    published_at: Optional[float],
    event_time: Optional[float],
    mention_time: Optional[float],
    valid_from: Optional[float],
    valid_until: Optional[float],
    extractor_version: str,
    content_sha256: str,
    supersedes_snapshot_id: Optional[str],
) -> str:
    return "snap-" + _domain_hash(
        "snapshot",
        {
            "provider": provider,
            "canonical_uri": canonical_uri_value,
            "fetched_at": fetched_at,
            "published_at": published_at,
            "event_time": event_time,
            "mention_time": mention_time,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "extractor_version": extractor_version,
            "content_sha256": content_sha256,
            "supersedes_snapshot_id": supersedes_snapshot_id,
        },
    )[:48]


def _fetch_metadata_hash(
    *,
    published_at: Optional[float],
    event_time: Optional[float],
    mention_time: Optional[float],
    valid_from: Optional[float],
    valid_until: Optional[float],
    extractor_version: str,
    supersedes_snapshot_id: Optional[str],
) -> str:
    return _domain_hash(
        "fetch-receipt-metadata",
        {
            "published_at": published_at,
            "event_time": event_time,
            "mention_time": mention_time,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "extractor_version": extractor_version,
            "supersedes_snapshot_id": supersedes_snapshot_id,
        },
    )


def _span_manifest_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    manifest = [
        {
            "span_id": row["span_id"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "byte_start": row["byte_start"],
            "byte_end": row["byte_end"],
            "span_sha256": row["span_sha256"],
        }
        for row in sorted(rows, key=lambda item: str(item["span_id"]))
    ]
    return _domain_hash("span-manifest", manifest)


def _sqlite_check_fragments(sql: Any) -> frozenset[str]:
    """Extract normalized top-level CHECK clauses without trusting SQL comments."""

    if not isinstance(sql, str) or re.search(r"/\*|--", sql):
        return frozenset()
    lowered = sql.lower()
    fragments: set[str] = set()
    index = 0
    while index < len(sql):
        char = sql[index]
        if char in {"'", '"', "`"}:
            quote = char
            index += 1
            while index < len(sql):
                if sql[index] == quote:
                    if index + 1 < len(sql) and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue
        if char == "[":
            index = sql.find("]", index + 1)
            if index < 0:
                return frozenset()
            index += 1
            continue
        if not lowered.startswith("check", index) or (
            index > 0 and (lowered[index - 1].isalnum() or lowered[index - 1] == "_")
        ):
            index += 1
            continue
        after_word = index + len("check")
        if after_word < len(sql) and (
            lowered[after_word].isalnum() or lowered[after_word] == "_"
        ):
            index += 1
            continue
        opening = after_word
        while opening < len(sql) and sql[opening].isspace():
            opening += 1
        if opening >= len(sql) or sql[opening] != "(":
            index += 1
            continue
        depth = 0
        cursor = opening
        quote: Optional[str] = None
        while cursor < len(sql):
            current = sql[cursor]
            if quote is not None:
                if current == quote:
                    if cursor + 1 < len(sql) and sql[cursor + 1] == quote:
                        cursor += 2
                        continue
                    quote = None
                cursor += 1
                continue
            if current in {"'", '"', "`"}:
                quote = current
            elif current == "(":
                depth += 1
            elif current == ")":
                depth -= 1
                if depth == 0:
                    cursor += 1
                    fragments.add(re.sub(r"\s+", "", lowered[index:cursor]))
                    break
            cursor += 1
        else:
            return frozenset()
        index = cursor
    return frozenset(fragments)


def _claim_binding_hash(
    *,
    turn_id: str,
    sentence_index: int,
    claim_text: str,
    snapshot_id: str,
    span_id: str,
    relation: str,
) -> str:
    return _claim_binding_hash_from_digest(
        turn_id=turn_id,
        sentence_index=sentence_index,
        claim_text_sha256=_text_hash(claim_text),
        snapshot_id=snapshot_id,
        span_id=span_id,
        relation=relation,
    )


def _claim_binding_hash_from_digest(
    *,
    turn_id: str,
    sentence_index: int,
    claim_text_sha256: str,
    snapshot_id: str,
    span_id: str,
    relation: str,
) -> str:
    return _domain_hash(
        "claim-binding",
        {
            "turn_id": turn_id,
            "sentence_index": sentence_index,
            "claim_text_sha256": claim_text_sha256,
            "snapshot_id": snapshot_id,
            "span_id": span_id,
            "relation": relation,
        },
    )


def _checker_receipt_hash(values: Mapping[str, Any]) -> str:
    return _domain_hash(
        "deterministic-checker-receipt",
        {
            key: values[key]
            for key in (
                "turn_id",
                "sentence_index",
                "claim_binding_sha256",
                "snapshot_id",
                "span_id",
                "span_sha256",
                "checker_id",
                "checker_version",
                "checker_source_sha256",
                "status",
                "algorithmically_independent",
                "reason_sha256",
                "run_nonce_sha256",
            )
        },
    )


def _claim_record_hash(
    *,
    turn_id: str,
    sentence_index: int,
    claim_text: str,
    snapshot_id: str,
    span_id: str,
    relation: str,
    quote_verified: bool,
    mechanically_verified: bool,
    checker_id: str,
    checker_receipt_sha256: str,
) -> str:
    return _domain_hash(
        "claim",
        {
            "turn_id": turn_id,
            "sentence_index": sentence_index,
            "claim_text": claim_text,
            "snapshot_id": snapshot_id,
            "span_id": span_id,
            "relation": relation,
            "quote_verified": quote_verified,
            "mechanically_verified": mechanically_verified,
            "checker_id": checker_id,
            "checker_receipt_sha256": checker_receipt_sha256,
        },
    )


def _turn_attestation_hash(values: Mapping[str, Any]) -> str:
    return _domain_hash(
        "turn-attestation",
        {
            key: values[key]
            for key in (
                "turn_id",
                "sealed_at",
                "evidence_set_sha256",
                "freshness_required",
                "status",
                "sealed_conflict_ids",
                "attested_at",
                "authority_granted",
            )
        },
    )


def _checker_run_hash(values: Mapping[str, Any]) -> str:
    return _domain_hash(
        "checker-run",
        {
            key: values[key]
            for key in (
                "run_nonce_sha256",
                "turn_id",
                "sentence_index",
                "claim_text_sha256",
                "claim_binding_sha256",
                "snapshot_id",
                "span_id",
                "span_sha256",
                "checker_id",
                "checker_version",
                "checker_source_sha256",
                "issued_at",
                "authority_granted",
            )
        },
    )


def _conflict_observation_hash(values: Mapping[str, Any]) -> str:
    return _domain_hash(
        "conflict-observation",
        {
            key: values[key]
            for key in (
                "turn_id",
                "conflict_id",
                "observed_at",
                "authority_granted",
            )
        },
    )


def untrusted_ephemeral_evidence(text: Any, *, source: Any = "caller") -> Dict[str, Any]:
    """Classify caller-provided text without persisting or granting authority."""

    bounded = _clean_text(text, field="text", limit=MAX_CONTENT_CHARS)
    provider = _clean_text(source, field="source", limit=MAX_PROVIDER_CHARS, required=False)
    return {
        "evidence_class": "untrusted_ephemeral",
        "authority": False,
        "persisted": False,
        "source": provider,
        "text": bounded,
    }


class SQLiteEvidenceLedger:
    """Append-only SQLite WAL ledger for shadow provenance records."""

    _SCHEMA_TABLE = "nexus_evidence_schema"
    _SNAPSHOTS = "nexus_evidence_snapshots"
    _SPANS = "nexus_evidence_spans"
    _TURNS = "nexus_evidence_turns"
    _TURN_ATTESTATIONS = "nexus_evidence_turn_attestations"
    _OPENED = "nexus_evidence_opened"
    _OUTPUTS = "nexus_evidence_outputs"
    _CLAIMS = "nexus_evidence_claims"
    _CHECKER_RUNS = "nexus_evidence_checker_runs"
    _CHECKER_RECEIPTS = "nexus_evidence_checker_receipts"
    _CONFLICTS = "nexus_evidence_conflicts"
    _CONFLICT_OBSERVATIONS = "nexus_evidence_conflict_observations"
    _REVISIONS = "nexus_evidence_revisions"
    _TABLES = (
        _SCHEMA_TABLE,
        _SNAPSHOTS,
        _SPANS,
        _TURNS,
        _TURN_ATTESTATIONS,
        _OPENED,
        _OUTPUTS,
        _CLAIMS,
        _CHECKER_RUNS,
        _CHECKER_RECEIPTS,
        _CONFLICTS,
        _CONFLICT_OBSERVATIONS,
        _REVISIONS,
    )
    _EXPECTED_COLUMNS = {
        _SCHEMA_TABLE: ("version", "applied_at"),
        _SNAPSHOTS: (
            "snapshot_id",
            "schema_version",
            "provider",
            "canonical_uri",
            "fetched_at",
            "published_at",
            "event_time",
            "mention_time",
            "valid_from",
            "valid_until",
            "extractor_version",
            "content",
            "content_sha256",
            "origin",
            "supersedes_snapshot_id",
            "created_at",
        ),
        _SPANS: (
            "snapshot_id",
            "span_id",
            "char_start",
            "char_end",
            "byte_start",
            "byte_end",
            "text",
            "span_sha256",
        ),
        _TURNS: (
            "turn_id",
            "sealed_at",
            "evidence_set_sha256",
            "freshness_required",
            "status",
            "authority_granted",
        ),
        _TURN_ATTESTATIONS: (
            "turn_id",
            "turn_sha256",
            "sealed_conflict_ids_json",
            "attested_at",
            "authority_granted",
        ),
        _OPENED: ("turn_id", "sequence", "snapshot_id", "span_id", "span_sha256"),
        _OUTPUTS: (
            "turn_id",
            "sentence_count",
            "sentence_hashes_json",
            "output_sha256",
            "manifest_sha256",
            "origin",
            "authority_granted",
        ),
        _CLAIMS: (
            "turn_id",
            "sentence_index",
            "claim_text",
            "snapshot_id",
            "span_id",
            "relation",
            "quote_verified",
            "mechanically_verified",
            "checker_id",
            "claim_sha256",
            "authority_granted",
        ),
        _CHECKER_RUNS: (
            "run_nonce_sha256",
            "turn_id",
            "sentence_index",
            "claim_text_sha256",
            "claim_binding_sha256",
            "snapshot_id",
            "span_id",
            "span_sha256",
            "checker_id",
            "checker_version",
            "checker_source_sha256",
            "issued_at",
            "run_sha256",
            "authority_granted",
        ),
        _CHECKER_RECEIPTS: (
            "turn_id",
            "sentence_index",
            "claim_binding_sha256",
            "claim_sha256",
            "snapshot_id",
            "span_id",
            "span_sha256",
            "checker_id",
            "checker_version",
            "checker_source_sha256",
            "status",
            "algorithmically_independent",
            "reason_sha256",
            "run_nonce_sha256",
            "receipt_sha256",
            "authority_granted",
        ),
        _CONFLICTS: (
            "conflict_id",
            "left_snapshot_id",
            "right_snapshot_id",
            "reason",
            "created_at",
        ),
        _CONFLICT_OBSERVATIONS: (
            "turn_id",
            "conflict_id",
            "observed_at",
            "observation_sha256",
            "authority_granted",
        ),
        _REVISIONS: ("snapshot_id", "supersedes_snapshot_id", "revision_kind", "created_at"),
    }
    _EXPECTED_PRIMARY_KEYS = {
        _SCHEMA_TABLE: ("version",),
        _SNAPSHOTS: ("snapshot_id",),
        _SPANS: ("snapshot_id", "span_id"),
        _TURNS: ("turn_id",),
        _TURN_ATTESTATIONS: ("turn_id",),
        _OPENED: ("turn_id", "sequence"),
        _OUTPUTS: ("turn_id",),
        _CLAIMS: ("turn_id", "sentence_index"),
        _CHECKER_RUNS: ("run_nonce_sha256",),
        _CHECKER_RECEIPTS: ("turn_id", "sentence_index"),
        _CONFLICTS: ("conflict_id",),
        _CONFLICT_OBSERVATIONS: ("turn_id", "conflict_id"),
        _REVISIONS: ("snapshot_id",),
    }
    _EXPECTED_FOREIGN_KEYS = {
        _SCHEMA_TABLE: frozenset(),
        _SNAPSHOTS: frozenset(
            {(_SNAPSHOTS, (("supersedes_snapshot_id", "snapshot_id"),))}
        ),
        _SPANS: frozenset({(_SNAPSHOTS, (("snapshot_id", "snapshot_id"),))}),
        _TURNS: frozenset(),
        _TURN_ATTESTATIONS: frozenset({(_TURNS, (("turn_id", "turn_id"),))}),
        _OPENED: frozenset(
            {
                (_TURNS, (("turn_id", "turn_id"),)),
                (
                    _SPANS,
                    (("snapshot_id", "snapshot_id"), ("span_id", "span_id")),
                ),
            }
        ),
        _OUTPUTS: frozenset({(_TURNS, (("turn_id", "turn_id"),))}),
        _CLAIMS: frozenset(
            {
                (_TURNS, (("turn_id", "turn_id"),)),
                (
                    _SPANS,
                    (("snapshot_id", "snapshot_id"), ("span_id", "span_id")),
                ),
            }
        ),
        _CHECKER_RUNS: frozenset(
            {
                (_TURNS, (("turn_id", "turn_id"),)),
                (
                    _SPANS,
                    (("snapshot_id", "snapshot_id"), ("span_id", "span_id")),
                ),
            }
        ),
        _CHECKER_RECEIPTS: frozenset(
            {
                (
                    _CLAIMS,
                    (("turn_id", "turn_id"), ("sentence_index", "sentence_index")),
                ),
                (
                    _SPANS,
                    (("snapshot_id", "snapshot_id"), ("span_id", "span_id")),
                ),
                (_CHECKER_RUNS, (("run_nonce_sha256", "run_nonce_sha256"),)),
            }
        ),
        _CONFLICTS: frozenset(
            {
                (_SNAPSHOTS, (("left_snapshot_id", "snapshot_id"),)),
                (_SNAPSHOTS, (("right_snapshot_id", "snapshot_id"),)),
            }
        ),
        _CONFLICT_OBSERVATIONS: frozenset(
            {
                (_TURNS, (("turn_id", "turn_id"),)),
                (_CONFLICTS, (("conflict_id", "conflict_id"),)),
            }
        ),
        _REVISIONS: frozenset(
            {
                (_SNAPSHOTS, (("snapshot_id", "snapshot_id"),)),
                (_SNAPSHOTS, (("supersedes_snapshot_id", "snapshot_id"),)),
            }
        ),
    }
    _REQUIRED_UNIQUE_KEYS = {
        _OPENED: frozenset({("turn_id", "snapshot_id", "span_id")}),
        _CHECKER_RUNS: frozenset(
            {
                (
                    "claim_binding_sha256",
                    "checker_id",
                    "checker_version",
                    "checker_source_sha256",
                )
            }
        ),
        _CHECKER_RECEIPTS: frozenset({("run_nonce_sha256",)}),
    }
    _REAL_COLUMNS = frozenset(
        {
            "applied_at",
            "fetched_at",
            "published_at",
            "event_time",
            "mention_time",
            "valid_from",
            "valid_until",
            "created_at",
            "sealed_at",
            "attested_at",
            "issued_at",
            "observed_at",
        }
    )
    _INTEGER_COLUMNS = frozenset(
        {
            "char_start",
            "char_end",
            "byte_start",
            "byte_end",
            "sequence",
            "sentence_count",
            "sentence_index",
            "freshness_required",
            "authority_granted",
            "quote_verified",
            "mechanically_verified",
            "algorithmically_independent",
        }
    )
    _NULLABLE_COLUMNS = frozenset(
        {
            (_SNAPSHOTS, "published_at"),
            (_SNAPSHOTS, "event_time"),
            (_SNAPSHOTS, "mention_time"),
            (_SNAPSHOTS, "valid_from"),
            (_SNAPSHOTS, "valid_until"),
            (_SNAPSHOTS, "supersedes_snapshot_id"),
        }
    )
    _REQUIRED_CHECK_FRAGMENTS = {
        _SNAPSHOTS: frozenset({"check(origin='server_fetch')"}),
        _SPANS: frozenset(
            {
                "check(char_start>=0)",
                "check(char_end>char_start)",
                "check(byte_start>=0)",
                "check(byte_end>byte_start)",
            }
        ),
        _TURNS: frozenset(
            {
                "check(freshness_requiredin(0,1))",
                "check(statusin('sealed','conflict_defer'))",
                "check(authority_granted=0)",
            }
        ),
        _TURN_ATTESTATIONS: frozenset({"check(authority_granted=0)"}),
        _OPENED: frozenset({"check(sequence>=0)"}),
        _OUTPUTS: frozenset(
            {
                f"check(sentence_countbetween1and{MAX_OUTPUT_SENTENCES})",
                "check(origin='server_generation')",
                "check(authority_granted=0)",
            }
        ),
        _CLAIMS: frozenset(
            {
                f"check(sentence_indexbetween0and{MAX_OUTPUT_SENTENCES - 1})",
                "check(relationin('quotation','compression','inference'))",
                "check(quote_verifiedin(0,1))",
                "check(mechanically_verifiedin(0,1))",
                "check(authority_granted=0)",
            }
        ),
        _CHECKER_RUNS: frozenset(
            {
                f"check(sentence_indexbetween0and{MAX_OUTPUT_SENTENCES - 1})",
                "check(authority_granted=0)",
            }
        ),
        _CHECKER_RECEIPTS: frozenset(
            {
                f"check(sentence_indexbetween0and{MAX_OUTPUT_SENTENCES - 1})",
                "check(statusin('passed','failed','not_applicable'))",
                "check(algorithmically_independentin(0,1))",
                "check(status!='passed'oralgorithmically_independent=1)",
                "check(authority_granted=0)",
            }
        ),
        _CONFLICTS: frozenset({"check(left_snapshot_id<right_snapshot_id)"}),
        _CONFLICT_OBSERVATIONS: frozenset({"check(authority_granted=0)"}),
        _REVISIONS: frozenset(
            {
                "check(revision_kind='explicit_revision')",
                "check(snapshot_id<>supersedes_snapshot_id)",
            }
        ),
    }

    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_ms: int = 5000,
        clock: Any = time.time,
        deterministic_checkers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> None:
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int) or busy_timeout_ms <= 0:
            raise ValueError("busy_timeout_ms must be a positive integer")
        self.path = Path(path)
        if self.path.exists() and self.path.is_dir():
            raise ValueError(f"evidence ledger path is a directory: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = int(busy_timeout_ms)
        self._clock = clock
        self._capability_token = object()
        self._capability_key = secrets.token_bytes(32)
        self._deterministic_checkers = self._checker_registry(deterministic_checkers)
        with _SCHEMA_INIT_LOCK:
            self._init_schema()

    @staticmethod
    def _checker_registry(
        supplied: Optional[Mapping[str, Mapping[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        if supplied is None:
            return {}
        if not isinstance(supplied, Mapping) or len(supplied) > 64:
            raise ValueError("deterministic_checkers must be a bounded mapping")
        result: Dict[str, Dict[str, Any]] = {}
        for raw_id, raw in supplied.items():
            checker_id = _identifier(raw_id, field="checker_id")
            if checker_id != raw_id or checker_id in result:
                raise ValueError("deterministic checker ids must be unique and canonical")
            if not isinstance(raw, Mapping):
                raise ValueError("deterministic checker descriptors must be mappings")
            version = _clean_text(
                raw.get("version"), field="checker_version", limit=MAX_PROVIDER_CHARS
            )
            if version != raw.get("version"):
                raise ValueError("deterministic checker versions must be canonical")
            source_sha256 = _sha256(
                raw.get("source_sha256"), field="checker_source_sha256"
            )
            check = raw.get("check")
            if not callable(check):
                raise ValueError("deterministic checker descriptors require a callable check")
            result[checker_id] = {
                "version": version,
                "source_sha256": source_sha256,
                "check": check,
            }
        return result

    def _connect(self, *, configure_wal: bool = False) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.path),
            timeout=self.busy_timeout_ms / 1000.0,
            isolation_level=None,
        )
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        conn.execute("PRAGMA foreign_keys=ON")
        if configure_wal:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    @classmethod
    def _validate_schema_contract(cls, conn: sqlite3.Connection) -> None:
        """Reject look-alike tables that omit relational or policy constraints."""

        for table, expected_columns in cls._EXPECTED_COLUMNS.items():
            column_rows = conn.execute(f"PRAGMA table_xinfo({table})").fetchall()
            actual_columns = tuple(str(row[1]) for row in column_rows)
            if actual_columns != expected_columns:
                raise EvidenceLedgerError(
                    f"evidence-ledger table shape mismatch for {table}: "
                    f"expected {expected_columns!r}, got {actual_columns!r}"
                )

            primary_key = tuple(
                str(row[1])
                for row in sorted(
                    (row for row in column_rows if int(row[5]) > 0),
                    key=lambda row: int(row[5]),
                )
            )
            if primary_key != cls._EXPECTED_PRIMARY_KEYS[table]:
                raise EvidenceLedgerError(
                    f"evidence-ledger primary-key mismatch for {table}"
                )

            for row in column_rows:
                column = str(row[1])
                expected_type = (
                    "INTEGER"
                    if column in cls._INTEGER_COLUMNS
                    else "REAL"
                    if column in cls._REAL_COLUMNS
                    else "TEXT"
                )
                if (
                    str(row[2]).upper() != expected_type
                    or row[4] is not None
                    or (len(row) > 6 and int(row[6]) != 0)
                ):
                    raise EvidenceLedgerError(
                        f"evidence-ledger column contract mismatch for {table}.{column}"
                    )
                expected_not_null = (table, column) not in cls._NULLABLE_COLUMNS
                if bool(row[3]) != expected_not_null:
                    raise EvidenceLedgerError(
                        f"evidence-ledger nullability mismatch for {table}.{column}"
                    )

            foreign_key_rows = conn.execute(
                f"PRAGMA foreign_key_list({table})"
            ).fetchall()
            foreign_key_groups: Dict[int, Dict[str, Any]] = {}
            for row in foreign_key_rows:
                group = foreign_key_groups.setdefault(
                    int(row[0]), {"table": str(row[2]), "columns": []}
                )
                if (
                    group["table"] != str(row[2])
                    or str(row[5]).upper() != "NO ACTION"
                    or str(row[6]).upper() != "NO ACTION"
                    or str(row[7]).upper() != "NONE"
                ):
                    raise EvidenceLedgerError(
                        f"evidence-ledger foreign-key policy mismatch for {table}"
                    )
                group["columns"].append(
                    (int(row[1]), str(row[3]), str(row[4]))
                )
            actual_foreign_keys = frozenset(
                (
                    str(group["table"]),
                    tuple(
                        (source, target)
                        for _sequence, source, target in sorted(group["columns"])
                    ),
                )
                for group in foreign_key_groups.values()
            )
            if actual_foreign_keys != cls._EXPECTED_FOREIGN_KEYS[table]:
                raise EvidenceLedgerError(
                    f"evidence-ledger foreign-key mismatch for {table}"
                )

            actual_unique_keys: set[Tuple[str, ...]] = set()
            for index_row in conn.execute(f"PRAGMA index_list({table})").fetchall():
                if int(index_row[2]) != 1:
                    continue
                if int(index_row[4]) != 0:
                    raise EvidenceLedgerError(
                        f"evidence-ledger partial unique-key mismatch for {table}"
                    )
                index_name = str(index_row[1]).replace("'", "''")
                key_rows = [
                    row
                    for row in sorted(
                        conn.execute(
                            f"PRAGMA index_xinfo('{index_name}')"
                        ).fetchall(),
                        key=lambda row: int(row[0]),
                    )
                    if int(row[5]) == 1
                ]
                if any(
                    row[2] is None
                    or int(row[3]) != 0
                    or str(row[4]).upper() != "BINARY"
                    for row in key_rows
                ):
                    raise EvidenceLedgerError(
                        f"evidence-ledger unique-key collation mismatch for {table}"
                    )
                index_columns = tuple(str(row[2]) for row in key_rows)
                if index_columns:
                    actual_unique_keys.add(index_columns)
            expected_unique_keys = set(
                cls._REQUIRED_UNIQUE_KEYS.get(table, frozenset())
            )
            expected_unique_keys.add(cls._EXPECTED_PRIMARY_KEYS[table])
            if actual_unique_keys != expected_unique_keys:
                raise EvidenceLedgerError(
                    f"evidence-ledger unique-key mismatch for {table}"
                )

            sql_row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
            actual_checks = _sqlite_check_fragments(
                "" if sql_row is None else sql_row[0]
            )
            expected_checks = cls._REQUIRED_CHECK_FRAGMENTS.get(table, frozenset())
            if actual_checks != expected_checks:
                raise EvidenceLedgerError(
                    f"evidence-ledger CHECK-constraint mismatch for {table}"
                )

    def _validate_append_only_contract(self, conn: sqlite3.Connection) -> None:
        """Require every update/delete guard to have the exact fail-closed body."""

        trigger_sql = {
            str(row[0]): re.sub(r"\s+", " ", str(row[1] or "")).lower()
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        for table in self._TABLES:
            for operation in ("update", "delete"):
                name = f"{table}_append_only_{operation}"
                expected_sql = (
                    f"create trigger {name} before {operation} on {table} "
                    "begin select raise(abort, 'evidence ledger is append-only'); end"
                )
                if trigger_sql.get(name, "") != expected_sql:
                    raise EvidenceLedgerError(
                        f"append-only trigger contract mismatch for {table} {operation}"
                    )

    def _clock_now(self, *, field: str = "clock") -> float:
        value = _timestamp(self._clock(), field=field)
        if value is None:
            raise EvidenceLedgerValidationError(f"{field} must be a finite timestamp")
        return value

    def _init_schema(self) -> None:
        conn = self._connect(configure_wal=True)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._SCHEMA_TABLE} (
                    version TEXT PRIMARY KEY NOT NULL,
                    applied_at REAL NOT NULL
                )
                """
            )
            existing_versions = {
                str(row[0])
                for row in conn.execute(f"SELECT version FROM {self._SCHEMA_TABLE}").fetchall()
            }
            supported_versions = LEGACY_LEDGER_SCHEMA_VERSIONS | {LEDGER_SCHEMA_VERSION}
            unsupported = existing_versions - supported_versions
            if unsupported:
                raise EvidenceLedgerError(
                    "unsupported evidence-ledger schema version: "
                    + ", ".join(sorted(unsupported))
                )
            if (
                existing_versions & LEGACY_LEDGER_SCHEMA_VERSIONS
                and LEDGER_SCHEMA_VERSION not in existing_versions
            ):
                raise EvidenceLedgerError(
                    "legacy v1 evidence ledger requires an explicit reviewed migration"
                )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._SNAPSHOTS} (
                    snapshot_id TEXT PRIMARY KEY NOT NULL,
                    schema_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    canonical_uri TEXT NOT NULL,
                    fetched_at REAL NOT NULL,
                    published_at REAL,
                    event_time REAL,
                    mention_time REAL,
                    valid_from REAL,
                    valid_until REAL,
                    extractor_version TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin = 'server_fetch'),
                    supersedes_snapshot_id TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(supersedes_snapshot_id)
                        REFERENCES {self._SNAPSHOTS}(snapshot_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._SPANS} (
                    snapshot_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    byte_start INTEGER NOT NULL,
                    byte_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    span_sha256 TEXT NOT NULL,
                    CHECK(char_start >= 0),
                    CHECK(char_end > char_start),
                    CHECK(byte_start >= 0),
                    CHECK(byte_end > byte_start),
                    PRIMARY KEY(snapshot_id, span_id),
                    FOREIGN KEY(snapshot_id) REFERENCES {self._SNAPSHOTS}(snapshot_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TURNS} (
                    turn_id TEXT PRIMARY KEY NOT NULL,
                    sealed_at REAL NOT NULL,
                    evidence_set_sha256 TEXT NOT NULL,
                    freshness_required INTEGER NOT NULL CHECK(freshness_required IN (0, 1)),
                    status TEXT NOT NULL CHECK(status IN ('sealed', 'conflict_defer')),
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._TURN_ATTESTATIONS} (
                    turn_id TEXT PRIMARY KEY NOT NULL,
                    turn_sha256 TEXT NOT NULL,
                    sealed_conflict_ids_json TEXT NOT NULL,
                    attested_at REAL NOT NULL,
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._OPENED} (
                    turn_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL CHECK(sequence >= 0),
                    snapshot_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    span_sha256 TEXT NOT NULL,
                    PRIMARY KEY(turn_id, sequence),
                    UNIQUE(turn_id, snapshot_id, span_id),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id),
                    FOREIGN KEY(snapshot_id, span_id)
                        REFERENCES {self._SPANS}(snapshot_id, span_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._OUTPUTS} (
                    turn_id TEXT PRIMARY KEY NOT NULL,
                    sentence_count INTEGER NOT NULL
                        CHECK(sentence_count BETWEEN 1 AND 4096),
                    sentence_hashes_json TEXT NOT NULL,
                    output_sha256 TEXT NOT NULL,
                    manifest_sha256 TEXT NOT NULL,
                    origin TEXT NOT NULL CHECK(origin = 'server_generation'),
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._CLAIMS} (
                    turn_id TEXT NOT NULL,
                    sentence_index INTEGER NOT NULL
                        CHECK(sentence_index BETWEEN 0 AND 4095),
                    claim_text TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    relation TEXT NOT NULL
                        CHECK(relation IN ('quotation', 'compression', 'inference')),
                    quote_verified INTEGER NOT NULL CHECK(quote_verified IN (0, 1)),
                    mechanically_verified INTEGER NOT NULL
                        CHECK(mechanically_verified IN (0, 1)),
                    checker_id TEXT NOT NULL,
                    claim_sha256 TEXT NOT NULL,
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    PRIMARY KEY(turn_id, sentence_index),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id),
                    FOREIGN KEY(snapshot_id, span_id)
                        REFERENCES {self._SPANS}(snapshot_id, span_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._CHECKER_RUNS} (
                    run_nonce_sha256 TEXT PRIMARY KEY NOT NULL,
                    turn_id TEXT NOT NULL,
                    sentence_index INTEGER NOT NULL
                        CHECK(sentence_index BETWEEN 0 AND 4095),
                    claim_text_sha256 TEXT NOT NULL,
                    claim_binding_sha256 TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    span_sha256 TEXT NOT NULL,
                    checker_id TEXT NOT NULL,
                    checker_version TEXT NOT NULL,
                    checker_source_sha256 TEXT NOT NULL,
                    issued_at REAL NOT NULL,
                    run_sha256 TEXT NOT NULL,
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    UNIQUE(
                        claim_binding_sha256, checker_id, checker_version,
                        checker_source_sha256
                    ),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id),
                    FOREIGN KEY(snapshot_id, span_id)
                        REFERENCES {self._SPANS}(snapshot_id, span_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._CHECKER_RECEIPTS} (
                    turn_id TEXT NOT NULL,
                    sentence_index INTEGER NOT NULL
                        CHECK(sentence_index BETWEEN 0 AND 4095),
                    claim_binding_sha256 TEXT NOT NULL,
                    claim_sha256 TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    span_sha256 TEXT NOT NULL,
                    checker_id TEXT NOT NULL,
                    checker_version TEXT NOT NULL,
                    checker_source_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK(status IN ('passed', 'failed', 'not_applicable')),
                    algorithmically_independent INTEGER NOT NULL
                        CHECK(algorithmically_independent IN (0, 1)),
                    reason_sha256 TEXT NOT NULL,
                    run_nonce_sha256 TEXT NOT NULL UNIQUE,
                    receipt_sha256 TEXT NOT NULL,
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    CHECK(status != 'passed' OR algorithmically_independent = 1),
                    PRIMARY KEY(turn_id, sentence_index),
                    FOREIGN KEY(turn_id, sentence_index)
                        REFERENCES {self._CLAIMS}(turn_id, sentence_index),
                    FOREIGN KEY(snapshot_id, span_id)
                        REFERENCES {self._SPANS}(snapshot_id, span_id),
                    FOREIGN KEY(run_nonce_sha256)
                        REFERENCES {self._CHECKER_RUNS}(run_nonce_sha256)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._CONFLICTS} (
                    conflict_id TEXT PRIMARY KEY NOT NULL,
                    left_snapshot_id TEXT NOT NULL,
                    right_snapshot_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    CHECK(left_snapshot_id < right_snapshot_id),
                    FOREIGN KEY(left_snapshot_id) REFERENCES {self._SNAPSHOTS}(snapshot_id),
                    FOREIGN KEY(right_snapshot_id) REFERENCES {self._SNAPSHOTS}(snapshot_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._CONFLICT_OBSERVATIONS} (
                    turn_id TEXT NOT NULL,
                    conflict_id TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    observation_sha256 TEXT NOT NULL,
                    authority_granted INTEGER NOT NULL CHECK(authority_granted = 0),
                    PRIMARY KEY(turn_id, conflict_id),
                    FOREIGN KEY(turn_id) REFERENCES {self._TURNS}(turn_id),
                    FOREIGN KEY(conflict_id) REFERENCES {self._CONFLICTS}(conflict_id)
                )
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._REVISIONS} (
                    snapshot_id TEXT PRIMARY KEY NOT NULL,
                    supersedes_snapshot_id TEXT NOT NULL,
                    revision_kind TEXT NOT NULL
                        CHECK(revision_kind = 'explicit_revision'),
                    created_at REAL NOT NULL,
                    CHECK(snapshot_id <> supersedes_snapshot_id),
                    FOREIGN KEY(snapshot_id) REFERENCES {self._SNAPSHOTS}(snapshot_id),
                    FOREIGN KEY(supersedes_snapshot_id) REFERENCES {self._SNAPSHOTS}(snapshot_id)
                )
                """
            )
            self._validate_schema_contract(conn)
            for table in self._TABLES:
                update_trigger = f"{table}_append_only_update"
                delete_trigger = f"{table}_append_only_delete"
                conn.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {update_trigger}
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'evidence ledger is append-only');
                    END
                    """
                )
                conn.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {delete_trigger}
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, 'evidence ledger is append-only');
                    END
                    """
                )
            self._validate_append_only_contract(conn)
            conn.execute(
                f"INSERT OR IGNORE INTO {self._SCHEMA_TABLE}(version, applied_at) VALUES (?, ?)",
                (LEDGER_SCHEMA_VERSION, self._clock_now()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _span_rows(content: str, spans: Optional[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
        if spans is None:
            supplied: List[Any] = []
        else:
            try:
                supplied = list(itertools.islice(iter(spans), MAX_SPANS_PER_SNAPSHOT + 1))
            except TypeError as exc:
                raise EvidenceLedgerValidationError("snapshot spans must be iterable") from exc
        if len(supplied) > MAX_SPANS_PER_SNAPSHOT:
            raise EvidenceLedgerValidationError("too many snapshot spans")
        if not supplied:
            supplied = [{"span_id": "whole", "start": 0, "end": len(content)}]
        result: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in supplied:
            if not isinstance(row, Mapping):
                raise EvidenceLedgerValidationError("snapshot spans must be mappings")
            span_id = _identifier(row.get("span_id"), field="span_id")
            if span_id in seen:
                raise EvidenceLedgerValidationError("snapshot span ids must be unique")
            seen.add(span_id)
            start = row.get("start", row.get("char_start"))
            end = row.get("end", row.get("char_end"))
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                raise EvidenceLedgerValidationError("snapshot span offsets must be integers")
            if start < 0 or end <= start or end > len(content):
                raise EvidenceLedgerValidationError("snapshot span offsets are out of bounds")
            text = content[start:end]
            supplied_text = row.get("text")
            if supplied_text is not None and supplied_text != text:
                raise EvidenceLedgerValidationError("snapshot span text does not match source bytes")
            byte_start = len(content[:start].encode("utf-8"))
            byte_end = len(content[:end].encode("utf-8"))
            result.append(
                {
                    "span_id": span_id,
                    "char_start": start,
                    "char_end": end,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "text": text,
                    "span_sha256": _domain_hash(
                        "span",
                        {
                            "span_id": span_id,
                            "char_start": start,
                            "char_end": end,
                            "byte_start": byte_start,
                            "byte_end": byte_end,
                            "text_sha256": _text_hash(text),
                        },
                    ),
                }
            )
        return result

    def _snapshot_row(self, conn: sqlite3.Connection, snapshot_id: str) -> Optional[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"SELECT * FROM {self._SNAPSHOTS} WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()

    def _span_row(
        self, conn: sqlite3.Connection, snapshot_id: str, span_id: str
    ) -> Optional[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"SELECT * FROM {self._SPANS} WHERE snapshot_id = ? AND span_id = ?",
            (snapshot_id, span_id),
        ).fetchone()

    def _validated_snapshot_row(
        self, conn: sqlite3.Connection, snapshot_id: str
    ) -> Optional[sqlite3.Row]:
        """Read and integrity-check a snapshot plus every persisted span."""

        row = self._snapshot_row(conn, snapshot_id)
        if row is None:
            return None
        try:
            if row["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
                raise EvidenceLedgerError("snapshot schema version integrity check failed")
            if row["origin"] != "server_fetch":
                raise EvidenceLedgerError("snapshot origin integrity check failed")
            provider = _clean_text(row["provider"], field="provider", limit=MAX_PROVIDER_CHARS)
            if provider != row["provider"]:
                raise EvidenceLedgerError("snapshot provider integrity check failed")
            uri = canonical_uri(row["canonical_uri"])
            if uri != row["canonical_uri"]:
                raise EvidenceLedgerError("snapshot URI integrity check failed")
            content = _source_content(row["content"])
            content_sha = _text_hash(content)
            if content_sha != row["content_sha256"]:
                raise EvidenceLedgerError("snapshot content integrity check failed")
            fetched = _timestamp(row["fetched_at"], field="fetched_at")
            published = _timestamp(row["published_at"], field="published_at", optional=True)
            event = _timestamp(row["event_time"], field="event_time", optional=True)
            mention = _timestamp(row["mention_time"], field="mention_time", optional=True)
            valid_start = _timestamp(row["valid_from"], field="valid_from", optional=True)
            valid_end = _timestamp(row["valid_until"], field="valid_until", optional=True)
            if valid_start is not None and valid_end is not None and valid_end <= valid_start:
                raise EvidenceLedgerError("snapshot validity-window integrity check failed")
            extractor = _clean_text(
                row["extractor_version"], field="extractor_version", limit=MAX_EXTRACTOR_CHARS
            )
            if extractor != row["extractor_version"]:
                raise EvidenceLedgerError("snapshot extractor integrity check failed")
            supersedes = row["supersedes_snapshot_id"]
            if supersedes is not None:
                supersedes = _identifier(
                    supersedes, field="supersedes_snapshot_id", prefix="snap-"
                )
                if supersedes != row["supersedes_snapshot_id"]:
                    raise EvidenceLedgerError("snapshot revision integrity check failed")
            elif row["supersedes_snapshot_id"] is not None:
                raise EvidenceLedgerError("snapshot revision integrity check failed")
            expected_id = _snapshot_identity(
                provider=provider,
                canonical_uri_value=uri,
                fetched_at=fetched,
                published_at=published,
                event_time=event,
                mention_time=mention,
                valid_from=valid_start,
                valid_until=valid_end,
                extractor_version=extractor,
                content_sha256=content_sha,
                supersedes_snapshot_id=supersedes,
            )
            if row["snapshot_id"] != expected_id or snapshot_id != expected_id:
                raise EvidenceLedgerError("snapshot identity integrity check failed")
            _timestamp(row["created_at"], field="created_at")
            spans = conn.execute(
                f"""
                SELECT snapshot_id, span_id, char_start, char_end, byte_start, byte_end,
                       text, span_sha256
                FROM {self._SPANS} WHERE snapshot_id = ? ORDER BY span_id
                """,
                (snapshot_id,),
            ).fetchall()
            if not spans or len(spans) > MAX_SPANS_PER_SNAPSHOT:
                raise EvidenceLedgerError("snapshot span cardinality integrity check failed")
            for span in spans:
                span_id = _identifier(span["span_id"], field="span_id")
                if span_id != span["span_id"] or span["snapshot_id"] != snapshot_id:
                    raise EvidenceLedgerError("snapshot span identity integrity check failed")
                start = span["char_start"]
                end = span["char_end"]
                byte_start = span["byte_start"]
                byte_end = span["byte_end"]
                if (
                    isinstance(start, bool)
                    or isinstance(end, bool)
                    or isinstance(byte_start, bool)
                    or isinstance(byte_end, bool)
                    or not all(isinstance(value, int) for value in (start, end, byte_start, byte_end))
                    or start < 0
                    or end <= start
                    or end > len(content)
                ):
                    raise EvidenceLedgerError("snapshot span bounds integrity check failed")
                expected_text = content[start:end]
                if span["text"] != expected_text:
                    raise EvidenceLedgerError("snapshot span text integrity check failed")
                expected_byte_start = len(content[:start].encode("utf-8"))
                expected_byte_end = len(content[:end].encode("utf-8"))
                if (byte_start, byte_end) != (expected_byte_start, expected_byte_end):
                    raise EvidenceLedgerError("snapshot span byte-offset integrity check failed")
                expected_span_sha = _domain_hash(
                    "span",
                    {
                        "span_id": span_id,
                        "char_start": start,
                        "char_end": end,
                        "byte_start": byte_start,
                        "byte_end": byte_end,
                        "text_sha256": _text_hash(expected_text),
                    },
                )
                if span["span_sha256"] != expected_span_sha:
                    raise EvidenceLedgerError("snapshot span hash integrity check failed")
        except EvidenceLedgerError:
            raise
        except (UnicodeError, OverflowError, TypeError, ValueError, KeyError) as exc:
            raise EvidenceLedgerError("snapshot integrity check failed") from exc
        return row

    def _issue_server_fetch_capability(
        self,
        *,
        provider: str,
        uri: str,
        content: str,
        fetched_at: float,
        published_at: Optional[float] = None,
        event_time: Optional[float] = None,
        mention_time: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        extractor_version: str = "text-extractor-v1",
        supersedes_snapshot_id: Optional[str] = None,
        spans: Optional[Iterable[Mapping[str, Any]]] = None,
    ) -> object:
        """Mint an in-process receipt for a trusted server-fetch adapter.

        This private hook is for the server/tool adapter, not for request
        payloads.  It is ledger-instance-bound and is not cryptographic
        authentication; deployments should keep the adapter in the trusted
        process boundary and pass the resulting receipt to ``record_snapshot``.
        """

        provider_text = _clean_text(provider, field="provider", limit=MAX_PROVIDER_CHARS)
        uri_text = canonical_uri(uri)
        content_text = _source_content(content)
        fetched = _timestamp(fetched_at, field="fetched_at")
        published = _timestamp(published_at, field="published_at", optional=True)
        event = _timestamp(event_time, field="event_time", optional=True)
        mention = _timestamp(mention_time, field="mention_time", optional=True)
        valid_start = _timestamp(valid_from, field="valid_from", optional=True)
        valid_end = _timestamp(valid_until, field="valid_until", optional=True)
        if valid_start is not None and valid_end is not None and valid_end <= valid_start:
            raise EvidenceLedgerValidationError("valid_until must be after valid_from")
        span_rows = self._span_rows(content_text, spans)
        extractor = _clean_text(
            extractor_version,
            field="extractor_version",
            limit=MAX_EXTRACTOR_CHARS,
        )
        supersedes = None
        if supersedes_snapshot_id is not None:
            supersedes = _identifier(
                supersedes_snapshot_id,
                field="supersedes_snapshot_id",
                prefix="snap-",
            )
        assert fetched is not None
        capability_values = {
            "provider": provider_text,
            "canonical_uri": uri_text,
            "fetched_at": fetched,
            "content_sha256": _text_hash(content_text),
            "metadata_sha256": _fetch_metadata_hash(
                published_at=published,
                event_time=event,
                mention_time=mention,
                valid_from=valid_start,
                valid_until=valid_end,
                extractor_version=extractor,
                supersedes_snapshot_id=supersedes,
            ),
            "span_manifest_sha256": _span_manifest_hash(span_rows),
        }
        return _ServerFetchCapability(
            self._capability_token,
            **capability_values,
            capability_mac=_capability_mac(
                self._capability_key, "server-fetch", capability_values
            ),
        )

    def record_snapshot(
        self,
        *,
        provider: str,
        uri: str,
        content: str,
        fetched_at: float,
        published_at: Optional[float] = None,
        event_time: Optional[float] = None,
        mention_time: Optional[float] = None,
        valid_from: Optional[float] = None,
        valid_until: Optional[float] = None,
        extractor_version: str = "text-extractor-v1",
        spans: Optional[Iterable[Mapping[str, Any]]] = None,
        origin: str = "server_fetch",
        supersedes_snapshot_id: Optional[str] = None,
        fetch_capability: Any = None,
    ) -> Dict[str, Any]:
        """Append one immutable server-produced source snapshot."""

        provider_text = _clean_text(provider, field="provider", limit=MAX_PROVIDER_CHARS)
        uri_text = canonical_uri(uri)
        content_text = _source_content(content)
        span_rows = self._span_rows(content_text, spans)
        fetched = _timestamp(fetched_at, field="fetched_at")
        published = _timestamp(published_at, field="published_at", optional=True)
        event = _timestamp(event_time, field="event_time", optional=True)
        mention = _timestamp(mention_time, field="mention_time", optional=True)
        valid_start = _timestamp(valid_from, field="valid_from", optional=True)
        valid_end = _timestamp(valid_until, field="valid_until", optional=True)
        if valid_start is not None and valid_end is not None and valid_end <= valid_start:
            raise EvidenceLedgerValidationError("valid_until must be after valid_from")
        extractor = _clean_text(
            extractor_version,
            field="extractor_version",
            limit=MAX_EXTRACTOR_CHARS,
        )
        if origin != "server_fetch":
            raise EvidenceLedgerValidationError(
                "caller-supplied evidence is untrusted_ephemeral and cannot be persisted"
            )
        if type(fetch_capability) is not _ServerFetchCapability or (
            fetch_capability._ledger_token is not self._capability_token
        ):
            raise EvidenceLedgerValidationError(
                "record_snapshot requires a server-produced fetch capability"
            )
        supersedes = None
        if supersedes_snapshot_id is not None:
            supersedes = _identifier(
                supersedes_snapshot_id,
                field="supersedes_snapshot_id",
                prefix="snap-",
            )
        content_sha = _text_hash(content_text)
        capability_values = {
            "provider": fetch_capability.provider,
            "canonical_uri": fetch_capability.canonical_uri,
            "fetched_at": fetch_capability.fetched_at,
            "content_sha256": fetch_capability.content_sha256,
            "metadata_sha256": fetch_capability.metadata_sha256,
            "span_manifest_sha256": fetch_capability.span_manifest_sha256,
        }
        if (
            fetch_capability.provider != provider_text
            or fetch_capability.canonical_uri != uri_text
            or fetch_capability.fetched_at != fetched
            or fetch_capability.content_sha256 != content_sha
            or fetch_capability.metadata_sha256
            != _fetch_metadata_hash(
                published_at=published,
                event_time=event,
                mention_time=mention,
                valid_from=valid_start,
                valid_until=valid_end,
                extractor_version=extractor,
                supersedes_snapshot_id=supersedes,
            )
            or fetch_capability.span_manifest_sha256 != _span_manifest_hash(span_rows)
            or not isinstance(fetch_capability._capability_mac, str)
            or _SHA256_RE.fullmatch(fetch_capability._capability_mac) is None
            or not hmac.compare_digest(
                fetch_capability._capability_mac,
                _capability_mac(self._capability_key, "server-fetch", capability_values),
            )
        ):
            raise EvidenceLedgerValidationError(
                "server fetch capability does not match snapshot"
            )
        snapshot_id = _snapshot_identity(
            provider=provider_text,
            canonical_uri_value=uri_text,
            fetched_at=fetched,
            published_at=published,
            event_time=event,
            mention_time=mention,
            valid_from=valid_start,
            valid_until=valid_end,
            extractor_version=extractor,
            content_sha256=content_sha,
            supersedes_snapshot_id=supersedes,
        )
        created = self._clock_now(field="created_at")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if supersedes is not None:
                if self._validated_snapshot_row(conn, supersedes) is None:
                    raise EvidenceLedgerValidationError("superseded snapshot does not exist")
            existing = self._snapshot_row(conn, snapshot_id)
            if existing is None:
                conn.execute(
                    f"""
                    INSERT INTO {self._SNAPSHOTS}(
                        snapshot_id, schema_version, provider, canonical_uri,
                        fetched_at, published_at, event_time, mention_time,
                        valid_from, valid_until, extractor_version, content,
                        content_sha256, origin, supersedes_snapshot_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        SNAPSHOT_SCHEMA_VERSION,
                        provider_text,
                        uri_text,
                        fetched,
                        published,
                        event,
                        mention,
                        valid_start,
                        valid_end,
                        extractor,
                        content_text,
                        content_sha,
                        origin,
                        supersedes,
                        created,
                    ),
                )
                for span in span_rows:
                    conn.execute(
                        f"""
                        INSERT INTO {self._SPANS}(
                            snapshot_id, span_id, char_start, char_end, byte_start,
                            byte_end, text, span_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            span["span_id"],
                            span["char_start"],
                            span["char_end"],
                            span["byte_start"],
                            span["byte_end"],
                            span["text"],
                            span["span_sha256"],
                        ),
                    )
                if supersedes is not None:
                    conn.execute(
                        f"""
                        INSERT INTO {self._REVISIONS}(
                            snapshot_id, supersedes_snapshot_id, revision_kind, created_at
                        ) VALUES (?, ?, 'explicit_revision', ?)
                        """,
                        (snapshot_id, supersedes, created),
                    )
            else:
                self._validated_snapshot_row(conn, snapshot_id)
                if (
                    existing["content_sha256"] != content_sha
                    or existing["canonical_uri"] != uri_text
                    or existing["provider"] != provider_text
                ):
                    raise EvidenceLedgerValidationError("snapshot id collision with different content")
                current_spans = conn.execute(
                    f"SELECT span_id, span_sha256 FROM {self._SPANS} WHERE snapshot_id = ? ORDER BY span_id",
                    (snapshot_id,),
                ).fetchall()
                expected_spans = sorted((row["span_id"], row["span_sha256"]) for row in span_rows)
                if [(row[0], row[1]) for row in current_spans] != expected_spans:
                    raise EvidenceLedgerValidationError("immutable snapshot spans do not match")
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return self.get_snapshot(snapshot_id) or {}

    def get_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        sid = _identifier(snapshot_id, field="snapshot_id", prefix="snap-")
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            row = self._validated_snapshot_row(conn, sid)
            if row is None:
                conn.commit()
                return None
            spans = conn.execute(
                f"""
                SELECT span_id, char_start, char_end, byte_start, byte_end, text, span_sha256
                FROM {self._SPANS} WHERE snapshot_id = ? ORDER BY span_id
                """,
                (sid,),
            ).fetchall()
            result = {
                "snapshot_id": row["snapshot_id"],
                "schema_version": row["schema_version"],
                "provider": row["provider"],
                "canonical_uri": row["canonical_uri"],
                "fetched_at": row["fetched_at"],
                "published_at": row["published_at"],
                "event_time": row["event_time"],
                "mention_time": row["mention_time"],
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "extractor_version": row["extractor_version"],
                "content": row["content"],
                "content_sha256": row["content_sha256"],
                "origin": row["origin"],
                "supersedes_snapshot_id": row["supersedes_snapshot_id"],
                "spans": [dict(span) for span in spans],
                "authority_granted": False,
            }
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _freshness(self, row: sqlite3.Row, now: float) -> str:
        valid_from = row["valid_from"]
        valid_until = row["valid_until"]
        if valid_from is not None and now < float(valid_from):
            return "not_yet_valid"
        if valid_until is not None and now >= float(valid_until):
            return "expired"
        if valid_from is None or valid_until is None:
            return "unknown"
        return "current"

    def snapshot_freshness(self, snapshot_id: str, *, now: Optional[float] = None) -> Dict[str, Any]:
        sid = _identifier(snapshot_id, field="snapshot_id", prefix="snap-")
        moment = self._clock_now(field="now") if now is None else _timestamp(now, field="now")
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            row = self._validated_snapshot_row(conn, sid)
            if row is None:
                raise EvidenceLedgerValidationError("snapshot does not exist")
            status = self._freshness(row, moment)
            result = {
                "snapshot_id": sid,
                "status": status,
                "freshness_known": status in {"current", "expired", "not_yet_valid"},
                "valid_from": row["valid_from"],
                "valid_until": row["valid_until"],
                "checked_at": moment,
                "authority_granted": False,
            }
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _opened_items(evidence: Iterable[Any]) -> List[Tuple[str, str]]:
        try:
            rows = list(itertools.islice(iter(evidence), MAX_OPENED_SPANS_PER_TURN + 1))
        except TypeError as exc:
            raise EvidenceLedgerValidationError("opened evidence must be iterable") from exc
        if not rows or len(rows) > MAX_OPENED_SPANS_PER_TURN:
            raise EvidenceLedgerValidationError("a turn needs 1-128 opened evidence spans")
        result: List[Tuple[str, str]] = []
        for row in rows:
            if isinstance(row, Mapping):
                snapshot_id = row.get("snapshot_id")
                span_id = row.get("span_id")
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)) and len(row) == 2:
                snapshot_id, span_id = row
            else:
                raise EvidenceLedgerValidationError("opened evidence must identify snapshot and span")
            item = (
                _identifier(snapshot_id, field="snapshot_id", prefix="snap-"),
                _identifier(span_id, field="span_id"),
            )
            if item in result:
                raise EvidenceLedgerValidationError("opened evidence spans must be unique")
            result.append(item)
        return result

    def _turn_row(self, conn: sqlite3.Connection, turn_id: str) -> Optional[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"SELECT * FROM {self._TURNS} WHERE turn_id = ?", (turn_id,)
        ).fetchone()

    def seal_turn(
        self,
        turn_id: str,
        evidence: Iterable[Any],
        *,
        freshness_required: bool = False,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Seal the exact ordered evidence spans opened for one generation turn."""

        tid = _identifier(turn_id, field="turn_id")
        if type(freshness_required) is not bool:
            raise EvidenceLedgerValidationError("freshness_required must be boolean")
        items = self._opened_items(evidence)
        moment = self._clock_now(field="now") if now is None else _timestamp(now, field="now")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = self._turn_row(conn, tid)
            rows: List[Dict[str, Any]] = []
            snapshot_ids: set[str] = set()
            for snapshot_id, span_id in items:
                snapshot = self._validated_snapshot_row(conn, snapshot_id)
                if snapshot is None:
                    raise EvidenceLedgerValidationError("opened evidence snapshot does not exist")
                span = self._span_row(conn, snapshot_id, span_id)
                if span is None:
                    raise EvidenceLedgerValidationError("opened evidence span does not exist")
                if span["span_sha256"] != _domain_hash(
                    "span",
                    {
                        "span_id": span["span_id"],
                        "char_start": span["char_start"],
                        "char_end": span["char_end"],
                        "byte_start": span["byte_start"],
                        "byte_end": span["byte_end"],
                        "text_sha256": _text_hash(span["text"]),
                    },
                ):
                    raise EvidenceLedgerError("opened evidence span integrity check failed")
                snapshot_ids.add(snapshot_id)
                rows.append(
                    {
                        "sequence": len(rows),
                        "snapshot_id": snapshot_id,
                        "span_id": span_id,
                        "span_sha256": span["span_sha256"],
                    }
                )
                if existing is None and freshness_required and self._freshness(snapshot, moment) != "current":
                    raise EvidenceLedgerFreshnessError(
                        f"freshness_required evidence is not current: {snapshot_id}"
                    )
            evidence_hash = _domain_hash("opened-evidence", rows)
            conflicts = self._conflict_pairs(conn, snapshot_ids)
            if existing is not None:
                self._validated_turn_state(conn, tid)
                existing_rows = conn.execute(
                    f"""
                    SELECT sequence, snapshot_id, span_id, span_sha256
                    FROM {self._OPENED} WHERE turn_id = ? ORDER BY sequence
                    """,
                    (tid,),
                ).fetchall()
                if (
                    existing["evidence_set_sha256"] != evidence_hash
                    or bool(existing["freshness_required"]) != bool(freshness_required)
                    or [
                        (row["sequence"], row["snapshot_id"], row["span_id"], row["span_sha256"])
                        for row in existing_rows
                    ] != [
                        (row["sequence"], row["snapshot_id"], row["span_id"], row["span_sha256"])
                        for row in rows
                    ]
                ):
                    raise EvidenceLedgerValidationError("a sealed turn cannot be reopened with different evidence")
                conn.commit()
                return {
                    "turn_id": tid,
                    "status": existing["status"],
                    "sealed": True,
                    "evidence_set_sha256": existing["evidence_set_sha256"],
                    "opened_spans": rows,
                    "conflicts": conflicts,
                    "freshness_required": bool(existing["freshness_required"]),
                    "authority_granted": False,
                }
            else:
                status = "conflict_defer" if conflicts else "sealed"
                conn.execute(
                    f"""
                    INSERT INTO {self._TURNS}(
                        turn_id, sealed_at, evidence_set_sha256, freshness_required,
                        status, authority_granted
                    ) VALUES (?, ?, ?, ?, ?, 0)
                    """,
                    (tid, moment, evidence_hash, int(bool(freshness_required)), status),
                )
                for row in rows:
                    conn.execute(
                        f"""
                        INSERT INTO {self._OPENED}(
                            turn_id, sequence, snapshot_id, span_id, span_sha256
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (tid, row["sequence"], row["snapshot_id"], row["span_id"], row["span_sha256"]),
                    )
                conflict_ids = tuple(sorted(str(c["conflict_id"]) for c in conflicts))
                attestation = {
                    "turn_id": tid,
                    "sealed_at": moment,
                    "evidence_set_sha256": evidence_hash,
                    "freshness_required": bool(freshness_required),
                    "status": status,
                    "sealed_conflict_ids": conflict_ids,
                    "attested_at": moment,
                    "authority_granted": False,
                }
                conn.execute(
                    f"""
                    INSERT INTO {self._TURN_ATTESTATIONS}(
                        turn_id, turn_sha256, sealed_conflict_ids_json, attested_at, authority_granted
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (tid, _turn_attestation_hash(attestation), json.dumps(list(conflict_ids)), moment),
                )
            conn.commit()
            return {
                "turn_id": tid,
                "status": status,
                "sealed": True,
                "evidence_set_sha256": evidence_hash,
                "opened_spans": rows,
                "conflicts": conflicts,
                "freshness_required": bool(freshness_required),
                "authority_granted": False,
            }
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _output_manifest(sentences: Iterable[Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        if isinstance(sentences, (str, bytes)):
            raise EvidenceLedgerValidationError("output sentences must not be a string")
        try:
            supplied = list(itertools.islice(iter(sentences), MAX_OUTPUT_SENTENCES + 1))
        except TypeError as exc:
            raise EvidenceLedgerValidationError("output sentences must be iterable") from exc
        if not supplied or len(supplied) > MAX_OUTPUT_SENTENCES:
            raise EvidenceLedgerValidationError("output needs 1-4096 bounded sentences")
        cleaned: List[str] = []
        manifest: List[Dict[str, Any]] = []
        total_chars = 0
        for index, value in enumerate(supplied):
            sentence = _clean_text(value, field="output_sentence", limit=MAX_CLAIM_CHARS)
            if sentence != value:
                raise EvidenceLedgerValidationError(
                    "output sentences must not have surrounding whitespace"
                )
            total_chars += len(sentence)
            if total_chars > MAX_OUTPUT_CHARS:
                raise EvidenceLedgerValidationError(
                    f"output sentences exceed {MAX_OUTPUT_CHARS} total characters"
                )
            cleaned.append(sentence)
            manifest.append(
                {
                    "sentence_index": index,
                    "text_sha256": _text_hash(sentence),
                }
            )
        return cleaned, manifest

    def _issue_generated_output_capability(
        self, turn_id: str, sentences: Iterable[Any]
    ) -> object:
        """Mint a manifest receipt inside the trusted server-generation adapter."""

        tid = _identifier(turn_id, field="turn_id")
        _cleaned, manifest = self._output_manifest(sentences)
        manifest_json = _canonical_json(manifest)
        manifest_sha = _domain_hash("output-sentence-manifest", manifest)
        output_sha = _domain_hash(
            "generated-output", {"turn_id": tid, "sentence_hashes": manifest}
        )
        capability_values = {
            "turn_id": tid,
            "sentence_count": len(manifest),
            "sentence_hashes_json": manifest_json,
            "output_sha256": output_sha,
            "manifest_sha256": manifest_sha,
        }
        return _GeneratedOutputCapability(
            self._capability_token,
            **capability_values,
            capability_mac=_capability_mac(
                self._capability_key, "generated-output", capability_values
            ),
        )

    def _output_row(self, conn: sqlite3.Connection, turn_id: str) -> Optional[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"SELECT * FROM {self._OUTPUTS} WHERE turn_id = ?", (turn_id,)
        ).fetchone()

    def _validated_output_row(
        self, conn: sqlite3.Connection, turn_id: str
    ) -> Optional[Tuple[sqlite3.Row, List[Dict[str, Any]]]]:
        row = self._output_row(conn, turn_id)
        if row is None:
            return None
        try:
            count = row["sentence_count"]
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                or count > MAX_OUTPUT_SENTENCES
                or row["origin"] != "server_generation"
                or row["authority_granted"] != 0
            ):
                raise EvidenceLedgerError("output manifest shape integrity check failed")
            manifest = json.loads(row["sentence_hashes_json"])
            if not isinstance(manifest, list) or len(manifest) != count:
                raise EvidenceLedgerError("output manifest cardinality integrity check failed")
            expected_manifest: List[Dict[str, Any]] = []
            for index, item in enumerate(manifest):
                if not isinstance(item, dict) or set(item) != {"sentence_index", "text_sha256"}:
                    raise EvidenceLedgerError("output manifest entry integrity check failed")
                if item["sentence_index"] != index:
                    raise EvidenceLedgerError("output manifest order integrity check failed")
                expected_manifest.append(
                    {
                        "sentence_index": index,
                        "text_sha256": _sha256(item["text_sha256"], field="text_sha256"),
                    }
                )
            canonical = _canonical_json(expected_manifest)
            if row["sentence_hashes_json"] != canonical:
                raise EvidenceLedgerError("output manifest canonicalization integrity check failed")
            expected_manifest_sha = _domain_hash("output-sentence-manifest", expected_manifest)
            expected_output_sha = _domain_hash(
                "generated-output",
                {"turn_id": turn_id, "sentence_hashes": expected_manifest},
            )
            if (
                row["manifest_sha256"] != expected_manifest_sha
                or row["output_sha256"] != expected_output_sha
            ):
                raise EvidenceLedgerError("output manifest hash integrity check failed")
        except EvidenceLedgerError:
            raise
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise EvidenceLedgerError("output manifest integrity check failed") from exc
        return row, expected_manifest

    def bind_output(
        self,
        turn_id: str,
        sentences: Iterable[Any],
        *,
        output_capability: object = None,
    ) -> Dict[str, Any]:
        """Bind a sealed turn to exact ordered sentence hashes before claim recording."""

        tid = _identifier(turn_id, field="turn_id")
        _cleaned, manifest = self._output_manifest(sentences)
        manifest_json = _canonical_json(manifest)
        manifest_sha = _domain_hash("output-sentence-manifest", manifest)
        output_sha = _domain_hash(
            "generated-output", {"turn_id": tid, "sentence_hashes": manifest}
        )
        if not isinstance(output_capability, _GeneratedOutputCapability):
            raise EvidenceLedgerValidationError(
                "output binding requires a trusted server-generation capability"
            )
        if output_capability._ledger_token is not self._capability_token:
            raise EvidenceLedgerValidationError("output capability belongs to another ledger")
        expected_capability = {
            "turn_id": tid,
            "sentence_count": len(manifest),
            "sentence_hashes_json": manifest_json,
            "output_sha256": output_sha,
            "manifest_sha256": manifest_sha,
        }
        supplied_capability = {
            key: output_capability._value(key) for key in expected_capability
        }
        supplied_mac = output_capability._value("capability_mac")
        if (
            any(
                supplied_capability[key] != value
                for key, value in expected_capability.items()
            )
            or not isinstance(supplied_mac, str)
            or _SHA256_RE.fullmatch(supplied_mac) is None
            or not hmac.compare_digest(
                supplied_mac,
                _capability_mac(
                    self._capability_key, "generated-output", supplied_capability
                ),
            )
        ):
            raise EvidenceLedgerValidationError("output capability does not match the output")
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if self._turn_row(conn, tid) is None:
                raise EvidenceLedgerValidationError("turn must be sealed before binding output")
            self._validated_turn_state(conn, tid)
            existing = self._validated_output_row(conn, tid)
            if existing is not None:
                row, _existing_manifest = existing
                if (
                    row["sentence_count"] != len(manifest)
                    or row["sentence_hashes_json"] != manifest_json
                    or row["output_sha256"] != output_sha
                    or row["manifest_sha256"] != manifest_sha
                ):
                    raise EvidenceLedgerValidationError("bound output is immutable")
            else:
                claim_count = int(
                    conn.execute(
                        f"SELECT COUNT(*) FROM {self._CLAIMS} WHERE turn_id = ?", (tid,)
                    ).fetchone()[0]
                )
                if claim_count:
                    raise EvidenceLedgerValidationError("output must be bound before claims")
                conn.execute(
                    f"""
                    INSERT INTO {self._OUTPUTS}(
                        turn_id, sentence_count, sentence_hashes_json, output_sha256,
                        manifest_sha256, origin, authority_granted
                    ) VALUES (?, ?, ?, ?, ?, 'server_generation', 0)
                    """,
                    (tid, len(manifest), manifest_json, output_sha, manifest_sha),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "turn_id": tid,
            "sentence_count": len(manifest),
            "sentence_hashes": manifest,
            "output_sha256": output_sha,
            "manifest_sha256": manifest_sha,
            "origin": "server_generation",
            "stores_output_text": False,
            "authority_granted": False,
        }

    def _conflict_pairs(self, conn: sqlite3.Connection, snapshot_ids: set[str]) -> List[Dict[str, Any]]:
        if len(snapshot_ids) < 2:
            return []
        rows = conn.execute(
            f"SELECT conflict_id, left_snapshot_id, right_snapshot_id, reason FROM {self._CONFLICTS}"
        ).fetchall()
        return [
            {
                "conflict_id": row[0],
                "left_snapshot_id": row[1],
                "right_snapshot_id": row[2],
                "reason": row[3],
            }
            for row in rows
            if row[1] in snapshot_ids and row[2] in snapshot_ids
        ]

    def _opened_span(self, conn: sqlite3.Connection, turn_id: str, snapshot_id: str, span_id: str) -> Optional[sqlite3.Row]:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            f"""
            SELECT o.sequence, o.span_sha256, s.text
            FROM {self._OPENED} o
            JOIN {self._SPANS} s ON s.snapshot_id = o.snapshot_id AND s.span_id = o.span_id
            WHERE o.turn_id = ? AND o.snapshot_id = ? AND o.span_id = ?
            """,
            (turn_id, snapshot_id, span_id),
        ).fetchone()

    def _run_deterministic_checker(
        self,
        turn_id: str,
        sentence_index: int,
        claim_text: str,
        *,
        snapshot_id: str,
        span_id: str,
        checker_id: str,
        run_nonce: str,
    ) -> object:
        """Run one configured checker and mint a ledger-bound immutable receipt."""

        tid = _identifier(turn_id, field="turn_id")
        if (
            isinstance(sentence_index, bool)
            or not isinstance(sentence_index, int)
            or sentence_index < 0
            or sentence_index >= MAX_OUTPUT_SENTENCES
        ):
            raise EvidenceLedgerValidationError("sentence_index must be a bounded non-negative integer")
        claim = _clean_text(claim_text, field="claim_text", limit=MAX_CLAIM_CHARS)
        if claim != claim_text:
            raise EvidenceLedgerValidationError(
                "claim_text must not have surrounding whitespace"
            )
        sid = _identifier(snapshot_id, field="snapshot_id", prefix="snap-")
        span = _identifier(span_id, field="span_id")
        checker = _identifier(checker_id, field="checker_id")
        nonce = _clean_text(run_nonce, field="run_nonce", limit=MAX_TURN_ID_CHARS)
        nonce_sha256 = _text_hash(nonce)
        descriptor = self._deterministic_checkers.get(checker)
        if descriptor is None:
            raise EvidenceLedgerValidationError("checker_id is not allowlisted for this ledger")
        binding_sha = _claim_binding_hash(
            turn_id=tid,
            sentence_index=sentence_index,
            claim_text=claim,
            snapshot_id=sid,
            span_id=span,
            relation="inference",
        )
        issued_at = self._clock_now(field="checker_issued_at")

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if self._turn_row(conn, tid) is None:
                raise EvidenceLedgerValidationError("turn must be sealed before checking a claim")
            self._validated_turn_state(conn, tid)
            if conn.execute(
                f"SELECT 1 FROM {self._CHECKER_RUNS} WHERE run_nonce_sha256 = ?",
                (nonce_sha256,),
            ).fetchone() is not None:
                raise EvidenceLedgerValidationError("checker run nonce has already been used")
            if conn.execute(
                f"""
                SELECT 1 FROM {self._CHECKER_RUNS}
                WHERE claim_binding_sha256 = ? AND checker_id = ?
                  AND checker_version = ? AND checker_source_sha256 = ?
                """,
                (
                    binding_sha,
                    checker,
                    descriptor["version"],
                    descriptor["source_sha256"],
                ),
            ).fetchone() is not None:
                raise EvidenceLedgerValidationError(
                    "this checker version has already run for the claim binding"
                )
            output = self._validated_output_row(conn, tid)
            if output is None:
                raise EvidenceLedgerValidationError("output must be bound before checking a claim")
            _output_row, manifest = output
            if sentence_index >= len(manifest) or manifest[sentence_index]["text_sha256"] != _text_hash(claim):
                raise EvidenceLedgerValidationError("claim text does not match the bound output sentence")
            if self._validated_snapshot_row(conn, sid) is None:
                raise EvidenceLedgerError("checker source snapshot integrity check failed")
            opened = self._opened_span(conn, tid, sid, span)
            if opened is None:
                raise EvidenceLedgerValidationError("checker must bind to an opened evidence span")
            source_span = self._span_row(conn, sid, span)
            if source_span is None or opened["span_sha256"] != source_span["span_sha256"]:
                raise EvidenceLedgerError("checker source span integrity check failed")
            span_sha256 = str(source_span["span_sha256"])
            source_text = str(source_span["text"])
            run_values = {
                "run_nonce_sha256": nonce_sha256,
                "turn_id": tid,
                "sentence_index": sentence_index,
                "claim_text_sha256": _text_hash(claim),
                "claim_binding_sha256": binding_sha,
                "snapshot_id": sid,
                "span_id": span,
                "span_sha256": span_sha256,
                "checker_id": checker,
                "checker_version": descriptor["version"],
                "checker_source_sha256": descriptor["source_sha256"],
                "issued_at": issued_at,
                "authority_granted": False,
            }
            conn.execute(
                f"""
                INSERT INTO {self._CHECKER_RUNS}(
                    run_nonce_sha256, turn_id, sentence_index, claim_text_sha256,
                    claim_binding_sha256,
                    snapshot_id, span_id, span_sha256, checker_id, checker_version,
                    checker_source_sha256, issued_at, run_sha256, authority_granted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    nonce_sha256,
                    tid,
                    sentence_index,
                    _text_hash(claim),
                    binding_sha,
                    sid,
                    span,
                    span_sha256,
                    checker,
                    descriptor["version"],
                    descriptor["source_sha256"],
                    issued_at,
                    _checker_run_hash(run_values),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise EvidenceLedgerValidationError(
                "checker run nonce or claim binding has already been used"
            ) from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        try:
            raw_result = descriptor["check"](
                claim_text=claim,
                source_text=source_text,
                turn_id=tid,
                sentence_index=sentence_index,
                snapshot_id=sid,
                span_id=span,
                span_sha256=span_sha256,
            )
        except Exception as exc:
            raise EvidenceLedgerError("deterministic checker execution failed") from exc
        if not isinstance(raw_result, Mapping):
            raise EvidenceLedgerError("deterministic checker returned an invalid result")
        status = raw_result.get("status")
        independent = raw_result.get("algorithmically_independent")
        if status not in CHECKER_STATUSES or type(independent) is not bool:
            raise EvidenceLedgerError("deterministic checker returned an invalid result")
        if status == "passed" and not independent:
            raise EvidenceLedgerError("a passing checker must be algorithmically independent")
        reason = _clean_text(
            raw_result.get("reason", ""),
            field="checker_reason",
            limit=MAX_CHECKER_REASON_CHARS,
            required=False,
        )
        values = {
            "turn_id": tid,
            "sentence_index": sentence_index,
            "claim_binding_sha256": binding_sha,
            "snapshot_id": sid,
            "span_id": span,
            "span_sha256": span_sha256,
            "checker_id": checker,
            "checker_version": descriptor["version"],
            "checker_source_sha256": descriptor["source_sha256"],
            "status": status,
            "algorithmically_independent": independent,
            "reason_sha256": _text_hash(reason),
            "run_nonce_sha256": nonce_sha256,
        }
        values["receipt_sha256"] = _checker_receipt_hash(values)
        values["capability_mac"] = _capability_mac(
            self._capability_key, "checker-receipt", values
        )
        return _DeterministicCheckerReceipt(self._capability_token, **values)

    def record_claim(
        self,
        turn_id: str,
        sentence_index: int,
        claim_text: str,
        *,
        snapshot_id: str,
        span_id: str,
        relation: str,
        deterministic_checker_id: str = "",
        deterministic_verified: bool = False,
        checker_receipt: object = None,
    ) -> Dict[str, Any]:
        """Record a sentence provenance relation after a turn has been sealed."""

        tid = _identifier(turn_id, field="turn_id")
        if isinstance(sentence_index, bool) or not isinstance(sentence_index, int) or sentence_index < 0:
            raise EvidenceLedgerValidationError("sentence_index must be a non-negative integer")
        if sentence_index >= MAX_OUTPUT_SENTENCES:
            raise EvidenceLedgerValidationError("sentence_index is too large")
        claim = _clean_text(claim_text, field="claim_text", limit=MAX_CLAIM_CHARS)
        if claim != claim_text:
            raise EvidenceLedgerValidationError(
                "claim_text must not have surrounding whitespace"
            )
        sid = _identifier(snapshot_id, field="snapshot_id", prefix="snap-")
        span = _identifier(span_id, field="span_id")
        if not isinstance(relation, str) or relation not in PROVENANCE_RELATIONS:
            raise EvidenceLedgerValidationError("relation must be quotation, compression, or inference")
        if type(deterministic_verified) is not bool:
            raise EvidenceLedgerValidationError("deterministic_verified must be boolean")
        legacy_checker = _clean_text(
            deterministic_checker_id,
            field="deterministic_checker_id",
            limit=MAX_PROVIDER_CHARS,
            required=False,
        )
        if deterministic_verified or legacy_checker:
            raise EvidenceLedgerValidationError(
                "caller-asserted deterministic verification is not accepted; use a checker receipt"
            )
        if checker_receipt is not None and relation != "inference":
            raise EvidenceLedgerValidationError("checker receipts apply only to inference claims")
        binding_sha = _claim_binding_hash(
            turn_id=tid,
            sentence_index=sentence_index,
            claim_text=claim,
            snapshot_id=sid,
            span_id=span,
            relation=relation,
        )
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            turn = self._turn_row(conn, tid)
            if turn is None:
                raise EvidenceLedgerValidationError("turn must be sealed before recording claims")
            self._validated_turn_state(conn, tid)
            output = self._validated_output_row(conn, tid)
            if output is None:
                raise EvidenceLedgerValidationError("output must be bound before recording claims")
            _output_row, output_manifest = output
            if (
                sentence_index >= len(output_manifest)
                or output_manifest[sentence_index]["text_sha256"] != _text_hash(claim)
            ):
                raise EvidenceLedgerValidationError("claim text does not match the bound output sentence")
            if self._validated_snapshot_row(conn, sid) is None:
                raise EvidenceLedgerError("claim source snapshot integrity check failed")
            opened = self._opened_span(conn, tid, sid, span)
            if opened is None:
                raise EvidenceLedgerValidationError("claim must bind to an opened evidence span")
            source_span = self._span_row(conn, sid, span)
            if source_span is None or opened["span_sha256"] != source_span["span_sha256"]:
                raise EvidenceLedgerError("claim source span integrity check failed")
            quote_verified = relation == "quotation" and _normalise_quote(claim) in _normalise_quote(opened["text"])
            if relation == "quotation" and not quote_verified:
                raise EvidenceLedgerValidationError("quotation does not match the opened source span")
            receipt_values: Optional[Dict[str, Any]] = None
            if checker_receipt is not None:
                if not isinstance(checker_receipt, _DeterministicCheckerReceipt):
                    raise EvidenceLedgerValidationError("checker receipt is invalid")
                if checker_receipt._ledger_token is not self._capability_token:
                    raise EvidenceLedgerValidationError("checker receipt belongs to another ledger")
                try:
                    raw_receipt = {
                        key: checker_receipt._value(key)
                        for key in (
                            "turn_id",
                            "sentence_index",
                            "claim_binding_sha256",
                            "snapshot_id",
                            "span_id",
                            "span_sha256",
                            "checker_id",
                            "checker_version",
                            "checker_source_sha256",
                            "status",
                            "algorithmically_independent",
                            "reason_sha256",
                            "run_nonce_sha256",
                            "receipt_sha256",
                        )
                    }
                    receipt_index = raw_receipt["sentence_index"]
                    if (
                        isinstance(receipt_index, bool)
                        or not isinstance(receipt_index, int)
                        or receipt_index < 0
                        or receipt_index >= MAX_OUTPUT_SENTENCES
                        or raw_receipt["status"] not in CHECKER_STATUSES
                        or type(raw_receipt["algorithmically_independent"]) is not bool
                    ):
                        raise EvidenceLedgerValidationError("checker receipt has invalid fields")
                    receipt_values = {
                        "turn_id": _identifier(raw_receipt["turn_id"], field="turn_id"),
                        "sentence_index": receipt_index,
                        "claim_binding_sha256": _sha256(
                            raw_receipt["claim_binding_sha256"], field="claim_binding_sha256"
                        ),
                        "snapshot_id": _identifier(
                            raw_receipt["snapshot_id"], field="snapshot_id", prefix="snap-"
                        ),
                        "span_id": _identifier(raw_receipt["span_id"], field="span_id"),
                        "span_sha256": _sha256(
                            raw_receipt["span_sha256"], field="span_sha256"
                        ),
                        "checker_id": _identifier(
                            raw_receipt["checker_id"], field="checker_id"
                        ),
                        "checker_version": _clean_text(
                            raw_receipt["checker_version"],
                            field="checker_version",
                            limit=MAX_PROVIDER_CHARS,
                        ),
                        "checker_source_sha256": _sha256(
                            raw_receipt["checker_source_sha256"],
                            field="checker_source_sha256",
                        ),
                        "status": raw_receipt["status"],
                        "algorithmically_independent": raw_receipt[
                            "algorithmically_independent"
                        ],
                        "reason_sha256": _sha256(
                            raw_receipt["reason_sha256"], field="reason_sha256"
                        ),
                        "run_nonce_sha256": _sha256(
                            raw_receipt["run_nonce_sha256"], field="run_nonce_sha256"
                        ),
                        "receipt_sha256": _sha256(
                            raw_receipt["receipt_sha256"], field="receipt_sha256"
                        ),
                    }
                    supplied_receipt_mac = _sha256(
                        checker_receipt._value("capability_mac"),
                        field="capability_mac",
                    )
                except (AttributeError, KeyError, TypeError, ValueError) as exc:
                    raise EvidenceLedgerValidationError("checker receipt is invalid") from exc
                if receipt_values != raw_receipt:
                    raise EvidenceLedgerValidationError("checker receipt is not canonical")
                expected_binding = {
                    "turn_id": tid,
                    "sentence_index": sentence_index,
                    "claim_binding_sha256": binding_sha,
                    "snapshot_id": sid,
                    "span_id": span,
                    "span_sha256": source_span["span_sha256"],
                }
                if any(receipt_values[key] != value for key, value in expected_binding.items()):
                    raise EvidenceLedgerValidationError("checker receipt does not match this claim")
                descriptor = self._deterministic_checkers.get(receipt_values["checker_id"])
                if (
                    descriptor is None
                    or descriptor["version"] != receipt_values["checker_version"]
                    or descriptor["source_sha256"] != receipt_values["checker_source_sha256"]
                    or receipt_values["status"] not in CHECKER_STATUSES
                    or receipt_values["receipt_sha256"]
                    != _checker_receipt_hash(receipt_values)
                    or not hmac.compare_digest(
                        supplied_receipt_mac,
                        _capability_mac(
                            self._capability_key, "checker-receipt", receipt_values
                        ),
                    )
                ):
                    raise EvidenceLedgerValidationError("checker receipt failed validation")
            checker = "" if receipt_values is None else str(receipt_values["checker_id"])
            receipt_sha = "" if receipt_values is None else str(receipt_values["receipt_sha256"])
            checker_passed = receipt_values is not None and receipt_values["status"] == "passed"
            mechanically_verified = bool(quote_verified or checker_passed)
            status = (
                "verified_quotation"
                if quote_verified
                else "checked_inference_no_authority"
                if checker_passed
                else "auditable_compression"
                if relation == "compression"
                else "defer_inference"
            )
            claim_hash = _claim_record_hash(
                turn_id=tid,
                sentence_index=sentence_index,
                claim_text=claim,
                snapshot_id=sid,
                span_id=span,
                relation=relation,
                quote_verified=quote_verified,
                mechanically_verified=mechanically_verified,
                checker_id=checker,
                checker_receipt_sha256=receipt_sha,
            )
            existing = conn.execute(
                f"SELECT claim_sha256 FROM {self._CLAIMS} WHERE turn_id = ? AND sentence_index = ?",
                (tid, sentence_index),
            ).fetchone()
            if existing is not None:
                if existing[0] != claim_hash:
                    raise EvidenceLedgerValidationError("sentence provenance is immutable")
            else:
                conn.execute(
                    f"""
                    INSERT INTO {self._CLAIMS}(
                        turn_id, sentence_index, claim_text, snapshot_id, span_id,
                        relation, quote_verified, mechanically_verified, checker_id,
                        claim_sha256, authority_granted
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        tid,
                        sentence_index,
                        claim,
                        sid,
                        span,
                        relation,
                        int(quote_verified),
                        int(mechanically_verified),
                        checker,
                        claim_hash,
                    ),
                )
                if receipt_values is not None:
                    conn.execute(
                        f"""
                        INSERT INTO {self._CHECKER_RECEIPTS}(
                            turn_id, sentence_index, claim_binding_sha256, claim_sha256,
                            snapshot_id, span_id, span_sha256, checker_id, checker_version,
                            checker_source_sha256, status, reason_sha256, run_nonce_sha256,
                            algorithmically_independent, receipt_sha256, authority_granted
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            tid,
                            sentence_index,
                            receipt_values["claim_binding_sha256"],
                            claim_hash,
                            sid,
                            span,
                            receipt_values["span_sha256"],
                            checker,
                            receipt_values["checker_version"],
                            receipt_values["checker_source_sha256"],
                            receipt_values["status"],
                            receipt_values["reason_sha256"],
                            receipt_values["run_nonce_sha256"],
                            int(receipt_values["algorithmically_independent"]),
                            receipt_sha,
                        ),
                    )
            conn.commit()
            return {
                "turn_id": tid,
                "sentence_index": sentence_index,
                "claim_text": claim,
                "snapshot_id": sid,
                "span_id": span,
                "relation": relation,
                "quote_verified": quote_verified,
                "mechanically_verified": mechanically_verified,
                "checker_id": checker,
                "checker_receipt_sha256": receipt_sha,
                "status": status,
                "claim_sha256": claim_hash,
                "authority_granted": False,
            }
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise EvidenceLedgerValidationError(
                "checker receipt replay or ledger constraint violation"
            ) from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _validated_turn_state(
        self, conn: sqlite3.Connection, turn_id: str
    ) -> Dict[str, Any]:
        """Recompute every hash and semantic invariant reachable from one turn."""

        turn = self._turn_row(conn, turn_id)
        if turn is None:
            raise EvidenceLedgerValidationError("turn does not exist")
        try:
            tid = _identifier(turn["turn_id"], field="turn_id")
            if tid != turn_id:
                raise EvidenceLedgerError("turn identity integrity check failed")
            sealed_at = _timestamp(turn["sealed_at"], field="sealed_at")
            evidence_set_sha = _sha256(
                turn["evidence_set_sha256"], field="evidence_set_sha256"
            )
            if turn["freshness_required"] not in (0, 1) or turn["authority_granted"] != 0:
                raise EvidenceLedgerError("turn policy integrity check failed")
            if turn["status"] not in {"sealed", "conflict_defer"}:
                raise EvidenceLedgerError("turn status integrity check failed")
            attestation = conn.execute(
                f"SELECT * FROM {self._TURN_ATTESTATIONS} WHERE turn_id = ?", (tid,)
            ).fetchone()
            if attestation is None or attestation["authority_granted"] != 0:
                raise EvidenceLedgerError("turn attestation is missing or invalid")
            attested_at = _timestamp(attestation["attested_at"], field="attested_at")
            sealed_conflict_ids = tuple(
                json.loads(attestation["sealed_conflict_ids_json"])
            ) if "sealed_conflict_ids_json" in attestation.keys() else ()
            attestation_values = {
                "turn_id": tid,
                "sealed_at": sealed_at,
                "evidence_set_sha256": evidence_set_sha,
                "freshness_required": bool(turn["freshness_required"]),
                "status": str(turn["status"]),
                "sealed_conflict_ids": sealed_conflict_ids,
                "attested_at": attested_at,
                "authority_granted": False,
            }
            if attestation["turn_sha256"] != _turn_attestation_hash(attestation_values):
                raise EvidenceLedgerError("turn attestation integrity check failed")

            opened = conn.execute(
                f"""
                SELECT sequence, snapshot_id, span_id, span_sha256
                FROM {self._OPENED} WHERE turn_id = ? ORDER BY sequence
                """,
                (tid,),
            ).fetchall()
            if not opened or len(opened) > MAX_OPENED_SPANS_PER_TURN:
                raise EvidenceLedgerError("opened evidence cardinality integrity check failed")
            opened_manifest: List[Dict[str, Any]] = []
            opened_keys: set[Tuple[str, str]] = set()
            snapshot_rows: Dict[str, sqlite3.Row] = {}
            for expected_sequence, row in enumerate(opened):
                if row["sequence"] != expected_sequence:
                    raise EvidenceLedgerError("opened evidence order integrity check failed")
                sid = _identifier(row["snapshot_id"], field="snapshot_id", prefix="snap-")
                span_id = _identifier(row["span_id"], field="span_id")
                if sid != row["snapshot_id"] or span_id != row["span_id"]:
                    raise EvidenceLedgerError("opened evidence identity integrity check failed")
                if (sid, span_id) in opened_keys:
                    raise EvidenceLedgerError("opened evidence uniqueness integrity check failed")
                opened_keys.add((sid, span_id))
                if sid not in snapshot_rows:
                    snapshot = self._validated_snapshot_row(conn, sid)
                    if snapshot is None:
                        raise EvidenceLedgerError("opened evidence snapshot integrity check failed")
                    snapshot_rows[sid] = snapshot
                span_row = self._span_row(conn, sid, span_id)
                if span_row is None or span_row["span_sha256"] != row["span_sha256"]:
                    raise EvidenceLedgerError("opened evidence span integrity check failed")
                opened_manifest.append(
                    {
                        "sequence": expected_sequence,
                        "snapshot_id": sid,
                        "span_id": span_id,
                        "span_sha256": str(row["span_sha256"]),
                    }
                )
            if evidence_set_sha != _domain_hash("opened-evidence", opened_manifest):
                raise EvidenceLedgerError("turn evidence-set integrity check failed")
            sealed_conflicts = self._conflict_pairs(conn, set(snapshot_rows))
            if turn["status"] == "conflict_defer" and not sealed_conflicts:
                raise EvidenceLedgerError("turn conflict-status integrity check failed")

            output = self._validated_output_row(conn, tid)
            output_manifest = [] if output is None else output[1]
            claim_rows = conn.execute(
                f"SELECT * FROM {self._CLAIMS} WHERE turn_id = ? ORDER BY sentence_index",
                (tid,),
            ).fetchall()
            receipt_rows = conn.execute(
                f"SELECT * FROM {self._CHECKER_RECEIPTS} WHERE turn_id = ? ORDER BY sentence_index",
                (tid,),
            ).fetchall()
            run_rows = conn.execute(
                f"SELECT * FROM {self._CHECKER_RUNS} WHERE turn_id = ? ORDER BY run_nonce_sha256",
                (tid,),
            ).fetchall()
            runs: Dict[str, Dict[str, Any]] = {}
            for row in run_rows:
                nonce_sha = _sha256(row["run_nonce_sha256"], field="run_nonce_sha256")
                index = row["sentence_index"]
                if (
                    nonce_sha in runs
                    or isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(output_manifest)
                    or row["authority_granted"] != 0
                ):
                    raise EvidenceLedgerError("checker run shape integrity check failed")
                claim_text_sha = _sha256(
                    row["claim_text_sha256"], field="claim_text_sha256"
                )
                binding_sha = _sha256(
                    row["claim_binding_sha256"], field="claim_binding_sha256"
                )
                sid = _identifier(row["snapshot_id"], field="snapshot_id", prefix="snap-")
                span_id = _identifier(row["span_id"], field="span_id")
                checker_id = _identifier(row["checker_id"], field="checker_id")
                checker_version = _clean_text(
                    row["checker_version"],
                    field="checker_version",
                    limit=MAX_PROVIDER_CHARS,
                )
                source_sha = _sha256(
                    row["checker_source_sha256"], field="checker_source_sha256"
                )
                issued_at = _timestamp(row["issued_at"], field="checker_issued_at")
                span_sha = _sha256(row["span_sha256"], field="span_sha256")
                if (
                    row["turn_id"] != tid
                    or sid != row["snapshot_id"]
                    or span_id != row["span_id"]
                    or checker_id != row["checker_id"]
                    or checker_version != row["checker_version"]
                    or (sid, span_id) not in opened_keys
                    or output_manifest[index]["text_sha256"] != claim_text_sha
                ):
                    raise EvidenceLedgerError("checker run binding integrity check failed")
                span_row = self._span_row(conn, sid, span_id)
                if span_row is None or span_row["span_sha256"] != span_sha:
                    raise EvidenceLedgerError("checker run source integrity check failed")
                expected_binding = _claim_binding_hash_from_digest(
                    turn_id=tid,
                    sentence_index=index,
                    claim_text_sha256=claim_text_sha,
                    snapshot_id=sid,
                    span_id=span_id,
                    relation="inference",
                )
                run_values = {
                    "run_nonce_sha256": nonce_sha,
                    "turn_id": tid,
                    "sentence_index": index,
                    "claim_text_sha256": claim_text_sha,
                    "claim_binding_sha256": binding_sha,
                    "snapshot_id": sid,
                    "span_id": span_id,
                    "span_sha256": span_sha,
                    "checker_id": checker_id,
                    "checker_version": checker_version,
                    "checker_source_sha256": source_sha,
                    "issued_at": issued_at,
                    "authority_granted": False,
                }
                if (
                    binding_sha != expected_binding
                    or row["run_sha256"] != _checker_run_hash(run_values)
                ):
                    raise EvidenceLedgerError("checker run hash integrity check failed")
                runs[nonce_sha] = run_values
            receipts = {int(row["sentence_index"]): row for row in receipt_rows}
            if len(receipts) != len(receipt_rows):
                raise EvidenceLedgerError("checker receipt uniqueness integrity check failed")
            validated_claims: List[Dict[str, Any]] = []
            seen_indices: set[int] = set()
            for row in claim_rows:
                index = row["sentence_index"]
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= MAX_OUTPUT_SENTENCES
                    or index in seen_indices
                ):
                    raise EvidenceLedgerError("claim sentence-index integrity check failed")
                seen_indices.add(index)
                claim = _clean_text(row["claim_text"], field="claim_text", limit=MAX_CLAIM_CHARS)
                sid = _identifier(row["snapshot_id"], field="snapshot_id", prefix="snap-")
                span_id = _identifier(row["span_id"], field="span_id")
                if (
                    claim != row["claim_text"]
                    or sid != row["snapshot_id"]
                    or span_id != row["span_id"]
                ):
                    raise EvidenceLedgerError("claim canonicalization integrity check failed")
                relation = row["relation"]
                if relation not in PROVENANCE_RELATIONS or (sid, span_id) not in opened_keys:
                    raise EvidenceLedgerError("claim source binding integrity check failed")
                if row["quote_verified"] not in (0, 1) or row["mechanically_verified"] not in (0, 1):
                    raise EvidenceLedgerError("claim verification-bit integrity check failed")
                if row["authority_granted"] != 0:
                    raise EvidenceLedgerError("claim authority integrity check failed")
                if output is None or index >= len(output_manifest):
                    raise EvidenceLedgerError("claim has no bound output sentence")
                if output_manifest[index]["text_sha256"] != _text_hash(claim):
                    raise EvidenceLedgerError("claim text/output binding integrity check failed")
                span_row = self._span_row(conn, sid, span_id)
                if span_row is None:
                    raise EvidenceLedgerError("claim source span integrity check failed")
                quote_matches = _normalise_quote(claim) in _normalise_quote(span_row["text"])
                receipt = receipts.pop(index, None)
                receipt_sha = ""
                expected_quote = relation == "quotation" and quote_matches
                expected_mechanical = expected_quote
                expected_checker = ""
                binding_sha = _claim_binding_hash(
                    turn_id=tid,
                    sentence_index=index,
                    claim_text=claim,
                    snapshot_id=sid,
                    span_id=span_id,
                    relation=relation,
                )
                if relation == "quotation" and not quote_matches:
                    raise EvidenceLedgerError("quotation relation integrity check failed")
                if relation != "inference" and receipt is not None:
                    raise EvidenceLedgerError("non-inference claim has a checker receipt")
                if receipt is not None:
                    independent = receipt["algorithmically_independent"]
                    if independent not in (0, 1) or receipt["authority_granted"] != 0:
                        raise EvidenceLedgerError("checker receipt policy integrity check failed")
                    receipt_values = {
                        "turn_id": tid,
                        "sentence_index": index,
                        "claim_binding_sha256": _sha256(
                            receipt["claim_binding_sha256"], field="claim_binding_sha256"
                        ),
                        "snapshot_id": _identifier(
                            receipt["snapshot_id"], field="snapshot_id", prefix="snap-"
                        ),
                        "span_id": _identifier(receipt["span_id"], field="span_id"),
                        "span_sha256": _sha256(receipt["span_sha256"], field="span_sha256"),
                        "checker_id": _identifier(receipt["checker_id"], field="checker_id"),
                        "checker_version": _clean_text(
                            receipt["checker_version"],
                            field="checker_version",
                            limit=MAX_PROVIDER_CHARS,
                        ),
                        "checker_source_sha256": _sha256(
                            receipt["checker_source_sha256"], field="checker_source_sha256"
                        ),
                        "status": receipt["status"],
                        "algorithmically_independent": bool(independent),
                        "reason_sha256": _sha256(receipt["reason_sha256"], field="reason_sha256"),
                        "run_nonce_sha256": _sha256(
                            receipt["run_nonce_sha256"], field="run_nonce_sha256"
                        ),
                    }
                    if receipt_values["status"] not in CHECKER_STATUSES:
                        raise EvidenceLedgerError("checker receipt status integrity check failed")
                    if (
                        receipt["turn_id"] != tid
                        or receipt_values["snapshot_id"] != receipt["snapshot_id"]
                        or receipt_values["span_id"] != receipt["span_id"]
                        or receipt_values["checker_id"] != receipt["checker_id"]
                        or receipt_values["checker_version"] != receipt["checker_version"]
                    ):
                        raise EvidenceLedgerError(
                            "checker receipt canonicalization integrity check failed"
                        )
                    checker_run = runs.get(receipt_values["run_nonce_sha256"])
                    if checker_run is None or any(
                        checker_run[key] != receipt_values[key]
                        for key in (
                            "turn_id",
                            "sentence_index",
                            "claim_binding_sha256",
                            "snapshot_id",
                            "span_id",
                            "span_sha256",
                            "checker_id",
                            "checker_version",
                            "checker_source_sha256",
                        )
                    ):
                        raise EvidenceLedgerError("checker receipt/run binding integrity check failed")
                    if receipt_values["status"] == "passed" and not bool(independent):
                        raise EvidenceLedgerError("checker receipt independence integrity check failed")
                    if (
                        receipt_values["claim_binding_sha256"] != binding_sha
                        or receipt_values["snapshot_id"] != sid
                        or receipt_values["span_id"] != span_id
                        or receipt_values["span_sha256"] != span_row["span_sha256"]
                    ):
                        raise EvidenceLedgerError("checker receipt claim binding integrity check failed")
                    receipt_sha = _sha256(receipt["receipt_sha256"], field="receipt_sha256")
                    if receipt_sha != _checker_receipt_hash(receipt_values):
                        raise EvidenceLedgerError("checker receipt hash integrity check failed")
                    expected_checker = receipt_values["checker_id"]
                    expected_mechanical = receipt_values["status"] == "passed"
                if (
                    bool(row["quote_verified"]) != expected_quote
                    or bool(row["mechanically_verified"]) != expected_mechanical
                    or row["checker_id"] != expected_checker
                ):
                    raise EvidenceLedgerError("claim verification integrity check failed")
                expected_claim_sha = _claim_record_hash(
                    turn_id=tid,
                    sentence_index=index,
                    claim_text=claim,
                    snapshot_id=sid,
                    span_id=span_id,
                    relation=relation,
                    quote_verified=expected_quote,
                    mechanically_verified=expected_mechanical,
                    checker_id=expected_checker,
                    checker_receipt_sha256=receipt_sha,
                )
                if row["claim_sha256"] != expected_claim_sha:
                    raise EvidenceLedgerError("claim hash integrity check failed")
                if receipt is not None and receipt["claim_sha256"] != expected_claim_sha:
                    raise EvidenceLedgerError("checker receipt claim-hash integrity check failed")
                validated_claims.append(
                    {
                        "sentence_index": index,
                        "mechanically_verified": expected_mechanical,
                        "relation": relation,
                    }
                )
            if receipts:
                raise EvidenceLedgerError("orphan checker receipt integrity check failed")
        except EvidenceLedgerError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError) as exc:
            raise EvidenceLedgerError("turn integrity check failed") from exc
        return {
            "turn": turn,
            "opened": opened_manifest,
            "snapshot_rows": snapshot_rows,
            "output_manifest": output_manifest,
            "claims": validated_claims,
        }

    def declare_conflict(
        self,
        left_snapshot_id: str,
        right_snapshot_id: str,
        *,
        reason: str,
        created_at: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retain an explicit contradiction; never silently supersede either side."""

        left = _identifier(left_snapshot_id, field="left_snapshot_id", prefix="snap-")
        right = _identifier(right_snapshot_id, field="right_snapshot_id", prefix="snap-")
        if left == right:
            raise EvidenceLedgerValidationError("a snapshot cannot conflict with itself")
        ordered = tuple(sorted((left, right)))
        explanation = _clean_text(reason, field="reason", limit=MAX_CLAIM_CHARS)
        moment = self._clock_now(field="created_at") if created_at is None else _timestamp(created_at, field="created_at")
        conflict_id = "conflict-" + _domain_hash(
            "conflict", {"left": ordered[0], "right": ordered[1], "reason": explanation}
        )[:48]
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if (
                self._validated_snapshot_row(conn, ordered[0]) is None
                or self._validated_snapshot_row(conn, ordered[1]) is None
            ):
                raise EvidenceLedgerValidationError("conflict snapshots must exist")
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {self._CONFLICTS}(
                    conflict_id, left_snapshot_id, right_snapshot_id, reason, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (conflict_id, ordered[0], ordered[1], explanation, moment),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {
            "conflict_id": conflict_id,
            "left_snapshot_id": ordered[0],
            "right_snapshot_id": ordered[1],
            "reason": explanation,
            "authority_granted": False,
        }

    def evaluate_turn(self, turn_id: str, *, now: Optional[float] = None) -> Dict[str, Any]:
        """Return a shadow decision for freshness/conflict/provenance inspection."""

        tid = _identifier(turn_id, field="turn_id")
        moment = self._clock_now(field="now") if now is None else _timestamp(now, field="now")
        conn = self._connect()
        try:
            conn.execute("BEGIN")
            self._validate_global_rows(conn)
            state = self._validated_turn_state(conn, tid)
            turn = state["turn"]
            opened = state["opened"]
            snapshot_ids = set(state["snapshot_rows"])
            conflicts = self._conflict_pairs(conn, snapshot_ids)
            freshness = [
                {
                    "snapshot_id": snapshot_id,
                    "status": self._freshness(state["snapshot_rows"][snapshot_id], moment),
                }
                for snapshot_id in sorted(snapshot_ids)
            ]
            freshness_bad = [row for row in freshness if row["status"] != "current"]
            claims = state["claims"]
            expected_indices = set(range(len(state["output_manifest"])))
            linked_indices = {row["sentence_index"] for row in claims}
            mechanically_verified = sum(bool(row["mechanically_verified"]) for row in claims)
            coverage_complete = bool(expected_indices) and linked_indices == expected_indices
            verification_complete = coverage_complete and mechanically_verified == len(expected_indices)
            if conflicts:
                status = "conflict_defer"
            elif bool(turn["freshness_required"]) and freshness_bad:
                status = "freshness_defer"
            elif not state["output_manifest"]:
                status = "output_unbound_defer"
            elif not verification_complete:
                status = "coverage_defer"
            else:
                status = "shadow_recorded"
            result = {
                "turn_id": tid,
                "status": status,
                "opened_span_count": len(opened),
                "claim_count": len(claims),
                "freshness": freshness,
                "conflicts": conflicts,
                "evidence_set_sha256": turn["evidence_set_sha256"],
                "output_sentence_count": len(state["output_manifest"]),
                "linked_sentence_count": len(linked_indices),
                "mechanically_verified_sentence_count": mechanically_verified,
                "coverage_complete": coverage_complete,
                "verification_complete": verification_complete,
                "authority_granted": False,
            }
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _validate_global_rows(self, conn: sqlite3.Connection) -> None:
        """Validate append-only rows not owned by a single turn."""

        conn.row_factory = sqlite3.Row
        try:
            self._validate_schema_contract(conn)
            if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
                raise EvidenceLedgerError("foreign-key integrity check failed")
            schema_rows = conn.execute(
                f"SELECT version, applied_at FROM {self._SCHEMA_TABLE} ORDER BY applied_at, version"
            ).fetchall()
            versions: set[str] = set()
            for version, applied_at in schema_rows:
                if version in versions:
                    raise EvidenceLedgerError("schema history uniqueness integrity check failed")
                versions.add(str(version))
                _timestamp(applied_at, field="schema_applied_at")
            if LEDGER_SCHEMA_VERSION not in versions or versions - (
                LEGACY_LEDGER_SCHEMA_VERSIONS | {LEDGER_SCHEMA_VERSION}
            ):
                raise EvidenceLedgerError("schema history integrity check failed")

            for row in conn.execute(
                f"SELECT * FROM {self._CONFLICTS} ORDER BY conflict_id"
            ).fetchall():
                left = _identifier(
                    row["left_snapshot_id"], field="left_snapshot_id", prefix="snap-"
                )
                right = _identifier(
                    row["right_snapshot_id"], field="right_snapshot_id", prefix="snap-"
                )
                reason = _clean_text(row["reason"], field="reason", limit=MAX_CLAIM_CHARS)
                _timestamp(row["created_at"], field="created_at")
                if (
                    left != row["left_snapshot_id"]
                    or right != row["right_snapshot_id"]
                    or reason != row["reason"]
                ):
                    raise EvidenceLedgerError("conflict canonicalization integrity check failed")
                if left >= right:
                    raise EvidenceLedgerError("conflict ordering integrity check failed")
                expected = "conflict-" + _domain_hash(
                    "conflict", {"left": left, "right": right, "reason": reason}
                )[:48]
                if row["conflict_id"] != expected:
                    raise EvidenceLedgerError("conflict identity integrity check failed")

            revision_pairs: set[Tuple[str, str]] = set()
            for row in conn.execute(
                f"SELECT * FROM {self._REVISIONS} ORDER BY snapshot_id"
            ).fetchall():
                sid = _identifier(row["snapshot_id"], field="snapshot_id", prefix="snap-")
                supersedes = _identifier(
                    row["supersedes_snapshot_id"],
                    field="supersedes_snapshot_id",
                    prefix="snap-",
                )
                created_at = _timestamp(row["created_at"], field="created_at")
                snapshot = self._snapshot_row(conn, sid)
                if (
                    sid != row["snapshot_id"]
                    or supersedes != row["supersedes_snapshot_id"]
                    or row["revision_kind"] != "explicit_revision"
                    or sid == supersedes
                    or snapshot is None
                    or snapshot["supersedes_snapshot_id"] != supersedes
                    or _timestamp(snapshot["created_at"], field="snapshot_created_at")
                    != created_at
                ):
                    raise EvidenceLedgerError("revision link integrity check failed")
                revision_pairs.add((sid, supersedes))

            expected_revision_pairs: set[Tuple[str, str]] = set()
            for row in conn.execute(
                f"""
                SELECT snapshot_id, supersedes_snapshot_id
                FROM {self._SNAPSHOTS}
                WHERE supersedes_snapshot_id IS NOT NULL
                ORDER BY snapshot_id
                """
            ).fetchall():
                sid = _identifier(row["snapshot_id"], field="snapshot_id", prefix="snap-")
                supersedes = _identifier(
                    row["supersedes_snapshot_id"],
                    field="supersedes_snapshot_id",
                    prefix="snap-",
                )
                if sid != row["snapshot_id"] or supersedes != row["supersedes_snapshot_id"]:
                    raise EvidenceLedgerError("snapshot revision canonicalization check failed")
                expected_revision_pairs.add((sid, supersedes))
            if revision_pairs != expected_revision_pairs:
                raise EvidenceLedgerError("revision coverage integrity check failed")
        except EvidenceLedgerError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeError, OverflowError) as exc:
            raise EvidenceLedgerError("global ledger integrity check failed") from exc

    def health(self) -> Dict[str, Any]:
        count_tables = (
            ("snapshots", self._SNAPSHOTS),
            ("spans", self._SPANS),
            ("turns", self._TURNS),
            ("turn_attestations", self._TURN_ATTESTATIONS),
            ("opened_spans", self._OPENED),
            ("outputs", self._OUTPUTS),
            ("claims", self._CLAIMS),
            ("checker_runs", self._CHECKER_RUNS),
            ("checker_receipts", self._CHECKER_RECEIPTS),
            ("conflicts", self._CONFLICTS),
            ("revisions", self._REVISIONS),
        )
        conn = self._connect()
        counts: Dict[str, Optional[int]] = {name: None for name, _table in count_tables}
        counts["unclaimed_checker_runs"] = None
        journal_mode = "unknown"
        append_only = False
        integrity_valid = False
        try:
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            conn.execute("BEGIN")
            integrity_valid = True
            snapshot_ids = [
                str(row[0])
                for row in conn.execute(
                    f"SELECT snapshot_id FROM {self._SNAPSHOTS} ORDER BY snapshot_id"
                ).fetchall()
            ]
            for snapshot_id in snapshot_ids:
                try:
                    if self._validated_snapshot_row(conn, snapshot_id) is None:
                        integrity_valid = False
                except EvidenceLedgerError:
                    integrity_valid = False
            turn_ids = [
                str(row[0])
                for row in conn.execute(
                    f"SELECT turn_id FROM {self._TURNS} ORDER BY turn_id"
                ).fetchall()
            ]
            for turn_id in turn_ids:
                try:
                    self._validated_turn_state(conn, turn_id)
                except EvidenceLedgerError:
                    integrity_valid = False
            try:
                self._validate_global_rows(conn)
                if str(conn.execute("PRAGMA integrity_check").fetchone()[0]).lower() != "ok":
                    integrity_valid = False
            except (EvidenceLedgerError, sqlite3.DatabaseError, TypeError, ValueError):
                integrity_valid = False
            for name, table in count_tables:
                counts[name] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            counts["unclaimed_checker_runs"] = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM {self._CHECKER_RUNS} AS runs
                    LEFT JOIN {self._CHECKER_RECEIPTS} AS receipts
                      ON receipts.run_nonce_sha256 = runs.run_nonce_sha256
                    WHERE receipts.run_nonce_sha256 IS NULL
                    """
                ).fetchone()[0]
            )
            try:
                self._validate_append_only_contract(conn)
                append_only = True
            except EvidenceLedgerError:
                append_only = False
            conn.commit()
        except (
            EvidenceLedgerError,
            sqlite3.DatabaseError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            UnicodeError,
            OverflowError,
        ):
            try:
                conn.rollback()
            except sqlite3.DatabaseError:
                pass
            integrity_valid = False
            append_only = False
            for name, table in count_tables:
                try:
                    counts[name] = int(
                        conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    )
                except (sqlite3.DatabaseError, TypeError, ValueError):
                    counts[name] = None
            try:
                counts["unclaimed_checker_runs"] = int(
                    conn.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {self._CHECKER_RUNS} AS runs
                        LEFT JOIN {self._CHECKER_RECEIPTS} AS receipts
                          ON receipts.run_nonce_sha256 = runs.run_nonce_sha256
                        WHERE receipts.run_nonce_sha256 IS NULL
                        """
                    ).fetchone()[0]
                )
            except (sqlite3.DatabaseError, TypeError, ValueError):
                counts["unclaimed_checker_runs"] = None
        finally:
            conn.close()
        return {
            "status": "ok" if append_only and integrity_valid else "degraded",
            "schema_version": LEDGER_SCHEMA_VERSION,
            "journal_mode": journal_mode,
            "append_only": append_only,
            "integrity_valid": integrity_valid,
            "stores_caller_evidence": False,
            "authority_granted": False,
            "counts": counts,
        }

    def __len__(self) -> int:
        conn = self._connect()
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {self._SNAPSHOTS}").fetchone()[0])
        finally:
            conn.close()


__all__ = [
    "EvidenceLedgerError",
    "EvidenceLedgerFreshnessError",
    "EvidenceLedgerValidationError",
    "LEDGER_SCHEMA_VERSION",
    "MAX_CONTENT_CHARS",
    "PROVENANCE_RELATIONS",
    "SNAPSHOT_SCHEMA_VERSION",
    "SQLiteEvidenceLedger",
    "canonical_uri",
    "untrusted_ephemeral_evidence",
]
