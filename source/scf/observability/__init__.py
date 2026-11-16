"""Telemetry, checkpoint integrity, and isolation receipts for SCF."""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


SCF_RECEIPT_SCHEMA = "supermix-scf91-execution-receipt-v1"

__all__ = [
    "SCF_RECEIPT_SCHEMA",
    "StageMetric",
    "SCFExecutionReceipt",
    "SCFObservabilityHub",
]


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class StageMetric:
    name: str
    elapsed_ms: float
    status: str = "ok"
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SCFExecutionReceipt:
    """Audit-ready provenance receipt for an SCF cognitive cycle."""

    schema_version: str = SCF_RECEIPT_SCHEMA
    task_id: str = ""
    fallback_to_legacy: bool = False
    selected_family: str = ""
    decoding_policy: Dict[str, Any] = field(default_factory=dict)
    flags_snapshot: Dict[str, bool] = field(default_factory=dict)
    experimental_components_used: List[str] = field(default_factory=list)
    production_ready_claim: bool = False
    verification_passed: Optional[bool] = None
    total_latency_ms: float = 0.0
    peak_traced_kb: float = 0.0
    stages: List[StageMetric] = field(default_factory=list)
    evidence_links: List[str] = field(default_factory=list)
    receipt_sha256: str = ""

    def finalize(self) -> "SCFExecutionReceipt":
        # Never claim experimental components are production-ready (Rule 6)
        if self.experimental_components_used:
            self.production_ready_claim = False
        body = asdict(self)
        body.pop("receipt_sha256", None)
        self.receipt_sha256 = _canonical_sha256(body)
        return self

    def to_dict(self) -> Dict[str, Any]:
        if not self.receipt_sha256:
            self.finalize()
        return asdict(self)


class SCFObservabilityHub:
    """Tracks execution latency, peak Python allocations, and checkpoint hashes."""

    def __init__(self) -> None:
        self.history: List[Dict[str, Any]] = []

    @staticmethod
    def verify_checkpoint_hash(path: Path, expected_sha256: str) -> Dict[str, Any]:
        resolved = Path(path)
        if not resolved.is_file():
            return {"path": str(path), "exists": False, "verified": False, "sha256": None}
        hasher = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        actual = hasher.hexdigest()
        return {
            "path": str(resolved),
            "exists": True,
            "sha256": actual,
            "expected_sha256": expected_sha256,
            "verified": actual == expected_sha256,
        }

    def measure_stage(self, name: str, fn, *args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return result, StageMetric(name=name, elapsed_ms=elapsed_ms)

    def start_memory_probe(self) -> bool:
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        return already_tracing

    @staticmethod
    def finish_memory_probe(was_already_tracing: bool) -> float:
        _, peak = tracemalloc.get_traced_memory()
        if not was_already_tracing:
            tracemalloc.stop()
        return round(float(peak) / 1024.0, 2)

    def record(self, receipt: SCFExecutionReceipt) -> Dict[str, Any]:
        payload = receipt.finalize().to_dict()
        self.history.append(payload)
        return payload
