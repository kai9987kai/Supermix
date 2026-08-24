"""Build integrity-bound Cognitive Leap model-store ZIPs.

The v50 package is a legacy checkpoint archive, v51 is the canonical bounded
runtime demo, and v51.1 is an explicitly unpromoted research candidate.  None
of these packages authorizes automatic routing or a broad chat/reasoning claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable

from materialize_v51_chat_demo import build_demo_metadata


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
FIXED_ZIP_TIME = (2026, 8, 12, 0, 0, 0)
BOUNDED_EVALUATION_SCHEMA = "supermix-cognitive-leap-bounded-evaluation-v2"
THREE_WAY_EVALUATION_SCHEMA = "supermix-cognitive-leap-three-way-evaluation-v1"
THREE_WAY_PREDICTION_SCHEMA = (
    "supermix-cognitive-leap-three-way-logits-jsonl-v1"
)
THREE_WAY_EVALUATION_PROFILE_SHA256 = (
    "3a018d1b9cde5d59c0431f0323a46993d71806604753e459200649a024332bbd"
)
ARTIFACT_MANIFEST_V2_SCHEMA = "supermix-model-store-artifact-manifest-v2"
CONTENT_BOUND_STATUS = "content_bound_not_authenticated"
BOUNDED_EVALUATION_AUTHORITY = {
    "activation": False,
    "auto_route": False,
    "default_model": False,
    "fallback": False,
    "consultant": False,
    "tools": False,
    "permissions": False,
    "safety": False,
    "promotion": False,
    "store_publication": False,
    "release": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not allowed: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read strict JSON from {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _safe_relative_member(raw_path: Any, *, label: str) -> str:
    """Return a portable ZIP member name for a repository-relative path."""

    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ValueError(f"{label} must be a non-empty relative path")
    windows_path = PureWindowsPath(raw_path)
    normalized = raw_path.replace("\\", "/")
    parts = normalized.split("/")
    if (
        windows_path.is_absolute()
        or bool(windows_path.drive)
        or PurePosixPath(normalized).is_absolute()
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise ValueError(f"{label} is not a safe repository-relative path: {raw_path!r}")
    return "/".join(parts)


def _resolve_repository_path(root: Path, raw_path: Any, *, label: str) -> tuple[str, Path]:
    member = _safe_relative_member(raw_path, label=label)
    resolved_root = root.resolve()
    resolved = resolved_root.joinpath(*member.split("/")).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} escapes the package root")
    return member, resolved


def _file_binding_from_reference(
    record: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, int]:
    """Read a byte-level file binding, preferring protocol ``file_sha256``."""

    hash_field = "file_sha256" if "file_sha256" in record else "sha256"
    expected_hash = record.get(hash_field)
    expected_size = record.get("size_bytes")
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or expected_hash != expected_hash.lower()
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"{label} {hash_field} is invalid")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError(f"{label} size_bytes is invalid")
    return expected_hash, expected_size


def _closure_manifest_rows(
    closure: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "archive_member": member,
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "reference_sites": sorted(record["reference_sites"]),
        }
        for member, record in sorted(closure.items())
    ]


def _collect_reproducibility_closure(
    receipt_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Resolve every byte-bound file reachable from a bounded-v2 receipt.

    Referenced JSON files are traversed recursively.  This deliberately does
    not infer unbound paths: every archived input must carry an exact byte hash
    and size in the signed-by-content evidence graph.
    """

    resolved_root = root.resolve()
    resolved_receipt = receipt_path.resolve()
    if not resolved_receipt.is_relative_to(resolved_root) or not resolved_receipt.is_file():
        raise ValueError("The bounded receipt must be a file beneath the package root")
    receipt_member = resolved_receipt.relative_to(resolved_root).as_posix()
    receipt_member = _safe_relative_member(receipt_member, label="bounded receipt path")
    receipt = _load_strict_json(resolved_receipt)
    closure: dict[str, dict[str, Any]] = {
        receipt_member: {
            "path": resolved_receipt,
            "sha256": sha256_file(resolved_receipt),
            "size_bytes": resolved_receipt.stat().st_size,
            "reference_sites": {"entry_receipt"},
        }
    }
    casefold_members = {receipt_member.casefold(): receipt_member}
    json_queue: list[tuple[str, Path]] = []
    parsed_json: set[Path] = set()

    def add_reference(record: Mapping[str, Any], *, site: str) -> None:
        member, path = _resolve_repository_path(
            resolved_root,
            record.get("path"),
            label=f"{site}.path",
        )
        expected_hash, expected_size = _file_binding_from_reference(record, label=site)
        if not path.is_file():
            raise ValueError(f"{site} is missing: {member}")
        if path.stat().st_size != expected_size or sha256_file(path) != expected_hash:
            raise ValueError(f"{site} content does not match its binding")
        folded = member.casefold()
        existing_name = casefold_members.get(folded)
        if existing_name is not None and existing_name != member:
            raise ValueError(
                "Case-insensitive reproducibility path collision: "
                f"{existing_name!r} and {member!r}"
            )
        casefold_members[folded] = member
        existing = closure.get(member)
        if existing is None:
            closure[member] = {
                "path": path,
                "sha256": expected_hash,
                "size_bytes": expected_size,
                "reference_sites": {site},
            }
        else:
            if (
                existing["path"] != path
                or existing["sha256"] != expected_hash
                or existing["size_bytes"] != expected_size
            ):
                raise ValueError(f"Conflicting bindings for reproducibility member {member}")
            existing["reference_sites"].add(site)
        if member.lower().endswith(".json"):
            json_queue.append((member, path))

    def visit(value: Any, *, site: str) -> None:
        if isinstance(value, Mapping):
            if "path" in value and ("sha256" in value or "file_sha256" in value):
                add_reference(value, site=site)
            for key, child in value.items():
                visit(child, site=f"{site}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, site=f"{site}[{index}]")

    visit(receipt, site="receipt")
    while json_queue:
        member, path = json_queue.pop(0)
        resolved = path.resolve()
        if resolved in parsed_json:
            continue
        parsed_json.add(resolved)
        payload = _load_strict_json(path)
        visit(payload, site=f"file:{member}")
    return receipt, closure


def _validate_receipt_semantics(receipt_path: Path, root: Path) -> Mapping[str, Any]:
    """Run the independent evaluator receipt verifier.

    The import is intentionally local: legacy v50/v51/v51.1 packaging must not
    depend on the v2 verifier unless the caller explicitly selects the v2 path.
    """

    from cognitive_leap_receipt import validate_receipt  # noqa: PLC0415

    result = validate_receipt(receipt_path, root=root)
    if not isinstance(result, Mapping):
        raise ValueError("Receipt validator returned a non-object result")
    return result


def _validate_three_way_receipt_semantics(
    receipt_path: Path,
    root: Path,
) -> Mapping[str, Any]:
    """Run the profiled three-way verifier with checkpoint replay enabled.

    ``verify_inference`` is deliberately omitted from the call so the
    validator's production-safe default (``True``) is the only behavior used
    by this packaging path.
    """

    from cognitive_leap_three_way_receipt import (  # noqa: PLC0415
        CANONICAL_EVALUATION_PROFILE_SHA256,
        validate_receipt,
    )

    if CANONICAL_EVALUATION_PROFILE_SHA256 != THREE_WAY_EVALUATION_PROFILE_SHA256:
        raise ValueError("Three-way validator profile hash does not match the packager")
    result = validate_receipt(receipt_path, root=root)
    if not isinstance(result, Mapping):
        raise ValueError("Three-way receipt validator returned a non-object result")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Receipt {label} must be an object")
    return value


def _resolve_bound_artifact(
    root: Path,
    record: Mapping[str, Any],
    *,
    label: str,
) -> Path:
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    expected_size = record.get("size_bytes")
    _member, resolved = _resolve_repository_path(
        root,
        raw_path,
        label=f"Receipt {label} path",
    )
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        raise ValueError(f"Receipt {label} SHA-256 is invalid")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise ValueError(f"Receipt {label} size is invalid")

    if not resolved.is_file():
        raise ValueError(f"Receipt {label} is missing: {raw_path}")
    if resolved.stat().st_size != expected_size or sha256_file(resolved) != expected_hash:
        raise ValueError(f"Receipt {label} content does not match its binding")
    return resolved


def _receipt_v2_evidence(
    receipt_path: Path,
    *,
    root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[Path, Mapping[str, Any]]],
    Mapping[str, Any],
]:
    resolved_root = root.resolve()
    resolved_receipt = receipt_path.resolve()
    if not resolved_receipt.is_relative_to(resolved_root) or not resolved_receipt.is_file():
        raise ValueError("The bounded receipt must be a file beneath the package root")

    receipt_hash_before_validation = sha256_file(resolved_receipt)
    receipt_size_before_validation = resolved_receipt.stat().st_size
    validation = _validate_receipt_semantics(resolved_receipt, resolved_root)
    if (
        validation.get("valid") is not True
        or validation.get("gate_outcome") != "pass"
    ):
        raise ValueError("Bounded evaluation receipt failed semantic validation")
    if (
        resolved_receipt.stat().st_size != receipt_size_before_validation
        or sha256_file(resolved_receipt) != receipt_hash_before_validation
    ):
        raise ValueError("Bounded evaluation receipt changed during semantic validation")

    receipt = _load_strict_json(resolved_receipt)
    if receipt.get("schema") != BOUNDED_EVALUATION_SCHEMA:
        raise ValueError("Only bounded-evaluation-v2 receipts can use v2 packaging")
    if receipt.get("gate_outcome") != "pass":
        raise ValueError("A semantically validated passing bounded gate is required")
    if (
        receipt.get("authentication") != "none"
        or receipt.get("integrity_status") != CONTENT_BOUND_STATUS
        or receipt.get("trusted_timestamp") is not False
    ):
        raise ValueError("Receipt authentication and integrity contract changed")
    if receipt.get("authority") != BOUNDED_EVALUATION_AUTHORITY:
        raise ValueError("Receipt must carry the exact all-false authority contract")

    selection = _require_mapping(receipt.get("selection"), "selection")
    artifacts = _require_mapping(receipt.get("artifacts"), "artifacts")
    candidate = _require_mapping(artifacts.get("candidate"), "candidate artifact")
    lineage = _require_mapping(
        selection.get("lineage_manifest"),
        "lineage manifest",
    )
    lineage_verification = _require_mapping(
        selection.get("lineage_verification"),
        "lineage verification",
    )
    predictions = _require_mapping(
        receipt.get("per_example_artifact"),
        "per-example artifact",
    )
    expected_semantic_bindings = {
        "receipt_id": receipt.get("receipt_id"),
        "receipt_file_sha256": receipt_hash_before_validation,
        "candidate_sha256": candidate.get("sha256"),
        "lineage_sha256": lineage.get("sha256"),
        "lineage_verification_sha256": lineage_verification.get("sha256"),
        "per_example_sha256": predictions.get("sha256"),
        "per_example_uncompressed_sha256": predictions.get("uncompressed_sha256"),
    }
    for field, expected in expected_semantic_bindings.items():
        if validation.get(field) != expected:
            raise ValueError(f"Receipt validator {field} cross-link mismatch")

    evidence = {
        "candidate": (
            _resolve_bound_artifact(resolved_root, candidate, label="candidate artifact"),
            candidate,
        ),
        "lineage": (
            _resolve_bound_artifact(resolved_root, lineage, label="lineage manifest"),
            lineage,
        ),
        "lineage_verification": (
            _resolve_bound_artifact(
                resolved_root,
                lineage_verification,
                label="lineage verification",
            ),
            lineage_verification,
        ),
        "predictions": (
            _resolve_bound_artifact(
                resolved_root,
                predictions,
                label="per-example artifact",
            ),
            predictions,
        ),
        "receipt": (
            resolved_receipt,
            {
                "path": resolved_receipt.relative_to(resolved_root).as_posix(),
                "sha256": sha256_file(resolved_receipt),
                "size_bytes": resolved_receipt.stat().st_size,
                "schema": receipt["schema"],
                "receipt_id": receipt.get("receipt_id"),
            },
        ),
    }
    return receipt, evidence, validation


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


def build_zip(
    output_path: Path,
    *,
    files: Iterable[tuple[str, Path]],
    generated: Iterable[tuple[str, bytes]],
    package_contract: dict[str, Any],
) -> dict[str, Any]:
    file_rows: list[dict[str, Any]] = []
    materialized: list[tuple[str, Path | None, bytes | None]] = []
    seen: set[str] = set()
    for name, path in files:
        if _safe_relative_member(name, label="ZIP member") != name:
            raise ValueError(f"ZIP member must use canonical forward slashes: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate ZIP member: {name}")
        seen.add(name.casefold())
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        file_rows.append(
            {
                "name": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        materialized.append((name, path, None))
    for name, data in generated:
        if _safe_relative_member(name, label="ZIP member") != name:
            raise ValueError(f"ZIP member must use canonical forward slashes: {name}")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate ZIP member: {name}")
        seen.add(name.casefold())
        file_rows.append(
            {
                "name": name,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        materialized.append((name, None, data))

    if "artifact_manifest.json".casefold() in seen:
        raise ValueError("artifact_manifest.json is reserved for the package manifest")
    artifact_manifest = canonical_json(
        {
            "schema": "supermix-model-store-artifact-manifest-v1",
            **package_contract,
            "members": sorted(file_rows, key=lambda row: row["name"]),
        }
    )
    materialized.append(("artifact_manifest.json", None, artifact_manifest))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    with zipfile.ZipFile(temp_path, "w", allowZip64=True) as archive:
        for name, path, data in sorted(materialized, key=lambda row: row[0]):
            info = _zip_info(name)
            with archive.open(info, "w") as target:
                if path is not None:
                    with path.open("rb") as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                else:
                    target.write(data or b"")
    temp_path.replace(output_path)
    with zipfile.ZipFile(output_path, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed for {bad}")
    return {
        "file_name": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "size_mb": round(output_path.stat().st_size / (1024 * 1024), 3),
        "sha256": sha256_file(output_path),
        "family": "cognitive_leap",
    }


def model_card(
    *,
    title: str,
    status: str,
    intended_use: str,
    evidence: str,
    limitations: str,
) -> bytes:
    return (
        f"# {title}\n\n"
        f"- Status: **{status}**\n"
        "- Automatic routing/default activation: **not allowed**\n"
        "- General-chat claim: **false**\n"
        "- General-reasoning claim: **false**\n\n"
        f"## Intended use\n\n{intended_use}\n\n"
        f"## Evidence\n\n{evidence}\n\n"
        f"## Limitations\n\n{limitations}\n"
    ).encode("utf-8")


def _manifest_evidence_link(
    archive_member: str,
    record: Mapping[str, Any],
    *,
    extra_keys: Iterable[str] = (),
) -> dict[str, Any]:
    link = {
        "archive_member": archive_member,
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }
    for key in extra_keys:
        if key in record:
            link[key] = record[key]
    return link


def _bounded_v2_chat_metadata(
    *,
    candidate_path: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    candidate_alias: str,
    receipt_alias: str,
) -> bytes:
    """Create deterministic runtime metadata without broad capability claims."""

    metadata = build_demo_metadata(
        weights=candidate_path,
        metrics_path=receipt_path,
    )
    metadata.update(
        {
            "created_by": "package_cognitive_leap_store_models.py",
            "purpose": (
                "manual-only Cognitive Leap v51.2 bounded arithmetic runtime demo"
            ),
            "training_task": (
                "synthetic four-operation chained modular arithmetic modulo 10"
            ),
            "fine_tuned_weights": candidate_alias,
            "checkpoint_path": candidate_alias,
            "benchmark_metrics": receipt_alias,
            "prediction_stability_metrics": "",
            "benchmark_summary": {
                "bounded_evaluation_schema": receipt.get("schema"),
                "bounded_receipt_id": receipt.get("receipt_id"),
                "bounded_gate_outcome": receipt.get("gate_outcome"),
                "claim_scope": receipt.get("claim_scope"),
                "summary": receipt.get("summary", {}),
            },
            "model_store_policy": {
                "manual_selection_only": True,
                "auto_route_allowed": False,
                "default_model_allowed": False,
                "general_chat_claim": False,
                "general_reasoning_claim": False,
                "receipt_grants_activation": False,
                "receipt_grants_store_publication": False,
            },
            "authentication": "none",
            "integrity_status": CONTENT_BOUND_STATUS,
            "authority": dict(BOUNDED_EVALUATION_AUTHORITY),
        }
    )
    return canonical_json(metadata)


def _three_way_v1_chat_metadata(
    *,
    candidate_path: Path,
    receipt_path: Path,
    receipt: Mapping[str, Any],
    candidate_alias: str,
    receipt_alias: str,
) -> bytes:
    """Create deterministic manual-only metadata for a profiled candidate."""

    comparisons = _require_mapping(receipt.get("comparisons"), "comparisons")
    release = _require_mapping(
        comparisons.get("release_continuity"),
        "release continuity comparison",
    )
    prior = _require_mapping(
        comparisons.get("prior_candidate_superiority"),
        "prior-candidate superiority comparison",
    )
    metadata = build_demo_metadata(
        weights=candidate_path,
        metrics_path=receipt_path,
    )
    metadata.update(
        {
            "created_by": "package_cognitive_leap_store_models.py",
            "purpose": (
                "manual-only Cognitive Leap v51.2 profiled three-way arithmetic demo"
            ),
            "training_task": (
                "synthetic four-operation chained modular arithmetic modulo 10"
            ),
            "fine_tuned_weights": candidate_alias,
            "checkpoint_path": candidate_alias,
            "benchmark_metrics": receipt_alias,
            "prediction_stability_metrics": "",
            "benchmark_summary": {
                "three_way_evaluation_schema": receipt.get("schema"),
                "three_way_receipt_id": receipt.get("receipt_id"),
                "evaluation_profile_sha256": receipt.get(
                    "evaluation_profile_sha256"
                ),
                "gate_outcome": receipt.get("gate_outcome"),
                "release_continuity_passed": release.get("passed"),
                "prior_candidate_superiority_passed": prior.get("passed"),
                "release_continuity": release.get("summary", {}),
                "prior_candidate_superiority": prior.get("summary", {}),
            },
            "model_store_policy": {
                "manual_selection_only": True,
                "auto_route_allowed": False,
                "default_model_allowed": False,
                "general_chat_claim": False,
                "general_reasoning_claim": False,
                "receipt_grants_activation": False,
                "receipt_grants_store_publication": False,
            },
            "authentication": "none",
            "integrity_status": CONTENT_BOUND_STATUS,
            "authority": dict(BOUNDED_EVALUATION_AUTHORITY),
        }
    )
    return canonical_json(metadata)


def _extract_archive_safely(archive: zipfile.ZipFile, destination: Path) -> None:
    seen: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if _safe_relative_member(name, label="archive member") != name:
            raise ValueError(f"Archive contains a noncanonical member: {name!r}")
        if name.casefold() in seen:
            raise ValueError(f"Archive contains a duplicate member: {name}")
        seen.add(name.casefold())
        if info.is_dir():
            raise ValueError(f"Bounded-v2 archives may not contain directory entries: {name}")
        target = destination.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info, "r") as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def _verify_content_bound_archive(
    output_path: Path,
    expected_links: Mapping[str, Mapping[str, Any]],
    expected_closure: list[dict[str, Any]],
    expected_validation: Mapping[str, Any],
    *,
    receipt_archive_member: str,
    semantic_validator: Callable[[Path, Path], Mapping[str, Any]],
    semantic_fields: Iterable[str],
    archive_label: str,
) -> None:
    try:
        with zipfile.ZipFile(output_path, "r") as archive:
            manifest_value = json.loads(
                archive.read("artifact_manifest.json"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
            manifest = _require_mapping(manifest_value, "artifact manifest")
            if manifest.get("schema") != ARTIFACT_MANIFEST_V2_SCHEMA:
                raise ValueError("Built package is missing artifact-manifest-v2")
            if manifest.get("evidence_links") != expected_links:
                raise ValueError("Built package evidence cross-links changed")
            if manifest.get("reproducibility_closure") != expected_closure:
                raise ValueError("Built package reproducibility closure changed")
            expected_closure_sha256 = hashlib.sha256(
                canonical_json(expected_closure)
            ).hexdigest()
            if manifest.get("reproducibility_closure_sha256") != expected_closure_sha256:
                raise ValueError("Built package reproducibility closure digest changed")
            raw_member_rows = manifest.get("members")
            if not isinstance(raw_member_rows, list):
                raise ValueError("Artifact manifest members must be a list")
            member_rows: dict[str, Mapping[str, Any]] = {}
            for index, row in enumerate(raw_member_rows):
                row_value = _require_mapping(row, f"artifact manifest members[{index}]")
                name = row_value.get("name")
                if not isinstance(name, str) or name in member_rows:
                    raise ValueError("Artifact manifest member names must be unique strings")
                member_rows[name] = row_value
            archive_names = [info.filename for info in archive.infolist()]
            if len({name.casefold() for name in archive_names}) != len(archive_names):
                raise ValueError("Built package contains duplicate member names")
            for name in archive_names:
                if _safe_relative_member(name, label="archive member") != name:
                    raise ValueError(f"Built package member path is unsafe: {name}")
            expected_member_names = set(archive_names) - {"artifact_manifest.json"}
            if set(member_rows) != expected_member_names:
                raise ValueError("Artifact manifest does not enumerate every package member")
            for name, member_row in member_rows.items():
                payload = archive.read(name)
                if (
                    member_row.get("size_bytes") != len(payload)
                    or member_row.get("sha256")
                    != hashlib.sha256(payload).hexdigest()
                ):
                    raise ValueError(f"Built package member binding is invalid: {name}")
            for label, link in expected_links.items():
                archive_member = str(link["archive_member"])
                try:
                    payload = archive.read(archive_member)
                except KeyError as exc:
                    raise ValueError(
                        f"Built package is missing the {label} evidence member"
                    ) from exc
                member_row = _require_mapping(
                    member_rows.get(archive_member),
                    f"artifact manifest {label} member",
                )
                actual_hash = hashlib.sha256(payload).hexdigest()
                if (
                    len(payload) != int(link["size_bytes"])
                    or actual_hash != link["sha256"]
                    or member_row.get("size_bytes") != link["size_bytes"]
                    or member_row.get("sha256") != link["sha256"]
                ):
                    raise ValueError(f"Built package {label} cross-link is invalid")
            for row in expected_closure:
                name = str(row["archive_member"])
                member_row = _require_mapping(
                    member_rows.get(name),
                    f"artifact manifest closure member {name}",
                )
                if (
                    member_row.get("size_bytes") != row["size_bytes"]
                    or member_row.get("sha256") != row["sha256"]
                ):
                    raise ValueError(f"Built package closure cross-link is invalid: {name}")

            with tempfile.TemporaryDirectory(
                prefix=f"supermix-{archive_label}-package-"
            ) as temp_name:
                extracted_root = Path(temp_name)
                _extract_archive_safely(archive, extracted_root)
                receipt_path = extracted_root.joinpath(
                    *receipt_archive_member.split("/")
                )
                validation = semantic_validator(receipt_path, extracted_root)
                if (
                    validation.get("valid") is not True
                    or validation.get("gate_outcome") != "pass"
                ):
                    raise ValueError(
                        f"Extracted {archive_label} receipt failed independent semantic "
                        "validation"
                    )
                for field in semantic_fields:
                    if validation.get(field) != expected_validation.get(field):
                        raise ValueError(
                            f"Extracted {archive_label} receipt {field} cross-link mismatch"
                        )
                _receipt, extracted_closure = _collect_reproducibility_closure(
                    receipt_path,
                    root=extracted_root,
                )
                expected_reachable = {
                    row["archive_member"]: (row["sha256"], row["size_bytes"])
                    for row in expected_closure
                    if any(
                        site != "entry_receipt"
                        for site in row.get("reference_sites", [])
                    )
                }
                extracted_reachable = {
                    member: (record["sha256"], record["size_bytes"])
                    for member, record in extracted_closure.items()
                    if any(
                        site != "entry_receipt"
                        for site in record["reference_sites"]
                    )
                }
                if extracted_reachable != expected_reachable:
                    raise ValueError(
                        "Extracted package does not reproduce the complete evidence closure"
                    )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not verify the built {archive_label} package") from exc


_BOUNDED_V2_SEMANTIC_FIELDS = (
    "receipt_id",
    "receipt_file_sha256",
    "candidate_sha256",
    "lineage_sha256",
    "lineage_verification_sha256",
    "per_example_sha256",
    "per_example_uncompressed_sha256",
)


def _verify_bounded_v2_archive(
    output_path: Path,
    expected_links: Mapping[str, Mapping[str, Any]],
    expected_closure: list[dict[str, Any]],
    expected_validation: Mapping[str, Any],
) -> None:
    _verify_content_bound_archive(
        output_path,
        expected_links,
        expected_closure,
        expected_validation,
        receipt_archive_member="bounded_evaluation_receipt.json",
        semantic_validator=_validate_receipt_semantics,
        semantic_fields=_BOUNDED_V2_SEMANTIC_FIELDS,
        archive_label="bounded-v2",
    )


def build_bounded_v2_package(
    output_path: Path,
    *,
    receipt_path: Path,
    root: Path = PROJECT_ROOT,
    model_key: str = "cognitive_leap_ultra_v51_2",
    allow_unsafe_unprofiled_v2_for_tests: bool = False,
) -> dict[str, Any]:
    """Build the obsolete unprofiled v2 package only for closure tests.

    Bounded-v2 predates the immutable normative evaluation profile and is not
    eligible for production packaging.  The loudly named override exists only
    to retain regression coverage for the generic transitive-closure code.
    """

    if allow_unsafe_unprofiled_v2_for_tests is not True:
        raise ValueError(
            "Unprofiled bounded-v2 production packaging is disabled; use the "
            "profiled three-way-v1 package path"
        )

    if (
        not isinstance(model_key, str)
        or not model_key
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in model_key)
    ):
        raise ValueError("model_key must contain only lowercase letters, digits, and underscores")

    receipt, evidence, validation = _receipt_v2_evidence(
        Path(receipt_path),
        root=Path(root),
    )
    closure_receipt, closure = _collect_reproducibility_closure(
        Path(receipt_path),
        root=Path(root),
    )
    if closure_receipt != receipt:
        raise ValueError("Bounded receipt changed while collecting package closure")
    candidate_path, candidate_record = evidence["candidate"]
    _lineage_path, lineage_record = evidence["lineage"]
    _lineage_verification_path, lineage_verification_record = evidence[
        "lineage_verification"
    ]
    _prediction_path, prediction_record = evidence["predictions"]
    resolved_receipt, receipt_record = evidence["receipt"]

    if lineage_record.get("schema") != "supermix-cognitive-leap-lineage-v2":
        raise ValueError("Receipt does not bind a Cognitive Leap lineage-v2 manifest")
    if (
        lineage_verification_record.get("schema")
        != "supermix-cognitive-leap-lineage-verification-v1"
    ):
        raise ValueError("Receipt does not bind a Cognitive Leap lineage verification")
    if (
        prediction_record.get("schema")
        != "supermix-cognitive-leap-paired-logits-jsonl-v1"
    ):
        raise ValueError("Receipt does not bind the paired-logits prediction schema")
    candidate_state_hash = candidate_record.get("canonical_state_sha256")
    if (
        not isinstance(candidate_state_hash, str)
        or len(candidate_state_hash) != 64
        or any(character not in "0123456789abcdef" for character in candidate_state_hash)
    ):
        raise ValueError("Candidate canonical state hash is missing or invalid")

    protocol_reference = _require_mapping(receipt.get("protocol"), "protocol reference")
    selection_reference = _require_mapping(receipt.get("selection"), "selection reference")
    artifacts = _require_mapping(receipt.get("artifacts"), "artifacts")
    baseline_record = _require_mapping(artifacts.get("baseline"), "baseline artifact")

    def referenced_member(record: Mapping[str, Any], label: str) -> str:
        member = _safe_relative_member(record.get("path"), label=f"{label} path")
        closure_record = closure.get(member)
        if closure_record is None:
            raise ValueError(f"Reproducibility closure is missing {label}: {member}")
        expected_hash, expected_size = _file_binding_from_reference(record, label=label)
        if (
            closure_record["sha256"] != expected_hash
            or closure_record["size_bytes"] != expected_size
        ):
            raise ValueError(f"Reproducibility closure binding changed for {label}")
        return member

    exact_members = {
        "protocol": referenced_member(protocol_reference, "protocol"),
        "selection": referenced_member(selection_reference, "selection"),
        "baseline": referenced_member(baseline_record, "baseline artifact"),
        "candidate": referenced_member(candidate_record, "candidate artifact"),
        "lineage": referenced_member(lineage_record, "lineage manifest"),
        "lineage_verification": referenced_member(
            lineage_verification_record,
            "lineage verification",
        ),
        "predictions": referenced_member(prediction_record, "per-example predictions"),
    }
    receipt_member = resolved_receipt.relative_to(Path(root).resolve()).as_posix()
    receipt_member = _safe_relative_member(receipt_member, label="receipt path")
    if receipt_member not in closure:
        raise ValueError("Reproducibility closure is missing its entry receipt")

    aliases = {
        "candidate": "cognitive_leap_ultra_v51_2.pth",
        "receipt": "bounded_evaluation_receipt.json",
        "chat_meta": "chat_demo_meta_v51_2.json",
    }
    chat_metadata = _bounded_v2_chat_metadata(
        candidate_path=candidate_path,
        receipt_path=resolved_receipt,
        receipt=receipt,
        candidate_alias=aliases["candidate"],
        receipt_alias=aliases["receipt"],
    )

    def closure_link(
        archive_member: str,
        source_record: Mapping[str, Any],
        *,
        extra_keys: Iterable[str] = (),
    ) -> dict[str, Any]:
        closure_record = closure[referenced_member(source_record, archive_member)]
        link = {
            "archive_member": archive_member,
            "sha256": closure_record["sha256"],
            "size_bytes": closure_record["size_bytes"],
        }
        for key in extra_keys:
            if key in source_record:
                link[key] = source_record[key]
        return link

    evidence_links = {
        "receipt": _manifest_evidence_link(
            aliases["receipt"],
            receipt_record,
            extra_keys=("schema", "receipt_id"),
        ),
        "protocol": closure_link(
            exact_members["protocol"],
            protocol_reference,
        ),
        "selection": closure_link(
            exact_members["selection"],
            selection_reference,
        ),
        "baseline": closure_link(
            exact_members["baseline"],
            baseline_record,
            extra_keys=(
                "canonical_state_sha256",
                "tensor_count",
                "element_count",
            ),
        ),
        "lineage": closure_link(
            exact_members["lineage"],
            lineage_record,
            extra_keys=("schema",),
        ),
        "lineage_verification": closure_link(
            exact_members["lineage_verification"],
            lineage_verification_record,
            extra_keys=("schema",),
        ),
        "predictions": closure_link(
            exact_members["predictions"],
            prediction_record,
            extra_keys=(
                "schema",
                "uncompressed_sha256",
                "row_count",
                "dataset_id",
            ),
        ),
        "candidate": _manifest_evidence_link(
            aliases["candidate"],
            candidate_record,
            extra_keys=(
                "canonical_state_sha256",
                "tensor_count",
                "element_count",
            ),
        ),
        "chat_meta": {
            "archive_member": aliases["chat_meta"],
            "sha256": hashlib.sha256(chat_metadata).hexdigest(),
            "size_bytes": len(chat_metadata),
            "checkpoint_archive_member": aliases["candidate"],
            "receipt_archive_member": aliases["receipt"],
            "manual_selection_only": True,
        },
    }

    closure_rows = _closure_manifest_rows(closure)
    closure_sha256 = hashlib.sha256(canonical_json(closure_rows)).hexdigest()
    package_files: list[tuple[str, Path]] = [
        (member, record["path"])
        for member, record in sorted(closure.items())
    ]
    existing_names = {member.casefold(): member for member, _path in package_files}

    def add_alias(alias: str, path: Path, exact_member: str) -> None:
        existing_name = existing_names.get(alias.casefold())
        if existing_name is not None:
            if existing_name != alias or exact_member != alias:
                raise ValueError(
                    f"Runtime alias collides with reproducibility member: {alias}"
                )
            return
        existing_names[alias.casefold()] = alias
        package_files.append((alias, path))

    add_alias(aliases["candidate"], candidate_path, exact_members["candidate"])
    add_alias(aliases["receipt"], resolved_receipt, receipt_member)

    result = build_zip(
        Path(output_path),
        files=package_files,
        generated=(
            (aliases["chat_meta"], chat_metadata),
            (
                "MODEL_CARD.md",
                model_card(
                    title="Cognitive Leap Ultra v51.2 bounded evaluation artifact",
                    status="bounded evaluation passed; manually selectable only",
                    intended_use=(
                        "Reproducing the content-bound synthetic arithmetic evaluation "
                        "and reviewing its checkpoint lineage and paired logits."
                    ),
                    evidence=(
                        "The independent semantic validator accepted the bundled v2 "
                        "receipt after clean temporary extraction. The complete protocol, "
                        "selection, source snapshots, baseline, continuation members, "
                        "training receipts, lineage, paired logits, and candidate are "
                        "cross-linked by SHA-256 in artifact_manifest.json."
                    ),
                    limitations=(
                        "The receipt is unauthenticated bounded evidence and grants no "
                        "publication, activation, Auto-routing, default-model, safety, "
                        "tool, or permission authority. It is not a general-chat or "
                        "general-reasoning claim."
                    ),
                ),
            ),
        ),
        package_contract={
            "schema": ARTIFACT_MANIFEST_V2_SCHEMA,
            "model_key": model_key,
            "status": "bounded_evaluation_pass_manual_only",
            "manual_selectable": True,
            "manual_selection_only": True,
            "auto_route_allowed": False,
            "default_model_allowed": False,
            "receipt_grants_activation": False,
            "receipt_grants_store_publication": False,
            "authentication": "none",
            "integrity_status": CONTENT_BOUND_STATUS,
            "runtime_status": CONTENT_BOUND_STATUS,
            "receipt_authority": dict(BOUNDED_EVALUATION_AUTHORITY),
            "gate_outcome": receipt["gate_outcome"],
            "reproducibility_closure_schema": (
                "supermix-cognitive-leap-reproducibility-closure-v1"
            ),
            "reproducibility_closure": closure_rows,
            "reproducibility_closure_sha256": closure_sha256,
            "portable_validation": {
                "receipt_archive_member": aliases["receipt"],
                "root": ".",
                "validator": "cognitive_leap_receipt.validate_receipt",
                "verified_after_temporary_extraction": True,
            },
            "runtime_files": {
                "checkpoint": aliases["candidate"],
                "chat_metadata": aliases["chat_meta"],
                "bounded_receipt": aliases["receipt"],
            },
            "evidence_links": evidence_links,
        },
    )
    try:
        _verify_bounded_v2_archive(
            Path(output_path),
            evidence_links,
            closure_rows,
            validation,
        )
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise
    return result


_THREE_WAY_SEMANTIC_FIELDS = (
    "schema",
    "receipt_id",
    "gate_outcome",
    "evaluation_profile_schema",
    "evaluation_profile_sha256",
    "protocol_sha256",
    "selection_sha256",
    "release_baseline_sha256",
    "prior_candidate_sha256",
    "candidate_sha256",
    "per_example_artifact_sha256",
    "release_continuity_passed",
    "prior_candidate_superiority_passed",
    "checkpoint_inference_replayed",
    "authority",
)


def _three_way_semantic_projection(
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [field for field in _THREE_WAY_SEMANTIC_FIELDS if field not in validation]
    if missing:
        raise ValueError(
            "Three-way validator omitted semantic bindings: " + ", ".join(missing)
        )
    return {field: validation[field] for field in _THREE_WAY_SEMANTIC_FIELDS}


def _three_way_receipt_for_packaging(
    receipt_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    resolved_root = root.resolve()
    resolved_receipt = receipt_path.resolve()
    if not resolved_receipt.is_relative_to(resolved_root) or not resolved_receipt.is_file():
        raise ValueError("The three-way receipt must be a file beneath the package root")

    receipt_hash_before = sha256_file(resolved_receipt)
    receipt_size_before = resolved_receipt.stat().st_size
    validation = _validate_three_way_receipt_semantics(
        resolved_receipt,
        resolved_root,
    )
    if (
        validation.get("valid") is not True
        or validation.get("schema") != THREE_WAY_EVALUATION_SCHEMA
        or validation.get("gate_outcome") != "pass"
        or validation.get("evaluation_profile_sha256")
        != THREE_WAY_EVALUATION_PROFILE_SHA256
        or validation.get("release_continuity_passed") is not True
        or validation.get("prior_candidate_superiority_passed") is not True
        or validation.get("checkpoint_inference_replayed") is not True
        or validation.get("authority") != BOUNDED_EVALUATION_AUTHORITY
    ):
        raise ValueError(
            "Three-way receipt requires a profiled passing validator result with "
            "both comparisons and checkpoint inference replay"
        )
    if (
        resolved_receipt.stat().st_size != receipt_size_before
        or sha256_file(resolved_receipt) != receipt_hash_before
    ):
        raise ValueError("Three-way receipt changed during semantic validation")

    receipt = _load_strict_json(resolved_receipt)
    if (
        receipt.get("schema") != THREE_WAY_EVALUATION_SCHEMA
        or receipt.get("gate_outcome") != "pass"
        or receipt.get("evaluation_profile_sha256")
        != THREE_WAY_EVALUATION_PROFILE_SHA256
        or receipt.get("authentication") != "none"
        or receipt.get("integrity_status") != CONTENT_BOUND_STATUS
        or receipt.get("trusted_timestamp") is not False
        or receipt.get("authority") != BOUNDED_EVALUATION_AUTHORITY
    ):
        raise ValueError("Three-way receipt trust, profile, gate, or authority changed")
    comparisons = _require_mapping(receipt.get("comparisons"), "comparisons")
    release = _require_mapping(
        comparisons.get("release_continuity"),
        "release continuity comparison",
    )
    prior = _require_mapping(
        comparisons.get("prior_candidate_superiority"),
        "prior-candidate superiority comparison",
    )
    if release.get("passed") is not True or prior.get("passed") is not True:
        raise ValueError("Both receipt comparisons must pass before packaging")

    protocol = _require_mapping(receipt.get("protocol"), "protocol reference")
    selection = _require_mapping(receipt.get("selection"), "selection reference")
    artifacts = _require_mapping(receipt.get("artifacts"), "artifacts")
    release_artifact = _require_mapping(
        artifacts.get("release_baseline"),
        "release baseline artifact",
    )
    prior_artifact = _require_mapping(
        artifacts.get("prior_candidate"),
        "prior candidate artifact",
    )
    candidate_artifact = _require_mapping(
        artifacts.get("candidate"),
        "candidate artifact",
    )
    predictions = _require_mapping(
        receipt.get("per_example_artifact"),
        "per-example artifact",
    )
    expected_bindings = {
        "receipt_id": receipt.get("receipt_id"),
        "protocol_sha256": protocol.get("content_sha256"),
        "selection_sha256": selection.get("content_sha256"),
        "release_baseline_sha256": release_artifact.get("sha256"),
        "prior_candidate_sha256": prior_artifact.get("sha256"),
        "candidate_sha256": candidate_artifact.get("sha256"),
        "per_example_artifact_sha256": predictions.get("sha256"),
    }
    for field, expected in expected_bindings.items():
        if validation.get(field) != expected:
            raise ValueError(f"Three-way validator {field} cross-link mismatch")
    _three_way_semantic_projection(validation)
    return receipt, validation


def _verify_three_way_v1_archive(
    output_path: Path,
    expected_links: Mapping[str, Mapping[str, Any]],
    expected_closure: list[dict[str, Any]],
    expected_validation: Mapping[str, Any],
) -> None:
    expected_semantics = _three_way_semantic_projection(expected_validation)
    expected_semantics_sha256 = hashlib.sha256(
        canonical_json(expected_semantics)
    ).hexdigest()
    try:
        with zipfile.ZipFile(output_path, "r") as archive:
            manifest_value = json.loads(
                archive.read("artifact_manifest.json"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
            manifest = _require_mapping(manifest_value, "artifact manifest")
            if manifest.get("semantic_validation") != expected_semantics:
                raise ValueError("Three-way semantic validation cross-links changed")
            if (
                manifest.get("semantic_validation_sha256")
                != expected_semantics_sha256
            ):
                raise ValueError("Three-way semantic validation digest changed")
            if (
                manifest.get("evaluation_profile_sha256")
                != THREE_WAY_EVALUATION_PROFILE_SHA256
                or manifest.get("release_continuity_passed") is not True
                or manifest.get("prior_candidate_superiority_passed") is not True
                or manifest.get("checkpoint_inference_replayed") is not True
            ):
                raise ValueError("Three-way manifest semantic admission changed")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Could not verify three-way semantic manifest") from exc
    _verify_content_bound_archive(
        output_path,
        expected_links,
        expected_closure,
        expected_validation,
        receipt_archive_member="three_way_evaluation_receipt.json",
        semantic_validator=_validate_three_way_receipt_semantics,
        semantic_fields=_THREE_WAY_SEMANTIC_FIELDS,
        archive_label="three-way-v1",
    )


def build_three_way_v1_package(
    output_path: Path,
    *,
    receipt_path: Path,
    root: Path = PROJECT_ROOT,
    model_key: str = "cognitive_leap_ultra_v51_2_three_way",
) -> dict[str, Any]:
    """Build a manual-only Store ZIP from the immutable three-way profile.

    The receipt is content-bound evidence only.  Even a passing three-way gate
    grants no publication, activation, routing, default-model, tool, safety,
    or permission authority.
    """

    if (
        not isinstance(model_key, str)
        or not model_key
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in model_key
        )
    ):
        raise ValueError(
            "model_key must contain only lowercase letters, digits, and underscores"
        )
    resolved_root = Path(root).resolve()
    resolved_receipt = Path(receipt_path).resolve()
    receipt, validation = _three_way_receipt_for_packaging(
        resolved_receipt,
        root=resolved_root,
    )
    closure_receipt, closure = _collect_reproducibility_closure(
        resolved_receipt,
        root=resolved_root,
    )
    if closure_receipt != receipt:
        raise ValueError("Three-way receipt changed while collecting package closure")

    def referenced_member(record: Mapping[str, Any], label: str) -> str:
        member = _safe_relative_member(record.get("path"), label=f"{label} path")
        closure_record = closure.get(member)
        if closure_record is None:
            raise ValueError(f"Reproducibility closure is missing {label}: {member}")
        expected_hash, expected_size = _file_binding_from_reference(
            record,
            label=label,
        )
        if (
            closure_record["sha256"] != expected_hash
            or closure_record["size_bytes"] != expected_size
        ):
            raise ValueError(f"Reproducibility closure binding changed for {label}")
        return member

    protocol_record = _require_mapping(receipt.get("protocol"), "protocol reference")
    selection_record = _require_mapping(receipt.get("selection"), "selection reference")
    artifacts = _require_mapping(receipt.get("artifacts"), "artifacts")
    release_record = _require_mapping(
        artifacts.get("release_baseline"),
        "release baseline artifact",
    )
    prior_record = _require_mapping(
        artifacts.get("prior_candidate"),
        "prior candidate artifact",
    )
    candidate_record = _require_mapping(
        artifacts.get("candidate"),
        "candidate artifact",
    )
    prediction_record = _require_mapping(
        receipt.get("per_example_artifact"),
        "per-example artifact",
    )
    protocol_member = referenced_member(protocol_record, "protocol")
    selection_member = referenced_member(selection_record, "selection")
    release_member = referenced_member(release_record, "release baseline")
    prior_member = referenced_member(prior_record, "prior candidate")
    candidate_member = referenced_member(candidate_record, "candidate")
    predictions_member = referenced_member(prediction_record, "three-way predictions")
    selection_payload = _load_strict_json(closure[selection_member]["path"])
    lineage_record = _require_mapping(
        selection_payload.get("lineage_manifest"),
        "lineage manifest",
    )
    lineage_verification_record = _require_mapping(
        selection_payload.get("lineage_verification"),
        "lineage verification",
    )
    lineage_member = referenced_member(lineage_record, "lineage manifest")
    lineage_verification_member = referenced_member(
        lineage_verification_record,
        "lineage verification",
    )
    if lineage_record.get("schema") != "supermix-cognitive-leap-lineage-v2":
        raise ValueError("Three-way receipt does not bind a lineage-v2 manifest")
    if (
        lineage_verification_record.get("schema")
        != "supermix-cognitive-leap-lineage-verification-v1"
    ):
        raise ValueError("Three-way receipt does not bind lineage verification")
    if prediction_record.get("schema") != THREE_WAY_PREDICTION_SCHEMA:
        raise ValueError("Three-way receipt does not bind the three-way logits schema")
    candidate_state_hash = candidate_record.get("canonical_state_sha256")
    if (
        not isinstance(candidate_state_hash, str)
        or len(candidate_state_hash) != 64
        or any(character not in "0123456789abcdef" for character in candidate_state_hash)
    ):
        raise ValueError("Candidate canonical state hash is missing or invalid")

    receipt_member = resolved_receipt.relative_to(resolved_root).as_posix()
    receipt_member = _safe_relative_member(receipt_member, label="receipt path")
    if receipt_member not in closure:
        raise ValueError("Reproducibility closure is missing its entry receipt")
    candidate_path = closure[candidate_member]["path"]
    aliases = {
        "candidate": "cognitive_leap_ultra_v51_2_three_way.pth",
        "receipt": "three_way_evaluation_receipt.json",
        "chat_meta": "chat_demo_meta_v51_2_three_way.json",
    }
    chat_metadata = _three_way_v1_chat_metadata(
        candidate_path=candidate_path,
        receipt_path=resolved_receipt,
        receipt=receipt,
        candidate_alias=aliases["candidate"],
        receipt_alias=aliases["receipt"],
    )

    def closure_link(
        archive_member: str,
        source_record: Mapping[str, Any],
        *,
        extra_keys: Iterable[str] = (),
    ) -> dict[str, Any]:
        source_member = referenced_member(source_record, archive_member)
        closure_record = closure[source_member]
        link = {
            "archive_member": archive_member,
            "sha256": closure_record["sha256"],
            "size_bytes": closure_record["size_bytes"],
        }
        for key in extra_keys:
            if key in source_record:
                link[key] = source_record[key]
        return link

    receipt_link_record = {
        "sha256": closure[receipt_member]["sha256"],
        "size_bytes": closure[receipt_member]["size_bytes"],
        "schema": receipt["schema"],
        "receipt_id": receipt.get("receipt_id"),
        "evaluation_profile_sha256": receipt.get("evaluation_profile_sha256"),
    }
    evidence_links = {
        "receipt": _manifest_evidence_link(
            aliases["receipt"],
            receipt_link_record,
            extra_keys=("schema", "receipt_id", "evaluation_profile_sha256"),
        ),
        "protocol": closure_link(protocol_member, protocol_record),
        "selection": closure_link(selection_member, selection_record),
        "release_baseline": closure_link(
            release_member,
            release_record,
            extra_keys=(
                "canonical_state_sha256",
                "tensor_count",
                "element_count",
            ),
        ),
        "prior_candidate": closure_link(
            prior_member,
            prior_record,
            extra_keys=(
                "status",
                "canonical_state_sha256",
                "tensor_count",
                "element_count",
            ),
        ),
        "candidate": _manifest_evidence_link(
            aliases["candidate"],
            candidate_record,
            extra_keys=(
                "canonical_state_sha256",
                "tensor_count",
                "element_count",
            ),
        ),
        "lineage": closure_link(
            lineage_member,
            lineage_record,
            extra_keys=("schema",),
        ),
        "lineage_verification": closure_link(
            lineage_verification_member,
            lineage_verification_record,
            extra_keys=("schema",),
        ),
        "predictions": closure_link(
            predictions_member,
            prediction_record,
            extra_keys=(
                "schema",
                "evaluation_profile_sha256",
                "uncompressed_sha256",
                "row_count",
                "dataset_id",
            ),
        ),
        "chat_meta": {
            "archive_member": aliases["chat_meta"],
            "sha256": hashlib.sha256(chat_metadata).hexdigest(),
            "size_bytes": len(chat_metadata),
            "checkpoint_archive_member": aliases["candidate"],
            "receipt_archive_member": aliases["receipt"],
            "manual_selection_only": True,
        },
    }
    closure_rows = _closure_manifest_rows(closure)
    closure_sha256 = hashlib.sha256(canonical_json(closure_rows)).hexdigest()
    semantic_validation = _three_way_semantic_projection(validation)
    semantic_validation_sha256 = hashlib.sha256(
        canonical_json(semantic_validation)
    ).hexdigest()
    package_files: list[tuple[str, Path]] = [
        (member, record["path"])
        for member, record in sorted(closure.items())
    ]
    existing_names = {member.casefold(): member for member, _path in package_files}

    def add_alias(alias: str, path: Path, exact_member: str) -> None:
        existing_name = existing_names.get(alias.casefold())
        if existing_name is not None:
            if existing_name != alias or exact_member != alias:
                raise ValueError(
                    f"Runtime alias collides with reproducibility member: {alias}"
                )
            return
        existing_names[alias.casefold()] = alias
        package_files.append((alias, path))

    add_alias(aliases["candidate"], candidate_path, candidate_member)
    add_alias(aliases["receipt"], resolved_receipt, receipt_member)
    result = build_zip(
        Path(output_path),
        files=package_files,
        generated=(
            (aliases["chat_meta"], chat_metadata),
            (
                "MODEL_CARD.md",
                model_card(
                    title="Cognitive Leap Ultra v51.2 profiled three-way artifact",
                    status="both bounded comparisons passed; manually selectable only",
                    intended_use=(
                        "Reproducing the immutable-profile synthetic arithmetic "
                        "evaluation against both the canonical v51 release and the "
                        "exact unpromoted v51.1 candidate."
                    ),
                    evidence=(
                        "The independent three-way validator replayed all three bound "
                        "checkpoints and accepted both paired comparisons. The full "
                        "protocol, selection, release baseline, prior candidate, "
                        "training members and receipts, source snapshots, lineage, "
                        "predictions, and candidate are included transitively."
                    ),
                    limitations=(
                        "This unauthenticated receipt is bounded evidence only and "
                        "grants no publication, activation, routing, default-model, "
                        "safety, tool, or permission authority. It is not a general-chat "
                        "or general-reasoning claim."
                    ),
                ),
            ),
        ),
        package_contract={
            "schema": ARTIFACT_MANIFEST_V2_SCHEMA,
            "model_key": model_key,
            "status": "three_way_evaluation_pass_manual_only",
            "manual_selectable": True,
            "manual_selection_only": True,
            "auto_route_allowed": False,
            "default_model_allowed": False,
            "receipt_grants_activation": False,
            "receipt_grants_store_publication": False,
            "authentication": "none",
            "integrity_status": CONTENT_BOUND_STATUS,
            "runtime_status": CONTENT_BOUND_STATUS,
            "receipt_authority": dict(BOUNDED_EVALUATION_AUTHORITY),
            "gate_outcome": "pass",
            "evaluation_profile_sha256": THREE_WAY_EVALUATION_PROFILE_SHA256,
            "release_continuity_passed": True,
            "prior_candidate_superiority_passed": True,
            "checkpoint_inference_replayed": True,
            "semantic_validation": semantic_validation,
            "semantic_validation_sha256": semantic_validation_sha256,
            "reproducibility_closure_schema": (
                "supermix-cognitive-leap-reproducibility-closure-v1"
            ),
            "reproducibility_closure": closure_rows,
            "reproducibility_closure_sha256": closure_sha256,
            "portable_validation": {
                "receipt_archive_member": aliases["receipt"],
                "root": ".",
                "validator": (
                    "cognitive_leap_three_way_receipt.validate_receipt"
                ),
                "verify_inference": True,
                "verified_after_temporary_extraction": True,
            },
            "runtime_files": {
                "checkpoint": aliases["candidate"],
                "chat_metadata": aliases["chat_meta"],
                "three_way_receipt": aliases["receipt"],
            },
            "evidence_links": evidence_links,
        },
    )
    try:
        _verify_three_way_v1_archive(
            Path(output_path),
            evidence_links,
            closure_rows,
            validation,
        )
    except Exception:
        Path(output_path).unlink(missing_ok=True)
        raise
    return result


def build_packages(output_dir: Path) -> dict[str, Any]:
    v50_weights = PROJECT_ROOT / "artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth"
    v50_meta = PROJECT_ROOT / "chat_model_meta_v50_cognitive_leap.json"
    v51_dir = PROJECT_ROOT / "output/benchmark_v51_cognitive_leap_ultra_latest"
    v51_weights = v51_dir / "cognitive_leap_ultra_v51_trained.pth"
    v51_meta = v51_dir / "chat_demo_meta.json"
    candidate_dir = PROJECT_ROOT / "output/training_candidates/cognitive_leap_ultra_v51_1_balanced_blend30_seed151"
    candidate_weights = candidate_dir / "cognitive_leap_ultra_v51_1_balanced_blend30.pth"
    candidate_receipt = candidate_dir / "bounded_promotion_receipt.json"
    candidate_metrics = json.loads(candidate_receipt.read_text(encoding="utf-8"))
    candidate_meta = build_demo_metadata(
        weights=candidate_weights,
        metrics_path=candidate_receipt,
    )
    candidate_meta.update(
        {
            "purpose": "unpromoted v51.1 balanced-blend bounded arithmetic research demo",
            "training_task": "synthetic four-operation chained modular arithmetic modulo 10",
            "fine_tuned_weights": "cognitive_leap_ultra_v51_1_balanced_blend30.pth",
            "checkpoint_path": "cognitive_leap_ultra_v51_1_balanced_blend30.pth",
            "benchmark_metrics": "bounded_promotion_receipt.json",
            "prediction_stability_metrics": "",
            "promotion": {
                "passed": False,
                "decision": "reject",
                "receipt": "bounded_promotion_receipt.json",
                "reason": "15 of 20 final seeds were non-regressing; the predeclared gate required 16.",
            },
            "benchmark_summary": candidate_metrics.get("summary", {}),
        }
    )

    rows = []
    rows.append(
        build_zip(
            output_dir / "supermix_cognitive_leap_v50_chat_20260812.zip",
            files=(
                ("champion_model_chat_v50_cognitive_leap.pth", v50_weights),
                ("chat_model_meta_v50_cognitive_leap.json", v50_meta),
            ),
            generated=(
                (
                    "MODEL_CARD.md",
                    model_card(
                        title="Cognitive Leap v50 legacy chat checkpoint",
                        status="legacy archive; manually selectable",
                        intended_use="Compatibility and historical comparison with the Champion chat runtime.",
                        evidence="The checkpoint and original metadata are preserved byte-for-byte and reload-tested before publication.",
                        limitations="The original metadata references training inputs that are not bundled. No new benchmark or broad capability claim is attached.",
                    ),
                ),
            ),
            package_contract={
                "model_key": "cognitive_leap_v50",
                "status": "legacy_archive",
                "auto_route_allowed": False,
            },
        )
    )
    rows.append(
        build_zip(
            output_dir / "supermix_cognitive_leap_ultra_v51_demo_20260812.zip",
            files=(
                ("cognitive_leap_ultra_v51_trained.pth", v51_weights),
                ("chat_demo_meta_v51.json", v51_meta),
                ("benchmark_results.json", v51_dir / "benchmark_results.json"),
                ("prediction_stability_gate.json", v51_dir / "prediction_stability_gate_4096_decision_v5.json"),
                ("chat_response_fidelity_gate.json", v51_dir / "chat_response_fidelity_gate.json"),
            ),
            generated=(
                (
                    "MODEL_CARD.md",
                    model_card(
                        title="Cognitive Leap Ultra v51 bounded runtime demo",
                        status="canonical bounded demo; manually selectable",
                        intended_use="Reproducing the synthetic arithmetic checkpoint and exercising recursive/adaptive runtime controls.",
                        evidence="Includes the canonical checkpoint, metadata, synthetic benchmark metrics, prediction-stability receipt, and frozen chat-response fidelity receipt.",
                        limitations="This is not a polished assistant. Its training task is four-operation chained modular arithmetic modulo 10.",
                    ),
                ),
            ),
            package_contract={
                "model_key": "cognitive_leap_ultra_v51_demo",
                "status": "bounded_demo",
                "auto_route_allowed": False,
            },
        )
    )
    rows.append(
        build_zip(
            output_dir / "supermix_cognitive_leap_ultra_v51_1_balanced_blend30_20260812.zip",
            files=(
                ("cognitive_leap_ultra_v51_1_balanced_blend30.pth", candidate_weights),
                ("bounded_promotion_receipt.json", candidate_receipt),
                ("blend_manifest.json", candidate_dir / "blend_manifest.json"),
                (
                    "balanced_continuation_training_receipt.json",
                    PROJECT_ROOT
                    / "output/training_candidates/cognitive_leap_ultra_v51_1_balanced_continue12k_lr1e4_seed151/training_receipt.json",
                ),
            ),
            generated=(
                ("chat_demo_meta_v51_1_balanced_blend30.json", canonical_json(candidate_meta)),
                (
                    "MODEL_CARD.md",
                    model_card(
                        title="Cognitive Leap Ultra v51.1 balanced blend",
                        status="experimental; bounded promotion gate failed",
                        intended_use="Reviewing a conservative v51 continuation/blend and its complete failed promotion receipt.",
                        evidence="On the fresh 40,000-example final cohort it improved aggregate accuracy and loss and every coarse operation family, but missed the predeclared seed gate (15/20 versus 16/20 required).",
                        limitations="Unpromoted research artifact. Manual selection only; it must not be described as better chat or general reasoning.",
                    ),
                ),
            ),
            package_contract={
                "model_key": "cognitive_leap_ultra_v51_1_balanced_blend30",
                "status": "unpromoted_experimental",
                "promotion_passed": False,
                "auto_route_allowed": False,
            },
        )
    )
    summary = {
        "schema": "supermix-cognitive-leap-store-packages-v1",
        "packages": rows,
    }
    (output_dir / "package_summary.json").write_bytes(canonical_json(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "output/model_store_publish_20260812"),
    )
    args = parser.parse_args()
    summary = build_packages(Path(args.output_dir).resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
