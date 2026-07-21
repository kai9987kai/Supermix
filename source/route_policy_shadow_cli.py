"""Operate the local, shadow-only whole-policy commitment registry.

All mutations are explicit local CLI actions.  The command never executes a
route, writes the executed-decision ledger, enables a policy, or prints private
seed material.  ``seal`` writes the private seed capsule to a separate file
before committing the public campaign package to SQLite so an interrupted
operation can be recovered safely.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

try:
    from .route_policy_shadow_registry import (
        MAX_SHADOW_VERIFY_BATCH,
        RouteShadowAssignmentRegistry,
        RouteShadowRegistryError,
        audit_shadow_seed_capsule,
        build_shadow_design_binding,
        create_shadow_campaign_artifacts,
        generate_shadow_seed,
    )
except ImportError:  # pragma: no cover - direct ``python source/...`` use
    from route_policy_shadow_registry import (
        MAX_SHADOW_VERIFY_BATCH,
        RouteShadowAssignmentRegistry,
        RouteShadowRegistryError,
        audit_shadow_seed_capsule,
        build_shadow_design_binding,
        create_shadow_campaign_artifacts,
        generate_shadow_seed,
    )


_SEED_CAPSULE_KEYS = {
    "schema_version",
    "campaign_id",
    "design_binding_hash",
    "seed_commitment",
    "seed_material_base64url",
}


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"input JSON contains duplicate object key: {key}")
        value[key] = item
    return value


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"input JSON contains non-finite number: {value}")


def _read_json(path: str, *, name: str) -> Any:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc.msg}") from exc


def _read_json_object(path: str, *, name: str) -> Dict[str, Any]:
    value = _read_json(path, name=name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _render(value: Any, *, compact: bool) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    )


def _decode_capsule_seed(capsule: Dict[str, Any]) -> bytes:
    if set(capsule) != _SEED_CAPSULE_KEYS:
        raise ValueError("private seed capsule has an unsupported field set")
    encoded = capsule.get("seed_material_base64url")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("private seed capsule does not contain canonical seed material")
    padding = "=" * (-len(encoded) % 4)
    try:
        seed = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("private seed capsule does not contain canonical seed material") from exc
    canonical = base64.urlsafe_b64encode(seed).decode("ascii").rstrip("=")
    if len(seed) != 32 or canonical != encoded:
        raise ValueError("private seed capsule does not contain canonical seed material")
    return seed


def _load_recovery_capsule(path: Path, review_bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], bytes]:
    capsule = _read_json_object(str(path), name="private seed capsule")
    seed = _decode_capsule_seed(capsule)
    artifacts = create_shadow_campaign_artifacts(review_bundle, seed)
    audit_shadow_seed_capsule(artifacts["public_package"], capsule)
    expected = artifacts["private_seed_capsule"]
    if _canonical_json(capsule) != _canonical_json(expected):
        raise ValueError("private seed capsule does not match the review bundle")
    return capsule, seed


def _write_private_capsule(path: Path, capsule: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(_render(capsule, compact=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            # Windows ACLs, rather than POSIX mode bits, are authoritative.
            pass
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _same_path(first: Path, second: Path) -> bool:
    return os.path.normcase(str(first.expanduser().resolve())) == os.path.normcase(
        str(second.expanduser().resolve())
    )


def _seal(args: argparse.Namespace) -> Dict[str, Any]:
    bundle = _read_json_object(str(args.bundle), name="review bundle")
    registry_path = Path(args.registry).expanduser()
    capsule_path = Path(args.seed_output).expanduser()
    if _same_path(registry_path, capsule_path):
        raise ValueError("registry and private seed capsule paths must be different")

    capsule_written = False
    capsule_recovered = False
    if capsule_path.exists():
        _capsule, seed = _load_recovery_capsule(capsule_path, bundle)
        capsule_recovered = True
    else:
        if registry_path.is_file():
            origin_hash = build_shadow_design_binding(bundle)["origin_review_bundle_hash"]
            existing = RouteShadowAssignmentRegistry(
                registry_path, read_only=True
            ).snapshot()
            if any(
                row["origin_review_bundle_hash"] == origin_hash
                for row in existing["campaigns"]
            ):
                raise ValueError(
                    "origin review bundle is already sealed; recover its original seed capsule"
                )
        seed = generate_shadow_seed()
        artifacts = create_shadow_campaign_artifacts(bundle, seed)
        try:
            _write_private_capsule(capsule_path, artifacts["private_seed_capsule"])
            capsule_written = True
        except FileExistsError:
            _capsule, seed = _load_recovery_capsule(capsule_path, bundle)
            capsule_recovered = True

    result = RouteShadowAssignmentRegistry(registry_path).seal_campaign(bundle, seed)

    # Deliberately reconstruct the result instead of mutating the registry return
    # so private seed material cannot reach stdout through a future extra field.
    return {
        "ok": bool(result["ok"]),
        "created": bool(result["created"]),
        "public_package": result["public_package"],
        "private_seed_capsule_written": capsule_written,
        "private_seed_capsule_recovered": capsule_recovered,
        "private_seed_material_persisted_in_registry": False,
        "private_seed_material_returned": False,
        "execution_enabled": False,
        "activation_available": False,
    }


def _commit(args: argparse.Namespace) -> Dict[str, Any]:
    capsule = _read_json_object(str(args.seed_input), name="private seed capsule")
    cluster = _read_json_object(str(args.cluster_input), name="cluster input")
    if set(cluster) != {"cluster_identifier"}:
        raise ValueError("cluster input must contain exactly one field: cluster_identifier")
    identifier = cluster["cluster_identifier"]
    if not isinstance(identifier, str):
        raise ValueError("cluster_identifier must be a JSON string")
    return RouteShadowAssignmentRegistry(args.registry).append_assignment_commitment(
        campaign_id=args.campaign,
        seed_capsule=capsule,
        cluster_identifier=identifier,
    )


def _close(args: argparse.Namespace) -> Dict[str, Any]:
    return RouteShadowAssignmentRegistry(args.registry).close_campaign(args.campaign)


def _reveal(args: argparse.Namespace) -> Dict[str, Any]:
    capsule = _read_json_object(str(args.seed_input), name="private seed capsule")
    artifact = RouteShadowAssignmentRegistry(args.registry).reveal_seed(
        campaign_id=args.campaign,
        seed_capsule=capsule,
    )
    reveal = artifact["reveal"]
    # The registry retains the public opening needed for reconstruction, but
    # command output remains a non-secret receipt by contract.
    return {
        "ok": True,
        "campaign_id": reveal["campaign_id"],
        "seal_hash": reveal["seal_hash"],
        "seed_commitment": reveal["seed_commitment"],
        "seed_material_fingerprint": reveal["seed_material_fingerprint"],
        "seed_material_revealed_in_registry": True,
        "seed_material_returned": False,
        "state": reveal["state"],
        "seed_reveal_hash": artifact["seed_reveal_hash"],
        "execution_enabled": False,
        "activation_available": False,
    }


def _verify(args: argparse.Namespace) -> Dict[str, Any]:
    return RouteShadowAssignmentRegistry(args.registry).verify_assignment_reveals(
        args.campaign,
        batch_size=args.batch_size,
    )


def _status(args: argparse.Namespace) -> Dict[str, Any]:
    registry_path = Path(args.registry).expanduser()
    if not registry_path.is_file():
        raise ValueError("shadow registry does not exist")
    snapshot = RouteShadowAssignmentRegistry(
        registry_path, read_only=True
    ).snapshot(args.campaign)
    return {**snapshot, "read_only": True}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--registry", required=True, help="local shadow-registry SQLite path")
    parser.add_argument("--compact", action="store_true", help="emit compact canonical JSON")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Operate a local shadow-only whole-policy commitment/reveal registry; "
            "no command executes or promotes a route policy."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    seal = commands.add_parser("seal", help="seal a source-bound review bundle")
    _add_common(seal)
    seal.add_argument("--bundle", required=True, help="review-bundle JSON path")
    seal.add_argument(
        "--seed-output",
        required=True,
        help="separate private seed-capsule JSON path (created exclusively)",
    )
    seal.set_defaults(handler=_seal)

    commit = commands.add_parser("commit", help="append one opaque cluster commitment")
    _add_common(commit)
    commit.add_argument("--campaign", required=True, help="sealed shadow campaign identifier")
    commit.add_argument("--seed-input", required=True, help="private seed-capsule JSON path")
    commit.add_argument(
        "--cluster-input",
        required=True,
        help='JSON path containing only {"cluster_identifier": "..."}',
    )
    commit.set_defaults(handler=_commit)

    close = commands.add_parser("close", help="freeze the commitment population")
    _add_common(close)
    close.add_argument("--campaign", required=True, help="sealed shadow campaign identifier")
    close.set_defaults(handler=_close)

    reveal = commands.add_parser("reveal", help="reveal the seed after commitment closure")
    _add_common(reveal)
    reveal.add_argument("--campaign", required=True, help="sealed shadow campaign identifier")
    reveal.add_argument("--seed-input", required=True, help="private seed-capsule JSON path")
    reveal.set_defaults(handler=_reveal)

    verify = commands.add_parser("verify", help="reconstruct a bounded batch of assignments")
    _add_common(verify)
    verify.add_argument("--campaign", required=True, help="sealed shadow campaign identifier")
    verify.add_argument(
        "--batch-size",
        type=int,
        default=MAX_SHADOW_VERIFY_BATCH,
        help=f"maximum assignments to reconstruct (1-{MAX_SHADOW_VERIFY_BATCH})",
    )
    verify.set_defaults(handler=_verify)

    status = commands.add_parser("status", help="read and verify registry state")
    _add_common(status)
    status.add_argument("--campaign", help="optional sealed shadow campaign identifier")
    status.set_defaults(handler=_status)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        result = args.handler(args)
        print(_render(result, compact=bool(args.compact)))
    except (KeyError, OSError, sqlite3.Error, TypeError, ValueError, RouteShadowRegistryError) as exc:
        print(f"route shadow registry error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
