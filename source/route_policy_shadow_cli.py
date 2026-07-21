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


def _parse_json(raw: str, *, name: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {exc.msg}") from exc


def _read_json(path: str, *, name: str) -> Any:
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8-sig")
    return _parse_json(raw, name=name)


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


def _is_windows() -> bool:
    return os.name == "nt"


def _windows_acl_runtime():
    """Load the small Win32 ACL surface lazily for source and frozen builds."""

    import ctypes
    import msvcrt
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class Acl(ctypes.Structure):
        _fields_ = [
            ("AclRevision", ctypes.c_ubyte),
            ("Sbz1", ctypes.c_ubyte),
            ("AclSize", wintypes.WORD),
            ("AceCount", wintypes.WORD),
            ("Sbz2", wintypes.WORD),
        ]

    class AceHeader(ctypes.Structure):
        _fields_ = [
            ("AceType", ctypes.c_ubyte),
            ("AceFlags", ctypes.c_ubyte),
            ("AceSize", wintypes.WORD),
        ]

    class AccessAllowedAce(ctypes.Structure):
        _fields_ = [
            ("Header", AceHeader),
            ("Mask", wintypes.DWORD),
            ("SidStart", wintypes.DWORD),
        ]

    class AclSizeInformation(ctypes.Structure):
        _fields_ = [
            ("AceCount", wintypes.DWORD),
            ("AclBytesInUse", wintypes.DWORD),
            ("AclBytesFree", wintypes.DWORD),
        ]

    class SidAndAttributes(ctypes.Structure):
        _fields_ = [("Sid", wintypes.LPVOID), ("Attributes", wintypes.DWORD)]

    class TokenUser(ctypes.Structure):
        _fields_ = [("User", SidAndAttributes)]

    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetLengthSid.argtypes = [wintypes.LPVOID]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.CopySid.argtypes = [wintypes.DWORD, wintypes.LPVOID, wintypes.LPVOID]
    advapi32.CopySid.restype = wintypes.BOOL
    advapi32.InitializeAcl.argtypes = [wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD]
    advapi32.InitializeAcl.restype = wintypes.BOOL
    advapi32.AddAccessAllowedAceEx.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
    ]
    advapi32.AddAccessAllowedAceEx.restype = wintypes.BOOL
    advapi32.SetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    advapi32.SetSecurityInfo.restype = wintypes.DWORD
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorControl.argtypes = [
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL
    advapi32.GetAclInformation.argtypes = [
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    ]
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [wintypes.LPVOID, wintypes.LPVOID]
    advapi32.EqualSid.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.LPVOID]
    kernel32.LocalFree.restype = wintypes.LPVOID

    return {
        "ctypes": ctypes,
        "wintypes": wintypes,
        "msvcrt": msvcrt,
        "advapi32": advapi32,
        "kernel32": kernel32,
        "Acl": Acl,
        "AceHeader": AceHeader,
        "AccessAllowedAce": AccessAllowedAce,
        "AclSizeInformation": AclSizeInformation,
        "TokenUser": TokenUser,
    }


def _windows_api_failure(runtime: Dict[str, Any], action: str) -> OSError:
    code = runtime["ctypes"].get_last_error() or 1
    return OSError(code, f"Windows refused to {action} the private seed capsule ACL")


def _windows_current_user_sid(runtime: Dict[str, Any]):
    """Return a stable copy of the process token's user SID."""

    ctypes = runtime["ctypes"]
    wintypes = runtime["wintypes"]
    advapi32 = runtime["advapi32"]
    kernel32 = runtime["kernel32"]
    token = wintypes.HANDLE()
    token_query = 0x0008
    token_user_class = 1
    error_insufficient_buffer = 122
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), token_query, ctypes.byref(token)
    ):
        raise _windows_api_failure(runtime, "query")
    try:
        required = wintypes.DWORD()
        ctypes.set_last_error(0)
        advapi32.GetTokenInformation(
            token, token_user_class, None, 0, ctypes.byref(required)
        )
        if ctypes.get_last_error() != error_insufficient_buffer or required.value == 0:
            raise _windows_api_failure(runtime, "query")
        token_buffer = ctypes.create_string_buffer(required.value)
        if not advapi32.GetTokenInformation(
            token,
            token_user_class,
            token_buffer,
            required.value,
            ctypes.byref(required),
        ):
            raise _windows_api_failure(runtime, "query")
        token_user = ctypes.cast(
            token_buffer, ctypes.POINTER(runtime["TokenUser"])
        ).contents
        sid_length = advapi32.GetLengthSid(token_user.User.Sid)
        if sid_length == 0:
            raise _windows_api_failure(runtime, "query")
        sid = ctypes.create_string_buffer(sid_length)
        if not advapi32.CopySid(sid_length, sid, token_user.User.Sid):
            raise _windows_api_failure(runtime, "copy")
        return sid
    finally:
        kernel32.CloseHandle(token)


def _windows_os_handle(runtime: Dict[str, Any], descriptor: int):
    raw_handle = runtime["msvcrt"].get_osfhandle(descriptor)
    if raw_handle == -1:
        raise OSError("Windows refused to resolve the private seed capsule handle")
    return runtime["wintypes"].HANDLE(raw_handle)


def _apply_windows_private_capsule_acl(path: Path) -> None:
    """Install a protected, current-user-only DACL on an empty capsule file."""

    runtime = _windows_acl_runtime()
    ctypes = runtime["ctypes"]
    wintypes = runtime["wintypes"]
    advapi32 = runtime["advapi32"]
    sid = _windows_current_user_sid(runtime)
    sid_length = advapi32.GetLengthSid(sid)
    if sid_length == 0:
        raise _windows_api_failure(runtime, "query")

    # FILE_ALL_ACCESS is deliberate: the creator remains able to rotate or
    # delete the capsule, while the protected DACL contains no other trustee.
    file_all_access = 0x001F01FF
    acl_revision = 2
    acl_size = (
        ctypes.sizeof(runtime["Acl"])
        + ctypes.sizeof(runtime["AccessAllowedAce"])
        - ctypes.sizeof(wintypes.DWORD)
        + sid_length
    )
    acl = ctypes.create_string_buffer(acl_size)
    if not advapi32.InitializeAcl(acl, acl_size, acl_revision):
        raise _windows_api_failure(runtime, "initialize")
    if not advapi32.AddAccessAllowedAceEx(
        acl, acl_revision, 0, file_all_access, sid
    ):
        raise _windows_api_failure(runtime, "populate")

    se_file_object = 1
    dacl_security_information = 0x00000004
    protected_dacl_security_information = 0x80000000
    # SetNamedSecurityInfo opens the newly-created empty file with WRITE_DAC,
    # which the CRT data handle intentionally does not request.  The still-open
    # CRT handle prevents replacement, and verification below uses that handle.
    result = advapi32.SetNamedSecurityInfoW(
        str(path),
        se_file_object,
        dacl_security_information | protected_dacl_security_information,
        None,
        None,
        acl,
        None,
    )
    if result != 0:
        raise OSError(result, "Windows refused to restrict the private seed capsule ACL")


def _verify_windows_private_capsule_acl(descriptor: int) -> None:
    """Verify the capsule has exactly one protected full-control user ACE."""

    runtime = _windows_acl_runtime()
    ctypes = runtime["ctypes"]
    wintypes = runtime["wintypes"]
    advapi32 = runtime["advapi32"]
    kernel32 = runtime["kernel32"]
    expected_sid = _windows_current_user_sid(runtime)

    se_file_object = 1
    dacl_security_information = 0x00000004
    security_descriptor = wintypes.LPVOID()
    dacl = wintypes.LPVOID()
    result = advapi32.GetSecurityInfo(
        _windows_os_handle(runtime, descriptor),
        se_file_object,
        dacl_security_information,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(security_descriptor),
    )
    if result != 0:
        raise OSError(result, "Windows refused to read the private seed capsule ACL")
    try:
        dacl_present = wintypes.BOOL()
        dacl_defaulted = wintypes.BOOL()
        descriptor_dacl = wintypes.LPVOID()
        if not advapi32.GetSecurityDescriptorDacl(
            security_descriptor,
            ctypes.byref(dacl_present),
            ctypes.byref(descriptor_dacl),
            ctypes.byref(dacl_defaulted),
        ):
            raise _windows_api_failure(runtime, "verify")
        if (
            not dacl_present.value
            or not descriptor_dacl.value
            or descriptor_dacl.value != dacl.value
        ):
            raise OSError("private seed capsule does not have an explicit Windows DACL")

        control = wintypes.WORD()
        revision = wintypes.DWORD()
        if not advapi32.GetSecurityDescriptorControl(
            security_descriptor, ctypes.byref(control), ctypes.byref(revision)
        ):
            raise _windows_api_failure(runtime, "verify")
        se_dacl_protected = 0x1000
        if not control.value & se_dacl_protected:
            raise OSError("private seed capsule Windows DACL still inherits permissions")

        acl_info = runtime["AclSizeInformation"]()
        acl_size_information_class = 2
        if not advapi32.GetAclInformation(
            dacl,
            ctypes.byref(acl_info),
            ctypes.sizeof(acl_info),
            acl_size_information_class,
        ):
            raise _windows_api_failure(runtime, "verify")
        if acl_info.AceCount != 1:
            raise OSError("private seed capsule Windows DACL is not current-user-only")

        ace_pointer = wintypes.LPVOID()
        if not advapi32.GetAce(dacl, 0, ctypes.byref(ace_pointer)):
            raise _windows_api_failure(runtime, "verify")
        ace = ctypes.cast(
            ace_pointer, ctypes.POINTER(runtime["AccessAllowedAce"])
        ).contents
        access_allowed_ace_type = 0
        file_all_access = 0x001F01FF
        if (
            ace.Header.AceType != access_allowed_ace_type
            or ace.Header.AceFlags != 0
            or ace.Mask != file_all_access
        ):
            raise OSError("private seed capsule Windows DACL has unexpected access rights")
        sid_address = ace_pointer.value + runtime["AccessAllowedAce"].SidStart.offset
        if not advapi32.EqualSid(wintypes.LPVOID(sid_address), expected_sid):
            raise OSError("private seed capsule Windows DACL trustee is not the current user")
    finally:
        if security_descriptor.value:
            kernel32.LocalFree(security_descriptor)


def _read_private_capsule(path: Path) -> Dict[str, Any]:
    """Read a capsule only after validating its Windows access boundary."""

    with path.open("r", encoding="utf-8-sig") as handle:
        if _is_windows():
            _verify_windows_private_capsule_acl(handle.fileno())
        raw = handle.read()
    value = _parse_json(raw, name="private seed capsule")
    if not isinstance(value, dict):
        raise ValueError("private seed capsule must be a JSON object")
    return value


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
    capsule = _read_private_capsule(path)
    seed = _decode_capsule_seed(capsule)
    artifacts = create_shadow_campaign_artifacts(review_bundle, seed)
    audit_shadow_seed_capsule(artifacts["public_package"], capsule)
    expected = artifacts["private_seed_capsule"]
    if _canonical_json(capsule) != _canonical_json(expected):
        raise ValueError("private seed capsule does not match the review bundle")
    return capsule, seed


def _write_private_capsule(path: Path, capsule: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_RDWR gives the verifier READ_CONTROL through GENERIC_READ on Windows.
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    descriptor = os.open(str(path), flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            if _is_windows():
                # The inherited DACL may be broad, so lock and verify the still
                # empty file before any private seed bytes are written.
                _apply_windows_private_capsule_acl(path)
                _verify_windows_private_capsule_acl(handle.fileno())
            handle.write(_render(capsule, compact=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            if _is_windows():
                _verify_windows_private_capsule_acl(handle.fileno())
        if not _is_windows():
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
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
    capsule = _read_private_capsule(Path(args.seed_input).expanduser())
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
    capsule = _read_private_capsule(Path(args.seed_input).expanduser())
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
