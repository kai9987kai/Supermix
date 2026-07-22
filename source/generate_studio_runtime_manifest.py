"""Generate the deterministic Supermix Studio runtime distribution contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


MANIFEST_SCHEMA_VERSION = "supermix-studio-runtime-manifest-v1"
STUDIO_APP_VERSION = "2026.07.18"
DEFAULT_MANIFEST_PATH = Path("source/studio_runtime_manifest.json")

RUNTIME_MODULES = (
    "source/chat_app.py",
    "source/chat_web_app.py",
    "source/model_variants.py",
    "source/route_policy_ledger.py",
    "source/route_policy_lab.py",
    "source/route_policy_explorer.py",
    "source/route_policy_protocol.py",
    "source/route_policy_shadow_registry.py",
    "source/route_policy_study_cli.py",
    "source/route_policy_protocol_cli.py",
    "source/route_policy_shadow_cli.py",
    "source/multimodel_runtime.py",
    "source/supermix_multimodel_web_app.py",
    "source/supermix_multimodel_desktop_app.py",
    "runtime_python/chat_app.py",
    "runtime_python/chat_web_app.py",
    "runtime_python/chat_pipeline.py",
    "runtime_python/chat_memory.py",
    "runtime_python/device_utils.py",
    "runtime_python/llm_database.py",
    "runtime_python/model_variants.py",
    "runtime_python/run.py",
)

CONTRACT_CONSTANTS = {
    "source/chat_app.py": (
        "AUTO_COMPUTE_PLAN_SCHEMA_VERSION",
        "AUTO_COMPUTE_STRATEGY",
        "DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K",
    ),
    "source/route_policy_ledger.py": (
        "LEDGER_SCHEMA_VERSION",
        "SUPPORT_SCHEMA_VERSION",
        "EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION",
        "OUTCOME_CONTRACT_SCHEMA_VERSION",
        "OUTCOME_MATURITY_SCHEMA_VERSION",
    ),
    "source/route_policy_lab.py": ("READINESS_SCHEMA_VERSION",),
    "source/route_policy_explorer.py": (
        "STUDY_PLAN_SCHEMA_VERSION",
        "STUDY_ASSIGNMENT_SCHEMA_VERSION",
        "REHEARSAL_RECEIPT_SCHEMA_VERSION",
        "REHEARSAL_SUPPORT_PROPOSAL_SCHEMA_VERSION",
        "STUDY_ID",
        "STUDY_VERSION",
    ),
    "source/route_policy_protocol.py": (
        "PROTOCOL_BUILD_INPUT_SCHEMA_VERSION",
        "PROTOCOL_SCHEMA_VERSION",
        "PROTOCOL_VERSION",
        "REVIEW_BUNDLE_SCHEMA_VERSION",
        "REVIEW_BUNDLE_VERSION",
        "TARGET_POLICY_CLASS_SCHEMA_VERSION",
        "POPULATION_SCHEMA_VERSION",
        "STATEFUL_DESIGN_SCHEMA_VERSION",
        "STOPPING_SCHEMA_VERSION",
        "RANDOMNESS_SCHEMA_VERSION",
    ),
    "source/route_policy_shadow_registry.py": (
        "SHADOW_REGISTRY_SCHEMA_VERSION",
        "SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION",
        "SHADOW_DESIGN_BINDING_SCHEMA_VERSION",
        "SHADOW_ASSIGNMENT_MANIFEST_SCHEMA_VERSION",
        "SHADOW_CAMPAIGN_SEAL_SCHEMA_VERSION",
        "SHADOW_SEED_CAPSULE_SCHEMA_VERSION",
        "SHADOW_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION",
        "SHADOW_CAMPAIGN_CLOSURE_SCHEMA_VERSION",
        "SHADOW_SEED_REVEAL_SCHEMA_VERSION",
        "SHADOW_ASSIGNMENT_REVEAL_SCHEMA_VERSION",
        "SHADOW_REGISTRY_EVENT_SCHEMA_VERSION",
        "SHADOW_REGISTRY_SNAPSHOT_SCHEMA_VERSION",
        "SHADOW_ASSIGNMENT_ALGORITHM",
        "SHADOW_CANONICALIZATION",
        "SHADOW_LEGACY_BUNDLE_ENCODING",
        "SHADOW_SCHEMA_OBJECTS_SHA256",
        "SHADOW_TOTAL_ALLOCATION_BPS",
        "SHADOW_CANDIDATE_ALLOCATION_BPS",
        "SHADOW_BLOCK_ID",
    ),
}


def _canonical_module_bytes(path: Path) -> bytes:
    """Return the Git-canonical text payload for a runtime module.

    Every declared runtime module is a Python text file covered by the
    repository's ``text=auto`` policy.  Git stores those files with LF line
    endings but may materialize CRLF in a Windows worktree.  Hashing the raw
    checkout made the supposedly deterministic manifest depend on the
    developer's ``core.autocrlf`` setting, so mirror Git's CRLF-to-LF clean
    conversion before recording the digest and size.
    """

    return path.read_bytes().replace(b"\r\n", b"\n")


def _validate_python_source(path: Path) -> None:
    try:
        # ``compile`` accepts bytes, so Python's normal source encoding rules
        # (including an UTF-8 BOM) apply exactly as they do at runtime.
        compile(path.read_bytes(), str(path), "exec", ast.PyCF_ONLY_AST, dont_inherit=True)
    except (SyntaxError, UnicodeError) as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise ValueError(f"required Studio runtime module is not valid Python: {path}: {detail}") from exc


def _literal_constants(path: Path, names: Iterable[str]) -> Dict[str, Any]:
    wanted = set(names)
    found: Dict[str, Any] = {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value
        for target in targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                try:
                    found[target.id] = ast.literal_eval(value_node)
                except (TypeError, ValueError):
                    pass
    missing = sorted(wanted - set(found))
    if missing:
        raise ValueError(f"{path.as_posix()} is missing literal contract constants: {', '.join(missing)}")
    return {name: found[name] for name in sorted(found)}


def _git_provenance(repo_root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = run("rev-parse", "HEAD") or None
    branch = run("branch", "--show-current") or None
    dirty = bool(run("status", "--porcelain"))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def build_manifest(repo_root: Path, *, include_git: bool = False) -> Dict[str, Any]:
    root = repo_root.resolve()
    modules = []
    for relative in RUNTIME_MODULES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required Studio runtime module is missing: {relative}")
        _validate_python_source(path)
        canonical_payload = _canonical_module_bytes(path)
        modules.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(canonical_payload).hexdigest(),
                "size_bytes": len(canonical_payload),
            }
        )

    contracts: Dict[str, Any] = {}
    for relative, names in CONTRACT_CONSTANTS.items():
        contracts[relative] = _literal_constants(root / relative, names)

    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "app_version": STUDIO_APP_VERSION,
        "active_runtime_tree": "source",
        "legacy_compatibility_tree": "runtime_python",
        "entrypoints": {
            "desktop": "source/supermix_multimodel_desktop_app.py",
            "web": "source/supermix_multimodel_web_app.py",
            "route_study_console": "source/route_policy_protocol_cli.py",
            "route_shadow_console": "source/route_policy_shadow_cli.py",
        },
        "expected_windows_artifacts": {
            "desktop_executable": "SupermixStudioDesktop.exe",
            "route_study_console": "SupermixRouteStudy.exe",
            "route_shadow_console": "SupermixRouteShadow.exe",
            "installer": "SupermixStudioDesktopSetup.exe",
        },
        "modules": modules,
        "contracts": contracts,
        "package_guards": {
            "route_protocol_activation_available": False,
            "route_protocol_assignment_implementation_available": False,
            "route_rehearsal_writes_ledger": False,
            "route_ledger_requires_executed_assignment_namespace": True,
            "route_ledger_requires_issued_execution_assignment_record": True,
            "route_review_bundle_full_source_reconstruction": True,
            "route_review_bundle_authenticity_proof_available": False,
            "route_review_bundle_trusted_timestamp_available": False,
            "route_shadow_registry_available": True,
            "route_shadow_assignment_executes_routes": False,
            "route_shadow_private_seed_persisted_before_reveal": False,
            "route_shadow_authenticity_proof_available": False,
            "route_shadow_trusted_timestamp_available": False,
            "automatic_policy_promotion_available": False,
            "progressive_auto_compute_accepted_probe_reuse": True,
            "mutual_stability_shadow_can_select_output": False,
        },
    }
    if include_git:
        manifest["release_provenance"] = _git_provenance(root)
    return manifest


def _render(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default="", help="repository root; defaults to this script's parent")
    parser.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH), help="manifest path relative to repo root")
    parser.add_argument("--check", action="store_true", help="fail if the checked manifest is stale")
    parser.add_argument(
        "--release-provenance",
        action="store_true",
        help="include commit, branch, and dirty state for a build-only release manifest",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = (
        Path(args.repo_root).expanduser().resolve()
        if str(args.repo_root).strip()
        else Path(__file__).resolve().parents[1]
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    try:
        rendered = _render(build_manifest(repo_root, include_git=bool(args.release_provenance)))
        if args.check:
            current = output.read_text(encoding="utf-8")
            if current != rendered:
                print(
                    f"Studio runtime manifest is stale: {output}. Regenerate it before packaging.",
                    file=sys.stderr,
                )
                return 1
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(f"Studio runtime manifest error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
