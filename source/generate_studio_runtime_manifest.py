"""Generate the deterministic Supermix Studio runtime distribution contract."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


MANIFEST_SCHEMA_VERSION = "supermix-studio-runtime-manifest-v1"
STUDIO_APP_VERSION = "71.0.0"
DEFAULT_MANIFEST_PATH = Path("source/studio_runtime_manifest.json")

RUNTIME_MODULES = (
    "source/chat_app.py",
    "source/chat_export.py",
    "source/chat_image_variant_app.py",
    "source/chat_memory.py",
    "source/chat_pipeline.py",
    "source/chat_web_app.py",
    "source/conversation_directive.py",
    "source/conversation_state.py",
    "source/dcgan_image_model.py",
    "source/device_utils.py",
    "source/grounding_runtime.py",
    "source/image_feature_utils.py",
    "source/image_recognition_model.py",
    "source/interaction_planner.py",
    "source/llm_database.py",
    "source/math_equation_model.py",
    "source/mattergen_generation_model.py",
    "source/memory_authority.py",
    "source/mesh_feature_utils.py",
    "source/model_frontier_native_image_v36.py",
    "source/model_frontier_v33.py",
    "source/model_frontier_v35.py",
    "source/model_frontier_v39.py",
    "source/model_native_image_lite_v37.py",
    "source/model_native_image_xlite_v38.py",
    "source/model_variants.py",
    "source/multimodel_catalog.py",
    "source/multimodel_memory.py",
    "source/multimodel_runtime.py",
    "source/multimodel_tools.py",
    "source/native_image_infer_v36.py",
    "source/native_image_infer_v37_lite.py",
    "source/native_image_infer_v38_xlite.py",
    "source/omni_collective_model.py",
    "source/omni_collective_v3_model.py",
    "source/omni_collective_v41_model.py",
    "source/omni_collective_v42_model.py",
    "source/omni_collective_v4_model.py",
    "source/omni_collective_v5_model.py",
    "source/omni_collective_v6_model.py",
    "source/omni_collective_v7_model.py",
    "source/omni_collective_v8_model.py",
    "source/protein_folding_model.py",
    "source/prompt_understanding.py",
    "source/qwen_adapter_promotion.py",
    "source/qwen_chat_desktop_app.py",
    "source/qwen_chat_web_app.py",
    "source/reasoning_engine.py",
    "source/science_plan.py",
    "source/route_policy_ledger.py",
    "source/route_policy_lab.py",
    "source/route_policy_explorer.py",
    "source/route_policy_protocol.py",
    "source/route_policy_shadow_registry.py",
    "source/route_policy_study_cli.py",
    "source/route_policy_protocol_cli.py",
    "source/route_policy_shadow_cli.py",
    "source/run.py",
    "source/score_fusion.py",
    "source/supermix_multimodel_web_app.py",
    "source/supermix_multimodel_desktop_app.py",
    "source/three_d_generation_model.py",
    "source/v40_benchmax_common.py",
    "source/video_feature_utils.py",
    "runtime_python/chat_app.py",
    "runtime_python/chat_web_app.py",
    "runtime_python/conversation_state.py",
    "runtime_python/memory_authority.py",
    "runtime_python/score_fusion.py",
    "runtime_python/grounding_runtime.py",
    "runtime_python/interaction_planner.py",
    "runtime_python/prompt_understanding.py",
    "runtime_python/reasoning_engine.py",
    "runtime_python/science_plan.py",
    "runtime_python/chat_pipeline.py",
    "runtime_python/chat_memory.py",
    "runtime_python/device_utils.py",
    "runtime_python/llm_database.py",
    "runtime_python/model_variants.py",
    "runtime_python/run.py",
)

STUDIO_RUNTIME_ENTRYPOINTS = (
    "source/supermix_multimodel_desktop_app.py",
    "source/supermix_multimodel_web_app.py",
    "source/route_policy_protocol_cli.py",
    "source/route_policy_shadow_cli.py",
)
RUNTIME_COMPATIBILITY_ENTRYPOINTS = (
    "runtime_python/chat_app.py",
    "runtime_python/chat_web_app.py",
)
DISTRIBUTION_RUNTIME_ENTRYPOINTS = (
    *STUDIO_RUNTIME_ENTRYPOINTS,
    *RUNTIME_COMPATIBILITY_ENTRYPOINTS,
)
DISTRIBUTION_RUNTIME_CLOSURE_ROOTS = tuple(
    dict.fromkeys((*DISTRIBUTION_RUNTIME_ENTRYPOINTS, *RUNTIME_MODULES))
)

# These modules are imported lazily by shared model metadata helpers, but only
# when constructing training datasets. They are not reachable from Studio
# inference or either packaged route console. Keeping the exclusions explicit
# lets the closure check fail closed for every other newly imported local file.
STUDIO_RUNTIME_IMPORT_EXCLUSIONS = {
    "source/build_reasoning_benchmix_v39.py": (
        "lazy v39 training-dataset builder used only by "
        "v40_benchmax_common.build_v39_style_rows"
    ),
    "source/build_v33_frontier_dataset.py": (
        "lazy v33 training-dataset builder used only by "
        "v40_benchmax_common.build_v33_style_rows"
    ),
}

CONTRACT_CONSTANTS = {
    "source/chat_app.py": (
        "AUTO_COMPUTE_PLAN_SCHEMA_VERSION",
        "AUTO_COMPUTE_STRATEGY",
        "DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K",
        "DEFAULT_PREDICTION_STABILITY_MARGIN",
        "DEFAULT_PREDICTION_STABILITY_RANK_DEPTH",
    ),
    "source/grounding_runtime.py": (
        "GROUNDING_SCHEMA_VERSION",
        "GROUNDING_RUNTIME_VERSION",
        "VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION",
    ),
    "runtime_python/grounding_runtime.py": (
        "GROUNDING_SCHEMA_VERSION",
        "GROUNDING_RUNTIME_VERSION",
        "VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION",
    ),
    "source/interaction_planner.py": ("PLANNER_VERSION",),
    "runtime_python/interaction_planner.py": ("PLANNER_VERSION",),
    "source/prompt_understanding.py": (
        "PROMPT_UNDERSTANDING_SCHEMA_VERSION",
        "PROMPT_UNDERSTANDING_VERSION",
    ),
    "runtime_python/prompt_understanding.py": (
        "PROMPT_UNDERSTANDING_SCHEMA_VERSION",
        "PROMPT_UNDERSTANDING_VERSION",
    ),
    "source/reasoning_engine.py": (
        "FINITE_BERNOULLI_SCHEMA_VERSION",
        "REASONING_SCHEMA_VERSION",
        "REASONING_ENGINE_VERSION",
    ),
    "runtime_python/reasoning_engine.py": (
        "FINITE_BERNOULLI_SCHEMA_VERSION",
        "REASONING_SCHEMA_VERSION",
        "REASONING_ENGINE_VERSION",
    ),
    "source/science_plan.py": (
        "SCIENCE_FORMULA_REGISTRY_VERSION",
        "SCIENCE_PLAN_ENGINE_VERSION",
        "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION",
        "SCIENCE_PLAN_SCHEMA_VERSION",
    ),
    "runtime_python/science_plan.py": (
        "SCIENCE_FORMULA_REGISTRY_VERSION",
        "SCIENCE_PLAN_ENGINE_VERSION",
        "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION",
        "SCIENCE_PLAN_SCHEMA_VERSION",
    ),
    "source/memory_authority.py": (
        "MEMORY_AUTHORITY_SCHEMA_VERSION",
        "MEMORY_AUTHORITY_POLICY_VERSION",
        "MEMORY_EXTRACTION_RULE_VERSION",
    ),
    "runtime_python/memory_authority.py": (
        "MEMORY_AUTHORITY_SCHEMA_VERSION",
        "MEMORY_AUTHORITY_POLICY_VERSION",
        "MEMORY_EXTRACTION_RULE_VERSION",
    ),
    "source/conversation_state.py": (
        "CONVERSATION_STATE_SCHEMA_VERSION",
        "CONVERSATION_STATE_VERSION",
    ),
    "runtime_python/conversation_state.py": (
        "CONVERSATION_STATE_SCHEMA_VERSION",
        "CONVERSATION_STATE_VERSION",
    ),
    "source/conversation_directive.py": (
        "CONVERSATION_DIRECTIVE_SCHEMA_VERSION",
        "CONVERSATION_DIRECTIVE_VERSION",
    ),
    "source/qwen_adapter_promotion.py": (
        "BENCHMARK_SCHEMA_VERSION",
        "PROMOTION_SCHEMA_VERSION",
        "GATE_SCHEMA_VERSION",
        "PRODUCTION_POLICY_ID",
        "SUPPORTED_VERIFIER_SCHEMA_VERSION",
    ),
    "source/multimodel_memory.py": ("MEMORY_SCHEMA_VERSION",),
    "source/score_fusion.py": (
        "SCORE_FUSION_SCHEMA_VERSION",
        "SCORE_FUSION_VERSION",
    ),
    "runtime_python/score_fusion.py": (
        "SCORE_FUSION_SCHEMA_VERSION",
        "SCORE_FUSION_VERSION",
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


def _parse_python_source(path: Path) -> ast.Module:
    try:
        # ``compile`` accepts bytes, so Python's normal source encoding rules
        # (including an UTF-8 BOM) apply exactly as they do at runtime.
        tree = compile(
            path.read_bytes(),
            str(path),
            "exec",
            ast.PyCF_ONLY_AST,
            dont_inherit=True,
        )
    except (SyntaxError, UnicodeError) as exc:
        detail = str(exc).strip() or exc.__class__.__name__
        raise ValueError(f"required Studio runtime module is not valid Python: {path}: {detail}") from exc
    if not isinstance(tree, ast.Module):  # pragma: no cover - compile contract
        raise ValueError(f"required Studio runtime module did not parse as a module: {path}")
    return tree


def _validate_python_source(path: Path) -> None:
    _parse_python_source(path)


def _resolve_local_module(
    repo_root: Path,
    current_path: Path,
    module_name: str,
    *,
    level: int = 0,
) -> Optional[Path]:
    """Resolve one import inside the current source or compatibility tree."""

    raw_name = str(module_name or "").strip(".")
    if level > 0:
        base = current_path.parent
        for _ in range(level - 1):
            base = base.parent
        module_base = base.joinpath(*([part for part in raw_name.split(".") if part]))
    else:
        parts = [part for part in raw_name.split(".") if part]
        if parts[:1] in (["source"], ["runtime_python"]):
            module_base = repo_root.joinpath(*parts)
        else:
            try:
                current_tree = current_path.resolve().relative_to(repo_root.resolve()).parts[0]
            except (IndexError, ValueError):
                return None
            if current_tree not in {"source", "runtime_python"}:
                return None
            module_base = (repo_root / current_tree).joinpath(*parts)

    candidates = (module_base.with_suffix(".py"), module_base / "__init__.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _local_imports(repo_root: Path, current_path: Path, tree: ast.Module) -> tuple[Path, ...]:
    resolved: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                candidate = _resolve_local_module(repo_root, current_path, alias.name)
                if candidate is not None:
                    resolved.add(candidate)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue

        module_name = str(node.module or "")
        candidate = _resolve_local_module(
            repo_root,
            current_path,
            module_name,
            level=int(node.level or 0),
        )
        if candidate is not None:
            resolved.add(candidate)

        # ``from . import helper`` and imports from a local package may name a
        # submodule only in the alias rather than in ``node.module``.
        for alias in node.names:
            if alias.name == "*":
                continue
            alias_module = ".".join(part for part in (module_name, alias.name) if part)
            alias_candidate = _resolve_local_module(
                repo_root,
                current_path,
                alias_module,
                level=int(node.level or 0),
            )
            if alias_candidate is not None:
                resolved.add(alias_candidate)
    return tuple(sorted(resolved, key=lambda path: path.as_posix()))


def _default_runtime_closure_roots(runtime_modules: Sequence[str]) -> tuple[str, ...]:
    """Build closure roots from the active manifest configuration.

    Keeping this calculation dynamic matters for both release tooling and the
    isolated tamper tests: callers may supply a reduced module set, while a
    real distribution that declares any compatibility modules must still root
    the compatibility entry points explicitly.
    """

    declared = tuple(Path(relative).as_posix() for relative in runtime_modules)
    compatibility_roots: Sequence[str] = ()
    if any(relative.startswith("runtime_python/") for relative in declared):
        compatibility_roots = RUNTIME_COMPATIBILITY_ENTRYPOINTS
    return tuple(
        dict.fromkeys((*STUDIO_RUNTIME_ENTRYPOINTS, *compatibility_roots, *declared))
    )


def discover_studio_runtime_import_closure(
    repo_root: Path,
    *,
    runtime_modules: Optional[Sequence[str]] = None,
    entrypoints: Optional[Sequence[str]] = None,
    exclusions: Optional[Mapping[str, str]] = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return recursive local imports and the explicit exclusions encountered."""

    root = repo_root.resolve()
    configured_modules = RUNTIME_MODULES if runtime_modules is None else runtime_modules
    configured_entrypoints = (
        _default_runtime_closure_roots(configured_modules)
        if entrypoints is None
        else entrypoints
    )
    configured_exclusions = (
        STUDIO_RUNTIME_IMPORT_EXCLUSIONS if exclusions is None else exclusions
    )
    normalized_exclusions: Dict[str, str] = {}
    for relative, reason in configured_exclusions.items():
        normalized = Path(relative).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Studio runtime import exclusion must stay inside the repository: {relative}")
        if not str(reason).strip():
            raise ValueError(f"Studio runtime import exclusion requires a justification: {normalized}")
        normalized_exclusions[normalized] = str(reason).strip()

    pending: list[Path] = []
    for relative in configured_entrypoints:
        entrypoint = (root / relative).resolve()
        if not entrypoint.is_file():
            raise FileNotFoundError(f"required Studio runtime entrypoint is missing: {relative}")
        pending.append(entrypoint)

    visited: set[Path] = set()
    encountered_exclusions: set[str] = set()
    while pending:
        path = pending.pop()
        relative = path.relative_to(root).as_posix()
        if relative in normalized_exclusions:
            encountered_exclusions.add(relative)
            continue
        if path in visited:
            continue
        visited.add(path)
        pending.extend(_local_imports(root, path, _parse_python_source(path)))

    closure = tuple(sorted(path.relative_to(root).as_posix() for path in visited))
    return closure, tuple(sorted(encountered_exclusions))


def validate_studio_runtime_import_closure(
    repo_root: Path,
    *,
    runtime_modules: Optional[Sequence[str]] = None,
    entrypoints: Optional[Sequence[str]] = None,
    exclusions: Optional[Mapping[str, str]] = None,
) -> tuple[str, ...]:
    """Fail when a Studio entrypoint imports unmanifested local runtime code."""

    configured_modules = RUNTIME_MODULES if runtime_modules is None else runtime_modules
    configured_exclusions = (
        STUDIO_RUNTIME_IMPORT_EXCLUSIONS if exclusions is None else exclusions
    )
    closure, encountered_exclusions = discover_studio_runtime_import_closure(
        repo_root,
        runtime_modules=configured_modules,
        entrypoints=entrypoints,
        exclusions=configured_exclusions,
    )
    declared = {Path(relative).as_posix() for relative in configured_modules}
    excluded = {Path(relative).as_posix() for relative in configured_exclusions}
    overlap = sorted(declared & excluded)
    if overlap:
        raise ValueError(
            "Studio runtime modules cannot also be import-closure exclusions: "
            + ", ".join(overlap)
        )

    missing = sorted(set(closure) - declared)
    if missing:
        raise ValueError(
            "Studio runtime manifest omits recursively imported local module(s): "
            + ", ".join(missing)
        )

    unused_exclusions = sorted(excluded - set(encountered_exclusions))
    if unused_exclusions:
        raise ValueError(
            "Studio runtime import exclusion(s) are stale or unreachable: "
            + ", ".join(unused_exclusions)
        )
    return closure


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
    validate_studio_runtime_import_closure(root)
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
            "qwen_adapter_promotion_enforced_in_studio": True,
            "qwen_adapter_production_policy_pinned": True,
            "verified_probabilistic_scenarios_available": True,
            "verified_probabilistic_scenarios_open_world_authority": False,
            "verified_scientific_scenarios_available": True,
            "science_plan_executes_model_code": False,
            "science_plan_open_world_authority": False,
            "science_plan_requires_explicit_assumptions": True,
            "science_plan_receipts_carry_prompt_or_source_text": False,
            "science_plan_public_receipts_carry_prompt_derived_digests": False,
            "progressive_auto_compute_accepted_probe_reuse": True,
            "mutual_stability_shadow_can_select_output": False,
            "prediction_stability_margin_guard_available": True,
            "prediction_stability_margin_default_is_universal": False,
            "prediction_stability_rank_depth_guard_available": True,
            "prediction_stability_rank_depth_default_is_universal": False,
            "prediction_stability_rank_depth_zero_disables_verifier": True,
            "prediction_stability_decision_margin_telemetry": True,
            "prediction_verifier_telemetry_requires_active_verifier": True,
            "full_output_verifier_includes_post_head_calibration": True,
            "prediction_stability_verifier_scoped_to_available_labels": True,
            "memory_authority_firewall_available": True,
            "memory_relevance_can_elevate_authority": False,
            "memory_claims_can_authorize_tools": False,
            "memory_factual_claims_enter_shared_planning_context": False,
            "memory_legacy_rows_prompt_eligible": False,
            "memory_content_digests_are_authentication": False,
            "assistant_memory_examples_automatically_injected": False,
            "verified_answer_receipts_available": True,
            "verified_answer_receipts_carry_prompt_or_answer_text": False,
            "verified_answer_receipts_carry_expression_proof_steps_or_evidence": False,
            "verified_answer_receipts_control_routes_or_permissions": False,
            "verified_answer_receipts_control_tools_or_safety": False,
            "verified_answer_receipts_control_runtime_authority": False,
            "distribution_recursive_import_closure_enforced": True,
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
