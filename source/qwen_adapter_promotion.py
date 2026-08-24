"""Content-addressed promotion receipts for Qwen adapters.

Training candidates are intentionally inert.  The GUI may prefer a candidate
only when its adapter weights, configuration, benchmark receipt, and explicit
``passed`` decision all match this manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional


BENCHMARK_SCHEMA_VERSION = "supermix-qwen-evaluation-v4"
PROMOTION_SCHEMA_VERSION = "supermix-qwen-adapter-promotion-v4"
GATE_SCHEMA_VERSION = "supermix-qwen-general-promotion-gate-v4"
PRODUCTION_POLICY_ID = "supermix-qwen-production-promotion-policy-v4"
PROMOTION_FILENAME = "promotion_manifest.json"
GATE_FILENAME = "promotion_gate.json"
SUPPORTED_VERIFIER_SCHEMA_VERSION = "supermix-verifier-v2"
SUPPORTED_PAIRED_EVIDENCE_SCHEMA_VERSION = "supermix-qwen-paired-evidence-v1"
PRODUCTION_THRESHOLD_FLOORS = MappingProxyType(
    {
        "min_verified_samples": 20,
        "min_verified_gain": 0.05,
        "min_tuned_accuracy": 0.20,
        "max_family_regression": 0.0,
        "max_loss_ratio": 1.05,
        "min_token_f1_delta": -0.02,
        "min_family_verified_samples": 1,
        "max_generation_cap_rate": 0.05,
        "max_paired_p_value": 0.05,
        "max_paired_regression_rate": 0.02,
        "min_paired_cluster_lower_bound": 0.0,
        "min_template_clusters": 5,
    }
)
PRODUCTION_PROTOCOL = MappingProxyType(
    {
        "curriculum_schema": "supermix-general-intelligence-curriculum-v3",
        "curriculum_seed": 6201,
        "curriculum_eval_rows": 150,
        "curriculum_eval_sha256": (
            "45a84eb8e95f2a687b8c8ab951e8c687948446f0c23266f4550671f3095c7617"
        ),
        "selection_seed": 6201,
        "samples_per_family": 2,
        "max_eval_samples": 0,
        "selected_eval_samples": 34,
        "max_length": 256,
        "max_new_tokens": 64,
        "paired_bootstrap_seed": 5203,
        "paired_bootstrap_resamples": 5_000,
    }
)

_EVALUATOR_FILES = (
    "run_research_baseline.py",
    "qwen_supermix_pipeline.py",
    "qwen_paired_evidence.py",
)
_VERIFIER_FILES = (
    "verifiable_reasoning.py",
    "logical_entailment.py",
)
_POLICY_FILES = (
    "run_qwen_general_promotion_gate.py",
    "qwen_adapter_promotion.py",
    "qwen_paired_evidence.py",
)
LEGACY_ADAPTER_ARTIFACT_PREFIXES = (
    "qwen_supermix_enhanced_",
    "qwen_supermix_auto_verify_",
)
# Receipt-free compatibility is deliberately content-addressed. Prefixes are
# retained only as descriptive namespace metadata; they never confer trust.
LEGACY_ADAPTER_ALLOWLIST: Mapping[str, Mapping[str, str]] = {
    "qwen_supermix_enhanced_v25_selective_pref": {
        "adapter_config_sha256": "c1e0177e529a83ffa042cc58f4a885cc45b4e02ac7fba530e8ce7eddf66faba3",
        "adapter_sha256": "67606005e5fce4ee240014a2d59c676583f3dd3ecc3eca0a684a1e63bda7a559",
    },
    "qwen_supermix_auto_verify_20260320": {
        "adapter_config_sha256": "3cf014103cd4c351a9ef04ad356a0b03ae4c91ea16e514f4d921507091af982d",
        "adapter_sha256": "d9f0596471014746db13ea6306c4032cd7d3c9f6add937fa5a4fbc4bd73b4b17",
    }
}
_BOUND_GATE_FIELDS = (
    "benchmark_schema",
    "benchmark_sha256",
    "curriculum_manifest_sha256",
    "curriculum_eval_sha256",
    "selected_eval_sha256",
    "base_samples_sha256",
    "tuned_samples_sha256",
    "sample_comparison_sha256",
    "paired_evidence_sha256",
    "paired_evidence_schema",
    "policy_id",
    "policy_mode",
    "production_eligible",
    "verifier_schema",
)
_MINIMUM_THRESHOLD_KEYS = frozenset(
    {
        "min_verified_samples",
        "min_verified_gain",
        "min_tuned_accuracy",
        "min_token_f1_delta",
        "min_family_verified_samples",
        "min_paired_cluster_lower_bound",
        "min_template_clusters",
    }
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_code_hashes(source_dir: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Hash the exact evaluator and verifier implementations used by a receipt."""

    root = Path(source_dir).resolve() if source_dir is not None else Path(__file__).resolve().parent
    return {
        "evaluator": {name: sha256_file(root / name) for name in _EVALUATOR_FILES},
        "verifier": {name: sha256_file(root / name) for name in _VERIFIER_FILES},
        "policy": {name: sha256_file(root / name) for name in _POLICY_FILES},
    }


def load_json_mapping(path: Path | str) -> Optional[dict[str, object]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_child(parent: Path, filename: object) -> Optional[Path]:
    raw = str(filename or "").strip()
    if not raw or Path(raw).name != raw:
        return None
    candidate = (parent / raw).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError:
        return None
    return candidate


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text)


def _same_hash(left: object, right: object) -> bool:
    return _is_sha256(left) and str(left).lower() == str(right or "").lower()


def _valid_code_hashes(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    expected_files = {
        "evaluator": frozenset(_EVALUATOR_FILES),
        "verifier": frozenset(_VERIFIER_FILES),
        "policy": frozenset(_POLICY_FILES),
    }
    if set(value) != set(expected_files):
        return False
    for group, expected_names in expected_files.items():
        rows = value.get(group)
        if not isinstance(rows, Mapping) or set(rows) != set(expected_names):
            return False
        if any(not str(name or "").strip() or not _is_sha256(digest) for name, digest in rows.items()):
            return False
    return True


def _supported_production_thresholds(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    for key, floor in PRODUCTION_THRESHOLD_FLOORS.items():
        candidate = value.get(key)
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            return False
        if not math.isfinite(float(candidate)):
            return False
        if key in _MINIMUM_THRESHOLD_KEYS:
            if float(candidate) < float(floor):
                return False
        elif float(candidate) > float(floor):
            return False
    return True


def _adapter_namespace_token(value: object) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _is_candidate_namespace(value: object) -> bool:
    token = _adapter_namespace_token(value)
    token_parts = set(token.split("_"))
    return bool({"candidate", "candidates"} & token_parts or "general_intelligence" in token)


def _canonical_huggingface_identity(value: object) -> tuple[str, str]:
    """Return ``(repo_id, revision)`` for a repo ID or HF snapshot path."""

    raw = str(value or "").strip().replace("\\", "/").rstrip("/")
    if not raw:
        return "", ""
    parts = [part for part in raw.split("/") if part]
    lowered = [part.casefold() for part in parts]
    if "snapshots" in lowered:
        snapshot_index = len(lowered) - 1 - lowered[::-1].index("snapshots")
        revision = parts[snapshot_index + 1] if snapshot_index + 1 < len(parts) else ""
        cache_token = next(
            (
                part
                for part in reversed(parts[:snapshot_index])
                if part.casefold().startswith("models--")
            ),
            "",
        )
        repo_id = cache_token[len("models--") :].replace("--", "/") if cache_token else ""
        return repo_id, revision
    return raw, ""


def _adapter_config_matches_receipt(
    config: Mapping[str, object],
    *,
    base_model: object,
    base_model_revision: object,
) -> bool:
    configured_id, configured_revision = _canonical_huggingface_identity(
        config.get("base_model_name_or_path")
    )
    expected_id = str(base_model or "").strip().rstrip("/")
    expected_revision = str(base_model_revision or "").strip()
    if not configured_id or configured_id.casefold() != expected_id.casefold():
        return False
    if configured_revision:
        return bool(expected_revision) and configured_revision.casefold() == expected_revision.casefold()
    return True


def _recognized_legacy_identity(
    adapter_dir: Path,
    *,
    legacy_artifact_name: str,
) -> Optional[tuple[str, str, str]]:
    config_path = adapter_dir / "adapter_config.json"
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        return None
    namespace_values = _adapter_namespace_values(adapter_dir, legacy_artifact_name)
    config_sha256 = sha256_file(config_path)
    adapter_sha256 = sha256_file(weights_path)
    for value in namespace_values:
        token = _adapter_namespace_token(value)
        expected = LEGACY_ADAPTER_ALLOWLIST.get(token)
        if expected is None:
            continue
        if _same_hash(expected.get("adapter_config_sha256"), config_sha256) and _same_hash(
            expected.get("adapter_sha256"), adapter_sha256
        ):
            return (
                token,
                str(expected["adapter_config_sha256"]).lower(),
                str(expected["adapter_sha256"]).lower(),
            )
    return None


def _has_promotion_receipt(adapter_dir: Path) -> bool:
    artifact_dir = adapter_dir.parent
    return any((artifact_dir / filename).exists() for filename in (PROMOTION_FILENAME, GATE_FILENAME))


def _adapter_namespace_values(adapter_dir: Path, legacy_artifact_name: str) -> tuple[str, ...]:
    parts = adapter_dir.parts
    boundary = -1
    for index, part in enumerate(parts):
        if _adapter_namespace_token(part) in {"artifacts", "output", "bundled_models"}:
            boundary = index
    scoped_parts = parts[boundary + 1 :] if boundary >= 0 else parts[-4:]
    return (*scoped_parts, legacy_artifact_name)


def _legacy_activation_identity(
    adapter_dir: Path,
    *,
    legacy_artifact_name: str,
) -> Optional[tuple[str, str, str]]:
    if _has_promotion_receipt(adapter_dir):
        return None
    namespace_values = _adapter_namespace_values(adapter_dir, legacy_artifact_name)
    if any(_is_candidate_namespace(value) for value in namespace_values):
        return None
    return _recognized_legacy_identity(
        adapter_dir,
        legacy_artifact_name=legacy_artifact_name,
    )


def _legacy_activation_kind(adapter_dir: Path, *, legacy_artifact_name: str) -> Optional[str]:
    identity = _legacy_activation_identity(
        adapter_dir,
        legacy_artifact_name=legacy_artifact_name,
    )
    return "legacy" if identity is not None else None


def adapter_activation_kind(
    adapter_dir: Path | str,
    *,
    legacy_artifact_name: str = "",
) -> Optional[str]:
    """Classify an adapter for implicit runtime activation.

    Explicit standalone Qwen paths remain a user-controlled compatibility
    escape hatch. Automatic selectors and Studio should call this classifier:
    content-validated promotions win, invalid receipts are revocations, and
    only exact content hashes for known historical artifacts receive
    receipt-free compatibility treatment.
    """

    adapter = Path(adapter_dir).expanduser().resolve()
    if validate_promoted_adapter(adapter) is not None:
        return "promoted"
    return _legacy_activation_kind(
        adapter,
        legacy_artifact_name=legacy_artifact_name,
    )


def attest_adapter_for_runtime(
    adapter_dir: Path | str,
    *,
    legacy_artifact_name: str = "",
    resolved_base_model: Path | str = "",
) -> dict[str, object]:
    """Return fail-closed trust metadata for an implicitly loaded adapter.

    The resolved base must be the receipt's exact Hugging Face repository and
    immutable ``snapshots/<revision>`` directory (or the exact repository ID
    when revision resolution is intentionally deferred). Unproven local model
    copies fail closed.
    """

    adapter = Path(adapter_dir).expanduser().resolve()
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError(f"Adapter weights or configuration are missing: {adapter}")

    manifest = validate_promoted_adapter(adapter)
    legacy_identity = (
        None
        if manifest is not None
        else _legacy_activation_identity(adapter, legacy_artifact_name=legacy_artifact_name)
    )
    activation_kind = "promoted" if manifest is not None else (
        "legacy" if legacy_identity is not None else None
    )
    if activation_kind is None:
        raise ValueError(
            "Adapter is not eligible for implicit runtime activation: valid promotion "
            "receipts or recognized legacy provenance are required."
        )

    expected_adapter_sha256 = (
        str(manifest.get("adapter_sha256") or "").lower()
        if manifest is not None
        else str(legacy_identity[2] if legacy_identity is not None else "")
    )
    expected_config_sha256 = (
        str(manifest.get("adapter_config_sha256") or "").lower()
        if manifest is not None
        else str(legacy_identity[1] if legacy_identity is not None else "")
    )
    attestation: dict[str, object] = {
        "activation_kind": activation_kind,
        "trusted": True,
        "promotion_schema": "",
        "gate_schema": "",
        "adapter_sha256": expected_adapter_sha256,
        "adapter_config_sha256": expected_config_sha256,
        "base_model": "",
        "base_model_revision": "",
        "base_revision_status": "legacy_not_applicable",
    }
    if manifest is None:
        return attestation

    gate_path = adapter.parent / str(manifest.get("gate_file") or GATE_FILENAME)
    gate = load_json_mapping(gate_path) or {}
    expected_revision = str(manifest.get("base_model_revision") or "").strip()
    attestation.update(
        {
            "promotion_schema": str(manifest.get("schema") or ""),
            "gate_schema": str(gate.get("schema") or ""),
            "base_model": str(manifest.get("base_model") or ""),
            "base_model_revision": expected_revision,
            "base_revision_status": "not_checked",
        }
    )

    raw_base = str(resolved_base_model or "").strip()
    if raw_base:
        base_path = Path(raw_base).expanduser()
        if base_path.is_dir():
            resolved = base_path.resolve()
            resolved_model, resolved_revision = _canonical_huggingface_identity(resolved)
            expected_model = str(manifest.get("base_model") or "").strip()
            if (
                not resolved_model
                or resolved_model.casefold() != expected_model.casefold()
                or not expected_revision
                or resolved_revision.casefold() != expected_revision.casefold()
            ):
                raise ValueError(
                    "Promoted adapter base-model identity or revision does not match the resolved snapshot."
                )
            attestation["base_revision_status"] = "verified_snapshot"
        elif raw_base.casefold() == str(manifest.get("base_model") or "").casefold():
            attestation["base_revision_status"] = "model_id_match_revision_unresolved"
        else:
            raise ValueError("Promoted adapter base-model identity could not be established.")
    return attestation


def validate_promoted_adapter(adapter_dir: Path | str) -> Optional[dict[str, object]]:
    """Return a validated promotion manifest, or ``None`` on any mismatch."""

    adapter = Path(adapter_dir).expanduser().resolve()
    artifact_dir = adapter.parent
    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    manifest_path = artifact_dir / PROMOTION_FILENAME
    if not (config_path.is_file() and weights_path.is_file() and manifest_path.is_file()):
        return None
    manifest = load_json_mapping(manifest_path)
    if manifest is None:
        return None
    if manifest.get("schema") != PROMOTION_SCHEMA_VERSION or manifest.get("passed") is not True:
        return None
    config = load_json_mapping(config_path)
    if config is None:
        return None
    if not _same_hash(manifest.get("adapter_sha256"), sha256_file(weights_path)):
        return None
    if not _same_hash(manifest.get("adapter_config_sha256"), sha256_file(config_path)):
        return None

    gate_path = _safe_child(artifact_dir, manifest.get("gate_file"))
    if gate_path is None or not gate_path.is_file():
        return None
    if not _same_hash(manifest.get("gate_sha256"), sha256_file(gate_path)):
        return None
    gate = load_json_mapping(gate_path)
    if gate is None:
        return None
    if gate.get("schema") != GATE_SCHEMA_VERSION or gate.get("passed") is not True:
        return None
    decision = gate.get("decision")
    if not isinstance(decision, Mapping) or decision.get("passed") is not True:
        return None
    for payload in (manifest, gate, decision):
        if (
            payload.get("policy_id") != PRODUCTION_POLICY_ID
            or payload.get("policy_mode") != "production"
            or payload.get("production_eligible") is not True
            or payload.get("production_threshold_floors") != dict(PRODUCTION_THRESHOLD_FLOORS)
            or payload.get("production_protocol") != dict(PRODUCTION_PROTOCOL)
        ):
            return None
    blockers = decision.get("blockers")
    if not isinstance(blockers, list) or blockers:
        return None
    if not isinstance(decision.get("metrics"), Mapping) or not _supported_production_thresholds(
        decision.get("thresholds")
    ):
        return None
    if not _same_hash(gate.get("adapter_sha256"), manifest.get("adapter_sha256")):
        return None
    if not _same_hash(gate.get("adapter_config_sha256"), manifest.get("adapter_config_sha256")):
        return None
    if not str(gate.get("base_model") or "").strip() or str(gate.get("base_model") or "") != str(
        manifest.get("base_model") or ""
    ):
        return None
    if not str(gate.get("base_model_revision") or "").strip() or str(
        gate.get("base_model_revision") or ""
    ) != str(manifest.get("base_model_revision") or ""):
        return None
    if not _adapter_config_matches_receipt(
        config,
        base_model=manifest.get("base_model"),
        base_model_revision=manifest.get("base_model_revision"),
    ):
        return None
    if gate.get("benchmark_schema") != BENCHMARK_SCHEMA_VERSION:
        return None
    if gate.get("verifier_schema") != SUPPORTED_VERIFIER_SCHEMA_VERSION:
        return None
    if gate.get("paired_evidence_schema") != SUPPORTED_PAIRED_EVIDENCE_SCHEMA_VERSION:
        return None
    for field in _BOUND_GATE_FIELDS:
        gate_value = gate.get(field)
        manifest_value = manifest.get(field)
        if field.endswith("sha256"):
            if not _same_hash(gate_value, manifest_value):
                return None
        elif str(gate_value or "") != str(manifest_value or ""):
            return None
    code_hashes = gate.get("code_hashes")
    if not _valid_code_hashes(code_hashes) or code_hashes != manifest.get("code_hashes"):
        return None
    binding = decision.get("binding")
    if not isinstance(binding, Mapping):
        return None
    for field in (
        "base_model",
        "base_model_revision",
        "adapter_sha256",
        "adapter_config_sha256",
        *_BOUND_GATE_FIELDS,
    ):
        gate_value = gate.get(field)
        binding_value = binding.get(field)
        if field.endswith("sha256"):
            if not _same_hash(gate_value, binding_value):
                return None
        elif str(gate_value or "") != str(binding_value or ""):
            return None
    if binding.get("code_hashes") != code_hashes:
        return None
    return manifest


def is_promoted_adapter(adapter_dir: Path | str) -> bool:
    return validate_promoted_adapter(adapter_dir) is not None


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "GATE_FILENAME",
    "GATE_SCHEMA_VERSION",
    "LEGACY_ADAPTER_ALLOWLIST",
    "LEGACY_ADAPTER_ARTIFACT_PREFIXES",
    "PRODUCTION_POLICY_ID",
    "PRODUCTION_PROTOCOL",
    "PRODUCTION_THRESHOLD_FLOORS",
    "PROMOTION_FILENAME",
    "PROMOTION_SCHEMA_VERSION",
    "adapter_activation_kind",
    "attest_adapter_for_runtime",
    "evaluation_code_hashes",
    "is_promoted_adapter",
    "load_json_mapping",
    "sha256_file",
    "validate_promoted_adapter",
]
