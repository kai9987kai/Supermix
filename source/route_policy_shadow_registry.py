"""Append-only, shadow-only whole-policy assignment commitments.

The registry is deliberately separate from the executed route-decision ledger.
It can freeze a reviewed campaign, commit opaque cluster assignments, close
enrollment, reveal the seed, and reconstruct every committed whole-policy arm.
It cannot execute a route, emit a logging propensity, estimate a causal effect,
or promote a policy.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    from .route_policy_protocol import (
        PROTOCOL_OPTION_KEYS,
        audit_route_study_review_bundle,
        build_route_study_review_bundle,
    )
except ImportError:  # pragma: no cover - direct ``python source/...`` use
    from route_policy_protocol import (
        PROTOCOL_OPTION_KEYS,
        audit_route_study_review_bundle,
        build_route_study_review_bundle,
    )


SHADOW_REGISTRY_SCHEMA_VERSION = 1
SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION = "route-study-shadow-public-package-v1"
SHADOW_DESIGN_BINDING_SCHEMA_VERSION = "route-study-shadow-design-binding-v1"
SHADOW_ASSIGNMENT_MANIFEST_SCHEMA_VERSION = "route-study-shadow-assignment-manifest-v1"
SHADOW_CAMPAIGN_SEAL_SCHEMA_VERSION = "route-study-shadow-campaign-seal-v1"
SHADOW_SEED_CAPSULE_SCHEMA_VERSION = "route-study-shadow-seed-capsule-v1"
SHADOW_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION = "route-study-shadow-assignment-commitment-v1"
SHADOW_CAMPAIGN_CLOSURE_SCHEMA_VERSION = "route-study-shadow-campaign-closure-v1"
SHADOW_SEED_REVEAL_SCHEMA_VERSION = "route-study-shadow-seed-reveal-v1"
SHADOW_ASSIGNMENT_REVEAL_SCHEMA_VERSION = "route-study-shadow-assignment-reveal-v1"
SHADOW_REGISTRY_EVENT_SCHEMA_VERSION = "route-study-shadow-registry-event-v1"
SHADOW_REGISTRY_SNAPSHOT_SCHEMA_VERSION = "route-study-shadow-registry-snapshot-v1"

SHADOW_ASSIGNMENT_ALGORITHM = "hkdf-sha256-hmac-sha256-whole-policy-bps-v1"
SHADOW_CLUSTER_KEY_SCHEMA_VERSION = "session-hash-v1"
SHADOW_CANONICALIZATION = "rfc8785-jcs-restricted-ijson-integer-v1"
SHADOW_LEGACY_BUNDLE_ENCODING = "route-review-bundle-v1-python-sorted-compact-utf8"
SHADOW_SCHEMA_OBJECTS_SHA256 = "fe12a50841e5f983adce9ef1eb4ebf6635ab079f8c0043f4ca68500b38f9a04c"
SHADOW_TOTAL_ALLOCATION_BPS = 10_000
SHADOW_CANDIDATE_ALLOCATION_BPS = 5_000
SHADOW_BLOCK_ID = 0
MAX_SHADOW_CAMPAIGNS = 1_000
MAX_SHADOW_COMMITMENTS_PER_CAMPAIGN = 1_000_000
MAX_SHADOW_VERIFY_BATCH = 1_000

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BASE64URL_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")

_DESIGN_HASH_DOMAIN = b"supermix.route-shadow.design-binding.v1\x00"
_ARM_HASH_DOMAIN = b"supermix.route-shadow.whole-policy-arm.v1\x00"
_SOURCE_POLICY_CLASS_HASH_DOMAIN = b"supermix.route-shadow.source-policy-class.v1\x00"
_MANIFEST_HASH_DOMAIN = b"supermix.route-shadow.assignment-manifest.v1\x00"
_SEED_COMMITMENT_DOMAIN = b"supermix.route-shadow.seed-commitment.v1\x00"
_SEAL_HASH_DOMAIN = b"supermix.route-shadow.campaign-seal.v1\x00"
_PACKAGE_HASH_DOMAIN = b"supermix.route-shadow.public-package.v1\x00"
_IDENTITY_INFO = b"supermix.route-shadow.identity-key.v1\x00"
_ASSIGNMENT_INFO = b"supermix.route-shadow.assignment-key.v1\x00"
_PSEUDONYM_DOMAIN = b"supermix.route-shadow.cluster-pseudonym.v1\x00"
_DRAW_DOMAIN = b"supermix.route-shadow.whole-policy-draw.v1\x00"
_DRAW_HASH_DOMAIN = b"supermix.route-shadow.draw-proof.v1\x00"
_ASSIGNMENT_REVEAL_HASH_DOMAIN = b"supermix.route-shadow.assignment-reveal.v1\x00"
_ASSIGNMENT_COMMITMENT_HASH_DOMAIN = b"supermix.route-shadow.assignment-commitment.v1\x00"
_CLOSURE_HASH_DOMAIN = b"supermix.route-shadow.campaign-closure.v1\x00"
_SEED_REVEAL_HASH_DOMAIN = b"supermix.route-shadow.seed-reveal.v1\x00"
_REGISTRY_EVENT_HASH_DOMAIN = b"supermix.route-shadow.registry-event.v1\x00"
_MAX_SAFE_JSON_INTEGER = (1 << 53) - 1

_BOUNDARIES = {
    "shadow_only": True,
    "ledger_eligible": False,
    "executed_logging_propensity": False,
    "route_execution_enabled": False,
    "model_inference_enabled": False,
    "causal_estimate_available": False,
    "activation_available": False,
    "automatic_promotion_allowed": False,
    "causal_design_certified": False,
    "pseudonymity_is_anonymity": False,
    "post_reveal_identity_unlinkability_guaranteed": False,
}

_REQUIRED_SHADOW_TABLES = frozenset(
    {
        "shadow_registry_metadata",
        "shadow_campaign_seals",
        "shadow_assignment_commitments",
        "shadow_campaign_closures",
        "shadow_seed_reveals",
        "shadow_assignment_reveals",
        "shadow_registry_events",
    }
)
_REQUIRED_SHADOW_TRIGGERS = frozenset(
    {
        "shadow_commitments_require_open_campaign",
        "shadow_seed_reveal_requires_closure",
        "shadow_assignment_reveal_requires_seed",
        "shadow_assignment_reveal_campaign_matches_commitment",
        "shadow_metadata_no_update",
        "shadow_metadata_no_delete",
        "shadow_seals_no_update",
        "shadow_seals_no_delete",
        "shadow_commitments_no_update",
        "shadow_commitments_no_delete",
        "shadow_closures_no_update",
        "shadow_closures_no_delete",
        "shadow_seed_reveals_no_update",
        "shadow_seed_reveals_no_delete",
        "shadow_assignment_reveals_no_update",
        "shadow_assignment_reveals_no_delete",
        "shadow_events_no_update",
        "shadow_events_no_delete",
    }
)
_REQUIRED_SHADOW_INDEXES = frozenset(
    {
        "shadow_commitments_campaign_order_idx",
        "shadow_reveals_campaign_order_idx",
    }
)


class RouteShadowRegistryError(RuntimeError):
    """Base class for stable registry failures."""


class ShadowRegistryConflictError(RouteShadowRegistryError):
    """Raised when an immutable campaign artifact conflicts with existing state."""


def _require_exact_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} does not match the v1 schema")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a closed ASCII identifier")
    cooked = value.strip()
    if not _IDENTIFIER_RE.fullmatch(cooked):
        raise ValueError(f"{name} must be a closed ASCII identifier")
    return cooked


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    cooked = value.strip()
    if not _SHA256_RE.fullmatch(cooked):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return cooked


def _bounded_int(value: Any, name: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    cooked = value
    if cooked < minimum or cooked > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return cooked


def _assert_shadow_json(value: Any, name: str = "shadow artifact") -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > _MAX_SAFE_JSON_INTEGER:
            raise ValueError(f"{name} must use I-JSON safe integers")
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{name} must not contain lone Unicode surrogates") from exc
        return
    if isinstance(value, float):
        raise ValueError(f"{name} must not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} object keys must be strings")
            if not key.isascii():
                raise ValueError(f"{name} object keys must be ASCII for portable sorting")
            _assert_shadow_json(item, name)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _assert_shadow_json(item, name)
        return
    raise ValueError(f"{name} contains a non-JSON value")


def _canonical_json(value: Any, name: str = "shadow artifact") -> str:
    _assert_shadow_json(value, name)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical finite JSON") from exc


def _domain_hash(domain: bytes, value: Any, name: str = "shadow artifact") -> str:
    return hashlib.sha256(domain + _canonical_json(value, name).encode("utf-8")).hexdigest()


def _schema_integrity(connection: sqlite3.Connection) -> Dict[str, Any]:
    rows = connection.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'trigger', 'index')
        """
    ).fetchall()
    found = {
        kind: {str(row["name"]) for row in rows if str(row["type"]) == kind}
        for kind in ("table", "trigger", "index")
    }
    missing_tables = sorted(_REQUIRED_SHADOW_TABLES - found["table"])
    missing_triggers = sorted(_REQUIRED_SHADOW_TRIGGERS - found["trigger"])
    missing_indexes = sorted(_REQUIRED_SHADOW_INDEXES - found["index"])
    required_names = (
        _REQUIRED_SHADOW_TABLES
        | _REQUIRED_SHADOW_TRIGGERS
        | _REQUIRED_SHADOW_INDEXES
    )
    definition_rows = sorted(
        (
            {
                "type": str(row["type"]),
                "name": str(row["name"]),
                "sql": " ".join(str(row["sql"] or "").split()),
            }
            for row in rows
            if str(row["name"]) in required_names
        ),
        key=lambda row: (row["type"], row["name"]),
    )
    definition_fingerprint = hashlib.sha256(
        _canonical_json(definition_rows, "shadow schema definitions").encode("utf-8")
    ).hexdigest()
    definitions_ok = hmac.compare_digest(
        definition_fingerprint, SHADOW_SCHEMA_OBJECTS_SHA256
    )
    return {
        "ok": not (missing_tables or missing_triggers or missing_indexes)
        and definitions_ok,
        "required_tables": len(_REQUIRED_SHADOW_TABLES),
        "required_triggers": len(_REQUIRED_SHADOW_TRIGGERS),
        "required_indexes": len(_REQUIRED_SHADOW_INDEXES),
        "missing_tables": missing_tables,
        "missing_triggers": missing_triggers,
        "missing_indexes": missing_indexes,
        "definitions_ok": definitions_ok,
        "definition_fingerprint": definition_fingerprint,
        "expected_definition_fingerprint": SHADOW_SCHEMA_OBJECTS_SHA256,
    }


def _seed_bytes(value: Any) -> bytes:
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("shadow seed material must be bytes")
    cooked = bytes(value)
    if len(cooked) != 32:
        raise ValueError("shadow seed material must contain exactly 32 bytes")
    return cooked


def generate_shadow_seed() -> bytes:
    """Return 256 bits from the operating-system CSPRNG."""

    return secrets.token_bytes(32)


def _encode_seed(seed: bytes) -> str:
    return base64.urlsafe_b64encode(_seed_bytes(seed)).decode("ascii").rstrip("=")


def _decode_seed(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ValueError("seed_material_base64url must encode exactly 32 bytes")
    cooked = value.strip()
    if not _BASE64URL_RE.fullmatch(cooked):
        raise ValueError("seed_material_base64url must encode exactly 32 bytes")
    try:
        seed = base64.urlsafe_b64decode(cooked + "=")
    except (ValueError, TypeError) as exc:
        raise ValueError("seed_material_base64url is invalid") from exc
    seed = _seed_bytes(seed)
    if not hmac.compare_digest(_encode_seed(seed), cooked):
        raise ValueError("seed_material_base64url is not canonical")
    return seed


def _reject_duplicate_object_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"review bundle JSON contains duplicate object key: {key}")
        value[key] = item
    return value


def _canonical_review_bundle_json(bundle: Any, name: str) -> str:
    """Freeze a legacy reviewed bundle without importing its floats into v1 artifacts."""

    audit_route_study_review_bundle(bundle)
    try:
        return json.dumps(
            bundle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical finite JSON") from exc


def _decode_review_bundle_json(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a canonical JSON string")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite number {token}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be a strict canonical JSON string") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{name} must decode to a JSON object")
    canonical = _canonical_review_bundle_json(decoded, name)
    if canonical != value:
        raise ValueError(f"{name} is not in canonical JSON form")
    return decoded


def _hkdf_expand(seed: bytes, *, salt: bytes, info: bytes) -> bytes:
    """Derive one SHA-256-sized subkey using RFC 5869 extract-and-expand."""

    prk = hmac.new(salt, _seed_bytes(seed), hashlib.sha256).digest()
    return hmac.new(prk, info + b"\x01", hashlib.sha256).digest()


def _review_bundle_parts(bundle: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    verification = audit_route_study_review_bundle(bundle)
    if verification.get("verification_level") != "full_source_bound_reconstruction":
        raise ValueError("shadow campaigns require full source-bound reconstruction")
    if not isinstance(bundle, Mapping):
        raise ValueError("review bundle must be a JSON object")
    protocol = bundle.get("protocol")
    builder = bundle.get("protocol_builder")
    if not isinstance(protocol, Mapping) or not isinstance(builder, Mapping):
        raise ValueError("review bundle protocol and builder are required")
    options = builder.get("options")
    if not isinstance(options, Mapping) or set(options) != set(PROTOCOL_OPTION_KEYS):
        raise ValueError("review bundle protocol options are incomplete")
    return dict(protocol), dict(options), dict(verification)


def _whole_policy_arms(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    charter = protocol.get("charter")
    if not isinstance(charter, Mapping):
        raise ValueError("review bundle protocol charter is missing")
    source = charter.get("source_studies")
    target = charter.get("target_policy_class")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("review bundle source and target policy bindings are missing")
    common = source.get("common_source_contract")
    if not isinstance(common, Mapping):
        raise ValueError("review bundle source policy cohort is missing")
    support_strata = source.get("support_strata")
    if not isinstance(support_strata, list) or not support_strata:
        raise ValueError("review bundle source support strata are missing")
    frozen_source_strata: List[Dict[str, Any]] = []
    for index, stratum in enumerate(support_strata):
        if not isinstance(stratum, Mapping):
            raise ValueError(f"source support stratum {index} must be an object")
        frozen_source_strata.append(
            {
                "study_design_hash": _sha256(
                    stratum.get("study_design_hash"), "source study_design_hash"
                ),
                "candidate_set_hash": _sha256(
                    stratum.get("candidate_set_hash"), "source candidate_set_hash"
                ),
                "distribution_hash": _sha256(
                    stratum.get("distribution_hash"), "source distribution_hash"
                ),
                "baseline_action": _identifier(
                    stratum.get("baseline_action"), "source baseline_action"
                ),
            }
        )
    frozen_source_strata.sort(key=lambda row: row["study_design_hash"])
    source_class_manifest = {
        "schema_version": "route-study-shadow-source-policy-class-v1",
        "common_source_contract": {
            "policy_id": _identifier(common.get("policy_id"), "source policy_id"),
            "policy_version": _identifier(
                common.get("policy_version"), "source policy_version"
            ),
            "feature_schema_version": _identifier(
                common.get("feature_schema_version"), "source feature_schema_version"
            ),
            "support_schema_version": _identifier(
                common.get("support_schema_version"), "source support_schema_version"
            ),
            "outcome_contract_schema_version": _identifier(
                common.get("outcome_contract_schema_version"),
                "source outcome_contract_schema_version",
            ),
        },
        "support_strata": frozen_source_strata,
    }
    source_class_hash = _domain_hash(
        _SOURCE_POLICY_CLASS_HASH_DOMAIN,
        source_class_manifest,
        "source whole-policy class",
    )
    source_binding = {
        "binding_type": "frozen_source_policy_class",
        "policy_class_hash": source_class_hash,
        "policy_class_manifest": source_class_manifest,
    }
    target_class_hash = _sha256(target.get("class_hash"), "target class_hash")
    target_class_manifest = json.loads(
        _canonical_json(
            {key: value for key, value in target.items() if key != "class_hash"},
            "target whole-policy class",
        )
    )
    if _domain_hash(
        b"supermix.route-study.target-class.v1\x00",
        target_class_manifest,
        "target whole-policy class",
    ) != target_class_hash:
        # The review-bundle auditor already validates this using the protocol's
        # source-of-truth profile catalog. This guard documents the invariant
        # without allowing a partially described treatment arm.
        raise ValueError("target policy class hash does not bind its full manifest")
    target_binding = {
        "binding_type": "frozen_target_policy_class",
        "policy_class_hash": target_class_hash,
        "policy_class_manifest": target_class_manifest,
    }
    rows = [
        {
            "arm_id": "incumbent_source_policy",
            "allocation_bps": SHADOW_TOTAL_ALLOCATION_BPS - SHADOW_CANDIDATE_ALLOCATION_BPS,
            "policy_binding": source_binding,
        },
        {
            "arm_id": "candidate_target_policy",
            "allocation_bps": SHADOW_CANDIDATE_ALLOCATION_BPS,
            "policy_binding": target_binding,
        },
    ]
    return [
        {**row, "arm_hash": _domain_hash(_ARM_HASH_DOMAIN, row, "whole-policy arm")}
        for row in rows
    ]


def build_shadow_design_binding(review_bundle: Any) -> Dict[str, Any]:
    """Bind explicit whole-policy arms without treating route actions as arms."""

    protocol, options, verification = _review_bundle_parts(review_bundle)
    if options.get("seed_commitment") is not None:
        raise ValueError("origin review bundle must not already contain seed material")
    charter = protocol["charter"]
    design = charter["stateful_design"]
    population = charter["population"]
    if design.get("selected_design_mode") != "sticky_session_cluster":
        raise ValueError("shadow registry v1 supports sticky_session_cluster only")
    if design.get("assignment_unit") != "session_hash":
        raise ValueError("shadow registry v1 requires the session_hash assignment unit")
    if population.get("cluster_key_schema_version") != SHADOW_CLUSTER_KEY_SCHEMA_VERSION:
        raise ValueError(
            "shadow registry v1 requires the session-hash-v1 cluster key schema"
        )
    binding = {
        "schema_version": SHADOW_DESIGN_BINDING_SCHEMA_VERSION,
        "origin_review_bundle_hash": _sha256(
            verification.get("bundle_hash"), "origin review bundle_hash"
        ),
        "origin_review_bundle_artifact_sha256": hashlib.sha256(
            _canonical_review_bundle_json(review_bundle, "origin review bundle").encode("utf-8")
        ).hexdigest(),
        "origin_protocol_hash": _sha256(
            verification.get("protocol_hash"), "origin protocol_hash"
        ),
        "canonicalization": SHADOW_CANONICALIZATION,
        "legacy_review_bundle_encoding": SHADOW_LEGACY_BUNDLE_ENCODING,
        "design_mode": "sticky_session_cluster",
        "assignment_unit": "study_scoped_session_cluster",
        "cluster_key_schema_version": SHADOW_CLUSTER_KEY_SCHEMA_VERSION,
        "population_rule_id": _identifier(
            population.get("population_rule_id"), "population_rule_id"
        ),
        "population_rule_version": _identifier(
            population.get("population_rule_version"), "population_rule_version"
        ),
        "planned_cluster_ceiling": _bounded_int(
            population.get("planned_clusters"),
            "planned_cluster_ceiling",
            minimum=2,
            maximum=MAX_SHADOW_COMMITMENTS_PER_CAMPAIGN,
        ),
        "admitted_support_strata": [
            _sha256(value, "admitted support stratum")
            for value in population.get("admitted_support_strata", [])
        ],
        "whole_policy_arms": _whole_policy_arms(protocol),
        "assignment_algorithm": SHADOW_ASSIGNMENT_ALGORITHM,
        "allocation_total_bps": SHADOW_TOTAL_ALLOCATION_BPS,
        "boundaries": dict(_BOUNDARIES),
    }
    if not binding["admitted_support_strata"]:
        raise ValueError("shadow design must bind at least one support stratum")
    if sum(row["allocation_bps"] for row in binding["whole_policy_arms"]) != SHADOW_TOTAL_ALLOCATION_BPS:
        raise ValueError("whole-policy arm allocation must sum to 10000 basis points")
    return {
        **binding,
        "design_binding_hash": _domain_hash(
            _DESIGN_HASH_DOMAIN, binding, "shadow design binding"
        ),
    }


def compute_shadow_seed_commitment(design_binding_hash: Any, seed_material: Any) -> str:
    design_hash = _sha256(design_binding_hash, "design_binding_hash")
    seed = _seed_bytes(seed_material)
    return hashlib.sha256(
        _SEED_COMMITMENT_DOMAIN + bytes.fromhex(design_hash) + seed
    ).hexdigest()


def _rebuild_review_bundle_with_seed(review_bundle: Mapping[str, Any], seed_commitment: str) -> Dict[str, Any]:
    builder = review_bundle["protocol_builder"]
    options = dict(builder["options"])
    options["seed_commitment"] = _sha256(seed_commitment, "seed_commitment")
    return build_route_study_review_bundle(review_bundle["source_study_plans"], **options)


def _assignment_manifest(
    design_binding: Mapping[str, Any],
    committed_bundle: Mapping[str, Any],
    seed_commitment: str,
) -> Dict[str, Any]:
    verification = audit_route_study_review_bundle(committed_bundle)
    payload = {
        "schema_version": SHADOW_ASSIGNMENT_MANIFEST_SCHEMA_VERSION,
        "manifest": {
            "version": 1,
            "label": "Sealed Shadow Whole-Policy Assignment Manifest",
            "state": "shadow_assignment_design",
            "design_binding_hash": _sha256(
                design_binding.get("design_binding_hash"), "design_binding_hash"
            ),
            "committed_review_bundle_hash": _sha256(
                verification.get("bundle_hash"), "committed review bundle_hash"
            ),
            "committed_review_bundle_artifact_sha256": hashlib.sha256(
                _canonical_review_bundle_json(
                    committed_bundle, "committed review bundle"
                ).encode("utf-8")
            ).hexdigest(),
            "committed_protocol_hash": _sha256(
                verification.get("protocol_hash"), "committed protocol_hash"
            ),
            "seed_commitment": _sha256(seed_commitment, "seed_commitment"),
            "assignment_algorithm": SHADOW_ASSIGNMENT_ALGORITHM,
            "canonicalization": SHADOW_CANONICALIZATION,
            "legacy_review_bundle_encoding": SHADOW_LEGACY_BUNDLE_ENCODING,
            "whole_policy_arms": json.loads(
                _canonical_json(
                    design_binding["whole_policy_arms"],
                    "whole-policy arms",
                )
            ),
            "allocation_total_bps": SHADOW_TOTAL_ALLOCATION_BPS,
            "block_id": SHADOW_BLOCK_ID,
            "boundaries": dict(_BOUNDARIES),
        },
    }
    return {
        **payload,
        "manifest_hash": _domain_hash(
            _MANIFEST_HASH_DOMAIN, payload, "shadow assignment manifest"
        ),
    }


def _campaign_seal(
    design_binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed_commitment: str,
) -> Dict[str, Any]:
    design_hash = _sha256(design_binding.get("design_binding_hash"), "design_binding_hash")
    campaign_id = f"shadow:{design_hash}"
    payload = {
        "schema_version": SHADOW_CAMPAIGN_SEAL_SCHEMA_VERSION,
        "seal": {
            "version": 1,
            "label": "Local Shadow Assignment Seal",
            "state": "sealed_shadow_only",
            "campaign_id": campaign_id,
            "design_binding_hash": design_hash,
            "origin_review_bundle_hash": _sha256(
                design_binding.get("origin_review_bundle_hash"), "origin review bundle_hash"
            ),
            "committed_review_bundle_hash": _sha256(
                manifest["manifest"]["committed_review_bundle_hash"],
                "committed review bundle_hash",
            ),
            "committed_protocol_hash": _sha256(
                manifest["manifest"]["committed_protocol_hash"], "committed protocol_hash"
            ),
            "manifest_hash": _sha256(manifest.get("manifest_hash"), "manifest_hash"),
            "seed_commitment": _sha256(seed_commitment, "seed_commitment"),
            "assignment_algorithm": SHADOW_ASSIGNMENT_ALGORITHM,
            "canonicalization": SHADOW_CANONICALIZATION,
            "private_seed_material_included": False,
            "local_append_only_registry_required": True,
            "authenticity_proof_available": False,
            "trusted_timestamp_available": False,
            "independent_seed_custody_verified": False,
            "nonce_grinding_resistance_proven": False,
            "boundaries": dict(_BOUNDARIES),
        },
    }
    return {**payload, "seal_hash": _domain_hash(_SEAL_HASH_DOMAIN, payload, "campaign seal")}


def _seed_capsule(seal: Mapping[str, Any], seed_material: bytes) -> Dict[str, Any]:
    return {
        "schema_version": SHADOW_SEED_CAPSULE_SCHEMA_VERSION,
        "campaign_id": _identifier(seal["seal"]["campaign_id"], "campaign_id"),
        "design_binding_hash": _sha256(
            seal["seal"]["design_binding_hash"], "design_binding_hash"
        ),
        "seed_commitment": _sha256(seal["seal"]["seed_commitment"], "seed_commitment"),
        "seed_material_base64url": _encode_seed(seed_material),
    }


def create_shadow_campaign_artifacts(review_bundle: Any, seed_material: Any) -> Dict[str, Any]:
    """Create a public campaign package and a separately handled seed capsule."""

    if not isinstance(review_bundle, Mapping):
        raise ValueError("review bundle must be a JSON object")
    seed = _seed_bytes(seed_material)
    design_binding = build_shadow_design_binding(review_bundle)
    seed_commitment = compute_shadow_seed_commitment(
        design_binding["design_binding_hash"], seed
    )
    committed_bundle = _rebuild_review_bundle_with_seed(review_bundle, seed_commitment)
    manifest = _assignment_manifest(design_binding, committed_bundle, seed_commitment)
    seal = _campaign_seal(design_binding, manifest, seed_commitment)
    payload = {
        "schema_version": SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION,
        "origin_review_bundle": _canonical_review_bundle_json(
            review_bundle, "origin review bundle"
        ),
        "committed_review_bundle": _canonical_review_bundle_json(
            committed_bundle, "committed review bundle"
        ),
        "design_binding": design_binding,
        "assignment_manifest": manifest,
        "campaign_seal": seal,
    }
    public_package = {
        **payload,
        "public_package_hash": _domain_hash(
            _PACKAGE_HASH_DOMAIN, payload, "shadow public package"
        ),
    }
    return {
        "public_package": public_package,
        "private_seed_capsule": _seed_capsule(seal, seed),
    }


def audit_shadow_campaign_artifacts(public_package: Any) -> Dict[str, Any]:
    package = _require_exact_keys(
        public_package,
        {
            "schema_version",
            "origin_review_bundle",
            "committed_review_bundle",
            "design_binding",
            "assignment_manifest",
            "campaign_seal",
            "public_package_hash",
        },
        "shadow public package",
    )
    if package.get("schema_version") != SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported shadow public package schema_version")
    origin_encoded = package["origin_review_bundle"]
    committed_encoded = package["committed_review_bundle"]
    origin = _decode_review_bundle_json(origin_encoded, "origin review bundle")
    committed = _decode_review_bundle_json(committed_encoded, "committed review bundle")
    origin_protocol, origin_options, origin_verification = _review_bundle_parts(origin)
    committed_protocol, committed_options, committed_verification = _review_bundle_parts(committed)
    if origin_options.get("seed_commitment") is not None:
        raise ValueError("origin review bundle must remain seedless")
    seed_commitment = _sha256(
        committed_options.get("seed_commitment"), "committed seed_commitment"
    )
    expected_options = dict(origin_options)
    expected_options["seed_commitment"] = seed_commitment
    if committed_options != expected_options:
        raise ValueError("committed review bundle changed fields other than seed_commitment")
    if committed["source_study_plans"] != origin["source_study_plans"]:
        raise ValueError("committed review bundle changed source study plans")

    expected_design = build_shadow_design_binding(origin)
    if _canonical_json(package["design_binding"]) != _canonical_json(expected_design):
        raise ValueError("shadow design binding does not reconstruct from the origin bundle")
    expected_committed = _rebuild_review_bundle_with_seed(origin, seed_commitment)
    if _canonical_review_bundle_json(
        committed, "committed review bundle"
    ) != _canonical_review_bundle_json(expected_committed, "expected committed review bundle"):
        raise ValueError("committed review bundle does not reconstruct")
    expected_manifest = _assignment_manifest(expected_design, expected_committed, seed_commitment)
    if _canonical_json(package["assignment_manifest"]) != _canonical_json(expected_manifest):
        raise ValueError("shadow assignment manifest does not reconstruct")
    expected_seal = _campaign_seal(expected_design, expected_manifest, seed_commitment)
    if _canonical_json(package["campaign_seal"]) != _canonical_json(expected_seal):
        raise ValueError("shadow campaign seal does not reconstruct")

    payload = {
        "schema_version": package["schema_version"],
        "origin_review_bundle": origin_encoded,
        "committed_review_bundle": committed_encoded,
        "design_binding": package["design_binding"],
        "assignment_manifest": package["assignment_manifest"],
        "campaign_seal": package["campaign_seal"],
    }
    supplied_hash = _sha256(package.get("public_package_hash"), "public_package_hash")
    expected_hash = _domain_hash(_PACKAGE_HASH_DOMAIN, payload, "shadow public package")
    if not hmac.compare_digest(supplied_hash, expected_hash):
        raise ValueError("public_package_hash does not match the canonical package")
    return {
        "ok": True,
        "schema_version": SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION,
        "campaign_id": expected_seal["seal"]["campaign_id"],
        "public_package_hash": supplied_hash,
        "origin_review_bundle_hash": origin_verification["bundle_hash"],
        "origin_review_bundle_artifact_sha256": hashlib.sha256(
            origin_encoded.encode("utf-8")
        ).hexdigest(),
        "committed_review_bundle_hash": committed_verification["bundle_hash"],
        "committed_review_bundle_artifact_sha256": hashlib.sha256(
            committed_encoded.encode("utf-8")
        ).hexdigest(),
        "committed_protocol_hash": committed_verification["protocol_hash"],
        "design_binding_hash": expected_design["design_binding_hash"],
        "manifest_hash": expected_manifest["manifest_hash"],
        "seal_hash": expected_seal["seal_hash"],
        "seed_commitment": seed_commitment,
        "whole_policy_arm_count": len(expected_manifest["manifest"]["whole_policy_arms"]),
        "support_stratum_count": committed_verification["support_stratum_count"],
        "verification_level": "full_source_bound_shadow_seal_reconstruction",
        "authenticity_proof_available": False,
        "trusted_timestamp_available": False,
        "execution_enabled": False,
        "activation_available": False,
    }


def _seed_from_capsule(public_package: Mapping[str, Any], capsule: Any) -> bytes:
    verification = audit_shadow_campaign_artifacts(public_package)
    value = _require_exact_keys(
        capsule,
        {
            "schema_version",
            "campaign_id",
            "design_binding_hash",
            "seed_commitment",
            "seed_material_base64url",
        },
        "shadow seed capsule",
    )
    if value.get("schema_version") != SHADOW_SEED_CAPSULE_SCHEMA_VERSION:
        raise ValueError("unsupported shadow seed capsule schema_version")
    if value.get("campaign_id") != verification["campaign_id"]:
        raise ValueError("shadow seed capsule campaign_id does not match")
    if value.get("design_binding_hash") != verification["design_binding_hash"]:
        raise ValueError("shadow seed capsule design binding does not match")
    if value.get("seed_commitment") != verification["seed_commitment"]:
        raise ValueError("shadow seed capsule commitment does not match")
    seed = _decode_seed(value.get("seed_material_base64url"))
    expected = compute_shadow_seed_commitment(verification["design_binding_hash"], seed)
    if not hmac.compare_digest(expected, verification["seed_commitment"]):
        raise ValueError("shadow seed material does not open the sealed commitment")
    return seed


def audit_shadow_seed_capsule(public_package: Any, capsule: Any) -> Dict[str, Any]:
    seed = _seed_from_capsule(public_package, capsule)
    verification = audit_shadow_campaign_artifacts(public_package)
    return {
        "ok": True,
        "schema_version": SHADOW_SEED_CAPSULE_SCHEMA_VERSION,
        "campaign_id": verification["campaign_id"],
        "seed_commitment": verification["seed_commitment"],
        "seed_length_bytes": len(seed),
        "seed_commitment_verified": True,
        "seed_material_returned": False,
        "execution_enabled": False,
        "activation_available": False,
    }


def _cluster_identifier(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(
            "cluster identifier must be a canonical session_hash string"
        )
    # The sealed v1 design binds ``assignment_unit=session_hash`` and
    # ``cluster_key_schema_version=session-hash-v1``.  Accept the exact wire
    # representation produced by ``route_policy_ledger.hash_session_identity``
    # rather than trimming, case-folding, or accepting a caller-defined alias.
    # The hash is HMAC-pseudonymized below and is never persisted directly.
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(
            "cluster identifier must be a canonical session_hash: "
            "exactly 64 lowercase hexadecimal characters"
        )
    return value


def _assignment_from_pseudonym(
    public_package: Mapping[str, Any],
    seed: bytes,
    cluster_pseudonym: str,
    *,
    verification: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    verified = (
        dict(verification)
        if verification is not None
        else audit_shadow_campaign_artifacts(public_package)
    )
    pseudonym = _sha256(cluster_pseudonym, "cluster_pseudonym")
    design_hash = verified["design_binding_hash"]
    seal_hash = verified["seal_hash"]
    manifest = public_package["assignment_manifest"]["manifest"]
    assignment_key = _hkdf_expand(
        seed,
        salt=bytes.fromhex(design_hash),
        info=_ASSIGNMENT_INFO + bytes.fromhex(seal_hash),
    )
    draw_digest = hmac.new(
        assignment_key,
        _DRAW_DOMAIN
        + bytes.fromhex(seal_hash)
        + bytes.fromhex(pseudonym)
        + SHADOW_BLOCK_ID.to_bytes(8, "big"),
        hashlib.sha256,
    ).digest()
    draw_bucket = int.from_bytes(draw_digest, "big") * SHADOW_TOTAL_ALLOCATION_BPS // (1 << 256)
    chosen: Optional[Mapping[str, Any]] = None
    cumulative = 0
    for arm in manifest["whole_policy_arms"]:
        cumulative += int(arm["allocation_bps"])
        if draw_bucket < cumulative:
            chosen = arm
            break
    if chosen is None:
        raise ValueError("whole-policy allocation is incomplete")
    reveal_payload = {
        "schema_version": SHADOW_ASSIGNMENT_REVEAL_SCHEMA_VERSION,
        "assignment": {
            "campaign_id": verified["campaign_id"],
            "seal_hash": seal_hash,
            "manifest_hash": verified["manifest_hash"],
            "cluster_pseudonym": pseudonym,
            "block_id": SHADOW_BLOCK_ID,
            "assignment_algorithm": SHADOW_ASSIGNMENT_ALGORITHM,
            "arm_id": chosen["arm_id"],
            "arm_hash": chosen["arm_hash"],
            "arm_probability_bps": chosen["allocation_bps"],
            "draw_bucket_bps": draw_bucket,
            "draw_hash": hashlib.sha256(_DRAW_HASH_DOMAIN + draw_digest).hexdigest(),
            "boundaries": dict(_BOUNDARIES),
        },
    }
    reveal = {
        **reveal_payload,
        "assignment_reveal_hash": _domain_hash(
            _ASSIGNMENT_REVEAL_HASH_DOMAIN,
            reveal_payload,
            "shadow assignment reveal",
        ),
    }
    commitment_payload = {
        "schema_version": SHADOW_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION,
        "commitment": {
            "campaign_id": verified["campaign_id"],
            "seal_hash": seal_hash,
            "manifest_hash": verified["manifest_hash"],
            "cluster_pseudonym": pseudonym,
            "block_id": SHADOW_BLOCK_ID,
            "assignment_algorithm": SHADOW_ASSIGNMENT_ALGORITHM,
            "seed_commitment": verified["seed_commitment"],
            "assignment_reveal_commitment": reveal["assignment_reveal_hash"],
            "chosen_arm_withheld_until_reveal": True,
            "boundaries": dict(_BOUNDARIES),
        },
    }
    commitment = {
        **commitment_payload,
        "commitment_hash": _domain_hash(
            _ASSIGNMENT_COMMITMENT_HASH_DOMAIN,
            commitment_payload,
            "shadow assignment commitment",
        ),
    }
    return {"commitment": commitment, "reveal": reveal}


def prepare_shadow_assignment_commitment(
    public_package: Any,
    seed_capsule: Any,
    cluster_identifier: Any,
) -> Dict[str, Any]:
    if not isinstance(public_package, Mapping):
        raise ValueError("shadow public package must be a JSON object")
    seed = _seed_from_capsule(public_package, seed_capsule)
    verification = audit_shadow_campaign_artifacts(public_package)
    identity_key = _hkdf_expand(
        seed,
        salt=bytes.fromhex(verification["design_binding_hash"]),
        info=_IDENTITY_INFO + bytes.fromhex(verification["seal_hash"]),
    )
    pseudonym = hmac.new(
        identity_key,
        _PSEUDONYM_DOMAIN + _cluster_identifier(cluster_identifier).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    prepared = _assignment_from_pseudonym(
        public_package,
        seed,
        pseudonym,
        verification=verification,
    )
    return {
        "commitment": prepared["commitment"],
        "private_reveal": prepared["reveal"],
    }


def _closure_artifact(public_package: Mapping[str, Any], frozen_count: int) -> Dict[str, Any]:
    verification = audit_shadow_campaign_artifacts(public_package)
    payload = {
        "schema_version": SHADOW_CAMPAIGN_CLOSURE_SCHEMA_VERSION,
        "closure": {
            "campaign_id": verification["campaign_id"],
            "seal_hash": verification["seal_hash"],
            "frozen_commitment_count": _bounded_int(
                frozen_count,
                "frozen_commitment_count",
                minimum=0,
                maximum=MAX_SHADOW_COMMITMENTS_PER_CAMPAIGN,
            ),
            "state": "commitments_closed",
            "boundaries": dict(_BOUNDARIES),
        },
    }
    return {**payload, "closure_hash": _domain_hash(_CLOSURE_HASH_DOMAIN, payload, "campaign closure")}


def _seed_reveal_artifact(public_package: Mapping[str, Any], seed: bytes) -> Dict[str, Any]:
    verification = audit_shadow_campaign_artifacts(public_package)
    payload = {
        "schema_version": SHADOW_SEED_REVEAL_SCHEMA_VERSION,
        "reveal": {
            "campaign_id": verification["campaign_id"],
            "seal_hash": verification["seal_hash"],
            "seed_commitment": verification["seed_commitment"],
            "seed_material_base64url": _encode_seed(seed),
            "seed_material_fingerprint": hashlib.sha256(
                _SEED_REVEAL_HASH_DOMAIN + _seed_bytes(seed)
            ).hexdigest(),
            "seed_material_revealed": True,
            "state": "seed_revealed",
            "boundaries": dict(_BOUNDARIES),
        },
    }
    return {
        **payload,
        "seed_reveal_hash": _domain_hash(_SEED_REVEAL_HASH_DOMAIN, payload, "seed reveal"),
    }


def _audit_assignment_commitment_artifact(
    value: Any,
    *,
    verification: Mapping[str, Any],
) -> Dict[str, Any]:
    artifact = _require_exact_keys(
        value,
        {"schema_version", "commitment", "commitment_hash"},
        "shadow assignment commitment",
    )
    if artifact.get("schema_version") != SHADOW_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION:
        raise ValueError("unsupported shadow assignment commitment schema_version")
    commitment = _require_exact_keys(
        artifact.get("commitment"),
        {
            "campaign_id",
            "seal_hash",
            "manifest_hash",
            "cluster_pseudonym",
            "block_id",
            "assignment_algorithm",
            "seed_commitment",
            "assignment_reveal_commitment",
            "chosen_arm_withheld_until_reveal",
            "boundaries",
        },
        "shadow assignment commitment payload",
    )
    if commitment.get("campaign_id") != verification.get("campaign_id"):
        raise ValueError("assignment commitment campaign_id does not match its seal")
    if commitment.get("seal_hash") != verification.get("seal_hash"):
        raise ValueError("assignment commitment seal_hash does not match")
    if commitment.get("manifest_hash") != verification.get("manifest_hash"):
        raise ValueError("assignment commitment manifest_hash does not match")
    if commitment.get("seed_commitment") != verification.get("seed_commitment"):
        raise ValueError("assignment commitment seed commitment does not match")
    _sha256(commitment.get("cluster_pseudonym"), "cluster_pseudonym")
    _sha256(
        commitment.get("assignment_reveal_commitment"),
        "assignment_reveal_commitment",
    )
    if _bounded_int(
        commitment.get("block_id"), "block_id", minimum=0, maximum=0
    ) != SHADOW_BLOCK_ID:
        raise ValueError("assignment commitment block_id is unsupported")
    if commitment.get("assignment_algorithm") != SHADOW_ASSIGNMENT_ALGORITHM:
        raise ValueError("assignment commitment algorithm is unsupported")
    if commitment.get("chosen_arm_withheld_until_reveal") is not True:
        raise ValueError("assignment commitment must withhold its arm")
    if _canonical_json(commitment.get("boundaries")) != _canonical_json(_BOUNDARIES):
        raise ValueError("assignment commitment boundaries do not match")
    payload = {
        "schema_version": artifact["schema_version"],
        "commitment": dict(commitment),
    }
    expected_hash = _domain_hash(
        _ASSIGNMENT_COMMITMENT_HASH_DOMAIN,
        payload,
        "shadow assignment commitment",
    )
    supplied_hash = _sha256(artifact.get("commitment_hash"), "commitment_hash")
    if not hmac.compare_digest(expected_hash, supplied_hash):
        raise ValueError("assignment commitment hash does not match")
    return dict(artifact)


def _now_us() -> int:
    return time.time_ns() // 1_000


class RouteShadowAssignmentRegistry:
    """SQLite registry isolated from executed route decisions and outcomes."""

    def __init__(
        self,
        db_path: Any,
        *,
        timeout_seconds: float = 30.0,
        read_only: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        if str(self.db_path) == ":memory:":
            raise ValueError("RouteShadowAssignmentRegistry requires a durable filesystem path")
        if not isinstance(read_only, bool):
            raise ValueError("read_only must be a boolean")
        self.read_only = read_only
        if self.read_only:
            if not self.db_path.is_file():
                raise FileNotFoundError(f"shadow registry does not exist: {self.db_path}")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        if self.read_only:
            self._validate_existing_schema()
        else:
            self._initialize()

    def _connect(self) -> sqlite3.Connection:
        database = (
            f"{self.db_path.expanduser().resolve().as_uri()}?mode=ro"
            if self.read_only
            else str(self.db_path)
        )
        connection = sqlite3.connect(
            database,
            timeout=self.timeout_seconds,
            isolation_level=None,
            uri=self.read_only,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}")
        connection.execute("PRAGMA foreign_keys = ON")
        if not self.read_only:
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _validate_existing_schema(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != SHADOW_REGISTRY_SCHEMA_VERSION:
                raise RouteShadowRegistryError(
                    f"unsupported shadow registry schema {version}; expected {SHADOW_REGISTRY_SCHEMA_VERSION}"
                )
            metadata = connection.execute(
                "SELECT value FROM shadow_registry_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if metadata is None or str(metadata["value"]) != str(
                SHADOW_REGISTRY_SCHEMA_VERSION
            ):
                raise RouteShadowRegistryError("shadow registry schema metadata is missing or invalid")
            schema = _schema_integrity(connection)
            if schema["missing_tables"]:
                raise RouteShadowRegistryError("shadow registry schema tables are incomplete")
        except sqlite3.Error as exc:
            raise RouteShadowRegistryError("shadow registry schema is incomplete") from exc
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, SHADOW_REGISTRY_SCHEMA_VERSION):
                raise RouteShadowRegistryError(
                    f"unsupported shadow registry schema {version}; expected {SHADOW_REGISTRY_SCHEMA_VERSION}"
                )
            if version == 0:
                existing_objects = connection.execute(
                    """
                    SELECT type, name
                    FROM sqlite_master
                    WHERE name NOT LIKE 'sqlite_%'
                    ORDER BY type, name
                    """
                ).fetchall()
                if existing_objects:
                    raise RouteShadowRegistryError(
                        "refusing to initialize a non-empty unversioned SQLite database"
                    )
            # Reject future schemas before any persistent journal-mode or DDL
            # mutation so an older binary cannot rewrite a newer registry.
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS shadow_registry_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS shadow_campaign_seals (
                    campaign_id TEXT PRIMARY KEY,
                    origin_bundle_hash TEXT NOT NULL UNIQUE,
                    committed_bundle_hash TEXT NOT NULL UNIQUE,
                    protocol_hash TEXT NOT NULL,
                    design_binding_hash TEXT NOT NULL UNIQUE,
                    manifest_hash TEXT NOT NULL UNIQUE,
                    seal_hash TEXT NOT NULL UNIQUE,
                    seed_commitment TEXT NOT NULL UNIQUE,
                    planned_cluster_ceiling INTEGER NOT NULL CHECK (planned_cluster_ceiling >= 2),
                    public_package_json TEXT NOT NULL,
                    registered_at_us INTEGER NOT NULL CHECK (registered_at_us > 0)
                );

                CREATE TABLE IF NOT EXISTS shadow_assignment_commitments (
                    commitment_hash TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL REFERENCES shadow_campaign_seals(campaign_id),
                    cluster_pseudonym TEXT NOT NULL,
                    block_id INTEGER NOT NULL CHECK (block_id = 0),
                    assignment_reveal_commitment TEXT NOT NULL,
                    commitment_json TEXT NOT NULL,
                    committed_at_us INTEGER NOT NULL CHECK (committed_at_us > 0),
                    UNIQUE (campaign_id, cluster_pseudonym, block_id)
                );

                CREATE TABLE IF NOT EXISTS shadow_campaign_closures (
                    campaign_id TEXT PRIMARY KEY REFERENCES shadow_campaign_seals(campaign_id),
                    frozen_commitment_count INTEGER NOT NULL CHECK (frozen_commitment_count >= 0),
                    closure_hash TEXT NOT NULL UNIQUE,
                    closure_json TEXT NOT NULL,
                    closed_at_us INTEGER NOT NULL CHECK (closed_at_us > 0)
                );

                CREATE TABLE IF NOT EXISTS shadow_seed_reveals (
                    campaign_id TEXT PRIMARY KEY REFERENCES shadow_campaign_seals(campaign_id),
                    seed_material_base64url TEXT NOT NULL,
                    seed_reveal_hash TEXT NOT NULL UNIQUE,
                    reveal_json TEXT NOT NULL,
                    revealed_at_us INTEGER NOT NULL CHECK (revealed_at_us > 0)
                );

                CREATE TABLE IF NOT EXISTS shadow_assignment_reveals (
                    commitment_hash TEXT PRIMARY KEY
                        REFERENCES shadow_assignment_commitments(commitment_hash),
                    campaign_id TEXT NOT NULL REFERENCES shadow_campaign_seals(campaign_id),
                    verification_status TEXT NOT NULL CHECK (verification_status IN ('matched', 'mismatch')),
                    assignment_reveal_hash TEXT NOT NULL,
                    reveal_json TEXT NOT NULL,
                    verified_at_us INTEGER NOT NULL CHECK (verified_at_us > 0)
                );

                CREATE TABLE IF NOT EXISTS shadow_registry_events (
                    event_sequence INTEGER PRIMARY KEY CHECK (event_sequence >= 1),
                    event_type TEXT NOT NULL
                        CHECK (event_type IN ('campaign_sealed', 'assignment_committed', 'commitments_closed', 'seed_revealed', 'assignment_verified')),
                    artifact_hash TEXT NOT NULL UNIQUE,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE,
                    event_json TEXT NOT NULL,
                    recorded_at_us INTEGER NOT NULL CHECK (recorded_at_us > 0)
                );

                CREATE INDEX IF NOT EXISTS shadow_commitments_campaign_order_idx
                ON shadow_assignment_commitments(
                    campaign_id, committed_at_us, commitment_hash
                );
                CREATE INDEX IF NOT EXISTS shadow_reveals_campaign_order_idx
                ON shadow_assignment_reveals(
                    campaign_id, verified_at_us, commitment_hash
                );

                CREATE TRIGGER IF NOT EXISTS shadow_commitments_require_open_campaign
                BEFORE INSERT ON shadow_assignment_commitments
                WHEN EXISTS (
                    SELECT 1 FROM shadow_campaign_closures c
                    WHERE c.campaign_id = NEW.campaign_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'shadow campaign commitments are closed');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_seed_reveal_requires_closure
                BEFORE INSERT ON shadow_seed_reveals
                WHEN NOT EXISTS (
                    SELECT 1 FROM shadow_campaign_closures c
                    WHERE c.campaign_id = NEW.campaign_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'shadow seed reveal requires campaign closure');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_assignment_reveal_requires_seed
                BEFORE INSERT ON shadow_assignment_reveals
                WHEN NOT EXISTS (
                    SELECT 1 FROM shadow_seed_reveals r
                    WHERE r.campaign_id = NEW.campaign_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment reveal requires seed reveal');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_assignment_reveal_campaign_matches_commitment
                BEFORE INSERT ON shadow_assignment_reveals
                WHEN NOT EXISTS (
                    SELECT 1 FROM shadow_assignment_commitments c
                    WHERE c.commitment_hash = NEW.commitment_hash
                      AND c.campaign_id = NEW.campaign_id
                )
                BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment reveal campaign does not match commitment');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_metadata_no_update
                BEFORE UPDATE ON shadow_registry_metadata BEGIN
                    SELECT RAISE(ABORT, 'shadow registry metadata is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_metadata_no_delete
                BEFORE DELETE ON shadow_registry_metadata BEGIN
                    SELECT RAISE(ABORT, 'shadow registry metadata is append-only');
                END;

                CREATE TRIGGER IF NOT EXISTS shadow_seals_no_update
                BEFORE UPDATE ON shadow_campaign_seals BEGIN
                    SELECT RAISE(ABORT, 'shadow campaign seals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_seals_no_delete
                BEFORE DELETE ON shadow_campaign_seals BEGIN
                    SELECT RAISE(ABORT, 'shadow campaign seals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_commitments_no_update
                BEFORE UPDATE ON shadow_assignment_commitments BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment commitments are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_commitments_no_delete
                BEFORE DELETE ON shadow_assignment_commitments BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment commitments are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_closures_no_update
                BEFORE UPDATE ON shadow_campaign_closures BEGIN
                    SELECT RAISE(ABORT, 'shadow campaign closures are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_closures_no_delete
                BEFORE DELETE ON shadow_campaign_closures BEGIN
                    SELECT RAISE(ABORT, 'shadow campaign closures are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_seed_reveals_no_update
                BEFORE UPDATE ON shadow_seed_reveals BEGIN
                    SELECT RAISE(ABORT, 'shadow seed reveals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_seed_reveals_no_delete
                BEFORE DELETE ON shadow_seed_reveals BEGIN
                    SELECT RAISE(ABORT, 'shadow seed reveals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_assignment_reveals_no_update
                BEFORE UPDATE ON shadow_assignment_reveals BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment reveals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_assignment_reveals_no_delete
                BEFORE DELETE ON shadow_assignment_reveals BEGIN
                    SELECT RAISE(ABORT, 'shadow assignment reveals are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_events_no_update
                BEFORE UPDATE ON shadow_registry_events BEGIN
                    SELECT RAISE(ABORT, 'shadow registry events are append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS shadow_events_no_delete
                BEFORE DELETE ON shadow_registry_events BEGIN
                    SELECT RAISE(ABORT, 'shadow registry events are append-only');
                END;
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO shadow_registry_metadata(key, value) VALUES ('schema_version', ?)",
                (str(SHADOW_REGISTRY_SCHEMA_VERSION),),
            )
            metadata = connection.execute(
                "SELECT value FROM shadow_registry_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if metadata is None or str(metadata["value"]) != str(
                SHADOW_REGISTRY_SCHEMA_VERSION
            ):
                raise RouteShadowRegistryError(
                    "shadow registry schema metadata is missing or invalid"
                )
            if not _schema_integrity(connection)["ok"]:
                raise RouteShadowRegistryError(
                    "shadow registry schema objects are incomplete"
                )
            connection.execute(f"PRAGMA user_version = {SHADOW_REGISTRY_SCHEMA_VERSION}")
        finally:
            connection.close()

    @contextmanager
    def _write_transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RouteShadowRegistryError("shadow registry was opened read-only")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def _read_transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _append_event(
        connection: sqlite3.Connection,
        event_type: str,
        artifact_hash: str,
        *,
        recorded_at_us: int,
    ) -> Dict[str, Any]:
        previous = connection.execute(
            "SELECT event_sequence, event_hash FROM shadow_registry_events ORDER BY event_sequence DESC LIMIT 1"
        ).fetchone()
        sequence = int(previous["event_sequence"]) + 1 if previous else 1
        previous_hash = str(previous["event_hash"]) if previous else None
        payload = {
            "schema_version": SHADOW_REGISTRY_EVENT_SCHEMA_VERSION,
            "event_sequence": sequence,
            "event_type": _identifier(event_type, "event_type"),
            "artifact_hash": _sha256(artifact_hash, "artifact_hash"),
            "previous_event_hash": previous_hash,
            "boundaries": dict(_BOUNDARIES),
        }
        event_hash = _domain_hash(_REGISTRY_EVENT_HASH_DOMAIN, payload, "registry event")
        connection.execute(
            """
            INSERT INTO shadow_registry_events(
                event_sequence, event_type, artifact_hash, previous_event_hash,
                event_hash, event_json, recorded_at_us
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sequence,
                event_type,
                artifact_hash,
                previous_hash,
                event_hash,
                _canonical_json(payload, "registry event"),
                recorded_at_us,
            ),
        )
        return {**payload, "event_hash": event_hash, "recorded_at_us": recorded_at_us}

    def seal_campaign(self, review_bundle: Any, seed_material: Any) -> Dict[str, Any]:
        artifacts = create_shadow_campaign_artifacts(review_bundle, seed_material)
        public_package = artifacts["public_package"]
        capsule = artifacts["private_seed_capsule"]
        verification = audit_shadow_campaign_artifacts(public_package)
        design = public_package["design_binding"]
        registered_at_us = _now_us()
        package_json = _canonical_json(public_package, "shadow public package")
        created = False
        with self._write_transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM shadow_campaign_seals WHERE origin_bundle_hash = ?",
                (verification["origin_review_bundle_hash"],),
            ).fetchone()
            if existing:
                if str(existing["public_package_json"]) != package_json:
                    raise ShadowRegistryConflictError(
                        "origin review bundle is already sealed with a different seed commitment"
                    )
            else:
                count = int(connection.execute("SELECT COUNT(*) FROM shadow_campaign_seals").fetchone()[0])
                if count >= MAX_SHADOW_CAMPAIGNS:
                    raise RouteShadowRegistryError("shadow registry campaign ceiling reached")
                connection.execute(
                    """
                    INSERT INTO shadow_campaign_seals(
                        campaign_id, origin_bundle_hash, committed_bundle_hash, protocol_hash,
                        design_binding_hash, manifest_hash, seal_hash, seed_commitment,
                        planned_cluster_ceiling, public_package_json, registered_at_us
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        verification["campaign_id"],
                        verification["origin_review_bundle_hash"],
                        verification["committed_review_bundle_hash"],
                        verification["committed_protocol_hash"],
                        verification["design_binding_hash"],
                        verification["manifest_hash"],
                        verification["seal_hash"],
                        verification["seed_commitment"],
                        design["planned_cluster_ceiling"],
                        package_json,
                        registered_at_us,
                    ),
                )
                self._append_event(
                    connection,
                    "campaign_sealed",
                    verification["seal_hash"],
                    recorded_at_us=registered_at_us,
                )
                created = True
        return {
            "ok": True,
            "created": created,
            "public_package": public_package,
            "private_seed_capsule": capsule if created else None,
            "private_seed_material_persisted": False,
            "seed_capsule_returned_once": created,
            "execution_enabled": False,
            "activation_available": False,
        }

    def _campaign_row(self, connection: sqlite3.Connection, campaign_id: Any) -> sqlite3.Row:
        cooked = _identifier(campaign_id, "campaign_id")
        row = connection.execute(
            "SELECT * FROM shadow_campaign_seals WHERE campaign_id = ?", (cooked,)
        ).fetchone()
        if row is None:
            raise KeyError(f"shadow campaign not found: {cooked}")
        return row

    def append_assignment_commitment(
        self,
        *,
        campaign_id: Any,
        seed_capsule: Any,
        cluster_identifier: Any,
    ) -> Dict[str, Any]:
        cooked_campaign = _identifier(campaign_id, "campaign_id")
        with self._read_transaction() as connection:
            row = self._campaign_row(connection, cooked_campaign)
            public_package = json.loads(str(row["public_package_json"]))
        seed = _seed_from_capsule(public_package, seed_capsule)
        identity_key = _hkdf_expand(
            seed,
            salt=bytes.fromhex(row["design_binding_hash"]),
            info=_IDENTITY_INFO + bytes.fromhex(row["seal_hash"]),
        )
        pseudonym = hmac.new(
            identity_key,
            _PSEUDONYM_DOMAIN + _cluster_identifier(cluster_identifier).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        prepared = _assignment_from_pseudonym(public_package, seed, pseudonym)
        commitment = prepared["commitment"]
        commitment_json = _canonical_json(commitment, "assignment commitment")
        committed_at_us = _now_us()
        created = False
        with self._write_transaction() as connection:
            campaign = self._campaign_row(connection, cooked_campaign)
            if connection.execute(
                "SELECT 1 FROM shadow_campaign_closures WHERE campaign_id = ?", (cooked_campaign,)
            ).fetchone():
                raise RouteShadowRegistryError("shadow campaign commitments are closed")
            existing = connection.execute(
                """
                SELECT * FROM shadow_assignment_commitments
                WHERE campaign_id = ? AND cluster_pseudonym = ? AND block_id = ?
                """,
                (cooked_campaign, pseudonym, SHADOW_BLOCK_ID),
            ).fetchone()
            if existing:
                if str(existing["commitment_json"]) != commitment_json:
                    raise ShadowRegistryConflictError("cluster already has a conflicting commitment")
            else:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM shadow_assignment_commitments WHERE campaign_id = ?",
                        (cooked_campaign,),
                    ).fetchone()[0]
                )
                ceiling = min(int(campaign["planned_cluster_ceiling"]), MAX_SHADOW_COMMITMENTS_PER_CAMPAIGN)
                if count >= ceiling:
                    raise RouteShadowRegistryError("shadow campaign planned cluster ceiling reached")
                connection.execute(
                    """
                    INSERT INTO shadow_assignment_commitments(
                        commitment_hash, campaign_id, cluster_pseudonym, block_id,
                        assignment_reveal_commitment, commitment_json, committed_at_us
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        commitment["commitment_hash"],
                        cooked_campaign,
                        pseudonym,
                        SHADOW_BLOCK_ID,
                        commitment["commitment"]["assignment_reveal_commitment"],
                        commitment_json,
                        committed_at_us,
                    ),
                )
                self._append_event(
                    connection,
                    "assignment_committed",
                    commitment["commitment_hash"],
                    recorded_at_us=committed_at_us,
                )
                created = True
        return {
            "ok": True,
            "created": created,
            "commitment": commitment,
            "chosen_arm_revealed": False,
            "private_reveal_persisted": False,
            "execution_enabled": False,
            "activation_available": False,
        }

    def close_campaign(self, campaign_id: Any) -> Dict[str, Any]:
        cooked = _identifier(campaign_id, "campaign_id")
        closed_at_us = _now_us()
        with self._write_transaction() as connection:
            row = self._campaign_row(connection, cooked)
            public_package = json.loads(str(row["public_package_json"]))
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM shadow_assignment_commitments WHERE campaign_id = ?",
                    (cooked,),
                ).fetchone()[0]
            )
            existing = connection.execute(
                "SELECT * FROM shadow_campaign_closures WHERE campaign_id = ?", (cooked,)
            ).fetchone()
            if existing:
                return json.loads(str(existing["closure_json"]))
            closure = _closure_artifact(public_package, count)
            connection.execute(
                """
                INSERT INTO shadow_campaign_closures(
                    campaign_id, frozen_commitment_count, closure_hash, closure_json, closed_at_us
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (cooked, count, closure["closure_hash"], _canonical_json(closure), closed_at_us),
            )
            self._append_event(
                connection,
                "commitments_closed",
                closure["closure_hash"],
                recorded_at_us=closed_at_us,
            )
        return closure

    def reveal_seed(self, *, campaign_id: Any, seed_capsule: Any) -> Dict[str, Any]:
        cooked = _identifier(campaign_id, "campaign_id")
        revealed_at_us = _now_us()
        with self._write_transaction() as connection:
            row = self._campaign_row(connection, cooked)
            public_package = json.loads(str(row["public_package_json"]))
            if not connection.execute(
                "SELECT 1 FROM shadow_campaign_closures WHERE campaign_id = ?", (cooked,)
            ).fetchone():
                raise RouteShadowRegistryError("shadow seed cannot be revealed before closure")
            seed = _seed_from_capsule(public_package, seed_capsule)
            seed_encoded = _encode_seed(seed)
            reveal = _seed_reveal_artifact(public_package, seed)
            existing = connection.execute(
                "SELECT * FROM shadow_seed_reveals WHERE campaign_id = ?", (cooked,)
            ).fetchone()
            if existing:
                if str(existing["seed_material_base64url"]) != seed_encoded:
                    raise ShadowRegistryConflictError("shadow campaign seed was already revealed differently")
                return json.loads(str(existing["reveal_json"]))
            connection.execute(
                """
                INSERT INTO shadow_seed_reveals(
                    campaign_id, seed_material_base64url, seed_reveal_hash,
                    reveal_json, revealed_at_us
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cooked,
                    seed_encoded,
                    reveal["seed_reveal_hash"],
                    _canonical_json(reveal),
                    revealed_at_us,
                ),
            )
            self._append_event(
                connection,
                "seed_revealed",
                reveal["seed_reveal_hash"],
                recorded_at_us=revealed_at_us,
            )
        return reveal

    def verify_assignment_reveals(
        self,
        campaign_id: Any,
        *,
        batch_size: int = MAX_SHADOW_VERIFY_BATCH,
    ) -> Dict[str, Any]:
        cooked = _identifier(campaign_id, "campaign_id")
        limit = _bounded_int(
            batch_size, "batch_size", minimum=1, maximum=MAX_SHADOW_VERIFY_BATCH
        )
        verified_at_us = _now_us()
        invalid_commitment_artifacts = 0
        with self._write_transaction() as connection:
            row = self._campaign_row(connection, cooked)
            public_package = json.loads(str(row["public_package_json"]))
            package_verification = audit_shadow_campaign_artifacts(public_package)
            commitment_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM shadow_assignment_commitments WHERE campaign_id = ?",
                    (cooked,),
                ).fetchone()[0]
            )
            closure_row = connection.execute(
                "SELECT * FROM shadow_campaign_closures WHERE campaign_id = ?", (cooked,)
            ).fetchone()
            if closure_row is None:
                raise RouteShadowRegistryError(
                    "shadow commitments must be closed before assignment verification"
                )
            try:
                expected_closure = _closure_artifact(public_package, commitment_count)
                if (
                    str(closure_row["closure_json"])
                    != _canonical_json(expected_closure, "campaign closure")
                    or str(closure_row["closure_hash"])
                    != expected_closure["closure_hash"]
                    or int(closure_row["frozen_commitment_count"])
                    != commitment_count
                ):
                    raise ValueError("closure artifact does not match frozen commitments")
            except (KeyError, TypeError, ValueError) as exc:
                raise RouteShadowRegistryError(
                    "shadow verification preflight found an invalid campaign closure"
                ) from exc
            seed_row = connection.execute(
                "SELECT * FROM shadow_seed_reveals WHERE campaign_id = ?", (cooked,)
            ).fetchone()
            if seed_row is None:
                raise RouteShadowRegistryError("shadow seed must be revealed before assignment verification")
            try:
                seed = _decode_seed(seed_row["seed_material_base64url"])
                expected_seed_reveal = _seed_reveal_artifact(public_package, seed)
                if (
                    str(seed_row["reveal_json"])
                    != _canonical_json(expected_seed_reveal, "seed reveal")
                    or str(seed_row["seed_reveal_hash"])
                    != expected_seed_reveal["seed_reveal_hash"]
                ):
                    raise ValueError("seed reveal artifact does not reconstruct")
            except (KeyError, TypeError, ValueError) as exc:
                raise RouteShadowRegistryError(
                    "shadow verification preflight found an invalid seed reveal"
                ) from exc
            pending = connection.execute(
                """
                SELECT c.*
                FROM shadow_assignment_commitments c
                LEFT JOIN shadow_assignment_reveals r ON r.commitment_hash = c.commitment_hash
                WHERE c.campaign_id = ? AND r.commitment_hash IS NULL
                ORDER BY c.committed_at_us, c.commitment_hash
                LIMIT ?
                """,
                (cooked, limit),
            ).fetchall()
            matched = 0
            mismatched = 0
            for commitment_row in pending:
                commitment_artifact_ok = False
                try:
                    commitment_text = str(commitment_row["commitment_json"])
                    commitment_value = json.loads(commitment_text)
                    if (
                        _canonical_json(
                            commitment_value, "stored assignment commitment"
                        )
                        != commitment_text
                    ):
                        raise ValueError("stored commitment is not canonical")
                    audited = _audit_assignment_commitment_artifact(
                        commitment_value,
                        verification=package_verification,
                    )
                    commitment_payload = audited["commitment"]
                    row_projection = {
                        "commitment_hash": audited["commitment_hash"],
                        "campaign_id": commitment_payload["campaign_id"],
                        "cluster_pseudonym": commitment_payload["cluster_pseudonym"],
                        "block_id": commitment_payload["block_id"],
                        "assignment_reveal_commitment": commitment_payload[
                            "assignment_reveal_commitment"
                        ],
                    }
                    for field, expected_value in row_projection.items():
                        actual_value: Any = commitment_row[field]
                        if field == "block_id":
                            actual_value = int(actual_value)
                        else:
                            actual_value = str(actual_value)
                        if actual_value != expected_value:
                            raise ValueError(
                                f"stored commitment column {field} does not match"
                            )
                    commitment_artifact_ok = True
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    invalid_commitment_artifacts += 1
                prepared = _assignment_from_pseudonym(
                    public_package,
                    seed,
                    str(commitment_row["cluster_pseudonym"]),
                    verification=package_verification,
                )
                expected_commitment = prepared["commitment"]
                reveal = prepared["reveal"]
                status = (
                    "matched"
                    if commitment_artifact_ok
                    and hmac.compare_digest(
                        expected_commitment["commitment_hash"],
                        str(commitment_row["commitment_hash"]),
                    )
                    and hmac.compare_digest(
                        reveal["assignment_reveal_hash"],
                        str(commitment_row["assignment_reveal_commitment"]),
                    )
                    else "mismatch"
                )
                if status == "matched":
                    matched += 1
                else:
                    mismatched += 1
                connection.execute(
                    """
                    INSERT INTO shadow_assignment_reveals(
                        commitment_hash, campaign_id, verification_status,
                        assignment_reveal_hash, reveal_json, verified_at_us
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        commitment_row["commitment_hash"],
                        cooked,
                        status,
                        reveal["assignment_reveal_hash"],
                        _canonical_json(reveal),
                        verified_at_us,
                    ),
                )
                self._append_event(
                    connection,
                    "assignment_verified",
                    reveal["assignment_reveal_hash"],
                    recorded_at_us=verified_at_us,
                )
            remaining = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM shadow_assignment_commitments c
                    LEFT JOIN shadow_assignment_reveals r ON r.commitment_hash = c.commitment_hash
                    WHERE c.campaign_id = ? AND r.commitment_hash IS NULL
                    """,
                    (cooked,),
                ).fetchone()[0]
            )
            total_mismatched = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM shadow_assignment_reveals
                    WHERE campaign_id = ? AND verification_status = 'mismatch'
                    """,
                    (cooked,),
                ).fetchone()[0]
            )
            processing_complete = remaining == 0
        campaign_audit_performed = processing_complete
        campaign_audit_ok: Optional[bool] = None
        if campaign_audit_performed:
            campaign_audit_ok = bool(self.snapshot(cooked)["ok"])
        batch_integrity_ok = invalid_commitment_artifacts == 0
        verification_complete = (
            processing_complete
            and total_mismatched == 0
            and campaign_audit_ok is True
        )
        return {
            "ok": total_mismatched == 0
            and batch_integrity_ok
            and (campaign_audit_ok is not False),
            "campaign_id": cooked,
            "processed": len(pending),
            "matched": matched,
            "mismatched": mismatched,
            "total_mismatched": total_mismatched,
            "invalid_commitment_artifacts": invalid_commitment_artifacts,
            "remaining": remaining,
            "processing_complete": processing_complete,
            "verification_complete": verification_complete,
            "complete": verification_complete,
            "campaign_audit_performed": campaign_audit_performed,
            "campaign_audit_ok": campaign_audit_ok,
            "execution_enabled": False,
            "activation_available": False,
        }

    @staticmethod
    def _verify_event_chain(rows: Sequence[sqlite3.Row]) -> Dict[str, Any]:
        previous_hash: Optional[str] = None
        expected_sequence = 1
        for row in rows:
            try:
                payload = json.loads(str(row["event_json"]))
                if not isinstance(payload, Mapping):
                    raise ValueError("event payload is not an object")
                if int(row["event_sequence"]) != expected_sequence:
                    return {"ok": False, "reason": "event_sequence_gap", "verified_events": expected_sequence - 1}
                if payload.get("event_sequence") != expected_sequence:
                    return {"ok": False, "reason": "event_payload_sequence_mismatch", "verified_events": expected_sequence - 1}
                if payload.get("event_type") != str(row["event_type"]):
                    return {"ok": False, "reason": "event_type_mismatch", "verified_events": expected_sequence - 1}
                if payload.get("artifact_hash") != str(row["artifact_hash"]):
                    return {"ok": False, "reason": "event_artifact_hash_mismatch", "verified_events": expected_sequence - 1}
                if payload.get("previous_event_hash") != previous_hash:
                    return {"ok": False, "reason": "previous_event_hash_mismatch", "verified_events": expected_sequence - 1}
                expected_hash = _domain_hash(_REGISTRY_EVENT_HASH_DOMAIN, payload, "registry event")
                if not hmac.compare_digest(expected_hash, str(row["event_hash"])):
                    return {"ok": False, "reason": "event_hash_mismatch", "verified_events": expected_sequence - 1}
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return {
                    "ok": False,
                    "reason": "event_payload_invalid",
                    "verified_events": expected_sequence - 1,
                }
            previous_hash = expected_hash
            expected_sequence += 1
        return {
            "ok": True,
            "reason": "verified",
            "verified_events": len(rows),
            "head_event_hash": previous_hash,
        }

    def snapshot(self, campaign_id: Optional[Any] = None) -> Dict[str, Any]:
        cooked = _identifier(campaign_id, "campaign_id") if campaign_id is not None else None
        with self._read_transaction() as connection:
            schema_integrity = _schema_integrity(connection)
            if cooked is None:
                seal_rows = connection.execute(
                    "SELECT * FROM shadow_campaign_seals ORDER BY registered_at_us, campaign_id"
                ).fetchall()
            else:
                seal_rows = [self._campaign_row(connection, cooked)]
            campaigns: List[Dict[str, Any]] = []
            for seal in seal_rows:
                cid = str(seal["campaign_id"])
                commitment_rows = connection.execute(
                    """
                    SELECT * FROM shadow_assignment_commitments
                    WHERE campaign_id = ? ORDER BY committed_at_us, commitment_hash
                    """,
                    (cid,),
                ).fetchall()
                commitment_count = len(commitment_rows)
                closure = connection.execute(
                    "SELECT * FROM shadow_campaign_closures WHERE campaign_id = ?", (cid,)
                ).fetchone()
                seed_reveal = connection.execute(
                    "SELECT * FROM shadow_seed_reveals WHERE campaign_id = ?", (cid,)
                ).fetchone()
                reveal_rows = connection.execute(
                    """
                    SELECT r.*, c.cluster_pseudonym, c.assignment_reveal_commitment,
                           c.commitment_json
                    FROM shadow_assignment_reveals r
                    JOIN shadow_assignment_commitments c
                      ON c.commitment_hash = r.commitment_hash
                    WHERE r.campaign_id = ?
                    ORDER BY r.verified_at_us, r.commitment_hash
                    """,
                    (cid,),
                ).fetchall()
                revealed = len(reveal_rows)
                matched = sum(
                    1 for row in reveal_rows if str(row["verification_status"]) == "matched"
                )
                mismatched = sum(
                    1 for row in reveal_rows if str(row["verification_status"]) == "mismatch"
                )
                if closure is None:
                    lifecycle_state = "accepting_commitments"
                elif seed_reveal is None:
                    lifecycle_state = "commitments_closed"
                elif revealed < commitment_count:
                    lifecycle_state = "seed_revealed"
                else:
                    lifecycle_state = "reveal_verification_complete"

                artifact_audit_errors: List[str] = []
                public_package: Optional[Dict[str, Any]] = None
                package_verification: Optional[Dict[str, Any]] = None
                try:
                    package_text = str(seal["public_package_json"])
                    decoded_package = json.loads(package_text)
                    if not isinstance(decoded_package, dict):
                        raise ValueError("stored public package is not an object")
                    if _canonical_json(decoded_package, "stored public package") != package_text:
                        raise ValueError("stored public package is not canonical")
                    package_verification = audit_shadow_campaign_artifacts(decoded_package)
                    public_package = decoded_package
                    expected_seal_fields = {
                        "campaign_id": package_verification["campaign_id"],
                        "origin_bundle_hash": package_verification["origin_review_bundle_hash"],
                        "committed_bundle_hash": package_verification["committed_review_bundle_hash"],
                        "protocol_hash": package_verification["committed_protocol_hash"],
                        "design_binding_hash": package_verification["design_binding_hash"],
                        "manifest_hash": package_verification["manifest_hash"],
                        "seal_hash": package_verification["seal_hash"],
                        "seed_commitment": package_verification["seed_commitment"],
                        "planned_cluster_ceiling": decoded_package["design_binding"][
                            "planned_cluster_ceiling"
                        ],
                    }
                    for field, expected_value in expected_seal_fields.items():
                        actual_value: Any = seal[field]
                        if field == "planned_cluster_ceiling":
                            actual_value = int(actual_value)
                        else:
                            actual_value = str(actual_value)
                        if actual_value != expected_value:
                            raise ValueError(f"stored seal column {field} does not match its package")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    artifact_audit_errors.append(f"campaign_seal:{exc}")

                if package_verification is not None:
                    for commitment_row in commitment_rows:
                        try:
                            commitment_text = str(commitment_row["commitment_json"])
                            commitment_value = json.loads(commitment_text)
                            if _canonical_json(
                                commitment_value, "stored assignment commitment"
                            ) != commitment_text:
                                raise ValueError("stored commitment is not canonical")
                            audited_commitment = _audit_assignment_commitment_artifact(
                                commitment_value,
                                verification=package_verification,
                            )
                            payload = audited_commitment["commitment"]
                            row_checks = {
                                "commitment_hash": audited_commitment["commitment_hash"],
                                "campaign_id": payload["campaign_id"],
                                "cluster_pseudonym": payload["cluster_pseudonym"],
                                "block_id": payload["block_id"],
                                "assignment_reveal_commitment": payload[
                                    "assignment_reveal_commitment"
                                ],
                            }
                            for field, expected_value in row_checks.items():
                                actual_value = commitment_row[field]
                                if field == "block_id":
                                    actual_value = int(actual_value)
                                else:
                                    actual_value = str(actual_value)
                                if actual_value != expected_value:
                                    raise ValueError(
                                        f"stored commitment column {field} does not match"
                                    )
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                            artifact_audit_errors.append(f"assignment_commitment:{exc}")

                if closure is not None and public_package is not None:
                    try:
                        expected_closure = _closure_artifact(public_package, commitment_count)
                        closure_text = str(closure["closure_json"])
                        if _canonical_json(expected_closure) != closure_text:
                            raise ValueError("closure artifact does not match frozen population")
                        if str(closure["closure_hash"]) != expected_closure["closure_hash"]:
                            raise ValueError("closure hash column does not match")
                        if int(closure["frozen_commitment_count"]) != commitment_count:
                            raise ValueError("closure count column does not match")
                    except (KeyError, TypeError, ValueError) as exc:
                        artifact_audit_errors.append(f"campaign_closure:{exc}")

                revealed_seed: Optional[bytes] = None
                if seed_reveal is not None and public_package is not None:
                    try:
                        revealed_seed = _decode_seed(seed_reveal["seed_material_base64url"])
                        expected_seed_reveal = _seed_reveal_artifact(
                            public_package, revealed_seed
                        )
                        reveal_text = str(seed_reveal["reveal_json"])
                        if _canonical_json(expected_seed_reveal) != reveal_text:
                            raise ValueError("seed reveal artifact does not reconstruct")
                        if str(seed_reveal["seed_reveal_hash"]) != expected_seed_reveal[
                            "seed_reveal_hash"
                        ]:
                            raise ValueError("seed reveal hash column does not match")
                    except (KeyError, TypeError, ValueError) as exc:
                        revealed_seed = None
                        artifact_audit_errors.append(f"seed_reveal:{exc}")

                arm_counts: Dict[str, int] = {}
                for reveal_row in reveal_rows:
                    try:
                        reveal_text = str(reveal_row["reveal_json"])
                        reveal = json.loads(reveal_text)
                        if revealed_seed is None or public_package is None:
                            raise ValueError("assignment reveal has no verified seed opening")
                        expected = _assignment_from_pseudonym(
                            public_package,
                            revealed_seed,
                            str(reveal_row["cluster_pseudonym"]),
                            verification=package_verification,
                        )
                        expected_reveal = expected["reveal"]
                        expected_commitment = expected["commitment"]
                        if _canonical_json(expected_reveal) != reveal_text:
                            raise ValueError("assignment reveal does not reconstruct")
                        if str(reveal_row["assignment_reveal_hash"]) != expected_reveal[
                            "assignment_reveal_hash"
                        ]:
                            raise ValueError("assignment reveal hash column does not match")
                        expected_status = (
                            "matched"
                            if hmac.compare_digest(
                                expected_commitment["commitment_hash"],
                                str(reveal_row["commitment_hash"]),
                            )
                            and hmac.compare_digest(
                                expected_reveal["assignment_reveal_hash"],
                                str(reveal_row["assignment_reveal_commitment"]),
                            )
                            else "mismatch"
                        )
                        if str(reveal_row["verification_status"]) != expected_status:
                            raise ValueError("assignment verification status does not match")
                        if expected_status == "matched":
                            arm_id = str(expected_reveal["assignment"]["arm_id"])
                            arm_counts[arm_id] = arm_counts.get(arm_id, 0) + 1
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        artifact_audit_errors.append(f"assignment_reveal:{exc}")
                if mismatched:
                    state = "reveal_verification_failed"
                elif artifact_audit_errors:
                    state = "artifact_verification_failed"
                else:
                    state = lifecycle_state
                campaigns.append(
                    {
                        "campaign_id": cid,
                        "state": state,
                        "lifecycle_state": lifecycle_state,
                        "origin_review_bundle_hash": str(seal["origin_bundle_hash"]),
                        "committed_review_bundle_hash": str(seal["committed_bundle_hash"]),
                        "protocol_hash": str(seal["protocol_hash"]),
                        "manifest_hash": str(seal["manifest_hash"]),
                        "seal_hash": str(seal["seal_hash"]),
                        "seed_commitment": str(seal["seed_commitment"]),
                        "planned_cluster_ceiling": int(seal["planned_cluster_ceiling"]),
                        "commitment_count": commitment_count,
                        "frozen_commitment_count": int(closure["frozen_commitment_count"]) if closure else None,
                        "seed_revealed": seed_reveal is not None,
                        "processed_reveal_count": revealed,
                        "verified_assignment_count": matched,
                        "matched_assignment_count": matched,
                        "mismatched_assignment_count": mismatched,
                        "whole_policy_arm_counts": arm_counts,
                        "artifact_audit_ok": not artifact_audit_errors,
                        "artifact_audit_errors": artifact_audit_errors[:8],
                        "boundaries": dict(_BOUNDARIES),
                    }
                )
            events = connection.execute(
                "SELECT * FROM shadow_registry_events ORDER BY event_sequence"
            ).fetchall()
            chain = self._verify_event_chain(events)
            registered_artifacts = sorted(
                (str(row["event_type"]), str(row["artifact_hash"])) for row in events
            )
            evidence_artifacts: List[Tuple[str, str]] = []
            evidence_artifacts.extend(
                ("campaign_sealed", str(row[0]))
                for row in connection.execute(
                    "SELECT seal_hash FROM shadow_campaign_seals"
                ).fetchall()
            )
            evidence_artifacts.extend(
                ("assignment_committed", str(row[0]))
                for row in connection.execute(
                    "SELECT commitment_hash FROM shadow_assignment_commitments"
                ).fetchall()
            )
            evidence_artifacts.extend(
                ("commitments_closed", str(row[0]))
                for row in connection.execute(
                    "SELECT closure_hash FROM shadow_campaign_closures"
                ).fetchall()
            )
            evidence_artifacts.extend(
                ("seed_revealed", str(row[0]))
                for row in connection.execute(
                    "SELECT seed_reveal_hash FROM shadow_seed_reveals"
                ).fetchall()
            )
            evidence_artifacts.extend(
                ("assignment_verified", str(row[0]))
                for row in connection.execute(
                    "SELECT assignment_reveal_hash FROM shadow_assignment_reveals"
                ).fetchall()
            )
            evidence_artifacts.sort()
            artifact_events = {
                "ok": registered_artifacts == evidence_artifacts,
                "registered_event_artifacts": len(registered_artifacts),
                "evidence_artifacts": len(evidence_artifacts),
                "reason": (
                    "verified"
                    if registered_artifacts == evidence_artifacts
                    else "event_evidence_artifact_mismatch"
                ),
            }
        return {
            "ok": (
                schema_integrity["ok"]
                and chain["ok"]
                and artifact_events["ok"]
                and all(
                    row["mismatched_assignment_count"] == 0
                    and row["artifact_audit_ok"]
                    for row in campaigns
                )
            ),
            "schema_version": SHADOW_REGISTRY_SNAPSHOT_SCHEMA_VERSION,
            "registry_schema_version": SHADOW_REGISTRY_SCHEMA_VERSION,
            "schema_integrity": schema_integrity,
            "campaign_count": len(campaigns),
            "campaigns": campaigns,
            "event_chain": chain,
            "event_artifact_consistency": artifact_events,
            "verification_level": "local_append_only_chain_without_external_anchor",
            "authenticity_proof_available": False,
            "trusted_timestamp_available": False,
            "external_transparency_anchor_available": False,
            "execution_enabled": False,
            "activation_available": False,
            "automatic_promotion_allowed": False,
        }


__all__ = [
    "MAX_SHADOW_VERIFY_BATCH",
    "SHADOW_ASSIGNMENT_ALGORITHM",
    "SHADOW_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION",
    "SHADOW_ASSIGNMENT_MANIFEST_SCHEMA_VERSION",
    "SHADOW_ASSIGNMENT_REVEAL_SCHEMA_VERSION",
    "SHADOW_CAMPAIGN_CLOSURE_SCHEMA_VERSION",
    "SHADOW_CAMPAIGN_SEAL_SCHEMA_VERSION",
    "SHADOW_CANONICALIZATION",
    "SHADOW_DESIGN_BINDING_SCHEMA_VERSION",
    "SHADOW_PUBLIC_PACKAGE_SCHEMA_VERSION",
    "SHADOW_LEGACY_BUNDLE_ENCODING",
    "SHADOW_REGISTRY_SCHEMA_VERSION",
    "SHADOW_REGISTRY_SNAPSHOT_SCHEMA_VERSION",
    "SHADOW_SEED_CAPSULE_SCHEMA_VERSION",
    "SHADOW_SEED_REVEAL_SCHEMA_VERSION",
    "RouteShadowAssignmentRegistry",
    "RouteShadowRegistryError",
    "ShadowRegistryConflictError",
    "audit_shadow_campaign_artifacts",
    "audit_shadow_seed_capsule",
    "build_shadow_design_binding",
    "compute_shadow_seed_commitment",
    "create_shadow_campaign_artifacts",
    "generate_shadow_seed",
    "prepare_shadow_assignment_commitment",
]
