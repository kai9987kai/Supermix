import json
from pathlib import Path

from source.generate_studio_runtime_manifest import (
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_MODULES,
    STUDIO_APP_VERSION,
    build_manifest,
    main,
)


ROOT = Path(__file__).resolve().parent


def test_checked_runtime_manifest_is_current_and_complete():
    assert main(["--repo-root", str(ROOT), "--check"]) == 0
    checked = json.loads((ROOT / "source/studio_runtime_manifest.json").read_text(encoding="utf-8"))
    generated = build_manifest(ROOT)
    assert checked == generated
    assert checked["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert checked["app_version"] == STUDIO_APP_VERSION
    assert [row["path"] for row in checked["modules"]] == list(RUNTIME_MODULES)
    assert all(len(row["sha256"]) == 64 and row["size_bytes"] > 0 for row in checked["modules"])
    assert checked["entrypoints"]["route_study_console"] == "source/route_policy_protocol_cli.py"
    assert checked["expected_windows_artifacts"]["route_study_console"] == "SupermixRouteStudy.exe"
    assert checked["entrypoints"]["route_shadow_console"] == "source/route_policy_shadow_cli.py"
    assert checked["expected_windows_artifacts"]["route_shadow_console"] == "SupermixRouteShadow.exe"
    assert checked["package_guards"] == {
        "route_protocol_activation_available": False,
        "route_protocol_assignment_implementation_available": False,
        "route_rehearsal_writes_ledger": False,
        "route_ledger_requires_executed_assignment_namespace": True,
        "route_review_bundle_full_source_reconstruction": True,
        "route_review_bundle_authenticity_proof_available": False,
        "route_review_bundle_trusted_timestamp_available": False,
        "route_shadow_registry_available": True,
        "route_shadow_assignment_executes_routes": False,
        "route_shadow_private_seed_persisted_before_reveal": False,
        "route_shadow_authenticity_proof_available": False,
        "route_shadow_trusted_timestamp_available": False,
        "automatic_policy_promotion_available": False,
    }


def test_manifest_exposes_route_contract_versions_without_importing_runtime():
    manifest = build_manifest(ROOT)
    contracts = manifest["contracts"]
    assert contracts["source/route_policy_ledger.py"]["OUTCOME_CONTRACT_SCHEMA_VERSION"] == (
        "route-outcome-contract-v1"
    )
    assert contracts["source/route_policy_ledger.py"][
        "EXECUTED_ASSIGNMENT_COMMITMENT_SCHEMA_VERSION"
    ] == "route-execution-assignment-v1"
    assert contracts["source/route_policy_explorer.py"]["STUDY_PLAN_SCHEMA_VERSION"] == (
        "route-exploration-plan-v1"
    )
    assert contracts["source/route_policy_protocol.py"]["PROTOCOL_SCHEMA_VERSION"] == (
        "route-study-protocol-preflight-v1"
    )
    assert contracts["source/route_policy_protocol.py"]["REVIEW_BUNDLE_SCHEMA_VERSION"] == (
        "route-study-review-bundle-v1"
    )
    assert contracts["source/route_policy_protocol.py"][
        "PROTOCOL_BUILD_INPUT_SCHEMA_VERSION"
    ] == "route-study-protocol-build-input-v1"
    assert contracts["source/route_policy_shadow_registry.py"][
        "SHADOW_REGISTRY_SCHEMA_VERSION"
    ] == 1
    assert contracts["source/route_policy_shadow_registry.py"][
        "SHADOW_ASSIGNMENT_ALGORITHM"
    ] == "hkdf-sha256-hmac-sha256-whole-policy-bps-v1"
    assert contracts["source/route_policy_shadow_registry.py"][
        "SHADOW_CANONICALIZATION"
    ] == "rfc8785-jcs-restricted-ijson-integer-v1"
    assert len(
        contracts["source/route_policy_shadow_registry.py"][
            "SHADOW_SCHEMA_OBJECTS_SHA256"
        ]
    ) == 64


def test_studio_packaging_sources_are_machine_portable_and_include_console_contract():
    spec = (ROOT / "SupermixStudioDesktop.spec").read_text(encoding="utf-8")
    build = (ROOT / "source/build_supermix_studio_desktop_exe.ps1").read_text(encoding="utf-8")
    installer = (ROOT / "installer/SupermixStudioDesktop.iss").read_text(encoding="utf-8")

    assert "Supermix_27" not in spec
    assert "C:\\Users\\" not in spec
    assert "Supermix_27" not in build
    assert "C:\\Users\\" not in build
    assert "studio_runtime_manifest.json" in spec
    assert "generate_studio_runtime_manifest.py" in build
    assert "--check" in build
    assert "SupermixRouteStudy" in build
    assert "route_policy_protocol_cli.py" in build
    assert "SupermixRouteShadow" in build
    assert "route_policy_shadow_cli.py" in build
    assert "studio_runtime_manifest.json" in build
    assert f'#define MyAppVersion "{STUDIO_APP_VERSION}"' in installer
    assert "DefaultDirName={localappdata}\\Programs\\Supermix Studio" in installer
    assert "PrivilegesRequired=lowest" in installer
