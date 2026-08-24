import json
from pathlib import Path

import pytest

import source.generate_studio_runtime_manifest as manifest_generator
from source.generate_studio_runtime_manifest import (
    DISTRIBUTION_RUNTIME_ENTRYPOINTS,
    DISTRIBUTION_RUNTIME_CLOSURE_ROOTS,
    MANIFEST_SCHEMA_VERSION,
    RUNTIME_COMPATIBILITY_ENTRYPOINTS,
    RUNTIME_MODULES,
    STUDIO_RUNTIME_ENTRYPOINTS,
    STUDIO_RUNTIME_IMPORT_EXCLUSIONS,
    STUDIO_APP_VERSION,
    build_manifest,
    discover_studio_runtime_import_closure,
    main,
    validate_studio_runtime_import_closure,
)


ROOT = Path(__file__).resolve().parent

EXPECTED_CHAT_RUNTIME_MODULES = {
    "source/chat_app.py",
    "source/chat_pipeline.py",
    "source/chat_web_app.py",
    "source/interaction_planner.py",
    "source/prompt_understanding.py",
    "source/science_plan.py",
    "source/model_variants.py",
    "source/qwen_adapter_promotion.py",
    "source/qwen_chat_web_app.py",
    "runtime_python/chat_app.py",
    "runtime_python/chat_web_app.py",
    "runtime_python/interaction_planner.py",
    "runtime_python/memory_authority.py",
    "runtime_python/prompt_understanding.py",
    "runtime_python/science_plan.py",
    "runtime_python/chat_pipeline.py",
    "runtime_python/chat_memory.py",
    "runtime_python/device_utils.py",
    "runtime_python/llm_database.py",
    "runtime_python/model_variants.py",
    "runtime_python/run.py",
}


def test_checked_runtime_manifest_is_current_and_complete():
    assert main(["--repo-root", str(ROOT), "--check"]) == 0
    checked = json.loads((ROOT / "source/studio_runtime_manifest.json").read_text(encoding="utf-8"))
    generated = build_manifest(ROOT)
    assert checked == generated
    assert checked["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert checked["app_version"] == STUDIO_APP_VERSION
    assert [row["path"] for row in checked["modules"]] == list(RUNTIME_MODULES)
    assert EXPECTED_CHAT_RUNTIME_MODULES.issubset(
        {row["path"] for row in checked["modules"]}
    )
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
    }


def test_manifest_exposes_route_contract_versions_without_importing_runtime():
    manifest = build_manifest(ROOT)
    contracts = manifest["contracts"]
    assert contracts["source/chat_app.py"] == {
        "AUTO_COMPUTE_PLAN_SCHEMA_VERSION": "runtime-auto-compute-plan-v2",
        "AUTO_COMPUTE_STRATEGY": "progressive_accepted_probe",
        "DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K": 5,
        "DEFAULT_PREDICTION_STABILITY_MARGIN": 0.0005,
        "DEFAULT_PREDICTION_STABILITY_RANK_DEPTH": 3,
    }
    assert contracts["source/interaction_planner.py"] == {
        "PLANNER_VERSION": "supermix-plan-evaluate-v4",
    }
    assert (
        contracts["runtime_python/interaction_planner.py"]
        == contracts["source/interaction_planner.py"]
    )
    assert contracts["source/grounding_runtime.py"] == {
        "GROUNDING_RUNTIME_VERSION": "supermix-grounding-runtime-v6",
        "GROUNDING_SCHEMA_VERSION": "supermix-grounding-v1",
        "VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION": "supermix-verified-answer-receipt-v2",
    }
    assert (
        contracts["runtime_python/grounding_runtime.py"]
        == contracts["source/grounding_runtime.py"]
    )
    assert contracts["source/prompt_understanding.py"] == {
        "PROMPT_UNDERSTANDING_SCHEMA_VERSION": "supermix-prompt-understanding-v1",
        "PROMPT_UNDERSTANDING_VERSION": "supermix-prompt-understanding-runtime-v3",
    }
    assert (
        contracts["runtime_python/prompt_understanding.py"]
        == contracts["source/prompt_understanding.py"]
    )
    assert contracts["source/reasoning_engine.py"] == {
        "FINITE_BERNOULLI_SCHEMA_VERSION": "supermix-finite-bernoulli-scenario-v1",
        "REASONING_ENGINE_VERSION": "supermix-reasoning-engine-v5",
        "REASONING_SCHEMA_VERSION": "supermix-reasoning-v2",
    }
    assert (
        contracts["runtime_python/reasoning_engine.py"]
        == contracts["source/reasoning_engine.py"]
    )
    assert contracts["source/science_plan.py"] == {
        "SCIENCE_FORMULA_REGISTRY_VERSION": "supermix-science-formula-registry-v1",
        "SCIENCE_PLAN_ENGINE_VERSION": "supermix-science-plan-engine-v1",
        "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION": "supermix-science-plan-receipt-v1",
        "SCIENCE_PLAN_SCHEMA_VERSION": "supermix-science-plan-v1",
    }
    assert (
        contracts["runtime_python/science_plan.py"]
        == contracts["source/science_plan.py"]
    )
    assert contracts["source/memory_authority.py"] == {
        "MEMORY_AUTHORITY_POLICY_VERSION": "supermix-memory-authority-firewall-v1",
        "MEMORY_AUTHORITY_SCHEMA_VERSION": "supermix-memory-authority-v1",
        "MEMORY_EXTRACTION_RULE_VERSION": "supermix-explicit-user-memory-v3",
    }
    assert (
        contracts["runtime_python/memory_authority.py"]
        == contracts["source/memory_authority.py"]
    )
    assert contracts["source/multimodel_memory.py"] == {
        "MEMORY_SCHEMA_VERSION": "supermix-conversation-memory-v3",
    }
    assert contracts["source/qwen_adapter_promotion.py"] == {
        "BENCHMARK_SCHEMA_VERSION": "supermix-qwen-evaluation-v4",
        "GATE_SCHEMA_VERSION": "supermix-qwen-general-promotion-gate-v4",
        "PRODUCTION_POLICY_ID": "supermix-qwen-production-promotion-policy-v4",
        "PROMOTION_SCHEMA_VERSION": "supermix-qwen-adapter-promotion-v4",
        "SUPPORTED_VERIFIER_SCHEMA_VERSION": "supermix-verifier-v2",
    }
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


def test_manifest_covers_recursive_local_import_closure():
    closure, encountered_exclusions = discover_studio_runtime_import_closure(ROOT)

    assert validate_studio_runtime_import_closure(ROOT) == closure
    assert set(STUDIO_RUNTIME_ENTRYPOINTS).issubset(closure)
    assert set(RUNTIME_COMPATIBILITY_ENTRYPOINTS).issubset(closure)
    assert set(DISTRIBUTION_RUNTIME_ENTRYPOINTS).issubset(closure)
    assert set(DISTRIBUTION_RUNTIME_CLOSURE_ROOTS).issubset(closure)
    assert set(RUNTIME_MODULES).issubset(closure)
    assert {
        "source/device_utils.py",
        "source/memory_authority.py",
        "source/multimodel_catalog.py",
        "source/qwen_chat_desktop_app.py",
        "source/run.py",
    }.issubset(closure)
    assert set(encountered_exclusions) == set(STUDIO_RUNTIME_IMPORT_EXCLUSIONS)
    assert set(encountered_exclusions).isdisjoint(closure)
    assert all(
        "training-dataset" in reason
        for reason in STUDIO_RUNTIME_IMPORT_EXCLUSIONS.values()
    )


def test_import_closure_rejects_an_unmanifested_transitive_module(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "entrypoint.py").write_text("import helper\n", encoding="utf-8")
    (source_dir / "helper.py").write_text("import nested\n", encoding="utf-8")
    (source_dir / "nested.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"omits recursively imported.*nested\.py"):
        validate_studio_runtime_import_closure(
            tmp_path,
            runtime_modules=("source/entrypoint.py", "source/helper.py"),
            entrypoints=("source/entrypoint.py",),
            exclusions={},
        )


def test_import_closure_rejects_an_unmanifested_runtime_python_dependency(
    tmp_path: Path,
):
    runtime_dir = tmp_path / "runtime_python"
    runtime_dir.mkdir()
    (runtime_dir / "entrypoint.py").write_text("import helper\n", encoding="utf-8")
    (runtime_dir / "helper.py").write_text("import nested\n", encoding="utf-8")
    (runtime_dir / "nested.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"omits recursively imported.*nested\.py"):
        validate_studio_runtime_import_closure(
            tmp_path,
            runtime_modules=(
                "runtime_python/entrypoint.py",
                "runtime_python/helper.py",
            ),
            entrypoints=("runtime_python/entrypoint.py",),
            exclusions={},
        )


def test_import_closure_allows_only_reachable_justified_exclusions(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "entrypoint.py").write_text("import runtime_helper\n", encoding="utf-8")
    (source_dir / "runtime_helper.py").write_text(
        "def build_rows():\n    import training_builder\n",
        encoding="utf-8",
    )
    (source_dir / "training_builder.py").write_text("ROWS = []\n", encoding="utf-8")

    closure = validate_studio_runtime_import_closure(
        tmp_path,
        runtime_modules=("source/entrypoint.py", "source/runtime_helper.py"),
        entrypoints=("source/entrypoint.py",),
        exclusions={
            "source/training_builder.py": "lazy training-dataset builder used only by build_rows",
        },
    )
    assert "source/training_builder.py" not in closure

    with pytest.raises(ValueError, match="requires a justification"):
        validate_studio_runtime_import_closure(
            tmp_path,
            runtime_modules=("source/entrypoint.py", "source/runtime_helper.py"),
            entrypoints=("source/entrypoint.py",),
            exclusions={"source/training_builder.py": ""},
        )

    reason = "lazy training-dataset builder used only by build_rows"
    with pytest.raises(ValueError, match="stale or unreachable"):
        validate_studio_runtime_import_closure(
            tmp_path,
            runtime_modules=("source/entrypoint.py", "source/runtime_helper.py"),
            entrypoints=("source/entrypoint.py",),
            exclusions={
                "source/training_builder.py": reason,
                "source/unused_builder.py": "unused training-dataset builder",
            },
        )

    with pytest.raises(ValueError, match="cannot also be import-closure exclusions"):
        validate_studio_runtime_import_closure(
            tmp_path,
            runtime_modules=(
                "source/entrypoint.py",
                "source/runtime_helper.py",
                "source/training_builder.py",
            ),
            entrypoints=("source/entrypoint.py",),
            exclusions={"source/training_builder.py": reason},
        )


def test_studio_packaging_sources_are_machine_portable_and_include_console_contract():
    spec = (ROOT / "SupermixStudioDesktop.spec").read_text(encoding="utf-8")
    build = (ROOT / "source/build_supermix_studio_desktop_exe.ps1").read_text(encoding="utf-8")
    installer = (ROOT / "installer/SupermixStudioDesktop.iss").read_text(encoding="utf-8")
    installer_build = (ROOT / "source/build_supermix_studio_desktop_installer.ps1").read_text(
        encoding="utf-8"
    )

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
    assert "source\\reasoning_engine.py" in build
    assert "source\\science_plan.py" in build
    assert 'repo_root / "source" / "reasoning_engine.py"' in spec
    assert 'repo_root / "source" / "science_plan.py"' in spec
    assert "RuntimeOnly" in build
    assert "runtime_only_base_model_plus_model_store" in build
    assert f'#define MyAppVersion "{STUDIO_APP_VERSION}"' in installer
    assert STUDIO_APP_VERSION == "71.0.0"
    assert "DefaultDirName={localappdata}\\Programs\\Supermix Studio" in installer
    assert "PrivilegesRequired=lowest" in installer
    assert "$EffectiveSetupBaseName" in installer_build
    assert '$SetupPath = Join-Path $InstallerDir "$EffectiveSetupBaseName.exe"' in installer_build
    assert 'Join-Path $InstallerDir "SupermixStudioDesktopSetup.exe"' not in installer_build
    assert "Expected freshly built installer" in installer_build


def test_qwen_packaging_sources_are_machine_portable():
    """The Studio spec has been guarded for portability; the Qwen ones were not.

    Both Qwen specs are regenerated PyInstaller outputs and had been committed
    with absolute paths into a *different* sibling checkout, so building from
    the spec resolved datas and the icon outside this repository.
    """

    for name in ("SupermixQwenDesktop.spec", "SupermixQwenDesktopV26.spec"):
        spec = (ROOT / name).read_text(encoding="utf-8")
        assert "Supermix_27" not in spec, f"{name} references a foreign checkout"
        assert "C:\\Users\\" not in spec, f"{name} hardcodes a machine path"

    for name in ("build_qwen_chat_desktop_exe.ps1", "source/build_qwen_chat_desktop_exe.ps1"):
        build = (ROOT / name).read_text(encoding="utf-8")
        assert "Supermix_27" not in build
        assert "C:\\Users\\" not in build


def test_qwen_packaging_includes_runtime_import_dependencies():
    specs = (
        ROOT / "SupermixQwenDesktop.spec",
        ROOT / "SupermixQwenDesktopV26.spec",
    )
    builds = (
        ROOT / "build_qwen_chat_desktop_exe.ps1",
        ROOT / "source/build_qwen_chat_desktop_exe.ps1",
    )

    for path in specs:
        text = path.read_text(encoding="utf-8")
        assert "source\\\\\\\\qwen_adapter_promotion.py" in text
        assert "source\\\\\\\\prompt_understanding.py" in text
        assert "runtime_python\\\\\\\\prompt_understanding.py" in text
        assert "source\\\\\\\\science_plan.py" in text
        assert "runtime_python\\\\\\\\science_plan.py" in text
        assert "source\\\\\\\\conversation_state.py" in text
        assert "source\\\\\\\\conversation_directive.py" in text

    for path in builds:
        text = path.read_text(encoding="utf-8")
        assert "source\\\\qwen_adapter_promotion.py;source" in text
        assert "source\\\\prompt_understanding.py;source" in text
        assert "runtime_python\\\\prompt_understanding.py;runtime_python" in text
        assert "source\\\\science_plan.py;source" in text
        assert "runtime_python\\\\science_plan.py;runtime_python" in text
        assert "source\\\\conversation_state.py;source" in text
        assert "source\\\\conversation_directive.py;source" in text
        assert '"promotion_manifest.json", "promotion_gate.json"' in text
        assert "validate_promoted_adapter" in text
        assert "adapter_activation = $AdapterActivation" in text
        assert "promotion_manifest_relative_path" in text
        assert "promotion_gate_relative_path" in text
        assert "Staged promoted adapter failed receipt validation" in text


def test_manifest_check_rejects_tampered_or_syntax_broken_runtime_module(
    tmp_path, monkeypatch, capsys
):
    runtime = tmp_path / "source/chat_web_app.py"
    runtime.parent.mkdir(parents=True)
    runtime.write_text("RUNTIME_SENTINEL = 1\n", encoding="utf-8")
    output = tmp_path / "source/studio_runtime_manifest.json"

    monkeypatch.setattr(
        manifest_generator,
        "RUNTIME_MODULES",
        ("source/chat_web_app.py",),
    )
    monkeypatch.setattr(
        manifest_generator,
        "STUDIO_RUNTIME_ENTRYPOINTS",
        ("source/chat_web_app.py",),
    )
    monkeypatch.setattr(manifest_generator, "STUDIO_RUNTIME_IMPORT_EXCLUSIONS", {})
    monkeypatch.setattr(manifest_generator, "CONTRACT_CONSTANTS", {})

    args = ["--repo-root", str(tmp_path), "--output", str(output)]
    assert main(args) == 0

    runtime.write_text("RUNTIME_SENTINEL = 2\n", encoding="utf-8")
    assert main([*args, "--check"]) == 1
    assert "manifest is stale" in capsys.readouterr().err

    runtime.write_text("def syntax_broken(:\n", encoding="utf-8")
    assert main([*args, "--check"]) == 2
    assert "is not valid Python" in capsys.readouterr().err


def test_manifest_module_hashes_are_portable_across_lf_and_crlf_checkouts(
    tmp_path, monkeypatch
):
    runtime = tmp_path / "source/chat_web_app.py"
    runtime.parent.mkdir(parents=True)
    lf_payload = b"RUNTIME_SENTINEL = 1\n"
    runtime.write_bytes(lf_payload)

    monkeypatch.setattr(
        manifest_generator,
        "RUNTIME_MODULES",
        ("source/chat_web_app.py",),
    )
    monkeypatch.setattr(
        manifest_generator,
        "STUDIO_RUNTIME_ENTRYPOINTS",
        ("source/chat_web_app.py",),
    )
    monkeypatch.setattr(manifest_generator, "STUDIO_RUNTIME_IMPORT_EXCLUSIONS", {})
    monkeypatch.setattr(manifest_generator, "CONTRACT_CONSTANTS", {})

    lf_manifest = build_manifest(tmp_path)
    runtime.write_bytes(lf_payload.replace(b"\n", b"\r\n"))
    crlf_manifest = build_manifest(tmp_path)

    assert crlf_manifest == lf_manifest
    assert lf_manifest["modules"][0]["size_bytes"] == len(lf_payload)
