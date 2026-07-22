import json
import sys
from pathlib import Path

import pytest

from source.multimodel_catalog import ModelRecord

SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import source.multimodel_runtime as runtime_module
from source.multimodel_runtime import ChatResult, UnifiedModelManager


def _record(key: str, kind: str, capabilities: tuple[str, ...], score: float | None = None) -> ModelRecord:
    return ModelRecord(
        key=key,
        label=key,
        family="test",
        kind=kind,
        capabilities=capabilities,
        zip_path=Path(f"{key}.zip"),
        common_row_key=key,
        common_overall_exact=score,
    )


class _FakeBackend:
    def __init__(self, record: ModelRecord) -> None:
        self.record = record

    def chat(self, session_id: str, prompt: str, settings: dict) -> ChatResult:
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=f"reply via {self.record.key}",
            prompt_used=prompt,
        )

    def clear(self, session_id: str) -> None:
        return None

    def unload(self) -> None:
        return None


def test_champion_backend_forwards_runtime_compute_controls_and_telemetry() -> None:
    record = _record("v51-runtime", "champion_chat", ("chat",), 0.9)
    captured = {}

    class _Engine:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {
                "response": "computed response",
                "timing_ms": {"total": 12.5},
                "compute": {
                    "applied": True,
                    "cycles_used": 5.0,
                    "exit_reason": "prediction_stable",
                },
            }

    backend = runtime_module.ChampionChatBackend.__new__(runtime_module.ChampionChatBackend)
    backend.record = record
    backend.engine = _Engine()
    result = backend.chat(
        "compute-session",
        "Solve this carefully.",
        {
            "reasoning_cycles": 5,
            "adaptive_compute": True,
            "adaptive_exit_tol": 0.04,
            "adaptive_exit_entropy": 0.25,
            "prediction_stability_patience": 3,
            "prediction_stability_tol": 0.01,
            "auto_compute": True,
        },
    )

    assert captured["reasoning_cycles"] == 5
    assert captured["adaptive_compute"] is True
    assert captured["adaptive_exit_tol"] == 0.04
    assert captured["adaptive_exit_entropy"] == 0.25
    assert captured["prediction_stability_patience"] == 3
    assert captured["prediction_stability_tol"] == 0.01
    assert captured["auto_compute"] is True
    assert result.compute == {
        "applied": True,
        "cycles_used": 5.0,
        "exit_reason": "prediction_stable",
    }
    assert result.to_dict()["compute"]["exit_reason"] == "prediction_stable"


def _auto_route_records() -> tuple[ModelRecord, ...]:
    return (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )


def test_manager_exposes_only_read_only_shadow_registry_status(tmp_path: Path) -> None:
    from source.route_policy_ledger import hash_session_identity
    from source.route_policy_protocol import build_route_study_review_bundle_from_input
    from source.route_policy_protocol_cli import _example_bundle_input
    from source.route_policy_shadow_registry import RouteShadowAssignmentRegistry

    manager = UnifiedModelManager(
        records=(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    absent = manager.route_shadow_registry_snapshot()
    assert absent["available"] is False
    assert absent["read_only"] is True
    assert absent["execution_enabled"] is False
    assert not manager.route_shadow_registry_path.exists()

    registry = RouteShadowAssignmentRegistry(manager.route_shadow_registry_path)
    bundle = build_route_study_review_bundle_from_input(_example_bundle_input())
    sealed = registry.seal_campaign(bundle, bytes(range(32)))
    campaign_id = sealed["public_package"]["campaign_seal"]["seal"]["campaign_id"]
    registry.append_assignment_commitment(
        campaign_id=campaign_id,
        seed_capsule=sealed["private_seed_capsule"],
        cluster_identifier=hash_session_identity("runtime-status-private-cluster"),
    )
    before_database = manager.route_shadow_registry_path.read_bytes()

    status = manager.route_shadow_registry_snapshot()
    assert status["available"] is True
    assert status["status"] == "verified"
    assert status["read_only"] is True
    assert status["registry_location"] == "memory/route-policy-shadow-registry.sqlite3"
    assert "registry_path" not in status
    assert status["campaign_count"] == 1
    assert status["campaigns"][0]["artifact_audit_ok"] is True
    assert status["campaigns"][0]["commitment_count"] == 1
    assert status["execution_enabled"] is False
    assert status["activation_available"] is False
    assert status["automatic_promotion_allowed"] is False
    assert manager.route_shadow_registry_path.read_bytes() == before_database
    rendered = json.dumps(status, sort_keys=True)
    assert sealed["private_seed_capsule"]["seed_material_base64url"] not in rendered
    assert "runtime-status-private-cluster" not in rendered


def test_collective_panel_includes_omni_collective_v2_v3_v4_v5_v6_v7_v8_v8_preview_v40_and_domain_specialists(tmp_path: Path) -> None:
    records = (
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("protein_folding_micro_v1", "protein_folding", ("chat",), None),
        _record("mattergen_micro_v1", "mattergen_generation", ("chat",), None),
        _record("three_d_generation_micro_v1", "three_d_generation", ("chat",), None),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), None),
        _record("omni_collective_v8_preview", "omni_collective_v8", ("chat", "vision"), None),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
        _record("science_vision_micro_v1", "image_recognition", ("chat", "vision"), None),
        _record("v38_native_xlite_fp16", "native_image", ("image",), 0.01),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    consultants = manager._collective_consultants()
    keys = [record.key for record in consultants]
    assert "protein_folding_micro_v1" in keys
    assert "mattergen_micro_v1" in keys
    assert "three_d_generation_micro_v1" in keys
    assert "omni_collective_v2" in keys
    assert "omni_collective_v3" in keys
    assert "omni_collective_v4" in keys
    assert "omni_collective_v5" in keys
    assert "omni_collective_v6" in keys
    assert "omni_collective_v7" in keys
    assert "omni_collective_v8" in keys
    assert "omni_collective_v8_preview" in keys
    assert "v40_benchmax" in keys
    assert "v38_native_xlite_fp16" not in keys


def test_default_text_record_prefers_v40_benchmax(tmp_path: Path) -> None:
    records = (
        _record("v33_final", "champion_chat", ("chat",), 0.18),
        _record("omni_collective_v2", "omni_collective", ("chat", "vision"), None),
        _record("omni_collective_v3", "omni_collective_v3", ("chat", "vision"), None),
        _record("omni_collective_v4", "omni_collective_v4", ("chat", "vision"), None),
        _record("omni_collective_v5", "omni_collective_v5", ("chat", "vision"), None),
        _record("omni_collective_v6", "omni_collective_v6", ("chat", "vision"), None),
        _record("omni_collective_v7", "omni_collective_v7", ("chat", "vision"), 0.1067),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), None),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    chosen = manager._default_text_record()
    assert chosen.key == "v40_benchmax"


def test_default_text_record_keeps_stable_v46_preference_even_when_v48_exists(tmp_path: Path) -> None:
    records = (
        _record("omni_collective_v48", "omni_collective_v48", ("chat", "vision"), 0.7557),
        _record("omni_collective_v47", "omni_collective_v47", ("chat", "vision"), 0.7110),
        _record("omni_collective_v46", "omni_collective_v46", ("chat", "vision"), 0.7477),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.24),
        _record("omni_collective_v41", "omni_collective_v41", ("chat", "vision"), 0.17),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    chosen = manager._default_text_record()
    assert chosen.key == "omni_collective_v46"


def test_collective_consultants_can_follow_configured_keys_and_keep_chosen_first(tmp_path: Path) -> None:
    records = (
        _record("omni_collective_v41", "omni_collective_v41", ("chat", "vision"), 0.17),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.24),
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.21),
        _record("qwen_v28", "qwen_adapter", ("chat",), 0.02),
        _record("math_equation_micro_v1", "math_equation", ("chat",), 0.01),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    chosen = records[4]
    consultants = manager._collective_consultants(
        settings={
            "collective_consultant_keys": ["omni_collective_v41", "v40_benchmax", "qwen_v28"],
            "collective_consultant_limit": 4,
        },
        chosen_record=chosen,
    )

    assert [record.key for record in consultants] == [
        "math_equation_micro_v1",
        "omni_collective_v41",
        "v40_benchmax",
        "qwen_v28",
    ]


def test_handle_prompt_falls_back_when_requested_backend_cannot_initialize(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v48", "omni_collective_v48", ("chat", "vision"), 0.7557),
        _record("omni_collective_v47", "omni_collective_v47", ("chat", "vision"), 0.7110),
        _record("omni_collective_v46", "omni_collective_v46", ("chat", "vision"), 0.7477),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    def fake_build_backend(record: ModelRecord):
        if record.key == "omni_collective_v48":
            raise RuntimeError("broken weights")
        return _FakeBackend(record)

    monkeypatch.setattr(manager, "_build_backend", fake_build_backend)

    payload = manager.handle_prompt(
        session_id="fallback-session",
        prompt="Explain the result.",
        model_key="omni_collective_v48",
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert payload["ok"] is True
    assert payload["model_key"] == "omni_collective_v47"
    assert "fell back" in payload["route_reason"].lower()


def test_auto_agent_mode_routes_complex_prompt_to_collective_loop(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        captured["pre_execution_decision"] = manager.route_policy_ledger.list_decisions(
            session_id="auto-complex-session"
        )[0]
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response="auto routed through collective loop",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)

    payload = manager.handle_prompt(
        session_id="auto-complex-session",
        prompt=(
            "Research the latest evidence, design and implement a multi-step runtime integration, "
            "debug regressions, verify with tests, and explain benchmark tradeoffs."
        ),
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    pre_execution_decision = captured["pre_execution_decision"]
    assert set(pre_execution_decision["outcome_contracts"]) == {
        "route_success",
        "user_quality_rating",
        "cost",
        "latency",
    }
    assert pre_execution_decision["outcome_contracts_precommitted_at_begin"] is True
    assert pre_execution_decision["outcome_contracts_defaulted_at_begin"] is False
    assert pre_execution_decision["outcome_events"] == []
    for outcome_name, contract in pre_execution_decision["outcome_contracts"].items():
        assert contract["schema_version"] == "route-outcome-contract-v1"
        assert contract["outcome_name"] == outcome_name
        assert contract["precommitted"] is True
        assert contract["commitment_source"] == "caller"
        assert len(contract["contract_hash"]) == 64
    assert captured["collective_mode"] is True
    assert captured["settings"]["agent_mode"] == "collective_loop"
    assert captured["settings"]["reasoning_cycles"] == policy["reasoning_cycles"]
    assert captured["settings"]["adaptive_compute"] is True
    assert policy["selected_agent_mode"] == "collective_loop"
    assert policy["policy_id"] == "auto-route-v2"
    assert policy["policy_version"] == "2.0.0"
    assert policy["feature_schema_version"] == "route-context-v1"
    assert policy["decision_type"] == "deterministic"
    assert policy["probability_stage"] == "post_filter"
    assert policy["logging_propensity"] == 1.0
    assert policy["post_filter_action_probabilities"]["collective_loop"] == 1.0
    assert sum(policy["post_filter_action_probabilities"].values()) == 1.0
    assert policy["counterfactual_support"] == "none_deterministic_logging"
    assert policy["logging_support"]["schema_version"] == "route-support-v1"
    assert policy["logging_support"]["decision_type"] == "deterministic"
    assert policy["logging_support"]["sampler"]["exploration_rate"] == 0.0
    assert policy["candidate_set_hash"] == policy["logging_support"]["candidate_set_hash"]
    assert policy["distribution_hash"] == policy["logging_support"]["distribution_hash"]
    assert payload["route_id"] == payload["agent_trace"]["route_id"]
    assert len(payload["route_id"]) == 32
    assert policy["runtime_compute_request"] == {
        "reasoning_cycles": policy["reasoning_cycles"],
        "adaptive_compute": True,
        "source": "auto_route_policy",
    }
    assert policy["collective_available"] is True
    assert policy["route_economics_estimate"]["selected_agent_mode"] == "collective_loop"
    assert policy["route_economics_estimate"]["estimated_model_calls"] >= 8
    assert payload["agent_trace"]["route_economics"]["estimate"]["selected_agent_mode"] == "collective_loop"
    assert payload["agent_trace"]["route_economics"]["actual"]["elapsed_ms"] >= 0
    assert payload["agent_trace"]["requested_agent_mode"] == "auto"
    assert payload["agent_trace"]["resolved_agent_mode"] == "collective_loop"
    assert "Auto orchestration selected collective_loop" in payload["route_reason"]

    policy_lab = manager.route_policy_lab_snapshot("auto-complex-session", profile="efficiency")
    assert policy_lab["profile"]["name"] == "efficiency"
    assert policy_lab["support"]["usage"]["unique_route_ids"] == 1
    assert policy_lab["evidence_source"] == "durable_sqlite_ledger"
    assert policy_lab["evaluation_readiness"]["schema_version"] == "route-readiness-v2"
    assert policy_lab["evaluation_readiness"]["logging_integrity"]["valid_routes"] == 0
    assert policy_lab["evaluation_readiness"]["target_overlap"]["effective_sample_size"] == 0.0
    assert policy_lab["evaluation_readiness"]["ready_for_external_ope"] is False
    assert "no_valid_randomized_overlap" in policy_lab["promotion_gate"]["blocking_reason_codes"]
    assert policy_lab["promotion_gate"]["deployment"] == "shadow_only"
    assert policy_lab["promotion_gate"]["automatic_promotion_allowed"] is False
    assert policy_lab["evaluation_readiness"]["policy_value_estimated"] is False
    outcome_maturity = policy_lab["outcome_contract_maturity"]
    assert outcome_maturity["schema_version"] == "route-outcome-maturity-v1"
    assert outcome_maturity["descriptive_only"] is True
    assert outcome_maturity["policy_value_estimate"] is None
    assert outcome_maturity["causal_identification"] == "not_performed"
    assert outcome_maturity["missingness_identification"] == "not_performed"
    assert any(
        "diagnostic-only telemetry" in warning
        and "not a policy-value estimator" in warning
        for warning in policy_lab["warnings"]
    )
    assert policy_lab["durable_ledger"]["counts"] == {
        "started": 1,
        "completed": 1,
        "failed": 0,
        "inflight": 0,
    }
    decision = manager.route_policy_ledger.get_decision(payload["route_id"])
    assert decision["status"] == "completed"
    assert decision["chosen_mode"] == "collective_loop"
    assert decision["executed_mode"] == "collective_loop"
    assert decision["decision_context"]["requested_agent_mode"] == "auto"
    assert decision["decision_type"] == "deterministic"
    assert decision["probability_stage"] == "post_filter"
    assert decision["candidate_set_hash"] == policy["candidate_set_hash"]
    assert decision["distribution_hash"] == policy["distribution_hash"]
    assert "actual" not in decision["actual_economics"]
    assert decision["actual_economics"]["cost_units"] == pytest.approx(
        payload["agent_trace"]["route_economics"]["actual"]["cost_units"]
    )
    assert decision["actual_economics"]["elapsed_ms"] == pytest.approx(
        payload["agent_trace"]["route_economics"]["actual"]["elapsed_ms"]
    )
    terminal_outcome_names = [event["outcome_name"] for event in decision["outcome_events"]]
    assert terminal_outcome_names.count("route_success") == 1
    assert terminal_outcome_names.count("cost") == 1
    assert terminal_outcome_names.count("latency") == 1
    assert "user_quality_rating" not in terminal_outcome_names
    terminal_events = {event["outcome_name"]: event for event in decision["outcome_events"]}
    assert terminal_events["route_success"]["value"] is True
    assert terminal_events["route_success"]["observation_status"] == "observed"
    assert terminal_events["route_success"]["event_source"] == "route_completion"
    assert terminal_events["cost"]["observation_status"] == "observed"
    assert terminal_events["cost"]["value"] == pytest.approx(
        decision["actual_economics"]["cost_units"]
    )
    assert terminal_events["latency"]["observation_status"] == "observed"
    assert terminal_events["latency"]["value"] == pytest.approx(
        decision["actual_economics"]["elapsed_ms"]
    )
    assert payload["agent_trace"]["route_ledger"]["status"] == "completed"
    assert payload["agent_trace"]["route_ledger"]["candidate_set_hash"] == policy["candidate_set_hash"]
    first_feedback = manager.record_route_feedback(
        session_id="auto-complex-session",
        feedback={"route_id": payload["route_id"], "rating": "up", "feedback_intent": "good"},
    )
    assert first_feedback["durable_feedback"] == {
        "status": "committed",
        "revision": 1,
        "idempotent": False,
        "eligible_for_readiness": True,
    }
    observed_lab = manager.route_policy_lab_snapshot("auto-complex-session", profile="efficiency")
    assert observed_lab["matched_observed"]["avg_cost_units"] == pytest.approx(
        decision["actual_economics"]["cost_units"]
    )
    assert observed_lab["matched_observed"]["avg_elapsed_ms"] == pytest.approx(
        decision["actual_economics"]["elapsed_ms"]
    )
    identical_retry = manager.record_route_feedback(
        session_id="auto-complex-session",
        feedback={"route_id": payload["route_id"], "rating": "up", "feedback_intent": "good"},
    )
    assert identical_retry["durable_feedback"]["revision"] == 1
    assert identical_retry["durable_feedback"]["idempotent"] is True
    assert identical_retry["feedback"]["feedback_revision"] == 1
    decision_with_feedback = manager.route_policy_ledger.get_decision(payload["route_id"])
    assert decision_with_feedback["feedback_status"] == "known"
    assert decision_with_feedback["feedback_revision_count"] == 1
    assert decision_with_feedback["latest_feedback"]["feedback"]["observation_status"] == "observed"
    feedback_outcome_names = [
        event["outcome_name"] for event in decision_with_feedback["outcome_events"]
    ]
    assert feedback_outcome_names.count("user_quality_rating") == 1
    quality_event = next(
        event
        for event in decision_with_feedback["outcome_events"]
        if event["outcome_name"] == "user_quality_rating"
    )
    assert quality_event["value"] == 1
    assert quality_event["observation_status"] == "observed"
    assert quality_event["event_source"] == "feedback_revision"
    non_quality_feedback = manager.record_route_feedback(
        session_id="auto-complex-session",
        feedback={
            "route_id": payload["route_id"],
            "rating": "down",
            "feedback_intent": "too_slow",
        },
    )
    assert non_quality_feedback["durable_feedback"]["revision"] == 2
    decision_with_non_quality_feedback = manager.route_policy_ledger.get_decision(payload["route_id"])
    assert decision_with_non_quality_feedback["feedback_revision_count"] == 2
    quality_events = [
        event
        for event in decision_with_non_quality_feedback["outcome_events"]
        if event["outcome_name"] == "user_quality_rating"
    ]
    assert len(quality_events) == 2
    assert [event["observation_status"] for event in quality_events] == [
        "observed",
        "not_observed",
    ]
    assert quality_events[1]["value"] is None
    assert quality_events[1]["event_source"] == "feedback_revision"
    with pytest.raises(ValueError, match="does not belong to this session"):
        manager.record_route_feedback(
            session_id="different-session",
            feedback={"route_id": payload["route_id"], "rating": "up", "feedback_intent": "good"},
        )


def test_durable_feedback_survives_compatibility_mirror_failure_and_reconciles(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    session_id = "mirror-reconciliation-session"
    route_id = "11111111-1111-4111-8111-111111111111"
    manager.route_policy_ledger.begin_decision(
        session_id=session_id,
        policy_name="explicit-route-v1",
        policy_version="1.0.0",
        policy_schema_version="route-context-v1",
        decision_context={"action_mode": "text", "selected_agent_mode": "off"},
        eligible_modes=["off"],
        chosen_mode="off",
        action_probabilities={"off": 1.0},
        estimated_economics={"estimated_cost_units": 1.0},
        route_id=route_id,
    )
    manager.route_policy_ledger.complete_decision(
        route_id,
        success=True,
        executed_mode="off",
        actual_economics={"cost_units": 1.0},
    )

    original_commit = manager.memory_store.commit_feedback

    def fail_mirror(**_kwargs):
        raise OSError("simulated compatibility mirror failure")

    monkeypatch.setattr(manager.memory_store, "commit_feedback", fail_mirror)
    accepted = manager.record_route_feedback(
        session_id=session_id,
        feedback={"route_id": route_id, "rating": "up", "feedback_intent": "good"},
    )

    assert accepted["ok"] is True
    assert accepted["compatibility_mirror"]["status"] == "pending_reconciliation"
    assert accepted["durable_feedback"]["revision"] == 1
    durable = manager.route_policy_ledger.get_decision(route_id)
    assert durable["feedback_revision_count"] == 1
    durable_quality_events = [
        event for event in durable["outcome_events"]
        if event["outcome_name"] == "user_quality_rating"
    ]
    assert len(durable_quality_events) == 1
    assert durable_quality_events[0]["value"] == 1
    assert durable_quality_events[0]["event_source"] == "feedback_revision"
    assert manager.memory_store.load_session(session_id)["route_feedback"] == []

    monkeypatch.setattr(manager.memory_store, "commit_feedback", original_commit)
    reconciled = manager.record_route_feedback(
        session_id=session_id,
        feedback={"route_id": route_id, "rating": "up", "feedback_intent": "good"},
    )
    assert reconciled["durable_feedback"]["idempotent"] is True
    assert reconciled["compatibility_mirror"]["status"] == "committed"
    assert reconciled["feedback"]["feedback_revision"] == 1
    assert manager.route_policy_ledger.get_decision(route_id)["feedback_revision_count"] == 1
    assert [
        event["outcome_name"]
        for event in manager.route_policy_ledger.get_decision(route_id)["outcome_events"]
    ].count("user_quality_rating") == 1


def test_legacy_json_policy_lab_fallback_is_descriptive_only(tmp_path: Path) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    session_id = "legacy-policy-lab-session"
    manager.memory_store.add_route_usage(
        session_id=session_id,
        route_id="legacy-route-1",
        prompt="Legacy compatibility evidence",
        selected_agent_mode="off",
        route_economics={"actual": {"cost_units": 1.0, "elapsed_ms": 20.0}},
        auto_agent_policy={
            "policy_id": "auto-route-v2",
            "policy_version": "2.0.0",
            "feature_schema_version": "route-context-v1",
            "decision_type": "deterministic",
            "probability_stage": "post_filter",
            "score": 0,
            "selected_agent_mode": "off",
            "allowed_agent_modes": ["off"],
            "eligible_actions": ["off"],
            "post_filter_action_probabilities": {"off": 1.0},
        },
    )

    report = manager.route_policy_lab_snapshot(session_id)
    lifecycle = report["evaluation_readiness"]["lifecycle_integrity"]
    assert report["evidence_source"] == "legacy_json_compatibility"
    assert report["compatibility_view"]["used_for_analysis"] is True
    assert report["compatibility_view"]["used_for_readiness"] is False
    assert report["compatibility_view"]["eligible_for_readiness"] is False
    assert lifecycle["durable_lifecycle_present"] is False
    assert lifecycle["reconciled"] is False
    assert "durable_lifecycle_required" in report["promotion_gate"]["blocking_reason_codes"]
    assert report["evaluation_readiness"]["ready_for_external_ope"] is False


def test_route_ledger_records_runtime_failure_before_reraising(tmp_path: Path, monkeypatch) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    observed = {}

    def fail_loop_agent(**_kwargs):
        observed["during"] = manager.route_policy_ledger.report(session_id="failed-route-session")
        observed["policy_lab_during"] = manager.route_policy_lab_snapshot(
            "failed-route-session",
            profile="balanced",
        )
        raise RuntimeError("simulated route backend failure")

    monkeypatch.setattr(manager, "_run_loop_agent_text", fail_loop_agent)

    with pytest.raises(RuntimeError, match="simulated route backend failure"):
        manager.handle_prompt(
            session_id="failed-route-session",
            prompt=(
                "Research the latest evidence, implement the whole project, verify every regression, "
                "and complete a multi-step production audit."
            ),
            model_key="omni_collective_v8",
            action_mode="text",
            settings={
                "agent_mode": "auto",
                "memory_enabled": False,
                "web_search_enabled": False,
                "cmd_open_enabled": False,
            },
        )

    assert observed["during"]["counts"] == {
        "started": 1,
        "completed": 0,
        "failed": 0,
        "inflight": 1,
    }
    during_lab = observed["policy_lab_during"]
    during_lifecycle = during_lab["evaluation_readiness"]["lifecycle_integrity"]
    assert during_lab["evidence_source"] == "durable_sqlite_ledger"
    assert during_lifecycle == {
        "durable_lifecycle_present": True,
        "durable_started_routes": 1,
        "durable_terminal_routes": 0,
        "durable_inflight_routes": 1,
        "replay_usage_routes": 1,
        "replay_decision_coverage_rate": 1.0,
        "reconciled": False,
    }
    assert "lifecycle_not_reconciled" in during_lab["promotion_gate"]["blocking_reason_codes"]
    report = manager.route_policy_ledger.report(session_id="failed-route-session")
    assert report["counts"] == {"started": 1, "completed": 0, "failed": 1, "inflight": 0}
    failed_lab = manager.route_policy_lab_snapshot("failed-route-session", profile="balanced")
    failed_lifecycle = failed_lab["evaluation_readiness"]["lifecycle_integrity"]
    assert failed_lab["durable_ledger"]["counts"] == {
        "started": 1,
        "completed": 0,
        "failed": 1,
        "inflight": 0,
    }
    assert failed_lifecycle == {
        "durable_lifecycle_present": True,
        "durable_started_routes": 1,
        "durable_terminal_routes": 1,
        "durable_inflight_routes": 0,
        "replay_usage_routes": 1,
        "replay_decision_coverage_rate": 1.0,
        "reconciled": True,
    }
    assert "lifecycle_not_reconciled" not in failed_lab["promotion_gate"]["blocking_reason_codes"]
    decision = manager.route_policy_ledger.list_decisions(session_id="failed-route-session")[0]
    assert decision["status"] == "failed"
    assert decision["success"] is False
    assert decision["error_category"] == "runtime_error"
    assert decision["error_message"] == "simulated route backend failure"
    assert set(decision["actual_economics"]) == {"elapsed_ms"}
    assert 0.0 <= decision["actual_economics"]["elapsed_ms"] < 5_000.0
    failure_outcome_names = [event["outcome_name"] for event in decision["outcome_events"]]
    assert failure_outcome_names.count("route_success") == 1
    assert failure_outcome_names.count("cost") == 1
    assert failure_outcome_names.count("latency") == 1
    assert "user_quality_rating" not in failure_outcome_names
    failure_events = {event["outcome_name"]: event for event in decision["outcome_events"]}
    assert failure_events["route_success"]["value"] is False
    assert failure_events["route_success"]["event_source"] == "route_completion"
    assert failure_events["cost"]["observation_status"] == "not_observed"
    assert failure_events["cost"]["value"] is None
    assert failure_events["latency"]["value"] == pytest.approx(
        decision["actual_economics"]["elapsed_ms"]
    )
    assert manager.memory_store.load_session("failed-route-session").get("route_usage") == []
    with pytest.raises(ValueError, match="successfully completed route"):
        manager.record_route_feedback(
            session_id="failed-route-session",
            feedback={"route_id": decision["route_id"], "rating": "down", "feedback_intent": "bad_quality"},
        )


def test_auto_agent_budget_fast_caps_complex_prompt_to_collective(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response="auto routed through fast collective",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)

    payload = manager.handle_prompt(
        session_id="auto-fast-session",
        prompt=(
            "Research the latest evidence, design and implement a multi-step runtime integration, "
            "debug regressions, verify with tests, and explain benchmark tradeoffs."
        ),
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "fast",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert captured["settings"]["agent_mode"] == "collective"
    assert policy["budget_profile"] == "fast"
    assert policy["selected_agent_mode"] == "collective"
    assert "loop" not in policy["allowed_agent_modes"]
    assert policy["score"] < policy["score_before_budget"]
    assert "fast budget profile" in payload["route_reason"]


def test_auto_agent_budget_max_promotes_moderate_prompt_to_collective_loop(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response="auto routed through max collective loop",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)

    payload = manager.handle_prompt(
        session_id="auto-max-session",
        prompt="Research latest runtime integration and verify tests.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "max",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert captured["collective_mode"] is True
    assert captured["settings"]["agent_mode"] == "collective_loop"
    assert policy["budget_profile"] == "max"
    assert policy["selected_agent_mode"] == "collective_loop"
    assert policy["score"] > policy["score_before_budget"]
    assert "max budget profile" in payload["route_reason"]


def test_auto_agent_mode_keeps_simple_prompt_single_pass(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="auto-simple-session",
        prompt="Say hello.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert payload["agent_trace"]["agent_mode"] == "off"
    assert payload["agent_trace"]["requested_agent_mode"] == "auto"
    assert payload["agent_trace"]["resolved_agent_mode"] == "off"
    assert policy["selected_agent_mode"] == "off"
    assert policy["reason"] == "low_complexity_single_pass"
    assert policy["route_confidence"]["level"] == "high"
    assert policy["uncertainty_adjustment"] is None
    economics = payload["agent_trace"]["route_economics"]
    assert economics["estimate"]["selected_agent_mode"] == "off"
    assert economics["estimate"]["estimated_model_calls"] == 1
    assert economics["actual"]["model_calls"] == 1
    assert economics["actual"]["elapsed_ms"] >= 0
    assert payload["timing"]["route_elapsed_ms"] == economics["actual"]["elapsed_ms"]
    assert policy["route_economics_actual"]["model_calls"] == 1
    assert payload["response"] == "reply via omni_collective_v8"


def test_auto_agent_uncertainty_margin_escalates_near_threshold(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 1, "cycles": 1, "reasons": ["borderline"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)

    payload = manager.handle_prompt(
        session_id="uncertainty-margin-session",
        prompt="Audit this answer for correctness risk.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["uncertainty_adjustment"]
    assert captured["settings"]["agent_mode"] == "collective"
    assert policy["selected_agent_mode"] == "collective"
    assert policy["reason"] == "borderline_score_with_uncertainty_signals"
    assert adjustment["direction"] == "upgrade"
    assert adjustment["from"] == "off"
    assert adjustment["to"] == "collective"
    assert adjustment["score_to_next_mode"] == 1
    assert {"audit", "correctness", "risk"}.issubset(set(adjustment["uncertainty_signals"]))
    assert policy["route_confidence"]["adjusted"] is True
    assert policy["route_confidence"]["score_to_next_mode"] == 1
    assert policy["route_confidence"]["selected_agent_mode"] == "collective"
    assert "borderline_score_with_uncertainty_signals" in payload["route_reason"]


def test_auto_agent_uncertainty_margin_respects_fast_budget(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 2, "cycles": 2, "reasons": ["borderline"]},
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="uncertainty-fast-budget-session",
        prompt="Audit this answer for correctness risk.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "fast",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert payload["agent_trace"]["resolved_agent_mode"] == "off"
    assert policy["budget_profile"] == "fast"
    assert policy["selected_agent_mode"] == "off"
    assert policy["uncertainty_adjustment"] is None
    assert policy["route_confidence"]["adjusted"] is False
    assert policy["route_confidence"]["score_to_next_mode"] == 1
    assert policy["score"] < policy["score_before_budget"]


def test_auto_agent_session_budget_paces_route_to_remaining_budget(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="session-budget-pacing",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)

    payload = manager.handle_prompt(
        session_id="session-budget-pacing",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 8,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert captured["settings"]["agent_mode"] == "collective"
    assert policy["selected_agent_mode"] == "collective"
    assert payload["agent_trace"]["resolved_agent_mode"] == "collective"
    assert adjustment["direction"] == "downgrade"
    assert adjustment["from"] == "collective_loop"
    assert adjustment["to"] == "collective"
    assert adjustment["remaining_cost_units"] == 3.0
    assert policy["session_budget"]["limit_cost_units"] == 8.0
    assert policy["session_budget"]["used_cost_units"] == 5.0
    assert policy["session_budget"]["estimated_cost_units"] <= policy["session_budget"]["remaining_cost_units"]
    assert policy["route_confidence"]["session_budget_adjusted"] is True
    assert policy["route_economics_estimate"]["selected_agent_mode"] == "collective"
    assert "Session budget pacing downgraded collective_loop to collective" in payload["route_reason"]

    health = manager.route_health_snapshot("session-budget-pacing")
    assert health["route_usage"]["total_routes"] == 2
    assert health["route_usage"]["economics"]["total_cost_units"] == 8.0


def test_auto_agent_session_budget_target_routes_paces_early_route(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)

    payload = manager.handle_prompt(
        session_id="session-budget-target-routes",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 12,
            "auto_session_budget_target_routes": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert captured["settings"]["agent_mode"] == "collective"
    assert policy["selected_agent_mode"] == "collective"
    assert adjustment["reason"] == "session_route_budget_target_pacing"
    assert adjustment["from"] == "collective_loop"
    assert adjustment["to"] == "collective"
    assert adjustment["target_route_count"] == 4
    assert adjustment["target_remaining_routes"] == 4
    assert adjustment["pacing_cap_cost_units"] == 3.0
    assert policy["session_budget"]["target_route_count"] == 4
    assert policy["session_budget"]["pacing_cap_cost_units"] == 3.0
    assert policy["session_budget"]["effective_cap_cost_units"] == 3.0
    assert policy["session_budget"]["would_exceed_pacing_cap"] is False
    assert policy["route_economics_estimate"]["estimated_cost_units"] <= 3.0
    assert "Session budget target pacing downgraded collective_loop to collective" in payload["route_reason"]


def test_auto_agent_session_budget_target_routes_preserves_remaining_budget_precedence(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="session-budget-target-hard-remaining",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="session-budget-target-hard-remaining",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 5.5,
            "auto_session_budget_target_routes": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert policy["selected_agent_mode"] == "off"
    assert adjustment["reason"] == "session_route_budget_would_exceed_remaining"
    assert adjustment["to"] == "off"
    assert adjustment["remaining_cost_units"] == 0.5
    assert policy["session_budget"]["would_exceed_remaining"] is True
    assert policy["session_budget"]["would_exceed_pacing_cap"] is True
    assert "Session budget pacing downgraded collective_loop to off" in payload["route_reason"]


def test_auto_agent_session_budget_target_routes_reports_single_pass_pacing_floor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="session-budget-target-single-pass-floor",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 1.5,
            "auto_session_budget_target_routes": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert policy["selected_agent_mode"] == "off"
    assert adjustment["reason"] == "session_route_budget_target_pacing"
    assert adjustment["to"] == "off"
    assert adjustment["pacing_cap_cost_units"] == 0.375
    assert policy["session_budget"]["would_exceed_remaining"] is False
    assert policy["session_budget"]["would_exceed_pacing_cap"] is True
    assert policy["route_economics_estimate"]["estimated_cost_units"] == 1.0
    assert "Session budget target pacing downgraded collective_loop to off" in payload["route_reason"]


def test_route_plan_preview_does_not_run_inference_or_write_memory(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    def fail(*args, **kwargs):
        raise AssertionError("preview must not run inference or write memory")

    monkeypatch.setattr(manager, "ensure_backend", fail)
    monkeypatch.setattr(manager, "_run_text_model", fail)
    monkeypatch.setattr(manager, "_run_agent_text", fail)
    monkeypatch.setattr(manager, "_run_loop_agent_text", fail)
    monkeypatch.setattr(manager.memory_store, "add_route_usage", fail)
    monkeypatch.setattr(manager.memory_store, "update", fail)

    plan = manager.preview_route_plan(
        session_id="route-plan-dry-run",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert plan["dry_run"] is True
    assert plan["execution_plan"]["will_run_inference"] is False
    assert plan["execution_plan"]["will_write_memory"] is False
    assert plan["auto_agent_policy"]["selected_agent_mode"] == plan["selected_agent_mode"]
    assert manager.route_health_snapshot("route-plan-dry-run")["route_usage"]["total_routes"] == 0
    assert manager.route_policy_ledger.report(session_id="route-plan-dry-run")["counts"]["started"] == 0


def test_route_study_preview_uses_final_budget_support_without_assigning_or_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    def fail(*args, **kwargs):
        raise AssertionError("study rehearsal must not infer, assign, or write")

    monkeypatch.setattr(manager, "ensure_backend", fail)
    monkeypatch.setattr(manager, "_run_text_model", fail)
    monkeypatch.setattr(manager, "_run_agent_text", fail)
    monkeypatch.setattr(manager, "_run_loop_agent_text", fail)
    monkeypatch.setattr(manager.memory_store, "add_route_usage", fail)
    monkeypatch.setattr(manager.memory_store, "update", fail)

    prompt = "Research, implement, and verify a private production integration."
    session_id = "route-study-private-session"
    result = manager.preview_route_study(
        session_id=session_id,
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 3,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
        exploration_rate=0.10,
        planned_routes=2000,
        assumed_feedback_rate=0.30,
        target_observed_labels=20,
    )

    assert result["dry_run"] is True
    assert result["baseline_agent_mode"] == "collective"
    assert result["execution_plan"] == {
        "will_run_inference": False,
        "will_write_memory": False,
        "will_write_ledger": False,
        "will_assign_route": False,
        "will_randomize": False,
        "activation_available": False,
    }
    study = result["route_study"]
    assert study["study"]["study_id"] == "auto-route-adjacent-explorer-v1"
    assert study["charter"]["source_contract"] == result["deterministic_support"]
    assert study["charter"]["enrollment"]["adjacent_feasible_actions"] == ["off"]
    assert study["charter"]["probability_design"]["action_probabilities"] == {
        "off": 0.1,
        "collective": 0.9,
    }
    assert study["charter"]["probability_design"]["assignment_performed"] is False
    label_forecast = study["charter"]["traffic_scenario"]["observed_label_scenario"]
    assert label_forecast["target_scope"] == (
        "at_least_target_observed_labels_on_every_alternate_action"
    )
    assert label_forecast["exact_simultaneous_target"]["method"] == (
        "exact_binomial_tail_inversion_single_alternate"
    )
    assert study["charter"]["causal_boundaries"]["execution_enabled"] is False
    assert study["charter"]["causal_boundaries"]["automatic_promotion_allowed"] is False
    assert study["charter"]["causal_boundaries"]["activation_blockers"]
    protocol = result["route_protocol_preflight"]
    assert result["route_protocol_preflight_reason"] == "draft_for_independent_review"
    assert protocol["protocol"]["label"] == "Stateful Route Experiment Preflight v1"
    assert protocol["protocol"]["activation_available"] is False
    assert protocol["charter"]["source_studies"]["support_strata"][0][
        "study_design_hash"
    ] == study["design_hash"]
    assert protocol["charter"]["stateful_design"]["selected_design_mode"] == (
        "sticky_session_cluster"
    )
    assert protocol["charter"]["stateful_design"]["selected_design_status"] == (
        "declaration_incomplete"
    )
    assert protocol["charter"]["randomness"]["assignment_performed"] is False
    assert protocol["charter"]["causal_boundaries"]["activation_blockers"]
    encoded = json.dumps(study, sort_keys=True)
    assert prompt not in encoded
    assert session_id not in encoded
    encoded_protocol = json.dumps(protocol, sort_keys=True)
    assert prompt not in encoded_protocol
    assert session_id not in encoded_protocol
    assert manager.route_health_snapshot(session_id)["route_usage"]["total_routes"] == 0
    assert manager.route_policy_ledger.report(session_id=session_id)["counts"]["started"] == 0


def test_route_study_preview_rejects_manual_agent_mode(tmp_path: Path) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    with pytest.raises(ValueError, match="requires Auto Router"):
        manager.preview_route_study(
            session_id="manual-study",
            prompt="Preview this route.",
            model_key="omni_collective_v8",
            action_mode="text",
            settings={"agent_mode": "off"},
        )


def test_route_study_protocol_preflight_binds_stateful_design_declarations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    result = manager.preview_route_study(
        session_id="stateful-preflight-session",
        prompt="Research a difficult private integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 3,
            "memory_enabled": False,
        },
        target_policy_profile="quality_first",
        protocol_design_mode="clustered_switchback",
        carryover_scope="within_session",
        interference_scope="none_declared",
        temporal_variation="stable_declared",
        planned_clusters=480,
        max_routes_per_cluster=12,
        analysis_every_clusters=60,
        block_length_routes=20,
        washout_routes=4,
    )

    protocol = result["route_protocol_preflight"]
    assert protocol["charter"]["target_policy_class"]["profile_name"] == "quality_first"
    assert protocol["charter"]["population"]["planned_clusters"] == 480
    assert protocol["charter"]["stopping_and_resources"]["planned_route_ceiling"] == 5760
    screen = protocol["charter"]["stateful_design"]
    assert screen["selected_design_mode"] == "clustered_switchback"
    assert screen["selected_design_status"] == "assumptions_declared_unvalidated"
    assert screen["assignment_unit"] == "session_hash_x_time_block"
    assert screen["block_length_routes"] == 20
    assert screen["washout_routes"] == 4
    assert protocol["charter"]["causal_boundaries"]["activation_available"] is False


def test_multi_stratum_review_bundle_is_pure_source_bound_and_fully_reconstructed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from source.route_policy_protocol_cli import _example_bundle_input

    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    before_ledger = manager.route_policy_ledger.report()

    def fail(*_args, **_kwargs):
        raise AssertionError("review-bundle assembly must not enter route or inference paths")

    monkeypatch.setattr(manager, "handle_prompt", fail)
    monkeypatch.setattr(manager, "preview_route_plan", fail)
    result = manager.build_route_protocol_review_bundle(_example_bundle_input())

    bundle = result["route_protocol_review_bundle"]
    verification = result["verification"]
    assert bundle["schema_version"] == "route-study-review-bundle-v1"
    assert verification["verification_level"] == "full_source_bound_reconstruction"
    assert verification["support_stratum_count"] == 2
    assert verification["source_plan_reconstruction_performed"] is True
    assert result["execution_plan"] == {
        "will_run_inference": False,
        "will_write_memory": False,
        "will_write_ledger": False,
        "will_assign_route": False,
        "will_randomize": False,
        "activation_available": False,
    }
    assert manager.route_policy_ledger.report() == before_ledger
    assert manager.audit_route_protocol_review_bundle(bundle)["verification"]["ok"] is True

    with pytest.raises(ValueError, match="non-prompt-free fields: prompt"):
        manager.build_route_protocol_review_bundle(
            {**_example_bundle_input(), "prompt": "must never enter the bundle"}
        )


def test_route_plan_preview_and_execution_share_budget_filtered_support_hashes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="route-support-preview-execution",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )

    settings = {
        "agent_mode": "auto",
        "auto_session_budget_units": 8,
        "memory_enabled": False,
        "web_search_enabled": False,
        "cmd_open_enabled": False,
    }
    prompt = "Research, implement, and verify a complex production integration."
    preview = manager.preview_route_plan(
        session_id="route-support-preview-execution",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings=settings,
    )

    def fake_run_agent_text(**kwargs):
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response="budget-filtered collective response",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    executed = manager.handle_prompt(
        session_id="route-support-preview-execution",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings=settings,
    )

    preview_policy = preview["auto_agent_policy"]
    executed_policy = executed["agent_trace"]["auto_agent_policy"]
    preview_support = preview_policy["logging_support"]
    executed_support = executed_policy["logging_support"]
    preview_exclusions = {
        row["action"]: row["reasons"] for row in preview_support["exclusions"]
    }

    assert preview["selected_agent_mode"] == "collective"
    assert executed["agent_trace"]["resolved_agent_mode"] == "collective"
    assert [row["action"] for row in preview_support["candidates"]] == ["off", "collective"]
    assert preview_exclusions == {
        "loop": ["session_budget_post_filter"],
        "collective_loop": ["session_budget_post_filter"],
    }
    assert executed_support["candidates"] == preview_support["candidates"]
    assert executed_support["exclusions"] == preview_support["exclusions"]
    assert executed_policy["candidate_set_hash"] == preview_policy["candidate_set_hash"]
    assert executed_policy["distribution_hash"] == preview_policy["distribution_hash"]

    ledger_row = manager.route_policy_ledger.get_decision(executed["route_id"])
    assert ledger_row["candidate_set_hash"] == preview_policy["candidate_set_hash"]
    assert ledger_row["distribution_hash"] == preview_policy["distribution_hash"]


def test_route_plan_preview_is_idempotent_for_session_budget_usage(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="route-plan-budget-idempotent",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )

    settings = {
        "agent_mode": "auto",
        "auto_session_budget_units": 8,
        "auto_session_budget_target_routes": 4,
        "memory_enabled": False,
        "web_search_enabled": False,
        "cmd_open_enabled": False,
    }
    first = manager.preview_route_plan(
        session_id="route-plan-budget-idempotent",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings=settings,
    )
    second = manager.preview_route_plan(
        session_id="route-plan-budget-idempotent",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings=settings,
    )

    usage = manager.memory_store.route_usage_summary("route-plan-budget-idempotent")
    assert usage["total_routes"] == 1
    assert usage["economics"]["total_cost_units"] == 5.0
    assert first["auto_agent_policy"]["session_budget"]["used_cost_units"] == 5.0
    assert second["auto_agent_policy"]["session_budget"]["used_cost_units"] == 5.0
    assert first["route_economics_estimate"]["selected_agent_mode"] == first["selected_agent_mode"]
    assert second["route_economics_estimate"]["selected_agent_mode"] == second["selected_agent_mode"]


def test_route_plan_preview_sanitizes_bad_numeric_settings(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-bad-settings",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "loop_max_steps": "bad",
            "web_search_enabled": True,
            "web_search_budget": -5,
            "web_search_results": "bad",
            "auto_session_budget_units": "bad",
            "auto_session_budget_target_routes": "bad",
            "memory_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    estimate = plan["route_economics_estimate"]
    assert plan["ok"] is True
    assert estimate["estimated_tool_calls"] == 0
    assert estimate["planned_loop_steps"] == 4
    assert estimate["estimated_cost_units"] >= 0.0
    assert plan["auto_agent_policy"]["session_budget_adjustment"] is None
    for row in plan["route_alternatives"]:
        row_estimate = row["estimate"]
        assert row_estimate["estimated_tool_calls"] >= 0
        assert 0 <= row_estimate["planned_loop_steps"] <= runtime_module.LOOP_AGENT_HARD_MAX_STEPS
        assert row_estimate["estimated_cost_units"] >= 0.0


def test_route_plan_preview_reports_allowed_route_alternatives(tmp_path: Path, monkeypatch) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-alternatives",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "balanced",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    alternatives = plan["route_alternatives"]
    modes = [row["selected_agent_mode"] for row in alternatives]
    selected = [row for row in alternatives if row["is_selected"]]
    assert modes == ["off", "collective", "loop", "collective_loop"]
    assert selected == [next(row for row in alternatives if row["selected_agent_mode"] == plan["selected_agent_mode"])]
    assert all(row["estimated_cost_units"] == row["estimate"]["estimated_cost_units"] for row in alternatives)
    assert alternatives[0]["estimated_cost_units"] < alternatives[-1]["estimated_cost_units"]
    assert all(row["frontier_rank"] >= 1 for row in alternatives)
    assert all(0.0 <= row["estimated_quality_score"] <= 1.0 for row in alternatives)
    assert all(row["estimated_quality_cost_score"] is not None for row in alternatives)
    assert all(row["quality_cost_source"] == "heuristic_cost_adjusted" for row in alternatives)
    assert all(row["quality_evidence_status"] == "heuristic_prior" for row in alternatives)
    assert all(row["budget_fit"] is True for row in alternatives)
    assert all(row["budget_feasible_pareto_frontier"] == row["pareto_frontier"] for row in alternatives)
    assert plan["route_frontier"]["budget_feasible_pareto_modes"] == plan["route_frontier"]["pareto_modes"]
    assert plan["route_frontier"]["recommended_agent_mode"] == plan["selected_agent_mode"]
    assert plan["route_frontier"]["selected_matches_recommendation"] is True
    assert plan["route_frontier"]["budget_fit_count"] == len(alternatives)
    assert plan["route_frontier"]["budget_blockers"]["none"] == len(alternatives)
    assert plan["route_frontier"]["selected_budget_blocker"] is None
    assert plan["route_frontier"]["ranked_modes"][0]["frontier_rank"] == 1


def test_route_plan_preview_alternatives_reflect_budget_adjusted_selection(tmp_path: Path, monkeypatch) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="route-plan-budget-frontier",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-budget-frontier",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 8,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    alternatives = plan["route_alternatives"]
    selected_rows = [row for row in alternatives if row["is_selected"]]
    expensive_row = next(row for row in alternatives if row["selected_agent_mode"] == "collective_loop")
    assert plan["selected_agent_mode"] == "collective"
    assert selected_rows == [next(row for row in alternatives if row["selected_agent_mode"] == "collective")]
    assert selected_rows[0]["estimate"] == plan["route_economics_estimate"]
    assert expensive_row["estimated_cost_units"] > selected_rows[0]["estimated_cost_units"]
    assert selected_rows[0]["budget_fit"] is True
    assert expensive_row["budget_fit"] is False
    assert selected_rows[0]["fits_remaining_budget"] is True
    assert selected_rows[0]["fits_pacing_cap"] is None
    assert selected_rows[0]["budget_blocker"] is None
    assert expensive_row["fits_remaining_budget"] is False
    assert expensive_row["budget_blocker"] == "remaining_budget"
    assert plan["route_frontier"]["budget_cap_cost_units"] == 3.0
    assert plan["route_frontier"]["remaining_cost_units"] == 3.0
    assert plan["route_frontier"]["pacing_cap_cost_units"] is None
    assert plan["route_frontier"]["effective_cap_cost_units"] == 3.0
    assert plan["route_frontier"]["recommended_agent_mode"] == "collective"
    assert plan["route_frontier"]["selected_matches_recommendation"] is True
    ranked_collective_loop = next(
        row for row in plan["route_frontier"]["ranked_modes"] if row["selected_agent_mode"] == "collective_loop"
    )
    assert ranked_collective_loop["budget_fit"] is False
    assert ranked_collective_loop["budget_blocker"] == "remaining_budget"
    assert ranked_collective_loop["pareto_frontier"] is True
    assert ranked_collective_loop["budget_feasible_pareto_frontier"] is False
    assert "collective_loop" in plan["route_frontier"]["pareto_modes"]
    assert "collective_loop" not in plan["route_frontier"]["budget_feasible_pareto_modes"]


def test_route_plan_preview_alternatives_report_single_pass_floor(tmp_path: Path, monkeypatch) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 0, "cycles": 0, "reasons": ["simple_prompt"]},
    )
    manager.memory_store.add_route_usage(
        session_id="route-plan-single-pass-floor",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-single-pass-floor",
        prompt="Say hello.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 5.5,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    selected = next(row for row in plan["route_alternatives"] if row["is_selected"])
    assert plan["selected_agent_mode"] == "off"
    assert plan["auto_agent_policy"]["session_budget_adjustment"]["reason"] == "session_route_budget_exhausted_single_pass_floor"
    assert selected["selected_agent_mode"] == "off"
    assert selected["estimated_cost_units"] == 1.0
    assert selected["budget_fit"] is False
    assert selected["fits_remaining_budget"] is False
    assert selected["fits_pacing_cap"] is None
    assert selected["minimum_route_floor"] is True
    assert selected["budget_blocker"] == "remaining_budget"
    assert plan["route_frontier"]["budget_fit_count"] == 0
    assert plan["route_frontier"]["recommended_agent_mode"] == "off"
    assert plan["route_frontier"]["recommended_reason"] == "no_budget_feasible_route"
    assert plan["route_frontier"]["recommended_budget_blocker"] == "remaining_budget"
    assert plan["route_frontier"]["selected_budget_blocker"] == "remaining_budget"
    assert plan["route_frontier"]["budget_blockers"]["remaining_budget"] == len(plan["route_alternatives"])
    assert plan["route_frontier"]["minimum_route_floor"] is True


def test_route_plan_frontier_distinguishes_remaining_fit_from_pacing_fit(tmp_path: Path, monkeypatch) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-pacing-floor",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 1.5,
            "auto_session_budget_target_routes": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    selected = next(row for row in plan["route_alternatives"] if row["is_selected"])
    frontier = plan["route_frontier"]
    assert plan["selected_agent_mode"] == "off"
    assert plan["auto_agent_policy"]["session_budget_adjustment"]["reason"] == "session_route_budget_target_pacing"
    assert selected["selected_agent_mode"] == "off"
    assert selected["estimated_cost_units"] == 1.0
    assert selected["fits_remaining_budget"] is True
    assert selected["fits_pacing_cap"] is False
    assert selected["budget_fit"] is False
    assert selected["budget_blocker"] == "pacing_cap"
    assert frontier["remaining_cost_units"] == 1.5
    assert frontier["pacing_cap_cost_units"] == 0.375
    assert frontier["effective_cap_cost_units"] == 0.375
    assert frontier["pacing_cap_applied"] is True
    assert frontier["selected_fits_remaining_budget"] is True
    assert frontier["selected_fits_pacing_cap"] is False
    assert frontier["selected_budget_blocker"] == "pacing_cap"
    assert frontier["recommended_agent_mode"] == "off"
    assert frontier["recommended_budget_blocker"] == "pacing_cap"
    assert frontier["minimum_route_floor"] is False
    assert frontier["recommended_reason"] == "no_pacing_feasible_route"
    assert frontier["budget_blockers"]["pacing_cap"] == 1
    assert frontier["budget_blockers"]["remaining_budget"] == len(plan["route_alternatives"]) - 1
    assert frontier["budget_feasible_pareto_modes"] == []


def test_route_plan_frontier_prefers_complete_adaptive_quality_cost_evidence(tmp_path: Path) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    alternatives = [
        {"selected_agent_mode": "collective", "estimated_cost_units": 2.0, "is_selected": False},
        {"selected_agent_mode": "loop", "estimated_cost_units": 8.0, "is_selected": True},
    ]

    frontier = manager._annotate_route_frontier(
        alternatives=alternatives,
        selected="loop",
        action_mode="text",
        auto_agent_policy={
            "score": 4,
            "budget_profile": "balanced",
            "feedback_summary": {
                "mode_scores": {
                    "collective": {
                        "adaptive": {
                            "sample_count": 3,
                            "weighted_count": 2.5,
                            "weighted_net": 2.5,
                            "quality_score": 0.86,
                            "quality_cost_score": 0.78,
                        },
                        "economics": {"sample_count": 3, "avg_cost_units": 2.0},
                    },
                    "loop": {
                        "adaptive": {
                            "sample_count": 3,
                            "weighted_count": 2.5,
                            "weighted_net": 2.5,
                            "quality_score": 0.94,
                            "quality_cost_score": 0.46,
                        },
                        "economics": {"sample_count": 3, "avg_cost_units": 8.0},
                    },
                }
            },
        },
    )

    collective = next(row for row in alternatives if row["selected_agent_mode"] == "collective")
    ranked_collective = next(row for row in frontier["ranked_modes"] if row["selected_agent_mode"] == "collective")
    assert collective["quality_source"] == "adaptive_feedback"
    assert collective["quality_cost_source"] == "adaptive_feedback"
    assert collective["quality_evidence_status"] == "adaptive_complete"
    assert collective["estimated_quality_cost_score"] == 0.78
    assert frontier["recommended_agent_mode"] == "collective"
    assert frontier["recommended_reason"] == "adaptive_quality_cost_frontier_recommended"
    assert frontier["recommended_estimated_quality_cost_score"] == 0.78
    assert frontier["selected_estimated_quality_cost_score"] == 0.46
    assert ranked_collective["frontier_rank"] == 1
    assert ranked_collective["estimated_quality_cost_score"] == 0.78
    assert ranked_collective["quality_cost_source"] == "adaptive_feedback"
    assert ranked_collective["quality_evidence_status"] == "adaptive_complete"


def test_route_plan_frontier_reports_incomplete_adaptive_cost_evidence_without_promoting_it(
    tmp_path: Path,
) -> None:
    manager = UnifiedModelManager(
        records=_auto_route_records(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    alternatives = [
        {"selected_agent_mode": "collective", "estimated_cost_units": 2.0, "is_selected": False},
        {"selected_agent_mode": "loop", "estimated_cost_units": 8.0, "is_selected": True},
    ]

    frontier = manager._annotate_route_frontier(
        alternatives=alternatives,
        selected="loop",
        action_mode="text",
        auto_agent_policy={
            "score": 4,
            "budget_profile": "max",
            "feedback_summary": {
                "mode_scores": {
                    "collective": {
                        "adaptive": {
                            "sample_count": 3,
                            "weighted_count": 2.5,
                            "weighted_net": 2.5,
                            "quality_score": 0.99,
                            "quality_cost_score": 0.95,
                        },
                        "economics": {"sample_count": 0},
                    }
                }
            },
        },
    )

    collective = next(row for row in alternatives if row["selected_agent_mode"] == "collective")
    assert collective["quality_source"] == "heuristic_policy"
    assert collective["quality_cost_source"] == "heuristic_cost_adjusted"
    assert collective["quality_evidence_status"] == "adaptive_incomplete_cost_evidence"
    assert collective["quality_evidence"] is None
    assert collective["estimated_quality_score"] == 0.7
    assert frontier["recommended_agent_mode"] == "loop"
    assert frontier["ranked_modes"][0]["selected_agent_mode"] == "loop"


@pytest.mark.parametrize(
    ("profile", "expected_modes"),
    [
        ("fast", ["off", "collective"]),
        ("balanced", ["off", "collective", "loop", "collective_loop"]),
        ("deep", ["off", "collective", "loop", "collective_loop"]),
        ("max", ["off", "collective", "loop", "collective_loop"]),
    ],
)
def test_route_plan_preview_alternatives_follow_auto_budget_profile(
    tmp_path: Path,
    monkeypatch,
    profile: str,
    expected_modes: list[str],
) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id=f"route-plan-profile-{profile}",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": profile,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert [row["selected_agent_mode"] for row in plan["route_alternatives"]] == expected_modes


@pytest.mark.parametrize(
    ("toggle_settings", "expected_modes"),
    [
        ({"auto_agent_collective": False}, ["off", "loop"]),
        ({"auto_agent_loop": False}, ["off", "collective"]),
    ],
)
def test_route_plan_preview_alternatives_follow_auto_agent_toggles(
    tmp_path: Path,
    monkeypatch,
    toggle_settings: dict[str, bool],
    expected_modes: list[str],
) -> None:
    records = _auto_route_records()
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    settings = {
        "agent_mode": "auto",
        "memory_enabled": False,
        "web_search_enabled": False,
        "cmd_open_enabled": False,
    }
    settings.update(toggle_settings)

    plan = manager.preview_route_plan(
        session_id="route-plan-toggle-frontier",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings=settings,
    )

    assert [row["selected_agent_mode"] for row in plan["route_alternatives"]] == expected_modes
    assert plan["selected_agent_mode"] in expected_modes


def test_route_plan_preview_alternatives_skip_unavailable_collective_modes(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-no-collective-frontier",
        prompt="Research, implement, and verify a complex integration.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert [row["selected_agent_mode"] for row in plan["route_alternatives"]] == ["off", "loop"]


def test_route_plan_preview_image_action_excludes_loop_alternatives(tmp_path: Path) -> None:
    records = (
        _record("dcgan_v2_in_progress", "dcgan_image", ("image",), None),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-image-loop-fallback",
        prompt="Generate a clean test image.",
        model_key="dcgan_v2_in_progress",
        action_mode="image",
        settings={
            "agent_mode": "loop",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    modes = [row["selected_agent_mode"] for row in plan["route_alternatives"]]
    assert plan["selected_agent_mode"] == "off"
    assert "loop" not in modes
    assert "collective_loop" not in modes
    assert plan["execution_plan"]["loop_enabled"] is False
    assert plan["route_economics_estimate"]["selected_agent_mode"] == "off"


def test_route_plan_preview_image_support_records_post_filter_exclusions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = UnifiedModelManager(
        records=(_record("dcgan_v2_in_progress", "dcgan_image", ("image",), None),),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-image-support",
        prompt="Generate and refine a complex production image.",
        model_key="dcgan_v2_in_progress",
        action_mode="image",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = plan["auto_agent_policy"]
    support = policy["logging_support"]
    exclusions = {row["action"]: row["reasons"] for row in support["exclusions"]}
    assert plan["selected_agent_mode"] == "off"
    assert [row["action"] for row in support["candidates"]] == ["off"]
    assert exclusions == {
        "collective": ["capability_or_policy_filter"],
        "loop": ["action_mode_unsupported"],
        "collective_loop": ["action_mode_unsupported"],
    }
    assert policy["eligible_actions"] == ["off"]
    assert policy["post_filter_action_probabilities"] == {"off": 1.0}
    assert support["probability_stage"] == "post_filter"
    assert len(policy["candidate_set_hash"]) == 64
    assert len(policy["distribution_hash"]) == 64


def test_route_plan_preview_manual_mode_skips_auto_policy_and_writes(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    plan = manager.preview_route_plan(
        session_id="route-plan-manual-loop",
        prompt="Iterate on this implementation.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "loop",
            "loop_max_steps": "bad",
            "auto_session_budget_units": 1.5,
            "auto_session_budget_target_routes": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert plan["auto_agent_policy"] is None
    assert plan["selected_agent_mode"] == "loop"
    assert plan["execution_plan"]["loop_enabled"] is True
    assert plan["execution_plan"]["will_write_memory"] is False
    assert plan["route_economics_estimate"]["planned_loop_steps"] == 4
    assert [row["selected_agent_mode"] for row in plan["route_alternatives"]] == ["loop"]
    assert plan["route_alternatives"][0]["is_selected"] is True
    assert manager.route_health_snapshot("route-plan-manual-loop")["route_usage"]["total_routes"] == 0


def test_auto_agent_session_budget_overrides_uncertainty_upgrade(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 1, "cycles": 1, "reasons": ["borderline"]},
    )
    manager.memory_store.add_route_usage(
        session_id="uncertainty-budget-session",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="uncertainty-budget-session",
        prompt="Audit this answer for correctness risk.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 5.5,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["uncertainty_adjustment"]["from"] == "off"
    assert policy["uncertainty_adjustment"]["to"] == "collective"
    assert policy["session_budget_adjustment"]["from"] == "collective"
    assert policy["session_budget_adjustment"]["to"] == "off"
    assert policy["selected_agent_mode"] == "off"
    assert policy["route_confidence"]["session_budget_adjusted"] is True
    assert payload["agent_trace"]["resolved_agent_mode"] == "off"


def test_auto_agent_session_budget_can_fall_back_to_single_pass_when_exhausted(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 5, "cycles": 5, "reasons": ["workflow_depth"]},
    )
    manager.memory_store.add_route_usage(
        session_id="session-budget-exhausted",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="session-budget-exhausted",
        prompt="Complex implementation task.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 5.5,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert payload["agent_trace"]["resolved_agent_mode"] == "off"
    assert policy["selected_agent_mode"] == "off"
    assert adjustment["from"] == "collective_loop"
    assert adjustment["to"] == "off"
    assert adjustment["remaining_cost_units"] == 0.5
    assert policy["session_budget"]["would_exceed_remaining"] is True
    assert policy["route_economics_estimate"]["selected_agent_mode"] == "off"
    assert payload["response"] == "reply via omni_collective_v8"


def test_auto_agent_session_budget_reports_single_pass_floor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 0, "cycles": 0, "reasons": ["simple_prompt"]},
    )
    manager.memory_store.add_route_usage(
        session_id="session-budget-floor",
        route_id="prior-route",
        prompt="Prior expensive work.",
        selected_agent_mode="collective_loop",
        route_economics={"actual": {"elapsed_ms": 1000.0, "model_calls": 5, "cost_units": 5.0}},
        auto_agent_policy={"selected_agent_mode": "collective_loop", "score": 5},
        route_reason="prior",
        model_key="omni_collective_v8",
    )
    monkeypatch.setattr(manager, "_build_backend", lambda record: _FakeBackend(record))

    payload = manager.handle_prompt(
        session_id="session-budget-floor",
        prompt="Say hello.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_session_budget_units": 5.5,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["session_budget_adjustment"]
    assert policy["selected_agent_mode"] == "off"
    assert adjustment["direction"] == "floor"
    assert adjustment["from"] == "off"
    assert adjustment["to"] == "off"
    assert adjustment["reason"] == "session_route_budget_exhausted_single_pass_floor"
    assert policy["session_budget"]["would_exceed_remaining"] is True
    assert "Session budget is exhausted" in payload["route_reason"]


def test_auto_agent_route_feedback_is_session_scoped_and_clearable(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    calls = []

    def fake_run_loop_agent_text(**kwargs):
        calls.append(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = (
        "Research the latest evidence, design and implement a multi-step runtime integration, "
        "debug regressions, verify with tests, and explain benchmark tradeoffs."
    )

    for idx in range(2):
        feedback = manager.record_route_feedback(
            session_id="feedback-session",
            feedback={
                "route_id": f"route-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective_loop",
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": "collective_loop", "score": 5},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 100 + idx,
                        "model_calls": 8,
                        "tool_calls": 0,
                        "cost_units": 8.0,
                    }
                },
            },
        )
    assert feedback["summary"]["mode_scores"]["collective_loop"]["net"] == -2
    health = manager.route_health_snapshot("feedback-session")
    assert health["economics"]["sample_count"] == 2
    assert health["economics"]["avg_cost_units"] == 8.0

    adjusted = manager.handle_prompt(
        session_id="feedback-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )
    adjusted_policy = adjusted["agent_trace"]["auto_agent_policy"]
    assert adjusted_policy["selected_agent_mode"] == "loop"
    assert adjusted_policy["feedback_adjustment"]["direction"] == "downgrade"
    assert calls[-1]["collective_mode"] is False

    clean = manager.handle_prompt(
        session_id="clean-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )
    assert clean["agent_trace"]["auto_agent_policy"]["selected_agent_mode"] == "collective_loop"
    assert calls[-1]["collective_mode"] is True

    manager.clear("feedback-session")
    reset = manager.handle_prompt(
        session_id="feedback-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )
    assert reset["agent_trace"]["auto_agent_policy"]["selected_agent_mode"] == "collective_loop"
    assert calls[-1]["collective_mode"] is True


def test_auto_agent_explicit_needs_deeper_feedback_moves_one_route_up(tmp_path: Path) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    prompt = "Implement and verify the runtime integration."
    manager.record_route_feedback(
        session_id="needs-deeper-session",
        feedback={
            "route_id": "collective-route",
            "prompt": prompt,
            "selected_agent_mode": "collective",
            "rating": "down",
            "feedback_intent": "needs_deeper",
        },
    )

    summary = manager.memory_store.route_feedback_summary("needs-deeper-session", prompt)
    selected, adjustment = manager._apply_auto_route_feedback(
        selected="collective",
        score=2,
        feedback_summary=summary,
        allowed_modes=("off", "collective", "loop", "collective_loop"),
    )

    assert selected == "loop"
    assert adjustment["direction"] == "upgrade"
    assert adjustment["reason"] == "explicit_feedback_requested_deeper_route"
    assert summary["mode_scores"]["collective"]["quality_negative"] == 0


def test_auto_agent_explicit_cost_feedback_moves_one_route_down_without_quality_regression(tmp_path: Path) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    prompt = "Implement and verify the runtime integration."
    manager.record_route_feedback(
        session_id="lower-cost-session",
        feedback={
            "route_id": "loop-route",
            "prompt": prompt,
            "selected_agent_mode": "loop",
            "rating": "down",
            "feedback_intent": "too_costly",
            "route_economics": {"actual": {"elapsed_ms": 7000, "model_calls": 8, "cost_units": 8.0}},
        },
    )

    summary = manager.memory_store.route_feedback_summary("lower-cost-session", prompt)
    selected, adjustment = manager._apply_auto_route_feedback(
        selected="loop",
        score=4,
        feedback_summary=summary,
        allowed_modes=("off", "collective", "loop", "collective_loop"),
    )

    assert selected == "collective"
    assert adjustment["direction"] == "downgrade"
    assert adjustment["reason"] == "explicit_feedback_requested_lower_cost_route"
    assert summary["mode_scores"]["loop"]["adaptive"]["quality_score"] is None


def test_auto_agent_adaptive_feedback_downgrades_recent_regression_over_stale_positives(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration."
    for idx in range(6):
        manager.record_route_feedback(
            session_id="adaptive-feedback-session",
            feedback={
                "route_id": f"old-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 100.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-feedback-session",
            feedback={
                "route_id": f"new-bad-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 120.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-feedback-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["feedback_adjustment"]
    assert policy["selected_agent_mode"] == "collective"
    assert captured["settings"]["agent_mode"] == "collective"
    assert adjustment["direction"] == "downgrade"
    assert adjustment["reason"] == "recent_weighted_feedback_regression"
    assert adjustment["from"] == "loop"
    assert adjustment["to"] == "collective"
    assert adjustment["weighted_net"] < 0
    assert policy["feedback_summary"]["mode_scores"]["loop"]["net"] == 4
    assert policy["feedback_summary"]["mode_scores"]["loop"]["adaptive"]["regression_signal"] is True
    assert "recent_weighted_feedback_regression" in payload["route_reason"]


def test_auto_agent_adaptive_feedback_ignores_unrelated_recent_fallback(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    unrelated_prompt = "Optimize the marketing dashboard."
    target_prompt = "Implement a runtime integration."
    for idx in range(6):
        manager.record_route_feedback(
            session_id="adaptive-unrelated-session",
            feedback={
                "route_id": f"old-good-{idx}",
                "prompt": unrelated_prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 100.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-unrelated-session",
            feedback={
                "route_id": f"new-bad-{idx}",
                "prompt": unrelated_prompt,
                "selected_agent_mode": "loop",
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 120.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-unrelated-session",
        prompt=target_prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["used_recent_fallback"] is True
    assert policy["feedback_summary"]["mode_scores"]["loop"]["adaptive"]["regression_signal"] is True
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_max_budget_still_honors_adaptive_quality_regression(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 2, "cycles": 2, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration."
    for idx in range(6):
        manager.record_route_feedback(
            session_id="adaptive-max-session",
            feedback={
                "route_id": f"old-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 100.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-max-session",
            feedback={
                "route_id": f"new-bad-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 120.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-max-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "max",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["budget_profile"] == "max"
    assert policy["selected_agent_mode"] == "collective"
    assert policy["feedback_adjustment"]["reason"] == "recent_weighted_feedback_regression"
    assert captured["settings"]["agent_mode"] == "collective"


def test_auto_agent_adaptive_quality_cost_prefers_cheaper_neighbor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration and verify the tests."

    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-cheaper-neighbor-session",
            feedback={
                "route_id": f"loop-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 7000.0, "model_calls": 8, "cost_units": 8.0}},
            },
        )
    for idx in range(3):
        manager.record_route_feedback(
            session_id="adaptive-cheaper-neighbor-session",
            feedback={
                "route_id": f"collective-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {"actual": {"elapsed_ms": 900.0, "model_calls": 2, "cost_units": 2.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-cheaper-neighbor-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["feedback_adjustment"]
    assert policy["selected_agent_mode"] == "collective"
    assert captured["settings"]["agent_mode"] == "collective"
    assert adjustment["direction"] == "downgrade"
    assert adjustment["reason"] == "adaptive_quality_cost_preferred_neighbor"
    assert adjustment["quality_cost_delta"] > 0
    assert policy["feedback_summary"]["mode_scores"]["collective"]["adaptive"]["quality_cost_score"] > (
        policy["feedback_summary"]["mode_scores"]["loop"]["adaptive"]["quality_cost_score"]
    )


def test_auto_agent_adaptive_quality_cost_requires_candidate_sample_floor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = "Implement a runtime integration and verify the tests."
    for idx in range(3):
        manager.record_route_feedback(
            session_id="adaptive-sample-floor-session",
            feedback={
                "route_id": f"loop-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 6500.0, "model_calls": 8, "cost_units": 8.0}},
            },
        )
    manager.record_route_feedback(
        session_id="adaptive-sample-floor-session",
        feedback={
            "route_id": "collective-lucky-good",
            "prompt": prompt,
            "selected_agent_mode": "collective",
            "rating": "up",
            "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
            "route_economics": {"actual": {"elapsed_ms": 500.0, "model_calls": 1, "cost_units": 1.0}},
        },
    )

    payload = manager.handle_prompt(
        session_id="adaptive-sample-floor-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["mode_scores"]["collective"]["adaptive"]["weighted_count"] == 1.0
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_adaptive_quality_cost_ignores_missing_cost_evidence(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = "Implement a runtime integration and verify the tests."
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-missing-cost-session",
            feedback={
                "route_id": f"loop-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 6500.0, "model_calls": 8, "cost_units": 8.0}},
            },
        )
    for idx in range(3):
        manager.record_route_feedback(
            session_id="adaptive-missing-cost-session",
            feedback={
                "route_id": f"collective-no-cost-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {"actual": {"elapsed_ms": 800.0, "model_calls": 2}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-missing-cost-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["mode_scores"]["collective"]["economics"].get("avg_cost_units") is None
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_adaptive_quality_cost_can_promote_stronger_neighbor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 3, "cycles": 3, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = "Plan and implement a careful multi-step runtime change."

    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-stronger-neighbor-session",
            feedback={
                "route_id": f"collective-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {"actual": {"elapsed_ms": 8000.0, "model_calls": 8, "cost_units": 8.0}},
            },
        )
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-stronger-neighbor-session",
            feedback={
                "route_id": f"loop-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 3},
                "route_economics": {"actual": {"elapsed_ms": 1800.0, "model_calls": 3, "cost_units": 3.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-stronger-neighbor-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["feedback_adjustment"]
    assert policy["selected_agent_mode"] == "loop"
    assert captured["settings"]["agent_mode"] == "loop"
    assert adjustment["direction"] == "upgrade"
    assert adjustment["reason"] == "adaptive_quality_cost_preferred_neighbor"
    assert adjustment["quality_cost_delta"] > 0


def test_auto_agent_max_budget_ignores_cost_only_adaptive_neighbor(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 2, "cycles": 2, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = "Implement a runtime integration."
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-max-cost-only-session",
            feedback={
                "route_id": f"loop-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {"actual": {"elapsed_ms": 8000.0, "model_calls": 8, "cost_units": 8.0}},
            },
        )
    for idx in range(2):
        manager.record_route_feedback(
            session_id="adaptive-max-cost-only-session",
            feedback={
                "route_id": f"collective-good-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {"actual": {"elapsed_ms": 900.0, "model_calls": 2, "cost_units": 2.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-max-cost-only-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "max",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["budget_profile"] == "max"
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_adaptive_quality_cost_ignores_unrelated_recent_fallback(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    unrelated_prompt = "Summarize customer churn by segment."
    target_prompt = "Implement a runtime integration and verify the tests."
    for idx in range(3):
        manager.record_route_feedback(
            session_id="adaptive-cost-unrelated-session",
            feedback={
                "route_id": f"collective-good-{idx}",
                "prompt": unrelated_prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {"actual": {"elapsed_ms": 900.0, "model_calls": 2, "cost_units": 2.0}},
            },
        )

    payload = manager.handle_prompt(
        session_id="adaptive-cost-unrelated-session",
        prompt=target_prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["used_recent_fallback"] is True
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_economics_pressure_downgrades_without_positive_signal(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration and verify the tests."

    for idx, selected_mode in enumerate(("loop", "collective")):
        manager.record_route_feedback(
            session_id="economic-pressure-session",
            feedback={
                "route_id": f"route-{idx}",
                "prompt": prompt,
                "selected_agent_mode": selected_mode,
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": selected_mode, "score": 4},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 9000 + idx,
                        "model_calls": 7,
                        "tool_calls": 0,
                        "cost_units": 7.0,
                    }
                },
            },
        )

    payload = manager.handle_prompt(
        session_id="economic-pressure-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["feedback_adjustment"]
    assert policy["selected_agent_mode"] == "collective"
    assert captured["settings"]["agent_mode"] == "collective"
    assert adjustment["direction"] == "downgrade"
    assert adjustment["reason"] == "session_route_economics_exceeded_budget_health"
    assert adjustment["economic_pressure"] == ["cost", "latency"]
    assert adjustment["pressure_scope"] == "loop"
    assert adjustment["sample_count"] == 1
    assert policy["feedback_summary"]["economics"]["avg_cost_units"] == 7.0
    assert policy["feedback_summary"]["mode_scores"]["loop"]["economics"]["avg_cost_units"] == 7.0
    assert "session_route_economics_exceeded_budget_health" in payload["route_reason"]


def test_auto_agent_economics_pressure_is_mode_scoped(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 2, "cycles": 2, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration and verify the tests."

    for idx in range(2):
        manager.record_route_feedback(
            session_id="mode-scoped-economic-pressure-session",
            feedback={
                "route_id": f"loop-route-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "loop",
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 12000 + idx,
                        "model_calls": 9,
                        "tool_calls": 0,
                        "cost_units": 9.0,
                    }
                },
            },
        )
    manager.record_route_feedback(
        session_id="mode-scoped-economic-pressure-session",
        feedback={
            "route_id": "collective-route",
            "prompt": prompt,
            "selected_agent_mode": "collective",
            "rating": "down",
            "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
            "route_economics": {
                "actual": {
                    "elapsed_ms": 1200,
                    "model_calls": 2,
                    "tool_calls": 0,
                    "cost_units": 2.0,
                }
            },
        },
    )

    payload = manager.handle_prompt(
        session_id="mode-scoped-economic-pressure-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "collective"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["mode_scores"]["loop"]["economics"]["avg_cost_units"] == 9.0
    assert policy["feedback_summary"]["mode_scores"]["collective"]["economics"]["avg_cost_units"] == 2.0
    assert captured["settings"]["agent_mode"] == "collective"


def test_auto_agent_economics_pressure_uses_selected_mode_not_aggregate(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_panel", "consulted_models": ["a", "b"]},
        )

    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)
    prompt = "Implement a runtime integration and verify the tests."

    manager.record_route_feedback(
        session_id="selected-mode-economic-pressure-session",
        feedback={
            "route_id": "loop-route",
            "prompt": prompt,
            "selected_agent_mode": "loop",
            "rating": "down",
            "auto_agent_policy": {"selected_agent_mode": "loop", "score": 4},
            "route_economics": {
                "actual": {
                    "elapsed_ms": 9000,
                    "model_calls": 7,
                    "tool_calls": 0,
                    "cost_units": 7.0,
                }
            },
        },
    )
    for idx in range(6):
        manager.record_route_feedback(
            session_id="selected-mode-economic-pressure-session",
            feedback={
                "route_id": f"collective-route-{idx}",
                "prompt": prompt,
                "selected_agent_mode": "collective",
                "rating": "up",
                "auto_agent_policy": {"selected_agent_mode": "collective", "score": 2},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 400,
                        "model_calls": 1,
                        "tool_calls": 0,
                        "cost_units": 1.0,
                    }
                },
            },
        )

    payload = manager.handle_prompt(
        session_id="selected-mode-economic-pressure-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    adjustment = policy["feedback_adjustment"]
    assert policy["selected_agent_mode"] == "collective"
    assert adjustment["reason"] == "adaptive_quality_cost_preferred_neighbor"
    assert adjustment["quality_cost_delta"] > 0
    assert policy["feedback_summary"]["economics"]["avg_cost_units"] < 6.0
    assert policy["feedback_summary"]["mode_scores"]["loop"]["economics"]["avg_cost_units"] == 7.0
    assert captured["settings"]["agent_mode"] == "collective"


def test_auto_agent_economics_pressure_ignores_unrelated_recent_feedback(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    unrelated_prompt = "Summarize the quarterly sales dashboard."
    target_prompt = "Implement a runtime integration and verify the tests."

    for idx, selected_mode in enumerate(("loop", "collective")):
        manager.record_route_feedback(
            session_id="unrelated-economic-pressure-session",
            feedback={
                "route_id": f"route-{idx}",
                "prompt": unrelated_prompt,
                "selected_agent_mode": selected_mode,
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": selected_mode, "score": 4},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 12000 + idx,
                        "model_calls": 9,
                        "tool_calls": 0,
                        "cost_units": 9.0,
                    }
                },
            },
        )

    payload = manager.handle_prompt(
        session_id="unrelated-economic-pressure-session",
        prompt=target_prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["selected_agent_mode"] == "loop"
    assert policy["feedback_adjustment"] is None
    assert policy["feedback_summary"]["used_recent_fallback"] is True
    assert policy["feedback_summary"]["relevant_feedback"] == 0
    assert captured["settings"]["agent_mode"] == "loop"


def test_auto_agent_max_budget_ignores_passive_economics_pressure(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(
        runtime_module.chat_app,
        "estimate_auto_reasoning_cycles",
        lambda prompt: {"score": 4, "cycles": 4, "reasons": ["workflow_depth"]},
    )
    captured = {}

    def fake_run_loop_agent_text(**kwargs):
        captured.update(kwargs)
        chosen_record = kwargs["chosen_record"]
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=kwargs["route_reason"],
            response=f"auto routed through {kwargs['settings']['agent_mode']}",
            prompt_used=kwargs["prompt"],
            agent_trace={"agent_mode": "collective_loop_agent", "loop_steps": []},
        )

    monkeypatch.setattr(manager, "_run_loop_agent_text", fake_run_loop_agent_text)
    prompt = "Implement a runtime integration and verify the tests."

    for idx, selected_mode in enumerate(("loop", "collective")):
        manager.record_route_feedback(
            session_id="max-economic-pressure-session",
            feedback={
                "route_id": f"route-{idx}",
                "prompt": prompt,
                "selected_agent_mode": selected_mode,
                "rating": "down",
                "auto_agent_policy": {"selected_agent_mode": selected_mode, "score": 4},
                "route_economics": {
                    "actual": {
                        "elapsed_ms": 25000 + idx,
                        "model_calls": 15,
                        "tool_calls": 0,
                        "cost_units": 15.0,
                    }
                },
            },
        )

    payload = manager.handle_prompt(
        session_id="max-economic-pressure-session",
        prompt=prompt,
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "auto",
            "auto_agent_budget": "max",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    policy = payload["agent_trace"]["auto_agent_policy"]
    assert policy["budget_profile"] == "max"
    assert policy["selected_agent_mode"] == "collective_loop"
    assert policy["feedback_adjustment"] is None
    assert captured["settings"]["agent_mode"] == "collective_loop"


def test_model_store_catalog_marks_installed_and_selectable_records(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    installed = models_dir / "dcgan_v2_in_progress.zip"
    installed.write_bytes(b"zip")
    records = (
        _record("dcgan_v2_in_progress", "dcgan_image", ("image",), None),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
        models_dir=models_dir,
        common_summary_path=tmp_path / "missing_summary.json",
    )

    manager._fetch_model_store_manifest_locked = lambda force_refresh=False: {  # type: ignore[method-assign]
        "models": [
            {"file_name": "dcgan_v2_in_progress.zip", "size_bytes": 3, "size_mb": 0.0, "family": "gan"},
            {"file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip", "size_bytes": 10, "size_mb": 0.0, "family": "fusion"},
        ]
    }

    payload = manager.model_store_catalog(force_refresh=True)
    by_name = {row["file_name"]: row for row in payload["models"]}
    assert by_name["dcgan_v2_in_progress.zip"]["installed"] is True
    assert by_name["dcgan_v2_in_progress.zip"]["selectable"] is True
    assert by_name["supermix_omni_collective_v8_preview_20260407_001155.zip"]["known"] is True
    assert by_name["supermix_omni_collective_v8_preview_20260407_001155.zip"]["installed"] is False


def test_model_store_catalog_skips_unsafe_remote_manifest_names(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manager = UnifiedModelManager(
        records=(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
        models_dir=models_dir,
        common_summary_path=tmp_path / "missing_summary.json",
    )

    manager._fetch_model_store_manifest_locked = lambda force_refresh=False: {  # type: ignore[method-assign]
        "models": [
            {"file_name": "../escape.zip", "size_bytes": 10},
            {"file_name": "nested/escape.zip", "size_bytes": 10},
            {"file_name": "dcgan_v2_in_progress.zip", "size_bytes": 3},
        ]
    }

    payload = manager.model_store_catalog(force_refresh=True)

    assert [row["file_name"] for row in payload["models"]] == ["dcgan_v2_in_progress.zip"]
    assert not (tmp_path / "escape.zip").exists()


def test_model_store_install_rejects_unsafe_artifact_names(tmp_path: Path) -> None:
    manager = UnifiedModelManager(
        records=(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
        models_dir=tmp_path / "models",
        common_summary_path=tmp_path / "missing_summary.json",
    )
    manager._fetch_model_store_manifest_locked = lambda force_refresh=False: {  # type: ignore[method-assign]
        "models": [{"file_name": "dcgan_v2_in_progress.zip", "size_bytes": 3}]
    }

    for bad_name in ("../escape.zip", "nested/escape.zip", "bad:name.zip", "not-a-zip.txt"):
        with pytest.raises(ValueError):
            manager.install_model_store_artifact(bad_name)


def test_loop_agent_runs_until_reviewer_marks_complete(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    counters = {"planner": 0, "worker": 0, "review": 0}

    def fake_run_text_model(record, *, session_id, prompt, settings, route_reason, tool_cache, allow_tool_calls):
        if "planner sub-agent" in prompt:
            counters["planner"] += 1
            return ChatResult(
                kind="text",
                model_key=record.key,
                model_label=record.label,
                route_reason=route_reason,
                response=(
                    "DONE: no\n"
                    "STEP_GOAL: produce the strongest final answer\n"
                    "SUCCESS_SIGNAL: the user request is fully handled\n"
                    "WORKING_NOTES: tighten the answer and check completeness"
                ),
                prompt_used=prompt,
            ), []
        if "worker sub-agent" in prompt:
            counters["worker"] += 1
            return ChatResult(
                kind="text",
                model_key=record.key,
                model_label=record.label,
                route_reason=route_reason,
                response=(
                    f"DONE: {'yes' if counters['worker'] >= 2 else 'no'}\n"
                    f"OUTPUT: draft answer pass {counters['worker']}\n"
                    "NEXT_FOCUS: close the remaining gaps"
                ),
                prompt_used=prompt,
            ), []
        counters["review"] += 1
        return ChatResult(
            kind="text",
            model_key=record.key,
            model_label=record.label,
            route_reason=route_reason,
            response=(
                f"COMPLETE: {'yes' if counters['review'] >= 2 else 'no'}\n"
                f"SCORE: {0.97 if counters['review'] >= 2 else 0.45}\n"
                f"CONFIDENCE: {0.96 if counters['review'] >= 2 else 0.40}\n"
                "RISK_SCORE: 0.05\n"
                f"FINAL_RESPONSE: final answer pass {counters['review']}\n"
                "REASON: the loop has converged on a complete response\n"
                "EVIDENCE: reviewer observed all requested work\n"
                "NEXT_STEP: none"
            ),
            prompt_used=prompt,
        ), []

    monkeypatch.setattr(manager, "_run_text_model", fake_run_text_model)

    payload = manager.handle_prompt(
        session_id="loop-session",
        prompt="Finish this task autonomously.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "loop",
            "loop_max_steps": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert payload["model_key"] == "omni_collective_v8"
    assert payload["agent_trace"]["agent_mode"] == "loop_agent"
    assert payload["agent_trace"]["loop_completed"] is True
    assert payload["agent_trace"]["loop_budget"] == 4
    assert len(payload["agent_trace"]["loop_steps"]) == 2
    assert payload["agent_trace"]["loop_steps"][0]["review_score"] == 0.45
    assert payload["agent_trace"]["loop_steps"][1]["review_score"] == 0.97
    assert payload["agent_trace"]["loop_steps"][1]["stop_decision"] == "stop"
    assert payload["agent_trace"]["loop_stop_reason_code"] == "reviewer_complete"
    assert payload["agent_trace"]["loop_stop_step"] == 2
    assert payload["agent_trace"]["loop_stop_score"] == 0.97
    economics = payload["agent_trace"]["route_economics"]
    assert economics["estimate"]["selected_agent_mode"] == "loop"
    assert economics["estimate"]["estimated_model_calls"] == 12
    assert economics["actual"]["model_calls"] == 6
    assert economics["actual"]["loop_steps"] == 2
    assert payload["timing"]["route_elapsed_ms"] == economics["actual"]["elapsed_ms"]
    assert "final answer pass 2" in payload["response"]
    assert counters == {"planner": 2, "worker": 2, "review": 2}


def test_loop_agent_records_budget_stop_telemetry(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    def fake_run_text_model(record, *, session_id, prompt, settings, route_reason, tool_cache, allow_tool_calls):
        if "planner sub-agent" in prompt:
            response = (
                "DONE: no\n"
                "STEP_GOAL: refine the incomplete answer\n"
                "SUCCESS_SIGNAL: the answer is complete\n"
                "WORKING_NOTES: continue iterating"
            )
        elif "worker sub-agent" in prompt:
            response = "DONE: no\nOUTPUT: partial answer\nNEXT_FOCUS: finish the remaining proof"
        else:
            response = (
                "COMPLETE: no\n"
                "SCORE: 0.4\n"
                "CONFIDENCE: 0.5\n"
                "RISK_SCORE: 0.2\n"
                "FINAL_RESPONSE: partial answer\n"
                "REASON: still missing proof\n"
                "EVIDENCE: reviewer found a remaining gap\n"
                "NEXT_STEP: add the missing proof"
            )
        return ChatResult(
            kind="text",
            model_key=record.key,
            model_label=record.label,
            route_reason=route_reason,
            response=response,
            prompt_used=prompt,
        ), []

    monkeypatch.setattr(manager, "_run_text_model", fake_run_text_model)

    payload = manager.handle_prompt(
        session_id="loop-budget-session",
        prompt="Finish this task autonomously.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "loop",
            "loop_max_steps": 2,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert payload["agent_trace"]["loop_completed"] is False
    assert payload["agent_trace"]["loop_stop_reason_code"] == "budget_exhausted"
    assert payload["agent_trace"]["loop_stop_step"] == 2
    assert payload["agent_trace"]["loop_stop_score"] == 0.4
    assert payload["agent_trace"]["loop_steps"][-1]["stop_decision"] == "continue"
    assert "loop budget ended" in payload["response"]


def test_loop_agent_can_stop_on_high_score_without_yes_no_label(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    calls = {"review": 0}

    def fake_run_text_model(record, *, session_id, prompt, settings, route_reason, tool_cache, allow_tool_calls):
        if "planner sub-agent" in prompt:
            response = (
                "DONE: no\n"
                "STEP_GOAL: produce a final answer\n"
                "SUCCESS_SIGNAL: the answer satisfies the request\n"
                "WORKING_NOTES: one pass should be enough"
            )
        elif "worker sub-agent" in prompt:
            response = "DONE: yes\nOUTPUT: complete answer\nNEXT_FOCUS: none"
        else:
            calls["review"] += 1
            response = (
                "SCORE: 0.91\n"
                "CONFIDENCE: 0.92\n"
                "RISK_SCORE: 0.10\n"
                "FINAL_RESPONSE: complete answer\n"
                "REASON: score evidence clears the completion threshold\n"
                "EVIDENCE: all requested parts are present\n"
                "NEXT_STEP: none"
            )
        return ChatResult(
            kind="text",
            model_key=record.key,
            model_label=record.label,
            route_reason=route_reason,
            response=response,
            prompt_used=prompt,
        ), []

    monkeypatch.setattr(manager, "_run_text_model", fake_run_text_model)

    payload = manager.handle_prompt(
        session_id="loop-score-session",
        prompt="Finish this task autonomously.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "loop",
            "loop_max_steps": 4,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    step = payload["agent_trace"]["loop_steps"][0]
    assert payload["agent_trace"]["loop_completed"] is True
    assert payload["agent_trace"]["loop_stop_reason_code"] == "score_threshold"
    assert payload["agent_trace"]["loop_stop_step"] == 1
    assert payload["agent_trace"]["loop_stop_score"] == 0.91
    assert step["review_complete"] is None
    assert step["stop_decision"] == "stop"
    assert calls["review"] == 1


def test_collective_loop_agent_uses_collective_worker(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("v40_benchmax", "omni_collective_v5", ("chat", "vision"), 0.2433),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    collective_calls = {"count": 0}

    def fake_run_text_model(record, *, session_id, prompt, settings, route_reason, tool_cache, allow_tool_calls):
        if "planner sub-agent" in prompt:
            response = (
                "DONE: no\n"
                "STEP_GOAL: consult the panel and produce the final answer\n"
                "SUCCESS_SIGNAL: the task is fully addressed\n"
                "WORKING_NOTES: use the panel once and stop if complete"
            )
        else:
            response = (
                "COMPLETE: yes\n"
                "FINAL_RESPONSE: collective loop final\n"
                "REASON: the task is complete\n"
                "NEXT_STEP: none"
            )
        return ChatResult(
            kind="text",
            model_key=record.key,
            model_label=record.label,
            route_reason=route_reason,
            response=response,
            prompt_used=prompt,
        ), []

    def fake_run_agent_text(*, session_id, prompt, chosen_record, settings, route_reason, action_mode, memory_bundle):
        collective_calls["count"] += 1
        return ChatResult(
            kind="text",
            model_key=chosen_record.key,
            model_label=chosen_record.label,
            route_reason=route_reason,
            response="DONE: yes\nOUTPUT: panel-produced answer\nNEXT_FOCUS: none",
            prompt_used=prompt,
            agent_trace={
                "agent_mode": "collective_panel",
                "consulted_models": ["omni_collective_v8", "v40_benchmax"],
                "consultation_rows": [{"model_key": "v40_benchmax", "model_label": "v40_benchmax", "response": "panel note"}],
                "tool_events": [],
            },
        )

    monkeypatch.setattr(manager, "_run_text_model", fake_run_text_model)
    monkeypatch.setattr(manager, "_run_agent_text", fake_run_agent_text)

    payload = manager.handle_prompt(
        session_id="collective-loop-session",
        prompt="Solve this with the collective loop.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "collective_loop",
            "loop_max_steps": 3,
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert payload["agent_trace"]["agent_mode"] == "collective_loop_agent"
    assert payload["agent_trace"]["loop_worker_mode"] == "collective"
    assert payload["agent_trace"]["loop_completed"] is True
    assert collective_calls["count"] == 1
    assert payload["agent_trace"]["consulted_models"] == ["omni_collective_v8", "v40_benchmax"]


def test_collective_agent_skips_broken_consultant_and_still_returns(tmp_path: Path, monkeypatch) -> None:
    records = (
        _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133),
        _record("qwen_v28", "qwen_adapter", ("chat",), 0.42),
    )
    manager = UnifiedModelManager(
        records=records,
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    def fake_run_text_model(record, *, session_id, prompt, settings, route_reason, tool_cache, allow_tool_calls):
        if record.key == "qwen_v28":
            raise FileNotFoundError("missing adapter_model.safetensors")
        return ChatResult(
            kind="text",
            model_key=record.key,
            model_label=record.label,
            route_reason=route_reason,
            response="safe panel answer",
            prompt_used=prompt,
        ), []

    monkeypatch.setattr(manager, "_run_text_model", fake_run_text_model)

    payload = manager.handle_prompt(
        session_id="collective-session",
        prompt="Answer despite one broken consultant.",
        model_key="omni_collective_v8",
        action_mode="text",
        settings={
            "agent_mode": "collective",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert payload["response"] == "safe panel answer"
    assert payload["agent_trace"]["agent_mode"] == "collective_panel"
    assert payload["agent_trace"]["consulted_models"] == ["omni_collective_v8"]
    assert payload["agent_trace"]["skipped_models"][0]["model_key"] == "qwen_v28"


def test_auto_route_feedback_ignores_unrelated_fallback_quality_votes(tmp_path: Path) -> None:
    record = _record("omni_collective_v8", "omni_collective_v8", ("chat", "vision"), 0.2133)
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    allowed_modes = ("off", "collective", "loop", "collective_loop")

    downgraded, downgrade_adjustment = manager._apply_auto_route_feedback(
        selected="loop",
        score=4,
        feedback_summary={
            "used_recent_fallback": True,
            "mode_scores": {
                "loop": {
                    "quality_net": -2,
                    "quality_negative": 2,
                    "adaptive": {},
                }
            },
        },
        allowed_modes=allowed_modes,
    )
    assert downgraded == "loop"
    assert downgrade_adjustment is None

    upgraded, upgrade_adjustment = manager._apply_auto_route_feedback(
        selected="collective",
        score=4,
        feedback_summary={
            "used_recent_fallback": True,
            "mode_scores": {
                "collective": {"adaptive": {}},
                "loop": {"quality_net": 2, "quality_positive": 2},
            },
        },
        allowed_modes=allowed_modes,
    )
    assert upgraded == "collective"
    assert upgrade_adjustment is None


def test_route_economics_counts_tool_followup_model_invocation(tmp_path: Path, monkeypatch) -> None:
    record = _record("tool-model", "champion_chat", ("chat",), 0.9)
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    class _ToolBackend(_FakeBackend):
        def __init__(self, backend_record: ModelRecord) -> None:
            super().__init__(backend_record)
            self.calls = 0

        def chat(self, session_id: str, prompt: str, settings: dict) -> ChatResult:
            self.calls += 1
            response = "TOOL:web_search: current docs" if self.calls == 1 else "Answer grounded in the tool result."
            return ChatResult(
                kind="text",
                model_key=self.record.key,
                model_label=self.record.label,
                route_reason=str(settings.get("route_reason") or ""),
                response=response,
                prompt_used=prompt,
            )

    backend = _ToolBackend(record)
    monkeypatch.setattr(manager, "ensure_backend", lambda _key: (record, backend))

    def fake_web_query(query, tool_cache, settings):
        event = runtime_module.ToolEvent(
            name="web_search",
            query=query,
            results=[{"title": "Docs", "url": "https://example.test/docs", "snippet": "Current."}],
        )
        tool_cache[str(query).lower()] = event
        return event

    monkeypatch.setattr(manager, "_run_web_query_cached", fake_web_query)
    payload = manager.handle_prompt(
        session_id="tool-economics-session",
        prompt="Answer using the available evidence.",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": True,
            "cmd_open_enabled": False,
        },
    )

    actual = payload["agent_trace"]["route_economics"]["actual"]
    assert backend.calls == 2
    assert actual["model_calls"] == 2
    assert actual["tool_calls"] == 1
    assert actual["cost_units"] == 2.25

    second_payload = manager.handle_prompt(
        session_id="tool-economics-session",
        prompt="Answer without a tool.",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )
    assert backend.calls == 3
    assert second_payload["agent_trace"]["route_economics"]["actual"]["model_calls"] == 1


def test_image_wrapper_counts_refiner_and_pipeline_in_route_economics(tmp_path: Path, monkeypatch) -> None:
    record = _record("image-wrapper", "image_wrapper", ("chat", "image"), 0.9)
    backend = runtime_module.ImageWrapperBackend.__new__(runtime_module.ImageWrapperBackend)
    backend.record = record

    class _ImageEngine:
        def generate_image(self, **kwargs):
            return {
                "timing_ms": 25.0,
                "model_calls": 2,
                "refiner_model_calls": 1,
                "image_url": "/generated/test.png",
                "output_path": str(tmp_path / "test.png"),
                "prompt_used": kwargs["prompt"],
                "refined_prompt": "refined image prompt",
            }

    backend.image_engine = _ImageEngine()
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(manager, "ensure_backend", lambda _key: (record, backend))

    payload = manager.handle_prompt(
        session_id="image-economics-session",
        prompt="Paint a lunar observatory.",
        model_key=record.key,
        action_mode="image",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
            "use_text_refiner": True,
        },
    )

    actual = payload["agent_trace"]["route_economics"]["actual"]
    assert payload["timing"]["model_calls"] == 2
    assert payload["timing"]["refiner_model_calls"] == 1
    assert actual["model_calls"] == 2
    assert actual["cost_units"] == 2.0


def test_image_prompt_refiner_marks_its_model_invocation(monkeypatch) -> None:
    from source.chat_image_variant_app import ImageVariantEngine

    engine = ImageVariantEngine.__new__(ImageVariantEngine)

    class _TextEngine:
        @staticmethod
        def status():
            return {"loaded": True}

    engine.text_engine = _TextEngine()
    monkeypatch.setattr(engine, "_refine_prompt_with_text_model", lambda _prompt: "A moon base under blue light")

    refined = engine._build_final_prompt("A moon base", style="auto", use_text_refiner=True)
    unrefined = engine._build_final_prompt("A moon base", style="auto", use_text_refiner=False)

    assert refined["refiner_model_calls"] == 1
    assert unrefined["refiner_model_calls"] == 0
