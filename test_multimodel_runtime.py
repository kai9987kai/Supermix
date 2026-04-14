import sys
from pathlib import Path

from source.multimodel_catalog import ModelRecord

SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

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
                f"FINAL_RESPONSE: final answer pass {counters['review']}\n"
                "REASON: the loop has converged on a complete response\n"
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
    assert "final answer pass 2" in payload["response"]
    assert counters == {"planner": 2, "worker": 2, "review": 2}


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
