from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from multimodel_catalog import ModelRecord, choose_auto_model  # noqa: E402
from multimodel_runtime import (  # noqa: E402
    ChampionChatBackend,
    ToolEvent,
    UnifiedModelManager,
)
from prompt_understanding import analyze_prompt  # noqa: E402
import supermix_multimodel_web_app as studio_web  # noqa: E402


def _record(key: str, score: float) -> ModelRecord:
    return ModelRecord(
        key=key,
        label=key,
        family="test",
        kind="champion_chat",
        capabilities=("chat",),
        zip_path=Path(f"{key}.zip"),
        common_row_key=key,
        common_overall_exact=score,
    )


def test_profile_prevents_short_typoed_reasoning_request_from_fast_misroute() -> None:
    records = (
        _record("v30_lite", 0.01),
        _record("v40_benchmax", 0.9),
    )
    prompt = "Plase compair."

    legacy, _ = choose_auto_model(records, prompt)
    profile = analyze_prompt(prompt)
    understood, reason = choose_auto_model(
        records,
        prompt,
        prompt_profile=profile,
    )

    assert legacy is not None and legacy.key == "v30_lite"
    assert understood is not None and understood.key == "v40_benchmax"
    assert "reasoning" in reason.lower()


def test_studio_route_preview_uses_same_profile_before_model_choice(
    tmp_path: Path,
) -> None:
    manager = UnifiedModelManager(
        records=(_record("v30_lite", 0.01), _record("v40_benchmax", 0.9)),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    preview = manager.preview_route_plan(
        session_id="preview-profile",
        prompt="Plase compair.",
        model_key="auto",
        action_mode="auto",
        settings={"agent_mode": "off", "memory_enabled": False},
    )

    assert preview["active_model_key"] == "v40_benchmax"
    diagnostics = preview["prompt_understanding"]
    assert "compare" in diagnostics["objective_acts"]
    assert diagnostics["normalization"]["correction_count"] >= 1


def test_external_search_redacts_secrets_without_expanding_permission(
    tmp_path: Path,
) -> None:
    manager = UnifiedModelManager(
        records=(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    observed: list[str] = []

    def fake_search(query: str, max_results: int = 5) -> ToolEvent:
        del max_results
        observed.append(query)
        return ToolEvent(name="web_search", query=query, results=[])

    manager.web_search.search = fake_search
    prompt = "Find the current docs for api_key=supersecretvalue123"
    profile = analyze_prompt(prompt)

    disabled = manager._seed_auto_tool_events(
        prompt,
        {"web_search_enabled": False, "_prompt_profile": profile},
    )
    assert disabled == []
    assert observed == []

    enabled = manager._seed_auto_tool_events(
        prompt,
        {
            "web_search_enabled": True,
            "web_search_results": 5,
            "_prompt_profile": profile,
        },
    )
    assert len(enabled) == 1
    assert len(observed) == 1
    assert "supersecretvalue123" not in observed[0]
    assert "[REDACTED_SECRET]" in observed[0]
    assert profile["authority"]["may_expand_permissions"] is False


def test_model_requested_search_is_redacted_before_provider_call(
    tmp_path: Path,
) -> None:
    manager = UnifiedModelManager(
        records=(),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    observed: list[str] = []

    def fake_search(query: str, max_results: int = 5) -> ToolEvent:
        del max_results
        observed.append(query)
        return ToolEvent(name="web_search", query=query, results=[])

    manager.web_search.search = fake_search
    cache: dict[str, ToolEvent] = {}
    event = manager._run_web_query_cached(
        "latest release password=hunter-example-secret",
        cache,
        {"web_search_budget": 2, "web_search_results": 5},
    )

    assert event is not None
    assert observed and "hunter-example-secret" not in observed[0]
    assert "[REDACTED_SECRET]" in observed[0]
    assert all("hunter-example-secret" not in key for key in cache)


def test_champion_backend_forwards_precomputed_prompt_profile() -> None:
    record = _record("v40_benchmax", 0.9)
    captured: dict[str, object] = {}

    class _Engine:
        def chat(self, **kwargs):
            captured.update(kwargs)
            return {"response": "ok", "timing_ms": {}, "compute": {}}

    backend = ChampionChatBackend.__new__(ChampionChatBackend)
    backend.record = record
    backend.engine = _Engine()
    profile = analyze_prompt("Explain this.")

    backend.chat(
        "profile-forwarding",
        "Explain this.",
        {
            "_prompt_profile": profile,
            "_interaction_user_text": "Explain this.",
        },
    )

    assert captured["prompt_profile"] is profile
    assert captured["interaction_user_text"] == "Explain this."


def test_studio_ui_renders_private_understanding_trace() -> None:
    assert "trace.prompt_understanding" in studio_web.HTML_TEMPLATE
    assert "Constraint audit" in studio_web.HTML_TEMPLATE
    assert "Cue typo recovered" in studio_web.HTML_TEMPLATE
