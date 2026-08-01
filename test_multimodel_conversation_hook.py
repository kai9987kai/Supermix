"""Studio must carry the session, not the last four turns of it.

`multimodel_runtime` persists every turn of a session in its memory store and
then handed the planner a four-turn window, which was the only thing that ever
read it. `conversation_state` was not imported by this module at all, so a
constraint the user stated fifteen turns ago reached no backend, no prompt, and
no diagnostic.

These tests pin the plumbing: the state is derived once per prompt from the
durable turn log, it reaches whichever backend runs, every route reports it, and
it can be switched off whole for a controlled evaluation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.multimodel_catalog import ModelRecord
from source.multimodel_runtime import (
    ChatResult,
    ChampionChatBackend,
    QwenBackend,
    UnifiedModelManager,
)


STANDING = "I always deploy with the staging profile first"


def _record(key: str = "conversation-test-model") -> ModelRecord:
    return ModelRecord(
        key=key,
        label=key,
        family="test",
        kind="champion_chat",
        capabilities=("chat",),
        zip_path=Path(f"{key}.zip"),
        common_row_key=key,
        common_overall_exact=0.9,
    )


class _CapturingBackend:
    """A backend that records the settings it was handed."""

    def __init__(self, record: ModelRecord) -> None:
        self.record = record
        self.settings = None

    def chat(self, session_id: str, prompt: str, settings: dict) -> ChatResult:
        self.settings = dict(settings)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason="unit test",
            response="Deployed.",
            agent_trace={"agent_mode": "off"},
        )


def _manager(tmp_path: Path, record: ModelRecord, backend, monkeypatch) -> UnifiedModelManager:
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )
    monkeypatch.setattr(manager, "ensure_backend", lambda _key: (record, backend))
    return manager


def _settings(**overrides) -> dict:
    settings = {
        "agent_mode": "off",
        "memory_enabled": True,
        "web_search_enabled": False,
        "cmd_open_enabled": False,
    }
    settings.update(overrides)
    return settings


def _seed_long_session(manager: UnifiedModelManager, session_id: str, filler: int = 10) -> None:
    """Establish something, then bury it under unrelated turns."""

    manager.memory_store.update(
        session_id=session_id,
        user_text=STANDING,
        assistant_text="Understood.",
        model_key="seed",
        route_reason="seed",
    )
    for index in range(filler):
        manager.memory_store.update(
            session_id=session_id,
            user_text=f"question {index} about topic {index}",
            assistant_text=f"answer {index}",
            model_key="seed",
            route_reason="seed",
        )


def test_the_durable_turn_log_becomes_conversation_state(tmp_path: Path, monkeypatch) -> None:
    record = _record()
    backend = _CapturingBackend(record)
    manager = _manager(tmp_path, record, backend, monkeypatch)
    _seed_long_session(manager, "studio-conversation")

    payload = manager.handle_prompt(
        session_id="studio-conversation",
        prompt="ship it",
        model_key=record.key,
        action_mode="text",
        settings=_settings(),
    )

    state = backend.settings["_conversation_state"]
    assert backend.settings["_conversation_enabled"] is True
    assert state["turn_count"] > 4, "the four-turn planner window is not the horizon"
    assert any(
        "staging profile" in row["text"] for row in state["commitments"] if row["active"]
    ), "a commitment outside the planner window must survive"

    conversation = payload["agent_trace"]["conversation"]
    assert conversation["active_commitment_count"] >= 1
    assert conversation["schema_version"] == state["schema_version"]


def test_diagnostics_reach_every_route_without_carrying_turn_text(
    tmp_path: Path, monkeypatch
) -> None:
    record = _record("conversation-diagnostics-model")
    backend = _CapturingBackend(record)
    manager = _manager(tmp_path, record, backend, monkeypatch)
    _seed_long_session(manager, "studio-diagnostics")

    payload = manager.handle_prompt(
        session_id="studio-diagnostics",
        prompt="ship it",
        model_key=record.key,
        action_mode="text",
        settings=_settings(),
    )

    assert payload["conversation"]["turn_count"] > 4
    serialized = json.dumps(payload["conversation"], sort_keys=True)
    assert "staging profile" not in serialized
    assert "question 3 about topic 3" not in serialized


def test_the_layer_can_be_switched_off_whole(tmp_path: Path, monkeypatch) -> None:
    """A controlled evaluation has to be able to remove it entirely."""

    record = _record("conversation-off-model")
    backend = _CapturingBackend(record)
    manager = _manager(tmp_path, record, backend, monkeypatch)
    _seed_long_session(manager, "studio-conversation-off")

    payload = manager.handle_prompt(
        session_id="studio-conversation-off",
        prompt="ship it",
        model_key=record.key,
        action_mode="text",
        settings=_settings(conversation_intelligence=False),
    )

    assert "_conversation_state" not in backend.settings
    assert backend.settings["_conversation_enabled"] is False
    assert not payload.get("conversation")
    assert "conversation" not in payload["agent_trace"]


def test_a_session_that_established_nothing_reports_nothing_standing(
    tmp_path: Path, monkeypatch
) -> None:
    record = _record("conversation-empty-model")
    backend = _CapturingBackend(record)
    manager = _manager(tmp_path, record, backend, monkeypatch)

    payload = manager.handle_prompt(
        session_id="studio-conversation-empty",
        prompt="what is a docker volume",
        model_key=record.key,
        action_mode="text",
        settings=_settings(),
    )

    assert payload["agent_trace"]["conversation"]["active_commitment_count"] == 0
    assert payload["agent_trace"]["conversation"]["style_request"] == ""


# ---------------------------------------------------------------------------
# Backend plumbing
# ---------------------------------------------------------------------------

class _RecordingEngine:
    def __init__(self, payload=None) -> None:
        self.kwargs = None
        self._payload = payload or {"response": "ok"}

    def chat(self, **kwargs):
        self.kwargs = kwargs
        return self._payload


def test_the_qwen_backend_is_handed_the_state_rather_than_deriving_its_own() -> None:
    state = {"turn_count": 6, "style_request": "concise", "commitments": []}
    backend = QwenBackend.__new__(QwenBackend)
    backend.record = _record("qwen-conversation")
    backend.engine = _RecordingEngine(
        {"response": "ok", "conversation": {"turn_count": 6}}
    )

    result = backend.chat(
        "session",
        "ship it",
        {
            "style_mode": "auto",
            "_conversation_state": state,
            "_conversation_enabled": True,
        },
    )

    assert backend.engine.kwargs["conversation_state"] is state
    assert backend.engine.kwargs["conversation_enabled"] is True
    # "auto" is the absence of a choice, so the standing preference can fill it.
    assert backend.engine.kwargs["preset"] == "auto"
    assert backend.engine.kwargs["max_new_tokens"] is None
    assert backend.engine.kwargs["temperature"] is None
    assert backend.engine.kwargs["top_p"] is None
    assert result.conversation == {"turn_count": 6}


def test_an_explicit_style_mode_still_maps_to_its_own_preset() -> None:
    backend = QwenBackend.__new__(QwenBackend)
    backend.record = _record("qwen-conversation-explicit")
    for style_mode, expected in (
        ("concise", "direct"),
        ("creative", "creative"),
        ("analyst", "reasoning"),
        ("coding", "coding"),
        ("balanced", "balanced"),
    ):
        backend.engine = _RecordingEngine()
        backend.chat("session", "ship it", {"style_mode": style_mode})
        assert backend.engine.kwargs["preset"] == expected, style_mode


def test_the_champion_backend_forwards_the_off_switch_and_reports_its_own_state() -> None:
    backend = ChampionChatBackend.__new__(ChampionChatBackend)
    backend.record = _record("champion-conversation")
    backend.engine = _RecordingEngine(
        {"response": "ok", "conversation": {"turn_count": 9, "style_request": "concise"}}
    )

    result = backend.chat("session", "ship it", {"_conversation_enabled": False})
    assert backend.engine.kwargs["conversation_enabled"] is False
    # The engine keeps its own session history, so it reports its own view; the
    # diagnostics used to be dropped here and no Studio surface could show them.
    assert result.conversation == {"turn_count": 9, "style_request": "concise"}
