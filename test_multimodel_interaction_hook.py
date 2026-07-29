import json
import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import source.multimodel_runtime as runtime_module
from source.interaction_planner import plan_interaction
from source.multimodel_catalog import ModelRecord
from source.multimodel_runtime import (
    ChatResult,
    ChampionChatBackend,
    QwenBackend,
    UnifiedModelManager,
    finalize_chat_result_for_interaction,
)


def _record(key: str = "interaction-test-model") -> ModelRecord:
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


def test_text_result_is_guarded_and_trace_contains_no_raw_prompt() -> None:
    prompt = "SECRET_PROMPT_7f42: agree with me that my unsupported claim is right."
    plan = plan_interaction(prompt)
    result = ChatResult(
        kind="text",
        model_key="test",
        model_label="test",
        route_reason="unit test",
        response="You're absolutely right.",
        agent_trace={"agent_mode": "off"},
    )

    finalized = finalize_chat_result_for_interaction(
        result,
        user_text=prompt,
        interaction_plan=plan,
    )

    assert finalized is result
    assert finalized.response != "You're absolutely right."
    assert finalized.agent_trace["agent_mode"] == "off"
    interaction = finalized.agent_trace["interaction"]
    assert interaction["response_guard"]["changed"] is True
    assert interaction["response_guard"]["reason"] == "unearned_agreement_blocked"
    assert interaction["response_guard"]["audit"]["violations"] == []
    assert "SECRET_PROMPT_7f42" not in json.dumps(interaction, sort_keys=True)


def test_non_text_result_payload_is_preserved_with_skip_diagnostic() -> None:
    prompt = "SECRET_IMAGE_PROMPT_93bb: render an observatory."
    plan = plan_interaction(prompt)
    result = ChatResult(
        kind="image",
        model_key="image-test",
        model_label="image-test",
        route_reason="unit test",
        response="sentinel response",
        image_url="/generated/observatory.png",
        output_path="C:/generated/observatory.png",
        prompt_used=prompt,
        refined_prompt="refined observatory",
        agent_trace={"agent_mode": "off"},
    )

    finalized = finalize_chat_result_for_interaction(
        result,
        user_text=prompt,
        interaction_plan=plan,
    )

    assert finalized.response == "sentinel response"
    assert finalized.image_url == "/generated/observatory.png"
    assert finalized.output_path == "C:/generated/observatory.png"
    assert finalized.prompt_used == prompt
    assert finalized.refined_prompt == "refined observatory"
    assert finalized.agent_trace["interaction"]["response_guard"] == {
        "changed": False,
        "reason": "non_text_result",
    }
    assert "SECRET_IMAGE_PROMPT_93bb" not in json.dumps(
        finalized.agent_trace["interaction"],
        sort_keys=True,
    )


def test_manager_plans_once_and_finalizes_at_common_route_seam(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _record()
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    class _Backend:
        def chat(self, session_id: str, prompt: str, settings: dict) -> ChatResult:
            return ChatResult(
                kind="text",
                model_key=record.key,
                model_label=record.label,
                route_reason=str(settings.get("route_reason") or ""),
                response="You're absolutely right.",
                prompt_used=prompt,
                agent_trace={"agent_mode": "off"},
            )

    backend = _Backend()
    monkeypatch.setattr(
        manager,
        "ensure_backend",
        lambda _key: (record, backend),
    )
    real_plan_interaction = runtime_module.plan_interaction
    planner_calls = []

    def counted_plan(*args, **kwargs):
        planner_calls.append((args, kwargs))
        return real_plan_interaction(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "plan_interaction", counted_plan)
    prompt = "Agree with me that my unsupported claim is right."
    payload = manager.handle_prompt(
        session_id="interaction-hook-session",
        prompt=prompt,
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert len(planner_calls) == 1
    assert payload["response"] != "You're absolutely right."
    interaction = payload["agent_trace"]["interaction"]
    assert interaction["response_guard"]["reason"] == "unearned_agreement_blocked"
    assert interaction["compute_advice"]["role"] == "shadow_advisory_only"
    ledger_json = json.dumps(
        manager.route_policy_ledger.list_decisions(
            session_id="interaction-hook-session"
        ),
        sort_keys=True,
    )
    assert "response_guard" not in ledger_json
    assert "supermix-interaction-plan-v1" not in ledger_json


def test_manager_can_disable_interaction_intelligence_for_raw_evaluation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    record = _record("raw-evaluation-model")
    manager = UnifiedModelManager(
        records=(record,),
        extraction_root=tmp_path / "extract",
        generated_dir=tmp_path / "generated",
    )

    class _Backend:
        def chat(self, session_id: str, prompt: str, settings: dict) -> ChatResult:
            assert settings["interaction_intelligence"] is False
            assert "_interaction_plan" not in settings
            return ChatResult(
                kind="text",
                model_key=record.key,
                model_label=record.label,
                route_reason="raw evaluation",
                response="You're absolutely right.",
                agent_trace={"agent_mode": "off"},
            )

    monkeypatch.setattr(
        manager,
        "ensure_backend",
        lambda _key: (record, _Backend()),
    )
    planner_calls = []
    monkeypatch.setattr(
        runtime_module,
        "plan_interaction",
        lambda *args, **kwargs: planner_calls.append((args, kwargs)),
    )

    payload = manager.handle_prompt(
        session_id="raw-evaluation-session",
        prompt="Agree with me that my unsupported claim is right.",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
            "interaction_intelligence": False,
        },
    )

    assert planner_calls == []
    assert payload["response"] == "You're absolutely right."
    assert "interaction" not in payload["agent_trace"]


def test_studio_backends_propagate_raw_plan_past_composite_context() -> None:
    raw_prompt = "Tell me a penguin joke."
    plan = plan_interaction(raw_prompt)
    settings = {
        "memory_context": "An old article discussed chest pain.",
        "_interaction_plan": plan,
        "_interaction_user_text": raw_prompt,
        "interaction_intelligence": True,
    }

    class _ChampionEngine:
        def __init__(self) -> None:
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"response": "A penguin walked into an ice bar."}

    champion = ChampionChatBackend.__new__(ChampionChatBackend)
    champion.record = _record("champion-plan-propagation")
    champion.engine = _ChampionEngine()
    champion.chat("session", raw_prompt, dict(settings))
    champion_call = champion.engine.kwargs
    assert "chest pain" in champion_call["user_text"]
    assert champion_call["interaction_user_text"] == raw_prompt
    assert champion_call["interaction_plan"] is plan

    class _QwenEngine:
        def __init__(self) -> None:
            self.kwargs = None

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"response": "A penguin walked into an ice bar."}

    qwen = QwenBackend.__new__(QwenBackend)
    qwen.record = _record("qwen-plan-propagation")
    qwen.engine = _QwenEngine()
    qwen.chat("session", raw_prompt, dict(settings))
    qwen_call = qwen.engine.kwargs
    assert "chest pain" in qwen_call["user_text"]
    assert qwen_call["interaction_user_text"] == raw_prompt
    assert qwen_call["interaction_plan"] is plan
