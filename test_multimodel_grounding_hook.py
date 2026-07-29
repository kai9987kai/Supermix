import sys
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import source.multimodel_runtime as runtime_module
from source.multimodel_catalog import ModelRecord
from source.multimodel_runtime import ChatResult, UnifiedModelManager


def _record(key: str) -> ModelRecord:
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


def _manager(tmp_path: Path, record: ModelRecord, monkeypatch, response: str):
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
                response=response,
                agent_trace={"agent_mode": "off"},
            )

    monkeypatch.setattr(manager, "ensure_backend", lambda _key: (record, _Backend()))
    return manager


def test_studio_plans_grounding_once_and_exact_solver_precedes_interaction_guard(
    tmp_path: Path,
    monkeypatch,
):
    record = _record("grounded-studio")
    manager = _manager(tmp_path, record, monkeypatch, "The answer might be 8.")
    real_plan = runtime_module.plan_grounding
    calls = []

    def counted_plan(*args, **kwargs):
        calls.append((args, kwargs))
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "plan_grounding", counted_plan)
    payload = manager.handle_prompt(
        session_id="grounding-on",
        prompt="Calculate (7 * 9) + 5.",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    assert len(calls) == 1
    assert payload["response"] == "The exact result is 68."
    grounding = payload["agent_trace"]["grounding"]
    assert grounding["response_guard"] == {
        "changed": True,
        "reason": "explicit_arithmetic_exact",
    }
    assert grounding["diagnostics"]["evidence_count"] == 0
    assert grounding["authority"]["controls_compute"] is False
    assert grounding["authority"]["controls_routes"] is False
    assert payload["agent_trace"]["interaction"]["response_guard"]["reason"] in {
        "candidate_aligned",
        "candidate_partially_aligned",
        "candidate_repaired",
    }


def test_studio_grounding_can_be_disabled_for_raw_evaluation(
    tmp_path: Path,
    monkeypatch,
):
    record = _record("ungrounded-studio")
    manager = _manager(tmp_path, record, monkeypatch, "The answer might be 8.")
    calls = []
    monkeypatch.setattr(
        runtime_module,
        "plan_grounding",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    payload = manager.handle_prompt(
        session_id="grounding-off",
        prompt="Calculate (7 * 9) + 5.",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
            "grounding_intelligence": False,
        },
    )

    assert calls == []
    assert payload["response"] == "The answer might be 8."
    assert "grounding" not in payload["agent_trace"]


def test_studio_solves_verified_word_problems_and_reports_prompt_free_reasoning(
    tmp_path: Path,
    monkeypatch,
):
    record = _record("reasoning-studio")
    manager = _manager(tmp_path, record, monkeypatch, "I am not sure, maybe 100 km/h.")

    payload = manager.handle_prompt(
        session_id="reasoning-on",
        prompt="A train travels 120 km in 2 hours. What is its speed?",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
        },
    )

    grounding = payload["agent_trace"]["grounding"]
    reasoning = grounding["reasoning"]

    assert "60" in payload["response"]
    assert grounding["response_guard"]["reason"] == "verified_reasoning_solution"
    assert reasoning["problem_class"] == "rate"
    assert reasoning["verified"] is True
    assert reasoning["override_allowed"] is True

    # The trace must carry metadata only, never the prompt or the answer.
    for leaked in ("120", "60", "train"):
        assert leaked not in str(reasoning)


def test_studio_reasoning_is_disabled_with_the_grounding_layer(
    tmp_path: Path,
    monkeypatch,
):
    record = _record("reasoning-off-studio")
    manager = _manager(tmp_path, record, monkeypatch, "I am not sure, maybe 100 km/h.")

    payload = manager.handle_prompt(
        session_id="reasoning-off",
        prompt="A train travels 120 km in 2 hours. What is its speed?",
        model_key=record.key,
        action_mode="text",
        settings={
            "agent_mode": "off",
            "memory_enabled": False,
            "web_search_enabled": False,
            "cmd_open_enabled": False,
            "grounding_intelligence": False,
        },
    )

    assert payload["response"] == "I am not sure, maybe 100 km/h."
    assert "grounding" not in payload["agent_trace"]
