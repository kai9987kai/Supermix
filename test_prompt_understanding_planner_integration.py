import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load(
    "source_prompt_understanding_planner_integration",
    "source/interaction_planner.py",
)
RUNTIME = _load(
    "runtime_prompt_understanding_planner_integration",
    "runtime_python/interaction_planner.py",
)
PROMPTS = _load(
    "source_prompt_understanding_planner_api",
    "source/prompt_understanding.py",
)


def _profile(*, blocking: bool = False, unresolved: bool = False):
    references = (
        [
            {
                "id": "ref_target",
                "kind": "deictic",
                "status": "unresolved",
                "resolved_id": "",
                "candidate_ids": [],
                "confidence": 0.95,
            }
        ]
        if unresolved
        else []
    )
    conflicts = (
        [
            {
                "id": "conflict_1",
                "members": ["constraint_a", "constraint_b"],
                "kind": "mutually_exclusive",
                "severity": "hard",
                "blocking": True,
            }
        ]
        if blocking
        else []
    )
    return {
        "schema_version": "supermix-prompt-understanding-v1",
        "version": "test",
        "normalization": {},
        "objectives": [
            {
                "id": "objective_1",
                "act": "solve",
                "mode": "required",
                "confidence": 0.96,
                "target_ref": "ref_target" if unresolved else "",
            }
        ],
        "constraints": [],
        "conflicts": conflicts,
        "references": references,
        "ambiguity": {
            "score": 0.9 if (blocking or unresolved) else 0.1,
            "status": (
                "clarification_required"
                if (blocking or unresolved)
                else "clear"
            ),
            "reasons": [],
            "clarification_required": bool(blocking or unresolved),
            "unresolved_reference_count": int(unresolved),
            "hard_conflict_count": int(blocking),
        },
        "knowledge": {
            "factual": False,
            "freshness_required": False,
            "evidence_requested": False,
            "citations_requested": False,
            "strict_evidence_only": False,
        },
        "safety": {
            "personal_crisis_signal": False,
            "urgent_health_signal": False,
        },
        "execution_policy": {},
        "response_contract": {
            "required_capabilities": ["steps"],
            "forbidden_capabilities": ["comparison"],
            "deterministic_constraint_ids": [],
            "semantic_constraint_ids": [],
            "mixed_objective": True,
        },
        "context": {
            "turn_relation": "standalone",
            "followup": False,
            "used_turn_ids": [],
            "available_turn_count": 0,
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
        },
    }


def _stub_api(profile, calls=None):
    calls = calls if calls is not None else []

    def analyze_prompt(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return profile

    def diagnostics(value):
        return {
            "schema_version": value.get("schema_version"),
            "objective_count": len(value.get("objectives", ())),
            "constraint_count": len(value.get("constraints", ())),
            "clarification_required": bool(
                value.get("ambiguity", {}).get("clarification_required")
            ),
        }

    def evaluate(response_text, _prompt, _profile):
        failed = "violates-format" in response_text
        return {
            "schema_version": "supermix-response-constraint-audit-v1",
            "accepted": not failed,
            "checked_constraint_ids": ["format_1"],
            "passed_constraint_ids": [] if failed else ["format_1"],
            "violations": (
                [
                    {
                        "constraint_id": "format_1",
                        "kind": "format",
                        "reason": "test_violation",
                    }
                ]
                if failed
                else []
            ),
            "unchecked_constraint_ids": [],
            "coverage": 0.0 if failed else 1.0,
        }

    return SimpleNamespace(
        analyze_prompt=analyze_prompt,
        prompt_understanding_diagnostics=diagnostics,
        evaluate_response_constraints=evaluate,
    )


def test_planner_computes_once_from_recent_turns_and_keeps_safe_diagnostics(
    monkeypatch,
):
    sentinel = "PRIVATE_SENTINEL_8b7c"
    profile = _profile()
    calls = []
    monkeypatch.setattr(
        SOURCE,
        "_PROMPT_UNDERSTANDING_MODULE",
        _stub_api(profile, calls),
    )
    recent_turns = [
        {
            "id": "turn_1",
            "user": "Earlier request.",
            "assistant": "Earlier answer.",
        }
    ]

    plan = SOURCE.plan_interaction(
        f"Solve this carefully. {sentinel}",
        context={"recent_turns": recent_turns},
    )
    diagnostics = SOURCE.interaction_plan_diagnostics(plan)

    assert len(calls) == 1
    assert calls[0][1]["recent_turns"] == recent_turns
    assert calls[0][1]["recent_user_messages"] == ["Earlier request."]
    assert calls[0][1]["recent_assistant_messages"] == ["Earlier answer."]
    assert plan["prompt_profile"] == profile
    assert plan["intent"]["primary"] == "problem_solving"
    assert "steps" in plan["response_contract"]["required_capabilities"]
    assert "comparison" in plan["response_contract"]["forbidden_capabilities"]
    assert sentinel not in json.dumps(diagnostics, sort_keys=True)


def test_unresolved_reference_and_hard_conflict_force_one_targeted_question(
    monkeypatch,
):
    for profile, expected_reason in (
        (_profile(unresolved=True), "unresolved_required_reference"),
        (_profile(blocking=True), "hard_constraint_conflict"),
    ):
        monkeypatch.setattr(
            SOURCE,
            "_PROMPT_UNDERSTANDING_MODULE",
            _stub_api(profile),
        )
        plan = SOURCE.plan_interaction(
            "Make it better.",
            prompt_profile=profile,
        )
        result = SOURCE.finalize_response_for_interaction(
            "I changed the implementation without asking.",
            "Make it better.",
            plan,
        )

        assert result["changed"] is True
        assert result["reason"] == expected_reason
        assert result["text"].count("?") == 1
        assert "changed the implementation" not in result["text"]


def test_raw_safety_fast_path_precedes_profile_clarification(monkeypatch):
    profile = _profile(blocking=True)
    monkeypatch.setattr(
        SOURCE,
        "_PROMPT_UNDERSTANDING_MODULE",
        _stub_api(profile),
    )
    user_text = "I want to kill myself and might act on it right now."
    plan = SOURCE.plan_interaction(user_text, prompt_profile=profile)
    result = SOURCE.finalize_response_for_interaction(
        "Wait until tomorrow.",
        user_text,
        plan,
    )

    assert plan["compute_advice"]["decision_exit_authority"] == (
        "checkpoint_bound_prediction_verifier"
    )
    assert result["reason"] == "crisis_safety_escalation"
    assert "emergency" in result["text"].lower()


def test_deterministic_constraint_penalty_is_bounded_and_audited(monkeypatch):
    profile = _profile()
    monkeypatch.setattr(
        SOURCE,
        "_PROMPT_UNDERSTANDING_MODULE",
        _stub_api(profile),
    )
    plan = SOURCE.plan_interaction(
        "Solve this in the requested format.",
        prompt_profile=profile,
    )
    compliant = SOURCE.score_candidate_for_interaction(
        "First, inspect the input. Second, verify the result.",
        plan,
    )
    violating = SOURCE.score_candidate_for_interaction(
        "First, inspect the input. Second, verify the result. violates-format",
        plan,
    )
    audit = SOURCE.evaluate_response_contract(
        "First, inspect the input. violates-format",
        "Solve this in the requested format.",
        plan,
    )

    assert 0.0 <= violating["contract_penalty"] <= 0.28
    assert compliant["total"] > violating["total"]
    assert "constraint:format_1" in audit["violations"]
    assert audit["constraint_audit"]["accepted"] is False


def test_real_exact_bullet_contract_changes_ranking_and_audit(monkeypatch):
    prompt = "Answer in exactly 2 bullets."
    profile = PROMPTS.analyze_prompt(prompt)
    monkeypatch.setattr(SOURCE, "_PROMPT_UNDERSTANDING_MODULE", PROMPTS)
    plan = SOURCE.plan_interaction(prompt, prompt_profile=profile)

    compliant = SOURCE.score_candidate_for_interaction(
        "- First result\n- Second result",
        plan,
    )
    violating = SOURCE.score_candidate_for_interaction(
        "- Only one result",
        plan,
    )
    audit = SOURCE.evaluate_response_contract(
        "- Only one result",
        prompt,
        plan,
    )

    assert compliant["total"] > violating["total"]
    assert compliant["constraint_audit"]["accepted"] is True
    assert audit["constraint_audit"]["accepted"] is False
    assert any(
        value.startswith("constraint:")
        for value in audit["violations"]
    )


def test_negated_capability_is_not_reintroduced_by_legacy_cue_matching():
    prompt = "Explain why the sky is blue in exactly one sentence and do not give steps."
    profile = PROMPTS.analyze_prompt(prompt)

    source_plan = SOURCE.plan_interaction(prompt, prompt_profile=profile)
    runtime_plan = RUNTIME.plan_interaction(prompt, prompt_profile=profile)

    for plan in (source_plan, runtime_plan):
        contract = plan["response_contract"]
        assert "steps" in contract["forbidden_capabilities"]
        assert "steps" not in contract["required_capabilities"]
        assert set(contract["required_capabilities"]).isdisjoint(
            contract["forbidden_capabilities"]
        )
        assert contract["mixed_objective"] is False
    assert source_plan["response_contract"] == runtime_plan["response_contract"]


def test_source_runtime_profile_integration_parity(monkeypatch):
    profile = _profile(unresolved=True)
    monkeypatch.setattr(
        SOURCE,
        "_PROMPT_UNDERSTANDING_MODULE",
        _stub_api(profile),
    )
    monkeypatch.setattr(
        RUNTIME,
        "_PROMPT_UNDERSTANDING_MODULE",
        _stub_api(profile),
    )
    context = {
        "recent_turns": [
            {"user": "Earlier request.", "assistant": "Earlier answer."}
        ]
    }

    assert SOURCE.plan_interaction(
        "Continue.",
        context=context,
        prompt_profile=profile,
    ) == RUNTIME.plan_interaction(
        "Continue.",
        context=context,
        prompt_profile=profile,
    )
