from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(os.getcwd(), "source"))

import chat_app  # noqa: E402
import chat_web_app  # noqa: E402
from chat_web_app import Engine  # noqa: E402


def _candidate(text: str) -> dict[str, object]:
    vec = chat_app.text_to_model_input(text, feature_mode="legacy")[0, 0].tolist()
    return {
        "text": text,
        "vec": vec,
        "ctx_vec": vec,
        "bucket_score": 1.0,
        "count": 1,
    }


class _Model(torch.nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], x.shape[1], 10, device=x.device)
        logits[..., 2] = 4.0
        return logits


def _engine() -> Engine:
    engine = Engine(torch.device("cpu"), {"resolved": "cpu"}, {"pool_mode": "all"})
    engine.model = _Model()
    engine.buckets = {2: [_candidate("A neutral candidate response.")]}
    engine.available_labels = [2]
    return engine


def _arithmetic_template_engine() -> Engine:
    engine = _engine()
    engine.buckets = {
        2: [
            _candidate(
                "Combine the first two amounts: 71 + 8 = 79. "
                "Apply the multiplier: 79 × 3 = 237. "
                "Remove 2: 237 - 2 = 235. Final answer: 235."
            )
        ]
    }
    return engine


def test_champion_analyzes_prompt_once_and_returns_private_diagnostics(
    monkeypatch,
) -> None:
    engine = _engine()
    real_analyze = chat_web_app.analyze_prompt
    calls: list[str] = []

    def counted(prompt, **kwargs):
        calls.append(str(prompt))
        return real_analyze(prompt, **kwargs)

    monkeypatch.setattr(chat_web_app, "analyze_prompt", counted)
    prompt = "Plase compair the optons and recomend the best one. SECRET_41329"

    result = engine.chat(
        session_id="understanding-once",
        user_text=prompt,
        response_temperature=0.0,
    )

    assert calls == [prompt]
    diagnostics = result["prompt_understanding"]
    assert diagnostics["schema_version"] == "supermix-prompt-understanding-v1"
    assert {"compare", "recommend"}.issubset(set(diagnostics["objective_acts"]))
    assert diagnostics["normalization"]["correction_count"] >= 2
    assert "SECRET_41329" not in json.dumps(diagnostics)
    assert result["interaction"]["prompt_understanding"]["schema_version"] == (
        "supermix-prompt-understanding-v1"
    )


def test_champion_asks_one_targeted_question_for_blocking_constraints() -> None:
    engine = _engine()

    result = engine.chat(
        session_id="understanding-conflict",
        user_text="Return exactly 2 bullets and exactly 3 bullets.",
        response_temperature=0.0,
    )

    assert result["response"].endswith("?")
    assert "conflicting requirements" in result["response"].lower()
    guard = result["interaction"]["response_guard"]
    assert guard["changed"] is True
    assert guard["reason"] == "hard_constraint_conflict"
    assert result["prompt_understanding"]["hard_conflict_count"] == 1
    assert (
        result["prompt_understanding"]["response_constraint_audit"]["accepted"]
        is False
    )


def test_champion_blocks_arithmetic_template_contamination_and_repairs_contract() -> None:
    engine = _arithmetic_template_engine()
    prompt = (
        "Do not give steps; answer in exactly one sentence. "
        "Explain why local caching helps."
    )

    result = engine.chat(
        session_id="template-contamination-fresh",
        user_text=prompt,
        response_temperature=0.0,
    )

    assert result["response"] == (
        "I don't have enough relevant information to answer that reliably."
    )
    assert "71" not in result["response"]
    assert "235" not in result["response"]
    guard = result["interaction"]["response_guard"]
    assert guard["changed"] is True
    assert guard["reason"] == "incompatible_arithmetic_template_blocked"
    assert guard["audit"]["accepted"] is True
    assert guard["audit"]["constraint_audit"]["accepted"] is True
    assert (
        result["prompt_understanding"]["response_constraint_audit"]["accepted"]
        is True
    )


def test_standalone_turn_does_not_inherit_prior_math_domain_permission() -> None:
    engine = _arithmetic_template_engine()
    session_id = "template-contamination-after-math"

    arithmetic = engine.chat(
        session_id=session_id,
        user_text="What is (71 + 8) * 3 - 2?",
        response_temperature=0.0,
    )
    result = engine.chat(
        session_id=session_id,
        user_text=(
            "Do not give steps; answer in exactly one sentence. "
            "Explain why local caching helps."
        ),
        response_temperature=0.0,
    )

    assert arithmetic["response"] == "The exact result is 235."
    assert result["interaction"]["prompt_understanding"]["context"]["followup"] is False
    assert result["response"] == (
        "I don't have enough relevant information to answer that reliably."
    )
    assert result["interaction"]["response_guard"]["reason"] == (
        "incompatible_arithmetic_template_blocked"
    )
    assert result["interaction"]["response_guard"]["audit"]["accepted"] is True


def test_current_arithmetic_request_is_not_blocked_as_template_contamination() -> None:
    engine = _arithmetic_template_engine()

    result = engine.chat(
        session_id="relevant-arithmetic-template",
        user_text="What is (71 + 8) * 3 - 2?",
        response_temperature=0.0,
    )

    assert result["response"] == "The exact result is 235."
    assert result["grounding"]["response_guard"]["reason"] == (
        "explicit_arithmetic_exact"
    )
    assert result["interaction"]["response_guard"]["reason"] != (
        "incompatible_arithmetic_template_blocked"
    )


def test_champion_web_surfaces_understanding_and_grounding_diagnostics() -> None:
    html = chat_web_app.HTML

    assert "function understandingText" in html
    assert "function groundingText" in html
    assert "contract pass" in html
    assert "response_constraint_audit" in html
    assert "data.prompt_understanding" in html
    assert "data.grounding" in html
