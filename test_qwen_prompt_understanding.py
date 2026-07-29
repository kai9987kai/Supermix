from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


SOURCE = Path(__file__).resolve().parent / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from prompt_understanding import (  # noqa: E402
    analyze_prompt,
    evaluate_response_constraints,
)
from qwen_chat_web_app import (  # noqa: E402
    Engine,
    enforce_response_contract,
)


class _Batch(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, decoded: str):
        self.decoded = decoded
        self.messages: list[dict[str, str]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        self.messages = list(messages)
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def __call__(self, _prompt, **_kwargs):
        return _Batch({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})

    def decode(self, _tokens, skip_special_tokens=True):
        del skip_special_tokens
        return self.decoded


class _Model:
    def generate(self, **inputs):
        return torch.cat(
            (inputs["input_ids"], torch.tensor([[9]], dtype=torch.long)),
            dim=1,
        )


def test_response_contract_never_substitutes_canned_factual_answers() -> None:
    cases = (
        (
            "What is the difference between precision and recall?",
            "Use a model-specific explanation backed by the supplied evidence.",
        ),
        (
            "Explain overfitting in 3 bullets.",
            "- Custom point A.\n- Custom point B.\n- Custom point C.",
        ),
        (
            "What is 7 * 9? Just the answer.",
            "The locally generated candidate is sixty-three.",
        ),
    )
    for prompt, candidate in cases:
        repaired = enforce_response_contract(prompt, candidate)
        assert repaired == candidate


def test_response_contract_allows_one_format_only_bullet_repair() -> None:
    prompt = "Return exactly 2 bullets."
    profile = analyze_prompt(prompt)

    repaired = enforce_response_contract(
        prompt,
        "Alpha is useful. Beta is useful. Gamma is extra.",
        prompt_profile=profile,
    )

    assert repaired == "- Alpha is useful.\n- Beta is useful."
    assert evaluate_response_constraints(repaired, prompt, profile)["accepted"] is True


def test_response_contract_does_not_choose_between_blocking_constraints() -> None:
    prompt = "Return exactly 2 bullets and exactly 3 bullets."
    profile = analyze_prompt(prompt)
    candidate = "One. Two. Three."

    assert profile["ambiguity"]["clarification_required"] is True
    assert enforce_response_contract(prompt, candidate, profile) == candidate


def test_qwen_engine_injects_bounded_contract_and_returns_private_diagnostics() -> None:
    tokenizer = _Tokenizer("Caching reuses prior results.")
    engine = Engine(_Model(), tokenizer, torch.device("cpu"), False, {})
    prompt = "Explain caching in exactly one sentence. PROMPT_SECRET_93742"

    payload = engine.chat(
        session_id="prompt-understanding",
        user_text=prompt,
        max_new_tokens=32,
        temperature=0.0,
        top_p=0.9,
        preset="balanced",
        system_hint="",
        grounding_enabled=False,
    )

    contracts = [
        message["content"]
        for message in tokenizer.messages
        if message["role"] == "system"
        and str(message["content"]).startswith("PROMPT_CONTRACT")
    ]
    assert len(contracts) == 1
    assert "supermix-prompt-understanding-v1" in contracts[0]
    assert '"may_force_route":false' in contracts[0]
    assert '"controls_compute_exit":false' in contracts[0]

    diagnostics = payload["prompt_understanding"]
    assert diagnostics["schema_version"] == "supermix-prompt-understanding-v1"
    assert diagnostics["constraint_count"] >= 1
    assert diagnostics["response_constraint_audit"]["accepted"] is True
    assert "PROMPT_SECRET_93742" not in json.dumps(diagnostics)
