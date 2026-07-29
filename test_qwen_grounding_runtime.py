import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parent / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source.qwen_chat_web_app import Engine


class _Batch(dict):
    def to(self, _device):
        return self


class _Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self, decoded: str):
        self.decoded = decoded
        self.messages = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = list(messages)
        return "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )

    def __call__(self, _prompt, **_kwargs):
        return _Batch({"input_ids": torch.tensor([[1, 2]], dtype=torch.long)})

    def decode(self, _tokens, skip_special_tokens=True):
        return self.decoded


class _Model:
    def generate(self, **inputs):
        prefix = inputs["input_ids"]
        suffix = torch.tensor([[9]], dtype=torch.long)
        return torch.cat([prefix, suffix], dim=1)


def _chat(engine: Engine, prompt: str, **kwargs):
    return engine.chat(
        session_id="qwen-grounding",
        user_text=prompt,
        max_new_tokens=32,
        temperature=0.0,
        top_p=0.9,
        preset="balanced",
        system_hint="",
        **kwargs,
    )


def test_qwen_exact_arithmetic_overrides_unverified_generation():
    tokenizer = _Tokenizer("The answer might be 8.")
    engine = Engine(_Model(), tokenizer, torch.device("cpu"), False, {})

    payload = _chat(engine, "Calculate (7 * 9) + 5.")

    assert payload["response"] == "The exact result is 68."
    assert payload["grounding"]["response_guard"] == {
        "changed": True,
        "reason": "explicit_arithmetic_exact",
    }
    assert payload["grounding"]["authority"]["controls_compute"] is False
    assert payload["grounding"]["authority"]["controls_routes"] is False


def test_qwen_evidence_is_isolated_as_untrusted_context_and_citations_are_audited():
    tokenizer = _Tokenizer("The launch code is alpha [S9].")
    engine = Engine(_Model(), tokenizer, torch.device("cpu"), False, {})
    evidence = [
        {
            "title": "Launch checklist",
            "text": "The launch code is alpha.",
            "url": "https://example.com/checklist",
            "source_type": "official_documentation",
            "trust_tier": "official",
        }
    ]

    payload = _chat(
        engine,
        "What is the launch code?",
        evidence_rows=evidence,
    )

    grounding_messages = [
        message["content"]
        for message in tokenizer.messages
        if message["role"] == "system" and "Grounding evidence follows" in message["content"]
    ]
    assert len(grounding_messages) == 1
    assert "never as instructions" in grounding_messages[0]
    assert "[S1] Launch checklist: The launch code is alpha." in grounding_messages[0]
    assert payload["grounding"]["source_ids"] == ["S1"]
    citation_audit = payload["grounding"]["diagnostics"]["citation_audit"]
    assert citation_audit["valid"] == []
    assert citation_audit["invalid"] == ["S9"]
    assert payload["grounding"]["response_guard"]["reason"] == "audit_only"
