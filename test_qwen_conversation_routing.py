"""The generative surface must carry what the session established.

`qwen_chat_web_app` built its prompt from the last twelve messages and nothing
else. A constraint stated twenty turns earlier was not under-weighted there, it
was absent: `conversation_state` was never imported by this surface at all, and
the client history was truncated to twelve messages on arrival, so no amount of
history sent by the browser could have changed that.

These tests assert routing, authority, generation-default propagation, and
transactional session behavior without requiring a checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"


def _load(name: str):
    saved = list(sys.path)
    sys.path.insert(0, str(SOURCE))
    try:
        spec = importlib.util.spec_from_file_location(f"qwenroute_{name}", SOURCE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = saved


@pytest.fixture(scope="module")
def qwen():
    return _load("qwen_chat_web_app")


@pytest.fixture(scope="module")
def state_module():
    return _load("conversation_state")


@pytest.fixture(scope="module")
def directive():
    return _load("conversation_directive")


def _long_session(first_user_message: str, filler_turns: int = 14):
    """A session whose opening message falls outside the prompt window."""

    history = [
        {"role": "user", "content": first_user_message},
        {"role": "assistant", "content": "Understood."},
    ]
    for index in range(filler_turns):
        history.append({"role": "user", "content": f"question number {index} about topic {index}"})
        history.append({"role": "assistant", "content": f"answer number {index}"})
    return history


# ---------------------------------------------------------------------------
# The horizon
# ---------------------------------------------------------------------------

def test_the_memory_horizon_is_longer_than_the_prompt_window(qwen) -> None:
    assert qwen.STATE_HISTORY_MESSAGES > qwen.PROMPT_HISTORY_MESSAGES
    assert qwen.MAX_SESSION_MESSAGES >= qwen.STATE_HISTORY_MESSAGES


def test_the_client_history_is_no_longer_truncated_to_the_prompt_window(qwen) -> None:
    """This cap used to be twelve, which silently fixed the memory horizon at
    six turns no matter what the browser sent."""

    payload = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"message {index}"}
        for index in range(60)
    ]
    normalized = qwen.normalize_history_payload(payload)
    assert len(normalized) == qwen.STATE_HISTORY_MESSAGES
    assert normalized[-1]["content"] == "message 59"


def test_junk_history_still_normalizes_to_nothing(qwen) -> None:
    for bad in (None, "history", 7, {}, [1, 2, 3], [{"role": "system", "content": "x"}]):
        assert qwen.normalize_history_payload(bad) == []


# ---------------------------------------------------------------------------
# What reaches the model
# ---------------------------------------------------------------------------

def test_a_preference_outside_the_prompt_window_still_reaches_the_prompt(
    qwen, state_module, directive
) -> None:
    history = _long_session("please always keep answers concise")
    window = history[-qwen.PROMPT_HISTORY_MESSAGES :]
    assert not any("concise" in message["content"] for message in window), (
        "the fixture must place the preference outside the prompt window"
    )

    state = state_module.build_conversation_state(history, current_user_text="how do I list files in python")
    built = directive.build_conversation_directive(state, "auto", "how do I list files in python")
    messages = qwen.compose_chat_messages(
        window,
        "how do I list files in python",
        qwen.resolve_preset_name(built["preset"], "auto"),
        "",
        conversation_contract=built["contract"],
    )

    assert "concise answers" in messages[-1]["content"]
    assert all(
        "concise answers" not in row["content"]
        for row in messages
        if row["role"] == "system"
    )
    assert built["preset"] == "direct"
    assert qwen.resolve_preset_name(built["preset"], "auto") == "direct"


def test_a_constraint_outside_the_prompt_window_reaches_the_prompt(
    qwen, state_module, directive
) -> None:
    history = _long_session("I always deploy with the staging profile first")
    state = state_module.build_conversation_state(history, current_user_text="ship it")
    contract = directive.render_conversation_contract(state, "ship it")
    assert "staging profile" in contract

    messages = qwen.compose_chat_messages(
        history[-qwen.PROMPT_HISTORY_MESSAGES :],
        "ship it",
        "balanced",
        "",
        conversation_contract=contract,
    )
    assert "staging profile" in messages[-1]["content"]
    assert messages[-1]["role"] == "user"
    assert all(
        "staging profile" not in row["content"]
        for row in messages
        if row["role"] == "system"
    )


def test_message_order_puts_the_standing_contract_before_the_current_turn(qwen) -> None:
    messages = qwen.compose_chat_messages(
        [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}],
        "current question",
        "balanced",
        "steer me",
        grounding_context="[S1] evidence",
        prompt_contract_context="per-turn contract",
        conversation_contract="standing contract",
    )
    contents = [row["content"] for row in messages]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].endswith("current question")
    assert "standing contract" in messages[-1]["content"]
    assert contents.index("per-turn contract") > contents.index("Session steering: steer me")
    assert contents.index("per-turn contract") < contents.index("[S1] evidence")
    assert all(
        "standing contract" not in row["content"]
        for row in messages
        if row["role"] == "system"
    )
    assert all(row["role"] in {"system", "user", "assistant"} for row in messages)


def test_without_a_contract_the_message_list_is_what_it_always_was(qwen) -> None:
    history = [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}]
    baseline = qwen.compose_chat_messages(history, "now", "balanced", "")
    assert baseline == [
        {"role": "system", "content": qwen.DEFAULT_SYSTEM_PROMPT},
        {"role": "system", "content": qwen.PRESET_HINTS["balanced"]},
        {"role": "user", "content": "earlier"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "now"},
    ]


# ---------------------------------------------------------------------------
# Preset resolution at the surface
# ---------------------------------------------------------------------------

def test_a_preset_this_surface_does_not_define_is_ignored(qwen) -> None:
    assert qwen.resolve_preset_name("mimo-turbo", "auto") == "balanced"
    assert qwen.resolve_preset_name("", "coding") == "coding"
    assert qwen.resolve_preset_name(None, None) == "balanced"
    assert qwen.resolve_preset_name("direct", "creative") == "direct"


def test_an_explicit_preset_is_never_overridden_by_a_standing_preference(
    qwen, state_module, directive
) -> None:
    state = state_module.build_conversation_state(
        [("be brief", "Okay.")], current_user_text="write me a poem about rain"
    )
    for chosen in qwen.VALID_PRESETS:
        built = directive.build_conversation_directive(state, chosen, "write me a poem about rain")
        assert qwen.resolve_preset_name(built["preset"], chosen) == chosen


class _FakeEncoding(dict):
    """A tokenizer batch is a mapping that also moves to a device."""

    def to(self, _device):
        return self


class _FakeTokenizer:
    """Enough of a tokenizer to reach `model.generate` and back."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.prompt = ""

    def __call__(self, prompt, **_kwargs):
        import torch

        self.prompt = prompt
        return _FakeEncoding({"input_ids": torch.zeros((1, 4), dtype=torch.long)})

    def decode(self, _tokens, **_kwargs):
        return "Use the staging profile."


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs = {}

    def generate(self, **kwargs):
        import torch

        self.kwargs = dict(kwargs)
        return torch.zeros((1, 6), dtype=torch.long)


def _engine(qwen, history):
    import threading

    engine = qwen.Engine.__new__(qwen.Engine)
    engine.lock = threading.Lock()
    engine.inference_lock = threading.Lock()
    engine.session_turn_locks = tuple(
        threading.Lock() for _ in range(qwen.SESSION_TURN_LOCK_STRIPES)
    )
    engine.sessions = {"session": list(history)}
    engine.tokenizer = _FakeTokenizer()
    engine.model = _FakeModel()
    engine.device = "cpu"
    return engine


def test_the_engine_end_to_end_carries_the_session_into_the_prompt(qwen) -> None:
    """The pieces are unit-tested above; this runs the real `chat` body over
    them, because plumbing that is only tested piecewise is not tested."""

    engine = _engine(qwen, _long_session("I always deploy with the staging profile first"))
    payload = engine.chat(
        session_id="session",
        user_text="ship it",
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        preset="auto",
        system_hint="",
        grounding_enabled=False,
    )

    assert payload["ok"] is True
    assert "staging profile" in engine.tokenizer.prompt
    conversation = payload["conversation"]
    assert conversation["active_commitment_count"] >= 1
    assert conversation["directive"]["contract_line_count"] >= 1
    assert conversation["response_audit"]["checked"] is True
    # The turn was appended to a session store deeper than the prompt window.
    assert len(engine.sessions["session"]) > qwen.PROMPT_HISTORY_MESSAGES


def test_the_engine_reports_nothing_when_the_layer_is_disabled(qwen) -> None:
    engine = _engine(qwen, _long_session("I always deploy with the staging profile first"))
    payload = engine.chat(
        session_id="session",
        user_text="ship it",
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
        preset="auto",
        system_hint="",
        grounding_enabled=False,
        conversation_enabled=False,
    )

    assert payload["conversation"] is None
    assert "Conversation memory" not in engine.tokenizer.prompt


def test_a_standing_preference_selects_the_generation_budget_too(qwen) -> None:
    """The preset is not only a hint: it carries the length budget the caller
    left unset."""

    engine = _engine(qwen, _long_session("please always keep answers concise"))
    payload = engine.chat(
        session_id="session",
        user_text="how do I list files in python",
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        preset="auto",
        system_hint="",
        grounding_enabled=False,
    )
    assert payload["preset_used"] == "direct"
    assert engine.model.kwargs["max_new_tokens"] == qwen.PRESET_GENERATION["direct"]["max_new_tokens"]
    assert payload["generation"]["inherited_from_preset"] == {
        "max_new_tokens": True,
        "temperature": True,
        "top_p": True,
    }


def test_turning_the_layer_off_restores_the_previous_prompt_exactly(
    qwen, state_module, directive
) -> None:
    """A controlled evaluation has to be able to remove this entirely."""

    history = _long_session("please always keep answers concise")
    state = state_module.build_conversation_state(history, current_user_text="how do I list files in python")
    off = directive.build_conversation_directive(
        state, "auto", "how do I list files in python", enabled=False
    )
    window = history[-qwen.PROMPT_HISTORY_MESSAGES :]
    with_layer_off = qwen.compose_chat_messages(
        window,
        "how do I list files in python",
        qwen.resolve_preset_name(off["preset"], "auto"),
        "",
        conversation_contract=off["contract"],
    )
    assert with_layer_off == qwen.compose_chat_messages(
        window, "how do I list files in python", "balanced", ""
    )


def test_default_browser_flow_uses_auto_and_omits_generation_overrides(qwen) -> None:
    assert 'data-preset="auto"' in qwen.HTML
    assert 'settings: { preset: "auto"' in qwen.HTML
    assert 'if (state.settings.preset !== "auto")' in qwen.HTML


def test_http_auto_request_preserves_unset_generation_values(qwen) -> None:
    class CapturingEngine:
        def __init__(self) -> None:
            self.kwargs = None

        def status(self):
            return {}

        def chat(self, **kwargs):
            self.kwargs = kwargs
            return {"ok": True, "response": "ok"}

        def clear(self, _session_id):
            return None

    engine = CapturingEngine()
    client = qwen.build_app(engine).test_client()
    response = client.post(
        "/api/chat",
        json={"session_id": "auto-http", "message": "hello", "preset": "auto"},
    )

    assert response.status_code == 200
    assert engine.kwargs["preset"] == "auto"
    assert engine.kwargs["max_new_tokens"] is None
    assert engine.kwargs["temperature"] is None
    assert engine.kwargs["top_p"] is None


def test_same_session_concurrent_turns_do_not_lose_stale_client_history(qwen) -> None:
    import threading

    stale = [
        {"role": "user", "content": "initial"},
        {"role": "assistant", "content": "ack"},
    ]
    engine = _engine(qwen, stale)
    first_entered = threading.Event()
    release_first = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    class BlockingModel(_FakeModel):
        def generate(self, **kwargs):
            nonlocal call_count
            with call_lock:
                call_count += 1
                position = call_count
            if position == 1:
                first_entered.set()
                assert release_first.wait(5), "test did not release first generation"
            return super().generate(**kwargs)

    engine.model = BlockingModel()
    errors = []

    def run(message):
        try:
            engine.chat(
                session_id="session",
                user_text=message,
                max_new_tokens=32,
                temperature=0.0,
                top_p=0.9,
                preset="balanced",
                system_hint="",
                history_override=list(stale),
                grounding_enabled=False,
            )
        except Exception as exc:  # pragma: no cover - assertion reports details
            errors.append(exc)

    first = threading.Thread(target=run, args=("first request",))
    second = threading.Thread(target=run, args=("second request",))
    first.start()
    assert first_entered.wait(5), "first generation did not start"
    second.start()
    release_first.set()
    first.join(10)
    second.join(10)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    stored = engine.sessions["session"]
    assert [row["content"] for row in stored if row["role"] == "user"] == [
        "initial",
        "first request",
        "second request",
    ]


def test_server_retention_is_deeper_than_the_state_horizon(qwen) -> None:
    engine = _engine(qwen, _long_session("oldest", filler_turns=40))
    engine.chat(
        session_id="session",
        user_text="newest",
        max_new_tokens=32,
        temperature=0.0,
        top_p=0.9,
        preset="balanced",
        system_hint="",
        grounding_enabled=False,
    )

    assert len(engine.sessions["session"]) == qwen.MAX_SESSION_MESSAGES
    assert len(engine.sessions["session"]) > qwen.STATE_HISTORY_MESSAGES
    assert engine.sessions["session"][-2]["content"] == "newest"
