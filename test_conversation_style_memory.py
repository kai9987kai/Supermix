"""A stated style preference must survive the turn it was stated on.

`conversation_state` detected a durable "keep answers concise" correctly and then
applied it as a bounded score nudge worth at most 0.036, which was far too small
to reorder anything. Meanwhile the ranker already had a fully weighted `concise`
style mode — `infer_style_mode` simply never saw the conversation, so it
re-inferred the mode from the current turn alone and returned `analyst` for
"how do I list files in python".

The fix routes the standing preference into the mode that already exists rather
than adding weight to the score. These tests pin the three properties that makes
load-bearing:

1. a standing preference reaches the ranker on later turns;
2. a fresh explicit request outranks the standing one; and
3. behaviour is untouched when there is no conversation state, so single-turn
   retrieval is unaffected.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RUNTIME = ROOT / "runtime_python"


def _load(directory: Path):
    saved = list(sys.path)
    for name in ("chat_pipeline", "conversation_state", "run"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(
            "style_memory_chat_pipeline", directory / "chat_pipeline.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state_spec = importlib.util.spec_from_file_location(
            "style_memory_conversation_state", directory / "conversation_state.py"
        )
        state = importlib.util.module_from_spec(state_spec)
        state_spec.loader.exec_module(state)
        return module, state
    finally:
        sys.path[:] = saved


@pytest.fixture(scope="module", params=["source", "runtime_python"])
def pipeline(request):
    directory = SOURCE if request.param == "source" else RUNTIME
    return _load(directory)


STANDING_PHRASINGS = [
    "Please always keep answers concise",
    "keep it short please",
    "I prefer short answers",
    "be brief",
    "be concise",
]


def test_standing_style_preference_reaches_later_turns(pipeline) -> None:
    chat_pipeline, conversation_state = pipeline

    for phrasing in STANDING_PHRASINGS:
        state = conversation_state.build_conversation_state(
            [(phrasing, "Understood.")],
            current_user_text="how do I list files in python",
        )
        assert state["style_request"] == "concise", phrasing
        assert (
            chat_pipeline.infer_style_mode(
                "how do I list files in python", "auto", state
            )
            == "concise"
        ), phrasing


def test_a_fresh_request_for_depth_outranks_the_standing_preference(pipeline) -> None:
    """Otherwise remembering the preference is worse than having no memory."""

    chat_pipeline, conversation_state = pipeline
    state = conversation_state.build_conversation_state(
        [("be brief", "Okay.")], current_user_text="explain that in detail"
    )
    assert state["style_request"] == "concise"

    for query in (
        "explain that in detail",
        "can you elaborate",
        "tell me more",
        "walk me through it step by step",
        "give me a thorough explanation",
    ):
        assert chat_pipeline.infer_style_mode(query, "auto", state) != "concise", query


def test_an_explicit_mode_always_wins(pipeline) -> None:
    chat_pipeline, conversation_state = pipeline
    state = conversation_state.build_conversation_state(
        [("be brief", "Okay.")], current_user_text="anything"
    )
    for mode in ("balanced", "creative", "analyst", "concise"):
        assert chat_pipeline.infer_style_mode("anything", mode, state) == mode


def test_no_conversation_state_leaves_behaviour_bit_identical(pipeline) -> None:
    """Single-turn retrieval must not move; it passes no state."""

    chat_pipeline, _ = pipeline
    queries = [
        "how do I list files in python",
        "what is a docker volume",
        "write me a poem about rain",
        "explain quantum entanglement",
        "summarize this briefly",
        "define serendipity",
        "what does git rebase do",
        "tell me more",
        "",
    ]
    for query in queries:
        assert chat_pipeline.infer_style_mode(query, "auto") == chat_pipeline.infer_style_mode(
            query, "auto", None
        ), query


def test_state_without_a_style_request_changes_nothing(pipeline) -> None:
    chat_pipeline, conversation_state = pipeline
    state = conversation_state.build_conversation_state(
        [("how do I index a table?", "Use CREATE INDEX.")],
        current_user_text="what about partial indexes",
    )
    assert state["style_request"] == ""
    for query in ("how do I list files in python", "what is a docker volume"):
        assert chat_pipeline.infer_style_mode(query, "auto", state) == (
            chat_pipeline.infer_style_mode(query, "auto")
        ), query


def test_malformed_state_is_ignored_rather_than_raising(pipeline) -> None:
    chat_pipeline, _ = pipeline
    baseline = chat_pipeline.infer_style_mode("what is a docker volume", "auto")
    for bad in (None, {}, {"style_request": None}, {"style_request": 123},
                {"style_request": "nonsense"}, [], "concise", 0):
        assert chat_pipeline.infer_style_mode("what is a docker volume", "auto", bad) == baseline


def test_bare_style_directives_are_recorded_as_commitments(pipeline) -> None:
    """"be brief" matches no cue pattern: no "I prefer", no "always", no negation."""

    _, conversation_state = pipeline
    for phrasing in ("be brief", "be concise", "keep it short", "make it short"):
        state = conversation_state.build_conversation_state(
            [(phrasing, "Okay.")], current_user_text="what is a docker volume"
        )
        assert state["style_request"] == "concise", phrasing
        assert state["flags"]["has_active_commitments"], phrasing


def test_detail_request_pattern_has_no_stray_control_characters(pipeline) -> None:
    """This regex shipped once with a literal 0x08 where a word boundary belonged.

    It compiled, imported, and silently matched nothing.
    """

    chat_pipeline, _ = pipeline
    pattern = chat_pipeline.DETAIL_REQUEST_RE.pattern
    assert "\x08" not in pattern
    assert not any(ord(character) < 32 for character in pattern)
    assert chat_pipeline.DETAIL_REQUEST_RE.search("explain that in detail")
    assert not chat_pipeline.DETAIL_REQUEST_RE.search("what is a docker volume")
