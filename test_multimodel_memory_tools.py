from pathlib import Path

from source.multimodel_memory import ConversationMemoryStore
from source.multimodel_tools import parse_tool_calls, should_offer_web_search, strip_tool_calls


def test_memory_store_extracts_facts_and_examples(tmp_path: Path) -> None:
    store = ConversationMemoryStore(tmp_path / "memory")
    session_id = "session-a"
    store.update(
        session_id=session_id,
        user_text="My name is Kai and I prefer concise answers. I am working on a multimodel desktop app.",
        assistant_text="Understood. I will keep answers compact and focused on the desktop app.",
        model_key="v33_final",
        route_reason="initial",
    )
    store.update(
        session_id=session_id,
        user_text="Debug the memory routing bug in the desktop app.",
        assistant_text="Start by checking where the session state is persisted and reloaded.",
        model_key="v33_final",
        route_reason="follow-up",
    )

    bundle = store.build_context(session_id, "Please help with the desktop app memory bug.")
    assert any("User name:" in item or "Preferred" in item for item in bundle["memory_notes"])
    assert bundle["example_count"] >= 1
    assert "Relevant prior conversation examples" in bundle["context_block"]


def test_tool_call_parsing_and_stripping() -> None:
    text = "TOOL:web_search: latest OpenAI model docs\nThen summarize the result."
    assert parse_tool_calls(text) == ["latest OpenAI model docs"]
    assert strip_tool_calls(text) == "Then summarize the result."
    assert should_offer_web_search("What is the latest OpenAI models page?")
