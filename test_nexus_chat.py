"""Tests for the NexusChatEngine — adaptive multi-turn conversational intelligence."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
for p in (ROOT, SOURCE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nexus_chat as nc


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

def test_chat_returns_result():
    result = nc.chat_turn("test_sess_1", "Hello, how are you?")
    assert isinstance(result, nc.ChatTurnResult)
    assert result.reply
    assert result.session_id == "test_sess_1"


def test_reply_is_non_empty():
    result = nc.chat_turn("test_sess_2", "What is the capital of France?")
    assert isinstance(result.reply, str)
    assert len(result.reply) > 10


def test_persona_profile_returned():
    result = nc.chat_turn("test_sess_3", "Hello!")
    assert isinstance(result.persona_used, nc.PersonaProfile)
    assert result.persona_used.display_name
    assert result.persona_used.tone


# ---------------------------------------------------------------------------
# Persona Selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("persona_key,expected_type", [
    ("socratic_mentor", nc.PersonaType.SOCRATIC_MENTOR),
    ("creative_catalyst", nc.PersonaType.CREATIVE_CATALYST),
    ("rigorous_scientist", nc.PersonaType.RIGOROUS_SCIENTIST),
    ("empathetic_conversationalist", nc.PersonaType.EMPATHETIC_CONVERSATIONALIST),
    ("executive_analyst", nc.PersonaType.EXECUTIVE_ANALYST),
])
def test_explicit_persona_selection(persona_key, expected_type):
    result = nc.chat_turn(f"sess_persona_{persona_key}", "Tell me something", persona=persona_key)
    assert result.persona_used.persona_type == expected_type


def test_unknown_persona_falls_back_gracefully():
    result = nc.chat_turn("sess_unknown_p", "Hi", persona="nonexistent_persona")
    # Should not raise; should fall back to an auto-detected persona
    assert result.reply


# ---------------------------------------------------------------------------
# Intent Detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message,expected_intent", [
    ("calculate the kinetic energy of mass 5 kg at velocity 10 m/s", "math_science_solve"),
    ("brainstorm new ideas for decentralized AI", "brainstorm_innovate"),
    ("make an executive summary of our cloud migration strategy", "executive_strategy"),
    ("Hello! What did you do today?", "general_conversation"),
])
def test_intent_detection(message, expected_intent):
    engine = nc.NexusChatEngine()
    intent, mode, persona = engine.detect_intent(message)
    assert intent == expected_intent


# ---------------------------------------------------------------------------
# Auto-Persona Routing from Intent
# ---------------------------------------------------------------------------

def test_science_query_routes_to_scientist():
    engine = nc.NexusChatEngine()
    _, _, auto_persona = engine.detect_intent("solve for the acceleration with force 100 N and mass 10 kg")
    assert auto_persona == nc.PersonaType.RIGOROUS_SCIENTIST


def test_brainstorm_query_routes_to_catalyst():
    engine = nc.NexusChatEngine()
    _, _, auto_persona = engine.detect_intent("brainstorm creative innovations for autonomous drones")
    assert auto_persona == nc.PersonaType.CREATIVE_CATALYST


def test_debate_query_routes_to_mentor():
    engine = nc.NexusChatEngine()
    _, _, auto_persona = engine.detect_intent("debate the pros and cons of centralized vs distributed computing")
    assert auto_persona == nc.PersonaType.SOCRATIC_MENTOR


# ---------------------------------------------------------------------------
# Multi-Turn Session Memory
# ---------------------------------------------------------------------------

def test_session_persists_messages():
    engine = nc.NexusChatEngine()
    session_id = "memory_test_session"
    engine.chat(session_id, "My name is Alice.")
    engine.chat(session_id, "How can I learn quantum mechanics?")
    sess = engine.sessions[session_id]
    assert len(sess.messages) >= 4  # 2 user + 2 assistant


def test_different_sessions_are_isolated():
    engine = nc.NexusChatEngine()
    engine.chat("sess_a", "Hello from A")
    engine.chat("sess_b", "Hello from B")
    assert "sess_a" in engine.sessions
    assert "sess_b" in engine.sessions
    assert len(engine.sessions["sess_a"].messages) == 2
    assert len(engine.sessions["sess_b"].messages) == 2


def test_session_tracks_active_variables():
    engine = nc.NexusChatEngine()
    engine.chat("var_sess", "The mass m = 10 kg and velocity v = 5 m/s")
    sess = engine.sessions["var_sess"]
    assert "m" in sess.active_variables or "v" in sess.active_variables


# ---------------------------------------------------------------------------
# Persona Profiles Catalog
# ---------------------------------------------------------------------------

def test_all_persona_profiles_defined():
    for pt in nc.PersonaType:
        assert pt in nc.PERSONA_PROFILES
        profile = nc.PERSONA_PROFILES[pt]
        assert profile.display_name
        assert profile.tone
        assert profile.greeting
        assert profile.prompt_prefix


def test_persona_profiles_serializable():
    for pt, profile in nc.PERSONA_PROFILES.items():
        d = profile.to_dict()
        assert "persona_type" in d
        assert "display_name" in d
        assert "tone" in d


# ---------------------------------------------------------------------------
# ChatMessage & ConversationSession
# ---------------------------------------------------------------------------

def test_conversation_session_to_dict():
    sess = nc.ConversationSession(session_id="test123")
    sess.add_user_message("Test message")
    d = sess.to_dict()
    assert d["session_id"] == "test123"
    assert len(d["messages"]) == 1


def test_chat_message_roles():
    sess = nc.ConversationSession(session_id="roles_test")
    sess.add_user_message("User input")
    sess.add_assistant_message("Assistant reply")
    assert sess.messages[0].role == "user"
    assert sess.messages[1].role == "assistant"


# ---------------------------------------------------------------------------
# ChatTurnResult serialization
# ---------------------------------------------------------------------------

def test_chat_turn_result_to_dict():
    import json
    result = nc.chat_turn("serial_sess", "How do I center a div in CSS?")
    d = result.to_dict()
    assert "reply" in d
    assert "persona_used" in d
    assert "intent_detected" in d
    json_str = json.dumps(d)
    assert len(json_str) > 50
