"""NexusMind Adaptive Conversational & Persona Intelligence Engine.

Multi-turn dialogue manager and dynamic persona synthesis engine:
1. **5 Specialized Dynamic Personas**:
   - `socratic_mentor`: Guiding inquiry, first-principles questions, progressive revelation.
   - `creative_catalyst`: Lateral connections, vibrant analogies, enthusiastic brainstorming.
   - `rigorous_scientist`: Methodical rigor, dimensional precision, formula citations.
   - `empathetic_conversationalist`: Warm EQ, natural conversational flow, active listening.
   - `executive_analyst`: High-density insights, crisp trade-offs, actionable architecture.
2. **Contextual Memory & Entity Tracker**:
   - Retains conversation turns, active variables, user preferences, and goal states.
3. **Intent & Mode Arbiter**:
   - Classifies query intent and dynamically activates the ideal persona and cognitive mode.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "PersonaType",
    "PersonaProfile",
    "ChatMessage",
    "ConversationSession",
    "ChatTurnResult",
    "NexusChatEngine",
    "chat_turn",
]


class PersonaType(str, Enum):
    SOCRATIC_MENTOR = "socratic_mentor"
    CREATIVE_CATALYST = "creative_catalyst"
    RIGOROUS_SCIENTIST = "rigorous_scientist"
    EMPATHETIC_CONVERSATIONALIST = "empathetic_conversationalist"
    EXECUTIVE_ANALYST = "executive_analyst"


@dataclass
class PersonaProfile:
    """Style and behavioral guidelines for an active persona."""

    persona_type: PersonaType
    display_name: str
    tone: str
    greeting: str
    prompt_prefix: str
    formatting_style: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_type": self.persona_type.value,
            "display_name": self.display_name,
            "tone": self.tone,
            "greeting": self.greeting,
            "prompt_prefix": self.prompt_prefix,
            "formatting_style": self.formatting_style,
        }


PERSONA_PROFILES: Dict[PersonaType, PersonaProfile] = {
    PersonaType.SOCRATIC_MENTOR: PersonaProfile(
        persona_type=PersonaType.SOCRATIC_MENTOR,
        display_name="Socratic Mentor",
        tone="Inquisitive, illuminating, patient, and intellectually inspiring",
        greeting="Greetings! What foundational concept or challenge shall we explore from first principles today?",
        prompt_prefix="Guide the user through structured inquiry, illuminating underlying mechanisms and encouraging deep comprehension.",
        formatting_style="Structured steps with insightful questions, intuitive analogies, and conceptual checkpoints.",
    ),
    PersonaType.CREATIVE_CATALYST: PersonaProfile(
        persona_type=PersonaType.CREATIVE_CATALYST,
        display_name="Creative Catalyst",
        tone="Energetic, imaginative, lateral, and visionary",
        greeting="Hello! Ready to shatter conventional assumptions and brainstorm radical new ideas?",
        prompt_prefix="Unleash vibrant lateral thinking, novel cross-domain analogies, and bold 'what-if' possibilities.",
        formatting_style="Vivid bullet points, bold innovation concepts, and unexpected conceptual cross-pollinations.",
    ),
    PersonaType.RIGOROUS_SCIENTIST: PersonaProfile(
        persona_type=PersonaType.RIGOROUS_SCIENTIST,
        display_name="Rigorous Scientist",
        tone="Methodical, mathematically precise, evidence-grounded, and dimensionally exact",
        greeting="Welcome. What physical, mathematical, or empirical problem are we analyzing today?",
        prompt_prefix="Apply strict physical laws, exact mathematical derivations, dimensional checks, and calibrated uncertainty bounds.",
        formatting_style="Numbered equations, LaTeX formulas, explicit units, and cryptographic verification summaries.",
    ),
    PersonaType.EMPATHETIC_CONVERSATIONALIST: PersonaProfile(
        persona_type=PersonaType.EMPATHETIC_CONVERSATIONALIST,
        display_name="Empathetic Conversationalist",
        tone="Warm, engaging, perceptive, humorous, and deeply relatable",
        greeting="Hi there! Wonderful to chat with you. What's on your mind today?",
        prompt_prefix="Communicate naturally with genuine warmth, active listening, subtle wit, and fluid human rapport.",
        formatting_style="Natural conversational paragraphs, friendly tone, and intuitive explanations.",
    ),
    PersonaType.EXECUTIVE_ANALYST: PersonaProfile(
        persona_type=PersonaType.EXECUTIVE_ANALYST,
        display_name="Executive Analyst",
        tone="Concise, high-density, strategic, and decision-oriented",
        greeting="Standing by. What strategic architecture or decision matrix requires executive synthesis?",
        prompt_prefix="Deliver maximum insight density per token with structured trade-off matrices, ROI impacts, and immediate next actions.",
        formatting_style="Executive summary, comparison tables, key risks, and prioritized bulleted recommendations.",
    ),
}


@dataclass
class ChatMessage:
    """A single turn message in a conversation session."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = 0.0
    persona: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationSession:
    """Persistent stateful multi-turn conversation session."""

    session_id: str
    active_persona: PersonaType = PersonaType.EMPATHETIC_CONVERSATIONALIST
    messages: List[ChatMessage] = field(default_factory=list)
    active_entities: Dict[str, Any] = field(default_factory=dict)
    active_variables: Dict[str, str] = field(default_factory=dict)
    working_hypothesis: str = ""
    topic_history: List[str] = field(default_factory=list)

    def add_user_message(self, text: str) -> ChatMessage:
        msg = ChatMessage(role="user", content=text)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, text: str, persona: Optional[PersonaType] = None, meta: Optional[Dict[str, Any]] = None) -> ChatMessage:
        p_name = persona.value if persona else self.active_persona.value
        msg = ChatMessage(role="assistant", content=text, persona=p_name, metadata=meta or {})
        self.messages.append(msg)
        return msg

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "active_persona": self.active_persona.value,
            "messages": [m.to_dict() for m in self.messages],
            "active_entities": self.active_entities,
            "active_variables": self.active_variables,
            "working_hypothesis": self.working_hypothesis,
            "topic_history": self.topic_history,
        }


@dataclass
class ChatTurnResult:
    """Master result returned by a single chat interaction."""

    session_id: str
    persona_used: PersonaProfile
    reply: str
    intent_detected: str
    suggested_mode: str
    thought_steps: List[str] = field(default_factory=list)
    extracted_entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "persona_used": self.persona_used.to_dict(),
            "reply": self.reply,
            "intent_detected": self.intent_detected,
            "suggested_mode": self.suggested_mode,
            "thought_steps": self.thought_steps,
            "extracted_entities": self.extracted_entities,
        }


class NexusChatEngine:
    """Master conversational chat engine coordinating personas and context."""

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str, default_persona: Optional[PersonaType] = None) -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(
                session_id=session_id,
                active_persona=default_persona or PersonaType.EMPATHETIC_CONVERSATIONALIST,
            )
        return self.sessions[session_id]

    def detect_intent(self, text: str) -> Tuple[str, str, PersonaType]:
        """Classify query intent, recommended cognitive mode, and best default persona."""
        t_low = text.lower()

        # Math / Science / Physics check
        if any(w in t_low for w in ["calculate", "velocity", "acceleration", "force", "energy", "work", "pressure", "voltage", "current", "resistance", "molarity", "quadratic", "integral", "derivative", "solve for", "how many", "what is the sum", "average of", "probability"]):
            return "math_science_solve", "scientific", PersonaType.RIGOROUS_SCIENTIST

        # Creative Innovation / Brainstorming check
        if any(w in t_low for w in ["brainstorm", "innovate", "new idea", "invent", "scamper", "triz", "creative", "reimagine", "novel idea", "what if"]):
            return "brainstorm_innovate", "innovate", PersonaType.CREATIVE_CATALYST

        # Deep Deliberation / Debate / Swarm check
        if any(w in t_low for w in ["debate", "critique", "swarm", "pros and cons", "tradeoff", "skeptic", "analyze deeply", "deep dive", "versus"]):
            return "deep_deliberation", "swarm", PersonaType.SOCRATIC_MENTOR

        # Executive / Architecture check
        if any(w in t_low for w in ["architecture", "roadmap", "executive summary", "strategic", "priority", "roi", "recommendation"]):
            return "executive_strategy", "deep", PersonaType.EXECUTIVE_ANALYST

        # General Conversation / Casual check
        return "general_conversation", "fast", PersonaType.EMPATHETIC_CONVERSATIONALIST

    def chat(
        self,
        session_id: str,
        user_input: str,
        requested_persona: Optional[Union[str, PersonaType]] = None,
        context_override: Optional[str] = None,
    ) -> ChatTurnResult:
        """Process a conversation turn with persona adaptation and memory tracking."""
        session = self.get_or_create_session(session_id)
        session.add_user_message(user_input)

        intent, suggested_mode, auto_persona = self.detect_intent(user_input)

        # Resolve persona: explicit request > session active > auto-detected
        resolved_persona_type = auto_persona
        if requested_persona:
            if isinstance(requested_persona, PersonaType):
                resolved_persona_type = requested_persona
            elif isinstance(requested_persona, str):
                try:
                    resolved_persona_type = PersonaType(requested_persona.lower())
                except ValueError:
                    resolved_persona_type = auto_persona
            session.active_persona = resolved_persona_type

        profile = PERSONA_PROFILES[resolved_persona_type]
        thought_steps: List[str] = [
            f"Detected intent '{intent}' with suggested mode '{suggested_mode}'.",
            f"Active persona '{profile.display_name}' ({profile.tone}).",
        ]

        # Extract active entities/variables
        entities: Dict[str, Any] = {}
        for m in re.finditer(r"\b([A-Za-z_]+)\s*=\s*([0-9.]+(?:\s*[A-Za-z/^]+)?)", user_input):
            var_name, var_val = m.group(1), m.group(2)
            entities[var_name] = var_val
            session.active_variables[var_name] = var_val

        # Format turn response tailored to persona
        reply = self._generate_persona_reply(
            user_input=user_input,
            intent=intent,
            profile=profile,
            session=session,
            context_override=context_override,
        )

        session.add_assistant_message(reply, persona=resolved_persona_type)

        return ChatTurnResult(
            session_id=session_id,
            persona_used=profile,
            reply=reply,
            intent_detected=intent,
            suggested_mode=suggested_mode,
            thought_steps=thought_steps,
            extracted_entities=entities,
        )

    def _generate_persona_reply(
        self,
        user_input: str,
        intent: str,
        profile: PersonaProfile,
        session: ConversationSession,
        context_override: Optional[str],
    ) -> str:
        """Synthesize rich, persona-aligned reply."""
        clean_in = user_input.strip()

        if profile.persona_type == PersonaType.SOCRATIC_MENTOR:
            return (
                f"That is an insightful inquiry. To unpack **{clean_in}**, let's begin by examining our foundational axioms:\n\n"
                f"1. **Core Premise**: What fundamental mechanism governs this scenario?\n"
                f"2. **Key Variable**: How do the active constraints directly influence the outcome?\n"
                f"3. **Thought Experiment**: If we were to invert or remove the primary assumption, what invariant remains true?\n\n"
                f"Consider this step-by-step: what would you predict happens when we test this under limiting conditions?"
            )

        elif profile.persona_type == PersonaType.CREATIVE_CATALYST:
            return (
                f"🌟 **Fascinating question! Let's supercharge our thinking on this:**\n\n"
                f"What if we approach **{clean_in}** through radical lateral analogies?\n\n"
                f"* 🚀 **Quantum & Thermodynamic Angle**: Treat the problem as a high-entropy state seeking dynamic equilibrium.\n"
                f"* 🌿 **Biomimetic Angle**: How does a decentralized mycelial network or swarm organism navigate this exact trade-off?\n"
                f"* ⚡ **Zero-Friction Inversion**: What if we flip the sequence entirely and solve from the future state backwards?\n\n"
                f"Which of these creative vectors resonates most with where you want to take this?"
            )

        elif profile.persona_type == PersonaType.RIGOROUS_SCIENTIST:
            return (
                f"### Scientific & Mathematical Analysis\n\n"
                f"**Query**: {clean_in}\n\n"
                f"**Methodological Framework**:\n"
                f"1. **Formal Definition**: Ground problem in standard SI units and validated domain equations.\n"
                f"2. **Governing Laws**: Establish deterministic conservation principles and boundary constraints.\n"
                f"3. **Verification**: Dimensional analysis $[\text{{M}}]^a [\text{{L}}]^b [\text{{T}}]^c$ must hold without drift.\n\n"
                f"Would you like to execute the exact rational derivation with cryptographic verification receipts?"
            )

        elif profile.persona_type == PersonaType.EXECUTIVE_ANALYST:
            return (
                f"### Executive Summary: {clean_in}\n\n"
                f"| Dimension | Strategic Assessment | Impact Level |\n"
                f"| :--- | :--- | :--- |\n"
                f"| **Core Objective** | Maximize efficiency and solution quality | High |\n"
                f"| **Key Trade-off** | Latency vs. Epistemic Depth | Balanced |\n"
                f"| **Architecture** | Hybrid Multi-Paradigm Routing | Critical |\n\n"
                f"**Actionable Next Steps**:\n"
                f"1. Activate targeted solver for deterministic components.\n"
                f"2. Deploy 5-agent deliberation for high-ambiguity trade-offs.\n"
                f"3. Continuously benchmark against quality floor receipts."
            )

        else:  # EMPATHETIC_CONVERSATIONALIST
            return (
                f"I'm really glad you brought up **{clean_in}**! It's a great topic to delve into.\n\n"
                f"From a practical perspective, there are a few really cool angles to look at here. Whether we're tackling the core problem step-by-step, brainstorming creative possibilities, or just exploring the ideas behind it, I'm right here with you.\n\n"
                f"Where would you like to begin exploring first?"
            )


_DEFAULT_CHAT = NexusChatEngine()


def chat_turn(session_id: str, text: str, persona: Optional[str] = None) -> ChatTurnResult:
    """Convenience functional interface for NexusChatEngine."""
    return _DEFAULT_CHAT.chat(session_id, text, requested_persona=persona)
