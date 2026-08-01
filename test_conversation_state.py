from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "conversation_state.py"
RUNTIME_PATH = ROOT / "runtime_python" / "conversation_state.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = _load_module("source_conversation_state_tests", SOURCE_PATH)
runtime = _load_module("runtime_conversation_state_tests", RUNTIME_PATH)


def test_state_is_deterministic_json_safe_and_advisory_only() -> None:
    turns = [
        ("I use Python 3.12 for this project", "Noted."),
        ("Please always keep answers concise", "Understood."),
    ]

    first = source.build_conversation_state(turns)
    second = source.build_conversation_state(turns)

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == "supermix-conversation-state-v1"
    assert first["scope"] == "conversation_context_only"
    assert first["advisory_only"] is True
    assert first["authority"] == {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
        "controls_permissions": False,
    }


def test_durable_commitments_are_extracted_and_classified() -> None:
    state = source.build_conversation_state(
        [
            ("My name is Kai", "Hello Kai."),
            ("I use Python 3.12 for this project", "Noted."),
            ("Please always keep answers concise", "Understood."),
            ("Don't use tabs for indentation", "Understood."),
        ]
    )

    kinds = {row["kind"] for row in state["commitments"] if row["active"]}
    assert "identity" in kinds
    assert "preference" in kinds
    assert "constraint" in kinds
    assert state["style_request"] == "concise"
    assert state["flags"]["has_active_commitments"] is True


def test_version_numbers_survive_sentence_splitting() -> None:
    state = source.build_conversation_state([("I use Python 3.12 for this project", "Noted.")])

    tooling = [row for row in state["commitments"] if row["kind"] == "tooling"]
    assert tooling, "tooling commitment should be extracted"
    assert "3.12" in tooling[0]["text"]


def test_later_statement_supersedes_earlier_and_records_contradiction() -> None:
    state = source.build_conversation_state(
        [
            ("I use Python 3.12 for this project", "Noted."),
            ("Actually I don't use Python 3.12 anymore, I moved to 3.13", "Understood."),
        ]
    )

    by_id = {row["id"]: row for row in state["commitments"]}
    superseded = [row for row in state["commitments"] if not row["active"]]
    assert superseded, "the earlier statement should be superseded"
    earlier = superseded[0]
    assert earlier["superseded_by"]
    assert by_id[earlier["superseded_by"]]["active"] is True

    assert state["contradictions"]
    assert state["flags"]["contradiction_present"] is True


def test_assistant_question_is_tracked_until_the_user_answers_it() -> None:
    unanswered = source.build_conversation_state(
        [("Help me pick a database", "Are you optimising for read throughput or write throughput?")]
    )
    assert unanswered["open_questions"]
    assert unanswered["flags"]["unresolved_open_question"] is True

    answered = source.build_conversation_state(
        [("Help me pick a database", "Are you optimising for read throughput or write throughput?")],
        current_user_text="Read throughput",
    )
    assert answered["open_questions"] == []
    assert answered["answered_now"]
    assert answered["flags"]["answered_open_question"] is True


def test_repeated_clarification_raises_a_loop_flag() -> None:
    state = source.build_conversation_state(
        [
            ("Fix my build", "Which build system are you using?"),
            ("the main one", "Which build system are you using?"),
            ("I said the main one", "Which build system are you using?"),
        ]
    )

    assert state["flags"]["clarification_loop"] is True

    audit = source.audit_response_against_state("Which build system are you using?", state)
    assert audit["checked"] is True
    assert audit["repeats_open_question"] is True
    assert audit["authority"] == "audit_only"


def test_response_audit_respects_a_newer_opposite_style_request() -> None:
    concise = source.build_conversation_state(
        [("Please always keep answers concise", "Understood.")],
        current_user_text="Explain that in detail",
    )
    detailed_response = " ".join(["detail"] * 100)
    assert source.audit_response_against_state(
        detailed_response,
        concise,
        current_user_text="Explain that in detail",
    )["ignores_style_request"] is False

    detailed = source.build_conversation_state(
        [("I prefer detailed answers", "Understood.")],
        current_user_text="Just briefly this time",
    )
    assert source.audit_response_against_state(
        "Short answer.",
        detailed,
        current_user_text="Just briefly this time",
    )["ignores_style_request"] is False


def test_turn_scoped_style_request_does_not_become_standing_memory() -> None:
    state = source.build_conversation_state(
        [
            ("I prefer detailed answers", "Understood."),
            ("Keep it short this time", "Short answer."),
        ],
        current_user_text="Now explain closures",
    )

    assert state["style_request"] == "detailed"
    assert all(
        "this time" not in row["text"].lower()
        for row in state["commitments"]
        if row["active"]
    )

    no_prior_style = source.build_conversation_state(
        [("Keep it short this time", "Short answer.")],
        current_user_text="Now explain closures",
    )
    assert no_prior_style["style_request"] == ""


def test_repetition_is_detected_across_the_whole_conversation() -> None:
    answer = (
        "A closure is a function that captures variables from its enclosing "
        "lexical scope and keeps them alive."
    )
    state = source.build_conversation_state([("What is a closure?", answer)])

    assert source.repetition_score(answer, state) > 0.9
    assert source.repetition_score("Closures matter most for callbacks.", state) < 0.2

    repeat_score = source.score_candidate_for_conversation(answer, state)
    fresh_score = source.score_candidate_for_conversation(
        "Closures matter for callbacks because each keeps its own captured counter.",
        state,
    )
    assert repeat_score < fresh_score


def test_related_turns_share_a_thread_and_a_new_subject_shifts_topic() -> None:
    state = source.build_conversation_state(
        [
            ("How do I index a Postgres table?", "Use CREATE INDEX on the queried column."),
            ("What about partial indexes?", "Partial indexes add a WHERE clause."),
        ]
    )
    assert len(state["threads"]) == 1, "index/indexes should fold into one thread"
    assert state["flags"]["topic_shift"] is False

    shifted = source.build_conversation_state(
        [
            ("How do I index a Postgres table?", "Use CREATE INDEX on the queried column."),
        ],
        current_user_text="What is the capital of Peru?",
    )
    assert shifted["flags"]["topic_shift"] is True


def test_score_is_zero_without_state_and_bounded_with_state() -> None:
    assert source.score_candidate_for_conversation("anything", None) == 0.0
    assert source.score_candidate_for_conversation("anything", {}) == 0.0

    conversations = [
        [("What is a closure?", "A closure captures its enclosing scope and keeps it alive.")],
        [("Please keep answers concise", "Understood.")],
        [("Fix my build", "Which build system are you using?")],
    ]
    candidates = [
        "yes",
        "A closure captures its enclosing scope and keeps it alive.",
        "word " * 400,
        "Which build system are you using?",
    ]
    for turns in conversations:
        state = source.build_conversation_state(turns)
        for candidate in candidates:
            score = source.score_candidate_for_conversation(candidate, state)
            assert abs(score) <= source.MAX_CONVERSATION_SCORE + 1e-9


def test_batch_scoring_matches_single_candidate_scoring_exactly() -> None:
    """Ranking uses the batch path; it must not drift from the single path."""

    state = source.build_conversation_state(
        [
            ("I use Python 3.12 and please keep answers concise", "Noted."),
            ("How do I index a Postgres table?", "Use CREATE INDEX on the queried column."),
            ("What about partial indexes?", "Which query are you optimising?"),
        ],
        current_user_text="the reporting query",
    )
    candidates = [
        "Use CREATE INDEX on the queried column.",
        "A partial index adds a WHERE clause so only matching rows are stored.",
        "Which query are you optimising?",
        "word " * 200,
        "",
        "yes",
    ]

    single = [source.score_candidate_for_conversation(text, state) for text in candidates]
    batch = source.score_candidates_for_conversation(candidates, state)

    assert batch == single
    assert source.score_candidates_for_conversation(candidates, None) == [0.0] * len(candidates)
    assert source.score_candidates_for_conversation([], state) == []
    assert all(abs(value) <= source.MAX_CONVERSATION_SCORE + 1e-9 for value in batch)

    # A verbatim repeat and a re-asked open question must both rank below a
    # fresh, on-topic answer.
    assert batch[0] < batch[1]
    assert batch[2] < batch[1]


def test_degenerate_inputs_produce_an_empty_state() -> None:
    for value in ((), None, "", [], [{}], [("", "")], {"messages": []}, 17):
        state = source.build_conversation_state(value)
        assert state["turn_count"] == 0
        assert state["commitments"] == []
        assert state["flags"]["clarification_loop"] is False
        assert source.score_candidate_for_conversation("anything", state) == 0.0


def test_diagnostics_are_counts_only_and_never_leak_turn_text() -> None:
    state = source.build_conversation_state(
        [
            ("My name is Kai and I use Python 3.12", "Hello Kai."),
            ("Never mention Kubernetes", "Understood."),
        ]
    )
    diagnostics = source.conversation_state_diagnostics(state)
    serialized = json.dumps(diagnostics, sort_keys=True)

    assert json.loads(serialized) == diagnostics
    for secret in ("Kai", "Python", "3.12", "Kubernetes"):
        assert secret not in serialized
    assert diagnostics["commitment_count"] >= 2
    assert diagnostics["authority"]["controls_routes"] is False


def test_brief_summarises_what_the_conversation_established() -> None:
    state = source.build_conversation_state(
        [("Please always keep answers concise", "Understood.")],
        current_user_text="Now explain closures",
    )
    brief = source.render_state_brief(state)

    assert "Established by the user" in brief
    assert "concise" in brief
    assert len(brief) <= 600
    assert source.render_state_brief(None) == ""


def test_a_turn_log_keyed_by_speaker_is_not_dropped_in_silence() -> None:
    """The Studio memory store writes `{"user": ..., "assistant": ...}` rows.

    They matched neither the role-dict branch nor the pair branch, so every turn
    produced an empty `content` and was skipped: the whole history read as an
    empty conversation, with no error anywhere.
    """

    turns = [
        {"user": "I always deploy with the staging profile first", "assistant": "Understood."},
        {"user": "what changed in the last release", "assistant": "Three fixes."},
    ]
    state = source.build_conversation_state(turns, current_user_text="ship it")

    assert state["turn_count"] == 5
    assert state["user_turn_count"] == 3
    assert any(
        "staging profile" in row["text"] for row in state["commitments"] if row["active"]
    )
    # The documented pair form must produce exactly the same state.
    assert state == source.build_conversation_state(
        [(row["user"], row["assistant"]) for row in turns], current_user_text="ship it"
    )


def test_a_style_statement_is_classified_the_same_way_everywhere() -> None:
    assert source.style_preference_of("please keep answers concise") == "concise"
    assert source.style_preference_of("I prefer detailed answers") == "detailed"
    assert source.style_preference_of("I use Python 3.12") == ""
    assert source.style_preference_of(None) == ""


def test_source_and_runtime_contracts_are_exact_mirrors() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    runtime_bytes = RUNTIME_PATH.read_bytes()
    assert source_bytes == runtime_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == hashlib.sha256(runtime_bytes).hexdigest()

    turns = [
        ("I use Python 3.12", "Noted."),
        ("Please keep answers concise", "Which module should I start with?"),
    ]
    assert source.build_conversation_state(turns) == runtime.build_conversation_state(turns)

    state = source.build_conversation_state(turns)
    assert source.conversation_state_diagnostics(state) == runtime.conversation_state_diagnostics(state)
    assert source.score_candidate_for_conversation("Noted.", state) == runtime.score_candidate_for_conversation(
        "Noted.", state
    )
