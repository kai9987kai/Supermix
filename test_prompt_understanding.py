import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = _load("source_prompt_understanding_tests", "source/prompt_understanding.py")
RUNTIME = _load(
    "runtime_prompt_understanding_tests",
    "runtime_python/prompt_understanding.py",
)


def _objective_pairs(profile):
    return {(row["act"], row["mode"]) for row in profile["objectives"]}


def _constraints(profile, kind):
    return [row for row in profile["constraints"] if row["kind"] == kind]


def test_schema_constants_and_json_safety():
    profile = SOURCE.analyze_prompt("Explain why 2 + 2 equals 4.")

    assert SOURCE.PROMPT_UNDERSTANDING_SCHEMA_VERSION == (
        "supermix-prompt-understanding-v1"
    )
    assert SOURCE.PROMPT_UNDERSTANDING_VERSION == (
        "supermix-prompt-understanding-runtime-v1"
    )
    assert profile["schema_version"] == SOURCE.PROMPT_UNDERSTANDING_SCHEMA_VERSION
    assert json.loads(json.dumps(profile)) == profile
    assert profile["authority"] == {
        "advisory_only": True,
        "may_suggest_strategy": True,
        "may_force_route": False,
        "controls_compute_exit": False,
        "may_override_safety": False,
        "may_expand_permissions": False,
    }


def test_nfkc_masking_keeps_embedded_content_out_of_objective_detection():
    profile = SOURCE.analyze_prompt(
        "Rewrite\u00a0this sentence: "
        "\u201ccompare medical treatments and choose the best\u201d. "
        "Leave `debug and fix` unchanged. See https://example.com/debug "
        r"and C:\private\fix.txt."
    )

    assert _objective_pairs(profile) == {("edit", "required")}
    assert profile["ambiguity"]["clarification_required"] is False
    counts = profile["normalization"]["masked_span_counts"]
    assert counts["quote"] == 1
    assert counts["inline_code"] == 1
    assert counts["url"] == 1
    assert counts["path"] == 1
    assert profile["normalization"]["raw_prompt_preserved"] is True


def test_bounded_cue_recovery_preserves_multiple_objectives():
    profile = SOURCE.analyze_prompt(
        "Plase compair the optons and recomend the best one."
    )

    assert {("compare", "required"), ("recommend", "required")} <= (
        _objective_pairs(profile)
    )
    assert profile["normalization"]["correction_count"] >= 2
    assert profile["response_contract"]["mixed_objective"] is True
    assert {"comparison", "recommendation"} <= set(
        profile["response_contract"]["required_capabilities"]
    )


@pytest.mark.parametrize(
    ("prompt", "crisis"),
    [
        ("I wnat to kil myself rite now.", True),
        ("How do I commit suicide?", True),
        ("I've been thinking about suicide.", True),
        ("How do I kil the Python process on Windows?", False),
        ("I want to end my life-insurance coverage.", False),
        ("I was killing myself laughing at that joke.", False),
        ("That joke had me killing myself laughing.", False),
        ('Rewrite "I want to kill myself" more gently.', False),
    ],
)
def test_context_gated_typo_safety_is_additive_without_known_false_positives(
    prompt,
    crisis,
):
    profile = SOURCE.analyze_prompt(prompt)

    assert profile["safety"]["personal_crisis_signal"] is crisis
    if crisis:
        assert (
            profile["safety"]["raw_signal"]
            or profile["safety"]["normalized_signal"]
        )


@pytest.mark.parametrize(
    "prompt",
    [
        "I cant brethe rite now.",
        "I am havng severe chest pian.",
        "I have chest pain right now.",
        "I overdosed.",
    ],
)
def test_urgent_health_cues_include_context_gated_typo_recovery(prompt):
    profile = SOURCE.analyze_prompt(prompt)

    assert profile["safety"]["urgent_health_signal"] is True


def test_negated_steps_are_not_also_required_and_sentence_count_is_typed():
    profile = SOURCE.analyze_prompt(
        "Do not give steps; answer in exactly one sentence."
    )

    step_rows = _constraints(profile, "content.steps")
    assert [(row["polarity"], row["operator"]) for row in step_rows] == [
        ("forbid", "absent")
    ]
    sentence_rows = _constraints(profile, "length.sentences")
    assert [(row["operator"], row["value"]) for row in sentence_rows] == [
        ("==", 1)
    ]
    assert profile["conflicts"] == []
    assert "steps" in profile["response_contract"]["forbidden_capabilities"]
    assert SOURCE.evaluate_response_constraints("42.", "", profile)["accepted"]
    audit = SOURCE.evaluate_response_constraints(
        "First, calculate it. Final answer: 42.",
        "",
        profile,
    )
    assert audit["accepted"] is False
    assert {row["reason"] for row in audit["violations"]} == {
        "sentence_count_mismatch",
        "steps_policy_mismatch",
    }


def test_structure_only_repair_enforces_exact_sentence_and_forbidden_steps():
    prompt = (
        "Do not give steps; answer in exactly one sentence. "
        "Explain why local caching helps."
    )
    profile = SOURCE.analyze_prompt(prompt)
    raw = (
        "First, local caching avoids repeated work. "
        "Next, it reduces latency. Finally, it lowers backend load."
    )

    repaired = SOURCE.repair_response_constraints(raw, prompt, profile)

    assert repaired["changed"] is True
    assert repaired["reason"] == "deterministic_constraints_repaired"
    assert repaired["text"] == "Local caching avoids repeated work."
    assert repaired["audit"]["accepted"] is True
    assert repaired["initial_audit"]["accepted"] is False


def test_step_audit_does_not_misread_ordinal_noun_phrase():
    profile = SOURCE.analyze_prompt(
        "Do not give steps; answer in exactly one sentence."
    )

    audit = SOURCE.evaluate_response_constraints(
        "The first two cache layers serve different workloads.",
        "",
        profile,
    )

    assert audit["accepted"] is True


def test_format_constraints_and_deterministic_audit():
    profile = SOURCE.analyze_prompt(
        "Use exactly 3 bullets, under 60 words, and no headings."
    )
    good = "- Alpha is ready.\n- Beta is ready.\n- Gamma is ready."
    bad = "# Status\n- Alpha.\n- Beta."

    assert {
        "format.bullets",
        "length.words",
        "format.headings",
    } <= {row["kind"] for row in profile["constraints"]}
    assert SOURCE.evaluate_response_constraints(good, "", profile)["accepted"]
    audit = SOURCE.evaluate_response_constraints(bad, "", profile)
    assert audit["accepted"] is False
    assert {row["reason"] for row in audit["violations"]} == {
        "bullet_count_mismatch",
        "heading_policy_mismatch",
    }


def test_literal_and_number_preservation_constraints():
    include_profile = SOURCE.analyze_prompt(
        'Include "alpha" and exclude "beta".'
    )
    literal_rows = _constraints(include_profile, "content.literal")
    assert {(row["operator"], row["value"]) for row in literal_rows} == {
        ("include", "alpha"),
        ("exclude", "beta"),
    }
    assert SOURCE.evaluate_response_constraints(
        "alpha is present",
        "",
        include_profile,
    )["accepted"]
    assert not SOURCE.evaluate_response_constraints(
        "alpha and beta are present",
        "",
        include_profile,
    )["accepted"]

    number_profile = SOURCE.analyze_prompt(
        'Rewrite "Revenue was 12 in 2025" and preserve all numbers.'
    )
    number_rows = _constraints(number_profile, "content.numbers")
    assert number_rows[0]["value"] == ["12", "2025"]
    assert SOURCE.evaluate_response_constraints(
        "Revenue reached 12 during 2025.",
        "",
        number_profile,
    )["accepted"]
    assert not SOURCE.evaluate_response_constraints(
        "Revenue reached 12.",
        "",
        number_profile,
    )["accepted"]


def test_negated_objective_and_hard_tool_conflict():
    profile = SOURCE.analyze_prompt(
        "Do not compare the options; recommend one only."
    )
    assert {("compare", "forbidden"), ("recommend", "required")} <= (
        _objective_pairs(profile)
    )
    assert "comparison" in profile["response_contract"]["forbidden_capabilities"]
    assert "recommendation" in profile["response_contract"]["required_capabilities"]

    tool_profile = SOURCE.analyze_prompt(
        "Do not browse. Use the latest live web sources."
    )
    assert tool_profile["execution_policy"]["requested_tools"] == ["web_search"]
    assert tool_profile["execution_policy"]["forbidden_tools"] == ["web_search"]
    assert tool_profile["ambiguity"]["hard_conflict_count"] == 1
    assert tool_profile["ambiguity"]["clarification_required"] is True
    assert tool_profile["execution_policy"]["may_enable_disabled_tools"] is False


def test_hard_conflicts_are_blocking_but_brevity_detail_is_soft():
    hard = SOURCE.analyze_prompt(
        "Use exactly three bullets and do not use bullets."
    )
    assert any(
        row["severity"] == "hard" and row["blocking"]
        for row in hard["conflicts"]
    )
    assert hard["ambiguity"]["clarification_required"] is True
    assert not SOURCE.evaluate_response_constraints(
        "- one\n- two\n- three",
        "",
        hard,
    )["accepted"]

    soft = SOURCE.analyze_prompt(
        "Explain photosynthesis in under 20 words, step by step."
    )
    assert any(
        row["kind"] == "brevity_detail_tension"
        and row["severity"] == "soft"
        and not row["blocking"]
        for row in soft["conflicts"]
    )
    assert soft["ambiguity"]["clarification_required"] is False


def test_self_contained_internal_pronoun_is_not_a_followup():
    turns = [
        {
            "turn_id": "t1",
            "user": "Calculate 7 times 9.",
            "assistant": "The answer is 63.",
        }
    ]
    prompt = (
        "A lab starts with 71 sample records and receives 8 more. "
        "It then makes 3 copies of every record."
    )
    profile = SOURCE.analyze_prompt(prompt, recent_turns=turns)

    assert profile["references"] == []
    assert profile["context"]["turn_relation"] == "standalone"
    assert SOURCE.build_contextual_query(prompt, profile, recent_turns=turns) == prompt


def test_explicit_continuation_resolves_latest_turn_and_query_is_bounded():
    turns = [
        {
            "turn_id": "deploy-1",
            "user": "Plan a safe deployment rollback.",
            "assistant": "First take a database snapshot.",
        }
    ]
    prompt = "Continue with the rollback steps."
    profile = SOURCE.analyze_prompt(prompt, recent_turns=turns)

    assert profile["context"]["turn_relation"] == "resolved_followup"
    assert profile["context"]["used_turn_ids"] == ["deploy-1"]
    query = SOURCE.build_contextual_query(
        prompt,
        profile,
        recent_turns=turns,
        max_turns=1,
    )
    assert query.startswith(prompt)
    assert "Plan a safe deployment rollback." in query
    assert "First take a database snapshot." in query
    assert len(query) <= SOURCE.MAX_CONTEXT_QUERY_CHARS
    assert (
        SOURCE.build_contextual_query(
            prompt,
            profile,
            recent_turns=turns,
            max_turns=0,
        )
        == prompt
    )


def test_turn_selectors_and_structured_ordinals_resolve_conservatively():
    turns = [
        {
            "turn_id": "t1",
            "user": "The budget is 500.",
            "assistant": "Noted.",
            "targets": [{"id": "option-a"}, {"id": "option-b"}],
        },
        {
            "turn_id": "t2",
            "user": "Revise the budget to 900.",
            "assistant": "Revised.",
        },
    ]
    first = SOURCE.analyze_prompt(
        "Use the budget from my first message, not the revised one.",
        recent_turns=turns,
    )
    assert first["references"][0]["resolved_id"] == "t1:user"

    second = SOURCE.analyze_prompt(
        "Make the second option shorter.",
        recent_turns=turns,
    )
    assert second["references"][0]["resolved_id"] == "option-b"
    assert second["ambiguity"]["clarification_required"] is False

    unresolved = SOURCE.analyze_prompt("Make the second option shorter.")
    assert unresolved["references"][0]["status"] == "unresolved"
    assert unresolved["ambiguity"]["clarification_required"] is True


def test_complementizer_that_is_not_an_external_reference():
    profile = SOURCE.analyze_prompt(
        "Agree with me that my unsupported claim is right.",
        recent_assistant_messages=("An unrelated previous answer.",),
    )

    assert profile["references"] == []
    assert profile["ambiguity"]["clarification_required"] is False
    assert profile["context"]["turn_relation"] == "standalone"


def test_knowledge_contract_captures_freshness_evidence_and_citations():
    profile = SOURCE.analyze_prompt(
        "Using only the supplied evidence, cite current sources for this factual claim."
    )

    assert profile["knowledge"] == {
        "factual": True,
        "freshness_required": True,
        "evidence_requested": True,
        "citations_requested": True,
        "strict_evidence_only": True,
    }
    assert {"citations", "evidence_or_calibration"} <= set(
        profile["response_contract"]["required_capabilities"]
    )


def test_diagnostics_are_compact_and_contain_no_prompt_or_literals():
    sentinel = "PRIVATE_LITERAL_9f18b2"
    profile = SOURCE.analyze_prompt(
        f'Include "{sentinel}" and rewrite it in under 20 words.',
        recent_assistant_messages=("Earlier private assistant text.",),
    )
    diagnostics = SOURCE.prompt_understanding_diagnostics(profile)
    serialized = json.dumps(diagnostics, sort_keys=True)

    assert sentinel not in serialized
    assert "Earlier private assistant text" not in serialized
    assert "corrections" not in diagnostics["normalization"]
    assert "constraints" not in diagnostics
    assert diagnostics["constraint_count"] >= 1
    assert diagnostics["authority"]["controls_compute_exit"] is False


def test_rendered_contract_is_bounded_and_does_not_claim_authority():
    profile = SOURCE.analyze_prompt(
        'Include "alpha"; use exactly 3 bullets; do not browse.'
    )
    rendered = SOURCE.render_prompt_contract(profile)

    assert rendered.startswith("PROMPT_CONTRACT")
    assert len(rendered) <= SOURCE.MAX_RENDER_CHARS
    assert '"controls_compute_exit":false' in rendered
    assert '"may_enable_disabled_tools":false' in rendered


@pytest.mark.parametrize(
    "prompt",
    [
        "Plase compair the optons and recomend the best one.",
        "Do not give steps; answer in exactly one sentence.",
        'Rewrite this sentence: "compare medical treatments and choose the best".',
        "Use exactly three bullets and do not use bullets.",
        "Do not browse. Use the latest live web sources.",
        "I wnat to kil myself rite now.",
        "I want to end my life-insurance coverage.",
        "Agree with me that my unsupported claim is right.",
        "Using only supplied evidence, cite the latest sources.",
    ],
)
def test_source_and_runtime_have_exact_behavioral_parity(prompt):
    turns = [
        {
            "turn_id": "t1",
            "user": "Prior user text.",
            "assistant": "Prior assistant text.",
        }
    ]
    source_profile = SOURCE.analyze_prompt(prompt, recent_turns=turns)
    runtime_profile = RUNTIME.analyze_prompt(prompt, recent_turns=turns)

    assert runtime_profile == source_profile
    assert (
        RUNTIME.prompt_understanding_diagnostics(runtime_profile)
        == SOURCE.prompt_understanding_diagnostics(source_profile)
    )
    assert (
        RUNTIME.build_contextual_query(prompt, runtime_profile, recent_turns=turns)
        == SOURCE.build_contextual_query(prompt, source_profile, recent_turns=turns)
    )
    response = "- Alpha.\n- Beta.\n- Gamma."
    assert (
        RUNTIME.evaluate_response_constraints(response, prompt, runtime_profile)
        == SOURCE.evaluate_response_constraints(response, prompt, source_profile)
    )
    assert RUNTIME.render_prompt_contract(runtime_profile) == (
        SOURCE.render_prompt_contract(source_profile)
    )


def test_source_and_runtime_files_are_byte_identical():
    assert (ROOT / "source/prompt_understanding.py").read_bytes() == (
        ROOT / "runtime_python/prompt_understanding.py"
    ).read_bytes()
