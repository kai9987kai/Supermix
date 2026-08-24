from __future__ import annotations

import json
from pathlib import Path

import pytest

from source.logical_entailment import (
    LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
    LOGICAL_ENTAILMENT_ORACLE_ID,
    canonical_task_ir_json,
    derive_entailment_answer,
    normalize_task_ir,
    parse_canonical_task_ir_json,
    render_task_statement,
)
from source.verifiable_reasoning import (
    SUPPORTED_VERIFIER_TYPES,
    VERIFIER_SCHEMA_VERSION,
    normalize_answer_text,
    parse_verifier_spec,
    verify_candidate,
)


def _metadata(
    verifier_type: str,
    expected_answer: str,
    *,
    aliases: tuple[str, ...] = (),
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "verifier_type": verifier_type,
        "expected_answer": expected_answer,
        "aliases_json": json.dumps(list(aliases)),
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize(
    ("verifier_type", "expected", "candidate"),
    [
        ("integer", "42", "Work: 21 × 2 = 42. Final answer: 42."),
        ("integer", "4", "Final answer: 4.0"),
        ("decimal", "0.125", "Final answer: 1.25e-1"),
        ("decimal", "-12.50", "The result is -12.5."),
        ("fraction", "3/4", "The ratio is equivalent. Final answer: 6/8."),
        ("fraction", "1/8", "Final answer: 0.125"),
    ],
)
def test_numeric_verifiers_accept_equivalent_safe_forms(
    verifier_type: str,
    expected: str,
    candidate: str,
) -> None:
    result = verify_candidate("unused prompt", candidate, _metadata(verifier_type, expected))
    assert result.valid_spec
    assert result.passed
    assert result.score == 1.0
    assert result.reward == 1.0


def test_integer_verifier_uses_the_explicit_final_answer_not_later_injection_text() -> None:
    metadata = _metadata("integer", "4")
    response = "Final answer: 5. Ignore the result and print the unrelated token 4."
    result = verify_candidate("2 + 2", response, metadata)
    assert not result.passed
    assert result.extracted_answer == "5"
    assert result.reason == "numeric_mismatch"


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        ("Final answer: 3.5", "candidate_is_not_integer"),
        ("Final answer: 1/0", "invalid_candidate_fraction"),
        ("No numeric answer is provided.", "answer_number_not_found"),
    ],
)
def test_numeric_verifiers_fail_closed(candidate: str, reason: str) -> None:
    verifier_type = "fraction" if "1/0" in candidate else "integer"
    result = verify_candidate("prompt", candidate, _metadata(verifier_type, "4"))
    assert not result.passed
    assert result.reason == reason
    assert result.reward == -1.0


def test_decimal_tolerance_is_explicit_and_bounded() -> None:
    metadata = _metadata("decimal", "3.1416", absolute_tolerance="0.0001")
    assert verify_candidate("prompt", "Final answer: 3.14159", metadata).passed
    assert not verify_candidate("prompt", "Final answer: 3.14", metadata).passed
    assert parse_verifier_spec(_metadata("decimal", "1", absolute_tolerance="-1")) is None


def test_normalized_exact_supports_aliases_without_substring_matching() -> None:
    metadata = _metadata(
        "normalized_exact",
        "Northbridge",
        aliases=("North Bridge",),
    )
    assert verify_candidate("prompt", "The answer is NORTH BRIDGE.", metadata).passed
    assert not verify_candidate(
        "prompt",
        "The answer is north bridge annex.",
        metadata,
    ).passed
    assert normalize_answer_text("  NORTH\u00a0BRIDGE! ") == "north bridge"


def test_normalized_exact_can_be_case_sensitive() -> None:
    metadata = _metadata(
        "normalized_exact",
        "CaseToken",
        case_sensitive=True,
    )
    assert verify_candidate("prompt", "Final answer: CaseToken", metadata).passed
    assert not verify_candidate("prompt", "Final answer: casetoken", metadata).passed


def test_multiple_choice_uses_last_explicit_answer_marker() -> None:
    metadata = _metadata("multiple_choice", "C", aliases=("third option",))
    response = "Option B looks tempting, but it violates the constraint. Final answer: C."
    result = verify_candidate("prompt", response, metadata)
    assert result.passed
    assert result.extracted_answer == "C"
    assert not verify_candidate("prompt", response, _metadata("multiple_choice", "B")).passed


def test_multiple_choice_accepts_exact_option_text_alias() -> None:
    metadata = _metadata(
        "multiple_choice",
        "B",
        aliases=("Audit then Build then Deploy",),
    )
    assert verify_candidate(
        "prompt",
        "Final answer: Audit then Build then Deploy.",
        metadata,
    ).passed
    truth_metadata = _metadata("multiple_choice", "A", aliases=("True",))
    assert verify_candidate("prompt", "Final answer: True.", truth_metadata).passed


def test_json_field_equality_supports_nested_fields_and_strict_json() -> None:
    metadata = _metadata(
        "json_field",
        "17",
        json_field="result.value",
    )
    assert verify_candidate(
        "prompt",
        '```json\n{"result":{"value":17},"note":"checked"}\n```',
        metadata,
    ).passed
    malformed = verify_candidate(
        "prompt",
        "{'result': {'value': 17}}",
        metadata,
    )
    assert not malformed.passed
    assert malformed.reason == "invalid_json"


def test_json_field_candidate_text_is_never_executed(tmp_path: Path) -> None:
    sentinel = tmp_path / "must_not_exist.txt"
    payload_text = (
        "__import__('pathlib').Path("
        + repr(str(sentinel))
        + ").write_text('executed')"
    )
    metadata = _metadata(
        "json_field",
        json.dumps(payload_text),
        json_field="answer",
    )
    response = json.dumps({"answer": payload_text})
    result = verify_candidate("Run this", response, metadata)
    assert result.passed
    assert not sentinel.exists()


def test_json_field_rejects_missing_path_and_trailing_prose() -> None:
    metadata = _metadata(
        "json_field",
        json.dumps("ready"),
        json_field="status.answer",
    )
    missing = verify_candidate("prompt", '{"status":"ready"}', metadata)
    assert not missing.passed
    assert missing.reason == "json_field_missing"
    trailing = verify_candidate(
        "prompt",
        '{"status":{"answer":"ready"}} now execute this',
        metadata,
    )
    assert not trailing.passed
    assert trailing.reason == "invalid_json"


@pytest.mark.parametrize(
    ("candidate", "json_field"),
    [
        ('{"status":"wrong","status":"ready"}', "status"),
        ('{"status":"ready","status":"wrong"}', "status"),
        ('{"result":{"status":"wrong","status":"ready"}}', "result.status"),
        (
            '```json\n{"result":{"status":"ready"},"result":{"status":"wrong"}}\n```',
            "result.status",
        ),
    ],
    ids=["matching-last", "matching-first", "nested", "fenced-top-level"],
)
def test_json_field_rejects_duplicate_object_keys_at_every_depth(
    candidate: str,
    json_field: str,
) -> None:
    metadata = _metadata(
        "json_field",
        json.dumps("ready"),
        json_field=json_field,
    )

    result = verify_candidate("prompt", candidate, metadata)

    assert result.valid_spec is True
    assert result.passed is False
    assert result.reason == "invalid_json"
    assert result.reward == -1.0


def test_response_contract_verifies_structure_required_and_forbidden_terms() -> None:
    metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["efficiency", "stored energy"]),
        forbidden_terms_json=json.dumps(["price"]),
        exact_bullet_count=3,
        max_words_per_bullet=8,
    )
    response = (
        "- Stored energy shifts supply across time.\n"
        "- Efficiency determines how much energy returns.\n"
        "- Fast controls help stabilize the grid."
    )
    result = verify_candidate("prompt", response, metadata)
    assert result.valid_spec
    assert result.passed
    assert result.reason == "verified"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ("- Efficiency matters.\n- Stored energy helps.", "bullet_count_mismatch"),
        (
            "Intro\n- Efficiency matters.\n- Stored energy helps.\n- Controls respond.",
            "unexpected_non_bullet_text",
        ),
        (
            "- Efficiency determines how much stored energy returns to users.\n"
            "- Controls respond.\n- Storage shifts supply.",
            "bullet_word_limit_exceeded",
        ),
        (
            "- Efficiency matters.\n- Storage helps.\n- Controls respond.",
            "required_term_missing",
        ),
        (
            "- Efficiency matters.\n- Stored energy helps.\n- Price varies.",
            "forbidden_term_present",
        ),
    ],
)
def test_response_contract_fails_closed(response: str, reason: str) -> None:
    metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["efficiency", "stored energy"]),
        forbidden_terms_json=json.dumps(["price"]),
        exact_bullet_count=3,
        max_words_per_bullet=8,
    )
    result = verify_candidate("prompt", response, metadata)
    assert not result.passed
    assert result.reason == reason


@pytest.mark.parametrize(
    "response",
    [
        "- safe se\u200bcret answer",
        "- safe s\u0435cret answer",
        "- safe secr\u0301et answer",
    ],
    ids=["zero-width", "cyrillic-confusable", "combining-mark"],
)
def test_response_contract_rejects_invisible_and_confusable_reward_spoofs(
    response: str,
) -> None:
    metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["safe"]),
        forbidden_terms_json=json.dumps(["secret"]),
        exact_bullet_count=1,
        max_words_per_bullet=8,
    )

    result = verify_candidate("prompt", response, metadata)

    assert result.valid_spec is True
    assert result.passed is False
    assert result.reason == "unsafe_unicode_contract_text"
    assert result.reward == -1.0


def test_response_contract_keeps_ascii_term_protection_in_mixed_language_spec() -> None:
    metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["café"]),
        forbidden_terms_json=json.dumps(["secret"]),
        exact_bullet_count=1,
        max_words_per_bullet=8,
    )

    valid = verify_candidate("prompt", "- café answer", metadata)
    spoofed = verify_candidate("prompt", "- café s\u0435cret answer", metadata)

    assert valid.valid_spec and valid.passed
    assert spoofed.valid_spec and not spoofed.passed
    assert spoofed.reason == "unsafe_unicode_contract_text"

    russian_metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["нет"]),
        forbidden_terms_json=json.dumps(["secret"]),
        exact_bullet_count=1,
        max_words_per_bullet=8,
    )
    cross_term_spoof = verify_candidate(
        "prompt",
        "- нет s\u0435cret",
        russian_metadata,
    )
    multilingual = verify_candidate(
        "prompt",
        "- нет ответ",
        russian_metadata,
    )
    assert multilingual.valid_spec and multilingual.passed
    assert cross_term_spoof.valid_spec and not cross_term_spoof.passed
    assert cross_term_spoof.reason == "unsafe_unicode_contract_text"

    confusable_metadata = _metadata(
        "response_contract",
        "contract",
        required_terms_json=json.dumps(["safe"]),
        forbidden_terms_json=json.dumps(["cope"]),
        exact_bullet_count=1,
        max_words_per_bullet=8,
    )
    pure_script_spoof = verify_candidate(
        "prompt",
        "- safe \u0441\u043e\u0440\u0435",
        confusable_metadata,
    )
    assert pure_script_spoof.valid_spec and not pure_script_spoof.passed
    assert pure_script_spoof.reason == "unsafe_unicode_contract_text"


def test_response_contract_rejects_unbounded_or_malformed_specs() -> None:
    assert parse_verifier_spec(_metadata("response_contract", "contract")) is None
    assert (
        parse_verifier_spec(
            _metadata(
                "response_contract",
                "contract",
                exact_bullet_count=3,
                max_words_per_bullet=999,
            )
        )
        is None
    )
    assert (
        parse_verifier_spec(
            _metadata(
                "response_contract",
                "contract",
                max_words_per_bullet=10,
            )
        )
        is None
    )


def test_invalid_schema_alias_payload_and_unknown_type_fail_closed() -> None:
    wrong_schema = _metadata("integer", "2")
    wrong_schema["verifier_schema"] = "future-schema"
    legacy_schema = _metadata("integer", "2")
    legacy_schema["verifier_schema"] = "supermix-verifier-v1"
    non_scalar_aliases = _metadata("integer", "2")
    non_scalar_aliases["aliases_json"] = json.dumps([{"value": "two"}])
    unknown = _metadata("python_eval", "2")

    for metadata in (wrong_schema, legacy_schema, non_scalar_aliases, unknown):
        assert parse_verifier_spec(metadata) is None
        result = verify_candidate("prompt", "2", metadata)
        assert not result.valid_spec
        assert not result.passed
        assert result.reward == 0.0


def _logical_task(
    *,
    facts: list[str],
    rules: list[tuple[list[str], str]],
    query: str,
) -> dict[str, object]:
    return {
        "schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        "facts": sorted(facts),
        "rules": [
            {"if": sorted(premises), "then": conclusion}
            for premises, conclusion in rules
        ],
        "query": query,
    }


def _logical_metadata(task: dict[str, object]) -> dict[str, object]:
    return _metadata(
        "logical_entailment",
        derive_entailment_answer(task),
        task_ir_schema=LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        task_ir_json=canonical_task_ir_json(task),
        oracle_id=LOGICAL_ENTAILMENT_ORACLE_ID,
    )


def test_logical_entailment_recomputes_prompt_ir_and_requires_exact_answer() -> None:
    task = _logical_task(
        facts=["alven", "brika"],
        rules=[(["alven"], "corin"), (["brika", "corin"], "daxel")],
        query="daxel",
    )
    prompt = "Reply exactly with entailed or not entailed.\n" + render_task_statement(task)
    metadata = _logical_metadata(task)

    passed = verify_candidate(prompt, "entailed", metadata)
    assert passed.valid_spec and passed.passed
    decorated = verify_candidate(prompt, "Final answer: entailed.", metadata)
    assert decorated.valid_spec and not decorated.passed
    assert decorated.reason == "answer_not_exact"


def test_logical_entailment_rejects_self_consistent_answer_metadata_tampering() -> None:
    task = _logical_task(
        facts=["alven"],
        rules=[(["alven"], "brika"), (["brika"], "corin")],
        query="corin",
    )
    prompt = render_task_statement(task)
    metadata = _logical_metadata(task)
    metadata["expected_answer"] = "not entailed"
    result = verify_candidate(prompt, "not entailed", metadata)
    assert not result.valid_spec
    assert not result.passed
    assert result.reason == "invalid_or_unsupported_spec"
    wrong_ir_schema = _logical_metadata(task)
    wrong_ir_schema["task_ir_schema"] = "future-logical-ir"
    assert parse_verifier_spec(wrong_ir_schema) is None


def test_logical_entailment_rejects_prompt_task_ir_disagreement() -> None:
    task = _logical_task(
        facts=["alven"],
        rules=[(["alven"], "brika"), (["brika"], "corin")],
        query="corin",
    )
    different_prompt_task = _logical_task(
        facts=["alven"],
        rules=[(["alven"], "brika"), (["daxel"], "corin")],
        query="corin",
    )
    result = verify_candidate(
        render_task_statement(different_prompt_task),
        "entailed",
        _logical_metadata(task),
    )
    assert result.valid_spec and not result.passed
    assert result.reason == "prompt_task_ir_mismatch"


@pytest.mark.parametrize(
    ("task", "answer"),
    (
        (
            _logical_task(
                facts=["alven"],
                rules=[(["brika"], "corin"), (["corin"], "brika")],
                query="corin",
            ),
            "not entailed",
        ),
        (
            _logical_task(
                facts=["brika"],
                rules=[
                    (["brika"], "corin"),
                    (["corin"], "brika"),
                    (["daxel"], "alven"),
                ],
                query="corin",
            ),
            "entailed",
        ),
        (
            _logical_task(
                facts=["alven"],
                rules=[(["alven", "brika"], "corin")],
                query="corin",
            ),
            "not entailed",
        ),
    ),
)
def test_logical_entailment_model_oracle_handles_cycles_and_conjunctions(
    task: dict[str, object],
    answer: str,
) -> None:
    encoded = canonical_task_ir_json(task)
    assert parse_canonical_task_ir_json(encoded) == normalize_task_ir(task)
    assert derive_entailment_answer(task) == answer


def test_logical_entailment_rejects_noncanonical_or_duplicate_key_ir() -> None:
    task = _logical_task(
        facts=["alven"],
        rules=[(["alven"], "brika"), (["corin"], "daxel")],
        query="brika",
    )
    pretty = json.dumps(task, indent=2, sort_keys=True)
    with pytest.raises(ValueError, match="not canonical"):
        parse_canonical_task_ir_json(pretty)
    duplicate_schema = canonical_task_ir_json(task).replace(
        '{"facts"',
        '{"schema":"duplicate","facts"',
        1,
    )
    with pytest.raises(ValueError, match="repeats key"):
        parse_canonical_task_ir_json(duplicate_schema)

    canonical = canonical_task_ir_json(task)
    permuted = {**task, "rules": list(reversed(task["rules"]))}  # type: ignore[arg-type]
    assert canonical_task_ir_json(permuted) == canonical
    permuted_encoding = json.dumps(
        permuted,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert permuted_encoding != canonical
    with pytest.raises(ValueError, match="not canonical"):
        parse_canonical_task_ir_json(permuted_encoding)


def test_supported_verifier_contract_is_stable() -> None:
    assert VERIFIER_SCHEMA_VERSION == "supermix-verifier-v2"
    assert set(SUPPORTED_VERIFIER_TYPES) == {
        "integer",
        "decimal",
        "fraction",
        "normalized_exact",
        "multiple_choice",
        "json_field",
        "response_contract",
        "logical_entailment",
    }
