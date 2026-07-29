from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_invalid_schema_alias_payload_and_unknown_type_fail_closed() -> None:
    wrong_schema = _metadata("integer", "2")
    wrong_schema["verifier_schema"] = "future-schema"
    non_scalar_aliases = _metadata("integer", "2")
    non_scalar_aliases["aliases_json"] = json.dumps([{"value": "two"}])
    unknown = _metadata("python_eval", "2")

    for metadata in (wrong_schema, non_scalar_aliases, unknown):
        assert parse_verifier_spec(metadata) is None
        result = verify_candidate("prompt", "2", metadata)
        assert not result.valid_spec
        assert not result.passed
        assert result.reward == 0.0


def test_supported_verifier_contract_is_stable() -> None:
    assert VERIFIER_SCHEMA_VERSION == "supermix-verifier-v1"
    assert set(SUPPORTED_VERIFIER_TYPES) == {
        "integer",
        "decimal",
        "fraction",
        "normalized_exact",
        "multiple_choice",
        "json_field",
    }
