from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "source" / "grounding_runtime.py"
RUNTIME_PATH = ROOT / "runtime_python" / "grounding_runtime.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = _load_module("source_grounding_runtime_tests", SOURCE_PATH)
runtime = _load_module("runtime_grounding_runtime_tests", RUNTIME_PATH)


def test_plan_is_deterministic_json_safe_and_advisory_only() -> None:
    interaction = {
        "appraisal": {"epistemic_risk": 0.83},
        "response_strategy": "unrelated_external_strategy",
    }
    first = source.plan_grounding(
        "What is the latest documented release? Cite sources.",
        interaction_plan=interaction,
    )
    second = source.plan_grounding(
        "What is the latest documented release? Cite sources.",
        interaction_plan=interaction,
    )

    assert first == second
    assert json.loads(json.dumps(first, sort_keys=True)) == first
    assert first["schema_version"] == "supermix-grounding-v1"
    assert first["scope"] == "grounding_only"
    assert first["evidence_recommended"] is True
    assert first["freshness_required"] is True
    assert first["citation_requested"] is True
    assert first["epistemic_risk"] == 0.83
    assert first["authority"] == {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
        "compute_exit_authority": "unchanged",
    }
    serialized = json.dumps(first, sort_keys=True)
    assert "reasoning_cycles" not in serialized
    assert "agent_mode" not in serialized
    assert "response_strategy" not in serialized


def test_plan_detects_high_stakes_strict_evidence_and_exact_arithmetic() -> None:
    strict = source.plan_grounding(
        "Use only the supplied evidence to answer this medical dosage question."
    )
    arithmetic = source.plan_grounding("What is (1.25 + 2.75) * 3?")

    assert strict["strict_evidence_only"] is True
    assert strict["high_stakes"] is True
    assert "strict_evidence_only" in strict["reasons"]
    assert arithmetic["exact_arithmetic"] == {
        "attempted": True,
        "solved": True,
        "reason": "solved_exactly",
    }
    assert arithmetic["evidence_recommended"] is False


def test_external_query_redacts_private_values_without_leaking_originals() -> None:
    raw = (
        r"Search docs for kai@example.com Bearer abcdefghijklmnop "
        r"api_key=topsecretvalue sk-abcdefghijklmnop "
        r"C:\Users\kai99\secret\notes.txt /home/kai/.ssh/id_rsa "
        r"4111 1111 1111 1111 192.168.1.24"
    )
    redacted = source.redact_external_query(raw)
    payload = json.dumps(redacted, sort_keys=True)

    assert redacted["redaction_count"] >= 7
    assert redacted["safe_to_send"] is True
    assert set(redacted["categories"]) >= {
        "bearer_token",
        "secret",
        "credential",
        "email",
        "credit_card",
        "ip_address",
        "windows_path",
        "home_path",
    }
    for secret in (
        "kai@example.com",
        "abcdefghijklmnop",
        "topsecretvalue",
        r"C:\Users\kai99",
        "/home/kai",
        "4111 1111 1111 1111",
        "192.168.1.24",
    ):
        assert secret not in payload


def test_external_query_is_bounded_and_all_redacted_query_is_not_sendable() -> None:
    bounded = source.redact_external_query("latest " + ("x" * 1000), max_chars=80)
    hidden = source.redact_external_query("kai@example.com")

    assert len(bounded["query"]) <= 80
    assert bounded["truncated"] is True
    assert hidden["query"] == "[REDACTED_EMAIL]"
    assert hidden["safe_to_send"] is False


def _evidence_rows():
    return [
        {
            "title": "Community summary",
            "text": "Paris is the capital of France.",
            "url": "https://community.example/france#fragment",
            "source_type": "forum",
            "score": 0.95,
        },
        {
            "title": "Official France profile",
            "text": "France's capital is Paris. Paris is the national capital.",
            "url": "https://official.example:443/france",
            "source": "Official Example",
            "source_type": "first_party",
            "published_at": "2026-07-01",
            "license": "CC-BY-4.0",
            "score": 0.8,
        },
        {
            "title": "Irrelevant but high score",
            "text": "Saturn has a prominent ring system.",
            "url": "https://science.example/saturn",
            "trust_tier": "primary",
            "score": 1.0,
        },
    ]


def test_evidence_normalization_is_stable_ranked_and_provenance_safe() -> None:
    query = "What is the capital of France?"
    forward = source.normalize_evidence_rows(_evidence_rows(), query=query, max_items=8)
    reverse = source.normalize_evidence_rows(reversed(_evidence_rows()), query=query, max_items=8)

    assert forward == reverse
    assert [item["id"] for item in forward] == ["S1", "S2", "S3"]
    assert [item["rank"] for item in forward] == [1, 2, 3]
    assert forward[0]["title"] == "Official France profile"
    assert forward[0]["trust_tier"] == "official"
    assert forward[0]["url"] == "https://official.example:443/france"
    assert forward[0]["published_at"] == "2026-07-01"
    assert forward[0]["license"] == "CC-BY-4.0"
    assert len(forward[0]["content_hash"]) == 64
    assert all(set(item) >= {
        "id",
        "rank",
        "title",
        "url",
        "text",
        "source",
        "source_type",
        "domain",
        "published_at",
        "trust_tier",
        "license",
        "content_hash",
        "input_score",
        "lexical_overlap",
        "rank_score",
    } for item in forward)
    assert json.loads(json.dumps(forward)) == forward


def test_evidence_normalization_deduplicates_and_rejects_unsafe_urls() -> None:
    rows = [
        {"title": "A", "text": "Same evidence.", "url": "file:///etc/passwd"},
        {"title": "A", "text": "Same evidence.", "url": "file:///etc/passwd"},
        {"title": "B", "text": "Other evidence.", "url": "https://user:pass@example.com/x"},
        {"title": "C", "text": "Port evidence.", "url": "https://example.com:bad/x"},
        {"title": "empty", "text": ""},
        "not a mapping",
    ]
    normalized = source.normalize_evidence_rows(rows, query="evidence", max_items=12)

    assert len(normalized) == 3
    assert all(item["url"] == "" for item in normalized)
    assert len({item["content_hash"] for item in normalized}) == 3


def test_sufficiency_and_lexical_coverage_are_explainable() -> None:
    bundle = source.build_evidence_bundle(
        "What is the capital of France?",
        [{"title": "France", "text": "Paris is the capital of France.", "score": 0.7}],
    )
    diagnostics = bundle["diagnostics"]

    assert diagnostics["evidence_count"] == 1
    assert diagnostics["query_coverage"] == 1.0
    assert diagnostics["best_item_coverage"] == 1.0
    assert diagnostics["sufficiency"] == "sufficient"
    assert diagnostics["sufficient"] is True

    weak = source.build_evidence_bundle(
        "What is the capital of France?",
        [{"title": "Saturn", "text": "Saturn has rings."}],
    )
    assert weak["diagnostics"]["sufficiency"] == "insufficient"


@pytest.mark.parametrize(
    ("left", "right", "kind"),
    [
        (
            "The reported population is 10 million residents.",
            "The reported population is 12 million residents.",
            "numeric",
        ),
        (
            "This medicine is safe for healthy adults.",
            "This medicine is not safe for healthy adults.",
            "polarity",
        ),
    ],
)
def test_conflict_diagnostics_detect_numeric_and_polarity_disagreement(
    left: str,
    right: str,
    kind: str,
) -> None:
    bundle = source.build_evidence_bundle(
        "What does the evidence report?",
        [
            {"title": "One", "text": left, "score": 0.9},
            {"title": "Two", "text": right, "score": 0.9},
        ],
    )

    diagnostics = bundle["diagnostics"]
    assert diagnostics["sufficiency"] == "conflicting"
    assert diagnostics["conflict_count"] >= 1
    assert any(row["kind"] == kind for row in diagnostics["conflicts"])
    assert all(row["source_ids"] == sorted(row["source_ids"]) for row in diagnostics["conflicts"])


def test_distinct_facts_do_not_create_false_conflicts() -> None:
    bundle = source.build_evidence_bundle(
        "Compare the two planets.",
        [
            {"title": "Earth", "text": "Earth has one natural moon."},
            {"title": "Mars", "text": "Mars has two small natural moons."},
        ],
    )
    assert bundle["diagnostics"]["conflict_count"] == 0


def test_citation_validation_accepts_only_existing_canonical_ids() -> None:
    evidence = source.normalize_evidence_rows(
        [
            {"title": "One", "text": "Evidence one."},
            {"title": "Two", "text": "Evidence two."},
        ],
        query="evidence",
    )
    audit = source.validate_citations(
        "Supported [S1], repeated [s1], fabricated [S9], malformed [S01], and [S0].",
        evidence,
    )

    assert audit["citations"] == ["S1", "S9", "S01", "S0"]
    assert audit["valid"] == ["S1"]
    assert audit["invalid"] == ["S9", "S01", "S0"]
    assert audit["all_valid"] is False
    assert audit["uncited_evidence_ids"] == ["S2"]


@pytest.mark.parametrize(
    ("query", "exact", "display"),
    [
        ("What is 2 + 3 * 4?", "14", "14"),
        ("calculate (1.25 + 2.75) * 3", "12", "12"),
        ("compute 1/3 + 2/3", "1", "1"),
        ("evaluate 5 / 6", "5/6", "5/6"),
        ("work out -7 // 3", "-3", "-3"),
        ("2^8", "256", "256"),
        ("7 % 4", "3", "3"),
    ],
)
def test_exact_arithmetic_solver_handles_bounded_expressions(
    query: str,
    exact: str,
    display: str,
) -> None:
    solved = source.solve_exact_arithmetic(query)
    assert solved["attempted"] is True
    assert solved["solved"] is True
    assert solved["reason"] == "solved_exactly"
    assert solved["exact"] == exact
    assert solved["display"] == display
    assert solved["operations"] >= 1


@pytest.mark.parametrize(
    ("query", "attempted", "reason"),
    [
        ("What is __import__('os').system('dir')?", False, "not_explicit_arithmetic"),
        ("open('/etc/passwd').read()", False, "not_explicit_arithmetic"),
        ("print(2 + 2)", False, "not_explicit_arithmetic"),
        ("solve x + 2 = 3", False, "not_explicit_arithmetic"),
        ("Version 3.12 plus build 4", False, "not_explicit_arithmetic"),
        ("2026-07-25", False, "not_explicit_arithmetic"),
        ("2026 / 07 / 25", False, "not_explicit_arithmetic"),
        ("555-123-4567", False, "not_explicit_arithmetic"),
        ("123-45-6789", False, "not_explicit_arithmetic"),
        ("1 / 0", True, "division_by_zero"),
        ("2 ** 1000000", True, "exponent_too_large"),
        ("9 ** (1 / 2)", True, "fractional_exponent_not_supported"),
        ("[1, 2, 3]", False, "not_explicit_arithmetic"),
        ("2 << 4", False, "not_explicit_arithmetic"),
    ],
)
def test_exact_arithmetic_rejects_code_ambiguity_and_resource_attacks(
    query: str,
    attempted: bool,
    reason: str,
) -> None:
    result = source.solve_exact_arithmetic(query)
    assert result["attempted"] is attempted
    assert result["solved"] is False
    assert result["reason"] == reason


def test_exact_arithmetic_is_decimal_exact_not_binary_float() -> None:
    result = source.solve_exact_arithmetic("calculate 0.1 + 0.2")
    assert result["solved"] is True
    assert result["exact"] == "3/10"
    assert result["display"] == "0.3"


def test_finalizer_overrides_only_explicit_arithmetic() -> None:
    finalized = source.finalize_grounded_response(
        "The answer is probably five.",
        "What is 2 + 2?",
    )
    assert finalized["text"] == "The exact result is 4."
    assert finalized["changed"] is True
    assert finalized["reason"] == "explicit_arithmetic_exact"
    assert finalized["authority"] == {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
    }


def test_finalizer_enforces_strict_supplied_evidence_insufficiency() -> None:
    no_evidence = source.finalize_grounded_response(
        "Paris.",
        "Use only the supplied evidence: what is the capital of France?",
        evidence_bundle=[],
    )
    weak_evidence = source.finalize_grounded_response(
        "Paris.",
        "Answer only from the provided sources: what is the capital of France?",
        evidence_bundle=[{"title": "Saturn", "text": "Saturn has rings."}],
    )

    assert no_evidence["changed"] is True
    assert no_evidence["reason"] == "strict_evidence_no_evidence"
    assert "no usable evidence" in no_evidence["text"]
    assert weak_evidence["changed"] is True
    assert weak_evidence["reason"] == "strict_evidence_insufficient"


def test_strict_evidence_constraint_precedes_arithmetic_override() -> None:
    result = source.finalize_grounded_response(
        "Four.",
        "Use only the supplied evidence to answer: what is 2 + 2?",
        evidence_bundle=[],
    )
    assert result["reason"] == "strict_evidence_no_evidence"
    assert result["text"] != "The exact result is 4."


def test_caller_plan_cannot_invent_strict_evidence_override_authority() -> None:
    response = "Keep this ordinary answer unchanged."
    result = source.finalize_grounded_response(
        response,
        "Tell me a short story about a lighthouse.",
        grounding_plan={"strict_evidence_only": True},
        evidence_bundle=[],
    )
    assert result["text"] == response
    assert result["changed"] is False
    assert result["reason"] == "audit_only"


def test_finalizer_is_audit_only_for_supported_or_nonstrict_factual_answers() -> None:
    evidence = [
        {
            "title": "France",
            "text": "Paris is the capital of France.",
            "trust_tier": "primary",
        }
    ]
    supported = source.finalize_grounded_response(
        "Paris is the capital [S1].",
        "Use only the supplied evidence: what is the capital of France?",
        evidence_bundle=evidence,
    )
    unsupported_but_nonstrict = source.finalize_grounded_response(
        "London is the capital of France.",
        "What is the capital of France?",
        evidence_bundle=[],
    )

    assert supported["changed"] is False
    assert supported["reason"] == "audit_only"
    assert supported["citations"]["all_valid"] is True
    assert unsupported_but_nonstrict["changed"] is False
    assert unsupported_but_nonstrict["reason"] == "audit_only"


def test_source_and_runtime_contracts_are_exact_mirrors() -> None:
    source_bytes = SOURCE_PATH.read_bytes()
    runtime_bytes = RUNTIME_PATH.read_bytes()
    assert source_bytes == runtime_bytes
    assert hashlib.sha256(source_bytes).hexdigest() == hashlib.sha256(runtime_bytes).hexdigest()

    query = "What is the latest documented release? Cite [S1]."
    evidence = [{"title": "Release", "text": "Version 3 is current.", "score": 0.8}]
    assert source.plan_grounding(query) == runtime.plan_grounding(query)
    assert source.build_evidence_bundle(query, evidence) == runtime.build_evidence_bundle(query, evidence)
    assert source.finalize_grounded_response(
        "Version 3 [S1].",
        query,
        evidence_bundle=evidence,
    ) == runtime.finalize_grounded_response(
        "Version 3 [S1].",
        query,
        evidence_bundle=evidence,
    )
