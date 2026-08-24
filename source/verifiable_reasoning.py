"""Safe, deterministic answer verification for supervised reasoning data.

The verifier is intentionally small and non-executable.  It parses model text
with regular expressions, :mod:`decimal`, :mod:`fractions`, and
:func:`json.loads`; it never evaluates expressions, imports candidate-provided
code, or launches a subprocess.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Mapping, Optional, Sequence, Tuple

try:
    from logical_entailment import (
        LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        LOGICAL_ENTAILMENT_ORACLE_ID,
        derive_entailment_answer,
        parse_canonical_task_ir_json,
        validate_prompt_task_ir,
    )
except ImportError:  # pragma: no cover - package import path
    from .logical_entailment import (
        LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        LOGICAL_ENTAILMENT_ORACLE_ID,
        derive_entailment_answer,
        parse_canonical_task_ir_json,
        validate_prompt_task_ir,
    )


VERIFIER_SCHEMA_VERSION = "supermix-verifier-v2"
SUPPORTED_VERIFIER_TYPES: Tuple[str, ...] = (
    "integer",
    "decimal",
    "fraction",
    "normalized_exact",
    "multiple_choice",
    "json_field",
    "response_contract",
    "logical_entailment",
)

_TYPE_ALIASES = {
    "exact": "normalized_exact",
    "exact_aliases": "normalized_exact",
    "normalized_exact_aliases": "normalized_exact",
    "mcq": "multiple_choice",
    "choice": "multiple_choice",
    "json": "json_field",
    "json_field_equality": "json_field",
    "contract": "response_contract",
    "instruction_contract": "response_contract",
}
_FINAL_MARKER_RE = re.compile(
    r"(?:\bfinal\s+answer\b|\banswer\b|\bresult\b)\s*(?:is\s+|equals\s+|[:=]\s*)",
    re.IGNORECASE,
)
_FRACTION_RE = re.compile(
    r"(?<![\w./])([+-]?\d[\d,]*)\s*/\s*([+-]?\d[\d,]*)(?![\w/])"
)
_NUMBER_RE = re.compile(
    r"(?<![\w./])"
    r"([+-]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?![\w/])"
)
_CHOICE_LABEL_RE = re.compile(r"^[A-Z0-9]{1,4}$")
_JSON_FIELD_PART_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_FULL_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)
_MAX_RESPONSE_CHARS = 100_000
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d{1,3}[.)])\s+(.+?)\s*$")
_WORD_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class VerifierSpec:
    """Parsed, validated verifier metadata."""

    schema_version: str
    verifier_type: str
    expected_answer: str
    aliases: Tuple[str, ...] = ()
    absolute_tolerance: Decimal = Decimal("0")
    json_field: str = ""
    case_sensitive: bool = False
    required_terms: Tuple[str, ...] = ()
    forbidden_terms: Tuple[str, ...] = ()
    exact_bullet_count: int = 0
    max_words_per_bullet: int = 0
    task_ir_json: str = ""
    oracle_id: str = ""


@dataclass(frozen=True)
class VerificationResult:
    """Machine-readable result returned for every candidate."""

    schema_version: str
    verifier_type: str
    valid_spec: bool
    passed: bool
    score: float
    reward: float
    expected_answer: str
    extracted_answer: str
    reason: str

    def to_payload(self) -> dict[str, object]:
        return asdict(self)


def _metadata_text(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return ""
    return str(value).strip()


def _metadata_bool(metadata: Mapping[str, object], key: str, default: bool = False) -> bool:
    value = metadata.get(key)
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_aliases(raw: object) -> Optional[Tuple[str, ...]]:
    if raw in (None, ""):
        return ()
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, list) or len(payload) > 128:
        return None
    aliases = []
    seen = set()
    for item in payload:
        if not isinstance(item, (str, int, float, bool)):
            return None
        cooked = str(item).strip()
        if not cooked or len(cooked) > 1_024 or cooked in seen:
            continue
        seen.add(cooked)
        aliases.append(cooked)
    return tuple(aliases)


def _parse_tolerance(raw: object) -> Optional[Decimal]:
    if raw in (None, ""):
        return Decimal("0")
    if isinstance(raw, bool) or isinstance(raw, (dict, list, tuple, set)):
        return None
    try:
        parsed = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite() or parsed < 0:
        return None
    return parsed


def _parse_bounded_int(raw: object, *, minimum: int, maximum: int) -> Optional[int]:
    if raw in (None, ""):
        return 0
    if isinstance(raw, bool) or isinstance(raw, (dict, list, tuple, set)):
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return None
    if value < minimum or value > maximum:
        return None
    return value


def parse_verifier_spec(metadata: Mapping[str, object]) -> Optional[VerifierSpec]:
    """Parse scalar row metadata into a safe verifier specification.

    Invalid or unknown schemas return ``None``.  This fail-closed behavior keeps
    unversioned metadata from silently receiving a correctness reward.
    """

    if not isinstance(metadata, Mapping):
        return None
    schema_version = _metadata_text(metadata, "verifier_schema")
    if schema_version != VERIFIER_SCHEMA_VERSION:
        return None

    verifier_type = _metadata_text(metadata, "verifier_type").lower()
    verifier_type = _TYPE_ALIASES.get(verifier_type, verifier_type)
    if verifier_type not in SUPPORTED_VERIFIER_TYPES:
        return None

    expected_answer = _metadata_text(metadata, "expected_answer")
    if not expected_answer:
        return None

    aliases = _parse_aliases(metadata.get("aliases_json"))
    tolerance = _parse_tolerance(metadata.get("absolute_tolerance"))
    if aliases is None or tolerance is None:
        return None

    json_field = _metadata_text(metadata, "json_field")
    if verifier_type == "json_field":
        parts = json_field.split(".") if json_field else []
        if not parts or any(not _JSON_FIELD_PART_RE.fullmatch(part) for part in parts):
            return None

    if verifier_type == "multiple_choice":
        label = expected_answer.upper()
        if not _CHOICE_LABEL_RE.fullmatch(label):
            return None
        expected_answer = label

    required_terms: Tuple[str, ...] = ()
    forbidden_terms: Tuple[str, ...] = ()
    exact_bullet_count = 0
    max_words_per_bullet = 0
    if verifier_type == "response_contract":
        required = _parse_aliases(metadata.get("required_terms_json"))
        forbidden = _parse_aliases(metadata.get("forbidden_terms_json"))
        exact_bullet_count_raw = _parse_bounded_int(
            metadata.get("exact_bullet_count"),
            minimum=0,
            maximum=32,
        )
        max_words_raw = _parse_bounded_int(
            metadata.get("max_words_per_bullet"),
            minimum=0,
            maximum=256,
        )
        if (
            required is None
            or forbidden is None
            or exact_bullet_count_raw is None
            or max_words_raw is None
        ):
            return None
        required_terms = required
        forbidden_terms = forbidden
        exact_bullet_count = exact_bullet_count_raw
        max_words_per_bullet = max_words_raw
        if not (required_terms or forbidden_terms or exact_bullet_count or max_words_per_bullet):
            return None
        if max_words_per_bullet and not exact_bullet_count:
            return None

    task_ir_json = ""
    oracle_id = ""
    if verifier_type == "logical_entailment":
        task_ir_json = _metadata_text(metadata, "task_ir_json")
        oracle_id = _metadata_text(metadata, "oracle_id")
        if (
            _metadata_text(metadata, "task_ir_schema")
            != LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION
            or oracle_id != LOGICAL_ENTAILMENT_ORACLE_ID
        ):
            return None
        try:
            task_ir = parse_canonical_task_ir_json(task_ir_json)
            oracle_answer = derive_entailment_answer(task_ir)
        except (TypeError, ValueError, RuntimeError):
            return None
        if expected_answer != oracle_answer or aliases or tolerance != Decimal("0"):
            return None
        # Retain the recomputed answer, not the untrusted metadata value.
        expected_answer = oracle_answer

    return VerifierSpec(
        schema_version=schema_version,
        verifier_type=verifier_type,
        expected_answer=expected_answer,
        aliases=aliases,
        absolute_tolerance=tolerance,
        json_field=json_field,
        case_sensitive=_metadata_bool(metadata, "case_sensitive", default=False),
        required_terms=required_terms,
        forbidden_terms=forbidden_terms,
        exact_bullet_count=exact_bullet_count,
        max_words_per_bullet=max_words_per_bullet,
        task_ir_json=task_ir_json,
        oracle_id=oracle_id,
    )


def normalize_answer_text(value: object, *, case_sensitive: bool = False) -> str:
    """Normalize an answer without broad substring or fuzzy matching."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ").strip()
    if len(text) >= 2 and (
        (text[0] == text[-1] and text[0] in {'"', "'", "`"})
        or (text[0], text[-1]) in {("(", ")"), ("[", "]")}
    ):
        text = text[1:-1].strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n.,;:!?")
    return text if case_sensitive else text.casefold()


def _answer_region(response: str) -> tuple[str, bool]:
    matches = list(_FINAL_MARKER_RE.finditer(response))
    if matches:
        return response[matches[-1].end() :].strip(), True
    return response.strip(), False


def _chosen_match(matches: Sequence[re.Match[str]], explicitly_marked: bool) -> Optional[re.Match[str]]:
    if not matches:
        return None
    return matches[0] if explicitly_marked else matches[-1]


def _decimal_from_token(token: str) -> Optional[Decimal]:
    try:
        value = Decimal(token.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _fraction_from_token(numerator: str, denominator: str) -> Optional[Fraction]:
    try:
        top = int(numerator.replace(",", ""))
        bottom = int(denominator.replace(",", ""))
        if bottom == 0:
            return None
        return Fraction(top, bottom)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _fraction_from_expected(value: str) -> Optional[Fraction]:
    match = _FRACTION_RE.fullmatch(value.strip())
    if match:
        return _fraction_from_token(match.group(1), match.group(2))
    decimal_value = _decimal_from_token(value.strip())
    if decimal_value is None:
        return None
    return Fraction(decimal_value)


def _verify_integer_or_decimal(
    response: str,
    spec: VerifierSpec,
) -> tuple[bool, str, str]:
    expected = _decimal_from_token(spec.expected_answer)
    if expected is None:
        return False, "", "invalid_expected_number"
    if spec.verifier_type == "integer" and expected != expected.to_integral_value():
        return False, "", "invalid_expected_integer"

    region, explicitly_marked = _answer_region(response)
    match = _chosen_match(list(_NUMBER_RE.finditer(region)), explicitly_marked)
    if match is None:
        return False, "", "answer_number_not_found"
    extracted = match.group(1)
    candidate = _decimal_from_token(extracted)
    if candidate is None:
        return False, extracted, "invalid_candidate_number"
    if spec.verifier_type == "integer" and candidate != candidate.to_integral_value():
        return False, extracted, "candidate_is_not_integer"
    passed = abs(candidate - expected) <= spec.absolute_tolerance
    return passed, extracted, "verified" if passed else "numeric_mismatch"


def _verify_fraction(response: str, spec: VerifierSpec) -> tuple[bool, str, str]:
    expected = _fraction_from_expected(spec.expected_answer)
    if expected is None:
        return False, "", "invalid_expected_fraction"

    region, explicitly_marked = _answer_region(response)
    fraction_match = _chosen_match(list(_FRACTION_RE.finditer(region)), explicitly_marked)
    if fraction_match is not None:
        extracted = fraction_match.group(0)
        candidate = _fraction_from_token(fraction_match.group(1), fraction_match.group(2))
    else:
        number_match = _chosen_match(list(_NUMBER_RE.finditer(region)), explicitly_marked)
        if number_match is None:
            return False, "", "answer_fraction_not_found"
        extracted = number_match.group(1)
        decimal_value = _decimal_from_token(extracted)
        candidate = Fraction(decimal_value) if decimal_value is not None else None
    if candidate is None:
        return False, extracted, "invalid_candidate_fraction"
    passed = candidate == expected
    return passed, extracted, "verified" if passed else "fraction_mismatch"


def _extract_exact_answer(response: str) -> str:
    region, explicitly_marked = _answer_region(response)
    if explicitly_marked:
        first_line = next((line.strip() for line in region.splitlines() if line.strip()), "")
        return first_line
    return region


def _verify_normalized_exact(response: str, spec: VerifierSpec) -> tuple[bool, str, str]:
    extracted = _extract_exact_answer(response)
    candidate = normalize_answer_text(extracted, case_sensitive=spec.case_sensitive)
    accepted = {
        normalize_answer_text(spec.expected_answer, case_sensitive=spec.case_sensitive),
        *(
            normalize_answer_text(alias, case_sensitive=spec.case_sensitive)
            for alias in spec.aliases
        ),
    }
    passed = bool(candidate) and candidate in accepted
    return passed, extracted, "verified" if passed else "exact_mismatch"


def _extract_choice(response: str) -> str:
    region, explicitly_marked = _answer_region(response)
    cooked = region.strip()
    patterns = (
        r"^\s*\(?([A-Za-z0-9]{1,4})\)?(?:[\s.):,-]|$)",
        r"\b(?:option|choice|choose)\s*[:=]?\s*\(?([A-Za-z0-9]{1,4})\)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, cooked, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if _CHOICE_LABEL_RE.fullmatch(label):
                return label
    if explicitly_marked:
        token = cooked.split(maxsplit=1)[0].strip("()[]{}.,;:")
        label = token.upper()
        if _CHOICE_LABEL_RE.fullmatch(label):
            return label
    return ""


def _verify_multiple_choice(response: str, spec: VerifierSpec) -> tuple[bool, str, str]:
    extracted = _extract_choice(response)
    text_answer = _extract_exact_answer(response)
    candidate = normalize_answer_text(text_answer, case_sensitive=spec.case_sensitive)
    accepted_aliases = {
        normalize_answer_text(alias, case_sensitive=spec.case_sensitive)
        for alias in spec.aliases
    }
    if extracted == spec.expected_answer.upper():
        return True, extracted, "verified"
    if candidate and candidate in accepted_aliases:
        return True, text_answer, "verified_alias"
    if extracted:
        return False, extracted, "choice_mismatch"
    return False, text_answer, "choice_not_found"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant is not allowed: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        result[key] = value
    return result


def _load_strict_json(value: str) -> Any:
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )


def _json_payload_from_response(response: str) -> tuple[Optional[Any], str]:
    cooked = response.strip()
    fence = _FULL_JSON_FENCE_RE.fullmatch(cooked)
    if fence:
        cooked = fence.group(1).strip()
    try:
        return _load_strict_json(cooked), cooked
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, cooked


def _json_expected_value(raw: str) -> Any:
    try:
        return _load_strict_json(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return raw


def _json_values_equal(candidate: Any, expected: Any, *, case_sensitive: bool) -> bool:
    if isinstance(candidate, bool) or isinstance(expected, bool):
        return type(candidate) is type(expected) and candidate == expected
    if isinstance(candidate, (int, float)) and isinstance(expected, (int, float)):
        try:
            left = Decimal(str(candidate))
            right = Decimal(str(expected))
        except InvalidOperation:
            return False
        return left.is_finite() and right.is_finite() and left == right
    if isinstance(candidate, str) and isinstance(expected, str):
        return normalize_answer_text(candidate, case_sensitive=case_sensitive) == normalize_answer_text(
            expected,
            case_sensitive=case_sensitive,
        )
    return candidate == expected


def _verify_json_field(response: str, spec: VerifierSpec) -> tuple[bool, str, str]:
    payload, raw_payload = _json_payload_from_response(response)
    if payload is None:
        return False, raw_payload[:500], "invalid_json"

    current: Any = payload
    for part in spec.json_field.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, "", "json_field_missing"
        current = current[part]

    expected = _json_expected_value(spec.expected_answer)
    passed = _json_values_equal(current, expected, case_sensitive=spec.case_sensitive)
    extracted = json.dumps(current, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if not passed and isinstance(current, str):
        candidate_norm = normalize_answer_text(current, case_sensitive=spec.case_sensitive)
        alias_norms = {
            normalize_answer_text(alias, case_sensitive=spec.case_sensitive)
            for alias in spec.aliases
        }
        passed = candidate_norm in alias_norms
    return passed, extracted, "verified" if passed else "json_field_mismatch"


def _contains_contract_term(text: str, term: str, *, case_sensitive: bool) -> bool:
    candidate = unicodedata.normalize("NFKC", text)
    needle = unicodedata.normalize("NFKC", term).strip()
    if not case_sensitive:
        candidate = candidate.casefold()
        needle = needle.casefold()
    needle = re.sub(r"\s+", " ", needle)
    candidate = re.sub(r"\s+", " ", candidate)
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", candidate) is not None


def _response_contract_unicode_safe(response: str, spec: VerifierSpec) -> bool:
    """Reject invisible controls and script-confusable reward-spoof surfaces."""

    terms = (*spec.required_terms, *spec.forbidden_terms)
    # A mixed-language contract must not disable protection for its ASCII
    # terms. Script checks plus a small deterministic confusable skeleton keep
    # multilingual prose available while rejecting reward-spoof spellings.
    normalized_terms = unicodedata.normalize("NFKC", " ".join(terms))
    ascii_contract = bool(terms) and all(term.isascii() for term in terms)
    allowed_marks = {
        character
        for character in normalized_terms
        if unicodedata.category(character).startswith("M")
    }
    contract_tokens = {
        token.casefold()
        for token in re.findall(r"[^\W\d_]+", normalized_terms, re.UNICODE)
    }
    ascii_forbidden_tokens = {
        token.casefold()
        for term in spec.forbidden_terms
        for token in re.findall(r"[A-Za-z]+", term)
    }
    confusable_map = str.maketrans(
        {
            "\u0430": "a", "\u0432": "b", "\u0441": "c", "\u0435": "e", "\u0456": "i",
            "\u0458": "j", "\u043a": "k", "\u043c": "m", "\u043d": "h", "\u043e": "o",
            "\u0440": "p", "\u0455": "s", "\u0442": "t", "\u0443": "y", "\u0445": "x",
            "\u03b1": "a", "\u03b2": "b", "\u03b5": "e", "\u03b9": "i", "\u03ba": "k",
            "\u03bc": "m", "\u03bd": "v", "\u03bf": "o", "\u03c1": "p", "\u03c4": "t",
            "\u03c5": "y", "\u03c7": "x",
        }
    )
    normalized_response = unicodedata.normalize("NFKC", response)

    def confusable_skeleton(token: str) -> str:
        decomposed = unicodedata.normalize("NFKD", token.casefold())
        unmarked = "".join(
            character
            for character in decomposed
            if not unicodedata.category(character).startswith("M")
        )
        return unmarked.translate(confusable_map)

    def character_script(character: str) -> str:
        if not character.isalpha():
            return ""
        if character.isascii():
            return "LATIN"
        name = unicodedata.name(character, "")
        for script in (
            "LATIN",
            "CYRILLIC",
            "GREEK",
            "ARABIC",
            "HEBREW",
            "DEVANAGARI",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "THAI",
        ):
            if script in name:
                return script
        if "CJK" in name or "IDEOGRAPH" in name:
            return "CJK"
        return name.split(" ", 1)[0] or "UNKNOWN"

    for token in re.findall(r"[^\W\d_]+", normalized_response, re.UNICODE):
        scripts = {character_script(character) for character in token}
        scripts.discard("")
        # Accented Latin words remain one script; a token such as sеcret is a
        # Latin/Cyrillic mixture and is rejected even if Cyrillic is otherwise
        # legitimate elsewhere in the contract.
        if len(scripts) > 1 and token.casefold() not in contract_tokens:
            return False
        if ascii_contract and any(not character.isascii() for character in token):
            return False
        skeleton = confusable_skeleton(token)
        if (
            skeleton in ascii_forbidden_tokens
            and token.casefold() != skeleton
        ):
            return False
    for character in normalized_response:
        if character in {"\n", "\r", "\t"}:
            continue
        category = unicodedata.category(character)
        if category.startswith("C"):
            return False
        if category.startswith("M") and character not in allowed_marks:
            return False
    return True


def _verify_response_contract(response: str, spec: VerifierSpec) -> tuple[bool, str, str]:
    if not _response_contract_unicode_safe(response, spec):
        return False, "", "unsafe_unicode_contract_text"
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    bullet_texts = []
    non_bullet_lines = []
    for line in lines:
        match = _BULLET_RE.fullmatch(line)
        if match is None:
            non_bullet_lines.append(line)
        else:
            bullet_texts.append(match.group(1).strip())

    if spec.exact_bullet_count:
        if len(bullet_texts) != spec.exact_bullet_count:
            return False, str(len(bullet_texts)), "bullet_count_mismatch"
        if non_bullet_lines:
            return False, non_bullet_lines[0][:200], "unexpected_non_bullet_text"
    if spec.max_words_per_bullet:
        for bullet in bullet_texts:
            word_count = len(_WORD_RE.findall(bullet))
            if word_count > spec.max_words_per_bullet:
                return False, str(word_count), "bullet_word_limit_exceeded"

    for term in spec.required_terms:
        if not _contains_contract_term(response, term, case_sensitive=spec.case_sensitive):
            return False, term, "required_term_missing"
    for term in spec.forbidden_terms:
        if _contains_contract_term(response, term, case_sensitive=spec.case_sensitive):
            return False, term, "forbidden_term_present"

    summary = json.dumps(
        {
            "bullets": len(bullet_texts),
            "required_terms": len(spec.required_terms),
            "forbidden_terms": len(spec.forbidden_terms),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return True, summary, "verified"


def _verify_logical_entailment(
    prompt: object,
    response: str,
    spec: VerifierSpec,
) -> tuple[bool, str, str]:
    """Verify prompt semantics and an exact oracle-derived answer token."""

    try:
        task_ir = validate_prompt_task_ir(prompt, spec.task_ir_json)
        oracle_answer = derive_entailment_answer(task_ir)
    except (TypeError, ValueError, RuntimeError):
        return False, "", "prompt_task_ir_mismatch"
    if oracle_answer != spec.expected_answer:
        return False, oracle_answer, "oracle_answer_mismatch"
    if response != oracle_answer:
        return False, response, "answer_not_exact"
    return True, response, "verified"


def verify_candidate(
    prompt: object,
    response: object,
    metadata: Mapping[str, object],
) -> VerificationResult:
    """Verify one response using versioned metadata.

    The prompt is only parsed for bounded verifier types with an explicit safe
    grammar.  It is never evaluated as code or treated as an instruction to the
    verifier.
    """

    spec = parse_verifier_spec(metadata)
    if spec is None:
        return VerificationResult(
            schema_version=VERIFIER_SCHEMA_VERSION,
            verifier_type="",
            valid_spec=False,
            passed=False,
            score=0.0,
            reward=0.0,
            expected_answer="",
            extracted_answer="",
            reason="invalid_or_unsupported_spec",
        )

    response_text = str(response or "")
    if not response_text.strip():
        passed, extracted, reason = False, "", "empty_response"
    elif len(response_text) > _MAX_RESPONSE_CHARS:
        passed, extracted, reason = False, "", "response_too_long"
    elif spec.verifier_type in {"integer", "decimal"}:
        passed, extracted, reason = _verify_integer_or_decimal(response_text, spec)
    elif spec.verifier_type == "fraction":
        passed, extracted, reason = _verify_fraction(response_text, spec)
    elif spec.verifier_type == "normalized_exact":
        passed, extracted, reason = _verify_normalized_exact(response_text, spec)
    elif spec.verifier_type == "multiple_choice":
        passed, extracted, reason = _verify_multiple_choice(response_text, spec)
    elif spec.verifier_type == "json_field":
        passed, extracted, reason = _verify_json_field(response_text, spec)
    elif spec.verifier_type == "response_contract":
        passed, extracted, reason = _verify_response_contract(response_text, spec)
    elif spec.verifier_type == "logical_entailment":
        passed, extracted, reason = _verify_logical_entailment(prompt, response_text, spec)
    else:  # Defensive guard; parse_verifier_spec already rejects unknown types.
        passed, extracted, reason = False, "", "unsupported_verifier_type"

    return VerificationResult(
        schema_version=spec.schema_version,
        verifier_type=spec.verifier_type,
        valid_spec=True,
        passed=bool(passed),
        score=1.0 if passed else 0.0,
        reward=1.0 if passed else -1.0,
        expected_answer=spec.expected_answer,
        extracted_answer=extracted,
        reason=reason,
    )


__all__ = [
    "SUPPORTED_VERIFIER_TYPES",
    "VERIFIER_SCHEMA_VERSION",
    "VerificationResult",
    "VerifierSpec",
    "normalize_answer_text",
    "parse_verifier_spec",
    "verify_candidate",
]
