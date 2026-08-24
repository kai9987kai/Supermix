"""Build a deterministic, exact-verifiable dataset for Mosaic Expert Revival.

The builder deliberately keeps train, development, and sealed holdout component
identities disjoint.  It never consults model outputs or evaluation scores.  A
prior manifest can be supplied as a forbidden set, which makes accidental
corpus reuse a hard error rather than a warning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import unicodedata
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BUNDLE_SCHEMA = "supermix-mosaic-expert-revival-dataset-v1"
MOSAIC_ROW_SCHEMA = "supermix-mosaic-row-v1"
ATOMIC_ROW_SCHEMA = "supermix-mosaic-atomic-row-v1"
SPLITS = ("train", "dev", "holdout")
KINDS = ("dialogue_math", "math_chain", "dialogue_dialogue")
MATH_FAMILIES = ("addition", "subtraction", "percentage", "average", "algebra")
SEQUENCE_LENGTH = 128
_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[A-Za-z]+")
_PLACEHOLDER_RE = re.compile(
    r"(?:\{\{[^{}]+\}\}|<\s*(?:placeholder|todo|insert|fill)[^>]*>|"
    r"\[\s*(?:placeholder|todo|insert|fill|redacted)[^]]*\])",
    re.IGNORECASE,
)
_TRUNCATION_RE = re.compile(r"(?:\.\.\.|…|\[truncated\]|<truncated>|to be continued)\s*$", re.IGNORECASE)
_SIGNED_INTEGER = r"(-?\d+)"
_MATH_TASK_MARKERS = {
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "arithmetic",
    "percent",
    "percentage",
    "average",
    "algebra",
    "algebra_one_step",
    "equation",
    "word_problem",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_sha256(tokenizer: Any) -> str:
    return sha256_bytes(canonical_json(tokenizer.to_dict()).encode("utf-8"))


def _pair_token_issue(tokenizer: Any, user: str, assistant: str, maximum_length: int) -> str | None:
    if tokenizer.unknown_rate(user) != 0.0 or tokenizer.unknown_rate(assistant) != 0.0:
        return "v70_unknown_token"
    encoded, _ = tokenizer.encode_turn(user, assistant)
    if len(encoded) > maximum_length:
        return "over_token_limit"
    return None


def load_bound_v70_tokenizer(
    checkpoint_path: Path,
    *,
    expected_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
) -> Any:
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError(
            f"v70 checkpoint hash mismatch: expected {expected_checkpoint_sha256}, got {actual_checkpoint_hash}"
        )
    import torch

    import mimomix_text

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != "supermix-v57-talk-checkpoint-v1":
        raise ValueError("dataset tokenizer source is not a compatible talk checkpoint")
    extra = payload.get("extra")
    if not isinstance(extra, Mapping) or extra.get("run_name") != "v70_moe":
        raise ValueError("dataset tokenizer source is not the frozen v70_moe run")
    tokenizer = mimomix_text.WordTokenizer.from_dict(payload["tokenizer"])
    actual_tokenizer_hash = tokenizer_sha256(tokenizer)
    if actual_tokenizer_hash != expected_tokenizer_sha256:
        raise ValueError(
            f"v70 tokenizer hash mismatch: expected {expected_tokenizer_sha256}, got {actual_tokenizer_hash}"
        )
    return tokenizer


def stable_id(namespace: str, value: Any) -> str:
    payload = f"{namespace}\0{canonical_json(value)}".encode("utf-8")
    return sha256_bytes(payload)


def normalize_text(value: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFKC", value)).strip()


def prompt_identifier(prompt: str) -> str:
    return f"prompt:{stable_id('normalized-prompt', normalize_text(prompt).casefold())}"


def _exact_integer(value: Any, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact integer")
    return value


def _canonical_semantic_spec(family: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(spec, Mapping):
        raise TypeError("math spec must be an object")
    expected_fields: dict[str, set[str]] = {
        "addition": {"left", "right"},
        "subtraction": {"left", "right"},
        "percentage": {"percent", "amount"},
        "average": {"values"},
        "algebra": {"solution", "offset"},
        "math_chain": {"left", "right", "subtract"},
    }
    if family not in expected_fields or set(spec) != expected_fields[family]:
        raise ValueError(f"invalid closed spec schema for {family!r}")
    if family == "average":
        values = spec["values"]
        if not isinstance(values, list) or not values:
            raise TypeError("average values must be a non-empty integer list")
        return {"values": sorted(_exact_integer(value, "average value") for value in values)}
    result = {key: _exact_integer(spec[key], key) for key in sorted(expected_fields[family])}
    if family in {"addition", "math_chain"}:
        result["left"], result["right"] = sorted((result["left"], result["right"]))
    return result


def math_semantic_identifier(family: str, spec: Mapping[str, Any]) -> str:
    canonical = _canonical_semantic_spec(family, spec)
    return f"semantic:{stable_id('math-semantic', {'family': family, 'spec': canonical})}"


def _chain_semantic_identifiers(spec: Mapping[str, Any]) -> set[str]:
    canonical = _canonical_semantic_spec("math_chain", spec)
    intermediate = canonical["left"] + canonical["right"]
    return {
        math_semantic_identifier("math_chain", canonical),
        math_semantic_identifier(
            "addition", {"left": canonical["left"], "right": canonical["right"]}
        ),
        math_semantic_identifier(
            "subtraction", {"left": intermediate, "right": canonical["subtract"]}
        ),
    }


def _semantic_identifiers_from_prompt(prompt: str) -> set[str]:
    text = normalize_text(prompt).casefold().rstrip(".?")
    match = re.fullmatch(
        rf"first add {_SIGNED_INTEGER} and {_SIGNED_INTEGER}\. then subtract {_SIGNED_INTEGER} from the answer",
        text,
    )
    if match:
        left, right, subtract = (int(value) for value in match.groups())
        return _chain_semantic_identifiers(
            {"left": left, "right": right, "subtract": subtract}
        )
    match = re.fullmatch(
        rf"a student has {_SIGNED_INTEGER} [a-z]+\. they get {_SIGNED_INTEGER} more and then "
        rf"give away {_SIGNED_INTEGER}\. how many [a-z]+ do they have now",
        text,
    )
    if match:
        left, right, subtract = (int(value) for value in match.groups())
        return _chain_semantic_identifiers(
            {"left": left, "right": right, "subtract": subtract}
        )
    body = text
    for prefix in (
        r"what is\s+",
        r"solve this basic math problem:\s*",
        r"quick question:\s*",
        r"please help with this\.\s*",
    ):
        stripped = re.sub(rf"^{prefix}", "", body)
        if stripped != body:
            body = stripped
            break
    match = re.fullmatch(rf"{_SIGNED_INTEGER}\s*(?:%|percent)\s*of\s*{_SIGNED_INTEGER}", body)
    if match:
        percent, amount = (int(value) for value in match.groups())
        return {math_semantic_identifier("percentage", {"percent": percent, "amount": amount})}
    match = re.fullmatch(r"(?:the )?average of (.+)", body)
    if match is None:
        match = re.fullmatch(r"find the average \(mean\) of these numbers: (.+)", text)
    if match:
        pieces = [piece.strip() for piece in match.group(1).split(",")]
        if pieces and all(re.fullmatch(r"-?\d+", piece) for piece in pieces):
            return {math_semantic_identifier("average", {"values": [int(piece) for piece in pieces]})}
    match = re.fullmatch(rf"solve(?: for)? x:? x\s*\+\s*{_SIGNED_INTEGER}\s*=\s*{_SIGNED_INTEGER}", text)
    if match:
        offset, total = (int(value) for value in match.groups())
        return {math_semantic_identifier("algebra", {"solution": total - offset, "offset": offset})}
    match = re.fullmatch(rf"{_SIGNED_INTEGER}\s*(\+|-|plus|minus)\s*{_SIGNED_INTEGER}", body)
    if match:
        left, operator, right = match.groups()
        family = "addition" if operator in {"+", "plus"} else "subtraction"
        return {math_semantic_identifier(family, {"left": int(left), "right": int(right)})}
    return set()


def _semantic_identifier_from_prompt(prompt: str) -> str | None:
    identifiers = _semantic_identifiers_from_prompt(prompt)
    return min(identifiers) if identifiers else None


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    reduced = denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced == 1:
        places = max(_factor_count(denominator, 2), _factor_count(denominator, 5))
        scaled = value.numerator * (10**places) // denominator
        sign = "-" if scaled < 0 else ""
        digits = str(abs(scaled)).rjust(places + 1, "0")
        rendered = f"{sign}{digits[:-places]}.{digits[-places:]}" if places else digits
        return rendered.rstrip("0").rstrip(".")
    return f"{value.numerator}/{value.denominator}"


def _factor_count(value: int, factor: int) -> int:
    count = 0
    while value % factor == 0:
        count += 1
        value //= factor
    return count


def _validated_generation_math_spec(family: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    _canonical_semantic_spec(family, spec)
    raw = dict(spec)
    if family == "addition" and not (
        11 <= raw["left"] <= 999 and 11 <= raw["right"] <= 999
    ):
        raise ValueError("addition operands are outside the generator schema")
    if family == "subtraction" and not (
        11 <= raw["right"] <= 499 and raw["right"] < raw["left"] <= 999
    ):
        raise ValueError("subtraction operands are outside the generator schema")
    if family == "percentage" and not (
        1 <= raw["percent"] <= 99
        and 40 <= raw["amount"] <= 200_000
        and raw["amount"] % 20 == 0
    ):
        raise ValueError("percentage operands are outside the generator schema")
    if family == "average":
        values = raw["values"]
        if len(values) != 4 or any(value < 2 or value > 200 for value in values):
            raise ValueError("average values are outside the generator schema")
        raw["values"] = list(values)
    if family == "algebra" and not (
        2 <= raw["solution"] <= 10_000 and 2 <= raw["offset"] <= 5_000
    ):
        raise ValueError("algebra operands are outside the generator schema")
    return raw


def _math_render(family: str, spec: Mapping[str, Any]) -> tuple[str, str, Fraction]:
    spec = _validated_generation_math_spec(family, spec)
    if family == "addition":
        left, right = spec["left"], spec["right"]
        answer = Fraction(left + right)
        return (
            f"What is {left} plus {right}?",
            f"{left} + {right} = {_fraction_text(answer)}. The answer is {_fraction_text(answer)}.",
            answer,
        )
    if family == "subtraction":
        left, right = spec["left"], spec["right"]
        answer = Fraction(left - right)
        return (
            f"What is {left} - {right}?",
            f"{left} - {right} = {_fraction_text(answer)}. The answer is {_fraction_text(answer)}.",
            answer,
        )
    if family == "percentage":
        percent, amount = spec["percent"], spec["amount"]
        answer = Fraction(percent * amount, 100)
        return (
            f"What is {percent} percent of {amount}?",
            f"{percent} percent of {amount} is {percent} / 100 times {amount} = "
            f"{_fraction_text(answer)}. The answer is {_fraction_text(answer)}.",
            answer,
        )
    if family == "average":
        values = tuple(spec["values"])
        answer = Fraction(sum(values), len(values))
        joined = ", ".join(str(value) for value in values)
        return (
            f"What is the average of {joined}?",
            f"The sum is {sum(values)} and there are {len(values)} values. "
            f"{sum(values)} / {len(values)} = {_fraction_text(answer)}. "
            f"The answer is {_fraction_text(answer)}.",
            answer,
        )
    if family == "algebra":
        solution, offset = spec["solution"], spec["offset"]
        total = solution + offset
        answer = Fraction(solution)
        return (
            f"Solve x + {offset} = {total}.",
            f"Subtract {offset} from {total}. x = {_fraction_text(answer)}. "
            f"The answer is {_fraction_text(answer)}.",
            answer,
        )
    raise ValueError(f"unsupported math family: {family!r}")


def _math_component(family: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    user, assistant, answer = _math_render(family, spec)
    spec = dict(spec)
    identity = {
        "domain": "math",
        "family": family,
        "spec": _canonical_semantic_spec(family, spec),
    }
    return {
        "component_id": stable_id("component", identity),
        "domain": "math",
        "family": family,
        "user": user,
        "assistant": assistant,
        "verification": {
            "family": family,
            "spec": dict(spec),
            "answer_fraction": f"{answer.numerator}/{answer.denominator}",
        },
    }


def _dialogue_component(user: str, assistant: str) -> dict[str, Any]:
    user, assistant = normalize_text(user), normalize_text(assistant)
    identity = {"domain": "dialogue", "user": user, "assistant": assistant}
    return {
        "component_id": stable_id("component", identity),
        "domain": "dialogue",
        "family": "dialogue",
        "user": user,
        "assistant": assistant,
    }


def verify_math_component(component: Mapping[str, Any]) -> bool:
    if set(component) != {
        "component_id",
        "domain",
        "family",
        "user",
        "assistant",
        "verification",
    } or component.get("domain") != "math":
        return False
    verification = component.get("verification")
    if not isinstance(verification, Mapping) or set(verification) != {
        "family",
        "spec",
        "answer_fraction",
    }:
        return False
    try:
        family = verification["family"]
        if not isinstance(family, str):
            return False
        spec = verification["spec"]
        user, assistant, answer = _math_render(family, spec)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    identity = {
        "domain": "math",
        "family": family,
        "spec": _canonical_semantic_spec(family, spec),
    }
    return (
        component.get("component_id") == stable_id("component", identity)
        and component.get("family") == family
        and component.get("user") == user
        and component.get("assistant") == assistant
        and verification.get("answer_fraction") == f"{answer.numerator}/{answer.denominator}"
    )


def verify_dialogue_component(component: Mapping[str, Any]) -> bool:
    if set(component) != {"component_id", "domain", "family", "user", "assistant"}:
        return False
    if component.get("domain") != "dialogue" or component.get("family") != "dialogue":
        return False
    user = component.get("user")
    assistant = component.get("assistant")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return False
    if not user or not assistant or user != normalize_text(user) or assistant != normalize_text(assistant):
        return False
    identity = {"domain": "dialogue", "user": user, "assistant": assistant}
    return component.get("component_id") == stable_id("component", identity)


def verify_component(component: Mapping[str, Any]) -> bool:
    return verify_math_component(component) if component.get("domain") == "math" else verify_dialogue_component(component)


def _validated_chain_spec(spec: Mapping[str, Any]) -> dict[str, int]:
    _canonical_semantic_spec("math_chain", spec)
    raw = dict(spec)
    if not (
        10 <= raw["left"] <= 500
        and 10 <= raw["right"] <= 500
        and 1 <= raw["subtract"] < raw["left"] + raw["right"]
    ):
        raise ValueError("math-chain operands are outside the generator schema")
    return raw


def _chain_render(spec: Mapping[str, Any]) -> tuple[str, str, Fraction]:
    spec = _validated_chain_spec(spec)
    left, right, subtract = spec["left"], spec["right"], spec["subtract"]
    intermediate = Fraction(left + right)
    answer = intermediate - subtract
    user = f"First add {left} and {right}. Then subtract {subtract} from the answer."
    assistant = (
        f"First, {left} + {right} = {_fraction_text(intermediate)}. "
        f"Then, {_fraction_text(intermediate)} - {subtract} = {_fraction_text(answer)}. "
        f"The answer is {_fraction_text(answer)}."
    )
    return user, assistant, answer


def _mosaic_prompt(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    return f"Answer both questions in order. First: {first['user']} Then: {second['user']}"


def _mosaic_answer(first: Mapping[str, Any], second: Mapping[str, Any]) -> str:
    return f"First answer: {first['assistant']} Then answer: {second['assistant']}"


def _make_mosaic_row(split: str, kind: str, first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    prompt = _mosaic_prompt(first, second)
    assistant = _mosaic_answer(first, second)
    identity = {
        "split": split,
        "kind": kind,
        "component_ids": [first["component_id"], second["component_id"]],
    }
    return {
        "schema": MOSAIC_ROW_SCHEMA,
        "row_id": stable_id("mosaic-row", identity),
        "split": split,
        "kind": kind,
        "user": prompt,
        "assistant": assistant,
        "components": [dict(first), dict(second)],
    }


def _make_chain_row(split: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    user, assistant, answer = _chain_render(spec)
    spec = dict(spec)
    identity = {
        "split": split,
        "kind": "math_chain",
        "spec": _canonical_semantic_spec("math_chain", spec),
    }
    return {
        "schema": MOSAIC_ROW_SCHEMA,
        "row_id": stable_id("mosaic-row", identity),
        "split": split,
        "kind": "math_chain",
        "user": user,
        "assistant": assistant,
        "verification": {
            "family": "math_chain",
            "spec": dict(spec),
            "answer_fraction": f"{answer.numerator}/{answer.denominator}",
        },
    }


def verify_mosaic_row(row: Mapping[str, Any]) -> bool:
    if row.get("schema") != MOSAIC_ROW_SCHEMA or row.get("split") not in SPLITS:
        return False
    kind = row.get("kind")
    if kind == "math_chain":
        if set(row) != {
            "schema",
            "row_id",
            "split",
            "kind",
            "user",
            "assistant",
            "verification",
        }:
            return False
        verification = row.get("verification")
        if not isinstance(verification, Mapping) or set(verification) != {
            "family",
            "spec",
            "answer_fraction",
        }:
            return False
        if verification.get("family") != "math_chain":
            return False
        try:
            spec = verification["spec"]
            user, assistant, answer = _chain_render(spec)
        except (KeyError, TypeError, ValueError):
            return False
        identity = {
            "split": row["split"],
            "kind": kind,
            "spec": _canonical_semantic_spec("math_chain", spec),
        }
        return (
            row.get("row_id") == stable_id("mosaic-row", identity)
            and row.get("user") == user
            and row.get("assistant") == assistant
            and verification.get("answer_fraction") == f"{answer.numerator}/{answer.denominator}"
        )
    if kind not in {"dialogue_math", "dialogue_dialogue"}:
        return False
    if set(row) != {"schema", "row_id", "split", "kind", "user", "assistant", "components"}:
        return False
    components = row.get("components")
    if not isinstance(components, list) or len(components) != 2:
        return False
    first, second = components
    if not all(isinstance(component, Mapping) and verify_component(component) for component in components):
        return False
    domains = (first["domain"], second["domain"])
    if kind == "dialogue_math" and domains != ("dialogue", "math"):
        return False
    if kind == "dialogue_dialogue" and domains != ("dialogue", "dialogue"):
        return False
    identity = {
        "split": row["split"],
        "kind": kind,
        "component_ids": [first["component_id"], second["component_id"]],
    }
    return (
        row.get("row_id") == stable_id("mosaic-row", identity)
        and row.get("user") == _mosaic_prompt(first, second)
        and row.get("assistant") == _mosaic_answer(first, second)
    )


def verify_atomic_row(row: Mapping[str, Any]) -> bool:
    if set(row) != {"schema", "row_id", "split", "domain", "component"}:
        return False
    if row.get("schema") != ATOMIC_ROW_SCHEMA or row.get("split") not in SPLITS:
        return False
    component = row.get("component")
    return (
        isinstance(component, Mapping)
        and row.get("domain") == component.get("domain")
        and verify_component(component)
        and row.get("row_id") == component.get("component_id")
    )


def prediction_matches(row: Mapping[str, Any], prediction: str) -> bool:
    """Strict verifier used by the preregistered composition gate."""

    if row.get("schema") == MOSAIC_ROW_SCHEMA:
        return verify_mosaic_row(row) and normalize_text(prediction) == normalize_text(str(row["assistant"]))
    if row.get("schema") == ATOMIC_ROW_SCHEMA:
        component = row.get("component")
        return verify_atomic_row(row) and normalize_text(prediction) == normalize_text(str(component["assistant"]))
    return False


def _extract_dialogue_pair(payload: Mapping[str, Any]) -> tuple[str, str] | None:
    candidates = (
        (payload.get("user"), payload.get("assistant")),
        (payload.get("prompt"), payload.get("response")),
        (payload.get("instruction"), payload.get("output")),
    )
    for user, assistant in candidates:
        if isinstance(user, str) and isinstance(assistant, str):
            return user, assistant
    messages = payload.get("messages")
    if isinstance(messages, list):
        user = next((item.get("content") for item in messages if isinstance(item, Mapping) and item.get("role") == "user"), None)
        assistant = next((item.get("content") for item in messages if isinstance(item, Mapping) and item.get("role") == "assistant"), None)
        if isinstance(user, str) and isinstance(assistant, str):
            return user, assistant
    return None


def _dialogue_quality_reason(user: str, assistant: str) -> str | None:
    user, assistant = normalize_text(user), normalize_text(assistant)
    if not user or not assistant:
        return "empty"
    if len(user) > 2_000 or len(assistant) > 4_000:
        return "over_length"
    if len(_WORD_RE.findall(user)) < 2 or len(_WORD_RE.findall(assistant)) < 2:
        return "too_few_words"
    if user.casefold() == assistant.casefold():
        return "echo"
    if _PLACEHOLDER_RE.search(user) or _PLACEHOLDER_RE.search(assistant):
        return "placeholder"
    if _TRUNCATION_RE.search(user) or _TRUNCATION_RE.search(assistant):
        return "truncated"
    if assistant[-1].isalnum() or assistant[-1] in ",-–—":
        return "incomplete_ending"
    if any(marker in user or marker in assistant for marker in ("\ufffd", "Ã", "â€")):
        return "mojibake"
    if any(ord(character) < 32 and character not in "\t\r\n" for character in user + assistant):
        return "control_character"
    return None


def _load_dialogue_components_with_stats(
    path: Path,
    *,
    tokenizer: Any | None = None,
    maximum_length: int = SEQUENCE_LENGTH,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    component_ids_by_prompt: dict[str, set[str]] = {}
    rejection_counts: Counter[str] = Counter()
    input_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            input_rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected an object at {path}:{line_number}")
            domain = payload.get("domain")
            if domain is not None and normalize_text(str(domain)).casefold() != "dialogue":
                rejection_counts["non_dialogue_domain"] += 1
                continue
            task = normalize_text(str(payload.get("task", ""))).casefold()
            topic = normalize_text(str(payload.get("topic", ""))).casefold()
            if domain is None and (task in _MATH_TASK_MARKERS or topic == "basic_math"):
                rejection_counts["domainless_math_marker"] += 1
                continue
            pair = _extract_dialogue_pair(payload)
            if pair is None:
                rejection_counts["missing_pair"] += 1
                continue
            quality_reason = _dialogue_quality_reason(*pair)
            if quality_reason is not None:
                rejection_counts[quality_reason] += 1
                continue
            if tokenizer is not None:
                token_issue = _pair_token_issue(tokenizer, pair[0], pair[1], maximum_length)
                if token_issue is not None:
                    rejection_counts[token_issue] += 1
                    continue
            component = _dialogue_component(*pair)
            if component["component_id"] in components:
                rejection_counts["duplicate"] += 1
                continue
            components[component["component_id"]] = component
            prompt_id = prompt_identifier(component["user"])
            component_ids_by_prompt.setdefault(prompt_id, set()).add(component["component_id"])
    ambiguous_component_ids = {
        component_id
        for component_ids in component_ids_by_prompt.values()
        if len(component_ids) > 1
        for component_id in component_ids
    }
    if ambiguous_component_ids:
        rejection_counts["ambiguous_prompt"] += len(ambiguous_component_ids)
        for component_id in ambiguous_component_ids:
            del components[component_id]
    result = list(components.values())
    if len(result) < 12:
        raise ValueError(
            "at least 12 quality-filtered unique dialogue pairs are required for disjoint "
            "train/dev/holdout splits"
        )
    return result, {
        "input_rows": input_rows,
        "accepted_unique_rows": len(result),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "domain_policy": (
            "domain-tagged rows must be dialogue; domainless rows are accepted only without a known "
            "math task marker or topic=basic_math"
        ),
        "quality_policy": (
            "reject empty, over-length, echo, placeholder, truncated/incomplete-ending, "
            "mojibake, control, sub-two-word rows, and every target variant of an ambiguous "
            "normalized prompt"
        ),
        "token_policy": (
            f"reject any v70 <unk> token or full turn over {maximum_length} tokens before deterministic splitting"
            if tokenizer is not None
            else "not applied by standalone dialogue inspection"
        ),
    }


def load_dialogue_components(path: Path) -> list[dict[str, Any]]:
    components, _ = _load_dialogue_components_with_stats(path)
    return components


def _payload_prompt(payload: Mapping[str, Any]) -> str | None:
    for key in ("user", "prompt", "instruction"):
        value = payload.get(key)
        if isinstance(value, str) and normalize_text(value):
            return value
    messages = payload.get("messages")
    if isinstance(messages, list):
        value = next(
            (
                item.get("content")
                for item in messages
                if isinstance(item, Mapping) and item.get("role") == "user"
            ),
            None,
        )
        if isinstance(value, str) and normalize_text(value):
            return value
    return None


def _scan_forbidden_corpus(path: Path) -> tuple[set[str], dict[str, Any]]:
    identifiers: set[str] = set()
    prompt_count = 0
    semantic_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid forbidden corpus JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"expected a forbidden corpus object at {path}:{line_number}")
            prompt = _payload_prompt(payload)
            if prompt is None:
                continue
            prompt_count += 1
            identifiers.add(prompt_identifier(prompt))
            semantic_identifiers = _semantic_identifiers_from_prompt(prompt)
            if semantic_identifiers:
                semantic_count += 1
                identifiers.update(semantic_identifiers)
    return identifiers, {
        "path": path.name,
        "sha256": sha256_file(path),
        "prompt_rows_scanned": prompt_count,
        "math_semantic_rows_recognized": semantic_count,
        "identifier_count": len(identifiers),
    }


def _split_dialogues(components: Sequence[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    ordered = sorted(components, key=lambda item: stable_id(f"dialogue-split-{seed}", item["component_id"]))
    count = len(ordered)
    dev_count = max(2, int(math.floor(count * 0.1)))
    holdout_count = max(2, int(math.floor(count * 0.1)))
    if count - dev_count - holdout_count < 2:
        raise ValueError("not enough dialogue pairs to retain two train components")
    return {
        "dev": ordered[:dev_count],
        "holdout": ordered[dev_count : dev_count + holdout_count],
        "train": ordered[dev_count + holdout_count :],
    }


def _random_math_spec(family: str, rng: random.Random) -> dict[str, Any]:
    if family == "addition":
        return {"left": rng.randint(11, 999), "right": rng.randint(11, 999)}
    if family == "subtraction":
        right = rng.randint(11, 499)
        return {"left": right + rng.randint(1, 500), "right": right}
    if family == "percentage":
        return {"percent": rng.randint(1, 99), "amount": rng.randint(2, 10_000) * 20}
    if family == "average":
        return {"values": [rng.randint(2, 200) for _ in range(4)]}
    if family == "algebra":
        return {"solution": rng.randint(2, 10_000), "offset": rng.randint(2, 5_000)}
    raise ValueError(f"unsupported family: {family}")


def math_family_capacities() -> dict[str, int]:
    return {
        "addition": 989 * 989,
        "subtraction": 489 * 500,
        "percentage": 99 * 9_999,
        "average": 199**4,
        "algebra": 9_999 * 4_999,
    }


def validate_math_component_capacity(total_count: int) -> dict[str, int]:
    if total_count <= 0:
        raise ValueError("math component count must be positive")
    required = Counter(MATH_FAMILIES[index % len(MATH_FAMILIES)] for index in range(total_count))
    capacities = math_family_capacities()
    impossible = {
        family: {"required": required[family], "capacity": capacities[family]}
        for family in MATH_FAMILIES
        if required[family] > capacities[family]
    }
    if impossible:
        raise ValueError(f"requested math components exceed exact family capacity: {impossible}")
    return dict(required)


def _generate_math_components(
    split: str,
    count: int,
    seed: int,
    globally_used: set[str],
    globally_used_semantics: set[str],
    externally_forbidden: set[str],
    tokenizer: Any,
    maximum_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_math_component_capacity(count)
    rng = random.Random(int(stable_id("math-seed", {"seed": seed, "split": split})[:16], 16))
    result: list[dict[str, Any]] = []
    attempts = 0
    rejections: Counter[str] = Counter()
    while len(result) < count:
        attempts += 1
        if attempts > count * 100 + 1000:
            raise RuntimeError("could not create enough unique math components")
        family = MATH_FAMILIES[len(result) % len(MATH_FAMILIES)]
        component = _math_component(family, _random_math_spec(family, rng))
        if component["component_id"] in globally_used:
            rejections["duplicate_semantics"] += 1
            continue
        semantic = math_semantic_identifier(family, component["verification"]["spec"])
        if semantic in globally_used_semantics:
            rejections["internal_semantic_collision"] += 1
            continue
        if prompt_identifier(component["user"]) in externally_forbidden or semantic in externally_forbidden:
            rejections["external_corpus_collision"] += 1
            continue
        token_issue = _pair_token_issue(
            tokenizer, component["user"], component["assistant"], maximum_length
        )
        if token_issue is not None:
            rejections[token_issue] += 1
            continue
        globally_used.add(component["component_id"])
        globally_used_semantics.add(semantic)
        result.append(component)
    return result, {"attempts": attempts, "rejection_counts": dict(sorted(rejections.items()))}


def _generate_mosaic_rows(
    split: str,
    count: int,
    dialogues: Sequence[dict[str, Any]],
    maths: Sequence[dict[str, Any]],
    seed: int,
    globally_used_semantics: set[str],
    externally_forbidden: set[str],
    tokenizer: Any,
    maximum_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(dialogues) < 2 or not maths:
        raise ValueError(f"split {split!r} lacks components")
    rng = random.Random(int(stable_id("mosaic-seed", {"seed": seed, "split": split})[:16], 16))
    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    attempts = 0
    rejections: Counter[str] = Counter()
    while len(rows) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise RuntimeError(f"could not create {count} unique rows for {split}")
        kind = KINDS[len(rows) % len(KINDS)]
        if kind == "dialogue_math":
            row = _make_mosaic_row(split, kind, rng.choice(dialogues), rng.choice(maths))
        elif kind == "dialogue_dialogue":
            first = rng.choice(dialogues)
            second = rng.choice(dialogues)
            if first["component_id"] == second["component_id"]:
                rejections["same_dialogue_component"] += 1
                continue
            row = _make_mosaic_row(split, kind, first, second)
        else:
            left, right = rng.randint(10, 500), rng.randint(10, 500)
            subtract = rng.randint(1, left + right - 1)
            row = _make_chain_row(split, {"left": left, "right": right, "subtract": subtract})
        if row["row_id"] in used_ids:
            rejections["duplicate_row"] += 1
            continue
        protected = {prompt_identifier(str(row["user"]))}
        chain_semantics: set[str] = set()
        if row["kind"] == "math_chain":
            chain_semantics = _chain_semantic_identifiers(row["verification"]["spec"])
            protected.update(chain_semantics)
            if chain_semantics & globally_used_semantics:
                rejections["internal_semantic_collision"] += 1
                continue
        if protected & externally_forbidden:
            rejections["external_corpus_collision"] += 1
            continue
        token_issue = _pair_token_issue(
            tokenizer, str(row["user"]), str(row["assistant"]), maximum_length
        )
        if token_issue is not None:
            rejections[token_issue] += 1
            continue
        if not verify_mosaic_row(row):
            raise AssertionError("internal mosaic verifier rejected a generated row")
        used_ids.add(row["row_id"])
        globally_used_semantics.update(chain_semantics)
        rows.append(row)
    return rows, {"attempts": attempts, "rejection_counts": dict(sorted(rejections.items()))}


def _atomic_rows(split: str, components: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "schema": ATOMIC_ROW_SCHEMA,
            "row_id": component["component_id"],
            "split": split,
            "domain": component["domain"],
            "component": component,
        }
        for component in components
    ]
    if not all(verify_atomic_row(row) for row in rows):
        raise AssertionError("internal atomic verifier rejected a generated row")
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_forbidden_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("forbidden manifest must contain an object")
    id_file = payload.get("id_file")
    if not isinstance(id_file, str):
        raise ValueError("forbidden manifest is missing id_file")
    resolved = path.parent / id_file
    expected = payload.get("files", {}).get(id_file, {}).get("sha256")
    if not isinstance(expected, str) or sha256_file(resolved) != expected:
        raise ValueError("forbidden manifest id_file hash mismatch")
    return {line.strip() for line in resolved.read_text(encoding="utf-8").splitlines() if line.strip()}


def _all_identifiers(
    dialogue_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    math_splits: Mapping[str, Sequence[Mapping[str, Any]]],
    mosaic_splits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[str], dict[str, str], set[str]]:
    identifiers: list[str] = []
    ownership: dict[str, str] = {}
    dialogue_replay_exemptions: set[str] = set()
    for split in SPLITS:
        for component in (*dialogue_splits[split], *math_splits[split]):
            component_identifiers = [
                f"component:{component['component_id']}",
                prompt_identifier(str(component["user"])),
            ]
            if component["domain"] == "math":
                component_identifiers.append(
                    math_semantic_identifier(
                        str(component["family"]), component["verification"]["spec"]
                    )
                )
            else:
                dialogue_replay_exemptions.add(component_identifiers[1])
            for identifier in component_identifiers:
                previous = ownership.setdefault(identifier, split)
                if previous != split:
                    raise ValueError(f"cross-split component collision: {identifier}")
                identifiers.append(identifier)
        for row in mosaic_splits[split]:
            row_identifier = f"row:{row['row_id']}"
            row_identifiers = [row_identifier, prompt_identifier(str(row["user"]))]
            if row["kind"] == "math_chain":
                row_identifiers.extend(
                    sorted(_chain_semantic_identifiers(row["verification"]["spec"]))
                )
            for identifier in row_identifiers:
                previous = ownership.setdefault(identifier, split)
                if previous != split:
                    raise ValueError(f"cross-split row collision: {identifier}")
                identifiers.append(identifier)
    if len(identifiers) != len(set(identifiers)):
        # Reuse within one split also risks overweighting a supposedly unique example.
        duplicates = [item for item, count in Counter(identifiers).items() if count > 1]
        raise ValueError(f"duplicate identifiers within dataset: {duplicates[:3]}")
    return sorted(identifiers), ownership, dialogue_replay_exemptions


def build_bundle(
    dialogue_jsonl: Path,
    output_dir: Path,
    *,
    tokenizer: Any,
    parent_checkpoint_sha256: str,
    expected_tokenizer_sha256: str,
    seed: int = 710_413,
    train_count: int = 12000,
    dev_count: int = 1200,
    holdout_count: int = 1200,
    forbidden_manifests: Sequence[Path] = (),
    forbidden_corpora: Sequence[Path] = (),
) -> dict[str, Any]:
    counts = {"train": int(train_count), "dev": int(dev_count), "holdout": int(holdout_count)}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("all split counts must be positive")
    for label, value in (
        ("parent checkpoint", parent_checkpoint_sha256),
        ("tokenizer", expected_tokenizer_sha256),
    ):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{label} binding must be a lowercase SHA-256")
    actual_tokenizer_hash = tokenizer_sha256(tokenizer)
    if actual_tokenizer_hash != expected_tokenizer_sha256:
        raise ValueError(
            f"builder tokenizer hash mismatch: expected {expected_tokenizer_sha256}, got {actual_tokenizer_hash}"
        )
    externally_forbidden: set[str] = set()
    external_corpus_scans: list[dict[str, Any]] = []
    for corpus_path in forbidden_corpora:
        corpus_ids, scan = _scan_forbidden_corpus(corpus_path)
        externally_forbidden.update(corpus_ids)
        external_corpus_scans.append(scan)
    dialogue_components, dialogue_filter = _load_dialogue_components_with_stats(
        dialogue_jsonl,
        tokenizer=tokenizer,
        maximum_length=SEQUENCE_LENGTH,
    )
    dialogue_splits = _split_dialogues(dialogue_components, seed)
    globally_used_math: set[str] = set()
    globally_used_semantics: set[str] = set()
    math_counts = {split: max(64, counts[split] * 2) for split in SPLITS}
    validate_math_component_capacity(sum(math_counts.values()))
    math_splits: dict[str, list[dict[str, Any]]] = {}
    math_generation_stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        generated, stats = _generate_math_components(
            split,
            math_counts[split],
            seed,
            globally_used_math,
            globally_used_semantics,
            externally_forbidden,
            tokenizer,
            SEQUENCE_LENGTH,
        )
        math_splits[split] = generated
        math_generation_stats[split] = stats
    mosaic_splits: dict[str, list[dict[str, Any]]] = {}
    mosaic_generation_stats: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        generated, stats = _generate_mosaic_rows(
            split,
            counts[split],
            dialogue_splits[split],
            math_splits[split],
            seed,
            globally_used_semantics,
            externally_forbidden,
            tokenizer,
            SEQUENCE_LENGTH,
        )
        mosaic_splits[split] = generated
        mosaic_generation_stats[split] = stats
    token_checked_rows = 0
    maximum_observed_turn_tokens = 0
    for split in SPLITS:
        pairs = [
            (str(component["user"]), str(component["assistant"]))
            for component in (*dialogue_splits[split], *math_splits[split])
        ]
        pairs.extend((str(row["user"]), str(row["assistant"])) for row in mosaic_splits[split])
        for user, assistant in pairs:
            issue = _pair_token_issue(tokenizer, user, assistant, SEQUENCE_LENGTH)
            if issue is not None:
                raise RuntimeError(f"pre-write tokenizer validation failed closed: {issue}")
            encoded, _ = tokenizer.encode_turn(user, assistant)
            maximum_observed_turn_tokens = max(maximum_observed_turn_tokens, len(encoded))
            token_checked_rows += 1
    identifiers, ownership, dialogue_replay_exemptions = _all_identifiers(
        dialogue_splits, math_splits, mosaic_splits
    )
    forbidden: set[str] = set()
    for manifest_path in forbidden_manifests:
        forbidden.update(_load_forbidden_ids(manifest_path))
    collisions = sorted(set(identifiers) & forbidden)
    if collisions:
        raise ValueError(f"forbidden corpus collision ({len(collisions)} ids): {collisions[:3]}")
    protected_external_ids = set(identifiers) - dialogue_replay_exemptions
    external_collisions = sorted(protected_external_ids & externally_forbidden)
    if external_collisions:
        raise ValueError(
            f"external v70/v71 corpus collision ({len(external_collisions)} ids): "
            f"{external_collisions[:3]}"
        )
    replay_overlaps = sorted(dialogue_replay_exemptions & externally_forbidden)

    file_payloads: dict[str, bytes] = {}
    for split in SPLITS:
        file_payloads[f"{split}.jsonl"] = _jsonl_bytes(mosaic_splits[split])
        file_payloads[f"{split}_dialogue.jsonl"] = _jsonl_bytes(_atomic_rows(split, dialogue_splits[split]))
        file_payloads[f"{split}_math.jsonl"] = _jsonl_bytes(_atomic_rows(split, math_splits[split]))
    id_file = "content_ids.txt"
    file_payloads[id_file] = ("\n".join(identifiers) + "\n").encode("utf-8")

    for relative, payload in file_payloads.items():
        _atomic_write(output_dir / relative, payload)

    files = {
        relative: {"sha256": sha256_bytes(payload), "bytes": len(payload)}
        for relative, payload in sorted(file_payloads.items())
    }
    source_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "seed": seed,
        "source_dialogue": {
            "path": dialogue_jsonl.name,
            "sha256": sha256_file(dialogue_jsonl),
            "filter": dialogue_filter,
        },
        "generator_sha256": sha256_file(source_path),
        "tokenizer_binding": {
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "tokenizer_sha256": actual_tokenizer_hash,
            "sequence_length": SEQUENCE_LENGTH,
            "unknown_token_policy": "reject_before_split_or_write",
        },
        "split_policy": {
            "dialogue": "hash-sort then 80/10/10 with a minimum of two dev and holdout components",
            "math": "split-scoped deterministic PRNG with global semantic-id rejection",
            "calibration": "fixed lowest component ids from train atomics only; dev is never used for expert selection",
            "holdout_role": "sealed final evaluation only; never ranks checkpoints and is forbidden for calibration",
        },
        "counts": {
            split: {
                "mosaic": len(mosaic_splits[split]),
                "dialogue": len(dialogue_splits[split]),
                "math": len(math_splits[split]),
                "kinds": dict(sorted(Counter(row["kind"] for row in mosaic_splits[split]).items())),
            }
            for split in SPLITS
        },
        "generation_filter": {
            "math": math_generation_stats,
            "mosaic": mosaic_generation_stats,
            "policy": "invalid candidates are counted and deterministically resampled before any split file is written",
        },
        "prewrite_token_validation": {
            "checked_rows": token_checked_rows,
            "unknown_rows": 0,
            "overlength_rows": 0,
            "maximum_observed_turn_tokens": maximum_observed_turn_tokens,
        },
        "id_file": id_file,
        "id_count": len(identifiers),
        "id_set_sha256": sha256_bytes(("\n".join(identifiers) + "\n").encode("utf-8")),
        "cross_split_collision_count": 0,
        "forbidden_manifest_sha256": [sha256_file(path) for path in forbidden_manifests],
        "external_corpus_scans": external_corpus_scans,
        "external_corpus_collision_count": 0,
        "dialogue_replay_exemption": {
            "scope": "atomic dialogue prompts sourced from the approved replay corpus only",
            "rationale": "legacy dialogue replay is intentional; Mosaic prompts, generated math prompts, and math operand tuples are never exempt",
            "identifier_count": len(dialogue_replay_exemptions),
            "identifier_set_sha256": sha256_bytes(
                ("\n".join(sorted(dialogue_replay_exemptions)) + "\n").encode("utf-8")
            ),
            "external_overlap_count": len(replay_overlaps),
            "external_overlap_set_sha256": sha256_bytes(
                ("\n".join(replay_overlaps) + ("\n" if replay_overlaps else "")).encode("utf-8")
            ),
        },
        "files": files,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    _atomic_write(output_dir / "manifest.json", manifest_bytes)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogue-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--v70-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v70-checkpoint-sha256", required=True)
    parser.add_argument("--expected-v70-tokenizer-sha256", required=True)
    parser.add_argument("--seed", type=int, default=710_413)
    parser.add_argument("--train-count", type=int, default=12_000)
    parser.add_argument("--dev-count", type=int, default=1_200)
    parser.add_argument("--holdout-count", type=int, default=1_200)
    parser.add_argument("--forbidden-manifest", type=Path, action="append", default=[])
    parser.add_argument("--forbidden-corpus-jsonl", type=Path, action="append", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    tokenizer = load_bound_v70_tokenizer(
        args.v70_checkpoint,
        expected_checkpoint_sha256=args.expected_v70_checkpoint_sha256,
        expected_tokenizer_sha256=args.expected_v70_tokenizer_sha256,
    )
    manifest = build_bundle(
        args.dialogue_jsonl,
        args.output_dir,
        tokenizer=tokenizer,
        parent_checkpoint_sha256=args.expected_v70_checkpoint_sha256,
        expected_tokenizer_sha256=args.expected_v70_tokenizer_sha256,
        seed=args.seed,
        train_count=args.train_count,
        dev_count=args.dev_count,
        holdout_count=args.holdout_count,
        forbidden_manifests=args.forbidden_manifest,
        forbidden_corpora=args.forbidden_corpus_jsonl,
    )
    print(canonical_json({"status": "built", "manifest": str(args.output_dir / "manifest.json"), "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
