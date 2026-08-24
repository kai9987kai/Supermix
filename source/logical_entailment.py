"""Bounded, non-executable verification for positive-Horn entailment tasks.

The task format deliberately has a tiny explicit grammar::

    Facts: atom, atom. Rules: a -> b; b & c -> d. Query: d.

Answers are derived by exhaustive Boolean-model enumeration.  The oracle does
not trust a stored expected answer, execute candidate text, or use the response
verifier's string-matching logic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION = "supermix-logical-entailment-ir-v1"
LOGICAL_ENTAILMENT_ORACLE_ID = "exhaustive-positive-horn-models-v1"
LOGICAL_ENTAILMENT_ANSWERS = ("entailed", "not entailed")

_ATOM_RE = re.compile(r"[a-z]{2,12}")
_STATEMENT_RE = re.compile(
    r"Facts: (?P<facts>[a-z]{2,12}(?:, [a-z]{2,12})*)\. "
    r"Rules: (?P<rules>.+)\. Query: (?P<query>[a-z]{2,12})\."
)
_RULE_RE = re.compile(
    r"(?P<premises>[a-z]{2,12}(?: & [a-z]{2,12}){0,2})"
    r" -> (?P<conclusion>[a-z]{2,12})"
)
_MAX_ATOMS = 10
_MAX_FACTS = 4
_MAX_RULES = 8
_MAX_PREMISES = 3
_MAX_TASK_TEXT_CHARS = 10_000


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Logical-entailment JSON repeats key {key!r}.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Logical-entailment JSON constant {value!r} is not supported.")


def _atom(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _ATOM_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be one lower-case opaque atom.")
    return value


def normalize_task_ir(value: Mapping[str, object]) -> dict[str, object]:
    """Validate and normalize one bounded positive-Horn task IR."""

    if not isinstance(value, Mapping):
        raise ValueError("Logical-entailment task IR must be a mapping.")
    if set(value) != {"schema", "facts", "rules", "query"}:
        raise ValueError("Logical-entailment task IR has unexpected fields.")
    if value.get("schema") != LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION:
        raise ValueError("Logical-entailment task IR schema is unsupported.")

    raw_facts = value.get("facts")
    if (
        isinstance(raw_facts, (str, bytes))
        or not isinstance(raw_facts, Sequence)
        or not 1 <= len(raw_facts) <= _MAX_FACTS
    ):
        raise ValueError(f"Logical-entailment facts require 1 to {_MAX_FACTS} atoms.")
    facts = [_atom(item, label="Logical-entailment fact") for item in raw_facts]
    if len(set(facts)) != len(facts) or facts != sorted(facts):
        raise ValueError("Logical-entailment facts must be unique and sorted.")

    raw_rules = value.get("rules")
    if (
        isinstance(raw_rules, (str, bytes))
        or not isinstance(raw_rules, Sequence)
        or not 1 <= len(raw_rules) <= _MAX_RULES
    ):
        raise ValueError(f"Logical-entailment rules require 1 to {_MAX_RULES} clauses.")
    rules: list[dict[str, object]] = []
    rule_keys: set[tuple[tuple[str, ...], str]] = set()
    for index, raw_rule in enumerate(raw_rules, start=1):
        if not isinstance(raw_rule, Mapping) or set(raw_rule) != {"if", "then"}:
            raise ValueError(f"Logical-entailment rule {index} has unexpected fields.")
        raw_premises = raw_rule.get("if")
        if (
            isinstance(raw_premises, (str, bytes))
            or not isinstance(raw_premises, Sequence)
            or not 1 <= len(raw_premises) <= _MAX_PREMISES
        ):
            raise ValueError(
                f"Logical-entailment rule {index} requires 1 to {_MAX_PREMISES} premises."
            )
        premises = [
            _atom(item, label=f"Logical-entailment rule {index} premise")
            for item in raw_premises
        ]
        if len(set(premises)) != len(premises) or premises != sorted(premises):
            raise ValueError(
                f"Logical-entailment rule {index} premises must be unique and sorted."
            )
        conclusion = _atom(
            raw_rule.get("then"),
            label=f"Logical-entailment rule {index} conclusion",
        )
        if conclusion in premises:
            raise ValueError(f"Logical-entailment rule {index} is tautological.")
        rule_key = (tuple(premises), conclusion)
        if rule_key in rule_keys:
            raise ValueError("Logical-entailment task IR repeats a rule.")
        rule_keys.add(rule_key)
        rules.append({"if": premises, "then": conclusion})
    rules.sort(key=lambda rule: (str(rule["then"]), tuple(rule["if"])))

    query = _atom(value.get("query"), label="Logical-entailment query")
    if query in facts:
        raise ValueError("Logical-entailment query must require reasoning beyond a stated fact.")

    atoms = set(facts)
    atoms.add(query)
    for rule in rules:
        atoms.update(rule["if"])  # type: ignore[arg-type]
        atoms.add(str(rule["then"]))
    if not 3 <= len(atoms) <= _MAX_ATOMS:
        raise ValueError(f"Logical-entailment tasks require 3 to {_MAX_ATOMS} atoms.")

    return {
        "schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        "facts": facts,
        "rules": rules,
        "query": query,
    }


def canonical_task_ir_json(value: Mapping[str, object]) -> str:
    """Return the unique compact JSON representation of a valid task IR."""

    return json.dumps(
        normalize_task_ir(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def parse_canonical_task_ir_json(value: object) -> dict[str, object]:
    """Parse task IR while rejecting duplicate keys and non-canonical encodings."""

    if not isinstance(value, str) or not value or len(value) > _MAX_TASK_TEXT_CHARS:
        raise ValueError("Logical-entailment task IR JSON must be non-empty text.")
    try:
        decoded = json.loads(
            value,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Logical-entailment task IR JSON is invalid.") from exc
    normalized = normalize_task_ir(decoded)
    if canonical_task_ir_json(normalized) != value:
        raise ValueError("Logical-entailment task IR JSON is not canonical.")
    return normalized


def render_task_statement(value: Mapping[str, object]) -> str:
    """Render a validated IR using the runtime's strict text grammar."""

    task = normalize_task_ir(value)
    facts = ", ".join(str(atom) for atom in task["facts"])
    rendered_rules = []
    for rule in task["rules"]:  # type: ignore[union-attr]
        premises = " & ".join(str(atom) for atom in rule["if"])
        rendered_rules.append(f"{premises} -> {rule['then']}")
    return (
        f"Facts: {facts}. Rules: {'; '.join(rendered_rules)}. "
        f"Query: {task['query']}."
    )


def parse_task_statement(value: object) -> dict[str, object]:
    """Parse one exact runtime-grammar statement into canonical task IR."""

    if not isinstance(value, str) or len(value) > _MAX_TASK_TEXT_CHARS:
        raise ValueError("Logical-entailment statement must be text.")
    match = _STATEMENT_RE.fullmatch(value)
    if match is None:
        raise ValueError("Logical-entailment statement does not match the strict grammar.")
    facts = match.group("facts").split(", ")
    rules: list[dict[str, object]] = []
    for raw_rule in match.group("rules").split("; "):
        rule_match = _RULE_RE.fullmatch(raw_rule)
        if rule_match is None:
            raise ValueError("Logical-entailment rule does not match the strict grammar.")
        rules.append(
            {
                "if": rule_match.group("premises").split(" & "),
                "then": rule_match.group("conclusion"),
            }
        )
    task = normalize_task_ir(
        {
            "schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
            "facts": facts,
            "rules": rules,
            "query": match.group("query"),
        }
    )
    if render_task_statement(task) != value:
        raise ValueError("Logical-entailment statement is not canonical.")
    return task


def task_ir_from_prompt(value: object) -> dict[str, object]:
    """Recover the canonical task IR from the final non-empty prompt line."""

    if not isinstance(value, str) or len(value) > _MAX_TASK_TEXT_CHARS:
        raise ValueError("Logical-entailment prompt must be text.")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Logical-entailment prompt is empty.")
    grammar_lines = [line for line in lines if line.startswith("Facts:")]
    if len(grammar_lines) != 1 or grammar_lines[0] != lines[-1]:
        raise ValueError("Logical-entailment prompt requires one final grammar statement.")
    return parse_task_statement(grammar_lines[0])


def validate_prompt_task_ir(prompt: object, task_ir_json: object) -> dict[str, object]:
    """Require the prompt grammar and separately stored canonical IR to agree exactly."""

    metadata_task = parse_canonical_task_ir_json(task_ir_json)
    prompt_task = task_ir_from_prompt(prompt)
    if canonical_task_ir_json(prompt_task) != canonical_task_ir_json(metadata_task):
        raise ValueError("Logical-entailment prompt and task IR disagree.")
    return metadata_task


def exhaustive_model_entailment(value: Mapping[str, object]) -> bool:
    """Return whether every positive-Horn model of the task satisfies its query."""

    task = normalize_task_ir(value)
    atoms = set(task["facts"])
    atoms.add(str(task["query"]))
    for rule in task["rules"]:  # type: ignore[union-attr]
        atoms.update(rule["if"])
        atoms.add(str(rule["then"]))
    ordered_atoms = sorted(str(atom) for atom in atoms)
    atom_indexes = {atom: index for index, atom in enumerate(ordered_atoms)}
    facts = tuple(str(atom) for atom in task["facts"])
    rules = tuple(
        (tuple(str(atom) for atom in rule["if"]), str(rule["then"]))
        for rule in task["rules"]  # type: ignore[union-attr]
    )
    query = str(task["query"])

    found_model = False
    for bits in range(1 << len(ordered_atoms)):
        def is_true(atom: str) -> bool:
            return bool(bits & (1 << atom_indexes[atom]))

        if any(not is_true(fact) for fact in facts):
            continue
        if any(
            all(is_true(premise) for premise in premises) and not is_true(conclusion)
            for premises, conclusion in rules
        ):
            continue
        found_model = True
        if not is_true(query):
            return False
    if not found_model:  # Positive Horn theories always have the all-true model.
        raise RuntimeError("Logical-entailment theory unexpectedly has no Boolean model.")
    return True


def derive_entailment_answer(value: Mapping[str, object]) -> str:
    """Derive the exact answer token from exhaustive model semantics."""

    return "entailed" if exhaustive_model_entailment(value) else "not entailed"


__all__ = [
    "LOGICAL_ENTAILMENT_ANSWERS",
    "LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION",
    "LOGICAL_ENTAILMENT_ORACLE_ID",
    "canonical_task_ir_json",
    "derive_entailment_answer",
    "exhaustive_model_entailment",
    "normalize_task_ir",
    "parse_canonical_task_ir_json",
    "parse_task_statement",
    "render_task_statement",
    "task_ir_from_prompt",
    "validate_prompt_task_ir",
]
