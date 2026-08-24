from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from source.build_general_intelligence_curriculum import (
    ADVANCED_FAMILIES,
    CURRICULUM_SCHEMA_VERSION,
    EVAL_FILENAME,
    MANIFEST_FILENAME,
    TRAIN_FILENAME,
    _apply_quantity_transitions,
    _format_exact_quantity,
    _validate_rows,
    build_curriculum,
    write_curriculum,
)
from source.logical_entailment import (
    LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
    LOGICAL_ENTAILMENT_ORACLE_ID,
    canonical_task_ir_json,
    derive_entailment_answer,
    parse_canonical_task_ir_json,
    task_ir_from_prompt,
)
from source.reasoning_engine import solve_problem
from source.verifiable_reasoning import verify_candidate


def _jsonl_hash(rows: tuple[dict[str, object], ...]) -> str:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _independent_quantity_replay(
    initial: object,
    transitions: list[dict[str, object]],
) -> tuple[Fraction, tuple[Fraction, ...]]:
    current = Fraction(Decimal(str(initial)))
    states: list[Fraction] = []
    for transition in transitions:
        amount = Fraction(Decimal(str(transition["amount"])))
        delta = current * amount / 100 if transition["kind"] == "percentage" else amount
        current = current + delta if transition["direction"] == "increase" else current - delta
        states.append(current)
    return current, tuple(states)


def test_general_curriculum_is_deterministic_disjoint_and_verified() -> None:
    first = build_curriculum(seed=77, train_rows=75, eval_rows=45)
    second = build_curriculum(seed=77, train_rows=75, eval_rows=45)
    assert first == second
    assert len(first.train_rows) == 75
    assert len(first.eval_rows) == 45
    assert CURRICULUM_SCHEMA_VERSION == "supermix-general-intelligence-curriculum-v3"
    assert first.manifest["curriculum_schema"] == CURRICULUM_SCHEMA_VERSION

    train_templates = {str(row["metadata"]["template_id"]) for row in first.train_rows}  # type: ignore[index]
    eval_templates = {str(row["metadata"]["template_id"]) for row in first.eval_rows}  # type: ignore[index]
    train_prompts = {str(row["user"]).casefold() for row in first.train_rows}
    eval_prompts = {str(row["user"]).casefold() for row in first.eval_rows}
    assert train_templates.isdisjoint(eval_templates)
    assert train_prompts.isdisjoint(eval_prompts)

    for split, rows in (("train", first.train_rows), ("eval", first.eval_rows)):
        for row in rows:
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            assert metadata["curriculum_split"] == split
            result = verify_candidate(row["user"], row["assistant"], metadata)
            assert result.valid_spec
            assert result.passed

    assert first.manifest["train"]["sha256"] == _jsonl_hash(first.train_rows)  # type: ignore[index]
    assert first.manifest["eval"]["sha256"] == _jsonl_hash(first.eval_rows)  # type: ignore[index]


def test_general_curriculum_covers_every_advanced_and_inherited_family() -> None:
    bundle = build_curriculum(seed=91, train_rows=150, eval_rows=75)
    train_families = Counter(
        str(row["metadata"]["problem_family"]) for row in bundle.train_rows  # type: ignore[index]
    )
    eval_families = Counter(
        str(row["metadata"]["problem_family"]) for row in bundle.eval_rows  # type: ignore[index]
    )
    for family in ADVANCED_FAMILIES:
        assert train_families[family] > 0
        assert eval_families[family] > 0
    assert train_families["multi_step_arithmetic"] > 0
    assert train_families["typo_noise_robustness"] > 0


def test_general_curriculum_writes_content_addressed_artifacts(tmp_path: Path) -> None:
    bundle = build_curriculum(seed=13, train_rows=75, eval_rows=45)
    paths = write_curriculum(bundle, tmp_path)
    assert Path(paths["train_jsonl"]).name == TRAIN_FILENAME
    assert Path(paths["eval_jsonl"]).name == EVAL_FILENAME
    assert Path(paths["manifest_json"]).name == MANIFEST_FILENAME
    assert len(Path(paths["train_jsonl"]).read_text(encoding="utf-8").splitlines()) == 75
    manifest = json.loads(Path(paths["manifest_json"]).read_text(encoding="utf-8"))
    assert manifest["all_targets_verified"] is True
    assert manifest["template_ids_disjoint"] is True
    with pytest.raises(FileExistsError):
        write_curriculum(bundle, tmp_path)


def test_quantity_transitions_generalize_across_disjoint_surfaces_and_orders() -> None:
    bundle = build_curriculum(seed=103, train_rows=120, eval_rows=60)
    family = "quantity_transition_reasoning"
    train_rows = [
        row for row in bundle.train_rows if row["metadata"]["problem_family"] == family  # type: ignore[index]
    ]
    eval_rows = [
        row for row in bundle.eval_rows if row["metadata"]["problem_family"] == family  # type: ignore[index]
    ]
    assert train_rows and eval_rows

    train_templates = {str(row["metadata"]["template_id"]) for row in train_rows}  # type: ignore[index]
    eval_templates = {str(row["metadata"]["template_id"]) for row in eval_rows}  # type: ignore[index]
    train_surfaces = {str(row["metadata"]["surface_domain"]) for row in train_rows}  # type: ignore[index]
    eval_surfaces = {str(row["metadata"]["surface_domain"]) for row in eval_rows}  # type: ignore[index]
    assert train_templates.isdisjoint(eval_templates)
    assert train_surfaces.isdisjoint(eval_surfaces)

    patterns: dict[str, dict[int, set[tuple[str, ...]]]] = {
        "train": {2: set(), 3: set(), 4: set()},
        "eval": {2: set(), 3: set(), 4: set()},
    }
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        assert {int(row["metadata"]["transition_count"]) for row in rows} == {2, 3, 4}  # type: ignore[index]
        for row in rows:
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            transitions = json.loads(str(metadata["transition_plan_json"]))
            assert isinstance(transitions, list)
            count = int(metadata["transition_count"])
            assert len(transitions) == count
            assert {transition["kind"] for transition in transitions} == {
                "percentage",
                "fixed",
            }
            final, states = _apply_quantity_transitions(
                metadata["initial_quantity"],
                transitions,
            )
            independent_final, independent_states = _independent_quantity_replay(
                metadata["initial_quantity"],
                transitions,
            )
            assert all(state >= Decimal(0) for state in states)
            assert independent_final == Fraction(final)
            assert independent_states == tuple(Fraction(state) for state in states)
            assert _format_exact_quantity(final) == metadata["expected_answer"]
            assert metadata["ordered_transition_semantics"] == "percentage_of_current_state"
            verified = verify_candidate(row["user"], row["assistant"], metadata)
            assert verified.valid_spec and verified.passed

            reverse_final, _ = _apply_quantity_transitions(
                metadata["initial_quantity"],
                list(reversed(transitions)),
            )
            assert reverse_final != final
            wrong = verify_candidate(
                row["user"],
                f"Final answer: {_format_exact_quantity(reverse_final)}.",
                metadata,
            )
            assert wrong.valid_spec and not wrong.passed
            runtime_result = solve_problem(row["user"])
            assert runtime_result["problem_class"] == "quantity_transition"
            assert Fraction(runtime_result["answer"]["exact"]) == independent_final
            patterns[split][count].add(
                tuple(str(transition["kind"]) for transition in transitions)
            )

    for count in (2, 3, 4):
        assert patterns["train"][count]
        assert patterns["eval"][count]
        assert patterns["train"][count].isdisjoint(patterns["eval"][count])


def test_quantity_transition_has_a_hand_worked_exact_golden_case() -> None:
    transitions = [
        {"kind": "percentage", "direction": "decrease", "amount": "25"},
        {"kind": "fixed", "direction": "increase", "amount": "15"},
    ]
    final, states = _apply_quantity_transitions("120", transitions)
    assert states == (Decimal("90"), Decimal("105"))
    assert final == Decimal("105")
    assert _format_exact_quantity(final) == "105"
    independent_final, independent_states = _independent_quantity_replay("120", transitions)
    assert independent_states == (Fraction(90), Fraction(105))
    assert independent_final == Fraction(105)


@pytest.mark.parametrize(
    ("initial", "transitions", "message"),
    (
        (
            "100",
            [{"kind": "fixed", "direction": "increase", "amount": "5"}],
            "2 to 4",
        ),
        (
            "100",
            [
                {"kind": "percentage", "direction": "increase", "amount": "10"},
                {"kind": "percentage", "direction": "decrease", "amount": "5"},
            ],
            "mix percentage and fixed",
        ),
        (
            "10",
            [
                {"kind": "fixed", "direction": "decrease", "amount": "20"},
                {"kind": "percentage", "direction": "increase", "amount": "10"},
            ],
            "negative state",
        ),
        (
            "100",
            [
                {"kind": "percentage", "direction": "decrease", "amount": "100"},
                {"kind": "fixed", "direction": "increase", "amount": "5"},
            ],
            "below 100 percent",
        ),
    ),
)
def test_quantity_transition_invariants_fail_closed(
    initial: str,
    transitions: list[dict[str, str]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _apply_quantity_transitions(initial, transitions)


def test_logical_entailment_has_semantically_disjoint_model_checked_holdouts() -> None:
    bundle = build_curriculum(seed=307, train_rows=180, eval_rows=90)
    family = "logical_entailment"
    split_sets: dict[str, dict[str, set[str]]] = {}
    for split, all_rows in (("train", bundle.train_rows), ("eval", bundle.eval_rows)):
        rows = [
            row
            for row in all_rows
            if row["metadata"]["problem_family"] == family  # type: ignore[index]
        ]
        assert rows
        atoms: set[str] = set()
        topologies: set[str] = set()
        surfaces: set[str] = set()
        answers: set[str] = set()
        templates: set[str] = set()
        for row in rows:
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            task = parse_canonical_task_ir_json(metadata["task_ir_json"])
            assert task_ir_from_prompt(row["user"]) == task
            assert metadata["task_ir_schema"] == LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION
            assert metadata["oracle_id"] == LOGICAL_ENTAILMENT_ORACLE_ID
            assert metadata["expected_answer"] == derive_entailment_answer(task)
            assert row["assistant"] == metadata["expected_answer"]
            assert metadata["expected_answer"] in {"entailed", "not entailed"}
            assert verify_candidate(row["user"], row["assistant"], metadata).passed
            atoms.update(str(atom) for atom in json.loads(metadata["atom_vocabulary_json"]))
            topologies.add(str(metadata["rule_topology"]))
            surfaces.add(str(metadata["surface_marker"]))
            answers.add(str(metadata["expected_answer"]))
            templates.add(str(metadata["template_id"]))
        assert answers == {"entailed", "not entailed"}
        split_sets[split] = {
            "atoms": atoms,
            "topologies": topologies,
            "surfaces": surfaces,
            "templates": templates,
        }

    for key in ("atoms", "topologies", "surfaces", "templates"):
        assert split_sets["train"][key].isdisjoint(split_sets["eval"][key])
    holdout = bundle.manifest["semantic_holdouts"][family]  # type: ignore[index]
    assert holdout["atom_vocabularies_disjoint"] is True
    assert holdout["rule_topologies_disjoint"] is True
    assert holdout["surface_markers_disjoint"] is True


def test_logical_entailment_answer_and_ir_tampering_fail_closed() -> None:
    bundle = build_curriculum(seed=401, train_rows=90, eval_rows=60)
    original = next(
        row
        for row in bundle.train_rows
        if row["metadata"]["problem_family"] == "logical_entailment"  # type: ignore[index]
        and row["metadata"]["rule_topology"] == "seeded_unary_ladder"  # type: ignore[index]
    )

    answer_tampered = deepcopy(original)
    answer_tampered["metadata"]["expected_answer"] = "not entailed"  # type: ignore[index]
    answer_tampered["assistant"] = "not entailed"
    with pytest.raises(ValueError, match="Target failed verifier"):
        _validate_rows([answer_tampered], "train")

    ir_tampered = deepcopy(original)
    metadata = ir_tampered["metadata"]
    assert isinstance(metadata, dict)
    task = parse_canonical_task_ir_json(metadata["task_ir_json"])
    facts = set(str(atom) for atom in task["facts"])
    old_query = str(task["query"])
    replacement = next(
        str(rule["then"])
        for rule in task["rules"]  # type: ignore[union-attr]
        if str(rule["then"]) not in facts and str(rule["then"]) != old_query
    )
    task["query"] = replacement
    metadata["task_ir_json"] = canonical_task_ir_json(task)
    metadata["expected_answer"] = derive_entailment_answer(task)
    ir_tampered["assistant"] = metadata["expected_answer"]
    with pytest.raises(ValueError, match="Target failed verifier"):
        _validate_rows([ir_tampered], "train")


def test_general_curriculum_requires_coverage_budget() -> None:
    with pytest.raises(ValueError, match="at least 17"):
        build_curriculum(train_rows=15, eval_rows=45)
