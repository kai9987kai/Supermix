"""Build a mixed, held-out curriculum for trained general capability.

The mixture deliberately combines three kinds of supervision:

* verifier-grounded maths, logic, probability, and evidence-in-prompt tasks;
* deterministic prompt-understanding tasks; and
* new science, compositional quantity-transition, model-checked logical
  entailment, calibrated-prediction, causal-evidence, and conversational tasks.

Train and evaluation template identifiers and prompt text are disjoint.  Every
target is checked by ``supermix-verifier-v2`` before it can be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

try:
    from build_prompt_understanding_curriculum import (
        build_curriculum as build_prompt_curriculum,
    )
    from build_verifiable_reasoning_curriculum import (
        build_curriculum as build_reasoning_curriculum,
    )
    from logical_entailment import (
        LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        LOGICAL_ENTAILMENT_ORACLE_ID,
        canonical_task_ir_json,
        derive_entailment_answer,
        parse_canonical_task_ir_json,
        render_task_statement,
    )
    from verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate
except ImportError:  # pragma: no cover - package import path
    from .build_prompt_understanding_curriculum import (
        build_curriculum as build_prompt_curriculum,
    )
    from .build_verifiable_reasoning_curriculum import (
        build_curriculum as build_reasoning_curriculum,
    )
    from .logical_entailment import (
        LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        LOGICAL_ENTAILMENT_ORACLE_ID,
        canonical_task_ir_json,
        derive_entailment_answer,
        parse_canonical_task_ir_json,
        render_task_statement,
    )
    from .verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate


CURRICULUM_SCHEMA_VERSION = "supermix-general-intelligence-curriculum-v3"
CURRICULUM_SOURCE = "supermix_general_intelligence_v3"
TRAIN_FILENAME = "general_intelligence_train.jsonl"
EVAL_FILENAME = "general_intelligence_eval.jsonl"
MANIFEST_FILENAME = "general_intelligence_manifest.json"

ADVANCED_FAMILIES: Tuple[str, ...] = (
    "quantitative_science",
    "quantity_transition_reasoning",
    "logical_entailment",
    "calibrated_prediction",
    "causal_evidence",
    "conversation_constraints",
    "multi_turn_instruction",
)
_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class GeneralCurriculumBundle:
    train_rows: Tuple[dict[str, object], ...]
    eval_rows: Tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _metadata(
    *,
    prompt: str,
    split: str,
    family: str,
    template_id: str,
    verifier_type: str,
    expected_answer: str,
    difficulty: float,
    aliases: Sequence[str] = (),
    required_terms: Sequence[str] = (),
    forbidden_terms: Sequence[str] = (),
    exact_bullet_count: int = 0,
    max_words_per_bullet: int = 0,
) -> dict[str, object]:
    example_id = hashlib.sha256(
        f"{split}|{family}|{template_id}|{prompt}".encode("utf-8")
    ).hexdigest()[:24]
    payload: dict[str, object] = {
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "verifier_type": verifier_type,
        "expected_answer": expected_answer,
        "aliases_json": json.dumps(list(aliases), ensure_ascii=False, separators=(",", ":")),
        "absolute_tolerance": "0",
        "problem_family": family,
        "template_id": template_id,
        "split_group": f"{split}:{template_id}",
        "curriculum_split": split,
        "verifier_difficulty": float(max(0.0, min(1.0, difficulty))),
        "verified_correct": True,
        "rule_reward": 1.0,
        "example_id": example_id,
    }
    if verifier_type == "response_contract":
        payload.update(
            {
                "required_terms_json": json.dumps(
                    list(required_terms), ensure_ascii=False, separators=(",", ":")
                ),
                "forbidden_terms_json": json.dumps(
                    list(forbidden_terms), ensure_ascii=False, separators=(",", ":")
                ),
                "exact_bullet_count": int(exact_bullet_count),
                "max_words_per_bullet": int(max_words_per_bullet),
            }
        )
    return payload


def _row(prompt: str, assistant: str, metadata: Mapping[str, object]) -> dict[str, object]:
    row = {
        "user": prompt,
        "assistant": assistant,
        "source": CURRICULUM_SOURCE,
        "metadata": dict(metadata),
    }
    result = verify_candidate(prompt, assistant, row["metadata"])
    if not result.valid_spec or not result.passed:
        raise ValueError(
            f"Generated target failed verification: {result.reason} "
            f"({row['metadata'].get('example_id', '-')})"
        )
    return row


def _generate_quantitative_science(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    variant = index % 3
    if variant == 0:
        mass = rng.randint(2, 24)
        acceleration = rng.randint(2, 12)
        answer = mass * acceleration
        setting = "bench cart" if split == "train" else "thruster sled"
        template_id = f"{prefix}.science.force_mass_acceleration.v1"
        prompt = (
            f"Science case {prefix.upper()}-F{index + 1:04d}: A {setting} has mass {mass} kg "
            f"and accelerates at {acceleration} m/s^2. "
            "Ignoring friction, what net force acts on it in newtons?"
        )
        assistant = (
            f"Use F = ma. F = {mass} x {acceleration} = {answer} N. "
            f"Final answer: {answer}."
        )
        difficulty = 0.38
    elif variant == 1:
        voltage = rng.randint(3, 24)
        current = rng.randint(2, 12)
        answer = voltage * current
        setting = "test supply" if split == "train" else "field circuit"
        template_id = f"{prefix}.science.electrical_power.v1"
        prompt = (
            f"Science case {prefix.upper()}-P{index + 1:04d}: A {setting} operates at "
            f"{voltage} volts and {current} amperes. "
            "Using P = VI, what electrical power does it draw in watts?"
        )
        assistant = (
            f"Apply P = VI: {voltage} x {current} = {answer} W. "
            f"Final answer: {answer}."
        )
        difficulty = 0.42
    else:
        volume = rng.randint(2, 15)
        density = rng.randint(2, 18)
        mass = volume * density
        answer = density
        setting = "alloy sample" if split == "train" else "mineral core"
        template_id = f"{prefix}.science.density_mass_volume.v1"
        prompt = (
            f"Science case {prefix.upper()}-D{index + 1:04d}: A {setting} has mass "
            f"{mass} g and volume {volume} cm^3. "
            "What is its density in g/cm^3?"
        )
        assistant = (
            f"Density is mass divided by volume: {mass} / {volume} = {answer} g/cm^3. "
            f"Final answer: {answer}."
        )
        difficulty = 0.40

    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="quantitative_science",
        template_id=template_id,
        verifier_type="integer",
        expected_answer=str(answer),
        difficulty=difficulty,
    )
    return _row(prompt, assistant, metadata)


def _exact_decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite decimal quantity.")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a finite decimal quantity.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{label} must be a finite decimal quantity.")
    return parsed


def _format_exact_quantity(value: Decimal) -> str:
    parsed = _exact_decimal(value, label="quantity")
    text = format(parsed.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _apply_quantity_transitions(
    initial: object,
    transitions: Sequence[Mapping[str, object]],
) -> tuple[Decimal, tuple[Decimal, ...]]:
    """Apply a bounded mixed transition plan to the current state exactly."""

    state = _exact_decimal(initial, label="initial quantity")
    if state <= 0:
        raise ValueError("Initial quantity must be positive.")
    if isinstance(transitions, (str, bytes)) or not isinstance(transitions, Sequence):
        raise TypeError("Quantity transitions must be a sequence of mappings.")
    if not 2 <= len(transitions) <= 4:
        raise ValueError("Quantity transition plans require 2 to 4 ordered changes.")

    kinds: set[str] = set()
    states: list[Decimal] = []
    hundred = Decimal(100)
    for step, raw_transition in enumerate(transitions, start=1):
        if not isinstance(raw_transition, Mapping):
            raise ValueError(f"Quantity transition {step} must be a mapping.")
        kind = str(raw_transition.get("kind") or "").strip().lower()
        direction = str(raw_transition.get("direction") or "").strip().lower()
        if kind not in {"percentage", "fixed"}:
            raise ValueError(f"Quantity transition {step} has an unsupported kind.")
        if direction not in {"increase", "decrease"}:
            raise ValueError(f"Quantity transition {step} has an unsupported direction.")
        amount = _exact_decimal(
            raw_transition.get("amount"),
            label=f"quantity transition {step} amount",
        )
        if amount <= 0:
            raise ValueError(f"Quantity transition {step} amount must be positive.")
        if kind == "percentage":
            if amount >= hundred:
                raise ValueError("Percentage changes must be below 100 percent.")
            delta = state * amount / hundred
        else:
            delta = amount
        state = state + delta if direction == "increase" else state - delta
        if state < 0:
            raise ValueError(f"Quantity transition {step} produces a negative state.")
        kinds.add(kind)
        states.append(state)
    if kinds != {"percentage", "fixed"}:
        raise ValueError("Quantity transition plans must mix percentage and fixed changes.")
    return state, tuple(states)


_QUANTITY_SURFACES: Mapping[str, Tuple[Tuple[str, str, str, str], ...]] = {
    "train": (
        ("warehouse_reserve", "warehouse fluid reserve", "liters", "volume"),
        ("mixing_tank", "mixing tank", "liters", "volume"),
        ("service_credit_pool", "service credit pool", "points", "quantity"),
    ),
    "eval": (
        ("orbital_allocation", "orbital allocation buffer", "points", "quantity"),
        ("ecology_nutrients", "ecology nutrient tank", "liters", "volume"),
        ("relief_water", "relief water cache", "liters", "volume"),
    ),
}
_QUANTITY_KIND_PATTERNS: Mapping[str, Mapping[int, Tuple[str, ...]]] = {
    "train": {
        2: ("percentage", "fixed"),
        3: ("fixed", "percentage", "fixed"),
        4: ("percentage", "fixed", "percentage", "fixed"),
    },
    "eval": {
        2: ("fixed", "percentage"),
        3: ("percentage", "fixed", "percentage"),
        4: ("fixed", "percentage", "fixed", "percentage"),
    },
}


def _quantity_change_text(
    split: str,
    transition: Mapping[str, object],
    *,
    unit: str,
) -> str:
    kind = str(transition["kind"])
    direction = str(transition["direction"])
    amount = str(transition["amount"])
    if split == "train":
        if kind == "percentage":
            verb = "increase" if direction == "increase" else "decrease"
            return f"{verb} the current amount by {amount}%"
        verb = "add" if direction == "increase" else "remove"
        return f"{verb} exactly {amount} {unit}"
    if kind == "percentage":
        action = "increase" if direction == "increase" else "decrease"
        return f"apply a {amount}% {action} to the current level"
    verb = "receive" if direction == "increase" else "withdraw"
    return f"{verb} a fixed {amount} {unit}"


def _generate_quantity_transition(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    transition_count = 2 + (index % 3)
    pattern = _QUANTITY_KIND_PATTERNS[split][transition_count]
    percentage_choices = (10, 15, 20, 25, 30)

    for _attempt in range(64):
        initial = Decimal(rng.randint(180, 720))
        transitions: list[dict[str, object]] = []
        for kind in pattern:
            amount = (
                rng.choice(percentage_choices)
                if kind == "percentage"
                else rng.randint(7, 59)
            )
            transitions.append(
                {
                    "kind": kind,
                    "direction": "increase" if rng.getrandbits(1) else "decrease",
                    "amount": str(amount),
                }
            )
        final, states = _apply_quantity_transitions(initial, transitions)
        reverse_final, _reverse_states = _apply_quantity_transitions(
            initial,
            list(reversed(transitions)),
        )
        if final != reverse_final:
            break
    else:  # pragma: no cover - deterministic choices make this unreachable
        raise RuntimeError("Unable to generate an order-sensitive quantity transition plan.")

    surface_key, surface_name, unit, target_label = _QUANTITY_SURFACES[split][
        index % len(_QUANTITY_SURFACES[split])
    ]
    template_id = f"{prefix}.quantity_transition.{surface_key}.ordered_current_state.v2"
    initial_text = _format_exact_quantity(initial)
    temporal_cues = ("Then", "Next", "After that", "Finally")
    rendered_changes = " ".join(
        f"{temporal_cues[step]} {_quantity_change_text(split, transition, unit=unit)}."
        for step, transition in enumerate(transitions)
    )
    if split == "train":
        prompt = (
            f"Quantity ledger {prefix.upper()}-Q{index + 1:04d}: A {surface_name} starts "
            f"with {initial_text} {unit}. Each percentage is calculated from the current "
            f"amount produced by the preceding step. {rendered_changes} What is the final "
            f"{target_label}?"
        )
    else:
        prompt = (
            f"Held-out transition {prefix.upper()}-Q{index + 1:04d}: The {surface_name} "
            f"begins with {initial_text} {unit}. A percentage always acts on the level "
            f"produced by the previous stage. {rendered_changes} Determine the final "
            f"{target_label}."
        )

    reasoning_lines = [f"Start: {initial_text} {unit}."]
    before = initial
    for step, (transition, after) in enumerate(zip(transitions, states), start=1):
        amount = str(transition["amount"])
        direction = str(transition["direction"])
        if transition["kind"] == "percentage":
            delta = before * Decimal(amount) / Decimal(100)
            operation = f"{amount}% of {_format_exact_quantity(before)} is {_format_exact_quantity(delta)}"
        else:
            operation = f"the fixed change is {amount}"
        sign = "+" if direction == "increase" else "-"
        reasoning_lines.append(
            f"Step {step}: {operation}; {sign} gives {_format_exact_quantity(after)} {unit}."
        )
        before = after
    answer = _format_exact_quantity(final)
    reasoning_lines.append(f"Final answer: {answer} {unit}.")
    assistant = "\n".join(reasoning_lines)
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="quantity_transition_reasoning",
        template_id=template_id,
        verifier_type="decimal",
        expected_answer=answer,
        difficulty=0.68 + 0.05 * (transition_count - 2),
    )
    metadata.update(
        {
            "initial_quantity": initial_text,
            "transition_count": transition_count,
            "transition_plan_json": json.dumps(
                transitions,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "ordered_transition_semantics": "percentage_of_current_state",
            "surface_domain": surface_key,
        }
    )
    return _row(prompt, assistant, metadata)


_LOGICAL_ATOM_VOCABULARIES: Mapping[str, Tuple[str, ...]] = {
    "train": (
        "alven",
        "brika",
        "corin",
        "daxel",
        "evora",
        "faron",
        "gimel",
        "helix",
        "ivren",
        "jural",
        "kemos",
        "lurin",
    ),
    "eval": (
        "mavik",
        "noxen",
        "orlan",
        "pirex",
        "quven",
        "roxil",
        "sovin",
        "turek",
        "uvran",
        "wexor",
        "xevin",
        "yulon",
    ),
}
_LOGICAL_TOPOLOGIES: Mapping[str, Tuple[str, ...]] = {
    "train": (
        "seeded_unary_ladder",
        "unseeded_chain_gap",
        "fact_conjunction_bridge",
        "missing_fact_conjunct",
    ),
    "eval": (
        "single_root_fork_join",
        "two_root_derived_join",
        "unseeded_cycle_exit",
        "derived_branch_gap",
    ),
}
_LOGICAL_SURFACE_MARKERS = {
    "train": "derivation-ledger",
    "eval": "model-audit",
}


def _logical_rule(premises: Sequence[str], conclusion: str) -> dict[str, object]:
    return {"if": sorted(premises), "then": conclusion}


def _logical_topology_task(
    topology: str,
    atoms: Sequence[str],
) -> dict[str, object]:
    """Construct one topology without consulting its expected answer."""

    if len(atoms) < 6:
        raise ValueError("Logical-entailment topology construction needs six atoms.")
    a, b, c, d, e, f = atoms[:6]
    if topology == "seeded_unary_ladder":
        facts = [a]
        rules = [
            _logical_rule([a], b),
            _logical_rule([b], c),
            _logical_rule([c], d),
            _logical_rule([e], f),
        ]
        query = d
    elif topology == "unseeded_chain_gap":
        facts = [a]
        rules = [
            _logical_rule([a], b),
            _logical_rule([c], d),
            _logical_rule([d], e),
        ]
        query = e
    elif topology == "fact_conjunction_bridge":
        facts = [a, b]
        rules = [
            _logical_rule([a, b], c),
            _logical_rule([c], d),
            _logical_rule([e], f),
        ]
        query = d
    elif topology == "missing_fact_conjunct":
        facts = [a]
        rules = [
            _logical_rule([a, b], c),
            _logical_rule([c], d),
            _logical_rule([e], f),
        ]
        query = d
    elif topology == "single_root_fork_join":
        facts = [a]
        rules = [
            _logical_rule([a], b),
            _logical_rule([a], c),
            _logical_rule([b, c], d),
            _logical_rule([e], f),
        ]
        query = d
    elif topology == "two_root_derived_join":
        facts = [a, b]
        rules = [
            _logical_rule([a], c),
            _logical_rule([b], d),
            _logical_rule([c, d], e),
        ]
        query = e
    elif topology == "unseeded_cycle_exit":
        facts = [e]
        rules = [
            _logical_rule([a], b),
            _logical_rule([b], a),
            _logical_rule([b], c),
        ]
        query = c
    elif topology == "derived_branch_gap":
        facts = [a]
        rules = [
            _logical_rule([a], b),
            _logical_rule([c], d),
            _logical_rule([b, d], e),
        ]
        query = e
    else:
        raise ValueError(f"Unsupported logical-entailment topology {topology!r}.")
    return {
        "schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        "facts": sorted(facts),
        "rules": rules,
        "query": query,
    }


def _generate_logical_entailment(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    topologies = _LOGICAL_TOPOLOGIES[split]
    topology = topologies[index % len(topologies)]
    atoms = rng.sample(list(_LOGICAL_ATOM_VOCABULARIES[split]), 6)
    task_ir = _logical_topology_task(topology, atoms)
    task_ir_json = canonical_task_ir_json(task_ir)
    # The target comes exclusively from bounded exhaustive Boolean-model search.
    answer = derive_entailment_answer(task_ir)
    surface_marker = _LOGICAL_SURFACE_MARKERS[split]
    case_id = f"{prefix.upper()}-L{index + 1:04d}"
    prompt = (
        f"{surface_marker} {case_id}: Decide positive-Horn entailment under classical "
        "Boolean model semantics.\n"
        "Reply exactly with entailed or not entailed.\n"
        f"{render_task_statement(task_ir)}"
    )
    template_id = f"{prefix}.logical_entailment.{topology}.v1"
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="logical_entailment",
        template_id=template_id,
        verifier_type="logical_entailment",
        expected_answer=answer,
        difficulty=0.72 if "conjunction" not in topology and "join" not in topology else 0.78,
    )
    metadata.update(
        {
            "task_ir_schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
            "task_ir_json": task_ir_json,
            "oracle_id": LOGICAL_ENTAILMENT_ORACLE_ID,
            "rule_topology": topology,
            "surface_marker": surface_marker,
            "atom_vocabulary_json": json.dumps(
                sorted(set(atoms)),
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        }
    )
    return _row(prompt, answer, metadata)


def _generate_calibrated_prediction(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    total = rng.randint(8, 36)
    successes = rng.randint(1, total - 1)
    estimate = (Decimal(successes + 1) / Decimal(total + 2)).quantize(
        Decimal("0.001"),
        rounding=ROUND_HALF_UP,
    )
    answer = format(estimate, ".3f")
    event = "sensor checks" if split == "train" else "replication attempts"
    template_id = f"{prefix}.prediction.laplace_smoothed_rate.v1"
    prompt = (
        f"Prediction case {prefix.upper()}-{index + 1:04d}: In {total} comparable "
        f"{event}, {successes} succeeded. Use the explicitly "
        "specified Laplace estimate (successes + 1) / (trials + 2). Report the "
        "result to three decimal places and describe it as an estimate, not a guarantee."
    )
    assistant = (
        f"The smoothed estimate is ({successes} + 1) / ({total} + 2) = {answer}. "
        f"This is an estimate, not a guarantee."
    )
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="calibrated_prediction",
        template_id=template_id,
        verifier_type="response_contract",
        expected_answer="contract",
        difficulty=0.54,
        required_terms=(answer, "estimate", "not a guarantee"),
    )
    return _row(prompt, assistant, metadata)


def _generate_causal_evidence(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    randomized = bool((index + rng.randint(0, 1)) % 2)
    if split == "train":
        exposure, outcome = "study reminders", "quiz completion"
    else:
        exposure, outcome = "irrigation schedules", "plant growth"
    if randomized:
        design = (
            f"Researchers randomly assign otherwise eligible units to receive {exposure} "
            f"or no {exposure}, then compare {outcome}."
        )
        answer = "A"
        explanation = "Random assignment supports a causal interpretation under the stated design."
    else:
        design = (
            f"Researchers observe which units already choose {exposure} and find that "
            f"they also have higher {outcome}; no assignment or confounder control is used."
        )
        answer = "B"
        explanation = "The observational association alone does not establish causation."
    template_id = f"{prefix}.causal_evidence.study_design.v1"
    prompt = (
        f"Evidence case {prefix.upper()}-{index + 1:04d}: {design}\n"
        "Which conclusion is warranted?\n"
        "A. The design can support a causal effect, subject to its assumptions.\n"
        "B. The evidence establishes association only, not a causal effect.\n"
        "Reply with the choice label."
    )
    assistant = f"{explanation} Final answer: {answer}."
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="causal_evidence",
        template_id=template_id,
        verifier_type="multiple_choice",
        expected_answer=answer,
        difficulty=0.60,
    )
    return _row(prompt, assistant, metadata)


_CONVERSATION_CASES = {
    "train": (
        (
            "battery storage",
            "efficiency",
            "price",
            (
                "Stored energy shifts renewable supply across time.",
                "Efficiency determines how much energy returns.",
                "Fast controls help stabilize the grid.",
            ),
        ),
        (
            "vaccination trials",
            "control group",
            "guarantee",
            (
                "Randomization reduces systematic differences between groups.",
                "A control group provides the comparison baseline.",
                "Confidence intervals show remaining statistical uncertainty.",
            ),
        ),
        (
            "database indexes",
            "query",
            "magic",
            (
                "Indexes reduce records scanned by a query.",
                "Writes spend extra work maintaining index entries.",
                "Measure real workloads before selecting columns.",
            ),
        ),
        (
            "urban trees",
            "evidence",
            "perfect",
            (
                "Shade can reduce local surface temperatures.",
                "Evidence quality varies across neighborhoods and methods.",
                "Maintenance determines whether benefits persist.",
            ),
        ),
    ),
    "eval": (
        (
            "heat pumps",
            "efficiency",
            "free",
            (
                "Heat pumps move energy instead of creating it.",
                "Efficiency changes with outdoor and delivery temperatures.",
                "Insulation can reduce the required heating load.",
            ),
        ),
        (
            "A/B experiments",
            "control group",
            "certainty",
            (
                "Random assignment balances many hidden differences.",
                "A control group anchors the treatment comparison.",
                "Confidence intervals quantify sampling uncertainty.",
            ),
        ),
        (
            "software caching",
            "latency",
            "always",
            (
                "Caching can reduce repeated computation and latency.",
                "Invalidation keeps stored results from becoming stale.",
                "Measure hit rates and memory pressure together.",
            ),
        ),
        (
            "wetland restoration",
            "evidence",
            "guaranteed",
            (
                "Wetlands can slow runoff and trap sediment.",
                "Evidence should compare conditions across time.",
                "Local hydrology shapes restoration outcomes.",
            ),
        ),
    ),
}


def _generate_conversation_constraints(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    del rng
    prefix = "train" if split == "train" else "eval"
    topic, required, forbidden, bullets = _CONVERSATION_CASES[split][
        index % len(_CONVERSATION_CASES[split])
    ]
    template_id = f"{prefix}.conversation.three_bounded_bullets.v1"
    prompt = (
        f"Conversation case {prefix.upper()}-{index + 1:04d}: Explain {topic} in exactly "
        "three bullet points. Keep every bullet to at most "
        f"nine words. Include the term '{required}' somewhere, and do not use the word "
        f"'{forbidden}'."
    )
    assistant = "\n".join(f"- {bullet}" for bullet in bullets)
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="conversation_constraints",
        template_id=template_id,
        verifier_type="response_contract",
        expected_answer="contract",
        difficulty=0.62,
        required_terms=(required,),
        forbidden_terms=(forbidden,),
        exact_bullet_count=3,
        max_words_per_bullet=9,
    )
    return _row(prompt, assistant, metadata)


_MULTI_TURN_CASES = {
    "train": (
        (
            "solar forecasting",
            "uncertainty",
            "measurement",
            "guaranteed",
            (
                "- Uncertainty grows when weather patterns shift.",
                "- Better measurement improves short-range calibration.",
            ),
        ),
        (
            "code review",
            "correctness",
            "maintainability",
            "perfect",
            (
                "- Correctness checks should target concrete failure cases.",
                "- Maintainability improves with small explicit interfaces.",
            ),
        ),
    ),
    "eval": (
        (
            "clinical screening",
            "sensitivity",
            "specificity",
            "certainty",
            (
                "- Sensitivity measures detection among affected people.",
                "- Specificity measures exclusion among unaffected people.",
            ),
        ),
        (
            "model monitoring",
            "drift",
            "calibration",
            "always",
            (
                "- Drift checks whether deployed inputs have changed.",
                "- Calibration compares confidence with observed outcomes.",
            ),
        ),
    ),
}


def _generate_multi_turn_instruction(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    del rng
    prefix = "train" if split == "train" else "eval"
    topic, first_term, second_term, forbidden, bullets = _MULTI_TURN_CASES[split][
        index % len(_MULTI_TURN_CASES[split])
    ]
    template_id = f"{prefix}.conversation.followup_refinement.v1"
    prompt = (
        f"Conversation case {prefix.upper()}-{index + 1:04d}:\n"
        f"Turn 1 user: Give a short explanation of {topic}.\n"
        "Turn 2 user: Make it exactly two bullets. Keep both earlier scope and this "
        f"format change. Include '{first_term}' and '{second_term}', and do not say "
        f"'{forbidden}'.\n"
        "Respond to Turn 2."
    )
    assistant = "\n".join(bullets)
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="multi_turn_instruction",
        template_id=template_id,
        verifier_type="response_contract",
        expected_answer="contract",
        difficulty=0.70,
        required_terms=(first_term, second_term),
        forbidden_terms=(forbidden,),
        exact_bullet_count=2,
        max_words_per_bullet=9,
    )
    return _row(prompt, assistant, metadata)


_ADVANCED_GENERATORS: Mapping[
    str,
    Callable[[str, int, random.Random], dict[str, object]],
] = {
    "quantitative_science": _generate_quantitative_science,
    "quantity_transition_reasoning": _generate_quantity_transition,
    "logical_entailment": _generate_logical_entailment,
    "calibrated_prediction": _generate_calibrated_prediction,
    "causal_evidence": _generate_causal_evidence,
    "conversation_constraints": _generate_conversation_constraints,
    "multi_turn_instruction": _generate_multi_turn_instruction,
}


def _generate_advanced_split(*, split: str, count: int, seed: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    indices: Counter[str] = Counter()
    for ordinal in range(count):
        family = ADVANCED_FAMILIES[ordinal % len(ADVANCED_FAMILIES)]
        index = indices[family]
        indices[family] += 1
        rng = random.Random(
            _stable_seed(CURRICULUM_SCHEMA_VERSION, seed, split, family, index)
        )
        rows.append(_ADVANCED_GENERATORS[family](split, index, rng))
    return rows


def _component_counts(total: int) -> tuple[int, int, int]:
    minimum_rows = 5 + 5 + len(ADVANCED_FAMILIES)
    if total < minimum_rows:
        raise ValueError(
            f"Each split needs at least {minimum_rows} rows to cover all families."
        )
    reasoning = max(5, int(round(total * 0.50)))
    prompt = max(5, int(round(total * 0.20)))
    advanced = total - reasoning - prompt
    if advanced < len(ADVANCED_FAMILIES):
        deficit = len(ADVANCED_FAMILIES) - advanced
        reasoning -= deficit
        advanced += deficit
    return reasoning, prompt, advanced


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _split_summary(rows: Sequence[Mapping[str, object]], filename: str) -> dict[str, object]:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    template_ids = set()
    for row in rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Curriculum metadata must be a mapping.")
        family_counts[str(metadata.get("problem_family") or "unknown")] += 1
        source_counts[str(row.get("source") or "unknown")] += 1
        verifier_counts[str(metadata.get("verifier_type") or "unknown")] += 1
        template_ids.add(str(metadata.get("template_id") or ""))
    payload = _jsonl_bytes(rows)
    return {
        "file": filename,
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "verifier_type_counts": dict(sorted(verifier_counts.items())),
        "template_ids": sorted(template_ids),
    }


def _logical_holdout_summary(
    train_rows: Sequence[Mapping[str, object]],
    eval_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    split_values: dict[str, dict[str, object]] = {}
    for split, rows in (("train", train_rows), ("eval", eval_rows)):
        atoms: set[str] = set()
        topologies: set[str] = set()
        surfaces: set[str] = set()
        answers: Counter[str] = Counter()
        count = 0
        for row in rows:
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                continue
            if metadata.get("problem_family") != "logical_entailment":
                continue
            task_ir = parse_canonical_task_ir_json(metadata.get("task_ir_json"))
            if derive_entailment_answer(task_ir) != metadata.get("expected_answer"):
                raise RuntimeError("Logical-entailment target disagrees with its model oracle.")
            atoms.update(str(atom) for atom in json.loads(str(metadata["atom_vocabulary_json"])))
            topologies.add(str(metadata.get("rule_topology") or ""))
            surfaces.add(str(metadata.get("surface_marker") or ""))
            answers[str(metadata.get("expected_answer") or "")] += 1
            count += 1
        if count < 1 or "" in topologies or "" in surfaces or "" in answers:
            raise RuntimeError(f"Logical-entailment {split} split lacks complete holdout metadata.")
        split_values[split] = {
            "rows": count,
            "atom_vocabulary": sorted(atoms),
            "rule_topologies": sorted(topologies),
            "surface_markers": sorted(surfaces),
            "answer_counts": dict(sorted(answers.items())),
        }

    train = split_values["train"]
    evaluation = split_values["eval"]
    checks = {
        "atom_vocabularies_disjoint": set(train["atom_vocabulary"]).isdisjoint(
            evaluation["atom_vocabulary"]
        ),
        "rule_topologies_disjoint": set(train["rule_topologies"]).isdisjoint(
            evaluation["rule_topologies"]
        ),
        "surface_markers_disjoint": set(train["surface_markers"]).isdisjoint(
            evaluation["surface_markers"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("Logical-entailment semantic holdout boundary overlaps.")
    return {
        "task_ir_schema": LOGICAL_ENTAILMENT_IR_SCHEMA_VERSION,
        "oracle_id": LOGICAL_ENTAILMENT_ORACLE_ID,
        **checks,
        "train": train,
        "eval": evaluation,
    }


def _validate_rows(rows: Sequence[Mapping[str, object]], split: str) -> None:
    seen_prompts = set()
    for row in rows:
        if set(row) != {"user", "assistant", "source", "metadata"}:
            raise ValueError("Curriculum rows require user, assistant, source, and metadata.")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("Curriculum metadata must be a mapping.")
        if metadata.get("curriculum_split") != split:
            raise ValueError("Curriculum split metadata mismatch.")
        if any(
            not isinstance(key, str) or not isinstance(value, _SCALAR_TYPES)
            for key, value in metadata.items()
        ):
            raise ValueError("Curriculum metadata values must be scalar.")
        prompt_key = str(row.get("user") or "").strip().casefold()
        if not prompt_key or prompt_key in seen_prompts:
            raise ValueError("Curriculum prompts must be non-empty and unique within a split.")
        seen_prompts.add(prompt_key)
        result = verify_candidate(row["user"], row["assistant"], metadata)
        if not result.valid_spec or not result.passed:
            raise ValueError(f"Target failed verifier: {result.reason}")


def build_curriculum(
    *,
    seed: int = 5201,
    train_rows: int = 1_200,
    eval_rows: int = 150,
) -> GeneralCurriculumBundle:
    """Build a deterministic mixed curriculum and its promotion manifest."""

    train_counts = _component_counts(int(train_rows))
    eval_counts = _component_counts(int(eval_rows))
    reasoning = build_reasoning_curriculum(
        seed=_stable_seed(seed, "reasoning") % (2**31),
        train_rows=train_counts[0],
        eval_rows=eval_counts[0],
    )
    prompt = build_prompt_curriculum(
        seed=_stable_seed(seed, "prompt") % (2**31),
        train_rows=train_counts[1],
        eval_rows=eval_counts[1],
    )
    advanced_train = _generate_advanced_split(
        split="train",
        count=train_counts[2],
        seed=seed,
    )
    advanced_eval = _generate_advanced_split(
        split="eval",
        count=eval_counts[2],
        seed=seed,
    )

    train = list(reasoning.train_rows) + list(prompt.train_rows) + advanced_train
    evaluation = list(reasoning.eval_rows) + list(prompt.eval_rows) + advanced_eval
    random.Random(_stable_seed(seed, "shuffle", "train")).shuffle(train)
    random.Random(_stable_seed(seed, "shuffle", "eval")).shuffle(evaluation)
    _validate_rows(train, "train")
    _validate_rows(evaluation, "eval")

    train_templates = {str(row["metadata"]["template_id"]) for row in train}  # type: ignore[index]
    eval_templates = {str(row["metadata"]["template_id"]) for row in evaluation}  # type: ignore[index]
    train_prompts = {str(row["user"]).strip().casefold() for row in train}
    eval_prompts = {str(row["user"]).strip().casefold() for row in evaluation}
    if not train_templates.isdisjoint(eval_templates):
        raise RuntimeError("Train/eval template IDs overlap.")
    if not train_prompts.isdisjoint(eval_prompts):
        raise RuntimeError("Train/eval prompt text overlaps.")

    manifest = {
        "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "seed": int(seed),
        "mixture": {
            "reasoning_fraction": 0.50,
            "prompt_understanding_fraction": 0.20,
            "advanced_fraction": 0.30,
            "advanced_families": list(ADVANCED_FAMILIES),
        },
        "template_ids_disjoint": True,
        "prompt_text_disjoint": True,
        "all_targets_verified": True,
        "all_logical_targets_oracle_verified": True,
        "semantic_holdouts": {
            "logical_entailment": _logical_holdout_summary(train, evaluation),
        },
        "train": _split_summary(train, TRAIN_FILENAME),
        "eval": _split_summary(evaluation, EVAL_FILENAME),
    }
    return GeneralCurriculumBundle(tuple(train), tuple(evaluation), manifest)


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite curriculum artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_curriculum(
    bundle: GeneralCurriculumBundle,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    payloads = {
        "train_jsonl": (root / TRAIN_FILENAME, _jsonl_bytes(bundle.train_rows)),
        "eval_jsonl": (root / EVAL_FILENAME, _jsonl_bytes(bundle.eval_rows)),
        "manifest_json": (
            root / MANIFEST_FILENAME,
            (json.dumps(bundle.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        ),
    }
    for _name, (path, payload) in payloads.items():
        _write_atomic(path, payload, overwrite=overwrite)
    return {name: str(path) for name, (path, _payload) in payloads.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mixed verifier-grounded Supermix general-capability curriculum."
    )
    parser.add_argument(
        "--output-dir",
        default="output/general_intelligence_curriculum_v3",
    )
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--train-rows", type=int, default=1_200)
    parser.add_argument("--eval-rows", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bundle = build_curriculum(
        seed=int(args.seed),
        train_rows=int(args.train_rows),
        eval_rows=int(args.eval_rows),
    )
    paths = write_curriculum(bundle, args.output_dir, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "status": "complete",
                "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
                "train_rows": len(bundle.train_rows),
                "eval_rows": len(bundle.eval_rows),
                "artifacts": paths,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADVANCED_FAMILIES",
    "CURRICULUM_SCHEMA_VERSION",
    "CURRICULUM_SOURCE",
    "EVAL_FILENAME",
    "GeneralCurriculumBundle",
    "MANIFEST_FILENAME",
    "TRAIN_FILENAME",
    "build_curriculum",
    "main",
    "parse_args",
    "write_curriculum",
]
