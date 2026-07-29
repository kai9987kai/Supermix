"""Build a deterministic, verifier-grounded reasoning curriculum.

The emitted JSONL rows are directly compatible with the Qwen ``ChatPair``
loader: each row has ``user``, ``assistant``, ``source``, and scalar
``metadata`` fields.  Train and evaluation examples use disjoint template IDs
and independently derived random streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from itertools import permutations
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    from verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate
except ImportError:  # pragma: no cover - package import path
    from .verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate


CURRICULUM_SCHEMA_VERSION = "supermix-verifiable-reasoning-curriculum-v1"
CURRICULUM_SOURCE = "supermix_verified_reasoning_v1"
TRAIN_FILENAME = "verified_reasoning_train.jsonl"
EVAL_FILENAME = "verified_reasoning_eval.jsonl"
MANIFEST_FILENAME = "verified_reasoning_manifest.json"

PROBLEM_FAMILIES: Tuple[str, ...] = (
    "multi_step_arithmetic",
    "ratios_probability",
    "sequences",
    "constraint_logic_tables",
    "evidence_in_prompt_qa",
)

_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class CurriculumBundle:
    train_rows: Tuple[dict[str, object], ...]
    eval_rows: Tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _row_rng(seed: int, split: str, family: str, index: int, attempt: int = 0) -> random.Random:
    return random.Random(_stable_seed(CURRICULUM_SCHEMA_VERSION, seed, split, family, index, attempt))


def _canonical_decimal(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


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
    json_field: str = "",
    absolute_tolerance: str = "0",
) -> dict[str, object]:
    example_id = hashlib.sha256(
        f"{split}|{family}|{template_id}|{prompt}".encode("utf-8")
    ).hexdigest()[:24]
    metadata: dict[str, object] = {
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "verifier_type": str(verifier_type),
        "expected_answer": str(expected_answer),
        "aliases_json": json.dumps(list(aliases), ensure_ascii=False, separators=(",", ":")),
        "absolute_tolerance": str(absolute_tolerance),
        "problem_family": str(family),
        "template_id": str(template_id),
        "split_group": f"{split}:{template_id}",
        "curriculum_split": str(split),
        "verifier_difficulty": float(max(0.0, min(1.0, difficulty))),
        "verified_correct": True,
        "rule_reward": 1.0,
        "example_id": example_id,
    }
    if json_field:
        metadata["json_field"] = str(json_field)
    return metadata


def _row(prompt: str, assistant: str, metadata: Mapping[str, object]) -> dict[str, object]:
    return {
        "user": str(prompt),
        "assistant": str(assistant),
        "source": CURRICULUM_SOURCE,
        "metadata": dict(metadata),
    }


def _generate_multi_step_arithmetic(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    if split == "train":
        template_ids = (
            "train.arithmetic.sample_copies.v1",
            "train.arithmetic.supply_batches.v1",
        )
    else:
        template_ids = (
            "eval.arithmetic.energy_counter.v1",
            "eval.arithmetic.archive_pages.v1",
        )
    template_id = template_ids[index % len(template_ids)]
    start = rng.randint(8, 75)
    added = rng.randint(3, 24)
    multiplier = rng.randint(2, 5)
    removed = rng.randint(1, min(18, (start + added) * multiplier - 1))
    answer = (start + added) * multiplier - removed

    if template_id.endswith("sample_copies.v1"):
        prompt = (
            f"A lab starts with {start} sample records and receives {added} more. "
            f"It then makes {multiplier} copies of every record in the combined set "
            f"and discards {removed} damaged copies. How many usable copies remain?"
        )
    elif template_id.endswith("supply_batches.v1"):
        prompt = (
            f"A workshop has {start} parts, adds {added} parts, prepares "
            f"{multiplier} identical batches of that combined amount, then removes "
            f"{removed} faulty parts. What is the final part count?"
        )
    elif template_id.endswith("energy_counter.v1"):
        prompt = (
            f"An energy counter begins at {start}, rises by {added}, is multiplied "
            f"by {multiplier} after calibration, and then loses {removed} units. "
            "What value does the counter show?"
        )
    else:
        prompt = (
            f"An archive has {start} indexed pages and adds {added}. It creates "
            f"{multiplier} complete indexed copies, then removes {removed} duplicate "
            "pages. How many indexed pages remain?"
        )

    assistant = (
        f"Combine the first two amounts: {start} + {added} = {start + added}. "
        f"Apply the multiplier: {start + added} × {multiplier} = {(start + added) * multiplier}. "
        f"Remove {removed}: {(start + added) * multiplier} - {removed} = {answer}. "
        f"Final answer: {answer}."
    )
    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="multi_step_arithmetic",
        template_id=template_id,
        verifier_type="integer",
        expected_answer=str(answer),
        difficulty=0.38 + 0.04 * (multiplier - 2),
    )
    return _row(prompt, assistant, metadata)


def _generate_ratios_probability(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    variant = index % 3
    if variant == 0:
        template_id = f"{prefix}.ratios_probability.draw_fraction.v1"
        favorable = rng.randint(2, 18)
        other = rng.randint(3, 22)
        total = favorable + other
        answer_fraction = Fraction(favorable, total)
        answer = f"{answer_fraction.numerator}/{answer_fraction.denominator}"
        vessel = "bag" if split == "train" else "sealed tray"
        prompt = (
            f"A {vessel} contains {favorable} red markers and {other} blue markers. "
            "One marker is selected uniformly at random. Give the probability of "
            "selecting red as a reduced fraction."
        )
        assistant = (
            f"There are {total} markers in total and {favorable} favorable outcomes. "
            f"The probability is {favorable}/{total}, which reduces to {answer}. "
            f"Final answer: {answer}."
        )
        verifier_type = "fraction"
        difficulty = 0.34
    elif variant == 1:
        template_id = f"{prefix}.ratios_probability.scale_ratio.v1"
        left = rng.randint(2, 7)
        right = rng.randint(3, 9)
        units = rng.randint(3, 14)
        known = left * units
        answer = str(right * units)
        noun = "sensors" if split == "train" else "inspection tags"
        prompt = (
            f"A plan uses {left} controllers for every {right} {noun}. "
            f"If it uses {known} controllers at the same ratio, how many {noun} are needed?"
        )
        assistant = (
            f"The scale factor is {known} ÷ {left} = {units}. "
            f"Scale the other side: {right} × {units} = {answer}. "
            f"Final answer: {answer}."
        )
        verifier_type = "integer"
        difficulty = 0.42
    else:
        template_id = f"{prefix}.ratios_probability.observed_decimal.v1"
        total = rng.choice((10, 20, 25, 40, 50, 100))
        successful = rng.randint(1, total - 1)
        answer = _canonical_decimal(Decimal(successful) / Decimal(total))
        noun = "checks" if split == "train" else "calibration trials"
        prompt = (
            f"Out of {total} independent {noun}, {successful} succeed. "
            "Write the observed success rate as a decimal, without a percent sign."
        )
        assistant = (
            f"Divide successes by total trials: {successful} ÷ {total} = {answer}. "
            f"Final answer: {answer}."
        )
        verifier_type = "decimal"
        difficulty = 0.30

    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="ratios_probability",
        template_id=template_id,
        verifier_type=verifier_type,
        expected_answer=answer,
        difficulty=difficulty,
    )
    return _row(prompt, assistant, metadata)


def _generate_sequences(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    if index % 2 == 0:
        template_id = f"{prefix}.sequences.arithmetic_term.v1"
        first = rng.randint(-12, 35)
        difference = rng.choice(tuple(value for value in range(-9, 10) if value != 0))
        term_number = rng.randint(6, 18)
        answer = first + (term_number - 1) * difference
        setting = "sensor log" if split == "train" else "field notebook"
        prompt = (
            f"Sequence case {prefix.upper()}-A{index + 1:04d}: A {setting} follows "
            f"an arithmetic sequence with first term {first} "
            f"and common difference {difference}. What is term {term_number}?"
        )
        assistant = (
            f"Use aₙ = a₁ + (n - 1)d. Here, {first} + ({term_number} - 1) × "
            f"{difference} = {answer}. Final answer: {answer}."
        )
        difficulty = 0.38
    else:
        template_id = f"{prefix}.sequences.geometric_term.v1"
        first = rng.randint(2, 6)
        ratio = rng.randint(2, 3)
        term_number = rng.randint(4, 7)
        answer = first * (ratio ** (term_number - 1))
        setting = "growth model" if split == "train" else "replication study"
        prompt = (
            f"Sequence case {prefix.upper()}-G{index + 1:04d}: A {setting} uses "
            f"a geometric sequence with first term {first} "
            f"and common ratio {ratio}. What is term {term_number}?"
        )
        assistant = (
            f"Use aₙ = a₁r^(n-1). Thus {first} × {ratio}^{term_number - 1} "
            f"= {answer}. Final answer: {answer}."
        )
        difficulty = 0.46

    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="sequences",
        template_id=template_id,
        verifier_type="integer",
        expected_answer=str(answer),
        difficulty=difficulty,
    )
    return _row(prompt, assistant, metadata)


def _logic_choice_prompt(
    *,
    title: str,
    expression: str,
    p: bool,
    q: bool,
    r: bool,
    result: bool,
) -> tuple[str, str, Tuple[str, ...]]:
    expected_label = "A" if result else "B"
    expected_word = "True" if result else "False"
    prompt = (
        f"{title}\n\n"
        "| Variable | Value |\n"
        "|---|---|\n"
        f"| P | {str(p)} |\n"
        f"| Q | {str(q)} |\n"
        f"| R | {str(r)} |\n\n"
        f"Evaluate `{expression}`.\n"
        "A. True\n"
        "B. False\n"
        "Reply with the choice label."
    )
    return prompt, expected_label, (expected_word,)


def _generate_constraint_logic_tables(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    if index % 2 == 0:
        template_id = f"{prefix}.constraint_logic.truth_table.v1"
        p, q, r = (bool(rng.getrandbits(1)) for _ in range(3))
        if split == "train":
            result = (p and q) or (not r)
            expression = "(P AND Q) OR (NOT R)"
            title = f"Use the supplied truth-value table for logic case T{index + 1:04d}."
        else:
            result = (p != q) and r
            expression = "(P XOR Q) AND R"
            title = f"Use only this evaluation table for logic case E{index + 1:04d}."
        prompt, answer, aliases = _logic_choice_prompt(
            title=title,
            expression=expression,
            p=p,
            q=q,
            r=r,
            result=result,
        )
        assistant = (
            f"Substituting the table values makes the expression "
            f"{'true' if result else 'false'}. Final answer: {answer}."
        )
        difficulty = 0.44
    else:
        template_id = f"{prefix}.constraint_logic.unique_schedule.v1"
        train_names = ("Audit", "Build", "Check", "Deploy", "Review", "Test")
        eval_names = ("Intake", "Label", "Measure", "Publish", "Sample", "Verify")
        pool = train_names if split == "train" else eval_names
        correct_order = tuple(rng.sample(pool, 4))
        all_orders = list(permutations(correct_order))
        wrong_orders = [order for order in all_orders if order != correct_order]
        rng.shuffle(wrong_orders)
        choices = [correct_order, *wrong_orders[:3]]
        rng.shuffle(choices)
        labels = ("A", "B", "C", "D")
        correct_index = choices.index(correct_order)
        answer = labels[correct_index]
        option_lines = [
            f"{label}. {' → '.join(order)}"
            for label, order in zip(labels, choices)
        ]
        prompt = (
            f"Schedule case {prefix.upper()}-S{index + 1:04d}: "
            "Four tasks must obey all three constraints:\n"
            f"- {correct_order[0]} occurs before {correct_order[1]}.\n"
            f"- {correct_order[1]} occurs before {correct_order[2]}.\n"
            f"- {correct_order[2]} occurs before {correct_order[3]}.\n\n"
            "Which schedule satisfies every constraint?\n"
            + "\n".join(option_lines)
            + "\nReply with the choice label."
        )
        assistant = (
            "The constraints form one complete order: "
            f"{' → '.join(correct_order)}. Final answer: {answer}."
        )
        aliases = (" → ".join(correct_order),)
        difficulty = 0.40

    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="constraint_logic_tables",
        template_id=template_id,
        verifier_type="multiple_choice",
        expected_answer=answer,
        difficulty=difficulty,
        aliases=aliases,
    )
    return _row(prompt, assistant, metadata)


def _generate_evidence_in_prompt_qa(
    split: str,
    index: int,
    rng: random.Random,
) -> dict[str, object]:
    prefix = "train" if split == "train" else "eval"
    if split == "train":
        projects = (
            ("Aurora Key", "Northbridge", "North Bridge"),
            ("Cedar Signal", "Westhaven", "West Haven"),
            ("Indigo Map", "Lakepoint", "Lake Point"),
            ("Silver Reed", "Eastmere", "East Mere"),
        )
        distractors = ("Grayfield", "Southport", "Rivermark", "Pinegate")
    else:
        projects = (
            ("Amber Lens", "Stoneford", "Stone Ford"),
            ("Kestrel Note", "Clearwater", "Clear Water"),
            ("Violet Path", "Oakridge", "Oak Ridge"),
            ("Willow Code", "Brighton", "Bright Town"),
        )
        distractors = ("Foxmere", "Hillcrest", "Redbrook", "Windham")

    project, location, alias = rng.choice(projects)
    other_location = rng.choice(tuple(item for item in distractors if item != location))
    record_id = hashlib.sha256(
        f"{split}|evidence|{index}|{project}|{location}".encode("utf-8")
    ).hexdigest()[:8].upper()
    context = (
        f"Record {record_id}\n"
        f"- Project {project} stores its primary archive in {location}.\n"
        f"- The backup archive is in {other_location}.\n"
        "- Only the primary archive answers the question below."
    )

    if index % 2 == 0:
        template_id = f"{prefix}.evidence_qa.normalized_exact.v1"
        prompt = (
            f"Use only the supplied record.\n\n{context}\n\n"
            f"Where is the primary archive for Project {project}?"
        )
        assistant = (
            f"The record states that Project {project}'s primary archive is in "
            f"{location}. Final answer: {location}."
        )
        verifier_type = "normalized_exact"
        expected_answer = location
        aliases = (alias,)
        json_field = ""
    else:
        template_id = f"{prefix}.evidence_qa.json_field.v1"
        prompt = (
            f"Use only the supplied record.\n\n{context}\n\n"
            f"Return a JSON object whose `answer` field is the primary archive "
            f"location for Project {project}. Do not add prose."
        )
        assistant = json.dumps({"answer": location}, ensure_ascii=False, separators=(",", ":"))
        verifier_type = "json_field"
        expected_answer = json.dumps(location, ensure_ascii=False)
        aliases = (alias,)
        json_field = "answer"

    metadata = _metadata(
        prompt=prompt,
        split=split,
        family="evidence_in_prompt_qa",
        template_id=template_id,
        verifier_type=verifier_type,
        expected_answer=expected_answer,
        difficulty=0.28,
        aliases=aliases,
        json_field=json_field,
    )
    return _row(prompt, assistant, metadata)


_FAMILY_GENERATORS: Mapping[
    str,
    Callable[[str, int, random.Random], dict[str, object]],
] = {
    "multi_step_arithmetic": _generate_multi_step_arithmetic,
    "ratios_probability": _generate_ratios_probability,
    "sequences": _generate_sequences,
    "constraint_logic_tables": _generate_constraint_logic_tables,
    "evidence_in_prompt_qa": _generate_evidence_in_prompt_qa,
}


def _normalize_families(families: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not families:
        return PROBLEM_FAMILIES
    cooked: List[str] = []
    seen = set()
    for raw in families:
        family = str(raw).strip().lower()
        if not family or family in seen:
            continue
        if family not in _FAMILY_GENERATORS:
            raise ValueError(
                f"Unknown problem family {family!r}; expected one of {', '.join(PROBLEM_FAMILIES)}"
            )
        seen.add(family)
        cooked.append(family)
    if not cooked:
        raise ValueError("At least one problem family is required.")
    return tuple(cooked)


def _validate_row(row: Mapping[str, object], expected_split: str) -> None:
    if set(row) != {"user", "assistant", "source", "metadata"}:
        raise ValueError("Curriculum rows must contain user, assistant, source, and metadata.")
    if not str(row.get("user") or "").strip() or not str(row.get("assistant") or "").strip():
        raise ValueError("Curriculum rows require non-empty user and assistant text.")
    if row.get("source") != CURRICULUM_SOURCE:
        raise ValueError("Unexpected curriculum source.")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Curriculum metadata must be a dictionary.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, _SCALAR_TYPES):
            raise ValueError(f"Metadata must be scalar; invalid field {key!r}.")
    if metadata.get("curriculum_split") != expected_split:
        raise ValueError("Row split metadata does not match its output split.")
    result = verify_candidate(row["user"], row["assistant"], metadata)
    if not result.valid_spec or not result.passed:
        raise ValueError(
            f"Generated assistant failed its verifier: {result.reason} "
            f"({metadata.get('example_id', '-')})"
        )


def _build_split(
    *,
    split: str,
    count: int,
    seed: int,
    families: Sequence[str],
) -> List[dict[str, object]]:
    if split not in {"train", "eval"}:
        raise ValueError("split must be train or eval")
    if int(count) < 0:
        raise ValueError("row count cannot be negative")
    rows: List[dict[str, object]] = []
    prompt_keys = set()
    family_indices: Counter[str] = Counter()
    for ordinal in range(int(count)):
        family = families[ordinal % len(families)]
        family_index = int(family_indices[family])
        family_indices[family] += 1
        generator = _FAMILY_GENERATORS[family]
        row: Optional[dict[str, object]] = None
        for attempt in range(32):
            rng = _row_rng(seed, split, family, family_index, attempt)
            candidate = generator(split, family_index, rng)
            prompt_key = str(candidate["user"]).strip().casefold()
            if prompt_key not in prompt_keys:
                row = candidate
                prompt_keys.add(prompt_key)
                break
        if row is None:
            raise RuntimeError(
                f"Could not generate a unique {family} prompt after 32 deterministic attempts."
            )
        _validate_row(row, split)
        rows.append(row)
    return rows


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    lines = [
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _split_summary(rows: Sequence[Mapping[str, object]], filename: str) -> dict[str, object]:
    family_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    template_ids = set()
    for row in rows:
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        family_counts[str(metadata["problem_family"])] += 1
        verifier_counts[str(metadata["verifier_type"])] += 1
        template_ids.add(str(metadata["template_id"]))
    payload = _jsonl_bytes(rows)
    return {
        "file": filename,
        "rows": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "family_counts": dict(sorted(family_counts.items())),
        "verifier_type_counts": dict(sorted(verifier_counts.items())),
        "template_ids": sorted(template_ids),
    }


def build_curriculum(
    *,
    seed: int = 51,
    train_rows: int = 2_000,
    eval_rows: int = 400,
    families: Optional[Sequence[str]] = None,
) -> CurriculumBundle:
    """Build deterministic train/eval rows and a content-addressed manifest."""

    if int(train_rows) <= 0 or int(eval_rows) <= 0:
        raise ValueError("train_rows and eval_rows must both be positive.")
    selected_families = _normalize_families(families)
    train_payload = _build_split(
        split="train",
        count=int(train_rows),
        seed=int(seed),
        families=selected_families,
    )
    eval_payload = _build_split(
        split="eval",
        count=int(eval_rows),
        seed=int(seed),
        families=selected_families,
    )

    train_template_ids = {
        str(row["metadata"]["template_id"])  # type: ignore[index]
        for row in train_payload
    }
    eval_template_ids = {
        str(row["metadata"]["template_id"])  # type: ignore[index]
        for row in eval_payload
    }
    overlap = sorted(train_template_ids & eval_template_ids)
    if overlap:
        raise RuntimeError(f"Train/eval template IDs overlap: {overlap}")
    train_prompts = {str(row["user"]).strip().casefold() for row in train_payload}
    eval_prompts = {str(row["user"]).strip().casefold() for row in eval_payload}
    prompt_overlap = train_prompts & eval_prompts
    if prompt_overlap:
        raise RuntimeError("Train/eval prompt text overlap detected.")

    manifest = {
        "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "source": CURRICULUM_SOURCE,
        "seed": int(seed),
        "families": list(selected_families),
        "template_ids_disjoint": True,
        "prompt_text_disjoint": True,
        "train": _split_summary(train_payload, TRAIN_FILENAME),
        "eval": _split_summary(eval_payload, EVAL_FILENAME),
    }
    return CurriculumBundle(
        train_rows=tuple(train_payload),
        eval_rows=tuple(eval_payload),
        manifest=manifest,
    )


def _write_atomic(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing curriculum artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_bytes(payload)
    os.replace(temp_path, path)


def write_curriculum(
    bundle: CurriculumBundle,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Write a bundle atomically and return its artifact paths."""

    root = Path(output_dir).expanduser().resolve()
    train_path = root / TRAIN_FILENAME
    eval_path = root / EVAL_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    targets = (train_path, eval_path, manifest_path)
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing curriculum artifacts: "
                + ", ".join(str(path) for path in existing)
            )

    train_bytes = _jsonl_bytes(bundle.train_rows)
    eval_bytes = _jsonl_bytes(bundle.eval_rows)
    if hashlib.sha256(train_bytes).hexdigest() != str(bundle.manifest["train"]["sha256"]):  # type: ignore[index]
        raise ValueError("Train payload hash does not match the manifest.")
    if hashlib.sha256(eval_bytes).hexdigest() != str(bundle.manifest["eval"]["sha256"]):  # type: ignore[index]
        raise ValueError("Eval payload hash does not match the manifest.")
    manifest_bytes = (
        json.dumps(bundle.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    _write_atomic(train_path, train_bytes, overwrite=overwrite)
    _write_atomic(eval_path, eval_bytes, overwrite=overwrite)
    _write_atomic(manifest_path, manifest_bytes, overwrite=overwrite)
    return {
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "manifest_json": str(manifest_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic verifier-grounded Supermix reasoning curriculum."
    )
    parser.add_argument(
        "--output-dir",
        default="output/verifiable_reasoning_curriculum_v1",
        help="Directory for train/eval JSONL and the manifest.",
    )
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--train-rows", type=int, default=2_000)
    parser.add_argument("--eval-rows", type=int, default=400)
    parser.add_argument(
        "--families",
        default=",".join(PROBLEM_FAMILIES),
        help="Comma-separated subset of supported problem families.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts in the selected output directory.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    families = tuple(item.strip() for item in str(args.families).split(",") if item.strip())
    bundle = build_curriculum(
        seed=int(args.seed),
        train_rows=int(args.train_rows),
        eval_rows=int(args.eval_rows),
        families=families,
    )
    paths = write_curriculum(bundle, args.output_dir, overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "status": "complete",
                "curriculum_schema": CURRICULUM_SCHEMA_VERSION,
                "verifier_schema": VERIFIER_SCHEMA_VERSION,
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
    "CURRICULUM_SCHEMA_VERSION",
    "CURRICULUM_SOURCE",
    "CurriculumBundle",
    "EVAL_FILENAME",
    "MANIFEST_FILENAME",
    "PROBLEM_FAMILIES",
    "TRAIN_FILENAME",
    "build_curriculum",
    "main",
    "parse_args",
    "write_curriculum",
]
