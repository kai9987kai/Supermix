"""Build a deterministic, verifier-grounded frontier-repair curriculum.

The repair mixture is derived from an existing general-intelligence training
artifact and its manifest. Probability and calibrated-prediction examples are
oversampled while every other family is replayed. The parent evaluation file is
copied byte-for-byte into the derived artifact so the frozen evaluator and
promotion gate can bind the repaired train set to the original held-out eval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

try:
    from verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate
except ImportError:  # pragma: no cover - package import path
    from .verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate


REPAIR_SCHEMA_VERSION = "supermix-general-intelligence-repair-v3"
DEFAULT_SOURCE_DIR = Path("output/general_intelligence_curriculum_v3")
DEFAULT_OUTPUT_DIR = Path("output/general_intelligence_repair_v3")
DEFAULT_TRAIN_FILENAME = "general_intelligence_train.jsonl"
DEFAULT_EVAL_FILENAME = "general_intelligence_eval.jsonl"
DEFAULT_MANIFEST_FILENAME = "general_intelligence_manifest.json"
REPAIR_FILENAME_PREFIX = "general_intelligence_repair"
REPAIR_SOURCE = "supermix_general_intelligence_repair_v3"

FOCUS_FAMILIES: Tuple[str, ...] = (
    "ratios_probability",
    "calibrated_prediction",
)
PRIORITY_REPLAY_FAMILIES: Tuple[str, ...] = (
    "constraint_logic_tables",
    "hard_conflict_ask_vs_act",
    "evidence_in_prompt_qa",
    "multi_step_arithmetic",
)
REQUIRED_FAMILIES: Tuple[str, ...] = (
    *FOCUS_FAMILIES,
    *PRIORITY_REPLAY_FAMILIES,
    "quantitative_science",
    "quantity_transition_reasoning",
    "logical_entailment",
    "causal_evidence",
    "conversation_constraints",
    "multi_turn_instruction",
    "constraint_polarity_composition",
    "multi_turn_reference_intent_drift",
    "quoted_code_instruction_data_separation",
    "sequences",
    "typo_noise_robustness",
)
MIN_TARGET_ROWS = max(
    30,
    2 * (len(REQUIRED_FAMILIES) - len(FOCUS_FAMILIES)),
)


@dataclass(frozen=True)
class RepairCurriculumBundle:
    train_rows: Tuple[dict[str, object], ...]
    eval_bytes: bytes
    manifest: dict[str, object]


@dataclass(frozen=True)
class _SplitProfile:
    family_counts: dict[str, int]
    source_counts: dict[str, int]
    verifier_type_counts: dict[str, int]
    template_ids: frozenset[str]
    prompts: frozenset[str]
    example_ids: frozenset[str]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    return b"\n".join(_canonical_json_bytes(row) for row in rows) + b"\n"


def _normalized_prompt(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _load_json_mapping(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Manifest must be valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("Curriculum manifest must be a JSON object.")
    return value, payload


def _decode_jsonl_rows(payload: bytes, *, label: str) -> list[dict[str, object]]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Curriculum JSONL must be valid UTF-8: {label}") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        cooked = line.strip()
        if not cooked:
            continue
        try:
            value = json.loads(cooked)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL row {line_number}: {label}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} must be an object: {label}")
        rows.append(value)
    if not rows:
        raise ValueError(f"Curriculum JSONL is empty: {label}")
    return rows


def _load_jsonl_rows(path: Path) -> tuple[list[dict[str, object]], bytes]:
    payload = path.read_bytes()
    rows = _decode_jsonl_rows(payload, label=str(path))
    return rows, payload


def _safe_manifest_artifact(manifest_path: Path, raw_name: object) -> Path:
    name = str(raw_name or "").strip()
    relative = Path(name)
    if not name or relative.is_absolute():
        raise ValueError("Manifest artifact path must be a non-empty relative path.")
    candidate = (manifest_path.parent / relative).resolve()
    try:
        candidate.relative_to(manifest_path.parent.resolve())
    except ValueError as exc:
        raise ValueError("Manifest artifact path escapes its directory.") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest artifact not found: {candidate}")
    return candidate


def _manifest_count_map(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"Manifest {label} must be a non-empty mapping.")
    counts: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key or isinstance(raw_value, bool):
            raise ValueError(f"Manifest {label} contains an invalid entry.")
        try:
            count = int(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Manifest {label} contains a non-integer count.") from exc
        if count < 1 or count != raw_value:
            raise ValueError(f"Manifest {label} counts must be positive integers.")
        counts[key] = count
    return dict(sorted(counts.items()))


def _manifest_template_ids(value: object, *, label: str) -> frozenset[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Manifest {label} must be a non-empty list.")
    template_ids = [str(item or "").strip() for item in value]
    if any(not item for item in template_ids) or len(template_ids) != len(set(template_ids)):
        raise ValueError(f"Manifest {label} contains empty or duplicate template IDs.")
    return frozenset(template_ids)


def _validate_rows(rows: Sequence[Mapping[str, object]], *, split: str) -> _SplitProfile:
    family_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()
    template_ids: set[str] = set()
    prompts: set[str] = set()
    example_ids: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != {"user", "assistant", "source", "metadata"}:
            raise ValueError(f"Row {index} must contain user, assistant, source, and metadata.")
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"Row {index} metadata must be a mapping.")
        if metadata.get("curriculum_split") != split:
            raise ValueError(f"Row {index} curriculum_split must remain {split}.")
        if metadata.get("verifier_schema") != VERIFIER_SCHEMA_VERSION:
            raise ValueError(f"Row {index} verifier schema is not current.")
        if metadata.get("verified_correct") is not True:
            raise ValueError(f"Row {index} is not marked verified_correct.")
        family = str(metadata.get("problem_family") or "").strip()
        template_id = str(metadata.get("template_id") or "").strip()
        example_id = str(metadata.get("example_id") or "").strip()
        prompt_key = _normalized_prompt(row.get("user"))
        if not family or not template_id or not example_id or not prompt_key:
            raise ValueError(f"Row {index} is missing family, template, example ID, or prompt.")
        example_key = example_id.casefold()
        if prompt_key in prompts:
            raise ValueError("Curriculum contains duplicate prompts.")
        if example_key in example_ids:
            raise ValueError("Curriculum contains duplicate example IDs.")
        result = verify_candidate(row.get("user"), row.get("assistant"), metadata)
        if not result.valid_spec or not result.passed:
            raise ValueError(
                f"Row {index} target failed verifier revalidation: {result.reason}."
            )
        prompts.add(prompt_key)
        example_ids.add(example_key)
        template_ids.add(template_id)
        family_counts[family] += 1
        source_counts[str(row.get("source") or "unknown")] += 1
        verifier_counts[str(metadata.get("verifier_type") or "unknown")] += 1
    return _SplitProfile(
        family_counts=dict(sorted(family_counts.items())),
        source_counts=dict(sorted(source_counts.items())),
        verifier_type_counts=dict(sorted(verifier_counts.items())),
        template_ids=frozenset(template_ids),
        prompts=frozenset(prompts),
        example_ids=frozenset(example_ids),
    )


def _verify_manifest_split(
    summary: Mapping[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
    payload: bytes,
    profile: _SplitProfile,
    label: str,
) -> None:
    expected_sha = str(summary.get("sha256") or "").strip().lower()
    if expected_sha != _sha256_bytes(payload):
        raise ValueError(f"Input {label} hash does not match its manifest.")
    expected_rows = summary.get("rows")
    if isinstance(expected_rows, bool) or not isinstance(expected_rows, int):
        raise ValueError(f"Manifest {label} row count must be an integer.")
    if expected_rows != len(rows):
        raise ValueError(f"Input {label} row count does not match its manifest.")
    expected_families = _manifest_count_map(
        summary.get("family_counts"),
        label=f"{label}.family_counts",
    )
    if expected_families != profile.family_counts:
        raise ValueError(f"Input {label} family counts do not match its manifest.")
    expected_sources = _manifest_count_map(
        summary.get("source_counts"),
        label=f"{label}.source_counts",
    )
    if expected_sources != profile.source_counts:
        raise ValueError(f"Input {label} source counts do not match its manifest.")
    expected_verifier_types = _manifest_count_map(
        summary.get("verifier_type_counts"),
        label=f"{label}.verifier_type_counts",
    )
    if expected_verifier_types != profile.verifier_type_counts:
        raise ValueError(f"Input {label} verifier counts do not match its manifest.")
    expected_templates = _manifest_template_ids(
        summary.get("template_ids"),
        label=f"{label}.template_ids",
    )
    if expected_templates != profile.template_ids:
        raise ValueError(f"Input {label} template IDs do not match its manifest.")


def _parse_focus_fraction(value: float | str | Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("focus_fraction must be a finite decimal.") from exc
    if not parsed.is_finite() or parsed < Decimal("0.50") or parsed > Decimal("0.90"):
        raise ValueError("focus_fraction must be between 0.50 and 0.90.")
    return parsed


def _weighted_allocation(
    total: int,
    families: Sequence[str],
    weights: Mapping[str, int],
) -> dict[str, int]:
    if total < len(families):
        raise ValueError("Allocation budget is too small to cover every family.")
    counts = {family: 1 for family in families}
    remaining = total - len(families)
    weight_total = sum(int(weights[family]) for family in families)
    remainders: list[tuple[int, int, str]] = []
    assigned = 0
    for index, family in enumerate(families):
        numerator = remaining * int(weights[family])
        share, remainder = divmod(numerator, weight_total)
        counts[family] += share
        assigned += share
        remainders.append((remainder, -index, family))
    for _remainder, _negative_index, family in sorted(remainders, reverse=True)[
        : remaining - assigned
    ]:
        counts[family] += 1
    return counts


def _family_targets(target_rows: int, focus_fraction: Decimal) -> dict[str, int]:
    if target_rows < MIN_TARGET_ROWS:
        raise ValueError(
            f"target_rows must be at least {MIN_TARGET_ROWS} for "
            f"{len(REQUIRED_FAMILIES)}-family coverage."
        )
    focus_rows = int(
        (Decimal(target_rows) * focus_fraction).to_integral_value(rounding=ROUND_CEILING)
    )
    replay_families = tuple(
        family for family in REQUIRED_FAMILIES if family not in FOCUS_FAMILIES
    )
    replay_rows = target_rows - focus_rows
    if focus_rows < len(FOCUS_FAMILIES) or replay_rows < len(replay_families):
        raise ValueError("target_rows and focus_fraction leave too little replay coverage.")
    focus_targets = _weighted_allocation(
        focus_rows,
        FOCUS_FAMILIES,
        {
            "ratios_probability": 3,
            "calibrated_prediction": 1,
        },
    )
    replay_targets = _weighted_allocation(
        replay_rows,
        replay_families,
        {
            family: 2 if family in PRIORITY_REPLAY_FAMILIES else 1
            for family in replay_families
        },
    )
    return {**focus_targets, **replay_targets}


def _stable_row_order(
    rows: Sequence[dict[str, object]],
    *,
    family: str,
    seed: int,
) -> list[dict[str, object]]:
    def key(row: Mapping[str, object]) -> tuple[str, str]:
        metadata = row.get("metadata")
        example_id = str(metadata.get("example_id") or "") if isinstance(metadata, Mapping) else ""
        prompt = str(row.get("user") or "")
        digest = hashlib.sha256(
            f"{REPAIR_SCHEMA_VERSION}|{seed}|{family}|{example_id}|{prompt}".encode("utf-8")
        ).hexdigest()
        return digest, example_id

    return sorted(rows, key=key)


def _canonicalize_ratio_target(row: Mapping[str, object]) -> dict[str, object]:
    cloned = copy.deepcopy(dict(row))
    metadata = cloned.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Ratio source metadata must be mutable after cloning.")
    verifier_type = str(metadata.get("verifier_type") or "")
    expected = str(metadata.get("expected_answer") or "").strip()
    if verifier_type not in {"decimal", "fraction", "integer"} or not expected:
        return cloned
    original_assistant = cloned.get("assistant")
    cloned["assistant"] = f"Answer: {expected}."
    result = verify_candidate(cloned.get("user"), cloned.get("assistant"), metadata)
    if not result.valid_spec or not result.passed:
        cloned["assistant"] = original_assistant
        return cloned
    metadata["repair_schema"] = REPAIR_SCHEMA_VERSION
    metadata["repair_target_form"] = "compact_answer_first"
    return cloned


def _ratio_repair_variant(
    rows: Sequence[dict[str, object]],
    *,
    seed: int,
    variant_index: int,
) -> dict[str, object]:
    kind = "leading_zero_minimal_pair" if variant_index % 3 else "compact_fraction"
    verifier_type = "normalized_exact" if kind == "leading_zero_minimal_pair" else "fraction"
    exemplar_verifier_type = "decimal" if kind == "leading_zero_minimal_pair" else "fraction"
    exemplar: Optional[dict[str, object]] = None
    for row in rows:
        metadata = row.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and metadata.get("verifier_type") == exemplar_verifier_type
        ):
            exemplar = row
            break
    if exemplar is None:
        raise ValueError(f"Ratio repair requires a {exemplar_verifier_type} verifier exemplar.")
    cloned = copy.deepcopy(exemplar)
    metadata = cloned.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Ratio repair metadata must be mutable after cloning.")

    if kind == "leading_zero_minimal_pair":
        decimal_ordinal = variant_index - (variant_index // 3)
        pair_index, pair_member = divmod(decimal_ordinal - 1, 2)
        base_successes = 1 + ((int(seed) + pair_index * 37) % 88)
        successes = base_successes + pair_member
        trials = 1_000
        expected = format(Decimal(successes) / Decimal(trials), ".3f")
        template_id = "repair.train.ratios_probability.leading_zero_minimal_pair.v1"
        prompt = (
            f"Frontier decimal minimal pair {pair_index + 1:04d}{'A' if pair_member == 0 else 'B'}: "
            f"Out of {trials} checks, {successes} succeed. Write the observed success "
            "rate to three decimal places, including the leading zero, without a percent sign."
        )
        pair_id = f"decimal-pair-{pair_index + 1:04d}"
    else:
        fraction_ordinal = variant_index // 3
        denominator = 11 + ((int(seed) + fraction_ordinal * 5) % 37)
        numerator = 1 + ((int(seed) + fraction_ordinal * 11) % (denominator - 1))
        reduced = Fraction(numerator, denominator)
        expected = f"{reduced.numerator}/{reduced.denominator}"
        template_id = "repair.train.ratios_probability.compact_fraction.v1"
        prompt = (
            f"Frontier compact fraction case {fraction_ordinal:04d}: A batch has {numerator} "
            f"flagged items and {denominator - numerator} clear items. Give the probability "
            "of drawing a flagged item as one reduced fraction in a compact answer-first response."
        )
        pair_id = f"fraction-{fraction_ordinal:04d}"

    origin_id = str(metadata.get("example_id") or "").strip()
    metadata.update(
        {
            "absolute_tolerance": "0",
            "aliases_json": "[]",
            "curriculum_split": "train",
            "example_id": hashlib.sha256(
                f"{REPAIR_SCHEMA_VERSION}|{seed}|{template_id}|{prompt}".encode("utf-8")
            ).hexdigest()[:24],
            "expected_answer": expected,
            "problem_family": "ratios_probability",
            "repair_origin_example_id": origin_id,
            "repair_pair_id": pair_id,
            "repair_schema": REPAIR_SCHEMA_VERSION,
            "repair_target_form": (
                "exact_leading_zero_three_decimal"
                if kind == "leading_zero_minimal_pair"
                else "compact_answer_first"
            ),
            "repair_variant_index": int(variant_index),
            "repair_variant_kind": kind,
            "split_group": f"train:{template_id}",
            "template_id": template_id,
            "verified_correct": True,
            "verifier_schema": VERIFIER_SCHEMA_VERSION,
            "verifier_type": verifier_type,
        }
    )
    cloned.update(
        {
            "user": prompt,
            "assistant": (
                expected
                if kind == "leading_zero_minimal_pair"
                else f"Answer: {expected}."
            ),
            "source": REPAIR_SOURCE,
        }
    )
    result = verify_candidate(cloned.get("user"), cloned.get("assistant"), metadata)
    if not result.valid_spec or not result.passed:
        raise ValueError(f"Generated ratio repair failed verification: {result.reason}.")
    if kind == "leading_zero_minimal_pair":
        numeric_metadata = dict(metadata)
        numeric_metadata["verifier_type"] = "decimal"
        numeric_result = verify_candidate(
            cloned.get("user"),
            cloned.get("assistant"),
            numeric_metadata,
        )
        if not numeric_result.valid_spec or not numeric_result.passed:
            raise ValueError(
                f"Generated leading-zero repair failed numeric verification: "
                f"{numeric_result.reason}."
            )
    return cloned


def _repair_variant(
    row: Mapping[str, object],
    *,
    family: str,
    seed: int,
    variant_index: int,
) -> dict[str, object]:
    cloned = copy.deepcopy(dict(row))
    metadata = cloned.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Repair source metadata must be mutable after cloning.")
    origin_id = str(metadata.get("example_id") or "").strip()
    marker = hashlib.sha256(
        f"{REPAIR_SCHEMA_VERSION}|variant|{seed}|{family}|{origin_id}|{variant_index}".encode(
            "utf-8"
        )
    ).hexdigest()[:12]
    original_prompt = str(cloned.get("user") or "").strip()
    repaired_prompt = (
        f"Independent frontier-repair practice {marker}. Solve the original task from "
        f"scratch and preserve its requested answer format.\n\n{original_prompt}"
    )
    metadata["repair_origin_example_id"] = origin_id
    metadata["repair_variant_index"] = int(variant_index)
    metadata["repair_schema"] = REPAIR_SCHEMA_VERSION
    metadata["example_id"] = hashlib.sha256(
        f"{REPAIR_SCHEMA_VERSION}|{seed}|{origin_id}|{marker}|{repaired_prompt}".encode("utf-8")
    ).hexdigest()[:24]
    metadata["curriculum_split"] = "train"
    template_id = f"repair.train.{family}.replay.v1"
    metadata["template_id"] = template_id
    metadata["split_group"] = f"train:{template_id}"
    cloned["user"] = repaired_prompt
    cloned["source"] = REPAIR_SOURCE
    return cloned


def _select_repair_rows(
    rows: Sequence[dict[str, object]],
    *,
    family_targets: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    pools: dict[str, list[dict[str, object]]] = {family: [] for family in REQUIRED_FAMILIES}
    for row in rows:
        metadata = row.get("metadata")
        family = str(metadata.get("problem_family") or "") if isinstance(metadata, Mapping) else ""
        if family in pools:
            pools[family].append(row)
    missing = [family for family, pool in pools.items() if not pool]
    if missing:
        raise ValueError(f"Input train set is missing required families: {', '.join(missing)}")

    selected: list[dict[str, object]] = []
    augmented_counts: dict[str, int] = {}
    for family in REQUIRED_FAMILIES:
        target = int(family_targets[family])
        ordered = _stable_row_order(pools[family], family=family, seed=seed)
        originals = ordered[:target]
        if family == "ratios_probability":
            selected.extend(_canonicalize_ratio_target(row) for row in originals)
        else:
            selected.extend(copy.deepcopy(originals))
        shortfall = target - len(originals)
        augmented_counts[family] = max(0, shortfall)
        for offset in range(shortfall):
            if family == "ratios_probability":
                selected.append(
                    _ratio_repair_variant(
                        ordered,
                        seed=seed,
                        variant_index=offset + 1,
                    )
                )
            else:
                source = ordered[offset % len(ordered)]
                selected.append(
                    _repair_variant(
                        source,
                        family=family,
                        seed=seed,
                        variant_index=offset + 1,
                    )
                )

    selected.sort(
        key=lambda row: hashlib.sha256(
            (
                f"{REPAIR_SCHEMA_VERSION}|shuffle|{seed}|"
                f"{row['metadata']['example_id']}|{row['user']}"  # type: ignore[index]
            ).encode("utf-8")
        ).hexdigest()
    )
    return selected, dict(sorted(augmented_counts.items()))


def _summary(
    rows: Sequence[Mapping[str, object]],
    *,
    filename: str,
    payload: bytes,
    profile: _SplitProfile,
) -> dict[str, object]:
    return {
        "file": filename,
        "rows": len(rows),
        "sha256": _sha256_bytes(payload),
        "family_counts": profile.family_counts,
        "source_counts": profile.source_counts,
        "verifier_type_counts": profile.verifier_type_counts,
        "template_ids": sorted(profile.template_ids),
    }


def _required_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return digest


def _content_identity(
    *,
    parent_manifest_sha256: str,
    parent_train_sha256: str,
    parent_eval_sha256: str,
    seed: int,
    target_rows: int,
    focus_fraction: str,
    family_targets: Mapping[str, int],
    train_sha256: str,
) -> dict[str, object]:
    return {
        "repair_schema": REPAIR_SCHEMA_VERSION,
        "parent_manifest_sha256": parent_manifest_sha256,
        "parent_train_sha256": parent_train_sha256,
        "parent_eval_sha256": parent_eval_sha256,
        "seed": int(seed),
        "target_rows": int(target_rows),
        "focus_fraction": focus_fraction,
        "family_targets": dict(sorted(family_targets.items())),
        "train_sha256": train_sha256,
    }


def _content_artifact_names(
    identity: Mapping[str, object],
) -> tuple[str, str, str, str]:
    content_digest = _sha256_bytes(_canonical_json_bytes(identity))
    stem = f"{REPAIR_FILENAME_PREFIX}_{content_digest[:16]}"
    return (
        f"sha256:{content_digest}",
        f"{stem}.train.jsonl",
        f"{stem}.eval.jsonl",
        f"{stem}.manifest.json",
    )


def _identity_from_manifest(
    manifest: Mapping[str, object],
    *,
    train_sha256: str,
) -> dict[str, object]:
    if manifest.get("curriculum_schema") != REPAIR_SCHEMA_VERSION:
        raise ValueError("Repair manifest schema is not current.")
    if manifest.get("verifier_schema") != VERIFIER_SCHEMA_VERSION:
        raise ValueError("Repair manifest verifier schema is not current.")
    parent = manifest.get("parent")
    repair = manifest.get("repair")
    if not isinstance(parent, Mapping) or not isinstance(repair, Mapping):
        raise ValueError("Repair manifest is missing parent or repair identity metadata.")
    seed = manifest.get("seed")
    target_rows = repair.get("target_rows")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Repair manifest seed must be an integer.")
    if isinstance(target_rows, bool) or not isinstance(target_rows, int) or target_rows < 1:
        raise ValueError("Repair manifest target_rows must be a positive integer.")
    raw_focus_fraction = str(repair.get("requested_focus_fraction") or "").strip()
    parsed_focus_fraction = _parse_focus_fraction(raw_focus_fraction)
    canonical_focus_fraction = format(parsed_focus_fraction, "f")
    if raw_focus_fraction != canonical_focus_fraction:
        raise ValueError("Repair manifest focus fraction is not canonical.")
    family_targets = _manifest_count_map(
        repair.get("family_targets"),
        label="repair.family_targets",
    )
    if set(family_targets) != set(REQUIRED_FAMILIES) or sum(family_targets.values()) != target_rows:
        raise ValueError("Repair manifest family targets do not match its target_rows.")
    return _content_identity(
        parent_manifest_sha256=_required_sha256(
            parent.get("manifest_sha256"),
            label="parent.manifest_sha256",
        ),
        parent_train_sha256=_required_sha256(
            parent.get("train_sha256"),
            label="parent.train_sha256",
        ),
        parent_eval_sha256=_required_sha256(
            parent.get("eval_sha256"),
            label="parent.eval_sha256",
        ),
        seed=seed,
        target_rows=target_rows,
        focus_fraction=canonical_focus_fraction,
        family_targets=family_targets,
        train_sha256=_required_sha256(train_sha256, label="train.sha256"),
    )


def build_repair_curriculum(
    *,
    train_jsonl_path: Path | str,
    manifest_path: Path | str,
    eval_jsonl_path: Path | str,
    seed: int = 5202,
    target_rows: int = 480,
    focus_fraction: float | str | Decimal = Decimal("0.55"),
) -> RepairCurriculumBundle:
    """Derive one deterministic repair set and a gate-consumable manifest."""

    train_path = Path(train_jsonl_path).expanduser().resolve()
    parent_manifest_path = Path(manifest_path).expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Input train JSONL not found: {train_path}")
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"Input manifest not found: {parent_manifest_path}")

    parent_manifest, parent_manifest_bytes = _load_json_mapping(parent_manifest_path)
    if parent_manifest.get("all_targets_verified") is not True:
        raise ValueError("Parent manifest does not attest that all targets are verified.")
    if parent_manifest.get("verifier_schema") != VERIFIER_SCHEMA_VERSION:
        raise ValueError("Parent manifest verifier schema is not current.")
    train_summary = parent_manifest.get("train")
    eval_summary = parent_manifest.get("eval")
    if not isinstance(train_summary, Mapping) or not isinstance(eval_summary, Mapping):
        raise ValueError("Parent manifest must declare train and eval summaries.")
    declared_train_path = _safe_manifest_artifact(parent_manifest_path, train_summary.get("file"))
    if declared_train_path != train_path:
        raise ValueError("Input train JSONL is not the artifact declared by the manifest.")
    parent_eval_path = _safe_manifest_artifact(parent_manifest_path, eval_summary.get("file"))
    requested_eval_path = Path(eval_jsonl_path).expanduser().resolve()
    if requested_eval_path != parent_eval_path:
        raise ValueError("Input eval JSONL is not the artifact declared by the manifest.")

    train_rows, train_bytes = _load_jsonl_rows(train_path)
    eval_rows, eval_bytes = _load_jsonl_rows(parent_eval_path)
    train_profile = _validate_rows(train_rows, split="train")
    eval_profile = _validate_rows(eval_rows, split="eval")
    _verify_manifest_split(
        train_summary,
        rows=train_rows,
        payload=train_bytes,
        profile=train_profile,
        label="train",
    )
    _verify_manifest_split(
        eval_summary,
        rows=eval_rows,
        payload=eval_bytes,
        profile=eval_profile,
        label="eval",
    )
    if not train_profile.template_ids.isdisjoint(eval_profile.template_ids):
        raise ValueError("Parent train template IDs overlap manifest eval template IDs.")
    if not train_profile.prompts.isdisjoint(eval_profile.prompts):
        raise ValueError("Parent train prompts overlap manifest eval prompts.")
    if not train_profile.example_ids.isdisjoint(eval_profile.example_ids):
        raise ValueError("Parent train example IDs overlap manifest eval example IDs.")
    if set(train_profile.family_counts) != set(REQUIRED_FAMILIES):
        raise ValueError("Parent train family set does not match the required family registry.")
    if set(eval_profile.family_counts) != set(REQUIRED_FAMILIES):
        raise ValueError("Parent eval family set does not match the required family registry.")

    parsed_focus_fraction = _parse_focus_fraction(focus_fraction)
    family_targets = _family_targets(int(target_rows), parsed_focus_fraction)
    selected, augmented_counts = _select_repair_rows(
        train_rows,
        family_targets=family_targets,
        seed=int(seed),
    )
    repair_profile = _validate_rows(selected, split="train")
    if repair_profile.family_counts != dict(sorted(family_targets.items())):
        raise RuntimeError("Repair family counts do not match their deterministic targets.")
    if not repair_profile.template_ids.isdisjoint(eval_profile.template_ids):
        raise RuntimeError("Repair template IDs overlap manifest eval template IDs.")
    if not repair_profile.prompts.isdisjoint(eval_profile.prompts):
        raise RuntimeError("Repair prompts overlap manifest eval prompts.")
    if not repair_profile.example_ids.isdisjoint(eval_profile.example_ids):
        raise RuntimeError("Repair example IDs overlap manifest eval example IDs.")
    focus_rows = sum(repair_profile.family_counts[family] for family in FOCUS_FAMILIES)
    if focus_rows * 2 < len(selected):
        raise RuntimeError("Repair curriculum focus share fell below fifty percent.")
    canonicalized_targets = 0
    repair_variant_counts: Counter[str] = Counter()
    for row in selected:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        if metadata.get("repair_target_form") == "compact_answer_first":
            canonicalized_targets += 1
        variant_kind = str(metadata.get("repair_variant_kind") or "").strip()
        if variant_kind:
            repair_variant_counts[variant_kind] += 1

    repair_train_bytes = _jsonl_bytes(selected)
    parent_manifest_sha = _sha256_bytes(parent_manifest_bytes)
    parent_train_sha = _sha256_bytes(train_bytes)
    parent_eval_sha = _sha256_bytes(eval_bytes)
    train_sha = _sha256_bytes(repair_train_bytes)
    identity = _content_identity(
        parent_manifest_sha256=parent_manifest_sha,
        parent_train_sha256=parent_train_sha,
        parent_eval_sha256=parent_eval_sha,
        seed=int(seed),
        target_rows=int(target_rows),
        focus_fraction=format(parsed_focus_fraction, "f"),
        family_targets=family_targets,
        train_sha256=train_sha,
    )
    content_id, train_filename, eval_filename, manifest_filename = _content_artifact_names(
        identity
    )

    manifest: dict[str, object] = {
        "curriculum_schema": REPAIR_SCHEMA_VERSION,
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "content_id": content_id,
        "manifest_file": manifest_filename,
        "seed": int(seed),
        "all_targets_verified": True,
        "template_ids_disjoint": True,
        "prompt_text_disjoint": True,
        "example_ids_disjoint": True,
        "parent": {
            "curriculum_schema": str(parent_manifest.get("curriculum_schema") or ""),
            "seed": parent_manifest.get("seed"),
            "manifest_file": parent_manifest_path.name,
            "manifest_sha256": parent_manifest_sha,
            "train_file": str(train_summary.get("file") or ""),
            "train_sha256": parent_train_sha,
            "eval_file": str(eval_summary.get("file") or ""),
            "eval_sha256": parent_eval_sha,
        },
        "repair": {
            "target_rows": len(selected),
            "requested_focus_fraction": format(parsed_focus_fraction, "f"),
            "actual_focus_rows": focus_rows,
            "actual_focus_fraction": focus_rows / len(selected),
            "focus_families": list(FOCUS_FAMILIES),
            "priority_replay_families": list(PRIORITY_REPLAY_FAMILIES),
            "required_families": list(REQUIRED_FAMILIES),
            "family_targets": dict(sorted(family_targets.items())),
            "augmented_counts": augmented_counts,
            "canonicalized_answer_first_targets": canonicalized_targets,
            "generated_variant_counts": dict(sorted(repair_variant_counts.items())),
        },
        "train": _summary(
            selected,
            filename=train_filename,
            payload=repair_train_bytes,
            profile=repair_profile,
        ),
        "eval": {
            **_summary(
                eval_rows,
                filename=eval_filename,
                payload=eval_bytes,
                profile=eval_profile,
            ),
            "byte_identical_to_parent": True,
            "parent_file": str(eval_summary.get("file") or ""),
            "parent_sha256": parent_eval_sha,
        },
    }
    return RepairCurriculumBundle(tuple(selected), eval_bytes, manifest)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_repair_curriculum(
    bundle: RepairCurriculumBundle,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    train_summary = bundle.manifest.get("train")
    eval_summary = bundle.manifest.get("eval")
    if not isinstance(train_summary, Mapping) or not isinstance(eval_summary, Mapping):
        raise ValueError("Repair manifest is missing train or eval summaries.")
    if (
        bundle.manifest.get("all_targets_verified") is not True
        or bundle.manifest.get("template_ids_disjoint") is not True
        or bundle.manifest.get("prompt_text_disjoint") is not True
        or bundle.manifest.get("example_ids_disjoint") is not True
    ):
        raise ValueError("Repair manifest is missing required validation attestations.")

    train_payload = _jsonl_bytes(bundle.train_rows)
    train_profile = _validate_rows(bundle.train_rows, split="train")
    _verify_manifest_split(
        train_summary,
        rows=bundle.train_rows,
        payload=train_payload,
        profile=train_profile,
        label="repair train",
    )
    eval_rows = _decode_jsonl_rows(bundle.eval_bytes, label="repair eval payload")
    eval_profile = _validate_rows(eval_rows, split="eval")
    _verify_manifest_split(
        eval_summary,
        rows=eval_rows,
        payload=bundle.eval_bytes,
        profile=eval_profile,
        label="repair eval",
    )
    if set(eval_profile.family_counts) != set(REQUIRED_FAMILIES):
        raise ValueError("Repair eval family set does not match the required family registry.")

    train_sha = _sha256_bytes(train_payload)
    eval_sha = _sha256_bytes(bundle.eval_bytes)
    parent = bundle.manifest.get("parent")
    if not isinstance(parent, Mapping):
        raise ValueError("Repair manifest is missing parent identity metadata.")
    if (
        eval_summary.get("byte_identical_to_parent") is not True
        or _required_sha256(eval_summary.get("sha256"), label="eval.sha256") != eval_sha
        or _required_sha256(eval_summary.get("parent_sha256"), label="eval.parent_sha256")
        != eval_sha
        or _required_sha256(parent.get("eval_sha256"), label="parent.eval_sha256")
        != eval_sha
    ):
        raise ValueError("Repair eval hash chain does not prove byte-identical parent evidence.")

    identity = _identity_from_manifest(bundle.manifest, train_sha256=train_sha)
    if (
        len(bundle.train_rows) != identity["target_rows"]
        or train_profile.family_counts != identity["family_targets"]
    ):
        raise ValueError("Repair train payload does not match its canonical family targets.")
    expected_content_id, train_filename, eval_filename, manifest_filename = (
        _content_artifact_names(identity)
    )
    if str(bundle.manifest.get("content_id") or "") != expected_content_id:
        raise ValueError("Repair manifest content_id is not canonical.")
    if (
        str(train_summary.get("file") or "") != train_filename
        or str(eval_summary.get("file") or "") != eval_filename
        or str(bundle.manifest.get("manifest_file") or "") != manifest_filename
    ):
        raise ValueError("Repair manifest artifact filenames are not canonical.")

    manifest_payload = (
        json.dumps(bundle.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    payloads = {
        "train_jsonl": (root / train_filename, train_payload),
        "eval_jsonl": (root / eval_filename, bundle.eval_bytes),
        "manifest_json": (root / manifest_filename, manifest_payload),
    }
    if not overwrite:
        existing = [path for path, _payload in payloads.values() if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite repair artifact: {existing[0]}")
    for path, payload in payloads.values():
        _write_atomic(path, payload)
    return {name: str(path) for name, (path, _payload) in payloads.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic frontier-repair general-intelligence curriculum."
    )
    parser.add_argument(
        "--train-jsonl",
        default=str(DEFAULT_SOURCE_DIR / DEFAULT_TRAIN_FILENAME),
    )
    parser.add_argument(
        "--eval-jsonl",
        default=str(DEFAULT_SOURCE_DIR / DEFAULT_EVAL_FILENAME),
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_SOURCE_DIR / DEFAULT_MANIFEST_FILENAME),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=5202)
    parser.add_argument("--target-rows", type=int, default=480)
    parser.add_argument("--focus-fraction", default="0.55")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    bundle = build_repair_curriculum(
        train_jsonl_path=args.train_jsonl,
        eval_jsonl_path=args.eval_jsonl,
        manifest_path=args.manifest,
        seed=int(args.seed),
        target_rows=int(args.target_rows),
        focus_fraction=str(args.focus_fraction),
    )
    paths = write_repair_curriculum(
        bundle,
        args.output_dir,
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "curriculum_schema": REPAIR_SCHEMA_VERSION,
                "content_id": bundle.manifest["content_id"],
                "train_rows": len(bundle.train_rows),
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
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_EVAL_FILENAME",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_SOURCE_DIR",
    "DEFAULT_TRAIN_FILENAME",
    "FOCUS_FAMILIES",
    "PRIORITY_REPLAY_FAMILIES",
    "REPAIR_FILENAME_PREFIX",
    "REPAIR_SCHEMA_VERSION",
    "REQUIRED_FAMILIES",
    "RepairCurriculumBundle",
    "build_repair_curriculum",
    "main",
    "parse_args",
    "write_repair_curriculum",
]
