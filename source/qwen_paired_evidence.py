"""Recomputable paired promotion evidence for Qwen evaluations.

The benchmark runner stores detailed base and tuned generations for the same
trusted held-out rows.  This module validates that one-to-one alignment,
re-runs the versioned verifier on each stored prediction, and derives the
paired statistics used by the promotion gate.  Stored ``verified_correct``
flags are deliberately ignored.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from difflib import SequenceMatcher
from math import comb, gcd
from typing import Mapping, Sequence

try:
    from verifiable_reasoning import verify_candidate
except ImportError:  # pragma: no cover - package import path
    from .verifiable_reasoning import verify_candidate


PAIRED_EVIDENCE_SCHEMA_VERSION = "supermix-qwen-paired-evidence-v1"
DEFAULT_BOOTSTRAP_SEED = 5203
DEFAULT_BOOTSTRAP_RESAMPLES = 5_000
MIN_BOOTSTRAP_RESAMPLES = 100
MAX_BOOTSTRAP_RESAMPLES = 100_000

_IDENTITY_FIELDS = ("example_id", "template_id", "split_group", "problem_family")
_ARTIFACT_HASH_FIELDS = (
    "base_samples_sha256",
    "tuned_samples_sha256",
    "sample_comparison_sha256",
)
_CORE_METRIC_KEYS = (
    "eval_samples",
    "eval_loss",
    "perplexity",
    "token_f1",
    "char_similarity",
    "avg_generated_tokens",
    "total_generated_tokens",
    "generation_cap",
    "generation_cap_hits",
    "generation_cap_rate",
    "verified_samples",
    "verified_accuracy",
)


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Paired evidence contains a non-canonical JSON value.") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def paired_evidence_sha256(evidence: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 identity of a paired evidence payload."""

    if not isinstance(evidence, Mapping):
        raise TypeError("Paired evidence must be a mapping.")
    return _sha256_json(dict(evidence))


def _required_sha256(value: object, *, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{label} must be a SHA-256 digest.")
    return digest


def _artifact_hashes(value: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("artifact_hashes must be a mapping.")
    return {
        field: _required_sha256(value.get(field), label=f"artifact_hashes.{field}")
        for field in _ARTIFACT_HASH_FIELDS
    }


def _mapping_or_attributes(value: object, *, index: int) -> tuple[str, str, str, Mapping[str, object]]:
    if isinstance(value, Mapping):
        user = value.get("user")
        assistant = value.get("assistant")
        source = value.get("source", "")
        metadata = value.get("metadata")
    else:
        user = getattr(value, "user", None)
        assistant = getattr(value, "assistant", None)
        source = getattr(value, "source", "")
        metadata = getattr(value, "metadata", None)
    if not isinstance(user, str) or not user.strip():
        raise ValueError(f"Trusted eval row {index} has an invalid user value.")
    if not isinstance(assistant, str) or not assistant.strip():
        raise ValueError(f"Trusted eval row {index} has an invalid assistant reference.")
    if not isinstance(source, str):
        raise ValueError(f"Trusted eval row {index} has an invalid source value.")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"Trusted eval row {index} is missing verifier metadata.")
    return (
        user.strip(),
        assistant.strip(),
        source.strip() or "dataset",
        _normalize_eval_metadata(metadata, index=index),
    )


def _normalize_eval_metadata(
    metadata: Mapping[object, object],
    *,
    index: int,
) -> dict[str, object]:
    """Mirror the evaluator's saved ChatPair metadata normalization."""

    normalized: dict[str, object] = {}
    for raw_key, raw_value in metadata.items():
        key = "" if raw_key is None else str(raw_key).strip()
        if not key:
            continue
        if key in normalized:
            raise ValueError(
                f"Trusted eval row {index} contains colliding normalized metadata keys."
            )
        if isinstance(raw_value, str):
            value = raw_value.strip()
            if value:
                normalized[key] = value
        elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
            normalized[key] = raw_value
        elif key == "verifier_spec" and isinstance(raw_value, Mapping):
            verifier_spec = _normalize_eval_metadata(raw_value, index=index)
            if verifier_spec:
                normalized[key] = verifier_spec
    return normalized


def _metadata_identity(metadata: Mapping[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Trusted eval metadata {key} must be a non-empty string.")
    return value.strip()


def _trusted_eval_rows(eval_rows: Sequence[object]) -> list[dict[str, object]]:
    if isinstance(eval_rows, (str, bytes)) or not isinstance(eval_rows, Sequence):
        raise TypeError("eval_rows must be a sequence of ChatPair-like objects or mappings.")
    trusted: list[dict[str, object]] = []
    example_ids: set[str] = set()
    for index, raw_row in enumerate(eval_rows):
        user, reference, source, metadata = _mapping_or_attributes(raw_row, index=index)
        identity = {key: _metadata_identity(metadata, key) for key in _IDENTITY_FIELDS}
        example_id = identity["example_id"]
        if example_id in example_ids:
            raise ValueError(f"Trusted eval rows contain duplicate example_id {example_id!r}.")
        example_ids.add(example_id)
        reference_verification = verify_candidate(user, reference, metadata)
        if not reference_verification.valid_spec or not reference_verification.passed:
            raise ValueError(
                f"Trusted eval row {index} reference does not pass its verifier specification."
            )
        family = identity["problem_family"]
        trusted.append(
            {
                "sample_index": index,
                "user": user,
                "reference": reference,
                "source": source,
                "metadata": dict(metadata),
                "evidence_family": family,
                **identity,
            }
        )
    if not trusted:
        raise ValueError("Paired evidence requires at least one trusted eval row.")
    return trusted


def _canonical_eval_row(trusted: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "user": trusted["user"],
        "assistant": trusted["reference"],
        "source": trusted["source"],
    }
    metadata = trusted.get("metadata")
    if isinstance(metadata, Mapping) and metadata:
        row["metadata"] = dict(metadata)
    return row


def deterministic_eval_selection(
    eval_rows: Sequence[object],
    *,
    seed: int,
    samples_per_family: int,
    max_eval_samples: int,
) -> list[dict[str, object]]:
    """Normalize, validate, and replay the evaluator's exact row selection.

    The returned mappings have the same logical shape and ordering as
    ``qwen_supermix_pipeline.save_jsonl`` after ``load_saved_chat_pairs``.
    Identity completeness and unique ``example_id`` values are checked before
    sampling, so this also validates the entire bound curriculum when called on
    its rows.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Evaluation selection seed must be an integer.")
    per_family = _nonnegative_int(
        samples_per_family,
        label="samples_per_family",
    )
    maximum = _nonnegative_int(max_eval_samples, label="max_eval_samples")
    if per_family > 0 and maximum > 0:
        raise ValueError("Use either samples_per_family or max_eval_samples, not both.")

    trusted = _trusted_eval_rows(eval_rows)
    normalized = [_canonical_eval_row(row) for row in trusted]
    selected = list(normalized)
    if per_family > 0:
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in normalized:
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):  # pragma: no cover - guarded above
                raise ValueError("Trusted eval row is missing normalized metadata.")
            family = str(metadata["problem_family"])
            grouped.setdefault(family, []).append(row)
        selected = []
        for family in sorted(grouped):
            family_seed = int.from_bytes(
                hashlib.sha256(f"{seed}|{family}".encode("utf-8")).digest()[:8],
                "big",
            )
            rows = list(grouped[family])
            random.Random(family_seed).shuffle(rows)
            selected.extend(rows[:per_family])
    if maximum > 0 and len(selected) > maximum:
        selected = random.Random(seed + 101).sample(selected, maximum)
    return selected


def _sample_index(value: object, *, side: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{side} sample_index must be a non-negative integer.")
    return value


def _indexed_samples(
    rows: Sequence[Mapping[str, object]],
    *,
    side: str,
    expected_count: int,
) -> dict[int, Mapping[str, object]]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError(f"{side} samples must be a sequence of mappings.")
    indexed: dict[int, Mapping[str, object]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{side} samples must contain only mappings.")
        index = _sample_index(raw_row.get("sample_index"), side=side)
        if index in indexed:
            raise ValueError(f"{side} samples contain duplicate sample_index {index}.")
        indexed[index] = raw_row
    expected_indices = set(range(expected_count))
    if set(indexed) != expected_indices:
        missing = sorted(expected_indices - set(indexed))
        unexpected = sorted(set(indexed) - expected_indices)
        raise ValueError(
            f"{side} samples are not a unique complete eval alignment "
            f"(missing={missing[:8]}, unexpected={unexpected[:8]})."
        )
    return indexed


def _finite_float(value: object, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{label} must be a finite number >= {minimum}.")
    return parsed


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


def _positive_int(value: object, *, label: str) -> int:
    parsed = _nonnegative_int(value, label=label)
    if parsed < 1:
        raise ValueError(f"{label} must be positive.")
    return parsed


def _comparison_float(value: object, *, label: str) -> float:
    if value is None:
        return 0.0
    return _finite_float(value, label=label)


def _comparison_int(value: object, *, label: str) -> int:
    if value is None:
        return 0
    return _nonnegative_int(value, label=label)


def derive_sample_comparison(
    base_samples: Sequence[Mapping[str, object]],
    tuned_samples: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Strictly derive the evaluator's sample-comparison artifact."""

    expected_count = len(base_samples)
    base_by_index = _indexed_samples(
        base_samples,
        side="base",
        expected_count=expected_count,
    )
    tuned_by_index = _indexed_samples(
        tuned_samples,
        side="tuned",
        expected_count=expected_count,
    )
    rows: list[dict[str, object]] = []
    for base_row in base_samples:
        sample_index = _sample_index(base_row.get("sample_index"), side="base")
        # Indexing both mappings above proves complete one-to-one alignment.
        tuned_row = tuned_by_index[sample_index]
        base_row = base_by_index[sample_index]
        base_f1 = _comparison_float(
            base_row.get("token_f1"),
            label=f"base sample {sample_index} token_f1",
        )
        tuned_f1 = _comparison_float(
            tuned_row.get("token_f1"),
            label=f"tuned sample {sample_index} token_f1",
        )
        base_char = _comparison_float(
            base_row.get("char_similarity"),
            label=f"base sample {sample_index} char_similarity",
        )
        tuned_char = _comparison_float(
            tuned_row.get("char_similarity"),
            label=f"tuned sample {sample_index} char_similarity",
        )
        base_gen = _comparison_float(
            base_row.get("gen_seconds"),
            label=f"base sample {sample_index} gen_seconds",
        )
        tuned_gen = _comparison_float(
            tuned_row.get("gen_seconds"),
            label=f"tuned sample {sample_index} gen_seconds",
        )
        base_generated_tokens = _comparison_int(
            base_row.get("generated_tokens"),
            label=f"base sample {sample_index} generated_tokens",
        )
        tuned_generated_tokens = _comparison_int(
            tuned_row.get("generated_tokens"),
            label=f"tuned sample {sample_index} generated_tokens",
        )
        rows.append(
            {
                "sample_index": sample_index,
                "source": str(base_row.get("source", tuned_row.get("source", "")) or ""),
                "example_id": str(
                    base_row.get("example_id", tuned_row.get("example_id", "")) or ""
                ),
                "template_id": str(
                    base_row.get("template_id", tuned_row.get("template_id", "")) or ""
                ),
                "split_group": str(
                    base_row.get("split_group", tuned_row.get("split_group", "")) or ""
                ),
                "problem_family": str(
                    base_row.get("problem_family", tuned_row.get("problem_family", "")) or ""
                ),
                "prompt_signature": str(
                    base_row.get("prompt_signature", tuned_row.get("prompt_signature", "")) or ""
                ),
                "prompt_complexity": _comparison_float(
                    base_row.get(
                        "prompt_complexity",
                        tuned_row.get("prompt_complexity", 0.0),
                    ),
                    label=f"base sample {sample_index} prompt_complexity",
                ),
                "user": str(base_row.get("user", tuned_row.get("user", "")) or ""),
                "reference": str(
                    base_row.get("reference", tuned_row.get("reference", "")) or ""
                ),
                "base_prediction": str(base_row.get("prediction", "") or ""),
                "tuned_prediction": str(tuned_row.get("prediction", "") or ""),
                "base_loss": _comparison_float(
                    base_row.get("loss"),
                    label=f"base sample {sample_index} loss",
                ),
                "tuned_loss": _comparison_float(
                    tuned_row.get("loss"),
                    label=f"tuned sample {sample_index} loss",
                ),
                "base_token_f1": base_f1,
                "tuned_token_f1": tuned_f1,
                "delta_token_f1": tuned_f1 - base_f1,
                "base_char_similarity": base_char,
                "tuned_char_similarity": tuned_char,
                "delta_char_similarity": tuned_char - base_char,
                "base_gen_seconds": base_gen,
                "tuned_gen_seconds": tuned_gen,
                "delta_gen_seconds": tuned_gen - base_gen,
                "base_generated_tokens": base_generated_tokens,
                "tuned_generated_tokens": tuned_generated_tokens,
                "delta_generated_tokens": tuned_generated_tokens - base_generated_tokens,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row["delta_token_f1"]),
            float(row["delta_char_similarity"]),
            -float(row["delta_gen_seconds"]),
            int(row["sample_index"]),
        )
    )
    return rows


def validate_sample_comparison(
    comparison_rows: Sequence[Mapping[str, object]],
    base_samples: Sequence[Mapping[str, object]],
    tuned_samples: Sequence[Mapping[str, object]],
) -> None:
    """Require the comparison JSONL to be the exact complete derivation."""

    if isinstance(comparison_rows, (str, bytes)) or not isinstance(comparison_rows, Sequence):
        raise TypeError("comparison_rows must be a sequence of mappings.")
    if any(not isinstance(row, Mapping) for row in comparison_rows):
        raise ValueError("Comparison artifact must contain only mappings.")
    expected = derive_sample_comparison(base_samples, tuned_samples)
    actual = [dict(row) for row in comparison_rows]
    if _canonical_json_bytes(actual) != _canonical_json_bytes(expected):
        raise ValueError(
            "Sample comparison artifact is not the exact complete base/tuned derivation."
        )


def _token_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    if not reference_tokens and not hypothesis_tokens:
        return 1.0
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in reference_tokens:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in hypothesis_tokens:
        available = counts.get(token, 0)
        if available > 0:
            overlap += 1
            counts[token] = available - 1
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _metric_family_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "unknown"


def _validated_sample(
    row: Mapping[str, object],
    trusted: Mapping[str, object],
    *,
    side: str,
) -> dict[str, object]:
    index = int(trusted["sample_index"])
    for sample_key, trusted_key in (
        ("user", "user"),
        ("reference", "reference"),
        ("source", "source"),
    ):
        actual = row.get(sample_key)
        expected = trusted[trusted_key]
        if not isinstance(actual, str) or actual != expected:
            raise ValueError(f"{side} sample {index} does not match trusted {sample_key} identity.")
    for key in _IDENTITY_FIELDS:
        actual = row.get(key, "")
        if actual is None:
            actual = ""
        if not isinstance(actual, str) or actual.strip() != trusted[key]:
            raise ValueError(f"{side} sample {index} does not match trusted {key} identity.")

    prediction = row.get("prediction")
    if not isinstance(prediction, str):
        raise ValueError(f"{side} sample {index} prediction must be text.")
    metadata = trusted.get("metadata")
    if not isinstance(metadata, Mapping):  # pragma: no cover - established above
        raise ValueError(f"Trusted eval row {index} lost verifier metadata.")
    verification = verify_candidate(trusted["user"], prediction, metadata)
    if not verification.valid_spec:
        raise ValueError(f"Trusted eval row {index} has an invalid verifier specification.")

    generation_cap = _positive_int(
        row.get("generation_cap"),
        label=f"{side} sample {index} generation_cap",
    )
    generated_tokens = _nonnegative_int(
        row.get("generated_tokens"),
        label=f"{side} sample {index} generated_tokens",
    )
    if generated_tokens > generation_cap:
        raise ValueError(f"{side} sample {index} exceeds its generation cap.")
    generation_cap_hit = row.get("generation_cap_hit")
    if not isinstance(generation_cap_hit, bool):
        raise ValueError(f"{side} sample {index} generation_cap_hit must be boolean.")
    if generation_cap_hit and generated_tokens < generation_cap:
        raise ValueError(f"{side} sample {index} reports an impossible generation cap hit.")

    reference = str(trusted["reference"])
    return {
        "sample_index": index,
        "template_id": trusted["template_id"],
        "problem_family": trusted["evidence_family"],
        "correct": bool(verification.passed),
        "loss": _finite_float(row.get("loss"), label=f"{side} sample {index} loss", minimum=0.0),
        "token_f1": _token_f1(reference, prediction),
        "char_similarity": float(SequenceMatcher(None, reference, prediction).ratio()),
        "generation_cap": generation_cap,
        "generation_cap_hit": generation_cap_hit,
        "generated_tokens": generated_tokens,
        "prediction_sha256": hashlib.sha256(prediction.encode("utf-8")).hexdigest(),
    }


def _side_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    count = len(rows)
    caps = {int(row["generation_cap"]) for row in rows}
    if len(caps) != 1:
        raise ValueError("Detailed samples must use one generation cap per evaluation side.")
    generation_cap = next(iter(caps))
    correct_count = sum(bool(row["correct"]) for row in rows)
    cap_hits = sum(bool(row["generation_cap_hit"]) for row in rows)
    total_generated_tokens = sum(int(row["generated_tokens"]) for row in rows)
    mean_loss = sum(float(row["loss"]) for row in rows) / count
    metrics: dict[str, float | int] = {
        "eval_samples": count,
        "eval_loss": mean_loss,
        "perplexity": math.exp(min(20.0, mean_loss)),
        "token_f1": sum(float(row["token_f1"]) for row in rows) / count,
        "char_similarity": sum(float(row["char_similarity"]) for row in rows) / count,
        "avg_generated_tokens": total_generated_tokens / count,
        "total_generated_tokens": total_generated_tokens,
        "generation_cap": generation_cap,
        "generation_cap_hits": cap_hits,
        "generation_cap_rate": cap_hits / count,
        "verified_samples": count,
        "verified_accuracy": correct_count / count,
    }
    family_rows: dict[str, list[Mapping[str, object]]] = {}
    normalized_families: dict[str, str] = {}
    for row in rows:
        family = str(row["problem_family"])
        metric_family = _metric_family_name(family)
        existing = normalized_families.get(metric_family)
        if existing is not None and existing != family:
            raise ValueError(
                f"Problem families {existing!r} and {family!r} collide after metric normalization."
            )
        normalized_families[metric_family] = family
        family_rows.setdefault(family, []).append(row)
    for metric_family, family in sorted(normalized_families.items()):
        grouped = family_rows[family]
        family_count = len(grouped)
        metrics[f"verified_samples_family_{metric_family}"] = family_count
        metrics[f"verified_accuracy_family_{metric_family}"] = (
            sum(bool(row["correct"]) for row in grouped) / family_count
        )
    return metrics


def _transition_summary(records: Sequence[Mapping[str, object]]) -> dict[str, float | int]:
    base_correct = sum(bool(row["base_correct"]) for row in records)
    tuned_correct = sum(bool(row["tuned_correct"]) for row in records)
    wins = sum(not bool(row["base_correct"]) and bool(row["tuned_correct"]) for row in records)
    regressions = sum(bool(row["base_correct"]) and not bool(row["tuned_correct"]) for row in records)
    both_correct = sum(bool(row["base_correct"]) and bool(row["tuned_correct"]) for row in records)
    both_incorrect = sum(
        not bool(row["base_correct"]) and not bool(row["tuned_correct"]) for row in records
    )
    count = len(records)
    return {
        "samples": count,
        "base_correct": base_correct,
        "tuned_correct": tuned_correct,
        "wins": wins,
        "regressions": regressions,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "ties": both_correct + both_incorrect,
        "discordant_pairs": wins + regressions,
        "base_accuracy": base_correct / count,
        "tuned_accuracy": tuned_correct / count,
        "accuracy_delta": (tuned_correct - base_correct) / count,
    }


def _exact_one_sided_mcnemar(*, wins: int, regressions: int) -> dict[str, object]:
    discordant = int(wins) + int(regressions)
    if discordant == 0:
        numerator, denominator = 1, 1
    else:
        denominator = 1 << discordant
        numerator = sum(comb(discordant, successes) for successes in range(int(wins), discordant + 1))
        divisor = gcd(numerator, denominator)
        numerator //= divisor
        denominator //= divisor
    return {
        "method": "exact_one_sided_mcnemar_binomial",
        "alternative": "tuned_accuracy_greater",
        "wins": int(wins),
        "regressions": int(regressions),
        "discordant_pairs": discordant,
        "p_value": numerator / denominator,
        "p_value_numerator": str(numerator),
        "p_value_denominator": str(denominator),
    }


def _percentile_r7(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("Bootstrap produced no values.")
    position = (len(sorted_values) - 1) * quantile
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )


def _template_cluster_bootstrap(
    records: Sequence[Mapping[str, object]],
    *,
    seed: int,
    resamples: int,
) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("bootstrap_seed must be an integer.")
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or not MIN_BOOTSTRAP_RESAMPLES <= resamples <= MAX_BOOTSTRAP_RESAMPLES
    ):
        raise ValueError(
            f"bootstrap_resamples must be between {MIN_BOOTSTRAP_RESAMPLES} "
            f"and {MAX_BOOTSTRAP_RESAMPLES}."
        )
    clusters: dict[str, list[int]] = {}
    for row in records:
        template_id = str(row.get("template_id") or "").strip()
        if not template_id:
            raise ValueError("Template-cluster bootstrap requires template_id on every row.")
        delta = int(bool(row["tuned_correct"])) - int(bool(row["base_correct"]))
        clusters.setdefault(template_id, []).append(delta)
    template_ids = sorted(clusters)
    if not template_ids:
        raise ValueError("Template-cluster bootstrap requires at least one cluster.")
    rng = random.Random(seed)
    estimates: list[float] = []
    cluster_count = len(template_ids)
    for _ in range(resamples):
        total_delta = 0
        total_rows = 0
        for _cluster_draw in range(cluster_count):
            selected = template_ids[rng.randrange(cluster_count)]
            values = clusters[selected]
            total_delta += sum(values)
            total_rows += len(values)
        estimates.append(total_delta / total_rows)
    estimates.sort()
    return {
        "method": "paired_template_cluster_percentile",
        "metric": "verified_accuracy_delta",
        "seed": seed,
        "resamples": resamples,
        "confidence_level": 0.95,
        "quantile_method": "linear_interpolation_r7",
        "cluster_count": cluster_count,
        "lower_95": _percentile_r7(estimates, 0.025),
        "upper_95": _percentile_r7(estimates, 0.975),
    }


def build_paired_evidence(
    base_samples: Sequence[Mapping[str, object]],
    tuned_samples: Sequence[Mapping[str, object]],
    eval_rows: Sequence[object],
    *,
    artifact_hashes: Mapping[str, object],
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
) -> dict[str, object]:
    """Build fail-closed paired evidence from detailed model generations."""

    trusted = _trusted_eval_rows(eval_rows)
    base_by_index = _indexed_samples(base_samples, side="base", expected_count=len(trusted))
    tuned_by_index = _indexed_samples(tuned_samples, side="tuned", expected_count=len(trusted))
    bound_hashes = _artifact_hashes(artifact_hashes)

    base_rows: list[dict[str, object]] = []
    tuned_rows: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    identity_rows: list[dict[str, object]] = []
    for trusted_row in trusted:
        index = int(trusted_row["sample_index"])
        base = _validated_sample(base_by_index[index], trusted_row, side="base")
        tuned = _validated_sample(tuned_by_index[index], trusted_row, side="tuned")
        if base["generation_cap"] != tuned["generation_cap"]:
            raise ValueError(f"Base and tuned sample {index} use different generation caps.")
        base_rows.append(base)
        tuned_rows.append(tuned)
        transitions.append(
            {
                "sample_index": index,
                "template_id": trusted_row["template_id"],
                "problem_family": trusted_row["evidence_family"],
                "base_correct": base["correct"],
                "tuned_correct": tuned["correct"],
                "base_prediction_sha256": base["prediction_sha256"],
                "tuned_prediction_sha256": tuned["prediction_sha256"],
            }
        )
        identity_rows.append(
            {
                "sample_index": index,
                "user": trusted_row["user"],
                "reference": trusted_row["reference"],
                "source": trusted_row["source"],
                "metadata": trusted_row["metadata"],
                **{key: trusted_row[key] for key in _IDENTITY_FIELDS},
            }
        )

    transition_summary = _transition_summary(transitions)
    family_records: dict[str, list[Mapping[str, object]]] = {}
    for row in transitions:
        family_records.setdefault(str(row["problem_family"]), []).append(row)
    per_family = {
        family: _transition_summary(rows)
        for family, rows in sorted(family_records.items())
    }
    template_ids = sorted({str(row["template_id"]) for row in transitions})
    base_metrics = _side_metrics(base_rows)
    tuned_metrics = _side_metrics(tuned_rows)
    aggregate_delta = {
        key: float(tuned_metrics[key]) - float(base_metrics[key])
        for key in sorted(set(base_metrics) & set(tuned_metrics))
    }
    return {
        "schema": PAIRED_EVIDENCE_SCHEMA_VERSION,
        "artifact_hashes": bound_hashes,
        "identity": {
            "eval_sample_count": len(trusted),
            "eval_identity_sha256": _sha256_json(identity_rows),
            "transition_records_sha256": _sha256_json(transitions),
            "template_cluster_count": len(template_ids),
            "template_ids_sha256": _sha256_json(template_ids),
        },
        "transitions": transition_summary,
        "per_family": per_family,
        "mcnemar_exact_one_sided": _exact_one_sided_mcnemar(
            wins=int(transition_summary["wins"]),
            regressions=int(transition_summary["regressions"]),
        ),
        "template_cluster_bootstrap": _template_cluster_bootstrap(
            transitions,
            seed=bootstrap_seed,
            resamples=bootstrap_resamples,
        ),
        "recomputed_metrics": {
            "base": base_metrics,
            "tuned": tuned_metrics,
            "delta_tuned_minus_base": aggregate_delta,
        },
    }


def validate_reported_metrics(
    reported: Mapping[str, object],
    recomputed: Mapping[str, object],
    *,
    side: str,
) -> None:
    """Fail when evaluator aggregates disagree with detailed sample evidence."""

    if not isinstance(reported, Mapping) or not isinstance(recomputed, Mapping):
        raise ValueError(f"{side} metrics must be mappings.")
    keys = [
        *_CORE_METRIC_KEYS,
        *sorted(key for key in recomputed if str(key).startswith("verified_")),
    ]
    for key in dict.fromkeys(keys):
        if key not in reported or key not in recomputed:
            raise ValueError(f"{side} metrics are missing paired-evidence field {key}.")
        actual = _finite_float(reported[key], label=f"{side}.{key}")
        expected = _finite_float(recomputed[key], label=f"recomputed {side}.{key}")
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"{side} aggregate metric {key} does not match detailed sample evidence."
            )


def recompute_and_validate_paired_evidence(
    expected_evidence: Mapping[str, object],
    base_samples: Sequence[Mapping[str, object]],
    tuned_samples: Sequence[Mapping[str, object]],
    eval_rows: Sequence[object],
    *,
    artifact_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Recompute evidence using trusted eval mappings and require exact equality.

    This is the gate-facing validation API.  ``eval_rows`` may be the mappings
    loaded directly from ``eval_pairs.jsonl``; importing the training pipeline
    or constructing ``ChatPair`` instances is unnecessary.
    """

    if not isinstance(expected_evidence, Mapping):
        raise ValueError("Expected paired evidence must be a mapping.")
    if expected_evidence.get("schema") != PAIRED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Expected paired evidence uses an unsupported schema.")
    bootstrap = expected_evidence.get("template_cluster_bootstrap")
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Expected paired evidence is missing bootstrap configuration.")
    seed = bootstrap.get("seed")
    resamples = bootstrap.get("resamples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Expected paired evidence bootstrap seed must be an integer.")
    if isinstance(resamples, bool) or not isinstance(resamples, int):
        raise ValueError("Expected paired evidence bootstrap resamples must be an integer.")
    recomputed = build_paired_evidence(
        base_samples,
        tuned_samples,
        eval_rows,
        artifact_hashes=artifact_hashes,
        bootstrap_seed=seed,
        bootstrap_resamples=resamples,
    )
    if _canonical_json_bytes(dict(expected_evidence)) != _canonical_json_bytes(recomputed):
        raise ValueError("Paired evidence does not match recomputed trusted sample evidence.")
    return recomputed


__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "MAX_BOOTSTRAP_RESAMPLES",
    "MIN_BOOTSTRAP_RESAMPLES",
    "PAIRED_EVIDENCE_SCHEMA_VERSION",
    "build_paired_evidence",
    "deterministic_eval_selection",
    "derive_sample_comparison",
    "paired_evidence_sha256",
    "recompute_and_validate_paired_evidence",
    "validate_sample_comparison",
    "validate_reported_metrics",
]
