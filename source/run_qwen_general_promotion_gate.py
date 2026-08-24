"""Fail-closed promotion gate for a trained Qwen adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

try:
    from qwen_adapter_promotion import (
        BENCHMARK_SCHEMA_VERSION,
        GATE_FILENAME,
        GATE_SCHEMA_VERSION,
        PRODUCTION_POLICY_ID,
        PRODUCTION_PROTOCOL,
        PRODUCTION_THRESHOLD_FLOORS,
        PROMOTION_FILENAME,
        PROMOTION_SCHEMA_VERSION,
        evaluation_code_hashes,
        sha256_file,
    )
    from qwen_paired_evidence import (
        PAIRED_EVIDENCE_SCHEMA_VERSION,
        deterministic_eval_selection,
        paired_evidence_sha256,
        recompute_and_validate_paired_evidence,
        validate_sample_comparison,
        validate_reported_metrics,
    )
    from verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate
except ImportError:  # pragma: no cover - package import path
    from .qwen_adapter_promotion import (
        BENCHMARK_SCHEMA_VERSION,
        GATE_FILENAME,
        GATE_SCHEMA_VERSION,
        PRODUCTION_POLICY_ID,
        PRODUCTION_PROTOCOL,
        PRODUCTION_THRESHOLD_FLOORS,
        PROMOTION_FILENAME,
        PROMOTION_SCHEMA_VERSION,
        evaluation_code_hashes,
        sha256_file,
    )
    from .qwen_paired_evidence import (
        PAIRED_EVIDENCE_SCHEMA_VERSION,
        deterministic_eval_selection,
        paired_evidence_sha256,
        recompute_and_validate_paired_evidence,
        validate_sample_comparison,
        validate_reported_metrics,
    )
    from .verifiable_reasoning import VERIFIER_SCHEMA_VERSION, verify_candidate

_MINIMUM_THRESHOLD_KEYS = frozenset(
    {
        "min_verified_samples",
        "min_verified_gain",
        "min_tuned_accuracy",
        "min_token_f1_delta",
        "min_family_verified_samples",
        "min_paired_cluster_lower_bound",
        "min_template_clusters",
    }
)


def _metric(payload: Mapping[str, object], key: str) -> Optional[float]:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _count_metric(payload: Mapping[str, object], key: str) -> Optional[int]:
    value = _metric(payload, key)
    if value is None or value < 0 or not value.is_integer():
        return None
    return int(value)


def _rate_metric(payload: Mapping[str, object], key: str) -> Optional[float]:
    value = _metric(payload, key)
    if value is None or value < 0.0 or value > 1.0:
        return None
    return value


def _finite_number(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _positive_count(value: object) -> Optional[int]:
    parsed = _finite_number(value)
    if parsed is None or parsed < 1.0 or not parsed.is_integer():
        return None
    return int(parsed)


def _threshold_policy_violations(thresholds: Mapping[str, object]) -> list[str]:
    violations: list[str] = []
    for key, floor in PRODUCTION_THRESHOLD_FLOORS.items():
        value = _finite_number(thresholds.get(key))
        if value is None:
            violations.append(key)
            continue
        if key in _MINIMUM_THRESHOLD_KEYS:
            if value < float(floor):
                violations.append(key)
        elif value > float(floor):
            violations.append(key)
    return violations


def _protocol_policy_violations(
    benchmark: Mapping[str, object],
    curriculum: Mapping[str, object],
    paired_evidence: Mapping[str, object],
) -> list[str]:
    config = benchmark.get("config")
    if not isinstance(config, Mapping):
        return list(PRODUCTION_PROTOCOL)
    actual: dict[str, object] = {
        "curriculum_schema": curriculum.get("curriculum_schema"),
        "curriculum_seed": curriculum.get("curriculum_seed"),
        "curriculum_eval_rows": curriculum.get("curriculum_eval_samples"),
        "curriculum_eval_sha256": curriculum.get("curriculum_eval_sha256"),
        "selection_seed": config.get("seed"),
        "samples_per_family": config.get("samples_per_family"),
        "max_eval_samples": config.get("max_eval_samples"),
        "selected_eval_samples": config.get("eval_samples"),
        "max_length": config.get("max_length"),
        "max_new_tokens": config.get("max_new_tokens"),
        "paired_bootstrap_seed": config.get("paired_bootstrap_seed"),
        "paired_bootstrap_resamples": config.get("paired_bootstrap_resamples"),
    }
    bootstrap = paired_evidence.get("template_cluster_bootstrap")
    if isinstance(bootstrap, Mapping):
        if actual["paired_bootstrap_seed"] != bootstrap.get("seed"):
            actual["paired_bootstrap_seed"] = None
        if actual["paired_bootstrap_resamples"] != bootstrap.get("resamples"):
            actual["paired_bootstrap_resamples"] = None
    else:
        actual["paired_bootstrap_seed"] = None
        actual["paired_bootstrap_resamples"] = None
    violations: list[str] = []
    for key, expected in PRODUCTION_PROTOCOL.items():
        value = actual.get(key)
        if isinstance(expected, int):
            matches = not isinstance(value, bool) and isinstance(value, int) and value == expected
        else:
            matches = isinstance(value, str) and value == expected
        if not matches:
            violations.append(key)
    return violations


def _realized_successes(rate: Optional[float], count: Optional[int]) -> Optional[int]:
    if rate is None or count is None or count < 0:
        return None
    raw_successes = rate * count
    rounded = round(raw_successes)
    if not math.isclose(raw_successes, rounded, rel_tol=0.0, abs_tol=1e-9):
        return None
    return int(rounded)


def _family_metrics(
    payload: Mapping[str, object],
    prefix: str,
    *,
    counts: bool,
) -> tuple[dict[str, float | int], bool]:
    rows: dict[str, float | int] = {}
    invalid = False
    for raw_key in sorted(payload):
        key = str(raw_key)
        if not key.startswith(prefix):
            continue
        family = key[len(prefix) :]
        value: float | int | None
        value = _count_metric(payload, key) if counts else _rate_metric(payload, key)
        if not family or value is None:
            invalid = True
            continue
        rows[family] = value
    return rows, invalid


def _metric_payload_blockers(payload: Mapping[str, object], side: str) -> list[str]:
    blockers: list[str] = []
    for raw_key, raw_value in payload.items():
        key = str(raw_key)
        value = _finite_number(raw_value)
        if not key or value is None:
            blockers.append(f"invalid_metric:{side}:{key or 'unnamed'}")
            continue
        if value < 0.0:
            blockers.append(f"out_of_range_metric:{side}:{key}")
            continue
        if (
            key in {"verified_accuracy", "token_f1", "char_similarity", "generation_cap_rate"}
            or key.startswith("verified_accuracy_family_")
        ) and value > 1.0:
            blockers.append(f"out_of_range_metric:{side}:{key}")
        if (
            key in {"eval_samples", "verified_samples", "generation_cap", "generation_cap_hits"}
            or key.startswith("verified_samples_family_")
        ) and not value.is_integer():
            blockers.append(f"non_integral_count_metric:{side}:{key}")
    return blockers


def evaluate_promotion(
    benchmark: Mapping[str, object],
    *,
    min_verified_samples: int = 20,
    min_verified_gain: float = 0.05,
    min_tuned_accuracy: float = 0.20,
    max_family_regression: float = 0.0,
    max_loss_ratio: float = 1.05,
    min_token_f1_delta: float = -0.02,
    min_family_verified_samples: int = 1,
    max_generation_cap_rate: float = 0.05,
    max_paired_p_value: float = 0.05,
    max_paired_regression_rate: float = 0.02,
    min_paired_cluster_lower_bound: float = 0.0,
    min_template_clusters: int = 5,
    expected_family_counts: Optional[Mapping[str, int]] = None,
    expected_eval_samples: Optional[int] = None,
) -> dict[str, object]:
    reference = benchmark.get("base")
    tuned = benchmark.get("tuned")
    if not isinstance(reference, Mapping) or not isinstance(tuned, Mapping):
        return {"passed": False, "blockers": ["missing_reference_or_tuned_metrics"]}

    min_samples_value = _positive_count(min_verified_samples)
    min_family_samples_value = _positive_count(min_family_verified_samples)
    min_gain_value = _finite_number(min_verified_gain)
    min_accuracy_value = _finite_number(min_tuned_accuracy)
    max_family_regression_value = _finite_number(max_family_regression)
    max_loss_ratio_value = _finite_number(max_loss_ratio)
    min_f1_delta_value = _finite_number(min_token_f1_delta)
    max_cap_rate_value = _finite_number(max_generation_cap_rate)
    max_paired_p_value_value = _finite_number(max_paired_p_value)
    max_paired_regression_rate_value = _finite_number(max_paired_regression_rate)
    min_paired_cluster_lower_bound_value = _finite_number(min_paired_cluster_lower_bound)
    min_template_clusters_value = _positive_count(min_template_clusters)
    thresholds = {
        "min_verified_samples": min_samples_value,
        "min_verified_gain": min_gain_value,
        "min_tuned_accuracy": min_accuracy_value,
        "max_family_regression": max_family_regression_value,
        "max_loss_ratio": max_loss_ratio_value,
        "min_token_f1_delta": min_f1_delta_value,
        "min_family_verified_samples": min_family_samples_value,
        "max_generation_cap_rate": max_cap_rate_value,
        "max_paired_p_value": max_paired_p_value_value,
        "max_paired_regression_rate": max_paired_regression_rate_value,
        "min_paired_cluster_lower_bound": min_paired_cluster_lower_bound_value,
        "min_template_clusters": min_template_clusters_value,
    }
    thresholds_valid = bool(
        min_samples_value is not None
        and min_family_samples_value is not None
        and min_gain_value is not None
        and 0.0 <= min_gain_value <= 1.0
        and min_accuracy_value is not None
        and 0.0 <= min_accuracy_value <= 1.0
        and max_family_regression_value is not None
        and 0.0 <= max_family_regression_value <= 1.0
        and max_loss_ratio_value is not None
        and max_loss_ratio_value > 0.0
        and min_f1_delta_value is not None
        and -1.0 <= min_f1_delta_value <= 1.0
        and max_cap_rate_value is not None
        and 0.0 <= max_cap_rate_value <= 1.0
        and max_paired_p_value_value is not None
        and 0.0 <= max_paired_p_value_value <= 1.0
        and max_paired_regression_rate_value is not None
        and 0.0 <= max_paired_regression_rate_value <= 1.0
        and min_paired_cluster_lower_bound_value is not None
        and -1.0 <= min_paired_cluster_lower_bound_value <= 1.0
        and min_template_clusters_value is not None
    )
    if not thresholds_valid:
        return {
            "passed": False,
            "blockers": ["invalid_thresholds"],
            "metrics": {},
            "thresholds": thresholds,
        }

    blockers: list[str] = []
    blockers.extend(_metric_payload_blockers(reference, "base"))
    blockers.extend(_metric_payload_blockers(tuned, "tuned"))
    ref_eval_samples = _count_metric(reference, "eval_samples")
    tuned_eval_samples = _count_metric(tuned, "eval_samples")
    ref_samples = _count_metric(reference, "verified_samples")
    tuned_samples = _count_metric(tuned, "verified_samples")
    if None in (ref_eval_samples, tuned_eval_samples, ref_samples, tuned_samples):
        blockers.append("missing_or_invalid_sample_counts")
    else:
        assert ref_eval_samples is not None and tuned_eval_samples is not None
        assert ref_samples is not None and tuned_samples is not None
        if ref_eval_samples != tuned_eval_samples or ref_samples != tuned_samples:
            blockers.append("base_tuned_sample_count_mismatch")
        if expected_eval_samples is not None:
            expected_eval_count = _positive_count(expected_eval_samples)
            if expected_eval_count is None:
                blockers.append("invalid_expected_eval_samples")
            elif ref_eval_samples != expected_eval_count or tuned_eval_samples != expected_eval_count:
                blockers.append("selected_eval_sample_count_mismatch")
        if ref_samples > ref_eval_samples or tuned_samples > tuned_eval_samples:
            blockers.append("verified_samples_exceed_eval_samples")
        if tuned_samples < min_samples_value:
            blockers.append("insufficient_verified_samples")

    ref_accuracy = _rate_metric(reference, "verified_accuracy")
    tuned_accuracy = _rate_metric(tuned, "verified_accuracy")
    ref_loss = _metric(reference, "eval_loss")
    tuned_loss = _metric(tuned, "eval_loss")
    ref_f1 = _rate_metric(reference, "token_f1")
    tuned_f1 = _rate_metric(tuned, "token_f1")
    ref_cap = _positive_count(reference.get("generation_cap"))
    tuned_cap = _positive_count(tuned.get("generation_cap"))
    ref_cap_hits = _count_metric(reference, "generation_cap_hits")
    tuned_cap_hits = _count_metric(tuned, "generation_cap_hits")
    ref_cap_rate = _rate_metric(reference, "generation_cap_rate")
    tuned_cap_rate = _rate_metric(tuned, "generation_cap_rate")
    ref_verified_correct = _realized_successes(ref_accuracy, ref_samples)
    tuned_verified_correct = _realized_successes(tuned_accuracy, tuned_samples)
    if ref_accuracy is None or tuned_accuracy is None:
        blockers.append("missing_verified_accuracy")
        verified_gain = None
    else:
        verified_gain = tuned_accuracy - ref_accuracy
        if verified_gain < min_gain_value:
            blockers.append("verified_accuracy_gain_below_threshold")
        if tuned_accuracy < min_accuracy_value:
            blockers.append("tuned_accuracy_below_floor")
        if ref_verified_correct is None:
            blockers.append("reference_verified_accuracy_not_realizable")
        if tuned_verified_correct is None:
            blockers.append("tuned_verified_accuracy_not_realizable")
    if ref_loss is None or tuned_loss is None or ref_loss <= 0 or tuned_loss <= 0:
        blockers.append("missing_or_invalid_eval_loss")
        loss_ratio = None
    else:
        loss_ratio = tuned_loss / ref_loss
        if loss_ratio > max_loss_ratio_value:
            blockers.append("eval_loss_regression")
    if ref_f1 is None or tuned_f1 is None:
        blockers.append("missing_token_f1")
        token_f1_delta = None
    else:
        token_f1_delta = tuned_f1 - ref_f1
        if token_f1_delta < min_f1_delta_value:
            blockers.append("token_f1_regression")

    if (
        ref_cap is None
        or tuned_cap is None
        or ref_cap_hits is None
        or tuned_cap_hits is None
        or ref_cap_rate is None
        or tuned_cap_rate is None
    ):
        blockers.append("missing_or_invalid_generation_cap_metrics")
    else:
        if ref_cap != tuned_cap:
            blockers.append("base_tuned_generation_cap_mismatch")
        if ref_eval_samples is None or tuned_eval_samples is None:
            blockers.append("generation_cap_rate_missing_denominator")
        else:
            if ref_cap_hits > ref_eval_samples or tuned_cap_hits > tuned_eval_samples:
                blockers.append("generation_cap_hits_exceed_eval_samples")
            expected_ref_rate = ref_cap_hits / ref_eval_samples if ref_eval_samples else 0.0
            expected_tuned_rate = tuned_cap_hits / tuned_eval_samples if tuned_eval_samples else 0.0
            if not math.isclose(ref_cap_rate, expected_ref_rate, rel_tol=0.0, abs_tol=1e-12):
                blockers.append("reference_generation_cap_rate_inconsistent")
            if not math.isclose(tuned_cap_rate, expected_tuned_rate, rel_tol=0.0, abs_tol=1e-12):
                blockers.append("tuned_generation_cap_rate_inconsistent")
        if ref_cap_rate > max_cap_rate_value:
            blockers.append("reference_generation_cap_rate_above_threshold")
        if tuned_cap_rate > max_cap_rate_value:
            blockers.append("tuned_generation_cap_rate_above_threshold")

    family_deltas: dict[str, float] = {}
    accuracy_prefix = "verified_accuracy_family_"
    count_prefix = "verified_samples_family_"
    ref_family_accuracy, ref_accuracy_invalid = _family_metrics(reference, accuracy_prefix, counts=False)
    tuned_family_accuracy, tuned_accuracy_invalid = _family_metrics(tuned, accuracy_prefix, counts=False)
    ref_family_counts, ref_counts_invalid = _family_metrics(reference, count_prefix, counts=True)
    tuned_family_counts, tuned_counts_invalid = _family_metrics(tuned, count_prefix, counts=True)
    if any((ref_accuracy_invalid, tuned_accuracy_invalid, ref_counts_invalid, tuned_counts_invalid)):
        blockers.append("invalid_family_metrics")
    family_sets = (
        set(ref_family_accuracy),
        set(tuned_family_accuracy),
        set(ref_family_counts),
        set(tuned_family_counts),
    )
    family_metric_sets_aligned = bool(
        family_sets[0] and all(families == family_sets[0] for families in family_sets[1:])
    )
    if not family_metric_sets_aligned:
        blockers.append("family_metric_set_mismatch")
    else:
        ref_family_correct_total = 0
        tuned_family_correct_total = 0
        family_correct_counts_valid = True
        for family in sorted(family_sets[0]):
            ref_count = int(ref_family_counts[family])
            tuned_count = int(tuned_family_counts[family])
            ref_correct = _realized_successes(float(ref_family_accuracy[family]), ref_count)
            tuned_correct = _realized_successes(float(tuned_family_accuracy[family]), tuned_count)
            if ref_correct is None:
                blockers.append(f"reference_family_accuracy_not_realizable:{family}")
                family_correct_counts_valid = False
            else:
                ref_family_correct_total += ref_correct
            if tuned_correct is None:
                blockers.append(f"tuned_family_accuracy_not_realizable:{family}")
                family_correct_counts_valid = False
            else:
                tuned_family_correct_total += tuned_correct
            if ref_count != tuned_count:
                blockers.append(f"family_sample_count_mismatch:{family}")
                continue
            if tuned_count < min_family_samples_value:
                blockers.append(f"insufficient_family_samples:{family}")
            delta = float(tuned_family_accuracy[family]) - float(ref_family_accuracy[family])
            family_deltas[family] = delta
            if delta < -max_family_regression_value:
                blockers.append(f"family_regression:{family}")
        if ref_samples is not None and sum(int(value) for value in ref_family_counts.values()) != ref_samples:
            blockers.append("reference_family_counts_do_not_cover_verified_samples")
        if tuned_samples is not None and sum(int(value) for value in tuned_family_counts.values()) != tuned_samples:
            blockers.append("tuned_family_counts_do_not_cover_verified_samples")
        ref_family_total = sum(int(value) for value in ref_family_counts.values())
        tuned_family_total = sum(int(value) for value in tuned_family_counts.values())
        if ref_accuracy is not None and ref_family_total > 0:
            weighted_ref_accuracy = sum(
                float(ref_family_accuracy[family]) * int(ref_family_counts[family])
                for family in family_sets[0]
            ) / ref_family_total
            if not math.isclose(ref_accuracy, weighted_ref_accuracy, rel_tol=0.0, abs_tol=1e-12):
                blockers.append("reference_verified_accuracy_inconsistent_with_families")
        if tuned_accuracy is not None and tuned_family_total > 0:
            weighted_tuned_accuracy = sum(
                float(tuned_family_accuracy[family]) * int(tuned_family_counts[family])
                for family in family_sets[0]
            ) / tuned_family_total
            if not math.isclose(tuned_accuracy, weighted_tuned_accuracy, rel_tol=0.0, abs_tol=1e-12):
                blockers.append("tuned_verified_accuracy_inconsistent_with_families")
        if family_correct_counts_valid:
            if (
                ref_verified_correct is not None
                and ref_family_correct_total != ref_verified_correct
            ):
                blockers.append("reference_family_correct_total_mismatch")
            if (
                tuned_verified_correct is not None
                and tuned_family_correct_total != tuned_verified_correct
            ):
                blockers.append("tuned_family_correct_total_mismatch")
        if expected_family_counts is not None:
            normalized_expected: dict[str, int] = {}
            expected_invalid = False
            for raw_family, raw_count in expected_family_counts.items():
                family = str(raw_family or "").strip()
                count = _positive_count(raw_count)
                if not family or count is None:
                    expected_invalid = True
                    continue
                normalized_expected[family] = count
            if expected_invalid or not normalized_expected:
                blockers.append("invalid_expected_family_counts")
            elif set(normalized_expected) != family_sets[0]:
                blockers.append("selected_eval_family_set_mismatch")
            else:
                for family, expected_count in sorted(normalized_expected.items()):
                    if int(ref_family_counts[family]) != expected_count:
                        blockers.append(f"selected_eval_family_count_mismatch:{family}")

    paired_p_value: Optional[float] = None
    paired_cluster_lower: Optional[float] = None
    paired_cluster_upper: Optional[float] = None
    paired_regression_rate: Optional[float] = None
    paired_wins: Optional[int] = None
    paired_regressions: Optional[int] = None
    paired_clusters: Optional[int] = None
    paired = benchmark.get("paired_evidence")
    if not isinstance(paired, Mapping) or paired.get("schema") != PAIRED_EVIDENCE_SCHEMA_VERSION:
        blockers.append("missing_or_invalid_paired_evidence")
    else:
        transitions = paired.get("transitions")
        mcnemar = paired.get("mcnemar_exact_one_sided")
        cluster_bootstrap = paired.get("template_cluster_bootstrap")
        identity = paired.get("identity")
        per_family = paired.get("per_family")
        if not all(
            isinstance(value, Mapping)
            for value in (transitions, mcnemar, cluster_bootstrap, identity, per_family)
        ):
            blockers.append("invalid_paired_evidence_structure")
        else:
            assert isinstance(transitions, Mapping)
            assert isinstance(mcnemar, Mapping)
            assert isinstance(cluster_bootstrap, Mapping)
            assert isinstance(identity, Mapping)
            assert isinstance(per_family, Mapping)
            paired_samples = _count_metric(transitions, "samples")
            paired_base_correct = _count_metric(transitions, "base_correct")
            paired_tuned_correct = _count_metric(transitions, "tuned_correct")
            paired_wins = _count_metric(transitions, "wins")
            paired_regressions = _count_metric(transitions, "regressions")
            paired_discordant = _count_metric(transitions, "discordant_pairs")
            paired_base_accuracy = _rate_metric(transitions, "base_accuracy")
            paired_tuned_accuracy = _rate_metric(transitions, "tuned_accuracy")
            paired_gain = _finite_number(transitions.get("accuracy_delta"))
            paired_p_value = _rate_metric(mcnemar, "p_value")
            paired_cluster_lower = _finite_number(cluster_bootstrap.get("lower_95"))
            paired_cluster_upper = _finite_number(cluster_bootstrap.get("upper_95"))
            paired_clusters = _count_metric(identity, "template_cluster_count")
            required_values = (
                paired_samples,
                paired_base_correct,
                paired_tuned_correct,
                paired_wins,
                paired_regressions,
                paired_discordant,
                paired_base_accuracy,
                paired_tuned_accuracy,
                paired_gain,
                paired_p_value,
                paired_cluster_lower,
                paired_cluster_upper,
                paired_clusters,
            )
            if any(value is None for value in required_values):
                blockers.append("invalid_paired_evidence_metrics")
            else:
                assert paired_samples is not None
                assert paired_base_correct is not None
                assert paired_tuned_correct is not None
                assert paired_wins is not None
                assert paired_regressions is not None
                assert paired_discordant is not None
                assert paired_base_accuracy is not None
                assert paired_tuned_accuracy is not None
                assert paired_gain is not None
                assert paired_p_value is not None
                assert paired_cluster_lower is not None
                assert paired_cluster_upper is not None
                assert paired_clusters is not None
                if paired_samples != ref_samples or paired_samples != tuned_samples:
                    blockers.append("paired_sample_count_mismatch")
                if paired_discordant != paired_wins + paired_regressions:
                    blockers.append("paired_discordant_count_mismatch")
                if paired_base_correct + paired_wins - paired_regressions != paired_tuned_correct:
                    blockers.append("paired_transition_count_mismatch")
                expected_base_accuracy = paired_base_correct / max(1, paired_samples)
                expected_tuned_accuracy = paired_tuned_correct / max(1, paired_samples)
                expected_gain = (paired_wins - paired_regressions) / max(1, paired_samples)
                if not math.isclose(paired_base_accuracy, expected_base_accuracy, rel_tol=0.0, abs_tol=1e-12):
                    blockers.append("paired_base_accuracy_inconsistent")
                if not math.isclose(paired_tuned_accuracy, expected_tuned_accuracy, rel_tol=0.0, abs_tol=1e-12):
                    blockers.append("paired_tuned_accuracy_inconsistent")
                if not math.isclose(paired_gain, expected_gain, rel_tol=0.0, abs_tol=1e-12):
                    blockers.append("paired_gain_inconsistent")
                if ref_accuracy is not None and not math.isclose(
                    paired_base_accuracy, ref_accuracy, rel_tol=0.0, abs_tol=1e-12
                ):
                    blockers.append("paired_base_accuracy_mismatch")
                if tuned_accuracy is not None and not math.isclose(
                    paired_tuned_accuracy, tuned_accuracy, rel_tol=0.0, abs_tol=1e-12
                ):
                    blockers.append("paired_tuned_accuracy_mismatch")
                if verified_gain is not None and not math.isclose(
                    paired_gain, verified_gain, rel_tol=0.0, abs_tol=1e-12
                ):
                    blockers.append("paired_gain_mismatch")
                paired_regression_rate = paired_regressions / max(1, paired_samples)
                if paired_p_value > max_paired_p_value_value:
                    blockers.append("paired_exact_p_value_above_threshold")
                if paired_regression_rate > max_paired_regression_rate_value:
                    blockers.append("paired_regression_rate_above_threshold")
                if paired_cluster_lower <= min_paired_cluster_lower_bound_value:
                    blockers.append("paired_cluster_lower_bound_not_above_threshold")
                if paired_cluster_upper < paired_cluster_lower:
                    blockers.append("paired_cluster_interval_invalid")
                if paired_clusters < min_template_clusters_value:
                    blockers.append("insufficient_template_clusters")

            normalized_paired_families: dict[str, Mapping[str, object]] = {}
            paired_family_collision = False
            for raw_family, raw_metrics in per_family.items():
                metric_family = _metric_family_name(raw_family)
                if not isinstance(raw_metrics, Mapping) or metric_family in normalized_paired_families:
                    paired_family_collision = True
                    continue
                normalized_paired_families[metric_family] = raw_metrics
            if (
                paired_family_collision
                or not family_metric_sets_aligned
                or set(normalized_paired_families) != family_sets[0]
            ):
                blockers.append("paired_family_set_mismatch")
            else:
                for family, family_metrics in sorted(normalized_paired_families.items()):
                    family_samples = _count_metric(family_metrics, "samples")
                    family_base_accuracy = _rate_metric(family_metrics, "base_accuracy")
                    family_tuned_accuracy = _rate_metric(family_metrics, "tuned_accuracy")
                    family_gain = _finite_number(family_metrics.get("accuracy_delta"))
                    if None in (family_samples, family_base_accuracy, family_tuned_accuracy, family_gain):
                        blockers.append(f"invalid_paired_family_metrics:{family}")
                        continue
                    assert family_samples is not None
                    assert family_base_accuracy is not None
                    assert family_tuned_accuracy is not None
                    assert family_gain is not None
                    if family_samples != int(ref_family_counts[family]):
                        blockers.append(f"paired_family_sample_count_mismatch:{family}")
                    if not math.isclose(
                        family_base_accuracy,
                        float(ref_family_accuracy[family]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        blockers.append(f"paired_reference_family_accuracy_mismatch:{family}")
                    if not math.isclose(
                        family_tuned_accuracy,
                        float(tuned_family_accuracy[family]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        blockers.append(f"paired_tuned_family_accuracy_mismatch:{family}")
                    if not math.isclose(
                        family_gain,
                        family_deltas[family],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        blockers.append(f"paired_family_gain_mismatch:{family}")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "metrics": {
            "reference_eval_samples": ref_eval_samples,
            "tuned_eval_samples": tuned_eval_samples,
            "reference_verified_samples": ref_samples,
            "tuned_verified_samples": tuned_samples,
            "reference_verified_accuracy": ref_accuracy,
            "tuned_verified_accuracy": tuned_accuracy,
            "verified_accuracy_gain": verified_gain,
            "eval_loss_ratio": loss_ratio,
            "token_f1_delta": token_f1_delta,
            "reference_generation_cap": ref_cap,
            "tuned_generation_cap": tuned_cap,
            "reference_generation_cap_hits": ref_cap_hits,
            "tuned_generation_cap_hits": tuned_cap_hits,
            "reference_generation_cap_rate": ref_cap_rate,
            "tuned_generation_cap_rate": tuned_cap_rate,
            "family_accuracy_deltas": family_deltas,
            "paired_wins": paired_wins,
            "paired_regressions": paired_regressions,
            "paired_regression_rate": paired_regression_rate,
            "paired_exact_one_sided_p_value": paired_p_value,
            "paired_cluster_lower_95": paired_cluster_lower,
            "paired_cluster_upper_95": paired_cluster_upper,
            "paired_template_clusters": paired_clusters,
        },
        "thresholds": thresholds,
    }


def _metric_family_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_") or "unknown"


def _safe_relative_artifact(parent: Path, raw_name: object) -> Path:
    name = str(raw_name or "").strip()
    candidate_name = Path(name)
    if not name or candidate_name.is_absolute():
        raise ValueError("Artifact path must be a non-empty relative path.")
    candidate = (parent.resolve() / candidate_name).resolve()
    try:
        candidate.relative_to(parent.resolve())
    except ValueError as exc:
        raise ValueError("Artifact path escapes its manifest directory.") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"Artifact not found: {candidate}")
    return candidate


def _curriculum_evidence(manifest_path: Path) -> dict[str, object]:
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Curriculum manifest must be valid UTF-8 JSON.") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Curriculum manifest must be a JSON object.")
    if manifest.get("all_targets_verified") is not True:
        raise ValueError("Curriculum manifest does not attest that every target was verified.")
    if str(manifest.get("verifier_schema") or "") != VERIFIER_SCHEMA_VERSION:
        raise ValueError("Curriculum verifier schema does not match the current verifier.")
    curriculum_schema = str(manifest.get("curriculum_schema") or "").strip()
    if not curriculum_schema:
        raise ValueError("Curriculum manifest is missing its schema identity.")
    curriculum_seed = manifest.get("seed")
    if isinstance(curriculum_seed, bool) or not isinstance(curriculum_seed, int):
        raise ValueError("Curriculum manifest seed must be an integer.")
    if manifest.get("template_ids_disjoint") is not True:
        raise ValueError("Curriculum manifest does not attest disjoint template identities.")
    if manifest.get("prompt_text_disjoint") is not True:
        raise ValueError("Curriculum manifest does not attest disjoint prompt text.")
    eval_metadata = manifest.get("eval")
    if not isinstance(eval_metadata, Mapping):
        raise ValueError("Curriculum manifest is missing eval artifact metadata.")
    eval_path = _safe_relative_artifact(manifest_path.parent, eval_metadata.get("file"))
    actual_eval_sha = sha256_file(eval_path)
    if str(eval_metadata.get("sha256") or "").strip().lower() != actual_eval_sha:
        raise ValueError("Curriculum eval artifact hash does not match its manifest.")
    raw_rows = _load_jsonl_mappings(eval_path, label="Curriculum eval artifact")
    normalized_rows = deterministic_eval_selection(
        raw_rows,
        seed=0,
        samples_per_family=0,
        max_eval_samples=0,
    )
    declared_rows = eval_metadata.get("rows")
    if (
        isinstance(declared_rows, bool)
        or not isinstance(declared_rows, int)
        or declared_rows != len(normalized_rows)
    ):
        raise ValueError("Curriculum eval row count does not match its manifest.")
    family_counts: dict[str, int] = {}
    for row in normalized_rows:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):  # pragma: no cover - validated above
            raise ValueError("Curriculum eval row is missing normalized metadata.")
        verification = verify_candidate(row.get("user"), row.get("assistant"), metadata)
        if not verification.valid_spec or not verification.passed:
            raise ValueError(
                "Curriculum eval reference failed current verifier replay: "
                f"{verification.reason}."
            )
        family = _metric_family_name(metadata.get("problem_family"))
        family_counts[family] = family_counts.get(family, 0) + 1
    declared_family_counts = eval_metadata.get("family_counts")
    if not isinstance(declared_family_counts, Mapping):
        raise ValueError("Curriculum eval manifest is missing family counts.")
    parsed_family_counts: dict[str, int] = {}
    for raw_family, raw_count in declared_family_counts.items():
        family = _metric_family_name(raw_family)
        if (
            family in parsed_family_counts
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 1
        ):
            raise ValueError("Curriculum eval manifest has invalid family counts.")
        parsed_family_counts[family] = raw_count
    if dict(sorted(parsed_family_counts.items())) != dict(sorted(family_counts.items())):
        raise ValueError("Curriculum eval family counts do not match its manifest.")
    return {
        "curriculum_manifest": str(manifest_path.resolve()),
        "curriculum_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "curriculum_eval": str(eval_path),
        "curriculum_eval_sha256": actual_eval_sha,
        "curriculum_schema": curriculum_schema,
        "curriculum_seed": curriculum_seed,
        "curriculum_eval_samples": len(normalized_rows),
        "curriculum_eval_family_counts": dict(sorted(family_counts.items())),
        "curriculum_eval_rows": normalized_rows,
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
    }


def _selected_eval_evidence(
    benchmark_path: Path,
    benchmark: Mapping[str, object],
    curriculum_evidence: Mapping[str, object],
) -> dict[str, object]:
    selected_eval_path = benchmark_path.parent / "eval_pairs.jsonl"
    if not selected_eval_path.is_file():
        raise FileNotFoundError(f"Selected eval artifact not found: {selected_eval_path}")
    selected_eval_bytes = selected_eval_path.read_bytes()
    try:
        selected_eval_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Selected eval artifact must be UTF-8 JSONL.") from exc
    selected_rows = _load_jsonl_mappings(selected_eval_path, label="Selected eval artifact")
    normalized_selected = deterministic_eval_selection(
        selected_rows,
        seed=0,
        samples_per_family=0,
        max_eval_samples=0,
    )
    config = benchmark.get("config")
    curriculum_rows = curriculum_evidence.get("curriculum_eval_rows")
    if not isinstance(config, Mapping) or not isinstance(curriculum_rows, Sequence):
        raise ValueError("Benchmark or curriculum is missing evaluation-selection configuration.")
    seed = config.get("seed")
    samples_per_family = config.get("samples_per_family")
    max_eval_samples = config.get("max_eval_samples")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Benchmark selection seed must be an integer.")
    if isinstance(samples_per_family, bool) or not isinstance(samples_per_family, int):
        raise ValueError("Benchmark samples_per_family must be a non-negative integer.")
    if isinstance(max_eval_samples, bool) or not isinstance(max_eval_samples, int):
        raise ValueError("Benchmark max_eval_samples must be a non-negative integer.")
    expected_selected = deterministic_eval_selection(
        curriculum_rows,
        seed=seed,
        samples_per_family=samples_per_family,
        max_eval_samples=max_eval_samples,
    )
    try:
        actual_identity = json.dumps(
            normalized_selected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        expected_identity = json.dumps(
            expected_selected,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Evaluation selection contains non-canonical JSON values.") from exc
    if actual_identity != expected_identity:
        raise ValueError(
            "Selected eval artifact is not the deterministic ordered subset of the curriculum."
        )

    family_counts: dict[str, int] = {}
    for row in normalized_selected:
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping):  # pragma: no cover - validated above
            raise ValueError("Selected eval row is missing normalized metadata.")
        family = _metric_family_name(metadata.get("problem_family"))
        family_counts[family] = family_counts.get(family, 0) + 1
    row_count = len(normalized_selected)
    if row_count < 1 or not family_counts:
        raise ValueError("Selected eval artifact is empty.")
    return {
        "selected_eval": str(selected_eval_path.resolve()),
        "selected_eval_sha256": hashlib.sha256(selected_eval_bytes).hexdigest(),
        "selected_eval_samples": row_count,
        "selected_eval_family_counts": dict(sorted(family_counts.items())),
        "selected_eval_identity_sha256": hashlib.sha256(actual_identity).hexdigest(),
        "selection_seed": seed,
        "samples_per_family": samples_per_family,
        "max_eval_samples": max_eval_samples,
    }


def _load_jsonl_mappings(path: Path, *, label: str) -> list[dict[str, object]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8 JSONL.") from exc
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        cooked = line.strip()
        if not cooked:
            continue
        try:
            row = json.loads(cooked)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {line_number} is invalid JSON.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{label} row {line_number} must be a JSON object.")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} is empty.")
    return rows


def _paired_artifact_evidence(
    benchmark_path: Path,
    benchmark: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str], dict[str, Path]]:
    """Recompute paired evidence from the three content-bound sibling artifacts."""

    artifact_paths = {
        "base_samples_sha256": benchmark_path.parent / "base_samples.jsonl",
        "tuned_samples_sha256": benchmark_path.parent / "tuned_samples.jsonl",
        "sample_comparison_sha256": benchmark_path.parent / "sample_comparison.jsonl",
    }
    missing = [path.name for path in artifact_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Paired evidence artifacts are missing: {', '.join(missing)}")
    actual_hashes = {
        field: sha256_file(path)
        for field, path in artifact_paths.items()
    }
    declared_hashes = benchmark.get("artifact_hashes")
    provenance = benchmark.get("provenance")
    if not isinstance(declared_hashes, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("Benchmark is missing paired artifact hash provenance.")
    for field, digest in actual_hashes.items():
        if str(declared_hashes.get(field) or "").strip().lower() != digest:
            raise ValueError(f"Benchmark {field} does not match its detailed artifact.")
        if str(provenance.get(field) or "").strip().lower() != digest:
            raise ValueError(f"Benchmark provenance {field} does not match its detailed artifact.")

    eval_path = benchmark_path.parent / "eval_pairs.jsonl"
    eval_rows = _load_jsonl_mappings(eval_path, label="Selected eval artifact")
    base_rows = _load_jsonl_mappings(
        artifact_paths["base_samples_sha256"],
        label="Base sample artifact",
    )
    tuned_rows = _load_jsonl_mappings(
        artifact_paths["tuned_samples_sha256"],
        label="Tuned sample artifact",
    )
    comparison_rows = _load_jsonl_mappings(
        artifact_paths["sample_comparison_sha256"],
        label="Sample comparison artifact",
    )
    validate_sample_comparison(comparison_rows, base_rows, tuned_rows)
    expected_evidence = benchmark.get("paired_evidence")
    evidence = recompute_and_validate_paired_evidence(
        expected_evidence if isinstance(expected_evidence, Mapping) else {},
        base_rows,
        tuned_rows,
        eval_rows,
        artifact_hashes=actual_hashes,
    )
    recomputed_metrics = evidence.get("recomputed_metrics")
    reference_metrics = benchmark.get("base")
    tuned_metrics = benchmark.get("tuned")
    if not isinstance(recomputed_metrics, Mapping):
        raise ValueError("Paired evidence is missing recomputed aggregate metrics.")
    validate_reported_metrics(
        reference_metrics if isinstance(reference_metrics, Mapping) else {},
        recomputed_metrics.get("base") if isinstance(recomputed_metrics.get("base"), Mapping) else {},
        side="base",
    )
    validate_reported_metrics(
        tuned_metrics if isinstance(tuned_metrics, Mapping) else {},
        recomputed_metrics.get("tuned") if isinstance(recomputed_metrics.get("tuned"), Mapping) else {},
        side="tuned",
    )
    evidence_digest = paired_evidence_sha256(evidence)
    if str(provenance.get("paired_evidence_sha256") or "").strip().lower() != evidence_digest:
        raise ValueError("Benchmark paired evidence digest does not match recomputed evidence.")
    if str(provenance.get("paired_evidence_schema") or "") != PAIRED_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Benchmark paired evidence schema is unsupported.")
    return evidence, {**actual_hashes, "paired_evidence_sha256": evidence_digest}, artifact_paths


def _same_existing_path(raw_path: object, expected: Path) -> bool:
    text = str(raw_path or "").strip()
    if not text:
        return False
    try:
        return Path(text).expanduser().resolve().samefile(expected.resolve())
    except (OSError, ValueError):
        return False


def _expected_hf_cache_model_dir(base_model: str) -> str:
    parts = [part for part in str(base_model).strip().split("/") if part]
    return "models--" + "--".join(parts) if len(parts) >= 2 else ""


def _valid_resolved_snapshot(
    path: Path,
    *,
    base_model: str,
    revision: str,
) -> bool:
    expected_cache_dir = _expected_hf_cache_model_dir(base_model)
    try:
        resolved = path.expanduser().resolve()
    except (OSError, ValueError):
        return False
    return bool(
        expected_cache_dir
        and resolved.is_dir()
        and resolved.name.casefold() == revision.casefold()
        and resolved.parent.name.casefold() == "snapshots"
        and resolved.parent.parent.name.casefold() == expected_cache_dir.casefold()
    )


def _adapter_config_base_model_matches(
    adapter_config: Mapping[str, object],
    *,
    base_model: str,
    base_model_revision: str,
    resolved_base_model: Path,
) -> bool:
    raw_value = adapter_config.get("base_model_name_or_path")
    if not isinstance(raw_value, str) or not raw_value.strip():
        return False
    raw = raw_value.strip().rstrip("/\\")
    expected_model = str(base_model).strip().rstrip("/\\")
    if raw == expected_model:
        return True
    candidate = Path(raw).expanduser()
    if not _valid_resolved_snapshot(
        candidate,
        base_model=expected_model,
        revision=base_model_revision,
    ):
        return False
    return _same_existing_path(candidate, resolved_base_model)


def _benchmark_binding_blockers(
    benchmark: Mapping[str, object],
    *,
    adapter: Path,
    adapter_sha256: str,
    adapter_config_sha256: str,
    adapter_config: Mapping[str, object],
    base_model: str,
    base_model_revision: str,
    curriculum_evidence: Mapping[str, object],
    selected_eval_evidence: Mapping[str, object],
    paired_evidence: Mapping[str, object],
    code_hashes: Mapping[str, object],
) -> list[str]:
    blockers: list[str] = []
    if benchmark.get("schema") != BENCHMARK_SCHEMA_VERSION:
        blockers.append("unsupported_benchmark_schema")
    config = benchmark.get("config")
    provenance = benchmark.get("provenance")
    if not isinstance(config, Mapping):
        blockers.append("missing_benchmark_config")
        config = {}
    if not isinstance(provenance, Mapping):
        blockers.append("missing_benchmark_provenance")
        provenance = {}

    revision = str(base_model_revision or "").strip()
    if not revision:
        blockers.append("missing_base_model_revision")
    if str(config.get("base_model") or "") != str(base_model):
        blockers.append("benchmark_base_model_mismatch")
    if str(config.get("base_model_revision") or "") != revision:
        blockers.append("benchmark_base_model_revision_mismatch")
    if str(provenance.get("base_model") or "") != str(base_model):
        blockers.append("provenance_base_model_mismatch")
    if str(provenance.get("base_model_revision") or "") != revision:
        blockers.append("provenance_base_model_revision_mismatch")
    resolved_base_model = Path(str(config.get("resolved_base_model_path") or "")).expanduser()
    if not _valid_resolved_snapshot(
        resolved_base_model,
        base_model=str(base_model),
        revision=revision,
    ):
        blockers.append("benchmark_resolved_base_model_revision_mismatch")
    if not _adapter_config_base_model_matches(
        adapter_config,
        base_model=str(base_model),
        base_model_revision=revision,
        resolved_base_model=resolved_base_model,
    ):
        blockers.append("adapter_config_base_model_mismatch")
    if not _same_existing_path(config.get("adapter_dir"), adapter):
        blockers.append("benchmark_adapter_path_mismatch")
    if str(config.get("reference_adapter_dir") or "").strip():
        blockers.append("benchmark_reference_is_not_base_model")

    if (
        str(provenance.get("adapter_sha256") or "").strip().lower()
        != adapter_sha256
    ):
        blockers.append("benchmark_adapter_weights_mismatch")
    if (
        str(provenance.get("adapter_config_sha256") or "").strip().lower()
        != adapter_config_sha256
    ):
        blockers.append("benchmark_adapter_config_mismatch")

    selected_sha = str(selected_eval_evidence.get("selected_eval_sha256") or "")
    if str(provenance.get("selected_eval_sha256") or "").strip().lower() != selected_sha:
        blockers.append("benchmark_selected_eval_mismatch")
    configured_eval_samples = _positive_count(config.get("eval_samples"))
    if configured_eval_samples != int(selected_eval_evidence.get("selected_eval_samples") or 0):
        blockers.append("benchmark_selected_eval_count_mismatch")
    configured_max_new_tokens = _positive_count(config.get("max_new_tokens"))
    reference = benchmark.get("base")
    tuned = benchmark.get("tuned")
    if configured_max_new_tokens is None:
        blockers.append("missing_or_invalid_benchmark_generation_cap")
    elif isinstance(reference, Mapping) and isinstance(tuned, Mapping):
        expected_generation_cap = max(8, configured_max_new_tokens)
        if (
            _positive_count(reference.get("generation_cap")) != expected_generation_cap
            or _positive_count(tuned.get("generation_cap")) != expected_generation_cap
        ):
            blockers.append("benchmark_generation_cap_mismatch")

    bootstrap = paired_evidence.get("template_cluster_bootstrap")
    if not isinstance(bootstrap, Mapping):
        blockers.append("missing_paired_bootstrap_configuration")
    else:
        for key in ("seed", "resamples"):
            configured = config.get(f"paired_bootstrap_{key}")
            evidenced = bootstrap.get(key)
            if (
                isinstance(configured, bool)
                or not isinstance(configured, int)
                or isinstance(evidenced, bool)
                or not isinstance(evidenced, int)
                or configured != evidenced
            ):
                blockers.append(f"benchmark_paired_bootstrap_{key}_mismatch")

    curriculum_manifest_sha = str(curriculum_evidence.get("curriculum_manifest_sha256") or "")
    curriculum_eval_sha = str(curriculum_evidence.get("curriculum_eval_sha256") or "")
    if not _same_existing_path(
        config.get("curriculum_manifest"),
        Path(str(curriculum_evidence.get("curriculum_manifest") or "")),
    ):
        blockers.append("benchmark_curriculum_manifest_path_mismatch")
    if not _same_existing_path(
        config.get("eval_source"),
        Path(str(curriculum_evidence.get("curriculum_eval") or "")),
    ):
        blockers.append("benchmark_curriculum_eval_path_mismatch")
    if (
        str(provenance.get("curriculum_manifest_sha256") or "").strip().lower()
        != curriculum_manifest_sha
    ):
        blockers.append("benchmark_curriculum_manifest_mismatch")
    if str(provenance.get("curriculum_eval_sha256") or "").strip().lower() != curriculum_eval_sha:
        blockers.append("benchmark_curriculum_eval_mismatch")
    if str(provenance.get("verifier_schema") or "") != VERIFIER_SCHEMA_VERSION:
        blockers.append("benchmark_verifier_schema_mismatch")
    if provenance.get("code_hashes") != dict(code_hashes):
        blockers.append("benchmark_evaluator_or_verifier_hash_mismatch")
    return blockers


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_gate(
    *,
    benchmark_path: Path,
    adapter_dir: Path,
    curriculum_manifest_path: Path,
    pointer_path: Path,
    base_model: str,
    base_model_revision: str = "",
    min_verified_samples: int,
    min_verified_gain: float,
    min_tuned_accuracy: float,
    max_family_regression: float,
    max_loss_ratio: float,
    min_token_f1_delta: float,
    min_family_verified_samples: int = 1,
    max_generation_cap_rate: float = 0.05,
    max_paired_p_value: float = 0.05,
    max_paired_regression_rate: float = 0.02,
    min_paired_cluster_lower_bound: float = 0.0,
    min_template_clusters: int = 5,
    write_pointer: bool,
) -> dict[str, object]:
    benchmark_file = benchmark_path.expanduser().resolve()
    try:
        benchmark_bytes = benchmark_file.read_bytes()
        benchmark = json.loads(benchmark_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Benchmark result must be valid UTF-8 JSON.") from exc
    if not isinstance(benchmark, dict):
        raise ValueError("Benchmark result must be a JSON object.")
    benchmark_sha = hashlib.sha256(benchmark_bytes).hexdigest()
    adapter = adapter_dir.resolve()
    weights_path = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Adapter weights or configuration are missing.")
    adapter_sha = sha256_file(weights_path)
    adapter_config_bytes = config_path.read_bytes()
    adapter_config_sha = hashlib.sha256(adapter_config_bytes).hexdigest()
    adapter_config: dict[str, object] = {}
    adapter_config_error = ""
    try:
        raw_adapter_config = json.loads(adapter_config_bytes.decode("utf-8"))
        if not isinstance(raw_adapter_config, dict):
            raise ValueError("Adapter configuration must be a JSON object.")
        adapter_config = raw_adapter_config
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        adapter_config_error = str(exc)
    curriculum_manifest = curriculum_manifest_path.expanduser().resolve()
    if not curriculum_manifest.is_file():
        raise FileNotFoundError(f"Curriculum manifest not found: {curriculum_manifest}")

    binding_blockers: list[str] = []
    if adapter_config_error:
        binding_blockers.append("invalid_adapter_config")
    curriculum: dict[str, object] = {
        "curriculum_manifest": str(curriculum_manifest),
        "curriculum_manifest_sha256": sha256_file(curriculum_manifest),
        "curriculum_eval": "",
        "curriculum_eval_sha256": "",
        "curriculum_schema": "",
        "curriculum_seed": None,
        "curriculum_eval_samples": 0,
        "curriculum_eval_rows": [],
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
    }
    try:
        curriculum = _curriculum_evidence(curriculum_manifest)
    except (OSError, TypeError, ValueError) as exc:
        binding_blockers.append("invalid_curriculum_manifest_or_eval")
        curriculum["error"] = str(exc)

    selected_eval: dict[str, object] = {
        "selected_eval": str((benchmark_file.parent / "eval_pairs.jsonl").resolve()),
        "selected_eval_sha256": "",
        "selected_eval_samples": 0,
        "selected_eval_family_counts": {},
    }
    try:
        selected_eval = _selected_eval_evidence(benchmark_file, benchmark, curriculum)
    except (OSError, TypeError, ValueError) as exc:
        binding_blockers.append("invalid_selected_eval_artifact")
        selected_eval["error"] = str(exc)

    paired_evidence: dict[str, object] = {}
    paired_hashes: dict[str, str] = {
        "base_samples_sha256": "",
        "tuned_samples_sha256": "",
        "sample_comparison_sha256": "",
        "paired_evidence_sha256": "",
    }
    paired_paths: dict[str, Path] = {}
    try:
        paired_evidence, paired_hashes, paired_paths = _paired_artifact_evidence(
            benchmark_file,
            benchmark,
        )
    except (OSError, TypeError, ValueError) as exc:
        binding_blockers.append("invalid_paired_evidence_or_artifacts")
        paired_evidence["error"] = str(exc)

    code_hashes = evaluation_code_hashes(Path(__file__).resolve().parent)
    binding_blockers.extend(
        _benchmark_binding_blockers(
            benchmark,
            adapter=adapter,
            adapter_sha256=adapter_sha,
            adapter_config_sha256=adapter_config_sha,
            adapter_config=adapter_config,
            base_model=str(base_model),
            base_model_revision=str(base_model_revision),
            curriculum_evidence=curriculum,
            selected_eval_evidence=selected_eval,
            paired_evidence=paired_evidence,
            code_hashes=code_hashes,
        )
    )
    if sha256_file(benchmark_file) != benchmark_sha:
        binding_blockers.append("benchmark_changed_during_gate")
    if sha256_file(weights_path) != adapter_sha or sha256_file(config_path) != adapter_config_sha:
        binding_blockers.append("adapter_changed_during_gate")
    if sha256_file(curriculum_manifest) != str(
        curriculum.get("curriculum_manifest_sha256") or ""
    ):
        binding_blockers.append("curriculum_manifest_changed_during_gate")
    curriculum_eval_path = Path(str(curriculum.get("curriculum_eval") or ""))
    if curriculum_eval_path.is_file() and sha256_file(curriculum_eval_path) != str(
        curriculum.get("curriculum_eval_sha256") or ""
    ):
        binding_blockers.append("curriculum_eval_changed_during_gate")
    selected_eval_path = Path(str(selected_eval.get("selected_eval") or ""))
    if selected_eval_path.is_file() and sha256_file(selected_eval_path) != str(
        selected_eval.get("selected_eval_sha256") or ""
    ):
        binding_blockers.append("selected_eval_changed_during_gate")
    if evaluation_code_hashes(Path(__file__).resolve().parent) != code_hashes:
        binding_blockers.append("evaluator_or_verifier_changed_during_gate")
    for field, path in paired_paths.items():
        if not path.is_file() or sha256_file(path) != paired_hashes.get(field):
            binding_blockers.append(f"paired_artifact_changed_during_gate:{path.name}")

    requested_thresholds: dict[str, object] = {
        "min_verified_samples": min_verified_samples,
        "min_verified_gain": min_verified_gain,
        "min_tuned_accuracy": min_tuned_accuracy,
        "max_family_regression": max_family_regression,
        "max_loss_ratio": max_loss_ratio,
        "min_token_f1_delta": min_token_f1_delta,
        "min_family_verified_samples": min_family_verified_samples,
        "max_generation_cap_rate": max_generation_cap_rate,
        "max_paired_p_value": max_paired_p_value,
        "max_paired_regression_rate": max_paired_regression_rate,
        "min_paired_cluster_lower_bound": min_paired_cluster_lower_bound,
        "min_template_clusters": min_template_clusters,
    }
    threshold_floor_violations = _threshold_policy_violations(requested_thresholds)
    protocol_violations = _protocol_policy_violations(
        benchmark,
        curriculum,
        paired_evidence,
    )
    production_eligible = not threshold_floor_violations and not protocol_violations
    policy_mode = (
        "production"
        if production_eligible and write_pointer
        else ("review" if production_eligible else "research")
    )
    policy_blockers: list[str] = []
    if write_pointer and not production_eligible:
        policy_blockers.append("nonproduction_policy_requires_no_write_pointer")
        policy_blockers.extend(
            f"production_threshold_floor_violation:{key}"
            for key in threshold_floor_violations
        )
        policy_blockers.extend(
            f"production_protocol_violation:{key}"
            for key in protocol_violations
        )
    policy_payload: dict[str, object] = {
        "policy_id": PRODUCTION_POLICY_ID,
        "policy_mode": policy_mode,
        "production_eligible": production_eligible,
        "write_pointer_requested": bool(write_pointer),
        "threshold_floor_violations": threshold_floor_violations,
        "protocol_violations": protocol_violations,
        "production_threshold_floors": dict(PRODUCTION_THRESHOLD_FLOORS),
        "production_protocol": dict(PRODUCTION_PROTOCOL),
    }

    metric_decision = evaluate_promotion(
        benchmark,
        min_verified_samples=min_verified_samples,
        min_verified_gain=min_verified_gain,
        min_tuned_accuracy=min_tuned_accuracy,
        max_family_regression=max_family_regression,
        max_loss_ratio=max_loss_ratio,
        min_token_f1_delta=min_token_f1_delta,
        min_family_verified_samples=min_family_verified_samples,
        max_generation_cap_rate=max_generation_cap_rate,
        max_paired_p_value=max_paired_p_value,
        max_paired_regression_rate=max_paired_regression_rate,
        min_paired_cluster_lower_bound=min_paired_cluster_lower_bound,
        min_template_clusters=min_template_clusters,
        expected_family_counts=(
            selected_eval.get("selected_eval_family_counts")
            if isinstance(selected_eval.get("selected_eval_family_counts"), Mapping)
            else None
        ),
        expected_eval_samples=(
            int(selected_eval["selected_eval_samples"])
            if int(selected_eval.get("selected_eval_samples") or 0) > 0
            else None
        ),
    )
    metric_blockers = list(metric_decision.get("blockers") or [])
    blockers = list(
        dict.fromkeys([*binding_blockers, *metric_blockers, *policy_blockers])
    )
    decision = dict(metric_decision)
    decision.update(
        {
            "passed": not blockers,
            "blockers": blockers,
            "binding_blockers": list(dict.fromkeys(binding_blockers)),
            "metric_blockers": metric_blockers,
            "policy_blockers": policy_blockers,
            **policy_payload,
            "binding": {
                "benchmark_schema": str(benchmark.get("schema") or ""),
                "benchmark_sha256": benchmark_sha,
                "base_model": str(base_model),
                "base_model_revision": str(base_model_revision),
                "adapter_sha256": adapter_sha,
                "adapter_config_sha256": adapter_config_sha,
                "selected_eval_sha256": str(selected_eval.get("selected_eval_sha256") or ""),
                "base_samples_sha256": paired_hashes["base_samples_sha256"],
                "tuned_samples_sha256": paired_hashes["tuned_samples_sha256"],
                "sample_comparison_sha256": paired_hashes["sample_comparison_sha256"],
                "paired_evidence_sha256": paired_hashes["paired_evidence_sha256"],
                "paired_evidence_schema": str(paired_evidence.get("schema") or ""),
                "policy_id": PRODUCTION_POLICY_ID,
                "policy_mode": policy_mode,
                "production_eligible": production_eligible,
                "curriculum_manifest_sha256": str(
                    curriculum.get("curriculum_manifest_sha256") or ""
                ),
                "curriculum_eval_sha256": str(curriculum.get("curriculum_eval_sha256") or ""),
                "verifier_schema": VERIFIER_SCHEMA_VERSION,
                "code_hashes": code_hashes,
            },
        }
    )
    artifact_dir = adapter.parent
    gate_payload: dict[str, object] = {
        "schema": GATE_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": bool(decision["passed"]),
        **policy_payload,
        "base_model": str(base_model),
        "base_model_revision": str(base_model_revision),
        "adapter_dir": str(adapter),
        "adapter_sha256": adapter_sha,
        "adapter_config_sha256": adapter_config_sha,
        "benchmark_schema": str(benchmark.get("schema") or ""),
        "benchmark_file": str(benchmark_file),
        "benchmark_sha256": benchmark_sha,
        "selected_eval": str(selected_eval.get("selected_eval") or ""),
        "selected_eval_sha256": str(selected_eval.get("selected_eval_sha256") or ""),
        "base_samples_sha256": paired_hashes["base_samples_sha256"],
        "tuned_samples_sha256": paired_hashes["tuned_samples_sha256"],
        "sample_comparison_sha256": paired_hashes["sample_comparison_sha256"],
        "paired_evidence_sha256": paired_hashes["paired_evidence_sha256"],
        "paired_evidence_schema": str(paired_evidence.get("schema") or ""),
        "curriculum_manifest": str(curriculum_manifest),
        "curriculum_manifest_sha256": str(curriculum.get("curriculum_manifest_sha256") or ""),
        "curriculum_eval": str(curriculum.get("curriculum_eval") or ""),
        "curriculum_eval_sha256": str(curriculum.get("curriculum_eval_sha256") or ""),
        "verifier_schema": VERIFIER_SCHEMA_VERSION,
        "code_hashes": code_hashes,
        "decision": decision,
    }
    gate_path = artifact_dir / GATE_FILENAME
    _atomic_json(gate_path, gate_payload)

    manifest_path = artifact_dir / PROMOTION_FILENAME
    if decision["passed"] and policy_mode == "production":
        promotion_payload = {
            "schema": PROMOTION_SCHEMA_VERSION,
            "passed": True,
            "policy_id": PRODUCTION_POLICY_ID,
            "policy_mode": policy_mode,
            "production_eligible": True,
            "production_threshold_floors": dict(PRODUCTION_THRESHOLD_FLOORS),
            "production_protocol": dict(PRODUCTION_PROTOCOL),
            "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model": str(base_model),
            "base_model_revision": str(base_model_revision),
            "adapter_sha256": adapter_sha,
            "adapter_config_sha256": adapter_config_sha,
            "benchmark_schema": str(benchmark.get("schema") or ""),
            "benchmark_sha256": benchmark_sha,
            "selected_eval_sha256": str(selected_eval.get("selected_eval_sha256") or ""),
            "base_samples_sha256": paired_hashes["base_samples_sha256"],
            "tuned_samples_sha256": paired_hashes["tuned_samples_sha256"],
            "sample_comparison_sha256": paired_hashes["sample_comparison_sha256"],
            "paired_evidence_sha256": paired_hashes["paired_evidence_sha256"],
            "paired_evidence_schema": str(paired_evidence.get("schema") or ""),
            "curriculum_manifest_sha256": str(
                curriculum.get("curriculum_manifest_sha256") or ""
            ),
            "curriculum_eval_sha256": str(curriculum.get("curriculum_eval_sha256") or ""),
            "verifier_schema": VERIFIER_SCHEMA_VERSION,
            "code_hashes": code_hashes,
            "gate_file": gate_path.name,
            "gate_sha256": sha256_file(gate_path),
        }
        _atomic_json(manifest_path, promotion_payload)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.write_text(str(adapter), encoding="utf-8")
    elif manifest_path.exists():
        # A newly failed gate must not leave a stale pass receipt active.
        manifest_path.unlink()

    return {
        "passed": bool(decision["passed"]),
        "blockers": blockers,
        "binding_blockers": list(dict.fromkeys(binding_blockers)),
        "metric_blockers": metric_blockers,
        "policy_blockers": policy_blockers,
        "policy_id": PRODUCTION_POLICY_ID,
        "policy_mode": policy_mode,
        "production_eligible": production_eligible,
        "gate_path": str(gate_path),
        "promotion_manifest_path": str(manifest_path) if manifest_path.exists() else "",
        "pointer_written": bool(decision["passed"] and policy_mode == "production"),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate and explicitly promote one Qwen adapter.")
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--curriculum-manifest", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, default=Path(".gui_default_adapter.txt"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument(
        "--base-model-revision",
        default="",
        help="Immutable model revision. A missing revision is a fail-closed gate blocker.",
    )
    parser.add_argument(
        "--min-verified-samples",
        type=int,
        default=PRODUCTION_THRESHOLD_FLOORS["min_verified_samples"],
    )
    parser.add_argument(
        "--min-verified-gain",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["min_verified_gain"],
    )
    parser.add_argument(
        "--min-tuned-accuracy",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["min_tuned_accuracy"],
    )
    parser.add_argument(
        "--max-family-regression",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["max_family_regression"],
    )
    parser.add_argument(
        "--max-loss-ratio",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["max_loss_ratio"],
    )
    parser.add_argument(
        "--min-token-f1-delta",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["min_token_f1_delta"],
    )
    parser.add_argument(
        "--min-family-verified-samples",
        type=int,
        default=PRODUCTION_THRESHOLD_FLOORS["min_family_verified_samples"],
    )
    parser.add_argument(
        "--max-generation-cap-rate",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["max_generation_cap_rate"],
    )
    parser.add_argument(
        "--max-paired-p-value",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["max_paired_p_value"],
    )
    parser.add_argument(
        "--max-paired-regression-rate",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["max_paired_regression_rate"],
    )
    parser.add_argument(
        "--min-paired-cluster-lower-bound",
        type=float,
        default=PRODUCTION_THRESHOLD_FLOORS["min_paired_cluster_lower_bound"],
    )
    parser.add_argument(
        "--min-template-clusters",
        type=int,
        default=PRODUCTION_THRESHOLD_FLOORS["min_template_clusters"],
    )
    parser.add_argument("--no-write-pointer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_gate(
        benchmark_path=args.benchmark_json,
        adapter_dir=args.adapter_dir,
        curriculum_manifest_path=args.curriculum_manifest,
        pointer_path=args.pointer,
        base_model=str(args.base_model),
        base_model_revision=str(args.base_model_revision),
        min_verified_samples=int(args.min_verified_samples),
        min_verified_gain=float(args.min_verified_gain),
        min_tuned_accuracy=float(args.min_tuned_accuracy),
        max_family_regression=float(args.max_family_regression),
        max_loss_ratio=float(args.max_loss_ratio),
        min_token_f1_delta=float(args.min_token_f1_delta),
        min_family_verified_samples=int(args.min_family_verified_samples),
        max_generation_cap_rate=float(args.max_generation_cap_rate),
        max_paired_p_value=float(args.max_paired_p_value),
        max_paired_regression_rate=float(args.max_paired_regression_rate),
        min_paired_cluster_lower_bound=float(args.min_paired_cluster_lower_bound),
        min_template_clusters=int(args.min_template_clusters),
        write_pointer=not bool(args.no_write_pointer),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PRODUCTION_PROTOCOL",
    "PRODUCTION_THRESHOLD_FLOORS",
    "evaluate_promotion",
    "main",
    "parse_args",
    "run_gate",
]
