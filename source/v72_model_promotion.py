"""Preregister and review-gate a collision-free MiMoMix successor.

The v65-v70 arithmetic reports repeatedly reused one deterministic 120-prompt
set.  Worse, exact copies of 40-41 of those supposedly novel prompts occur in
the compared training corpora.  That set remains useful for development, but it
cannot be the authority for another promotion.

This module creates a separate v72 protocol with two deliberately distinct
phases:

``freeze``
    Runs before the final candidate checkpoint exists.  It deterministically
    constructs a balanced, multi-seed prompt set, rejects exact prompt
    collisions against both training corpora, and atomically writes a manifest
    binding the prompts, corpora, v70 checkpoint, policy, and evaluator source.

``evaluate``
    Runs only after the expected candidate checkpoint appears.  It verifies
    every frozen hash again, scores v70 and the candidate on exactly the same
    prompts, applies conjunctive non-regression gates, and writes a content-
    bound review receipt.  It has no code path that writes a model pointer.

Example::

    python source/v72_model_promotion.py freeze \
      --baseline-checkpoint output/v70_moe/v70_moe.pt \
      --baseline-corpus datasets/v70/v70_combined.jsonl \
      --candidate-corpus datasets/v71/v71_combined.jsonl \
      --candidate-checkpoint output/v71_decomposed/v71_decomposed.pt \
      --output output/v72_promotion/frozen_manifest.json

    python source/v72_model_promotion.py evaluate \
      --manifest output/v72_promotion/frozen_manifest.json \
      --candidate-checkpoint output/v71_decomposed/v71_decomposed.pt \
      --output output/v72_promotion/promotion_receipt.json \
      --no-write-pointer
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA = "supermix-v72-collision-free-promotion-manifest-v1"
RECEIPT_SCHEMA = "supermix-v72-collision-free-promotion-receipt-v1"
POLICY_ID = "supermix-v72-collision-free-promotion-policy-v1"

# Production comparator identity.  A path called ``v70_moe.pt`` is not enough:
# the bytes are the comparator.  Tests replace these module constants in-memory
# with tiny synthetic identities; the CLI has no override flag.
V70_CHECKPOINT_SHA256 = "f3b97bd2eeae38b4a91a98d0bf71642307b3a3f0110b38d70b95e978fbc28567"
V70_CORPUS_SHA256 = "ac1b844608743f8e6750554722bf9b34a8980f55ce8fade3b66e153e8c5648fd"

_PREDECESSOR_IDENTITIES = {
    "v68_average_fix": {
        "checkpoint_sha256": "777648137669fa1fc0984fc11aa2eb7ac20347fb1266436a455bc97d2dfef502",
        "corpus_sha256": "36460825add48b17631b89ccb0f413441e4dcb6006f6c291cf759afaaec1c7e9",
        "role": "archived_math_specialist",
    },
    "v69_unified_b": {
        "checkpoint_sha256": "2c1cdbd5f1af2a4e7d955838feac239d2da1a882c14ec424373b8412952da5b8",
        "corpus_sha256": "ba26445ed4b2693c0309dfe28dc31100d61f3e53bebc4c061f75f4fa672bd322",
        "role": "archived_unified_predecessor",
    },
}

FAMILIES: Tuple[str, ...] = (
    "arithmetic",
    "percent",
    "average",
    "algebra_one_step",
    "word_problem",
)
EVALUATION_SEEDS: Tuple[int, ...] = (7201, 7211, 7221, 7231)
SAMPLES_PER_FAMILY_PER_SEED = 8
POOL_MULTIPLIER = 16

MATH_MAX_NEW_TOKENS = 96
CHAT_MAX_NEW_TOKENS = 48
TORCH_NUM_THREADS = 4

MIN_OVERALL_ACCURACY_GAIN = 0.05
MAX_FAMILY_REGRESSION = 0.0
MAX_UNPARSED_RATE = 0.05
MAX_GENERATION_CAP_RATE = 0.05
MAX_PROMPT_UNKNOWN_RATE = 0.05
MAX_PAIRED_ONE_SIDED_P_VALUE = 0.05
MIN_CHAT_OPERATIONAL = 7
MIN_CHAT_SIMILAR = 6
MIN_CHAT_TOKEN_F1 = 0.50

LEGACY_DEVELOPMENT_SEED = 65
LEGACY_DEVELOPMENT_COUNT = 120

PROMPT_FORMS: Tuple[str, ...] = (
    "Solve this basic math problem: {expression}",
    "Quick question: {expression}",
    "Please help with this. {expression}",
    "What is {expression}?",
)

CHAT_PROMPTS: Tuple[str, ...] = (
    "hello",
    "can you help me with tests",
    "why is my script failing",
    "what is your name",
    "write a unit test for login",
    "please explain this error",
    "help me debug a Python function",
    "how should I test an API",
)

_FAMILY_OFFSETS = {
    "arithmetic": 101,
    "percent": 211,
    "average": 307,
    "algebra_one_step": 401,
    "word_problem": 503,
}

_NUMBER_RE = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?:/[-+]?\d+)?"
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _records_sha256(records: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json_bytes(record))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write once via same-directory replace; never overwrite a frozen artifact."""

    destination = path.resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(
                f"immutable artifact appeared while writing: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _required_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def production_protocol() -> Dict[str, object]:
    return {
        "comparator_id": "v70_moe",
        "comparator_checkpoint_sha256": V70_CHECKPOINT_SHA256,
        "comparator_corpus_sha256": V70_CORPUS_SHA256,
        "families": list(FAMILIES),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "samples_per_family_per_seed": SAMPLES_PER_FAMILY_PER_SEED,
        "selected_math_prompts": (
            len(FAMILIES) * len(EVALUATION_SEEDS) * SAMPLES_PER_FAMILY_PER_SEED
        ),
        "chat_prompts": len(CHAT_PROMPTS),
        "math_max_new_tokens": MATH_MAX_NEW_TOKENS,
        "chat_max_new_tokens": CHAT_MAX_NEW_TOKENS,
        "decoding": "greedy_non_speculative",
        "torch_num_threads": TORCH_NUM_THREADS,
        "exact_prompt_collision_policy": "reject_against_both_corpora",
        "candidate_checkpoint_must_be_absent_at_freeze": True,
        "pointer_mode": "review_only_no_write_implementation",
    }


def comparator_lineage() -> Dict[str, object]:
    return {
        **_PREDECESSOR_IDENTITIES,
        "v70_moe": {
            "checkpoint_sha256": V70_CHECKPOINT_SHA256,
            "corpus_sha256": V70_CORPUS_SHA256,
            "role": "immutable_production_comparator",
        },
    }


def production_thresholds() -> Dict[str, object]:
    return {
        "min_overall_accuracy_gain": MIN_OVERALL_ACCURACY_GAIN,
        "max_family_regression": MAX_FAMILY_REGRESSION,
        "require_strict_family_gain": True,
        "require_no_seed_regression": True,
        "max_unparsed_rate": MAX_UNPARSED_RATE,
        "max_generation_cap_rate": MAX_GENERATION_CAP_RATE,
        "max_prompt_unknown_rate": MAX_PROMPT_UNKNOWN_RATE,
        "max_paired_one_sided_p_value": MAX_PAIRED_ONE_SIDED_P_VALUE,
        "min_chat_operational_prompts": MIN_CHAT_OPERATIONAL,
        "min_chat_similar_prompts": MIN_CHAT_SIMILAR,
        "min_chat_token_f1": MIN_CHAT_TOKEN_F1,
    }


def _fraction_payload(value: Fraction) -> Dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _outside_training_integer(rng: random.Random, low: int, high: int) -> int:
    magnitude = rng.randint(low, high)
    return magnitude if rng.randrange(2) else -magnitude


def _promotion_problem(family: str, rng: random.Random) -> Dict[str, object]:
    """Generate one task outside the old numeric support, using familiar wording."""

    if family == "arithmetic":
        a, b = rng.randint(1000, 9999), rng.randint(1000, 9999)
        op = rng.choice(("+", "-"))
        answer = a + b if op == "+" else a - b
        form_index = rng.randrange(len(PROMPT_FORMS))
        prompt = PROMPT_FORMS[form_index].format(expression=f"{a} {op} {b}")
        return {
            "family": family,
            "prompt": prompt,
            "expected": _fraction_payload(Fraction(answer, 1)),
            "template_id": f"binary_form_{form_index}",
        }

    if family == "percent":
        percent = rng.choice((5, 10, 20, 25, 50))
        base = rng.randint(2001, 9999)
        return {
            "family": family,
            "prompt": f"What is {percent}% of {base}?",
            "expected": _fraction_payload(Fraction(percent * base, 100)),
            "template_id": "percent_outside_base_range",
        }

    if family == "average":
        values = [rng.randint(100, 999) for _ in range(rng.choice((4, 5, 6)))]
        prompt = "Find the average (mean) of these numbers: " + ", ".join(
            str(value) for value in values
        )
        return {
            "family": family,
            "prompt": prompt,
            "expected": _fraction_payload(Fraction(sum(values), len(values))),
            "template_id": f"average_{len(values)}_outside_value_range",
        }

    if family == "algebra_one_step":
        x = _outside_training_integer(rng, 31, 99)
        offset = _outside_training_integer(rng, 31, 99)
        return {
            "family": family,
            "prompt": f"Solve for x: x + {offset} = {x + offset}",
            "expected": _fraction_payload(Fraction(x, 1)),
            "template_id": "algebra_outside_operand_range",
        }

    if family == "word_problem":
        start, gain, lose = (
            rng.randint(100, 999),
            rng.randint(100, 999),
            rng.randint(100, 999),
        )
        item = rng.choice(("notebooks", "cookies", "marbles", "stickers"))
        return {
            "family": family,
            "prompt": (
                f"A student has {start} {item}. They get {gain} more and then give "
                f"away {lose}. How many {item} do they have now?"
            ),
            "expected": _fraction_payload(Fraction(start + gain - lose, 1)),
            "template_id": "word_problem_outside_operand_range",
        }

    raise ValueError(f"unknown promotion family: {family}")


def _build_prompt_pool() -> Dict[Tuple[str, int], List[Dict[str, object]]]:
    groups: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
    globally_seen: set[str] = set()
    target = SAMPLES_PER_FAMILY_PER_SEED * POOL_MULTIPLIER
    for family in FAMILIES:
        for seed in EVALUATION_SEEDS:
            rng = random.Random(seed + _FAMILY_OFFSETS[family])
            records: List[Dict[str, object]] = []
            attempts = 0
            while len(records) < target:
                attempts += 1
                if attempts > target * 100:
                    raise RuntimeError(f"could not build unique prompt pool for {family}/{seed}")
                record = _promotion_problem(family, rng)
                prompt = str(record["prompt"])
                if prompt in globally_seen:
                    continue
                globally_seen.add(prompt)
                records.append({**record, "generation_seed": seed, "attempt": attempts})
            groups[(family, seed)] = records
    return groups


def _legacy_seed65_prompts() -> Dict[str, str]:
    """Reproduce the old evaluator's deterministic 120-prompt development set."""

    rng = random.Random(LEGACY_DEVELOPMENT_SEED)
    prompts: Dict[str, str] = {}
    for index in range(LEGACY_DEVELOPMENT_COUNT):
        family = FAMILIES[index % len(FAMILIES)]
        if family == "arithmetic":
            a, b = rng.randint(100, 999), rng.randint(10, 999)
            op = rng.choice(("+", "-"))
            prompt = f"Solve this basic math problem: {a} {op} {b}"
        elif family == "percent":
            percent = rng.choice((5, 10, 12, 15, 20, 25))
            base = rng.randint(20, 2000)
            prompt = f"What is {percent}% of {base}?"
        elif family == "average":
            values = [rng.randint(5, 99) for _ in range(rng.choice((4, 5, 6)))]
            prompt = "Find the average (mean) of these numbers: " + ", ".join(
                str(value) for value in values
            )
        elif family == "algebra_one_step":
            x, offset = rng.randint(-30, 30), rng.randint(-30, 30)
            prompt = f"Solve for x: x + {offset} = {x + offset}"
        else:
            start, gain, lose = (
                rng.randint(20, 99),
                rng.randint(5, 60),
                rng.randint(5, 60),
            )
            item = rng.choice(("notebooks", "cookies", "marbles", "stickers"))
            prompt = (
                f"A student has {start} {item}. They get {gain} more and then give "
                f"away {lose}. How many {item} do they have now?"
            )
        if prompt in prompts:
            raise AssertionError("legacy seed-65 prompt generator produced a duplicate")
        prompts[prompt] = family
    return prompts


def _scan_corpus(
    path: Path,
    promotion_pool: Mapping[str, Mapping[str, object]],
    legacy_prompts: Mapping[str, str],
) -> Dict[str, object]:
    promotion_hits: set[str] = set()
    legacy_hits: set[str] = set()
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, Mapping) or not isinstance(record.get("user"), str):
                raise ValueError(f"missing string user prompt in {path}:{line_number}")
            prompt = str(record["user"])
            rows += 1
            if prompt in promotion_pool:
                promotion_hits.add(prompt)
            if prompt in legacy_prompts:
                legacy_hits.add(prompt)
    legacy_family_counts = Counter(legacy_prompts[prompt] for prompt in legacy_hits)
    return {
        "rows": rows,
        "promotion_pool_collision_prompts": sorted(promotion_hits),
        "promotion_pool_collision_count": len(promotion_hits),
        "legacy_seed65_unique_prompt_hits": len(legacy_hits),
        "legacy_seed65_family_hits": {
            family: int(legacy_family_counts.get(family, 0)) for family in FAMILIES
        },
    }


def _select_prompts(
    groups: Mapping[Tuple[str, int], Sequence[Mapping[str, object]]],
    rejected_prompts: set[str],
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for family in FAMILIES:
        for seed in EVALUATION_SEEDS:
            available = [
                dict(record)
                for record in groups[(family, seed)]
                if str(record["prompt"]) not in rejected_prompts
            ]
            if len(available) < SAMPLES_PER_FAMILY_PER_SEED:
                raise ValueError(
                    f"only {len(available)} collision-free prompts for {family}/{seed}; "
                    f"need {SAMPLES_PER_FAMILY_PER_SEED}"
                )
            for ordinal, record in enumerate(
                available[:SAMPLES_PER_FAMILY_PER_SEED], 1
            ):
                identity = {
                    "family": family,
                    "generation_seed": seed,
                    "prompt": record["prompt"],
                    "expected": record["expected"],
                    "template_id": record["template_id"],
                }
                record.pop("attempt", None)
                record["ordinal_within_seed"] = ordinal
                record["id"] = "v72-" + hashlib.sha256(
                    _canonical_json_bytes(identity)
                ).hexdigest()[:20]
                selected.append(record)
    return selected


def _chat_records() -> List[Dict[str, str]]:
    return [
        {"id": f"chat-{index + 1:02d}", "prompt": prompt}
        for index, prompt in enumerate(CHAT_PROMPTS)
    ]


def freeze_manifest(
    *,
    baseline_checkpoint: str | Path,
    baseline_corpus: str | Path,
    candidate_corpus: str | Path,
    candidate_checkpoint: str | Path,
    output: str | Path,
) -> Dict[str, object]:
    """Freeze a manifest while the expected final candidate is still absent."""

    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"frozen manifest already exists: {destination}")

    baseline_checkpoint_path = _required_file(baseline_checkpoint, "baseline checkpoint")
    baseline_corpus_path = _required_file(baseline_corpus, "baseline corpus")
    candidate_corpus_path = _required_file(candidate_corpus, "candidate corpus")
    candidate_checkpoint_path = Path(candidate_checkpoint).expanduser().resolve()
    if candidate_checkpoint_path.exists():
        raise FileExistsError(
            "candidate checkpoint already exists; the holdout must be frozen before "
            f"candidate scoring: {candidate_checkpoint_path}"
        )
    if candidate_checkpoint_path == baseline_checkpoint_path:
        raise ValueError("candidate checkpoint path must differ from the baseline")

    evaluator_path = Path(__file__).resolve()
    before_hashes = {
        "evaluator": sha256_file(evaluator_path),
        "baseline_checkpoint": sha256_file(baseline_checkpoint_path),
        "baseline_corpus": sha256_file(baseline_corpus_path),
        "candidate_corpus": sha256_file(candidate_corpus_path),
    }
    if before_hashes["baseline_checkpoint"] != V70_CHECKPOINT_SHA256:
        raise ValueError("baseline checkpoint does not match the immutable v70 identity")
    if before_hashes["baseline_corpus"] != V70_CORPUS_SHA256:
        raise ValueError("baseline corpus does not match the immutable v70 identity")

    groups = _build_prompt_pool()
    pool_by_prompt: Dict[str, Mapping[str, object]] = {
        str(record["prompt"]): record
        for records in groups.values()
        for record in records
    }
    legacy_prompts = _legacy_seed65_prompts()
    baseline_scan = _scan_corpus(
        baseline_corpus_path, pool_by_prompt, legacy_prompts
    )
    candidate_scan = _scan_corpus(
        candidate_corpus_path, pool_by_prompt, legacy_prompts
    )
    rejected = set(baseline_scan["promotion_pool_collision_prompts"]) | set(
        candidate_scan["promotion_pool_collision_prompts"]
    )
    prompts = _select_prompts(groups, rejected)
    selected_prompt_values = {str(record["prompt"]) for record in prompts}
    if selected_prompt_values & rejected:
        raise AssertionError("a colliding prompt survived selection")

    prompt_family_counts = Counter(str(record["family"]) for record in prompts)
    prompt_seed_counts = Counter(int(record["generation_seed"]) for record in prompts)
    chat = _chat_records()

    after_hashes = {
        "evaluator": sha256_file(evaluator_path),
        "baseline_checkpoint": sha256_file(baseline_checkpoint_path),
        "baseline_corpus": sha256_file(baseline_corpus_path),
        "candidate_corpus": sha256_file(candidate_corpus_path),
    }
    changed = [key for key in before_hashes if before_hashes[key] != after_hashes[key]]
    if changed:
        raise RuntimeError(f"artifact changed while freezing manifest: {changed}")
    if candidate_checkpoint_path.exists():
        raise FileExistsError(
            "candidate checkpoint appeared while freezing; refusing retrospective "
            f"preregistration: {candidate_checkpoint_path}"
        )

    frozen_at_unix_ns = time.time_ns()
    manifest: Dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "policy_id": POLICY_ID,
        "status": "frozen_unscored",
        "created_at_utc": _utc_now(),
        "frozen_at_unix_ns": frozen_at_unix_ns,
        "evaluator": {
            "path": str(evaluator_path),
            "sha256": before_hashes["evaluator"],
        },
        "protocol": production_protocol(),
        "thresholds": production_thresholds(),
        "comparator_lineage": comparator_lineage(),
        "baseline": {
            "checkpoint": str(baseline_checkpoint_path),
            "checkpoint_sha256": before_hashes["baseline_checkpoint"],
            "corpus": str(baseline_corpus_path),
            "corpus_sha256": before_hashes["baseline_corpus"],
            "corpus_scan": baseline_scan,
        },
        "candidate": {
            "checkpoint_expected": str(candidate_checkpoint_path),
            "checkpoint_present_at_freeze": False,
            "checkpoint_sha256_at_freeze": None,
            "corpus": str(candidate_corpus_path),
            "corpus_sha256": before_hashes["candidate_corpus"],
            "corpus_scan": candidate_scan,
        },
        "prompt_set": {
            "sha256": _records_sha256(prompts),
            "count": len(prompts),
            "family_counts": {
                family: int(prompt_family_counts.get(family, 0)) for family in FAMILIES
            },
            "seed_counts": {
                str(seed): int(prompt_seed_counts.get(seed, 0))
                for seed in EVALUATION_SEEDS
            },
            "pool_prompts": len(pool_by_prompt),
            "rejected_exact_collision_prompts": len(rejected),
            "records": prompts,
        },
        "chat_set": {
            "sha256": _records_sha256(chat),
            "count": len(chat),
            "records": chat,
        },
        "legacy_seed65_development_contamination": {
            "seed": LEGACY_DEVELOPMENT_SEED,
            "prompt_count": LEGACY_DEVELOPMENT_COUNT,
            "baseline_unique_prompt_hits": baseline_scan[
                "legacy_seed65_unique_prompt_hits"
            ],
            "candidate_unique_prompt_hits": candidate_scan[
                "legacy_seed65_unique_prompt_hits"
            ],
            "baseline_family_hits": baseline_scan["legacy_seed65_family_hits"],
            "candidate_family_hits": candidate_scan["legacy_seed65_family_hits"],
            "classification": "adaptive_contaminated_development_benchmark",
            "promotion_authority": False,
            "note": (
                "The repeatedly inspected seed-65 set contains exact training-prompt "
                "collisions and is excluded from this promotion decision."
            ),
        },
        "pointer_policy": {
            "mode": "review_only",
            "write_supported": False,
            "pointer_path": None,
            "pointer_written": False,
        },
    }
    if candidate_checkpoint_path.exists():
        raise FileExistsError("candidate checkpoint appeared before manifest commit")
    _atomic_write_json(destination, manifest)
    return manifest


def _load_json_mapping(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _manifest_records(manifest: Mapping[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    prompt_set = manifest.get("prompt_set")
    chat_set = manifest.get("chat_set")
    if not isinstance(prompt_set, Mapping) or not isinstance(chat_set, Mapping):
        raise ValueError("manifest is missing prompt_set/chat_set")
    raw_prompts = prompt_set.get("records")
    raw_chat = chat_set.get("records")
    if not isinstance(raw_prompts, list) or not all(
        isinstance(record, dict) for record in raw_prompts
    ):
        raise ValueError("manifest prompt records are invalid")
    if not isinstance(raw_chat, list) or not all(isinstance(record, dict) for record in raw_chat):
        raise ValueError("manifest chat records are invalid")
    prompts: List[Dict[str, object]] = [dict(record) for record in raw_prompts]
    chat: List[Dict[str, str]] = [
        {"id": str(record.get("id", "")), "prompt": str(record.get("prompt", ""))}
        for record in raw_chat
    ]
    return prompts, chat


def _validate_manifest(manifest: Mapping[str, object]) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported v72 manifest schema")
    if manifest.get("policy_id") != POLICY_ID or manifest.get("status") != "frozen_unscored":
        raise ValueError("manifest policy/status is not production frozen-unscored")
    if manifest.get("protocol") != production_protocol():
        raise ValueError("manifest protocol differs from the immutable production protocol")
    if manifest.get("thresholds") != production_thresholds():
        raise ValueError("manifest thresholds differ from the immutable production policy")
    if manifest.get("comparator_lineage") != comparator_lineage():
        raise ValueError("manifest comparator lineage differs from immutable identities")
    if manifest.get("pointer_policy") != {
        "mode": "review_only",
        "write_supported": False,
        "pointer_path": None,
        "pointer_written": False,
    }:
        raise ValueError("manifest pointer policy is not the non-writing review policy")

    evaluator = manifest.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ValueError("manifest evaluator binding is missing")
    current_evaluator = Path(__file__).resolve()
    if Path(str(evaluator.get("path", ""))).resolve() != current_evaluator:
        raise ValueError("manifest was frozen by a different evaluator path")
    if str(evaluator.get("sha256", "")) != sha256_file(current_evaluator):
        raise ValueError("evaluator source changed after manifest freeze")

    candidate = manifest.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("candidate binding is missing")
    if candidate.get("checkpoint_present_at_freeze") is not False:
        raise ValueError("manifest does not prove candidate absence at freeze")
    if candidate.get("checkpoint_sha256_at_freeze") is not None:
        raise ValueError("candidate hash was populated retrospectively at freeze")

    prompts, chat = _manifest_records(manifest)
    expected_count = len(FAMILIES) * len(EVALUATION_SEEDS) * SAMPLES_PER_FAMILY_PER_SEED
    if len(prompts) != expected_count:
        raise ValueError(f"manifest has {len(prompts)} math prompts, expected {expected_count}")
    ids = [str(record.get("id", "")) for record in prompts]
    texts = [str(record.get("prompt", "")) for record in prompts]
    if any(not value for value in ids + texts) or len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("prompt IDs/text must be non-empty and unique")
    family_seed_counts: Counter[Tuple[str, int]] = Counter()
    for record in prompts:
        family = str(record.get("family", ""))
        seed = record.get("generation_seed")
        expected = record.get("expected")
        if family not in FAMILIES or seed not in EVALUATION_SEEDS:
            raise ValueError("prompt family/seed is outside the frozen protocol")
        if not isinstance(expected, Mapping):
            raise ValueError("prompt expected value is missing")
        numerator, denominator = expected.get("numerator"), expected.get("denominator")
        if isinstance(numerator, bool) or not isinstance(numerator, int):
            raise ValueError("expected numerator must be an integer")
        if isinstance(denominator, bool) or not isinstance(denominator, int) or denominator <= 0:
            raise ValueError("expected denominator must be a positive integer")
        family_seed_counts[(family, int(seed))] += 1
    for family in FAMILIES:
        for seed in EVALUATION_SEEDS:
            if family_seed_counts[(family, seed)] != SAMPLES_PER_FAMILY_PER_SEED:
                raise ValueError(f"unbalanced prompt cell: {family}/{seed}")

    prompt_set = manifest["prompt_set"]
    assert isinstance(prompt_set, Mapping)
    if str(prompt_set.get("sha256", "")) != _records_sha256(prompts):
        raise ValueError("prompt-set digest mismatch")
    if int(prompt_set.get("count", -1)) != len(prompts):
        raise ValueError("prompt-set count mismatch")
    expected_family_counts = {
        family: len(EVALUATION_SEEDS) * SAMPLES_PER_FAMILY_PER_SEED
        for family in FAMILIES
    }
    if prompt_set.get("family_counts") != expected_family_counts:
        raise ValueError("prompt-set family counts mismatch")

    expected_chat = _chat_records()
    if chat != expected_chat:
        raise ValueError("chat retention set differs from the immutable protocol")
    chat_set = manifest["chat_set"]
    assert isinstance(chat_set, Mapping)
    if str(chat_set.get("sha256", "")) != _records_sha256(chat):
        raise ValueError("chat-set digest mismatch")
    return prompts, chat


def _legacy_scan_matches_manifest(
    manifest: Mapping[str, object], baseline_scan: Mapping[str, object], candidate_scan: Mapping[str, object]
) -> bool:
    disclosure = manifest.get("legacy_seed65_development_contamination")
    if not isinstance(disclosure, Mapping):
        return False
    return bool(
        disclosure.get("seed") == LEGACY_DEVELOPMENT_SEED
        and disclosure.get("prompt_count") == LEGACY_DEVELOPMENT_COUNT
        and disclosure.get("baseline_unique_prompt_hits")
        == baseline_scan.get("legacy_seed65_unique_prompt_hits")
        and disclosure.get("candidate_unique_prompt_hits")
        == candidate_scan.get("legacy_seed65_unique_prompt_hits")
        and disclosure.get("baseline_family_hits")
        == baseline_scan.get("legacy_seed65_family_hits")
        and disclosure.get("candidate_family_hits")
        == candidate_scan.get("legacy_seed65_family_hits")
        and disclosure.get("promotion_authority") is False
    )


def _parse_last_number(text: str) -> Optional[Fraction]:
    matches = _NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    raw = matches[-1].rstrip(".")
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(raw))
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None


def _is_correct(predicted: Optional[Fraction], expected: Fraction) -> bool:
    if predicted is None:
        return False
    difference = abs(predicted - expected)
    tolerance = max(Fraction(1, 1_000_000), abs(expected) / 1_000_000)
    return difference <= tolerance


def _normalised_words(text: str) -> List[str]:
    return _WORD_RE.findall(text.casefold())


def _token_f1(reference: str, candidate: str) -> float:
    reference_counts = Counter(_normalised_words(reference))
    candidate_counts = Counter(_normalised_words(candidate))
    if not reference_counts and not candidate_counts:
        return 1.0
    if not reference_counts or not candidate_counts:
        return 0.0
    overlap = sum((reference_counts & candidate_counts).values())
    precision = overlap / sum(candidate_counts.values())
    recall = overlap / sum(reference_counts.values())
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def _index_runner_rows(
    rows: object, expected_ids: Sequence[str], label: str
) -> Dict[str, Mapping[str, object]]:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"runner {label} rows are invalid")
    indexed: Dict[str, Mapping[str, object]] = {}
    for row in rows:
        identifier = str(row.get("id", ""))
        if not identifier or identifier in indexed:
            raise ValueError(f"runner {label} IDs are empty or duplicated")
        indexed[identifier] = row
    if set(indexed) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(indexed))
        extra = sorted(set(indexed) - set(expected_ids))
        raise ValueError(f"runner {label} identity mismatch; missing={missing}, extra={extra}")
    return indexed


def _coerce_generation_row(row: Mapping[str, object], cap: int) -> Dict[str, object]:
    reply = str(row.get("reply", ""))
    tokens = row.get("tokens")
    unknown_rate = row.get("prompt_unknown_rate")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0 or tokens > cap:
        raise ValueError("runner token count is invalid")
    if isinstance(unknown_rate, bool) or not isinstance(unknown_rate, (int, float)):
        raise ValueError("runner prompt unknown rate is invalid")
    unknown_rate_float = float(unknown_rate)
    if not math.isfinite(unknown_rate_float) or not 0.0 <= unknown_rate_float <= 1.0:
        raise ValueError("runner prompt unknown rate is out of range")
    return {
        "reply": reply,
        "tokens": tokens,
        "prompt_unknown_rate": unknown_rate_float,
        "generation_cap_hit": tokens >= cap,
        "error": str(row.get("error", "")),
    }


def _score_math(
    prompt_records: Sequence[Mapping[str, object]], raw_rows: object, cap: int
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    expected_ids = [str(record["id"]) for record in prompt_records]
    indexed = _index_runner_rows(raw_rows, expected_ids, "math")
    samples: List[Dict[str, object]] = []
    family_counts: Dict[str, Dict[str, int]] = {
        family: {"n": 0, "correct": 0} for family in FAMILIES
    }
    seed_counts: Dict[int, Dict[str, int]] = {
        seed: {"n": 0, "correct": 0} for seed in EVALUATION_SEEDS
    }
    unparsed = cap_hits = errors = correct_total = 0
    unknown_rates: List[float] = []
    for record in prompt_records:
        generation = _coerce_generation_row(indexed[str(record["id"])], cap)
        expected_payload = record["expected"]
        assert isinstance(expected_payload, Mapping)
        expected = Fraction(
            int(expected_payload["numerator"]), int(expected_payload["denominator"])
        )
        predicted = _parse_last_number(str(generation["reply"]))
        correct = _is_correct(predicted, expected) and not generation["error"]
        family = str(record["family"])
        seed = int(record["generation_seed"])
        family_counts[family]["n"] += 1
        family_counts[family]["correct"] += int(correct)
        seed_counts[seed]["n"] += 1
        seed_counts[seed]["correct"] += int(correct)
        correct_total += int(correct)
        unparsed += int(predicted is None)
        cap_hits += int(generation["generation_cap_hit"])
        errors += int(bool(generation["error"]))
        unknown_rates.append(float(generation["prompt_unknown_rate"]))
        samples.append(
            {
                "id": record["id"],
                "family": family,
                "generation_seed": seed,
                "prompt": record["prompt"],
                "expected": record["expected"],
                "predicted": (
                    _fraction_payload(predicted) if predicted is not None else None
                ),
                "correct": bool(correct),
                **generation,
            }
        )

    total = len(samples)
    per_family = {
        family: {
            **counts,
            "accuracy": counts["correct"] / max(1, counts["n"]),
        }
        for family, counts in family_counts.items()
    }
    per_seed = {
        str(seed): {
            **counts,
            "accuracy": counts["correct"] / max(1, counts["n"]),
        }
        for seed, counts in seed_counts.items()
    }
    metrics: Dict[str, object] = {
        "n": total,
        "correct": correct_total,
        "accuracy": correct_total / max(1, total),
        "unparsed": unparsed,
        "unparsed_rate": unparsed / max(1, total),
        "generation_cap_hits": cap_hits,
        "generation_cap_rate": cap_hits / max(1, total),
        "generation_errors": errors,
        "mean_prompt_unknown_rate": sum(unknown_rates) / max(1, len(unknown_rates)),
        "max_prompt_unknown_rate": max(unknown_rates, default=0.0),
        "per_family": per_family,
        "per_seed": per_seed,
    }
    return metrics, samples


def _score_chat(
    chat_records: Sequence[Mapping[str, object]], raw_rows: object, cap: int
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    expected_ids = [str(record["id"]) for record in chat_records]
    indexed = _index_runner_rows(raw_rows, expected_ids, "chat")
    samples: List[Dict[str, object]] = []
    for record in chat_records:
        generation = _coerce_generation_row(indexed[str(record["id"])], cap)
        word_count = len(_normalised_words(str(generation["reply"])))
        operational = bool(
            word_count >= 3
            and not generation["generation_cap_hit"]
            and not generation["error"]
        )
        samples.append(
            {
                "id": record["id"],
                "prompt": record["prompt"],
                "word_count": word_count,
                "operational": operational,
                **generation,
            }
        )
    metrics: Dict[str, object] = {
        "n": len(samples),
        "operational": sum(int(row["operational"]) for row in samples),
        "generation_cap_hits": sum(int(row["generation_cap_hit"]) for row in samples),
        "generation_errors": sum(int(bool(row["error"])) for row in samples),
        "max_prompt_unknown_rate": max(
            (float(row["prompt_unknown_rate"]) for row in samples), default=0.0
        ),
    }
    return metrics, samples


def _paired_evidence(
    baseline_samples: Sequence[Mapping[str, object]],
    candidate_samples: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    baseline = {str(row["id"]): row for row in baseline_samples}
    candidate = {str(row["id"]): row for row in candidate_samples}
    if set(baseline) != set(candidate):
        raise ValueError("paired sample identities differ")
    wins = regressions = ties = 0
    family_rows = {
        family: {"wins": 0, "regressions": 0, "ties": 0} for family in FAMILIES
    }
    seed_rows = {
        str(seed): {"wins": 0, "regressions": 0, "ties": 0}
        for seed in EVALUATION_SEEDS
    }
    for identifier in sorted(baseline):
        base_ok = bool(baseline[identifier]["correct"])
        candidate_ok = bool(candidate[identifier]["correct"])
        family = str(baseline[identifier]["family"])
        seed = str(baseline[identifier]["generation_seed"])
        if candidate_ok and not base_ok:
            key = "wins"
            wins += 1
        elif base_ok and not candidate_ok:
            key = "regressions"
            regressions += 1
        else:
            key = "ties"
            ties += 1
        family_rows[family][key] += 1
        seed_rows[seed][key] += 1
    discordant = wins + regressions
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = sum(
            math.comb(discordant, value) for value in range(wins, discordant + 1)
        ) / (2**discordant)
    return {
        "n": len(baseline),
        "wins": wins,
        "regressions": regressions,
        "ties": ties,
        "discordant": discordant,
        "exact_one_sided_sign_p_value": p_value,
        "per_family": family_rows,
        "per_seed": seed_rows,
    }


def _promotion_decision(
    baseline_math: Mapping[str, object],
    candidate_math: Mapping[str, object],
    baseline_chat: Mapping[str, object],
    candidate_chat: Mapping[str, object],
    chat_similarity: Mapping[str, object],
    paired: Mapping[str, object],
) -> Dict[str, object]:
    blockers: List[str] = []
    for side, metrics in (("baseline", baseline_math), ("candidate", candidate_math)):
        if int(metrics["generation_errors"]):
            blockers.append(f"{side}_generation_errors")
        if float(metrics["unparsed_rate"]) > MAX_UNPARSED_RATE:
            blockers.append(f"{side}_unparsed_rate_above_threshold")
        if float(metrics["generation_cap_rate"]) > MAX_GENERATION_CAP_RATE:
            blockers.append(f"{side}_generation_cap_rate_above_threshold")
        if float(metrics["max_prompt_unknown_rate"]) > MAX_PROMPT_UNKNOWN_RATE:
            blockers.append(f"{side}_prompt_unknown_rate_above_threshold")

    overall_gain = float(candidate_math["accuracy"]) - float(baseline_math["accuracy"])
    if overall_gain < MIN_OVERALL_ACCURACY_GAIN:
        blockers.append("overall_accuracy_gain_below_threshold")
    if float(candidate_math["accuracy"]) <= float(baseline_math["accuracy"]):
        blockers.append("candidate_not_strictly_better_overall")

    base_families = baseline_math["per_family"]
    candidate_families = candidate_math["per_family"]
    assert isinstance(base_families, Mapping) and isinstance(candidate_families, Mapping)
    family_deltas: Dict[str, float] = {}
    for family in FAMILIES:
        base = base_families[family]
        tuned = candidate_families[family]
        assert isinstance(base, Mapping) and isinstance(tuned, Mapping)
        delta = float(tuned["accuracy"]) - float(base["accuracy"])
        family_deltas[family] = delta
        if delta < -MAX_FAMILY_REGRESSION:
            blockers.append(f"family_regression:{family}")
    if not any(delta > 0.0 for delta in family_deltas.values()):
        blockers.append("no_strict_family_gain")

    base_seeds = baseline_math["per_seed"]
    candidate_seeds = candidate_math["per_seed"]
    assert isinstance(base_seeds, Mapping) and isinstance(candidate_seeds, Mapping)
    seed_deltas: Dict[str, float] = {}
    for seed in EVALUATION_SEEDS:
        key = str(seed)
        base = base_seeds[key]
        tuned = candidate_seeds[key]
        assert isinstance(base, Mapping) and isinstance(tuned, Mapping)
        delta = float(tuned["accuracy"]) - float(base["accuracy"])
        seed_deltas[key] = delta
        if delta < 0.0:
            blockers.append(f"seed_regression:{key}")

    if int(paired["wins"]) <= int(paired["regressions"]):
        blockers.append("paired_wins_not_above_regressions")
    if float(paired["exact_one_sided_sign_p_value"]) > MAX_PAIRED_ONE_SIDED_P_VALUE:
        blockers.append("paired_exact_p_value_above_threshold")

    for side, metrics in (("baseline", baseline_chat), ("candidate", candidate_chat)):
        if int(metrics["generation_errors"]):
            blockers.append(f"{side}_chat_generation_errors")
        if int(metrics["generation_cap_hits"]):
            blockers.append(f"{side}_chat_generation_cap_hits")
        if float(metrics["max_prompt_unknown_rate"]) > MAX_PROMPT_UNKNOWN_RATE:
            blockers.append(f"{side}_chat_prompt_unknown_rate_above_threshold")
        if int(metrics["operational"]) < MIN_CHAT_OPERATIONAL:
            blockers.append(f"{side}_chat_operational_below_threshold")
    if int(candidate_chat["operational"]) < int(baseline_chat["operational"]):
        blockers.append("chat_operational_regression")
    if int(chat_similarity["similar_prompts"]) < MIN_CHAT_SIMILAR:
        blockers.append("chat_similarity_below_threshold")

    return {
        "passed": not blockers,
        "blockers": blockers,
        "overall_accuracy_gain": overall_gain,
        "family_accuracy_deltas": family_deltas,
        "seed_accuracy_deltas": seed_deltas,
        "thresholds": production_thresholds(),
    }


def _run_checkpoint(
    checkpoint: Path,
    prompt_records: Sequence[Mapping[str, object]],
    chat_records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object],
) -> Dict[str, object]:
    """Production CPU runner, imported lazily so freeze and tests stay light."""

    import torch

    from train_mimomix_talk import generate_reply, load_talk_checkpoint

    torch.manual_seed(0)
    torch.set_num_threads(int(protocol["torch_num_threads"]))
    torch.use_deterministic_algorithms(True)
    model, tokenizer, payload = load_talk_checkpoint(checkpoint)
    model.eval()

    def generate(
        active_model: object,
        active_tokenizer: object,
        records: Sequence[Mapping[str, object]],
        cap: int,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for record in records:
            prompt = str(record["prompt"])
            try:
                result = generate_reply(
                    active_model,
                    active_tokenizer,
                    prompt,
                    max_new_tokens=cap,
                    speculative=False,
                )
                rows.append(
                    {
                        "id": record["id"],
                        "reply": str(result.get("reply", "")),
                        "tokens": int(result.get("tokens", 0)),
                        "prompt_unknown_rate": active_tokenizer.unknown_rate(prompt),
                        "error": "",
                    }
                )
            except Exception as exc:  # fail closed while preserving prompt identity
                rows.append(
                    {
                        "id": record["id"],
                        "reply": "",
                        "tokens": 0,
                        "prompt_unknown_rate": active_tokenizer.unknown_rate(prompt),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return rows

    result: Dict[str, object] = {
        "checkpoint_metadata": {
            "schema": str(payload.get("schema", "")),
            "config": payload.get("config", {}),
            "extra": payload.get("extra", {}),
            "tokenizer_size": tokenizer.vocab_size,
            "digit_tokens": tokenizer.digit_tokens,
        },
        "math": generate(
            model, tokenizer, prompt_records, int(protocol["math_max_new_tokens"])
        ),
        "chat": generate(
            model, tokenizer, chat_records, int(protocol["chat_max_new_tokens"])
        ),
    }
    del model
    gc.collect()
    return result


Runner = Callable[
    [Path, Sequence[Mapping[str, object]], Sequence[Mapping[str, object]], Mapping[str, object]],
    Mapping[str, object],
]


def evaluate_manifest(
    *,
    manifest_path: str | Path,
    candidate_checkpoint: str | Path,
    output: str | Path,
    no_write_pointer: bool,
    runner: Optional[Runner] = None,
) -> Dict[str, object]:
    """Evaluate the frozen pair and emit a review-only, content-bound receipt."""

    if not no_write_pointer:
        raise ValueError("v72 evaluation requires explicit --no-write-pointer review mode")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"promotion receipt already exists: {destination}")
    frozen_path = _required_file(manifest_path, "frozen manifest")
    manifest_sha256 = sha256_file(frozen_path)
    manifest = _load_json_mapping(frozen_path)
    prompt_records, chat_records = _validate_manifest(manifest)

    baseline_binding = manifest.get("baseline")
    candidate_binding = manifest.get("candidate")
    if not isinstance(baseline_binding, Mapping) or not isinstance(candidate_binding, Mapping):
        raise ValueError("manifest model bindings are invalid")
    baseline_checkpoint = _required_file(
        str(baseline_binding.get("checkpoint", "")), "baseline checkpoint"
    )
    baseline_corpus = _required_file(
        str(baseline_binding.get("corpus", "")), "baseline corpus"
    )
    candidate_corpus = _required_file(
        str(candidate_binding.get("corpus", "")), "candidate corpus"
    )
    candidate_checkpoint_path = _required_file(candidate_checkpoint, "candidate checkpoint")
    expected_candidate = Path(str(candidate_binding.get("checkpoint_expected", ""))).resolve()
    if candidate_checkpoint_path != expected_candidate:
        raise ValueError("candidate checkpoint path differs from the preregistered path")
    frozen_at = manifest.get("frozen_at_unix_ns")
    if isinstance(frozen_at, bool) or not isinstance(frozen_at, int) or frozen_at <= 0:
        raise ValueError("manifest freeze timestamp is invalid")
    if candidate_checkpoint_path.stat().st_mtime_ns < frozen_at:
        raise ValueError("candidate checkpoint predates the frozen manifest")
    if candidate_checkpoint_path.stat().st_mtime_ns <= frozen_path.stat().st_mtime_ns:
        raise ValueError("candidate checkpoint was not created after manifest commit")

    evaluator_path = Path(__file__).resolve()
    start_hashes = {
        "manifest": manifest_sha256,
        "evaluator": sha256_file(evaluator_path),
        "baseline_checkpoint": sha256_file(baseline_checkpoint),
        "candidate_checkpoint": sha256_file(candidate_checkpoint_path),
        "baseline_corpus": sha256_file(baseline_corpus),
        "candidate_corpus": sha256_file(candidate_corpus),
    }
    if start_hashes["baseline_checkpoint"] != str(
        baseline_binding.get("checkpoint_sha256", "")
    ):
        raise ValueError("baseline checkpoint changed after manifest freeze")
    if start_hashes["baseline_corpus"] != str(baseline_binding.get("corpus_sha256", "")):
        raise ValueError("baseline corpus changed after manifest freeze")
    if start_hashes["candidate_corpus"] != str(candidate_binding.get("corpus_sha256", "")):
        raise ValueError("candidate corpus changed after manifest freeze")
    if start_hashes["baseline_checkpoint"] == start_hashes["candidate_checkpoint"]:
        raise ValueError("candidate checkpoint is byte-identical to the baseline")

    prompt_map = {str(record["prompt"]): record for record in prompt_records}
    legacy_prompts = _legacy_seed65_prompts()
    baseline_scan = _scan_corpus(baseline_corpus, prompt_map, legacy_prompts)
    candidate_scan = _scan_corpus(candidate_corpus, prompt_map, legacy_prompts)
    if baseline_scan["promotion_pool_collision_count"] or candidate_scan[
        "promotion_pool_collision_count"
    ]:
        raise ValueError("a frozen promotion prompt collides with a training corpus")
    if not _legacy_scan_matches_manifest(manifest, baseline_scan, candidate_scan):
        raise ValueError("legacy seed-65 contamination disclosure no longer matches corpora")

    run = runner or _run_checkpoint
    protocol = production_protocol()
    baseline_raw = run(baseline_checkpoint, prompt_records, chat_records, protocol)
    candidate_raw = run(candidate_checkpoint_path, prompt_records, chat_records, protocol)
    if not isinstance(baseline_raw, Mapping) or not isinstance(candidate_raw, Mapping):
        raise ValueError("runner returned an invalid checkpoint result")

    baseline_math, baseline_math_samples = _score_math(
        prompt_records, baseline_raw.get("math"), MATH_MAX_NEW_TOKENS
    )
    candidate_math, candidate_math_samples = _score_math(
        prompt_records, candidate_raw.get("math"), MATH_MAX_NEW_TOKENS
    )
    baseline_chat, baseline_chat_samples = _score_chat(
        chat_records, baseline_raw.get("chat"), CHAT_MAX_NEW_TOKENS
    )
    candidate_chat, candidate_chat_samples = _score_chat(
        chat_records, candidate_raw.get("chat"), CHAT_MAX_NEW_TOKENS
    )

    base_chat_by_id = {str(row["id"]): row for row in baseline_chat_samples}
    candidate_chat_by_id = {str(row["id"]): row for row in candidate_chat_samples}
    similarity_rows: List[Dict[str, object]] = []
    for identifier in sorted(base_chat_by_id):
        score = _token_f1(
            str(base_chat_by_id[identifier]["reply"]),
            str(candidate_chat_by_id[identifier]["reply"]),
        )
        similarity_rows.append(
            {
                "id": identifier,
                "token_f1": score,
                "retained": score >= MIN_CHAT_TOKEN_F1,
            }
        )
    chat_similarity = {
        "threshold": MIN_CHAT_TOKEN_F1,
        "similar_prompts": sum(int(row["retained"]) for row in similarity_rows),
        "mean_token_f1": sum(float(row["token_f1"]) for row in similarity_rows)
        / max(1, len(similarity_rows)),
        "rows": similarity_rows,
    }
    paired = _paired_evidence(baseline_math_samples, candidate_math_samples)
    decision = _promotion_decision(
        baseline_math,
        candidate_math,
        baseline_chat,
        candidate_chat,
        chat_similarity,
        paired,
    )

    end_hashes = {
        "manifest": sha256_file(frozen_path),
        "evaluator": sha256_file(evaluator_path),
        "baseline_checkpoint": sha256_file(baseline_checkpoint),
        "candidate_checkpoint": sha256_file(candidate_checkpoint_path),
        "baseline_corpus": sha256_file(baseline_corpus),
        "candidate_corpus": sha256_file(candidate_corpus),
    }
    changed_during_evaluation = [
        key for key in start_hashes if start_hashes[key] != end_hashes[key]
    ]
    if changed_during_evaluation:
        blockers = list(decision["blockers"])
        blockers.extend(
            f"artifact_changed_during_evaluation:{key}"
            for key in changed_during_evaluation
        )
        decision = {**decision, "passed": False, "blockers": blockers}

    receipt: Dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "policy_id": POLICY_ID,
        "generated_at_utc": _utc_now(),
        "mode": "review_only_no_write_pointer",
        "manifest": {"path": str(frozen_path), "sha256": manifest_sha256},
        "protocol": protocol,
        "thresholds": production_thresholds(),
        "artifact_binding": {
            "evaluator_path": str(evaluator_path),
            "evaluator_sha256": start_hashes["evaluator"],
            "baseline_checkpoint": str(baseline_checkpoint),
            "baseline_checkpoint_sha256": start_hashes["baseline_checkpoint"],
            "candidate_checkpoint": str(candidate_checkpoint_path),
            "candidate_checkpoint_sha256": start_hashes["candidate_checkpoint"],
            "baseline_corpus": str(baseline_corpus),
            "baseline_corpus_sha256": start_hashes["baseline_corpus"],
            "candidate_corpus": str(candidate_corpus),
            "candidate_corpus_sha256": start_hashes["candidate_corpus"],
            "prompt_set_sha256": manifest["prompt_set"]["sha256"],  # type: ignore[index]
            "chat_set_sha256": manifest["chat_set"]["sha256"],  # type: ignore[index]
            "changed_during_evaluation": changed_during_evaluation,
        },
        "legacy_seed65_development_contamination": manifest[
            "legacy_seed65_development_contamination"
        ],
        "baseline": {
            "checkpoint_metadata": baseline_raw.get("checkpoint_metadata", {}),
            "math": baseline_math,
            "math_samples_sha256": _records_sha256(baseline_math_samples),
            "math_samples": baseline_math_samples,
            "chat": baseline_chat,
            "chat_samples_sha256": _records_sha256(baseline_chat_samples),
            "chat_samples": baseline_chat_samples,
        },
        "candidate": {
            "checkpoint_metadata": candidate_raw.get("checkpoint_metadata", {}),
            "math": candidate_math,
            "math_samples_sha256": _records_sha256(candidate_math_samples),
            "math_samples": candidate_math_samples,
            "chat": candidate_chat,
            "chat_samples_sha256": _records_sha256(candidate_chat_samples),
            "chat_samples": candidate_chat_samples,
        },
        "paired_evidence": paired,
        "chat_similarity": chat_similarity,
        "decision": decision,
        "passed": bool(decision["passed"]),
        "pointer": {
            "write_requested": False,
            "write_supported": False,
            "pointer_path": None,
            "pointer_written": False,
        },
        "non_claims": [
            "Passing proves improvement only on the five frozen arithmetic families "
            "and retention of the eight frozen chat probes.",
            "The numeric ranges are intentionally outside the old generators, so the "
            "gate measures bounded extrapolation rather than broad intelligence.",
            "Chat similarity to v70 preserves recognisable behaviour; it does not prove "
            "novel conversation because v70 chat is known to reproduce training text.",
        ],
    }
    _atomic_write_json(destination, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze", help="freeze prompts before candidate exists")
    freeze.add_argument("--baseline-checkpoint", required=True)
    freeze.add_argument("--baseline-corpus", required=True)
    freeze.add_argument("--candidate-corpus", required=True)
    freeze.add_argument("--candidate-checkpoint", required=True)
    freeze.add_argument("--output", required=True)

    evaluate = subparsers.add_parser("evaluate", help="review-gate the frozen pair")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--candidate-checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--no-write-pointer",
        action="store_true",
        help="required explicit acknowledgement; v72 contains no pointer-write path",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        manifest = freeze_manifest(
            baseline_checkpoint=args.baseline_checkpoint,
            baseline_corpus=args.baseline_corpus,
            candidate_corpus=args.candidate_corpus,
            candidate_checkpoint=args.candidate_checkpoint,
            output=args.output,
        )
        print(
            json.dumps(
                {
                    "schema": manifest["schema"],
                    "status": manifest["status"],
                    "prompt_set_sha256": manifest["prompt_set"]["sha256"],  # type: ignore[index]
                    "prompt_count": manifest["prompt_set"]["count"],  # type: ignore[index]
                    "legacy_seed65": manifest[
                        "legacy_seed65_development_contamination"
                    ],
                    "output": str(Path(args.output).resolve()),
                },
                indent=2,
            )
        )
        return 0
    receipt = evaluate_manifest(
        manifest_path=args.manifest,
        candidate_checkpoint=args.candidate_checkpoint,
        output=args.output,
        no_write_pointer=bool(args.no_write_pointer),
    )
    print(
        json.dumps(
            {
                "schema": receipt["schema"],
                "passed": receipt["passed"],
                "blockers": receipt["decision"]["blockers"],  # type: ignore[index]
                "output": str(Path(args.output).resolve()),
                "pointer_written": False,
            },
            indent=2,
        )
    )
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
