"""Original-row partitions for matched v87 data experiments.

Partition membership depends on the original corpus, never transformed wording.
Test novelty tiers are recomputed honestly against each arm's actual train text.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import mimomix_eval_splits as splits
from v87_reasoning import digest_json, group_id, parse_problem

SCHEMA = "supermix-v87-frozen-source-split-v1"


def _file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _valid_row(raw: bytes, index: int) -> dict:
    row = json.loads(raw)
    if not isinstance(row, dict) or not all(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("user", "assistant")
    ):
        raise ValueError(f"invalid frozen-split corpus row {index + 1}")
    return row


def source_group(row: dict) -> str:
    """Group target permutations/contrasts and other exact normalized pairs."""
    prompt, task = row["user"].strip(), row.get("task")
    if task not in ("average", "two_step"):
        if prompt.startswith("Find the average (mean) of these numbers:"):
            task = "average"
        elif prompt.startswith("What is ") and ", then " in prompt:
            task = "two_step"
    if task in ("average", "two_step"):
        return "semantic:" + group_id(parse_problem(prompt, task))
    return "pair:" + digest_json([prompt, row["assistant"].strip()])


def _source_groups(source: Path, indices: list[int] | None) -> tuple[list[str], str, int]:
    if indices is not None and (
        not isinstance(indices, list) or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("source row indices must be unique, increasing nonnegative integers")
    wanted = iter(indices) if indices is not None else None
    next_index = next(wanted, None) if wanted is not None else None
    groups, digest, row_count = [], hashlib.sha256(), 0
    with source.open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            row_count += 1
            if wanted is None or index == next_index:
                groups.append(source_group(_valid_row(raw, index)))
                if wanted is not None:
                    next_index = next(wanted, None)
    if next_index is not None:
        raise ValueError("source row index is outside the original corpus")
    return groups, digest.hexdigest(), row_count


def _assign(groups: list[str], seed: int, dev_fraction: float, test_fraction: float) -> str:
    if type(seed) is not int or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and 0 < value < 1
        for value in (dev_fraction, test_fraction)
    ) or dev_fraction + test_fraction >= 1:
        raise ValueError("invalid frozen split seed or fractions")
    ordered = sorted(set(groups), key=lambda group: (digest_json([seed, group]), group))
    dev_count = max(1, int(len(ordered) * dev_fraction))
    test_count = max(1, int(len(ordered) * test_fraction))
    if len(ordered) <= dev_count + test_count:
        raise ValueError("frozen split needs at least one train, dev, and test group")
    assigned = {group: "d" if index < dev_count else
                "e" if index < dev_count + test_count else "t"
                for index, group in enumerate(ordered)}
    return "".join(assigned[group] for group in groups)


def write_frozen_split(source_path: Path, corpus_path: Path, destination: Path, *,
                       seed: int = 58, dev_fraction: float = 0.01,
                       test_fraction: float = 0.04,
                       source_row_indices: list[int] | None = None,
                       expected_source_sha256: str | None = None) -> dict:
    """Write a compact hash-bound recipe; source indices refer to JSONL lines.

    Full-corpus arms omit indices. Rehearsals supply the original zero-based
    indices of retained rows in output order. The preparer owns verification of
    transformed text; this function binds its file bytes to original membership.
    """
    source, corpus, destination = Path(source_path).resolve(), Path(corpus_path), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    groups, source_hash, source_count = _source_groups(source, source_row_indices)
    if expected_source_sha256 is not None and source_hash != expected_source_sha256.lower():
        raise ValueError("original source hash changed before freezing the split")
    assignments = _assign(groups, seed, dev_fraction, test_fraction)
    digest, corpus_count = hashlib.sha256(), 0
    with corpus.open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            _valid_row(raw, index)
            corpus_count += 1
    if corpus_count != len(groups):
        raise ValueError("transformed corpus row count differs from retained source rows")
    receipt = {
        "schema": SCHEMA, "source": str(source), "source_sha256": source_hash,
        "source_rows": source_count, "source_row_indices": source_row_indices,
        "corpus_sha256": digest.hexdigest(), "corpus_rows": corpus_count,
        "seed": seed, "dev_fraction": dev_fraction, "test_fraction": test_fraction,
        "source_group_sequence_sha256": digest_json(groups),
        "partition_sha256": hashlib.sha256(assignments.encode()).hexdigest(),
        "partition_rows": dict(Counter(assignments)), "source_groups": len(set(groups)),
        "scope": "original source semantic/exact-pair groups; source row order",
        "tier_scope": "test response novelty is reclassified separately for each transformed arm",
    }
    destination.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def load_frozen_split(corpus_path: Path, receipt_path: Path, *,
                      min_response_characters: int = 1,
                      limit: int | None = None) -> tuple[splits.GeneralisationSplit, dict]:
    """Validate bytes and partitions before exposing rows to the trainer."""
    if limit is not None:
        raise ValueError("--pairs cannot filter a frozen split; prepare a separate rehearsal")
    receipt_path = Path(receipt_path)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("unsupported frozen split receipt")
    groups, source_hash, source_count = _source_groups(
        Path(receipt["source"]), receipt["source_row_indices"]
    )
    assignments = _assign(groups, receipt["seed"], receipt["dev_fraction"], receipt["test_fraction"])
    observed = {
        "source_sha256": source_hash, "source_rows": source_count,
        "corpus_rows": len(groups), "source_group_sequence_sha256": digest_json(groups),
        "partition_sha256": hashlib.sha256(assignments.encode()).hexdigest(),
        "partition_rows": dict(Counter(assignments)), "source_groups": len(set(groups)),
    }
    if any(receipt.get(key) != value for key, value in observed.items()):
        raise ValueError("frozen source membership or partition receipt mismatch")
    partitions: dict[str, list[tuple[str, str]]] = {"t": [], "d": [], "e": []}
    digest, count = hashlib.sha256(), 0
    with Path(corpus_path).open("rb") as handle:
        for index, raw in enumerate(handle):
            digest.update(raw)
            row = _valid_row(raw, index)
            if index >= len(assignments):
                raise ValueError("frozen corpus has extra rows")
            pair = row["user"].strip(), row["assistant"].strip()
            if len(pair[1]) < min_response_characters:
                raise ValueError("response filtering would change the frozen training population")
            partitions[assignments[index]].append(pair)
            count += 1
    if count != receipt["corpus_rows"] or digest.hexdigest() != receipt["corpus_sha256"]:
        raise ValueError("frozen corpus bytes or row count mismatch")
    train, dev, test = partitions["t"], partitions["d"], partitions["e"]
    train_responses = {answer for _, answer in train}
    train_sentences = set(splits.sentence_inventory(train))
    tier1, tier2, tier3, held_out = [], [], [], set()
    for pair in test:
        missing = set(splits.split_sentences(pair[1])) - train_sentences
        if pair[1] in train_responses:
            tier1.append(pair)
        elif missing:
            tier3.append(pair)
            held_out.update(missing)
        else:
            tier2.append(pair)
    split = splits.GeneralisationSplit(
        train=train, dev=dev, tier1_seen_response=tier1,
        tier2_unseen_response=tier2, tier3_unseen_sentence=tier3,
        held_out_sentences=sorted(held_out), source=str(corpus_path),
        settings={"scope": receipt["scope"], "seed": receipt["seed"],
                  "dev_fraction": receipt["dev_fraction"], "test_fraction": receipt["test_fraction"],
                  "tier_scope": receipt["tier_scope"],
                  "partition_sha256": receipt["partition_sha256"],
                  "source_group_sequence_sha256": receipt["source_group_sequence_sha256"]},
    )
    splits.verify_split(split)
    train_set, dev_set, test_set = set(train), set(dev), set(test)
    if train_set & dev_set or train_set & test_set or dev_set & test_set:
        raise ValueError("transformed rows collide across frozen partitions")
    provenance = {**receipt, "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
                  "receipt_path": str(receipt_path.resolve()),
                  "verified": True, "actual_train_rows": len(train),
                  "actual_dev_rows": len(dev), "actual_test_rows": len(test)}
    provenance.pop("source_row_indices", None)
    return split, provenance
