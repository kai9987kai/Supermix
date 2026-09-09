"""Freeze an algebra repair with original-training replay and separate evaluation.

Reconstructs the original sentence split before sampling. Numeric task groups
are deliberately coarse (task and operands); this may withhold extra rows but
prevents differently worded copies from crossing the continuation boundary.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def group_id(task: str, prompt: str) -> str:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    # Mean and product inputs commute; sorting also conservatively groups
    # other permutations within a task. Language uses exact input identity.
    payload = [task, sorted(numbers) if numbers else prompt]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def repair_algebra(prompt: str) -> str:
    match = re.fullmatch(r"Solve for x: x \+ (-?\d+) = (-?\d+)", prompt)
    if not match:
        raise ValueError(f"unsupported algebra prompt: {prompt!r}")
    constant, right = map(int, match.groups())
    # Same sign-resolved target measured in v87; no other task changes format.
    from build_scratchpad_math import _split_tens
    magnitude = abs(constant)
    rt, rr = _split_tens(right)
    mt, mr = _split_tens(magnitude)
    if constant >= 0:
        word, prep, op, high, low = "subtract", "from", "-", rt - mt, rr - mr
    else:
        word, prep, op, high, low = "add", "to", "+", rt + mt, rr + mr
    answer = right - constant
    if high + low != answer:
        raise ValueError("algebra decomposition invariant failed")
    return (f"{word} {magnitude} {prep} both sides, {rt} {op} {mt} = {high}, "
            f"{rr} {op} {mr} = {low}, total {answer}")


def prepare(corpus: Path, original_report: Path, checkpoint: Path, output: Path,
            seed: int = 907) -> dict:
    if output.exists():
        raise ValueError("preparation output already exists")
    import torch
    from mimomix_eval_splits import build_generalisation_split, split_sentences
    from mimomix_text import WordTokenizer, UNK
    from train_mimomix_generalisation import load_corpus_pairs
    from eval_problem_solving import extract_answer, GENERATORS

    source_hashes = {str(p.resolve()): digest(p) for p in (corpus, original_report, checkpoint)}
    report = json.loads(original_report.read_text(encoding="utf-8"))
    settings = report["split"]["settings"]
    pairs = load_corpus_pairs(str(corpus), min_response_characters=1)
    split = build_generalisation_split(pairs, **{k: settings[k] for k in
        ("dev_fraction", "test_fraction", "target_row_fraction", "max_row_fraction_per_sentence", "seed")})
    if (len(split.train) != report["split"]["train_pairs"]
            or len(split.dev) != report["split"]["dev_pairs"]
            or split.held_out_sentences != report["held_out_sentences"]):
        raise ValueError("original split cannot be reproduced")
    del pairs
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tokenizer = WordTokenizer.from_dict(payload["tokenizer"])
    context = int(payload["config"]["max_position_embeddings"])
    del payload
    task_for_pair = {}
    with corpus.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_for_pair[(row["user"], row["assistant"])] = row.get("task", "language")
    held_sentences = set(split.held_out_sentences)
    rejects = Counter()

    def convert(pair, *, training=False):
        prompt, response = pair
        task = task_for_pair[pair]
        if task == "algebra_one_step":
            response = repair_algebra(prompt)
        ids, _ = tokenizer.encode_turn(prompt, response)
        if UNK in ids or len(ids) > context:
            rejects["unknown_or_over_context"] += 1
            return None
        if training and held_sentences.intersection(split_sentences(response)):
            rejects["original_held_sentence"] += 1
            return None
        return {"task": task, "user": prompt, "response": response,
                "group_id": group_id(task, prompt)}

    # Select once using only source membership and labels, never model replies.
    reserved = set()
    def choose(rows, per_task, algebra, *, evaluation=False):
        unique = sorted(set(rows))
        random.Random(seed + int(evaluation)).shuffle(unique)
        chosen, counts = [], Counter()
        for pair in unique:
            task = task_for_pair[pair]
            if evaluation and task not in GENERATORS:
                continue
            cap = algebra if task == "algebra_one_step" else per_task
            if counts[task] >= cap:
                continue
            row = convert(pair)
            if row is None or row["group_id"] in reserved:
                continue
            if evaluation:
                expected = extract_answer(row["response"])
                if expected is None:
                    raise ValueError("evaluation target cannot be read")
                row["expected"] = expected
            reserved.add(row["group_id"])
            chosen.append(row)
            counts[task] += 1
        return chosen

    dev = choose(split.dev, 8, 32)
    test = choose(split.tier1_seen_response + split.tier2_unseen_response + split.tier3_unseen_sentence,
                  6, 40, evaluation=True)
    original_train_groups = {group_id(task_for_pair[p], p[0]) for p in set(split.train)}
    base_semantic_exposure = sum(r["group_id"] in original_train_groups for r in test)
    buckets = defaultdict(list)
    for pair in sorted(set(split.train)):
        task = task_for_pair[pair]
        gid = group_id(task, pair[0])
        if gid not in reserved:
            buckets[task].append(pair)
    train = []
    for task, bucket in sorted(buckets.items()):
        random.Random(f"{seed}:{task}").shuffle(bucket)
        selected_groups = set()
        cap = 4000 if task == "algebra_one_step" else 400
        for pair in bucket:
            if len(selected_groups) >= cap:
                break
            row = convert(pair, training=True)
            if row is None or row["group_id"] in selected_groups:
                continue
            selected_groups.add(row["group_id"])
            train.append(row)
    random.Random(seed).shuffle(train)
    if not train or not dev or not test:
        raise ValueError("empty partition")
    if any(digest(Path(p)) != value for p, value in source_hashes.items()):
        raise ValueError("source changed during preparation")
    output.mkdir(parents=True)
    for filename, rows in (("train.jsonl", train), ("dev.jsonl", dev)):
        (output / filename).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(test, indent=2) + "\n", encoding="utf-8")
    receipt = {"schema": "supermix-replay-preparation-v1", "seed": seed,
        "source_sha256": source_hashes,
        "original_split_reproduced": True,
        "files": {name: digest(output / name) for name in ("train.jsonl", "dev.jsonl", "evaluation.json")},
        "counts": {name: dict(Counter(r["task"] for r in rows)) for name, rows in
                   (("train", train), ("dev", dev), ("test", test))},
        "filtered": dict(rejects), "continuation_group_overlap": 0,
        "test_groups_exposed_to_original_training": base_semantic_exposure,
        "promotion_authorized": False,
        "limitations": "Continuation groups are disjoint. Original training may contain semantically related test inputs. This is a bounded template-distribution experiment, not an untouched general-intelligence benchmark.",
        "preparation_code_sha256": digest(Path(__file__))}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("corpus", "original-report", "checkpoint", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.original_report, args.checkpoint, args.output), indent=2))
