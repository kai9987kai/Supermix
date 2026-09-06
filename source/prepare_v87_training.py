"""Prepare an isolated, reproducible v87 curriculum and frozen evaluation set.

The input v86 corpus is immutable. Only average/two_step rows can change, and
all arms preserve row order, task exposure and operands. No trainer is launched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import tempfile

import build_scratchpad_math as scratch
from v87_reasoning import canonical_prompt, digest_json, group_id, parse_problem, render_working, verify_working

SCHEMA = "supermix-v87-training-preparation-v1"
TASKS_V86 = ("arithmetic", "percent", "average", "algebra_one_step", "word_problem",
             "multiplication", "division", "sequence", "two_step", "force", "acceleration",
             "momentum", "kinetic_energy", "work", "power", "voltage", "electrical_power",
             "wave_speed", "molarity", "combination", "arithmetic_series")
ARMS = ("control", "average", "two_step", "paraphrases", "combined")
SOURCE_FILES = ("prepare_v87_training.py", "v87_reasoning.py", "eval_prompt_robustness.py",
                "v87_frozen_split.py", "run_v87_training.py", "train_supervised.py",
                "build_scratchpad_math.py", "train_mimomix_generalisation.py", "train_mimomix_talk.py",
                "mimomix_text.py", "mimomix_core.py", "mimomix_eval_splits.py", "mimomix_decoding.py",
                "eval_problem_solving.py", "answer_check.py")
# These phrasings are reserved for evaluation, never used by the training builder.
EVAL_TEMPLATES = {
    "average": (
        "What mean do you get when averaging {csv}?",
        "Determine the arithmetic mean for this list: {csv}.",
        "Add {csv} together and divide the sum by how many numbers there are.",
    ),
    "two_step": (
        "Find {pct} percent of {base}. {sentence_op} {delta} to get the final value.",
        "Use {fraction} of {base} as the starting value and {natural_op} {delta}.",
        "The first value is {pct}% of {base}; {adjustment} that result by {delta}.",
    ),
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def row_case(row: dict) -> dict | None:
    task, prompt = row.get("task"), row["user"]
    if task not in ("average", "two_step"):
        if prompt.startswith("Find the average (mean) of these numbers:"):
            task = "average"
        elif prompt.startswith("What is ") and ", then " in prompt:
            task = "two_step"
        else:
            return None
    return parse_problem(prompt, task)


def training_prompt(case: dict, index: int, seed: int) -> tuple[str, str]:
    if case["task"] == "average":
        values = case["values"]
        fields = {"csv": ", ".join(map(str, values)), "natural": scratch._natural_join(values)}
    else:
        fields = {**case, "word": case["op"], "fraction": scratch.PERCENT_FRACTIONS[case["pct"]][0],
                  "natural_word": "add" if case["op"] == "add" else "take away",
                  "follow_word": "increase it by" if case["op"] == "add" else "reduce it by"}
    return scratch._paraphrase_prompt({"task": case["task"], "expression": canonical_prompt(case),
                                       "_prompt_fields": fields}, seed=seed, row_index=index)


def evaluation_prompts(case: dict) -> list[str]:
    fields = ({"csv": ", ".join(map(str, case["values"]))} if case["task"] == "average" else
              {**case, "fraction": scratch.PERCENT_FRACTIONS[case["pct"]][0],
               "sentence_op": "Add" if case["op"] == "add" else "Subtract",
               "natural_op": case["op"],
               "adjustment": "increase" if case["op"] == "add" else "decrease"})
    return [canonical_prompt(case)] + [t.format(**fields) for t in EVAL_TEMPLATES[case["task"]]]


def freeze_cases(occupied: set[str], per_family: int, seed: int) -> list[dict]:
    if per_family < 1:
        raise ValueError("per_family must be positive")
    rows, used = [], set(occupied)
    for task in ("average", "two_step"):
        rng = random.Random(int(digest_json([seed, task]), 16))
        count = 0
        for _ in range(per_family * 1000):
            if count >= per_family:
                break
            if task == "average":
                case = {"task": task, "values": [rng.randint(5, 99) for _ in range(rng.choice((4, 5, 6)))]}
            else:
                pct = rng.choice((10, 20, 25, 50))
                divisor = scratch.PERCENT_FRACTIONS[pct][1]
                base = rng.choice([v for v in range(40, 900) if v % divisor == 0])
                case = {"task": task, "pct": pct, "base": base, "delta": rng.randint(5, 60), "op": "add"}
            identity = group_id(case)
            if identity in used:
                continue
            used.add(identity)
            count += 1
            contrasts = [case] if task == "average" else [case, {**case, "op": "subtract"}]
            for contrast in contrasts:
                prompts = evaluation_prompts(contrast)
                case_id = digest_json(contrast)
                for variant, prompt in enumerate(prompts):
                    rows.append({"id": digest_json([case_id, variant]), "case_id": case_id,
                                 "group_id": identity, "case": contrast, "prompt": prompt,
                                 "variant": "canonical" if variant == 0 else f"{task}.eval.{variant}"})
        if count != per_family:
            raise ValueError(f"cannot find {per_family} unused {task} groups")
    return rows


def validate_frozen(rows: list[dict]) -> None:
    if not isinstance(rows, list) or not rows:
        raise ValueError("frozen evaluation is empty or invalid")
    seen, cases, groups = set(), {}, {}
    for row in rows:
        case = row["case"]
        # Reparse the canonical form to enforce the same bounded domain.
        if parse_problem(canonical_prompt(case), case["task"]) != case:
            raise ValueError("invalid frozen case")
        if row["case_id"] != digest_json(case) or row["group_id"] != group_id(case):
            raise ValueError("frozen case identity mismatch")
        variant = row["variant"]
        variants = ["canonical"] + [f"{case['task']}.eval.{i}" for i in range(1, 4)]
        if variant not in variants:
            raise ValueError("unknown frozen variant")
        index = variants.index(variant)
        if row["prompt"] != evaluation_prompts(case)[index]:
            raise ValueError("frozen prompt does not express its declared case")
        if row["id"] != digest_json([row["case_id"], index]) or row["id"] in seen:
            raise ValueError("invalid or duplicate frozen item")
        seen.add(row["id"])
        cases.setdefault(row["case_id"], set()).add(variant)
        groups.setdefault(row["group_id"], {})[row["case_id"]] = case
    if any(len(variants) != 4 for variants in cases.values()):
        raise ValueError("frozen evaluation has incomplete paraphrase groups")
    for group in groups.values():
        members = list(group.values())
        if members[0]["task"] == "average":
            if len(members) != 1:
                raise ValueError("duplicate average semantic group")
        elif len(members) != 2 or {c["op"] for c in members} != {"add", "subtract"}:
            raise ValueError("frozen evaluation has incomplete operation contrasts")


def prepare(source: Path, output_dir: Path, *, expected_source_sha256: str,
            arm: str = "combined", seed: int = 87, per_family: int = 100,
            limit_per_task: int = 0) -> dict:
    from mimomix_text import WordTokenizer
    from v87_frozen_split import write_frozen_split

    source, output_dir = source.resolve(), output_dir.resolve()
    initial_code = {name: sha256_file(Path(__file__).parent / name) for name in SOURCE_FILES}
    if arm not in ARMS or limit_per_task < 0:
        raise ValueError("invalid arm or limit_per_task")
    source_hash = sha256_file(source)
    if source_hash != expected_source_sha256.lower():
        raise ValueError("source corpus SHA256 mismatch")
    if output_dir.exists():
        raise FileExistsError(f"preparation directory already exists: {output_dir}")
    tokenizer = WordTokenizer([], digit_tokens=True)
    occupied: set[str] = set()
    source_counts: Counter = Counter()
    written_counts: Counter = Counter()
    changed_counts: Counter = Counter()
    max_turn: Counter = Counter()
    max_reply: Counter = Counter()
    rolling_source = hashlib.sha256()
    retained_indices = [] if limit_per_task else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v87-prepare-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        with source.open("rb") as handle, (stage / "train.jsonl").open("wb") as destination:
            for index, raw in enumerate(handle):
                rolling_source.update(raw)
                row = json.loads(raw)
                if not isinstance(row, dict) or not all(isinstance(row.get(k), str) and row[k].strip() for k in ("user", "assistant")):
                    raise ValueError(f"invalid training row at line {index+1}")
                label = row.get("task") or "unlabelled_inherited"
                source_counts[label] += 1
                case = row_case(row)
                if case:
                    occupied.add(group_id(case))
                if limit_per_task and written_counts[label] >= limit_per_task:
                    continue
                changed = dict(row)
                if case:
                    original = verify_working(case, row["assistant"])
                    if not original["process_correct"]:
                        raise ValueError(f"source reasoning row fails verification at line {index+1}")
                    if arm == "combined" or arm == case["task"]:
                        changed["assistant"] = render_working(case)
                    if arm in ("combined", "paraphrases"):
                        changed["user"], _ = training_prompt(case, index, seed)
                    if not verify_working(case, changed["assistant"])["process_correct"]:
                        raise ValueError(f"transformed reasoning row fails at line {index+1}")
                    turn = len(tokenizer.encode_turn(changed["user"], changed["assistant"])[0])
                    response = len(tokenizer.pattern.findall(changed["assistant"]))
                    max_turn[case["task"]] = max(max_turn[case["task"]], turn)
                    max_reply[case["task"]] = max(max_reply[case["task"]], response)
                    if turn > 128 or response > 96:
                        raise ValueError(f"reasoning row exceeds training/evaluation budget at line {index+1}")
                if changed != row:
                    changed_counts[label] += 1
                    destination.write((json.dumps(changed, ensure_ascii=True) + "\n").encode())
                else:
                    destination.write(raw if raw.endswith(b"\n") else raw + b"\n")
                written_counts[label] += 1
                if retained_indices is not None:
                    retained_indices.append(index)
        if rolling_source.hexdigest() != source_hash:
            raise ValueError("source corpus changed during preparation")
        if not all(source_counts[t] > 0 for t in ("average", "two_step")):
            raise ValueError("source corpus lacks required reasoning families")
        evaluation = freeze_cases(occupied, per_family, seed + 10000)
        validate_frozen(evaluation)
        if any(row["group_id"] in occupied for row in evaluation):
            raise ValueError("evaluation overlaps training semantic groups")
        (stage / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
        frozen_split = write_frozen_split(source, stage / "train.jsonl", stage / "frozen_split.json",
                                         seed=58, source_row_indices=retained_indices,
                                         expected_source_sha256=source_hash)
        trainer_args = ["--steps", "18000", "--run_name", f"v87_{arm}", "--output_dir", f"output/v87_{arm}",
                        "--corpus_jsonl", str(output_dir / "train.jsonl"), "--min_response_characters", "1",
                        "--frozen_split", str(output_dir / "frozen_split.json"), "--split_seed", "58", "--seed", str(seed),
                        "--digit_tokens", "--sequence_length", "128", "--max_vocab", "16384",
                        "--hidden_size", "256", "--n_layers", "4", "--n_heads", "8", "--n_kv_heads", "2",
                        "--n_routed_experts", "48", "--turn_aligned_packing", "--checkpoint_every_improvement",
                        "--eval_every", "500", "--accuracy_every", "3000", "--accuracy_problems", "420",
                        "--probe_max_new_tokens", "112", "--select_on", "accuracy", "--strict", "--torch_threads", "8"]
        for task in TASKS_V86:
            trainer_args.extend(("--accuracy_task", task))
        if initial_code != {name: sha256_file(Path(__file__).parent / name) for name in SOURCE_FILES}:
            raise ValueError("source implementation changed during preparation")
        manifest = {
            "schema": SCHEMA, "arm": arm, "seed": seed, "source": str(source), "source_sha256": source_hash,
            "source_rows_by_task": dict(source_counts), "rows_by_task": dict(written_counts),
            "changed_rows_by_task": dict(changed_counts), "rehearsal": bool(limit_per_task),
            "train_sha256": sha256_file(stage / "train.jsonl"),
            "evaluation_sha256": sha256_file(stage / "evaluation.json"),
            "frozen_split_sha256": sha256_file(stage / "frozen_split.json"),
            "partition_sha256": frozen_split["partition_sha256"],
            "partition_rows": frozen_split["partition_rows"],
            "evaluation_groups_per_family": per_family, "evaluation_items": len(evaluation),
            "semantic_group_overlap": 0, "source_semantic_groups": len(occupied),
            "target_budget": {"max_turn_tokens": dict(max_turn), "max_reply_tokens": dict(max_reply)},
            "source_code_sha256": initial_code,
            "trainer_args": trainer_args, "trainer_args_sha256": digest_json(trainer_args), "preparation_complete": True,
            "training_started": False, "promotion_authorized": False, "pointer_written": False,
            "acceptance_policy": {"average_accuracy_floor": 0.33, "two_step_accuracy_floor": 0.6333,
                                  "require_process_and_group_robustness_gain": True,
                                  "require_original_21_task_paired_nonregression": True},
            "limitations": ["This is a controlled two-family intervention, not evidence of broad intelligence.",
                            "Other rows are inherited unchanged; provenance/privacy of legacy unlabelled rows is not re-established.",
                            "Training membership and row order are paired to source identities, but tokenizer vocabulary and response novelty tiers may differ by arm.",
                            "Frozen evaluation excludes average permutations and both two-step operation contrasts from the complete source corpus.",
                            "The generator-based 21-task probe is development evidence and may overlap training.",
                            "Training and promotion need completed candidate-versus-control evidence; this manifest cannot activate a model."],
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        stage.rename(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected_source_sha256", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, default="combined")
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--per_family", type=int, default=100)
    parser.add_argument("--limit_per_task", type=int, default=0, help="rehearsal only; zero preserves the entire corpus")
    args = vars(parser.parse_args())
    result = prepare(**args)
    print(json.dumps({k: result[k] for k in ("arm", "rows_by_task", "changed_rows_by_task", "target_budget", "evaluation_items", "rehearsal", "training_started")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
