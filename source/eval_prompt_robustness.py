"""Evaluate raw prompt understanding and bounded working on a frozen v87 set.

Equivalent phrasings and operation contrasts are kept together for scoring.
This is behavioral evidence, never an authorization to select a runtime model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from prepare_v87_training import SCHEMA, sha256_file, validate_frozen
from v87_reasoning import digest_json, expected_answer, verify_working

REPORT_SCHEMA = "supermix-v87-prompt-robustness-v1"


def load_evaluation(bundle: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unsupported preparation manifest")
    if sha256_file(bundle / "evaluation.json") != manifest["evaluation_sha256"]:
        raise ValueError("frozen evaluation SHA256 mismatch")
    rows = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    validate_frozen(rows)
    counts = defaultdict(set)
    for row in rows:
        counts[row["case"]["task"]].add(row["group_id"])
    if (set(counts) != {"average", "two_step"} or len(rows) != manifest["evaluation_items"]
            or any(len(groups) != manifest["evaluation_groups_per_family"] for groups in counts.values())):
        raise ValueError("frozen evaluation coverage differs from manifest")
    return manifest, rows


def select_groups(rows: list[dict], limit: int) -> list[dict]:
    if limit < 0:
        raise ValueError("group limit must not be negative")
    if not limit:
        return rows
    chosen = defaultdict(set)
    selected = []
    for row in rows:
        task, identity = row["case"]["task"], row["group_id"]
        if identity in chosen[task] or len(chosen[task]) < limit:
            chosen[task].add(identity)
            selected.append(row)
    return selected


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot score an empty evaluation")
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["case"]["task"]].append(row)
    result = {}
    for task, items in by_task.items():
        groups = defaultdict(list)
        for row in items:
            groups[row["group_id"]].append(row)
        canonical = [r for r in items if r["variant"] == "canonical"]
        checked = sum(r["process"]["checked_steps"] for r in items)
        correct_steps = sum(r["process"]["correct_steps"] for r in items)
        group_results = {key: {
            "items": len(members),
            "all_answers_correct": all(r["correct"] for r in members),
            "all_processes_correct": all(r["process"]["process_correct"] for r in members),
            "canonical_correct": all(r["correct"] for r in members if r["variant"] == "canonical"),
        } for key, members in groups.items()}
        result[task] = {
            "items": len(items), "correct": sum(r["correct"] for r in items),
            "accuracy": sum(r["correct"] for r in items) / len(items),
            "canonical_accuracy": sum(r["correct"] for r in canonical) / len(canonical),
            "all_variant_group_accuracy": sum(g["all_answers_correct"] for g in group_results.values()) / len(groups),
            "process_accuracy": sum(r["process"]["process_correct"] for r in items) / len(items),
            "supported_processes": sum(r["process"]["supported"] for r in items),
            "checked_steps": checked, "correct_steps": correct_steps,
            "supported_step_accuracy": correct_steps / checked if checked else None,
            "hit_token_cap": sum(r["hit_token_cap"] for r in items),
            "groups": group_results,
        }
    return result


def compare_reports(candidate: dict, baseline: dict, *, draws: int = 2000) -> dict:
    for key in ("schema", "evaluation_sha256", "scoring_sha256", "settings", "complete"):
        if candidate.get(key) != baseline.get(key):
            raise ValueError(f"unpaired reports: {key} differs")
    if candidate.get("schema") != REPORT_SCHEMA:
        raise ValueError("unsupported evaluation report")
    # Recompute aggregates from the recorded replies rather than trusting summary fields.
    def checked_rows(report):
        rows = report["results"]
        validate_frozen(rows)
        import eval_problem_solving as solving
        for row in rows:
            process = verify_working(row["case"], row["reply"])
            correct = solving.is_correct(solving.extract_answer(row["reply"]), float(expected_answer(row["case"])))
            if row["process"] != process or row["correct"] != correct:
                raise ValueError("evaluation report has altered scoring")
        return rows
    candidate_rows, baseline_rows = checked_rows(candidate), checked_rows(baseline)
    if [r["id"] for r in candidate_rows] != [r["id"] for r in baseline_rows]:
        raise ValueError("unpaired reports: ordered evaluation items differ")
    left, right = summarize(candidate_rows), summarize(baseline_rows)
    result = {}
    for task in left:
        groups = left[task]["groups"]
        differences = [int(group["all_answers_correct"]) - int(right[task]["groups"][key]["all_answers_correct"])
                       for key, group in groups.items()]
        interval = None
        if len(differences) >= 20:
            rng = random.Random(87)
            samples = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences)
                             for _ in range(draws))
            interval = [samples[int(draws * .025)], samples[min(draws-1, int(draws * .975))]]
        result[task] = {"paired_groups": len(differences),
                        "all_variant_group_accuracy_delta": sum(differences) / len(differences),
                        "group_bootstrap_95_interval": interval,
                        "process_accuracy_delta": left[task]["process_accuracy"] - right[task]["process_accuracy"]}
    return {"by_task": result, "promotion_authorized": False,
            "note": "Groups, not correlated paraphrases, are the resampling unit. Small smoke sets have no interval; no multiplicity or selection correction is applied."}


def evaluate(bundle: Path, checkpoint: Path, *, max_new_tokens: int = 96,
             limit_groups: int = 0, torch_threads: int = 6) -> dict:
    import torch
    import eval_problem_solving as solving
    from mimomix_text import UNK
    from train_mimomix_talk import generate_reply, load_talk_checkpoint

    if not 1 <= max_new_tokens <= 256 or torch_threads < 1:
        raise ValueError("invalid token cap or thread count")
    scoring_files = ("eval_prompt_robustness.py", "prepare_v87_training.py", "v87_reasoning.py",
                     "eval_problem_solving.py", "train_mimomix_talk.py", "mimomix_decoding.py",
                     "mimomix_core.py", "mimomix_text.py")
    initial_code = {p: sha256_file(Path(__file__).parent / p) for p in scoring_files}
    manifest, frozen = load_evaluation(bundle)
    rows = select_groups(frozen, limit_groups)
    checkpoint_hash = sha256_file(checkpoint)
    torch.set_num_threads(torch_threads)
    model, tokenizer, _ = load_talk_checkpoint(checkpoint)
    results = []
    for index, row in enumerate(rows):
        prompt_ids, _ = tokenizer.encode_turn(row["prompt"], None)
        output = generate_reply(model, tokenizer, row["prompt"], max_new_tokens, speculative=False)
        reply, tokens = output["reply"], output["tokens"]
        results.append({**row, "reply": reply, "tokens": tokens,
                        "prompt_tokens": len(prompt_ids),
                        "unknown_prompt_tokens": sum(t == UNK for t in prompt_ids),
                        "beyond_native_context": len(prompt_ids) + tokens > model.config.native_context,
                        "hit_token_cap": tokens >= max_new_tokens,
                        "truncated": solving.is_truncated(reply, tokens, max_new_tokens),
                        "correct": solving.is_correct(solving.extract_answer(reply), float(expected_answer(row["case"]))),
                        "process": verify_working(row["case"], reply),
                        "latency_ms": output["latency_ms"]})
        print(f"{index+1}/{len(rows)} {row['case']['task']} {row['variant']}: correct={results[-1]['correct']}", flush=True)
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ValueError("checkpoint changed during evaluation")
    if initial_code != {p: sha256_file(Path(__file__).parent / p) for p in scoring_files}:
        raise ValueError("scoring implementation changed during evaluation")
    return {"schema": REPORT_SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
            "evaluation_sha256": manifest["evaluation_sha256"],
            "scoring_sha256": digest_json(initial_code),
            "settings": {"max_new_tokens": max_new_tokens, "normalisation": "none", "decoding": "greedy", "limit_groups": limit_groups},
            "native_context": model.config.native_context,
            "complete": len(rows) == len(frozen), "training_rehearsal_bundle": manifest["rehearsal"],
            "by_task": summarize(results), "results": results,
            "promotion_authorized": False, "pointer_written": False,
            "limitations": ["Supported text equations do not prove internal reasoning.",
                            "Two synthetic families do not establish general instruction-following.",
                            "Keep this frozen set out of training and checkpoint selection."]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit_groups", type=int, default=0)
    parser.add_argument("--torch_threads", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    report = evaluate(args.bundle, args.checkpoint, max_new_tokens=args.max_new_tokens,
                      limit_groups=args.limit_groups, torch_threads=args.torch_threads)
    if args.baseline:
        report["comparison"] = compare_reports(report, json.loads(args.baseline.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"complete": report["complete"], "promotion_authorized": False,
                      "accuracy": {t: s["accuracy"] for t, s in report["by_task"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
