"""Bounded, task-balanced warm-start experiments using a checkpoint's vocabulary.

This is a fresh AdamW optimisation experiment, not crash recovery. Development
loss selects among the untouched baseline and trained snapshots. A selected
snapshot is only an experimental artifact; independent answer evaluation and
promotion remain separate. No active model pointer is written.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import torch

from mimomix_text import UNK, build_training_tensors
from train_mimomix_talk import evaluate, load_talk_checkpoint, save_talk_checkpoint


SCHEMA = "supermix-replay-experiment-v1"


def digest_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    staging = path.with_suffix(path.suffix + ".tmp")
    staging.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a record")
            user = row.get("user", row.get("prompt"))
            fields = {"task": row.get("task"), "user": user,
                      "response": row.get("response"), "group_id": row.get("group_id")}
            if any(not isinstance(value, str) or not value.strip() for value in fields.values()):
                raise ValueError(f"{path}:{line_number}: task, user, response, group_id must be nonempty strings")
            rows.append(fields)
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def validate_partitions(train: list[dict], dev: list[dict]) -> None:
    """Reject both semantic-group leakage and prompts relabelled to evade it."""
    groups = {row["group_id"] for row in train} & {row["group_id"] for row in dev}
    prompts = {row["user"].strip() for row in train} & {row["user"].strip() for row in dev}
    if groups or prompts:
        raise ValueError(f"train/dev overlap: {len(groups)} groups, {len(prompts)} prompts")
    if not {row["task"] for row in train}.issubset({row["task"] for row in dev}):
        raise ValueError("development must cover every training task")


def task_probabilities(rows: list[dict], specifications: list[str]) -> tuple[torch.Tensor, dict]:
    """Each task has equal mass unless overridden; row count never sets mass."""
    counts = Counter(row["task"] for row in rows)
    weights = dict.fromkeys(counts, 1.0)
    supplied = set()
    for item in specifications:
        task, separator, raw = item.partition("=")
        if not separator or task not in counts or task in supplied:
            raise ValueError(f"invalid or duplicate task weight: {item!r}")
        weight = float(raw)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("task weights must be finite and positive")
        weights[task] = weight
        supplied.add(task)
    # Scale before summing so valid large finite weights do not overflow.
    scale = max(weights.values())
    denominator = sum(value / scale for value in weights.values())
    masses = {task: value / scale / denominator for task, value in weights.items()}
    probabilities = torch.tensor([masses[row["task"]] / counts[row["task"]]
                                  for row in rows], dtype=torch.float64)
    if not bool(torch.isfinite(probabilities).all()) or bool((probabilities <= 0).any()):
        raise ValueError("task weights exceed the representable sampling range")
    return probabilities, masses


def encode_rows(rows: list[dict], tokenizer, sequence_length: int):
    maximum = 0
    tokens = Counter()
    for index, row in enumerate(rows):
        ids, prompt_length = tokenizer.encode_turn(row["user"], row["response"])
        if UNK in ids:
            raise ValueError(f"row {index} ({row['task']}) contains checkpoint-unknown tokens")
        if len(ids) > sequence_length:
            raise ValueError(f"row {index} ({row['task']}) needs {len(ids)} tokens; context is {sequence_length}")
        maximum = max(maximum, len(ids))
        tokens[row["task"]] += len(ids) - prompt_length
    x, y = build_training_tensors([(row["user"], row["response"]) for row in rows],
                                 tokenizer, sequence_length, turn_aligned=True)
    if x.shape[0] != len(rows):
        raise RuntimeError("complete-turn packing unexpectedly dropped rows")
    return x, y, {"rows": len(rows), "max_turn_tokens": maximum,
                  "rows_by_task": dict(Counter(row["task"] for row in rows)),
                  "supervised_tokens_by_task": dict(tokens), "dropped_rows": 0}


def validate_arguments(args) -> None:
    if min(args.steps, args.batch_size, args.eval_every, args.torch_threads) <= 0:
        raise ValueError("steps, batch-size, eval-every and torch-threads must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and nonnegative")
    if Path(args.output_dir).exists():
        raise ValueError("output directory already exists; choose a fresh experiment directory")


def run(args) -> dict:
    validate_arguments(args)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    paths = {name: Path(getattr(args, name)).resolve() for name in
             ("checkpoint", "train_jsonl", "dev_jsonl", "evaluation_manifest")}
    bindings = {name: {"path": str(path), "sha256": digest_file(path)} for name, path in paths.items()}
    source_dir = Path(__file__).resolve().parent
    bindings["code"] = {name: digest_file(source_dir / name) for name in
                        ("train_mimomix_replay.py", "train_mimomix_talk.py", "mimomix_text.py",
                         "mimomix_core.py", "mimomix_decoding.py")}
    train, dev = load_rows(paths["train_jsonl"]), load_rows(paths["dev_jsonl"])
    validate_partitions(train, dev)
    model, tokenizer, payload = load_talk_checkpoint(paths["checkpoint"])
    if tokenizer.to_dict() != payload["tokenizer"] or model.config.to_dict() != payload["config"]:
        raise ValueError("loading changed the checkpoint tokenizer or architecture")
    sequence_length = int(model.config.max_position_embeddings)
    train_x, train_y, train_stats = encode_rows(train, tokenizer, sequence_length)
    dev_x, dev_y, dev_stats = encode_rows(dev, tokenizer, sequence_length)
    probabilities, masses = task_probabilities(train, args.task_weight)
    for name, path in paths.items():
        if digest_file(path) != bindings[name]["sha256"]:
            raise ValueError(f"input changed during preparation: {name}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "running",
              "created_at": datetime.now(timezone.utc).isoformat(), "inputs": bindings,
              "tokenizer_sha256": digest_json(tokenizer.to_dict()),
              "config_sha256": digest_json(model.config.to_dict()),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "torch_threads": torch.get_num_threads()},
              "hyperparameters": {"steps": args.steps, "batch_size": args.batch_size,
                                  "lr": args.lr, "weight_decay": args.weight_decay,
                                  "seed": args.seed, "eval_every": args.eval_every,
                                  "sequence_length": sequence_length,
                                  "task_sampling_mass": masses, "optimiser": "fresh AdamW",
                                  "schedule": "constant", "gradient_clip": 1.0},
              "train": train_stats, "dev": dev_stats, "history": [],
              "selection": {"criterion": "development reply-token loss", "baseline_eligible": True},
              "heldout_evaluated": False, "promotion_authorized": False, "pointer_written": False,
              "limitations": ["The input group identifiers are supplied by the data preparation step.",
                              "Development loss selection does not establish answer-accuracy improvement.",
                              "The holdout file is bound by hash but never scored by this trainer.",
                              "Fresh optimisation state makes this a warm start, not exact crash recovery."]}
    write_json(output / "replay_run.json", report)
    started = time.perf_counter()
    try:
        baseline = evaluate(model, dev_x, dev_y, args.batch_size)
        if not math.isfinite(baseline["loss"]) or baseline["scored_tokens"] <= 0:
            raise ValueError("baseline development loss is invalid")
        best_loss, best_step = baseline["loss"], 0
        report["history"].append({"step": 0, "dev": baseline, "elapsed_seconds": round(time.perf_counter()-started, 2)})
        extra = {"schema": SCHEMA, "source_checkpoint": bindings["checkpoint"],
                 "corpus_jsonl": str(paths["train_jsonl"]), "input_bindings": bindings,
                 "run_name": output.name, "promotion_authorized": False, "pointer_written": False}
        save_talk_checkpoint(output / "selected.pt", model, tokenizer,
                             extra={**extra, "steps": 0, "best_step": 0, "baseline_selected": True})
        print(f"baseline development loss {best_loss:.6f}", flush=True)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        generator = torch.Generator().manual_seed(args.seed + 1)
        exposures, token_exposures = Counter(), Counter()
        running_loss, loss_steps = 0.0, 0
        for step in range(1, args.steps + 1):
            indices = torch.multinomial(probabilities, args.batch_size, replacement=True, generator=generator)
            x, y = train_x[indices].long(), train_y[indices].long()
            for offset, index in enumerate(indices.tolist()):
                task = train[index]["task"]
                exposures[task] += 1
                token_exposures[task] += int((y[offset, 1:] != -100).sum())
            model.train()
            result = model(x, labels=y)
            if not bool(torch.isfinite(result.loss)):
                raise ValueError(f"nonfinite training loss at step {step}")
            optimiser.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimiser.step()
            model.step_router_bias()
            running_loss += float(result.lm_loss.detach())
            loss_steps += 1
            if step % args.eval_every == 0 or step == args.steps:
                metrics = evaluate(model, dev_x, dev_y, args.batch_size)
                if not math.isfinite(metrics["loss"]):
                    raise ValueError(f"nonfinite development loss at step {step}")
                record = {"step": step, "train_lm_loss": running_loss/loss_steps,
                          "dev": metrics, "elapsed_seconds": round(time.perf_counter()-started, 2)}
                report["history"].append(record)
                if metrics["loss"] < best_loss:
                    best_loss, best_step = metrics["loss"], step
                    save_talk_checkpoint(output / "selected.pt", model, tokenizer,
                                         extra={**extra, "steps": step, "best_step": step,
                                                "baseline_selected": False})
                report["selection"].update(best_step=best_step, best_dev_loss=best_loss)
                report["sampled_rows_by_task"] = dict(exposures)
                report["sampled_supervised_tokens_by_task"] = dict(token_exposures)
                write_json(output / "replay_run.json", report)
                print(f"step {step}/{args.steps}, dev {metrics['loss']:.6f}, selected {best_step}, "
                      f"elapsed {record['elapsed_seconds']:.1f}s", flush=True)
                running_loss, loss_steps = 0.0, 0
        for name, path in paths.items():
            if digest_file(path) != bindings[name]["sha256"]:
                raise ValueError(f"bound input changed during training: {name}")
        for name, digest in bindings["code"].items():
            if digest_file(source_dir / name) != digest:
                raise ValueError(f"bound source changed during training: {name}")
        save_talk_checkpoint(output / "candidate.pt", model, tokenizer,
                             extra={**extra, "steps": args.steps, "best_step": best_step})
        report.update(status="completed", elapsed_seconds=round(time.perf_counter()-started, 2))
        report["selection"].update(best_step=best_step, best_dev_loss=best_loss,
                                   baseline_selected=best_step == 0)
        report["artifacts"] = {name: {"path": str(output/name), "sha256": digest_file(output/name)}
                               for name in ("candidate.pt", "selected.pt")}
        report["weights_changed"] = any(not torch.equal(value.cpu(), payload["state_dict"][name])
                                        for name, value in model.state_dict().items())
        write_json(output / "replay_run.json", report)
        return report
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(output / "replay_run.json", report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "train-jsonl", "dev-jsonl", "evaluation-manifest", "output-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=907)
    parser.add_argument("--task-weight", action="append", default=[])
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
