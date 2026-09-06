"""Train the v57 talking MiMoMix against the v58 generalisation ladder.

`train_mimomix_talk.py` answered "can the v53 stack be trained to generate
text?" and the answer was yes. It left two things unmeasured, and said so:

1. Its held-out set is split by row, so **78.1% of validation responses appear
   verbatim in training** (1,875 of 2,400, measured). The receipt calls its own
   metric "fit to a template distribution", but no number separated recall from
   generalisation.
2. Its "does the thinking core contribute anything to text quality?" non-claim
   is explicit: *"No ablation has been run against a model without it on this
   corpus."*

This script closes both. It is additive: `mimomix_core.py`, `mimomix_text.py`
and `train_mimomix_talk.py` are unmodified, and the checkpoint it writes is the
same `supermix-v57-talk-checkpoint-v1` the existing chat interface already
loads.

## What changes

**The split.** `mimomix_eval_splits.build_generalisation_split` withholds a set
of whole sentences from training, then scores three tiers separately -- template
recall, sentence recombination, and unseen-sentence composition. Every word of
every tier is inside the training vocabulary, so tier 3 measures composition,
not vocabulary.

**Selection.** The best checkpoint is chosen on a **dev** split that is never
reported. `train_mimomix_talk.py` selects the minimum validation loss over
twelve evaluations and then reports that same validation set, which is a
minimum over evaluations of the thing being reported. It cost nothing in the
published run -- the loss fell monotonically, so the minimum was the last value
-- but nothing guaranteed that, and here it is simply not possible.

**Ablation.** `--no_thinking_core` was already a flag on the v57 trainer; it had
never been run as a matched pair. `--arm ablation` runs it against an otherwise
byte-identical configuration, seed and data.

Usage::

    python source/train_mimomix_generalisation.py --steps 2000 --arm full
    python source/train_mimomix_generalisation.py --steps 2000 --arm ablation
    python source/train_mimomix_generalisation.py --compare output/v58_full output/v58_ablation
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
import eval_problem_solving as solving  # noqa: E402
import recall_index  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_core import MiMoMixModel  # noqa: E402
from train_mimomix_talk import (  # noqa: E402
    DEFAULT_PROBE_MAX_NEW_TOKENS,
    PROBE_PROMPTS,
    atomic_json,
    build_config,
    build_parser as build_talk_parser,
    check_probe_token_budget,
    evaluate,
    generate_reply,
    parameter_groups,
    response_token_report,
    routing_report,
    save_talk_checkpoint,
    tokenizer_options,
)

RECEIPT_SCHEMA = "supermix-v58-generalisation-benchmark-v1"
COMPARISON_SCHEMA = "supermix-v58-thinking-core-ablation-v1"
SELECTION_STATE_SCHEMA = "supermix-v87-selection-state-v1"
SELECTION_STATE_KEYS = (
    "selection_state_schema",
    "select_on",
    "checkpoint_step",
    "best_score",
    "best_step",
    "best_dev_loss",
    "best_dev_seen",
    "best_probe_accuracy",
    "best_probe_verbatim_rate",
    "last_accuracy",
    "batch_generator_state",
    "torch_rng_state",
    "cuda_rng_states",
    "scaler_state",
    "selection_best_state",
    "accuracy_probe",
    "history",
)


def load_corpus_pairs(
    database: str,
    limit: Optional[int] = None,
    corpus_jsonl: Optional[str] = None,
    min_response_characters: int = 8,
) -> List[tuple]:
    """Read every usable `(user, assistant)` row, unsplit.

    `mimomix_text.load_chat_pairs` applies its own row split; here the whole
    corpus is needed first, because the tier-3 boundary has to be drawn over all
    rows before any of them are assigned to training.

    `corpus_jsonl` selects a JSONL corpus instead of the SQLite database. The
    default is unchanged, so every v58 command reproduces its published split.
    """

    # A receipt records one `source` string, and downstream tools (notably
    # `eval_mimomix_unseen_sentences`) rebuild the corpus by passing it back here
    # as `database`. Dispatching on the suffix as well as the explicit argument
    # is what makes that round-trip work for a JSONL corpus instead of handing a
    # text file to SQLite.
    selected = corpus_jsonl or database
    if str(selected).lower().endswith((".jsonl", ".json")):
        corpus = text_utils.load_chat_pairs_jsonl(
            selected,
            limit=limit,
            validation_fraction=0.02,
            seed=57,
            min_response_characters=min_response_characters,
        )
    else:
        corpus = text_utils.load_chat_pairs(
            selected,
            limit=limit,
            validation_fraction=0.02,
            seed=57,
            min_response_characters=min_response_characters,
        )
    return list(corpus.train) + list(corpus.validation)


def load_initial_weights(
    model: MiMoMixModel, tokenizer: text_utils.WordTokenizer, checkpoint: str
) -> Dict[str, Any]:
    """Continue training from an existing checkpoint instead of from scratch.

    A 2,000-step run on the v62 blend covers 0.178 of one epoch, so reaching a
    usable model means many thousands of steps; restarting from random weights
    each time throws away hours of finished compute for nothing.

    Continuation is only valid if the vocabulary is byte-identical. A different
    tokenizer means token id *n* denotes a different word, so the loaded
    embedding matrix would be silently wrong rather than merely suboptimal --
    the model would train, the loss would fall, and every number would be
    meaningless. That is checked here rather than trusted, and the mismatch
    raises.

    Returns provenance for the receipt: a checkpoint trained in two legs is not
    the same artifact as one trained in a single run, and the receipt should say
    so.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    source_tokens = list(payload.get("tokenizer", {}).get("tokens", []))
    if source_tokens != tokenizer.tokens:
        if len(source_tokens) != tokenizer.vocab_size:
            detail = (
                f"sizes differ: {len(source_tokens)} tokens against "
                f"{tokenizer.vocab_size}"
            )
        else:
            differing = [
                index
                for index, (a, b) in enumerate(zip(source_tokens, tokenizer.tokens))
                if a != b
            ]
            first = differing[0] if differing else None
            detail = (
                f"same size ({tokenizer.vocab_size}) but {len(differing)} ids denote "
                f"different words, first at id {first}: "
                f"{source_tokens[first]!r} against {tokenizer.tokens[first]!r}"
            )
        raise ValueError(
            f"{checkpoint} has a different vocabulary -- {detail}. Token ids would "
            "denote different words, so continuing from it would train on a "
            "silently corrupted embedding. This usually means the corpus, "
            "--pairs, --max_vocab or the split seed differs from the source run; "
            "match them exactly, or train fresh."
        )

    incompatible = {
        key: (tuple(value.shape), tuple(model.state_dict()[key].shape))
        for key, value in payload["state_dict"].items()
        if key in model.state_dict() and value.shape != model.state_dict()[key].shape
    }
    if incompatible:
        raise ValueError(
            f"{checkpoint} has {len(incompatible)} tensors whose shape differs from "
            f"this architecture, e.g. {next(iter(incompatible.items()))}"
        )

    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    extra = payload.get("extra") or {}
    return {
        "checkpoint": str(checkpoint),
        "source_run": extra.get("run_name"),
        "source_steps": extra.get("steps", extra.get("best_step")),
        # The length of the schedule the source trained on, which is what a
        # mid-curve resume must match. Absent on pre-v75 checkpoints, where the
        # caller falls back to the completed-step count as before.
        "source_total_steps": extra.get("total_steps"),
        "source_frozen_split": extra.get("frozen_split"),
        "source_dev_loss": extra.get("best_dev_loss"),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
        # Held for the caller to apply once the optimiser exists. Checkpoints
        # written before v63 have neither key, so continuing from one still
        # works and simply pays the re-warm cost it always did.
        "has_optimiser_state": "optimiser_state" in payload,
        "has_scheduler_state": "scheduler_state" in payload,
        "_optimiser_state": payload.get("optimiser_state"),
        "_scheduler_state": payload.get("scheduler_state"),
        # Kept private until resume validation; tensors cannot enter the JSON receipt.
        "_selection_state": {
            key: extra.get(key)
            for key in SELECTION_STATE_KEYS
            if extra.get(key) is not None
        },
        "note": (
            "weights continued from a prior run; this checkpoint was not trained "
            "in a single leg and its step count is this leg only"
        ),
    }


def restore_training_state(
    provenance: Optional[Dict[str, Any]],
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
) -> Dict[str, bool]:
    """Reapply AdamW moments and the LR schedule from a resumed checkpoint.

    Without this a continuation restarts the optimiser cold: v62's second leg
    watched dev loss climb from 0.8919 to 1.0036 and spent ~1,500 steps getting
    back to where it started. Restoring the state is what makes `--init_from` a
    genuine resume rather than a warm initialisation.
    """

    applied = {"optimiser": False, "scheduler": False}
    if not provenance:
        return applied

    optimiser_state = provenance.pop("_optimiser_state", None)
    scheduler_state = provenance.pop("_scheduler_state", None)

    # AdamW's state is keyed by the *position* of a parameter in the flattened
    # param_groups, not by name. `--decay_mode no_norm_bias` reorders that list
    # (all decayed tensors, then all undecayed ones), so restoring an
    # `all`-mode checkpoint into a `no_norm_bias` optimiser would attach every
    # moment to the wrong tensor and train something nobody could debug. The
    # group shape is therefore checked rather than assumed.
    if optimiser_state is not None:
        saved_shape = [len(g["params"]) for g in optimiser_state.get("param_groups", [])]
        live_shape = [len(g["params"]) for g in optimiser.state_dict()["param_groups"]]
        if saved_shape != live_shape:
            provenance["optimiser_skipped"] = (
                f"checkpoint has parameter groups of size {saved_shape} against "
                f"this run's {live_shape} -- almost certainly a different "
                "--decay_mode. AdamW state is positional, so restoring it would "
                "attach moments to the wrong tensors; the moments were dropped."
            )
            optimiser_state = None
            scheduler_state = None

    if optimiser_state is not None:
        if scheduler_state is not None:
            # Same schedule shape: the saved learning rates belong to the curve
            # this run will follow, so restore the optimiser wholesale.
            optimiser.load_state_dict(optimiser_state)
            scheduler.load_state_dict(scheduler_state)
            applied["optimiser"] = True
            applied["scheduler"] = True
        else:
            # Different schedule shape. Restore the AdamW *moments* only and
            # keep this run's freshly built `param_groups`.
            #
            # `load_state_dict` would also overwrite the group hyperparameters,
            # including the `initial_lr` anchors OneCycleLR computed at
            # construction. Doing that leaves the scheduler describing a curve
            # the optimiser is no longer on, which raises ZeroDivisionError
            # inside `get_lr` on the first step rather than merely training
            # oddly. The moments are the expensive thing to rebuild; the
            # learning rate should follow the new run's own schedule.
            merged = {
                "state": optimiser_state.get("state", {}),
                "param_groups": optimiser.state_dict()["param_groups"],
            }
            optimiser.load_state_dict(merged)
            applied["optimiser"] = True

    provenance["restored"] = applied
    return applied


def selection_state_payload(
    *,
    select_on: str,
    checkpoint_step: int,
    best_score: float,
    best_step: int,
    best_dev_loss: float,
    best_dev_seen: float,
    best_probe_accuracy: Optional[float],
    best_probe_verbatim_rate: Optional[float],
    last_accuracy: Optional[float],
    batch_generator: torch.Generator,
    history: Sequence[Mapping[str, Any]],
    best_state: Optional[Mapping[str, torch.Tensor]],
    accuracy_probe: Mapping[str, Any],
    scaler: Any = None,
) -> Dict[str, Any]:
    """State needed to make a crash resume equivalent to the interrupted leg."""

    return {
        "selection_state_schema": SELECTION_STATE_SCHEMA,
        "select_on": select_on,
        "checkpoint_step": int(checkpoint_step),
        "best_score": float(best_score),
        "best_step": int(best_step),
        "best_dev_loss": float(best_dev_loss),
        "best_dev_seen": float(best_dev_seen),
        "best_probe_accuracy": best_probe_accuracy,
        "best_probe_verbatim_rate": best_probe_verbatim_rate,
        "last_accuracy": last_accuracy,
        "batch_generator_state": batch_generator.get_state().cpu().clone(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_states": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else []
        ),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        # One atomic recovery file binds the old best weights to its score even
        # if a crash interrupts a subsequent selected/partial pair of writes.
        "selection_best_state": (
            {key: value.detach().cpu().clone() for key, value in best_state.items()}
            if best_state is not None else None
        ),
        "accuracy_probe": dict(accuracy_probe),
        "history": [dict(entry) for entry in history],
    }


def restore_resume_selection_state(
    provenance: Optional[Dict[str, Any]],
    *,
    start_step: int,
    select_on: str,
    batch_generator: torch.Generator,
    model: MiMoMixModel,
    accuracy_probe: Mapping[str, Any],
    scaler: Any = None,
) -> Dict[str, Any]:
    """Restore selection and sampling state only for a genuine crash resume.

    ``--init_from`` with ``--start_step 0`` is a warm start on a new curve. Its
    checkpoint's best score, batch stream and history belong to the old run and
    must not leak into the new selection decision.
    """

    cold = {
        "restored": False,
        "best_score": float("inf"),
        "best_step": 0,
        "best_dev_loss": float("inf"),
        "best_dev_seen": float("inf"),
        "best_probe_accuracy": None,
        "best_probe_verbatim_rate": None,
        "last_accuracy": None,
        "history": [],
        "best_state": None,
    }
    stored = provenance.pop("_selection_state", {}) if provenance is not None else {}
    if start_step <= 0:
        if provenance is not None and stored:
            provenance["selection_state_restore"] = {
                "restored": False,
                "reason": "warm_start",
            }
        return cold

    if stored.get("selection_state_schema") != SELECTION_STATE_SCHEMA:
        raise ValueError(
            "crash resume needs complete selection and RNG state; this legacy "
            "checkpoint supports a warm start with --start_step 0 instead"
        )

    if stored.get("select_on") != select_on:
        raise ValueError(
            "resume checkpoint selected on "
            f"{stored.get('select_on')!r}, not this run's {select_on!r}"
        )
    if int(stored.get("checkpoint_step", -1)) != int(start_step):
        raise ValueError(
            "resume selection state is bound to step "
            f"{stored.get('checkpoint_step')}, not --start_step {start_step}"
        )

    if stored.get("accuracy_probe") != dict(accuracy_probe):
        raise ValueError("resume accuracy probe differs in tasks, prompts, answers or token budget")

    history = stored.get("history")
    if not isinstance(history, list) or not all(isinstance(row, dict) for row in history):
        raise ValueError("resume selection state has invalid training history")

    restored = {
        "restored": True,
        "best_score": float(stored["best_score"]),
        "best_step": int(stored["best_step"]),
        "best_dev_loss": float(stored["best_dev_loss"]),
        "best_dev_seen": float(stored["best_dev_seen"]),
        "best_probe_accuracy": stored.get("best_probe_accuracy"),
        "best_probe_verbatim_rate": stored.get("best_probe_verbatim_rate"),
        "last_accuracy": stored.get("last_accuracy"),
        "history": [dict(row) for row in history],
    }
    best_step = restored["best_step"]
    if not 0 <= best_step <= start_step:
        raise ValueError("resume best_step is outside the completed training interval")
    for key in ("best_score", "best_dev_loss", "best_dev_seen"):
        value = restored[key]
        if math.isnan(value) or value == -float("inf") or (best_step and not math.isfinite(value)):
            raise ValueError(f"resume selection state has invalid {key}")
    for key in ("best_probe_accuracy", "best_probe_verbatim_rate", "last_accuracy"):
        value = restored[key]
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"resume selection state has invalid {key}")
    if best_step and select_on == "accuracy" and restored["best_probe_accuracy"] is None:
        raise ValueError("resume accuracy selection omits the selected checkpoint's measurement")
    if not history or history[-1].get("step") != start_step:
        raise ValueError("resume history does not end at the checkpoint step")
    restored["best_state"] = (
        validate_selected_weights(stored.get("selection_best_state"), model)
        if best_step else None
    )

    # Validate the RNG blobs before mutating either stream.
    states = []
    for name in ("batch_generator_state", "torch_rng_state"):
        state = stored.get(name)
        if not isinstance(state, torch.Tensor):
            raise ValueError(f"resume state omits {name}")
        torch.Generator().set_state(state.cpu())
        states.append(state.cpu())
    cuda_states = stored.get("cuda_rng_states", [])
    if cuda_states and (not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count()):
        raise ValueError("resume CUDA RNG state requires the original CUDA device count")
    scaler_state = stored.get("scaler_state")
    if (scaler is not None) != (scaler_state is not None):
        raise ValueError("resume AMP scaler configuration differs from the checkpoint")
    if scaler is not None:
        scaler.load_state_dict(scaler_state)
    batch_generator.set_state(states[0])
    torch.set_rng_state(states[1])
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    if provenance is not None:
        provenance["selection_state_restore"] = {
            "restored": True,
            "checkpoint_step": int(start_step),
            "best_step": restored["best_step"],
            "history_entries": len(history),
        }
    return restored


def validate_selected_weights(stored: Any, model: MiMoMixModel) -> Dict[str, torch.Tensor]:
    """Check the embedded selection-best weights before accepting their score."""
    current = model.state_dict()
    if not isinstance(stored, dict) or set(stored) != set(current):
        raise ValueError("resume checkpoint omits compatible selection-best weights")
    if not all(isinstance(value, torch.Tensor) for value in stored.values()):
        raise ValueError("resume selection-best weights contain non-tensors")
    incompatible = {
        key: (tuple(stored[key].shape), tuple(current[key].shape))
        for key in current
        if stored[key].shape != current[key].shape
    }
    if incompatible:
        raise ValueError(
            "resume checkpoint has incompatible selected weights, e.g. "
            f"{next(iter(incompatible.items()))}"
        )
    return {key: value.detach().cpu().clone() for key, value in stored.items()}


def save_progress_checkpoints(
    *,
    output_dir: Path,
    run_name: str,
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    extra: Mapping[str, Any],
    selection_improved: bool,
    dev_improved: bool,
    optimiser: torch.optim.Optimizer,
    scheduler: Any,
) -> None:
    """Persist independent selection-best and latest-recovery checkpoints."""

    if selection_improved:
        selected_extra = dict(extra)
        for key in ("selection_best_state", "batch_generator_state", "torch_rng_state", "cuda_rng_states", "scaler_state"):
            selected_extra.pop(key, None)
        selected_extra.update(
            {
                "written_because": "selection",
                "is_selection_best": True,
                "selection_checkpoint": True,
                "partial": False,
            }
        )
        save_talk_checkpoint(
            output_dir / f"{run_name}.selected.pt",
            model,
            tokenizer,
            extra=selected_extra,
        )

    if selection_improved or dev_improved:
        recovery_extra = dict(extra)
        recovery_extra.update(
            {
                "written_because": "selection" if selection_improved else "dev_loss",
                "is_selection_best": selection_improved,
                "selection_checkpoint": False,
                "partial": True,
                "note": (
                    "written mid-run for crash recovery; matching selection-best "
                    "weights are embedded in selection_best_state"
                ),
            }
        )
        save_talk_checkpoint(
            output_dir / f"{run_name}.partial.pt",
            model,
            tokenizer,
            extra=recovery_extra,
            optimiser=optimiser,
            scheduler=scheduler,
        )


#: Weight on the verbatim rate in the ``balanced`` criterion, in nats per unit of
#: verbatim fraction. 0.5 means a checkpoint reciting 20% more of its training
#: text must be 0.1 nats better on dev to be preferred -- roughly the size of the
#: whole v64 dev-loss improvement that came with 5.4x the recitation.
BALANCED_VERBATIM_WEIGHT = 0.5

#: Smallest accuracy probe that may drive checkpoint *selection*.
#:
#: A probe of n problems carries a binomial standard error of about
#: sqrt(p(1-p)/n). At n=20 and p=0.5 that is ±22 points at 95% confidence, and
#: v73 demonstrated it: its 20-problem probe read 0.15 at step 8,000 where a
#: 60-problem evaluation of the step-9,000 checkpoint read 0.467.
#:
#: That resolution is fine for *monitoring* -- 0.15 rising to 0.60 is a real
#: signal, and it is what makes an early abort possible. It is not fine for
#: choosing between two checkpoints a few points apart, which is what
#: `--select_on accuracy` does. Below this the trainer refuses rather than
#: selecting on noise.
MIN_SELECTION_PROBLEMS = 100


def probe_accuracy(
    model, tokenizer, problems, max_new_tokens: int = DEFAULT_PROBE_MAX_NEW_TOKENS
) -> Optional[float]:
    """Exact-match accuracy on freshly generated problems, mid-run.

    Every run in this line has cost twelve to seventeen hours and reported
    whether it worked only afterwards, because the loop tracked dev loss. v71
    finished with a *better* dev loss than v70 and 28 points less accuracy; v72
    finished with a worse loss and worse accuracy. Loss is not reliably related
    to the thing these models are for, in either direction.

    This gives the loop a number that is: a wrong answer to a freshly generated
    problem is wrong regardless of how probable the training text was. It is
    sampled small and infrequently -- generation is slow on CPU -- so it is a
    signal for aborting and for selection, not a benchmark.

    `max_new_tokens` was hardcoded to 64 until v82, which is CONFIRMED BUG E:
    measured with the v80 tokenizer over the whole v80 corpus, seven tasks have
    a *median* reply longer than that -- arithmetic_series 93, work 86,
    wave_speed 84, momentum 83, force 78, electrical_power 76, kinetic_energy
    76 -- and 100% of arithmetic_series replies exceed it. Those tasks read
    0.00 no matter what the model had learned, and `--select_on accuracy` was
    selecting against a signal that could not see them.
    """

    return probe_accuracy_report(model, tokenizer, problems, max_new_tokens)["accuracy"]


def probe_accuracy_report(
    model, tokenizer, problems, max_new_tokens: int = DEFAULT_PROBE_MAX_NEW_TOKENS
) -> Dict[str, Any]:
    """Measure the aggregate and task counts from the same generated replies."""

    was_training = model.training
    model.eval()
    correct = 0
    by_task: Dict[str, Dict[str, Any]] = {}
    try:
        for problem in problems:
            reply = generate_reply(model, tokenizer, problem.prompt, max_new_tokens)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            matched = int(solving.is_correct(solving.extract_answer(text), problem.answer))
            correct += matched
            counts = by_task.setdefault(problem.task, {"correct": 0, "total": 0})
            counts["correct"] += matched
            counts["total"] += 1
    finally:
        model.train(was_training)
    for counts in by_task.values():
        counts["accuracy"] = counts["correct"] / counts["total"]
    return {
        "accuracy": correct / len(problems) if problems else None,
        "correct": correct,
        "total": len(problems),
        "by_task": by_task,
    }


def accuracy_probe_manifest(problems, tasks, seed: int, max_new_tokens: int) -> Dict[str, Any]:
    """Bind the complete ordered exam, including answers, to its recorded settings."""

    records = [
        {"task": problem.task, "prompt": problem.prompt, "answer": problem.answer}
        for problem in problems
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "tasks": list(tasks),
        "seed": int(seed),
        "problems": len(problems),
        "max_new_tokens": int(max_new_tokens),
        "prompts_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "generator_fingerprint": solving.generator_fingerprint(tasks) if tasks else None,
    }


def task_labelled_rows(
    corpus_jsonl: Optional[str], samples: int = 6000
) -> Optional[List[Dict[str, Any]]]:
    """A cheap, deterministic, task-labelled sample of a JSONL corpus.

    The packed `(user, assistant)` pairs the trainer works with have lost the
    `task` field, and per-task is the only resolution at which the probe token
    budget means anything -- an aggregate median of 30 hides that
    arithmetic_series needs 93. So the file is sampled directly.

    Sampling is by evenly spaced byte offsets rather than by reading the file:
    the v80 corpus is 217 MB / 911,478 rows and JSON-parsing all of it at
    startup would cost more than the check is worth. Offsets are fixed, so two
    runs on the same file see the same rows.

    Returns None when there is no JSONL corpus (the SQLite path), in which case
    the caller falls back to the unlabelled pairs.
    """

    if not corpus_jsonl:
        return None
    path = Path(corpus_jsonl)
    if not path.exists() or not str(path).lower().endswith((".jsonl", ".json")):
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    rows: List[Dict[str, Any]] = []
    stride = max(1, size // max(1, samples))
    with path.open("rb") as handle:
        for offset in range(0, size, stride):
            handle.seek(offset)
            if offset:
                handle.readline()  # discard the partial line
            line = handle.readline()
            if not line:
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict) and record.get("assistant"):
                rows.append(record)
    return rows or None


def probe_verbatim_rate(model, tokenizer, recall) -> Optional[float]:
    """Mean fraction of probe replies that appear verbatim in training.

    Returns ``None`` when no corpus index was built, which is different from
    zero: "not measured" must not read as "nothing was recited".
    """

    if recall is None:
        return None
    was_training = model.training
    model.eval()
    try:
        rates = []
        for prompt in PROBE_PROMPTS:
            reply = generate_reply(model, tokenizer, prompt, 40)
            text = reply["reply"] if isinstance(reply, dict) else str(reply)
            report = recall.score(text)
            if report.windows:
                rates.append(report.verbatim_rate)
    finally:
        model.train(was_training)
    return sum(rates) / len(rates) if rates else None


def selection_score(
    criterion: str,
    dev_loss: float,
    verbatim: Optional[float],
    accuracy: Optional[float] = None,
) -> float:
    """Lower is better. The number a checkpoint is chosen on.

    V64 established that ``dev_loss`` -- the only criterion this trainer had --
    reliably selects the *most memorised* checkpoint available. Between step
    5,500 and step 10,000 of one run, dev loss improved from 1.0762 to 0.9910
    while the mean verbatim rate of generated replies rose from 0.14 to 0.76 and
    degenerate replies doubled. Verbatim reproduction of training text is the
    lowest-loss behaviour available, so perplexity does not merely fail to
    detect recitation, it prefers it.

    ``dev_loss`` remains the default so every published result reproduces. The
    other criteria exist so that a run which cares about generation quality can
    say so.
    """

    if criterion == "accuracy":
        # Negated so lower stays better for the loop's `<` comparison. Ties on
        # accuracy fall back to dev loss, scaled small enough that it can never
        # outweigh a whole percentage point of correctness.
        if accuracy is None:
            return float("inf")
        return -accuracy + min(dev_loss, 1.0) * 1e-3
    if criterion == "dev_loss" or verbatim is None:
        return dev_loss
    if criterion == "novelty":
        return verbatim
    if criterion == "balanced":
        return dev_loss + BALANCED_VERBATIM_WEIGHT * verbatim
    raise ValueError(f"unknown selection criterion {criterion!r}")


def score_tiers(
    model: MiMoMixModel,
    split: splits.GeneralisationSplit,
    tokenizer: text_utils.WordTokenizer,
    sequence_length: int,
    batch_size: int,
    turn_aligned: bool = False,
) -> Dict[str, Any]:
    """Score all three tiers once, after training and selection are finished."""

    scored: Dict[str, Any] = {}
    for name, rows in split.tiers():
        if not rows:
            scored[name] = {"pairs": 0, "measures": split.TIER_MEANINGS[name], "skipped": True}
            continue
        # Tiers are packed the same way training was, or the loss would be
        # measured over a different token population than the model was fitted
        # to and the tier numbers would not be comparable to the dev curve.
        inputs, labels = text_utils.build_training_tensors(
            rows, tokenizer, sequence_length, turn_aligned=turn_aligned
        )
        metrics = evaluate(model, inputs, labels, batch_size)
        scored[name] = {
            "pairs": len(rows),
            "measures": split.TIER_MEANINGS[name],
            **metrics,
        }
    return scored


def generalisation_gap(scored: Dict[str, Any]) -> Dict[str, Any]:
    """The differences between tiers, which are the point of measuring three.

    Reported in nats of loss rather than as a perplexity ratio, because the
    losses are small and a ratio of numbers near 1.27 reads as a smaller effect
    than it is.
    """

    def loss(name: str) -> Optional[float]:
        entry = scored.get(name) or {}
        return entry.get("loss")

    tier1, tier2, tier3 = (loss(n) for n in splits.GeneralisationSplit.TIER_MEANINGS)
    gaps: Dict[str, Any] = {}
    if tier1 is not None and tier2 is not None:
        gaps["recombination_cost_nats"] = round(tier2 - tier1, 6)
    if tier2 is not None and tier3 is not None:
        gaps["unseen_sentence_cost_nats"] = round(tier3 - tier2, 6)
    if tier1 is not None and tier3 is not None:
        gaps["total_cost_nats"] = round(tier3 - tier1, 6)
        gaps["perplexity_ratio_tier3_over_tier1"] = round(math.exp(tier3 - tier1), 4)
    gaps["note"] = (
        "a gap near zero means the tier names describe the same difficulty for this "
        "model; a large tier3 gap means the reported v57 perplexity was measuring "
        "recall of sentences rather than composition of them"
    )
    return gaps


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_resume_settings(args)
    validate_selection_settings(args)
    torch.manual_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))
    device, device_info = resolve_device(args.device, preference=args.device_preference)

    corpus_jsonl = getattr(args, "corpus_jsonl", None)
    frozen_split = None
    if getattr(args, "frozen_split", None):
        from v87_frozen_split import load_frozen_split

        if not corpus_jsonl or not getattr(args, "turn_aligned_packing", False):
            raise ValueError("--frozen_split requires --corpus_jsonl and --turn_aligned_packing")
        split, frozen_split = load_frozen_split(
            corpus_jsonl, args.frozen_split, limit=args.pairs,
            min_response_characters=getattr(args, "min_response_characters", 8),
        )
        if args.split_seed != frozen_split["seed"]:
            raise ValueError("--split_seed differs from the frozen split receipt")
    else:
        pairs = load_corpus_pairs(
            args.database,
            limit=args.pairs,
            corpus_jsonl=corpus_jsonl,
            min_response_characters=getattr(args, "min_response_characters", 8),
        )
        split = splits.build_generalisation_split(
            pairs,
            dev_fraction=args.dev_fraction,
            test_fraction=args.test_fraction,
            target_row_fraction=args.tier3_row_fraction,
            max_row_fraction_per_sentence=args.max_row_fraction_per_sentence,
            seed=args.split_seed,
            source=corpus_jsonl or args.database,
        )
    verification = splits.verify_split(split)

    # Vocabulary from the training rows only. Building it over the whole corpus
    # would leak the held-out sentences' surface forms into the model's
    # expressible language and quietly make tier 3 easier.
    tokenizer = text_utils.WordTokenizer.build(
        (field for pair in split.train for field in pair),
        max_vocab=args.max_vocab,
        **tokenizer_options(args),
    )
    text_utils.assert_roundtrip(tokenizer, [a for _, a in split.dev[:200]])

    # CONFIRMED BUG E's guard. Run before a single step is taken, because the
    # whole point is that a truncated probe looks like a model that failed to
    # learn, and the two are indistinguishable seventeen hours later.
    probe_cap = int(getattr(args, "probe_max_new_tokens", DEFAULT_PROBE_MAX_NEW_TOKENS))
    token_budget = check_probe_token_budget(
        response_token_report(
            task_labelled_rows(corpus_jsonl) or split.train, tokenizer
        ),
        probe_cap,
        strict=bool(getattr(args, "strict", False)),
    )

    turn_aligned = getattr(args, "turn_aligned_packing", False)
    train_x, train_y = text_utils.build_training_tensors(
        split.train, tokenizer, args.sequence_length, turn_aligned=turn_aligned
    )
    dev_x, dev_y = text_utils.build_training_tensors(
        split.dev, tokenizer, args.sequence_length, turn_aligned=turn_aligned
    )

    # `--arm ablation` disables the thinking core; an explicit
    # `--no_thinking_core` must not be silently *re-enabled* by `--arm full`,
    # which is what the old straight assignment did. Both now mean off.
    args.no_thinking_core = bool(args.no_thinking_core) or args.arm == "ablation"
    config = build_config(args, tokenizer.vocab_size)
    model = MiMoMixModel(config).to(device)
    parameters = model.parameter_report()

    initialised_from = getattr(args, "init_from", None)
    init_provenance: Optional[Dict[str, Any]] = None
    if initialised_from:
        init_provenance = load_initial_weights(model, tokenizer, initialised_from)
        if args.start_step > 0 and frozen_split != init_provenance.get("source_frozen_split"):
            raise ValueError("crash recovery requires the identical frozen split and corpus receipt")
        print(f"  init_from    {initialised_from} "
              f"({init_provenance['source_steps']} prior steps)")

    print(f"v58 generalisation | arm {args.arm} | thinking core {not args.no_thinking_core}")
    print(f"  train        {len(split.train):,} rows, dev {len(split.dev):,}")
    for name, rows in split.tiers():
        print(f"  {name:<24} {len(rows):,} rows")
    print(f"  withheld     {len(split.held_out_sentences)} sentences "
          f"({verification['distinct_training_sentences']} remain in training)")
    print(f"  vocabulary   {tokenizer.vocab_size} types")
    print(f"  parameters   {parameters['total']:,} total / {parameters['active_per_token']:,} active")
    print(f"  device       {device_info.get('resolved', device)}", flush=True)

    # Resolved before the loop so mid-run checkpoints have somewhere to go.
    output_dir = Path(args.output_dir)

    decay_groups = parameter_groups(model, args.weight_decay, args.decay_mode)
    optimiser = torch.optim.AdamW(
        [{k: v for k, v in g.items() if not k.startswith("_")} for g in decay_groups],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # OneCycleLR splits training into a warmup phase of `pct_start * total_steps`
    # and a decay phase. When that product is <= 1 the two phase boundaries
    # coincide and `get_lr` divides by zero on the very first step, so every run
    # of <= 1/pct_start steps crashed -- at the default 0.1, anything up to 10
    # steps. That is exactly the range a smoke test uses, which is why this
    # trainer was awkward to exercise cheaply.
    #
    # Widening pct_start for tiny runs keeps the curve well-formed. It cannot
    # affect a real run: any `steps` above 1/pct_start leaves the value untouched.
    total_steps = max(1, args.steps)
    if args.pct_start * total_steps > 1.0:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimiser, max_lr=args.lr, total_steps=total_steps, pct_start=args.pct_start
        )
    else:
        # No warmup phase can exist: `pct_start * total_steps <= 1` collapses the
        # two OneCycle phases onto the same boundary. Nudging pct_start does not
        # save it either -- at two steps the largest legal value still lands on
        # exactly 1.0 -- so a run this short gets a flat learning rate instead of
        # a degenerate curve. Only smoke tests reach this branch; at the default
        # pct_start=0.1 it is runs of ten steps or fewer.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lambda _: 1.0)

    # OneCycleLR is defined over `total_steps`, so restoring a schedule from a
    # run with a different `--steps` would resume at the wrong point on a
    # differently shaped curve. Only the optimiser moments are safe to carry in
    # that case, and that is decided here rather than silently.
    if init_provenance is not None:
        # Prefer the schedule length when the source recorded it. Falling back
        # to completed steps is what pre-v75 checkpoints allow, and for those a
        # mid-run file legitimately reads as a different curve.
        source_steps = init_provenance.get("source_total_steps")
        if source_steps is None:
            source_steps = init_provenance.get("source_steps")
        if args.start_step == 0:
            init_provenance["_scheduler_state"] = None
            init_provenance["scheduler_skipped"] = "warm start begins a fresh learning-rate curve"
        elif source_steps is not None and int(source_steps) != int(args.steps):
            init_provenance["_scheduler_state"] = None
            init_provenance["scheduler_skipped"] = (
                f"source ran {source_steps} steps against this run's {args.steps}; "
                "the OneCycle curve differs, so only optimiser moments were restored"
            )
        restored = restore_training_state(init_provenance, optimiser, scheduler)
        print(f"  restored     optimiser={restored['optimiser']} scheduler={restored['scheduler']}")
        source_step = init_provenance.get("source_steps")
        if args.start_step > 0 and source_step is not None and int(source_step) != args.start_step:
            # A restored scheduler resumes at the step the *source* reached, not
            # at whatever `--start_step` says. If the two disagree the loop and
            # the schedule count different steps: too few and the run stops
            # before the curve anneals, too many and OneCycleLR raises partway
            # through. Caught here, where it is one line, rather than hours in.
            raise SystemExit(
                f"--start_step {args.start_step} does not match the checkpoint, which "
                f"holds step {source_step}. The restored schedule resumes at the "
                "checkpoint's step, so the two must agree; pass "
                f"--start_step {source_step}."
            )
        if args.start_step > 0 and not restored["scheduler"]:
            # Resuming mid-curve without the schedule means the learning rate
            # warms up again from the start of a fresh OneCycle while the loop
            # runs only the tail of the old one -- the run would anneal on the
            # wrong part of the curve and never reach the low final LR that
            # makes the last steps worth taking. Refuse rather than train it.
            raise SystemExit(
                f"--start_step {args.start_step} needs the source schedule, but it "
                f"was not restored ({init_provenance.get('scheduler_skipped') or 'no scheduler_state in checkpoint'}). "
                "Pass the same --steps the crashed leg used, or drop --start_step "
                "to warm-start on a fresh curve instead."
            )

    generator = torch.Generator().manual_seed(args.seed)
    # Sampled fresh, never from the corpus, so a memorised answer scores 0.
    accuracy_tasks = list(getattr(args, "accuracy_task", None) or solving.GENERATORS)
    accuracy_probes = (
        solving.generate_novel(args.accuracy_problems, seed=args.seed + 900, tasks=accuracy_tasks)
        if args.accuracy_every > 0 else []
    )
    probe_manifest = accuracy_probe_manifest(
        accuracy_probes, accuracy_tasks if accuracy_probes else [], args.seed + 900, probe_cap
    )
    # Built only when a criterion needs it: indexing costs ~30s, and the
    # default criterion must not pay for a feature it does not use.
    recall = None
    if args.select_on in ("novelty", "balanced") and corpus_jsonl:
        recall = recall_index.RecallIndex.from_jsonl(corpus_jsonl)
        print(f"  recall index  {recall.hashes.size:,} windows / {recall.rows:,} rows")
    started = time.perf_counter()
    running, seen = 0.0, 0

    # On an accelerator, keep the packed corpus resident instead of copying a
    # batch across the bus every step. The v62 blend is 179,320 x 128 int64 =
    # 183 MB, negligible against any GPU this would run on, and the transfer it
    # removes is per-step overhead that dominates at these model sizes.
    # CPU is left exactly as it was: the tensors are already in host memory, so
    # moving them would be a no-op copy.
    accelerated = device.type != "cpu"
    if accelerated and args.resident_corpus:
        train_x, train_y = train_x.to(device), train_y.to(device)

    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    # fp16 needs loss scaling to keep small gradients from flushing to zero;
    # bf16 has the exponent range of fp32 and does not.
    scaler = torch.amp.GradScaler(device.type) if args.amp == "fp16" else None
    resumed = restore_resume_selection_state(
        init_provenance,
        start_step=args.start_step,
        select_on=args.select_on,
        batch_generator=generator,
        model=model,
        accuracy_probe=probe_manifest,
        scaler=scaler,
    )
    history: List[Dict[str, Any]] = resumed["history"]
    best_dev, best_score = resumed["best_dev_loss"], resumed["best_score"]
    best_dev_seen = resumed["best_dev_seen"]
    best_verbatim, best_accuracy = resumed["best_probe_verbatim_rate"], resumed["best_probe_accuracy"]
    best_state, best_step = resumed["best_state"], resumed["best_step"]
    last_accuracy = resumed["last_accuracy"]

    # `--start_step` resumes mid-curve: the run keeps the *same* `--steps`
    # OneCycle schedule and simply picks up where the crashed leg stopped, so
    # the learning rate continues down the curve instead of warming up again.
    # v74 segfaulted at step 11,500 of 18,000 after 9.2 hours; without this the
    # only options were to restart from zero or to re-warm on a fresh curve.
    for step in range(args.start_step + 1, args.steps + 1):
        model.train()
        pick = torch.randint(0, train_x.shape[0], (args.batch_size,), generator=generator)
        # The packed corpus is stored in the narrowest integer type that holds
        # the vocabulary (see `mimomix_text.compact_dtype`), which is what
        # keeps a 900k-row corpus off the pagefile. Embedding lookup and
        # cross_entropy both need int64, so the batch -- 16 x 128 values -- is
        # widened here rather than the whole corpus being held wide.
        batch_x = (train_x[pick] if accelerated and args.resident_corpus
                   else train_x[pick].to(device)).long()
        batch_y = (train_y[pick] if accelerated and args.resident_corpus
                   else train_y[pick].to(device)).long()

        if autocast_dtype is not None:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(batch_x, labels=batch_y)
        else:
            out = model(batch_x, labels=batch_y)

        optimiser.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        scheduler.step()
        model.step_router_bias()
        running += float(out.lm_loss.detach())
        seen += 1

        if step % args.eval_every == 0 or step == args.steps:
            dev_metrics = evaluate(model, dev_x, dev_y, args.eval_batch_size)
            verbatim = probe_verbatim_rate(model, tokenizer, recall)
            accuracy = None
            if accuracy_probes and (
                step % args.accuracy_every == 0 or step == args.steps
            ):
                accuracy_report = probe_accuracy_report(
                    model, tokenizer, accuracy_probes, max_new_tokens=probe_cap
                )
                accuracy = accuracy_report["accuracy"]
            entry = {
                "step": step,
                "train_lm_loss": round(running / max(1, seen), 6),
                "dev_loss": dev_metrics["loss"],
                "dev_perplexity": dev_metrics["perplexity"],
                "elapsed_seconds": round(time.perf_counter() - started, 1),
            }
            if verbatim is not None:
                entry["probe_verbatim_rate"] = round(verbatim, 4)
            if accuracy is not None:
                entry["probe_accuracy"] = round(accuracy, 4)
                entry["probe_by_task"] = accuracy_report["by_task"]
            history.append(entry)
            running, seen = 0.0, 0

            # Retain the last reading for recovery, but selection requires this
            # checkpoint's own measurement. A previous step's accuracy cannot
            # justify promoting these different weights on a loss tie-break.
            if accuracy is not None:
                last_accuracy = accuracy
            score = selection_score(
                args.select_on, dev_metrics["loss"], verbatim,
                accuracy,
            )
            selection_improved = score < best_score
            if selection_improved:
                best_score = score
                best_dev = dev_metrics["loss"]
                best_verbatim = verbatim
                best_accuracy = accuracy
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # Crash safety is decoupled from selection.
            #
            # This write used to sit inside the branch above, which was fine
            # while dev loss drove selection: it improves most evaluations, so a
            # checkpoint existed within minutes of any crash. Under
            # `--select_on accuracy` the score only moves when the accuracy probe
            # moves, and that is sampled every `--accuracy_every` steps -- v73
            # went **6.3 hours** without writing one, which is exactly the
            # exposure the v63 protection was added to remove.
            #
            # Dev loss is therefore the trigger, whatever criterion selects the
            # final model. The file records which improvement produced it, so a
            # resumed run is not silently assumed to hold the selected-best
            # weights when it holds the most recent dev-best instead.
            dev_improved = dev_metrics["loss"] < best_dev_seen
            if dev_improved:
                best_dev_seen = dev_metrics["loss"]
            if args.checkpoint_every_improvement and (selection_improved or dev_improved):
                save_progress_checkpoints(
                    output_dir=output_dir,
                    run_name=args.run_name,
                    model=model,
                    tokenizer=tokenizer,
                    extra={
                        "run_name": args.run_name,
                        "arm": args.arm,
                        "dev_loss_at_write": round(dev_metrics["loss"], 6),
                        "steps": step,
                        # Steps *completed* and the length of the schedule they
                        # were completed on are different numbers. A resume
                        # needs the second to know it is rejoining the same
                        # OneCycle curve; comparing against the first made a
                        # mid-run checkpoint look like a differently-shaped run.
                        "total_steps": args.steps,
                        "start_step": args.start_step,
                        "corpus_jsonl": str(corpus_jsonl) if corpus_jsonl else None,
                        "frozen_split": frozen_split,
                        "database": str(args.database) if args.database else None,
                        **selection_state_payload(
                            select_on=args.select_on,
                            checkpoint_step=step,
                            best_score=best_score,
                            best_step=best_step,
                            best_dev_loss=best_dev,
                            best_dev_seen=best_dev_seen,
                            best_probe_accuracy=best_accuracy,
                            best_probe_verbatim_rate=best_verbatim,
                            last_accuracy=last_accuracy,
                            batch_generator=generator,
                            history=history,
                            best_state=best_state,
                            accuracy_probe=probe_manifest,
                            scaler=scaler,
                        ),
                    },
                    selection_improved=selection_improved,
                    dev_improved=dev_improved,
                    optimiser=optimiser,
                    scheduler=scheduler,
                )
            print(
                f"step {step:>5}/{args.steps}  train {history[-1]['train_lm_loss']:.4f}  "
                f"dev {dev_metrics['loss']:.4f}  ppl {dev_metrics['perplexity']:.2f}  "
                + (f"acc {accuracy:.2f}  " if accuracy is not None else "")
                +
                f"{history[-1]['elapsed_seconds']:.0f}s",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = round(time.perf_counter() - started, 1)

    # Only now, once and never again, are the tiers touched.
    scored = score_tiers(
        model, split, tokenizer, args.sequence_length, args.eval_batch_size,
        turn_aligned=turn_aligned,
    )
    conversations = [
        generate_reply(model, tokenizer, prompt, args.sample_tokens) for prompt in PROBE_PROMPTS
    ]
    parity = generate_reply(model, tokenizer, PROBE_PROMPTS[1], args.sample_tokens, speculative=False)
    speculative = [c for c in conversations if c["prompt"] == PROBE_PROMPTS[1]][0]

    checkpoint_path = output_dir / f"{args.run_name}.pt"
    report: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.run_name,
        "arm": args.arm,
        "thinking_core": not args.no_thinking_core,
        "model": "v58_mimomix_generalisation",
        # Absent for a fresh run; present when weights were continued, because a
        # checkpoint trained in two legs is not the artifact its step count
        # suggests.
        "initialised_from": init_provenance,
        "architecture": "mimomix_core.MiMoMixModel (v53), unmodified",
        "split": split.report(tokenizer),
        "split_verification": verification,
        "frozen_split": frozen_split,
        "held_out_sentences": split.held_out_sentences,
        "tokenizer": tokenizer.vocabulary_report([a for _, a in split.dev]),
        "config": config.to_dict(),
        "parameters": parameters,
        # Provenance, not decoration. `--compare` diffs this block to decide
        # whether two arms are comparable, so a setting that is missing here is
        # a setting that can differ silently between them. Everything below
        # changes what a run measures.
        "hyperparameters": {
            "steps": args.steps,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "split_seed": args.split_seed,
            "eval_every": getattr(args, "eval_every", None),
            "accuracy_every": getattr(args, "accuracy_every", None),
            "accuracy_problems": getattr(args, "accuracy_problems", None),
            "select_on": getattr(args, "select_on", None),
            "turn_aligned_packing": bool(getattr(args, "turn_aligned_packing", False)),
            "digit_tokens": bool(getattr(args, "digit_tokens", False)),
            "reverse_digits": bool(getattr(args, "reverse_digits", False)),
            "min_response_characters": getattr(args, "min_response_characters", None),
            "max_vocab": getattr(args, "max_vocab", None),
            "corpus_jsonl": str(corpus_jsonl) if corpus_jsonl else None,
            "frozen_split_sha256": frozen_split["receipt_sha256"] if frozen_split else None,
            "amp": getattr(args, "amp", None),
            "decay_mode": getattr(args, "decay_mode", None),
            "repeat_subset_fraction": getattr(args, "repeat_subset_fraction", None),
            "repeat_subset_prob": getattr(args, "repeat_subset_prob", None),
            "mtp_loss_weight_final": getattr(args, "mtp_loss_weight_final", None),
            "mtp_weight_warmup_fraction": getattr(args, "mtp_weight_warmup_fraction", None),
            # The v85 headline. A receipt that does not say what the probe could
            # see cannot be read: a task truncated by the cap and a task the
            # model never learned produce the same 0.00.
            "probe_max_new_tokens": probe_cap,
        },
        "probe_token_budget": token_budget,
        "accuracy_probe": probe_manifest,
        "device": str(device_info.get("resolved", device)),
        "train_seconds": train_seconds,
        "history": history,
        "selection": {
            "selected_on": args.select_on,
            "best_step": best_step,
            "best_dev_loss": round(best_dev, 6),
            "best_probe_verbatim_rate": (
                round(best_verbatim, 4) if best_verbatim is not None else None
            ),
            "best_probe_accuracy": (
                round(best_accuracy, 4) if best_accuracy is not None else None
            ),
            "note": "no tier was evaluated before selection finished",
        },
        "tiers": scored,
        "gaps": generalisation_gap(scored),
        "uniform_baseline_loss": round(math.log(tokenizer.vocab_size), 6),
        "conversations": conversations,
        "decoding_parity": {
            "prompt": PROBE_PROMPTS[1],
            "greedy_reply": parity["reply"],
            "speculative_reply": speculative["reply"],
            "identical": parity["reply"] == speculative["reply"],
            "acceptance_length": speculative["acceptance_length"],
        },
        "routing": routing_report(model, dev_x, args.eval_batch_size),
        "checkpoint_path": str(checkpoint_path),
    }
    # Fall back to whichever tier was actually scored.
    #
    # A tier with no rows is recorded as `{"skipped": True}` with no "loss", and
    # indexing it raised `KeyError: 'loss'` *after* training finished -- losing
    # the whole run at the last step. Small or narrow corpora hit this routinely:
    # 6,000 rows of the scratchpad corpus produce 0 tier-1 and 0 tier-2 rows,
    # because nearly every response is unique and lands in tier 3.
    learned_from = next(
        (entry for entry in scored.values() if "loss" in entry), None
    )
    checks = {
        "split_verified": True,
        "learned_something": (
            learned_from["loss"] < 0.5 * math.log(tokenizer.vocab_size)
            if learned_from is not None
            else False
        ),
        "produces_non_empty_replies": all(c["reply"].strip() for c in conversations),
        "speculative_matches_greedy": report["decoding_parity"]["identical"],
        "selection_never_read_a_tier": True,
    }
    report["checks"] = checks
    report["passed"] = all(checks.values())

    save_talk_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        extra={
            "run_name": args.run_name,
            "arm": args.arm,
            "best_dev_loss": round(best_dev, 6),
            "select_on": args.select_on,
            "best_step": best_step,
            "steps": args.steps,
            "total_steps": args.steps,
            "start_step": args.start_step,
            "accuracy_probe": probe_manifest,
            "selection_checkpoint": True,
            "is_selection_best": True,
            "note": "selected weights for inference or warm start; use partial.pt for crash recovery",
            # The corpus travels with the weights. `eval_problem_solving`'s
            # "seen" arm is only a memorisation control if its rows are rows
            # this checkpoint trained on, and until v85 that arm defaulted to a
            # hard-coded v62 path -- so every run after v62 compared itself
            # against a corpus it had never seen and reported the difference as
            # a memorisation gap. A checkpoint that carries its own corpus lets
            # the benchmark check instead of assume.
            "corpus_jsonl": str(corpus_jsonl) if corpus_jsonl else None,
            "frozen_split": frozen_split,
            "database": str(args.database) if getattr(args, "database", None) else None,
            "created_at": report["created_at"],
        },
    )
    atomic_json(output_dir / "generalisation_results.json", report)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print()
    print(f"== v58 generalisation ladder | arm {report['arm']} ==")
    print(f"  parameters      {report['parameters']['total']:,} "
          f"({report['parameters']['active_per_token']:,} active/token)")
    print(f"  selected        step {report['selection']['best_step']} on "
          f"{describe_selection(report['selection'])}")
    print()
    print(f"  {'tier':<26} {'rows':>6} {'loss':>8} {'ppl':>8}")
    for name, entry in report["tiers"].items():
        if entry.get("skipped"):
            continue
        print(f"  {name:<26} {entry['pairs']:>6} {entry['loss']:>8.4f} {entry['perplexity']:>8.4f}")
    print()
    gaps = report["gaps"]
    for key in ("recombination_cost_nats", "unseen_sentence_cost_nats", "total_cost_nats"):
        if key in gaps:
            print(f"  {key:<32} {gaps[key]:+.4f}")
    if "perplexity_ratio_tier3_over_tier1" in gaps:
        print(f"  {'perplexity ratio tier3/tier1':<32} {gaps['perplexity_ratio_tier3_over_tier1']:.3f}x")
    print()
    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n  checkpoint  {report['checkpoint_path']}")


def compare(directories: Sequence[str]) -> Dict[str, Any]:
    """Compare a full arm against an ablation arm from their receipts.

    The comparison is only meaningful when the two runs differ in exactly one
    field, so that is checked rather than trusted: matching steps, batch size,
    sequence length, learning rate, seed and split seed are required, and the
    split itself must be the same set of withheld sentences.
    """

    reports = []
    for directory in directories:
        path = Path(directory) / "generalisation_results.json"
        if not path.exists():
            raise FileNotFoundError(f"no receipt at {path}")
        reports.append(json.loads(path.read_text(encoding="utf-8")))

    arms = {report["arm"]: report for report in reports}
    if set(arms) != {"full", "ablation"}:
        raise ValueError(f"need one 'full' and one 'ablation' arm, got {sorted(arms)}")
    full, ablation = arms["full"], arms["ablation"]

    mismatched = {
        key: (full["hyperparameters"][key], ablation["hyperparameters"][key])
        for key in full["hyperparameters"]
        if full["hyperparameters"][key] != ablation["hyperparameters"][key]
    }
    if sorted(full["held_out_sentences"]) != sorted(ablation["held_out_sentences"]):
        mismatched["held_out_sentences"] = ("differ", "differ")

    rows = []
    for name in splits.GeneralisationSplit.TIER_MEANINGS:
        a, b = full["tiers"].get(name), ablation["tiers"].get(name)
        if not a or not b or a.get("skipped") or b.get("skipped"):
            continue
        rows.append(
            {
                "tier": name,
                "full_loss": a["loss"],
                "ablation_loss": b["loss"],
                "delta_nats": round(b["loss"] - a["loss"], 6),
                "full_perplexity": a["perplexity"],
                "ablation_perplexity": b["perplexity"],
            }
        )

    return {
        "schema": COMPARISON_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": "does the recursive thinking core change text quality on this corpus?",
        "matched": not mismatched,
        "mismatched_hyperparameters": mismatched,
        "parameters": {
            "full": full["parameters"]["total"],
            "ablation": ablation["parameters"]["total"],
            "difference": full["parameters"]["total"] - ablation["parameters"]["total"],
        },
        "train_seconds": {"full": full["train_seconds"], "ablation": ablation["train_seconds"]},
        "tiers": rows,
        "interpretation_note": (
            "one seed per arm. A delta smaller than the seed-to-seed spread of this "
            "setup is not evidence of an effect in either direction, and no "
            "multi-seed spread has been measured here"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_talk_parser()
    parser.description = "Train MiMoMix against the v58 generalisation ladder"
    parser.add_argument("--arm", choices=("full", "ablation"), default="full",
                        help="'ablation' disables the recursive thinking core")
    parser.add_argument("--dev_fraction", type=float, default=0.01)
    parser.add_argument("--test_fraction", type=float, default=0.02)
    parser.add_argument("--tier3_row_fraction", type=float, default=0.02)
    parser.add_argument("--max_row_fraction_per_sentence", type=float, default=0.002)
    parser.add_argument("--split_seed", type=int, default=58)
    parser.add_argument("--frozen_split", default=None,
                        help="v87 original-source row split receipt; requires JSONL and turn-aligned packing")
    parser.add_argument(
        "--amp",
        choices=("off", "bf16", "fp16"),
        default="off",
        help=(
            "mixed-precision autocast for the forward pass. 'off' (default) "
            "keeps the fp32 path every published result was produced under. "
            "'bf16' is the right choice on Ampere or newer; 'fp16' adds a "
            "GradScaler and is for older accelerators. No effect on CPU results"
        ),
    )
    parser.add_argument(
        "--accuracy_every",
        type=int,
        default=0,
        help=(
            "measure exact-match accuracy on freshly generated problems every N "
            "steps. 0 (default) is off. Dev loss has twice proved unrelated to "
            "task accuracy here -- v71 finished with better loss and 28 points "
            "less accuracy -- so this gives the loop a number worth aborting on"
        ),
    )
    parser.add_argument(
        "--accuracy_problems",
        type=int,
        default=20,
        help="problems per accuracy probe; generation is slow on CPU",
    )
    parser.add_argument(
        "--accuracy_task",
        action="append",
        choices=tuple(solving.GENERATORS),
        default=None,
        help="repeat to pin the ordered task list used by every accuracy probe",
    )
    parser.add_argument(
        "--select_on",
        choices=("dev_loss", "novelty", "balanced", "accuracy"),
        default="dev_loss",
        help=(
            "what to choose the best checkpoint on. 'dev_loss' is the default "
            "and reproduces every published run, but v64 showed it reliably "
            "picks the most memorised checkpoint: dev improved 1.0762 -> 0.9910 "
            "while the verbatim rate of generated replies went 0.14 -> 0.76. "
            "'novelty' minimises that verbatim rate; 'balanced' minimises "
            "dev_loss + 0.5 * verbatim. Both require --corpus_jsonl"
        ),
    )
    parser.add_argument(
        "--digit_tokens",
        action="store_true",
        help=(
            "split numbers into single digits. The default makes '498' one "
            "opaque token, which puts arithmetic out of reach in principle -- "
            "measured on a 240k arithmetic corpus, 94.8%% of the vocabulary was "
            "numbers and accuracy was 1.7%%, identical on seen and unseen "
            "problems. Splitting digits took that vocabulary from 16,390 to 876 "
            "at coverage 1.0000"
        ),
    )
    parser.add_argument(
        "--turn_aligned_packing",
        action="store_true",
        help=(
            "give every turn its own padded block instead of chopping a "
            "concatenated stream on a fixed stride. Measured on the v63 corpus, "
            "the stream packing leaves 56.0%% of supervised tokens in a block "
            "with no prompt in it, which trains the model to emit the corpus's "
            "modal reply regardless of the question. Default off, so every "
            "result up to v62 reproduces"
        ),
    )
    parser.add_argument(
        "--checkpoint_every_improvement",
        action="store_true",
        help=(
            "write <run_name>.partial.pt whenever dev loss improves, so a long "
            "run survives a crash or a kill. Without it the best weights live "
            "only in memory until the loop finishes"
        ),
    )
    parser.add_argument(
        "--resident_corpus",
        action="store_true",
        help=(
            "keep the packed corpus on the accelerator instead of copying each "
            "batch across the bus per step. Costs ~183 MB of device memory for "
            "the v62 blend and removes a per-step transfer that dominates at "
            "these model sizes. Ignored on CPU"
        ),
    )
    parser.add_argument(
        "--init_from",
        default=None,
        help=(
            "continue training from this checkpoint instead of from random "
            "weights. The vocabulary must be byte-identical, which is verified; "
            "a mismatch raises rather than silently training a wrong embedding"
        ),
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help=(
            "resume a crashed leg at this step of the same --steps curve. Use "
            "with --init_from pointing at that leg's checkpoint; --steps must "
            "match the original run so the OneCycle schedule is the same one"
        ),
    )
    parser.add_argument(
        "--min_response_characters",
        type=int,
        default=8,
        help=(
            "drop rows whose answer is shorter than this. The default 8 comes "
            "from a dialogue corpus where short replies were truncation "
            "artifacts; it deletes 73.5%% of arithmetic rows, whose correct "
            "answers are values like '79' and '9/14'. Use 1 for maths corpora"
        ),
    )
    parser.add_argument(
        "--corpus_jsonl",
        default=None,
        help=(
            "read the corpus from a JSONL file of {user, assistant} records "
            "instead of --database. Use this to run the ladder on a corpus with "
            "measured diversity beyond the 292 word types of llm_chat.db, which "
            "v58 names as an unmet promotion gate"
        ),
    )
    parser.add_argument("--compare", nargs=2, metavar=("FULL_DIR", "ABLATION_DIR"),
                        help="compare two finished arms instead of training")
    # Architecture defaults are the *published v57 configuration*, not the v57
    # parser's defaults, which differ (hidden 256 / 6 layers). Matching the
    # shipped run is what makes a v58 tier number comparable to the 1.27 headline
    # rather than merely adjacent to it.
    parser.set_defaults(
        run_name="v58_full",
        output_dir=str(SOURCE_DIR.parent / "output" / "v58_full"),
        steps=2000,
        hidden_size=192,
        n_layers=4,
        n_heads=6,
        n_kv_heads=2,
        intermediate_size=384,
        moe_intermediate_size=96,
    )
    return parser


def describe_selection(selection: Dict[str, Any]) -> str:
    """Say which criterion chose the checkpoint, and what it read.

    This line used to print "on dev (dev loss ...)" whatever `--select_on`
    was, which states the opposite of what v64 established: dev loss is not
    the criterion, and under `--select_on accuracy` it is not even consulted
    except as a tie-break. A run selected on a 0.89 accuracy probe reporting
    only its dev loss is how a summary quietly becomes wrong.
    """

    criterion = selection.get("selected_on") or "dev_loss"
    dev_loss = selection.get("best_dev_loss")
    dev_text = f"dev loss {dev_loss:.4f}" if dev_loss is not None else "dev loss unmeasured"

    accuracy = selection.get("best_probe_accuracy")
    verbatim = selection.get("best_probe_verbatim_rate")
    if criterion == "accuracy" and accuracy is not None:
        return f"accuracy (probe {accuracy:.2f}, {dev_text})"
    if criterion in ("novelty", "balanced") and verbatim is not None:
        return f"{criterion} (verbatim {verbatim:.2f}, {dev_text})"
    return f"{criterion} ({dev_text})"


def validate_resume_settings(args) -> None:
    """Reject resume settings that would train a curve nobody intended."""

    start = getattr(args, "start_step", 0)
    if start <= 0:
        return
    if not getattr(args, "init_from", None):
        raise SystemExit(
            "--start_step resumes a crashed leg and needs --init_from pointing "
            "at that leg's checkpoint; without it the weights would be random."
        )
    if start >= args.steps:
        raise SystemExit(
            f"--start_step {start} leaves no steps to run against --steps "
            f"{args.steps}; the leg it resumes is already complete."
        )


def validate_selection_settings(args) -> None:
    """Refuse a selection criterion the run cannot measure well enough.

    Checked before the corpus is read, so a misconfiguration costs seconds
    rather than being discovered after a fourteen-hour run has selected on
    sampling noise.
    """

    if args.select_on != "accuracy":
        return
    if args.accuracy_every <= 0:
        raise SystemExit(
            "--select_on accuracy requires --accuracy_every > 0; without a probe "
            "there is no accuracy to select on."
        )
    if args.accuracy_problems < MIN_SELECTION_PROBLEMS:
        raise SystemExit(
            f"--select_on accuracy needs --accuracy_problems >= "
            f"{MIN_SELECTION_PROBLEMS}, got {args.accuracy_problems}. At n=20 the "
            "95% interval is about +-22 points, and v73's 20-problem probe read "
            "0.15 where a 60-problem evaluation read 0.467. Use a larger probe, "
            "or keep the small probe for monitoring and select on dev_loss."
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_selection_settings(args)
    validate_resume_settings(args)
    if args.compare:
        result = compare(args.compare)
        print(json.dumps(result, indent=2))
        atomic_json(Path(args.output_dir).parent / "v58_thinking_core_ablation.json", result)
        return 0 if result["matched"] else 1
    report = run(args)
    print_summary(report)
    if args.enforce_gates and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
