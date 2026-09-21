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
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_eval_splits as splits  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
import eval_problem_solving as solving  # noqa: E402
import recall_index  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_core import MiMoMixModel, SparseMoEFeedForward, pin_layout_for_growth  # noqa: E402
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


#: Parameter-name prefixes of modules grafted onto a trained checkpoint. They
#: start untrained while everything else is converged, so they may take a
#: larger learning rate (`--new_param_lr_mult`). This is the v91 value and
#: the default of `split_new_parameter_groups`; a v93 run derives its own
#: list with `new_parameter_prefixes(config)`.
NEW_PARAMETER_PREFIXES = ("cns_core.",)
#: Grafted parameters that are gains, gates, biases or logits rather than
#: matrices. Decaying `edge_logit` toward 0 would pull every connectome edge
#: toward softplus(0) = 0.69 -- a strength nobody measured -- so they are not
#: decayed. `read_in`/`read_out` are ordinary matrices and are. The v93 sites
#: need no new entries: `extra_gates.<j>` and `thinking_gate` are 1-D and
#: `split_new_parameter_groups` already leaves every 1-D tensor undecayed,
#: while `extra_read_in`/`extra_read_out`/`to_thinking` are matrices.
NEW_UNDECAYED_LEAVES = ("edge_logit", "node_bias", "leak_logit", "gate", "weight_norm")


def new_parameter_prefixes(config: Any) -> Tuple[str, ...]:
    """Grafted-parameter prefixes for this config: the core plus grown blocks.

    A block appended by `--grow_layers` (v93 D4) is as untrained as the
    connectome graft -- its `o_proj`/`down_proj` are zeroed after the warm
    start -- so it takes the same `--new_param_lr_mult` and the same
    decayed/undecayed split. With `grow_layers == 0` this is exactly
    `NEW_PARAMETER_PREFIXES`, so a v91 command builds the v91 optimiser.
    """

    grown = int(getattr(config, "grow_layers", 0) or 0)
    n_layers = int(getattr(config, "n_layers", 0) or 0)
    return NEW_PARAMETER_PREFIXES + tuple(
        f"layers.{index}." for index in range(n_layers - grown, n_layers)
    )


def split_new_parameter_groups(
    model: torch.nn.Module,
    groups: List[Dict[str, Any]],
    prefixes: Sequence[str] = NEW_PARAMETER_PREFIXES,
    lr_mult: float = 1.0,
) -> List[Dict[str, Any]]:
    """Move grafted parameters into their own AdamW groups.

    A model without grafted modules gets `groups` back unchanged -- same
    tensors, same order, same group count -- so the v89 control arm builds the
    identical optimiser it would have built before v91. Grafted parameters are
    removed from the existing groups and appended as a decayed group and an
    undecayed group, each carrying `_lr_mult` for the per-group OneCycle peak.
    """

    names = {id(p): n for n, p in model.named_parameters()}
    new_ids = {
        id(p) for n, p in model.named_parameters()
        if p.requires_grad and any(n.startswith(prefix) for prefix in prefixes)
    }
    if not new_ids:
        return groups
    kept = []
    for group in groups:
        params = [p for p in group["params"] if id(p) not in new_ids]
        if params:
            kept.append({**group, "params": params})
    decayed, undecayed = [], []
    for name, parameter in model.named_parameters():
        if id(parameter) not in new_ids:
            continue
        leaf = name.rsplit(".", 1)[-1]
        is_norm_gain = name.endswith("norm.weight")
        if leaf in NEW_UNDECAYED_LEAVES or is_norm_gain or parameter.ndim <= 1:
            undecayed.append(parameter)
        else:
            decayed.append(parameter)
    weight_decay = float(groups[0].get("weight_decay", 0.0)) if groups else 0.0
    lr_note = float(lr_mult)
    for params, decay in ((decayed, weight_decay), (undecayed, 0.0)):
        if params:
            kept.append({
                "params": params,
                "weight_decay": decay,
                "_lr_mult": lr_note,
                "_names": [names[id(p)] for p in params],
            })
    return kept


#: Fraction of the old embedding's scalar std used for a new token's row
#: (v93 D5). Small enough that a new id starts as "the average token" rather
#: than as noise the trunk has never seen, non-zero so that two new rows are
#: distinguishable from the first gradient step.
NEW_TOKEN_NOISE_SCALE = 0.1

#: Tensors that carry one row per expert *slot* and therefore grow by
#: `--moe_spare_experts` while every other MoE tensor keeps its shape (v93 D3).
#: A checkpoint row is copied into the leading slice; the spare rows keep the
#: construction init, which is what a dead slot is (its logit is -inf'd).
#: `expert_alive` is absent from every checkpoint written before v93 (the
#: module fills it as all-alive) and present at the live count in one written
#: since; padding it keeps the source's mask and leaves the spares dead.
SLOT_GROWN_LEAF_SUFFIXES = ("mlp.gate.weight", "mlp.expert_bias", "mlp.expert_alive")


def _grow_rows(matrix: torch.Tensor, new_rows: int, generator: torch.Generator) -> torch.Tensor:
    """Append ``new_rows - rows`` rows of ``mean(old) + 0.1 std(old) N(0,1)``."""

    old = matrix.detach().float()
    mean = old.mean(dim=0, keepdim=True)
    std = float(old.std()) if old.numel() > 1 else 0.0
    noise = torch.randn(int(new_rows) - old.shape[0], old.shape[1], generator=generator)
    fresh = mean + NEW_TOKEN_NOISE_SCALE * std * noise
    return torch.cat([matrix.detach(), fresh.to(matrix.dtype)], dim=0)


def grow_embedding_rows(
    state: Dict[str, torch.Tensor], old_rows: int, new_rows: int, seed: int
) -> Dict[str, Any]:
    """Grow the tied embedding of a checkpoint state to ``new_rows`` (v93 D5).

    Writes ``embed_tokens.weight`` and ``lm_head.weight`` in ``state`` in
    place. With tied weights the two keys hold the same values, so the head
    receives the *same* grown tensor rather than a second noise draw; an
    untied head (``--no_tie_word_embeddings``) is grown with its own draw from
    the same generator. Returns the receipt block.
    """

    embed = state["embed_tokens.weight"]
    if int(embed.shape[0]) != int(old_rows):
        raise ValueError(
            f"checkpoint embedding has {embed.shape[0]} rows but its tokenizer has "
            f"{old_rows} tokens; the checkpoint is internally inconsistent"
        )
    generator = torch.Generator().manual_seed(int(seed))
    grown = _grow_rows(embed, new_rows, generator)
    state["embed_tokens.weight"] = grown
    tied = None
    head = state.get("lm_head.weight")
    if head is not None and int(head.shape[0]) == int(old_rows):
        tied = bool(head.shape == embed.shape and torch.equal(head, embed))
        state["lm_head.weight"] = grown if tied else _grow_rows(head, new_rows, generator)
    new = grown[int(old_rows):].float()
    old = embed.detach().float()
    return {
        "old": int(old_rows),
        "new": int(new_rows),
        "added": int(new_rows) - int(old_rows),
        "seed": int(seed),
        "init": f"mean(old rows) + {NEW_TOKEN_NOISE_SCALE} * std(old) * N(0, 1)",
        "head_tied_to_embedding": tied,
        "old_row_std": round(float(old.std()), 6),
        "new_row_distance_from_mean": round(float((new - old.mean(dim=0)).norm(dim=1).mean()), 6),
    }


def _vocabulary_mismatch(source_tokens: List[str], tokenizer: text_utils.WordTokenizer) -> str:
    if len(source_tokens) != tokenizer.vocab_size:
        return (
            f"sizes differ: {len(source_tokens)} tokens against "
            f"{tokenizer.vocab_size}"
        )
    differing = [
        index
        for index, (a, b) in enumerate(zip(source_tokens, tokenizer.tokens))
        if a != b
    ]
    first = differing[0] if differing else None
    return (
        f"same size ({tokenizer.vocab_size}) but {len(differing)} ids denote "
        f"different words, first at id {first}: "
        f"{source_tokens[first]!r} against {tokenizer.tokens[first]!r}"
    )


def load_initial_weights(
    model: MiMoMixModel,
    tokenizer: text_utils.WordTokenizer,
    checkpoint: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    extend_vocab: bool = False,
    seed: int = 0,
    grow_layers: int = 0,
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

    **v93 growth (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D3-D5).** Three
    departures from identity are accepted, each only in its named case, and
    everything else keeps the identical-shape rule:

    * ``extend_vocab``: the checkpoint's token list may be a byte-identical
      *prefix* of the live one (same ``digit_tokens``/``reverse_digits``); the
      tied embedding is grown with `grow_embedding_rows` before the shape
      scan, seeded by ``seed`` so the receipt can reproduce the new rows.
    * spare expert slots: ``mlp.gate.weight`` / ``mlp.expert_bias`` whose
      live dim 0 is larger get the checkpoint rows in the leading slice.
    * missing keys are allowed only under ``cns_core.`` (the graft),
      ``layers.<k>.`` for the ``grow_layers`` appended blocks and
      ``mlp.experts.<j>.`` for the spare slots; any other missing key raises,
      because a tensor that trains from random init while the receipt says
      "warm start" is exactly the kind of silent difference the vocabulary
      check exists to catch. Appended blocks whose keys were missing are
      zeroed into identity (:meth:`MiMoMixModel.zero_new_blocks`); on a crash
      resume they are present, trained, and left alone.

    ``payload`` lets the caller pass an already-loaded checkpoint (the trainer
    reads it once for the tokenizer when extending). Returns provenance for
    the receipt: a checkpoint trained in two legs is not the same artifact as
    one trained in a single run, and the receipt should say so.
    """

    if payload is None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    source_tokenizer = payload.get("tokenizer", {}) or {}
    source_tokens = list(source_tokenizer.get("tokens", []))
    # A shallow copy: grown tensors replace entries here and never touch the
    # caller's payload (a resume test compares against it later).
    state: Dict[str, torch.Tensor] = dict(payload["state_dict"])
    grown_vocab: Optional[Dict[str, Any]] = None
    if source_tokens != tokenizer.tokens:
        same_flags = (
            bool(source_tokenizer.get("digit_tokens", False)) == bool(tokenizer.digit_tokens)
            and bool(source_tokenizer.get("reverse_digits", False)) == bool(tokenizer.reverse_digits)
        )
        is_prefix = (
            0 < len(source_tokens) < tokenizer.vocab_size
            and tokenizer.tokens[: len(source_tokens)] == source_tokens
        )
        if extend_vocab and is_prefix and not same_flags:
            detail = (
                f"the {len(source_tokens)} checkpoint tokens are a prefix of the live "
                f"{tokenizer.vocab_size} but digit_tokens/reverse_digits differ "
                f"({source_tokenizer.get('digit_tokens', False)}/"
                f"{source_tokenizer.get('reverse_digits', False)} against "
                f"{tokenizer.digit_tokens}/{tokenizer.reverse_digits}), so numbers "
                "would be segmented differently"
            )
        elif extend_vocab and is_prefix:
            detail = None
        else:
            detail = _vocabulary_mismatch(source_tokens, tokenizer)
            if not extend_vocab and is_prefix and same_flags:
                detail += " (a prefix: pass --extend_vocab to grow the embedding instead)"
        if detail is not None:
            raise ValueError(
                f"{checkpoint} has a different vocabulary -- {detail}. Token ids would "
                "denote different words, so continuing from it would train on a "
                "silently corrupted embedding. This usually means the corpus, "
                "--pairs, --max_vocab or the split seed differs from the source run; "
                "match them exactly, or train fresh."
            )
        grown_vocab = grow_embedding_rows(state, len(source_tokens), tokenizer.vocab_size, seed)

    live_state = model.state_dict()
    grown_experts: List[Dict[str, Any]] = []
    for key, value in list(state.items()):
        live = live_state.get(key)
        if live is None or value.shape == live.shape:
            continue
        slot_grown = (
            key.endswith(SLOT_GROWN_LEAF_SUFFIXES)
            and value.ndim == live.ndim
            and tuple(value.shape[1:]) == tuple(live.shape[1:])
            and int(value.shape[0]) < int(live.shape[0])
        )
        if slot_grown:
            padded = live.detach().clone()
            padded[: value.shape[0]] = value
            state[key] = padded
            grown_experts.append({"key": key, "rows": [int(value.shape[0]), int(live.shape[0])]})

    incompatible = {
        key: (tuple(value.shape), tuple(live_state[key].shape))
        for key, value in state.items()
        if key in live_state and value.shape != live_state[key].shape
    }
    if incompatible:
        raise ValueError(
            f"{checkpoint} has {len(incompatible)} tensors whose shape differs from "
            f"this architecture, e.g. {next(iter(incompatible.items()))}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)

    grow_layers = int(grow_layers or 0)
    n_layers = int(model.config.n_layers)
    first_new_layer = n_layers - grow_layers
    live_experts = int(model.config.n_routed_experts)
    spare_experts = int(getattr(model.config, "moe_spare_experts", 0) or 0)

    def allowed_group(key: str) -> Optional[str]:
        if key.startswith("cns_core."):
            return "cns_core"
        block = re.match(r"layers\.(\d+)\.", key)
        if block and grow_layers > 0 and int(block.group(1)) >= first_new_layer:
            return "grown_layers"
        expert = re.match(r"layers\.\d+\.mlp\.experts\.(\d+)\.", key)
        if expert and spare_experts > 0 and int(expert.group(1)) >= live_experts:
            return "spare_experts"
        return None

    allowed_missing: Dict[str, int] = {}
    disallowed: List[str] = []
    for key in sorted(missing):
        group = allowed_group(key)
        if group is None:
            disallowed.append(key)
        else:
            allowed_missing[group] = allowed_missing.get(group, 0) + 1
    if disallowed:
        raise ValueError(
            f"{checkpoint} lacks {len(disallowed)} tensor(s) this architecture needs, "
            f"e.g. {disallowed[0]!r}. Only the connectome graft (cns_core.*), blocks "
            f"appended by --grow_layers (layers.{{k}}.* for k >= {first_new_layer}) and "
            f"spare expert slots (mlp.experts.{{j}}.* for j >= {live_experts} under "
            "--moe_spare_experts) may be absent from a warm start; anything else "
            "would train from random weights while the receipt calls it continued."
        )

    grown_layers: Optional[Dict[str, Any]] = None
    if grow_layers > 0:
        grown_keys = [
            key for key in live_state
            if re.match(r"layers\.(\d+)\.", key) and int(key.split(".")[1]) >= first_new_layer
        ]
        absent = allowed_missing.get("grown_layers", 0)
        if absent == len(grown_keys):
            zeroed: Optional[Dict[str, int]] = model.zero_new_blocks(first_new_layer)
            source_had_blocks = False
        elif absent == 0:
            # A crash resume: the appended blocks are in the checkpoint,
            # trained, and must not be reset.
            zeroed = None
            source_had_blocks = True
        else:
            raise ValueError(
                f"{checkpoint} carries {len(grown_keys) - absent} of the {len(grown_keys)} "
                f"tensors of the appended blocks (layers.{first_new_layer}+); a block "
                "that is half loaded is neither the source's nor an identity"
            )
        grown_layers = {
            "first_new_layer": first_new_layer,
            "count": grow_layers,
            "source_had_blocks": source_had_blocks,
            "zeroed": zeroed,
        }

    if bool(model.config.tie_word_embeddings) and model.lm_head.weight is not model.embed_tokens.weight:
        raise RuntimeError("embedding tie lost during load: lm_head.weight is no longer embed_tokens.weight")

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
        # v93 growth provenance (all None/empty for a v89/v91 warm start).
        "grown_vocab": grown_vocab,
        "grown_experts": grown_experts or None,
        "grown_layers": grown_layers,
        "allowed_missing": allowed_missing,
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
        # The neurogenesis controller's apoptosis counters (v93 D6), restored
        # by the trainer on a crash resume; popped before the receipt.
        "_neurogenesis_state": extra.get("neurogenesis_state"),
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


# ---------------------------------------------------------------------------
# v93 neurogenesis instruments (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D6, D8)
# ---------------------------------------------------------------------------

#: Counters of a neurogenesis event record that the per-eval history keeps.
#: The full record (per-pair lists, seconds, by-block detail) lives in
#: output/<run>/neurogenesis.jsonl; the history carries enough to read the
#: training curve against what grew at that step.
EVENT_COUNT_KEYS = (
    "modules_split", "modules_killed", "edges_opened", "edges_pruned",
    "taps_opened", "experts_born", "experts_killed",
)


def _count(value: Any) -> int:
    """An event field as one integer: a list's length, a dict's ``count`` or
    the sum of its integer entries, an int itself."""

    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, Mapping):
        if "count" in value:
            return _count(value["count"])
        return sum(_count(v) for v in value.values() if isinstance(v, (int, float, list, tuple)))
    return 0


def event_counts(event: Mapping[str, Any]) -> Dict[str, Any]:
    """The compact history form of one controller event record."""

    summary: Dict[str, Any] = {key: _count(event.get(key)) for key in EVENT_COUNT_KEYS}
    for key in ("witness_loss_before", "witness_loss_after", "witness_delta"):
        if event.get(key) is not None:
            summary[key] = round(float(event[key]), 6)
    if "flagged" in event:
        summary["flagged"] = bool(event["flagged"])
    for key in ("alive_modules", "alive_edges"):
        if event.get(key) is not None:
            summary[key] = int(event[key])
    if event.get("alive_experts_per_layer") is not None:
        summary["alive_experts_per_layer"] = [int(v) for v in event["alive_experts_per_layer"]]
    return summary


def growth_telemetry(model: MiMoMixModel) -> Optional[Dict[str, Any]]:
    """Alive counts and gate sizes per site, for the history and the receipt.

    ``None`` for a model with neither a connectome core nor spare expert
    slots, so a v89/v91 history entry keeps its exact keys.
    """

    core = getattr(model, "cns_core", None)
    moe_layers = [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]
    spares = any(int(getattr(m, "n_spare", 0)) > 0 for m in moe_layers)
    if core is None and not spares:
        return None
    snapshot: Dict[str, Any] = {}
    if core is not None:
        t = core.telemetry()
        gates: List[Dict[str, Any]] = [{
            "site": int(t["write_layers"][0]),
            "gate_mean_abs": round(float(t["gate_mean_abs"]), 8),
            "gate_max_abs": round(float(t["gate_max_abs"]), 8),
        }]
        for extra in t.get("extra_gates", []):
            gates.append({
                "site": int(extra["layer"]),
                "gate_mean_abs": round(float(extra["gate_mean_abs"]), 8),
                "gate_max_abs": round(float(extra["gate_max_abs"]), 8),
            })
        if "thinking_gate_mean_abs" in t:
            gates.append({
                "site": "thinking",
                "gate_mean_abs": round(float(t["thinking_gate_mean_abs"]), 8),
                "gate_max_abs": round(float(t["thinking_gate_max_abs"]), 8),
            })
        snapshot.update({
            "alive_modules": int(t["alive_nodes"]),
            "grown_modules": int(t["grown_nodes"]),
            "hemisphere_modules": list(t["hemisphere_nodes"]),
            "alive_edges": int(t["edges_installed"]),
            "edges_by_block": dict(t["edges_by_block"]),
            "grown_edges": int(t["grown_edges"]),
            "taps": {
                "in": int(t["taps_in"]), "out": int(t["taps_out"]),
                "in_grown": int(t["taps_in_grown"]), "out_grown": int(t["taps_out_grown"]),
            },
            "gates": gates,
        })
    if moe_layers:
        snapshot["alive_experts_per_layer"] = [int(m.alive_count()) for m in moe_layers]
    return snapshot


def v93_ablation_report(
    model: MiMoMixModel,
    dev_x: torch.Tensor,
    dev_y: torch.Tensor,
    batch_size: int,
    rows: int,
    *,
    grown: bool,
    reference_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Dev-loss cost of each v93 wiring element on the same weights and rows.

    ``rows`` limits the dev rows (0 = all); each ablation is one extra dev
    pass, ~15 min each on the full v89 dev set, which is why the cap exists.
    With ``rows == 0`` the unablated reference is ``reference_loss`` when the
    caller measured it on the full set already (the v91 block's
    ``dev_loss_core_on``), so nothing is scored twice. Every switch is
    restored afterwards, whatever happens in between.
    """

    core = model.cns_core
    x, y = (dev_x[:rows], dev_y[:rows]) if rows > 0 else (dev_x, dev_y)
    if rows <= 0 and reference_loss is not None:
        base = float(reference_loss)
    else:
        base = float(evaluate(model, x, y, batch_size)["loss"])
    alive = core.alive.bool()
    sides = sorted(set(int(s) for s in core.hemisphere[alive].tolist()))
    two_sided = sides == [0, 1]
    ablations: Dict[str, Dict[str, Any]] = {}

    def measure(name: str, detail: Optional[Dict[str, Any]] = None) -> None:
        loss = float(evaluate(model, x, y, batch_size)["loss"])
        ablations[name] = {
            "dev_loss": round(loss, 6),
            "cost_nats": round(loss - base, 6),
            **(detail or {}),
        }

    gates = [core.gate, *list(core.extra_gates)]
    if core.thinking_gate is not None:
        gates.append(core.thinking_gate)
    saved = [g.detach().clone() for g in gates]
    try:
        with torch.no_grad():
            for gate in gates:
                gate.zero_()
        measure("gates_off", {"gates": len(gates), "note": "every write site and the thinking bond shut"})
    finally:
        with torch.no_grad():
            for gate, value in zip(gates, saved):
                gate.copy_(value)

    if two_sided:
        try:
            core.ablate_cross = True
            measure("cross_off", {"note": "LR and RL (commissural) blocks masked"})
        finally:
            core.ablate_cross = False
        # ablate_side removes the named side, so "left only" masks side 1.
        for name, removed in (("left_only", 1), ("right_only", 0)):
            try:
                core.ablate_side = removed
                measure(name, {"masked_hemisphere": "R" if removed == 1 else "L"})
            finally:
                core.ablate_side = None
    else:
        skipped = {"skipped": "single hemisphere graph", "hemispheres": sides}
        ablations["cross_off"] = dict(skipped)
        ablations["left_only"] = dict(skipped)
        ablations["right_only"] = dict(skipped)

    if grown:
        try:
            core.ablate_grown = True
            measure("grown_off", {"note": "every grown module, edge and tap returned to its birth state"})
        finally:
            core.ablate_grown = False
    else:
        ablations["grown_off"] = {"skipped": "neurogenesis was off (--grow_every 0)"}

    if core.ablate_cross or core.ablate_grown or core.ablate_side is not None:
        raise RuntimeError("v93 ablation switches were not restored")
    return {
        "rows": int(x.shape[0]),
        "rows_note": "first --ablation_rows dev rows" if rows > 0 else "full dev set",
        "dev_loss": round(base, 6),
        "reference": (
            "reused cns_core.dev_loss_core_on (same rows)"
            if rows <= 0 and reference_loss is not None else "measured on these rows"
        ),
        "ablations": ablations,
        "note": (
            "cost_nats = ablated dev loss - unablated dev loss on the same rows; "
            "> 0 means the selected weights rely on that element"
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    validate_resume_settings(args)
    validate_selection_settings(args)
    validate_growth_settings(args)
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
    initialised_from = getattr(args, "init_from", None)
    init_payload: Optional[Dict[str, Any]] = None
    vocabulary_provenance: Optional[Dict[str, Any]] = None
    if getattr(args, "extend_vocab", False):
        # v93 D5. `build` orders ids by frequency, so a corpus that differs
        # by one row from the source run's renumbers the vocabulary and the
        # warm start is (correctly) refused. Extending keeps the checkpoint's
        # ids as a prefix and appends only what the new corpus adds. The
        # digit settings come from the checkpoint, not the flags: a tokenizer
        # reloaded under the other setting segments every number differently.
        if not initialised_from:
            raise SystemExit("--extend_vocab appends to a checkpoint's vocabulary and needs --init_from")
        init_payload = torch.load(initialised_from, map_location="cpu", weights_only=False)
        base_tokenizer = text_utils.WordTokenizer.from_dict(init_payload["tokenizer"])
        wanted = tokenizer_options(args)
        for flag in ("digit_tokens", "reverse_digits"):
            if bool(wanted.get(flag, False)) != bool(getattr(base_tokenizer, flag)):
                raise SystemExit(
                    f"--extend_vocab takes {flag} from {initialised_from} "
                    f"({getattr(base_tokenizer, flag)}), but the command says "
                    f"{bool(wanted.get(flag, False))}; numbers would be segmented "
                    "differently from the embedding that was trained on them. "
                    "Match the checkpoint's setting."
                )
        max_new = int(getattr(args, "max_new_tokens_vocab", 4000))
        if args.start_step > 0:
            # A crash resume rebuilds the crashed leg's model exactly. The
            # recovery checkpoint already carries the extended token list, and
            # extending it again would append whatever `max_new` had cut off
            # in the first leg -- a different vocabulary, a different
            # embedding shape, and an optimiser state that no longer fits.
            tokenizer = base_tokenizer
            vocabulary_provenance = {
                "source": "checkpoint",
                "base_vocab": base_tokenizer.vocab_size,
                "added": 0,
                "max_new": max_new,
                "reason": "crash resume takes the recovery checkpoint's token list verbatim",
            }
        else:
            tokenizer = text_utils.WordTokenizer.extend(
                base_tokenizer,
                (field for pair in split.train for field in pair),
                max_new=max_new,
            )
            vocabulary_provenance = {
                "source": "extended",
                "base_vocab": base_tokenizer.vocab_size,
                "added": tokenizer.vocab_size - base_tokenizer.vocab_size,
                "max_new": max_new,
                "capped": tokenizer.vocab_size - base_tokenizer.vocab_size >= max_new,
            }
        print(f"  vocabulary   {vocabulary_provenance['source']}: {base_tokenizer.vocab_size} from "
              f"{initialised_from} + {vocabulary_provenance['added']} new ids", flush=True)
    else:
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
    grow_layers = int(getattr(args, "grow_layers", 0) or 0)
    if grow_layers > 0:
        # v93 D4. Applied to the built config rather than through the config
        # table: it changes n_layers and pins global_layers to the existing
        # layout, which is a transformation of a config, not a field of one.
        if not initialised_from:
            raise SystemExit(
                "--grow_layers appends identity blocks to a warm-started model and "
                "needs --init_from; to train a deeper model from scratch raise --n_layers"
            )
        config = pin_layout_for_growth(config, grow_layers)
    model = MiMoMixModel(config).to(device)
    cns_graph_info: Optional[Dict[str, Any]] = None
    if getattr(model, "cns_core", None) is not None:
        if not getattr(args, "cns_graph", ""):
            raise SystemExit("--cns_core needs --cns_graph (see source/malecns_connectome.py)")
        # Installed before --init_from: the source checkpoint has no cns_core
        # keys, so the non-strict load below leaves this wiring untouched.
        # (A crash resume's checkpoint does carry them, grown state included,
        # and overwrites this install -- which is the point of the resume.)
        # v93: load_graph fills the leading `cns_nodes - cns_spare_nodes`
        # slots from the npz and leaves the spares dead.
        cns_graph_info = model.cns_core.load_graph(
            args.cns_graph, wiring=args.cns_wiring,
            spectral_radius=float(getattr(args, "cns_spectral_radius", 0.9)),
        )
        cns_snapshot = model.cns_core.telemetry()
        print(f"  cns core     {cns_graph_info['wiring']} | {cns_graph_info['nodes']} modules, "
              f"{cns_graph_info['edges']:,} edges, {cns_graph_info['input_nodes']} in / "
              f"{cns_graph_info['output_nodes']} out, after block {config.cns_run_after_layer}")
        if cns_snapshot.get("dead_nodes") or len(cns_snapshot.get("read_layers", [])) > 1 \
                or len(cns_snapshot.get("write_layers", [])) > 1 or "hemisphere_nodes" in cns_graph_info:
            blocks = cns_snapshot["edges_by_block"]
            print(f"  cns v93      capacity {model.cns_core.n_nodes} ({cns_snapshot['dead_nodes']} spare) | "
                  f"hemispheres L {cns_snapshot['hemisphere_nodes'][0]} / R {cns_snapshot['hemisphere_nodes'][1]} | "
                  f"edges LL {blocks['LL']:,} RR {blocks['RR']:,} LR {blocks['LR']:,} RL {blocks['RL']:,} | "
                  f"reads {cns_snapshot['read_layers']} writes {cns_snapshot['write_layers']}"
                  f"{' + thinking' if 'thinking_gate_mean_abs' in cns_snapshot else ''}"
                  f"{' | temporal' if cns_snapshot.get('temporal') else ''}", flush=True)
    parameters = model.parameter_report()

    init_provenance: Optional[Dict[str, Any]] = None
    if initialised_from:
        init_provenance = load_initial_weights(
            model, tokenizer, initialised_from,
            payload=init_payload,
            extend_vocab=bool(getattr(args, "extend_vocab", False)),
            seed=int(args.seed),
            grow_layers=grow_layers,
        )
        init_payload = None  # the 121-500 MB payload is not needed past this point
        if vocabulary_provenance is not None:
            init_provenance["vocabulary"] = vocabulary_provenance
        if args.start_step > 0 and frozen_split != init_provenance.get("source_frozen_split"):
            raise ValueError("crash recovery requires the identical frozen split and corpus receipt")
        print(f"  init_from    {initialised_from} "
              f"({init_provenance['source_steps']} prior steps)")
        if init_provenance.get("grown_vocab"):
            grown = init_provenance["grown_vocab"]
            print(f"  grown vocab  {grown['old']} -> {grown['new']} rows (seed {grown['seed']})")
        if init_provenance.get("grown_layers"):
            grown = init_provenance["grown_layers"]
            print(f"  grown depth  {grown['count']} block(s) from layer {grown['first_new_layer']}"
                  f"{' zeroed to identity' if grown['zeroed'] else ' restored from checkpoint'}")
        if init_provenance.get("grown_experts"):
            print(f"  grown moe    {len(init_provenance['grown_experts'])} router tensors padded "
                  f"for {int(getattr(config, 'moe_spare_experts', 0))} spare slots per layer")
    neurogenesis_state = (init_provenance or {}).pop("_neurogenesis_state", None)

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

    decay_groups = split_new_parameter_groups(
        model,
        parameter_groups(model, args.weight_decay, args.decay_mode),
        prefixes=new_parameter_prefixes(config),
        lr_mult=float(getattr(args, "new_param_lr_mult", 1.0)),
    )
    group_max_lr = [args.lr * float(g.get("_lr_mult", 1.0)) for g in decay_groups]
    optimiser = torch.optim.AdamW(
        [
            {
                **{k: v for k, v in g.items() if not k.startswith("_")},
                # Only grafted groups carry a rate; the rest inherit args.lr
                # exactly as before v91.
                **({"lr": args.lr * float(g["_lr_mult"])} if "_lr_mult" in g else {}),
            }
            for g in decay_groups
        ],
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
            optimiser,
            # A scalar when every group shares the rate, exactly as before v91,
            # so an unchanged command builds an unchanged scheduler.
            max_lr=group_max_lr if len(set(group_max_lr)) > 1 else args.lr,
            total_steps=total_steps,
            pct_start=args.pct_start,
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

    # v93 D6: the neurogenesis controller. Built after every restore so it
    # sees the model the loop will train, and only when growth is on, so a
    # v89/v91 command never imports it. Its witness batch is the first
    # --witness_rows dev rows, fixed for the run.
    controller = None
    grow_every = int(getattr(args, "grow_every", 0) or 0)
    if grow_every > 0:
        if args.eval_every <= 0 or grow_every % int(args.eval_every) != 0:
            raise SystemExit(
                f"--grow_every {grow_every} must be a positive multiple of --eval_every "
                f"{args.eval_every}: growth decisions use the statistics of the dev "
                "pass that precedes them, so an event can only sit on an eval step."
            )
        import neurogenesis  # noqa: E402  (source/neurogenesis.py; lazy so v91 commands never need it)

        witness_rows = max(1, int(getattr(args, "witness_rows", 8) or 8))
        witness = (dev_x[:witness_rows].long(), dev_y[:witness_rows].long())
        controller = neurogenesis.NeurogenesisController(
            model,
            neurogenesis.build_settings_from_args(args),
            str(output_dir / "neurogenesis.jsonl"),
            witness,
        )
        if args.start_step > 0 and neurogenesis_state is not None:
            controller.load_state_dict(neurogenesis_state)
        print(f"  neurogenesis every {grow_every} steps | witness {int(witness[0].shape[0])} rows | "
              f"log {output_dir / 'neurogenesis.jsonl'}"
              f"{' | counters restored' if args.start_step > 0 and neurogenesis_state is not None else ''}",
              flush=True)
    growth_watch = growth_telemetry(model) is not None

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
            # The dev pass doubles as the statistics pass for growth (D6):
            # module rates, rate covariance and expert load accumulate only in
            # eval mode between begin/end, so the numbers are exactly the ones
            # the reported dev loss was measured on.
            if controller is not None:
                controller.begin_dev_pass()
            dev_metrics = evaluate(model, dev_x, dev_y, args.eval_batch_size)
            if controller is not None:
                controller.end_dev_pass()
            verbatim = probe_verbatim_rate(model, tokenizer, recall)
            accuracy = None
            if accuracy_probes and (
                step % args.accuracy_every == 0 or step == args.steps
            ):
                accuracy_report = probe_accuracy_report(
                    model, tokenizer, accuracy_probes, max_new_tokens=probe_cap
                )
                accuracy = accuracy_report["accuracy"]
            # One growth event, after every measurement of these weights and
            # before anything is written: the recovery checkpoint then holds
            # the post-event slots, moments and counters, which is what makes
            # a crash resume replay the continuous run exactly.
            event = controller.maybe_grow(step, optimiser) if controller is not None else None
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
            if event is not None:
                entry["neurogenesis"] = event_counts(event)
            if growth_watch:
                entry["growth"] = growth_telemetry(model)
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
                        # v93 D6: the apoptosis counters, so a resumed leg
                        # continues "two consecutive events" where the
                        # crashed one left it. None when growth is off.
                        "neurogenesis_state": (
                            controller.state_dict() if controller is not None else None
                        ),
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

    # v91 mechanism test. A grafted branch can train, move the loss, and still
    # be doing nothing -- v58's thinking-core gate reached 6.4e-4 and closing
    # it changed none of 12,192 predictions. So the selected weights are scored
    # on dev twice: as trained, and with the connectome core's gate forced shut.
    cns_report: Optional[Dict[str, Any]] = None
    core = getattr(model, "cns_core", None)
    if core is not None:
        dev_on = evaluate(model, dev_x, dev_y, args.eval_batch_size)["loss"]
        saved_gate = core.gate.detach().clone()
        with torch.no_grad():
            core.gate.zero_()
        dev_off = evaluate(model, dev_x, dev_y, args.eval_batch_size)["loss"]
        with torch.no_grad():
            core.gate.copy_(saved_gate)
        cns_report = {
            **core.telemetry(),
            "graph": cns_graph_info,
            "dev_loss_core_on": round(float(dev_on), 6),
            "dev_loss_core_off": round(float(dev_off), 6),
            "ablation_cost_nats": round(float(dev_off - dev_on), 6),
            "note": (
                "ablation_cost_nats > 0 means the trained model relies on the "
                "connectome branch; ~0 means the gate never became load-bearing"
            ),
        }
        print(f"  cns ablation dev {dev_on:.5f} with core, {dev_off:.5f} without "
              f"({dev_off - dev_on:+.5f} nats)", flush=True)

    # v93 mechanism tests (the pre-registered readout): every write site and
    # the thinking bond shut at once, the commissure alone, each hemisphere
    # alone, and everything neurogenesis grew returned to its birth state.
    # Same weights, same rows, each a dev-loss difference against the
    # unablated pass on those rows. The v91 block above is left byte-identical
    # (primary gate only, full dev set) so v91 receipts still compare.
    v93_ablations: Optional[Dict[str, Any]] = None
    if core is not None:
        ablation_rows = int(getattr(args, "ablation_rows", 0) or 0)
        v93_ablations = v93_ablation_report(
            model, dev_x, dev_y, args.eval_batch_size, ablation_rows,
            grown=controller is not None,
            reference_loss=cns_report["dev_loss_core_on"] if cns_report else None,
        )
        for name, result in v93_ablations["ablations"].items():
            if "cost_nats" in result:
                print(f"  v93 ablation {name:<11} dev {result['dev_loss']:.5f} "
                      f"({result['cost_nats']:+.5f} nats over {v93_ablations['rows']} rows)", flush=True)

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
    v93_active = any((
        int(getattr(config, "cns_spare_nodes", 0) or 0) > 0,
        len(getattr(config, "cns_read_layers", ()) or ()) > 0,
        len(getattr(config, "cns_write_layers", ()) or ()) > 0,
        bool(getattr(config, "cns_to_thinking", False)),
        bool(getattr(config, "cns_temporal", False)),
        int(getattr(config, "moe_spare_experts", 0) or 0) > 0,
        grow_layers > 0,
        controller is not None,
        bool(getattr(args, "extend_vocab", False)),
    ))
    neurogenesis_report: Optional[Dict[str, Any]] = None
    if controller is not None:
        log_path = output_dir / "neurogenesis.jsonl"
        neurogenesis_report = {
            **controller.summary(),
            "log": str(log_path),
            # The controller only knows the events of this leg; the log holds
            # every leg's (append-only), so a resumed run's count is read here.
            "events_logged": (
                sum(1 for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
                if log_path.exists() else 0
            ),
        }
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
        "architecture": (
            "mimomix_core.MiMoMixModel (v53) + v91 ConnectomeCore + v93 neurogenesis"
            if v93_active and cns_report is not None
            else "mimomix_core.MiMoMixModel (v53) + v93 neurogenesis"
            if v93_active
            else "mimomix_core.MiMoMixModel (v53) + v91 ConnectomeCore"
            if cns_report is not None
            else "mimomix_core.MiMoMixModel (v53), unmodified"
        ),
        "cns_core": cns_report,
        "v93_ablations": v93_ablations,
        "neurogenesis": neurogenesis_report,
        "optimiser_groups": [
            {"params": len(g["params"]), "weight_decay": g.get("weight_decay"),
             "lr_mult": g.get("_lr_mult", 1.0)}
            for g in decay_groups
        ],
        "split": split.report(tokenizer),
        "split_verification": verification,
        "frozen_split": frozen_split,
        "held_out_sentences": split.held_out_sentences,
        "tokenizer": tokenizer.vocabulary_report([a for _, a in split.dev]),
        "config": config.to_dict(),
        # Recomputed on the selected weights, after every growth event.
        # Slot-based growth never changes a tensor's shape, so the total
        # equals the construction-time count; the alive counts that did
        # change are in `growth` (final telemetry) and per eval in `history`.
        "parameters": model.parameter_report(),
        "parameters_at_construction": parameters,
        "growth": growth_telemetry(model),
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
            "new_param_lr_mult": getattr(args, "new_param_lr_mult", 1.0),
            "cns_core": bool(getattr(args, "cns_core", False)),
            "cns_wiring": getattr(args, "cns_wiring", None) if getattr(args, "cns_core", False) else None,
            "cns_graph": getattr(args, "cns_graph", None) if getattr(args, "cns_core", False) else None,
            "repeat_subset_fraction": getattr(args, "repeat_subset_fraction", None),
            "repeat_subset_prob": getattr(args, "repeat_subset_prob", None),
            "mtp_loss_weight_final": getattr(args, "mtp_loss_weight_final", None),
            "mtp_weight_warmup_fraction": getattr(args, "mtp_weight_warmup_fraction", None),
            # The v85 headline. A receipt that does not say what the probe could
            # see cannot be read: a task truncated by the cap and a task the
            # model never learned produce the same 0.00.
            "probe_max_new_tokens": probe_cap,
            # v93 neurogenesis (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md). Every
            # one changes what grew, so two arms differing in any of them are
            # not a matched pair.
            "extend_vocab": bool(getattr(args, "extend_vocab", False)),
            "max_new_tokens_vocab": getattr(args, "max_new_tokens_vocab", None),
            "cns_spare_nodes": getattr(args, "cns_spare_nodes", None),
            "cns_read_layers": list(getattr(args, "cns_read_layers", None) or []) or None,
            "cns_write_layers": list(getattr(args, "cns_write_layers", None) or []) or None,
            "cns_to_thinking": bool(getattr(args, "cns_to_thinking", False)),
            "cns_temporal": bool(getattr(args, "cns_temporal", False)),
            "moe_spare_experts": getattr(args, "moe_spare_experts", None),
            "grow_layers": grow_layers,
            "grow_every": getattr(args, "grow_every", None),
            "grow_modules": getattr(args, "grow_modules", None),
            "grow_edges": getattr(args, "grow_edges", None),
            "grow_taps": getattr(args, "grow_taps", None),
            "grow_experts": bool(getattr(args, "grow_experts", False)),
            "prune_threshold": getattr(args, "prune_threshold", None),
            "witness_rows": getattr(args, "witness_rows", None),
            "ablation_rows": getattr(args, "ablation_rows", None),
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
            "a mismatch raises rather than silently training a wrong embedding "
            "(with --extend_vocab the checkpoint's list may be a prefix instead)"
        ),
    )
    parser.add_argument(
        "--new_param_lr_mult",
        type=float,
        default=1.0,
        help=(
            "peak-LR multiplier for grafted modules (parameters under "
            "NEW_PARAMETER_PREFIXES, i.e. the v91 cns_core). 1.0 (default) "
            "changes nothing; a model with no grafted modules is unaffected"
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
    # v93 neurogenesis (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md D5, D6, D8).
    # Every flag is inert at its default; the capacity flags they act on
    # (--cns_spare_nodes, --moe_spare_experts, --grow_layers ...) are on the
    # shared talk parser because they map to config fields.
    neuro = parser.add_argument_group("v93 neurogenesis")
    neuro.add_argument(
        "--extend_vocab",
        action="store_true",
        help=(
            "take the tokenizer from --init_from and APPEND the new corpus's "
            "tokens after its ids instead of rebuilding by frequency (which "
            "renumbers ids and is refused as a different vocabulary). The tied "
            "embedding grows by mean(old) + 0.1 std(old) N(0,1) rows, seeded "
            "by --seed. digit_tokens/reverse_digits are taken from the "
            "checkpoint and must match the flags"
        ),
    )
    neuro.add_argument(
        "--max_new_tokens_vocab", type=int, default=4000,
        help="cap on the ids --extend_vocab may append (both spacing forms count)",
    )
    neuro.add_argument(
        "--grow_every", type=int, default=0,
        help=(
            "run one neurogenesis event every N steps, from the statistics of "
            "the dev pass at that step; must be a multiple of --eval_every. "
            "0 (default) is off"
        ),
    )
    neuro.add_argument("--grow_modules", type=int, default=0,
                       help="module splits per event (into --cns_spare_nodes slots)")
    neuro.add_argument("--grow_edges", type=int, default=0,
                       help="synapses opened per event at logit -7; at least half cross-hemisphere")
    neuro.add_argument("--grow_taps", type=int, default=0,
                       help="afferent taps and efferent taps opened per event (each)")
    neuro.add_argument("--grow_experts", action="store_true",
                       help="allow expert births into --moe_spare_experts slots")
    neuro.add_argument(
        "--prune_threshold", type=float, default=1e-4,
        help="softplus edge strength below which a grown edge is pruned after 2 consecutive events",
    )
    neuro.add_argument(
        "--witness_rows", type=int, default=8,
        help="dev rows of the fixed witness batch whose loss is measured before and after each event",
    )
    neuro.add_argument(
        "--ablation_rows", type=int, default=0,
        help=(
            "dev rows for the end-of-run v93 ablations (gates, commissure, each "
            "hemisphere, grown elements); each is one dev pass. 0 (default) "
            "uses the whole dev set"
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


def validate_growth_settings(args) -> None:
    """Refuse v93 growth settings that cannot mean what they say.

    Checked before the corpus is read for the same reason the selection
    settings are: a growth schedule that never fires, or a vocabulary
    extension with nothing to extend, should cost seconds, not a run.
    """

    grow_every = int(getattr(args, "grow_every", 0) or 0)
    eval_every = int(getattr(args, "eval_every", 0) or 0)
    if grow_every < 0:
        raise SystemExit(f"--grow_every must be >= 0, got {grow_every}")
    if grow_every > 0 and (eval_every <= 0 or grow_every % eval_every != 0):
        raise SystemExit(
            f"--grow_every {grow_every} must be a positive multiple of --eval_every "
            f"{eval_every}: growth decisions use the statistics of the dev pass "
            "that precedes them, so an event can only sit on an eval step."
        )
    if getattr(args, "extend_vocab", False) and not getattr(args, "init_from", None):
        raise SystemExit("--extend_vocab appends to a checkpoint's vocabulary and needs --init_from")
    if int(getattr(args, "grow_layers", 0) or 0) > 0 and not getattr(args, "init_from", None):
        raise SystemExit(
            "--grow_layers appends identity blocks to a warm-started model and "
            "needs --init_from; to train a deeper model from scratch raise --n_layers"
        )
    for name in ("grow_modules", "grow_edges", "grow_taps", "witness_rows", "ablation_rows",
                 "max_new_tokens_vocab", "cns_spare_nodes", "moe_spare_experts"):
        value = getattr(args, name, None)
        if value is not None and int(value) < 0:
            raise SystemExit(f"--{name} must be >= 0, got {value}")


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
    validate_growth_settings(args)
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
