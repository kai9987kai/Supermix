"""MTP self-speculative decoding for MiMoMix.

The Multi-Token Prediction depths trained in :mod:`mimomix_core` are reused at
inference as a *draft model that costs one block each*, which is the trick MiMo
uses to roughly triple output throughput without a second model in memory.

The loop implemented here is the standard draft/verify schedule specialised to
greedy decoding, where acceptance has an exact form:

    accept a drafted token iff it equals the trunk's own argmax at that position

On the first mismatch the trunk's argmax is emitted instead and the rest of the
draft is discarded. This makes the emitted sequence **bit-identical** to plain
autoregressive greedy decoding -- speculation buys throughput and changes
nothing else. :func:`assert_greedy_equivalence` checks exactly that, and the
test-suite runs it on random models.

Two implementation details are load-bearing and easy to get wrong:

* **Cache rollback.** Rejecting ``r`` tokens means the KV entries written for
  them must go. Under sliding-window attention a cache trimmed to exactly
  ``window`` has already dropped keys that rollback brings back into range, so
  the decoder asks the model for ``cache_slack = draft_length`` extra entries.
* **Block-independence.** Verification feeds several positions at once. That is
  only equivalent to one-at-a-time decoding if the model's per-position output
  does not depend on what else is in the block. The adaptive thinking core
  makes a *batch-level* halting decision, so speculative decoding refuses to
  run with ``adaptive_thinking=True`` rather than quietly breaking the
  guarantee.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from mimomix_core import MiMoMixModel


__all__ = [
    "DecodeStats",
    "GenerationResult",
    "greedy_generate",
    "speculative_generate",
    "assert_greedy_equivalence",
    "hybrid_cache_footprint",
    "trim_past",
]


PastKV = List[Optional[Tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class DecodeStats:
    """Throughput accounting for one generation call."""

    mode: str
    new_tokens: int = 0
    #: forward passes through the full trunk (excluding the prefill)
    verify_forwards: int = 0
    prefill_forwards: int = 1
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    seconds: float = 0.0

    @property
    def acceptance_length(self) -> float:
        """Mean tokens committed per trunk forward -- the headline MTP number.

        Plain greedy decoding scores exactly ``1.0``. MiMo reports up to 3.6
        with three MTP layers on a real checkpoint; an untrained toy model will
        score near 1 because its drafts are noise, and that is the correct
        behaviour, not a bug.
        """

        if self.verify_forwards == 0:
            return 0.0
        return self.new_tokens / self.verify_forwards

    @property
    def acceptance_rate(self) -> float:
        """Fraction of speculative tokens that survived verification."""

        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.drafted_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.seconds <= 0.0:
            return 0.0
        return self.new_tokens / self.seconds

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["acceptance_length"] = round(self.acceptance_length, 4)
        payload["acceptance_rate"] = round(self.acceptance_rate, 4)
        payload["tokens_per_second"] = round(self.tokens_per_second, 3)
        return payload


@dataclass
class GenerationResult:
    sequences: torch.Tensor
    new_tokens: torch.Tensor
    stats: DecodeStats
    telemetry: Dict[str, object] = field(default_factory=dict)


def trim_past(past: Optional[Sequence], drop: int) -> Optional[PastKV]:
    """Remove the last ``drop`` cached positions from every layer."""

    if past is None or drop <= 0:
        return None if past is None else list(past)
    trimmed: PastKV = []
    for entry in past:
        if entry is None or entry[0].numel() == 0:
            trimmed.append(entry)
            continue
        keys, values = entry
        length = keys.shape[2]
        keep = max(0, length - drop)
        trimmed.append((keys[:, :, :keep], values[:, :, :keep]))
    return trimmed


def _reject_adaptive(adaptive_thinking: bool) -> None:
    if adaptive_thinking:
        raise ValueError(
            "speculative decoding requires a block-independent target model; "
            "the adaptive thinking core halts on a batch-level statistic, so "
            "pass adaptive_thinking=False (a fixed cycle budget) instead"
        )


@torch.no_grad()
def greedy_generate(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    thinking_cycles: Optional[int] = None,
    adaptive_thinking: bool = False,
) -> GenerationResult:
    """Reference one-token-at-a-time greedy decoding.

    This is the correctness oracle for :func:`speculative_generate`.
    """

    model.eval()
    device = input_ids.device
    started = time.perf_counter()

    out = model(
        input_ids,
        use_cache=True,
        thinking_cycles=thinking_cycles,
        adaptive_thinking=adaptive_thinking,
        return_mtp=False,
        past_length=0,
    )
    past = out.past_key_values
    position = int(input_ids.shape[1])
    token = out.logits[:, -1].argmax(dim=-1, keepdim=True)

    emitted: List[torch.Tensor] = []
    stats = DecodeStats(mode="greedy")
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        emitted.append(token)
        stats.new_tokens += 1
        if eos_token_id is not None:
            finished = finished | token.squeeze(1).eq(int(eos_token_id))
            if bool(finished.all()):
                break
        if stats.new_tokens >= max_new_tokens:
            break
        step = model(
            token,
            past_key_values=past,
            use_cache=True,
            thinking_cycles=thinking_cycles,
            adaptive_thinking=adaptive_thinking,
            return_mtp=False,
            past_length=position,
        )
        stats.verify_forwards += 1
        past = step.past_key_values
        position += 1
        token = step.logits[:, -1].argmax(dim=-1, keepdim=True)

    stats.seconds = time.perf_counter() - started
    new_tokens = torch.cat(emitted, dim=1) if emitted else input_ids.new_zeros((input_ids.shape[0], 0))
    return GenerationResult(
        sequences=torch.cat([input_ids, new_tokens], dim=1),
        new_tokens=new_tokens,
        stats=stats,
        telemetry=out.telemetry,
    )


@torch.no_grad()
def speculative_generate(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    thinking_cycles: Optional[int] = None,
    adaptive_thinking: bool = False,
    draft_length: Optional[int] = None,
) -> GenerationResult:
    """Greedy decoding accelerated by the model's own MTP depths.

    Emits exactly what :func:`greedy_generate` emits, using fewer trunk
    forwards whenever the draft is right.
    """

    _reject_adaptive(adaptive_thinking)
    model.eval()
    device = input_ids.device
    batch = int(input_ids.shape[0])
    max_draft = len(model.mtp_modules) if draft_length is None else int(draft_length)
    max_draft = max(0, min(max_draft, len(model.mtp_modules)))
    started = time.perf_counter()

    prefill = model(
        input_ids,
        use_cache=True,
        thinking_cycles=thinking_cycles,
        adaptive_thinking=False,
        return_mtp=False,
        cache_slack=max_draft,
        past_length=0,
    )
    past = prefill.past_key_values
    committed_length = int(input_ids.shape[1])

    # The trunk's own argmax at the last prompt position: exact, not a draft.
    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    trunk_state = prefill.trunk_hidden[:, -1:]

    emitted: List[torch.Tensor] = [token]
    stats = DecodeStats(mode="speculative")
    stats.new_tokens = 1
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    if eos_token_id is not None:
        finished = token.squeeze(1).eq(int(eos_token_id))

    while stats.new_tokens < max_new_tokens and not bool(finished.all()):
        draft = model.propose_draft(trunk_state, token, position=committed_length - 1)
        if max_draft < draft.shape[1]:
            draft = draft[:, :max_draft]
        block = torch.cat([token, draft], dim=1) if draft.numel() else token
        block_len = int(block.shape[1])
        n_draft = block_len - 1

        step = model(
            block,
            past_key_values=past,
            use_cache=True,
            thinking_cycles=thinking_cycles,
            adaptive_thinking=False,
            return_mtp=False,
            cache_slack=max_draft,
            past_length=committed_length,
        )
        stats.verify_forwards += 1
        stats.drafted_tokens += n_draft * batch

        target = step.logits.argmax(dim=-1)  # (B, block_len)
        # Accept the longest prefix that every batch row agrees with. Batching
        # forces a common accept length; per-row divergence just costs speed.
        accepted = 0
        for index in range(n_draft):
            if bool(torch.equal(block[:, index + 1], target[:, index])):
                accepted += 1
            else:
                break
        stats.accepted_draft_tokens += accepted * batch

        for index in range(accepted):
            emitted.append(block[:, index + 1 : index + 2])
        bonus = target[:, accepted : accepted + 1]
        emitted.append(bonus)

        committed_length += accepted + 1
        rejected = n_draft - accepted
        if rejected > 0:
            past = trim_past(step.past_key_values, rejected)
        else:
            past = step.past_key_values

        trunk_state = step.trunk_hidden[:, accepted : accepted + 1]
        token = bonus
        stats.new_tokens += accepted + 1
        if eos_token_id is not None:
            for index in range(accepted + 1):
                offset = len(emitted) - (accepted + 1) + index
                finished = finished | emitted[offset].squeeze(1).eq(int(eos_token_id))

    stats.seconds = time.perf_counter() - started
    new_tokens = torch.cat(emitted, dim=1)[:, :max_new_tokens]
    stats.new_tokens = int(new_tokens.shape[1])
    return GenerationResult(
        sequences=torch.cat([input_ids, new_tokens], dim=1),
        new_tokens=new_tokens,
        stats=stats,
        telemetry=prefill.telemetry,
    )


def assert_greedy_equivalence(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    thinking_cycles: Optional[int] = None,
) -> Dict[str, object]:
    """Run both decoders and require identical output. Raises on divergence."""

    reference = greedy_generate(
        model, input_ids, max_new_tokens=max_new_tokens, thinking_cycles=thinking_cycles
    )
    fast = speculative_generate(
        model, input_ids, max_new_tokens=max_new_tokens, thinking_cycles=thinking_cycles
    )
    if reference.new_tokens.shape != fast.new_tokens.shape or not torch.equal(
        reference.new_tokens, fast.new_tokens
    ):
        raise AssertionError(
            "speculative decoding diverged from greedy decoding\n"
            f"  greedy:      {reference.new_tokens.tolist()}\n"
            f"  speculative: {fast.new_tokens.tolist()}"
        )
    return {
        "tokens": int(reference.new_tokens.shape[1]),
        "greedy_forwards": reference.stats.verify_forwards,
        "speculative_forwards": fast.stats.verify_forwards,
        "acceptance_length": round(fast.stats.acceptance_length, 4),
        "acceptance_rate": round(fast.stats.acceptance_rate, 4),
        "forward_reduction": round(
            1.0 - (fast.stats.verify_forwards / max(1, reference.stats.verify_forwards)), 4
        ),
    }


def hybrid_cache_footprint(model: MiMoMixModel, sequence_length: int) -> Dict[str, object]:
    """KV-cache entries a hybrid layout holds versus an all-global one.

    This is the arithmetic behind the "hybrid attention shrinks the KV cache"
    claim, evaluated for *this* model's layout. It counts cache entries, not
    bytes, and assumes the cache is already at steady state.
    """

    window = int(model.config.sliding_window)
    per_layer: List[int] = []
    for kind in model.layout:
        per_layer.append(sequence_length if kind == "global" else min(window, sequence_length))
    hybrid_total = sum(per_layer)
    dense_total = sequence_length * len(model.layout)
    return {
        "sequence_length": int(sequence_length),
        "sliding_window": window,
        "layout": list(model.layout),
        "per_layer_entries": per_layer,
        "hybrid_entries": int(hybrid_total),
        "all_global_entries": int(dense_total),
        "reduction_factor": round(dense_total / hybrid_total, 4) if hybrid_total else 0.0,
        "saved_fraction": round(1.0 - (hybrid_total / dense_total), 4) if dense_total else 0.0,
    }
