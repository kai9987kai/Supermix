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
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from mimomix_core import MiMoMixModel


__all__ = [
    "DecodeStats",
    "GenerationResult",
    "AdaptiveDraftController",
    "PrefixKVCache",
    "greedy_generate",
    "speculative_generate",
    "chunked_prefill",
    "chunked_greedy_generate",
    "speculative_tree_verify",
    "assert_greedy_equivalence",
    "hybrid_cache_footprint",
    "quantize_past_kv",
    "dequantize_past_kv",
    "kv_cache_memory_report",
    "quantize_past_kv_int4",
    "dequantize_past_kv_int4",
    "kv_cache_int4_memory_report",
    "quantize_past_kv_mixed_outlier",
    "dequantize_past_kv_mixed_outlier",
    "kv_cache_outlier_memory_report",
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

        The prompt prefill both consumes a trunk forward and produces the first
        generated token.  Decode throughput deliberately excludes that common
        setup cost from *both* sides of the ratio: ``verify_forwards`` does not
        include prefill, so the token produced by prefill must not be counted in
        the numerator either.  Mixing those conventions makes greedy score
        ``N / (N - 1)`` and can put speculation above its theoretical
        ``draft_length + 1`` maximum.
        """

        if self.verify_forwards == 0:
            return 1.0 if self.mode == "greedy" and self.new_tokens > 0 else 0.0
        return self.decoding_tokens / self.verify_forwards

    @property
    def decoding_tokens(self) -> int:
        """Generated tokens attributable to post-prefill decode forwards."""

        prefill_token = 1 if self.prefill_forwards > 0 and self.new_tokens > 0 else 0
        return max(0, int(self.new_tokens) - prefill_token)

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
        payload["decoding_tokens"] = self.decoding_tokens
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


def _sequence_axis(tensor: torch.Tensor) -> int:
    """Which axis of a cache tensor is the time axis.

    Not every layer caches the same rank. Grouped-query attention stores
    ``(B, H, T, D)``; MLA stores its compressed latent as ``(B, T, latent)`` and
    its decoupled rope keys as ``(B, 1, T, pe)``. A rank-4 tensor is
    ``(B, H, T, D)`` so the time axis is 2; a rank-3 tensor is ``(B, T, D)`` so
    it is 1.

    Before v82 this function assumed rank 4 unconditionally and sliced axis 2 of
    every tensor, which for an MLA latent cache trimmed the *latent* dimension
    instead of time -- so ``speculative_generate`` on a ``use_mla`` model raised
    a RuntimeError on the first rejected token. ``greedy_generate`` never trims
    and was unaffected. The rank test here is deliberately structural, not an
    ``isinstance(layer, MultiLatentAttention)`` special case, so any future
    cache layout of either rank works.
    """

    if tensor.ndim == 4:
        return 2
    if tensor.ndim == 3:
        return 1
    raise ValueError(
        f"cannot locate the time axis of a rank-{tensor.ndim} cache tensor "
        f"with shape {tuple(tensor.shape)}"
    )


def trim_past(past: Optional[Sequence], drop: int) -> Optional[PastKV]:
    """Remove the last ``drop`` cached positions from every layer.

    Each tensor is trimmed on its own time axis (see :func:`_sequence_axis`), so
    a hybrid model whose layers cache different ranks -- MLA on the global
    layers, grouped-query attention on the sliding-window ones -- trims
    correctly throughout.
    """

    if past is None or drop <= 0:
        return None if past is None else list(past)
    trimmed: PastKV = []
    for entry in past:
        if entry is None or entry[0].numel() == 0:
            trimmed.append(entry)
            continue
        cut = []
        for tensor in entry:
            axis = _sequence_axis(tensor)
            keep = max(0, tensor.shape[axis] - drop)
            cut.append(tensor.narrow(axis, 0, keep))
        trimmed.append((cut[0], cut[1]))
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

    skip_step_telemetry = isinstance(model, MiMoMixModel)
    step_extra_kwargs: Dict[str, Any] = {"collect_telemetry": False} if skip_step_telemetry else {}

    for _ in range(max_new_tokens):
        if eos_token_id is not None and bool(finished.any()):
            # Keep already-finished batch rows pinned to EOS while unfinished
            # rows continue.  For batch size one the loop exits immediately;
            # for larger batches this gives every row standard stop semantics
            # without returning ragged tensors.
            token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(token, int(eos_token_id)),
                token,
            )
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
            **step_extra_kwargs,
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


class AdaptiveDraftController:
    """EMA-driven dynamic speculative draft-depth controller (v105).

    Scales the active draft depth in ``[0, max_draft]`` according to observed
    prefix acceptance so low-acceptance / high-entropy decoding skips wasted
    draft + rollback compute while high-acceptance regions draft full depth.
    Verification remains exact greedy argmax prefix matching, so emitted
    tokens stay **bit-identical** to plain greedy decoding.
    """

    def __init__(
        self,
        max_draft: int,
        *,
        ema_alpha: float = 0.35,
        probe_interval: int = 4,
    ) -> None:
        self.max_draft = max(0, int(max_draft))
        self.ema_alpha = float(ema_alpha)
        self.probe_interval = max(1, int(probe_interval))
        self.current_draft = self.max_draft
        self.acceptance_ema = 0.5
        self.steps = 0
        self.zero_draft_steps = 0

    def next_budget(self) -> int:
        if self.max_draft <= 0:
            return 0
        if self.current_draft == 0 and self.steps > 0 and (self.steps % self.probe_interval) == 0:
            return 1
        return self.current_draft

    def observe(self, drafted: int, accepted: int) -> None:
        self.steps += 1
        if drafted <= 0:
            self.zero_draft_steps += 1
            return
        rate = float(accepted) / float(drafted)
        self.acceptance_ema = (1.0 - self.ema_alpha) * self.acceptance_ema + self.ema_alpha * rate
        if accepted == drafted:
            self.current_draft = min(self.max_draft, max(1, self.current_draft + 1))
        elif accepted == 0:
            if self.acceptance_ema < 0.15:
                self.current_draft = 0
            else:
                self.current_draft = max(1, self.current_draft - 1)
        else:
            scaled = int(round(self.max_draft * min(1.0, self.acceptance_ema + 0.15)))
            self.current_draft = max(1, min(self.max_draft, scaled))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_draft": int(self.max_draft),
            "final_draft": int(self.current_draft),
            "acceptance_ema": round(float(self.acceptance_ema), 4),
            "steps": int(self.steps),
            "zero_draft_steps": int(self.zero_draft_steps),
        }


@torch.no_grad()
def speculative_generate(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 16,
    eos_token_id: Optional[int] = None,
    thinking_cycles: Optional[int] = None,
    adaptive_thinking: bool = False,
    draft_length: Optional[int] = None,
    adaptive_draft: bool = False,
) -> GenerationResult:
    """Greedy decoding accelerated by the model's own MTP depths.

    Emits exactly what :func:`greedy_generate` emits, using fewer trunk
    forwards whenever the draft is right. Set ``adaptive_draft=True`` (v105)
    to dynamically throttle draft depth during low-acceptance stretches while
    preserving bit-exact greedy equivalence.
    """

    _reject_adaptive(adaptive_thinking)
    model.eval()
    device = input_ids.device
    batch = int(input_ids.shape[0])
    max_draft = len(model.mtp_modules) if draft_length is None else int(draft_length)
    max_draft = max(0, min(max_draft, len(model.mtp_modules)))
    draft_ctrl = AdaptiveDraftController(max_draft) if adaptive_draft else None
    skip_step_telemetry = isinstance(model, MiMoMixModel)
    step_extra_kwargs: Dict[str, Any] = {"collect_telemetry": False} if skip_step_telemetry else {}
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
        remaining = max_new_tokens - stats.new_tokens
        active_max = draft_ctrl.next_budget() if draft_ctrl is not None else max_draft
        draft_budget = min(active_max, max(0, remaining - 1))
        if draft_budget > 0:
            draft = model.propose_draft(trunk_state, token, position=committed_length - 1)
            if draft_budget < draft.shape[1]:
                draft = draft[:, :draft_budget]
        else:
            draft = token.new_zeros((batch, 0))
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
            **step_extra_kwargs,
        )
        stats.verify_forwards += 1
        stats.drafted_tokens += n_draft * batch

        target = step.logits.argmax(dim=-1)  # (B, block_len)
        # Accept the longest prefix that every *unfinished* batch row agrees
        # with. Batching forces a common accept length; per-row divergence just
        # costs speed. Rows that reached EOS inside the block are ignored at
        # later positions and are pinned to EOS in the returned tensor.
        accepted = 0
        if batch == 1 and n_draft > 0:
            block_row = block[0].tolist()
            target_row = target[0].tolist()
            eos_int = int(eos_token_id) if eos_token_id is not None else None
            for index in range(n_draft):
                cand_tok = int(block_row[index + 1])
                if cand_tok != int(target_row[index]):
                    break
                accepted += 1
                if eos_int is not None and cand_tok == eos_int:
                    break
        elif n_draft > 0:
            verification_finished = finished.clone()
            for index in range(n_draft):
                candidate = block[:, index + 1]
                matches = candidate.eq(target[:, index])
                if not bool(matches[~verification_finished].all()):
                    break
                accepted += 1
                if eos_token_id is not None:
                    verification_finished = verification_finished | candidate.eq(int(eos_token_id))
                    if bool(verification_finished.all()):
                        break
        stats.accepted_draft_tokens += accepted * batch
        if draft_ctrl is not None:
            draft_ctrl.observe(n_draft, accepted)

        committed: List[torch.Tensor] = []
        for index in range(accepted):
            candidate = block[:, index + 1 : index + 2]
            if eos_token_id is not None:
                candidate = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(candidate, int(eos_token_id)),
                    candidate,
                )
                finished = finished | candidate.squeeze(1).eq(int(eos_token_id))
            committed.append(candidate)

        # If every row ended on an accepted draft token, greedy decoding would
        # stop there.  Do not append the block's bonus token after EOS.
        bonus: Optional[torch.Tensor] = None
        if not bool(finished.all()):
            bonus = target[:, accepted : accepted + 1]
            if eos_token_id is not None:
                bonus = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(bonus, int(eos_token_id)),
                    bonus,
                )
                finished = finished | bonus.squeeze(1).eq(int(eos_token_id))
            committed.append(bonus)

        emitted.extend(committed)

        committed_length += accepted + 1
        rejected = n_draft - accepted
        if rejected > 0:
            past = trim_past(step.past_key_values, rejected)
        else:
            past = step.past_key_values

        stats.new_tokens += len(committed)
        if bonus is None:
            break
        trunk_state = step.trunk_hidden[:, accepted : accepted + 1]
        token = bonus

    stats.seconds = time.perf_counter() - started
    new_tokens = torch.cat(emitted, dim=1)[:, :max_new_tokens]
    stats.new_tokens = int(new_tokens.shape[1])
    out_telemetry = dict(prefill.telemetry) if isinstance(prefill.telemetry, dict) else {}
    if draft_ctrl is not None:
        out_telemetry["adaptive_draft"] = draft_ctrl.snapshot()
    return GenerationResult(
        sequences=torch.cat([input_ids, new_tokens], dim=1),
        new_tokens=new_tokens,
        stats=stats,
        telemetry=out_telemetry,
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


# ---------------------------------------------------------------------------
# v105 Prefix KV Cache Reuse & Symmetric Int8 KV Compression
# ---------------------------------------------------------------------------


def _clone_past(past: Optional[PastKV]) -> Optional[PastKV]:
    if past is None:
        return None
    cloned: PastKV = []
    for entry in past:
        if entry is None:
            cloned.append(None)
        else:
            cloned.append((entry[0].detach().clone(), entry[1].detach().clone()))
    return cloned


class PrefixKVCache:
    """Bounded LRU prefix KV-cache for zero-recompute shared-prefix prefills (v105).

    Caches prefill states keyed by exact token prefix tuples (batch size 1).
    When a new prompt shares a cached prefix of length ``L`` (``1 <= L < len(prompt)``),
    only the suffix tokens ``prompt[:, L:]`` are processed through the trunk,
    producing **bit-identical** greedy generations with significantly lower TTFT.
    """

    def __init__(self, max_entries: int = 16) -> None:
        self.max_entries = max(1, int(max_entries))
        self._store: "OrderedDict[Tuple[int, ...], Dict[str, Any]]" = OrderedDict()
        self.hits: int = 0
        self.misses: int = 0
        self.tokens_reused: int = 0

    def clear(self) -> None:
        self._store.clear()

    def put(
        self,
        prefix_ids: Sequence[int],
        *,
        past_key_values: PastKV,
        last_logits: torch.Tensor,
        telemetry: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = tuple(int(t) for t in prefix_ids)
        if not key:
            return
        self._store[key] = {
            "past": _clone_past(past_key_values),
            "last_logits": last_logits.detach().clone(),
            "telemetry": dict(telemetry or {}),
        }
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)

    def lookup_longest_prefix(self, prompt_ids: Sequence[int]) -> Optional[Tuple[int, Dict[str, Any]]]:
        tokens = tuple(int(t) for t in prompt_ids)
        best_len = 0
        best_key: Optional[Tuple[int, ...]] = None
        for key in self._store:
            k_len = len(key)
            if k_len <= len(tokens) and k_len > best_len and tokens[:k_len] == key:
                best_len = k_len
                best_key = key
        if best_key is None:
            return None
        self._store.move_to_end(best_key)
        return best_len, self._store[best_key]

    @torch.no_grad()
    def greedy_generate(
        self,
        model: MiMoMixModel,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 16,
        eos_token_id: Optional[int] = None,
        thinking_cycles: Optional[int] = None,
        adaptive_thinking: bool = False,
        cache_prefix_length: Optional[int] = None,
    ) -> GenerationResult:
        """Exact greedy decoding leveraging cached shared prefixes when available."""
        if int(input_ids.shape[0]) != 1:
            return greedy_generate(
                model,
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_token_id,
                thinking_cycles=thinking_cycles,
                adaptive_thinking=adaptive_thinking,
            )

        model.eval()
        prompt_list = input_ids[0].tolist()
        started = time.perf_counter()
        hit = self.lookup_longest_prefix(prompt_list)
        skip_step_telemetry = isinstance(model, MiMoMixModel)
        step_extra: Dict[str, Any] = {"collect_telemetry": False} if skip_step_telemetry else {}

        if hit is not None:
            prefix_len, entry = hit
            self.hits += 1
            self.tokens_reused += prefix_len
            past = _clone_past(entry["past"])
            if prefix_len == len(prompt_list):
                last_logits = entry["last_logits"].to(device=input_ids.device)
                prefill_tele = dict(entry["telemetry"])
            else:
                suffix = input_ids[:, prefix_len:]
                out = model(
                    suffix,
                    past_key_values=past,
                    use_cache=True,
                    thinking_cycles=thinking_cycles,
                    adaptive_thinking=adaptive_thinking,
                    return_mtp=False,
                    past_length=prefix_len,
                )
                past = out.past_key_values
                last_logits = out.logits[:, -1]
                prefill_tele = out.telemetry
            position = len(prompt_list)
        else:
            self.misses += 1
            if cache_prefix_length is not None and 0 < int(cache_prefix_length) < len(prompt_list):
                p_len = int(cache_prefix_length)
                p_out = model(
                    input_ids[:, :p_len],
                    use_cache=True,
                    thinking_cycles=thinking_cycles,
                    adaptive_thinking=adaptive_thinking,
                    return_mtp=False,
                    past_length=0,
                )
                self.put(
                    prompt_list[:p_len],
                    past_key_values=p_out.past_key_values,
                    last_logits=p_out.logits[:, -1],
                    telemetry=p_out.telemetry,
                )
                s_out = model(
                    input_ids[:, p_len:],
                    past_key_values=p_out.past_key_values,
                    use_cache=True,
                    thinking_cycles=thinking_cycles,
                    adaptive_thinking=adaptive_thinking,
                    return_mtp=False,
                    past_length=p_len,
                )
                past = s_out.past_key_values
                last_logits = s_out.logits[:, -1]
                prefill_tele = s_out.telemetry
            else:
                out = model(
                    input_ids,
                    use_cache=True,
                    thinking_cycles=thinking_cycles,
                    adaptive_thinking=adaptive_thinking,
                    return_mtp=False,
                    past_length=0,
                )
                past = out.past_key_values
                last_logits = out.logits[:, -1]
                prefill_tele = out.telemetry
                self.put(
                    prompt_list,
                    past_key_values=past,
                    last_logits=last_logits,
                    telemetry=prefill_tele,
                )
            position = len(prompt_list)

        token = last_logits.argmax(dim=-1, keepdim=True)
        emitted: List[torch.Tensor] = []
        stats = DecodeStats(mode="greedy")
        for _ in range(max_new_tokens):
            emitted.append(token)
            stats.new_tokens += 1
            if eos_token_id is not None and int(token[0, 0].item()) == int(eos_token_id):
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
                **step_extra,
            )
            stats.verify_forwards += 1
            past = step.past_key_values
            position += 1
            token = step.logits[:, -1].argmax(dim=-1, keepdim=True)

        stats.seconds = time.perf_counter() - started
        new_tokens = (
            torch.cat(emitted, dim=1) if emitted else input_ids.new_zeros((1, 0))
        )
        return GenerationResult(
            sequences=torch.cat([input_ids, new_tokens], dim=1),
            new_tokens=new_tokens,
            stats=stats,
            telemetry=prefill_tele,
        )

    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / float(total), 4) if total else 0.0,
            "tokens_reused": self.tokens_reused,
        }


def quantize_past_kv(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Per-token/channel symmetric int8 quantization for ``past_key_values`` (v105).

    Compresses floating-point cache tensors (both rank-4 GQA and rank-3 MLA/memory
    states) into ``int8`` + ``float16`` per-row scales while preserving empty or
    integer entries (such as Engram token-ID tail caches) untouched.
    """
    if past is None:
        return None
    out: List[Optional[Dict[str, Any]]] = []
    for entry in past:
        if entry is None:
            out.append(None)
            continue
        q_pair: List[Dict[str, Any]] = []
        for tensor in entry:
            if tensor.numel() == 0 or not tensor.is_floating_point():
                q_pair.append({"quantized": False, "tensor": tensor.detach().clone()})
                continue
            t_f = tensor.detach().float()
            scale = (t_f.abs().amax(dim=-1, keepdim=True) / 127.0).clamp_min(1e-8)
            q_val = torch.round(t_f / scale).clamp(-127.0, 127.0).to(torch.int8)
            q_pair.append(
                {
                    "quantized": True,
                    "q": q_val,
                    "scale": scale.to(torch.float16),
                    "dtype": tensor.dtype,
                }
            )
        out.append({"pair": q_pair})
    return out


def dequantize_past_kv(
    quantized_past: Optional[Sequence[Optional[Dict[str, Any]]]],
) -> Optional[PastKV]:
    """Reconstruct floating-point ``past_key_values`` from :func:`quantize_past_kv`."""
    if quantized_past is None:
        return None
    restored: PastKV = []
    for entry in quantized_past:
        if entry is None:
            restored.append(None)
            continue
        tensors: List[torch.Tensor] = []
        for item in entry["pair"]:
            if not item.get("quantized"):
                tensors.append(item["tensor"])
            else:
                dt = item["dtype"]
                val = item["q"].float() * item["scale"].float()
                tensors.append(val.to(dtype=dt))
        restored.append((tensors[0], tensors[1]))
    return restored


def kv_cache_memory_report(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    quantized_past: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Measure exact uncompressed vs int8-quantized KV-cache bytes and fidelity."""
    raw_bytes = 0
    quant_bytes = 0
    max_abs_err = 0.0
    cos_vals: List[float] = []

    if past is None:
        return {
            "raw_bytes": 0,
            "quantized_bytes": 0,
            "compression_ratio": 1.0,
            "saved_fraction": 0.0,
            "max_abs_error": 0.0,
            "mean_cosine_similarity": 1.0,
        }

    q_past = quantized_past if quantized_past is not None else quantize_past_kv(past)
    dq_past = dequantize_past_kv(q_past)
    assert q_past is not None and dq_past is not None

    for raw_entry, q_entry, dq_entry in zip(past, q_past, dq_past):
        if raw_entry is None or q_entry is None or dq_entry is None:
            continue
        for raw_t, q_item, dq_t in zip(raw_entry, q_entry["pair"], dq_entry):
            raw_bytes += int(raw_t.numel() * raw_t.element_size())
            if q_item.get("quantized"):
                q_t = q_item["q"]
                s_t = q_item["scale"]
                quant_bytes += int(q_t.numel() * q_t.element_size() + s_t.numel() * s_t.element_size())
                diff = float((raw_t.detach().float() - dq_t.detach().float()).abs().max().item())
                if diff > max_abs_err:
                    max_abs_err = diff
                a_flat = raw_t.detach().float().reshape(-1)
                b_flat = dq_t.detach().float().reshape(-1)
                denom = float(a_flat.norm().item() * b_flat.norm().item())
                if denom > 1e-12:
                    cos_vals.append(float(torch.dot(a_flat, b_flat).item() / denom))
            else:
                t_raw = q_item["tensor"]
                quant_bytes += int(t_raw.numel() * t_raw.element_size())

    ratio = (raw_bytes / float(quant_bytes)) if quant_bytes > 0 else 1.0
    saved = (1.0 - (quant_bytes / float(raw_bytes))) if raw_bytes > 0 else 0.0
    mean_cos = (sum(cos_vals) / float(len(cos_vals))) if cos_vals else 1.0
    return {
        "raw_bytes": int(raw_bytes),
        "quantized_bytes": int(quant_bytes),
        "compression_ratio": round(ratio, 4),
        "saved_fraction": round(saved, 4),
        "max_abs_error": round(max_abs_err, 6),
        "mean_cosine_similarity": round(mean_cos, 6),
    }


# ---------------------------------------------------------------------------
# v106 Chunked Prefill, Packed Int4 KV Compression & Multi-Branch Tree Verify
# ---------------------------------------------------------------------------


@torch.no_grad()
def chunked_prefill(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    *,
    chunk_size: int = 32,
    thinking_cycles: Optional[int] = None,
    adaptive_thinking: bool = False,
    cache_slack: int = 0,
) -> Dict[str, Any]:
    """Prefill ``input_ids`` in bounded chunks to cap peak attention memory (v106).

    Processes the prompt in slices of ``chunk_size`` tokens while threading
    ``past_key_values`` and exact ``past_length``. When ``adaptive_thinking=False``
    the resulting ``last_logits``, ``trunk_hidden_last``, and ``past_key_values``
    are mathematically equivalent to a monolithic prefill while bounding
    intermediate attention matrix size to ``O(chunk_size * seq_len)``.
    """
    step_size = int(chunk_size)
    if step_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    seq_len = int(input_ids.shape[1])
    if seq_len < 1:
        raise ValueError("input_ids must have at least 1 token")

    model.eval()
    past: Optional[PastKV] = None
    pos = 0
    chunks_run = 0
    last_out = None
    is_mimo = isinstance(model, MiMoMixModel)

    while pos < seq_len:
        end = min(seq_len, pos + step_size)
        is_final = end >= seq_len
        extra: Dict[str, Any] = {}
        if is_mimo and not is_final:
            extra["collect_telemetry"] = False
        last_out = model(
            input_ids[:, pos:end],
            past_key_values=past,
            use_cache=True,
            thinking_cycles=thinking_cycles,
            adaptive_thinking=adaptive_thinking,
            return_mtp=False,
            cache_slack=cache_slack,
            past_length=pos,
            **extra,
        )
        past = last_out.past_key_values
        pos = end
        chunks_run += 1

    assert last_out is not None
    return {
        "last_logits": last_out.logits[:, -1:],
        "trunk_hidden_last": (
            last_out.trunk_hidden[:, -1:] if last_out.trunk_hidden is not None else None
        ),
        "past_key_values": past,
        "past_length": seq_len,
        "chunks_executed": chunks_run,
        "chunk_size": step_size,
        "telemetry": last_out.telemetry,
    }


@torch.no_grad()
def chunked_greedy_generate(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 16,
    chunk_size: int = 32,
    eos_token_id: Optional[int] = None,
    thinking_cycles: Optional[int] = None,
    adaptive_thinking: bool = False,
) -> GenerationResult:
    """Exact greedy decoding using bounded-memory :func:`chunked_prefill` (v106)."""
    model.eval()
    device = input_ids.device
    started = time.perf_counter()

    pre = chunked_prefill(
        model,
        input_ids,
        chunk_size=chunk_size,
        thinking_cycles=thinking_cycles,
        adaptive_thinking=adaptive_thinking,
    )
    past = pre["past_key_values"]
    position = int(pre["past_length"])
    token = pre["last_logits"][:, -1].argmax(dim=-1, keepdim=True)

    emitted: List[torch.Tensor] = []
    stats = DecodeStats(mode="chunked_greedy", prefill_forwards=int(pre["chunks_executed"]))
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    step_extra: Dict[str, Any] = {"collect_telemetry": False} if isinstance(model, MiMoMixModel) else {}

    for _ in range(max_new_tokens):
        if eos_token_id is not None and bool(finished.any()):
            token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(token, int(eos_token_id)),
                token,
            )
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
            **step_extra,
        )
        stats.verify_forwards += 1
        past = step.past_key_values
        position += 1
        token = step.logits[:, -1].argmax(dim=-1, keepdim=True)

    stats.seconds = time.perf_counter() - started
    new_tokens = torch.cat(emitted, dim=1) if emitted else input_ids.new_zeros((input_ids.shape[0], 0))
    tele = dict(pre["telemetry"]) if isinstance(pre["telemetry"], dict) else {}
    tele["chunked_prefill"] = {
        "chunk_size": int(pre["chunk_size"]),
        "chunks_executed": int(pre["chunks_executed"]),
    }
    return GenerationResult(
        sequences=torch.cat([input_ids, new_tokens], dim=1),
        new_tokens=new_tokens,
        stats=stats,
        telemetry=tele,
    )


def quantize_past_kv_int4(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    *,
    group_size: int = 32,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Packed symmetric Int4 group-quantization for ``past_key_values`` (v106).

    Packs two signed 4-bit integers in ``[-7, 7]`` (offset by +8 into ``[1, 15]``)
    into each ``uint8`` byte with per-group ``float16`` scales, targeting
    ~5.5x-7.0x compression over FP32 caches while preserving empty and non-float
    entries (e.g. Engram token-ID tail caches) verbatim.
    """
    if past is None:
        return None
    g_size = max(2, int(group_size))
    if g_size % 2 != 0:
        g_size += 1
    out: List[Optional[Dict[str, Any]]] = []
    for entry in past:
        if entry is None:
            out.append(None)
            continue
        q_pair: List[Dict[str, Any]] = []
        for tensor in entry:
            if tensor.numel() == 0 or not tensor.is_floating_point():
                q_pair.append({"quantized": False, "tensor": tensor.detach().clone()})
                continue
            t_f = tensor.detach().float()
            orig_shape = tuple(int(s) for s in t_f.shape)
            last_dim = orig_shape[-1]
            eff_group = min(g_size, max(2, ((last_dim + 1) // 2) * 2))
            pad_rem = (-last_dim) % eff_group
            if pad_rem > 0:
                t_pad = torch.nn.functional.pad(t_f, (0, pad_rem))
            else:
                t_pad = t_f
            n_groups = t_pad.shape[-1] // eff_group
            grouped = t_pad.reshape(*orig_shape[:-1], n_groups, eff_group)
            scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
            q_signed = torch.round(grouped / scale).clamp(-7.0, 7.0).to(torch.int16)
            q_u4 = (q_signed + 8).to(torch.uint8)  # in [1, 15]
            q_flat_last = q_u4.reshape(*orig_shape[:-1], n_groups * eff_group)
            hi = q_flat_last[..., 0::2]
            lo = q_flat_last[..., 1::2]
            packed = (hi << 4) | lo
            q_pair.append(
                {
                    "quantized": True,
                    "bits": 4,
                    "packed": packed,
                    "scale": scale.squeeze(-1).to(torch.float16),
                    "orig_shape": orig_shape,
                    "group_size": eff_group,
                    "dtype": tensor.dtype,
                }
            )
        out.append({"pair": q_pair})
    return out


def dequantize_past_kv_int4(
    quantized_past: Optional[Sequence[Optional[Dict[str, Any]]]],
) -> Optional[PastKV]:
    """Reconstruct floating-point ``past_key_values`` from :func:`quantize_past_kv_int4`."""
    if quantized_past is None:
        return None
    restored: PastKV = []
    for entry in quantized_past:
        if entry is None:
            restored.append(None)
            continue
        tensors: List[torch.Tensor] = []
        for item in entry["pair"]:
            if not item.get("quantized"):
                tensors.append(item["tensor"])
                continue
            packed = item["packed"]
            scale = item["scale"].float()
            orig_shape = item["orig_shape"]
            eff_group = int(item["group_size"])
            hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
            lo = (packed & 0x0F).to(torch.int16) - 8
            unpacked = torch.stack([hi, lo], dim=-1).reshape(
                *orig_shape[:-1], scale.shape[-1], eff_group
            )
            val = (unpacked.float() * scale.unsqueeze(-1)).reshape(
                *orig_shape[:-1], scale.shape[-1] * eff_group
            )
            val = val[..., : orig_shape[-1]].to(dtype=item["dtype"])
            tensors.append(val)
        restored.append((tensors[0], tensors[1]))
    return restored


def kv_cache_int4_memory_report(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    *,
    group_size: int = 32,
) -> Dict[str, Any]:
    """Measure exact uncompressed vs packed Int4 group-quantized KV bytes & fidelity."""
    if past is None:
        return {
            "bits": 4,
            "group_size": int(group_size),
            "raw_bytes": 0,
            "quantized_bytes": 0,
            "compression_ratio": 1.0,
            "saved_fraction": 0.0,
            "max_abs_error": 0.0,
            "mean_cosine_similarity": 1.0,
        }
    q_past = quantize_past_kv_int4(past, group_size=group_size)
    dq_past = dequantize_past_kv_int4(q_past)
    assert q_past is not None and dq_past is not None

    raw_bytes = 0
    quant_bytes = 0
    max_abs_err = 0.0
    cos_vals: List[float] = []

    for raw_entry, q_entry, dq_entry in zip(past, q_past, dq_past):
        if raw_entry is None or q_entry is None or dq_entry is None:
            continue
        for raw_t, q_item, dq_t in zip(raw_entry, q_entry["pair"], dq_entry):
            raw_bytes += int(raw_t.numel() * raw_t.element_size())
            if q_item.get("quantized"):
                p_t = q_item["packed"]
                s_t = q_item["scale"]
                quant_bytes += int(
                    p_t.numel() * p_t.element_size() + s_t.numel() * s_t.element_size()
                )
                diff = float((raw_t.detach().float() - dq_t.detach().float()).abs().max().item())
                if diff > max_abs_err:
                    max_abs_err = diff
                a_flat = raw_t.detach().float().reshape(-1)
                b_flat = dq_t.detach().float().reshape(-1)
                denom = float(a_flat.norm().item() * b_flat.norm().item())
                if denom > 1e-12:
                    cos_vals.append(float(torch.dot(a_flat, b_flat).item() / denom))
            else:
                t_raw = q_item["tensor"]
                quant_bytes += int(t_raw.numel() * t_raw.element_size())

    ratio = (raw_bytes / float(quant_bytes)) if quant_bytes > 0 else 1.0
    saved = (1.0 - (quant_bytes / float(raw_bytes))) if raw_bytes > 0 else 0.0
    mean_cos = (sum(cos_vals) / float(len(cos_vals))) if cos_vals else 1.0
    return {
        "bits": 4,
        "group_size": int(group_size),
        "raw_bytes": int(raw_bytes),
        "quantized_bytes": int(quant_bytes),
        "compression_ratio": round(ratio, 4),
        "saved_fraction": round(saved, 4),
        "max_abs_error": round(max_abs_err, 6),
        "mean_cosine_similarity": round(mean_cos, 6),
    }


def quantize_past_kv_mixed_outlier(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    *,
    group_size: int = 32,
    outlier_channels: int = 2,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Outlier-protected mixed-precision Int4 + FP16 KV-cache compression (v107).

    Inspired by KIVI / Atom / QuaRot heavy-tailed KV quantization:
    identifies the top-``outlier_channels`` highest-RMS feature channels in each
    floating-point cache tensor, stores those channels in exact ``float16``, and
    packs the well-conditioned residual tensor into symmetric 4-bit nibbles
    (``uint8``) with tight per-group ``float16`` scales. Non-float tensors (e.g.
    Engram token-ID histories) and empty companions are preserved verbatim.
    """
    if past is None:
        return None
    if int(outlier_channels) < 0:
        raise ValueError(f"outlier_channels must be >= 0, got {outlier_channels}")
    g_size = max(2, int(group_size))
    if g_size % 2 != 0:
        g_size += 1
    out: List[Optional[Dict[str, Any]]] = []
    for entry in past:
        if entry is None:
            out.append(None)
            continue
        q_pair: List[Dict[str, Any]] = []
        for tensor in entry:
            if tensor.numel() == 0 or not tensor.is_floating_point():
                q_pair.append({"quantized": False, "tensor": tensor.detach().clone()})
                continue
            t_f = tensor.detach().float()
            orig_shape = tuple(int(s) for s in t_f.shape)
            last_dim = orig_shape[-1]
            k_out = min(int(outlier_channels), max(0, last_dim - 2))

            outlier_idx: Optional[torch.Tensor] = None
            outlier_vals: Optional[torch.Tensor] = None
            residual = t_f
            if k_out > 0:
                flat_rows = t_f.reshape(-1, last_dim)
                chan_rms = flat_rows.pow(2).mean(dim=0).sqrt()
                _, top_idx = torch.topk(chan_rms, k_out, dim=0, largest=True, sorted=True)
                outlier_idx = top_idx.to(torch.int16)
                outlier_vals = t_f.index_select(-1, top_idx).to(torch.float16)
                residual = t_f.clone()
                residual.index_fill_(-1, top_idx, 0.0)

            eff_group = min(g_size, max(2, ((last_dim + 1) // 2) * 2))
            pad_rem = (-last_dim) % eff_group
            t_pad = torch.nn.functional.pad(residual, (0, pad_rem)) if pad_rem > 0 else residual
            n_groups = t_pad.shape[-1] // eff_group
            grouped = t_pad.reshape(*orig_shape[:-1], n_groups, eff_group)
            scale = (grouped.abs().amax(dim=-1, keepdim=True) / 7.0).clamp_min(1e-8)
            q_signed = torch.round(grouped / scale).clamp(-7.0, 7.0).to(torch.int16)
            q_u4 = (q_signed + 8).to(torch.uint8)
            q_flat_last = q_u4.reshape(*orig_shape[:-1], n_groups * eff_group)
            packed = (q_flat_last[..., 0::2] << 4) | q_flat_last[..., 1::2]
            q_pair.append(
                {
                    "quantized": True,
                    "bits": 4,
                    "mixed_outlier": True,
                    "packed": packed,
                    "scale": scale.squeeze(-1).to(torch.float16),
                    "outlier_idx": outlier_idx,
                    "outlier_vals": outlier_vals,
                    "orig_shape": orig_shape,
                    "group_size": eff_group,
                    "dtype": tensor.dtype,
                }
            )
        out.append({"pair": q_pair})
    return out


def dequantize_past_kv_mixed_outlier(
    quantized_past: Optional[Sequence[Optional[Dict[str, Any]]]],
) -> Optional[PastKV]:
    """Reconstruct ``past_key_values`` from :func:`quantize_past_kv_mixed_outlier`."""
    if quantized_past is None:
        return None
    restored: PastKV = []
    for entry in quantized_past:
        if entry is None:
            restored.append(None)
            continue
        tensors: List[torch.Tensor] = []
        for item in entry["pair"]:
            if not item.get("quantized"):
                tensors.append(item["tensor"])
                continue
            packed = item["packed"]
            scale = item["scale"].float()
            orig_shape = item["orig_shape"]
            eff_group = int(item["group_size"])
            hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
            lo = (packed & 0x0F).to(torch.int16) - 8
            unpacked = torch.stack([hi, lo], dim=-1).reshape(
                *orig_shape[:-1], scale.shape[-1], eff_group
            )
            val = (unpacked.float() * scale.unsqueeze(-1)).reshape(
                *orig_shape[:-1], scale.shape[-1] * eff_group
            )[..., : orig_shape[-1]]
            outlier_idx = item.get("outlier_idx")
            outlier_vals = item.get("outlier_vals")
            if outlier_idx is not None and outlier_vals is not None and outlier_idx.numel() > 0:
                idx_long = outlier_idx.to(dtype=torch.long, device=val.device)
                val = val.clone()
                val.index_copy_(-1, idx_long, outlier_vals.to(dtype=val.dtype, device=val.device))
            tensors.append(val.to(dtype=item["dtype"]))
        restored.append((tensors[0], tensors[1]))
    return restored


def kv_cache_outlier_memory_report(
    past: Optional[Sequence[Optional[Tuple[torch.Tensor, torch.Tensor]]]],
    *,
    group_size: int = 32,
    outlier_channels: int = 2,
) -> Dict[str, Any]:
    """Measure exact bytes, compression ratio, and cosine fidelity for outlier-mixed Int4 KV."""
    if past is None:
        return {
            "bits": 4,
            "mixed_outlier": True,
            "group_size": int(group_size),
            "outlier_channels": int(outlier_channels),
            "raw_bytes": 0,
            "quantized_bytes": 0,
            "compression_ratio": 1.0,
            "saved_fraction": 0.0,
            "max_abs_error": 0.0,
            "mean_cosine_similarity": 1.0,
        }
    q_past = quantize_past_kv_mixed_outlier(
        past, group_size=group_size, outlier_channels=outlier_channels
    )
    dq_past = dequantize_past_kv_mixed_outlier(q_past)
    assert q_past is not None and dq_past is not None

    raw_bytes = 0
    quant_bytes = 0
    max_abs_err = 0.0
    cos_vals: List[float] = []

    for raw_entry, q_entry, dq_entry in zip(past, q_past, dq_past):
        if raw_entry is None or q_entry is None or dq_entry is None:
            continue
        for raw_t, q_item, dq_t in zip(raw_entry, q_entry["pair"], dq_entry):
            raw_bytes += int(raw_t.numel() * raw_t.element_size())
            if q_item.get("quantized"):
                p_t = q_item["packed"]
                s_t = q_item["scale"]
                o_i = q_item.get("outlier_idx")
                o_v = q_item.get("outlier_vals")
                item_bytes = int(
                    p_t.numel() * p_t.element_size() + s_t.numel() * s_t.element_size()
                )
                if o_i is not None and o_v is not None:
                    item_bytes += int(
                        o_i.numel() * o_i.element_size() + o_v.numel() * o_v.element_size()
                    )
                quant_bytes += item_bytes
                diff = float((raw_t.detach().float() - dq_t.detach().float()).abs().max().item())
                if diff > max_abs_err:
                    max_abs_err = diff
                a_flat = raw_t.detach().float().reshape(-1)
                b_flat = dq_t.detach().float().reshape(-1)
                denom = float(a_flat.norm().item() * b_flat.norm().item())
                if denom > 1e-12:
                    cos_vals.append(float(torch.dot(a_flat, b_flat).item() / denom))
            else:
                t_raw = q_item["tensor"]
                quant_bytes += int(t_raw.numel() * t_raw.element_size())

    ratio = (raw_bytes / float(quant_bytes)) if quant_bytes > 0 else 1.0
    saved = (1.0 - (quant_bytes / float(raw_bytes))) if raw_bytes > 0 else 0.0
    mean_cos = (sum(cos_vals) / float(len(cos_vals))) if cos_vals else 1.0
    return {
        "bits": 4,
        "mixed_outlier": True,
        "group_size": int(group_size),
        "outlier_channels": int(outlier_channels),
        "raw_bytes": int(raw_bytes),
        "quantized_bytes": int(quant_bytes),
        "compression_ratio": round(ratio, 4),
        "saved_fraction": round(saved, 4),
        "max_abs_error": round(max_abs_err, 6),
        "mean_cosine_similarity": round(mean_cos, 6),
    }


@torch.no_grad()
def speculative_tree_verify(
    model: MiMoMixModel,
    input_ids: torch.Tensor,
    candidate_branches: Sequence[Sequence[int]],
    *,
    thinking_cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """Verify multiple candidate continuation branches over one shared prompt KV cache (v106).

    Computes exact target-model log-probabilities and exact greedy-match prefix
    lengths for every branch without re-encoding the prompt prefix.
    """
    if int(input_ids.shape[0]) != 1:
        raise ValueError("speculative_tree_verify expects batch size 1 prompt input_ids")
    if not candidate_branches:
        raise ValueError("candidate_branches must be non-empty")

    model.eval()
    device = input_ids.device
    prefill = model(
        input_ids,
        use_cache=True,
        thinking_cycles=thinking_cycles,
        adaptive_thinking=False,
        return_mtp=False,
        past_length=0,
    )
    prompt_len = int(input_ids.shape[1])
    first_log_probs = torch.nn.functional.log_softmax(prefill.logits[0, -1].float(), dim=-1)
    first_greedy = int(first_log_probs.argmax().item())
    step_extra: Dict[str, Any] = {"collect_telemetry": False} if isinstance(model, MiMoMixModel) else {}

    branch_reports: List[Dict[str, Any]] = []
    for idx, raw_branch in enumerate(candidate_branches):
        toks = [int(t) for t in raw_branch]
        if not toks:
            raise ValueError(f"candidate_branches[{idx}] must be non-empty")
        total_lp = float(first_log_probs[toks[0]].item())
        greedy_prefix = 1 if toks[0] == first_greedy else 0
        still_greedy = toks[0] == first_greedy

        if len(toks) > 1:
            branch_tensor = torch.tensor([toks[:-1]], dtype=torch.long, device=device)
            past_copy = _clone_past(prefill.past_key_values)
            out = model(
                branch_tensor,
                past_key_values=past_copy,
                use_cache=False,
                thinking_cycles=thinking_cycles,
                adaptive_thinking=False,
                return_mtp=False,
                past_length=prompt_len,
                **step_extra,
            )
            step_lps = torch.nn.functional.log_softmax(out.logits[0].float(), dim=-1)
            step_argmax = step_lps.argmax(dim=-1).tolist()
            for pos_i in range(len(toks) - 1):
                nxt = toks[pos_i + 1]
                total_lp += float(step_lps[pos_i, nxt].item())
                if still_greedy and nxt == int(step_argmax[pos_i]):
                    greedy_prefix += 1
                else:
                    still_greedy = False

        branch_reports.append(
            {
                "branch_index": idx,
                "tokens": toks,
                "length": len(toks),
                "cumulative_logprob": round(total_lp, 6),
                "mean_logprob": round(total_lp / float(len(toks)), 6),
                "greedy_accepted_prefix": greedy_prefix,
                "exact_greedy_match": greedy_prefix == len(toks),
            }
        )

    best_by_greedy = max(
        branch_reports,
        key=lambda b: (b["greedy_accepted_prefix"], b["mean_logprob"]),
    )
    best_by_logprob = max(branch_reports, key=lambda b: b["mean_logprob"])
    return {
        "prompt_length": prompt_len,
        "prompt_prefill_forwards": 1,
        "branches_evaluated": len(branch_reports),
        "best_greedy_branch_index": best_by_greedy["branch_index"],
        "best_logprob_branch_index": best_by_logprob["branch_index"],
        "branches": branch_reports,
    }
