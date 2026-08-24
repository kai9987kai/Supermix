"""v59: measure what each v53 mechanism actually *does* to held-out loss.

V58 ran a thinking-core ablation by training two arms and differencing their
tier losses. It reported deltas between 0.0006 and 0.007 nats, called them "no
measurable effect", and said in the same document that this sits below a noise
floor which "has not itself been quantified".

The floor was the wrong thing to chase. The mechanism is *inert*: the whole
recursive core is gated through one scalar, ``ThinkingCore.residual_scale``,
which ``mimomix_core.py`` initialises to exactly ``0.0`` and which reaches only
6.410e-04 after 1,000 training steps. Forcing that scalar to zero changes
held-out loss by ~1e-07 nats and leaves every argmax identical. V58's two arms
were therefore functionally the same model, and its deltas measure run-to-run
variance rather than the mechanism.

An ablation that retrains cannot tell those two cases apart, because a retrain
moves every weight. This module measures the counterfactual directly instead:
take one trained checkpoint, intervene on one mechanism, and re-score the same
tokens. Everything else is held bit-identical by construction.

Three properties make the verdicts falsifiable rather than asserted:

* **The threshold is measured, not chosen.** ``numerical_noise_floor`` re-scores
  the identical tokens under different batch groupings. Float addition is not
  associative, so this returns the loss difference attributable purely to
  arithmetic order. A mechanism whose causal effect is smaller than that is
  inert in the strongest available sense -- it moves the loss less than the
  order of summation does.
* **The instrument self-checks.** ``IDENTITY`` is an intervention that rewires
  the routing path through a reimplementation and changes nothing. If it fails
  to reproduce the baseline bit-exactly, the harness raises, because a patch
  that silently alters behaviour would contaminate every other verdict.
* **It reports non-effects as results.** ``mtp`` is expected to be inert at
  evaluation time; recording that is how the audit shows it can find a null.

Nothing here modifies ``mimomix_core``. Interventions are context managers that
patch and restore, so importing this module changes no behaviour anywhere.

    python source/mechanism_causality.py --checkpoint output/v58_full/v58_full.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import mimomix_text as text_utils  # noqa: E402
from mimomix_core import (  # noqa: E402
    MiMoMixConfig,
    MiMoMixModel,
    SparseMoEFeedForward,
)

RECEIPT_SCHEMA = "supermix-v59-mechanism-causality-v1"

#: Verdicts. A mechanism is ``active`` if it changes at least one of the model's
#: argmax predictions **or** moves held-out loss by at least
#: :data:`RELEVANCE_SCALE`; it is ``inert`` only when both tests fail.
#:
#: Neither test alone is sufficient, and both failure modes are real. The
#: numerical floor is tight enough (~4e-09 nats) that a mechanism can sit 24x
#: above it and still change nothing the model says, so a floor-based rule calls
#: dead mechanisms live. Conversely an untrained or degenerate model has a
#: near-constant argmax that no perturbation disturbs, so a decision-only rule
#: calls live mechanisms dead -- measured on a randomly initialised model, a
#: fully open thinking core moves loss by 2.4e-03 nats and changes zero
#: decisions. Requiring both to fail before declaring inertness is what makes
#: the verdict survive both cases.
INERT = "inert"
ACTIVE = "active"

#: The smallest arm-to-arm delta v58 reported and interpreted as a result, in
#: nats. Used as the relevance scale, so "inert" means "moves held-out loss by
#: less than the smallest effect this project has ever published as a finding"
#: -- a reference drawn from the repo's own history rather than a chosen epsilon.
V58_SMALLEST_REPORTED_DELTA = 5.9e-4
RELEVANCE_SCALE = V58_SMALLEST_REPORTED_DELTA

#: Scoring options an intervention may flip. Kept as module state rather than a
#: :func:`score` argument so an intervention can change *how* the forward runs,
#: not only what the weights are.
_SCORE_OPTIONS: Dict[str, Any] = {"return_mtp": False}


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


@dataclass
class ScoreResult:
    """Held-out loss plus the decisions that produced it."""

    loss: float
    tokens: int
    argmax: torch.Tensor

    def agreement_with(self, other: "ScoreResult") -> float:
        if self.argmax.shape != other.argmax.shape:
            raise ValueError("argmax shapes differ; the two scores are not comparable")
        return float((self.argmax == other.argmax).float().mean().item())


@torch.no_grad()
def score(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
) -> ScoreResult:
    """Mean cross-entropy over every supervised token, plus the argmax decisions.

    The logits are shifted against the labels exactly as
    ``train_mimomix_talk.evaluate`` does -- position ``t`` predicts token
    ``t + 1`` -- and ``return_mtp=False`` matches it too, so this reproduces the
    published tier losses rather than measuring a different quantity that merely
    responds to the same interventions.

    The reduction is ``sum`` divided by the true token count rather than a mean
    of per-batch means, so the result does not depend on how the rows happen to
    divide into batches -- except through float associativity, which is exactly
    what :func:`numerical_noise_floor` measures.
    """

    was_training = model.training
    model.eval()
    total = 0.0
    counted = 0
    decisions: List[torch.Tensor] = []
    try:
        for start in range(0, inputs.shape[0], batch_size):
            chunk_x = inputs[start : start + batch_size]
            chunk_y = labels[start : start + batch_size]
            logits = model(chunk_x, return_mtp=_SCORE_OPTIONS["return_mtp"]).logits[:, :-1]
            targets = chunk_y[:, 1:]
            decisions.append(logits.argmax(dim=-1))
            if int((targets != -100).sum().item()) == 0:
                continue
            total += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                ).item()
            )
            counted += int((targets != -100).sum().item())
    finally:
        model.train(was_training)

    if counted == 0:
        raise ValueError("no supervised tokens in the evaluation set")
    return ScoreResult(loss=total / counted, tokens=counted, argmax=torch.cat(decisions))


def numerical_noise_floor(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_sizes: Sequence[int] = (8, 12, 16),
) -> Dict[str, Any]:
    """The loss spread caused by arithmetic order alone.

    The same tokens are scored under several batch groupings. Every grouping is
    mathematically identical -- a sum over the same terms -- so any difference
    is float non-associativity. That spread is the smallest effect this rig can
    resolve, and it is a *lower* bound on the true run-to-run noise floor: a
    retrain also moves every weight, which is strictly larger. A mechanism below
    this bound is inert without needing the larger bound to be measured.
    """

    losses = {int(size): score(model, inputs, labels, batch_size=size).loss for size in batch_sizes}
    values = list(losses.values())
    floor = max(values) - min(values)
    return {
        "losses_by_batch_size": {str(k): v for k, v in losses.items()},
        "floor_nats": floor,
        "note": (
            "spread from float associativity only; a lower bound on the "
            "seed-to-seed floor, which is larger and is not measured here"
        ),
    }


# --------------------------------------------------------------------------
# interventions
# --------------------------------------------------------------------------


@dataclass
class Intervention:
    """One counterfactual: perturb a mechanism, score, restore."""

    name: str
    description: str
    apply: Callable[[MiMoMixModel], Callable[[], None]]
    expected: str = ""
    requires: Optional[Callable[[MiMoMixModel], bool]] = None

    def available(self, model: MiMoMixModel) -> bool:
        return True if self.requires is None else bool(self.requires(model))


@contextmanager
def applied(model: MiMoMixModel, intervention: Intervention) -> Iterator[None]:
    restore = intervention.apply(model)
    try:
        yield
    finally:
        restore()


def _has_thinking_core(model: MiMoMixModel) -> bool:
    return getattr(model, "thinking_core", None) is not None


def _moe_layers(model: MiMoMixModel) -> List[SparseMoEFeedForward]:
    return [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]


def _thinking_core_off(model: MiMoMixModel) -> Callable[[], None]:
    """Close the gate the entire recursive core is multiplied by."""

    core = model.thinking_core
    saved = core.residual_scale.data.clone()

    def restore() -> None:
        core.residual_scale.data.copy_(saved)

    core.residual_scale.data.zero_()
    return restore


def _routing_forward(mode: str) -> Callable[[SparseMoEFeedForward, torch.Tensor], torch.Tensor]:
    """Rebuild :meth:`SparseMoEFeedForward.forward` with the expert choice swapped.

    This mirrors the original computation exactly and changes only how
    ``expert_indices`` is chosen. ``mode='identity'`` keeps the real choice and
    exists so the harness can prove this reimplementation is faithful before
    trusting any verdict that depends on it.
    """

    def forward(self: SparseMoEFeedForward, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])

        logits = self.gate(flat)
        scores = self._scores(logits)
        selection_scores = scores + self.expert_bias.to(scores.dtype).unsqueeze(0)

        if mode == "identity":
            _, expert_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        elif mode == "inverted":
            # The experts the router ranked *worst*: an upper bound on how much
            # the learned assignment is worth.
            _, expert_indices = torch.topk(
                selection_scores, self.top_k, dim=-1, largest=False
            )
        elif mode == "random":
            generator = torch.Generator(device=flat.device).manual_seed(
                _ROUTING_SEED + self._causality_layer_index
            )
            noise = torch.rand(
                selection_scores.shape, generator=generator, device=flat.device
            )
            _, expert_indices = torch.topk(noise, self.top_k, dim=-1)
        else:  # pragma: no cover - guarded by _ROUTING_MODES
            raise ValueError(f"unknown routing mode {mode!r}")

        gate_weights = scores.gather(-1, expert_indices)
        if self.norm_topk_prob and self.top_k > 1:
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        output = torch.zeros_like(flat)
        one_hot = F.one_hot(expert_indices, num_classes=self.n_routed).sum(dim=1)
        for expert_id, expert in enumerate(self.experts):
            token_ids = torch.nonzero(one_hot[:, expert_id], as_tuple=False).flatten()
            if token_ids.numel() == 0:
                continue
            expert_out = expert(flat.index_select(0, token_ids))
            weight = (gate_weights * (expert_indices == expert_id)).sum(dim=-1)
            weight = weight.index_select(0, token_ids).unsqueeze(-1)
            # Mirrors mimomix_core: `index_add_` needs matching dtypes, which
            # differ under autocast. The identity self-check would fail here
            # first if this rebuild drifted from the original.
            contribution = expert_out * weight.to(expert_out.dtype)
            output.index_add_(0, token_ids, contribution.to(output.dtype))

        if self.shared_expert is not None:
            output = output + self.shared_expert(flat)

        self._aux_loss = logits.new_zeros(())
        return output.reshape(original_shape)

    return forward


_ROUTING_SEED = 59
_ROUTING_MODES = ("identity", "random", "inverted")


def _patch_routing(mode: str) -> Callable[[MiMoMixModel], Callable[[], None]]:
    def apply(model: MiMoMixModel) -> Callable[[], None]:
        layers = _moe_layers(model)
        if not layers:
            raise ValueError("model has no sparse MoE layers to intervene on")
        saved = SparseMoEFeedForward.forward
        for index, layer in enumerate(layers):
            layer._causality_layer_index = index

        def restore() -> None:
            SparseMoEFeedForward.forward = saved
            for layer in layers:
                if hasattr(layer, "_causality_layer_index"):
                    del layer._causality_layer_index

        SparseMoEFeedForward.forward = _routing_forward(mode)
        return restore

    return apply


def _shared_expert_off(model: MiMoMixModel) -> Callable[[], None]:
    """Drop the always-on expert, leaving only the routed ones."""

    layers = [layer for layer in _moe_layers(model) if layer.shared_expert is not None]
    saved = [(layer, layer.shared_expert) for layer in layers]

    def restore() -> None:
        for layer, module in saved:
            layer.shared_expert = module

    for layer in layers:
        layer.shared_expert = None
    return restore


def _mtp_side_effect(model: MiMoMixModel) -> Callable[[], None]:
    """Run the MTP chain during scoring instead of skipping it.

    This is not an ablation, it is a leak test. The audit scores with
    ``return_mtp=False``, matching the repo's own ``evaluate``; turning MTP on
    should leave the main-path logits untouched, because the speculative heads
    are supposed to read the trunk without writing to it. A non-zero delta here
    would mean MTP mutates shared state and that every published tier loss
    depends on whether the heads happened to run.

    MTP's actual value is decoding throughput, which is a different measurement
    (``benchmark_mimomix.measure_decoding``) and is deliberately out of scope --
    this audit only measures effects on held-out next-token loss.
    """

    modules = getattr(model, "mtp_modules", None)
    if modules is None or len(modules) == 0:
        return lambda: None

    saved = _SCORE_OPTIONS.get("return_mtp", False)

    def restore() -> None:
        _SCORE_OPTIONS["return_mtp"] = saved

    _SCORE_OPTIONS["return_mtp"] = True
    return restore


IDENTITY = Intervention(
    name="identity_routing_rebuild",
    description=(
        "Route through the reimplemented MoE forward without changing the "
        "expert choice. Must reproduce the baseline bit-exactly."
    ),
    apply=_patch_routing("identity"),
    expected="no change (self-check)",
    requires=lambda model: bool(_moe_layers(model)),
)


INTERVENTIONS: Tuple[Intervention, ...] = (
    Intervention(
        name="thinking_core",
        description="Force ThinkingCore.residual_scale to 0, closing the gate the whole recursive core passes through.",
        apply=_thinking_core_off,
        expected="v58 assumed this was a live mechanism",
        requires=_has_thinking_core,
    ),
    Intervention(
        name="moe_routing_random",
        description="Choose top-k experts uniformly at random, discarding the learned assignment but keeping the gate weights.",
        apply=_patch_routing("random"),
        expected="cost of discarding learned routing",
        requires=lambda model: bool(_moe_layers(model)),
    ),
    Intervention(
        name="moe_routing_inverted",
        description="Choose the experts the router ranked worst: an upper bound on the value of the learned assignment.",
        apply=_patch_routing("inverted"),
        expected="worst case; should exceed the random cost",
        requires=lambda model: bool(_moe_layers(model)),
    ),
    Intervention(
        name="moe_shared_expert",
        description="Remove the always-on shared expert, leaving only routed capacity.",
        apply=_shared_expert_off,
        expected="cost of the dense residual path",
        requires=lambda model: any(layer.shared_expert is not None for layer in _moe_layers(model)),
    ),
    Intervention(
        name="mtp_main_path_leak",
        description="Run the speculative MTP chain during scoring; the main-path logits must not move.",
        apply=_mtp_side_effect,
        expected="inert -- a leak test, not an ablation; MTP's value is decoding speed, measured elsewhere",
        requires=lambda model: bool(len(getattr(model, "mtp_modules", []) or [])),
    ),
)


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------


def self_check(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    baseline: ScoreResult,
    batch_size: int,
) -> Dict[str, Any]:
    """Prove the routing reimplementation is faithful before trusting verdicts.

    Raises if the identity rebuild moves the loss at all. Every routing verdict
    is a difference against this path, so an unfaithful rebuild would show up as
    a mechanism effect that is really a transcription bug.
    """

    if not IDENTITY.available(model):
        return {"ran": False, "reason": "model has no MoE layers"}

    with applied(model, IDENTITY):
        rebuilt = score(model, inputs, labels, batch_size=batch_size)

    delta = rebuilt.loss - baseline.loss
    agreement = rebuilt.agreement_with(baseline)
    if delta != 0.0 or agreement != 1.0:
        raise AssertionError(
            "identity routing rebuild is not faithful: "
            f"delta={delta:.3e}, argmax agreement={agreement:.6f}. "
            "Every routing verdict differences against this path, so the audit "
            "refuses to report numbers it cannot trust."
        )
    return {"ran": True, "delta_nats": delta, "argmax_agreement": agreement}


def audit(
    model: MiMoMixModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 16,
    interventions: Sequence[Intervention] = INTERVENTIONS,
) -> Dict[str, Any]:
    """Measure every mechanism's causal effect on held-out loss."""

    baseline = score(model, inputs, labels, batch_size=batch_size)
    floor = numerical_noise_floor(model, inputs, labels)
    checked = self_check(model, inputs, labels, baseline, batch_size)

    results: List[Dict[str, Any]] = []
    for intervention in interventions:
        if not intervention.available(model):
            results.append(
                {
                    "mechanism": intervention.name,
                    "available": False,
                    "reason": "mechanism absent from this checkpoint",
                }
            )
            continue

        with applied(model, intervention):
            perturbed = score(model, inputs, labels, batch_size=batch_size)

        delta = perturbed.loss - baseline.loss
        magnitude = abs(delta)
        changed = int((perturbed.argmax != baseline.argmax).sum().item())
        positions = int(baseline.argmax.numel())
        verdict = ACTIVE if (changed > 0 or magnitude >= RELEVANCE_SCALE) else INERT
        results.append(
            {
                "mechanism": intervention.name,
                "available": True,
                "description": intervention.description,
                "expectation": intervention.expected,
                "baseline_loss": baseline.loss,
                "perturbed_loss": perturbed.loss,
                "delta_nats": delta,
                "abs_delta_nats": magnitude,
                "argmax_agreement": perturbed.agreement_with(baseline),
                "decisions_changed": changed,
                "decision_positions": positions,
                "floor_multiple": (
                    magnitude / floor["floor_nats"] if floor["floor_nats"] > 0 else None
                ),
                "v58_reported_delta_multiple": magnitude / V58_SMALLEST_REPORTED_DELTA,
                "verdict": verdict,
            }
        )

    # Restoration is only credible if it is checked.
    after = score(model, inputs, labels, batch_size=batch_size)
    if after.loss != baseline.loss:
        raise AssertionError(
            f"model was not restored: baseline {baseline.loss!r} became {after.loss!r}. "
            "An intervention leaked, so these verdicts are void."
        )

    active = [r for r in results if r.get("verdict") == ACTIVE]
    inert = [r for r in results if r.get("verdict") == INERT]
    return {
        "schema": RECEIPT_SCHEMA,
        "baseline": {"loss_nats": baseline.loss, "tokens": baseline.tokens},
        "numerical_noise_floor": floor,
        "self_check": checked,
        "mechanisms": results,
        "summary": {
            "active": [r["mechanism"] for r in active],
            "inert": [r["mechanism"] for r in inert],
            "ranked_by_effect": [
                r["mechanism"]
                for r in sorted(active, key=lambda r: r["abs_delta_nats"], reverse=True)
            ],
        },
        "non_claims": [
            "This is one checkpoint, one corpus and one evaluation set. A mechanism "
            "inert in this trained model is not inert in general -- it may be inert "
            "because training never gave it a gradient path, which is a fact about "
            "this training run.",
            "The floor is numerical only. The seed-to-seed floor is larger and is "
            "not measured here, so 'active' means 'clears arithmetic noise', not "
            "'would survive a retrain'.",
            "Intervening is not the same as removing. Deleting a mechanism and "
            "retraining could give a different result, because the rest of the "
            "model would adapt.",
            "Held-out loss is the only quantity measured. Nothing here evaluates "
            "reply quality, latency, or behaviour on any downstream task.",
        ],
    }


def load_checkpoint(path: str) -> Tuple[MiMoMixModel, text_utils.WordTokenizer, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = MiMoMixConfig(**payload["config"])
    model = MiMoMixModel(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    tokenizer = text_utils.WordTokenizer.from_dict(payload["tokenizer"])
    return model, tokenizer, payload


def build_evaluation(
    tokenizer: text_utils.WordTokenizer,
    database: str,
    sequence_length: int,
    rows: int,
    blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack held-out rows for scoring.

    Dispatches on suffix so a checkpoint can be audited against the corpus it was
    actually trained on. Auditing a v60/v61 checkpoint against `llm_chat.db`
    would encode 292-word-type text with a 10,538-type tokenizer and measure
    mechanisms on a distribution the model never saw.
    """

    if str(database).lower().endswith((".jsonl", ".json")):
        corpus = text_utils.load_chat_pairs_jsonl(
            database, limit=rows, validation_fraction=0.2, seed=57
        )
    else:
        corpus = text_utils.load_chat_pairs(
            database, limit=rows, validation_fraction=0.2, seed=57
        )
    inputs, labels = text_utils.build_training_tensors(
        corpus.validation, tokenizer, sequence_length=sequence_length
    )
    return inputs[:blocks], labels[:blocks]


def print_summary(report: Dict[str, Any]) -> None:
    floor = report["numerical_noise_floor"]["floor_nats"]
    print(f"baseline loss        {report['baseline']['loss_nats']:.8f} nats "
          f"over {report['baseline']['tokens']:,} tokens")
    print(f"numerical floor      {floor:.3e} nats (batch-order spread)")
    print()
    print(f"{'mechanism':24s} {'delta (nats)':>14s} {'decisions changed':>19s}  verdict")
    print("-" * 70)
    for row in report["mechanisms"]:
        if not row.get("available"):
            print(f"{row['mechanism']:24s} {'--':>14s} {'--':>19s}  absent")
            continue
        changed = f"{row['decisions_changed']:,} / {row['decision_positions']:,}"
        print(
            f"{row['mechanism']:24s} {row['delta_nats']:>+14.6e} {changed:>19s}  {row['verdict']}"
        )
    print()
    ranked = report["summary"]["ranked_by_effect"]
    if ranked:
        print("active, strongest first: " + ", ".join(ranked))
    if report["summary"]["inert"]:
        print("inert:                   " + ", ".join(report["summary"]["inert"]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="output/v58_full/v58_full.pt")
    parser.add_argument("--database", default="databases/llm_chat.db")
    parser.add_argument("--rows", type=int, default=4000, help="corpus rows to read")
    parser.add_argument("--blocks", type=int, default=96, help="packed evaluation blocks to score")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", default=None, help="write the receipt JSON here")
    parser.add_argument("--torch_threads", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    model, tokenizer, payload = load_checkpoint(args.checkpoint)
    inputs, labels = build_evaluation(
        tokenizer,
        args.database,
        sequence_length=model.config.native_context,
        rows=args.rows,
        blocks=args.blocks,
    )
    report = audit(model, inputs, labels, batch_size=args.batch_size)
    report["checkpoint"] = {
        "path": str(args.checkpoint),
        "schema": payload.get("schema"),
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }
    report["evaluation"] = {
        "database": args.database,
        "rows_read": args.rows,
        "blocks_scored": int(inputs.shape[0]),
        "sequence_length": int(model.config.native_context),
    }
    print_summary(report)

    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
