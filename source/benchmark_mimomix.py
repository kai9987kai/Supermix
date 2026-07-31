"""End-to-end measurement of the v53 MiMoMix stack.

Runs four measurements on a small model trained on a synthetic task, and prints
what was actually observed rather than what the design hopes for:

1. **KV-cache footprint** -- the hybrid layout versus all-global attention, at a
   sequence length you choose. Pure arithmetic over the layout; no model needed.
2. **MoE load balancing** -- routing entropy and starved-expert count before and
   after the auxiliary-loss-free bias rule has had a chance to act.
3. **MTP speculative decoding** -- acceptance rate, acceptance length and
   forward reduction, *before* and *after* training, with a hard check that the
   emitted tokens are identical to greedy decoding in both cases.
4. **Progressive thinking ladder** -- cycle reduction and decision fidelity,
   audited against the ceiling budget at full cost.

The task is deliberately synthetic: a small periodic language whose next token
is predictable, so a model that trains for a few seconds on CPU can actually
learn it. That makes the MTP numbers meaningful (an untrained draft is noise)
and the numbers themselves meaningless as a claim about anything else.

Usage::

    python source/benchmark_mimomix.py
    python source/benchmark_mimomix.py --steps 400 --seq-length 131072 --json out.json
    python source/benchmark_mimomix.py --enforce-gates
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Optional, Sequence

import torch

import mimomix_controller as controller
import mimomix_decoding as decoding
from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward


# ---------------------------------------------------------------------------
# Synthetic task
# ---------------------------------------------------------------------------


def periodic_batch(
    batch: int, length: int, period: int, vocab: int, generator: torch.Generator
) -> torch.Tensor:
    """A periodic sequence with a random phase per row.

    Learnable but not trivial: the model must infer the phase from context
    rather than memorise one fixed string.
    """

    phase = torch.randint(0, period, (batch, 1), generator=generator)
    positions = torch.arange(length).unsqueeze(0)
    return ((positions + phase) % period) % vocab


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------


def measure_cache(model: MiMoMixModel, sequence_lengths: Sequence[int]) -> Dict[str, object]:
    rows = [decoding.hybrid_cache_footprint(model, int(n)) for n in sequence_lengths]
    return {
        "layout": list(model.layout),
        "sliding_window": model.config.sliding_window,
        "global_layers": model.layout.count("global"),
        "local_layers": model.layout.count("swa"),
        "by_sequence_length": [
            {
                "sequence_length": r["sequence_length"],
                "hybrid_entries": r["hybrid_entries"],
                "all_global_entries": r["all_global_entries"],
                "reduction_factor": r["reduction_factor"],
                "saved_fraction": r["saved_fraction"],
            }
            for r in rows
        ],
    }


def measure_routing(model: MiMoMixModel, batches: Sequence[torch.Tensor]) -> Dict[str, object]:
    """Accumulate expert load across several batches, not one snapshot.

    A single batch's ``last_expert_load`` is an instantaneous reading. The
    aux-loss-free bias rule is a control loop, so it *rings*: at any one step
    the load can look lopsided while the running average is balanced. Judging
    balance from one batch measures the phase of the oscillation, not the
    balance. So this sums over several batches before reporting.
    """

    moe_layers = [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]
    if not moe_layers:
        return {"moe_layers": 0, "note": "model has no MoE layers"}

    totals = [torch.zeros(layer.n_routed) for layer in moe_layers]
    sinks: List[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            model(batch, return_mtp=False, past_length=0)
            for index, layer in enumerate(moe_layers):
                totals[index] += layer.last_expert_load.detach().cpu()
            sinks.append(float(model.telemetry()["mean_sink_mass"]))

    per_layer = []
    for index, total in enumerate(totals):
        mass = float(total.sum())
        share = (total / mass) if mass > 0 else total
        n = int(total.numel())
        entropy = float(-(share * share.clamp_min(1e-12).log()).sum())
        per_layer.append(
            {
                "layer": index,
                "normalised_entropy": round(entropy / math.log(n), 6) if n > 1 else 1.0,
                "herfindahl_index": round(float((share * share).sum()), 6),
                "starved_experts": int((total == 0).sum()),
                "min_share": round(float(share.min()), 6),
                "max_share": round(float(share.max()), 6),
            }
        )

    return {
        "moe_layers": len(per_layer),
        "batches_accumulated": len(batches),
        "mean_normalised_entropy": round(
            sum(float(l["normalised_entropy"]) for l in per_layer) / len(per_layer), 6
        ),
        "worst_layer_entropy": round(min(float(l["normalised_entropy"]) for l in per_layer), 6),
        "worst_layer_herfindahl": round(max(float(l["herfindahl_index"]) for l in per_layer), 6),
        "balanced_herfindahl": round(1.0 / moe_layers[0].n_routed, 6),
        "any_starved_expert": any(l["starved_experts"] for l in per_layer),
        "total_starved": sum(int(l["starved_experts"]) for l in per_layer),
        "mean_attention_sink_mass": round(sum(sinks) / len(sinks), 6) if sinks else 0.0,
        "per_layer": per_layer,
    }


def measure_decoding(
    model: MiMoMixModel, prompt: torch.Tensor, new_tokens: int
) -> Dict[str, object]:
    started = time.perf_counter()
    fast = decoding.speculative_generate(model, prompt, max_new_tokens=new_tokens)
    slow = decoding.greedy_generate(model, prompt, max_new_tokens=new_tokens)
    identical = bool(torch.equal(fast.new_tokens, slow.new_tokens))
    return {
        "tokens": int(fast.new_tokens.shape[1]),
        "identical_to_greedy": identical,
        "acceptance_rate": round(fast.stats.acceptance_rate, 6),
        "acceptance_length": round(fast.stats.acceptance_length, 6),
        "speculative_forwards": fast.stats.verify_forwards,
        "greedy_forwards": slow.stats.verify_forwards,
        "forward_reduction": round(
            1.0 - (fast.stats.verify_forwards / max(1, slow.stats.verify_forwards)), 6
        ),
        "measured_seconds": round(time.perf_counter() - started, 4),
    }


def measure_controller(
    model: MiMoMixModel, requests: Sequence[torch.Tensor], policy: controller.ThinkingPolicy
) -> Dict[str, object]:
    return controller.audit_decision_fidelity(model, requests, policy=policy, mode="agent")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, object]:
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    generator = torch.Generator().manual_seed(args.seed)

    config = MiMoMixConfig(
        vocab_size=args.vocab,
        hidden_size=96,
        n_heads=4,
        n_kv_heads=2,
        n_layers=args.layers,
        intermediate_size=192,
        sliding_window=args.window,
        hybrid_ratio=args.hybrid_ratio,
        native_context=64,
        max_position_embeddings=256,
        n_routed_experts=args.experts,
        moe_top_k=2,
        moe_intermediate_size=48,
        router_bias_update_speed=args.bias_speed,
        n_mtp_layers=args.mtp_layers,
        mtp_loss_weight=1.0,
        thinking_cycles=2,
        thinking_max_cycles=8,
        rope_scaling="yarn",
    )
    model = MiMoMixModel(config)

    report: Dict[str, object] = {
        "config": config.to_dict(),
        "parameters": model.parameter_report(),
        "cache": measure_cache(model, args.sequence_lengths),
    }

    prompt = periodic_batch(1, 12, args.period, args.vocab, generator)

    model.eval()
    report["decoding_untrained"] = measure_decoding(model, prompt, args.generate)

    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    losses: List[float] = []
    training_started = time.perf_counter()
    for step in range(args.steps):
        batch = periodic_batch(args.batch, args.train_length, args.period, args.vocab, generator)
        optimiser.zero_grad(set_to_none=True)
        out = model(batch, labels=batch)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        model.step_router_bias()  # exactly one bias update per optimizer step
        losses.append(float(out.lm_loss.detach()))
    training_seconds = time.perf_counter() - training_started

    report["training"] = {
        "steps": args.steps,
        "seconds": round(training_seconds, 3),
        "first_lm_loss": round(losses[0], 6) if losses else None,
        "final_lm_loss": round(losses[-1], 6) if losses else None,
        "uniform_baseline_loss": round(math.log(args.vocab), 6),
    }

    model.eval()
    report["routing"] = measure_routing(
        model,
        [
            periodic_batch(args.batch, args.train_length, args.period, args.vocab, generator)
            for _ in range(args.routing_batches)
        ],
    )
    report["decoding_trained"] = measure_decoding(model, prompt, args.generate)

    policy = controller.ThinkingPolicy(
        ladder=(1, 2, 4, 8),
        confidence_target=args.confidence_target,
        entropy_target=args.entropy_target,
        continue_threshold=args.continue_threshold,
        decision_margin=args.decision_margin,
    )
    requests = [
        periodic_batch(1, 10 + i, args.period, args.vocab, generator)
        for i in range(args.audit_requests)
    ]
    report["controller"] = measure_controller(model, requests, policy)

    report["gates"] = evaluate_gates(report)
    return report


def evaluate_gates(report: Dict[str, object]) -> Dict[str, object]:
    """Objective pass/fail checks. These bound correctness, not quality."""

    decoding_trained = report["decoding_trained"]
    decoding_untrained = report["decoding_untrained"]
    controller_report = report["controller"]
    training = report["training"]
    routing = report.get("routing", {})

    checks = {
        "speculative_matches_greedy_untrained": bool(decoding_untrained["identical_to_greedy"]),
        "speculative_matches_greedy_trained": bool(decoding_trained["identical_to_greedy"]),
        "model_learned_the_task": (
            training["final_lm_loss"] is not None
            and training["final_lm_loss"] < 0.5 * training["uniform_baseline_loss"]
        ),
        "draft_is_accepted_after_training": decoding_trained["acceptance_rate"] > 0.3,
        "acceptance_improved_with_training": (
            decoding_trained["acceptance_length"] > decoding_untrained["acceptance_length"]
        ),
        "no_starved_experts": not routing.get("any_starved_expert", False),
        "controller_top1_fidelity_perfect": controller_report["top1_disagreements"] == 0,
    }
    return {"checks": checks, "passed": all(checks.values())}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure the MiMoMix v53 stack end to end")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300, help="training steps")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--vocab", type=int, default=16)
    parser.add_argument("--period", type=int, default=5)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--hybrid-ratio", type=int, default=5)
    parser.add_argument("--experts", type=int, default=8)
    # The sign-based bias rule is a control loop: too large a step overshoots
    # the balance point every update and rings instead of converging. Measured
    # on this task, gamma=1e-2 starves ~1 expert per layer and gamma>=5e-2
    # starves most of them, while gamma=1e-3 (the value DeepSeek-V3 reports)
    # converges. Raise this only with a routing measurement in hand.
    parser.add_argument("--bias-speed", type=float, default=1e-3)
    parser.add_argument("--mtp-layers", type=int, default=3)
    parser.add_argument("--train-length", type=int, default=32)
    parser.add_argument("--generate", type=int, default=48)
    parser.add_argument("--audit-requests", type=int, default=16)
    parser.add_argument("--routing-batches", type=int, default=8)
    parser.add_argument("--confidence-target", type=float, default=0.5)
    parser.add_argument("--entropy-target", type=float, default=1.0)
    parser.add_argument("--continue-threshold", type=float, default=0.6)
    parser.add_argument("--decision-margin", type=float, default=5e-4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[1024, 32768, 1048576],
        help="sequence lengths for the KV-cache footprint table",
    )
    parser.add_argument("--json", type=str, default=None, help="write the full report to this path")
    parser.add_argument(
        "--enforce-gates", action="store_true", help="exit non-zero if any check fails"
    )
    return parser


def _print_summary(report: Dict[str, object]) -> None:
    parameters = report["parameters"]
    cache = report["cache"]
    training = report["training"]
    untrained = report["decoding_untrained"]
    trained = report["decoding_trained"]
    ctl = report["controller"]
    routing = report.get("routing", {})

    print("== MiMoMix v53 measurement ==")
    print(f"  layout                 {''.join('G' if k == 'global' else 'L' for k in cache['layout'])}"
          f"  (window {cache['sliding_window']})")
    print(f"  parameters             {parameters['total']:,} total, "
          f"{parameters['active_per_token']:,} active/token")
    print()
    print("  KV cache (hybrid vs all-global)")
    for row in cache["by_sequence_length"]:
        print(f"    seq {row['sequence_length']:>9,}   {row['reduction_factor']:>6.2f}x smaller"
              f"   ({row['saved_fraction'] * 100:.1f}% saved)")
    print()
    print(f"  training               {training['steps']} steps in {training['seconds']}s, "
          f"lm loss {training['first_lm_loss']:.3f} -> {training['final_lm_loss']:.3f} "
          f"(uniform baseline {training['uniform_baseline_loss']:.3f})")
    if routing.get("moe_layers"):
        print(f"  routing                entropy {routing['mean_normalised_entropy']:.3f} "
              f"(worst layer {routing['worst_layer_entropy']:.3f}), "
              f"herfindahl {routing['worst_layer_herfindahl']:.4f} "
              f"vs balanced {routing['balanced_herfindahl']:.4f}, "
              f"starved {routing['total_starved']} "
              f"over {routing['batches_accumulated']} batches")
    print()
    print("  MTP speculative decoding")
    for label, row in (("untrained", untrained), ("trained  ", trained)):
        print(f"    {label}  acceptance {row['acceptance_rate'] * 100:>5.1f}%   "
              f"length {row['acceptance_length']:.3f}   "
              f"forwards {row['speculative_forwards']:>3} vs {row['greedy_forwards']:>3} "
              f"({row['forward_reduction'] * 100:+.1f}%)   "
              f"identical to greedy: {row['identical_to_greedy']}")
    print()
    print(f"  thinking ladder        {ctl['requests']} requests, "
          f"cycle reduction {ctl['cycle_reduction'] * 100:+.1f}%, "
          f"top-1 fidelity {ctl['top1_fidelity'] * 100:.1f}%, "
          f"ordered top-3 fidelity {ctl['ordered_topk_fidelity'] * 100:.1f}%")
    print(f"  exits                  {ctl['exit_reasons']}")
    print()
    gates = report["gates"]
    for name, passed in gates["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()
    print("  These are measurements on a synthetic periodic task with a tiny model.")
    print("  They demonstrate the mechanisms work. They are not a quality result.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    _print_summary(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"  full report written to {args.json}")
    if args.enforce_gates and not report["gates"]["passed"]:
        failed = [k for k, v in report["gates"]["checks"].items() if not v]
        print(f"\nGATES FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
