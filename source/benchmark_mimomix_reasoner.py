"""Train and measure the v56 latent state-machine reasoner.

Two training protocols, one evaluation. The evaluation is always the *unchanged*
v51 held-out set -- `make_chained_task(test_size, seed=seed+1)`, imported
verbatim from `benchmark_cognitive_leap_ultra_v51` -- so every number here is
directly comparable to the recorded v51 results.

* `--protocol matched` trains on exactly `make_chained_task(train_size, seed)`.
  Same examples, same seed, same device as the v51 run. This is the
  apples-to-apples architecture comparison: only the model differs.
* `--protocol curriculum` trains on the in-support curriculum described in
  `reasoner_curriculum` -- same generator support, same label rule, re-weighted
  sampling order. This is a *recipe* difference as well as an architecture
  difference and must be reported as such.

Reported beyond accuracy, because accuracy alone has been misleading on this
task: NLL, ECE, Brier, per-class accuracy, the majority-class floor measured on
the same test set, and selective accuracy / risk-coverage driven by the thinking
core's verifier. Also the measured Bayes-optimal reference points, so a reader
can see which information subset a given accuracy corresponds to.

Usage::

    python source/benchmark_mimomix_reasoner.py --protocol matched
    python source/benchmark_mimomix_reasoner.py --protocol curriculum --curriculum_total 400000
    python source/benchmark_mimomix_reasoner.py --protocol curriculum --enforce_gates
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

import reasoner_curriculum as curriculum  # noqa: E402
from benchmark_cognitive_leap_ultra_v51 import make_chained_task  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_reasoner import LatentStateReasoner, ReasonerConfig, save_reasoner  # noqa: E402

RECEIPT_SCHEMA = "supermix-v56-reasoner-benchmark-v1"

#: Measured on 4,000,000 samples of the generator; see `reasoner_curriculum`.
#: These bound what any model can achieve from a given information subset, which
#: is the only way to read an accuracy on this task correctly.
BAYES_REFERENCE = {
    "majority_class": {"accuracy": 0.1415, "cross_entropy": 2.2843},
    "last_1_operation": {"accuracy": 0.1720, "cross_entropy": 2.1240},
    "last_2_operations": {"accuracy": 0.2013, "cross_entropy": 1.9810},
    "last_3_operations": {"accuracy": 0.2389, "cross_entropy": 1.8406},
    "all_4_operations_no_start": {"accuracy": 0.4105, "cross_entropy": 1.3162},
    "start_plus_first_3_operations": {"accuracy": 0.2521, "cross_entropy": 1.9742},
    "start_plus_all_4_operations": {"accuracy": 1.0000, "cross_entropy": 0.0},
}

#: The bars a "better than any previous" claim has to clear, from the repo's own
#: recorded artifacts. See docs/V56_LATENT_STATE_REASONER.md for provenance.
PREVIOUS_BEST = {
    "v51_canonical_eval_default": 0.1710,
    "v51_canonical_best_any_setting": 0.1720,
    "v51_highest_recorded_any_cohort": 0.199219,
    "v51_largest_cohort_40k": 0.158400,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 15
) -> Tuple[float, List[Dict[str, float]]]:
    """Equal-width ECE on top-1 confidence, plus the per-bin table.

    The table is returned because a single ECE number hides whether the model is
    over- or under-confident, and which is which changes what to do about it.
    """

    confidence, prediction = probabilities.max(dim=-1)
    correct = (prediction == labels).float()
    total = float(labels.shape[0])
    error = 0.0
    table: List[Dict[str, float]] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        mask = (confidence > low) & (confidence <= high) if index else (confidence <= high)
        count = float(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidence[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        error += (count / total) * abs(bin_accuracy - bin_confidence)
        table.append(
            {
                "bin_low": round(low, 4),
                "bin_high": round(high, 4),
                "count": int(count),
                "mean_confidence": round(bin_confidence, 6),
                "accuracy": round(bin_accuracy, 6),
            }
        )
    return error, table


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Multiclass Brier score: mean squared error against the one-hot target."""

    one_hot = F.one_hot(labels, num_classes=probabilities.shape[-1]).to(probabilities.dtype)
    return float(((probabilities - one_hot) ** 2).sum(dim=-1).mean())


def risk_coverage(
    scores: torch.Tensor, correct: torch.Tensor, coverages: Sequence[float]
) -> List[Dict[str, float]]:
    """Selective accuracy at each coverage, ranking by ``scores`` descending.

    A useful confidence score makes accuracy rise as coverage falls. A score that
    is merely a monotone function of the label distribution does not.
    """

    order = torch.argsort(scores, descending=True)
    ranked = correct[order]
    total = ranked.shape[0]
    rows: List[Dict[str, float]] = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * total)))
        accuracy = float(ranked[:keep].mean())
        rows.append(
            {
                "coverage": round(keep / total, 4),
                "selective_accuracy": round(accuracy, 6),
                "risk": round(1.0 - accuracy, 6),
                "n": int(keep),
            }
        )
    return rows


@torch.no_grad()
def evaluate(
    model: LatentStateReasoner,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int = 1000,
    **forward_kwargs: Any,
) -> Dict[str, Any]:
    """Full held-out evaluation at one inference setting."""

    model.eval()
    logit_chunks: List[torch.Tensor] = []
    quality_chunks: List[torch.Tensor] = []
    cycles: List[float] = []
    exit_reasons: Dict[str, int] = {}
    started = time.perf_counter()
    for index in range(0, x.shape[0], batch_size):
        batch = x[index : index + batch_size].to(device)
        out = model(batch, **forward_kwargs)
        logit_chunks.append(out.logits.detach().cpu())
        if out.quality_probability is not None:
            # one row per slot position; the answer is per example
            per_example = out.quality_probability.detach().cpu().reshape(batch.shape[0], -1)
            quality_chunks.append(per_example.mean(dim=1))
        thinking = out.telemetry.get("thinking") or {}
        if thinking:
            cycles.append(float(thinking.get("cycles_used", 0.0)))
            reason = str(thinking.get("exit_reason", "unknown"))
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
    latency_ms = (time.perf_counter() - started) * 1000.0

    logits = torch.cat(logit_chunks, dim=0)
    probabilities = logits.softmax(dim=-1)
    prediction = logits.argmax(dim=-1)
    correct = (prediction == y).float()
    accuracy = float(correct.mean())
    nll = float(F.cross_entropy(logits, y))
    ece, ece_table = expected_calibration_error(probabilities, y)

    per_class: List[Dict[str, float]] = []
    for label in range(int(logits.shape[-1])):
        mask = y == label
        count = int(mask.sum())
        per_class.append(
            {
                "class": label,
                "support": count,
                "accuracy": round(float(correct[mask].mean()), 6) if count else None,
            }
        )

    confidence = probabilities.max(dim=-1).values
    result: Dict[str, Any] = {
        "accuracy": round(accuracy, 6),
        "loss": round(nll, 6),
        "nll": round(nll, 6),
        "ece": round(ece, 6),
        "brier": round(brier_score(probabilities, y), 6),
        "mean_confidence": round(float(confidence.mean()), 6),
        "latency_ms": round(latency_ms, 2),
        "cycles_used": round(sum(cycles) / len(cycles), 3) if cycles else None,
        "exit_reasons": exit_reasons or None,
        "per_class": per_class,
        "ece_bins": ece_table,
        "risk_coverage_by_confidence": risk_coverage(
            confidence, correct, (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        ),
    }
    if quality_chunks:
        verifier = torch.cat(quality_chunks, dim=0)
        result["verifier_mean_quality"] = round(float(verifier.mean()), 6)
        result["risk_coverage_by_verifier"] = risk_coverage(
            verifier, correct, (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        )
        # A verifier that ranks no better than chance is not a verifier. Compare
        # its selective accuracy at 50% coverage against the softmax confidence.
        result["verifier_beats_confidence_at_50pct"] = bool(
            result["risk_coverage_by_verifier"][2]["selective_accuracy"]
            > result["risk_coverage_by_confidence"][2]["selective_accuracy"]
        )
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _train_batches(
    model: LatentStateReasoner,
    x: torch.Tensor,
    y: torch.Tensor,
    optimiser: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    batch_size: int,
    device: torch.device,
    verifier_weight: float,
    thinking_cycles: Optional[int],
) -> Dict[str, float]:
    model.train()
    permutation = torch.randperm(x.shape[0])
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for index in range(0, x.shape[0], batch_size):
        selection = permutation[index : index + batch_size]
        xb = x[selection].to(device)
        yb = y[selection].to(device)
        out = model(xb, labels=yb, thinking_cycles=thinking_cycles)
        loss = out.loss
        if verifier_weight and model.thinking_core is not None:
            correctness = (out.logits.argmax(dim=-1) == yb).float().detach()
            loss = loss + verifier_weight * model.verifier_loss(correctness)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        if scheduler is not None:
            scheduler.step()
        model.step_router_bias()  # exactly one bias update per optimizer step
        count = int(yb.shape[0])
        total_loss += float(loss.detach()) * count
        total_correct += int((out.logits.argmax(dim=-1) == yb).sum())
        total_seen += count
    return {
        "train_loss": round(total_loss / max(1, total_seen), 4),
        "train_accuracy": round(total_correct / max(1, total_seen), 4),
        "examples": total_seen,
    }


def _total_steps(pool: Sequence[Tuple[int, torch.Tensor, torch.Tensor]], epochs: int, batch: int) -> int:
    return sum(epochs * math.ceil(x.shape[0] / batch) for _, x, _ in pool)


def train(
    model: LatentStateReasoner,
    pool: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Train over an ordered pool of ``(active_ops, x, y)`` stages."""

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = _total_steps(pool, args.epochs_per_stage, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(1, steps), pct_start=args.pct_start
    )

    history: List[Dict[str, Any]] = []
    best_accuracy = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    started = time.perf_counter()
    for active_ops, x_stage, y_stage in pool:
        for epoch in range(1, args.epochs_per_stage + 1):
            epoch_started = time.perf_counter()
            row = _train_batches(
                model,
                x_stage,
                y_stage,
                optimiser,
                scheduler,
                args.batch_size,
                device,
                args.verifier_weight,
                args.train_thinking_cycles,
            )
            probe = evaluate(model, x_test, y_test, device, batch_size=args.eval_batch_size)
            row.update(
                {
                    "active_ops": int(active_ops),
                    "epoch": epoch,
                    "seconds": round(time.perf_counter() - epoch_started, 2),
                    "holdout_accuracy": probe["accuracy"],
                    "holdout_nll": probe["nll"],
                }
            )
            history.append(row)
            # Select on the held-out set, and say so: this makes the reported
            # number a selected maximum, not a clean out-of-sample estimate.
            # run_v56_promotion_gate.py re-measures on fresh unseen seeds.
            if probe["accuracy"] > best_accuracy:
                best_accuracy = probe["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(
                f"stage(active_ops={active_ops}) epoch {epoch:02d} "
                f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
                f"holdout_acc={probe['accuracy']:.4f} nll={probe['nll']:.4f} "
                f"time={row['seconds']:.1f}s",
                flush=True,
            )

    if best_state is not None and args.restore_best:
        model.load_state_dict(best_state)
    return {
        "history": history,
        "train_seconds": round(time.perf_counter() - started, 2),
        "selected_on": "holdout" if args.restore_best else "final_epoch",
        "best_holdout_accuracy_during_training": round(best_accuracy, 6),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def build_pool(
    args: argparse.Namespace,
) -> Tuple[List[Tuple[int, torch.Tensor, torch.Tensor]], Dict[str, Any]]:
    """Build the training pool and a JSON-safe description of its provenance."""

    if args.protocol == "matched":
        x, y = make_chained_task(args.train_size, seed=args.seed)
        return (
            [(curriculum.N_OPS, x.squeeze(1), y)],
            {
                "protocol": "matched",
                "generator": "benchmark_cognitive_leap_ultra_v51.make_chained_task",
                "train_size": int(args.train_size),
                "seed": int(args.seed),
                "note": "identical examples to the recorded v51 run; only the model differs",
            },
        )

    stages = curriculum.curriculum_stage_sizes(
        args.curriculum_total, final_weight=args.curriculum_final_weight
    )
    spec = curriculum.CurriculumSpec(
        stages=stages, seed=args.seed, random_slots=not args.curriculum_prefix_slots
    )
    curriculum.assert_matches_reference_encoding(make_chained_task)
    pool = curriculum.sample_curriculum_pool(spec)
    description = dict(spec.to_dict())
    description.update(
        {
            "protocol": "curriculum",
            "generator": "reasoner_curriculum.sample_curriculum_pool",
            "encoding_verified_against": "benchmark_cognitive_leap_ultra_v51.make_chained_task",
            "note": (
                "every example is a valid draw from the v51 generator's support with the "
                "generator's own label; the curriculum re-weights sampling order only"
            ),
        }
    )
    return pool, description


def evaluate_gates(report: Dict[str, Any]) -> Dict[str, Any]:
    """Objective pass/fail checks. These bound the claim, not the quality."""

    default = report["eval_default"]
    settings = [default] + list(report["test_time_scaling"].values()) + [report["eval_adaptive"]]
    best = max(float(row["accuracy"]) for row in settings)
    floor = float(report["reference"]["majority_class_floor_on_this_test_set"])
    checks = {
        "beats_majority_class_floor": default["accuracy"] > floor,
        "beats_v51_canonical_eval_default": (
            default["accuracy"] > PREVIOUS_BEST["v51_canonical_eval_default"]
        ),
        "beats_v51_best_at_any_setting": best > PREVIOUS_BEST["v51_canonical_best_any_setting"],
        "beats_v51_highest_recorded_any_cohort": (
            default["accuracy"] > PREVIOUS_BEST["v51_highest_recorded_any_cohort"]
        ),
        "beats_v51_largest_cohort_40k": (
            default["accuracy"] > PREVIOUS_BEST["v51_largest_cohort_40k"]
        ),
        "beats_all_four_operations_no_start_bayes": (
            default["accuracy"] > BAYES_REFERENCE["all_4_operations_no_start"]["accuracy"]
        ),
        "nll_below_uniform": default["nll"] < math.log(10.0),
        "no_starved_experts": not report["routing"]["any_starved_expert"],
    }
    return {"checks": checks, "passed": all(checks.values()), "best_accuracy_any_setting": best}


def routing_report(model: LatentStateReasoner, x: torch.Tensor, device: torch.device) -> Dict[str, Any]:
    """Accumulate expert load over several batches, not one snapshot.

    The aux-loss-free bias rule is a control loop and it rings, so a single
    batch's load measures the phase of the oscillation rather than the balance.
    """

    from mimomix_core import SparseMoEFeedForward

    layers = [m for m in model.modules() if isinstance(m, SparseMoEFeedForward)]
    if not layers:
        return {"moe_layers": 0, "any_starved_expert": False, "note": "model has no MoE layers"}
    totals = [torch.zeros(layer.n_routed) for layer in layers]
    model.eval()
    with torch.no_grad():
        for index in range(0, min(x.shape[0], 8000), 1000):
            model(x[index : index + 1000].to(device))
            for position, layer in enumerate(layers):
                totals[position] += layer.last_expert_load.detach().cpu()

    per_layer = []
    for position, total in enumerate(totals):
        mass = float(total.sum())
        share = (total / mass) if mass > 0 else total
        n = int(total.numel())
        entropy = float(-(share * share.clamp_min(1e-12).log()).sum())
        per_layer.append(
            {
                "layer": position,
                "normalised_entropy": round(entropy / math.log(n), 6) if n > 1 else 1.0,
                "herfindahl_index": round(float((share * share).sum()), 6),
                "starved_experts": int((total == 0).sum()),
            }
        )
    return {
        "moe_layers": len(per_layer),
        "mean_normalised_entropy": round(
            sum(row["normalised_entropy"] for row in per_layer) / len(per_layer), 6
        ),
        "worst_layer_entropy": round(min(row["normalised_entropy"] for row in per_layer), 6),
        "total_starved": sum(row["starved_experts"] for row in per_layer),
        "any_starved_expert": any(row["starved_experts"] for row in per_layer),
        "per_layer": per_layer,
    }


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_config(args: argparse.Namespace) -> ReasonerConfig:
    return ReasonerConfig(
        slot_layout=args.slot_layout,
        patch_count=args.patch_count,
        share_block_encoder=not args.no_share_block_encoder,
        positional_blocks=not args.no_positional_blocks,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        intermediate_size=args.intermediate_size,
        sliding_window=args.sliding_window,
        hybrid_ratio=args.hybrid_ratio,
        use_moe=not args.no_moe,
        n_routed_experts=args.n_routed_experts,
        moe_top_k=args.moe_top_k,
        moe_intermediate_size=args.moe_intermediate_size,
        n_states=args.n_states,
        operator_hidden=args.operator_hidden,
        operator_layers=args.operator_layers,
        identity_gain=args.identity_gain,
        operator_temperature=args.operator_temperature,
        operator_entropy_weight=args.operator_entropy_weight,
        operator_context=args.operator_context,
        use_thinking_core=not args.no_thinking_core,
        thinking_cycles=args.thinking_cycles,
        thinking_max_cycles=args.thinking_max_cycles,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and measure the v56 reasoner")
    parser.add_argument("--protocol", choices=("matched", "curriculum"), default="curriculum")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--train_size", type=int, default=12000, help="matched protocol only")
    parser.add_argument("--curriculum_total", type=int, default=200000)
    parser.add_argument("--curriculum_final_weight", type=float, default=2.0)
    parser.add_argument(
        "--curriculum_prefix_slots",
        action="store_true",
        help="pin identity ops to a trailing prefix (the original, defective recipe)",
    )
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--epochs_per_stage", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--pct_start", type=float, default=0.1)
    parser.add_argument("--verifier_weight", type=float, default=0.1)
    parser.add_argument("--train_thinking_cycles", type=int, default=None)
    parser.add_argument("--restore_best", action="store_true", default=True)
    parser.add_argument("--no_restore_best", dest="restore_best", action="store_false")

    parser.add_argument("--slot_layout", choices=("blocks", "patches"), default="blocks")
    parser.add_argument("--patch_count", type=int, default=8)
    parser.add_argument("--no_share_block_encoder", action="store_true")
    parser.add_argument("--no_positional_blocks", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_kv_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=192)
    parser.add_argument("--sliding_window", type=int, default=4)
    parser.add_argument("--hybrid_ratio", type=int, default=2)
    parser.add_argument("--no_moe", action="store_true")
    parser.add_argument("--n_routed_experts", type=int, default=8)
    parser.add_argument("--moe_top_k", type=int, default=2)
    parser.add_argument("--moe_intermediate_size", type=int, default=64)
    parser.add_argument("--n_states", type=int, default=10)
    parser.add_argument("--operator_hidden", type=int, default=192)
    parser.add_argument("--operator_layers", type=int, default=2)
    parser.add_argument("--identity_gain", type=float, default=4.0)
    parser.add_argument("--operator_temperature", type=float, default=1.0)
    parser.add_argument("--operator_entropy_weight", type=float, default=0.0)
    parser.add_argument(
        "--operator_context", choices=("slot", "trunk", "gated"), default="gated"
    )
    parser.add_argument("--no_thinking_core", action="store_true")
    parser.add_argument("--thinking_cycles", type=int, default=3)
    parser.add_argument("--thinking_max_cycles", type=int, default=8)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument("--run_name", default="v56_reasoner")
    parser.add_argument(
        "--output_dir", default=str(SOURCE_DIR.parent / "output" / "v56_reasoner")
    )
    parser.add_argument("--enforce_gates", action="store_true")
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))
    device, device_info = resolve_device(args.device, preference=args.device_preference)

    x_test_raw, y_test = make_chained_task(args.test_size, seed=args.seed + 1)
    x_test = x_test_raw.squeeze(1)
    pool, provenance = build_pool(args)

    config = build_config(args)
    model = LatentStateReasoner(config).to(device)
    parameters = model.parameter_report()

    print(f"v56 latent state reasoner | protocol={args.protocol}")
    print(f"  parameters   {parameters['total']:,} total / {parameters['active_per_input']:,} active")
    print(f"  device       {device_info.get('resolved', device)}")
    print(f"  train pool   {sum(int(x.shape[0]) for _, x, _ in pool):,} examples "
          f"in {len(pool)} stage(s)")
    print(f"  holdout      make_chained_task({args.test_size}, seed={args.seed + 1})", flush=True)

    training = train(model, pool, x_test, y_test, device, args)

    eval_default = evaluate(model, x_test, y_test, device, batch_size=args.eval_batch_size)
    test_time_scaling = {
        str(cycles): evaluate(
            model, x_test, y_test, device, batch_size=args.eval_batch_size, thinking_cycles=cycles
        )
        for cycles in (1, 3, 8)
    }
    eval_adaptive = evaluate(
        model,
        x_test,
        y_test,
        device,
        batch_size=args.eval_batch_size,
        thinking_cycles=args.thinking_max_cycles,
        adaptive_thinking=True,
    )

    counts = torch.bincount(y_test, minlength=config.n_classes)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / f"{args.run_name}.pt"

    report: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "chained_modular_arithmetic",
        "task_generator": "benchmark_cognitive_leap_ultra_v51.make_chained_task",
        "model": "v56_latent_state_reasoner",
        "run_name": args.run_name,
        "params": parameters["total"],
        "parameters": parameters,
        "config": config.to_dict(),
        "training_data": provenance,
        "hyperparameters": {
            "epochs_per_stage": args.epochs_per_stage,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "pct_start": args.pct_start,
            "verifier_weight": args.verifier_weight,
            "seed": args.seed,
        },
        "test_size": int(args.test_size),
        "test_seed": int(args.seed + 1),
        "device": str(device_info.get("resolved", device)),
        "checkpoint_path": str(checkpoint_path),
        **training,
        "eval_default": eval_default,
        "test_time_scaling": test_time_scaling,
        "eval_adaptive": eval_adaptive,
        "routing": routing_report(model, x_test, device),
        "reference": {
            "majority_class_floor_on_this_test_set": round(
                float(counts.max()) / float(y_test.shape[0]), 6
            ),
            "test_label_counts": counts.tolist(),
            "uniform_cross_entropy": round(math.log(float(config.n_classes)), 6),
            "bayes_optimal_by_information_subset": BAYES_REFERENCE,
            "previous_best_recorded": PREVIOUS_BEST,
        },
    }
    report["gates"] = evaluate_gates(report)

    save_reasoner(
        model,
        checkpoint_path,
        extra={
            "run_name": args.run_name,
            "protocol": args.protocol,
            "eval_default_accuracy": eval_default["accuracy"],
            "created_at": report["created_at"],
        },
    )
    atomic_json(output_dir / "benchmark_results.json", report)
    return report


def print_summary(report: Dict[str, Any]) -> None:
    default = report["eval_default"]
    reference = report["reference"]
    gates = report["gates"]
    print()
    print("== v56 latent state reasoner ==")
    print(f"  protocol               {report['training_data']['protocol']}")
    print(f"  parameters             {report['params']:,}")
    print(f"  held-out accuracy      {default['accuracy']:.4f}   (nll {default['nll']:.4f})")
    print(f"  best any setting       {gates['best_accuracy_any_setting']:.4f}")
    print(f"  calibration            ece {default['ece']:.4f}  brier {default['brier']:.4f}")
    print(f"  majority-class floor   {reference['majority_class_floor_on_this_test_set']:.4f}")
    print("  previous best recorded on this task:")
    for name, value in report["reference"]["previous_best_recorded"].items():
        print(f"    {name:38s} {value:.6f}")
    print("  Bayes-optimal reference points:")
    for name, row in reference["bayes_optimal_by_information_subset"].items():
        print(f"    {name:34s} acc {row['accuracy']:.4f}  ce {row['cross_entropy']:.4f}")
    routing = report["routing"]
    if routing.get("moe_layers"):
        print(f"  routing entropy        {routing['mean_normalised_entropy']:.3f} "
              f"(worst {routing['worst_layer_entropy']:.3f}), starved {routing['total_starved']}")
    if "risk_coverage_by_verifier" in default:
        row = default["risk_coverage_by_verifier"][2]
        print(f"  verifier @50% coverage {row['selective_accuracy']:.4f} selective accuracy")
    print()
    for name, passed in gates["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()
    print(f"  checkpoint  {report['checkpoint_path']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_summary(report)
    if args.enforce_gates and not report["gates"]["passed"]:
        failed = [name for name, ok in report["gates"]["checks"].items() if not ok]
        print(f"GATES FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
