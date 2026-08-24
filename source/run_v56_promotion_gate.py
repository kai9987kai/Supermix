"""Paired multi-seed promotion gate for a v56 reasoner against the v51 baseline.

A single 1000-example held-out draw has a standard error of about 1.2 points, so
"beat 0.1710 once" is not evidence. This gate does what the repo's own bounded
promotion receipts did: score both models on **the same** fresh cohorts, then
test the difference with a paired test rather than comparing two independent
point estimates.

What it enforces:

* **Fresh cohorts only.** Seeds 51 and 52 (train and canonical test) and every
  seed already used by an existing v51 gate are refused, so a pass cannot come
  from a cohort either model has been tuned against.
* **Identical inputs.** Both models see the same tensor from the same
  `make_chained_task` call. The v51 head takes `(B, 1, 128)` and the v56 reasoner
  accepts exactly that shape, so no re-encoding stands between them.
* **A paired test.** McNemar's test on the discordant pairs -- exact binomial
  where it is tractable, normal approximation with continuity correction beyond
  that, and the choice is reported.
* **Per-seed non-regression.** A pooled win driven by two lucky cohorts is not a
  win, so the gate also requires v56 to lead on a declared majority of seeds and
  reports a sign test over them.
* **A floor check.** Both arms are compared against each cohort's own
  majority-class accuracy, because on this task that floor is only ~1.3 points
  below the recorded baseline.

Usage::

    python source/run_v56_promotion_gate.py \
        --candidate output/v56b_randslots_entropy/v56b_randslots_entropy.pt \
        --baseline output/v56_baseline_replication_v51_12k_seed51/cognitive_leap_ultra_v51_trained.pth
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

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from benchmark_cognitive_leap_ultra_v51 import make_chained_task  # noqa: E402
from device_utils import resolve_device  # noqa: E402
from mimomix_reasoner import load_reasoner  # noqa: E402
from model_variants import ChampionNetCognitiveLeapUltraExpert  # noqa: E402

RECEIPT_SCHEMA = "supermix-v56-bounded-promotion-v1"

#: Seeds that are already spoken for. 51/52 are the v51 train/test pair;
#: the rest are cohorts existing v51 gates and receipts have already scored, so
#: reusing them would let a result be selected rather than measured.
RESERVED_SEEDS = frozenset(
    {51, 52}
    | {641, 643, 647, 653, 659, 661, 673, 677}
    | {719, 727, 733, 739}
    | set(range(21052, 40053))
    | set(range(41052, 60053))
)


def default_seeds(count: int, start: int = 900_001, stride: int = 7) -> Tuple[int, ...]:
    """Deterministic fresh seeds, skipping anything reserved."""

    seeds: List[int] = []
    candidate = start
    while len(seeds) < count:
        if candidate not in RESERVED_SEEDS:
            seeds.append(candidate)
        candidate += stride
    return tuple(seeds)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def mcnemar(baseline_only: int, candidate_only: int) -> Dict[str, Any]:
    """McNemar's test on discordant pairs.

    ``candidate_only`` is the count of items the candidate got right and the
    baseline got wrong; ``baseline_only`` the reverse. Concordant pairs carry no
    information about the difference and are excluded by construction.
    """

    discordant = baseline_only + candidate_only
    if discordant == 0:
        return {
            "discordant_pairs": 0,
            "candidate_only_wins": 0,
            "baseline_only_wins": 0,
            "method": "none",
            "statistic": None,
            "p_value": 1.0,
            "note": "the two models agreed on every item",
        }

    if discordant <= 1000:
        smaller = min(baseline_only, candidate_only)
        tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
        method = "exact_binomial"
        statistic = float(smaller)
    else:
        numerator = (abs(candidate_only - baseline_only) - 1.0) ** 2
        statistic = numerator / discordant
        # chi-square with 1 df: p = erfc(sqrt(chi2 / 2))
        p_value = math.erfc(math.sqrt(statistic / 2.0))
        method = "chi_square_continuity_corrected"

    return {
        "discordant_pairs": int(discordant),
        "candidate_only_wins": int(candidate_only),
        "baseline_only_wins": int(baseline_only),
        "method": method,
        "statistic": round(float(statistic), 6),
        "p_value": float(f"{p_value:.6g}"),
    }


def sign_test(wins: int, losses: int) -> Dict[str, Any]:
    """Two-sided exact sign test over per-seed outcomes (ties dropped)."""

    trials = wins + losses
    if trials == 0:
        return {"trials": 0, "wins": 0, "p_value": 1.0}
    smaller = min(wins, losses)
    tail = sum(math.comb(trials, k) for k in range(smaller + 1))
    return {
        "trials": int(trials),
        "wins": int(wins),
        "losses": int(losses),
        "p_value": float(f"{min(1.0, 2.0 * tail / (2.0 ** trials)):.6g}"),
    }


def wilson_interval(successes: int, total: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval -- honest near the ends, unlike the normal one."""

    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (z / denominator) * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Model arms
# ---------------------------------------------------------------------------


class CandidateArm:
    """The v56 reasoner."""

    name = "v56_latent_state_reasoner"

    def __init__(self, path: Path, device: torch.device):
        self.model, payload = load_reasoner(path, map_location=str(device))
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.path = path
        self.extra: Dict[str, Any] = dict(payload.get("extra") or {})
        self.parameters = int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for index in range(0, x.shape[0], batch_size):
            out = self.model(x[index : index + batch_size].to(self.device))
            chunks.append(out.logits.argmax(dim=-1).cpu())
        return torch.cat(chunks, dim=0)


class BaselineArm:
    """The recorded v51 `ChampionNetCognitiveLeapUltraExpert`."""

    name = "v51_cognitive_leap_ultra_expert"

    def __init__(self, path: Path, device: torch.device, reasoning_cycles: Optional[int] = None):
        self.model = ChampionNetCognitiveLeapUltraExpert()
        state = torch.load(path, map_location=str(device), weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.path = path
        self.reasoning_cycles = reasoning_cycles
        self.parameters = int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def predict(self, x: torch.Tensor, batch_size: int = 1000) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        kwargs: Dict[str, Any] = {}
        if self.reasoning_cycles is not None:
            kwargs["reasoning_cycles"] = int(self.reasoning_cycles)
        for index in range(0, x.shape[0], batch_size):
            logits = self.model(x[index : index + batch_size].to(self.device), **kwargs)
            chunks.append(logits.squeeze(1).argmax(dim=-1).cpu())
        return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> Dict[str, Any]:
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    if args.torch_threads:
        torch.set_num_threads(max(1, args.torch_threads))

    seeds = tuple(args.seeds) if args.seeds else default_seeds(args.n_seeds)
    reserved = sorted(set(seeds) & RESERVED_SEEDS)
    if reserved:
        raise SystemExit(
            f"refusing to score reserved seeds {reserved}: these cohorts are already "
            "used by the v51 train/test split or an existing gate"
        )

    candidate = CandidateArm(Path(args.candidate), device)
    baseline = BaselineArm(Path(args.baseline), device, args.baseline_reasoning_cycles)

    print(f"v56 bounded promotion gate on {len(seeds)} fresh cohorts x {args.samples} samples")
    print(f"  candidate  {candidate.path}  ({candidate.parameters:,} params)")
    print(f"  baseline   {baseline.path}  ({baseline.parameters:,} params)")
    print(f"  device     {device_info.get('resolved', device)}", flush=True)

    per_seed: List[Dict[str, Any]] = []
    candidate_correct_total = 0
    baseline_correct_total = 0
    floor_total = 0
    total = 0
    both_wrong = candidate_only = baseline_only = both_right = 0
    started = time.perf_counter()

    for seed in seeds:
        x, y = make_chained_task(args.samples, seed=seed)
        candidate_prediction = candidate.predict(x, args.batch_size)
        baseline_prediction = baseline.predict(x, args.batch_size)

        candidate_hit = candidate_prediction == y
        baseline_hit = baseline_prediction == y
        counts = torch.bincount(y, minlength=10)
        floor = int(counts.max())

        both_right += int((candidate_hit & baseline_hit).sum())
        candidate_only += int((candidate_hit & ~baseline_hit).sum())
        baseline_only += int((~candidate_hit & baseline_hit).sum())
        both_wrong += int((~candidate_hit & ~baseline_hit).sum())

        candidate_correct = int(candidate_hit.sum())
        baseline_correct = int(baseline_hit.sum())
        candidate_correct_total += candidate_correct
        baseline_correct_total += baseline_correct
        floor_total += floor
        total += args.samples

        row = {
            "seed": int(seed),
            "samples": int(args.samples),
            "candidate_accuracy": round(candidate_correct / args.samples, 6),
            "baseline_accuracy": round(baseline_correct / args.samples, 6),
            "majority_class_floor": round(floor / args.samples, 6),
            "delta": round((candidate_correct - baseline_correct) / args.samples, 6),
            "candidate_wins": candidate_correct > baseline_correct,
        }
        per_seed.append(row)
        print(
            f"  seed {seed:>8}  candidate {row['candidate_accuracy']:.4f}  "
            f"baseline {row['baseline_accuracy']:.4f}  floor {row['majority_class_floor']:.4f}  "
            f"delta {row['delta']:+.4f}",
            flush=True,
        )

    candidate_accuracy = candidate_correct_total / total
    baseline_accuracy = baseline_correct_total / total
    floor_accuracy = floor_total / total
    candidate_interval = wilson_interval(candidate_correct_total, total)
    baseline_interval = wilson_interval(baseline_correct_total, total)

    wins = sum(1 for row in per_seed if row["candidate_accuracy"] > row["baseline_accuracy"])
    losses = sum(1 for row in per_seed if row["candidate_accuracy"] < row["baseline_accuracy"])
    beats_floor = sum(
        1 for row in per_seed if row["candidate_accuracy"] > row["majority_class_floor"]
    )

    paired = mcnemar(baseline_only, candidate_only)
    signs = sign_test(wins, losses)
    required_wins = max(1, math.ceil(args.min_win_fraction * len(seeds)))

    checks = {
        "candidate_beats_baseline_pooled": candidate_accuracy > baseline_accuracy,
        "paired_test_significant": paired["p_value"] < args.alpha,
        "per_seed_majority_win": wins >= required_wins,
        "candidate_beats_floor_on_every_seed": beats_floor == len(seeds),
        "confidence_intervals_disjoint": candidate_interval[0] > baseline_interval[1],
        "no_reserved_seed_used": True,
    }

    report: Dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "chained_modular_arithmetic",
        "task_generator": "benchmark_cognitive_leap_ultra_v51.make_chained_task",
        "device": str(device_info.get("resolved", device)),
        "protocol": {
            "seeds": [int(s) for s in seeds],
            "samples_per_seed": int(args.samples),
            "total_samples": int(total),
            "alpha": float(args.alpha),
            "min_win_fraction": float(args.min_win_fraction),
            "required_seed_wins": int(required_wins),
            "reserved_seeds_excluded": True,
            "note": (
                "both arms score the identical tensor from the same make_chained_task call; "
                "(seed, size) jointly determine a cohort, so size is held fixed across arms"
            ),
        },
        "candidate": {
            "name": candidate.name,
            "checkpoint": str(candidate.path),
            "parameters": candidate.parameters,
            "extra": candidate.extra,
            "accuracy": round(candidate_accuracy, 6),
            "correct": candidate_correct_total,
            "wilson_95": [round(candidate_interval[0], 6), round(candidate_interval[1], 6)],
        },
        "baseline": {
            "name": baseline.name,
            "checkpoint": str(baseline.path),
            "parameters": baseline.parameters,
            "reasoning_cycles": args.baseline_reasoning_cycles,
            "accuracy": round(baseline_accuracy, 6),
            "correct": baseline_correct_total,
            "wilson_95": [round(baseline_interval[0], 6), round(baseline_interval[1], 6)],
        },
        "summary": {
            "delta": round(candidate_accuracy - baseline_accuracy, 6),
            "relative_improvement": (
                round((candidate_accuracy - baseline_accuracy) / baseline_accuracy, 6)
                if baseline_accuracy
                else None
            ),
            "majority_class_floor": round(floor_accuracy, 6),
            "candidate_margin_over_floor": round(candidate_accuracy - floor_accuracy, 6),
            "baseline_margin_over_floor": round(baseline_accuracy - floor_accuracy, 6),
            "seed_wins": wins,
            "seed_losses": losses,
            "seed_ties": len(seeds) - wins - losses,
            "seeds_beating_floor": beats_floor,
        },
        "contingency": {
            "both_right": both_right,
            "candidate_only": candidate_only,
            "baseline_only": baseline_only,
            "both_wrong": both_wrong,
        },
        "mcnemar": paired,
        "sign_test": signs,
        "per_seed": per_seed,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "checks": checks,
        "passed": all(checks.values()),
        "claim_scope": {
            "task": "one synthetic 4-step chained modular-arithmetic task",
            "establishes": (
                "that this checkpoint predicts this task's held-out labels more accurately "
                "than the recorded v51 checkpoint, on fresh paired cohorts"
            ),
            "does_not_establish": [
                "general reasoning ability",
                "language ability of any kind",
                "transfer to any other task or dataset",
                "that the architecture is better outside this task family",
                "production readiness or a default-route change",
            ],
            "production_default_allowed": False,
        },
    }
    report["decision"] = "promote_for_this_task" if report["passed"] else "reject"
    return report


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


def print_summary(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print()
    print("== v56 bounded promotion gate ==")
    print(
        f"  candidate  {report['candidate']['accuracy']:.6f}  "
        f"95% CI [{report['candidate']['wilson_95'][0]:.4f}, {report['candidate']['wilson_95'][1]:.4f}]"
    )
    print(
        f"  baseline   {report['baseline']['accuracy']:.6f}  "
        f"95% CI [{report['baseline']['wilson_95'][0]:.4f}, {report['baseline']['wilson_95'][1]:.4f}]"
    )
    print(f"  floor      {summary['majority_class_floor']:.6f} (majority class)")
    print(
        f"  delta      {summary['delta']:+.6f}  "
        f"({summary['relative_improvement'] * 100:+.1f}% relative)"
        if summary["relative_improvement"] is not None
        else f"  delta      {summary['delta']:+.6f}"
    )
    print(
        f"  per-seed   {summary['seed_wins']} wins / {summary['seed_losses']} losses / "
        f"{summary['seed_ties']} ties  (sign test p={report['sign_test']['p_value']})"
    )
    print(
        f"  McNemar    {report['mcnemar']['method']}  "
        f"candidate-only {report['mcnemar']['candidate_only_wins']} vs "
        f"baseline-only {report['mcnemar']['baseline_only_wins']}  "
        f"p={report['mcnemar']['p_value']}"
    )
    print(f"  total      {report['protocol']['total_samples']:,} paired samples")
    print()
    for name, passed in report["checks"].items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print()
    print(f"  decision: {report['decision'].upper()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paired promotion gate for a v56 checkpoint")
    parser.add_argument("--candidate", required=True, help="v56 checkpoint (.pt)")
    parser.add_argument("--baseline", required=True, help="v51 checkpoint (.pth)")
    parser.add_argument("--baseline_reasoning_cycles", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--min_win_fraction", type=float, default=0.8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--device_preference", default="cuda,npu,xpu,dml,mps,cpu")
    parser.add_argument("--torch_threads", type=int, default=0)
    parser.add_argument(
        "--output", default=str(SOURCE_DIR.parent / "output" / "v56_promotion_gate.json")
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run(args)
    print_summary(report)
    atomic_json(Path(args.output), report)
    print(f"  receipt written to {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
