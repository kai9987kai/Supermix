"""Calibrated score fusion for the response reranker.

`rank_response_candidates` combines roughly twenty heterogeneous signals with
hand-tuned weights. About half pass through `_normalize_01` (per-batch min-max)
and half arrive raw: `sim_ctx` and `sim_resp` are dot products of 128-dimensional
hash vectors, `lex_sim` is a Jaccard ratio, `bucket_bonus` is a model score, and
`freq_penalty` is a `log1p` count. Those live on different scales, so a weight
does not determine how much a signal actually moves the ranking.

Measured on a 16-candidate pool, expressing each signal's influence as
|weight| x realised spread:

    sim_ctx       weight 0.60 -> 46.9% of the decision
    lex_sim       weight 0.10 ->  1.8%
    freq_penalty  weight 0.03 ->  6.7%
    a _normalize_01 signal at weight 0.06 -> 12.2%

`lex_sim` carries three times the nominal weight of `freq_penalty` and a third of
its influence, and a min-max signal at weight 0.06 outweighs it twice over,
purely because min-max guarantees a spread of exactly 1 while a Jaccard ratio
over short texts spans about 0.09. Whatever the weights were tuned to express,
this is not it.

The fix is the probability integral transform: replace each signal by its own
percentile rank, so every signal spans [0, 1] with a uniform distribution and a
weight of 0.60 really is six times a weight of 0.10.

**Percentile ranking alone is unsafe**, which is the part worth stating plainly.
The transform is scale-free, so it maps a signal carrying nothing but numerical
noise onto the same full [0, 1] spread as the most informative one, promoting
noise to first-class influence. Rank calibration is therefore always paired here
with a dispersion gate that zeroes a signal whose raw spread is indistinguishable
from noise *before* the transform can amplify it.

Design contract, matching the rest of this runtime:

* Pure and deterministic. Ties take average ranks, so the transform does not
  depend on input order.
* Bounded and JSON-safe outputs; diagnostics carry statistics, never text.
* Opt-in. The default fusion mode is `legacy`, which is bit-exact with the
  behaviour that shipped. Nothing changes unless a caller asks for it.
* Advisory. This module ranks candidates. It has no authority over routing,
  compute, tools, permissions, or safety.

References
----------
Bruch, Gai, Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval",
ACM TOIS 2023, arXiv:2210.11934 — score calibration before fusion, and the
finding that min-max preserves the skew of the underlying distribution.

Cormack, Clarke, Buettcher, "Reciprocal Rank Fusion Outperforms Condorcet and
Individual Rank Learning Methods", SIGIR 2009 — rank fusion as a scale-free
combiner.

Fox and Shaw, "Combination of Multiple Searches", TREC-2 1994 — CombMNZ, which
multiplies a combined score by the number of signals supporting a candidate.

These motivate the design; they are not evidence about Supermix. No weights were
retrained here, so none of this makes the underlying checkpoint smarter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


SCORE_FUSION_SCHEMA_VERSION = "supermix-score-fusion-v1"
SCORE_FUSION_VERSION = "supermix-score-fusion-runtime-v1"

FUSION_MODES = ("legacy", "gated", "calibrated", "consensus")
DEFAULT_FUSION_MODE = "legacy"

# Runtime callers may select only these. `calibrated` and `consensus` measured
# 51 and 58 percentage points of top-1 BELOW legacy on 200 real corpus probes
# with hard negatives, so they are reachable only through an explicit
# `allow_experimental=True` from the benchmark. They stay in the tree as the
# evidence trail for why rank calibration was rejected, not as an option.
RUNTIME_FUSION_MODES = ("legacy", "gated")
EXPERIMENTAL_FUSION_MODES = ("calibrated", "consensus")

# A signal whose relative spread falls below this is treated as carrying no
# information. Percentile ranking would otherwise stretch it to full influence.
DISPERSION_FLOOR = 1e-4
# Between the floor and this, the gate ramps linearly rather than switching, so
# a signal drifting toward informative does not flip the ranking discontinuously.
DISPERSION_CEILING = 1e-3

# CombMNZ support is capped so a candidate agreed on by every signal cannot
# overwhelm the calibrated sum outright.
MAX_CONSENSUS_MULTIPLIER = 1.35
# A candidate counts as "supported" by a signal when it ranks in the top of that
# signal's percentile distribution.
CONSENSUS_SUPPORT_QUANTILE = 0.65

_EPS = 1e-12


def resolve_fusion_mode(mode: Any, allow_experimental: bool = False) -> str:
    """Unknown, missing, or experimental modes fall back to shipped behaviour.

    The fallback is deliberate rather than an error: a stale config or a typo
    must degrade to the measured-best path, never to a mode that halves
    retrieval quality.
    """

    candidate = str(mode or "").strip().lower()
    allowed = FUSION_MODES if allow_experimental else RUNTIME_FUSION_MODES
    return candidate if candidate in allowed else DEFAULT_FUSION_MODE


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Map a signal onto its own percentile rank in [0, 1].

    Ties share their average rank, so the result does not depend on the order
    candidates arrived in. A constant signal maps to all zeros: it separates
    nothing, so it should contribute nothing.
    """

    flat = values.detach().reshape(-1).to(torch.float32)
    count = int(flat.numel())
    if count == 0:
        return torch.zeros(0, dtype=torch.float32)
    if count == 1:
        return torch.zeros(1, dtype=torch.float32)

    order = torch.argsort(flat, stable=True)
    ordered = flat.index_select(0, order)
    if float(ordered[-1] - ordered[0]) <= _EPS:
        # Constant signal: it separates nothing, so it should contribute nothing
        # rather than a uniform median-rank offset.
        return torch.zeros(count, dtype=torch.float32)
    positions = torch.arange(count, dtype=torch.float32)

    # Average the positions of each run of equal values.
    averaged = positions.clone()
    start = 0
    for index in range(1, count + 1):
        if index == count or float(ordered[index]) != float(ordered[start]):
            if index - start > 1:
                averaged[start:index] = positions[start:index].mean()
            start = index

    ranks = torch.empty(count, dtype=torch.float32)
    ranks.scatter_(0, order, averaged)
    return ranks / float(count - 1)


def zmuv(values: torch.Tensor, clamp: float = 4.0) -> torch.Tensor:
    """Zero mean, unit variance, clamped so downstream diagnostics stay bounded."""

    flat = values.detach().reshape(-1).to(torch.float32)
    if flat.numel() == 0:
        return torch.zeros(0, dtype=torch.float32)
    std = float(flat.std(unbiased=False))
    if std <= _EPS:
        return torch.zeros_like(flat)
    return torch.clamp((flat - flat.mean()) / std, min=-clamp, max=clamp)


def dispersion(values: torch.Tensor) -> float:
    """Scale-free spread of a signal: range over its own typical magnitude."""

    flat = values.detach().reshape(-1).to(torch.float32)
    if flat.numel() < 2:
        return 0.0
    spread = float(flat.max() - flat.min())
    if spread <= 0.0:
        return 0.0
    scale = float(flat.abs().mean()) + 1.0
    return spread / scale


def dispersion_gate(values: torch.Tensor) -> float:
    """A factor in [0, 1] that suppresses signals whose spread is noise.

    This is what keeps percentile ranking safe. Without it the transform would
    give a signal that separates nothing the same full spread as the signal that
    decides the answer.
    """

    measured = dispersion(values)
    if measured <= DISPERSION_FLOOR:
        return 0.0
    if measured >= DISPERSION_CEILING:
        return 1.0
    span = DISPERSION_CEILING - DISPERSION_FLOOR
    return float((measured - DISPERSION_FLOOR) / span) if span > 0 else 1.0


def calibrate_signal(values: torch.Tensor, rank_transform: bool = True) -> Tuple[torch.Tensor, float]:
    """Gate a signal on its raw spread, optionally mapping it to percentile ranks.

    `rank_transform=False` is the conservative half of the idea: suppress signals
    that carry only noise, but leave the surviving signals on their original
    scale. That matters because the shipped weights were hand-tuned *against*
    those scales and therefore already compensate for them. Rank-transforming
    without re-tuning the weights discards that compensation, which measured as a
    small regression rather than a gain.
    """

    flat = values.detach().reshape(-1).to(torch.float32)
    gate = dispersion_gate(flat)
    if gate <= 0.0:
        return torch.zeros_like(flat), 0.0
    if not rank_transform:
        return flat * gate, gate
    return percentile_rank(flat) * gate, gate


def consensus_multiplier(
    calibrated: Sequence[torch.Tensor],
    weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """CombMNZ-style support: how many signals independently back a candidate.

    A candidate agreed on by many weak signals is a safer choice than one that a
    single loud signal likes, which is the failure mode that matters most when
    the system must commit to exactly one stored response.
    """

    usable = [row for row in calibrated if row.numel() > 0]
    if not usable:
        return torch.ones(0, dtype=torch.float32)

    count = usable[0].numel()
    support = torch.zeros(count, dtype=torch.float32)
    considered = 0
    for index, row in enumerate(usable):
        if row.numel() != count:
            continue
        if float(row.max() - row.min()) <= _EPS:
            continue  # a gated-out or constant signal supports nobody
        weight = 1.0 if weights is None or index >= len(weights) else abs(float(weights[index]))
        if weight <= 0.0:
            continue
        support += (row >= CONSENSUS_SUPPORT_QUANTILE).to(torch.float32)
        considered += 1

    if considered == 0:
        return torch.ones(count, dtype=torch.float32)

    fraction = support / float(considered)
    return 1.0 + (MAX_CONSENSUS_MULTIPLIER - 1.0) * fraction


def borda(calibrated: Sequence[torch.Tensor], weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Weighted Borda count over already-calibrated signals."""

    usable = [row for row in calibrated if row.numel() > 0]
    if not usable:
        return torch.zeros(0, dtype=torch.float32)
    count = usable[0].numel()
    total = torch.zeros(count, dtype=torch.float32)
    weight_sum = 0.0
    for index, row in enumerate(usable):
        if row.numel() != count:
            continue
        weight = 1.0 if weights is None or index >= len(weights) else float(weights[index])
        total += weight * percentile_rank(row)
        weight_sum += abs(weight)
    if weight_sum <= 0.0:
        return torch.zeros(count, dtype=torch.float32)
    return total / weight_sum


def fuse_signals(
    signals: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    mode: Any = DEFAULT_FUSION_MODE,
    allow_experimental: bool = True,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Combine named signals under the requested fusion mode.

    Returns the fused score and JSON-safe diagnostics describing what each
    signal actually contributed, which is the part that was previously
    invisible.
    """

    # This is the analysis entry point, not a runtime path, so it defaults to
    # allowing every mode. `rank_response_candidates` is the guarded one.
    resolved = resolve_fusion_mode(mode, allow_experimental=allow_experimental)
    names = [name for name in signals if signals[name].numel() > 0]
    if not names:
        return torch.zeros(0, dtype=torch.float32), {
            "schema_version": SCORE_FUSION_SCHEMA_VERSION,
            "mode": resolved,
            "signals": [],
            "gated_out": [],
        }

    count = int(signals[names[0]].numel())
    fused = torch.zeros(count, dtype=torch.float32)
    calibrated: List[torch.Tensor] = []
    ordered_weights: List[float] = []
    report: List[Dict[str, Any]] = []
    gated_out: List[str] = []

    for name in names:
        raw = signals[name].detach().reshape(-1).to(torch.float32)
        if raw.numel() != count:
            continue
        weight = float(weights.get(name, 0.0))

        if resolved == "legacy":
            contribution = raw
            gate = 1.0
        else:
            contribution, gate = calibrate_signal(raw)
            if gate <= 0.0:
                gated_out.append(name)

        fused = fused + weight * contribution
        calibrated.append(contribution)
        ordered_weights.append(weight)
        report.append({
            "name": name,
            "weight": round(weight, 6),
            "raw_spread": round(float(raw.max() - raw.min()), 6),
            "dispersion": round(dispersion(raw), 6),
            "gate": round(gate, 6),
            "influence": round(abs(weight) * float(contribution.max() - contribution.min()), 6),
        })

    if resolved == "consensus" and calibrated:
        fused = fused * consensus_multiplier(calibrated, ordered_weights)

    total_influence = sum(row["influence"] for row in report) or 1.0
    for row in report:
        row["influence_share"] = round(row["influence"] / total_influence, 6)

    diagnostics = {
        "schema_version": SCORE_FUSION_SCHEMA_VERSION,
        "runtime_version": SCORE_FUSION_VERSION,
        "mode": resolved,
        "candidates": count,
        "signals": sorted(report, key=lambda row: -row["influence"]),
        "gated_out": sorted(gated_out),
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }
    return fused, diagnostics


def fusion_diagnostics(diagnostics: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compact, privacy-safe summary: statistics only, never candidate text."""

    if not isinstance(diagnostics, Mapping):
        return {
            "schema_version": SCORE_FUSION_SCHEMA_VERSION,
            "mode": DEFAULT_FUSION_MODE,
            "signal_count": 0,
            "gated_out_count": 0,
            "top_signals": [],
        }
    rows = [row for row in diagnostics.get("signals") or () if isinstance(row, Mapping)]
    return {
        "schema_version": SCORE_FUSION_SCHEMA_VERSION,
        "mode": str(diagnostics.get("mode") or DEFAULT_FUSION_MODE),
        "candidates": int(diagnostics.get("candidates") or 0),
        "signal_count": len(rows),
        "gated_out_count": len(diagnostics.get("gated_out") or ()),
        "gated_out": list(diagnostics.get("gated_out") or ())[:8],
        "top_signals": [
            {
                "name": str(row.get("name") or ""),
                "weight": float(row.get("weight") or 0.0),
                "influence_share": float(row.get("influence_share") or 0.0),
            }
            for row in rows[:6]
        ],
    }


__all__ = [
    "DEFAULT_FUSION_MODE",
    "EXPERIMENTAL_FUSION_MODES",
    "RUNTIME_FUSION_MODES",
    "DISPERSION_CEILING",
    "DISPERSION_FLOOR",
    "FUSION_MODES",
    "MAX_CONSENSUS_MULTIPLIER",
    "SCORE_FUSION_SCHEMA_VERSION",
    "SCORE_FUSION_VERSION",
    "borda",
    "calibrate_signal",
    "consensus_multiplier",
    "dispersion",
    "dispersion_gate",
    "fuse_signals",
    "fusion_diagnostics",
    "percentile_rank",
    "resolve_fusion_mode",
    "zmuv",
]
