"""The Dem-Lab observatory: rigorous instrumentation for MiMoMix.

``AI-Dem-Lab`` is a single-file browser sandbox of speculative panels -- an
entropy/randomness bench, a quantum-vs-LLM randomness comparison, a Bell
locality sandbox, a PEAR-style evidence critique, a Wolfram-ish computational
universe, a multi-agent ecosystem, semantic resonance mapping, Q-learning
feedback, and RSI novelty/stability meters.

This module keeps the *questions* those panels ask and replaces the sandbox
heuristics with defensible statistics:

======================  =====================================================
Dem-Lab panel           What lives here instead
======================  =====================================================
entropy / randomness    Shannon, min-entropy, perplexity, and a real
                        uniformity battery (chi-square with an exact
                        regularised-incomplete-gamma p-value, monobit, runs)
quantum comparison      the same battery applied to any two streams, reported
                        as a comparison rather than a verdict
Bell locality sandbox   :func:`chsh_value` -- used as a **self-test of this
                        harness**: a classical strategy must not exceed 2, so
                        a run that reports 2.4 means the statistics code is
                        wrong, not that physics happened
PEAR evidence critique  :func:`sequential_evidence` -- a log-likelihood ratio
                        that carries an explicit optional-stopping penalty,
                        which is the actual methodological lesson
mechanistic explorer    :func:`routing_attribution` -- which experts fire, how
                        concentrated the routing is, per-layer sink mass
semantic resonance      :func:`semantic_resonance` -- cosine geometry of hidden
                        states with a deterministic clustering threshold
RSI meters              :func:`novelty_score`, :func:`stability_score` and a
                        bounded composite, all deterministic
multi-agent ecosystem   :func:`replicator_step` -- discrete replicator dynamics
                        over controller policies
Q-learning feedback     :class:`BudgetPolicyLearner` -- tabular Q-learning that
                        proposes a *starting budget* for the thinking
                        controller from observed fidelity/cost outcomes
======================  =====================================================

Every function here is **deterministic**: no unseeded randomness, no wall-clock
input, no network. Two calls with the same inputs return the same floats. That
is a hard requirement -- the controller consumes some of these numbers, and a
non-reproducible control signal is not a control signal.

Nothing in this module is evidence about cognition. These are measurements of a
running system's telemetry, and they are only as meaningful as the model
producing that telemetry.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


__all__ = [
    "shannon_entropy",
    "min_entropy",
    "perplexity",
    "jensen_shannon_divergence",
    "chi_square_uniformity",
    "monobit_test",
    "runs_test",
    "randomness_report",
    "chsh_value",
    "sequential_evidence",
    "novelty_score",
    "stability_score",
    "recursive_improvement_index",
    "semantic_resonance",
    "routing_attribution",
    "robust_anomalies",
    "replicator_step",
    "run_ecosystem",
    "BudgetPolicyLearner",
    "Observatory",
]


# ---------------------------------------------------------------------------
# Special functions (kept dependency-free on purpose)
# ---------------------------------------------------------------------------


def _gamma_series(a: float, x: float, iterations: int = 500, eps: float = 3e-12) -> float:
    """Regularised lower incomplete gamma ``P(a, x)`` by series expansion."""

    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(iterations):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * eps:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a: float, x: float, iterations: int = 500, eps: float = 3e-12) -> float:
    """Regularised upper incomplete gamma ``Q(a, x)`` by continued fraction."""

    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _regularised_upper_gamma(a: float, x: float) -> float:
    """``Q(a, x) = 1 - P(a, x)``, the chi-square survival function's core."""

    if x < 0.0 or a <= 0.0:
        raise ValueError("regularised gamma requires a > 0 and x >= 0")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_series(a, x)))
    return max(0.0, min(1.0, _gamma_continued_fraction(a, x)))


def chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Upper-tail p-value of a chi-square statistic. Exact, not tabulated."""

    if degrees_of_freedom <= 0:
        return 1.0
    return _regularised_upper_gamma(degrees_of_freedom / 2.0, max(0.0, statistic) / 2.0)


# ---------------------------------------------------------------------------
# Entropy / randomness lab
# ---------------------------------------------------------------------------


def _as_probabilities(values: Sequence[float]) -> List[float]:
    total = float(sum(values))
    if total <= 0.0:
        raise ValueError("probability vector must have positive mass")
    return [float(v) / total for v in values]


def shannon_entropy(distribution: Sequence[float], base: Optional[float] = None) -> float:
    """Shannon entropy in nats (or ``base``-logarithm units)."""

    probs = _as_probabilities(distribution)
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    if base is not None:
        entropy /= math.log(base)
    return entropy


def min_entropy(distribution: Sequence[float]) -> float:
    """``-log(max p)`` -- the entropy a guessing adversary actually faces.

    Shannon entropy overstates unpredictability when one outcome dominates;
    min-entropy is the conservative number and is what randomness extraction
    is budgeted against.
    """

    probs = _as_probabilities(distribution)
    return -math.log(max(probs))


def perplexity(distribution: Sequence[float]) -> float:
    return math.exp(shannon_entropy(distribution))


def jensen_shannon_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Symmetric, bounded divergence in nats (``0 <= JSD <= ln 2``)."""

    if len(p) != len(q):
        raise ValueError("distributions must have equal length")
    pp = _as_probabilities(p)
    qq = _as_probabilities(q)
    mm = [(a + b) / 2.0 for a, b in zip(pp, qq)]

    def _kl(a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * math.log(x / y) for x, y in zip(a, b) if x > 0.0 and y > 0.0)

    return 0.5 * _kl(pp, mm) + 0.5 * _kl(qq, mm)


def chi_square_uniformity(samples: Sequence[int], n_categories: int) -> Dict[str, float]:
    """Test whether integer samples look uniform over ``n_categories``."""

    if n_categories < 2:
        raise ValueError("n_categories must be >= 2")
    counts = [0] * n_categories
    for value in samples:
        index = int(value)
        if not 0 <= index < n_categories:
            raise ValueError(f"sample {value!r} outside [0, {n_categories})")
        counts[index] += 1
    n = len(samples)
    expected = n / n_categories
    if expected <= 0.0:
        return {"statistic": 0.0, "degrees_of_freedom": 0, "p_value": 1.0, "samples": 0}
    statistic = sum((c - expected) ** 2 / expected for c in counts)
    dof = n_categories - 1
    return {
        "statistic": statistic,
        "degrees_of_freedom": dof,
        "p_value": chi_square_survival(statistic, dof),
        "samples": n,
        # A rule-of-thumb validity flag: chi-square needs expected >= 5/cell.
        "expected_per_cell": expected,
        "approximation_valid": expected >= 5.0,
    }


def monobit_test(bits: Sequence[int]) -> Dict[str, float]:
    """NIST monobit frequency test: are 0s and 1s balanced?"""

    n = len(bits)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "bits": 0}
    total = 0
    for bit in bits:
        if bit not in (0, 1):
            raise ValueError("monobit_test expects a sequence of 0/1")
        total += 1 if bit else -1
    statistic = abs(total) / math.sqrt(n)
    return {"statistic": statistic, "p_value": math.erfc(statistic / math.sqrt(2.0)), "bits": n}


def runs_test(bits: Sequence[int]) -> Dict[str, float]:
    """NIST runs test: are alternations as frequent as chance predicts?

    Only meaningful once the monobit proportion is close to 1/2; the NIST
    prerequisite is reported rather than silently assumed.
    """

    n = len(bits)
    if n < 2:
        return {"runs": 0, "p_value": 1.0, "bits": n, "prerequisite_met": False}
    ones = sum(1 for b in bits if b)
    pi = ones / n
    prerequisite = abs(pi - 0.5) <= (2.0 / math.sqrt(n))
    runs = 1 + sum(1 for i in range(1, n) if bits[i] != bits[i - 1])
    if not prerequisite or pi in (0.0, 1.0):
        return {"runs": runs, "p_value": 0.0, "bits": n, "prerequisite_met": False, "proportion": pi}
    numerator = abs(runs - 2.0 * n * pi * (1.0 - pi))
    denominator = 2.0 * math.sqrt(2.0 * n) * pi * (1.0 - pi)
    return {
        "runs": runs,
        "p_value": math.erfc(numerator / denominator),
        "bits": n,
        "prerequisite_met": True,
        "proportion": pi,
    }


def randomness_report(samples: Sequence[int], n_categories: int) -> Dict[str, object]:
    """Full battery over an integer stream, plus its parity bit stream.

    This is the honest form of Dem-Lab's "is the LLM stream as random as a
    quantum stream?" panel: it reports test statistics and p-values for each
    stream and lets the reader compare them. It does not declare a winner, and
    a p-value here is evidence about *this* battery, not about the source's
    physics.
    """

    counts = [0] * n_categories
    for value in samples:
        counts[int(value)] += 1
    distribution = [c for c in counts]
    bits = [int(v) & 1 for v in samples]
    return {
        "samples": len(samples),
        "categories": int(n_categories),
        "shannon_entropy_nats": shannon_entropy(distribution) if any(distribution) else 0.0,
        "shannon_entropy_bits": shannon_entropy(distribution, base=2.0) if any(distribution) else 0.0,
        "max_entropy_bits": math.log2(n_categories),
        "min_entropy_nats": min_entropy(distribution) if any(distribution) else 0.0,
        "uniformity": chi_square_uniformity(samples, n_categories),
        "monobit": monobit_test(bits),
        "runs": runs_test(bits),
    }


# ---------------------------------------------------------------------------
# Harness self-tests: CHSH, and evidence under optional stopping
# ---------------------------------------------------------------------------


def chsh_value(correlations: Mapping[Tuple[int, int], float]) -> Dict[str, float]:
    """CHSH combination ``S = E(0,0) + E(0,1) + E(1,0) - E(1,1)``.

    Used here as a **self-test of this module's statistics**, not as a physics
    claim. Any local strategy satisfies ``|S| <= 2``; a correlation table built
    from classical data that scores above 2 means the estimator or the data
    plumbing is broken. The quantum bound ``2 sqrt(2)`` is reported for scale
    only -- nothing in this repository can produce it.
    """

    required = [(0, 0), (0, 1), (1, 0), (1, 1)]
    missing = [k for k in required if k not in correlations]
    if missing:
        raise ValueError(f"missing correlation settings: {missing}")
    for key, value in correlations.items():
        if not -1.0 <= float(value) <= 1.0:
            raise ValueError(f"correlation {key} must lie in [-1, 1], got {value}")
    s = (
        float(correlations[(0, 0)])
        + float(correlations[(0, 1)])
        + float(correlations[(1, 0)])
        - float(correlations[(1, 1)])
    )
    return {
        "s_value": s,
        "classical_bound": 2.0,
        "tsirelson_bound": 2.0 * math.sqrt(2.0),
        "within_classical_bound": abs(s) <= 2.0 + 1e-9,
    }


def sequential_evidence(
    successes: int,
    trials: int,
    null_rate: float = 0.5,
    alternative_rate: Optional[float] = None,
    looks: int = 1,
) -> Dict[str, float]:
    """Log-likelihood ratio with an explicit optional-stopping penalty.

    The methodological lesson from the PEAR-era literature is not "the effect
    was small" -- it is that an unbounded number of peeks at an accumulating
    dataset inflates apparent evidence. So this reports:

    * the raw log-LR of ``alternative`` versus ``null``;
    * a **penalised** log-LR that subtracts ``ln(looks)``, the standard
      order-of-magnitude correction for ``looks`` independent opportunities to
      stop at a favourable moment;
    * the observed effect size, because a decisive LR on a trivial effect is
      still a trivial effect.

    ``alternative_rate`` defaults to the maximum-likelihood estimate, which is
    the *most generous possible* alternative and therefore an upper bound on
    the evidence any specific alternative could earn.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 < null_rate < 1.0:
        raise ValueError("null_rate must lie in (0, 1)")
    if looks < 1:
        raise ValueError("looks must be >= 1")

    observed = successes / trials
    alt = observed if alternative_rate is None else float(alternative_rate)
    alt = min(1.0 - 1e-12, max(1e-12, alt))

    def _log_likelihood(rate: float) -> float:
        return successes * math.log(rate) + (trials - successes) * math.log(1.0 - rate)

    log_lr = _log_likelihood(alt) - _log_likelihood(null_rate)
    penalty = math.log(looks)
    return {
        "successes": successes,
        "trials": trials,
        "observed_rate": observed,
        "null_rate": null_rate,
        "alternative_rate": alt,
        "log_likelihood_ratio": log_lr,
        "optional_stopping_penalty": penalty,
        "penalised_log_likelihood_ratio": log_lr - penalty,
        "effect_size": observed - null_rate,
        "alternative_was_fitted": alternative_rate is None,
    }


# ---------------------------------------------------------------------------
# RSI meters: novelty and stability
# ---------------------------------------------------------------------------


def _ngrams(sequence: Sequence[int], n: int) -> List[Tuple[int, ...]]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return [tuple(sequence[i : i + n]) for i in range(max(0, len(sequence) - n + 1))]


def novelty_score(sequence: Sequence[int], history: Sequence[Sequence[int]], n: int = 3) -> Dict[str, float]:
    """Fraction of ``n``-grams in ``sequence`` never seen in ``history``.

    Bounded in ``[0, 1]``. High novelty is not automatically good: a broken
    model scores 1.0. Pair it with :func:`stability_score`.
    """

    seen = set()
    for past in history:
        seen.update(_ngrams(past, n))
    grams = _ngrams(sequence, n)
    if not grams:
        return {"novelty": 0.0, "ngrams": 0, "unseen": 0, "n": n}
    unseen = sum(1 for g in grams if g not in seen)
    distinct = len(set(grams))
    return {
        "novelty": unseen / len(grams),
        "ngrams": len(grams),
        "unseen": unseen,
        "distinct_ratio": distinct / len(grams),
        "n": n,
    }


def stability_score(series: Sequence[float]) -> Dict[str, float]:
    """How steady a telemetry series is, as ``1 / (1 + coefficient of variation)``.

    Bounded in ``(0, 1]``; 1.0 means perfectly constant. Uses the mean's
    magnitude so a series centred near zero does not report false instability.
    """

    values = [float(v) for v in series]
    if len(values) < 2:
        return {"stability": 1.0, "mean": values[0] if values else 0.0, "std": 0.0, "samples": len(values)}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    std = math.sqrt(variance)
    scale = max(abs(mean), 1e-9)
    coefficient = std / scale
    return {
        "stability": 1.0 / (1.0 + coefficient),
        "mean": mean,
        "std": std,
        "coefficient_of_variation": coefficient,
        "samples": len(values),
    }


def recursive_improvement_index(
    novelty: float, stability: float, quality_delta: float, cost_delta: float
) -> Dict[str, float]:
    """A bounded composite in ``[0, 1]``: is the system getting better *cheaply*?

    ``quality_delta`` is the change in whatever quality metric the caller
    trusts; ``cost_delta`` is the change in compute spent, on the same scale.
    The index rewards novelty only when stability holds and quality rose per
    unit of extra cost.

    This is a **dashboard aggregate**, not a measurement of self-improvement.
    It has no units and no ground truth; it exists so a long run can be
    eyeballed, and it should never gate a promotion decision on its own.
    """

    novelty = min(1.0, max(0.0, float(novelty)))
    stability = min(1.0, max(0.0, float(stability)))
    efficiency = float(quality_delta) / (1.0 + max(0.0, float(cost_delta)))
    squashed = 1.0 / (1.0 + math.exp(-4.0 * efficiency))
    index = novelty * stability * squashed
    return {
        "index": min(1.0, max(0.0, index)),
        "novelty": novelty,
        "stability": stability,
        "efficiency": efficiency,
        "efficiency_squashed": squashed,
    }


# ---------------------------------------------------------------------------
# Semantic resonance and mechanistic attribution
# ---------------------------------------------------------------------------


def semantic_resonance(
    hidden_states: torch.Tensor, threshold: float = 0.5
) -> Dict[str, object]:
    """Cosine geometry of a set of representations.

    ``hidden_states`` is ``(N, H)``. Returns mean/max off-diagonal cosine
    similarity, the fraction of pairs above ``threshold`` (the "resonance
    density"), and a deterministic connected-component clustering at that
    threshold. Deterministic: no random projections, no seeded initialisation.
    """

    if hidden_states.dim() != 2:
        raise ValueError(f"expected (N, H), got {tuple(hidden_states.shape)}")
    n = int(hidden_states.shape[0])
    if n < 2:
        return {"pairs": 0, "mean_similarity": 0.0, "max_similarity": 0.0,
                "resonance_density": 0.0, "clusters": [[0]] if n else [], "n_clusters": n}

    normalised = torch.nn.functional.normalize(hidden_states.float(), dim=-1)
    similarity = normalised @ normalised.T
    mask = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
    off_diagonal = similarity[mask]

    # Deterministic single-linkage clustering by union-find at the threshold.
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    linked = (similarity >= threshold).nonzero(as_tuple=False).tolist()
    for i, j in linked:
        if i != j:
            union(i, j)
    groups: Dict[int, List[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    clusters = [sorted(v) for _, v in sorted(groups.items())]

    return {
        "pairs": int(off_diagonal.numel()),
        "mean_similarity": float(off_diagonal.mean()),
        "max_similarity": float(off_diagonal.max()),
        "min_similarity": float(off_diagonal.min()),
        "resonance_density": float((off_diagonal >= threshold).float().mean()),
        "threshold": float(threshold),
        "clusters": clusters,
        "n_clusters": len(clusters),
    }


def routing_attribution(telemetry: Mapping[str, object]) -> Dict[str, object]:
    """Mechanistic read of a MiMoMix telemetry snapshot.

    Reports, per MoE layer: which experts carried load, how concentrated the
    routing was (normalised entropy, and the Herfindahl index used for market
    concentration), and whether any expert was starved. Plus the per-layer
    attention-sink mass, which says how often heads chose to attend to nothing.
    """

    loads = telemetry.get("expert_load") or []
    layers: List[Dict[str, object]] = []
    for index, layer_load in enumerate(loads):
        values = [float(v) for v in layer_load]
        total = sum(values)
        if total <= 0.0:
            layers.append({"layer": index, "degenerate": True})
            continue
        share = [v / total for v in values]
        n = len(share)
        entropy = -sum(p * math.log(p) for p in share if p > 0.0)
        herfindahl = sum(p * p for p in share)
        starved = [i for i, v in enumerate(values) if v == 0.0]
        layers.append(
            {
                "layer": index,
                "load": values,
                "share": share,
                "normalised_entropy": entropy / math.log(n) if n > 1 else 1.0,
                "herfindahl_index": herfindahl,
                "balanced_herfindahl": 1.0 / n,
                "starved_experts": starved,
                "busiest_expert": max(range(n), key=lambda i: values[i]),
                "degenerate": False,
            }
        )

    sinks = [float(v) for v in (telemetry.get("per_layer_sink_mass") or [])]
    return {
        "moe_layers": layers,
        "mean_normalised_entropy": (
            sum(float(l["normalised_entropy"]) for l in layers if not l["degenerate"])
            / max(1, sum(1 for l in layers if not l["degenerate"]))
        )
        if layers
        else 0.0,
        "any_starved_expert": any(l.get("starved_experts") for l in layers),
        "attention_sink_mass": sinks,
        "mean_attention_sink_mass": sum(sinks) / len(sinks) if sinks else 0.0,
        "attention_layout": list(telemetry.get("attention_layout") or []),
    }


def robust_anomalies(series: Sequence[float], z_threshold: float = 3.5) -> Dict[str, object]:
    """Median/MAD outlier detection -- the mean-and-sigma version breaks itself.

    Uses the modified z-score ``0.6745 * (x - median) / MAD``. When MAD is zero
    (a constant series with a few spikes) it falls back to the mean absolute
    deviation, and reports which estimator was used.
    """

    values = [float(v) for v in series]
    n = len(values)
    if n == 0:
        return {"samples": 0, "anomalies": [], "estimator": "none"}
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    deviations = sorted(abs(v - median) for v in values)
    mad = deviations[mid] if n % 2 else 0.5 * (deviations[mid - 1] + deviations[mid])
    if mad > 0.0:
        estimator = "mad"
        scale = mad / 0.6745
    else:
        mean_abs = sum(abs(v - median) for v in values) / n
        estimator = "mean_absolute_deviation"
        scale = (mean_abs / 0.7979) if mean_abs > 0.0 else 0.0
    if scale <= 0.0:
        return {"samples": n, "median": median, "anomalies": [], "estimator": "degenerate_constant"}
    scores = [(v - median) / scale for v in values]
    anomalies = [
        {"index": i, "value": values[i], "z": scores[i]}
        for i in range(n)
        if abs(scores[i]) >= z_threshold
    ]
    return {
        "samples": n,
        "median": median,
        "scale": scale,
        "estimator": estimator,
        "z_threshold": float(z_threshold),
        "modified_z": scores,
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# Multi-agent ecosystem: replicator dynamics over controller policies
# ---------------------------------------------------------------------------


def replicator_step(
    population: Sequence[float], payoffs: Sequence[float]
) -> List[float]:
    """One discrete replicator step: ``x_i' = x_i * f_i / mean(f)``.

    Deterministic and mass-preserving. Strategies that beat the population mean
    grow; the rest shrink. Payoffs must be non-negative -- a negative fitness
    has no meaning in this update, so it is rejected rather than clipped.
    """

    if len(population) != len(payoffs):
        raise ValueError("population and payoffs must be the same length")
    if any(p < 0.0 for p in payoffs):
        raise ValueError("replicator dynamics require non-negative payoffs")
    total = sum(population)
    if total <= 0.0:
        raise ValueError("population must have positive mass")
    shares = [float(x) / total for x in population]
    mean_fitness = sum(s * f for s, f in zip(shares, payoffs))
    if mean_fitness <= 0.0:
        return shares
    updated = [s * f / mean_fitness for s, f in zip(shares, payoffs)]
    normaliser = sum(updated)
    return [u / normaliser for u in updated]


def run_ecosystem(
    population: Sequence[float], payoffs: Sequence[float], steps: int = 32
) -> Dict[str, object]:
    """Iterate :func:`replicator_step` and report where the population settles."""

    if steps < 1:
        raise ValueError("steps must be >= 1")
    current = list(population)
    trajectory = [list(_normalise(current))]
    for _ in range(steps):
        current = replicator_step(current, payoffs)
        trajectory.append(list(current))
    dominant = max(range(len(current)), key=lambda i: current[i])
    return {
        "steps": steps,
        "final_shares": current,
        "dominant_strategy": dominant,
        "dominant_share": current[dominant],
        "converged": abs(current[dominant] - trajectory[-2][dominant]) < 1e-6,
        "trajectory": trajectory,
    }


def _normalise(values: Sequence[float]) -> List[float]:
    total = sum(values)
    if total <= 0.0:
        raise ValueError("cannot normalise a non-positive vector")
    return [float(v) / total for v in values]


# ---------------------------------------------------------------------------
# Q-learning feedback into the thinking controller
# ---------------------------------------------------------------------------


@dataclass
class BudgetPolicyLearner:
    """Tabular Q-learning over ``(difficulty bucket, risk bucket) -> budget``.

    This is the Dem-Lab Q-learning panel doing real work: it observes what the
    thinking controller actually spent and whether the accepted decision
    matched the ceiling decision, and learns a *starting budget* per request
    bucket.

    Boundaries, deliberately narrow:

    * it proposes a **starting** budget only. Policy floors, ceilings, the
      verifier gate, and the cross-budget agreement rule all still apply, so a
      badly-learned value cannot authorise an unsafe early exit -- at worst it
      wastes compute or triggers one extra probe.
    * updates are deterministic given the same observation order. There is no
      exploration noise; exploration is the caller's business.
    * an unvisited bucket returns ``None``, not a guess.
    """

    budgets: Tuple[int, ...] = (1, 2, 4)
    buckets: int = 3
    learning_rate: float = 0.2
    fidelity_weight: float = 1.0
    cost_weight: float = 0.15
    table: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)
    visits: Dict[Tuple[int, int], List[int]] = field(default_factory=dict)

    def bucket(self, value: float) -> int:
        clamped = min(1.0, max(0.0, float(value)))
        index = int(clamped * self.buckets)
        return min(self.buckets - 1, index)

    def state(self, difficulty: float, risk: float) -> Tuple[int, int]:
        return (self.bucket(difficulty), self.bucket(risk))

    def reward(self, decision_matched_ceiling: bool, cycles_spent: int, ceiling_cycles: int) -> float:
        """Fidelity first, cost second. Getting the answer wrong is never cheap."""

        fidelity = 1.0 if decision_matched_ceiling else 0.0
        normalised_cost = cycles_spent / max(1, ceiling_cycles)
        return self.fidelity_weight * fidelity - self.cost_weight * normalised_cost

    def observe(
        self,
        difficulty: float,
        risk: float,
        budget: int,
        decision_matched_ceiling: bool,
        cycles_spent: int,
        ceiling_cycles: int,
    ) -> float:
        if budget not in self.budgets:
            raise ValueError(f"budget {budget} is not one of {self.budgets}")
        key = self.state(difficulty, risk)
        action = self.budgets.index(int(budget))
        row = self.table.setdefault(key, [0.0] * len(self.budgets))
        counts = self.visits.setdefault(key, [0] * len(self.budgets))
        reward = self.reward(decision_matched_ceiling, cycles_spent, ceiling_cycles)
        row[action] += self.learning_rate * (reward - row[action])
        counts[action] += 1
        return reward

    def suggest(self, difficulty: float, risk: float, min_visits: int = 3) -> Optional[int]:
        """Best-known starting budget, or ``None`` if the bucket is unproven."""

        key = self.state(difficulty, risk)
        row = self.table.get(key)
        counts = self.visits.get(key)
        if row is None or counts is None:
            return None
        eligible = [i for i, c in enumerate(counts) if c >= min_visits]
        if not eligible:
            return None
        # Ties break toward the cheaper budget: an untested tie is not a reason
        # to spend more.
        best = min(eligible, key=lambda i: (-row[i], self.budgets[i]))
        return int(self.budgets[best])

    def to_dict(self) -> Dict[str, object]:
        return {
            "budgets": list(self.budgets),
            "buckets": self.buckets,
            "learning_rate": self.learning_rate,
            "table": {f"{k[0]},{k[1]}": v for k, v in sorted(self.table.items())},
            "visits": {f"{k[0]},{k[1]}": v for k, v in sorted(self.visits.items())},
        }


# ---------------------------------------------------------------------------
# The observatory front end
# ---------------------------------------------------------------------------


@dataclass
class Observatory:
    """Accumulates telemetry across turns and reports the full dashboard.

    Usage::

        obs = Observatory()
        obs.record(output.telemetry, decision=decision, tokens=new_tokens.tolist()[0])
        report = obs.report()

    Everything in ``report()`` is JSON-safe and deterministic given the same
    recorded history.
    """

    history: List[Dict[str, object]] = field(default_factory=list)
    token_history: List[List[int]] = field(default_factory=list)
    learner: BudgetPolicyLearner = field(default_factory=BudgetPolicyLearner)

    def record(
        self,
        telemetry: Mapping[str, object],
        decision: Optional[object] = None,
        tokens: Optional[Sequence[int]] = None,
    ) -> None:
        entry: Dict[str, object] = {"telemetry": dict(telemetry)}
        if decision is not None:
            entry["decision"] = decision.to_dict() if hasattr(decision, "to_dict") else dict(decision)
        if tokens is not None:
            entry["tokens"] = [int(t) for t in tokens]
        self.history.append(entry)
        if tokens is not None:
            self.token_history.append([int(t) for t in tokens])

    def _series(self, path: Sequence[str]) -> List[float]:
        values: List[float] = []
        for entry in self.history:
            node: object = entry
            for key in path:
                if not isinstance(node, Mapping) or key not in node:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                values.append(float(node))
        return values

    def report(self, vocab_size: Optional[int] = None) -> Dict[str, object]:
        if not self.history:
            return {"turns": 0}

        latest = self.history[-1]["telemetry"]
        sink_series = self._series(("telemetry", "mean_sink_mass"))
        cycles_series = self._series(("telemetry", "thinking", "cycles_used"))
        continue_series = self._series(("telemetry", "thinking", "continue_probability"))

        report: Dict[str, object] = {
            "turns": len(self.history),
            "attribution": routing_attribution(latest),
            "stability": {
                "attention_sink_mass": stability_score(sink_series),
                "thinking_cycles": stability_score(cycles_series),
                "continue_probability": stability_score(continue_series),
            },
            "anomalies": {
                "attention_sink_mass": robust_anomalies(sink_series),
                "thinking_cycles": robust_anomalies(cycles_series),
            },
            "policy": self.learner.to_dict(),
        }

        if self.token_history:
            current = self.token_history[-1]
            report["novelty"] = novelty_score(current, self.token_history[:-1], n=3)
            if vocab_size:
                report["randomness"] = randomness_report(current, int(vocab_size))

        quality_series = self._series(("telemetry", "thinking", "quality_probability"))
        if len(quality_series) >= 2 and len(cycles_series) >= 2 and self.token_history:
            report["rsi"] = recursive_improvement_index(
                novelty=float(report.get("novelty", {}).get("novelty", 0.0)),
                stability=float(report["stability"]["thinking_cycles"]["stability"]),
                quality_delta=quality_series[-1] - quality_series[0],
                cost_delta=cycles_series[-1] - cycles_series[0],
            )
        return report
