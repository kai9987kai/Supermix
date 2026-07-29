"""Retrieval-quality harness for the response reranker.

`rank_response_candidates` fuses about twenty signals with hand-tuned weights,
and until now there was no way to tell whether a change to it helped, hurt, or
did nothing. Every weight in that function is a judgement call that nothing
measured.

This harness fixes that narrow gap: top-1 / top-3 / MRR per fusion mode, with
paired bootstrap confidence intervals.

It prefers the corpus fixture written by `build_ranking_eval_set.py` — real
(query, gold response) pairs drawn from `llm_chat.db`, with hard negatives — and
falls back to the hand-written probes below when that file is absent.

Two properties carry the result, and both were arrived at by getting them wrong
first:

**Paired intervals.** Differences here are small and the probe set is finite. An
unpaired eyeball comparison reads sampling noise as a result. Only a paired
interval that excludes zero is reported as significant.

**Equivalence-aware scoring.** The corpus stores many responses that say the same
thing with different framing, and exactly one is labelled gold. Scoring only the
labelled string counted 58% of apparent failures against responses that were
equivalent to the gold, reporting 70.7% top-1 where the system was really at
87.7%. Strict scoring is still available via `--strict`, but it measures the
labelling, not the ranking.

Scope, stated plainly:

* This is a **regression and comparison harness**, not a validation corpus. A win
  here is evidence that a change did something on this fixture, not evidence
  that the system is smarter.
* It measures the *ranking* only, holding the featurizer and the corpus fixed,
  so it says nothing about the checkpoint.
* `bucket_score` now replicates the first stage of `llm_database.fetch_candidates`
  rather than being pinned to a constant. This made no material difference
  (top-1 93.0% either way), because that first-stage score is built from the same
  two cosine similarities the reranker already weights, so it carries almost no
  independent information. The genuinely independent prior — the trained
  classifier's softmax over k-means buckets, which production applies to
  *model-bucket* candidates rather than DB candidates — is still not measured
  here, because this fixture contains only DB-style candidates.
* Absolute numbers are not comparable across fixtures. Only differences between
  modes on the same run mean anything.

Usage
-----

    python source/benchmark_ranking_quality.py
    python source/benchmark_ranking_quality.py --json
    python source/benchmark_ranking_quality.py --bootstrap 10000
    python source/benchmark_ranking_quality.py --strict
    python source/benchmark_ranking_quality.py --constant-bucket-score
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_pipeline import featurize_text, rank_response_candidates  # noqa: E402
from score_fusion import FUSION_MODES  # noqa: E402


BENCHMARK_VERSION = "supermix-ranking-quality-v1"

# (response text, domain tag)
CORPUS: Sequence[Tuple[str, str]] = (
    ("Use CREATE INDEX on the column you filter by most often.", "db"),
    ("Partial indexes add a WHERE clause so only matching rows are stored.", "db"),
    ("A B-tree keeps keys sorted and supports efficient range scans.", "db"),
    ("Normalization reduces redundancy across relational tables.", "db"),
    ("Run EXPLAIN ANALYZE to see which step of the query plan is slow.", "db"),
    ("VACUUM reclaims storage occupied by dead tuples in Postgres.", "db"),
    ("A foreign key constraint enforces referential integrity between tables.", "db"),
    ("Connection pooling reuses database connections instead of reopening them.", "db"),
    ("A closure captures variables from its enclosing lexical scope.", "prog"),
    ("Python's GIL prevents true parallel execution of bytecode threads.", "prog"),
    ("def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)", "prog"),
    ("List comprehensions build a list in one expression and are usually faster than append loops.", "prog"),
    ("A decorator wraps a function to extend its behaviour without editing it.", "prog"),
    ("Rust ownership rules prevent data races at compile time.", "prog"),
    ("A generator yields values lazily instead of building the whole list.", "prog"),
    ("Static typing catches type errors before the program runs.", "prog"),
    ("Photosynthesis converts light energy into chemical energy in chloroplasts.", "sci"),
    ("The mitochondrion produces ATP and is often called the powerhouse of the cell.", "sci"),
    ("The speed of light is roughly 300,000 kilometres per second in vacuum.", "sci"),
    ("DNA stores genetic information as sequences of four nucleotide bases.", "sci"),
    ("Newton's third law states every action has an equal and opposite reaction.", "sci"),
    ("Entropy in a closed system tends to increase over time.", "sci"),
    ("Water boils at 100 degrees Celsius at standard atmospheric pressure.", "sci"),
    ("HTTP 429 means too many requests; back off and retry later.", "web"),
    ("Docker containers share the host kernel, unlike virtual machines.", "web"),
    ("A CDN caches static assets closer to the user to cut latency.", "web"),
    ("TLS encrypts traffic between the client and the server.", "web"),
    ("A vector database stores embeddings for similarity search.", "web"),
    ("Load balancers spread incoming requests across several backend servers.", "web"),
    ("A reverse proxy sits in front of servers and forwards client requests.", "web"),
)

# (query, index of the one correct response in CORPUS)
PROBES: Sequence[Tuple[str, int]] = (
    ("how do I find which part of my query is slow", 4),
    ("what does EXPLAIN ANALYZE do", 4),
    ("how do I speed up filtering on a column", 0),
    ("what is a partial index", 1),
    ("why use a b-tree for range queries", 2),
    ("how do I reclaim dead tuple storage", 5),
    ("what is database normalization", 3),
    ("how do I enforce referential integrity", 6),
    ("why reuse database connections", 7),
    ("what is a closure in programming", 8),
    ("why can't python threads run in parallel", 9),
    ("write a recursive factorial function", 10),
    ("are list comprehensions faster than loops", 11),
    ("what does a decorator do", 12),
    ("how does rust avoid data races", 13),
    ("how do I produce values lazily", 14),
    ("what catches type errors before running", 15),
    ("how do plants convert light to energy", 16),
    ("what makes ATP in the cell", 17),
    ("how fast does light travel", 18),
    ("how is genetic information stored", 19),
    ("what is newton's third law", 20),
    ("what happens to entropy over time", 21),
    ("at what temperature does water boil", 22),
    ("what does http 429 mean", 23),
    ("difference between containers and vms", 24),
    ("how do I reduce latency for static files", 25),
    ("what encrypts client server traffic", 26),
    ("where are embeddings stored for search", 27),
    ("how do I spread requests across servers", 28),
    ("what forwards client requests to backends", 29),
)


EVAL_SET_PATH = Path(__file__).resolve().parent / "ranking_eval_set.json"

# The corpus stores many responses that say the same thing with different
# framing: "Okay. Hello. Tell me what you need." versus "Hello. Tell me what you
# need. Let me know if you want a deeper walkthrough." Exactly one is labelled
# gold, so strict rank scoring counts retrieving the other as a failure.
#
# Measured on the 300-probe dev split: 88 top-1 failures, of which 51 (58%) were
# near-duplicates of the gold. Strict scoring therefore reported 70.7% top-1
# where equivalence-aware scoring reports 87.7%. Chasing that 17-point gap would
# be optimising a labelling artefact, and worse, it rewards changes that shuffle
# equivalent responses while penalising none of the genuinely wrong retrievals.
_BOILERPLATE_PREFIX_RE = re.compile(
    r"^(?:okay|ok|got it|understood|short answer|sure|alright|hi there|yes|great)[.:,]?\s*",
    re.IGNORECASE,
)
_BOILERPLATE_SUFFIX_RE = re.compile(
    r"(?:let me know if.*|if you share logs.*|send the next.*|"
    r"tell me what you need.*|i can be more specific.*)$",
    re.IGNORECASE,
)
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")

# Two responses count as equivalent above this Jaccard overlap of content words.
EQUIVALENCE_THRESHOLD = 0.6


def response_core(text: Any) -> frozenset:
    """Content words of a response, with corpus boilerplate stripped."""

    cooked = str(text or "").strip()
    for _ in range(3):
        cooked = _BOILERPLATE_PREFIX_RE.sub("", cooked).strip()
    cooked = _BOILERPLATE_SUFFIX_RE.sub("", cooked).strip()
    return frozenset(_NON_WORD_RE.sub(" ", cooked.lower()).split())


def responses_are_equivalent(left: Any, right: Any) -> bool:
    first, second = response_core(left), response_core(right)
    if not first or not second:
        return False
    return len(first & second) / len(first | second) >= EQUIVALENCE_THRESHOLD


def load_eval_set(path: Path = EVAL_SET_PATH) -> Dict[str, Any] | None:
    """The corpus-derived fixture, when `build_ranking_eval_set.py` has run.

    Preferred over the hand-written probes: it is drawn from real query/response
    pairs and uses hard negatives, so it can actually discriminate between
    ranking changes. With random negatives the reranker scores 100% top-1 and
    the measurement is worthless.
    """

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if payload.get("cases") else None


# Replicates the first stage in `llm_database.fetch_candidates`. Production does
# NOT hand the reranker a constant prior: every DB candidate arrives carrying
# sigmoid() of a first-stage retrieval score, and `rank_response_candidates`
# weights that at 0.14-0.18 as `bucket_bonus`. Holding it constant makes the
# bucket term identical across candidates, so it drops out of the ranking and the
# reranker is measured without its strongest prior.
#
# `bm25_penalty` and `exact_bonus` are omitted: the fixture has no FTS ranks, and
# a hard-negative set contains no exact user-text matches. Both are small
# relative to the similarity terms, but this is an approximation of the first
# stage, not a reproduction of it.
FIRST_STAGE_CTX_WEIGHT = 0.72
FIRST_STAGE_RESP_WEIGHT = 0.28
FIRST_STAGE_COUNT_WEIGHT = 0.04


def first_stage_bucket_score(query_vector, response_vector, context_vector, count: int) -> float:
    sim_ctx = float(torch.dot(context_vector, query_vector))
    sim_resp = float(torch.dot(response_vector, query_vector))
    score = (
        FIRST_STAGE_CTX_WEIGHT * sim_ctx
        + FIRST_STAGE_RESP_WEIGHT * sim_resp
        + FIRST_STAGE_COUNT_WEIGHT * math.log1p(max(1, int(count)))
    )
    return float(torch.sigmoid(torch.tensor(score)))


def _corpus_gold_ranks(
    payload: Dict[str, Any],
    mode: str,
    strict: bool = False,
    real_bucket_score: bool = True,
) -> List[int]:
    ranks: List[int] = []
    for case in payload["cases"]:
        rows = [{
            "text": case["gold"],
            "query": case["query"],
            "count": max(1, int(case.get("gold_count", 1))),
        }]
        rows.extend(
            {
                "text": row["text"],
                "query": row.get("query") or row["text"],
                "count": max(1, int(row.get("count", 1))),
            }
            for row in case["distractors"]
        )
        candidates = []
        query_vector = featurize_text(case["query"])
        for row in rows:
            # Mirror production: `vec` is the response embedding, `ctx_vec` is
            # the embedding of the context the response was stored under.
            vector = featurize_text(row["text"])
            context_vector = featurize_text(row["query"])
            candidates.append({
                "text": row["text"],
                "vec": vector.tolist(),
                "ctx_vec": context_vector.tolist(),
                "count": row["count"],
                "bucket_score": (
                    first_stage_bucket_score(query_vector, vector, context_vector, row["count"])
                    if real_bucket_score
                    else 0.5
                ),
            })
        order, _ = rank_response_candidates(
            candidates, case["query"], [], fusion_mode=mode,
            allow_experimental_fusion=True,
        )
        if strict:
            ranks.append(order.index(0) + 1)
        else:
            # Credit the best-ranked candidate that says what the gold says.
            gold = case["gold"]
            best = order.index(0) + 1
            for position, index in enumerate(order, start=1):
                if position >= best:
                    break
                if responses_are_equivalent(rows[index]["text"], gold):
                    best = position
                    break
            ranks.append(best)
    return ranks


def build_candidates() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, (text, _domain) in enumerate(CORPUS):
        vector = featurize_text(text)
        rows.append({
            "text": text,
            "vec": vector.tolist(),
            "ctx_vec": vector.tolist(),
            # Deterministic, mode-independent stand-ins for corpus statistics.
            "count": 1 + (index % 4),
            "bucket_score": 0.25 + 0.03 * (index % 6),
        })
    return rows


def gold_ranks(candidates: Sequence[Dict[str, Any]], mode: str) -> List[int]:
    """1-based rank of the gold response for each probe."""

    ranks: List[int] = []
    for query, gold in PROBES:
        ranked, _ = rank_response_candidates(
            list(candidates), query, [], fusion_mode=mode,
            allow_experimental_fusion=True,
        )
        ranks.append(ranked.index(gold) + 1)
    return ranks


def _metrics(ranks: Sequence[int]) -> Dict[str, float]:
    count = max(1, len(ranks))
    return {
        "top1": sum(1 for r in ranks if r == 1) / count,
        "top3": sum(1 for r in ranks if r <= 3) / count,
        "mrr": sum(1.0 / r for r in ranks) / count,
    }


def _bootstrap_ci(
    ranks: Sequence[int],
    metric: str,
    samples: int,
    seed: int = 20260728,
) -> Tuple[float, float]:
    """Percentile bootstrap interval over probes."""

    if not ranks:
        return (0.0, 0.0)
    rng = random.Random(seed)
    count = len(ranks)
    draws: List[float] = []
    for _ in range(max(1, samples)):
        resampled = [ranks[rng.randrange(count)] for _ in range(count)]
        draws.append(_metrics(resampled)[metric])
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return (low, high)


def _paired_delta_ci(
    baseline: Sequence[int],
    variant: Sequence[int],
    metric: str,
    samples: int,
    seed: int = 20260728,
) -> Tuple[float, float, float]:
    """Paired bootstrap on the difference, which is the honest comparison.

    The modes are evaluated on identical probes, so resampling them jointly
    removes probe difficulty from the comparison and gives a much tighter,
    fairer interval than comparing two independent intervals.
    """

    rng = random.Random(seed)
    count = min(len(baseline), len(variant))
    if count == 0:
        return (0.0, 0.0, 0.0)
    observed = _metrics(variant)[metric] - _metrics(baseline)[metric]
    draws: List[float] = []
    for _ in range(max(1, samples)):
        picks = [rng.randrange(count) for _ in range(count)]
        base_sample = [baseline[i] for i in picks]
        var_sample = [variant[i] for i in picks]
        draws.append(_metrics(var_sample)[metric] - _metrics(base_sample)[metric])
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return (observed, low, high)


def run(
    bootstrap: int = 4000,
    modes: Sequence[str] = FUSION_MODES,
    use_corpus: bool = True,
    strict: bool = False,
    real_bucket_score: bool = True,
) -> Dict[str, Any]:
    torch.set_num_threads(1)

    payload = load_eval_set() if use_corpus else None
    if payload is not None:
        ranks_by_mode = {
            mode: _corpus_gold_ranks(
                payload, mode, strict=strict, real_bucket_score=real_bucket_score
            )
            for mode in modes
        }
        fixture = {
            "kind": "corpus",
            "eval_set_version": payload.get("eval_set_version"),
            "negatives": payload.get("negatives"),
            "fingerprint": payload.get("fingerprint"),
            "distractors_per_probe": payload.get("distractors_per_probe"),
            "scoring": "strict" if strict else "equivalence_aware",
            "bucket_score": "first_stage" if real_bucket_score else "constant",
        }
        probe_count = len(payload["cases"])
        corpus_size = int(payload.get("distractors_per_probe", 0)) + 1
    else:
        candidates = build_candidates()
        ranks_by_mode = {mode: gold_ranks(candidates, mode) for mode in modes}
        fixture = {"kind": "synthetic", "negatives": "hand-written"}
        probe_count = len(PROBES)
        corpus_size = len(CORPUS)
    baseline = ranks_by_mode.get("legacy") or next(iter(ranks_by_mode.values()))

    report: Dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "fixture": fixture,
        "corpus_size": corpus_size,
        "probe_count": probe_count,
        "bootstrap_samples": int(bootstrap),
        "modes": {},
    }
    for mode, ranks in ranks_by_mode.items():
        entry: Dict[str, Any] = {"metrics": {}, "deltas_vs_legacy": {}}
        for metric in ("top1", "top3", "mrr"):
            value = _metrics(ranks)[metric]
            low, high = _bootstrap_ci(ranks, metric, bootstrap)
            entry["metrics"][metric] = {
                "value": round(value, 6),
                "ci95": [round(low, 6), round(high, 6)],
            }
            observed, dlow, dhigh = _paired_delta_ci(baseline, ranks, metric, bootstrap)
            entry["deltas_vs_legacy"][metric] = {
                "delta": round(observed, 6),
                "ci95": [round(dlow, 6), round(dhigh, 6)],
                # A difference whose interval straddles zero is not a result.
                "significant": bool(dlow > 0.0 or dhigh < 0.0),
            }
        report["modes"][mode] = entry
    return report


def _print(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"RANKING QUALITY  {report['benchmark_version']}")
    fixture = report.get("fixture") or {}
    print(f"{report['probe_count']} probes over {report['corpus_size']} candidates, "
          f"{report['bootstrap_samples']} bootstrap samples")
    print(f"fixture: {fixture.get('kind')} / {fixture.get('negatives')} negatives"
          f" / scoring {fixture.get('scoring', 'strict')}"
          f" / bucket {fixture.get('bucket_score', 'constant')}"
          + (f" / {fixture.get('fingerprint')}" if fixture.get("fingerprint") else ""))
    print("=" * 78)
    print(f"\n  {'mode':12} {'top-1':>20} {'top-3':>20} {'MRR':>20}")
    print("  " + "-" * 74)
    for mode, entry in report["modes"].items():
        cells = []
        for metric in ("top1", "top3", "mrr"):
            row = entry["metrics"][metric]
            low, high = row["ci95"]
            if metric == "mrr":
                cells.append(f"{row['value']:.3f} [{low:.3f},{high:.3f}]")
            else:
                cells.append(f"{row['value']*100:.1f}% [{low*100:.0f},{high*100:.0f}]")
        print(f"  {mode:12} {cells[0]:>20} {cells[1]:>20} {cells[2]:>20}")

    print("\n  paired difference vs legacy (95% CI; significant only if it excludes 0):")
    for mode, entry in report["modes"].items():
        if mode == "legacy":
            continue
        parts = []
        for metric in ("top1", "mrr"):
            row = entry["deltas_vs_legacy"][metric]
            low, high = row["ci95"]
            mark = "*" if row["significant"] else " "
            parts.append(f"{metric} {row['delta']:+.4f} [{low:+.4f},{high:+.4f}]{mark}")
        print(f"    {mode:12} " + "   ".join(parts))

    significant = [
        mode
        for mode, entry in report["modes"].items()
        if mode != "legacy"
        and any(entry["deltas_vs_legacy"][m]["significant"] for m in ("top1", "mrr"))
    ]
    print()
    if significant:
        print(f"  Modes with a statistically distinguishable difference: {', '.join(significant)}")
    else:
        print("  No mode differs from legacy beyond sampling noise on this probe set.")
        print("  Treat any apparent ranking change as unmeasured, not as an improvement.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--modes", nargs="*", default=list(FUSION_MODES))
    parser.add_argument("--constant-bucket-score", action="store_true",
                        help="Pin bucket_score to 0.5, removing the first-stage prior.")
    parser.add_argument("--strict", action="store_true",
                        help="Score only the labelled gold, counting an equivalent "
                             "response as a failure. Reports ~17 points lower.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force the hand-written probes instead of the corpus fixture.")
    args = parser.parse_args(argv)

    report = run(bootstrap=args.bootstrap, modes=tuple(args.modes), use_corpus=not args.synthetic, strict=args.strict,
                 real_bucket_score=not args.constant_bucket_score)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
