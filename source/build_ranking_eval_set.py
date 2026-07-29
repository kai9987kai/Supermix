"""Build a labelled retrieval evaluation set from the real chat corpus.

The hand-written probe set in `benchmark_ranking_quality.py` is a smoke fixture:
31 queries, and small enough that a single query moves top-1 by three points. It
cannot settle whether a ranking change helps.

`llm_chat.db` already contains the labels. Each row pairs a `user_text` with the
`response_text` stored for it, so a row whose `user_text` appears exactly once is
an unambiguous (query, gold response) pair. There are 57,934 such rows.

Two design decisions carry the usefulness of the output:

**Hard negatives.** With randomly drawn distractors the existing reranker scores
100% top-1, because an unrelated response is trivially dissimilar. That measures
nothing. Drawing the distractors from the query's nearest neighbours in context
space instead drops it to 75.8%, which leaves real headroom for a change to move.

**Text, not vectors.** The fixture stores response text and re-featurizes at
evaluation time, so it is a small file rather than tens of megabytes of floats
and does not require the 460 MB database to be present. The stored corpus
vectors are averaged over duplicate rows, so re-featurizing is not bit-identical
to what production retrieves with; the fixture is for comparing ranking changes
against each other, not for predicting production accuracy.

Stated limits:

* `bucket_score` is held constant. The classifier's contribution is deliberately
  excluded so the fixture measures the *text-signal fusion*, which is the part
  under test. A change that only affects `bucket_score` handling will not show up
  here.
* Singleton rows skew short and specific, since a repeated query is more likely
  to be a greeting. This is not a uniform sample of traffic.
* Absolute numbers are not comparable across fixtures. Only differences between
  modes on the same fixture mean anything.

Usage
-----

    python source/build_ranking_eval_set.py
    python source/build_ranking_eval_set.py --probes 400 --distractors 49
    python source/build_ranking_eval_set.py --negatives random
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_pipeline import featurize_text  # noqa: E402


EVAL_SET_VERSION = "supermix-ranking-eval-set-v1"
DEFAULT_DB = "databases/llm_chat.db"
DEFAULT_OUT = "source/ranking_eval_set.json"
DEFAULT_SEED = 20260728

# Longer than a greeting, short enough to stay a single request.
MIN_QUERY_CHARS = 12
MAX_TEXT_CHARS = 600


def _load_singletons(db_path: Path, limit: int) -> List[Dict[str, Any]]:
    """Rows whose `user_text` occurs exactly once, so the gold is unambiguous."""

    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """
            SELECT e.user_text, e.response_text, e.count
            FROM llm_entries e
            JOIN (
                SELECT user_text
                FROM llm_entries
                GROUP BY user_text
                HAVING COUNT(*) = 1
            ) unique_queries ON unique_queries.user_text = e.user_text
            WHERE length(e.user_text) > ?
            ORDER BY e.id
            LIMIT ?
            """,
            (MIN_QUERY_CHARS, int(limit)),
        ).fetchall()
    finally:
        connection.close()

    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        query = str(row["user_text"] or "").strip()[:MAX_TEXT_CHARS]
        response = str(row["response_text"] or "").strip()[:MAX_TEXT_CHARS]
        if not query or not response:
            continue
        cleaned.append({
            "query": query,
            "response": response,
            "count": max(1, int(row["count"] or 1)),
        })
    return cleaned


def build(
    db_path: Path,
    probes: int,
    distractors: int,
    negatives: str = "hard",
    seed: int = DEFAULT_SEED,
    corpus_limit: int = 12000,
) -> Dict[str, Any]:
    rows = _load_singletons(db_path, corpus_limit)
    if len(rows) < probes + distractors + 1:
        raise SystemExit(
            f"corpus too small: {len(rows)} usable rows for "
            f"{probes} probes x {distractors} distractors"
        )

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    probe_rows = shuffled[:probes]
    pool_rows = shuffled[probes:]

    pool_vectors = None
    if negatives == "hard":
        pool_vectors = torch.stack(
            [featurize_text(row["query"]) for row in pool_rows], dim=0
        )

    cases: List[Dict[str, Any]] = []
    for index, probe in enumerate(probe_rows):
        if negatives == "hard":
            # Nearest neighbours in query space: responses stored for the most
            # similar other questions. This is what makes the fixture discriminative.
            similarity = torch.mv(pool_vectors, featurize_text(probe["query"]))
            picked = torch.topk(similarity, min(distractors, len(pool_rows))).indices.tolist()
        else:
            picked = random.Random(seed + index).sample(
                range(len(pool_rows)), min(distractors, len(pool_rows))
            )
        cases.append({
            "query": probe["query"],
            "gold": probe["response"],
            "gold_count": probe["count"],
            "distractors": [
                {
                    "text": pool_rows[j]["response"],
                    # Production scores a candidate's stored *context* vector
                    # against the live query (`sim_ctx`), which is the dominant
                    # signal. The fixture must carry the query each response was
                    # stored under or it measures something else entirely.
                    "query": pool_rows[j]["query"],
                    "count": pool_rows[j]["count"],
                }
                for j in picked
            ],
        })

    payload = {
        "eval_set_version": EVAL_SET_VERSION,
        "source_db": db_path.name,
        "seed": seed,
        "negatives": negatives,
        "probe_count": len(cases),
        "distractors_per_probe": distractors,
        "corpus_rows_considered": len(rows),
        "cases": cases,
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--probes", type=int, default=200)
    parser.add_argument("--distractors", type=int, default=39)
    parser.add_argument("--negatives", choices=("hard", "random"), default="hard")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--corpus-limit", type=int, default=12000)
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = root / db_path
    if not db_path.exists():
        raise SystemExit(f"corpus database not found: {db_path}")

    payload = build(
        db_path=db_path,
        probes=args.probes,
        distractors=args.distractors,
        negatives=args.negatives,
        seed=args.seed,
        corpus_limit=args.corpus_limit,
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    size_kb = out_path.stat().st_size / 1024
    print(
        f"wrote {out_path.name}: {payload['probe_count']} probes x "
        f"{payload['distractors_per_probe']} {payload['negatives']} negatives "
        f"({size_kb:.0f} KB, fingerprint {payload['fingerprint']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
