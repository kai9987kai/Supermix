"""Behavioural harness for multi-turn conversation quality.

Everything measured for this project so far has been single-turn retrieval. The
conversation layer in `conversation_state.py` — durable user commitments, open
questions, topic threads, repetition — was built and wired into ranking without
anything ever checking that it helps. This harness is that check.

It cannot be built from the corpus. Every `context_text` in `llm_chat.db` carries
exactly one turn marker, so there is no real multi-turn material to draw on. The
cases here are therefore **constructed**, and that shapes what the numbers mean.

What this is
------------

A behavioural contract suite. Each case is a conversation, a next user turn, and
a candidate pool holding one good continuation, one *trap*, and filler drawn from
the real corpus. The trap is a response that is only wrong for a conversational
reason: it repeats what was already said, re-asks a question the user answered,
ignores a stated style preference, or abandons the active topic. The measurement
is whether the good continuation outranks the trap.

What this is not
----------------

Not a validation corpus and not evidence about real traffic. The cases were
written by inspecting the system, so they are biased towards failures the author
could imagine. A pass rate here means the ranker respects a contract on
constructed probes; it does not estimate how often the contract holds in use.

The value is in the trap categories being *separable*: a change can be shown to
fix repetition without disturbing topic continuity, which single-turn top-1
cannot express.

Usage
-----

    python source/benchmark_conversation_quality.py
    python source/benchmark_conversation_quality.py --json
    python source/benchmark_conversation_quality.py --bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chat_pipeline import (  # noqa: E402
    featurize_text,
    infer_style_mode,
    rank_response_candidates,
)
from conversation_state import build_conversation_state  # noqa: E402


BENCHMARK_VERSION = "supermix-conversation-quality-v1"
DEFAULT_DB = "databases/llm_chat.db"
FILLER_COUNT = 25


# Each case: history, the next user turn, a good continuation, and one trap that
# is wrong for a purely conversational reason.
CASES: Sequence[Dict[str, Any]] = (
    # ---- repetition: the assistant already said this ----
    dict(
        trap_kind="repetition",
        history=[("How do I index a Postgres table?",
                  "Use CREATE INDEX on the column you filter by most often.")],
        query="what about for text search?",
        good="For text search use a GIN index on a tsvector column.",
        trap="Use CREATE INDEX on the column you filter by most often.",
    ),
    dict(
        trap_kind="repetition",
        history=[("what is a closure?",
                  "A closure is a function that captures variables from its enclosing scope.")],
        query="show me why that matters",
        good="It matters for callbacks: each one keeps its own captured counter value.",
        trap="A closure is a function that captures variables from its enclosing scope.",
    ),
    dict(
        trap_kind="repetition",
        history=[("how do I read a file in python?",
                  "Open it with a context manager: with open(path) as handle.")],
        query="and how do I write to one",
        good="Open it in write mode: with open(path, 'w') as handle, then handle.write(text).",
        trap="Open it with a context manager: with open(path) as handle.",
    ),
    dict(
        trap_kind="repetition",
        history=[("why is my container restarting?",
                  "Check the exit code with docker inspect and read the container logs.")],
        query="the exit code is 137",
        good="Exit code 137 means the process was killed by the OOM killer. Raise the memory limit.",
        trap="Check the exit code with docker inspect and read the container logs.",
    ),

    # ---- re-asked question: the user already answered it ----
    dict(
        trap_kind="reasked_question",
        history=[("my build is failing", "Which build system are you using?")],
        query="gradle",
        good="For Gradle, run with --stacktrace to see which task failed.",
        trap="Which build system are you using?",
    ),
    dict(
        trap_kind="reasked_question",
        history=[("the query is slow", "Are you optimising for read or write throughput?")],
        query="read throughput",
        good="For read throughput, add a covering index and consider a read replica.",
        trap="Are you optimising for read or write throughput?",
    ),
    dict(
        trap_kind="reasked_question",
        history=[("help me pick a language", "Is this for scripting or for a long-lived service?")],
        query="a long lived service",
        good="For a long-lived service, prefer a statically typed language such as Go or Rust.",
        trap="Is this for scripting or for a long-lived service?",
    ),
    dict(
        trap_kind="reasked_question",
        history=[("deployment keeps failing", "Which environment is it failing in, staging or production?")],
        query="staging",
        good="In staging, compare the env vars against production and check the image tag.",
        trap="Which environment is it failing in, staging or production?",
    ),

    # ---- style: the user stated a durable preference ----
    dict(
        trap_kind="style_concise",
        history=[("Please always keep answers concise", "Understood.")],
        query="how do I list files in python",
        good="Use os.listdir(path), or Path(path).iterdir() for a Path object.",
        trap="Great question. There are several ways to list files in Python and the right "
             "one depends on your needs. You could use os.listdir, or pathlib.Path.iterdir, "
             "or glob.glob, each with tradeoffs around recursion, hidden files and "
             "performance. Let me know if you want a deeper walkthrough of each option.",
    ),
    dict(
        trap_kind="style_concise",
        history=[("keep it short please", "Got it.")],
        query="what does git rebase do",
        good="It replays your commits on top of another branch.",
        trap="Git rebase is a powerful and somewhat subtle command that takes the commits "
             "on your current branch and replays them one at a time on top of another "
             "branch, which rewrites history and can be confusing if you have already "
             "pushed. There are interactive and non-interactive forms, and each has "
             "tradeoffs worth understanding before you use them on shared branches.",
    ),
    dict(
        trap_kind="style_concise",
        history=[("I prefer short answers", "Noted.")],
        query="how do I check python version",
        good="Run python --version.",
        trap="There are a number of ways to check which version of Python you are running, "
             "and which one you want depends on whether you mean the interpreter on your "
             "PATH, the one inside a virtual environment, or the one a particular script "
             "will use when executed. The most direct approach is running python --version.",
    ),
    dict(
        trap_kind="style_concise",
        history=[("be brief", "Okay.")],
        query="what is a docker volume",
        good="A volume is storage that outlives the container.",
        trap="A Docker volume is the mechanism Docker provides for persisting data beyond "
             "the lifetime of any individual container, which matters because container "
             "filesystems are ephemeral and everything written to them disappears when the "
             "container is removed. Volumes can be named or anonymous, and can be mounted "
             "into several containers at once for sharing state between them.",
    ),

    # ---- a fresh explicit request must outrank a standing preference ----
    # Without this contract, remembering "be brief" is worse than having no
    # memory: the user asks for depth and is refused it by their own earlier
    # instruction.
    dict(
        trap_kind="standing_yields_to_fresh",
        history=[("be brief", "Okay.")],
        query="explain that in detail",
        good="In detail: the scheduler assigns each task a priority, then walks the "
             "queue in priority order, preempting anything lower when a higher "
             "priority task arrives, which is why starvation is possible.",
        trap="It uses priority order.",
    ),
    dict(
        trap_kind="standing_yields_to_fresh",
        history=[("Please always keep answers concise", "Understood."),
                 ("what is a database index?", "A sorted structure for fast lookup.")],
        query="can you elaborate",
        good="An index stores keys in sorted order with pointers to rows, so a lookup "
             "walks a tree instead of scanning every row. The cost is slower writes "
             "and extra storage, since each write must maintain the index too.",
        trap="A sorted structure for fast lookup.",
    ),
    dict(
        trap_kind="standing_yields_to_fresh",
        history=[("keep it short please", "Got it.")],
        query="walk me through how tls works step by step",
        good="First the client sends a hello with supported ciphers. The server picks "
             "one and returns its certificate. The client verifies it, they derive a "
             "shared key, and all later traffic is encrypted with that key.",
        trap="It encrypts traffic.",
    ),

    # ---- topic continuity: the user is still on the same subject ----
    dict(
        trap_kind="topic_drift",
        history=[("How do I speed up a slow Postgres query?",
                  "Start by running EXPLAIN ANALYZE to find the slow step.")],
        query="it says sequential scan",
        good="A sequential scan means no usable index. Add one on the filtered column.",
        trap="Docker containers share the host kernel, unlike virtual machines.",
    ),
    dict(
        trap_kind="topic_drift",
        history=[("my python script leaks memory",
                  "Use tracemalloc to find which allocations grow between snapshots.")],
        query="the growth is in a list",
        good="A growing list usually means you are appending without ever clearing it.",
        trap="TLS encrypts traffic between the client and the server.",
    ),
    dict(
        trap_kind="topic_drift",
        history=[("help me write unit tests",
                  "Start with the core logic and the edge cases around it.")],
        query="what about mocking",
        good="Mock only what crosses a boundary: network, filesystem, clock.",
        trap="A CDN caches static assets closer to the user to cut latency.",
    ),
    dict(
        trap_kind="topic_drift",
        history=[("how do I reduce docker image size?",
                  "Use a smaller base image and a multi-stage build.")],
        query="my final stage is still large",
        good="Copy only the built artifact into the final stage and drop the build tools.",
        trap="Newton's third law states every action has an equal and opposite reaction.",
    ),
)


def load_filler(db_path: Path, count: int = FILLER_COUNT) -> List[Dict[str, Any]]:
    """Realistic distractors so the pool is not just the good answer and a trap."""

    if not db_path.exists():
        return []
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT user_text, response_text, count FROM llm_entries "
            "WHERE length(response_text) > 40 ORDER BY id LIMIT ?",
            (int(count),),
        ).fetchall()
    finally:
        connection.close()
    return [
        {
            "query": str(row["user_text"] or ""),
            "text": str(row["response_text"] or ""),
            "count": max(1, int(row["count"] or 1)),
        }
        for row in rows
    ]


def _candidate(text: str, query: str, count: int) -> Dict[str, Any]:
    response_vector = featurize_text(text)
    context_vector = featurize_text(query)
    return {
        "text": text,
        "vec": response_vector.tolist(),
        "ctx_vec": context_vector.tolist(),
        "count": count,
        "bucket_score": 0.5,
    }


def evaluate_case(
    case: Dict[str, Any],
    filler: Sequence[Dict[str, Any]],
    use_state: bool,
) -> Tuple[bool, int, int, str]:
    """Return (good_beats_trap, good_rank, trap_rank, resolved_style_mode)."""

    candidates = [
        _candidate(case["good"], case["query"], 1),
        _candidate(case["trap"], case["query"], 1),
    ]
    for row in filler:
        candidates.append(_candidate(row["text"], row["query"], row["count"]))

    recent = [reply for _, reply in case["history"] if reply]
    state = (
        build_conversation_state(case["history"], current_user_text=case["query"])
        if use_state
        else None
    )
    # Mirror the apps: they resolve a style mode first, then rank. Calling the
    # ranker directly with a defaulted mode skips the path a standing style
    # preference travels, so the harness would report no effect from a fix that
    # works in production.
    resolved_style = infer_style_mode(
        case["query"], requested_mode="auto", conversation_state=state
    )
    order, _ = rank_response_candidates(
        candidates,
        case["query"],
        recent,
        style_mode=resolved_style,
        conversation_state=state,
    )
    good_rank = order.index(0) + 1
    trap_rank = order.index(1) + 1
    return good_rank < trap_rank, good_rank, trap_rank, resolved_style


def _bootstrap_pass_rate(
    outcomes: Sequence[bool], iterations: int, seed: int
) -> Tuple[float, float]:
    if not outcomes:
        return 0.0, 0.0
    rng = random.Random(seed)
    size = len(outcomes)
    samples = []
    for _ in range(iterations):
        samples.append(
            sum(outcomes[rng.randrange(size)] for _ in range(size)) / size
        )
    samples.sort()
    return samples[int(0.025 * iterations)], samples[int(0.975 * iterations)]


def run(bootstrap: int = 4000, db_path: Path | None = None) -> Dict[str, Any]:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent.parent
    filler = load_filler(db_path or (root / DEFAULT_DB))

    kinds = sorted({case["trap_kind"] for case in CASES})
    report: Dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": len(CASES),
        "filler_per_pool": len(filler),
        "fixture": "constructed behavioural probes, not real traffic",
        "bootstrap_samples": bootstrap,
        "by_state": {},
    }

    for label, use_state in (("without_state", False), ("with_state", True)):
        per_kind: Dict[str, Any] = {}
        all_outcomes: List[bool] = []
        for kind in kinds:
            outcomes = []
            details = []
            for case in CASES:
                if case["trap_kind"] != kind:
                    continue
                passed, good_rank, trap_rank, style = evaluate_case(case, filler, use_state)
                outcomes.append(passed)
                details.append({
                    "query": case["query"],
                    "passed": passed,
                    "good_rank": good_rank,
                    "trap_rank": trap_rank,
                    "style_mode": style,
                })
            all_outcomes.extend(outcomes)
            per_kind[kind] = {
                "cases": len(outcomes),
                "passed": sum(outcomes),
                "pass_rate": round(sum(outcomes) / max(1, len(outcomes)), 4),
                "details": details,
            }
        low, high = _bootstrap_pass_rate(all_outcomes, bootstrap, seed=17)
        report["by_state"][label] = {
            "overall_pass_rate": round(sum(all_outcomes) / max(1, len(all_outcomes)), 4),
            "overall_ci": [round(low, 4), round(high, 4)],
            "by_trap_kind": per_kind,
        }

    return report


def print_report(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"CONVERSATION QUALITY  {report['benchmark_version']}")
    print(f"{report['case_count']} constructed cases, "
          f"{report['filler_per_pool']} corpus filler candidates per pool")
    print("NOTE: constructed probes, not real traffic. Pass rate is a contract "
          "check,\n      not an estimate of behaviour in use.")
    print("=" * 78)

    kinds = sorted(report["by_state"]["without_state"]["by_trap_kind"])
    print(f"\n  {'trap kind':22} {'without state':>16} {'with state':>16}")
    print("  " + "-" * 56)
    for kind in kinds:
        off = report["by_state"]["without_state"]["by_trap_kind"][kind]
        on = report["by_state"]["with_state"]["by_trap_kind"][kind]
        print(f"  {kind:22} {off['passed']:>7}/{off['cases']:<8} {on['passed']:>7}/{on['cases']:<8}")

    print()
    for label in ("without_state", "with_state"):
        block = report["by_state"][label]
        low, high = block["overall_ci"]
        print(f"  {label:14} overall {block['overall_pass_rate']*100:5.1f}%  "
              f"[{low*100:.0f}, {high*100:.0f}]")

    failures = [
        (kind, detail)
        for kind, block in report["by_state"]["with_state"]["by_trap_kind"].items()
        for detail in block["details"]
        if not detail["passed"]
    ]
    if failures:
        print(f"\n  still failing with state ({len(failures)}):")
        for kind, detail in failures:
            print(f"    [{kind}] good rank {detail['good_rank']}, "
                  f"trap rank {detail['trap_rank']}, style {detail['style_mode']}"
                  f"  <- {detail['query'][:38]}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--db", default=None)
    args = parser.parse_args(argv)

    report = run(bootstrap=args.bootstrap, db_path=Path(args.db) if args.db else None)
    if args.json:
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
