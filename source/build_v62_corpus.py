"""v62: blend a domain-balanced corpus and label every row with its domain.

V61 established that this model is **data-limited, not capacity-limited**: 3.18x
the parameters changed held-out loss by less than the noise floor. So the way to
a better model is a better corpus, and the way to know whether it worked is to
measure per domain rather than in aggregate.

That second half is the point of this module. A single perplexity cannot say
whether a model got better at arithmetic, because arithmetic is a rounding error
in a corpus dominated by dialogue. Every row emitted here carries a `domain`, so
`eval_v62_domains.py` can score each one separately and a claim like "better at
maths" becomes falsifiable.

Two measured facts drove the blend:

* The large corpora are **templated**. `supermix_plus_v27_500k` has 509,126 rows
  and 3,433 word types; `hybrid_v6_live_knowledge` has 182,034 rows and **251**
  types, thinner than the 292-type corpus v58 called its binding constraint.
  Row count is not diversity, so the big files are capped, not consumed whole.
* The vocabulary backbone is `book_extracts_public_domain_v2_120k` at 22,962
  types in its first 20,000 rows -- real literary English rather than generated
  templates. It is weighted accordingly.

Balance is by explicit per-domain caps rather than by natural proportion. Left
unbalanced, dialogue templates would supply most rows and the model would learn
the register it already speaks.

    python source/build_v62_corpus.py --output datasets/v62/v62_blend.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE = REPO_ROOT / "bundles" / "champion_datasets_current_pipeline_bundle.zip"
DEFAULT_MATH = REPO_ROOT / "datasets" / "v62" / "english_math_40k.jsonl"

RECEIPT_SCHEMA = "supermix-v62-corpus-blend-v1"

WORD = re.compile(r"[a-z0-9']+")

#: Rows whose answer is shorter than this are dropped; matches
#: `mimomix_text.load_chat_pairs`, which inherited it from a chat corpus where a
#: three-character reply was a truncation artifact.
MIN_RESPONSE_CHARACTERS = 8

#: Domains where a very short answer is the *correct* answer.
#:
#: Applying the prose minimum to arithmetic silently deleted 73.5% of the maths
#: rows -- "79", "9/14" and "59.4" are complete answers, not truncations. The
#: filter was right for dialogue and wrong for every domain whose replies are
#: values rather than sentences, so the threshold is per domain.
SHORT_ANSWER_DOMAINS = {"maths": 1, "language": 1, "vocabulary": 2}


@dataclass
class Source:
    """One contributing corpus and the domain its rows belong to."""

    domain: str
    member: str
    cap: int
    #: Keep a row only if this returns True. Used to split one file across
    #: domains by its own `category`/`task` label.
    keep: Optional[Callable[[Dict[str, Any]], bool]] = None
    note: str = ""


def _category_in(*values: str) -> Callable[[Dict[str, Any]], bool]:
    wanted = set(values)
    return lambda record: record.get("category") in wanted


def _topic_is(*values: str) -> Callable[[Dict[str, Any]], bool]:
    wanted = set(values)
    return lambda record: record.get("topic") in wanted


BUNDLE_SOURCES: Tuple[Source, ...] = (
    # Real literary English: the vocabulary backbone.
    Source("writing", "conversation_data.book_extracts_public_domain_v2_120k.jsonl", 45000,
           note="22,962 word types in the first 20k rows; the richest text available"),
    # Reasoning, split out of the creative files by their own category label.
    Source("logic", "conversation_data.mega_reasoning_creative_v25_75582.jsonl", 30000,
           keep=_category_in("chain_of_thought", "socratic", "debate")),
    Source("logic", "conversation_data.mega_creative_250k_v2.jsonl", 15000,
           keep=_category_in("chain_of_thought", "socratic", "debate")),
    Source("creativity", "conversation_data.mega_creative_250k_v2.jsonl", 30000,
           keep=_category_in("storytelling", "analogy")),
    Source("creativity", "conversation_data.mega_creative_100k.jsonl", 15000,
           keep=_category_in("storytelling", "analogy")),
    Source("conversation", "conversation_data.mega_creative_250k_v2.jsonl", 20000,
           keep=_category_in("real_conversation", "empathy")),
    Source("conversation", "conversation_data.supermix_plus_v27_500k.jsonl", 25000,
           note="509k rows but only 3,433 word types; capped hard because it is templated"),
    Source("scripture", "conversation_data.bible_kjv_public_domain_smoke.jsonl", 12000,
           note="archaic register, 10,248 types"),
    Source("vocabulary", "conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 5000,
           note="definitions; 8,112 types in 5,000 rows"),
    Source("science", "conversation_data.science_essentials_smoke.jsonl", 500),
    Source("science", "conversation_data.science_novel_examples_smoke.jsonl", 200),
    Source("coding", "conversation_data.coding_knowledge_2026_02_19.jsonl", 380),
    Source("literary_study", "conversation_data.finnegans_wake_study_noninfringing_smoke.jsonl", 1000),
)

#: The generated file, which the bundle has almost none of (100 rows).
MATH_SOURCES: Tuple[Source, ...] = (
    Source("maths", "@math", 30000, keep=_topic_is("basic_math")),
    Source("language", "@math", 10000, keep=_topic_is("english_foundations")),
)


#: See the note above `UNLEARNABLE_AT_THIS_SCALE`. Real prose leads; templated
#: dialogue is retained only so the model keeps answering in turns.
MEANING_PROFILE: Tuple[Source, ...] = (
    Source("writing", "conversation_data.book_extracts_public_domain_v2_120k.jsonl", 90000,
           note="real literary prose, 22,962 word types in the first 20k rows"),
    Source("scripture", "conversation_data.bible_kjv_public_domain_smoke.jsonl", 30000,
           note="real archaic prose, 10,248 types"),
    Source("vocabulary", "conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 5000,
           note="explicit word -> definition pairs; the only 'meaning' supervision on disk"),
    Source("literary_study", "conversation_data.finnegans_wake_study_noninfringing_smoke.jsonl", 1000),
    # Minority dialogue, to preserve the question-and-answer shape.
    Source("conversation", "conversation_data.supermix_plus_v27_500k.jsonl", 20000),
    Source("logic", "conversation_data.mega_reasoning_creative_v25_75582.jsonl", 15000,
           keep=_category_in("chain_of_thought", "socratic")),
    Source("creativity", "conversation_data.mega_creative_250k_v2.jsonl", 10000,
           keep=_category_in("storytelling", "analogy")),
    Source("science", "conversation_data.science_essentials_smoke.jsonl", 500),
    Source("coding", "conversation_data.coding_knowledge_2026_02_19.jsonl", 380),
    Source("maths", "@math", 15000, keep=_topic_is("basic_math")),
    Source("language", "@math", 8000, keep=_topic_is("english_foundations")),
)


def _clean(record: Dict[str, Any], domain: str) -> Optional[Tuple[str, str]]:
    user = str(record.get("user") or record.get("prompt") or "").strip()
    assistant = str(record.get("assistant") or record.get("response") or "").strip()
    minimum = SHORT_ANSWER_DOMAINS.get(domain, MIN_RESPONSE_CHARACTERS)
    if not user or len(assistant) < minimum:
        return None
    return user, assistant


def _iter_jsonl(handle: Iterator[str]) -> Iterator[Dict[str, Any]]:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            yield record


def collect(
    bundle: Path,
    math_path: Optional[Path],
    sources: Sequence[Source],
    seed: int,
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Read each source up to its cap, tagging every row with its domain."""

    archive = zipfile.ZipFile(bundle)
    available = set(archive.namelist())
    rows: List[Dict[str, str]] = []
    stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"rows": 0, "characters": 0, "sources": []}
    )
    skipped: List[str] = []

    for source in sources:
        if source.member == "@math":
            if math_path is None or not math_path.exists():
                skipped.append(f"{source.domain}: {math_path} missing")
                continue
            stream = math_path.open(encoding="utf-8")
        elif source.member in available:
            stream = io.TextIOWrapper(
                archive.open(source.member), encoding="utf-8", errors="replace"
            )
        else:
            skipped.append(f"{source.domain}: {source.member} not in bundle")
            continue

        taken = 0
        try:
            for record in _iter_jsonl(stream):
                if taken >= source.cap:
                    break
                if source.keep is not None and not source.keep(record):
                    continue
                cleaned = _clean(record, source.domain)
                if cleaned is None:
                    continue
                user, assistant = cleaned
                rows.append({"user": user, "assistant": assistant, "domain": source.domain})
                stats[source.domain]["rows"] += 1
                stats[source.domain]["characters"] += len(user) + len(assistant)
                taken += 1
        finally:
            stream.close()

        stats[source.domain]["sources"].append(
            {
                "member": source.member,
                "rows": taken,
                "cap": source.cap,
                "hit_cap": taken >= source.cap,
                "note": source.note,
            }
        )

    archive.close()
    random.Random(seed).shuffle(rows)
    return rows, {"per_domain": dict(stats), "skipped": skipped}


def measure(rows: Sequence[Dict[str, str]]) -> Dict[str, Any]:
    """Word-type counts overall and per domain -- the diversity the blend exists for."""

    overall: set = set()
    per_domain: Dict[str, set] = defaultdict(set)
    for row in rows:
        words = WORD.findall((row["user"] + " " + row["assistant"]).lower())
        overall.update(words)
        per_domain[row["domain"]].update(words)
    return {
        "word_types": len(overall),
        "word_types_by_domain": {k: len(v) for k, v in sorted(per_domain.items())},
    }


def select_sources(
    exclude: Sequence[str] = (), scale: float = 1.0, profile: str = "default"
) -> Tuple[Source, ...]:
    """Apply the chosen profile, domain exclusions and cap scaling."""

    base = MEANING_PROFILE if profile == "meaning" else tuple(BUNDLE_SOURCES) + tuple(MATH_SOURCES)
    dropped = set(exclude)
    chosen = []
    for source in base:
        if source.domain in dropped:
            continue
        cap = source.cap if scale == 1.0 else max(1, int(source.cap * scale))
        chosen.append(
            Source(source.domain, source.member, cap, source.keep, source.note)
        )
    if not chosen:
        raise ValueError("every source was excluded")
    return tuple(chosen)


def build(
    bundle: Path,
    math_path: Optional[Path],
    output: Path,
    seed: int = 62,
    exclude: Sequence[str] = (),
    scale: float = 1.0,
    profile: str = "default",
) -> Dict[str, Any]:
    rows, collected = collect(bundle, math_path, select_sources(exclude, scale, profile), seed)
    if not rows:
        raise ValueError("blend is empty; no source produced usable rows")

    diversity = measure(rows)
    counts = Counter(row["domain"] for row in rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "schema": RECEIPT_SCHEMA,
        "output": str(output),
        "rows": len(rows),
        "domains": dict(sorted(counts.items())),
        "diversity": diversity,
        "collection": collected,
        "seed": seed,
        "excluded_domains": sorted(exclude),
        "cap_scale": scale,
        "profile": profile,
        "non_claims": [
            "Word types measure diversity, not quality. None of these corpora were "
            "audited for correctness, licensing or contamination.",
            "The domain labels come from each source file's own category/task/topic "
            "field. They describe where a row came from, not a verified competence.",
            "Balancing by cap makes domains comparable in size, which is what makes "
            "per-domain evaluation meaningful. It does not make them equally hard.",
        ],
    }


def print_summary(receipt: Dict[str, Any]) -> None:
    print(f"rows        {receipt['rows']:,}")
    print(f"word types  {receipt['diversity']['word_types']:,}")
    print()
    print(f"{'domain':16s} {'rows':>8s} {'word types':>11s}")
    print("-" * 38)
    by_domain = receipt["diversity"]["word_types_by_domain"]
    for domain, count in sorted(receipt["domains"].items(), key=lambda kv: -kv[1]):
        print(f"{domain:16s} {count:8,d} {by_domain.get(domain, 0):11,d}")
    if receipt["collection"]["skipped"]:
        print("\nskipped:")
        for entry in receipt["collection"]["skipped"]:
            print(f"  {entry}")


#: v64: a corpus weighted toward text a human wrote.
#:
#: The v62/v63 blends are dominated by generator output, and it shows: 289,169
#: rows collapse to 399,029 distinct 8-grams, and one sentence appears in 17.6%
#: of them. A model fitted to that reproduces it, which the recall meter now
#: makes visible per reply.
#:
#: This profile inverts the ratio. `book_extracts_public_domain` (22,962 word
#: types) and `bible_kjv` (10,248) are real prose; `dictionary_wordnet_meanings`
#: is 5,000 explicit word-to-definition pairs, which is the only material on disk
#: that teaches what words *mean* rather than how replies are shaped. Templated
#: dialogue is kept, but as a minority, so the model still answers in turns.
#:
#: v62 measured these same domains at perplexity 12-19 and concluded they were
#: unlearnable. That measurement was taken with the packer that orphaned 56% of
#: supervised tokens from their prompt, so it is worth re-running rather than
#: treating as settled.
#: Domains v62 measured as unlearnable at this scale.
#:
#: Held-out perplexity after 8,000 steps: writing 18.65, vocabulary 19.09,
#: scripture 12.67, against 1.26-1.39 for the templated domains. Those three
#: carry the real human text, and including them cost capacity without producing
#: competence -- the model defaulted to the templates it *could* fit. Excluding
#: them trades breadth for a corpus consistent enough to learn.
UNLEARNABLE_AT_THIS_SCALE = ("writing", "vocabulary", "scripture", "science")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--math", default=str(DEFAULT_MATH))
    parser.add_argument("--output", default=str(REPO_ROOT / "datasets" / "v62" / "v62_blend.jsonl"))
    parser.add_argument("--receipt", default=None)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="drop a domain entirely; repeatable",
    )
    parser.add_argument(
        "--coherent",
        action="store_true",
        help=(
            "keep only the domains v62 measured as learnable at this scale, "
            f"excluding {', '.join(UNLEARNABLE_AT_THIS_SCALE)}"
        ),
    )
    parser.add_argument(
        "--profile",
        choices=("default", "meaning"),
        default="default",
        help="'meaning' weights real prose and dictionary definitions over generated dialogue",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every per-domain cap, to use more of the available data",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exclude = list(args.exclude)
    if args.coherent:
        exclude.extend(UNLEARNABLE_AT_THIS_SCALE)
    receipt = build(
        Path(args.bundle),
        Path(args.math),
        Path(args.output),
        seed=args.seed,
        exclude=sorted(set(exclude)),
        scale=args.scale,
        profile=args.profile,
    )
    print_summary(receipt)

    destination = Path(args.receipt) if args.receipt else Path(args.output).with_suffix(".receipt.json")
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nreceipt -> {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
