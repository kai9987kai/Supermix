"""Tell a generated reply apart from a remembered one.

The v63 investigation ended on an uncomfortable fact: this model's most fluent
output was not composition. *"The moment hung in the air like a held breath"*
reads like writing, and it appears verbatim in **51,022 of 289,169** training
rows. A chat interface that prints that sentence with no further comment is
presenting recall as though it were generation, which is the single most
misleading thing it could do.

This module makes the distinction measurable at serving time. It indexes every
word n-gram in a corpus, then scores a reply by how much of it the corpus already
contained.

Two numbers, because they answer different questions:

* ``verbatim_rate`` -- the fraction of the reply's n-gram windows found in the
  corpus. A reply built entirely from remembered spans scores 1.0.
* ``longest_verbatim_words`` -- the longest unbroken stretch that appears in
  training. A high rate made of scattered common phrases ("I can help you with")
  is ordinary language use; a single 40-word run is recitation, and the two look
  identical under an average.

The index stores 64-bit hashes in a sorted array and answers with binary search:
~92 MB and microseconds per lookup for a 12M-window corpus, against several
hundred megabytes for a Python set of tuples. Collisions are possible at 2**64
and would slightly *overstate* recall; nothing here rounds in the flattering
direction, which is deliberate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Window size in words. Eight is long enough that hitting one by chance is
#: unlikely in this vocabulary, and short enough to catch a recycled clause.
DEFAULT_N = 8

#: Below this the reply is too short for an n-gram window to exist at all.
MIN_WORDS = DEFAULT_N

_WORD = re.compile(r"[a-z0-9']+")


def normalise(text: str) -> List[str]:
    """Lowercase word tokens, so recall is not hidden by punctuation or case."""

    return _WORD.findall(text.lower())


def _hash_window(words: Sequence[str]) -> int:
    joined = " ".join(words).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


def _windows(words: Sequence[str], n: int) -> Iterable[Tuple[int, int]]:
    """Yield ``(start, hash)`` for each n-word window."""

    for start in range(0, len(words) - n + 1):
        yield start, _hash_window(words[start : start + n])


@dataclass
class RecallReport:
    """How much of one reply the corpus already contained."""

    windows: int
    matched: int
    verbatim_rate: float
    longest_verbatim_words: int
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "windows": self.windows,
            "matched": self.matched,
            "verbatim_rate": round(self.verbatim_rate, 4),
            "longest_verbatim_words": self.longest_verbatim_words,
            "verdict": self.verdict,
        }


class RecallIndex:
    """Every n-gram of a corpus, searchable."""

    def __init__(self, hashes: np.ndarray, n: int, rows: int, source: str = ""):
        self.hashes = hashes
        self.n = n
        self.rows = rows
        self.source = source

    # -- construction ------------------------------------------------------

    @classmethod
    def from_texts(
        cls, texts: Iterable[str], n: int = DEFAULT_N, source: str = ""
    ) -> "RecallIndex":
        collected: List[int] = []
        rows = 0
        for text in texts:
            rows += 1
            words = normalise(text)
            collected.extend(value for _, value in _windows(words, n))
        if not collected:
            # An empty index must still answer, and must answer "nothing
            # matched" rather than crash or claim everything matched.
            return cls(np.zeros(0, dtype=np.uint64), n, rows, source)
        array = np.array(collected, dtype=np.uint64)
        array.sort()
        return cls(np.unique(array), n, rows, source)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        n: int = DEFAULT_N,
        field: str = "assistant",
        limit: Optional[int] = None,
    ) -> "RecallIndex":
        """Index the replies of a JSONL corpus.

        Only the assistant side is indexed: the question is what the model is
        reproducing when it *answers*, and including prompts would score a reply
        as recalled for echoing the user.
        """

        def rows() -> Iterable[str]:
            with Path(path).open(encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    if limit is not None and index >= limit:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and record.get(field):
                        yield str(record[field])

        return cls.from_texts(rows(), n=n, source=str(path))

    # -- query -------------------------------------------------------------

    def contains(self, value: int) -> bool:
        if self.hashes.size == 0:
            return False
        position = int(np.searchsorted(self.hashes, np.uint64(value)))
        return position < self.hashes.size and int(self.hashes[position]) == value

    def score(self, reply: str) -> RecallReport:
        """Measure how much of ``reply`` the corpus already contained."""

        words = normalise(reply)
        if len(words) < self.n:
            # Too short to judge. Saying "composed" here would let a one-word
            # answer masquerade as originality.
            return RecallReport(0, 0, 0.0, 0, "too_short_to_judge")

        flags = [self.contains(value) for _, value in _windows(words, self.n)]
        matched = sum(flags)
        total = len(flags)

        longest_run = 0
        current = 0
        for hit in flags:
            current = current + 1 if hit else 0
            longest_run = max(longest_run, current)
        # A run of k consecutive matching windows covers k + n - 1 words.
        longest_words = longest_run + self.n - 1 if longest_run else 0

        rate = matched / total
        if rate >= 0.6 or longest_words >= 25:
            verdict = "largely_recalled"
        elif rate >= 0.2 or longest_words >= self.n:
            verdict = "part_recalled"
        else:
            verdict = "mostly_novel"
        return RecallReport(total, matched, rate, longest_words, verdict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "rows_indexed": self.rows,
            "windows_indexed": int(self.hashes.size),
            "n": self.n,
        }


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", required=True, help="JSONL corpus to index")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--text", action="append", default=[], help="score this text; repeatable")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    index = RecallIndex.from_jsonl(args.corpus, n=args.n, limit=args.limit)
    print(json.dumps(index.to_dict(), indent=2))
    for text in args.text:
        report = index.score(text)
        print()
        print(f"  {text[:90]!r}")
        print(f"  -> {json.dumps(report.to_dict())}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
