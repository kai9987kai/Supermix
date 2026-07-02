#!/usr/bin/env python3
"""
Build a public-domain Bible conversation dataset (KJV via Project Gutenberg).

Design goals:
- Whole-Bible coverage: every parsed verse is included at least once.
- Grounded rows: exact verse, reference lookup, local passage context/explanation.
- Low-risk parsing: line-wrapped KJV verses are reconstructed from pg10.txt format.

Usage:
  python build_bible_dataset.py --output conversation_data.bible_kjv_public_domain_v1_500k.jsonl --target 500000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


SOURCE_URL = "https://www.gutenberg.org/cache/epub/10/pg10.txt"
START_MARKER = "*** START OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
END_MARKER = "*** END OF THE PROJECT GUTENBERG EBOOK THE KING JAMES VERSION OF THE BIBLE ***"
VERSE_RE = re.compile(r"^(\d+):(\d+)\s+(.*)$")
# Include marker-only lines like "1:15" where the verse text starts on the next line.
VERSE_INLINE_RE = re.compile(r"(\d+):(\d+)(?=\s|$)")


VerseRow = Dict[str, Any]


def _download_if_needed(url: str, cache_path: Path) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    return cache_path


def _extract_gutenberg_body(raw_text: str) -> str:
    start = raw_text.find(START_MARKER)
    if start < 0:
        raise RuntimeError("Could not find Project Gutenberg start marker.")
    start += len(START_MARKER)
    end = raw_text.find(END_MARKER, start)
    if end < 0:
        end = len(raw_text)
    return raw_text[start:end]


def _looks_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if VERSE_RE.match(s):
        return False
    if s.startswith("***"):
        return False
    if re.search(r"\d:\d", s):
        return False
    # Verse continuations frequently end with punctuation and are much longer.
    if len(s) < 4:
        return False
    letters = sum(ch.isalpha() for ch in s)
    if letters < max(3, int(0.5 * len(s.replace(" ", "")))):
        return False
    return True


def _is_book_heading(line: str) -> bool:
    s = (line or "").strip()
    if not _looks_heading(s):
        return False
    sl = s.lower()
    if sl.startswith("the old testament") or sl.startswith("the new testament"):
        return False
    if sl.startswith("the project gutenberg") or sl.startswith("title:") or sl.startswith("release date:"):
        return False
    # Book titles are typically title-like and short-ish.
    return True


def _extract_known_book_headings(text: str) -> List[str]:
    """Use the Gutenberg table-of-contents block to get exact book heading strings."""
    lines = [ln.strip() for ln in text.splitlines()]
    headings: List[str] = []
    seen_first_ot = False
    in_toc = False
    for s in lines[:1200]:
        if not s:
            continue
        if s == "The Old Testament of the King James Version of the Bible":
            if seen_first_ot:
                break  # second occurrence marks the start of the body
            seen_first_ot = True
            in_toc = True
            continue
        if s == "The New Testament of the King James Bible":
            in_toc = True
            continue
        if not in_toc:
            continue
        if VERSE_RE.match(s):
            break
        if _is_book_heading(s):
            headings.append(s)
    # Preserve order, dedupe exact strings.
    seen = set()
    out: List[str] = []
    for h in headings:
        if h not in seen:
            out.append(h)
            seen.add(h)
    return out


def parse_kjv_verses(text: str) -> List[VerseRow]:
    verses: List[VerseRow] = []
    known_book_headings = set(_extract_known_book_headings(text))
    current: Optional[VerseRow] = None
    current_book = "Unknown"
    pending_headings: List[str] = []
    last_blank = True

    def finalize_current() -> None:
        nonlocal current
        if current is None:
            return
        current["text"] = re.sub(r"\s+", " ", str(current["text"]).strip())
        if current["text"]:
            verses.append(current)
        current = None

    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            last_blank = True
            continue

        markers = list(VERSE_INLINE_RE.finditer(stripped))
        if markers:
            prefix = stripped[: markers[0].start()].strip()
            if prefix and current is not None:
                current["text"] = f"{current['text']} {prefix}".strip()
            for i, mk in enumerate(markers):
                finalize_current()
                if pending_headings:
                    for h in reversed(pending_headings):
                        if h in known_book_headings or _is_book_heading(h):
                            current_book = h.strip()
                            break
                    pending_headings.clear()
                seg_start = mk.end()
                seg_end = markers[i + 1].start() if i + 1 < len(markers) else len(stripped)
                seg_text = stripped[seg_start:seg_end].strip()
                current = {
                    "book": current_book,
                    "chapter": int(mk.group(1)),
                    "verse": int(mk.group(2)),
                    "text": seg_text,
                }
            last_blank = False
            continue

        if current is not None:
            # At a book boundary, there is usually a blank line followed by a heading.
            if last_blank and (stripped in known_book_headings):
                finalize_current()
                pending_headings = [stripped]
            else:
                current["text"] = f"{current['text']} {stripped}".strip()
            last_blank = False
            continue

        if stripped in known_book_headings:
            pending_headings.append(stripped)
            if len(pending_headings) > 6:
                pending_headings = pending_headings[-6:]
        elif _looks_heading(stripped):
            pending_headings.append(stripped)
            if len(pending_headings) > 6:
                pending_headings = pending_headings[-6:]
        last_blank = False

    finalize_current()
    # Drop any malformed trailing parse entries if present.
    return [v for v in verses if v.get("book") and isinstance(v.get("chapter"), int) and isinstance(v.get("verse"), int)]


def _ref(v: VerseRow) -> str:
    return f"{v['book']} {v['chapter']}:{v['verse']}"


def _chapter_groups(verses: Sequence[VerseRow]) -> Dict[Tuple[str, int], List[VerseRow]]:
    grouped: Dict[Tuple[str, int], List[VerseRow]] = defaultdict(list)
    for v in verses:
        grouped[(str(v["book"]), int(v["chapter"]))].append(v)
    return grouped


THEME_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("creation", ("create", "created", "made", "heaven", "earth", "light")),
    ("faith", ("faith", "believe", "believed", "trust")),
    ("love", ("love", "charity", "beloved")),
    ("wisdom", ("wisdom", "understanding", "knowledge", "wise")),
    ("justice", ("judge", "judgment", "justice", "righteous", "oppress")),
    ("mercy", ("mercy", "forgive", "compassion", "grace")),
    ("prayer", ("pray", "prayer", "supplication", "cry", "called unto")),
    ("salvation", ("save", "salvation", "redeem", "deliver")),
    ("hope", ("hope", "promise", "wait", "comfort")),
    ("warning", ("woe", "judgment", "wrath", "destroy", "repent")),
]


def _themes_for_text(text: str) -> List[str]:
    t = (text or "").lower()
    hits: List[str] = []
    for name, kws in THEME_KEYWORDS:
        if any(kw in t for kw in kws):
            hits.append(name)
    if not hits:
        if re.search(r"\b(lord|god|jesus|christ)\b", t):
            hits.append("teaching")
        else:
            hits.append("scripture")
    return hits[:3]


def _write_jsonl(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_rows(verses: List[VerseRow], target: int, rng: random.Random) -> List[Dict[str, Any]]:
    if target < len(verses):
        raise ValueError(f"target={target} is smaller than verse coverage count={len(verses)}")

    rows: List[Dict[str, Any]] = []
    by_ref: Dict[Tuple[str, int, int], VerseRow] = {(v["book"], v["chapter"], v["verse"]): v for v in verses}
    chapter_map = _chapter_groups(verses)
    chapter_keys = list(chapter_map.keys())

    # Coverage pass: include every verse exactly at least once.
    for i, v in enumerate(verses):
        ref = _ref(v)
        user = [
            f"What does {ref} say in the KJV Bible?",
            f"Quote {ref} (KJV).",
            f"Give me the text of {ref} from the King James Bible.",
        ][i % 3]
        rows.append(
            {
                "user": user,
                "assistant": f"{ref} (KJV): {v['text']}",
                "category": "bible_quote",
                "source": "Project Gutenberg KJV (eBook #10)",
                "reference": ref,
            }
        )

    def add_explanation_row(v: VerseRow) -> None:
        ref = _ref(v)
        themes = _themes_for_text(str(v["text"]))
        user = rng.choice(
            [
                f"Explain the meaning of {ref} in simple English.",
                f"What is {ref} saying?",
                f"Help me understand {ref} (KJV).",
                f"Summarize the message of {ref}.",
            ]
        )
        theme_text = ", ".join(themes)
        assistant = (
            f"{ref} says: {v['text']} "
            f"In simple terms, this verse emphasizes {theme_text}. "
            f"The explanation here is a text-based summary, not a doctrinal interpretation."
        )
        rows.append(
            {
                "user": user,
                "assistant": assistant,
                "category": "bible_explain",
                "source": "Project Gutenberg KJV (eBook #10)",
                "reference": ref,
            }
        )

    def add_next_verse_row(v: VerseRow) -> bool:
        nxt = by_ref.get((v["book"], v["chapter"], int(v["verse"]) + 1))
        if not nxt:
            return False
        ref = _ref(v)
        nref = _ref(nxt)
        rows.append(
            {
                "user": rng.choice(
                    [
                        f"What verse comes immediately after {ref}?",
                        f"Give me the next verse after {ref} (KJV).",
                        f"Continue after {ref}.",
                    ]
                ),
                "assistant": f"The next verse is {nref} (KJV): {nxt['text']}",
                "category": "bible_next_verse",
                "source": "Project Gutenberg KJV (eBook #10)",
                "reference": nref,
            }
        )
        return True

    def add_passage_row() -> bool:
        if not chapter_keys:
            return False
        book, chap = rng.choice(chapter_keys)
        chapter_verses = chapter_map[(book, chap)]
        if len(chapter_verses) < 2:
            return False
        win = min(rng.choice([2, 3, 4, 5]), len(chapter_verses))
        start_idx = rng.randint(0, len(chapter_verses) - win)
        passage = chapter_verses[start_idx : start_idx + win]
        start_ref = _ref(passage[0])
        end_ref = _ref(passage[-1])
        joined = " ".join(str(x["text"]) for x in passage)
        themes = _themes_for_text(joined)
        assistant = (
            f"{start_ref}-{end_ref} (KJV): {joined} "
            f"Key themes in this passage include {', '.join(themes)}. "
            f"This summary is based on the wording of the passage."
        )
        rows.append(
            {
                "user": rng.choice(
                    [
                        f"Summarize {start_ref}-{end_ref} in plain English.",
                        f"What is happening in {start_ref}-{end_ref}?",
                        f"Give me a short explanation of the passage {start_ref}-{end_ref}.",
                    ]
                ),
                "assistant": assistant,
                "category": "bible_passage",
                "source": "Project Gutenberg KJV (eBook #10)",
                "reference": f"{start_ref}-{end_ref}",
            }
        )
        return True

    # Augment to target using grounded row types.
    verse_index = 0
    while len(rows) < target:
        roll = rng.random()
        if roll < 0.45:
            add_explanation_row(verses[verse_index % len(verses)])
            verse_index += 1
        elif roll < 0.75:
            ok = add_next_verse_row(verses[verse_index % len(verses)])
            verse_index += 1
            if not ok:
                add_explanation_row(verses[(verse_index - 1) % len(verses)])
        else:
            if not add_passage_row():
                add_explanation_row(verses[verse_index % len(verses)])
                verse_index += 1

    return rows[:target]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Bible-based KJV conversation dataset (public domain).")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--target", type=int, default=500000, help="Target row count (must be >= verse count)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--cache", default="corpora/kjv_pg10.txt", help="Local cache path for pg10 text")
    ap.add_argument("--source_url", default=SOURCE_URL, help="Source URL (default: Project Gutenberg KJV)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache_path = Path(args.cache)
    out_path = Path(args.output)

    _download_if_needed(args.source_url, cache_path)
    raw = cache_path.read_text(encoding="utf-8-sig", errors="replace")
    body = _extract_gutenberg_body(raw)
    verses = parse_kjv_verses(body)
    if len(verses) < 30000:
        raise RuntimeError(f"Parsed too few verses: {len(verses)} (expected around 31k)")

    rows = build_rows(verses, target=args.target, rng=rng)
    _write_jsonl(rows, out_path)

    print(f"Parsed verses: {len(verses)}")
    print(f"Wrote rows: {len(rows)} -> {out_path}")
    print(f"Output bytes: {out_path.stat().st_size}")
    print(f"Source cache: {cache_path} ({cache_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
