import argparse
import json
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_SOURCES: List[Tuple[str, str]] = [
    ("alice_in_wonderland", "https://www.gutenberg.org/cache/epub/11/pg11.txt"),
    ("pride_and_prejudice", "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"),
    ("frankenstein", "https://www.gutenberg.org/cache/epub/84/pg84.txt"),
    ("sherlock_holmes", "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"),
    ("dracula", "https://www.gutenberg.org/cache/epub/345/pg345.txt"),
    ("moby_dick", "https://www.gutenberg.org/cache/epub/2701/pg2701.txt"),
    ("great_expectations", "https://www.gutenberg.org/cache/epub/1400/pg1400.txt"),
    ("jane_eyre", "https://www.gutenberg.org/cache/epub/1260/pg1260.txt"),
]

SENTENCE_RE = re.compile(r"[^.!?]+[.!?]?")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
CAP_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
SPACE_RE = re.compile(r"\s+")
GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG", re.I)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG", re.I)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "at", "for", "from",
    "with", "without", "by", "as", "is", "are", "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "he", "she", "they", "them", "his", "her", "their", "you", "your", "i", "we", "our", "us",
    "not", "no", "so", "very", "into", "out", "up", "down", "over", "under", "again", "there", "here", "when",
    "where", "why", "how", "what", "who", "whom", "which", "had", "has", "have", "do", "does", "did", "can", "could",
    "would", "should", "may", "might", "must", "shall", "will", "just", "only", "also", "such", "much", "many",
}


def _normalize_spaces(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "").replace("\r", " ").replace("\t", " ")).strip()


def _mojibake_score(text: str) -> int:
    score = 0
    for tok in ("â€™", "â€œ", "â€", "Ã", "Â"):
        score += text.count(tok)
    # Broader signals for cp1252/utf-8 mojibake sequences.
    score += max(0, text.count("â") - 2)
    score += max(0, text.count("Ã") - 1)
    return score


def _maybe_repair_mojibake(text: str) -> str:
    candidates = [text]
    for enc in ("cp1252", "latin-1"):
        try:
            repaired = text.encode(enc, errors="ignore").decode("utf-8", errors="ignore")
            if repaired:
                candidates.append(repaired)
        except Exception:
            continue
    return min(candidates, key=lambda t: (_mojibake_score(t), -len(t)))


def _decode_payload(payload: bytes) -> str:
    candidates: List[str] = []
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            candidates.append(payload.decode(enc, errors="strict"))
        except Exception:
            continue
    if not candidates:
        return payload.decode("utf-8", errors="ignore")
    # Prefer lowest mojibake score; tie-break on longer decoded text.
    best = min(candidates, key=lambda t: (_mojibake_score(t), -len(t)))
    return best


def _looks_like_book_text(text: str) -> bool:
    t = (text or "").lower()
    if len(t) < 20000:
        return False
    if "<html" in t and "</html>" in t:
        return False
    return True


def _download_text(url: str, cache_path: Path, timeout: int = 30) -> str:
    cached = ""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        cached = cache_path.read_text(encoding="utf-8", errors="ignore")
        cached = _maybe_repair_mojibake(cached)
        if _looks_like_book_text(cached) and _mojibake_score(cached) <= 5:
            return cached

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Codex dataset builder)"})
    last_err = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            text = _decode_payload(payload)
            text = _maybe_repair_mojibake(text)
            if not _looks_like_book_text(text):
                if cached:
                    return cached
                raise RuntimeError(f"Downloaded non-book payload from {url}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        except (urllib.error.URLError, TimeoutError) as exc:
            last_err = exc
            time.sleep(1.0)
        except Exception as exc:
            last_err = exc
            time.sleep(1.0)
    if cached:
        return cached
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def _strip_gutenberg_boilerplate(text: str) -> str:
    lines = text.splitlines()
    start_idx = 0
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if GUTENBERG_START_RE.search(line):
            start_idx = i + 1
            break
    for i in range(len(lines) - 1, -1, -1):
        if GUTENBERG_END_RE.search(lines[i]):
            end_idx = i
            break
    core = "\n".join(lines[start_idx:end_idx]).strip()
    return core if core else text


def _is_headingish(paragraph: str) -> bool:
    p = paragraph.strip()
    if not p:
        return True
    if len(p) < 10:
        return True
    if len(p) > 2200:
        return True
    if re.fullmatch(r"[IVXLCDM0-9 .,-]+", p):
        return True
    if re.search(r"\bchapter\b", p, re.I) and len(p.split()) <= 8:
        return True
    alpha = [ch for ch in p if ch.isalpha()]
    if alpha:
        upper_ratio = sum(1 for ch in alpha if ch.isupper()) / max(1, len(alpha))
        if upper_ratio > 0.75 and len(p.split()) <= 16:
            return True
    return False


def _paragraphs_from_text(text: str) -> List[str]:
    raw_paras = re.split(r"\n\s*\n+", text)
    out: List[str] = []
    for para in raw_paras:
        p = _normalize_spaces(para)
        if _is_headingish(p):
            continue
        words = WORD_RE.findall(p)
        if len(words) < 20:
            continue
        out.append(p)
    if len(out) >= 20:
        return out

    # Fallback for wrapped/plaintext formats that separate nearly every line.
    lines = [_normalize_spaces(ln) for ln in text.splitlines()]
    usable_lines: List[str] = []
    for ln in lines:
        if _is_headingish(ln):
            continue
        if len(WORD_RE.findall(ln)) < 5:
            continue
        usable_lines.append(ln)

    if len(usable_lines) < 6:
        return out

    windows: List[str] = []
    for i in range(0, len(usable_lines) - 2):
        joined = " ".join(usable_lines[i : i + 4])
        joined = _normalize_spaces(joined)
        words = WORD_RE.findall(joined)
        if 24 <= len(words) <= 160:
            windows.append(joined)
        joined2 = " ".join(usable_lines[i : i + 6])
        joined2 = _normalize_spaces(joined2)
        words2 = WORD_RE.findall(joined2)
        if 30 <= len(words2) <= 220:
            windows.append(joined2)
    if windows:
        out.extend(windows)
    return out


def _sentences(text: str, max_sentences: int = 6) -> List[str]:
    out: List[str] = []
    for s in SENTENCE_RE.findall(text):
        s = _normalize_spaces(s)
        if len(s) >= 12:
            out.append(s)
        if len(out) >= max_sentences:
            break
    if not out:
        out.append(_normalize_spaces(text))
    return out


def _top_keywords(text: str, k: int = 4) -> List[str]:
    counts: Dict[str, int] = {}
    for w in WORD_RE.findall(text.lower()):
        if len(w) < 4 or w in STOPWORDS:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


def _named_entities(text: str, k: int = 4) -> List[str]:
    counts: Dict[str, int] = {}
    for n in CAP_NAME_RE.findall(text):
        if n.lower() in STOPWORDS:
            continue
        counts[n] = counts.get(n, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [n for n, _ in ranked[:k]]


def _guess_tone(text: str) -> List[str]:
    lower = text.lower()
    tone: List[str] = []
    if re.search(r"\b(dark|death|blood|fear|storm|grave|terror|cold)\b", lower):
        tone.append("tense")
    if re.search(r"\b(laugh|smile|curious|wonder|playful|peculiar)\b", lower):
        tone.append("playful")
    if re.search(r"\b(softly|quiet|silence|gentle|calm)\b", lower):
        tone.append("reflective")
    if re.search(r"\b(angry|rage|shout|cried|furious)\b", lower):
        tone.append("heated")
    if re.search(r"\b(love|heart|dear|affection)\b", lower):
        tone.append("emotional")
    if not tone:
        tone = ["descriptive", "narrative"]
    return tone[:3]


def _clip_excerpt_words(text: str, min_words: int, max_words: int) -> str:
    words = text.split()
    if len(words) < min_words:
        return ""
    if len(words) <= max_words:
        return " ".join(words)
    start = 0
    end = min(len(words), max_words)
    clipped = " ".join(words[start:end]).strip()
    if not re.search(r"[.!?]$", clipped):
        clipped += " ..."
    return clipped


def _build_excerpt_pool(paragraphs: Sequence[str], rng: random.Random, min_words: int, max_words: int) -> List[str]:
    pool: List[str] = []
    if not paragraphs:
        return pool
    for i, para in enumerate(paragraphs):
        cand = _clip_excerpt_words(para, min_words=min_words, max_words=max_words)
        if cand:
            pool.append(cand)
        if i + 1 < len(paragraphs):
            pair = para + " " + paragraphs[i + 1]
            cand2 = _clip_excerpt_words(pair, min_words=min_words, max_words=max_words)
            if cand2:
                pool.append(cand2)
    rng.shuffle(pool)
    dedup: List[str] = []
    seen = set()
    for p in pool:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def _summary_answer(excerpt: str) -> str:
    sents = _sentences(excerpt, max_sentences=4)
    lead = sents[0]
    if len(sents) > 1:
        return f"Summary: {lead} The passage mainly develops {', '.join(_top_keywords(excerpt, k=3)) or 'a scene and its emotional tension'}."
    return f"Summary: {lead}"


def _tone_answer(excerpt: str) -> str:
    tones = _guess_tone(excerpt)
    reasons = _top_keywords(excerpt, k=4)
    reason_text = ", ".join(reasons) if reasons else "the descriptive wording and pacing"
    return f"Tone: {', '.join(tones)}. This comes through in the language choices around {reason_text}."


def _theme_answer(excerpt: str) -> str:
    kws = _top_keywords(excerpt, k=5)
    themes = []
    if any(w in kws for w in ("duty", "honor", "promise", "obedience")):
        themes.append("duty and obligation")
    if any(w in kws for w in ("fear", "danger", "death", "night")):
        themes.append("fear and uncertainty")
    if any(w in kws for w in ("love", "heart", "dear", "friend")):
        themes.append("attachment and loyalty")
    if any(w in kws for w in ("house", "room", "window", "street")):
        themes.append("setting shaping emotion")
    if not themes:
        themes = ["character perspective", "social tension", "scene-building detail"]
    return f"Likely themes: {', '.join(themes[:3])}. Key signals: {', '.join(kws[:4]) or 'the narrative focus and repeated details'}."


def _character_answer(excerpt: str) -> str:
    names = _named_entities(excerpt, k=4)
    if names:
        return (
            f"Character focus: {', '.join(names)}. The excerpt suggests relationships and motives through "
            f"their reactions, word choice, and the scene details around them."
        )
    return "Character focus: the narrator and nearby figures. The passage reveals character mainly through reactions, pacing, and description."


def _modern_answer(excerpt: str) -> str:
    sents = _sentences(excerpt, max_sentences=3)
    plain = " ".join(sents)
    plain = re.sub(r"\bshall\b", "will", plain, flags=re.I)
    plain = re.sub(r"\bthus\b", "so", plain, flags=re.I)
    plain = re.sub(r"\bwhilst\b", "while", plain, flags=re.I)
    plain = _normalize_spaces(plain)
    return f"In modern wording: {plain}"


def _continuation_answer(excerpt: str, rng: random.Random) -> str:
    names = _named_entities(excerpt, k=2)
    kws = _top_keywords(excerpt, k=4)
    subject = ", ".join(names) if names else "the narrator"
    pivot = rng.choice(
        [
            "a small detail suddenly changes the meaning of the scene",
            "a quiet observation raises the tension",
            "an interruption forces a decision",
            "a remembered thought reframes what just happened",
        ]
    )
    cue = ", ".join(kws[:3]) if kws else "the mood and setting"
    return (
        f"Creative continuation: {subject} pauses, and {pivot}. "
        f"The next lines should preserve the original cadence while extending {cue}."
    )


def _dialogue_answer(excerpt: str) -> str:
    sents = _sentences(excerpt, max_sentences=2)
    line_a = sents[0] if sents else "The scene is tense."
    line_b = sents[1] if len(sents) > 1 else "Someone responds carefully."
    return (
        "Dialogue-style rewrite: "
        f"\"{line_a}\" "
        f"\"{line_b}\" "
        "Keep the emotional intent, but make the exchange more explicit."
    )


TASKS = [
    ("summary", "Summarize this book excerpt in 2-3 sentences.", _summary_answer),
    ("tone", "What tone and mood do you notice in this excerpt? Be specific.", _tone_answer),
    ("themes", "Identify likely themes in this excerpt and explain the clues.", _theme_answer),
    ("character", "What does this excerpt suggest about the character(s)?", _character_answer),
    ("modernize", "Rewrite this excerpt in plain modern English while keeping the meaning.", _modern_answer),
    ("continue", "Continue this excerpt in a similar style for a few lines (high level, not full prose).", _continuation_answer),
    ("dialogue", "Turn the key idea of this excerpt into a short dialogue-style version.", _dialogue_answer),
]


def _make_row(book_name: str, excerpt: str, rng: random.Random) -> Dict[str, str]:
    task_name, prompt, fn = rng.choice(TASKS)
    preface = rng.choice(
        [
            "Please analyze this excerpt from a book.",
            "Help me understand this book passage.",
            "Can you work with this literary excerpt?",
            "I am studying this novel excerpt.",
            "Review this passage for me.",
        ]
    )
    if task_name == "continue":
        preface = rng.choice(
            [
                "Be creative but stay consistent with the style of this excerpt.",
                "Write a careful stylistic continuation based on this passage.",
                "Please continue the scene while preserving tone and voice.",
            ]
        )
    user = f"{preface}\n\nExcerpt ({book_name}):\n{excerpt}\n\n{prompt}"
    if task_name == "continue":
        assistant = fn(excerpt, rng)
    else:
        assistant = fn(excerpt)
    assistant = _normalize_spaces(assistant)
    if len(user) > 900:
        user = user[:900].rstrip() + " ..."
    if len(assistant) > 520:
        assistant = assistant[:520].rstrip() + " ..."
    return {
        "user": user,
        "assistant": assistant,
        "topic": "book_extracts_public_domain",
        "source_book": book_name,
        "task": task_name,
    }


def _fallback_excerpt_pool() -> Dict[str, List[str]]:
    # Fallback if downloads fail; keeps the pipeline usable.
    passages = [
        "The corridor was quiet except for the distant ticking of a clock, and every small sound seemed larger because no one spoke.",
        "She stood by the window and watched the rain collect on the stone, thinking not only of what had happened, but of what must be said next.",
        "He laughed at first, lightly and almost without care, but the letter in his hand altered his expression before the fire had burned half its edge.",
        "The road bent through the mist and the horse slowed of its own accord, as if it recognized a place the rider wished not to remember.",
        "At the table they spoke politely, yet each answer arrived a moment late, and the silence between them carried more meaning than the words.",
    ]
    return {"fallback_public_domain_style": passages}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a book-extract conversation dataset from public-domain texts.")
    ap.add_argument("--output", default="conversation_data.book_extracts_public_domain.jsonl")
    ap.add_argument("--target", type=int, default=1500000, help="Number of rows to generate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache_dir", default="book_sources_cache", help="Where downloaded book texts are cached.")
    ap.add_argument("--min_excerpt_words", type=int, default=38)
    ap.add_argument("--max_excerpt_words", type=int, default=88)
    ap.add_argument("--max_excerpts_per_book", type=int, default=6000)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    cache_dir = Path(args.cache_dir)
    book_to_excerpts: Dict[str, List[str]] = {}
    failed_sources: List[str] = []

    for book_name, url in DEFAULT_SOURCES:
        cache_path = cache_dir / f"{book_name}.txt"
        try:
            raw = _download_text(url, cache_path)
            core = _strip_gutenberg_boilerplate(raw)
            paras = _paragraphs_from_text(core)
            excerpts = _build_excerpt_pool(
                paras,
                rng=rng,
                min_words=max(10, int(args.min_excerpt_words)),
                max_words=max(int(args.min_excerpt_words) + 5, int(args.max_excerpt_words)),
            )
            if args.max_excerpts_per_book > 0:
                excerpts = excerpts[: int(args.max_excerpts_per_book)]
            if excerpts:
                book_to_excerpts[book_name] = excerpts
                print(f"Loaded {len(excerpts)} excerpts from {book_name}")
            else:
                failed_sources.append(book_name)
        except Exception as exc:
            failed_sources.append(f"{book_name}: {exc}")

    if not book_to_excerpts:
        print("All downloads failed; using fallback excerpt pool.")
        book_to_excerpts = _fallback_excerpt_pool()

    books = sorted(book_to_excerpts.keys())
    excerpt_counts = {name: len(book_to_excerpts[name]) for name in books}
    total_excerpt_pool = sum(excerpt_counts.values())
    if total_excerpt_pool <= 0:
        raise RuntimeError("No excerpts available to build dataset.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = max(1, int(args.target))

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(target):
            book = rng.choice(books)
            excerpt = rng.choice(book_to_excerpts[book])
            row = _make_row(book, excerpt, rng)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 250000 == 0:
                print(f"Wrote {i + 1}/{target}")

    print(f"Books used: {len(books)}")
    print(f"Excerpt pool: {total_excerpt_pool}")
    print(f"Output rows: {target}")
    print(f"Wrote: {out_path}")
    if failed_sources:
        print("Failed sources:")
        for item in failed_sources:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
