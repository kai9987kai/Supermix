import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _safe_word(text: str) -> str:
    return _clean(text.replace("_", " "))


def _ensure_wordnet(download_dir: Optional[str] = None):
    import nltk
    from nltk.corpus import wordnet as wn

    try:
        _ = wn.synsets("test")
        return wn
    except LookupError:
        dl_kwargs: Dict[str, Any] = {"quiet": True}
        if download_dir:
            dl_kwargs["download_dir"] = str(download_dir)
        nltk.download("wordnet", **dl_kwargs)
        nltk.download("omw-1.4", **dl_kwargs)
        _ = wn.synsets("test")
        return wn


def _pos_name(pos: str) -> str:
    return {
        "n": "noun",
        "v": "verb",
        "a": "adjective",
        "s": "adjective",
        "r": "adverb",
    }.get(str(pos), "word")


def _normalize_gloss(gloss: str) -> Tuple[str, str]:
    raw = _clean(gloss)
    if ";" in raw:
        head, tail = raw.split(";", 1)
        return _clean(head), _clean(tail)
    return raw, ""


def _lemma_list_text(lemmas: List[str], exclude: str = "", max_items: int = 4) -> str:
    seen = set()
    out: List[str] = []
    ex = exclude.lower().strip()
    for x in lemmas:
        s = _safe_word(x)
        if not s:
            continue
        key = s.lower()
        if key == ex or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= max_items:
            break
    return ", ".join(out)


def _collect_wordnet_senses(wn) -> Tuple[List[Dict[str, Any]], int]:
    senses: List[Dict[str, Any]] = []
    lemma_vocab = set()
    for syn in wn.all_synsets():
        pos = _pos_name(syn.pos())
        gloss_main, gloss_tail = _normalize_gloss(str(syn.definition() or syn.name()))
        examples = [_clean(x) for x in syn.examples() if _clean(x)]
        lemmas = [str(l.name()) for l in syn.lemmas()]
        if not lemmas:
            continue
        for lemma in lemmas:
            lemma_clean = _safe_word(lemma)
            if not lemma_clean:
                continue
            lemma_vocab.add(lemma_clean.lower())
            senses.append(
                {
                    "lemma": lemma_clean,
                    "lemma_raw": lemma,
                    "pos": pos,
                    "synset": syn.name(),
                    "definition": gloss_main or _clean(syn.name().replace(".", " ")),
                    "gloss_extra": gloss_tail,
                    "examples": examples[:3],
                    "lemmas": lemmas[:8],
                }
            )
    return senses, len(lemma_vocab)


def _make_define_row(sense: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    word = sense["lemma"]
    pos = sense["pos"]
    definition = sense["definition"]
    extra = sense.get("gloss_extra", "")
    examples: List[str] = list(sense.get("examples", []))
    user_frames = [
        "What does \"{w}\" mean?",
        "Define the word {w}.",
        "Dictionary meaning of {w}?",
        "What is the meaning of {w} as a {pos}?",
        "Give a clear definition of {w}.",
    ]
    user = rng.choice(user_frames).format(w=word, pos=pos)
    ans = f"{word} ({pos}) means {definition}."
    if extra and rng.random() < 0.65:
        ans += f" Note: {extra}."
    if examples and rng.random() < 0.50:
        ans += f" Example: {examples[0]}"
    return {
        "user": _clean(user),
        "assistant": _clean(ans),
        "topic": "dictionary_meanings",
        "task": "define",
        "source": "wordnet",
        "word": word,
        "pos": pos,
        "synset": sense["synset"],
    }


def _make_synonym_row(sense: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    word = sense["lemma"]
    pos = sense["pos"]
    lemmas = list(sense.get("lemmas", []))
    syns = _lemma_list_text(lemmas, exclude=word, max_items=5)
    if not syns:
        return None
    user_frames = [
        "Give synonyms for {w}.",
        "What are some synonyms of {w} ({pos})?",
        "List a few words close in meaning to {w}.",
    ]
    user = rng.choice(user_frames).format(w=word, pos=pos)
    ans = f"For this sense of {word} ({pos}), similar words include {syns}."
    return {
        "user": _clean(user),
        "assistant": _clean(ans),
        "topic": "dictionary_meanings",
        "task": "synonyms",
        "source": "wordnet",
        "word": word,
        "pos": pos,
        "synset": sense["synset"],
    }


def _make_usage_row(sense: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    word = sense["lemma"]
    pos = sense["pos"]
    definition = sense["definition"]
    examples: List[str] = list(sense.get("examples", []))
    user_frames = [
        "Use the word {w} in a sentence and explain its meaning.",
        "Give a sentence using {w} correctly.",
        "Show how to use {w} in context.",
    ]
    user = rng.choice(user_frames).format(w=word)
    if examples:
        sentence = examples[0]
        ans = f"Example sentence: {sentence} Meaning here: {word} ({pos}) means {definition}."
    else:
        # Synthetic example when WordNet has no example sentence.
        starter = rng.choice(
            [
                f"In context, {word} can describe an idea clearly.",
                f"The teacher explained {word} during the lesson.",
                f"They used the word {word} in the discussion.",
            ]
        )
        ans = f"{starter} Meaning: {word} ({pos}) means {definition}."
    return {
        "user": _clean(user),
        "assistant": _clean(ans),
        "topic": "dictionary_meanings",
        "task": "usage",
        "source": "wordnet",
        "word": word,
        "pos": pos,
        "synset": sense["synset"],
    }


def _make_compare_row(sense: Dict[str, Any], rng: random.Random) -> Optional[Dict[str, Any]]:
    word = sense["lemma"]
    pos = sense["pos"]
    lemmas = [_safe_word(x) for x in sense.get("lemmas", [])]
    alt = ""
    for x in lemmas:
        if x and x.lower() != word.lower():
            alt = x
            break
    if not alt:
        return None
    user = rng.choice(
        [
            "What is the difference between {w} and {alt}?",
            "Compare {w} with {alt} in meaning.",
        ]
    ).format(w=word, alt=alt)
    definition = sense["definition"]
    ans = (
        f"In this WordNet sense, {word} ({pos}) means {definition}. "
        f"{alt} is related, but context determines whether it is an exact substitute."
    )
    return {
        "user": _clean(user),
        "assistant": _clean(ans),
        "topic": "dictionary_meanings",
        "task": "compare",
        "source": "wordnet",
        "word": word,
        "pos": pos,
        "synset": sense["synset"],
    }


def _make_row_for_sense(sense: Dict[str, Any], rng: random.Random, mode: str = "coverage") -> Dict[str, Any]:
    if mode == "coverage":
        return _make_define_row(sense, rng)

    choices = ["define", "usage", "syn", "compare"]
    weights = [40, 28, 18, 14]
    pick = rng.choices(choices, weights=weights, k=1)[0]
    if pick == "usage":
        return _make_usage_row(sense, rng)
    if pick == "syn":
        row = _make_synonym_row(sense, rng)
        if row is not None:
            return row
        return _make_define_row(sense, rng)
    if pick == "compare":
        row = _make_compare_row(sense, rng)
        if row is not None:
            return row
        return _make_usage_row(sense, rng)
    return _make_define_row(sense, rng)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a WordNet dictionary-meanings conversation dataset (all senses coverage + augmentation).")
    ap.add_argument("--output", default="conversation_data.dictionary_wordnet_meanings_v1.jsonl")
    ap.add_argument("--target", type=int, default=3000000, help="Target number of rows; coverage rows are written first, then augmented to target.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress_every", type=int, default=250000)
    ap.add_argument("--nltk_data_dir", default="", help="Optional NLTK data directory for WordNet downloads/cache.")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    nltk_data_dir = _clean(args.nltk_data_dir)
    wn = _ensure_wordnet(download_dir=nltk_data_dir or None)
    senses, unique_words = _collect_wordnet_senses(wn)
    if not senses:
        raise RuntimeError("No WordNet senses found.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = max(1, int(args.target))

    task_counts: Dict[str, int] = {}
    pos_counts: Dict[str, int] = {}
    rows_written = 0

    with out_path.open("w", encoding="utf-8") as f:
        # Coverage pass: one definition row per lemma-sense entry.
        for s in senses:
            row = _make_row_for_sense(s, rng, mode="coverage")
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1
            task = str(row.get("task", ""))
            pos = str(row.get("pos", ""))
            task_counts[task] = task_counts.get(task, 0) + 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            if args.progress_every > 0 and rows_written % int(args.progress_every) == 0:
                print(f"Wrote {rows_written}/{target} (coverage)")
            if rows_written >= target:
                break

        # Augmentation pass: richer tasks, still grounded in WordNet senses.
        while rows_written < target:
            s = senses[rng.randrange(len(senses))]
            row = _make_row_for_sense(s, rng, mode="augment")
            if rng.random() < 0.06:
                row["user"] = _clean("Please answer like a dictionary teacher. " + str(row["user"]))
            elif rng.random() < 0.05:
                row["user"] = _clean("Quick vocabulary question: " + str(row["user"]))
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1
            task = str(row.get("task", ""))
            pos = str(row.get("pos", ""))
            task_counts[task] = task_counts.get(task, 0) + 1
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            if args.progress_every > 0 and rows_written % int(args.progress_every) == 0:
                print(f"Wrote {rows_written}/{target}")

    print(f"Unique WordNet lemmas covered: {unique_words}")
    print(f"WordNet lemma-sense entries covered: {len(senses)}")
    print(f"Output rows: {rows_written}")
    print("POS:", ", ".join(f"{k}={v}" for k, v in sorted(pos_counts.items())))
    print("Tasks:", ", ".join(f"{k}={v}" for k, v in sorted(task_counts.items())))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
