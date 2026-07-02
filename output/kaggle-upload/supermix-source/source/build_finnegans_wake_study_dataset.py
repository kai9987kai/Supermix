#!/usr/bin/env python3
"""
Build a non-verbatim Finnegans Wake study/style dataset.

Important:
- This script does NOT include or reproduce the copyrighted text of Finnegans Wake.
- It generates synthetic study prompts, analysis tasks, reading strategies, and
  style-oriented exercises about the work.

This is intended as a safe training supplement for:
- literary difficulty handling
- dense, allusive, multilingual wordplay discussion
- reading-strategy coaching
- interpretation and paraphrase skills (without source-text copying)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


THEMES = [
    "dream logic",
    "cyclical structure",
    "multilingual wordplay",
    "portmanteau language",
    "mythic allusion",
    "historical layering",
    "family conflict",
    "river imagery",
    "night language",
    "sound-based meaning",
    "comic distortion",
    "identity shifting",
    "recurrence and repetition",
    "folk speech textures",
    "dense symbolism",
    "narrative ambiguity",
]

READING_GOALS = [
    "finding a stable thread through a difficult passage",
    "noticing patterns without needing a total decoding",
    "tracking recurring names and transformations",
    "balancing sound and sense while reading",
    "staying engaged when a passage feels opaque",
    "using annotations without over-relying on them",
    "reading for rhythm before interpretation",
    "building confidence with short daily sessions",
]

READING_STRATEGIES = [
    "read aloud first to hear rhythm and sound echoes before trying to paraphrase",
    "underline recurring words, sounds, or images and treat them as anchors",
    "summarize only what is clear, then mark ambiguities instead of forcing certainty",
    "use small chunks and revisit them later after reading surrounding passages",
    "track transformations of names and roles rather than expecting fixed identities",
    "separate literal guesses from symbolic or tonal impressions in your notes",
    "compare multiple interpretations and keep a 'working hypothesis' instead of a final answer",
    "focus on one layer at a time: sound, imagery, references, or narrative movement",
]

STYLE_FEATURES = [
    "compressed wordplay",
    "cross-language punning",
    "phonetic echoes",
    "stacked allusions",
    "syntax that slides between voices",
    "dreamlike transitions",
    "shifts in register from comic to elevated",
    "repetition with variation",
    "blended identities and pronouns",
    "fragmented but musical phrasing",
]

COMPARE_WORKS = [
    "Ulysses",
    "modernist poetry",
    "myth retellings",
    "oral storytelling traditions",
    "experimental prose",
    "stream-of-consciousness fiction",
    "surrealist writing",
    "annotated epic literature",
]

STUDENT_CONTEXTS = [
    "a first-time reader",
    "a literature student",
    "a book club participant",
    "a writer studying style",
    "a reader who enjoys poetry but struggles with dense prose",
    "someone returning after a long break from the book",
    "a teacher designing a discussion session",
    "a self-guided reader using annotations",
]

DISCUSSION_ANGLES = [
    "how ambiguity changes the reading experience",
    "whether difficulty can create meaning instead of just blocking it",
    "what sound contributes beyond literal paraphrase",
    "how humor and seriousness coexist in dense writing",
    "how recurring images help orientation",
    "how readers can disagree productively on interpretation",
    "what counts as a 'good' reading of a difficult text",
    "how to read allusion-heavy work without encyclopedic knowledge",
]

PSEUDO_MORPHEMES_LEFT = [
    "river", "night", "whisper", "thunder", "dream", "laugh", "murmur", "echo",
    "tumble", "riddle", "lantern", "brook", "moon", "tongue", "hush", "wake",
]
PSEUDO_MORPHEMES_RIGHT = [
    "glint", "drift", "song", "fold", "trace", "spark", "spill", "bloom",
    "clatter", "ripple", "thread", "mingle", "flare", "murk", "twine", "glow",
]

AFFECT_SCENARIOS = [
    "I feel stupid when I read pages I cannot fully understand.",
    "I keep rereading the same paragraph and getting frustrated.",
    "I want to enjoy difficult literature but I shut down fast.",
    "I feel lost because I cannot tell who is speaking.",
    "I like the sound of the writing but I cannot summarize it.",
    "I worry I am reading it the wrong way.",
]

BOOK_CHAPTER_HINTS = [
    "Book I, early chapters",
    "Book I, mid-book sequence",
    "Book II classroom-focused sections",
    "Book II family-centered scenes",
    "Book III river/listening passages",
    "Book IV closing cyclical movement",
]

LANGUAGES_HINTS = [
    "English with echoes of multiple European languages",
    "mixed registers and mock-formal diction",
    "street-speech textures blended with learned references",
    "high/low style collisions for comic and symbolic effect",
]


def _pseudo_word(rng: random.Random) -> str:
    a = rng.choice(PSEUDO_MORPHEMES_LEFT)
    b = rng.choice(PSEUDO_MORPHEMES_RIGHT)
    join = rng.choice(["", "", "-", ""])
    out = a + join + b
    if rng.random() < 0.35:
        out = out + rng.choice(["ish", "ling", "ward", "wise", "tone", "sway"])
    return out


def _pseudo_line(rng: random.Random) -> str:
    n = rng.randint(6, 14)
    words = []
    for _ in range(n):
        if rng.random() < 0.45:
            words.append(_pseudo_word(rng))
        else:
            words.append(rng.choice([
                "the", "and", "over", "under", "through", "while", "where",
                "a", "that", "this", "near", "again", "between", "round",
            ]))
    line = " ".join(words)
    return line[:1].upper() + line[1:] + "."


def _row(user: str, assistant: str, category: str) -> Dict[str, str]:
    return {
        "user": user,
        "assistant": assistant,
        "category": category,
        "source": "Non-verbatim synthetic study/style tasks about Finnegans Wake (no copyrighted text included)",
    }


def generate_rows(target: int, seed: int) -> Iterable[Dict[str, str]]:
    rng = random.Random(seed)
    for i in range(target):
        mode = i % 10
        theme = THEMES[i % len(THEMES)]
        goal = READING_GOALS[(i * 3) % len(READING_GOALS)]
        strat = READING_STRATEGIES[(i * 5) % len(READING_STRATEGIES)]
        feature = STYLE_FEATURES[(i * 7) % len(STYLE_FEATURES)]
        work = COMPARE_WORKS[(i * 11) % len(COMPARE_WORKS)]
        learner = STUDENT_CONTEXTS[(i * 13) % len(STUDENT_CONTEXTS)]
        angle = DISCUSSION_ANGLES[(i * 17) % len(DISCUSSION_ANGLES)]
        chapter_hint = BOOK_CHAPTER_HINTS[(i * 19) % len(BOOK_CHAPTER_HINTS)]
        lang_hint = LANGUAGES_HINTS[(i * 23) % len(LANGUAGES_HINTS)]

        if mode == 0:
            user = f"How should {learner} approach Finnegans Wake when the main challenge is {goal}?"
            assistant = (
                f"For {learner}, a useful approach is to treat Finnegans Wake as a layered reading experience. "
                f"Start with {strat}. Focus first on {theme}, and do not require a full decoding on the first pass. "
                f"Use short sections, mark recurring patterns, and return later with notes."
            )
            yield _row(user, assistant, "fw_reading_strategy")
        elif mode == 1:
            user = f"Explain how {feature} can create meaning in Finnegans Wake."
            assistant = (
                f"{feature.capitalize()} can carry meaning through sound, association, and repetition even when the syntax is unstable. "
                f"In Finnegans Wake, readers often build interpretation by tracking recurring sonic or imagistic cues rather than only literal paraphrase. "
                f"That makes {feature} part of the narrative logic, not just ornament."
            )
            yield _row(user, assistant, "fw_style_analysis")
        elif mode == 2:
            pseudo = _pseudo_word(rng)
            user = f"Give a plausible interpretation strategy for a Joycean-style portmanteau like '{pseudo}' without pretending there is only one correct answer."
            assistant = (
                f"A good strategy is to split '{pseudo}' into possible sound/word fragments, list 2-4 candidate roots, and test which ones fit the local tone or theme. "
                f"For a Finnegans Wake-style reading, the goal is not a single final answer but a small set of plausible layers and how they interact."
            )
            yield _row(user, assistant, "fw_portmanteau_strategy")
        elif mode == 3:
            user = f"Compare reading Finnegans Wake with reading {work} for someone interested in interpretation rather than plot summary."
            assistant = (
                f"Compared with {work}, Finnegans Wake usually demands more tolerance for ambiguity and stronger attention to sound-patterns, recurring motifs, and allusive density. "
                f"A plot-first method is often less effective. A pattern-first reading can be more productive, especially when tracking {theme}."
            )
            yield _row(user, assistant, "fw_comparison")
        elif mode == 4:
            pseudo_line = _pseudo_line(rng)
            user = (
                "Interpret this synthetic Joycean-style line (not from the book) in plain English and mention two possible layers of meaning:\n"
                f"{pseudo_line}"
            )
            assistant = (
                f"One plain-English reading is that the line creates a drifting, nighttime scene with movement and overlapping voices. "
                f"Two possible layers are: (1) a literal sensory scene built from sound and motion words, and (2) a dreamlike or symbolic layer tied to {theme}. "
                f"Because the phrasing is deliberately blended, multiple readings can coexist."
            )
            yield _row(user, assistant, "fw_synthetic_interpretation")
        elif mode == 5:
            user = f"Create a discussion prompt for a Finnegans Wake reading group about {angle}."
            assistant = (
                f"Discussion prompt: In a difficult passage, how can readers evaluate interpretations when the language is highly allusive and unstable? "
                f"Use {angle} as the focus, and ask participants to cite patterns (sound, repetition, imagery, tone) rather than only plot claims."
            )
            yield _row(user, assistant, "fw_discussion_prompt")
        elif mode == 6:
            user = f"How would you teach a short seminar on Finnegans Wake focused on {theme} in {chapter_hint}?"
            assistant = (
                f"I would structure the seminar in three passes: sound/rhythm, image tracking, and interpretive hypotheses. "
                f"For {chapter_hint}, ask students first to notice recurring motifs tied to {theme}, then compare interpretations in pairs. "
                f"End with a brief reflection on uncertainty as part of the reading process."
            )
            yield _row(user, assistant, "fw_teaching")
        elif mode == 7:
            user = f"What makes Finnegans Wake feel difficult, and how can a reader make progress anyway?"
            assistant = (
                f"It often feels difficult because of compressed wordplay, allusions, shifting voice, and {lang_hint}. "
                f"Progress comes from changing the success metric: aim for pattern recognition, tonal understanding, and selective paraphrase rather than total decoding. "
                f"Small repeated sessions and annotated notes help."
            )
            yield _row(user, assistant, "fw_reading_support")
        elif mode == 8:
            scenario = AFFECT_SCENARIOS[(i * 29) % len(AFFECT_SCENARIOS)]
            user = scenario + " Can you help me keep going with Finnegans Wake?"
            assistant = (
                "Yes. That reaction is common with very dense literature, and it does not mean you are reading badly. "
                "Try a shorter goal: read one paragraph aloud, note three repeated sounds/images, and write one 'best guess' sentence. "
                "Treat uncertainty as part of the method, not a sign of failure."
            )
            yield _row(user, assistant, "fw_empathy_support")
        else:
            user = (
                f"Give me a creative writing exercise inspired by Finnegans Wake's {feature} without copying its text."
            )
            assistant = (
                f"Exercise: Write 6-8 lines using {feature} and {theme}. Blend at least five invented compound words, keep a recurring sound pattern, and let the scene shift as if in a dream. "
                f"After writing, annotate two places where sound influences meaning. This studies technique without reproducing the source text."
            )
            yield _row(user, assistant, "fw_creative_exercise")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a non-verbatim Finnegans Wake study/style dataset.")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--target", type=int, default=500000, help="Number of rows to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in generate_rows(args.target, args.seed):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} rows to {out_path}")
    print(f"Bytes: {out_path.stat().st_size}")


if __name__ == "__main__":
    main()
