import argparse
import json
import random
import re
from typing import List, Sequence, Tuple

from chat_pipeline import load_conversation_examples


def extract_last_user(context: str) -> str:
    for line in reversed(context.splitlines()):
        if line.strip().lower().startswith("user:"):
            return line.split(":", 1)[1].strip()
    return context.strip()


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def apply_replacements(text: str, replacements: Sequence[Tuple[str, str]], rng: random.Random) -> str:
    out = text
    picks = list(replacements)
    rng.shuffle(picks)
    num = rng.randint(0, min(3, len(picks)))
    for src, dst in picks[:num]:
        if src in out:
            out = out.replace(src, dst)
    return out


def maybe_add_prefix(text: str, prefixes: Sequence[str], rng: random.Random, p: float) -> str:
    if rng.random() < p:
        return rng.choice(prefixes) + text
    return text


def maybe_add_suffix(text: str, suffixes: Sequence[str], rng: random.Random, p: float) -> str:
    if rng.random() < p:
        return text + rng.choice(suffixes)
    return text


def mutate_user(text: str, rng: random.Random, creative_prob: float) -> str:
    replacements = [
        ("how do i", "how can i"),
        ("can you", "could you"),
        ("fix", "resolve"),
        ("error", "issue"),
        ("script", "program"),
        ("app", "application"),
        ("help", "assist"),
        ("please", ""),
        ("thanks", "thank you"),
    ]
    prefixes = ["please ", "quick question: ", "hey, ", "can you ", "i need help: "]
    suffixes = [" please", " thanks", " when you can", ""]

    out = text.lower()
    out = apply_replacements(out, replacements, rng)
    out = maybe_add_prefix(out, prefixes, rng, p=0.45)
    out = maybe_add_suffix(out, suffixes, rng, p=0.40)
    out = normalize_spaces(out)
    if not out:
        out = text.strip().lower()
    if rng.random() < 0.3:
        out = out.capitalize()
    if rng.random() < 0.25 and not out.endswith("?"):
        out += "?"
    if rng.random() < creative_prob:
        creative_frames = [
            "explain this in plain english: {}",
            "give me a practical way to handle this: {}",
            "i need a smart approach for this: {}",
            "can you reason through this step by step: {}",
            "help me solve this like an expert: {}",
        ]
        out = rng.choice(creative_frames).format(out)
    return out


def _creative_polish(text: str, rng: random.Random) -> str:
    analogies = [
        "treat debugging like narrowing suspects in an investigation",
        "think of optimization as removing bottlenecks in a pipeline",
        "approach this like proving a theorem with small lemmas",
        "treat this as hypothesis testing, not guesswork",
    ]
    closers = [
        "I can propose a concrete plan if you want.",
        "If you share details, I can make this specific.",
        "We can turn this into a repeatable workflow.",
    ]
    styles = [
        "{}",
        "Short answer: {}",
        "Recommended path: {}",
        "{} " + rng.choice(closers),
        "{} Think of it this way: " + rng.choice(analogies) + ".",
    ]
    return normalize_spaces(rng.choice(styles).format(text))


def mutate_assistant(text: str, rng: random.Random, creative_prob: float) -> str:
    replacements = [
        ("you can", "you should"),
        ("check", "verify"),
        ("then", "and then"),
        ("share", "provide"),
        ("run", "execute"),
        ("start", "begin"),
        ("please", ""),
        ("I will", "I'll"),
        ("do not", "don't"),
    ]
    prefixes = ["", "Sure. ", "Okay. ", "Got it. ", "Understood. "]
    suffixes = ["", " Let me know if you want a deeper walkthrough.", " If you share logs, I can be more specific."]

    out = text.strip()
    out = apply_replacements(out, replacements, rng)
    out = maybe_add_prefix(out, prefixes, rng, p=0.55)
    out = maybe_add_suffix(out, suffixes, rng, p=0.30)
    out = normalize_spaces(out)
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    if rng.random() < creative_prob:
        out = _creative_polish(out, rng)
    return out


def blend_pairs(pair_a: Tuple[str, str], pair_b: Tuple[str, str], rng: random.Random) -> Tuple[str, str]:
    ua, aa = pair_a
    ub, ab = pair_b
    user = normalize_spaces(rng.choice([f"{ua} and also {ub}", f"{ua}; plus {ub}", f"{ua}. also {ub}"]))
    assistant = normalize_spaces(
        rng.choice(
            [
                f"{aa} Then {ab}",
                f"{aa} Also, {ab[:1].lower() + ab[1:] if ab else ab}",
                f"{ab} Then {aa[:1].lower() + aa[1:] if aa else aa}",
            ]
        )
    )
    return user, assistant


def sanitize_pair(user: str, assistant: str) -> Tuple[str, str]:
    user = normalize_spaces(user)
    assistant = normalize_spaces(assistant)
    if len(user) > 280:
        user = user[:280].rstrip()
    if len(assistant) > 420:
        assistant = assistant[:420].rstrip()
    return user, assistant


def main():
    ap = argparse.ArgumentParser(description="Expand small conversation JSONL into a larger synthetic dataset.")
    ap.add_argument("--input", required=True, help="Input JSONL with conversation data.")
    ap.add_argument("--output", default="conversation_data.large.jsonl", help="Output expanded JSONL file.")
    ap.add_argument("--target", type=int, default=5000, help="Target number of examples to generate.")
    ap.add_argument("--creative_prob", type=float, default=0.35, help="Probability of creative-style augmentation.")
    ap.add_argument("--blend_prob", type=float, default=0.20, help="Probability of blending two pairs into one example.")
    ap.add_argument("--allow_duplicates", action="store_true", help="Allow duplicate user/assistant rows.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = load_conversation_examples(args.input)
    pairs: List[Tuple[str, str]] = []
    for ex in examples:
        user = extract_last_user(ex.context)
        assistant = ex.response.strip()
        if user and assistant:
            pairs.append((user, assistant))

    if not pairs:
        raise RuntimeError("No usable user/assistant pairs found in input dataset.")

    rows: List[dict] = []
    seen = set()

    def maybe_add_row(user: str, assistant: str) -> bool:
        user, assistant = sanitize_pair(user, assistant)
        if not user or not assistant:
            return False
        key = (user.lower(), assistant.lower())
        if (not args.allow_duplicates) and key in seen:
            return False
        rows.append({"user": user, "assistant": assistant})
        seen.add(key)
        return True

    # Keep originals first.
    for u, a in pairs:
        maybe_add_row(u, a)

    target = max(args.target, len(rows))
    max_attempts = max(target * 20, 10000)
    attempts = 0
    while len(rows) < target and attempts < max_attempts:
        attempts += 1
        if len(pairs) >= 2 and rng.random() < args.blend_prob:
            p1 = rng.choice(pairs)
            p2 = rng.choice(pairs)
            if p1 == p2:
                continue
            u, a = blend_pairs(p1, p2, rng)
        else:
            u, a = rng.choice(pairs)
            u = mutate_user(u, rng, creative_prob=args.creative_prob)
            a = mutate_assistant(a, rng, creative_prob=args.creative_prob)
        maybe_add_row(u, a)

    # If dedupe exhausted the search space, fill with duplicates to hit exact target.
    while len(rows) < target:
        u, a = rng.choice(pairs)
        u = mutate_user(u, rng, creative_prob=args.creative_prob)
        a = mutate_assistant(a, rng, creative_prob=args.creative_prob)
        rows.append({"user": normalize_spaces(u), "assistant": normalize_spaces(a)})

    with open(args.output, "w", encoding="utf-8") as f:
        for row in rows[:target]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Input examples: {len(examples)}")
    print(f"Expanded examples: {target}")
    print(f"Unique pairs tracked: {len(seen)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
