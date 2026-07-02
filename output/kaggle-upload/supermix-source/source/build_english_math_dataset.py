import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple


NOUNS = [
    "student", "teacher", "engineer", "writer", "designer", "chef", "driver", "doctor",
    "musician", "manager", "neighbor", "friend", "team", "company", "family", "class",
    "project", "meeting", "report", "email", "idea", "plan", "schedule", "budget",
    "garden", "library", "museum", "restaurant", "computer", "phone", "notebook", "lesson",
]
VERBS = [
    "write", "finish", "review", "send", "prepare", "improve", "explain", "organize",
    "clean", "build", "test", "read", "share", "plan", "deliver", "study", "practice",
    "update", "check", "solve", "design", "draft", "revise", "discuss",
]
TRANSITIVE_VERBS = [
    "write", "finish", "review", "send", "prepare", "improve", "explain", "organize",
    "clean", "build", "test", "read", "share", "plan", "deliver", "study", "practice",
    "update", "check", "solve", "design", "draft", "revise", "discuss",
]
INTRANSITIVE_VERB_PHRASES = [
    "work carefully",
    "arrive early",
    "practice daily",
    "study consistently",
    "speak clearly",
    "respond politely",
]
ADJECTIVES = [
    "clear", "useful", "careful", "quick", "detailed", "friendly", "formal", "short",
    "helpful", "creative", "strong", "simple", "effective", "professional", "polite",
]
TIME_WORDS = ["today", "tomorrow", "this week", "before lunch", "after class", "by Friday", "soon"]
PLACES = ["at the office", "in class", "at home", "during the meeting", "at the library", "online"]
COMMON_MISSPELLINGS = [
    ("recieve", "receive"),
    ("definately", "definitely"),
    ("seperate", "separate"),
    ("occured", "occurred"),
    ("wierd", "weird"),
    ("enviroment", "environment"),
    ("goverment", "government"),
    ("adress", "address"),
    ("untill", "until"),
    ("succesful", "successful"),
]
SYNONYMS = [
    ("happy", "glad"),
    ("angry", "upset"),
    ("big", "large"),
    ("small", "tiny"),
    ("quick", "rapid"),
    ("careful", "cautious"),
    ("helpful", "useful"),
    ("start", "begin"),
    ("finish", "complete"),
    ("fix", "repair"),
]
DEFINITIONS = [
    ("punctuation", "the system of marks such as commas and periods used to clarify meaning in writing"),
    ("paragraph", "a group of related sentences focused on one main idea"),
    ("thesis statement", "a sentence that states the main claim of a piece of writing"),
    ("synonym", "a word with the same or nearly the same meaning as another word"),
    ("verb", "a word that shows an action, occurrence, or state of being"),
    ("adjective", "a word that describes a noun or pronoun"),
    ("topic sentence", "a sentence that introduces the main idea of a paragraph"),
]
UNITS = [
    ("cm", "m", 100.0),
    ("m", "cm", 0.01),
    ("mm", "cm", 10.0),
    ("kg", "g", 0.001),
    ("g", "kg", 1000.0),
    ("L", "mL", 0.001),
    ("mL", "L", 1000.0),
]
WORD_PROBLEM_ITEMS = ["apples", "books", "notebooks", "tickets", "stickers", "pencils", "cookies"]
COMPARATORS = [">", "<", "="]


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


def _cap(s: str) -> str:
    s = _clean(s)
    if not s:
        return s
    return s[0].upper() + s[1:]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _english_grammar_fix(rng: random.Random) -> Dict[str, str]:
    subject = rng.choice(["He", "She", "It", "The team", "My friend", "The teacher"])
    verb_obj_pairs = [
        ("write", "the report"),
        ("write", "the email"),
        ("go", "to school"),
        ("have", "a plan"),
        ("make", "the schedule"),
        ("need", "the lesson notes"),
        ("plan", "the meeting"),
    ]
    verb, obj = rng.choice(verb_obj_pairs)
    time = rng.choice(TIME_WORDS)
    wrong = f"{subject} {verb} {obj} {time}."
    fixes = {
        "write": "writes",
        "go": "goes",
        "have": "has",
        "make": "makes",
        "need": "needs",
        "plan": "plans",
    }
    corrected = f"{subject} {fixes[verb]} {obj} {time}."
    user = f"Correct the grammar in this sentence and give the corrected version only: {wrong}"
    return {"user": user, "assistant": corrected, "topic": "english_foundations", "task": "grammar_fix"}


def _english_punctuation(rng: random.Random) -> Dict[str, str]:
    name = rng.choice(["maria", "john", "alex", "samira", "david", "olivia"])
    sentence = f"{name} said we should finish the draft today however we can revise it tomorrow"
    corrected = f"{_cap(name)} said, \"We should finish the draft today; however, we can revise it tomorrow.\""
    user = f"Add punctuation and capitalization to this sentence: {sentence}"
    return {"user": user, "assistant": corrected, "topic": "english_foundations", "task": "punctuation_capitalization"}


def _english_spelling(rng: random.Random) -> Dict[str, str]:
    wrong, right = rng.choice(COMMON_MISSPELLINGS)
    user = f"Fix the spelling in this sentence: The {wrong} of the document needs one more review."
    assistant = f"The {right} of the document needs one more review."
    return {"user": user, "assistant": assistant, "topic": "english_foundations", "task": "spelling_fix"}


def _english_rewrite_tone(rng: random.Random) -> Dict[str, str]:
    tone = rng.choice(["formal", "friendly", "concise"])
    base = rng.choice(
        [
            "send me the files when you get a chance",
            "we need to reschedule the meeting",
            "your draft is good but needs a few edits",
            "can you explain this part again",
        ]
    )
    if tone == "formal":
        assistant = {
            "send me the files when you get a chance": "Please send me the files when you have a moment.",
            "we need to reschedule the meeting": "We need to reschedule the meeting to a different time.",
            "your draft is good but needs a few edits": "Your draft is strong, but it requires a few revisions.",
            "can you explain this part again": "Could you please explain this section again?",
        }[base]
    elif tone == "friendly":
        assistant = {
            "send me the files when you get a chance": "Hey, could you send me the files when you have a minute?",
            "we need to reschedule the meeting": "Looks like we need to move the meeting to another time.",
            "your draft is good but needs a few edits": "Nice draft so far! It just needs a few small edits.",
            "can you explain this part again": "Could you go over this part one more time?",
        }[base]
    else:
        assistant = {
            "send me the files when you get a chance": "Please send the files when available.",
            "we need to reschedule the meeting": "We need to reschedule the meeting.",
            "your draft is good but needs a few edits": "Your draft is good but needs edits.",
            "can you explain this part again": "Please explain this part again.",
        }[base]
    user = f"Rewrite this in a {tone} tone: \"{base}\""
    return {"user": user, "assistant": assistant, "topic": "english_foundations", "task": "tone_rewrite"}


def _english_vocab(rng: random.Random) -> Dict[str, str]:
    if rng.random() < 0.55:
        w, syn = rng.choice(SYNONYMS)
        user = f"Give a simple synonym for the word \"{w}\" and use it in a short sentence."
        assistant = f"Synonym: {syn}. Example: I felt {syn} after finishing the project."
        task = "synonym_example"
    else:
        term, definition = rng.choice(DEFINITIONS)
        user = f"Define \"{term}\" in plain English."
        assistant = _cap(definition) + "."
        task = "definition"
    return {"user": user, "assistant": assistant, "topic": "english_foundations", "task": task}


def _english_sentence_build(rng: random.Random) -> Dict[str, str]:
    noun = rng.choice(NOUNS)
    adj = rng.choice(ADJECTIVES)
    place = rng.choice(PLACES)
    time = rng.choice(TIME_WORDS)
    if rng.random() < 0.75:
        verb = rng.choice(TRANSITIVE_VERBS)
        obj = rng.choice(["report", "draft", "plan", "message", "task", "lesson", "summary", "schedule"])
        assistant = _cap(f"the {noun} will {verb} a {adj} {obj} {place} {time}.")
    else:
        verb = rng.choice(INTRANSITIVE_VERB_PHRASES)
        assistant = _cap(f"the {noun} will {verb} {place} {time} and stay {adj}.")
    # `verb` is needed in the prompt after branch selection.
    user = (
        f"Write one clear English sentence using these words: {noun}, {verb.split()[0]}, {adj}. "
        f"Keep it natural and grammatical."
    )
    return {"user": user, "assistant": assistant, "topic": "english_foundations", "task": "sentence_construction"}


def _english_paragraph_edit(rng: random.Random) -> Dict[str, str]:
    topic = rng.choice(["study habits", "healthy routines", "teamwork", "time management", "writing practice"])
    singular_topic = topic in {"teamwork", "time management", "writing practice"}
    be_wrong = "are" if singular_topic else "is"
    help_wrong = "help" if singular_topic else "helps"
    make_wrong = "make" if singular_topic else "makes"
    be_correct = "is" if singular_topic else "are"
    subject_pron = "it" if singular_topic else "they"
    wrong = (
        f"i think {topic} {be_wrong} important because it {help_wrong} people stay focused, "
        f"and it also {make_wrong} daily work easier"
    )
    corrected = (
        f"I think {topic} {be_correct} important because {subject_pron} help{'s' if subject_pron == 'it' else ''} people stay focused, "
        f"and {subject_pron} also make{'s' if subject_pron == 'it' else ''} daily work easier."
    )
    user = "Proofread this sentence for capitalization, punctuation, and grammar: " + wrong
    return {"user": user, "assistant": corrected, "topic": "english_foundations", "task": "proofread"}


def _english_short_writing(rng: random.Random) -> Dict[str, str]:
    scenario = rng.choice(
        [
            "ask a teacher for an extension",
            "thank a coworker for helpful feedback",
            "invite a friend to study together",
            "decline a meeting politely and suggest another time",
            "request clarification about an assignment",
        ]
    )
    user = f"Write a short, polite message in English to {scenario}."
    templates = {
        "ask a teacher for an extension": "Hello, I am writing to ask whether I could have a short extension on the assignment due date. I have made progress, but I need a little more time to submit my best work. Thank you for considering my request.",
        "thank a coworker for helpful feedback": "Hi, thank you for the helpful feedback on my draft. Your comments made the next steps much clearer, and I appreciate the time you took to review it.",
        "invite a friend to study together": "Hey, would you like to study together this week? I think it would help us review the material and compare notes before the quiz.",
        "decline a meeting politely and suggest another time": "Thank you for the invitation. I am not available at that time, but I would be happy to meet tomorrow afternoon or later this week if that works for you.",
        "request clarification about an assignment": "Hello, I have a quick question about the assignment instructions. Could you clarify what should be included in the final section? Thank you.",
    }
    assistant = templates[scenario]
    return {"user": user, "assistant": assistant, "topic": "english_foundations", "task": "message_writing"}


def _english_row(rng: random.Random) -> Dict[str, str]:
    fn = rng.choices(
        [
            _english_grammar_fix,
            _english_punctuation,
            _english_spelling,
            _english_rewrite_tone,
            _english_vocab,
            _english_sentence_build,
            _english_paragraph_edit,
            _english_short_writing,
        ],
        weights=[18, 14, 10, 16, 12, 12, 10, 8],
        k=1,
    )[0]
    return fn(rng)


def _fmt_num(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return s


def _math_arithmetic(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(2, 500)
    b = rng.randint(2, 500)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    elif op == "*":
        ans = a * b
    else:
        b = rng.randint(2, 30)
        q = rng.randint(2, 80)
        a = b * q
        ans = a // b
    user = f"Solve this basic math problem: {a} {op} {b}"
    if rng.random() < 0.35:
        assistant = f"{a} {op} {b} = {ans}"
    else:
        assistant = str(ans)
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "arithmetic"}


def _math_order_ops(rng: random.Random) -> Dict[str, str]:
    a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 20)
    d = rng.randint(1, 12)
    expr_type = rng.choice([0, 1, 2])
    if expr_type == 0:
        expr = f"{a} + {b} * {c}"
        ans = a + b * c
    elif expr_type == 1:
        expr = f"({a} + {b}) * {d}"
        ans = (a + b) * d
    else:
        expr = f"{a * d} / {d} + {c}"
        ans = (a * d) / d + c
    user = f"Use order of operations to evaluate: {expr}"
    assistant = _fmt_num(ans)
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "order_of_operations"}


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _math_fraction(rng: random.Random) -> Dict[str, str]:
    b = rng.randint(2, 12)
    d = rng.randint(2, 12)
    a = rng.randint(1, b - 1)
    c = rng.randint(1, d - 1)
    op = rng.choice(["+", "-"])
    num = a * d + c * b if op == "+" else a * d - c * b
    den = b * d
    g = _gcd(abs(num), den) if num != 0 else den
    sn, sd = num // g, den // g
    user = f"Compute and simplify: {a}/{b} {op} {c}/{d}"
    if sd == 1:
        assistant = str(sn)
    else:
        assistant = f"{sn}/{sd}"
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "fractions"}


def _math_percent(rng: random.Random) -> Dict[str, str]:
    percent = rng.choice([5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90])
    base = rng.randint(20, 2000)
    ans = base * percent / 100.0
    user = f"What is {percent}% of {base}?"
    assistant = _fmt_num(ans)
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "percent"}


def _math_ratio(rng: random.Random) -> Dict[str, str]:
    a = rng.randint(2, 60)
    b = rng.randint(2, 60)
    g = _gcd(a, b)
    user = f"Simplify this ratio: {a}:{b}"
    assistant = f"{a//g}:{b//g}"
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "ratio"}


def _math_algebra(rng: random.Random) -> Dict[str, str]:
    x = rng.randint(-20, 20)
    form = rng.choice(["x_plus", "ax", "x_minus"])
    if form == "x_plus":
        b = rng.randint(-20, 20)
        c = x + b
        user = f"Solve for x: x + {b} = {c}"
    elif form == "x_minus":
        b = rng.randint(-20, 20)
        c = x - b
        user = f"Solve for x: x - {b} = {c}"
    else:
        a = rng.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
        c = a * x
        user = f"Solve for x: {a}x = {c}"
    assistant = f"x = {x}"
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "algebra_one_step"}


def _math_average(rng: random.Random) -> Dict[str, str]:
    nums = [rng.randint(1, 100) for _ in range(rng.randint(3, 6))]
    ans = sum(nums) / len(nums)
    user = f"Find the average (mean) of these numbers: {', '.join(map(str, nums))}"
    assistant = _fmt_num(ans)
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "average"}


def _math_conversion(rng: random.Random) -> Dict[str, str]:
    from_unit, to_unit, factor = rng.choice(UNITS)
    value = rng.randint(1, 5000)
    if factor >= 1.0:
        ans = value / factor
    else:
        ans = value / factor
    user = f"Convert {value} {from_unit} to {to_unit}."
    assistant = f"{_fmt_num(ans)} {to_unit}"
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "unit_conversion"}


def _math_word_problem(rng: random.Random) -> Dict[str, str]:
    item = rng.choice(WORD_PROBLEM_ITEMS)
    start = rng.randint(5, 80)
    add = rng.randint(2, 40)
    take = rng.randint(1, min(30, start + add - 1))
    total = start + add - take
    user = (
        f"A student has {start} {item}. They get {add} more and then give away {take}. "
        f"How many {item} do they have now?"
    )
    if rng.random() < 0.4:
        assistant = f"{start} + {add} - {take} = {total}"
    else:
        assistant = str(total)
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "word_problem"}


def _math_compare(rng: random.Random) -> Dict[str, str]:
    a = round(rng.uniform(-50, 50), 2)
    b = round(rng.uniform(-50, 50), 2)
    comp = "=" if abs(a - b) < 1e-9 else (">" if a > b else "<")
    user = f"Compare these decimals using >, <, or = : {_fmt_num(a)} __ {_fmt_num(b)}"
    assistant = comp
    return {"user": user, "assistant": assistant, "topic": "basic_math", "task": "decimal_compare"}


def _math_row(rng: random.Random) -> Dict[str, str]:
    fn = rng.choices(
        [
            _math_arithmetic,
            _math_order_ops,
            _math_fraction,
            _math_percent,
            _math_ratio,
            _math_algebra,
            _math_average,
            _math_conversion,
            _math_word_problem,
            _math_compare,
        ],
        weights=[18, 12, 12, 10, 8, 10, 8, 8, 10, 4],
        k=1,
    )[0]
    row = fn(rng)
    # Add occasional short step-by-step reasoning for training variety.
    if row.get("topic") == "basic_math" and rng.random() < 0.22 and row["task"] in {"arithmetic", "percent", "word_problem", "algebra_one_step"}:
        ans = row["assistant"]
        if row["task"] == "algebra_one_step" and "x =" in ans:
            row["assistant"] = f"Step 1: isolate x. Step 2: {ans}."
        elif row["task"] == "word_problem":
            row["assistant"] = f"Compute the net change and then the total. Answer: {ans}."
        else:
            row["assistant"] = f"Calculate directly. Answer: {ans}."
    return row


def generate_row(rng: random.Random, english_ratio: float) -> Dict[str, str]:
    if rng.random() < english_ratio:
        row = _english_row(rng)
    else:
        row = _math_row(rng)
    # Light style diversification to improve conversational coverage.
    if rng.random() < 0.08:
        row["user"] = "Please help with this.\n" + row["user"]
    elif rng.random() < 0.08:
        row["user"] = "Quick question: " + row["user"]
    if rng.random() < 0.05 and row.get("topic") == "english_foundations":
        row["assistant"] = "Recommended answer: " + row["assistant"]
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a large synthetic English + basic math conversation dataset.")
    ap.add_argument("--output", default="conversation_data.english_math_core_v1.jsonl")
    ap.add_argument("--target", type=int, default=4000000)
    ap.add_argument("--english_ratio", type=float, default=0.72, help="Fraction of English rows (rest are basic math).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--progress_every", type=int, default=250000)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    target = max(1, int(args.target))
    english_ratio = min(0.98, max(0.02, float(args.english_ratio)))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"english_foundations": 0, "basic_math": 0}
    task_counts: Dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(target):
            row = generate_row(rng, english_ratio=english_ratio)
            row["user"] = _clean(row["user"])
            row["assistant"] = _clean(row["assistant"])
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            topic = str(row.get("topic", ""))
            counts[topic] = counts.get(topic, 0) + 1
            task = str(row.get("task", ""))
            task_counts[task] = task_counts.get(task, 0) + 1
            if args.progress_every > 0 and (i + 1) % int(args.progress_every) == 0:
                print(f"Wrote {i + 1}/{target}")

    print(f"Output rows: {target}")
    print(f"English rows: {counts.get('english_foundations', 0)}")
    print(f"Math rows: {counts.get('basic_math', 0)}")
    top_tasks = sorted(task_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    print("Top tasks:", ", ".join(f"{k}={v}" for k, v in top_tasks))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
