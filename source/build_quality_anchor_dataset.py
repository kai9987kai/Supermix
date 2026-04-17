import json
import random
from pathlib import Path


def add_row(rows, user, assistant, source):
    rows.append(
        {
            "user": str(user).strip(),
            "assistant": str(assistant).strip(),
            "source": source,
        }
    )


def build_rows(seed: int = 42):
    rng = random.Random(seed)
    rows = []

    # Arithmetic and concise answer anchors.
    for _ in range(120):
        a = rng.randint(2, 180)
        b = rng.randint(2, 180)
        op = rng.choice(["add", "sub", "mul", "div"])
        if op == "add":
            user = f"What is {a} + {b}? Give just the answer."
            assistant = str(a + b)
        elif op == "sub":
            user = f"What is {a} - {b}? Give just the answer."
            assistant = str(a - b)
        elif op == "mul":
            user = f"What is {a} * {b}? Give just the answer."
            assistant = str(a * b)
        else:
            denom = max(1, b)
            user = f"What is {a} / {denom}? Round to 2 decimals and give just the answer."
            assistant = f"{(a / denom):.2f}"
        add_row(rows, user, assistant, "quality_anchor_math")

    # Core ML concept anchors.
    concept_pairs = [
        (
            "Explain overfitting in 3 short bullet points.",
            "- Overfitting means a model memorizes training examples instead of learning general patterns.\n- It usually looks strong on training data but weak on new, unseen data.\n- Common fixes include simpler models, regularization, more data, and early stopping.",
        ),
        (
            "Explain underfitting in 3 short bullet points.",
            "- Underfitting means a model is too simple to capture important patterns.\n- It performs poorly on both training and test data.\n- Common fixes include better features, more training time, or a more expressive model.",
        ),
        (
            "What is the difference between precision and recall?",
            "Precision is the share of predicted positives that are truly positive. Recall is the share of true positives the model successfully finds.",
        ),
        (
            "What is bias-variance tradeoff?",
            "It is the balance between a model that is too simple (high bias) and too sensitive to noise (high variance). Good models minimize both enough to generalize well.",
        ),
    ]
    for _ in range(25):
        for user, assistant in concept_pairs:
            add_row(rows, user, assistant, "quality_anchor_ml")

    # Formatting and instruction-following anchors.
    format_prompts = [
        (
            "Summarize why testing matters in exactly 2 bullet points.",
            "- Testing catches regressions early.\n- Testing makes refactoring safer and faster.",
        ),
        (
            "Answer in one sentence: Why use version control?",
            "Version control tracks changes, enables collaboration, and lets teams safely recover from mistakes.",
        ),
        (
            "Give a concise definition of API in one sentence.",
            "An API is a defined interface that lets one software system communicate with another.",
        ),
        (
            "How can I make model responses more reliable? Give 3 bullet points.",
            "- Use cleaner, higher-quality training data.\n- Add preference tuning or human feedback for response quality.\n- Use stable decoding settings and evaluate with representative tests.",
        ),
    ]
    for _ in range(20):
        for user, assistant in format_prompts:
            add_row(rows, user, assistant, "quality_anchor_format")

    # Deduplicate while preserving order.
    seen = set()
    out = []
    for row in rows:
        key = (row["user"], row["assistant"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main():
    rows = build_rows(seed=42)
    out_path = Path("datasets/conversation_data.quality_anchor_v2.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[done] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
