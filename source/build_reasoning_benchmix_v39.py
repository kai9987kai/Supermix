#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - resolved on the pod
    raise RuntimeError("The `datasets` package is required to build the v39 reasoning benchmix dataset.") from exc


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _extract_gsm8k_answer(answer_text: str) -> str:
    text = str(answer_text or "")
    if "####" in text:
        text = text.split("####", 1)[1]
    matches = NUMBER_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else _normalize_text(text)


def _clean_gsm8k_solution(answer_text: str) -> str:
    text = str(answer_text or "").replace("<<", "").replace(">>", "")
    if "####" in text:
        body, final_answer = text.split("####", 1)
        body = "\n".join(part.strip() for part in body.splitlines() if part.strip())
        final_answer = _extract_gsm8k_answer(final_answer)
        cleaned = _normalize_text(body)
        if cleaned:
            return f"{cleaned}\nFinal answer: {final_answer}"
        return f"Final answer: {final_answer}"
    return _normalize_text(text)


def _choice_labels(count: int) -> List[str]:
    if count < 1 or count > len(OPTION_LABELS):
        raise ValueError(f"Unsupported number of options: {count}")
    return list(OPTION_LABELS[:count])


def _format_options(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def _sample_rows(rows: Sequence[dict], limit: int, seed: int) -> List[dict]:
    items = list(rows)
    rng = random.Random(seed)
    rng.shuffle(items)
    if limit > 0:
        return items[: min(len(items), limit)]
    return items


def _sample_stratified(rows: Sequence[dict], limit: int, seed: int, key_name: str) -> List[dict]:
    items = list(rows)
    if limit <= 0 or limit >= len(items):
        return items
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in items:
        grouped[_normalize_text(row.get(key_name, "")) or "unknown"].append(row)
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
    ordered_keys = sorted(grouped.keys())
    selected: List[dict] = []
    while len(selected) < limit:
        made_progress = False
        for key in ordered_keys:
            group = grouped[key]
            if not group:
                continue
            selected.append(group.pop())
            made_progress = True
            if len(selected) >= limit:
                break
        if not made_progress:
            break
    rng.shuffle(selected)
    return selected


def _row(
    *,
    user: str,
    assistant: str,
    source: str,
    benchmark: str,
    split: str,
    task_mode: str,
    reasoning_budget: str,
    extra_metadata: Dict[str, object] | None = None,
) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        "benchmark": benchmark,
        "benchmark_split": split,
        "task_mode": task_mode,
        "reasoning_budget": reasoning_budget,
        "creativity_level": "precise",
        "knowledge_recency": "evergreen",
        "source_quality": "benchmark_train",
        "research_tags": ["v39", benchmark, split],
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "user": _normalize_text(user),
        "assistant": str(assistant).strip(),
        "source": source,
        "metadata": metadata,
    }


def _build_arc_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for item in _sample_rows(ds, limit, seed):
        labels = [str(label).upper() for label in item["choices"]["label"]]
        texts = [_normalize_text(text) for text in item["choices"]["text"]]
        choices = dict(zip(labels, texts))
        answer_key = str(item["answerKey"]).upper()
        options = _format_options(choices)
        user = (
            "Answer the multiple-choice science question. End with 'Final answer: <letter>'.\n"
            f"Question: {_normalize_text(item['question'])}\n{options}"
        )
        assistant = f"Final answer: {answer_key}. {choices.get(answer_key, '')}"
        rows.append(
            _row(
                user=user,
                assistant=assistant,
                source="v39_arc_train_direct",
                benchmark="arc_challenge",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
            )
        )
        wrong = rng.choice([label for label in labels if label != answer_key])
        verify_user = (
            "Verify the draft answer to the multiple-choice science question. If it is wrong, correct it. "
            "Return only the corrected final answer in the same format.\n"
            f"Question: {_normalize_text(item['question'])}\n{options}\n"
            f"Draft answer: Final answer: {wrong}. {choices.get(wrong, '')}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=assistant,
                source="v39_arc_train_verify",
                benchmark="arc_challenge",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
                extra_metadata={"repair_kind": "multiple_choice_verification"},
            )
        )
    return rows


def _build_boolq_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("google/boolq", split="train")
    rows: List[Dict[str, object]] = []
    for item in _sample_rows(ds, limit, seed):
        answer = "yes" if bool(item["answer"]) else "no"
        draft = "no" if answer == "yes" else "yes"
        passage = _normalize_text(item["passage"])
        question = _normalize_text(item["question"])
        user = (
            "Read the passage and answer the yes/no question. End with 'Final answer: yes' or "
            "'Final answer: no'.\n"
            f"Passage: {passage}\nQuestion: {question}"
        )
        assistant = f"Final answer: {answer}"
        rows.append(
            _row(
                user=user,
                assistant=assistant,
                source="v39_boolq_train_direct",
                benchmark="boolq",
                split="train",
                task_mode="knowledge",
                reasoning_budget="medium",
            )
        )
        verify_user = (
            "Verify the draft answer to the yes/no question. If it is wrong, correct it. "
            "Return only the corrected final answer.\n"
            f"Passage: {passage}\nQuestion: {question}\nDraft answer: Final answer: {draft}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=assistant,
                source="v39_boolq_train_verify",
                benchmark="boolq",
                split="train",
                task_mode="knowledge",
                reasoning_budget="medium",
                extra_metadata={"repair_kind": "boolq_verification"},
            )
        )
    return rows


def _build_gsm8k_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("gsm8k", "main", split="train")
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for item in _sample_rows(ds, limit, seed):
        question = _normalize_text(item["question"])
        cleaned_solution = _clean_gsm8k_solution(item["answer"])
        answer = _extract_gsm8k_answer(item["answer"])
        user = (
            "Solve the math word problem carefully. End with 'Final answer: <number>'.\n"
            f"Question: {question}"
        )
        rows.append(
            _row(
                user=user,
                assistant=cleaned_solution,
                source="v39_gsm8k_train_direct",
                benchmark="gsm8k",
                split="train",
                task_mode="reasoning",
                reasoning_budget="deep",
            )
        )
        try:
            numeric = float(answer)
            delta = rng.choice([1, 2, 3, 5, 10]) * (-1 if rng.random() < 0.5 else 1)
            wrong = str(int(numeric + delta) if numeric.is_integer() else numeric + delta)
        except Exception:
            wrong = answer + "0"
        verify_user = (
            "Verify the draft solution to the math problem. Fix any reasoning or arithmetic mistakes and return the corrected answer. "
            "End with 'Final answer: <number>'.\n"
            f"Question: {question}\nDraft answer: Final answer: {wrong}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=f"Final answer: {answer}",
                source="v39_gsm8k_train_verify",
                benchmark="gsm8k",
                split="train",
                task_mode="reasoning",
                reasoning_budget="deep",
                extra_metadata={"repair_kind": "math_verification"},
            )
        )
    return rows


def _build_hellaswag_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("Rowan/hellaswag", split="train")
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for item in _sample_rows(ds, limit, seed):
        endings = [_normalize_text(text) for text in item["endings"]]
        labels = _choice_labels(len(endings))
        choices = dict(zip(labels, endings))
        answer_key = labels[int(item["label"])]
        activity = _normalize_text(item.get("activity_label", ""))
        context = _normalize_text(item.get("ctx", ""))
        options = _format_options(choices)
        user = (
            "Choose the most plausible next sentence for the situation. End with 'Final answer: <letter>'.\n"
            f"Activity: {activity}\nContext: {context}\n{options}"
        )
        assistant = f"Final answer: {answer_key}. {choices.get(answer_key, '')}"
        rows.append(
            _row(
                user=user,
                assistant=assistant,
                source="v39_hellaswag_train_direct",
                benchmark="hellaswag",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
            )
        )
        wrong = rng.choice([label for label in labels if label != answer_key])
        verify_user = (
            "Verify the draft continuation choice. If it is wrong, correct it and return only the corrected final answer.\n"
            f"Activity: {activity}\nContext: {context}\n{options}\n"
            f"Draft answer: Final answer: {wrong}. {choices.get(wrong, '')}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=assistant,
                source="v39_hellaswag_train_verify",
                benchmark="hellaswag",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
                extra_metadata={"repair_kind": "commonsense_verification"},
            )
        )
    return rows


def _build_mmlu_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    sampled = _sample_stratified(ds, limit, seed, "subject")
    for item in sampled:
        choices_list = [_normalize_text(text) for text in item["choices"]]
        labels = _choice_labels(len(choices_list))
        choices = dict(zip(labels, choices_list))
        answer_key = labels[int(item["answer"])]
        subject = _normalize_text(str(item["subject"]).replace("_", " "))
        options = _format_options(choices)
        user = (
            "Answer the multiple-choice knowledge question. End with 'Final answer: <letter>'.\n"
            f"Subject: {subject}\nQuestion: {_normalize_text(item['question'])}\n{options}"
        )
        assistant = f"Final answer: {answer_key}. {choices.get(answer_key, '')}"
        rows.append(
            _row(
                user=user,
                assistant=assistant,
                source="v39_mmlu_aux_train_direct",
                benchmark="mmlu",
                split="auxiliary_train",
                task_mode="knowledge",
                reasoning_budget="deep",
                extra_metadata={"subject": subject},
            )
        )
        wrong = rng.choice([label for label in labels if label != answer_key])
        verify_user = (
            "Verify the draft answer to the multiple-choice knowledge question. If it is wrong, correct it. "
            "Return only the corrected final answer.\n"
            f"Subject: {subject}\nQuestion: {_normalize_text(item['question'])}\n{options}\n"
            f"Draft answer: Final answer: {wrong}. {choices.get(wrong, '')}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=assistant,
                source="v39_mmlu_aux_train_verify",
                benchmark="mmlu",
                split="auxiliary_train",
                task_mode="knowledge",
                reasoning_budget="deep",
                extra_metadata={"repair_kind": "knowledge_verification", "subject": subject},
            )
        )
    return rows


def _build_piqa_rows(limit: int, seed: int) -> List[Dict[str, object]]:
    ds = load_dataset("nthngdy/piqa", split="train")
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for item in _sample_rows(ds, limit, seed):
        choices = {
            "A": _normalize_text(item["sol1"]),
            "B": _normalize_text(item["sol2"]),
        }
        answer_key = "A" if int(item["label"]) == 0 else "B"
        options = _format_options(choices)
        goal = _normalize_text(item["goal"])
        user = (
            "Choose the better physical commonsense solution. End with 'Final answer: <letter>'.\n"
            f"Goal: {goal}\n{options}"
        )
        assistant = f"Final answer: {answer_key}. {choices.get(answer_key, '')}"
        rows.append(
            _row(
                user=user,
                assistant=assistant,
                source="v39_piqa_train_direct",
                benchmark="piqa",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
            )
        )
        wrong = "B" if answer_key == "A" else "A"
        verify_user = (
            "Verify the draft commonsense solution. If it is wrong, correct it and return only the corrected final answer.\n"
            f"Goal: {goal}\n{options}\nDraft answer: Final answer: {wrong}. {choices.get(wrong, '')}"
        )
        rows.append(
            _row(
                user=verify_user,
                assistant=assistant,
                source="v39_piqa_train_verify",
                benchmark="piqa",
                split="train",
                task_mode="reasoning",
                reasoning_budget="medium",
                extra_metadata={"repair_kind": "commonsense_verification"},
            )
        )
    return rows


def build_dataset(args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    rows.extend(_build_arc_rows(limit=int(args.arc_count), seed=int(args.seed) + 11))
    rows.extend(_build_boolq_rows(limit=int(args.boolq_count), seed=int(args.seed) + 17))
    rows.extend(_build_gsm8k_rows(limit=int(args.gsm8k_count), seed=int(args.seed) + 23))
    rows.extend(_build_hellaswag_rows(limit=int(args.hellaswag_count), seed=int(args.seed) + 29))
    rows.extend(_build_mmlu_rows(limit=int(args.mmlu_count), seed=int(args.seed) + 31))
    rows.extend(_build_piqa_rows(limit=int(args.piqa_count), seed=int(args.seed) + 37))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the v39 reasoning and benchmark-train expansion dataset.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary_out", default="")
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--arc_count", type=int, default=1119)
    parser.add_argument("--boolq_count", type=int, default=3000)
    parser.add_argument("--gsm8k_count", type=int, default=2500)
    parser.add_argument("--hellaswag_count", type=int, default=4000)
    parser.add_argument("--mmlu_count", type=int, default=5000)
    parser.add_argument("--piqa_count", type=int, default=4000)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    summary_out = Path(args.summary_out).resolve() if str(args.summary_out).strip() else output.with_suffix(".summary.json")
    rows = build_dataset(args)
    write_jsonl(output, rows)

    source_counts = Counter(str(row["source"]) for row in rows)
    benchmark_counts = Counter(str(row["metadata"].get("benchmark", "unknown")) for row in rows)
    repair_counts = Counter(str(row["metadata"].get("repair_kind", "direct")) for row in rows)
    summary = {
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
        "output": str(output),
        "row_count": len(rows),
        "source_counts": dict(source_counts),
        "benchmark_counts": dict(benchmark_counts),
        "repair_counts": dict(repair_counts),
        "note": (
            "This v39 benchmix file adds train-only reasoning and commonsense rows from ARC-Challenge, BoolQ, "
            "GSM8K, HellaSwag, MMLU auxiliary_train, and PIQA, plus paired verification/correction rows."
        ),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
