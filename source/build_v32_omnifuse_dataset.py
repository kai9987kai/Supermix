#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from qwen_supermix_pipeline import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SupermixTeacher,
    _load_adapter_with_compat,
    _set_model_use_cache,
    token_f1,
)


USER_KEYS = ("user", "prompt", "input", "question", "instruction")
ASSISTANT_KEYS = ("assistant", "response", "output", "answer", "completion", "target")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
SYMBOL_ONLY_RE = re.compile(r"^[-=+*/%0-9.,:;(){}\[\]\s]+$")
URL_RE = re.compile(r"https?://\S+")

DEFAULT_SYSTEM_PROMPT = (
    "You are Supermix Qwen v32, a practical local assistant. "
    "Answer directly, stay grounded in the user's request, and do not narrate hidden reasoning."
)
PRESET_HINT = "Give a direct answer with only the detail needed to be useful."
PROMPT_STYLE_SUFFIXES = {
    "photo": "photorealistic, natural lighting, high detail, sharp focus",
    "cinematic": "cinematic composition, dramatic lighting, rich atmosphere, high detail",
    "illustration": "detailed illustration, clean shapes, polished color palette",
    "anime": "anime style, expressive character design, crisp line art, vibrant color",
}
IMAGE_STYLE_ORDER = ("photo", "cinematic", "illustration", "anime")
IMAGE_STYLE_HINTS = {
    "photo": ("photo", "realistic", "photograph", "product", "portrait", "camera"),
    "cinematic": ("scene", "movie", "cinematic", "dramatic", "neon", "storm", "space", "night"),
    "illustration": ("diagram", "poster", "concept", "cover", "book", "editorial", "explain"),
    "anime": ("anime", "manga", "stylized", "character"),
}
TEACHER_PRIOR = {
    "qwen_base": -0.010,
    "qwen_v28": 0.012,
    "qwen_v29": -0.020,
    "qwen_v30": -0.030,
    "champion_v28": 0.028,
    "lite_v30": 0.040,
    "hybrid_v31": 0.085,
}


@dataclass(frozen=True)
class PromptRow:
    user: str
    assistant: str
    source: str
    metadata: Dict[str, object]


class QwenGenerator:
    def __init__(self, *, base_model: str, adapter_dir: Optional[Path], device: str) -> None:
        self.device = torch.device(device)
        self.base_model = base_model
        self.adapter_dir = adapter_dir
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float32,
            local_files_only=True,
            low_cpu_mem_usage=False,
        ).to(self.device)
        _set_model_use_cache(model, enabled=True)
        if adapter_dir is not None and adapter_dir.exists():
            model = _load_adapter_with_compat(
                model=model,
                adapter_dir=adapter_dir,
                device=self.device,
                is_trainable=False,
                merge_for_inference=True,
            )
        else:
            model = model.to(self.device)
        _set_model_use_cache(model, enabled=True)
        self.model = model.eval()

    def _build_prompt(self, user_text: str) -> str:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "system", "content": PRESET_HINT},
            {"role": "user", "content": user_text},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        lines = []
        for message in messages:
            lines.append(f"{str(message.get('role', 'user')).upper()}: {str(message.get('content', ''))}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)

    @torch.no_grad()
    def generate(self, user_text: str, max_new_tokens: int) -> str:
        prompt = self._build_prompt(user_text)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max(24, int(max_new_tokens)),
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=1.08,
            no_repeat_ngram_size=4,
            use_cache=True,
        )
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return normalize_response(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def coerce_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def normalize_whitespace(text: str) -> str:
    return " ".join(coerce_text(text).split())


def normalize_response(text: str) -> str:
    out = normalize_whitespace(text)
    if not out:
        return ""
    for prefix in ("assistant:", "bot:", "answer:", "response:"):
        if out.lower().startswith(prefix):
            out = normalize_whitespace(out[len(prefix) :])
    out = out.replace("<|assistant|>", "").replace("</s>", "").strip()
    return normalize_whitespace(out)


def extract_pair(record: Dict[str, object]) -> Optional[PromptRow]:
    user_text = ""
    for key in USER_KEYS:
        if key in record:
            user_text = coerce_text(record.get(key))
            if user_text:
                break

    assistant_text = ""
    for key in ASSISTANT_KEYS:
        if key in record:
            assistant_text = coerce_text(record.get(key))
            if assistant_text:
                break

    if not user_text or not assistant_text:
        return None

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return PromptRow(
        user=normalize_whitespace(user_text),
        assistant=normalize_response(assistant_text),
        source=coerce_text(record.get("source")) or "unknown",
        metadata={str(k): v for k, v in metadata.items()},
    )


def load_prompt_rows(paths: Sequence[Path], exclude_source_prefix: str = "") -> List[PromptRow]:
    rows: List[PromptRow] = []
    source_prefix = coerce_text(exclude_source_prefix).lower()
    seen_users: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                pair = extract_pair(json.loads(raw))
                if pair is None:
                    continue
                if source_prefix and pair.source.lower().startswith(source_prefix):
                    continue
                key = pair.user.lower()
                if key in seen_users:
                    continue
                seen_users.add(key)
                rows.append(pair)
    return rows


def merge_unique_rows(*row_groups: Sequence[PromptRow]) -> List[PromptRow]:
    merged: List[PromptRow] = []
    seen_users: set[str] = set()
    for group in row_groups:
        for row in group:
            key = row.user.lower()
            if key in seen_users:
                continue
            seen_users.add(key)
            merged.append(row)
    return merged


def split_holdout_rows(rows: Sequence[PromptRow], *, eval_fraction: float, eval_cap: int, seed: int) -> Tuple[List[PromptRow], List[PromptRow]]:
    if not rows:
        return [], []
    rng = random.Random(seed)
    groups: Dict[str, List[PromptRow]] = defaultdict(list)
    for row in rows:
        groups[row.source].append(row)
    holdout: List[PromptRow] = []
    train: List[PromptRow] = []
    for source, items in groups.items():
        shuffled = list(items)
        rng.shuffle(shuffled)
        if len(shuffled) < 6:
            train.extend(shuffled)
            continue
        quota = max(1, int(round(len(shuffled) * max(0.0, eval_fraction))))
        quota = min(quota, max(1, len(shuffled) // 4))
        holdout.extend(shuffled[:quota])
        train.extend(shuffled[quota:])
    if eval_cap > 0 and len(holdout) > eval_cap:
        rng.shuffle(holdout)
        keep = holdout[:eval_cap]
        keep_users = {row.user.lower() for row in keep}
        extra = [row for row in holdout if row.user.lower() not in keep_users]
        train.extend(extra)
        holdout = keep
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def stratified_sample(rows: Sequence[PromptRow], sample_size: int, seed: int) -> List[PromptRow]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    groups: Dict[str, List[PromptRow]] = defaultdict(list)
    for row in rows:
        groups[row.source].append(row)
    for items in groups.values():
        rng.shuffle(items)
    total = len(rows)
    chosen: List[PromptRow] = []
    leftovers: List[PromptRow] = []
    for source in sorted(groups):
        items = groups[source]
        quota = max(1, round(sample_size * (len(items) / max(1, total))))
        quota = min(len(items), int(quota))
        chosen.extend(items[:quota])
        leftovers.extend(items[quota:])
    if len(chosen) > sample_size:
        rng.shuffle(chosen)
        chosen = chosen[:sample_size]
    elif len(chosen) < sample_size:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: sample_size - len(chosen)])
    rng.shuffle(chosen)
    return chosen


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def candidate_penalty(reference: str, candidate: str) -> float:
    candidate = normalize_response(candidate)
    reference = normalize_response(reference)
    if not candidate:
        return 1.0

    words = candidate.split()
    ref_words = reference.split()
    visible = sum(1 for ch in candidate if not ch.isspace())
    alpha = sum(1 for ch in candidate if ch.isalpha())
    alpha_ratio = (alpha / visible) if visible else 0.0
    tokens = [tok.lower() for tok in WORD_RE.findall(candidate)]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    penalty = 0.0
    if len(words) < 3:
        penalty += 0.18
    if len(words) > 140:
        penalty += 0.05
    if SYMBOL_ONLY_RE.fullmatch(candidate):
        penalty += 0.45
    if alpha_ratio < 0.42:
        penalty += 0.12
    if tokens and unique_ratio < 0.40:
        penalty += 0.08
    if URL_RE.search(candidate):
        penalty += 0.05
    if reference and len(ref_words) <= 4 and len(words) > 10:
        penalty += 0.24
    if ref_words and len(words) > max(18, len(ref_words) * 4):
        penalty += 0.08
    if any(fragment in candidate for fragment in ("=0", "-3", "-)", "-.", "The elegant answer is")):
        penalty += 0.18
    return penalty


def score_candidate(reference: str, candidate: str, teacher: str) -> Dict[str, float]:
    candidate = normalize_response(candidate)
    reference = normalize_response(reference)
    if not candidate:
        return {
            "token_f1": 0.0,
            "char_similarity": 0.0,
            "penalty": 1.0,
            "composite": -1.0,
        }
    f1 = float(token_f1(reference, candidate))
    similarity = float(SequenceMatcher(None, reference.lower(), candidate.lower()).ratio())
    penalty = float(candidate_penalty(reference, candidate))
    word_count = len(candidate.split())
    ref_words = len(reference.split())
    length_bonus = 0.02 if 4 <= word_count <= 96 else 0.0
    if 0 < ref_words <= 12 and abs(word_count - ref_words) <= max(2, int(ref_words * 0.6)):
        length_bonus += 0.03
    composite = 0.62 * f1 + 0.33 * similarity + TEACHER_PRIOR.get(teacher, 0.0) + length_bonus - penalty
    return {
        "token_f1": f1,
        "char_similarity": similarity,
        "penalty": penalty,
        "composite": float(composite),
    }


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_qwen_outputs(
    prompts: Sequence[PromptRow],
    *,
    base_model: str,
    adapter_dir: Optional[Path],
    device: str,
    label: str,
    max_new_tokens: int,
    log_every: int,
) -> List[str]:
    print(f"[v32-distill] loading {label} qwen engine...")
    engine = QwenGenerator(base_model=base_model, adapter_dir=adapter_dir, device=device)
    outputs: List[str] = []
    try:
        for idx, row in enumerate(prompts, start=1):
            outputs.append(engine.generate(row.user, max_new_tokens=max_new_tokens))
            if log_every > 0 and idx % log_every == 0:
                print(f"[v32-distill] {label} generated {idx}/{len(prompts)}")
    finally:
        del engine
        release_cuda_memory()
    return outputs


def generate_champion_outputs(
    prompts: Sequence[PromptRow],
    *,
    weights_path: str,
    meta_path: str,
    device: str,
    label: str,
    log_every: int,
) -> List[str]:
    print(f"[v32-distill] loading {label} champion teacher...")
    teacher = SupermixTeacher(weights_path=weights_path, meta_path=meta_path, device=device)
    outputs: List[str] = []
    try:
        for idx, row in enumerate(prompts, start=1):
            outputs.append(normalize_response(teacher.generate(row.user)))
            if log_every > 0 and idx % log_every == 0:
                print(f"[v32-distill] {label} generated {idx}/{len(prompts)}")
    finally:
        del teacher
        release_cuda_memory()
    return outputs


def pick_image_style(text: str) -> str:
    lowered = coerce_text(text).lower()
    for style, hints in IMAGE_STYLE_HINTS.items():
        if any(hint in lowered for hint in hints):
            return style
    return IMAGE_STYLE_ORDER[hash(lowered) % len(IMAGE_STYLE_ORDER)] if lowered else "cinematic"


def make_image_prompt(text: str, style: str) -> str:
    cleaned = normalize_response(URL_RE.sub("", text))
    cleaned = cleaned.strip(" .,:;")
    if not cleaned:
        cleaned = "interesting subject"
    if len(cleaned.split()) > 28:
        cleaned = " ".join(cleaned.split()[:28])
    if style == "photo":
        base = f"{cleaned}, professional product photo"
    elif style == "illustration":
        base = f"editorial illustration about {cleaned}"
    elif style == "anime":
        base = f"{cleaned}, anime character concept"
    else:
        base = f"{cleaned}, cinematic scene"
    suffix = PROMPT_STYLE_SUFFIXES.get(style, "")
    final = ", ".join(part for part in (base, suffix, "high detail, coherent composition") if part)
    return normalize_whitespace(final)


def build_image_variant_rows(rows: Sequence[PromptRow], *, limit: int, seed: int) -> List[Dict[str, object]]:
    if limit <= 0 or not rows:
        return []
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    selected = selected[: min(limit, len(selected))]
    output: List[Dict[str, object]] = []
    for row in selected:
        topic = row.user if 4 <= len(row.user.split()) <= 20 else row.assistant
        style = pick_image_style(topic)
        prompt = make_image_prompt(topic, style)
        output.append(
            {
                "user": f"Create a concise image-generation prompt for this idea: {topic}",
                "assistant": prompt,
                "source": "v32_image_variant_aux",
                "metadata": {
                    "derived_from": row.source,
                    "style": style,
                    "teacher_choice": "image_variant_logic",
                },
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a v32 omnifuse dataset from all available teachers.")
    parser.add_argument("--prompt_data", nargs="+", required=True, help="Input JSONL prompt/answer files.")
    parser.add_argument("--eval_data", nargs="*", default=(), help="Optional eval-only JSONL files.")
    parser.add_argument("--distill_output", required=True, help="Output distilled JSONL file.")
    parser.add_argument("--stage2_train_output", required=True, help="Output stage2 train mix JSONL.")
    parser.add_argument("--stage2_eval_output", required=True, help="Output stage2 eval mix JSONL.")
    parser.add_argument("--summary_out", required=True, help="Output summary JSON.")
    parser.add_argument("--audit_out", default=None, help="Optional per-row teacher audit JSONL.")
    parser.add_argument("--base_model", required=True, help="Local Qwen base model directory.")
    parser.add_argument("--qwen_v28_adapter", required=True)
    parser.add_argument("--qwen_v29_adapter", required=True)
    parser.add_argument("--qwen_v30_adapter", required=True)
    parser.add_argument("--champion_v28_weights", default="")
    parser.add_argument("--champion_v28_meta", default="")
    parser.add_argument("--lite_weights", required=True)
    parser.add_argument("--lite_meta", required=True)
    parser.add_argument("--hybrid_weights", required=True)
    parser.add_argument("--hybrid_meta", required=True)
    parser.add_argument("--sample_size", type=int, default=900)
    parser.add_argument("--image_aux_limit", type=int, default=128)
    parser.add_argument("--eval_fraction", type=float, default=0.12)
    parser.add_argument("--eval_cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--log_every", type=int, default=32)
    parser.add_argument("--min_teacher_score", type=float, default=0.12)
    args = parser.parse_args()

    prompt_paths = [Path(path).resolve() for path in args.prompt_data]
    eval_paths = [Path(path).resolve() for path in args.eval_data]
    distill_output = Path(args.distill_output).resolve()
    stage2_train_output = Path(args.stage2_train_output).resolve()
    stage2_eval_output = Path(args.stage2_eval_output).resolve()
    summary_out = Path(args.summary_out).resolve()
    audit_out = Path(args.audit_out).resolve() if args.audit_out else None

    raw_rows = load_prompt_rows(prompt_paths)
    extra_eval_rows = load_prompt_rows(eval_paths)
    if not raw_rows:
        raise SystemExit("No eligible prompt rows were loaded.")

    train_rows, heldout_rows = split_holdout_rows(
        raw_rows,
        eval_fraction=float(args.eval_fraction),
        eval_cap=int(args.eval_cap),
        seed=int(args.seed),
    )
    eval_rows = merge_unique_rows(extra_eval_rows, heldout_rows)
    eval_user_keys = {row.user.lower() for row in eval_rows}
    train_rows = [row for row in train_rows if row.user.lower() not in eval_user_keys]
    distill_rows = stratified_sample(train_rows, sample_size=int(args.sample_size), seed=int(args.seed))

    print(
        f"[v32-distill] loaded={len(raw_rows)} train={len(train_rows)} eval={len(eval_rows)} "
        f"sampled={len(distill_rows)}"
    )

    qwen_base_outputs = generate_qwen_outputs(
        distill_rows,
        base_model=str(Path(args.base_model).resolve()),
        adapter_dir=None,
        device=args.device,
        label="qwen_base",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    qwen_v28_outputs = generate_qwen_outputs(
        distill_rows,
        base_model=str(Path(args.base_model).resolve()),
        adapter_dir=Path(args.qwen_v28_adapter).resolve(),
        device=args.device,
        label="qwen_v28",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    qwen_v29_outputs = generate_qwen_outputs(
        distill_rows,
        base_model=str(Path(args.base_model).resolve()),
        adapter_dir=Path(args.qwen_v29_adapter).resolve(),
        device=args.device,
        label="qwen_v29",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    qwen_v30_outputs = generate_qwen_outputs(
        distill_rows,
        base_model=str(Path(args.base_model).resolve()),
        adapter_dir=Path(args.qwen_v30_adapter).resolve(),
        device=args.device,
        label="qwen_v30",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    lite_outputs = generate_champion_outputs(
        distill_rows,
        weights_path=str(Path(args.lite_weights).resolve()),
        meta_path=str(Path(args.lite_meta).resolve()),
        device=args.device,
        label="lite_v30",
        log_every=int(args.log_every),
    )
    hybrid_outputs = generate_champion_outputs(
        distill_rows,
        weights_path=str(Path(args.hybrid_weights).resolve()),
        meta_path=str(Path(args.hybrid_meta).resolve()),
        device=args.device,
        label="hybrid_v31",
        log_every=int(args.log_every),
    )

    classic_outputs: Optional[List[str]] = None
    classic_weights = coerce_text(args.champion_v28_weights)
    classic_meta = coerce_text(args.champion_v28_meta)
    if classic_weights and classic_meta and Path(classic_weights).exists() and Path(classic_meta).exists():
        classic_outputs = generate_champion_outputs(
            distill_rows,
            weights_path=str(Path(classic_weights).resolve()),
            meta_path=str(Path(classic_meta).resolve()),
            device=args.device,
            label="champion_v28",
            log_every=int(args.log_every),
        )

    distilled_json_rows: List[Dict[str, object]] = []
    audit_json_rows: List[Dict[str, object]] = []
    teacher_choice_counts: Counter[str] = Counter()
    prompt_source_counts: Counter[str] = Counter()
    teacher_mean_composite: Counter[str] = Counter()
    teacher_mean_count: Counter[str] = Counter()

    iterator = zip(
        distill_rows,
        qwen_base_outputs,
        qwen_v28_outputs,
        qwen_v29_outputs,
        qwen_v30_outputs,
        lite_outputs,
        hybrid_outputs,
        classic_outputs if classic_outputs is not None else [None] * len(distill_rows),
    )
    for row, base_text, v28_text, v29_text, v30_text, lite_text, hybrid_text, classic_text in iterator:
        candidates = {
            "qwen_base": base_text,
            "qwen_v28": v28_text,
            "qwen_v29": v29_text,
            "qwen_v30": v30_text,
            "lite_v30": lite_text,
            "hybrid_v31": hybrid_text,
        }
        if classic_text:
            candidates["champion_v28"] = classic_text

        scored: Dict[str, Dict[str, float]] = {}
        for teacher_name, candidate in candidates.items():
            metrics = score_candidate(reference=row.assistant, candidate=str(candidate), teacher=teacher_name)
            metrics["text"] = normalize_response(str(candidate))
            scored[teacher_name] = metrics

        best_teacher, best_metrics = max(scored.items(), key=lambda item: float(item[1]["composite"]))
        use_reference = (
            float(best_metrics["composite"]) < float(args.min_teacher_score)
            or (
                float(best_metrics["token_f1"]) < 0.06
                and float(best_metrics["char_similarity"]) < 0.16
            )
        )
        if use_reference:
            chosen_text = row.assistant
            chosen_from = "reference"
        else:
            chosen_text = normalize_response(str(best_metrics["text"]))
            chosen_from = best_teacher
            teacher_mean_composite[best_teacher] += float(best_metrics["composite"])
            teacher_mean_count[best_teacher] += 1

        teacher_choice_counts[chosen_from] += 1
        prompt_source_counts[row.source] += 1
        distilled_json_rows.append(
            {
                "user": row.user,
                "assistant": chosen_text,
                "source": f"v32_omnifuse_{chosen_from}",
                "metadata": {
                    "reference_source": row.source,
                    "teacher_choice": chosen_from,
                },
            }
        )
        audit_json_rows.append(
            {
                "user": row.user,
                "reference": row.assistant,
                "reference_source": row.source,
                "chosen_from": chosen_from,
                "chosen_text": chosen_text,
                "scores": scored,
            }
        )

    image_aux_rows = build_image_variant_rows(train_rows, limit=int(args.image_aux_limit), seed=int(args.seed) + 13)
    stage2_train_rows: List[Dict[str, object]] = []
    seen_train = set()
    for row in train_rows:
        item = {
            "user": row.user,
            "assistant": row.assistant,
            "source": row.source,
            "metadata": row.metadata,
        }
        key = (item["user"].lower(), item["assistant"].lower(), item["source"])
        if key not in seen_train:
            stage2_train_rows.append(item)
            seen_train.add(key)
    for item in distilled_json_rows + image_aux_rows:
        key = (str(item["user"]).lower(), str(item["assistant"]).lower(), str(item["source"]))
        if key not in seen_train:
            stage2_train_rows.append(item)
            seen_train.add(key)

    stage2_eval_rows: List[Dict[str, object]] = []
    seen_eval = set()
    for row in eval_rows:
        item = {
            "user": row.user,
            "assistant": row.assistant,
            "source": row.source,
            "metadata": row.metadata,
        }
        key = (item["user"].lower(), item["assistant"].lower(), item["source"])
        if key not in seen_eval:
            stage2_eval_rows.append(item)
            seen_eval.add(key)

    write_jsonl(distill_output, distilled_json_rows)
    write_jsonl(stage2_train_output, stage2_train_rows)
    write_jsonl(stage2_eval_output, stage2_eval_rows)
    if audit_out is not None:
        write_jsonl(audit_out, audit_json_rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_files": [str(path) for path in prompt_paths],
        "eval_files": [str(path) for path in eval_paths],
        "rows_loaded": len(raw_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "distill_rows": len(distilled_json_rows),
        "stage2_train_rows": len(stage2_train_rows),
        "stage2_eval_rows": len(stage2_eval_rows),
        "image_aux_rows": len(image_aux_rows),
        "teacher_selection_counts": dict(teacher_choice_counts),
        "prompt_source_counts": dict(prompt_source_counts),
        "teacher_mean_composite": {
            teacher: float(teacher_mean_composite[teacher] / max(1, teacher_mean_count[teacher]))
            for teacher in teacher_mean_count
        },
        "teacher_prior": dict(TEACHER_PRIOR),
        "student_note": (
            "The image-generation variant is represented indirectly through image-prompt auxiliary rows. "
            "The diffusion renderer itself is not distilled into the text student."
        ),
        "device": str(args.device),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[v32-distill] wrote distill data to {distill_output}")
    print(f"[v32-distill] wrote stage2 train data to {stage2_train_output}")
    print(f"[v32-distill] wrote stage2 eval data to {stage2_eval_output}")
    print(f"[v32-distill] teacher choices: {dict(teacher_choice_counts)}")
    print(f"[v32-distill] wrote summary to {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
