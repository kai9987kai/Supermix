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
TEACHER_PRIOR = {"tuned": 0.055, "base": 0.020, "lite": 0.000}
SYMBOL_ONLY_RE = re.compile(r"^[-=+*/%0-9.,:;(){}\[\]\s]+$")
WORD_RE = re.compile(r"[A-Za-z0-9']+")
DEFAULT_SYSTEM_PROMPT = (
    "You are Supermix Qwen v31, a practical local assistant. "
    "Answer directly, stay grounded in the user's request, and do not narrate hidden reasoning."
)
PRESET_HINT = "Give a direct answer with only the detail needed to be useful."


@dataclass(frozen=True)
class PromptRow:
    user: str
    assistant: str
    source: str


class QwenGenerator:
    def __init__(self, *, base_model: str, adapter_dir: Path, device: str) -> None:
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
        if adapter_dir.exists():
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
        return _normalize_response(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_whitespace(text: str) -> str:
    return " ".join(_coerce_text(text).split())


def _normalize_response(text: str) -> str:
    out = _normalize_whitespace(text)
    if not out:
        return ""
    for prefix in ("assistant:", "bot:", "answer:"):
        if out.lower().startswith(prefix):
            out = _normalize_whitespace(out[len(prefix) :])
    out = out.replace("<|assistant|>", "").replace("</s>", "").strip()
    return _normalize_whitespace(out)


def _extract_pair(record: Dict[str, object]) -> Optional[PromptRow]:
    user_text = ""
    for key in USER_KEYS:
        if key in record:
            user_text = _coerce_text(record.get(key))
            if user_text:
                break

    assistant_text = ""
    for key in ASSISTANT_KEYS:
        if key in record:
            assistant_text = _coerce_text(record.get(key))
            if assistant_text:
                break

    if not user_text or not assistant_text:
        return None

    return PromptRow(
        user=_normalize_whitespace(user_text),
        assistant=_normalize_response(assistant_text),
        source=_coerce_text(record.get("source")) or "unknown",
    )


def load_prompt_rows(paths: Sequence[Path], exclude_source_prefix: str) -> List[PromptRow]:
    rows: List[PromptRow] = []
    seen_users: set[str] = set()
    source_prefix = _coerce_text(exclude_source_prefix).lower()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                pair = _extract_pair(json.loads(raw))
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


def _candidate_penalty(text: str) -> float:
    candidate = _normalize_response(text)
    if not candidate:
        return 1.0

    words = candidate.split()
    visible = sum(1 for ch in candidate if not ch.isspace())
    alpha = sum(1 for ch in candidate if ch.isalpha())
    alpha_ratio = (alpha / visible) if visible else 0.0
    tokens = [tok.lower() for tok in WORD_RE.findall(candidate)]
    unique_ratio = (len(set(tokens)) / len(tokens)) if tokens else 0.0

    penalty = 0.0
    if len(words) < 5:
        penalty += 0.14
    if len(words) > 120:
        penalty += 0.03
    if SYMBOL_ONLY_RE.fullmatch(candidate):
        penalty += 0.45
    if alpha_ratio < 0.42:
        penalty += 0.10
    if tokens and unique_ratio < 0.42:
        penalty += 0.08
    if any(fragment in candidate for fragment in ("=0", "-3", "-)", "-.")):
        penalty += 0.12
    return penalty


def score_candidate(reference: str, candidate: str, teacher: str) -> Dict[str, float]:
    candidate = _normalize_response(candidate)
    reference = _normalize_response(reference)
    if not candidate:
        return {
            "token_f1": 0.0,
            "char_similarity": 0.0,
            "penalty": 1.0,
            "composite": -1.0,
        }

    f1 = float(token_f1(reference, candidate))
    similarity = float(SequenceMatcher(None, reference.lower(), candidate.lower()).ratio())
    penalty = float(_candidate_penalty(candidate))
    word_count = len(candidate.split())
    length_bonus = 0.02 if 8 <= word_count <= 80 else 0.0
    composite = 0.60 * f1 + 0.34 * similarity + TEACHER_PRIOR.get(teacher, 0.0) + length_bonus - penalty
    return {
        "token_f1": f1,
        "char_similarity": similarity,
        "penalty": penalty,
        "composite": float(composite),
    }


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def generate_qwen_outputs(
    prompts: Sequence[PromptRow],
    *,
    base_model: str,
    adapter_dir: Path,
    device: str,
    label: str,
    max_new_tokens: int,
    log_every: int,
) -> List[str]:
    print(f"[v31-distill] loading {label} qwen engine...")
    engine = QwenGenerator(base_model=base_model, adapter_dir=adapter_dir, device=device)
    outputs: List[str] = []
    try:
        for idx, row in enumerate(prompts, start=1):
            outputs.append(engine.generate(row.user, max_new_tokens=max_new_tokens))
            if log_every > 0 and idx % log_every == 0:
                print(f"[v31-distill] {label} generated {idx}/{len(prompts)}")
    finally:
        del engine
        _release_cuda_memory()
    return outputs


def generate_lite_outputs(
    prompts: Sequence[PromptRow],
    *,
    weights_path: str,
    meta_path: str,
    device: str,
    log_every: int,
) -> List[str]:
    print("[v31-distill] loading lite teacher...")
    teacher = SupermixTeacher(weights_path=weights_path, meta_path=meta_path, device=device)
    outputs: List[str] = []
    try:
        for idx, row in enumerate(prompts, start=1):
            outputs.append(_normalize_response(teacher.generate(row.user)))
            if log_every > 0 and idx % log_every == 0:
                print(f"[v31-distill] lite generated {idx}/{len(prompts)}")
    finally:
        del teacher
        _release_cuda_memory()
    return outputs


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a v31 three-teacher distilled dataset.")
    parser.add_argument("--prompt_data", nargs="+", required=True, help="Input JSONL prompt/answer data files.")
    parser.add_argument("--output_data", required=True, help="Output distilled JSONL file.")
    parser.add_argument("--summary_out", required=True, help="Output summary JSON.")
    parser.add_argument("--audit_out", default=None, help="Optional per-row teacher audit JSONL.")
    parser.add_argument("--base_model", required=True, help="Local Qwen base model directory.")
    parser.add_argument("--adapter_dir", required=True, help="Adapter directory for the tuned Qwen teacher.")
    parser.add_argument("--lite_weights", required=True, help="Lite student checkpoint path.")
    parser.add_argument("--lite_meta", required=True, help="Lite student metadata path.")
    parser.add_argument("--sample_size", type=int, default=240)
    parser.add_argument("--exclude_source_prefix", default="official_refresh_")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--log_every", type=int, default=16)
    parser.add_argument("--min_teacher_score", type=float, default=0.18)
    args = parser.parse_args()

    prompt_paths = [Path(path).resolve() for path in args.prompt_data]
    output_data = Path(args.output_data).resolve()
    summary_out = Path(args.summary_out).resolve()
    audit_out = Path(args.audit_out).resolve() if args.audit_out else None
    base_model = str(Path(args.base_model).resolve())
    adapter_dir = Path(args.adapter_dir).resolve()
    lite_weights = str(Path(args.lite_weights).resolve())
    lite_meta = str(Path(args.lite_meta).resolve())

    rows = load_prompt_rows(prompt_paths, exclude_source_prefix=args.exclude_source_prefix)
    if not rows:
        raise SystemExit("No eligible prompt rows were loaded.")

    sampled = stratified_sample(rows, sample_size=int(args.sample_size), seed=int(args.seed))
    print(
        f"[v31-distill] loaded={len(rows)} sampled={len(sampled)} "
        f"exclude_prefix={args.exclude_source_prefix or '-'}"
    )

    nonexistent_adapter = adapter_dir.parent / "__no_adapter__"
    base_outputs = generate_qwen_outputs(
        sampled,
        base_model=base_model,
        adapter_dir=nonexistent_adapter,
        device=args.device,
        label="base",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    tuned_outputs = generate_qwen_outputs(
        sampled,
        base_model=base_model,
        adapter_dir=adapter_dir,
        device=args.device,
        label="tuned",
        max_new_tokens=int(args.max_new_tokens),
        log_every=int(args.log_every),
    )
    lite_outputs = generate_lite_outputs(
        sampled,
        weights_path=lite_weights,
        meta_path=lite_meta,
        device=args.device,
        log_every=int(args.log_every),
    )

    distilled_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []
    chosen_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    composite_sums: Counter[str] = Counter()

    for row, base_text, tuned_text, lite_text in zip(sampled, base_outputs, tuned_outputs, lite_outputs):
        candidates = {
            "base": base_text,
            "tuned": tuned_text,
            "lite": lite_text,
        }
        scored: Dict[str, Dict[str, float]] = {}
        for teacher_name, candidate in candidates.items():
            metrics = score_candidate(reference=row.assistant, candidate=candidate, teacher=teacher_name)
            metrics["text"] = candidate
            scored[teacher_name] = metrics

        best_teacher, best_metrics = max(scored.items(), key=lambda item: float(item[1]["composite"]))
        use_reference = (
            float(best_metrics["composite"]) < float(args.min_teacher_score)
            or (
                float(best_metrics["token_f1"]) < 0.08
                and float(best_metrics["char_similarity"]) < 0.18
            )
        )

        if use_reference:
            assistant_text = row.assistant
            chosen_from = "reference"
        else:
            assistant_text = _normalize_response(str(best_metrics["text"]))
            chosen_from = best_teacher
            composite_sums[best_teacher] += float(best_metrics["composite"])

        chosen_counts[chosen_from] += 1
        source_counts[row.source] += 1

        distilled_rows.append(
            {
                "user": row.user,
                "assistant": assistant_text,
                "source": f"v31_hybrid_{chosen_from}",
                "metadata": {
                    "reference_source": row.source,
                    "teacher_choice": chosen_from,
                },
            }
        )
        audit_rows.append(
            {
                "user": row.user,
                "reference": row.assistant,
                "reference_source": row.source,
                "chosen_from": chosen_from,
                "chosen_text": assistant_text,
                "scores": scored,
            }
        )

    write_jsonl(output_data, distilled_rows)
    if audit_out is not None:
        write_jsonl(audit_out, audit_rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_files": [str(path) for path in prompt_paths],
        "rows_available": len(rows),
        "rows_sampled": len(sampled),
        "sample_size_requested": int(args.sample_size),
        "exclude_source_prefix": str(args.exclude_source_prefix),
        "output_data": str(output_data),
        "audit_out": str(audit_out) if audit_out is not None else None,
        "teacher_selection_counts": dict(chosen_counts),
        "prompt_source_counts": dict(source_counts),
        "teacher_mean_composite": {
            teacher: float(composite_sums[teacher] / max(1, chosen_counts[teacher]))
            for teacher in ("base", "tuned", "lite")
            if chosen_counts[teacher] > 0
        },
        "teacher_prior": dict(TEACHER_PRIOR),
        "device": str(args.device),
        "base_model": base_model,
        "adapter_dir": str(adapter_dir),
        "lite_weights": lite_weights,
        "lite_meta": lite_meta,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[v31-distill] wrote dataset to {output_data}")
    print(f"[v31-distill] teacher choices: {dict(chosen_counts)}")
    print(f"[v31-distill] wrote summary to {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
