#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

from build_v31_hybrid_distill_dataset import QwenGenerator, _normalize_response
from chat_pipeline import (
    build_context,
    choose_bucket_from_logits,
    pick_response,
    resolve_feature_mode,
    text_to_model_input,
)
from model_variants import build_model, detect_model_size_from_state_dict
from qwen_supermix_pipeline import _fast_cleanup_response_text, token_f1
from run import safe_load_state_dict

try:
    from datasets import load_dataset
except Exception as exc:  # pragma: no cover - resolved in the RunPod launcher
    raise RuntimeError("The `datasets` package is required. Install it before running this benchmark.") from exc


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r]+)", re.IGNORECASE)
YES_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class BenchmarkItem:
    benchmark: str
    prompt: str
    reference_text: str
    reference_extracted: str
    max_new_tokens: int
    scoring_data: Optional[Dict[str, object]] = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    kind: str
    weights_path: Optional[Path] = None
    meta_path: Optional[Path] = None
    adapter_dir: Optional[Path] = None


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _extract_gsm8k_answer(answer_text: str) -> str:
    text = str(answer_text)
    if "####" in text:
        text = text.split("####", 1)[1]
    matches = NUMBER_RE.findall(text.replace(",", ""))
    return matches[-1] if matches else _normalize_text(text)


def _extract_last_number(text: str) -> str:
    cleaned = _normalize_text(text)
    tagged = FINAL_ANSWER_RE.search(cleaned)
    if tagged:
        cleaned = tagged.group(1)
    matches = NUMBER_RE.findall(cleaned.replace(",", ""))
    return matches[-1] if matches else ""


def _extract_mc_choice(text: str, choices: Dict[str, str]) -> str:
    cleaned = _normalize_text(text)
    tagged = FINAL_ANSWER_RE.search(cleaned)
    if tagged:
        cleaned = tagged.group(1)
    labels = [str(label).strip().upper() for label in choices if str(label).strip()]
    if labels:
        pattern = r"\b(" + "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True)) + r")\b"
        match = re.search(pattern, cleaned.upper())
        if match:
            return match.group(1).upper()
    lowered = cleaned.lower()
    best = ""
    best_len = -1
    for label, choice_text in choices.items():
        option = _normalize_text(choice_text).lower()
        if option and option in lowered and len(option) > best_len:
            best = str(label).upper()
            best_len = len(option)
    return best


def _extract_yes_no(text: str) -> str:
    cleaned = _normalize_text(text)
    tagged = FINAL_ANSWER_RE.search(cleaned)
    if tagged:
        cleaned = tagged.group(1)
    match = YES_RE.search(cleaned)
    return match.group(1).lower() if match else ""


def _sample_rows(rows: Sequence[dict], sample_size: int, seed: int) -> List[dict]:
    items = list(rows)
    rng = random.Random(seed)
    rng.shuffle(items)
    if sample_size > 0:
        return items[: min(len(items), sample_size)]
    return items


def _try_load_dataset(candidates: Sequence[Tuple[str, Optional[str], str]]):
    errors: List[str] = []
    for path_name, config_name, split_name in candidates:
        try:
            if config_name is None:
                return load_dataset(path_name, split=split_name)
            return load_dataset(path_name, config_name, split=split_name)
        except Exception as exc:
            errors.append(f"{path_name}/{config_name or '-'}:{split_name} -> {exc}")
    raise RuntimeError("Could not load benchmark dataset. Attempts:\n" + "\n".join(errors))


def _choice_labels(count: int) -> List[str]:
    if count < 1 or count > len(OPTION_LABELS):
        raise ValueError(f"Unsupported number of choices: {count}")
    return list(OPTION_LABELS[:count])


def _format_options_block(choices: Dict[str, str]) -> str:
    return "\n".join(f"{label}. {text}" for label, text in choices.items())


def build_benchmark_items(sample_per_benchmark: int, seed: int) -> List[BenchmarkItem]:
    items: List[BenchmarkItem] = []

    gsm8k = _try_load_dataset(
        (
            ("gsm8k", "main", "test"),
            ("openai/gsm8k", "main", "test"),
        )
    )
    for row in _sample_rows(gsm8k, sample_per_benchmark, seed + 11):
        question = _normalize_text(row["question"])
        answer = _extract_gsm8k_answer(row["answer"])
        prompt = (
            "Solve the math word problem. Show only brief reasoning and end with "
            "'Final answer: <number>'.\n"
            f"Question: {question}"
        )
        items.append(
            BenchmarkItem(
                benchmark="gsm8k",
                prompt=prompt,
                reference_text=f"Final answer: {answer}",
                reference_extracted=answer,
                max_new_tokens=80,
            )
        )

    arc = _try_load_dataset(
        (
            ("allenai/ai2_arc", "ARC-Challenge", "test"),
            ("ai2_arc", "ARC-Challenge", "test"),
        )
    )
    for row in _sample_rows(arc, sample_per_benchmark, seed + 17):
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [_normalize_text(text) for text in row["choices"]["text"]]
        choices = dict(zip(labels, texts))
        options_block = _format_options_block(choices)
        answer_key = str(row["answerKey"]).upper()
        prompt = (
            "Answer the multiple-choice science question. End with 'Final answer: <letter>'.\n"
            f"Question: {_normalize_text(row['question'])}\n{options_block}"
        )
        items.append(
            BenchmarkItem(
                benchmark="arc_challenge",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    boolq = _try_load_dataset(
        (
            ("google/boolq", None, "validation"),
            ("boolq", None, "validation"),
        )
    )
    for row in _sample_rows(boolq, sample_per_benchmark, seed + 23):
        answer = "yes" if bool(row["answer"]) else "no"
        prompt = (
            "Read the passage and answer the yes/no question. End with 'Final answer: yes' or "
            "'Final answer: no'.\n"
            f"Passage: {_normalize_text(row['passage'])}\n"
            f"Question: {_normalize_text(row['question'])}"
        )
        items.append(
            BenchmarkItem(
                benchmark="boolq",
                prompt=prompt,
                reference_text=f"Final answer: {answer}",
                reference_extracted=answer,
                max_new_tokens=48,
            )
        )

    hellaswag = _try_load_dataset(
        (
            ("Rowan/hellaswag", None, "validation"),
            ("hellaswag", None, "validation"),
        )
    )
    for row in _sample_rows(hellaswag, sample_per_benchmark, seed + 29):
        endings = [_normalize_text(text) for text in row["endings"]]
        labels = _choice_labels(len(endings))
        choices = dict(zip(labels, endings))
        options_block = _format_options_block(choices)
        answer_index = int(row["label"])
        answer_key = labels[answer_index]
        activity = _normalize_text(row.get("activity_label", ""))
        context = _normalize_text(row.get("ctx", ""))
        prompt = (
            "Choose the most plausible next sentence for the situation. End with "
            "'Final answer: <letter>'.\n"
            f"Activity: {activity}\n"
            f"Context: {context}\n{options_block}"
        )
        items.append(
            BenchmarkItem(
                benchmark="hellaswag",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=48,
                scoring_data={"choices": choices},
            )
        )

    piqa = _try_load_dataset(
        (
            ("gimmaru/piqa", None, "validation"),
        )
    )
    for row in _sample_rows(piqa, sample_per_benchmark, seed + 31):
        choices = {
            "A": _normalize_text(row["sol1"]),
            "B": _normalize_text(row["sol2"]),
        }
        answer_key = "A" if int(row["label"]) == 0 else "B"
        options_block = _format_options_block(choices)
        prompt = (
            "Choose the better physical commonsense solution. End with 'Final answer: <letter>'.\n"
            f"Goal: {_normalize_text(row['goal'])}\n{options_block}"
        )
        items.append(
            BenchmarkItem(
                benchmark="piqa",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=48,
                scoring_data={"choices": choices},
            )
        )

    mmlu = _try_load_dataset(
        (
            ("cais/mmlu", "all", "test"),
        )
    )
    for row in _sample_rows(mmlu, sample_per_benchmark, seed + 37):
        choice_texts = [_normalize_text(text) for text in row["choices"]]
        labels = _choice_labels(len(choice_texts))
        choices = dict(zip(labels, choice_texts))
        answer_key = labels[int(row["answer"])]
        options_block = _format_options_block(choices)
        subject = _normalize_text(str(row.get("subject", "")).replace("_", " "))
        prompt = (
            "Answer the multiple-choice knowledge question. End with 'Final answer: <letter>'.\n"
            f"Subject: {subject}\n"
            f"Question: {_normalize_text(row['question'])}\n{options_block}"
        )
        items.append(
            BenchmarkItem(
                benchmark="mmlu",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=48,
                scoring_data={"choices": choices},
            )
        )

    return items


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_bucket_meta(meta_path: Path) -> Dict[str, object]:
    visited: set[str] = set()
    current = meta_path
    while True:
        resolved = str(current.resolve(strict=False))
        if resolved in visited:
            raise RuntimeError(f"Bucket-meta cycle detected at {resolved}")
        visited.add(resolved)
        meta = _load_json(current)
        buckets = meta.get("buckets")
        if isinstance(buckets, dict) and buckets:
            return meta
        for key in ("student_base_meta", "base_meta", "teacher_meta"):
            next_meta = _normalize_text(meta.get(key, ""))
            if next_meta:
                current = Path(next_meta)
                break
        else:
            raise RuntimeError(f"Could not resolve buckets for {meta_path}")


class ChampionBenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.device = torch.device(device)
        current_meta = _load_json(meta_path)
        bucket_meta = _resolve_bucket_meta(meta_path)

        self.feature_mode = resolve_feature_mode(str(current_meta.get("feature_mode", "legacy")), smarter_auto=True)
        self.buckets: Dict[int, List[Dict[str, object]]] = {}
        for key, value in bucket_meta.get("buckets", {}).items():
            try:
                label = int(key)
            except Exception:
                continue
            if isinstance(value, list):
                self.buckets[label] = value
        self.available_labels = sorted(self.buckets.keys()) or list(range(10))

        state_dict = safe_load_state_dict(str(weights_path))
        model_size = _normalize_text(current_meta.get("model_size")) or detect_model_size_from_state_dict(state_dict)
        model = build_model(
            model_size=model_size,
            expansion_dim=int(current_meta.get("expansion_dim", 512) or 512),
            extra_expansion_dim=int(current_meta.get("extra_expansion_dim", 1024) or 1024),
            third_expansion_dim=int(current_meta.get("third_expansion_dim", 3072) or 3072),
            fourth_expansion_dim=int(current_meta.get("fourth_expansion_dim", 4096) or 4096),
            fifth_expansion_dim=int(current_meta.get("fifth_expansion_dim", 6144) or 6144),
            sixth_expansion_dim=int(current_meta.get("sixth_expansion_dim", 8192) or 8192),
            dropout=float(current_meta.get("adapter_dropout", 0.1) or 0.1),
        ).to(self.device)
        target_state = model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key in target_state and tuple(target_state[key].shape) == tuple(value.shape)
        }
        model.load_state_dict(filtered, strict=False)
        self.model = model.eval()

    @torch.no_grad()
    def generate(self, user_text: str, max_new_tokens: int) -> str:
        context = build_context(history=[], user_text=user_text, max_turns=0)
        x = text_to_model_input(context, feature_mode=self.feature_mode).to(self.device)
        logits = self.model(x)[0, 0]
        bucket = choose_bucket_from_logits(logits, self.available_labels, temperature=0.0)
        candidates = self.buckets.get(int(bucket), [])
        if not candidates:
            return ""
        response = pick_response(
            candidates=candidates,
            query_text=user_text,
            recent_assistant_messages=[],
            response_temperature=0.0,
            style_mode="balanced",
            creativity=0.0,
        )
        return _fast_cleanup_response_text(response)


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _extract_answer(item: BenchmarkItem, prediction: str) -> str:
    if item.benchmark == "gsm8k":
        return _extract_last_number(prediction)
    if item.benchmark == "boolq":
        return _extract_yes_no(prediction)
    choices = (item.scoring_data or {}).get("choices")
    if isinstance(choices, dict) and choices:
        normalized_choices = {str(key).upper(): _normalize_text(value) for key, value in choices.items()}
        return _extract_mc_choice(prediction, normalized_choices)
    return _normalize_text(prediction)


def _overall_exact_score(per_benchmark: Dict[str, float]) -> float:
    if not per_benchmark:
        return 0.0
    return float(sum(per_benchmark.values()) / len(per_benchmark))


def discover_models(persist_root: Path, include_qwen_base: bool) -> Tuple[List[ModelSpec], List[Dict[str, str]]]:
    models: List[ModelSpec] = []
    skipped: List[Dict[str, str]] = []

    qwen_base_model = persist_root / "base_models" / "qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775"
    if include_qwen_base and qwen_base_model.exists():
        models.append(ModelSpec(name="qwen_base", family="qwen", kind="qwen", adapter_dir=Path("__no_adapter__")))
    elif include_qwen_base:
        skipped.append({"name": "qwen_base", "reason": f"base model missing: {qwen_base_model}"})

    qwen_dirs = [
        ("qwen_v28", persist_root / "artifacts" / "qwen_supermix_enhanced_v28_cloud_plus_runpod_budget"),
        ("qwen_v29", persist_root / "artifacts" / "qwen_supermix_enhanced_v29_delta_official_refresh_20260326"),
        ("qwen_v30", persist_root / "artifacts" / "qwen_supermix_enhanced_v30_anchor_refresh_20260326"),
    ]
    for name, run_dir in qwen_dirs:
        latest = run_dir / "latest_adapter_checkpoint.txt"
        if latest.exists():
            adapter_dir = Path(_normalize_text(latest.read_text(encoding="utf-8")))
            if adapter_dir.exists():
                models.append(ModelSpec(name=name, family="qwen", kind="qwen", adapter_dir=adapter_dir))
                continue
        skipped.append({"name": name, "reason": f"missing latest_adapter_checkpoint.txt or adapter dir in {run_dir}"})

    v39_dir = persist_root / "artifacts" / "champion_v39_frontier_reasoning_plus_20260327"
    v39_chosen = v39_dir / "v39_frontier_reasoning_plus_chosen_checkpoint.json"
    if v39_chosen.exists():
        try:
            chosen_payload = _load_json(v39_chosen)
            chosen_stage = _normalize_text(chosen_payload.get("chosen_stage"))
            path_map = chosen_payload.get("paths") if isinstance(chosen_payload.get("paths"), dict) else {}
            chosen_info = path_map.get(chosen_stage) if isinstance(path_map, dict) else {}
            if not isinstance(chosen_info, dict):
                chosen_info = {}
            weights_path = Path(_normalize_text(chosen_info.get("weights", "")))
            meta_path = Path(_normalize_text(chosen_info.get("meta", "")))
            if weights_path.exists() and meta_path.exists():
                models.append(
                    ModelSpec(
                        name="v39_final",
                        family="champion",
                        kind="champion",
                        weights_path=weights_path,
                        meta_path=meta_path,
                    )
                )
            else:
                skipped.append(
                    {
                        "name": "v39_final",
                        "reason": f"chosen v39 checkpoint missing weights/meta: {weights_path} | {meta_path}",
                    }
                )
        except Exception as exc:
            skipped.append({"name": "v39_final", "reason": f"failed to parse chosen checkpoint metadata: {exc}"})
    else:
        skipped.append({"name": "v39_final", "reason": f"missing chosen checkpoint metadata: {v39_chosen}"})

    champion_specs = [
        ("v30_lite", "champion", "champion_v30_lite_student_20260326/champion_model_chat_v30_lite_student.pth", "champion_v30_lite_student_20260326/chat_model_meta_v30_lite_student.json"),
        ("v31_stage1", "champion", "champion_v31_hybrid_plus_refresh_20260326/champion_model_chat_v31_hybrid_student_stage1.pth", "champion_v31_hybrid_plus_refresh_20260326/chat_model_meta_v31_hybrid_student_stage1.json"),
        ("v31_final", "champion", "champion_v31_hybrid_plus_refresh_20260326/champion_model_chat_v31_hybrid_plus_refresh.pth", "champion_v31_hybrid_plus_refresh_20260326/chat_model_meta_v31_hybrid_plus_refresh.json"),
        ("v32_smoke", "champion", "champion_v32_smoke_test/smoke_model.pth", "champion_v32_smoke_test/smoke_meta.json"),
        ("v32_stage1", "champion", "champion_v32_omnifuse_20260326/champion_model_chat_v32_omnifuse_stage1.pth", "champion_v32_omnifuse_20260326/chat_model_meta_v32_omnifuse_stage1.json"),
        ("v32_final", "champion", "champion_v32_omnifuse_20260326/champion_model_chat_v32_omnifuse_final.pth", "champion_v32_omnifuse_20260326/chat_model_meta_v32_omnifuse_final.json"),
        ("v33_stage1", "champion", "champion_v33_frontier_full_20260326/champion_model_chat_v33_frontier_stage1.pth", "champion_v33_frontier_full_20260326/chat_model_meta_v33_frontier_stage1.json"),
        ("v33_stage2", "champion", "champion_v33_frontier_full_20260326/champion_model_chat_v33_frontier_stage2.pth", "champion_v33_frontier_full_20260326/chat_model_meta_v33_frontier_stage2.json"),
        ("v33_final", "champion", "champion_v33_frontier_full_20260326/champion_model_chat_v33_frontier_full_final.pth", "champion_v33_frontier_full_20260326/chat_model_meta_v33_frontier_full_final.json"),
        ("v34_stage1", "champion", "champion_v34_frontier_plus_20260326/champion_model_chat_v34_frontier_plus_stage1.pth", "champion_v34_frontier_plus_20260326/chat_model_meta_v34_frontier_plus_stage1.json"),
        ("v34_stage2", "champion", "champion_v34_frontier_plus_20260326/champion_model_chat_v34_frontier_plus_stage2.pth", "champion_v34_frontier_plus_20260326/chat_model_meta_v34_frontier_plus_stage2.json"),
        ("v34_stage3", "champion", "champion_v34_frontier_plus_20260326/champion_model_chat_v34_frontier_plus_stage3.pth", "champion_v34_frontier_plus_20260326/chat_model_meta_v34_frontier_plus_stage3.json"),
        ("v35_stage1", "champion", "champion_v35_collective_allteachers_20260326/champion_model_chat_v35_collective_allteachers_stage1.pth", "champion_v35_collective_allteachers_20260326/chat_model_meta_v35_collective_allteachers_stage1.json"),
        ("v35_stage2", "champion", "champion_v35_collective_allteachers_20260326/champion_model_chat_v35_collective_allteachers_stage2.pth", "champion_v35_collective_allteachers_20260326/chat_model_meta_v35_collective_allteachers_stage2.json"),
        ("v35_stage3", "champion", "champion_v35_collective_allteachers_20260326/champion_model_chat_v35_collective_allteachers_stage3.pth", "champion_v35_collective_allteachers_20260326/chat_model_meta_v35_collective_allteachers_stage3.json"),
        ("v36_native", "native_image", "champion_v36_native_image_20260327/champion_model_chat_v36_native_image_single_checkpoint.pth", "champion_v36_native_image_20260327/chat_model_meta_v36_native_image_single_checkpoint.json"),
        ("v37_native_lite", "native_image", "champion_v37_native_image_lite_20260327/champion_model_chat_v37_native_image_lite_single_checkpoint.pth", "champion_v37_native_image_lite_20260327/chat_model_meta_v37_native_image_lite_single_checkpoint.json"),
        ("v38_native_xlite", "native_image", "champion_v38_native_image_xlite_20260327/champion_model_chat_v38_native_image_xlite_single_checkpoint.pth", "champion_v38_native_image_xlite_20260327/chat_model_meta_v38_native_image_xlite_single_checkpoint.json"),
    ]
    for name, family, weights_rel, meta_rel in champion_specs:
        weights_path = persist_root / "artifacts" / weights_rel
        meta_path = persist_root / "artifacts" / meta_rel
        if weights_path.exists() and meta_path.exists():
            models.append(ModelSpec(name=name, family=family, kind="champion", weights_path=weights_path, meta_path=meta_path))
        else:
            skipped.append({"name": name, "reason": f"missing weights/meta: {weights_path} | {meta_path}"})

    models.sort(key=lambda item: item.name)
    return models, skipped


def _build_generator(spec: ModelSpec, *, device: str, qwen_base_model: Path):
    if spec.kind == "qwen":
        adapter_dir = spec.adapter_dir or Path("__no_adapter__")
        if adapter_dir.name == "__no_adapter__":
            adapter_path = qwen_base_model.parent / "__no_adapter__"
        else:
            adapter_path = adapter_dir
        return QwenGenerator(base_model=str(qwen_base_model), adapter_dir=adapter_path, device=device)
    if spec.kind == "champion":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete champion spec: {spec}")
        return ChampionBenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    raise ValueError(f"Unsupported model spec: {spec}")


def benchmark_models(models: Sequence[ModelSpec], items: Sequence[BenchmarkItem], *, device: str, qwen_base_model: Path, log_every: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    benchmark_names = sorted({item.benchmark for item in items})
    details: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    for index, spec in enumerate(models, start=1):
        print(f"[bench] loading {spec.name} ({index}/{len(models)})")
        start_model = time.time()
        generator = _build_generator(spec, device=device, qwen_base_model=qwen_base_model)
        per_benchmark_exact: Dict[str, List[float]] = {name: [] for name in benchmark_names}
        token_scores: List[float] = []
        char_scores: List[float] = []
        gen_seconds: List[float] = []
        try:
            for item_index, item in enumerate(items, start=1):
                t0 = time.time()
                prediction = _normalize_response(generator.generate(item.prompt, item.max_new_tokens))
                elapsed = time.time() - t0
                extracted = _extract_answer(item, prediction)
                exact = 1.0 if _normalize_text(extracted).lower() == _normalize_text(item.reference_extracted).lower() else 0.0
                token = float(token_f1(item.reference_text, prediction))
                char = float(__import__("difflib").SequenceMatcher(None, item.reference_text.lower(), prediction.lower()).ratio())
                per_benchmark_exact[item.benchmark].append(exact)
                token_scores.append(token)
                char_scores.append(char)
                gen_seconds.append(elapsed)
                details.append(
                    {
                        "model": spec.name,
                        "family": spec.family,
                        "benchmark": item.benchmark,
                        "prompt": item.prompt,
                        "reference_text": item.reference_text,
                        "reference_extracted": item.reference_extracted,
                        "prediction": prediction,
                        "prediction_extracted": extracted,
                        "exact": exact,
                        "token_f1": token,
                        "char_similarity": char,
                        "gen_seconds": elapsed,
                    }
                )
                if log_every > 0 and item_index % log_every == 0:
                    print(f"[bench] {spec.name} {item_index}/{len(items)} done")
        finally:
            del generator
            _release_cuda_memory()

        benchmark_scores = {
            name: float(sum(values) / max(1, len(values)))
            for name, values in per_benchmark_exact.items()
        }
        summary_rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "overall_exact": _overall_exact_score(benchmark_scores),
                "avg_token_f1": float(sum(token_scores) / max(1, len(token_scores))),
                "avg_char_similarity": float(sum(char_scores) / max(1, len(char_scores))),
                "avg_gen_seconds": float(sum(gen_seconds) / max(1, len(gen_seconds))),
                "model_seconds": float(time.time() - start_model),
                "benchmarks": benchmark_scores,
            }
        )
        print(f"[bench] finished {spec.name} overall_exact={summary_rows[-1]['overall_exact']:.4f}")

    summary_rows.sort(key=lambda row: float(row["overall_exact"]), reverse=True)
    return summary_rows, details


def _filter_models(models: Sequence[ModelSpec], skipped: Sequence[Dict[str, str]], requested: Sequence[str]) -> Tuple[List[ModelSpec], List[Dict[str, str]]]:
    wanted = {name.strip().lower() for name in requested if name.strip()}
    if not wanted:
        return list(models), list(skipped)
    selected = [model for model in models if model.name.lower() in wanted]
    selected_names = {model.name.lower() for model in selected}
    filtered_skipped = list(skipped)
    for missing in sorted(wanted - selected_names):
        filtered_skipped.append({"name": missing, "reason": "requested model was not discovered"})
    return selected, filtered_skipped


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], benchmark_names: Sequence[str]) -> None:
    fieldnames = ["model", "family", "overall_exact", "avg_token_f1", "avg_char_similarity", "avg_gen_seconds", "model_seconds"] + list(benchmark_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "model": row["model"],
                "family": row["family"],
                "overall_exact": f"{float(row['overall_exact']):.6f}",
                "avg_token_f1": f"{float(row['avg_token_f1']):.6f}",
                "avg_char_similarity": f"{float(row['avg_char_similarity']):.6f}",
                "avg_gen_seconds": f"{float(row['avg_gen_seconds']):.6f}",
                "model_seconds": f"{float(row['model_seconds']):.6f}",
            }
            payload.update({name: f"{float(row['benchmarks'].get(name, 0.0)):.6f}" for name in benchmark_names})
            writer.writerow(payload)


def _family_color(family: str) -> str:
    if family == "qwen":
        return "#d97706"
    if family == "native_image":
        return "#15803d"
    return "#2563eb"


def draw_graph(path: Path, summary_rows: Sequence[Dict[str, object]], benchmark_names: Sequence[str]) -> None:
    model_names = [str(row["model"]) for row in summary_rows]
    families = [str(row["family"]) for row in summary_rows]
    exacts = [float(row["overall_exact"]) for row in summary_rows]
    heatmap = [[float(row["benchmarks"].get(name, 0.0)) for name in benchmark_names] for row in summary_rows]

    fig_height = max(8.0, 0.36 * len(summary_rows) + 2.5)
    fig_width = max(16.0, 11.0 + 1.05 * len(benchmark_names))
    fig, (ax_heatmap, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
        constrained_layout=True,
    )

    im = ax_heatmap.imshow(heatmap, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_heatmap.set_title("Exact Accuracy by Benchmark")
    ax_heatmap.set_xticks(range(len(benchmark_names)))
    ax_heatmap.set_xticklabels(benchmark_names, rotation=20, ha="right")
    ax_heatmap.set_yticks(range(len(model_names)))
    ax_heatmap.set_yticklabels(model_names)
    cbar = fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    cbar.set_label("Accuracy")

    y_pos = list(range(len(model_names)))
    colors = [_family_color(family) for family in families]
    ax_bar.barh(y_pos, exacts, color=colors)
    ax_bar.set_title("Overall Exact Accuracy")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(model_names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, max(0.25, max(exacts) * 1.15 if exacts else 0.25))
    ax_bar.set_xlabel("Mean exact score")
    for yi, score in zip(y_pos, exacts):
        ax_bar.text(score + 0.005, yi, f"{score:.3f}", va="center", fontsize=8)

    legend_handles = []
    for family in ("qwen", "champion", "native_image"):
        if family in families:
            legend_handles.append(plt.Line2D([0], [0], color=_family_color(family), lw=8, label=family))
    if legend_handles:
        ax_bar.legend(handles=legend_handles, loc="lower right")

    fig.suptitle("Supermix Model Comparison on Expanded Sampled Common Benchmarks", fontsize=14)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark all trained Supermix models on sampled common benchmarks.")
    parser.add_argument("--persist_root", default="/workspace/supermix_runpod_budget/persistent")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample_per_benchmark", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--log_every", type=int, default=12)
    parser.add_argument("--include_qwen_base", action="store_true")
    parser.add_argument("--model_name", action="append", default=[], help="Benchmark only the named discovered model. Repeat for multiple models.")
    args = parser.parse_args()

    persist_root = Path(args.persist_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = build_benchmark_items(sample_per_benchmark=int(args.sample_per_benchmark), seed=int(args.seed))
    benchmark_names = sorted({item.benchmark for item in items})

    prompts_jsonl = output_dir / "benchmark_items.jsonl"
    _write_jsonl(
        prompts_jsonl,
        (
            {
                "benchmark": item.benchmark,
                "prompt": item.prompt,
                "reference_text": item.reference_text,
                "reference_extracted": item.reference_extracted,
                "max_new_tokens": item.max_new_tokens,
            }
            for item in items
        ),
    )

    models, skipped = discover_models(persist_root, include_qwen_base=bool(args.include_qwen_base))
    models, skipped = _filter_models(models, skipped, requested=args.model_name)
    if not models:
        raise RuntimeError("No benchmarkable models were selected.")
    qwen_base_model = persist_root / "base_models" / "qwen2_5_0_5b_instruct_7ae557604adf67be50417f59c2c2f167def9a775"
    summary_rows, details = benchmark_models(
        models,
        items,
        device=str(args.device),
        qwen_base_model=qwen_base_model,
        log_every=int(args.log_every),
    )

    graph_path = output_dir / "benchmark_all_models_common_graph.png"
    draw_graph(graph_path, summary_rows, benchmark_names)

    summary_path = output_dir / "benchmark_all_models_common_summary.json"
    details_path = output_dir / "benchmark_all_models_common_details.jsonl"
    csv_path = output_dir / "benchmark_all_models_common_table.csv"

    _write_jsonl(details_path, details)
    _write_csv(csv_path, summary_rows, benchmark_names)
    summary_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "persist_root": str(persist_root),
                "output_dir": str(output_dir),
                "device": str(args.device),
                "sample_per_benchmark": int(args.sample_per_benchmark),
                "benchmarks": benchmark_names,
                "models_benchmarked": [row["model"] for row in summary_rows],
                "skipped_models": skipped,
                "summary_rows": summary_rows,
                "artifacts": {
                    "prompts_jsonl": str(prompts_jsonl),
                    "details_jsonl": str(details_path),
                    "table_csv": str(csv_path),
                    "graph_png": str(graph_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[bench] wrote summary to {summary_path}")
    print(f"[bench] wrote graph to {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
