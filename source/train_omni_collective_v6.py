from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from multimodel_catalog import discover_model_records
    from multimodel_runtime import UnifiedModelManager
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v6_model import OmniCollectiveEngineV6, OmniCollectiveNetV6
    from train_image_recognition_model import ensure_base_images
    from train_math_equation_model import build_rows as build_math_prompt_rows
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from train_omni_collective_v3 import _score_candidate
    from train_omni_collective_v4 import (
        DEFAULT_QWEN_ADAPTER_ZIP,
        _load_expanded_state_from_zip,
        _resolve_local_qwen_base_model,
        _train_stage,
    )
    from train_omni_collective_v5 import _build_rows_v5
    from v40_benchmax_common import build_protein_folding_rows
    from math_equation_model import solve_intent
except ImportError:  # pragma: no cover
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .multimodel_catalog import discover_model_records
    from .multimodel_runtime import UnifiedModelManager
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v6_model import OmniCollectiveEngineV6, OmniCollectiveNetV6
    from .train_image_recognition_model import ensure_base_images
    from .train_math_equation_model import build_rows as build_math_prompt_rows
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from .train_omni_collective_v3 import _score_candidate
    from .train_omni_collective_v4 import (
        DEFAULT_QWEN_ADAPTER_ZIP,
        _load_expanded_state_from_zip,
        _resolve_local_qwen_base_model,
        _train_stage,
    )
    from .train_omni_collective_v5 import _build_rows_v5
    from .v40_benchmax_common import build_protein_folding_rows
    from .math_equation_model import solve_intent


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v5_frontier_20260330.zip")


def _conversation_alignment_rows_v6() -> List[OmniRow]:
    items = [
        ("Reply in exactly two sentences explaining why regression tests matter.", "Strong regression tests catch behavior changes early. They make refactors safer because you can verify that old guarantees still hold.", "planning"),
        ("Answer in one short paragraph and stay on topic: why should I write smaller functions?", "Smaller functions are easier to test, review, reuse, and debug because each one does less and exposes fewer hidden side effects.", "coding"),
        ("Give me three bullet points only: how do I keep a code change reversible?", "- Make the diff small.\n- Add or update tests before broader cleanup.\n- Keep a clear rollback path for the changed behavior.", "planning"),
        ("The user asked for a concise answer. What should the assistant do?", "It should answer directly, keep the wording tight, and avoid drifting into unrelated explanation or filler.", "general"),
        ("How should an assistant respond when the prompt gives an explicit output format?", "It should follow the requested format exactly unless doing so would make the answer invalid or unsafe.", "general"),
        ("What is a good behavior when a request is underspecified but still answerable?", "Make the smallest reasonable assumption, state it briefly if it matters, and move the task forward instead of stalling.", "planning"),
        ("Rewrite this to sound more natural in conversation: I am unable to comply with that formatting request.", "I can't follow that formatting exactly, but I can still help in a nearby format.", "language"),
        ("Explain why preserving the user's constraint matters in a chat app.", "Preserving constraints keeps the response aligned with the user's real task instead of giving a generic answer that technically says something but fails the request.", "general"),
        ("What should a grounded assistant do with a request for live current facts it cannot verify locally?", "It should say that current facts need web verification instead of inventing an answer from stale local knowledge.", "knowledge"),
        ("Answer in JSON with keys cause and fix: why might a local model hallucinate more than expected?", "{\"cause\":\"The prompt is underspecified or the model is overconfident outside its grounded knowledge.\",\"fix\":\"Use tighter prompts, stronger retrieval or verification, and training examples that reward uncertainty over invention.\"}", "coding"),
    ]
    return [
        OmniRow(
            prompt=prompt,
            intent="planning" if domain == "planning" else ("coding" if domain == "coding" else ("language" if domain == "language" else "general")),
            response_text=_normalize(response, 420),
            domain=domain if domain in OMNI_DOMAIN_LABELS_V2 else "general",
            source="conversation_alignment_v6",
        )
        for prompt, response, domain in items
    ]


def _grounding_rows_v6() -> List[OmniRow]:
    items = [
        ("What should you do if you are unsure about a specific current event and cannot verify it locally?", "Say that you cannot verify the current fact locally and recommend using web search instead of guessing.", "knowledge"),
        ("Is it okay to invent a citation if you think it is probably right?", "No. If you cannot verify a citation, do not invent it. Say that you do not have a verified source.", "knowledge"),
        ("How should a grounded assistant answer when an image has not been uploaded but the user asks what is in it?", "It should say that no image was provided yet and ask for the upload instead of pretending to see one.", "general"),
        ("What is the safe way to answer a question whose exact value depends on the latest data?", "Give the stable background if useful, but clearly separate it from the current value and say that the latest figure needs verification.", "knowledge"),
        ("How should an assistant behave when a prompt contains ambiguity but the answer space is risky?", "State the uncertainty, avoid overclaiming, and prefer a narrower answer or a short clarifying question over a fabricated detail.", "planning"),
        ("What makes an answer more grounded and less hallucinatory?", "Grounded answers stay close to the prompt, avoid unsupported specifics, and openly signal uncertainty when evidence is weak.", "knowledge"),
        ("If you have only local training data and no live retrieval, how should you answer 'what happened today'?", "Say that you do not have verified live updates in local memory and that today's events require web or external lookup.", "knowledge"),
        ("What is the correct behavior if a user asks for a file path you cannot confirm exists?", "Avoid claiming the path exists. Say what you know, then ask for verification or inspect the filesystem before asserting specifics.", "coding"),
    ]
    return [
        OmniRow(
            prompt=prompt,
            intent="knowledge" if domain == "knowledge" else "planning",
            response_text=_normalize(response, 420),
            domain=domain if domain in OMNI_DOMAIN_LABELS_V2 else "general",
            source="grounding_v6",
        )
        for prompt, response, domain in items
    ]


def _conversation_focus_rows_v6(repo_root: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    counts: Dict[str, int] = defaultdict(int)
    dataset_specs = [
        ("conversation_data.hybrid_v6_live_knowledge.jsonl", 1400, "general", "conversation_hybrid_v6"),
        ("conversation_data.supermix_plus_v27_500k.jsonl", 1800, "knowledge", "conversation_supermix_plus_v6"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 960, "planning", "conversation_reasoning_v6"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 1100, "creative", "conversation_creative_v6"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 720, "language", "conversation_books_v6"),
        ("conversation_data.quality_anchor_v2.jsonl", 180, "general", "conversation_quality_anchor_v6"),
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 320, "knowledge", "conversation_delta_anchor_v6"),
        ("conversation_data.delta_official_refresh_2026_03_26.jsonl", 160, "knowledge", "conversation_delta_refresh_v6"),
        ("conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 220, "language", "conversation_dictionary_v6"),
        ("conversation_data.bible_kjv_public_domain_smoke.jsonl", 180, "language", "conversation_bible_v6"),
        ("conversation_data.science_essentials_smoke.jsonl", 180, "knowledge", "conversation_science_v6"),
        ("conversation_data.science_novel_examples_smoke.jsonl", 140, "knowledge", "conversation_science_novel_v6"),
        ("conversation_data.world_events_2026_02_19.jsonl", 180, "knowledge", "conversation_world_events_v6"),
    ]
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(rows), domain=domain, source_tag=source_tag)
        rows.extend(sampled)
        counts[source_tag] += len(sampled)
    return rows, dict(sorted(counts.items()))


def _math_exact_rows_v6(repo_root: Path, seed: int, limit: int) -> List[OmniRow]:
    rows: List[OmniRow] = []
    prompts = build_math_prompt_rows(samples_per_intent=max(18, int(limit)), seed=seed)
    for prompt, label in prompts[: max(24, int(limit) * 6)]:
        try:
            solved = solve_intent(prompt, label)
            response = _normalize(str(solved.get("response") or ""), 420)
        except Exception:
            continue
        if len(response) < 4:
            continue
        rows.append(
            OmniRow(
                prompt=_normalize(prompt, 240),
                intent="math",
                response_text=response,
                domain="math",
                source="math_exact_v6",
            )
        )
    datasets_dir = repo_root / "datasets"
    math_jsonl = datasets_dir / "conversation_data.english_math_smoke_v3.jsonl"
    if math_jsonl.exists():
        rows.extend(
            _rows_from_jsonl(
                math_jsonl,
                limit=max(40, int(limit) // 2),
                seed=seed + 71,
                domain="math",
                source_tag="math_jsonl_v6",
            )
        )
    return rows


def _protein_rows_v6(seed: int, limit: int) -> List[OmniRow]:
    protein_rows, _summary = build_protein_folding_rows(seed=seed, max_rows=limit)
    cooked: List[OmniRow] = []
    for item in protein_rows:
        prompt = _normalize(str(item.get("prompt") or ""), 260)
        response = _normalize(str(item.get("response_text") or item.get("assistant") or ""), 420)
        if len(prompt) < 8 or len(response) < 8:
            continue
        cooked.append(
            OmniRow(
                prompt=prompt,
                intent=str(item.get("intent") or "knowledge"),
                response_text=response,
                domain=str(item.get("domain") or "knowledge"),
                source="protein_folding_v6",
            )
        )
    return cooked


def _build_rows_v6(repo_root: Path, models_dir: Path, images_dir: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows, source_counts = _build_rows_v5(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    conversation_rows, conversation_counts = _conversation_focus_rows_v6(repo_root=repo_root, seed=seed + 401)
    grounding_rows = _grounding_rows_v6()
    alignment_rows = _conversation_alignment_rows_v6()
    math_rows = _math_exact_rows_v6(repo_root=repo_root, seed=seed + 433, limit=60)
    protein_rows = _protein_rows_v6(seed=seed + 457, limit=180)

    rows.extend(conversation_rows)
    rows.extend(grounding_rows)
    rows.extend(alignment_rows)
    rows.extend(math_rows)
    rows.extend(protein_rows)

    for key, value in conversation_counts.items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    source_counts["grounding_v6"] = len(grounding_rows)
    source_counts["conversation_alignment_v6"] = len(alignment_rows)
    source_counts["math_exact_v6"] = len([row for row in math_rows if row.source == "math_exact_v6"])
    source_counts["math_jsonl_v6"] = len([row for row in math_rows if row.source == "math_jsonl_v6"])
    source_counts["protein_folding_v6"] = len(protein_rows)
    return rows, dict(sorted(source_counts.items()))


def _teacher_order_key(record_key: str) -> Tuple[int, str]:
    priority = {
        "qwen_v28": 0,
        "qwen_v30": 1,
    }
    return (priority.get(record_key, 10), record_key)


def _sample_teacher_rows(rows: Sequence[OmniRow], *, seed: int, limit: int) -> List[OmniRow]:
    eligible = [
        row
        for row in rows
        if row.domain in {"coding", "knowledge", "language", "creative", "planning", "general", "image_prompt", "model_selection", "math"}
    ]
    grouped: Dict[str, List[OmniRow]] = defaultdict(list)
    for row in eligible:
        grouped[row.domain].append(row)
    rng = random.Random(int(seed))
    sample: List[OmniRow] = []
    per_domain = max(6, int(limit) // max(1, len(grouped)))
    for domain, items in sorted(grouped.items()):
        cooked = list(items)
        rng.shuffle(cooked)
        sample.extend(cooked[:per_domain])
    rng.shuffle(sample)
    return sample[: int(limit)]


def _model_style_for_row(row: OmniRow) -> str:
    if row.domain in {"coding", "model_selection"}:
        return "coding"
    if row.domain in {"creative", "image_prompt", "language"}:
        return "creative"
    return "analyst"


def _all_model_distill_rows(
    rows: Sequence[OmniRow],
    *,
    repo_root: Path,
    models_dir: Path,
    seed: int,
    limit: int,
    teacher_model_limit: int,
) -> Tuple[List[OmniRow], Dict[str, object]]:
    sample = _sample_teacher_rows(rows, seed=seed, limit=limit)
    if not str(os.environ.get("SUPERMIX_QWEN_BASE_MODEL_DIR") or "").strip():
        try:
            os.environ["SUPERMIX_QWEN_BASE_MODEL_DIR"] = str(_resolve_local_qwen_base_model())
        except Exception:
            pass
    teacher_records = [record for record in discover_model_records(models_dir=models_dir) if record.supports_chat]
    teacher_records.sort(key=lambda item: _teacher_order_key(item.key))
    if teacher_model_limit > 0:
        teacher_records = teacher_records[: int(teacher_model_limit)]

    extraction_root = (repo_root / "output" / "omni_v6_teacher_cache").resolve()
    generated_dir = (repo_root / "output" / "omni_v6_teacher_generated").resolve()
    manager = UnifiedModelManager(tuple(teacher_records), extraction_root=extraction_root, generated_dir=generated_dir, device_preference="cpu")

    best_by_index: Dict[int, Tuple[float, str, str]] = {}
    direct_counts: Dict[str, int] = defaultdict(int)
    repair_counts: Dict[str, int] = defaultdict(int)
    empty_counts: Dict[str, int] = defaultdict(int)
    unavailable_teachers: Dict[str, str] = {}

    for teacher_idx, record in enumerate(teacher_records, start=1):
        try:
            _, backend = manager.ensure_backend(record.key)
        except Exception as exc:
            unavailable_teachers[record.key] = f"{type(exc).__name__}: {exc}"
            print(
                json.dumps(
                    {
                        "event": "teacher_model_unavailable",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "reason": unavailable_teachers[record.key],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        accepted_for_teacher = 0
        for row_index, row in enumerate(sample, start=1):
            try:
                result = backend.chat(
                    session_id=f"v6distill::{record.key}::{row_index}",
                    prompt=row.prompt,
                    settings={
                        "style_mode": _model_style_for_row(row),
                        "response_temperature": 0.0,
                        "temperature": 0.0,
                        "max_new_tokens": 144,
                        "top_p": 0.92,
                    },
                )
                candidate = str(result.response or "").strip()
            except Exception:
                candidate = ""
            finally:
                backend.clear(f"v6distill::{record.key}::{row_index}")
            if not candidate:
                empty_counts[record.key] += 1
                continue
            score = float(_score_candidate(row.response_text, candidate)["composite"])
            best = best_by_index.get(row_index)
            if best is None or score > best[0]:
                best_by_index[row_index] = (score, record.key, candidate)
                accepted_for_teacher += 1
        print(
            json.dumps(
                {
                    "event": "teacher_model_done",
                    "teacher_index": teacher_idx,
                    "teacher_total": len(teacher_records),
                    "teacher": record.key,
                    "sample_rows": len(sample),
                    "non_empty": len(sample) - empty_counts.get(record.key, 0),
                    "best_updates": accepted_for_teacher,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    accepted: List[OmniRow] = []
    discarded = 0
    for row_index, row in enumerate(sample, start=1):
        best = best_by_index.get(row_index)
        if best is None:
            discarded += 1
            continue
        score, teacher_key, candidate = best
        if score >= 0.24:
            accepted.append(
                OmniRow(
                    prompt=row.prompt,
                    intent=row.intent,
                    response_text=_normalize(candidate, 420),
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_distill_v6",
                )
            )
            direct_counts[teacher_key] += 1
        elif score >= 0.10:
            accepted.append(
                OmniRow(
                    prompt=_normalize(
                        "Repair and ground this draft answer so it becomes concise, correct, and less speculative.\n"
                        f"Request: {row.prompt}\n"
                        f"Draft: {candidate}",
                        340,
                    ),
                    intent=row.intent,
                    response_text=row.response_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_repair_v6",
                )
            )
            repair_counts[teacher_key] += 1
        else:
            discarded += 1
        if row_index % 8 == 0 or row_index == len(sample):
            print(
                json.dumps(
                    {
                        "event": "teacher_league_progress",
                        "completed": row_index,
                        "total": len(sample),
                        "accepted": len(accepted),
                        "discarded": discarded,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    if manager._backend is not None:
        manager._backend.unload()
    gc.collect()
    return accepted, {
        "requested": int(limit),
        "sampled": len(sample),
        "accepted_total": len(accepted),
        "accepted_direct": dict(sorted(direct_counts.items())),
        "accepted_repair": dict(sorted(repair_counts.items())),
        "empty_counts": dict(sorted(empty_counts.items())),
        "discarded": discarded,
        "teacher_keys": [record.key for record in teacher_records],
        "unavailable_teachers": dict(sorted(unavailable_teachers.items())),
        "forced_qwen_keys": ["qwen_v28", "qwen_v30"],
    }


def build_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    full_rows, source_counts = _build_rows_v6(repo_root=repo_root, models_dir=models_dir, images_dir=images_dir, seed=seed)
    distill_rows, distill_summary = _all_model_distill_rows(
        full_rows,
        repo_root=repo_root,
        models_dir=models_dir,
        seed=seed + 521,
        limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
    )
    full_rows.extend(distill_rows)
    source_counts["all_model_distill_total_v6"] = len(distill_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    return stage1_rows, list(full_rows), {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(full_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_league": distill_summary,
    }


def train_model(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    base_zip: Path,
    images_dir: Path,
    image_size: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    _stage1_rows, full_rows, dataset_summary = build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        distill_limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
    )
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    vocab = build_char_vocab([row.prompt for row in train_rows], min_frequency=1)
    response_bank = sorted({row.response_text for row in full_rows if row.response_text})
    print(
        json.dumps(
            {
                "event": "label_space",
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "vocab_size": len(vocab),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    max_len = 360
    max_words = 80
    word_buckets = 16384
    device = torch.device("cpu")
    model = OmniCollectiveNetV6(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=104,
        text_hidden=208,
        image_channels=44,
        word_buckets=word_buckets,
        word_embed_dim=96,
        deep_text_channels=272,
        deep_image_channels=96,
        fusion_hidden=768,
        memory_slots=16,
        depth_steps=7,
        expert_count=6,
        expert_hidden=1152,
        context_top_k=3,
        expert_top_k=2,
    ).to(device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)

    stage1 = _train_stage(
        model=model,
        train_rows=train_stage1,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=batch_size,
        learning_rate=stage1_lr,
        epochs=stage1_epochs,
        seed=seed + 101,
        device=device,
        loss_weights={"intent": 0.60, "response": 1.00, "domain": 0.78, "vision": 0.70},
        balance_weight=0.030,
    )
    stage2 = _train_stage(
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=max(16, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        device=device,
        loss_weights={"intent": 0.54, "response": 1.00, "domain": 0.74, "vision": 1.02},
        balance_weight=0.040,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v6_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v6_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v6_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v6_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v6_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 6,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 104,
        "text_hidden": 208,
        "image_channels": 44,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 96,
        "deep_text_channels": 272,
        "deep_image_channels": 96,
        "fusion_hidden": 768,
        "memory_slots": 16,
        "depth_steps": 7,
        "expert_count": 6,
        "expert_hidden": 1152,
        "context_top_k": 3,
        "expert_top_k": 2,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "deliberation_passes": 6,
        "minimum_passes": 3,
        "grounding_threshold": 0.46,
        "prompt_understanding_mode": "grounded_conversation_math_consensus",
        "notes": [
            "v6 distills the local chat-capable model catalog, forcing qwen_v28 and qwen_v30 into the teacher league.",
            "The continuation adds more conversation, grounding, math exact-solve, and protein-folding knowledge rows while increasing internal deliberation depth.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV6(weights_path=weights_path, meta_path=meta_path, device=device)
    sample_prompts = [
        "Reply in exactly two sentences explaining why regression tests matter.",
        "Solve 3*x + 7 = 19.",
        "What should a grounded assistant do when it cannot verify a current fact locally?",
        "Write a tiny OpenSCAD snippet for a centered cylinder with a hole.",
        "Why do multiple-sequence alignments help protein structure prediction?",
    ]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts],
        "notes": [
            "v6 grows v5 again and blends full-catalog distillation with conversation, grounding, math, and protein-folding supervision.",
            "Inference uses a longer grounded deliberation loop to reduce hallucinated specifics on weak-confidence prompts.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "dataset_summary": dataset_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the omni_collective_v6 continuation model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00038)
    parser.add_argument("--stage2_lr", type=float, default=0.00018)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--distill_limit", type=int, default=64)
    parser.add_argument("--teacher_model_limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_model(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        models_dir=Path(args.models_dir).resolve(),
        base_zip=Path(args.base_zip).resolve(),
        images_dir=Path(args.images_dir).resolve(),
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
        distill_limit=int(args.distill_limit),
        teacher_model_limit=int(args.teacher_model_limit),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
