from __future__ import annotations

import argparse
import gc
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.append(str(SOURCE_DIR))

try:
    from dcgan_image_model import DCGAN_SPECS
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from mattergen_generation_model import MATTERGEN_CONCEPT_SPECS
    from multimodel_catalog import ModelRecord, discover_model_records
    from multimodel_runtime import UnifiedModelManager
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_training_runtime import ModelEma, TrainingRuntime, create_warmup_cosine_scheduler, maybe_compile_model, resolve_training_runtime
    from omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8
    from protein_folding_model import PROTEIN_CONCEPT_LABELS
    from three_d_generation_model import THREE_D_GENERATION_SPECS
    from train_image_recognition_model import ensure_base_images
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniDatasetV2, OmniRow, _normalize, _rows_from_jsonl, _weighted_score, evaluate, split_rows
    from train_omni_collective_v3 import _score_candidate
    from train_omni_collective_v4 import _load_expanded_state_from_zip, _resolve_local_qwen_base_model
    from train_omni_collective_v6 import (
        _build_rows_v6,
        _conversation_alignment_rows_v6,
        _grounding_rows_v6,
        _math_exact_rows_v6,
        _protein_rows_v6,
        _sample_teacher_rows,
    )
    from v40_benchmax_common import build_protein_folding_rows, build_v33_style_rows, build_v39_style_rows, prompt_row_to_omni
except ImportError:  # pragma: no cover
    from .dcgan_image_model import DCGAN_SPECS
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .mattergen_generation_model import MATTERGEN_CONCEPT_SPECS
    from .multimodel_catalog import ModelRecord, discover_model_records
    from .multimodel_runtime import UnifiedModelManager
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_training_runtime import ModelEma, TrainingRuntime, create_warmup_cosine_scheduler, maybe_compile_model, resolve_training_runtime
    from .omni_collective_v8_model import OmniCollectiveEngineV8, OmniCollectiveNetV8
    from .protein_folding_model import PROTEIN_CONCEPT_LABELS
    from .three_d_generation_model import THREE_D_GENERATION_SPECS
    from .train_image_recognition_model import ensure_base_images
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniDatasetV2, OmniRow, _normalize, _rows_from_jsonl, _weighted_score, evaluate, split_rows
    from .train_omni_collective_v3 import _score_candidate
    from .train_omni_collective_v4 import _load_expanded_state_from_zip, _resolve_local_qwen_base_model
    from .train_omni_collective_v6 import (
        _build_rows_v6,
        _conversation_alignment_rows_v6,
        _grounding_rows_v6,
        _math_exact_rows_v6,
        _protein_rows_v6,
        _sample_teacher_rows,
    )
    from .v40_benchmax_common import build_protein_folding_rows, build_v33_style_rows, build_v39_style_rows, prompt_row_to_omni


DEFAULT_BASE_ZIP = Path("output/supermix_omni_collective_v7_frontier_20260403.zip")


def _safe_domain(domain: str) -> str:
    cooked = str(domain or "general")
    return cooked if cooked in OMNI_DOMAIN_LABELS_V2 else "general"


def _dedupe_rows(rows: Sequence[OmniRow]) -> List[OmniRow]:
    deduped: List[OmniRow] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.prompt).strip().lower(),
            str(row.response_text).strip().lower(),
            str(row.domain).strip().lower(),
            str(row.source).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _omni_row_to_payload(row: OmniRow) -> Dict[str, Any]:
    return {
        "prompt": row.prompt,
        "intent": row.intent,
        "response_text": row.response_text,
        "domain": row.domain,
        "source": row.source,
        "image_path": row.image_path,
        "vision_label": row.vision_label,
    }


def _omni_row_from_payload(payload: Dict[str, Any]) -> OmniRow:
    return OmniRow(
        prompt=str(payload.get("prompt") or ""),
        intent=str(payload.get("intent") or "general"),
        response_text=str(payload.get("response_text") or ""),
        domain=_safe_domain(str(payload.get("domain") or "general")),
        source=str(payload.get("source") or "teacher_resume_v8"),
        image_path=str(payload.get("image_path")) if payload.get("image_path") else None,
        vision_label=str(payload.get("vision_label")) if payload.get("vision_label") else None,
    )


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    if text:
        text += "\n"
    temp_path.write_text(text, encoding="utf-8")
    temp_path.replace(path)


def _stage_resume_dir_v8(output_dir: Path, *, seed: int, distill_limit: int, teacher_model_limit: int) -> Path:
    teacher_tag = int(teacher_model_limit) if int(teacher_model_limit) > 0 else 0
    return (output_dir / "omni_v8_stage_resume" / f"seed_{int(seed)}_distill_{int(distill_limit)}_teacherlimit_{teacher_tag}").resolve()


def _run_state_path_v8(output_dir: Path) -> Path:
    return (output_dir / "omni_collective_v8_train_state.json").resolve()


def _write_run_state_v8(path: Path, payload: Dict[str, Any]) -> None:
    cooked = dict(payload)
    cooked["updated_at"] = datetime.now().isoformat()
    _write_json_atomic(path, cooked)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sorted_model_records_v8(models_dir: Path, allowed_model_keys: Optional[Sequence[str]] = None) -> List[ModelRecord]:
    records = sorted(discover_model_records(models_dir=models_dir), key=lambda item: item.key)
    if not allowed_model_keys:
        return records
    allowed = {str(key) for key in allowed_model_keys}
    return [record for record in records if record.key in allowed]


def _load_frozen_dataset_summary_v8(stage_resume_dir: Path) -> Optional[Dict[str, Any]]:
    payload = _load_json_if_exists(stage_resume_dir / "dataset_summary.json")
    if not isinstance(payload, dict):
        return None
    summary = payload.get("dataset_summary")
    if not isinstance(summary, dict):
        return None
    resume_artifacts = [
        _stage_progress_path_v8(stage_resume_dir, "stage1"),
        _stage_progress_path_v8(stage_resume_dir, "stage2"),
        _stage_complete_weights_path_v8(stage_resume_dir, "stage1"),
        _stage_complete_meta_path_v8(stage_resume_dir, "stage1"),
        _stage_complete_weights_path_v8(stage_resume_dir, "stage2"),
        _stage_complete_meta_path_v8(stage_resume_dir, "stage2"),
    ]
    if not any(path.exists() for path in resume_artifacts):
        return None
    return summary


def _stage_progress_path_v8(stage_dir: Path, stage_name: str) -> Path:
    return stage_dir / f"{stage_name}_progress.pt"


def _stage_complete_weights_path_v8(stage_dir: Path, stage_name: str) -> Path:
    return stage_dir / f"{stage_name}_complete_weights.pth"


def _stage_complete_meta_path_v8(stage_dir: Path, stage_name: str) -> Path:
    return stage_dir / f"{stage_name}_complete.json"


def _clone_state_dict_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def _stage_progress_temp_path_v8(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")


def _load_stage_progress_checkpoint_v8(checkpoint_path: Path) -> Tuple[Optional[Dict[str, object]], Optional[Path]]:
    candidates = [checkpoint_path, _stage_progress_temp_path_v8(checkpoint_path)]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None, None
    existing.sort(key=lambda path: (path.stat().st_mtime, path.name.endswith(".tmp")), reverse=True)
    last_error: Optional[str] = None
    for candidate in existing:
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            return payload, candidate
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    if last_error is not None:
        print(
            json.dumps(
                {
                    "event": "stage_progress_load_warning",
                    "checkpoint_candidates": [str(path) for path in existing],
                    "error": last_error,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    return None, None


def _restore_scheduler_state_v8(
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scheduler_state: Optional[Dict[str, object]],
    *,
    optimizer_steps_done: int,
    stage_name: str,
    checkpoint_source: Optional[Path],
) -> int:
    state = dict(scheduler_state or {})
    if not state:
        return max(0, int(optimizer_steps_done))
    try:
        scheduler.load_state_dict(state)
        resumed_steps = max(
            0,
            int(
                optimizer_steps_done
                or state.get("_step_count")
                or state.get("last_epoch")
                or 0
            ),
        )
        return resumed_steps
    except Exception as exc:
        resumed_steps = max(
            0,
            int(
                optimizer_steps_done
                or state.get("_step_count")
                or state.get("last_epoch")
                or 0
            ),
        )
        if resumed_steps > 0:
            for _ in range(resumed_steps):
                scheduler.step()
        print(
            json.dumps(
                {
                    "event": "scheduler_resume_fallback",
                    "stage": stage_name,
                    "checkpoint_source": str(checkpoint_source) if checkpoint_source is not None else None,
                    "resumed_steps": resumed_steps,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return resumed_steps


def _save_stage_progress_checkpoint_v8(
    *,
    checkpoint_path: Path,
    stage_name: str,
    epoch: int,
    next_batch_index: int,
    total_batches: int,
    total_loss: float,
    total_items: int,
    optimizer_steps_done: int,
    best_score: float,
    best_state: Optional[Dict[str, torch.Tensor]],
    ema_state: Optional[Dict[str, torch.Tensor]],
    history: Sequence[Dict[str, float]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    save_best_state: bool = False,
    save_ema_state: bool = False,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_name": str(stage_name),
        "epoch": int(epoch),
        "next_batch_index": int(next_batch_index),
        "total_batches": int(total_batches),
        "total_loss": float(total_loss),
        "total_items": int(total_items),
        "optimizer_steps_done": int(optimizer_steps_done),
        "best_score": float(best_score),
        "history": list(history),
        "saved_at": datetime.now().isoformat(),
        "model_state": _clone_state_dict_cpu(model.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_state": _clone_state_dict_cpu(best_state) if save_best_state and best_state is not None else None,
        "ema_state": _clone_state_dict_cpu(ema_state) if save_ema_state and ema_state is not None else None,
    }
    temp_path = _stage_progress_temp_path_v8(checkpoint_path)
    try:
        torch.save(payload, temp_path)
    except Exception as exc:
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "event": "checkpoint_write_warning",
                    "checkpoint_path": str(checkpoint_path),
                    "temp_path": str(temp_path),
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return
    replace_error: Optional[str] = None
    for attempt in range(8):
        try:
            temp_path.replace(checkpoint_path)
            return
        except PermissionError as exc:
            replace_error = f"{type(exc).__name__}: {exc}"
        except OSError as exc:
            replace_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(0.25 * float(attempt + 1), 2.0))
    print(
        json.dumps(
            {
                "event": "checkpoint_replace_warning",
                "checkpoint_path": str(checkpoint_path),
                "temp_path": str(temp_path),
                "error": replace_error or "unknown replace error",
            },
            ensure_ascii=True,
        ),
        flush=True,
    )


def _save_stage_complete_v8(
    *,
    stage_dir: Path,
    stage_name: str,
    model: torch.nn.Module,
    result: Dict[str, object],
) -> Tuple[Path, Path]:
    weights_path = _stage_complete_weights_path_v8(stage_dir, stage_name)
    meta_path = _stage_complete_meta_path_v8(stage_dir, stage_name)
    stage_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_clone_state_dict_cpu(model.state_dict()), weights_path)
    meta_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    return weights_path, meta_path


def _teacher_resume_dir_v7(repo_root: Path, seed: int, limit: int, teacher_keys: Sequence[str]) -> Path:
    teacher_tag = f"teachers_{len(teacher_keys)}"
    return (repo_root / "output" / "omni_v8_teacher_resume" / f"seed_{int(seed)}_limit_{int(limit)}_{teacher_tag}").resolve()


def _teacher_sample_path_v7(resume_dir: Path) -> Path:
    return resume_dir / "teacher_sample.jsonl"


def _teacher_manifest_path_v7(resume_dir: Path) -> Path:
    return resume_dir / "teacher_manifest.json"


def _teacher_state_path_v7(resume_dir: Path, teacher_key: str) -> Path:
    return resume_dir / f"{teacher_key}.json"


def _save_teacher_sample_v7(sample_path: Path, rows: Sequence[OmniRow]) -> None:
    _write_jsonl_atomic(sample_path, [_omni_row_to_payload(row) for row in rows])


def _load_teacher_sample_v7(sample_path: Path) -> List[OmniRow]:
    rows: List[OmniRow] = []
    if not sample_path.exists():
        return rows
    for line in sample_path.read_text(encoding="utf-8").splitlines():
        cooked = line.strip()
        if not cooked:
            continue
        try:
            payload = json.loads(cooked)
        except Exception:
            continue
        rows.append(_omni_row_from_payload(payload))
    return rows


def _load_teacher_state_v7(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "status": "missing",
            "rows": {},
            "empty_row_indices": set(),
            "processed_rows": 0,
        }
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "status": "corrupt",
            "rows": {},
            "empty_row_indices": set(),
            "processed_rows": 0,
        }
    rows_payload = payload.get("rows") or []
    rows: Dict[int, Tuple[float, str]] = {}
    for item in rows_payload:
        try:
            row_index = int(item.get("row_index"))
            rows[row_index] = (float(item.get("score") or 0.0), str(item.get("candidate") or ""))
        except Exception:
            continue
    empty_indices = {int(value) for value in (payload.get("empty_row_indices") or []) if str(value).isdigit()}
    return {
        "status": str(payload.get("status") or "partial"),
        "rows": rows,
        "empty_row_indices": empty_indices,
        "processed_rows": int(payload.get("processed_rows") or (len(rows) + len(empty_indices))),
        "teacher": str(payload.get("teacher") or ""),
        "sample_total": int(payload.get("sample_total") or 0),
    }


def _conversation_alignment_rows_v7() -> List[OmniRow]:
    base = list(_conversation_alignment_rows_v6())
    items = [
        (
            "The user wants a direct answer with no filler. What should the assistant optimize for?",
            "Optimize for directness, preserve the requested format, and remove stock framing that does not advance the task.",
            "general",
        ),
        (
            "Why is it useful for a local model to mention uncertainty instead of inventing details?",
            "Because an explicit uncertainty statement preserves trust and still moves the task forward, while invented details create confident but fragile failures.",
            "knowledge",
        ),
        (
            "What should a model do if a user asks for a benchmark comparison between two local models?",
            "Compare the actual recorded strengths, benchmark scores, and capabilities instead of guessing from names alone.",
            "model_selection",
        ),
        (
            "Answer in one short paragraph: why should a coding assistant stay close to the repo context?",
            "Staying close to the repo context keeps fixes compatible with the real code, avoids generic advice, and reduces the chance of proposing changes that do not fit the current implementation.",
            "coding",
        ),
    ]
    base.extend(
        [
            OmniRow(
                prompt=prompt,
                intent="model_selection" if domain == "model_selection" else ("coding" if domain == "coding" else "general"),
                response_text=_normalize(response, 420),
                domain=_safe_domain(domain),
                source="conversation_alignment_v7",
            )
            for prompt, response, domain in items
        ]
    )
    return base


def _grounding_rows_v7() -> List[OmniRow]:
    base = list(_grounding_rows_v6())
    items = [
        (
            "How should a grounded assistant answer a question about a local model it has never loaded?",
            "It should describe only the catalog facts it can verify and avoid claiming benchmark or runtime behavior it has not actually observed.",
            "knowledge",
        ),
        (
            "What is the safer answer when a request depends on a file that might not exist?",
            "Inspect or verify the file path first; if that cannot be done yet, say so instead of pretending the file is present.",
            "coding",
        ),
        (
            "Why is narrowing the answer often better than improvising extra details?",
            "A narrower answer stays closer to the evidence you have, which makes it more reliable than a broader answer padded with unsupported specifics.",
            "general",
        ),
        (
            "How should a local model behave when it is only weakly confident about a domain classification?",
            "It should default to a grounded general answer, keep claims bounded, and avoid overcommitting to a specialist tone it cannot support.",
            "planning",
        ),
    ]
    base.extend(
        [
            OmniRow(
                prompt=prompt,
                intent="coding" if domain == "coding" else "knowledge",
                response_text=_normalize(response, 420),
                domain=_safe_domain(domain),
                source="grounding_v7",
            )
            for prompt, response, domain in items
        ]
    )
    return base


def _conversation_expansion_rows_v7(repo_root: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    counts: Dict[str, int] = defaultdict(int)
    dataset_specs = [
        ("conversation_data.supermix_plus_v27_500k.jsonl", 3400, "knowledge", "conversation_supermix_plus_v8"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 1800, "planning", "conversation_reasoning_v8"),
        ("conversation_data.mega_creative_250k_v2.jsonl", 1750, "creative", "conversation_creative_v8"),
        ("conversation_data.mega_creative_100k.jsonl", 900, "creative", "conversation_creative_alt_v8"),
        ("conversation_data.book_extracts_public_domain_v2_120k.jsonl", 1200, "language", "conversation_books_v8"),
        ("conversation_data.book_extracts_smoke_v3.jsonl", 260, "language", "conversation_books_smoke_v8"),
        ("conversation_data.finnegans_wake_study_noninfringing_smoke.jsonl", 120, "language", "conversation_style_edge_v8"),
        ("conversation_data.hybrid_v6_live_knowledge.jsonl", 1500, "general", "conversation_hybrid_live_v8"),
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 780, "coding", "conversation_coding_v8"),
        ("conversation_data.dictionary_wordnet_meanings_smoke.jsonl", 320, "language", "conversation_dictionary_v8"),
        ("conversation_data.science_essentials_smoke.jsonl", 300, "knowledge", "conversation_science_v8"),
        ("conversation_data.science_novel_examples_smoke.jsonl", 260, "knowledge", "conversation_science_novel_v8"),
        ("conversation_data.world_events_2026_02_19.jsonl", 320, "knowledge", "conversation_world_events_v8"),
        ("conversation_data.english_math_smoke_v3.jsonl", 360, "math", "conversation_math_v8"),
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 260, "knowledge", "conversation_delta_anchor_v8"),
        ("conversation_data.delta_official_refresh_2026_03_26.jsonl", 160, "knowledge", "conversation_delta_refresh_v8"),
    ]
    for rel_name, limit, domain, source_tag in dataset_specs:
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(path, limit=limit, seed=seed + len(rows) + limit, domain=domain, source_tag=source_tag)
        rows.extend(sampled)
        counts[source_tag] += len(sampled)
    return rows, dict(sorted(counts.items()))


def _benchmax_rows_v7(repo_root: Path, seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    counts: Dict[str, int] = {}
    v33_rows, _ = build_v33_style_rows(repo_root, seed=seed + 31, sample_size=2304)
    v39_rows, _ = build_v39_style_rows(repo_root, seed=seed + 57, sample_size=2560)
    for source_tag, payload_rows in (("benchmax_v33_v8", v33_rows), ("benchmax_v39_v8", v39_rows)):
        local_count = 0
        for item in payload_rows:
            cooked = prompt_row_to_omni(item, source=source_tag)
            prompt = _normalize(cooked.get("prompt") or "", 260)
            response = _normalize(cooked.get("response_text") or "", 420)
            if len(prompt) < 8 or len(response) < 4:
                continue
            rows.append(
                OmniRow(
                    prompt=prompt,
                    intent=str(cooked.get("intent") or "general"),
                    response_text=response,
                    domain=_safe_domain(str(cooked.get("domain") or "general")),
                    source=source_tag,
                )
            )
            local_count += 1
        counts[source_tag] = local_count
    return rows, counts


def _record_score_fragment(repo_root: Path, record: ModelRecord) -> str:
    if record.common_overall_exact is not None:
        return f"Its common benchmark score is {record.common_overall_exact:.4f}."
    if record.recipe_eval_accuracy is not None:
        return f"Its internal validation score is {record.recipe_eval_accuracy:.4f}."
    if record.kind == "dcgan_image":
        for path in sorted((repo_root / "output").rglob(f"{record.key}_benchmark_summary.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            score = payload.get("specialist_score")
            if score is not None:
                return f"Its specialist GAN score is {float(score):.4f}."
    return ""


def _specialist_profile_rows_v7(
    repo_root: Path,
    models_dir: Path,
    seed: int,
    allowed_record_keys: Optional[Sequence[str]] = None,
) -> Tuple[List[OmniRow], Dict[str, object]]:
    del seed
    records = _sorted_model_records_v8(models_dir=models_dir, allowed_model_keys=allowed_record_keys)
    rows: List[OmniRow] = []
    family_map: Dict[str, List[str]] = defaultdict(list)
    non_chat_keys: List[str] = []
    for record in records:
        family_map[record.family].append(record.label)
        if not record.supports_chat:
            non_chat_keys.append(record.key)
        family_text = record.family.replace("_", " ")
        kind_text = record.kind.replace("_", " ")
        capability_text = ", ".join(sorted(record.capabilities)) or "specialist"
        score_text = _record_score_fragment(repo_root, record)
        use_text = str(record.benchmark_hint or record.note or f"{record.label} is a local specialist model.").strip()
        base_answer = _normalize(
            f"{record.label} is a {family_text} {kind_text} model with capabilities {capability_text}. "
            f"{record.note or ''} Best use: {use_text} {score_text}".strip(),
            420,
        )
        rows.append(
            OmniRow(
                prompt=f"What is {record.label} best used for in the local Supermix catalog?",
                intent="model_selection",
                response_text=base_answer,
                domain="model_selection",
                source=f"{record.key}_profile_v7",
            )
        )
        if record.supports_chat:
            response = _normalize(
                f"Choose {record.label} when the task matches its strongest local use case: {use_text} {score_text}".strip(),
                420,
            )
        else:
            response = _normalize(
                f"{record.label} is not a normal chat consultant. Use it through its specialist pipeline for {use_text.lower()} rather than expecting ordinary back-and-forth chat.",
                420,
            )
        rows.append(
            OmniRow(
                prompt=f"When should I choose {record.label} instead of a generic local chat model?",
                intent="model_selection",
                response_text=response,
                domain="model_selection",
                source=f"{record.key}_selection_v7",
            )
        )
        if record.kind == "dcgan_image":
            spec = DCGAN_SPECS.get(record.key)
            if spec is not None:
                rows.append(
                    OmniRow(
                        prompt=f"How does {record.label} behave when prompted with text?",
                        intent="model_selection",
                        response_text=_normalize(
                            f"{record.label} is an unconditional GAN. The text prompt only seeds latent sampling; it does not understand the prompt semantically like a chat model.",
                            420,
                        ),
                        domain="image_prompt",
                        source=f"{record.key}_gan_v7",
                    )
                )
        elif record.kind == "image_recognition":
            rows.append(
                OmniRow(
                    prompt=f"What kind of uploaded images is {record.label} strongest at?",
                    intent="vision",
                    response_text=_normalize(
                        f"{record.label} is strongest on the bundled science-diagram style categories such as {', '.join(name.replace('_', ' ') for name in SCIENCE_IMAGE_CLASSES[:5])}.",
                        420,
                    ),
                    domain="vision",
                    source=f"{record.key}_vision_v7",
                )
            )
        elif record.kind == "protein_folding":
            rows.append(
                OmniRow(
                    prompt="Which local model is the protein-folding specialist?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the protein-folding specialist in the current local model catalog.", 240),
                    domain="model_selection",
                    source=f"{record.key}_protein_v7",
                )
            )
        elif record.kind == "three_d_generation":
            rows.append(
                OmniRow(
                    prompt="Which local model is best for OpenSCAD, CAD, and small 3D geometry prompts?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the local 3D/OpenSCAD specialist for parametric geometry prompts.", 240),
                    domain="model_selection",
                    source=f"{record.key}_three_d_v8",
                )
            )
        elif record.kind == "mattergen_generation":
            rows.append(
                OmniRow(
                    prompt="Which local model is best for materials discovery, CIF-style seeds, and crystal design prompts?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the local materials-generation specialist for crystal and property-conditioned prototype prompts.", 260),
                    domain="model_selection",
                    source=f"{record.key}_mattergen_v8",
                )
            )
        elif record.kind == "math_equation":
            rows.append(
                OmniRow(
                    prompt="Which local model is best for exact symbolic math and equation solving?",
                    intent="model_selection",
                    response_text=_normalize(f"{record.label} is the exact symbolic math specialist in the local model catalog.", 240),
                    domain="model_selection",
                    source=f"{record.key}_math_v7",
                )
            )
    for family, labels in sorted(family_map.items()):
        label_text = ", ".join(sorted(labels))
        rows.append(
            OmniRow(
                prompt=f"Which local models belong to the {family.replace('_', ' ')} family?",
                intent="model_selection",
                response_text=_normalize(f"The {family.replace('_', ' ')} family currently includes: {label_text}.", 420),
                domain="model_selection",
                source=f"{family}_family_map_v7",
            )
        )
    if non_chat_keys:
        rows.append(
            OmniRow(
                prompt="Which local models are specialist generators or non-chat systems rather than normal chat assistants?",
                intent="model_selection",
                response_text=_normalize(
                    "The non-chat specialist models include: "
                    + ", ".join(sorted(non_chat_keys))
                    + ". Use them through image generation or specialist pipelines instead of normal text chat.",
                    420,
                ),
                domain="model_selection",
                source="non_chat_specialists_v7",
            )
        )
    return rows, {
        "record_keys": [record.key for record in records],
        "record_count": len(records),
        "non_chat_record_keys": sorted(non_chat_keys),
        "family_counts": {family: len(labels) for family, labels in sorted(family_map.items())},
        "row_count": len(rows),
    }


def _materials_rows_v8() -> List[OmniRow]:
    rows: List[OmniRow] = []
    for spec in MATTERGEN_CONCEPT_SPECS:
        prototype = dict(spec.get("prototype") or {})
        display = str(spec.get("display_name") or spec.get("concept") or "materials candidate")
        formula = str(prototype.get("formula") or "unknown formula")
        system = str(prototype.get("crystal_system") or "unknown system")
        space_group = str(prototype.get("space_group") or "unknown space group")
        focus = str(prototype.get("screening_focus") or "property screening")
        note = str(prototype.get("synthesis_note") or "validate synthesis constraints before trusting the candidate")
        rows.append(
            OmniRow(
                prompt=f"Generate a grounded materials-design answer for a {display} candidate.",
                intent="general",
                response_text=_normalize(
                    f"A grounded {display} seed can start from {formula} in the {system} system ({space_group}). "
                    f"Focus screening on {focus}, and keep the synthesis note in view: {note}.",
                    420,
                ),
                domain="knowledge",
                source="materials_grounding_v8",
            )
        )
        rows.append(
            OmniRow(
                prompt=f"What should a local model mention when proposing a CIF-style seed for {display}?",
                intent="knowledge",
                response_text=_normalize(
                    f"It should name the prototype family, a plausible formula like {formula}, the crystal system {system}, the space group {space_group}, and the property targets that still need verification rather than claiming full materials validation.",
                    420,
                ),
                domain="knowledge",
                source="materials_grounding_v8",
            )
        )
    return rows


def _three_d_rows_v8() -> List[OmniRow]:
    rows: List[OmniRow] = []
    for spec in THREE_D_GENERATION_SPECS:
        answers = dict(spec.get("answers") or {})
        display = str(spec.get("display_name") or spec.get("concept") or "3D object")
        code_answer = _normalize(str(answers.get("code") or answers.get("concise_answer") or ""), 420)
        concise_answer = _normalize(str(answers.get("concise_answer") or code_answer), 420)
        if code_answer:
            rows.append(
                OmniRow(
                    prompt=f"Write a grounded OpenSCAD answer for {display}.",
                    intent="coding",
                    response_text=code_answer,
                    domain="spatial_3d",
                    source="three_d_generation_v8",
                )
            )
        rows.append(
            OmniRow(
                prompt=f"What is the local 3D generation pattern for {display}?",
                intent="knowledge",
                response_text=_normalize(
                    f"For {display}, keep the answer parametric, valid OpenSCAD, and concise. A grounded local pattern is: {concise_answer}",
                    420,
                ),
                domain="spatial_3d",
                source="three_d_generation_v8",
            )
        )
    return rows


def _cross_modal_selection_rows_v8(models_dir: Path, allowed_record_keys: Optional[Sequence[str]] = None) -> List[OmniRow]:
    records = {
        record.key: record
        for record in _sorted_model_records_v8(models_dir=models_dir, allowed_model_keys=allowed_record_keys)
    }
    prompts_and_keys = [
        ("Which local model should I use for benchmark-focused reasoning prompts?", "v40_benchmax"),
        ("Which local model should I use for protein folding prompts?", "protein_folding_micro_v1"),
        ("Which local model should I use for OpenSCAD or 3D design prompts?", "three_d_generation_micro_v1"),
        ("Which local model should I use for materials discovery or CIF-style crystal prompts?", "mattergen_micro_v1"),
        ("Which local model should I use for uploaded science image recognition?", "science_vision_micro_v1"),
        ("Which local model should I use for exact symbolic math?", "math_equation_micro_v1"),
        ("Which local model should I use for unconditional digit-like image generation?", "dcgan_mnist_model"),
        ("Which local model should I use for the broadest all-model distilled chat ability?", "omni_collective_v7"),
    ]
    rows: List[OmniRow] = []
    for prompt, key in prompts_and_keys:
        record = records.get(key)
        if record is None:
            continue
        use_text = str(record.benchmark_hint or record.note or record.label).strip()
        rows.append(
            OmniRow(
                prompt=prompt,
                intent="model_selection",
                response_text=_normalize(f"Use {record.label} for that task. {use_text}", 320),
                domain="model_selection",
                source="cross_modal_selection_v8",
            )
        )
    return rows


def _gemma_four_slice_rows_v8(rows: Sequence[OmniRow], *, seed: int, limit: int) -> Tuple[List[OmniRow], Dict[str, object]]:
    model_target = str(os.environ.get("SUPERMIX_GEMMA4_MODEL_DIR") or os.environ.get("SUPERMIX_GEMMA4_MODEL_ID") or "").strip()
    if not model_target:
        return [], {"status": "skipped_no_local_gemma4", "requested": int(limit), "accepted": 0}
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        return [], {"status": "transformers_unavailable", "requested": int(limit), "accepted": 0, "error": f"{type(exc).__name__}: {exc}"}

    sample = _sample_teacher_rows(rows, seed=seed, limit=max(4, int(limit)))
    local_only = not Path(model_target).exists()
    accepted: List[OmniRow] = []
    summary: Dict[str, object] = {
        "status": "started",
        "requested": len(sample),
        "accepted": 0,
        "model_target": model_target,
        "local_files_only": bool(local_only),
    }
    try:  # pragma: no cover - heavy external path
        tokenizer = AutoTokenizer.from_pretrained(model_target, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(model_target, local_files_only=local_only, torch_dtype=torch.float32)
        model.eval()
        for row in sample:
            prompt = (
                "Answer the request directly, concisely, and with grounded uncertainty when needed.\n"
                f"Request: {row.prompt}\n"
                "Answer:"
            )
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=96,
                    pad_token_id=tokenizer.eos_token_id,
                )
            candidate = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if not candidate:
                continue
            score = float(_score_candidate(row.response_text, candidate)["composite"])
            if score < 0.08:
                continue
            accepted.append(
                OmniRow(
                    prompt=row.prompt,
                    intent=row.intent,
                    response_text=_normalize(candidate, 420),
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="gemma4_slice_v8",
                )
            )
        summary["status"] = "complete"
        summary["accepted"] = len(accepted)
        return accepted, summary
    except Exception as exc:
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return [], summary


def _teacher_order_key_v7(record_key: str) -> Tuple[int, str]:
    priority = {
        "v40_benchmax": 0,
        "omni_collective_v7": 1,
        "qwen_v28": 2,
        "qwen_v30": 3,
        "omni_collective_v6": 4,
        "omni_collective_v5": 5,
        "v33_final": 6,
        "omni_collective_v4": 7,
        "omni_collective_v3": 8,
    }
    return (priority.get(record_key, 20), record_key)


def _model_style_for_row_v7(row: OmniRow) -> str:
    if row.domain in {"coding", "model_selection"}:
        return "coding"
    if row.domain in {"creative", "image_prompt", "language"}:
        return "creative"
    return "analyst"


_teacher_resume_dir_v8 = _teacher_resume_dir_v7
_teacher_sample_path_v8 = _teacher_sample_path_v7
_teacher_manifest_path_v8 = _teacher_manifest_path_v7
_teacher_state_path_v8 = _teacher_state_path_v7
_save_teacher_sample_v8 = _save_teacher_sample_v7
_load_teacher_sample_v8 = _load_teacher_sample_v7
_load_teacher_state_v8 = _load_teacher_state_v7
_teacher_order_key_v8 = _teacher_order_key_v7
_model_style_for_row_v8 = _model_style_for_row_v7


def _teacher_worker_python() -> str:
    venv_python = Path(str(os.environ.get("VIRTUAL_ENV") or "")).resolve() / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_teacher_worker_v7(
    *,
    repo_root: Path,
    models_dir: Path,
    teacher_key: str,
    sample_path: Path,
    state_path: Path,
    stall_seconds: int = 1200,
    poll_seconds: int = 10,
) -> int:
    worker_script = SOURCE_DIR / "run_omni_collective_v8_teacher.py"
    cmd = [
        _teacher_worker_python(),
        "-u",
        str(worker_script),
        "--repo_root",
        str(repo_root),
        "--models_dir",
        str(models_dir),
        "--teacher_key",
        str(teacher_key),
        "--sample_jsonl",
        str(sample_path),
        "--state_json",
        str(state_path),
    ]
    process = subprocess.Popen(cmd, cwd=str(repo_root))
    last_state_mtime: Optional[float] = state_path.stat().st_mtime if state_path.exists() else None
    last_progress_ts = time.time()
    while True:
        return_code = process.poll()
        current_mtime = state_path.stat().st_mtime if state_path.exists() else None
        if current_mtime is not None and current_mtime != last_state_mtime:
            last_state_mtime = current_mtime
            last_progress_ts = time.time()
        if return_code is not None:
            return int(return_code)
        if time.time() - last_progress_ts > max(int(stall_seconds), 60):
            try:
                process.kill()
            except Exception:
                pass
            try:
                process.wait(timeout=30)
            except Exception:
                pass
            return 124
        time.sleep(max(int(poll_seconds), 1))


def _aggregate_teacher_states_v7(
    teacher_states: Dict[str, Dict[str, Any]],
    sample_total: int,
) -> Tuple[Dict[int, Tuple[float, str, str]], Dict[int, List[Tuple[float, str, str]]], Dict[str, int], List[str], List[str]]:
    best_by_index: Dict[int, Tuple[float, str, str]] = {}
    candidates_by_index: Dict[int, List[Tuple[float, str, str]]] = defaultdict(list)
    empty_counts: Dict[str, int] = {}
    complete_teachers: List[str] = []
    partial_teachers: List[str] = []
    for teacher_key, state in teacher_states.items():
        rows = state.get("rows") or {}
        empty_indices = state.get("empty_row_indices") or set()
        empty_counts[teacher_key] = len(empty_indices)
        if str(state.get("status") or "") == "complete":
            complete_teachers.append(teacher_key)
        elif rows or empty_indices:
            partial_teachers.append(teacher_key)
        for row_index, payload in rows.items():
            score, candidate = payload
            candidates_by_index[int(row_index)].append((float(score), teacher_key, str(candidate)))
            best = best_by_index.get(int(row_index))
            if best is None or float(score) > best[0]:
                best_by_index[int(row_index)] = (float(score), teacher_key, str(candidate))
    for teacher_key in empty_counts:
        empty_counts[teacher_key] = min(sample_total, max(0, int(empty_counts[teacher_key])))
    return best_by_index, candidates_by_index, dict(sorted(empty_counts.items())), sorted(complete_teachers), sorted(partial_teachers)


def _all_model_distill_rows_v7(
    rows: Sequence[OmniRow],
    *,
    repo_root: Path,
    models_dir: Path,
    seed: int,
    limit: int,
    teacher_model_limit: int,
    teacher_keys_override: Optional[Sequence[str]] = None,
    resume_dir_override: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, object]]:
    sample = _sample_teacher_rows(rows, seed=seed, limit=limit)
    if not str(os.environ.get("SUPERMIX_QWEN_BASE_MODEL_DIR") or "").strip():
        try:
            os.environ["SUPERMIX_QWEN_BASE_MODEL_DIR"] = str(_resolve_local_qwen_base_model())
        except Exception:
            pass
    all_records = _sorted_model_records_v8(models_dir=models_dir)
    teacher_records = [record for record in all_records if record.supports_chat and record.key != "omni_collective_v8"]
    if teacher_keys_override:
        teacher_order = {str(key): idx for idx, key in enumerate(teacher_keys_override)}
        teacher_records = [record for record in teacher_records if record.key in teacher_order]
        teacher_records.sort(key=lambda item: teacher_order[item.key])
    else:
        teacher_records.sort(key=lambda item: _teacher_order_key_v7(item.key))
        if teacher_model_limit > 0:
            teacher_records = teacher_records[: int(teacher_model_limit)]

    if int(limit) <= 0 or not sample:
        return [], {
            "requested": int(limit),
            "sampled": 0,
            "accepted_total": 0,
            "accepted_direct": {},
            "accepted_repair": {},
            "accepted_consensus": 0,
            "empty_counts": {},
            "discarded": 0,
            "teacher_keys": [record.key for record in teacher_records],
            "all_record_keys": [record.key for record in all_records],
            "non_chat_record_keys": [record.key for record in all_records if not record.supports_chat],
            "unavailable_teachers": {},
            "timed_out_teachers": [],
            "complete_teachers": [],
            "partial_teachers": [],
            "resume_dir": "",
            "forced_priority_keys": ["v40_benchmax", "omni_collective_v7", "qwen_v28", "qwen_v30", "omni_collective_v6"],
        }

    resume_dir = (
        Path(resume_dir_override).resolve()
        if resume_dir_override is not None
        else _teacher_resume_dir_v7(repo_root=repo_root, seed=seed, limit=limit, teacher_keys=[record.key for record in teacher_records])
    )
    resume_dir.mkdir(parents=True, exist_ok=True)
    sample_path = _teacher_sample_path_v7(resume_dir)
    if sample_path.exists():
        cached_sample = _load_teacher_sample_v7(sample_path)
        if cached_sample:
            sample = cached_sample
    else:
        _save_teacher_sample_v7(sample_path, sample)
    _write_json_atomic(
        _teacher_manifest_path_v7(resume_dir),
        {
            "seed": int(seed),
            "limit": int(limit),
            "sample_total": len(sample),
            "teacher_keys": [record.key for record in teacher_records],
        },
    )

    direct_counts: Dict[str, int] = defaultdict(int)
    repair_counts: Dict[str, int] = defaultdict(int)
    unavailable_teachers: Dict[str, str] = {}
    timed_out_teachers: List[str] = []
    teacher_states: Dict[str, Dict[str, Any]] = {}

    for teacher_idx, record in enumerate(teacher_records, start=1):
        state_path = _teacher_state_path_v7(resume_dir, record.key)
        teacher_states[record.key] = _load_teacher_state_v7(state_path)
        if teacher_states[record.key].get("status") == "complete":
            print(
                json.dumps(
                    {
                        "event": "teacher_model_resume_hit",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            continue
        print(
            json.dumps(
                {
                    "event": "teacher_model_start",
                    "teacher_index": teacher_idx,
                    "teacher_total": len(teacher_records),
                    "teacher": record.key,
                    "resume_processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return_code = _run_teacher_worker_v7(
            repo_root=repo_root,
            models_dir=models_dir,
            teacher_key=record.key,
            sample_path=sample_path,
            state_path=state_path,
        )
        teacher_states[record.key] = _load_teacher_state_v7(state_path)
        if return_code == 124:
            timed_out_teachers.append(record.key)
            print(
                json.dumps(
                    {
                        "event": "teacher_model_timeout",
                        "teacher_index": teacher_idx,
                        "teacher_total": len(teacher_records),
                        "teacher": record.key,
                        "processed_rows": int(teacher_states[record.key].get("processed_rows") or 0),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        if return_code != 0 and not teacher_states[record.key].get("rows") and not teacher_states[record.key].get("empty_row_indices"):
            unavailable_teachers[record.key] = f"worker_exit_{return_code}"
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

    best_by_index, candidates_by_index, empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample),
    )

    accepted: List[OmniRow] = []
    consensus_rows: List[OmniRow] = []
    discarded = 0
    for row_index, row in enumerate(sample, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        best = best_by_index.get(row_index)
        if best is None:
            discarded += 1
            continue
        score, teacher_key, candidate = best
        if score >= 0.26:
            accepted.append(
                OmniRow(
                    prompt=row.prompt,
                    intent=row.intent,
                    response_text=_normalize(candidate, 420),
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_distill_v7",
                )
            )
            direct_counts[teacher_key] += 1
        elif score >= 0.12:
            accepted.append(
                OmniRow(
                    prompt=_normalize(
                        "Repair and ground this draft answer so it becomes concise, correct, and less speculative.\n"
                        f"Request: {row.prompt}\n"
                        f"Draft: {candidate}",
                        360,
                    ),
                    intent=row.intent,
                    response_text=row.response_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source=f"{teacher_key}_repair_v7",
                )
            )
            repair_counts[teacher_key] += 1
        else:
            discarded += 1

        if len(candidates) >= 2 and len(consensus_rows) < max(18, int(limit) // 3):
            first, second = candidates[0], candidates[1]
            if first[0] >= 0.18 and second[0] >= 0.18 and first[2].strip().lower() != second[2].strip().lower():
                consensus_rows.append(
                    OmniRow(
                        prompt=_normalize(
                            "Synthesize the strongest grounded answer from these teacher drafts.\n"
                            f"Request: {row.prompt}\n"
                            f"Draft A ({first[1]}): {first[2]}\n"
                            f"Draft B ({second[1]}): {second[2]}",
                            360,
                        ),
                        intent=row.intent,
                        response_text=row.response_text,
                        domain=row.domain,
                        image_path=row.image_path,
                        vision_label=row.vision_label,
                        source="teacher_consensus_v7",
                    )
                )

        if row_index % 8 == 0 or row_index == len(sample):
            print(
                json.dumps(
                    {
                        "event": "teacher_league_progress",
                        "completed": row_index,
                        "total": len(sample),
                        "accepted": len(accepted),
                        "consensus": len(consensus_rows),
                        "discarded": discarded,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
    gc.collect()
    accepted.extend(consensus_rows)
    return accepted, {
        "requested": int(limit),
        "sampled": len(sample),
        "accepted_total": len(accepted),
        "accepted_direct": dict(sorted(direct_counts.items())),
        "accepted_repair": dict(sorted(repair_counts.items())),
        "accepted_consensus": len(consensus_rows),
        "empty_counts": dict(sorted(empty_counts.items())),
        "discarded": discarded,
        "teacher_keys": [record.key for record in teacher_records],
        "all_record_keys": [record.key for record in all_records],
        "non_chat_record_keys": [record.key for record in all_records if not record.supports_chat],
        "unavailable_teachers": dict(sorted(unavailable_teachers.items())),
        "timed_out_teachers": sorted(timed_out_teachers),
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "resume_dir": str(resume_dir),
        "forced_priority_keys": ["v40_benchmax", "omni_collective_v7", "qwen_v28", "qwen_v30", "omni_collective_v6"],
    }


def build_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
    frozen_dataset_summary: Optional[Dict[str, object]] = None,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, object]]:
    allowed_model_keys: Optional[List[str]] = None
    teacher_keys_override: Optional[List[str]] = None
    teacher_resume_dir_override: Optional[Path] = None
    if isinstance(frozen_dataset_summary, dict):
        specialist_payload = frozen_dataset_summary.get("specialist_profiles")
        if isinstance(specialist_payload, dict):
            record_keys = specialist_payload.get("record_keys")
            if isinstance(record_keys, list) and record_keys:
                allowed_model_keys = [str(key) for key in record_keys]
        teacher_payload = frozen_dataset_summary.get("teacher_league")
        if isinstance(teacher_payload, dict):
            teacher_keys = teacher_payload.get("teacher_keys")
            if isinstance(teacher_keys, list) and teacher_keys:
                teacher_keys_override = [str(key) for key in teacher_keys]
            resume_dir_value = str(teacher_payload.get("resume_dir") or "").strip()
            if resume_dir_value:
                teacher_resume_dir_override = Path(resume_dir_value).resolve()

    full_rows, source_counts = _build_rows_v6(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        allowed_model_keys=allowed_model_keys,
    )
    conversation_rows, conversation_counts = _conversation_expansion_rows_v7(repo_root=repo_root, seed=seed + 601)
    benchmax_rows, benchmax_counts = _benchmax_rows_v7(repo_root=repo_root, seed=seed + 641)
    specialist_rows, specialist_summary = _specialist_profile_rows_v7(
        repo_root=repo_root,
        models_dir=models_dir,
        seed=seed + 659,
        allowed_record_keys=allowed_model_keys,
    )
    materials_rows = _materials_rows_v8()
    three_d_rows = _three_d_rows_v8()
    cross_modal_rows = _cross_modal_selection_rows_v8(models_dir=models_dir, allowed_record_keys=allowed_model_keys)
    grounding_rows = _grounding_rows_v7()
    alignment_rows = _conversation_alignment_rows_v7()
    math_rows = _math_exact_rows_v6(repo_root=repo_root, seed=seed + 677, limit=132)
    protein_rows = _protein_rows_v6(seed=seed + 701, limit=300)
    protein_pack_rows, protein_pack_summary = build_protein_folding_rows(seed=seed + 727, max_rows=320)
    protein_pack_cooked = [
        OmniRow(
            prompt=_normalize(str(item.get("prompt") or ""), 260),
            intent=str(item.get("intent") or "knowledge"),
            response_text=_normalize(str(item.get("response_text") or item.get("assistant") or ""), 420),
            domain=_safe_domain(str(item.get("domain") or "knowledge")),
            source="protein_pack_v7",
        )
        for item in protein_pack_rows
        if len(_normalize(str(item.get("prompt") or ""), 260)) >= 8 and len(_normalize(str(item.get("response_text") or item.get("assistant") or ""), 420)) >= 8
    ]

    full_rows.extend(conversation_rows)
    full_rows.extend(benchmax_rows)
    full_rows.extend(specialist_rows)
    full_rows.extend(materials_rows)
    full_rows.extend(three_d_rows)
    full_rows.extend(cross_modal_rows)
    full_rows.extend(grounding_rows)
    full_rows.extend(alignment_rows)
    full_rows.extend(math_rows)
    full_rows.extend(protein_rows)
    full_rows.extend(protein_pack_cooked)

    for key, value in conversation_counts.items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    for key, value in benchmax_counts.items():
        source_counts[key] = source_counts.get(key, 0) + int(value)
    source_counts["specialist_profiles_v7"] = len(specialist_rows)
    source_counts["materials_grounding_v8"] = len(materials_rows)
    source_counts["three_d_generation_v8"] = len(three_d_rows)
    source_counts["cross_modal_selection_v8"] = len(cross_modal_rows)
    source_counts["grounding_v7"] = len([row for row in grounding_rows if row.source == "grounding_v7"])
    source_counts["conversation_alignment_v7"] = len([row for row in alignment_rows if row.source == "conversation_alignment_v7"])
    source_counts["math_exact_v7_added"] = len(math_rows)
    source_counts["protein_folding_v7_added"] = len(protein_rows)
    source_counts["protein_pack_v7"] = len(protein_pack_cooked)

    distill_rows, distill_summary = _all_model_distill_rows_v7(
        full_rows,
        repo_root=repo_root,
        models_dir=models_dir,
        seed=seed + 787,
        limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
        teacher_keys_override=teacher_keys_override,
        resume_dir_override=teacher_resume_dir_override,
    )
    full_rows.extend(distill_rows)
    source_counts["all_model_distill_total_v7"] = len(distill_rows)
    gemma_rows, gemma_summary = _gemma_four_slice_rows_v8(full_rows, seed=seed + 829, limit=max(6, int(distill_limit) // 10))
    full_rows.extend(gemma_rows)
    source_counts["gemma4_slice_v8"] = len(gemma_rows)

    pre_dedupe_rows = len(full_rows)
    full_rows = _dedupe_rows(full_rows)
    rng = random.Random(int(seed))
    rng.shuffle(full_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    return stage1_rows, list(full_rows), {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(full_rows),
        "pre_dedupe_rows": pre_dedupe_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "teacher_league": distill_summary,
        "gemma_slice": gemma_summary,
        "specialist_profiles": specialist_summary,
        "protein_pack_summary": protein_pack_summary,
    }


def _train_stage_resumable_v8(
    *,
    model: OmniCollectiveNetV8,
    forward_model: torch.nn.Module,
    train_rows: Sequence[OmniRow],
    val_rows: Sequence[OmniRow],
    vocab: Dict[str, int],
    response_bank: Sequence[str],
    image_size: int,
    max_len: int,
    max_words: int,
    word_buckets: int,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    seed: int,
    runtime: TrainingRuntime,
    loss_weights: Dict[str, float],
    balance_weight: float,
    stage_name: str,
    stage_dir: Path,
    run_state_path: Path,
    progress_every_batches: Optional[int] = None,
    checkpoint_every_batches: Optional[int] = None,
    checkpoint_every_seconds: int = 1200,
    grad_accum_steps: int = 1,
    inject_interrupt_after_batch: Optional[int] = None,
) -> Dict[str, object]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    complete_weights_path = _stage_complete_weights_path_v8(stage_dir, stage_name)
    complete_meta_path = _stage_complete_meta_path_v8(stage_dir, stage_name)
    progress_path = _stage_progress_path_v8(stage_dir, stage_name)

    if complete_weights_path.exists() and complete_meta_path.exists():
        model.load_state_dict(torch.load(complete_weights_path, map_location="cpu", weights_only=True))
        complete_payload = _load_json_if_exists(complete_meta_path)
        if isinstance(complete_payload, dict):
            print(
                json.dumps(
                    {
                        "event": "stage_resume_complete_hit",
                        "stage": stage_name,
                        "weights_path": str(complete_weights_path),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            _write_run_state_v8(
                run_state_path,
                {
                    "status": "stage_complete_reused",
                    "stage": stage_name,
                    "weights_path": str(complete_weights_path),
                    "meta_path": str(complete_meta_path),
                    "result": complete_payload,
                },
            )
            return complete_payload

    response_to_index = {text: idx for idx, text in enumerate(response_bank)}
    train_ds = OmniDatasetV2(
        train_rows,
        vocab=vocab,
        max_len=max_len,
        image_size=image_size,
        max_words=max_words,
        word_buckets=word_buckets,
        response_to_index=response_to_index,
    )
    val_ds = OmniDatasetV2(
        val_rows,
        vocab=vocab,
        max_len=max_len,
        image_size=image_size,
        max_words=max_words,
        word_buckets=word_buckets,
        response_to_index=response_to_index,
    )
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False)
    total_batches = max(1, len(train_loader))
    grad_accum = max(1, int(grad_accum_steps))
    optimizer_steps_per_epoch = max(1, (total_batches + grad_accum - 1) // grad_accum)
    total_optimizer_steps = max(1, int(epochs) * optimizer_steps_per_epoch)
    progress_every = max(25, total_batches // 24) if progress_every_batches is None else max(1, int(progress_every_batches))
    checkpoint_every = max(120, total_batches // 8) if checkpoint_every_batches is None else max(1, int(checkpoint_every_batches))

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.022)
    scheduler = create_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=runtime.warmup_steps,
        warmup_ratio=runtime.warmup_ratio,
        min_lr_scale=runtime.min_lr_scale,
    )
    intent_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.03)
    response_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    vision_loss_fn = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.01)
    domain_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.02)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=bool(runtime.scaler_enabled))
    else:  # pragma: no cover
        scaler = torch.cuda.amp.GradScaler(enabled=bool(runtime.scaler_enabled))
    ema = ModelEma(model, runtime.ema_decay) if float(runtime.ema_decay) > 0.0 else None

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -1.0
    history: List[Dict[str, float]] = []
    start_epoch = 1
    start_batch_index = 1
    total_loss = 0.0
    total_items = 0
    resumed_from_progress = False
    optimizer_steps_done = 0
    avg_balance_loss = 0.0
    balance_count = 0

    checkpoint, checkpoint_source = _load_stage_progress_checkpoint_v8(progress_path)
    if checkpoint is not None:
        if str(checkpoint.get("stage_name") or "") == stage_name:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer_steps_done = _restore_scheduler_state_v8(
                scheduler,
                checkpoint.get("scheduler_state"),
                optimizer_steps_done=int(checkpoint.get("optimizer_steps_done") or 0),
                stage_name=stage_name,
                checkpoint_source=checkpoint_source,
            )
            if checkpoint.get("best_state") is not None:
                best_state = checkpoint["best_state"]
            if ema is not None:
                if checkpoint.get("ema_state") is not None:
                    ema.load_state_dict(checkpoint["ema_state"])
                else:
                    ema = ModelEma(model, runtime.ema_decay)
            best_score = float(checkpoint.get("best_score", -1.0))
            history = [dict(item) for item in (checkpoint.get("history") or [])]
            start_epoch = max(1, int(checkpoint.get("epoch") or 1))
            start_batch_index = max(1, int(checkpoint.get("next_batch_index") or 1))
            total_loss = float(checkpoint.get("total_loss") or 0.0)
            total_items = int(checkpoint.get("total_items") or 0)
            resumed_from_progress = True
            print(
                json.dumps(
                    {
                        "event": "stage_progress_resume_hit",
                        "stage": stage_name,
                        "epoch": start_epoch,
                        "next_batch_index": start_batch_index,
                        "total_batches": total_batches,
                        "checkpoint_path": str(progress_path),
                        "checkpoint_source": str(checkpoint_source) if checkpoint_source is not None else str(progress_path),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    print(
        json.dumps(
            {
                "event": "stage_start",
                "stage": stage_name,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "epochs": int(epochs),
                "loss_weights": loss_weights,
                "balance_weight": float(balance_weight),
                "resumed_from_progress": bool(resumed_from_progress),
                "device": runtime.resolved_device,
                "amp_enabled": bool(runtime.amp_enabled),
                "amp_dtype": runtime.amp_dtype_name,
                "grad_accum_steps": grad_accum,
                "effective_batch_size": int(batch_size) * grad_accum,
                "ema_decay": float(runtime.ema_decay),
                "warmup_steps": int(runtime.warmup_steps),
                "warmup_ratio": float(runtime.warmup_ratio),
                "compile_enabled": bool(runtime.compile_enabled),
                "compile_mode": runtime.compile_mode,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    _write_run_state_v8(
        run_state_path,
        {
            "status": "stage_running",
            "stage": stage_name,
            "epoch": start_epoch,
            "next_batch_index": start_batch_index,
            "total_batches": total_batches,
            "checkpoint_path": str(progress_path),
            "complete_weights_path": str(complete_weights_path),
            "complete_meta_path": str(complete_meta_path),
            "resumed_from_progress": bool(resumed_from_progress),
            "runtime": runtime.to_payload(),
        },
    )

    stage_wall_start = time.time()
    processed_batches_this_stage = 0
    last_progress_log = time.time()
    last_checkpoint_save = time.time()
    for epoch in range(start_epoch, int(epochs) + 1):
        if epoch != start_epoch:
            start_batch_index = 1
            total_loss = 0.0
            total_items = 0
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for batch_index, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            token_ids = batch["token_ids"].to(runtime.device)
            image_tensor = batch["image_tensor"].to(runtime.device)
            has_image = batch["has_image"].to(runtime.device)
            word_ids = batch["word_ids"].to(runtime.device)
            prompt_features = batch["prompt_features"].to(runtime.device)
            intent_targets = batch["intent"].to(runtime.device)
            response_targets = batch["response"].to(runtime.device)
            domain_targets = batch["domain"].to(runtime.device)
            vision_targets = batch["vision"].to(runtime.device)
            with torch.amp.autocast(
                device_type="cuda" if runtime.device_type == "cuda" else "cpu",
                dtype=runtime.amp_dtype,
                enabled=bool(runtime.amp_enabled and runtime.amp_dtype is not None),
            ):
                outputs = forward_model(
                    token_ids,
                    image_tensor,
                    has_image,
                    word_ids,
                    prompt_features,
                )
                raw_loss = (
                    float(loss_weights["intent"]) * intent_loss_fn(outputs["intent"], intent_targets)
                    + float(loss_weights["response"]) * response_loss_fn(outputs["response"], response_targets)
                    + float(loss_weights["domain"]) * domain_loss_fn(outputs["domain"], domain_targets)
                )
                if bool(vision_targets.ge(0).any()):
                    raw_loss = raw_loss + float(loss_weights["vision"]) * vision_loss_fn(outputs["vision"], vision_targets)
                if "balance_loss" in outputs:
                    raw_loss = raw_loss + float(balance_weight) * outputs["balance_loss"]
                    avg_balance_loss += float(outputs["balance_loss"].detach().item())
                    balance_count += 1
                loss = raw_loss / float(grad_accum)
            if bool(runtime.scaler_enabled):
                scaler.scale(loss).backward()
            else:
                loss.backward()
            should_step = (batch_index % grad_accum == 0) or (batch_index == total_batches)
            if should_step:
                if bool(runtime.scaler_enabled):
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if bool(runtime.scaler_enabled):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps_done += 1
                if ema is not None:
                    ema.update(model)
            batch_items = int(batch["intent"].size(0))
            total_loss += float(raw_loss.detach().item()) * batch_items
            total_items += batch_items
            processed_batches_this_stage += 1

            now_ts = time.time()
            if (
                batch_index == total_batches
                or batch_index == 1
                or batch_index % progress_every == 0
                or now_ts - last_progress_log >= 90
            ):
                elapsed = max(now_ts - stage_wall_start, 1e-6)
                completed_batches = max(1, (epoch - 1) * total_batches + batch_index)
                total_stage_batches = max(1, int(epochs) * total_batches)
                rate = completed_batches / elapsed
                remaining_batches = max(0, total_stage_batches - completed_batches)
                eta_seconds = remaining_batches / rate if rate > 1e-9 else None
                avg_loss = float(total_loss / max(total_items, 1))
                print(
                    json.dumps(
                        {
                            "event": "batch_progress",
                            "stage": stage_name,
                            "epoch": int(epoch),
                            "batch_index": int(batch_index),
                            "total_batches": int(total_batches),
                            "stage_progress_percent": round((completed_batches / total_stage_batches) * 100.0, 3),
                            "avg_train_loss": avg_loss,
                            "lr": float(scheduler.get_last_lr()[0]),
                            "optimizer_steps": optimizer_steps_done,
                            "avg_balance_loss": None if balance_count == 0 else round(avg_balance_loss / balance_count, 6),
                            "elapsed_seconds": round(elapsed, 3),
                            "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
                _write_run_state_v8(
                    run_state_path,
                    {
                        "status": "stage_running",
                        "stage": stage_name,
                        "epoch": int(epoch),
                        "batch_index": int(batch_index),
                        "total_batches": int(total_batches),
                        "avg_train_loss": avg_loss,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "optimizer_steps": optimizer_steps_done,
                        "avg_balance_loss": None if balance_count == 0 else round(avg_balance_loss / balance_count, 6),
                        "elapsed_seconds": round(elapsed, 3),
                        "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                        "checkpoint_path": str(progress_path),
                        "complete_weights_path": str(complete_weights_path),
                        "complete_meta_path": str(complete_meta_path),
                    },
                )
                last_progress_log = now_ts

            if batch_index < total_batches and (
                batch_index % checkpoint_every == 0 or now_ts - last_checkpoint_save >= max(int(checkpoint_every_seconds), 60)
            ):
                _save_stage_progress_checkpoint_v8(
                    checkpoint_path=progress_path,
                    stage_name=stage_name,
                    epoch=epoch,
                    next_batch_index=batch_index + 1,
                    total_batches=total_batches,
                    total_loss=total_loss,
                    total_items=total_items,
                    optimizer_steps_done=optimizer_steps_done,
                    best_score=best_score,
                    best_state=best_state,
                    ema_state=ema.state_dict() if ema is not None else None,
                    history=history,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_best_state=False,
                    save_ema_state=False,
                )
                last_checkpoint_save = now_ts

            if inject_interrupt_after_batch is not None and processed_batches_this_stage >= int(inject_interrupt_after_batch):
                _save_stage_progress_checkpoint_v8(
                    checkpoint_path=progress_path,
                    stage_name=stage_name,
                    epoch=epoch,
                    next_batch_index=min(batch_index + 1, total_batches),
                    total_batches=total_batches,
                    total_loss=total_loss,
                    total_items=total_items,
                    optimizer_steps_done=optimizer_steps_done,
                    best_score=best_score,
                    best_state=best_state,
                    ema_state=ema.state_dict() if ema is not None else None,
                    history=history,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    save_best_state=False,
                    save_ema_state=False,
                )
                raise RuntimeError("Injected stage interruption for tests.")

        if ema is not None:
            with ema.apply_to(model):
                val_metrics = evaluate(model, val_loader, runtime.device)
                candidate_best_state = _clone_state_dict_cpu(model.state_dict())
        else:
            val_metrics = evaluate(model, val_loader, runtime.device)
            candidate_best_state = _clone_state_dict_cpu(model.state_dict())
        score = _weighted_score(val_metrics)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(total_loss / max(total_items, 1)),
            "val_intent_accuracy": val_metrics["intent_accuracy"],
            "val_response_accuracy": val_metrics["response_accuracy"],
            "val_vision_accuracy": val_metrics["vision_accuracy"],
            "val_domain_accuracy": val_metrics["domain_accuracy"],
            "score": score,
            "lr": float(scheduler.get_last_lr()[0]),
            "optimizer_steps": float(optimizer_steps_done),
            "avg_balance_loss": float(avg_balance_loss / balance_count) if balance_count else 0.0,
            "ema_enabled": 1.0 if ema is not None else 0.0,
        }
        history.append(epoch_record)
        print(json.dumps({"event": "epoch_end", "stage": stage_name, **epoch_record}, ensure_ascii=True), flush=True)
        if score >= best_score:
            best_score = score
            best_state = candidate_best_state
        if epoch < int(epochs):
            _save_stage_progress_checkpoint_v8(
                checkpoint_path=progress_path,
                stage_name=stage_name,
                epoch=epoch + 1,
                next_batch_index=1,
                total_batches=total_batches,
                total_loss=0.0,
                total_items=0,
                optimizer_steps_done=optimizer_steps_done,
                best_score=best_score,
                best_state=best_state,
                ema_state=ema.state_dict() if ema is not None else None,
                history=history,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                save_best_state=True,
                save_ema_state=True,
            )
            last_checkpoint_save = time.time()

    if best_state is None:
        raise RuntimeError(f"No model state captured during {stage_name}.")
    model.load_state_dict(best_state)
    result = {
        "history": history,
        "best_score": best_score,
        "val_metrics": evaluate(model, val_loader, runtime.device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "runtime": runtime.to_payload(),
        "effective_batch_size": int(batch_size) * grad_accum,
    }
    weights_path, meta_path = _save_stage_complete_v8(stage_dir=stage_dir, stage_name=stage_name, model=model, result=result)
    if progress_path.exists():
        progress_path.unlink()
    _write_run_state_v8(
        run_state_path,
        {
            "status": "stage_complete",
            "stage": stage_name,
            "weights_path": str(weights_path),
            "meta_path": str(meta_path),
            "best_score": float(best_score),
            "val_metrics": result["val_metrics"],
        },
    )
    return result


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
    requested_device: str,
    amp_mode: str,
    amp_dtype: str,
    compile_model: bool,
    compile_mode: str,
    grad_accum_steps: int,
    ema_decay: float,
    warmup_steps: int,
    warmup_ratio: float,
    min_lr_scale: float,
) -> Dict[str, object]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_resume_dir = _stage_resume_dir_v8(
        output_dir,
        seed=int(seed),
        distill_limit=int(distill_limit),
        teacher_model_limit=int(teacher_model_limit),
    )
    stage_resume_dir.mkdir(parents=True, exist_ok=True)
    frozen_dataset_summary = _load_frozen_dataset_summary_v8(stage_resume_dir)
    run_state_path = _run_state_path_v8(output_dir)
    _write_run_state_v8(
        run_state_path,
        {
            "status": "building_dataset_from_resume" if frozen_dataset_summary is not None else "building_dataset",
            "stage": "dataset",
            "resume_dir": str(stage_resume_dir),
            "frozen_dataset": bool(frozen_dataset_summary is not None),
        },
    )
    _stage1_rows, full_rows, dataset_summary = build_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        distill_limit=distill_limit,
        teacher_model_limit=teacher_model_limit,
        frozen_dataset_summary=frozen_dataset_summary,
    )
    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    _write_json_atomic(stage_resume_dir / "dataset_summary.json", {"dataset_summary": dataset_summary})
    _write_run_state_v8(
        run_state_path,
        {
            "status": "dataset_built",
            "stage": "dataset",
            "resume_dir": str(stage_resume_dir),
            "dataset_summary": dataset_summary,
        },
    )
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
    _write_run_state_v8(
        run_state_path,
        {
            "status": "label_space_ready",
            "stage": "label_space",
            "resume_dir": str(stage_resume_dir),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "response_bank": len(response_bank),
            "vocab_size": len(vocab),
        },
    )

    max_len = 384
    max_words = 84
    word_buckets = 16384
    runtime = resolve_training_runtime(
        repo_root=repo_root,
        requested_device=requested_device,
        amp_mode=amp_mode,
        amp_dtype=amp_dtype,
        compile_requested=compile_model,
        compile_mode=compile_mode,
        grad_accum_steps=grad_accum_steps,
        ema_decay=ema_decay,
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        min_lr_scale=min_lr_scale,
        batch_size=batch_size,
    )
    print(json.dumps({"event": "runtime_config", "runtime": runtime.to_payload()}, ensure_ascii=True), flush=True)
    model = OmniCollectiveNetV8(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=132,
        text_hidden=272,
        image_channels=56,
        word_buckets=word_buckets,
        word_embed_dim=120,
        deep_text_channels=384,
        deep_image_channels=128,
        fusion_hidden=1088,
        memory_slots=28,
        depth_steps=11,
        expert_count=10,
        expert_hidden=1792,
        context_top_k=4,
        expert_top_k=2,
    ).to(runtime.device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    forward_model, runtime = maybe_compile_model(model, runtime)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)
    _write_run_state_v8(
        run_state_path,
        {
            "status": "warm_start_complete",
            "stage": "warm_start",
            "resume_dir": str(stage_resume_dir),
            "warm_start": warm_start,
            "runtime": runtime.to_payload(),
        },
    )

    stage1 = _train_stage_resumable_v8(
        model=model,
        forward_model=forward_model,
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
        runtime=runtime,
        loss_weights={"intent": 0.58, "response": 1.00, "domain": 0.82, "vision": 0.78},
        balance_weight=0.036,
        stage_name="stage1",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        grad_accum_steps=grad_accum_steps,
    )
    stage2 = _train_stage_resumable_v8(
        model=model,
        forward_model=forward_model,
        train_rows=train_rows,
        val_rows=val_rows,
        vocab=vocab,
        response_bank=response_bank,
        image_size=image_size,
        max_len=max_len,
        max_words=max_words,
        word_buckets=word_buckets,
        batch_size=max(10, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        runtime=runtime,
        loss_weights={"intent": 0.54, "response": 1.00, "domain": 0.80, "vision": 1.06},
        balance_weight=0.050,
        stage_name="stage2",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        grad_accum_steps=grad_accum_steps,
    )

    stamp = datetime.now().strftime("%Y%m%d")
    artifact_dir = output_dir / f"supermix_omni_collective_v8_frontier_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "omni_collective_v8_frontier.pth"
    meta_path = artifact_dir / "omni_collective_v8_frontier_meta.json"
    summary_path = artifact_dir / "omni_collective_v8_frontier_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v8_frontier_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    class_info = ensure_base_images(images_dir)
    meta = {
        "architecture_version": 8,
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": class_info,
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 132,
        "text_hidden": 272,
        "image_channels": 56,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 120,
        "deep_text_channels": 384,
        "deep_image_channels": 128,
        "fusion_hidden": 1088,
        "memory_slots": 28,
        "depth_steps": 11,
        "expert_count": 10,
        "expert_hidden": 1792,
        "context_top_k": 4,
        "expert_top_k": 2,
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "training_runtime": runtime.to_payload(),
        "deliberation_passes": 10,
        "minimum_passes": 5,
        "grounding_threshold": 0.54,
        "prompt_understanding_mode": "all_model_multitype_grounded_consensus_math_protein_materials_three_d_conversation",
        "notes": [
            "v8 grows v7 again and keeps full-catalog local distillation across every discovered model family, with explicit multi-type specialist supervision for non-chat systems.",
            "The continuation broadens conversation, benchmark, protein, 3D, and materials rows, increases internal deliberation depth, and adds an optional bounded Gemma slice when local Gemma 4 weights are available.",
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")
    engine = OmniCollectiveEngineV8(weights_path=weights_path, meta_path=meta_path, device=runtime.device)
    sample_prompts = [
        "Reply in exactly two sentences explaining why regression tests matter.",
        "Solve 3*x + 7 = 19.",
        "What should a grounded assistant do when it cannot verify a current fact locally?",
        "Write a tiny OpenSCAD snippet for a centered cylinder with a hole.",
        "Why do multiple-sequence alignments help protein structure prediction?",
        "Which local model is best for benchmark-focused reasoning prompts?",
        "Generate a CIF-style seed for a thermoelectric oxide candidate with a low thermal conductivity target.",
    ]
    summary = {
        "artifact": zip_path.name,
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "training_runtime": runtime.to_payload(),
        "sample_outputs": [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts],
        "notes": [
            "v8 grows v7 again and blends full-catalog distillation, specialist-profile supervision, broader benchmark and conversation rows, and heavier protein/3D/materials grounding.",
            "Inference uses a longer all-model multi-type grounded deliberation loop to improve prompt understanding and reduce hallucinated specifics.",
            "Future runs support configurable device selection, AMP, gradient accumulation, EMA, and warmup-plus-cosine scheduling.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_path, arcname=summary_path.name)
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    _write_run_state_v8(
        run_state_path,
        {
            "status": "complete",
            "stage": "done",
            "resume_dir": str(stage_resume_dir),
            "zip_path": str(zip_path),
            "desktop_zip_path": str(desktop_zip_path),
            "artifact_dir": str(artifact_dir),
            "parameter_count": parameter_count,
            "stage1_best_score": float(stage1["best_score"]),
            "stage2_best_score": float(stage2["best_score"]),
            "runtime": runtime.to_payload(),
        },
    )
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path),
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "dataset_summary": dataset_summary,
        "training_runtime": runtime.to_payload(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the omni_collective_v8 continuation model.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP))
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00030)
    parser.add_argument("--stage2_lr", type=float, default=0.00013)
    parser.add_argument("--seed", type=int, default=403)
    parser.add_argument("--distill_limit", type=int, default=160)
    parser.add_argument("--teacher_model_limit", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="auto")
    parser.add_argument("--amp_dtype", default="auto")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_scale", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    models_dir = Path(args.models_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    if args.dry_run:
        stage1_rows, full_rows, summary = build_training_rows(
            repo_root=repo_root,
            models_dir=models_dir,
            images_dir=images_dir,
            seed=int(args.seed),
            distill_limit=int(args.distill_limit),
            teacher_model_limit=int(args.teacher_model_limit),
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "stage1_rows": len(stage1_rows),
                    "stage2_rows": len(full_rows),
                    "dataset_summary": summary,
                },
                indent=2,
            )
        )
        return
    result = train_model(
        repo_root=repo_root,
        output_dir=output_dir,
        models_dir=models_dir,
        base_zip=Path(args.base_zip).resolve(),
        images_dir=images_dir,
        image_size=int(args.image_size),
        batch_size=int(args.batch_size),
        stage1_epochs=int(args.stage1_epochs),
        stage2_epochs=int(args.stage2_epochs),
        stage1_lr=float(args.stage1_lr),
        stage2_lr=float(args.stage2_lr),
        seed=int(args.seed),
        distill_limit=int(args.distill_limit),
        teacher_model_limit=int(args.teacher_model_limit),
        requested_device=str(args.device),
        amp_mode=str(args.amp),
        amp_dtype=str(args.amp_dtype),
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
        grad_accum_steps=int(args.grad_accum_steps),
        ema_decay=float(args.ema_decay),
        warmup_steps=int(args.warmup_steps),
        warmup_ratio=float(args.warmup_ratio),
        min_lr_scale=float(args.min_lr_scale),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
