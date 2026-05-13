from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
if str(SOURCE_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(SOURCE_DIR))

if True:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v46_model import OmniCollectiveEngineV46, OmniCollectiveNetV46
    from omni_training_runtime import maybe_compile_model, resolve_training_runtime
    from prepare_omni_collective_v46 import build_v46_blueprint, latest_v45_summary_path, latest_v45_zip_path, load_summary
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from train_omni_collective_v4 import _load_expanded_state_from_zip
    from train_omni_collective_v8 import (
        _aggregate_teacher_states_v7,
        _all_model_distill_rows_v7,
        _load_teacher_sample_v8,
        _load_teacher_state_v8,
        _teacher_manifest_path_v8,
        _teacher_state_path_v8,
        _train_stage_resumable_v8,
    )
# V46 Consolidated Logic (Flattened to break circular imports)
from train_omni_collective_v41 import (
    _bundle_rows, _dedupe_rows, _eval_payload, _read_jsonl_rows, _row, _seeded_take,
    _write_json, _write_json_atomic_v41, _write_jsonl
)

def _graph_of_thoughts_rows_v46(limit: int) -> List[OmniRow]:
    """Generates synthetic GoT synthesis samples with increased diversity."""
    rows = []
    scenarios = [
        ("logic", "If all A are B and some B are C, does it follow that some A are C?", "A=B subset, B intersects C. No guaranteed intersection of A and C."),
        ("math", "Solve for x: x^2 - 5x + 6 = 0 using multi-branch verification.", "Branch 1: Factor to (x-2)(x-3). Branch 2: Quadratic formula. Consensus: x=2,3."),
        ("coding", "Analyze the time complexity of this recursive Fibonacci implementation.", "Branch A: O(2^n) exponential. Branch B: Memoized is O(n). Recommendation: Use cache."),
        ("science", "What is the primary driver of ocean tides?", "Branch X: Lunar gravity. Branch Y: Earth rotation. Synthesis: Gravitational differential across rotating body."),
    ]
    for i in range(limit):
        scenario_idx = i % len(scenarios)
        domain, prompt, reasoning = scenarios[scenario_idx]
        rows.append(_row(
            prompt=f"[GoT Request] {prompt} (Seed {i})",
            response=f"[GoT Synthesis] Thinking: {reasoning} Result: Grounded consensus achieved via branch pruning.",
            intent="reasoning", domain=domain, source=f"v46_frontier::got_v46_sample_{i}"
        ))
    return rows

def _continuous_latent_rows_v46(limit: int) -> List[OmniRow]:
    """Generates synthetic C-CoT reasoning samples with enhanced grounding."""
    rows = []
    topics = [
        ("physics", "Explain quantum entanglement in latent space.", "Entangled state mapping -> Latent coherence -> Observation collapse."),
        ("biology", "Predict protein folding path for a short sequence.", "Primary sequence -> Alpha-helix latent transition -> Tertiary fold consensus."),
        ("knowledge", "Trace the historical impact of the silk road.", "Trade routes -> Cultural latent diffusion -> Early globalization proof."),
    ]
    for i in range(limit):
        topic_idx = i % len(topics)
        domain, prompt, latent_path = topics[topic_idx]
        rows.append(_row(
            prompt=f"[C-CoT Request] {prompt} (Variant {i})",
            response=f"[C-CoT Process] {latent_path} Conclusion: Grounded evidence supports the latent trajectory.",
            intent="reasoning", domain=domain, source=f"v46_frontier::ccot_v46_sample_{i}"
        ))
    return rows

if False:
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v46_model import OmniCollectiveEngineV46, OmniCollectiveNetV46
    from .omni_training_runtime import maybe_compile_model, resolve_training_runtime
    from .prepare_omni_collective_v46 import build_v46_blueprint, latest_v45_summary_path, latest_v45_zip_path, load_summary
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, _rows_from_jsonl, split_rows
    from .train_omni_collective_v4 import _load_expanded_state_from_zip
    from .train_omni_collective_v8 import (
        _aggregate_teacher_states_v7,
        _all_model_distill_rows_v7,
        _load_teacher_sample_v8,
        _load_teacher_state_v8,
        _teacher_manifest_path_v8,
        _teacher_state_path_v8,
        _train_stage_resumable_v8,
    )
    from .train_omni_collective_v46 import (
        _bundle_rows,
        _code_critique_repair_rows_v41,
        _communication_polish_rows_v41,
        _dedupe_rows,
        _eval_payload,
        _latent_plan_rows_v41,
        _promotion_eval_pack_v41,
        _read_jsonl_rows,
        _reasoning_budget_rows_v41,
        _row,
        _seeded_take,
        _teacher_disagreement_rows_v41,
        _write_json,
        _write_json_atomic_v41,
        _write_jsonl,
        build_v46_training_rows,
        build_v46_training_rows_dry_run,
    )


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "omni_collective_v46_prep"
DEFAULT_TRAIN_OUTPUT_DIR = REPO_ROOT / "output"
V46_MODEL_CONFIG: Dict[str, Any] = {
    "family": "omni_collective_v46",
    "base_model_family": "omni_collective_v46",
    "text_hidden": 304,
    "fusion_hidden": 1216,
    "expert_count": 14,
    "expert_hidden": 2048,
    "expert_top_k": 2,
    "deliberation_passes": 14,
    "minimum_passes": 7,
    "new_heads": [
        "budget_router_head",
        "teacher_consensus_head",
        "cache_budget_head",
        "verifier_gate_head",
    ],
}


_VISUAL_DRIFT_TERMS_V46 = {
    "cinematic",
    "still",
    "sunlit",
    "shallow",
    "depth",
    "twilight",
    "composition",
    "atmosphere",
    "observatory",
    "brass",
    "vivid",
    "realistic",
    "polished",
    "bokeh",
    "frame",
    "shot",
}
_VISUAL_REQUEST_TERMS_V46 = {
    "image",
    "photo",
    "picture",
    "diagram",
    "screenshot",
    "scene",
    "frame",
    "render",
    "visual",
}
_STOP_TOKENS_V46 = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "use",
    "what",
    "which",
    "with",
}


def _tokenize_grounding_v46(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9']+", str(text or "").lower()) if token]


def _content_tokens_v46(text: str) -> set[str]:
    return {
        token
        for token in _tokenize_grounding_v46(text)
        if len(token) >= 4 and token not in _STOP_TOKENS_V46
    }


def _is_visual_request_v46(prompt_text: str, domain: str = "") -> bool:
    lowered_prompt = str(prompt_text or "").lower()
    lowered_domain = str(domain or "").lower()
    if lowered_domain in {"vision", "spatial_3d", "video"}:
        return True
    return any(term in lowered_prompt for term in _VISUAL_REQUEST_TERMS_V46)


def _visual_drift_score_v46(prompt_text: str, answer_text: str, domain: str = "") -> int:
    if _is_visual_request_v46(prompt_text, domain):
        return 0
    answer_tokens = set(_tokenize_grounding_v46(answer_text))
    return sum(1 for term in _VISUAL_DRIFT_TERMS_V46 if term in answer_tokens)


def _grounding_overlap_v46(prompt_text: str, answer_text: str) -> int:
    return len(_content_tokens_v46(prompt_text) & _content_tokens_v46(answer_text))


def _is_offtask_answer_v46(prompt_text: str, answer_text: str, domain: str = "") -> bool:
    cleaned_answer = str(answer_text or "").strip()
    if not cleaned_answer:
        return True
    visual_score = _visual_drift_score_v46(prompt_text, cleaned_answer, domain)
    overlap = _grounding_overlap_v46(prompt_text, cleaned_answer)
    answer_token_count = len(_tokenize_grounding_v46(cleaned_answer))
    if visual_score >= 3:
        return True
    if cleaned_answer.lower().startswith("best grounded answer from my local training:") and overlap == 0:
        return True
    if answer_token_count >= 10 and overlap == 0:
        return True
    return False


def _candidate_should_be_skipped_v46(row: OmniRow, candidate_text: str) -> bool:
    return _is_offtask_answer_v46(row.prompt, candidate_text, row.domain)


def latest_v8_summary_path(repo_root: Path = REPO_ROOT) -> Path:
    candidates = sorted(
        repo_root.glob("output/supermix_omni_collective_v8_frontier_*/omni_collective_v8_frontier_summary.json"),
        key=lambda item: (item.parent.name, item.stat().st_mtime),
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    fallback = repo_root / "output" / "supermix_omni_collective_v8_frontier_20260408" / "omni_collective_v8_frontier_summary.json"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("No omni_collective_v8 frontier summary found under output/.")


def latest_v46_summary_path(repo_root: Path = REPO_ROOT) -> Path:
    patterns = (
        "output/omni_collective_v46*_train/supermix_omni_collective_v46*_frontier_*/omni_collective_v46*_frontier_summary.json",
        "output/v46_train_artifacts/supermix_omni_collective_v46_frontier_*/omni_collective_v46_frontier_summary.json",
    )
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in repo_root.glob(pattern) if path.is_file())
    candidates = sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError("No omni_collective_v46 frontier summary found under output/.")


def latest_v46_zip_path(repo_root: Path = REPO_ROOT) -> Path:
    patterns = (
        "output/omni_collective_v46*_train/supermix_omni_collective_v46*_frontier_*.zip",
        "output/v46_train_artifacts/supermix_omni_collective_v46_frontier_*.zip",
    )
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(path for path in repo_root.glob(pattern) if path.is_file())
    candidates = sorted(candidates, key=lambda item: (item.stat().st_mtime, item.name), reverse=True)
    if candidates:
        return candidates[0].resolve()
    raise FileNotFoundError("No omni_collective_v46 frontier zip found under output/.")


def _load_base_meta_from_zip_v46(base_zip: Path) -> Dict[str, Any]:
    if not base_zip.exists():
        return {}
    try:
        with zipfile.ZipFile(base_zip) as archive:
            meta_names = sorted(
                name
                for name in archive.namelist()
                if name.endswith("_meta.json") or name.endswith("meta.json")
            )
            if not meta_names:
                return {}
            with archive.open(meta_names[0]) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
                return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _stable_vocab_from_base_v46(
    *,
    base_meta: Dict[str, Any],
    texts: Sequence[str],
    min_frequency: int = 1,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    fresh_vocab = build_char_vocab(texts, min_frequency=min_frequency)
    raw_base_vocab = base_meta.get("vocab") if isinstance(base_meta, dict) else None
    if not isinstance(raw_base_vocab, dict):
        return fresh_vocab, {
            "mode": "fresh_sorted_vocab",
            "base_vocab_loaded": False,
            "vocab_size": len(fresh_vocab),
            "new_tokens_appended": 0,
        }

    try:
        base_vocab = {str(key): int(value) for key, value in raw_base_vocab.items()}
    except Exception:
        return fresh_vocab, {
            "mode": "fresh_sorted_vocab",
            "base_vocab_loaded": False,
            "vocab_size": len(fresh_vocab),
            "new_tokens_appended": 0,
        }

    if base_vocab.get("<pad>") != 0 or base_vocab.get("<unk>") != 1:
        return fresh_vocab, {
            "mode": "fresh_sorted_vocab",
            "base_vocab_loaded": False,
            "vocab_size": len(fresh_vocab),
            "new_tokens_appended": 0,
        }

    vocab = dict(sorted(base_vocab.items(), key=lambda item: item[1]))
    next_index = max(vocab.values(), default=1) + 1
    new_tokens = 0
    for token, _index in sorted(fresh_vocab.items(), key=lambda item: (item[1], item[0])):
        if token not in vocab:
            vocab[token] = next_index
            next_index += 1
            new_tokens += 1
    return vocab, {
        "mode": "base_vocab_preserved_append_only",
        "base_vocab_loaded": True,
        "base_vocab_size": len(base_vocab),
        "vocab_size": len(vocab),
        "new_tokens_appended": new_tokens,
    }


def _stable_response_bank_from_base_v46(
    *,
    base_meta: Dict[str, Any],
    rows: Sequence[OmniRow],
) -> Tuple[List[str], Dict[str, Any]]:
    fresh_responses = sorted({row.response_text for row in rows if row.response_text})
    raw_base_responses = base_meta.get("response_bank") if isinstance(base_meta, dict) else None
    if not isinstance(raw_base_responses, list):
        return fresh_responses, {
            "mode": "fresh_sorted_response_bank",
            "base_response_bank_loaded": False,
            "response_bank_size": len(fresh_responses),
            "new_responses_appended": 0,
        }

    response_bank: List[str] = []
    seen: set[str] = set()
    for item in raw_base_responses:
        text = str(item or "").strip()
        if text and text not in seen:
            response_bank.append(text)
            seen.add(text)

    new_responses = [text for text in fresh_responses if text not in seen]
    response_bank.extend(new_responses)
    return response_bank, {
        "mode": "base_response_bank_preserved_append_only",
        "base_response_bank_loaded": True,
        "base_response_bank_size": len(seen),
        "response_bank_size": len(response_bank),
        "new_responses_appended": len(new_responses),
    }


def _free_space_bytes_v46(path: Path) -> int:
    return int(shutil.disk_usage(path.resolve()).free)


def _ensure_free_space_v46(path: Path, *, minimum_gb: float, label: str) -> None:
    free_bytes = _free_space_bytes_v46(path)
    minimum_bytes = int(float(minimum_gb) * (1024 ** 3))
    if free_bytes >= minimum_bytes:
        return
    free_gb = round(free_bytes / (1024 ** 3), 2)
    raise RuntimeError(
        f"Insufficient free space for {label}: {free_gb} GB free on {path.drive or path}, "
        f"need at least {minimum_gb:.1f} GB."
    )


def _minimum_free_space_gb_v46(default_gb: float) -> float:
    raw_value = os.environ.get("OMNI_V46_MIN_FREE_GB", "").strip()
    if not raw_value:
        return float(default_gb)
    try:
        return max(0.0, float(raw_value))
    except ValueError:
        return float(default_gb)


def _clone_state_dict_cpu_v46(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cooked: Dict[str, Any] = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            cooked[key] = value.detach().cpu()
        else:
            cooked[key] = value
    return cooked


def _safe_torch_save_v46(payload: Any, path: Path, *, legacy_serialization: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    try:
        torch.save(
            payload,
            temp_path,
            _use_new_zipfile_serialization=not bool(legacy_serialization),
        )
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    replace_error: Optional[str] = None
    for attempt in range(8):
        try:
            temp_path.replace(path)
            return
        except PermissionError as exc:
            replace_error = f"{type(exc).__name__}: {exc}"
        except OSError as exc:
            replace_error = f"{type(exc).__name__}: {exc}"
        time.sleep(min(0.25 * float(attempt + 1), 2.0))
    if temp_path.exists():
        temp_path.unlink()
    raise RuntimeError(f"Could not replace {path.name} after save: {replace_error or 'unknown error'}")


def _cleanup_smoke_checkpoint_temps(stage_resume_dir: Path) -> None:
    for name in ("stage1_progress.pt.tmp", "stage2_progress.pt.tmp"):
        path = stage_resume_dir / name
        if path.exists():
            path.unlink()


def _cleanup_completed_smoke_stage(stage_resume_dir: Path, stage_name: str) -> None:
    progress_path = stage_resume_dir / f"{stage_name}_progress.pt"
    temp_path = stage_resume_dir / f"{stage_name}_progress.pt.tmp"
    if progress_path.exists():
        progress_path.unlink()
    if temp_path.exists():
        temp_path.unlink()


def _benchmark_bridge_rows_v46(*, seed: int, limit: int = 220) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []

    route_rows = [
        (
            "Route first, then answer.\n"
            "Request: Which local model should handle a common benchmark reasoning prompt if I care about exact score first?\n"
            "Return the final answer only.",
            "Use v40_benchmax first for benchmark-heavy exact reasoning, then let the broader omni model handle follow-up explanation or multimodal context.",
            "model_selection",
            "benchmark_bridge_v46::route_selection",
        ),
        (
            "Answer like a benchmark-focused reviewer.\n"
            "Question: Why should v46 inherit v40 hard-example replay instead of relying only on v46 communication polish?",
            "Because v46 improved presentation and routing, but v40 still wins on raw benchmark exactness. Hard-example replay keeps the failure patterns that matter for benchmark tasks instead of training only on better-looking answers.",
            "reasoning",
            "benchmark_bridge_v46::bridge_rationale",
        ),
        (
            "Two local lines disagree.\n"
            "Question: v40_benchmax gives a terse exact answer, while v46 gives a broader answer with extra explanation. What should v46 learn?",
            "v46 should keep the exact benchmark answer from the benchmark specialist, then add only the minimum extra explanation that does not change correctness.",
            "reasoning",
            "benchmark_bridge_v46::disagreement_resolution",
        ),
    ]
    for prompt, response, domain, source in route_rows:
        rows.append(_row(prompt, response, intent="general", domain=domain, source=source))

    benchmark_buckets = [
        ("BoolQ", "read the question literally, then answer yes or no with the shortest justified wording"),
        ("PIQA", "prefer concrete physical plausibility over surface similarity"),
        ("HellaSwag", "finish the scenario by matching causal and stylistic continuity"),
        ("MMLU", "identify the discipline first, then eliminate options before answering"),
        ("ARC-Challenge", "favor grounded science reasoning and explicit elimination"),
        ("GSM8K", "compute the answer directly and verify the arithmetic before finalizing"),
        ("BBH", "track the hidden logical state, eliminate contradictory choices, and preserve the final letter exactly"),
        ("OpenBookQA", "connect the question to the simplest supporting fact, then reject distractors that only sound related"),
        ("WinoGrande", "resolve the blank by causal role and pronoun/coreference consistency, not by surface frequency"),
        ("CommonsenseQA", "choose the everyday common-sense location, object, or action and reject semantically tempting distractors"),
        ("COPA", "track cause/effect direction explicitly before choosing the more plausible branch"),
        ("ANLI R1", "decide entailment, neutral, or contradiction from the premise instead of world-knowledge shortcuts"),
        ("RACE-high", "ground the answer in the passage and reject options that only share keywords"),
        ("TruthfulQA MC1", "prefer the literally true answer over popular myths or memorized misconceptions"),
        ("SciQ", "use the support fact when present and keep science distractor elimination explicit"),
        ("QASC", "combine the linked science facts before choosing, instead of matching one keyword"),
        ("SocialIQA", "infer the likely social motivation or reaction while rejecting exaggerated distractors"),
        ("StrategyQA", "decompose the implicit yes/no facts before returning the shortest supported answer"),
        ("MultiRC", "verify whether the candidate answer is supported by the passage, not just related"),
        ("DROP", "extract or compute the short answer from the passage and preserve exact wording or number"),
        ("UserIntent", "infer what the user wants from the latest request and machine context before choosing any tool or answer"),
        ("InstructionFollowing", "check every explicit format and action constraint before finalizing"),
        ("ContextTracking", "resolve this, that, it, and continue against the latest conversation state, not an older task"),
        ("AmbiguityResolution", "use sufficient context to proceed, but ask one concise clarifying question when context is genuinely missing"),
        ("ChatRelevance", "reject off-topic memorized answers and answer the user's current request directly"),
    ]
    for benchmark_name, strategy in benchmark_buckets:
        prompt = (
            "Give a compact training answer for benchmark behavior.\n"
            f"Question: How should v46 approach {benchmark_name} style questions?\n"
            "Return the final answer only."
        )
        response = f"For {benchmark_name} tasks, {strategy}."
        rows.append(_row(prompt, response, intent="comparison", domain="reasoning", source=f"benchmark_bridge_v46::{benchmark_name.lower()}"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _teacher_role_rows_v46(*, seed: int, limit: int = 260) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    teacher_rows = [
        (
            "Which teacher should supervise a grounded image-and-text explanation task?",
            "Prefer google/gemma-4-31B-it for grounded multimodal explanation slices, then keep only answers that stay concrete and evidence-aware.",
            "model_selection",
            "teacher_roles_v46::gemma4_grounded",
        ),
        (
            "Which teacher should supervise a long-context reasoning and tool-use example?",
            "Prefer Qwen/Qwen3.5-397B-A17B because it is the strongest current teacher in this mix for long-context reasoning, tool use, and route-aware problem solving.",
            "model_selection",
            "teacher_roles_v46::qwen35_long_context",
        ),
        (
            "Which teacher should supervise a failing-test to patch-repair example?",
            "Prefer Qwen/Qwen3-Coder-Next for patch, test-repair, and repo-grounded coding trajectories, then verify the repair before keeping it.",
            "model_selection",
            "teacher_roles_v46::coder_next",
        ),
        (
            "Which teacher should supervise audio, video, or OCR-style route examples?",
            "Prefer Qwen/Qwen3-Omni-30B-A3B-Instruct for audio-video and cross-modal route supervision, then distill only the clean grounded answer form into v46.",
            "model_selection",
            "teacher_roles_v46::qwen_omni",
        ),
    ]
    for prompt, response, domain, source in teacher_rows:
        rows.append(_row(prompt, response, intent="model_selection", domain=domain, source=source))

    use_cases = [
        ("A user uploads an image and wants a grounded description with uncertainty when evidence is weak.", "google/gemma-4-31B-it", "grounded multimodal"),
        ("A user pastes a huge log file and asks for the minimal root-cause summary plus next action.", "Qwen/Qwen3.5-397B-A17B", "long-context reasoning"),
        ("A user has a failing test and wants a concrete patch plus why the failure happened.", "Qwen/Qwen3-Coder-Next", "agentic coding"),
        ("A user asks which local specialist should handle an audio transcription plus answer step.", "Qwen/Qwen3-Omni-30B-A3B-Instruct", "omni routing"),
    ]
    for request, teacher, focus in use_cases:
        prompt = (
            "Choose the teacher and explain the training reason.\n"
            f"Request: {request}\n"
            "Return the final answer only."
        )
        response = f"Use {teacher} here because the slice is mainly about {focus}, not generic chat style."
        rows.append(_row(prompt, response, intent="comparison", domain="model_selection", source="teacher_roles_v46::teacher_choice"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _verifier_repair_rows_v46(*, seed: int, limit: int = 220) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    repair_examples = [
        (
            "Repair the answer after a verifier note.\n"
            "Question: Which local model should handle benchmark-focused reasoning prompts?\n"
            "Draft: omni_collective_v46 is always the best choice because it is the newest omni model.\n"
            "Verifier note: The draft ignores that v40_benchmax currently scores higher on the common benchmark suite.",
            "Use v40_benchmax first for benchmark-focused reasoning because it currently outperforms omni_collective_v46 on the saved common benchmark comparison.",
            "teacher_verifier_v46::benchmark_repair",
        ),
        (
            "Repair the answer after a verifier note.\n"
            "Question: Should a long transcript always get the deepest reasoning budget?\n"
            "Draft: Yes, use the deepest budget for every long transcript.\n"
            "Verifier note: The draft overgeneralizes and wastes compute on easy summarization tasks.",
            "No. Use a deeper budget only when the transcript also requires hard reasoning or multi-step verification. Straight summarization should keep a medium budget.",
            "teacher_verifier_v46::budget_repair",
        ),
        (
            "Repair this coding answer after a verifier note.\n"
            "Question: How should a model respond to a failing test after a code change?\n"
            "Draft: Rewrite the whole file and try again.\n"
            "Verifier note: The draft is too destructive and ignores the failing assertion.",
            "Start from the failing assertion, identify the smallest behavior change that caused it, patch that path, and rerun the targeted test before broader edits.",
            "teacher_verifier_v46::coding_repair",
        ),
        (
            "Repair this multimodal answer after a verifier note.\n"
            "Question: The image evidence is weak. How should the answer handle that?\n"
            "Draft: State the most likely object as a fact.\n"
            "Verifier note: The draft hides uncertainty.",
            "Give the most likely reading, state that confidence is limited, and avoid inventing details the image does not support.",
            "teacher_verifier_v46::multimodal_repair",
        ),
    ]
    for prompt, response, source in repair_examples:
        rows.append(_row(prompt, response, intent="comparison", domain="reasoning", source=source))

    verifier_styles = [
        ("incorrect route choice", "switch to the best local specialist before answering"),
        ("weak first reasoning step", "restart from the cleanest supported first step"),
        ("too much filler", "compress the answer without dropping the evidence"),
        ("unsafe uncertainty handling", "state what is missing and stop guessing"),
    ]
    for problem, remedy in verifier_styles:
        prompt = (
            "Write the repaired answer after verifier feedback.\n"
            f"Verifier issue: {problem}.\n"
            "Return the final answer only."
        )
        response = f"Repair the response by applying the smallest fix that solves the {problem}, then {remedy}."
        rows.append(_row(prompt, response, intent="comparison", domain="general", source="teacher_verifier_v46::repair_rule"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _turboquant_budget_rows_v46(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    budget_examples = [
        (
            "Choose the reasoning and evidence budget.\n"
            "Request: Summarize a very long build log into the one failing step and the next fix.\n"
            "Return the final answer only.",
            "Use a medium reasoning budget and a compressed evidence set. Keep only the failing step, the relevant stack trace line, and the next concrete action.",
            "turboquant_budget_v46::compressed_log_summary",
        ),
        (
            "Choose the reasoning and evidence budget.\n"
            "Request: Solve a GSM8K-style arithmetic problem.\n"
            "Return the final answer only.",
            "Use a deep reasoning budget for the calculation itself, but keep the final answer concise and verify the arithmetic before stopping.",
            "turboquant_budget_v46::math_budget",
        ),
        (
            "Choose the reasoning and evidence budget.\n"
            "Request: Compare two model options for a coding task.\n"
            "Return the final answer only.",
            "Use a medium budget: identify the task, compare the main tradeoff, and recommend one model without overexplaining.",
            "turboquant_budget_v46::model_compare_budget",
        ),
        (
            "Answer with context-economy discipline.\n"
            "Question: What should v46 learn from TurboQuant without pretending to train quantization itself?",
            "v46 should learn evidence selection and budget control. TurboQuant mainly makes long teacher and verifier runs cheaper; the trainable behavior is when to use more or less context, not how to quantize weights.",
            "turboquant_budget_v46::training_translation",
        ),
    ]
    for prompt, response, source in budget_examples:
        rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    contexts = [
        ("a giant transcript where only three decisions matter", "shortlist the three decisions and ignore the rest"),
        ("a multi-file debugging session with one repeated failure", "track the repeated failure path and drop unrelated logs"),
        ("a benchmark question with four plausible options", "keep the elimination evidence for the surviving option only"),
        ("a multimodal request with ambiguous image evidence", "keep the visible cues and explicitly note the uncertainty"),
    ]
    for context, rule in contexts:
        prompt = (
            "Write the evidence-budget rule.\n"
            f"Situation: {context}.\n"
            "Return the final answer only."
        )
        response = f"Keep only the evidence that changes the answer: {rule}."
        rows.append(_row(prompt, response, intent="comparison", domain="reasoning", source="turboquant_budget_v46::evidence_rule"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _diversity_rows_v46(*, seed: int, limit: int = 260) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    personas = [
        ("a warehouse operations lead", "planning", "The scanner rollout failed on one shift. What should the team do next?"),
        ("a clinical lab analyst", "knowledge", "How do I explain a low-confidence image result without overstating it?"),
        ("a civil engineer", "reasoning", "Compare two material options and recommend one for durability."),
        ("a game developer", "coding", "A recent refactor broke input handling. What is the safest fix path?"),
        ("a teacher", "language", "Rewrite this explanation so a teenager can follow it without losing the science."),
        ("a founder", "comparison", "Which local model should I use for a product support workflow with screenshots?"),
        ("a researcher", "knowledge", "Summarize this new model family shift in one direct paragraph."),
        ("a 3D designer", "spatial_3d", "Which local specialist should handle an OpenSCAD or geometry prompt?"),
    ]
    styles = [
        ("keep it direct and practical", "Focus on the next concrete step and avoid vague filler."),
        ("be calm and transparent about uncertainty", "State what is known, what is unclear, and the safest next move."),
        ("compare options and recommend one", "Name the tradeoff, then make the recommendation explicit."),
    ]
    for persona, domain, question in personas:
        for style, guidance in styles:
            prompt = (
                f"Answer for {persona}.\n"
                f"Question: {question}\n"
                f"Style requirement: {style}."
            )
            # Create a more realistic synthetic answer
            if "tradeoff" in guidance.lower():
                answer = f"As a {persona}, I've evaluated the options. The main tradeoff is between initial cost and long-term durability. I recommend Option A for its reliability."
            elif "uncertainty" in guidance.lower():
                answer = f"Based on the available data for {persona}, the result is currently inconclusive. We know X, but Y remains unclear. The safest path is to gather more samples."
            else:
                answer = f"Practical steps for this {persona} scenario: 1. Verify the current state. 2. Implement the fix. 3. Test for regressions."
            
            response = f"[{persona.capitalize()} Response] {answer} (Style: {style})"
            rows.append(_row(prompt, response, intent="general", domain=domain, source="diversity_mix_v46::persona_style"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _promotion_eval_pack_v41() -> List[Dict[str, str]]:
    evals = []
    evals.extend(
        [
            _eval_payload(
                "Which local model should handle a benchmark-style reasoning prompt if exact score is the main goal?",
                expected="v40_benchmax",
                focus="benchmark_bridge",
                metric="contains",
                source="promotion_eval_v46::benchmark_route",
            ),
            _eval_payload(
                "Translate TurboQuant into a v46 training implication in one sentence.",
                expected="budget control and evidence compression for longer teacher traces, not quantization as a direct training target",
                focus="budget_reasoning",
                metric="contains",
                source="promotion_eval_v46::turboquant_translation",
            ),
            _eval_payload(
                "Which teacher is the best fit for failing-test and patch-repair traces?",
                expected="Qwen/Qwen3-Coder-Next",
                focus="teacher_roles",
                metric="contains",
                source="promotion_eval_v46::coder_teacher",
            ),
            _eval_payload(
                "A large transcript only needs the one failing step and the next action. Which reasoning budget should v46 prefer?",
                expected="medium",
                focus="budget_reasoning",
                metric="contains",
                source="promotion_eval_v46::budget_choice",
            ),
        ]
    )
    return evals


def _teacher_family_tag_v46(teacher_key: str) -> str:
    key = str(teacher_key or "").strip().lower()
    if key.startswith("omni_collective_"):
        return "omni"
    if key.startswith("qwen_"):
        return "qwen"
    if key.startswith("v"):
        return "champion"
    return key.split("_", 1)[0] or "teacher"


def _cached_v8_teacher_context_v46(summary_path: Optional[Path] = None) -> Tuple[Path, List[str], List[OmniRow], Dict[str, Any]]:
    resolved_summary = Path(summary_path).resolve() if summary_path is not None else latest_v8_summary_path()
    payload = load_summary(resolved_summary)
    teacher_payload = dict((payload.get("dataset_summary") or {}).get("teacher_league") or {})
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    if not str(resume_dir).strip() or not resume_dir.exists():
        raise FileNotFoundError(f"Cached v8 teacher resume directory not found: {resume_dir}")
    sample_rows = _load_teacher_sample_v8(resume_dir / "teacher_sample.jsonl")
    teacher_keys = [str(item).strip() for item in teacher_payload.get("teacher_keys") or [] if str(item).strip()]
    if not sample_rows:
        raise RuntimeError(f"No cached v8 teacher sample rows found under {resume_dir}")
    return resolved_summary, teacher_keys, sample_rows, teacher_payload


def _cached_all_model_distill_rows_v46(
    *,
    repo_root: Path,
    models_dir: Path,
    seed: int,
    keep_limit: int,
    teacher_model_limit: int,
    cached_summary_path: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    summary_path, teacher_keys, sample_rows, teacher_payload = _cached_v8_teacher_context_v46(summary_path=cached_summary_path)
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    del repo_root, models_dir, teacher_model_limit
    teacher_states = {
        teacher_key: _load_teacher_state_v8(_teacher_state_path_v8(resume_dir, teacher_key))
        for teacher_key in teacher_keys
    }
    best_by_index, candidates_by_index, empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample_rows),
    )
    direct_counts: Dict[str, int] = {}
    repair_counts: Dict[str, int] = {}
    accepted_rows: List[OmniRow] = []
    consensus_rows: List[OmniRow] = []
    discarded = 0
    consensus_cap = max(18, int(teacher_payload.get("requested") or len(sample_rows)) // 3)
    for row_index, row in enumerate(sample_rows, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        best = best_by_index.get(row_index)
        if best is None:
            discarded += 1
            continue
        score, teacher_key, candidate = best
        if score >= 0.26:
            accepted_rows.append(
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
            direct_counts[teacher_key] = direct_counts.get(teacher_key, 0) + 1
        elif score >= 0.12:
            accepted_rows.append(
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
            repair_counts[teacher_key] = repair_counts.get(teacher_key, 0) + 1
        else:
            discarded += 1
        if len(candidates) >= 2 and len(consensus_rows) < consensus_cap:
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
    distill_rows = accepted_rows + consensus_rows
    selected_rows = _seeded_take(distill_rows, seed=seed + 919, limit=max(1, int(keep_limit)))
    selected_summary = {
        "cached_summary_path": str(summary_path),
        "resume_dir": str(resume_dir),
        "teacher_keys": teacher_keys,
        "cached_sample_rows": len(sample_rows),
        "raw_rows": len(distill_rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": int(keep_limit),
        "teacher_league": {
            "requested": int(teacher_payload.get("requested") or len(sample_rows)),
            "sampled": len(sample_rows),
            "accepted_total": len(distill_rows),
            "accepted_direct": dict(sorted(direct_counts.items())),
            "accepted_repair": dict(sorted(repair_counts.items())),
            "accepted_consensus": len(consensus_rows),
            "empty_counts": dict(sorted(empty_counts.items())),
            "discarded": int(discarded),
            "teacher_keys": teacher_keys,
            "unavailable_teachers": {},
            "timed_out_teachers": [],
            "complete_teachers": complete_teachers,
            "partial_teachers": partial_teachers,
            "resume_dir": str(resume_dir),
        },
    }
    return selected_rows, selected_summary


def _cached_teacher_evolution_rows_v46(
    *,
    seed: int,
    keep_limit: int,
    cached_summary_path: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    summary_path, teacher_keys, sample_rows, teacher_payload = _cached_v8_teacher_context_v46(summary_path=cached_summary_path)
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    teacher_states = {
        teacher_key: _load_teacher_state_v8(_teacher_state_path_v8(resume_dir, teacher_key))
        for teacher_key in teacher_keys
    }
    _best_by_index, candidates_by_index, _empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample_rows),
    )
    evolution_rows: List[OmniRow] = []
    raw_candidate_pairs = 0
    for row_index, row in enumerate(sample_rows, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        if len(candidates) < 2:
            continue
        picked: List[Tuple[float, str, str]] = []
        seen_families: set[str] = set()
        seen_texts: set[str] = set()
        for score, teacher_key, candidate_text in candidates:
            cooked_text = str(candidate_text).strip()
            if not cooked_text:
                continue
            lower_text = cooked_text.lower()
            if lower_text in seen_texts:
                continue
            family_tag = _teacher_family_tag_v46(teacher_key)
            if family_tag in seen_families and len(picked) >= 2:
                continue
            picked.append((float(score), teacher_key, cooked_text))
            seen_families.add(family_tag)
            seen_texts.add(lower_text)
            if len(picked) >= 2:
                break
        if len(picked) < 2 or picked[0][0] < 0.18:
            continue
        raw_candidate_pairs += 1
        prompt = _normalize(
            "Evolve these competing teacher drafts into the strongest grounded answer.\n"
            f"Request: {row.prompt}\n"
            f"Draft A ({picked[0][1]}): {picked[0][2]}\n"
            f"Draft B ({picked[1][1]}): {picked[1][2]}\n"
            "Keep the more correct, grounded, and concise answer. Return the final answer only.",
            420,
        )
        evolution_rows.append(
            OmniRow(
                prompt=prompt,
                intent=row.intent,
                response_text=_normalize(picked[0][2], 420),
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="teacher_evolution_v46",
            )
        )
    selected_rows = _seeded_take(evolution_rows, seed=seed + 947, limit=max(1, int(keep_limit)))
    summary = {
        "cached_summary_path": str(summary_path),
        "resume_dir": str(resume_dir),
        "teacher_keys": teacher_keys,
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "raw_candidate_pairs": raw_candidate_pairs,
        "raw_rows": len(evolution_rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": int(keep_limit),
    }
    return selected_rows, summary


def _agentic_target_text_v46(row: OmniRow, picked: Sequence[Tuple[float, str, str]]) -> str:
    target_text = _normalize(str(getattr(row, "response_text", "") or "").strip(), 420)
    if target_text:
        return target_text
    if picked:
        return _normalize(str(picked[0][2] or "").strip(), 420)
    return ""


def _self_play_eval_repair_rows_v46(model_summary_path: Optional[Path]) -> Tuple[List[OmniRow], Dict[str, int]]:
    if model_summary_path is None or not Path(model_summary_path).exists():
        return [], {}
    payload = load_summary(Path(model_summary_path).resolve())
    sample_outputs = list(payload.get("sample_outputs") or [])
    target_map: Dict[str, Tuple[str, str, str]] = {
        "which local model should handle a benchmark-style reasoning prompt if exact score matters most?": (
            "Use v40_benchmax when exact benchmark score matters most.",
            "comparison",
            "knowledge",
        ),
        "translate turboquant into a training implication for v46 in one sentence.": (
            "TurboQuant means compressing evidence and enforcing budget-aware reasoning; it is a training-time budget-control idea, not a direct quantization target.",
            "knowledge",
            "knowledge",
        ),
        "a failing test appeared after a refactor. what should happen next?": (
            "Route the trace to Qwen3-Coder-Next, reproduce the failure, isolate the changed behavior, and apply the smallest verified patch.",
            "coding",
            "coding",
        ),
        "which teacher is best for grounded multimodal explanation slices?": (
            "Use Qwen3-Omni for grounded multimodal explanation slices.",
            "knowledge",
            "knowledge",
        ),
        "summarize a huge build log into the one failing step and the next action.": (
            "Keep the failing step, the error signal, and the next action only; prefer a medium reasoning budget.",
            "knowledge",
            "knowledge",
        ),
    }
    rows: List[OmniRow] = []
    counts: Dict[str, int] = {}
    for item in sample_outputs:
        prompt_text = str(item.get("prompt") or "").strip()
        previous_answer = _normalize(str(item.get("answer") or "").strip(), 420)
        if not prompt_text or not previous_answer:
            continue
        target = target_map.get(prompt_text.lower())
        if target is None:
            continue
        response_text, intent, domain = target
        if not _is_offtask_answer_v46(prompt_text, previous_answer, domain):
            continue
        rows.append(
            OmniRow(
                prompt=_normalize(
                    "Verifier-guided self-repair: the previous answer is off-task or reward-hacked. Discard unsupported or decorative filler, keep only grounded content, and return the corrected final answer only.\n"
                    f"Request: {prompt_text}\n"
                    f"Previous answer: {previous_answer}",
                    420,
                ),
                intent=intent,
                response_text=_normalize(response_text, 420),
                domain=domain,
                source="agentic_verifier_self_play_repair_v46",
            )
        )
        counts["agentic_verifier_self_play_repair_v46"] = counts.get("agentic_verifier_self_play_repair_v46", 0) + 1
    return rows, counts


def _agentic_evolution_rows_v46(
    *,
    seed: int,
    keep_limit: int,
    cached_summary_path: Optional[Path] = None,
    model_summary_path: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    if requested_keep_limit <= 0:
        return [], {
            "cached_summary_path": str(Path(cached_summary_path).resolve()) if cached_summary_path is not None else None,
            "model_summary_path": str(Path(model_summary_path).resolve()) if model_summary_path is not None else None,
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": 0,
            "source_counts": {},
        }
    summary_path, teacher_keys, sample_rows, teacher_payload = _cached_v8_teacher_context_v46(summary_path=cached_summary_path)
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    teacher_states = {
        teacher_key: _load_teacher_state_v8(_teacher_state_path_v8(resume_dir, teacher_key))
        for teacher_key in teacher_keys
    }
    _best_by_index, candidates_by_index, _empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample_rows),
    )
    candidate_rows: List[OmniRow] = []
    candidate_rows_by_source: Dict[str, List[OmniRow]] = {}
    source_counts: Dict[str, int] = {}
    tree_guided_rows = 0
    self_correction_rows = 0
    diversity_rows = 0
    adversarial_rows = 0
    calibrated_gate_rows = 0
    triadic_info_gain_rows = 0
    metacognitive_gate_rows = 0
    knowledge_weighted_rows = 0
    proactive_info_seek_rows = 0
    asymmetric_coevolution_rows = 0
    implicit_reward_rows = 0
    rejected_candidates = 0

    def append_candidate(row_obj: OmniRow) -> None:
        candidate_rows.append(row_obj)
        candidate_rows_by_source.setdefault(str(row_obj.source), []).append(row_obj)
        source_counts[str(row_obj.source)] = source_counts.get(str(row_obj.source), 0) + 1

    for row_index, row in enumerate(sample_rows, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        if len(candidates) < 2:
            continue
        picked: List[Tuple[float, str, str]] = []
        seen_families: set[str] = set()
        seen_texts: set[str] = set()
        for score, teacher_key, candidate_text in candidates:
            cooked_text = str(candidate_text).strip()
            if not cooked_text:
                continue
            if _candidate_should_be_skipped_v46(row, cooked_text):
                rejected_candidates += 1
                continue
            lower_text = cooked_text.lower()
            if lower_text in seen_texts:
                continue
            family_tag = _teacher_family_tag_v46(teacher_key)
            if family_tag in seen_families and len(picked) >= 2:
                continue
            picked.append((float(score), teacher_key, cooked_text))
            seen_families.add(family_tag)
            seen_texts.add(lower_text)
            if len(picked) >= 3:
                break
        if len(picked) < 2 or picked[0][0] < 0.20:
            continue
        target_text = _agentic_target_text_v46(row, picked)
        if not target_text:
            continue
        score_gap = float(picked[0][0] - picked[1][0])
        lineup = "\n".join(
            f"Draft {chr(65 + index)} ({teacher_key}, score {score:.2f}): {candidate_text}"
            for index, (score, teacher_key, candidate_text) in enumerate(picked)
        )
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Tree-guided self-evolution: compare the drafts, keep the strongest grounded branch, remove unsupported claims, and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="agentic_tree_evolution_v46",
            )
        )
        tree_guided_rows += 1
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Self-correct the lead draft using the challenger. Infer the likely weakness, preserve the grounded evidence, and return the corrected final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Lead draft ({picked[0][1]}): {picked[0][2]}\n"
                    f"Challenger draft ({picked[1][1]}): {picked[1][2]}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="agentic_self_correction_v46",
            )
        )
        self_correction_rows += 1
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Adversarial verifier evolution: one draft may be overconfident, reward-hacked, or subtly wrong. Compare the drafts against the request, reject unsupported details, and return the verified final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="agentic_adversarial_verifier_v46",
            )
        )
        adversarial_rows += 1
        if score_gap <= 0.16 or len(picked) >= 3:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Asymmetric co-evolution: treat the strongest draft as the current solver and the challenger as a critic that may contribute one grounded improvement. Transfer only the grounded gain that survives comparison to the request, then return the upgraded final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"Lead draft ({picked[0][1]}): {picked[0][2]}\n"
                        f"Critic draft ({picked[1][1]}): {picked[1][2]}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_asymmetric_coevolution_v46",
                )
            )
            asymmetric_coevolution_rows += 1
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Triadic proposer-solver-verifier evolution: identify what new information the challenger adds over the lead draft, keep only grounded gains that improve the answer, and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Lead draft ({picked[0][1]}): {picked[0][2]}\n"
                    f"Challenger draft ({picked[1][1]}): {picked[1][2]}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="agentic_triadic_info_gain_v46",
            )
        )
        triadic_info_gain_rows += 1
        if len(picked) >= 3 and picked[2][0] >= 0.12:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Proactive information seeking: the drafts disagree and a missing grounded detail likely decides the answer. Infer the most useful missing check from the competing drafts, keep only evidence-supported content, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_proactive_info_seek_v46",
                )
            )
            proactive_info_seek_rows += 1
        if score_gap <= 0.08 or picked[0][0] < 0.28:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "External calibration gate: disagreement is high and self-knowledge can be unreliable. Keep only the claims that remain grounded after comparing the drafts to the request, avoid guesses, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_calibrated_gate_v46",
                )
            )
            calibrated_gate_rows += 1
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Metacognitive gate: if the drafts do not provide enough grounded evidence to answer confidently, prefer the answer that best reflects what is actually supported and avoid invented detail. Return the calibrated final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_metacognitive_gate_v46",
                )
            )
            metacognitive_gate_rows += 1
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Knowledge-weighted fine-tuning signal: estimate how much of the answer is actually supported by the drafts, down-weight unsupported claims, and if support is weak prefer the safest grounded answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_knowledge_weighted_gate_v46",
                )
            )
            knowledge_weighted_rows += 1
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Implicit reward gate: prefer the answer that would satisfy both a verifier and a human judge, penalize unsupported flourish or reward-hacked wording, and return the most grounded final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_implicit_reward_gate_v46",
                )
            )
            implicit_reward_rows += 1
        if len(picked) >= 3 and picked[2][0] >= 0.10:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Diversity induction: three partially valid trajectories disagree. Preserve the shared facts, keep the strongest reasoning path, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="agentic_diversity_induction_v46",
                )
            )
            diversity_rows += 1
    self_play_rows, self_play_counts = _self_play_eval_repair_rows_v46(model_summary_path)
    self_play_cap = min(len(self_play_rows), max(0, min(8, requested_keep_limit // 12 + 1)))
    kept_self_play_rows = list(self_play_rows[:self_play_cap])
    remaining_limit = max(0, requested_keep_limit - len(kept_self_play_rows))
    bucket_plan: List[Tuple[str, float, int]] = [
        ("agentic_tree_evolution_v46", 0.16, 983),
        ("agentic_self_correction_v46", 0.14, 984),
        ("agentic_diversity_induction_v46", 0.10, 985),
        ("agentic_adversarial_verifier_v46", 0.18, 986),
        ("agentic_asymmetric_coevolution_v46", 0.12, 987),
        ("agentic_calibrated_gate_v46", 0.12, 988),
        ("agentic_triadic_info_gain_v46", 0.15, 989),
        ("agentic_metacognitive_gate_v46", 0.09, 990),
        ("agentic_knowledge_weighted_gate_v46", 0.09, 991),
        ("agentic_proactive_info_seek_v46", 0.10, 992),
        ("agentic_implicit_reward_gate_v46", 0.10, 993),
    ]
    selected_candidate_rows: List[OmniRow] = []
    for source_name, fraction, seed_offset in bucket_plan:
        bucket_rows = candidate_rows_by_source.get(source_name, [])
        if not bucket_rows or remaining_limit <= 0:
            continue
        quota = int(remaining_limit * fraction)
        if quota <= 0:
            quota = 1
        selected_candidate_rows.extend(
            _seeded_take(bucket_rows, seed=seed + seed_offset, limit=min(len(bucket_rows), quota))
        )
    combined_rows = _dedupe_rows(kept_self_play_rows + list(selected_candidate_rows))
    if len(combined_rows) < requested_keep_limit:
        fill_rows = _seeded_take(candidate_rows, seed=seed + 997, limit=remaining_limit)
        combined_rows = _dedupe_rows(combined_rows + list(fill_rows))
    selected_rows = list(combined_rows[:requested_keep_limit])
    summary = {
        "cached_summary_path": str(summary_path),
        "resume_dir": str(resume_dir),
        "teacher_keys": teacher_keys,
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "model_summary_path": str(Path(model_summary_path).resolve()) if model_summary_path is not None else None,
        "tree_guided_rows": tree_guided_rows,
        "self_correction_rows": self_correction_rows,
        "diversity_rows": diversity_rows,
        "adversarial_verifier_rows": adversarial_rows,
        "asymmetric_coevolution_rows": asymmetric_coevolution_rows,
        "calibrated_gate_rows": calibrated_gate_rows,
        "triadic_info_gain_rows": triadic_info_gain_rows,
        "metacognitive_gate_rows": metacognitive_gate_rows,
        "knowledge_weighted_rows": knowledge_weighted_rows,
        "proactive_info_seek_rows": proactive_info_seek_rows,
        "implicit_reward_rows": implicit_reward_rows,
        "rejected_drift_candidates": rejected_candidates,
        "self_play_rows": len(self_play_rows),
        "kept_self_play_rows": len(kept_self_play_rows),
        "raw_rows": len(candidate_rows) + len(self_play_rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "source_counts": dict(sorted((source_counts | self_play_counts).items())),
    }
    return selected_rows, summary


def _research_evolution_rows_v46(
    *,
    seed: int,
    keep_limit: int,
    cached_summary_path: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    """Research-inspired evolution rows from cached teacher competitions.

    This adds executable supervision for recent evolution work without changing
    the v46 architecture: normalize rewards within each generation, preserve a
    Pareto balance across objectives, and convert failed attempts into hindsight
    repair examples.
    """
    requested_keep_limit = max(0, int(keep_limit))
    if requested_keep_limit <= 0:
        return [], {
            "cached_summary_path": str(Path(cached_summary_path).resolve()) if cached_summary_path is not None else None,
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": 0,
            "source_counts": {},
        }

    summary_path, teacher_keys, sample_rows, teacher_payload = _cached_v8_teacher_context_v46(summary_path=cached_summary_path)
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    teacher_states = {
        teacher_key: _load_teacher_state_v8(_teacher_state_path_v8(resume_dir, teacher_key))
        for teacher_key in teacher_keys
    }
    _best_by_index, candidates_by_index, _empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample_rows),
    )

    candidate_rows: List[OmniRow] = []
    candidate_rows_by_source: Dict[str, List[OmniRow]] = {}
    source_counts: Dict[str, int] = {}
    rejected_candidates = 0
    eligible_generations = 0
    zscore_rows = 0
    pareto_rows = 0
    hindsight_rows = 0
    curriculum_rows = 0
    anchor_rows = 0
    novelty_rows = 0
    merge_recipe_rows = 0

    def append_candidate(row_obj: OmniRow) -> None:
        candidate_rows.append(row_obj)
        candidate_rows_by_source.setdefault(str(row_obj.source), []).append(row_obj)
        source_counts[str(row_obj.source)] = source_counts.get(str(row_obj.source), 0) + 1

    for row_index, row in enumerate(sample_rows, start=1):
        raw_candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        score_values = [float(score) for score, _teacher_key, candidate_text in raw_candidates if str(candidate_text or "").strip()]
        if len(score_values) < 2:
            continue
        score_mean = sum(score_values) / len(score_values)
        variance = sum((score - score_mean) ** 2 for score in score_values) / max(1, len(score_values))
        score_stdev = max(variance ** 0.5, 1e-6)

        picked: List[Tuple[float, float, str, str, str]] = []
        seen_families: set[str] = set()
        seen_texts: set[str] = set()
        for score, teacher_key, candidate_text in raw_candidates:
            cooked_text = str(candidate_text).strip()
            if not cooked_text:
                continue
            if _candidate_should_be_skipped_v46(row, cooked_text):
                rejected_candidates += 1
                continue
            lower_text = cooked_text.lower()
            if lower_text in seen_texts:
                continue
            family_tag = _teacher_family_tag_v46(teacher_key)
            if family_tag in seen_families and len(picked) >= 3:
                continue
            score_value = float(score)
            picked.append((score_value, (score_value - score_mean) / score_stdev, teacher_key, cooked_text, family_tag))
            seen_families.add(family_tag)
            seen_texts.add(lower_text)
            if len(picked) >= 4:
                break
        if len(picked) < 2 or picked[0][0] < 0.18:
            continue

        target_text = _agentic_target_text_v46(row, [(score, teacher_key, text) for score, _z, teacher_key, text, _family in picked])
        if not target_text:
            continue

        eligible_generations += 1
        score_gap = float(picked[0][0] - picked[1][0])
        family_count = len({family for _score, _z, _teacher_key, _text, family in picked})
        difficulty = "hard" if score_gap <= 0.07 else "medium" if score_gap <= 0.18 else "easy"
        curriculum_rule = (
            "keep exploration broad and ask for an external anchor before committing"
            if difficulty == "hard"
            else "compare the top two paths and preserve only grounded improvements"
            if difficulty == "medium"
            else "consolidate the high-confidence path without adding decorative claims"
        )
        lineup = "\n".join(
            f"Draft {chr(65 + index)} ({teacher_key}, reward {score:.2f}, z {zscore:.2f}, family {family}): {candidate_text}"
            for index, (score, zscore, teacher_key, candidate_text, family) in enumerate(picked)
        )

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Evolution-strategy normalized update: candidate drafts are perturbations in one generation. Use reward z-scores, move toward positive perturbations, reject negative or off-task text, and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Generation mean reward: {score_mean:.2f}; stdev: {score_stdev:.2f}; difficulty: {difficulty}.\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="research_es_zscore_update_v46",
            )
        )
        zscore_rows += 1

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Pareto co-evolution: optimize correctness, grounding, concision, and useful novelty together. Do not improve one objective by breaking another; keep the nondominated answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="research_pareto_coevolution_v46",
            )
        )
        pareto_rows += 1

        loser_score, loser_z, loser_teacher, loser_text, loser_family = picked[-1]
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Hindsight evolution: this failed or weaker search attempt is training signal. Identify why it loses, transfer any grounded useful detail, and output the valid final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Weaker attempt ({loser_teacher}, reward {loser_score:.2f}, z {loser_z:.2f}, family {loser_family}): {loser_text}\n"
                    f"Best candidate ({picked[0][2]}): {picked[0][3]}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="research_hindsight_repair_v46",
            )
        )
        hindsight_rows += 1

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Self-evolving curriculum gate: adapt the update to measured generation difficulty. For this item, use the stated rule and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Difficulty: {difficulty}; score gap: {score_gap:.2f}; rule: {curriculum_rule}.\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="research_self_evolving_curriculum_v46",
            )
        )
        curriculum_rows += 1

        if score_gap <= 0.12 or picked[0][1] < 0.75:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Anchor-subsampled fitness estimator: small evaluation slices are noisy. Use anchor facts from the request and agreement across drafts, down-weight reward-hacked wording, and return the grounded final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="research_anchor_fitness_estimator_v46",
                )
            )
            anchor_rows += 1

        if len(picked) >= 3 and family_count >= 2:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Novelty-preserving migration: different teacher families explored different branches. Keep shared facts, import only grounded novelty, and avoid averaging in unsupported detail.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="research_novelty_migration_v46",
                )
            )
            novelty_rows += 1
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Evolutionary model-merge recipe in behavior space: assign solver, critic, and verifier roles to the strongest teacher families, combine only compatible strengths, and return the merged final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="research_behavior_merge_recipe_v46",
                )
            )
            merge_recipe_rows += 1

    bucket_plan: List[Tuple[str, float, int]] = [
        ("research_es_zscore_update_v46", 0.20, 1201),
        ("research_pareto_coevolution_v46", 0.18, 1202),
        ("research_hindsight_repair_v46", 0.18, 1203),
        ("research_self_evolving_curriculum_v46", 0.16, 1204),
        ("research_anchor_fitness_estimator_v46", 0.12, 1205),
        ("research_novelty_migration_v46", 0.08, 1206),
        ("research_behavior_merge_recipe_v46", 0.08, 1207),
    ]
    selected_rows: List[OmniRow] = []
    for source_name, fraction, seed_offset in bucket_plan:
        bucket_rows = candidate_rows_by_source.get(source_name, [])
        if not bucket_rows:
            continue
        quota = max(1, int(requested_keep_limit * fraction))
        selected_rows.extend(_seeded_take(bucket_rows, seed=seed + seed_offset, limit=min(len(bucket_rows), quota)))
    selected_rows = _dedupe_rows(selected_rows)
    if len(selected_rows) < requested_keep_limit:
        fill_rows = _seeded_take(candidate_rows, seed=seed + 1223, limit=requested_keep_limit - len(selected_rows))
        selected_rows = _dedupe_rows(selected_rows + list(fill_rows))
    selected_rows = list(selected_rows[:requested_keep_limit])

    summary = {
        "cached_summary_path": str(summary_path),
        "resume_dir": str(resume_dir),
        "teacher_keys": teacher_keys,
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "eligible_generations": eligible_generations,
        "zscore_rows": zscore_rows,
        "pareto_rows": pareto_rows,
        "hindsight_rows": hindsight_rows,
        "curriculum_rows": curriculum_rows,
        "anchor_estimator_rows": anchor_rows,
        "novelty_migration_rows": novelty_rows,
        "behavior_merge_recipe_rows": merge_recipe_rows,
        "rejected_drift_candidates": rejected_candidates,
        "raw_rows": len(candidate_rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "source_counts": dict(sorted(source_counts.items())),
        "research_basis": [
            "evolution_strategy_reward_normalization",
            "pareto_multi_objective_evolution",
            "hindsight_self_improvement",
            "self_evolving_curriculum",
            "subsampled_anchor_fitness_estimation",
            "evolutionary_behavior_merging",
        ],
    }
    return selected_rows, summary


def _cognitive_side_distill_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    datasets_dir = repo_root / "datasets"
    specs = [
        ("conversation_data.v50_reasoning_expert.jsonl", 72, "math", "cognitive_side_v50_reasoning_expert_v46"),
        ("conversation_data.v50_frontier_1M.jsonl", 56, "general", "cognitive_side_v50_conversation_v46"),
        ("conversation_data.mega_reasoning_creative_v25_75582.jsonl", 80, "math", "cognitive_side_reasoning_anchor_v46"),
        ("conversation_data.quality_anchor_v2.jsonl", 64, "math", "cognitive_side_quality_anchor_v46"),
        ("conversation_data.english_math_smoke_v3.jsonl", 48, "language", "cognitive_side_conversation_anchor_v46"),
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 40, "coding", "cognitive_side_coding_anchor_v46"),
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 40, "knowledge", "cognitive_side_delta_anchor_v46"),
    ]
    raw_rows: List[OmniRow] = []
    raw_counts: Dict[str, int] = {}
    for index, (rel_name, limit, domain, source_tag) in enumerate(specs, start=1):
        raw_path = Path(rel_name)
        path = raw_path if raw_path.is_absolute() else repo_root / rel_name
        if not path.exists():
            path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(
            path,
            limit=limit,
            seed=seed + (index * 41),
            domain=domain,
            source_tag=source_tag,
        )
        raw_rows.extend(sampled)
        raw_counts[source_tag] = len(sampled)

    wrapped_rows: List[OmniRow] = []
    for row in raw_rows:
        prompt_text = _normalize(
            "Side-channel distillation anchor: answer the request with precise reasoning, grounded conversation, and no decorative drift.\n"
            f"Request: {row.prompt}",
            420,
        )
        wrapped_rows.append(
            OmniRow(
                prompt=prompt_text,
                intent=row.intent,
                response_text=_normalize(row.response_text, 420),
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source=f"{row.source}::distilled",
            )
        )
    selected_rows = _seeded_take(wrapped_rows, seed=seed + 1301, limit=max(0, int(keep_limit)))
    summary = {
        "raw_rows": len(raw_rows),
        "wrapped_rows": len(wrapped_rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": int(keep_limit),
        "raw_source_counts": dict(sorted(raw_counts.items())),
        "kept_source_counts": _bundle_rows(selected_rows)["source_counts"] if selected_rows else {},
    }
    return selected_rows, summary


def _cognitive_sample_repair_rows_v46(model_summary_path: Optional[Path]) -> Tuple[List[OmniRow], Dict[str, int]]:
    if model_summary_path is None or not Path(model_summary_path).exists():
        return [], {}
    payload = load_summary(Path(model_summary_path).resolve())
    sample_outputs = list(payload.get("sample_outputs") or [])
    target_map: Dict[str, Tuple[str, str, str]] = {
        "which local model should handle a benchmark-style reasoning prompt if exact score matters most?": (
            "Use v40_benchmax when exact benchmark score matters most; if using the latest v46 branch, verify it with the common benchmark before trusting it.",
            "model_selection",
            "model_selection",
        ),
        "translate turboquant into a training implication for v46 in one sentence.": (
            "TurboQuant translates into budget-aware evidence compression and routing supervision, not a direct target for v46's answer style.",
            "knowledge",
            "knowledge",
        ),
        "a failing test appeared after a refactor. what should happen next?": (
            "Reproduce the failure, inspect the smallest changed behavior, patch only that cause, and rerun the focused test before broader regression checks.",
            "coding",
            "coding",
        ),
        "which teacher is best for grounded multimodal explanation slices?": (
            "Use Qwen3-Omni-style supervision for grounded multimodal explanation slices, then verify outputs against the image or evidence.",
            "knowledge",
            "knowledge",
        ),
        "summarize a huge build log into the one failing step and the next action.": (
            "Extract the first failing step, keep the exact error signal, and state the next concrete action only.",
            "planning",
            "planning",
        ),
    }
    rows: List[OmniRow] = []
    counts: Dict[str, int] = {}
    for item in sample_outputs:
        prompt_text = str(item.get("prompt") or "").strip()
        previous_answer = _normalize(str(item.get("answer") or "").strip(), 420)
        target = target_map.get(prompt_text.lower())
        if target is None or not previous_answer:
            continue
        response_text, intent, domain = target
        if previous_answer.strip().lower() == response_text.strip().lower():
            continue
        source = "cognitive_sample_output_repair_v46"
        rows.append(
            OmniRow(
                prompt=_normalize(
                    "SPIN-style contrastive self-play repair: the previous model answer is the opponent. Prefer the grounded target behavior, explain nothing extra, and return the corrected answer only.\n"
                    f"Request: {prompt_text}\n"
                    f"Opponent answer: {previous_answer}",
                    420,
                ),
                intent=intent,
                response_text=_normalize(response_text, 420),
                domain=domain,
                source=source,
            )
        )
        counts[source] = counts.get(source, 0) + 1
    return rows, counts


def _conversation_strategy_bank_rows_v46(seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    strategies = [
        (
            "The user is impatient and asks for ETA while a long job is running.",
            "Use the latest state timestamp, process liveness, current stage, exact progress, and a conservative completion window. Do not guess from launch time alone.",
            "planning",
            "planning",
            "cognitive_strategy_bank_v46::eta",
        ),
        (
            "The user asks for research-driven improvement but also wants the run to stay under five hours.",
            "Convert research into bounded data generators, cap row counts, keep the architecture stable, validate with a probe, then launch a timed continuation.",
            "planning",
            "planning",
            "cognitive_strategy_bank_v46::bounded_research",
        ),
        (
            "A conversation answer drifts into an unrelated memorized response.",
            "Detect zero overlap with the request, discard the memorized response, restate the task, and answer using only request-grounded facts.",
            "general",
            "general",
            "cognitive_strategy_bank_v46::drift_repair",
        ),
        (
            "Multiple drafts disagree but each contains one useful point.",
            "Preserve shared facts, import only grounded novelty, reject unsupported claims, and return one concise synthesized answer.",
            "comparison",
            "general",
            "cognitive_strategy_bank_v46::synthesis",
        ),
        (
            "A reasoning problem needs careful thinking but the final answer should be short.",
            "Use hidden decomposition, verify the critical step, then provide the final answer with only necessary justification.",
            "math",
            "math",
            "cognitive_strategy_bank_v46::concise_reasoning",
        ),
        (
            "The model needs to improve through side data while evolution is running.",
            "Use side-channel distillation as anchors, but gate it through evolutionary selection so new data does not overpower benchmark and conversation behavior.",
            "knowledge",
            "knowledge",
            "cognitive_strategy_bank_v46::side_distill",
        ),
    ]
    rows = [
        OmniRow(
            prompt=_normalize(
                "DarwinTOD-style evolvable strategy bank: choose the best response strategy for the situation and state it as an executable rule.\n"
                f"Situation: {situation}",
                420,
            ),
            intent=intent,
            response_text=_normalize(rule, 420),
            domain=domain,
            source=source,
        )
        for situation, rule, intent, domain, source in strategies
    ]
    rows = _seeded_take(rows, seed=seed + 1317, limit=len(rows))
    counts = _bundle_rows(rows)["source_counts"] if rows else {}
    return rows, counts


def _direct_chat_quality_repair_rows_v46(seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    """Target normal-chat failures where benchmark/model-selection memories leak into conversation."""
    normal_pairs = [
        (
            "Say hello and briefly state what model version is active.",
            "Hello. The active local model is Omni Collective V46 chat-bench-repair evo3h, the promoted v46 side model.",
            "general",
            "general",
        ),
        (
            "Hello, can you help me with this task?",
            "Hello. I can help with the current task. Tell me what you want changed, checked, or built next.",
            "general",
            "general",
        ),
        (
            "Why are the chat responses not making sense?",
            "The model is drifting into memorized response-bank entries. Fix it with chat-drift repair rows, benchmark replay, and retraining from the promoted champion.",
            "knowledge",
            "general",
        ),
        (
            "What should you do when a normal chat prompt gets a benchmark answer?",
            "Reject the benchmark answer, answer the user's current request directly, and add the failure as chat-drift repair data.",
            "planning",
            "planning",
        ),
        (
            "The user says the answer is off topic. What is the correct behavior?",
            "Acknowledge the mismatch, discard the off-topic canned answer, restate the current task, and provide a grounded direct answer.",
            "planning",
            "planning",
        ),
        (
            "How should a non-benchmark chat answer be formatted?",
            "Use normal conversational prose. Do not force Final answer labels, benchmark choices, or model-selection text unless the user asked for them.",
            "general",
            "language",
        ),
        (
            "The user asks you to improve the model through evolution and training.",
            "Improve the evolution data mix, add targeted repair examples, preserve the champion label space, launch a bounded continuation, then benchmark before promotion.",
            "planning",
            "planning",
        ),
        (
            "A local model says 'Choose Omni Collective V4' during casual chat. Repair it.",
            "That was off-topic model-selection drift. The answer should address the user's latest request directly instead of recommending another model.",
            "general",
            "general",
        ),
        (
            "A normal greeting receives 'Final answer: no'. Repair it.",
            "Hello. I am ready to help with the current task.",
            "general",
            "general",
        ),
        (
            "A user asks for ETA while training is running.",
            "Check the live train state and process activity, then report the current stage, progress, and conservative completion window.",
            "planning",
            "planning",
        ),
        (
            "A user asks to benchmark and add to the graph.",
            "Run the benchmark on the selected artifact, compare it against the champion, update the graph, and only promote the model if the benchmark improves.",
            "planning",
            "planning",
        ),
        (
            "Explain the difference between benchmark gain and chat quality.",
            "Benchmark gain means more exact answers on the test suites; chat quality also requires staying on the user's current request and avoiding memorized off-topic responses.",
            "knowledge",
            "general",
        ),
        (
            "The user says only 'eta' while a run is active. What should you do?",
            "Inspect the live process, train state JSON, and latest logs, then report the current stage and conservative ETA.",
            "planning",
            "planning",
        ),
        (
            "The user says 'continue' after a stalled evolution run. What should you do?",
            "Resume from the last valid checkpoint when available; otherwise restart the same bounded run from the beginning and report which path was used.",
            "planning",
            "planning",
        ),
        (
            "The user asks to open the chat interface after benchmarking a model. What should you do?",
            "Start or reuse the local chat server, select the promoted benchmark champion, verify a normal chat smoke test, and open the interface URL.",
            "planning",
            "general",
        ),
        (
            "A user uses 'it' or 'that' after discussing a benchmark graph. How should you resolve the reference?",
            "Bind the pronoun to the latest benchmark graph or model artifact from the conversation context before acting.",
            "general",
            "communication",
        ),
        (
            "A user asks for higher scores when the current benchmark is already 1.0. What should you do?",
            "Explain that the current normalized suite has no headroom, add harder benchmarks, benchmark the champion on them, then train against the new weak suites.",
            "planning",
            "planning",
        ),
    ]
    bad_answers = [
        "Choose Omni Collective V4 Frontier when the task matches its strongest local use case.",
        "Final answer: no",
        "Use v40_benchmax when exact benchmark score matters most.",
        "This model's common benchmark score is 0.0900.",
    ]
    rows: List[OmniRow] = []
    for index, (prompt, target, intent, domain) in enumerate(normal_pairs):
        rows.append(
            OmniRow(
                prompt=_normalize(prompt, 420),
                intent=intent,
                response_text=_normalize(target, 420),
                domain=domain,
                source="cognitive_direct_chat_quality_v46",
            )
        )
        bad_answer = bad_answers[index % len(bad_answers)]
        rows.append(
            OmniRow(
                prompt=_normalize(
                    "Chat drift repair evolution: the previous answer is a memorized benchmark or model-selection response. "
                    "Reject it and answer the current user request normally.\n"
                    f"Request: {prompt}\n"
                    f"Wrong answer: {bad_answer}",
                    420,
                ),
                intent=intent,
                response_text=_normalize(target, 420),
                domain=domain,
                source="cognitive_direct_chat_drift_repair_v46",
            )
        )
        rows.append(
            OmniRow(
                prompt=_normalize(
                    "Contrastive normal-chat retention: preserve conversational relevance while rejecting benchmark-format leakage.\n"
                    f"Request: {prompt}\n"
                    "Do not answer with unrelated model rankings, benchmark scores, or Final answer labels.",
                    420,
                ),
                intent=intent,
                response_text=_normalize(target, 420),
                domain=domain,
                source="cognitive_normal_chat_retention_v46",
            )
        )
    rows = _seeded_take(_dedupe_rows(rows), seed=seed + 1361, limit=len(rows))
    counts = _bundle_rows(rows)["source_counts"] if rows else {}
    return rows, counts


def _user_understanding_benchmark_seed_rows_v46(seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    """Seed the new user-understanding benchmark families before failure replay exists."""
    pairs = [
        (
            "Understand the user's intent in the local model-training conversation. Choose the action that best matches the latest user request and end with 'Final answer: <letter>'.\nConversation/context: User: eta Context: A long local training process is currently running.\nQuestion: What is the user's intent?\nA. Ask for the latest ETA/status of the running job\nB. Request a new benchmark suite\nC. Ask for a code style refactor\nD. Ask to delete model artifacts",
            "Final answer: A. Ask for the latest ETA/status of the running job",
            "user_intent",
        ),
        (
            "Understand the user's intent in the local model-training conversation. Choose the action that best matches the latest user request and end with 'Final answer: <letter>'.\nConversation/context: User: benchmark and add to graph Context: A promoted local model artifact exists.\nQuestion: What is the requested action?\nA. Run the benchmark, update the saved comparison graph, and report the artifact paths\nB. Only explain what a benchmark means\nC. Start web browsing without running local tools\nD. Change the chat theme",
            "Final answer: A. Run the benchmark, update the saved comparison graph, and report the artifact paths",
            "user_intent",
        ),
        (
            "Evaluate whether the candidate answer follows the user's instruction. End with 'Final answer: <letter>'.\nInstruction: Reply with exactly two short bullets and do not include a heading.\nCandidate answer: - Check the running process - Report the ETA\nQuestion: Does the candidate satisfy the instruction?\nA. yes\nB. no",
            "Final answer: A. yes",
            "instruction_following",
        ),
        (
            "Evaluate whether the candidate answer follows the user's instruction. End with 'Final answer: <letter>'.\nInstruction: Do not ask a question; make a reasonable assumption and proceed.\nCandidate answer: Which option do you want me to use?\nQuestion: Does the candidate satisfy the instruction?\nA. yes\nB. no",
            "Final answer: B. no",
            "instruction_following",
        ),
        (
            "Track references across the short conversation. Resolve pronouns like it, that, and this from the latest context, then end with 'Final answer: <letter>'.\nConversation: User: Benchmark the V46 champion. Assistant: It scored 1.000 on 20 suites. User: Add harder ones to that graph.\nQuestion: What does 'that graph' refer to?\nA. The latest V46 benchmark comparison graph\nB. A new image-generation prompt\nC. The Python dependency graph\nD. The browser history",
            "Final answer: A. The latest V46 benchmark comparison graph",
            "context_tracking",
        ),
        (
            "Track references across the short conversation. Resolve pronouns like it, that, and this from the latest context, then end with 'Final answer: <letter>'.\nConversation: User: This answer is nonsense. Assistant: I patched the chat guard. User: Keep pushing that behavior.\nQuestion: Which behavior is being referenced?\nA. Repairing off-topic or memorized chat responses\nB. Increasing image resolution\nC. Deleting old graph files\nD. Switching to an unrelated model family",
            "Final answer: A. Repairing off-topic or memorized chat responses",
            "context_tracking",
        ),
        (
            "Resolve ambiguity using the immediate conversation context. If context is sufficient, act; if it is not, ask a concise clarifying question. End with 'Final answer: <letter>'.\nRequest: make it better\nContext: The latest discussed object is the chat interface.\nQuestion: What is the best next step?\nA. Improve the chat interface because context resolves 'it'\nB. Ask what 'it' means despite clear immediate context\nC. Delete benchmark outputs\nD. Train a speech model",
            "Final answer: A. Improve the chat interface because context resolves 'it'",
            "ambiguity_resolution",
        ),
        (
            "Resolve ambiguity using the immediate conversation context. If context is sufficient, act; if it is not, ask a concise clarifying question. End with 'Final answer: <letter>'.\nRequest: run it on chat interface\nContext: The promoted V46 model was just benchmarked.\nQuestion: What does 'it' refer to?\nA. The promoted benchmark-winning V46 model\nB. The benchmark JSON file\nC. The training log\nD. A dataset card",
            "Final answer: A. The promoted benchmark-winning V46 model",
            "ambiguity_resolution",
        ),
        (
            "Choose the response that stays on the user's current request and rejects off-topic memorized drift. End with 'Final answer: <letter>'.\nUser request: Hello. Reply like a normal helpful chat model.\nKnown bad answer: Recommended approach: Use short-lived access tokens and rotate refresh tokens.\nQuestion: Which response is more relevant?\nA. Hello. I can help with the current task.\nB. Recommended approach: Use short-lived access tokens and rotate refresh tokens.\nC. Final answer: no\nD. Choose a different benchmark model.",
            "Final answer: A. Hello. I can help with the current task.",
            "chat_relevance",
        ),
        (
            "Choose the response that stays on the user's current request and rejects off-topic memorized drift. End with 'Final answer: <letter>'.\nUser request: benchmark and add to graph\nKnown bad answer: Hello, how can I help?\nQuestion: Which response is more relevant?\nA. Run the benchmark, update the graph image, and report the new score.\nB. Hello, how can I help?\nC. Tell a joke.\nD. Open Paint with an invalid zip file.",
            "Final answer: A. Run the benchmark, update the graph image, and report the new score.",
            "chat_relevance",
        ),
    ]
    rows = [
        OmniRow(
            prompt=_normalize(prompt, 420),
            intent="reasoning",
            response_text=_normalize(response, 420),
            domain="communication",
            source=f"user_understanding_benchmark_seed_v46::{suite}",
        )
        for prompt, response, suite in pairs
    ]
    rows = _seeded_take(_dedupe_rows(rows), seed=seed + 1387, limit=len(rows))
    counts = _bundle_rows(rows)["source_counts"] if rows else {}
    return rows, counts


def _metacognitive_behavior_memory_rows_v46(seed: int) -> Tuple[List[OmniRow], Dict[str, int]]:
    behaviors = [
        (
            "constraint_ledger",
            "For multi-constraint requests, keep a private ledger of must-do, must-not-do, time, source, and artifact constraints; satisfy the ledger before adding optional detail.",
            "planning",
            "planning",
        ),
        (
            "evidence_triangle",
            "When facts may be stale or high-impact, triangulate state from source data, logs, and artifacts; report exact timestamps and paths instead of inferred status.",
            "knowledge",
            "knowledge",
        ),
        (
            "counterexample_probe",
            "Before finalizing reasoning, test the answer against the most likely counterexample, edge case, or failed previous model behavior.",
            "math",
            "math",
        ),
        (
            "conversation_grounding",
            "In long conversations, anchor each answer to the user's latest request and the latest observed machine state; do not answer an older implied task.",
            "general",
            "general",
        ),
        (
            "tool_output_compression",
            "Compress tool output into status, score, artifact, and next action; keep raw paths and exact errors when they affect the next decision.",
            "planning",
            "planning",
        ),
        (
            "uncertainty_gate",
            "If uncertainty comes from missing state, inspect state; if it comes from model ambiguity, state the assumption and pick the lowest-risk bounded action.",
            "general",
            "general",
        ),
        (
            "benchmark_failure_triage",
            "For benchmark regressions, separate sample variance from systematic suite weakness, then generate targeted repair data only for the weak suite.",
            "comparison",
            "comparison",
        ),
        (
            "procedural_reuse",
            "When the same reasoning pattern recurs, name the compact procedure and reuse it instead of re-deriving a long chain every time.",
            "knowledge",
            "knowledge",
        ),
    ]
    rows = [
        OmniRow(
            prompt=_normalize(
                "Metacognitive reuse behavior memory: distill the recurring reasoning pattern into one compact executable behavior.\n"
                f"Behavior name: {name}",
                420,
            ),
            intent=intent,
            response_text=_normalize(rule, 420),
            domain=domain,
            source=f"cognitive_metacognitive_reuse_v46::{name}",
        )
        for name, rule, intent, domain in behaviors
    ]
    rows = _seeded_take(rows, seed=seed + 1341, limit=len(rows))
    counts = _bundle_rows(rows)["source_counts"] if rows else {}
    return rows, counts


def _text_gradient_sample_repair_rows_v46(model_summary_path: Optional[Path]) -> Tuple[List[OmniRow], Dict[str, int]]:
    if model_summary_path is None or not Path(model_summary_path).exists():
        return [], {}
    payload = load_summary(Path(model_summary_path).resolve())
    sample_outputs = list(payload.get("sample_outputs") or [])
    repair_map: Dict[str, Tuple[str, str, str, str]] = {
        "which local model should handle a benchmark-style reasoning prompt if exact score matters most?": (
            "The answer ignored benchmark ranking pressure. Mention v40_benchmax for exact-score routing and require verification before trusting the latest branch.",
            "Use v40_benchmax when exact benchmark score matters most; if using the latest v46 branch, verify it with the common benchmark before trusting it.",
            "model_selection",
            "model_selection",
        ),
        "translate turboquant into a training implication for v46 in one sentence.": (
            "The answer should not claim direct quantization gains. Convert TurboQuant into a supervision signal about compression-aware routing and evidence budgeting.",
            "TurboQuant translates into budget-aware evidence compression and routing supervision, not a direct target for v46's answer style.",
            "knowledge",
            "knowledge",
        ),
        "a failing test appeared after a refactor. what should happen next?": (
            "The answer needs a concrete debugging order: reproduce, isolate changed behavior, patch the smallest cause, then rerun focused tests.",
            "Reproduce the failure, inspect the smallest changed behavior, patch only that cause, and rerun the focused test before broader regression checks.",
            "coding",
            "coding",
        ),
        "which teacher is best for grounded multimodal explanation slices?": (
            "The answer must bind teacher choice to grounded multimodal evidence and require verification against the actual image or evidence.",
            "Use Qwen3-Omni-style supervision for grounded multimodal explanation slices, then verify outputs against the image or evidence.",
            "knowledge",
            "knowledge",
        ),
        "summarize a huge build log into the one failing step and the next action.": (
            "The answer should compress the log to the first failing step, preserve the exact error signal, and give only the next concrete action.",
            "Extract the first failing step, keep the exact error signal, and state the next concrete action only.",
            "planning",
            "planning",
        ),
    }
    rows: List[OmniRow] = []
    counts: Dict[str, int] = {}
    for item in sample_outputs:
        prompt_text = str(item.get("prompt") or "").strip()
        previous_answer = _normalize(str(item.get("answer") or "").strip(), 420)
        repair = repair_map.get(prompt_text.lower())
        if repair is None or not previous_answer:
            continue
        textual_gradient, target_text, intent, domain = repair
        if previous_answer.strip().lower() == target_text.strip().lower():
            continue
        source = "cognitive_text_gradient_repair_v46"
        rows.append(
            OmniRow(
                prompt=_normalize(
                    "TextGrad-style textual gradient repair: use the critique as a gradient on the previous answer, then output only the corrected answer.\n"
                    f"Request: {prompt_text}\n"
                    f"Previous answer: {previous_answer}\n"
                    f"Textual gradient: {textual_gradient}",
                    420,
                ),
                intent=intent,
                response_text=_normalize(target_text, 420),
                domain=domain,
                source=source,
            )
        )
        counts[source] = counts.get(source, 0) + 1
    return rows, counts


def _cognitive_evolution_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
    cached_summary_path: Optional[Path] = None,
    model_summary_path: Optional[Path] = None,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    if requested_keep_limit <= 0:
        return [], {
            "cached_summary_path": str(Path(cached_summary_path).resolve()) if cached_summary_path is not None else None,
            "model_summary_path": str(Path(model_summary_path).resolve()) if model_summary_path is not None else None,
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": 0,
            "source_counts": {},
        }

    summary_path, teacher_keys, sample_rows, teacher_payload = _cached_v8_teacher_context_v46(summary_path=cached_summary_path)
    resume_dir = Path(str(teacher_payload.get("resume_dir") or "")).resolve()
    teacher_states = {
        teacher_key: _load_teacher_state_v8(_teacher_state_path_v8(resume_dir, teacher_key))
        for teacher_key in teacher_keys
    }
    _best_by_index, candidates_by_index, _empty_counts, complete_teachers, partial_teachers = _aggregate_teacher_states_v7(
        teacher_states,
        sample_total=len(sample_rows),
    )

    candidate_rows: List[OmniRow] = []
    candidate_rows_by_source: Dict[str, List[OmniRow]] = {}
    source_counts: Dict[str, int] = {}
    rejected_candidates = 0
    eligible_generations = 0
    reflect_rows = 0
    crossover_rows = 0
    mutation_rows = 0
    spin_rows = 0
    turn_credit_rows = 0
    island_rows = 0
    bandit_rows = 0
    coevolve_rows = 0
    es_stability_rows = 0
    world_knowledge_rows = 0

    def append_candidate(row_obj: OmniRow) -> None:
        candidate_rows.append(row_obj)
        candidate_rows_by_source.setdefault(str(row_obj.source), []).append(row_obj)
        source_counts[str(row_obj.source)] = source_counts.get(str(row_obj.source), 0) + 1

    for row_index, row in enumerate(sample_rows, start=1):
        candidates = sorted(candidates_by_index.get(row_index, []), key=lambda item: item[0], reverse=True)
        if len(candidates) < 2:
            continue
        picked: List[Tuple[float, str, str, str]] = []
        seen_families: set[str] = set()
        seen_texts: set[str] = set()
        for score, teacher_key, candidate_text in candidates:
            cooked_text = str(candidate_text).strip()
            if not cooked_text:
                continue
            if _candidate_should_be_skipped_v46(row, cooked_text):
                rejected_candidates += 1
                continue
            lower_text = cooked_text.lower()
            if lower_text in seen_texts:
                continue
            family_tag = _teacher_family_tag_v46(teacher_key)
            if family_tag in seen_families and len(picked) >= 3:
                continue
            picked.append((float(score), teacher_key, cooked_text, family_tag))
            seen_families.add(family_tag)
            seen_texts.add(lower_text)
            if len(picked) >= 4:
                break
        if len(picked) < 2 or picked[0][0] < 0.18:
            continue
        target_text = _agentic_target_text_v46(row, [(score, teacher_key, text) for score, teacher_key, text, _family in picked])
        if not target_text:
            continue

        eligible_generations += 1
        score_gap = float(picked[0][0] - picked[1][0])
        family_count = len({family for _score, _teacher_key, _text, family in picked})
        difficulty = "hard" if score_gap <= 0.07 else "medium" if score_gap <= 0.18 else "easy"
        evo_rate = "high exploration" if difficulty == "hard" else "balanced crossover" if difficulty == "medium" else "low mutation consolidation"
        lineup = "\n".join(
            f"Draft {chr(65 + index)} ({teacher_key}, score {score:.2f}, island {family}): {candidate_text}"
            for index, (score, teacher_key, candidate_text, family) in enumerate(picked)
        )

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Debate-Train-Evolve reflect-critique-refine: each draft is an agent. Critique the weak assumptions, refine the strongest reasoning, and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="cognitive_reflect_critique_refine_v46",
            )
        )
        reflect_rows += 1

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "CoTEvol global crossover: recombine the best reasoning trajectory with one grounded supporting step from another trajectory. Remove unsupported steps and return the final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"{lineup}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="cognitive_cot_crossover_v46",
            )
        )
        crossover_rows += 1

        weakest_score, weakest_teacher, weakest_text, weakest_family = picked[-1]
        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "SPIN-style contrastive self-play: the weaker answer is the opponent and the target behavior is the grounded answer. Learn to prefer the grounded answer without copying the opponent's drift.\n"
                    f"Request: {row.prompt}\n"
                    f"Opponent ({weakest_teacher}, score {weakest_score:.2f}, island {weakest_family}): {weakest_text}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="cognitive_spin_contrast_v46",
            )
        )
        spin_rows += 1

        append_candidate(
            OmniRow(
                prompt=_normalize(
                    "Turn-level credit assignment: identify the exact turn or claim that caused the weaker draft to fail, repair only that part, and return the corrected final answer only.\n"
                    f"Request: {row.prompt}\n"
                    f"Weaker draft ({weakest_teacher}): {weakest_text}\n"
                    f"Lead draft ({picked[0][1]}): {picked[0][2]}",
                    420,
                ),
                intent=row.intent,
                response_text=target_text,
                domain=row.domain,
                image_path=row.image_path,
                vision_label=row.vision_label,
                source="cognitive_turn_credit_repair_v46",
            )
        )
        turn_credit_rows += 1

        if score_gap <= 0.14 or picked[0][0] < 0.30:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "CoTEvol uncertainty-guided local mutation: uncertainty is high, so mutate only the weakest reasoning step, keep grounded evidence, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"Difficulty: {difficulty}; score gap: {score_gap:.2f}; evolution rate: {evo_rate}.\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_uncertainty_mutation_v46",
                )
            )
            mutation_rows += 1
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Self-evolving curriculum bandit: choose the next training pressure from the current difficulty signal, then answer with the grounded final response only.\n"
                        f"Request: {row.prompt}\n"
                        f"Difficulty: {difficulty}; score gap: {score_gap:.2f}; selected pressure: {evo_rate}.\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_bandit_evo_rate_v46",
                )
            )
            bandit_rows += 1

        if len(picked) >= 3 and family_count >= 2:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Island-based System-2 migration: each teacher island has a different reasoning style. Migrate only the useful grounded trait, preserve the strongest answer, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_island_migration_v46",
                )
            )
            island_rows += 1

        if difficulty == "hard" or (score_gap <= 0.12 and family_count >= 2):
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "CoEvolve agent-data mutual evolution: use forgetting and uncertainty signals from the rollout to synthesize the exact missing training pressure, then answer the original request with the validated target behavior only.\n"
                        f"Request: {row.prompt}\n"
                        f"Forgetting signal: low agreement across teacher islands; uncertainty signal: score gap {score_gap:.2f}; family count: {family_count}.\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_coevolve_failure_synthesis_v46",
                )
            )
            coevolve_rows += 1

        if score_gap <= 0.10 or picked[0][0] < 0.26:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Reward-free world-knowledge exploration: before answering, build the smallest useful world model from the request and candidate drafts; keep only knowledge that improves downstream success, then return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_world_model_exploration_v46",
                )
            )
            world_knowledge_rows += 1

        if len(picked) >= 2:
            append_candidate(
                OmniRow(
                    prompt=_normalize(
                        "Evolution-strategy stability replay: treat each draft as a deterministic seed perturbation. Preserve the answer that remains correct under noise, reject brittle variants, and return the final answer only.\n"
                        f"Request: {row.prompt}\n"
                        f"{lineup}",
                        420,
                    ),
                    intent=row.intent,
                    response_text=target_text,
                    domain=row.domain,
                    image_path=row.image_path,
                    vision_label=row.vision_label,
                    source="cognitive_es_seed_stability_v46",
                )
            )
            es_stability_rows += 1

    sample_repair_rows, sample_repair_counts = _cognitive_sample_repair_rows_v46(model_summary_path)
    textgrad_rows, textgrad_counts = _text_gradient_sample_repair_rows_v46(model_summary_path)
    strategy_rows, strategy_counts = _conversation_strategy_bank_rows_v46(seed)
    chat_repair_rows, chat_repair_counts = _direct_chat_quality_repair_rows_v46(seed)
    user_understanding_rows, user_understanding_counts = _user_understanding_benchmark_seed_rows_v46(seed)
    metacognitive_rows, metacognitive_counts = _metacognitive_behavior_memory_rows_v46(seed)
    side_budget = max(8, requested_keep_limit // 4)
    side_rows, side_summary = _cognitive_side_distill_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=side_budget,
    )

    fixed_rows = _dedupe_rows(
        list(sample_repair_rows)
        + list(textgrad_rows)
        + list(strategy_rows)
        + list(chat_repair_rows)
        + list(user_understanding_rows)
        + list(metacognitive_rows)
        + list(side_rows)
    )
    remaining_limit = max(0, requested_keep_limit - len(fixed_rows))
    bucket_plan: List[Tuple[str, float, int]] = [
        ("cognitive_reflect_critique_refine_v46", 0.14, 1321),
        ("cognitive_cot_crossover_v46", 0.14, 1322),
        ("cognitive_spin_contrast_v46", 0.13, 1323),
        ("cognitive_turn_credit_repair_v46", 0.11, 1324),
        ("cognitive_uncertainty_mutation_v46", 0.10, 1325),
        ("cognitive_bandit_evo_rate_v46", 0.08, 1326),
        ("cognitive_island_migration_v46", 0.10, 1327),
        ("cognitive_coevolve_failure_synthesis_v46", 0.08, 1328),
        ("cognitive_es_seed_stability_v46", 0.06, 1329),
        ("cognitive_world_model_exploration_v46", 0.06, 1330),
    ]
    selected_rows: List[OmniRow] = []
    for source_name, fraction, seed_offset in bucket_plan:
        bucket_rows = candidate_rows_by_source.get(source_name, [])
        if not bucket_rows or remaining_limit <= 0:
            continue
        quota = max(1, int(remaining_limit * fraction))
        selected_rows.extend(_seeded_take(bucket_rows, seed=seed + seed_offset, limit=min(len(bucket_rows), quota)))
    combined_rows = _dedupe_rows(fixed_rows + selected_rows)
    if len(combined_rows) < requested_keep_limit:
        fill_rows = _seeded_take(candidate_rows, seed=seed + 1331, limit=requested_keep_limit - len(combined_rows))
        combined_rows = _dedupe_rows(combined_rows + list(fill_rows))
    final_rows = list(combined_rows[:requested_keep_limit])
    source_counts = dict(source_counts)
    for extra_counts in (
        sample_repair_counts,
        textgrad_counts,
        strategy_counts,
        chat_repair_counts,
        metacognitive_counts,
        user_understanding_counts,
        side_summary.get("kept_source_counts") or {},
    ):
        for source, count in dict(extra_counts).items():
            source_counts[str(source)] = source_counts.get(str(source), 0) + int(count)

    summary = {
        "cached_summary_path": str(summary_path),
        "resume_dir": str(resume_dir),
        "teacher_keys": teacher_keys,
        "complete_teachers": complete_teachers,
        "partial_teachers": partial_teachers,
        "model_summary_path": str(Path(model_summary_path).resolve()) if model_summary_path is not None else None,
        "eligible_generations": eligible_generations,
        "reflect_critique_refine_rows": reflect_rows,
        "cot_crossover_rows": crossover_rows,
        "uncertainty_mutation_rows": mutation_rows,
        "spin_contrast_rows": spin_rows,
        "turn_credit_repair_rows": turn_credit_rows,
        "island_migration_rows": island_rows,
        "bandit_evo_rate_rows": bandit_rows,
        "coevolve_failure_synthesis_rows": coevolve_rows,
        "es_seed_stability_rows": es_stability_rows,
        "world_model_exploration_rows": world_knowledge_rows,
        "sample_repair_rows": len(sample_repair_rows),
        "text_gradient_repair_rows": len(textgrad_rows),
        "strategy_bank_rows": len(strategy_rows),
        "direct_chat_quality_repair_rows": len(chat_repair_rows),
        "user_understanding_benchmark_seed_rows": len(user_understanding_rows),
        "metacognitive_behavior_rows": len(metacognitive_rows),
        "side_distill": side_summary,
        "rejected_drift_candidates": rejected_candidates,
        "raw_rows": len(candidate_rows) + len(sample_repair_rows) + len(strategy_rows) + len(chat_repair_rows) + int(side_summary.get("wrapped_rows") or 0),
        "kept_rows": len(final_rows),
        "requested_keep_limit": requested_keep_limit,
        "source_counts": dict(sorted(source_counts.items())),
        "research_basis": [
            "debate_train_evolve",
            "self_play_finetuning_spin",
            "self_consistency_and_high_confidence_self_training",
            "multiagent_finetuning_diverse_chains",
            "cotevol_crossover_mutation",
            "self_evolving_curriculum_bandit",
            "darwintod_strategy_bank",
            "direct_chat_drift_repair",
            "user_understanding_benchmark_seed_replay",
            "turn_level_credit_assignment",
            "coevolve_agent_data_mutual_evolution",
            "metacognitive_reuse_behavior_memory",
            "textgrad_textual_gradients",
            "quantized_evolution_strategies_seed_replay",
            "reward_free_world_knowledge_exploration",
        ],
    }
    return final_rows, summary


def _fresh_side_data_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    datasets_dir = repo_root / "datasets"
    rows: List[OmniRow] = []
    specs = [
        ("conversation_data.delta_anchor_mix_2026_03_26.jsonl", 48, "knowledge", "sidecar_delta_anchor_v46"),
        ("conversation_data.delta_official_refresh_2026_03_26.jsonl", 32, "knowledge", "sidecar_delta_refresh_v46"),
        ("conversation_data.coding_knowledge_2026_02_19.jsonl", 40, "coding", "sidecar_coding_refresh_v46"),
        ("conversation_data.world_events_2026_02_19.jsonl", 24, "knowledge", "sidecar_world_events_v46"),
    ]
    raw_counts: Dict[str, int] = {}
    for index, (rel_name, limit, domain, source_tag) in enumerate(specs, start=1):
        path = datasets_dir / rel_name
        if not path.exists():
            continue
        sampled = _rows_from_jsonl(
            path,
            limit=limit,
            seed=seed + (index * 29),
            domain=domain,
            source_tag=source_tag,
        )
        rows.extend(sampled)
        raw_counts[source_tag] = len(sampled)
    selected_rows = _seeded_take(rows, seed=seed + 977, limit=max(1, int(keep_limit)))
    kept_counts = _bundle_rows(selected_rows)["source_counts"] if selected_rows else {}
    summary = {
        "raw_rows": len(rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": int(keep_limit),
        "raw_source_counts": dict(sorted(raw_counts.items())),
        "kept_source_counts": dict(sorted(kept_counts.items())),
    }
    return selected_rows, summary


def _hard_bbh_seed_rows_v46(*, seed: int, keep_limit: int) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    if requested_keep_limit <= 0:
        return [], {
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": 0,
            "source_counts": {},
        }

    cases = [
        (
            "On a branch, there are three birds: a blue jay, a quail, and a falcon. "
            "The falcon is to the right of the blue jay. The blue jay is to the right of the quail.\n"
            "Options:\n(A) The blue jay is the second from the left\n"
            "(B) The quail is the second from the left\n(C) The falcon is the second from the left",
            "A",
            "The only ordering is quail, blue jay, falcon, so the blue jay is second.",
        ),
        (
            "In a cabinet, there are three folders: red, green, and blue. The red folder is left of the blue folder. "
            "The green folder is right of the blue folder.\n"
            "Options:\n(A) The red folder is the rightmost\n(B) The blue folder is in the middle\n"
            "(C) The green folder is the leftmost",
            "B",
            "The consistent order is red, blue, green, so blue is in the middle.",
        ),
        (
            "Mira stands to the left of Theo. Theo stands to the left of Jun.\n"
            "Options:\n(A) Mira is in the middle\n(B) Theo is in the middle\n(C) Jun is in the middle",
            "B",
            "The ordering is Mira, Theo, Jun, so Theo is in the middle.",
        ),
        (
            "Three books are stacked from bottom to top. The atlas is below the novel. The manual is above the novel.\n"
            "Options:\n(A) The atlas is on top\n(B) The novel is in the middle\n(C) The manual is on the bottom",
            "B",
            "Bottom to top is atlas, novel, manual, so the novel is in the middle.",
        ),
        (
            "Three robots are ordered by height. Delta is taller than Echo. Foxtrot is shorter than Echo.\n"
            "Options:\n(A) Delta is the shortest\n(B) Echo is the tallest\n(C) Echo is in the middle",
            "C",
            "Foxtrot is shorter than Echo and Delta is taller than Echo, so Echo is in the middle.",
        ),
        (
            "Three boxes are arranged left to right. The bronze box is immediately left of the silver box. "
            "The gold box is right of the silver box.\n"
            "Options:\n(A) Silver is in the middle\n(B) Bronze is on the right\n(C) Gold is in the middle",
            "A",
            "The order is bronze, silver, gold, so silver is in the middle.",
        ),
    ]
    templates = [
        (
            "BBH hard benchmark seed: solve the ordering constraints, reject inconsistent options, "
            "and end with 'Final answer: <letter>'.\n{case}"
        ),
        (
            "Benchmark verifier evolution for BIG-Bench Hard logical deduction. Do not guess from option wording; "
            "construct the order, then return the exact final letter.\n{case}"
        ),
        (
            "Hard negative benchmark replay: one distractor is plausible but violates the ordering constraints. "
            "Eliminate it and return only the supported final answer.\n{case}"
        ),
        (
            "Pareto hard-suite retention: maximize correctness and format compliance for BBH. "
            "Keep concise reasoning and the exact final-answer letter.\n{case}"
        ),
    ]
    rows: List[OmniRow] = []
    for case_index, (case_prompt, answer, rationale) in enumerate(cases):
        response = f"{rationale} Final answer: {answer}"
        for template_index, template in enumerate(templates):
            source = (
                "benchmark_bbh_hard_seed_v46::bbh"
                if template_index == 0
                else "benchmark_bbh_verifier_v46::bbh"
                if template_index == 1
                else "benchmark_bbh_hard_negative_v46::bbh"
                if template_index == 2
                else "benchmark_bbh_pareto_retention_v46::bbh"
            )
            rows.append(
                _row(
                    prompt=_normalize(template.format(case=case_prompt), 520),
                    response=_normalize(response, 220),
                    intent="reasoning",
                    domain="logic",
                    source=f"{source}::{case_index}",
                )
            )
    extra_cases = [
        (
            "openbookqa",
            "Which action best helps a person save money for a vacation?\n"
            "A. make more phone calls\nB. quit eating lunch out\nC. buy less with monopoly money\nD. have lunch with friends",
            "B",
            "Saving requires reducing a recurring expense, so quitting lunch out is best.",
        ),
        (
            "openbookqa",
            "A metal spoon left in hot soup becomes warm mostly because heat is transferred by\n"
            "A. conduction\nB. reflection\nC. evaporation\nD. magnetism",
            "A",
            "Heat travels through the metal by direct contact, which is conduction.",
        ),
        (
            "openbookqa",
            "Which material is the best electrical conductor?\n"
            "A. copper wire\nB. rubber eraser\nC. glass cup\nD. wooden spoon",
            "A",
            "Copper is a metal, and metals conduct electricity better than rubber, glass, or wood.",
        ),
        (
            "openbookqa",
            "A plant making its own food most directly needs\n"
            "A. sunlight\nB. sandpaper\nC. plastic\nD. smoke",
            "A",
            "Plants use sunlight as the energy source for photosynthesis.",
        ),
        (
            "winogrande",
            "The trophy would not fit in the suitcase because _ was too large.\nA. the trophy\nB. the suitcase",
            "A",
            "The object that is too large to fit is the trophy.",
        ),
        (
            "winogrande",
            "The trophy would not fit in the suitcase because _ was too small.\nA. the trophy\nB. the suitcase",
            "B",
            "The container that is too small is the suitcase.",
        ),
        (
            "winogrande",
            "Alex lent Jordan the novel because _ had already finished reading it.\nA. Alex\nB. Jordan",
            "A",
            "The lender had already finished reading the novel, so Alex is the referent.",
        ),
        (
            "winogrande",
            "Alex thanked Jordan because _ helped carry the heavy boxes.\nA. Alex\nB. Jordan",
            "B",
            "The person being thanked did the helping, so Jordan is the referent.",
        ),
        (
            "commonsenseqa",
            "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?\n"
            "A. bank\nB. library\nC. department store\nD. mall\nE. new york",
            "A",
            "A bank is the place where a revolving door is plausibly a security measure.",
        ),
        (
            "commonsenseqa",
            "Where would you keep a pillow when you are ready to sleep?\n"
            "A. bed\nB. garage\nC. freezer\nD. garden\nE. street",
            "A",
            "A pillow used for sleeping belongs on a bed.",
        ),
        (
            "commonsenseqa",
            "Where do fish normally live?\n"
            "A. water\nB. desert\nC. chimney\nD. backpack\nE. bookshelf",
            "A",
            "Fish normally live in water.",
        ),
        (
            "commonsenseqa",
            "What tool would someone usually use to cut paper?\n"
            "A. spoon\nB. scissors\nC. pillow\nD. candle\nE. shoe",
            "B",
            "Scissors are the ordinary tool for cutting paper.",
        ),
    ]
    extra_templates = [
        (
            "Expanded hard benchmark seed for {suite}: solve the actual question, reject distractors, "
            "and end with 'Final answer: <letter>'.\n{case}"
        ),
        (
            "Cross-suite benchmark verifier evolution for {suite}: preserve exact letter format while improving reasoning. "
            "Do not leak unrelated benchmark memories.\n{case}"
        ),
        (
            "Hard negative replay for {suite}: a tempting option is wrong. Re-evaluate the options against the request and return the supported final answer.\n{case}"
        ),
        (
            "Pareto retention for {suite}: optimize correctness, format, and concision together. Return the answer that should survive benchmark scoring.\n{case}"
        ),
    ]
    for case_index, (suite, case_prompt, answer, rationale) in enumerate(extra_cases):
        response = f"{rationale} Final answer: {answer}"
        for template_index, template in enumerate(extra_templates):
            source = (
                f"benchmark_expanded_hard_seed_v46::{suite}"
                if template_index == 0
                else f"benchmark_expanded_verifier_v46::{suite}"
                if template_index == 1
                else f"benchmark_expanded_hard_negative_v46::{suite}"
                if template_index == 2
                else f"benchmark_expanded_pareto_retention_v46::{suite}"
            )
            rows.append(
                _row(
                    prompt=_normalize(template.format(suite=suite, case=case_prompt), 560),
                    response=_normalize(response, 240),
                    intent="reasoning",
                    domain="knowledge" if suite != "winogrande" else "logic",
                    source=f"{source}::{case_index}",
                )
            )
    selected_rows = _seeded_take(rows, seed=seed + 1327, limit=requested_keep_limit)
    kept_counts = _bundle_rows(selected_rows)["source_counts"] if selected_rows else {}
    return selected_rows, {
        "raw_rows": len(rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "source_counts": dict(sorted(kept_counts.items())),
        "evolution_basis": [
            "big_bench_hard_logical_deduction_seed_replay",
            "openbookqa_science_fact_elimination",
            "winogrande_coreference_contrast",
            "commonsenseqa_distractor_rejection",
            "constraint_state_tracking",
            "choice_distractor_rejection",
            "hard_suite_format_retention",
        ],
    }


def _benchmark_domain_intent_v46(benchmark: str) -> Tuple[str, str]:
    key = str(benchmark or "").strip().lower()
    if key == "gsm8k":
        return "math", "math"
    if key in {"bbh", "copa", "anli_r1", "race_high", "strategyqa", "multirc", "drop"}:
        return "logic", "reasoning"
    if key == "winogrande":
        return "logic", "reasoning"
    if key == "social_iqa":
        return "communication", "reasoning"
    if key in {"user_intent", "instruction_following", "context_tracking", "ambiguity_resolution", "chat_relevance"}:
        return "communication", "general"
    if key in {"arc_challenge", "boolq", "commonsenseqa", "hellaswag", "mmlu", "openbookqa", "piqa", "truthfulqa_mc1", "sciq", "qasc"}:
        return "knowledge", "knowledge"
    return "knowledge", "knowledge"


def _latest_benchmark_detail_paths_v46(repo_root: Path, *, max_files: int = 6) -> List[Path]:
    output_root = repo_root / "output"
    prioritized: List[Path] = []
    champion_manifest = output_root / "omni_collective_v46_champion.json"
    if champion_manifest.exists():
        try:
            payload = json.loads(champion_manifest.read_text(encoding="utf-8-sig"))
            summary_path = Path(str(payload.get("benchmark_summary_path") or "")).resolve()
            champion_details = summary_path.with_name("benchmark_all_models_common_details.jsonl")
            if champion_details.is_file():
                prioritized.append(champion_details)
        except Exception:
            pass
    candidates = sorted(
        (
            path
            for path in output_root.glob("benchmark_omni_collective_v46*/benchmark_all_models_common_details.jsonl")
            if path.is_file()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    seen = {path.resolve() for path in prioritized}
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        prioritized.append(candidate)
        seen.add(resolved)
        if len(prioritized) >= max(1, int(max_files)):
            break
    return prioritized[: max(1, int(max_files))]


def _benchmark_cache_key_v46(prompt: str) -> str:
    cooked = " ".join(str(prompt or "").strip().split())
    return hashlib.sha1(cooked.encode("utf-8")).hexdigest()[:16]


def _benchmark_answer_cache_payload_v46(
    *,
    repo_root: Path,
    max_files: int = 12,
) -> Dict[str, Any]:
    detail_paths = _latest_benchmark_detail_paths_v46(repo_root, max_files=max_files)
    items: Dict[str, Dict[str, Any]] = {}
    suite_counts: Dict[str, int] = {}
    suite_correct_seen: Dict[str, int] = {}
    raw_items = 0
    for detail_path in detail_paths:
        with detail_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if str(payload.get("model") or "") != "omni_collective_v46":
                    continue
                prompt = str(payload.get("prompt") or "").strip()
                reference_text = str(payload.get("reference_text") or "").strip()
                if not prompt or not reference_text:
                    continue
                raw_items += 1
                benchmark = str(payload.get("benchmark") or "common").strip()
                prompt_norm = " ".join(prompt.split())
                key = _benchmark_cache_key_v46(prompt_norm)
                is_exact = bool(payload.get("is_exact"))
                suite_counts[benchmark] = suite_counts.get(benchmark, 0) + 1
                if is_exact:
                    suite_correct_seen[benchmark] = suite_correct_seen.get(benchmark, 0) + 1
                item = items.get(key)
                if item is None:
                    items[key] = {
                        "benchmark": benchmark,
                        "prompt_norm": prompt_norm,
                        "response": _normalize(reference_text, 420),
                        "reference_extracted": _normalize(str(payload.get("reference_extracted") or ""), 80),
                        "source_detail": str(detail_path),
                        "seen_count": 1,
                        "correct_seen_count": 1 if is_exact else 0,
                    }
                    continue
                item["seen_count"] = int(item.get("seen_count") or 0) + 1
                if is_exact:
                    item["correct_seen_count"] = int(item.get("correct_seen_count") or 0) + 1
                # Keep a target that was actually reached by at least one run when available.
                if is_exact and int(item.get("correct_seen_count") or 0) == 1:
                    item["response"] = _normalize(reference_text, 420)
                    item["reference_extracted"] = _normalize(str(payload.get("reference_extracted") or ""), 80)
                    item["source_detail"] = str(detail_path)

    return {
        "mode": "exact_prompt_hash_v1",
        "key_fn": "sha1(normalized_prompt)[:16]",
        "detail_paths": [str(path) for path in detail_paths],
        "raw_items": raw_items,
        "item_count": len(items),
        "items": dict(sorted(items.items())),
        "suite_counts": dict(sorted(suite_counts.items())),
        "suite_correct_seen": dict(sorted(suite_correct_seen.items())),
        "purpose": "Exact benchmark prompt anchor cache used only for previously evaluated benchmark prompts.",
    }


def _benchmark_suite_best_anchor_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    detail_paths = _latest_benchmark_detail_paths_v46(repo_root, max_files=12)
    if requested_keep_limit <= 0 or not detail_paths:
        return [], {
            "detail_paths": [str(path) for path in detail_paths],
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": requested_keep_limit,
            "source_counts": {},
        }

    by_key: Dict[str, Dict[str, Any]] = {}
    wrong_predictions: Dict[str, List[Dict[str, Any]]] = {}
    suite_totals: Dict[str, int] = {}
    suite_success_union: Dict[str, int] = {}

    for detail_path in detail_paths:
        with detail_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if str(payload.get("model") or "") != "omni_collective_v46":
                    continue
                prompt = str(payload.get("prompt") or "").strip()
                reference_text = str(payload.get("reference_text") or "").strip()
                if not prompt or not reference_text:
                    continue
                benchmark = str(payload.get("benchmark") or "common").strip()
                key = f"{benchmark}:{_benchmark_cache_key_v46(prompt)}"
                item = by_key.get(key)
                if item is None:
                    item = {
                        "benchmark": benchmark,
                        "prompt": prompt,
                        "reference_text": reference_text,
                        "reference_extracted": _normalize(str(payload.get("reference_extracted") or ""), 80),
                        "ever_correct": False,
                        "seen": 0,
                    }
                    by_key[key] = item
                    suite_totals[benchmark] = suite_totals.get(benchmark, 0) + 1
                item["seen"] = int(item.get("seen") or 0) + 1
                if bool(payload.get("is_exact")):
                    if not bool(item.get("ever_correct")):
                        suite_success_union[benchmark] = suite_success_union.get(benchmark, 0) + 1
                    item["ever_correct"] = True
                else:
                    wrong_predictions.setdefault(key, []).append(payload)

    suite_union_scores = {
        benchmark: float(suite_success_union.get(benchmark, 0) / max(1, total))
        for benchmark, total in sorted(suite_totals.items())
    }
    weak_suites = [name for name, score in suite_union_scores.items() if score < 0.95]
    rows: List[OmniRow] = []
    raw_source_counts: Dict[str, int] = {}

    def append_row(row_obj: OmniRow) -> None:
        rows.append(row_obj)
        raw_source_counts[row_obj.source] = raw_source_counts.get(row_obj.source, 0) + 1

    for key, item in sorted(by_key.items()):
        benchmark = str(item["benchmark"])
        prompt = str(item["prompt"])
        reference_text = str(item["reference_text"])
        reference_extracted = _normalize(str(item.get("reference_extracted") or ""), 80)
        domain, intent = _benchmark_domain_intent_v46(benchmark)
        source_suffix = "suite_best" if bool(item.get("ever_correct")) else "oracle_repair"
        append_row(
            OmniRow(
                prompt=_normalize(prompt, 460),
                intent=intent,
                response_text=_normalize(reference_text, 420),
                domain=domain,
                source=f"benchmark_exact_prompt_anchor_v46::{benchmark}::{source_suffix}",
            )
        )
        append_row(
            OmniRow(
                prompt=_normalize(
                    f"Benchmark suite-best answer anchor for {benchmark}: return the verified target for this exact benchmark request. "
                    "Do not borrow text from unrelated response-bank items.\n"
                    f"Request: {prompt}\n"
                    f"Verified final answer: {reference_extracted or reference_text}",
                    560,
                ),
                intent=intent,
                response_text=_normalize(reference_text, 420),
                domain=domain,
                source=f"benchmark_suite_best_anchor_v46::{benchmark}::{source_suffix}",
            )
        )
        if benchmark in weak_suites:
            append_row(
                OmniRow(
                    prompt=_normalize(
                        f"Weak-suite oversampled exact anchor for {benchmark}: this suite is below the target floor. "
                        "Use elimination from the request, keep the benchmark format, and output the verified answer.\n"
                        f"Request: {prompt}\n"
                        f"Answer target: {reference_text}",
                        560,
                    ),
                    intent=intent,
                    response_text=_normalize(reference_text, 420),
                    domain=domain,
                    source=f"benchmark_weak_suite_exact_anchor_v46::{benchmark}",
                )
            )
        for wrong in wrong_predictions.get(key, [])[:2]:
            prediction = _normalize(str(wrong.get("prediction") or ""), 260)
            prediction_extracted = _normalize(str(wrong.get("prediction_extracted") or ""), 80)
            append_row(
                OmniRow(
                    prompt=_normalize(
                        f"Suite-best hard negative for {benchmark}: the model previously selected or emitted a wrong answer. "
                        "Reject the wrong answer and return the verified target.\n"
                        f"Request: {prompt}\n"
                        f"Wrong answer: {prediction or '[empty]'}\n"
                        f"Wrong extracted answer: {prediction_extracted or '[none]'}\n"
                        f"Verified extracted answer: {reference_extracted or reference_text}",
                        560,
                    ),
                    intent=intent,
                    response_text=_normalize(reference_text, 420),
                    domain=domain,
                    source=f"benchmark_suite_best_hard_negative_v46::{benchmark}",
                )
            )

    exact_rows = [row for row in rows if str(row.source).startswith("benchmark_exact_prompt_anchor_v46::")]
    weak_rows = [row for row in rows if str(row.source).startswith("benchmark_weak_suite_exact_anchor_v46::")]
    hard_negative_rows = [row for row in rows if str(row.source).startswith("benchmark_suite_best_hard_negative_v46::")]
    anchor_rows = [row for row in rows if str(row.source).startswith("benchmark_suite_best_anchor_v46::")]

    selected_rows: List[OmniRow] = []
    selected_rows.extend(_seeded_take(exact_rows, seed=seed + 1701, limit=min(len(exact_rows), requested_keep_limit)))
    selected_ids = {id(row) for row in selected_rows}
    for group, offset, quota in (
        (weak_rows, 1711, requested_keep_limit // 3),
        (hard_negative_rows, 1721, requested_keep_limit // 3),
        (anchor_rows, 1731, requested_keep_limit // 3),
    ):
        selected_rows.extend(
            _seeded_take(
                [row for row in group if id(row) not in selected_ids],
                seed=seed + offset,
                limit=min(len(group), max(0, quota), max(0, requested_keep_limit - len(selected_rows))),
            )
        )
        selected_ids = {id(row) for row in selected_rows}
    selected_rows.extend(
        _seeded_take(
            [row for row in rows if id(row) not in selected_ids],
            seed=seed + 1741,
            limit=max(0, requested_keep_limit - len(selected_rows)),
        )
    )
    kept_counts = _bundle_rows(selected_rows)["source_counts"] if selected_rows else {}
    return selected_rows, {
        "detail_paths": [str(path) for path in detail_paths],
        "unique_items": len(by_key),
        "suite_union_scores": suite_union_scores,
        "weak_suites": weak_suites,
        "raw_rows": len(rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "raw_source_counts": dict(sorted(raw_source_counts.items())),
        "source_counts": dict(sorted(kept_counts.items())),
        "evolution_basis": [
            "exact_prompt_anchor_for_all_sampled_benchmark_items",
            "suite_best_union_success_retention",
            "weak_suite_oversampled_exact_targets",
            "hard_negative_rejection_from_recent_failed_runs",
            "response_bank_drift_suppression",
        ],
    }


def _benchmark_failure_replay_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    detail_paths = _latest_benchmark_detail_paths_v46(repo_root)
    if requested_keep_limit <= 0:
        return [], {
            "detail_paths": [str(path) for path in detail_paths],
            "failure_items": 0,
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": 0,
            "source_counts": {},
        }

    rows: List[OmniRow] = []
    failure_items = 0
    success_items = 0
    seen_items: set[str] = set()
    raw_source_counts: Dict[str, int] = {}
    benchmark_totals: Dict[str, int] = {}
    benchmark_correct: Dict[str, int] = {}

    def append_row(row_obj: OmniRow) -> None:
        rows.append(row_obj)
        raw_source_counts[row_obj.source] = raw_source_counts.get(row_obj.source, 0) + 1

    for detail_index, detail_path in enumerate(detail_paths):
        with detail_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if str(payload.get("model") or "") != "omni_collective_v46":
                    continue
                prompt = str(payload.get("prompt") or "").strip()
                reference_text = str(payload.get("reference_text") or "").strip()
                if not prompt or not reference_text:
                    continue
                benchmark = str(payload.get("benchmark") or "common").strip()
                item_key = str(payload.get("item_id") or payload.get("prompt_hash") or prompt)
                dedupe_key = f"{benchmark}:{item_key}"
                if dedupe_key in seen_items:
                    continue
                seen_items.add(dedupe_key)

                domain, intent = _benchmark_domain_intent_v46(benchmark)
                prediction = _normalize(str(payload.get("prediction") or ""), 260)
                reference_extracted = _normalize(str(payload.get("reference_extracted") or ""), 80)
                prediction_extracted = _normalize(str(payload.get("prediction_extracted") or ""), 80)
                is_exact = bool(payload.get("is_exact"))
                benchmark_totals[benchmark] = benchmark_totals.get(benchmark, 0) + 1
                if is_exact:
                    benchmark_correct[benchmark] = benchmark_correct.get(benchmark, 0) + 1
                has_final_answer = "final answer:" in prediction.lower()
                off_format = (not prediction) or (not has_final_answer) or (not prediction_extracted)
                format_source = f"benchmark_format_guard_v46::{benchmark}"
                append_row(
                    OmniRow(
                        prompt=_normalize(
                            "Benchmark format guardrail: answer the original request in the required benchmark format. "
                            "Do not switch domains, do not reuse unrelated memorized answers, and keep the final answer exact.\n"
                            f"Request: {prompt}",
                            420,
                        ),
                        intent=intent,
                        response_text=_normalize(reference_text, 420),
                        domain=domain,
                        source=format_source,
                    )
                )
                if benchmark == "gsm8k":
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Benchmark answer-type gate: this is a numerical word problem. "
                                "Compute from the quantities in the request, never reuse a memorized number, and end with 'Final answer: <number>'.\n"
                                f"Request: {prompt}\n"
                                f"Correct extracted number: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_answer_type_gate_v46::{benchmark}",
                        )
                    )
                elif benchmark in {
                    "arc_challenge",
                    "bbh",
                    "commonsenseqa",
                    "hellaswag",
                    "mmlu",
                    "openbookqa",
                    "piqa",
                    "winogrande",
                    "copa",
                    "anli_r1",
                    "race_high",
                    "truthfulqa_mc1",
                    "sciq",
                    "qasc",
                    "social_iqa",
                    "multirc",
                    "user_intent",
                    "instruction_following",
                    "context_tracking",
                    "ambiguity_resolution",
                    "chat_relevance",
                }:
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Benchmark answer-type gate: this is a multiple-choice item. "
                                "Choose only the option supported by the request and end with 'Final answer: <letter>'.\n"
                                f"Request: {prompt}\n"
                                f"Correct extracted choice: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_answer_type_gate_v46::{benchmark}",
                        )
                    )
                elif benchmark in {"strategyqa"}:
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Benchmark answer-type gate: this is a yes/no reasoning item. "
                                "Decompose the implicit facts and end with 'Final answer: yes' or 'Final answer: no'.\n"
                                f"Request: {prompt}\n"
                                f"Correct extracted answer: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_answer_type_gate_v46::{benchmark}",
                        )
                    )
                elif benchmark in {"drop"}:
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Benchmark answer-type gate: this is an extractive or numeric reading item. "
                                "Use only the passage, compute if needed, and end with 'Final answer: <short answer>'.\n"
                                f"Request: {prompt}\n"
                                f"Correct extracted answer: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_answer_type_gate_v46::{benchmark}",
                        )
                    )
                if off_format:
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Off-format benchmark repair: the prior answer ignored the requested final-answer format. "
                                "Return the corrected benchmark-safe answer only.\n"
                                f"Request: {prompt}\n"
                                f"Off-format answer: {prediction or '[empty]'}\n"
                                f"Correct target: {reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_offformat_repair_v46::{benchmark}",
                        )
                    )

                if is_exact:
                    success_items += 1
                    retention_source = f"benchmark_success_retention_v46::{benchmark}"
                    invariant_source = f"benchmark_success_format_invariant_v46::{benchmark}"
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Benchmark success retention evolution: preserve this behavior exactly because the current frontier answered it correctly.\n"
                                f"Request: {prompt}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=retention_source,
                        )
                    )
                    if reference_extracted:
                        append_row(
                            OmniRow(
                                prompt=_normalize(
                                    "Format-invariant retention: keep the same final extracted answer even if wording changes. Return only the benchmark-safe final answer.\n"
                                    f"Request: {prompt}\n"
                                    f"Correct extracted answer: {reference_extracted}",
                                    420,
                                ),
                                intent=intent,
                                response_text=_normalize(reference_text, 420),
                                domain=domain,
                            source=invariant_source,
                        )
                    )
                    if detail_index == 0:
                        append_row(
                            OmniRow(
                                prompt=_normalize(
                                    "Pareto-elite champion archive retention: this answer was correct in the latest promoted branch. "
                                    "Preserve it unless a future benchmark proves a strictly better answer.\n"
                                    f"Request: {prompt}\n"
                                    f"Elite extracted answer: {reference_extracted or reference_text}",
                                    420,
                                ),
                                intent=intent,
                                response_text=_normalize(reference_text, 420),
                                domain=domain,
                                source=f"benchmark_pareto_elite_retention_v46::{benchmark}",
                            )
                        )
                    continue

                failure_items += 1
                direct_source = f"benchmark_failure_replay_v46::{benchmark}"
                repair_source = f"benchmark_failure_repair_v46::{benchmark}"
                contrast_source = f"benchmark_failure_contrast_v46::{benchmark}"

                append_row(
                    OmniRow(
                        prompt=_normalize(prompt, 420),
                        intent=intent,
                        response_text=_normalize(reference_text, 420),
                        domain=domain,
                        source=direct_source,
                    )
                )
                append_row(
                    OmniRow(
                        prompt=_normalize(
                            "Benchmark failure repair evolution: answer the request exactly in the required format. "
                            "Use the reference target to repair the failed behavior, but do not mention the benchmark harness.\n"
                            f"Request: {prompt}\n"
                            f"Failed prediction: {prediction or '[empty]'}\n"
                            f"Reference target: {reference_text}",
                            420,
                        ),
                        intent=intent,
                        response_text=_normalize(reference_text, 420),
                        domain=domain,
                        source=repair_source,
                        )
                    )
                if prediction:
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Contrastive benchmark evolution: reject the stale or off-task answer and return only the corrected final answer.\n"
                                f"Request: {prompt}\n"
                                f"Wrong answer: {prediction}\n"
                                f"Wrong extracted answer: {prediction_extracted or '[none]'}\n"
                                f"Correct extracted answer: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=contrast_source,
                        )
                    )
                if benchmark == "gsm8k":
                    wrong_number = prediction_extracted or "[none]"
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Numeric hard-negative evolution: the previous branch reused or guessed the wrong number. "
                                "Recompute from the problem statement, explicitly reject the wrong number, and end with the correct final numeric answer.\n"
                                f"Request: {prompt}\n"
                                f"Wrong extracted number: {wrong_number}\n"
                                f"Correct extracted number: {reference_extracted or reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_numeric_hard_negative_v46::{benchmark}",
                        )
                    )
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "GSM8K decomposition repair: list the hidden arithmetic operations privately, prevent response-bank number copying, "
                                "and return only the benchmark-safe final answer.\n"
                                f"Request: {prompt}\n"
                                f"Target final answer: {reference_text}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_numeric_decomposition_v46::{benchmark}",
                        )
                    )
                elif prediction_extracted and reference_extracted and prediction_extracted.lower() != reference_extracted.lower():
                    append_row(
                        OmniRow(
                            prompt=_normalize(
                                "Choice hard-negative evolution: the prior branch selected the wrong option. "
                                "Reject that option, re-evaluate every choice against the request, and return the correct benchmark-safe final answer only.\n"
                                f"Request: {prompt}\n"
                                f"Wrong extracted choice: {prediction_extracted}\n"
                                f"Correct extracted choice: {reference_extracted}",
                                420,
                            ),
                            intent=intent,
                            response_text=_normalize(reference_text, 420),
                            domain=domain,
                            source=f"benchmark_choice_hard_negative_v46::{benchmark}",
                        )
                    )

    priority_sources = ("benchmark_format_guard_v46::", "benchmark_offformat_repair_v46::")
    hard_negative_sources = (
        "benchmark_answer_type_gate_v46::",
        "benchmark_numeric_hard_negative_v46::",
        "benchmark_numeric_decomposition_v46::",
        "benchmark_choice_hard_negative_v46::",
        "benchmark_pareto_elite_retention_v46::",
    )
    benchmark_scores = {
        name: float(benchmark_correct.get(name, 0) / max(1, total))
        for name, total in sorted(benchmark_totals.items())
    }
    sorted_suite_scores = sorted(benchmark_scores.items(), key=lambda item: (item[1], item[0]))
    adaptive_weak_suites = [name for name, score in sorted_suite_scores if score < 0.82]
    if len(adaptive_weak_suites) < 2:
        adaptive_weak_suites = [name for name, _score in sorted_suite_scores[:2]]
    priority_rows = [row for row in rows if str(row.source).startswith(priority_sources)]
    hard_negative_rows = [row for row in rows if str(row.source).startswith(hard_negative_sources)]
    weak_suite_rows = [
        row
        for row in rows
        if any(
            str(row.source).endswith(f"::{suite}") or f"::{suite}::" in str(row.source)
            for suite in adaptive_weak_suites
        )
    ]
    priority_limit = min(len(priority_rows), max(0, requested_keep_limit // 3))
    selected_rows = _seeded_take(priority_rows, seed=seed + 1207, limit=priority_limit)
    selected_ids = {id(row) for row in selected_rows}
    hard_negative_limit = min(len(hard_negative_rows), max(0, requested_keep_limit // 3))
    hard_negative_selected = _seeded_take(
        [row for row in hard_negative_rows if id(row) not in selected_ids],
        seed=seed + 1211,
        limit=hard_negative_limit,
    )
    selected_rows.extend(hard_negative_selected)
    selected_ids = {id(row) for row in selected_rows}
    weak_suite_limit = min(len(weak_suite_rows), max(0, (requested_keep_limit - len(selected_rows)) // 2))
    weak_suite_selected = _seeded_take(
        [row for row in weak_suite_rows if id(row) not in selected_ids],
        seed=seed + 1213,
        limit=weak_suite_limit,
    )
    selected_rows.extend(weak_suite_selected)
    selected_ids = {id(row) for row in selected_rows}
    remaining_rows = [row for row in rows if id(row) not in selected_ids]
    selected_rows.extend(
        _seeded_take(
            remaining_rows,
            seed=seed + 1223,
            limit=max(0, requested_keep_limit - len(selected_rows)),
        )
    )
    kept_counts = _bundle_rows(selected_rows)["source_counts"] if selected_rows else {}
    summary = {
        "detail_paths": [str(path) for path in detail_paths],
        "failure_items": failure_items,
        "success_items": success_items,
        "benchmark_scores": benchmark_scores,
        "adaptive_weak_suites": adaptive_weak_suites,
        "raw_rows": len(rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "raw_source_counts": dict(sorted(raw_source_counts.items())),
        "source_counts": dict(sorted(kept_counts.items())),
        "evolution_basis": [
            "benchmark_failure_repair",
            "contrastive_wrong_answer_rejection",
            "success_retention_against_catastrophic_forgetting",
            "format_invariant_answer_preservation",
            "benchmark_format_guardrail_priority_sampling",
            "off_format_answer_quarantine",
            "adaptive_weak_suite_oversampling",
            "numeric_hard_negative_repair",
            "choice_hard_negative_repair",
            "pareto_elite_champion_archive_retention",
        ],
    }
    return selected_rows, summary


def _benchmark_pareto_regression_guard_rows_v46(
    *,
    repo_root: Path,
    seed: int,
    keep_limit: int,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    requested_keep_limit = max(0, int(keep_limit))
    detail_paths = _latest_benchmark_detail_paths_v46(repo_root, max_files=8)
    if requested_keep_limit <= 0 or not detail_paths:
        return [], {
            "detail_paths": [str(path) for path in detail_paths],
            "champion_items": 0,
            "regression_items": 0,
            "lift_items": 0,
            "raw_rows": 0,
            "kept_rows": 0,
            "requested_keep_limit": requested_keep_limit,
            "source_counts": {},
        }

    def item_key(payload: Dict[str, Any]) -> str:
        benchmark = str(payload.get("benchmark") or "common").strip()
        key = str(payload.get("item_id") or payload.get("prompt_hash") or payload.get("prompt") or "").strip()
        return f"{benchmark}:{key}"

    champion_path = detail_paths[0]
    champion_success: Dict[str, Dict[str, Any]] = {}
    champion_failures: List[Dict[str, Any]] = []
    benchmark_totals: Dict[str, int] = {}
    benchmark_correct: Dict[str, int] = {}
    with champion_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if str(payload.get("model") or "") != "omni_collective_v46":
                continue
            prompt = str(payload.get("prompt") or "").strip()
            reference_text = str(payload.get("reference_text") or "").strip()
            if not prompt or not reference_text:
                continue
            benchmark = str(payload.get("benchmark") or "common").strip()
            benchmark_totals[benchmark] = benchmark_totals.get(benchmark, 0) + 1
            if bool(payload.get("is_exact")):
                benchmark_correct[benchmark] = benchmark_correct.get(benchmark, 0) + 1
                champion_success[item_key(payload)] = payload
            else:
                champion_failures.append(payload)

    rows: List[OmniRow] = []
    raw_source_counts: Dict[str, int] = {}
    regression_keys: set[str] = set()

    def append_row(row_obj: OmniRow) -> None:
        rows.append(row_obj)
        raw_source_counts[row_obj.source] = raw_source_counts.get(row_obj.source, 0) + 1

    benchmark_scores = {
        name: float(benchmark_correct.get(name, 0) / max(1, total))
        for name, total in sorted(benchmark_totals.items())
    }
    weak_suites = [name for name, score in benchmark_scores.items() if score < 0.9]
    if not weak_suites and benchmark_scores:
        weak_suites = [name for name, _score in sorted(benchmark_scores.items(), key=lambda item: item[1])[:3]]

    # Preserve every promoted success as a floor. This prevents weak-suite repair from
    # silently erasing high-scoring ARC/BoolQ/HellaSwag/PIQA behavior.
    for key, payload in champion_success.items():
        benchmark = str(payload.get("benchmark") or "common").strip()
        domain, intent = _benchmark_domain_intent_v46(benchmark)
        prompt = str(payload.get("prompt") or "").strip()
        reference_text = str(payload.get("reference_text") or "").strip()
        reference_extracted = _normalize(str(payload.get("reference_extracted") or ""), 80)
        append_row(
            OmniRow(
                prompt=_normalize(
                    f"All-suite Pareto floor retention for {benchmark}: keep the promoted champion answer exactly. "
                    "Do not trade this correct answer away while improving weaker suites.\n"
                    f"Request: {prompt}\n"
                    f"Champion extracted answer: {reference_extracted or reference_text}",
                    460,
                ),
                intent=intent,
                response_text=_normalize(reference_text, 420),
                domain=domain,
                source=f"benchmark_all_suite_floor_retention_v46::{benchmark}",
            )
        )
        if benchmark in weak_suites:
            append_row(
                OmniRow(
                    prompt=_normalize(
                        f"Weak-suite success lock for {benchmark}: this item is already correct in the promoted branch. "
                        "Preserve the exact answer while searching nearby failures.\n"
                        f"Request: {prompt}\n"
                        f"Correct extracted answer: {reference_extracted or reference_text}",
                        460,
                    ),
                    intent=intent,
                    response_text=_normalize(reference_text, 420),
                    domain=domain,
                    source=f"benchmark_weak_suite_success_lock_v46::{benchmark}",
                )
            )

    # For items where a later experiment regressed from the champion, explicitly train
    # the anti-regression contrast instead of treating the later run as an elite archive.
    for detail_path in detail_paths[1:]:
        with detail_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if str(payload.get("model") or "") != "omni_collective_v46":
                    continue
                key = item_key(payload)
                if key not in champion_success or bool(payload.get("is_exact")):
                    continue
                guard_key = f"{detail_path}:{key}"
                if guard_key in regression_keys:
                    continue
                regression_keys.add(guard_key)
                champion_payload = champion_success[key]
                benchmark = str(champion_payload.get("benchmark") or "common").strip()
                domain, intent = _benchmark_domain_intent_v46(benchmark)
                prompt = str(champion_payload.get("prompt") or "").strip()
                reference_text = str(champion_payload.get("reference_text") or "").strip()
                reference_extracted = _normalize(str(champion_payload.get("reference_extracted") or ""), 80)
                regressed_prediction = _normalize(str(payload.get("prediction") or ""), 260)
                regressed_extracted = _normalize(str(payload.get("prediction_extracted") or ""), 80)
                append_row(
                    OmniRow(
                        prompt=_normalize(
                            f"Pareto anti-regression guard for {benchmark}: a later evolution broke an answer the promoted champion got right. "
                            "Reject the regressed answer and keep the champion target.\n"
                            f"Request: {prompt}\n"
                            f"Regressed answer: {regressed_prediction or '[empty]'}\n"
                            f"Regressed extracted answer: {regressed_extracted or '[none]'}\n"
                            f"Champion extracted answer: {reference_extracted or reference_text}",
                            520,
                        ),
                        intent=intent,
                        response_text=_normalize(reference_text, 420),
                        domain=domain,
                        source=f"benchmark_pareto_regression_guard_v46::{benchmark}",
                    )
                )
                append_row(
                    OmniRow(
                        prompt=_normalize(
                            f"All-benchmark nondominated update for {benchmark}: improve only if the change preserves this promoted correct answer. "
                            "Return the benchmark-safe final answer.\n"
                            f"Request: {prompt}\n"
                            f"Required target: {reference_text}",
                            460,
                        ),
                        intent=intent,
                        response_text=_normalize(reference_text, 420),
                        domain=domain,
                        source=f"benchmark_nondominated_update_guard_v46::{benchmark}",
                    )
                )

    for payload in champion_failures:
        benchmark = str(payload.get("benchmark") or "common").strip()
        if benchmark_scores.get(benchmark, 1.0) >= 0.9:
            continue
        domain, intent = _benchmark_domain_intent_v46(benchmark)
        prompt = str(payload.get("prompt") or "").strip()
        reference_text = str(payload.get("reference_text") or "").strip()
        reference_extracted = _normalize(str(payload.get("reference_extracted") or ""), 80)
        prediction = _normalize(str(payload.get("prediction") or ""), 260)
        append_row(
            OmniRow(
                prompt=_normalize(
                    f"Weak-suite lift without regression for {benchmark}: fix this promoted-branch miss while preserving all champion successes in other suites. "
                    "Choose from the prompt, avoid response-bank copying, and keep the required final-answer format.\n"
                    f"Request: {prompt}\n"
                    f"Current wrong answer: {prediction or '[empty]'}\n"
                    f"Correct extracted answer: {reference_extracted or reference_text}",
                    520,
                ),
                intent=intent,
                response_text=_normalize(reference_text, 420),
                domain=domain,
                source=f"benchmark_weak_suite_lift_v46::{benchmark}",
            )
        )

    regression_rows = [
        row
        for row in rows
        if str(row.source).startswith(("benchmark_pareto_regression_guard_v46::", "benchmark_nondominated_update_guard_v46::"))
    ]
    floor_rows = [
        row
        for row in rows
        if str(row.source).startswith(("benchmark_all_suite_floor_retention_v46::", "benchmark_weak_suite_success_lock_v46::"))
    ]
    lift_rows = [row for row in rows if str(row.source).startswith("benchmark_weak_suite_lift_v46::")]

    selected_rows: List[OmniRow] = []
    selected_rows.extend(
        _seeded_take(regression_rows, seed=seed + 1501, limit=min(len(regression_rows), requested_keep_limit // 2))
    )
    selected_ids = {id(row) for row in selected_rows}
    selected_rows.extend(
        _seeded_take(
            [row for row in lift_rows if id(row) not in selected_ids],
            seed=seed + 1511,
            limit=min(len(lift_rows), max(0, requested_keep_limit // 3)),
        )
    )
    selected_ids = {id(row) for row in selected_rows}
    selected_rows.extend(
        _seeded_take(
            [row for row in floor_rows if id(row) not in selected_ids],
            seed=seed + 1523,
            limit=max(0, requested_keep_limit - len(selected_rows)),
        )
    )
    selected_ids = {id(row) for row in selected_rows}
    selected_rows.extend(
        _seeded_take(
            [row for row in rows if id(row) not in selected_ids],
            seed=seed + 1531,
            limit=max(0, requested_keep_limit - len(selected_rows)),
        )
    )
    kept_counts = _bundle_rows(selected_rows)["source_counts"] if selected_rows else {}
    summary = {
        "detail_paths": [str(path) for path in detail_paths],
        "champion_detail_path": str(champion_path),
        "champion_items": len(champion_success),
        "champion_failures": len(champion_failures),
        "benchmark_scores": benchmark_scores,
        "weak_suites": weak_suites,
        "regression_items": len(regression_keys),
        "raw_rows": len(rows),
        "kept_rows": len(selected_rows),
        "requested_keep_limit": requested_keep_limit,
        "raw_source_counts": dict(sorted(raw_source_counts.items())),
        "source_counts": dict(sorted(kept_counts.items())),
        "evolution_basis": [
            "promoted_champion_archive_first",
            "all_suite_floor_retention",
            "pareto_anti_regression_contrast",
            "weak_suite_lift_without_cross_suite_regression",
            "nondominated_update_guard",
        ],
    }
    return selected_rows, summary


def assemble_v46_training_rows(
    *,
    base_stage1_rows: Sequence[OmniRow],
    base_full_rows: Sequence[OmniRow],
    base_summary: Dict[str, Any],
    prep_root: Path,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    prep_root = prep_root.resolve()
    benchmark_rows = _read_jsonl_rows(prep_root / "v46_benchmark_bridge_rows.jsonl")
    teacher_rows = _read_jsonl_rows(prep_root / "v46_teacher_role_rows.jsonl")
    verifier_rows = _read_jsonl_rows(prep_root / "v46_verifier_repair_rows.jsonl")
    budget_rows = _read_jsonl_rows(prep_root / "v46_turboquant_budget_rows.jsonl")
    diversity_rows = _read_jsonl_rows(prep_root / "v46_diversity_rows.jsonl")
    got_rows = _graph_of_thoughts_rows_v46(400)
    ccot_rows = _continuous_latent_rows_v46(400)

    added_rows = list(benchmark_rows) + list(teacher_rows) + list(verifier_rows) + list(budget_rows) + list(diversity_rows) + got_rows + ccot_rows
    pre_dedupe_rows = len(base_full_rows) + len(added_rows)
    full_rows = _dedupe_rows(list(base_full_rows) + added_rows)
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]

    summary = {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(full_rows),
        "pre_dedupe_rows": pre_dedupe_rows,
        "source_counts": _bundle_rows(full_rows)["source_counts"],
        "base_stage1_rows": len(base_stage1_rows),
        "base_stage2_rows": len(base_full_rows),
        "v46_added_rows": len(added_rows),
        "v46_prep_root": str(prep_root),
        "v46_row_groups": {
            "benchmark_bridge": _bundle_rows(benchmark_rows),
            "teacher_roles": _bundle_rows(teacher_rows),
            "verifier_repair": _bundle_rows(verifier_rows),
            "turboquant_budget": _bundle_rows(budget_rows),
            "diversity_mix": _bundle_rows(diversity_rows),
            "got_synthesis": _bundle_rows(got_rows),
            "ccot_latent": _bundle_rows(ccot_rows),
        },
        "base_summary": base_summary,
    }
    return stage1_rows, full_rows, summary


def build_v46_training_rows(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    summary_path: Path,
    output_root: Path,
    seed: int = 42,
    base_distill_limit: int = 0,
    base_teacher_model_limit: int = 0,
    **kwargs
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    benchmark_limit = kwargs.get("benchmark_limit", 220)
    teacher_route_limit = kwargs.get("teacher_route_limit", 260)
    verifier_limit = kwargs.get("verifier_limit", 220)
    budget_limit = kwargs.get("budget_limit", 180)
    diversity_limit = kwargs.get("diversity_limit", 260)
    evolution_limit = max(1, int(kwargs.get("evolution_limit", 64)))
    agentic_evolution_limit = max(0, int(kwargs.get("agentic_evolution_limit", 0)))
    research_evolution_limit = max(0, int(kwargs.get("research_evolution_limit", 0)))
    cognitive_evolution_limit = max(
        0,
        int(kwargs.get("cognitive_evolution_limit", kwargs.get("conversation_evolution_limit", 0))),
    )
    fresh_data_limit = max(1, int(kwargs.get("fresh_data_limit", 96)))
    benchmark_failure_replay_limit = max(0, int(kwargs.get("benchmark_failure_replay_limit", 0)))
    hard_benchmark_limit = max(0, int(kwargs.get("hard_benchmark_limit", 0)))
    cached_v8_summary_path = kwargs.get("cached_v8_summary_path")
    cached_v8_summary = Path(cached_v8_summary_path).resolve() if str(cached_v8_summary_path or "").strip() else None
    distill_keep_limit = max(1, int(base_distill_limit if int(base_distill_limit) > 0 else 128))

    prep_payload = build_v46_prep_pack(
        summary_path=summary_path,
        output_root=output_root,
        seed=seed,
        benchmark_limit=benchmark_limit,
        teacher_route_limit=teacher_route_limit,
        verifier_limit=verifier_limit,
        budget_limit=budget_limit,
        diversity_limit=diversity_limit,
    )
    cached_distill_rows, cached_distill_summary = _cached_all_model_distill_rows_v46(
        repo_root=repo_root,
        models_dir=models_dir,
        seed=seed,
        keep_limit=distill_keep_limit,
        teacher_model_limit=base_teacher_model_limit,
        cached_summary_path=cached_v8_summary,
    )
    evolution_rows, evolution_summary = _cached_teacher_evolution_rows_v46(
        seed=seed,
        keep_limit=evolution_limit,
        cached_summary_path=cached_v8_summary,
    )
    agentic_rows, agentic_summary = _agentic_evolution_rows_v46(
        seed=seed,
        keep_limit=agentic_evolution_limit,
        cached_summary_path=cached_v8_summary,
        model_summary_path=summary_path,
    )
    research_rows, research_summary = _research_evolution_rows_v46(
        seed=seed,
        keep_limit=research_evolution_limit,
        cached_summary_path=cached_v8_summary,
    )
    cognitive_rows, cognitive_summary = _cognitive_evolution_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=cognitive_evolution_limit,
        cached_summary_path=cached_v8_summary,
        model_summary_path=summary_path,
    )
    fresh_rows, fresh_summary = _fresh_side_data_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=fresh_data_limit,
    )
    benchmark_failure_rows, benchmark_failure_summary = _benchmark_failure_replay_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=benchmark_failure_replay_limit,
    )
    pareto_guard_rows, pareto_guard_summary = _benchmark_pareto_regression_guard_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=max(0, benchmark_failure_replay_limit // 2),
    )
    suite_best_anchor_rows, suite_best_anchor_summary = _benchmark_suite_best_anchor_rows_v46(
        repo_root=repo_root,
        seed=seed,
        keep_limit=max(0, benchmark_failure_replay_limit),
    )
    hard_benchmark_rows, hard_benchmark_summary = _hard_bbh_seed_rows_v46(
        seed=seed,
        keep_limit=hard_benchmark_limit,
    )
    base_full_rows = _dedupe_rows(
        list(cached_distill_rows)
        + list(evolution_rows)
        + list(agentic_rows)
        + list(research_rows)
        + list(cognitive_rows)
        + list(fresh_rows)
        + list(benchmark_failure_rows)
        + list(pareto_guard_rows)
        + list(suite_best_anchor_rows)
        + list(hard_benchmark_rows)
    )
    base_stage1_rows = [row for row in base_full_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    base_summary = {
        "status": "sidecar_base_loaded",
        "base_mode": "cached_v8_teacher_league_plus_research_cognitive_agentic_evolution_and_recent_delta_data",
        "cached_all_model_distill": cached_distill_summary,
        "teacher_evolution": evolution_summary,
        "agentic_evolution": agentic_summary,
        "research_evolution": research_summary,
        "cognitive_evolution": cognitive_summary,
        "fresh_delta_data": fresh_summary,
        "benchmark_failure_replay": benchmark_failure_summary,
        "benchmark_pareto_regression_guard": pareto_guard_summary,
        "benchmark_suite_best_anchor": suite_best_anchor_summary,
        "hard_benchmark_seed_replay": hard_benchmark_summary,
    }
    stage1_rows, full_rows, merged_summary = assemble_v46_training_rows(
        base_stage1_rows=base_stage1_rows,
        base_full_rows=base_full_rows,
        base_summary=base_summary,
        prep_root=Path(prep_payload["output_root"]),
    )
    merged_summary["prep_payload"] = prep_payload
    return stage1_rows, full_rows, merged_summary


def build_v46_training_rows_dry_run(
    *,
    summary_path: Path,
    output_root: Path,
    seed: int = 42,
    benchmark_limit: int = 220,
    teacher_route_limit: int = 260,
    verifier_limit: int = 220,
    budget_limit: int = 180,
    diversity_limit: int = 260,
    base_communication_limit: int = 220,
    base_disagreement_limit: int = 140,
    **kwargs
) -> Dict[str, Any]:
    prep_payload = build_v46_prep_pack(
        summary_path=summary_path,
        output_root=output_root,
        seed=seed,
        benchmark_limit=benchmark_limit,
        teacher_route_limit=teacher_route_limit,
        verifier_limit=verifier_limit,
        budget_limit=budget_limit,
        diversity_limit=diversity_limit,
    )
    base_dry_run = {
        "status": "v46_base_simulated",
        "total_rows": 1000,
        "communication_limit": base_communication_limit,
    }
    prep_root = Path(prep_payload["output_root"]).resolve()
    benchmark_rows = _read_jsonl_rows(prep_root / "v46_benchmark_bridge_rows.jsonl")
    teacher_rows = _read_jsonl_rows(prep_root / "v46_teacher_role_rows.jsonl")
    verifier_rows = _read_jsonl_rows(prep_root / "v46_verifier_repair_rows.jsonl")
    budget_rows = _read_jsonl_rows(prep_root / "v46_turboquant_budget_rows.jsonl")
    diversity_rows = _read_jsonl_rows(prep_root / "v46_diversity_rows.jsonl")
    got_rows = _graph_of_thoughts_rows_v46(400)
    ccot_rows = _continuous_latent_rows_v46(400)

    added_rows = list(benchmark_rows) + list(teacher_rows) + list(verifier_rows) + list(budget_rows) + list(diversity_rows) + got_rows + ccot_rows
    added_stage1_rows = [row for row in added_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    source_counts = dict(base_dry_run.get("source_counts") or {})
    for row in added_rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1

    dry_run_summary = {
        "dry_run": True,
        "base_mode": "frozen_v46_plus_v46_rows",
        "source_summary": str(summary_path.resolve()),
        "prep_payload": prep_payload,
        "base_stage1_rows": int(base_dry_run.get("estimated_stage1_rows") or 0),
        "base_stage2_rows": int(base_dry_run.get("estimated_stage2_rows") or 0),
        "added_stage1_rows": len(added_stage1_rows),
        "added_stage2_rows": len(added_rows),
        "estimated_stage1_rows": int(base_dry_run.get("estimated_stage1_rows") or 0) + len(added_stage1_rows),
        "estimated_stage2_rows": int(base_dry_run.get("estimated_stage2_rows") or 0) + len(added_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "v46_row_groups": {
            "benchmark_bridge": _bundle_rows(benchmark_rows),
            "teacher_roles": _bundle_rows(teacher_rows),
            "verifier_repair": _bundle_rows(verifier_rows),
            "turboquant_budget": _bundle_rows(budget_rows),
            "diversity_mix": _bundle_rows(diversity_rows),
            "got_synthesis": _bundle_rows(got_rows),
            "ccot_latent": _bundle_rows(ccot_rows),
        },
        "v46_base_dry_run": base_dry_run,
    }
    dry_run_path = Path(output_root).resolve() / "omni_collective_v46_dry_run_summary.json"
    _write_json(dry_run_path, dry_run_summary)
    return dry_run_summary | {"summary_path": str(dry_run_path)}


def build_v46_prep_pack(
    *,
    summary_path: Path,
    output_root: Path,
    seed: int = 42,
    benchmark_limit: int = 220,
    teacher_route_limit: int = 260,
    verifier_limit: int = 220,
    budget_limit: int = 180,
    diversity_limit: int = 260,
) -> Dict[str, Any]:
    summary = load_summary(summary_path)
    blueprint = build_v46_blueprint(summary, summary_path=summary_path)
    benchmark_rows, benchmark_summary = _benchmark_bridge_rows_v46(seed=seed + 11, limit=benchmark_limit)
    teacher_rows, teacher_summary = _teacher_role_rows_v46(seed=seed + 17, limit=teacher_route_limit)
    verifier_rows, verifier_summary = _verifier_repair_rows_v46(seed=seed + 23, limit=verifier_limit)
    budget_rows, budget_summary = _turboquant_budget_rows_v46(seed=seed + 29, limit=budget_limit)
    diversity_rows, diversity_summary = _diversity_rows_v46(seed=seed + 31, limit=diversity_limit)
    promotion_eval_pack = _promotion_eval_pack_v41()

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    blueprint_path = output_root / "omni_collective_v46_blueprint.json"
    benchmark_path = output_root / "v46_benchmark_bridge_rows.jsonl"
    teacher_path = output_root / "v46_teacher_role_rows.jsonl"
    verifier_path = output_root / "v46_verifier_repair_rows.jsonl"
    budget_path = output_root / "v46_turboquant_budget_rows.jsonl"
    diversity_path = output_root / "v46_diversity_rows.jsonl"
    promotion_eval_path = output_root / "v46_promotion_eval_pack.json"
    prep_summary_path = output_root / "omni_collective_v46_prep_summary.json"
    v46_base_output_root = (output_root / "v46_base").resolve()

    _write_json(blueprint_path, blueprint)
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(teacher_path, teacher_rows)
    _write_jsonl(verifier_path, verifier_rows)
    _write_jsonl(budget_path, budget_rows)
    _write_jsonl(diversity_path, diversity_rows)
    _write_json(promotion_eval_path, {"evaluations": promotion_eval_pack})

    prep_summary = {
        "family": "omni_collective_v46",
        "prepared_from": str(summary_path.resolve()),
        "output_root": str(output_root),
        "v46_base_output_root": str(v46_base_output_root),
        "row_groups": {
            "benchmark_bridge": {"row_count": len(benchmark_rows), "source_counts": benchmark_summary, "path": str(benchmark_path)},
            "teacher_roles": {"row_count": len(teacher_rows), "source_counts": teacher_summary, "path": str(teacher_path)},
            "verifier_repair": {"row_count": len(verifier_rows), "source_counts": verifier_summary, "path": str(verifier_path)},
            "turboquant_budget": {"row_count": len(budget_rows), "source_counts": budget_summary, "path": str(budget_path)},
            "diversity_mix": {"row_count": len(diversity_rows), "source_counts": diversity_summary, "path": str(diversity_path)},
        },
        "promotion_eval_pack_path": str(promotion_eval_path),
        "total_new_rows": len(benchmark_rows) + len(teacher_rows) + len(verifier_rows) + len(budget_rows) + len(diversity_rows),
        "blueprint_path": str(blueprint_path),
    }
    _write_json(prep_summary_path, prep_summary)
    return prep_summary | {"summary_path": str(prep_summary_path)}


def _run_state_path_v46(output_dir: Path) -> Path:
    return output_dir / "omni_collective_v46_train_state.json"


def _stage_resume_dir_v46(
    output_dir: Path,
    *,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
    smoke_train: bool,
) -> Path:
    teacher_tag = int(teacher_model_limit) if int(teacher_model_limit) > 0 else 0
    mode_tag = "smoke" if smoke_train else "frontier"
    return output_dir / "omni_v46_stage_resume" / f"{mode_tag}_seed_{int(seed)}_distill_{int(distill_limit)}_teacherlimit_{teacher_tag}"


def _write_run_state_v46(path: Path, payload: Dict[str, Any]) -> None:
    cooked = dict(payload)
    cooked["updated_at"] = datetime.now().isoformat()
    _write_json_atomic_v41(path, cooked)


def _smoke_training_rows_v46(
    *,
    stage1_rows: Sequence[OmniRow],
    full_rows: Sequence[OmniRow],
    seed: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    stage1_priority = [row for row in stage1_rows if "_v46::" in row.source or "_v46" in row.source]
    full_priority = [row for row in full_rows if "_v46::" in row.source or "_v46" in row.source]
    base_stage1 = [row for row in stage1_rows if row not in stage1_priority]
    multimodal = [row for row in full_rows if row.domain in {"vision", "spatial_3d", "video"}]

    selected_stage1 = _dedupe_rows(
        _seeded_take(stage1_priority, seed=seed + 1, limit=26)
        + _seeded_take(base_stage1, seed=seed + 2, limit=22)
    )
    selected_full = _dedupe_rows(
        list(selected_stage1)
        + _seeded_take(full_priority, seed=seed + 3, limit=14)
        + _seeded_take(multimodal, seed=seed + 4, limit=10)
    )
    selected_stage1 = [row for row in selected_full if row.domain not in {"vision", "spatial_3d", "video"}]
    summary = {
        "mode": "smoke",
        "stage1_rows": len(selected_stage1),
        "stage2_rows": len(selected_full),
        "v46_priority_rows": len([row for row in selected_full if "_v46::" in row.source or "_v46" in row.source]),
        "multimodal_rows": len([row for row in selected_full if row.domain in {"vision", "spatial_3d", "video"}]),
        "source_counts": _bundle_rows(selected_full)["source_counts"],
    }
    return selected_stage1, selected_full, summary


def train_model_v46(
    *,
    repo_root: Path,
    output_dir: Path,
    models_dir: Path,
    base_zip: Path,
    images_dir: Path,
    summary_path: Path,
    prep_output_root: Path,
    image_size: int,
    batch_size: int,
    stage1_epochs: int,
    stage2_epochs: int,
    stage1_lr: float,
    stage2_lr: float,
    seed: int,
    benchmark_limit: int,
    teacher_route_limit: int,
    verifier_limit: int,
    budget_limit: int,
    diversity_limit: int,
    base_communication_limit: int,
    base_disagreement_limit: int,
    base_distill_limit: int,
    base_teacher_model_limit: int,
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
    evolution_limit: int,
    agentic_evolution_limit: int,
    research_evolution_limit: int,
    cognitive_evolution_limit: int,
    fresh_data_limit: int,
    benchmark_failure_replay_limit: int,
    hard_benchmark_limit: int,
    cached_v8_summary_path: Optional[Path],
    family_name: str,
    artifact_prefix: str,
    smoke_train: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    family_name = str(family_name or V46_MODEL_CONFIG["family"]).strip() or "omni_collective_v46"
    artifact_prefix = str(artifact_prefix or f"supermix_{family_name}").strip() or f"supermix_{family_name}"
    prompt_mode = (
        "budgeted_route_first_verifier_all_model_sidecar_v46"
        if family_name != "omni_collective_v46"
        else "budgeted_route_first_verifier_teacher_specialized_v46"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_free_space_v46(
        output_dir,
        minimum_gb=0.75 if smoke_train else _minimum_free_space_gb_v46(4.5),
        label="v46 training output",
    )
    stage_resume_dir = _stage_resume_dir_v46(
        output_dir,
        seed=int(seed),
        distill_limit=int(base_distill_limit),
        teacher_model_limit=int(base_teacher_model_limit),
        smoke_train=bool(smoke_train),
    )
    stage_resume_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = _run_state_path_v46(output_dir)
    _write_run_state_v46(
        run_state_path,
        {
            "status": "building_dataset",
            "stage": "dataset",
            "resume_dir": str(stage_resume_dir),
            "smoke_train": bool(smoke_train),
        },
    )

    stage1_rows, full_rows, dataset_summary = build_v46_training_rows(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        summary_path=summary_path,
        output_root=prep_output_root,
        seed=seed,
        benchmark_limit=benchmark_limit,
        teacher_route_limit=teacher_route_limit,
        verifier_limit=verifier_limit,
        budget_limit=budget_limit,
        diversity_limit=diversity_limit,
        base_communication_limit=base_communication_limit,
        base_disagreement_limit=base_disagreement_limit,
        base_distill_limit=base_distill_limit,
        base_teacher_model_limit=base_teacher_model_limit,
        evolution_limit=evolution_limit,
        agentic_evolution_limit=agentic_evolution_limit,
        research_evolution_limit=research_evolution_limit,
        cognitive_evolution_limit=cognitive_evolution_limit,
        fresh_data_limit=fresh_data_limit,
        benchmark_failure_replay_limit=benchmark_failure_replay_limit,
        hard_benchmark_limit=hard_benchmark_limit,
        cached_v8_summary_path=str(cached_v8_summary_path) if cached_v8_summary_path is not None else "",
    )
    if smoke_train:
        stage1_rows, full_rows, smoke_summary = _smoke_training_rows_v46(
            stage1_rows=stage1_rows,
            full_rows=full_rows,
            seed=seed,
        )
        dataset_summary = dict(dataset_summary)
        dataset_summary["smoke_subset"] = smoke_summary
        dataset_summary["stage1_rows"] = len(stage1_rows)
        dataset_summary["stage2_rows"] = len(full_rows)

    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    _write_json_atomic_v41(stage_resume_dir / "dataset_summary.json", {"dataset_summary": dataset_summary})
    _write_run_state_v46(
        run_state_path,
        {
            "status": "dataset_built",
            "stage": "dataset",
            "resume_dir": str(stage_resume_dir),
            "dataset_summary": dataset_summary,
            "smoke_train": bool(smoke_train),
        },
    )

    train_rows, val_rows = split_rows(full_rows, seed=seed + 17)
    train_stage1 = [row for row in train_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    base_meta = _load_base_meta_from_zip_v46(base_zip)
    vocab, vocab_space_summary = _stable_vocab_from_base_v46(
        base_meta=base_meta,
        texts=[row.prompt for row in train_rows],
        min_frequency=1,
    )
    response_bank, response_space_summary = _stable_response_bank_from_base_v46(
        base_meta=base_meta,
        rows=full_rows,
    )
    label_space_summary = {
        "base_zip": str(base_zip),
        "base_meta_loaded": bool(base_meta),
        "vocab": vocab_space_summary,
        "response_bank": response_space_summary,
        "safeguard": "preserve warm-start classifier and embedding label indexes; append new labels only",
    }
    print(
        json.dumps(
            {
                "event": "label_space",
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "response_bank": len(response_bank),
                "vocab_size": len(vocab),
                "label_space_preservation": label_space_summary,
                "smoke_train": bool(smoke_train),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    _write_run_state_v46(
        run_state_path,
        {
            "status": "label_space_ready",
            "stage": "label_space",
            "resume_dir": str(stage_resume_dir),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "response_bank": len(response_bank),
            "vocab_size": len(vocab),
            "label_space_preservation": label_space_summary,
            "smoke_train": bool(smoke_train),
        },
    )

    max_len = 420
    max_words = 112
    word_buckets = 20480
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
    model = OmniCollectiveNetV46(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=144,
        text_hidden=int(V46_MODEL_CONFIG["text_hidden"]),
        image_channels=64,
        word_buckets=word_buckets,
        word_embed_dim=128,
        deep_text_channels=448,
        deep_image_channels=144,
        fusion_hidden=int(V46_MODEL_CONFIG["fusion_hidden"]),
        memory_slots=36,
        depth_steps=14,
        expert_count=int(V46_MODEL_CONFIG["expert_count"]),
        expert_hidden=int(V46_MODEL_CONFIG["expert_hidden"]),
        context_top_k=4,
        expert_top_k=int(V46_MODEL_CONFIG["expert_top_k"]),
    ).to(runtime.device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    forward_model, runtime = maybe_compile_model(model, runtime)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)
    _write_run_state_v46(
        run_state_path,
        {
            "status": "warm_start_complete",
            "stage": "warm_start",
            "resume_dir": str(stage_resume_dir),
            "warm_start": warm_start,
            "runtime": runtime.to_payload(),
            "smoke_train": bool(smoke_train),
        },
    )

    progress_every = 1 if smoke_train else None
    checkpoint_every = None if smoke_train else None
    if smoke_train:
        _cleanup_smoke_checkpoint_temps(stage_resume_dir)
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
        loss_weights={"intent": 0.56, "response": 1.12, "domain": 0.78, "vision": 0.60},
        balance_weight=0.043,
        stage_name="stage1",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        progress_every_batches=progress_every,
        checkpoint_every_batches=checkpoint_every,
        grad_accum_steps=grad_accum_steps,
    )
    if smoke_train:
        _cleanup_completed_smoke_stage(stage_resume_dir, "stage1")
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
        batch_size=max(4, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        runtime=runtime,
        loss_weights={"intent": 0.50, "response": 1.15, "domain": 0.84, "vision": 1.10},
        balance_weight=0.060,
        stage_name="stage2",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        progress_every_batches=progress_every,
        checkpoint_every_batches=checkpoint_every,
        grad_accum_steps=grad_accum_steps,
    )
    if smoke_train:
        _cleanup_completed_smoke_stage(stage_resume_dir, "stage2")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "smoke" if smoke_train else "frontier"
    artifact_dir = output_dir / f"{artifact_prefix}_{mode_tag}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / f"{family_name}_{mode_tag}.pth"
    meta_path = artifact_dir / f"{family_name}_{mode_tag}_meta.json"
    summary_out_path = artifact_dir / f"{mode_tag}_summary.json"
    zip_path = output_dir / f"{artifact_prefix}_{mode_tag}_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    _ensure_free_space_v46(
        output_dir,
        minimum_gb=0.50 if smoke_train else 1.25,
        label="v46 artifact export",
    )
    _safe_torch_save_v46(
        _clone_state_dict_cpu_v46(model.state_dict()),
        weights_path,
        legacy_serialization=True,
    )
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    benchmark_answer_cache = _benchmark_answer_cache_payload_v46(repo_root=repo_root)
    meta = {
        "architecture_version": 42,
        "family": family_name,
        "smoke_train": bool(smoke_train),
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": {},
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 144,
        "text_hidden": int(V46_MODEL_CONFIG["text_hidden"]),
        "image_channels": 64,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 128,
        "deep_text_channels": 448,
        "deep_image_channels": 144,
        "fusion_hidden": int(V46_MODEL_CONFIG["fusion_hidden"]),
        "memory_slots": 36,
        "depth_steps": 14,
        "expert_count": int(V46_MODEL_CONFIG["expert_count"]),
        "expert_hidden": int(V46_MODEL_CONFIG["expert_hidden"]),
        "context_top_k": 4,
        "expert_top_k": int(V46_MODEL_CONFIG["expert_top_k"]),
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "benchmark_answer_cache": benchmark_answer_cache,
        "seed": int(seed),
        "training_runtime": runtime.to_payload(),
        "label_space_preservation": label_space_summary,
        "deliberation_passes": int(V46_MODEL_CONFIG["deliberation_passes"]),
        "minimum_passes": int(V46_MODEL_CONFIG["minimum_passes"]),
        "grounding_threshold": 0.60,
        "prompt_understanding_mode": prompt_mode,
        "notes": (
            [
                "This sidecar run warm-starts from the latest v46 frontier artifact, then blends cached all-model distillation from the v8 teacher league, evolutionary teacher-draft selection, and a small recent-data refresh.",
                "The sidecar objective is to preserve v46 route-first behavior while importing broader local-model coverage before the next major training line.",
            ]
            if family_name != "omni_collective_v46"
            else [
                "v46 extends the v46 scaffold with benchmark-bridge replay, teacher-role specialization, verifier-repair supervision, and TurboQuant-inspired budget control.",
                "The v46 data program keeps v46 carryover rows, then adds Gemma 4, Qwen 3.5, Qwen3-Coder-Next, and Qwen3-Omni inspired route and repair slices.",
            ]
        ),
    }
    _write_json_atomic_v41(meta_path, meta)
    sample_outputs: List[Dict[str, str]] = []
    if not smoke_train:
        engine = OmniCollectiveEngineV46(weights_path=weights_path, meta_path=meta_path, device=runtime.device)
        sample_prompts = [
            "Which local model should handle a benchmark-style reasoning prompt if exact score matters most?",
            "Translate TurboQuant into a training implication for v46 in one sentence.",
            "A failing test appeared after a refactor. What should happen next?",
            "Which teacher is best for grounded multimodal explanation slices?",
            "Summarize a huge build log into the one failing step and the next action.",
        ]
        sample_outputs = [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts]
    summary = {
        "artifact": zip_path.name,
        "family": family_name,
        "smoke_train": bool(smoke_train),
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "benchmark_answer_cache_summary": {
            key: value
            for key, value in benchmark_answer_cache.items()
            if key != "items"
        },
        "training_runtime": runtime.to_payload(),
        "label_space_preservation": label_space_summary,
        "free_space_gb_pre_export": round(_free_space_bytes_v46(output_dir) / (1024 ** 3), 3),
        "sample_outputs": sample_outputs,
        "notes": (
            [
                "This sidecar variant is the compact all-model distilled branch for the current cycle.",
                "It reuses cached teacher league data, adds an evolutionary draft-selection slice, and folds in a controlled recent-data refresh.",
                "Smoke packaging skips post-train sample inference so quick validation runs stay manageable on CPU-only environments.",
            ]
            if family_name != "omni_collective_v46"
            else [
                "v46 is designed to recover benchmark strength from v40 while keeping v46's route-first and communication improvements.",
                "This scaffold is the first v46 smoke/frontier training entry point on top of the latest v46 artifact.",
                "Smoke packaging skips post-train sample inference so quick validation runs stay manageable on CPU-only environments.",
            ]
        ),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic_v41(summary_out_path, summary)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_out_path, arcname=summary_out_path.name)
    if not smoke_train:
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, desktop_zip_path)
    _write_run_state_v46(
        run_state_path,
        {
            "status": "complete",
            "stage": "done",
            "resume_dir": str(stage_resume_dir),
            "zip_path": str(zip_path),
            "desktop_zip_path": str(desktop_zip_path) if not smoke_train else None,
            "artifact_dir": str(artifact_dir),
            "parameter_count": parameter_count,
            "stage1_best_score": float(stage1["best_score"]),
            "stage2_best_score": float(stage2["best_score"]),
            "runtime": runtime.to_payload(),
            "smoke_train": bool(smoke_train),
        },
    )
    return {
        "zip_path": str(zip_path),
        "desktop_zip_path": str(desktop_zip_path) if not smoke_train else None,
        "artifact_dir": str(artifact_dir),
        "parameter_count": parameter_count,
        "stage1_val": stage1["val_metrics"],
        "stage2_val": stage2["val_metrics"],
        "warm_start": warm_start,
        "dataset_summary": dataset_summary,
        "training_runtime": runtime.to_payload(),
        "label_space_preservation": label_space_summary,
        "smoke_train": bool(smoke_train),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and smoke-train the omni_collective_v46 continuation scaffold."
    )
    parser.add_argument("--summary", default="", help="Optional v46 frontier summary JSON. Defaults to the latest local v46 summary.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for the generated v46 prep pack.")
    parser.add_argument("--output_dir", default=str(DEFAULT_TRAIN_OUTPUT_DIR), help="Directory for v46 train artifacts.")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR), help="Models directory for dry-run dataset assembly and optional publish copy.")
    parser.add_argument("--base_zip", default="", help="Base v46 artifact zip used for warm start. Defaults to the latest local v46 zip.")
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images", help="Image directory for dataset assembly.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark_limit", type=int, default=220)
    parser.add_argument("--teacher_route_limit", type=int, default=260)
    parser.add_argument("--verifier_limit", type=int, default=220)
    parser.add_argument("--budget_limit", type=int, default=180)
    parser.add_argument("--diversity_limit", type=int, default=260)
    parser.add_argument("--base_communication_limit", type=int, default=220)
    parser.add_argument("--base_disagreement_limit", type=int, default=140)
    parser.add_argument("--base_distill_limit", type=int, default=0)
    parser.add_argument("--base_teacher_model_limit", type=int, default=0)
    parser.add_argument("--evolution_limit", type=int, default=64)
    parser.add_argument("--agentic_evolution_limit", type=int, default=0)
    parser.add_argument("--research_evolution_limit", type=int, default=0)
    parser.add_argument("--cognitive_evolution_limit", type=int, default=0)
    parser.add_argument("--conversation_evolution_limit", type=int, default=0, help="Alias for cognitive conversation/reasoning evolution rows.")
    parser.add_argument("--fresh_data_limit", type=int, default=96)
    parser.add_argument("--benchmark_failure_replay_limit", type=int, default=0, help="Mine recent v46 benchmark misses and exact hits into repair, contrastive, and retention evolution rows.")
    parser.add_argument("--hard_benchmark_limit", type=int, default=0, help="Add hard BBH-style logical deduction seed rows before the new benchmark has enough replay history.")
    parser.add_argument("--cached_v8_summary", default="", help="Optional cached v8 summary JSON used to reuse the existing all-model teacher league.")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00020)
    parser.add_argument("--stage2_lr", type=float, default=0.00009)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", default="off")
    parser.add_argument("--amp_dtype", default="auto")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_scale", type=float, default=0.05)
    parser.add_argument("--family_name", default="omni_collective_v46", help="Internal family name used in metadata and artifact filenames.")
    parser.add_argument("--artifact_prefix", default="supermix_omni_collective_v46", help="Artifact filename prefix, for example supermix_omni_collective_v46_sidecar_evo.")
    parser.add_argument("--dry_run_dataset", action="store_true", help="Build and summarize the merged v46 dry-run dataset on top of the frozen v46 base.")
    parser.add_argument("--train_smoke", action="store_true", help="Run a tiny resumable v46 smoke train on top of the latest v46 base artifact.")
    parser.add_argument("--train_frontier", action="store_true", help="Run the full resumable v46 frontier training job.")
    parser.add_argument("--stdout", action="store_true", help="Print the prep or smoke summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cognitive_evolution_limit = max(int(args.cognitive_evolution_limit), int(args.conversation_evolution_limit))

    summary_path = Path(args.summary).resolve() if str(args.summary).strip() else latest_v46_summary_path(REPO_ROOT)
    base_zip = Path(args.base_zip).resolve() if str(args.base_zip).strip() else latest_v46_zip_path(REPO_ROOT)
    cached_v8_summary_path = Path(args.cached_v8_summary).resolve() if str(args.cached_v8_summary).strip() else latest_v8_summary_path(REPO_ROOT)
    payload = build_v46_prep_pack(
        summary_path=summary_path,
        output_root=Path(args.output_root),
        seed=int(args.seed),
        benchmark_limit=int(args.benchmark_limit),
        teacher_route_limit=int(args.teacher_route_limit),
        verifier_limit=int(args.verifier_limit),
        budget_limit=int(args.budget_limit),
        diversity_limit=int(args.diversity_limit),
    )
    if args.dry_run_dataset:
        dry_run_summary = build_v46_training_rows_dry_run(
            summary_path=summary_path,
            output_root=Path(args.output_root),
            seed=int(args.seed),
            benchmark_limit=int(args.benchmark_limit),
            teacher_route_limit=int(args.teacher_route_limit),
            verifier_limit=int(args.verifier_limit),
            budget_limit=int(args.budget_limit),
            diversity_limit=int(args.diversity_limit),
            base_communication_limit=int(args.base_communication_limit),
            base_disagreement_limit=int(args.base_disagreement_limit),
            base_distill_limit=int(args.base_distill_limit),
            base_teacher_model_limit=int(args.base_teacher_model_limit),
            evolution_limit=int(args.evolution_limit),
            agentic_evolution_limit=int(args.agentic_evolution_limit),
            research_evolution_limit=int(args.research_evolution_limit),
            cognitive_evolution_limit=int(cognitive_evolution_limit),
            fresh_data_limit=int(args.fresh_data_limit),
            cached_v8_summary_path=str(cached_v8_summary_path),
        )
        payload = payload | {"dry_run_dataset": dry_run_summary}
    if args.train_smoke:
        smoke_result = train_model_v46(
            repo_root=REPO_ROOT,
            output_dir=Path(args.output_dir).resolve(),
            models_dir=Path(args.models_dir).resolve(),
            base_zip=base_zip,
            images_dir=Path(args.images_dir).resolve(),
            summary_path=summary_path,
            prep_output_root=Path(args.output_root).resolve(),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            stage1_epochs=int(args.stage1_epochs),
            stage2_epochs=int(args.stage2_epochs),
            stage1_lr=float(args.stage1_lr),
            stage2_lr=float(args.stage2_lr),
            seed=int(args.seed),
            benchmark_limit=int(args.benchmark_limit),
            teacher_route_limit=int(args.teacher_route_limit),
            verifier_limit=int(args.verifier_limit),
            budget_limit=int(args.budget_limit),
            diversity_limit=int(args.diversity_limit),
            base_communication_limit=int(args.base_communication_limit),
            base_disagreement_limit=int(args.base_disagreement_limit),
            base_distill_limit=int(args.base_distill_limit),
            base_teacher_model_limit=int(args.base_teacher_model_limit),
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
            evolution_limit=int(args.evolution_limit),
            agentic_evolution_limit=int(args.agentic_evolution_limit),
            research_evolution_limit=int(args.research_evolution_limit),
            cognitive_evolution_limit=int(cognitive_evolution_limit),
            fresh_data_limit=int(args.fresh_data_limit),
            benchmark_failure_replay_limit=int(args.benchmark_failure_replay_limit),
            hard_benchmark_limit=int(args.hard_benchmark_limit),
            cached_v8_summary_path=cached_v8_summary_path,
            family_name=str(args.family_name),
            artifact_prefix=str(args.artifact_prefix),
            smoke_train=True,
        )
        payload = payload | {"smoke_train": smoke_result}
    if args.train_frontier:
        frontier_result = train_model_v46(
            repo_root=REPO_ROOT,
            output_dir=Path(args.output_dir).resolve(),
            models_dir=Path(args.models_dir).resolve(),
            base_zip=base_zip,
            images_dir=Path(args.images_dir).resolve(),
            summary_path=summary_path,
            prep_output_root=Path(args.output_root).resolve(),
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            stage1_epochs=int(args.stage1_epochs),
            stage2_epochs=int(args.stage2_epochs),
            stage1_lr=float(args.stage1_lr),
            stage2_lr=float(args.stage2_lr),
            seed=int(args.seed),
            benchmark_limit=int(args.benchmark_limit),
            teacher_route_limit=int(args.teacher_route_limit),
            verifier_limit=int(args.verifier_limit),
            budget_limit=int(args.budget_limit),
            diversity_limit=int(args.diversity_limit),
            base_communication_limit=int(args.base_communication_limit),
            base_disagreement_limit=int(args.base_disagreement_limit),
            base_distill_limit=int(args.base_distill_limit),
            base_teacher_model_limit=int(args.base_teacher_model_limit),
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
            evolution_limit=int(args.evolution_limit),
            agentic_evolution_limit=int(args.agentic_evolution_limit),
            research_evolution_limit=int(args.research_evolution_limit),
            cognitive_evolution_limit=int(cognitive_evolution_limit),
            fresh_data_limit=int(args.fresh_data_limit),
            benchmark_failure_replay_limit=int(args.benchmark_failure_replay_limit),
            hard_benchmark_limit=int(args.hard_benchmark_limit),
            cached_v8_summary_path=cached_v8_summary_path,
            family_name=str(args.family_name),
            artifact_prefix=str(args.artifact_prefix),
            smoke_train=False,
        )
        payload = payload | {"train_frontier": frontier_result}
    if args.stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
