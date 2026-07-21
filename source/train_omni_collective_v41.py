from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
if str(SOURCE_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(SOURCE_DIR))

try:
    from image_recognition_model import SCIENCE_IMAGE_CLASSES
    from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from omni_collective_v41_model import OmniCollectiveEngineV41, OmniCollectiveNetV41
    from omni_training_runtime import maybe_compile_model, resolve_training_runtime
    from prepare_omni_collective_v41 import build_v41_blueprint, latest_v8_summary_path, load_summary
    from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, split_rows
    from train_omni_collective_v4 import _load_expanded_state_from_zip
    from train_omni_collective_v8 import (
        _train_stage_resumable_v8,
        build_training_rows as build_training_rows_v8,
    )
except ImportError:  # pragma: no cover
    from .image_recognition_model import SCIENCE_IMAGE_CLASSES
    from .omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
    from .omni_collective_v41_model import OmniCollectiveEngineV41, OmniCollectiveNetV41
    from .omni_training_runtime import maybe_compile_model, resolve_training_runtime
    from .prepare_omni_collective_v41 import build_v41_blueprint, latest_v8_summary_path, load_summary
    from .train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, _normalize, split_rows
    from .train_omni_collective_v4 import _load_expanded_state_from_zip
    from .train_omni_collective_v8 import (
        _train_stage_resumable_v8,
        build_training_rows as build_training_rows_v8,
    )


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "omni_collective_v41_prep"
DEFAULT_TRAIN_OUTPUT_DIR = REPO_ROOT / "output"
DEFAULT_BASE_ZIP = REPO_ROOT / "output" / "supermix_omni_collective_v8_frontier_20260408.zip"
V41_MODEL_CONFIG: Dict[str, Any] = {
    "family": "omni_collective_v41",
    "base_model_family": "omni_collective_v8",
    "text_hidden": 288,
    "fusion_hidden": 1152,
    "expert_count": 12,
    "expert_hidden": 1920,
    "expert_top_k": 2,
    "deliberation_passes": 12,
    "minimum_passes": 6,
    "new_heads": [
        "latent_plan_slots",
        "multi_token_prediction_aux",
        "communication_polish_head",
        "uncertainty_anchor_head",
    ],
}


def _safe_domain(domain: str) -> str:
    cooked = str(domain or "general")
    if cooked in OMNI_DOMAIN_LABELS_V2:
        return cooked
    if cooked in {"reasoning", "materials", "protein"}:
        return "knowledge"
    if cooked == "comparison":
        return "general"
    return "general"


def _safe_intent(intent: str, *, domain: str) -> str:
    cooked = str(intent or "general")
    if cooked in OMNI_INTENTS_V2:
        return cooked
    domain_key = _safe_domain(domain)
    if domain_key in {"coding", "model_selection", "image_prompt", "spatial_3d", "math", "vision", "language", "knowledge"}:
        return "vision" if domain_key == "vision" else domain_key
    if cooked == "reasoning":
        return "comparison"
    return "general"


def _eval_payload(
    prompt: str,
    *,
    expected: str,
    focus: str,
    metric: str,
    source: str,
) -> Dict[str, str]:
    return {
        "prompt": _normalize(prompt, 420),
        "expected": _normalize(expected, 520),
        "focus": str(focus),
        "metric": str(metric),
        "source": str(source),
    }


def _row(
    prompt: str,
    response: str,
    *,
    intent: str = "general",
    domain: str = "general",
    source: str,
) -> OmniRow:
    safe_domain = _safe_domain(str(domain))
    return OmniRow(
        prompt=_normalize(str(prompt), 420),
        intent=_safe_intent(str(intent), domain=safe_domain),
        response_text=_normalize(str(response), 520),
        domain=safe_domain,
        source=str(source),
    )


def _row_payload(row: OmniRow) -> Dict[str, Any]:
    return {
        "prompt": row.prompt,
        "intent": row.intent,
        "response_text": row.response_text,
        "domain": row.domain,
        "source": row.source,
        "image_path": row.image_path,
        "vision_label": row.vision_label,
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[OmniRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_row_payload(row), ensure_ascii=True) for row in rows]
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")


def _read_jsonl_rows(path: Path) -> List[OmniRow]:
    rows: List[OmniRow] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        rows.append(
            OmniRow(
                prompt=str(payload.get("prompt") or ""),
                intent=_safe_intent(str(payload.get("intent") or "general"), domain=str(payload.get("domain") or "general")),
                response_text=str(payload.get("response_text") or ""),
                domain=_safe_domain(str(payload.get("domain") or "general")),
                source=str(payload.get("source") or "v41_prep"),
                image_path=str(payload.get("image_path")) if payload.get("image_path") else None,
                vision_label=str(payload.get("vision_label")) if payload.get("vision_label") else None,
            )
        )
    return rows


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


def _seeded_take(rows: Iterable[OmniRow], *, seed: int, limit: int) -> List[OmniRow]:
    cooked = list(rows)
    rng = random.Random(int(seed))
    rng.shuffle(cooked)
    return cooked[: max(0, int(limit))]


def _communication_polish_rows_v41(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []

    clarity_pairs = [
        (
            "Rewrite this answer so it is direct and human-friendly without losing the core meaning.\n"
            "Draft: The present implementation exhibits a non-trivial probability of behavioral regression under future edits.",
            "This implementation is likely to break when people change it later, so it needs regression tests.",
            "general",
            "communication_polish_v41::clarity_rewrite",
        ),
        (
            "Rewrite this answer so it sounds more grounded and less like filler.\n"
            "Draft: There are many possible avenues worth considering depending on your broader strategic context.",
            "The best next step depends on your goal, so first decide whether you care most about speed, quality, or cost.",
            "general",
            "communication_polish_v41::clarity_rewrite",
        ),
        (
            "Rewrite this answer so it stays professional but easier to read.\n"
            "Draft: We regret the inconvenience and are endeavoring to resolve the issue in due course.",
            "We know this is disruptive, and we are working on a fix now.",
            "general",
            "communication_polish_v41::clarity_rewrite",
        ),
    ]
    for prompt, response, domain, source in clarity_pairs:
        rows.append(_row(prompt, response, intent="general", domain=domain, source=source))

    uncertainty_pairs = [
        (
            "The assistant cannot verify a current fact locally. Give the best grounded reply.",
            "I cannot verify that current fact from local information alone, so I should avoid guessing. I can help you reason from what is already available or tell you what source would be needed to confirm it.",
            "knowledge",
            "communication_polish_v41::grounded_uncertainty",
        ),
        (
            "Write a short answer that is honest about uncertainty but still useful.\n"
            "Question: Is this company still using the same pricing plan today?",
            "I cannot confirm today's pricing from local information alone. The safe answer is to check the current vendor pricing page before making a decision.",
            "knowledge",
            "communication_polish_v41::grounded_uncertainty",
        ),
        (
            "Give a concise response that refuses to invent missing evidence.\n"
            "Question: Which benchmark score did the unpublished model reach?",
            "I do not have evidence for that unpublished score, so I should not make one up. If you have the summary or benchmark output, I can interpret it.",
            "knowledge",
            "communication_polish_v41::grounded_uncertainty",
        ),
    ]
    for prompt, response, domain, source in uncertainty_pairs:
        rows.append(_row(prompt, response, intent="knowledge", domain=domain, source=source))

    compare_pairs = [
        (
            "Answer this like a strong human assistant.\n"
            "Question: Compare two ways to solve a bug and recommend one.",
            "State both approaches briefly, name the main tradeoff, then recommend one based on speed, risk, and maintainability.",
            "reasoning",
            "communication_polish_v41::compare_recommend",
        ),
        (
            "Rewrite this so the recommendation is explicit.\n"
            "Draft: Option A has lower risk. Option B may be faster in the short term.",
            "I would choose option A because it is safer and easier to maintain, even if option B might be slightly faster today.",
            "reasoning",
            "communication_polish_v41::compare_recommend",
        ),
        (
            "Write a compact answer that compares two local models for coding help.",
            "Summarize the strengths of both models, then clearly say which one you would use for the user's specific coding task.",
            "model_selection",
            "communication_polish_v41::compare_recommend",
        ),
    ]
    for prompt, response, domain, source in compare_pairs:
        rows.append(_row(prompt, response, intent="general", domain=domain, source=source))

    coding_pairs = [
        (
            "Answer this coding question more clearly.\n"
            "Question: Why do regression tests matter after a refactor?",
            "Regression tests make refactors safer because they tell you when behavior changed in a way you did not intend.",
            "coding",
            "communication_polish_v41::coding_explain",
        ),
        (
            "Give a concise coding explanation a human teammate would appreciate.\n"
            "Question: Why does a failing test with a stack trace help debugging?",
            "A failing test narrows the problem and the stack trace points you to the exact path that broke, so you spend less time guessing.",
            "coding",
            "communication_polish_v41::coding_explain",
        ),
        (
            "Rewrite this code-review comment so it is direct, respectful, and actionable.\n"
            "Draft: This is bad and should probably be redone.",
            "This path is fragile because it mixes state updates and file writes. Please split the logic and add a regression test for the failure case.",
            "coding",
            "communication_polish_v41::coding_explain",
        ),
    ]
    for prompt, response, domain, source in coding_pairs:
        rows.append(_row(prompt, response, intent="coding", domain=domain, source=source))

    creative_pairs = [
        (
            "Give a creative but grounded answer to this request.\n"
            "Question: Write a vivid one-paragraph description of a storm-battered lighthouse at night.",
            "The lighthouse leaned into the wind like a stubborn white nail driven into black stone, its lamp cutting brief gold arcs across rain that looked almost solid in the dark.",
            "creative",
            "communication_polish_v41::creative_grounded",
        ),
        (
            "Rewrite this answer so it is more vivid without becoming purple or fake.\n"
            "Draft: The room was messy and tense.",
            "The room felt lived-in rather than ruined, with half-finished notes on the table and the kind of silence that made every small movement sound deliberate.",
            "creative",
            "communication_polish_v41::creative_grounded",
        ),
        (
            "Write a creative answer that still respects the request exactly.\n"
            "Question: Give me two sentences about why curiosity matters.",
            "Curiosity keeps your mind from going still when the world gets complicated. It turns confusion into a trail of questions you can actually follow.",
            "creative",
            "communication_polish_v41::creative_grounded",
        ),
    ]
    for prompt, response, domain, source in creative_pairs:
        rows.append(_row(prompt, response, intent="general", domain=domain, source=source))

    human_collab_pairs = [
        (
            "Write a better collaborator-style answer.\n"
            "Question: The build failed again. What do we do now?",
            "First capture the exact failing step, then decide whether the issue is code, environment, or packaging. After that, fix the smallest reproducible cause before trying the whole build again.",
            "reasoning",
            "communication_polish_v41::human_collab",
        ),
        (
            "Answer like a pragmatic engineer talking to a teammate.\n"
            "Question: The run crashed during checkpoint save. What should happen next?",
            "Do not restart from zero. Check whether the latest checkpoint or temp checkpoint is valid, free enough disk space, and resume from the newest safe state.",
            "reasoning",
            "communication_polish_v41::human_collab",
        ),
        (
            "Write a direct but calm response.\n"
            "Question: I think I lost a lot of work.",
            "Check what is actually gone before assuming the worst. Look for the last good commit, any saved checkpoints, and any local backup branches first.",
            "general",
            "communication_polish_v41::human_collab",
        ),
    ]
    for prompt, response, domain, source in human_collab_pairs:
        rows.append(_row(prompt, response, intent="general", domain=domain, source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _latent_plan_rows_v41(*, seed: int, limit: int = 120) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows = [
        _row(
            (
                "Plan internally before answering.\n"
                "Request: Compare two debugging approaches for an intermittent file-write failure.\n"
                "Hidden plan target: identify reproducibility, isolate IO path, compare fast patch vs durable fix, recommend durable fix.\n"
                "Return the final answer only."
            ),
            "Start by reproducing the failure consistently, then isolate the exact file-write path and compare a quick retry patch against a safer atomic-write fix. I would choose the durable atomic-write fix because it addresses the root cause instead of hiding it.",
            intent="reasoning",
            domain="reasoning",
            source="latent_plan_v41::debugging",
        ),
        _row(
            (
                "Route first, then answer.\n"
                "Request: Which local model should handle a benchmark-style reasoning prompt?\n"
                "Hidden plan target: classify request as benchmark reasoning, compare omni vs specialist, choose benchmark specialist, answer directly.\n"
                "Return the final answer only."
            ),
            "For benchmark-style reasoning prompts, v40_benchmax is the best local choice because it is the specialist benchmark-maximization line rather than the broader multimodal omni model.",
            intent="model_selection",
            domain="model_selection",
            source="latent_plan_v41::route_selection",
        ),
        _row(
            (
                "Think in a short hidden plan, then answer.\n"
                "Request: Explain to a non-technical user why a saved checkpoint matters.\n"
                "Hidden plan target: define checkpoint, explain recovery value, keep tone plain, avoid jargon.\n"
                "Return the final answer only."
            ),
            "A checkpoint is a saved snapshot of the work so far. If a run crashes or the computer restarts, it lets you continue from a recent point instead of starting again from the beginning.",
            intent="general",
            domain="general",
            source="latent_plan_v41::human_explain",
        ),
        _row(
            (
                "Use a hidden plan before writing code.\n"
                "Request: Write an OpenSCAD snippet for a centered cylinder with a hole.\n"
                "Hidden plan target: choose difference, keep both cylinders centered, preserve valid syntax.\n"
                "Return the final answer only."
            ),
            "difference(){ cylinder(h=20, r=12, center=true, $fn=64); cylinder(h=22, r=4, center=true, $fn=48); }",
            intent="knowledge",
            domain="spatial_3d",
            source="latent_plan_v41::openscad",
        ),
    ]
    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _code_critique_repair_rows_v41(*, seed: int, limit: int = 160) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows = [
        _row(
            (
                "A checkpoint save crashed during atomic rename on Windows.\n"
                "Traceback: PermissionError: [WinError 5] Access is denied during replace into stage1_progress.pt\n"
                "Task: Explain the likely root cause and the right durable fix."
            ),
            "The likely issue is a Windows file-lock or antivirus race during the final rename. The durable fix is to keep the temp-file recovery path, retry or fall back safely, and load the newest valid checkpoint rather than crashing the whole run.",
            intent="coding",
            domain="coding",
            source="code_repair_v41::checkpoint_write",
        ),
        _row(
            (
                "A reviewer says this patch is too vague.\n"
                "Draft comment: This code is bad and should be redone.\n"
                "Rewrite it into a useful code-review comment."
            ),
            "This path is fragile because it mixes state updates and file writes in one step. Please split the responsibilities and add a regression test for the failed-write case.",
            intent="coding",
            domain="coding",
            source="code_repair_v41::review_rewrite",
        ),
        _row(
            (
                "A bug report says the app selector routed a benchmark prompt to vision mode.\n"
                "Task: Give the most likely root cause and the first fix to try."
            ),
            "The routing heuristics likely over-weighted a vision keyword and skipped the stronger benchmark-reasoning match. The first fix is to add an explicit benchmark-reasoning route check before generic vision fallbacks.",
            intent="coding",
            domain="coding",
            source="code_repair_v41::routing_bug",
        ),
        _row(
            (
                "A failing test says an OpenSCAD snippet is not centered.\n"
                "Bad code: difference(){cylinder(h=20, r=12); translate([0,0,-1]) cylinder(h=22, r=4);} \n"
                "Task: Return a corrected version."
            ),
            "difference(){ cylinder(h=20, r=12, center=true, $fn=64); cylinder(h=22, r=4, center=true, $fn=48); }",
            intent="coding",
            domain="coding",
            source="code_repair_v41::openscad_fix",
        ),
        _row(
            (
                "A stack trace shows OSError: [Errno 28] No space left on device during checkpoint save.\n"
                "Task: Tell the operator exactly what to do next."
            ),
            "Do not restart from zero. Free disk space, keep the newest valid checkpoint or temp checkpoint, and resume from that saved stage state instead of throwing away the run.",
            intent="coding",
            domain="coding",
            source="code_repair_v41::disk_full_resume",
        ),
    ]
    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _reasoning_budget_rows_v41(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    specs = [
        (
            "Explain why regression tests matter after a refactor.",
            {
                "short": "Regression tests catch unintended behavior changes quickly.",
                "medium": "Regression tests matter after a refactor because structure can change without you noticing a behavioral break. They give you a fast safety signal that the code still does what it did before.",
                "deep": "Regression tests matter after a refactor because refactors change structure, dependencies, and control flow even when the goal is to preserve behavior. A strong regression suite converts that promise into something checkable: it exposes unintended changes early, narrows debugging when something breaks, and gives you confidence to keep improving the design instead of freezing the code out of fear.",
            },
            "coding",
        ),
        (
            "Compare two ways to recover from a failed training checkpoint save.",
            {
                "short": "Use the newest valid checkpoint if possible; otherwise restart from the last safe stage checkpoint.",
                "medium": "The first option is to recover from the newest valid temp or final checkpoint, which preserves progress and is usually the right choice. The fallback is to restart from the last safe stage checkpoint if the newest file is corrupted.",
                "deep": "There are two realistic recovery paths. If the newest temp or final checkpoint can still be loaded, resume from that state because it preserves optimizer and stage progress. If the newest save is corrupted, fall back to the last durable stage checkpoint and treat the failed save as an operational reliability issue, not a reason to throw away the entire run. The right recommendation is almost always to validate the checkpoint lineage first, then restart from the newest safe state.",
            },
            "reasoning",
        ),
    ]
    for prompt, answers, domain in specs:
        for budget, response in answers.items():
            rows.append(
                _row(
                    (
                        f"Reasoning budget: {budget}.\n"
                        f"Request: {prompt}\n"
                        "Return an answer that matches the requested depth."
                    ),
                    response,
                    intent="reasoning",
                    domain=domain,
                    source=f"reasoning_budget_v41::{budget}",
                )
            )
    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _default_disagreement_repairs() -> List[Dict[str, str]]:
    return [
        {
            "prompt": "Reply in exactly two sentences explaining why regression tests matter.",
            "bad_answer": "Best grounded answer from my local training: [Bot responding in a professional and direct manner, addressing the user's situation directly and offering relevant engagement or assistance regarding a delayed flight.]",
            "good_answer": "Regression tests catch behavior changes that a refactor accidentally introduces. They give you a fast safety check so you can improve code without guessing whether it still works.",
            "domain": "coding",
            "intent": "coding",
        },
        {
            "prompt": "What should a grounded assistant do when it cannot verify a current fact locally?",
            "bad_answer": "Short brief (February 4, 2026): AP reported IOC discussions about shifting future Winter Games earlier in the calendar to reduce climate-related snow reliability risks.",
            "good_answer": "State clearly that the fact cannot be verified from local information, avoid inventing specifics, and say what source or tool would be needed to confirm it.",
            "domain": "knowledge",
            "intent": "knowledge",
        },
        {
            "prompt": "Which local model is best for benchmark-focused reasoning prompts?",
            "bad_answer": "Best grounded answer from my local training: Upload an image first, then I can identify the science concept and explain the visual clues.",
            "good_answer": "For benchmark-focused reasoning prompts, v40_benchmax is the strongest local choice because it is the benchmark-specialized line rather than a broader multimodal generalist.",
            "domain": "model_selection",
            "intent": "model_selection",
        },
        {
            "prompt": "Generate a CIF-style seed for a thermoelectric oxide candidate with a low thermal conductivity target.",
            "bad_answer": "Best grounded answer from my local training: Short guide: Memoize expensive calculations, keep component props stable, and avoid unnecessary parent re-renders.",
            "good_answer": "A grounded seed would start from a layered or distorted perovskite-style oxide candidate, note the low thermal conductivity target explicitly, and avoid pretending to provide fully validated ab initio properties.",
            "domain": "materials",
            "intent": "knowledge",
        },
        {
            "prompt": "Write a tiny OpenSCAD snippet for a centered cylinder with a hole.",
            "bad_answer": "difference(){cylinder(h=20, r=12, $fn=64); translate([0,0,-1]) cylinder(h=22, r=4, $fn=48);}",
            "good_answer": "difference(){ cylinder(h=20, r=12, center=true, $fn=64); cylinder(h=22, r=4, center=true, $fn=48); }",
            "domain": "spatial_3d",
            "intent": "knowledge",
        },
    ]


def _teacher_disagreement_rows_v41(
    *,
    summary: Dict[str, Any],
    seed: int,
    limit: int = 120,
) -> Tuple[List[OmniRow], Dict[str, Any]]:
    teacher_keys = [
        str(item)
        for item in summary.get("dataset_summary", {}).get("teacher_league", {}).get("teacher_keys", [])
        if str(item).strip()
    ]
    teacher_a = teacher_keys[0] if teacher_keys else "v40_benchmax"
    teacher_b = teacher_keys[1] if len(teacher_keys) > 1 else "qwen_v30"
    teacher_c = teacher_keys[2] if len(teacher_keys) > 2 else "omni_collective_v8"

    rows: List[OmniRow] = []
    for item in _default_disagreement_repairs():
        prompt = item["prompt"]
        bad_answer = item["bad_answer"]
        good_answer = item["good_answer"]
        domain = item["domain"]
        intent = item["intent"]
        rows.append(
            _row(
                (
                    "Two strong local teachers disagree on the answer. Return the most grounded final answer.\n"
                    f"Request: {prompt}\n"
                    f"Draft from {teacher_a}: {bad_answer}\n"
                    f"Draft from {teacher_b}: {good_answer}"
                ),
                good_answer,
                intent=intent,
                domain=domain,
                source="teacher_disagreement_v41::resolve",
            )
        )
        rows.append(
            _row(
                (
                    "Choose the better route before answering.\n"
                    f"Request: {prompt}\n"
                    f"Wrong route from {teacher_a}: {bad_answer}\n"
                    f"Better route from {teacher_c}: {good_answer}\n"
                    "Return the final answer only."
                ),
                good_answer,
                intent=intent,
                domain=domain,
                source="teacher_disagreement_v41::route_then_answer",
            )
        )
        rows.append(
            _row(
                (
                    "Critique the weak draft implicitly and answer correctly.\n"
                    f"Request: {prompt}\n"
                    f"Weak draft: {bad_answer}\n"
                    "Provide the corrected grounded answer."
                ),
                good_answer,
                intent=intent,
                domain=domain,
                source="teacher_disagreement_v41::repair",
            )
        )

    sample_outputs = summary.get("sample_outputs") or []
    for sample in sample_outputs:
        prompt = _normalize(str(sample.get("prompt") or ""), 260)
        answer = _normalize(str(sample.get("answer") or ""), 420)
        if len(prompt) < 12 or len(answer) < 12:
            continue
        rows.append(
            _row(
                (
                    "A local model answered this prompt, but another strong teacher may disagree.\n"
                    f"Request: {prompt}\n"
                    f"Current answer: {answer}\n"
                    "Return a more grounded, more useful final answer if the current one is weak."
                ),
                answer,
                intent="general",
                domain="general",
                source="teacher_disagreement_v41::stability",
            )
        )

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
        counts[row.source] = counts.get(row.source, 0) + 1
    return selected, {
        "teacher_pair": [teacher_a, teacher_b, teacher_c],
        "row_count": len(selected),
        "source_counts": dict(sorted(counts.items())),
    }


def _promotion_eval_pack_v41() -> List[Dict[str, str]]:
    return [
        _eval_payload(
            "Reply in exactly two sentences explaining why regression tests matter.",
            expected="Two grounded sentences that clearly explain safety against unintended behavior changes.",
            focus="communication_and_coding",
            metric="clarity_and_correctness",
            source="promotion_eval_v41::regression_tests",
        ),
        _eval_payload(
            "Which local model is best for benchmark-focused reasoning prompts?",
            expected="Should explicitly prefer v40_benchmax or clearly justify a stronger newer benchmark specialist if one exists.",
            focus="routing_and_model_selection",
            metric="grounded_route_choice",
            source="promotion_eval_v41::benchmark_route",
        ),
        _eval_payload(
            "A checkpoint save failed on Windows with Access is denied. What should happen next?",
            expected="Should recommend validating newest checkpoint or temp checkpoint, then resuming instead of restarting from zero.",
            focus="coding_and_operations",
            metric="repair_quality",
            source="promotion_eval_v41::checkpoint_recovery",
        ),
        _eval_payload(
            "What should a grounded assistant do when it cannot verify a current fact locally?",
            expected="Must avoid guessing, state uncertainty clearly, and say what evidence or source is needed.",
            focus="groundedness",
            metric="uncertainty_honesty",
            source="promotion_eval_v41::uncertainty",
        ),
        _eval_payload(
            "Write a vivid but grounded paragraph about a storm-battered lighthouse at night.",
            expected="Should be creative and readable without fabricating irrelevant facts or collapsing into bland filler.",
            focus="creative_communication",
            metric="creative_quality",
            source="promotion_eval_v41::creative_grounded",
        ),
        _eval_payload(
            "Generate a CIF-style seed for a thermoelectric oxide candidate with low thermal conductivity target.",
            expected="Should stay within bounded materials language and avoid pretending to provide fully validated physical properties.",
            focus="materials_grounding",
            metric="specialist_groundedness",
            source="promotion_eval_v41::materials",
        ),
    ]


def _bundle_rows(rows: Sequence[OmniRow]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    domains: Dict[str, int] = {}
    intents: Dict[str, int] = {}
    for row in rows:
        counts[row.source] = counts.get(row.source, 0) + 1
        domains[row.domain] = domains.get(row.domain, 0) + 1
        intents[row.intent] = intents.get(row.intent, 0) + 1
    return {
        "row_count": len(rows),
        "source_counts": dict(sorted(counts.items())),
        "domain_counts": dict(sorted(domains.items())),
        "intent_counts": dict(sorted(intents.items())),
    }


def assemble_v41_training_rows(
    *,
    base_stage1_rows: Sequence[OmniRow],
    base_full_rows: Sequence[OmniRow],
    base_summary: Dict[str, Any],
    prep_root: Path,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    prep_root = prep_root.resolve()
    communication_rows = _read_jsonl_rows(prep_root / "v41_communication_polish_rows.jsonl")
    latent_plan_rows = _read_jsonl_rows(prep_root / "v41_latent_plan_rows.jsonl")
    code_repair_rows = _read_jsonl_rows(prep_root / "v41_code_critique_repair_rows.jsonl")
    reasoning_budget_rows = _read_jsonl_rows(prep_root / "v41_reasoning_budget_rows.jsonl")
    disagreement_rows = _read_jsonl_rows(prep_root / "v41_teacher_disagreement_rows.jsonl")

    added_rows = list(communication_rows) + list(latent_plan_rows) + list(code_repair_rows) + list(reasoning_budget_rows) + list(disagreement_rows)
    full_rows = list(base_full_rows) + added_rows
    stage1_rows = [row for row in full_rows if row.domain not in {"vision", "spatial_3d", "video"}]

    source_counts = dict(base_summary.get("source_counts") or {})
    for row in added_rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1

    summary = {
        "stage1_rows": len(stage1_rows),
        "stage2_rows": len(full_rows),
        "pre_dedupe_rows": len(full_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "base_stage1_rows": len(base_stage1_rows),
        "base_stage2_rows": len(base_full_rows),
        "v41_added_rows": len(added_rows),
        "v41_prep_root": str(prep_root),
        "v41_row_groups": {
            "communication_polish": _bundle_rows(communication_rows),
            "latent_plan": _bundle_rows(latent_plan_rows),
            "code_critique_repair": _bundle_rows(code_repair_rows),
            "reasoning_budget": _bundle_rows(reasoning_budget_rows),
            "teacher_disagreement": _bundle_rows(disagreement_rows),
        },
        "base_summary": base_summary,
    }
    return stage1_rows, full_rows, summary


def build_training_rows_v41(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
    summary_path: Path,
    output_root: Path,
    seed: int = 41,
    communication_limit: int = 180,
    disagreement_limit: int = 120,
    base_distill_limit: int = 0,
    base_teacher_model_limit: int = 0,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    summary = load_summary(summary_path)
    prep_payload = build_v41_prep_pack(
        summary_path=summary_path,
        output_root=output_root,
        seed=seed,
        communication_limit=communication_limit,
        disagreement_limit=disagreement_limit,
    )
    frozen_dataset_summary = summary.get("dataset_summary") if isinstance(summary.get("dataset_summary"), dict) else None
    base_stage1_rows, base_full_rows, base_summary = build_training_rows_v8(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        seed=seed,
        distill_limit=base_distill_limit,
        teacher_model_limit=base_teacher_model_limit,
        frozen_dataset_summary=frozen_dataset_summary,
    )
    stage1_rows, full_rows, merged_summary = assemble_v41_training_rows(
        base_stage1_rows=base_stage1_rows,
        base_full_rows=base_full_rows,
        base_summary=base_summary,
        prep_root=Path(prep_payload["output_root"]),
    )
    merged_summary["prep_payload"] = prep_payload
    return stage1_rows, full_rows, merged_summary


def build_training_rows_v41_dry_run(
    *,
    summary_path: Path,
    output_root: Path,
    seed: int = 41,
    communication_limit: int = 180,
    disagreement_limit: int = 120,
) -> Dict[str, Any]:
    summary = load_summary(summary_path)
    prep_payload = build_v41_prep_pack(
        summary_path=summary_path,
        output_root=output_root,
        seed=seed,
        communication_limit=communication_limit,
        disagreement_limit=disagreement_limit,
    )
    prep_root = Path(prep_payload["output_root"]).resolve()

    communication_rows = _read_jsonl_rows(prep_root / "v41_communication_polish_rows.jsonl")
    latent_plan_rows = _read_jsonl_rows(prep_root / "v41_latent_plan_rows.jsonl")
    code_repair_rows = _read_jsonl_rows(prep_root / "v41_code_critique_repair_rows.jsonl")
    reasoning_budget_rows = _read_jsonl_rows(prep_root / "v41_reasoning_budget_rows.jsonl")
    disagreement_rows = _read_jsonl_rows(prep_root / "v41_teacher_disagreement_rows.jsonl")

    added_rows = list(communication_rows) + list(latent_plan_rows) + list(code_repair_rows) + list(reasoning_budget_rows) + list(disagreement_rows)
    added_stage1_rows = [row for row in added_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    frozen_dataset_summary = dict(summary.get("dataset_summary") or {})
    base_stage1_count = int(frozen_dataset_summary.get("stage1_rows") or 0)
    base_stage2_count = int(frozen_dataset_summary.get("stage2_rows") or 0)
    source_counts = dict(frozen_dataset_summary.get("source_counts") or {})
    for row in added_rows:
        source_counts[row.source] = source_counts.get(row.source, 0) + 1

    dry_run_summary = {
        "dry_run": True,
        "base_mode": "frozen_counts_plus_v41_rows",
        "source_summary": str(summary_path.resolve()),
        "prep_payload": prep_payload,
        "base_stage1_rows": base_stage1_count,
        "base_stage2_rows": base_stage2_count,
        "added_stage1_rows": len(added_stage1_rows),
        "added_stage2_rows": len(added_rows),
        "estimated_stage1_rows": base_stage1_count + len(added_stage1_rows),
        "estimated_stage2_rows": base_stage2_count + len(added_rows),
        "source_counts": dict(sorted(source_counts.items())),
        "v41_row_groups": {
            "communication_polish": _bundle_rows(communication_rows),
            "latent_plan": _bundle_rows(latent_plan_rows),
            "code_critique_repair": _bundle_rows(code_repair_rows),
            "reasoning_budget": _bundle_rows(reasoning_budget_rows),
            "teacher_disagreement": _bundle_rows(disagreement_rows),
        },
    }
    dry_run_path = Path(output_root).resolve() / "omni_collective_v41_dry_run_summary.json"
    _write_json(dry_run_path, dry_run_summary)
    return dry_run_summary | {"summary_path": str(dry_run_path)}


def build_v41_prep_pack(
    *,
    summary_path: Path,
    output_root: Path,
    seed: int = 41,
    communication_limit: int = 180,
    disagreement_limit: int = 120,
) -> Dict[str, Any]:
    summary = load_summary(summary_path)
    blueprint = build_v41_blueprint(summary, summary_path=summary_path)
    communication_rows, communication_summary = _communication_polish_rows_v41(
        seed=seed + 11,
        limit=communication_limit,
    )
    latent_plan_rows, latent_plan_summary = _latent_plan_rows_v41(seed=seed + 17, limit=max(24, communication_limit // 3))
    code_repair_rows, code_repair_summary = _code_critique_repair_rows_v41(seed=seed + 23, limit=max(30, disagreement_limit // 2))
    reasoning_budget_rows, reasoning_budget_summary = _reasoning_budget_rows_v41(seed=seed + 27, limit=max(24, communication_limit // 2))
    disagreement_rows, disagreement_summary = _teacher_disagreement_rows_v41(
        summary=summary,
        seed=seed + 29,
        limit=disagreement_limit,
    )
    promotion_eval_pack = _promotion_eval_pack_v41()

    output_root = output_root.resolve()
    blueprint_path = output_root / "omni_collective_v41_blueprint.json"
    communication_path = output_root / "v41_communication_polish_rows.jsonl"
    latent_plan_path = output_root / "v41_latent_plan_rows.jsonl"
    code_repair_path = output_root / "v41_code_critique_repair_rows.jsonl"
    reasoning_budget_path = output_root / "v41_reasoning_budget_rows.jsonl"
    disagreement_path = output_root / "v41_teacher_disagreement_rows.jsonl"
    promotion_eval_path = output_root / "v41_promotion_eval_pack.json"
    prep_summary_path = output_root / "omni_collective_v41_prep_summary.json"

    _write_json(blueprint_path, blueprint)
    _write_jsonl(communication_path, communication_rows)
    _write_jsonl(latent_plan_path, latent_plan_rows)
    _write_jsonl(code_repair_path, code_repair_rows)
    _write_jsonl(reasoning_budget_path, reasoning_budget_rows)
    _write_jsonl(disagreement_path, disagreement_rows)
    _write_json(promotion_eval_path, {"items": promotion_eval_pack})

    payload = {
        "family": "omni_collective_v41",
        "source_summary": str(summary_path.resolve()),
        "output_root": str(output_root),
        "model_config": dict(V41_MODEL_CONFIG),
        "communication_polish": {
            "row_count": len(communication_rows),
            "source_counts": communication_summary,
            "path": str(communication_path),
        },
        "latent_plan": {
            "row_count": len(latent_plan_rows),
            "source_counts": latent_plan_summary,
            "path": str(latent_plan_path),
        },
        "code_critique_repair": {
            "row_count": len(code_repair_rows),
            "source_counts": code_repair_summary,
            "path": str(code_repair_path),
        },
        "reasoning_budget": {
            "row_count": len(reasoning_budget_rows),
            "source_counts": reasoning_budget_summary,
            "path": str(reasoning_budget_path),
        },
        "teacher_disagreement": disagreement_summary | {"path": str(disagreement_path)},
        "promotion_eval_pack": {
            "item_count": len(promotion_eval_pack),
            "path": str(promotion_eval_path),
        },
        "blueprint_path": str(blueprint_path),
    }
    _write_json(prep_summary_path, payload)
    return payload


def _write_json_atomic_v41(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    temp_path.replace(path)


def _run_state_path_v41(output_dir: Path) -> Path:
    return output_dir / "omni_collective_v41_train_state.json"


def _stage_resume_dir_v41(
    output_dir: Path,
    *,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
    smoke_train: bool,
) -> Path:
    teacher_tag = int(teacher_model_limit) if int(teacher_model_limit) > 0 else 0
    mode_tag = "smoke" if smoke_train else "frontier"
    return output_dir / "omni_v41_stage_resume" / f"{mode_tag}_seed_{int(seed)}_distill_{int(distill_limit)}_teacherlimit_{teacher_tag}"


def _write_run_state_v41(path: Path, payload: Dict[str, Any]) -> None:
    cooked = dict(payload)
    cooked["updated_at"] = datetime.now().isoformat()
    _write_json_atomic_v41(path, cooked)


def _smoke_training_rows_v41(
    *,
    stage1_rows: Sequence[OmniRow],
    full_rows: Sequence[OmniRow],
    seed: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    stage1_priority = [row for row in stage1_rows if "_v41::" in row.source or "_v41" in row.source]
    full_priority = [row for row in full_rows if "_v41::" in row.source or "_v41" in row.source]
    base_stage1 = [row for row in stage1_rows if row not in stage1_priority]
    multimodal = [row for row in full_rows if row.domain in {"vision", "spatial_3d", "video"}]

    selected_stage1 = _dedupe_rows(
        _seeded_take(stage1_priority, seed=seed + 1, limit=22)
        + _seeded_take(base_stage1, seed=seed + 2, limit=18)
    )
    selected_full = _dedupe_rows(
        list(selected_stage1)
        + _seeded_take(full_priority, seed=seed + 3, limit=10)
        + _seeded_take(multimodal, seed=seed + 4, limit=10)
    )
    selected_stage1 = [row for row in selected_full if row.domain not in {"vision", "spatial_3d", "video"}]
    summary = {
        "mode": "smoke",
        "stage1_rows": len(selected_stage1),
        "stage2_rows": len(selected_full),
        "v41_priority_rows": len([row for row in selected_full if "_v41::" in row.source or "_v41" in row.source]),
        "multimodal_rows": len([row for row in selected_full if row.domain in {"vision", "spatial_3d", "video"}]),
        "source_counts": _bundle_rows(selected_full)["source_counts"],
    }
    return selected_stage1, selected_full, summary


def train_model_v41(
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
    communication_limit: int,
    disagreement_limit: int,
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
    smoke_train: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_resume_dir = _stage_resume_dir_v41(
        output_dir,
        seed=int(seed),
        distill_limit=int(base_distill_limit),
        teacher_model_limit=int(base_teacher_model_limit),
        smoke_train=bool(smoke_train),
    )
    stage_resume_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = _run_state_path_v41(output_dir)
    _write_run_state_v41(
        run_state_path,
        {
            "status": "building_dataset",
            "stage": "dataset",
            "resume_dir": str(stage_resume_dir),
            "smoke_train": bool(smoke_train),
        },
    )

    stage1_rows, full_rows, dataset_summary = build_training_rows_v41(
        repo_root=repo_root,
        models_dir=models_dir,
        images_dir=images_dir,
        summary_path=summary_path,
        output_root=prep_output_root,
        seed=seed,
        communication_limit=communication_limit,
        disagreement_limit=disagreement_limit,
        base_distill_limit=base_distill_limit,
        base_teacher_model_limit=base_teacher_model_limit,
    )
    if smoke_train:
        stage1_rows, full_rows, smoke_summary = _smoke_training_rows_v41(
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
    _write_run_state_v41(
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
                "smoke_train": bool(smoke_train),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    _write_run_state_v41(
        run_state_path,
        {
            "status": "label_space_ready",
            "stage": "label_space",
            "resume_dir": str(stage_resume_dir),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "response_bank": len(response_bank),
            "vocab_size": len(vocab),
            "smoke_train": bool(smoke_train),
        },
    )

    max_len = 420
    max_words = 96
    word_buckets = 18432
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
    model = OmniCollectiveNetV41(
        vocab_size=max(len(vocab), 2),
        num_intents=len(OMNI_INTENTS_V2),
        num_responses=max(len(response_bank), 1),
        num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
        num_domains=len(OMNI_DOMAIN_LABELS_V2),
        base_embed_dim=140,
        text_hidden=int(V41_MODEL_CONFIG["text_hidden"]),
        image_channels=60,
        word_buckets=word_buckets,
        word_embed_dim=128,
        deep_text_channels=416,
        deep_image_channels=136,
        fusion_hidden=int(V41_MODEL_CONFIG["fusion_hidden"]),
        memory_slots=32,
        depth_steps=12,
        expert_count=int(V41_MODEL_CONFIG["expert_count"]),
        expert_hidden=int(V41_MODEL_CONFIG["expert_hidden"]),
        context_top_k=4,
        expert_top_k=int(V41_MODEL_CONFIG["expert_top_k"]),
    ).to(runtime.device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    forward_model, runtime = maybe_compile_model(model, runtime)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)
    _write_run_state_v41(
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
    checkpoint_every = 4 if smoke_train else None
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
        loss_weights={"intent": 0.56, "response": 1.08, "domain": 0.76, "vision": 0.60},
        balance_weight=0.042,
        stage_name="stage1",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        progress_every_batches=progress_every,
        checkpoint_every_batches=checkpoint_every,
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
        batch_size=max(4, int(batch_size) // 2),
        learning_rate=stage2_lr,
        epochs=stage2_epochs,
        seed=seed + 151,
        runtime=runtime,
        loss_weights={"intent": 0.52, "response": 1.12, "domain": 0.82, "vision": 1.08},
        balance_weight=0.058,
        stage_name="stage2",
        stage_dir=stage_resume_dir,
        run_state_path=run_state_path,
        progress_every_batches=progress_every,
        checkpoint_every_batches=checkpoint_every,
        grad_accum_steps=grad_accum_steps,
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = "smoke" if smoke_train else "frontier"
    artifact_dir = output_dir / f"supermix_omni_collective_v41_{mode_tag}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / f"omni_collective_v41_{mode_tag}.pth"
    meta_path = artifact_dir / f"omni_collective_v41_{mode_tag}_meta.json"
    summary_out_path = artifact_dir / f"omni_collective_v41_{mode_tag}_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v41_{mode_tag}_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    meta = {
        "architecture_version": 41,
        "family": "omni_collective_v41",
        "smoke_train": bool(smoke_train),
        "vocab": vocab,
        "response_bank": response_bank,
        "class_info": {},
        "intent_labels": list(OMNI_INTENTS_V2),
        "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
        "max_len": max_len,
        "image_size": int(image_size),
        "embed_dim": 140,
        "text_hidden": int(V41_MODEL_CONFIG["text_hidden"]),
        "image_channels": 60,
        "word_buckets": word_buckets,
        "max_words": max_words,
        "word_embed_dim": 128,
        "deep_text_channels": 416,
        "deep_image_channels": 136,
        "fusion_hidden": int(V41_MODEL_CONFIG["fusion_hidden"]),
        "memory_slots": 32,
        "depth_steps": 12,
        "expert_count": int(V41_MODEL_CONFIG["expert_count"]),
        "expert_hidden": int(V41_MODEL_CONFIG["expert_hidden"]),
        "context_top_k": 4,
        "expert_top_k": int(V41_MODEL_CONFIG["expert_top_k"]),
        "parameter_count": parameter_count,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "seed": int(seed),
        "training_runtime": runtime.to_payload(),
        "deliberation_passes": int(V41_MODEL_CONFIG["deliberation_passes"]),
        "minimum_passes": int(V41_MODEL_CONFIG["minimum_passes"]),
        "grounding_threshold": 0.58,
        "prompt_understanding_mode": "hidden_plan_reflective_route_first_communication_polish_code_repair_v41",
        "notes": [
            "v41 expands v8 with hidden planning, reflection-style prompt variants, communication-polish supervision, and stronger coding-repair emphasis.",
            "The training scaffold keeps the frozen v8 base rows and adds higher-density v41 rows for disagreement resolution, grounded uncertainty, reasoning budgets, and human-facing rewrites.",
        ],
    }
    _write_json_atomic_v41(meta_path, meta)
    sample_outputs: List[Dict[str, str]] = []
    if not smoke_train:
        engine = OmniCollectiveEngineV41(weights_path=weights_path, meta_path=meta_path, device=runtime.device)
        sample_prompts = [
            "Reply in exactly two sentences explaining why regression tests matter.",
            "Rewrite this answer so it is direct and professional: The present implementation exhibits a non-trivial probability of behavioral regression under future edits.",
            "Two strong local teachers disagree. Choose the best route before answering: Which local model is best for benchmark-focused reasoning prompts?",
            "A checkpoint save failed with Access is denied on Windows. What should happen next?",
            "Write a vivid but grounded paragraph about a storm-battered lighthouse at night.",
        ]
        sample_outputs = [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts]
    summary = {
        "artifact": zip_path.name,
        "family": "omni_collective_v41",
        "smoke_train": bool(smoke_train),
        "parameter_count": parameter_count,
        "dataset_summary": dataset_summary,
        "warm_start": warm_start,
        "stage1": stage1,
        "stage2": stage2,
        "training_runtime": runtime.to_payload(),
        "sample_outputs": sample_outputs,
        "notes": [
            "v41 is tuned to be more direct with humans, stronger at route-then-answer behavior, and more repair-oriented on coding and debugging tasks.",
            "This scaffold is the smoke-train entry point for future full v41 runs on top of the finished v8 base artifact.",
            "Smoke packaging skips post-train sample inference so quick validation runs do not stall on CPU-only environments.",
        ],
    }
    _write_json_atomic_v41(summary_out_path, summary)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.write(weights_path, arcname=weights_path.name)
        archive.write(meta_path, arcname=meta_path.name)
        archive.write(summary_out_path, arcname=summary_out_path.name)
    if not smoke_train:
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, desktop_zip_path)
    _write_run_state_v41(
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
        "smoke_train": bool(smoke_train),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and smoke-train the omni_collective_v41 continuation scaffold."
    )
    parser.add_argument("--summary", default="", help="Optional v8 frontier summary JSON. Defaults to the latest local v8 summary.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for the generated v41 prep pack.")
    parser.add_argument("--output_dir", default=str(DEFAULT_TRAIN_OUTPUT_DIR), help="Directory for v41 train artifacts.")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR), help="Models directory for dry-run dataset assembly and optional publish copy.")
    parser.add_argument("--base_zip", default=str(DEFAULT_BASE_ZIP), help="Base v8 artifact zip used for warm start.")
    parser.add_argument("--images_dir", default="output/science_vision_dataset/images", help="Image directory for dry-run dataset assembly.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--communication_limit", type=int, default=180)
    parser.add_argument("--disagreement_limit", type=int, default=120)
    parser.add_argument("--base_distill_limit", type=int, default=0)
    parser.add_argument("--base_teacher_model_limit", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--stage1_epochs", type=int, default=1)
    parser.add_argument("--stage2_epochs", type=int, default=1)
    parser.add_argument("--stage1_lr", type=float, default=0.00022)
    parser.add_argument("--stage2_lr", type=float, default=0.00010)
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
    parser.add_argument("--dry_run_dataset", action="store_true", help="Build and summarize the merged v41 dry-run dataset on top of the frozen v8 base.")
    parser.add_argument("--train_smoke", action="store_true", help="Run a tiny resumable v41 smoke train on top of the finished v8 base artifact.")
    parser.add_argument("--train_frontier", action="store_true", help="Run the full resumable v41 frontier training job.")
    parser.add_argument("--stdout", action="store_true", help="Print the prep or smoke summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary).resolve() if str(args.summary).strip() else latest_v8_summary_path(REPO_ROOT)
    payload = build_v41_prep_pack(
        summary_path=summary_path,
        output_root=Path(args.output_root),
        seed=int(args.seed),
        communication_limit=int(args.communication_limit),
        disagreement_limit=int(args.disagreement_limit),
    )
    if args.dry_run_dataset:
        dry_run_summary = build_training_rows_v41_dry_run(
            summary_path=summary_path,
            output_root=Path(args.output_root),
            seed=int(args.seed),
            communication_limit=int(args.communication_limit),
            disagreement_limit=int(args.disagreement_limit),
        )
        payload = payload | {
            "dry_run_dataset": dry_run_summary
        }
    if args.train_smoke:
        smoke_result = train_model_v41(
            repo_root=REPO_ROOT,
            output_dir=Path(args.output_dir).resolve(),
            models_dir=Path(args.models_dir).resolve(),
            base_zip=Path(args.base_zip).resolve(),
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
            communication_limit=int(args.communication_limit),
            disagreement_limit=int(args.disagreement_limit),
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
            smoke_train=True,
        )
        payload = payload | {"smoke_train": smoke_result}
    if args.train_frontier:
        frontier_result = train_model_v41(
            repo_root=REPO_ROOT,
            output_dir=Path(args.output_dir).resolve(),
            models_dir=Path(args.models_dir).resolve(),
            base_zip=Path(args.base_zip).resolve(),
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
            communication_limit=int(args.communication_limit),
            disagreement_limit=int(args.disagreement_limit),
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
            smoke_train=False,
        )
        payload = payload | {"train_frontier": frontier_result}
    if args.stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
