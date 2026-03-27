#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from build_v32_omnifuse_dataset import (
    PromptRow,
    candidate_penalty,
    extract_pair,
    generate_champion_outputs,
    generate_qwen_outputs,
    normalize_response,
    split_holdout_rows,
    stratified_sample,
)
from qwen_supermix_pipeline import token_f1


BASE_TEACHER_PRIOR = {
    "qwen_base": 0.015,
    "qwen_v28": 0.060,
    "v32_latest": 0.105,
    "v33_stage1": 0.095,
    "v33_stage2_best": 0.130,
    "hybrid_v31": 0.040,
    "lite_v30": 0.016,
    "champion_v28": 0.024,
}

PAPER_NOTES = (
    {
        "slug": "moda_2026",
        "title": "Mixture-of-Depths Attention",
        "url": "https://arxiv.org/abs/2603.15619",
        "summary": "Mixture-of-Depths Attention reuses useful hidden-state structure from earlier depth so deeper computation can preserve signals instead of rewriting them.",
        "takeaway": "For a compact Supermix student, later reasoning blocks should be able to revisit earlier hidden drafts under a learned compute budget.",
        "tradeoff": "It improves depth efficiency, but it introduces routing and state-management complexity that must stay stable on small hardware.",
        "analogy": "It is like keeping the strongest earlier drafts on the desk so the final draft can borrow them rather than start over.",
    },
    {
        "slug": "fast_quietstar_2025",
        "title": "Fast Quiet-STaR: Thinking Without Thought Tokens",
        "url": "https://aclanthology.org/2025.findings-emnlp.1020.pdf",
        "summary": "Fast Quiet-STaR trains reasoning traces first and then compresses them so the model keeps the quality of thought without emitting long hidden traces at inference time.",
        "takeaway": "For Supermix, teach richer reasoning in stage one, then polish toward shorter direct answers while preserving the same correctness.",
        "tradeoff": "It avoids test-time reasoning bloat, but the curriculum becomes more sensitive to data quality and stage ordering.",
        "analogy": "It is like teaching a student to show the full working first, then expecting the same accuracy from a clean final answer.",
    },
    {
        "slug": "titans_2025",
        "title": "Titans: Learning to Memorize at Test Time",
        "url": "https://www.kognition.info/wp-content/uploads/2025/02/2501.00663v1.pdf",
        "summary": "Titans adds a learned neural memory that writes selectively so difficult contexts can reuse important intermediate state later in the computation.",
        "takeaway": "The compact adaptation is a small memory bank with surprise-gated writes so harder prompts preserve useful internal notes.",
        "tradeoff": "Memory helps on harder prompts, but stale or noisy writes can destabilize small models if the write policy is weak.",
        "analogy": "It is like updating a notebook only when something surprising happens so the next pass starts from the best notes.",
    },
    {
        "slug": "spin_2024",
        "title": "Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models",
        "url": "https://arxiv.org/abs/2401.01335",
        "summary": "SPIN sharpens a model by comparing its current behavior with older weaker behavior and moving outputs toward the target data distribution.",
        "takeaway": "A final repair pass should focus on concise, high-value corrections instead of repeating the whole training mix again.",
        "tradeoff": "Self-play style repair can sharpen strengths, but it can also reinforce existing errors if the repair set is noisy.",
        "analogy": "It is like revising a draft by comparing it with an older weaker draft and only keeping edits that move closer to the editor-approved answer.",
    },
    {
        "slug": "mod_2024",
        "title": "Mixture-of-Depths",
        "url": "https://arxiv.org/abs/2404.02258",
        "summary": "Mixture-of-Depths allocates compute adaptively so only the most useful positions receive deeper processing under a fixed budget.",
        "takeaway": "The compact-model lesson is adaptive compute: spend more routed capacity on hard reasoning or coding prompts and less on easy ones.",
        "tradeoff": "It saves compute, but the routing policy becomes another optimization problem that can overfit if it is not regularized.",
        "analogy": "It is like triage: easy cases move quickly while complex cases get more specialist attention.",
    },
    {
        "slug": "agent_distill_2025",
        "title": "Distilling LLM Agent into Small Models with Retrieval and Code Tools",
        "url": "https://arxiv.org/abs/2505.17612",
        "summary": "Agent Distillation shows that small models improve when they inherit not only answers but full task-solving behavior, including retrieval and code-tool use.",
        "takeaway": "For Supermix, add tool-aware verification and retrieval-style repair rows so factual and calculation-heavy prompts are trained with explicit checking behavior.",
        "tradeoff": "Richer trajectories improve robustness, but they add data complexity and can inject tool-specific noise if the traces are weak.",
        "analogy": "It is like teaching a junior engineer not just the final fix, but also when to search docs and when to run a quick sanity script.",
    },
    {
        "slug": "tshirt_2025",
        "title": "T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning",
        "url": "https://arxiv.org/abs/2506.01317",
        "summary": "T-SHIRT argues that curation should reward informative tokens and robust neighborhoods, not just coarse sample-level scores.",
        "takeaway": "For Supermix, expand the dataset but select the stage-two reference pool with an informativeness-and-diversity score instead of raw random sampling.",
        "tradeoff": "Better selection can beat using all data, but the scoring heuristic can bias the dataset if it over-rewards flashy prompts.",
        "analogy": "It is like choosing training drills by how much they teach, not just by how many pages they occupy in the workbook.",
    },
    {
        "slug": "t1_2025",
        "title": "T1: Tool-integrated Self-verification for Test-time Compute Scaling in Small Language Models",
        "url": "https://arxiv.org/abs/2504.04718",
        "summary": "T1 shows that small models verify better when memorization-heavy checks are delegated to tools such as code execution.",
        "takeaway": "In post-training, create explicit self-check and correction rows for numerical, factual, and code-heavy tasks so the model learns verification behavior.",
        "tradeoff": "Verification improves reliability, but overemphasizing it can make responses stiff or overly procedural on easy prompts.",
        "analogy": "It is like teaching someone to use a calculator or quick script for the brittle parts instead of trusting mental math every time.",
    },
    {
        "slug": "efficient_reasoners_2025",
        "title": "Making Small Language Models Efficient Reasoners: Intervention, Supervision, Reinforcement",
        "url": "https://arxiv.org/abs/2505.07961",
        "summary": "This work shows that small models benefit when reasoning length is controlled explicitly instead of letting verbose traces expand without discipline.",
        "takeaway": "For Supermix, keep the reasoning budget tags and strengthen concise direct-answer supervision rather than assuming longer always means better.",
        "tradeoff": "Length control improves token efficiency, but too much pressure toward short answers can cut out genuinely useful reasoning steps.",
        "analogy": "It is like teaching someone to show enough working to be correct, but not to turn every answer into a page of rambling notes.",
    },
    {
        "slug": "limo_2025",
        "title": "LIMO: Less is More for Reasoning",
        "url": "https://arxiv.org/abs/2502.03387",
        "summary": "LIMO argues that a small number of carefully orchestrated reasoning demonstrations can unlock capabilities better than much larger noisy sets.",
        "takeaway": "For Supermix, keep a curated high-signal repair subset even while expanding the broader dataset, because not every extra row deserves equal weight.",
        "tradeoff": "Curated examples can punch above their weight, but over-trimming can leave obvious coverage gaps across domains.",
        "analogy": "It is like a short set of master-level solved examples that teach the method better than a huge pile of mediocre homework.",
    },
    {
        "slug": "intrinsic_self_correction_2024",
        "title": "Large Language Models have Intrinsic Self-Correction Ability",
        "url": "https://arxiv.org/pdf/2406.15673",
        "summary": "This work argues that self-correction improves when the model is prompted to verify calmly and deterministically instead of with noisy sampling.",
        "takeaway": "For Supermix, build low-temperature self-repair rows that ask for verification and correction directly, especially for reasoning and knowledge prompts.",
        "tradeoff": "Self-correction is helpful, but without a clean target it can easily turn into stylistic rewrites rather than real error repair.",
        "analogy": "It is like rereading a draft slowly with a checklist instead of trying to improve it by improvising another flashy draft.",
    },
    {
        "slug": "quietstar_2024",
        "title": "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking",
        "url": "https://arxiv.org/abs/2403.09629",
        "summary": "Quiet-STaR shows that latent reasoning traces can improve downstream answers even when the final response remains concise.",
        "takeaway": "For Supermix, keep high-signal reasoning supervision in the training mix while still optimizing the visible answer to stay direct and compact.",
        "tradeoff": "Latent-thought style training can boost hard-task accuracy, but weak traces can become noise if they are not filtered aggressively.",
        "analogy": "It is like practicing on scratch paper, then handing in only the clean final answer.",
    },
    {
        "slug": "ih_challenge_2026",
        "title": "IH-Challenge: A Training Dataset to Improve Instruction Hierarchy",
        "url": "https://cdn.openai.com/pdf/14e541fa-7e48-4d79-9cbf-61c3cde3e263/ih-challenge-paper.pdf",
        "summary": "IH-Challenge trains models to respect higher-priority instructions consistently instead of being distracted by conflicting lower-priority requests.",
        "takeaway": "For Supermix, keep system-tag supervision explicit and include more rows where the answer stays aligned with the top-level constraints and formatting tags.",
        "tradeoff": "Hierarchy training improves robustness, but overdoing it can make answers overly rigid when prompts are benign and straightforward.",
        "analogy": "It is like teaching a junior teammate to follow the project spec first, even if an offhand chat message asks for something conflicting.",
    },
)


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


def load_prompt_rows_safe(paths: Sequence[Path]) -> List[PromptRow]:
    rows: List[PromptRow] = []
    seen_users: set[str] = set()
    for path in paths:
        skipped_bad = 0
        skipped_lfs = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                if raw.startswith("version https://git-lfs.github.com/spec/v1") or raw.startswith("oid sha256:") or raw.startswith("size "):
                    skipped_lfs += 1
                    continue
                try:
                    pair = extract_pair(json.loads(raw))
                except Exception:
                    skipped_bad += 1
                    continue
                if pair is None:
                    continue
                key = pair.user.lower()
                if key in seen_users:
                    continue
                seen_users.add(key)
                rows.append(pair)
        if skipped_lfs > 0 or skipped_bad > 0:
            print(f"[v33-frontier] skipped file noise in {path.name}: lfs={skipped_lfs} bad_json={skipped_bad}")
    return rows


def infer_task_mode(row: PromptRow) -> str:
    query = row.user.lower()
    if any(tok in query for tok in ("creative", "story", "poem", "analogy", "metaphor", "brainstorm", "vivid")):
        return "creative"
    if any(tok in query for tok in ("code", "python", "javascript", "api", "bug", "patch", "refactor", "test")):
        return "coding"
    if any(tok in query for tok in ("latest", "recent", "newest", "2026", "2025", "paper", "research", "arxiv")):
        return "knowledge"
    return "reasoning"


def infer_budget(row: PromptRow) -> str:
    query = row.user.lower()
    if any(tok in query for tok in ("one line", "brief", "short", "concise")):
        return "short"
    if infer_task_mode(row) in {"reasoning", "coding"}:
        return "deep"
    return "medium"


def infer_creativity_level(row: PromptRow) -> str:
    mode = infer_task_mode(row)
    if mode == "creative":
        return "vivid"
    if mode == "knowledge":
        return "precise"
    return "balanced"


def infer_recency(row: PromptRow) -> str:
    explicit = row.metadata.get("knowledge_recency")
    if explicit:
        return str(explicit).strip().lower()
    if row.source.startswith("frontier_paper"):
        return "latest"
    return "evergreen"


def infer_source_quality(row: PromptRow) -> str:
    explicit = row.metadata.get("source_quality")
    if explicit:
        return str(explicit).strip().lower()
    if row.source.startswith("frontier_paper"):
        return "research"
    if "official" in row.source:
        return "official"
    return "synthetic" if row.source.startswith("v33_") else "official"


def dynamic_teacher_prior(row: PromptRow, teacher: str) -> float:
    prior = float(BASE_TEACHER_PRIOR.get(teacher, 0.0))
    mode = infer_task_mode(row)
    recency = infer_recency(row)
    creativity = infer_creativity_level(row)
    normalized_teacher = teacher.strip().lower()

    if normalized_teacher.startswith("qwen_"):
        prior += 0.012
    if "stage2" in normalized_teacher or "chosen" in normalized_teacher:
        prior += 0.035
    elif "stage1" in normalized_teacher:
        prior += 0.012
    elif "final" in normalized_teacher:
        prior += 0.020
    if "smoke" in normalized_teacher:
        prior -= 0.055

    if mode == "creative":
        if teacher == "hybrid_v31":
            prior += 0.08
        if teacher == "lite_v30":
            prior += 0.03
        if teacher == "v33_stage1":
            prior += 0.03
        if teacher == "v32_latest":
            prior += 0.01
        if "v34" in normalized_teacher:
            prior += 0.02
    if mode in {"reasoning", "coding"}:
        if teacher == "v33_stage2_best":
            prior += 0.08
        if teacher == "v33_stage1":
            prior += 0.05
        if teacher == "v32_latest":
            prior += 0.03
        if teacher == "champion_v28":
            prior += 0.02
        if "v34" in normalized_teacher:
            prior += 0.06
        if "v35" in normalized_teacher:
            prior += 0.07
    if mode == "knowledge":
        if teacher == "v33_stage2_best":
            prior += 0.04
        if teacher == "qwen_v28":
            prior += 0.03
        if normalized_teacher.startswith("qwen_v29") or normalized_teacher.startswith("qwen_v30"):
            prior += 0.025
    if recency == "latest":
        if teacher == "qwen_v28":
            prior += 0.04
        if teacher == "qwen_base":
            prior += 0.01
        if teacher == "v33_stage2_best":
            prior += 0.03
        if teacher == "v32_latest":
            prior += 0.02
    if creativity == "precise" and teacher == "qwen_v28":
        prior += 0.02
    return prior


def parse_named_path_spec(spec: str) -> Tuple[str, Path]:
    raw = str(spec).strip()
    if "=" not in raw:
        raise ValueError(f"Expected NAME=PATH spec, got: {spec!r}")
    name, path_text = raw.split("=", 1)
    name = name.strip()
    path_text = path_text.strip()
    if not name or not path_text:
        raise ValueError(f"Expected non-empty NAME and PATH in spec: {spec!r}")
    return name, Path(path_text).expanduser().resolve()


def parse_named_pair_spec(spec: str) -> Tuple[str, Path, Path]:
    raw = str(spec).strip()
    parts = raw.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected NAME=WEIGHTS::META spec, got: {spec!r}")
    name = parts[0].strip()
    payload = parts[1].strip()
    if "::" not in payload:
        raise ValueError(f"Expected WEIGHTS::META payload in spec: {spec!r}")
    weights_text, meta_text = payload.split("::", 1)
    weights_path = Path(weights_text.strip()).expanduser().resolve()
    meta_path = Path(meta_text.strip()).expanduser().resolve()
    if not name or not str(weights_path) or not str(meta_path):
        raise ValueError(f"Expected non-empty NAME, WEIGHTS, and META in spec: {spec!r}")
    return name, weights_path, meta_path


def row_informativeness(row: PromptRow) -> float:
    user_words = row.user.split()
    assistant_words = row.assistant.split()
    combined_words = [word.lower() for word in user_words + assistant_words]
    unique_ratio = len(set(combined_words)) / max(1, len(combined_words))
    digit_ratio = sum(char.isdigit() for char in row.user + row.assistant) / max(1, len(row.user) + len(row.assistant))
    code_hint = 0.0
    if any(tok in row.user.lower() for tok in ("```", "def ", "class ", "import ", "api", "json", "sql", "bash", "python", "javascript")):
        code_hint += 0.20
    if any(tok in row.assistant.lower() for tok in ("return", "function", "SELECT ", "pip ", "http", "{", "}", "[]")):
        code_hint += 0.10

    mode = infer_task_mode(row)
    source_quality = infer_source_quality(row)
    recency = infer_recency(row)

    score = 0.0
    score += min(len(user_words), 56) / 56.0 * 0.18
    score += min(len(assistant_words), 88) / 88.0 * 0.16
    score += unique_ratio * 0.22
    score += min(digit_ratio * 18.0, 0.10)
    score += code_hint
    if mode == "reasoning":
        score += 0.16
    elif mode == "coding":
        score += 0.20
    elif mode == "knowledge":
        score += 0.14
    else:
        score += 0.10
    if source_quality in {"official", "research"}:
        score += 0.12
    if recency == "latest":
        score += 0.10
    if len(assistant_words) > 168:
        score -= 0.05
    return float(score)


def select_reference_rows(rows: Sequence[PromptRow], *, sample_size: int, seed: int) -> List[PromptRow]:
    if sample_size <= 0 or sample_size >= len(rows):
        return list(rows)

    rng = random.Random(seed)
    groups: Dict[str, List[PromptRow]] = {}
    for row in rows:
        groups.setdefault(row.source, []).append(row)

    weighted_sources = []
    for source, items in groups.items():
        weighted_sources.append((source, items, math.sqrt(max(1, len(items)))))
    rng.shuffle(weighted_sources)

    total_weight = sum(weight for _, _, weight in weighted_sources) or 1.0
    quotas: Dict[str, int] = {}
    for source, items, weight in weighted_sources:
        quota = max(1, int(round(sample_size * weight / total_weight)))
        quotas[source] = min(quota, len(items))

    current_total = sum(quotas.values())
    if current_total > sample_size:
        ranked_sources = sorted(
            weighted_sources,
            key=lambda item: (quotas[item[0]] > 1, quotas[item[0]] - 1, len(item[1])),
            reverse=True,
        )
        overflow = current_total - sample_size
        for source, _, _ in ranked_sources:
            while overflow > 0 and quotas[source] > 1:
                quotas[source] -= 1
                overflow -= 1
            if overflow <= 0:
                break
    elif current_total < sample_size:
        deficit = sample_size - current_total
        ranked_sources = sorted(weighted_sources, key=lambda item: len(item[1]), reverse=True)
        while deficit > 0:
            changed = False
            for source, items, _ in ranked_sources:
                if quotas[source] >= len(items):
                    continue
                quotas[source] += 1
                deficit -= 1
                changed = True
                if deficit <= 0:
                    break
            if not changed:
                break

    selected: List[PromptRow] = []
    seen_users: set[str] = set()
    leftovers: List[PromptRow] = []

    for source, items, _ in weighted_sources:
        ranked_items = sorted(
            items,
            key=lambda row: (
                row_informativeness(row) + rng.random() * 0.015,
                infer_task_mode(row),
                infer_recency(row),
            ),
            reverse=True,
        )
        keep = quotas.get(source, 0)
        for row in ranked_items[:keep]:
            key = row.user.lower()
            if key in seen_users:
                continue
            seen_users.add(key)
            selected.append(row)
        leftovers.extend(ranked_items[keep:])

    if len(selected) < sample_size:
        ranked_leftovers = sorted(
            leftovers,
            key=lambda row: row_informativeness(row) + rng.random() * 0.010,
            reverse=True,
        )
        for row in ranked_leftovers:
            key = row.user.lower()
            if key in seen_users:
                continue
            seen_users.add(key)
            selected.append(row)
            if len(selected) >= sample_size:
                break

    rng.shuffle(selected)
    return selected[:sample_size]


def score_candidate(row: PromptRow, candidate: str, teacher: str) -> Dict[str, float]:
    candidate = normalize_response(candidate)
    reference = normalize_response(row.assistant)
    if not candidate:
        return {"token_f1": 0.0, "char_similarity": 0.0, "penalty": 1.0, "composite": -1.0}
    f1 = float(token_f1(reference, candidate))
    similarity = float(SequenceMatcher(None, reference.lower(), candidate.lower()).ratio())
    penalty = float(candidate_penalty(reference, candidate))
    word_count = len(candidate.split())
    length_bonus = 0.02 if 4 <= word_count <= 112 else 0.0
    if infer_budget(row) == "short" and word_count <= 32:
        length_bonus += 0.03
    if infer_task_mode(row) == "creative" and any(tok in candidate.lower() for tok in ("imagine", "like", "as if", "vivid", "scene")):
        length_bonus += 0.02
    composite = 0.60 * f1 + 0.32 * similarity + dynamic_teacher_prior(row, teacher) + length_bonus - penalty
    return {
        "token_f1": f1,
        "char_similarity": similarity,
        "penalty": penalty,
        "composite": float(composite),
    }


def make_paper_prompt_rows() -> List[PromptRow]:
    rows: List[PromptRow] = []
    for note in PAPER_NOTES:
        title = str(note["title"])
        research_tags = [title, "frontier", "latest"]
        rows.extend(
            [
                PromptRow(
                    user=f"What is {title} and why does it matter for small local models?",
                    assistant=str(note["summary"]),
                    source=f"frontier_paper_{note['slug']}",
                    metadata={"knowledge_recency": "latest", "source_quality": "research", "research_tags": research_tags + ["summary"]},
                ),
                PromptRow(
                    user=f"How would you adapt the core idea of {title} to a compact Supermix-style model?",
                    assistant=str(note["takeaway"]),
                    source=f"frontier_paper_{note['slug']}",
                    metadata={"knowledge_recency": "latest", "source_quality": "research", "research_tags": research_tags + ["adaptation"]},
                ),
                PromptRow(
                    user=f"What is the main tradeoff of {title}?",
                    assistant=str(note["tradeoff"]),
                    source=f"frontier_paper_{note['slug']}",
                    metadata={"knowledge_recency": "latest", "source_quality": "research", "research_tags": research_tags + ["tradeoff"]},
                ),
                PromptRow(
                    user=f"Explain {title} with a vivid analogy for an engineer.",
                    assistant=str(note["analogy"]),
                    source=f"frontier_paper_{note['slug']}",
                    metadata={
                        "knowledge_recency": "latest",
                        "source_quality": "research",
                        "task_mode": "creative",
                        "creativity_level": "vivid",
                        "research_tags": research_tags + ["analogy"],
                    },
                ),
            ]
        )
    return rows


def compress_answer(text: str) -> str:
    text = normalize_response(text)
    if not text:
        return ""
    parts = []
    for raw in text.replace("?", ".").replace("!", ".").split("."):
        cleaned = raw.strip()
        if cleaned:
            parts.append(cleaned)
    kept = ". ".join(parts[:2]).strip()
    if kept and not kept.endswith("."):
        kept += "."
    words = kept.split()
    if len(words) > 52:
        kept = " ".join(words[:52]).rstrip(",;:") + "."
    return normalize_response(kept)


def build_polish_rows(rows: Sequence[PromptRow], *, limit: int, seed: int, recipe_tag: str) -> List[PromptRow]:
    if limit <= 0 or not rows:
        return []
    rng = random.Random(seed)
    selected = list(rows)
    rng.shuffle(selected)
    out: List[PromptRow] = []
    for row in selected[: min(limit, len(selected))]:
        mode = infer_task_mode(row)
        assistant = row.assistant if mode == "creative" else (compress_answer(row.assistant) or row.assistant)
        metadata = dict(row.metadata)
        metadata.update(
            {
                "task_mode": mode,
                "reasoning_budget": "short" if mode != "creative" else "medium",
                "creativity_level": infer_creativity_level(row),
                "knowledge_recency": infer_recency(row),
                "source_quality": infer_source_quality(row),
                "research_tags": list(metadata.get("research_tags") or [])[:8] + ["polish", mode],
            }
        )
        out.append(
            PromptRow(
                user=row.user,
                assistant=assistant,
                source=f"{recipe_tag}_polish_{row.source}",
                metadata=metadata,
            )
        )
    return out


def build_self_repair_rows(
    repair_candidates: Sequence[Dict[str, object]],
    *,
    limit: int,
    seed: int,
    recipe_tag: str,
) -> List[PromptRow]:
    if limit <= 0 or not repair_candidates:
        return []

    rng = random.Random(seed)
    ranked = sorted(
        repair_candidates,
        key=lambda item: (float(item.get("challenge", 0.0)) + rng.random() * 0.01),
        reverse=True,
    )

    out: List[PromptRow] = []
    seen_prompts: set[str] = set()
    for item in ranked:
        row = item.get("row")
        if not isinstance(row, PromptRow):
            continue
        mode = infer_task_mode(row)
        if mode == "creative":
            continue
        draft = normalize_response(str(item.get("draft") or ""))
        if not draft or draft.lower() == row.assistant.lower():
            continue

        if mode == "coding":
            repair_instruction = "Revise the draft. Fix any logic, syntax, or grounding problems. Return only the corrected final answer."
        elif mode == "knowledge":
            repair_instruction = "Revise the draft. Fix any stale or incorrect claims and return only the corrected final answer."
        else:
            repair_instruction = "Revise the draft. Fix the reasoning mistakes and return only the corrected final answer."

        user_prompt = (
            f"User request:\n{row.user}\n\n"
            f"Draft answer:\n{draft}\n\n"
            f"{repair_instruction}"
        )
        key = user_prompt.lower()
        if key in seen_prompts:
            continue
        seen_prompts.add(key)

        metadata = dict(row.metadata)
        metadata.update(
            {
                "task_mode": mode,
                "reasoning_budget": "medium" if mode != "coding" else "deep",
                "creativity_level": infer_creativity_level(row),
                "knowledge_recency": infer_recency(row),
                "source_quality": infer_source_quality(row),
                "teacher_choice": str(item.get("chosen_from") or "reference"),
                "repair_source": row.source,
                "repair_kind": "draft_correction",
                "research_tags": list(metadata.get("research_tags") or [])[:8] + ["self_repair", mode],
            }
        )
        out.append(
            PromptRow(
                user=user_prompt,
                assistant=row.assistant,
                source=f"{recipe_tag}_self_repair_{metadata['teacher_choice']}",
                metadata=metadata,
            )
        )
        if len(out) >= limit:
            break
    return out


def build_messages_row(
    row: PromptRow,
    *,
    assistant_text: Optional[str] = None,
    chosen_from: str,
    source_name: Optional[str] = None,
) -> Dict[str, object]:
    task_mode = str(row.metadata.get("task_mode") or infer_task_mode(row))
    reasoning_budget = str(row.metadata.get("reasoning_budget") or infer_budget(row))
    creativity_level = str(row.metadata.get("creativity_level") or infer_creativity_level(row))
    knowledge_recency = str(row.metadata.get("knowledge_recency") or infer_recency(row))
    source_quality = str(row.metadata.get("source_quality") or infer_source_quality(row))
    research_tags = row.metadata.get("research_tags") or []
    if not isinstance(research_tags, list):
        research_tags = [str(research_tags)] if str(research_tags).strip() else []
    tags = [task_mode, reasoning_budget, creativity_level, knowledge_recency, source_quality]
    if chosen_from and chosen_from != "reference":
        tags.append(chosen_from)

    assistant = normalize_response(assistant_text if assistant_text is not None else row.assistant)
    metadata = dict(row.metadata)
    metadata.update(
        {
            "teacher_choice": chosen_from,
            "reference_source": row.source,
            "task_mode": task_mode,
            "reasoning_budget": reasoning_budget,
            "creativity_level": creativity_level,
            "knowledge_recency": knowledge_recency,
            "source_quality": source_quality,
            "research_tags": research_tags[:8],
        }
    )
    return {
        "messages": [
            {"role": "system", "content": f"conversation_tags={','.join(sorted(set(tags)))}"},
            {"role": "system", "content": f"reasoning_budget={reasoning_budget}"},
            {"role": "system", "content": f"creativity_level={creativity_level}"},
            {"role": "system", "content": f"knowledge_recency={knowledge_recency}"},
            {"role": "system", "content": f"source_quality={source_quality}"},
            {"role": "system", "content": f"task_mode={task_mode}"},
            {"role": "system", "content": f"research_tags={','.join(str(tag) for tag in research_tags[:8])}"},
            {"role": "user", "content": row.user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source_name or row.source,
        "metadata": metadata,
    }


def write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_unique_json_rows(output: List[Dict[str, object]], seen: set[tuple[str, str, str]], rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        messages = row.get("messages") or []
        user_text = ""
        assistant_text = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content
        key = (user_text.lower(), assistant_text.lower(), str(row.get("source", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        output.append(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frontier dataset with latest-paper controls and multi-teacher distillation.")
    parser.add_argument("--prompt_data", nargs="+", required=True, help="Input JSONL prompt/answer files.")
    parser.add_argument("--eval_data", nargs="*", default=(), help="Optional eval-only JSONL files.")
    parser.add_argument("--distill_output", required=True)
    parser.add_argument("--stage2_train_output", required=True)
    parser.add_argument("--stage2_eval_output", required=True)
    parser.add_argument("--stage3_polish_output", required=True)
    parser.add_argument("--summary_out", required=True)
    parser.add_argument("--audit_out", default=None)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--qwen_v28_adapter", required=True)
    parser.add_argument("--skip_qwen_base", action="store_true")
    parser.add_argument("--v32_weights", required=True)
    parser.add_argument("--v32_meta", required=True)
    parser.add_argument("--v33_stage1_weights", default="")
    parser.add_argument("--v33_stage1_meta", default="")
    parser.add_argument("--v33_stage2_weights", default="")
    parser.add_argument("--v33_stage2_meta", default="")
    parser.add_argument("--hybrid_weights", required=True)
    parser.add_argument("--hybrid_meta", required=True)
    parser.add_argument("--lite_weights", required=True)
    parser.add_argument("--lite_meta", required=True)
    parser.add_argument("--champion_v28_weights", default="")
    parser.add_argument("--champion_v28_meta", default="")
    parser.add_argument("--skip_champion_v28", action="store_true")
    parser.add_argument("--extra_qwen_teacher", action="append", default=[], help="Additional Qwen teacher as NAME=ADAPTER_PATH.")
    parser.add_argument("--extra_champion_teacher", action="append", default=[], help="Additional Champion teacher as NAME=WEIGHTS_PATH::META_PATH.")
    parser.add_argument("--sample_size", type=int, default=960)
    parser.add_argument("--stage2_reference_cap", type=int, default=2200)
    parser.add_argument("--eval_fraction", type=float, default=0.14)
    parser.add_argument("--eval_cap", type=int, default=160)
    parser.add_argument("--paper_eval_fraction", type=float, default=0.20)
    parser.add_argument("--paper_eval_cap", type=int, default=20)
    parser.add_argument("--polish_limit", type=int, default=288)
    parser.add_argument("--self_repair_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=112)
    parser.add_argument("--log_every", type=int, default=32)
    parser.add_argument("--min_teacher_score", type=float, default=0.14)
    parser.add_argument("--recipe_tag", default="v33_frontier")
    parser.add_argument("--summary_note", default="")
    args = parser.parse_args()

    prompt_paths = [Path(path).resolve() for path in args.prompt_data]
    eval_paths = [Path(path).resolve() for path in args.eval_data]
    distill_output = Path(args.distill_output).resolve()
    stage2_train_output = Path(args.stage2_train_output).resolve()
    stage2_eval_output = Path(args.stage2_eval_output).resolve()
    stage3_polish_output = Path(args.stage3_polish_output).resolve()
    summary_out = Path(args.summary_out).resolve()
    audit_out = Path(args.audit_out).resolve() if args.audit_out else None

    raw_rows = load_prompt_rows_safe(prompt_paths)
    extra_eval_rows = load_prompt_rows_safe(eval_paths)
    paper_rows = make_paper_prompt_rows()
    if not raw_rows:
        raise SystemExit("No eligible prompt rows were loaded.")

    train_rows, heldout_rows = split_holdout_rows(
        raw_rows,
        eval_fraction=float(args.eval_fraction),
        eval_cap=int(args.eval_cap),
        seed=int(args.seed),
    )
    paper_train_rows, paper_eval_rows = split_holdout_rows(
        paper_rows,
        eval_fraction=float(args.paper_eval_fraction),
        eval_cap=int(args.paper_eval_cap),
        seed=int(args.seed) + 17,
    )

    eval_rows = merge_unique_rows(extra_eval_rows, heldout_rows, paper_eval_rows)
    eval_user_keys = {row.user.lower() for row in eval_rows}
    train_rows = [row for row in merge_unique_rows(train_rows, paper_train_rows) if row.user.lower() not in eval_user_keys]
    distill_rows = stratified_sample(train_rows, sample_size=int(args.sample_size), seed=int(args.seed))

    print(
        f"[{args.recipe_tag}] loaded={len(raw_rows)} paper={len(paper_rows)} "
        f"train={len(train_rows)} eval={len(eval_rows)} sampled={len(distill_rows)}"
    )

    qwen_teacher_specs: List[Tuple[str, Optional[Path]]] = []
    if not args.skip_qwen_base:
        qwen_teacher_specs.append(("qwen_base", None))
    qwen_teacher_specs.append(("qwen_v28", Path(args.qwen_v28_adapter).resolve()))
    for spec in args.extra_qwen_teacher:
        name, adapter_path = parse_named_path_spec(spec)
        if adapter_path.exists():
            qwen_teacher_specs.append((name, adapter_path))

    qwen_teacher_outputs: Dict[str, List[str]] = {}
    for teacher_name, adapter_path in qwen_teacher_specs:
        qwen_teacher_outputs[teacher_name] = generate_qwen_outputs(
            distill_rows,
            base_model=str(Path(args.base_model).resolve()),
            adapter_dir=adapter_path,
            device=args.device,
            label=teacher_name,
            max_new_tokens=int(args.max_new_tokens),
            log_every=int(args.log_every),
        )

    champion_teacher_specs: List[Tuple[str, Path, Path]] = [
        ("v32_latest", Path(args.v32_weights).resolve(), Path(args.v32_meta).resolve()),
        ("hybrid_v31", Path(args.hybrid_weights).resolve(), Path(args.hybrid_meta).resolve()),
        ("lite_v30", Path(args.lite_weights).resolve(), Path(args.lite_meta).resolve()),
    ]
    if args.v33_stage1_weights and args.v33_stage1_meta:
        v33_stage1_weights = Path(args.v33_stage1_weights).resolve()
        v33_stage1_meta = Path(args.v33_stage1_meta).resolve()
        if v33_stage1_weights.exists() and v33_stage1_meta.exists():
            champion_teacher_specs.append(("v33_stage1", v33_stage1_weights, v33_stage1_meta))
    if args.v33_stage2_weights and args.v33_stage2_meta:
        v33_stage2_weights = Path(args.v33_stage2_weights).resolve()
        v33_stage2_meta = Path(args.v33_stage2_meta).resolve()
        if v33_stage2_weights.exists() and v33_stage2_meta.exists():
            champion_teacher_specs.append(("v33_stage2_best", v33_stage2_weights, v33_stage2_meta))
    if not args.skip_champion_v28 and args.champion_v28_weights and args.champion_v28_meta:
        classic_weights = Path(args.champion_v28_weights).resolve()
        classic_meta = Path(args.champion_v28_meta).resolve()
        if classic_weights.exists() and classic_meta.exists():
            champion_teacher_specs.append(("champion_v28", classic_weights, classic_meta))
    for spec in args.extra_champion_teacher:
        name, weights_path, meta_path = parse_named_pair_spec(spec)
        if weights_path.exists() and meta_path.exists():
            champion_teacher_specs.append((name, weights_path, meta_path))

    champion_teacher_outputs: Dict[str, List[str]] = {}
    for teacher_name, weights_path, meta_path in champion_teacher_specs:
        champion_teacher_outputs[teacher_name] = generate_champion_outputs(
            distill_rows,
            weights_path=str(weights_path),
            meta_path=str(meta_path),
            device=args.device,
            label=teacher_name,
            log_every=int(args.log_every),
        )

    distilled_prompt_rows: List[PromptRow] = []
    distilled_json_rows: List[Dict[str, object]] = []
    audit_json_rows: List[Dict[str, object]] = []
    repair_candidates: List[Dict[str, object]] = []
    teacher_choice_counts: Counter[str] = Counter()
    prompt_source_counts: Counter[str] = Counter()
    teacher_mean_composite: Counter[str] = Counter()
    teacher_mean_count: Counter[str] = Counter()

    for idx, row in enumerate(distill_rows):
        candidates: Dict[str, str] = {}
        for teacher_name, outputs in qwen_teacher_outputs.items():
            if idx < len(outputs):
                candidate = normalize_response(str(outputs[idx]))
                if candidate:
                    candidates[teacher_name] = candidate
        for teacher_name, outputs in champion_teacher_outputs.items():
            if idx < len(outputs):
                candidate = normalize_response(str(outputs[idx]))
                if candidate:
                    candidates[teacher_name] = candidate

        scored: Dict[str, Dict[str, float]] = {}
        for teacher_name, candidate in candidates.items():
            metrics = score_candidate(row, str(candidate), teacher_name)
            metrics["text"] = normalize_response(str(candidate))
            scored[teacher_name] = metrics

        best_teacher, best_metrics = max(scored.items(), key=lambda item: float(item[1]["composite"]))
        use_reference = (
            float(best_metrics["composite"]) < float(args.min_teacher_score)
            or (float(best_metrics["token_f1"]) < 0.05 and float(best_metrics["char_similarity"]) < 0.15)
            or float(best_metrics["penalty"]) > 0.55
        )
        if use_reference:
            chosen_text = row.assistant
            chosen_from = "reference"
        else:
            chosen_text = normalize_response(str(best_metrics["text"]))
            chosen_from = best_teacher
            teacher_mean_composite[best_teacher] += float(best_metrics["composite"])
            teacher_mean_count[best_teacher] += 1

        prompt_source_counts[row.source] += 1
        teacher_choice_counts[chosen_from] += 1

        distilled_row = PromptRow(
            user=row.user,
            assistant=chosen_text,
            source=f"{args.recipe_tag}_distill_{chosen_from}",
            metadata={
                **dict(row.metadata),
                "teacher_choice": chosen_from,
                "reference_source": row.source,
                "task_mode": infer_task_mode(row),
                "reasoning_budget": infer_budget(row),
                "creativity_level": infer_creativity_level(row),
                "knowledge_recency": infer_recency(row),
                "source_quality": infer_source_quality(row),
                "research_tags": list(row.metadata.get("research_tags") or []),
            },
        )
        distilled_prompt_rows.append(distilled_row)
        distilled_json_rows.append(
            build_messages_row(
                distilled_row,
                assistant_text=chosen_text,
                chosen_from=chosen_from,
                source_name=distilled_row.source,
            )
        )

        best_text = normalize_response(str(best_metrics["text"]))
        challenge = max(
            0.0,
            (1.0 - float(best_metrics["token_f1"]))
            + 0.55 * (1.0 - float(best_metrics["char_similarity"]))
            + 0.60 * float(best_metrics["penalty"]),
        )
        if best_text and best_text.lower() != row.assistant.lower():
            repair_candidates.append(
                {
                    "row": row,
                    "draft": best_text,
                    "chosen_from": chosen_from if chosen_from != "reference" else best_teacher,
                    "challenge": challenge,
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

    stage2_reference_rows = select_reference_rows(
        train_rows,
        sample_size=int(args.stage2_reference_cap),
        seed=int(args.seed) + 7,
    )
    self_repair_prompt_rows = build_self_repair_rows(
        repair_candidates,
        limit=int(args.self_repair_limit),
        seed=int(args.seed) + 23,
        recipe_tag=str(args.recipe_tag),
    )
    raw_train_rows = [build_messages_row(row, chosen_from="reference", source_name=row.source) for row in stage2_reference_rows]
    self_repair_json_rows = [build_messages_row(row, chosen_from=str(row.metadata.get("teacher_choice", "reference")), source_name=row.source) for row in self_repair_prompt_rows]
    stage2_train_rows: List[Dict[str, object]] = []
    seen_stage2: set[tuple[str, str, str]] = set()
    append_unique_json_rows(stage2_train_rows, seen_stage2, raw_train_rows)
    append_unique_json_rows(stage2_train_rows, seen_stage2, distilled_json_rows)
    append_unique_json_rows(stage2_train_rows, seen_stage2, self_repair_json_rows)

    stage2_eval_rows = [build_messages_row(row, chosen_from="reference", source_name=row.source) for row in eval_rows]

    polish_pool = list(train_rows) + distilled_prompt_rows + paper_train_rows
    polish_rows = build_polish_rows(
        polish_pool,
        limit=int(args.polish_limit),
        seed=int(args.seed) + 31,
        recipe_tag=str(args.recipe_tag),
    )
    stage3_polish_rows = [build_messages_row(row, chosen_from=str(row.metadata.get("teacher_choice", "reference")), source_name=row.source) for row in polish_rows]

    write_jsonl(distill_output, distilled_json_rows)
    write_jsonl(stage2_train_output, stage2_train_rows)
    write_jsonl(stage2_eval_output, stage2_eval_rows)
    write_jsonl(stage3_polish_output, stage3_polish_rows)
    if audit_out is not None:
        write_jsonl(audit_out, audit_json_rows)

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_files": [str(path) for path in prompt_paths],
        "eval_files": [str(path) for path in eval_paths],
        "rows_loaded": len(raw_rows),
        "paper_rows": len(paper_rows),
        "paper_train_rows": len(paper_train_rows),
        "paper_eval_rows": len(paper_eval_rows),
        "train_rows": len(train_rows),
        "stage2_reference_rows": len(stage2_reference_rows),
        "eval_rows": len(eval_rows),
        "distill_rows": len(distilled_json_rows),
        "stage2_train_rows": len(stage2_train_rows),
        "stage2_eval_rows": len(stage2_eval_rows),
        "self_repair_rows": len(self_repair_json_rows),
        "stage3_polish_rows": len(stage3_polish_rows),
        "teacher_selection_counts": dict(teacher_choice_counts),
        "prompt_source_counts": dict(prompt_source_counts),
        "teacher_mean_composite": {
            teacher: float(teacher_mean_composite[teacher] / max(1, teacher_mean_count[teacher]))
            for teacher in teacher_mean_count
        },
        "teacher_prior": dict(BASE_TEACHER_PRIOR),
        "configured_qwen_teachers": [name for name, _ in qwen_teacher_specs],
        "configured_champion_teachers": [name for name, _, _ in champion_teacher_specs],
        "paper_urls": [{key: note[key] for key in ("slug", "title", "url")} for note in PAPER_NOTES],
        "student_note": (
            str(args.summary_note).strip()
            or "This frontier recipe combines adaptive depth reuse, memory-gated reasoning, curated stage-two reference selection, and explicit self-repair rows. "
            "The paper rows are concise synthesis notes, not full-paper reproductions."
        ),
        "recipe_tag": str(args.recipe_tag),
        "device": str(args.device),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{args.recipe_tag}] wrote distill data to {distill_output}")
    print(f"[{args.recipe_tag}] wrote stage2 train data to {stage2_train_output}")
    print(f"[{args.recipe_tag}] wrote stage2 eval data to {stage2_eval_output}")
    print(f"[{args.recipe_tag}] wrote stage3 polish data to {stage3_polish_output}")
    print(f"[{args.recipe_tag}] teacher choices: {dict(teacher_choice_counts)}")
    print(f"[{args.recipe_tag}] wrote summary to {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
