from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

SOURCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SOURCE_DIR.parent
if str(SOURCE_DIR) not in __import__("sys").path:
    __import__("sys").path.append(str(SOURCE_DIR))

from image_recognition_model import SCIENCE_IMAGE_CLASSES
from omni_collective_model import OMNI_DOMAIN_LABELS_V2, OMNI_INTENTS_V2, build_char_vocab
from omni_collective_v45_model import OmniCollectiveEngineV45, OmniCollectiveNetV45
from omni_training_runtime import maybe_compile_model, resolve_training_runtime
from prepare_omni_collective_v45 import build_v45_blueprint, latest_v44_summary_path, latest_v44_zip_path, load_summary
from train_omni_collective_v2 import DEFAULT_MODELS_DIR, OmniRow, split_rows
from train_omni_collective_v4 import _load_expanded_state_from_zip
from train_omni_collective_v8 import _train_stage_resumable_v8
from train_omni_collective_v44 import (
    _bundle_rows,
    _code_critique_repair_rows_v41,
    _communication_polish_rows_v41,
    _dedupe_rows,
    _eval_payload,
    _latent_plan_rows_v42,
    _promotion_eval_pack_v42,
    _read_jsonl_rows,
    _reasoning_budget_rows_v43,
    _row,
    _seeded_take,
    _teacher_disagreement_rows_v42,
    _write_json,
    _write_json_atomic_v42,
    _write_jsonl,
    build_training_rows_v44,
    build_training_rows_v44_dry_run,
    )


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "omni_collective_v45_prep"
DEFAULT_TRAIN_OUTPUT_DIR = REPO_ROOT / "output"
V45_MODEL_CONFIG: Dict[str, Any] = {
    "family": "omni_collective_v45",
    "base_model_family": "omni_collective_v44",
    "text_hidden": 320,       # V45: Widened for richer internal representations
    "fusion_hidden": 1280,    # V45: Scaled fusion layer for better cross-modal blending
    "expert_count": 16,       # V45: 2 additional experts for PRM and adversarial reasoning
    "expert_hidden": 2048,
    "expert_top_k": 3,        # V45: Route to 3 experts per token (finer-grained MoE)
    "deliberation_passes": 18,  # V45: Dynamic — this is the max; actual is difficulty-adaptive
    "minimum_passes": 3,      # V45: Lower minimum since SpecCoT can exit at pass 1
    "speculative_threshold": 0.82,  # V45: Confidence cutoff for speculative early-exit
    "new_heads": [
    "budget_router_head",
    "teacher_consensus_head",
    "cache_budget_head",
    "verifier_gate_head",
    "process_reward_head",      # V45: Scores reasoning step quality
    "adversarial_verifier_head", # V45: Self-play challenge scoring
    "difficulty_estimator_head", # V45: Adaptive compute allocation
    ],
}


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


def _benchmark_bridge_rows_v44(*, seed: int, limit: int = 220) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []

    route_rows = [
    (
        "Route first, then answer.\n"
        "Request: Which local model should handle a common benchmark reasoning prompt if I care about exact score first?\n"
        "Return the final answer only.",
        "Use v42_benchmax first for benchmark-heavy exact reasoning, then let the broader omni model handle follow-up explanation or multimodal context.",
        "model_selection",
        "benchmark_bridge_v45::route_selection",
    ),
    (
        "Answer like a benchmark-focused reviewer.\n"
        "Question: Why should v45 inherit v42 hard-example replay instead of relying only on v44 communication polish?",
        "Because v44 improved presentation and routing, but v42 still wins on raw benchmark exactness. Hard-example replay keeps the failure patterns that matter for benchmark tasks instead of training only on better-looking answers.",
        "reasoning",
        "benchmark_bridge_v45::bridge_rationale",
    ),
    (
        "Two local lines disagree.\n"
        "Question: v42_benchmax gives a terse exact answer, while v44 gives a broader answer with extra explanation. What should v45 learn?",
        "v45 should keep the exact benchmark answer from the benchmark specialist, then add only the minimum extra explanation that does not change correctness.",
        "reasoning",
        "benchmark_bridge_v45::disagreement_resolution",
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
    ]
    for benchmark_name, strategy in benchmark_buckets:
    prompt = (
        "Give a compact training answer for benchmark behavior.\n"
        f"Question: How should v45 approach {benchmark_name} style questions?\n"
        "Return the final answer only."
    )
    response = f"For {benchmark_name} tasks, {strategy}."
    rows.append(_row(prompt, response, intent="comparison", domain="reasoning", source=f"benchmark_bridge_v45::{benchmark_name.lower()}"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _teacher_role_rows_v44(*, seed: int, limit: int = 260) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    teacher_rows = [
    (
        "Which teacher should supervise a grounded image-and-text explanation task?",
        "Prefer google/gemma-4-31B-it for grounded multimodal explanation slices, then keep only answers that stay concrete and evidence-aware.",
        "model_selection",
        "teacher_roles_v45::gemma4_grounded",
    ),
    (
        "Which teacher should supervise a long-context reasoning and tool-use example?",
        "Prefer Qwen/Qwen3.5-397B-A17B because it is the strongest current teacher in this mix for long-context reasoning, tool use, and route-aware problem solving.",
        "model_selection",
        "teacher_roles_v45::qwen35_long_context",
    ),
    (
        "Which teacher should supervise a failing-test to patch-repair example?",
        "Prefer Qwen/Qwen3-Coder-Next for patch, test-repair, and repo-grounded coding trajectories, then verify the repair before keeping it.",
        "model_selection",
        "teacher_roles_v45::coder_next",
    ),
    (
        "Which teacher should supervise audio, video, or OCR-style route examples?",
        "Prefer Qwen/Qwen3-Omni-30B-A3B-Instruct for audio-video and cross-modal route supervision, then distill only the clean grounded answer form into v45.",
        "model_selection",
        "teacher_roles_v45::qwen_omni",
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
    rows.append(_row(prompt, response, intent="comparison", domain="model_selection", source="teacher_roles_v45::teacher_choice"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _verifier_repair_rows_v44(*, seed: int, limit: int = 220) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    repair_examples = [
    (
        "Repair the answer after a verifier note.\n"
        "Question: Which local model should handle benchmark-focused reasoning prompts?\n"
        "Draft: omni_collective_v44 is always the best choice because it is the newest omni model.\n"
        "Verifier note: The draft ignores that v42_benchmax currently scores higher on the common benchmark suite.",
        "Use v42_benchmax first for benchmark-focused reasoning because it currently outperforms omni_collective_v44 on the saved common benchmark comparison.",
        "teacher_verifier_v45::benchmark_repair",
    ),
    (
        "Repair the answer after a verifier note.\n"
        "Question: Should a long transcript always get the deepest reasoning budget?\n"
        "Draft: Yes, use the deepest budget for every long transcript.\n"
        "Verifier note: The draft overgeneralizes and wastes compute on easy summarization tasks.",
        "No. Use a deeper budget only when the transcript also requires hard reasoning or multi-step verification. Straight summarization should keep a medium budget.",
        "teacher_verifier_v45::budget_repair",
    ),
    (
        "Repair this coding answer after a verifier note.\n"
        "Question: How should a model respond to a failing test after a code change?\n"
        "Draft: Rewrite the whole file and try again.\n"
        "Verifier note: The draft is too destructive and ignores the failing assertion.",
        "Start from the failing assertion, identify the smallest behavior change that caused it, patch that path, and rerun the targeted test before broader edits.",
        "teacher_verifier_v45::coding_repair",
    ),
    (
        "Repair this multimodal answer after a verifier note.\n"
        "Question: The image evidence is weak. How should the answer handle that?\n"
        "Draft: State the most likely object as a fact.\n"
        "Verifier note: The draft hides uncertainty.",
        "Give the most likely reading, state that confidence is limited, and avoid inventing details the image does not support.",
        "teacher_verifier_v45::multimodal_repair",
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
    rows.append(_row(prompt, response, intent="comparison", domain="general", source="teacher_verifier_v45::repair_rule"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _tree_of_thought_rows_v44(*, seed: int, limit: int = 200) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    tot_examples = [
    (
        "Use a Tree-of-Thoughts approach to solve this.\nQuestion: If I have a 3 gallon and 5 gallon jug, how do I measure 4 gallons?",
        (
            "Let's explore possible paths.\n"
            "Path A: Fill 3G, pour to 5G. 5G has 3G. Fill 3G again, pour to 5G. 5G is full, 1G left in 3G.\n"
            "Path B: Fill 5G, pour to 3G. 5G has 2G left.\n"
            "Evaluating... Path B looks faster. Let's trace it completely: "
            "1. Fill 5G. 2. Pour to 3G -> 5G has 2G, 3G has 3G. 3. Empty 3G. "
            "4. Pour the 2G from 5G to 3G -> 5G is empty, 3G has 2G. "
            "5. Fill 5G again. 6. Pour from 5G to 3G until 3G is full. Since 3G has 2G, we pour 1G. "
            "7. 5G started at 5G, poured 1G -> 4G left in 5G. "
            "This path works. Final output: Fill 5G, pour 3G to 3G, empty 3G, pour remainder (2G) to 3G, fill 5G, and fill 3G."
        ),
        "tree_of_thought_v45::math_puzzle",
    ),
    (
        "Your initial route confidence is low. Start a self-reflection pass.\nQuestion: Why did the training script fail at epoch 3 with CUDA Out of Memory, but not epoch 1?",
        (
            "Initial thought: Maybe the model got larger. Wait, weights don't grow during training unless it's dynamic.\n"
            "What changes over epochs? Gradients accumulate, but usually loop resets them. Tensors attached to lists? Yes, if training loss is stored as a tensor with history.\n"
            "Self-correction: The most common cause is keeping `loss` instead of `loss.item()` in the epoch stat tracker, maintaining the entire backward graph in memory.\n"
            "Final answer: You are likely storing the loss tensor with its computation graph instead of taking the float value (`loss.item()`). This causes the memory to accumulate until it crashes in later epochs."
        ),
        "tree_of_thought_v45::code_debug",
    )
    ]
    for prompt, response, source in tot_examples:
    rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _contrastive_hallucination_rows_v44(*, seed: int, limit: int = 150) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    contrastive_examples = [
    (
        "Use Contrastive Decoding against Hallucination.\nRequest: Name the 2024 Nobel Laureate in Computer Science.",
        (
            "Plausible trap: Suggesting a famous computer scientist or saying Geoff Hinton won the Turing Award in 2024.\n"
            "Refusing trap: There is no Nobel Prize in Computer Science (the Turing Award is equivalent, and the Nobel is physics/chemistry/etc.).\n"
            "Final answer: The Nobel Prize does not have a category for Computer Science. The equivalent honor is the ACM A.M. Turing Award."
        ),
        "anti_hallucinate_v45::fake_fact",
    ),
    (
        "Use Contrastive Decoding against Hallucination.\nRequest: Can you provide the source code for the proprietary Google search ranking algorithm?",
        (
            "Plausible trap: Generating a fake Python script with `def pagerank():` and presenting it as the real Google algorithm.\n"
            "Refusing trap: I do not have access to proprietary codebases, nor is the actual modern algorithm entirely public.\n"
            "Final answer: The source code for Google's proprietary search ranking algorithm is not publicly available."
        ),
        "anti_hallucinate_v45::fake_code",
    )
    ]
    for prompt, response, source in contrastive_examples:
    rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# V45 NOVEL: Process Reward Model training data
# ---------------------------------------------------------------------------
def _process_reward_rows_v45(*, seed: int, limit: int = 200) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    prm_examples = [
    (
        "Use step-by-step Process Reward verification.\n"
        "Request: A train leaves city A at 60 mph. Another leaves city B (300 miles away) at 40 mph toward A. When do they meet?\n"
        "Break your reasoning into numbered steps and verify each.",
        (
            "Step 1: Combined closing speed = 60 + 40 = 100 mph. [VERIFY: correct, they approach each other]\n"
            "Step 2: Distance to cover = 300 miles. [VERIFY: given directly]\n"
            "Step 3: Time = 300 / 100 = 3 hours. [VERIFY: arithmetic checks out]\n"
            "Step 4: They meet in 3 hours, 180 miles from A and 120 miles from B. [VERIFY: 60×3=180, 40×3=120, 180+120=300 ✓]\n"
            "Final answer: The trains meet after 3 hours."
        ),
        "process_reward_v45::math_verify_steps",
    ),
    (
        "Use step-by-step Process Reward verification.\n"
        "Request: Debug why this function returns None: def get_max(lst): m = lst[0]; for x in lst: if x > m: m = x\n"
        "Break your reasoning into numbered steps and verify each.",
        (
            "Step 1: The function signature is correct, takes a list. [VERIFY: ok]\n"
            "Step 2: m = lst[0] initializes to first element. [VERIFY: ok, but will crash on empty list]\n"
            "Step 3: Loop compares and updates m. [VERIFY: logic is correct for finding max]\n"
            "Step 4: There is no return statement. [VERIFY: THIS IS THE BUG — Python returns None implicitly]\n"
            "Step 5: Fix: add 'return m' at the end of the function. [VERIFY: this resolves the issue]\n"
            "Final answer: The function is missing a 'return m' statement at the end."
        ),
        "process_reward_v45::code_debug_steps",
    ),
    (
        "Use step-by-step Process Reward verification.\n"
        "Request: Is it safe to use aluminium cookware with acidic foods like tomatoes?\n"
        "Break your reasoning into numbered steps and verify each.",
        (
            "Step 1: Aluminium is reactive with acids. [VERIFY: yes, well-established chemistry]\n"
            "Step 2: Acidic foods (tomatoes, citrus) can cause aluminium to leach into food. [VERIFY: FDA and studies confirm trace leaching]\n"
            "Step 3: The amount leached is generally very small and within safe limits for occasional use. [VERIFY: consensus from health authorities]\n"
            "Step 4: However, prolonged cooking or storage of acidic foods in bare aluminium is discouraged. [VERIFY: consistent recommendation]\n"
            "Final answer: Occasional use is generally safe, but avoid prolonged cooking or storing acidic foods in uncoated aluminium. Anodized aluminium cookware is a safer alternative."
        ),
        "process_reward_v45::science_verify_steps",
    ),
    ]
    for prompt, response, source in prm_examples:
    rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# V45 NOVEL: GRPO Multi-Candidate generation training data
# ---------------------------------------------------------------------------
def _grpo_multi_candidate_rows_v45(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    grpo_examples = [
    (
        "Generate three candidate answers internally using different approaches.\n"
        "Request: What causes the seasons on Earth?\n"
        "Select the candidate with the strongest evidence chain.",
        (
            "Candidate 1 (First principles): Earth's axial tilt of 23.5° means different hemispheres receive more direct sunlight at different times of year. More direct sunlight = more energy per unit area = warmer.\n"
            "Candidate 2 (Elimination): It's NOT distance from the Sun (Earth is actually closer in Northern Hemisphere winter). It's NOT the length of day alone. It IS the angle of incidence of sunlight due to axial tilt.\n"
            "Candidate 3 (Analogy): Like a flashlight — shine it straight down and you get a bright spot; tilt it and the same light spreads over a larger area, dimmer.\n"
            "Selection: Candidate 1 is most complete and directly explains the mechanism. Candidate 2 helpfully eliminates a common misconception. Combining: Seasons are caused by Earth's 23.5° axial tilt, which changes the angle and duration of sunlight each hemisphere receives throughout the year."
        ),
        "grpo_candidates_v45::science_multi",
    ),
    (
        "Generate three candidate answers internally using different approaches.\n"
        "Request: Should I use a SQL or NoSQL database for a social media app?\n"
        "Select the candidate with the strongest evidence chain.",
        (
            "Candidate 1 (First principles): Social media has varied data shapes (posts, likes, follows, messages) and needs horizontal scaling. NoSQL fits better for flexible schemas and scale.\n"
            "Candidate 2 (Elimination): SQL requires predefined schemas — bad for evolving features. SQL struggles with horizontal scaling at social-media volume. NoSQL handles both.\n"
            "Candidate 3 (Analogy): Instagram and Twitter both started SQL and migrated to mixed/NoSQL at scale. The pattern suggests starting with what you know, but planning for NoSQL.\n"
            "Selection: Candidate 1 gives the clearest reasoning. Final answer: Use NoSQL (like MongoDB or Cassandra) for the primary data store due to flexible schemas and horizontal scalability, but consider SQL for structured transactional data like payments."
        ),
        "grpo_candidates_v45::engineering_multi",
    ),
    ]
    for prompt, response, source in grpo_examples:
    rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# V45 NOVEL: Self-Play Adversarial Verification training data
# ---------------------------------------------------------------------------
def _adversarial_selfplay_rows_v45(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    adversarial_examples = [
    (
        "You are a strict adversarial verifier. Your job is to find flaws.\n"
        "Original request: What is the time complexity of binary search?\n"
        "Proposed answer: O(n) because you look at every element.\n"
        "Identify the single most critical flaw.",
        (
            "CRITICAL FLAW: The proposed answer is factually wrong. Binary search has O(log n) time complexity, not O(n). "
            "Binary search eliminates half the search space on each comparison, which is the defining property that gives it logarithmic performance. "
            "O(n) describes linear search, not binary search. The answer must be corrected to O(log n)."
        ),
        "adversarial_selfplay_v45::wrong_complexity",
    ),
    (
        "You are a strict adversarial verifier. Your job is to find flaws.\n"
        "Original request: Is Python interpreted or compiled?\n"
        "Proposed answer: Python is a purely interpreted language.\n"
        "Identify the single most critical flaw.",
        (
            "FLAW: The answer oversimplifies. CPython (the reference implementation) first compiles Python source code to bytecode (.pyc files), "
            "then the bytecode is interpreted by the Python Virtual Machine (PVM). So Python is both compiled (to bytecode) and interpreted (bytecode execution). "
            "Additionally, implementations like PyPy use JIT compilation. The answer should say 'Python is typically compiled to bytecode and then interpreted' rather than 'purely interpreted'."
        ),
        "adversarial_selfplay_v45::oversimplification",
    ),
    (
        "You are a strict adversarial verifier. Your job is to find flaws.\n"
        "Original request: How far is the Moon from Earth?\n"
        "Proposed answer: The Moon is approximately 384,400 km (238,855 miles) from Earth on average.\n"
        "Identify the single most critical flaw.",
        (
            "VERIFIED CORRECT: The proposed answer is accurate. The average Earth-Moon distance (semi-major axis of the lunar orbit) is approximately 384,400 km. "
            "The answer correctly notes this is an average, which is important since the actual distance varies from about 356,500 km (perigee) to 406,700 km (apogee). "
            "No flaws found. High confidence in this answer."
        ),
        "adversarial_selfplay_v45::correct_verification",
    ),
    ]
    for prompt, response, source in adversarial_examples:
    rows.append(_row(prompt, response, intent="general", domain="reasoning", source=source))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _turboquant_budget_rows_v44(*, seed: int, limit: int = 180) -> Tuple[List[OmniRow], Dict[str, int]]:
    rows: List[OmniRow] = []
    budget_examples = [
    (
        "Choose the reasoning and evidence budget.\n"
        "Request: Summarize a very long build log into the one failing step and the next fix.\n"
        "Return the final answer only.",
        "Use a medium reasoning budget and a compressed evidence set. Keep only the failing step, the relevant stack trace line, and the next concrete action.",
        "turboquant_budget_v45::compressed_log_summary",
    ),
    (
        "Choose the reasoning and evidence budget.\n"
        "Request: Solve a GSM8K-style arithmetic problem.\n"
        "Return the final answer only.",
        "Use a deep reasoning budget for the calculation itself, but keep the final answer concise and verify the arithmetic before stopping.",
        "turboquant_budget_v45::math_budget",
    ),
    (
        "Choose the reasoning and evidence budget.\n"
        "Request: Compare two model options for a coding task.\n"
        "Return the final answer only.",
        "Use a medium budget: identify the task, compare the main tradeoff, and recommend one model without overexplaining.",
        "turboquant_budget_v45::model_compare_budget",
    ),
    (
        "Answer with context-economy discipline.\n"
        "Question: What should v45 learn from TurboQuant without pretending to train quantization itself?",
        "v45 should learn evidence selection and budget control. TurboQuant mainly makes long teacher and verifier runs cheaper; the trainable behavior is when to use more or less context, not how to quantize weights.",
        "turboquant_budget_v45::training_translation",
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
    rows.append(_row(prompt, response, intent="comparison", domain="reasoning", source="turboquant_budget_v45::evidence_rule"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _diversity_rows_v44(*, seed: int, limit: int = 260) -> Tuple[List[OmniRow], Dict[str, int]]:
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
        response = f"{guidance} Tailor the wording to {persona} while staying concise."
        rows.append(_row(prompt, response, intent="general", domain=domain, source="diversity_mix_v45::persona_style"))

    selected = _seeded_take(rows, seed=seed, limit=limit)
    counts: Dict[str, int] = {}
    for row in selected:
    counts[row.source] = counts.get(row.source, 0) + 1
    return selected, dict(sorted(counts.items()))


def _promotion_eval_pack_v42() -> List[Dict[str, str]]:
    evals = list(_promotion_eval_pack_v42())
    evals.extend(
    [
        _eval_payload(
            "Which local model should handle a benchmark-style reasoning prompt if exact score is the main goal?",
            expected="v42_benchmax",
            focus="benchmark_bridge",
            metric="contains",
            source="promotion_eval_v45::benchmark_route",
        ),
        _eval_payload(
            "Translate TurboQuant into a v45 training implication in one sentence.",
            expected="budget control and evidence compression for longer teacher traces, not quantization as a direct training target",
            focus="budget_reasoning",
            metric="contains",
            source="promotion_eval_v45::turboquant_translation",
        ),
        _eval_payload(
            "Which teacher is the best fit for failing-test and patch-repair traces?",
            expected="Qwen/Qwen3-Coder-Next",
            focus="teacher_roles",
            metric="contains",
            source="promotion_eval_v45::coder_teacher",
        ),
        _eval_payload(
            "A large transcript only needs the one failing step and the next action. Which reasoning budget should v45 prefer?",
            expected="medium",
            focus="budget_reasoning",
            metric="contains",
            source="promotion_eval_v45::budget_choice",
        ),
    ]
    )
    return evals


def assemble_v45_training_rows(
    *,
    base_stage1_rows: Sequence[OmniRow],
    base_full_rows: Sequence[OmniRow],
    base_summary: Dict[str, Any],
    prep_root: Path,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    prep_root = prep_root.resolve()
    benchmark_rows = _read_jsonl_rows(prep_root / "v45_benchmark_bridge_rows.jsonl")
    teacher_rows = _read_jsonl_rows(prep_root / "v45_teacher_role_rows.jsonl")
    verifier_rows = _read_jsonl_rows(prep_root / "v45_verifier_repair_rows.jsonl")
    budget_rows = _read_jsonl_rows(prep_root / "v45_turboquant_budget_rows.jsonl")
    diversity_rows = _read_jsonl_rows(prep_root / "v45_diversity_rows.jsonl")

    tree_of_thought_rows = _read_jsonl_rows(prep_root / "v45_tree_of_thought_rows.jsonl")
    contrastive_rows = _read_jsonl_rows(prep_root / "v45_contrastive_rows.jsonl")
    prm_rows = _read_jsonl_rows(prep_root / "v45_process_reward_rows.jsonl")
    grpo_rows = _read_jsonl_rows(prep_root / "v45_grpo_multi_candidate_rows.jsonl")
    adversarial_rows = _read_jsonl_rows(prep_root / "v45_adversarial_selfplay_rows.jsonl")

    added_rows = (list(benchmark_rows) + list(teacher_rows) + list(verifier_rows) + list(budget_rows)
              + list(diversity_rows) + list(tree_of_thought_rows) + list(contrastive_rows)
              + list(prm_rows) + list(grpo_rows) + list(adversarial_rows))
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
    "v45_added_rows": len(added_rows),
    "v45_prep_root": str(prep_root),
    "v45_row_groups": {
        "benchmark_bridge": _bundle_rows(benchmark_rows),
        "teacher_roles": _bundle_rows(teacher_rows),
        "verifier_repair": _bundle_rows(verifier_rows),
        "turboquant_budget": _bundle_rows(budget_rows),
        "diversity_mix": _bundle_rows(diversity_rows),
        "tree_of_thought": _bundle_rows(tree_of_thought_rows),
        "contrastive_hallucination": _bundle_rows(contrastive_rows),
        "process_reward": _bundle_rows(prm_rows),
        "grpo_multi_candidate": _bundle_rows(grpo_rows),
        "adversarial_selfplay": _bundle_rows(adversarial_rows),
    },
    "base_summary": base_summary,
    }
    return stage1_rows, full_rows, summary


def build_training_rows_v45(
    *,
    repo_root: Path,
    models_dir: Path,
    images_dir: Path,
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
    base_distill_limit: int = 0,
    base_teacher_model_limit: int = 0,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    prep_payload = build_v45_prep_pack(
    summary_path=summary_path,
    output_root=output_root,
    seed=seed,
    benchmark_limit=benchmark_limit,
    teacher_route_limit=teacher_route_limit,
    verifier_limit=verifier_limit,
    budget_limit=budget_limit,
    diversity_limit=diversity_limit,
    )
    base_stage1_rows, base_full_rows, base_summary = build_training_rows_v44(
    repo_root=repo_root,
    models_dir=models_dir,
    images_dir=images_dir,
    summary_path=summary_path,
    output_root=Path(prep_payload["v44_base_output_root"]),
    seed=seed,
    communication_limit=base_communication_limit,
    disagreement_limit=base_disagreement_limit,
    base_distill_limit=base_distill_limit,
    base_teacher_model_limit=base_teacher_model_limit,
    )
    stage1_rows, full_rows, merged_summary = assemble_v45_training_rows(
    base_stage1_rows=base_stage1_rows,
    base_full_rows=base_full_rows,
    base_summary=base_summary,
    prep_root=Path(prep_payload["output_root"]),
    )
    merged_summary["prep_payload"] = prep_payload
    return stage1_rows, full_rows, merged_summary


def build_training_rows_v45_dry_run(
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
) -> Dict[str, Any]:
    prep_payload = build_v45_prep_pack(
    summary_path=summary_path,
    output_root=output_root,
    seed=seed,
    benchmark_limit=benchmark_limit,
    teacher_route_limit=teacher_route_limit,
    verifier_limit=verifier_limit,
    budget_limit=budget_limit,
    diversity_limit=diversity_limit,
    )
    base_dry_run = build_training_rows_v44_dry_run(
    summary_path=summary_path,
    output_root=Path(prep_payload["v44_base_output_root"]),
    seed=seed,
    communication_limit=base_communication_limit,
    disagreement_limit=base_disagreement_limit,
    )
    prep_root = Path(prep_payload["output_root"]).resolve()
    benchmark_rows = _read_jsonl_rows(prep_root / "v45_benchmark_bridge_rows.jsonl")
    teacher_rows = _read_jsonl_rows(prep_root / "v45_teacher_role_rows.jsonl")
    verifier_rows = _read_jsonl_rows(prep_root / "v45_verifier_repair_rows.jsonl")
    budget_rows = _read_jsonl_rows(prep_root / "v45_turboquant_budget_rows.jsonl")
    diversity_rows = _read_jsonl_rows(prep_root / "v45_diversity_rows.jsonl")

    tree_of_thought_rows = _read_jsonl_rows(prep_root / "v45_tree_of_thought_rows.jsonl")
    contrastive_rows = _read_jsonl_rows(prep_root / "v45_contrastive_rows.jsonl")
    prm_rows = _read_jsonl_rows(prep_root / "v45_process_reward_rows.jsonl")
    grpo_rows = _read_jsonl_rows(prep_root / "v45_grpo_multi_candidate_rows.jsonl")
    adversarial_rows = _read_jsonl_rows(prep_root / "v45_adversarial_selfplay_rows.jsonl")

    added_rows = (list(benchmark_rows) + list(teacher_rows) + list(verifier_rows) + list(budget_rows)
              + list(diversity_rows) + list(tree_of_thought_rows) + list(contrastive_rows)
              + list(prm_rows) + list(grpo_rows) + list(adversarial_rows))
    added_stage1_rows = [row for row in added_rows if row.domain not in {"vision", "spatial_3d", "video"}]
    source_counts = dict(base_dry_run.get("source_counts") or {})
    for row in added_rows:
    source_counts[row.source] = source_counts.get(row.source, 0) + 1

    dry_run_summary = {
    "dry_run": True,
    "base_mode": "frozen_v44_plus_v45_rows",
    "source_summary": str(summary_path.resolve()),
    "prep_payload": prep_payload,
    "base_stage1_rows": int(base_dry_run.get("estimated_stage1_rows") or 0),
    "base_stage2_rows": int(base_dry_run.get("estimated_stage2_rows") or 0),
    "added_stage1_rows": len(added_stage1_rows),
    "added_stage2_rows": len(added_rows),
    "estimated_stage1_rows": int(base_dry_run.get("estimated_stage1_rows") or 0) + len(added_stage1_rows),
    "estimated_stage2_rows": int(base_dry_run.get("estimated_stage2_rows") or 0) + len(added_rows),
    "source_counts": dict(sorted(source_counts.items())),
    "v45_row_groups": {
        "benchmark_bridge": _bundle_rows(benchmark_rows),
        "teacher_roles": _bundle_rows(teacher_rows),
        "verifier_repair": _bundle_rows(verifier_rows),
        "turboquant_budget": _bundle_rows(budget_rows),
        "diversity_mix": _bundle_rows(diversity_rows),
        "tree_of_thought": _bundle_rows(tree_of_thought_rows),
        "contrastive_hallucination": _bundle_rows(contrastive_rows),
        "process_reward": _bundle_rows(prm_rows),
        "grpo_multi_candidate": _bundle_rows(grpo_rows),
        "adversarial_selfplay": _bundle_rows(adversarial_rows),
    },
    "v44_base_dry_run": base_dry_run,
    }
    dry_run_path = Path(output_root).resolve() / "omni_collective_v45_dry_run_summary.json"
    _write_json(dry_run_path, dry_run_summary)
    return dry_run_summary | {"summary_path": str(dry_run_path)}


def build_v45_prep_pack(
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
    blueprint = build_v45_blueprint(summary, summary_path=summary_path)
    benchmark_rows, benchmark_summary = _benchmark_bridge_rows_v44(seed=seed + 11, limit=benchmark_limit)
    teacher_rows, teacher_summary = _teacher_role_rows_v44(seed=seed + 17, limit=teacher_route_limit)
    verifier_rows, verifier_summary = _verifier_repair_rows_v44(seed=seed + 23, limit=verifier_limit)
    budget_rows, budget_summary = _turboquant_budget_rows_v44(seed=seed + 29, limit=budget_limit)
    diversity_rows, diversity_summary = _diversity_rows_v44(seed=seed + 31, limit=diversity_limit)
    tot_rows, tot_summary = _tree_of_thought_rows_v44(seed=seed + 35, limit=200)
    con_rows, con_summary = _contrastive_hallucination_rows_v44(seed=seed + 37, limit=150)
    prm_rows, prm_summary = _process_reward_rows_v45(seed=seed + 41, limit=200)  # V45 novel
    grpo_rows, grpo_summary = _grpo_multi_candidate_rows_v45(seed=seed + 43, limit=180)  # V45 novel
    adv_rows, adv_summary = _adversarial_selfplay_rows_v45(seed=seed + 47, limit=180)  # V45 novel
    promotion_eval_pack = _promotion_eval_pack_v42()

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    blueprint_path = output_root / "omni_collective_v45_blueprint.json"
    benchmark_path = output_root / "v45_benchmark_bridge_rows.jsonl"
    teacher_path = output_root / "v45_teacher_role_rows.jsonl"
    verifier_path = output_root / "v45_verifier_repair_rows.jsonl"
    budget_path = output_root / "v45_turboquant_budget_rows.jsonl"
    diversity_path = output_root / "v45_diversity_rows.jsonl"
    tot_path = output_root / "v45_tree_of_thought_rows.jsonl"
    con_path = output_root / "v45_contrastive_rows.jsonl"
    prm_path = output_root / "v45_process_reward_rows.jsonl"
    grpo_path = output_root / "v45_grpo_multi_candidate_rows.jsonl"
    adv_path = output_root / "v45_adversarial_selfplay_rows.jsonl"
    promotion_eval_path = output_root / "v45_promotion_eval_pack.json"
    prep_summary_path = output_root / "omni_collective_v45_prep_summary.json"
    v44_base_output_root = (output_root / "v44_base").resolve()

    _write_json(blueprint_path, blueprint)
    _write_jsonl(benchmark_path, benchmark_rows)
    _write_jsonl(teacher_path, teacher_rows)
    _write_jsonl(verifier_path, verifier_rows)
    _write_jsonl(budget_path, budget_rows)
    _write_jsonl(diversity_path, diversity_rows)
    _write_jsonl(tot_path, tot_rows)
    _write_jsonl(con_path, con_rows)
    _write_jsonl(prm_path, prm_rows)
    _write_jsonl(grpo_path, grpo_rows)
    _write_jsonl(adv_path, adv_rows)
    _write_json(promotion_eval_path, {"evaluations": promotion_eval_pack})

    prep_summary = {
    "family": "omni_collective_v45",
    "prepared_from": str(summary_path.resolve()),
    "output_root": str(output_root),
    "v44_base_output_root": str(v44_base_output_root),
    "row_groups": {
        "benchmark_bridge": {"row_count": len(benchmark_rows), "source_counts": benchmark_summary, "path": str(benchmark_path)},
        "teacher_roles": {"row_count": len(teacher_rows), "source_counts": teacher_summary, "path": str(teacher_path)},
        "verifier_repair": {"row_count": len(verifier_rows), "source_counts": verifier_summary, "path": str(verifier_path)},
        "turboquant_budget": {"row_count": len(budget_rows), "source_counts": budget_summary, "path": str(budget_path)},
        "diversity_mix": {"row_count": len(diversity_rows), "source_counts": diversity_summary, "path": str(diversity_path)},
        "tree_of_thought": {"row_count": len(tot_rows), "source_counts": tot_summary, "path": str(tot_path)},
        "contrastive_hallucination": {"row_count": len(con_rows), "source_counts": con_summary, "path": str(con_path)},
        "process_reward": {"row_count": len(prm_rows), "source_counts": prm_summary, "path": str(prm_path)},
        "grpo_multi_candidate": {"row_count": len(grpo_rows), "source_counts": grpo_summary, "path": str(grpo_path)},
        "adversarial_selfplay": {"row_count": len(adv_rows), "source_counts": adv_summary, "path": str(adv_path)},
    },
    "promotion_eval_pack_path": str(promotion_eval_path),
    "total_new_rows": len(benchmark_rows) + len(teacher_rows) + len(verifier_rows) + len(budget_rows) + len(diversity_rows) + len(tot_rows) + len(con_rows) + len(prm_rows) + len(grpo_rows) + len(adv_rows),
    "blueprint_path": str(blueprint_path),
    }
    _write_json(prep_summary_path, prep_summary)
    return prep_summary | {"summary_path": str(prep_summary_path)}


def _run_state_path_v45(output_dir: Path) -> Path:
    return output_dir / "omni_collective_v45_train_state.json"


def _stage_resume_dir_v45(
    output_dir: Path,
    *,
    seed: int,
    distill_limit: int,
    teacher_model_limit: int,
    smoke_train: bool,
) -> Path:
    teacher_tag = int(teacher_model_limit) if int(teacher_model_limit) > 0 else 0
    mode_tag = "smoke" if smoke_train else "frontier"
    return output_dir / "omni_v45_stage_resume" / f"{mode_tag}_seed_{int(seed)}_distill_{int(distill_limit)}_teacherlimit_{teacher_tag}"


def _write_run_state_v45(path: Path, payload: Dict[str, Any]) -> None:
    cooked = dict(payload)
    cooked["updated_at"] = datetime.now().isoformat()
    _write_json_atomic_v42(path, cooked)


def _smoke_training_rows_v45(
    *,
    stage1_rows: Sequence[OmniRow],
    full_rows: Sequence[OmniRow],
    seed: int,
) -> Tuple[List[OmniRow], List[OmniRow], Dict[str, Any]]:
    stage1_priority = [row for row in stage1_rows if "_v45::" in row.source or "_v45" in row.source]
    full_priority = [row for row in full_rows if "_v45::" in row.source or "_v45" in row.source]
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
    "v45_priority_rows": len([row for row in selected_full if "_v45::" in row.source or "_v45" in row.source]),
    "multimodal_rows": len([row for row in selected_full if row.domain in {"vision", "spatial_3d", "video"}]),
    "source_counts": _bundle_rows(selected_full)["source_counts"],
    }
    return selected_stage1, selected_full, summary


def train_model_v45(
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
    smoke_train: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(int(seed))
    random.seed(int(seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_resume_dir = _stage_resume_dir_v45(
    output_dir,
    seed=int(seed),
    distill_limit=int(base_distill_limit),
    teacher_model_limit=int(base_teacher_model_limit),
    smoke_train=bool(smoke_train),
    )
    stage_resume_dir.mkdir(parents=True, exist_ok=True)
    run_state_path = _run_state_path_v45(output_dir)
    _write_run_state_v45(
    run_state_path,
    {
        "status": "building_dataset",
        "stage": "dataset",
        "resume_dir": str(stage_resume_dir),
        "smoke_train": bool(smoke_train),
    },
    )

    stage1_rows, full_rows, dataset_summary = build_training_rows_v45(
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
    )
    if smoke_train:
    stage1_rows, full_rows, smoke_summary = _smoke_training_rows_v45(
        stage1_rows=stage1_rows,
        full_rows=full_rows,
        seed=seed,
    )
    dataset_summary = dict(dataset_summary)
    dataset_summary["smoke_subset"] = smoke_summary
    dataset_summary["stage1_rows"] = len(stage1_rows)
    dataset_summary["stage2_rows"] = len(full_rows)

    print(json.dumps({"event": "dataset_built", "summary": dataset_summary}, ensure_ascii=True), flush=True)
    _write_json_atomic_v42(stage_resume_dir / "dataset_summary.json", {"dataset_summary": dataset_summary})
    _write_run_state_v45(
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
    _write_run_state_v45(
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
    model = OmniCollectiveNetV45(
    vocab_size=max(len(vocab), 2),
    num_intents=len(OMNI_INTENTS_V2),
    num_responses=max(len(response_bank), 1),
    num_vision_classes=len(SCIENCE_IMAGE_CLASSES),
    num_domains=len(OMNI_DOMAIN_LABELS_V2),
    base_embed_dim=144,
    text_hidden=int(V45_MODEL_CONFIG["text_hidden"]),
    image_channels=64,
    word_buckets=word_buckets,
    word_embed_dim=128,
    deep_text_channels=448,
    deep_image_channels=144,
    fusion_hidden=int(V45_MODEL_CONFIG["fusion_hidden"]),
    memory_slots=36,
    depth_steps=14,
    expert_count=int(V45_MODEL_CONFIG["expert_count"]),
    expert_hidden=int(V45_MODEL_CONFIG["expert_hidden"]),
    context_top_k=4,
    expert_top_k=int(V45_MODEL_CONFIG["expert_top_k"]),
    ).to(runtime.device)
    warm_start = _load_expanded_state_from_zip(model, base_zip)
    forward_model, runtime = maybe_compile_model(model, runtime)
    print(json.dumps({"event": "warm_start", "info": warm_start}, ensure_ascii=True), flush=True)
    _write_run_state_v45(
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
    artifact_dir = output_dir / f"supermix_omni_collective_v45_{mode_tag}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / f"omni_collective_v45_{mode_tag}.pth"
    meta_path = artifact_dir / f"omni_collective_v45_{mode_tag}_meta.json"
    summary_out_path = artifact_dir / f"omni_collective_v45_{mode_tag}_summary.json"
    zip_path = output_dir / f"supermix_omni_collective_v45_{mode_tag}_{stamp}.zip"
    desktop_zip_path = models_dir / zip_path.name

    torch.save(model.state_dict(), weights_path)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    meta = {
    "architecture_version": 42,
    "family": "omni_collective_v45",
    "smoke_train": bool(smoke_train),
    "vocab": vocab,
    "response_bank": response_bank,
    "class_info": {},
    "intent_labels": list(OMNI_INTENTS_V2),
    "domain_labels": list(OMNI_DOMAIN_LABELS_V2),
    "max_len": max_len,
    "image_size": int(image_size),
    "embed_dim": 144,
    "text_hidden": int(V45_MODEL_CONFIG["text_hidden"]),
    "image_channels": 64,
    "word_buckets": word_buckets,
    "max_words": max_words,
    "word_embed_dim": 128,
    "deep_text_channels": 448,
    "deep_image_channels": 144,
    "fusion_hidden": int(V45_MODEL_CONFIG["fusion_hidden"]),
    "memory_slots": 36,
    "depth_steps": 14,
    "expert_count": int(V45_MODEL_CONFIG["expert_count"]),
    "expert_hidden": int(V45_MODEL_CONFIG["expert_hidden"]),
    "context_top_k": 4,
    "expert_top_k": int(V45_MODEL_CONFIG["expert_top_k"]),
    "parameter_count": parameter_count,
    "warm_start": warm_start,
    "stage1": stage1,
    "stage2": stage2,
    "seed": int(seed),
    "training_runtime": runtime.to_payload(),
    "deliberation_passes": int(V45_MODEL_CONFIG["deliberation_passes"]),
    "minimum_passes": int(V45_MODEL_CONFIG["minimum_passes"]),
    "grounding_threshold": 0.60,
    "prompt_understanding_mode": "budgeted_route_first_verifier_teacher_specialized_v45",
    "notes": [
        "v45 extends the v44 scaffold with benchmark-bridge replay, teacher-role specialization, verifier-repair supervision, and TurboQuant-inspired budget control.",
        "The v45 data program keeps v44 carryover rows, then adds Gemma 4, Qwen 3.5, Qwen3-Coder-Next, and Qwen3-Omni inspired route and repair slices.",
    ],
    }
    _write_json_atomic_v42(meta_path, meta)
    sample_outputs: List[Dict[str, str]] = []
    if not smoke_train:
    engine = OmniCollectiveEngineV45(weights_path=weights_path, meta_path=meta_path, device=runtime.device)
    sample_prompts = [
        "Which local model should handle a benchmark-style reasoning prompt if exact score matters most?",
        "Translate TurboQuant into a training implication for v45 in one sentence.",
        "A failing test appeared after a refactor. What should happen next?",
        "Which teacher is best for grounded multimodal explanation slices?",
        "Summarize a huge build log into the one failing step and the next action.",
    ]
    sample_outputs = [{"prompt": prompt, "answer": engine.answer(prompt)} for prompt in sample_prompts]
    summary = {
    "artifact": zip_path.name,
    "family": "omni_collective_v45",
    "smoke_train": bool(smoke_train),
    "parameter_count": parameter_count,
    "dataset_summary": dataset_summary,
    "warm_start": warm_start,
    "stage1": stage1,
    "stage2": stage2,
    "training_runtime": runtime.to_payload(),
    "sample_outputs": sample_outputs,
    "notes": [
        "v45 is designed to recover benchmark strength from v42 while keeping v44's route-first and communication improvements.",
        "This scaffold is the first v45 smoke/frontier training entry point on top of the latest v44 artifact.",
        "Smoke packaging skips post-train sample inference so quick validation runs stay manageable on CPU-only environments.",
    ],
    }
    _write_json_atomic_v42(summary_out_path, summary)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    archive.write(weights_path, arcname=weights_path.name)
    archive.write(meta_path, arcname=meta_path.name)
    archive.write(summary_out_path, arcname=summary_out_path.name)
    if not smoke_train:
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(zip_path, desktop_zip_path)
    _write_run_state_v45(
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
    description="Prepare and smoke-train the omni_collective_v45 continuation scaffold."
    )
    parser.add_argument("--summary", default="", help="Optional v44 frontier summary JSON. Defaults to the latest local v44 summary.")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory for the generated v45 prep pack.")
    parser.add_argument("--output_dir", default=str(DEFAULT_TRAIN_OUTPUT_DIR), help="Directory for v45 train artifacts.")
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR), help="Models directory for dry-run dataset assembly and optional publish copy.")
    parser.add_argument("--base_zip", default="", help="Base v44 artifact zip used for warm start. Defaults to the latest local v44 zip.")
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
    parser.add_argument("--dry_run_dataset", action="store_true", help="Build and summarize the merged v45 dry-run dataset on top of the frozen v44 base.")
    parser.add_argument("--train_smoke", action="store_true", help="Run a tiny resumable v45 smoke train on top of the latest v44 base artifact.")
    parser.add_argument("--train_frontier", action="store_true", help="Run the full resumable v45 frontier training job.")
    parser.add_argument("--stdout", action="store_true", help="Print the prep or smoke summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary).resolve() if str(args.summary).strip() else latest_v44_summary_path(REPO_ROOT)
    base_zip = Path(args.base_zip).resolve() if str(args.base_zip).strip() else latest_v44_zip_path(REPO_ROOT)
    payload = build_v45_prep_pack(
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
    dry_run_summary = build_training_rows_v45_dry_run(
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
    )
    payload = payload | {"dry_run_dataset": dry_run_summary}
    if args.train_smoke:
    smoke_result = train_model_v45(
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
        smoke_train=True,
    )
    payload = payload | {"smoke_train": smoke_result}
    if args.train_frontier:
    frontier_result = train_model_v45(
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
        smoke_train=False,
    )
    payload = payload | {"train_frontier": frontier_result}
    if args.stdout:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

# Export fallbacks
from train_omni_collective_v44 import _communication_polish_rows_v41 as _communication_polish_rows_v41
_code_critique_repair_rows_v41 = _code_critique_repair_rows_v41
