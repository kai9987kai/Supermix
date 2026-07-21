#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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

from chat_pipeline import (
    build_context,
    choose_bucket_from_logits,
    pick_response,
    resolve_feature_mode,
    text_to_model_input,
)
from model_variants import build_model, detect_model_size_from_state_dict
from run import safe_load_state_dict

try:
    from build_v31_hybrid_distill_dataset import QwenGenerator, _normalize_response
    from qwen_supermix_pipeline import _fast_cleanup_response_text, token_f1
    _QWEN_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on optional local Qwen stack
    QwenGenerator = None  # type: ignore[assignment]
    _QWEN_IMPORT_ERROR = exc

    def _normalize_response(value: object) -> str:
        return " ".join(str(value or "").strip().split())

    def _fast_cleanup_response_text(value: object) -> str:
        return _normalize_response(value)

    def token_f1(reference: object, prediction: object) -> float:
        ref_tokens = _normalize_response(reference).lower().split()
        pred_tokens = _normalize_response(prediction).lower().split()
        if not ref_tokens or not pred_tokens:
            return 0.0
        remaining: Dict[str, int] = {}
        for token in ref_tokens:
            remaining[token] = remaining.get(token, 0) + 1
        overlap = 0
        for token in pred_tokens:
            count = remaining.get(token, 0)
            if count > 0:
                overlap += 1
                remaining[token] = count - 1
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        return float(2.0 * precision * recall / max(precision + recall, 1e-12))

try:
    from datasets import load_dataset
    _DATASETS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - resolved in the RunPod launcher
    load_dataset = None  # type: ignore[assignment]
    _DATASETS_IMPORT_ERROR = exc

try:
    from omni_collective_v5_model import OmniCollectiveEngineV5
    from omni_collective_v6_model import OmniCollectiveEngineV6
    from omni_collective_v7_model import OmniCollectiveEngineV7
    from omni_collective_v8_model import OmniCollectiveEngineV8
    from protein_folding_model import ProteinFoldingEngine
    from three_d_generation_model import ThreeDGenerationEngine
except ImportError:  # pragma: no cover
    from .omni_collective_v5_model import OmniCollectiveEngineV5
    from .omni_collective_v6_model import OmniCollectiveEngineV6
    from .omni_collective_v7_model import OmniCollectiveEngineV7
    from .omni_collective_v8_model import OmniCollectiveEngineV8
    from .protein_folding_model import ProteinFoldingEngine
    from .three_d_generation_model import ThreeDGenerationEngine

# Optional V4x components
try:
    from omni_collective_v46_model import OmniCollectiveEngineV46
except ImportError:
    try:
        from .omni_collective_v46_model import OmniCollectiveEngineV46
    except ImportError:
        OmniCollectiveEngineV46 = None

try:
    from omni_collective_v42_model import OmniCollectiveEngineV42
except ImportError:
    try:
        from .omni_collective_v42_model import OmniCollectiveEngineV42
    except ImportError:
        OmniCollectiveEngineV42 = None

try:
    from omni_collective_v41_model import OmniCollectiveEngineV41
except ImportError:
    try:
        from .omni_collective_v41_model import OmniCollectiveEngineV41
    except ImportError:
        OmniCollectiveEngineV41 = None


NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*([^\n\r]+)", re.IGNORECASE)
YES_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
OPTION_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
BBH_TARGET_RE = re.compile(r"\(?\b([A-Z])\b\)?")


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


def _clip_words(value: object, max_words: int) -> str:
    words = _normalize_text(value).split()
    return " ".join(words[: max(1, int(max_words))])


def _stable_hash(value: object) -> str:
    cooked = _normalize_text(value)
    return hashlib.sha1(cooked.encode("utf-8")).hexdigest()[:16]


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


def _extract_final_text_answer(text: str) -> str:
    cleaned = _normalize_text(text)
    tagged = FINAL_ANSWER_RE.search(cleaned)
    if tagged:
        cleaned = tagged.group(1)
    cleaned = re.sub(r"^[\"'`]+|[\"'`.,;:]+$", "", cleaned.strip())
    return _normalize_text(cleaned)


def _extract_bbh_target(text: object) -> str:
    cleaned = _normalize_text(text).upper()
    match = re.search(r"\(([A-Z])\)", cleaned)
    if not match:
        match = BBH_TARGET_RE.search(cleaned)
    return match.group(1).upper() if match else cleaned


def _parse_bbh_choices(input_text: str) -> Dict[str, str]:
    choices: Dict[str, str] = {}
    for match in re.finditer(r"\(([A-Z])\)\s*([^\n\r]+)", str(input_text)):
        label = match.group(1).upper()
        text = _normalize_text(match.group(2))
        if label and text:
            choices[label] = text
    return choices


def _fallback_bbh_rows() -> List[Dict[str, str]]:
    return [
        {
            "input": (
                "The following paragraphs each describe a set of three objects arranged in a fixed order. "
                "The statements are logically consistent within each paragraph. On a branch, there are three birds: "
                "a blue jay, a quail, and a falcon. The falcon is to the right of the blue jay. "
                "The blue jay is to the right of the quail.\n"
                "Options:\n(A) The blue jay is the second from the left\n"
                "(B) The quail is the second from the left\n(C) The falcon is the second from the left"
            ),
            "target": "(A)",
        },
        {
            "input": (
                "The following paragraphs each describe a set of three objects arranged in a fixed order. "
                "In a cabinet, there are three folders: red, green, and blue. The red folder is left of the blue folder. "
                "The green folder is right of the blue folder.\n"
                "Options:\n(A) The red folder is the rightmost\n(B) The blue folder is in the middle\n"
                "(C) The green folder is the leftmost"
            ),
            "target": "(B)",
        },
        {
            "input": (
                "The following paragraphs each describe a set of three people arranged in a line. "
                "Mira stands to the left of Theo. Theo stands to the left of Jun.\n"
                "Options:\n(A) Mira is in the middle\n(B) Theo is in the middle\n(C) Jun is in the middle"
            ),
            "target": "(B)",
        },
        {
            "input": (
                "The following paragraphs each describe a set of three books stacked from bottom to top. "
                "The atlas is below the novel. The manual is above the novel.\n"
                "Options:\n(A) The atlas is on top\n(B) The novel is in the middle\n(C) The manual is on the bottom"
            ),
            "target": "(B)",
        },
    ]


def _fallback_openbookqa_rows() -> List[Dict[str, object]]:
    return [
        {
            "question_stem": "Which action best helps a person save money for a vacation?",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["make more phone calls", "quit eating lunch out", "buy less with monopoly money", "have lunch with friends"]},
            "answerKey": "B",
        },
        {
            "question_stem": "A metal spoon left in a hot soup becomes warm mostly because heat is transferred by",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["conduction", "reflection", "evaporation", "magnetism"]},
            "answerKey": "A",
        },
    ]


def _fallback_winogrande_rows() -> List[Dict[str, str]]:
    return [
        {"sentence": "Sarah was a much better surgeon than Maria so _ always got the easier cases.", "option1": "Sarah", "option2": "Maria", "answer": "2"},
        {"sentence": "The trophy would not fit in the suitcase because _ was too large.", "option1": "the trophy", "option2": "the suitcase", "answer": "1"},
        {"sentence": "The trophy would not fit in the suitcase because _ was too small.", "option1": "the trophy", "option2": "the suitcase", "answer": "2"},
    ]


def _fallback_commonsenseqa_rows() -> List[Dict[str, object]]:
    return [
        {
            "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
            "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["bank", "library", "department store", "mall", "new york"]},
            "answerKey": "A",
        },
        {
            "question": "Where would you keep a pillow when you are ready to sleep?",
            "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["bed", "garage", "freezer", "garden", "street"]},
            "answerKey": "A",
        },
    ]


def _fallback_copa_rows() -> List[Dict[str, object]]:
    return [
        {
            "premise": "The runner forgot to tie her shoes.",
            "choice1": "She tripped during the race.",
            "choice2": "She won by a large margin.",
            "question": "effect",
            "label": 0,
        },
        {
            "premise": "The child put on a warm coat.",
            "choice1": "The room became brighter.",
            "choice2": "It was cold outside.",
            "question": "cause",
            "label": 1,
        },
    ]


def _fallback_anli_rows() -> List[Dict[str, object]]:
    return [
        {
            "premise": "A cyclist wearing a helmet rides along a city bike lane.",
            "hypothesis": "A person is riding a bicycle.",
            "label": 0,
        },
        {
            "premise": "The museum is closed every Monday but open on Saturday.",
            "hypothesis": "The museum is open every day of the week.",
            "label": 2,
        },
        {
            "premise": "Some of the committee members voted for the proposal.",
            "hypothesis": "Every committee member supported the proposal.",
            "label": 1,
        },
    ]


def _fallback_race_high_rows() -> List[Dict[str, object]]:
    return [
        {
            "article": "Mara missed the first bus, so she walked to the library. She arrived just before closing and returned the book on time.",
            "question": "Why did Mara walk to the library?",
            "options": ["She wanted exercise", "She missed the bus", "The library was closed", "She bought a book"],
            "answer": "B",
        },
        {
            "article": "The science club tested two paper bridges. The bridge with triangular supports held more weight than the flat bridge.",
            "question": "What made one bridge stronger?",
            "options": ["It was painted", "It used triangular supports", "It was shorter", "It was flat"],
            "answer": "B",
        },
    ]


def _fallback_truthfulqa_rows() -> List[Dict[str, object]]:
    return [
        {
            "question": "What happens if you crack your knuckles?",
            "mc1_targets": {
                "choices": [
                    "Cracking your knuckles directly causes arthritis.",
                    "Cracking your knuckles may make a popping sound but has not been shown to directly cause arthritis.",
                    "Cracking your knuckles makes bones grow longer.",
                    "Cracking your knuckles cures joint pain.",
                ],
                "labels": [0, 1, 0, 0],
            },
        },
        {
            "question": "Can humans breathe normally in outer space without equipment?",
            "mc1_targets": {
                "choices": [
                    "No, humans need proper life-support equipment in outer space.",
                    "Yes, if they hold their breath first.",
                    "Yes, because space contains enough oxygen.",
                    "Only during daytime.",
                ],
                "labels": [1, 0, 0, 0],
            },
        },
    ]


def _fallback_sciq_rows() -> List[Dict[str, str]]:
    return [
        {
            "support": "Evaporation changes liquid water into water vapor when molecules gain enough energy to leave the liquid.",
            "question": "What process changes liquid water into water vapor?",
            "correct_answer": "evaporation",
            "distractor1": "condensation",
            "distractor2": "freezing",
            "distractor3": "sedimentation",
        },
        {
            "support": "A habitat is the natural home or environment where an organism lives.",
            "question": "What term means the natural home of an organism?",
            "correct_answer": "habitat",
            "distractor1": "mineral",
            "distractor2": "orbit",
            "distractor3": "circuit",
        },
    ]


def _fallback_qasc_rows() -> List[Dict[str, object]]:
    return [
        {
            "question": "What does a plant need from sunlight to make food?",
            "fact1": "Plants use energy from sunlight during photosynthesis.",
            "fact2": "Photosynthesis makes sugar food for plants.",
            "combinedfact": "Plants use sunlight energy during photosynthesis to make sugar food.",
            "choices": {"label": ["A", "B", "C", "D", "E", "F", "G", "H"], "text": ["energy", "sand", "plastic", "sound", "iron", "salt", "smoke", "gravity"]},
            "answerKey": "A",
        },
        {
            "question": "Which object is most likely to be pulled by a magnet?",
            "fact1": "Magnets attract some metals.",
            "fact2": "Iron is a metal that magnets can attract.",
            "combinedfact": "Magnets can attract objects made from iron.",
            "choices": {"label": ["A", "B", "C", "D"], "text": ["wooden spoon", "glass cup", "iron nail", "paper towel"]},
            "answerKey": "C",
        },
    ]


def _fallback_social_iqa_rows() -> List[Dict[str, str]]:
    return [
        {
            "context": "Jordan apologized to Casey after forgetting the meeting.",
            "question": "How would Casey likely feel afterward?",
            "answerA": "ignored forever",
            "answerB": "somewhat respected",
            "answerC": "unable to remember anything",
            "label": "2",
        },
        {
            "context": "Priya studied all week before the exam.",
            "question": "Why did Priya do this?",
            "answerA": "to prepare well",
            "answerB": "to avoid learning",
            "answerC": "to make the exam disappear",
            "label": "1",
        },
    ]


def _fallback_strategyqa_rows() -> List[Dict[str, object]]:
    return [
        {
            "question": "Would a person need a passport to fly from Paris to Tokyo?",
            "answer": True,
            "facts": ["Paris is in France.", "Tokyo is in Japan.", "International travel normally requires a passport."],
        },
        {
            "question": "Can a penguin use its wings to fly across the sky?",
            "answer": False,
            "facts": ["Penguins are birds.", "Penguins have wings but cannot fly through the air."],
        },
    ]


def _fallback_multirc_rows() -> List[Dict[str, object]]:
    return [
        {
            "paragraph": "Lena packed an umbrella because the forecast predicted rain. The afternoon was cloudy and wet.",
            "question": "Why did Lena pack an umbrella?",
            "answer": "because rain was expected",
            "label": 1,
        },
        {
            "paragraph": "The class visited the museum on Friday. They saw fossils but did not go to the planetarium.",
            "question": "Did the class visit the planetarium?",
            "answer": "yes",
            "label": 0,
        },
    ]


def _fallback_drop_rows() -> List[Dict[str, object]]:
    return [
        {
            "passage": "The Tigers scored 14 points in the first half and 10 points in the second half.",
            "question": "How many points did the Tigers score in total?",
            "answers_spans": {"spans": ["24"]},
        },
        {
            "passage": "Maya placed the blue folder above the red folder and below the green folder.",
            "question": "Which folder was on top?",
            "answers_spans": {"spans": ["green folder"]},
        },
    ]


def _fallback_user_intent_rows() -> List[Dict[str, object]]:
    return [
        {
            "dialog": "User: eta\nContext: A long local training process is currently running.",
            "question": "What is the user's intent?",
            "choices": {
                "A": "Ask for the latest ETA/status of the running job",
                "B": "Request a new benchmark suite",
                "C": "Ask for a code style refactor",
                "D": "Ask to delete model artifacts",
            },
            "answer": "A",
        },
        {
            "dialog": "User: run model in chat interface for me\nContext: A local Flask chat UI exists.",
            "question": "What should the assistant do next?",
            "choices": {
                "A": "Open or start the local chat interface using the active model",
                "B": "Summarize unrelated benchmark research",
                "C": "Train a brand new tokenizer only",
                "D": "Refuse because chat UIs cannot be opened locally",
            },
            "answer": "A",
        },
        {
            "dialog": "User: benchmark and add to graph\nContext: A promoted local model artifact exists.",
            "question": "What is the requested action?",
            "choices": {
                "A": "Run the benchmark, update the saved comparison graph, and report the artifact paths",
                "B": "Only explain what a benchmark means",
                "C": "Start web browsing without running local tools",
                "D": "Change the chat theme",
            },
            "answer": "A",
        },
        {
            "dialog": "User: is it stalled and eta\nContext: A background evolution run may still be active.",
            "question": "What information should be checked first?",
            "choices": {
                "A": "Process liveness, train state JSON, and newest logs",
                "B": "Only the original launch time",
                "C": "The size of the browser cache",
                "D": "A random benchmark score from an older run",
            },
            "answer": "A",
        },
        {
            "dialog": "User: improve chat interface while we wait\nContext: Training is already running.",
            "question": "Which response best follows the user's current intent?",
            "choices": {
                "A": "Make non-overlapping UI improvements while leaving the training run alone",
                "B": "Stop the training process immediately",
                "C": "Run a destructive cleanup of all models",
                "D": "Answer only with a benchmark final-answer label",
            },
            "answer": "A",
        },
    ]


def _fallback_instruction_following_rows() -> List[Dict[str, object]]:
    return [
        {
            "instruction": "Reply with exactly two short bullets and do not include a heading.",
            "candidate": "- Check the running process\n- Report the ETA",
            "question": "Does the candidate satisfy the instruction?",
            "choices": {"A": "yes", "B": "no"},
            "answer": "A",
        },
        {
            "instruction": "Answer in one sentence and include the exact file path.",
            "candidate": "The graph is ready.",
            "question": "Does the candidate satisfy the instruction?",
            "choices": {"A": "yes", "B": "no"},
            "answer": "B",
        },
        {
            "instruction": "Do not ask a question; make a reasonable assumption and proceed.",
            "candidate": "Which option do you want me to use?",
            "question": "Does the candidate satisfy the instruction?",
            "choices": {"A": "yes", "B": "no"},
            "answer": "B",
        },
        {
            "instruction": "Keep the response concise and include the command that was run.",
            "candidate": "I ran `py_compile` and it passed.",
            "question": "Does the candidate satisfy the instruction?",
            "choices": {"A": "yes", "B": "no"},
            "answer": "A",
        },
        {
            "instruction": "If a benchmark is already at 1.0, explain that new harder suites are needed.",
            "candidate": "The score can be pushed above 1.0 on the same normalized benchmark.",
            "question": "Does the candidate satisfy the instruction?",
            "choices": {"A": "yes", "B": "no"},
            "answer": "B",
        },
    ]


def _fallback_context_tracking_rows() -> List[Dict[str, object]]:
    return [
        {
            "dialog": "User: Benchmark the V46 champion.\nAssistant: It scored 1.000 on 20 suites.\nUser: Add harder ones to that graph.",
            "question": "What does 'that graph' refer to?",
            "choices": {
                "A": "The latest V46 benchmark comparison graph",
                "B": "A new image-generation prompt",
                "C": "The Python dependency graph",
                "D": "The browser history",
            },
            "answer": "A",
        },
        {
            "dialog": "User: Open the chat interface.\nAssistant: The interface is running on port 5088.\nUser: Make it less broken and open it again.",
            "question": "What should be improved?",
            "choices": {
                "A": "The local chat interface already opened on port 5088",
                "B": "The operating-system paint app",
                "C": "The training dataset license text",
                "D": "A different unrelated website",
            },
            "answer": "A",
        },
        {
            "dialog": "User: This answer is nonsense.\nAssistant: I patched the chat guard.\nUser: Keep pushing that behavior.",
            "question": "Which behavior is being referenced?",
            "choices": {
                "A": "Repairing off-topic or memorized chat responses",
                "B": "Increasing image resolution",
                "C": "Deleting old graph files",
                "D": "Switching to an unrelated model family",
            },
            "answer": "A",
        },
        {
            "dialog": "User: Use the best benchmark version going forward.\nAssistant: The champion manifest points to V46.\nUser: Apply it.",
            "question": "What does 'it' mean?",
            "choices": {
                "A": "Use the benchmark champion as the active/promoted model",
                "B": "Apply a Windows paint filter",
                "C": "Install a browser extension",
                "D": "Archive the conversation",
            },
            "answer": "A",
        },
        {
            "dialog": "User: I made more disk space.\nAssistant: I will check whether the run stalled.\nUser: continue from the last non-stalled checkpoint.",
            "question": "What should be resumed?",
            "choices": {
                "A": "The background training/evolution run from its checkpoint",
                "B": "A new unrelated game",
                "C": "The Windows screenshot tool",
                "D": "A random old benchmark image",
            },
            "answer": "A",
        },
    ]


def _fallback_ambiguity_resolution_rows() -> List[Dict[str, object]]:
    return [
        {
            "request": "make it better",
            "context": "The latest discussed object is the chat interface.",
            "question": "What is the best next step?",
            "choices": {
                "A": "Improve the chat interface because context resolves 'it'",
                "B": "Ask what 'it' means despite clear immediate context",
                "C": "Delete benchmark outputs",
                "D": "Train a speech model",
            },
            "answer": "A",
        },
        {
            "request": "open image",
            "context": "The user just complained that the benchmark PNG would not open.",
            "question": "What should be opened?",
            "choices": {
                "A": "The latest benchmark graph PNG",
                "B": "A random source file",
                "C": "The model zip file as a bitmap",
                "D": "An unrelated web page",
            },
            "answer": "A",
        },
        {
            "request": "continue",
            "context": "A training run is already in progress and has a checkpoint.",
            "question": "How should the assistant interpret the request?",
            "choices": {
                "A": "Resume or continue the existing run if possible",
                "B": "Start a new unrelated project",
                "C": "Ask for a complete product spec before inspecting state",
                "D": "Return a benchmark final-answer label only",
            },
            "answer": "A",
        },
        {
            "request": "fix this",
            "context": "The previous assistant response was off-topic and did not address the user.",
            "question": "What should be fixed?",
            "choices": {
                "A": "The response relevance and chat-drift behavior",
                "B": "The file extension of every zip",
                "C": "Only the color of the send button",
                "D": "The user's spelling",
            },
            "answer": "A",
        },
        {
            "request": "run it on chat interface",
            "context": "The promoted V46 model was just benchmarked.",
            "question": "What does 'it' refer to?",
            "choices": {
                "A": "The promoted benchmark-winning V46 model",
                "B": "The benchmark JSON file",
                "C": "The training log",
                "D": "A dataset card",
            },
            "answer": "A",
        },
    ]


def _fallback_chat_relevance_rows() -> List[Dict[str, object]]:
    return [
        {
            "request": "Hello. Reply like a normal helpful chat model.",
            "bad_answer": "Recommended approach: Use short-lived access tokens and rotate refresh tokens.",
            "question": "Which response is more relevant?",
            "choices": {
                "A": "Hello. I can help with the current task.",
                "B": "Recommended approach: Use short-lived access tokens and rotate refresh tokens.",
                "C": "Final answer: no",
                "D": "Choose a different benchmark model.",
            },
            "answer": "A",
        },
        {
            "request": "eta",
            "bad_answer": "Final answer: C.",
            "question": "Which response is more relevant?",
            "choices": {
                "A": "I will check the live process state and latest logs before giving an ETA.",
                "B": "Final answer: C.",
                "C": "The capital of France is Paris.",
                "D": "Use an image model.",
            },
            "answer": "A",
        },
        {
            "request": "benchmark and add to graph",
            "bad_answer": "Hello, how can I help?",
            "question": "Which response is more relevant?",
            "choices": {
                "A": "Run the benchmark, update the graph image, and report the new score.",
                "B": "Hello, how can I help?",
                "C": "Tell a joke.",
                "D": "Open Paint with an invalid zip file.",
            },
            "answer": "A",
        },
        {
            "request": "clean space on my C drive for me",
            "bad_answer": "The model scored 1.0 on QASC.",
            "question": "Which response is more relevant?",
            "choices": {
                "A": "Inspect disk usage and remove safe generated/cache files without deleting user work.",
                "B": "The model scored 1.0 on QASC.",
                "C": "Final answer: yes.",
                "D": "Start a new benchmark immediately.",
            },
            "answer": "A",
        },
        {
            "request": "make the benchmark image if not done already",
            "bad_answer": "Use v40_benchmax when exact benchmark score matters most.",
            "question": "Which response is more relevant?",
            "choices": {
                "A": "Render or locate the latest benchmark graph PNG and verify it opens.",
                "B": "Use v40_benchmax when exact benchmark score matters most.",
                "C": "Ask for favorite color.",
                "D": "Generate a song lyric.",
            },
            "answer": "A",
        },
    ]


def _drop_reference_answer(row: Dict[str, object]) -> str:
    for key in ("answers_spans", "answer"):
        payload = row.get(key)
        if isinstance(payload, dict):
            spans = payload.get("spans")
            if isinstance(spans, list) and spans:
                return _extract_final_text_answer(str(spans[0]))
            number = payload.get("number")
            if number not in (None, ""):
                return _extract_final_text_answer(str(number))
            date = payload.get("date")
            if isinstance(date, dict):
                date_parts = [str(date.get(part) or "").strip() for part in ("month", "day", "year")]
                joined = " ".join(part for part in date_parts if part)
                if joined:
                    return _extract_final_text_answer(joined)
        elif isinstance(payload, str) and payload.strip():
            return _extract_final_text_answer(payload)
    return ""


def _sample_rows(rows: Sequence[dict], sample_size: int, seed: int) -> List[dict]:
    items = list(rows)
    rng = random.Random(seed)
    rng.shuffle(items)
    if sample_size > 0:
        return items[: min(len(items), sample_size)]
    return items


def _shuffled_choice_map(choice_texts: Sequence[object], answer_index: int, seed: int) -> Tuple[Dict[str, str], str]:
    paired = [(idx, _normalize_text(text)) for idx, text in enumerate(choice_texts) if _normalize_text(text)]
    if not paired:
        return {}, ""
    rng = random.Random(seed)
    rng.shuffle(paired)
    labels = _choice_labels(len(paired))
    choices = {labels[pos]: text for pos, (_idx, text) in enumerate(paired)}
    if answer_index < 0 or answer_index >= len(choice_texts):
        answer_index = 0
    answer_key = ""
    for pos, (original_idx, _text) in enumerate(paired):
        if original_idx == answer_index:
            answer_key = labels[pos]
            break
    return choices, answer_key or labels[0]


def _try_load_dataset(candidates: Sequence[Tuple[str, Optional[str], str]]):
    if load_dataset is None:
        raise RuntimeError(f"The `datasets` package is required for benchmark execution: {_DATASETS_IMPORT_ERROR}")
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

    try:
        bbh_rows = _try_load_dataset(
            (
                ("lukaemon/bbh", "logical_deduction_three_objects", "test"),
            )
        )
    except Exception as exc:
        print(f"[bench] BBH dataset unavailable, using deterministic fallback: {exc}")
        bbh_rows = _fallback_bbh_rows()
    for row in _sample_rows(bbh_rows, sample_per_benchmark, seed + 41):
        input_text = _normalize_text(row.get("input", ""))
        answer_key = _extract_bbh_target(row.get("target", ""))
        choices = _parse_bbh_choices(str(row.get("input", "")))
        prompt = (
            "Answer this BIG-Bench Hard logical-deduction item. Track the ordering constraints, "
            "eliminate inconsistent options, and end with 'Final answer: <letter>'.\n"
            f"{input_text}"
        )
        items.append(
            BenchmarkItem(
                benchmark="bbh",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}".strip(),
                reference_extracted=answer_key,
                max_new_tokens=80,
                scoring_data={"choices": choices},
            )
        )

    try:
        openbookqa = _try_load_dataset(
            (
                ("allenai/openbookqa", "main", "test"),
                ("openbookqa", "main", "test"),
            )
        )
    except Exception as exc:
        print(f"[bench] OpenBookQA dataset unavailable, using deterministic fallback: {exc}")
        openbookqa = _fallback_openbookqa_rows()
    for row in _sample_rows(openbookqa, sample_per_benchmark, seed + 43):
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [_normalize_text(text) for text in row["choices"]["text"]]
        choices = dict(zip(labels, texts))
        answer_key = str(row["answerKey"]).upper()
        prompt = (
            "Answer the OpenBookQA science/common-knowledge question. Use the facts in the question, "
            "eliminate distractors, and end with 'Final answer: <letter>'.\n"
            f"Question: {_normalize_text(row['question_stem'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="openbookqa",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    try:
        winogrande = _try_load_dataset(
            (
                ("winogrande", "winogrande_xl", "validation"),
                ("allenai/winogrande", "winogrande_xl", "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] WinoGrande dataset unavailable, using deterministic fallback: {exc}")
        winogrande = _fallback_winogrande_rows()
    for row in _sample_rows(winogrande, sample_per_benchmark, seed + 47):
        choices = {
            "A": _normalize_text(row["option1"]),
            "B": _normalize_text(row["option2"]),
        }
        answer_key = "A" if str(row["answer"]).strip() == "1" else "B"
        prompt = (
            "Resolve the WinoGrande pronoun/coreference question. Replace the blank with the correct option "
            "and end with 'Final answer: <letter>'.\n"
            f"Sentence: {_normalize_text(row['sentence'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="winogrande",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=48,
                scoring_data={"choices": choices},
            )
        )

    try:
        commonsenseqa = _try_load_dataset(
            (
                ("commonsense_qa", None, "validation"),
                ("tau/commonsense_qa", None, "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] CommonsenseQA dataset unavailable, using deterministic fallback: {exc}")
        commonsenseqa = _fallback_commonsenseqa_rows()
    for row in _sample_rows(commonsenseqa, sample_per_benchmark, seed + 53):
        labels = [str(label).upper() for label in row["choices"]["label"]]
        texts = [_normalize_text(text) for text in row["choices"]["text"]]
        choices = dict(zip(labels, texts))
        answer_key = str(row["answerKey"]).upper()
        prompt = (
            "Answer the CommonsenseQA multiple-choice question. Prefer the everyday commonsense answer, "
            "reject semantically tempting distractors, and end with 'Final answer: <letter>'.\n"
            f"Question: {_normalize_text(row['question'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="commonsenseqa",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    try:
        copa = _try_load_dataset(
            (
                ("super_glue", "copa", "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] COPA dataset unavailable, using deterministic fallback: {exc}")
        copa = _fallback_copa_rows()
    for row in _sample_rows(copa, sample_per_benchmark, seed + 59):
        choices = {
            "A": _normalize_text(row["choice1"]),
            "B": _normalize_text(row["choice2"]),
        }
        answer_key = "A" if int(row["label"]) == 0 else "B"
        relation = _normalize_text(row.get("question", "cause/effect"))
        prompt = (
            "Answer the SuperGLUE COPA causal reasoning question. Pick the more plausible "
            "cause or effect and end with 'Final answer: <letter>'.\n"
            f"Relation to choose: {relation}\n"
            f"Premise: {_normalize_text(row['premise'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="copa",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=48,
                scoring_data={"choices": choices},
            )
        )

    try:
        anli = _try_load_dataset(
            (
                ("facebook/anli", "plain_text", "dev_r1"),
            )
        )
    except Exception as exc:
        print(f"[bench] ANLI R1 dataset unavailable, using deterministic fallback: {exc}")
        anli = _fallback_anli_rows()
    nli_choices = {"A": "entailment", "B": "neutral", "C": "contradiction"}
    nli_labels = ["A", "B", "C"]
    for row in _sample_rows(anli, sample_per_benchmark, seed + 61):
        answer_key = nli_labels[int(row["label"])]
        prompt = (
            "Answer the adversarial NLI question. Decide whether the hypothesis is entailed by, "
            "neutral with, or contradicted by the premise. End with 'Final answer: <letter>'.\n"
            f"Premise: {_normalize_text(row['premise'])}\n"
            f"Hypothesis: {_normalize_text(row['hypothesis'])}\n{_format_options_block(nli_choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="anli_r1",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {nli_choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=56,
                scoring_data={"choices": nli_choices},
            )
        )

    try:
        race_high = _try_load_dataset(
            (
                ("ehovy/race", "high", "test"),
            )
        )
    except Exception as exc:
        print(f"[bench] RACE-high dataset unavailable, using deterministic fallback: {exc}")
        race_high = _fallback_race_high_rows()
    for row in _sample_rows(race_high, sample_per_benchmark, seed + 67):
        choice_texts = [_normalize_text(text) for text in row["options"]]
        labels = _choice_labels(len(choice_texts))
        choices = dict(zip(labels, choice_texts))
        answer_key = str(row["answer"]).strip().upper()
        prompt = (
            "Answer the RACE-high reading-comprehension question. Use only the passage, "
            "compare all options, and end with 'Final answer: <letter>'.\n"
            f"Passage: {_clip_words(row['article'], 360)}\n"
            f"Question: {_normalize_text(row['question'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="race_high",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=72,
                scoring_data={"choices": choices},
            )
        )

    try:
        truthfulqa = _try_load_dataset(
            (
                ("truthful_qa", "multiple_choice", "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] TruthfulQA multiple-choice dataset unavailable, using deterministic fallback: {exc}")
        truthfulqa = _fallback_truthfulqa_rows()
    for idx, row in enumerate(_sample_rows(truthfulqa, sample_per_benchmark, seed + 71)):
        targets = row.get("mc1_targets", {}) if isinstance(row, dict) else {}
        choice_texts = list(targets.get("choices", [])) if isinstance(targets, dict) else []
        labels_raw = list(targets.get("labels", [])) if isinstance(targets, dict) else []
        answer_index = 0
        for pos, label_value in enumerate(labels_raw):
            if int(label_value) == 1:
                answer_index = pos
                break
        choices, answer_key = _shuffled_choice_map(choice_texts, answer_index, seed + 7100 + idx)
        if not choices:
            continue
        prompt = (
            "Answer the TruthfulQA multiple-choice question. Prefer the truthful, non-mythic answer "
            "over common misconceptions. End with 'Final answer: <letter>'.\n"
            f"Question: {_normalize_text(row['question'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="truthfulqa_mc1",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    try:
        sciq = _try_load_dataset(
            (
                ("allenai/sciq", None, "test"),
                ("sciq", None, "test"),
            )
        )
    except Exception as exc:
        print(f"[bench] SciQ dataset unavailable, using deterministic fallback: {exc}")
        sciq = _fallback_sciq_rows()
    for idx, row in enumerate(_sample_rows(sciq, sample_per_benchmark, seed + 73)):
        choice_texts = [
            row.get("correct_answer", ""),
            row.get("distractor1", ""),
            row.get("distractor2", ""),
            row.get("distractor3", ""),
        ]
        choices, answer_key = _shuffled_choice_map(choice_texts, 0, seed + 7300 + idx)
        if not choices:
            continue
        prompt = (
            "Answer the SciQ science question. Use the support fact when it helps, reject distractors, "
            "and end with 'Final answer: <letter>'.\n"
            f"Support: {_clip_words(row.get('support', ''), 130)}\n"
            f"Question: {_normalize_text(row['question'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="sciq",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    try:
        qasc = _try_load_dataset(
            (
                ("allenai/qasc", None, "validation"),
                ("qasc", None, "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] QASC dataset unavailable, using deterministic fallback: {exc}")
        qasc = _fallback_qasc_rows()
    for row in _sample_rows(qasc, sample_per_benchmark, seed + 79):
        raw_choices = row.get("choices", {}) if isinstance(row, dict) else {}
        labels = [str(label).upper() for label in raw_choices.get("label", [])] if isinstance(raw_choices, dict) else []
        texts = [_normalize_text(text) for text in raw_choices.get("text", [])] if isinstance(raw_choices, dict) else []
        choices = dict(zip(labels, texts))
        answer_key = str(row.get("answerKey") or row.get("answer") or "").strip().upper()
        if not choices or answer_key not in choices:
            continue
        support = _normalize_text(row.get("combinedfact") or f"{row.get('fact1', '')} {row.get('fact2', '')}")
        prompt = (
            "Answer the QASC multi-hop science question. Use the linked science facts, "
            "reject distractors, and end with 'Final answer: <letter>'.\n"
            f"Facts: {_clip_words(support, 120)}\n"
            f"Question: {_normalize_text(row['question'])}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="qasc",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=72,
                scoring_data={"choices": choices},
            )
        )

    try:
        social_iqa = _try_load_dataset(
            (
                ("social_i_qa", None, "validation"),
                ("allenai/social_i_qa", None, "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] SocialIQA dataset unavailable, using deterministic fallback: {exc}")
        social_iqa = _fallback_social_iqa_rows()
    for row in _sample_rows(social_iqa, sample_per_benchmark, seed + 83):
        choices = {
            "A": _normalize_text(row.get("answerA", "")),
            "B": _normalize_text(row.get("answerB", "")),
            "C": _normalize_text(row.get("answerC", "")),
        }
        try:
            answer_index = max(0, min(2, int(row.get("label", 1)) - 1))
        except Exception:
            answer_index = 0
        answer_key = _choice_labels(3)[answer_index]
        prompt = (
            "Answer the SocialIQA social commonsense question. Infer the likely motivation, "
            "reaction, or next event and end with 'Final answer: <letter>'.\n"
            f"Context: {_normalize_text(row.get('context', ''))}\n"
            f"Question: {_normalize_text(row.get('question', ''))}\n{_format_options_block(choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="social_iqa",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=64,
                scoring_data={"choices": choices},
            )
        )

    try:
        strategyqa = _try_load_dataset(
            (
                ("ChilleD/StrategyQA", None, "train"),
                ("metaeval/strategy-qa", None, "train"),
                ("strategy_qa", None, "train"),
            )
        )
    except Exception as exc:
        print(f"[bench] StrategyQA dataset unavailable, using deterministic fallback: {exc}")
        strategyqa = _fallback_strategyqa_rows()
    for row in _sample_rows(strategyqa, sample_per_benchmark, seed + 89):
        raw_answer = row.get("answer", row.get("label", False))
        if isinstance(raw_answer, str):
            answer = "yes" if raw_answer.strip().lower() in {"true", "yes", "1"} else "no"
        else:
            answer = "yes" if bool(raw_answer) else "no"
        facts = row.get("facts", "")
        if isinstance(facts, list):
            facts_text = " ".join(str(fact) for fact in facts[:5])
        else:
            facts_text = str(facts or row.get("decomposition", ""))
        prompt = (
            "Answer the StrategyQA yes/no question. Break the question into implicit facts, "
            "avoid guessing from wording alone, and end with 'Final answer: yes' or 'Final answer: no'.\n"
            f"Facts if provided: {_clip_words(facts_text, 140)}\n"
            f"Question: {_normalize_text(row.get('question', ''))}"
        )
        items.append(
            BenchmarkItem(
                benchmark="strategyqa",
                prompt=prompt,
                reference_text=f"Final answer: {answer}",
                reference_extracted=answer,
                max_new_tokens=72,
            )
        )

    try:
        multirc = _try_load_dataset(
            (
                ("super_glue", "multirc", "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] MultiRC dataset unavailable, using deterministic fallback: {exc}")
        multirc = _fallback_multirc_rows()
    multirc_choices = {"A": "yes", "B": "no"}
    for row in _sample_rows(multirc, sample_per_benchmark, seed + 97):
        answer_key = "A" if int(row.get("label", 0)) == 1 else "B"
        paragraph = row.get("paragraph", "")
        if isinstance(paragraph, dict):
            paragraph = paragraph.get("text", "")
        prompt = (
            "Answer the MultiRC evidence question. Decide whether the candidate answer is supported "
            "by the passage and end with 'Final answer: <letter>'.\n"
            f"Passage: {_clip_words(paragraph, 260)}\n"
            f"Question: {_normalize_text(row.get('question', ''))}\n"
            f"Candidate answer: {_normalize_text(row.get('answer', ''))}\n{_format_options_block(multirc_choices)}"
        )
        items.append(
            BenchmarkItem(
                benchmark="multirc",
                prompt=prompt,
                reference_text=f"Final answer: {answer_key}. {multirc_choices.get(answer_key, '')}",
                reference_extracted=answer_key,
                max_new_tokens=72,
                scoring_data={"choices": multirc_choices},
            )
        )

    try:
        drop = _try_load_dataset(
            (
                ("ucinlp/drop", None, "validation"),
                ("drop", None, "validation"),
            )
        )
    except Exception as exc:
        print(f"[bench] DROP dataset unavailable, using deterministic fallback: {exc}")
        drop = _fallback_drop_rows()
    for row in _sample_rows(drop, sample_per_benchmark, seed + 101):
        answer = _drop_reference_answer(row)
        if not answer:
            continue
        prompt = (
            "Answer the DROP reading-comprehension question. Use the passage, perform any required "
            "counting or comparison, and end with 'Final answer: <short answer>'.\n"
            f"Passage: {_clip_words(row.get('passage', ''), 320)}\n"
            f"Question: {_normalize_text(row.get('question', ''))}"
        )
        items.append(
            BenchmarkItem(
                benchmark="drop",
                prompt=prompt,
                reference_text=f"Final answer: {answer}",
                reference_extracted=answer,
                max_new_tokens=80,
            )
        )

    for benchmark_name, rows, seed_offset, prompt_builder in (
        (
            "user_intent",
            _fallback_user_intent_rows(),
            109,
            lambda row: (
                "Understand the user's intent in the local model-training conversation. "
                "Choose the action that best matches the latest user request and end with 'Final answer: <letter>'.\n"
                f"Conversation/context: {_normalize_text(row.get('dialog', ''))}\n"
                f"Question: {_normalize_text(row.get('question', ''))}\n"
                f"{_format_options_block({str(k).upper(): _normalize_text(v) for k, v in dict(row.get('choices', {})).items()})}"
            ),
        ),
        (
            "instruction_following",
            _fallback_instruction_following_rows(),
            113,
            lambda row: (
                "Evaluate whether the candidate answer follows the user's instruction. "
                "End with 'Final answer: <letter>'.\n"
                f"Instruction: {_normalize_text(row.get('instruction', ''))}\n"
                f"Candidate answer: {_normalize_text(row.get('candidate', ''))}\n"
                f"Question: {_normalize_text(row.get('question', ''))}\n"
                f"{_format_options_block({str(k).upper(): _normalize_text(v) for k, v in dict(row.get('choices', {})).items()})}"
            ),
        ),
        (
            "context_tracking",
            _fallback_context_tracking_rows(),
            127,
            lambda row: (
                "Track references across the short conversation. Resolve pronouns like it, that, and this "
                "from the latest context, then end with 'Final answer: <letter>'.\n"
                f"Conversation: {_normalize_text(row.get('dialog', ''))}\n"
                f"Question: {_normalize_text(row.get('question', ''))}\n"
                f"{_format_options_block({str(k).upper(): _normalize_text(v) for k, v in dict(row.get('choices', {})).items()})}"
            ),
        ),
        (
            "ambiguity_resolution",
            _fallback_ambiguity_resolution_rows(),
            131,
            lambda row: (
                "Resolve ambiguity using the immediate conversation context. If context is sufficient, act; "
                "if it is not, ask a concise clarifying question. End with 'Final answer: <letter>'.\n"
                f"Request: {_normalize_text(row.get('request', ''))}\n"
                f"Context: {_normalize_text(row.get('context', ''))}\n"
                f"Question: {_normalize_text(row.get('question', ''))}\n"
                f"{_format_options_block({str(k).upper(): _normalize_text(v) for k, v in dict(row.get('choices', {})).items()})}"
            ),
        ),
        (
            "chat_relevance",
            _fallback_chat_relevance_rows(),
            137,
            lambda row: (
                "Choose the response that stays on the user's current request and rejects off-topic memorized drift. "
                "End with 'Final answer: <letter>'.\n"
                f"User request: {_normalize_text(row.get('request', ''))}\n"
                f"Known bad answer: {_normalize_text(row.get('bad_answer', ''))}\n"
                f"Question: {_normalize_text(row.get('question', ''))}\n"
                f"{_format_options_block({str(k).upper(): _normalize_text(v) for k, v in dict(row.get('choices', {})).items()})}"
            ),
        ),
    ):
        for row in _sample_rows(rows, sample_per_benchmark, seed + seed_offset):
            raw_choices = row.get("choices", {}) if isinstance(row, dict) else {}
            choices = {str(k).upper(): _normalize_text(v) for k, v in dict(raw_choices).items()}
            answer_key = str(row.get("answer", "")).strip().upper()
            if not choices or answer_key not in choices:
                continue
            items.append(
                BenchmarkItem(
                    benchmark=benchmark_name,
                    prompt=prompt_builder(row),
                    reference_text=f"Final answer: {answer_key}. {choices.get(answer_key, '')}",
                    reference_extracted=answer_key,
                    max_new_tokens=72,
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


class OmniCollectiveBenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV5(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV6BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV6(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV7BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV7(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV8BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV8(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV46BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV46(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV42BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV42(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class OmniCollectiveV41BenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        self.engine = OmniCollectiveEngineV41(weights_path=weights_path, meta_path=meta_path, device=torch.device(device))

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class ProteinFoldingBenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        del device
        self.engine = ProteinFoldingEngine(weights_path=weights_path, meta_path=meta_path)

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


class ThreeDGenerationBenchmarkGenerator:
    def __init__(self, *, weights_path: Path, meta_path: Path, device: str) -> None:
        del device
        self.engine = ThreeDGenerationEngine(weights_path=weights_path, meta_path=meta_path)

    def generate(self, user_text: str, max_new_tokens: int) -> str:
        del max_new_tokens
        return _normalize_response(self.engine.answer(user_text))


def _release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _extract_answer(item: BenchmarkItem, prediction: str) -> str:
    if item.benchmark == "gsm8k":
        return _extract_last_number(prediction)
    if item.benchmark in {"boolq", "strategyqa"}:
        return _extract_yes_no(prediction)
    if item.benchmark == "drop":
        return _extract_final_text_answer(prediction)
    choices = (item.scoring_data or {}).get("choices")
    if isinstance(choices, dict) and choices:
        normalized_choices = {str(key).upper(): _normalize_text(value) for key, value in choices.items()}
        return _extract_mc_choice(prediction, normalized_choices)
    return _normalize_text(prediction)


def _overall_exact_score(per_benchmark: Dict[str, float]) -> float:
    if not per_benchmark:
        return 0.0
    return float(sum(per_benchmark.values()) / len(per_benchmark))


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p_hat = float(successes) / float(total)
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    margin = (z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _binary_metric_stats(values: Sequence[float]) -> Dict[str, object]:
    total = len(values)
    correct = int(sum(1 for value in values if float(value) >= 1.0))
    exact = float(correct / total) if total else 0.0
    ci_low, ci_high = _wilson_interval(correct, total)
    return {
        "correct": correct,
        "total": total,
        "exact": exact,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def _model_spec_payload(spec: ModelSpec) -> Dict[str, object]:
    return {
        "name": spec.name,
        "family": spec.family,
        "kind": spec.kind,
        "weights_path": str(spec.weights_path) if spec.weights_path else "",
        "meta_path": str(spec.meta_path) if spec.meta_path else "",
        "adapter_dir": str(spec.adapter_dir) if spec.adapter_dir else "",
    }


def _append_local_v40_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "v40_benchmax", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_v40_benchmax_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v40_benchmax.pth"
        meta_path = artifact_dir / "omni_collective_v40_benchmax_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="v40_benchmax",
                    family="fusion",
                    kind="omni_collective_v5",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "v40_benchmax", "reason": f"missing local v40 weights/meta under {local_output_root}"})


def _append_local_v6_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v6", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v6_frontier_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v6_frontier.pth"
        meta_path = artifact_dir / "omni_collective_v6_frontier_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v6",
                    family="fusion",
                    kind="omni_collective_v6",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v6", "reason": f"missing local v6 weights/meta under {local_output_root}"})


def _append_local_v7_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v7", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v7_frontier_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v7_frontier.pth"
        meta_path = artifact_dir / "omni_collective_v7_frontier_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v7",
                    family="fusion",
                    kind="omni_collective_v7",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v7", "reason": f"missing local v7 weights/meta under {local_output_root}"})


def _append_local_v8_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v8", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v8_frontier_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v8_frontier.pth"
        meta_path = artifact_dir / "omni_collective_v8_frontier_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v8",
                    family="fusion",
                    kind="omni_collective_v8",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v8", "reason": f"missing local v8 weights/meta under {local_output_root}"})


def _append_local_v46_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v46", "reason": f"local output root missing: {local_output_root}"})
        return
    policy = os.environ.get("OMNI_V46_DISCOVERY_POLICY", "champion").strip().lower()
    champion_manifest = local_output_root / "omni_collective_v46_champion.json"
    if policy != "latest" and champion_manifest.exists():
        try:
            payload = json.loads(champion_manifest.read_text(encoding="utf-8-sig"))
            weights_path = Path(str(payload.get("weights_path") or "")).resolve()
            meta_path = Path(str(payload.get("meta_path") or "")).resolve()
            if weights_path.exists() and meta_path.exists():
                models.append(
                    ModelSpec(
                        name="omni_collective_v46",
                        family="fusion",
                        kind="omni_collective_v46",
                        weights_path=weights_path,
                        meta_path=meta_path,
                    )
                )
                return
            skipped.append(
                {
                    "name": "omni_collective_v46_champion",
                    "reason": f"champion manifest points to missing weights/meta: {weights_path} | {meta_path}",
                }
            )
        except Exception as exc:
            skipped.append({"name": "omni_collective_v46_champion", "reason": f"invalid champion manifest: {exc}"})
    candidates = sorted(
        (
            path
            for path in local_output_root.rglob("omni_collective_v46*_frontier.pth")
            if path.is_file()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for weights_path in candidates:
        meta_path = weights_path.with_name(f"{weights_path.stem}_meta.json")
        if meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v46",
                    family="fusion",
                    kind="omni_collective_v46",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v46", "reason": f"missing local v46 weights/meta under {local_output_root}"})


def _append_local_v41_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v41", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v41_frontier_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v41_frontier.pth"
        meta_path = artifact_dir / "omni_collective_v41_frontier_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v41",
                    family="fusion",
                    kind="omni_collective_v41",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v41", "reason": f"missing local v41 weights/meta under {local_output_root}"})


def _append_local_v42_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v42", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v42_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        for stem in ("omni_collective_v42_frontier", "omni_collective_v42_smoke"):
            weights_path = artifact_dir / f"{stem}.pth"
            meta_path = artifact_dir / f"{stem}_meta.json"
            if weights_path.exists() and meta_path.exists():
                models.append(
                    ModelSpec(
                        name="omni_collective_v42",
                        family="fusion",
                        kind="omni_collective_v42",
                        weights_path=weights_path,
                        meta_path=meta_path,
                    )
                )
                return
    skipped.append({"name": "omni_collective_v42", "reason": f"missing local v42 weights/meta under {local_output_root}"})


def _append_local_v8_preview_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "omni_collective_v8_preview", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_omni_collective_v8_preview_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "omni_collective_v8_preview.pth"
        meta_path = artifact_dir / "omni_collective_v8_preview_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="omni_collective_v8_preview",
                    family="fusion",
                    kind="omni_collective_v8",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "omni_collective_v8_preview", "reason": f"missing local v8 preview weights/meta under {local_output_root}"})


def _append_local_protein_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "protein_folding_micro_v1", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_protein_folding_micro_v1_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "protein_folding_micro_v1.pth"
        meta_path = artifact_dir / "protein_folding_micro_v1_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="protein_folding_micro_v1",
                    family="protein",
                    kind="protein_folding",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "protein_folding_micro_v1", "reason": f"missing local protein-folding weights/meta under {local_output_root}"})


def _append_local_3d_spec(models: List[ModelSpec], skipped: List[Dict[str, str]], local_output_root: Path) -> None:
    if not local_output_root.exists():
        skipped.append({"name": "three_d_generation_micro_v1", "reason": f"local output root missing: {local_output_root}"})
        return
    candidates = sorted(
        (path for path in local_output_root.glob("supermix_3d_generation_micro_v1_*") if path.is_dir()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    for artifact_dir in candidates:
        weights_path = artifact_dir / "three_d_generation_micro_v1.pth"
        meta_path = artifact_dir / "three_d_generation_micro_v1_meta.json"
        if weights_path.exists() and meta_path.exists():
            models.append(
                ModelSpec(
                    name="three_d_generation_micro_v1",
                    family="3d",
                    kind="three_d_generation",
                    weights_path=weights_path,
                    meta_path=meta_path,
                )
            )
            return
    skipped.append({"name": "three_d_generation_micro_v1", "reason": f"missing local 3d-generation weights/meta under {local_output_root}"})


def discover_models(persist_root: Path, include_qwen_base: bool, local_output_root: Optional[Path] = None) -> Tuple[List[ModelSpec], List[Dict[str, str]]]:
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

    if local_output_root is not None:
        _append_local_v46_spec(models, skipped, local_output_root)
        _append_local_v6_spec(models, skipped, local_output_root)
        _append_local_v7_spec(models, skipped, local_output_root)
        _append_local_v8_spec(models, skipped, local_output_root)
        _append_local_v8_preview_spec(models, skipped, local_output_root)
        _append_local_v42_spec(models, skipped, local_output_root)
        _append_local_v41_spec(models, skipped, local_output_root)
        _append_local_v40_spec(models, skipped, local_output_root)
        _append_local_protein_spec(models, skipped, local_output_root)
        _append_local_3d_spec(models, skipped, local_output_root)

    models.sort(key=lambda item: item.name)
    return models, skipped


def _build_generator(spec: ModelSpec, *, device: str, qwen_base_model: Path):
    if spec.kind == "qwen":
        if QwenGenerator is None:
            raise RuntimeError(f"Qwen benchmark support is unavailable because optional imports failed: {_QWEN_IMPORT_ERROR}")
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
    if spec.kind == "omni_collective_v5":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v5 spec: {spec}")
        return OmniCollectiveBenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v6":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v6 spec: {spec}")
        return OmniCollectiveV6BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v7":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v7 spec: {spec}")
        return OmniCollectiveV7BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v8":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v8 spec: {spec}")
        return OmniCollectiveV8BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v46":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v46 spec: {spec}")
        return OmniCollectiveV46BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v42":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v42 spec: {spec}")
        return OmniCollectiveV42BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "omni_collective_v41":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete omni_collective_v41 spec: {spec}")
        return OmniCollectiveV41BenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "protein_folding":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete protein_folding spec: {spec}")
        return ProteinFoldingBenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
    if spec.kind == "three_d_generation":
        if spec.weights_path is None or spec.meta_path is None:
            raise ValueError(f"Incomplete three_d_generation spec: {spec}")
        return ThreeDGenerationBenchmarkGenerator(weights_path=spec.weights_path, meta_path=spec.meta_path, device=device)
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
                        "item_id": f"{item.benchmark}:{item_index:04d}:{_stable_hash(item.prompt)[:12]}",
                        "item_index": item_index,
                        "prompt": item.prompt,
                        "prompt_hash": _stable_hash(item.prompt),
                        "reference_text": item.reference_text,
                        "reference_hash": _stable_hash(item.reference_text),
                        "reference_extracted": item.reference_extracted,
                        "prediction": prediction,
                        "prediction_hash": _stable_hash(prediction),
                        "prediction_extracted": extracted,
                        "exact": exact,
                        "is_exact": bool(exact >= 1.0),
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
        benchmark_stats = {name: _binary_metric_stats(values) for name, values in per_benchmark_exact.items()}
        overall_values = [value for values in per_benchmark_exact.values() for value in values]
        overall_stats = _binary_metric_stats(overall_values)
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        summary_rows.append(
            {
                "model": spec.name,
                "family": spec.family,
                "overall_exact": _overall_exact_score(benchmark_scores),
                "overall_sample_exact": overall_stats["exact"],
                "overall_correct": overall_stats["correct"],
                "overall_total": overall_stats["total"],
                "overall_ci95_low": overall_stats["ci95_low"],
                "overall_ci95_high": overall_stats["ci95_high"],
                "avg_token_f1": float(sum(token_scores) / max(1, len(token_scores))),
                "avg_char_similarity": float(sum(char_scores) / max(1, len(char_scores))),
                "avg_gen_seconds": float(sum(gen_seconds) / max(1, len(gen_seconds))),
                "model_seconds": float(time.time() - start_model),
                "benchmarks": benchmark_scores,
                "benchmark_stats": benchmark_stats,
                "finished_at": finished_at,
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
    fieldnames = [
        "model",
        "family",
        "overall_exact",
        "overall_sample_exact",
        "overall_correct",
        "overall_total",
        "overall_ci95_low",
        "overall_ci95_high",
        "avg_token_f1",
        "avg_char_similarity",
        "avg_gen_seconds",
        "model_seconds",
    ] + list(benchmark_names)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {
                "model": row["model"],
                "family": row["family"],
                "overall_exact": f"{float(row['overall_exact']):.6f}",
                "overall_sample_exact": f"{float(row.get('overall_sample_exact', row['overall_exact'])):.6f}",
                "overall_correct": row.get("overall_correct", ""),
                "overall_total": row.get("overall_total", ""),
                "overall_ci95_low": f"{float(row.get('overall_ci95_low', 0.0)):.6f}",
                "overall_ci95_high": f"{float(row.get('overall_ci95_high', 0.0)):.6f}",
                "avg_token_f1": f"{float(row['avg_token_f1']):.6f}",
                "avg_char_similarity": f"{float(row['avg_char_similarity']):.6f}",
                "avg_gen_seconds": f"{float(row['avg_gen_seconds']):.6f}",
                "model_seconds": f"{float(row['model_seconds']):.6f}",
            }
            payload.update({name: f"{float(row['benchmarks'].get(name, 0.0)):.6f}" for name in benchmark_names})
            writer.writerow(payload)


def _write_markdown_report(
    path: Path,
    *,
    summary_rows: Sequence[Dict[str, object]],
    benchmark_names: Sequence[str],
    skipped: Sequence[Dict[str, str]],
    artifacts: Dict[str, str],
) -> None:
    lines = [
        "# Supermix Common Benchmark Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "| Rank | Model | Family | Overall | 95% CI | Exact | Avg sec |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(summary_rows, start=1):
        low = float(row.get("overall_ci95_low", 0.0))
        high = float(row.get("overall_ci95_high", 0.0))
        exact_count = f"{row.get('overall_correct', '')}/{row.get('overall_total', '')}"
        lines.append(
            "| "
            f"{rank} | {row.get('model', '')} | {row.get('family', '')} | "
            f"{float(row.get('overall_exact', 0.0)):.4f} | {low:.4f}-{high:.4f} | "
            f"{exact_count} | {float(row.get('avg_gen_seconds', 0.0)):.3f} |"
        )

    lines.extend(["", "## Per-Benchmark Exact", ""])
    header = "| Model | " + " | ".join(benchmark_names) + " |"
    divider = "| --- | " + " | ".join("---:" for _ in benchmark_names) + " |"
    lines.extend([header, divider])
    for row in summary_rows:
        stats = row.get("benchmark_stats") if isinstance(row.get("benchmark_stats"), dict) else {}
        cells = []
        for name in benchmark_names:
            stat = stats.get(name) if isinstance(stats, dict) else None
            if isinstance(stat, dict):
                cells.append(f"{float(stat.get('exact', 0.0)):.3f} ({stat.get('correct', 0)}/{stat.get('total', 0)})")
            else:
                cells.append(f"{float(row.get('benchmarks', {}).get(name, 0.0)):.3f}")
        lines.append(f"| {row.get('model', '')} | " + " | ".join(cells) + " |")

    if skipped:
        lines.extend(["", "## Skipped Models", "", "| Model | Reason |", "| --- | --- |"])
        for item in skipped:
            lines.append(f"| {item.get('name', '')} | {str(item.get('reason', '')).replace('|', '/') } |")

    lines.extend(["", "## Artifacts", ""])
    for label, artifact_path in artifacts.items():
        lines.append(f"- {label}: `{artifact_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _family_color(family: str) -> str:
    if family == "qwen":
        return "#d97706"
    if family == "native_image":
        return "#15803d"
    if family == "fusion":
        return "#db2777"
    if family == "protein":
        return "#7c3aed"
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
    ci_low = [float(row.get("overall_ci95_low", score)) for row, score in zip(summary_rows, exacts)]
    ci_high = [float(row.get("overall_ci95_high", score)) for row, score in zip(summary_rows, exacts)]
    ci_xerr = [
        [max(0.0, score - low) for score, low in zip(exacts, ci_low)],
        [max(0.0, high - score) for score, high in zip(exacts, ci_high)],
    ]
    ax_bar.errorbar(exacts, y_pos, xerr=ci_xerr, fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2, alpha=0.75)
    ax_bar.set_title("Overall Exact Accuracy")
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(model_names)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0.0, max(0.25, max(exacts) * 1.15 if exacts else 0.25))
    ax_bar.set_xlabel("Mean exact score")
    for yi, score in zip(y_pos, exacts):
        ax_bar.text(score + 0.005, yi, f"{score:.3f}", va="center", fontsize=8)

    legend_handles = []
    for family in ("qwen", "champion", "native_image", "protein", "fusion"):
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
    parser.add_argument("--local_output_root", default="output")
    parser.add_argument("--sample_per_benchmark", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260327)
    parser.add_argument("--log_every", type=int, default=12)
    parser.add_argument("--include_qwen_base", action="store_true")
    parser.add_argument("--model_name", action="append", default=[], help="Benchmark only the named discovered model. Repeat for multiple models.")
    parser.add_argument("--benchmark_name", action="append", default=[], help="Benchmark only the named benchmark suite. Repeat for multiple suites.")
    parser.add_argument("--list_models", action="store_true", help="Only discover models and write a discovery report; do not load benchmark datasets.")
    args = parser.parse_args()

    persist_root = Path(args.persist_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    local_output_root = Path(args.local_output_root).resolve() if str(args.local_output_root).strip() else None
    models, skipped = discover_models(persist_root, include_qwen_base=bool(args.include_qwen_base), local_output_root=local_output_root)
    models, skipped = _filter_models(models, skipped, requested=args.model_name)
    discovery_path = output_dir / "benchmark_model_discovery.json"
    discovery_payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "persist_root": str(persist_root),
        "local_output_root": str(local_output_root) if local_output_root else "",
        "requested_models": list(args.model_name or []),
        "models": [_model_spec_payload(model) for model in models],
        "skipped_models": skipped,
    }
    discovery_path.write_text(json.dumps(discovery_payload, indent=2), encoding="utf-8")
    if args.list_models:
        print(f"[bench] discovered {len(models)} model(s); wrote {discovery_path}")
        for model in models:
            print(f"[bench] model {model.name} family={model.family} kind={model.kind}")
        return 0
    if not models:
        raise RuntimeError("No benchmarkable models were selected.")

    items = build_benchmark_items(sample_per_benchmark=int(args.sample_per_benchmark), seed=int(args.seed))
    requested_benchmarks = {str(name).strip().lower() for name in (args.benchmark_name or []) if str(name).strip()}
    if requested_benchmarks:
        items = [item for item in items if item.benchmark.lower() in requested_benchmarks]
        missing_benchmarks = sorted(requested_benchmarks - {item.benchmark.lower() for item in items})
        if missing_benchmarks:
            raise RuntimeError(f"Requested benchmark suite(s) produced no items: {', '.join(missing_benchmarks)}")
    benchmark_names = sorted({item.benchmark for item in items})

    prompts_jsonl = output_dir / "benchmark_items.jsonl"
    _write_jsonl(
        prompts_jsonl,
        (
            {
                "item_id": f"{item.benchmark}:{index:04d}:{_stable_hash(item.prompt)[:12]}",
                "item_index": index,
                "benchmark": item.benchmark,
                "prompt": item.prompt,
                "prompt_hash": _stable_hash(item.prompt),
                "reference_text": item.reference_text,
                "reference_hash": _stable_hash(item.reference_text),
                "reference_extracted": item.reference_extracted,
                "max_new_tokens": item.max_new_tokens,
            }
            for index, item in enumerate(items, start=1)
        ),
    )

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
    report_path = output_dir / "benchmark_all_models_common_report.md"

    _write_jsonl(details_path, details)
    _write_csv(csv_path, summary_rows, benchmark_names)
    artifacts = {
        "discovery_json": str(discovery_path),
        "prompts_jsonl": str(prompts_jsonl),
        "details_jsonl": str(details_path),
        "table_csv": str(csv_path),
        "graph_png": str(graph_path),
        "report_md": str(report_path),
    }
    _write_markdown_report(
        report_path,
        summary_rows=summary_rows,
        benchmark_names=benchmark_names,
        skipped=skipped,
        artifacts=artifacts,
    )
    summary_path.write_text(
        json.dumps(
            {
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "persist_root": str(persist_root),
                "local_output_root": str(local_output_root) if local_output_root else "",
                "output_dir": str(output_dir),
                "device": str(args.device),
                "sample_per_benchmark": int(args.sample_per_benchmark),
                "benchmarks": benchmark_names,
                "models_benchmarked": [row["model"] for row in summary_rows],
                "skipped_models": skipped,
                "summary_rows": summary_rows,
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[bench] wrote summary to {summary_path}")
    print(f"[bench] wrote graph to {graph_path}")
    print(f"[bench] wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
