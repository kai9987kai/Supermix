import argparse
import gc
import hashlib
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
try:
    from peft.tuners.lora.dora import DoraLinearLayer, dequantize_module_weight, transpose
except Exception:
    DoraLinearLayer = None
    dequantize_module_weight = None
    transpose = None
from peft.utils.save_and_load import set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat_pipeline import (
    MODEL_CLASSES,
    build_context,
    choose_bucket_from_logits,
    pick_response,
    resolve_feature_mode,
    text_to_model_input,
)
from device_utils import configure_torch_runtime, resolve_device as resolve_runtime_device
from model_variants import (
    build_model,
    detect_large_head_expansion_dim,
    detect_model_size_from_state_dict,
    detect_xlarge_aux_expansion_dim,
    detect_xxlarge_third_expansion_dim,
    detect_xxxlarge_fourth_expansion_dim,
    detect_ultralarge_fifth_expansion_dim,
    detect_megalarge_sixth_expansion_dim,
    load_weights_for_model,
)
from run import safe_load_state_dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


@dataclass
class ChatPair:
    user: str
    assistant: str
    source: str = "dataset"
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class PreferencePair:
    user: str
    chosen: str
    rejected: str
    weight: float = 1.0
    quality_gap: float = 0.0
    rejected_similarity: float = 0.0
    prompt_complexity: float = 0.0
    selection_score: float = 0.0
    conversation_score: float = 0.0
    reasoning_score: float = 0.0
    creativity_score: float = 0.0
    knowledge_density_score: float = 0.0
    is_followup: bool = False


@dataclass
class PairAlignmentMetrics:
    conversation: float = 0.0
    reasoning: float = 0.0
    creativity: float = 0.0
    clarification: float = 0.0
    constraint: float = 0.0
    is_ambiguous: bool = False
    is_followup: bool = False


@dataclass
class ResumeCheckpoint:
    adapter_dir: Path
    stage: str
    sft_steps: int = 0
    preference_steps: int = 0
    sft_loss_mean: float = 0.0
    preference_loss_mean: float = 0.0


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_chat_pair_metadata(raw_metadata: Optional[Dict[str, object]]) -> Dict[str, object]:
    if not isinstance(raw_metadata, dict):
        return {}
    out: Dict[str, object] = {}
    for raw_key, raw_value in raw_metadata.items():
        key = _coerce_text(raw_key)
        if not key:
            continue
        if isinstance(raw_value, str):
            value = raw_value.strip()
            if value:
                out[key] = value
        elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
            out[key] = raw_value
    return out


def _extract_chat_pair_metadata(record: Dict[str, object]) -> Dict[str, object]:
    raw_meta: Dict[str, object] = {}
    for raw_key, raw_value in record.items():
        key = _coerce_text(raw_key)
        if not key or key in {"user", "assistant", "messages"}:
            continue
        if key == "source":
            key = "record_source"
        raw_meta[key] = raw_value
    return _normalize_chat_pair_metadata(raw_meta)


def _pairs_from_messages(
    messages: Sequence[Dict[str, object]],
    source: str = "dataset",
    metadata: Optional[Dict[str, object]] = None,
) -> List[ChatPair]:
    out: List[ChatPair] = []
    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    pair_metadata = _normalize_chat_pair_metadata(metadata)
    for msg in messages:
        role = _coerce_text(msg.get("role", "")).lower()
        text = _coerce_text(msg.get("content", msg.get("text")))
        if not text:
            continue
        if role == "user":
            pending_user = text
            continue
        if role == "assistant" and pending_user:
            user_text = pending_user
            if history:
                max_turns = 5 if (FOLLOWUP_HINT_RE.search(pending_user) or len(pending_user.split()) <= 12) else 4
                user_text = build_context(
                    history=history[-max_turns:],
                    user_text=pending_user,
                    max_turns=max_turns,
                )
            out.append(
                ChatPair(
                    user=user_text,
                    assistant=text,
                    source=source,
                    metadata=dict(pair_metadata),
                )
            )
            history.append((pending_user, text))
            pending_user = None
    return out


ARTIFACT_TAG_RE = re.compile(
    r"\[[^\]\n]*(?:variant|worked solution|set\d+|reflective|counterexample|debug|planning|mentor|teaching)[^\]\n]*\]",
    flags=re.IGNORECASE,
)
SYNTHETIC_SET_TOKEN_RE = re.compile(r"\b[a-z]+(?:-[a-z]+){0,3}-set\d+\b", flags=re.IGNORECASE)
GENRE_VARIANT_PAREN_RE = re.compile(
    r"\(\s*[^()\n]{0,120}\bgenre variant\b[^()\n]{0,80}\)",
    flags=re.IGNORECASE,
)
ASSISTANT_STYLE_LEAD_RE = re.compile(
    r"^(?:start with|try|using|use|let's use|take|consider|from|build)\s+(?:an?\s+)?"
    r"[a-z][a-z0-9 \-]{0,72}\s+(?:frame|lens|angle|comparison|walkthrough|interpretation|thought experiment|"
    r"mental model|breakdown|analysis|approach|explanation)(?:\s+for [^:]{0,72})?\s*[:\-]\s*",
    flags=re.IGNORECASE,
)
SYNTHETIC_META_PHRASES = (
    "debate framing",
    "genre variant",
    "worked solution",
    "this socratic seed trains",
    "this variant adds",
    "for beginners and advanced learners",
)
LEAD_ARTIFACT_PHRASES = (
    "let me reason through this carefully.",
    "let me work through this step by step.",
    "let me work through this carefully.",
    "solve this step by step:",
    "walk me through the solution:",
)
LOW_QUALITY_ASSISTANT_SNIPPETS = (
    "angle for problem-solving",
    "for beginners and advanced learners",
    "core mechanismbeh",
    "mechanismbehordes",
    "rests on several pillars",
    "that said, a thoughtful discussion",
    "this socratic seed trains",
    "this variant adds",
)
PLACEHOLDER_BRACKET_RESPONSE_RE = re.compile(r"^\[[^\]\n]{12,260}\]$", flags=re.IGNORECASE)
PLACEHOLDER_ASSISTANT_SNIPPETS = (
    "bot responding in a",
    "assistant responding in a",
    "addressing the user's situation directly",
    "offering relevant engagement or assistance",
    "the assistant should",
    "assistant should respond",
    "placeholder response",
    "insert response here",
)
SYNTHETIC_PROMPT_RE = re.compile(
    r"\bin a [a-z][a-z0-9 \-]{2,48}\b(?:worksheet|scenario|simulation|drill|exercise|study set|training|lab)\b",
    flags=re.IGNORECASE,
)
SMART_USER_KEYWORDS = (
    "why",
    "how",
    "explain",
    "derive",
    "proof",
    "analyze",
    "compare",
    "tradeoff",
    "algorithm",
    "complexity",
    "optimize",
    "debug",
    "python",
    "javascript",
    "sql",
    "physics",
    "chemistry",
    "biology",
    "statistics",
    "probability",
    "calculus",
)
CODING_PROMPT_KEYWORDS = (
    "python",
    "javascript",
    "typescript",
    "java",
    "c++",
    "c#",
    "rust",
    "golang",
    "sql",
    "regex",
    "debug",
    "bug",
    "algorithm",
    "leetcode",
    "complexity",
    "big-o",
)
REASONING_PROMPT_KEYWORDS = (
    "explain",
    "reason",
    "why",
    "derive",
    "proof",
    "tradeoff",
    "analyze",
    "compare",
    "counterexample",
    "step by step",
    "formal",
    "logic",
    "math",
    "probability",
    "statistics",
    "physics",
)
FINAL_ANSWER_HINTS = (
    "final answer",
    "answer:",
    "thus",
    "therefore",
    "so the answer",
    "hence",
)
CODE_BUG_OP_FLIPS = (
    ("<=", "<"),
    (">=", ">"),
    ("==", "!="),
    ("!=", "=="),
    (" and ", " or "),
    (" or ", " and "),
)
LEGACY_NESTED_ADAPTER_PREFIX = "base_model.model.base_model.model.model."
NORMAL_ADAPTER_PREFIX = "base_model.model.model."
NESTED_PREFIX_FRAGMENT = "base_model.model.base_model.model."
CONTEXT_ROLE_LINE_RE = re.compile(r"^(user|assistant)\s*:\s*(.*)$", flags=re.IGNORECASE)
FOLLOWUP_HINT_RE = re.compile(
    r"\b(it|that|this|they|them|same|again|continue|deeper|expand|shorter|longer|rewrite|rephrase|refine|improve|make it|more like)\b",
    flags=re.IGNORECASE,
)
AMBIGUOUS_EDIT_RE = re.compile(
    r"\b(make it better|improve it|fix this|change it|same but better|more like that|do that again)\b",
    flags=re.IGNORECASE,
)
SHORTEN_HINT_RE = re.compile(
    r"\b(shorter|short version|brief|concise|trim|compress|tldr|one sentence|summarize)\b",
    flags=re.IGNORECASE,
)
EXPAND_HINT_RE = re.compile(
    r"\b(expand|elaborate|deeper|more detail|step by step|walk through|longer|explain more|unpack)\b",
    flags=re.IGNORECASE,
)
REWRITE_HINT_RE = re.compile(
    r"\b(rewrite|rephrase|say it better|clearer|fix the wording|make it clearer|polish)\b",
    flags=re.IGNORECASE,
)
CREATIVE_REQUEST_RE = re.compile(
    r"\b(creative|story|brainstorm|metaphor|analogy|invent|novel|poem|vivid|imaginative)\b",
    flags=re.IGNORECASE,
)
REASONING_REQUEST_RE = re.compile(
    r"\b(explain|why|how|analy[sz]e|reason|derive|prove|tradeoff|step by step|debug)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_TARGET_RE = re.compile(r"`[^`]{2,}`|\"[^\"]{2,}\"|'[^']{2,}'")
CLARIFICATION_RESPONSE_RE = re.compile(
    r"\b(which part|what part|what exactly|what do you want me to|which answer|paste the text|share the text|what topic|what are you referring to|what would you like)\b",
    flags=re.IGNORECASE,
)
SIGNIFICANT_TOKEN_RE = re.compile(r"[a-z0-9_+\-']+")
NUMBER_RE = re.compile(r"\b-?\d+(?:\.\d+)?\b")
STOPWORDS = {
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
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "to",
    "us",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}
KNOWLEDGE_STRUCTURE_RE = re.compile(
    r"(^|\n)\s*(?:[-*]|\d+[.)]|step\s+\d+|example:|definition:|formula:|algorithm:|key idea:|note:|code:)",
    flags=re.IGNORECASE,
)
KNOWLEDGE_DENSE_TOKEN_RE = re.compile(
    r"`[^`]+`|\b[a-z_][a-z0-9_]*\(|\b[a-z]+[0-9][a-z0-9_]*\b|\b(?:api|fft|json|sql|http|https|cuda|gpu|lora|qwen|python|torch|transformer)\b",
    flags=re.IGNORECASE,
)
LOW_DENSITY_FILLER_PHRASES = (
    "let me know if you want",
    "i hope this helps",
    "feel free to ask",
    "generally speaking",
    "it depends",
    "useful and important",
    "really helpful",
    "in many ways",
)


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_leading_bracket_tags(text: str) -> str:
    out = text.strip()
    for _ in range(4):
        nxt = re.sub(r"^\s*(\[[^\]\n]{1,90}\]\s*)+", "", out).strip()
        if nxt == out:
            break
        out = nxt
    return out


def _looks_like_placeholder_assistant(text: str) -> bool:
    t = _normalize_whitespace(_coerce_text(text))
    if not t:
        return False
    low = t.lower()

    for snippet in PLACEHOLDER_ASSISTANT_SNIPPETS:
        if snippet in low:
            return True

    if low.startswith("assistant should ") or low.startswith("the assistant should "):
        return True

    if ("<assistant" in low and "response" in low) or ("[assistant" in low and "response" in low):
        return True

    if PLACEHOLDER_BRACKET_RESPONSE_RE.fullmatch(t):
        cue_words = (
            "bot ",
            "assistant ",
            "respond",
            "manner",
            "tone",
            "style",
            "addressing the user's situation",
            "offering relevant",
            "regarding ",
        )
        cue_hits = sum(1 for cue in cue_words if cue in low)
        if cue_hits >= 2:
            return True
    return False


def _synthetic_artifact_hits(text: str) -> int:
    t = _normalize_whitespace(_coerce_text(text))
    if not t:
        return 0
    low = t.lower()
    hits = len(SYNTHETIC_SET_TOKEN_RE.findall(t))
    hits += sum(1 for phrase in SYNTHETIC_META_PHRASES if phrase in low)
    if ASSISTANT_STYLE_LEAD_RE.match(t):
        hits += 1
    if "rests on several pillars" in low and ("the for case" in low or "the against case" in low):
        hits += 1
    return hits


def _latest_user_text(text: str) -> str:
    raw = _normalize_whitespace(_coerce_text(text))
    if not raw:
        return ""
    latest = ""
    for line in raw.splitlines():
        match = CONTEXT_ROLE_LINE_RE.match(line.strip())
        if match and str(match.group(1)).strip().lower() == "user":
            latest = str(match.group(2)).strip()
    return latest or raw


def _is_short_answer_prompt(user_text: str) -> bool:
    low = _latest_user_text(user_text).lower()
    return (
        "just the answer" in low
        or "one word" in low
        or "true or false" in low
        or "yes or no" in low
    )


def _preference_stop_target_words(user_text: str) -> int:
    low = _latest_user_text(user_text).lower()
    if not low:
        return 0
    if "one word" in low:
        return 2
    if "yes or no" in low or "true or false" in low:
        return 3
    if "one sentence" in low:
        return 18
    if any(
        phrase in low
        for phrase in (
            "just the answer",
            "answer only",
            "only answer",
            "output only",
            "just output",
            "no explanation",
            "without explanation",
        )
    ):
        return 12
    if SHORTEN_HINT_RE.search(low) or any(
        phrase in low
        for phrase in ("briefly", "succinct", "keep it short", "as short as possible", "be concise")
    ):
        return 40
    return 0


def _preference_stop_alignment_score(
    user_text: str,
    assistant_text: str,
    strength: float = 1.0,
) -> float:
    scale = max(0.0, float(strength))
    if scale <= 0.0:
        return 0.0
    target_words = _preference_stop_target_words(user_text)
    text = _fast_cleanup_response_text(assistant_text)
    if target_words <= 0:
        return 0.0
    if not text:
        return float(-0.35 * scale)

    words = max(1, _word_token_count(text))
    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    sentence_count = max(1, len(sentences))
    paragraph_count = max(1, len([ln for ln in text.splitlines() if ln.strip()]))
    low = text.lower()

    bonus = 0.0
    penalty = 0.0
    if words <= target_words:
        bonus += min(0.18, 0.02 * float(target_words - words + 1))
    else:
        excess_ratio = float(words - target_words) / float(max(1, target_words))
        penalty += 0.22 * min(1.8, excess_ratio)

    if target_words <= 4 and sentence_count > 1:
        penalty += 0.28
    elif target_words <= 12 and sentence_count > 1:
        penalty += 0.12 * min(2.0, float(sentence_count - 1))
    elif target_words <= 18 and sentence_count > 2:
        penalty += 0.08 * min(2.0, float(sentence_count - 2))

    if paragraph_count > 1:
        penalty += 0.12
    if "```" in text and target_words <= 18:
        penalty += 0.18
    filler_hits = sum(1 for phrase in LOW_DENSITY_FILLER_PHRASES if phrase in low)
    if filler_hits > 0:
        penalty += 0.05 * min(3, filler_hits)

    return float(scale * max(-0.75, min(0.20, bonus - penalty)))


def _is_synthetic_template_prompt(user_text: str) -> bool:
    text = _normalize_whitespace(_latest_user_text(user_text))
    if not text:
        return False
    low = text.lower()
    if low.startswith("solve this step by step: in a "):
        return True
    if low.startswith("walk me through the solution: in a "):
        return True
    if low.startswith("in a ") and (" worksheet" in low or " scenario" in low):
        return True
    if SYNTHETIC_PROMPT_RE.search(low):
        return True
    if SYNTHETIC_SET_TOKEN_RE.search(text):
        return True
    if any(phrase in low for phrase in ("debate framing", "genre variant", "worked solution")):
        return True
    return False


def _word_token_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", _coerce_text(text).lower()))


def _prompt_signature(text: str) -> str:
    low = _normalize_whitespace(_latest_user_text(text).lower())
    low = re.sub(
        r"\bin a [a-z0-9 \-]{2,48}\b(?= (?:worksheet|scenario|simulation|drill|exercise|study set|training|lab))",
        "in a <domain>",
        low,
    )
    low = SYNTHETIC_SET_TOKEN_RE.sub(" <settag> ", low)
    low = GENRE_VARIANT_PAREN_RE.sub(" ", low)
    low = re.sub(r"\b(?:debate framing|genre variant|worked solution)\b", " ", low, flags=re.IGNORECASE)
    low = re.sub(r"\d+", " <num> ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low[:220]


def _replace_last_user_turn(prompt_text: str, new_latest_user: str) -> str:
    raw = _normalize_whitespace(_coerce_text(prompt_text))
    new_latest_user = _normalize_whitespace(_coerce_text(new_latest_user))
    if not raw or not new_latest_user:
        return new_latest_user or raw
    lines = raw.splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        match = CONTEXT_ROLE_LINE_RE.match(lines[idx].strip())
        if match and str(match.group(1)).strip().lower() == "user":
            lines[idx] = f"User: {new_latest_user}"
            return "\n".join(lines)
    return new_latest_user


def _followup_paraphrase_variants(user_text: str, max_variants: int) -> List[str]:
    limit = max(0, int(max_variants))
    if limit <= 0:
        return []

    latest_user = _latest_user_text(user_text)
    if not latest_user:
        return []
    low = latest_user.lower()

    variants: List[str] = []
    if SHORTEN_HINT_RE.search(low) and REWRITE_HINT_RE.search(low):
        variants.append("Give me a shorter, clearer version.")
    if EXPAND_HINT_RE.search(low) and REASONING_REQUEST_RE.search(low):
        variants.append("Go deeper and walk through it step by step.")
    if SHORTEN_HINT_RE.search(low):
        variants.extend(
            [
                "Give me a shorter version.",
                "Can you make that more concise?",
            ]
        )
    if EXPAND_HINT_RE.search(low):
        variants.extend(
            [
                "Go deeper on that.",
                "Can you expand on that with more detail?",
            ]
        )
    if REWRITE_HINT_RE.search(low):
        variants.extend(
            [
                "Rephrase that more clearly.",
                "Say the same thing, but clearer.",
            ]
        )
    if CREATIVE_REQUEST_RE.search(low):
        variants.extend(
            [
                "Make it more vivid and creative.",
                "Give it a more imaginative spin.",
            ]
        )
    if REASONING_REQUEST_RE.search(low):
        variants.extend(
            [
                "Explain it step by step.",
                "Walk me through the reasoning.",
            ]
        )
    if FOLLOWUP_HINT_RE.search(low) and not variants:
        variants.extend(
            [
                "Continue from the last answer.",
                "Build on what you just said.",
            ]
        )

    out: List[str] = []
    seen: set = set()
    base_key = " ".join(latest_user.lower().split())
    for variant in variants:
        cleaned = _normalize_whitespace(variant)
        key = " ".join(cleaned.lower().split())
        if not cleaned or key == base_key or key in seen:
            continue
        seen.add(key)
        out.append(_replace_last_user_turn(user_text, cleaned))
        if len(out) >= limit:
            break
    return out


def _clean_training_text(text: str, is_user: bool) -> str:
    out = _normalize_whitespace(_coerce_text(text))
    if not out:
        return ""

    raw_artifact_hits = _synthetic_artifact_hits(out)
    out = ARTIFACT_TAG_RE.sub(" ", out)
    out = _strip_leading_bracket_tags(out)
    out = GENRE_VARIANT_PAREN_RE.sub(" ", out)
    if not is_user:
        out = SYNTHETIC_SET_TOKEN_RE.sub(" ", out)
        out = re.sub(r"\b(?:debate framing|genre variant|worked solution)\b", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\b(?:this socratic seed trains|this variant adds)\b[^.?!]*[.?!]?", " ", out, flags=re.IGNORECASE)
        out = re.sub(r"\bfor beginners and advanced learners\b", " ", out, flags=re.IGNORECASE)
        if raw_artifact_hits > 0:
            for _ in range(3):
                nxt = ASSISTANT_STYLE_LEAD_RE.sub("", out, count=1).strip()
                if nxt == out:
                    break
                out = nxt
    if not is_user:
        lowered = out.lower()
        changed = True
        while changed:
            changed = False
            for phrase in LEAD_ARTIFACT_PHRASES:
                if lowered.startswith(phrase):
                    out = out[len(phrase) :].lstrip(" :-\n")
                    lowered = out.lower()
                    changed = True
    out = _normalize_whitespace(out)
    return out


def _fast_cleanup_response_text(text: str) -> str:
    out = _normalize_whitespace(_coerce_text(text))
    if not out:
        return ""
    out = ARTIFACT_TAG_RE.sub(" ", out)
    out = _strip_leading_bracket_tags(out)
    out = GENRE_VARIANT_PAREN_RE.sub(" ", out)
    out = SYNTHETIC_SET_TOKEN_RE.sub(" ", out)
    out = re.sub(r"\b(?:debate framing|genre variant|worked solution)\b", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\b(?:this socratic seed trains|this variant adds)\b[^.?!]*[.?!]?", " ", out, flags=re.IGNORECASE)
    out = re.sub(r"\bfor beginners and advanced learners\b", " ", out, flags=re.IGNORECASE)
    for _ in range(3):
        nxt = ASSISTANT_STYLE_LEAD_RE.sub("", out, count=1).strip()
        if nxt == out:
            break
        out = nxt
    return _normalize_whitespace(out)


def _is_quality_pair(user_text: str, assistant_text: str, min_chars: int) -> bool:
    if len(user_text) < min_chars:
        return False
    if _looks_like_placeholder_assistant(assistant_text):
        return False
    short_answer_allowed = _is_short_answer_prompt(user_text)
    if len(assistant_text) < max(8, min_chars) and not short_answer_allowed:
        return False
    if assistant_text.count("[") + assistant_text.count("]") > 4:
        return False
    if ARTIFACT_TAG_RE.search(assistant_text):
        return False
    if _synthetic_artifact_hits(assistant_text) > 0:
        return False
    low = assistant_text.lower()
    for snippet in LOW_QUALITY_ASSISTANT_SNIPPETS:
        if snippet in low:
            return False
    if SequenceMatcher(None, user_text.lower(), assistant_text.lower()).ratio() > 0.92:
        return False
    tokens = assistant_text.lower().split()
    if len(tokens) >= 10:
        uniq_ratio = len(set(tokens)) / float(len(tokens))
        if uniq_ratio < 0.28:
            return False
    return True


def _prompt_complexity_score(user_text: str) -> float:
    text = _latest_user_text(user_text)
    if not text:
        return 0.0
    low = text.lower()
    tokens = re.findall(r"[a-z0-9_']+", low)
    score = min(2.0, float(len(tokens)) / 18.0)
    if "?" in text:
        score += 0.25
    if any(ch.isdigit() for ch in text):
        score += 0.35
    if "```" in text or "def " in low or "class " in low:
        score += 0.60
    keyword_hits = 0
    for kw in SMART_USER_KEYWORDS:
        if kw in low:
            keyword_hits += 1
    score += min(2.0, 0.35 * float(keyword_hits))
    if len(text) > 350:
        score += 0.35
    if _is_synthetic_template_prompt(text):
        score -= 0.80
    return float(score)


def _looks_like_coding_prompt(text: str) -> bool:
    low = _latest_user_text(text).lower()
    if not low:
        return False
    if "```" in low:
        return True
    return any(k in low for k in CODING_PROMPT_KEYWORDS)


def _looks_like_reasoning_prompt(text: str) -> bool:
    low = _latest_user_text(text).lower()
    if not low:
        return False
    hits = sum(1 for k in REASONING_PROMPT_KEYWORDS if k in low)
    return hits >= 1


def _iter_clean_pairs_from_jsonl(
    path: Path,
    min_chars: int,
    seen: set,
    stats: Dict[str, int],
) -> Iterator[ChatPair]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue

            local_pairs: List[ChatPair] = []
            pair_metadata = _extract_chat_pair_metadata(record)
            user = _coerce_text(record.get("user"))
            assistant = _coerce_text(record.get("assistant"))
            if user and assistant:
                local_pairs.append(
                    ChatPair(
                        user=user,
                        assistant=assistant,
                        source=path.name,
                        metadata=dict(pair_metadata),
                    )
                )
            elif isinstance(record.get("messages"), list):
                local_pairs.extend(
                    _pairs_from_messages(
                        record["messages"],
                        source=path.name,
                        metadata=pair_metadata,
                    )
                )

            for pair in local_pairs:
                stats["raw"] += 1
                user_text = _clean_training_text(pair.user, is_user=True)
                assistant_text = _clean_training_text(pair.assistant, is_user=False)
                if not user_text or not assistant_text:
                    stats["empty_after_clean"] += 1
                    continue
                if _looks_like_placeholder_assistant(assistant_text):
                    stats["filtered_placeholder"] += 1
                    continue
                if not _is_quality_pair(user_text, assistant_text, min_chars=min_chars):
                    stats["filtered_quality"] += 1
                    continue
                key = (user_text, assistant_text)
                if key in seen:
                    stats["deduped"] += 1
                    continue
                seen.add(key)
                stats["kept"] += 1
                yield ChatPair(
                    user=user_text,
                    assistant=assistant_text,
                    source=path.name,
                    metadata=dict(pair.metadata),
                )


def load_jsonl_pairs(
    paths: Sequence[str],
    max_records: int,
    min_chars: int = 4,
    max_source_fraction: float = 0.0,
    max_synthetic_fraction: float = 0.0,
    max_prompt_signature_count: int = 0,
    prompt_signature_cap_exempt_sources: Optional[Sequence[str]] = None,
    log_every_records: int = 5000,
) -> List[ChatPair]:
    pairs: List[ChatPair] = []
    seen: set = set()
    stats = {
        "raw": 0,
        "kept": 0,
        "empty_after_clean": 0,
        "filtered_placeholder": 0,
        "filtered_quality": 0,
        "deduped": 0,
        "filtered_source_cap": 0,
        "filtered_synthetic_cap": 0,
        "filtered_prompt_signature": 0,
        "cap_relaxations": 0,
    }
    per_source_kept: Dict[str, int] = {}
    prompt_sig_kept: Dict[str, int] = {}
    synthetic_kept = 0
    exempt_sources = set()
    for src in prompt_signature_cap_exempt_sources or []:
        s = str(src or "").strip().lower()
        if s:
            exempt_sources.add(s)
    path_objs: List[Path] = []
    for path_str in paths:
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        path_objs.append(path)

    iterators = [
        _iter_clean_pairs_from_jsonl(
            path=path,
            min_chars=min_chars,
            seen=seen,
            stats=stats,
        )
        for path in path_objs
    ]
    active = [True for _ in iterators]
    source_cap = max(0.0, min(1.0, float(max_source_fraction)))
    synthetic_cap = max(0.0, min(1.0, float(max_synthetic_fraction)))
    prompt_sig_cap = max(0, int(max_prompt_signature_count))
    base_source_cap = source_cap
    base_synthetic_cap = synthetic_cap
    base_prompt_sig_cap = prompt_sig_cap
    log_every_records = max(0, int(log_every_records))
    started = time.time()
    source_names = ", ".join(p.name for p in path_objs)
    print(f"[data] sources={len(path_objs)} -> {source_names}")

    while len(pairs) < max_records and any(active):
        progressed = False
        for idx, it in enumerate(iterators):
            if not active[idx]:
                continue
            while True:
                try:
                    pair = next(it)
                except StopIteration:
                    active[idx] = False
                    break

                next_total = len(pairs) + 1
                src = str(pair.source)
                src_count = per_source_kept.get(src, 0)
                src_low = src.strip().lower()
                is_synth = _is_synthetic_template_prompt(pair.user)
                prompt_sig = _prompt_signature(pair.user)
                prompt_sig_count = prompt_sig_kept.get(prompt_sig, 0)

                if source_cap > 0:
                    max_src_count = max(1, int(math.floor(source_cap * float(next_total))))
                    if src_count + 1 > max_src_count:
                        stats["filtered_source_cap"] += 1
                        continue

                if synthetic_cap > 0 and is_synth:
                    max_synth_count = max(1, int(math.floor(synthetic_cap * float(next_total))))
                    if synthetic_kept + 1 > max_synth_count:
                        stats["filtered_synthetic_cap"] += 1
                        continue
                if (
                    prompt_sig_cap > 0
                    and src_low not in exempt_sources
                    and prompt_sig_count + 1 > int(prompt_sig_cap)
                ):
                    stats["filtered_prompt_signature"] += 1
                    continue

                pairs.append(pair)
                per_source_kept[src] = src_count + 1
                prompt_sig_kept[prompt_sig] = prompt_sig_count + 1
                if is_synth:
                    synthetic_kept += 1
                if log_every_records > 0 and len(pairs) % log_every_records == 0:
                    elapsed = max(1e-6, time.time() - started)
                    print(
                        "[data] progress: "
                        f"pairs={len(pairs)}/{max_records} "
                        f"raw={stats['raw']} kept={stats['kept']} "
                        f"rate={len(pairs) / elapsed:.2f}/s"
                    )
                progressed = True
                break
            if len(pairs) >= max_records:
                break
        if not progressed:
            relaxed = False
            if source_cap > 0.0 and source_cap < 1.0:
                source_cap = min(1.0, source_cap + 0.05)
                relaxed = True
            if synthetic_cap > 0.0 and synthetic_cap < 1.0:
                synthetic_cap = min(1.0, synthetic_cap + 0.02)
                relaxed = True
            if prompt_sig_cap > 0 and prompt_sig_cap < 64:
                prompt_sig_cap += 1
                relaxed = True
            if relaxed:
                stats["cap_relaxations"] += 1
                continue
            break

    source_mix = ", ".join(f"{k}:{v}" for k, v in sorted(per_source_kept.items()))
    print(
        "[data] quality filter: "
        f"raw={stats['raw']} kept={stats['kept']} "
        f"empty={stats['empty_after_clean']} "
        f"placeholder={stats['filtered_placeholder']} "
        f"filtered={stats['filtered_quality']} "
        f"deduped={stats['deduped']} "
        f"source_cap={stats['filtered_source_cap']} "
        f"synthetic_cap={stats['filtered_synthetic_cap']} "
        f"prompt_cap={stats['filtered_prompt_signature']} "
        f"cap_relax={stats['cap_relaxations']}"
    )
    if base_source_cap > 0.0 or base_synthetic_cap > 0.0 or base_prompt_sig_cap > 0:
        print(
            "[data] cap_state: "
            f"source={base_source_cap:.2f}->{source_cap:.2f} "
            f"synthetic={base_synthetic_cap:.2f}->{synthetic_cap:.2f} "
            f"prompt_sig={base_prompt_sig_cap}->{prompt_sig_cap}"
        )
    print(f"[data] synthetic_kept={synthetic_kept}/{len(pairs)}")
    if source_mix:
        print(f"[data] source_mix: {source_mix}")
    return pairs


def _split_group_token(value: object) -> str:
    text = _coerce_text(value)
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _chat_pair_split_group_key(pair: ChatPair) -> Tuple[str, str]:
    metadata = pair.metadata if isinstance(pair.metadata, dict) else {}
    split_group = _split_group_token(metadata.get("split_group"))
    if split_group:
        return "split_group", f"split_group:{split_group}"

    event_id = _split_group_token(metadata.get("event_id"))
    if event_id:
        return "event_id", f"event_id:{event_id}"

    record_source = _split_group_token(metadata.get("record_source"))
    if record_source:
        return "record_source", f"record_source:{record_source}"

    pair_source = _split_group_token(pair.source)
    if "://" in pair_source:
        return "source_url", f"source_url:{pair_source}"

    topic = _split_group_token(metadata.get("topic"))
    as_of = _split_group_token(metadata.get("as_of"))
    if topic and as_of:
        return "topic_as_of", f"topic_as_of:{topic}|{as_of}"
    if topic:
        return "topic", f"topic:{topic}"

    return "", ""


def split_train_eval(
    pairs: List[ChatPair],
    eval_size: int,
    seed: int,
    split_mode: str = "auto",
) -> Tuple[List[ChatPair], List[ChatPair]]:
    if len(pairs) < 2:
        raise ValueError("Need at least 2 samples to split train/eval.")
    rng = random.Random(seed)
    eval_n = max(1, min(eval_size, len(pairs) - 1))
    mode = str(split_mode or "auto").strip().lower()
    if mode not in {"auto", "random"}:
        mode = "auto"

    if mode == "auto":
        grouped_indices: Dict[str, List[int]] = {}
        group_type_counts: Dict[str, int] = {}
        ungrouped_indices: List[int] = []
        for idx, pair in enumerate(pairs):
            group_type, group_key = _chat_pair_split_group_key(pair)
            if group_key:
                grouped_indices.setdefault(group_key, []).append(idx)
                group_type_counts[group_type] = group_type_counts.get(group_type, 0) + 1
            else:
                ungrouped_indices.append(idx)

        if len(grouped_indices) >= 2:
            grouped_items = list(grouped_indices.items())
            rng.shuffle(grouped_items)
            grouped_items.sort(key=lambda item: len(item[1]))
            eval_idx: set = set()
            eval_group_count = 0
            for _group_key, indices in grouped_items:
                if len(eval_idx) >= eval_n:
                    break
                if len(eval_idx) + len(indices) >= len(pairs):
                    continue
                eval_idx.update(indices)
                eval_group_count += 1

            if len(eval_idx) < eval_n and ungrouped_indices:
                rng.shuffle(ungrouped_indices)
                remaining_budget = min(eval_n - len(eval_idx), len(pairs) - len(eval_idx) - 1)
                if remaining_budget > 0:
                    eval_idx.update(ungrouped_indices[:remaining_budget])

            if eval_idx and len(eval_idx) < len(pairs):
                train = [pairs[i] for i in range(len(pairs)) if i not in eval_idx]
                eval_pairs = [pairs[i] for i in range(len(pairs)) if i in eval_idx]
                group_summary = ", ".join(
                    f"{group_type}:{count}" for group_type, count in sorted(group_type_counts.items())
                )
                print(
                    "[data] split: "
                    f"mode=auto eval={len(eval_pairs)}/{len(pairs)} "
                    f"grouped_groups={len(grouped_indices)} eval_groups={eval_group_count} "
                    f"ungrouped={len(ungrouped_indices)} "
                    f"group_types={group_summary if group_summary else '-'}"
                )
                return train, eval_pairs

    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    eval_idx = set(idx[:eval_n])
    train = [pairs[i] for i in range(len(pairs)) if i not in eval_idx]
    eval_pairs = [pairs[i] for i in range(len(pairs)) if i in eval_idx]
    print(f"[data] split: mode=random eval={len(eval_pairs)}/{len(pairs)}")
    return train, eval_pairs


def _resolve_expansion_dim(
    arg_val: Optional[int],
    meta: Dict[str, object],
    meta_key: str,
    default_val: int,
    inferred_size: str,
    allowed_sizes: set,
    detect_fn,
    sd: Dict[str, torch.Tensor],
) -> int:
    if arg_val is not None:
        return int(arg_val)
    raw = meta.get(meta_key, default_val)
    try:
        val = int(raw) if raw is not None else int(default_val)
    except Exception:
        val = int(default_val)
    if inferred_size in allowed_sizes and detect_fn is not None:
        return int(detect_fn(sd, default=val))
    return int(val)


class SupermixTeacher:
    def __init__(self, weights_path: str, meta_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        with Path(meta_path).open("r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_mode = resolve_feature_mode(str(meta.get("feature_mode", "legacy")), smarter_auto=True)
        self.buckets: Dict[int, List[Dict[str, object]]] = {}
        for k, v in meta.get("buckets", {}).items():
            try:
                label = int(k)
            except Exception:
                continue
            if isinstance(v, list):
                self.buckets[label] = v
        self.available_labels = sorted(self.buckets.keys()) or list(range(MODEL_CLASSES))

        sd = safe_load_state_dict(weights_path)
        inferred_size = detect_model_size_from_state_dict(sd)
        expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "expansion_dim",
            512,
            inferred_size,
            {"large", "xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"},
            detect_large_head_expansion_dim,
            sd,
        )
        extra_expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "extra_expansion_dim",
            1024,
            inferred_size,
            {"xlarge", "xxlarge", "xxxlarge", "ultralarge", "megalarge"},
            detect_xlarge_aux_expansion_dim,
            sd,
        )
        third_expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "third_expansion_dim",
            3072,
            inferred_size,
            {"xxlarge", "xxxlarge", "ultralarge", "megalarge"},
            detect_xxlarge_third_expansion_dim,
            sd,
        )
        fourth_expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "fourth_expansion_dim",
            4096,
            inferred_size,
            {"xxxlarge", "ultralarge", "megalarge"},
            detect_xxxlarge_fourth_expansion_dim,
            sd,
        )
        fifth_expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "fifth_expansion_dim",
            6144,
            inferred_size,
            {"ultralarge", "megalarge"},
            detect_ultralarge_fifth_expansion_dim,
            sd,
        )
        sixth_expansion_dim = _resolve_expansion_dim(
            None,
            meta,
            "sixth_expansion_dim",
            8192,
            inferred_size,
            {"megalarge"},
            detect_megalarge_sixth_expansion_dim,
            sd,
        )
        adapter_dropout = float(meta.get("adapter_dropout", 0.1))

        model = build_model(
            model_size=inferred_size,
            expansion_dim=expansion_dim,
            dropout=adapter_dropout,
            extra_expansion_dim=extra_expansion_dim,
            third_expansion_dim=third_expansion_dim,
            fourth_expansion_dim=fourth_expansion_dim,
            fifth_expansion_dim=fifth_expansion_dim,
            sixth_expansion_dim=sixth_expansion_dim,
        ).to(self.device)
        missing, unexpected = load_weights_for_model(model, sd, model_size=inferred_size)
        if missing or unexpected:
            raise RuntimeError(f"Supermix weight mismatch. missing={missing}, unexpected={unexpected}")
        self.model = model.eval()

    @torch.no_grad()
    def generate(self, user_text: str) -> str:
        candidates = self.generate_candidates(user_text, temperatures=(0.0,))
        return candidates[0] if candidates else ""

    @torch.no_grad()
    def generate_candidates(self, user_text: str, temperatures: Sequence[float]) -> List[str]:
        context = build_context(history=[], user_text=user_text, max_turns=0)
        x = text_to_model_input(context, feature_mode=self.feature_mode).to(self.device)
        logits = self.model(x)[0, 0]
        bucket = choose_bucket_from_logits(logits, self.available_labels, temperature=0.0)
        candidates = self.buckets.get(int(bucket), [])
        if not candidates:
            return []
        out: List[str] = []
        seen: set = set()
        temp_list = list(temperatures) if temperatures else [0.0]
        for temp in temp_list:
            response = pick_response(
                candidates=candidates,
                query_text=user_text,
                recent_assistant_messages=[],
                response_temperature=max(0.0, float(temp)),
                style_mode="balanced",
                creativity=0.0,
            )
            cleaned = _fast_cleanup_response_text(response)
            if not cleaned:
                continue
            key = " ".join(cleaned.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out


def _distill_candidate_temperatures(best_of: int) -> Tuple[float, ...]:
    best = max(1, int(best_of))
    preset = (0.0, 0.22, 0.45, 0.72)
    return tuple(float(x) for x in preset[:best])


def _distillation_compactness_score(
    user_text: str,
    reference_text: str,
    candidate_text: str,
) -> float:
    reference_words = max(1, _word_token_count(reference_text))
    candidate_words = max(1, _word_token_count(candidate_text))
    ratio = float(candidate_words) / float(reference_words)
    wants_reasoning = _looks_like_reasoning_prompt(user_text)
    wants_coding = _looks_like_coding_prompt(user_text)

    if candidate_words <= 8:
        return 0.42
    if candidate_words <= 16 and bool(wants_reasoning or wants_coding):
        return 0.78

    soft_limit = 1.02
    if wants_coding:
        soft_limit = 1.10
    if wants_reasoning:
        soft_limit = 1.18

    score = 1.0
    if ratio <= 0.92 and candidate_words >= 18:
        score += 0.06
    if ratio > soft_limit:
        score -= min(0.70, (ratio - soft_limit) / (1.10 if wants_reasoning else 0.85))
    if candidate_words > 150:
        score -= min(0.35, float(candidate_words - 150) / 260.0)
    if ratio < 0.55 and bool(wants_reasoning or wants_coding):
        score -= min(0.28, (0.55 - ratio) / 0.55)
    return float(max(0.30, min(1.10, score)))


def _distillation_candidate_rank(
    user_text: str,
    candidate_text: str,
    reference_text: str,
    density_bias: float,
    gain_bias: float,
    compactness_bias: float,
    source: str = "dataset",
) -> Tuple[float, float, float, Dict[str, float]]:
    quality_score, alignment = _paired_response_score(user_text, candidate_text)
    knowledge_density = _pair_knowledge_density_score(
        user_text=user_text,
        assistant_text=candidate_text,
        source=source,
    )
    reference_quality, _reference_alignment = _paired_response_score(user_text, reference_text)
    reference_density = _pair_knowledge_density_score(
        user_text=user_text,
        assistant_text=reference_text,
        source=source,
    )
    quality_gain = float(quality_score - reference_quality)
    density_gain = float(knowledge_density - reference_density)
    compactness = _distillation_compactness_score(
        user_text=user_text,
        reference_text=reference_text,
        candidate_text=candidate_text,
    )
    alignment_bonus = (
        0.08 * max(0.0, float(alignment.conversation))
        + 0.06 * max(0.0, float(alignment.reasoning))
        + 0.04 * max(0.0, float(alignment.creativity))
        + 0.04 * max(0.0, float(alignment.constraint))
    )
    base_utility = float(quality_score) + max(0.0, float(density_bias)) * float(knowledge_density)
    compactness_mix = max(0.0, min(1.0, float(compactness_bias)))
    utility_scale = (1.0 - compactness_mix) + compactness_mix * float(compactness)
    rank = (
        base_utility * utility_scale
        + max(0.0, float(gain_bias)) * float(quality_gain)
        + 0.12 * float(density_gain)
        + alignment_bonus
    )
    return float(rank), float(quality_score), float(knowledge_density), {
        "quality_gain": float(quality_gain),
        "density_gain": float(density_gain),
        "compactness": float(compactness),
        "utility_scale": float(utility_scale),
        "alignment_bonus": float(alignment_bonus),
    }


def _should_log_progress_heartbeat(
    visited: int,
    log_every: int,
    now: float,
    last_log_time: float,
    heartbeat_seconds: float,
) -> bool:
    if int(visited) <= 0:
        return False
    if int(log_every) > 0 and int(visited) % int(log_every) == 0:
        return True
    if float(heartbeat_seconds) > 0.0 and float(now - last_log_time) >= float(heartbeat_seconds):
        return True
    return False


def apply_supermix_distillation(
    train_pairs: List[ChatPair],
    teacher: SupermixTeacher,
    ratio: float,
    max_teacher_samples: int,
    seed: int,
    min_quality_score: float = 0.0,
    min_quality_gain: float = 0.0,
    skip_synthetic_prompts: bool = True,
    log_every: int = 50,
    max_seconds: float = 0.0,
    best_of: int = 1,
    density_bias: float = 0.0,
    gain_bias: float = 0.0,
    compactness_bias: float = 0.0,
    rank_margin: float = 0.0,
) -> Tuple[List[ChatPair], int]:
    if ratio <= 0 or max_teacher_samples <= 0:
        return train_pairs, 0
    n_target = int(round(len(train_pairs) * ratio))
    n_target = max(0, min(n_target, len(train_pairs), max_teacher_samples))
    if n_target == 0:
        return train_pairs, 0

    rng = random.Random(seed)
    indices = list(range(len(train_pairs)))
    rng.shuffle(indices)
    chosen = set(indices[:n_target])
    seen_keys = {(p.user, p.assistant) for p in train_pairs}
    log_every = max(0, int(log_every))
    max_seconds = max(0.0, float(max_seconds))
    started = time.time()
    print(
        "[distill] config: "
        f"target={n_target} ratio={float(ratio):.3f} "
        f"max_seconds={max_seconds if max_seconds > 0 else 'off'} "
        f"best_of={max(1, int(best_of))} "
        f"min_gain={max(0.0, float(min_quality_gain)):.3f} "
        f"density_bias={max(0.0, float(density_bias)):.3f} "
        f"gain_bias={max(0.0, float(gain_bias)):.3f} "
        f"compactness_bias={max(0.0, float(compactness_bias)):.3f} "
        f"rank_margin={max(0.0, float(rank_margin)):.3f}"
    )

    mixed = list(train_pairs)
    generated = 0
    chosen_sorted = sorted(chosen)
    visited = 0
    candidate_temperatures = _distill_candidate_temperatures(best_of=int(best_of))
    progress_heartbeat_seconds = 30.0
    last_progress_log_time = float(started)
    for idx in chosen_sorted:
        visited += 1
        if max_seconds > 0 and (time.time() - started) >= max_seconds:
            print(
                "[distill] stop: "
                f"elapsed={time.time() - started:.1f}s reached max_seconds={max_seconds:.1f}s "
                f"generated={generated} visited={visited}/{len(chosen_sorted)}"
            )
            break
        pair = train_pairs[idx]
        if bool(skip_synthetic_prompts) and _is_synthetic_template_prompt(pair.user):
            continue
        base_score, _base_alignment = _paired_response_score(pair.user, pair.assistant)
        base_rank, _base_rank_score, _base_rank_density, _base_rank_metrics = _distillation_candidate_rank(
            user_text=pair.user,
            candidate_text=pair.assistant,
            reference_text=pair.assistant,
            density_bias=float(density_bias),
            gain_bias=float(gain_bias),
            compactness_bias=float(compactness_bias),
            source=str(pair.source or "dataset"),
        )
        required_score = max(float(min_quality_score), base_score + max(0.0, float(min_quality_gain)))
        best_response = ""
        best_score = -1e9
        best_rank = -1e9
        best_density = 0.0
        for teacher_resp in teacher.generate_candidates(pair.user, temperatures=candidate_temperatures):
            teacher_resp = _clean_training_text(teacher_resp, is_user=False)
            if not teacher_resp:
                continue
            if _looks_like_placeholder_assistant(teacher_resp):
                continue
            if not _is_quality_pair(pair.user, teacher_resp, min_chars=4):
                continue
            teacher_rank, teacher_score, teacher_density, teacher_metrics = _distillation_candidate_rank(
                user_text=pair.user,
                candidate_text=teacher_resp,
                reference_text=pair.assistant,
                density_bias=float(density_bias),
                gain_bias=float(gain_bias),
                compactness_bias=float(compactness_bias),
                source=str(pair.source or "dataset"),
            )
            if teacher_score < required_score:
                continue
            if teacher_rank < (base_rank + max(0.0, float(rank_margin))):
                continue
            key = (pair.user, teacher_resp)
            if key in seen_keys:
                continue
            if teacher_rank > best_rank or (
                abs(teacher_rank - best_rank) <= 1e-9
                and (
                    teacher_metrics.get("quality_gain", 0.0) > 0.0
                    or teacher_density > best_density
                )
            ):
                best_rank = teacher_rank
                best_score = teacher_score
                best_density = teacher_density
                best_response = teacher_resp
        if best_response:
            seen_keys.add((pair.user, best_response))
            mixed.append(ChatPair(user=pair.user, assistant=best_response, source="supermix_teacher"))
            generated += 1

        now = time.time()
        if _should_log_progress_heartbeat(
            visited=visited,
            log_every=log_every,
            now=now,
            last_log_time=last_progress_log_time,
            heartbeat_seconds=progress_heartbeat_seconds,
        ):
            elapsed = max(1e-6, now - started)
            print(
                "[distill] progress: "
                f"visited={visited}/{len(chosen_sorted)} generated={generated} "
                f"rate={visited / elapsed:.2f}/s"
            )
            last_progress_log_time = float(now)
    rng.shuffle(mixed)
    elapsed = max(1e-6, time.time() - started)
    print(
        "[distill] complete: "
        f"generated={generated} visited={visited}/{len(chosen_sorted)} elapsed={elapsed:.1f}s"
    )
    return mixed, generated


def _chat_text_pair(tokenizer, user_text: str, assistant_text: str) -> Tuple[str, str]:
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        prompt = f"User: {user_text}\nAssistant:"
        full = f"{prompt} {assistant_text}"
    return prompt, full


def _chat_prompt_only(tokenizer, user_text: str) -> str:
    has_template = getattr(tokenizer, "chat_template", None)
    if has_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"User: {user_text}\nAssistant:"


def _encode_user_assistant(
    tokenizer,
    user_text: str,
    assistant_text: str,
    max_length: int,
) -> Optional[Dict[str, List[int]]]:
    prompt, full = _chat_text_pair(tokenizer, user_text, assistant_text)
    full_enc = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
    prompt_enc = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)
    input_ids = list(full_enc["input_ids"])
    if not input_ids:
        return None
    labels = list(input_ids)
    prompt_len = min(len(prompt_enc["input_ids"]), max(0, len(labels) - 1))
    for i in range(prompt_len):
        labels[i] = -100
    if all(v == -100 for v in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": list(full_enc["attention_mask"]),
        "labels": labels,
    }


def encode_for_causal_lm(tokenizer, pair: ChatPair, max_length: int) -> Optional[Dict[str, List[int]]]:
    return _encode_user_assistant(
        tokenizer=tokenizer,
        user_text=pair.user,
        assistant_text=pair.assistant,
        max_length=max_length,
    )


class PackedChatDataset(Dataset):
    def __init__(self, rows: List[Dict[str, List[int]]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.rows[idx]


class LengthBucketBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_window_multiplier: int = 16,
        seed: int = 0,
    ) -> None:
        self.lengths = [max(1, int(x)) for x in lengths]
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.bucket_window_multiplier = max(2, int(bucket_window_multiplier))
        self.seed = int(seed)
        self._epoch = 0

    def __len__(self) -> int:
        if not self.lengths:
            return 0
        return int(math.ceil(float(len(self.lengths)) / float(self.batch_size)))

    def __iter__(self):
        if not self.lengths:
            return iter([])

        if not self.shuffle:
            ordered = sorted(range(len(self.lengths)), key=lambda idx: self.lengths[idx], reverse=True)
            batches = [
                ordered[start : start + self.batch_size]
                for start in range(0, len(ordered), self.batch_size)
            ]
            return iter(batches)

        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)
        window_size = max(self.batch_size, self.batch_size * self.bucket_window_multiplier)
        batches: List[List[int]] = []
        for start in range(0, len(indices), window_size):
            chunk = indices[start : start + window_size]
            chunk.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for offset in range(0, len(chunk), self.batch_size):
                batch = chunk[offset : offset + self.batch_size]
                if batch:
                    batches.append(batch)
        rng.shuffle(batches)
        return iter(batches)


def _build_bucketed_dataloader(
    dataset: Dataset,
    lengths: Sequence[int],
    batch_size: int,
    collate_fn,
    shuffle: bool,
    enabled: bool,
    bucket_window_multiplier: int,
    seed: int,
    label: str,
) -> DataLoader:
    if bool(enabled) and len(lengths) > max(1, int(batch_size)):
        sampler = LengthBucketBatchSampler(
            lengths=lengths,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            bucket_window_multiplier=int(bucket_window_multiplier),
            seed=int(seed),
        )
        sorted_lengths = sorted(int(x) for x in lengths)
        median_len = sorted_lengths[len(sorted_lengths) // 2]
        print(
            f"[{label}] length-bucketed batches: "
            f"rows={len(lengths)} batch_size={max(1, int(batch_size))} "
            f"window_mult={max(2, int(bucket_window_multiplier))} "
            f"median_tokens={median_len}"
        )
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        collate_fn=collate_fn,
    )


def build_rows(
    tokenizer,
    pairs: Sequence[ChatPair],
    max_length: int,
    row_weight_fn=None,
    followup_paraphrase_aug: int = 0,
    followup_paraphrase_weight: float = 0.72,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    augmented_rows = 0
    for pair in pairs:
        base_weight = 1.0
        if row_weight_fn is not None:
            try:
                base_weight = float(row_weight_fn(pair))
            except Exception:
                base_weight = 1.0
        row = encode_for_causal_lm(tokenizer, pair, max_length=max_length)
        if row is not None:
            if row_weight_fn is not None:
                row["sample_weight"] = float(max(0.05, base_weight))
            row["sequence_tokens"] = int(len(row["input_ids"]))
            row["supervised_tokens"] = int(sum(1 for token in row["labels"] if int(token) != -100))
            row["source"] = str(pair.source or "dataset")
            rows.append(row)
        if int(followup_paraphrase_aug) > 0:
            for variant_user in _followup_paraphrase_variants(pair.user, max_variants=int(followup_paraphrase_aug)):
                aug_pair = ChatPair(user=variant_user, assistant=pair.assistant, source=pair.source)
                aug_row = encode_for_causal_lm(tokenizer, aug_pair, max_length=max_length)
                if aug_row is None:
                    continue
                aug_weight = float(max(0.05, base_weight * float(followup_paraphrase_weight)))
                aug_row["sample_weight"] = aug_weight
                aug_row["sequence_tokens"] = int(len(aug_row["input_ids"]))
                aug_row["supervised_tokens"] = int(sum(1 for token in aug_row["labels"] if int(token) != -100))
                aug_row["source"] = str(pair.source or "dataset")
                rows.append(aug_row)
                augmented_rows += 1
    if not rows:
        raise ValueError("No valid rows produced after tokenization.")
    if augmented_rows > 0:
        print(f"[sft] added {augmented_rows} follow-up paraphrase augmentation rows")
    return rows


def _row_supervised_token_count(row: Dict[str, object]) -> int:
    value = row.get("supervised_tokens", 0)
    try:
        count = int(value)
    except Exception:
        count = 0
    if count > 0:
        return count
    labels = row.get("labels", [])
    if isinstance(labels, (list, tuple)):
        return int(sum(1 for token in labels if int(token) != -100))
    return 0


def _pack_sft_rows(
    rows: Sequence[Dict[str, object]],
    max_length: int,
    separator_token_id: int,
    max_samples_per_row: int = 0,
) -> List[Dict[str, object]]:
    if not rows:
        return []
    max_row_length = max(1, int(max_length))
    sample_cap = max(0, int(max_samples_per_row))
    ordered_rows = sorted(
        list(rows),
        key=lambda row: int(row.get("sequence_tokens", len(row.get("input_ids", [])))),
        reverse=True,
    )
    bins: List[Dict[str, object]] = []
    for row in ordered_rows:
        row_len = int(row.get("sequence_tokens", len(row.get("input_ids", []))))
        if row_len <= 0:
            continue
        best_idx = -1
        best_remaining = max_row_length + 1
        for idx, packed_bin in enumerate(bins):
            existing_rows = packed_bin["rows"]
            if sample_cap > 0 and len(existing_rows) >= sample_cap:
                continue
            separator_cost = 1 if existing_rows else 0
            used = int(packed_bin["used_tokens"])
            if used + separator_cost + row_len > max_row_length:
                continue
            remaining = max_row_length - (used + separator_cost + row_len)
            if remaining < best_remaining:
                best_idx = idx
                best_remaining = remaining
        if best_idx < 0:
            bins.append({"rows": [row], "used_tokens": row_len})
            continue
        bins[best_idx]["rows"].append(row)
        bins[best_idx]["used_tokens"] = int(bins[best_idx]["used_tokens"]) + 1 + row_len

    packed_rows: List[Dict[str, object]] = []
    for packed_bin in bins:
        segment_rows = list(packed_bin["rows"])
        packed_input_ids: List[int] = []
        packed_attention_mask: List[int] = []
        packed_labels: List[int] = []
        segment_lengths: List[int] = []
        source_values: List[str] = []
        total_supervised = 0
        weight_numer = 0.0
        weight_denom = 0
        has_sample_weight = False
        for idx, row in enumerate(segment_rows):
            row_input = list(row["input_ids"])
            row_attention = list(row["attention_mask"])
            row_labels = list(row["labels"])
            row_supervised = max(0, _row_supervised_token_count(row))
            row_weight = float(row.get("sample_weight", 1.0))
            if idx > 0:
                packed_input_ids.append(int(separator_token_id))
                packed_attention_mask.append(1)
                packed_labels.append(-100)
            packed_input_ids.extend(int(token) for token in row_input)
            packed_attention_mask.extend(int(token) for token in row_attention)
            packed_labels.extend(int(token) for token in row_labels)
            segment_lengths.append(len(row_input))
            total_supervised += row_supervised
            if "sample_weight" in row:
                has_sample_weight = True
                weight_numer += float(max(1, row_supervised)) * row_weight
                weight_denom += max(1, row_supervised)
            source_values.append(str(row.get("source", "dataset") or "dataset"))
        packed_row: Dict[str, object] = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "labels": packed_labels,
            "sequence_tokens": int(len(packed_input_ids)),
            "supervised_tokens": int(total_supervised),
            "packed_sample_count": int(len(segment_rows)),
            "segment_lengths": segment_lengths,
        }
        unique_sources = sorted(set(source_values))
        if len(unique_sources) == 1:
            packed_row["source"] = unique_sources[0]
        elif unique_sources:
            packed_row["source"] = "packed_mix"
        if has_sample_weight:
            packed_row["sample_weight"] = float(weight_numer / max(1, weight_denom))
        packed_rows.append(packed_row)
    return packed_rows


def _median_numeric(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return 0.5 * float(ordered[mid - 1] + ordered[mid])


def collate_rows(rows: Sequence[Dict[str, object]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(r["input_ids"]) for r in rows)
    batch_input_ids = []
    batch_attention = []
    batch_labels = []
    for r in rows:
        n = len(r["input_ids"])
        pad = max_len - n
        batch_input_ids.append(r["input_ids"] + [pad_token_id] * pad)
        batch_attention.append(r["attention_mask"] + [0] * pad)
        batch_labels.append(r["labels"] + [-100] * pad)
    out = {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }
    if any("sample_weight" in r for r in rows):
        out["weights"] = torch.tensor(
            [float(r.get("sample_weight", 1.0)) for r in rows],
            dtype=torch.float32,
        )
    if any("segment_lengths" in r for r in rows):
        segment_counts = [max(1, len(r.get("segment_lengths", []))) for r in rows]
        max_segments = max(segment_counts)
        batch_segment_lengths = []
        batch_segment_weights = []
        for r in rows:
            lengths = [int(value) for value in r.get("segment_lengths", [len(r["input_ids"])])]
            weights = [float(value) for value in r.get("segment_weights", [float(r.get("sample_weight", 1.0))])]
            if len(weights) < len(lengths):
                weights.extend([1.0] * (len(lengths) - len(weights)))
            elif len(weights) > len(lengths):
                weights = weights[: len(lengths)]
            pad_segments = max_segments - len(lengths)
            batch_segment_lengths.append(lengths + [0] * pad_segments)
            batch_segment_weights.append(weights + [0.0] * pad_segments)
        out["segment_count"] = torch.tensor(segment_counts, dtype=torch.long)
        out["segment_lengths"] = torch.tensor(batch_segment_lengths, dtype=torch.long)
        out["segment_weights"] = torch.tensor(batch_segment_weights, dtype=torch.float32)
    return out


def _set_model_use_cache(model, enabled: bool) -> None:
    use_cache = bool(enabled)
    cfg = getattr(model, "config", None)
    if cfg is not None:
        cfg.use_cache = use_cache
    base_model = getattr(model, "base_model", None)
    base_cfg = getattr(base_model, "config", None)
    if base_cfg is not None:
        base_cfg.use_cache = use_cache


def _device_backend_name(device: Any, resolved_backend: str = "") -> str:
    backend = str(resolved_backend or "").strip().lower()
    if backend:
        return backend
    device_type = str(getattr(device, "type", "") or "").strip().lower()
    if device_type == "privateuseone":
        return "dml"
    if device_type:
        return device_type
    return str(device).strip().lower() or "cpu"


def _resolve_device(device_spec: str, device_preference: str = "cuda,npu,xpu,dml,mps,cpu") -> Tuple[Any, Dict[str, str]]:
    device, info = resolve_runtime_device(
        device_spec=str(device_spec),
        preference=str(device_preference),
    )
    resolved_backend = _device_backend_name(device=device, resolved_backend=str(info.get("resolved", "")))
    requested = str(info.get("requested", device_spec or "auto")).strip().lower() or "auto"
    if requested != resolved_backend:
        print(
            "[runtime] device resolve: "
            f"requested={requested} preference={str(device_preference).strip()} "
            f"resolved={resolved_backend} device={device}"
        )
    return device, {
        "requested": requested,
        "resolved": resolved_backend,
        "device_repr": str(device),
        "preference": str(device_preference).strip(),
    }


def _resolve_torch_dtype(dtype_spec: str, device: Any, resolved_backend: str = "") -> torch.dtype:
    mode = str(dtype_spec).strip().lower()
    backend = _device_backend_name(device=device, resolved_backend=resolved_backend)
    explicit = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if mode in explicit:
        resolved = explicit[mode]
    else:
        if backend == "cuda":
            supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            resolved = torch.bfloat16 if supports_bf16 else torch.float16
        elif backend in {"mps", "dml", "npu", "xpu"}:
            resolved = torch.float16
        else:
            resolved = torch.float32
    if backend == "cpu" and resolved != torch.float32:
        print("[runtime] cpu training requires float32 weights; overriding requested dtype.")
        return torch.float32
    return resolved


def _resolve_peft_bootstrap(
    runtime_device: Any,
    resolved_backend: str,
    runtime_dtype: torch.dtype,
) -> Tuple[Any, torch.dtype, str]:
    backend = _device_backend_name(device=runtime_device, resolved_backend=resolved_backend)
    if backend == "dml":
        return (
            torch.device("cpu"),
            torch.float32,
            "DirectML PEFT init uses a CPU/float32 bootstrap to avoid PiSSA/DoRA init incompatibilities.",
        )
    return runtime_device, runtime_dtype, ""


def _patch_dora_linear_forward_for_dml() -> bool:
    if DoraLinearLayer is None or dequantize_module_weight is None or transpose is None:
        return False
    if getattr(DoraLinearLayer.forward, "_supermix_dml_safe", False):
        return False

    original_forward = DoraLinearLayer.forward

    def _dml_safe_forward(self, x, *, lora_A, lora_B, scaling, base_layer, base_result=None):
        if _device_backend_name(device=x.device) != "dml":
            return original_forward(
                self,
                x,
                lora_A=lora_A,
                lora_B=lora_B,
                scaling=scaling,
                base_layer=base_layer,
                base_result=base_result,
            )

        lora_weight = (lora_B.weight @ lora_A.weight).to(dtype=x.dtype)
        magnitude = self.weight.to(dtype=x.dtype)
        weight = dequantize_module_weight(base_layer).to(x.dtype)
        weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scaling).detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)

        lora_result = lora_B(lora_A(x))

        if base_result is not None:
            bias = base_layer.bias
            if bias is not None:
                base_result = base_result - bias
        else:
            base_result = torch.nn.functional.linear(x, transpose(weight, self.fan_in_fan_out))

        return (mag_norm_scale - 1) * base_result + mag_norm_scale * lora_result * scaling

    _dml_safe_forward._supermix_dml_safe = True
    DoraLinearLayer.forward = _dml_safe_forward
    return True


def _disable_peft_init_for_weight_load(peft_cfg: Any) -> str:
    init_mode = getattr(peft_cfg, "init_lora_weights", None)
    if init_mode in (None, False):
        return ""
    try:
        peft_cfg.init_lora_weights = False
        return str(init_mode)
    except Exception:
        return ""


def _load_base_model_and_tokenizer(
    base_model: str,
    device: Any,
    for_training: bool = True,
    model_dtype: torch.dtype = torch.float32,
    gradient_checkpointing: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if bool(for_training) and bool(gradient_checkpointing):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        else:
            print("[train] gradient checkpointing requested but unsupported by this model class.")
    _set_model_use_cache(model, enabled=not bool(for_training))
    return model, tokenizer


def _target_modules_from_arg(spec: str) -> List[str]:
    raw = [x.strip() for x in spec.split(",")]
    return [x for x in raw if x]


def _parse_lora_init_mode(value: str):
    v = str(value).strip().lower()
    if v == "true":
        return True
    if v == "false":
        return False
    return value


def _is_lora_b_parameter(name: str) -> bool:
    low = str(name).strip().lower()
    return any(token in low for token in ("lora_b", "lora_embedding_b"))


def _build_optimizer_param_groups(
    model,
    base_lr: float,
    weight_decay: float,
    lora_plus_ratio: float = 0.0,
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    named_trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    base_lr = float(base_lr)
    ratio = float(lora_plus_ratio)
    weight_decay = float(weight_decay)
    if ratio <= 1.0:
        return (
            [{"params": [param for _name, param in named_trainable], "lr": base_lr, "weight_decay": weight_decay}],
            {
                "trainable_params": float(len(named_trainable)),
                "lora_plus_ratio": 0.0,
                "lora_plus_base_group_params": float(len(named_trainable)),
                "lora_plus_fast_group_params": 0.0,
            },
        )

    base_group: List[torch.nn.Parameter] = []
    fast_group: List[torch.nn.Parameter] = []
    for name, param in named_trainable:
        if _is_lora_b_parameter(name):
            fast_group.append(param)
        else:
            base_group.append(param)

    if not fast_group:
        return (
            [{"params": [param for _name, param in named_trainable], "lr": base_lr, "weight_decay": weight_decay}],
            {
                "trainable_params": float(len(named_trainable)),
                "lora_plus_ratio": 0.0,
                "lora_plus_base_group_params": float(len(named_trainable)),
                "lora_plus_fast_group_params": 0.0,
            },
        )

    groups: List[Dict[str, object]] = []
    if base_group:
        groups.append({"params": base_group, "lr": base_lr, "weight_decay": weight_decay})
    groups.append({"params": fast_group, "lr": base_lr * ratio, "weight_decay": weight_decay})
    return (
        groups,
        {
            "trainable_params": float(len(named_trainable)),
            "lora_plus_ratio": float(ratio),
            "lora_plus_base_group_params": float(len(base_group)),
            "lora_plus_fast_group_params": float(len(fast_group)),
        },
    )


def _load_init_adapter_config(adapter_dir: Path) -> Dict[str, object]:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        return {}
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception as e:
        print(f"[adapter] failed to read adapter_config.json from {adapter_dir}: {e}")
    return {}


def _build_lr_lambda(
    schedule: str,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float,
    restart_period: int = 0,
):
    schedule_mode = str(schedule).strip().lower()
    if schedule_mode not in {"constant", "cosine", "cosine_restarts"}:
        schedule_mode = "constant"
    warmup = max(0, int(warmup_steps))
    total = max(1, int(total_steps))
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))
    period = max(0, int(restart_period))

    def _lambda(step_idx: int) -> float:
        step = int(step_idx) + 1
        if warmup > 0 and step <= warmup:
            return max(1e-6, float(step) / float(max(1, warmup)))
        if schedule_mode == "constant" or total <= warmup:
            return 1.0
        post_warmup = step - warmup
        if schedule_mode == "cosine_restarts" and period > 0:
            # Cosine annealing with periodic warm restarts.
            cycle_pos = post_warmup % period
            progress = float(cycle_pos) / float(max(1, period))
        else:
            progress = float(post_warmup) / float(max(1, total - warmup))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_ratio + (1.0 - min_ratio) * cosine)

    return _lambda


def _register_neftune_hook(model, noise_alpha: float):
    alpha = float(noise_alpha)
    if alpha <= 0:
        return None

    emb = model.get_input_embeddings()
    if emb is None:
        return None

    def _hook(_module, _inputs, output):
        if not _module.training or not isinstance(output, torch.Tensor):
            return output
        hidden = max(1, int(output.shape[-1]))
        mag = alpha / math.sqrt(hidden)
        noise = torch.empty_like(output).uniform_(-mag, mag)
        return output + noise

    return emb.register_forward_hook(_hook)


def _replace_neftune_hook(model, hook, noise_alpha: float):
    if hook is not None:
        hook.remove()
    return _register_neftune_hook(model, noise_alpha=float(noise_alpha))


def _load_adapter_state_dict(adapter_dir: Path) -> Dict[str, torch.Tensor]:
    safetensors_path = adapter_dir / "adapter_model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file  # local import to keep startup light

        return load_file(str(safetensors_path))
    bin_path = adapter_dir / "adapter_model.bin"
    if bin_path.exists():
        state = torch.load(bin_path, map_location="cpu")
        if isinstance(state, dict):
            return state
        raise TypeError(f"Unexpected adapter_model.bin payload type: {type(state)}")
    raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")


def _is_legacy_nested_adapter_state(state_dict: Dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    first_key = next(iter(state_dict.keys()))
    return first_key.startswith(LEGACY_NESTED_ADAPTER_PREFIX)


def _canonicalize_adapter_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    remapped: Dict[str, torch.Tensor] = {}
    remapped_count = 0
    for key, value in state_dict.items():
        new_key = key
        while NESTED_PREFIX_FRAGMENT in new_key:
            new_key = new_key.replace(NESTED_PREFIX_FRAGMENT, "base_model.model.")
        if new_key != key:
            remapped_count += 1
        remapped[new_key] = value
    return remapped, remapped_count


def _maybe_merge_adapter_for_inference(model, device: torch.device):
    if hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
        _set_model_use_cache(merged, enabled=True)
        return merged.to(device)
    _set_model_use_cache(model, enabled=True)
    return model.to(device)


def _load_adapter_with_compat(
    model,
    adapter_dir: Path,
    device: Any,
    is_trainable: bool,
    merge_for_inference: bool = False,
):
    state_dict = _load_adapter_state_dict(adapter_dir)
    if _is_legacy_nested_adapter_state(state_dict):
        print(f"[adapter] legacy nested format detected: {adapter_dir}")
    canonical_state, remapped_count = _canonicalize_adapter_state_dict(state_dict)

    peft_model = model
    if not isinstance(peft_model, PeftModel):
        peft_cfg = PeftConfig.from_pretrained(str(adapter_dir))
        peft_cfg.inference_mode = not bool(is_trainable)
        if _device_backend_name(device=device) == "dml" and bool(getattr(peft_cfg, "use_dora", False)):
            if _patch_dora_linear_forward_for_dml():
                print("[adapter] applied DirectML DoRA forward compatibility patch.")
        disabled_init_mode = _disable_peft_init_for_weight_load(peft_cfg)
        if disabled_init_mode:
            print(
                "[adapter] disabled init_lora_weights while loading saved adapter: "
                f"{disabled_init_mode}"
            )
        peft_model = get_peft_model(peft_model, peft_cfg)

    incompat = set_peft_model_state_dict(peft_model, canonical_state, adapter_name="default")
    missing_keys = list(getattr(incompat, "missing_keys", []) or [])
    unexpected_keys = list(getattr(incompat, "unexpected_keys", []) or [])

    def _is_adapter_key(key: str) -> bool:
        return any(
            marker in str(key)
            for marker in (
                ".lora_",
                "lora_magnitude_vector",
                "modules_to_save",
                "prompt_embeddings",
                "trainable_tokens_",
            )
        )

    adapter_missing_count = sum(1 for key in missing_keys if _is_adapter_key(key))
    adapter_unexpected_count = sum(1 for key in unexpected_keys if _is_adapter_key(key))
    base_missing_count = max(0, len(missing_keys) - adapter_missing_count)
    base_unexpected_count = max(0, len(unexpected_keys) - adapter_unexpected_count)
    print(
        "[adapter] loaded "
        f"remapped={remapped_count} "
        f"adapter_missing={adapter_missing_count} "
        f"adapter_unexpected={adapter_unexpected_count} "
        f"base_missing={base_missing_count} "
        f"base_unexpected={base_unexpected_count}"
    )
    peft_model = peft_model.to(device)
    if merge_for_inference and not bool(is_trainable):
        return _maybe_merge_adapter_for_inference(peft_model, device=device)
    return peft_model


class PackedPreferenceDataset(Dataset):
    def __init__(self, rows: List[Dict[str, object]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.rows[idx]


def build_preference_rows(
    tokenizer,
    pref_pairs: Sequence[PreferencePair],
    max_length: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pair in pref_pairs:
        chosen = _encode_user_assistant(tokenizer, pair.user, pair.chosen, max_length=max_length)
        rejected = _encode_user_assistant(tokenizer, pair.user, pair.rejected, max_length=max_length)
        if chosen is None or rejected is None:
            continue
        rows.append(
            {
                "chosen": chosen,
                "rejected": rejected,
                "weight": float(max(0.1, pair.weight)),
                "base_weight": float(max(0.1, pair.weight)),
                "quality_gap": float(max(0.0, pair.quality_gap)),
                "rejected_similarity": float(max(0.0, min(1.0, pair.rejected_similarity))),
                "prompt_complexity": float(max(0.0, pair.prompt_complexity)),
                "selection_score": float(max(0.0, pair.selection_score)),
                "conversation_score": float(max(0.0, pair.conversation_score)),
                "reasoning_score": float(max(0.0, pair.reasoning_score)),
                "creativity_score": float(max(0.0, pair.creativity_score)),
                "knowledge_density_score": float(max(0.0, pair.knowledge_density_score)),
                "is_followup": bool(pair.is_followup),
                "pair_tokens": int(len(chosen["input_ids"]) + len(rejected["input_ids"])),
            }
        )
    return rows


def _preference_row_rescore_weight(
    row: Dict[str, object],
    progress: float,
    target_hardness: float,
    hardness_bandwidth: float,
) -> float:
    base_weight = max(0.1, float(row.get("base_weight", row.get("weight", 1.0))))
    quality_gap = max(0.0, float(row.get("quality_gap", 0.0)))
    rejected_similarity = max(0.0, min(1.0, float(row.get("rejected_similarity", 0.0))))
    prompt_complexity = max(0.0, float(row.get("prompt_complexity", 0.0)))
    selection_score = max(0.0, float(row.get("selection_score", 0.0)))
    conversation_score = max(0.0, float(row.get("conversation_score", 0.0)))
    reasoning_score = max(0.0, float(row.get("reasoning_score", 0.0)))
    creativity_score = max(0.0, float(row.get("creativity_score", 0.0)))
    knowledge_density_score = max(0.0, float(row.get("knowledge_density_score", 0.0)))
    followup_bonus = 0.04 if bool(row.get("is_followup", False)) else 0.0

    easy_focus = 0.84 + 0.46 * min(1.0, quality_gap / 0.24)
    z = (rejected_similarity - target_hardness) / max(0.05, hardness_bandwidth)
    hardness_window = math.exp(-0.5 * z * z)
    hard_focus = 0.78 + 0.58 * hardness_window
    difficulty_focus = ((1.0 - progress) * easy_focus) + (progress * hard_focus)

    novelty_focus = (
        1.0
        + 0.05 * min(2.0, conversation_score)
        + 0.07 * min(2.0, reasoning_score)
        + 0.04 * min(2.0, creativity_score)
        + 0.08 * min(2.0, knowledge_density_score)
        + followup_bonus
    )
    prompt_focus = 0.96 + 0.04 * min(2.5, prompt_complexity)
    selection_focus = 1.0 + 0.03 * min(4.0, selection_score)
    raw_weight = base_weight * difficulty_focus * novelty_focus * prompt_focus * selection_focus
    return float(max(0.1, raw_weight))


def _rescore_preference_rows(
    pref_rows: List[Dict[str, object]],
    step: int,
    total_steps: int,
    round_index: int,
) -> Dict[str, float]:
    if not pref_rows:
        return {
            "weight_mean": 0.0,
            "weight_min": 0.0,
            "weight_max": 0.0,
            "progress": 0.0,
            "target_hardness": 0.0,
            "hardness_bandwidth": 0.0,
        }

    progress = max(0.0, min(1.0, float(step) / float(max(1, total_steps))))
    target_hardness = 0.18 + 0.50 * progress
    hardness_bandwidth = 0.28 - 0.10 * progress

    base_weights = [max(0.1, float(row.get("base_weight", row.get("weight", 1.0)))) for row in pref_rows]
    raw_weights = [
        _preference_row_rescore_weight(
            row=row,
            progress=progress,
            target_hardness=target_hardness,
            hardness_bandwidth=hardness_bandwidth,
        )
        for row in pref_rows
    ]

    target_mean = float(sum(base_weights) / max(1, len(base_weights)))
    raw_mean = float(sum(raw_weights) / max(1, len(raw_weights)))
    scale = target_mean / max(1e-6, raw_mean)

    final_weights: List[float] = []
    for row, raw_weight in zip(pref_rows, raw_weights):
        updated_weight = float(max(0.1, min(4.5, raw_weight * scale)))
        row["weight"] = updated_weight
        final_weights.append(updated_weight)

    summary = {
        "weight_mean": float(sum(final_weights) / max(1, len(final_weights))),
        "weight_min": float(min(final_weights)),
        "weight_max": float(max(final_weights)),
        "progress": float(progress),
        "target_hardness": float(target_hardness),
        "hardness_bandwidth": float(hardness_bandwidth),
    }
    print(
        "[pref] rescored rows: "
        f"round={int(round_index)} step={int(step)}/{int(max(1, total_steps))} "
        f"progress={summary['progress']:.3f} "
        f"target_hardness={summary['target_hardness']:.3f} "
        f"bandwidth={summary['hardness_bandwidth']:.3f} "
        f"weight_mean={summary['weight_mean']:.3f} "
        f"range=[{summary['weight_min']:.3f}, {summary['weight_max']:.3f}]"
    )
    return summary


def collate_preference_rows(
    rows: Sequence[Dict[str, object]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    chosen_rows = [r["chosen"] for r in rows]
    rejected_rows = [r["rejected"] for r in rows]
    weights = torch.tensor([float(r.get("weight", 1.0)) for r in rows], dtype=torch.float32)
    chosen = collate_rows(chosen_rows, pad_token_id=pad_token_id)
    rejected = collate_rows(rejected_rows, pad_token_id=pad_token_id)
    out = {
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "chosen_labels": chosen["labels"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
        "rejected_labels": rejected["labels"],
        "weights": weights,
    }
    if rows and all(("ref_chosen_logp" in r and "ref_rejected_logp" in r) for r in rows):
        out["ref_chosen_logp"] = torch.tensor(
            [float(r.get("ref_chosen_logp", 0.0)) for r in rows],
            dtype=torch.float32,
        )
        out["ref_rejected_logp"] = torch.tensor(
            [float(r.get("ref_rejected_logp", 0.0)) for r in rows],
            dtype=torch.float32,
        )
    return out


def _sequence_average_log_prob(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_f = valid.to(token_log_probs.dtype)
    token_log_probs = token_log_probs * valid_f
    lengths = valid_f.sum(dim=1).clamp_min(1.0)
    avg_log_prob = token_log_probs.sum(dim=1) / lengths
    return avg_log_prob, lengths


def _packed_segment_average_values(
    token_values: torch.Tensor,
    valid_mask: torch.Tensor,
    segment_lengths: torch.Tensor,
    segment_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_f = valid_mask.to(token_values.dtype)
    masked_values = token_values * valid_f
    flat_values: List[torch.Tensor] = []
    flat_lengths: List[torch.Tensor] = []
    batch_size = int(token_values.shape[0])
    for batch_idx in range(batch_size):
        count = int(segment_count[batch_idx].item())
        cursor = 0
        for seg_idx in range(max(0, count)):
            seg_len = int(segment_lengths[batch_idx, seg_idx].item())
            if seg_len <= 0:
                continue
            shift_start = 0 if cursor <= 0 else cursor - 1
            shift_end = max(shift_start, cursor + seg_len - 1)
            seg_values = masked_values[batch_idx, shift_start:shift_end]
            seg_valid = valid_f[batch_idx, shift_start:shift_end]
            seg_token_count = seg_valid.sum().clamp_min(1.0)
            flat_values.append(seg_values.sum() / seg_token_count)
            flat_lengths.append(seg_token_count)
            cursor += seg_len + 1
    if not flat_values:
        zero = token_values.new_zeros((1,))
        one = token_values.new_ones((1,))
        return zero, one
    return torch.stack(flat_values), torch.stack(flat_lengths)


def _packed_segment_average_log_prob(
    logits: torch.Tensor,
    labels: torch.Tensor,
    segment_lengths: torch.Tensor,
    segment_count: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return _packed_segment_average_values(
        token_values=token_log_probs,
        valid_mask=valid,
        segment_lengths=segment_lengths,
        segment_count=segment_count,
    )


def _flatten_packed_segment_values(values: torch.Tensor, segment_count: torch.Tensor) -> torch.Tensor:
    flat_values: List[torch.Tensor] = []
    for batch_idx in range(int(values.shape[0])):
        count = int(segment_count[batch_idx].item())
        if count <= 0:
            continue
        flat_values.append(values[batch_idx, :count])
    if not flat_values:
        return values.new_zeros((0,))
    return torch.cat(flat_values, dim=0)


def _packed_segment_symmetric_token_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    labels: torch.Tensor,
    segment_lengths: torch.Tensor,
    segment_count: torch.Tensor,
) -> torch.Tensor:
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        return logits_a.new_zeros((1,))

    shift_logits_a = logits_a[:, :-1, :]
    shift_logits_b = logits_b[:, :-1, :]
    logp_a = torch.log_softmax(shift_logits_a, dim=-1)
    logp_b = torch.log_softmax(shift_logits_b, dim=-1)
    kl_ab = torch.nn.functional.kl_div(logp_b, logp_a, reduction="none", log_target=True).sum(dim=-1)
    kl_ba = torch.nn.functional.kl_div(logp_a, logp_b, reduction="none", log_target=True).sum(dim=-1)
    sym = 0.5 * (kl_ab + kl_ba)
    flat_sym, _ = _packed_segment_average_values(
        token_values=sym,
        valid_mask=valid,
        segment_lengths=segment_lengths,
        segment_count=segment_count,
    )
    return flat_sym


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    norm_w = weights.float() / weights.float().mean().clamp_min(1e-6)
    return (values * norm_w).sum() / norm_w.sum().clamp_min(1e-6)


def _symmetric_token_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        return logits_a.new_zeros((labels.shape[0],))

    shift_logits_a = logits_a[:, :-1, :]
    shift_logits_b = logits_b[:, :-1, :]
    logp_a = torch.log_softmax(shift_logits_a, dim=-1)
    logp_b = torch.log_softmax(shift_logits_b, dim=-1)
    kl_ab = torch.nn.functional.kl_div(logp_b, logp_a, reduction="none", log_target=True).sum(dim=-1)
    kl_ba = torch.nn.functional.kl_div(logp_a, logp_b, reduction="none", log_target=True).sum(dim=-1)
    sym = 0.5 * (kl_ab + kl_ba)
    sym = sym * valid.to(sym.dtype)
    lengths = valid.sum(dim=1).clamp_min(1)
    return sym.sum(dim=1) / lengths


def _wpo_pair_weights(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    alpha: float,
    clip: float,
) -> torch.Tensor:
    if alpha <= 0.0:
        return torch.ones_like(chosen_logp)

    pair_log_weight = chosen_logp.detach().float() + rejected_logp.detach().float()
    pair_log_weight = pair_log_weight - pair_log_weight.mean()
    weights = torch.exp(float(alpha) * pair_log_weight)
    max_scale = max(1.0, float(clip))
    weights = weights.clamp(min=1.0 / max_scale, max=max_scale)
    return weights / weights.mean().clamp_min(1e-6)


def _log_odds_from_avg_log_prob(avg_log_prob: torch.Tensor) -> torch.Tensor:
    # ORPO-style odds ratio needs sequence probabilities in (0, 1).
    safe_log_prob = avg_log_prob.clamp(min=math.log(1e-8), max=math.log(1.0 - 1e-6))
    prob = torch.exp(safe_log_prob).clamp(min=1e-8, max=1.0 - 1e-6)
    return safe_log_prob - torch.log1p(-prob)


def _linear_schedule_value(start: float, end: float, step_idx: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(start)
    t = float(max(0, min(step_idx, total_steps - 1))) / float(total_steps - 1)
    return float(start + (end - start) * t)


def _sigmoid_preference_loss(logits_delta: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    eps = max(0.0, min(0.49, float(label_smoothing)))
    pos = torch.nn.functional.logsigmoid(logits_delta)
    if eps <= 0.0:
        return -pos
    neg = torch.nn.functional.logsigmoid(-logits_delta)
    return -((1.0 - eps) * pos + eps * neg)


def _tensor_one_minus(x: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(x) - x


def _broadcast_margin_like(delta: torch.Tensor, margin: Any) -> torch.Tensor:
    if isinstance(margin, torch.Tensor):
        return margin.to(device=delta.device, dtype=delta.dtype)
    return delta.new_full(delta.shape, float(margin))


def _dpo_preference_loss(
    delta: torch.Tensor,
    ref_delta: torch.Tensor,
    beta: float,
    margin: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    margin_t = _broadcast_margin_like(delta, margin)
    logits_delta = float(beta) * ((delta - ref_delta) - margin_t)
    return _sigmoid_preference_loss(logits_delta, label_smoothing=label_smoothing)


def _ipo_target_gap(beta: float, margin: float = 0.0) -> float:
    base_gap = 0.5 / max(1e-6, float(beta))
    if isinstance(margin, torch.Tensor):
        return margin + base_gap
    return float(margin) + base_gap


def _ipo_preference_loss(
    delta: torch.Tensor,
    ref_delta: torch.Tensor,
    beta: float,
    margin: float = 0.0,
) -> torch.Tensor:
    target_gap = _ipo_target_gap(beta=beta, margin=margin)
    target_gap_t = _broadcast_margin_like(delta, target_gap)
    return ((delta - ref_delta) - target_gap_t).pow(2)


def _preference_length_control_margin(
    chosen_len: torch.Tensor,
    rejected_len: torch.Tensor,
    target_ratio: float,
    max_penalty: float,
) -> torch.Tensor:
    chosen = chosen_len.float().clamp_min(1.0)
    rejected = rejected_len.float().clamp_min(1.0)
    ratio = chosen / rejected
    target = max(0.75, float(target_ratio))
    penalty_cap = max(0.0, float(max_penalty))
    excess = torch.relu(ratio - target) / target
    if penalty_cap <= 0.0:
        return torch.zeros_like(excess)
    return excess.clamp(max=penalty_cap)


def _xpo_preference_logits_delta(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
    clip: float = 0.0,
) -> torch.Tensor:
    chosen_log_ratio = chosen_logp - ref_chosen_logp
    rejected_log_ratio = rejected_logp - ref_rejected_logp
    logits_delta = float(beta) * (
        _xpo_log_ratio_transform(chosen_log_ratio) - _xpo_log_ratio_transform(rejected_log_ratio)
    )
    if float(clip) > 0.0:
        logits_delta = logits_delta.clamp(min=-float(clip), max=float(clip))
    return logits_delta


def _xpo_log_ratio_transform(log_ratio: torch.Tensor) -> torch.Tensor:
    return torch.exp(log_ratio) + log_ratio


def _xpo_preference_loss(
    chosen_logp: torch.Tensor,
    rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
    clip: float = 0.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits_delta = _xpo_preference_logits_delta(
        chosen_logp=chosen_logp,
        rejected_logp=rejected_logp,
        ref_chosen_logp=ref_chosen_logp,
        ref_rejected_logp=ref_rejected_logp,
        beta=beta,
        clip=clip,
    )
    return _sigmoid_preference_loss(logits_delta, label_smoothing=label_smoothing)


def _robust_preference_correctness_weights(
    logits_delta: torch.Tensor,
    alpha: float,
    noise_eta: float,
    clip: float,
) -> torch.Tensor:
    robust_alpha = max(0.0, min(1.0, float(alpha)))
    eta = max(0.0, min(0.49, float(noise_eta)))
    if robust_alpha <= 0.0 or eta <= 0.0:
        return torch.ones_like(logits_delta)

    prob_correct = torch.sigmoid(logits_delta.detach().float())
    eta_t = prob_correct.new_tensor(eta)
    robust_alpha_t = prob_correct.new_tensor(robust_alpha)
    numerator = (torch.ones_like(prob_correct) - eta_t) * prob_correct
    denom = numerator + eta_t * _tensor_one_minus(prob_correct)
    posterior = numerator / denom.clamp_min(1e-6)
    weights = posterior / posterior.mean().clamp_min(1e-6)
    max_scale = max(1.0, float(clip))
    weights = weights.clamp(min=1.0 / max_scale, max=max_scale)
    weights = weights / weights.mean().clamp_min(1e-6)
    blended = (torch.ones_like(weights) - robust_alpha_t) + robust_alpha_t * weights
    return blended / blended.mean().clamp_min(1e-6)


def _pair_sft_weight(
    pair: ChatPair,
    mode: str,
    min_weight: float,
    max_weight: float,
    synthetic_prompt_weight: float,
    teacher_source_weight: float,
    quality_anchor_boost: float,
    coding_boost: float,
    events_boost: float,
    reasoning_boost: float,
    prompt_skill_boost: float,
    conversation_boost: float,
    creativity_boost: float,
    knowledge_density_boost: float,
) -> float:
    if str(mode).strip().lower() == "none":
        return 1.0

    source = str(pair.source or "").lower()
    user = _coerce_text(pair.user)
    assistant = _fast_cleanup_response_text(str(pair.assistant or ""))
    paired_score, alignment = _paired_response_score(user, assistant)
    knowledge_density = _pair_knowledge_density_score(user, assistant, source=source)

    # Map quality to a mild multiplier around 1.0.
    quality_factor = 1.0 + 0.22 * max(-1.0, min(1.5, (paired_score - 1.2) / 1.2))
    w = float(quality_factor)

    if _is_synthetic_template_prompt(user):
        w *= float(synthetic_prompt_weight)
    if _is_short_answer_prompt(user):
        w *= 0.95
    if source == "supermix_teacher":
        w *= float(teacher_source_weight)
    if "quality_anchor" in source:
        w *= float(quality_anchor_boost)
    if "coding_knowledge" in source:
        w *= float(coding_boost)
    if "reasoning" in source:
        w *= float(reasoning_boost)
    if "world_events" in source:
        w *= float(events_boost)
    if _looks_like_coding_prompt(user):
        w *= float(max(1.0, prompt_skill_boost))
    if _looks_like_reasoning_prompt(user):
        w *= float(max(1.0, prompt_skill_boost))
    if float(alignment.conversation) > 0.10:
        conv_gain = (max(1.0, float(conversation_boost)) - 1.0) * min(1.0, float(alignment.conversation) / 1.15)
        w *= 1.0 + conv_gain
    if float(alignment.creativity) > 0.05:
        creative_gain = (max(1.0, float(creativity_boost)) - 1.0) * min(1.0, float(alignment.creativity))
        w *= 1.0 + creative_gain
    if bool(alignment.is_followup):
        w *= 1.0 + 0.05 * min(1.0, max(0.0, float(alignment.conversation)))
    if knowledge_density > 0.0:
        density_gain = (max(1.0, float(knowledge_density_boost)) - 1.0) * min(1.0, knowledge_density)
        w *= 1.0 + density_gain

    return float(max(float(min_weight), min(float(max_weight), w)))


def _normalized_quality_signal(score: float) -> float:
    return float(max(0.0, min(1.25, (float(score) + 0.4) / 2.4)))


def _sft_pair_selection_score(
    pair: ChatPair,
    mode: str,
    hardness_target: float,
    hardness_bandwidth: float,
) -> Tuple[float, Dict[str, float]]:
    assistant = _fast_cleanup_response_text(pair.assistant)
    if not assistant:
        return -1e9, {}

    quality_score, alignment = _paired_response_score(pair.user, assistant)
    quality_signal = _normalized_quality_signal(quality_score)
    density = _pair_knowledge_density_score(pair.user, assistant, source=str(pair.source or ""))
    reasoning_signal = max(float(alignment.reasoning), _response_reasoning_signal(assistant))
    conversation_signal = max(0.0, float(alignment.conversation))
    creativity_signal = max(0.0, float(alignment.creativity))
    prompt_complexity = min(1.0, float(_prompt_complexity_score(pair.user)) / 1.35)
    word_count = max(1, _word_token_count(assistant))

    if word_count <= 18:
        compactness = 0.55
    elif word_count <= 96:
        compactness = 1.0
    else:
        compactness = max(0.35, 1.0 - (float(word_count - 96) / 260.0))

    utility = (
        0.40 * quality_signal
        + 0.18 * density
        + 0.14 * reasoning_signal
        + 0.10 * conversation_signal
        + 0.08 * creativity_signal
        + 0.10 * compactness
    )
    if _looks_like_coding_prompt(pair.user):
        utility += 0.04
    if _looks_like_reasoning_prompt(pair.user):
        utility += 0.05
    if "quality_anchor" in str(pair.source or "").lower():
        utility += 0.03

    difficulty = min(
        1.0,
        0.42 * prompt_complexity
        + 0.22 * reasoning_signal
        + 0.18 * min(1.0, float(word_count) / 180.0)
        + 0.18 * quality_signal,
    )

    score = utility
    mode_norm = str(mode).strip().lower()
    if mode_norm in {"capacity_aware", "coverage_topk"}:
        bw = max(0.08, float(hardness_bandwidth))
        z = (difficulty - float(hardness_target)) / bw
        band = math.exp(-0.5 * z * z)
        score = utility * (0.35 + 0.65 * band)

    return float(score), {
        "quality": float(quality_score),
        "quality_signal": float(quality_signal),
        "density": float(density),
        "reasoning": float(reasoning_signal),
        "conversation": float(conversation_signal),
        "creativity": float(creativity_signal),
        "prompt_complexity": float(prompt_complexity),
        "difficulty": float(difficulty),
        "compactness": float(compactness),
        "words": float(word_count),
        "utility": float(utility),
        "score": float(score),
    }


def _estimated_sft_pair_tokens(pair: ChatPair) -> int:
    user_text = _clean_training_text(pair.user, is_user=True)
    assistant_text = _fast_cleanup_response_text(pair.assistant)
    user_tokens = max(1, _word_token_count(user_text))
    assistant_tokens = max(1, _word_token_count(assistant_text))
    return max(8, int(round(user_tokens * 1.10 + assistant_tokens * 1.05 + 6.0)))


def _selection_rarity_bonus(count: int, total: int) -> float:
    count_i = max(1, int(count))
    total_i = max(1, int(total))
    if total_i <= 1:
        return 0.0
    return float(
        max(0.0, min(1.0, math.log1p(float(total_i) / float(count_i)) / math.log1p(float(total_i))))
    )


def _word_count_bucket(word_count: float) -> str:
    words = max(1.0, float(word_count))
    if words <= 24.0:
        return "short"
    if words <= 72.0:
        return "medium"
    if words <= 144.0:
        return "long"
    return "xlong"


def _sft_pair_style_bucket(pair: ChatPair, metrics: Dict[str, float]) -> str:
    if _looks_like_coding_prompt(pair.user):
        return "coding"
    style_scores = {
        "reasoning": float(metrics.get("reasoning", 0.0)),
        "knowledge": float(metrics.get("density", 0.0)),
        "creative": float(metrics.get("creativity", 0.0)),
        "conversation": float(metrics.get("conversation", 0.0)),
    }
    return max(style_scores.items(), key=lambda item: (item[1], item[0]))[0]


def _apply_sft_coverage_topk_scores(
    scored: Sequence[Tuple[float, float, ChatPair, Dict[str, float]]],
    budget_mode: str,
    budget_power: float,
) -> List[Tuple[float, float, ChatPair, Dict[str, float]]]:
    total = len(scored)
    if total <= 1:
        return list(scored)

    source_counts: Dict[str, int] = {}
    group_counts: Dict[str, int] = {}
    style_counts: Dict[str, int] = {}
    length_counts: Dict[str, int] = {}
    keyed_rows: List[Tuple[str, str, str, str]] = []
    for _budget_value, _score, pair, metrics in scored:
        source_key = _split_group_token(Path(str(pair.source or "dataset")).name) or "dataset"
        _group_type, raw_group_key = _chat_pair_split_group_key(pair)
        group_key = raw_group_key or "ungrouped"
        style_key = _sft_pair_style_bucket(pair, metrics)
        length_key = _word_count_bucket(float(metrics.get("words", 0.0)))
        keyed_rows.append((source_key, group_key, style_key, length_key))
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        style_counts[style_key] = style_counts.get(style_key, 0) + 1
        length_counts[length_key] = length_counts.get(length_key, 0) + 1

    adjusted: List[Tuple[float, float, ChatPair, Dict[str, float]]] = []
    for (budget_value, score, pair, metrics), (source_key, group_key, style_key, length_key) in zip(scored, keyed_rows):
        quality_signal = max(0.0, min(1.0, float(metrics.get("quality_signal", 0.0))))
        source_rarity = _selection_rarity_bonus(source_counts.get(source_key, 1), total)
        group_rarity = _selection_rarity_bonus(group_counts.get(group_key, 1), total)
        style_rarity = _selection_rarity_bonus(style_counts.get(style_key, 1), total)
        length_rarity = _selection_rarity_bonus(length_counts.get(length_key, 1), total)
        diversity_bonus = (
            0.10 * source_rarity
            + 0.18 * group_rarity
            + 0.08 * style_rarity
            + 0.04 * length_rarity
        ) * (0.60 + 0.40 * quality_signal)
        adjusted_score = float(score) + float(diversity_bonus)
        estimated_tokens = float(metrics.get("estimated_tokens", 0.0))
        adjusted_budget = float(adjusted_score)
        if str(budget_mode).strip().lower() == "tokens":
            adjusted_budget = float(adjusted_score) / math.pow(max(8.0, estimated_tokens), float(budget_power))
        next_metrics = dict(metrics)
        next_metrics["source_key"] = source_key
        next_metrics["group_key"] = group_key
        next_metrics["style_key"] = style_key
        next_metrics["length_key"] = length_key
        next_metrics["source_rarity"] = float(source_rarity)
        next_metrics["group_rarity"] = float(group_rarity)
        next_metrics["style_rarity"] = float(style_rarity)
        next_metrics["length_rarity"] = float(length_rarity)
        next_metrics["diversity_bonus"] = float(diversity_bonus)
        next_metrics["score"] = float(adjusted_score)
        next_metrics["budget_value"] = float(adjusted_budget)
        adjusted.append((float(adjusted_budget), float(adjusted_score), pair, next_metrics))
    return adjusted


def _is_scoped_sft_selection_pair(
    pair: ChatPair,
    scope: str,
    scope_min_words: int,
) -> bool:
    scope_norm = str(scope).strip().lower()
    if scope_norm in {"", "all"}:
        return True

    assistant_words = max(1, _word_token_count(_fast_cleanup_response_text(pair.assistant)))
    source_low = str(pair.source or "").strip().lower()
    is_teacher = source_low == "supermix_teacher"
    is_synthetic = _is_synthetic_template_prompt(pair.user)

    if scope_norm == "verbose_synthetic_teacher":
        return bool((is_teacher or is_synthetic) and assistant_words >= max(1, int(scope_min_words)))

    return True


def _select_sft_training_pairs(
    pairs: Sequence[ChatPair],
    strategy: str,
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
    hardness_target: float,
    hardness_bandwidth: float,
    budget_mode: str = "pairs",
    budget_power: float = 0.5,
    scope: str = "all",
    scope_min_words: int = 40,
) -> List[ChatPair]:
    mode = str(strategy).strip().lower()
    if mode not in {"none", "utility_topk", "capacity_aware", "coverage_topk"}:
        mode = "none"
    if mode == "none" or not pairs:
        return list(pairs)

    scope = str(scope).strip().lower()
    if scope not in {"all", "verbose_synthetic_teacher"}:
        scope = "all"
    scope_min_words = max(1, int(scope_min_words))
    budget_mode = str(budget_mode).strip().lower()
    if budget_mode not in {"pairs", "tokens"}:
        budget_mode = "pairs"
    budget_power = max(0.0, min(1.0, float(budget_power)))
    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    min_keep = max(0, int(min_keep))
    max_keep = max(0, int(max_keep))
    if keep_ratio >= 0.999 and max_keep <= 0:
        return list(pairs)

    passthrough_ids = set()
    selectable_pairs: List[ChatPair] = []
    for pair in pairs:
        if _is_scoped_sft_selection_pair(pair, scope=scope, scope_min_words=scope_min_words):
            selectable_pairs.append(pair)
        else:
            passthrough_ids.add(id(pair))
    if not selectable_pairs:
        return list(pairs)

    if len(selectable_pairs) != len(pairs):
        print(
            "[sft] pair selection scope: "
            f"scope={scope} candidates={len(selectable_pairs)}/{len(pairs)} "
            f"passthrough={len(pairs) - len(selectable_pairs)} min_words={scope_min_words}"
        )

    scored: List[Tuple[float, float, ChatPair, Dict[str, float]]] = []
    for pair in selectable_pairs:
        score, metrics = _sft_pair_selection_score(
            pair,
            mode=mode,
            hardness_target=float(hardness_target),
            hardness_bandwidth=float(hardness_bandwidth),
        )
        if metrics:
            estimated_tokens = float(_estimated_sft_pair_tokens(pair))
            budget_value = float(score)
            if budget_mode == "tokens":
                budget_value = float(score) / math.pow(max(8.0, estimated_tokens), budget_power)
            metrics["estimated_tokens"] = float(estimated_tokens)
            metrics["budget_value"] = float(budget_value)
            scored.append((float(budget_value), float(score), pair, metrics))
    if not scored:
        return list(pairs)

    if mode == "coverage_topk":
        scored = _apply_sft_coverage_topk_scores(
            scored,
            budget_mode=str(budget_mode),
            budget_power=float(budget_power),
        )

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    total = len(scored)
    selected: List[Tuple[float, float, ChatPair, Dict[str, float]]] = []
    total_estimated_tokens = float(sum(float(entry[3].get("estimated_tokens", 0.0)) for entry in scored))
    selected_estimated_tokens = 0.0

    if budget_mode == "tokens" and keep_ratio < 0.999:
        token_budget = max(0.0, total_estimated_tokens * keep_ratio)
        min_keep_target = min(min_keep, total) if min_keep > 0 else 0
        for entry in scored:
            if max_keep > 0 and len(selected) >= max_keep:
                break
            entry_tokens = float(entry[3].get("estimated_tokens", 0.0))
            must_keep = len(selected) < max(1, min_keep_target)
            under_budget = (selected_estimated_tokens + entry_tokens) <= token_budget
            if not selected or must_keep or under_budget:
                selected.append(entry)
                selected_estimated_tokens += entry_tokens
        if min_keep_target > 0 and len(selected) < min_keep_target:
            selected_ids = {id(entry[2]) for entry in selected}
            for entry in scored:
                if id(entry[2]) in selected_ids:
                    continue
                selected.append(entry)
                selected_estimated_tokens += float(entry[3].get("estimated_tokens", 0.0))
                selected_ids.add(id(entry[2]))
                if len(selected) >= min_keep_target:
                    break
        if not selected:
            selected = [scored[0]]
            selected_estimated_tokens = float(selected[0][3].get("estimated_tokens", 0.0))
    else:
        keep_n = total
        if keep_ratio < 0.999:
            keep_n = max(1, int(round(total * keep_ratio)))
        if max_keep > 0:
            keep_n = min(keep_n, max_keep)
        if min_keep > 0:
            keep_n = max(keep_n, min(min_keep, total))
        keep_n = max(1, min(total, keep_n))
        selected = scored[:keep_n]
        selected_estimated_tokens = float(
            sum(float(entry[3].get("estimated_tokens", 0.0)) for entry in selected)
        )

    def _mean(metric_name: str, rows: Sequence[Tuple[float, float, ChatPair, Dict[str, float]]]) -> float:
        if not rows:
            return 0.0
        vals = [float(entry[3].get(metric_name, 0.0)) for entry in rows]
        return float(sum(vals) / float(len(vals)))

    print(
        "[sft] pair selection: "
        f"strategy={mode} scope={scope} budget_mode={budget_mode} keep={len(selected)}/{total} "
        f"keep_ratio={len(selected)/max(1,total):.3f} "
        f"quality={_mean('quality', scored):.3f}->{_mean('quality', selected):.3f} "
        f"density={_mean('density', scored):.3f}->{_mean('density', selected):.3f} "
        f"reason={_mean('reasoning', scored):.3f}->{_mean('reasoning', selected):.3f} "
        f"difficulty={_mean('difficulty', scored):.3f}->{_mean('difficulty', selected):.3f} "
        f"compactness={_mean('compactness', scored):.3f}->{_mean('compactness', selected):.3f} "
        f"words={_mean('words', scored):.1f}->{_mean('words', selected):.1f} "
        f"est_tokens={total_estimated_tokens:.0f}->{selected_estimated_tokens:.0f} "
        f"budget_power={budget_power:.2f} "
        f"selected_score_mean={_mean('score', selected):.4f}"
        + (
            f" diversity_bonus={_mean('diversity_bonus', scored):.4f}->{_mean('diversity_bonus', selected):.4f} "
            f"coverage_sources={len({str(entry[3].get('source_key', '')) for entry in scored})}"
            f"->{len({str(entry[3].get('source_key', '')) for entry in selected})} "
            f"coverage_groups={len({str(entry[3].get('group_key', '')) for entry in scored})}"
            f"->{len({str(entry[3].get('group_key', '')) for entry in selected})} "
            f"coverage_styles={len({str(entry[3].get('style_key', '')) for entry in scored})}"
            f"->{len({str(entry[3].get('style_key', '')) for entry in selected})}"
            if mode == "coverage_topk"
            else ""
        )
    )
    selected_ids = {id(pair) for _budget_value, _score, pair, _metrics in selected}
    return [pair for pair in pairs if id(pair) in passthrough_ids or id(pair) in selected_ids]


def filter_sft_training_pairs(
    pairs: Sequence[ChatPair],
    min_quality_score: float,
    keep_short_answer_prompts: bool,
    exempt_sources: Optional[Sequence[str]] = None,
    drop_synthetic_prompts: bool = False,
    min_keep_pairs: int = 32,
) -> List[ChatPair]:
    threshold = float(min_quality_score)
    exempt = {str(x or "").strip().lower() for x in (exempt_sources or []) if str(x or "").strip()}
    if threshold <= -1e8:
        if not bool(drop_synthetic_prompts):
            return list(pairs)

    kept: List[ChatPair] = []
    dropped_quality = 0
    dropped_short = 0
    dropped_synthetic = 0
    for pair in pairs:
        src_low = str(pair.source or "").strip().lower()
        if src_low in exempt:
            kept.append(pair)
            continue
        if bool(drop_synthetic_prompts) and _is_synthetic_template_prompt(pair.user):
            dropped_synthetic += 1
            continue
        if bool(keep_short_answer_prompts) and _is_short_answer_prompt(pair.user):
            kept.append(pair)
            continue
        if (not keep_short_answer_prompts) and _is_short_answer_prompt(pair.user):
            dropped_short += 1
        paired_score, _alignment = _paired_response_score(pair.user, pair.assistant)
        if paired_score < threshold:
            dropped_quality += 1
            continue
        kept.append(pair)

    if len(kept) < max(8, int(min_keep_pairs)):
        print(
            "[sft] quality filter fallback: "
            f"kept={len(kept)} too small; using unfiltered set={len(pairs)}"
        )
        return list(pairs)

    print(
        "[sft] quality filter: "
        f"threshold={threshold:.2f} kept={len(kept)} dropped_quality={dropped_quality} "
        f"dropped_short={dropped_short} dropped_synthetic={dropped_synthetic} "
        f"exempt_sources={len(exempt)}"
    )
    return kept


def filter_eval_pairs(
    pairs: Sequence[ChatPair],
    min_quality_score: float,
    drop_synthetic_prompts: bool,
    min_keep_pairs: int = 64,
) -> List[ChatPair]:
    threshold = float(min_quality_score)
    if threshold <= -1e8 and not bool(drop_synthetic_prompts):
        return list(pairs)

    kept: List[ChatPair] = []
    non_synthetic_ranked: List[Tuple[float, ChatPair]] = []
    dropped_quality = 0
    dropped_synthetic = 0
    for pair in pairs:
        if bool(drop_synthetic_prompts) and _is_synthetic_template_prompt(pair.user):
            dropped_synthetic += 1
            continue
        paired_score, _alignment = _paired_response_score(pair.user, pair.assistant)
        non_synthetic_ranked.append((paired_score, pair))
        if threshold > -1e8 and paired_score < threshold:
            dropped_quality += 1
            continue
        kept.append(pair)

    if len(kept) < max(8, int(min_keep_pairs)) and non_synthetic_ranked:
        target_keep = min(len(non_synthetic_ranked), max(8, int(min_keep_pairs)))
        non_synthetic_ranked.sort(key=lambda x: x[0], reverse=True)
        kept = [pair for _score, pair in non_synthetic_ranked[:target_keep]]
        print(
            "[eval] quality filter fallback: "
            f"kept={len(kept)} after ranking top non-synthetic pairs "
            f"(threshold={threshold:.2f}, dropped_quality={dropped_quality}, "
            f"dropped_synthetic={dropped_synthetic})"
        )
        return kept

    print(
        "[eval] quality filter: "
        f"threshold={threshold:.2f} kept={len(kept)} dropped_quality={dropped_quality} "
        f"dropped_synthetic={dropped_synthetic}"
    )
    return kept


def _compute_source_balance_factors(
    pairs: Sequence[ChatPair],
    strength: float,
    max_scale: float,
) -> Dict[str, float]:
    if not pairs:
        return {}
    src_counts: Dict[str, int] = {}
    for p in pairs:
        src = str(p.source or "dataset")
        src_counts[src] = src_counts.get(src, 0) + 1
    if not src_counts:
        return {}
    target = float(len(pairs)) / float(len(src_counts))
    max_s = max(1.0, float(max_scale))
    strength = max(0.0, float(strength))
    factors: Dict[str, float] = {}
    for src, count in src_counts.items():
        raw = (target / max(1.0, float(count))) ** strength
        factors[src] = float(max(1.0 / max_s, min(max_s, raw)))
    return factors


def _conversation_prompt_state(user_text: str) -> Tuple[List[Tuple[str, str]], str, str]:
    raw = _normalize_whitespace(_coerce_text(user_text))
    if not raw:
        return [], "", ""

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    history: List[Tuple[str, str]] = []
    latest_user = raw
    last_assistant = ""
    pending_user: Optional[str] = None
    parsed_role_lines = False
    for line in lines:
        match = CONTEXT_ROLE_LINE_RE.match(line)
        if not match:
            continue
        parsed_role_lines = True
        role = str(match.group(1)).strip().lower()
        content = str(match.group(2)).strip()
        if not content:
            continue
        if role == "user":
            latest_user = content
            pending_user = content
            continue
        if role == "assistant":
            last_assistant = content
            if pending_user:
                history.append((pending_user, content))
                pending_user = None
    if not parsed_role_lines:
        return [], raw, ""
    return history, latest_user, last_assistant


def _significant_token_set(text: str, max_tokens: int = 96) -> set:
    out = set()
    for tok in SIGNIFICANT_TOKEN_RE.findall(_coerce_text(text).lower())[:max_tokens]:
        if tok in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in tok):
            out.add(tok)
            continue
        if len(tok) >= 3:
            out.add(tok)
    return out


def _prompt_ambiguity_profile(latest_user: str, has_anchor: bool) -> Dict[str, bool]:
    text = _normalize_whitespace(_coerce_text(latest_user))
    low = text.lower()
    content_terms = _significant_token_set(text, max_tokens=24)
    explicit_target = bool(
        EXPLICIT_TARGET_RE.search(text)
        or "`" in text
        or "\n" in text
        or (":" in text and len(text) >= 20)
    )
    short_query = len(content_terms) <= 2 and len(text.split()) <= 5
    vague_followup = bool(FOLLOWUP_HINT_RE.search(low)) and len(content_terms) <= 4
    generic_edit = bool(AMBIGUOUS_EDIT_RE.search(low))
    conflict = bool(SHORTEN_HINT_RE.search(low) and EXPAND_HINT_RE.search(low))
    missing_anchor = (not bool(has_anchor)) and (generic_edit or vague_followup or short_query)
    ambiguous = bool(conflict or (missing_anchor and not explicit_target))
    return {
        "ambiguous": ambiguous,
        "missing_anchor": missing_anchor,
        "conflict": conflict,
        "generic_edit": generic_edit,
    }


def _clarification_response_signal(text: str) -> float:
    low = _fast_cleanup_response_text(text).lower()
    if not low:
        return 0.0
    score = 0.0
    if "?" in low:
        score += 0.35
    if CLARIFICATION_RESPONSE_RE.search(low):
        score += 0.45
    if re.search(r"\b(paste|share|point me to|tell me which|show me)\b", low):
        score += 0.15
    return float(min(1.0, score))


def _response_reasoning_signal(text: str) -> float:
    low = _fast_cleanup_response_text(text).lower()
    if not low:
        return 0.0
    score = 0.0
    if re.search(r"(^|\s)(1\)|1\.|first\b|step\s*1)", low):
        score += 0.35
    if re.search(r"\b(then|next|finally|therefore|thus|because|since|so)\b", low):
        score += 0.30
    if re.search(r"\b(let me|let's|consider|assume|suppose|verify)\b", low):
        score += 0.20
    if re.search(r"\b(tradeoff|edge case|counterexample|check)\b", low):
        score += 0.15
    return float(min(1.0, score))


def _response_creativity_signal(text: str) -> float:
    low = _fast_cleanup_response_text(text).lower()
    if not low:
        return 0.0
    score = 0.0
    if re.search(r"\b(imagine|picture|creative|story|invent|novel|poem|vivid)\b", low):
        score += 0.35
    if re.search(r"\b(analogy|metaphor|like\b|as if)\b", low):
        score += 0.30
    if re.search(r"\b(brainstorm|possibility|perspective|angle)\b", low):
        score += 0.20
    if "?" in low:
        score += 0.10
    token_words = re.findall(r"[a-z0-9']+", low)
    if token_words:
        uniq_ratio = len(set(token_words)) / float(len(token_words))
        score += max(0.0, min(0.15, uniq_ratio - 0.55))
    return float(min(1.0, score))


def _response_knowledge_density_score(text: str) -> float:
    raw = _fast_cleanup_response_text(text)
    if not raw:
        return 0.0
    low = raw.lower()
    tokens = re.findall(r"[a-z0-9_+#./'-]+", low)
    if not tokens:
        return 0.0

    content_tokens = [
        tok
        for tok in tokens
        if tok not in STOPWORDS and (len(tok) >= 4 or any(ch.isdigit() for ch in tok))
    ]
    content_ratio = float(len(content_tokens)) / float(max(1, len(tokens)))
    unique_content_ratio = (
        float(len(set(content_tokens))) / float(max(1, len(content_tokens)))
        if content_tokens
        else 0.0
    )
    numeric_score = min(1.0, float(len(NUMBER_RE.findall(raw))) / 4.0)
    dense_token_score = min(1.0, float(len(KNOWLEDGE_DENSE_TOKEN_RE.findall(raw))) / 4.0)
    structure_score = min(1.0, float(len(KNOWLEDGE_STRUCTURE_RE.findall(raw))) / 3.0)
    long_term_score = 0.0
    if content_tokens:
        long_terms = sum(1 for tok in content_tokens if len(tok) >= 8 or any(ch.isdigit() for ch in tok))
        long_term_score = min(1.0, float(long_terms) / float(max(1, len(content_tokens) // 2)))

    word_count = max(1, _word_token_count(raw))
    if word_count <= 5:
        brevity_factor = 0.30
    elif word_count <= 16:
        brevity_factor = 0.76
    elif word_count <= 120:
        brevity_factor = 1.0
    else:
        brevity_factor = max(0.58, 1.0 - (float(word_count - 120) / 260.0))

    filler_hits = sum(1 for phrase in LOW_DENSITY_FILLER_PHRASES if phrase in low)
    score = (
        0.34 * content_ratio
        + 0.16 * unique_content_ratio
        + 0.14 * numeric_score
        + 0.14 * dense_token_score
        + 0.12 * structure_score
        + 0.10 * long_term_score
    )
    score *= brevity_factor
    score -= 0.08 * min(3, filler_hits)
    return float(max(0.0, min(1.0, score)))


def _pair_knowledge_density_score(user_text: str, assistant_text: str, source: str = "") -> float:
    response_density = _response_knowledge_density_score(assistant_text)
    prompt_complexity = min(1.0, _prompt_complexity_score(user_text) / 1.35)
    alignment = _response_alignment_metrics(user_text, assistant_text)
    source_low = str(source or "").lower()

    source_bonus = 0.0
    if "coding_knowledge" in source_low:
        source_bonus += 0.12
    elif "quality_anchor" in source_low:
        source_bonus += 0.08
    elif "science" in source_low or "dictionary" in source_low:
        source_bonus += 0.06
    elif "world_events" in source_low:
        source_bonus += 0.05

    if _looks_like_coding_prompt(user_text):
        source_bonus += 0.06
    if _looks_like_reasoning_prompt(user_text):
        source_bonus += 0.05

    score = (
        0.56 * response_density
        + 0.18 * prompt_complexity
        + 0.10 * min(1.0, max(0.0, float(alignment.reasoning)))
        + 0.08 * min(1.0, max(0.0, float(alignment.conversation)))
        + 0.05 * min(1.0, max(0.0, float(alignment.creativity)))
        + source_bonus
    )
    return float(max(0.0, min(1.0, score)))


def _response_quality_score(text: str) -> float:
    t = _fast_cleanup_response_text(str(text or "")).strip()
    if not t:
        return -1e9
    if _looks_like_placeholder_assistant(t):
        return -3.0

    score = 0.0
    token_words = re.findall(r"[a-z0-9']+", t.lower())
    word_count = len(token_words)
    char_count = len(t)

    if word_count < 4:
        score -= 1.5
    else:
        score += min(1.5, float(word_count) / 30.0)

    if char_count > 900:
        score -= min(2.0, float(char_count - 900) / 240.0)

    if ARTIFACT_TAG_RE.search(t):
        score -= 2.0
    if "[[" in t or "]]" in t:
        score -= 1.0
    artifact_hits = _synthetic_artifact_hits(t)
    if artifact_hits > 0:
        score -= min(2.4, 0.9 * float(artifact_hits))

    low = t.lower()
    for bad in ("assistant:", "user:", "as an ai", "i cannot", "i'm unable"):
        if bad in low:
            score -= 0.8

    if token_words:
        uniq_ratio = len(set(token_words)) / float(len(token_words))
        score += (uniq_ratio - 0.45) * 2.0

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if len(lines) > 1:
        line_uniq_ratio = len(set(lines)) / float(len(lines))
        score += (line_uniq_ratio - 0.5)

    return float(score)


def _response_alignment_metrics(user_text: str, assistant_text: str) -> PairAlignmentMetrics:
    history, latest_user, last_assistant = _conversation_prompt_state(user_text)
    response = _fast_cleanup_response_text(assistant_text)
    if not response:
        return PairAlignmentMetrics()

    latest_tokens = _significant_token_set(latest_user)
    response_tokens = _significant_token_set(response)
    coverage = 0.0
    if latest_tokens:
        coverage = float(len(latest_tokens & response_tokens)) / float(max(1, len(latest_tokens)))

    prompt_numbers = set(NUMBER_RE.findall(latest_user))
    response_numbers = set(NUMBER_RE.findall(response))
    number_coverage = 0.0
    if prompt_numbers:
        number_coverage = float(len(prompt_numbers & response_numbers)) / float(max(1, len(prompt_numbers)))

    latest_low = latest_user.lower()
    followup = bool(history) and bool(FOLLOWUP_HINT_RE.search(latest_low))
    shorten = bool(SHORTEN_HINT_RE.search(latest_low))
    expand = bool(EXPAND_HINT_RE.search(latest_low))
    rewrite = bool(REWRITE_HINT_RE.search(latest_low))
    wants_creative = bool(CREATIVE_REQUEST_RE.search(latest_low))
    wants_reasoning = bool(REASONING_REQUEST_RE.search(latest_low) or _looks_like_reasoning_prompt(latest_user))
    ambiguity = _prompt_ambiguity_profile(latest_user, has_anchor=bool(last_assistant))
    clarification = _clarification_response_signal(response) if ambiguity["ambiguous"] else 0.0

    conversation = 0.55 * coverage + 0.18 * number_coverage
    reasoning = _response_reasoning_signal(response) if wants_reasoning else 0.0
    creativity = _response_creativity_signal(response) if wants_creative else 0.0
    constraint = 0.0

    if followup and last_assistant:
        prev_tokens = _significant_token_set(last_assistant)
        prev_overlap = 0.0
        if prev_tokens:
            prev_overlap = float(len(prev_tokens & response_tokens)) / float(max(1, len(prev_tokens)))
        prev_numbers = set(NUMBER_RE.findall(last_assistant))
        prev_number_overlap = 0.0
        if prev_numbers:
            prev_number_overlap = float(len(prev_numbers & response_numbers)) / float(max(1, len(prev_numbers)))
        prev_word_count = max(1, _word_token_count(last_assistant))
        resp_word_count = max(1, _word_token_count(response))
        len_ratio = float(resp_word_count) / float(prev_word_count)
        conversation += 0.18 * min(1.0, prev_overlap / 0.45)
        constraint += 0.22 * min(1.0, prev_overlap / 0.45) + 0.10 * prev_number_overlap
        if shorten:
            if len_ratio < 0.90:
                shorten_gain = 0.28 * min(1.0, (0.90 - len_ratio) / 0.55)
                conversation += shorten_gain
                constraint += shorten_gain
            else:
                shorten_penalty = 0.10 * min(1.0, (len_ratio - 0.90) / 0.80)
                conversation -= shorten_penalty
                constraint -= shorten_penalty
        elif expand:
            if len_ratio > 1.05:
                expand_gain = 0.24 * min(1.0, (len_ratio - 1.05) / 1.10)
                conversation += expand_gain
                constraint += expand_gain
            conversation += 0.18 * reasoning
            constraint += 0.14 * reasoning
        elif rewrite:
            sim = float(SequenceMatcher(None, last_assistant.lower(), response.lower()).ratio())
            if 0.18 <= sim <= 0.94:
                conversation += 0.18
                constraint += 0.18
            elif sim > 0.98:
                conversation -= 0.10
                constraint -= 0.10
        elif wants_creative:
            conversation += 0.16 * creativity
            constraint += 0.12 * creativity
        if wants_reasoning:
            constraint += 0.12 * reasoning

    if wants_reasoning:
        conversation += 0.12 * reasoning
    if wants_creative:
        conversation += 0.10 * creativity
    if ambiguity["ambiguous"]:
        conversation += 0.22 * clarification
        if clarification < 0.12 and coverage < 0.08:
            conversation -= 0.16
    constraint = float(max(-0.30, min(1.40, constraint)))

    conversation = float(max(-0.30, min(1.60, conversation)))
    return PairAlignmentMetrics(
        conversation=conversation,
        reasoning=float(max(0.0, min(1.0, reasoning))),
        creativity=float(max(0.0, min(1.0, creativity))),
        clarification=float(max(0.0, min(1.0, clarification))),
        constraint=float(max(0.0, min(1.0, constraint))),
        is_ambiguous=bool(ambiguity["ambiguous"]),
        is_followup=bool(followup),
    )


def _paired_response_score(user_text: str, assistant_text: str) -> Tuple[float, PairAlignmentMetrics]:
    alignment = _response_alignment_metrics(user_text, assistant_text)
    score = _response_quality_score(assistant_text)
    score += 0.32 * float(alignment.conversation)
    score += 0.18 * float(alignment.reasoning)
    score += 0.16 * float(alignment.creativity)
    score += 0.14 * float(alignment.constraint)
    if alignment.is_ambiguous:
        score += 0.18 * float(alignment.clarification)
    if alignment.is_followup:
        score += 0.06 * max(0.0, float(alignment.conversation))
    return float(score), alignment


def _preference_pair_weight(
    chosen_quality: float,
    rejected_quality: float,
    rejected_similarity: float,
    chosen_text: str,
    rejected_text: str,
    short_reject_boost: float = 0.55,
    long_reject_boost: float = 0.25,
) -> float:
    gap = max(0.0, float(chosen_quality) - float(rejected_quality))
    hardness = max(0.0, min(1.0, float(rejected_similarity)))
    base = 1.0 + 0.9 * min(1.0, gap / 2.5) + 0.35 * hardness

    chosen_words = max(1, len(re.findall(r"[a-z0-9']+", str(chosen_text).lower())))
    rejected_words = max(1, len(re.findall(r"[a-z0-9']+", str(rejected_text).lower())))
    length_ratio = float(rejected_words) / float(chosen_words)
    if length_ratio < 0.72:
        short_gap = (0.72 - length_ratio) / 0.72
        base += float(short_reject_boost) * min(1.0, short_gap)
    elif length_ratio > 1.55:
        long_gap = (length_ratio - 1.55) / 1.55
        base += float(long_reject_boost) * min(1.0, long_gap)

    if float(chosen_quality) + 0.25 < float(rejected_quality):
        base *= 0.7
    return float(max(0.75, min(2.75, base)))


def _preference_selection_score(
    pair: PreferencePair,
    strategy: str,
    hardness_target: float,
    hardness_bandwidth: float,
) -> float:
    mode = str(strategy or "none").strip().lower()
    if mode == "none":
        return float(max(0.25, pair.weight))

    quality_gap = max(0.0, float(pair.quality_gap))
    rejected_similarity = max(0.0, min(1.0, float(pair.rejected_similarity)))
    prompt_complexity = max(0.0, float(pair.prompt_complexity))
    base_weight = max(0.25, float(pair.weight))
    conversation_score = max(0.0, float(pair.conversation_score))
    reasoning_score = max(0.0, float(pair.reasoning_score))
    creativity_score = max(0.0, float(pair.creativity_score))
    knowledge_density_score = max(0.0, float(pair.knowledge_density_score))
    followup_bonus = 0.10 if bool(pair.is_followup) else 0.0

    if mode == "margin_topk":
        return float(base_weight * (quality_gap + 0.35 * rejected_similarity + 0.08 * prompt_complexity))

    if mode == "capacity_aware":
        target = max(0.0, min(1.0, float(hardness_target)))
        bw = max(0.05, float(hardness_bandwidth))
        z = (rejected_similarity - target) / bw
        hardness_window = math.exp(-0.5 * z * z)
        return float(base_weight * hardness_window * (quality_gap + 0.12 * prompt_complexity + 0.05))

    if mode == "innovation_mix":
        target = max(0.0, min(1.0, float(hardness_target)))
        bw = max(0.05, float(hardness_bandwidth))
        z = (rejected_similarity - target) / bw
        hardness_window = math.exp(-0.5 * z * z)
        novelty_bonus = (
            0.12 * conversation_score
            + 0.10 * reasoning_score
            + 0.10 * creativity_score
            + 0.12 * knowledge_density_score
            + followup_bonus
        )
        return float(
            base_weight
            * (0.35 + 0.65 * hardness_window)
            * (quality_gap + 0.12 * prompt_complexity + novelty_bonus + 0.05)
        )

    if mode == "coverage_margin":
        target = max(0.0, min(1.0, float(hardness_target)))
        bw = max(0.05, float(hardness_bandwidth))
        z = (rejected_similarity - target) / bw
        hardness_window = math.exp(-0.5 * z * z)
        novelty_bonus = (
            0.10 * conversation_score
            + 0.12 * reasoning_score
            + 0.10 * creativity_score
            + 0.12 * knowledge_density_score
            + followup_bonus
        )
        return float(
            base_weight
            * (0.25 + 0.75 * hardness_window)
            * (quality_gap + 0.14 * prompt_complexity + novelty_bonus + 0.05)
        )

    return float(max(0.25, pair.weight))


def _preference_pair_style_bucket(pair: PreferencePair) -> str:
    if _looks_like_coding_prompt(pair.user):
        return "coding"
    style_scores = {
        "reasoning": float(pair.reasoning_score),
        "knowledge": float(pair.knowledge_density_score),
        "creative": float(pair.creativity_score),
        "conversation": float(pair.conversation_score),
    }
    return max(style_scores.items(), key=lambda item: (item[1], item[0]))[0]


def _preference_pair_hardness_bucket(pair: PreferencePair) -> str:
    similarity = max(0.0, min(1.0, float(pair.rejected_similarity)))
    quality_gap = max(0.0, float(pair.quality_gap))
    if similarity >= 0.72 and quality_gap <= 0.18:
        return "near_miss"
    if similarity >= 0.46:
        return "hard"
    if similarity >= 0.22:
        return "mid"
    return "easy"


def _apply_preference_coverage_scores(
    scored: Sequence[Tuple[float, PreferencePair]],
) -> List[Tuple[float, PreferencePair]]:
    total = len(scored)
    if total <= 1:
        return list(scored)

    signature_counts: Dict[str, int] = {}
    style_counts: Dict[str, int] = {}
    hardness_counts: Dict[str, int] = {}
    length_counts: Dict[str, int] = {}
    keyed_rows: List[Tuple[str, str, str, str]] = []
    for _score, pair in scored:
        signature_key = _prompt_signature(pair.user)[:96] or "<empty>"
        style_key = _preference_pair_style_bucket(pair)
        hardness_key = _preference_pair_hardness_bucket(pair)
        length_key = _word_count_bucket(
            max(
                _word_token_count(_fast_cleanup_response_text(pair.chosen)),
                _word_token_count(_fast_cleanup_response_text(pair.rejected)),
            )
        )
        keyed_rows.append((signature_key, style_key, hardness_key, length_key))
        signature_counts[signature_key] = signature_counts.get(signature_key, 0) + 1
        style_counts[style_key] = style_counts.get(style_key, 0) + 1
        hardness_counts[hardness_key] = hardness_counts.get(hardness_key, 0) + 1
        length_counts[length_key] = length_counts.get(length_key, 0) + 1

    adjusted: List[Tuple[float, PreferencePair]] = []
    for (score, pair), (signature_key, style_key, hardness_key, length_key) in zip(scored, keyed_rows):
        signature_rarity = _selection_rarity_bonus(signature_counts.get(signature_key, 1), total)
        style_rarity = _selection_rarity_bonus(style_counts.get(style_key, 1), total)
        hardness_rarity = _selection_rarity_bonus(hardness_counts.get(hardness_key, 1), total)
        length_rarity = _selection_rarity_bonus(length_counts.get(length_key, 1), total)
        quality_anchor = 0.60 + 0.40 * min(1.0, max(0.0, float(pair.quality_gap)) / 0.45)
        diversity_bonus = (
            0.10 * signature_rarity
            + 0.08 * style_rarity
            + 0.08 * hardness_rarity
            + 0.04 * length_rarity
        ) * quality_anchor
        pair.selection_score = float(score + diversity_bonus)
        adjusted.append((float(pair.selection_score), pair))
    return adjusted


def _select_preference_pairs(
    pairs: Sequence[PreferencePair],
    strategy: str,
    keep_ratio: float,
    min_keep: int,
    max_keep: int,
    hardness_target: float,
    hardness_bandwidth: float,
) -> List[PreferencePair]:
    if not pairs:
        return []

    mode = str(strategy or "none").strip().lower()
    if mode not in {"none", "margin_topk", "capacity_aware", "innovation_mix", "coverage_margin"}:
        mode = "none"

    keep_ratio = max(0.0, min(1.0, float(keep_ratio)))
    min_keep = max(0, int(min_keep))
    max_keep = max(0, int(max_keep))

    if mode == "none" and keep_ratio >= 0.999 and max_keep <= 0:
        selected = list(pairs)
        for p in selected:
            p.selection_score = float(max(0.25, p.weight))
        return selected

    scored: List[Tuple[float, PreferencePair]] = []
    for pair in pairs:
        score = _preference_selection_score(
            pair=pair,
            strategy=mode,
            hardness_target=float(hardness_target),
            hardness_bandwidth=float(hardness_bandwidth),
        )
        pair.selection_score = float(score)
        scored.append((float(score), pair))
    if mode == "coverage_margin":
        scored = _apply_preference_coverage_scores(scored)
    scored.sort(key=lambda x: x[0], reverse=True)

    total = len(scored)
    keep_n = total
    if keep_ratio < 0.999:
        keep_n = max(1, int(round(total * keep_ratio)))
    if max_keep > 0:
        keep_n = min(keep_n, max_keep)
    if min_keep > 0:
        keep_n = max(keep_n, min(min_keep, total))
    keep_n = max(1, min(total, keep_n))
    selected = [p for _s, p in scored[:keep_n]]

    def _mean(vals: Sequence[float]) -> float:
        if not vals:
            return 0.0
        return float(sum(vals) / float(len(vals)))

    full_gap = _mean([max(0.0, float(p.quality_gap)) for p in pairs])
    full_sim = _mean([max(0.0, min(1.0, float(p.rejected_similarity))) for p in pairs])
    sel_gap = _mean([max(0.0, float(p.quality_gap)) for p in selected])
    sel_sim = _mean([max(0.0, min(1.0, float(p.rejected_similarity))) for p in selected])
    sel_score = _mean([float(p.selection_score) for p in selected])
    full_conv = _mean([max(0.0, float(p.conversation_score)) for p in pairs])
    sel_conv = _mean([max(0.0, float(p.conversation_score)) for p in selected])
    full_reason = _mean([max(0.0, float(p.reasoning_score)) for p in pairs])
    sel_reason = _mean([max(0.0, float(p.reasoning_score)) for p in selected])
    full_creative = _mean([max(0.0, float(p.creativity_score)) for p in pairs])
    sel_creative = _mean([max(0.0, float(p.creativity_score)) for p in selected])
    full_density = _mean([max(0.0, float(p.knowledge_density_score)) for p in pairs])
    sel_density = _mean([max(0.0, float(p.knowledge_density_score)) for p in selected])
    print(
        "[pref] pair selection: "
        f"strategy={mode} keep={len(selected)}/{total} keep_ratio={len(selected)/max(1,total):.3f} "
        f"gap={full_gap:.3f}->{sel_gap:.3f} sim={full_sim:.3f}->{sel_sim:.3f} "
        f"conv={full_conv:.3f}->{sel_conv:.3f} "
        f"reason={full_reason:.3f}->{sel_reason:.3f} "
        f"creative={full_creative:.3f}->{sel_creative:.3f} "
        f"density={full_density:.3f}->{sel_density:.3f} "
        f"selected_score_mean={sel_score:.4f}"
        + (
            f" coverage_styles={len({_preference_pair_style_bucket(p) for p in pairs})}"
            f"->{len({_preference_pair_style_bucket(p) for p in selected})} "
            f"coverage_hardness={len({_preference_pair_hardness_bucket(p) for p in pairs})}"
            f"->{len({_preference_pair_hardness_bucket(p) for p in selected})}"
            if mode == "coverage_margin"
            else ""
        )
    )
    return selected


def _pick_rejected_candidate(
    user_text: str,
    chosen_text: str,
    generated: Sequence[str],
    similarity_threshold: float,
    similarity_min: float = 0.0,
    stop_signal_strength: float = 0.0,
) -> Tuple[str, float, float]:
    candidates: List[Tuple[str, float, float]] = []
    for cand in generated:
        c = _fast_cleanup_response_text(str(cand or "")).strip()
        if not c or c == chosen_text:
            continue
        if _looks_like_placeholder_assistant(c):
            continue
        sim = float(SequenceMatcher(None, chosen_text, c).ratio())
        if sim < float(similarity_min):
            continue
        if sim >= float(similarity_threshold):
            continue
        score, _alignment = _paired_response_score(user_text, c)
        score += _preference_stop_alignment_score(
            user_text=user_text,
            assistant_text=c,
            strength=float(stop_signal_strength),
        )
        candidates.append((c, sim, score))

    if not candidates:
        return "", 0.0, -1e9

    hard = [x for x in candidates if x[1] >= 0.25]
    if hard:
        # Prefer semantically-close but still inferior responses as harder negatives.
        hard.sort(key=lambda x: (-x[1], x[2]))
        return hard[0]

    # Fall back to the lowest-quality generated answer.
    candidates.sort(key=lambda x: x[2])
    return candidates[0]


def _stop_overlong_reject_variants(
    user_text: str,
    chosen_text: str,
    rng: random.Random,
    max_variants: int = 2,
) -> List[str]:
    target_words = _preference_stop_target_words(user_text)
    limit = max(0, int(max_variants))
    text = _fast_cleanup_response_text(str(chosen_text or "")).strip()
    if target_words <= 0 or limit <= 0 or not text:
        return []
    if "```" in text:
        return []
    if _word_token_count(text) > max(12, int(round(1.55 * float(target_words)))):
        return []

    bridge = [
        "The short answer is enough, but the fuller explanation is that this follows from the core idea and standard reasoning.",
        "More explicitly, the same answer still holds when you unpack the details step by step instead of stopping at the concise reply.",
        "To elaborate a bit, this conclusion comes from the main constraint in the prompt and the usual interpretation of the terms.",
    ]
    rng.shuffle(bridge)

    base = text.rstrip()
    if base and base[-1] not in ".!?":
        base += "."
    variants: List[str] = []
    for suffix in bridge[:limit]:
        candidate = _fast_cleanup_response_text(f"{base} {suffix}")
        if candidate and candidate != text:
            variants.append(candidate)

    dedup: List[str] = []
    seen = set()
    for cand in variants:
        norm = _normalize_whitespace(cand.lower())
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(cand)
    return dedup


def _reorder_scored_ids_for_self_play(
    scored_ids: Sequence[Tuple[float, int, str, float, str, float, PairAlignmentMetrics, float]],
    budget: int,
    curriculum: str = "easy_to_hard",
    pool_multiplier: int = 4,
) -> Tuple[List[Tuple[float, int, str, float, str, float, PairAlignmentMetrics, float]], set]:
    budget = max(0, int(budget))
    if budget <= 0 or not scored_ids:
        return list(scored_ids), set()

    mode = str(curriculum or "easy_to_hard").strip().lower()
    if mode not in {"easy_to_hard", "hard_first"}:
        mode = "easy_to_hard"

    pool_size = min(
        len(scored_ids),
        max(budget, max(1, int(pool_multiplier)) * budget),
    )
    candidate_pool = list(scored_ids[:pool_size])
    if mode == "hard_first":
        candidate_pool.sort(key=lambda x: (-float(x[5]), -float(x[0]), int(x[1])))
    else:
        candidate_pool.sort(key=lambda x: (float(x[5]), -float(x[0]), int(x[1])))

    selected = candidate_pool[:budget]
    selected_ids = {int(item[1]) for item in selected}
    if not selected_ids:
        return list(scored_ids), set()

    selected_order = selected
    remainder = [item for item in scored_ids if int(item[1]) not in selected_ids]
    return selected_order + remainder, selected_ids


def _compact_reasoning_variant(
    chosen_text: str,
    max_words: int = 96,
    max_sentences: int = 4,
) -> str:
    text = _fast_cleanup_response_text(str(chosen_text or "")).strip()
    if not text or "```" in text:
        return ""
    if _word_token_count(text) < 40:
        return ""
    if (
        _response_reasoning_signal(text) < 0.15
        and _response_knowledge_density_score(text) < 0.42
    ):
        return ""

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    compact_parts: List[str] = []
    for part in parts:
        candidate = str(part)
        candidate = re.sub(r"^step\s*\d+[\s:.)-]*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(
            r"^(?:first|second|third|next|then|finally|therefore|so|because|to solve this|the key idea is|in short)\b[\s,:-]*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        if _word_token_count(candidate) > 18:
            clauses = [x.strip() for x in re.split(r"(?:;|, and |, but |, so | because | which )", candidate, maxsplit=1, flags=re.IGNORECASE) if x.strip()]
            if clauses:
                candidate = clauses[0]
        candidate = _normalize_whitespace(candidate)
        if _word_token_count(candidate) < 4:
            continue
        compact_parts.append(candidate)
        if len(compact_parts) >= max(1, int(max_sentences)):
            break

    compact = _normalize_whitespace(" ".join(compact_parts))
    if not compact:
        return ""
    for phrase in LOW_DENSITY_FILLER_PHRASES:
        compact = re.sub(re.escape(phrase), "", compact, flags=re.IGNORECASE)
    compact = _normalize_whitespace(compact)
    words = compact.split()
    if len(words) > max(8, int(max_words)):
        compact = " ".join(words[: max(8, int(max_words))]).strip()
    compact = compact.rstrip(",;:-")
    if compact and compact[-1] not in ".!?":
        compact += "."
    compact = _fast_cleanup_response_text(compact)
    if not compact or compact == text:
        return ""
    if _word_token_count(compact) >= _word_token_count(text):
        return ""
    if _word_token_count(compact) < 12:
        return ""
    if _response_knowledge_density_score(compact) < 0.20:
        return ""
    return compact


def _counterfactual_reject_variants(chosen_text: str, rng: random.Random) -> List[str]:
    text = _fast_cleanup_response_text(str(chosen_text or "")).strip()
    if not text:
        return []
    out: List[str] = []

    compact = _compact_reasoning_variant(text)
    if compact and compact != text:
        out.append(compact)

    parts = re.split(r"(?<=[.!?])\s+", text)
    if len(parts) >= 3:
        drop_idx = rng.randrange(len(parts))
        dropped = " ".join(p for i, p in enumerate(parts) if i != drop_idx).strip()
        if dropped and dropped != text:
            out.append(dropped)

    def _nudge_num(match: re.Match) -> str:
        raw = match.group(0)
        try:
            val = int(raw)
        except Exception:
            return raw
        step = 1 if val >= 0 else -1
        if rng.random() < 0.5:
            return str(val + step)
        return str(val - step)

    nudged = re.sub(r"\b-?\d+\b", _nudge_num, text, count=2)
    if nudged != text:
        out.append(nudged)

    flip_map = (
        ("must", "might"),
        ("always", "sometimes"),
        ("never", "often"),
        ("correct", "incorrect"),
        ("optimal", "suboptimal"),
        ("increases", "decreases"),
        ("true", "false"),
    )
    low = text.lower()
    for a, b in flip_map:
        if a in low:
            flipped = re.sub(rf"\b{re.escape(a)}\b", b, text, count=1, flags=re.IGNORECASE)
            if flipped != text:
                out.append(flipped)
            break

    dedup: List[str] = []
    seen = set()
    for cand in out:
        norm = _normalize_whitespace(cand.lower())
        if norm in seen:
            continue
        seen.add(norm)
        dedup.append(cand.strip())
    return dedup


@torch.no_grad()
def build_preference_pairs_with_generation(
    model,
    tokenizer,
    train_pairs: Sequence[ChatPair],
    max_pairs: int,
    max_new_tokens: int,
    prompt_max_tokens: int,
    seed: int,
    similarity_threshold: float = 0.90,
    reject_similarity_min: float = 0.10,
    candidate_count: int = 3,
    include_greedy_candidate: bool = True,
    short_reject_boost: float = 0.55,
    long_reject_boost: float = 0.25,
    min_chosen_quality: float = 0.90,
    min_chosen_words: int = 6,
    min_quality_gap: float = 0.05,
    skip_template_prompts: bool = True,
    max_pairs_per_user: int = 1,
    max_pairs_per_source: int = 0,
    fallback_same_source_first: bool = True,
    mining_mode: str = "auto",
    progress_every: int = 100,
    mining_max_seconds: float = 0.0,
    mining_max_attempt_factor: int = 12,
    coding_focus_boost: float = 1.0,
    reasoning_focus_boost: float = 1.0,
    counterfactual_rejects_per_prompt: int = 2,
    self_play_budget: int = 0,
    self_play_curriculum: str = "easy_to_hard",
    self_play_max_new_tokens: int = 0,
    stop_signal_strength: float = 0.0,
    stop_rejects_per_prompt: int = 0,
    selection_strategy: str = "none",
    selection_keep_ratio: float = 1.0,
    selection_min_keep: int = 0,
    selection_max_keep: int = 0,
    selection_hardness_target: float = 0.45,
    selection_hardness_bandwidth: float = 0.24,
) -> List[PreferencePair]:
    if max_pairs <= 0:
        return []

    rng = random.Random(seed)
    scored_ids: List[Tuple[float, int, str, float, str, float, PairAlignmentMetrics, float]] = []
    for idx, pair in enumerate(train_pairs):
        chosen_text = _fast_cleanup_response_text(pair.assistant)
        if not chosen_text:
            continue
        if _looks_like_placeholder_assistant(chosen_text):
            continue
        if bool(skip_template_prompts) and _is_synthetic_template_prompt(pair.user):
            continue
        short_prompt = _is_short_answer_prompt(pair.user)
        chosen_score, chosen_alignment = _paired_response_score(pair.user, chosen_text)
        chosen_density = _pair_knowledge_density_score(pair.user, chosen_text, source=str(pair.source or ""))
        chosen_words = _word_token_count(chosen_text)
        if not short_prompt and chosen_words < max(1, int(min_chosen_words)):
            continue
        if not short_prompt and chosen_score < float(min_chosen_quality):
            continue
        chosen_stop_score = _preference_stop_alignment_score(
            user_text=pair.user,
            assistant_text=chosen_text,
            strength=float(stop_signal_strength),
        )
        prompt_complexity = _prompt_complexity_score(pair.user)
        score = prompt_complexity + 0.18 * max(-1.0, min(2.5, chosen_score + chosen_stop_score))
        score += 0.14 * float(chosen_alignment.conversation)
        score += 0.10 * float(chosen_alignment.reasoning)
        score += 0.08 * float(chosen_alignment.creativity)
        score += 0.12 * float(chosen_density)
        score += 0.15 * rng.random()
        scored_ids.append(
            (
                score,
                idx,
                chosen_text,
                chosen_score,
                _prompt_signature(pair.user),
                prompt_complexity,
                chosen_alignment,
                chosen_density,
            )
        )
    if not scored_ids:
        return []
    scored_ids.sort(key=lambda x: x[0], reverse=True)
    fallback_answers = [
        _fast_cleanup_response_text(p.assistant)
        for p in train_pairs
        if p.assistant.strip() and not _looks_like_placeholder_assistant(p.assistant)
    ]
    if not fallback_answers:
        return []
    fallback_by_source: Dict[str, List[str]] = {}
    for p in train_pairs:
        ans = _fast_cleanup_response_text(p.assistant)
        if not ans or _looks_like_placeholder_assistant(ans):
            continue
        src_key = str(p.source or "dataset")
        fallback_by_source.setdefault(src_key, []).append(ans)

    device = next(model.parameters()).device
    mining_mode_norm = str(mining_mode).strip().lower()
    if mining_mode_norm not in {"auto", "hybrid", "dataset", "generation"}:
        mining_mode_norm = "auto"
    if mining_mode_norm == "auto":
        use_generation = device.type != "cpu"
    elif mining_mode_norm == "dataset":
        use_generation = False
    else:
        use_generation = True
    self_play_budget = max(0, int(self_play_budget))
    self_play_curriculum = str(self_play_curriculum or "easy_to_hard").strip().lower()
    if self_play_curriculum not in {"easy_to_hard", "hard_first"}:
        self_play_curriculum = "easy_to_hard"
    self_play_max_new_tokens = max(0, int(self_play_max_new_tokens))
    self_play_generation_ids: set = set()
    if not use_generation and self_play_budget > 0:
        scored_ids, self_play_generation_ids = _reorder_scored_ids_for_self_play(
            scored_ids,
            budget=min(self_play_budget, len(scored_ids)),
            curriculum=self_play_curriculum,
        )

    max_pairs = max(1, int(max_pairs))
    progress_every = max(0, int(progress_every))
    mining_max_seconds = max(0.0, float(mining_max_seconds))
    mining_max_attempt_factor = max(0, int(mining_max_attempt_factor))
    max_attempts = len(scored_ids)
    if mining_max_attempt_factor > 0:
        max_attempts = min(max_attempts, max(max_pairs, mining_max_attempt_factor * max_pairs))

    print(
        "[pref] mining config: "
        f"mode={mining_mode_norm} generation={'on' if use_generation else 'off'} "
        f"target_pairs={max_pairs} candidates={len(scored_ids)} "
        f"max_attempts={max_attempts} "
        f"self_play_budget={len(self_play_generation_ids)} "
        f"self_play_curriculum={self_play_curriculum} "
        f"self_play_max_new_tokens={self_play_max_new_tokens if self_play_max_new_tokens > 0 else max_new_tokens} "
        f"selection={str(selection_strategy).strip().lower()} "
        f"keep_ratio={max(0.0, min(1.0, float(selection_keep_ratio))):.3f} "
        f"max_seconds={mining_max_seconds if mining_max_seconds > 0 else 'off'}"
    )

    was_training = bool(model.training)
    prev_use_cache = bool(getattr(getattr(model, "config", None), "use_cache", True))
    model.eval()
    _set_model_use_cache(model, enabled=True)
    out: List[PreferencePair] = []
    user_pair_counts: Dict[str, int] = {}
    source_pair_counts: Dict[str, int] = {}
    temps = (0.65, 0.8, 0.95, 1.1)
    mine_started = time.time()
    visited = 0
    generation_failures = 0
    brevity_filtered = 0
    stop_reject_variants_added = 0
    self_play_prompts_used = 0
    self_play_candidates_generated = 0
    self_play_generation_failures = 0

    try:
        for (
            _score,
            idx,
            chosen_text,
            chosen_score,
            user_sig,
            prompt_complexity,
            chosen_alignment,
            chosen_density,
        ) in scored_ids:
            visited += 1
            if len(out) >= max_pairs:
                break
            if visited > max_attempts:
                print(
                    "[pref] mining stop: "
                    f"visited={visited} exceeded max_attempts={max_attempts} with pairs={len(out)}"
                )
                break
            if mining_max_seconds > 0 and (time.time() - mine_started) >= mining_max_seconds:
                print(
                    "[pref] mining stop: "
                    f"elapsed={time.time() - mine_started:.1f}s reached max_seconds={mining_max_seconds:.1f}s "
                    f"with pairs={len(out)}"
                )
                break
            if user_pair_counts.get(user_sig, 0) >= max(1, int(max_pairs_per_user)):
                continue
            pair = train_pairs[idx]
            prompt_is_coding = _looks_like_coding_prompt(pair.user)
            prompt_is_reasoning = _looks_like_reasoning_prompt(pair.user)
            source_key = str(pair.source or "dataset")
            stop_target_words = _preference_stop_target_words(pair.user)
            if int(max_pairs_per_source) > 0:
                if source_pair_counts.get(source_key, 0) >= int(max_pairs_per_source):
                    continue
            chosen_stop_score = _preference_stop_alignment_score(
                user_text=pair.user,
                assistant_text=chosen_text,
                strength=float(stop_signal_strength),
            )
            if (
                stop_target_words > 0
                and chosen_stop_score < -0.08
                and _word_token_count(chosen_text) > max(6, int(round(1.45 * float(stop_target_words))))
            ):
                brevity_filtered += 1
                continue
            chosen_effective_score = float(chosen_score + chosen_stop_score)
            generated: List[str] = []
            use_self_play_generation = (not use_generation) and (int(idx) in self_play_generation_ids)
            if use_generation or use_self_play_generation:
                prompt = _chat_prompt_only(tokenizer, pair.user)
                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max(32, int(prompt_max_tokens)),
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                try:
                    if use_self_play_generation:
                        self_play_prompts_used += 1
                    gen_max_new_tokens = max(16, int(max_new_tokens))
                    if use_self_play_generation and self_play_max_new_tokens > 0:
                        gen_max_new_tokens = max(16, min(int(max_new_tokens), int(self_play_max_new_tokens)))
                    if bool(include_greedy_candidate) or use_self_play_generation:
                        greedy = model.generate(
                            **enc,
                            max_new_tokens=gen_max_new_tokens,
                            do_sample=False,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        greedy_new_tokens = greedy[0, enc["input_ids"].shape[1] :]
                        greedy_pred = tokenizer.decode(greedy_new_tokens, skip_special_tokens=True).strip()
                        greedy_pred = _fast_cleanup_response_text(greedy_pred)
                        if greedy_pred:
                            generated.append(greedy_pred)
                            if use_self_play_generation:
                                self_play_candidates_generated += 1

                    n_candidates = max(1, int(candidate_count))
                    if use_self_play_generation:
                        n_candidates = min(1, n_candidates)
                    for ci in range(n_candidates):
                        temp = float(temps[ci % len(temps)])
                        top_p = min(0.96, 0.86 + 0.03 * float(ci % 3))
                        gen = model.generate(
                            **enc,
                            max_new_tokens=gen_max_new_tokens,
                            do_sample=True,
                            temperature=temp,
                            top_p=top_p,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        new_tokens = gen[0, enc["input_ids"].shape[1] :]
                        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                        pred = _fast_cleanup_response_text(pred)
                        if pred:
                            generated.append(pred)
                            if use_self_play_generation:
                                self_play_candidates_generated += 1
                except Exception as e:
                    generation_failures += 1
                    if use_self_play_generation:
                        self_play_generation_failures += 1
                    if generation_failures <= 3:
                        print(f"[pref] generation warning #{generation_failures}: {e}")

            if bool(prompt_is_coding or prompt_is_reasoning):
                cf_variants = _counterfactual_reject_variants(chosen_text, rng=rng)
                if int(counterfactual_rejects_per_prompt) > 0:
                    cf_variants = cf_variants[: max(1, int(counterfactual_rejects_per_prompt))]
                generated.extend(cf_variants)
            if int(stop_rejects_per_prompt) > 0:
                stop_variants = _stop_overlong_reject_variants(
                    user_text=pair.user,
                    chosen_text=chosen_text,
                    rng=rng,
                    max_variants=int(stop_rejects_per_prompt),
                )
                stop_reject_variants_added += len(stop_variants)
                generated.extend(stop_variants)

            rejected, rejected_sim, rejected_score = _pick_rejected_candidate(
                user_text=pair.user,
                chosen_text=chosen_text,
                generated=generated,
                similarity_threshold=float(similarity_threshold),
                similarity_min=float(reject_similarity_min),
                stop_signal_strength=float(stop_signal_strength),
            )

            if not rejected:
                candidate_pools: List[List[str]] = []
                if bool(fallback_same_source_first):
                    local_pool = fallback_by_source.get(source_key, [])
                    if local_pool:
                        candidate_pools.append(local_pool)
                candidate_pools.append(fallback_answers)
                fallback_tries = 8
                if prompt_is_coding or prompt_is_reasoning:
                    fallback_tries = 14

                for pool in candidate_pools:
                    if not pool:
                        continue
                    for _ in range(fallback_tries):
                        candidate = pool[rng.randrange(len(pool))].strip()
                        if not candidate or candidate == chosen_text:
                            continue
                        if _looks_like_placeholder_assistant(candidate):
                            continue
                        sim = float(SequenceMatcher(None, chosen_text, candidate).ratio())
                        if sim < float(reject_similarity_min):
                            continue
                        if sim >= float(similarity_threshold):
                            continue
                        rejected = candidate
                        rejected_sim = sim
                        rejected_score, _alignment = _paired_response_score(pair.user, candidate)
                        rejected_score += _preference_stop_alignment_score(
                            user_text=pair.user,
                            assistant_text=candidate,
                            strength=float(stop_signal_strength),
                        )
                        break
                    if rejected:
                        break

            if not rejected:
                continue
            if _looks_like_placeholder_assistant(rejected):
                continue
            if float(chosen_effective_score) < float(rejected_score) + float(min_quality_gap):
                continue

            pair_weight = _preference_pair_weight(
                chosen_quality=chosen_effective_score,
                rejected_quality=rejected_score,
                rejected_similarity=rejected_sim,
                chosen_text=chosen_text,
                rejected_text=rejected,
                short_reject_boost=float(short_reject_boost),
                long_reject_boost=float(long_reject_boost),
            )
            if prompt_is_coding:
                pair_weight *= float(max(0.7, coding_focus_boost))
            if prompt_is_reasoning:
                pair_weight *= float(max(0.7, reasoning_focus_boost))
            pair_weight = float(max(0.75, min(3.25, pair_weight)))
            out.append(
                PreferencePair(
                    user=pair.user,
                    chosen=chosen_text,
                    rejected=rejected,
                    weight=pair_weight,
                    quality_gap=float(max(0.0, float(chosen_effective_score) - float(rejected_score))),
                    rejected_similarity=float(max(0.0, min(1.0, float(rejected_sim)))),
                    prompt_complexity=float(max(0.0, float(prompt_complexity))),
                    conversation_score=float(max(0.0, float(chosen_alignment.conversation))),
                    reasoning_score=float(max(0.0, float(chosen_alignment.reasoning))),
                    creativity_score=float(max(0.0, float(chosen_alignment.creativity))),
                    knowledge_density_score=float(max(0.0, float(chosen_density))),
                    is_followup=bool(chosen_alignment.is_followup),
                )
            )
            user_pair_counts[user_sig] = user_pair_counts.get(user_sig, 0) + 1
            source_pair_counts[source_key] = source_pair_counts.get(source_key, 0) + 1
            if progress_every > 0 and (visited % progress_every == 0 or len(out) % progress_every == 0):
                elapsed = max(1e-6, time.time() - mine_started)
                print(
                    "[pref] mining progress: "
                    f"visited={visited}/{len(scored_ids)} accepted={len(out)} "
                    f"rate={visited / elapsed:.2f}/s"
                )
    finally:
        if was_training:
            model.train()
        _set_model_use_cache(model, enabled=prev_use_cache)
    mined_pairs = len(out)
    out = _select_preference_pairs(
        out,
        strategy=str(selection_strategy),
        keep_ratio=float(selection_keep_ratio),
        min_keep=int(selection_min_keep),
        max_keep=int(selection_max_keep),
        hardness_target=float(selection_hardness_target),
        hardness_bandwidth=float(selection_hardness_bandwidth),
    )
    elapsed = max(1e-6, time.time() - mine_started)
    print(
        "[pref] mining complete: "
        f"pairs={len(out)} mined={mined_pairs} visited={visited} generation_failures={generation_failures} "
        f"brevity_filtered={brevity_filtered} stop_reject_variants={stop_reject_variants_added} "
        f"self_play_prompts={self_play_prompts_used} self_play_candidates={self_play_candidates_generated} "
        f"self_play_failures={self_play_generation_failures} "
        f"elapsed={elapsed:.1f}s"
    )
    return out


@torch.no_grad()
def annotate_preference_rows_with_reference_logps(
    model,
    pref_rows: List[Dict[str, object]],
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    length_bucketed_batches: bool = False,
    length_bucket_window_mult: int = 16,
    seed: int = 0,
) -> int:
    if not pref_rows:
        return 0

    ref_dataset = PackedPreferenceDataset(pref_rows)
    ref_loader = _build_bucketed_dataloader(
        dataset=ref_dataset,
        lengths=[int(row.get("pair_tokens", 1)) for row in pref_rows],
        batch_size=max(1, int(batch_size)),
        collate_fn=lambda b: collate_preference_rows(b, pad_token_id),
        shuffle=False,
        enabled=bool(length_bucketed_batches),
        bucket_window_multiplier=int(length_bucket_window_mult),
        seed=int(seed),
        label="pref-ref",
    )

    was_training = bool(model.training)
    prev_use_cache = bool(getattr(getattr(model, "config", None), "use_cache", True))
    model.eval()
    _set_model_use_cache(model, enabled=True)
    offset = 0
    try:
        for batch in ref_loader:
            chosen_input_ids = batch["chosen_input_ids"].to(device)
            chosen_attention_mask = batch["chosen_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_input_ids = batch["rejected_input_ids"].to(device)
            rejected_attention_mask = batch["rejected_attention_mask"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            out_chosen = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            out_rejected = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)
            chosen_logp, _ = _sequence_average_log_prob(out_chosen.logits, chosen_labels)
            rejected_logp, _ = _sequence_average_log_prob(out_rejected.logits, rejected_labels)

            bs = int(chosen_logp.shape[0])
            for j in range(bs):
                pref_rows[offset + j]["ref_chosen_logp"] = float(chosen_logp[j].item())
                pref_rows[offset + j]["ref_rejected_logp"] = float(rejected_logp[j].item())
            offset += bs
    finally:
        if was_training:
            model.train()
        _set_model_use_cache(model, enabled=prev_use_cache)
    return int(offset)


def finetune_qwen(
    base_model: str,
    train_pairs: Sequence[ChatPair],
    output_dir: Path,
    device: Any,
    runtime_device_requested: str,
    runtime_device_resolved: str,
    runtime_device_preference: str,
    model_dtype: torch.dtype,
    gradient_checkpointing: bool,
    max_length: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float,
    sft_lr_schedule: str,
    sft_warmup_steps: int,
    sft_min_lr_ratio: float,
    sft_max_grad_norm: float,
    train_log_every_steps: int,
    weight_decay: float,
    max_steps: int,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_targets: str,
    use_dora: bool,
    use_rslora: bool,
    lora_init: str,
    lora_plus_ratio: float,
    neftune_noise_alpha: float,
    preference_neftune_noise_alpha: float,
    sft_weight_mode: str,
    sft_min_weight: float,
    sft_max_weight: float,
    sft_synthetic_prompt_weight: float,
    sft_teacher_source_weight: float,
    sft_quality_anchor_boost: float,
    sft_coding_boost: float,
    sft_events_boost: float,
    sft_reasoning_boost: float,
    sft_prompt_skill_boost: float,
    sft_conversation_boost: float,
    sft_creativity_boost: float,
    sft_knowledge_density_boost: float,
    sft_followup_paraphrase_aug: int,
    sft_followup_paraphrase_weight: float,
    sft_rdrop_alpha: float,
    sft_min_quality_score: float,
    sft_filter_keep_short_answers: bool,
    sft_drop_synthetic_prompts: bool,
    sft_quality_filter_exempt_sources: Optional[Sequence[str]],
    sft_auto_balance_sources: bool,
    sft_source_balance_strength: float,
    sft_source_balance_max_scale: float,
    sft_lr_restart_period: int,
    sft_focal_gamma: float,
    sft_eval_every_steps: int,
    sft_early_stop_patience: int,
    sft_curriculum_quality_ramp: float,
    sft_grad_noise_eta: float,
    sft_length_bucketed_batches: bool,
    sft_length_bucket_window_mult: int,
    sft_true_packing: bool,
    sft_packing_max_samples_per_row: int,
    sft_selection_strategy: str,
    sft_selection_keep_ratio: float,
    sft_selection_min_keep: int,
    sft_selection_max_keep: int,
    sft_selection_hardness_target: float,
    sft_selection_hardness_bandwidth: float,
    sft_selection_budget_mode: str,
    sft_selection_budget_power: float,
    sft_selection_scope: str,
    sft_selection_scope_min_words: int,
    eval_pairs: Sequence[ChatPair],
    preference_objective: str,
    preference_steps: int,
    preference_pairs: int,
    preference_beta: float,
    preference_beta_end: float,
    preference_margin: float,
    preference_margin_end: float,
    preference_label_smoothing: float,
    preference_xpo_clip: float,
    preference_sft_weight: float,
    preference_length_weight: float,
    preference_length_control_strength: float,
    preference_length_control_target_ratio: float,
    preference_length_control_max_penalty: float,
    preference_hardness_gamma: float,
    preference_robust_alpha: float,
    preference_robust_eta: float,
    preference_robust_clip: float,
    preference_wpo_alpha: float,
    preference_wpo_clip: float,
    preference_reference_anchor_weight: float,
    preference_reference_anchor_batch_size: int,
    preference_lr: float,
    preference_lr_schedule: str,
    preference_warmup_steps: int,
    preference_min_lr_ratio: float,
    preference_max_grad_norm: float,
    preference_max_new_tokens: int,
    preference_prompt_max_tokens: int,
    preference_reject_similarity_min: float,
    preference_candidate_count: int,
    preference_include_greedy_candidate: bool,
    preference_short_reject_boost: float,
    preference_long_reject_boost: float,
    preference_min_chosen_quality: float,
    preference_min_chosen_words: int,
    preference_min_quality_gap: float,
    preference_skip_template_prompts: bool,
    preference_max_pairs_per_user: int,
    preference_max_pairs_per_source: int,
    preference_fallback_same_source_first: bool,
    preference_mining_mode: str,
    preference_mining_progress_every: int,
    preference_mining_max_seconds: float,
    preference_mining_max_attempt_factor: int,
    preference_coding_focus_boost: float,
    preference_reasoning_focus_boost: float,
    preference_counterfactual_rejects_per_prompt: int,
    preference_self_play_budget: int,
    preference_self_play_curriculum: str,
    preference_self_play_max_new_tokens: int,
    preference_stop_signal_strength: float,
    preference_stop_rejects_per_prompt: int,
    preference_selection_strategy: str,
    preference_selection_keep_ratio: float,
    preference_selection_min_keep: int,
    preference_selection_max_keep: int,
    preference_selection_hardness_target: float,
    preference_selection_hardness_bandwidth: float,
    preference_length_bucketed_batches: bool,
    preference_length_bucket_window_mult: int,
    seed: int,
    save_every_steps: int,
    keep_last_checkpoints: int,
    preference_rescore_every: int,
    skip_sft: bool,
    init_adapter_match_lora: bool,
    init_adapter_dir: Optional[str] = None,
    resume_sft_steps: int = 0,
    resume_preference_steps: int = 0,
    resume_sft_loss_mean: float = 0.0,
    resume_preference_loss_mean: float = 0.0,
) -> Tuple[Path, Dict[str, float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    runtime_backend = _device_backend_name(device=device, resolved_backend=runtime_device_resolved)
    if runtime_backend == "cpu" and bool(gradient_checkpointing):
        print("[train] gradient checkpointing disabled on cpu (stability + no memory benefit).")
        gradient_checkpointing = False
    if runtime_backend == "cpu" and float(sft_rdrop_alpha) > 0.0:
        print("[train] R-Drop disabled on cpu (it doubles forward cost and makes progress look stalled).")
        sft_rdrop_alpha = 0.0

    init_path: Optional[Path] = None
    if init_adapter_dir:
        init_path = Path(str(init_adapter_dir))

    bootstrap_device, bootstrap_dtype, bootstrap_reason = _resolve_peft_bootstrap(
        runtime_device=device,
        resolved_backend=runtime_device_resolved,
        runtime_dtype=model_dtype,
    )
    if bootstrap_reason:
        print(
            "[train] PEFT bootstrap: "
            f"{bootstrap_reason} load_device={bootstrap_device} "
            f"load_dtype={str(bootstrap_dtype)} runtime_device={device} "
            f"runtime_dtype={str(model_dtype)}"
        )
    model, tokenizer = _load_base_model_and_tokenizer(
        base_model,
        bootstrap_device,
        for_training=True,
        model_dtype=bootstrap_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    print(
        "[train] runtime config: "
        f"requested={runtime_device_requested} resolved={runtime_backend} "
        f"device={device} model_dtype={str(model_dtype)} "
        f"gradient_checkpointing={bool(gradient_checkpointing)} "
        f"torch_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"preference={runtime_device_preference}"
    )
    target_modules = _target_modules_from_arg(lora_targets)
    if bool(init_adapter_match_lora) and init_path is not None and init_path.exists():
        init_cfg = _load_init_adapter_config(init_path)
        if init_cfg:
            inherited_target_modules = init_cfg.get("target_modules")
            if isinstance(inherited_target_modules, (list, tuple)):
                matched_targets = [str(x).strip() for x in inherited_target_modules if str(x).strip()]
                if matched_targets:
                    target_modules = matched_targets
            inherited_r = int(init_cfg.get("r", lora_r))
            inherited_alpha = int(init_cfg.get("lora_alpha", lora_alpha))
            inherited_dropout = float(init_cfg.get("lora_dropout", lora_dropout))
            inherited_use_dora = bool(init_cfg.get("use_dora", use_dora))
            inherited_use_rslora = bool(init_cfg.get("use_rslora", use_rslora))
            changed = (
                inherited_r != int(lora_r)
                or inherited_alpha != int(lora_alpha)
                or abs(inherited_dropout - float(lora_dropout)) > 1e-9
                or inherited_use_dora != bool(use_dora)
                or inherited_use_rslora != bool(use_rslora)
            )
            lora_r = inherited_r
            lora_alpha = inherited_alpha
            lora_dropout = inherited_dropout
            use_dora = inherited_use_dora
            use_rslora = inherited_use_rslora
            if changed:
                print(
                    "[train] matched LoRA config to init adapter: "
                    f"r={lora_r} alpha={lora_alpha} dropout={lora_dropout:.4f} "
                    f"use_dora={use_dora} use_rslora={use_rslora} "
                    f"targets={','.join(target_modules)}"
                )

    lora_init_mode = _parse_lora_init_mode(lora_init)
    if runtime_backend == "dml" and bool(use_dora):
        if _patch_dora_linear_forward_for_dml():
            print("[train] applied DirectML DoRA forward compatibility patch.")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=float(lora_dropout),
        target_modules=target_modules,
        use_dora=bool(use_dora),
        use_rslora=bool(use_rslora),
        init_lora_weights=lora_init_mode,
    )
    print("[train] stage=lora_wrap start")
    model = get_peft_model(model, peft_config)
    if str(bootstrap_device) != str(device) or bootstrap_dtype != model_dtype:
        print("[train] stage=lora_wrap transfer_to_runtime start")
        model = model.to(device=device, dtype=model_dtype)
        print("[train] stage=lora_wrap transfer_to_runtime done")
    print("[train] stage=lora_wrap done")
    if init_path is not None:
        if init_path.exists():
            print(f"[train] warm-starting from adapter: {init_path}")
            try:
                model = _load_adapter_with_compat(
                    model=model,
                    adapter_dir=init_path,
                    device=device,
                    is_trainable=True,
                )
            except Exception as e:
                print(f"[train] warm-start failed; continuing with fresh LoRA init: {e}")
        else:
            print(f"[train] init adapter not found, skipping warm-start: {init_path}")
    print("[train] stage=prepare_train_mode start")
    model.train()
    noise_hook = _register_neftune_hook(model, noise_alpha=float(neftune_noise_alpha))
    print(
        "[train] NEFTune config: "
        f"sft_noise_alpha={float(neftune_noise_alpha):.3f} "
        f"preference_noise_alpha={float(preference_neftune_noise_alpha):.3f}"
    )
    print("[train] stage=prepare_train_mode done")

    print("[train] stage=filter_sft_pairs start")
    filtered_train_pairs = filter_sft_training_pairs(
        train_pairs,
        min_quality_score=float(sft_min_quality_score),
        keep_short_answer_prompts=bool(sft_filter_keep_short_answers),
        drop_synthetic_prompts=bool(sft_drop_synthetic_prompts),
        exempt_sources=sft_quality_filter_exempt_sources,
    )
    filtered_train_pairs = _select_sft_training_pairs(
        filtered_train_pairs,
        strategy=str(sft_selection_strategy),
        keep_ratio=float(sft_selection_keep_ratio),
        min_keep=int(sft_selection_min_keep),
        max_keep=int(sft_selection_max_keep),
        hardness_target=float(sft_selection_hardness_target),
        hardness_bandwidth=float(sft_selection_hardness_bandwidth),
        budget_mode=str(sft_selection_budget_mode),
        budget_power=float(sft_selection_budget_power),
        scope=str(sft_selection_scope),
        scope_min_words=int(sft_selection_scope_min_words),
    )
    print("[train] stage=filter_sft_pairs done")
    source_balance_factors: Dict[str, float] = {}
    if bool(sft_auto_balance_sources):
        source_balance_factors = _compute_source_balance_factors(
            filtered_train_pairs,
            strength=float(sft_source_balance_strength),
            max_scale=float(sft_source_balance_max_scale),
        )
        if source_balance_factors:
            src_summary = ", ".join(
                f"{k}:{v:.2f}" for k, v in sorted(source_balance_factors.items())
            )
            print(f"[sft] source balance factors: {src_summary}")

    train_rows = build_rows(
        tokenizer,
        filtered_train_pairs,
        max_length=max_length,
        row_weight_fn=lambda p: max(
            0.05,
            min(
                3.0,
                _pair_sft_weight(
                    p,
                    mode=str(sft_weight_mode),
                    min_weight=float(sft_min_weight),
                    max_weight=float(sft_max_weight),
                    synthetic_prompt_weight=float(sft_synthetic_prompt_weight),
                    teacher_source_weight=float(sft_teacher_source_weight),
                    quality_anchor_boost=float(sft_quality_anchor_boost),
                    coding_boost=float(sft_coding_boost),
                    events_boost=float(sft_events_boost),
                    reasoning_boost=float(sft_reasoning_boost),
                    prompt_skill_boost=float(sft_prompt_skill_boost),
                    conversation_boost=float(sft_conversation_boost),
                    creativity_boost=float(sft_creativity_boost),
                    knowledge_density_boost=float(sft_knowledge_density_boost),
                )
                * float(source_balance_factors.get(str(p.source or "dataset"), 1.0)),
            ),
        ),
        followup_paraphrase_aug=int(sft_followup_paraphrase_aug),
        followup_paraphrase_weight=float(sft_followup_paraphrase_weight),
    )
    sft_rows_before_packing = int(len(train_rows))
    sft_supervised_tokens_before_packing = int(
        sum(_row_supervised_token_count(row) for row in train_rows)
    )
    if bool(sft_true_packing):
        separator_token_id = getattr(tokenizer, "eos_token_id", None)
        if separator_token_id is None:
            separator_token_id = getattr(tokenizer, "pad_token_id", 0)
        train_rows = _pack_sft_rows(
            train_rows,
            max_length=max_length,
            separator_token_id=int(separator_token_id if separator_token_id is not None else 0),
            max_samples_per_row=int(sft_packing_max_samples_per_row),
        )
        packed_rows_multi = sum(
            1 for row in train_rows if int(row.get("packed_sample_count", 1)) > 1
        )
        packed_ratio = float(len(train_rows)) / float(max(1, sft_rows_before_packing))
        avg_samples = sum(int(row.get("packed_sample_count", 1)) for row in train_rows) / float(
            max(1, len(train_rows))
        )
        print(
            "[sft] true packing: "
            f"rows_before={sft_rows_before_packing} rows_after={len(train_rows)} "
            f"packed_rows={packed_rows_multi} avg_samples_per_row={avg_samples:.2f} "
            f"compression={packed_ratio:.3f}"
        )
    sft_rows_after_packing = int(len(train_rows))
    sft_supervised_tokens_after_packing = int(
        sum(_row_supervised_token_count(row) for row in train_rows)
    )
    sft_packed_rows = int(sum(1 for row in train_rows if int(row.get("packed_sample_count", 1)) > 1))
    sft_avg_samples_per_row = float(
        sum(int(row.get("packed_sample_count", 1)) for row in train_rows) / float(max(1, len(train_rows)))
    )
    sft_median_train_sequence_tokens = float(
        _median_numeric([int(row.get("sequence_tokens", len(row["input_ids"]))) for row in train_rows])
    )
    dataset = PackedChatDataset(train_rows)
    loader = _build_bucketed_dataloader(
        dataset=dataset,
        lengths=[int(row.get("sequence_tokens", len(row["input_ids"]))) for row in train_rows],
        batch_size=max(1, int(batch_size)),
        collate_fn=lambda b: collate_rows(b, tokenizer.pad_token_id),
        shuffle=True,
        enabled=bool(sft_length_bucketed_batches),
        bucket_window_multiplier=int(sft_length_bucket_window_mult),
        seed=int(seed),
        label="sft",
    )

    optim_groups, lora_plus_stats = _build_optimizer_param_groups(
        model=model,
        base_lr=float(lr),
        weight_decay=float(weight_decay),
        lora_plus_ratio=float(lora_plus_ratio),
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(optim_groups)
    if float(lora_plus_stats.get("lora_plus_ratio", 0.0)) > 0.0:
        print(
            "[train] LoRA+ optimizer groups: "
            f"ratio={lora_plus_stats['lora_plus_ratio']:.2f} "
            f"base_group={int(lora_plus_stats['lora_plus_base_group_params'])} "
            f"fast_group={int(lora_plus_stats['lora_plus_fast_group_params'])}"
        )
    sft_total_steps = max(1, int(max_steps))
    resume_sft_steps = max(0, min(int(resume_sft_steps), sft_total_steps))
    for group in optim.param_groups:
        group.setdefault("initial_lr", group["lr"])
    sft_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=_build_lr_lambda(
            schedule=str(sft_lr_schedule),
            warmup_steps=int(sft_warmup_steps),
            total_steps=sft_total_steps,
            min_lr_ratio=float(sft_min_lr_ratio),
            restart_period=int(sft_lr_restart_period),
        ),
        last_epoch=resume_sft_steps - 1,
    )
    restored_sft_training_state = False
    if resume_sft_steps > 0:
        restored_sft_training_state = _restore_checkpoint_training_state(
            adapter_dir_or_checkpoint_dir=init_path,
            stage="sft",
            optimizer=optim,
            scheduler=sft_scheduler,
            device=device,
        )
    # -- v28 improvements config --
    focal_gamma = max(0.0, float(sft_focal_gamma))
    eval_every = max(0, int(sft_eval_every_steps))
    early_stop_patience = max(0, int(sft_early_stop_patience))
    curriculum_ramp = max(0.0, float(sft_curriculum_quality_ramp))
    grad_noise_eta = max(0.0, float(sft_grad_noise_eta))
    pref_rescore_every = max(0, int(preference_rescore_every))
    if focal_gamma > 0 or eval_every > 0 or curriculum_ramp > 0 or grad_noise_eta > 0:
        print(
            "[train] v28 improvements: "
            f"focal_gamma={focal_gamma:.2f} eval_every={eval_every} "
            f"early_stop_patience={early_stop_patience} "
            f"curriculum_ramp={curriculum_ramp:.3f} "
            f"grad_noise_eta={grad_noise_eta:.4f} "
            f"lr_restart_period={int(sft_lr_restart_period)} "
            f"pref_rescore_every={pref_rescore_every}"
        )
    # Build eval set for validation monitoring
    eval_loader = None
    if eval_every > 0 and eval_pairs:
        eval_rows = build_rows(tokenizer, list(eval_pairs), max_length=max_length)
        if eval_rows:
            eval_dataset = PackedChatDataset(eval_rows)
            eval_loader = _build_bucketed_dataloader(
                dataset=eval_dataset,
                lengths=[int(row.get("sequence_tokens", len(row["input_ids"]))) for row in eval_rows],
                batch_size=max(1, int(batch_size)),
                collate_fn=lambda b: collate_rows(b, tokenizer.pad_token_id),
                shuffle=False,
                enabled=bool(sft_length_bucketed_batches),
                bucket_window_multiplier=int(sft_length_bucket_window_mult),
                seed=int(seed),
                label="sft-eval",
            )
            print(f"[train] eval monitoring: {len(eval_rows)} eval samples, check every {eval_every} steps")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(save_every_steps))
    steps = resume_sft_steps
    token_count = 0
    sft_weight_accum = 0.0
    sft_weight_count = 0
    sft_loss_sum = float(resume_sft_loss_mean) * float(steps)
    sft_rdrop_sum = 0.0
    best_eval_loss = float('inf')
    eval_no_improve_count = 0
    sft_early_stopped = False
    t0 = time.time()
    optim.zero_grad(set_to_none=True)
    if steps > 0:
        if restored_sft_training_state:
            print(
                "[resume] continuing SFT from adapter "
                f"{init_path} at step={steps} (optimizer/scheduler state restored)"
            )
        else:
            print(
                "[resume] continuing SFT from adapter "
                f"{init_path} at step={steps} (optimizer state is reinitialized)"
            )

    if bool(skip_sft):
        print("[train] skipping SFT stage (--skip_sft).")
    else:
        for epoch in range(max(1, int(epochs))):
            for i, batch in enumerate(loader):
                model_batch = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "labels": batch["labels"].to(device),
                }
                segment_count = batch.get("segment_count")
                segment_lengths = batch.get("segment_lengths")
                segment_weights = batch.get("segment_weights")
                if segment_count is not None:
                    segment_count = segment_count.to(device)
                if segment_lengths is not None:
                    segment_lengths = segment_lengths.to(device)
                sample_weights = None
                if segment_weights is not None and segment_count is not None:
                    sample_weights = _flatten_packed_segment_values(
                        segment_weights.to(device).float(),
                        segment_count,
                    )
                else:
                    sample_weights = batch.get("weights")
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device).float()
                    # Curriculum quality ramp: linearly reduce weight of lower-quality
                    # samples over the first 50% of training by modulating weights.
                    if curriculum_ramp > 0.0 and steps < sft_total_steps // 2:
                        # Ramp progress goes 0 -> 1 over first half of training.
                        ramp_progress = float(steps) / float(max(1, sft_total_steps // 2))
                        # Scale: weights below median get reduced early, restored later.
                        ramp_scale = 1.0 - curriculum_ramp * (1.0 - ramp_progress)
                        one_weights = torch.ones_like(sample_weights)
                        ramp_scale_t = sample_weights.new_tensor(ramp_scale)
                        weight_median = sample_weights.median()
                        below_median = (sample_weights < weight_median).float()
                        sample_weights = sample_weights * (
                            one_weights - below_median * (one_weights - ramp_scale_t)
                        )
                    sft_weight_accum += float(sample_weights.sum().item())
                    sft_weight_count += int(sample_weights.numel())

                out = model(
                    input_ids=model_batch["input_ids"],
                    attention_mask=model_batch["attention_mask"],
                )
                if segment_lengths is not None and segment_count is not None:
                    seq_logp, _ = _packed_segment_average_log_prob(
                        out.logits,
                        model_batch["labels"],
                        segment_lengths=segment_lengths,
                        segment_count=segment_count,
                    )
                else:
                    seq_logp, _ = _sequence_average_log_prob(out.logits, model_batch["labels"])
                seq_loss = -seq_logp

                # Focal loss: down-weight easy samples, up-weight hard ones.
                if focal_gamma > 0.0:
                    focal_weight = _tensor_one_minus(torch.exp(-seq_loss.detach())).pow(focal_gamma)
                    if sample_weights is not None:
                        sample_weights = sample_weights * focal_weight
                    else:
                        sample_weights = focal_weight

                loss_value = _weighted_mean(seq_loss, sample_weights)

                rdrop_value = seq_loss.new_zeros(())
                if float(sft_rdrop_alpha) > 0.0:
                    out_b = model(
                        input_ids=model_batch["input_ids"],
                        attention_mask=model_batch["attention_mask"],
                    )
                    if segment_lengths is not None and segment_count is not None:
                        seq_logp_b, _ = _packed_segment_average_log_prob(
                            out_b.logits,
                            model_batch["labels"],
                            segment_lengths=segment_lengths,
                            segment_count=segment_count,
                        )
                    else:
                        seq_logp_b, _ = _sequence_average_log_prob(out_b.logits, model_batch["labels"])
                    seq_loss_b = -seq_logp_b
                    base_loss = 0.5 * (
                        _weighted_mean(seq_loss, sample_weights)
                        + _weighted_mean(seq_loss_b, sample_weights)
                    )
                    if segment_lengths is not None and segment_count is not None:
                        rdrop_term = _packed_segment_symmetric_token_kl(
                            out.logits,
                            out_b.logits,
                            model_batch["labels"],
                            segment_lengths=segment_lengths,
                            segment_count=segment_count,
                        )
                    else:
                        rdrop_term = _symmetric_token_kl(out.logits, out_b.logits, model_batch["labels"])
                    rdrop_value = _weighted_mean(rdrop_term, sample_weights)
                    loss_value = base_loss + float(sft_rdrop_alpha) * rdrop_value
                    sft_rdrop_sum += float(rdrop_value.item())
                loss = loss_value / max(1, int(grad_accum_steps))
                loss.backward()

                non_mask = int((model_batch["labels"] != -100).sum().item())
                token_count += non_mask
                sft_loss_sum += float(loss_value.item())

                if (i + 1) % max(1, int(grad_accum_steps)) == 0:
                    if float(sft_max_grad_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, float(sft_max_grad_norm))
                    # Gradient noise injection for escaping sharp minima.
                    if grad_noise_eta > 0.0:
                        noise_std = grad_noise_eta / (1.0 + float(steps)) ** 0.55
                        for p in trainable_params:
                            if p.grad is not None:
                                p.grad.add_(torch.randn_like(p.grad) * noise_std)
                    optim.step()
                    sft_scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    steps += 1
                    if steps % max(1, int(train_log_every_steps)) == 0:
                        current_lr = float(optim.param_groups[0]["lr"])
                        print(
                            f"[train] step={steps} loss={sft_loss_sum / max(1, steps):.4f} "
                            f"lr={current_lr:.6g} rdrop={sft_rdrop_sum / max(1, steps):.4f}"
                        )
                    if checkpoint_every > 0 and steps % checkpoint_every == 0:
                        ckpt_adapter_dir = output_dir / "checkpoints" / f"sft_step_{steps:05d}" / "adapter"
                        ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_adapter_dir)
                        tokenizer.save_pretrained(ckpt_adapter_dir)
                        trainer_state_path = _save_checkpoint_training_state(
                            adapter_dir_or_checkpoint_dir=ckpt_adapter_dir,
                            stage="sft",
                            optimizer=optim,
                            scheduler=sft_scheduler,
                        )
                        checkpoint_meta = {
                            "stage": "sft",
                            "sft_steps": int(steps),
                            "sft_loss_mean": float(sft_loss_sum / max(1, steps)),
                            "checkpoint_adapter_dir": str(ckpt_adapter_dir),
                            "checkpoint_training_state_path": str(trainer_state_path),
                        }
                        (ckpt_adapter_dir.parent / "checkpoint_meta.json").write_text(
                            json.dumps(checkpoint_meta, indent=2),
                            encoding="utf-8",
                        )
                        (output_dir / "latest_adapter_checkpoint.txt").write_text(
                            str(ckpt_adapter_dir),
                            encoding="utf-8",
                        )
                        print(f"[checkpoint] saved stage=sft step={steps} -> {ckpt_adapter_dir}")
                        _prune_checkpoint_dirs(output_dir, keep_last_checkpoints)
                    # Eval monitoring with optional early stopping.
                    if eval_every > 0 and eval_loader is not None and steps % eval_every == 0:
                        model.eval()
                        eval_loss_sum = 0.0
                        eval_count = 0
                        with torch.no_grad():
                            for eval_batch in eval_loader:
                                e_ids = eval_batch["input_ids"].to(device)
                                e_mask = eval_batch["attention_mask"].to(device)
                                e_labels = eval_batch["labels"].to(device)
                                e_out = model(input_ids=e_ids, attention_mask=e_mask)
                                e_logp, _ = _sequence_average_log_prob(e_out.logits, e_labels)
                                eval_loss_sum += float((-e_logp).mean().item())
                                eval_count += 1
                        model.train()
                        eval_loss = eval_loss_sum / max(1, eval_count)
                        improved = eval_loss < best_eval_loss
                        if improved:
                            best_eval_loss = eval_loss
                            eval_no_improve_count = 0
                        else:
                            eval_no_improve_count += 1
                        print(
                            f"[eval] step={steps} eval_loss={eval_loss:.4f} "
                            f"best={best_eval_loss:.4f} no_improve={eval_no_improve_count}"
                        )
                        if early_stop_patience > 0 and eval_no_improve_count >= early_stop_patience:
                            print(
                                f"[eval] early stopping at step={steps}: "
                                f"no improvement for {eval_no_improve_count} evaluations"
                            )
                            sft_early_stopped = True
                            break
                    if steps >= max(1, int(max_steps)):
                        break
            if sft_early_stopped or steps >= max(1, int(max_steps)):
                break
        if checkpoint_every > 0 and steps > 0 and steps % checkpoint_every != 0:
            ckpt_adapter_dir = output_dir / "checkpoints" / f"sft_step_{steps:05d}" / "adapter"
            ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_adapter_dir)
            tokenizer.save_pretrained(ckpt_adapter_dir)
            trainer_state_path = _save_checkpoint_training_state(
                adapter_dir_or_checkpoint_dir=ckpt_adapter_dir,
                stage="sft",
                optimizer=optim,
                scheduler=sft_scheduler,
            )
            checkpoint_meta = {
                "stage": "sft",
                "sft_steps": int(steps),
                "sft_loss_mean": float(sft_loss_sum / max(1, steps)),
                "checkpoint_adapter_dir": str(ckpt_adapter_dir),
                "checkpoint_training_state_path": str(trainer_state_path),
            }
            (ckpt_adapter_dir.parent / "checkpoint_meta.json").write_text(
                json.dumps(checkpoint_meta, indent=2),
                encoding="utf-8",
            )
            (output_dir / "latest_adapter_checkpoint.txt").write_text(
                str(ckpt_adapter_dir),
                encoding="utf-8",
            )
            print(f"[checkpoint] saved stage=sft step={steps} -> {ckpt_adapter_dir}")
            _prune_checkpoint_dirs(output_dir, keep_last_checkpoints)

    noise_hook = _replace_neftune_hook(
        model=model,
        hook=noise_hook,
        noise_alpha=float(preference_neftune_noise_alpha),
    )

    pref_steps_done = max(0, int(resume_preference_steps))
    pref_loss_sum = float(resume_preference_loss_mean) * float(pref_steps_done)
    pref_pair_count = 0
    pref_weight_mean = 0.0
    pref_reference_pairs = 0
    pref_wpo_std_sum = 0.0
    pref_robust_std_sum = 0.0
    pref_length_control_sum = 0.0
    pref_objective_mode = str(preference_objective).strip().lower()
    pref_beta_start = float(preference_beta)
    pref_beta_final = float(preference_beta)
    if float(preference_beta_end) > 0:
        pref_beta_final = float(preference_beta_end)
    pref_margin_start = float(preference_margin)
    pref_margin_final = float(preference_margin)
    if float(preference_margin_end) >= 0.0:
        pref_margin_final = float(preference_margin_end)
    pref_hardness_gamma = max(0.0, float(preference_hardness_gamma))
    pref_label_smoothing = max(0.0, min(0.49, float(preference_label_smoothing)))
    pref_xpo_clip = max(0.0, float(preference_xpo_clip))
    pref_robust_alpha = max(0.0, min(1.0, float(preference_robust_alpha)))
    pref_robust_eta = max(0.0, min(0.49, float(preference_robust_eta)))
    pref_robust_clip = max(1.0, float(preference_robust_clip))
    pref_length_control_strength = max(0.0, float(preference_length_control_strength))
    pref_length_control_target_ratio = max(0.75, float(preference_length_control_target_ratio))
    pref_length_control_max_penalty = max(0.0, float(preference_length_control_max_penalty))
    pref_self_play_budget = max(0, int(preference_self_play_budget))
    pref_self_play_curriculum = str(preference_self_play_curriculum or "easy_to_hard").strip().lower()
    if pref_self_play_curriculum not in {"easy_to_hard", "hard_first"}:
        pref_self_play_curriculum = "easy_to_hard"
    pref_self_play_max_new_tokens = max(0, int(preference_self_play_max_new_tokens))
    pref_stop_signal_strength = max(0.0, float(preference_stop_signal_strength))
    pref_stop_rejects_per_prompt = max(0, int(preference_stop_rejects_per_prompt))
    pref_anchor_weight = max(0.0, float(preference_reference_anchor_weight))
    pref_ipo_target_start = (
        _ipo_target_gap(beta=pref_beta_start, margin=pref_margin_start)
        if pref_objective_mode == "ipo"
        else 0.0
    )
    pref_ipo_target_final = (
        _ipo_target_gap(beta=pref_beta_final, margin=pref_margin_final)
        if pref_objective_mode == "ipo"
        else 0.0
    )
    if pref_objective_mode != "none" and int(preference_steps) > 0 and int(preference_pairs) > 0:
        print(f"[pref] building preference pairs (mode={pref_objective_mode})...")
        ipo_schedule = (
            f"ipo_target={pref_ipo_target_start:.4f}->{pref_ipo_target_final:.4f} "
            if pref_objective_mode == "ipo"
            else ""
        )
        print(
            "[pref] objective schedule: "
            f"beta={pref_beta_start:.4f}->{pref_beta_final:.4f} "
            f"margin={pref_margin_start:.4f}->{pref_margin_final:.4f} "
            f"{ipo_schedule}"
            f"label_smoothing={pref_label_smoothing:.3f} "
            f"xpo_clip={pref_xpo_clip:.3f} "
            f"hardness_gamma={pref_hardness_gamma:.3f} "
            f"wpo_alpha={float(preference_wpo_alpha):.3f} "
            f"robust_alpha={pref_robust_alpha:.3f} "
            f"robust_eta={pref_robust_eta:.3f} "
            f"len_ctrl={pref_length_control_strength:.3f} "
            f"len_ratio={pref_length_control_target_ratio:.3f} "
            f"self_play_budget={pref_self_play_budget} "
            f"self_play_curriculum={pref_self_play_curriculum} "
            f"self_play_max_new_tokens={pref_self_play_max_new_tokens if pref_self_play_max_new_tokens > 0 else preference_max_new_tokens} "
            f"stop_signal={pref_stop_signal_strength:.3f} "
            f"stop_rejects={pref_stop_rejects_per_prompt} "
            f"anchor_weight={pref_anchor_weight:.4f}"
        )
        pref_pairs = build_preference_pairs_with_generation(
            model=model,
            tokenizer=tokenizer,
            train_pairs=filtered_train_pairs,
            max_pairs=int(preference_pairs),
            max_new_tokens=int(preference_max_new_tokens),
            prompt_max_tokens=int(preference_prompt_max_tokens),
            seed=int(seed) + 17,
            reject_similarity_min=float(preference_reject_similarity_min),
            candidate_count=int(preference_candidate_count),
            include_greedy_candidate=bool(preference_include_greedy_candidate),
            short_reject_boost=float(preference_short_reject_boost),
            long_reject_boost=float(preference_long_reject_boost),
            min_chosen_quality=float(preference_min_chosen_quality),
            min_chosen_words=int(preference_min_chosen_words),
            min_quality_gap=float(preference_min_quality_gap),
            skip_template_prompts=bool(preference_skip_template_prompts),
            max_pairs_per_user=int(preference_max_pairs_per_user),
            max_pairs_per_source=int(preference_max_pairs_per_source),
            fallback_same_source_first=bool(preference_fallback_same_source_first),
            mining_mode=str(preference_mining_mode),
            progress_every=int(preference_mining_progress_every),
            mining_max_seconds=float(preference_mining_max_seconds),
            mining_max_attempt_factor=int(preference_mining_max_attempt_factor),
            coding_focus_boost=float(preference_coding_focus_boost),
            reasoning_focus_boost=float(preference_reasoning_focus_boost),
            counterfactual_rejects_per_prompt=int(preference_counterfactual_rejects_per_prompt),
            self_play_budget=int(pref_self_play_budget),
            self_play_curriculum=str(pref_self_play_curriculum),
            self_play_max_new_tokens=int(pref_self_play_max_new_tokens),
            stop_signal_strength=float(pref_stop_signal_strength),
            stop_rejects_per_prompt=int(pref_stop_rejects_per_prompt),
            selection_strategy=str(preference_selection_strategy),
            selection_keep_ratio=float(preference_selection_keep_ratio),
            selection_min_keep=int(preference_selection_min_keep),
            selection_max_keep=int(preference_selection_max_keep),
            selection_hardness_target=float(preference_selection_hardness_target),
            selection_hardness_bandwidth=float(preference_selection_hardness_bandwidth),
        )
        pref_pair_count = len(pref_pairs)
        pref_rescore_rounds = 0
        if pref_pairs:
            pref_weight_mean = float(sum(float(p.weight) for p in pref_pairs) / max(1, len(pref_pairs)))
        print(f"[pref] pairs={pref_pair_count}")
        pref_rows = build_preference_rows(tokenizer, pref_pairs, max_length=max_length)
        if pref_rows:
            if pref_rescore_every > 0:
                pref_rescore_rounds += 1
                pref_weight_summary = _rescore_preference_rows(
                    pref_rows=pref_rows,
                    step=pref_steps_done,
                    total_steps=max(1, int(preference_steps)),
                    round_index=pref_rescore_rounds,
                )
                pref_weight_mean = float(pref_weight_summary.get("weight_mean", pref_weight_mean))
            pref_needs_reference = pref_anchor_weight > 0.0 or pref_objective_mode in {"dpo", "ipo", "xpo"}
            if pref_needs_reference:
                print("[pref] caching reference log-probs for reference-aware preference optimization...")
                pref_reference_pairs = annotate_preference_rows_with_reference_logps(
                    model=model,
                    pref_rows=pref_rows,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                    batch_size=int(preference_reference_anchor_batch_size),
                    length_bucketed_batches=bool(preference_length_bucketed_batches),
                    length_bucket_window_mult=int(preference_length_bucket_window_mult),
                    seed=int(seed),
                )
                print(f"[pref] reference margins cached for {pref_reference_pairs} pairs")
            pref_dataset = PackedPreferenceDataset(pref_rows)
            pref_loader = _build_bucketed_dataloader(
                dataset=pref_dataset,
                lengths=[int(row.get("pair_tokens", 1)) for row in pref_rows],
                batch_size=max(1, int(batch_size)),
                collate_fn=lambda b: collate_preference_rows(b, tokenizer.pad_token_id),
                shuffle=True,
                enabled=bool(preference_length_bucketed_batches),
                bucket_window_multiplier=int(preference_length_bucket_window_mult),
                seed=int(seed),
                label="pref",
            )

            pref_optim = torch.optim.AdamW(
                _build_optimizer_param_groups(
                    model=model,
                    base_lr=float(preference_lr),
                    weight_decay=float(weight_decay),
                    lora_plus_ratio=float(lora_plus_ratio),
                )[0],
            )
            for group in pref_optim.param_groups:
                group.setdefault("initial_lr", group["lr"])
            pref_scheduler = torch.optim.lr_scheduler.LambdaLR(
                pref_optim,
                lr_lambda=_build_lr_lambda(
                    schedule=str(preference_lr_schedule),
                    warmup_steps=int(preference_warmup_steps),
                    total_steps=max(1, int(preference_steps)),
                    min_lr_ratio=float(preference_min_lr_ratio),
                ),
                last_epoch=max(-1, pref_steps_done - 1),
            )
            restored_pref_training_state = False
            if pref_steps_done > 0:
                restored_pref_training_state = _restore_checkpoint_training_state(
                    adapter_dir_or_checkpoint_dir=init_path,
                    stage="preference",
                    optimizer=pref_optim,
                    scheduler=pref_scheduler,
                    device=device,
                )
            pref_optim.zero_grad(set_to_none=True)
            model.train()
            if pref_steps_done > 0:
                if restored_pref_training_state:
                    print(
                        "[resume] continuing preference stage from adapter "
                        f"{init_path} at step={pref_steps_done} (optimizer/scheduler state restored)"
                    )
                else:
                    print(
                        "[resume] continuing preference stage from adapter "
                        f"{init_path} at step={pref_steps_done} (optimizer state is reinitialized)"
                    )

            target_steps = max(1, int(preference_steps))
            pref_accum = 0
            while pref_steps_done < target_steps:
                for batch in pref_loader:
                    chosen_input_ids = batch["chosen_input_ids"].to(device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                    chosen_labels = batch["chosen_labels"].to(device)
                    rejected_input_ids = batch["rejected_input_ids"].to(device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                    rejected_labels = batch["rejected_labels"].to(device)
                    sample_weights = batch["weights"].to(device).float()
                    sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)

                    out_chosen = model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
                    out_rejected = model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)

                    chosen_logp, chosen_len = _sequence_average_log_prob(out_chosen.logits, chosen_labels)
                    rejected_logp, rejected_len = _sequence_average_log_prob(out_rejected.logits, rejected_labels)
                    delta = chosen_logp - rejected_logp
                    beta_t = _linear_schedule_value(
                        pref_beta_start,
                        pref_beta_final,
                        pref_steps_done,
                        target_steps,
                    )
                    margin_t = _linear_schedule_value(
                        pref_margin_start,
                        pref_margin_final,
                        pref_steps_done,
                        target_steps,
                    )
                    length_control_margin = torch.zeros_like(delta)
                    if pref_length_control_strength > 0.0:
                        length_control_margin = (
                            _preference_length_control_margin(
                                chosen_len=chosen_len,
                                rejected_len=rejected_len,
                                target_ratio=pref_length_control_target_ratio,
                                max_penalty=pref_length_control_max_penalty,
                            )
                            * pref_length_control_strength
                        )
                        pref_length_control_sum += float(length_control_margin.mean().item())
                    effective_margin_t = margin_t + length_control_margin
                    robust_logits = beta_t * (delta - effective_margin_t)

                    if pref_objective_mode == "repo":
                        pref_core = torch.relu(effective_margin_t - delta)
                    elif pref_objective_mode == "dpo":
                        ref_chosen = batch.get("ref_chosen_logp")
                        ref_rejected = batch.get("ref_rejected_logp")
                        if ref_chosen is None or ref_rejected is None:
                            raise RuntimeError("DPO objective requires cached reference log-probabilities.")
                        ref_delta = ref_chosen.to(device).float() - ref_rejected.to(device).float()
                        robust_logits = beta_t * ((delta - ref_delta) - effective_margin_t)
                        pref_core = _dpo_preference_loss(
                            delta=delta,
                            ref_delta=ref_delta,
                            beta=beta_t,
                            margin=effective_margin_t,
                            label_smoothing=pref_label_smoothing,
                        )
                    elif pref_objective_mode == "ipo":
                        ref_chosen = batch.get("ref_chosen_logp")
                        ref_rejected = batch.get("ref_rejected_logp")
                        if ref_chosen is None or ref_rejected is None:
                            raise RuntimeError("IPO objective requires cached reference log-probabilities.")
                        ref_delta = ref_chosen.to(device).float() - ref_rejected.to(device).float()
                        robust_logits = beta_t * ((delta - ref_delta) - effective_margin_t)
                        pref_core = _ipo_preference_loss(
                            delta=delta,
                            ref_delta=ref_delta,
                            beta=beta_t,
                            margin=effective_margin_t,
                        )
                    elif pref_objective_mode == "xpo":
                        ref_chosen = batch.get("ref_chosen_logp")
                        ref_rejected = batch.get("ref_rejected_logp")
                        if ref_chosen is None or ref_rejected is None:
                            raise RuntimeError("χPO objective requires cached reference log-probabilities.")
                        robust_logits = _xpo_preference_logits_delta(
                            chosen_logp=chosen_logp,
                            rejected_logp=rejected_logp,
                            ref_chosen_logp=ref_chosen.to(device).float(),
                            ref_rejected_logp=ref_rejected.to(device).float(),
                            beta=beta_t,
                            clip=pref_xpo_clip,
                        ) - (beta_t * length_control_margin)
                        pref_core = _sigmoid_preference_loss(robust_logits, label_smoothing=pref_label_smoothing)
                    elif pref_objective_mode == "orpo":
                        chosen_odds = _log_odds_from_avg_log_prob(chosen_logp)
                        rejected_odds = _log_odds_from_avg_log_prob(rejected_logp)
                        odds_delta = chosen_odds - rejected_odds
                        robust_logits = beta_t * (odds_delta - effective_margin_t)
                        pref_core = torch.nn.functional.softplus(-beta_t * (odds_delta - effective_margin_t))
                    else:
                        z = beta_t * (delta - effective_margin_t)
                        robust_logits = z
                        pref_core = _sigmoid_preference_loss(z, label_smoothing=pref_label_smoothing)

                    if pref_hardness_gamma > 0.0:
                        hardness_weight = _tensor_one_minus(
                            torch.sigmoid((beta_t * delta).detach())
                        ).pow(pref_hardness_gamma)
                        sample_weights = sample_weights * (1.0 + hardness_weight)
                        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)

                    if float(preference_wpo_alpha) > 0.0:
                        wpo_weights = _wpo_pair_weights(
                            chosen_logp=chosen_logp,
                            rejected_logp=rejected_logp,
                            alpha=float(preference_wpo_alpha),
                            clip=float(preference_wpo_clip),
                        )
                        sample_weights = sample_weights * wpo_weights
                        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
                        pref_wpo_std_sum += float(wpo_weights.std(unbiased=False).item())

                    if pref_robust_alpha > 0.0 and pref_robust_eta > 0.0:
                        robust_weights = _robust_preference_correctness_weights(
                            logits_delta=robust_logits,
                            alpha=pref_robust_alpha,
                            noise_eta=pref_robust_eta,
                            clip=pref_robust_clip,
                        )
                        sample_weights = sample_weights * robust_weights
                        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
                        pref_robust_std_sum += float(robust_weights.std(unbiased=False).item())

                    anchor_term = torch.zeros_like(pref_core)
                    if pref_anchor_weight > 0.0:
                        ref_chosen = batch.get("ref_chosen_logp")
                        ref_rejected = batch.get("ref_rejected_logp")
                        if ref_chosen is not None and ref_rejected is not None:
                            ref_delta = ref_chosen.to(device).float() - ref_rejected.to(device).float()
                            anchor_term = (delta - ref_delta).pow(2)

                    sft_term = -chosen_logp
                    length_gap = torch.abs(chosen_len.float() - rejected_len.float()) / (chosen_len.float() + 1.0)
                    pref_obj = (
                        pref_core
                        + float(preference_sft_weight) * sft_term
                        + float(preference_length_weight) * length_gap
                        + pref_anchor_weight * anchor_term
                    )
                    pref_loss = (pref_obj * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
                    (pref_loss / max(1, int(grad_accum_steps))).backward()
                    pref_loss_sum += float(pref_loss.item())
                    pref_accum += 1

                    pref_steps_done += 1
                    if pref_rescore_every > 0 and pref_steps_done < target_steps and pref_steps_done % pref_rescore_every == 0:
                        pref_rescore_rounds += 1
                        pref_weight_summary = _rescore_preference_rows(
                            pref_rows=pref_rows,
                            step=pref_steps_done,
                            total_steps=target_steps,
                            round_index=pref_rescore_rounds,
                        )
                        pref_weight_mean = float(pref_weight_summary.get("weight_mean", pref_weight_mean))
                    should_step = (pref_accum % max(1, int(grad_accum_steps))) == 0 or pref_steps_done >= target_steps
                    if should_step:
                        if float(preference_max_grad_norm) > 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, float(preference_max_grad_norm))
                        pref_optim.step()
                        pref_scheduler.step()
                        pref_optim.zero_grad(set_to_none=True)

                    if pref_steps_done % max(1, int(train_log_every_steps)) == 0:
                        pref_lr_now = float(pref_optim.param_groups[0]["lr"])
                        beta_now = _linear_schedule_value(
                            pref_beta_start,
                            pref_beta_final,
                            pref_steps_done,
                            target_steps,
                        )
                        margin_now = _linear_schedule_value(
                            pref_margin_start,
                            pref_margin_final,
                            pref_steps_done,
                            target_steps,
                        )
                        print(
                            f"[pref] step={pref_steps_done} loss={pref_loss_sum / max(1, pref_steps_done):.4f} "
                            f"lr={pref_lr_now:.6g} beta={beta_now:.4f} margin={margin_now:.4f} "
                            f"wpo_std={pref_wpo_std_sum / max(1, pref_steps_done):.4f}"
                        )
                    if checkpoint_every > 0 and pref_steps_done % checkpoint_every == 0:
                        ckpt_adapter_dir = output_dir / "checkpoints" / f"pref_step_{pref_steps_done:05d}" / "adapter"
                        ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_adapter_dir)
                        tokenizer.save_pretrained(ckpt_adapter_dir)
                        trainer_state_path = _save_checkpoint_training_state(
                            adapter_dir_or_checkpoint_dir=ckpt_adapter_dir,
                            stage="preference",
                            optimizer=pref_optim,
                            scheduler=pref_scheduler,
                        )
                        checkpoint_meta = {
                            "stage": "preference",
                            "sft_steps": int(steps),
                            "preference_steps": int(pref_steps_done),
                            "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
                            "checkpoint_adapter_dir": str(ckpt_adapter_dir),
                            "checkpoint_training_state_path": str(trainer_state_path),
                        }
                        (ckpt_adapter_dir.parent / "checkpoint_meta.json").write_text(
                            json.dumps(checkpoint_meta, indent=2),
                            encoding="utf-8",
                        )
                        (output_dir / "latest_adapter_checkpoint.txt").write_text(
                            str(ckpt_adapter_dir),
                            encoding="utf-8",
                        )
                        print(f"[checkpoint] saved stage=preference step={pref_steps_done} -> {ckpt_adapter_dir}")
                        _prune_checkpoint_dirs(output_dir, keep_last_checkpoints)
                    if pref_steps_done >= target_steps:
                        break
                if pref_steps_done >= target_steps:
                    break
            if checkpoint_every > 0 and pref_steps_done > 0 and pref_steps_done % checkpoint_every != 0:
                ckpt_adapter_dir = output_dir / "checkpoints" / f"pref_step_{pref_steps_done:05d}" / "adapter"
                ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_adapter_dir)
                tokenizer.save_pretrained(ckpt_adapter_dir)
                trainer_state_path = _save_checkpoint_training_state(
                    adapter_dir_or_checkpoint_dir=ckpt_adapter_dir,
                    stage="preference",
                    optimizer=pref_optim,
                    scheduler=pref_scheduler,
                )
                checkpoint_meta = {
                    "stage": "preference",
                    "sft_steps": int(steps),
                    "preference_steps": int(pref_steps_done),
                    "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
                    "checkpoint_adapter_dir": str(ckpt_adapter_dir),
                    "checkpoint_training_state_path": str(trainer_state_path),
                }
                (ckpt_adapter_dir.parent / "checkpoint_meta.json").write_text(
                    json.dumps(checkpoint_meta, indent=2),
                    encoding="utf-8",
                )
                (output_dir / "latest_adapter_checkpoint.txt").write_text(
                    str(ckpt_adapter_dir),
                    encoding="utf-8",
                )
                print(f"[checkpoint] saved stage=preference step={pref_steps_done} -> {ckpt_adapter_dir}")
                _prune_checkpoint_dirs(output_dir, keep_last_checkpoints)

    if noise_hook is not None:
        noise_hook.remove()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    elapsed = max(1e-6, time.time() - t0)
    stats = {
        "resume_sft_steps": float(max(0, int(resume_sft_steps))),
        "resume_preference_steps": float(max(0, int(resume_preference_steps))),
        "sft_steps": float(steps),
        "sft_loss_mean": float(sft_loss_sum / max(1, steps)),
        "sft_lr_schedule": str(sft_lr_schedule).strip().lower(),
        "sft_warmup_steps": float(max(0, int(sft_warmup_steps))),
        "sft_min_lr_ratio": float(sft_min_lr_ratio),
        "sft_max_grad_norm": float(sft_max_grad_norm),
        "sft_weight_mode": str(sft_weight_mode).strip().lower(),
        "sft_weight_mean": float(sft_weight_accum / max(1, sft_weight_count)),
        "sft_pairs_used": float(len(filtered_train_pairs)),
        "sft_auto_balance_sources": bool(sft_auto_balance_sources),
        "sft_source_balance_strength": float(sft_source_balance_strength),
        "sft_source_balance_max_scale": float(sft_source_balance_max_scale),
        "sft_reasoning_boost": float(sft_reasoning_boost),
        "sft_prompt_skill_boost": float(sft_prompt_skill_boost),
        "sft_conversation_boost": float(sft_conversation_boost),
        "sft_creativity_boost": float(sft_creativity_boost),
        "sft_knowledge_density_boost": float(sft_knowledge_density_boost),
        "sft_followup_paraphrase_aug": float(max(0, int(sft_followup_paraphrase_aug))),
        "sft_followup_paraphrase_weight": float(sft_followup_paraphrase_weight),
        "sft_rdrop_alpha": float(max(0.0, sft_rdrop_alpha)),
        "sft_rdrop_mean": float(sft_rdrop_sum / max(1, steps)),
        "sft_focal_gamma": float(focal_gamma),
        "sft_eval_every_steps": float(eval_every),
        "sft_early_stop_patience": float(early_stop_patience),
        "sft_early_stopped": bool(sft_early_stopped),
        "sft_best_eval_loss": float(best_eval_loss) if best_eval_loss < float('inf') else 0.0,
        "sft_curriculum_quality_ramp": float(curriculum_ramp),
        "sft_grad_noise_eta": float(grad_noise_eta),
        "sft_lr_restart_period": float(int(sft_lr_restart_period)),
        "sft_length_bucketed_batches": bool(sft_length_bucketed_batches),
        "sft_length_bucket_window_mult": float(max(2, int(sft_length_bucket_window_mult))),
        "sft_true_packing": bool(sft_true_packing),
        "sft_packing_max_samples_per_row": float(max(0, int(sft_packing_max_samples_per_row))),
        "sft_rows_before_packing": float(sft_rows_before_packing),
        "sft_rows_after_packing": float(sft_rows_after_packing),
        "sft_packed_rows": float(sft_packed_rows),
        "sft_avg_samples_per_row": float(sft_avg_samples_per_row),
        "sft_supervised_tokens_before_packing": float(sft_supervised_tokens_before_packing),
        "sft_supervised_tokens_after_packing": float(sft_supervised_tokens_after_packing),
        "sft_median_train_sequence_tokens": float(sft_median_train_sequence_tokens),
        "sft_selection_strategy": str(sft_selection_strategy).strip().lower(),
        "sft_selection_keep_ratio": float(max(0.0, min(1.0, sft_selection_keep_ratio))),
        "sft_selection_min_keep": float(max(0, int(sft_selection_min_keep))),
        "sft_selection_max_keep": float(max(0, int(sft_selection_max_keep))),
        "sft_selection_hardness_target": float(sft_selection_hardness_target),
        "sft_selection_hardness_bandwidth": float(sft_selection_hardness_bandwidth),
        "sft_selection_budget_mode": str(sft_selection_budget_mode).strip().lower(),
        "sft_selection_budget_power": float(max(0.0, min(1.0, sft_selection_budget_power))),
        "sft_selection_scope": str(sft_selection_scope).strip().lower(),
        "sft_selection_scope_min_words": float(max(1, int(sft_selection_scope_min_words))),
        "preference_rescore_every": float(pref_rescore_every),
        "preference_steps": float(pref_steps_done),
        "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
        "preference_pairs": float(pref_pair_count),
        "preference_weight_mean": float(pref_weight_mean),
        "preference_objective": pref_objective_mode,
        "preference_beta_start": float(pref_beta_start),
        "preference_beta_end": float(pref_beta_final),
        "preference_margin_start": float(pref_margin_start),
        "preference_margin_end": float(pref_margin_final),
        "preference_ipo_target_start": float(pref_ipo_target_start),
        "preference_ipo_target_end": float(pref_ipo_target_final),
        "preference_label_smoothing": float(pref_label_smoothing),
        "preference_xpo_clip": float(pref_xpo_clip),
        "preference_hardness_gamma": float(pref_hardness_gamma),
        "preference_length_control_strength": float(pref_length_control_strength),
        "preference_length_control_target_ratio": float(pref_length_control_target_ratio),
        "preference_length_control_max_penalty": float(pref_length_control_max_penalty),
        "preference_length_control_mean": float(pref_length_control_sum / max(1, pref_steps_done)),
        "preference_self_play_budget": float(pref_self_play_budget),
        "preference_self_play_curriculum": str(pref_self_play_curriculum),
        "preference_self_play_max_new_tokens": float(pref_self_play_max_new_tokens),
        "preference_stop_signal_strength": float(pref_stop_signal_strength),
        "preference_stop_rejects_per_prompt": float(pref_stop_rejects_per_prompt),
        "preference_robust_alpha": float(pref_robust_alpha),
        "preference_robust_eta": float(pref_robust_eta),
        "preference_robust_clip": float(pref_robust_clip),
        "preference_robust_std_mean": float(pref_robust_std_sum / max(1, pref_steps_done)),
        "preference_wpo_alpha": float(max(0.0, preference_wpo_alpha)),
        "preference_wpo_clip": float(max(1.0, preference_wpo_clip)),
        "preference_wpo_std_mean": float(pref_wpo_std_sum / max(1, pref_steps_done)),
        "preference_reference_anchor_weight": float(pref_anchor_weight),
        "preference_reference_pairs": float(pref_reference_pairs),
        "preference_mining_mode": str(preference_mining_mode).strip().lower(),
        "preference_mining_max_seconds": float(max(0.0, preference_mining_max_seconds)),
        "preference_mining_max_attempt_factor": float(max(0, preference_mining_max_attempt_factor)),
        "preference_coding_focus_boost": float(preference_coding_focus_boost),
        "preference_reasoning_focus_boost": float(preference_reasoning_focus_boost),
        "preference_counterfactual_rejects_per_prompt": float(
            max(0, int(preference_counterfactual_rejects_per_prompt))
        ),
        "preference_selection_strategy": str(preference_selection_strategy).strip().lower(),
        "preference_selection_keep_ratio": float(max(0.0, min(1.0, preference_selection_keep_ratio))),
        "preference_selection_min_keep": float(max(0, int(preference_selection_min_keep))),
        "preference_selection_max_keep": float(max(0, int(preference_selection_max_keep))),
        "preference_selection_hardness_target": float(preference_selection_hardness_target),
        "preference_selection_hardness_bandwidth": float(preference_selection_hardness_bandwidth),
        "preference_length_bucketed_batches": bool(preference_length_bucketed_batches),
        "preference_length_bucket_window_mult": float(
            max(2, int(preference_length_bucket_window_mult))
        ),
        "preference_lr_schedule": str(preference_lr_schedule).strip().lower(),
        "preference_warmup_steps": float(max(0, int(preference_warmup_steps))),
        "preference_min_lr_ratio": float(preference_min_lr_ratio),
        "preference_max_grad_norm": float(preference_max_grad_norm),
        "train_tokens_per_sec": float(token_count / elapsed),
        "train_seconds": float(elapsed),
        "runtime_device": str(device),
        "runtime_device_requested": str(runtime_device_requested),
        "runtime_device_resolved": str(runtime_backend),
        "runtime_device_preference": str(runtime_device_preference),
        "runtime_model_dtype": str(model_dtype),
        "runtime_gradient_checkpointing": bool(gradient_checkpointing),
        "runtime_torch_num_threads": float(torch.get_num_threads()),
        "runtime_torch_interop_threads": float(torch.get_num_interop_threads()),
        "lora_use_dora": bool(use_dora),
        "lora_use_rslora": bool(use_rslora),
        "lora_plus_ratio": float(lora_plus_stats.get("lora_plus_ratio", 0.0)),
        "lora_plus_base_group_params": float(lora_plus_stats.get("lora_plus_base_group_params", 0.0)),
        "lora_plus_fast_group_params": float(lora_plus_stats.get("lora_plus_fast_group_params", 0.0)),
        "neftune_noise_alpha": float(neftune_noise_alpha),
        "preference_neftune_noise_alpha": float(preference_neftune_noise_alpha),
        "init_adapter_match_lora": bool(init_adapter_match_lora),
        "save_every_steps": float(checkpoint_every),
        "keep_last_checkpoints": float(max(0, int(keep_last_checkpoints))),
        "skip_sft": bool(skip_sft),
    }
    return adapter_dir, stats


def token_f1(reference: str, hypothesis: str) -> float:
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    ref_counts: Dict[str, int] = {}
    for t in ref:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = 0
    for t in hyp:
        c = ref_counts.get(t, 0)
        if c > 0:
            overlap += 1
            ref_counts[t] = c - 1
    precision = overlap / max(1, len(hyp))
    recall = overlap / max(1, len(ref))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _preview_text(value: object, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", _coerce_text(value))
    if limit <= 3 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _evaluate_model_internal(
    base_model: str,
    eval_pairs: Sequence[ChatPair],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model, tokenizer = _load_base_model_and_tokenizer(base_model, device, for_training=False)
    metrics: Dict[str, float] = {}
    sample_rows: List[Dict[str, object]] = []
    try:
        if adapter_dir is not None:
            model = _load_adapter_with_compat(
                model=model,
                adapter_dir=Path(adapter_dir),
                device=device,
                is_trainable=False,
                merge_for_inference=True,
            )
        _set_model_use_cache(model, enabled=True)
        model.eval()

        losses = []
        f1s = []
        sims = []
        latencies = []
        prompt_token_count = 0.0
        generated_token_count = 0.0
        eval_start = time.time()
        with torch.inference_mode():
            for sample_index, pair in enumerate(eval_pairs):
                row = encode_for_causal_lm(tokenizer, pair, max_length=max_length)
                if row is None:
                    continue
                batch = collate_rows([row], tokenizer.pad_token_id)
                for k in batch:
                    batch[k] = batch[k].to(device)
                out = model(**batch)
                loss_value = float(out.loss.item())
                losses.append(loss_value)

                prompt, _ = _chat_text_pair(tokenizer, pair.user, pair.assistant)
                enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                prompt_tokens = int(enc["input_ids"].shape[1])
                t0 = time.time()
                gen = model.generate(
                    **enc,
                    max_new_tokens=max(8, int(max_new_tokens)),
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                latency_value = float(time.time() - t0)
                latencies.append(latency_value)
                prompt_token_count += float(prompt_tokens)
                new_tokens = gen[0, enc["input_ids"].shape[1] :]
                generated_tokens = int(new_tokens.shape[0])
                generated_token_count += float(generated_tokens)
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                token_f1_value = float(token_f1(pair.assistant, pred))
                char_similarity_value = float(SequenceMatcher(None, pair.assistant, pred).ratio())
                f1s.append(token_f1_value)
                sims.append(char_similarity_value)
                sample_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "source": str(pair.source),
                        "prompt_signature": _prompt_signature(pair.user),
                        "prompt_complexity": float(_prompt_complexity_score(pair.user)),
                        "prompt_words": int(len(pair.user.split())),
                        "reference_words": int(len(pair.assistant.split())),
                        "prompt_tokens": int(prompt_tokens),
                        "generated_tokens": int(generated_tokens),
                        "gen_seconds": float(latency_value),
                        "loss": float(loss_value),
                        "token_f1": float(token_f1_value),
                        "char_similarity": float(char_similarity_value),
                        "user": pair.user,
                        "reference": pair.assistant,
                        "prediction": pred,
                    }
                )

        mean_loss = float(sum(losses) / max(1, len(losses)))
        eval_seconds = float(max(1e-6, time.time() - eval_start))
        metrics = {
            "eval_samples": float(len(losses)),
            "eval_loss": mean_loss,
            "perplexity": float(math.exp(min(20.0, mean_loss))),
            "token_f1": float(sum(f1s) / max(1, len(f1s))),
            "char_similarity": float(sum(sims) / max(1, len(sims))),
            "avg_gen_seconds": float(sum(latencies) / max(1, len(latencies))),
            "avg_prompt_tokens": float(prompt_token_count / max(1, len(losses))),
            "avg_generated_tokens": float(generated_token_count / max(1, len(losses))),
            "total_prompt_tokens": float(prompt_token_count),
            "total_generated_tokens": float(generated_token_count),
            "eval_seconds": eval_seconds,
            "generated_tokens_per_sec": float(generated_token_count / eval_seconds),
        }
    finally:
        try:
            del model
        except Exception:
            pass
        gc.collect()
    return metrics, sample_rows


def evaluate_model(
    base_model: str,
    eval_pairs: Sequence[ChatPair],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: Optional[Path] = None,
) -> Dict[str, float]:
    metrics, _sample_rows = _evaluate_model_internal(
        base_model=base_model,
        eval_pairs=eval_pairs,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
    )
    return metrics


def evaluate_model_detailed(
    base_model: str,
    eval_pairs: Sequence[ChatPair],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    return _evaluate_model_internal(
        base_model=base_model,
        eval_pairs=eval_pairs,
        device=device,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        adapter_dir=adapter_dir,
    )


def save_jsonl(path: Path, pairs: Sequence[ChatPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            row = {"user": pair.user, "assistant": pair.assistant, "source": pair.source}
            metadata = _normalize_chat_pair_metadata(pair.metadata)
            if metadata:
                row["metadata"] = metadata
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_jsonl_records(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if not isinstance(row, dict):
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_benchmark_sample_comparison(
    base_samples: Sequence[Dict[str, object]],
    tuned_samples: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    tuned_by_index: Dict[int, Dict[str, object]] = {}
    for row in tuned_samples:
        if not isinstance(row, dict):
            continue
        try:
            sample_index = int(row.get("sample_index", -1))
        except Exception:
            continue
        if sample_index >= 0:
            tuned_by_index[sample_index] = row

    comparison_rows: List[Dict[str, object]] = []
    for base_row in base_samples:
        if not isinstance(base_row, dict):
            continue
        try:
            sample_index = int(base_row.get("sample_index", -1))
        except Exception:
            continue
        if sample_index < 0:
            continue
        tuned_row = tuned_by_index.get(sample_index)
        if tuned_row is None:
            continue
        base_f1 = float(base_row.get("token_f1", 0.0) or 0.0)
        tuned_f1 = float(tuned_row.get("token_f1", 0.0) or 0.0)
        base_char = float(base_row.get("char_similarity", 0.0) or 0.0)
        tuned_char = float(tuned_row.get("char_similarity", 0.0) or 0.0)
        base_gen = float(base_row.get("gen_seconds", 0.0) or 0.0)
        tuned_gen = float(tuned_row.get("gen_seconds", 0.0) or 0.0)
        base_generated_tokens = int(base_row.get("generated_tokens", 0) or 0)
        tuned_generated_tokens = int(tuned_row.get("generated_tokens", 0) or 0)
        comparison_rows.append(
            {
                "sample_index": int(sample_index),
                "source": str(base_row.get("source", tuned_row.get("source", "")) or ""),
                "prompt_signature": str(base_row.get("prompt_signature", tuned_row.get("prompt_signature", "")) or ""),
                "prompt_complexity": float(
                    base_row.get("prompt_complexity", tuned_row.get("prompt_complexity", 0.0)) or 0.0
                ),
                "user": str(base_row.get("user", tuned_row.get("user", "")) or ""),
                "reference": str(base_row.get("reference", tuned_row.get("reference", "")) or ""),
                "base_prediction": str(base_row.get("prediction", "") or ""),
                "tuned_prediction": str(tuned_row.get("prediction", "") or ""),
                "base_loss": float(base_row.get("loss", 0.0) or 0.0),
                "tuned_loss": float(tuned_row.get("loss", 0.0) or 0.0),
                "base_token_f1": float(base_f1),
                "tuned_token_f1": float(tuned_f1),
                "delta_token_f1": float(tuned_f1 - base_f1),
                "base_char_similarity": float(base_char),
                "tuned_char_similarity": float(tuned_char),
                "delta_char_similarity": float(tuned_char - base_char),
                "base_gen_seconds": float(base_gen),
                "tuned_gen_seconds": float(tuned_gen),
                "delta_gen_seconds": float(tuned_gen - base_gen),
                "base_generated_tokens": int(base_generated_tokens),
                "tuned_generated_tokens": int(tuned_generated_tokens),
                "delta_generated_tokens": int(tuned_generated_tokens - base_generated_tokens),
            }
        )

    comparison_rows.sort(
        key=lambda row: (
            float(row.get("delta_token_f1", 0.0)),
            float(row.get("delta_char_similarity", 0.0)),
            -float(row.get("delta_gen_seconds", 0.0)),
            int(row.get("sample_index", 0)),
        )
    )
    return comparison_rows


def save_benchmark_sample_artifacts(
    output_dir: Path,
    base_samples: Sequence[Dict[str, object]],
    tuned_samples: Optional[Sequence[Dict[str, object]]] = None,
) -> Tuple[Dict[str, str], Dict[str, object]]:
    artifacts: Dict[str, str] = {}
    summary: Dict[str, object] = {}

    base_path = output_dir / "base_samples.jsonl"
    save_jsonl_records(base_path, base_samples)
    artifacts["base_samples_jsonl"] = str(base_path)
    summary["base_sample_count"] = int(len(base_samples))

    tuned_rows = list(tuned_samples or [])
    if tuned_rows:
        tuned_path = output_dir / "tuned_samples.jsonl"
        save_jsonl_records(tuned_path, tuned_rows)
        artifacts["tuned_samples_jsonl"] = str(tuned_path)
        summary["tuned_sample_count"] = int(len(tuned_rows))

        comparison_rows = build_benchmark_sample_comparison(base_samples, tuned_rows)
        comparison_path = output_dir / "sample_comparison.jsonl"
        save_jsonl_records(comparison_path, comparison_rows)
        artifacts["sample_comparison_jsonl"] = str(comparison_path)
        summary["comparison_sample_count"] = int(len(comparison_rows))

        if comparison_rows:
            worst_row = comparison_rows[0]
            summary["worst_regression"] = {
                "sample_index": int(worst_row.get("sample_index", 0) or 0),
                "source": str(worst_row.get("source", "") or ""),
                "prompt_signature": str(worst_row.get("prompt_signature", "") or ""),
                "prompt_complexity": float(worst_row.get("prompt_complexity", 0.0) or 0.0),
                "delta_token_f1": float(worst_row.get("delta_token_f1", 0.0) or 0.0),
                "delta_char_similarity": float(worst_row.get("delta_char_similarity", 0.0) or 0.0),
                "delta_gen_seconds": float(worst_row.get("delta_gen_seconds", 0.0) or 0.0),
                "user_preview": _preview_text(worst_row.get("user", ""), limit=180),
                "reference_preview": _preview_text(worst_row.get("reference", ""), limit=140),
                "base_prediction_preview": _preview_text(worst_row.get("base_prediction", ""), limit=140),
                "tuned_prediction_preview": _preview_text(worst_row.get("tuned_prediction", ""), limit=140),
            }

    return artifacts, summary


def load_saved_chat_pairs(path: Path) -> List[ChatPair]:
    pairs: List[ChatPair] = []
    if not path.exists():
        return pairs
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            user = _coerce_text(record.get("user"))
            assistant = _coerce_text(record.get("assistant"))
            if not user or not assistant:
                continue
            metadata = _normalize_chat_pair_metadata(record.get("metadata"))
            if not metadata:
                metadata = _extract_chat_pair_metadata(record)
            pairs.append(
                ChatPair(
                    user=user,
                    assistant=assistant,
                    source=_coerce_text(record.get("source")) or "dataset",
                    metadata=metadata,
                )
            )
    return pairs


def _path_fingerprint(raw_path: str) -> Dict[str, object]:
    path = Path(raw_path)
    try:
        resolved = str(path.resolve(strict=False))
    except Exception:
        resolved = str(path)
    try:
        stat = path.stat()
        size = int(stat.st_size)
        mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    except OSError:
        size = -1
        mtime_ns = -1
    return {
        "path": resolved,
        "size": size,
        "mtime_ns": mtime_ns,
    }


def _prepared_data_cache_key(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepared_data_cache_paths(output_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        output_dir / "prepared_data_cache_meta.json",
        output_dir / "prepared_train_pairs.jsonl",
        output_dir / "prepared_eval_pairs.jsonl",
    )


def _load_prepared_data_cache(
    output_dir: Path,
    cache_key: str,
) -> Optional[Tuple[List[ChatPair], List[ChatPair], Dict[str, object]]]:
    meta_path, train_path, eval_path = _prepared_data_cache_paths(output_dir)
    if not (meta_path.exists() and train_path.exists() and eval_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[data] failed to read prepared cache meta {meta_path}: {e}")
        return None
    if not isinstance(meta, dict) or str(meta.get("cache_key") or "").strip() != str(cache_key):
        return None
    train_pairs = load_saved_chat_pairs(train_path)
    eval_pairs = load_saved_chat_pairs(eval_path)
    expected_train = int(meta.get("train_count", 0) or 0)
    expected_eval = int(meta.get("eval_count", 0) or 0)
    if len(train_pairs) != expected_train or len(eval_pairs) != expected_eval:
        print(
            "[data] prepared cache counts mismatch; rebuilding "
            f"(train={len(train_pairs)}/{expected_train} eval={len(eval_pairs)}/{expected_eval})"
        )
        return None
    return train_pairs, eval_pairs, meta


def _save_prepared_data_cache(
    output_dir: Path,
    cache_key: str,
    cache_payload: Dict[str, object],
    train_pairs: Sequence[ChatPair],
    eval_pairs: Sequence[ChatPair],
    raw_eval_count: int,
) -> None:
    meta_path, train_path, eval_path = _prepared_data_cache_paths(output_dir)
    save_jsonl(train_path, train_pairs)
    save_jsonl(eval_path, eval_pairs)
    meta = {
        "cache_key": str(cache_key),
        "cache_version": 1,
        "raw_eval_count": int(raw_eval_count),
        "train_count": int(len(train_pairs)),
        "eval_count": int(len(eval_pairs)),
        "created_at": float(time.time()),
        "config": cache_payload,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[data] cached prepared split -> {meta_path}")


def _merge_distillation_pairs(
    train_pairs: Sequence[ChatPair],
    distilled_pairs: Sequence[ChatPair],
    seed: int,
) -> Tuple[List[ChatPair], int]:
    seen = {(pair.user, pair.assistant) for pair in train_pairs}
    mixed = list(train_pairs)
    added = 0
    for pair in distilled_pairs:
        key = (pair.user, pair.assistant)
        if key in seen:
            continue
        seen.add(key)
        mixed.append(pair)
        added += 1
    random.Random(seed).shuffle(mixed)
    return mixed, added


def _checkpoint_training_state_path(adapter_dir_or_checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(adapter_dir_or_checkpoint_dir)
    if checkpoint_dir.name.strip().lower() == "adapter":
        checkpoint_dir = checkpoint_dir.parent
    return checkpoint_dir / "trainer_state.pt"


def _move_state_value_to_device(value: object, device: Any) -> object:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_state_value_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_state_value_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_state_value_to_device(v, device) for v in value)
    return value


def _move_optimizer_state_to_device(optimizer, device: Any) -> None:
    for state_key, state_value in list(optimizer.state.items()):
        optimizer.state[state_key] = _move_state_value_to_device(state_value, device)


def _save_checkpoint_training_state(
    adapter_dir_or_checkpoint_dir: Path,
    stage: str,
    optimizer,
    scheduler=None,
) -> Path:
    state_path = _checkpoint_training_state_path(adapter_dir_or_checkpoint_dir)
    payload = {
        "stage": str(stage).strip().lower() or "sft",
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(payload, state_path)
    return state_path


def _restore_checkpoint_training_state(
    adapter_dir_or_checkpoint_dir: Optional[Path],
    stage: str,
    optimizer,
    scheduler,
    device: Any,
) -> bool:
    if adapter_dir_or_checkpoint_dir is None:
        return False
    state_path = _checkpoint_training_state_path(Path(adapter_dir_or_checkpoint_dir))
    if not state_path.exists():
        return False
    try:
        payload = torch.load(state_path, map_location="cpu")
    except Exception as e:
        print(f"[resume] failed to load training state {state_path}: {e}")
        return False
    if not isinstance(payload, dict):
        print(f"[resume] invalid training state payload at {state_path}")
        return False
    payload_stage = str(payload.get("stage") or "").strip().lower()
    expected_stage = str(stage or "").strip().lower() or "sft"
    if payload_stage and payload_stage != expected_stage:
        print(
            "[resume] training state stage mismatch: "
            f"expected={expected_stage} found={payload_stage} path={state_path}"
        )
        return False
    optimizer_state = payload.get("optimizer_state")
    if not isinstance(optimizer_state, dict):
        print(f"[resume] missing optimizer_state in {state_path}")
        return False
    try:
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state_to_device(optimizer, device)
        scheduler_state = payload.get("scheduler_state")
        if scheduler is not None and isinstance(scheduler_state, dict):
            scheduler.load_state_dict(scheduler_state)
    except Exception as e:
        print(f"[resume] failed to restore training state from {state_path}: {e}")
        return False
    return True


def _read_resume_checkpoint(meta_path: Path) -> Optional[ResumeCheckpoint]:
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[resume] failed to read checkpoint meta {meta_path}: {e}")
        return None
    if not isinstance(raw, dict):
        return None
    adapter_dir_raw = str(raw.get("checkpoint_adapter_dir") or "").strip()
    if not adapter_dir_raw:
        return None
    adapter_dir = Path(adapter_dir_raw)
    if not adapter_dir.exists():
        adapter_dir = meta_path.parent / "adapter"
    if not adapter_dir.exists():
        return None
    return ResumeCheckpoint(
        adapter_dir=adapter_dir,
        stage=str(raw.get("stage") or "").strip().lower() or "sft",
        sft_steps=max(0, int(raw.get("sft_steps", 0) or 0)),
        preference_steps=max(0, int(raw.get("preference_steps", 0) or 0)),
        sft_loss_mean=float(raw.get("sft_loss_mean", 0.0) or 0.0),
        preference_loss_mean=float(raw.get("preference_loss_mean", 0.0) or 0.0),
    )


def _prune_checkpoint_dirs(output_dir: Path, keep_last_checkpoints: int) -> None:
    keep = max(0, int(keep_last_checkpoints))
    if keep <= 0:
        return
    checkpoints_dir = Path(output_dir) / "checkpoints"
    if not checkpoints_dir.exists():
        return
    ranked: List[Tuple[Tuple[int, int, float], Path]] = []
    for meta_path in checkpoints_dir.glob("*/checkpoint_meta.json"):
        state = _read_resume_checkpoint(meta_path)
        if state is None:
            continue
        try:
            mtime = float(meta_path.stat().st_mtime)
        except OSError:
            mtime = -1.0
        stage_rank = 1 if state.stage == "preference" else 0
        step_rank = int(state.preference_steps if state.stage == "preference" else state.sft_steps)
        ranked.append(((stage_rank, step_rank, mtime), meta_path.parent))
    if len(ranked) <= keep:
        return
    ranked.sort(key=lambda item: item[0], reverse=True)
    for _, checkpoint_dir in ranked[keep:]:
        try:
            shutil.rmtree(checkpoint_dir)
            print(f"[checkpoint] pruned old checkpoint -> {checkpoint_dir}")
        except OSError as e:
            print(f"[checkpoint] failed to prune {checkpoint_dir}: {e}")


def _resolve_latest_resume_checkpoint(output_dir: Path) -> Optional[ResumeCheckpoint]:
    latest_ptr = output_dir / "latest_adapter_checkpoint.txt"
    if latest_ptr.exists():
        adapter_dir = Path(latest_ptr.read_text(encoding="utf-8").strip())
        meta_path = adapter_dir.parent / "checkpoint_meta.json"
        if meta_path.exists():
            state = _read_resume_checkpoint(meta_path)
            if state is not None:
                return state

    latest_state: Optional[ResumeCheckpoint] = None
    latest_mtime = -1.0
    latest_progress_key = (-1, -1, -1.0)
    for meta_path in output_dir.glob("checkpoints/*/checkpoint_meta.json"):
        state = _read_resume_checkpoint(meta_path)
        if state is None:
            continue
        try:
            mtime = meta_path.stat().st_mtime
        except OSError:
            mtime = -1.0
        progress_key = (
            1 if state.stage == "preference" else 0,
            int(state.preference_steps if state.stage == "preference" else state.sft_steps),
            float(state.sft_steps),
        )
        if mtime > latest_mtime or (mtime == latest_mtime and progress_key >= latest_progress_key):
            latest_state = state
            latest_mtime = mtime
            latest_progress_key = progress_key
    return latest_state


def _build_explicit_resume_checkpoint(
    init_adapter_dir: str,
    sft_steps: int = 0,
    preference_steps: int = 0,
    sft_loss_mean: float = 0.0,
    preference_loss_mean: float = 0.0,
) -> Optional[ResumeCheckpoint]:
    adapter_dir_raw = str(init_adapter_dir or "").strip()
    if not adapter_dir_raw:
        return None
    sft_steps = max(0, int(sft_steps or 0))
    preference_steps = max(0, int(preference_steps or 0))
    sft_loss_mean = float(sft_loss_mean or 0.0)
    preference_loss_mean = float(preference_loss_mean or 0.0)
    if (
        sft_steps <= 0
        and preference_steps <= 0
        and abs(sft_loss_mean) <= 1e-12
        and abs(preference_loss_mean) <= 1e-12
    ):
        return None
    return ResumeCheckpoint(
        adapter_dir=Path(adapter_dir_raw),
        stage="preference" if preference_steps > 0 else "sft",
        sft_steps=sft_steps,
        preference_steps=preference_steps,
        sft_loss_mean=sft_loss_mean,
        preference_loss_mean=preference_loss_mean,
    )


def plot_benchmark(results: Dict[str, Dict[str, float]], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    models = ["base", "tuned"]
    higher_metrics = ["token_f1", "char_similarity"]
    lower_metrics = ["eval_loss", "perplexity", "avg_gen_seconds"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    xh = range(len(higher_metrics))
    xl = range(len(lower_metrics))
    width = 0.35

    axes[0].bar([x - width / 2 for x in xh], [results["base"][m] for m in higher_metrics], width, label="base")
    axes[0].bar([x + width / 2 for x in xh], [results["tuned"][m] for m in higher_metrics], width, label="tuned")
    axes[0].set_xticks(list(xh))
    axes[0].set_xticklabels(higher_metrics, rotation=15)
    axes[0].set_title("Higher Is Better")
    axes[0].legend()

    axes[1].bar([x - width / 2 for x in xl], [results["base"][m] for m in lower_metrics], width, label="base")
    axes[1].bar([x + width / 2 for x in xl], [results["tuned"][m] for m in lower_metrics], width, label="tuned")
    axes[1].set_xticks(list(xl))
    axes[1].set_xticklabels(lower_metrics, rotation=15)
    axes[1].set_title("Lower Is Better")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fine-tune Qwen on Supermix data and benchmark base vs tuned.")
    ap.add_argument(
        "--data",
        nargs="+",
        default=["datasets/conversation_data.supermix_plus_v27_500k.jsonl"],
        help="One or more Supermix JSONL files with user/assistant pairs.",
    )
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--output_dir", default="artifacts/qwen_supermix_run")
    ap.add_argument("--max_records", type=int, default=480)
    ap.add_argument(
        "--max_source_fraction",
        type=float,
        default=0.0,
        help="Optional cap on per-source share in loaded records (0 disables).",
    )
    ap.add_argument(
        "--max_synthetic_fraction",
        type=float,
        default=0.0,
        help="Optional cap on synthetic-template prompt share in loaded records (0 disables).",
    )
    ap.add_argument(
        "--max_prompt_signature_count",
        type=int,
        default=0,
        help="Optional cap on repeats per normalized prompt signature (0 disables).",
    )
    ap.add_argument(
        "--prompt_signature_cap_exempt_sources",
        default="",
        help="Comma-separated source filenames exempt from prompt-signature caps.",
    )
    ap.add_argument(
        "--data_log_every_records",
        type=int,
        default=5000,
        help="Log dataset loading progress every N accepted pairs (0 disables).",
    )
    ap.add_argument("--eval_size", type=int, default=64)
    ap.add_argument(
        "--eval_split_mode",
        choices=["auto", "random"],
        default="auto",
        help="Use metadata-aware grouped eval splits when possible, otherwise random.",
    )
    ap.add_argument(
        "--eval_min_quality_score",
        type=float,
        default=-1e9,
        help="Drop eval pairs below this paired-response score (disabled by default).",
    )
    ap.add_argument(
        "--eval_drop_synthetic_prompts",
        action="store_true",
        help="Drop synthetic/template prompts from eval_pairs.jsonl and benchmark inputs.",
    )
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--sft_lr_schedule",
        choices=["constant", "cosine", "cosine_restarts"],
        default="constant",
        help="Learning-rate schedule for SFT optimizer.",
    )
    ap.add_argument(
        "--sft_lr_restart_period",
        type=int,
        default=0,
        help="Period (in steps) for cosine_restarts LR schedule (0 disables restarts).",
    )
    ap.add_argument(
        "--sft_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for SFT LR schedule.",
    )
    ap.add_argument(
        "--sft_min_lr_ratio",
        type=float,
        default=0.15,
        help="Minimum LR ratio after cosine decay for SFT schedule.",
    )
    ap.add_argument(
        "--sft_max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for SFT stage (<=0 disables clipping).",
    )
    ap.add_argument(
        "--train_log_every_steps",
        type=int,
        default=2,
        help="Log train/pref metrics every N optimizer steps.",
    )
    ap.add_argument(
        "--save_every_steps",
        type=int,
        default=0,
        help="Save adapter checkpoints every N optimizer steps across SFT/preference (0 disables).",
    )
    ap.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=0,
        help="Keep only the most recent N saved checkpoints under --output_dir/checkpoints (0 disables pruning).",
    )
    ap.add_argument(
        "--resume_sft_steps",
        type=int,
        default=0,
        help="Explicit SFT step provenance for --init_adapter_dir when branching from an external checkpoint.",
    )
    ap.add_argument(
        "--resume_sft_loss_mean",
        type=float,
        default=0.0,
        help="Optional SFT mean-loss provenance paired with --resume_sft_steps for external checkpoint branches.",
    )
    ap.add_argument(
        "--resume_preference_steps",
        type=int,
        default=0,
        help="Explicit preference step provenance for --init_adapter_dir when branching from an external checkpoint.",
    )
    ap.add_argument(
        "--resume_preference_loss_mean",
        type=float,
        default=0.0,
        help="Optional preference mean-loss provenance paired with --resume_preference_steps for external checkpoint branches.",
    )
    ap.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip SFT stage and run only preference stage (requires --init_adapter_dir).",
    )
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--use_dora", action="store_true")
    ap.add_argument("--use_rslora", action="store_true")
    ap.add_argument(
        "--lora_init",
        default="true",
        help="LoRA init mode: true|false|gaussian|pissa_niter_4|olora|eva|corda",
    )
    ap.add_argument(
        "--lora_plus_ratio",
        type=float,
        default=0.0,
        help="LoRA+ learning-rate ratio for LoRA-B parameters (>1 enables separate fast group).",
    )
    ap.add_argument("--neftune_noise_alpha", type=float, default=0.0)
    ap.add_argument(
        "--preference_neftune_noise_alpha",
        type=float,
        default=0.0,
        help="Optional NEFTune embedding-noise alpha used only during preference training.",
    )
    ap.add_argument(
        "--sft_weight_mode",
        choices=["none", "quality"],
        default="none",
        help="Optional weighting strategy for SFT samples.",
    )
    ap.add_argument("--sft_min_weight", type=float, default=0.65)
    ap.add_argument("--sft_max_weight", type=float, default=1.45)
    ap.add_argument(
        "--sft_synthetic_prompt_weight",
        type=float,
        default=0.72,
        help="Multiplier for synthetic template prompts under quality SFT weighting.",
    )
    ap.add_argument(
        "--sft_teacher_source_weight",
        type=float,
        default=0.92,
        help="Multiplier for Supermix teacher-generated responses under quality SFT weighting.",
    )
    ap.add_argument("--sft_quality_anchor_boost", type=float, default=1.08)
    ap.add_argument("--sft_coding_boost", type=float, default=1.06)
    ap.add_argument("--sft_events_boost", type=float, default=1.06)
    ap.add_argument(
        "--sft_reasoning_boost",
        type=float,
        default=1.10,
        help="Extra SFT weight multiplier for reasoning-heavy sources/prompts.",
    )
    ap.add_argument(
        "--sft_prompt_skill_boost",
        type=float,
        default=1.05,
        help="Extra SFT weight multiplier for coding/reasoning user prompts.",
    )
    ap.add_argument(
        "--sft_conversation_boost",
        type=float,
        default=1.08,
        help="Extra SFT weight multiplier for context-following and follow-up coherence.",
    )
    ap.add_argument(
        "--sft_creativity_boost",
        type=float,
        default=1.06,
        help="Extra SFT weight multiplier for prompts that explicitly ask for creative answers.",
    )
    ap.add_argument(
        "--sft_knowledge_density_boost",
        type=float,
        default=1.0,
        help="Extra SFT weight multiplier for information-dense responses (1.0 disables).",
    )
    ap.add_argument(
        "--sft_followup_paraphrase_aug",
        type=int,
        default=1,
        help="How many deterministic paraphrase variants to add per follow-up SFT prompt (0 disables).",
    )
    ap.add_argument(
        "--sft_followup_paraphrase_weight",
        type=float,
        default=0.72,
        help="Sample-weight multiplier for follow-up paraphrase augmentation rows.",
    )
    ap.add_argument(
        "--sft_rdrop_alpha",
        type=float,
        default=0.0,
        help="R-Drop symmetric-KL regularization weight during SFT (0 disables).",
    )
    ap.add_argument(
        "--sft_focal_gamma",
        type=float,
        default=0.0,
        help="Focal-loss gamma for SFT: down-weight easy samples, amplify hard ones (0 disables).",
    )
    ap.add_argument(
        "--sft_eval_every_steps",
        type=int,
        default=0,
        help="Run eval monitoring every N SFT steps and log eval loss (0 disables).",
    )
    ap.add_argument(
        "--sft_early_stop_patience",
        type=int,
        default=0,
        help="Stop SFT early if eval loss doesn't improve for N consecutive evaluations (0 disables).",
    )
    ap.add_argument(
        "--sft_curriculum_quality_ramp",
        type=float,
        default=0.0,
        help="Curriculum ramp: reduce weight of below-median samples early in training, restored over 50%% of steps (0 disables).",
    )
    ap.add_argument(
        "--sft_grad_noise_eta",
        type=float,
        default=0.0,
        help="Gradient noise injection eta (Neelakantan-style, decaying) for escaping sharp minima (0 disables).",
    )
    ap.add_argument(
        "--sft_length_bucketed_batches",
        action="store_true",
        help="Bucket SFT batches by sequence length to reduce padding waste and improve throughput.",
    )
    ap.add_argument(
        "--sft_length_bucket_window_mult",
        type=int,
        default=16,
        help="Window multiplier used for SFT length bucketing.",
    )
    ap.add_argument(
        "--sft_true_packing",
        action="store_true",
        help="Pack multiple SFT train sequences into each row with masked separators for higher token utilization.",
    )
    ap.add_argument(
        "--sft_packing_max_samples_per_row",
        type=int,
        default=0,
        help="Optional cap on the number of original SFT samples packed into one train row (0 disables the cap).",
    )
    ap.add_argument(
        "--sft_selection_strategy",
        choices=["none", "utility_topk", "capacity_aware", "coverage_topk"],
        default="none",
        help="Optional post-filter SFT pair selection strategy for faster, denser training.",
    )
    ap.add_argument(
        "--sft_selection_keep_ratio",
        type=float,
        default=1.0,
        help="Fraction of filtered SFT pairs to keep after selection (1.0 disables trimming).",
    )
    ap.add_argument(
        "--sft_selection_min_keep",
        type=int,
        default=0,
        help="Minimum number of filtered SFT pairs to keep after selection.",
    )
    ap.add_argument(
        "--sft_selection_max_keep",
        type=int,
        default=0,
        help="Maximum number of filtered SFT pairs to keep after selection (0 disables cap).",
    )
    ap.add_argument(
        "--sft_selection_hardness_target",
        type=float,
        default=0.55,
        help="Target difficulty for capacity_aware SFT selection.",
    )
    ap.add_argument(
        "--sft_selection_hardness_bandwidth",
        type=float,
        default=0.28,
        help="Difficulty bandwidth for capacity_aware SFT selection.",
    )
    ap.add_argument(
        "--sft_selection_budget_mode",
        choices=["pairs", "tokens"],
        default="pairs",
        help="Whether SFT selection keep_ratio applies to pair count or estimated token budget.",
    )
    ap.add_argument(
        "--sft_selection_budget_power",
        type=float,
        default=0.5,
        help="Length penalty power for token-budgeted SFT selection (0 ignores length, 1 is score-per-token).",
    )
    ap.add_argument(
        "--sft_selection_scope",
        choices=["all", "verbose_synthetic_teacher"],
        default="all",
        help="Restrict SFT selection to a subset of rows while passing the rest through unchanged.",
    )
    ap.add_argument(
        "--sft_selection_scope_min_words",
        type=int,
        default=40,
        help="Minimum assistant word count for scoped SFT selection candidates.",
    )
    ap.add_argument(
        "--sft_auto_balance_sources",
        action="store_true",
        help="Auto-balance SFT source contributions via per-source sample-weight scaling.",
    )
    ap.add_argument(
        "--sft_source_balance_strength",
        type=float,
        default=0.45,
        help="Strength of source-balance scaling (0 disables effect).",
    )
    ap.add_argument(
        "--sft_source_balance_max_scale",
        type=float,
        default=1.35,
        help="Maximum per-source up/down scale factor for source balancing.",
    )
    ap.add_argument(
        "--sft_min_quality_score",
        type=float,
        default=-1e9,
        help="Drop SFT samples below this assistant quality score (disabled by default).",
    )
    ap.add_argument(
        "--sft_filter_drop_short_answers",
        action="store_true",
        help="Drop short-answer prompts in SFT quality filtering.",
    )
    ap.add_argument(
        "--sft_quality_filter_exempt_sources",
        default="",
        help="Comma-separated source filenames exempt from SFT quality filtering.",
    )
    ap.add_argument(
        "--sft_drop_synthetic_prompts",
        action="store_true",
        help="Drop synthetic/template prompts from SFT even if they pass quality filtering.",
    )
    ap.add_argument(
        "--lora_targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    ap.add_argument(
        "--preference_objective",
        choices=["none", "simpo", "dpo", "ipo", "xpo", "repo", "orpo"],
        default="none",
        help="Preference stage objective. simpo uses sigmoid log-loss; dpo uses reference-relative preference optimization; ipo regresses toward a finite reference-relative target gap; xpo uses chi-square-link preference optimization; repo uses ReLU-margin; orpo uses odds-ratio logistic.",
    )
    ap.add_argument("--preference_steps", type=int, default=0)
    ap.add_argument("--preference_pairs", type=int, default=0)
    ap.add_argument("--preference_beta", type=float, default=2.0)
    ap.add_argument(
        "--preference_beta_end",
        type=float,
        default=-1.0,
        help="Optional final beta for linear schedule over preference steps (<=0 keeps beta constant).",
    )
    ap.add_argument("--preference_margin", type=float, default=0.3)
    ap.add_argument(
        "--preference_margin_end",
        type=float,
        default=-1.0,
        help="Optional final margin for linear schedule over preference steps (<0 keeps margin constant).",
    )
    ap.add_argument(
        "--preference_label_smoothing",
        type=float,
        default=0.0,
        help="Conservative label smoothing for logistic preference losses such as DPO/SimPO (0 disables).",
    )
    ap.add_argument(
        "--preference_xpo_clip",
        type=float,
        default=0.0,
        help="Optional clip radius for χPO preference logits delta (0 disables clipping).",
    )
    ap.add_argument("--preference_sft_weight", type=float, default=0.2)
    ap.add_argument("--preference_length_weight", type=float, default=0.05)
    ap.add_argument(
        "--preference_length_control_strength",
        type=float,
        default=0.0,
        help="LMPO-style extra margin penalty when chosen responses are much longer than rejected ones.",
    )
    ap.add_argument(
        "--preference_length_control_target_ratio",
        type=float,
        default=1.08,
        help="Chosen/rejected length ratio tolerated before length-control margin activates.",
    )
    ap.add_argument(
        "--preference_length_control_max_penalty",
        type=float,
        default=0.35,
        help="Clamp on the per-pair length-control margin penalty.",
    )
    ap.add_argument(
        "--preference_hardness_gamma",
        type=float,
        default=0.0,
        help="Hard-example amplification gamma for preference sample weights (0 disables).",
    )
    ap.add_argument(
        "--preference_robust_alpha",
        type=float,
        default=0.0,
        help="RE-PO-style posterior-correctness reweighting blend for noisy preference pairs (0 disables).",
    )
    ap.add_argument(
        "--preference_robust_eta",
        type=float,
        default=0.0,
        help="Assumed label-flip rate for RE-PO-style robust preference weighting.",
    )
    ap.add_argument(
        "--preference_robust_clip",
        type=float,
        default=3.0,
        help="Clamp RE-PO-style robust preference weights to [1/clip, clip] after normalization.",
    )
    ap.add_argument(
        "--preference_wpo_alpha",
        type=float,
        default=0.0,
        help="Tempered WPO-style current-policy pair reweighting strength (0 disables).",
    )
    ap.add_argument(
        "--preference_wpo_clip",
        type=float,
        default=3.0,
        help="Clamp WPO-style pair weights to [1/clip, clip] after normalization.",
    )
    ap.add_argument(
        "--preference_reference_anchor_weight",
        type=float,
        default=0.0,
        help="Trust-region anchor weight against pre-preference reference margins (0 disables).",
    )
    ap.add_argument(
        "--preference_reference_anchor_batch_size",
        type=int,
        default=4,
        help="Batch size for reference-margin caching used by trust-region anchoring.",
    )
    ap.add_argument("--preference_lr", type=float, default=8e-5)
    ap.add_argument(
        "--preference_lr_schedule",
        choices=["constant", "cosine", "cosine_restarts"],
        default="constant",
        help="Learning-rate schedule for preference optimizer.",
    )
    ap.add_argument(
        "--preference_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps for preference LR schedule.",
    )
    ap.add_argument(
        "--preference_min_lr_ratio",
        type=float,
        default=0.20,
        help="Minimum LR ratio after cosine decay for preference schedule.",
    )
    ap.add_argument(
        "--preference_max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm for preference stage (<=0 disables clipping).",
    )
    ap.add_argument("--preference_max_new_tokens", type=int, default=48)
    ap.add_argument("--preference_prompt_max_tokens", type=int, default=192)
    ap.add_argument(
        "--preference_reject_similarity_min",
        type=float,
        default=0.10,
        help="Minimum similarity for rejected preference candidates to avoid unrelated negatives.",
    )
    ap.add_argument(
        "--preference_candidate_count",
        type=int,
        default=3,
        help="Number of sampled candidates per prompt when mining rejected preference responses.",
    )
    ap.add_argument(
        "--preference_no_greedy_negative",
        action="store_true",
        help="Disable greedy-decoded negative mining in preference pair generation.",
    )
    ap.add_argument(
        "--preference_short_reject_boost",
        type=float,
        default=0.55,
        help="Extra preference weight when rejected candidates are much shorter than chosen responses.",
    )
    ap.add_argument(
        "--preference_long_reject_boost",
        type=float,
        default=0.25,
        help="Extra preference weight when rejected candidates are much longer than chosen responses.",
    )
    ap.add_argument(
        "--preference_min_chosen_quality",
        type=float,
        default=0.9,
        help="Minimum chosen-response quality score for preference mining (non-short prompts).",
    )
    ap.add_argument(
        "--preference_min_chosen_words",
        type=int,
        default=6,
        help="Minimum chosen-response word count for preference mining (non-short prompts).",
    )
    ap.add_argument(
        "--preference_min_quality_gap",
        type=float,
        default=0.05,
        help="Require chosen_quality >= rejected_quality + gap for preference pairs.",
    )
    ap.add_argument(
        "--preference_allow_template_prompts",
        action="store_true",
        help="Allow synthetic template prompts in preference pair generation (disabled by default).",
    )
    ap.add_argument(
        "--preference_max_pairs_per_user",
        type=int,
        default=1,
        help="Maximum number of preference pairs mined per normalized user prompt.",
    )
    ap.add_argument(
        "--preference_max_pairs_per_source",
        type=int,
        default=0,
        help="Optional cap on preference pairs mined per source (0 disables).",
    )
    ap.add_argument(
        "--preference_global_fallback_only",
        action="store_true",
        help="Disable same-source-first fallback candidate selection for preference negatives.",
    )
    ap.add_argument(
        "--preference_mining_mode",
        choices=["auto", "hybrid", "dataset", "generation"],
        default="auto",
        help="Preference pair mining mode. auto disables on-the-fly generation on CPU to avoid stalls.",
    )
    ap.add_argument(
        "--preference_mining_progress_every",
        type=int,
        default=100,
        help="Log preference mining progress every N visited prompts (0 disables).",
    )
    ap.add_argument(
        "--preference_mining_max_seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock budget (seconds) for preference mining (0 disables).",
    )
    ap.add_argument(
        "--preference_mining_max_attempt_factor",
        type=int,
        default=12,
        help="Limit preference mining attempts to factor * target_pairs (<=0 disables cap).",
    )
    ap.add_argument(
        "--preference_coding_focus_boost",
        type=float,
        default=1.15,
        help="Extra preference pair weight for coding prompts.",
    )
    ap.add_argument(
        "--preference_reasoning_focus_boost",
        type=float,
        default=1.12,
        help="Extra preference pair weight for reasoning/problem-solving prompts.",
    )
    ap.add_argument(
        "--preference_counterfactual_rejects_per_prompt",
        type=int,
        default=2,
        help="Inject up to N counterfactual rejected variants for coding/reasoning prompts during preference mining.",
    )
    ap.add_argument(
        "--preference_self_play_budget",
        type=int,
        default=0,
        help="Generate current-policy self-play negatives for up to N prompts during preference mining, even on CPU.",
    )
    ap.add_argument(
        "--preference_self_play_curriculum",
        choices=["easy_to_hard", "hard_first"],
        default="easy_to_hard",
        help="Order prompts used for self-play negative generation.",
    )
    ap.add_argument(
        "--preference_self_play_max_new_tokens",
        type=int,
        default=40,
        help="Token cap for self-play negative generation (0 uses preference_max_new_tokens).",
    )
    ap.add_argument(
        "--preference_stop_signal_strength",
        type=float,
        default=0.0,
        help="Prompt-aware brevity penalty/bonus applied during preference mining for answer-only or concise prompts.",
    )
    ap.add_argument(
        "--preference_stop_rejects_per_prompt",
        type=int,
        default=0,
        help="Inject up to N synthetic overlong rejected variants for prompts that explicitly want brevity.",
    )
    ap.add_argument(
        "--preference_selection_strategy",
        choices=["none", "margin_topk", "capacity_aware", "innovation_mix", "coverage_margin"],
        default="none",
        help="Post-mining preference-pair selection strategy for data curation.",
    )
    ap.add_argument(
        "--preference_selection_keep_ratio",
        type=float,
        default=1.0,
        help="Keep top fraction of mined preference pairs after selection scoring (1.0 keeps all).",
    )
    ap.add_argument(
        "--preference_selection_min_keep",
        type=int,
        default=0,
        help="Minimum preference pairs to keep after selection (0 disables floor).",
    )
    ap.add_argument(
        "--preference_selection_max_keep",
        type=int,
        default=0,
        help="Maximum preference pairs to keep after selection (0 disables cap).",
    )
    ap.add_argument(
        "--preference_selection_hardness_target",
        type=float,
        default=0.45,
        help="Target rejected-similarity level for capacity_aware pair selection.",
    )
    ap.add_argument(
        "--preference_selection_hardness_bandwidth",
        type=float,
        default=0.24,
        help="Bandwidth around hardness target for capacity_aware pair selection.",
    )
    ap.add_argument(
        "--preference_length_bucketed_batches",
        action="store_true",
        help="Bucket preference-stage batches by pair length to reduce padding waste.",
    )
    ap.add_argument(
        "--preference_length_bucket_window_mult",
        type=int,
        default=16,
        help="Window multiplier used for preference length bucketing.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--preference_rescore_every",
        type=int,
        default=0,
        help="Re-score preference pair weights every N steps during preference training (0 disables).",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps", "xpu", "npu", "dml"])
    ap.add_argument(
        "--device_preference",
        default="cuda,npu,xpu,mps,cpu,dml",
        help="Priority order used when --device auto (supports cuda,npu,xpu,mps,cpu,dml).",
    )
    ap.add_argument(
        "--model_dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Base model load dtype. auto picks GPU-friendly dtype and uses float32 on CPU.",
    )
    ap.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable model gradient checkpointing for larger-model memory savings.",
    )
    ap.add_argument(
        "--torch_num_threads",
        type=int,
        default=0,
        help="Torch intra-op CPU threads (0 uses automatic host-core count).",
    )
    ap.add_argument(
        "--torch_interop_threads",
        type=int,
        default=0,
        help="Torch inter-op CPU threads (0 uses a conservative automatic value).",
    )
    ap.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="Torch float32 matmul precision hint.",
    )
    ap.add_argument(
        "--disable_tf32",
        action="store_true",
        help="Disable TF32 when running on CUDA-capable hardware.",
    )
    ap.add_argument(
        "--strict_determinism",
        action="store_true",
        help="Request deterministic CUDA/cuDNN algorithms where supported (may reduce throughput).",
    )
    ap.add_argument("--supermix_distill_ratio", type=float, default=0.25)
    ap.add_argument("--supermix_distill_max", type=int, default=120)
    ap.add_argument(
        "--supermix_distill_best_of",
        type=int,
        default=1,
        help="Sample up to N Supermix teacher candidates at different temperatures and keep the best-scoring response.",
    )
    ap.add_argument(
        "--supermix_distill_log_every",
        type=int,
        default=50,
        help="Log distillation progress every N visited prompts (0 disables).",
    )
    ap.add_argument(
        "--supermix_distill_max_seconds",
        type=float,
        default=0.0,
        help="Optional wall-clock cap for teacher distillation in seconds (0 disables).",
    )
    ap.add_argument(
        "--supermix_distill_min_quality",
        type=float,
        default=0.0,
        help="Minimum quality score required for teacher-generated distillation responses.",
    )
    ap.add_argument(
        "--supermix_distill_min_gain",
        type=float,
        default=0.0,
        help="Require teacher-generated distillation responses to beat the original assistant answer by this margin.",
    )
    ap.add_argument(
        "--supermix_distill_density_bias",
        type=float,
        default=0.0,
        help="Bias teacher distillation candidate choice toward denser answers when quality is close.",
    )
    ap.add_argument(
        "--supermix_distill_gain_bias",
        type=float,
        default=0.35,
        help="Extra ranking weight on teacher quality gain over the original assistant answer.",
    )
    ap.add_argument(
        "--supermix_distill_compactness_bias",
        type=float,
        default=0.45,
        help="Extra ranking weight favoring concise teacher rewrites when quality is comparable.",
    )
    ap.add_argument(
        "--supermix_distill_rank_margin",
        type=float,
        default=0.04,
        help="Minimum ranking margin a teacher response must clear over the original answer to be kept.",
    )
    ap.add_argument(
        "--supermix_distill_allow_synthetic_prompts",
        action="store_true",
        help="Allow teacher distillation on synthetic template prompts.",
    )
    ap.add_argument(
        "--supermix_weights",
        default="runtime_python/champion_model_chat_supermix_v27_500k_ft.pth",
    )
    ap.add_argument(
        "--supermix_meta",
        default="runtime_python/chat_model_meta_supermix_v27_500k.json",
    )
    ap.add_argument(
        "--init_adapter_dir",
        default="",
        help="Optional existing LoRA adapter directory to warm-start from.",
    )
    ap.add_argument(
        "--init_adapter_match_lora",
        action="store_true",
        help="Match LoRA rank/alpha/dropout/targets to init adapter config for warm-start compatibility.",
    )
    ap.add_argument(
        "--resume_from_latest_checkpoint",
        action="store_true",
        help="Resume from the latest checkpoint under --output_dir and continue step counts/LR schedules.",
    )
    ap.add_argument(
        "--skip_benchmark",
        action="store_true",
        help="Skip base/tuned benchmark generation to speed up training iteration.",
    )
    ap.add_argument(
        "--benchmark_eval_limit",
        type=int,
        default=0,
        help="Optional cap on eval pairs used during base/tuned benchmarking (0 uses the full filtered eval set).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
        strict_determinism=bool(args.strict_determinism),
    )
    device, device_info = _resolve_device(
        str(args.device),
        device_preference=str(args.device_preference),
    )
    runtime_backend = str(device_info.get("resolved", _device_backend_name(device))).strip().lower() or "cpu"
    model_dtype = _resolve_torch_dtype(
        str(args.model_dtype),
        device=device,
        resolved_backend=runtime_backend,
    )
    print(
        "[runtime] "
        f"requested={device_info.get('requested', str(args.device))} "
        f"resolved={runtime_backend} "
        f"device={device_info.get('device_repr', str(device))} "
        f"model_dtype={str(model_dtype)} "
        f"gradient_checkpointing={bool(args.gradient_checkpointing)} "
        f"strict_determinism={bool(args.strict_determinism)} "
        f"torch_threads={torch.get_num_threads()} "
        f"interop_threads={torch.get_num_interop_threads()} "
        f"preference={device_info.get('preference', str(args.device_preference))}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_state: Optional[ResumeCheckpoint] = None
    if bool(args.resume_from_latest_checkpoint):
        explicit_init = str(args.init_adapter_dir or "").strip()
        if explicit_init:
            print("[resume] explicit --init_adapter_dir set; skipping auto latest-checkpoint lookup.")
        else:
            resume_state = _resolve_latest_resume_checkpoint(output_dir)
            if resume_state is None:
                print(f"[resume] no checkpoint found under {output_dir}; starting fresh.")
            else:
                args.init_adapter_dir = str(resume_state.adapter_dir)
                if resume_state.stage == "preference":
                    args.skip_sft = True
                print(
                    "[resume] resolved latest checkpoint: "
                    f"stage={resume_state.stage} "
                    f"sft_steps={resume_state.sft_steps} "
                    f"preference_steps={resume_state.preference_steps} "
                    f"adapter={resume_state.adapter_dir}"
                )
    if resume_state is None:
        explicit_resume_state = _build_explicit_resume_checkpoint(
            init_adapter_dir=str(args.init_adapter_dir or "").strip(),
            sft_steps=int(args.resume_sft_steps),
            preference_steps=int(args.resume_preference_steps),
            sft_loss_mean=float(args.resume_sft_loss_mean),
            preference_loss_mean=float(args.resume_preference_loss_mean),
        )
        if explicit_resume_state is not None:
            resume_state = explicit_resume_state
            if resume_state.stage == "preference":
                args.skip_sft = True
            print(
                "[resume] using explicit resume metadata: "
                f"stage={resume_state.stage} "
                f"sft_steps={resume_state.sft_steps} "
                f"preference_steps={resume_state.preference_steps} "
                f"adapter={resume_state.adapter_dir}"
            )
    if bool(args.skip_sft) and not str(args.init_adapter_dir or "").strip():
        raise ValueError("--skip_sft requires --init_adapter_dir to load an existing adapter.")
    prompt_sig_exempt_sources = [
        x.strip()
        for x in str(args.prompt_signature_cap_exempt_sources).split(",")
        if x.strip()
    ]
    sft_filter_exempt_sources = [
        x.strip()
        for x in str(args.sft_quality_filter_exempt_sources).split(",")
        if x.strip()
    ]

    print("[data] loading Supermix pairs...")
    prepared_cache_payload = {
        "data_files": [_path_fingerprint(p) for p in args.data],
        "max_records": int(args.max_records),
        "max_source_fraction": float(args.max_source_fraction),
        "max_synthetic_fraction": float(args.max_synthetic_fraction),
        "max_prompt_signature_count": int(args.max_prompt_signature_count),
        "prompt_signature_cap_exempt_sources": sorted(prompt_sig_exempt_sources),
        "eval_size": int(args.eval_size),
        "eval_split_mode": str(args.eval_split_mode),
        "eval_min_quality_score": float(args.eval_min_quality_score),
        "eval_drop_synthetic_prompts": bool(args.eval_drop_synthetic_prompts),
        "seed": int(args.seed),
    }
    prepared_cache_key = _prepared_data_cache_key(prepared_cache_payload)
    cached_prepared = _load_prepared_data_cache(output_dir, prepared_cache_key)
    if cached_prepared is not None:
        train_pairs, eval_pairs, prepared_cache_meta = cached_prepared
        raw_eval_count = int(prepared_cache_meta.get("raw_eval_count", len(eval_pairs)) or len(eval_pairs))
        print(
            "[data] reused prepared cache: "
            f"train={len(train_pairs)} eval={len(eval_pairs)} raw_eval={raw_eval_count}"
        )
    else:
        all_pairs = load_jsonl_pairs(
            args.data,
            max_records=max(2, int(args.max_records)),
            max_source_fraction=float(args.max_source_fraction),
            max_synthetic_fraction=float(args.max_synthetic_fraction),
            max_prompt_signature_count=int(args.max_prompt_signature_count),
            prompt_signature_cap_exempt_sources=prompt_sig_exempt_sources,
            log_every_records=int(args.data_log_every_records),
        )
        print(f"[data] loaded={len(all_pairs)}")
        train_pairs, eval_pairs = split_train_eval(
            all_pairs,
            eval_size=int(args.eval_size),
            seed=int(args.seed),
            split_mode=str(args.eval_split_mode),
        )
        raw_eval_count = len(eval_pairs)
        eval_pairs = filter_eval_pairs(
            eval_pairs,
            min_quality_score=float(args.eval_min_quality_score),
            drop_synthetic_prompts=bool(args.eval_drop_synthetic_prompts),
        )
        _save_prepared_data_cache(
            output_dir=output_dir,
            cache_key=prepared_cache_key,
            cache_payload=prepared_cache_payload,
            train_pairs=train_pairs,
            eval_pairs=eval_pairs,
            raw_eval_count=raw_eval_count,
        )
    print(f"[data] train={len(train_pairs)} eval={len(eval_pairs)} (raw_eval={raw_eval_count})")

    teacher_generated = 0
    distill_cache_jsonl = output_dir / "teacher_distill_pairs.jsonl"
    if args.supermix_distill_ratio > 0 and args.supermix_distill_max > 0:
        reused_cache = False
        if resume_state is not None and distill_cache_jsonl.exists():
            cached_teacher_pairs = load_saved_chat_pairs(distill_cache_jsonl)
            if cached_teacher_pairs:
                train_pairs, teacher_generated = _merge_distillation_pairs(
                    train_pairs,
                    cached_teacher_pairs,
                    seed=int(args.seed),
                )
                reused_cache = True
                print(
                    f"[distill] reused cached teacher pairs={teacher_generated} from {distill_cache_jsonl}"
                )
        if not reused_cache:
            print("[distill] loading Supermix teacher...")
            teacher_device = str(device)
            if runtime_backend == "dml":
                # DirectML currently fails in the teacher path on scatter_.
                teacher_device = "cpu"
                print("[distill] forcing Supermix teacher to cpu because DirectML inference is unsupported for this path")
            teacher = SupermixTeacher(
                weights_path=args.supermix_weights,
                meta_path=args.supermix_meta,
                device=teacher_device,
            )
            train_pairs, teacher_generated = apply_supermix_distillation(
                train_pairs=train_pairs,
                teacher=teacher,
                ratio=float(args.supermix_distill_ratio),
                max_teacher_samples=int(args.supermix_distill_max),
                seed=int(args.seed),
                min_quality_score=float(args.supermix_distill_min_quality),
                min_quality_gain=float(args.supermix_distill_min_gain),
                skip_synthetic_prompts=not bool(args.supermix_distill_allow_synthetic_prompts),
                log_every=int(args.supermix_distill_log_every),
                max_seconds=float(args.supermix_distill_max_seconds),
                best_of=int(args.supermix_distill_best_of),
                density_bias=float(args.supermix_distill_density_bias),
                gain_bias=float(args.supermix_distill_gain_bias),
                compactness_bias=float(args.supermix_distill_compactness_bias),
                rank_margin=float(args.supermix_distill_rank_margin),
            )
            teacher_pairs = [pair for pair in train_pairs if str(pair.source) == "supermix_teacher"]
            if teacher_pairs:
                save_jsonl(distill_cache_jsonl, teacher_pairs)
                print(f"[distill] cached teacher pairs -> {distill_cache_jsonl}")
        print(f"[distill] teacher_generated={teacher_generated} train_after_mix={len(train_pairs)}")

    eval_jsonl = output_dir / "eval_pairs.jsonl"
    save_jsonl(eval_jsonl, eval_pairs)

    print("[train] fine-tuning Qwen with LoRA...")
    adapter_dir, train_stats = finetune_qwen(
        base_model=args.base_model,
        train_pairs=train_pairs,
        output_dir=output_dir,
        device=device,
        runtime_device_requested=str(device_info.get("requested", str(args.device))),
        runtime_device_resolved=runtime_backend,
        runtime_device_preference=str(device_info.get("preference", str(args.device_preference))),
        model_dtype=model_dtype,
        gradient_checkpointing=bool(args.gradient_checkpointing),
        max_length=int(args.max_length),
        batch_size=int(args.batch_size),
        grad_accum_steps=int(args.grad_accum_steps),
        lr=float(args.lr),
        sft_lr_schedule=str(args.sft_lr_schedule),
        sft_warmup_steps=int(args.sft_warmup_steps),
        sft_min_lr_ratio=float(args.sft_min_lr_ratio),
        sft_max_grad_norm=float(args.sft_max_grad_norm),
        train_log_every_steps=int(args.train_log_every_steps),
        weight_decay=float(args.weight_decay),
        max_steps=int(args.max_steps),
        epochs=int(args.epochs),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        lora_targets=str(args.lora_targets),
        use_dora=bool(args.use_dora),
        use_rslora=bool(args.use_rslora),
        lora_init=str(args.lora_init),
        lora_plus_ratio=float(args.lora_plus_ratio),
        neftune_noise_alpha=float(args.neftune_noise_alpha),
        preference_neftune_noise_alpha=float(args.preference_neftune_noise_alpha),
        sft_weight_mode=str(args.sft_weight_mode),
        sft_min_weight=float(args.sft_min_weight),
        sft_max_weight=float(args.sft_max_weight),
        sft_synthetic_prompt_weight=float(args.sft_synthetic_prompt_weight),
        sft_teacher_source_weight=float(args.sft_teacher_source_weight),
        sft_quality_anchor_boost=float(args.sft_quality_anchor_boost),
        sft_coding_boost=float(args.sft_coding_boost),
        sft_events_boost=float(args.sft_events_boost),
        sft_reasoning_boost=float(args.sft_reasoning_boost),
        sft_prompt_skill_boost=float(args.sft_prompt_skill_boost),
        sft_conversation_boost=float(args.sft_conversation_boost),
        sft_creativity_boost=float(args.sft_creativity_boost),
        sft_knowledge_density_boost=float(args.sft_knowledge_density_boost),
        sft_followup_paraphrase_aug=int(args.sft_followup_paraphrase_aug),
        sft_followup_paraphrase_weight=float(args.sft_followup_paraphrase_weight),
        sft_rdrop_alpha=float(args.sft_rdrop_alpha),
        sft_min_quality_score=float(args.sft_min_quality_score),
        sft_filter_keep_short_answers=not bool(args.sft_filter_drop_short_answers),
        sft_drop_synthetic_prompts=bool(args.sft_drop_synthetic_prompts),
        sft_quality_filter_exempt_sources=sft_filter_exempt_sources,
        sft_auto_balance_sources=bool(args.sft_auto_balance_sources),
        sft_source_balance_strength=float(args.sft_source_balance_strength),
        sft_source_balance_max_scale=float(args.sft_source_balance_max_scale),
        sft_lr_restart_period=int(args.sft_lr_restart_period),
        sft_focal_gamma=float(args.sft_focal_gamma),
        sft_eval_every_steps=int(args.sft_eval_every_steps),
        sft_early_stop_patience=int(args.sft_early_stop_patience),
        sft_curriculum_quality_ramp=float(args.sft_curriculum_quality_ramp),
        sft_grad_noise_eta=float(args.sft_grad_noise_eta),
        sft_length_bucketed_batches=bool(args.sft_length_bucketed_batches),
        sft_length_bucket_window_mult=int(args.sft_length_bucket_window_mult),
        sft_true_packing=bool(args.sft_true_packing),
        sft_packing_max_samples_per_row=int(args.sft_packing_max_samples_per_row),
        sft_selection_strategy=str(args.sft_selection_strategy),
        sft_selection_keep_ratio=float(args.sft_selection_keep_ratio),
        sft_selection_min_keep=int(args.sft_selection_min_keep),
        sft_selection_max_keep=int(args.sft_selection_max_keep),
        sft_selection_hardness_target=float(args.sft_selection_hardness_target),
        sft_selection_hardness_bandwidth=float(args.sft_selection_hardness_bandwidth),
        sft_selection_budget_mode=str(args.sft_selection_budget_mode),
        sft_selection_budget_power=float(args.sft_selection_budget_power),
        sft_selection_scope=str(args.sft_selection_scope),
        sft_selection_scope_min_words=int(args.sft_selection_scope_min_words),
        eval_pairs=eval_pairs,
        preference_objective=str(args.preference_objective),
        preference_steps=int(args.preference_steps),
        preference_pairs=int(args.preference_pairs),
        preference_beta=float(args.preference_beta),
        preference_beta_end=float(args.preference_beta_end),
        preference_margin=float(args.preference_margin),
        preference_margin_end=float(args.preference_margin_end),
        preference_label_smoothing=float(args.preference_label_smoothing),
        preference_xpo_clip=float(args.preference_xpo_clip),
        preference_sft_weight=float(args.preference_sft_weight),
        preference_length_weight=float(args.preference_length_weight),
        preference_length_control_strength=float(args.preference_length_control_strength),
        preference_length_control_target_ratio=float(args.preference_length_control_target_ratio),
        preference_length_control_max_penalty=float(args.preference_length_control_max_penalty),
        preference_hardness_gamma=float(args.preference_hardness_gamma),
        preference_robust_alpha=float(args.preference_robust_alpha),
        preference_robust_eta=float(args.preference_robust_eta),
        preference_robust_clip=float(args.preference_robust_clip),
        preference_wpo_alpha=float(args.preference_wpo_alpha),
        preference_wpo_clip=float(args.preference_wpo_clip),
        preference_reference_anchor_weight=float(args.preference_reference_anchor_weight),
        preference_reference_anchor_batch_size=int(args.preference_reference_anchor_batch_size),
        preference_lr=float(args.preference_lr),
        preference_lr_schedule=str(args.preference_lr_schedule),
        preference_warmup_steps=int(args.preference_warmup_steps),
        preference_min_lr_ratio=float(args.preference_min_lr_ratio),
        preference_max_grad_norm=float(args.preference_max_grad_norm),
        preference_max_new_tokens=int(args.preference_max_new_tokens),
        preference_prompt_max_tokens=int(args.preference_prompt_max_tokens),
        preference_reject_similarity_min=float(args.preference_reject_similarity_min),
        preference_candidate_count=int(args.preference_candidate_count),
        preference_include_greedy_candidate=not bool(args.preference_no_greedy_negative),
        preference_short_reject_boost=float(args.preference_short_reject_boost),
        preference_long_reject_boost=float(args.preference_long_reject_boost),
        preference_min_chosen_quality=float(args.preference_min_chosen_quality),
        preference_min_chosen_words=int(args.preference_min_chosen_words),
        preference_min_quality_gap=float(args.preference_min_quality_gap),
        preference_skip_template_prompts=not bool(args.preference_allow_template_prompts),
        preference_max_pairs_per_user=int(args.preference_max_pairs_per_user),
        preference_max_pairs_per_source=int(args.preference_max_pairs_per_source),
        preference_fallback_same_source_first=not bool(args.preference_global_fallback_only),
        preference_mining_mode=str(args.preference_mining_mode),
        preference_mining_progress_every=int(args.preference_mining_progress_every),
        preference_mining_max_seconds=float(args.preference_mining_max_seconds),
        preference_mining_max_attempt_factor=int(args.preference_mining_max_attempt_factor),
        preference_coding_focus_boost=float(args.preference_coding_focus_boost),
        preference_reasoning_focus_boost=float(args.preference_reasoning_focus_boost),
        preference_counterfactual_rejects_per_prompt=int(args.preference_counterfactual_rejects_per_prompt),
        preference_self_play_budget=int(args.preference_self_play_budget),
        preference_self_play_curriculum=str(args.preference_self_play_curriculum),
        preference_self_play_max_new_tokens=int(args.preference_self_play_max_new_tokens),
        preference_stop_signal_strength=float(args.preference_stop_signal_strength),
        preference_stop_rejects_per_prompt=int(args.preference_stop_rejects_per_prompt),
        preference_selection_strategy=str(args.preference_selection_strategy),
        preference_selection_keep_ratio=float(args.preference_selection_keep_ratio),
        preference_selection_min_keep=int(args.preference_selection_min_keep),
        preference_selection_max_keep=int(args.preference_selection_max_keep),
        preference_selection_hardness_target=float(args.preference_selection_hardness_target),
        preference_selection_hardness_bandwidth=float(args.preference_selection_hardness_bandwidth),
        preference_length_bucketed_batches=bool(args.preference_length_bucketed_batches),
        preference_length_bucket_window_mult=int(args.preference_length_bucket_window_mult),
        seed=int(args.seed),
        save_every_steps=int(args.save_every_steps),
        keep_last_checkpoints=int(args.keep_last_checkpoints),
        preference_rescore_every=int(args.preference_rescore_every),
        skip_sft=bool(args.skip_sft),
        init_adapter_match_lora=bool(args.init_adapter_match_lora),
        init_adapter_dir=str(args.init_adapter_dir or "").strip() or None,
        resume_sft_steps=int(resume_state.sft_steps if resume_state is not None else 0),
        resume_preference_steps=int(resume_state.preference_steps if resume_state is not None else 0),
        resume_sft_loss_mean=float(resume_state.sft_loss_mean if resume_state is not None else 0.0),
        resume_preference_loss_mean=float(
            resume_state.preference_loss_mean if resume_state is not None else 0.0
        ),
    )

    if bool(args.skip_benchmark):
        print("[eval] skipped (--skip_benchmark)")
        base_metrics = {}
        tuned_metrics = {}
        benchmark_artifacts = {}
        benchmark_sample_summary = {}
    else:
        benchmark_eval_pairs = list(eval_pairs)
        benchmark_eval_limit = max(0, int(args.benchmark_eval_limit))
        if benchmark_eval_limit > 0:
            benchmark_eval_pairs = benchmark_eval_pairs[:benchmark_eval_limit]
            print(
                "[eval] benchmark eval limit: "
                f"{len(benchmark_eval_pairs)}/{len(eval_pairs)} filtered eval samples"
            )
        print("[eval] benchmarking base model...")
        base_metrics, base_samples = evaluate_model_detailed(
            base_model=args.base_model,
            eval_pairs=benchmark_eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=None,
        )
        print("[eval] benchmarking fine-tuned model...")
        tuned_metrics, tuned_samples = evaluate_model_detailed(
            base_model=args.base_model,
            eval_pairs=benchmark_eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=adapter_dir,
        )
        benchmark_artifacts, benchmark_sample_summary = save_benchmark_sample_artifacts(
            output_dir=output_dir,
            base_samples=base_samples,
            tuned_samples=tuned_samples,
        )

    results = {
        "config": {
            "base_model": args.base_model,
            "device": str(device),
            "device_requested": str(device_info.get("requested", str(args.device))),
            "device_resolved": runtime_backend,
            "device_preference": str(device_info.get("preference", str(args.device_preference))),
            "model_dtype": str(model_dtype),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_interop_threads": int(torch.get_num_interop_threads()),
            "matmul_precision": str(args.matmul_precision),
            "tf32_enabled": not bool(args.disable_tf32),
            "strict_determinism": bool(args.strict_determinism),
            "resume_from_latest_checkpoint": bool(args.resume_from_latest_checkpoint),
            "resolved_resume_stage": str(resume_state.stage) if resume_state is not None else "",
            "resolved_resume_adapter": str(resume_state.adapter_dir) if resume_state is not None else "",
            "max_records": int(args.max_records),
            "max_source_fraction": float(args.max_source_fraction),
            "max_synthetic_fraction": float(args.max_synthetic_fraction),
            "max_prompt_signature_count": int(args.max_prompt_signature_count),
            "data_log_every_records": int(args.data_log_every_records),
            "prompt_signature_cap_exempt_sources": prompt_sig_exempt_sources,
            "eval_size": int(args.eval_size),
            "eval_split_mode": str(args.eval_split_mode),
            "eval_min_quality_score": float(args.eval_min_quality_score),
            "eval_drop_synthetic_prompts": bool(args.eval_drop_synthetic_prompts),
            "max_steps": int(args.max_steps),
            "max_length": int(args.max_length),
            "lr": float(args.lr),
            "sft_lr_schedule": str(args.sft_lr_schedule),
            "sft_warmup_steps": int(args.sft_warmup_steps),
            "sft_min_lr_ratio": float(args.sft_min_lr_ratio),
            "sft_max_grad_norm": float(args.sft_max_grad_norm),
            "train_log_every_steps": int(args.train_log_every_steps),
            "save_every_steps": int(args.save_every_steps),
            "keep_last_checkpoints": int(args.keep_last_checkpoints),
            "skip_sft": bool(args.skip_sft),
            "supermix_distill_ratio": float(args.supermix_distill_ratio),
            "supermix_distill_max": int(args.supermix_distill_max),
            "supermix_distill_best_of": int(args.supermix_distill_best_of),
            "supermix_distill_log_every": int(args.supermix_distill_log_every),
            "supermix_distill_max_seconds": float(args.supermix_distill_max_seconds),
            "supermix_distill_min_quality": float(args.supermix_distill_min_quality),
            "supermix_distill_min_gain": float(args.supermix_distill_min_gain),
            "supermix_distill_density_bias": float(args.supermix_distill_density_bias),
            "supermix_distill_gain_bias": float(args.supermix_distill_gain_bias),
            "supermix_distill_compactness_bias": float(args.supermix_distill_compactness_bias),
            "supermix_distill_rank_margin": float(args.supermix_distill_rank_margin),
            "supermix_distill_allow_synthetic_prompts": bool(
                args.supermix_distill_allow_synthetic_prompts
            ),
            "distill_cache_path": str(distill_cache_jsonl),
            "teacher_generated": int(teacher_generated),
            "use_dora": bool(args.use_dora),
            "use_rslora": bool(args.use_rslora),
            "lora_init": str(args.lora_init),
            "lora_plus_ratio": float(args.lora_plus_ratio),
            "neftune_noise_alpha": float(args.neftune_noise_alpha),
            "preference_neftune_noise_alpha": float(args.preference_neftune_noise_alpha),
            "sft_weight_mode": str(args.sft_weight_mode),
            "sft_min_weight": float(args.sft_min_weight),
            "sft_max_weight": float(args.sft_max_weight),
            "sft_synthetic_prompt_weight": float(args.sft_synthetic_prompt_weight),
            "sft_teacher_source_weight": float(args.sft_teacher_source_weight),
            "sft_quality_anchor_boost": float(args.sft_quality_anchor_boost),
            "sft_coding_boost": float(args.sft_coding_boost),
            "sft_events_boost": float(args.sft_events_boost),
            "sft_reasoning_boost": float(args.sft_reasoning_boost),
            "sft_prompt_skill_boost": float(args.sft_prompt_skill_boost),
            "sft_conversation_boost": float(args.sft_conversation_boost),
            "sft_creativity_boost": float(args.sft_creativity_boost),
            "sft_knowledge_density_boost": float(args.sft_knowledge_density_boost),
            "sft_followup_paraphrase_aug": int(args.sft_followup_paraphrase_aug),
            "sft_followup_paraphrase_weight": float(args.sft_followup_paraphrase_weight),
            "sft_rdrop_alpha": float(args.sft_rdrop_alpha),
            "sft_focal_gamma": float(args.sft_focal_gamma),
            "sft_eval_every_steps": int(args.sft_eval_every_steps),
            "sft_early_stop_patience": int(args.sft_early_stop_patience),
            "sft_curriculum_quality_ramp": float(args.sft_curriculum_quality_ramp),
            "sft_grad_noise_eta": float(args.sft_grad_noise_eta),
            "sft_length_bucketed_batches": bool(args.sft_length_bucketed_batches),
            "sft_length_bucket_window_mult": int(args.sft_length_bucket_window_mult),
            "sft_true_packing": bool(args.sft_true_packing),
            "sft_packing_max_samples_per_row": int(args.sft_packing_max_samples_per_row),
            "sft_selection_strategy": str(args.sft_selection_strategy),
            "sft_selection_keep_ratio": float(args.sft_selection_keep_ratio),
            "sft_selection_min_keep": int(args.sft_selection_min_keep),
            "sft_selection_max_keep": int(args.sft_selection_max_keep),
            "sft_selection_hardness_target": float(args.sft_selection_hardness_target),
            "sft_selection_hardness_bandwidth": float(args.sft_selection_hardness_bandwidth),
            "sft_selection_budget_mode": str(args.sft_selection_budget_mode),
            "sft_selection_budget_power": float(args.sft_selection_budget_power),
            "sft_selection_scope": str(args.sft_selection_scope),
            "sft_selection_scope_min_words": int(args.sft_selection_scope_min_words),
            "sft_lr_restart_period": int(args.sft_lr_restart_period),
            "preference_rescore_every": int(args.preference_rescore_every),
            "sft_min_quality_score": float(args.sft_min_quality_score),
            "sft_filter_drop_short_answers": bool(args.sft_filter_drop_short_answers),
            "sft_drop_synthetic_prompts": bool(args.sft_drop_synthetic_prompts),
            "sft_quality_filter_exempt_sources": sft_filter_exempt_sources,
            "sft_auto_balance_sources": bool(args.sft_auto_balance_sources),
            "sft_source_balance_strength": float(args.sft_source_balance_strength),
            "sft_source_balance_max_scale": float(args.sft_source_balance_max_scale),
            "preference_objective": str(args.preference_objective),
            "preference_steps": int(args.preference_steps),
            "preference_pairs": int(args.preference_pairs),
            "preference_beta": float(args.preference_beta),
            "preference_beta_end": float(args.preference_beta_end),
            "preference_margin": float(args.preference_margin),
            "preference_margin_end": float(args.preference_margin_end),
            "preference_label_smoothing": float(args.preference_label_smoothing),
            "preference_xpo_clip": float(args.preference_xpo_clip),
            "preference_hardness_gamma": float(args.preference_hardness_gamma),
            "preference_length_control_strength": float(args.preference_length_control_strength),
            "preference_length_control_target_ratio": float(args.preference_length_control_target_ratio),
            "preference_length_control_max_penalty": float(args.preference_length_control_max_penalty),
            "preference_robust_alpha": float(args.preference_robust_alpha),
            "preference_robust_eta": float(args.preference_robust_eta),
            "preference_robust_clip": float(args.preference_robust_clip),
            "preference_wpo_alpha": float(args.preference_wpo_alpha),
            "preference_wpo_clip": float(args.preference_wpo_clip),
            "preference_reference_anchor_weight": float(args.preference_reference_anchor_weight),
            "preference_reference_anchor_batch_size": int(args.preference_reference_anchor_batch_size),
            "preference_lr": float(args.preference_lr),
            "preference_lr_schedule": str(args.preference_lr_schedule),
            "preference_warmup_steps": int(args.preference_warmup_steps),
            "preference_min_lr_ratio": float(args.preference_min_lr_ratio),
            "preference_max_grad_norm": float(args.preference_max_grad_norm),
            "preference_reject_similarity_min": float(args.preference_reject_similarity_min),
            "preference_candidate_count": int(args.preference_candidate_count),
            "preference_no_greedy_negative": bool(args.preference_no_greedy_negative),
            "preference_short_reject_boost": float(args.preference_short_reject_boost),
            "preference_long_reject_boost": float(args.preference_long_reject_boost),
            "preference_min_chosen_quality": float(args.preference_min_chosen_quality),
            "preference_min_chosen_words": int(args.preference_min_chosen_words),
            "preference_min_quality_gap": float(args.preference_min_quality_gap),
            "preference_allow_template_prompts": bool(args.preference_allow_template_prompts),
            "preference_max_pairs_per_user": int(args.preference_max_pairs_per_user),
            "preference_max_pairs_per_source": int(args.preference_max_pairs_per_source),
            "preference_global_fallback_only": bool(args.preference_global_fallback_only),
            "preference_mining_mode": str(args.preference_mining_mode),
            "preference_mining_progress_every": int(args.preference_mining_progress_every),
            "preference_mining_max_seconds": float(args.preference_mining_max_seconds),
            "preference_mining_max_attempt_factor": int(args.preference_mining_max_attempt_factor),
            "preference_coding_focus_boost": float(args.preference_coding_focus_boost),
            "preference_reasoning_focus_boost": float(args.preference_reasoning_focus_boost),
            "preference_counterfactual_rejects_per_prompt": int(
                args.preference_counterfactual_rejects_per_prompt
            ),
            "preference_self_play_budget": int(args.preference_self_play_budget),
            "preference_self_play_curriculum": str(args.preference_self_play_curriculum),
            "preference_self_play_max_new_tokens": int(args.preference_self_play_max_new_tokens),
            "preference_stop_signal_strength": float(args.preference_stop_signal_strength),
            "preference_stop_rejects_per_prompt": int(args.preference_stop_rejects_per_prompt),
            "preference_selection_strategy": str(args.preference_selection_strategy),
            "preference_selection_keep_ratio": float(args.preference_selection_keep_ratio),
            "preference_selection_min_keep": int(args.preference_selection_min_keep),
            "preference_selection_max_keep": int(args.preference_selection_max_keep),
            "preference_selection_hardness_target": float(args.preference_selection_hardness_target),
            "preference_selection_hardness_bandwidth": float(args.preference_selection_hardness_bandwidth),
            "preference_length_bucketed_batches": bool(args.preference_length_bucketed_batches),
            "preference_length_bucket_window_mult": int(args.preference_length_bucket_window_mult),
            "init_adapter_dir": str(args.init_adapter_dir or ""),
            "init_adapter_match_lora": bool(args.init_adapter_match_lora),
            "skip_benchmark": bool(args.skip_benchmark),
            "benchmark_eval_limit": int(args.benchmark_eval_limit),
        },
        "train_stats": train_stats,
        "artifacts": benchmark_artifacts,
        "sample_summary": benchmark_sample_summary,
        "base": base_metrics,
        "tuned": tuned_metrics,
    }
    out_json = output_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out_png = output_dir / "benchmark_comparison.png"
    if base_metrics and tuned_metrics:
        plot_benchmark({"base": base_metrics, "tuned": tuned_metrics}, out_png)
    else:
        out_png = None

    print("[done] artifacts:")
    print(f"  - {adapter_dir}")
    print(f"  - {eval_jsonl}")
    print(f"  - {out_json}")
    if out_png is not None:
        print(f"  - {out_png}")
    print("[summary]")
    print(json.dumps({"train_stats": train_stats, "base": base_metrics, "tuned": tuned_metrics}, indent=2))


if __name__ == "__main__":
    main()
