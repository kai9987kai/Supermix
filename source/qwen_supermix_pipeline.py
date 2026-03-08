import argparse
import gc
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from peft.utils.save_and_load import set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat_pipeline import (
    MODEL_CLASSES,
    build_context,
    choose_bucket_from_logits,
    pick_response,
    resolve_feature_mode,
    text_to_model_input,
)
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
    is_followup: bool = False


@dataclass
class PairAlignmentMetrics:
    conversation: float = 0.0
    reasoning: float = 0.0
    creativity: float = 0.0
    is_followup: bool = False


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pairs_from_messages(messages: Sequence[Dict[str, object]]) -> List[ChatPair]:
    out: List[ChatPair] = []
    history: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
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
                user_text = build_context(
                    history=history[-3:],
                    user_text=pending_user,
                    max_turns=3,
                )
            out.append(ChatPair(user=user_text, assistant=text))
            history.append((pending_user, text))
            pending_user = None
    return out


ARTIFACT_TAG_RE = re.compile(
    r"\[[^\]\n]*(?:variant|worked solution|set\d+|reflective|counterexample|debug|planning|mentor|teaching)[^\]\n]*\]",
    flags=re.IGNORECASE,
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
    low = re.sub(r"\d+", " <num> ", low)
    low = re.sub(r"\s+", " ", low).strip()
    return low[:220]


def _clean_training_text(text: str, is_user: bool) -> str:
    out = _normalize_whitespace(_coerce_text(text))
    if not out:
        return ""

    out = ARTIFACT_TAG_RE.sub(" ", out)
    out = _strip_leading_bracket_tags(out)
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
            user = _coerce_text(record.get("user"))
            assistant = _coerce_text(record.get("assistant"))
            if user and assistant:
                local_pairs.append(ChatPair(user=user, assistant=assistant))
            elif isinstance(record.get("messages"), list):
                local_pairs.extend(_pairs_from_messages(record["messages"]))

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
                yield ChatPair(user=user_text, assistant=assistant_text, source=path.name)


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


def split_train_eval(pairs: List[ChatPair], eval_size: int, seed: int) -> Tuple[List[ChatPair], List[ChatPair]]:
    if len(pairs) < 2:
        raise ValueError("Need at least 2 samples to split train/eval.")
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    eval_n = max(1, min(eval_size, len(pairs) - 1))
    eval_idx = set(idx[:eval_n])
    train = [pairs[i] for i in range(len(pairs)) if i not in eval_idx]
    eval_pairs = [pairs[i] for i in range(len(pairs)) if i in eval_idx]
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


def apply_supermix_distillation(
    train_pairs: List[ChatPair],
    teacher: SupermixTeacher,
    ratio: float,
    max_teacher_samples: int,
    seed: int,
    min_quality_score: float = 0.0,
    skip_synthetic_prompts: bool = True,
    log_every: int = 50,
    max_seconds: float = 0.0,
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
        f"max_seconds={max_seconds if max_seconds > 0 else 'off'}"
    )

    mixed = list(train_pairs)
    generated = 0
    chosen_sorted = sorted(chosen)
    visited = 0
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
        teacher_resp = teacher.generate(pair.user)
        if not teacher_resp:
            continue
        teacher_resp = _clean_training_text(teacher_resp, is_user=False)
        if not teacher_resp:
            continue
        if _looks_like_placeholder_assistant(teacher_resp):
            continue
        if not _is_quality_pair(pair.user, teacher_resp, min_chars=4):
            continue
        teacher_score, _alignment = _paired_response_score(pair.user, teacher_resp)
        if teacher_score < float(min_quality_score):
            continue
        key = (pair.user, teacher_resp)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        mixed.append(ChatPair(user=pair.user, assistant=teacher_resp, source="supermix_teacher"))
        generated += 1
        if log_every > 0 and visited % log_every == 0:
            elapsed = max(1e-6, time.time() - started)
            print(
                "[distill] progress: "
                f"visited={visited}/{len(chosen_sorted)} generated={generated} "
                f"rate={visited / elapsed:.2f}/s"
            )
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


def build_rows(
    tokenizer,
    pairs: Sequence[ChatPair],
    max_length: int,
    row_weight_fn=None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for pair in pairs:
        row = encode_for_causal_lm(tokenizer, pair, max_length=max_length)
        if row is not None:
            if row_weight_fn is not None:
                try:
                    w = float(row_weight_fn(pair))
                except Exception:
                    w = 1.0
                row["sample_weight"] = float(max(0.05, w))
            rows.append(row)
    if not rows:
        raise ValueError("No valid rows produced after tokenization.")
    return rows


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


def _resolve_device(device_spec: str) -> torch.device:
    mode = str(device_spec).strip().lower()
    if mode == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if mode == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[runtime] requested cuda but unavailable; falling back to cpu.")
        return torch.device("cpu")
    if mode == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[runtime] requested mps but unavailable; falling back to cpu.")
        return torch.device("cpu")
    return torch.device("cpu")


def _resolve_torch_dtype(dtype_spec: str, device: torch.device) -> torch.dtype:
    mode = str(dtype_spec).strip().lower()
    explicit = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if mode in explicit:
        resolved = explicit[mode]
    else:
        if device.type == "cuda":
            supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            resolved = torch.bfloat16 if supports_bf16 else torch.float16
        elif device.type == "mps":
            resolved = torch.float16
        else:
            resolved = torch.float32
    if device.type == "cpu" and resolved != torch.float32:
        print("[runtime] cpu training requires float32 weights; overriding requested dtype.")
        return torch.float32
    return resolved


def _load_base_model_and_tokenizer(
    base_model: str,
    device: torch.device,
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
):
    schedule_mode = str(schedule).strip().lower()
    if schedule_mode not in {"constant", "cosine"}:
        schedule_mode = "constant"
    warmup = max(0, int(warmup_steps))
    total = max(1, int(total_steps))
    min_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def _lambda(step_idx: int) -> float:
        step = int(step_idx) + 1
        if warmup > 0 and step <= warmup:
            return max(1e-6, float(step) / float(max(1, warmup)))
        if schedule_mode == "constant" or total <= warmup:
            return 1.0
        progress = float(step - warmup) / float(max(1, total - warmup))
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
    device: torch.device,
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
        peft_model = get_peft_model(peft_model, peft_cfg)

    incompat = set_peft_model_state_dict(peft_model, canonical_state, adapter_name="default")
    missing_count = len(getattr(incompat, "missing_keys", []) or [])
    unexpected_count = len(getattr(incompat, "unexpected_keys", []) or [])
    print(
        f"[adapter] loaded remapped={remapped_count} missing={missing_count} unexpected={unexpected_count}"
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
        rows.append({"chosen": chosen, "rejected": rejected, "weight": float(max(0.1, pair.weight))})
    return rows


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
    token_log_probs = token_log_probs * valid
    lengths = valid.sum(dim=1).clamp_min(1)
    avg_log_prob = token_log_probs.sum(dim=1) / lengths
    return avg_log_prob, lengths


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
) -> float:
    if str(mode).strip().lower() == "none":
        return 1.0

    source = str(pair.source or "").lower()
    user = _coerce_text(pair.user)
    assistant = _fast_cleanup_response_text(str(pair.assistant or ""))
    paired_score, alignment = _paired_response_score(user, assistant)

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

    return float(max(float(min_weight), min(float(max_weight), w)))


def filter_sft_training_pairs(
    pairs: Sequence[ChatPair],
    min_quality_score: float,
    keep_short_answer_prompts: bool,
    exempt_sources: Optional[Sequence[str]] = None,
    min_keep_pairs: int = 32,
) -> List[ChatPair]:
    threshold = float(min_quality_score)
    exempt = {str(x or "").strip().lower() for x in (exempt_sources or []) if str(x or "").strip()}
    if threshold <= -1e8:
        return list(pairs)

    kept: List[ChatPair] = []
    dropped_quality = 0
    dropped_short = 0
    for pair in pairs:
        src_low = str(pair.source or "").strip().lower()
        if src_low in exempt:
            kept.append(pair)
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
        f"dropped_short={dropped_short} exempt_sources={len(exempt)}"
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

    conversation = 0.55 * coverage + 0.18 * number_coverage
    reasoning = _response_reasoning_signal(response) if wants_reasoning else 0.0
    creativity = _response_creativity_signal(response) if wants_creative else 0.0

    if followup and last_assistant:
        prev_tokens = _significant_token_set(last_assistant)
        prev_overlap = 0.0
        if prev_tokens:
            prev_overlap = float(len(prev_tokens & response_tokens)) / float(max(1, len(prev_tokens)))
        prev_word_count = max(1, _word_token_count(last_assistant))
        resp_word_count = max(1, _word_token_count(response))
        len_ratio = float(resp_word_count) / float(prev_word_count)
        conversation += 0.18 * min(1.0, prev_overlap / 0.45)
        if shorten:
            if len_ratio < 0.90:
                conversation += 0.28 * min(1.0, (0.90 - len_ratio) / 0.55)
            else:
                conversation -= 0.10 * min(1.0, (len_ratio - 0.90) / 0.80)
        elif expand:
            if len_ratio > 1.05:
                conversation += 0.24 * min(1.0, (len_ratio - 1.05) / 1.10)
            conversation += 0.18 * reasoning
        elif rewrite:
            sim = float(SequenceMatcher(None, last_assistant.lower(), response.lower()).ratio())
            if 0.18 <= sim <= 0.94:
                conversation += 0.18
            elif sim > 0.98:
                conversation -= 0.10
        elif wants_creative:
            conversation += 0.16 * creativity

    if wants_reasoning:
        conversation += 0.12 * reasoning
    if wants_creative:
        conversation += 0.10 * creativity

    conversation = float(max(-0.30, min(1.60, conversation)))
    return PairAlignmentMetrics(
        conversation=conversation,
        reasoning=float(max(0.0, min(1.0, reasoning))),
        creativity=float(max(0.0, min(1.0, creativity))),
        is_followup=bool(followup),
    )


def _paired_response_score(user_text: str, assistant_text: str) -> Tuple[float, PairAlignmentMetrics]:
    alignment = _response_alignment_metrics(user_text, assistant_text)
    score = _response_quality_score(assistant_text)
    score += 0.32 * float(alignment.conversation)
    score += 0.18 * float(alignment.reasoning)
    score += 0.16 * float(alignment.creativity)
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
            + followup_bonus
        )
        return float(
            base_weight
            * (0.35 + 0.65 * hardness_window)
            * (quality_gap + 0.12 * prompt_complexity + novelty_bonus + 0.05)
        )

    return float(max(0.25, pair.weight))


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
    if mode not in {"none", "margin_topk", "capacity_aware", "innovation_mix"}:
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
    print(
        "[pref] pair selection: "
        f"strategy={mode} keep={len(selected)}/{total} keep_ratio={len(selected)/max(1,total):.3f} "
        f"gap={full_gap:.3f}->{sel_gap:.3f} sim={full_sim:.3f}->{sel_sim:.3f} "
        f"conv={full_conv:.3f}->{sel_conv:.3f} "
        f"reason={full_reason:.3f}->{sel_reason:.3f} "
        f"creative={full_creative:.3f}->{sel_creative:.3f} "
        f"selected_score_mean={sel_score:.4f}"
    )
    return selected


def _pick_rejected_candidate(
    user_text: str,
    chosen_text: str,
    generated: Sequence[str],
    similarity_threshold: float,
    similarity_min: float = 0.0,
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


def _counterfactual_reject_variants(chosen_text: str, rng: random.Random) -> List[str]:
    text = _fast_cleanup_response_text(str(chosen_text or "")).strip()
    if not text:
        return []
    out: List[str] = []
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
    scored_ids: List[Tuple[float, int, str, float, str, float, PairAlignmentMetrics]] = []
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
        chosen_words = _word_token_count(chosen_text)
        if not short_prompt and chosen_words < max(1, int(min_chosen_words)):
            continue
        if not short_prompt and chosen_score < float(min_chosen_quality):
            continue
        prompt_complexity = _prompt_complexity_score(pair.user)
        score = prompt_complexity + 0.18 * max(-1.0, min(2.5, chosen_score))
        score += 0.14 * float(chosen_alignment.conversation)
        score += 0.10 * float(chosen_alignment.reasoning)
        score += 0.08 * float(chosen_alignment.creativity)
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

    try:
        for _score, idx, chosen_text, chosen_score, user_sig, prompt_complexity, chosen_alignment in scored_ids:
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
            if int(max_pairs_per_source) > 0:
                if source_pair_counts.get(source_key, 0) >= int(max_pairs_per_source):
                    continue
            generated: List[str] = []
            if use_generation:
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
                    if bool(include_greedy_candidate):
                        greedy = model.generate(
                            **enc,
                            max_new_tokens=max(16, int(max_new_tokens)),
                            do_sample=False,
                            eos_token_id=tokenizer.eos_token_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                        greedy_new_tokens = greedy[0, enc["input_ids"].shape[1] :]
                        greedy_pred = tokenizer.decode(greedy_new_tokens, skip_special_tokens=True).strip()
                        greedy_pred = _fast_cleanup_response_text(greedy_pred)
                        if greedy_pred:
                            generated.append(greedy_pred)

                    n_candidates = max(1, int(candidate_count))
                    for ci in range(n_candidates):
                        temp = float(temps[ci % len(temps)])
                        top_p = min(0.96, 0.86 + 0.03 * float(ci % 3))
                        gen = model.generate(
                            **enc,
                            max_new_tokens=max(16, int(max_new_tokens)),
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
                except Exception as e:
                    generation_failures += 1
                    if generation_failures <= 3:
                        print(f"[pref] generation warning #{generation_failures}: {e}")

            if bool(prompt_is_coding or prompt_is_reasoning):
                cf_variants = _counterfactual_reject_variants(chosen_text, rng=rng)
                if int(counterfactual_rejects_per_prompt) > 0:
                    cf_variants = cf_variants[: max(1, int(counterfactual_rejects_per_prompt))]
                generated.extend(cf_variants)

            rejected, rejected_sim, rejected_score = _pick_rejected_candidate(
                user_text=pair.user,
                chosen_text=chosen_text,
                generated=generated,
                similarity_threshold=float(similarity_threshold),
                similarity_min=float(reject_similarity_min),
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
                        break
                    if rejected:
                        break

            if not rejected:
                continue
            if _looks_like_placeholder_assistant(rejected):
                continue
            if float(chosen_score) < float(rejected_score) + float(min_quality_gap):
                continue

            pair_weight = _preference_pair_weight(
                chosen_quality=chosen_score,
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
                    quality_gap=float(max(0.0, float(chosen_score) - float(rejected_score))),
                    rejected_similarity=float(max(0.0, min(1.0, float(rejected_sim)))),
                    prompt_complexity=float(max(0.0, float(prompt_complexity))),
                    conversation_score=float(max(0.0, float(chosen_alignment.conversation))),
                    reasoning_score=float(max(0.0, float(chosen_alignment.reasoning))),
                    creativity_score=float(max(0.0, float(chosen_alignment.creativity))),
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
) -> int:
    if not pref_rows:
        return 0

    ref_dataset = PackedPreferenceDataset(pref_rows)
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        collate_fn=lambda b: collate_preference_rows(b, pad_token_id),
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
    device: torch.device,
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
    neftune_noise_alpha: float,
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
    sft_min_quality_score: float,
    sft_filter_keep_short_answers: bool,
    sft_quality_filter_exempt_sources: Optional[Sequence[str]],
    sft_auto_balance_sources: bool,
    sft_source_balance_strength: float,
    sft_source_balance_max_scale: float,
    preference_objective: str,
    preference_steps: int,
    preference_pairs: int,
    preference_beta: float,
    preference_beta_end: float,
    preference_margin: float,
    preference_margin_end: float,
    preference_sft_weight: float,
    preference_length_weight: float,
    preference_hardness_gamma: float,
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
    preference_selection_strategy: str,
    preference_selection_keep_ratio: float,
    preference_selection_min_keep: int,
    preference_selection_max_keep: int,
    preference_selection_hardness_target: float,
    preference_selection_hardness_bandwidth: float,
    seed: int,
    save_every_steps: int,
    skip_sft: bool,
    init_adapter_match_lora: bool,
    init_adapter_dir: Optional[str] = None,
) -> Tuple[Path, Dict[str, float]]:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cpu" and bool(gradient_checkpointing):
        print("[train] gradient checkpointing disabled on cpu (stability + no memory benefit).")
        gradient_checkpointing = False

    init_path: Optional[Path] = None
    if init_adapter_dir:
        init_path = Path(str(init_adapter_dir))

    model, tokenizer = _load_base_model_and_tokenizer(
        base_model,
        device,
        for_training=True,
        model_dtype=model_dtype,
        gradient_checkpointing=gradient_checkpointing,
    )
    print(
        "[train] runtime config: "
        f"device={device.type} model_dtype={str(model_dtype)} "
        f"gradient_checkpointing={bool(gradient_checkpointing)}"
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
    print("[train] stage=prepare_train_mode done")

    print("[train] stage=filter_sft_pairs start")
    filtered_train_pairs = filter_sft_training_pairs(
        train_pairs,
        min_quality_score=float(sft_min_quality_score),
        keep_short_answer_prompts=bool(sft_filter_keep_short_answers),
        exempt_sources=sft_quality_filter_exempt_sources,
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
                )
                * float(source_balance_factors.get(str(p.source or "dataset"), 1.0)),
            ),
        ),
    )
    dataset = PackedChatDataset(train_rows)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        collate_fn=lambda b: collate_rows(b, tokenizer.pad_token_id),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )
    sft_total_steps = max(1, int(max_steps))
    sft_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=_build_lr_lambda(
            schedule=str(sft_lr_schedule),
            warmup_steps=int(sft_warmup_steps),
            total_steps=sft_total_steps,
            min_lr_ratio=float(sft_min_lr_ratio),
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_every = max(0, int(save_every_steps))
    steps = 0
    token_count = 0
    sft_weight_accum = 0.0
    sft_weight_count = 0
    sft_loss_sum = 0.0
    t0 = time.time()
    optim.zero_grad(set_to_none=True)

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
                sample_weights = batch.get("weights")
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device).float()
                    sft_weight_accum += float(sample_weights.sum().item())
                    sft_weight_count += int(sample_weights.numel())

                out = model(
                    input_ids=model_batch["input_ids"],
                    attention_mask=model_batch["attention_mask"],
                )
                seq_logp, _ = _sequence_average_log_prob(out.logits, model_batch["labels"])
                seq_loss = -seq_logp
                if sample_weights is not None:
                    norm_w = sample_weights / sample_weights.mean().clamp_min(1e-6)
                    loss_value = (seq_loss * norm_w).sum() / norm_w.sum().clamp_min(1e-6)
                else:
                    loss_value = seq_loss.mean()
                loss = loss_value / max(1, int(grad_accum_steps))
                loss.backward()

                non_mask = int((model_batch["labels"] != -100).sum().item())
                token_count += non_mask
                sft_loss_sum += float(loss_value.item())

                if (i + 1) % max(1, int(grad_accum_steps)) == 0:
                    if float(sft_max_grad_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, float(sft_max_grad_norm))
                    optim.step()
                    sft_scheduler.step()
                    optim.zero_grad(set_to_none=True)
                    steps += 1
                    if steps % max(1, int(train_log_every_steps)) == 0:
                        current_lr = float(optim.param_groups[0]["lr"])
                        print(
                            f"[train] step={steps} loss={sft_loss_sum / max(1, steps):.4f} "
                            f"lr={current_lr:.6g}"
                        )
                    if checkpoint_every > 0 and steps % checkpoint_every == 0:
                        ckpt_adapter_dir = output_dir / "checkpoints" / f"sft_step_{steps:05d}" / "adapter"
                        ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_adapter_dir)
                        tokenizer.save_pretrained(ckpt_adapter_dir)
                        checkpoint_meta = {
                            "stage": "sft",
                            "sft_steps": int(steps),
                            "sft_loss_mean": float(sft_loss_sum / max(1, steps)),
                            "checkpoint_adapter_dir": str(ckpt_adapter_dir),
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
                    if steps >= max(1, int(max_steps)):
                        break
            if steps >= max(1, int(max_steps)):
                break
        if checkpoint_every > 0 and steps > 0 and steps % checkpoint_every != 0:
            ckpt_adapter_dir = output_dir / "checkpoints" / f"sft_step_{steps:05d}" / "adapter"
            ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_adapter_dir)
            tokenizer.save_pretrained(ckpt_adapter_dir)
            checkpoint_meta = {
                "stage": "sft",
                "sft_steps": int(steps),
                "sft_loss_mean": float(sft_loss_sum / max(1, steps)),
                "checkpoint_adapter_dir": str(ckpt_adapter_dir),
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

    pref_steps_done = 0
    pref_loss_sum = 0.0
    pref_pair_count = 0
    pref_weight_mean = 0.0
    pref_reference_pairs = 0
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
    pref_anchor_weight = max(0.0, float(preference_reference_anchor_weight))
    if pref_objective_mode != "none" and int(preference_steps) > 0 and int(preference_pairs) > 0:
        print(f"[pref] building preference pairs (mode={pref_objective_mode})...")
        print(
            "[pref] objective schedule: "
            f"beta={pref_beta_start:.4f}->{pref_beta_final:.4f} "
            f"margin={pref_margin_start:.4f}->{pref_margin_final:.4f} "
            f"hardness_gamma={pref_hardness_gamma:.3f} "
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
            selection_strategy=str(preference_selection_strategy),
            selection_keep_ratio=float(preference_selection_keep_ratio),
            selection_min_keep=int(preference_selection_min_keep),
            selection_max_keep=int(preference_selection_max_keep),
            selection_hardness_target=float(preference_selection_hardness_target),
            selection_hardness_bandwidth=float(preference_selection_hardness_bandwidth),
        )
        pref_pair_count = len(pref_pairs)
        if pref_pairs:
            pref_weight_mean = float(sum(float(p.weight) for p in pref_pairs) / max(1, len(pref_pairs)))
        print(f"[pref] pairs={pref_pair_count}")
        pref_rows = build_preference_rows(tokenizer, pref_pairs, max_length=max_length)
        if pref_rows:
            if pref_anchor_weight > 0.0:
                print("[pref] caching reference margins for trust-region anchoring...")
                pref_reference_pairs = annotate_preference_rows_with_reference_logps(
                    model=model,
                    pref_rows=pref_rows,
                    device=device,
                    pad_token_id=tokenizer.pad_token_id,
                    batch_size=int(preference_reference_anchor_batch_size),
                )
                print(f"[pref] reference margins cached for {pref_reference_pairs} pairs")
            pref_dataset = PackedPreferenceDataset(pref_rows)
            pref_loader = DataLoader(
                pref_dataset,
                batch_size=max(1, int(batch_size)),
                shuffle=True,
                collate_fn=lambda b: collate_preference_rows(b, tokenizer.pad_token_id),
            )

            pref_optim = torch.optim.AdamW(
                trainable_params,
                lr=float(preference_lr),
                weight_decay=float(weight_decay),
            )
            pref_scheduler = torch.optim.lr_scheduler.LambdaLR(
                pref_optim,
                lr_lambda=_build_lr_lambda(
                    schedule=str(preference_lr_schedule),
                    warmup_steps=int(preference_warmup_steps),
                    total_steps=max(1, int(preference_steps)),
                    min_lr_ratio=float(preference_min_lr_ratio),
                ),
            )
            pref_optim.zero_grad(set_to_none=True)
            model.train()

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

                    if pref_objective_mode == "repo":
                        pref_core = torch.relu(margin_t - delta)
                    elif pref_objective_mode == "orpo":
                        chosen_odds = _log_odds_from_avg_log_prob(chosen_logp)
                        rejected_odds = _log_odds_from_avg_log_prob(rejected_logp)
                        odds_delta = chosen_odds - rejected_odds
                        pref_core = torch.nn.functional.softplus(-beta_t * (odds_delta - margin_t))
                    else:
                        z = beta_t * (delta - margin_t)
                        pref_core = -torch.log(torch.sigmoid(z).clamp_min(1e-8))

                    if pref_hardness_gamma > 0.0:
                        hardness_weight = (1.0 - torch.sigmoid((beta_t * delta).detach())).pow(pref_hardness_gamma)
                        sample_weights = sample_weights * (1.0 + hardness_weight)
                        sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)

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
                            f"lr={pref_lr_now:.6g} beta={beta_now:.4f} margin={margin_now:.4f}"
                        )
                    if checkpoint_every > 0 and pref_steps_done % checkpoint_every == 0:
                        ckpt_adapter_dir = output_dir / "checkpoints" / f"pref_step_{pref_steps_done:05d}" / "adapter"
                        ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(ckpt_adapter_dir)
                        tokenizer.save_pretrained(ckpt_adapter_dir)
                        checkpoint_meta = {
                            "stage": "preference",
                            "sft_steps": int(steps),
                            "preference_steps": int(pref_steps_done),
                            "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
                            "checkpoint_adapter_dir": str(ckpt_adapter_dir),
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
                    if pref_steps_done >= target_steps:
                        break
                if pref_steps_done >= target_steps:
                    break
            if checkpoint_every > 0 and pref_steps_done > 0 and pref_steps_done % checkpoint_every != 0:
                ckpt_adapter_dir = output_dir / "checkpoints" / f"pref_step_{pref_steps_done:05d}" / "adapter"
                ckpt_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ckpt_adapter_dir)
                tokenizer.save_pretrained(ckpt_adapter_dir)
                checkpoint_meta = {
                    "stage": "preference",
                    "sft_steps": int(steps),
                    "preference_steps": int(pref_steps_done),
                    "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
                    "checkpoint_adapter_dir": str(ckpt_adapter_dir),
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

    if noise_hook is not None:
        noise_hook.remove()

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    elapsed = max(1e-6, time.time() - t0)
    stats = {
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
        "preference_steps": float(pref_steps_done),
        "preference_loss_mean": float(pref_loss_sum / max(1, pref_steps_done)),
        "preference_pairs": float(pref_pair_count),
        "preference_weight_mean": float(pref_weight_mean),
        "preference_objective": pref_objective_mode,
        "preference_beta_start": float(pref_beta_start),
        "preference_beta_end": float(pref_beta_final),
        "preference_margin_start": float(pref_margin_start),
        "preference_margin_end": float(pref_margin_final),
        "preference_hardness_gamma": float(pref_hardness_gamma),
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
        "preference_lr_schedule": str(preference_lr_schedule).strip().lower(),
        "preference_warmup_steps": float(max(0, int(preference_warmup_steps))),
        "preference_min_lr_ratio": float(preference_min_lr_ratio),
        "preference_max_grad_norm": float(preference_max_grad_norm),
        "train_tokens_per_sec": float(token_count / elapsed),
        "train_seconds": float(elapsed),
        "runtime_device": str(device),
        "runtime_model_dtype": str(model_dtype),
        "runtime_gradient_checkpointing": bool(gradient_checkpointing),
        "lora_use_dora": bool(use_dora),
        "lora_use_rslora": bool(use_rslora),
        "init_adapter_match_lora": bool(init_adapter_match_lora),
        "save_every_steps": float(checkpoint_every),
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


def evaluate_model(
    base_model: str,
    eval_pairs: Sequence[ChatPair],
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    adapter_dir: Optional[Path] = None,
) -> Dict[str, float]:
    model, tokenizer = _load_base_model_and_tokenizer(base_model, device, for_training=False)
    metrics: Dict[str, float] = {}
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
            for pair in eval_pairs:
                row = encode_for_causal_lm(tokenizer, pair, max_length=max_length)
                if row is None:
                    continue
                batch = collate_rows([row], tokenizer.pad_token_id)
                for k in batch:
                    batch[k] = batch[k].to(device)
                out = model(**batch)
                losses.append(float(out.loss.item()))

                prompt, _ = _chat_text_pair(tokenizer, pair.user, pair.assistant)
                enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
                t0 = time.time()
                gen = model.generate(
                    **enc,
                    max_new_tokens=max(8, int(max_new_tokens)),
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
                latencies.append(float(time.time() - t0))
                prompt_token_count += float(enc["input_ids"].shape[1])
                new_tokens = gen[0, enc["input_ids"].shape[1] :]
                generated_token_count += float(new_tokens.shape[0])
                pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                f1s.append(float(token_f1(pair.assistant, pred)))
                sims.append(float(SequenceMatcher(None, pair.assistant, pred).ratio()))

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
    return metrics


def save_jsonl(path: Path, pairs: Sequence[ChatPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            row = {"user": pair.user, "assistant": pair.assistant, "source": pair.source}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--sft_lr_schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="Learning-rate schedule for SFT optimizer.",
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
    ap.add_argument("--neftune_noise_alpha", type=float, default=0.0)
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
        "--lora_targets",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    ap.add_argument(
        "--preference_objective",
        choices=["none", "simpo", "repo", "orpo"],
        default="none",
        help="Preference stage objective. simpo uses sigmoid log-loss; repo uses ReLU-margin; orpo uses odds-ratio logistic.",
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
    ap.add_argument("--preference_sft_weight", type=float, default=0.2)
    ap.add_argument("--preference_length_weight", type=float, default=0.05)
    ap.add_argument(
        "--preference_hardness_gamma",
        type=float,
        default=0.0,
        help="Hard-example amplification gamma for preference sample weights (0 disables).",
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
        choices=["constant", "cosine"],
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
        "--preference_selection_strategy",
        choices=["none", "margin_topk", "capacity_aware", "innovation_mix"],
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
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
    ap.add_argument("--supermix_distill_ratio", type=float, default=0.25)
    ap.add_argument("--supermix_distill_max", type=int, default=120)
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
        "--skip_benchmark",
        action="store_true",
        help="Skip base/tuned benchmark generation to speed up training iteration.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = _resolve_device(str(args.device))
    model_dtype = _resolve_torch_dtype(str(args.model_dtype), device=device)
    print(
        "[runtime] "
        f"device={device.type} model_dtype={str(model_dtype)} "
        f"gradient_checkpointing={bool(args.gradient_checkpointing)}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    train_pairs, eval_pairs = split_train_eval(all_pairs, eval_size=int(args.eval_size), seed=int(args.seed))
    print(f"[data] train={len(train_pairs)} eval={len(eval_pairs)}")

    teacher_generated = 0
    if args.supermix_distill_ratio > 0 and args.supermix_distill_max > 0:
        print("[distill] loading Supermix teacher...")
        teacher = SupermixTeacher(
            weights_path=args.supermix_weights,
            meta_path=args.supermix_meta,
            device=str(device),
        )
        train_pairs, teacher_generated = apply_supermix_distillation(
            train_pairs=train_pairs,
            teacher=teacher,
            ratio=float(args.supermix_distill_ratio),
            max_teacher_samples=int(args.supermix_distill_max),
            seed=int(args.seed),
            min_quality_score=float(args.supermix_distill_min_quality),
            skip_synthetic_prompts=not bool(args.supermix_distill_allow_synthetic_prompts),
            log_every=int(args.supermix_distill_log_every),
            max_seconds=float(args.supermix_distill_max_seconds),
        )
        print(f"[distill] teacher_generated={teacher_generated} train_after_mix={len(train_pairs)}")

    eval_jsonl = output_dir / "eval_pairs.jsonl"
    save_jsonl(eval_jsonl, eval_pairs)

    print("[train] fine-tuning Qwen with LoRA...")
    adapter_dir, train_stats = finetune_qwen(
        base_model=args.base_model,
        train_pairs=train_pairs,
        output_dir=output_dir,
        device=device,
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
        neftune_noise_alpha=float(args.neftune_noise_alpha),
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
        sft_min_quality_score=float(args.sft_min_quality_score),
        sft_filter_keep_short_answers=not bool(args.sft_filter_drop_short_answers),
        sft_quality_filter_exempt_sources=sft_filter_exempt_sources,
        sft_auto_balance_sources=bool(args.sft_auto_balance_sources),
        sft_source_balance_strength=float(args.sft_source_balance_strength),
        sft_source_balance_max_scale=float(args.sft_source_balance_max_scale),
        preference_objective=str(args.preference_objective),
        preference_steps=int(args.preference_steps),
        preference_pairs=int(args.preference_pairs),
        preference_beta=float(args.preference_beta),
        preference_beta_end=float(args.preference_beta_end),
        preference_margin=float(args.preference_margin),
        preference_margin_end=float(args.preference_margin_end),
        preference_sft_weight=float(args.preference_sft_weight),
        preference_length_weight=float(args.preference_length_weight),
        preference_hardness_gamma=float(args.preference_hardness_gamma),
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
        preference_selection_strategy=str(args.preference_selection_strategy),
        preference_selection_keep_ratio=float(args.preference_selection_keep_ratio),
        preference_selection_min_keep=int(args.preference_selection_min_keep),
        preference_selection_max_keep=int(args.preference_selection_max_keep),
        preference_selection_hardness_target=float(args.preference_selection_hardness_target),
        preference_selection_hardness_bandwidth=float(args.preference_selection_hardness_bandwidth),
        seed=int(args.seed),
        save_every_steps=int(args.save_every_steps),
        skip_sft=bool(args.skip_sft),
        init_adapter_match_lora=bool(args.init_adapter_match_lora),
        init_adapter_dir=str(args.init_adapter_dir or "").strip() or None,
    )

    if bool(args.skip_benchmark):
        print("[eval] skipped (--skip_benchmark)")
        base_metrics = {}
        tuned_metrics = {}
    else:
        print("[eval] benchmarking base model...")
        base_metrics = evaluate_model(
            base_model=args.base_model,
            eval_pairs=eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=None,
        )
        print("[eval] benchmarking fine-tuned model...")
        tuned_metrics = evaluate_model(
            base_model=args.base_model,
            eval_pairs=eval_pairs,
            device=device,
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            adapter_dir=adapter_dir,
        )

    results = {
        "config": {
            "base_model": args.base_model,
            "device": str(device),
            "model_dtype": str(model_dtype),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "max_records": int(args.max_records),
            "max_source_fraction": float(args.max_source_fraction),
            "max_synthetic_fraction": float(args.max_synthetic_fraction),
            "max_prompt_signature_count": int(args.max_prompt_signature_count),
            "data_log_every_records": int(args.data_log_every_records),
            "prompt_signature_cap_exempt_sources": prompt_sig_exempt_sources,
            "eval_size": int(args.eval_size),
            "max_steps": int(args.max_steps),
            "max_length": int(args.max_length),
            "lr": float(args.lr),
            "sft_lr_schedule": str(args.sft_lr_schedule),
            "sft_warmup_steps": int(args.sft_warmup_steps),
            "sft_min_lr_ratio": float(args.sft_min_lr_ratio),
            "sft_max_grad_norm": float(args.sft_max_grad_norm),
            "train_log_every_steps": int(args.train_log_every_steps),
            "save_every_steps": int(args.save_every_steps),
            "skip_sft": bool(args.skip_sft),
            "supermix_distill_ratio": float(args.supermix_distill_ratio),
            "supermix_distill_max": int(args.supermix_distill_max),
            "supermix_distill_log_every": int(args.supermix_distill_log_every),
            "supermix_distill_max_seconds": float(args.supermix_distill_max_seconds),
            "supermix_distill_min_quality": float(args.supermix_distill_min_quality),
            "supermix_distill_allow_synthetic_prompts": bool(
                args.supermix_distill_allow_synthetic_prompts
            ),
            "teacher_generated": int(teacher_generated),
            "use_dora": bool(args.use_dora),
            "use_rslora": bool(args.use_rslora),
            "lora_init": str(args.lora_init),
            "neftune_noise_alpha": float(args.neftune_noise_alpha),
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
            "sft_min_quality_score": float(args.sft_min_quality_score),
            "sft_filter_drop_short_answers": bool(args.sft_filter_drop_short_answers),
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
            "preference_hardness_gamma": float(args.preference_hardness_gamma),
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
            "preference_selection_strategy": str(args.preference_selection_strategy),
            "preference_selection_keep_ratio": float(args.preference_selection_keep_ratio),
            "preference_selection_min_keep": int(args.preference_selection_min_keep),
            "preference_selection_max_keep": int(args.preference_selection_max_keep),
            "preference_selection_hardness_target": float(args.preference_selection_hardness_target),
            "preference_selection_hardness_bandwidth": float(args.preference_selection_hardness_bandwidth),
            "init_adapter_dir": str(args.init_adapter_dir or ""),
            "init_adapter_match_lora": bool(args.init_adapter_match_lora),
            "skip_benchmark": bool(args.skip_benchmark),
        },
        "train_stats": train_stats,
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
