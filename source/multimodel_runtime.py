from __future__ import annotations

import copy
import gc
import hashlib
import io
import json
import logging
import math
import re
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote
from urllib.request import urlopen

import torch
from PIL import Image

import chat_app
import chat_web_app
from chat_export import copy_generated_image, render_chat_transcript_image
from chat_image_variant_app import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_NEGATIVE_PROMPT,
    ImageVariantEngine,
)
from conversation_state import (
    build_conversation_state,
    conversation_state_diagnostics,
)
from device_utils import configure_torch_runtime, resolve_device
from grounding_runtime import (
    build_evidence_bundle,
    finalize_grounded_response,
    plan_grounding,
    redact_external_query,
)
from reasoning_engine import reasoning_diagnostics
from interaction_planner import (
    finalize_response_for_interaction,
    interaction_plan_diagnostics,
    plan_interaction,
)
from prompt_understanding import (
    analyze_prompt,
    evaluate_response_constraints,
    prompt_understanding_diagnostics,
)
from multimodel_catalog import (
    DEFAULT_COMMON_SUMMARY,
    DEFAULT_MODELS_DIR,
    ModelRecord,
    choose_auto_model,
    describe_model_artifact_name,
    discover_model_records,
)
from multimodel_memory import ConversationMemoryStore
from route_policy_ledger import (
    OUTCOME_CONTRACT_SCHEMA_VERSION,
    SUPPORT_SCHEMA_VERSION,
    DecisionNotFoundError,
    RoutePolicyLedger,
    build_logging_support_envelope,
    build_route_outcome_contracts,
    hash_session_identity,
)
from route_policy_lab import POLICY_PROFILES, analyze_route_policy
from route_policy_explorer import plan_adjacent_route_study
from route_policy_protocol import (
    audit_route_study_review_bundle,
    build_route_study_protocol,
    build_route_study_review_bundle_from_input,
)
from route_policy_shadow_registry import RouteShadowAssignmentRegistry
from multimodel_tools import (
    CmdOpenTool,
    ToolEvent,
    WebSearchTool,
    format_tool_results,
    parse_tool_requests,
    should_offer_web_search,
    should_offer_open_cmd,
    strip_tool_calls,
)
from dcgan_image_model import DCGANImageEngine, DCGAN_SPECS, _find_generator_weights
from math_equation_model import MathEquationEngine, format_math_response
from mattergen_generation_model import MatterGenMicroEngine, format_mattergen_response
from protein_folding_model import ProteinFoldingEngine, format_protein_response
from three_d_generation_model import ThreeDGenerationEngine, format_3d_generation_response
from native_image_infer_v36 import ChampionNetFrontierCollectiveNativeImage, save_prompt_image as save_prompt_image_v36
from native_image_infer_v37_lite import ChampionNetUltraExpertNativeImageLite, save_prompt_image as save_prompt_image_v37
from native_image_infer_v38_xlite import ChampionNetUltraExpertNativeImageExtraLite, save_prompt_image as save_prompt_image_v38
from image_recognition_model import ScienceImageRecognitionEngine
from omni_collective_model import OmniCollectiveEngine
from omni_collective_v3_model import OmniCollectiveEngineV3
from omni_collective_v4_model import OmniCollectiveEngineV4
from omni_collective_v5_model import OmniCollectiveEngineV5
from omni_collective_v6_model import OmniCollectiveEngineV6
from omni_collective_v7_model import OmniCollectiveEngineV7
from omni_collective_v8_model import OmniCollectiveEngineV8
try:
    from omni_collective_v42_model import OmniCollectiveEngineV42
except ImportError:
    OmniCollectiveEngineV42 = None

try:
    from omni_collective_v41_model import OmniCollectiveEngineV41
except ImportError:
    OmniCollectiveEngineV41 = None

try:
    from omni_collective_v46_model import OmniCollectiveEngineV46
except ImportError:
    OmniCollectiveEngineV46 = None

try:
    from omni_collective_v47_model import OmniCollectiveEnginev47, OmniPredictionv47
except ImportError:
    OmniCollectiveEnginev47, OmniPredictionv47 = None, None

try:
    from omni_collective_v48_model import OmniCollectiveEnginev48, OmniPredictionv48
except ImportError:
    OmniCollectiveEnginev48, OmniPredictionv48 = None, None

from run import safe_load_state_dict


@dataclass
class ChatResult:
    kind: str
    model_key: str
    model_label: str
    route_reason: str
    response: str = ""
    timing: Optional[Dict[str, Any]] = None
    compute: Optional[Dict[str, Any]] = None
    image_url: str = ""
    output_path: str = ""
    prompt_used: str = ""
    refined_prompt: str = ""
    agent_trace: Optional[Dict[str, Any]] = None
    conversation: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "kind": self.kind,
            "model_key": self.model_key,
            "model_label": self.model_label,
            "route_reason": self.route_reason,
            "response": self.response,
            "timing": self.timing or {},
            "compute": self.compute or {},
            "image_url": self.image_url,
            "output_path": self.output_path,
            "prompt_used": self.prompt_used,
            "refined_prompt": self.refined_prompt,
            "agent_trace": self.agent_trace or {},
            "conversation": self.conversation or {},
        }


def finalize_chat_result_for_interaction(
    result: ChatResult,
    *,
    user_text: str,
    interaction_plan: Mapping[str, Any],
    relevance_context: str = "",
) -> ChatResult:
    """Finalize one routed result without changing routing or compute decisions."""

    interaction = interaction_plan_diagnostics(interaction_plan)
    if result.kind == "text" and str(result.response or "").strip():
        guard = finalize_response_for_interaction(
            result.response,
            user_text,
            interaction_plan,
            relevance_context=relevance_context,
        )
        result.response = str(guard.get("text") or "")
        audit = dict(guard.get("audit") or {})
        response_guard: Dict[str, Any] = {
            "changed": bool(guard.get("changed", False)),
            "reason": str(guard.get("reason") or "candidate_aligned"),
            "audit": {
                "accepted": bool(audit.get("accepted", False)),
                "coverage": float(audit.get("coverage") or 0.0),
                "missing": list(audit.get("missing") or []),
                "violations": list(audit.get("violations") or []),
                "lexical_relevance": float(audit.get("lexical_relevance") or 0.0),
            },
        }
    else:
        response_guard = {
            "changed": False,
            "reason": (
                "non_text_result"
                if result.kind != "text"
                else "empty_text_result"
            ),
        }
    interaction["response_guard"] = response_guard
    trace = dict(result.agent_trace or {})
    trace["interaction"] = interaction
    result.agent_trace = trace
    return result


def _safe_slug(text: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cleaned.split("-") if part)
    return cooked[:72] or "artifact"


def _trim_text(text: str, limit: int = 320) -> str:
    cooked = " ".join(str(text or "").strip().split())
    return cooked[:limit]


def _coerce_int_setting(value: Any, default: int, *, minimum: int = 0, maximum: int = 100_000) -> int:
    try:
        cooked = int(float(value))
    except (TypeError, ValueError):
        cooked = int(default)
    return max(int(minimum), min(int(maximum), cooked))


def _record_route_model_call(settings: Dict[str, Any], count: int = 1) -> None:
    """Increment the request-local model invocation counter when one is active."""
    counter = settings.get("_route_model_call_counter")
    if not isinstance(counter, dict):
        return
    counter["count"] = max(0, int(counter.get("count") or 0)) + max(0, int(count))


_CHAT_DRIFT_MARKERS = (
    "choose omni collective",
    "common benchmark score",
    "latest fused multimodal frontier checkpoint",
    "request matches its strongest local use case",
    "use v40_benchmax",
)


def _looks_like_model_selection_prompt(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return any(
        marker in lowered
        for marker in (
            "which model",
            "route to",
            "should handle",
        )
    )


def _looks_like_active_model_prompt(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return any(marker in lowered for marker in ("what model", "active model", "model version"))


def _looks_like_benchmark_prompt(prompt: str) -> bool:
    lowered = str(prompt or "").lower()
    return any(
        marker in lowered
        for marker in (
            "final answer",
            "gsm8k",
            "mmlu",
            "arc_challenge",
            "hellaswag",
            "boolq",
            "piqa",
            "benchmark",
        )
    )


def _v46_response_is_obvious_chat_drift(prompt: str, response: str) -> bool:
    lowered_prompt = str(prompt or "").lower()
    lowered_response = str(response or "").lower()
    if not lowered_response.strip():
        return True
    greeting_prompt = any(token in lowered_prompt for token in ("hello", "hi ", "hey", "greetings"))
    if greeting_prompt and not any(
        token in lowered_response
        for token in ("hello", "hi", "greetings", "active local model", "how can i help")
    ):
        return not _looks_like_benchmark_prompt(lowered_prompt)
    if "<thought>" in lowered_response or "finalizing synthesis" in lowered_response:
        return not _looks_like_benchmark_prompt(lowered_prompt)
    if re.search(r"\bthe answer is\s+-?\d+(?:\.\d+)?\b", lowered_response):
        return not _looks_like_benchmark_prompt(lowered_prompt)
    if "short-lived access tokens" in lowered_response and not any(
        token in lowered_prompt for token in ("auth", "login", "oauth", "security", "token", "jwt")
    ):
        return not _looks_like_benchmark_prompt(lowered_prompt)
    if lowered_response.startswith("recommended approach:") and not any(
        token in lowered_prompt for token in ("approach", "recommend", "how should", "security", "token", "auth")
    ):
        return not _looks_like_benchmark_prompt(lowered_prompt)
    if "choose omni collective" in lowered_response:
        return not _looks_like_model_selection_prompt(lowered_prompt)
    if "common benchmark score" in lowered_response:
        return not ("benchmark score" in lowered_prompt or _looks_like_model_selection_prompt(lowered_prompt))
    if any(marker in lowered_response for marker in _CHAT_DRIFT_MARKERS):
        return not _looks_like_model_selection_prompt(lowered_prompt)
    if "final answer:" in lowered_response and not _looks_like_benchmark_prompt(lowered_prompt):
        return True
    if lowered_response.strip() in {"yes", "no", "final answer: yes", "final answer: no"}:
        return not _looks_like_benchmark_prompt(lowered_prompt)
    return False


def _v46_chat_guard_response(prompt: str, response: str, record: ModelRecord) -> Tuple[str, bool]:
    if not _v46_response_is_obvious_chat_drift(prompt, response):
        return response, False

    lowered = str(prompt or "").lower()
    model_label = record.label or "Omni Collective V46"
    if any(token in lowered for token in ("hello", "hi ", "hey", "greetings")):
        return f"Hello. The active local model is {model_label}, running through the Supermix chat interface.", True
    if _looks_like_active_model_prompt(lowered) or _looks_like_model_selection_prompt(lowered):
        score = record.common_overall_exact
        score_text = f" with common benchmark score {score:.4f}" if score is not None else ""
        return f"The active local model is {model_label}{score_text}.", True
    if any(token in lowered for token in ("not making sense", "nonsense", "not normal", "off topic", "wrong response")):
        return (
            "The chat response drifted into a memorized response-bank entry. "
            "The correct fix is to add chat-drift repair examples, preserve benchmark replay, "
            "and retrain from the promoted v46 champion instead of accepting the off-topic answer."
        ), True
    if any(token in lowered for token in ("train", "training", "evolution", "evolve", "benchmark higher")):
        return (
            "The next improvement pass should train from the promoted v46 champion with two priorities: "
            "normal-chat drift repair and weak-suite benchmark replay. It should reject off-topic canned answers "
            "while keeping exact final-answer formatting for benchmarks."
        ), True
    return (
        "I do not have a reliable grounded response for that prompt yet. "
        "The previous candidate was off-topic, so I am rejecting it rather than returning a memorized answer."
    ), True


def _extract_labeled_section(text: str, label: str) -> str:
    cooked = str(text or "")
    if not cooked.strip():
        return ""
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)\s*(?=^\s*[A-Z_ ]+\s*:|\Z)"
    )
    match = pattern.search(cooked)
    if match:
        return " ".join(match.group(1).strip().split())
    return ""


def _parse_yes_no_section(text: str, label: str) -> Optional[bool]:
    section = _extract_labeled_section(text, label)
    if not section:
        return None
    lowered = section.strip().lower()
    if lowered.startswith(("yes", "true", "done", "complete")):
        return True
    if lowered.startswith(("no", "false", "not", "continue", "incomplete")):
        return False
    return None


def _normalize_score_0_1(value: float) -> float:
    cooked = float(value)
    if cooked > 1.0:
        cooked = cooked / (10.0 if cooked <= 10.0 else 100.0)
    return max(0.0, min(1.0, cooked))


def _parse_labeled_score_0_1(text: str, label: str) -> Optional[float]:
    section = _extract_labeled_section(text, label)
    if not section:
        return None
    fraction = re.search(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", section)
    if fraction:
        denominator = float(fraction.group(2))
        if denominator:
            return round(_normalize_score_0_1(float(fraction.group(1)) / denominator), 4)
    match = re.search(r"([-+]?\d+(?:\.\d+)?)\s*%?", section)
    if not match:
        return None
    value = float(match.group(1))
    if "%" in match.group(0):
        value /= 100.0
    return round(_normalize_score_0_1(value), 4)


def _first_labeled_score_0_1(text: str, labels: Sequence[str]) -> Optional[float]:
    for label in labels:
        value = _parse_labeled_score_0_1(text, label)
        if value is not None:
            return value
    return None


def _loop_score_threshold(raw_value: Any) -> float:
    if raw_value is None or raw_value == "":
        return LOOP_AGENT_DEFAULT_SCORE_THRESHOLD
    try:
        return max(0.55, min(0.98, _normalize_score_0_1(float(raw_value))))
    except (TypeError, ValueError):
        return LOOP_AGENT_DEFAULT_SCORE_THRESHOLD


def _loop_review_metrics(reviewer_text: str, review_complete: Optional[bool]) -> Dict[str, Any]:
    explicit_score = _first_labeled_score_0_1(
        reviewer_text,
        ("SCORE", "VERIFIER_SCORE", "COMPLETION_SCORE", "QUALITY_SCORE"),
    )
    progress = _first_labeled_score_0_1(
        reviewer_text,
        ("PROGRESS_SCORE", "COMPLETION_SCORE", "COMPLETENESS", "QUALITY_SCORE", "SCORE"),
    )
    confidence = _first_labeled_score_0_1(reviewer_text, ("CONFIDENCE", "READY_CONFIDENCE", "CERTAINTY"))
    risk = _first_labeled_score_0_1(reviewer_text, ("RISK_SCORE", "RISK", "ERROR_RISK"))

    if progress is None:
        progress = 1.0 if review_complete is True else 0.35 if review_complete is False else 0.5
    if confidence is None:
        confidence = 0.9 if review_complete is True else 0.45 if review_complete is False else 0.55
    if risk is None:
        risk = 0.05 if review_complete is True else 0.45 if review_complete is False else 0.3

    verifier_score = round((progress * 0.50) + (confidence * 0.30) + ((1.0 - risk) * 0.20), 4)
    review_score = round(explicit_score if explicit_score is not None else verifier_score, 4)
    evidence = (
        _extract_labeled_section(reviewer_text, "EVIDENCE")
        or _extract_labeled_section(reviewer_text, "COMPLETION_EVIDENCE")
        or _extract_labeled_section(reviewer_text, "REASON")
    )
    return {
        "progress_score": round(progress, 4),
        "confidence_score": round(confidence, 4),
        "risk_score": round(risk, 4),
        "review_score": review_score,
        "verifier_score": verifier_score,
        "completion_evidence": _trim_text(evidence, limit=220),
    }


def _safe_upload_name(filename: str) -> str:
    cooked = re.sub(r"[^A-Za-z0-9._-]+", "-", str(filename or "").strip()).strip(".-")
    return cooked[:96] or "upload.png"


DEFAULT_MODEL_STORE_REPO_ID = "Kai9987kai/supermix-model-zoo"
MODEL_STORE_CACHE_TTL_SECONDS = 300.0
LOOP_AGENT_DEFAULT_MAX_STEPS = 4
LOOP_AGENT_HARD_MAX_STEPS = 8
LOOP_AGENT_DEFAULT_SCORE_THRESHOLD = 0.88
MODEL_STORE_ARTIFACT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,180}\.zip$", re.IGNORECASE)
AUTO_AGENT_MODE_ORDER = ("off", "collective", "loop", "collective_loop")
AUTO_ROUTE_POLICY_ID = "auto-route-v2"
AUTO_ROUTE_POLICY_VERSION = "2.0.0"
AUTO_ROUTE_FEATURE_SCHEMA_VERSION = "route-context-v1"
AUTO_AGENT_POSITIVE_THRESHOLDS = {
    "collective": 1,
    "loop": 3,
    "collective_loop": 4,
}
AUTO_AGENT_ADAPTIVE_MIN_WEIGHTED_COUNT = 1.2
AUTO_AGENT_ADAPTIVE_QUALITY_FLOOR = 0.7
AUTO_AGENT_ADAPTIVE_QUALITY_DELTA = 0.08
AUTO_AGENT_ADAPTIVE_QUALITY_COST_DELTA = 0.08
AUTO_AGENT_SELECTION_THRESHOLDS = {
    "collective": 2,
    "loop": 4,
    "collective_loop": 5,
}


def _stamp_auto_route_policy(
    policy: Dict[str, Any],
    *,
    selected_agent_mode: str,
    action_mode: str,
    logging_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    support_candidates = (
        logging_support.get("candidates")
        if isinstance(logging_support, dict) and isinstance(logging_support.get("candidates"), list)
        else []
    )
    allowed_modes = [
        str(candidate.get("action"))
        for candidate in support_candidates
        if isinstance(candidate, dict) and str(candidate.get("action")) in AUTO_AGENT_MODE_ORDER
    ]
    if not allowed_modes:
        allowed_modes = [
            str(mode)
            for mode in list(policy.get("allowed_agent_modes") or [])
            if str(mode) in AUTO_AGENT_MODE_ORDER
        ]
    if selected_agent_mode not in allowed_modes:
        allowed_modes.append(selected_agent_mode)
    policy["policy_id"] = AUTO_ROUTE_POLICY_ID
    policy["policy_version"] = AUTO_ROUTE_POLICY_VERSION
    policy["feature_schema_version"] = AUTO_ROUTE_FEATURE_SCHEMA_VERSION
    policy["decision_type"] = "deterministic"
    policy["action_mode"] = action_mode
    policy["eligible_agent_modes"] = list(allowed_modes)
    policy["eligible_actions"] = list(allowed_modes)
    policy["selected_agent_mode"] = selected_agent_mode
    probabilities = {
        mode: 1.0 if mode == selected_agent_mode else 0.0
        for mode in AUTO_AGENT_MODE_ORDER
        if mode in allowed_modes
    }
    policy["action_probabilities"] = dict(probabilities)
    policy["post_filter_action_probabilities"] = dict(probabilities)
    policy["probability_stage"] = "post_filter"
    policy["logging_propensity"] = 1.0
    policy["selection_thresholds"] = dict(AUTO_AGENT_SELECTION_THRESHOLDS)
    policy["counterfactual_support"] = "none_deterministic_logging"
    if isinstance(logging_support, dict):
        policy["logging_support"] = dict(logging_support)
        policy["candidate_set_hash"] = logging_support.get("candidate_set_hash")
        policy["distribution_hash"] = logging_support.get("distribution_hash")
    policy["decision_context"] = {
        "action_mode": action_mode,
        "budget_profile": str(policy.get("budget_profile") or "balanced"),
        "score": int(policy.get("score") or 0),
        "allowed_agent_modes": list(allowed_modes),
    }
    return policy


AUTO_AGENT_UNCERTAINTY_SIGNAL_PATTERNS = (
    ("audit", re.compile(r"\baudit(?:ing)?\b", re.IGNORECASE)),
    ("verification", re.compile(r"\b(verif(?:y|ies|ied|ication)|validat(?:e|es|ed|ion))\b", re.IGNORECASE)),
    ("risk", re.compile(r"\b(risk|safety|high[- ]stakes|production)\b", re.IGNORECASE)),
    ("correctness", re.compile(r"\b(correctness|correct|accuracy|accurate)\b", re.IGNORECASE)),
    ("edge_cases", re.compile(r"\bedge[- ]cases?\b", re.IGNORECASE)),
    ("proof", re.compile(r"\b(proof|prove|formal)\b", re.IGNORECASE)),
    ("review", re.compile(r"\breview\b", re.IGNORECASE)),
    ("regression", re.compile(r"\bregression\b", re.IGNORECASE)),
    ("evidence", re.compile(r"\b(source|cite|citation|evidence)\b", re.IGNORECASE)),
)
AUTO_AGENT_BUDGET_PROFILES = {
    "fast": {
        "label": "Fast",
        "cost_preference": "low_latency",
        "score_bias": -1,
        "max_agent_mode": "collective",
    },
    "balanced": {
        "label": "Balanced",
        "cost_preference": "balanced",
        "score_bias": 0,
        "max_agent_mode": "collective_loop",
    },
    "deep": {
        "label": "Deep",
        "cost_preference": "quality",
        "score_bias": 1,
        "max_agent_mode": "collective_loop",
    },
    "max": {
        "label": "Max",
        "cost_preference": "frontier",
        "score_bias": 2,
        "max_agent_mode": "collective_loop",
    },
}


def _validate_model_store_file_name(file_name: str) -> str:
    cooked = str(file_name or "").strip()
    if not cooked:
        raise ValueError("file_name is required")
    if (
        cooked in {".", ".."}
        or "/" in cooked
        or "\\" in cooked
        or ":" in cooked
        or Path(cooked).name != cooked
        or not MODEL_STORE_ARTIFACT_RE.fullmatch(cooked)
    ):
        raise ValueError(f"Unsafe model store artifact name: {cooked!r}")
    return cooked


def _is_safe_model_store_manifest_item(item: Dict[str, Any]) -> bool:
    try:
        _validate_model_store_file_name(item.get("file_name") or "")
        return True
    except ValueError:
        return False


def _hf_dataset_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{quote(filename, safe='')}?download=true"


def _missing_zip_members(archive: zipfile.ZipFile, target: Path) -> List[str]:
    missing: List[str] = []
    for member in archive.infolist():
        if member.is_dir():
            continue
        candidate = target / member.filename
        if not candidate.exists():
            missing.append(member.filename)
    return missing


def _extract_zip_once(zip_path: Path, extraction_root: Path) -> Path:
    # If the artifact is already a raw model file (.pth), skip extraction
    if zip_path.suffix.lower() == ".pth":
        return zip_path.parent
    extraction_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{zip_path.name}|{zip_path.stat().st_size}|{zip_path.stat().st_mtime_ns}"
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:12]
    target = extraction_root / f"{_safe_slug(zip_path.stem)}-{digest}"
    marker = target / ".extract_complete.json"
    expected_meta = {
        "zip_name": zip_path.name,
        "zip_size": zip_path.stat().st_size,
        "zip_mtime_ns": zip_path.stat().st_mtime_ns,
    }
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if payload == expected_meta:
            return target
    if target.exists():
        for child in sorted(target.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            target.rmdir()
        except OSError:
            pass
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(target)
        missing = _missing_zip_members(archive, target)
        if missing:
            for member in missing:
                archive.extract(member, target)
            missing = _missing_zip_members(archive, target)
        if missing:
            raise FileNotFoundError(
                f"Extraction of {zip_path.name} is incomplete; missing {len(missing)} file(s), including {missing[:3]}"
            )
    marker.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return target


def _find_matching_file(root: Path, preferred_names: Tuple[str, ...], suffix: str) -> Optional[Path]:
    if preferred_names:
        for name in preferred_names:
            candidate = root / name
            if candidate.exists():
                return candidate
        for name in preferred_names:
            matches = list(root.rglob(Path(name).name))
            if matches:
                return sorted(matches)[0]
    matches = sorted(root.rglob(f"*{suffix}"))
    if suffix.lower() == ".json":
        meta_matches = [
            path
            for path in matches
            if path.name.lower().endswith("_meta.json") and not path.name.startswith(".")
        ]
        if meta_matches:
            return meta_matches[0]
        matches = [
            path
            for path in matches
            if not path.name.startswith(".") and "summary" not in path.name.lower()
        ]
    return matches[0] if matches else None


def _find_adapter_dir(root: Path, markers: Tuple[str, ...]) -> Path:
    for marker in markers:
        candidate = root / marker
        if candidate.exists():
            return candidate.parent.resolve()
    matches = sorted(root.rglob("adapter_config.json"))
    for match in matches:
        if (match.parent / "adapter_model.safetensors").exists():
            return match.parent.resolve()
    raise FileNotFoundError(f"Could not find a Qwen adapter directory under {root}")


def _compose_text_prompt(prompt: str, settings: Dict[str, Any]) -> str:
    blocks: List[str] = []
    if str(settings.get("tool_instruction") or "").strip():
        blocks.append(str(settings.get("tool_instruction")).strip())
    if str(settings.get("memory_context") or "").strip():
        blocks.append(str(settings.get("memory_context")).strip())
    if str(settings.get("tool_context") or "").strip():
        blocks.append("Tool results:\n" + str(settings.get("tool_context")).strip())
    if str(settings.get("consultation_context") or "").strip():
        blocks.append("Cross-model consultation:\n" + str(settings.get("consultation_context")).strip())
    if not blocks:
        return str(prompt)
    blocks.append("Current user request:\n" + str(prompt).strip())
    blocks.append("Answer the current user request directly and use the context above when it is relevant.")
    return "\n\n".join(blocks)


def _compose_image_prompt(prompt: str, settings: Dict[str, Any]) -> str:
    notes: List[str] = []
    if str(settings.get("memory_context") or "").strip():
        notes.append(str(settings.get("memory_context")).strip())
    if str(settings.get("consultation_context") or "").strip():
        notes.append("Prompt planning notes:\n" + str(settings.get("consultation_context")).strip())
    if not notes:
        return str(prompt)
    return "\n\n".join([str(prompt).strip(), *notes])


class BaseBackend:
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        self.record = record
        self.extracted_dir = extracted_dir
        self.generated_dir = generated_dir
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        raise NotImplementedError

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        raise RuntimeError(f"{self.record.label} does not support image generation")

    def clear(self, session_id: str) -> None:
        return None

    def unload(self) -> None:
        return None


class ChampionChatBackend(BaseBackend):
    def __init__(
        self,
        record: ModelRecord,
        extracted_dir: Path,
        generated_dir: Path,
        device: Any,
        device_info: Dict[str, Any],
    ) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = chat_web_app.Engine(
            device,
            device_info,
            {
                "model_size": "auto",
                "max_turns": 2,
                "top_labels": 3,
                "pool_mode": "all",
                "response_temperature": 0.08,
                "temperature": 0.0,
                "style_mode": "auto",
                "creativity": 0.25,
            },
        )
        self.engine.load(str(self.weights_path), str(self.meta_path))

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "champion_chat",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_text_prompt(prompt, settings)
        payload = self.engine.chat(
            session_id=session_id,
            user_text=effective_prompt,
            style_mode=str(settings.get("style_mode") or "auto"),
            response_temperature=float(settings.get("response_temperature") or 0.08),
            show_top_responses=int(settings.get("show_top_responses") or 0),
            reasoning_cycles=settings.get("reasoning_cycles"),
            adaptive_compute=settings.get("adaptive_compute"),
            adaptive_exit_tol=settings.get("adaptive_exit_tol"),
            adaptive_exit_entropy=settings.get("adaptive_exit_entropy"),
            prediction_stability_patience=settings.get("prediction_stability_patience"),
            prediction_stability_tol=settings.get("prediction_stability_tol"),
            prediction_stability_margin=settings.get("prediction_stability_margin"),
            prediction_stability_rank_depth=settings.get("prediction_stability_rank_depth"),
            auto_compute=settings.get("auto_compute"),
            interaction_enabled=bool(
                settings.get("interaction_intelligence", True)
            ),
            interaction_plan=(
                settings.get("_interaction_plan")
                if isinstance(settings.get("_interaction_plan"), Mapping)
                else None
            ),
            interaction_user_text=str(
                settings.get("_interaction_user_text") or prompt
            ),
            prompt_profile=(
                settings.get("_prompt_profile")
                if isinstance(settings.get("_prompt_profile"), Mapping)
                else None
            ),
            grounding_enabled=False,
            conversation_enabled=bool(settings.get("_conversation_enabled", True)),
        )
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=str(payload.get("response") or ""),
            timing=dict(payload.get("timing_ms") or {}),
            compute=dict(payload.get("compute") or {}),
            prompt_used=effective_prompt,
            # The engine derives its own state from the history it keeps for
            # this session. It was already doing so; the diagnostics were simply
            # dropped on the floor here, so no Studio surface could report it.
            conversation=(
                dict(payload.get("conversation") or {})
                if isinstance(payload.get("conversation"), Mapping)
                else None
            ),
        )

    def clear(self, session_id: str) -> None:
        self.engine.clear(session_id)

    def unload(self) -> None:
        if hasattr(self.engine, "model"):
            self.engine.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ImageWrapperBackend(ChampionChatBackend):
    def __init__(
        self,
        record: ModelRecord,
        extracted_dir: Path,
        generated_dir: Path,
        device: Any,
        device_info: Dict[str, Any],
    ) -> None:
        super().__init__(record, extracted_dir, generated_dir, device, device_info)
        self.image_engine = ImageVariantEngine(
            text_engine=self.engine,
            output_dir=self.generated_dir / self.record.key,
            default_image_model=DEFAULT_IMAGE_MODEL,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )

    def status(self) -> Dict[str, Any]:
        payload = super().status()
        payload["backend"] = "image_wrapper"
        payload["image_status"] = self.image_engine.status()
        return payload

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_image_prompt(prompt, settings)
        result = self.image_engine.generate_image(
            prompt=effective_prompt,
            image_model=str(settings.get("image_model") or DEFAULT_IMAGE_MODEL),
            negative_prompt=str(settings.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT),
            style=str(settings.get("image_style") or "auto"),
            width=int(settings.get("image_width") or 512),
            height=int(settings.get("image_height") or 512),
            steps=int(settings.get("image_steps") or 2),
            seed=None if settings.get("image_seed") in (None, "") else int(settings.get("image_seed")),
            guidance_scale=float(settings.get("guidance_scale") or 0.0),
            use_text_refiner=bool(settings.get("use_text_refiner", True)),
        )
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            timing={
                "total_ms": result.get("timing_ms"),
                "model_calls": max(1, int(result.get("model_calls") or 1)),
                "refiner_model_calls": max(0, int(result.get("refiner_model_calls") or 0)),
            },
            image_url=str(result.get("image_url") or ""),
            output_path=str(result.get("output_path") or ""),
            prompt_used=str(result.get("prompt_used") or effective_prompt),
            refined_prompt=str(result.get("refined_prompt") or ""),
        )


class QwenBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        import qwen_chat_web_app  # lazy import so source runtime can start without the Qwen stack installed

        self._qwen = qwen_chat_web_app
        self.adapter_dir = _find_adapter_dir(extracted_dir, record.adapter_markers)
        self.device = self._qwen.resolve_device("auto")
        self.base_model = self._qwen.resolve_base_model_path("", self.adapter_dir)
        self.engine = self._qwen.load_engine(
            base_model=self.base_model,
            adapter_dir=self.adapter_dir,
            device=self.device,
        )

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "qwen_adapter",
            "record": self.record.to_dict(),
            "adapter_dir": str(self.adapter_dir),
            "base_model": str(self.base_model),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        style_mode = str(settings.get("style_mode") or "auto").strip().lower()
        # "auto" is the absence of a choice, so it stays unresolved here and a
        # style the user asked for earlier in the session is allowed to fill it
        # in. Every other style mode is an explicit choice and maps as before.
        preset = {
            "concise": "direct",
            "creative": "creative",
            "analyst": "reasoning",
            "coding": "coding",
        }.get(style_mode, "auto" if style_mode in ("", "auto") else "balanced")
        effective_prompt = _compose_text_prompt(prompt, settings)
        max_new_tokens = settings.get("max_new_tokens")
        temperature = settings.get("temperature")
        top_p = settings.get("top_p")
        payload = self.engine.chat(
            session_id=session_id,
            user_text=effective_prompt,
            max_new_tokens=(
                int(max_new_tokens) if max_new_tokens not in (None, "") else None
            ),
            temperature=(
                float(temperature) if temperature not in (None, "") else None
            ),
            top_p=float(top_p) if top_p not in (None, "") else None,
            preset=preset,
            system_hint=str(settings.get("system_hint") or ""),
            interaction_enabled=bool(
                settings.get("interaction_intelligence", True)
            ),
            interaction_plan=(
                settings.get("_interaction_plan")
                if isinstance(settings.get("_interaction_plan"), Mapping)
                else None
            ),
            interaction_user_text=str(
                settings.get("_interaction_user_text") or prompt
            ),
            prompt_profile=(
                settings.get("_prompt_profile")
                if isinstance(settings.get("_prompt_profile"), Mapping)
                else None
            ),
            grounding_enabled=False,
            conversation_enabled=bool(settings.get("_conversation_enabled", True)),
            conversation_state=(
                settings.get("_conversation_state")
                if isinstance(settings.get("_conversation_state"), Mapping)
                else None
            ),
        )
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=str(payload.get("response") or ""),
            timing=dict(payload.get("timing") or {}),
            prompt_used=effective_prompt,
            conversation=(
                dict(payload.get("conversation") or {})
                if isinstance(payload.get("conversation"), Mapping)
                else None
            ),
        )

    def clear(self, session_id: str) -> None:
        self.engine.clear(session_id)

    def unload(self) -> None:
        if hasattr(self.engine, "model"):
            self.engine.model = None
        if hasattr(self.engine, "tokenizer"):
            self.engine.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class NativeImageBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path, device: Any) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.device = torch.device("cuda" if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu")
        self.model, self.save_fn = self._load_model()

    def _load_model(self) -> Tuple[Any, Any]:
        image_size = int(self.meta.get("image_size") or 64)
        if self.record.key == "v36_native":
            model = ChampionNetFrontierCollectiveNativeImage(image_size=image_size).to(self.device).eval()
            state_dict = safe_load_state_dict(str(self.weights_path))
            model.load_state_dict(state_dict, strict=False)
            return model, save_prompt_image_v36
        if self.record.key == "v37_native_lite":
            model = ChampionNetUltraExpertNativeImageLite(image_size=image_size).to(self.device).eval()
            model.load_state_dict(safe_load_state_dict(str(self.weights_path)), strict=True)
            return model, save_prompt_image_v37
        model = ChampionNetUltraExpertNativeImageExtraLite(image_size=image_size).to(self.device).eval()
        model.load_state_dict(safe_load_state_dict(str(self.weights_path)), strict=True)
        return model, save_prompt_image_v38

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "native_image",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "device": str(self.device),
            "image_size": int(self.meta.get("image_size") or 64),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        return self.generate_image(session_id=session_id, prompt=prompt, settings=settings)

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        effective_prompt = _compose_image_prompt(prompt, settings)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.generated_dir / self.record.key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{_safe_slug(effective_prompt)[:40]}.png"
        feature_mode = str(self.meta.get("feature_mode") or "context_mix_v4")
        started = time.perf_counter()
        self.save_fn(self.model, str(effective_prompt), str(out_path), feature_mode=feature_mode, device=self.device)
        total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            timing={"total_ms": total_ms},
            image_url=f"/generated/{self.record.key}/{out_path.name}",
            output_path=str(out_path),
            prompt_used=str(effective_prompt),
            refined_prompt=str(settings.get("consultation_context") or ""),
        )

    def unload(self) -> None:
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DCGANImageBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        spec = DCGAN_SPECS.get(record.key)
        if spec is None:
            raise KeyError(f"Unsupported DCGAN model key: {record.key}")
        self.spec = spec
        self.weights_path = _find_generator_weights(extracted_dir, record.preferred_weights or spec.preferred_weights)
        self.engine = DCGANImageEngine(key=record.key, weights_path=self.weights_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "dcgan_image",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "runtime": self.engine.status(),
        }

    def generate_image(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        del session_id
        effective_prompt = _compose_image_prompt(prompt, settings)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.generated_dir / self.record.key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stamp}_{_safe_slug(effective_prompt)[:40]}.png"
        sample_count = 16
        lowered = effective_prompt.lower()
        if "single" in lowered or "one sample" in lowered:
            sample_count = 1
        elif "large grid" in lowered or "many samples" in lowered:
            sample_count = 25
        started = time.perf_counter()
        self.engine.render_to_path(prompt=effective_prompt, output_path=out_path, sample_count=sample_count)
        total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return ChatResult(
            kind="image",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=(
                f"{self.record.label} is an unconditional GAN, so the prompt only seeds the sample grid rather than controlling content directly."
            ),
            timing={"total_ms": total_ms},
            image_url=f"/generated/{self.record.key}/{out_path.name}",
            output_path=str(out_path),
            prompt_used=str(effective_prompt),
            refined_prompt=str(settings.get("consultation_context") or ""),
        )


class MathEquationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing math weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = MathEquationEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "math_equation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        solved = self.engine.solve(prompt)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=format_math_response(solved),
            timing={},
            prompt_used=str(prompt),
        )


class ProteinFoldingBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing protein-folding weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ProteinFoldingEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "protein_folding",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_protein_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class MatterGenGenerationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing MatterGen weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = MatterGenMicroEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "mattergen_generation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_mattergen_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class ThreeDGenerationBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing 3D-generation weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ThreeDGenerationEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "three_d_generation",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        prediction = self.engine.predict(prompt)
        response = format_3d_generation_response(self.engine.answer(prompt), prediction)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class ImageRecognitionBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing image-recognition weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = ScienceImageRecognitionEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "image_recognition",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        if not image_path:
            response = "Upload an image first, then I can identify the science concept and explain the visual clues."
        else:
            response = self.engine.answer(prompt, image_path=image_path)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class OmniCollectiveBackend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngine(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": self.engine.status(),
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        response = self.engine.answer(prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=str(prompt),
        )


class OmniCollectiveV3Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV3(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v3",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV4Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV4(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v4",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV5Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV5(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v5",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV6Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV6(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v6",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV7Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV7(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v7",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV8Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        self.engine = OmniCollectiveEngineV8(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v8",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV41Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        if OmniCollectiveEngineV41 is None:
            raise ImportError("OmniCollectiveEngineV41 is not available. Please ensure omni_collective_v41_model.py exists.")
        self.engine = OmniCollectiveEngineV41(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v41",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV42Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        if OmniCollectiveEngineV42 is None:
            raise ImportError("OmniCollectiveEngineV42 is not available. Please ensure omni_collective_v42_model.py exists.")
        self.engine = OmniCollectiveEngineV42(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v42",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "deliberation_passes": int(self.engine.deliberation_passes),
                "minimum_passes": int(self.engine.minimum_passes),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        response = self.engine.answer(effective_prompt, image_path=image_path or None)
        return ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing={},
            prompt_used=effective_prompt,
        )


class OmniCollectiveV46Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        if OmniCollectiveEngineV46 is None:
            raise ImportError("OmniCollectiveEngineV46 is not available. Please ensure omni_collective_v46_model.py exists.")
        self.engine = OmniCollectiveEngineV46(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v46",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        response, guard_repaired = _v46_chat_guard_response(effective_prompt, pred.response_text, self.record)
        timing["chat_guard_repaired"] = guard_repaired
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=response,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "grpo_group_size": pred.grpo_group_size,
            "raw_response": pred.response_text if guard_repaired else "",
        }
        return res


class OmniCollectiveV47Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        if OmniCollectiveEnginev47 is None:
            raise ImportError("OmniCollectiveEnginev47 is not available. Please ensure omni_collective_v47_model.py exists.")
        self.engine = OmniCollectiveEnginev47(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v47",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=pred.response_text,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "grpo_group_size": pred.grpo_group_size,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
        }
        return res


class OmniCollectiveV48Backend(BaseBackend):
    def __init__(self, record: ModelRecord, extracted_dir: Path, generated_dir: Path) -> None:
        super().__init__(record, extracted_dir, generated_dir)
        weights_path = _find_matching_file(extracted_dir, record.preferred_weights, ".pth")
        meta_path = _find_matching_file(extracted_dir, record.preferred_meta, ".json")
        if weights_path is None or meta_path is None:
            raise FileNotFoundError(f"Missing omnibus weights/meta for {record.label} in {extracted_dir}")
        self.weights_path = weights_path.resolve()
        self.meta_path = meta_path.resolve()
        if OmniCollectiveEnginev48 is None:
            raise ImportError("OmniCollectiveEnginev48 is not available. Please ensure omni_collective_v48_model.py exists.")
        self.engine = OmniCollectiveEnginev48(weights_path=self.weights_path, meta_path=self.meta_path)

    def status(self) -> Dict[str, Any]:
        return {
            "backend": "omni_collective_v48",
            "record": self.record.to_dict(),
            "weights_path": str(self.weights_path),
            "meta_path": str(self.meta_path),
            "runtime": {
                "device": str(self.engine.device),
                "image_size": int(self.engine.image_size),
                "vocab_size": len(self.engine.vocab),
                "response_count": len(self.engine.responses),
                "grounding_threshold": float(self.engine.grounding_threshold),
            },
        }

    def chat(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> ChatResult:
        image_path = str(settings.get("uploaded_image_path") or "").strip()
        effective_prompt = _compose_text_prompt(prompt, settings)
        pred = self.engine.predict(effective_prompt, image_path=image_path or None)
        timing = {
            "reasoning_passes": pred.reasoning_passes,
            "planned_budget": pred.planned_budget,
            "difficulty_estimate": pred.difficulty_estimate,
            "mixture_of_depths_skipped": pred.mixture_of_depths_skipped,
            "graph_synthesis_applied": pred.graph_synthesis_applied,
            "continuous_latent_active": pred.continuous_latent_active,
            "hierarchical_routing_applied": pred.hierarchical_routing_applied,
            "agat_node_count": pred.agat_node_count,
        }
        res = ChatResult(
            kind="text",
            model_key=self.record.key,
            model_label=self.record.label,
            route_reason=str(settings.get("route_reason") or ""),
            response=pred.response_text,
            timing=timing,
            prompt_used=effective_prompt,
        )
        res.agent_trace = {
            "reasoning_mode": pred.reasoning_mode,
            "speculative_accepted": pred.speculative_accepted,
            "adversarial_verified": pred.adversarial_verified,
            "dpo_alignment_score": pred.dpo_alignment_score,
        }
        return res


class UnifiedModelManager:
    def __init__(
        self,
        records: Tuple[ModelRecord, ...],
        extraction_root: Path,
        generated_dir: Path,
        device_preference: str = "cuda,npu,xpu,cpu,dml,mps",
        models_dir: Path = DEFAULT_MODELS_DIR,
        common_summary_path: Path = DEFAULT_COMMON_SUMMARY,
        model_store_repo_id: str = DEFAULT_MODEL_STORE_REPO_ID,
        backend_cache_size: int = 1,
    ) -> None:
        configure_torch_runtime(
            torch_num_threads=0,
            torch_interop_threads=0,
            allow_tf32=True,
            matmul_precision="high",
        )
        device, device_info = resolve_device("auto", preference=device_preference)
        self.records = list(records)
        self.record_map = {record.key: record for record in self.records}
        self.models_dir = Path(models_dir).resolve()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.common_summary_path = Path(common_summary_path).resolve()
        self.model_store_repo_id = str(model_store_repo_id or DEFAULT_MODEL_STORE_REPO_ID).strip() or DEFAULT_MODEL_STORE_REPO_ID
        self.extraction_root = extraction_root.resolve()
        self.generated_dir = generated_dir.resolve()
        self.uploads_dir = self.extraction_root.parent / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir = self.extraction_root.parent / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.device_info = device_info
        self.selected_model_key = "auto"
        self.last_route_reason = ""
        self._backend_cache_size = max(1, int(backend_cache_size))
        self._backend: Optional[BaseBackend] = None
        self._backend_key = ""
        self._backend_cache: Dict[str, BaseBackend] = {}
        self._backend_lru: List[str] = []
        self._backend_init_failures: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._model_store_manifest_cache: Optional[Dict[str, Any]] = None
        self._model_store_manifest_ts = 0.0
        self._model_store_jobs: Dict[str, Dict[str, Any]] = {}
        memory_dir = self.extraction_root.parent / "memory"
        self.memory_store = ConversationMemoryStore(memory_dir)
        self.route_policy_ledger = RoutePolicyLedger(memory_dir / "route-policy-ledger.sqlite3")
        self.route_shadow_registry_path = memory_dir / "route-policy-shadow-registry.sqlite3"
        self._route_shadow_registry_cache_signature: Optional[
            Tuple[Tuple[Any, ...], ...]
        ] = None
        self._route_shadow_registry_cache_snapshot: Optional[Dict[str, Any]] = None
        self._route_execution = threading.local()
        self.web_search = WebSearchTool()
        self.cmd_open = CmdOpenTool()

    def _build_backend(self, record: ModelRecord) -> BaseBackend:
        extracted_dir = _extract_zip_once(record.zip_path, self.extraction_root)
        if record.kind == "champion_chat":
            return ChampionChatBackend(record, extracted_dir, self.generated_dir, self.device, self.device_info)
        if record.kind == "image_wrapper":
            return ImageWrapperBackend(record, extracted_dir, self.generated_dir, self.device, self.device_info)
        if record.kind == "native_image":
            return NativeImageBackend(record, extracted_dir, self.generated_dir, self.device)
        if record.kind == "dcgan_image":
            return DCGANImageBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "math_equation":
            return MathEquationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "protein_folding":
            return ProteinFoldingBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "mattergen_generation":
            return MatterGenGenerationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "three_d_generation":
            return ThreeDGenerationBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "image_recognition":
            return ImageRecognitionBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective":
            return OmniCollectiveBackend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v3":
            return OmniCollectiveV3Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v4":
            return OmniCollectiveV4Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v5":
            return OmniCollectiveV5Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v6":
            return OmniCollectiveV6Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v7":
            return OmniCollectiveV7Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v8":
            return OmniCollectiveV8Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v42":
            return OmniCollectiveV42Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v46":
            return OmniCollectiveV46Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v47":
            return OmniCollectiveV47Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v48":
            return OmniCollectiveV48Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "omni_collective_v41":
            return OmniCollectiveV41Backend(record, extracted_dir, self.generated_dir)
        if record.kind == "qwen_adapter":
            return QwenBackend(record, extracted_dir, self.generated_dir)
        raise RuntimeError(f"Unsupported model kind: {record.kind}")

    def _fallback_model_keys(self, failed_record: ModelRecord) -> List[str]:
        ordered: List[str] = []
        if failed_record.supports_image and not failed_record.supports_chat:
            ordered.extend(("v36_native", "v37_native_lite", "v38_native_xlite", "v38_native_xlite_fp16"))
        elif failed_record.supports_chat and failed_record.supports_vision:
            ordered.extend((
                "omni_collective_v47",
                "omni_collective_v46",
                "omni_collective_v42",
                "omni_collective_v41",
                "v40_benchmax",
                "omni_collective_v8",
                "omni_collective_v7",
                "omni_collective_v6",
                "omni_collective_v5",
                "omni_collective_v4",
                "omni_collective_v3",
                "omni_collective_v2",
                "omni_collective_v1",
                "v33_final",
                "v35_final",
                "v34_final",
                "qwen_v28",
                "v31_final",
                "v30_lite",
            ))
        else:
            ordered.extend((
                "omni_collective_v47",
                "omni_collective_v46",
                "omni_collective_v42",
                "omni_collective_v41",
                "v40_benchmax",
                "v33_final",
                "qwen_v28",
                "v35_final",
                "v34_final",
                "v31_final",
                "v30_lite",
            ))
        seen = {failed_record.key}
        keys: List[str] = []
        for key in ordered:
            if key in seen or key not in self.record_map:
                continue
            seen.add(key)
            keys.append(key)
        for record in self.records:
            if record.key in seen:
                continue
            if failed_record.supports_chat and not record.supports_chat:
                continue
            if failed_record.supports_vision and not record.supports_vision:
                continue
            if failed_record.supports_image and not record.supports_image and not record.supports_chat:
                continue
            seen.add(record.key)
            keys.append(record.key)
        return keys

    def _ensure_backend_direct_locked(self, record: ModelRecord) -> Tuple[ModelRecord, BaseBackend]:
        if self._backend_cache_size <= 1:
            if self._backend is not None and self._backend_key == record.key:
                return record, self._backend
            if self._backend is not None:
                self._backend.unload()
            self._backend = self._build_backend(record)
            self._backend_key = record.key
            self._backend_init_failures.pop(record.key, None)
            return record, self._backend

        cached = self._backend_cache.get(record.key)
        if cached is not None:
            if record.key in self._backend_lru:
                self._backend_lru.remove(record.key)
            self._backend_lru.append(record.key)
            self._backend = cached
            self._backend_key = record.key
            return record, cached

        backend = self._build_backend(record)
        self._backend_cache[record.key] = backend
        if record.key in self._backend_lru:
            self._backend_lru.remove(record.key)
        self._backend_lru.append(record.key)
        while len(self._backend_lru) > self._backend_cache_size:
            evict_key = self._backend_lru.pop(0)
            evicted = self._backend_cache.pop(evict_key, None)
            if evicted is not None:
                evicted.unload()
            if self._backend_key == evict_key:
                self._backend = None
                self._backend_key = ""

        self._backend = backend
        self._backend_key = record.key
        self._backend_init_failures.pop(record.key, None)
        return record, backend

    def ensure_backend(self, model_key: str) -> Tuple[ModelRecord, BaseBackend]:
        with self._lock:
            requested = self.record_map[model_key]
            candidate_keys = [requested.key]
            if requested.key in self._backend_init_failures:
                candidate_keys = []
            candidate_keys.extend(self._fallback_model_keys(requested))
            attempts: List[str] = []

            for key in candidate_keys:
                record = self.record_map.get(key)
                if record is None:
                    continue
                try:
                    return self._ensure_backend_direct_locked(record)
                except Exception as exc:
                    cooked = _trim_text(str(exc), limit=220)
                    self._backend_init_failures[record.key] = cooked
                    attempts.append(f"{record.key}: {cooked}")
                    logging.exception("Failed to initialize backend for %s", record.key)

            summary = "; ".join(attempts[:3]) or f"{requested.key}: no fallback candidates were available"
            raise RuntimeError(f"Failed to initialize a usable backend for {requested.label}. {summary}")

    def _refresh_records_locked(self) -> None:
        refreshed = discover_model_records(
            models_dir=self.models_dir,
            common_summary_path=self.common_summary_path,
        )
        self.records = list(refreshed)
        self.record_map = {record.key: record for record in self.records}
        self._backend_init_failures = {
            key: value for key, value in self._backend_init_failures.items() if key in self.record_map
        }
        valid_keys = set(self.record_map)
        for key in list(self._backend_cache):
            if key not in valid_keys:
                backend = self._backend_cache.pop(key, None)
                if backend is not None:
                    backend.unload()
                if key in self._backend_lru:
                    self._backend_lru.remove(key)
        if self.selected_model_key != "auto" and self.selected_model_key not in self.record_map:
            self.selected_model_key = "auto"
        if self._backend_key and self._backend_key not in self.record_map:
            if self._backend is not None:
                self._backend.unload()
            self._backend = None
            self._backend_key = ""

    def _fetch_model_store_manifest_locked(self, force_refresh: bool = False) -> Dict[str, Any]:
        now = time.time()
        if (
            not force_refresh
            and self._model_store_manifest_cache is not None
            and (now - self._model_store_manifest_ts) < MODEL_STORE_CACHE_TTL_SECONDS
        ):
            return dict(self._model_store_manifest_cache)
        url = _hf_dataset_file_url(self.model_store_repo_id, "manifest.json")
        with urlopen(url, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self._model_store_manifest_cache = payload
        self._model_store_manifest_ts = now
        return dict(payload)

    def model_store_catalog(self, force_refresh: bool = False) -> Dict[str, Any]:
        with self._lock:
            manifest = self._fetch_model_store_manifest_locked(force_refresh=force_refresh)
            installed_names = {record.zip_path.name for record in self.records}
            rows: List[Dict[str, Any]] = []
            for item in manifest.get("models") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    file_name = _validate_model_store_file_name(item.get("file_name") or "")
                except ValueError:
                    continue
                details = describe_model_artifact_name(file_name)
                local_path = self.models_dir / file_name
                installed = local_path.exists() or file_name in installed_names
                rows.append(
                    {
                        "file_name": file_name,
                        "size_bytes": int(item.get("size_bytes") or 0),
                        "size_mb": float(item.get("size_mb") or 0.0),
                        "family": str(item.get("family") or details.get("family") or "other"),
                        "known": bool(details.get("known")),
                        "model_key": str(details.get("key") or ""),
                        "label": str(details.get("label") or Path(file_name).stem),
                        "kind": str(details.get("kind") or ""),
                        "capabilities": list(details.get("capabilities") or []),
                        "note": str(details.get("note") or ""),
                        "benchmark_hint": str(details.get("benchmark_hint") or ""),
                        "download_url": _hf_dataset_file_url(self.model_store_repo_id, file_name),
                        "installed": installed,
                        "local_path": str(local_path.resolve()) if local_path.exists() else "",
                        "selectable": bool(details.get("known")) and str(details.get("key") or "") in self.record_map,
                    }
                )
            rows.sort(key=lambda item: (not item["installed"], item["label"].lower(), item["file_name"].lower()))
            return {
                "repo_id": self.model_store_repo_id,
                "model_count": len(rows),
                "models": rows,
            }

    def model_store_jobs(self) -> Dict[str, Any]:
        with self._lock:
            jobs = sorted(
                (dict(job) for job in self._model_store_jobs.values()),
                key=lambda item: str(item.get("started_at") or ""),
                reverse=True,
            )
            return {"jobs": jobs}

    def _set_model_store_job(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            payload = dict(self._model_store_jobs.get(job_id) or {})
            payload.update(updates)
            self._model_store_jobs[job_id] = payload

    def _install_model_store_worker(self, job_id: str, file_name: str, expected_size: int) -> None:
        file_name = _validate_model_store_file_name(file_name)
        target = self.models_dir / file_name
        temp_target = self.models_dir / f"{file_name}.{job_id}.part"
        try:
            self._set_model_store_job(job_id, status="downloading")
            url = _hf_dataset_file_url(self.model_store_repo_id, file_name)
            downloaded = 0
            with urlopen(url, timeout=60) as response:
                header_size = int(response.headers.get("Content-Length") or 0)
                total_bytes = expected_size or header_size
                self._set_model_store_job(job_id, total_bytes=total_bytes)
                with temp_target.open("wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        self._set_model_store_job(job_id, downloaded_bytes=downloaded)
            if target.exists():
                target.unlink()
            temp_target.replace(target)
            with self._lock:
                self._refresh_records_locked()
            self._set_model_store_job(
                job_id,
                status="completed",
                downloaded_bytes=target.stat().st_size,
                total_bytes=target.stat().st_size,
                local_path=str(target.resolve()),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                selectable=bool(describe_model_artifact_name(file_name).get("key") in self.record_map),
            )
        except Exception as exc:
            try:
                temp_target.unlink(missing_ok=True)
            except Exception:
                pass
            self._set_model_store_job(
                job_id,
                status="error",
                error=str(exc),
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )

    def install_model_store_artifact(self, file_name: str) -> Dict[str, Any]:
        cooked = _validate_model_store_file_name(file_name)
        with self._lock:
            manifest = self._fetch_model_store_manifest_locked(force_refresh=False)
            manifest_rows = {
                _validate_model_store_file_name(item.get("file_name") or ""): item
                for item in (manifest.get("models") or [])
                if isinstance(item, dict)
                and _is_safe_model_store_manifest_item(item)
            }
            if cooked not in manifest_rows:
                raise FileNotFoundError(f"{cooked} is not present in {self.model_store_repo_id}")
            for job in self._model_store_jobs.values():
                if job.get("file_name") == cooked and job.get("status") in {"queued", "downloading"}:
                    return dict(job)
            target = self.models_dir / cooked
            expected_size = int(manifest_rows[cooked].get("size_bytes") or 0)
            if target.exists() and (expected_size <= 0 or target.stat().st_size == expected_size):
                self._refresh_records_locked()
                payload = {
                    "job_id": f"already-{int(time.time())}",
                    "file_name": cooked,
                    "status": "completed",
                    "downloaded_bytes": target.stat().st_size,
                    "total_bytes": target.stat().st_size,
                    "local_path": str(target.resolve()),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                self._model_store_jobs[payload["job_id"]] = payload
                return payload
            job_id = f"store-{int(time.time() * 1000)}"
            payload = {
                "job_id": job_id,
                "file_name": cooked,
                "status": "queued",
                "downloaded_bytes": 0,
                "total_bytes": expected_size,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "local_path": "",
                "error": "",
            }
            self._model_store_jobs[job_id] = payload
            worker = threading.Thread(
                target=self._install_model_store_worker,
                args=(job_id, cooked, expected_size),
                daemon=True,
                name=f"model-store-{job_id}",
            )
            worker.start()
            return dict(payload)

    def _session_scope(self, session_id: str, record_key: str, purpose: str) -> str:
        return f"{session_id}::{purpose}::{record_key}"

    def _default_text_record(self) -> ModelRecord:
        for key in ("omni_collective_v46", "omni_collective_v42", "omni_collective_v41", "v40_benchmax", "omni_collective_v47", "omni_collective_v8", "omni_collective_v7", "omni_collective_v6", "omni_collective_v5", "omni_collective_v4", "omni_collective_v3", "v33_final", "omni_collective_v2", "v35_final", "v34_final", "qwen_v28", "v31_final", "v30_lite"):
            if key in self.record_map and self.record_map[key].supports_chat:
                return self.record_map[key]
        for record in self.records:
            if record.supports_chat:
                return record
        raise RuntimeError("No text-capable local models were discovered.")

    def _collective_consultants(
        self,
        settings: Optional[Dict[str, Any]] = None,
        chosen_record: Optional[ModelRecord] = None,
    ) -> List[ModelRecord]:
        consultants = [record for record in self.records if record.supports_chat]
        if not consultants:
            return []

        settings = dict(settings or {})
        raw_keys = settings.get("collective_consultant_keys")
        preferred_keys: List[str] = []
        if isinstance(raw_keys, str):
            preferred_keys = [part.strip() for part in raw_keys.split(",") if part.strip()]
        elif isinstance(raw_keys, (list, tuple)):
            preferred_keys = [str(part).strip() for part in raw_keys if str(part).strip()]

        ranked: List[ModelRecord] = []
        seen: set[str] = set()

        def add_record(record: Optional[ModelRecord]) -> None:
            if record is None or not record.supports_chat or record.key in seen:
                return
            ranked.append(record)
            seen.add(record.key)

        if bool(settings.get("collective_include_chosen", True)):
            add_record(chosen_record)
        for key in preferred_keys:
            add_record(self.record_map.get(key))
        for record in consultants:
            add_record(record)

        limit = int(settings.get("collective_consultant_limit") or 0)
        if limit > 0:
            ranked = ranked[:limit]
        return ranked

    def _prepare_memory_bundle(self, session_id: str, prompt: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        if settings.get("memory_enabled", True) is False:
            return {"memory_notes": [], "context_block": "", "example_count": 0, "turn_count": 0}
        return self.memory_store.build_context(session_id, prompt)

    def _seed_auto_tool_events(self, prompt: str, settings: Dict[str, Any]) -> List[ToolEvent]:
        events: List[ToolEvent] = []
        profile = (
            settings.get("_prompt_profile")
            if isinstance(settings.get("_prompt_profile"), Mapping)
            else {}
        )
        knowledge = (
            profile.get("knowledge")
            if isinstance(profile.get("knowledge"), Mapping)
            else {}
        )
        search_recommended = bool(
            should_offer_web_search(prompt)
            or knowledge.get("freshness_required", False)
        )
        redaction = redact_external_query(prompt)
        safe_query = str(redaction.get("query") or "").strip()
        if (
            bool(settings.get("web_search_enabled", False))
            and search_recommended
            and bool(redaction.get("safe_to_send", False))
            and safe_query
        ):
            try:
                events.append(
                    self.web_search.search(
                        safe_query,
                        max_results=_coerce_int_setting(settings.get("web_search_results"), 5, minimum=1, maximum=20),
                    )
                )
            except Exception:
                pass
        if bool(settings.get("cmd_open_enabled", True)) and should_offer_open_cmd(prompt):
            try:
                events.append(self.cmd_open.open(""))
            except Exception:
                pass
        return events

    def _run_web_query_cached(
        self,
        query: str,
        tool_cache: Dict[str, ToolEvent],
        settings: Dict[str, Any],
    ) -> Optional[ToolEvent]:
        redaction = redact_external_query(query)
        if not bool(redaction.get("safe_to_send", False)):
            return None
        safe_query = str(redaction.get("query") or "").strip()
        key = _trim_text(safe_query, limit=220).lower()
        if not key:
            return None
        if key in tool_cache:
            return tool_cache[key]
        if len(tool_cache) >= _coerce_int_setting(settings.get("web_search_budget"), 3, minimum=0, maximum=100):
            return None
        try:
            event = self.web_search.search(
                safe_query,
                max_results=_coerce_int_setting(settings.get("web_search_results"), 5, minimum=1, maximum=20),
            )
        except Exception:
            return None
        tool_cache[key] = event
        return event

    def _run_cmd_open_cached(
        self,
        working_dir: str,
        tool_cache: Dict[str, ToolEvent],
    ) -> Optional[ToolEvent]:
        cooked_dir = _trim_text(working_dir or "", limit=220)
        key = f"open_cmd::{cooked_dir.lower()}"
        if key in tool_cache:
            return tool_cache[key]
        try:
            event = self.cmd_open.open(cooked_dir)
        except Exception:
            return None
        tool_cache[key] = event
        return event

    def _run_text_model(
        self,
        record: ModelRecord,
        *,
        session_id: str,
        prompt: str,
        settings: Dict[str, Any],
        route_reason: str,
        tool_cache: Dict[str, ToolEvent],
        allow_tool_calls: bool,
    ) -> Tuple[ChatResult, List[ToolEvent]]:
        resolved_record, backend = self.ensure_backend(record.key)
        local_events: List[ToolEvent] = []
        run_settings = dict(settings)
        run_settings.pop("_route_model_call_counter", None)
        effective_route_reason = route_reason
        if resolved_record.key != record.key:
            effective_route_reason = (
                f"{route_reason} Requested {record.label} could not be initialized, so the system fell back to "
                f"{resolved_record.label}."
            )
        run_settings["route_reason"] = effective_route_reason
        if tool_cache:
            run_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        if allow_tool_calls and (bool(settings.get("web_search_enabled", False)) or bool(settings.get("cmd_open_enabled", True))):
            run_settings["tool_instruction"] = (
                "Available tools:\n"
                "TOOL:web_search: <query>\n"
                "TOOL:open_cmd: <optional working directory>\n"
                "Use a tool line only when it is explicitly needed, otherwise answer normally."
            )
        _record_route_model_call(settings)
        result = backend.chat(session_id, prompt, run_settings)
        raw_response = str(result.response or "")
        requests = parse_tool_requests(raw_response) if allow_tool_calls else []
        result.response = strip_tool_calls(raw_response)
        if requests:
            for request in requests:
                if request["name"] == "web_search" and bool(settings.get("web_search_enabled", False)):
                    tool_event = self._run_web_query_cached(request["argument"], tool_cache, settings)
                elif request["name"] == "open_cmd" and bool(settings.get("cmd_open_enabled", True)):
                    tool_event = self._run_cmd_open_cached(request["argument"], tool_cache)
                else:
                    tool_event = None
                if tool_event is not None:
                    local_events.append(tool_event)
            if local_events:
                follow_settings = dict(settings)
                follow_settings.pop("_route_model_call_counter", None)
                follow_settings["route_reason"] = route_reason
                follow_settings["memory_context"] = str(settings.get("memory_context") or "")
                follow_settings["consultation_context"] = str(settings.get("consultation_context") or "")
                follow_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
                follow_settings["tool_instruction"] = "Tool results are already available below. Use them and answer directly."
                _record_route_model_call(settings)
                follow = backend.chat(session_id, prompt, follow_settings)
                follow.response = strip_tool_calls(follow.response)
                result = follow
        return result, local_events

    def _build_consult_prompt(self, prompt: str, action_mode: str) -> str:
        if action_mode == "image":
            return (
                "You are one consultant in a multimodel image-planning panel.\n"
                "Return short art-direction notes only: subject, composition, style, colors, and any constraints.\n\n"
                f"Request:\n{prompt}"
            )
        return (
            "You are one consultant in a multimodel answer panel.\n"
            "Give a concise answer draft with the key reasoning or caveat in under 130 words.\n\n"
            f"Request:\n{prompt}"
        )

    def _format_consultations(self, consultation_rows: Sequence[Dict[str, str]]) -> str:
        lines: List[str] = []
        for row in consultation_rows:
            label = _trim_text(row.get("model_label") or row.get("model_key") or "model", limit=80)
            response = _trim_text(row.get("response") or "", limit=280)
            if not response:
                continue
            lines.append(f"- {label}: {response}")
        return "\n".join(lines)

    def _build_synthesis_prompt(self, prompt: str, action_mode: str) -> str:
        if action_mode == "image":
            return (
                "Synthesize the cross-model notes into one final image prompt.\n"
                "Output a single polished prompt only, without bullets or explanation.\n\n"
                f"Original request:\n{prompt}"
            )
        return (
            "Synthesize the memory, tool results, and cross-model consultation into one final answer.\n"
            "Be direct, coherent, and avoid repeating the panel format.\n\n"
            f"Original request:\n{prompt}"
        )

    def _normalized_agent_mode(self, raw_mode: Any) -> str:
        cooked = str(raw_mode or "off").strip().lower()
        aliases = {
            "adaptive": "auto",
            "smart": "auto",
            "collective_all": "collective",
            "panel": "collective",
            "loop_agent": "loop",
            "collective_loop_agent": "collective_loop",
        }
        normalized = aliases.get(cooked, cooked)
        if normalized in {"off", "auto", "collective", "loop", "collective_loop"}:
            return normalized
        return "off"

    def _normalized_auto_budget_profile(self, raw_profile: Any) -> str:
        cooked = str(raw_profile or "balanced").strip().lower()
        aliases = {
            "default": "balanced",
            "normal": "balanced",
            "standard": "balanced",
            "cheap": "fast",
            "economy": "fast",
            "latency": "fast",
            "quality": "deep",
            "thorough": "deep",
            "frontier": "max",
            "maximum": "max",
            "unbounded": "max",
        }
        normalized = aliases.get(cooked, cooked)
        if normalized in AUTO_AGENT_BUDGET_PROFILES:
            return normalized
        return "balanced"

    def _allowed_auto_agent_modes(
        self,
        *,
        action_mode: str,
        allow_collective: bool,
        collective_available: bool,
        allow_loop: bool,
    ) -> List[str]:
        modes = ["off"]
        if allow_collective and collective_available:
            modes.append("collective")
        if action_mode != "image" and allow_loop:
            modes.append("loop")
        if action_mode != "image" and allow_loop and allow_collective and collective_available:
            modes.append("collective_loop")
        return modes

    def _budget_allowed_auto_agent_modes(self, allowed_modes: Sequence[str], max_agent_mode: str) -> List[str]:
        if max_agent_mode not in AUTO_AGENT_MODE_ORDER:
            max_agent_mode = "collective_loop"
        max_idx = AUTO_AGENT_MODE_ORDER.index(max_agent_mode)
        allowed = set(allowed_modes)
        return [mode for mode in AUTO_AGENT_MODE_ORDER[: max_idx + 1] if mode in allowed]

    def _effective_agent_mode_for_action(self, agent_mode: str, action_mode: str) -> str:
        if action_mode != "image":
            return agent_mode
        if agent_mode == "loop":
            return "off"
        if agent_mode == "collective_loop":
            return "collective"
        return agent_mode

    def _route_latency_tier(self, cost_units: float) -> str:
        if cost_units <= 1.5:
            return "low"
        if cost_units <= 4.0:
            return "moderate"
        if cost_units <= 10.0:
            return "high"
        return "frontier"

    def _estimate_route_economics(
        self,
        *,
        selected_agent_mode: str,
        action_mode: str,
        settings: Dict[str, Any],
        auto_agent_policy: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        loop_budget = _coerce_int_setting(
            settings.get("loop_max_steps") or settings.get("loop_budget"),
            LOOP_AGENT_DEFAULT_MAX_STEPS,
            minimum=2,
            maximum=LOOP_AGENT_HARD_MAX_STEPS,
        )
        collective_count = int((auto_agent_policy or {}).get("collective_model_count") or 0)
        if selected_agent_mode in {"collective", "collective_loop"}:
            collective_count = max(2, collective_count)
        tool_budget = (
            _coerce_int_setting(settings.get("web_search_budget"), 0, minimum=0, maximum=100)
            if bool(settings.get("web_search_enabled", False))
            else 0
        )

        if selected_agent_mode == "collective":
            estimated_model_calls = max(2, collective_count + 1)
            planned_steps = 0
        elif selected_agent_mode == "loop":
            estimated_model_calls = loop_budget * 3
            planned_steps = loop_budget
        elif selected_agent_mode == "collective_loop":
            estimated_model_calls = loop_budget * (3 + max(0, collective_count - 1))
            planned_steps = loop_budget
        else:
            estimated_model_calls = 1
            planned_steps = 0

        if action_mode == "image":
            estimated_model_calls += 1

        estimated_cost_units = round(float(estimated_model_calls) + (float(tool_budget) * 0.25), 2)
        budget_profile = str((auto_agent_policy or {}).get("budget_profile") or "manual")
        budget_policy = (auto_agent_policy or {}).get("budget_policy")
        cost_preference = ""
        if isinstance(budget_policy, dict):
            cost_preference = _trim_text(budget_policy.get("cost_preference") or "", limit=80)
        return {
            "selected_agent_mode": selected_agent_mode,
            "action_mode": action_mode,
            "budget_profile": budget_profile,
            "cost_preference": cost_preference,
            "estimated_model_calls": estimated_model_calls,
            "estimated_tool_calls": tool_budget,
            "estimated_cost_units": estimated_cost_units,
            "latency_tier": self._route_latency_tier(estimated_cost_units),
            "planned_loop_steps": planned_steps,
            "collective_model_count": collective_count,
            "requested_reasoning_cycles": settings.get("reasoning_cycles"),
            "adaptive_compute": bool(settings.get("adaptive_compute")) if settings.get("adaptive_compute") is not None else None,
        }

    def _finalize_route_economics(
        self,
        *,
        estimate: Dict[str, Any],
        trace: Dict[str, Any],
        elapsed_ms: float,
        actual_model_calls: Optional[int] = None,
    ) -> Dict[str, Any]:
        loop_steps = list(trace.get("loop_steps") or [])
        consultation_rows = list(trace.get("consultation_rows") or [])
        consulted_models = list(trace.get("consulted_models") or [])
        tool_events = list(trace.get("tool_events") or [])
        compute = trace.get("compute") if isinstance(trace.get("compute"), dict) else {}
        mode = str(estimate.get("selected_agent_mode") or trace.get("resolved_agent_mode") or trace.get("agent_mode") or "off")

        if mode == "collective_loop":
            inferred_model_calls = max(1, len(loop_steps) * 3 + len(consultation_rows))
        elif mode == "loop":
            inferred_model_calls = max(1, len(loop_steps) * 3)
        elif mode == "collective":
            inferred_model_calls = max(1, 1 + max(len(consultation_rows), len(consulted_models)))
        else:
            inferred_model_calls = 1
        tracked_model_calls = int(actual_model_calls or 0)
        resolved_model_calls = tracked_model_calls if tracked_model_calls > 0 else inferred_model_calls
        actual_cost_units = round(float(resolved_model_calls) + (float(len(tool_events)) * 0.25), 2)
        return {
            "estimate": estimate,
            "actual": {
                "elapsed_ms": round(float(elapsed_ms), 2),
                "model_calls": int(resolved_model_calls),
                "tool_calls": len(tool_events),
                "loop_steps": len(loop_steps),
                "consultation_count": max(len(consultation_rows), len(consulted_models)),
                "cost_units": actual_cost_units,
                "latency_tier": self._route_latency_tier(actual_cost_units),
                "compute_applied": bool(compute.get("applied")),
                "reasoning_cycles_used": compute.get("cycles_used"),
                "compute_exit_reason": compute.get("exit_reason"),
                "prediction_confidence_delta": compute.get("prediction_confidence_delta"),
            },
        }

    def _auto_session_budget_limit(self, settings: Dict[str, Any]) -> Optional[float]:
        for key in ("auto_session_budget_units", "session_route_budget_units", "route_budget_units"):
            raw = settings.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0.0:
                return round(min(value, 100_000.0), 3)
        return None

    def _auto_session_budget_target_routes(self, settings: Dict[str, Any]) -> Optional[int]:
        for key in ("auto_session_budget_target_routes", "session_route_budget_target_routes", "route_budget_target_routes"):
            raw = settings.get(key)
            if raw in (None, ""):
                continue
            try:
                value = int(float(raw))
            except (TypeError, ValueError):
                continue
            if value > 1:
                return min(value, 10_000)
        return None

    def _session_route_budget_snapshot(self, session_id: str, budget_limit: Optional[float]) -> Optional[Dict[str, Any]]:
        if budget_limit is None:
            return None
        usage = self.memory_store.route_usage_summary(session_id)
        economics = usage.get("economics") if isinstance(usage.get("economics"), dict) else {}
        used = float(economics.get("total_cost_units") or 0.0)
        remaining = round(max(0.0, float(budget_limit) - used), 3)
        return {
            "limit_cost_units": round(float(budget_limit), 3),
            "used_cost_units": round(used, 3),
            "remaining_cost_units": remaining,
            "route_count": int(usage.get("total_routes") or 0),
            "recent_route_count": int(usage.get("recent_routes") or 0),
        }

    def _apply_auto_session_budget(
        self,
        *,
        selected: str,
        action_mode: str,
        settings: Dict[str, Any],
        auto_agent_policy: Dict[str, Any],
        budget_snapshot: Optional[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        estimate = self._estimate_route_economics(
            selected_agent_mode=selected,
            action_mode=action_mode,
            settings=settings,
            auto_agent_policy=auto_agent_policy,
        )
        if not budget_snapshot:
            return selected, estimate, None, None

        remaining = float(budget_snapshot.get("remaining_cost_units") or 0.0)
        target_routes = self._auto_session_budget_target_routes(settings)
        route_count = int(budget_snapshot.get("route_count") or 0)
        target_remaining_routes: Optional[int] = None
        pacing_cap: Optional[float] = None
        target_pacing_active = False
        if target_routes is not None:
            target_remaining_routes = max(1, int(target_routes) - route_count)
            pacing_cap = round(max(0.0, remaining) / float(target_remaining_routes), 3)
            target_pacing_active = target_remaining_routes > 1
        allowed_modes = list(auto_agent_policy.get("allowed_agent_modes") or AUTO_AGENT_MODE_ORDER)
        original = selected
        original_estimated_cost = float(estimate.get("estimated_cost_units") or 0.0)
        effective_cap = remaining if not target_pacing_active else min(remaining, float(pacing_cap or 0.0))
        estimates = [
            {
                "selected_agent_mode": selected,
                "estimated_cost_units": estimate.get("estimated_cost_units"),
                "cap_cost_units": round(effective_cap, 3),
            }
        ]

        while selected != "off" and float(estimate.get("estimated_cost_units") or 0.0) > effective_cap:
            next_selected = self._neighbor_auto_agent_mode(selected, allowed_modes, -1)
            if next_selected == selected:
                break
            selected = next_selected
            estimate = self._estimate_route_economics(
                selected_agent_mode=selected,
                action_mode=action_mode,
                settings=settings,
                auto_agent_policy=auto_agent_policy,
            )
            estimates.append(
                {
                    "selected_agent_mode": selected,
                    "estimated_cost_units": estimate.get("estimated_cost_units"),
                    "cap_cost_units": round(effective_cap, 3),
                }
            )

        estimated_cost = float(estimate.get("estimated_cost_units") or 0.0)
        budget_state = dict(budget_snapshot)
        if target_routes is not None:
            budget_state.update(
                {
                    "target_route_count": int(target_routes),
                    "target_remaining_routes": int(target_remaining_routes or 1),
                    "pacing_cap_cost_units": round(float(pacing_cap or 0.0), 3),
                    "effective_cap_cost_units": round(float(effective_cap), 3),
                    "pacing_cap_applied": bool(target_pacing_active),
                }
            )
        budget_state.update(
            {
                "selected_agent_mode": selected,
                "estimated_cost_units": round(estimated_cost, 3),
                "would_exceed_remaining": bool(estimated_cost > remaining),
                "would_exceed_pacing_cap": bool(
                    target_pacing_active and estimated_cost > float(pacing_cap or 0.0)
                ),
            }
        )
        if selected == original == "off" and estimated_cost > remaining:
            adjustment = {
                "direction": "floor",
                "from": "off",
                "to": "off",
                "reason": "session_route_budget_exhausted_single_pass_floor",
                "remaining_cost_units": round(remaining, 3),
                "limit_cost_units": budget_state["limit_cost_units"],
                "used_cost_units": budget_state["used_cost_units"],
                "estimated_cost_units": round(estimated_cost, 3),
                "candidate_estimates": estimates,
            }
            return selected, estimate, adjustment, budget_state
        if selected == original:
            return selected, estimate, None, budget_state

        target_pacing_triggered = (
            target_pacing_active
            and original_estimated_cost > float(pacing_cap or 0.0)
            and estimated_cost <= remaining
        )
        adjustment = {
            "direction": "downgrade",
            "from": original,
            "to": selected,
            "reason": (
                "session_route_budget_target_pacing"
                if target_pacing_triggered
                else "session_route_budget_would_exceed_remaining"
            ),
            "remaining_cost_units": round(remaining, 3),
            "limit_cost_units": budget_state["limit_cost_units"],
            "used_cost_units": budget_state["used_cost_units"],
            "estimated_cost_units": round(estimated_cost, 3),
            "candidate_estimates": estimates,
        }
        if target_routes is not None:
            adjustment.update(
                {
                    "target_route_count": int(target_routes),
                    "target_remaining_routes": int(target_remaining_routes or 1),
                    "pacing_cap_cost_units": round(float(pacing_cap or 0.0), 3),
                    "effective_cap_cost_units": round(float(effective_cap), 3),
                }
            )
        return selected, estimate, adjustment, budget_state

    def _build_post_filter_logging_support(
        self,
        *,
        selected: str,
        action_mode: str,
        settings: Dict[str, Any],
        auto_agent_policy: Dict[str, Any],
        selected_estimate: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Freeze the final feasible action set after capability and budget filters."""

        allowed = {
            str(mode)
            for mode in list(auto_agent_policy.get("allowed_agent_modes") or AUTO_AGENT_MODE_ORDER)
            if str(mode) in AUTO_AGENT_MODE_ORDER
        }
        allowed.add(selected)
        budget_state = (
            auto_agent_policy.get("session_budget")
            if isinstance(auto_agent_policy.get("session_budget"), dict)
            else {}
        )
        cap_raw = budget_state.get("effective_cap_cost_units")
        if cap_raw is None:
            cap_raw = budget_state.get("remaining_cost_units")
        try:
            effective_cap = float(cap_raw) if cap_raw is not None else None
        except (TypeError, ValueError, OverflowError):
            effective_cap = None
        if effective_cap is not None and (not math.isfinite(effective_cap) or effective_cap < 0.0):
            effective_cap = None

        candidates: List[Dict[str, Any]] = []
        exclusions: List[Dict[str, Any]] = []
        for mode in AUTO_AGENT_MODE_ORDER:
            reasons: List[str] = []
            if mode not in allowed:
                reasons.append(
                    "action_mode_unsupported"
                    if action_mode == "image" and mode in {"loop", "collective_loop"}
                    else "capability_or_policy_filter"
                )
            estimate = (
                dict(selected_estimate)
                if mode == selected
                else self._estimate_route_economics(
                    selected_agent_mode=mode,
                    action_mode=action_mode,
                    settings=settings,
                    auto_agent_policy=auto_agent_policy,
                )
            )
            estimated_cost = float(estimate.get("estimated_cost_units") or 0.0)
            if (
                not reasons
                and effective_cap is not None
                and mode != "off"
                and estimated_cost > effective_cap
            ):
                reasons.append("session_budget_post_filter")

            if mode == selected:
                if reasons:
                    raise RuntimeError("selected route was excluded from its post-filter logging support")
                candidates.append(
                    {
                        "action": mode,
                        "selected": True,
                        "estimated_cost_units": round(estimated_cost, 3),
                        "estimated_model_calls": int(estimate.get("estimated_model_calls") or 1),
                        "planned_loop_steps": int(estimate.get("planned_loop_steps") or 0),
                        "latency_tier": str(estimate.get("latency_tier") or "unknown"),
                    }
                )
            elif reasons:
                exclusions.append({"action": mode, "reasons": reasons})
            else:
                candidates.append(
                    {
                        "action": mode,
                        "selected": False,
                        "estimated_cost_units": round(estimated_cost, 3),
                        "estimated_model_calls": int(estimate.get("estimated_model_calls") or 1),
                        "planned_loop_steps": int(estimate.get("planned_loop_steps") or 0),
                        "latency_tier": str(estimate.get("latency_tier") or "unknown"),
                    }
                )

        candidates.sort(key=lambda row: AUTO_AGENT_MODE_ORDER.index(str(row["action"])))
        exclusions.sort(key=lambda row: AUTO_AGENT_MODE_ORDER.index(str(row["action"])))
        eligible = [str(row["action"]) for row in candidates]
        probabilities = {mode: 1.0 if mode == selected else 0.0 for mode in eligible}
        raw_support = {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "decision_type": "deterministic",
            "probability_stage": "post_filter",
            "sampler": {
                "name": "threshold_budget_argmax",
                "version": "1",
                "exploration_rate": 0.0,
                "assignment_unit": "route",
                "assignment_commitment": None,
            },
            "candidates": candidates,
            "exclusions": exclusions,
        }
        return build_logging_support_envelope(
            raw_support,
            eligible_modes=eligible,
            action_probabilities=probabilities,
            chosen_mode=selected,
        )

    def _preview_route_alternatives(
        self,
        *,
        selected: str,
        action_mode: str,
        settings: Dict[str, Any],
        auto_agent_policy: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if auto_agent_policy is not None:
            allowed_agent_modes = auto_agent_policy.get("allowed_agent_modes")
            raw_candidate_modes = allowed_agent_modes if allowed_agent_modes is not None else AUTO_AGENT_MODE_ORDER
            candidate_modes = [
                str(mode)
                for mode in list(raw_candidate_modes)
                if str(mode) in AUTO_AGENT_MODE_ORDER
            ]
        else:
            candidate_modes = [selected] if selected in AUTO_AGENT_MODE_ORDER else ["off"]
        if selected in AUTO_AGENT_MODE_ORDER and selected not in candidate_modes:
            candidate_modes.append(selected)

        rows: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for mode in sorted(set(candidate_modes), key=lambda item: AUTO_AGENT_MODE_ORDER.index(item)):
            if mode in seen:
                continue
            seen.add(mode)
            estimate = self._estimate_route_economics(
                selected_agent_mode=mode,
                action_mode=action_mode,
                settings=settings,
                auto_agent_policy=auto_agent_policy,
            )
            rows.append(
                {
                    "selected_agent_mode": mode,
                    "is_selected": mode == selected,
                    "estimate": estimate,
                    "estimated_cost_units": estimate.get("estimated_cost_units"),
                    "estimated_model_calls": estimate.get("estimated_model_calls"),
                    "planned_loop_steps": estimate.get("planned_loop_steps"),
                    "latency_tier": estimate.get("latency_tier"),
                }
            )
        return rows

    def _estimated_route_quality(
        self,
        *,
        mode: str,
        action_mode: str,
        estimated_cost_units: Optional[float] = None,
        auto_agent_policy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        feedback_summary = (
            auto_agent_policy.get("feedback_summary")
            if isinstance(auto_agent_policy, dict) and isinstance(auto_agent_policy.get("feedback_summary"), dict)
            else {}
        )
        mode_scores = feedback_summary.get("mode_scores") if isinstance(feedback_summary.get("mode_scores"), dict) else {}
        stats = mode_scores.get(mode) if isinstance(mode_scores.get(mode), dict) else {}
        evidence = self._adaptive_route_evidence(stats)
        if evidence is not None:
            return {
                "estimated_quality_score": round(float(evidence["quality_score"]), 3),
                "estimated_quality_cost_score": round(float(evidence["quality_cost_score"]), 3),
                "estimated_quality_lower_bound": evidence.get("quality_lower_bound"),
                "estimated_quality_upper_bound": evidence.get("quality_upper_bound"),
                "risk_adjusted_quality_cost_score": evidence.get("risk_adjusted_quality_cost_score"),
                "effective_sample_size": evidence.get("effective_sample_size"),
                "confidence_level": evidence.get("confidence_level"),
                "confidence_status": evidence.get("confidence_status"),
                "quality_source": "adaptive_feedback",
                "quality_cost_source": "adaptive_feedback",
                "quality_evidence_status": "adaptive_complete",
                "quality_evidence": evidence,
            }
        quality_evidence_status = self._adaptive_route_evidence_status(stats)

        score = int((auto_agent_policy or {}).get("score") or 0)
        if mode == "off":
            quality = max(0.25, 0.62 - (0.055 * float(score)))
        else:
            thresholds = {"collective": 2, "loop": 4, "collective_loop": 5}
            bases = {"collective": 0.62, "loop": 0.72, "collective_loop": 0.88}
            threshold = int(thresholds.get(mode, 99))
            base = float(bases.get(mode, 0.5))
            if action_mode == "image" and mode == "collective":
                threshold = 3
            if score >= threshold:
                quality = base + (0.04 * float(min(score - threshold, 4)))
            else:
                quality = base - (0.06 * float(threshold - score))
        estimated_quality_score = round(max(0.0, min(0.99, quality)), 3)
        try:
            cost = float(estimated_cost_units or 0.0)
        except (TypeError, ValueError):
            cost = 0.0
        if not math.isfinite(cost):
            cost = 0.0
        cost = max(0.0, cost)
        return {
            "estimated_quality_score": estimated_quality_score,
            "estimated_quality_cost_score": round(estimated_quality_score / (1.0 + (cost / 10.0)), 3),
            "estimated_quality_lower_bound": None,
            "estimated_quality_upper_bound": None,
            "risk_adjusted_quality_cost_score": None,
            "effective_sample_size": 0.0,
            "confidence_level": None,
            "confidence_status": "heuristic_prior",
            "quality_source": "heuristic_policy",
            "quality_cost_source": "heuristic_cost_adjusted",
            "quality_evidence_status": quality_evidence_status,
            "quality_evidence": None,
        }

    def _route_frontier_budget_state(self, auto_agent_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        session_budget = (
            auto_agent_policy.get("session_budget")
            if isinstance(auto_agent_policy, dict) and isinstance(auto_agent_policy.get("session_budget"), dict)
            else None
        )
        if not session_budget:
            return {
                "remaining_cost_units": None,
                "pacing_cap_cost_units": None,
                "effective_cap_cost_units": None,
                "pacing_cap_applied": False,
                "limit_cost_units": None,
                "used_cost_units": None,
                "route_count": None,
                "target_route_count": None,
                "target_remaining_routes": None,
            }

        def optional_float(key: str) -> Optional[float]:
            raw = session_budget.get(key)
            if raw in (None, ""):
                return None
            try:
                return round(max(0.0, float(raw)), 3)
            except (TypeError, ValueError):
                return None

        def optional_int(key: str) -> Optional[int]:
            raw = session_budget.get(key)
            if raw in (None, ""):
                return None
            try:
                return int(float(raw))
            except (TypeError, ValueError):
                return None

        remaining = optional_float("remaining_cost_units")
        pacing_cap_applied = bool(session_budget.get("pacing_cap_applied"))
        pacing_cap = optional_float("pacing_cap_cost_units") if pacing_cap_applied else None
        effective_cap = optional_float("effective_cap_cost_units")
        if effective_cap is None:
            if remaining is not None and pacing_cap is not None:
                effective_cap = min(remaining, pacing_cap)
            elif remaining is not None:
                effective_cap = remaining
            elif pacing_cap is not None:
                effective_cap = pacing_cap
        return {
            "remaining_cost_units": remaining,
            "pacing_cap_cost_units": pacing_cap,
            "effective_cap_cost_units": effective_cap,
            "pacing_cap_applied": pacing_cap is not None,
            "limit_cost_units": optional_float("limit_cost_units"),
            "used_cost_units": optional_float("used_cost_units"),
            "route_count": optional_int("route_count"),
            "target_route_count": optional_int("target_route_count"),
            "target_remaining_routes": optional_int("target_remaining_routes"),
        }

    def _annotate_route_frontier(
        self,
        *,
        alternatives: List[Dict[str, Any]],
        selected: str,
        action_mode: str,
        auto_agent_policy: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not alternatives:
            return {
                "recommended_agent_mode": selected,
                "recommended_reason": "no_alternatives_available",
                "recommended_budget_blocker": None,
                "ranked_modes": [],
                "pareto_modes": [],
                "budget_feasible_pareto_modes": [],
                "budget_fit_count": 0,
                "budget_blockers": {"remaining_budget": 0, "pacing_cap": 0, "effective_cap": 0, "none": 0},
                "budget_cap_cost_units": None,
                "remaining_cost_units": None,
                "pacing_cap_cost_units": None,
                "effective_cap_cost_units": None,
                "pacing_cap_applied": False,
                "recommended_estimated_quality_cost_score": None,
                "selected_estimated_quality_cost_score": None,
                "minimum_route_floor": False,
                "selected_budget_fit": None,
                "selected_budget_blocker": None,
                "selected_fits_remaining_budget": None,
                "selected_fits_pacing_cap": None,
                "selected_frontier_rank": None,
                "selected_matches_recommendation": False,
                "selected_agent_mode": selected,
            }

        budget_profile = str((auto_agent_policy or {}).get("budget_profile") or "manual")
        cost_penalty = {
            "fast": 0.35,
            "balanced": 0.06,
            "deep": 0.025,
            "max": 0.0,
            "manual": 0.03,
        }.get(budget_profile, 0.06)
        budget_state = self._route_frontier_budget_state(auto_agent_policy)
        effective_cap = budget_state["effective_cap_cost_units"]
        remaining_cap = budget_state["remaining_cost_units"]
        pacing_cap = budget_state["pacing_cap_cost_units"]
        session_budget_adjustment = (
            auto_agent_policy.get("session_budget_adjustment")
            if isinstance(auto_agent_policy, dict) and isinstance(auto_agent_policy.get("session_budget_adjustment"), dict)
            else {}
        )
        minimum_route_floor = (
            selected == "off"
            and session_budget_adjustment.get("reason") == "session_route_budget_exhausted_single_pass_floor"
        )
        max_cost = max(float(row.get("estimated_cost_units") or 0.0) for row in alternatives) or 1.0

        adaptive_quality_cost_available = False
        for row in alternatives:
            mode = str(row.get("selected_agent_mode") or "off")
            cost = max(0.0, float(row.get("estimated_cost_units") or 0.0))
            quality = self._estimated_route_quality(
                mode=mode,
                action_mode=action_mode,
                estimated_cost_units=cost,
                auto_agent_policy=auto_agent_policy,
            )
            adaptive_quality_cost_available = (
                adaptive_quality_cost_available or quality.get("quality_source") == "adaptive_feedback"
            )
            fits_remaining = True if remaining_cap is None else cost <= (float(remaining_cap) + 1e-9)
            fits_pacing = None if pacing_cap is None else cost <= (float(pacing_cap) + 1e-9)
            budget_fit = True if effective_cap is None else cost <= (float(effective_cap) + 1e-9)
            budget_overage = 0.0 if effective_cap is None else max(0.0, cost - float(effective_cap))
            if not fits_remaining:
                budget_blocker = "remaining_budget"
            elif fits_pacing is False:
                budget_blocker = "pacing_cap"
            elif not budget_fit:
                budget_blocker = "effective_cap"
            else:
                budget_blocker = None
            row.update(
                {
                    "estimated_quality_score": quality["estimated_quality_score"],
                    "estimated_quality_cost_score": quality["estimated_quality_cost_score"],
                    "estimated_quality_lower_bound": quality["estimated_quality_lower_bound"],
                    "estimated_quality_upper_bound": quality["estimated_quality_upper_bound"],
                    "risk_adjusted_quality_cost_score": quality["risk_adjusted_quality_cost_score"],
                    "effective_sample_size": quality["effective_sample_size"],
                    "confidence_level": quality["confidence_level"],
                    "confidence_status": quality["confidence_status"],
                    "quality_source": quality["quality_source"],
                    "quality_cost_source": quality["quality_cost_source"],
                    "quality_evidence_status": quality["quality_evidence_status"],
                    "quality_evidence": quality["quality_evidence"],
                    "budget_fit": bool(budget_fit),
                    "fits_remaining_budget": bool(fits_remaining),
                    "fits_pacing_cap": fits_pacing,
                    "effective_cap_cost_units": effective_cap,
                    "minimum_route_floor": bool(minimum_route_floor and mode == "off"),
                    "budget_blocker": budget_blocker,
                    "budget_overage_units": round(budget_overage, 3),
                }
            )

        for row in alternatives:
            cost = max(0.0, float(row.get("estimated_cost_units") or 0.0))
            budget_fit = bool(row.get("budget_fit"))
            budget_overage = float(row.get("budget_overage_units") or 0.0)
            quality_score = float(row.get("estimated_quality_score") or 0.0)
            if adaptive_quality_cost_available and budget_profile != "max":
                adaptive_utility = row.get("risk_adjusted_quality_cost_score")
                frontier_score = float(
                    adaptive_utility
                    if isinstance(adaptive_utility, (int, float))
                    else row.get("estimated_quality_cost_score") or 0.0
                ) + (0.02 * quality_score)
            else:
                frontier_score = quality_score - (cost_penalty * (cost / max_cost))
            if not budget_fit:
                frontier_score -= 0.5 + min(0.5, budget_overage / max_cost)
            row["frontier_score"] = round(frontier_score, 4)

        for row in alternatives:
            row_quality = float(row.get("estimated_quality_score") or 0.0)
            row_cost = float(row.get("estimated_cost_units") or 0.0)
            dominated = False
            for other in alternatives:
                if other is row:
                    continue
                other_quality = float(other.get("estimated_quality_score") or 0.0)
                other_cost = float(other.get("estimated_cost_units") or 0.0)
                if (
                    other_quality >= row_quality
                    and other_cost <= row_cost
                    and (other_quality > row_quality or other_cost < row_cost)
                ):
                    dominated = True
                    break
            row["pareto_frontier"] = not dominated

        feasible_for_pareto = [row for row in alternatives if bool(row.get("budget_fit"))]
        for row in alternatives:
            if not bool(row.get("budget_fit")):
                row["budget_feasible_pareto_frontier"] = False
                continue
            row_quality = float(row.get("estimated_quality_score") or 0.0)
            row_cost = float(row.get("estimated_cost_units") or 0.0)
            dominated = False
            for other in feasible_for_pareto:
                if other is row:
                    continue
                other_quality = float(other.get("estimated_quality_score") or 0.0)
                other_cost = float(other.get("estimated_cost_units") or 0.0)
                if (
                    other_quality >= row_quality
                    and other_cost <= row_cost
                    and (other_quality > row_quality or other_cost < row_cost)
                ):
                    dominated = True
                    break
            row["budget_feasible_pareto_frontier"] = not dominated

        ranked = sorted(
            alternatives,
            key=lambda row: (
                bool(row.get("budget_fit")),
                float(row.get("frontier_score") or 0.0),
                float(row.get("estimated_quality_score") or 0.0),
                -float(row.get("estimated_cost_units") or 0.0),
                -AUTO_AGENT_MODE_ORDER.index(str(row.get("selected_agent_mode") or "off")),
            ),
            reverse=True,
        )
        for idx, row in enumerate(ranked, start=1):
            row["frontier_rank"] = idx

        feasible = [row for row in ranked if bool(row.get("budget_fit"))]
        if feasible:
            recommended = feasible[0]
        else:
            recommended = min(
                ranked,
                key=lambda row: (
                    float(row.get("budget_overage_units") or 0.0),
                    float(row.get("estimated_cost_units") or 0.0),
                    -float(row.get("estimated_quality_score") or 0.0),
                    AUTO_AGENT_MODE_ORDER.index(str(row.get("selected_agent_mode") or "off")),
                ),
            )
        recommended_mode = str(recommended.get("selected_agent_mode") or selected)
        selected_row = next((row for row in alternatives if row.get("selected_agent_mode") == selected), None)
        cheapest_feasible = min(feasible, key=lambda row: float(row.get("estimated_cost_units") or 0.0)) if feasible else None
        highest_quality_feasible = (
            max(feasible, key=lambda row: float(row.get("estimated_quality_score") or 0.0)) if feasible else None
        )
        if not feasible:
            remaining_fit_count = sum(1 for row in alternatives if bool(row.get("fits_remaining_budget")))
            reason = (
                "no_pacing_feasible_route"
                if remaining_fit_count > 0 and pacing_cap is not None
                else "no_budget_feasible_route"
            )
        elif recommended_mode == selected:
            reason = "selected_route_is_frontier_recommended"
        elif adaptive_quality_cost_available and recommended.get("quality_source") == "adaptive_feedback":
            reason = "adaptive_quality_cost_frontier_recommended"
        elif float(recommended.get("estimated_cost_units") or 0.0) < float((selected_row or {}).get("estimated_cost_units") or 0.0):
            reason = "cheaper_feasible_quality_cost_tradeoff"
        else:
            reason = "higher_quality_feasible_tradeoff"
        budget_blockers = {
            "remaining_budget": sum(1 for row in alternatives if row.get("budget_blocker") == "remaining_budget"),
            "pacing_cap": sum(1 for row in alternatives if row.get("budget_blocker") == "pacing_cap"),
            "effective_cap": sum(1 for row in alternatives if row.get("budget_blocker") == "effective_cap"),
            "none": sum(1 for row in alternatives if row.get("budget_blocker") is None),
        }

        return {
            "selected_agent_mode": selected,
            "recommended_agent_mode": recommended_mode,
            "recommended_reason": reason,
            "recommended_budget_blocker": recommended.get("budget_blocker"),
            "recommended_estimated_cost_units": recommended.get("estimated_cost_units"),
            "recommended_estimated_quality_score": recommended.get("estimated_quality_score"),
            "recommended_estimated_quality_cost_score": recommended.get("estimated_quality_cost_score"),
            "recommended_risk_adjusted_quality_cost_score": recommended.get("risk_adjusted_quality_cost_score"),
            "recommended_quality_lower_bound": recommended.get("estimated_quality_lower_bound"),
            "recommended_quality_upper_bound": recommended.get("estimated_quality_upper_bound"),
            "budget_profile": budget_profile,
            "budget_blockers": budget_blockers,
            "budget_cap_cost_units": effective_cap,
            "remaining_cost_units": remaining_cap,
            "pacing_cap_cost_units": pacing_cap,
            "effective_cap_cost_units": effective_cap,
            "pacing_cap_applied": bool(budget_state["pacing_cap_applied"]),
            "limit_cost_units": budget_state["limit_cost_units"],
            "used_cost_units": budget_state["used_cost_units"],
            "route_count": budget_state["route_count"],
            "target_route_count": budget_state["target_route_count"],
            "target_remaining_routes": budget_state["target_remaining_routes"],
            "minimum_route_floor": bool(minimum_route_floor),
            "budget_fit_count": len(feasible),
            "selected_budget_fit": bool(selected_row.get("budget_fit")) if selected_row else None,
            "selected_budget_blocker": selected_row.get("budget_blocker") if selected_row else None,
            "selected_estimated_quality_cost_score": (
                selected_row.get("estimated_quality_cost_score") if selected_row else None
            ),
            "selected_risk_adjusted_quality_cost_score": (
                selected_row.get("risk_adjusted_quality_cost_score") if selected_row else None
            ),
            "selected_quality_lower_bound": (
                selected_row.get("estimated_quality_lower_bound") if selected_row else None
            ),
            "selected_quality_upper_bound": (
                selected_row.get("estimated_quality_upper_bound") if selected_row else None
            ),
            "selected_fits_remaining_budget": (
                bool(selected_row.get("fits_remaining_budget")) if selected_row else None
            ),
            "selected_fits_pacing_cap": selected_row.get("fits_pacing_cap") if selected_row else None,
            "selected_frontier_rank": selected_row.get("frontier_rank") if selected_row else None,
            "selected_matches_recommendation": recommended_mode == selected,
            "cheapest_feasible_agent_mode": (
                cheapest_feasible.get("selected_agent_mode") if cheapest_feasible else None
            ),
            "highest_quality_feasible_agent_mode": (
                highest_quality_feasible.get("selected_agent_mode") if highest_quality_feasible else None
            ),
            "pareto_modes": [
                str(row.get("selected_agent_mode"))
                for row in alternatives
                if bool(row.get("pareto_frontier"))
            ],
            "budget_feasible_pareto_modes": [
                str(row.get("selected_agent_mode"))
                for row in alternatives
                if bool(row.get("budget_feasible_pareto_frontier"))
            ],
            "ranked_modes": [
                {
                    "selected_agent_mode": str(row.get("selected_agent_mode") or "off"),
                    "frontier_rank": int(row.get("frontier_rank") or 0),
                    "frontier_score": row.get("frontier_score"),
                    "estimated_quality_score": row.get("estimated_quality_score"),
                    "estimated_quality_cost_score": row.get("estimated_quality_cost_score"),
                    "estimated_quality_lower_bound": row.get("estimated_quality_lower_bound"),
                    "estimated_quality_upper_bound": row.get("estimated_quality_upper_bound"),
                    "risk_adjusted_quality_cost_score": row.get("risk_adjusted_quality_cost_score"),
                    "effective_sample_size": row.get("effective_sample_size"),
                    "confidence_level": row.get("confidence_level"),
                    "confidence_status": row.get("confidence_status"),
                    "quality_source": row.get("quality_source"),
                    "quality_cost_source": row.get("quality_cost_source"),
                    "quality_evidence_status": row.get("quality_evidence_status"),
                    "estimated_cost_units": row.get("estimated_cost_units"),
                    "budget_fit": bool(row.get("budget_fit")),
                    "fits_remaining_budget": bool(row.get("fits_remaining_budget")),
                    "fits_pacing_cap": row.get("fits_pacing_cap"),
                    "budget_blocker": row.get("budget_blocker"),
                    "budget_overage_units": row.get("budget_overage_units"),
                    "pareto_frontier": bool(row.get("pareto_frontier")),
                    "budget_feasible_pareto_frontier": bool(row.get("budget_feasible_pareto_frontier")),
                }
                for row in ranked
            ],
        }

    def _neighbor_auto_agent_mode(self, selected: str, allowed_modes: Sequence[str], direction: int) -> str:
        ordered = [mode for mode in AUTO_AGENT_MODE_ORDER if mode in set(allowed_modes)]
        if selected not in ordered:
            return selected
        idx = ordered.index(selected)
        next_idx = idx + direction
        if next_idx < 0 or next_idx >= len(ordered):
            return selected
        return ordered[next_idx]

    def _auto_route_uncertainty_signals(self, prompt: str) -> List[str]:
        signals: List[str] = []
        raw_prompt = str(prompt or "")
        for name, pattern in AUTO_AGENT_UNCERTAINTY_SIGNAL_PATTERNS:
            if pattern.search(raw_prompt):
                signals.append(name)
        return signals

    def _auto_route_score_to_mode(self, mode: str, action_mode: str) -> Optional[int]:
        if action_mode == "image" and mode == "collective":
            return 3
        return AUTO_AGENT_SELECTION_THRESHOLDS.get(mode)

    def _auto_route_confidence(
        self,
        *,
        selected: str,
        score: int,
        allowed_modes: Sequence[str],
        action_mode: str,
        uncertainty_signals: Sequence[str],
    ) -> Dict[str, Any]:
        next_mode = self._neighbor_auto_agent_mode(selected, allowed_modes, 1)
        next_threshold = self._auto_route_score_to_mode(next_mode, action_mode) if next_mode != selected else None
        score_to_next_mode: Optional[int] = None
        if next_threshold is not None:
            score_to_next_mode = max(0, int(next_threshold) - int(score))

        level = "high"
        if score_to_next_mode == 1 and uncertainty_signals:
            level = "low"
        elif score_to_next_mode == 1:
            level = "medium"

        return {
            "level": level,
            "score": int(score),
            "selected_agent_mode": selected,
            "next_agent_mode": next_mode if next_mode != selected else None,
            "next_agent_mode_threshold": next_threshold,
            "score_to_next_mode": score_to_next_mode,
            "uncertainty_signals": list(uncertainty_signals)[:6],
            "adjusted": False,
        }

    def _maybe_apply_auto_uncertainty_margin(
        self,
        *,
        selected: str,
        score: int,
        allowed_modes: Sequence[str],
        action_mode: str,
        budget_profile: str,
        route_confidence: Dict[str, Any],
        uncertainty_signals: Sequence[str],
    ) -> tuple[str, Optional[Dict[str, Any]], Dict[str, Any]]:
        if (
            budget_profile == "fast"
            or action_mode == "image"
            or len(uncertainty_signals) < 2
            or route_confidence.get("score_to_next_mode") != 1
        ):
            return selected, None, route_confidence

        upgraded = self._neighbor_auto_agent_mode(selected, allowed_modes, 1)
        if upgraded == selected:
            return selected, None, route_confidence

        adjustment = {
            "direction": "upgrade",
            "from": selected,
            "to": upgraded,
            "reason": "borderline_score_with_uncertainty_signals",
            "score": int(score),
            "score_to_next_mode": route_confidence.get("score_to_next_mode"),
            "uncertainty_signals": list(uncertainty_signals)[:6],
            "budget_profile": budget_profile,
        }
        adjusted_confidence = dict(route_confidence)
        adjusted_confidence.update(
            {
                "adjusted": True,
                "adjusted_from": selected,
                "selected_agent_mode": upgraded,
                "adjustment_reason": adjustment["reason"],
                "level": "medium",
            }
        )
        return upgraded, adjustment, adjusted_confidence

    def _adaptive_route_evidence(self, stats: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(stats, dict):
            return None
        adaptive = stats.get("adaptive") if isinstance(stats.get("adaptive"), dict) else {}
        economics = stats.get("economics") if isinstance(stats.get("economics"), dict) else {}
        try:
            economics_samples = int(economics.get("sample_count") or 0)
            avg_cost = float(economics.get("avg_cost_units"))
        except (TypeError, ValueError):
            return None
        try:
            weighted_count = float(adaptive.get("weighted_count") or 0.0)
            quality_score = float(adaptive.get("quality_score"))
            quality_cost_score = float(adaptive.get("quality_cost_score"))
        except (TypeError, ValueError):
            return None
        if (
            economics_samples < 1
            or not math.isfinite(avg_cost)
            or not math.isfinite(weighted_count)
            or not math.isfinite(quality_score)
            or not math.isfinite(quality_cost_score)
            or weighted_count < AUTO_AGENT_ADAPTIVE_MIN_WEIGHTED_COUNT
            or quality_score < 0.0
            or quality_cost_score < 0.0
            or avg_cost < 0.0
            or bool(adaptive.get("regression_signal"))
        ):
            return None
        quality_lower_bound = adaptive.get("quality_lower_bound")
        quality_upper_bound = adaptive.get("quality_upper_bound")
        quality_cost_lower_bound = adaptive.get("quality_cost_lower_bound")
        quality_cost_upper_bound = adaptive.get("quality_cost_upper_bound")
        effective_sample_size = adaptive.get("effective_sample_size")
        try:
            quality_lower_bound = float(quality_lower_bound)
            quality_upper_bound = float(quality_upper_bound)
            quality_cost_lower_bound = float(quality_cost_lower_bound)
            quality_cost_upper_bound = float(quality_cost_upper_bound)
            effective_sample_size = float(effective_sample_size)
        except (TypeError, ValueError):
            quality_lower_bound = None
            quality_upper_bound = None
            quality_cost_lower_bound = None
            quality_cost_upper_bound = None
            effective_sample_size = 0.0
        confidence_status = str(adaptive.get("confidence_status") or "no_evidence")
        return {
            "sample_count": int(adaptive.get("sample_count") or stats.get("count") or 0),
            "economics_sample_count": economics_samples,
            "avg_cost_units": round(avg_cost, 3),
            "weighted_count": round(weighted_count, 3),
            "quality_score": round(quality_score, 3),
            "quality_cost_score": round(quality_cost_score, 3),
            "quality_lower_bound": (
                round(quality_lower_bound, 3) if isinstance(quality_lower_bound, float) and math.isfinite(quality_lower_bound) else None
            ),
            "quality_upper_bound": (
                round(quality_upper_bound, 3) if isinstance(quality_upper_bound, float) and math.isfinite(quality_upper_bound) else None
            ),
            "quality_cost_lower_bound": (
                round(quality_cost_lower_bound, 3)
                if isinstance(quality_cost_lower_bound, float) and math.isfinite(quality_cost_lower_bound)
                else None
            ),
            "quality_cost_upper_bound": (
                round(quality_cost_upper_bound, 3)
                if isinstance(quality_cost_upper_bound, float) and math.isfinite(quality_cost_upper_bound)
                else None
            ),
            "effective_sample_size": (
                round(effective_sample_size, 3) if math.isfinite(effective_sample_size) else 0.0
            ),
            "confidence_level": adaptive.get("confidence_level"),
            "confidence_status": confidence_status,
            "risk_adjusted_quality_cost_score": (
                round(quality_cost_lower_bound, 3)
                if confidence_status == "established"
                and isinstance(quality_cost_lower_bound, float)
                and math.isfinite(quality_cost_lower_bound)
                else round(quality_cost_score, 3)
            ),
            "weighted_net": adaptive.get("weighted_net"),
            "decay": adaptive.get("decay"),
        }

    def _adaptive_route_evidence_status(self, stats: Any) -> str:
        if not isinstance(stats, dict):
            return "heuristic_prior"
        adaptive = stats.get("adaptive") if isinstance(stats.get("adaptive"), dict) else {}
        economics = stats.get("economics") if isinstance(stats.get("economics"), dict) else {}
        sample_count = 0
        try:
            sample_count = int(adaptive.get("sample_count") or stats.get("count") or 0)
        except (TypeError, ValueError):
            sample_count = 0
        economics_sample_count = 0
        try:
            economics_sample_count = int(economics.get("sample_count") or 0)
        except (TypeError, ValueError):
            economics_sample_count = 0
        if sample_count <= 0 and economics_sample_count <= 0:
            return "heuristic_prior"
        if bool(adaptive.get("regression_signal")):
            return "adaptive_regression_blocked"
        try:
            avg_cost = float(economics.get("avg_cost_units"))
        except (TypeError, ValueError):
            return "adaptive_incomplete_cost_evidence"
        if economics_sample_count < 1 or not math.isfinite(avg_cost) or avg_cost < 0.0:
            return "adaptive_incomplete_cost_evidence"
        try:
            weighted_count = float(adaptive.get("weighted_count") or 0.0)
            quality_score = float(adaptive.get("quality_score"))
            quality_cost_score = float(adaptive.get("quality_cost_score"))
        except (TypeError, ValueError):
            return "adaptive_incomplete_feedback_evidence"
        if (
            not math.isfinite(weighted_count)
            or not math.isfinite(quality_score)
            or not math.isfinite(quality_cost_score)
            or weighted_count < AUTO_AGENT_ADAPTIVE_MIN_WEIGHTED_COUNT
            or quality_score < 0.0
            or quality_cost_score < 0.0
        ):
            return "adaptive_incomplete_feedback_evidence"
        return "adaptive_complete"

    def _prefer_adaptive_neighbor_route(
        self,
        *,
        selected: str,
        score: int,
        mode_scores: Dict[str, Any],
        allowed_modes: Sequence[str],
        budget_profile: str,
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        selected_idx = AUTO_AGENT_MODE_ORDER.index(selected) if selected in AUTO_AGENT_MODE_ORDER else 0
        selected_stats = mode_scores.get(selected) if isinstance(mode_scores.get(selected), dict) else {}
        selected_evidence = self._adaptive_route_evidence(selected_stats)
        selected_quality = float((selected_evidence or {}).get("quality_score", 0.5))
        selected_confident = bool((selected_evidence or {}).get("confidence_status") == "established")
        selected_quality_cost = float(
            (selected_evidence or {}).get(
                "quality_cost_upper_bound" if selected_confident else "quality_cost_score",
                0.5,
            )
        )
        selected_has_evidence = selected_evidence is not None

        candidates: List[Dict[str, Any]] = []
        for direction in (-1, 1):
            candidate = self._neighbor_auto_agent_mode(selected, allowed_modes, direction)
            if candidate == selected:
                continue
            stats = mode_scores.get(candidate) if isinstance(mode_scores.get(candidate), dict) else {}
            evidence = self._adaptive_route_evidence(stats)
            if evidence is None:
                continue
            candidate_idx = AUTO_AGENT_MODE_ORDER.index(candidate)
            moving_deeper = candidate_idx > selected_idx
            threshold = int(AUTO_AGENT_POSITIVE_THRESHOLDS.get(candidate, 99))
            if moving_deeper and score < threshold:
                continue
            candidate_quality = float(evidence["quality_score"])
            use_confidence_bound = selected_confident and evidence.get("confidence_status") == "established"
            candidate_quality_cost = float(
                evidence["risk_adjusted_quality_cost_score" if use_confidence_bound else "quality_cost_score"]
            )
            if candidate_quality < AUTO_AGENT_ADAPTIVE_QUALITY_FLOOR:
                continue

            quality_delta = round(candidate_quality - selected_quality, 3)
            quality_cost_delta = round(candidate_quality_cost - selected_quality_cost, 3)
            if budget_profile == "max":
                if quality_delta < AUTO_AGENT_ADAPTIVE_QUALITY_DELTA:
                    continue
                rank_score = candidate_quality
            elif selected_has_evidence:
                if (
                    quality_cost_delta < AUTO_AGENT_ADAPTIVE_QUALITY_COST_DELTA
                    and quality_delta < AUTO_AGENT_ADAPTIVE_QUALITY_DELTA
                ):
                    continue
                rank_score = candidate_quality_cost + max(0.0, quality_delta * 0.5)
            else:
                if moving_deeper:
                    if candidate_quality < 0.78:
                        continue
                elif quality_cost_delta < AUTO_AGENT_ADAPTIVE_QUALITY_COST_DELTA:
                    continue
                rank_score = candidate_quality_cost

            candidates.append(
                {
                    "mode": candidate,
                    "direction": "upgrade" if moving_deeper else "downgrade",
                    "rank_score": rank_score,
                    "quality_delta": quality_delta,
                    "quality_cost_delta": quality_cost_delta,
                    "evidence": evidence,
                    "utility_source": (
                        "confidence_lower_bound"
                        if use_confidence_bound
                        else "adaptive_mean"
                    ),
                }
            )

        if not candidates:
            return selected, None
        chosen = max(candidates, key=lambda item: (float(item["rank_score"]), float(item["evidence"]["weighted_count"])))
        if chosen["mode"] == selected:
            return selected, None
        evidence = chosen["evidence"]
        return str(chosen["mode"]), {
            "direction": chosen["direction"],
            "from": selected,
            "to": str(chosen["mode"]),
            "reason": "adaptive_quality_cost_preferred_neighbor",
            "budget_profile": budget_profile,
            "quality_score": evidence.get("quality_score"),
            "quality_cost_score": evidence.get("quality_cost_score"),
            "risk_adjusted_quality_cost_score": evidence.get("risk_adjusted_quality_cost_score"),
            "quality_lower_bound": evidence.get("quality_lower_bound"),
            "quality_upper_bound": evidence.get("quality_upper_bound"),
            "effective_sample_size": evidence.get("effective_sample_size"),
            "confidence_status": evidence.get("confidence_status"),
            "utility_source": chosen.get("utility_source"),
            "quality_delta": chosen["quality_delta"],
            "quality_cost_delta": chosen["quality_cost_delta"],
            "weighted_count": evidence.get("weighted_count"),
            "weighted_net": evidence.get("weighted_net"),
            "decay": evidence.get("decay"),
        }

    def _apply_auto_route_feedback(
        self,
        *,
        selected: str,
        score: int,
        feedback_summary: Dict[str, Any],
        allowed_modes: Sequence[str],
        budget_profile: str = "balanced",
    ) -> tuple[str, Optional[Dict[str, Any]]]:
        mode_scores = feedback_summary.get("mode_scores") if isinstance(feedback_summary, dict) else {}
        if not isinstance(mode_scores, dict):
            return selected, None

        selected_stats = mode_scores.get(selected) if isinstance(mode_scores.get(selected), dict) else {}
        selected_adaptive = selected_stats.get("adaptive") if isinstance(selected_stats.get("adaptive"), dict) else {}
        selected_net = int(selected_stats.get("quality_net", selected_stats.get("net")) or 0)
        selected_negative = int(selected_stats.get("quality_negative", selected_stats.get("negative")) or 0)

        if not bool(feedback_summary.get("used_recent_fallback")) and bool(selected_adaptive.get("preference_signal")):
            preference_direction = str(selected_adaptive.get("preference_direction") or "")
            direction = 1 if preference_direction == "deeper" else -1 if preference_direction == "shallower" else 0
            preferred = self._neighbor_auto_agent_mode(selected, allowed_modes, direction) if direction else selected
            if preferred != selected:
                cost_pressure = float(selected_adaptive.get("weighted_cost_pressure") or 0.0)
                latency_pressure = float(selected_adaptive.get("weighted_latency_pressure") or 0.0)
                if direction > 0:
                    reason = "explicit_feedback_requested_deeper_route"
                elif latency_pressure > cost_pressure:
                    reason = "explicit_feedback_requested_lower_latency_route"
                else:
                    reason = "explicit_feedback_requested_lower_cost_route"
                return preferred, {
                    "direction": "upgrade" if direction > 0 else "downgrade",
                    "from": selected,
                    "to": preferred,
                    "reason": reason,
                    "preference_direction": preference_direction,
                    "weighted_depth_preference": selected_adaptive.get("weighted_depth_preference"),
                    "weighted_cost_pressure": selected_adaptive.get("weighted_cost_pressure"),
                    "weighted_latency_pressure": selected_adaptive.get("weighted_latency_pressure"),
                    "feedback_scope": "prompt_session",
                }

        if (
            not bool(feedback_summary.get("used_recent_fallback"))
            and selected_negative >= 2
            and selected_net <= -2
        ):
            downgraded = self._neighbor_auto_agent_mode(selected, allowed_modes, -1)
            if downgraded != selected:
                return downgraded, {
                    "direction": "downgrade",
                    "from": selected,
                    "to": downgraded,
                    "reason": "recent_session_feedback_rejected_selected_route",
                    "net": selected_net,
                    "negative": selected_negative,
                }

        if (
            selected != "off"
            and not bool(feedback_summary.get("used_recent_fallback"))
            and bool(selected_adaptive.get("regression_signal"))
        ):
            downgraded = self._neighbor_auto_agent_mode(selected, allowed_modes, -1)
            if downgraded != selected:
                return downgraded, {
                    "direction": "downgrade",
                    "from": selected,
                    "to": downgraded,
                    "reason": "recent_weighted_feedback_regression",
                    "weighted_net": selected_adaptive.get("weighted_net"),
                    "weighted_negative": selected_adaptive.get("weighted_negative"),
                    "quality_score": selected_adaptive.get("quality_score"),
                    "recent_negative_rate": selected_adaptive.get("recent_negative_rate"),
                    "decay": selected_adaptive.get("decay"),
                }

        if not bool(feedback_summary.get("used_recent_fallback")):
            adjusted, adaptive_adjustment = self._prefer_adaptive_neighbor_route(
                selected=selected,
                score=score,
                mode_scores=mode_scores,
                allowed_modes=allowed_modes,
                budget_profile=budget_profile,
            )
            if adjusted != selected:
                return adjusted, adaptive_adjustment

        upgraded = self._neighbor_auto_agent_mode(selected, allowed_modes, 1)
        if not bool(feedback_summary.get("used_recent_fallback")) and upgraded != selected:
            upgraded_stats = mode_scores.get(upgraded) if isinstance(mode_scores.get(upgraded), dict) else {}
            upgraded_net = int(upgraded_stats.get("quality_net", upgraded_stats.get("net")) or 0)
            upgraded_positive = int(upgraded_stats.get("quality_positive", upgraded_stats.get("positive")) or 0)
            threshold = int(AUTO_AGENT_POSITIVE_THRESHOLDS.get(upgraded, 99))
            if upgraded_positive >= 2 and upgraded_net >= 2 and score >= threshold:
                return upgraded, {
                    "direction": "upgrade",
                    "from": selected,
                    "to": upgraded,
                    "reason": "recent_session_feedback_preferred_deeper_route",
                    "net": upgraded_net,
                    "positive": upgraded_positive,
                }

        if selected != "off" and budget_profile != "max":
            selected_economics = (
                selected_stats.get("economics")
                if isinstance(selected_stats.get("economics"), dict)
                else {}
            )
            economics = selected_economics
            if isinstance(economics, dict):
                sample_count = int(economics.get("sample_count") or 0)
                relevant_feedback = int(feedback_summary.get("relevant_feedback") or 0)
                used_recent_fallback = bool(feedback_summary.get("used_recent_fallback"))
                avg_cost = float(economics.get("avg_cost_units") or 0.0)
                avg_elapsed_ms = float(economics.get("avg_elapsed_ms") or 0.0)
                budget_limits = {
                    "fast": {"cost": 3.0, "elapsed_ms": 2500.0},
                    "balanced": {"cost": 6.0, "elapsed_ms": 8000.0},
                    "deep": {"cost": 10.0, "elapsed_ms": 15000.0},
                }
                limits = budget_limits.get(budget_profile, budget_limits["balanced"])
                selected_positive = int(selected_stats.get("quality_positive", selected_stats.get("positive")) or 0)
                pressure_reasons: List[str] = []
                if avg_cost >= float(limits["cost"]):
                    pressure_reasons.append("cost")
                if avg_elapsed_ms >= float(limits["elapsed_ms"]):
                    pressure_reasons.append("latency")
                if (
                    sample_count >= 1
                    and relevant_feedback >= 2
                    and not used_recent_fallback
                    and pressure_reasons
                    and selected_positive <= 0
                    and selected_net <= 0
                ):
                    downgraded = self._neighbor_auto_agent_mode(selected, allowed_modes, -1)
                    if downgraded != selected:
                        return downgraded, {
                            "direction": "downgrade",
                            "from": selected,
                            "to": downgraded,
                            "reason": "session_route_economics_exceeded_budget_health",
                            "budget_profile": budget_profile,
                            "economic_pressure": pressure_reasons,
                            "pressure_scope": selected,
                            "sample_count": sample_count,
                            "relevant_feedback": relevant_feedback,
                            "avg_cost_units": round(avg_cost, 3),
                            "avg_elapsed_ms": round(avg_elapsed_ms, 3),
                            "cost_limit": limits["cost"],
                            "elapsed_ms_limit": limits["elapsed_ms"],
                        }

        return selected, None

    def _resolve_auto_agent_mode(
        self,
        *,
        session_id: str,
        prompt: str,
        action_mode: str,
        settings: Dict[str, Any],
        chosen_record: ModelRecord,
    ) -> Dict[str, Any]:
        compute_policy = chat_app.estimate_auto_reasoning_cycles(prompt)
        score = int(compute_policy.get("score") or 0)
        reasons = list(compute_policy.get("reasons") or [])
        lowered = str(prompt or "").lower()

        if bool(settings.get("web_search_enabled", False)) and should_offer_web_search(prompt):
            score += 1
            reasons.append("fresh_knowledge")
        if settings.get("uploaded_image_path"):
            score += 1
            reasons.append("uploaded_artifact")
        if re.search(r"\b(latest|recent|state of the art|sota|paper|github|cite|source)\b", lowered):
            score += 1
            reasons.append("external_evidence")
        if re.search(r"\b(end to end|whole project|multi[- ]agent|sub[- ]agent|regression|production)\b", lowered):
            score += 1
            reasons.append("workflow_depth")

        consultants = self._collective_consultants(settings=settings, chosen_record=chosen_record)
        collective_model_count = len(consultants)
        collective_available = collective_model_count >= 2
        allow_collective = settings.get("auto_agent_collective", True) is not False
        allow_loop = settings.get("auto_agent_loop", True) is not False
        budget_profile = self._normalized_auto_budget_profile(
            settings.get("auto_agent_budget") or settings.get("auto_route_budget") or settings.get("budget_profile")
        )
        budget_policy = dict(AUTO_AGENT_BUDGET_PROFILES[budget_profile])
        score_before_budget = score
        score = max(0, score + int(budget_policy.get("score_bias") or 0))
        if budget_policy.get("score_bias"):
            reasons.append(f"budget_{budget_profile}")

        base_allowed_modes = self._allowed_auto_agent_modes(
            action_mode=action_mode,
            allow_collective=allow_collective,
            collective_available=collective_available,
            allow_loop=allow_loop,
        )
        allowed_modes = self._budget_allowed_auto_agent_modes(
            base_allowed_modes,
            str(budget_policy.get("max_agent_mode") or "collective_loop"),
        )

        if action_mode == "image":
            if allow_collective and collective_available and score >= 3:
                selected = "collective"
                reason = "complex_image_or_multimodal_prompt"
            else:
                selected = "off"
                reason = "image_generation_prefers_single_pass"
        elif score >= 5 and "collective_loop" in allowed_modes:
            selected = "collective_loop"
            reason = "high_complexity_with_collective_capacity"
        elif score >= 4 and "loop" in allowed_modes:
            selected = "loop"
            reason = "high_complexity_iterative_work"
        elif score >= 2 and "collective" in allowed_modes:
            selected = "collective"
            reason = "moderate_complexity_panel_synthesis"
        else:
            selected = "off"
            reason = "low_complexity_single_pass"
        if selected not in allowed_modes:
            selected = self._neighbor_auto_agent_mode(allowed_modes[-1], allowed_modes, 0) if allowed_modes else "off"
            reason = "budget_profile_limited_route_depth"
        base_selected_agent_mode = selected
        uncertainty_signals = self._auto_route_uncertainty_signals(prompt)
        route_confidence = self._auto_route_confidence(
            selected=selected,
            score=score,
            allowed_modes=allowed_modes,
            action_mode=action_mode,
            uncertainty_signals=uncertainty_signals,
        )
        uncertainty_adjustment: Optional[Dict[str, Any]] = None
        adjusted, uncertainty_adjustment, route_confidence = self._maybe_apply_auto_uncertainty_margin(
            selected=selected,
            score=score,
            allowed_modes=allowed_modes,
            action_mode=action_mode,
            budget_profile=budget_profile,
            route_confidence=route_confidence,
            uncertainty_signals=uncertainty_signals,
        )
        if adjusted != selected:
            selected = adjusted
            reason = str(uncertainty_adjustment.get("reason") or reason)
        feedback_summary = self.memory_store.route_feedback_summary(session_id, prompt)
        feedback_adjustment: Optional[Dict[str, Any]] = None
        if int(feedback_summary.get("total_feedback") or 0) > 0:
            adjusted, feedback_adjustment = self._apply_auto_route_feedback(
                selected=selected,
                score=score,
                feedback_summary=feedback_summary,
                allowed_modes=allowed_modes,
                budget_profile=budget_profile,
            )
            if adjusted != selected:
                selected = adjusted
                reason = str(feedback_adjustment.get("reason") or reason)
                route_confidence["feedback_adjusted"] = True
                route_confidence["selected_agent_mode"] = selected

        policy = {
            "mode": "auto",
            "selected_agent_mode": selected,
            "base_selected_agent_mode": base_selected_agent_mode,
            "reason": reason,
            "score": score,
            "score_before_budget": score_before_budget,
            "reasons": reasons or ["simple_prompt"],
            "reasoning_cycles": int(compute_policy.get("cycles") or 0),
            "collective_model_count": collective_model_count,
            "collective_available": collective_available,
            "loop_available": action_mode != "image",
            "budget_profile": budget_profile,
            "budget_policy": budget_policy,
            "allowed_agent_modes": allowed_modes,
            "route_confidence": route_confidence,
            "uncertainty_adjustment": uncertainty_adjustment,
            "feedback_summary": feedback_summary,
            "feedback_adjustment": feedback_adjustment,
        }
        return _stamp_auto_route_policy(
            policy,
            selected_agent_mode=selected,
            action_mode=action_mode,
        )

    def _format_loop_history(self, steps: Sequence[Dict[str, Any]]) -> str:
        if not steps:
            return "No prior loop work yet."
        lines: List[str] = []
        for row in steps:
            lines.append(f"Step {row.get('step')}:")
            if row.get("goal"):
                lines.append(f"Goal: {row['goal']}")
            if row.get("worker_excerpt"):
                lines.append(f"Worker: {row['worker_excerpt']}")
            if row.get("review_note"):
                lines.append(f"Reviewer: {row['review_note']}")
            if row.get("review_score") is not None:
                lines.append(f"Verifier score: {row['review_score']}")
            if row.get("next_step"):
                lines.append(f"Next: {row['next_step']}")
        return "\n".join(lines[-20:])

    def _build_loop_planner_prompt(self, prompt: str, history: str, step_index: int, max_steps: int) -> str:
        return (
            "You are the planner sub-agent inside a loop agent.\n"
            "Break the user's task into the single highest-value next action.\n"
            "Respond using exactly these headings:\n"
            "DONE: yes or no\n"
            "STEP_GOAL: one concise sentence\n"
            "SUCCESS_SIGNAL: how to know the task is complete\n"
            "WORKING_NOTES: short reasoning\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n\n"
            f"Work log:\n{history}"
        )

    def _build_loop_worker_prompt(
        self,
        prompt: str,
        *,
        history: str,
        step_goal: str,
        step_index: int,
        max_steps: int,
    ) -> str:
        return (
            "You are the worker sub-agent inside a loop agent.\n"
            "Execute the next best action for the task. Prefer doing the work over talking about the work.\n"
            "Respond using exactly these headings:\n"
            "DONE: yes or no\n"
            "OUTPUT: the actual deliverable or best current progress\n"
            "NEXT_FOCUS: what should happen next if not done\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n"
            f"Current goal:\n{step_goal}\n\n"
            f"Work log:\n{history}"
        )

    def _build_loop_review_prompt(
        self,
        prompt: str,
        *,
        history: str,
        latest_output: str,
        success_signal: str,
        step_index: int,
        max_steps: int,
    ) -> str:
        return (
            "You are the reviewer sub-agent inside a loop agent.\n"
            "Judge whether the user's task is complete enough to stop.\n"
            "Respond using exactly these headings:\n"
            "COMPLETE: yes or no\n"
            "SCORE: number from 0.0 to 1.0 where 1.0 means fully complete and safe to stop\n"
            "CONFIDENCE: number from 0.0 to 1.0\n"
            "RISK_SCORE: number from 0.0 to 1.0 where 1.0 means high error or missing-work risk\n"
            "FINAL_RESPONSE: the reply that should be shown to the user right now\n"
            "REASON: why you marked it complete or incomplete\n"
            "EVIDENCE: concrete evidence behind the score\n"
            "NEXT_STEP: the next action if it is not complete\n\n"
            f"Original task:\n{prompt}\n\n"
            f"Loop step: {step_index} of {max_steps}\n"
            f"Success signal:\n{success_signal or 'Task is complete when the user request has been fully addressed.'}\n\n"
            f"Work log:\n{history}\n\n"
            f"Latest worker output:\n{latest_output}"
        )

    def _run_loop_agent_text(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        action_mode: str,
        memory_bundle: Dict[str, Any],
        collective_mode: bool,
    ) -> ChatResult:
        controller_record = chosen_record if chosen_record.supports_chat else self._default_text_record()
        max_steps = _coerce_int_setting(
            settings.get("loop_max_steps") or settings.get("loop_budget"),
            LOOP_AGENT_DEFAULT_MAX_STEPS,
            minimum=2,
            maximum=LOOP_AGENT_HARD_MAX_STEPS,
        )
        score_threshold = _loop_score_threshold(
            settings.get("loop_score_threshold") or settings.get("loop_completion_threshold")
        )
        tool_cache = {
            _trim_text(event.query, limit=220).lower(): event
            for event in self._seed_auto_tool_events(prompt, settings)
        }
        tool_event_rows: List[Dict[str, Any]] = [event.to_dict() for event in list(tool_cache.values())]
        consult_rows: List[Dict[str, str]] = []
        consulted_labels: List[str] = []
        skipped_models: List[Dict[str, str]] = []
        loop_steps: List[Dict[str, Any]] = []
        final_response = ""
        completion_reason = ""
        stop_reason_code = "budget_exhausted"
        loop_stop_score: Optional[float] = None
        completed = False

        for step_index in range(1, max_steps + 1):
            history = self._format_loop_history(loop_steps)
            planner_result, planner_events = self._run_text_model(
                controller_record,
                session_id=self._session_scope(session_id, controller_record.key, f"loop-plan-{step_index}"),
                prompt=self._build_loop_planner_prompt(prompt, history, step_index, max_steps),
                settings={
                    **settings,
                    "memory_context": memory_bundle.get("context_block") or "",
                },
                route_reason=f"{route_reason} Loop agent planner step {step_index} via {controller_record.label}.",
                tool_cache=tool_cache,
                allow_tool_calls=True,
            )
            tool_event_rows.extend(event.to_dict() for event in planner_events)
            step_goal = (
                _extract_labeled_section(planner_result.response, "STEP_GOAL")
                or _extract_labeled_section(planner_result.response, "WORKING_NOTES")
                or _trim_text(planner_result.response, limit=220)
                or "Advance the task toward a complete solution."
            )
            success_signal = _extract_labeled_section(planner_result.response, "SUCCESS_SIGNAL")

            if collective_mode:
                worker_result = self._run_agent_text(
                    session_id=self._session_scope(session_id, controller_record.key, f"loop-work-{step_index}"),
                    prompt=self._build_loop_worker_prompt(
                        prompt,
                        history=history,
                        step_goal=step_goal,
                        step_index=step_index,
                        max_steps=max_steps,
                    ),
                    chosen_record=chosen_record,
                    settings=settings,
                    route_reason=f"{route_reason} Loop agent worker step {step_index} used the collective panel.",
                    action_mode=action_mode,
                    memory_bundle=memory_bundle,
                )
                consult_rows.extend(list((worker_result.agent_trace or {}).get("consultation_rows") or []))
                consulted_labels.extend(list((worker_result.agent_trace or {}).get("consulted_models") or []))
                skipped_models.extend(list((worker_result.agent_trace or {}).get("skipped_models") or []))
                tool_event_rows.extend(list((worker_result.agent_trace or {}).get("tool_events") or []))
            else:
                worker_result, worker_events = self._run_text_model(
                    controller_record,
                    session_id=self._session_scope(session_id, controller_record.key, f"loop-work-{step_index}"),
                    prompt=self._build_loop_worker_prompt(
                        prompt,
                        history=history,
                        step_goal=step_goal,
                        step_index=step_index,
                        max_steps=max_steps,
                    ),
                    settings={
                        **settings,
                        "memory_context": memory_bundle.get("context_block") or "",
                    },
                    route_reason=f"{route_reason} Loop agent worker step {step_index} via {controller_record.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
                tool_event_rows.extend(event.to_dict() for event in worker_events)

            worker_output = (
                _extract_labeled_section(worker_result.response, "OUTPUT")
                or _extract_labeled_section(worker_result.response, "FINAL_RESPONSE")
                or _trim_text(worker_result.response, limit=1800)
            )
            reviewer_result, reviewer_events = self._run_text_model(
                controller_record,
                session_id=self._session_scope(session_id, controller_record.key, f"loop-review-{step_index}"),
                prompt=self._build_loop_review_prompt(
                    prompt,
                    history=history,
                    latest_output=worker_output,
                    success_signal=success_signal,
                    step_index=step_index,
                    max_steps=max_steps,
                ),
                settings={
                    **settings,
                    "memory_context": memory_bundle.get("context_block") or "",
                },
                route_reason=f"{route_reason} Loop agent reviewer step {step_index} via {controller_record.label}.",
                tool_cache=tool_cache,
                allow_tool_calls=True,
            )
            tool_event_rows.extend(event.to_dict() for event in reviewer_events)

            review_complete = _parse_yes_no_section(reviewer_result.response, "COMPLETE")
            if review_complete is None:
                review_complete = _parse_yes_no_section(reviewer_result.response, "DONE")
            review_metrics = _loop_review_metrics(reviewer_result.response, review_complete)
            review_reason = (
                _extract_labeled_section(reviewer_result.response, "REASON")
                or _extract_labeled_section(reviewer_result.response, "MISSING")
            )
            next_step = (
                _extract_labeled_section(reviewer_result.response, "NEXT_STEP")
                or _extract_labeled_section(worker_result.response, "NEXT_FOCUS")
                or step_goal
            )
            final_candidate = (
                _extract_labeled_section(reviewer_result.response, "FINAL_RESPONSE")
                or worker_output
                or _trim_text(reviewer_result.response, limit=1800)
            )
            score_stop = (
                review_complete is None
                and review_metrics["review_score"] >= score_threshold
                and review_metrics["risk_score"] <= 0.35
            )
            step_stop_reason = ""
            stop_decision = "continue"
            if review_complete is True:
                stop_decision = "stop"
                step_stop_reason = "reviewer_complete"
            elif score_stop:
                stop_decision = "stop"
                step_stop_reason = "score_threshold"
            elif review_complete is False:
                step_stop_reason = "reviewer_continue"
            else:
                step_stop_reason = "score_below_threshold"

            loop_steps.append(
                {
                    "step": str(step_index),
                    "goal": step_goal,
                    "worker_excerpt": _trim_text(worker_output, limit=320),
                    "review_note": _trim_text(review_reason or "", limit=220),
                    "next_step": _trim_text(next_step, limit=220),
                    "review_complete": review_complete,
                    "review_score": review_metrics["review_score"],
                    "loop_score": review_metrics["review_score"],
                    "verifier_score": review_metrics["verifier_score"],
                    "progress_score": review_metrics["progress_score"],
                    "confidence_score": review_metrics["confidence_score"],
                    "risk_score": review_metrics["risk_score"],
                    "completion_evidence": review_metrics["completion_evidence"],
                    "stop_decision": stop_decision,
                    "stop_reason_code": step_stop_reason,
                }
            )
            final_response = final_candidate
            completion_reason = (
                review_reason
                or review_metrics["completion_evidence"]
                or success_signal
                or f"Loop agent finished step {step_index}."
            )
            loop_stop_score = review_metrics["review_score"]

            if stop_decision == "stop":
                completed = True
                stop_reason_code = step_stop_reason
                break

        if not completed and final_response:
            final_response = (
                f"{final_response}\n\nLoop agent note: the loop budget ended before it confidently marked the task complete."
            )
            completion_reason = completion_reason or "Loop budget reached before completion."
        elif completed and stop_reason_code == "score_threshold":
            completion_reason = completion_reason or (
                f"Reviewer score met the {score_threshold:.2f} completion threshold."
            )

        deduped_consulted = list(dict.fromkeys(label for label in consulted_labels if label))
        result = ChatResult(
            kind="text",
            model_key=controller_record.key,
            model_label=controller_record.label,
            route_reason=(
                f"{route_reason} Loop agent ran {len(loop_steps)} autonomous step(s)"
                + (" with collective worker consultations." if collective_mode else f" on {controller_record.label}.")
            ),
            response=final_response or "Loop agent did not produce a final answer.",
            timing={
                "loop_steps": len(loop_steps),
                "loop_max_steps": max_steps,
                "loop_stop_step": len(loop_steps),
                "loop_stop_score": loop_stop_score,
                "loop_stop_reason_code": stop_reason_code,
            },
            prompt_used=prompt,
        )
        result.agent_trace = {
            "agent_mode": "collective_loop_agent" if collective_mode else "loop_agent",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": deduped_consulted,
            "consultation_rows": consult_rows,
            "skipped_models": skipped_models,
            "tool_events": tool_event_rows,
            "loop_steps": loop_steps,
            "loop_completed": completed,
            "loop_completion_reason": completion_reason,
            "loop_budget": max_steps,
            "loop_score_threshold": score_threshold,
            "loop_stop_reason_code": stop_reason_code,
            "loop_stop_step": len(loop_steps),
            "loop_stop_score": loop_stop_score,
            "loop_controller_model": controller_record.label,
            "loop_worker_mode": "collective" if collective_mode else "single_model",
        }
        return result

    def _run_agent_text(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        action_mode: str,
        memory_bundle: Dict[str, Any],
    ) -> ChatResult:
        tool_events = self._seed_auto_tool_events(prompt, settings)
        tool_cache = {_trim_text(event.query, limit=220).lower(): event for event in tool_events}
        consult_rows: List[Dict[str, str]] = []
        skipped_rows: List[Dict[str, str]] = []
        for consultant in self._collective_consultants(settings=settings, chosen_record=chosen_record):
            consult_settings = dict(settings)
            consult_settings["memory_context"] = memory_bundle.get("context_block") or ""
            try:
                consult_result, new_tools = self._run_text_model(
                    consultant,
                    session_id=self._session_scope(session_id, consultant.key, "consult"),
                    prompt=self._build_consult_prompt(prompt, action_mode=action_mode),
                    settings=consult_settings,
                    route_reason=f"{route_reason} Agent consultation via {consultant.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "model_key": consultant.key,
                        "model_label": consultant.label,
                        "error": _trim_text(str(exc), limit=220),
                    }
                )
                continue
            tool_events.extend(new_tools)
            consult_rows.append(
                {
                    "model_key": consultant.key,
                    "model_label": consultant.label,
                    "response": consult_result.response,
                }
            )

        consultation_context = self._format_consultations(consult_rows)
        synthesis_record = chosen_record if chosen_record.supports_chat else self._default_text_record()
        synthesis_settings = dict(settings)
        synthesis_settings["memory_context"] = memory_bundle.get("context_block") or ""
        synthesis_settings["consultation_context"] = consultation_context
        if tool_cache:
            synthesis_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        synthesis_result, new_tools = self._run_text_model(
            synthesis_record,
            session_id=self._session_scope(session_id, synthesis_record.key, "answer"),
            prompt=self._build_synthesis_prompt(prompt, action_mode=action_mode),
            settings=synthesis_settings,
            route_reason=f"{route_reason} Agent mode consulted {len(consult_rows)} text models before synthesis.",
            tool_cache=tool_cache,
            allow_tool_calls=True,
        )
        tool_events.extend(new_tools)

        synthesis_result.agent_trace = {
            "agent_mode": "collective_panel",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": [row["model_label"] for row in consult_rows],
            "consultation_rows": consult_rows,
            "skipped_models": skipped_rows,
            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
        }
        return synthesis_result

    def _run_agent_image(
        self,
        *,
        session_id: str,
        prompt: str,
        chosen_record: ModelRecord,
        settings: Dict[str, Any],
        route_reason: str,
        memory_bundle: Dict[str, Any],
    ) -> ChatResult:
        if not chosen_record.supports_image:
            raise RuntimeError(f"{chosen_record.label} does not support image generation.")
        tool_events = self._seed_auto_tool_events(prompt, settings)
        tool_cache = {_trim_text(event.query, limit=220).lower(): event for event in tool_events}
        consult_rows: List[Dict[str, str]] = []
        skipped_rows: List[Dict[str, str]] = []
        for consultant in self._collective_consultants(settings=settings, chosen_record=chosen_record):
            consult_settings = dict(settings)
            consult_settings["memory_context"] = memory_bundle.get("context_block") or ""
            try:
                consult_result, new_tools = self._run_text_model(
                    consultant,
                    session_id=self._session_scope(session_id, consultant.key, "consult-image"),
                    prompt=self._build_consult_prompt(prompt, action_mode="image"),
                    settings=consult_settings,
                    route_reason=f"{route_reason} Agent image consultation via {consultant.label}.",
                    tool_cache=tool_cache,
                    allow_tool_calls=True,
                )
            except Exception as exc:
                skipped_rows.append(
                    {
                        "model_key": consultant.key,
                        "model_label": consultant.label,
                        "error": _trim_text(str(exc), limit=220),
                    }
                )
                continue
            tool_events.extend(new_tools)
            consult_rows.append(
                {
                    "model_key": consultant.key,
                    "model_label": consultant.label,
                    "response": consult_result.response,
                }
            )

        planner = self._default_text_record()
        planner_settings = dict(settings)
        planner_settings["memory_context"] = memory_bundle.get("context_block") or ""
        planner_settings["consultation_context"] = self._format_consultations(consult_rows)
        if tool_cache:
            planner_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
        planner_result, new_tools = self._run_text_model(
            planner,
            session_id=self._session_scope(session_id, planner.key, "image-synth"),
            prompt=self._build_synthesis_prompt(prompt, action_mode="image"),
            settings=planner_settings,
            route_reason=f"{route_reason} Agent mode refined the image prompt with {len(consult_rows)} text consultants.",
            tool_cache=tool_cache,
            allow_tool_calls=True,
        )
        tool_events.extend(new_tools)

        _record, backend = self.ensure_backend(chosen_record.key)
        final_settings = dict(settings)
        final_settings["memory_context"] = memory_bundle.get("context_block") or ""
        final_settings["consultation_context"] = self._format_consultations(consult_rows)
        effective_route_reason = (
            f"{route_reason} Agent mode consulted {len(consult_rows)} text models and refined the final image prompt."
        )
        if _record.key != chosen_record.key:
            effective_route_reason = (
                f"{effective_route_reason} Requested {chosen_record.label} could not be initialized, so the system fell "
                f"back to {_record.label}."
            )
        final_settings["route_reason"] = effective_route_reason
        generation_settings = dict(final_settings)
        generation_settings.pop("_route_model_call_counter", None)
        image_result = backend.generate_image(session_id, planner_result.response or prompt, generation_settings)
        _record_route_model_call(
            final_settings,
            max(1, int((image_result.timing or {}).get("model_calls") or 1)),
        )
        image_result.agent_trace = {
            "agent_mode": "collective_panel",
            "memory_notes": list(memory_bundle.get("memory_notes") or []),
            "consulted_models": [row["model_label"] for row in consult_rows],
            "consultation_rows": consult_rows,
            "skipped_models": skipped_rows,
            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
            "planner_model": planner.label,
        }
        image_result.refined_prompt = planner_result.response or prompt
        return image_result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            active = self.record_map.get(self._backend_key) if self._backend_key else None
            return {
                "selected_model_key": self.selected_model_key,
                "active_model_key": active.key if active else "",
                "active_model_label": active.label if active else "",
                "last_route_reason": self.last_route_reason,
                "models_available": len(self.records),
                "device": self.device_info.get("resolved", str(self.device)),
                "generated_dir": str(self.generated_dir),
                "uploads_dir": str(self.uploads_dir),
                "exports_dir": str(self.exports_dir),
                "extraction_root": str(self.extraction_root),
                "memory_status": self.memory_store.global_status(),
                "route_policy_ledger": self.route_policy_ledger.report(),
                "active_backend_status": self._backend.status() if self._backend is not None else None,
            }

    def session_memory_snapshot(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            return self.memory_store.session_snapshot(session_id)

    def record_route_feedback(self, *, session_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            route_id = str(feedback.get("route_id") or "").strip()
            try:
                durable_decision = self.route_policy_ledger.get_decision(route_id)
            except (DecisionNotFoundError, KeyError):
                durable_decision = None
            if durable_decision is not None:
                if durable_decision.get("session_hash") != hash_session_identity(session_id):
                    raise ValueError("route_id does not belong to this session")
                if durable_decision.get("status") != "completed":
                    raise ValueError("feedback requires a successfully completed route")
            if durable_decision is None:
                # Legacy feedback can predate the durable decision ledger. It
                # remains descriptive in the JSON compatibility store and is
                # never eligible for readiness certification.
                result = self.memory_store.add_feedback(session_id=session_id, feedback=feedback)
                result["durable_feedback"] = {
                    "status": "legacy_unjoined",
                    "eligible_for_readiness": False,
                }
                return result

            row = self.memory_store.prepare_feedback(session_id=session_id, feedback=feedback)
            durable_payload = {
                "rating": row.get("rating"),
                "feedback_intent": row.get("feedback_intent"),
                "feedback_tags": list(row.get("feedback_tags") or []),
                "feedback_axes": dict(row.get("feedback_axes") or {}),
                "reason": row.get("reason"),
                "source": "explicit_user_route_feedback",
                "observation_status": "observed",
            }
            # Commit the source-of-truth row first. Its content-derived
            # idempotency key makes an identical retry return the same revision.
            durable_feedback = self.route_policy_ledger.record_feedback(route_id, durable_payload)
            durable_revision = max(1, int(durable_feedback.get("revision") or 1))
            try:
                result = self.memory_store.commit_feedback(
                    session_id=session_id,
                    feedback_row=row,
                    feedback_revision=durable_revision,
                )
            except Exception as exc:
                # A compatibility-mirror failure must not turn an already
                # durable user acknowledgement into an API failure. An
                # identical retry safely reconciles the pinned revision.
                logging.exception("Route feedback compatibility mirror is pending reconciliation")
                accepted_row = dict(row)
                accepted_row.update(
                    {
                        "feedback_revision": durable_revision,
                        "durable_feedback_revision": durable_revision,
                    }
                )
                result = {
                    "ok": True,
                    "feedback": accepted_row,
                    "summary": None,
                    "compatibility_mirror": {
                        "status": "pending_reconciliation",
                        "durable_revision": durable_revision,
                        "error_category": type(exc).__name__,
                    },
                }
            result["durable_feedback"] = {
                "status": "committed",
                "revision": durable_revision,
                "idempotent": bool(durable_feedback.get("idempotent")),
                "eligible_for_readiness": True,
            }
            return result

    def route_health_snapshot(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            summary = self.memory_store.route_feedback_summary(session_id)
            summary["route_usage"] = self.memory_store.route_usage_summary(session_id)
            return summary

    def route_shadow_registry_snapshot(self) -> Dict[str, Any]:
        """Return the isolated shadow-assignment registry without changing it."""

        registry_path = self.route_shadow_registry_path
        with self._lock:
            if not registry_path.is_file():
                self._route_shadow_registry_cache_signature = None
                self._route_shadow_registry_cache_snapshot = None
                return {
                    "ok": True,
                    "available": False,
                    "status": "not_initialized",
                    "registry_location": f"memory/{registry_path.name}",
                    "read_only": True,
                    "campaign_count": 0,
                    "campaigns": [],
                    "event_chain": None,
                    "execution_enabled": False,
                    "activation_available": False,
                    "automatic_promotion_allowed": False,
                }

            signature_before = self._route_shadow_registry_signature(registry_path)
            if (
                signature_before == self._route_shadow_registry_cache_signature
                and self._route_shadow_registry_cache_snapshot is not None
            ):
                return copy.deepcopy(self._route_shadow_registry_cache_snapshot)

            snapshot = RouteShadowAssignmentRegistry(registry_path, read_only=True).snapshot()
            result = {
                **snapshot,
                "available": True,
                "status": "verified" if snapshot.get("ok") else "verification_failed",
                "registry_location": f"memory/{registry_path.name}",
                "read_only": True,
            }
            signature_after = self._route_shadow_registry_signature(registry_path)
            if signature_before == signature_after:
                self._route_shadow_registry_cache_signature = signature_after
                self._route_shadow_registry_cache_snapshot = copy.deepcopy(result)
            else:
                # A concurrent registry write may have landed during the audit.
                # The transactional snapshot is still coherent, but do not reuse it.
                self._route_shadow_registry_cache_signature = None
                self._route_shadow_registry_cache_snapshot = None
            return result

    @staticmethod
    def _route_shadow_registry_signature(
        registry_path: Path,
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Fingerprint durable SQLite state without reading or mutating the registry."""

        rows: List[Tuple[Any, ...]] = []
        for path in (registry_path, Path(f"{registry_path}-wal")):
            try:
                stat = path.stat()
            except FileNotFoundError:
                rows.append((False,))
            else:
                # SQLite may materialize an empty WAL while opening a read-only
                # connection. It carries no committed frames, so normalize it
                # to the same signature as an absent WAL.
                if path != registry_path and stat.st_size == 0:
                    rows.append((False,))
                    continue
                rows.append(
                    (
                        True,
                        stat.st_dev,
                        stat.st_ino,
                        stat.st_size,
                        stat.st_mtime_ns,
                        stat.st_ctime_ns,
                    )
                )
        return tuple(rows)

    def route_policy_lab_snapshot(self, session_id: str, profile: str = "balanced") -> Dict[str, Any]:
        with self._lock:
            payload = self.memory_store.load_session(session_id)
            durable_evidence = self.route_policy_ledger.policy_evidence_snapshot(
                session_id=session_id,
                policy_name=AUTO_ROUTE_POLICY_ID,
                policy_version=AUTO_ROUTE_POLICY_VERSION,
                limit=1000,
            )
            durable_usage = list(durable_evidence.get("usage_rows") or [])
            durable_feedback = list(durable_evidence.get("feedback_rows") or [])
            if durable_usage:
                usage_rows = durable_usage
                feedback_rows = durable_feedback
                expected_contexts = durable_evidence.get("expected_context_by_route_id")
                evidence_source = "durable_sqlite_ledger"
                lifecycle_source: Optional[Mapping[str, Any]] = durable_evidence
            else:
                usage_rows = list(payload.get("route_usage") or [])
                feedback_rows = list(payload.get("route_feedback") or [])
                expected_contexts = None
                evidence_source = "legacy_json_compatibility"
                lifecycle_source = None
            report = analyze_route_policy(
                usage_rows,
                feedback_rows,
                profile=profile,
                expected_policy_id=AUTO_ROUTE_POLICY_ID,
                expected_policy_version=AUTO_ROUTE_POLICY_VERSION,
                expected_feature_schema=AUTO_ROUTE_FEATURE_SCHEMA_VERSION,
                expected_context_by_route_id=(
                    expected_contexts if isinstance(expected_contexts, Mapping) else None
                ),
                durable_lifecycle=lifecycle_source,
            )
            report["evidence_source"] = evidence_source
            report["durable_evidence_window"] = durable_evidence.get("analysis_window")
            outcome_contract_maturity = durable_evidence.get("outcome_contract_maturity")
            report["outcome_contract_maturity"] = (
                dict(outcome_contract_maturity)
                if isinstance(outcome_contract_maturity, Mapping)
                else {}
            )
            report["compatibility_view"] = {
                "usage_rows": len(payload.get("route_usage") or []),
                "feedback_rows": len(payload.get("route_feedback") or []),
                "used_for_analysis": not bool(durable_usage),
                "used_for_readiness": False,
                "eligible_for_readiness": False,
            }
            report["available_profiles"] = [item.as_dict() for item in POLICY_PROFILES.values()]
            report["active_logging_policy"] = {
                "policy_id": AUTO_ROUTE_POLICY_ID,
                "policy_version": AUTO_ROUTE_POLICY_VERSION,
                "feature_schema_version": AUTO_ROUTE_FEATURE_SCHEMA_VERSION,
                "decision_type": "deterministic",
                "counterfactual_support": "none",
                "support_schema_version": SUPPORT_SCHEMA_VERSION,
                "failure_observation": "two_phase_started_and_terminal",
                "durable_store": "sqlite_wal",
                "retention": {
                    "durable_decisions": "uncapped",
                    "compatibility_usage_rows": 240,
                    "compatibility_feedback_rows": 120,
                },
            }
            durable = durable_evidence.get("lifecycle") or self.route_policy_ledger.report(
                session_id=session_id,
                policy_name=AUTO_ROUTE_POLICY_ID,
                policy_version=AUTO_ROUTE_POLICY_VERSION,
            )
            report["durable_ledger"] = durable
            report["warnings"].append(
                "Durable failure counts are descriptive runtime diagnostics; they do not create counterfactual support."
            )
            report["warnings"].append(
                "Outcome-contract maturity is diagnostic-only telemetry; it is not a policy-value estimator and "
                "cannot authorize route promotion."
            )
            if int((durable.get("counts") or {}).get("inflight") or 0) > 0:
                report["warnings"].append(
                    "In-flight rows can include active work or interrupted processes and require terminal reconciliation."
                )
            return report

    def three_d_model_view(self) -> Dict[str, Any]:
        record = self.record_map.get("three_d_generation_micro_v1")
        if record is None:
            raise FileNotFoundError("three_d_generation_micro_v1 is not available in the local catalog.")
        extracted_dir = _extract_zip_once(record.zip_path, self.extraction_root)
        summary_path = _find_matching_file(
            extracted_dir,
            ("three_d_generation_micro_v1_summary.json",),
            "_summary.json",
        )
        meta_path = _find_matching_file(
            extracted_dir,
            ("three_d_generation_micro_v1_meta.json",),
            ".json",
        )
        summary_payload: Dict[str, Any] = {}
        meta_payload: Dict[str, Any] = {}
        if summary_path is not None:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if meta_path is not None:
            meta_payload = json.loads(meta_path.read_text(encoding="utf-8"))
        row_summary = summary_payload.get("row_summary") if isinstance(summary_payload.get("row_summary"), dict) else {}
        sample_predictions = summary_payload.get("sample_predictions") if isinstance(summary_payload.get("sample_predictions"), list) else []
        return {
            "key": record.key,
            "label": record.label,
            "zip_path": str(record.zip_path.resolve()),
            "zip_name": record.zip_path.name,
            "zip_size_bytes": record.zip_path.stat().st_size,
            "summary_path": str(summary_path.resolve()) if summary_path is not None else "",
            "summary_name": summary_path.name if summary_path is not None else "",
            "meta_path": str(meta_path.resolve()) if meta_path is not None else "",
            "meta_name": meta_path.name if meta_path is not None else "",
            "parameter_count": summary_payload.get("parameter_count"),
            "train_accuracy": summary_payload.get("train_accuracy"),
            "val_accuracy": summary_payload.get("val_accuracy"),
            "concept_count": row_summary.get("concepts"),
            "source_rows": row_summary.get("source_rows"),
            "train_rows": row_summary.get("train_rows"),
            "val_rows": row_summary.get("val_rows"),
            "concept_labels": list(meta_payload.get("labels") or []),
            "sample_predictions": sample_predictions[:6],
        }

    def select_model(self, model_key: str, eager: bool = False) -> Dict[str, Any]:
        if model_key != "auto" and model_key not in self.record_map:
            raise KeyError(f"Unknown model key: {model_key}")
        self.selected_model_key = model_key
        if eager and model_key != "auto":
            record, _backend = self.ensure_backend(model_key)
            self.last_route_reason = f"Loaded {record.label}."
        return self.status()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self.memory_store.clear_session(session_id)
            if self._backend is not None:
                self._backend.clear(session_id)
                self._backend.clear(self._session_scope(session_id, self._backend_key, "consult"))
                self._backend.clear(self._session_scope(session_id, self._backend_key, "answer"))
            session_upload_dir = self.uploads_dir / _safe_slug(session_id)
            if session_upload_dir.exists():
                for child in sorted(session_upload_dir.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    else:
                        try:
                            child.rmdir()
                        except OSError:
                            pass
                try:
                    session_upload_dir.rmdir()
                except OSError:
                    pass

    def store_uploaded_image(self, *, session_id: str, filename: str, raw_bytes: bytes) -> Dict[str, Any]:
        with self._lock:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            session_dir = self.uploads_dir / _safe_slug(session_id)
            session_dir.mkdir(parents=True, exist_ok=True)
            safe_name = _safe_upload_name(filename)
            target = session_dir / f"{Path(safe_name).stem}.png"
            image.save(target, format="PNG")
            return {
                "ok": True,
                "saved_path": str(target),
                "image_url": f"/uploads/{_safe_slug(session_id)}/{target.name}",
                "filename": target.name,
            }

    def save_generated_image(self, *, source_path: str, destination_hint: str = "") -> Dict[str, Any]:
        with self._lock:
            target = copy_generated_image(source_path, destination_hint, self.exports_dir / "saved_images")
            return {
                "ok": True,
                "saved_path": str(target),
            }

    def export_chat_image(
        self,
        *,
        session_id: str,
        transcript: Sequence[Dict[str, object]],
        destination_hint: str = "",
    ) -> Dict[str, Any]:
        with self._lock:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            default_name = f"supermix_chat_{stamp}.png"
            target = render_chat_transcript_image(
                transcript,
                destination_hint=destination_hint or str(self.exports_dir / "chat_images" / default_name),
                default_dir=self.exports_dir / "chat_images",
                session_id=session_id,
            )
            return {
                "ok": True,
                "saved_path": str(target),
            }

    @staticmethod
    def _route_error_category(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, MemoryError):
            return "resource_exhausted"
        if isinstance(exc, FileNotFoundError):
            return "model_artifact_missing"
        if isinstance(exc, PermissionError):
            return "permission_error"
        if isinstance(exc, (KeyError, ValueError)):
            return "validation_error"
        return "runtime_error"

    def handle_prompt(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._route_execution.route_id = ""
        self._route_execution.ledger_started = False
        self._route_execution.started_at_perf = None
        try:
            return self._handle_prompt_impl(
                session_id=session_id,
                prompt=prompt,
                model_key=model_key,
                action_mode=action_mode,
                settings=settings,
            )
        except Exception as exc:
            route_id = str(getattr(self._route_execution, "route_id", "") or "")
            if route_id and bool(getattr(self._route_execution, "ledger_started", False)):
                try:
                    failure_economics: Optional[Dict[str, float]] = None
                    started_at_perf = getattr(self._route_execution, "started_at_perf", None)
                    if (
                        isinstance(started_at_perf, (int, float))
                        and not isinstance(started_at_perf, bool)
                        and math.isfinite(float(started_at_perf))
                    ):
                        completed_at_perf = time.perf_counter()
                        if math.isfinite(completed_at_perf) and completed_at_perf >= float(started_at_perf):
                            failure_economics = {
                                "elapsed_ms": round(
                                    (completed_at_perf - float(started_at_perf)) * 1000.0,
                                    2,
                                )
                            }
                    self.route_policy_ledger.complete_decision(
                        route_id,
                        success=False,
                        executed_mode=str(getattr(self._route_execution, "selected_mode", "") or "") or None,
                        actual_economics=failure_economics,
                        error_category=self._route_error_category(exc),
                        error_message=str(exc)[:500],
                    )
                except Exception:
                    logging.exception("Route decision failure completion could not be persisted")
            raise
        finally:
            self._route_execution.route_id = ""
            self._route_execution.ledger_started = False
            self._route_execution.selected_mode = ""
            self._route_execution.started_at_perf = None

    def _handle_prompt_impl(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            route_started = time.perf_counter()
            settings = dict(settings or {})
            route_model_call_counter = {"count": 0}
            settings["_route_model_call_counter"] = route_model_call_counter
            settings.setdefault("memory_enabled", True)
            settings.setdefault("agent_mode", "off")
            settings.setdefault("grounding_intelligence", True)
            settings.setdefault("web_search_enabled", False)
            settings.setdefault("cmd_open_enabled", True)
            settings.setdefault("web_search_budget", 3)
            settings.setdefault("web_search_results", 5)
            settings.setdefault("loop_max_steps", LOOP_AGENT_DEFAULT_MAX_STEPS)
            memory_bundle = self._prepare_memory_bundle(session_id, prompt, settings)
            memory_raw = memory_bundle.get("raw")
            raw_turns = (
                memory_raw.get("turns")
                if isinstance(memory_raw, Mapping)
                else ()
            )
            recent_turns = [
                turn
                for turn in (
                    raw_turns[-4:]
                    if isinstance(raw_turns, (list, tuple))
                    else ()
                )
                if isinstance(turn, Mapping)
            ]
            recent_user_messages = [
                str(turn.get("user") or "")
                for turn in recent_turns
                if str(turn.get("user") or "").strip()
            ]
            recent_assistant_messages = [
                str(turn.get("assistant") or "")
                for turn in recent_turns
                if str(turn.get("assistant") or "").strip()
            ]
            prompt_profile = analyze_prompt(
                prompt,
                recent_turns=recent_turns,
                recent_user_messages=recent_user_messages,
                recent_assistant_messages=recent_assistant_messages,
            )
            settings["_prompt_profile"] = prompt_profile

            # Studio persists every turn of the session but only ever handed the
            # last four to the planner, so nothing here accumulated. The state
            # is built once per prompt from the durable turn log and passed to
            # whichever backend runs, rather than each backend deriving its own
            # from whatever short window it happens to keep.
            conversation_enabled = bool(settings.get("conversation_intelligence", True))
            conversation_state: Optional[Dict[str, Any]] = None
            if conversation_enabled:
                conversation_state = build_conversation_state(
                    raw_turns if isinstance(raw_turns, (list, tuple)) else (),
                    current_user_text=prompt,
                )
                settings["_conversation_state"] = conversation_state
            settings["_conversation_enabled"] = conversation_enabled

            requested_key = model_key or self.selected_model_key or "auto"
            if requested_key == "auto":
                chosen_record, route_reason = choose_auto_model(
                    self.records,
                    prompt,
                    action_mode=action_mode,
                    uploaded_image_path=str(settings.get("uploaded_image_path") or ""),
                    prompt_profile=prompt_profile,
                )
                if chosen_record is None:
                    raise RuntimeError("No local models were discovered.")
            else:
                chosen_record = self.record_map.get(requested_key)
                if chosen_record is None:
                    raise KeyError(f"Unknown model key: {requested_key}")
                route_reason = f"Manual selection kept {chosen_record.label}."

            resolved_action = action_mode
            if resolved_action == "auto":
                if chosen_record.supports_image and not chosen_record.supports_chat:
                    resolved_action = "image"
                else:
                    resolved_action = "text"

            interaction_enabled = bool(
                settings.get("interaction_intelligence", True)
            )
            interaction_plan: Optional[Dict[str, Any]] = None
            if interaction_enabled:
                interaction_plan = plan_interaction(
                    prompt,
                    recent_assistant_messages=recent_assistant_messages,
                    context={
                        "recent_user_messages": recent_user_messages,
                        "recent_turns": recent_turns,
                    },
                    prompt_profile=prompt_profile,
                )
                settings["_interaction_plan"] = interaction_plan
                settings["_interaction_user_text"] = prompt
            grounding_plan = (
                plan_grounding(
                    prompt,
                    interaction_plan=interaction_plan,
                    prompt_profile=prompt_profile,
                )
                if bool(settings.get("grounding_intelligence", True))
                else None
            )
            agent_mode = self._normalized_agent_mode(settings.get("agent_mode"))
            requested_agent_mode = agent_mode
            auto_agent_policy: Optional[Dict[str, Any]] = None
            if agent_mode == "auto":
                auto_agent_policy = self._resolve_auto_agent_mode(
                    session_id=session_id,
                    prompt=prompt,
                    action_mode=resolved_action,
                    settings=settings,
                    chosen_record=chosen_record,
                )
                if settings.get("reasoning_cycles") in (None, "", 0, "0", "model", "default"):
                    settings["reasoning_cycles"] = int(auto_agent_policy.get("reasoning_cycles") or 1)
                if settings.get("adaptive_compute") is None:
                    settings["adaptive_compute"] = True
                auto_agent_policy["runtime_compute_request"] = {
                    "reasoning_cycles": settings.get("reasoning_cycles"),
                    "adaptive_compute": bool(settings.get("adaptive_compute")),
                    "source": "auto_route_policy",
                }
                agent_mode = str(auto_agent_policy["selected_agent_mode"])
                settings["agent_mode"] = agent_mode
                budget_profile = str(auto_agent_policy.get("budget_profile") or "balanced")
                budget_phrase = "" if budget_profile == "balanced" else f" under the {budget_profile} budget profile"
                route_reason = (
                    f"{route_reason} Auto orchestration selected {agent_mode} "
                    f"because {auto_agent_policy['reason']}{budget_phrase}."
                )
            effective_agent_mode = self._effective_agent_mode_for_action(agent_mode, resolved_action)
            if effective_agent_mode != agent_mode:
                previous_agent_mode = agent_mode
                agent_mode = effective_agent_mode
                settings["agent_mode"] = agent_mode
                if auto_agent_policy is not None:
                    auto_agent_policy["pre_action_agent_mode"] = previous_agent_mode
                    auto_agent_policy["selected_agent_mode"] = agent_mode
                    route_confidence = auto_agent_policy.get("route_confidence")
                    if isinstance(route_confidence, dict):
                        route_confidence["selected_agent_mode"] = agent_mode
                        route_confidence["action_adjusted"] = True
                route_reason = (
                    f"{route_reason} Loop agent currently supports text and vision replies; "
                    f"image generation used {agent_mode} instead of {previous_agent_mode}."
                )
            route_economics_estimate = self._estimate_route_economics(
                selected_agent_mode=agent_mode,
                action_mode=resolved_action,
                settings=settings,
                auto_agent_policy=auto_agent_policy,
            )
            if auto_agent_policy is not None:
                budget_limit = self._auto_session_budget_limit(settings)
                budget_snapshot = self._session_route_budget_snapshot(session_id, budget_limit)
                adjusted_agent_mode, route_economics_estimate, session_budget_adjustment, session_budget = (
                    self._apply_auto_session_budget(
                        selected=agent_mode,
                        action_mode=resolved_action,
                        settings=settings,
                        auto_agent_policy=auto_agent_policy,
                        budget_snapshot=budget_snapshot,
                    )
                )
                if session_budget is not None:
                    auto_agent_policy["session_budget"] = session_budget
                if session_budget_adjustment is not None:
                    previous_agent_mode = agent_mode
                    agent_mode = adjusted_agent_mode
                    settings["agent_mode"] = agent_mode
                    auto_agent_policy["pre_session_budget_agent_mode"] = previous_agent_mode
                    auto_agent_policy["selected_agent_mode"] = agent_mode
                    auto_agent_policy["reason"] = str(session_budget_adjustment.get("reason") or auto_agent_policy["reason"])
                    auto_agent_policy["session_budget_adjustment"] = session_budget_adjustment
                    route_confidence = auto_agent_policy.get("route_confidence")
                    if isinstance(route_confidence, dict):
                        route_confidence["session_budget_adjusted"] = True
                        route_confidence["selected_agent_mode"] = agent_mode
                    if previous_agent_mode == agent_mode:
                        route_reason = (
                            f"{route_reason} Session budget is exhausted; Auto kept {agent_mode} "
                            "because single-pass execution is the minimum route."
                        )
                    else:
                        if session_budget_adjustment.get("reason") == "session_route_budget_target_pacing":
                            route_reason = (
                                f"{route_reason} Session budget target pacing downgraded {previous_agent_mode} "
                                f"to {agent_mode} to preserve budget for the configured route horizon."
                            )
                        else:
                            route_reason = (
                                f"{route_reason} Session budget pacing downgraded {previous_agent_mode} to {agent_mode} "
                                f"because the estimated route cost would exceed the remaining budget."
                            )
                else:
                    auto_agent_policy["session_budget_adjustment"] = None
                auto_agent_policy["route_economics_estimate"] = dict(route_economics_estimate)
                logging_support = self._build_post_filter_logging_support(
                    selected=agent_mode,
                    action_mode=resolved_action,
                    settings=settings,
                    auto_agent_policy=auto_agent_policy,
                    selected_estimate=route_economics_estimate,
                )
                _stamp_auto_route_policy(
                    auto_agent_policy,
                    selected_agent_mode=agent_mode,
                    action_mode=resolved_action,
                    logging_support=logging_support,
                )
            route_id = uuid.uuid4().hex
            ledger_policy = auto_agent_policy if isinstance(auto_agent_policy, dict) else {}
            ledger_context = dict(ledger_policy.get("decision_context") or {})
            ledger_context.update(
                {
                    "action_mode": resolved_action,
                    "requested_model_key": requested_key,
                    "selected_model_key": chosen_record.key,
                    "requested_agent_mode": requested_agent_mode,
                    "selected_agent_mode": agent_mode,
                    "feedback_observation_status": "pending",
                }
            )
            eligible_modes = list(
                ledger_policy.get("eligible_agent_modes")
                or ledger_policy.get("eligible_actions")
                or [agent_mode]
            )
            action_probabilities = dict(
                ledger_policy.get("post_filter_action_probabilities")
                or ledger_policy.get("action_probabilities")
                or {agent_mode: 1.0}
            )
            self.route_policy_ledger.begin_decision(
                session_id=session_id,
                policy_name=str(ledger_policy.get("policy_id") or "explicit-route-v1"),
                policy_version=str(ledger_policy.get("policy_version") or "1.0.0"),
                policy_schema_version=str(
                    ledger_policy.get("feature_schema_version") or AUTO_ROUTE_FEATURE_SCHEMA_VERSION
                ),
                decision_context=ledger_context,
                eligible_modes=eligible_modes,
                chosen_mode=agent_mode,
                action_probabilities=action_probabilities,
                logging_support=(
                    ledger_policy.get("logging_support")
                    if isinstance(ledger_policy.get("logging_support"), dict)
                    else None
                ),
                estimated_economics=route_economics_estimate,
                outcome_contracts=build_route_outcome_contracts(commitment_source="caller"),
                route_id=route_id,
            )
            self._route_execution.route_id = route_id
            self._route_execution.ledger_started = True
            self._route_execution.selected_mode = agent_mode
            self._route_execution.started_at_perf = route_started
            collective_enabled = agent_mode in {"collective", "collective_loop"}
            loop_enabled = agent_mode in {"loop", "collective_loop"}

            if loop_enabled:
                result = self._run_loop_agent_text(
                    session_id=session_id,
                    prompt=prompt,
                    chosen_record=chosen_record,
                    settings=settings,
                    route_reason=route_reason,
                    action_mode=resolved_action,
                    memory_bundle=memory_bundle,
                    collective_mode=collective_enabled,
                )
            elif collective_enabled:
                if resolved_action == "image":
                    result = self._run_agent_image(
                        session_id=session_id,
                        prompt=prompt,
                        chosen_record=chosen_record,
                        settings=settings,
                        route_reason=route_reason,
                        memory_bundle=memory_bundle,
                    )
                else:
                    result = self._run_agent_text(
                        session_id=session_id,
                        prompt=prompt,
                        chosen_record=chosen_record,
                        settings=settings,
                        route_reason=route_reason,
                        action_mode=resolved_action,
                        memory_bundle=memory_bundle,
                    )
            else:
                base_settings = dict(settings)
                base_settings["memory_context"] = memory_bundle.get("context_block") or ""
                base_settings["route_reason"] = route_reason
                tool_cache = {
                    _trim_text(event.query, limit=220).lower(): event
                    for event in self._seed_auto_tool_events(prompt, settings)
                }
                if tool_cache:
                    base_settings["tool_context"] = format_tool_results(list(tool_cache.values()))
                record, backend = self.ensure_backend(chosen_record.key)
                effective_route_reason = route_reason
                if record.key != chosen_record.key:
                    effective_route_reason = (
                        f"{route_reason} Requested {chosen_record.label} could not be initialized, so the system fell "
                        f"back to {record.label}."
                    )
                base_settings["route_reason"] = effective_route_reason
                if resolved_action == "image":
                    if not record.supports_image:
                        raise RuntimeError(f"{record.label} does not support image generation.")
                    generation_settings = dict(base_settings)
                    generation_settings.pop("_route_model_call_counter", None)
                    result = backend.generate_image(session_id, prompt, generation_settings)
                    _record_route_model_call(
                        base_settings,
                        max(1, int((result.timing or {}).get("model_calls") or 1)),
                    )
                    result.agent_trace = {
                        "agent_mode": "off",
                        "memory_notes": list(memory_bundle.get("memory_notes") or []),
                        "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                    }
                else:
                    if not record.supports_chat and record.supports_image:
                        generation_settings = dict(base_settings)
                        generation_settings.pop("_route_model_call_counter", None)
                        result = backend.generate_image(session_id, prompt, generation_settings)
                        _record_route_model_call(
                            base_settings,
                            max(1, int((result.timing or {}).get("model_calls") or 1)),
                        )
                        result.agent_trace = {
                            "agent_mode": "off",
                            "memory_notes": list(memory_bundle.get("memory_notes") or []),
                            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                        }
                    else:
                        result, _new_tools = self._run_text_model(
                            record,
                            session_id=self._session_scope(session_id, record.key, "model"),
                            prompt=prompt,
                            settings=base_settings,
                            route_reason=effective_route_reason,
                            tool_cache=tool_cache,
                            allow_tool_calls=bool(settings.get("web_search_enabled", False)),
                        )
                        result.agent_trace = {
                            "agent_mode": "off",
                            "memory_notes": list(memory_bundle.get("memory_notes") or []),
                            "tool_events": [event.to_dict() for event in list(tool_cache.values())],
                        }

            grounding_guard: Optional[Dict[str, Any]] = None
            if grounding_plan is not None:
                trace_before_grounding = dict(result.agent_trace or {})
                evidence_rows: List[Dict[str, Any]] = []
                for event in trace_before_grounding.get("tool_events") or []:
                    if not isinstance(event, Mapping):
                        continue
                    event_source = str(event.get("source") or event.get("name") or "web_search")
                    for row in event.get("results") or []:
                        if not isinstance(row, Mapping):
                            continue
                        evidence_rows.append(
                            {
                                "title": str(row.get("title") or ""),
                                "text": str(row.get("snippet") or row.get("text") or ""),
                                "url": str(row.get("url") or row.get("href") or ""),
                                "source": event_source,
                                "source_type": "web_snippet",
                                "domain": str(row.get("domain") or ""),
                                "trust_tier": "web_snippet",
                            }
                        )
                grounding_bundle = build_evidence_bundle(
                    prompt,
                    evidence_rows,
                    interaction_plan=interaction_plan,
                    max_items=int(grounding_plan.get("max_evidence_items") or 6),
                    grounding_plan=grounding_plan,
                    prompt_profile=prompt_profile,
                )
                if result.kind == "text":
                    grounding_guard = finalize_grounded_response(
                        result.response,
                        prompt,
                        grounding_plan=grounding_plan,
                        evidence_bundle=grounding_bundle,
                        prompt_profile=prompt_profile,
                    )
                    result.response = str(grounding_guard["text"])
                trace_before_grounding["grounding"] = {
                    "schema_version": str(grounding_bundle.get("schema_version") or ""),
                    "plan": grounding_plan,
                    "source_ids": [
                        str(item.get("id") or "")
                        for item in grounding_bundle.get("evidence") or []
                    ],
                    "sources": [
                        {
                            "id": str(item.get("id") or ""),
                            "title": str(item.get("title") or ""),
                            "url": str(item.get("url") or ""),
                            "domain": str(item.get("domain") or ""),
                            "source_type": str(item.get("source_type") or ""),
                        }
                        for item in grounding_bundle.get("evidence") or []
                    ],
                    "diagnostics": dict(
                        (grounding_guard or {}).get("grounding")
                        or grounding_bundle.get("diagnostics")
                        or {}
                    ),
                    "response_guard": {
                        "changed": bool((grounding_guard or {}).get("changed", False)),
                        "reason": str(
                            (grounding_guard or {}).get("reason")
                            or ("non_text_result" if result.kind != "text" else "audit_only")
                        ),
                    },
                    # Prompt-free reasoning metadata: class, verification, and
                    # budget only. It carries no routing or compute authority.
                    "reasoning": reasoning_diagnostics(
                        (grounding_guard or {}).get("reasoning")
                    ),
                    "authority": dict(
                        (grounding_guard or {}).get("authority")
                        or grounding_plan.get("authority")
                        or {}
                    ),
                }
                result.agent_trace = trace_before_grounding

            if interaction_plan is not None:
                result = finalize_chat_result_for_interaction(
                    result,
                    user_text=prompt,
                    interaction_plan=interaction_plan,
                    relevance_context=str(memory_bundle.get("context_block") or ""),
                )
            elapsed_ms = (time.perf_counter() - route_started) * 1000.0
            trace = dict(result.agent_trace or {})
            understanding_diag = prompt_understanding_diagnostics(prompt_profile)
            if result.kind == "text":
                understanding_diag["response_constraint_audit"] = (
                    evaluate_response_constraints(
                        result.response,
                        prompt,
                        prompt_profile,
                    )
                )
            trace["prompt_understanding"] = understanding_diag
            if conversation_state is not None:
                # A backend that routed the state reports its own view; the rest
                # get the session-level one, so the trace answers the question
                # either way rather than only for one backend.
                trace["conversation"] = (
                    dict(result.conversation)
                    if isinstance(result.conversation, Mapping) and result.conversation
                    else conversation_state_diagnostics(conversation_state)
                )
            trace["route_id"] = route_id
            if result.compute:
                trace["compute"] = dict(result.compute)
            route_economics = self._finalize_route_economics(
                estimate=route_economics_estimate,
                trace=trace,
                elapsed_ms=elapsed_ms,
                actual_model_calls=int(route_model_call_counter.get("count") or 0),
            )
            trace["route_economics"] = route_economics
            if auto_agent_policy:
                trace["requested_agent_mode"] = "auto"
                trace["resolved_agent_mode"] = agent_mode
                auto_agent_policy["route_economics_actual"] = dict(route_economics["actual"])
                trace["auto_agent_policy"] = auto_agent_policy
            result.agent_trace = trace
            timing = dict(result.timing or {})
            timing["route_elapsed_ms"] = round(float(elapsed_ms), 2)
            timing.setdefault("route_cost_units", route_economics["actual"]["cost_units"])
            result.timing = timing

            assistant_summary = result.response or result.prompt_used or result.refined_prompt or ""
            tools_for_memory = list((result.agent_trace or {}).get("tool_events") or [])
            consultants_for_memory = list((result.agent_trace or {}).get("consultation_rows") or [])
            self.memory_store.add_route_usage(
                session_id=session_id,
                route_id=route_id,
                prompt=prompt,
                selected_agent_mode=agent_mode,
                route_economics=route_economics,
                auto_agent_policy=auto_agent_policy,
                route_reason=result.route_reason,
                model_key=result.model_key,
            )
            self.memory_store.update(
                session_id=session_id,
                user_text=prompt,
                assistant_text=assistant_summary,
                model_key=result.model_key,
                route_reason=result.route_reason,
                tools=tools_for_memory,
                consultants=consultants_for_memory,
            )

            self.selected_model_key = model_key or self.selected_model_key
            self.last_route_reason = result.route_reason
            payload = result.to_dict()
            if conversation_state is not None and not payload.get("conversation"):
                # Backends that route their own state report it themselves. The
                # rest — agent modes, image routes, the wrappers — reported
                # nothing at all, so the session-level state is attached here
                # and every Studio route now answers the same question.
                payload["conversation"] = conversation_state_diagnostics(conversation_state)
            payload["selected_model_key"] = self.selected_model_key
            payload["active_model_key"] = result.model_key
            payload["active_model_label"] = result.model_label
            payload["active_model_kind"] = self.record_map[result.model_key].kind
            payload["route_id"] = route_id
            # The durable ledger stores a flat terminal economics record.  Do
            # not wrap the already-partitioned trace object a second time or
            # replay will miss cost_units/elapsed_ms at the expected level.
            ledger_actual_economics = dict(route_economics.get("actual") or {})
            ledger_actual_economics["executed_model_key"] = result.model_key
            # A terminal-ledger failure is not a model failure. Clear the
            # exception bridge before the atomic pending -> completed update so
            # a persistence error leaves an honest in-flight row for repair.
            self._route_execution.route_id = ""
            self._route_execution.ledger_started = False
            ledger_row = self.route_policy_ledger.complete_decision(
                route_id,
                success=True,
                executed_mode=agent_mode,
                actual_economics=ledger_actual_economics,
            )
            payload_trace = payload.get("agent_trace")
            if isinstance(payload_trace, dict):
                payload_trace["route_ledger"] = {
                    "schema_version": ledger_row.get("ledger_schema_version"),
                    "status": ledger_row.get("status"),
                    "session_sequence": ledger_row.get("session_sequence"),
                    "feedback_status": ledger_row.get("feedback_status"),
                    "decision_type": ledger_row.get("decision_type"),
                    "probability_stage": ledger_row.get("probability_stage"),
                    "candidate_set_hash": ledger_row.get("candidate_set_hash"),
                    "distribution_hash": ledger_row.get("distribution_hash"),
                }
            return payload

    def preview_route_plan(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._lock:
            settings = dict(settings or {})
            settings.setdefault("memory_enabled", True)
            settings.setdefault("agent_mode", "off")
            settings.setdefault("web_search_enabled", False)
            settings.setdefault("cmd_open_enabled", True)
            settings.setdefault("web_search_budget", 3)
            settings.setdefault("web_search_results", 5)
            settings.setdefault("loop_max_steps", LOOP_AGENT_DEFAULT_MAX_STEPS)
            memory_bundle = self._prepare_memory_bundle(session_id, prompt, settings)
            memory_raw = memory_bundle.get("raw")
            raw_turns = (
                memory_raw.get("turns")
                if isinstance(memory_raw, Mapping)
                else ()
            )
            recent_turns = [
                turn
                for turn in (
                    raw_turns[-4:]
                    if isinstance(raw_turns, (list, tuple))
                    else ()
                )
                if isinstance(turn, Mapping)
            ]
            prompt_profile = analyze_prompt(
                prompt,
                recent_turns=recent_turns,
                recent_user_messages=[
                    str(turn.get("user") or "")
                    for turn in recent_turns
                    if str(turn.get("user") or "").strip()
                ],
                recent_assistant_messages=[
                    str(turn.get("assistant") or "")
                    for turn in recent_turns
                    if str(turn.get("assistant") or "").strip()
                ],
            )
            settings["_prompt_profile"] = prompt_profile

            requested_key = model_key or self.selected_model_key or "auto"
            if requested_key == "auto":
                chosen_record, route_reason = choose_auto_model(
                    self.records,
                    prompt,
                    action_mode=action_mode,
                    uploaded_image_path=str(settings.get("uploaded_image_path") or ""),
                    prompt_profile=prompt_profile,
                )
                if chosen_record is None:
                    raise RuntimeError("No local models were discovered.")
            else:
                chosen_record = self.record_map.get(requested_key)
                if chosen_record is None:
                    raise KeyError(f"Unknown model key: {requested_key}")
                route_reason = f"Manual selection kept {chosen_record.label}."

            resolved_action = action_mode
            if resolved_action == "auto":
                resolved_action = "image" if chosen_record.supports_image and not chosen_record.supports_chat else "text"

            requested_agent_mode = self._normalized_agent_mode(settings.get("agent_mode"))
            agent_mode = requested_agent_mode
            auto_agent_policy: Optional[Dict[str, Any]] = None
            if agent_mode == "auto":
                auto_agent_policy = self._resolve_auto_agent_mode(
                    session_id=session_id,
                    prompt=prompt,
                    action_mode=resolved_action,
                    settings=settings,
                    chosen_record=chosen_record,
                )
                if settings.get("reasoning_cycles") in (None, "", 0, "0", "model", "default"):
                    settings["reasoning_cycles"] = int(auto_agent_policy.get("reasoning_cycles") or 1)
                if settings.get("adaptive_compute") is None:
                    settings["adaptive_compute"] = True
                auto_agent_policy["runtime_compute_request"] = {
                    "reasoning_cycles": settings.get("reasoning_cycles"),
                    "adaptive_compute": bool(settings.get("adaptive_compute")),
                    "source": "auto_route_policy",
                }
                agent_mode = str(auto_agent_policy["selected_agent_mode"])
                settings["agent_mode"] = agent_mode
                budget_profile = str(auto_agent_policy.get("budget_profile") or "balanced")
                budget_phrase = "" if budget_profile == "balanced" else f" under the {budget_profile} budget profile"
                route_reason = (
                    f"{route_reason} Auto orchestration selected {agent_mode} "
                    f"because {auto_agent_policy['reason']}{budget_phrase}."
                )
            effective_agent_mode = self._effective_agent_mode_for_action(agent_mode, resolved_action)
            if effective_agent_mode != agent_mode:
                previous_agent_mode = agent_mode
                agent_mode = effective_agent_mode
                settings["agent_mode"] = agent_mode
                if auto_agent_policy is not None:
                    auto_agent_policy["pre_action_agent_mode"] = previous_agent_mode
                    auto_agent_policy["selected_agent_mode"] = agent_mode
                    route_confidence = auto_agent_policy.get("route_confidence")
                    if isinstance(route_confidence, dict):
                        route_confidence["selected_agent_mode"] = agent_mode
                        route_confidence["action_adjusted"] = True
                route_reason = (
                    f"{route_reason} Loop agent currently supports text and vision replies; "
                    f"image generation would use {agent_mode} instead of {previous_agent_mode}."
                )

            route_economics_estimate = self._estimate_route_economics(
                selected_agent_mode=agent_mode,
                action_mode=resolved_action,
                settings=settings,
                auto_agent_policy=auto_agent_policy,
            )
            if auto_agent_policy is not None:
                budget_limit = self._auto_session_budget_limit(settings)
                budget_snapshot = self._session_route_budget_snapshot(session_id, budget_limit)
                adjusted_agent_mode, route_economics_estimate, session_budget_adjustment, session_budget = (
                    self._apply_auto_session_budget(
                        selected=agent_mode,
                        action_mode=resolved_action,
                        settings=settings,
                        auto_agent_policy=auto_agent_policy,
                        budget_snapshot=budget_snapshot,
                    )
                )
                if session_budget is not None:
                    auto_agent_policy["session_budget"] = session_budget
                if session_budget_adjustment is not None:
                    previous_agent_mode = agent_mode
                    agent_mode = adjusted_agent_mode
                    settings["agent_mode"] = agent_mode
                    auto_agent_policy["pre_session_budget_agent_mode"] = previous_agent_mode
                    auto_agent_policy["selected_agent_mode"] = agent_mode
                    auto_agent_policy["reason"] = str(session_budget_adjustment.get("reason") or auto_agent_policy["reason"])
                    auto_agent_policy["session_budget_adjustment"] = session_budget_adjustment
                    route_confidence = auto_agent_policy.get("route_confidence")
                    if isinstance(route_confidence, dict):
                        route_confidence["session_budget_adjusted"] = True
                        route_confidence["selected_agent_mode"] = agent_mode
                    if previous_agent_mode == agent_mode:
                        route_reason = (
                            f"{route_reason} Session budget is exhausted; Auto kept {agent_mode} "
                            "because single-pass execution is the minimum route."
                        )
                    elif session_budget_adjustment.get("reason") == "session_route_budget_target_pacing":
                        route_reason = (
                            f"{route_reason} Session budget target pacing downgraded {previous_agent_mode} "
                            f"to {agent_mode} to preserve budget for the configured route horizon."
                        )
                    else:
                        route_reason = (
                            f"{route_reason} Session budget pacing downgraded {previous_agent_mode} to {agent_mode} "
                            f"because the estimated route cost would exceed the remaining budget."
                        )
                else:
                    auto_agent_policy["session_budget_adjustment"] = None
                auto_agent_policy["route_economics_estimate"] = dict(route_economics_estimate)
                logging_support = self._build_post_filter_logging_support(
                    selected=agent_mode,
                    action_mode=resolved_action,
                    settings=settings,
                    auto_agent_policy=auto_agent_policy,
                    selected_estimate=route_economics_estimate,
                )
                _stamp_auto_route_policy(
                    auto_agent_policy,
                    selected_agent_mode=agent_mode,
                    action_mode=resolved_action,
                    logging_support=logging_support,
                )

            collective_enabled = agent_mode in {"collective", "collective_loop"}
            loop_enabled = agent_mode in {"loop", "collective_loop"} and resolved_action != "image"
            route_alternatives = self._preview_route_alternatives(
                selected=agent_mode,
                action_mode=resolved_action,
                settings=settings,
                auto_agent_policy=auto_agent_policy,
            )
            route_frontier = self._annotate_route_frontier(
                alternatives=route_alternatives,
                selected=agent_mode,
                action_mode=resolved_action,
                auto_agent_policy=auto_agent_policy,
            )
            return {
                "ok": True,
                "dry_run": True,
                "session_id": session_id,
                "requested_model_key": requested_key,
                "active_model_key": chosen_record.key,
                "active_model_label": chosen_record.label,
                "active_model_kind": chosen_record.kind,
                "action_mode": resolved_action,
                "requested_agent_mode": requested_agent_mode,
                "selected_agent_mode": agent_mode,
                "route_reason": route_reason,
                "prompt_understanding": prompt_understanding_diagnostics(
                    prompt_profile
                ),
                "route_economics_estimate": route_economics_estimate,
                "route_alternatives": route_alternatives,
                "route_frontier": route_frontier,
                "auto_agent_policy": auto_agent_policy,
                "execution_plan": {
                    "collective_enabled": collective_enabled,
                    "loop_enabled": loop_enabled,
                    "single_pass": not collective_enabled and not loop_enabled,
                    "will_write_memory": False,
                    "will_run_inference": False,
                },
            }

    def preview_route_study(
        self,
        *,
        session_id: str,
        prompt: str,
        model_key: str,
        action_mode: str,
        settings: Dict[str, Any],
        exploration_rate: float = 0.10,
        planned_routes: int = 2_000,
        scenario_confidence: float = 0.95,
        assumed_feedback_rate: float = 0.30,
        target_observed_labels: int = 20,
        target_policy_profile: str = "balanced",
        protocol_design_mode: str = "sticky_session_cluster",
        carryover_scope: str = "unknown",
        interference_scope: str = "unknown",
        temporal_variation: str = "unknown",
        planned_clusters: int = 200,
        max_routes_per_cluster: int = 20,
        analysis_every_clusters: int = 50,
        block_length_routes: int = 20,
        washout_routes: int = 0,
    ) -> Dict[str, Any]:
        """Rehearse a prompt-specific adjacent-route charter without side effects.

        The normal preview path performs the final capability and session-budget
        filtering.  This method projects that immutable support into a distinct
        study cohort, but deliberately never samples, executes, or persists an
        assignment.  Rehearsed probabilities must not be treated as executed
        logging propensities.
        """

        preview_settings = dict(settings or {})
        if self._normalized_agent_mode(preview_settings.get("agent_mode")) != "auto":
            raise ValueError("Adjacent-route study rehearsal requires Auto Router mode")
        preview = self.preview_route_plan(
            session_id=session_id,
            prompt=prompt,
            model_key=model_key,
            action_mode=action_mode,
            settings=preview_settings,
        )
        policy = preview.get("auto_agent_policy")
        if not isinstance(policy, Mapping):
            raise ValueError("Auto Router did not produce a route policy for study rehearsal")
        support = policy.get("logging_support")
        if not isinstance(support, Mapping):
            raise ValueError("Auto Router did not produce final post-filter support")
        study = plan_adjacent_route_study(
            str(preview.get("selected_agent_mode") or ""),
            list(support.get("candidates") or []),
            list(support.get("exclusions") or []),
            source_contract={
                "policy_id": policy.get("policy_id"),
                "policy_version": policy.get("policy_version"),
                "feature_schema_version": policy.get("feature_schema_version"),
                "support_schema_version": support.get("schema_version"),
                "candidate_set_hash": support.get("candidate_set_hash"),
                "distribution_hash": support.get("distribution_hash"),
                "outcome_contract_schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
            },
            exploration_rate=exploration_rate,
            planned_routes=planned_routes,
            scenario_confidence=scenario_confidence,
            assumed_feedback_rate=assumed_feedback_rate,
            target_observed_labels=target_observed_labels,
        )
        protocol_preflight = None
        protocol_preflight_reason = "no_eligible_adjacent_support"
        if study["charter"]["enrollment"]["eligible"] is True:
            protocol_preflight = build_route_study_protocol(
                study,
                target_policy_profile=target_policy_profile,
                design_mode=protocol_design_mode,
                carryover_scope=carryover_scope,
                interference_scope=interference_scope,
                temporal_variation=temporal_variation,
                population_rule_id="interactive-auto-route-opt-in",
                population_rule_version="1",
                cluster_key_schema_version="session-hash-v1",
                planned_clusters=planned_clusters,
                max_routes_per_cluster=max_routes_per_cluster,
                analysis_every_clusters=analysis_every_clusters,
                block_length_routes=block_length_routes,
                washout_routes=washout_routes,
            )
            protocol_preflight_reason = "draft_for_independent_review"
        return {
            "ok": True,
            "dry_run": True,
            "session_id": session_id,
            "requested_model_key": preview.get("requested_model_key"),
            "active_model_key": preview.get("active_model_key"),
            "active_model_label": preview.get("active_model_label"),
            "action_mode": preview.get("action_mode"),
            "baseline_agent_mode": preview.get("selected_agent_mode"),
            "route_reason": preview.get("route_reason"),
            "deterministic_support": {
                "policy_id": policy.get("policy_id"),
                "policy_version": policy.get("policy_version"),
                "feature_schema_version": policy.get("feature_schema_version"),
                "support_schema_version": support.get("schema_version"),
                "candidate_set_hash": policy.get("candidate_set_hash"),
                "distribution_hash": policy.get("distribution_hash"),
                "outcome_contract_schema_version": OUTCOME_CONTRACT_SCHEMA_VERSION,
            },
            "route_study": study,
            "route_protocol_preflight": protocol_preflight,
            "route_protocol_preflight_reason": protocol_preflight_reason,
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }

    def build_route_protocol_review_bundle(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Build and fully reconstruct a prompt-free multi-stratum review bundle."""

        bundle = build_route_study_review_bundle_from_input(payload)
        verification = audit_route_study_review_bundle(bundle)
        return {
            "ok": True,
            "dry_run": True,
            "route_protocol_review_bundle": bundle,
            "verification": verification,
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }

    def audit_route_protocol_review_bundle(self, bundle: Any) -> Dict[str, Any]:
        """Perform full source-bound reconstruction without touching runtime state."""

        verification = audit_route_study_review_bundle(bundle)
        return {
            "ok": True,
            "dry_run": True,
            "verification": verification,
            "execution_plan": {
                "will_run_inference": False,
                "will_write_memory": False,
                "will_write_ledger": False,
                "will_assign_route": False,
                "will_randomize": False,
                "activation_available": False,
            },
        }
