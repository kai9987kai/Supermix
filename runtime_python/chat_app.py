import argparse
import inspect
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import torch

from device_utils import configure_torch_runtime, resolve_device
from chat_pipeline import (
    MODEL_CLASSES,
    build_context,
    choose_bucket_from_logits,
    cleanup_response_text,
    infer_style_mode,
    pick_response,
    rank_response_candidates,
    resolve_feature_mode,
    text_to_model_input,
)
from llm_database import LLMDatabase
from model_variants import (
    build_model,
    detect_large_head_expansion_dim,
    detect_model_size_from_state_dict,
    detect_xlarge_aux_expansion_dim,
    detect_xxlarge_third_expansion_dim,
    detect_xxxlarge_fourth_expansion_dim,
    detect_ultralarge_fifth_expansion_dim,
    detect_megalarge_sixth_expansion_dim,
    EXPANSION_DIM_MODEL_SIZES,
    EXTRA_EXPANSION_DIM_MODEL_SIZES,
    FIFTH_EXPANSION_DIM_MODEL_SIZES,
    FOURTH_EXPANSION_DIM_MODEL_SIZES,
    load_weights_for_model,
    SIXTH_EXPANSION_DIM_MODEL_SIZES,
    SUPPORTED_MODEL_SIZES,
    THIRD_EXPANSION_DIM_MODEL_SIZES,
)
from chat_memory import ChatMemoryDB, render_memory_block
from run import safe_load_state_dict


# UI color constants
class TerminalColors:
    USER = "\033[94m"     # Blue
    BOT = "\033[92m"      # Green
    SYSTEM = "\033[93m"   # Yellow
    RESET = "\033[0m"


def load_metadata(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{TerminalColors.SYSTEM}Error loading metadata from {path}: {e}{TerminalColors.RESET}")
        return {}


def _int_or_default(value, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


MAX_RUNTIME_REASONING_CYCLES = 64
DEFAULT_ADAPTIVE_EXIT_TOL = 1e-3
DEFAULT_AUTO_COMPUTE_CONFIDENCE = 0.55
DEFAULT_AUTO_COMPUTE_ENTROPY = 1.85
AUTO_COMPUTE_PLAN_SCHEMA_VERSION = "runtime-auto-compute-plan-v2"
AUTO_COMPUTE_STRATEGY = "progressive_accepted_probe"
DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K = 5


def resolve_runtime_compute_cycles(cycles, default: Optional[List[int]] = None, limit: int = 8) -> List[int]:
    if default is None:
        default = [1, 3, 8]
    if cycles is None or cycles == "":
        raw_values = list(default)
    elif isinstance(cycles, str):
        raw_values = [part.strip() for part in cycles.split(",")]
    elif isinstance(cycles, (list, tuple)):
        raw_values = list(cycles)
    else:
        raw_values = [cycles]

    resolved: List[int] = []
    seen = set()
    for value in raw_values:
        parsed = _coerce_optional_positive_int(
            value,
            default=None,
            max_value=MAX_RUNTIME_REASONING_CYCLES,
        )
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        resolved.append(parsed)
        if len(resolved) >= max(1, int(limit)):
            break
    return resolved or list(default)


def runtime_auto_compute_cycles(preferred_cycles: Optional[int] = None) -> List[int]:
    preferred = _coerce_optional_positive_int(
        preferred_cycles,
        default=None,
        max_value=MAX_RUNTIME_REASONING_CYCLES,
    )
    if preferred is None:
        return [1, 3, 8]
    return resolve_runtime_compute_cycles(
        [1, preferred, min(MAX_RUNTIME_REASONING_CYCLES, max(preferred + 1, preferred * 2))]
    )


def evaluate_runtime_compute_budgets(
    model,
    x,
    available_labels: List[int],
    *,
    cycles=None,
    adaptive_compute: bool = False,
    exit_tol: Optional[float] = None,
) -> List[Dict[str, object]]:
    labels = list(available_labels) or list(range(MODEL_CLASSES))
    idx_device = x.device if isinstance(x, torch.Tensor) else None
    idx = torch.tensor(labels, dtype=torch.long, device=idx_device)
    resolved_exit_tol = _coerce_nonnegative_float(
        DEFAULT_ADAPTIVE_EXIT_TOL if exit_tol is None else exit_tol,
        default=DEFAULT_ADAPTIVE_EXIT_TOL,
    )
    rows: List[Dict[str, object]] = []
    for requested_cycles in resolve_runtime_compute_cycles(cycles):
        t0 = time.perf_counter()
        with torch.no_grad():
            logits_tensor, compute_metrics = forward_with_runtime_compute(
                model,
                x,
                reasoning_cycles=requested_cycles,
                adaptive_compute=adaptive_compute,
                exit_tol=resolved_exit_tol,
                return_diagnostics=True,
            )
            logits = logits_tensor[0, 0]
            avail_logits = logits.index_select(0, idx.to(logits.device))
            probs = torch.softmax(avail_logits, dim=0)
            confidence_tensor, pred_pos_tensor = torch.max(probs, dim=0)
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())
        pred_pos = int(pred_pos_tensor.item())
        rows.append(
            {
                "requested_cycles": int(requested_cycles),
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "cycles_used": compute_metrics.get("cycles_used"),
                "predicted_label": int(labels[pred_pos]),
                "confidence": round(float(confidence_tensor.item()), 6),
                "entropy": round(entropy, 6),
                "compute": compute_metrics,
            }
        )
    return rows


def select_auto_runtime_compute_budget(
    rows: List[Dict[str, object]],
    *,
    confidence_target: float = DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    entropy_target: float = DEFAULT_AUTO_COMPUTE_ENTROPY,
) -> Dict[str, object]:
    confidence_target = max(0.0, min(1.0, float(confidence_target)))
    entropy_target = max(0.0, float(entropy_target))
    if not rows:
        return {
            "enabled": True,
            "selected_reasoning_cycles": None,
            "reason": "no_rows",
            "confidence_target": confidence_target,
            "entropy_target": entropy_target,
            "rows": [],
        }
    best = max(
        rows,
        key=lambda row: (
            float(row.get("confidence", 0.0)),
            -float(row.get("entropy", 999.0)),
            -float(row.get("latency_ms", 0.0)),
        ),
    )
    selected = best
    reason = "best_confidence"
    for row in rows:
        if float(row.get("confidence", 0.0)) >= confidence_target:
            selected = row
            reason = "confidence_target"
            break
        if float(row.get("entropy", 999.0)) <= entropy_target:
            selected = row
            reason = "entropy_target"
            break
    return {
        "enabled": True,
        "selected_reasoning_cycles": int(selected["requested_cycles"]),
        "selected_index": rows.index(selected),
        "reason": reason,
        "confidence_target": confidence_target,
        "entropy_target": entropy_target,
        "rows": rows,
    }


_FOLLOWUP_QUERY_RE = re.compile(r"\b(it|that|this|they|them|same|previous|above|more|continue|deeper|expand)\b", re.I)
VALID_RUNTIME_MODEL_SIZES = SUPPORTED_MODEL_SIZES
MAX_RUNTIME_REASONING_CYCLES = 64
DEFAULT_ADAPTIVE_EXIT_ENTROPY = 0.2
DEFAULT_PREDICTION_STABILITY_PATIENCE = 2
DEFAULT_PREDICTION_STABILITY_TOL = 5e-3
AUTO_REASONING_CYCLE_BUCKETS = (1, 3, 8, 16)


def _build_db_query(user: str, history: List[Tuple[str, str]], memory_rows: List[Dict], max_turns: int = 2) -> str:
    """
    Conversation-aware retrieval query (lightweight QRHEAD-style expansion).
    Uses the raw user query when standalone, and appends recent context for follow-ups.
    """
    user_text = (user or "").strip()
    if not user_text:
        return ""

    use_context = bool(_FOLLOWUP_QUERY_RE.search(user_text)) or len(user_text.split()) <= 4
    if not use_context and history:
        # If the user explicitly asks about a prior topic, keep some context anyway.
        use_context = user_text.endswith("?") and any(k in user_text.lower() for k in ("also", "again", "same", "earlier"))
    if not use_context:
        return user_text

    parts = [user_text]
    for hu, ha in history[-max(0, int(max_turns)):]:
        if hu:
            parts.append(hu)
        if ha:
            parts.append(ha[:220])
    for row in memory_rows[:2]:
        u = str(row.get("user_text", "")).strip()
        a = str(row.get("assistant_text", "")).strip()
        if u:
            parts.append(u)
        if a:
            parts.append(a[:160])
            
    # Deduplicate exact segments while preserving order.
    dedup: List[str] = []
    seen = set()
    for p in parts:
        s = " ".join(p.split())
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(s)
    return " | ".join(dedup[:6])


def _resolve_expansion_dim(arg_val: Optional[int], meta: Dict, meta_key: str, default_val: int, 
                           inferred_size: str, allowed_sizes: set, detect_fn, sd: Dict) -> int:
    """Helper to dynamically resolve and infer model dimensionalities, keeping main() DRY."""
    if arg_val is not None:
        return arg_val
    val = _int_or_default(meta.get(meta_key), default_val)
    if inferred_size in allowed_sizes and detect_fn is not None:
        return detect_fn(sd, default=val)
    return val


def resolve_runtime_model_size(requested_model_size: str, meta_model_size: str, inferred_from_weights: str) -> Tuple[str, str]:
    resolved_model_size = str(requested_model_size or "auto").strip().lower() or "auto"
    warning = ""
    if resolved_model_size == "auto":
        meta_model_size = str(meta_model_size or "").strip().lower()
        if meta_model_size in VALID_RUNTIME_MODEL_SIZES and meta_model_size != inferred_from_weights:
            warning = (
                f"{TerminalColors.SYSTEM}Warning: metadata model_size="
                f"{meta_model_size} but weights look like {inferred_from_weights}; using weights.{TerminalColors.RESET}"
            )
        resolved_model_size = inferred_from_weights
    if resolved_model_size not in VALID_RUNTIME_MODEL_SIZES:
        raise RuntimeError(f"Invalid model_size={resolved_model_size!r} (from args/meta).")
    return resolved_model_size, warning


def _coerce_optional_positive_int(
    value: Any,
    max_value: Optional[int] = None,
    *,
    default: Optional[int] = None,
) -> Optional[int]:
    if value is None:
        return default
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "default", "none", "off", "auto"}:
            return default
        value = stripped
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    parsed = max(1, parsed)
    if max_value is not None:
        parsed = min(parsed, max(1, int(max_value)))
    return parsed


def _is_auto_reasoning_cycles(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in {"auto", "adaptive", "smart"}


def _format_reasoning_cycles_setting(value: Any) -> Any:
    if _is_auto_reasoning_cycles(value):
        return "auto"
    return _coerce_optional_positive_int(value, MAX_RUNTIME_REASONING_CYCLES) or "default"


def estimate_auto_reasoning_cycles(context: str, max_value: Optional[int] = None) -> Dict[str, Any]:
    text = str(context or "")
    lowered = text.lower()
    reasons: List[str] = []
    score = 0

    char_count = len(text)
    if char_count >= 1600:
        score += 3
        reasons.append("long_context")
    elif char_count >= 700:
        score += 2
        reasons.append("medium_context")
    elif char_count >= 280:
        score += 1
        reasons.append("short_context")

    if re.search(r"```|traceback|exception|error|bug|fix|debug|refactor|class\s+|def\s+|function|import\s+|select\s+", lowered):
        score += 2
        reasons.append("code_or_debug")
    if re.search(r"\b(prove|derive|equation|calculate|solve|theorem|complexity|probability|optimi[sz]e|benchmarks?)\b|[=+\-*/^]{2,}", lowered):
        score += 2
        reasons.append("math_or_verification")
    if re.search(r"\b(compare|tradeoff|plan|architect|design|multi-step|step by step|analy[sz]e|research|evidence)\b", lowered):
        score += 1
        reasons.append("deliberation")
    if re.search(r"\b(agent|tool|browser|github|file|tests?|verify|integration|runtime)\b", lowered):
        score += 1
        reasons.append("tool_or_runtime")
    if re.search(r"\b(quick|brief|concise|short answer|one line)\b", lowered):
        score -= 1
        reasons.append("brevity_requested")

    if score <= 0:
        cycles = AUTO_REASONING_CYCLE_BUCKETS[0]
    elif score <= 2:
        cycles = AUTO_REASONING_CYCLE_BUCKETS[1]
    elif score <= 4:
        cycles = AUTO_REASONING_CYCLE_BUCKETS[2]
    else:
        cycles = AUTO_REASONING_CYCLE_BUCKETS[3]
    cycles = _coerce_optional_positive_int(cycles, max_value) or AUTO_REASONING_CYCLE_BUCKETS[0]

    return {
        "mode": "auto",
        "score": int(score),
        "reasons": reasons or ["simple_prompt"],
        "cycles": int(cycles),
    }


def _coerce_nonnegative_float(value: Any, default: float, max_value: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    if not math.isfinite(parsed):
        parsed = float(default)
    parsed = max(0.0, parsed)
    if max_value is not None:
        parsed = min(parsed, float(max_value))
    return parsed


def _coerce_nonnegative_int(value: Any, default: int, max_value: Optional[int] = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = int(default)
    parsed = max(0, parsed)
    if max_value is not None:
        parsed = min(parsed, max(0, int(max_value)))
    return parsed


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on", "y", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "n", "disable", "disabled"}:
            return False
    return bool(default)


def _model_forward_accepts_kwarg(model: Any, kwarg: str) -> bool:
    forward = getattr(model, "forward", None)
    if forward is None:
        return False
    try:
        params = inspect.signature(forward).parameters.values()
    except (TypeError, ValueError):
        return False
    for param in params:
        if param.kind == inspect.Parameter.VAR_KEYWORD or param.name == kwarg:
            return True
    return False


def model_supports_runtime_compute(model: Any) -> bool:
    return any(
        _model_forward_accepts_kwarg(model, key)
        for key in ("reasoning_cycles", "adaptive_compute", "exit_tol")
    )


def _to_optional_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            return float(value.detach().cpu().item())
        return float(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def collect_runtime_compute_metrics(
    model: Any,
    *,
    requested_reasoning_cycles: Optional[int] = None,
    adaptive_compute: bool = False,
    exit_tol: Optional[float] = None,
    applied_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "cycles_used": None,
        "ponder_cost": None,
        "consistency_loss": None,
        "gating_entropy": None,
        "prediction_streak": None,
        "prediction_confidence_delta": None,
        "exit_reason": None,
    }
    attr_map = {
        "last_cycles_used": "cycles_used",
        "last_ponder_cost": "ponder_cost",
        "last_consistency_loss": "consistency_loss",
        "last_gating_entropy": "gating_entropy",
        "last_prediction_streak": "prediction_streak",
        "last_prediction_confidence_delta": "prediction_confidence_delta",
    }
    modules = list(model.modules()) if hasattr(model, "modules") else [model]
    for module in modules:
        for attr, key in attr_map.items():
            if metrics[key] is not None:
                continue
            scalar = _to_optional_scalar(getattr(module, attr, None))
            if scalar is not None:
                metrics[key] = round(scalar, 6)
        if metrics["exit_reason"] is None:
            reason = getattr(module, "last_exit_reason", None)
            if isinstance(reason, str) and reason.strip():
                metrics["exit_reason"] = reason.strip()
    metrics.update({
        "supported": model_supports_runtime_compute(model),
        "requested_reasoning_cycles": requested_reasoning_cycles,
        "max_reasoning_cycles": MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": bool(adaptive_compute),
        "exit_tol": exit_tol,
        "applied": bool(applied_kwargs),
        "applied_kwargs": dict(applied_kwargs or {}),
    })
    return metrics


def get_last_cycles_used(model: Any) -> Optional[float]:
    return collect_runtime_compute_metrics(model).get("cycles_used")


def forward_with_runtime_compute(
    model: Any,
    x: torch.Tensor,
    reasoning_cycles: Any = None,
    adaptive_compute: Any = False,
    exit_tol: Any = 1e-3,
    exit_entropy_threshold: Any = DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    prediction_stability_patience: Any = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    prediction_stability_tol: Any = DEFAULT_PREDICTION_STABILITY_TOL,
    auto_reasoning_context: str = "",
    return_diagnostics: Optional[bool] = None,
):
    """Forward a model while applying optional v50 runtime-compute controls.

    Legacy model variants ignore these controls because their forward signatures
    do not accept them. This keeps the runtime knobs safe for mixed checkpoints.
    """
    auto_policy: Optional[Dict[str, Any]] = None
    if _is_auto_reasoning_cycles(reasoning_cycles):
        auto_policy = estimate_auto_reasoning_cycles(
            auto_reasoning_context,
            MAX_RUNTIME_REASONING_CYCLES,
        )
        cycles = int(auto_policy["cycles"])
    else:
        cycles = _coerce_optional_positive_int(reasoning_cycles, MAX_RUNTIME_REASONING_CYCLES)
    adaptive = _coerce_bool(adaptive_compute)
    tol = _coerce_nonnegative_float(exit_tol, 1e-3)
    entropy_threshold = _coerce_nonnegative_float(
        exit_entropy_threshold,
        DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    )
    stability_patience = _coerce_nonnegative_int(
        prediction_stability_patience,
        DEFAULT_PREDICTION_STABILITY_PATIENCE,
        MAX_RUNTIME_REASONING_CYCLES,
    )
    stability_tol = _coerce_nonnegative_float(
        prediction_stability_tol,
        DEFAULT_PREDICTION_STABILITY_TOL,
    )
    kwargs: Dict[str, Any] = {}

    if cycles is not None and _model_forward_accepts_kwarg(model, "reasoning_cycles"):
        kwargs["reasoning_cycles"] = cycles
    if adaptive and _model_forward_accepts_kwarg(model, "adaptive_compute"):
        kwargs["adaptive_compute"] = True
        if _model_forward_accepts_kwarg(model, "exit_tol"):
            kwargs["exit_tol"] = tol
        if _model_forward_accepts_kwarg(model, "exit_entropy_threshold"):
            kwargs["exit_entropy_threshold"] = entropy_threshold
        if _model_forward_accepts_kwarg(model, "prediction_stability_patience"):
            kwargs["prediction_stability_patience"] = stability_patience
        if _model_forward_accepts_kwarg(model, "prediction_stability_tol"):
            kwargs["prediction_stability_tol"] = stability_tol

    output = model(x, **kwargs) if kwargs else model(x)
    metrics = collect_runtime_compute_metrics(
        model,
        requested_reasoning_cycles=cycles,
        adaptive_compute=adaptive,
        exit_tol=tol,
        applied_kwargs=kwargs,
    )
    diagnostics = {
        "requested_reasoning_cycles": cycles,
        "selected_reasoning_cycles": cycles,
        "reasoning_budget_mode": "auto" if auto_policy is not None else ("manual" if cycles is not None else "default"),
        "auto_reasoning_policy": auto_policy,
        "max_reasoning_cycles": MAX_RUNTIME_REASONING_CYCLES,
        "adaptive_compute": adaptive,
        "exit_tol": tol,
        "exit_entropy_threshold": entropy_threshold,
        "prediction_stability_patience": stability_patience,
        "prediction_stability_tol": stability_tol,
        "applied": bool(kwargs),
        "applied_kwargs": dict(kwargs),
        **metrics,
    }
    if return_diagnostics is False:
        return output
    return output, diagnostics


def _topk_distribution_js_divergence(
    previous: torch.Tensor,
    current: torch.Tensor,
    top_k: int = DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
) -> float:
    """Return midpoint-support JSD while preserving tail probability mass."""

    left = previous.detach().to(dtype=torch.float64).reshape(-1)
    right = current.detach().to(dtype=torch.float64).reshape(-1)
    if left.numel() != right.numel() or left.numel() == 0:
        return 0.0
    k = min(max(1, int(top_k)), int(left.numel()))
    midpoint = 0.5 * (left + right)
    indices = torch.topk(midpoint, k=k).indices
    left_top = left.index_select(0, indices)
    right_top = right.index_select(0, indices)
    left_compact = torch.cat(
        [left_top, (1.0 - left_top.sum()).clamp_min(0.0).reshape(1)]
    )
    right_compact = torch.cat(
        [right_top, (1.0 - right_top.sum()).clamp_min(0.0).reshape(1)]
    )
    compact_midpoint = 0.5 * (left_compact + right_compact)
    epsilon = torch.finfo(left_compact.dtype).tiny

    def _kl(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (
            source
            * (
                torch.log(source.clamp_min(epsilon))
                - torch.log(target.clamp_min(epsilon))
            )
        ).sum()

    value = 0.5 * (
        _kl(left_compact, compact_midpoint)
        + _kl(right_compact, compact_midpoint)
    )
    return round(max(0.0, float(value.item())), 10)


def progressive_auto_compute_forward(
    model: Any,
    x: torch.Tensor,
    available_labels: List[int],
    *,
    cycles: Any = None,
    confidence_target: Any = DEFAULT_AUTO_COMPUTE_CONFIDENCE,
    entropy_target: Any = DEFAULT_AUTO_COMPUTE_ENTROPY,
    adaptive_compute: Any = False,
    exit_tol: Any = DEFAULT_ADAPTIVE_EXIT_TOL,
    exit_entropy_threshold: Any = DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    prediction_stability_patience: Any = DEFAULT_PREDICTION_STABILITY_PATIENCE,
    prediction_stability_tol: Any = DEFAULT_PREDICTION_STABILITY_TOL,
    auto_reasoning_context: str = "",
    distribution_top_k: int = DEFAULT_AUTO_COMPUTE_DISTRIBUTION_TOP_K,
) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
    """Evaluate auto-compute budgets progressively and reuse the accepted probe.

    This preserves the v1 selection policy and row order exactly: the first
    confidence/entropy target wins, otherwise the best-confidence row wins
    after the full ladder. Probes use the complete current runtime-control set,
    and the accepted probe output is returned directly, removing the old N+1
    rerun. Cross-budget persistence and JSD are telemetry only and never select
    or reject an output.
    """

    if bool(getattr(model, "training", False)):
        raise RuntimeError("progressive auto compute requires model.eval()")
    labels = list(available_labels) or list(range(MODEL_CLASSES))
    resolved_cycles = resolve_runtime_compute_cycles(cycles)
    confidence_target = min(
        1.0,
        _coerce_nonnegative_float(
            confidence_target,
            DEFAULT_AUTO_COMPUTE_CONFIDENCE,
        ),
    )
    entropy_target = _coerce_nonnegative_float(
        entropy_target,
        DEFAULT_AUTO_COMPUTE_ENTROPY,
    )
    adaptive = _coerce_bool(adaptive_compute)
    idx = torch.tensor(labels, dtype=torch.long, device=x.device)
    records: List[Dict[str, Any]] = []
    cached_outputs: List[torch.Tensor] = []
    previous_probs: Optional[torch.Tensor] = None
    previous_label: Optional[int] = None
    previous_confidence: Optional[float] = None
    accepted_index: Optional[int] = None
    reason = "best_confidence"

    for requested_cycles in resolved_cycles:
        started = time.perf_counter()
        with torch.no_grad():
            model_out, compute = forward_with_runtime_compute(
                model,
                x,
                reasoning_cycles=requested_cycles,
                adaptive_compute=adaptive,
                exit_tol=exit_tol,
                exit_entropy_threshold=exit_entropy_threshold,
                prediction_stability_patience=prediction_stability_patience,
                prediction_stability_tol=prediction_stability_tol,
                auto_reasoning_context=auto_reasoning_context,
            )
            logits = model_out[0, 0]
            available_logits = logits.index_select(0, idx.to(logits.device))
            probs = torch.softmax(available_logits, dim=0)
            confidence_tensor, predicted_position = torch.max(probs, dim=0)
            entropy_tensor = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        # v1 serialized these values to six decimals before the selector read
        # them. Keep that boundary exact so values such as 0.5499996 still
        # satisfy a 0.55 target just as they did before.
        confidence = round(float(confidence_tensor.item()), 6)
        entropy = round(float(entropy_tensor.item()), 6)
        predicted_label = int(labels[int(predicted_position.item())])
        # Freeze selection latency before shadow-only diagnostics. Otherwise a
        # costly JSD calculation could alter the legacy fallback tie-break.
        probe_latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        has_previous = previous_probs is not None
        shadow = {
            "role": "shadow_diagnostic_only",
            "metric": "midpoint_support_top_k_jensen_shannon_divergence_nats",
            "top_k": min(max(1, int(distribution_top_k)), len(labels)),
            "has_previous_probe": has_previous,
            "top1_persistent": (
                bool(previous_label == predicted_label) if has_previous else None
            ),
            "confidence_delta": (
                round(abs(confidence - float(previous_confidence)), 8)
                if previous_confidence is not None
                else None
            ),
            "js_divergence": (
                _topk_distribution_js_divergence(
                    previous_probs,
                    probs,
                    distribution_top_k,
                )
                if previous_probs is not None
                else None
            ),
            "selection_enabled": False,
        }
        row = {
            "requested_cycles": int(requested_cycles),
            "latency_ms": probe_latency_ms,
            "cycles_used": compute.get("cycles_used"),
            "predicted_label": predicted_label,
            "confidence": confidence,
            "entropy": entropy,
            "confidence_target_met": bool(confidence >= confidence_target),
            "entropy_target_met": bool(entropy <= entropy_target),
            "mutual_stability_shadow": shadow,
            "compute": compute,
        }
        records.append(row)
        cached_outputs.append(model_out)

        if confidence >= confidence_target:
            accepted_index = len(records) - 1
            reason = "confidence_target"
            break
        if entropy <= entropy_target:
            accepted_index = len(records) - 1
            reason = "entropy_target"
            break
        previous_probs = probs.detach()
        previous_label = predicted_label
        previous_confidence = confidence

    if accepted_index is None:
        accepted_index = max(
            range(len(records)),
            key=lambda index: (
                float(records[index].get("confidence", 0.0)),
                -float(records[index].get("entropy", 999.0)),
                -float(records[index].get("latency_ms", 0.0)),
            ),
        )

    selected_row = records[accepted_index]
    selected_output = cached_outputs[accepted_index]
    selected_compute = dict(selected_row["compute"])
    evaluated_cycles = [int(row["requested_cycles"]) for row in records]
    skipped_cycles = [
        int(value) for value in resolved_cycles[len(evaluated_cycles) :]
    ]
    legacy_forward_evaluations = len(resolved_cycles) + 1
    forward_evaluations = len(records)
    plan = {
        "schema_version": AUTO_COMPUTE_PLAN_SCHEMA_VERSION,
        "enabled": True,
        "strategy": AUTO_COMPUTE_STRATEGY,
        "selected_reasoning_cycles": int(selected_row["requested_cycles"]),
        "selected_index": int(accepted_index),
        "accepted_probe_index": int(accepted_index),
        "reason": reason,
        "confidence_target": confidence_target,
        "entropy_target": entropy_target,
        "candidate_cycles": [int(value) for value in resolved_cycles],
        "evaluated_cycles": evaluated_cycles,
        "skipped_cycles": skipped_cycles,
        "forward_evaluations": forward_evaluations,
        "legacy_forward_evaluations": legacy_forward_evaluations,
        "forward_reduction_percent": round(
            100.0
            * (legacy_forward_evaluations - forward_evaluations)
            / max(1, legacy_forward_evaluations),
            3,
        ),
        "reused_probe_output": True,
        "selection_semantics": "legacy_v1_selection_policy",
        "probe_control_scope": "full_runtime_controls_v2",
        "mutual_stability_role": "shadow_diagnostic_only",
        "rows": records,
    }
    selected_compute["auto_compute_plan"] = plan
    selected_compute["inference_reused"] = True
    return selected_output, selected_compute, plan


def compact_auto_compute_plan_metrics(plan: Any) -> Dict[str, Any]:
    """Return terminal-safe metrics for the probe that supplied the output."""

    if not isinstance(plan, dict):
        return {}
    rows = plan.get("rows") if isinstance(plan.get("rows"), list) else []
    try:
        selected_index = int(plan.get("selected_index"))
    except (TypeError, ValueError):
        selected_index = -1
    if not 0 <= selected_index < len(rows):
        selected_index = len(rows) - 1
    selected_row = rows[selected_index] if selected_index >= 0 else {}
    shadow = (
        selected_row.get("mutual_stability_shadow", {})
        if isinstance(selected_row, dict)
        else {}
    )
    metrics: Dict[str, Any] = {
        "auto_selected": plan.get("selected_reasoning_cycles"),
        "auto_reason": plan.get("reason"),
        "auto_forwards": (
            f"{plan.get('forward_evaluations')}/"
            f"{plan.get('legacy_forward_evaluations')}"
        ),
        "auto_reused": bool(plan.get("reused_probe_output")),
    }
    if isinstance(shadow, dict) and shadow.get("js_divergence") is not None:
        metrics["auto_shadow_js"] = shadow.get("js_divergence")
    return metrics


def _default_expansion_dim_for_model_size(model_size: str) -> int:
    if model_size in THIRD_EXPANSION_DIM_MODEL_SIZES:
        return 1024
    if model_size == "xlarge":
        return 768
    return 512


def _default_extra_expansion_dim_for_model_size(model_size: str, expansion_dim: int) -> int:
    base = 2048 if model_size in THIRD_EXPANSION_DIM_MODEL_SIZES else 1024
    return max(base, expansion_dim * 2)


def _format_ms(seconds: float) -> str:
    return f"{max(0.0, float(seconds)) * 1000.0:.1f} ms"


def _format_duration(seconds: float) -> str:
    s = int(max(0.0, float(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _print_chat_help() -> None:
    print(f"{TerminalColors.SYSTEM}Commands:{TerminalColors.RESET}")
    print("  /help               Show this help")
    print("  /stats              Show session stats")
    print("  /clear              Clear in-memory conversation history")
    print("  /style <mode>       Set style: auto|balanced|creative|concise|analyst")
    print("  /creativity <0-1>   Set creative rewrite strength")
    print("  /cycles <n|auto|default> Set runtime reasoning cycles for future turns")
    print("  /adaptive on|off    Toggle v50 convergence early-exit")
    print("  /exit_tol <float>   Set v50 adaptive-compute convergence tolerance")
    print("  /exit_entropy <f>   Set v50 adaptive-compute entropy exit threshold")
    print("  /stability <n>      Stable prediction cycles required (0 disables)")
    print("  /stability_tol <f>  Maximum confidence drift across the stable streak")
    print("  /top <n>            Show top reranked candidates each turn (0 disables)")
    print("  /timing on|off      Toggle per-turn timing output")
    print("  /auto_compute on|off Toggle confidence/entropy based cycle selection")
    print("  /auto_targets <confidence> <entropy> Set auto-compute selection gates")
    print("  /memory on|off      Toggle memory retrieval/writes for this session")
    print("  /db on|off          Toggle local LLM DB retrieval for this session")
    print("  /config             Print current runtime config")
    print("  /quit               Exit")


def main():
    ap = argparse.ArgumentParser(description="Run an advanced retrieval-style chat app on fine-tuned ChampionNet.")
    ap.add_argument("--weights", default="champion_model_chat_ft.pth")
    ap.add_argument("--meta", default="chat_model_meta.json")
    ap.add_argument(
        "--model_size",
        choices=["auto", *VALID_RUNTIME_MODEL_SIZES],
        default="auto",
    )
    ap.add_argument("--expansion_dim", type=int, default=None)
    ap.add_argument("--extra_expansion_dim", type=int, default=None)
    ap.add_argument("--third_expansion_dim", type=int, default=None)
    ap.add_argument("--fourth_expansion_dim", type=int, default=None)
    ap.add_argument("--fifth_expansion_dim", type=int, default=None)
    ap.add_argument("--sixth_expansion_dim", type=int, default=None)
    ap.add_argument("--adapter_dropout", type=float, default=None)
    ap.add_argument(
        "--reasoning_cycles",
        type=str,
        default=None,
        help="Optional runtime reasoning cycles, or 'auto' to choose from prompt difficulty; omit for checkpoint default.",
    )
    ap.add_argument(
        "--adaptive_compute",
        action="store_true",
        help="Enable v50 convergence early-exit during inference when the model supports it.",
    )
    ap.add_argument(
        "--adaptive_exit_tol",
        type=float,
        default=1e-3,
        help="Convergence tolerance for v50 adaptive compute.",
    )
    ap.add_argument(
        "--adaptive_exit_entropy",
        type=float,
        default=DEFAULT_ADAPTIVE_EXIT_ENTROPY,
        help="Prediction entropy threshold for v50 adaptive compute.",
    )
    ap.add_argument(
        "--prediction_stability_patience",
        type=int,
        default=DEFAULT_PREDICTION_STABILITY_PATIENCE,
        help="Stable full-output predictions required before adaptive exit (0 disables).",
    )
    ap.add_argument(
        "--prediction_stability_tol",
        type=float,
        default=DEFAULT_PREDICTION_STABILITY_TOL,
        help="Maximum confidence drift allowed across a stable prediction streak.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--device_preference",
        default="cuda,npu,xpu,dml,mps,cpu",
        help="Priority order used when --device auto.",
    )
    ap.add_argument("--torch_num_threads", type=int, default=0, help="PyTorch intra-op CPU threads (0=auto).")
    ap.add_argument("--torch_interop_threads", type=int, default=0, help="PyTorch inter-op CPU threads (0=auto).")
    ap.add_argument(
        "--matmul_precision",
        choices=["highest", "high", "medium"],
        default="high",
        help="torch float32 matmul precision when supported.",
    )
    ap.add_argument("--disable_tf32", action="store_true", help="Disable TF32 on supported CUDA devices.")
    ap.add_argument("--max_turns", type=int, default=2, help="How many previous turns to include in context.")
    ap.add_argument("--top_labels", type=int, default=3, help="How many top predicted buckets to fuse for retrieval.")
    ap.add_argument("--llm_db", default="llm_chat.db", help="Optional local LLM retrieval DB (SQLite).")
    ap.add_argument("--db_top_k", type=int, default=120, help="Top DB candidates to include per turn.")
    ap.add_argument(
        "--db_query_context_turns",
        type=int,
        default=2,
        help="How many recent turns to use for conversation-aware DB query rewriting.",
    )
    ap.add_argument(
        "--db_score_scale",
        type=float,
        default=1.0,
        help="Scale factor for DB candidate confidence contribution.",
    )
    ap.add_argument(
        "--pool_mode",
        choices=["topk", "all"],
        default="all",
        help="Candidate pool source: top-k labels only or all labels weighted by classifier probability.",
    )
    ap.add_argument(
        "--response_temperature",
        type=float,
        default=0.10,
        help="Response sampling temperature over top reranked candidates; 0 disables sampling.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 for argmax bucket choice, >0 for sampling from class probabilities.",
    )
    ap.add_argument(
        "--style_mode",
        choices=["auto", "balanced", "creative", "concise", "analyst"],
        default="auto",
        help="Response style mode; auto infers from the user query.",
    )
    ap.add_argument(
        "--creativity",
        type=float,
        default=0.25,
        help="Creative rewrite strength in [0,1] when style is creative.",
    )
    ap.add_argument(
        "--show_top_responses",
        type=int,
        default=0,
        help="Print top reranked response candidates each turn (debug).",
    )
    ap.add_argument(
        "--show_timing",
        action="store_true",
        help="Print per-turn timing breakdown (memory/db/inference/total).",
    )
    ap.add_argument(
        "--auto_compute",
        action="store_true",
        help="Probe a small reasoning-cycle ladder and choose the first budget meeting confidence/entropy targets.",
    )
    ap.add_argument(
        "--auto_compute_confidence",
        type=float,
        default=DEFAULT_AUTO_COMPUTE_CONFIDENCE,
        help="Confidence target for --auto_compute early budget selection.",
    )
    ap.add_argument(
        "--auto_compute_entropy",
        type=float,
        default=DEFAULT_AUTO_COMPUTE_ENTROPY,
        help="Entropy target for --auto_compute early budget selection.",
    )
    ap.add_argument("--memory_db", default="chat_memory.db", help="Persistent memory SQLite DB.")
    ap.add_argument("--memory_top_k", type=int, default=4, help="Number of memory snippets to retrieve each turn.")
    ap.add_argument("--memory_pool_size", type=int, default=400, help="How many recent memory rows to score per turn.")
    ap.add_argument(
        "--memory_recency_half_life_hours",
        type=float,
        default=168.0,
        help="Recency half-life for memory ranking.",
    )
    ap.add_argument(
        "--memory_score_scale",
        type=float,
        default=0.45,
        help="Scale factor for memory-derived candidate confidence.",
    )
    ap.add_argument("--disable_memory", action="store_true", help="Disable persistent memory retrieval and writes.")
    args = ap.parse_args()

    configure_torch_runtime(
        torch_num_threads=int(args.torch_num_threads),
        torch_interop_threads=int(args.torch_interop_threads),
        allow_tf32=not bool(args.disable_tf32),
        matmul_precision=str(args.matmul_precision),
    )
    device, device_info = resolve_device(args.device, preference=args.device_preference)
    meta = load_metadata(args.meta)
    
    raw_feature_mode = str(meta.get("feature_mode", "legacy")).strip().lower()
    feature_mode = resolve_feature_mode(raw_feature_mode, smarter_auto=True)
    feature_mode_note = ""
    if feature_mode != raw_feature_mode:
        feature_mode_note = f" (auto-upgraded from {raw_feature_mode or 'legacy'})"
        
    sd = safe_load_state_dict(args.weights)
    inferred_from_weights = detect_model_size_from_state_dict(sd)
    resolved_model_size, model_size_warning = resolve_runtime_model_size(
        args.model_size,
        str(meta.get("model_size", "")),
        inferred_from_weights,
    )
    if model_size_warning:
        print(model_size_warning)

    # Resolve architectural dimensions via unified helper function
    expansion_dim = _resolve_expansion_dim(
        args.expansion_dim, meta, "expansion_dim",
        _default_expansion_dim_for_model_size(resolved_model_size),
        inferred_from_weights, EXPANSION_DIM_MODEL_SIZES,
        detect_large_head_expansion_dim, sd
    )

    extra_expansion_dim = _resolve_expansion_dim(
        args.extra_expansion_dim, meta, "extra_expansion_dim",
        _default_extra_expansion_dim_for_model_size(resolved_model_size, expansion_dim),
        inferred_from_weights, EXTRA_EXPANSION_DIM_MODEL_SIZES,
        detect_xlarge_aux_expansion_dim, sd
    )

    third_expansion_dim = _resolve_expansion_dim(
        args.third_expansion_dim, meta, "third_expansion_dim",
        max(3072, extra_expansion_dim + expansion_dim),
        inferred_from_weights, THIRD_EXPANSION_DIM_MODEL_SIZES,
        detect_xxlarge_third_expansion_dim, sd
    )

    fourth_expansion_dim = _resolve_expansion_dim(
        args.fourth_expansion_dim, meta, "fourth_expansion_dim",
        max(4096, third_expansion_dim + expansion_dim),
        inferred_from_weights, FOURTH_EXPANSION_DIM_MODEL_SIZES,
        detect_xxxlarge_fourth_expansion_dim, sd
    )

    fifth_expansion_dim = _resolve_expansion_dim(
        args.fifth_expansion_dim, meta, "fifth_expansion_dim",
        max(6144, fourth_expansion_dim + expansion_dim),
        inferred_from_weights, FIFTH_EXPANSION_DIM_MODEL_SIZES,
        detect_ultralarge_fifth_expansion_dim, sd
    )

    sixth_expansion_dim = _resolve_expansion_dim(
        args.sixth_expansion_dim, meta, "sixth_expansion_dim",
        max(8192, fifth_expansion_dim + expansion_dim),
        inferred_from_weights, SIXTH_EXPANSION_DIM_MODEL_SIZES,
        detect_megalarge_sixth_expansion_dim, sd
    )

    adapter_dropout = float(meta.get("adapter_dropout", 0.1)) if args.adapter_dropout is None else args.adapter_dropout

    model = build_model(
        model_size=resolved_model_size,
        expansion_dim=expansion_dim,
        dropout=adapter_dropout,
        extra_expansion_dim=extra_expansion_dim,
        third_expansion_dim=third_expansion_dim,
        fourth_expansion_dim=fourth_expansion_dim,
        fifth_expansion_dim=fifth_expansion_dim,
        sixth_expansion_dim=sixth_expansion_dim,
    ).to(device).eval()
    
    missing, unexpected = load_weights_for_model(model, sd, model_size=resolved_model_size)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. Missing={missing}, Unexpected={unexpected}")

    buckets_raw = meta.get("buckets", {})
    buckets: Dict[int, List[Dict]] = {}
    for k, v in buckets_raw.items():
        try:
            label = int(k)
        except ValueError:
            continue
        if isinstance(v, list) and v:
            buckets[label] = v

    available_labels = sorted(buckets.keys())
    if not available_labels:
        available_labels = list(range(MODEL_CLASSES))
        print(f"{TerminalColors.SYSTEM}Warning: metadata has no buckets; relying on DB/memory retrieval + classifier priors.{TerminalColors.RESET}")
        
    recent_assistant_messages: List[str] = []
    history: List[Tuple[str, str]] = []
    
    llm_db: Optional[LLMDatabase] = None
    if args.llm_db and Path(args.llm_db).exists():
        llm_db = LLMDatabase(str(Path(args.llm_db)))
    elif args.llm_db:
        print(f"{TerminalColors.SYSTEM}Warning: llm_db not found at {args.llm_db}; continuing without DB retrieval.{TerminalColors.RESET}")

    memory_db: Optional[ChatMemoryDB] = None
    if not args.disable_memory and args.memory_db:
        memory_db = ChatMemoryDB(args.memory_db)
    session_memory_enabled = memory_db is not None
    session_db_enabled = llm_db is not None
    show_timing = bool(args.show_timing)
    session_started_at = time.time()
    turn_count = 0
    last_turn_timing: Dict[str, float] = {}
    last_compute_metrics: Dict[str, Any] = {}
    if _is_auto_reasoning_cycles(args.reasoning_cycles):
        args.reasoning_cycles = "auto"
    else:
        args.reasoning_cycles = _coerce_optional_positive_int(
            args.reasoning_cycles,
            default=None,
            max_value=MAX_RUNTIME_REASONING_CYCLES,
        )
    args.adaptive_compute = bool(args.adaptive_compute)
    args.adaptive_exit_tol = _coerce_nonnegative_float(
        args.adaptive_exit_tol,
        default=DEFAULT_ADAPTIVE_EXIT_TOL,
    )
    args.auto_compute = bool(args.auto_compute)
    args.auto_compute_confidence = max(0.0, min(1.0, float(args.auto_compute_confidence)))
    args.auto_compute_entropy = max(0.0, float(args.auto_compute_entropy))
    args.adaptive_exit_entropy = _coerce_nonnegative_float(
        args.adaptive_exit_entropy,
        DEFAULT_ADAPTIVE_EXIT_ENTROPY,
    )
    args.prediction_stability_patience = _coerce_nonnegative_int(
        args.prediction_stability_patience,
        DEFAULT_PREDICTION_STABILITY_PATIENCE,
        MAX_RUNTIME_REASONING_CYCLES,
    )
    args.prediction_stability_tol = _coerce_nonnegative_float(
        args.prediction_stability_tol,
        DEFAULT_PREDICTION_STABILITY_TOL,
    )

    print(f"\n{TerminalColors.SYSTEM}--- Session Info ---")
    print(f"Loaded: {Path(args.weights).name} [{resolved_model_size}] | Available labels: {len(available_labels)}")
    print(f"Device: {device_info.get('resolved', args.device)} | Threads intra={torch.get_num_threads()} interop={torch.get_num_interop_threads()}")
    print(f"Feature mode: {feature_mode}{feature_mode_note} | Style mode: {args.style_mode} (creativity={max(0.0, min(1.0, float(args.creativity))):.2f})")
    compute_support = "yes" if model_supports_runtime_compute(model) else "no"
    print(
        "Runtime compute: "
        f"supported={compute_support} cycles={_format_reasoning_cycles_setting(args.reasoning_cycles)} "
        f"adaptive={'on' if bool(args.adaptive_compute) else 'off'} "
        f"exit_tol={_coerce_nonnegative_float(args.adaptive_exit_tol, DEFAULT_ADAPTIVE_EXIT_TOL):.4g} "
        f"exit_entropy={_coerce_nonnegative_float(args.adaptive_exit_entropy, DEFAULT_ADAPTIVE_EXIT_ENTROPY):.4g} "
        f"stability={_coerce_nonnegative_int(args.prediction_stability_patience, DEFAULT_PREDICTION_STABILITY_PATIENCE)} "
        f"stability_tol={_coerce_nonnegative_float(args.prediction_stability_tol, DEFAULT_PREDICTION_STABILITY_TOL):.4g} "
        f"auto={'on' if args.auto_compute else 'off'}"
    )
    
    if llm_db:
        print(f"LLM DB: {args.llm_db} (top_k={args.db_top_k})")
    if memory_db:
        print(f"Memory DB: {args.memory_db} (top_k={max(1, int(args.memory_top_k))}, pool={max(1, int(args.memory_pool_size))})")
    print(f"--------------------{TerminalColors.RESET}")
    print(f"{TerminalColors.BOT}Chat app ready. Type 'exit'/'quit' or use /help for commands.{TerminalColors.RESET}\n")

    # Initialize ThreadPoolExecutor for concurrent DB operations and computational inference
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while True:
            try:
                user = input(f"{TerminalColors.USER}You: {TerminalColors.RESET}").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{TerminalColors.SYSTEM}Closing chat...{TerminalColors.RESET}")
                break

            if not user:
                continue
            if user.lower() in {"exit", "quit"}:
                break
            if user.startswith("/"):
                cmdline = user[1:].strip()
                parts = cmdline.split()
                cmd = parts[0].lower() if parts else ""
                arg = parts[1] if len(parts) > 1 else ""

                if cmd in {"help", "h", "?"}:
                    _print_chat_help()
                elif cmd == "quit":
                    break
                elif cmd == "clear":
                    history.clear()
                    recent_assistant_messages.clear()
                    print(f"{TerminalColors.SYSTEM}Cleared session conversation history.{TerminalColors.RESET}")
                elif cmd == "stats":
                    uptime = time.time() - session_started_at
                    print(f"{TerminalColors.SYSTEM}Session stats:{TerminalColors.RESET}")
                    print(f"  turns={turn_count} history_turns={len(history)} uptime={_format_duration(uptime)}")
                    print(
                        f"  style={args.style_mode} creativity={float(args.creativity):.2f} "
                        f"top_debug={int(args.show_top_responses)} timing={'on' if show_timing else 'off'}"
                    )
                    print(
                        f"  compute_supported={'yes' if model_supports_runtime_compute(model) else 'no'} "
                        f"cycles={_format_reasoning_cycles_setting(args.reasoning_cycles)} "
                        f"adaptive={'on' if bool(args.adaptive_compute) else 'off'} "
                        f"exit_tol={_coerce_nonnegative_float(args.adaptive_exit_tol, 1e-3):.4g} "
                        f"exit_entropy={_coerce_nonnegative_float(args.adaptive_exit_entropy, DEFAULT_ADAPTIVE_EXIT_ENTROPY):.4g} "
                        f"stability={_coerce_nonnegative_int(args.prediction_stability_patience, DEFAULT_PREDICTION_STABILITY_PATIENCE)} "
                        f"stability_tol={_coerce_nonnegative_float(args.prediction_stability_tol, DEFAULT_PREDICTION_STABILITY_TOL):.4g} "
                        f"auto={'on' if args.auto_compute else 'off'}"
                    )
                    print(
                        f"  memory={'on' if session_memory_enabled else 'off'} "
                        f"(db={'ready' if memory_db is not None else 'missing'}) | "
                        f"llm_db={'on' if session_db_enabled else 'off'} "
                        f"(db={'ready' if llm_db is not None else 'missing'})"
                    )
                    if last_turn_timing:
                        print(
                            "  last_timing="
                            + ", ".join(f"{k}={_format_ms(v)}" for k, v in last_turn_timing.items())
                        )
                    if last_compute_metrics:
                        compact_compute = dict(last_compute_metrics)
                        plan = compact_compute.pop("auto_compute_plan", None)
                        if isinstance(plan, dict):
                            compact_compute.update(compact_auto_compute_plan_metrics(plan))
                        print(
                            "  last_compute="
                            + ", ".join(f"{k}={v}" for k, v in compact_compute.items())
                        )
                elif cmd == "config":
                    print(f"{TerminalColors.SYSTEM}Runtime config:{TerminalColors.RESET}")
                    print(
                        f"  style={args.style_mode} creativity={float(args.creativity):.2f} "
                        f"response_temp={float(args.response_temperature):.3f} class_temp={float(args.temperature):.3f}"
                    )
                    print(
                        f"  pool_mode={args.pool_mode} top_labels={int(args.top_labels)} "
                        f"show_top_responses={int(args.show_top_responses)}"
                    )
                    print(
                        f"  compute_supported={'yes' if model_supports_runtime_compute(model) else 'no'} "
                        f"cycles={_format_reasoning_cycles_setting(args.reasoning_cycles)} "
                        f"adaptive={'on' if bool(args.adaptive_compute) else 'off'} "
                        f"exit_tol={_coerce_nonnegative_float(args.adaptive_exit_tol, 1e-3):.4g} "
                        f"exit_entropy={_coerce_nonnegative_float(args.adaptive_exit_entropy, DEFAULT_ADAPTIVE_EXIT_ENTROPY):.4g} "
                        f"stability={_coerce_nonnegative_int(args.prediction_stability_patience, DEFAULT_PREDICTION_STABILITY_PATIENCE)} "
                        f"stability_tol={_coerce_nonnegative_float(args.prediction_stability_tol, DEFAULT_PREDICTION_STABILITY_TOL):.4g} "
                        f"auto={'on' if args.auto_compute else 'off'} "
                        f"auto_conf={float(args.auto_compute_confidence):.3f} "
                        f"auto_entropy={float(args.auto_compute_entropy):.3f}"
                    )
                    print(
                        f"  memory={'on' if session_memory_enabled else 'off'} top_k={int(args.memory_top_k)} "
                        f"pool={int(args.memory_pool_size)} score_scale={float(args.memory_score_scale):.3f}"
                    )
                    print(
                        f"  llm_db={'on' if session_db_enabled else 'off'} top_k={int(args.db_top_k)} "
                        f"score_scale={float(args.db_score_scale):.3f}"
                    )
                elif cmd == "style":
                    if arg not in {"auto", "balanced", "creative", "concise", "analyst"}:
                        print(f"{TerminalColors.SYSTEM}Usage: /style auto|balanced|creative|concise|analyst{TerminalColors.RESET}")
                    else:
                        args.style_mode = arg
                        print(f"{TerminalColors.SYSTEM}Style mode set to {arg}.{TerminalColors.RESET}")
                elif cmd == "creativity":
                    try:
                        args.creativity = max(0.0, min(1.0, float(arg)))
                        print(f"{TerminalColors.SYSTEM}Creativity set to {float(args.creativity):.2f}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /creativity 0.0-1.0{TerminalColors.RESET}")
                elif cmd == "cycles":
                    raw_cycles = arg.strip().lower()
                    if raw_cycles in {"auto", "adaptive", "smart"}:
                        args.reasoning_cycles = "auto"
                        print(f"{TerminalColors.SYSTEM}Reasoning cycles set to auto.{TerminalColors.RESET}")
                    else:
                        cycles = _coerce_optional_positive_int(arg)
                        if cycles is None and raw_cycles not in {"", "default", "off", "none"}:
                            print(f"{TerminalColors.SYSTEM}Usage: /cycles <positive-int|auto|default>{TerminalColors.RESET}")
                            continue
                        args.reasoning_cycles = cycles
                        label = cycles if cycles is not None else "default"
                        print(f"{TerminalColors.SYSTEM}Reasoning cycles set to {label}.{TerminalColors.RESET}")
                elif cmd == "adaptive":
                    if arg.lower() in {"on", "1", "true"}:
                        args.adaptive_compute = True
                    elif arg.lower() in {"off", "0", "false"}:
                        args.adaptive_compute = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /adaptive on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Adaptive compute {'enabled' if args.adaptive_compute else 'disabled'}.{TerminalColors.RESET}")
                elif cmd == "exit_tol":
                    try:
                        args.adaptive_exit_tol = _coerce_nonnegative_float(arg, args.adaptive_exit_tol)
                        print(f"{TerminalColors.SYSTEM}Adaptive exit tolerance set to {float(args.adaptive_exit_tol):.4g}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /exit_tol <float>{TerminalColors.RESET}")
                elif cmd == "exit_entropy":
                    try:
                        args.adaptive_exit_entropy = _coerce_nonnegative_float(
                            arg,
                            args.adaptive_exit_entropy,
                        )
                        print(f"{TerminalColors.SYSTEM}Adaptive exit entropy set to {float(args.adaptive_exit_entropy):.4g}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /exit_entropy <float>{TerminalColors.RESET}")
                elif cmd == "stability":
                    try:
                        args.prediction_stability_patience = _coerce_nonnegative_int(
                            arg,
                            args.prediction_stability_patience,
                            MAX_RUNTIME_REASONING_CYCLES,
                        )
                        print(
                            f"{TerminalColors.SYSTEM}Prediction stability patience set to "
                            f"{int(args.prediction_stability_patience)}.{TerminalColors.RESET}"
                        )
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /stability <nonnegative-int>{TerminalColors.RESET}")
                elif cmd == "stability_tol":
                    try:
                        args.prediction_stability_tol = _coerce_nonnegative_float(
                            arg,
                            args.prediction_stability_tol,
                        )
                        print(
                            f"{TerminalColors.SYSTEM}Prediction stability tolerance set to "
                            f"{float(args.prediction_stability_tol):.4g}.{TerminalColors.RESET}"
                        )
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /stability_tol <float>{TerminalColors.RESET}")
                elif cmd == "top":
                    try:
                        args.show_top_responses = max(0, int(arg))
                        print(f"{TerminalColors.SYSTEM}Top-candidate debug set to {int(args.show_top_responses)}.{TerminalColors.RESET}")
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /top <int>{TerminalColors.RESET}")
                elif cmd == "timing":
                    if arg.lower() in {"on", "1", "true"}:
                        show_timing = True
                    elif arg.lower() in {"off", "0", "false"}:
                        show_timing = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /timing on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Per-turn timing {'enabled' if show_timing else 'disabled'}.{TerminalColors.RESET}")
                elif cmd in {"auto_compute", "auto"}:
                    if arg.lower() in {"on", "1", "true", "yes"}:
                        args.auto_compute = True
                    elif arg.lower() in {"off", "0", "false", "no"}:
                        args.auto_compute = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /auto_compute on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Auto compute {'enabled' if args.auto_compute else 'disabled'}.{TerminalColors.RESET}")
                elif cmd in {"auto_targets", "compute_targets"}:
                    try:
                        if len(parts) < 3:
                            raise ValueError
                        args.auto_compute_confidence = max(0.0, min(1.0, float(parts[1])))
                        args.auto_compute_entropy = max(0.0, float(parts[2]))
                        print(
                            f"{TerminalColors.SYSTEM}Auto compute targets set to "
                            f"confidence={float(args.auto_compute_confidence):.3f}, "
                            f"entropy={float(args.auto_compute_entropy):.3f}.{TerminalColors.RESET}"
                        )
                    except Exception:
                        print(f"{TerminalColors.SYSTEM}Usage: /auto_targets <confidence 0-1> <entropy>{TerminalColors.RESET}")
                elif cmd == "memory":
                    if memory_db is None and arg.lower() in {"on", "1", "true"}:
                        print(f"{TerminalColors.SYSTEM}Memory DB is not available in this session.{TerminalColors.RESET}")
                        continue
                    if arg.lower() in {"on", "1", "true"}:
                        session_memory_enabled = True
                    elif arg.lower() in {"off", "0", "false"}:
                        session_memory_enabled = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /memory on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}Memory retrieval/writes {'enabled' if session_memory_enabled else 'disabled'} for this session.{TerminalColors.RESET}")
                elif cmd == "db":
                    if llm_db is None and arg.lower() in {"on", "1", "true"}:
                        print(f"{TerminalColors.SYSTEM}LLM DB is not available in this session.{TerminalColors.RESET}")
                        continue
                    if arg.lower() in {"on", "1", "true"}:
                        session_db_enabled = True
                    elif arg.lower() in {"off", "0", "false"}:
                        session_db_enabled = False
                    else:
                        print(f"{TerminalColors.SYSTEM}Usage: /db on|off{TerminalColors.RESET}")
                        continue
                    print(f"{TerminalColors.SYSTEM}LLM DB retrieval {'enabled' if session_db_enabled else 'disabled'} for this session.{TerminalColors.RESET}")
                else:
                    print(f"{TerminalColors.SYSTEM}Unknown command: /{cmd}. Use /help.{TerminalColors.RESET}")
                continue

            turn_t0 = time.perf_counter()
            t_memory = 0.0
            t_db_wait = 0.0
            t_infer = 0.0
            t_rank = 0.0

            # 1. Evaluate memory queries synchronously (needed for model context & LLM DB query)
            memory_rows: List[Dict] = []
            if session_memory_enabled and memory_db is not None:
                _t = time.perf_counter()
                memory_rows = memory_db.query(
                    user,
                    top_k=max(1, int(args.memory_top_k)),
                    pool_size=max(1, int(args.memory_pool_size)),
                    recency_half_life_hours=max(1.0, float(args.memory_recency_half_life_hours)),
                )
                t_memory += max(0.0, time.perf_counter() - _t)

            # 2. Fire off background Thread for heavy LLM DB querying
            future_llm_db = None
            if session_db_enabled and llm_db is not None:
                db_query = _build_db_query(
                    user=user,
                    history=history,
                    memory_rows=memory_rows,
                    max_turns=max(0, int(args.db_query_context_turns)),
                )
                future_llm_db = executor.submit(llm_db.query, db_query or user, top_k=max(1, args.db_top_k))

            # 3. Proceed with model context building & inference while thread waits on DB IO
            _t = time.perf_counter()
            context = build_context(history, user_text=user, max_turns=args.max_turns)
            if memory_rows:
                memory_block = render_memory_block(memory_rows)
                if memory_block:
                    context = memory_block + "\n" + context
                    
            x = text_to_model_input(context, feature_mode=feature_mode).to(device)
            effective_reasoning_cycles = args.reasoning_cycles
            auto_compute_plan = None
            if args.auto_compute and model_supports_runtime_compute(model):
                model_out, last_compute_metrics, auto_compute_plan = progressive_auto_compute_forward(
                    model,
                    x,
                    available_labels,
                    cycles=runtime_auto_compute_cycles(args.reasoning_cycles),
                    confidence_target=args.auto_compute_confidence,
                    entropy_target=args.auto_compute_entropy,
                    adaptive_compute=args.adaptive_compute,
                    exit_tol=args.adaptive_exit_tol,
                    exit_entropy_threshold=args.adaptive_exit_entropy,
                    prediction_stability_patience=args.prediction_stability_patience,
                    prediction_stability_tol=args.prediction_stability_tol,
                    auto_reasoning_context=context,
                )
                effective_reasoning_cycles = auto_compute_plan.get("selected_reasoning_cycles")
            else:
                with torch.no_grad():
                    model_out, last_compute_metrics = forward_with_runtime_compute(
                        model,
                        x,
                        reasoning_cycles=effective_reasoning_cycles,
                        adaptive_compute=args.adaptive_compute,
                        exit_tol=args.adaptive_exit_tol,
                        exit_entropy_threshold=args.adaptive_exit_entropy,
                        prediction_stability_patience=args.prediction_stability_patience,
                        prediction_stability_tol=args.prediction_stability_tol,
                        auto_reasoning_context=context,
                    )
            logits = model_out[0, 0]  # (10,)
            t_infer += max(0.0, time.perf_counter() - _t)

            idx = torch.tensor(available_labels, dtype=torch.long, device=logits.device)
            avail_logits = logits.index_select(0, idx)
            probs = torch.softmax(avail_logits, dim=0)
            
            if args.pool_mode == "all":
                top_pos = list(range(len(available_labels)))
            else:
                k = max(1, min(args.top_labels, len(available_labels)))
                top_pos = torch.topk(avail_logits, k=k).indices.tolist()

            pooled_candidates: List[Dict] = []
            for pos in top_pos:
                label = available_labels[int(pos)]
                bucket_score = float(probs[int(pos)].item())
                for row in buckets.get(label, []):
                    merged = dict(row)
                    merged["bucket_score"] = bucket_score
                    merged["_source"] = "model"
                    pooled_candidates.append(merged)

            # 4. Await LLM DB fetch result and ingest
            if future_llm_db is not None:
                _t = time.perf_counter()
                db_candidates = future_llm_db.result()
                t_db_wait += max(0.0, time.perf_counter() - _t)
                for row in db_candidates:
                    merged = dict(row)
                    merged["bucket_score"] = float(merged.get("bucket_score", 0.0)) * float(args.db_score_scale)
                    merged["_source"] = "llm_db"
                    pooled_candidates.append(merged)

            # Memory candidates help continuity and long-term preference alignment.
            if memory_rows:
                for row in memory_rows:
                    text = str(row.get("assistant_text", "")).strip()
                    vec = row.get("assistant_vec")
                    ctx_vec = row.get("user_vec")
                    if not text or not isinstance(vec, list) or not isinstance(ctx_vec, list):
                        continue
                    pooled_candidates.append(
                        {
                            "text": text,
                            "count": 1,
                            "vec": vec,
                            "ctx_vec": ctx_vec,
                            "bucket_score": float(max(0.0, float(row.get("score", 0.0))) * float(args.memory_score_scale)),
                            "_source": "memory"
                        }
                    )

            # Deduplicate responses and Ensemble Boost Cross-Validated candidates
            dedup: Dict[str, Dict] = {}
            for row in pooled_candidates:
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                
                prev = dedup.get(text)
                if prev is None:
                    dedup[text] = row
                    dedup[text]["_sources_set"] = {row.get("_source", "unknown")}
                    continue
                
                # Boost algorithm: 10% bonus if validated by multiple sources
                source = row.get("_source", "unknown")
                base_score = max(float(prev.get("bucket_score", 0.0)), float(row.get("bucket_score", 0.0)))
                
                if source not in prev["_sources_set"]:
                    base_score *= 1.10
                    prev["_sources_set"].add(source)

                prev["bucket_score"] = base_score
                prev["count"] = int(prev.get("count", 1)) + int(row.get("count", 1))
            
            # Clean up temporary boost tracker vars
            for k in dedup:
                dedup[k].pop("_sources_set", None)
                dedup[k].pop("_source", None)

            pooled_candidates = list(dedup.values())

            if (not pooled_candidates) and buckets:
                label = choose_bucket_from_logits(logits, available_labels, temperature=args.temperature)
                pooled_candidates = list(buckets.get(label, []))

            resolved_style = infer_style_mode(user, requested_mode=args.style_mode)

            if args.show_top_responses > 0 and pooled_candidates:
                _t = time.perf_counter()
                ranked, scores = rank_response_candidates(
                    pooled_candidates,
                    query_text=user,
                    recent_assistant_messages=recent_assistant_messages,
                    style_mode=resolved_style,
                )
                t_rank += max(0.0, time.perf_counter() - _t)
                n_show = max(1, min(int(args.show_top_responses), len(ranked)))
                print(f"{TerminalColors.SYSTEM}Top candidates:{TerminalColors.RESET}")
                shown = 0
                for ridx in ranked:
                    cand_text = str(pooled_candidates[ridx].get("text", "")).strip()
                    if not cand_text:
                        continue
                    shown += 1
                    preview = cand_text if len(cand_text) <= 120 else (cand_text[:117] + "...")
                    print(f"  {shown}. ({float(scores[ridx].item()):.3f}) {preview}")
                    if shown >= n_show:
                        break

            _t = time.perf_counter()
            response = pick_response(
                pooled_candidates,
                query_text=user,
                recent_assistant_messages=recent_assistant_messages,
                response_temperature=args.response_temperature,
                style_mode=resolved_style,
                creativity=max(0.0, min(1.0, float(args.creativity))),
            )
            t_rank += max(0.0, time.perf_counter() - _t)
            response = cleanup_response_text(response)
            if not response:
                response = "I do not have a trained response for that yet."

            print(f"{TerminalColors.BOT}Bot: {TerminalColors.RESET}{response}")
            history.append((user, response))
            recent_assistant_messages.append(response)
            turn_count += 1
            
            if session_memory_enabled and memory_db is not None:
                memory_db.add_turn(user, response)

            total_turn = max(0.0, time.perf_counter() - turn_t0)
            last_turn_timing = {
                "memory": t_memory,
                "db_wait": t_db_wait,
                "infer": t_infer,
                "rank_pick": t_rank,
                "total": total_turn,
            }
            if show_timing:
                print(
                    f"{TerminalColors.SYSTEM}Timing:{TerminalColors.RESET} "
                    + ", ".join(f"{k}={_format_ms(v)}" for k, v in last_turn_timing.items())
                )
                if last_compute_metrics:
                    requested = last_compute_metrics.get("requested_reasoning_cycles") or "model default"
                    plan = last_compute_metrics.get("auto_compute_plan") if isinstance(last_compute_metrics, dict) else None
                    plan_suffix = ""
                    if isinstance(plan, dict):
                        plan_metrics = compact_auto_compute_plan_metrics(plan)
                        plan_suffix = " " + " ".join(
                            f"{key}={value}" for key, value in plan_metrics.items()
                        )
                    print(
                        f"{TerminalColors.SYSTEM}Compute:{TerminalColors.RESET} "
                        f"supported={last_compute_metrics.get('supported')} "
                        f"requested={requested} cycles_used={last_compute_metrics.get('cycles_used', 'n/a')} "
                        f"adaptive={last_compute_metrics.get('adaptive_compute')} "
                        f"applied={last_compute_metrics.get('applied')} "
                        f"ponder={last_compute_metrics.get('ponder_cost')} "
                        f"gate_entropy={last_compute_metrics.get('gating_entropy')} "
                        f"exit={last_compute_metrics.get('exit_reason')}{plan_suffix}"
                    )
                
    finally:
        executor.shutdown(wait=False)
        if llm_db is not None:
            llm_db.close()
        if memory_db is not None:
            memory_db.close()

    print(f"{TerminalColors.SYSTEM}Session ended.{TerminalColors.RESET}")


if __name__ == "__main__":
    main()
