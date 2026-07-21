from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


TOKEN_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)
MULTISPACE_RE = re.compile(r"\s+")
ROUTE_MODES = ("off", "collective", "loop", "collective_loop")
ROUTE_FEEDBACK_DECAY = 0.6
ROUTE_FEEDBACK_CONFIDENCE_Z = 1.645
ROUTE_PREFERENCE_SIGNAL_FLOOR = 0.75
ROUTE_EVIDENCE_INTERVAL_KIND = "recency_weighted_wilson_heuristic"
FEEDBACK_RATINGS = {
    "up": ("up", 1),
    "good": ("up", 1),
    "approve": ("up", 1),
    "approved": ("up", 1),
    "positive": ("up", 1),
    "down": ("down", -1),
    "bad": ("down", -1),
    "reject": ("down", -1),
    "rejected": ("down", -1),
    "negative": ("down", -1),
}
FEEDBACK_INTENTS = {
    "good": {
        "quality": 1,
        "depth_preference": 0,
        "cost_pressure": 0,
        "latency_pressure": 0,
    },
    "bad_quality": {
        "quality": -1,
        "depth_preference": 0,
        "cost_pressure": 0,
        "latency_pressure": 0,
    },
    "needs_deeper": {
        "quality": None,
        "depth_preference": 1,
        "cost_pressure": 0,
        "latency_pressure": 0,
    },
    "too_costly": {
        "quality": None,
        "depth_preference": -1,
        "cost_pressure": 1,
        "latency_pressure": 0,
    },
    "too_slow": {
        "quality": None,
        "depth_preference": -1,
        "cost_pressure": 0,
        "latency_pressure": 1,
    },
}
FEEDBACK_INTENT_ALIASES = {
    "up": "good",
    "positive": "good",
    "approve": "good",
    "approved": "good",
    "bad": "bad_quality",
    "down": "bad_quality",
    "negative": "bad_quality",
    "deeper": "needs_deeper",
    "more_depth": "needs_deeper",
    "cost": "too_costly",
    "expensive": "too_costly",
    "slow": "too_slow",
    "latency": "too_slow",
}

MEMORY_PATTERNS = (
    (
        re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z '\-]{0,48})", re.IGNORECASE),
        "identity",
        lambda match: f"User name: {match.group(1).strip()}",
    ),
    (
        re.compile(r"\bcall me\s+([A-Za-z][A-Za-z '\-]{0,48})", re.IGNORECASE),
        "identity",
        lambda match: f"Preferred name: {match.group(1).strip()}",
    ),
    (
        re.compile(r"\bi prefer\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"User preference: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bplease use\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"Preferred approach: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bi(?:'m| am) working on\s+(.{3,120})", re.IGNORECASE),
        "project",
        lambda match: f"Current project: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bthis project is\s+(.{3,120})", re.IGNORECASE),
        "project",
        lambda match: f"Project detail: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bremember that\s+(.{3,160})", re.IGNORECASE),
        "fact",
        lambda match: f"Remembered fact: {match.group(1).strip().rstrip('.!?')}",
    ),
    (
        re.compile(r"\bi like\s+(.{3,120})", re.IGNORECASE),
        "preference",
        lambda match: f"User likes: {match.group(1).strip().rstrip('.!?')}",
    ),
)


def _norm(text: str, limit: int = 260) -> str:
    cooked = MULTISPACE_RE.sub(" ", str(text or "").strip())
    return cooked[:limit]


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(text or ""))}


def _safe_slug(text: str) -> str:
    cooked = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text or ""))
    cooked = "-".join(part for part in cooked.split("-") if part)
    return cooked[:80] or "session"


def _now_ts() -> float:
    return float(time.time())


def _safe_float(value: Any, *, limit: float = 1_000_000.0, digits: int = 3) -> Optional[float]:
    try:
        cooked = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(cooked):
        return None
    return round(max(0.0, min(limit, cooked)), digits)


def _safe_int(value: Any, *, limit: int = 1_000_000) -> Optional[int]:
    cooked = _safe_float(value, limit=float(limit), digits=0)
    if cooked is None:
        return None
    return int(cooked)


def _compact_route_economics(raw: Any, *, trace: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    if not source and isinstance(trace.get("route_economics"), dict):
        source = trace["route_economics"]

    estimate_src = source.get("estimate") if isinstance(source.get("estimate"), dict) else {}
    actual_src = source.get("actual") if isinstance(source.get("actual"), dict) else {}
    if not estimate_src and isinstance(policy.get("route_economics_estimate"), dict):
        estimate_src = policy["route_economics_estimate"]
    if not actual_src and isinstance(policy.get("route_economics_actual"), dict):
        actual_src = policy["route_economics_actual"]

    estimate: Dict[str, Any] = {}
    for key in ("selected_agent_mode", "action_mode", "budget_profile", "cost_preference", "latency_tier"):
        text = _norm(estimate_src.get(key) or "", limit=80)
        if text:
            estimate[key] = text
    for key in ("estimated_model_calls", "estimated_tool_calls", "planned_loop_steps", "collective_model_count"):
        value = _safe_int(estimate_src.get(key), limit=10_000)
        if value is not None:
            estimate[key] = value
    cost = _safe_float(estimate_src.get("estimated_cost_units"), limit=100_000.0, digits=3)
    if cost is not None:
        estimate["estimated_cost_units"] = cost

    actual: Dict[str, Any] = {}
    for key in ("latency_tier",):
        text = _norm(actual_src.get(key) or "", limit=80)
        if text:
            actual[key] = text
    for key in ("model_calls", "tool_calls", "loop_steps", "consultation_count"):
        value = _safe_int(actual_src.get(key), limit=10_000)
        if value is not None:
            actual[key] = value
    for key in ("elapsed_ms", "cost_units"):
        value = _safe_float(actual_src.get(key), limit=100_000_000.0 if key == "elapsed_ms" else 100_000.0, digits=3)
        if value is not None:
            actual[key] = value

    compact: Dict[str, Any] = {}
    if estimate:
        compact["estimate"] = estimate
    if actual:
        compact["actual"] = actual
    return compact


def _route_economics_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    actual_rows: List[Dict[str, Any]] = []
    for row in rows:
        economics = row.get("route_economics") if isinstance(row.get("route_economics"), dict) else {}
        actual = economics.get("actual") if isinstance(economics.get("actual"), dict) else {}
        if actual:
            actual_rows.append(actual)

    def avg(key: str) -> Optional[float]:
        values = [
            float(value)
            for value in (actual.get(key) for actual in actual_rows)
            if isinstance(value, (int, float))
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    cost_values = [
        float(value)
        for value in (actual.get("cost_units") for actual in actual_rows)
        if isinstance(value, (int, float))
    ]
    latency_values = [
        float(value)
        for value in (actual.get("elapsed_ms") for actual in actual_rows)
        if isinstance(value, (int, float))
    ]
    summary: Dict[str, Any] = {"sample_count": len(actual_rows)}
    if cost_values:
        summary["avg_cost_units"] = round(sum(cost_values) / len(cost_values), 3)
        summary["total_cost_units"] = round(sum(cost_values), 3)
        summary["max_cost_units"] = round(max(cost_values), 3)
    if latency_values:
        summary["avg_elapsed_ms"] = round(sum(latency_values) / len(latency_values), 3)
        summary["max_elapsed_ms"] = round(max(latency_values), 3)
    model_calls = avg("model_calls")
    if model_calls is not None:
        summary["avg_model_calls"] = model_calls
    tool_calls = avg("tool_calls")
    if tool_calls is not None:
        summary["avg_tool_calls"] = tool_calls
    return summary


def _feedback_axes(row: Dict[str, Any]) -> Dict[str, Any]:
    stored = row.get("feedback_axes") if isinstance(row.get("feedback_axes"), dict) else {}
    if stored:
        quality_raw = stored.get("quality")
        quality = None if quality_raw is None else max(-1, min(1, int(quality_raw)))
        return {
            "quality": quality,
            "depth_preference": max(-1, min(1, int(stored.get("depth_preference") or 0))),
            "cost_pressure": max(0, min(1, int(stored.get("cost_pressure") or 0))),
            "latency_pressure": max(0, min(1, int(stored.get("latency_pressure") or 0))),
        }

    # Old route-feedback rows only had a scalar score. Treat them as generic
    # quality feedback so existing sessions keep their previous semantics.
    return {
        "quality": max(-1, min(1, int(row.get("score_delta") or 0))),
        "depth_preference": 0,
        "cost_pressure": 0,
        "latency_pressure": 0,
    }


def _weighted_wilson_interval(
    *,
    positive_weight: float,
    total_weight: float,
    squared_weight: float,
    z: float = ROUTE_FEEDBACK_CONFIDENCE_Z,
) -> Dict[str, Optional[float]]:
    if total_weight <= 0.0 or squared_weight <= 0.0:
        return {
            "effective_sample_size": 0.0,
            "quality_lower_bound": None,
            "quality_upper_bound": None,
            "quality_confidence_width": None,
        }
    effective_n = (total_weight * total_weight) / squared_weight
    probability = max(0.0, min(1.0, positive_weight / total_weight))
    z2 = z * z
    denominator = 1.0 + (z2 / effective_n)
    center = (probability + (z2 / (2.0 * effective_n))) / denominator
    radius = (
        z
        * math.sqrt(
            max(0.0, (probability * (1.0 - probability) / effective_n) + (z2 / (4.0 * effective_n * effective_n)))
        )
        / denominator
    )
    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    return {
        "effective_sample_size": round(effective_n, 3),
        "quality_lower_bound": round(lower, 3),
        "quality_upper_bound": round(upper, 3),
        "quality_confidence_width": round(upper - lower, 3),
    }


def _compact_auto_agent_policy(policy: Dict[str, Any], selected_mode: str) -> Dict[str, Any]:
    raw_allowed_modes = policy.get("allowed_agent_modes") or policy.get("eligible_agent_modes") or []
    if not isinstance(raw_allowed_modes, (list, tuple)):
        raw_allowed_modes = []
    allowed_modes = [
        _norm(item, limit=48)
        for item in raw_allowed_modes
        if _norm(item, limit=48) in ROUTE_MODES
    ]
    action_probabilities: Dict[str, float] = {}
    raw_probabilities = policy.get("action_probabilities")
    if isinstance(raw_probabilities, dict):
        for mode in ROUTE_MODES:
            value = raw_probabilities.get(mode)
            if isinstance(value, (int, float)):
                action_probabilities[mode] = round(max(0.0, min(1.0, float(value))), 6)
    post_filter_probabilities: Dict[str, float] = {}
    raw_post_filter_probabilities = policy.get("post_filter_action_probabilities")
    if isinstance(raw_post_filter_probabilities, dict):
        for mode in ROUTE_MODES:
            value = raw_post_filter_probabilities.get(mode)
            if isinstance(value, (int, float)):
                post_filter_probabilities[mode] = round(max(0.0, min(1.0, float(value))), 6)
    raw_eligible_actions = policy.get("eligible_actions")
    eligible_actions = [
        _norm(item, limit=48)
        for item in (raw_eligible_actions if isinstance(raw_eligible_actions, (list, tuple)) else [])
        if _norm(item, limit=48) in ROUTE_MODES
    ]
    raw_context = policy.get("decision_context") if isinstance(policy.get("decision_context"), dict) else {}
    decision_context = {
        "action_mode": _norm(raw_context.get("action_mode") or policy.get("action_mode") or "", limit=32),
        "budget_profile": _norm(raw_context.get("budget_profile") or policy.get("budget_profile") or "", limit=32),
        "score": raw_context.get("score", policy.get("score")),
        "allowed_agent_modes": [
            _norm(item, limit=48)
            for item in (
                raw_context.get("allowed_agent_modes")
                if isinstance(raw_context.get("allowed_agent_modes"), (list, tuple))
                else allowed_modes
            )
            if _norm(item, limit=48) in ROUTE_MODES
        ],
    }
    logging_propensity = policy.get("logging_propensity")
    logging_support = policy.get("logging_support") if isinstance(policy.get("logging_support"), dict) else {}
    return {
        "policy_id": _norm(policy.get("policy_id") or "", limit=80),
        "policy_version": _norm(policy.get("policy_version") or "", limit=80),
        "feature_schema_version": _norm(policy.get("feature_schema_version") or "", limit=80),
        "decision_type": _norm(policy.get("decision_type") or "legacy_unknown", limit=40),
        "action_mode": _norm(policy.get("action_mode") or "", limit=32),
        "score": policy.get("score"),
        "score_before_budget": policy.get("score_before_budget"),
        "reason": _norm(policy.get("reason") or "", limit=120),
        "reasons": [_norm(item, limit=80) for item in list(policy.get("reasons") or [])[:6]],
        "base_selected_agent_mode": _norm(policy.get("base_selected_agent_mode") or "", limit=48),
        "selected_agent_mode": _norm(policy.get("selected_agent_mode") or selected_mode, limit=48),
        "allowed_agent_modes": allowed_modes,
        "eligible_actions": eligible_actions,
        "budget_profile": _norm(policy.get("budget_profile") or "", limit=32),
        "logging_propensity": (
            round(max(0.0, min(1.0, float(logging_propensity))), 6)
            if isinstance(logging_propensity, (int, float))
            else None
        ),
        "action_probabilities": action_probabilities,
        "post_filter_action_probabilities": post_filter_probabilities,
        "probability_stage": _norm(policy.get("probability_stage") or "", limit=32),
        "support_schema_version": _norm(logging_support.get("schema_version") or "", limit=80),
        "candidate_set_hash": _norm(policy.get("candidate_set_hash") or "", limit=80),
        "distribution_hash": _norm(policy.get("distribution_hash") or "", limit=80),
        "decision_context": decision_context,
    }


def _route_adaptive_feedback_summary(
    rows: Sequence[Dict[str, Any]],
    economics: Optional[Dict[str, Any]] = None,
    *,
    decay: float = ROUTE_FEEDBACK_DECAY,
) -> Dict[str, Any]:
    if not rows:
        return {
            "sample_count": 0,
            "quality_sample_count": 0,
            "decay": decay,
            "weighted_count": 0.0,
            "weighted_net": 0.0,
            "weighted_positive": 0.0,
            "weighted_negative": 0.0,
            "quality_score": None,
            "quality_cost_score": None,
            "effective_sample_size": 0.0,
            "quality_lower_bound": None,
            "quality_upper_bound": None,
            "quality_confidence_width": None,
            "quality_cost_lower_bound": None,
            "quality_cost_upper_bound": None,
            "confidence_level": 0.9,
            "confidence_status": "no_evidence",
            "interval_kind": ROUTE_EVIDENCE_INTERVAL_KIND,
            "coverage_claim": "heuristic_associational_only",
            "effective_sample_size_ceiling": round((1.0 + decay) / (1.0 - decay), 3),
            "weighted_depth_preference": 0.0,
            "weighted_cost_pressure": 0.0,
            "weighted_latency_pressure": 0.0,
            "preference_direction": None,
            "preference_signal": False,
            "recent_negative_rate": None,
            "regression_signal": False,
        }

    weighted_count = 0.0
    squared_weight = 0.0
    weighted_net = 0.0
    weighted_positive = 0.0
    weighted_negative = 0.0
    weighted_depth_preference = 0.0
    weighted_cost_pressure = 0.0
    weighted_latency_pressure = 0.0
    quality_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        age = len(rows) - idx - 1
        weight = float(decay) ** age
        axes = _feedback_axes(row)
        weighted_depth_preference += weight * int(axes["depth_preference"] or 0)
        weighted_cost_pressure += weight * int(axes["cost_pressure"] or 0)
        weighted_latency_pressure += weight * int(axes["latency_pressure"] or 0)
        if axes["quality"] is None:
            continue
        delta = int(axes["quality"] or 0)
        quality_rows.append(row)
        weighted_count += weight
        squared_weight += weight * weight
        weighted_net += weight * delta
        if delta > 0:
            weighted_positive += weight
        elif delta < 0:
            weighted_negative += weight

    recent_tail = quality_rows[-3:]
    recent_negative = sum(1 for row in recent_tail if int(_feedback_axes(row)["quality"] or 0) < 0)
    recent_negative_rate = round(recent_negative / len(recent_tail), 3) if recent_tail else None
    quality_score = None
    if weighted_count > 0:
        quality_score = round(max(0.0, min(1.0, ((weighted_net / weighted_count) + 1.0) / 2.0)), 3)

    confidence = _weighted_wilson_interval(
        positive_weight=weighted_positive,
        total_weight=weighted_count,
        squared_weight=squared_weight,
    )
    effective_sample_size = float(confidence["effective_sample_size"] or 0.0)
    if effective_sample_size >= 3.0 and len(quality_rows) >= 4:
        confidence_status = "established"
    elif effective_sample_size >= 1.5 and len(quality_rows) >= 2:
        confidence_status = "emerging"
    elif quality_rows:
        confidence_status = "sparse"
    else:
        confidence_status = "no_quality_evidence"

    quality_cost_score = None
    quality_cost_lower_bound = None
    quality_cost_upper_bound = None
    avg_cost = None
    if isinstance(economics, dict):
        raw_cost = economics.get("avg_cost_units")
        if isinstance(raw_cost, (int, float)):
            avg_cost = float(raw_cost)
    if quality_score is not None:
        cost_divisor = 1.0 + (max(0.0, avg_cost or 0.0) / 10.0)
        quality_cost_score = round(quality_score / cost_divisor, 3)
        lower = confidence.get("quality_lower_bound")
        upper = confidence.get("quality_upper_bound")
        if isinstance(lower, (int, float)):
            quality_cost_lower_bound = round(float(lower) / cost_divisor, 3)
        if isinstance(upper, (int, float)):
            quality_cost_upper_bound = round(float(upper) / cost_divisor, 3)

    if weighted_depth_preference >= ROUTE_PREFERENCE_SIGNAL_FLOOR:
        preference_direction = "deeper"
    elif weighted_depth_preference <= -ROUTE_PREFERENCE_SIGNAL_FLOOR:
        preference_direction = "shallower"
    else:
        preference_direction = None

    regression_signal = bool(
        len(quality_rows) >= 3
        and weighted_negative >= 1.45
        and weighted_net <= -0.25
        and (recent_negative_rate or 0.0) >= 0.667
    )
    return {
        "sample_count": len(rows),
        "quality_sample_count": len(quality_rows),
        "decay": decay,
        "weighted_count": round(weighted_count, 3),
        "weighted_net": round(weighted_net, 3),
        "weighted_positive": round(weighted_positive, 3),
        "weighted_negative": round(weighted_negative, 3),
        "quality_score": quality_score,
        "quality_cost_score": quality_cost_score,
        **confidence,
        "quality_cost_lower_bound": quality_cost_lower_bound,
        "quality_cost_upper_bound": quality_cost_upper_bound,
        "confidence_level": 0.9,
        "confidence_status": confidence_status,
        "interval_kind": ROUTE_EVIDENCE_INTERVAL_KIND,
        "coverage_claim": "heuristic_associational_only",
        "effective_sample_size_ceiling": round((1.0 + decay) / (1.0 - decay), 3),
        "weighted_depth_preference": round(weighted_depth_preference, 3),
        "weighted_cost_pressure": round(weighted_cost_pressure, 3),
        "weighted_latency_pressure": round(weighted_latency_pressure, 3),
        "preference_direction": preference_direction,
        "preference_signal": preference_direction is not None,
        "recent_negative_rate": recent_negative_rate,
        "regression_signal": regression_signal,
    }


class ConversationMemoryStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        session_digest = hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()
        return self.root_dir / f"{_safe_slug(session_id)}-{session_digest}.json"

    def _legacy_path_for(self, session_id: str) -> Path:
        return self.root_dir / f"{_safe_slug(session_id)}.json"

    @staticmethod
    def _read_session_file(path: Path) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _empty_session(session_id: str) -> Dict[str, Any]:
        now = _now_ts()
        return {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "memories": [],
            "turns": [],
            "route_feedback": [],
            "route_usage": [],
        }

    def load_session(self, session_id: str) -> Dict[str, Any]:
        path = self._path_for(session_id)
        if not path.exists():
            legacy_path = self._legacy_path_for(session_id)
            legacy_payload = self._read_session_file(legacy_path) if legacy_path.exists() else None
            if legacy_payload is None or legacy_payload.get("session_id") != session_id:
                return self._empty_session(session_id)
            self.save_session(session_id, legacy_payload)
            legacy_path.unlink(missing_ok=True)
            payload = self._read_session_file(path)
        else:
            payload = self._read_session_file(path)
        if payload is None or payload.get("session_id") != session_id:
            return self._empty_session(session_id)
        payload.setdefault("created_at", _now_ts())
        payload.setdefault("updated_at", payload["created_at"])
        payload.setdefault("memories", [])
        payload.setdefault("turns", [])
        payload.setdefault("route_feedback", [])
        payload.setdefault("route_usage", [])
        return payload

    def save_session(self, session_id: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["session_id"] = session_id
        payload["updated_at"] = _now_ts()
        path = self._path_for(session_id)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)

    def clear_session(self, session_id: str) -> None:
        self._path_for(session_id).unlink(missing_ok=True)
        legacy_path = self._legacy_path_for(session_id)
        legacy_payload = self._read_session_file(legacy_path) if legacy_path.exists() else None
        if legacy_payload is not None and legacy_payload.get("session_id") == session_id:
            legacy_path.unlink(missing_ok=True)

    def _extract_memories(self, user_text: str, assistant_text: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        lower_user = _norm(user_text, limit=320)
        for pattern, kind, builder in MEMORY_PATTERNS:
            match = pattern.search(lower_user)
            if not match:
                continue
            note = _norm(builder(match), limit=220)
            if not note:
                continue
            found.append(
                {
                    "kind": kind,
                    "text": note,
                    "source": "user",
                    "score": 1.0,
                    "updated_at": _now_ts(),
                }
            )
        assistant = _norm(assistant_text, limit=320)
        if assistant:
            found.append(
                {
                    "kind": "lesson",
                    "text": f"Successful answer pattern: {assistant}",
                    "source": "assistant",
                    "score": 0.35,
                    "updated_at": _now_ts(),
                }
            )
        return found

    def update(
        self,
        *,
        session_id: str,
        user_text: str,
        assistant_text: str,
        model_key: str,
        route_reason: str,
        tools: Optional[Sequence[Dict[str, Any]]] = None,
        consultants: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        turn = {
            "ts": _now_ts(),
            "user": _norm(user_text, limit=1200),
            "assistant": _norm(assistant_text, limit=1600),
            "model_key": _norm(model_key, limit=120),
            "route_reason": _norm(route_reason, limit=240),
            "tools": list(tools or []),
            "consultants": list(consultants or []),
        }
        turns = list(payload.get("turns") or [])
        turns.append(turn)
        payload["turns"] = turns[-80:]

        existing = list(payload.get("memories") or [])
        for item in self._extract_memories(user_text, assistant_text):
            note = item["text"]
            match = next((row for row in existing if str(row.get("text") or "").strip().lower() == note.lower()), None)
            if match is None:
                existing.append(item)
                continue
            match["updated_at"] = _now_ts()
            match["score"] = round(float(match.get("score") or 0.0) + 0.2, 3)
        payload["memories"] = existing[-60:]
        self.save_session(session_id, payload)
        return payload

    def prepare_feedback(self, *, session_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize one feedback request without mutating the JSON mirror."""

        payload = self.load_session(session_id)
        route_id = _norm(feedback.get("route_id") or "", limit=120)
        usage_row = next(
            (
                row
                for row in reversed(list(payload.get("route_usage") or []))
                if isinstance(row, dict) and route_id and _norm(row.get("route_id") or "", limit=120) == route_id
            ),
            None,
        )
        authoritative = usage_row if isinstance(usage_row, dict) else {}
        trace = feedback.get("agent_trace") if isinstance(feedback.get("agent_trace"), dict) else {}
        policy = authoritative.get("auto_agent_policy") if authoritative else feedback.get("auto_agent_policy")
        if not isinstance(policy, dict):
            policy = trace.get("auto_agent_policy") if isinstance(trace.get("auto_agent_policy"), dict) else {}

        rating_raw = str(feedback.get("rating") or "").strip().lower()
        rating_pair = FEEDBACK_RATINGS.get(rating_raw)
        if rating_pair is None:
            raise ValueError("rating must be one of: up, down, good, bad, approve, reject")
        rating, score_delta = rating_pair

        intent_raw = str(feedback.get("feedback_intent") or feedback.get("intent") or "").strip().lower()
        intent_key = FEEDBACK_INTENT_ALIASES.get(intent_raw, intent_raw)
        if not intent_key:
            intent_key = "good" if score_delta > 0 else "bad_quality"
        if intent_key not in FEEDBACK_INTENTS:
            raise ValueError(
                "feedback_intent must be one of: good, bad_quality, needs_deeper, too_costly, too_slow"
            )
        feedback_axes = dict(FEEDBACK_INTENTS[intent_key])
        feedback_tags: List[str] = []
        raw_tags = feedback.get("feedback_tags") if isinstance(feedback.get("feedback_tags"), list) else []
        for raw_tag in raw_tags[:4]:
            tag = FEEDBACK_INTENT_ALIASES.get(str(raw_tag).strip().lower(), str(raw_tag).strip().lower())
            if tag in FEEDBACK_INTENTS and tag != intent_key and tag not in feedback_tags:
                feedback_tags.append(tag)
                tag_axes = FEEDBACK_INTENTS[tag]
                for axis_name in ("depth_preference", "cost_pressure", "latency_pressure"):
                    feedback_axes[axis_name] = max(
                        -1,
                        min(1, int(feedback_axes.get(axis_name) or 0) + int(tag_axes.get(axis_name) or 0)),
                    )
        quality_score_delta = feedback_axes.get("quality")
        quality_score_delta = int(quality_score_delta) if quality_score_delta is not None else 0

        selected = _norm(
            authoritative.get("selected_agent_mode")
            or feedback.get("selected_agent_mode")
            or feedback.get("resolved_agent_mode")
            or trace.get("resolved_agent_mode")
            or policy.get("selected_agent_mode")
            or "off",
            limit=48,
        )
        if selected not in ROUTE_MODES:
            selected = "off"

        row = {
            "ts": _now_ts(),
            "route_id": route_id,
            "prompt": _norm(
                authoritative.get("prompt") or feedback.get("prompt") or feedback.get("message") or "",
                limit=1200,
            ),
            "response": _norm(feedback.get("response") or "", limit=1200),
            "selected_agent_mode": selected,
            "rating": rating,
            "score_delta": quality_score_delta,
            "feedback_intent": intent_key,
            "feedback_tags": feedback_tags,
            "feedback_axes": feedback_axes,
            "reason": _norm(feedback.get("reason") or "", limit=240),
            "auto_agent_policy": _compact_auto_agent_policy(policy, selected),
            "model_key": _norm(authoritative.get("model_key") or feedback.get("model_key") or "", limit=120),
            "route_reason": _norm(
                authoritative.get("route_reason") or feedback.get("route_reason") or "",
                limit=240,
            ),
            "evidence_source": "server_route_join" if authoritative else "legacy_unjoined",
            "feedback_revision": 1,
        }
        route_economics = _compact_route_economics(
            authoritative.get("route_economics") if authoritative else feedback.get("route_economics"),
            trace=trace,
            policy=policy,
        )
        if route_economics:
            row["route_economics"] = route_economics
        return row

    def commit_feedback(
        self,
        *,
        session_id: str,
        feedback_row: Dict[str, Any],
        feedback_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upsert a prepared row, optionally pinned to its durable revision.

        When ``feedback_revision`` is supplied, retries replace the same mirror
        revision and stale retries cannot roll the compatibility view backward.
        The durable SQLite ledger remains the source of truth.
        """

        if not isinstance(feedback_row, dict):
            raise ValueError("feedback_row must be a JSON object")
        row = dict(feedback_row)
        route_id = _norm(row.get("route_id") or "", limit=120)
        payload = self.load_session(session_id)
        rows = list(payload.get("route_feedback") or [])
        previous_index = next(
            (
                idx
                for idx in range(len(rows) - 1, -1, -1)
                if isinstance(rows[idx], dict) and route_id and _norm(rows[idx].get("route_id") or "", limit=120) == route_id
            ),
            None,
        )
        previous_revision = (
            int(rows[previous_index].get("feedback_revision") or 1)
            if previous_index is not None
            else 0
        )
        durable_revision = feedback_revision is not None
        if durable_revision:
            if isinstance(feedback_revision, bool):
                raise ValueError("feedback_revision must be a positive integer")
            try:
                revision = int(feedback_revision)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("feedback_revision must be a positive integer") from exc
            if revision < 1:
                raise ValueError("feedback_revision must be a positive integer")
            if previous_index is not None and previous_revision > revision:
                previous = dict(rows[previous_index])
                return {
                    "ok": True,
                    "feedback": previous,
                    "summary": self.route_feedback_summary(session_id, previous.get("prompt") or ""),
                    "compatibility_mirror": {
                        "status": "current_newer_revision",
                        "durable_revision": revision,
                        "mirrored_revision": previous_revision,
                    },
                }
        else:
            revision = previous_revision + 1

        row["feedback_revision"] = revision
        if durable_revision:
            row["durable_feedback_revision"] = revision
        idempotent = previous_index is not None and previous_revision == revision
        if idempotent:
            row["ts"] = rows[previous_index].get("ts", row.get("ts"))
        if previous_index is None:
            rows.append(row)
        else:
            rows[previous_index] = row
        payload["route_feedback"] = rows[-120:]
        self.save_session(session_id, payload)
        return {
            "ok": True,
            "feedback": row,
            "summary": self.route_feedback_summary(session_id, row["prompt"]),
            "compatibility_mirror": {
                "status": "committed",
                "durable_revision": revision if durable_revision else None,
                "mirrored_revision": revision,
                "idempotent": idempotent,
            },
        }

    def add_feedback(self, *, session_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Backward-compatible local feedback write without a durable revision."""

        row = self.prepare_feedback(session_id=session_id, feedback=feedback)
        return self.commit_feedback(session_id=session_id, feedback_row=row)

    def add_route_usage(
        self,
        *,
        session_id: str,
        route_id: str,
        prompt: str,
        selected_agent_mode: str,
        route_economics: Dict[str, Any],
        auto_agent_policy: Optional[Dict[str, Any]] = None,
        route_reason: str = "",
        model_key: str = "",
    ) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        mode = _norm(selected_agent_mode or "off", limit=48)
        if mode not in ROUTE_MODES:
            mode = "off"
        policy = auto_agent_policy if isinstance(auto_agent_policy, dict) else {}
        compact_economics = _compact_route_economics(
            route_economics,
            trace={"route_economics": route_economics},
            policy=policy,
        )
        row = {
            "ts": _now_ts(),
            "route_id": _norm(route_id or "", limit=120),
            "prompt": _norm(prompt or "", limit=1200),
            "selected_agent_mode": mode,
            "route_economics": compact_economics,
            "auto_agent_policy": _compact_auto_agent_policy(policy, mode),
            "model_key": _norm(model_key or "", limit=120),
            "route_reason": _norm(route_reason or "", limit=240),
        }
        rows = list(payload.get("route_usage") or [])
        rows.append(row)
        payload["route_usage"] = rows[-240:]
        self.save_session(session_id, payload)
        return {
            "ok": True,
            "usage": row,
            "summary": self.route_usage_summary(session_id),
        }

    def build_context(self, session_id: str, prompt: str, *, max_memories: int = 5, max_examples: int = 2) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        prompt_tokens = _tokens(prompt)

        ranked_memories: List[tuple[float, Dict[str, Any]]] = []
        for item in list(payload.get("memories") or []):
            text = _norm(item.get("text") or "", limit=220)
            if not text:
                continue
            score = float(item.get("score") or 0.0)
            overlap = len(prompt_tokens & _tokens(text))
            if overlap:
                score += overlap * 0.55
            ranked_memories.append((score, dict(item)))
        ranked_memories.sort(key=lambda pair: pair[0], reverse=True)
        selected_memories = [row for _score, row in ranked_memories[:max_memories]]

        ranked_turns: List[tuple[float, Dict[str, Any]]] = []
        for turn in list(payload.get("turns") or [])[:-1]:
            score = len(prompt_tokens & _tokens(turn.get("user") or ""))
            if score <= 0:
                continue
            ranked_turns.append((float(score), dict(turn)))
        ranked_turns.sort(key=lambda pair: pair[0], reverse=True)
        selected_turns = [row for _score, row in ranked_turns[:max_examples]]

        blocks: List[str] = []
        memory_notes = [_norm(row.get("text") or "", limit=220) for row in selected_memories if _norm(row.get("text") or "", limit=220)]
        if memory_notes:
            blocks.append("Persistent conversation memory:\n" + "\n".join(f"- {note}" for note in memory_notes))
        exemplar_rows: List[str] = []
        for row in selected_turns:
            user = _norm(row.get("user") or "", limit=180)
            assistant = _norm(row.get("assistant") or "", limit=220)
            if not user or not assistant:
                continue
            exemplar_rows.append(f"- User: {user}\n  Assistant: {assistant}")
        if exemplar_rows:
            blocks.append("Relevant prior conversation examples:\n" + "\n".join(exemplar_rows))

        return {
            "memory_notes": memory_notes,
            "example_count": len(exemplar_rows),
            "turn_count": len(payload.get("turns") or []),
            "context_block": "\n\n".join(blocks).strip(),
            "raw": payload,
        }

    def route_feedback_summary(self, session_id: str, prompt: str = "", *, max_items: int = 30) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        rows = [row for row in list(payload.get("route_feedback") or []) if isinstance(row, dict)]
        recent = rows[-max_items:]
        prompt_tokens = _tokens(prompt)
        relevant: List[Dict[str, Any]] = []
        for row in recent:
            row_prompt = _norm(row.get("prompt") or "", limit=1200)
            overlap = len(prompt_tokens & _tokens(row_prompt)) if prompt_tokens else 0
            if not prompt_tokens or overlap >= 2:
                item = dict(row)
                item["prompt_overlap"] = overlap
                relevant.append(item)
        selected_rows = relevant if relevant else recent
        mode_scores: Dict[str, Dict[str, Any]] = {
            mode: {
                "count": 0,
                "positive": 0,
                "negative": 0,
                "net": 0,
                "quality_positive": 0,
                "quality_negative": 0,
                "quality_net": 0,
                "depth_preference_net": 0,
                "cost_pressure": 0,
                "latency_pressure": 0,
            }
            for mode in ROUTE_MODES
        }
        mode_rows: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in ROUTE_MODES}
        for row in selected_rows:
            mode = str(row.get("selected_agent_mode") or "off")
            if mode not in mode_scores:
                mode = "off"
            mode_rows[mode].append(row)
            delta = int(row.get("score_delta") or 0)
            mode_scores[mode]["count"] += 1
            mode_scores[mode]["net"] += delta
            if delta > 0:
                mode_scores[mode]["positive"] += 1
            elif delta < 0:
                mode_scores[mode]["negative"] += 1
            axes = _feedback_axes(row)
            quality_delta = axes["quality"]
            if quality_delta is not None:
                mode_scores[mode]["quality_net"] += int(quality_delta)
                if quality_delta > 0:
                    mode_scores[mode]["quality_positive"] += 1
                elif quality_delta < 0:
                    mode_scores[mode]["quality_negative"] += 1
            mode_scores[mode]["depth_preference_net"] += int(axes["depth_preference"] or 0)
            mode_scores[mode]["cost_pressure"] += int(axes["cost_pressure"] or 0)
            mode_scores[mode]["latency_pressure"] += int(axes["latency_pressure"] or 0)
        for mode, rows_for_mode in mode_rows.items():
            economics = _route_economics_summary(rows_for_mode)
            mode_scores[mode]["economics"] = economics
            mode_scores[mode]["adaptive"] = _route_adaptive_feedback_summary(rows_for_mode, economics)
        economics_summary = _route_economics_summary(selected_rows)
        return {
            "total_feedback": len(rows),
            "recent_feedback": len(recent),
            "relevant_feedback": len(relevant),
            "used_recent_fallback": bool(prompt_tokens and not relevant and recent),
            "mode_scores": mode_scores,
            "economics": economics_summary,
            "adaptive": _route_adaptive_feedback_summary(selected_rows, economics_summary),
            "recent": [
                {
                    "selected_agent_mode": _norm(row.get("selected_agent_mode") or "off", limit=48),
                    "rating": _norm(row.get("rating") or "", limit=16),
                    "score_delta": int(row.get("score_delta") or 0),
                    "feedback_intent": _norm(row.get("feedback_intent") or "bad_quality", limit=32),
                    "feedback_tags": [
                        _norm(item, limit=32) for item in list(row.get("feedback_tags") or [])[:4]
                    ],
                    "feedback_axes": _feedback_axes(row),
                    "reason": _norm(row.get("reason") or "", limit=120),
                    "route_economics": (
                        row.get("route_economics")
                        if isinstance(row.get("route_economics"), dict)
                        else {}
                    ),
                }
                for row in rows[-5:]
            ],
        }

    def route_usage_summary(self, session_id: str, *, max_items: int = 240) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        rows = [row for row in list(payload.get("route_usage") or []) if isinstance(row, dict)]
        recent = rows[-max_items:]
        mode_rows: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in ROUTE_MODES}
        for row in recent:
            mode = str(row.get("selected_agent_mode") or "off")
            if mode not in mode_rows:
                mode = "off"
            mode_rows[mode].append(row)
        return {
            "total_routes": len(rows),
            "recent_routes": len(recent),
            "mode_economics": {
                mode: _route_economics_summary(rows_for_mode)
                for mode, rows_for_mode in mode_rows.items()
            },
            "economics": _route_economics_summary(recent),
            "recent": [
                {
                    "selected_agent_mode": _norm(row.get("selected_agent_mode") or "off", limit=48),
                    "route_economics": (
                        row.get("route_economics")
                        if isinstance(row.get("route_economics"), dict)
                        else {}
                    ),
                }
                for row in rows[-5:]
            ],
        }

    def session_snapshot(self, session_id: str) -> Dict[str, Any]:
        payload = self.load_session(session_id)
        memories = [
            _norm(item.get("text") or "", limit=160)
            for item in list(payload.get("memories") or [])[-8:]
            if _norm(item.get("text") or "", limit=160)
        ]
        turns = list(payload.get("turns") or [])
        recent_turns = [
            {
                "user": _norm(turn.get("user") or "", limit=140),
                "assistant": _norm(turn.get("assistant") or "", limit=180),
                "model_key": _norm(turn.get("model_key") or "", limit=80),
            }
            for turn in turns[-4:]
        ]
        return {
            "session_id": session_id,
            "memory_count": len(payload.get("memories") or []),
            "turn_count": len(turns),
            "route_feedback_count": len(payload.get("route_feedback") or []),
            "route_feedback": self.route_feedback_summary(session_id),
            "route_usage": self.route_usage_summary(session_id),
            "memories": memories,
            "recent_turns": recent_turns,
            "updated_at": payload.get("updated_at"),
        }

    def global_status(self) -> Dict[str, Any]:
        files = list(self.root_dir.glob("*.json"))
        return {
            "session_files": len(files),
            "root_dir": str(self.root_dir),
        }
