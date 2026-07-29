from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit


GROUNDING_SCHEMA_VERSION = "supermix-grounding-v1"
GROUNDING_RUNTIME_VERSION = "supermix-grounding-runtime-v1"

MAX_QUERY_CHARS = 4000
MAX_EXTERNAL_QUERY_CHARS = 400
MAX_EVIDENCE_ITEMS = 12
MAX_EVIDENCE_TEXT_CHARS = 2400
MAX_ARITHMETIC_EXPRESSION_CHARS = 160
MAX_ARITHMETIC_AST_NODES = 64
MAX_ARITHMETIC_DEPTH = 12
MAX_ARITHMETIC_OPERATIONS = 32
MAX_ARITHMETIC_EXPONENT = 12
MAX_ARITHMETIC_RESULT_BITS = 4096


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'’-]*", re.IGNORECASE)
_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
_CITATION_RE = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
_VALID_CITATION_ID_RE = re.compile(r"^S[1-9]\d*$")
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b",
    re.IGNORECASE,
)
_FRESHNESS_RE = re.compile(
    r"\b(?:latest|newest|current|currently|today|recent|recently|up[- ]to[- ]date|"
    r"this (?:week|month|year)|as of|news|price|version|release|schedule|score)\b",
    re.IGNORECASE,
)
_CITATION_REQUEST_RE = re.compile(
    r"\b(?:cite|citation|citations|source|sources|reference|references|link|links|evidence)\b",
    re.IGNORECASE,
)
_HIGH_STAKES_RE = re.compile(
    r"\b(?:medical|medicine|diagnosis|dose|dosage|symptom|chest pain|stroke|overdose|"
    r"legal|lawyer|lawsuit|contract|tax|financial advice|investment|mortgage|"
    r"security vulnerability|malware|credential|password|suicid|self[- ]?harm)\b",
    re.IGNORECASE,
)
_FACTUAL_REQUEST_RE = re.compile(
    r"(?:^\s*(?:who|what|when|where|which|how many|how much|is|are|does|did|can)\b|"
    r"\b(?:fact|facts|explain|compare|difference|definition|documentation|research|paper)\b)",
    re.IGNORECASE,
)
_STRICT_EVIDENCE_ONLY_RE = re.compile(
    r"(?:"
    r"\b(?:use|using|based on|answer from|rely on)\s+only\s+(?:the\s+)?"
    r"(?:(?:supplied|provided|attached|following|given)\s+)?"
    r"(?:evidence|sources?|context|passage|documents?|text)\b"
    r"|"
    r"\banswer\s+only\s+from\s+(?:the\s+)?(?:supplied|provided|attached|following|given|these)?\s*"
    r"(?:evidence|sources?|context|passage|documents?|text)\b"
    r")",
    re.IGNORECASE,
)
_ARITHMETIC_PREFIX_RE = re.compile(
    r"^\s*(?:what\s+is|calculate|compute|evaluate|work\s+out|solve(?:\s+the\s+expression)?)"
    r"\s*[:=]?\s*(?P<expression>.+?)\s*$",
    re.IGNORECASE,
)
_ARITHMETIC_ALLOWED_RE = re.compile(r"^[0-9eE\s+\-*/().%^]+$")
_ARITHMETIC_BINARY_RE = re.compile(r"(?:\*\*|[+\-*/%^])")
_AMBIGUOUS_DATE_RE = re.compile(r"^\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}$")
_AMBIGUOUS_PHONE_RE = re.compile(
    r"^(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}$"
)
_AMBIGUOUS_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")

_STOPWORDS = {
    "a",
    "about",
    "after",
    "all",
    "also",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "based",
    "be",
    "because",
    "before",
    "by",
    "can",
    "cite",
    "context",
    "could",
    "did",
    "do",
    "does",
    "document",
    "documents",
    "evidence",
    "explain",
    "for",
    "from",
    "given",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "link",
    "me",
    "of",
    "on",
    "only",
    "or",
    "passage",
    "please",
    "provided",
    "reference",
    "source",
    "sources",
    "supplied",
    "text",
    "that",
    "the",
    "these",
    "this",
    "to",
    "use",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}
_NEGATION_TOKENS = {
    "not",
    "no",
    "never",
    "without",
    "cannot",
    "can't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "doesn't",
    "don't",
    "didn't",
}

_TRUST_WEIGHTS = {
    "official": 1000,
    "primary": 850,
    "secondary": 650,
    "community": 400,
    "unknown": 200,
}
_TRUST_ALIASES = {
    "first_party": "official",
    "first-party": "official",
    "government": "official",
    "standards_body": "official",
    "standards-body": "official",
    "research": "primary",
    "paper": "primary",
    "peer_reviewed": "primary",
    "peer-reviewed": "primary",
    "news": "secondary",
    "reference": "secondary",
    "blog": "community",
    "forum": "community",
    "social": "community",
}

_SHOW_WORK_RE = re.compile(
    r"\b(?:show (?:your |the )?(?:work|working|steps)|step[- ]by[- ]step|explain (?:your |the )?"
    r"(?:reasoning|working|steps|how)|walk me through|how did you get)\b",
    re.IGNORECASE,
)

_PROMPT_UNDERSTANDING_MODULE: Any = None
_REASONING_MODULE: Any = None


def _load_prompt_understanding_module() -> Any:
    """Load the mirrored sibling module even under file-based test imports."""

    global _PROMPT_UNDERSTANDING_MODULE
    if _PROMPT_UNDERSTANDING_MODULE is not None:
        return _PROMPT_UNDERSTANDING_MODULE
    try:
        import prompt_understanding as module
    except ImportError:
        module_path = Path(__file__).with_name("prompt_understanding.py")
        module_name = f"_supermix_{module_path.parent.name}_prompt_understanding"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load prompt understanding API from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        spec.loader.exec_module(module)
    _PROMPT_UNDERSTANDING_MODULE = module
    return module


def _load_reasoning_module() -> Any:
    """Load the mirrored deliberate-reasoning sibling module, or None."""

    global _REASONING_MODULE
    if _REASONING_MODULE is not None:
        return _REASONING_MODULE
    try:
        import reasoning_engine as module
    except ImportError:
        module_path = Path(__file__).with_name("reasoning_engine.py")
        module_name = f"_supermix_{module_path.parent.name}_reasoning_engine"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault(module_name, module)
        try:
            spec.loader.exec_module(module)
        except Exception:  # pragma: no cover - defensive: reasoning stays optional
            return None
    _REASONING_MODULE = module
    return module


def solve_reasoned_problem(query: Any, tier: str = "auto") -> Dict[str, Any]:
    """Run the deliberate reasoning engine, degrading to an empty attempt."""

    module = _load_reasoning_module()
    if module is None:
        return {"attempted": False, "solved": False, "override_allowed": False, "reason": "engine_unavailable"}
    try:
        result = module.solve_problem(_clean_text(query, MAX_QUERY_CHARS), tier=tier)
    except Exception:  # pragma: no cover - defensive: never break a chat turn
        return {"attempted": False, "solved": False, "override_allowed": False, "reason": "engine_error"}
    return dict(result) if isinstance(result, Mapping) else {
        "attempted": False,
        "solved": False,
        "override_allowed": False,
        "reason": "engine_bad_result",
    }


def _reasoning_frame(query: Any) -> Dict[str, Any]:
    module = _load_reasoning_module()
    if module is None:
        return {"available": False, "complexity": 0.0, "recommended_tier": "", "numbers_found": 0}
    try:
        frame = module.frame_problem(_clean_text(query, MAX_QUERY_CHARS))
    except Exception:  # pragma: no cover - defensive
        return {"available": False, "complexity": 0.0, "recommended_tier": "", "numbers_found": 0}
    return {
        "available": True,
        "complexity": _json_safe_float(frame.get("complexity")),
        "recommended_tier": str(frame.get("recommended_tier") or ""),
        "numbers_found": _bounded_int(frame.get("numbers_found"), 0, 0, 999),
    }


def _resolve_prompt_profile(
    query: Any,
    prompt_profile: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if isinstance(prompt_profile, Mapping):
        return dict(prompt_profile)
    module = _load_prompt_understanding_module()
    profile = module.analyze_prompt(str(query or ""))
    return dict(profile) if isinstance(profile, Mapping) else {}


def _profile_bool(
    profile: Mapping[str, Any],
    section_name: str,
    *keys: str,
) -> bool:
    section = profile.get(section_name)
    if not isinstance(section, Mapping):
        return False
    return any(bool(section.get(key)) for key in keys)


def _clean_text(value: Any, limit: int) -> str:
    cooked = re.sub(r"\s+", " ", str(value or "")).strip()
    return cooked[: max(0, int(limit))]


def _json_safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return cooked if math.isfinite(cooked) else float(default)


def _content_terms(value: Any) -> set[str]:
    return {
        token.lower().replace("’", "'")
        for token in _TOKEN_RE.findall(str(value or ""))
        if len(token) > 1 and token.lower().replace("’", "'") not in _STOPWORDS
    }


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(Fraction(max(0, numerator), denominator)), 6)


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        cooked = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, cooked))


def _safe_url(value: Any) -> str:
    raw = _clean_text(value, 600)
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return ""
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password:
        return ""
    host = (parts.hostname or "").lower().strip(".")
    if not host:
        return ""
    try:
        parsed_port = parts.port
    except ValueError:
        return ""
    port = f":{parsed_port}" if parsed_port is not None else ""
    netloc = host + port
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _url_domain(url: str) -> str:
    if not url:
        return ""
    try:
        return (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def redact_external_query(query: Any, max_chars: int = MAX_EXTERNAL_QUERY_CHARS) -> Dict[str, Any]:
    """Return a bounded query safe to hand to an external search provider."""

    cooked = _clean_text(query, MAX_QUERY_CHARS)
    categories: List[str] = []
    count = 0

    patterns: Sequence[Tuple[str, re.Pattern[str], str]] = (
        (
            "bearer_token",
            re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
            "[REDACTED_TOKEN]",
        ),
        (
            "secret",
            re.compile(
                r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|secret|password|passwd)"
                r"\s*[:=]\s*[\"']?[^\s\"',;]{4,}",
                re.IGNORECASE,
            ),
            "[REDACTED_SECRET]",
        ),
        (
            "credential",
            re.compile(
                r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{12,}|AKIA[A-Z0-9]{16})\b"
            ),
            "[REDACTED_CREDENTIAL]",
        ),
        (
            "email",
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
            "[REDACTED_EMAIL]",
        ),
        (
            "ssn",
            re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
            "[REDACTED_SSN]",
        ),
        (
            "phone",
            re.compile(r"(?<!\d)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\d)"),
            "[REDACTED_PHONE]",
        ),
        (
            "ip_address",
            re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
            "[REDACTED_IP]",
        ),
        (
            "credit_card",
            re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
            "[REDACTED_CARD]",
        ),
        (
            "windows_path",
            re.compile(r"(?<!\w)(?:[A-Za-z]:\\|\\\\)[^\s\"'<>|]+", re.IGNORECASE),
            "[REDACTED_PATH]",
        ),
        (
            "home_path",
            re.compile(r"(?<!\w)(?:/home/[^/\s]+|/Users/[^/\s]+|~/)[^\s\"'<>|]*"),
            "[REDACTED_PATH]",
        ),
    )

    for category, pattern, replacement in patterns:
        cooked, matches = pattern.subn(replacement, cooked)
        if matches:
            count += int(matches)
            if category not in categories:
                categories.append(category)

    cooked = re.sub(r"\s+", " ", cooked).strip()
    limit = _bounded_int(max_chars, MAX_EXTERNAL_QUERY_CHARS, 32, MAX_EXTERNAL_QUERY_CHARS)
    truncated = len(cooked) > limit
    if truncated:
        cooked = cooked[:limit].rstrip()
    meaningful = _content_terms(re.sub(r"\[REDACTED_[A-Z_]+\]", " ", cooked))
    return {
        "query": cooked,
        "redaction_count": count,
        "categories": categories,
        "truncated": bool(truncated),
        "safe_to_send": bool(cooked and meaningful),
    }


def _interaction_epistemic_risk(interaction_plan: Optional[Mapping[str, Any]]) -> float:
    if not isinstance(interaction_plan, Mapping):
        return 0.0
    risk = interaction_plan.get("risk")
    if isinstance(risk, Mapping):
        value = _json_safe_float(risk.get("epistemic_score"), -1.0)
        if value >= 0.0:
            return max(0.0, min(1.0, value))
    deliberation = interaction_plan.get("deliberation")
    if isinstance(deliberation, Mapping):
        value = _json_safe_float(deliberation.get("epistemic_risk"), -1.0)
        if value >= 0.0:
            return max(0.0, min(1.0, value))
    # Retain compatibility with early v1 plans that placed this field here.
    appraisal = interaction_plan.get("appraisal")
    if isinstance(appraisal, Mapping):
        return max(0.0, min(1.0, _json_safe_float(appraisal.get("epistemic_risk"), 0.0)))
    return 0.0


def plan_grounding(
    query: Any,
    interaction_plan: Optional[Mapping[str, Any]] = None,
    prompt_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a prompt-safe grounding plan that has no routing or compute authority."""

    text = _clean_text(query, MAX_QUERY_CHARS)
    profile = _resolve_prompt_profile(text, prompt_profile)
    arithmetic = solve_exact_arithmetic(text)
    strict_only = bool(_STRICT_EVIDENCE_ONLY_RE.search(text)) or _profile_bool(
        profile,
        "knowledge",
        "strict_evidence_only",
        "evidence_only",
    )
    freshness = bool(_FRESHNESS_RE.search(text)) or _profile_bool(
        profile,
        "knowledge",
        "freshness_required",
        "freshness_sensitive",
    )
    citation_requested = bool(_CITATION_REQUEST_RE.search(text)) or _profile_bool(
        profile,
        "knowledge",
        "citation_requested",
        "citations_requested",
    )
    high_stakes = (
        bool(_HIGH_STAKES_RE.search(text))
        or _profile_bool(profile, "knowledge", "high_stakes")
        or _profile_bool(profile, "safety", "high_stakes")
    )
    evidence_requested = _profile_bool(
        profile,
        "knowledge",
        "evidence_requested",
    )
    factual = bool(_FACTUAL_REQUEST_RE.search(text)) or _profile_bool(
        profile,
        "knowledge",
        "factual",
        "factual_request",
        "evidence_recommended",
    )
    epistemic_risk = _interaction_epistemic_risk(interaction_plan)
    external = redact_external_query(text)

    reasons: List[str] = []
    if strict_only:
        reasons.append("strict_evidence_only")
    if bool(arithmetic.get("attempted")):
        reasons.append("explicit_arithmetic")
    if freshness:
        reasons.append("freshness_required")
    if citation_requested:
        reasons.append("citations_requested")
    if evidence_requested:
        reasons.append("evidence_requested")
    if high_stakes:
        reasons.append("high_stakes_factuality")
    if factual:
        reasons.append("factual_request")
    if epistemic_risk >= 0.65:
        reasons.append("interaction_epistemic_risk")

    evidence_recommended = bool(
        strict_only
        or freshness
        or citation_requested
        or evidence_requested
        or high_stakes
        or factual
        or epistemic_risk >= 0.65
    ) and not bool(arithmetic.get("solved"))
    return {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "runtime_version": GROUNDING_RUNTIME_VERSION,
        "scope": "grounding_only",
        "advisory_only": True,
        "evidence_recommended": evidence_recommended,
        "reasons": reasons,
        "freshness_required": freshness,
        "evidence_requested": evidence_requested,
        "citation_requested": citation_requested,
        "high_stakes": high_stakes,
        "strict_evidence_only": strict_only,
        "exact_arithmetic": {
            "attempted": bool(arithmetic.get("attempted")),
            "solved": bool(arithmetic.get("solved")),
            "reason": str(arithmetic.get("reason") or ""),
        },
        "reasoning_frame": _reasoning_frame(text),
        "epistemic_risk": round(epistemic_risk, 6),
        "max_evidence_items": 6 if evidence_recommended else 0,
        "external_query": external,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "compute_exit_authority": "unchanged",
        },
    }


def _normalized_trust_tier(row: Mapping[str, Any]) -> str:
    raw = _clean_text(row.get("trust_tier") or row.get("trust") or "", 40).lower().replace(" ", "_")
    raw = _TRUST_ALIASES.get(raw, raw)
    if raw in _TRUST_WEIGHTS:
        return raw
    source_type = _clean_text(row.get("source_type") or row.get("kind") or "", 40).lower().replace(" ", "_")
    inferred = _TRUST_ALIASES.get(source_type, source_type)
    return inferred if inferred in _TRUST_WEIGHTS else "unknown"


def _provided_score_basis_points(row: Mapping[str, Any]) -> int:
    raw = row.get("score", row.get("relevance_score", row.get("rank_score", 0.0)))
    score = max(0.0, min(1.0, _json_safe_float(raw, 0.0)))
    return int(round(score * 1000.0))


def _normalize_evidence_candidate(row: Mapping[str, Any], query_terms: set[str]) -> Optional[Dict[str, Any]]:
    title = _clean_text(row.get("title") or row.get("name") or "", 300)
    text = _clean_text(
        row.get("text")
        or row.get("snippet")
        or row.get("content")
        or row.get("answer")
        or row.get("excerpt")
        or "",
        MAX_EVIDENCE_TEXT_CHARS,
    )
    url = _safe_url(row.get("url") or row.get("href") or row.get("link") or "")
    if not text:
        return None
    source_type = _clean_text(row.get("source_type") or row.get("kind") or "unknown", 40).lower()
    source = _clean_text(row.get("source") or row.get("provider") or _url_domain(url) or "unknown", 120)
    domain = _url_domain(url) or _clean_text(row.get("domain") or "", 160).lower()
    published_at = _clean_text(
        row.get("published_at") or row.get("date") or row.get("updated_at") or "",
        80,
    )
    license_name = _clean_text(row.get("license") or row.get("licence") or "", 120)
    trust_tier = _normalized_trust_tier(row)
    item_terms = _content_terms(f"{title} {text}")
    overlap = len(query_terms & item_terms)
    lexical_bp = int(Fraction(overlap, max(1, len(query_terms))) * 1000) if query_terms else 0
    provided_bp = _provided_score_basis_points(row)
    trust_bp = _TRUST_WEIGHTS[trust_tier]
    completeness_bp = min(1000, int(Fraction(min(len(text), 240), 240) * 1000))
    rank_bp = (
        lexical_bp * 55
        + provided_bp * 25
        + trust_bp * 15
        + completeness_bp * 5
    ) // 100
    canonical = json.dumps(
        {
            "title": title,
            "url": url,
            "text": text,
            "source": source,
            "published_at": published_at,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "title": title,
        "url": url,
        "text": text,
        "source": source,
        "source_type": source_type,
        "domain": domain,
        "published_at": published_at,
        "trust_tier": trust_tier,
        "license": license_name,
        "content_hash": content_hash,
        "input_score": round(float(Fraction(provided_bp, 1000)), 6),
        "lexical_overlap": round(float(Fraction(lexical_bp, 1000)), 6),
        "rank_score": round(float(Fraction(rank_bp, 1000)), 6),
        "_rank_bp": rank_bp,
        "_trust_bp": trust_bp,
    }


def normalize_evidence_rows(
    rows: Optional[Iterable[Mapping[str, Any]]],
    query: Any = "",
    max_items: int = 6,
) -> List[Dict[str, Any]]:
    """Normalize, deduplicate, and deterministically rank evidence with stable S1 IDs."""

    query_terms = _content_terms(_clean_text(query, MAX_QUERY_CHARS))
    candidates: Dict[str, Dict[str, Any]] = {}
    for raw in rows or ():
        if not isinstance(raw, Mapping):
            continue
        item = _normalize_evidence_candidate(raw, query_terms)
        if item is None:
            continue
        key = item["content_hash"]
        previous = candidates.get(key)
        if previous is None:
            candidates[key] = item
            continue
        previous_key = (
            int(previous["_rank_bp"]),
            int(previous["_trust_bp"]),
            json.dumps(previous, sort_keys=True, ensure_ascii=True),
        )
        item_key = (
            int(item["_rank_bp"]),
            int(item["_trust_bp"]),
            json.dumps(item, sort_keys=True, ensure_ascii=True),
        )
        if item_key > previous_key:
            candidates[key] = item

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            -int(item["_rank_bp"]),
            -int(item["_trust_bp"]),
            str(item["content_hash"]),
        ),
    )
    limit = _bounded_int(max_items, 6, 0, MAX_EVIDENCE_ITEMS)
    result: List[Dict[str, Any]] = []
    for index, item in enumerate(ordered[:limit], start=1):
        cooked = {key: value for key, value in item.items() if not key.startswith("_")}
        cooked["id"] = f"S{index}"
        cooked["rank"] = index
        result.append(cooked)
    return result


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(Fraction(len(left & right), len(union))) if union else 0.0


def _evidence_conflicts(evidence: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    sentence_rows: List[Tuple[str, str, set[str], set[str], bool]] = []
    for item in evidence:
        source_id = str(item.get("id") or "")
        text = str(item.get("text") or "")
        for match in _SENTENCE_RE.finditer(text):
            sentence = match.group(0).strip()
            terms = _content_terms(sentence)
            if not terms:
                continue
            numbers = {value.lower() for value in _NUMBER_RE.findall(sentence)}
            normalized_terms = {
                term
                for term in (terms - _NEGATION_TOKENS)
                if not term[:1].isdigit() and not _NUMBER_RE.fullmatch(term)
            }
            sentence_rows.append(
                (
                    source_id,
                    sentence,
                    normalized_terms,
                    numbers,
                    bool(_NEGATION_RE.search(sentence)),
                )
            )

    conflicts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for left_index, left in enumerate(sentence_rows):
        for right in sentence_rows[left_index + 1 :]:
            if not left[0] or left[0] == right[0]:
                continue
            shared = left[2] & right[2]
            similarity = _jaccard(left[2], right[2])
            kind = ""
            if len(shared) >= 2 and similarity >= 0.55 and left[3] and right[3] and left[3] != right[3]:
                kind = "numeric"
            elif len(shared) >= 3 and similarity >= 0.75 and left[4] != right[4]:
                kind = "polarity"
            if not kind:
                continue
            ids = tuple(sorted((left[0], right[0])))
            key = (ids[0], ids[1], kind)
            conflicts[key] = {
                "source_ids": [ids[0], ids[1]],
                "kind": kind,
                "shared_terms": sorted(shared)[:6],
            }
    return [conflicts[key] for key in sorted(conflicts)]


def validate_citations(
    response_text: Any,
    evidence: Any,
) -> Dict[str, Any]:
    if isinstance(evidence, Mapping):
        evidence_rows = evidence.get("evidence")
    else:
        evidence_rows = evidence
    valid_ids = {
        str(item.get("id") or "").upper()
        for item in (evidence_rows or ())
        if isinstance(item, Mapping) and _VALID_CITATION_ID_RE.fullmatch(str(item.get("id") or "").upper())
    }
    seen: set[str] = set()
    citations: List[str] = []
    for match in _CITATION_RE.finditer(str(response_text or "")):
        citation = match.group(1).upper()
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    valid = [citation for citation in citations if citation in valid_ids and _VALID_CITATION_ID_RE.fullmatch(citation)]
    invalid = [citation for citation in citations if citation not in valid_ids or not _VALID_CITATION_ID_RE.fullmatch(citation)]
    return {
        "citations": citations,
        "valid": valid,
        "invalid": invalid,
        "citation_count": len(citations),
        "has_citations": bool(citations),
        "all_valid": not invalid,
        "uncited_evidence_ids": sorted(valid_ids - set(valid), key=lambda value: int(value[1:])),
    }


def evidence_diagnostics(
    query: Any,
    evidence: Any,
    response_text: Any = "",
) -> Dict[str, Any]:
    if isinstance(evidence, Mapping):
        raw_rows = list(evidence.get("evidence") or ())
    else:
        raw_rows = list(evidence or ())
    # Re-normalize even apparently canonical rows. Public callers may supply
    # forged IDs, oversized fields, or unsafe provenance URLs.
    rows = normalize_evidence_rows(raw_rows, query=query, max_items=MAX_EVIDENCE_ITEMS)
    query_terms = _content_terms(query)
    union_terms: set[str] = set()
    item_coverages: List[float] = []
    for item in rows:
        terms = _content_terms(f"{item.get('title', '')} {item.get('text', '')}")
        union_terms |= terms
        item_coverages.append(_ratio(len(query_terms & terms), len(query_terms)))
    query_coverage = _ratio(len(query_terms & union_terms), len(query_terms))
    best_item_coverage = max(item_coverages, default=0.0)
    conflicts = _evidence_conflicts(rows)

    if not rows:
        sufficiency = "no_evidence"
    elif conflicts:
        sufficiency = "conflicting"
    elif not query_terms:
        sufficiency = "partial"
    elif query_coverage >= 0.70 or best_item_coverage >= 0.55:
        sufficiency = "sufficient"
    elif query_coverage >= 0.25 or best_item_coverage >= 0.20:
        sufficiency = "partial"
    else:
        sufficiency = "insufficient"

    response_terms = _content_terms(response_text)
    response_coverage = _ratio(len(response_terms & union_terms), len(response_terms))
    citation_audit = validate_citations(response_text, rows)
    return {
        "evidence_count": len(rows),
        "query_term_count": len(query_terms),
        "query_coverage": query_coverage,
        "best_item_coverage": round(best_item_coverage, 6),
        "response_coverage": response_coverage,
        "sufficiency": sufficiency,
        "sufficient": sufficiency == "sufficient",
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "citation_audit": citation_audit,
    }


def build_evidence_bundle(
    query: Any,
    rows: Optional[Iterable[Mapping[str, Any]]],
    interaction_plan: Optional[Mapping[str, Any]] = None,
    max_items: int = 6,
    prompt_profile: Optional[Mapping[str, Any]] = None,
    grounding_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    plan = (
        dict(grounding_plan)
        if isinstance(grounding_plan, Mapping)
        else plan_grounding(
            query,
            interaction_plan=interaction_plan,
            prompt_profile=prompt_profile,
        )
    )
    evidence = normalize_evidence_rows(rows, query=query, max_items=max_items)
    diagnostics = evidence_diagnostics(query, evidence)
    return {
        "schema_version": GROUNDING_SCHEMA_VERSION,
        "runtime_version": GROUNDING_RUNTIME_VERSION,
        "plan": plan,
        "evidence": evidence,
        "diagnostics": diagnostics,
    }


class _ArithmeticError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _extract_arithmetic_expression(query: Any) -> Optional[str]:
    text = _clean_text(query, MAX_QUERY_CHARS)
    if not text or len(text) > MAX_ARITHMETIC_EXPRESSION_CHARS + 64:
        return None
    match = _ARITHMETIC_PREFIX_RE.fullmatch(text)
    expression = match.group("expression") if match else text
    expression = expression.strip()
    if expression.endswith("?") or expression.endswith("!"):
        expression = expression[:-1].rstrip()
    if expression.endswith(".") and expression.count(".") == 1:
        expression = expression[:-1].rstrip()
    expression = (
        expression.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("^", "**")
    )
    if match is None and (
        _AMBIGUOUS_DATE_RE.fullmatch(expression)
        or _AMBIGUOUS_PHONE_RE.fullmatch(expression)
        or _AMBIGUOUS_SSN_RE.fullmatch(expression)
    ):
        return None
    if (
        not expression
        or len(expression) > MAX_ARITHMETIC_EXPRESSION_CHARS
        or not _ARITHMETIC_ALLOWED_RE.fullmatch(expression)
        or not _ARITHMETIC_BINARY_RE.search(expression)
    ):
        return None
    return expression


def _fraction_from_constant(node: ast.Constant, expression: str) -> Fraction:
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise _ArithmeticError("unsupported_literal")
    segment = ast.get_source_segment(expression, node) or repr(node.value)
    segment = segment.strip()
    if isinstance(node.value, int):
        if len(segment.lstrip("+-")) > 80:
            raise _ArithmeticError("literal_too_large")
        value = Fraction(int(node.value), 1)
    else:
        try:
            decimal = Decimal(segment)
        except (InvalidOperation, ValueError):
            raise _ArithmeticError("unsupported_literal") from None
        if not decimal.is_finite():
            raise _ArithmeticError("non_finite_literal")
        value = Fraction(decimal)
    return _check_fraction_bounds(value)


def _check_fraction_bounds(value: Fraction) -> Fraction:
    if (
        abs(value.numerator).bit_length() > MAX_ARITHMETIC_RESULT_BITS
        or value.denominator.bit_length() > MAX_ARITHMETIC_RESULT_BITS
    ):
        raise _ArithmeticError("result_too_large")
    return value


def _solve_ast(
    node: ast.AST,
    expression: str,
    *,
    depth: int,
    operation_counter: List[int],
) -> Fraction:
    if depth > MAX_ARITHMETIC_DEPTH:
        raise _ArithmeticError("expression_too_deep")
    if isinstance(node, ast.Expression):
        return _solve_ast(node.body, expression, depth=depth + 1, operation_counter=operation_counter)
    if isinstance(node, ast.Constant):
        return _fraction_from_constant(node, expression)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operation_counter[0] += 1
        value = _solve_ast(node.operand, expression, depth=depth + 1, operation_counter=operation_counter)
        return value if isinstance(node.op, ast.UAdd) else _check_fraction_bounds(-value)
    if not isinstance(node, ast.BinOp):
        raise _ArithmeticError("unsupported_syntax")
    operation_counter[0] += 1
    if operation_counter[0] > MAX_ARITHMETIC_OPERATIONS:
        raise _ArithmeticError("too_many_operations")
    left = _solve_ast(node.left, expression, depth=depth + 1, operation_counter=operation_counter)
    right = _solve_ast(node.right, expression, depth=depth + 1, operation_counter=operation_counter)
    if isinstance(node.op, ast.Add):
        result = left + right
    elif isinstance(node.op, ast.Sub):
        result = left - right
    elif isinstance(node.op, ast.Mult):
        result = left * right
    elif isinstance(node.op, ast.Div):
        if right == 0:
            raise _ArithmeticError("division_by_zero")
        result = left / right
    elif isinstance(node.op, ast.FloorDiv):
        if right == 0:
            raise _ArithmeticError("division_by_zero")
        result = Fraction(left // right, 1)
    elif isinstance(node.op, ast.Mod):
        if right == 0:
            raise _ArithmeticError("division_by_zero")
        result = left % right
    elif isinstance(node.op, ast.Pow):
        if right.denominator != 1:
            raise _ArithmeticError("fractional_exponent_not_supported")
        exponent = int(right.numerator)
        if abs(exponent) > MAX_ARITHMETIC_EXPONENT:
            raise _ArithmeticError("exponent_too_large")
        if exponent < 0 and left == 0:
            raise _ArithmeticError("division_by_zero")
        result = left**exponent
    else:
        raise _ArithmeticError("unsupported_operator")
    return _check_fraction_bounds(result)


def _terminating_decimal(value: Fraction) -> Optional[str]:
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return None
    with localcontext() as context:
        context.prec = min(220, max(32, len(str(abs(value.numerator))) + len(str(value.denominator)) + 8))
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    cooked = format(decimal, "f")
    if "." in cooked:
        cooked = cooked.rstrip("0").rstrip(".")
    return cooked or "0"


def _arithmetic_presentation(value: Fraction) -> Dict[str, Any]:
    exact_fraction = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    terminating = _terminating_decimal(value)
    if terminating is not None:
        display = terminating
        approximation = ""
    else:
        with localcontext() as context:
            context.prec = 16
            decimal = Decimal(value.numerator) / Decimal(value.denominator)
        approximation = format(decimal, ".12f").rstrip("0").rstrip(".")
        display = exact_fraction
    return {
        "exact": exact_fraction,
        "display": display,
        "approximation": approximation,
    }


def solve_exact_arithmetic(query: Any) -> Dict[str, Any]:
    """Solve a bounded explicit arithmetic expression without eval or side effects."""

    expression = _extract_arithmetic_expression(query)
    if expression is None:
        return {
            "attempted": False,
            "solved": False,
            "reason": "not_explicit_arithmetic",
            "expression": "",
            "exact": "",
            "display": "",
            "approximation": "",
            "operations": 0,
        }
    result: Dict[str, Any] = {
        "attempted": True,
        "solved": False,
        "reason": "",
        "expression": expression,
        "exact": "",
        "display": "",
        "approximation": "",
        "operations": 0,
    }
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > MAX_ARITHMETIC_AST_NODES:
            raise _ArithmeticError("too_many_nodes")
        operations = [0]
        value = _solve_ast(tree, expression, depth=0, operation_counter=operations)
        presentation = _arithmetic_presentation(value)
        result.update(
            {
                "solved": True,
                "reason": "solved_exactly",
                "operations": operations[0],
                **presentation,
            }
        )
    except SyntaxError:
        result["reason"] = "invalid_syntax"
    except _ArithmeticError as exc:
        result["reason"] = exc.reason
    except (ArithmeticError, InvalidOperation, OverflowError, ValueError):
        result["reason"] = "arithmetic_error"
    return result


def _coerce_evidence_bundle(
    query: Any,
    evidence_bundle: Any,
    interaction_plan: Optional[Mapping[str, Any]],
    prompt_profile: Optional[Mapping[str, Any]],
    grounding_plan: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if isinstance(evidence_bundle, Mapping):
        rows = evidence_bundle.get("evidence") or ()
    elif isinstance(evidence_bundle, Sequence) and not isinstance(evidence_bundle, (str, bytes, bytearray)):
        rows = evidence_bundle
    else:
        rows = ()
    return build_evidence_bundle(
        query,
        rows,
        interaction_plan=interaction_plan,
        max_items=MAX_EVIDENCE_ITEMS,
        prompt_profile=prompt_profile,
        grounding_plan=grounding_plan,
    )


def finalize_grounded_response(
    response_text: Any,
    user_text: Any,
    grounding_plan: Optional[Mapping[str, Any]] = None,
    evidence_bundle: Any = None,
    prompt_profile: Optional[Mapping[str, Any]] = None,
    interaction_plan: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Audit grounding and make only two conservative classes of override."""

    raw = str(response_text or "").strip()
    plan = (
        dict(grounding_plan)
        if isinstance(grounding_plan, Mapping)
        else plan_grounding(
            user_text,
            interaction_plan=interaction_plan,
            prompt_profile=prompt_profile,
        )
    )
    bundle = _coerce_evidence_bundle(
        user_text,
        evidence_bundle,
        interaction_plan=interaction_plan,
        prompt_profile=prompt_profile,
        grounding_plan=plan,
    )
    diagnostics = evidence_diagnostics(user_text, bundle["evidence"], response_text=raw)
    arithmetic = solve_exact_arithmetic(user_text)
    reasoning = solve_reasoned_problem(user_text)
    # The raw user wording is the authority for a strict-evidence override. A
    # caller-supplied or stale plan cannot invent permission to replace text.
    strict_only = bool(_STRICT_EVIDENCE_ONLY_RE.search(_clean_text(user_text, MAX_QUERY_CHARS)))

    text = raw
    reason = "audit_only"
    if strict_only and diagnostics["sufficiency"] != "sufficient":
        if diagnostics["sufficiency"] == "conflicting":
            text = (
                "I can't answer that from the supplied evidence alone because the supplied "
                "evidence conflicts."
            )
            reason = "strict_evidence_conflicting"
        elif diagnostics["sufficiency"] == "no_evidence":
            text = (
                "I can't answer that from the supplied evidence alone because no usable "
                "evidence was provided."
            )
            reason = "strict_evidence_no_evidence"
        else:
            text = (
                "I can't answer that from the supplied evidence alone because it does not "
                "directly support enough of the requested answer."
            )
            reason = "strict_evidence_insufficient"
    elif bool(arithmetic.get("solved")):
        text = f"The exact result is {arithmetic['display']}."
        if arithmetic.get("approximation"):
            text = (
                f"The exact result is {arithmetic['display']} "
                f"(approximately {arithmetic['approximation']})."
            )
        reason = "explicit_arithmetic_exact"
    elif bool(reasoning.get("override_allowed")):
        # Only a solved problem whose own verification passed, with no
        # disagreement between applicable solvers, may replace the response.
        module = _load_reasoning_module()
        wants_steps = bool(_SHOW_WORK_RE.search(_clean_text(user_text, MAX_QUERY_CHARS)))
        rendered = ""
        if module is not None:
            try:
                rendered = str(module.render_reasoning_answer(reasoning, include_steps=wants_steps))
            except Exception:  # pragma: no cover - defensive
                rendered = ""
        rendered = rendered.strip() or str(reasoning.get("text") or "").strip()
        if rendered:
            text = rendered
            reason = "verified_reasoning_solution"

    return {
        "text": text,
        "changed": text != raw,
        "reason": reason,
        "grounding": diagnostics,
        "citations": diagnostics["citation_audit"],
        "arithmetic": arithmetic,
        "reasoning": reasoning,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }


__all__ = [
    "GROUNDING_RUNTIME_VERSION",
    "GROUNDING_SCHEMA_VERSION",
    "build_evidence_bundle",
    "evidence_diagnostics",
    "finalize_grounded_response",
    "normalize_evidence_rows",
    "plan_grounding",
    "redact_external_query",
    "solve_exact_arithmetic",
    "solve_reasoned_problem",
    "validate_citations",
]
