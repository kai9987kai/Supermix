"""Conversation state v1 for Supermix.

The existing understanding layers are per-turn: `prompt_understanding` profiles
the current request against a bounded window of recent turns, and
`interaction_planner` chooses a strategy for that one turn. Nothing accumulates.
A constraint the user stated ten turns ago is gone, an unanswered clarifying
question can be asked again on the next turn, and a response that repeats an
earlier answer verbatim looks exactly like a fresh one to the ranker.

This module derives a bounded, accumulating view of the whole conversation:

* durable user commitments (preferences, constraints, identity, tooling) with
  supersession when the user changes their mind;
* questions the assistant asked, and whether a later user turn answered them,
  including detection of a clarification loop;
* topic threads and topic shift;
* what the assistant has already delivered, for repetition detection;
* user requests that the next assistant turn did not engage with;
* contradictions between what the user established earlier and says now.

Design contract
---------------
* Pure and deterministic. The state is derived from the turn list every time
  rather than mutated in place, so the same conversation always produces the
  same state and any turn sequence can be replayed in a test.
* Bounded everywhere: turns, commitments, questions, threads, signatures, and
  text lengths all have hard caps.
* JSON-safe output; diagnostics carry counts and flags only, never turn text.
* Advisory. This layer contributes a bounded ranking term and audit flags. It
  has no authority over routing, compute, tools, permissions, or safety.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple


CONVERSATION_STATE_SCHEMA_VERSION = "supermix-conversation-state-v1"
CONVERSATION_STATE_VERSION = "supermix-conversation-state-runtime-v2"

MAX_TURNS = 40
MAX_TEXT_CHARS = 400
MAX_COMMITMENTS = 24
MAX_OPEN_QUESTIONS = 8
MAX_THREADS = 6
MAX_DELIVERED = 24
MAX_UNADDRESSED = 8
MAX_CONTRADICTIONS = 8
MAX_TERMS = 24

# A candidate may move by at most this much in total from conversation signals,
# so continuity can break a tie but can never override retrieval relevance.
MAX_CONVERSATION_SCORE = 0.12

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_'-]*")
# A sentence ends at .!? only when the next character is not a digit, so
# "Python 3.12" and "v1.2.3" stay inside one sentence.
_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?!\d)|$)", re.DOTALL)

_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being
    but by can could did do does doing done for from get got had has have he
    her him his how i if in into is it its just like make me more most my no
    not now of on once only or other our out over own please put same she should
    so some such than that the their them then there these they this those to
    too under up us use used using very was way we were what when where which
    while who why will with would you your yours
    """.split()
)

# Durable things a user says about how they want to be helped.
_PREFERENCE_RE = re.compile(
    r"\b(?:i\s+(?:prefer|like|want|need|expect)|please\s+(?:always|never)|"
    r"from\s+now\s+on|going\s+forward|in\s+future|always|make\s+sure\s+(?:to|you)|"
    r"be\s+sure\s+to|remember\s+to|keep\s+it)\b",
    re.IGNORECASE,
)
_PROHIBITION_RE = re.compile(
    r"\b(?:don't|do\s+not|never|stop|avoid|no\s+more|without|must\s+not|"
    r"i\s+don't\s+want|please\s+don't)\b",
    re.IGNORECASE,
)
_IDENTITY_RE = re.compile(
    r"\b(?:my\s+name\s+is|call\s+me|i\s+am\s+called|i'm\s+called|i\s+go\s+by)\b",
    re.IGNORECASE,
)
_TOOLING_RE = re.compile(
    r"\b(?:i\s+(?:use|am\s+using|'m\s+using|work\s+(?:in|with)|write|code\s+in|run|"
    r"develop\s+(?:in|with))|we\s+(?:use|are\s+using)|our\s+stack\s+is|the\s+project\s+uses)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|don't|doesn't|didn't|isn't|aren't|"
    r"wasn't|weren't|won't|stop|avoid)\b",
    re.IGNORECASE,
)
_QUESTION_CUE_RE = re.compile(
    r"\b(?:which|what|who|when|where|why|how|do you|did you|are you|is it|"
    r"would you|should i|could you|can you|shall i)\b",
    re.IGNORECASE,
)
_REQUEST_RE = re.compile(
    r"\b(?:please|can you|could you|would you|write|build|create|make|fix|add|"
    r"show|explain|give me|help me|implement|generate|find|list|compare|"
    r"summarize|summarise|convert|refactor|review)\b",
    re.IGNORECASE,
)
_ACKNOWLEDGEMENT_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|no|nope|sure|ok|okay|correct|right|exactly|that's right|"
    r"the\s+(?:first|second|third|latter|former)|option\s+\w+|both|neither|either)\b",
    re.IGNORECASE,
)
# A bare imperative is the most natural way to state a style preference and it
# matches none of the cue patterns above: "be brief" has no "I prefer", no
# "always", no negation. Without this the preference is never recorded, so it
# cannot reach the ranker.
_STYLE_DIRECTIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:be|keep\s+(?:it|them|things)|make\s+(?:it|them))\s+"
    r"(?:\w+\s+){0,2}"
    r"(?:brief|concise|short|shorter|terse|succinct|detailed|thorough|comprehensive)\b",
    re.IGNORECASE,
)
_CONCISE_RE = re.compile(r"\b(?:concise|brief|short|shorter|terse|to the point|tl;?dr|summar)\w*\b", re.IGNORECASE)
_DETAIL_RE = re.compile(r"\b(?:detailed?|thorough|in depth|in-depth|elaborate|comprehensive|verbose|explain fully)\b", re.IGNORECASE)

# A fresh request on the *current* turn, in either direction. A standing
# preference must never override what the user just asked for, so every surface
# that routes `style_request` needs the same guard. These live here, next to the
# commitment patterns they override, rather than being restated per surface:
# `chat_pipeline` re-exports DETAIL_REQUEST_RE for the ranker and
# `conversation_directive` uses both for the generative surfaces.
DETAIL_REQUEST_RE = re.compile(
    r"\b(?:in (?:more )?detail|detailed|thorough(?:ly)?|in[- ]depth|elaborate|"
    r"expand|comprehensive|walk me through|walkthrough|explain (?:it |this |that )?fully|"
    r"tell me more|say more|more detail|at length|step by step)\b",
    re.IGNORECASE,
)
BREVITY_REQUEST_RE = re.compile(
    r"\b(?:briefly|be brief|in brief|keep it (?:short|brief)|short(?:er)? (?:answer|version)|"
    r"concisely|in one (?:line|sentence)|one[- ](?:line|sentence) (?:answer|version)|"
    r"tl;?dr|just the answer|summari[sz]e (?:it|this|that))\b",
    re.IGNORECASE,
)
# Explicitly turn-scoped instructions are useful for the current request but
# must not become durable session commitments. The current prompt still carries
# them; excluding them here only prevents "this time" from silently becoming
# "from now on" on later turns.
_TRANSIENT_COMMITMENT_RE = re.compile(
    r"\b(?:this time|for (?:just )?this (?:reply|answer|request|task|turn)|"
    r"(?:only|just) for (?:this (?:reply|answer|request|task|turn)|now)|"
    r"on this (?:turn|occasion))\b",
    re.IGNORECASE,
)
_MIXED_ORIGIN_FENCE_RE = re.compile(
    r"```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)"
)
_MIXED_ORIGIN_INLINE_CODE_RE = re.compile(r"`[^`]{1,500}`")
_MIXED_ORIGIN_QUOTED_RE = re.compile(
    r'"[^"\r\n]*(?:"|$)|“[^”\r\n]*(?:”|$)|‘[^’\r\n]*(?:’|$)'
)
_MIXED_ORIGIN_BLOCKQUOTE_RE = re.compile(
    r"(?:^|\s)(?:[-*+]\s*)?>\s*.*?(?:[.!?]+|$)",
    re.DOTALL,
)
_EXTERNAL_ATTRIBUTION_RE = re.compile(
    r"(?:\b(?:quote|quoted|example|sample|hypothetical|attachment|document|file|"
    r"readme|web\s*page|website|search\s+result|email|article|post|transcript|"
    r"tool\s+output|model\s+output|assistant\s+output|external\s+(?:text|content))\b"
    r"[^.!?]{0,140}(?::|\b(?:says?|said|states?|stated|reads?|wrote|writes|"
    r"contains?|includes?|shows?|suggests?|recommends?))\s*$|"
    r"\b(?:according\s+to|copied\s+from|quoted\s+from)\b[^.!?]{0,140}$|"
    r"\b(?:says?|said|states?|stated|reads?|wrote|writes)\s*:?\s*$)",
    re.IGNORECASE,
)
_EXTERNAL_SOURCE_RE = re.compile(
    r"\b(?:quote|quoted|example|sample|hypothetical|attachment|document|file|"
    r"readme|web\s*page|website|search\s+result|email|article|post|transcript|"
    r"tool\s+(?:output|result|response)|model\s+(?:output|result|response)|"
    r"assistant\s+(?:output|result|response|message)|system\s+message|developer\s+message|"
    r"retrieval\s+result|external\s+(?:text|content))\b",
    re.IGNORECASE,
)
_MIXED_ORIGIN_ROLE_WRAPPER_RE = re.compile(
    r"(?:<\/?(?:tool|assistant|model|system|developer|result|response)[^>]{0,80}>|"
    r"\[(?:tool|assistant|model|system|developer|result|response)\]|"
    r"(?:^|[,{]\s*)[\"']?(?:tool|assistant|model|system|developer)[\"']?\s*:|"
    r"[\"']?role[\"']?\s*:\s*[\"']?(?:tool|assistant|model|system|developer)\b|"
    r"^\s*(?:tool|assistant|model|system|developer|retrieval)\s+"
    r"(?:output|result|response|message)\s*[:\-—])",
    re.IGNORECASE,
)
_DIRECT_USER_FRAME = (
    r"(?:actually|personally|generally|normally|usually|currently|also|instead|"
    r"fyi|for\s+your\s+information|by\s+the\s+way|just\s+so\s+you\s+know|"
    r"as\s+a\s+reminder|to\s+clarify|for\s+the\s+record|"
    r"for\s+(?:this\s+)?(?:project|conversation|context)|"
    r"in\s+this\s+(?:project|conversation|context)|"
    r"(?:hi|hello|hey)(?:\s+there)?)"
)
_DIRECT_USER_CUE_PREFIX_RE = re.compile(
    rf"^\s*(?:{_DIRECT_USER_FRAME}\s*[,;:—-]?\s*){{0,2}}"
    r"(?:(?:i|we|please)\s+|you\s+(?:can|may)\s+)?$",
    re.IGNORECASE,
)
_DIRECT_FIRST_PERSON_COMPOUND_PREFIX_RE = re.compile(
    rf"^\s*(?:{_DIRECT_USER_FRAME}\s*[,;:—-]?\s*){{0,2}}"
    r"(?:i(?:'m|\s+am)?|my|we(?:'re|\s+are)?|our)\b[^.!?;:]{1,160}"
    r"\b(?:and|but)\s+(?:(?:i|we|please)\s+)?$",
    re.IGNORECASE,
)
_REPORTED_SPEECH_VERB_RE = re.compile(
    r"\b(?:says?|said|repl(?:y|ies|ied)|respond(?:s|ed)?|wrote|writes|quoted?|states?|stated)\b",
    re.IGNORECASE,
)


def style_preference_of(text: Any) -> str:
    """Which standing style, if any, a single statement expresses.

    Published because a surface that quotes commitments into a prompt has to be
    able to tell which of them are already carried by the style request, and
    saying the same instruction twice in one prompt is worse than saying it once.
    """

    value = str(text or "")
    if _CONCISE_RE.search(value):
        return "concise"
    if _DETAIL_RE.search(value):
        return "detailed"
    return ""


def _clean(value: Any, limit: int = MAX_TEXT_CHARS) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    without_format_controls = "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Cf"
    )
    return _WS_RE.sub(" ", without_format_controls).strip()[: max(0, int(limit))]


def _stem(token: str) -> str:
    """Light suffix folding so "index" and "indexes" land on one thread."""

    for suffix, keep in (("ies", 3), ("ing", 3), ("es", 2), ("s", 1)):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            base = token[: -keep]
            return base + "y" if suffix == "ies" else base
    return token


def _terms(value: Any, limit: int = MAX_TERMS) -> List[str]:
    return _terms_from_tokens(_TOKEN_RE.findall(str(value or "").lower()), limit)


def _term_set(value: Any) -> frozenset:
    return frozenset(_terms(value, MAX_TERMS * 2))


def _overlap(left: Iterable[str], right: Iterable[str]) -> float:
    first, second = set(left), set(right)
    if not first or not second:
        return 0.0
    return round(len(first & second) / len(first | second), 6)


def _containment(inner: Iterable[str], outer: Iterable[str]) -> float:
    first, second = set(inner), set(outer)
    if not first:
        return 0.0
    return round(len(first & second) / len(first), 6)


def _shingles_from_tokens(tokens: Sequence[str], size: int = 5) -> frozenset:
    if len(tokens) < size:
        return frozenset([" ".join(tokens)]) if tokens else frozenset()
    return frozenset(
        " ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    )


def _shingles(text: str, size: int = 5) -> frozenset:
    return _shingles_from_tokens(_TOKEN_RE.findall(str(text or "").lower()), size)


def _terms_from_tokens(tokens: Sequence[str], limit: int = MAX_TERMS) -> List[str]:
    seen: List[str] = []
    for token in tokens:
        if len(token) < 2 or token in _STOPWORDS:
            continue
        stemmed = _stem(token)
        if stemmed in seen:
            continue
        seen.append(stemmed)
        if len(seen) >= limit:
            break
    return seen


def _sentences(text: str) -> List[str]:
    return [part.strip() for part in _SENTENCE_RE.findall(str(text or "")) if part.strip()]


def _is_question(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    return stripped.endswith("?") or bool(_QUESTION_CUE_RE.match(stripped))


def _last_question(text: str) -> str:
    for sentence in reversed(_sentences(text)):
        if sentence.endswith("?"):
            return sentence
    return ""


class _Turn(NamedTuple):
    index: int
    role: str
    text: str
    commitment_eligible: bool


def _normalize_turn_value(value: Any, role: str) -> Tuple[str, bool]:
    raw = str(value or "")
    text = _clean(raw)
    eligible = bool(
        role == "user"
        and len(raw) <= MAX_TEXT_CHARS
        and not _EXTERNAL_SOURCE_RE.search(raw)
        and not _MIXED_ORIGIN_ROLE_WRAPPER_RE.search(raw)
    )
    return text, eligible


def _normalize_turns(turns: Any, current_user_text: Any = "") -> List[_Turn]:
    """Accept role dicts, (user, assistant) pairs, or a flat message list."""

    rows: List[Tuple[str, str, bool]] = []
    if isinstance(turns, Mapping):
        turns = turns.get("turns") or turns.get("messages") or ()
    if isinstance(turns, (str, bytes, bytearray)) or not isinstance(turns, (list, tuple)):
        # Callers hand this layer whatever their history happens to be; anything
        # that is not a turn sequence contributes no turns rather than raising.
        turns = ()
    for item in list(turns)[-MAX_TURNS:]:
        if isinstance(item, Mapping):
            # A turn log keyed by speaker rather than by role. The Studio memory
            # store writes this shape, and it used to fall through to the
            # `content` lookup, produce nothing, and be dropped in silence.
            if "role" not in item and ("user" in item or "assistant" in item):
                user_text, user_eligible = _normalize_turn_value(item.get("user"), "user")
                assistant_text, _ = _normalize_turn_value(item.get("assistant"), "assistant")
                if user_text:
                    rows.append(("user", user_text, user_eligible))
                if assistant_text:
                    rows.append(("assistant", assistant_text, False))
                continue
            role = str(item.get("role") or "").strip().lower()
            content, commitment_eligible = _normalize_turn_value(
                item.get("content") or item.get("text") or "", role
            )
            # Unknown, tool, system, and future roles are not user statements.
            # Dropping them is safer than the historical default that mapped
            # every non-assistant role to user and could manufacture a durable
            # commitment from tool output or imported metadata.
            if role not in {"user", "assistant"} or not content:
                continue
            rows.append((role, content, commitment_eligible))
            continue
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_text, user_eligible = _normalize_turn_value(item[0], "user")
            assistant_text, _ = _normalize_turn_value(item[1], "assistant")
            if user_text:
                rows.append(("user", user_text, user_eligible))
            if assistant_text:
                rows.append(("assistant", assistant_text, False))
            continue
        text, commitment_eligible = _normalize_turn_value(item, "user")
        if text:
            rows.append(("user", text, commitment_eligible))

    current, current_eligible = _normalize_turn_value(current_user_text, "user")
    if current and not (rows and rows[-1][:2] == ("user", current)):
        rows.append(("user", current, current_eligible))

    rows = rows[-MAX_TURNS:]
    return [
        _Turn(index=index, role=role, text=text, commitment_eligible=eligible)
        for index, (role, text, eligible) in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _commitment_kind(text: str) -> str:
    if _IDENTITY_RE.search(text):
        return "identity"
    if _TOOLING_RE.search(text):
        return "tooling"
    if _PROHIBITION_RE.search(text):
        return "constraint"
    if _PREFERENCE_RE.search(text) or _STYLE_DIRECTIVE_RE.search(text):
        return "preference"
    return ""


def _commitment_source_text(text: str) -> str:
    """Remove embedded data spans before interpreting durable user intent."""

    cooked = _MIXED_ORIGIN_FENCE_RE.sub(" ", str(text or ""))
    cooked = _MIXED_ORIGIN_INLINE_CODE_RE.sub(" ", cooked)
    cooked = _MIXED_ORIGIN_QUOTED_RE.sub(" ", cooked)
    cooked = _MIXED_ORIGIN_BLOCKQUOTE_RE.sub(" ", cooked)
    return cooked


def _commitment_has_external_attribution(sentence: str) -> bool:
    if _EXTERNAL_SOURCE_RE.search(sentence) or _MIXED_ORIGIN_ROLE_WRAPPER_RE.search(sentence):
        return True
    cue_matches = [
        match
        for pattern in (
            _IDENTITY_RE,
            _TOOLING_RE,
            _PROHIBITION_RE,
            _PREFERENCE_RE,
            _STYLE_DIRECTIVE_RE,
        )
        if (match := pattern.search(sentence)) is not None
    ]
    if not cue_matches:
        return False
    cue_start = min(match.start() for match in cue_matches)
    prefix = sentence[:cue_start]
    if prefix.rstrip().endswith((":", '"', "'", "`", ">")):
        return True
    return bool(_EXTERNAL_ATTRIBUTION_RE.search(prefix[-200:]))


def _commitment_is_direct_user_statement(sentence: str) -> bool:
    """Require the first durable cue to belong to a direct user clause.

    The cue regexes intentionally recognize both first-person statements and
    direct imperatives.  Reported speech can contain either form, so a durable
    commitment is admitted only when the text before its first cue is a bounded
    user framing such as ``Actually,`` or ``For this project,``.  A compound
    direct statement such as ``My name is Kai and I prefer concise answers``
    starts with an identity cue and remains eligible as one commitment.
    """

    cue_matches = [
        match
        for pattern in (
            _IDENTITY_RE,
            _TOOLING_RE,
            _PROHIBITION_RE,
            _PREFERENCE_RE,
            _STYLE_DIRECTIVE_RE,
        )
        if (match := pattern.search(sentence)) is not None
    ]
    if not cue_matches:
        return False
    cue_start = min(match.start() for match in cue_matches)
    prefix = sentence[:cue_start]
    return bool(
        _DIRECT_USER_CUE_PREFIX_RE.fullmatch(prefix)
        or (
            not _REPORTED_SPEECH_VERB_RE.search(prefix)
            and _DIRECT_FIRST_PERSON_COMPOUND_PREFIX_RE.fullmatch(prefix)
        )
    )


def _extract_commitments(turns: Sequence[_Turn]) -> List[Dict[str, Any]]:
    commitments: List[Dict[str, Any]] = []
    for turn in turns:
        if turn.role != "user" or not turn.commitment_eligible:
            continue
        if _EXTERNAL_SOURCE_RE.search(turn.text) or _MIXED_ORIGIN_ROLE_WRAPPER_RE.search(
            turn.text
        ):
            continue
        for sentence in _sentences(_commitment_source_text(turn.text)):
            if _is_question(sentence):
                continue
            if _TRANSIENT_COMMITMENT_RE.search(sentence):
                continue
            if _commitment_has_external_attribution(sentence):
                continue
            if not _commitment_is_direct_user_statement(sentence):
                continue
            kind = _commitment_kind(sentence)
            if not kind:
                continue
            terms = _terms(sentence)
            if not terms:
                continue
            commitments.append(
                {
                    "id": f"C{len(commitments) + 1}",
                    "kind": kind,
                    "text": _clean(sentence, 200),
                    "turn": turn.index,
                    "polarity": "negate" if _NEGATION_RE.search(sentence) else "affirm",
                    "terms": terms,
                    "superseded_by": "",
                    "active": True,
                }
            )
            if len(commitments) >= MAX_COMMITMENTS:
                return commitments
    return commitments


_DURABLE_KINDS = frozenset({"identity", "tooling", "constraint", "preference"})
# Words that carry the framing of a statement rather than its subject. Two
# statements are about the same thing when their *subjects* match, so "I use
# Python" and "I don't use Python" must compare as the same subject even though
# one is classified as tooling and the other as a constraint.
_FRAMING_TERMS = frozenset(
    """
    actually anymore always never please prefer like want need expect sure
    remember keep make stop avoid dont don't do not no more longer instead
    now going forward future
    """.split()
)


def _subject_terms(terms: Sequence[str]) -> List[str]:
    return [term for term in terms if term not in _FRAMING_TERMS]


def _resolve_supersession(commitments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A later statement about the same subject replaces an earlier one."""

    for later_index in range(len(commitments) - 1, -1, -1):
        later = commitments[later_index]
        if later["kind"] not in _DURABLE_KINDS:
            continue
        later_subject = _subject_terms(later["terms"])
        if not later_subject:
            continue
        for earlier in commitments[:later_index]:
            if not earlier["active"] or earlier["kind"] not in _DURABLE_KINDS:
                continue
            earlier_subject = _subject_terms(earlier["terms"])
            if not earlier_subject:
                continue
            # Kind-agnostic on purpose: changing your mind usually changes the
            # grammatical framing too.
            if _overlap(earlier_subject, later_subject) < 0.4 and (
                _containment(earlier_subject, later_subject) < 0.6
            ):
                continue
            earlier["active"] = False
            earlier["superseded_by"] = later["id"]
    return commitments


def _extract_contradictions(commitments: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """A superseding statement that flips polarity is a stated change of mind."""

    rows: List[Dict[str, Any]] = []
    by_id = {row["id"]: row for row in commitments}
    for earlier in commitments:
        successor_id = str(earlier.get("superseded_by") or "")
        if not successor_id:
            continue
        later = by_id.get(successor_id)
        if later is None or later["polarity"] == earlier["polarity"]:
            continue
        rows.append(
            {
                "id": f"X{len(rows) + 1}",
                "earlier": earlier["id"],
                "later": later["id"],
                "earlier_turn": int(earlier["turn"]),
                "later_turn": int(later["turn"]),
                "kind": str(earlier["kind"]),
                "shared_terms": sorted(
                    set(_subject_terms(earlier["terms"])) & set(_subject_terms(later["terms"]))
                )[:8],
            }
        )
        if len(rows) >= MAX_CONTRADICTIONS:
            break
    return rows


def _answers_question(user_text: str, question_terms: Sequence[str]) -> bool:
    if not user_text:
        return False
    if _ACKNOWLEDGEMENT_RE.match(user_text):
        return True
    words = _TOKEN_RE.findall(user_text.lower())
    if len(words) <= 24 and _containment(question_terms, _terms(user_text, 64)) >= 0.25:
        return True
    return False


def _extract_open_questions(turns: Sequence[_Turn]) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    for turn in turns:
        if turn.role != "assistant":
            continue
        question = _last_question(turn.text)
        if not question:
            continue
        terms = _terms(question)
        if not terms:
            continue
        status = "open"
        answered_by = -1
        for later in turns[turn.index + 1 :]:
            if later.role != "user":
                continue
            if _answers_question(later.text, terms):
                status = "answered"
                answered_by = later.index
            break
        else:
            status = "open"
        questions.append(
            {
                "id": f"Q{len(questions) + 1}",
                "text": _clean(question, 200),
                "turn": turn.index,
                "terms": terms,
                "status": status,
                "answered_by_turn": answered_by,
            }
        )
        if len(questions) >= MAX_OPEN_QUESTIONS:
            break
    return questions


def _detect_clarification_loop(questions: Sequence[Mapping[str, Any]]) -> bool:
    """The assistant asked substantially the same question more than once."""

    for index, question in enumerate(questions):
        for other in questions[index + 1 :]:
            if _overlap(question["terms"], other["terms"]) >= 0.6:
                return True
    return False


def _extract_threads(turns: Sequence[_Turn]) -> List[Dict[str, Any]]:
    threads: List[Dict[str, Any]] = []
    for turn in turns:
        if turn.role != "user":
            continue
        terms = _terms(turn.text)
        if not terms:
            continue
        best: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for thread in threads:
            score = _overlap(thread["terms"], terms)
            if score > best_score:
                best, best_score = thread, score
        if best is not None and best_score >= 0.2:
            best["terms"] = sorted(set(best["terms"]) | set(terms))[:MAX_TERMS]
            best["last_turn"] = turn.index
            best["turn_count"] = int(best["turn_count"]) + 1
            continue
        threads.append(
            {
                "id": f"T{len(threads) + 1}",
                "terms": terms,
                "first_turn": turn.index,
                "last_turn": turn.index,
                "turn_count": 1,
            }
        )
        if len(threads) > MAX_THREADS:
            threads.sort(key=lambda row: (int(row["last_turn"]), int(row["turn_count"])))
            threads.pop(0)
    latest = max((int(row["last_turn"]) for row in threads), default=-1)
    for thread in threads:
        thread["active"] = int(thread["last_turn"]) == latest
    threads.sort(key=lambda row: row["id"])
    return threads


def _extract_delivered(turns: Sequence[_Turn]) -> List[Dict[str, Any]]:
    delivered: List[Dict[str, Any]] = []
    for turn in turns:
        if turn.role != "assistant" or not turn.text:
            continue
        delivered.append(
            {
                "turn": turn.index,
                "terms": _terms(turn.text),
                "shingles": sorted(_shingles(turn.text))[:64],
            }
        )
    return delivered[-MAX_DELIVERED:]


def _extract_unaddressed(turns: Sequence[_Turn]) -> List[Dict[str, Any]]:
    """User requests the immediately following assistant turn did not engage."""

    rows: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn.role != "user":
            continue
        if not (_REQUEST_RE.search(turn.text) or _is_question(turn.text)):
            continue
        # A durable preference or constraint is not an unmet request; a short
        # acknowledgement is a perfectly good reply to it.
        if any(_commitment_kind(sentence) for sentence in _sentences(turn.text)):
            continue
        reply = next((later for later in turns[index + 1 :] if later.role == "assistant"), None)
        if reply is None:
            continue
        request_terms = _terms(turn.text)
        if not request_terms:
            continue
        if _containment(request_terms, _terms(reply.text, 96)) >= 0.2:
            continue
        rows.append(
            {
                "id": f"U{len(rows) + 1}",
                "turn": turn.index,
                # Kept as user-authored data, never diagnostics or a system
                # instruction.  The directive layer may surface one bounded,
                # sanitised row only after an explicit "you missed..." repair
                # request from the current user.
                "text": _clean(turn.text),
                "terms": request_terms[:8],
                "reply_turn": reply.index,
            }
        )
    rows = rows[-MAX_UNADDRESSED:]
    for row_index, row in enumerate(rows, start=1):
        row["id"] = f"U{row_index}"
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_conversation_state(
    turns: Any = (),
    current_user_text: Any = "",
) -> Dict[str, Any]:
    """Derive the accumulated, JSON-safe state of a conversation."""

    rows = _normalize_turns(turns, current_user_text)
    if not rows:
        return _empty_state()

    commitments = _resolve_supersession(_extract_commitments(rows))
    contradictions = _extract_contradictions(commitments)
    questions = _extract_open_questions(rows)
    threads = _extract_threads(rows)
    delivered = _extract_delivered(rows)
    unaddressed = _extract_unaddressed(rows)

    user_rows = [row for row in rows if row.role == "user"]
    current = user_rows[-1] if user_rows else None
    current_terms = _terms(current.text) if current is not None else []
    previous_user = user_rows[-2] if len(user_rows) >= 2 else None
    active_thread = next((row for row in threads if row.get("active")), None)

    # A topic shift is simply the current request not continuing the previous
    # one. Comparing against the immediately previous user turn keeps this
    # readable and avoids thread-clustering artefacts.
    topic_shift = bool(
        current_terms
        and previous_user is not None
        and _overlap(_terms(previous_user.text), current_terms) < 0.15
    )

    open_questions = [row for row in questions if row["status"] == "open"]
    just_answered = [
        row
        for row in questions
        if row["status"] == "answered"
        and current is not None
        and int(row["answered_by_turn"]) == current.index
    ]

    style = ""
    for row in reversed(commitments):
        if not row["active"]:
            continue
        style = style_preference_of(row["text"])
        if style:
            break

    flags = {
        "clarification_loop": _detect_clarification_loop(questions),
        "unresolved_open_question": bool(open_questions),
        "answered_open_question": bool(just_answered),
        "topic_shift": topic_shift,
        "contradiction_present": bool(contradictions),
        "unaddressed_request": bool(unaddressed),
        "has_active_commitments": any(row["active"] for row in commitments),
    }

    return {
        "schema_version": CONVERSATION_STATE_SCHEMA_VERSION,
        "runtime_version": CONVERSATION_STATE_VERSION,
        "scope": "conversation_context_only",
        "advisory_only": True,
        "turn_count": len(rows),
        "user_turn_count": sum(1 for row in rows if row.role == "user"),
        "assistant_turn_count": sum(1 for row in rows if row.role == "assistant"),
        "commitments": commitments,
        "active_commitments": [row["id"] for row in commitments if row["active"]],
        "contradictions": contradictions,
        "questions": questions,
        "open_questions": [row["id"] for row in open_questions],
        "answered_now": [row["id"] for row in just_answered],
        "threads": threads,
        "active_thread": str(active_thread["id"]) if active_thread else "",
        "delivered": delivered,
        "unaddressed": unaddressed,
        "style_request": style,
        "flags": flags,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_permissions": False,
        },
    }


def _empty_state() -> Dict[str, Any]:
    return {
        "schema_version": CONVERSATION_STATE_SCHEMA_VERSION,
        "runtime_version": CONVERSATION_STATE_VERSION,
        "scope": "conversation_context_only",
        "advisory_only": True,
        "turn_count": 0,
        "user_turn_count": 0,
        "assistant_turn_count": 0,
        "commitments": [],
        "active_commitments": [],
        "contradictions": [],
        "questions": [],
        "open_questions": [],
        "answered_now": [],
        "threads": [],
        "active_thread": "",
        "delivered": [],
        "unaddressed": [],
        "style_request": "",
        "flags": {
            "clarification_loop": False,
            "unresolved_open_question": False,
            "answered_open_question": False,
            "topic_shift": False,
            "contradiction_present": False,
            "unaddressed_request": False,
            "has_active_commitments": False,
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_permissions": False,
        },
    }


def repetition_score(candidate_text: Any, state: Optional[Mapping[str, Any]]) -> float:
    """How much of this candidate the assistant has already said, in [0, 1]."""

    if not isinstance(state, Mapping):
        return 0.0
    candidate = _shingles(_clean(candidate_text, 2000))
    if not candidate:
        return 0.0
    worst = 0.0
    for row in state.get("delivered") or ():
        if not isinstance(row, Mapping):
            continue
        previous = frozenset(row.get("shingles") or ())
        if not previous:
            continue
        overlap = len(candidate & previous) / len(candidate)
        worst = max(worst, overlap)
    return round(min(1.0, worst), 6)


def _prepare_scoring(state: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Derive everything a candidate is scored against, once per turn.

    Ranking calls this for a whole candidate pool, so anything that depends only
    on the conversation is hoisted out of the per-candidate loop.
    """

    if not isinstance(state, Mapping) or not state.get("turn_count"):
        return None

    delivered = [
        frozenset(row.get("shingles") or ())
        for row in state.get("delivered") or ()
        if isinstance(row, Mapping) and row.get("shingles")
    ]

    answered_now = set(state.get("answered_now") or ())
    answered_terms = [
        frozenset(question.get("terms") or ())
        for question in state.get("questions") or ()
        if isinstance(question, Mapping) and question.get("id") in answered_now
    ]

    flags = state.get("flags") if isinstance(state.get("flags"), Mapping) else {}
    active_terms: frozenset = frozenset()
    if not flags.get("topic_shift"):
        active_id = str(state.get("active_thread") or "")
        for thread in state.get("threads") or ():
            if isinstance(thread, Mapping) and thread.get("id") == active_id:
                active_terms = frozenset(thread.get("terms") or ())
                break

    return {
        "delivered": delivered,
        "answered_terms": answered_terms,
        "style": str(state.get("style_request") or ""),
        "active_terms": active_terms,
    }


def _score_prepared(candidate_text: Any, prepared: Mapping[str, Any]) -> float:
    text = _clean(candidate_text, 2000)
    if not text:
        return 0.0

    # One tokenization feeds the shingles, the term set, and the word count.
    tokens = _TOKEN_RE.findall(text.lower())
    score = 0.0

    # Saying the same thing again is the most visible conversational failure.
    delivered = prepared["delivered"]
    if delivered and tokens:
        candidate_shingles = _shingles_from_tokens(tokens)
        if candidate_shingles:
            size = len(candidate_shingles)
            worst = 0.0
            for previous in delivered:
                overlap = len(candidate_shingles & previous) / size
                if overlap > worst:
                    worst = overlap
            repetition = round(min(1.0, worst), 6)
            if repetition >= 0.55:
                score -= 1.0 * repetition

    candidate_terms = _terms_from_tokens(tokens, 96)

    # When the user has just answered a question, reward continuing that thread
    # rather than restating the question.
    if prepared["answered_terms"]:
        ends_with_question = text.rstrip().endswith("?")
        term_set = set(candidate_terms)
        for question_terms in prepared["answered_terms"]:
            if _containment(question_terms, term_set) >= 0.3:
                score += 0.35
            if ends_with_question:
                score -= 0.45

    # Respect a durable style request.
    style = prepared["style"]
    word_count = len(tokens)
    if style == "concise" and word_count > 0:
        score += 0.3 if word_count <= 45 else -0.3
    elif style == "detailed" and word_count > 0:
        score += 0.3 if word_count >= 40 else -0.2

    # Stay on the active thread unless the user has shifted topic.
    active_terms = prepared["active_terms"]
    if active_terms:
        score += 0.3 * _containment(candidate_terms[:24], active_terms)

    clamped = max(-1.0, min(1.0, score))
    return round(clamped * MAX_CONVERSATION_SCORE, 6)


def score_candidate_for_conversation(
    candidate_text: Any,
    state: Optional[Mapping[str, Any]],
) -> float:
    """Bounded continuity adjustment in [-MAX, +MAX]; 0.0 without state."""

    prepared = _prepare_scoring(state)
    if prepared is None:
        return 0.0
    return _score_prepared(candidate_text, prepared)


def score_candidates_for_conversation(
    candidate_texts: Sequence[Any],
    state: Optional[Mapping[str, Any]],
) -> List[float]:
    """Score a whole candidate pool, preparing the conversation view once."""

    texts = list(candidate_texts or ())
    prepared = _prepare_scoring(state)
    if prepared is None:
        return [0.0] * len(texts)
    return [_score_prepared(text, prepared) for text in texts]


def audit_response_against_state(
    response_text: Any,
    state: Optional[Mapping[str, Any]],
    current_user_text: Any = "",
) -> Dict[str, Any]:
    """Report conversational problems with a response. Audit only."""

    text = _clean(response_text, 2000)
    violations: List[str] = []
    if not isinstance(state, Mapping) or not state.get("turn_count") or not text:
        return {
            "schema_version": CONVERSATION_STATE_SCHEMA_VERSION,
            "checked": False,
            "violations": [],
            "repetition": 0.0,
            "repeats_prior_answer": False,
            "repeats_open_question": False,
            "ignores_style_request": False,
            "authority": "audit_only",
        }

    repetition = repetition_score(text, state)
    if repetition >= 0.55:
        violations.append("repeats_prior_answer")

    repeats_question = False
    flags = state.get("flags") if isinstance(state.get("flags"), Mapping) else {}
    if flags.get("clarification_loop") and text.rstrip().endswith("?"):
        candidate_terms = _terms(text, 96)
        for question in state.get("questions") or ():
            if not isinstance(question, Mapping):
                continue
            if _overlap(question.get("terms") or (), candidate_terms) >= 0.5:
                repeats_question = True
                break
    if repeats_question:
        violations.append("repeats_open_question")

    style = str(state.get("style_request") or "")
    current = str(current_user_text or "")
    if (
        (style == "concise" and DETAIL_REQUEST_RE.search(current))
        or (style == "detailed" and BREVITY_REQUEST_RE.search(current))
    ):
        # The current request has the same precedence as it does in the
        # directive and ranker. Do not label a response as wrong for correctly
        # following the user's newer, opposite request.
        style = ""
    word_count = len(_TOKEN_RE.findall(text))
    ignores_style = bool(
        (style == "concise" and word_count > 90)
        or (style == "detailed" and 0 < word_count < 20)
    )
    if ignores_style:
        violations.append("ignores_style_request")

    return {
        "schema_version": CONVERSATION_STATE_SCHEMA_VERSION,
        "checked": True,
        "violations": violations,
        "repetition": repetition,
        "repeats_prior_answer": "repeats_prior_answer" in violations,
        "repeats_open_question": repeats_question,
        "ignores_style_request": ignores_style,
        "authority": "audit_only",
    }


def render_state_brief(state: Optional[Mapping[str, Any]], max_chars: int = 600) -> str:
    """A compact brief of what the conversation has established."""

    if not isinstance(state, Mapping) or not state.get("turn_count"):
        return ""
    lines: List[str] = []
    active = {row["id"]: row for row in state.get("commitments") or () if isinstance(row, Mapping) and row.get("active")}
    if active:
        lines.append("Established by the user:")
        for row in list(active.values())[:6]:
            lines.append(f"- ({row['kind']}) {row['text']}")
    open_ids = set(state.get("open_questions") or ())
    open_rows = [
        row
        for row in state.get("questions") or ()
        if isinstance(row, Mapping) and row.get("id") in open_ids
    ]
    if open_rows:
        lines.append("Still waiting on:")
        for row in open_rows[:3]:
            lines.append(f"- {row['text']}")
    flags = state.get("flags") if isinstance(state.get("flags"), Mapping) else {}
    raised = sorted(name for name, value in flags.items() if value)
    if raised:
        lines.append("Flags: " + ", ".join(raised))
    return "\n".join(lines)[: max(0, int(max_chars))]


def conversation_state_diagnostics(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Privacy-safe metadata: counts and flags only, never turn text."""

    if not isinstance(state, Mapping):
        state = _empty_state()
    flags = state.get("flags") if isinstance(state.get("flags"), Mapping) else {}
    commitments = [row for row in state.get("commitments") or () if isinstance(row, Mapping)]
    return {
        "schema_version": CONVERSATION_STATE_SCHEMA_VERSION,
        "turn_count": int(state.get("turn_count") or 0),
        "user_turn_count": int(state.get("user_turn_count") or 0),
        "assistant_turn_count": int(state.get("assistant_turn_count") or 0),
        "commitment_count": len(commitments),
        "active_commitment_count": sum(1 for row in commitments if row.get("active")),
        "commitment_kinds": sorted({str(row.get("kind") or "") for row in commitments if row.get("active")}),
        "contradiction_count": len(state.get("contradictions") or ()),
        "question_count": len(state.get("questions") or ()),
        "open_question_count": len(state.get("open_questions") or ()),
        "answered_now_count": len(state.get("answered_now") or ()),
        "thread_count": len(state.get("threads") or ()),
        "unaddressed_count": len(state.get("unaddressed") or ()),
        "style_request": str(state.get("style_request") or ""),
        "flags": {str(key): bool(value) for key, value in sorted(flags.items())},
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_permissions": False,
        },
    }


__all__ = [
    "BREVITY_REQUEST_RE",
    "CONVERSATION_STATE_SCHEMA_VERSION",
    "CONVERSATION_STATE_VERSION",
    "DETAIL_REQUEST_RE",
    "MAX_CONVERSATION_SCORE",
    "audit_response_against_state",
    "build_conversation_state",
    "conversation_state_diagnostics",
    "render_state_brief",
    "repetition_score",
    "score_candidate_for_conversation",
    "score_candidates_for_conversation",
    "style_preference_of",
]
