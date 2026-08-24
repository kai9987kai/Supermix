from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit


GROUNDING_SCHEMA_VERSION = "supermix-grounding-v1"
GROUNDING_RUNTIME_VERSION = "supermix-grounding-runtime-v6"
VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION = "supermix-verified-answer-receipt-v2"

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
MAX_REASONING_QUERY_CHARS = 2000


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
_ARITHMETIC_REQUEST_SUFFIX_RE = re.compile(
    r"\s*(?:[?!.]\s*|\band\s+)(?:please\s+)?(?:"
    r"explain\s+(?:(?:your|the)\s+)?(?:reasoning|working|steps)"
    r"|show\s+(?:(?:your|the)\s+)?(?:work|working|steps)"
    r"|(?:verify|check)\s+(?:(?:the|your)\s+)?(?:answer|result|calculation|work)"
    r")\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_VERIFY_RESULT_RE = re.compile(
    r"\b(?:verify|check)\s+(?:(?:the|your)\s+)?(?:answer|result|calculation|work)\b",
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

# The reasoning engine is optional and can be supplied by a packaged sibling.
# Recheck the narrow probability/prediction assumptions at this grounding
# boundary so a stale or caller-replaced engine cannot acquire answer-rewrite
# authority from an inapplicable computation.
_REASONING_FAIR_DIE_RE = re.compile(
    r"\b(?:a\s+)?(?:standard\s+fair\s+die|fair\s+(?:(?P<sides>\d{1,3})|six)[ -]sided\s+die)\b",
    re.IGNORECASE,
)
_REASONING_PARITY_RE = re.compile(r"\b(?P<parity>odd|even)(?:\s+number)?\b", re.IGNORECASE)
_REASONING_EQUIPROBABLE_RE = re.compile(
    r"\b(?:equally\s+likely|equiprobable)\b|"
    r"\buniform(?:ly)?\s+(?:at\s+)?random\b|"
    r"\bfair\s+(?:experiment|selection|cases?|outcomes?)\b",
    re.IGNORECASE,
)
_REASONING_UNEQUAL_PROBABILITY_RE = re.compile(
    r"\b(?:unequal(?:ly)?|weighted|biased|unfair|non[- ]?uniform|"
    r"different\s+(?:probabilities|rates)|not\s+(?:a\s+)?fair)\b|"
    r"\b(?:not|isn't|aren't|without)\s+(?:equally\s+likely|equiprobable|"
    r"uniform(?:ly)?(?:\s+(?:at\s+)?random)?)\b",
    re.IGNORECASE,
)
_REASONING_UNSUPPORTED_PROBABILITY_EVENT_RE = re.compile(
    r"\b(?:not|except|excluding|other\s+than)\b|"
    r"\b(?:numbered|labelled|labeled)\b|\bfaces?\s*(?:are|=|:)",
    re.IGNORECASE,
)
_REASONING_REPEATED_TRIAL_RE = re.compile(
    r"\b(?:twice|thrice|again|another\s+(?:time|roll|flip|toss)|"
    r"multiple\s+(?:times|rolls?|flips?|tosses?)|repeated(?:ly)?)\b|"
    r"\b(?:[2-9]\d{0,3}|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:(?:coin\s+)?(?:flips?|tosses?)|(?:die\s+)?rolls?|"
    r"(?:fair\s+)?(?:\d{1,3}[ -]sided\s+)?dice)\b|"
    r"\b(?:roll(?:ed|ing)?|flip(?:ped|ping)?|toss(?:ed|ing)?)\b[^?.]{0,36}\b"
    r"(?:[2-9]\d{0,3}|two|three|four|five|six|seven|eight|nine|ten)\s+times?\b|"
    r"\b(?:both|each)\s+(?:of\s+the\s+)?(?:rolls?|flips?|tosses?)\b",
    re.IGNORECASE,
)
_REASONING_ASSUMPTION_CLAUSE_RE = re.compile(
    r"\b(?:assuming|assume|suppose(?:\s+that)?|under\s+the\s+assumption\s+that)\b"
    r"(?P<body>[^.;?!]{1,240})",
    re.IGNORECASE,
)
_REASONING_IID_RE = re.compile(
    r"\b(?:i\s*\.?\s*i\s*\.?\s*d\s*\.?|independent\s+and\s+identically\s+distributed)\b",
    re.IGNORECASE,
)
_REASONING_INDEPENDENCE_RE = re.compile(r"\bindependent(?:ly)?\b", re.IGNORECASE)
_REASONING_STATIONARITY_RE = re.compile(
    r"\b(?:same|constant|fixed|unchanged|stationary)\s+"
    r"(?:(?:underlying|success|event)\s+){0,2}(?:probability|rate)\b|"
    r"\b(?:probability|rate)\s+(?:is|remains|stays)\s+"
    r"(?:the\s+)?(?:same|constant|fixed|unchanged|stationary)\b",
    re.IGNORECASE,
)
_REASONING_NEGATED_ASSUMPTION_RE = re.compile(
    r"\b(?:not|never|without|isn't|aren't|wasn't|weren't)\s+"
    r"(?:(?:mutually|statistically)\s+)?(?:i\s*\.?\s*i\s*\.?\s*d\s*\.?|"
    r"independent|identically\s+distributed|stationary|constant|fixed|unchanged|"
    r"the\s+same)\b|"
    r"\b(?:dependent|non[- ]?independent|non[- ]?stationary|non[- ]?iid)\b|"
    r"\b(?:changing|varying|different|unknown)\s+(?:success\s+)?(?:probability|rate)\b|"
    r"\b(?:probability|rate)\s+(?:is|may\s+be|can\s+be|could\s+be)?\s*"
    r"(?:changing|varying|different|unknown)\b|"
    r"\b(?:may|can|could)\s+change\b",
    re.IGNORECASE,
)
_REASONING_NEXT_TRIAL_RE = re.compile(r"\bnext\s+(?:trial|outcome)\b", re.IGNORECASE)
_REASONING_EMPIRICAL_COUNTS_RE = re.compile(
    r"\b\d{1,10}\s+success(?:es)?\s+(?:in|out\s+of)\s+"
    r"\d{1,10}\s+(?:bernoulli\s+)?trials?\b",
    re.IGNORECASE,
)
_REASONING_REQUEST_CUE_RE = re.compile(
    r"\b(?:what\s+(?:is|are)|what's|calculate|compute|derive|evaluate|find|solve|"
    r"determine|work\s+out|convert|give(?:\s+me)?|provide|return|tell\s+me|show(?:\s+me)?)\b",
    re.IGNORECASE,
)
_REASONING_NON_CALCULATION_RE = re.compile(
    r"\b(?:word|term|phrase|definition|meaning|concept|conceptual|quote|quoted|"
    r"text|string|occurrence)\b",
    re.IGNORECASE,
)
_REASONING_NEGATED_REQUEST_RE = re.compile(
    r"\b(?:do\s+not|don't|never|not|without)\b",
    re.IGNORECASE,
)
_REASONING_REQUEST_CANCELLATION_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+"
    r"(?:calculate|compute|solve|answer|evaluate|determine|work\s+out)\b",
    re.IGNORECASE,
)
_REASONING_LATE_CORRECTION_RE = re.compile(
    r"[.!?]\s*(?:actually\b|correction\b|no(?:\s|,)|rather\b|instead\b)",
    re.IGNORECASE,
)
_REASONING_UNTRUSTED_PROBLEM_DATA_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+use\b|"
    r"\bignore\s+(?:the\s+)?(?:previous|prior|quoted|following|example|data)\b|"
    r"\b(?:quoted\s+)?(?:text|data|example)\s+(?:is|are)\s+untrusted\b|"
    r"\bnot\s+(?:an?\s+)?instructions?\b",
    re.IGNORECASE,
)
_REASONING_EXCLUDED_SETUP_RE = re.compile(
    r"\b(?:ignore|discard|exclude|omit)\b|"
    r"\b(?:do\s+not|don't|never)\s+(?:consider|include|use|rely\s+on)\b|"
    r"\b(?:incorrect|invalid|untrusted|irrelevant|decoy|counterexample|fake|"
    r"hypothetical|alleged|blockquote|code\s+example|quoted\s+example|markdown\s+quote)\b|"
    r"\b(?:should|must)\s+not\s+count\b|"
    r"\b(?:neither|nor|without|except|excluding|other\s+than)\b|"
    r"\bset\s+aside\b|\bset\s+(?:it|them|these|those|the\s+values?)\s+aside\b|"
    r"\bdo\s+not\s+take\s+into\s+account\b|"
    r"\b(?:just|only)\s+an?\s+example\b|"
    r"\b(?:for|as)\s+comparison\s+only\b",
    re.IGNORECASE,
)
_REASONING_UNCONSUMED_ACTION_RE = re.compile(
    r"\b(?:translate|summari[sz]e|write|compose|describe|define|definition|discuss|"
    r"explain|list|create|draw|sing|haiku|poem|story|benefits?|useful|real[- ]world\s+use|"
    r"recommend|compare|analy[sz]e|critique|proofread|suggest|format|generate)\b",
    re.IGNORECASE,
)
_REASONING_COORDINATED_TAIL_RE = re.compile(
    r"(?:\band\b|\balso\b|\bthen\b|;)\s*"
    r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?P<next>[a-z]+|\d+)",
    re.IGNORECASE,
)
_REASONING_SAFE_COORDINATED_TASK_WORDS = frozenset(
    {
        "acceleration", "area", "base", "current", "day", "days", "diameter",
        "difference", "distance", "energy", "failure", "failures", "force",
        "height", "hour", "hours", "january", "february", "march", "april",
        "may", "june", "july", "august", "september", "october", "november",
        "december", "length", "mass", "minute", "minutes", "outcome", "outcomes",
        "perimeter", "probability", "radius", "rate", "resistance", "speed",
        "success", "successes", "sum", "tax", "time", "total", "trial", "trials",
        "value", "voltage", "volume", "width",
    }
)
_REASONING_ALLOWED_ACTION_RE = re.compile(
    r"\b(?:explain\s+(?:(?:your|the)\s+)?(?:reasoning|working|steps)|"
    r"show\s+(?:(?:your|the)\s+)?(?:work|working|steps)|"
    r"(?:verify|check)\s+(?:(?:the|your)\s+)?(?:answer|result|calculation|work))\b",
    re.IGNORECASE,
)
_REASONING_AUTHORITATIVE_QUOTED_INPUT_RE = re.compile(
    r"\b(?:use|treat)\s+(?:the|this)\s+(?:quoted|following)\s+"
    r"(?:problem|text|example|data)\s+(?:as\s+)?(?:the\s+)?"
    r"(?:authoritative|input|given\s+data)\b",
    re.IGNORECASE,
)
_REASONING_QUOTED_SPAN_RE = re.compile(
    r'''(?:"[^"\n]{0,2000}"|`[^`\n]{0,2000}`|'''
    r'''\u201c[^\u201d\n]{0,2000}\u201d|\u2018[^\u2019\n]{0,2000}\u2019|'''
    r'''\u00ab[^\u00bb\n]{0,2000}\u00bb|\u300c[^\u300d\n]{0,2000}\u300d|'''
    r'''\u300e[^\u300f\n]{0,2000}\u300f|'''
    r'''(?<![a-z0-9])'[^'\n]{0,2000}'(?![a-z0-9]))''',
    re.IGNORECASE,
)
_REASONING_QUOTE_DELIMITER_RE = re.compile(
    r'''["`\u201c\u201d\u2018\u00ab\u00bb\u300c\u300d\u300e\u300f]|'''
    r'''(?<![a-z0-9])['\u2019]|['\u2019](?![a-z0-9])''',
    re.IGNORECASE,
)
_REASONING_CLAUSE_RE = re.compile(r"[^.;?!\n]+")
_REASONING_HORN_SHAPE_RE = re.compile(
    r"^facts\s*:.*\brules\s*:.*\bquery\s*:",
    re.IGNORECASE,
)
_REASONING_POSITIVE_NUMBER = r"(?:\d+(?:\.\d+)?)"
_REASONING_BARE_IN_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s+in\b",
    re.IGNORECASE,
)
_REASONING_MASS_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:kilograms?|kg|milligrams?|mg|grams?|g)(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_VOLUME_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:cubic\s+meters?|m\s*\^?\s*3|m3|cubic\s+centimeters?|"
    r"cm\s*\^?\s*3|cm3|millilit(?:er|re)s?|ml|lit(?:er|re)s?|l)(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_ACCELERATION_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:m\s*/\s*s\s*(?:\^\s*2|2|\u00b2)|met(?:er|re)s?\s+per\s+second\s+squared)"
    r"(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_SPEED_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:m\s*/\s*s|met(?:er|re)s?\s+per\s+second|km\s*/\s*h|kph|kmh|"
    r"kilomet(?:er|re)s?\s+per\s+hour)(?!\s*(?:\^\s*2|2|\u00b2))(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_RESISTANCE_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:kiloohms?|kohms?|k\s*ohms?|ohms?)(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_CURRENT_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:milliamperes?|milliamps?|ma|amperes?|amps?|a)(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_VOLTAGE_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}\s*"
    r"(?:millivolts?|mv|volts?|v)(?![a-z0-9])",
    re.IGNORECASE,
)
_REASONING_NEWTON_CONTEXT_RE = re.compile(
    r"\b(?:net\s+force|newton(?:'s)?\s+second\s+law)\b|"
    r"\bf\s*=\s*m\s*(?:\*|x)?\s*a\b",
    re.IGNORECASE,
)
_REASONING_FORCE_CAVEAT_RE = re.compile(
    r"\b(?:friction(?:al)?|applied\s+force|tension|drag|air\s+resistance|"
    r"normal\s+force|weight|gravity|incline|slope|force\s+components?|"
    r"multiple\s+forces?|several\s+forces?|two\s+forces?|three\s+forces?)\b",
    re.IGNORECASE,
)
_REASONING_DENSITY_CAVEAT_RE = re.compile(
    r"\b(?:mixture|layered|composite|porous|non[- ]?uniform|variable\s+density|"
    r"relative\s+density|specific\s+gravity|buoyancy|multiple\s+(?:materials?|phases?))\b",
    re.IGNORECASE,
)
_REASONING_KINETIC_CAVEAT_RE = re.compile(
    r"\b(?:rotational|rolling|angular|relativistic|collision|system\s+of|"
    r"multiple\s+objects?|several\s+objects?)\b",
    re.IGNORECASE,
)
_REASONING_OHM_CONTEXT_RE = re.compile(
    r"\bohm(?:'s)?\s+law\b|\bv\s*=\s*i\s*(?:\*|x)?\s*r\b|"
    r"\bi\s*=\s*v\s*/\s*r\b|\br\s*=\s*v\s*/\s*i\b",
    re.IGNORECASE,
)
_REASONING_SIMPLE_RESISTOR_RE = re.compile(
    r"\b(?:(?:single|one|same)\s+(?:resistor|resistive\s+element|element)|"
    r"a\s+[^.,;?]{0,32}\b(?:ohms?)\s+(?:resistor|element))\b",
    re.IGNORECASE,
)
_REASONING_CIRCUIT_CAVEAT_RE = re.compile(
    r"\b(?:branches?|series|parallel|network|multiple|multi[- ]component|"
    r"equivalent\s+resistance|two\s+resistors?|three\s+resistors?|four\s+resistors?)\b",
    re.IGNORECASE,
)
_REASONING_GEOMETRY_METHODS = frozenset(
    {
        "rectangle_area",
        "rectangle_perimeter",
        "triangle_area",
        "circle_area",
        "circle_circumference",
        "pythagorean_hypotenuse",
        "pythagorean_missing_leg",
    }
)
_REASONING_PHYSICS_METHODS = frozenset(
    {
        "newtons_second_law_force",
        "density_mass_over_volume",
        "kinetic_energy",
        "ohms_law_voltage",
        "ohms_law_current",
        "ohms_law_resistance",
    }
)
_REASONING_SCIENCE_METHOD_TARGETS = {
    "constant_acceleration.final_velocity": (
        "constant_acceleration",
        "final_velocity",
    ),
    "constant_acceleration.displacement": (
        "constant_acceleration",
        "displacement",
    ),
    "ideal_gas.pressure": ("ideal_gas", "pressure"),
    "ideal_gas.volume": ("ideal_gas", "volume"),
    "ideal_gas.temperature": ("ideal_gas", "temperature"),
    "ideal_gas.amount": ("ideal_gas", "amount"),
}
_SCIENCE_PLAN_CHECK_KEYS = (
    "registry_integrity",
    "plan_integrity",
    "input_bindings",
    "dimensions",
    "domain",
    "substitution",
)
_SCIENCE_PLAN_AUTHORITY_KEYS = (
    "controls_compute",
    "controls_routes",
    "controls_interaction_strategy",
    "controls_tools",
    "controls_permissions",
    "controls_safety",
)
_SCIENCE_RECEIPT_AUTHORITY_KEYS = (
    *_SCIENCE_PLAN_AUTHORITY_KEYS,
    "controls_promotion",
)

_REASONING_RECEIPT_REASON_CATEGORIES = {
    "query_too_long": "unsupported_or_ambiguous",
    "empty_query": "not_applicable",
    "ambiguous_or_superseded_request": "unsupported_or_ambiguous",
    "untrusted_problem_data": "unsupported_or_ambiguous",
    "multiple_calculation_requests": "unsupported_or_ambiguous",
    "mixed_or_unconsumed_request": "unsupported_or_ambiguous",
    "no_quantities": "not_applicable",
    "no_applicable_solver": "unsupported_or_ambiguous",
    "engine_unavailable": "engine_unavailable",
    "engine_error": "engine_unavailable",
    "engine_bad_result": "unrecognized_result",
    "geometry_intent_not_established": "unsupported_or_ambiguous",
    "physics_applicability_not_established": "unsupported_or_ambiguous",
    "probability_assumptions_not_established": "assumptions_not_established",
    "repeated_trials_not_single_trial": "assumptions_not_established",
    "probability_event_not_established": "assumptions_not_established",
    "finite_bernoulli_model_not_established": "assumptions_not_established",
    "prediction_assumptions_not_established": "assumptions_not_established",
    "science_plan_not_established": "assumptions_not_established",
    "science_plan_result_mismatch": "verification_failed",
    "solver_consensus_incomplete": "verification_failed",
    "reasoning_result_mismatch": "verification_failed",
    "high_stakes_override_suppressed": "high_stakes_suppressed",
    "unverified_solution": "verification_failed",
    "verified_non_overriding_estimate": "model_conditional",
    "verified_conflict": "conflict",
    "verified_solution": "verified",
}
_ARITHMETIC_RECEIPT_REASON_CATEGORIES = {
    "not_explicit_arithmetic": "not_applicable",
    "solved_exactly": "verified",
    "invalid_syntax": "unsupported_or_ambiguous",
    "arithmetic_error": "verification_failed",
    "division_by_zero": "unsupported_or_ambiguous",
    "exponent_too_large": "bounded_limit",
    "expression_too_deep": "bounded_limit",
    "fractional_exponent_not_supported": "unsupported_or_ambiguous",
    "literal_too_large": "bounded_limit",
    "non_finite_literal": "unsupported_or_ambiguous",
    "result_too_large": "bounded_limit",
    "too_many_nodes": "bounded_limit",
    "too_many_operations": "bounded_limit",
    "unsupported_literal": "unsupported_or_ambiguous",
    "unsupported_operator": "unsupported_or_ambiguous",
    "unsupported_syntax": "unsupported_or_ambiguous",
}
_RECEIPT_SELECTION_REASON_BY_GUARD = {
    "explicit_arithmetic_exact": "exact_arithmetic",
    "verified_reasoning_solution": "verified_reasoning",
    "verified_model_conditional_estimate": "model_conditional_estimate",
    "high_stakes_suppressed": "high_stakes_suppressed",
    "strict_evidence_conflicting": "strict_evidence_precedence",
    "strict_evidence_no_evidence": "strict_evidence_precedence",
    "strict_evidence_insufficient": "strict_evidence_precedence",
}

_PROMPT_UNDERSTANDING_MODULE: Any = None
_REASONING_MODULE: Any = None
_TRUSTED_REASONING_MODULE: Any = None
_TRUSTED_SCIENCE_PLAN_MODULE: Any = None


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


def _reasoning_validator(name: str, query: Any, *args: Any) -> Any:
    """Call a current sibling admission validator; absence fails closed."""

    module = _load_reasoning_module()
    validator = getattr(module, name, None) if module is not None else None
    if not callable(validator):
        return None
    try:
        return validator(query, *args)
    except Exception:
        return None


def _load_trusted_reasoning_module() -> Any:
    """Load the local reasoning implementation outside replaceable wrappers."""

    global _TRUSTED_REASONING_MODULE
    if _TRUSTED_REASONING_MODULE is not None:
        return _TRUSTED_REASONING_MODULE
    module_path = Path(__file__).with_name("reasoning_engine.py")
    module_name = f"_supermix_{module_path.parent.name}_trusted_reasoning_engine"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    _TRUSTED_REASONING_MODULE = module
    return module


def _load_trusted_science_plan_module() -> Any:
    """Load the local closed-world science-plan implementation fail-closed."""

    global _TRUSTED_SCIENCE_PLAN_MODULE
    if _TRUSTED_SCIENCE_PLAN_MODULE is not None:
        return _TRUSTED_SCIENCE_PLAN_MODULE
    module_path = Path(__file__).with_name("science_plan.py")
    module_name = f"_supermix_{module_path.parent.name}_trusted_science_plan"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(module_name, module)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None
    _TRUSTED_SCIENCE_PLAN_MODULE = module
    return module


def _trusted_reasoning_result(query: Any) -> Optional[Dict[str, Any]]:
    """Freshly recompute one prompt with the packaged canonical implementation."""

    module = _load_trusted_reasoning_module()
    solver = getattr(module, "solve_problem", None) if module is not None else None
    if not callable(solver):
        return None
    try:
        result = solver(str(query or ""), tier="auto")
    except Exception:
        return None
    return dict(result) if isinstance(result, Mapping) else None


def _science_plan_contract() -> Optional[Dict[str, str]]:
    module = _load_trusted_science_plan_module()
    if module is None:
        return None
    names = (
        "SCIENCE_PLAN_SCHEMA_VERSION",
        "SCIENCE_PLAN_ENGINE_VERSION",
        "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION",
        "SCIENCE_FORMULA_REGISTRY_VERSION",
        "SCIENCE_FORMULA_REGISTRY_SHA256",
    )
    values = {name: getattr(module, name, None) for name in names}
    if any(not isinstance(value, str) or not value for value in values.values()):
        return None
    registry_sha = values["SCIENCE_FORMULA_REGISTRY_SHA256"]
    if re.fullmatch(r"[0-9a-f]{64}", registry_sha) is None:
        return None
    return values


def _science_authority_is_false(value: Any, keys: Sequence[str]) -> bool:
    return bool(
        isinstance(value, Mapping)
        and all(value.get(key) is False for key in keys)
    )


def _science_count(value: Any, limit: int) -> Optional[int]:
    if type(value) is not int or value < 0 or value > limit:
        return None
    return value


def _science_hash(value: Any) -> str:
    text = value if isinstance(value, str) else ""
    return text if re.fullmatch(r"[0-9a-f]{64}", text) is not None else ""


def _science_summary_from_claim(
    diagnostics: Any,
    receipt: Any,
    expected_formula: str,
) -> Optional[Dict[str, Any]]:
    """Admit only the integrity-bound, prompt-free science-plan contract."""

    if expected_formula not in _REASONING_SCIENCE_METHOD_TARGETS:
        return None
    if not isinstance(diagnostics, Mapping) or not isinstance(receipt, Mapping):
        return None
    contract = _science_plan_contract()
    if contract is None:
        return None
    scenario, target = _REASONING_SCIENCE_METHOD_TARGETS[expected_formula]
    if not (
        diagnostics.get("schema_version") == contract["SCIENCE_PLAN_SCHEMA_VERSION"]
        and diagnostics.get("engine_version") == contract["SCIENCE_PLAN_ENGINE_VERSION"]
        and diagnostics.get("registry_version")
        == contract["SCIENCE_FORMULA_REGISTRY_VERSION"]
        and diagnostics.get("registry_sha256")
        == contract["SCIENCE_FORMULA_REGISTRY_SHA256"]
        and diagnostics.get("scenario") == scenario
        and diagnostics.get("target") == target
        and diagnostics.get("formula_id") == expected_formula
        and diagnostics.get("reason") == "verified_science_plan"
        and diagnostics.get("attempted") is True
        and diagnostics.get("solved") is True
        and diagnostics.get("override_allowed") is True
        and diagnostics.get("verification_passed") is True
        and diagnostics.get("model_conditional") is True
        and diagnostics.get("assumptions_explicit") is True
        and diagnostics.get("calibration_claimed") is False
        and _science_authority_is_false(
            diagnostics.get("authority"), _SCIENCE_PLAN_AUTHORITY_KEYS
        )
    ):
        return None
    quantities = _science_count(diagnostics.get("quantities"), 8)
    steps = _science_count(diagnostics.get("steps"), 4)
    checks = receipt.get("checks")
    epistemics = receipt.get("epistemics")
    formula_ids = receipt.get("formula_ids")
    query_sha256 = _science_hash(receipt.get("query_sha256"))
    plan_sha256 = _science_hash(receipt.get("plan_sha256"))
    if not (
        quantities is not None
        and steps is not None
        and receipt.get("schema_version")
        == contract["SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION"]
        and receipt.get("decision") == "verified"
        and receipt.get("scenario") == scenario
        and receipt.get("target") == target
        and isinstance(formula_ids, list)
        and formula_ids == [expected_formula]
        and receipt.get("registry_version")
        == contract["SCIENCE_FORMULA_REGISTRY_VERSION"]
        and receipt.get("registry_sha256")
        == contract["SCIENCE_FORMULA_REGISTRY_SHA256"]
        and bool(query_sha256)
        and bool(plan_sha256)
        and isinstance(checks, Mapping)
        and set(checks) == set(_SCIENCE_PLAN_CHECK_KEYS)
        and all(checks.get(key) is True for key in _SCIENCE_PLAN_CHECK_KEYS)
        and isinstance(epistemics, Mapping)
        and epistemics.get("model_conditional") is True
        and epistemics.get("assumptions_explicit") is True
        and epistemics.get("calibration_claimed") is False
        and receipt.get("diagnostic_only") is True
        and _science_authority_is_false(
            receipt.get("authority"), _SCIENCE_PLAN_AUTHORITY_KEYS
        )
    ):
        return None
    return {
        "present": True,
        "schema_version": contract["SCIENCE_PLAN_SCHEMA_VERSION"],
        "engine_version": contract["SCIENCE_PLAN_ENGINE_VERSION"],
        "receipt_schema_version": contract["SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION"],
        "registry_version": contract["SCIENCE_FORMULA_REGISTRY_VERSION"],
        "registry_sha256": contract["SCIENCE_FORMULA_REGISTRY_SHA256"],
        "scenario": scenario,
        "target": target,
        "formula_id": expected_formula,
        "query_sha256": query_sha256,
        "plan_sha256": plan_sha256,
        "checks": {key: True for key in _SCIENCE_PLAN_CHECK_KEYS},
        "counts": {"quantities": quantities, "steps": steps},
        "verification": {
            "passed": True,
            "independent": bool(diagnostics.get("verification_independent", False)),
        },
        "epistemics": {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        },
        "diagnostic_only": True,
        "authority": {key: False for key in _SCIENCE_RECEIPT_AUTHORITY_KEYS},
    }


def _trusted_science_summary(query: Any, expected_formula: str) -> Optional[Dict[str, Any]]:
    """Reparse the complete raw prompt and execute its registry-bound plan."""

    module = _load_trusted_science_plan_module()
    parser = getattr(module, "parse_science_scenario", None) if module is not None else None
    solver = getattr(module, "solve_science_scenario", None) if module is not None else None
    diagnostics_fn = getattr(module, "science_plan_diagnostics", None) if module is not None else None
    if not (callable(parser) and callable(solver) and callable(diagnostics_fn)):
        return None
    raw_query = str(query or "")
    canonical_query = _canonical_science_query(raw_query)
    try:
        plan = parser(canonical_query)
        result = solver(canonical_query)
        diagnostics = diagnostics_fn(result)
    except Exception:
        return None
    if not isinstance(plan, Mapping) or not isinstance(result, Mapping):
        return None
    summary = _science_summary_from_claim(
        diagnostics,
        result.get("receipt"),
        expected_formula,
    )
    if summary is None:
        return None
    expected_assumption = (
        "constant_acceleration"
        if summary["scenario"] == "constant_acceleration"
        else "ideal_gas"
    )
    steps = plan.get("steps")
    if not (
        plan.get("schema_version") == summary["schema_version"]
        and plan.get("registry_version") == summary["registry_version"]
        and plan.get("registry_sha256") == summary["registry_sha256"]
        and plan.get("scenario") == summary["scenario"]
        and plan.get("target") == summary["target"]
        and plan.get("assumptions") == [expected_assumption]
        and isinstance(steps, list)
        and len(steps) == 1
        and isinstance(steps[0], Mapping)
        and steps[0].get("formula_id") == expected_formula
        and plan.get("query_sha256") == summary["query_sha256"]
        and plan.get("plan_sha256") == summary["plan_sha256"]
        and summary["query_sha256"]
        == hashlib.sha256(canonical_query.encode("utf-8")).hexdigest()
    ):
        return None
    return summary


def _science_summary_is_safe(value: Any, expected_formula: str) -> bool:
    """Validate an already-sanitized summary before emitting it in a receipt."""

    if not isinstance(value, Mapping):
        return False
    contract = _science_plan_contract()
    if contract is None or expected_formula not in _REASONING_SCIENCE_METHOD_TARGETS:
        return False
    scenario, target = _REASONING_SCIENCE_METHOD_TARGETS[expected_formula]
    checks = value.get("checks")
    counts = value.get("counts")
    verification = value.get("verification")
    epistemics = value.get("epistemics")
    expected_keys = {
        "present",
        "schema_version",
        "engine_version",
        "receipt_schema_version",
        "registry_version",
        "registry_sha256",
        "scenario",
        "target",
        "formula_id",
        "query_sha256",
        "plan_sha256",
        "checks",
        "counts",
        "verification",
        "epistemics",
        "diagnostic_only",
        "authority",
    }
    return bool(
        set(value) == expected_keys
        and value.get("present") is True
        and value.get("schema_version") == contract["SCIENCE_PLAN_SCHEMA_VERSION"]
        and value.get("engine_version") == contract["SCIENCE_PLAN_ENGINE_VERSION"]
        and value.get("receipt_schema_version")
        == contract["SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION"]
        and value.get("registry_version")
        == contract["SCIENCE_FORMULA_REGISTRY_VERSION"]
        and value.get("registry_sha256")
        == contract["SCIENCE_FORMULA_REGISTRY_SHA256"]
        and value.get("scenario") == scenario
        and value.get("target") == target
        and value.get("formula_id") == expected_formula
        and bool(_science_hash(value.get("query_sha256")))
        and bool(_science_hash(value.get("plan_sha256")))
        and isinstance(checks, Mapping)
        and set(checks) == set(_SCIENCE_PLAN_CHECK_KEYS)
        and all(checks.get(key) is True for key in _SCIENCE_PLAN_CHECK_KEYS)
        and isinstance(counts, Mapping)
        and set(counts) == {"quantities", "steps"}
        and _science_count(counts.get("quantities"), 8) is not None
        and _science_count(counts.get("steps"), 4) is not None
        and isinstance(verification, Mapping)
        and set(verification) == {"passed", "independent"}
        and verification.get("passed") is True
        and type(verification.get("independent")) is bool
        and isinstance(epistemics, Mapping)
        and epistemics
        == {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        }
        and value.get("diagnostic_only") is True
        and _science_authority_is_false(
            value.get("authority"), _SCIENCE_RECEIPT_AUTHORITY_KEYS
        )
    )


def _empty_science_plan_receipt() -> Dict[str, Any]:
    return {
        "present": False,
        "schema_version": "",
        "engine_version": "",
        "receipt_schema_version": "",
        "registry_version": "",
        "registry_sha256": "",
        "scenario": "",
        "target": "",
        "formula_id": "",
        "checks": {key: False for key in _SCIENCE_PLAN_CHECK_KEYS},
        "counts": {"quantities": 0, "steps": 0},
        "verification": {"passed": False, "independent": False},
        "epistemics": {
            "model_conditional": False,
            "assumptions_explicit": False,
            "calibration_claimed": False,
        },
        "diagnostic_only": True,
        "authority": {key: False for key in _SCIENCE_RECEIPT_AUTHORITY_KEYS},
    }


def _science_plan_receipt_section(
    reasoning: Mapping[str, Any],
    problem_class: str,
    method: str,
) -> Dict[str, Any]:
    if (
        problem_class != "scientific_scenario"
        or method not in _REASONING_SCIENCE_METHOD_TARGETS
    ):
        return _empty_science_plan_receipt()
    value = reasoning.get("science_plan")
    if not _science_summary_is_safe(value, method):
        return _empty_science_plan_receipt()
    assert isinstance(value, Mapping)
    checks = value["checks"]
    counts = value["counts"]
    verification = value["verification"]
    return {
        "present": True,
        "schema_version": str(value["schema_version"]),
        "engine_version": str(value["engine_version"]),
        "receipt_schema_version": str(value["receipt_schema_version"]),
        "registry_version": str(value["registry_version"]),
        "registry_sha256": str(value["registry_sha256"]),
        "scenario": str(value["scenario"]),
        "target": str(value["target"]),
        "formula_id": method,
        "checks": {key: bool(checks[key]) for key in _SCIENCE_PLAN_CHECK_KEYS},
        "counts": {
            "quantities": int(counts["quantities"]),
            "steps": int(counts["steps"]),
        },
        "verification": {
            "passed": bool(verification["passed"]),
            "independent": bool(verification["independent"]),
        },
        "epistemics": {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        },
        "diagnostic_only": True,
        "authority": {key: False for key in _SCIENCE_RECEIPT_AUTHORITY_KEYS},
    }


def _reasoning_result_matches_trusted(
    claimed: Mapping[str, Any],
    trusted: Mapping[str, Any],
) -> bool:
    """Bind hard-override authority to a fresh method/class/answer recompute."""

    trusted_answer = trusted.get("answer")
    claimed_answer = claimed.get("answer")
    if not isinstance(trusted_answer, Mapping) or not isinstance(claimed_answer, Mapping):
        return False
    return bool(
        claimed.get("schema_version") == trusted.get("schema_version")
        and claimed.get("engine_version") == trusted.get("engine_version")
        and claimed.get("problem_class") == trusted.get("problem_class")
        and claimed.get("method") == trusted.get("method")
        and claimed_answer.get("exact") == trusted_answer.get("exact")
        and claimed_answer.get("display") == trusted_answer.get("display")
        and claimed_answer.get("unit") == trusted_answer.get("unit")
        and trusted.get("solved") is True
        and trusted.get("override_allowed") is True
        and _reasoning_consensus_complete(trusted)
    )


def _public_reasoning_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove prompt-derived bindings from the externally returned diagnostics."""

    published = dict(result)
    for key in ("science_plan", "science_plan_receipt"):
        value = published.get(key)
        if not isinstance(value, Mapping):
            continue
        redacted = dict(value)
        redacted.pop("query_sha256", None)
        redacted.pop("plan_sha256", None)
        published[key] = redacted
    return published


def _reasoning_estimate_matches_trusted(
    claimed: Mapping[str, Any],
    trusted: Mapping[str, Any],
) -> bool:
    """Bind a selected estimate to a fresh canonical non-overriding result."""

    trusted_answer = trusted.get("answer")
    claimed_answer = claimed.get("answer")
    verification = trusted.get("verification")
    return bool(
        isinstance(trusted_answer, Mapping)
        and isinstance(claimed_answer, Mapping)
        and isinstance(verification, Mapping)
        and claimed.get("schema_version") == trusted.get("schema_version")
        and claimed.get("engine_version") == trusted.get("engine_version")
        and claimed.get("problem_class") == trusted.get("problem_class") == "prediction"
        and claimed.get("method") == trusted.get("method") == "empirical_bernoulli_plugin"
        and claimed_answer.get("exact") == trusted_answer.get("exact")
        and claimed_answer.get("display") == trusted_answer.get("display")
        and claimed_answer.get("unit") == trusted_answer.get("unit")
        and trusted.get("solved") is True
        and trusted.get("override_allowed") is False
        and verification.get("checked") is True
        and verification.get("passed") is True
        and _reasoning_consensus_complete(trusted)
    )


def solve_reasoned_problem(query: Any, tier: str = "auto") -> Dict[str, Any]:
    """Run the deliberate reasoning engine, degrading to an empty attempt."""

    module = _load_reasoning_module()
    if module is None:
        return {"attempted": False, "solved": False, "override_allowed": False, "reason": "engine_unavailable"}
    try:
        result = module.solve_problem(str(query or ""), tier=tier)
    except Exception:  # pragma: no cover - defensive: never break a chat turn
        return {"attempted": False, "solved": False, "override_allowed": False, "reason": "engine_error"}
    cooked = dict(result) if isinstance(result, Mapping) else {
        "attempted": False,
        "solved": False,
        "override_allowed": False,
        "reason": "engine_bad_result",
    }
    return _ground_reasoning_result(query, cooked)


def _trusted_reasoning_diagnostics(result: Any) -> Dict[str, Any]:
    """Sanitize a result with the canonical, allowlisted diagnostics contract."""

    module = _load_trusted_reasoning_module()
    diagnostics_fn = getattr(module, "reasoning_diagnostics", None) if module is not None else None
    if not callable(diagnostics_fn):
        return {}
    try:
        diagnostics = diagnostics_fn(result if isinstance(result, Mapping) else None)
    except Exception:
        return {}
    return dict(diagnostics) if isinstance(diagnostics, Mapping) else {}


def _receipt_count(value: Any) -> int:
    if type(value) is not int:
        return 0
    return max(0, min(64, value))


def build_verified_answer_receipt(
    reasoning_result: Any,
    arithmetic_result: Any = None,
    *,
    response_guard_reason: Any = "",
) -> Dict[str, Any]:
    """Build prompt-free, answer-free diagnostics for a deterministic selection.

    The receipt is observational metadata only. It never contains the prompt,
    computed answer, expression, proof steps, or evidence, and every string
    derived from a result is admitted through a fixed allowlist.
    """

    reasoning = reasoning_result if isinstance(reasoning_result, Mapping) else {}
    arithmetic = arithmetic_result if isinstance(arithmetic_result, Mapping) else {}
    diagnostics = _trusted_reasoning_diagnostics(reasoning)
    raw_reason = str(reasoning.get("reason") or "")
    arithmetic_reason = str(arithmetic.get("reason") or "")
    guard_reason = str(response_guard_reason or "")
    selection_reason = _RECEIPT_SELECTION_REASON_BY_GUARD.get(
        guard_reason,
        "not_selected",
    )

    reasoning_attempted = bool(diagnostics.get("attempted"))
    reasoning_claimed_solved = bool(diagnostics.get("solved"))
    problem_class = str(diagnostics.get("problem_class") or "")
    method = str(diagnostics.get("method") or "")
    recognized_reasoning_result = bool(problem_class and method)
    reasoning_solved = bool(reasoning_claimed_solved and recognized_reasoning_result)
    reasoning_verified = bool(
        diagnostics.get("verified") and reasoning_solved
    )
    reasoning_signal = bool(
        reasoning_attempted
        or reasoning_solved
        or raw_reason
        not in {"", "empty_query", "no_quantities"}
    )
    arithmetic_attempted = bool(arithmetic.get("attempted"))
    arithmetic_solved = bool(arithmetic.get("solved"))

    if selection_reason == "exact_arithmetic" and arithmetic_solved:
        kind = "exact_arithmetic"
    elif selection_reason == "verified_reasoning" and reasoning_solved:
        kind = "deliberate_reasoning"
    elif reasoning_solved:
        kind = "deliberate_reasoning"
    elif arithmetic_solved:
        kind = "exact_arithmetic"
    elif reasoning_signal:
        kind = "deliberate_reasoning"
    elif arithmetic_attempted:
        kind = "exact_arithmetic"
    else:
        kind = "none"

    if kind == "exact_arithmetic":
        attempted = arithmetic_attempted
        solved = arithmetic_solved
        verified = arithmetic_solved
        independent = False
        receipt_problem_class = "arithmetic"
        receipt_method = "bounded_exact_arithmetic"
        reason_code = (
            arithmetic_reason
            if arithmetic_reason in _ARITHMETIC_RECEIPT_REASON_CATEGORIES
            else ""
        )
        reason_category = _ARITHMETIC_RECEIPT_REASON_CATEGORIES.get(
            reason_code,
            "unrecognized_result" if attempted else "not_applicable",
        )
        model_conditional = False
        assumptions_explicit = False
        conflicting = False
        consensus_paths = 0
        reasoning_schema_version = ""
        reasoning_engine_version = ""
    elif kind == "deliberate_reasoning":
        attempted = reasoning_attempted
        solved = reasoning_solved
        verified = reasoning_verified
        independent = bool(
            verified and diagnostics.get("verification_independent")
        )
        receipt_problem_class = problem_class if recognized_reasoning_result else ""
        receipt_method = method if recognized_reasoning_result else ""
        reason_code = (
            raw_reason if raw_reason in _REASONING_RECEIPT_REASON_CATEGORIES else ""
        )
        reason_category = _REASONING_RECEIPT_REASON_CATEGORIES.get(
            reason_code,
            "unrecognized_result" if reasoning_signal else "not_applicable",
        )
        model_conditional = bool(
            recognized_reasoning_result and diagnostics.get("model_conditional")
        )
        assumptions_explicit = bool(
            recognized_reasoning_result and diagnostics.get("assumptions_explicit")
        )
        conflicting = bool(
            recognized_reasoning_result and diagnostics.get("conflicting")
        )
        consensus_paths = (
            _receipt_count(diagnostics.get("paths"))
            if recognized_reasoning_result
            else 0
        )
        reasoning_schema_version = str(diagnostics.get("schema_version") or "")
        reasoning_engine_version = str(diagnostics.get("engine_version") or "")
    else:
        attempted = False
        solved = False
        verified = False
        independent = False
        receipt_problem_class = ""
        receipt_method = ""
        reason_code = ""
        reason_category = "not_applicable"
        model_conditional = False
        assumptions_explicit = False
        conflicting = False
        consensus_paths = 0
        reasoning_schema_version = ""
        reasoning_engine_version = ""

    if selection_reason in {
        "high_stakes_suppressed",
        "strict_evidence_precedence",
    }:
        reason_category = selection_reason

    selected = bool(
        verified
        and (
            (selection_reason == "exact_arithmetic" and kind == "exact_arithmetic")
            or (selection_reason == "verified_reasoning" and kind == "deliberate_reasoning")
            or (
                selection_reason == "model_conditional_estimate"
                and kind == "deliberate_reasoning"
                and model_conditional
            )
        )
    )
    if selected and selection_reason == "model_conditional_estimate":
        decision = "verified_estimate_selected"
    elif selected and verified:
        decision = "verified_selected"
    elif verified:
        decision = "verified_not_selected"
    elif solved:
        decision = "unverified_solution"
    elif attempted or reasoning_signal:
        decision = "abstained"
    else:
        decision = "not_attempted"
    science_plan = _science_plan_receipt_section(
        reasoning,
        receipt_problem_class,
        receipt_method,
    )

    return {
        "schema_version": VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION,
        "runtime_version": GROUNDING_RUNTIME_VERSION,
        "kind": kind,
        "decision": decision,
        "selection_reason": selection_reason,
        "selected": selected,
        "attempted": bool(attempted),
        "solved": bool(solved),
        "problem_class": receipt_problem_class,
        "method": receipt_method,
        "reason_code": reason_code,
        "reason_category": reason_category,
        "reasoning_schema_version": reasoning_schema_version,
        "reasoning_engine_version": reasoning_engine_version,
        "verification": {
            "passed": bool(verified),
            "independent": bool(independent),
        },
        "epistemics": {
            "model_conditional": bool(model_conditional),
            "assumptions_explicit": bool(assumptions_explicit),
            "calibration_claimed": False,
        },
        "consensus": {
            "conflicting": bool(conflicting),
            "paths": consensus_paths,
        },
        "science_plan": science_plan,
        "diagnostic_only": True,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
            "controls_tools": False,
            "controls_permissions": False,
            "controls_safety": False,
            "controls_promotion": False,
        },
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


def _profile_high_stakes(profile: Mapping[str, Any]) -> bool:
    """Interpret only explicit high-stakes facets from the prompt profile."""

    return _profile_bool(profile, "knowledge", "high_stakes") or _profile_bool(
        profile,
        "safety",
        "high_stakes",
        "personal_crisis_signal",
        "urgent_health_signal",
    )


def _clean_text(value: Any, limit: int) -> str:
    cooked = re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", str(value or "")),
    ).strip()
    return cooked[: max(0, int(limit))]


def _canonical_science_query(value: Any) -> str:
    """Match the reasoning frame while retaining unit-significant superscripts."""

    cooked = unicodedata.normalize("NFC", str(value or "")).strip()
    return cooked[:MAX_REASONING_QUERY_CHARS]


def _json_safe_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        cooked = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return cooked if math.isfinite(cooked) else float(default)


def _reject_reasoning_result(
    result: Mapping[str, Any],
    reason: str,
    *,
    assumptions_explicit: Optional[bool] = None,
) -> Dict[str, Any]:
    """Remove rewrite authority from a result that fails a raw-prompt gate."""

    guarded = dict(result)
    guarded.update(
        {
            "attempted": bool(guarded.get("attempted", True)),
            "solved": False,
            "override_allowed": False,
            "reason": str(reason),
            "answer": {
                "exact": "",
                "display": "",
                "approximation": "",
                "approximate": False,
                "unit": "",
            },
            "text": "",
            "steps": [],
        }
    )
    guarded["verification"] = {
        "checked": True,
        "passed": False,
        "method": f"grounding_gate:{reason}",
        "independent": True,
    }
    # A rejected caller-supplied result must not retain plan traces or receipt
    # extensions. Empty mappings preserve the reasoning schema without echoing
    # any untrusted nested content.
    guarded["science_plan"] = {}
    guarded["science_plan_receipt"] = {}
    epistemics = dict(guarded.get("epistemics") or {})
    epistemics.setdefault("model_conditional", False)
    epistemics.setdefault("calibration_claimed", False)
    if assumptions_explicit is not None:
        epistemics["assumptions_explicit"] = bool(assumptions_explicit)
    else:
        epistemics.setdefault("assumptions_explicit", False)
    guarded["epistemics"] = epistemics
    return guarded


def _positive_prediction_assumptions(query: Any) -> bool:
    """Require one explicit, positive IID or independence+stationarity clause."""

    text = re.sub(
        r"\bi\s*\.\s*i\s*\.\s*d\s*\.?",
        "iid",
        _clean_text(query, MAX_QUERY_CHARS),
        flags=re.IGNORECASE,
    )
    if _REASONING_NEGATED_ASSUMPTION_RE.search(text) is not None:
        return False
    clauses = [
        match.group("body").strip()
        for match in _REASONING_ASSUMPTION_CLAUSE_RE.finditer(text)
    ]
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    if _REASONING_NEGATED_ASSUMPTION_RE.search(clause) is not None:
        return False
    if _REASONING_IID_RE.search(clause) is not None:
        return True
    return (
        _REASONING_INDEPENDENCE_RE.search(clause) is not None
        and _REASONING_STATIONARITY_RE.search(clause) is not None
    )


def _is_empirical_prediction_result(query: Any, result: Mapping[str, Any]) -> bool:
    method = str(result.get("method") or "")
    if method == "empirical_bernoulli_plugin":
        return True
    return (
        str(result.get("problem_class") or "") == "prediction"
        and _REASONING_NEXT_TRIAL_RE.search(_clean_text(query, MAX_QUERY_CHARS)) is not None
        and _REASONING_EMPIRICAL_COUNTS_RE.search(_clean_text(query, MAX_QUERY_CHARS)) is not None
    )


def _bounded_fraction_decimal(value: Fraction, places: int = 6) -> str:
    with localcontext() as context:
        context.prec = max(24, places + 12)
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    return format(decimal, f".{max(0, int(places))}f").rstrip("0").rstrip(".") or "0"


def _correct_odd_sided_die_parity(
    query: Any,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    """Independently recount odd/even faces on one explicitly fair odd-sided die."""

    text = _clean_text(query, MAX_QUERY_CHARS)
    die_matches = list(_REASONING_FAIR_DIE_RE.finditer(text))
    parity_matches = list(_REASONING_PARITY_RE.finditer(text))
    if len(die_matches) != 1 or len(parity_matches) != 1:
        return dict(result)
    sides_token = die_matches[0].group("sides")
    sides = int(sides_token) if sides_token is not None else 6
    if sides < 3 or sides > 999 or sides % 2 == 0:
        return dict(result)

    parity = parity_matches[0].group("parity").lower()
    favourable = (sides + 1) // 2 if parity == "odd" else sides // 2
    value = Fraction(favourable, sides)
    exact = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    terminating = _terminating_decimal(value)
    display = terminating if terminating is not None else _bounded_fraction_decimal(value)
    percent = _bounded_fraction_decimal(value * 100)

    guarded = dict(result)
    answer = dict(guarded.get("answer") or {})
    answer.update(
        {
            "exact": exact,
            "display": display,
            "approximation": "" if terminating is not None else display,
            "approximate": terminating is None,
            "unit": "",
        }
    )
    guarded["answer"] = answer
    guarded["text"] = f"The probability is {exact} ({percent}%)."
    guarded["steps"] = [
        f"A fair {sides}-sided die has {sides} equiprobable faces.",
        f"There are {favourable} {parity} faces from 1 through {sides}.",
    ]
    guarded["verification"] = {
        "checked": True,
        "passed": True,
        "method": "grounding_odd_sided_die_face_recount",
        "independent": True,
    }
    # Preserve a conflict or other denial from the reasoning engine. The
    # recount corrects values; it does not manufacture override permission.
    guarded["override_allowed"] = bool(guarded.get("override_allowed"))
    if guarded["override_allowed"]:
        guarded["reason"] = "verified_solution"
    return guarded


def _reasoning_request_window(text: str, target: str) -> str:
    """Return a target request clause plus at most one adjacent setup clause."""

    clauses = list(_REASONING_CLAUSE_RE.finditer(text))
    target_re = re.compile(rf"\b(?:{target})\b", re.IGNORECASE)
    for index, clause_match in enumerate(clauses):
        clause = clause_match.group(0)
        for cue in _REASONING_REQUEST_CUE_RE.finditer(clause):
            target_match = target_re.search(clause, cue.end())
            if target_match is None:
                continue
            intent = clause[: target_match.end()]
            if (
                _REASONING_NEGATED_REQUEST_RE.search(clause) is not None
                or _REASONING_NON_CALCULATION_RE.search(intent) is not None
            ):
                continue
            start = clauses[max(0, index - 1)].start()
            return text[start : clause_match.end()].strip()
    return ""


def _reasoning_has_unconsumed_trailing_content(text: str) -> bool:
    quoted_spans = [
        (match.start(), match.end())
        for match in _REASONING_QUOTED_SPAN_RE.finditer(text)
    ]
    request = next(
        (
            cue
            for cue in _REASONING_REQUEST_CUE_RE.finditer(text)
            if not any(start <= cue.start() < end for start, end in quoted_spans)
        ),
        None,
    )
    if request is None:
        return False
    boundary = next(
        (
            match.end()
            for match in re.finditer(r"[;?!]|\.(?=\s|$)", text[request.end() :])
            if not any(
                start <= request.end() + match.start() < end
                for start, end in quoted_spans
            )
        ),
        None,
    )
    if boundary is None:
        return False
    absolute_boundary = request.end() + boundary
    trailing = _REASONING_ALLOWED_ACTION_RE.sub(" ", text[absolute_boundary:])
    trailing = re.sub(r"\b(?:and|please|then)\b", " ", trailing, flags=re.IGNORECASE)
    return re.search(r"[a-z0-9]", trailing, re.IGNORECASE) is not None


def _reasoning_has_unconsumed_action(text: str) -> bool:
    if _REASONING_HORN_SHAPE_RE.search(text) is not None:
        return False
    action_text = _REASONING_ALLOWED_ACTION_RE.sub(" ", text)
    if _REASONING_UNCONSUMED_ACTION_RE.search(action_text) is not None:
        return True
    request = _REASONING_REQUEST_CUE_RE.search(action_text)
    if request is None:
        return False
    for match in _REASONING_COORDINATED_TAIL_RE.finditer(action_text):
        if match.start() < request.start():
            continue
        following = match.group("next").lower()
        if following.isdigit() or following in _REASONING_SAFE_COORDINATED_TASK_WORDS:
            continue
        return True
    return False


def _single_reasoning_quantity(pattern: re.Pattern[str], text: str) -> bool:
    return len(list(pattern.finditer(text))) == 1


def _has_labeled_geometry_measure(text: str, labels: str) -> bool:
    unit = (
        r"(?:kilometers?|kilometres?|km|meters?|metres?|m|centimeters?|centimetres?|cm|"
        r"millimeters?|millimetres?|mm|miles?|mi|yards?|yd|feet|foot|ft|inches?|inch)"
    )
    connector = r"(?:\s+(?:is|are|of|equals?))?\s*(?:=|:)?\s*"
    forward = re.compile(
        rf"\b(?:{labels})\b{connector}{_REASONING_POSITIVE_NUMBER}"
        rf"(?:\s*{unit})?(?![a-z0-9])",
        re.IGNORECASE,
    )
    reverse = re.compile(
        rf"(?<![a-z0-9.]){_REASONING_POSITIVE_NUMBER}(?:\s*{unit})?\s+"
        rf"\b(?:{labels})\b",
        re.IGNORECASE,
    )
    return forward.search(text) is not None or reverse.search(text) is not None


def _geometry_intent_established(text: str, method: str) -> bool:
    if method not in _REASONING_GEOMETRY_METHODS:
        return True
    # Bare ``in`` is ambiguous between an inch abbreviation and a
    # preposition. Only the unambiguous words ``inch``/``inches`` are accepted
    # for hard-override authority at this boundary.
    if _REASONING_BARE_IN_RE.search(text) is not None:
        return False

    target_by_method = {
        "rectangle_area": "area",
        "rectangle_perimeter": "perimeter",
        "triangle_area": "area",
        "circle_area": "area",
        "circle_circumference": "circumference",
        "pythagorean_hypotenuse": "hypotenuse",
        "pythagorean_missing_leg": r"missing\s+(?:leg|side)",
    }
    window = _reasoning_request_window(text, target_by_method[method])
    if not window:
        return False

    shapes = {
        shape
        for shape in ("rectangle", "triangle", "circle")
        if re.search(rf"\b{shape}\b", window, re.IGNORECASE) is not None
    }
    required_shape = {
        "rectangle_area": "rectangle",
        "rectangle_perimeter": "rectangle",
        "triangle_area": "triangle",
        "circle_area": "circle",
        "circle_circumference": "circle",
        "pythagorean_hypotenuse": "triangle",
        "pythagorean_missing_leg": "triangle",
    }[method]
    if shapes != {required_shape}:
        return False

    if method in {"rectangle_area", "rectangle_perimeter"}:
        return _has_labeled_geometry_measure(
            window,
            r"length|long",
        ) and _has_labeled_geometry_measure(window, r"width|wide")
    if method == "triangle_area":
        return _has_labeled_geometry_measure(
            window,
            "base",
        ) and _has_labeled_geometry_measure(window, r"height|high")
    if method in {"circle_area", "circle_circumference"}:
        radius = _has_labeled_geometry_measure(window, "radius")
        diameter = _has_labeled_geometry_measure(window, "diameter")
        return radius != diameter

    if re.search(
        r"\b(?:right(?:-angled)?\s+triangle|pythagorean)\b",
        window,
        re.IGNORECASE,
    ) is None:
        return False
    if method == "pythagorean_hypotenuse":
        unit = (
            r"(?:kilometers?|kilometres?|km|meters?|metres?|m|centimeters?|"
            r"centimetres?|cm|millimeters?|millimetres?|mm|miles?|mi|yards?|"
            r"yd|feet|foot|ft|inches?|inch)"
        )
        return re.search(
            rf"\b(?:legs?|leg\s+lengths?)\b[^.;?!]{{0,24}}"
            rf"{_REASONING_POSITIVE_NUMBER}(?:\s*{unit})?\s*(?:and|,)\s*"
            rf"{_REASONING_POSITIVE_NUMBER}(?:\s*{unit})?(?![a-z0-9])",
            window,
            re.IGNORECASE,
        ) is not None
    return _has_labeled_geometry_measure(
        window,
        "hypotenuse",
    ) and _has_labeled_geometry_measure(window, r"known\s+leg|leg")


def _physics_applicability_established(text: str, method: str) -> bool:
    if method not in _REASONING_PHYSICS_METHODS:
        return True

    if method == "newtons_second_law_force":
        window = _reasoning_request_window(text, r"net\s+force|force")
        return bool(
            window
            and _REASONING_NEWTON_CONTEXT_RE.search(window) is not None
            and _REASONING_FORCE_CAVEAT_RE.search(text) is None
            and _single_reasoning_quantity(_REASONING_MASS_QUANTITY_RE, window)
            and _single_reasoning_quantity(_REASONING_ACCELERATION_QUANTITY_RE, window)
        )

    if method == "density_mass_over_volume":
        window = _reasoning_request_window(text, "density")
        return bool(
            window
            and _REASONING_DENSITY_CAVEAT_RE.search(text) is None
            and re.search(r"\b(?:object|material|substance|sample)\b", window, re.IGNORECASE)
            is not None
            and re.search(r"\bmass\b", window, re.IGNORECASE) is not None
            and re.search(r"\bvolume\b", window, re.IGNORECASE) is not None
            and _single_reasoning_quantity(_REASONING_MASS_QUANTITY_RE, window)
            and _single_reasoning_quantity(_REASONING_VOLUME_QUANTITY_RE, window)
        )

    if method == "kinetic_energy":
        window = _reasoning_request_window(text, r"kinetic\s+energy|ke")
        return bool(
            window
            and _REASONING_KINETIC_CAVEAT_RE.search(text) is None
            and re.search(r"\b(?:object|body|particle)\b", window, re.IGNORECASE)
            is not None
            and re.search(r"\b(?:moving|speed|velocity)\b", window, re.IGNORECASE)
            is not None
            and _single_reasoning_quantity(_REASONING_MASS_QUANTITY_RE, window)
            and _single_reasoning_quantity(_REASONING_SPEED_QUANTITY_RE, window)
        )

    target = {
        "ohms_law_voltage": "voltage",
        "ohms_law_current": "current",
        "ohms_law_resistance": "resistance",
    }[method]
    window = _reasoning_request_window(text, target)
    if not (
        window
        and _REASONING_OHM_CONTEXT_RE.search(window) is not None
        and _REASONING_SIMPLE_RESISTOR_RE.search(window) is not None
        and _REASONING_CIRCUIT_CAVEAT_RE.search(text) is None
    ):
        return False
    if method == "ohms_law_voltage":
        return (
            _single_reasoning_quantity(_REASONING_CURRENT_QUANTITY_RE, window)
            and _single_reasoning_quantity(_REASONING_RESISTANCE_QUANTITY_RE, window)
            and _REASONING_VOLTAGE_QUANTITY_RE.search(window) is None
        )
    if method == "ohms_law_current":
        return (
            _single_reasoning_quantity(_REASONING_VOLTAGE_QUANTITY_RE, window)
            and _single_reasoning_quantity(_REASONING_RESISTANCE_QUANTITY_RE, window)
            and _REASONING_CURRENT_QUANTITY_RE.search(window) is None
        )
    return (
        _single_reasoning_quantity(_REASONING_VOLTAGE_QUANTITY_RE, window)
        and _single_reasoning_quantity(_REASONING_CURRENT_QUANTITY_RE, window)
        and _REASONING_RESISTANCE_QUANTITY_RE.search(window) is None
    )


def _reasoning_consensus_complete(result: Mapping[str, Any]) -> bool:
    """Require verified, exhaustive consensus before granting hard override."""

    if not bool(result.get("solved")):
        return False
    verification = result.get("verification")
    consensus = result.get("consensus")
    budget = result.get("budget")
    if not all(isinstance(value, Mapping) for value in (verification, consensus, budget)):
        return False
    assert isinstance(verification, Mapping)
    assert isinstance(consensus, Mapping)
    assert isinstance(budget, Mapping)
    if verification.get("checked") is not True or verification.get("passed") is not True:
        return False
    paths = consensus.get("paths")
    agreeing = consensus.get("agreeing")
    if (
        type(paths) is not int
        or type(agreeing) is not int
        or paths < 1
        or agreeing < 1
        or agreeing > paths
        or consensus.get("conflicting") is not False
    ):
        return False
    solvers_run = budget.get("solvers_run")
    solver_limit = budget.get("solver_limit")
    solvers_considered = budget.get("solvers_considered")
    return bool(
        budget.get("tier") in {"fast", "deep"}
        and budget.get("early_exit") is False
        and budget.get("all_solvers_exhausted") is True
        and type(solvers_run) is int
        and type(solver_limit) is int
        and type(solvers_considered) is int
        and solver_limit >= 1
        and solvers_run == solver_limit
        and solvers_considered >= solver_limit
    )


def _ground_reasoning_result(query: Any, result: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply conservative raw-prompt gates to an optional reasoning result."""

    guarded = dict(result)
    method = str(guarded.get("method") or "")
    raw_query = str(query or "")
    text = _clean_text(raw_query, MAX_QUERY_CHARS)

    if len(raw_query) > MAX_REASONING_QUERY_CHARS:
        return _reject_reasoning_result(guarded, "query_too_long")
    if (
        _REASONING_REQUEST_CANCELLATION_RE.search(text) is not None
        or _REASONING_LATE_CORRECTION_RE.search(text) is not None
    ):
        return _reject_reasoning_result(guarded, "ambiguous_or_superseded_request")
    quoted_matches = list(_REASONING_QUOTED_SPAN_RE.finditer(text))
    quoted_numeric_input = any(
        _NUMBER_RE.search(match.group(0)) is not None for match in quoted_matches
    )
    quote_remainder = list(text)
    for match in quoted_matches:
        quote_remainder[match.start() : match.end()] = " " * (match.end() - match.start())
    unmatched_quote_delimiter = (
        _REASONING_QUOTE_DELIMITER_RE.search("".join(quote_remainder)) is not None
    )
    if (
        _REASONING_UNTRUSTED_PROBLEM_DATA_RE.search(text) is not None
        or _REASONING_EXCLUDED_SETUP_RE.search(text) is not None
        or (
            quoted_numeric_input
            and _REASONING_AUTHORITATIVE_QUOTED_INPUT_RE.search(text) is None
        )
        or (unmatched_quote_delimiter and _NUMBER_RE.search(text) is not None)
    ):
        return _reject_reasoning_result(guarded, "untrusted_problem_data")
    if (
        _reasoning_has_unconsumed_action(text)
        or _reasoning_has_unconsumed_trailing_content(text)
    ):
        return _reject_reasoning_result(guarded, "mixed_or_unconsumed_request")

    trusted_science_summary: Optional[Dict[str, Any]] = None
    if method in _REASONING_SCIENCE_METHOD_TARGETS:
        claimed_science = _science_summary_from_claim(
            guarded.get("science_plan"),
            guarded.get("science_plan_receipt"),
            method,
        )
        if claimed_science is None:
            summarized = guarded.get("science_plan")
            summarized_receipt = guarded.get("science_plan_receipt")
            if (
                summarized == summarized_receipt
                and _science_summary_is_safe(summarized, method)
            ):
                assert isinstance(summarized, Mapping)
                claimed_science = dict(summarized)
        if claimed_science is None:
            return _reject_reasoning_result(guarded, "science_plan_not_established")
        trusted_science = _trusted_science_summary(raw_query, method)
        if trusted_science is None or claimed_science != trusted_science:
            return _reject_reasoning_result(guarded, "science_plan_result_mismatch")
        trusted_science_summary = dict(trusted_science)
        # Never retain caller-supplied plan objects, source spans, trace strings,
        # or receipt extras. Only this allowlisted, recomputed summary crosses the
        # grounding boundary and can later enter a verified-answer receipt.
        guarded["science_plan"] = dict(trusted_science)
        guarded["science_plan_receipt"] = dict(trusted_science)
        epistemics = dict(guarded.get("epistemics") or {})
        epistemics.update(
            {
                "model_conditional": True,
                "assumptions_explicit": True,
                "calibration_claimed": False,
            }
        )
        guarded["epistemics"] = epistemics

    if not _geometry_intent_established(text, method):
        return _reject_reasoning_result(guarded, "geometry_intent_not_established")

    if not _physics_applicability_established(text, method):
        return _reject_reasoning_result(guarded, "physics_applicability_not_established")

    if method == "explicit_favourable_over_total" and (
        _REASONING_EQUIPROBABLE_RE.search(text) is None
        or _REASONING_UNEQUAL_PROBABILITY_RE.search(text) is not None
    ):
        return _reject_reasoning_result(
            guarded,
            "probability_assumptions_not_established",
        )

    if method in {"fair_coin_single_toss", "fair_die_equiprobable_faces"}:
        if _REASONING_REPEATED_TRIAL_RE.search(text) is not None:
            return _reject_reasoning_result(guarded, "repeated_trials_not_single_trial")
        if (
            _REASONING_UNEQUAL_PROBABILITY_RE.search(text) is not None
            or _REASONING_UNSUPPORTED_PROBABILITY_EVENT_RE.search(text) is not None
            or _reasoning_validator(
                "fair_probability_request_admissible",
                raw_query,
                method,
            )
            is not True
        ):
            return _reject_reasoning_result(
                guarded,
                "probability_event_not_established",
            )

    if method == "finite_binomial_event_probability":
        scenario = _reasoning_validator(
            "parse_finite_bernoulli_scenario",
            raw_query,
        )
        if (
            not isinstance(scenario, Mapping)
            or scenario.get("schema") != "supermix-finite-bernoulli-scenario-v1"
            or scenario.get("full_query_consumed") is not True
        ):
            return _reject_reasoning_result(
                guarded,
                "finite_bernoulli_model_not_established",
                assumptions_explicit=False,
            )
        epistemics = dict(guarded.get("epistemics") or {})
        epistemics.update(
            {
                "model_conditional": True,
                "assumptions_explicit": True,
                "calibration_claimed": False,
            }
        )
        guarded["epistemics"] = epistemics

    if _is_empirical_prediction_result(text, guarded):
        if not _positive_prediction_assumptions(text):
            return _reject_reasoning_result(
                guarded,
                "prediction_assumptions_not_established",
                assumptions_explicit=False,
            )
        trusted = _trusted_reasoning_result(raw_query)
        if trusted is None or not _reasoning_estimate_matches_trusted(guarded, trusted):
            return _reject_reasoning_result(
                guarded,
                "reasoning_result_mismatch",
            )
        guarded = dict(trusted)
        # An observed rate is a model-conditional estimate. Verification can
        # check the arithmetic but cannot turn it into a hard next-trial fact.
        guarded["override_allowed"] = False
        if bool(guarded.get("solved")):
            guarded["reason"] = "verified_non_overriding_estimate"
        epistemics = dict(guarded.get("epistemics") or {})
        epistemics.update(
            {
                "model_conditional": True,
                "assumptions_explicit": True,
                "calibration_claimed": False,
            }
        )
        guarded["epistemics"] = epistemics
        return guarded

    if bool(guarded.get("override_allowed")) and not _reasoning_consensus_complete(guarded):
        return _reject_reasoning_result(guarded, "solver_consensus_incomplete")
    if bool(guarded.get("override_allowed")):
        trusted = _trusted_reasoning_result(raw_query)
        if trusted is None or not _reasoning_result_matches_trusted(guarded, trusted):
            return _reject_reasoning_result(
                guarded,
                "reasoning_result_mismatch",
            )
        # The caller may control descriptive fields even when it copied the
        # correct numeric answer. Publish only the freshly recomputed result.
        guarded = dict(trusted)
        method = str(guarded.get("method") or "")
        if method in _REASONING_SCIENCE_METHOD_TARGETS:
            if trusted_science_summary is None:
                return _reject_reasoning_result(
                    guarded,
                    "science_plan_result_mismatch",
                )
            guarded["science_plan"] = dict(trusted_science_summary)
            guarded["science_plan_receipt"] = dict(trusted_science_summary)

    if method == "fair_die_equiprobable_faces":
        guarded = _correct_odd_sided_die_parity(text, guarded)
    if bool(guarded.get("override_allowed")) and not _reasoning_consensus_complete(guarded):
        return _reject_reasoning_result(guarded, "solver_consensus_incomplete")
    return guarded


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
        or _profile_high_stakes(profile)
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
    reasoning_profile = (
        dict(profile.get("reasoning") or {})
        if isinstance(profile.get("reasoning"), Mapping)
        else {}
    )
    response_contract = (
        dict(profile.get("response_contract") or {})
        if isinstance(profile.get("response_contract"), Mapping)
        else {}
    )
    forbidden_capabilities = {
        str(value)
        for value in (response_contract.get("forbidden_capabilities") or ())
    }
    evidence_forbidden = "evidence_or_calibration" in forbidden_capabilities
    scientific_request = bool(reasoning_profile.get("scientific"))
    prediction_request = bool(reasoning_profile.get("predictive")) and (
        "calibrated_prediction" not in forbidden_capabilities
    )
    mathematical_request = bool(reasoning_profile.get("mathematical"))
    causal_request = bool(reasoning_profile.get("causal")) and (
        "causal_reasoning" not in forbidden_capabilities
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
    if scientific_request:
        reasons.append("scientific_reasoning")
    if prediction_request:
        reasons.append("prediction_requires_calibration")
    if mathematical_request:
        reasons.append("quantitative_reasoning")
    if causal_request:
        reasons.append("causal_reasoning")
    if epistemic_risk >= 0.65:
        reasons.append("interaction_epistemic_risk")
    if evidence_forbidden:
        reasons.append("evidence_capability_forbidden")

    evidence_recommended = bool(
        strict_only
        or freshness
        or citation_requested
        or evidence_requested
        or high_stakes
        or factual
        or (
            not evidence_forbidden
            and (scientific_request or prediction_request or causal_request)
        )
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
        "reasoning_domains": [
            name
            for name, enabled in (
                ("mathematics", mathematical_request),
                ("science", scientific_request),
                ("prediction", prediction_request),
                ("causal", causal_request),
            )
            if enabled
        ],
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
    # A bounded explanation/verification request does not make an otherwise
    # explicit expression ambiguous. Strip only this closed suffix allowlist;
    # arbitrary prose, second calculations, code, and injected instructions
    # continue through the character allowlist below and fail closed.
    suffix = _ARITHMETIC_REQUEST_SUFFIX_RE.search(expression)
    if suffix is not None:
        expression = expression[: suffix.start()].rstrip()
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
    profile = _resolve_prompt_profile(user_text, prompt_profile)
    plan = (
        dict(grounding_plan)
        if isinstance(grounding_plan, Mapping)
        else plan_grounding(
            user_text,
            interaction_plan=interaction_plan,
            prompt_profile=profile,
        )
    )
    bundle = _coerce_evidence_bundle(
        user_text,
        evidence_bundle,
        interaction_plan=interaction_plan,
        prompt_profile=profile,
        grounding_plan=plan,
    )
    diagnostics = evidence_diagnostics(user_text, bundle["evidence"], response_text=raw)
    arithmetic = solve_exact_arithmetic(user_text)
    reasoning = _ground_reasoning_result(user_text, solve_reasoned_problem(user_text))
    # The raw user wording is the authority for a strict-evidence override. A
    # caller-supplied or stale plan cannot invent permission to replace text.
    strict_only = bool(_STRICT_EVIDENCE_ONLY_RE.search(_clean_text(user_text, MAX_QUERY_CHARS)))
    high_stakes = bool(
        _HIGH_STAKES_RE.search(_clean_text(user_text, MAX_QUERY_CHARS))
        or _profile_high_stakes(profile)
    )
    if high_stakes and bool(reasoning.get("override_allowed")):
        reasoning = dict(reasoning)
        reasoning["override_allowed"] = False
        reasoning["reason"] = "high_stakes_override_suppressed"

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
    elif bool(arithmetic.get("solved")) and not high_stakes:
        clean_user_text = _clean_text(user_text, MAX_QUERY_CHARS)
        wants_steps = bool(_SHOW_WORK_RE.search(clean_user_text))
        wants_verification = bool(_VERIFY_RESULT_RE.search(clean_user_text))
        if wants_steps or wants_verification:
            lead = "Using exact arithmetic" if wants_steps else "Verified with exact arithmetic"
            text = f"{lead}: {arithmetic['expression']} = {arithmetic['display']}."
            if arithmetic.get("approximation"):
                text = text[:-1] + f" (approximately {arithmetic['approximation']})."
        else:
            text = f"The exact result is {arithmetic['display']}."
            if arithmetic.get("approximation"):
                text = (
                    f"The exact result is {arithmetic['display']} "
                    f"(approximately {arithmetic['approximation']})."
                )
        reason = "explicit_arithmetic_exact"
    elif (
        not high_stakes
        and _is_empirical_prediction_result(user_text, reasoning)
        and reasoning.get("solved") is True
        and reasoning.get("reason") == "verified_non_overriding_estimate"
        and isinstance(reasoning.get("verification"), Mapping)
        and reasoning["verification"].get("passed") is True
    ):
        module = _load_trusted_reasoning_module()
        rendered = ""
        if module is not None:
            try:
                rendered = str(module.render_reasoning_answer(reasoning, include_steps=False))
            except Exception:  # pragma: no cover - defensive
                rendered = ""
        rendered = rendered.strip() or str(reasoning.get("text") or "").strip()
        if rendered:
            text = rendered
            reason = "verified_model_conditional_estimate"
    elif bool(reasoning.get("override_allowed")) and not high_stakes and not _is_empirical_prediction_result(
        user_text,
        reasoning,
    ):
        # Only a solved problem whose own verification passed, with no
        # disagreement between applicable solvers, may replace the response.
        module = _load_trusted_reasoning_module()
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

    receipt_guard_reason = (
        "high_stakes_suppressed"
        if high_stakes
        and reason == "audit_only"
        and (bool(arithmetic.get("solved")) or bool(reasoning.get("solved")))
        else reason
    )
    answer_receipt = build_verified_answer_receipt(
        reasoning,
        arithmetic,
        response_guard_reason=receipt_guard_reason,
    )
    return {
        "text": text,
        "changed": text != raw,
        "reason": reason,
        "grounding": diagnostics,
        "citations": diagnostics["citation_audit"],
        "arithmetic": arithmetic,
        "reasoning": _public_reasoning_result(reasoning),
        "answer_receipt": answer_receipt,
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }


__all__ = [
    "GROUNDING_RUNTIME_VERSION",
    "GROUNDING_SCHEMA_VERSION",
    "VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION",
    "build_evidence_bundle",
    "build_verified_answer_receipt",
    "evidence_diagnostics",
    "finalize_grounded_response",
    "normalize_evidence_rows",
    "plan_grounding",
    "redact_external_query",
    "solve_exact_arithmetic",
    "solve_reasoned_problem",
    "validate_citations",
]
