"""Deliberate reasoning v2 for Supermix.

This module turns a bounded natural-language request into an explicit,
verifiable problem-solving attempt. It is deterministic, dependency-free,
never calls `eval`, never touches the network, and never mutates state.

Design contract
---------------
* Every answer is computed with exact rational arithmetic (`fractions.Fraction`)
  so that no floating-point drift can enter a stated result.
* Every solver must publish an independent-or-inverse verification. A solution
  that fails its own check is reported, but is never eligible to override a
  response.
* Solvers are ranked, cross-checked for agreement, and any disagreement between
  two applicable solvers marks the attempt as conflicting and non-overriding.
* Consensus is bounded and exhaustive: every registered solver is considered
  before an answer can receive override authority. The requested `fast` or
  `deep` tier is advisory metadata only; this module has no authority over
  model routing, adaptive exit, or permissions.
* Diagnostics never contain the raw prompt or reconstructed prompt text.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


REASONING_SCHEMA_VERSION = "supermix-reasoning-v2"
REASONING_ENGINE_VERSION = "supermix-reasoning-engine-v5"
FINITE_BERNOULLI_SCHEMA_VERSION = "supermix-finite-bernoulli-scenario-v1"

MAX_QUERY_CHARS = 2000
MAX_NUMBERS = 32
MAX_STEPS = 10
MAX_RESULT_BITS = 4096
MAX_LITERAL_DIGITS = 40
MAX_SOLVER_INVOCATIONS = 24
MAX_SEQUENCE_TERMS = 16
MAX_STAT_VALUES = 64
MAX_FACTORIAL_N = 20
MAX_CHOOSE_N = 200
MAX_PRIME_CANDIDATE = 10**12
MAX_DATE_DELTA_DAYS = 400_000
MAX_EQUATION_CHARS = 160
MAX_PERCENT_CHAIN_OPS = 4
MAX_BERNOULLI_TRIALS = 1_000_000_000
MAX_LOGIC_ATOMS = 12
MAX_LOGIC_RULES = 16
MAX_LOGIC_ANTECEDENTS = 3
MAX_BINOMIAL_TRIALS = 200


class _ReasoningLimit(ValueError):
    """Raised internally when a bounded computation exceeds its budget."""


_WS_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}\b)")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9._])-?\d+(?:\.\d+)?(?![\d.]*[A-Za-z_])")
_WORD_RE = re.compile(r"[a-z][a-z'’-]*")

_NUM = r"-?\d+(?:\.\d+)?"

_PERCENT_OF_RE = re.compile(
    rf"(?P<pct>{_NUM})\s*(?:%|percent|per\s?cent)\s+(?:of|off\s+of)\s+(?P<whole>{_NUM})",
)
_PART_IS_WHAT_PERCENT_RE = re.compile(
    rf"(?P<part>{_NUM})\s+(?:is|are)\s+what\s+(?:%|percent|per\s?cent|percentage)\s+of\s+(?P<whole>{_NUM})",
)
_WHAT_PERCENT_OF_IS_RE = re.compile(
    rf"what\s+(?:%|percent|per\s?cent|percentage)\s+of\s+(?P<whole>{_NUM})\s+(?:is|are)\s+(?P<part>{_NUM})",
)
_PERCENT_OF_WHAT_RE = re.compile(
    rf"(?P<part>{_NUM})\s+(?:is|are)\s+(?P<pct>{_NUM})\s*(?:%|percent|per\s?cent)\s+of\s+what",
)
_PERCENT_CHANGE_RE = re.compile(
    rf"(?:%|percent|per\s?cent|percentage)\s+(?P<kind>change|increase|decrease|difference|drop|rise|growth)"
    rf"[^0-9\n]{{0,32}}?from\s+(?P<start>{_NUM})\s+to\s+(?P<end>{_NUM})",
)
_CHANGE_PERCENT_RE = re.compile(
    rf"(?:from|went\s+from|rose\s+from|fell\s+from|increased\s+from|decreased\s+from|grew\s+from|dropped\s+from)"
    rf"\s+(?P<start>{_NUM})\s+to\s+(?P<end>{_NUM})",
)

_GCD_RE = re.compile(rf"(?:gcd|greatest\s+common\s+(?:divisor|factor)|hcf)\D{{0,16}}?(?P<a>{_NUM})\D{{1,12}}?(?P<b>{_NUM})")
_LCM_RE = re.compile(rf"(?:lcm|least\s+common\s+multiple|lowest\s+common\s+multiple)\D{{0,16}}?(?P<a>{_NUM})\D{{1,12}}?(?P<b>{_NUM})")
_PRIME_RE = re.compile(r"is\s+(?P<n>\d+)\s+(?:a\s+)?prime")
_FACTORS_RE = re.compile(r"(?:prime\s+factor(?:s|ization|ise|ize)?|factorise|factorize)\D{0,16}?(?P<n>\d+)")

_FACTORIAL_RE = re.compile(r"(?<![\d.])(?P<n>\d{1,3})\s*!")
_FACTORIAL_WORD_RE = re.compile(r"(?:factorial\s+of\s+(?P<a>\d{1,3})|(?P<b>\d{1,3})\s+factorial)")
_CHOOSE_RE = re.compile(
    r"(?:"
    r"(?P<n1>\d{1,3})\s+choose\s+(?P<r1>\d{1,3})"
    r"|c\s*\(\s*(?P<n2>\d{1,3})\s*,\s*(?P<r2>\d{1,3})\s*\)"
    r"|(?:combinations?|ways)\b[^.\n]{0,48}?\bchoos\w*\s+(?P<r3>\d{1,3})\b[^.\n]{0,24}?\b(?:from|of|out\s+of)\s+(?P<n3>\d{1,3})"
    r"|(?:choose|select|pick)\s+(?P<r4>\d{1,3})\b[^.\n]{0,24}?\b(?:from|of|out\s+of)\s+(?P<n4>\d{1,3})"
    r")"
)
_PERMUTE_RE = re.compile(
    r"(?:"
    r"p\s*\(\s*(?P<n1>\d{1,3})\s*,\s*(?P<r1>\d{1,3})\s*\)"
    r"|permutations?\b[^.\n]{0,32}?\b(?P<r2>\d{1,3})\b[^.\n]{0,24}?\b(?:from|of|out\s+of)\s+(?P<n2>\d{1,3})"
    r"|(?:arrange|order)\s+(?P<r3>\d{1,3})\b[^.\n]{0,24}?\b(?:from|of|out\s+of)\s+(?P<n3>\d{1,3})"
    r")"
)

_STAT_RE = re.compile(r"\b(?P<kind>mean|average|median|mode|range|sum|total)\b")
# A statistics list must be numbers separated only by commas or "and". Requiring
# adjacency keeps phrases like "sum to 30 and differ by 6" out of this solver.
_STAT_LIST_RE = re.compile(rf"{_NUM}(?:\s*(?:,\s*(?:and\s+)?|\s+and\s+){_NUM})+")
_STAT_GAP_RE = re.compile(
    r"[\s:=]*(?:(?:of|for|is|are|the|these|those|numbers?|values?|list|set|data|following)[\s:]+)*",
)

_DAYS_BETWEEN_RE = re.compile(
    r"(?:how\s+many\s+)?(?P<unit>days?|weeks?)\s+(?:are\s+)?(?:there\s+)?between\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)\s*[?.]?\s*$",
)
_DATE_OFFSET_RE = re.compile(
    r"(?P<n>\d{1,6})\s+(?P<unit>days?|weeks?)\s+(?P<dir>after|before|from|prior\s+to)\s+(?P<anchor>.+?)\s*[?.]?\s*$",
)
_ISO_DATE_RE = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_NAMED_DATE_RE = re.compile(
    rf"\b(?:(?P<m1>{_MONTH_ALT})\s+(?P<d1>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y1>\d{{4}})"
    rf"|(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<m2>{_MONTH_ALT}),?\s+(?P<y2>\d{{4}}))\b",
)

_SIMPLE_INTEREST_RE = re.compile(r"simple\s+interest")
_COMPOUND_INTEREST_RE = re.compile(r"compound(?:ed|ing)?\s+(?:interest|annually|yearly)|interest\s+compounded")

_SUM_DIFF_RE = re.compile(
    rf"(?:sum|add\s+(?:up\s+)?to|total|adds?\s+up\s+to)\D{{0,24}}?(?P<sum>{_NUM})"
    rf"\D{{0,48}}?(?:difference|differ(?:ence)?\s+by|differ\s+by|apart)\D{{0,24}}?(?P<diff>{_NUM})",
)
_DIFF_SUM_RE = re.compile(
    rf"(?:difference|differ(?:ence)?\s+by|differ\s+by|apart)\D{{0,24}}?(?P<diff>{_NUM})"
    rf"\D{{0,48}}?(?:sum|add\s+(?:up\s+)?to|total|adds?\s+up\s+to)\D{{0,24}}?(?P<sum>{_NUM})",
)

_SEQUENCE_CUE_RE = re.compile(r"\b(?:next|sequence|series|pattern|continue|comes\s+after|following\s+term)\b")
_SEQUENCE_LIST_RE = re.compile(rf"(?:{_NUM})(?:\s*,\s*(?:{_NUM})){{2,}}")

_WORK_RATE_RE = re.compile(
    rf"(?P<a>{_NUM})\s*(?P<au>hours?|hrs?|h|minutes?|mins?|days?)\b[^.\n]{{0,80}}?"
    rf"(?P<b>{_NUM})\s*(?P<bu>hours?|hrs?|h|minutes?|mins?|days?)\b",
)
_TOGETHER_RE = re.compile(r"\b(?:together|combined|both\s+work|working\s+together|at\s+the\s+same\s+time)\b")

_PROPORTION_CUE_RE = re.compile(r"\b(?:same\s+rate|that\s+rate|this\s+rate|proportion|proportional|per)\b")

_EQUATION_VAR_RE = re.compile(r"(?:solve\s+for|find)\s+(?P<var>[a-z])\b")

# ---------------------------------------------------------------------------
# Unit conversion tables. Every factor is stored as an exact Fraction of the
# dimension base unit, so conversions never lose precision.
# ---------------------------------------------------------------------------

def _F(numerator: int, denominator: int = 1) -> Fraction:
    return Fraction(numerator, denominator)


_LENGTH_M: Dict[str, Fraction] = {
    "km": _F(1000), "m": _F(1), "cm": _F(1, 100), "mm": _F(1, 1000),
    "um": _F(1, 1_000_000), "nm": _F(1, 1_000_000_000),
    "mi": Fraction(1609344, 1000), "yd": Fraction(9144, 10000),
    "ft": Fraction(3048, 10000), "in": Fraction(254, 10000),
    "nmi": _F(1852),
}
_MASS_G: Dict[str, Fraction] = {
    "t": _F(1_000_000), "kg": _F(1000), "g": _F(1), "mg": _F(1, 1000),
    "ug": _F(1, 1_000_000),
    "lb": Fraction(45359237, 100000), "oz": Fraction(45359237, 1600000),
    "st": Fraction(45359237 * 14, 100000),
}
_VOLUME_L: Dict[str, Fraction] = {
    "kl": _F(1000), "l": _F(1), "dl": _F(1, 10), "cl": _F(1, 100), "ml": _F(1, 1000),
    "m3": _F(1000), "cm3": _F(1, 1000),
    "gal": Fraction(3785411784, 1000000000), "qt": Fraction(3785411784, 4000000000),
    "pt": Fraction(3785411784, 8000000000), "cup": Fraction(2365882365, 10000000000),
    "floz": Fraction(295735295625, 10000000000000), "tbsp": Fraction(295735295625, 20000000000000),
    "tsp": Fraction(295735295625, 60000000000000),
}
_TIME_S: Dict[str, Fraction] = {
    "ns": _F(1, 1_000_000_000), "us": _F(1, 1_000_000), "ms": _F(1, 1000),
    "s": _F(1), "min": _F(60), "h": _F(3600), "day": _F(86400),
    "week": _F(604800), "fortnight": _F(1209600), "year": _F(31_536_000),
}
_DATA_B: Dict[str, Fraction] = {
    "bit": _F(1, 8), "b": _F(1),
    "kb": _F(1000), "mb": _F(1_000_000), "gb": _F(1_000_000_000), "tb": _F(10**12),
    "kib": _F(1024), "mib": _F(1024**2), "gib": _F(1024**3), "tib": _F(1024**4),
}
_AREA_M2: Dict[str, Fraction] = {
    "km2": _F(1_000_000), "m2": _F(1), "cm2": _F(1, 10_000), "mm2": _F(1, 1_000_000),
    "ha": _F(10_000), "acre": Fraction(40468564224, 10000000),
    "ft2": Fraction(3048, 10000) ** 2, "in2": Fraction(254, 10000) ** 2,
    "mi2": Fraction(1609344, 1000) ** 2, "yd2": Fraction(9144, 10000) ** 2,
}
_SPEED_MS: Dict[str, Fraction] = {
    "m/s": _F(1), "km/h": Fraction(1000, 3600), "mph": Fraction(1609344, 1000 * 3600),
    "ft/s": Fraction(3048, 10000), "kn": Fraction(1852, 3600),
}

_DIMENSIONS: Dict[str, Dict[str, Fraction]] = {
    "length": _LENGTH_M,
    "mass": _MASS_G,
    "volume": _VOLUME_L,
    "time": _TIME_S,
    "data": _DATA_B,
    "area": _AREA_M2,
    "speed": _SPEED_MS,
}

_UNIT_ALIASES: Dict[str, Tuple[str, str]] = {}


def _register_units(dimension: str, aliases: Mapping[str, Sequence[str]]) -> None:
    for canonical, names in aliases.items():
        for name in names:
            _UNIT_ALIASES[name] = (dimension, canonical)


_register_units("length", {
    "km": ("km", "kms", "kilometer", "kilometers", "kilometre", "kilometres"),
    "m": ("m", "meter", "meters", "metre", "metres"),
    "cm": ("cm", "centimeter", "centimeters", "centimetre", "centimetres"),
    "mm": ("mm", "millimeter", "millimeters", "millimetre", "millimetres"),
    "um": ("micrometer", "micrometers", "micron", "microns"),
    "nm": ("nanometer", "nanometers"),
    "mi": ("mi", "mile", "miles"),
    "yd": ("yd", "yds", "yard", "yards"),
    "ft": ("ft", "foot", "feet"),
    "in": ("in", "inch", "inches"),
    "nmi": ("nmi", "nautical mile", "nautical miles"),
})
_register_units("mass", {
    "t": ("tonne", "tonnes", "metric ton", "metric tons"),
    "kg": ("kg", "kgs", "kilogram", "kilograms", "kilo", "kilos"),
    "g": ("g", "gram", "grams", "gramme", "grammes"),
    "mg": ("mg", "milligram", "milligrams"),
    "ug": ("microgram", "micrograms"),
    "lb": ("lb", "lbs", "pound", "pounds"),
    "oz": ("oz", "ounce", "ounces"),
    "st": ("stone", "stones"),
})
_register_units("volume", {
    "kl": ("kiloliter", "kiloliters", "kilolitre", "kilolitres"),
    "l": ("l", "liter", "liters", "litre", "litres"),
    "dl": ("dl", "deciliter", "deciliters"),
    "cl": ("cl", "centiliter", "centiliters"),
    "ml": ("ml", "milliliter", "milliliters", "millilitre", "millilitres"),
    "m3": ("cubic meter", "cubic meters", "cubic metre", "cubic metres"),
    "cm3": ("cubic centimeter", "cubic centimeters", "cc"),
    "gal": ("gal", "gallon", "gallons"),
    "qt": ("qt", "quart", "quarts"),
    "pt": ("pint", "pints"),
    "cup": ("cup", "cups"),
    "floz": ("fl oz", "fluid ounce", "fluid ounces"),
    "tbsp": ("tbsp", "tablespoon", "tablespoons"),
    "tsp": ("tsp", "teaspoon", "teaspoons"),
})
_register_units("time", {
    "ns": ("nanosecond", "nanoseconds"),
    "us": ("microsecond", "microseconds"),
    "ms": ("ms", "millisecond", "milliseconds"),
    "s": ("s", "sec", "secs", "second", "seconds"),
    "min": ("min", "mins", "minute", "minutes"),
    "h": ("h", "hr", "hrs", "hour", "hours"),
    "day": ("day", "days"),
    "week": ("week", "weeks"),
    "fortnight": ("fortnight", "fortnights"),
    "year": ("year", "years"),
})
_register_units("data", {
    "bit": ("bit", "bits"),
    "b": ("byte", "bytes"),
    "kb": ("kb", "kilobyte", "kilobytes"),
    "mb": ("mb", "megabyte", "megabytes"),
    "gb": ("gb", "gigabyte", "gigabytes"),
    "tb": ("tb", "terabyte", "terabytes"),
    "kib": ("kib", "kibibyte", "kibibytes"),
    "mib": ("mib", "mebibyte", "mebibytes"),
    "gib": ("gib", "gibibyte", "gibibytes"),
    "tib": ("tib", "tebibyte", "tebibytes"),
})
_register_units("area", {
    "km2": ("square kilometer", "square kilometers", "square kilometre", "square kilometres", "km2"),
    "m2": ("square meter", "square meters", "square metre", "square metres", "m2"),
    "cm2": ("square centimeter", "square centimeters", "cm2"),
    "mm2": ("square millimeter", "square millimeters", "mm2"),
    "ha": ("hectare", "hectares"),
    "acre": ("acre", "acres"),
    "ft2": ("square foot", "square feet", "ft2"),
    "in2": ("square inch", "square inches", "in2"),
    "mi2": ("square mile", "square miles", "mi2"),
    "yd2": ("square yard", "square yards", "yd2"),
})
_register_units("speed", {
    "m/s": ("m/s", "meters per second", "metres per second"),
    "km/h": ("km/h", "kmh", "kph", "kilometers per hour", "kilometres per hour"),
    "mph": ("mph", "miles per hour"),
    "ft/s": ("ft/s", "feet per second"),
    "kn": ("kn", "knot", "knots"),
})

_TEMPERATURE_ALIASES: Dict[str, str] = {
    "c": "c", "celsius": "c", "centigrade": "c", "degrees celsius": "c", "°c": "c",
    "f": "f", "fahrenheit": "f", "degrees fahrenheit": "f", "°f": "f",
    "k": "k", "kelvin": "k", "kelvins": "k",
}

_UNIT_DISPLAY: Dict[str, str] = {
    "km": "km", "m": "m", "cm": "cm", "mm": "mm", "um": "µm", "nm": "nm",
    "mi": "miles", "yd": "yd", "ft": "ft", "in": "in", "nmi": "nmi",
    "t": "tonnes", "kg": "kg", "g": "g", "mg": "mg", "ug": "µg",
    "lb": "lb", "oz": "oz", "st": "stone",
    "kl": "kL", "l": "L", "dl": "dL", "cl": "cL", "ml": "mL",
    "m3": "m³", "cm3": "cm³", "gal": "gal", "qt": "qt", "pt": "pt",
    "cup": "cups", "floz": "fl oz", "tbsp": "tbsp", "tsp": "tsp",
    "ns": "ns", "us": "µs", "ms": "ms", "s": "seconds", "min": "minutes",
    "h": "hours", "day": "days", "week": "weeks", "fortnight": "fortnights",
    "year": "years",
    "bit": "bits", "b": "bytes", "kb": "KB", "mb": "MB", "gb": "GB", "tb": "TB",
    "kib": "KiB", "mib": "MiB", "gib": "GiB", "tib": "TiB",
    "km2": "km²", "m2": "m²", "cm2": "cm²", "mm2": "mm²", "ha": "hectares",
    "acre": "acres", "ft2": "ft²", "in2": "in²", "mi2": "mi²", "yd2": "yd²",
    "m/s": "m/s", "km/h": "km/h", "mph": "mph", "ft/s": "ft/s", "kn": "knots",
    "c": "°C", "f": "°F", "k": "K",
}

_UNIT_TOKEN_ALT = "|".join(
    re.escape(name)
    for name in sorted(
        list(_UNIT_ALIASES) + list(_TEMPERATURE_ALIASES),
        key=len,
        reverse=True,
    )
)
_UNIT_TOKEN_RE = re.compile(rf"(?<![a-z0-9])(?:{_UNIT_TOKEN_ALT})(?![a-z0-9])")

_CONVERT_RE = re.compile(
    rf"(?:convert\s+)?(?P<value>{_NUM})\s*(?P<from>{_UNIT_TOKEN_ALT})"
    rf"\s+(?:to|in|into|as)\s+(?P<to>{_UNIT_TOKEN_ALT})\b",
)
_CONVERT_HOWMANY_RE = re.compile(
    rf"how\s+many\s+(?P<to>{_UNIT_TOKEN_ALT})\s+(?:are\s+|is\s+)?(?:there\s+)?(?:in|is|are|equal\s+to|make)\s+"
    rf"(?P<value>{_NUM})\s*(?P<from>{_UNIT_TOKEN_ALT})\b",
)

_RATE_SPEED_RE = re.compile(
    rf"(?:travels?|covers?|drives?|runs?|walks?|flies|goes|moved?|cycles?)\s+"
    rf"(?P<dist>{_NUM})\s*(?P<dunit>{_UNIT_TOKEN_ALT})\s+(?:in|over|during)\s+"
    rf"(?P<time>{_NUM})\s*(?P<tunit>{_UNIT_TOKEN_ALT})\b",
)
_RATE_DISTANCE_RE = re.compile(
    rf"(?:at|of)\s+(?P<speed>{_NUM})\s*(?P<sunit>mph|km/h|kmh|kph|m/s|miles per hour|kilometers per hour|"
    rf"kilometres per hour|meters per second|metres per second|feet per second|knots?)\s+"
    rf"(?:for|over|during)\s+(?P<time>{_NUM})\s*(?P<tunit>{_UNIT_TOKEN_ALT})\b",
)

_PERCENT_OP_RE = re.compile(
    rf"(?P<pct>{_NUM})\s*(?:%|percent|per\s?cent)\s*(?P<word>off|discount|tax|tip|gratuity|increase|"
    rf"raise|markup|added|more|less|reduction|fee|commission|interest)?",
)
_PERCENT_DOWN_WORDS = {"off", "discount", "reduction", "less", "decrease"}
_PERCENT_UP_WORDS = {"tax", "tip", "gratuity", "increase", "raise", "markup", "added", "more", "fee", "commission", "interest"}
_CURRENCY_AMOUNT_RE = re.compile(rf"(?:[$£€]\s*(?P<sym>{_NUM})|(?P<word>{_NUM})\s*(?:dollars|pounds|euros|usd|gbp|eur))")

# An intentionally narrow grammar for evidence-backed state-transition plans.
# Each operation must live in its own temporal clause so unrelated quantities
# cannot be silently assembled into a synthetic multi-step problem.
_STATE_COUNT_UNIT_ALT = r"items?|widgets?|tokens?|points?"
_STATE_QUANTITY_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>{_UNIT_TOKEN_ALT}|{_STATE_COUNT_UNIT_ALT})(?![a-z0-9])"
)
_STATE_START_RE = re.compile(
    r"\b(?:(?:starts?|begins?)\s+(?:with|at)|initial\s+"
    r"(?P<label>amount|balance|quantity|volume|mass|distance|total|inventory)\s*(?:is|=|of))\b"
)
_STATE_TRANSITION_RE = re.compile(r"\b(?:then|after\s+that|next|finally)\b")
_STATE_FINAL_TARGET_RE = re.compile(
    r"\b(?:what\s+is|what's|calculate|compute|find|determine)\b[^?.]{0,64}"
    r"\bfinal\s+(?P<label>amount|balance|quantity|volume|mass|distance|total|inventory)\b|"
    r"\bhow\s+(?:much|many)\b[^?.]{0,64}\b(?:remain(?:s|ing)?|left)\b"
)
_STATE_PERCENT_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?:%|percent|per\s?cent)(?![a-z0-9])"
)
_STATE_UP_RE = re.compile(
    r"\b(?:add(?:s|ed|ing)?|deposit(?:s|ed|ing)?|receive(?:s|d|ing)?|gain(?:s|ed|ing)?|"
    r"increase(?:s|d|ing)?|grow(?:s|n|ing)?|credit(?:s|ed|ing)?)\b"
)
_STATE_DOWN_RE = re.compile(
    r"\b(?:subtract(?:s|ed|ing)?|remove(?:s|d|ing)?|withdraw(?:s|n|ing)?|spend(?:s|ing)?|"
    r"spent|lose(?:s|ing)?|lost|decrease(?:s|d|ing)?|reduce(?:s|d|ing)?|drop(?:s|ped|ping)?|"
    r"use(?:s|d|ing)?)\b"
)

# A deliberately explicit positive-Horn grammar. Atoms are opaque identifiers:
# the engine composes only the facts and implications the user supplied and
# never assigns real-world meaning to a label.
_HORN_ATOM_RE = re.compile(r"[a-z][a-z0-9_]{0,31}")
_HORN_PROBLEM_RE = re.compile(
    r"^facts\s*:\s*(?P<facts>[^.]{1,384})\s*\.\s*"
    r"rules\s*:\s*(?P<rules>[^.]{1,1024})\s*\.\s*"
    r"query\s*:\s*(?P<query>[a-z][a-z0-9_]{0,31})\s*[?.]\s*$"
)
_HORN_RESERVED_ATOMS = frozenset({"facts", "rules", "query"})

# Deliberate Reasoning v3 uses narrow, verifier-backed grammars. These unit
# expressions deliberately cover only dimensions for which the solver can do
# an exact conversion and an inverse check. A phrase that falls outside this
# grammar is left to the language model rather than guessed at.
_GEO_UNIT_ALT = (
    r"kilometers?|kilometres?|km|meters?|metres?|m|centimeters?|centimetres?|cm|"
    r"millimeters?|millimetres?|mm|miles?|mi|yards?|yd|feet|foot|ft|inches?|inch"
)
_QUESTION_VERB_RE = re.compile(
    r"\b(?:what\s+(?:is|are)|what's|calculate|compute|derive|evaluate|find|solve|"
    r"determine|work\s+out|convert|give(?:\s+me)?|provide|return|tell\s+me|show(?:\s+me)?|"
    r"translate|summari[sz]e|write|compose|describe|define|discuss|explain|list|create|"
    r"recommend|compare|analy[sz]e|critique|proofread|suggest|format|generate)\b"
)
_NON_CALCULATION_INTENT_RE = re.compile(
    r"\b(?:word|term|phrase|definition|meaning|concept|conceptual|quote|quoted|text|string|occurrence)\b"
)
_NEGATED_REQUEST_RE = re.compile(r"\b(?:do\s+not|don't|never|not)\b")
_REQUEST_CANCELLATION_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+"
    r"(?:calculate|compute|solve|answer|evaluate|determine|work\s+out)\b"
)
_LATE_CORRECTION_RE = re.compile(
    r"[.!?]\s*(?:actually\b|correction\b|no(?:\s|,)|rather\b|instead\b)"
)
_UNTRUSTED_PROBLEM_DATA_RE = re.compile(
    r"\b(?:do\s+not|don't|never)\s+use\b|"
    r"\bignore\s+(?:the\s+)?(?:previous|prior|quoted|following|example|data)\b|"
    r"\b(?:quoted\s+)?(?:text|data|example)\s+(?:is|are)\s+untrusted\b|"
    r"\bnot\s+(?:an?\s+)?instructions?\b"
)
_AUTHORITATIVE_QUOTED_INPUT_RE = re.compile(
    r"\b(?:use|treat)\s+(?:the|this)\s+(?:quoted|following)\s+"
    r"(?:problem|text|example|data)\s+(?:as\s+)?(?:the\s+)?"
    r"(?:authoritative|input|given\s+data)\b"
)
_EXCLUDED_SETUP_RE = re.compile(
    r"\b(?:ignore|discard|exclude|omit)\b|"
    r"\b(?:do\s+not|don't|never)\s+(?:consider|include|use|rely\s+on)\b|"
    r"\b(?:incorrect|invalid|untrusted|irrelevant|decoy|counterexample|fake|"
    r"hypothetical|alleged|blockquote|code\s+example|quoted\s+example|markdown\s+quote)\b|"
    r"\b(?:should|must)\s+not\s+count\b|"
    r"\b(?:neither|nor|without|except|excluding|other\s+than)\b|"
    r"\bset\s+aside\b|\bset\s+(?:it|them|these|those|the\s+values?)\s+aside\b|"
    r"\bdo\s+not\s+take\s+into\s+account\b|"
    r"\b(?:just|only)\s+an?\s+example\b|"
    r"\b(?:for|as)\s+comparison\s+only\b"
)
_UNCONSUMED_ACTION_RE = re.compile(
    r"\b(?:translate|summari[sz]e|write|compose|describe|define|definition|discuss|"
    r"explain|list|create|draw|sing|haiku|poem|story|benefits?|useful|real[- ]world\s+use|"
    r"recommend|compare|analy[sz]e|critique|proofread|suggest|format|generate)\b"
)
_COORDINATED_TAIL_RE = re.compile(
    r"(?:\band\b|\balso\b|\bthen\b|;)\s*"
    r"(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?(?P<next>[a-z]+|\d+)"
)
_SAFE_COORDINATED_TASK_WORDS = frozenset(
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
_ALLOWED_REASONING_ACTION_RE = re.compile(
    r"\b(?:explain\s+(?:(?:your|the)\s+)?(?:reasoning|working|steps)|"
    r"show\s+(?:(?:your|the)\s+)?(?:work|working|steps)|"
    r"(?:verify|check)\s+(?:(?:the|your)\s+)?(?:answer|result|calculation|work))\b"
)
_AFFIRMATIVE_GEOMETRY_SETUP_RE = re.compile(
    r"\b(?:a|the)\s+(?:right(?:-angled)?\s+)?(?:rectangle|triangle|circle)\s+"
    r"(?:has|with|whose)\b|"
    r"\b(?:rectangle|triangle|circle)\b[^.;?!]{0,80}"
    r"\b(?:length|width|base|height|radius|diameter|hypotenuse|legs?)\s*(?:is|are|=|:)"
)
_AMBIGUOUS_IN_UNIT_RE = re.compile(rf"(?<![a-z0-9.]){_NUM}\s+in\b")
_QUOTED_SPAN_RE = re.compile(
    r'''(?:"[^"\n]{0,2000}"|`[^`\n]{0,2000}`|'''
    r'''\u201c[^\u201d\n]{0,2000}\u201d|\u2018[^\u2019\n]{0,2000}\u2019|'''
    r'''\u00ab[^\u00bb\n]{0,2000}\u00bb|\u300c[^\u300d\n]{0,2000}\u300d|'''
    r'''\u300e[^\u300f\n]{0,2000}\u300f|'''
    r'''(?<![a-z0-9])'[^'\n]{0,2000}'(?![a-z0-9]))'''
)
_QUOTE_DELIMITER_RE = re.compile(
    r'''["`\u201c\u201d\u2018\u00ab\u00bb\u300c\u300d\u300e\u300f]|'''
    r'''(?<![a-z0-9])['\u2019]|['\u2019](?![a-z0-9])'''
)
_FAIR_COIN_RE = re.compile(r"\b(?:a\s+)?fair\s+coin\b")
_FAIR_DIE_RE = re.compile(
    r"\b(?:a\s+)?(?:standard\s+fair\s+die|fair\s+(?:(?P<sides>\d{1,3})|six)[ -]sided\s+die)\b"
)
_REPEATED_COIN_RE = re.compile(
    r"\b(?:twice|thrice|multiple\s+times|repeated(?:ly)?)\b|"
    r"\b(?:flip(?:ped|ping)?|toss(?:ed|ing)?)\b[^?.]{0,20}\b(?:\d{1,4}|two|three|four|five|six|seven|eight|nine|ten)\s+times?\b|"
    r"\b(?:\d{1,4}|two|three|four|five|six|seven|eight|nine|ten)\s+(?:coin\s+)?(?:flips?|tosses?)\b|"
    r"\b(?:flips?|tosses?)\s+(?:\d{1,4}|two|three|four|five|six|seven|eight|nine|ten)\b"
)
_REPEATED_DIE_RE = re.compile(
    r"\b(?:roll(?:ed|ing)?|die)\b[^?.]{0,20}\b(?:twice|thrice|\d{1,4}\s+times?|"
    r"two\s+times?|three\s+times?|four\s+times?|five\s+times?|ten\s+times?)\b|"
    r"\b(?:\d{1,4}|two|three|four|five|six|seven|eight|nine|ten)\s+(?:die\s+)?rolls?\b|"
    r"\bmultiple\s+rolls?\b"
)
_FAVOURABLE_COUNT_RE = re.compile(
    rf"(?:(?P<a>{_NUM})\s+(?:favourable|favorable|successful|winning)\s+(?:cases?|outcomes?)"
    rf"|(?:favourable|favorable|successful|winning)\s+(?:cases?|outcomes?)\s*(?:is|are|=|:)\s*(?P<b>{_NUM}))"
)
_TOTAL_COUNT_RE = re.compile(
    rf"(?:(?P<a>{_NUM})\s+(?:(?:equally\s+likely|equiprobable)\s+)?total\s+(?:cases?|outcomes?)"
    rf"|total\s+(?:cases?|outcomes?)\s*(?:is|are|=|:)\s*(?P<b>{_NUM}))"
)
_EQUIPROBABLE_CUE_RE = re.compile(
    r"\b(?:equally\s+likely|equiprobable)(?:\s+(?:cases?|outcomes?))?\b|"
    r"\bfair\s+(?:experiment|selection|cases?|outcomes?)\b"
)
_UNEQUAL_PROBABILITY_RE = re.compile(
    r"\b(?:unequal(?:ly)?|weighted|biased|unfair|not\s+equally\s+likely|"
    r"not\s+equiprobable|different\s+probabilities|not\s+(?:a\s+)?fair)\b"
)
_UNSUPPORTED_PROBABILITY_EVENT_RE = re.compile(
    r"\b(?:not|except|excluding|other\s+than)\b|"
    r"\b(?:numbered|labelled|labeled)\b|\bfaces?\s*(?:are|=|:)"
)
_WITHOUT_REPLACEMENT_RE = re.compile(r"\bwithout\s+replacement\b")
_BERNOULLI_COUNTS_RE = re.compile(
    r"(?P<successes>\d{1,10})\s+success(?:es)?\s+(?:in|out\s+of)\s+"
    r"(?P<trials>\d{1,10})\s+(?:bernoulli\s+)?trials?\b"
)
_ASSUMPTION_CLAUSE_RE = re.compile(
    r"\b(?:assuming|assume|under\s+the\s+assumption\s+that)\b(?P<body>[^,.;?!]{1,180})"
)
_IID_CUE_RE = re.compile(r"\b(?:i\.?i\.?d\.?|independent\s+and\s+identically\s+distributed)\b")
_INDEPENDENCE_CUE_RE = re.compile(r"\bindependent(?:ly)?\b")
_STATIONARY_CUE_RE = re.compile(
    r"\b(?:same|constant|unchanged|stationary)\s+(?:success\s+)?(?:probability|rate)\b"
)
_NEGATED_ASSUMPTION_RE = re.compile(
    r"\b(?:not|isn't|aren't|without|dependent|non-?independent|non-?stationary|"
    r"changing|varying|different|unknown|may\s+change)\b"
)
_NEXT_TRIAL_RE = re.compile(
    r"\b(?:(?:predict(?:ed|ive)?\s+)?(?:probability|chance)\b[^?.]{0,48}\bnext\s+(?:trial|outcome)"
    r"|next\s+(?:trial|outcome)\b[^?.]{0,48}\b(?:probability|chance))"
)
_CERTAINTY_REQUEST_RE = re.compile(r"\b(?:guarantee|guaranteed|certain(?:ly)?|definite(?:ly)?|surely)\b")
_HIGH_STAKES_PREDICTION_RE = re.compile(
    r"\b(?:medical|clinical|patient|diagnos(?:is|e)|disease|treatment|drug|dose|pregnan\w*|"
    r"mortality|survival|cancer|stock|shares?|crypto|invest(?:ment|ing)|trade|market|price|"
    r"election|votes?|credit|loan|mortgage|insurance|sentence|parole|bail|recidivism|"
    r"employee|hiring|firing|admission|school|crime|weather|disaster)\b"
)

_FINITE_PROBABILITY_TOKEN = (
    r"(?:100(?:\.0{1,4})?|\d{1,2}(?:\.\d{1,4})?)%|"
    r"(?:0|1)(?:\.\d{1,6})?|\d{1,6}/\d{1,6}"
)
_FINITE_BERNOULLI_REQUEST_RE = re.compile(
    r"^(?:assuming|assume|under\s+the\s+assumption\s+that)\s+"
    r"(?P<model>[^,;?!]{1,240})\s*[,;]\s*"
    r"(?P<question>[^?]{1,240})\s*\?\s*$"
)
_FINITE_BERNOULLI_EXPLICIT_MODEL_RE = re.compile(
    rf"^(?P<trials>\d{{1,3}})\s+"
    r"(?P<mode>i\.?\s*i\.?\s*d\.?|independent\s+and\s+identically\s+distributed|independent)\s+"
    r"(?:bernoulli\s+)?trials?\s+with\s+"
    r"(?:(?:a|the)\s+)?(?:(?P<stationary>constant|fixed|same)\s+)?"
    rf"success\s+probability\s*(?:of|=|is)\s*(?P<probability>{_FINITE_PROBABILITY_TOKEN})$"
)
_FINITE_BERNOULLI_COIN_MODEL_RE = re.compile(
    r"^(?P<trials>\d{1,3})\s+"
    r"(?P<mode>i\.?\s*i\.?\s*d\.?|independent\s+and\s+identically\s+distributed|independent)\s+"
    r"fair\s+coin\s+(?:tosses|flips)$"
)
_FINITE_BERNOULLI_QUESTION_RE = re.compile(
    r"^(?:what\s+is|what's|calculate|compute|find|determine)\s+"
    r"(?:the\s+)?(?:probability|chance)\s+(?:of|that)\s+"
    r"(?P<event>exactly|at\s+least|at\s+most)\s+"
    r"(?P<count>\d{1,3})\s+"
    r"(?P<outcome>success(?:es)?|heads?|tails?)"
    r"(?:\s+(?:occurs?|occurring|in\s+(?:the|those)\s+trials?))?\s*$"
)

_PHYS_MASS_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>kilograms?|kg|milligrams?|mg|grams?|g)(?![a-z0-9])"
)
_PHYS_VOLUME_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>cubic\s+meters?|m\s*\^?\s*3|m3|"
    rf"cubic\s+centimeters?|cm\s*\^?\s*3|cm3|millilit(?:er|re)s?|ml|lit(?:er|re)s?|l)(?![a-z0-9])"
)
_PHYS_ACCELERATION_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>m\s*/\s*s\s*(?:\^\s*2|2|²)|"
    rf"met(?:er|re)s?\s+per\s+second\s+squared)(?![a-z0-9])"
)
_PHYS_SPEED_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>km\s*/\s*h|kph|kmh|"
    rf"kilomet(?:er|re)s?\s+per\s+hour|m\s*/\s*s(?!\s*(?:\^\s*2|2|²))|"
    rf"met(?:er|re)s?\s+per\s+second(?!\s+squared))(?![a-z0-9])"
)
_PHYS_FORCE_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>newtons?|n)(?![a-z0-9])"
)
_PHYS_RESISTANCE_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>kiloohms?|kohms?|kω|ohms?|ω)(?![a-z0-9])"
)
_PHYS_CURRENT_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>milliamps?|milliamperes?|ma|amps?|amperes?|a)(?![a-z0-9])"
)
_PHYS_VOLTAGE_RE = re.compile(
    rf"(?<![a-z0-9.])(?P<value>{_NUM})\s*(?P<unit>millivolts?|mv|volts?|v)(?![a-z0-9])"
)
_NEWTON_CONTEXT_RE = re.compile(
    r"\b(?:net\s+force|newton(?:'s)?\s+second\s+law)\b|\bf\s*=\s*m\s*(?:\*|x)?\s*a\b"
)
_NON_NET_FORCE_RE = re.compile(
    r"\b(?:friction(?:al)?|applied\s+force|tension|drag|air\s+resistance|normal\s+force|"
    r"weight|gravity|incline|slope)\b"
)
_OHM_LAW_CONTEXT_RE = re.compile(r"\bohm(?:'s)?\s+law\b|\bv\s*=\s*i\s*(?:\*|x)?\s*r\b")
_SIMPLE_OHM_ELEMENT_RE = re.compile(
    r"\b(?:(?:single|one|same)\s+(?:resistor|resistive\s+element|element)"
    r"|a\s+[^.,;?]{0,32}\b(?:ohms?|ω)\s+(?:resistor|element)"
    rf"|(?:through|across)\s+{_NUM}\s*(?:ohms?|ω))\b"
)
_MULTI_COMPONENT_CIRCUIT_RE = re.compile(
    r"\b(?:branches?|series|parallel|network|multiple|multi-component|equivalent\s+resistance|"
    r"two\s+resistors|three\s+resistors|four\s+resistors)\b"
)
_DENSITY_CONTEXT_RE = re.compile(r"\b(?:object|material|substance|sample)\b")
_DENSITY_AMBIGUITY_RE = re.compile(
    r"\b(?:mixture|layered|composite|porous|relative\s+density|specific\s+gravity|buoyancy)\b"
)
_KINETIC_CONTEXT_RE = re.compile(r"\b(?:object|body|particle)\b[^?.]{0,64}\b(?:moving|speed|velocity)\b")
_KINETIC_AMBIGUITY_RE = re.compile(
    r"\b(?:rotational|rolling|angular|relativistic|collision|system\s+of|multiple\s+objects)\b"
)


# ---------------------------------------------------------------------------
# Bounded numeric helpers
# ---------------------------------------------------------------------------

def _clean_text(value: Any, limit: int = MAX_QUERY_CHARS) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return _WS_RE.sub(" ", normalized).strip()[: max(0, int(limit))]


def _source_text(value: Any, limit: int = MAX_QUERY_CHARS) -> str:
    """Preserve stripped NFC source text for integrity-bound parsers."""

    normalized = unicodedata.normalize("NFC", str(value or ""))
    return normalized.strip()[: max(0, int(limit))]


def _guard(value: Fraction) -> Fraction:
    if (
        abs(value.numerator).bit_length() > MAX_RESULT_BITS
        or value.denominator.bit_length() > MAX_RESULT_BITS
    ):
        raise _ReasoningLimit("result_too_large")
    return value


def _fraction(token: Any) -> Optional[Fraction]:
    text = str(token or "").strip().replace(",", "")
    if not text or len(text.lstrip("+-").replace(".", "")) > MAX_LITERAL_DIGITS:
        return None
    try:
        return _guard(Fraction(Decimal(text)))
    except (InvalidOperation, ValueError, ZeroDivisionError, _ReasoningLimit):
        return None


def _terminating_decimal(value: Fraction) -> Optional[str]:
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return None
    with localcontext() as context:
        context.prec = min(
            260,
            max(32, len(str(abs(value.numerator))) + len(str(value.denominator)) + 8),
        )
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
    cooked = format(decimal, "f")
    if "." in cooked:
        cooked = cooked.rstrip("0").rstrip(".")
    return cooked or "0"


def _rounded_decimal(value: Fraction, places: int) -> str:
    places = max(0, min(12, int(places)))
    try:
        with localcontext() as context:
            context.prec = 60
            decimal = Decimal(value.numerator) / Decimal(value.denominator)
            quantized = decimal.quantize(Decimal(1).scaleb(-places))
    except (InvalidOperation, ValueError, ArithmeticError):
        return ""
    cooked = format(quantized, "f")
    if "." in cooked:
        cooked = cooked.rstrip("0").rstrip(".")
    return cooked or "0"


def _present(value: Fraction, *, prefer_decimal: bool = True, decimals: int = 6) -> Dict[str, Any]:
    exact = (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )
    terminating = _terminating_decimal(value)
    if terminating is not None:
        digits = len(terminating.split(".")[1]) if "." in terminating else 0
        if digits <= decimals:
            return {"exact": exact, "display": terminating, "approximation": "", "approximate": False}
        rounded = _rounded_decimal(value, decimals)
        return {"exact": exact, "display": rounded or terminating, "approximation": rounded, "approximate": True}
    rounded = _rounded_decimal(value, decimals)
    if prefer_decimal and rounded:
        return {"exact": exact, "display": rounded, "approximation": rounded, "approximate": True}
    return {"exact": exact, "display": exact, "approximation": rounded, "approximate": True}


def _plain(value: Fraction, decimals: int = 6) -> str:
    return str(_present(value, decimals=decimals)["display"])


def _is_integer(value: Fraction) -> bool:
    return value.denominator == 1


def _unit_label(canonical: str) -> str:
    return _UNIT_DISPLAY.get(canonical, canonical)


# ---------------------------------------------------------------------------
# Solution container
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """One candidate answer plus the check that decides whether it is trusted."""

    problem_class: str
    method: str
    headline: str
    value: Optional[Fraction] = None
    text_value: str = ""
    unit: str = ""
    steps: List[str] = field(default_factory=list)
    confidence: float = 0.6
    verified: bool = False
    verification_method: str = "none"
    verification_independent: bool = False
    prefer_decimal: bool = True
    decimals: int = 6
    model_conditional: bool = False
    assumptions_explicit: bool = False
    symbolic_exact: str = ""
    override_eligible: bool = True
    presentation_override: Dict[str, Any] = field(default_factory=dict)
    science_plan: Dict[str, Any] = field(default_factory=dict)
    science_plan_receipt: Dict[str, Any] = field(default_factory=dict)

    def answer_key(self) -> str:
        if self.value is not None:
            return f"{self.value.numerator}/{self.value.denominator}|{self.unit}"
        if self.symbolic_exact:
            return f"symbolic:{self.symbolic_exact}|{self.unit}"
        return f"text:{self.text_value.strip().lower()}"

    def answer_text(self) -> str:
        if self.value is None:
            return self.text_value
        presentation = self.presentation()
        rendered = str(presentation["display"])
        if self.unit:
            rendered = f"{rendered} {self.unit}" if not self.unit.startswith(("°", "%")) else f"{rendered}{self.unit}"
        return rendered

    def presentation(self) -> Dict[str, Any]:
        if self.value is None:
            return {
                "exact": self.symbolic_exact,
                "display": self.text_value,
                "approximation": "",
                "approximate": False,
            }
        if self.presentation_override:
            exact = self.presentation_override.get("exact")
            display = self.presentation_override.get("display")
            approximation = self.presentation_override.get("approximation")
            approximate = self.presentation_override.get("approximate")
            if (
                isinstance(exact, str)
                and exact == str(self.value)
                and isinstance(display, str)
                and 0 < len(display) <= 80
                and isinstance(approximation, str)
                and len(approximation) <= 80
                and isinstance(approximate, bool)
            ):
                return {
                    "exact": exact,
                    "display": display,
                    "approximation": approximation,
                    "approximate": approximate,
                }
        return _present(self.value, prefer_decimal=self.prefer_decimal, decimals=self.decimals)


# ---------------------------------------------------------------------------
# Problem frame
# ---------------------------------------------------------------------------

@dataclass
class _Frame:
    raw: str
    text: str
    numbers: List[Fraction]
    words: List[str]
    unit_tokens: List[str]
    has_equals: bool
    has_percent: bool
    clause_count: int


def _normalize(raw: str) -> str:
    text = _clean_text(raw).lower()
    text = _THOUSANDS_RE.sub("", text)
    text = (
        text.replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("–", "-")
        .replace("’", "'")
    )
    return text


def _build_frame(query: Any) -> _Frame:
    raw = _source_text(query)
    text = _normalize(raw)
    numbers: List[Fraction] = []
    for token in _NUMBER_RE.findall(text)[:MAX_NUMBERS]:
        parsed = _fraction(token)
        if parsed is not None:
            numbers.append(parsed)
    unit_tokens = _UNIT_TOKEN_RE.findall(text)
    return _Frame(
        raw=raw,
        text=text,
        numbers=numbers,
        words=_WORD_RE.findall(text)[:400],
        unit_tokens=unit_tokens,
        has_equals="=" in text,
        has_percent=("%" in text or "percent" in text or "per cent" in text),
        clause_count=1 + text.count(",") + text.count(";") + len(re.findall(r"\b(?:then|after that|next|and then)\b", text)),
    )


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def _solve_percent(frame: _Frame) -> Optional[Solution]:
    if not frame.has_percent:
        return None
    text = frame.text

    matches: List[Tuple[str, re.Match[str]]] = []
    for kind, pattern in (
        ("reverse", _PERCENT_OF_WHAT_RE),
        ("part_of", _PART_IS_WHAT_PERCENT_RE),
        ("what_of", _WHAT_PERCENT_OF_IS_RE),
        ("of", _PERCENT_OF_RE),
    ):
        matches.extend((kind, match) for match in pattern.finditer(text))
    unique = {
        (match.start(), match.end(), kind): (kind, match)
        for kind, match in matches
    }
    if len(unique) != 1:
        return None
    kind, match = next(iter(unique.values()))

    if kind == "reverse":
        part = _fraction(match.group("part"))
        pct = _fraction(match.group("pct"))
        if part is None or pct is None or pct == 0:
            return None
        value = _guard(part * 100 / pct)
        verified = value * pct / 100 == part
        return Solution(
            problem_class="percent",
            method="percent_reverse_whole",
            headline=f"{_plain(part)} is {_plain(pct)}% of {{answer}}.",
            value=value,
            steps=[f"whole = part * 100 / percent = {_plain(part)} * 100 / {_plain(pct)}"],
            confidence=0.88,
            verified=verified,
            verification_method="inverse_percent_reverse",
            verification_independent=True,
        )

    if kind in {"part_of", "what_of"}:
        part = _fraction(match.group("part"))
        whole = _fraction(match.group("whole"))
        if part is None or whole is None or whole == 0:
            return None
        value = _guard(part / whole * 100)
        verified = (value / 100) * whole == part
        return Solution(
            problem_class="percent",
            method="percent_of_whole",
            headline=f"{_plain(part)} is {{answer}} of {_plain(whole)}.",
            value=value,
            unit="%",
            steps=[
                f"part / whole = {_plain(part)} / {_plain(whole)}",
                f"multiply by 100 -> {_plain(value)}%",
            ],
            confidence=0.9,
            verified=verified,
            verification_method="inverse_percent_of",
            verification_independent=True,
        )

    if kind == "of":
        pct = _fraction(match.group("pct"))
        whole = _fraction(match.group("whole"))
        if pct is None or whole is None:
            return None
        value = _guard(pct / 100 * whole)
        verified = whole == 0 or (value / whole * 100 == pct)
        return Solution(
            problem_class="percent",
            method="percent_of",
            headline=f"{_plain(pct)}% of {_plain(whole)} is {{answer}}.",
            value=value,
            steps=[
                f"{_plain(pct)}% = {_plain(pct)}/100",
                f"multiply by {_plain(whole)} -> {_plain(value)}",
            ],
            confidence=0.92,
            verified=verified,
            verification_method="inverse_ratio_check",
            verification_independent=True,
        )
    return None


def _solve_percent_change(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    match = _PERCENT_CHANGE_RE.search(text)
    if match is None:
        if not frame.has_percent:
            return None
        match = _CHANGE_PERCENT_RE.search(text)
        if match is None:
            return None
    start = _fraction(match.group("start"))
    end = _fraction(match.group("end"))
    if start is None or end is None or start == 0:
        return None
    value = _guard((end - start) / start * 100)
    verified = start * (1 + value / 100) == end
    direction = "increase" if value > 0 else ("decrease" if value < 0 else "change")
    return Solution(
        problem_class="percent_change",
        method="percent_change",
        headline=f"Going from {_plain(start)} to {_plain(end)} is a {{answer}} {direction}.",
        value=value,
        unit="%",
        steps=[
            f"change = {_plain(end)} - {_plain(start)} = {_plain(end - start)}",
            f"relative = change / {_plain(start)}",
            f"as a percentage -> {_plain(value)}%",
        ],
        confidence=0.9,
        verified=verified,
        verification_method="forward_reconstruction",
        verification_independent=True,
    )


def _resolve_unit(token: str) -> Optional[Tuple[str, str]]:
    token = token.strip().lower()
    if token in _TEMPERATURE_ALIASES:
        return ("temperature", _TEMPERATURE_ALIASES[token])
    return _UNIT_ALIASES.get(token)


def _temperature_to_celsius(value: Fraction, unit: str) -> Fraction:
    if unit == "c":
        return value
    if unit == "f":
        return (value - 32) * Fraction(5, 9)
    return value - Fraction(27315, 100)


def _celsius_to_unit(value: Fraction, unit: str) -> Fraction:
    if unit == "c":
        return value
    if unit == "f":
        return value * Fraction(9, 5) + 32
    return value + Fraction(27315, 100)


def _solve_unit_conversion(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    matches = [
        *list(_CONVERT_HOWMANY_RE.finditer(text)),
        *list(_CONVERT_RE.finditer(text)),
    ]
    unique = {
        (
            match.start(),
            match.end(),
            match.group("value"),
            match.group("from"),
            match.group("to"),
        ): match
        for match in matches
    }
    if len(unique) != 1:
        return None
    match = next(iter(unique.values()))
    value = _fraction(match.group("value"))
    source = _resolve_unit(match.group("from"))
    target = _resolve_unit(match.group("to"))
    if value is None or source is None or target is None:
        return None
    if source[0] != target[0] or source[1] == target[1]:
        return None

    dimension, from_unit = source
    _, to_unit = target

    if dimension == "temperature":
        celsius = _temperature_to_celsius(value, from_unit)
        converted = _guard(_celsius_to_unit(celsius, to_unit))
        # Independent check: convert the answer back and compare to the input.
        round_trip = _celsius_to_unit(_temperature_to_celsius(converted, to_unit), from_unit)
        verified = round_trip == value
        return Solution(
            problem_class="unit_conversion",
            method="temperature_conversion",
            headline=(
                f"{_plain(value)}{_unit_label(from_unit)} is {{answer}}."
            ),
            value=converted,
            unit=_unit_label(to_unit),
            steps=[
                f"convert {_unit_label(from_unit)} to celsius -> {_plain(celsius)}°C",
                f"convert celsius to {_unit_label(to_unit)} -> {_plain(converted)}",
            ],
            confidence=0.93,
            verified=verified,
            verification_method="round_trip_conversion",
            verification_independent=True,
        )

    table = _DIMENSIONS[dimension]
    from_factor = table[from_unit]
    to_factor = table[to_unit]
    converted = _guard(value * from_factor / to_factor)

    # Two independent checks: an exact round trip, and a direction/magnitude
    # check that catches an inverted factor even when the round trip cancels.
    round_trip = converted * to_factor / from_factor == value
    if value == 0:
        direction_ok = converted == 0
    elif from_factor == to_factor:
        direction_ok = converted == value
    else:
        bigger_source = from_factor > to_factor
        bigger_result = abs(converted) > abs(value)
        direction_ok = bigger_source == bigger_result
    verified = bool(round_trip and direction_ok)

    return Solution(
        problem_class="unit_conversion",
        method="scale_conversion",
        headline=f"{_plain(value)} {_unit_label(from_unit)} is {{answer}}.",
        value=converted,
        unit=_unit_label(to_unit),
        steps=[
            f"1 {_unit_label(from_unit)} = {_plain(from_factor / to_factor)} {_unit_label(to_unit)}",
            f"{_plain(value)} * {_plain(from_factor / to_factor)} = {_plain(converted)}",
        ],
        confidence=0.94,
        verified=verified,
        verification_method="round_trip_and_direction",
        verification_independent=True,
    )


# ---------------------------------------------------------------------------
# Deliberate Reasoning v3: geometry, probability, prediction, physics, and logic
# ---------------------------------------------------------------------------

_PI_APPROX = Fraction(3_141_592_653_589_793, 1_000_000_000_000_000)


def _quoted_numeric_input_is_ambiguous(text: str) -> bool:
    matches = list(_QUOTED_SPAN_RE.finditer(text))
    quoted_numeric = any(
        _NUMBER_RE.search(match.group(0)) is not None for match in matches
    )
    remainder = list(text)
    for match in matches:
        remainder[match.start() : match.end()] = " " * (match.end() - match.start())
    unmatched_delimiter = _QUOTE_DELIMITER_RE.search("".join(remainder)) is not None
    if unmatched_delimiter and _NUMBER_RE.search(text) is not None:
        return True
    return bool(
        quoted_numeric
        and _AUTHORITATIVE_QUOTED_INPUT_RE.search(text) is None
    )


def _positive_calculation_request_count(text: str) -> int:
    """Count positive, unquoted request cues to reject silent multi-task collapse."""

    masked = _ALLOWED_REASONING_ACTION_RE.sub(
        lambda match: " " * (match.end() - match.start()),
        text,
    )
    quoted_spans = [(match.start(), match.end()) for match in _QUOTED_SPAN_RE.finditer(masked)]

    def quoted(position: int) -> bool:
        return any(start <= position < end for start, end in quoted_spans)

    count = 0
    for cue in _QUESTION_VERB_RE.finditer(masked):
        if quoted(cue.start()):
            continue
        count += 1
    return count


def _has_unconsumed_action(text: str) -> bool:
    if _HORN_PROBLEM_RE.fullmatch(text) is not None:
        return False
    masked = _ALLOWED_REASONING_ACTION_RE.sub(" ", text)
    if _UNCONSUMED_ACTION_RE.search(masked) is not None:
        return True
    request = _QUESTION_VERB_RE.search(masked)
    if request is None:
        return False
    for match in _COORDINATED_TAIL_RE.finditer(masked):
        if match.start() < request.start():
            continue
        following = match.group("next")
        if following.isdigit() or following in _SAFE_COORDINATED_TASK_WORDS:
            continue
        return True
    return False


def _has_unconsumed_trailing_content(text: str) -> bool:
    """Reject a second clause after one request unless it is a work/check modifier."""

    quoted_spans = [(match.start(), match.end()) for match in _QUOTED_SPAN_RE.finditer(text)]
    request = next(
        (
            cue
            for cue in _QUESTION_VERB_RE.finditer(text)
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
    trailing = _ALLOWED_REASONING_ACTION_RE.sub(" ", text[absolute_boundary:])
    trailing = re.sub(r"\b(?:and|please|then)\b", " ", trailing)
    return re.search(r"[a-z0-9]", trailing) is not None


def _request_clause_bounds(text: str, target: str) -> Optional[Tuple[int, int]]:
    """Return bounds for one positive, unquoted calculation request clause."""

    quoted_spans = [(match.start(), match.end()) for match in _QUOTED_SPAN_RE.finditer(text)]

    def quoted(position: int) -> bool:
        return any(start <= position < end for start, end in quoted_spans)

    for cue in _QUESTION_VERB_RE.finditer(text):
        if quoted(cue.start()):
            continue
        clause_start = max(text.rfind(mark, 0, cue.start()) for mark in ".;?!") + 1
        endings = [position for mark in ".;?!" if (position := text.find(mark, cue.end())) >= 0]
        clause_end = min(endings) if endings else len(text)
        clause_end = min(clause_end, cue.start() + 112)
        prefix = text[clause_start : cue.start()]
        segment = text[cue.start() : clause_end]
        target_match = re.search(rf"\b(?:{target})\b", segment)
        if target_match is None or quoted(cue.start() + target_match.start()):
            continue
        intent_span = prefix + " " + segment[: target_match.end()]
        if _NEGATED_REQUEST_RE.search(intent_span) or _NON_CALCULATION_INTENT_RE.search(intent_span):
            continue
        return clause_start, clause_end
    return None


def _request_clause_end(text: str, target: str) -> Optional[int]:
    bounds = _request_clause_bounds(text, target)
    return bounds[1] if bounds is not None else None


def _geometry_request_scope(text: str, target: str) -> Optional[str]:
    """Bind geometry quantities to the request or one affirmative setup clause."""

    bounds = _request_clause_bounds(text, target)
    if bounds is None:
        return None
    clause_start, clause_end = bounds
    scope_start = clause_start
    if clause_start > 0:
        previous_end = clause_start - 1
        previous_start = (
            max(text.rfind(mark, 0, previous_end) for mark in ".;?!") + 1
        )
        previous = text[previous_start:previous_end].strip()
        if (
            previous
            and _EXCLUDED_SETUP_RE.search(previous) is None
            and _AFFIRMATIVE_GEOMETRY_SETUP_RE.search(previous) is not None
        ):
            scope_start = previous_start
    return text[scope_start:clause_end].strip()


def _physics_request_scope(text: str, target: str) -> Optional[str]:
    """Bind physical quantities to one positive request and adjacent setup."""

    bounds = _request_clause_bounds(text, target)
    if bounds is None:
        return None
    clause_start, clause_end = bounds
    scope_start = clause_start
    if clause_start > 0:
        previous_end = clause_start - 1
        previous_start = (
            max(text.rfind(mark, 0, previous_end) for mark in ".;?!") + 1
        )
        previous = text[previous_start:previous_end].strip()
        quantity_patterns = (
            _PHYS_MASS_RE,
            _PHYS_ACCELERATION_RE,
            _PHYS_VOLUME_RE,
            _PHYS_SPEED_RE,
            _PHYS_RESISTANCE_RE,
            _PHYS_CURRENT_RE,
            _PHYS_VOLTAGE_RE,
        )
        affirmative_context = re.search(
            r"\b(?:object|body|material|substance|sample|circuit)\b|"
            r"\b(?:has|have|with|whose|weighs?|accelerat(?:e|es|ing)|moving|"
            r"mass|volume|speed|velocity|current|voltage|resistance)\b",
            previous,
        )
        if (
            previous
            and _EXCLUDED_SETUP_RE.search(previous) is None
            and affirmative_context is not None
            and any(pattern.search(previous) is not None for pattern in quantity_patterns)
        ):
            scope_start = previous_start
    return text[scope_start:clause_end].strip()


def _asks_for(text: str, target: str) -> bool:
    return _request_clause_end(text, target) is not None


def _unique_geometry_measure(text: str, labels: str) -> Optional[Tuple[Fraction, str]]:
    connector = r"(?:\s+(?:is|are|of|equals?))?\s*(?:=|:)?\s*"
    patterns = (
        re.compile(
            rf"\b(?:{labels})\b{connector}(?P<value>{_NUM})(?:\s*(?P<unit>{_GEO_UNIT_ALT}))?(?![a-z0-9])"
        ),
        re.compile(
            rf"(?<![a-z0-9.])(?P<value>{_NUM})(?:\s*(?P<unit>{_GEO_UNIT_ALT}))?\s+\b(?:{labels})\b"
        ),
    )
    found: List[Tuple[int, int, Fraction, str]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            value = _fraction(match.group("value"))
            if value is not None:
                found.append((match.start(), match.end(), value, (match.group("unit") or "").strip()))
    # A repeated label is ambiguous even when repeated values happen to agree.
    unique_spans = {(start, end): (value, unit) for start, end, value, unit in found}
    if len(unique_spans) != 1:
        return None
    return next(iter(unique_spans.values()))


def _geometry_unit(measures: Sequence[Tuple[Fraction, str]], *, squared: bool) -> Optional[str]:
    raw_units = [unit for _, unit in measures]
    if any(raw_units) and not all(raw_units):
        return None
    if not raw_units or not any(raw_units):
        return "square units" if squared else "units"
    canonical: List[str] = []
    for raw_unit in raw_units:
        resolved = _resolve_unit(raw_unit)
        if resolved is None or resolved[0] != "length":
            return None
        canonical.append(resolved[1])
    if len(set(canonical)) != 1:
        return None
    base = canonical[0]
    return f"{base}^2" if squared else base


def _exact_sqrt(value: Fraction) -> Optional[Fraction]:
    if value < 0:
        return None
    numerator = math.isqrt(value.numerator)
    denominator = math.isqrt(value.denominator)
    if numerator * numerator != value.numerator or denominator * denominator != value.denominator:
        return None
    return _guard(Fraction(numerator, denominator))


def _circle_symbolic_token(coefficient: Fraction) -> str:
    return "pi" if coefficient == 1 else f"{_plain(coefficient, 12)}*pi"


def _circle_symbolic_answer(coefficient: Fraction, unit: str) -> str:
    token = _circle_symbolic_token(coefficient)
    approximation = _plain(_guard(coefficient * _PI_APPROX), 6)
    return f"{token} {unit} (approximately {approximation} {unit})"


def _solve_geometry(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if _AMBIGUOUS_IN_UNIT_RE.search(text) is not None:
        return None
    asks_area = _asks_for(text, "area")
    asks_perimeter = _asks_for(text, "perimeter")
    asks_circumference = _asks_for(text, "circumference")
    asks_hypotenuse = _asks_for(text, "hypotenuse")
    asks_missing_leg = _asks_for(text, r"missing\s+(?:leg|side)")
    if sum((asks_area, asks_perimeter, asks_circumference, asks_hypotenuse, asks_missing_leg)) != 1:
        return None

    active_target = next(
        target
        for target, enabled in (
            ("area", asks_area),
            ("perimeter", asks_perimeter),
            ("circumference", asks_circumference),
            ("hypotenuse", asks_hypotenuse),
            (r"missing\s+(?:leg|side)", asks_missing_leg),
        )
        if enabled
    )
    scoped_text = _geometry_request_scope(text, active_target)
    if scoped_text is None:
        return None
    # Only the request clause or one affirmative adjacent setup sentence may
    # supply dimensions. Excluded examples and unrelated earlier values cannot.
    text = scoped_text

    shapes = {
        "rectangle": re.search(r"\brectangle\b", text) is not None,
        "triangle": re.search(r"\btriangle\b", text) is not None,
        "circle": re.search(r"\bcircle\b", text) is not None,
    }

    if asks_area and shapes["rectangle"] and sum(shapes.values()) == 1:
        length = _unique_geometry_measure(text, r"length|long")
        width = _unique_geometry_measure(text, r"width|wide")
        if length is None or width is None or length[0] <= 0 or width[0] <= 0:
            return None
        unit = _geometry_unit((length, width), squared=True)
        if unit is None:
            return None
        value = _guard(length[0] * width[0])
        verified = value / length[0] == width[0] and value / width[0] == length[0]
        return Solution(
            problem_class="geometry",
            method="rectangle_area",
            headline="The rectangle's area is {answer}.",
            value=value,
            unit=unit,
            steps=[
                f"area = length * width = {_plain(length[0])} * {_plain(width[0])}",
                f"area = {_plain(value)} {unit}",
            ],
            confidence=0.96,
            verified=verified,
            verification_method="recover_both_rectangle_sides",
            verification_independent=False,
        )

    if asks_perimeter and shapes["rectangle"] and sum(shapes.values()) == 1:
        length = _unique_geometry_measure(text, r"length|long")
        width = _unique_geometry_measure(text, r"width|wide")
        if length is None or width is None or length[0] <= 0 or width[0] <= 0:
            return None
        unit = _geometry_unit((length, width), squared=False)
        if unit is None:
            return None
        value = _guard(2 * (length[0] + width[0]))
        verified = value / 2 - length[0] == width[0]
        return Solution(
            problem_class="geometry",
            method="rectangle_perimeter",
            headline="The rectangle's perimeter is {answer}.",
            value=value,
            unit=unit,
            steps=[
                f"perimeter = 2 * (length + width) = 2 * ({_plain(length[0])} + {_plain(width[0])})",
                f"perimeter = {_plain(value)} {unit}",
            ],
            confidence=0.95,
            verified=verified,
            verification_method="recover_width_from_semiperimeter",
            verification_independent=False,
        )

    if asks_area and shapes["triangle"] and sum(shapes.values()) == 1:
        base = _unique_geometry_measure(text, r"base")
        height = _unique_geometry_measure(text, r"height|high")
        if base is None or height is None or base[0] <= 0 or height[0] <= 0:
            return None
        unit = _geometry_unit((base, height), squared=True)
        if unit is None:
            return None
        value = _guard(base[0] * height[0] / 2)
        verified = 2 * value / base[0] == height[0]
        return Solution(
            problem_class="geometry",
            method="triangle_area",
            headline="The triangle's area is {answer}.",
            value=value,
            unit=unit,
            steps=[
                f"area = base * perpendicular height / 2 = {_plain(base[0])} * {_plain(height[0])} / 2",
                f"area = {_plain(value)} {unit}",
            ],
            confidence=0.95,
            verified=verified,
            verification_method="recover_triangle_height",
            verification_independent=False,
        )

    if (asks_area or asks_circumference) and shapes["circle"] and sum(shapes.values()) == 1:
        radius = _unique_geometry_measure(text, r"radius")
        diameter = _unique_geometry_measure(text, r"diameter")
        if (radius is None) == (diameter is None):
            return None
        measure = radius or diameter
        assert measure is not None
        if measure[0] <= 0:
            return None
        unit = _geometry_unit((measure,), squared=asks_area)
        if unit is None:
            return None
        if radius is not None:
            diameter_value = _guard(2 * radius[0])
            coefficient = _guard(radius[0] * radius[0]) if asks_area else diameter_value
            verified = coefficient == (diameter_value * diameter_value / 4 if asks_area else diameter_value)
        else:
            assert diameter is not None
            radius_value = _guard(diameter[0] / 2)
            coefficient = _guard(diameter[0] * diameter[0] / 4) if asks_area else diameter[0]
            verified = coefficient == (radius_value * radius_value if asks_area else 2 * radius_value)
        quantity = "area" if asks_area else "circumference"
        formula = "pi * radius^2" if asks_area else "pi * diameter"
        return Solution(
            problem_class="geometry",
            method=f"circle_{quantity}",
            headline=f"The circle's {quantity} is {{answer}}.",
            text_value=_circle_symbolic_answer(coefficient, unit),
            unit=unit,
            steps=[f"{quantity} = {formula}", f"exact coefficient of pi = {_plain(coefficient, 12)}"],
            confidence=0.94,
            verified=verified,
            verification_method="radius_diameter_identity",
            verification_independent=False,
            symbolic_exact=_circle_symbolic_token(coefficient),
        )

    if (asks_hypotenuse or asks_missing_leg) and shapes["triangle"]:
        if re.search(r"\b(?:right(?:-angled)?\s+triangle|pythagorean)\b", text) is None:
            return None
        if asks_hypotenuse:
            leg_pattern = re.compile(
                rf"\b(?:legs?|leg\s+lengths?)\b\s*(?:are|of|=|:)?\s*"
                rf"(?P<a>{_NUM})(?:\s*(?P<au>{_GEO_UNIT_ALT}))?\s*(?:and|,)\s*"
                rf"(?P<b>{_NUM})(?:\s*(?P<bu>{_GEO_UNIT_ALT}))?(?![a-z0-9])"
            )
            matches = list(leg_pattern.finditer(text))
            if len(matches) != 1:
                return None
            match = matches[0]
            first = _fraction(match.group("a"))
            second = _fraction(match.group("b"))
            if first is None or second is None or first <= 0 or second <= 0:
                return None
            measures = ((first, match.group("au") or ""), (second, match.group("bu") or ""))
            unit = _geometry_unit(measures, squared=False)
            if unit is None:
                return None
            squared = _guard(first * first + second * second)
            value = _exact_sqrt(squared)
            if value is None:
                return None
            verified = value * value - first * first == second * second
            return Solution(
                problem_class="geometry",
                method="pythagorean_hypotenuse",
                headline="The right triangle's hypotenuse is {answer}.",
                value=value,
                unit=unit,
                steps=[
                    f"hypotenuse^2 = {_plain(first)}^2 + {_plain(second)}^2 = {_plain(squared)}",
                    f"hypotenuse = {_plain(value)} {unit}",
                ],
                confidence=0.97,
                verified=verified,
                verification_method="subtract_one_leg_square",
                verification_independent=False,
            )

        hypotenuse = _unique_geometry_measure(text, r"hypotenuse")
        leg = _unique_geometry_measure(text, r"known\s+leg|leg")
        if hypotenuse is None or leg is None or hypotenuse[0] <= leg[0] or leg[0] <= 0:
            return None
        unit = _geometry_unit((hypotenuse, leg), squared=False)
        if unit is None:
            return None
        squared = _guard(hypotenuse[0] * hypotenuse[0] - leg[0] * leg[0])
        value = _exact_sqrt(squared)
        if value is None:
            return None
        verified = value * value + leg[0] * leg[0] == hypotenuse[0] * hypotenuse[0]
        return Solution(
            problem_class="geometry",
            method="pythagorean_missing_leg",
            headline="The right triangle's missing leg is {answer}.",
            value=value,
            unit=unit,
            steps=[
                f"missing leg^2 = {_plain(hypotenuse[0])}^2 - {_plain(leg[0])}^2 = {_plain(squared)}",
                f"missing leg = {_plain(value)} {unit}",
            ],
            confidence=0.97,
            verified=verified,
            verification_method="pythagorean_sum_reconstruction",
            verification_independent=False,
        )
    return None


def _probability_request_disqualified(text: str) -> bool:
    return bool(
        _UNEQUAL_PROBABILITY_RE.search(text)
        or _UNSUPPORTED_PROBABILITY_EVENT_RE.search(text)
        or _WITHOUT_REPLACEMENT_RE.search(text)
        or _REQUEST_CANCELLATION_RE.search(text)
        or _LATE_CORRECTION_RE.search(text)
    )


def fair_probability_request_admissible(query: Any, method: str) -> bool:
    """Revalidate one single-trial fair experiment from the complete prompt."""

    raw = str(query or "")
    if not raw or len(raw) > MAX_QUERY_CHARS:
        return False
    text = _normalize(raw)
    if _probability_request_disqualified(text) or not _asks_for(text, r"probability|chance"):
        return False

    numeric_tokens = _NUMBER_RE.findall(text)
    if method == "fair_coin_single_toss":
        return bool(
            len(list(_FAIR_COIN_RE.finditer(text))) == 1
            and _REPEATED_COIN_RE.search(text) is None
            and re.search(
                r"\b(?:tosses|flips|two|three|four|five|six|seven|eight|nine|ten)\b",
                text,
            ) is None
            and len(re.findall(r"\b(?:heads?|tails?)\b", text)) == 1
            and not numeric_tokens
        )

    if method != "fair_die_equiprobable_faces":
        return False
    die_matches = list(_FAIR_DIE_RE.finditer(text))
    if len(die_matches) != 1 or _REPEATED_DIE_RE.search(text) is not None:
        return False
    die_match = die_matches[0]
    sides_token = die_match.group("sides")
    sides = int(sides_token) if sides_token is not None else 6
    if sides < 2 or sides > 1_000:
        return False
    face_matches = list(re.finditer(r"\b(?:roll|rolling)\s+(?:a\s+)?(?P<face>\d{1,3})\b", text))
    parity = [word for word in ("even", "odd") if re.search(rf"\b{word}(?:\s+number)?\b", text)]
    if face_matches and parity:
        return False
    if len(face_matches) == 1:
        face = int(face_matches[0].group("face"))
        tail = text[face_matches[0].end() : face_matches[0].end() + 24]
        expected_numeric_tokens = 1 + int(sides_token is not None)
        return bool(
            1 <= face <= sides
            and re.search(r"\b(?:or|and)\s+\d", tail) is None
            and len(numeric_tokens) == expected_numeric_tokens
        )
    if len(face_matches) == 0 and len(parity) == 1:
        return len(numeric_tokens) == int(sides_token is not None)
    return False


def _finite_probability_fraction(token: str) -> Optional[Fraction]:
    cooked = str(token or "").strip()
    try:
        if cooked.endswith("%"):
            value = Fraction(Decimal(cooked[:-1])) / 100
        elif "/" in cooked:
            numerator, denominator = cooked.split("/", 1)
            value = Fraction(int(numerator), int(denominator))
        else:
            value = Fraction(Decimal(cooked))
        value = _guard(value)
    except (ArithmeticError, InvalidOperation, ValueError, ZeroDivisionError, _ReasoningLimit):
        return None
    return value if 0 <= value <= 1 else None


def parse_finite_bernoulli_scenario(query: Any) -> Optional[Dict[str, Any]]:
    """Parse one complete, explicit finite Bernoulli scenario into canonical IR."""

    raw = str(query or "")
    if not raw or len(raw) > MAX_QUERY_CHARS:
        return None
    text = _normalize(raw)
    if (
        _probability_request_disqualified(text)
        or _NEGATED_ASSUMPTION_RE.search(text) is not None
        or _CERTAINTY_REQUEST_RE.search(text) is not None
        or _HIGH_STAKES_PREDICTION_RE.search(text) is not None
    ):
        return None
    request_match = _FINITE_BERNOULLI_REQUEST_RE.fullmatch(text)
    if request_match is None:
        return None
    model_text = request_match.group("model").strip()
    question_text = request_match.group("question").strip()
    question_match = _FINITE_BERNOULLI_QUESTION_RE.fullmatch(question_text)
    if question_match is None:
        return None

    model_kind = ""
    probability: Optional[Fraction] = None
    model_match = _FINITE_BERNOULLI_EXPLICIT_MODEL_RE.fullmatch(model_text)
    if model_match is not None:
        mode = re.sub(r"[^a-z]", "", model_match.group("mode"))
        if mode == "independent" and model_match.group("stationary") is None:
            return None
        probability = _finite_probability_fraction(model_match.group("probability"))
        trials = int(model_match.group("trials"))
        model_kind = "explicit_probability"
    else:
        coin_match = _FINITE_BERNOULLI_COIN_MODEL_RE.fullmatch(model_text)
        if coin_match is None:
            return None
        trials = int(coin_match.group("trials"))
        probability = Fraction(1, 2)
        model_kind = "fair_coin"

    count = int(question_match.group("count"))
    event = question_match.group("event").replace(" ", "_")
    outcome_token = question_match.group("outcome")
    outcome = "success" if outcome_token.startswith("success") else outcome_token.rstrip("s")
    if model_kind == "explicit_probability" and outcome != "success":
        return None
    if model_kind == "fair_coin" and outcome not in {"head", "tail"}:
        return None
    if (
        probability is None
        or not 1 <= trials <= MAX_BINOMIAL_TRIALS
        or not 0 <= count <= trials
    ):
        return None
    return {
        "schema": FINITE_BERNOULLI_SCHEMA_VERSION,
        "model": model_kind,
        "trials": trials,
        "event": event,
        "count": count,
        "outcome": outcome,
        "probability_numerator": probability.numerator,
        "probability_denominator": probability.denominator,
        "full_query_consumed": True,
    }


def _binomial_direct_event_probability(
    trials: int,
    probability: Fraction,
    event: str,
    count: int,
) -> Fraction:
    failure_probability = _guard(1 - probability)
    if event == "exactly":
        indices = range(count, count + 1)
    elif event == "at_least":
        indices = range(count, trials + 1)
    elif event == "at_most":
        indices = range(0, count + 1)
    else:
        raise ValueError("unsupported_binomial_event")
    total = Fraction(0)
    for successes in indices:
        mass = _guard(
            math.comb(trials, successes)
            * probability**successes
            * failure_probability ** (trials - successes)
        )
        total = _guard(total + mass)
    return total


def _bernoulli_convolution_distribution(
    trials: int,
    probability: Fraction,
) -> Tuple[Fraction, ...]:
    """Independently rebuild every outcome mass by repeated convolution."""

    failure_probability = _guard(1 - probability)
    distribution: List[Fraction] = [Fraction(1)]
    for _ in range(trials):
        updated = [Fraction(0) for _ in range(len(distribution) + 1)]
        for successes, mass in enumerate(distribution):
            updated[successes] = _guard(updated[successes] + mass * failure_probability)
            updated[successes + 1] = _guard(updated[successes + 1] + mass * probability)
        distribution = updated
    return tuple(distribution)


def _solve_finite_bernoulli(frame: _Frame) -> Optional[Solution]:
    scenario = parse_finite_bernoulli_scenario(frame.raw)
    if scenario is None:
        return None
    trials = int(scenario["trials"])
    count = int(scenario["count"])
    event = str(scenario["event"])
    outcome = str(scenario["outcome"])
    probability = Fraction(
        int(scenario["probability_numerator"]),
        int(scenario["probability_denominator"]),
    )

    direct = _binomial_direct_event_probability(trials, probability, event, count)
    distribution = _bernoulli_convolution_distribution(trials, probability)
    if event == "exactly":
        checked_event = distribution[count]
        checked_complement = _guard(sum(distribution[:count], Fraction(0)) + sum(distribution[count + 1 :], Fraction(0)))
        notation = f"P(X = {count})"
        direct_step = f"{notation} = C({trials}, {count}) p^{count} (1-p)^{trials - count}."
    elif event == "at_least":
        checked_event = _guard(sum(distribution[count:], Fraction(0)))
        checked_complement = _guard(sum(distribution[:count], Fraction(0)))
        notation = f"P(X >= {count})"
        direct_step = f"{notation} is the exact sum of binomial masses from {count} through {trials}."
    else:
        checked_event = _guard(sum(distribution[: count + 1], Fraction(0)))
        checked_complement = _guard(sum(distribution[count + 1 :], Fraction(0)))
        notation = f"P(X <= {count})"
        direct_step = f"{notation} is the exact sum of binomial masses from 0 through {count}."
    total_mass = _guard(sum(distribution, Fraction(0)))
    verified = bool(
        direct == checked_event
        and total_mass == 1
        and checked_event + checked_complement == 1
        and all(mass >= 0 for mass in distribution)
    )
    probability_text = (
        str(probability.numerator)
        if probability.denominator == 1
        else f"{probability.numerator}/{probability.denominator}"
    )
    event_words = event.replace("_", " ")
    outcome_words = outcome if count == 1 else {
        "success": "successes",
        "head": "heads",
        "tail": "tails",
    }[outcome]
    percent = _plain(_guard(direct * 100), 6)
    return Solution(
        problem_class="probability",
        method="finite_binomial_event_probability",
        headline=(
            "Because the exact binomial event sum applies under the stated finite "
            "independent, constant-probability model, "
            f"the probability of {event_words} {count} {outcome_words} is {{answer}} ({percent}%)."
        ),
        value=direct,
        steps=[
            f"Canonical model: n = {trials}, p = {probability_text}, event = {notation}.",
            direct_step,
            f"Independent check: repeated Bernoulli convolution rebuilt all {trials + 1} masses and they sum to 1.",
        ],
        confidence=0.99,
        verified=verified,
        verification_method="bernoulli_convolution_and_mass_check",
        verification_independent=True,
        model_conditional=True,
        assumptions_explicit=True,
    )


def _probability_solution(
    *,
    favourable: int,
    total: int,
    method: str,
    steps: Sequence[str],
) -> Optional[Solution]:
    if total <= 0 or favourable < 0 or favourable > total:
        return None
    value = _guard(Fraction(favourable, total))
    complement = _guard(Fraction(total - favourable, total))
    exact = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    percent = _plain(_guard(value * 100), 6)
    return Solution(
        problem_class="probability",
        method=method,
        headline=f"The probability is {exact} ({percent}%).",
        value=value,
        steps=list(steps),
        confidence=0.96,
        verified=value + complement == 1 and value * total == favourable,
        verification_method="complement_and_count_reconstruction",
        verification_independent=False,
    )


def _one_count(pattern: re.Pattern[str], text: str) -> Optional[int]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return None
    token = matches[0].group("a") or matches[0].group("b")
    value = _fraction(token)
    if value is None or not _is_integer(value):
        return None
    return int(value)


def _solve_probability(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if (
        not _asks_for(text, r"probability|chance")
        or _probability_request_disqualified(text)
    ):
        return None

    favourable = _one_count(_FAVOURABLE_COUNT_RE, text)
    total = _one_count(_TOTAL_COUNT_RE, text)
    if favourable is not None or total is not None:
        if (
            favourable is None
            or total is None
            or _EQUIPROBABLE_CUE_RE.search(text) is None
            or _UNEQUAL_PROBABILITY_RE.search(text) is not None
        ):
            return None
        return _probability_solution(
            favourable=favourable,
            total=total,
            method="explicit_favourable_over_total",
            steps=(
                f"favourable cases = {favourable}; total cases = {total}",
                f"probability = {favourable} / {total}",
            ),
        )

    if _FAIR_COIN_RE.search(text) is not None:
        if not fair_probability_request_admissible(frame.raw, "fair_coin_single_toss"):
            return None
        events = re.findall(r"\b(?:heads?|tails?)\b", text)
        return _probability_solution(
            favourable=1,
            total=2,
            method="fair_coin_single_toss",
            steps=("A fair coin has two equiprobable outcomes.", f"{events[0]} is one favourable outcome."),
        )

    die_match = _FAIR_DIE_RE.search(text)
    if die_match is None:
        return None
    if not fair_probability_request_admissible(frame.raw, "fair_die_equiprobable_faces"):
        return None
    sides_token = die_match.group("sides")
    sides = int(sides_token) if sides_token is not None else 6
    if sides < 2 or sides > 1_000:
        return None
    face_matches = list(re.finditer(r"\b(?:roll|rolling)\s+(?:a\s+)?(?P<face>\d{1,3})\b", text))
    parity = [word for word in ("even", "odd") if re.search(rf"\b{word}(?:\s+number)?\b", text)]
    if face_matches and parity:
        return None
    if len(face_matches) == 1:
        face = int(face_matches[0].group("face"))
        tail = text[face_matches[0].end() : face_matches[0].end() + 24]
        if re.search(r"\b(?:or|and)\s+\d", tail) or face < 1 or face > sides:
            return None
        favourable = 1
        event_step = f"Exactly one face ({face}) is favourable."
    elif len(face_matches) == 0 and len(parity) == 1:
        favourable = (sides + 1) // 2 if parity[0] == "odd" else sides // 2
        event_step = f"There are {favourable} {parity[0]} faces from 1 through {sides}."
    else:
        return None
    return _probability_solution(
        favourable=favourable,
        total=sides,
        method="fair_die_equiprobable_faces",
        steps=(f"A fair {sides}-sided die has {sides} equiprobable faces.", event_step),
    )


def _explicit_prediction_assumptions(text: str) -> bool:
    # Normalize the dotted abbreviation before extracting one punctuation-
    # bounded assumption clause. Cues from later sentences cannot be assembled
    # into a synthetic assumption.
    normalized = re.sub(r"\bi\.?i\.?d\.?\b", "iid", text)
    clauses = [match.group("body").strip() for match in _ASSUMPTION_CLAUSE_RE.finditer(normalized)]
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    if _NEGATED_ASSUMPTION_RE.search(clause) is not None:
        return False
    if _IID_CUE_RE.search(clause) is not None:
        return True
    return _INDEPENDENCE_CUE_RE.search(clause) is not None and _STATIONARY_CUE_RE.search(clause) is not None


def _solve_empirical_prediction(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if (
        _NEXT_TRIAL_RE.search(text) is None
        or not _explicit_prediction_assumptions(text)
        or _CERTAINTY_REQUEST_RE.search(text) is not None
        or _HIGH_STAKES_PREDICTION_RE.search(text) is not None
    ):
        return None
    matches = list(_BERNOULLI_COUNTS_RE.finditer(text))
    if len(matches) != 1:
        return None
    successes = int(matches[0].group("successes"))
    trials = int(matches[0].group("trials"))
    if trials <= 0 or trials > MAX_BERNOULLI_TRIALS or successes < 0 or successes > trials:
        return None
    value = _guard(Fraction(successes, trials))
    failures = trials - successes
    complement = _guard(Fraction(failures, trials))
    percent = _plain(_guard(value * 100), 6)
    verified = value * trials == successes and value + complement == 1
    return Solution(
        problem_class="prediction",
        method="empirical_bernoulli_plugin",
        headline=(
            "Under the stated independent, constant-probability Bernoulli model, "
            f"the plug-in estimate for the next trial is {percent}%. "
            "This is model-conditional, not a guarantee, and calibration has not been established."
        ),
        value=value,
        steps=[
            "Assumption gate: trials are explicitly independent with a constant success probability.",
            f"empirical success rate = {successes} / {trials} = {percent}%",
            f"complement check uses {failures} observed failures.",
        ],
        confidence=0.76,
        verified=verified,
        verification_method="count_reconstruction_and_complement",
        verification_independent=False,
        model_conditional=True,
        assumptions_explicit=True,
        override_eligible=False,
    )


def _scaled_quantity(
    text: str,
    pattern: re.Pattern[str],
    unit_factor: Callable[[str], Optional[Fraction]],
) -> Optional[Tuple[Fraction, Fraction]]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        return None
    raw = _fraction(matches[0].group("value"))
    factor = unit_factor(matches[0].group("unit").lower())
    if raw is None or factor is None:
        return None
    return raw, _guard(raw * factor)


def _mass_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"\s+", "", unit)
    if cooked in {"kg", "kilogram", "kilograms"}:
        return Fraction(1)
    if cooked in {"g", "gram", "grams"}:
        return Fraction(1, 1_000)
    if cooked in {"mg", "milligram", "milligrams"}:
        return Fraction(1, 1_000_000)
    return None


def _volume_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"[\s^]", "", unit)
    if cooked in {"m3", "cubicmeter", "cubicmeters"}:
        return Fraction(1)
    if cooked in {
        "cm3", "cubiccentimeter", "cubiccentimeters", "ml", "milliliter",
        "milliliters", "millilitre", "millilitres",
    }:
        return Fraction(1, 1_000_000)
    if cooked in {"l", "liter", "liters", "litre", "litres"}:
        return Fraction(1, 1_000)
    return None


def _speed_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"\s+", "", unit)
    if cooked in {"m/s", "meterpersecond", "meterspersecond", "metrepersecond", "metrespersecond"}:
        return Fraction(1)
    if cooked in {
        "km/h", "kph", "kmh", "kilometerperhour", "kilometersperhour",
        "kilometreperhour", "kilometresperhour",
    }:
        return Fraction(5, 18)
    return None


def _unit_factor_one(_unit: str) -> Optional[Fraction]:
    return Fraction(1)


def _resistance_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"\s+", "", unit)
    return Fraction(1_000) if cooked.startswith(("kilo", "k")) else Fraction(1)


def _current_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"\s+", "", unit)
    return Fraction(1, 1_000) if cooked.startswith(("milli", "m")) else Fraction(1)


def _voltage_factor(unit: str) -> Optional[Fraction]:
    cooked = re.sub(r"\s+", "", unit)
    return Fraction(1, 1_000) if cooked.startswith(("milli", "m")) else Fraction(1)


def _solve_physics(frame: _Frame) -> Optional[Solution]:
    full_text = frame.text
    targets = {
        "force": _asks_for(full_text, "force"),
        "density": _asks_for(full_text, "density"),
        "kinetic_energy": _asks_for(full_text, r"kinetic\s+energy|ke"),
        "voltage": _asks_for(full_text, "voltage"),
        "current": _asks_for(full_text, "current"),
        "resistance": _asks_for(full_text, "resistance"),
    }
    active = [name for name, enabled in targets.items() if enabled]
    if len(active) != 1:
        return None
    target = active[0]
    target_pattern = r"kinetic\s+energy|ke" if target == "kinetic_energy" else target
    scoped_text = _physics_request_scope(full_text, target_pattern)
    if scoped_text is None:
        return None
    text = scoped_text

    if target == "force":
        if _NEWTON_CONTEXT_RE.search(text) is None or _NON_NET_FORCE_RE.search(text) is not None:
            return None
        mass = _scaled_quantity(text, _PHYS_MASS_RE, _mass_factor)
        acceleration = _scaled_quantity(text, _PHYS_ACCELERATION_RE, _unit_factor_one)
        if mass is None or acceleration is None or mass[1] <= 0 or acceleration[1] <= 0:
            return None
        value = _guard(mass[1] * acceleration[1])
        verified = value / acceleration[1] == mass[1] and value / mass[1] == acceleration[1]
        return Solution(
            problem_class="physics",
            method="newtons_second_law_force",
            headline="Using F = ma, the force is {answer}.",
            value=value,
            unit="N",
            steps=[
                f"mass in kg = {_plain(mass[1])}; acceleration in m/s^2 = {_plain(acceleration[1])}",
                f"F = m * a = {_plain(mass[1])} * {_plain(acceleration[1])}",
            ],
            confidence=0.97,
            verified=verified,
            verification_method="recover_mass_and_acceleration",
            verification_independent=False,
        )

    if target == "density":
        if (
            _DENSITY_CONTEXT_RE.search(text) is None
            or _DENSITY_AMBIGUITY_RE.search(text) is not None
            or re.search(r"\bmass\b", text) is None
            or re.search(r"\bvolume\b", text) is None
        ):
            return None
        mass = _scaled_quantity(text, _PHYS_MASS_RE, _mass_factor)
        volume = _scaled_quantity(text, _PHYS_VOLUME_RE, _volume_factor)
        if mass is None or volume is None or mass[1] <= 0 or volume[1] <= 0:
            return None
        value = _guard(mass[1] / volume[1])
        verified = value * volume[1] == mass[1]
        return Solution(
            problem_class="physics",
            method="density_mass_over_volume",
            headline="Using density = mass / volume, the density is {answer}.",
            value=value,
            unit="kg/m^3",
            steps=[
                f"mass in kg = {_plain(mass[1])}; volume in m^3 = {_plain(volume[1], 12)}",
                f"density = {_plain(mass[1])} / {_plain(volume[1], 12)}",
            ],
            confidence=0.96,
            verified=verified,
            verification_method="mass_reconstruction_from_density_volume",
            verification_independent=False,
        )

    if target == "kinetic_energy":
        if _KINETIC_CONTEXT_RE.search(text) is None or _KINETIC_AMBIGUITY_RE.search(text) is not None:
            return None
        mass = _scaled_quantity(text, _PHYS_MASS_RE, _mass_factor)
        speed = _scaled_quantity(text, _PHYS_SPEED_RE, _speed_factor)
        if mass is None or speed is None or mass[1] <= 0 or speed[1] <= 0:
            return None
        value = _guard(mass[1] * speed[1] * speed[1] / 2)
        verified = 2 * value / (speed[1] * speed[1]) == mass[1]
        return Solution(
            problem_class="physics",
            method="kinetic_energy",
            headline="Using KE = 1/2 mv^2, the kinetic energy is {answer}.",
            value=value,
            unit="J",
            steps=[
                f"mass in kg = {_plain(mass[1])}; speed in m/s = {_plain(speed[1])}",
                f"KE = 1/2 * {_plain(mass[1])} * {_plain(speed[1])}^2",
            ],
            confidence=0.97,
            verified=verified,
            verification_method="recover_mass_from_energy_speed",
            verification_independent=False,
        )

    if (
        _OHM_LAW_CONTEXT_RE.search(text) is None
        or _SIMPLE_OHM_ELEMENT_RE.search(text) is None
        or _MULTI_COMPONENT_CIRCUIT_RE.search(text) is not None
    ):
        return None
    resistance = _scaled_quantity(text, _PHYS_RESISTANCE_RE, _resistance_factor)
    current = _scaled_quantity(text, _PHYS_CURRENT_RE, _current_factor)
    voltage = _scaled_quantity(text, _PHYS_VOLTAGE_RE, _voltage_factor)
    if target == "voltage":
        if resistance is None or current is None or resistance[1] <= 0 or current[1] <= 0 or voltage is not None:
            return None
        value = _guard(current[1] * resistance[1])
        verified = value / resistance[1] == current[1]
        method = "ohms_law_voltage"
        headline = "Using Ohm's law V = IR, the voltage is {answer}."
        unit = "V"
        steps = [f"V = I * R = {_plain(current[1])} * {_plain(resistance[1])}"]
    elif target == "current":
        if voltage is None or resistance is None or voltage[1] <= 0 or resistance[1] <= 0 or current is not None:
            return None
        value = _guard(voltage[1] / resistance[1])
        verified = value * resistance[1] == voltage[1]
        method = "ohms_law_current"
        headline = "Using Ohm's law I = V/R, the current is {answer}."
        unit = "A"
        steps = [f"I = V / R = {_plain(voltage[1])} / {_plain(resistance[1])}"]
    else:
        if voltage is None or current is None or voltage[1] <= 0 or current[1] <= 0 or resistance is not None:
            return None
        value = _guard(voltage[1] / current[1])
        verified = value * current[1] == voltage[1]
        method = "ohms_law_resistance"
        headline = "Using Ohm's law R = V/I, the resistance is {answer}."
        unit = "ohm"
        steps = [f"R = V / I = {_plain(voltage[1])} / {_plain(current[1])}"]
    return Solution(
        problem_class="physics",
        method=method,
        headline=headline,
        value=value,
        unit=unit,
        steps=steps,
        confidence=0.97,
        verified=verified,
        verification_method="ohms_law_inverse_reconstruction",
        verification_independent=False,
    )


def _parse_linear_side(side: str, variable: str) -> Optional[Tuple[Fraction, Fraction]]:
    """Collect a side of an equation into (coefficient, constant)."""

    text = side.replace(" ", "")
    if not text or len(text) > MAX_EQUATION_CHARS:
        return None
    coefficient = Fraction(0)
    constant = Fraction(0)
    index = 0
    terms = 0
    while index < len(text):
        sign = Fraction(1)
        if text[index] in "+-":
            sign = Fraction(-1) if text[index] == "-" else Fraction(1)
            index += 1
        start = index
        while index < len(text) and (text[index].isdigit() or text[index] == "."):
            index += 1
        digits = text[start:index]
        if index < len(text) and text[index] == "*":
            index += 1
        is_variable = index < len(text) and text[index] == variable
        if is_variable:
            index += 1
        if digits:
            magnitude = _fraction(digits)
            if magnitude is None:
                return None
        else:
            if not is_variable:
                return None
            magnitude = Fraction(1)
        if index < len(text) and text[index] == "/":
            index += 1
            start = index
            while index < len(text) and (text[index].isdigit() or text[index] == "."):
                index += 1
            divisor = _fraction(text[start:index])
            if divisor is None or divisor == 0:
                return None
            magnitude = magnitude / divisor
        if is_variable:
            coefficient += sign * magnitude
        else:
            constant += sign * magnitude
        terms += 1
        if terms > 12:
            return None
        if index < len(text) and text[index] not in "+-":
            return None
    return (_guard(coefficient), _guard(constant))


def _evaluate_linear_side(side: str, variable: str, value: Fraction) -> Optional[Fraction]:
    """Evaluate a side directly with the variable substituted.

    This is a second, independent implementation: it accumulates a numeric total
    instead of collecting symbolic coefficients, so a mistake in the symbolic
    collector cannot silently validate itself.
    """

    text = side.replace(" ", "")
    if not text:
        return None
    total = Fraction(0)
    index = 0
    while index < len(text):
        sign = Fraction(1)
        if text[index] in "+-":
            sign = Fraction(-1) if text[index] == "-" else Fraction(1)
            index += 1
        start = index
        while index < len(text) and (text[index].isdigit() or text[index] == "."):
            index += 1
        digits = text[start:index]
        if index < len(text) and text[index] == "*":
            index += 1
        term: Optional[Fraction]
        if index < len(text) and text[index] == variable:
            index += 1
            factor = _fraction(digits) if digits else Fraction(1)
            if factor is None:
                return None
            term = factor * value
        else:
            term = _fraction(digits) if digits else None
            if term is None:
                return None
        if index < len(text) and text[index] == "/":
            index += 1
            start = index
            while index < len(text) and (text[index].isdigit() or text[index] == "."):
                index += 1
            divisor = _fraction(text[start:index])
            if divisor is None or divisor == 0:
                return None
            term = term / divisor
        total += sign * term
        if index < len(text) and text[index] not in "+-":
            return None
    return _guard(total)


def _solve_linear_equation(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if text.count("=") != 1:
        return None
    variable_match = _EQUATION_VAR_RE.search(text)
    equation = text
    for prefix in ("solve for", "solve", "find", "what is", "calculate", "compute"):
        if equation.startswith(prefix):
            equation = equation[len(prefix):].strip()
    equation = re.sub(r"\bfor\s+[a-z]\b\s*$", "", equation).strip()
    equation = equation.rstrip("?.! ").strip()
    # Drop any leading prose that ends in a colon, e.g. "solve for x: 2x = 8".
    if ":" in equation:
        head, _, tail = equation.rpartition(":")
        if "=" in tail and "=" not in head:
            equation = tail.strip()
    if len(equation) > MAX_EQUATION_CHARS or "=" not in equation:
        return None

    letters = {char for char in equation if char.isalpha()}
    if variable_match is not None and variable_match.group("var") in letters:
        variable = variable_match.group("var")
    elif len(letters) == 1:
        variable = next(iter(letters))
    elif "x" in letters:
        variable = "x"
    else:
        return None
    if letters - {variable}:
        return None

    left_text, right_text = equation.split("=", 1)
    left = _parse_linear_side(left_text, variable)
    right = _parse_linear_side(right_text, variable)
    if left is None or right is None:
        return None
    coefficient = left[0] - right[0]
    constant = right[1] - left[1]
    if coefficient == 0:
        return None
    value = _guard(constant / coefficient)

    left_value = _evaluate_linear_side(left_text, variable, value)
    right_value = _evaluate_linear_side(right_text, variable, value)
    verified = left_value is not None and right_value is not None and left_value == right_value

    return Solution(
        problem_class="linear_equation",
        method="linear_isolation",
        headline=f"{variable} = {{answer}}.",
        value=value,
        steps=[
            f"collect terms: {_plain(coefficient)}{variable} = {_plain(constant)}",
            f"divide both sides by {_plain(coefficient)} -> {variable} = {_plain(value)}",
            f"check: left = {_plain(left_value) if left_value is not None else 'n/a'}, "
            f"right = {_plain(right_value) if right_value is not None else 'n/a'}",
        ],
        confidence=0.95,
        verified=verified,
        verification_method="substitution_recheck",
        verification_independent=True,
        prefer_decimal=False,
    )


def _solve_rate(frame: _Frame) -> Optional[Solution]:
    text = frame.text

    match = _RATE_SPEED_RE.search(text)
    if match is not None:
        distance = _fraction(match.group("dist"))
        time_value = _fraction(match.group("time"))
        dist_unit = _resolve_unit(match.group("dunit"))
        time_unit = _resolve_unit(match.group("tunit"))
        if (
            distance is None
            or time_value is None
            or time_value == 0
            or dist_unit is None
            or time_unit is None
            or dist_unit[0] != "length"
            or time_unit[0] != "time"
        ):
            return None
        speed = _guard(distance / time_value)
        unit = f"{_unit_label(dist_unit[1])} per {_unit_label(time_unit[1]).rstrip('s')}"
        verified = speed * time_value == distance
        return Solution(
            problem_class="rate",
            method="speed_from_distance_time",
            headline="The speed is {answer}.",
            value=speed,
            unit=unit,
            steps=[
                f"speed = distance / time = {_plain(distance)} / {_plain(time_value)}",
                f"speed = {_plain(speed)} {unit}",
            ],
            confidence=0.88,
            verified=verified,
            verification_method="distance_reconstruction",
            verification_independent=True,
        )

    match = _RATE_DISTANCE_RE.search(text)
    if match is not None:
        speed = _fraction(match.group("speed"))
        time_value = _fraction(match.group("time"))
        speed_unit = _resolve_unit(match.group("sunit"))
        time_unit = _resolve_unit(match.group("tunit"))
        if speed is None or time_value is None or speed_unit is None or time_unit is None:
            return None
        if speed_unit[0] != "speed" or time_unit[0] != "time":
            return None
        speed_ms = speed * _SPEED_MS[speed_unit[1]]
        seconds = time_value * _TIME_S[time_unit[1]]
        metres = _guard(speed_ms * seconds)
        if speed_unit[1] in {"mph"}:
            distance_unit, factor = "mi", _LENGTH_M["mi"]
        elif speed_unit[1] in {"km/h"}:
            distance_unit, factor = "km", _LENGTH_M["km"]
        elif speed_unit[1] in {"ft/s"}:
            distance_unit, factor = "ft", _LENGTH_M["ft"]
        elif speed_unit[1] in {"kn"}:
            distance_unit, factor = "nmi", _LENGTH_M["nmi"]
        else:
            distance_unit, factor = "m", _LENGTH_M["m"]
        distance = _guard(metres / factor)
        verified = distance * factor == speed_ms * seconds
        return Solution(
            problem_class="rate",
            method="distance_from_speed_time",
            headline="The distance is {answer}.",
            value=distance,
            unit=_unit_label(distance_unit),
            steps=[
                f"distance = speed * time = {_plain(speed)} * {_plain(time_value)}",
                f"distance = {_plain(distance)} {_unit_label(distance_unit)}",
            ],
            confidence=0.86,
            verified=verified,
            verification_method="base_unit_reconstruction",
            verification_independent=True,
        )
    return None


def _solve_work_rate(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if _TOGETHER_RE.search(text) is None:
        return None
    if not re.search(r"\b(?:how\s+long|how\s+many|time|take)\b", text):
        return None
    match = _WORK_RATE_RE.search(text)
    if match is None:
        return None
    first = _fraction(match.group("a"))
    second = _fraction(match.group("b"))
    first_unit = _resolve_unit(match.group("au"))
    second_unit = _resolve_unit(match.group("bu"))
    if (
        first is None
        or second is None
        or first <= 0
        or second <= 0
        or first_unit is None
        or second_unit is None
        or first_unit[0] != "time"
        or second_unit[0] != "time"
    ):
        return None
    base_first = first * _TIME_S[first_unit[1]]
    base_second = second * _TIME_S[second_unit[1]]
    combined_rate = 1 / base_first + 1 / base_second
    if combined_rate == 0:
        return None
    seconds = _guard(1 / combined_rate)
    display_unit = first_unit[1]
    together = _guard(seconds / _TIME_S[display_unit])
    verified = seconds * combined_rate == 1
    return Solution(
        problem_class="work_rate",
        method="combined_work_rate",
        headline="Working together they finish in {answer}.",
        value=together,
        unit=_unit_label(display_unit),
        steps=[
            f"rate one = 1/{_plain(first)}, rate two = 1/{_plain(second)}",
            "combined rate = rate one + rate two",
            f"time = 1 / combined rate = {_plain(together)} {_unit_label(display_unit)}",
        ],
        confidence=0.84,
        verified=verified,
        verification_method="unit_job_reconstruction",
        verification_independent=True,
    )


def _solve_proportion(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if _PROPORTION_CUE_RE.search(text) is None:
        return None
    match = re.search(
        rf"(?:if|when)\s+(?P<n1>{_NUM})\s+(?P<item>[a-z][a-z ]{{0,24}}?)\s+"
        rf"(?:costs?|weighs?|takes?|needs?|requires?|uses?|produces?|makes?)\s+"
        rf"(?P<n2>{_NUM})\s*(?P<unit>[a-z][a-z ]{{0,20}}?)?\s*[,.]",
        text,
    )
    if match is None:
        return None
    tail = text[match.end():]
    tail_match = re.search(
        rf"(?:how\s+(?:much|many|long)|what)\b[^0-9\n]{{0,64}}?(?P<n3>{_NUM})",
        tail,
    )
    if tail_match is None:
        return None
    first = _fraction(match.group("n1"))
    second = _fraction(match.group("n2"))
    third = _fraction(tail_match.group("n3"))
    if first is None or second is None or third is None or first == 0:
        return None
    item = _clean_text(match.group("item"), 40).strip()
    if item and item not in tail[: tail_match.end() + 40]:
        return None
    value = _guard(second * third / first)
    verified = value * first == second * third
    unit = _clean_text(match.group("unit") or "", 24).strip()
    return Solution(
        problem_class="proportion",
        method="cross_multiplication",
        headline="At the same rate the answer is {answer}.",
        value=value,
        unit=unit,
        steps=[
            f"{_plain(first)} -> {_plain(second)}",
            f"scale factor = {_plain(third)} / {_plain(first)}",
            f"answer = {_plain(second)} * {_plain(third)} / {_plain(first)} = {_plain(value)}",
        ],
        confidence=0.8,
        verified=verified,
        verification_method="cross_multiplication_check",
        verification_independent=True,
    )


def _sequence_terms(text: str) -> List[Fraction]:
    match = _SEQUENCE_LIST_RE.search(text)
    if match is None:
        return []
    terms: List[Fraction] = []
    for token in match.group(0).split(","):
        parsed = _fraction(token)
        if parsed is None:
            return []
        terms.append(parsed)
        if len(terms) > MAX_SEQUENCE_TERMS:
            return []
    return terms


def _solve_sequence(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if _SEQUENCE_CUE_RE.search(text) is None:
        return None
    terms = _sequence_terms(text)
    if len(terms) < 3:
        return None

    differences = [terms[i + 1] - terms[i] for i in range(len(terms) - 1)]
    if all(delta == differences[0] for delta in differences):
        step = differences[0]
        value = _guard(terms[-1] + step)
        verified = all(terms[i + 1] - terms[i] == step for i in range(len(terms) - 1))
        return Solution(
            problem_class="sequence",
            method="arithmetic_progression",
            headline="The next term is {answer}.",
            value=value,
            steps=[
                f"constant difference = {_plain(step)}",
                f"{_plain(terms[-1])} + {_plain(step)} = {_plain(value)}",
            ],
            confidence=0.9,
            verified=verified,
            verification_method="rule_holds_for_all_terms",
            verification_independent=True,
            prefer_decimal=False,
        )

    if all(term != 0 for term in terms[:-1]):
        ratios = [terms[i + 1] / terms[i] for i in range(len(terms) - 1)]
        if all(ratio == ratios[0] for ratio in ratios):
            ratio = ratios[0]
            value = _guard(terms[-1] * ratio)
            verified = all(terms[i] * ratio == terms[i + 1] for i in range(len(terms) - 1))
            return Solution(
                problem_class="sequence",
                method="geometric_progression",
                headline="The next term is {answer}.",
                value=value,
                steps=[
                    f"constant ratio = {_plain(ratio)}",
                    f"{_plain(terms[-1])} * {_plain(ratio)} = {_plain(value)}",
                ],
                confidence=0.9,
                verified=verified,
                verification_method="rule_holds_for_all_terms",
                verification_independent=True,
                prefer_decimal=False,
            )

    if len(terms) >= 4:
        second_differences = [differences[i + 1] - differences[i] for i in range(len(differences) - 1)]
        if all(delta == second_differences[0] for delta in second_differences):
            step = differences[-1] + second_differences[0]
            value = _guard(terms[-1] + step)
            verified = all(
                differences[i + 1] - differences[i] == second_differences[0]
                for i in range(len(differences) - 1)
            )
            return Solution(
                problem_class="sequence",
                method="quadratic_progression",
                headline="The next term is {answer}.",
                value=value,
                steps=[
                    f"second difference = {_plain(second_differences[0])}",
                    f"next difference = {_plain(step)}",
                    f"{_plain(terms[-1])} + {_plain(step)} = {_plain(value)}",
                ],
                confidence=0.82,
                verified=verified,
                verification_method="second_difference_holds",
                verification_independent=True,
                prefer_decimal=False,
            )

        if all(terms[i] + terms[i + 1] == terms[i + 2] for i in range(len(terms) - 2)):
            value = _guard(terms[-1] + terms[-2])
            return Solution(
                problem_class="sequence",
                method="additive_recurrence",
                headline="The next term is {answer}.",
                value=value,
                steps=[
                    "each term is the sum of the previous two",
                    f"{_plain(terms[-2])} + {_plain(terms[-1])} = {_plain(value)}",
                ],
                confidence=0.82,
                verified=True,
                verification_method="rule_holds_for_all_terms",
                verification_independent=True,
                prefer_decimal=False,
            )
    return None


def _statistics_values(text: str, cue_end: int) -> List[Fraction]:
    """Read the adjacent numeric list that follows a statistics cue word."""

    tail = text[cue_end:]
    match = _STAT_LIST_RE.search(tail)
    if match is None:
        return []
    # Only a short, purely structural gap may sit between the cue and its list.
    # Anything else means the numbers belong to a different clause, as in
    # "total nonsense, 5 and 7 have nothing to do with it".
    if not _STAT_GAP_RE.fullmatch(tail[: match.start()]):
        return []
    values: List[Fraction] = []
    for token in _NUMBER_RE.findall(match.group(0)):
        parsed = _fraction(token)
        if parsed is None:
            return []
        values.append(parsed)
        if len(values) > MAX_STAT_VALUES:
            return []
    return values


def _solve_statistics(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    match = _STAT_RE.search(text)
    if match is None:
        return None
    if _UNIT_TOKEN_RE.search(text) and "average" in text and "speed" in text:
        return None
    kind = match.group("kind")
    values = _statistics_values(text, match.end())
    if len(values) < 2:
        return None

    if kind in {"mean", "average"}:
        total = sum(values, Fraction(0))
        value = _guard(total / len(values))
        verified = value * len(values) == total
        return Solution(
            problem_class="statistics",
            method="arithmetic_mean",
            headline="The mean is {answer}.",
            value=value,
            steps=[
                f"sum = {_plain(total)}",
                f"count = {len(values)}",
                f"mean = {_plain(total)} / {len(values)} = {_plain(value)}",
            ],
            confidence=0.88,
            verified=verified,
            verification_method="sum_reconstruction",
            verification_independent=True,
            prefer_decimal=False,
        )

    if kind == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            value = ordered[middle]
        else:
            value = _guard((ordered[middle - 1] + ordered[middle]) / 2)
        below = sum(1 for item in ordered if item < value)
        above = sum(1 for item in ordered if item > value)
        verified = abs(below - above) <= 1
        return Solution(
            problem_class="statistics",
            method="median",
            headline="The median is {answer}.",
            value=value,
            steps=[
                f"sorted values: {', '.join(_plain(item) for item in ordered)}",
                f"median = {_plain(value)}",
            ],
            confidence=0.86,
            verified=verified,
            verification_method="balanced_split_check",
            verification_independent=True,
            prefer_decimal=False,
        )

    if kind == "range":
        value = _guard(max(values) - min(values))
        verified = min(values) + value == max(values)
        return Solution(
            problem_class="statistics",
            method="range",
            headline="The range is {answer}.",
            value=value,
            steps=[
                f"max = {_plain(max(values))}, min = {_plain(min(values))}",
                f"range = {_plain(value)}",
            ],
            confidence=0.85,
            verified=verified,
            verification_method="min_plus_range_check",
            verification_independent=True,
            prefer_decimal=False,
        )

    if kind in {"sum", "total"}:
        value = _guard(sum(values, Fraction(0)))
        running = Fraction(0)
        for item in reversed(values):
            running += item
        verified = running == value
        return Solution(
            problem_class="statistics",
            method="sum",
            headline="The total is {answer}.",
            value=value,
            steps=[f"added {len(values)} values -> {_plain(value)}"],
            confidence=0.84,
            verified=verified,
            verification_method="reverse_order_resum",
            verification_independent=True,
            prefer_decimal=False,
        )

    if kind == "mode":
        counts: Dict[Fraction, int] = {}
        for item in values:
            counts[item] = counts.get(item, 0) + 1
        best = max(counts.values())
        if best < 2:
            return None
        winners = sorted(item for item, count in counts.items() if count == best)
        if len(winners) != 1:
            return None
        value = winners[0]
        verified = sum(1 for item in values if item == value) == best
        return Solution(
            problem_class="statistics",
            method="mode",
            headline="The mode is {answer}.",
            value=value,
            steps=[f"{_plain(value)} appears {best} times"],
            confidence=0.83,
            verified=verified,
            verification_method="frequency_recount",
            verification_independent=True,
            prefer_decimal=False,
        )
    return None


def _prime_factors(number: int) -> List[int]:
    factors: List[int] = []
    remaining = number
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
            if len(factors) > 96:
                raise _ReasoningLimit("too_many_factors")
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors.append(remaining)
    return factors


def _is_prime(number: int) -> bool:
    if number < 2:
        return False
    if number in (2, 3):
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    divisor = 5
    while divisor * divisor <= number:
        if number % divisor == 0 or number % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


def _solve_number_theory(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    gcd_matches = list(_GCD_RE.finditer(text))
    lcm_matches = list(_LCM_RE.finditer(text))
    prime_matches = list(_PRIME_RE.finditer(text))
    factor_matches = list(_FACTORS_RE.finditer(text))
    if sum(map(len, (gcd_matches, lcm_matches, prime_matches, factor_matches))) != 1:
        return None

    match = gcd_matches[0] if gcd_matches else None
    if match is not None:
        first = _fraction(match.group("a"))
        second = _fraction(match.group("b"))
        if first is None or second is None or not _is_integer(first) or not _is_integer(second):
            return None
        left, right = abs(int(first)), abs(int(second))
        if left == 0 or right == 0 or max(left, right) > MAX_PRIME_CANDIDATE:
            return None
        value = math.gcd(left, right)
        verified = left % value == 0 and right % value == 0 and math.gcd(left // value, right // value) == 1
        return Solution(
            problem_class="number_theory",
            method="gcd",
            headline=f"The greatest common divisor of {left} and {right} is {{answer}}.",
            value=Fraction(value),
            steps=[f"gcd({left}, {right}) = {value}"],
            confidence=0.93,
            verified=verified,
            verification_method="divisibility_and_coprime_check",
            verification_independent=True,
            prefer_decimal=False,
        )

    match = lcm_matches[0] if lcm_matches else None
    if match is not None:
        first = _fraction(match.group("a"))
        second = _fraction(match.group("b"))
        if first is None or second is None or not _is_integer(first) or not _is_integer(second):
            return None
        left, right = abs(int(first)), abs(int(second))
        if left == 0 or right == 0 or max(left, right) > MAX_PRIME_CANDIDATE:
            return None
        value = left // math.gcd(left, right) * right
        verified = value % left == 0 and value % right == 0 and value * math.gcd(left, right) == left * right
        return Solution(
            problem_class="number_theory",
            method="lcm",
            headline=f"The least common multiple of {left} and {right} is {{answer}}.",
            value=Fraction(value),
            steps=[f"lcm = {left} * {right} / gcd({left}, {right}) = {value}"],
            confidence=0.93,
            verified=verified,
            verification_method="multiple_and_product_identity",
            verification_independent=True,
            prefer_decimal=False,
        )

    match = prime_matches[0] if prime_matches else None
    if match is not None:
        number = int(match.group("n"))
        if number > MAX_PRIME_CANDIDATE:
            return None
        prime = _is_prime(number)
        try:
            factors = _prime_factors(number) if number >= 2 else []
        except _ReasoningLimit:
            return None
        verified = (len(factors) == 1) == prime if number >= 2 else not prime
        if prime:
            statement = f"Yes, {number} is prime."
        elif factors:
            statement = f"No, {number} is not prime: {number} = {' x '.join(str(item) for item in factors)}."
        else:
            statement = f"No, {number} is not prime."
        return Solution(
            problem_class="number_theory",
            method="primality",
            headline="{answer}",
            text_value=statement,
            steps=[f"trial division up to sqrt({number})"],
            confidence=0.94,
            verified=verified,
            verification_method="factorization_agreement",
            verification_independent=True,
        )

    match = factor_matches[0] if factor_matches else None
    if match is not None:
        number = int(match.group("n"))
        if number < 2 or number > MAX_PRIME_CANDIDATE:
            return None
        try:
            factors = _prime_factors(number)
        except _ReasoningLimit:
            return None
        product = 1
        for item in factors:
            product *= item
        verified = product == number and all(_is_prime(item) for item in factors)
        statement = f"{number} = {' x '.join(str(item) for item in factors)}"
        return Solution(
            problem_class="number_theory",
            method="prime_factorization",
            headline="{answer}",
            text_value=statement,
            steps=[f"repeated division -> {len(factors)} prime factors"],
            confidence=0.92,
            verified=verified,
            verification_method="product_and_primality_check",
            verification_independent=True,
        )
    return None


def _solve_combinatorics(frame: _Frame) -> Optional[Solution]:
    text = frame.text

    match = _CHOOSE_RE.search(text)
    if match is not None:
        groups = match.groupdict()
        total = next((groups[key] for key in ("n1", "n2", "n3", "n4") if groups.get(key)), None)
        pick = next((groups[key] for key in ("r1", "r2", "r3", "r4") if groups.get(key)), None)
        if total is None or pick is None:
            return None
        n_value, r_value = int(total), int(pick)
        if n_value > MAX_CHOOSE_N or r_value > n_value:
            return None
        value = math.comb(n_value, r_value)
        verified = value == math.comb(n_value, n_value - r_value) and value >= 1
        return Solution(
            problem_class="combinatorics",
            method="combinations",
            headline=f"There are {{answer}} ways to choose {r_value} from {n_value}.",
            value=Fraction(value),
            steps=[
                f"C({n_value}, {r_value}) = {n_value}! / ({r_value}! * {n_value - r_value}!)",
                f"= {value}",
            ],
            confidence=0.9,
            verified=verified,
            verification_method="symmetry_identity",
            verification_independent=True,
            prefer_decimal=False,
        )

    match = _PERMUTE_RE.search(text)
    if match is not None:
        groups = match.groupdict()
        total = next((groups[key] for key in ("n1", "n2", "n3") if groups.get(key)), None)
        pick = next((groups[key] for key in ("r1", "r2", "r3") if groups.get(key)), None)
        if total is None or pick is None:
            return None
        n_value, r_value = int(total), int(pick)
        if n_value > MAX_CHOOSE_N or r_value > n_value:
            return None
        value = math.perm(n_value, r_value)
        verified = value == math.comb(n_value, r_value) * math.factorial(r_value)
        return Solution(
            problem_class="combinatorics",
            method="permutations",
            headline=f"There are {{answer}} ordered arrangements of {r_value} from {n_value}.",
            value=Fraction(value),
            steps=[f"P({n_value}, {r_value}) = {n_value}! / ({n_value - r_value}!) = {value}"],
            confidence=0.88,
            verified=verified,
            verification_method="combination_times_factorial",
            verification_independent=True,
            prefer_decimal=False,
        )

    match = _FACTORIAL_RE.search(text) or _FACTORIAL_WORD_RE.search(text)
    if match is not None:
        raw = match.groupdict().get("n") or match.groupdict().get("a") or match.groupdict().get("b")
        if raw is None:
            return None
        n_value = int(raw)
        if n_value > MAX_FACTORIAL_N:
            return None
        value = math.factorial(n_value)
        running = 1
        for step in range(2, n_value + 1):
            running *= step
        verified = running == value
        return Solution(
            problem_class="combinatorics",
            method="factorial",
            headline=f"{n_value}! is {{answer}}.",
            value=Fraction(value),
            steps=[f"multiply 1 through {n_value} -> {value}"],
            confidence=0.9,
            verified=verified,
            verification_method="iterative_recompute",
            verification_independent=True,
            prefer_decimal=False,
        )
    return None


def _parse_date(text: str) -> Optional[date]:
    match = _ISO_DATE_RE.search(text)
    if match is not None:
        try:
            return date(int(match.group("y")), int(match.group("m")), int(match.group("d")))
        except ValueError:
            return None
    match = _NAMED_DATE_RE.search(text)
    if match is not None:
        month_name = match.group("m1") or match.group("m2")
        day_token = match.group("d1") or match.group("d2")
        year_token = match.group("y1") or match.group("y2")
        if not month_name or not day_token or not year_token:
            return None
        try:
            return date(int(year_token), _MONTHS[month_name], int(day_token))
        except (ValueError, KeyError):
            return None
    return None


def _solve_date(frame: _Frame) -> Optional[Solution]:
    text = frame.text

    match = _DAYS_BETWEEN_RE.search(text)
    if match is not None:
        first = _parse_date(match.group("a"))
        second = _parse_date(match.group("b"))
        if first is None or second is None:
            return None
        delta_days = abs((second - first).days)
        if delta_days > MAX_DATE_DELTA_DAYS:
            return None
        weeks = match.group("unit").startswith("week")
        value = Fraction(delta_days, 7) if weeks else Fraction(delta_days)
        earlier, later = (first, second) if first <= second else (second, first)
        verified = (earlier.toordinal() + delta_days) == later.toordinal()
        label = "weeks" if weeks else "days"
        return Solution(
            problem_class="date",
            method="date_difference",
            headline=f"There are {{answer}} between {earlier.isoformat()} and {later.isoformat()}.",
            value=value,
            unit=label,
            steps=[f"{later.isoformat()} - {earlier.isoformat()} = {delta_days} days"],
            confidence=0.9,
            verified=verified,
            verification_method="ordinal_reconstruction",
            verification_independent=True,
            prefer_decimal=False,
        )

    match = _DATE_OFFSET_RE.search(text)
    if match is not None:
        anchor = _parse_date(match.group("anchor"))
        if anchor is None:
            return None
        count = int(match.group("n"))
        if match.group("unit").startswith("week"):
            count *= 7
        if count > MAX_DATE_DELTA_DAYS:
            return None
        backwards = match.group("dir") in {"before", "prior to"}
        try:
            target = date.fromordinal(anchor.toordinal() + (-count if backwards else count))
        except (ValueError, OverflowError):
            return None
        verified = abs(target.toordinal() - anchor.toordinal()) == count
        return Solution(
            problem_class="date",
            method="date_offset",
            headline="{answer}",
            text_value=f"That date is {target.isoformat()}.",
            steps=[
                f"anchor {anchor.isoformat()} {'-' if backwards else '+'} {count} days",
                f"result {target.isoformat()}",
            ],
            confidence=0.89,
            verified=verified,
            verification_method="ordinal_reconstruction",
            verification_independent=True,
        )
    return None


def _solve_interest(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    simple = _SIMPLE_INTEREST_RE.search(text) is not None
    compound = _COMPOUND_INTEREST_RE.search(text) is not None
    if not simple and not compound:
        return None
    principal_match = _CURRENCY_AMOUNT_RE.search(text)
    rate_match = re.search(rf"(?P<rate>{_NUM})\s*(?:%|percent|per\s?cent)", text)
    years_match = re.search(rf"(?P<years>{_NUM})\s*(?:years?|yrs?)", text)
    if principal_match is None or rate_match is None or years_match is None:
        return None
    principal = _fraction(principal_match.group("sym") or principal_match.group("word"))
    rate = _fraction(rate_match.group("rate"))
    years = _fraction(years_match.group("years"))
    if principal is None or rate is None or years is None or principal <= 0 or years < 0:
        return None

    if compound and not simple:
        if not _is_integer(years) or years > 60:
            return None
        periods = int(years)
        growth = 1 + rate / 100
        amount = _guard(principal * growth**periods)
        interest = _guard(amount - principal)
        running = principal
        for _ in range(periods):
            running = running * growth
        verified = running == amount
        return Solution(
            problem_class="interest",
            method="compound_interest",
            headline="The compound interest is {answer}.",
            value=interest,
            steps=[
                f"amount = {_plain(principal)} * (1 + {_plain(rate)}/100)^{periods}",
                f"amount = {_plain(amount)}",
                f"interest = amount - principal = {_plain(interest)}",
            ],
            confidence=0.85,
            verified=verified,
            verification_method="iterative_growth_recompute",
            verification_independent=True,
        )

    interest = _guard(principal * rate / 100 * years)
    verified = principal == 0 or years == 0 or (interest / years / principal * 100 == rate)
    return Solution(
        problem_class="interest",
        method="simple_interest",
        headline="The simple interest is {answer}.",
        value=interest,
        steps=[
            f"interest = principal * rate * time = {_plain(principal)} * {_plain(rate)}% * {_plain(years)}",
            f"interest = {_plain(interest)}",
        ],
        confidence=0.86,
        verified=verified,
        verification_method="inverse_rate_recovery",
        verification_independent=True,
    )


def _solve_sum_difference(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    match = _SUM_DIFF_RE.search(text) or _DIFF_SUM_RE.search(text)
    if match is None:
        return None
    total = _fraction(match.group("sum"))
    difference = _fraction(match.group("diff"))
    if total is None or difference is None:
        return None
    larger = _guard((total + difference) / 2)
    smaller = _guard((total - difference) / 2)
    verified = larger + smaller == total and larger - smaller == difference
    return Solution(
        problem_class="sum_difference",
        method="sum_and_difference",
        headline="{answer}",
        text_value=f"The two numbers are {_plain(larger)} and {_plain(smaller)}.",
        steps=[
            f"larger = (sum + difference) / 2 = ({_plain(total)} + {_plain(difference)}) / 2 = {_plain(larger)}",
            f"smaller = (sum - difference) / 2 = {_plain(smaller)}",
        ],
        confidence=0.87,
        verified=verified,
        verification_method="both_constraints_recheck",
        verification_independent=True,
    )


def _parse_horn_problem(
    text: str,
) -> Optional[Tuple[Tuple[str, ...], Tuple[Tuple[Tuple[str, ...], str], ...], str]]:
    """Parse one explicit, bounded positive-Horn entailment problem.

    The section markers and operators are part of the trust boundary. Free-form
    prose, negation, disjunction, quantifiers, and implicit world knowledge are
    intentionally outside this grammar.
    """

    match = _HORN_PROBLEM_RE.fullmatch(text)
    if match is None:
        return None

    def valid_atom(atom: str) -> bool:
        return _HORN_ATOM_RE.fullmatch(atom) is not None and atom not in _HORN_RESERVED_ATOMS

    facts = tuple(part.strip() for part in match.group("facts").split(","))
    if (
        not facts
        or len(facts) > MAX_LOGIC_ATOMS
        or any(not valid_atom(atom) for atom in facts)
        or len(set(facts)) != len(facts)
    ):
        return None

    raw_rules = tuple(part.strip() for part in match.group("rules").split(";"))
    if not (1 <= len(raw_rules) <= MAX_LOGIC_RULES) or any(not part for part in raw_rules):
        return None

    parsed_rules: List[Tuple[Tuple[str, ...], str]] = []
    seen_rules = set()
    all_atoms = set(facts)
    for raw_rule in raw_rules:
        if raw_rule.count("->") != 1:
            return None
        raw_antecedents, raw_consequent = (part.strip() for part in raw_rule.split("->", 1))
        antecedents = tuple(part.strip() for part in raw_antecedents.split("&"))
        if (
            not (1 <= len(antecedents) <= MAX_LOGIC_ANTECEDENTS)
            or any(not valid_atom(atom) for atom in antecedents)
            or len(set(antecedents)) != len(antecedents)
            or not valid_atom(raw_consequent)
        ):
            return None
        canonical_rule = (tuple(sorted(antecedents)), raw_consequent)
        if canonical_rule in seen_rules:
            return None
        seen_rules.add(canonical_rule)
        parsed_rules.append(canonical_rule)
        all_atoms.update(antecedents)
        all_atoms.add(raw_consequent)

    query = match.group("query")
    if not valid_atom(query):
        return None
    all_atoms.add(query)
    if len(all_atoms) > MAX_LOGIC_ATOMS:
        return None

    # Canonical ordering makes proof selection invariant to fact, antecedent,
    # and rule ordering without changing the Horn theory.
    return (
        tuple(sorted(facts)),
        tuple(sorted(parsed_rules, key=lambda item: (item[1], item[0]))),
        query,
    )


def _horn_forward_closure(
    facts: Sequence[str],
    rules: Sequence[Tuple[Tuple[str, ...], str]],
) -> Tuple[frozenset[str], Dict[str, Tuple[str, ...]]]:
    """Compute the least Horn model and retain one canonical derivation."""

    known = set(facts)
    derivations: Dict[str, Tuple[str, ...]] = {}
    for _ in range(MAX_LOGIC_ATOMS + 1):
        changed = False
        for antecedents, consequent in rules:
            if consequent not in known and all(atom in known for atom in antecedents):
                known.add(consequent)
                derivations[consequent] = antecedents
                changed = True
        if not changed:
            return frozenset(known), derivations
    raise _ReasoningLimit("logic_closure_exceeded")


def _horn_finite_model_entailment(
    facts: Sequence[str],
    rules: Sequence[Tuple[Tuple[str, ...], str]],
    query: str,
) -> Tuple[bool, int]:
    """Independently check entailment by enumerating every bounded model."""

    atoms = sorted(
        set(facts)
        | {query}
        | {atom for antecedents, consequent in rules for atom in (*antecedents, consequent)}
    )
    if len(atoms) > MAX_LOGIC_ATOMS:
        raise _ReasoningLimit("logic_model_space_exceeded")
    positions = {atom: index for index, atom in enumerate(atoms)}
    fact_mask = sum(1 << positions[atom] for atom in facts)
    query_mask = 1 << positions[query]
    rule_masks = [
        (
            sum(1 << positions[atom] for atom in antecedents),
            1 << positions[consequent],
        )
        for antecedents, consequent in rules
    ]

    satisfying_models = 0
    counterexample_found = False
    for assignment in range(1 << len(atoms)):
        if assignment & fact_mask != fact_mask:
            continue
        if any(
            assignment & antecedent_mask == antecedent_mask
            and assignment & consequent_mask == 0
            for antecedent_mask, consequent_mask in rule_masks
        ):
            continue
        satisfying_models += 1
        if assignment & query_mask == 0:
            counterexample_found = True
    return satisfying_models > 0 and not counterexample_found, satisfying_models


def _horn_proof_steps(
    query: str,
    facts: Sequence[str],
    derivations: Mapping[str, Tuple[str, ...]],
) -> List[str]:
    """Render only the canonical derivations needed by the queried atom."""

    fact_set = set(facts)
    emitted = set()
    visiting = set()
    steps: List[str] = []

    def visit(atom: str) -> None:
        if atom in fact_set or atom in emitted or atom in visiting:
            return
        antecedents = derivations.get(atom)
        if antecedents is None:
            return
        visiting.add(atom)
        for antecedent in antecedents:
            visit(antecedent)
        visiting.remove(atom)
        joined = " and ".join(antecedents)
        steps.append(f"Apply {joined} -> {atom}; infer {atom}.")
        emitted.add(atom)

    visit(query)
    return steps


def _solve_logical_entailment(frame: _Frame) -> Optional[Solution]:
    parsed = _parse_horn_problem(frame.text)
    if parsed is None:
        return None
    facts, rules, query = parsed

    closure, derivations = _horn_forward_closure(facts, rules)
    forward_entailed = query in closure
    model_entailed, satisfying_models = _horn_finite_model_entailment(facts, rules, query)
    verified = satisfying_models > 0 and forward_entailed == model_entailed

    steps = [f"Start from the supplied facts: {', '.join(facts)}."]
    if forward_entailed:
        if query in facts:
            steps.append(f"The query atom {query} is an explicit fact.")
        else:
            proof_steps = _horn_proof_steps(query, facts, derivations)
            visible_budget = MAX_STEPS - 2
            if len(proof_steps) > visible_budget:
                visible_budget -= 1
            steps.extend(proof_steps[:visible_budget])
            omitted = len(proof_steps) - visible_budget
            if omitted > 0:
                steps.append(f"Continue {omitted} additional validated derivations to reach {query}.")
        steps.append(
            f"Countercheck: all {satisfying_models} satisfying finite models make {query} true."
        )
        status = "entailed"
        text_value = f"Entailed: {query} follows from the supplied facts and rules."
    else:
        steps.append(f"Forward chaining cannot derive {query}.")
        steps.append(
            f"Countercheck: among {satisfying_models} satisfying finite models, at least one keeps {query} false."
        )
        status = "not_entailed"
        text_value = f"Not entailed: {query} does not follow from the supplied facts and rules."

    return Solution(
        problem_class="logical_entailment",
        method="bounded_horn_entailment",
        headline="{answer}",
        text_value=text_value,
        steps=steps,
        confidence=0.98,
        verified=verified,
        verification_method="finite_model_entailment_check",
        verification_independent=True,
        model_conditional=True,
        assumptions_explicit=True,
        symbolic_exact=f"{status}:{query}",
    )


_STATE_COUNT_CANONICAL = {
    "item": "items",
    "widget": "widgets",
    "token": "tokens",
    "point": "points",
}
_STATE_LABEL_DIMENSIONS = {
    "volume": "volume",
    "mass": "mass",
    "distance": "length",
    "inventory": "count",
}


def _state_label_matches_dimension(label: Optional[str], dimension: str) -> bool:
    if not label or label not in _STATE_LABEL_DIMENSIONS:
        return True
    expected = _STATE_LABEL_DIMENSIONS[label]
    return dimension.startswith("count:") if expected == "count" else dimension == expected


def _parse_state_quantity(match: re.Match[str]) -> Optional[Tuple[Fraction, str, str, Fraction]]:
    """Normalize one explicitly unit-labelled quantity to a dimension base."""

    raw = _fraction(match.group("value"))
    unit_token = match.group("unit").strip().lower()
    if raw is None:
        return None

    count_key = unit_token[:-1] if unit_token.endswith("s") else unit_token
    if count_key in _STATE_COUNT_CANONICAL:
        canonical = _STATE_COUNT_CANONICAL[count_key]
        return raw, f"count:{canonical}", canonical, Fraction(1)

    resolved = _resolve_unit(unit_token)
    if resolved is None:
        return None
    dimension, canonical = resolved
    factor = _DIMENSIONS.get(dimension, {}).get(canonical)
    if factor is None:
        # Affine temperatures are intentionally excluded: applying a relative
        # change to a temperature scale would not be invariant under conversion.
        return None
    return raw, dimension, canonical, factor


def _parse_state_transition(
    clause: str,
    *,
    expected_dimension: str,
) -> Optional[Tuple[str, str, Fraction, Fraction, str]]:
    """Return one unambiguous operation from one temporal evidence clause."""

    if not clause or len(clause) > 240:
        return None
    increases = _STATE_UP_RE.search(clause) is not None
    decreases = _STATE_DOWN_RE.search(clause) is not None
    if increases == decreases:
        # Neither direction, or contradictory directions in the same clause.
        return None

    direction = "up" if increases else "down"
    percentages = list(_STATE_PERCENT_RE.finditer(clause))
    quantities = list(_STATE_QUANTITY_RE.finditer(clause))
    if len(percentages) == 1 and not quantities:
        percent = _fraction(percentages[0].group("value"))
        if percent is None or percent <= 0 or percent > 1_000:
            return None
        if direction == "down" and percent >= 100:
            # A zero/negative state is not invertible, so it cannot pass the
            # reverse-evidence reconstruction used below.
            return None
        factor = 1 + percent / 100 if direction == "up" else 1 - percent / 100
        return "scale", direction, _guard(factor), percent, "%"

    if len(quantities) != 1 or percentages:
        return None
    parsed = _parse_state_quantity(quantities[0])
    if parsed is None:
        return None
    raw, dimension, canonical, factor = parsed
    if raw <= 0 or dimension != expected_dimension:
        return None
    return "delta", direction, _guard(raw * factor), raw, _unit_label(canonical)


def _solve_quantity_transition(frame: _Frame) -> Optional[Solution]:
    """Solve a bounded ordered state plan and reconstruct it in reverse.

    This is deliberately not a general story-problem parser. The initial state,
    every operation, its direction, and the final target must all be explicit.
    Each temporal clause supplies exactly one piece of transition evidence.
    """

    text = frame.text
    transitions = list(_STATE_TRANSITION_RE.finditer(text))
    targets = list(_STATE_FINAL_TARGET_RE.finditer(text))
    if not (2 <= len(transitions) <= MAX_PERCENT_CHAIN_OPS) or len(targets) != 1:
        return None
    target = targets[0]
    if target.start() <= transitions[-1].start():
        return None

    target_clause_start = max(text.rfind(mark, 0, target.start()) for mark in ".;?!") + 1
    if _NEGATED_REQUEST_RE.search(text[target_clause_start : target.end()]) is not None:
        return None
    quoted_spans = [(match.start(), match.end()) for match in _QUOTED_SPAN_RE.finditer(text)]
    if any(start <= target.start() < end for start, end in quoted_spans):
        return None

    base_segment = text[: transitions[0].start()]
    starts = list(_STATE_START_RE.finditer(base_segment))
    if len(starts) != 1 or any(start <= starts[0].start() < end for start, end in quoted_spans):
        return None
    base_candidates = [
        match
        for match in _STATE_QUANTITY_RE.finditer(base_segment)
        if match.start() >= starts[0].end()
    ]
    if len(base_candidates) != 1 or base_candidates[0].start() - starts[0].end() > 24:
        return None
    base = _parse_state_quantity(base_candidates[0])
    if base is None:
        return None
    base_raw, dimension, canonical, base_factor = base
    if (
        base_raw <= 0
        or not _state_label_matches_dimension(starts[0].group("label"), dimension)
        or not _state_label_matches_dimension(target.group("label"), dimension)
    ):
        return None

    clauses: List[str] = []
    for index, cue in enumerate(transitions):
        end = transitions[index + 1].start() if index + 1 < len(transitions) else len(text)
        clauses.append(text[cue.end() : end].strip())

    operations: List[Tuple[str, str, Fraction, Fraction, str]] = []
    for clause in clauses:
        operation = _parse_state_transition(clause, expected_dimension=dimension)
        if operation is None:
            return None
        operations.append(operation)

    display_unit = _unit_label(canonical)
    initial = _guard(base_raw * base_factor)
    current = initial
    steps = [f"Start with {_plain(base_raw)} {display_unit} from the explicit initial-state evidence."]
    for index, (kind, direction, operand, shown, shown_unit) in enumerate(operations, start=1):
        if kind == "scale":
            current = _guard(current * operand)
            action = "increase" if direction == "up" else "decrease"
            evidence = f"{_plain(shown)}% {action}"
        else:
            current = _guard(current + operand if direction == "up" else current - operand)
            action = "add" if direction == "up" else "remove"
            evidence = f"{action} {_plain(shown)} {shown_unit}"
        if current < 0:
            return None
        steps.append(
            f"Step {index}: {evidence} -> {_plain(_guard(current / base_factor))} {display_unit}."
        )

    # Countercheck the chosen plan by traversing the evidence in reverse with
    # inverse operations. This is separate from the forward state accumulator.
    reconstructed = current
    for kind, direction, operand, _shown, _shown_unit in reversed(operations):
        if kind == "scale":
            if operand <= 0:
                return None
            reconstructed = _guard(reconstructed / operand)
        else:
            reconstructed = _guard(
                reconstructed - operand if direction == "up" else reconstructed + operand
            )
    verified = reconstructed == initial
    value = _guard(current / base_factor)
    steps.append(
        "Countercheck: reverse every transition and recover the exact initial state."
    )
    return Solution(
        problem_class="quantity_transition",
        method="ordered_quantity_transitions",
        headline="The final quantity is {answer}.",
        value=value,
        unit=display_unit,
        steps=steps,
        confidence=0.93,
        verified=verified,
        verification_method="reverse_state_reconstruction",
        verification_independent=True,
    )


def _solve_percent_chain(frame: _Frame) -> Optional[Solution]:
    text = frame.text
    if not frame.has_percent:
        return None
    base_match = _CURRENCY_AMOUNT_RE.search(text)
    if base_match is None:
        return None
    base = _fraction(base_match.group("sym") or base_match.group("word"))
    if base is None or base <= 0:
        return None

    operations: List[Tuple[Fraction, str]] = []
    for match in _PERCENT_OP_RE.finditer(text):
        if match.start() < base_match.end():
            continue
        pct = _fraction(match.group("pct"))
        if pct is None:
            continue
        word = (match.group("word") or "").strip()
        if not word:
            window = text[match.end(): match.end() + 28]
            for candidate in _PERCENT_DOWN_WORDS | _PERCENT_UP_WORDS:
                if re.search(rf"\b{re.escape(candidate)}\b", window):
                    word = candidate
                    break
        if word in _PERCENT_DOWN_WORDS:
            operations.append((pct, "down"))
        elif word in _PERCENT_UP_WORDS:
            operations.append((pct, "up"))
        else:
            return None
        if len(operations) > MAX_PERCENT_CHAIN_OPS:
            return None
    if len(operations) < 2:
        return None

    value = base
    steps = [f"start at {_plain(base)}"]
    for pct, direction in operations:
        factor = (1 - pct / 100) if direction == "down" else (1 + pct / 100)
        value = _guard(value * factor)
        steps.append(
            f"apply {_plain(pct)}% {'decrease' if direction == 'down' else 'increase'} -> {_plain(value)}"
        )

    # Independent check: replay the same ordered operations as explicit
    # add/subtract deltas rather than multiplicative factors.
    replay = base
    for pct, direction in operations:
        delta = replay * pct / 100
        replay = replay - delta if direction == "down" else replay + delta
    verified = replay == value

    return Solution(
        problem_class="percent_chain",
        method="ordered_percent_operations",
        headline="The final amount is {answer}.",
        value=value,
        steps=steps[:MAX_STEPS],
        confidence=0.78,
        verified=verified,
        verification_method="delta_replay",
        verification_independent=False,
    )


_SCIENCE_PLAN_SCHEMA_VERSION = "supermix-science-plan-v1"
_SCIENCE_PLAN_ENGINE_VERSION = "supermix-science-plan-engine-v1"
_SCIENCE_REGISTRY_VERSION = "supermix-science-formula-registry-v1"
_SCIENCE_RECEIPT_SCHEMA_VERSION = "supermix-science-plan-receipt-v1"
_SCIENCE_VERIFICATION_METHOD = "registry_dimension_domain_and_substitution"
_SCIENCE_CHECK_KEYS = (
    "registry_integrity",
    "plan_integrity",
    "input_bindings",
    "dimensions",
    "domain",
    "substitution",
)
_SCIENCE_REASON_ALLOWLIST = frozenset({
    "",
    "verified_science_plan",
    "empty_query",
    "query_too_long",
    "invalid_query_text",
    "high_stakes_or_open_world",
    "prompt_control_or_mixed_request",
    "missing_explicit_assumption",
    "ambiguous_scenario",
    "multiple_targets",
    "unsupported_target",
    "missing_or_ambiguous_quantity",
    "mixed_or_unconsumed_request",
    "invalid_quantity_domain",
    "invalid_plan",
    "verification_failed",
    "numeric_limit",
})
_SCIENCE_FORMULAS: Dict[str, Dict[str, Any]] = {
    "constant_acceleration.final_velocity": {
        "scenario": "constant_acceleration",
        "target": "final_velocity",
        "unit": "m/s",
        "inputs": ("u", "a", "t"),
        "headline": (
            "Because v = u + a*t under the stated constant-acceleration model, "
            "the verified final velocity is {answer}."
        ),
    },
    "constant_acceleration.displacement": {
        "scenario": "constant_acceleration",
        "target": "displacement",
        "unit": "m",
        "inputs": ("u", "a", "t"),
        "headline": (
            "Because s = u*t + (a*t^2)/2 under the stated constant-acceleration model, "
            "the verified displacement is {answer}."
        ),
    },
    "ideal_gas.pressure": {
        "scenario": "ideal_gas",
        "target": "pressure",
        "unit": "Pa",
        "inputs": ("V", "n", "T"),
        "headline": (
            "Because P*V = n*R*T under the stated ideal-gas model, "
            "the verified pressure is {answer}."
        ),
    },
    "ideal_gas.volume": {
        "scenario": "ideal_gas",
        "target": "volume",
        "unit": "m^3",
        "inputs": ("P", "n", "T"),
        "headline": (
            "Because P*V = n*R*T under the stated ideal-gas model, "
            "the verified volume is {answer}."
        ),
    },
    "ideal_gas.temperature": {
        "scenario": "ideal_gas",
        "target": "temperature",
        "unit": "K",
        "inputs": ("P", "V", "n"),
        "headline": (
            "Because P*V = n*R*T under the stated ideal-gas model, "
            "the verified temperature is {answer}."
        ),
    },
    "ideal_gas.amount": {
        "scenario": "ideal_gas",
        "target": "amount",
        "unit": "mol",
        "inputs": ("P", "V", "T"),
        "headline": (
            "Because P*V = n*R*T under the stated ideal-gas model, "
            "the verified amount is {answer}."
        ),
    },
}
_SCIENCE_SCENARIOS = frozenset({"constant_acceleration", "ideal_gas"})
_SCIENCE_TARGETS = frozenset(
    str(formula["target"]) for formula in _SCIENCE_FORMULAS.values()
)
_SCIENCE_SYMBOLS = frozenset({"u", "a", "t", "P", "V", "n", "T"})
_SCIENCE_EXACT_RE = re.compile(r"-?(?:0|[1-9]\d*)(?:/[1-9]\d*)?")
_SCIENCE_DECIMAL_RE = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?",
    re.IGNORECASE,
)
_SCIENCE_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SCIENCE_PLAN_MODULE: Any = None


def _science_authority() -> Dict[str, bool]:
    return {
        "controls_compute": False,
        "controls_routes": False,
        "controls_interaction_strategy": False,
        "controls_tools": False,
        "controls_permissions": False,
        "controls_safety": False,
    }


def _science_authority_is_safe(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        value.get(key) is False for key in _science_authority()
    )


def _science_sha256(value: Any) -> str:
    if isinstance(value, str) and _SCIENCE_SHA256_RE.fullmatch(value):
        return value
    return ""


def _load_science_plan_module() -> Any:
    """Load the exact sibling module, including under file-based imports."""

    global _SCIENCE_PLAN_MODULE
    if _SCIENCE_PLAN_MODULE is not None:
        return _SCIENCE_PLAN_MODULE

    sibling = Path(__file__).resolve().with_name("science_plan.py")
    if not sibling.is_file():
        return None
    module_name = f"{__name__.replace('.', '_')}__science_plan_sibling"
    spec = importlib.util.spec_from_file_location(module_name, sibling)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        return None
    _SCIENCE_PLAN_MODULE = module
    return module


def _science_registry_sha256(module: Any = None) -> str:
    loaded = module if module is not None else _load_science_plan_module()
    return _science_sha256(getattr(loaded, "SCIENCE_FORMULA_REGISTRY_SHA256", ""))


def _empty_science_plan_diagnostics() -> Dict[str, Any]:
    return {
        "schema_version": _SCIENCE_PLAN_SCHEMA_VERSION,
        "engine_version": _SCIENCE_PLAN_ENGINE_VERSION,
        "registry_version": _SCIENCE_REGISTRY_VERSION,
        "registry_sha256": _science_registry_sha256(),
        "attempted": False,
        "solved": False,
        "override_allowed": False,
        "scenario": "",
        "target": "",
        "formula_id": "",
        "reason": "",
        "verification_passed": False,
        "verification_independent": False,
        "model_conditional": False,
        "assumptions_explicit": False,
        "calibration_claimed": False,
        "quantities": 0,
        "steps": 0,
        "authority": _science_authority(),
    }


def _empty_science_plan_receipt() -> Dict[str, Any]:
    return {
        "schema_version": _SCIENCE_RECEIPT_SCHEMA_VERSION,
        "decision": "abstained",
        "scenario": "",
        "target": "",
        "formula_ids": [],
        "registry_version": _SCIENCE_REGISTRY_VERSION,
        "registry_sha256": _science_registry_sha256(),
        "query_sha256": "",
        "plan_sha256": "",
        "input_spans": [],
        "checks": {key: False for key in _SCIENCE_CHECK_KEYS},
        "epistemics": {
            "model_conditional": False,
            "assumptions_explicit": False,
            "calibration_claimed": False,
        },
        "diagnostic_only": True,
        "authority": _science_authority(),
    }


def _bounded_science_count(value: Any, limit: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(limit, value))


def _sanitize_science_plan_diagnostics(raw: Any, module: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return _empty_science_plan_diagnostics()
    scenario = raw.get("scenario") if raw.get("scenario") in _SCIENCE_SCENARIOS else ""
    target = raw.get("target") if raw.get("target") in _SCIENCE_TARGETS else ""
    formula_id = raw.get("formula_id") if raw.get("formula_id") in _SCIENCE_FORMULAS else ""
    reason = raw.get("reason") if raw.get("reason") in _SCIENCE_REASON_ALLOWLIST else ""
    return {
        "schema_version": (
            _SCIENCE_PLAN_SCHEMA_VERSION
            if raw.get("schema_version") == _SCIENCE_PLAN_SCHEMA_VERSION
            else ""
        ),
        "engine_version": (
            _SCIENCE_PLAN_ENGINE_VERSION
            if raw.get("engine_version") == _SCIENCE_PLAN_ENGINE_VERSION
            else ""
        ),
        "registry_version": (
            _SCIENCE_REGISTRY_VERSION
            if raw.get("registry_version") == _SCIENCE_REGISTRY_VERSION
            else ""
        ),
        "registry_sha256": (
            _science_registry_sha256(module)
            if raw.get("registry_sha256") == _science_registry_sha256(module)
            else ""
        ),
        "attempted": raw.get("attempted") is True,
        "solved": raw.get("solved") is True,
        "override_allowed": raw.get("override_allowed") is True,
        "scenario": scenario,
        "target": target,
        "formula_id": formula_id,
        "reason": reason,
        "verification_passed": raw.get("verification_passed") is True,
        "verification_independent": raw.get("verification_independent") is True,
        "model_conditional": raw.get("model_conditional") is True,
        "assumptions_explicit": raw.get("assumptions_explicit") is True,
        "calibration_claimed": raw.get("calibration_claimed") is True,
        "quantities": _bounded_science_count(raw.get("quantities"), 8),
        "steps": _bounded_science_count(raw.get("steps"), 4),
        "authority": _science_authority(),
    }


def _sanitize_science_plan_receipt(raw: Any, module: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return _empty_science_plan_receipt()
    formula_ids_raw = raw.get("formula_ids")
    formula_ids = []
    if isinstance(formula_ids_raw, list) and len(formula_ids_raw) == 1:
        formula_id = formula_ids_raw[0]
        if formula_id in _SCIENCE_FORMULAS:
            formula_ids = [formula_id]

    input_spans = []
    raw_spans = raw.get("input_spans")
    if isinstance(raw_spans, list):
        for raw_span in raw_spans[:8]:
            if not isinstance(raw_span, Mapping):
                continue
            symbol = raw_span.get("symbol")
            start = raw_span.get("start")
            end = raw_span.get("end")
            digest = _science_sha256(raw_span.get("sha256"))
            if (
                symbol in _SCIENCE_SYMBOLS
                and isinstance(start, int)
                and not isinstance(start, bool)
                and isinstance(end, int)
                and not isinstance(end, bool)
                and 0 <= start < end <= MAX_QUERY_CHARS
                and digest
            ):
                input_spans.append({
                    "symbol": symbol,
                    "start": start,
                    "end": end,
                    "sha256": digest,
                })

    raw_checks = raw.get("checks") if isinstance(raw.get("checks"), Mapping) else {}
    raw_epistemics = (
        raw.get("epistemics") if isinstance(raw.get("epistemics"), Mapping) else {}
    )
    scenario = raw.get("scenario") if raw.get("scenario") in _SCIENCE_SCENARIOS else ""
    target = raw.get("target") if raw.get("target") in _SCIENCE_TARGETS else ""
    return {
        "schema_version": (
            _SCIENCE_RECEIPT_SCHEMA_VERSION
            if raw.get("schema_version") == _SCIENCE_RECEIPT_SCHEMA_VERSION
            else ""
        ),
        "decision": raw.get("decision") if raw.get("decision") in {"verified", "abstained"} else "",
        "scenario": scenario,
        "target": target,
        "formula_ids": formula_ids,
        "registry_version": (
            _SCIENCE_REGISTRY_VERSION
            if raw.get("registry_version") == _SCIENCE_REGISTRY_VERSION
            else ""
        ),
        "registry_sha256": (
            _science_registry_sha256(module)
            if raw.get("registry_sha256") == _science_registry_sha256(module)
            else ""
        ),
        "query_sha256": _science_sha256(raw.get("query_sha256")),
        "plan_sha256": _science_sha256(raw.get("plan_sha256")),
        "input_spans": input_spans,
        "checks": {key: raw_checks.get(key) is True for key in _SCIENCE_CHECK_KEYS},
        "epistemics": {
            "model_conditional": raw_epistemics.get("model_conditional") is True,
            "assumptions_explicit": raw_epistemics.get("assumptions_explicit") is True,
            "calibration_claimed": raw_epistemics.get("calibration_claimed") is True,
        },
        "diagnostic_only": raw.get("diagnostic_only") is True,
        "authority": _science_authority(),
    }


def _science_answer(raw: Any, expected_unit: str) -> Optional[Tuple[Fraction, Dict[str, Any]]]:
    if not isinstance(raw, Mapping) or raw.get("unit") != expected_unit:
        return None
    exact = raw.get("exact")
    display = raw.get("display")
    approximation = raw.get("approximation")
    approximate = raw.get("approximate")
    if (
        not isinstance(exact, str)
        or len(exact) > 160
        or _SCIENCE_EXACT_RE.fullmatch(exact) is None
        or not isinstance(display, str)
        or not 0 < len(display) <= 80
        or _SCIENCE_DECIMAL_RE.fullmatch(display) is None
        or not isinstance(approximation, str)
        or len(approximation) > 80
        or not isinstance(approximate, bool)
    ):
        return None
    exponent_match = re.search(r"e(?P<exponent>[+-]?\d+)$", display, re.IGNORECASE)
    if exponent_match is not None and abs(int(exponent_match.group("exponent"))) > 1_300:
        return None
    try:
        value = _guard(Fraction(exact))
        display_value = _guard(Fraction(display))
    except (ArithmeticError, ValueError, ZeroDivisionError, _ReasoningLimit):
        return None
    if str(value) != exact:
        return None
    if approximate:
        if (
            approximation != display
            or _SCIENCE_DECIMAL_RE.fullmatch(approximation) is None
            or (value == 0 and display_value != 0)
            or (
                value != 0
                and abs(display_value - value) * 100_000_000_000 > abs(value)
            )
        ):
            return None
    elif approximation or display_value != value:
        return None
    return value, {
        "exact": exact,
        "display": display,
        "approximation": approximation,
        "approximate": approximate,
    }


def _solve_scientific_scenario(frame: _Frame) -> Optional[Solution]:
    module = _load_science_plan_module()
    if module is None:
        return None
    required_api = (
        "parse_science_scenario",
        "execute_science_plan",
        "solve_science_scenario",
        "science_plan_diagnostics",
    )
    if not all(callable(getattr(module, name, None)) for name in required_api):
        return None
    if (
        getattr(module, "SCIENCE_PLAN_SCHEMA_VERSION", None) != _SCIENCE_PLAN_SCHEMA_VERSION
        or getattr(module, "SCIENCE_PLAN_ENGINE_VERSION", None) != _SCIENCE_PLAN_ENGINE_VERSION
        or getattr(module, "SCIENCE_FORMULA_REGISTRY_VERSION", None) != _SCIENCE_REGISTRY_VERSION
        or getattr(module, "SCIENCE_PLAN_RECEIPT_SCHEMA_VERSION", None)
        != _SCIENCE_RECEIPT_SCHEMA_VERSION
        or not _science_registry_sha256(module)
    ):
        return None
    try:
        raw_result = module.solve_science_scenario(frame.raw)
        raw_diagnostics = module.science_plan_diagnostics(raw_result)
    except Exception:
        return None
    if not isinstance(raw_result, Mapping) or not isinstance(raw_diagnostics, Mapping):
        return None

    formula_id = raw_result.get("formula_id")
    formula = _SCIENCE_FORMULAS.get(formula_id)
    if formula is None:
        return None
    scenario = str(formula["scenario"])
    target = str(formula["target"])
    if (
        raw_result.get("schema_version") != _SCIENCE_PLAN_SCHEMA_VERSION
        or raw_result.get("engine_version") != _SCIENCE_PLAN_ENGINE_VERSION
        or raw_result.get("registry_version") != _SCIENCE_REGISTRY_VERSION
        or raw_result.get("registry_sha256") != _science_registry_sha256(module)
        or raw_result.get("attempted") is not True
        or raw_result.get("solved") is not True
        or raw_result.get("override_allowed") is not True
        or raw_result.get("reason") != "verified_science_plan"
        or raw_result.get("scenario") != scenario
        or raw_result.get("target") != target
        or not _science_authority_is_safe(raw_result.get("authority"))
    ):
        return None

    verification = raw_result.get("verification")
    epistemics = raw_result.get("epistemics")
    if not isinstance(verification, Mapping) or not isinstance(epistemics, Mapping):
        return None
    checks = verification.get("checks")
    if (
        verification.get("checked") is not True
        or verification.get("passed") is not True
        or verification.get("method") != _SCIENCE_VERIFICATION_METHOD
        or verification.get("independent") is not False
        or not isinstance(checks, Mapping)
        or any(checks.get(key) is not True for key in _SCIENCE_CHECK_KEYS)
        or epistemics.get("model_conditional") is not True
        or epistemics.get("assumptions_explicit") is not True
        or epistemics.get("calibration_claimed") is not False
    ):
        return None

    parsed_answer = _science_answer(raw_result.get("answer"), str(formula["unit"]))
    if parsed_answer is None:
        return None
    value, presentation = parsed_answer

    diagnostics = _sanitize_science_plan_diagnostics(raw_diagnostics, module)
    if (
        diagnostics["schema_version"] != _SCIENCE_PLAN_SCHEMA_VERSION
        or diagnostics["engine_version"] != _SCIENCE_PLAN_ENGINE_VERSION
        or diagnostics["registry_version"] != _SCIENCE_REGISTRY_VERSION
        or not diagnostics["registry_sha256"]
        or diagnostics["attempted"] is not True
        or diagnostics["solved"] is not True
        or diagnostics["override_allowed"] is not True
        or diagnostics["scenario"] != scenario
        or diagnostics["target"] != target
        or diagnostics["formula_id"] != formula_id
        or diagnostics["reason"] != "verified_science_plan"
        or diagnostics["verification_passed"] is not True
        or diagnostics["verification_independent"] is not False
        or diagnostics["model_conditional"] is not True
        or diagnostics["assumptions_explicit"] is not True
        or diagnostics["calibration_claimed"] is not False
        or not _science_authority_is_safe(raw_diagnostics.get("authority"))
    ):
        return None

    raw_receipt = raw_result.get("receipt")
    receipt = _sanitize_science_plan_receipt(raw_receipt, module)
    expected_query_sha256 = hashlib.sha256(frame.raw.strip().encode("utf-8")).hexdigest()
    if (
        not isinstance(raw_receipt, Mapping)
        or receipt["schema_version"] != _SCIENCE_RECEIPT_SCHEMA_VERSION
        or receipt["decision"] != "verified"
        or receipt["scenario"] != scenario
        or receipt["target"] != target
        or receipt["formula_ids"] != [formula_id]
        or receipt["registry_version"] != _SCIENCE_REGISTRY_VERSION
        or not receipt["registry_sha256"]
        or receipt["query_sha256"] != expected_query_sha256
        or not receipt["plan_sha256"]
        or [span["symbol"] for span in receipt["input_spans"]] != list(formula["inputs"])
        or any(span["end"] > len(frame.raw) for span in receipt["input_spans"])
        or any(receipt["checks"].get(key) is not True for key in _SCIENCE_CHECK_KEYS)
        or receipt["epistemics"] != {
            "model_conditional": True,
            "assumptions_explicit": True,
            "calibration_claimed": False,
        }
        or receipt["diagnostic_only"] is not True
        or not _science_authority_is_safe(raw_receipt.get("authority"))
    ):
        return None

    return Solution(
        problem_class="scientific_scenario",
        method=str(formula_id),
        headline=str(formula["headline"]),
        value=value,
        unit=str(formula["unit"]),
        steps=[
            f"Apply allowlisted formula {formula_id} in canonical SI units.",
            "Check registry integrity, bindings, dimensions, domain, and substitution.",
        ],
        confidence=0.96,
        verified=True,
        verification_method=_SCIENCE_VERIFICATION_METHOD,
        verification_independent=False,
        model_conditional=True,
        assumptions_explicit=True,
        override_eligible=True,
        presentation_override=presentation,
        science_plan=diagnostics,
        science_plan_receipt=receipt,
    )


_SOLVERS: Tuple[Tuple[str, Callable[[_Frame], Optional[Solution]]], ...] = (
    ("logical_entailment", _solve_logical_entailment),
    ("linear_equation", _solve_linear_equation),
    ("geometry", _solve_geometry),
    ("physics", _solve_physics),
    ("scientific_scenario", _solve_scientific_scenario),
    ("finite_bernoulli", _solve_finite_bernoulli),
    ("empirical_prediction", _solve_empirical_prediction),
    ("probability", _solve_probability),
    ("unit_conversion", _solve_unit_conversion),
    ("percent", _solve_percent),
    ("percent_change", _solve_percent_change),
    ("quantity_transition", _solve_quantity_transition),
    ("percent_chain", _solve_percent_chain),
    ("number_theory", _solve_number_theory),
    ("combinatorics", _solve_combinatorics),
    ("sequence", _solve_sequence),
    ("statistics", _solve_statistics),
    ("date", _solve_date),
    ("rate", _solve_rate),
    ("work_rate", _solve_work_rate),
    ("interest", _solve_interest),
    ("proportion", _solve_proportion),
    ("sum_difference", _solve_sum_difference),
)

_DIAGNOSTIC_CLASS_ALLOWLIST = frozenset({
    "percent", "percent_change", "unit_conversion", "linear_equation", "rate",
    "work_rate", "proportion", "sequence", "statistics", "number_theory",
    "combinatorics", "date", "interest", "sum_difference", "percent_chain",
    "geometry", "probability", "prediction", "physics", "quantity_transition",
    "logical_entailment", "scientific_scenario",
})
_DIAGNOSTIC_METHOD_ALLOWLIST = frozenset({
    "percent_reverse_whole", "percent_of_whole", "percent_of", "temperature_conversion",
    "scale_conversion", "rectangle_area", "rectangle_perimeter", "triangle_area",
    "circle_area", "circle_circumference", "pythagorean_hypotenuse",
    "pythagorean_missing_leg", "explicit_favourable_over_total", "fair_coin_single_toss",
    "fair_die_equiprobable_faces", "finite_binomial_event_probability",
    "empirical_bernoulli_plugin", "newtons_second_law_force",
    "density_mass_over_volume", "kinetic_energy", "ohms_law_voltage", "ohms_law_current",
    "ohms_law_resistance", "linear_isolation", "speed_from_distance_time",
    "distance_from_speed_time", "combined_work_rate", "cross_multiplication",
    "arithmetic_progression", "geometric_progression", "quadratic_progression",
    "additive_recurrence", "arithmetic_mean", "median", "range", "sum", "mode", "gcd",
    "lcm", "primality", "prime_factorization", "combinations", "permutations", "factorial",
    "date_difference", "date_offset", "compound_interest", "simple_interest",
    "sum_and_difference", "ordered_percent_operations", "percent_change",
    "ordered_quantity_transitions", "bounded_horn_entailment",
    "constant_acceleration.final_velocity", "constant_acceleration.displacement",
    "ideal_gas.pressure", "ideal_gas.volume", "ideal_gas.temperature", "ideal_gas.amount",
})


# ---------------------------------------------------------------------------
# Adaptive budget
# ---------------------------------------------------------------------------

def _complexity_score(frame: _Frame) -> float:
    score = 0.0
    score += min(len(frame.numbers), 8) * 0.06
    score += min(frame.clause_count - 1, 6) * 0.09
    score += min(len(frame.unit_tokens), 6) * 0.05
    if frame.has_equals:
        score += 0.12
    if frame.has_percent:
        score += 0.08
    if len(frame.words) > 28:
        score += 0.12
    if re.search(r"\b(?:then|after that|finally|and then|step by step)\b", frame.text):
        score += 0.15
    finite_bernoulli = parse_finite_bernoulli_scenario(frame.raw)
    if finite_bernoulli is not None:
        score += 0.18
        score += min(int(finite_bernoulli["trials"]), 100) * 0.002
    horn_problem = _parse_horn_problem(frame.text)
    if horn_problem is not None:
        facts, rules, _query = horn_problem
        score += 0.08
        score += min(len(facts), 4) * 0.02
        score += min(len(rules), 6) * 0.07
    return round(min(1.0, score), 4)


def _resolve_tier(requested: str, complexity: float) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized in {"fast", "deep"}:
        return normalized
    return "deep" if complexity >= 0.28 else "fast"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def frame_problem(query: Any) -> Dict[str, Any]:
    """Describe the request as a bounded, prompt-free problem frame."""

    frame = _build_frame(query)
    complexity = _complexity_score(frame)
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "engine_version": REASONING_ENGINE_VERSION,
        "numbers_found": len(frame.numbers),
        "unit_tokens_found": len(frame.unit_tokens),
        "clause_count": int(frame.clause_count),
        "has_equation": bool(frame.has_equals),
        "has_percent": bool(frame.has_percent),
        "has_probability_cue": bool(re.search(r"\b(?:probability|chance|fair\s+coin|fair\s+die)\b", frame.text)),
        "has_prediction_cue": bool(_NEXT_TRIAL_RE.search(frame.text)),
        "has_explicit_prediction_assumptions": bool(_explicit_prediction_assumptions(frame.text)),
        "has_finite_bernoulli_scenario": parse_finite_bernoulli_scenario(frame.raw) is not None,
        "has_logic_cue": _parse_horn_problem(frame.text) is not None,
        "word_count": len(frame.words),
        "complexity": complexity,
        "recommended_tier": _resolve_tier("auto", complexity),
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }


def _empty_result(reason: str, frame: Optional[_Frame] = None, tier: str = "fast") -> Dict[str, Any]:
    complexity = _complexity_score(frame) if frame is not None else 0.0
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "engine_version": REASONING_ENGINE_VERSION,
        "attempted": False,
        "solved": False,
        "override_allowed": False,
        "problem_class": "",
        "method": "",
        "reason": reason,
        "answer": {"exact": "", "display": "", "approximation": "", "approximate": False, "unit": ""},
        "text": "",
        "steps": [],
        "verification": {
            "checked": False,
            "passed": False,
            "method": "none",
            "independent": False,
        },
        "epistemics": {
            "model_conditional": False,
            "assumptions_explicit": False,
            "calibration_claimed": False,
        },
        "science_plan": {},
        "science_plan_receipt": {},
        "consensus": {"paths": 0, "agreeing": 0, "conflicting": False, "classes": []},
        "budget": {
            "tier": tier,
            "complexity": complexity,
            "solvers_considered": len(_SOLVERS),
            "solvers_run": 0,
            "solver_limit": min(len(_SOLVERS), MAX_SOLVER_INVOCATIONS),
            "early_exit": False,
            "all_solvers_exhausted": False,
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }


def solve_problem(
    query: Any,
    *,
    tier: str = "auto",
    prompt_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Attempt a verified solution for a bounded natural-language problem.

    The result is deterministic and JSON-safe. `override_allowed` is the only
    flag a caller should use to decide whether the computed answer may replace
    a retrieved response: it requires a solved problem, a passing verification,
    and no disagreement between applicable solvers.

    `prompt_profile` is retained for API compatibility, but is deliberately
    authority-free: it cannot relax applicability, verification, or override
    rules and is not inspected by this deterministic engine.
    """

    _ = prompt_profile

    raw = str(query or "")
    if len(raw) > MAX_QUERY_CHARS:
        return _empty_result("query_too_long")
    text = _clean_text(raw)
    if not text:
        return _empty_result("empty_query")

    frame = _build_frame(raw)
    complexity = _complexity_score(frame)
    resolved_tier = _resolve_tier(tier, complexity)
    limit = min(len(_SOLVERS), MAX_SOLVER_INVOCATIONS)

    if (
        _REQUEST_CANCELLATION_RE.search(frame.text) is not None
        or _LATE_CORRECTION_RE.search(frame.text) is not None
    ):
        return _empty_result("ambiguous_or_superseded_request", frame, resolved_tier)
    if (
        _UNTRUSTED_PROBLEM_DATA_RE.search(frame.text) is not None
        or _EXCLUDED_SETUP_RE.search(frame.text) is not None
        or _quoted_numeric_input_is_ambiguous(frame.text)
    ):
        return _empty_result("untrusted_problem_data", frame, resolved_tier)
    if _positive_calculation_request_count(frame.text) > 1:
        return _empty_result("multiple_calculation_requests", frame, resolved_tier)
    if (
        _has_unconsumed_action(frame.text)
        or _has_unconsumed_trailing_content(frame.text)
    ):
        return _empty_result("mixed_or_unconsumed_request", frame, resolved_tier)

    symbolic_probability = (
        _FAIR_COIN_RE.search(frame.text) is not None
        or _FAIR_DIE_RE.search(frame.text) is not None
        or parse_finite_bernoulli_scenario(frame.raw) is not None
    )
    symbolic_logic = _parse_horn_problem(frame.text) is not None
    if not frame.numbers and not frame.has_equals and not symbolic_probability and not symbolic_logic:
        result = _empty_result("no_quantities", frame, resolved_tier)
        return result

    solutions: List[Solution] = []
    solvers_run = 0
    for _, solver in _SOLVERS[:limit]:
        solvers_run += 1
        try:
            candidate = solver(frame)
        except (_ReasoningLimit, ArithmeticError, ValueError, OverflowError, InvalidOperation, KeyError, IndexError):
            candidate = None
        if candidate is None:
            continue
        solutions.append(candidate)

    if not solutions:
        result = _empty_result("no_applicable_solver", frame, resolved_tier)
        result["attempted"] = True
        result["budget"]["solvers_run"] = solvers_run
        result["budget"]["all_solvers_exhausted"] = solvers_run == limit
        return result

    ordered = sorted(
        solutions,
        key=lambda item: (not item.verified, -item.confidence, item.method),
    )
    chosen = ordered[0]

    agreeing = sum(
        1
        for item in solutions
        if item.answer_key() == chosen.answer_key()
        and item.problem_class == chosen.problem_class
        and item.method == chosen.method
    )
    conflicting = any(
        (
            item.answer_key() != chosen.answer_key()
            or item.problem_class != chosen.problem_class
            or item.method != chosen.method
        )
        and item.verified == chosen.verified
        for item in solutions
    )

    presentation = chosen.presentation()
    answer_text = chosen.answer_text()
    headline = chosen.headline.replace("{answer}", answer_text)
    if chosen.value is not None and presentation["approximate"] and len(str(presentation["exact"])) <= 24:
        headline = f"{headline.rstrip('.')} (exact value {presentation['exact']})."

    override_allowed = bool(chosen.verified and chosen.override_eligible and not conflicting)
    if not chosen.verified:
        reason = "unverified_solution"
    elif not chosen.override_eligible:
        reason = "verified_non_overriding_estimate"
    elif conflicting:
        reason = "verified_conflict"
    else:
        reason = "verified_solution"

    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "engine_version": REASONING_ENGINE_VERSION,
        "attempted": True,
        "solved": True,
        "override_allowed": override_allowed,
        "problem_class": chosen.problem_class,
        "method": chosen.method,
        "reason": reason,
        "answer": {
            "exact": str(presentation["exact"]),
            "display": str(presentation["display"]),
            "approximation": str(presentation["approximation"]),
            "approximate": bool(presentation["approximate"]),
            "unit": chosen.unit,
        },
        "text": headline,
        "steps": [str(step) for step in chosen.steps[:MAX_STEPS]],
        "verification": {
            "checked": True,
            "passed": bool(chosen.verified),
            "method": chosen.verification_method,
            "independent": bool(chosen.verification_independent),
        },
        "epistemics": {
            "model_conditional": bool(chosen.model_conditional),
            "assumptions_explicit": bool(chosen.assumptions_explicit),
            "calibration_claimed": False,
        },
        "science_plan": dict(chosen.science_plan),
        "science_plan_receipt": dict(chosen.science_plan_receipt),
        "consensus": {
            "paths": len(solutions),
            "agreeing": agreeing,
            "conflicting": bool(conflicting),
            "classes": sorted({item.problem_class for item in solutions}),
        },
        "budget": {
            "tier": resolved_tier,
            "complexity": complexity,
            "solvers_considered": len(_SOLVERS),
            "solvers_run": solvers_run,
            "solver_limit": limit,
            "early_exit": False,
            "all_solvers_exhausted": solvers_run == limit,
        },
        "authority": {
            "controls_compute": False,
            "controls_routes": False,
            "controls_interaction_strategy": False,
        },
    }


def render_reasoning_answer(result: Mapping[str, Any], *, include_steps: bool = False) -> str:
    """Render a solved result as one plain sentence, optionally with steps."""

    if not isinstance(result, Mapping) or not result.get("solved"):
        return ""
    text = str(result.get("text") or "").strip()
    if not text:
        return ""
    if not include_steps:
        return text
    steps = [str(step).strip() for step in (result.get("steps") or []) if str(step).strip()]
    if not steps:
        return text
    body = "\n".join(f"- {step}" for step in steps[:MAX_STEPS])
    return f"{text}\n\nHow I got there:\n{body}"


def _diagnostic_allow(value: Any, allowlist: frozenset[str]) -> str:
    return value if isinstance(value, str) and value in allowlist else ""


def _diagnostic_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(MAX_SOLVER_INVOCATIONS, value))


def reasoning_diagnostics(result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return privacy-safe metadata: no prompt text, no computed answer."""

    if not isinstance(result, Mapping):
        return {
            "schema_version": REASONING_SCHEMA_VERSION,
            "engine_version": REASONING_ENGINE_VERSION,
            "attempted": False,
            "solved": False,
            "override_allowed": False,
            "problem_class": "",
            "method": "",
            "verified": False,
            "verification_independent": False,
            "model_conditional": False,
            "assumptions_explicit": False,
            "conflicting": False,
            "paths": 0,
            "tier": "",
            "solvers_run": 0,
        }
    verification = result.get("verification") if isinstance(result.get("verification"), Mapping) else {}
    epistemics = result.get("epistemics") if isinstance(result.get("epistemics"), Mapping) else {}
    consensus = result.get("consensus") if isinstance(result.get("consensus"), Mapping) else {}
    budget = result.get("budget") if isinstance(result.get("budget"), Mapping) else {}
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "engine_version": REASONING_ENGINE_VERSION,
        "attempted": bool(result.get("attempted")),
        "solved": bool(result.get("solved")),
        "override_allowed": bool(result.get("override_allowed")),
        "problem_class": _diagnostic_allow(result.get("problem_class"), _DIAGNOSTIC_CLASS_ALLOWLIST),
        "method": _diagnostic_allow(result.get("method"), _DIAGNOSTIC_METHOD_ALLOWLIST),
        "verified": bool(verification.get("passed")),
        "verification_independent": bool(verification.get("independent")),
        "model_conditional": bool(epistemics.get("model_conditional")),
        "assumptions_explicit": bool(epistemics.get("assumptions_explicit")),
        "conflicting": bool(consensus.get("conflicting")),
        "paths": _diagnostic_count(consensus.get("paths")),
        "tier": _diagnostic_allow(budget.get("tier"), frozenset({"fast", "deep"})),
        "solvers_run": _diagnostic_count(budget.get("solvers_run")),
    }


__all__ = [
    "FINITE_BERNOULLI_SCHEMA_VERSION",
    "REASONING_ENGINE_VERSION",
    "REASONING_SCHEMA_VERSION",
    "Solution",
    "fair_probability_request_admissible",
    "frame_problem",
    "parse_finite_bernoulli_scenario",
    "reasoning_diagnostics",
    "render_reasoning_answer",
    "solve_problem",
]
