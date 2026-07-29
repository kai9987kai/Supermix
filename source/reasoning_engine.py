"""Deliberate reasoning v1 for Supermix.

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
* Compute is adaptive but bounded: an easy request stops at the first verified
  path (`fast`), a complex request explores every applicable path and requires
  agreement (`deep`). The chosen tier is advisory metadata only; this module
  has no authority over model routing, adaptive exit, or permissions.
* Diagnostics never contain the raw prompt or reconstructed prompt text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple


REASONING_SCHEMA_VERSION = "supermix-reasoning-v1"
REASONING_ENGINE_VERSION = "supermix-reasoning-engine-v1"

MAX_QUERY_CHARS = 2000
MAX_NUMBERS = 32
MAX_STEPS = 10
MAX_RESULT_BITS = 4096
MAX_LITERAL_DIGITS = 40
MAX_SOLVER_INVOCATIONS = 16
MAX_SEQUENCE_TERMS = 16
MAX_STAT_VALUES = 64
MAX_FACTORIAL_N = 20
MAX_CHOOSE_N = 200
MAX_PRIME_CANDIDATE = 10**12
MAX_DATE_DELTA_DAYS = 400_000
MAX_EQUATION_CHARS = 160
MAX_PERCENT_CHAIN_OPS = 4


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
_PRIME_RE = re.compile(rf"is\s+(?P<n>\d+)\s+(?:a\s+)?prime")
_FACTORS_RE = re.compile(rf"(?:prime\s+factor(?:s|ization|ise|ize)?|factorise|factorize)\D{{0,16}}?(?P<n>\d+)")

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


# ---------------------------------------------------------------------------
# Bounded numeric helpers
# ---------------------------------------------------------------------------

def _clean_text(value: Any, limit: int = MAX_QUERY_CHARS) -> str:
    return _WS_RE.sub(" ", str(value or "")).strip()[: max(0, int(limit))]


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

    def answer_key(self) -> str:
        if self.value is not None:
            return f"{self.value.numerator}/{self.value.denominator}|{self.unit}"
        return f"text:{self.text_value.strip().lower()}"

    def answer_text(self) -> str:
        if self.value is None:
            return self.text_value
        presentation = _present(self.value, prefer_decimal=self.prefer_decimal, decimals=self.decimals)
        rendered = str(presentation["display"])
        if self.unit:
            rendered = f"{rendered} {self.unit}" if not self.unit.startswith(("°", "%")) else f"{rendered}{self.unit}"
        return rendered

    def presentation(self) -> Dict[str, Any]:
        if self.value is None:
            return {"exact": "", "display": self.text_value, "approximation": "", "approximate": False}
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
    raw = _clean_text(query)
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

    reverse = _PERCENT_OF_WHAT_RE.search(text)
    if reverse is not None:
        part = _fraction(reverse.group("part"))
        pct = _fraction(reverse.group("pct"))
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

    match = _PART_IS_WHAT_PERCENT_RE.search(text) or _WHAT_PERCENT_OF_IS_RE.search(text)
    if match is not None:
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

    match = _PERCENT_OF_RE.search(text)
    if match is not None:
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
    match = _CONVERT_HOWMANY_RE.search(text) or _CONVERT_RE.search(text)
    if match is None:
        return None
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

    match = _GCD_RE.search(text)
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

    match = _LCM_RE.search(text)
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

    match = _PRIME_RE.search(text)
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

    match = _FACTORS_RE.search(text)
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


_SOLVERS: Tuple[Tuple[str, Callable[[_Frame], Optional[Solution]]], ...] = (
    ("linear_equation", _solve_linear_equation),
    ("unit_conversion", _solve_unit_conversion),
    ("percent", _solve_percent),
    ("percent_change", _solve_percent_change),
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
        "numbers_found": len(frame.numbers),
        "unit_tokens_found": len(frame.unit_tokens),
        "clause_count": int(frame.clause_count),
        "has_equation": bool(frame.has_equals),
        "has_percent": bool(frame.has_percent),
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
        "consensus": {"paths": 0, "agreeing": 0, "conflicting": False, "classes": []},
        "budget": {
            "tier": tier,
            "complexity": complexity,
            "solvers_considered": len(_SOLVERS),
            "solvers_run": 0,
            "solver_limit": min(len(_SOLVERS), MAX_SOLVER_INVOCATIONS),
            "early_exit": False,
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
    """

    text = _clean_text(query)
    if not text:
        return _empty_result("empty_query")

    frame = _build_frame(text)
    complexity = _complexity_score(frame)
    resolved_tier = _resolve_tier(tier, complexity)
    limit = min(len(_SOLVERS), MAX_SOLVER_INVOCATIONS)

    if not frame.numbers and not frame.has_equals:
        result = _empty_result("no_quantities", frame, resolved_tier)
        return result

    solutions: List[Solution] = []
    solvers_run = 0
    early_exit = False
    for _, solver in _SOLVERS[:limit]:
        solvers_run += 1
        try:
            candidate = solver(frame)
        except (_ReasoningLimit, ArithmeticError, ValueError, OverflowError, InvalidOperation, KeyError, IndexError):
            candidate = None
        if candidate is None:
            continue
        solutions.append(candidate)
        # Fast tier stops at the first self-verified path. Deep tier keeps going
        # so that disagreement between applicable solvers can be detected.
        if resolved_tier == "fast" and candidate.verified:
            early_exit = True
            break

    if not solutions:
        result = _empty_result("no_applicable_solver", frame, resolved_tier)
        result["attempted"] = True
        result["budget"]["solvers_run"] = solvers_run
        return result

    ordered = sorted(
        solutions,
        key=lambda item: (not item.verified, -item.confidence, item.method),
    )
    chosen = ordered[0]

    keys = [item.answer_key() for item in solutions]
    agreeing = sum(1 for key in keys if key == chosen.answer_key())
    conflicting = any(
        item.answer_key() != chosen.answer_key() and item.verified == chosen.verified
        for item in solutions
    )

    presentation = chosen.presentation()
    answer_text = chosen.answer_text()
    headline = chosen.headline.replace("{answer}", answer_text)
    if chosen.value is not None and presentation["approximate"] and len(str(presentation["exact"])) <= 24:
        headline = f"{headline.rstrip('.')} (exact value {presentation['exact']})."

    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "engine_version": REASONING_ENGINE_VERSION,
        "attempted": True,
        "solved": True,
        "override_allowed": bool(chosen.verified and not conflicting),
        "problem_class": chosen.problem_class,
        "method": chosen.method,
        "reason": "verified_solution" if chosen.verified else "unverified_solution",
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
            "early_exit": bool(early_exit),
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


def reasoning_diagnostics(result: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return privacy-safe metadata: no prompt text, no computed answer."""

    if not isinstance(result, Mapping):
        return {
            "schema_version": REASONING_SCHEMA_VERSION,
            "attempted": False,
            "solved": False,
            "override_allowed": False,
            "problem_class": "",
            "method": "",
            "verified": False,
            "verification_independent": False,
            "conflicting": False,
            "paths": 0,
            "tier": "",
            "solvers_run": 0,
        }
    verification = result.get("verification") if isinstance(result.get("verification"), Mapping) else {}
    consensus = result.get("consensus") if isinstance(result.get("consensus"), Mapping) else {}
    budget = result.get("budget") if isinstance(result.get("budget"), Mapping) else {}
    return {
        "schema_version": REASONING_SCHEMA_VERSION,
        "attempted": bool(result.get("attempted")),
        "solved": bool(result.get("solved")),
        "override_allowed": bool(result.get("override_allowed")),
        "problem_class": str(result.get("problem_class") or ""),
        "method": str(result.get("method") or ""),
        "verified": bool(verification.get("passed")),
        "verification_independent": bool(verification.get("independent")),
        "conflicting": bool(consensus.get("conflicting")),
        "paths": int(consensus.get("paths") or 0),
        "tier": str(budget.get("tier") or ""),
        "solvers_run": int(budget.get("solvers_run") or 0),
    }


__all__ = [
    "REASONING_ENGINE_VERSION",
    "REASONING_SCHEMA_VERSION",
    "Solution",
    "frame_problem",
    "reasoning_diagnostics",
    "render_reasoning_answer",
    "solve_problem",
]
