"""Independent, bounded checker for Nexus exact-arithmetic certificates.

This module intentionally has no dependency on ``grounding_runtime`` or
``nexus_proof``. It parses a small arithmetic expression with its own allowlist
and evaluates it with ``Fraction``. It also independently reparses the two
closed-world science families (constant acceleration and ideal gas) using a
separate unit table and formula dispatch. The result is supplementary
evidence: it can strengthen a capsule, but it is never a model confidence,
signature, or routing/promotion decision.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from typing import Any, Dict, Optional


CHECKER_ID = "nexus-independent-arithmetic-checker-v1"
SCIENCE_CHECKER_ID = "nexus-independent-science-checker-v1"
CHECKER_SCHEMA_VERSION = "nexus-independent-check-v1"
MAX_EXPRESSION_CHARS = 160
MAX_AST_NODES = 64
MAX_DEPTH = 12
MAX_OPERATIONS = 32
MAX_EXPONENT = 12
MAX_RESULT_BITS = 4096

_PREFIX_RE = re.compile(
    r"^\s*(?:what\s+is|calculate|compute|evaluate|work\s+out|solve(?:\s+the\s+expression)?)"
    r"\s*[:=]?\s*(?P<expression>.+?)\s*$",
    re.IGNORECASE,
)
_ALLOWED_RE = re.compile(r"^[0-9eE\s+\-*/().%^]+$")
_BINARY_RE = re.compile(r"(?:\*\*|[+\-*/%^])")
_INTEGER_RE = re.compile(r"^[+-]?[0-9]+$")

_SCIENCE_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SCIENCE_NUMBER_RE = re.compile(_SCIENCE_NUM)
_SCIENCE_VELOCITY_UNIT = (
    r"(?:kilomet(?:er|re)s?\s+per\s+hour|meters?\s+per\s+second|"
    r"metres?\s+per\s+second|centimet(?:er|re)s?\s+per\s+second|"
    r"miles?\s+per\s+hour|km\s*/\s*h|cm\s*/\s*s|m\s*/\s*s|mph)"
)
_SCIENCE_ACCELERATION_UNIT = (
    r"(?:meters?\s+per\s+second\s+squared|metres?\s+per\s+second\s+squared|"
    r"kilomet(?:er|re)s?\s+per\s+hour\s+squared|"
    r"centimet(?:er|re)s?\s+per\s+second\s+squared|"
    r"feet\s+per\s+second\s+squared|"
    r"km\s*/\s*h\s*(?:\^\s*2|²)|cm\s*/\s*s\s*(?:\^\s*2|²)|"
    r"ft\s*/\s*s\s*(?:\^\s*2|²)|m\s*/\s*s\s*(?:\^\s*2|²))"
)
_SCIENCE_TIME_UNIT = r"(?:seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|hr|h)"
_SCIENCE_PRESSURE_UNIT = r"(?:megapascals?|kilopascals?|pascals?|MPa|kPa|Pa|bars?|atmospheres?|atm)"
_SCIENCE_VOLUME_UNIT = (
    r"(?:cubic\s+meters?|cubic\s+metres?|cubic\s+centimeters?|"
    r"cubic\s+centimetres?|millilit(?:er|re)s?|lit(?:er|re)s?|"
    r"m\s*(?:\^\s*3|³)|cm\s*(?:\^\s*3|³)|mL|ml|L|l)"
)
_SCIENCE_TEMPERATURE_UNIT = r"(?:kelvins?|K|degrees?\s+celsius|celsius|°\s*C)"
_SCIENCE_AMOUNT_UNIT = r"(?:kilomoles?|millimoles?|moles?|kmol|mmol|mol)"

_SCIENCE_INITIAL_RE = re.compile(
    rf"\binitial\s+velocity\s*(?:of|is|equals?|=|:)?\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_VELOCITY_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_REST_RE = re.compile(r"\b(?:starts?\s+from\s+rest|initially\s+at\s+rest)\b", re.IGNORECASE)
_SCIENCE_ACCELERATION_RE = re.compile(
    rf"\b(?:acceleration\s*(?:of|is|equals?|=|:)?|accelerat(?:es|ing|ed)\s+at)\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_ACCELERATION_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_TIME_RE = re.compile(
    rf"\b(?:for|over|during|time\s*(?:of|is|equals?|=|:)?)\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_TIME_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_PRESSURE_RE = re.compile(
    rf"\b(?:pressure\s*(?:of|is|equals?|=|:)?|at)\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_PRESSURE_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_VOLUME_RE = re.compile(
    rf"\bvolume\s*(?:of|is|equals?|=|:)?\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_VOLUME_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_TEMPERATURE_RE = re.compile(
    rf"\b(?:temperature\s*(?:of|is|equals?|=|:)?|at)\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_TEMPERATURE_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_AMOUNT_RE = re.compile(
    rf"\b(?:amount(?:\s+of\s+substance)?\s*(?:of|is|equals?|=|:)?|contains?|has)\s*"
    rf"(?P<value>{_SCIENCE_NUM})\s*(?P<unit>{_SCIENCE_AMOUNT_UNIT})(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_SCIENCE_CONSTANT_RE = re.compile(
    r"\b(?:(?:assuming|under|with)\s+(?:(?:an?|the)\s+)?constant\s+acceleration|"
    r"uniform\s+acceleration|uniformly\s+accelerat(?:es|ing|ed)|"
    r"accelerat(?:es|ing|ed)\s+uniformly)\b",
    re.IGNORECASE,
)
_SCIENCE_IDEAL_GAS_RE = re.compile(
    r"\b(?:(?:assuming|under|using)\s+(?:(?:an?|the)\s+)?ideal\s+gas"
    r"(?:\s+(?:law|model|equation))?|according\s+to\s+the\s+ideal\s+gas\s+law)\b",
    re.IGNORECASE,
)
_SCIENCE_TARGET_RE = re.compile(
    r"\b(?:what\s+is|calculate|find|determine)\s+(?:(?:the|its)\s+)?"
    r"(?P<target>final\s+velocity|displacement|pressure|volume|temperature|"
    r"amount(?:\s+of\s+substance)?|number\s+of\s+moles)\b",
    re.IGNORECASE,
)
_SCIENCE_PROMPT_CONTROL_RE = re.compile(
    r"\b(?:ignore|disregard|override|system\s+prompt|developer\s+message|"
    r"instruction|write\s+(?:code|a\s+poem|an\s+email)|run\s+code|execute|"
    r"http|https|www|patient|medical|weather|stock|investment)\b",
    re.IGNORECASE,
)

_SCIENCE_FORMULAS = {
    "constant_acceleration.final_velocity": {
        "scenario": "constant_acceleration",
        "target": "final velocity",
        "unit": "m/s",
    },
    "constant_acceleration.displacement": {
        "scenario": "constant_acceleration",
        "target": "displacement",
        "unit": "m",
    },
    "ideal_gas.pressure": {
        "scenario": "ideal_gas",
        "target": "pressure",
        "unit": "Pa",
    },
    "ideal_gas.volume": {
        "scenario": "ideal_gas",
        "target": "volume",
        "unit": "m^3",
    },
    "ideal_gas.temperature": {
        "scenario": "ideal_gas",
        "target": "temperature",
        "unit": "K",
    },
    "ideal_gas.amount": {
        "scenario": "ideal_gas",
        "target": "amount",
        "unit": "mol",
    },
}
_MOLAR_GAS_CONSTANT = Fraction(831446261815324, 100000000000000)


class _CheckError(ValueError):
    pass


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _bounded(value: Fraction) -> Fraction:
    if abs(value.numerator).bit_length() > MAX_RESULT_BITS or value.denominator.bit_length() > MAX_RESULT_BITS:
        raise _CheckError("result_too_large")
    return value


def _expression(query: str) -> Optional[str]:
    text = str(query or "").strip()
    if not text or len(text) > MAX_EXPRESSION_CHARS + 64:
        return None
    match = _PREFIX_RE.fullmatch(text)
    expression = match.group("expression") if match else text
    expression = expression.strip()
    expression = re.sub(
        r"\s+and\s+(?:please\s+)?(?:explain|show|verify|check)\b.*$",
        "",
        expression,
        flags=re.IGNORECASE,
    ).strip()
    while expression and expression[-1] in "?!.":
        expression = expression[:-1].rstrip()
    expression = expression.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    if (
        not expression
        or len(expression) > MAX_EXPRESSION_CHARS
        or not _ALLOWED_RE.fullmatch(expression)
        or not _BINARY_RE.search(expression)
    ):
        return None
    return expression


def _literal(node: ast.Constant, expression: str) -> Fraction:
    if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
        raise _CheckError("unsupported_literal")
    segment = ast.get_source_segment(expression, node) or repr(node.value)
    if isinstance(node.value, int):
        return _bounded(Fraction(node.value, 1))
    try:
        decimal = Decimal(segment)
    except (InvalidOperation, ValueError):
        raise _CheckError("unsupported_literal") from None
    if not decimal.is_finite():
        raise _CheckError("non_finite_literal")
    return _bounded(Fraction(decimal))


def _evaluate(node: ast.AST, expression: str, depth: int, operations: list[int]) -> Fraction:
    if depth > MAX_DEPTH:
        raise _CheckError("expression_too_deep")
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, expression, depth + 1, operations)
    if isinstance(node, ast.Constant):
        return _literal(node, expression)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operations[0] += 1
        value = _evaluate(node.operand, expression, depth + 1, operations)
        return value if isinstance(node.op, ast.UAdd) else _bounded(-value)
    if not isinstance(node, ast.BinOp):
        raise _CheckError("unsupported_syntax")
    operations[0] += 1
    if operations[0] > MAX_OPERATIONS:
        raise _CheckError("too_many_operations")
    left = _evaluate(node.left, expression, depth + 1, operations)
    right = _evaluate(node.right, expression, depth + 1, operations)
    if isinstance(node.op, ast.Add):
        result = left + right
    elif isinstance(node.op, ast.Sub):
        result = left - right
    elif isinstance(node.op, ast.Mult):
        result = left * right
    elif isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
        if right == 0:
            raise _CheckError("division_by_zero")
        if isinstance(node.op, ast.Div):
            result = left / right
        elif isinstance(node.op, ast.FloorDiv):
            result = Fraction(left // right, 1)
        else:
            result = left % right
    elif isinstance(node.op, ast.Pow):
        if right.denominator != 1 or abs(right.numerator) > MAX_EXPONENT:
            raise _CheckError("exponent_not_supported")
        if right.numerator < 0 and left == 0:
            raise _CheckError("division_by_zero")
        result = left ** int(right.numerator)
    else:
        raise _CheckError("unsupported_operator")
    return _bounded(result)


def _display(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    with localcontext() as context:
        context.prec = min(220, max(32, len(str(abs(value.numerator))) + len(str(value.denominator)) + 8))
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _parse_display(value: str) -> Fraction:
    text = str(value or "").strip().replace(",", "")
    if any(char.isnumeric() and char not in "0123456789" for char in text):
        raise _CheckError("non_ascii_numeric_display")
    if _INTEGER_RE.fullmatch(text):
        return Fraction(int(text), 1)
    if re.fullmatch(r"[+-]?[0-9]+/[0-9]+", text):
        numerator, denominator = text.split("/", 1)
        if int(denominator) == 0:
            raise _CheckError("division_by_zero")
        return Fraction(int(numerator), int(denominator))
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        raise _CheckError("display_not_numeric") from None
    if not decimal.is_finite():
        raise _CheckError("display_not_finite")
    return Fraction(decimal)


def check_arithmetic_certificate(
    *,
    query: str,
    display_answer: str,
    problem_class: str,
) -> Dict[str, Any]:
    """Cross-check an arithmetic answer without importing Nexus's grounder."""

    if problem_class != "arithmetic":
        return {
            "schema_version": CHECKER_SCHEMA_VERSION,
            "checker_id": CHECKER_ID,
            "status": "not_applicable",
            "algorithmically_independent": False,
            "reason": "non_arithmetic_claim_scope",
        }
    expression = _expression(query)
    base: Dict[str, Any] = {
        "schema_version": CHECKER_SCHEMA_VERSION,
        "checker_id": CHECKER_ID,
        "status": "failed",
        "algorithmically_independent": True,
        "query_sha256": _digest(str(query)),
        "display_sha256": _digest(str(display_answer)),
    }
    if expression is None:
        base["algorithmically_independent"] = False
        base["reason"] = "expression_not_admitted"
        return base
    base["expression_sha256"] = _digest(expression)
    try:
        tree = ast.parse(expression, mode="eval")
        if len(list(ast.walk(tree))) > MAX_AST_NODES:
            raise _CheckError("too_many_nodes")
        operations = [0]
        expected = _evaluate(tree, expression, 0, operations)
        observed = _parse_display(display_answer)
    except (SyntaxError, _CheckError) as exc:
        base["algorithmically_independent"] = False
        base["reason"] = "invalid_expression_or_display:" + str(exc)
        return base
    base.update(
        {
            "status": "passed" if expected == observed else "failed",
            "reason": "independent_fraction_match" if expected == observed else "display_mismatch",
            "operations": operations[0],
            "expected_display": _display(expected),
            "observed_display": str(display_answer),
        }
    )
    if expected != observed:
        base["algorithmically_independent"] = False
    return base


def _science_not_applicable() -> Dict[str, Any]:
    return {
        "schema_version": CHECKER_SCHEMA_VERSION,
        "checker_id": SCIENCE_CHECKER_ID,
        "status": "not_applicable",
        "algorithmically_independent": False,
        "reason": "non_scientific_claim_scope",
    }


def _science_text(query: str) -> Optional[str]:
    text = str(query or "").strip()
    if not text or len(text) > 1200:
        return None
    if any(char.isnumeric() and char not in "0123456789²³" for char in text):
        return None
    if _SCIENCE_PROMPT_CONTROL_RE.search(text):
        return None
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        return None
    return re.sub(r"\s+", " ", text)


def _science_number(value: str) -> Fraction:
    if any(char.isnumeric() and char not in "0123456789" for char in value):
        raise _CheckError("non_ascii_numeric_literal")
    try:
        decimal = Decimal(value)
    except (InvalidOperation, ValueError):
        raise _CheckError("invalid_science_literal") from None
    if not decimal.is_finite():
        raise _CheckError("non_finite_science_literal")
    return _bounded(Fraction(decimal))


def _science_unit_key(raw_unit: str) -> str:
    unit = str(raw_unit).casefold().strip()
    unit = unit.replace("²", "^2").replace("³", "^3").replace("·", "")
    unit = re.sub(r"\s+", " ", unit)
    unit = (
        unit.replace("metres", "meters")
        .replace("metre", "meter")
        .replace("kilometres", "kilometers")
        .replace("kilometre", "kilometer")
        .replace("centimetres", "centimeters")
        .replace("centimetre", "centimeter")
        .replace("litres", "liters")
        .replace("litre", "liter")
        .replace("millilitres", "milliliters")
        .replace("millilitre", "milliliter")
    )
    return re.sub(r"\s+", "", unit)


_SCIENCE_UNIT_FACTORS = {
    "velocity": {
        "m/s": Fraction(1),
        "meterpersecond": Fraction(1),
        "meterspersecond": Fraction(1),
        "km/h": Fraction(5, 18),
        "kilometerperhour": Fraction(5, 18),
        "kilometersperhour": Fraction(5, 18),
        "cm/s": Fraction(1, 100),
        "centimeterpersecond": Fraction(1, 100),
        "centimeterspersecond": Fraction(1, 100),
        "mph": Fraction(1397, 3125),
        "mileperhour": Fraction(1397, 3125),
        "milesperhour": Fraction(1397, 3125),
    },
    "acceleration": {
        "m/s^2": Fraction(1),
        "meterpersecondsquared": Fraction(1),
        "meterspersecondsquared": Fraction(1),
        "km/h^2": Fraction(1, 12960),
        "kilometerperhoursquared": Fraction(1, 12960),
        "kilometersperhoursquared": Fraction(1, 12960),
        "cm/s^2": Fraction(1, 100),
        "centimeterpersecondsquared": Fraction(1, 100),
        "centimeterspersecondsquared": Fraction(1, 100),
        "ft/s^2": Fraction(381, 1250),
        "feetpersecondsquared": Fraction(381, 1250),
    },
    "time": {
        "s": Fraction(1),
        "sec": Fraction(1),
        "secs": Fraction(1),
        "second": Fraction(1),
        "seconds": Fraction(1),
        "min": Fraction(60),
        "mins": Fraction(60),
        "minute": Fraction(60),
        "minutes": Fraction(60),
        "h": Fraction(3600),
        "hr": Fraction(3600),
        "hrs": Fraction(3600),
        "hour": Fraction(3600),
        "hours": Fraction(3600),
    },
    "pressure": {
        "pa": Fraction(1),
        "pascal": Fraction(1),
        "pascals": Fraction(1),
        "kpa": Fraction(1000),
        "kilopascal": Fraction(1000),
        "kilopascals": Fraction(1000),
        "mpa": Fraction(1_000_000),
        "megapascal": Fraction(1_000_000),
        "megapascals": Fraction(1_000_000),
        "bar": Fraction(100_000),
        "bars": Fraction(100_000),
        "atm": Fraction(101_325),
        "atmosphere": Fraction(101_325),
        "atmospheres": Fraction(101_325),
    },
    "volume": {
        "m^3": Fraction(1),
        "cubicmeter": Fraction(1),
        "cubicmeters": Fraction(1),
        "l": Fraction(1, 1000),
        "liter": Fraction(1, 1000),
        "liters": Fraction(1, 1000),
        "ml": Fraction(1, 1_000_000),
        "milliliter": Fraction(1, 1_000_000),
        "milliliters": Fraction(1, 1_000_000),
        "cm^3": Fraction(1, 1_000_000),
        "cubiccentimeter": Fraction(1, 1_000_000),
        "cubiccentimeters": Fraction(1, 1_000_000),
    },
    "temperature": {
        "k": Fraction(1),
        "kelvin": Fraction(1),
        "kelvins": Fraction(1),
        "degc": Fraction(1),
        "°c": Fraction(1),
        "celsius": Fraction(1),
        "degreecelsius": Fraction(1),
        "degreescelsius": Fraction(1),
    },
    "amount": {
        "mol": Fraction(1),
        "mole": Fraction(1),
        "moles": Fraction(1),
        "mmol": Fraction(1, 1000),
        "millimole": Fraction(1, 1000),
        "millimoles": Fraction(1, 1000),
        "kmol": Fraction(1000),
        "kilomole": Fraction(1000),
        "kilomoles": Fraction(1000),
    },
}


def _science_convert(value: str, raw_unit: str, dimension: str) -> Fraction:
    factor = _SCIENCE_UNIT_FACTORS.get(dimension, {}).get(_science_unit_key(raw_unit))
    if factor is None:
        raise _CheckError("unsupported_science_unit")
    numeric = _science_number(value)
    # Celsius is the only affine unit in this table. Keep the offset explicit
    # rather than silently treating it as Kelvin.
    key = _science_unit_key(raw_unit)
    if dimension == "temperature" and key in {"degc", "°c", "celsius", "degreecelsius", "degreescelsius"}:
        return _bounded(numeric + Fraction(5463, 20))
    return _bounded(numeric * factor)


def _science_display(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator == 1:
        with localcontext() as context:
            context.prec = min(220, max(32, len(str(abs(value.numerator))) + len(str(value.denominator)) + 8))
            rendered = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
        return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered
    with localcontext() as context:
        context.prec = 16
        return format(Decimal(value.numerator) / Decimal(value.denominator), ".12g")


def _science_match_value(match: re.Match[str], dimension: str) -> Fraction:
    return _science_convert(match.group("value"), match.group("unit"), dimension)


def _science_numbers_are_consumed(text: str, spans: list[tuple[int, int]]) -> bool:
    for number in _SCIENCE_NUMBER_RE.finditer(text):
        if not any(start <= number.start() and number.end() <= end for start, end in spans):
            return False
    return True


def _science_check_constant(text: str, method: str) -> tuple[Fraction, str]:
    target = "final velocity" if method.endswith("final_velocity") else "displacement"
    target_matches = [m for m in _SCIENCE_TARGET_RE.finditer(text) if m.group("target").casefold() == target]
    if len(target_matches) != 1 or len(_SCIENCE_CONSTANT_RE.findall(text)) != 1:
        raise _CheckError("science_target_or_assumption_not_unique")
    initial = list(_SCIENCE_INITIAL_RE.finditer(text))
    rests = list(_SCIENCE_REST_RE.finditer(text))
    accelerations = list(_SCIENCE_ACCELERATION_RE.finditer(text))
    times = list(_SCIENCE_TIME_RE.finditer(text))
    if len(initial) + len(rests) != 1 or len(accelerations) != 1 or len(times) != 1:
        raise _CheckError("science_quantity_not_unique")
    spans = [
        (m.start(), m.end()) for m in initial + rests + accelerations + times
    ]
    if not _science_numbers_are_consumed(text, spans):
        raise _CheckError("science_unconsumed_numeric_literal")
    u = Fraction(0) if rests else _science_match_value(initial[0], "velocity")
    a = _science_match_value(accelerations[0], "acceleration")
    t = _science_match_value(times[0], "time")
    if t <= 0:
        raise _CheckError("science_time_domain")
    if method.endswith("final_velocity"):
        return _bounded(u + a * t), "m/s"
    return _bounded(u * t + a * t * t / 2), "m"


def _science_check_ideal_gas(text: str, method: str) -> tuple[Fraction, str]:
    formula = _SCIENCE_FORMULAS.get(method)
    if formula is None or len(_SCIENCE_IDEAL_GAS_RE.findall(text)) != 1:
        raise _CheckError("science_target_or_assumption_not_unique")
    target = str(formula["target"])
    target_matches = [m for m in _SCIENCE_TARGET_RE.finditer(text) if m.group("target").casefold().startswith(target)]
    if len(target_matches) != 1:
        raise _CheckError("science_target_not_unique")
    patterns = {
        "P": (_SCIENCE_PRESSURE_RE, "pressure"),
        "V": (_SCIENCE_VOLUME_RE, "volume"),
        "T": (_SCIENCE_TEMPERATURE_RE, "temperature"),
        "n": (_SCIENCE_AMOUNT_RE, "amount"),
    }
    target_symbol = {"pressure": "P", "volume": "V", "temperature": "T", "amount": "n"}[target]
    values: Dict[str, Fraction] = {}
    spans: list[tuple[int, int]] = []
    for symbol, (pattern, dimension) in patterns.items():
        matches = list(pattern.finditer(text))
        if symbol == target_symbol:
            if matches:
                raise _CheckError("science_target_quantity_supplied")
            continue
        if len(matches) != 1:
            raise _CheckError("science_quantity_not_unique")
        values[symbol] = _science_match_value(matches[0], dimension)
        spans.append((matches[0].start(), matches[0].end()))
    if not _science_numbers_are_consumed(text, spans):
        raise _CheckError("science_unconsumed_numeric_literal")
    if any(value <= 0 for value in values.values()):
        raise _CheckError("science_positive_domain")
    if target == "pressure":
        answer = values["n"] * _MOLAR_GAS_CONSTANT * values["T"] / values["V"]
    elif target == "volume":
        answer = values["n"] * _MOLAR_GAS_CONSTANT * values["T"] / values["P"]
    elif target == "temperature":
        answer = values["P"] * values["V"] / (values["n"] * _MOLAR_GAS_CONSTANT)
    else:
        answer = values["P"] * values["V"] / (_MOLAR_GAS_CONSTANT * values["T"])
    return _bounded(answer), str(formula["unit"])


def check_science_certificate(
    *,
    query: str,
    display_answer: str,
    method: str,
    unit: str,
) -> Dict[str, Any]:
    """Independently parse and evaluate one allowlisted science formula."""

    base: Dict[str, Any] = {
        "schema_version": CHECKER_SCHEMA_VERSION,
        "checker_id": SCIENCE_CHECKER_ID,
        "status": "failed",
        "algorithmically_independent": True,
        "query_sha256": _digest(str(query)),
        "display_sha256": _digest(str(display_answer)),
        "method": str(method),
    }
    if method not in _SCIENCE_FORMULAS:
        base["algorithmically_independent"] = False
        base["reason"] = "method_not_admitted"
        return base
    text = _science_text(query)
    if text is None:
        base["algorithmically_independent"] = False
        base["reason"] = "query_not_admitted"
        return base
    try:
        formula = _SCIENCE_FORMULAS[method]
        expected, expected_unit = (
            _science_check_constant(text, method)
            if formula["scenario"] == "constant_acceleration"
            else _science_check_ideal_gas(text, method)
        )
    except _CheckError as exc:
        base["algorithmically_independent"] = False
        base["reason"] = str(exc)
        return base
    expected_display = _science_display(expected)
    observed_display = str(display_answer).strip()
    passed = observed_display == expected_display and str(unit) == expected_unit
    base.update(
        {
            "status": "passed" if passed else "failed",
            "algorithmically_independent": bool(passed),
            "reason": "independent_science_match" if passed else "science_display_or_unit_mismatch",
            "expected_display": expected_display,
            "observed_display": observed_display,
            "expected_unit": expected_unit,
            "observed_unit": str(unit),
            "calculation_sha256": _digest({"method": method, "expected": expected_display, "unit": expected_unit}),
        }
    )
    return base


def check_certificate(
    *,
    query: str,
    display_answer: str,
    problem_class: str,
    method: str = "",
    unit: str = "",
) -> Dict[str, Any]:
    """Dispatch to the independent checker matching the claim family."""

    if problem_class == "arithmetic":
        return check_arithmetic_certificate(
            query=query,
            display_answer=display_answer,
            problem_class=problem_class,
        )
    if problem_class == "scientific_scenario":
        return check_science_certificate(
            query=query,
            display_answer=display_answer,
            method=method,
            unit=unit,
        )
    return _science_not_applicable()


__all__ = [
    "CHECKER_ID",
    "CHECKER_SCHEMA_VERSION",
    "SCIENCE_CHECKER_ID",
    "check_arithmetic_certificate",
    "check_certificate",
    "check_science_certificate",
]
