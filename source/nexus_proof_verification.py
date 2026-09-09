"""Exact, bounded arithmetic trace checking and first-error localization.

Supported steps are single binary numeric equations and terminal totals. Literal
membership is only a register check: this module does not verify the meaning of
the question, units, premise truth, or whether the chosen operation answers it.
Suggested repairs have no authority until the complete trace is checked again.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Set


class ProofErrorCategory:
    NONE = "NONE"
    ARITHMETIC_ERROR = "ARITHMETIC_ERROR"
    PHANTOM_REGISTER = "PHANTOM_REGISTER"
    PREMISE_CONTRADICTION = "PREMISE_CONTRADICTION"
    UNSUPPORTED_LEAP = "UNSUPPORTED_LEAP"
    SYNTAX_ERROR = "SYNTAX_ERROR"


@dataclass
class StepVerificationRecord:
    step_index: int
    step_text: str
    is_valid: bool
    detected_error_category: str
    declared_operands: List[float] = field(default_factory=list)
    declared_operator: Optional[str] = None
    declared_result: Optional[float] = None
    expected_result: Optional[float] = None
    repaired_step_text: Optional[str] = None
    diagnostic_note: str = ""
    exact_declared_result: Optional[str] = None
    exact_expected_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FirstErrorResult:
    problem: str
    has_error: bool
    first_error_index: int
    error_category: str
    error_step_text: Optional[str]
    diagnostic_explanation: str
    step_records: List[StepVerificationRecord]
    repaired_trace: List[str]
    verified_final_answer: Optional[str]
    proof_fidelity_score: float
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FirstErrorLocalizer:
    """Check every assertion under a deliberately small arithmetic grammar."""

    DECIMAL = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
    NUMBER = rf"(?:[+-]?\d+/\d+|{DECIMAL})"
    EQ_PATTERN = re.compile(rf"({NUMBER})\s*([+*/-])\s*({NUMBER})\s*=\s*({NUMBER})[.!]?")
    TOTAL_PATTERN = re.compile(
        rf"(?:the\s+)?(?:total|answer|result)(?:\s+is|\s*[:=])?\s+({NUMBER})[.!]?",
        re.IGNORECASE,
    )
    # A sign directly after a number is a binary operator, not its next operand's sign.
    NUMBER_PATTERN = re.compile(rf"(?<![\w.]){NUMBER}(?![\w.])")
    MAX_STEP_CHARS = 4096

    def __init__(self, tolerance: float = 1e-5):
        # Retained for caller compatibility; equality and grounding are always exact.
        self.tolerance = tolerance

    @staticmethod
    def parse_number(text: str) -> Fraction:
        if len(text) > 128:
            raise ValueError("Numeric literal exceeds the supported length")
        exponent = re.search(r"[eE]([+-]?\d+)$", text)
        if exponent and abs(int(exponent.group(1))) > 100:
            raise ValueError("Numeric exponent exceeds the supported range")
        value = Fraction(text)
        if not math.isfinite(float(value)):
            raise ValueError("Numeric literal is outside the finite display range")
        return value

    def _problem_registers(self, problem: str) -> Set[Fraction]:
        registers: Set[Fraction] = set()
        for match in self.NUMBER_PATTERN.finditer(problem):
            try:
                registers.add(self.parse_number(match.group(0)))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
        return registers

    def extract_problem_numbers(self, problem: str) -> Set[float]:
        """Legacy float view for diagnostic consumers; verification uses exact values."""
        return {float(value) for value in self._problem_registers(problem)}

    def evaluate_step(
        self, step_idx: int, step_text: str, active_registers: Set[float],
    ) -> StepVerificationRecord:
        text = step_text.strip()
        record = StepVerificationRecord(
            step_idx, step_text, False, ProofErrorCategory.UNSUPPORTED_LEAP,
            diagnostic_note="Unsupported assertion; expected one complete numeric equation or terminal total.",
        )
        if len(text) > self.MAX_STEP_CHARS:
            record.detected_error_category = ProofErrorCategory.SYNTAX_ERROR
            record.diagnostic_note = "Step exceeds the supported length."
            return record
        equation = self.EQ_PATTERN.fullmatch(text)
        terminal = self.TOTAL_PATTERN.fullmatch(text)
        if equation is None and terminal is None:
            return record
        try:
            registers = {self.parse_number(str(value)) for value in active_registers}
            if terminal:
                value = self.parse_number(terminal.group(1))
                record.declared_result = float(value)
                record.exact_declared_result = str(value)
                record.is_valid = value in registers
                record.detected_error_category = (
                    ProofErrorCategory.NONE if record.is_valid else ProofErrorCategory.PHANTOM_REGISTER
                )
                record.diagnostic_note = (
                    "Terminal value occurs in active registers; full-trace dependency check still required."
                    if record.is_valid else "Terminal value does not occur in any verified active register."
                )
                return record

            left_text, operator, right_text, result_text = equation.groups()
            left, right, declared = map(self.parse_number, (left_text, right_text, result_text))
            record.declared_operands = [float(left), float(right)]
            record.declared_operator = operator
            record.declared_result = float(declared)
            record.exact_declared_result = str(declared)
            if left not in registers or right not in registers:
                missing = left if left not in registers else right
                record.detected_error_category = ProofErrorCategory.PHANTOM_REGISTER
                record.diagnostic_note = f"Operand {missing} is a phantom register absent from premises and valid prior steps."
                return record
            if operator == "+":
                expected = left + right
            elif operator == "-":
                expected = left - right
            elif operator == "*":
                expected = left * right
            else:
                if right == 0:
                    record.detected_error_category = ProofErrorCategory.ARITHMETIC_ERROR
                    record.diagnostic_note = "Division by zero cannot be repaired to a numeric result."
                    return record
                expected = left / right
            record.expected_result = float(expected)
            record.exact_expected_result = str(expected)
            record.is_valid = declared == expected
            record.detected_error_category = (
                ProofErrorCategory.NONE if record.is_valid else ProofErrorCategory.ARITHMETIC_ERROR
            )
            record.repaired_step_text = (
                step_text if record.is_valid else f"{left_text} {operator} {right_text} = {expected}"
            )
            record.diagnostic_note = (
                "Exact arithmetic and literal register membership checked; problem semantics are unchecked."
                if record.is_valid else f"Arithmetic error: expected exact result {expected}, received {result_text}."
            )
            return record
        except (ValueError, ZeroDivisionError, OverflowError):
            record.detected_error_category = ProofErrorCategory.SYNTAX_ERROR
            record.diagnostic_note = "Invalid or out-of-range numeric literal."
            return record

    @staticmethod
    def _answer_text(value: Fraction) -> str:
        # Keep the historic small-integer display without rounding large integers.
        if value.denominator == 1 and abs(value.numerator) <= 2**53:
            return f"{value.numerator}.0"
        return str(value)

    def verify_and_localize(self, problem: str, trace_steps: List[str]) -> FirstErrorResult:
        initial_registers = self._problem_registers(problem)
        running_registers = set(initial_registers)
        records: List[StepVerificationRecord] = []
        repaired_trace: List[str] = []
        last_computed: Optional[Fraction] = None
        terminal_seen = False

        for index, step in enumerate(trace_steps):
            record = self.evaluate_step(index, step, running_registers)
            terminal = self.TOTAL_PATTERN.fullmatch(step.strip()) is not None
            if terminal_seen:
                record.is_valid = False
                record.detected_error_category = ProofErrorCategory.UNSUPPORTED_LEAP
                record.diagnostic_note = "Assertions after the terminal answer are unsupported."
            elif terminal and record.is_valid:
                if last_computed is None or Fraction(record.exact_declared_result) != last_computed:
                    record.is_valid = False
                    record.detected_error_category = ProofErrorCategory.UNSUPPORTED_LEAP
                    record.diagnostic_note = "Terminal answer must equal the most recent verified equation result."
            records.append(record)
            repaired_trace.append(record.repaired_step_text or step)
            if terminal:
                terminal_seen = True
            # Invalid equations never create registers, including their suggested repairs.
            if record.is_valid and record.exact_expected_result is not None:
                last_computed = Fraction(record.exact_expected_result)
                running_registers.add(last_computed)

        first_error = next((record for record in records if not record.is_valid), None)
        has_error = first_error is not None or not records
        valid_count = sum(record.is_valid for record in records)
        answer = self._answer_text(last_computed) if not has_error and last_computed is not None else None
        if first_error:
            diagnostic = f"First error at step {first_error.step_index} [{first_error.detected_error_category}]: {first_error.diagnostic_note}"
        elif not records:
            diagnostic = "Empty reasoning trace cannot satisfy arithmetic verification."
        else:
            diagnostic = f"All {len(records)} supported arithmetic steps checked. Problem semantics and answer relevance are not verified."
        return FirstErrorResult(
            problem=problem,
            has_error=has_error,
            first_error_index=first_error.step_index if first_error else (0 if not records else -1),
            error_category=first_error.detected_error_category if first_error else (
                ProofErrorCategory.UNSUPPORTED_LEAP if not records else ProofErrorCategory.NONE
            ),
            error_step_text=first_error.step_text if first_error else None,
            diagnostic_explanation=diagnostic,
            step_records=records,
            repaired_trace=repaired_trace,
            verified_final_answer=answer,
            proof_fidelity_score=round(valid_count / len(records), 3) if records else 0.0,
            telemetry={
                "steps_analyzed": len(records),
                "valid_steps": valid_count,
                "initial_registers_count": len(initial_registers),
                "terminal_registers_count": len(running_registers),
                "verification_scope": "arithmetic_trace_only",
                "problem_semantics_verified": False,
                "answer_authority": False,
                "exact_final_register": str(last_computed) if not has_error and last_computed is not None else None,
                "repair_requires_reverification": has_error,
            },
        )
