"""Bounded arithmetic repair proposals with independent full-trace replay.

The memory buffer records diagnostic observations, not learned constraints or
answer authority. A repair is successful only within the localizer's arithmetic
grammar; problem semantics and general reasoning correctness remain unchecked.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import nexus_proof_verification as proof_ver


@dataclass
class EpistemicReflexionCapsule:
    failure_index: int
    failure_mode: str
    failed_step_text: str
    counterfactual_root_cause: str
    negative_avoidance_constraint: str
    suggested_pivot_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflexionCorrectionResult:
    problem: str
    original_trace: List[str]
    had_failure: bool
    reflexion_capsule: Optional[EpistemicReflexionCapsule]
    corrected_trace: List[str]
    corrected_final_answer: Optional[str]
    correction_fidelity: float
    memory_buffer_updated: bool
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem": self.problem,
            "original_trace": self.original_trace,
            "had_failure": self.had_failure,
            "reflexion_capsule": self.reflexion_capsule.to_dict() if self.reflexion_capsule else None,
            "corrected_trace": self.corrected_trace,
            "corrected_final_answer": self.corrected_final_answer,
            "correction_fidelity": self.correction_fidelity,
            "memory_buffer_updated": self.memory_buffer_updated,
            "diagnostic_summary": self.diagnostic_summary,
            "telemetry": self.telemetry,
        }


class ReflexiveCorrectionEngine:
    """Diagnose local arithmetic errors and recheck bounded repair proposals."""

    def __init__(self, memory_capacity: int = 128):
        if isinstance(memory_capacity, bool) or not isinstance(memory_capacity, int) or memory_capacity < 0:
            raise ValueError("memory_capacity must be a nonnegative integer")
        self.memory_capacity = memory_capacity
        self.memory_buffer: List[Dict[str, Any]] = []
        self.localizer = proof_ver.FirstErrorLocalizer()

    def diagnose_and_correct(
        self,
        problem: str,
        trace_steps: List[str],
        ground_truth: Optional[str] = None,
        max_iterations: int = 3,
    ) -> ReflexionCorrectionResult:
        """Run step-level proof localization and synthesize a reflexive self-correction if broken."""
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or not 0 <= max_iterations <= 32:
            raise ValueError("max_iterations must be an integer between 0 and 32")
        fel_res = self.localizer.verify_and_localize(problem, trace_steps)

        def reference_matches(result: proof_ver.FirstErrorResult) -> Optional[bool]:
            if ground_truth is None:
                return None
            exact_answer = result.telemetry.get("exact_final_register")
            if exact_answer is None:
                return False
            try:
                return self.localizer.parse_number(ground_truth.strip()) == self.localizer.parse_number(exact_answer)
            except (ValueError, ZeroDivisionError, OverflowError):
                return False

        if not fel_res.has_error:
            # Clean derivation; no reflexion needed
            return ReflexionCorrectionResult(
                problem=problem,
                original_trace=trace_steps,
                had_failure=False,
                reflexion_capsule=None,
                corrected_trace=trace_steps,
                corrected_final_answer=fel_res.verified_final_answer if reference_matches(fel_res) is not False else None,
                correction_fidelity=1.0 if reference_matches(fel_res) is not False else 0.0,
                memory_buffer_updated=False,
                diagnostic_summary=(
                    "Supported arithmetic trace checked without repair; problem semantics remain unchecked."
                    if reference_matches(fel_res) is not False else
                    "Arithmetic trace passed, but the supplied reference answer did not match; no answer certified."
                ),
                telemetry={
                    "verified_steps": len(trace_steps), "iterations_used": 0,
                    "correction_verified": reference_matches(fel_res) is not False,
                    "reference_answer_matches": reference_matches(fel_res),
                    "answer_authority": False, "verification_scope": "arithmetic_trace_only",
                },
            )

        # Failure identified: construct Epistemic Reflexion Capsule
        fail_idx = fel_res.first_error_index
        fail_cat = fel_res.error_category
        fail_txt = fel_res.error_step_text or ""

        if fail_cat == proof_ver.ProofErrorCategory.ARITHMETIC_ERROR:
            cause = "LHS and RHS diverge under exact arithmetic evaluation."
            constraint = "ENFORCE_EXACT_RATIONAL_EQUIVALENCE: recompute operator transition using exact fractions."
            pivot = "RECOMPUTE_EQUATION_RESULT"
        elif fail_cat == proof_ver.ProofErrorCategory.PHANTOM_REGISTER:
            cause = "Operand was not present in problem premises or prior intermediate state registers."
            constraint = "RESTRICT_TO_ACTIVE_REGISTERS: strictly discard ungrounded constants."
            pivot = "SUBSTITUTE_GROUNDED_PREMISE"
        else:
            cause = f"Formal verification breakdown: {fel_res.diagnostic_explanation}"
            constraint = "RE-ANCHOR_TO_CANONICAL_SPECIFICATION"
            pivot = "BACKTRACK_AND_REINITIALIZE"

        capsule = EpistemicReflexionCapsule(
            failure_index=fail_idx,
            failure_mode=fail_cat,
            failed_step_text=fail_txt,
            counterfactual_root_cause=cause,
            negative_avoidance_constraint=constraint,
            suggested_pivot_action=pivot,
        )

        # Store in episodic memory buffer
        memory_entry = {
            "problem": problem,
            "failure_mode": fail_cat,
            "failed_step": fail_txt,
            "constraint": constraint,
        }
        if self.memory_capacity:
            self.memory_buffer.append(memory_entry)
            del self.memory_buffer[:-self.memory_capacity]

        corrected_trace = list(trace_steps)
        checked = fel_res
        iterations_used = 0
        for _ in range(max_iterations):
            candidate = list(checked.repaired_trace)
            # A final total directly copies the preceding equation result. Only
            # update that explicit dependency; do not guess what phantom operands mean.
            if len(candidate) >= 2:
                last = self.localizer.TOTAL_PATTERN.fullmatch(candidate[-1].strip())
                previous = checked.step_records[-2]
                if (
                    last and previous.detected_error_category == proof_ver.ProofErrorCategory.ARITHMETIC_ERROR
                    and previous.repaired_step_text and previous.exact_expected_result is not None
                    and previous.exact_declared_result is not None
                ):
                    try:
                        copied_result = self.localizer.parse_number(last.group(1)) == self.localizer.parse_number(previous.exact_declared_result)
                    except (ValueError, ZeroDivisionError, OverflowError):
                        copied_result = False
                    if copied_result:
                        candidate[-1] = f"total {previous.exact_expected_result}"
            if candidate == corrected_trace:
                break
            corrected_trace = candidate
            checked = self.localizer.verify_and_localize(problem, corrected_trace)
            iterations_used += 1
            if not checked.has_error:
                break

        reference_match = reference_matches(checked)
        verified = not checked.has_error and checked.verified_final_answer is not None and reference_match is not False
        corrected_ans = checked.verified_final_answer if verified else None
        fidelity = checked.proof_fidelity_score if reference_match is not False else 0.0
        summary = (
            f"Arithmetic repair rechecked successfully after {iterations_used} pass(es); problem semantics remain unchecked."
            if verified else
            f"Repair unresolved after {iterations_used} pass(es): no verified arithmetic answer. {checked.diagnostic_explanation}"
        )
        if reference_match is False:
            summary += " Supplied reference answer did not match the verified trace."

        return ReflexionCorrectionResult(
            problem=problem,
            original_trace=trace_steps,
            had_failure=True,
            reflexion_capsule=capsule,
            corrected_trace=corrected_trace,
            corrected_final_answer=corrected_ans,
            correction_fidelity=fidelity,
            memory_buffer_updated=bool(self.memory_capacity),
            diagnostic_summary=summary,
            telemetry={
                "memory_buffer_size": len(self.memory_buffer),
                "repaired_step_count": len(corrected_trace),
                "iterations_used": iterations_used,
                "correction_verified": verified,
                "reference_answer_matches": reference_match,
                "remaining_first_error_index": checked.first_error_index,
                "verification_scope": "arithmetic_trace_only",
                "answer_authority": False,
            },
        )
