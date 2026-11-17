"""Multi-path reasoning coordinator across GoT, Causal DAG, Active Inference, and Reflexion."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf.config import SCFFeatureFlags  # noqa: E402

__all__ = ["SCFReasoningCoordinator"]


class SCFReasoningCoordinator:
    """Coordinates Graph-of-Thought, Active Inference, Causal DAG, Ideation, and Problem Solving."""

    @staticmethod
    def decompose_and_probe(
        query: str,
        *,
        working_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Decompose a problem, detect CROSSWIRE ambiguity, and run SONAPSE critic checks."""
        text = str(query or "").strip()
        lower = text.lower()
        stop = {
            "the", "and", "for", "with", "that", "this", "how", "can", "why", "what",
            "let", "lets", "our", "your", "you", "are", "were", "was", "does", "did",
            "when", "where", "which", "would", "could", "should", "make", "about",
            "into", "from", "discuss", "explain", "help", "need", "want", "more",
        }
        terms = [w for w in re.findall(r"[a-zA-Z0-9_+-]{3,}", lower) if w not in stop]
        unique_terms = list(dict.fromkeys(terms))
        focus = ", ".join(unique_terms[:5]) or "this task"
        ws = dict(working_state or {})
        commitments = [c.get("text", "") for c in ws.get("active_commitments", []) if isinstance(c, dict) and c.get("text")]

        # CROSSWIRE underspecification / ambiguity check
        clarifying_questions: List[str] = []
        vague_pronouns = bool(re.search(r"\b(it|this|that|my script|my app|my code|things)\b", lower))
        if vague_pronouns and len(terms) < 6 and not ws.get("salient_turns"):
            clarifying_questions.append("Which specific error message, stack trace, or observed vs expected behavior should we isolate first?")
        if any(k in lower for k in ("optimize", "faster", "improve", "better", "choose", "compare", "design")):
            clarifying_questions.append("What is your hard constraint priority (accuracy, latency, memory footprint, or simplicity)?")

        if any(k in lower for k in ("bug", "error", "fail", "failing", "crash", "race", "debug", "fix", "broken")):
            steps = [
                f"Reproduce & isolate minimum failing case around `{focus}` with deterministic seed/input.",
                "Separate state mutation, external I/O, and invariant violations into single-variable checks.",
                "Test candidate hypothesis with a minimal assertion before editing shared state.",
                "Lock the fix with a regression test covering both the failing edge case and existing baseline.",
            ]
        elif any(k in lower for k in ("choose", "decide", "tradeoff", "compare", "architecture", "design", "plan")):
            steps = [
                f"Define explicit success metrics and non-negotiable constraints for `{focus}`.",
                "Evaluate options on worst-case failure mode (Faultline joint-failure check), not just average case.",
                "Prototype the smallest reversible slice behind a feature flag with paired A/B measurement.",
                "Keep a clean rollback condition if held-out metrics regress.",
            ]
        else:
            steps = [
                f"Clarify exact target outcome and boundaries for `{focus}`.",
                "Decompose into small testable sub-problems and verify intermediate invariants early.",
                "Check edge cases, resource limits, and potential contradictions against prior constraints.",
            ]

        critic_checks = [
            "Falsification test: define what metric or assertion would disprove this approach.",
            "Boundary guard: preserve baseline behavior via default-off flag or rollback point until verified.",
        ]
        if commitments:
            critic_checks.append(f"Memory alignment: verify plan respects active user constraints ({'; '.join(commitments[:2])}).")

        return {
            "focus_terms": unique_terms[:6],
            "steps": steps,
            "clarifying_questions": clarifying_questions[:2],
            "critic_checks": critic_checks,
            "ambiguity_detected": bool(clarifying_questions),
            "answer_authority": False,
        }

    def reason(
        self,
        query: str,
        *,
        flags: SCFFeatureFlags,
        max_steps: int = 8,
        trace_steps: Optional[Sequence[str]] = None,
        ground_truth: Optional[str] = None,
        working_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import nexus_got as ngot
        import nexus_reflexion as nref

        depth = max(1, min(4, int(max_steps)))
        got_engine = ngot.GraphOfThoughts(max_depth=min(3, depth), beam_width=2)
        got_res = got_engine.search(query).to_dict()
        problem_probe = self.decompose_and_probe(query, working_state=working_state)

        causal_res: Optional[Dict[str, Any]] = None
        if any(k in query.lower() for k in ("causal", "do(", "counterfactual", "backdoor", "force", "drug", "debug", "fail", "latency", "why")):
            import nexus_causal_dag as ncd

            dag = ncd.CausalDAGEngine()
            scenario = "medical_treatment" if "drug" in query.lower() else "physics_newton"
            causal_res = dag.evaluate_causal_query(scenario=scenario).to_dict()

        active_inf_res: Optional[Dict[str, Any]] = None
        if flags.SCF_ACTIVE_INFERENCE_ENABLED:
            import nexus_active_inference as nai

            ctrl = nai.ActiveInferenceController()
            active_inf_res = ctrl.decide(
                query=query,
                current_trace_steps=list(trace_steps or []),
            ).to_dict()

        reflexion_res: Optional[Dict[str, Any]] = None
        if trace_steps:
            ref = nref.ReflexiveCorrectionEngine()
            reflexion_res = ref.diagnose_and_correct(
                query,
                list(trace_steps),
                ground_truth=ground_truth,
            ).to_dict()

        return {
            "answer_authority": False,
            "got": got_res,
            "problem_decomposition": problem_probe,
            "causal_dag": causal_res,
            "active_inference": active_inf_res,
            "reflexion": reflexion_res,
        }

    def shield_plan_action(
        self,
        *,
        candidate_action: str = "EXEC",
        observed_trace: Optional[Sequence[Dict[str, Any]]] = None,
        hypotheses: Optional[Sequence[Dict[str, Any]]] = None,
        unsafe_outputs: Optional[Sequence[str]] = None,
        unsafe_states: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run fail-closed Crosswire all-hypothesis version-space interface shield."""
        import nexus_v95_ecosystem as v95

        res = v95.CrosswireInterfaceEngine().verify_and_probe(
            hypotheses=hypotheses,
            observed_trace=observed_trace,
            candidate_action=candidate_action,
            unsafe_outputs=unsafe_outputs,
            unsafe_states=unsafe_states,
        )
        return {**res, "answer_authority": False}
