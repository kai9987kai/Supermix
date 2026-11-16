"""Multi-path reasoning coordinator across GoT, Causal DAG, Active Inference, and Reflexion."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf.config import SCFFeatureFlags  # noqa: E402

__all__ = ["SCFReasoningCoordinator"]


class SCFReasoningCoordinator:
    """Coordinates Graph-of-Thought, Active Inference, Causal DAG, and Reflexion."""

    def reason(
        self,
        query: str,
        *,
        flags: SCFFeatureFlags,
        max_steps: int = 8,
        trace_steps: Optional[Sequence[str]] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        import nexus_got as ngot
        import nexus_reflexion as nref

        depth = max(1, min(4, int(max_steps)))
        got_engine = ngot.GraphOfThoughts(max_depth=min(3, depth), beam_width=2)
        got_res = got_engine.search(query).to_dict()

        causal_res: Optional[Dict[str, Any]] = None
        if any(k in query.lower() for k in ("causal", "do(", "counterfactual", "backdoor", "force", "drug")):
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
