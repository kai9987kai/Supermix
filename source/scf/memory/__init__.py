"""Layered cognitive memory (working, episodic, firewall, latent state)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFLayeredMemory"]


class SCFLayeredMemory:
    """Combines working task state, conversation state, authority firewall, and MiMoMix slots."""

    def __init__(self, *, latent_slot_count: int = 222) -> None:
        self.working_state: Dict[str, Any] = {}
        self.admitted_memories: List[Dict[str, Any]] = []
        self.latent_slot_count = int(latent_slot_count)

    def admit_user_memory(
        self,
        *,
        kind: str,
        text: str,
        source_turn_id: str,
        origin: str = "direct_user",
    ) -> Dict[str, Any]:
        """Enforce memory_authority firewall so recalled strings never become tools/evidence."""

        import memory_authority as ma

        record = ma.build_memory_authority(
            kind=kind,
            text=text,
            source_turn_id=source_turn_id,
            origin=origin,
        )
        inspection = ma.inspect_memory_authority({**record, "kind": kind, "text": text})
        entry = {
            "kind": kind,
            "text": text,
            "authority": record,
            "inspection": inspection,
            "label": ma.authority_label(inspection),
            "claims_can_authorize_tools": False,
        }
        self.admitted_memories.append(entry)
        return entry

    def build_context(
        self,
        *,
        prompt: str,
        recent_turns: Sequence[Any] = (),
        enable_latent_memory: bool = False,
    ) -> Dict[str, Any]:
        """Construct unified working/episodic/latent state summary for SCF."""

        import conversation_state as cs

        conv_state = cs.build_conversation_state(turns=list(recent_turns), current_user_text=prompt)
        brief = cs.render_state_brief(conv_state)
        self.working_state = {
            "turn_count": len(recent_turns),
            "state_brief": brief,
            "active_memory_count": len([m for m in self.admitted_memories if m["authority"].get("active")]),
        }

        latent_summary: Optional[Dict[str, Any]] = None
        if enable_latent_memory:
            try:
                import mimomix_reasoner as mr

                cfg = mr.ReasonerConfig()
                tokenizer = mr.SlotTokenizer(cfg)
                latent_summary = {
                    "enabled": True,
                    "slot_sequence_length": int(tokenizer.sequence_length),
                    "d_model": int(cfg.hidden_size),
                    "n_layers": int(cfg.n_layers),
                    "n_states": int(cfg.n_states),
                    "state_authority": "advisory_slot_context_only",
                }
            except Exception as exc:
                latent_summary = {"enabled": False, "error": f"{type(exc).__name__}: {exc}"}

        return {
            "working_state": dict(self.working_state),
            "conversation_state": cs.conversation_state_diagnostics(conv_state),
            "memory_firewall_enforced": True,
            "latent_memory": latent_summary,
        }

    def run_memory_assay(
        self,
        *,
        assay_observations: Optional[Sequence[Dict[str, Any]]] = None,
        recovery_trajectory: Optional[Sequence[float]] = None,
        prior: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run Bayesian Mnemorph / MORPHEUS connectome-engram recovery assay (advisory only)."""
        import nexus_v95_ecosystem as v95

        res = v95.MnemorphArchaeologyEngine().analyze_assays(
            assay_observations=assay_observations or [{"intervention": "commissure_cut", "observed_retention": 0.85}],
            recovery_trajectory=recovery_trajectory,
            prior=prior,
        )
        return {**res, "answer_authority": False}
