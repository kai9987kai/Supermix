"""Layered cognitive memory (working, episodic, firewall, latent state)."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

__all__ = ["SCFLayeredMemory"]

_MEM_TERM_RE = re.compile(r"[a-z0-9_+-]{3,}")


class SCFLayeredMemory:
    """Combines working task state, conversation state, authority firewall, and MiMoMix slots."""

    _LATENT_SUMMARY_CACHE: Optional[Dict[str, Any]] = None

    def __init__(self, *, latent_slot_count: int = 222) -> None:
        self.working_state: Dict[str, Any] = {}
        self.admitted_memories: List[Dict[str, Any]] = []
        self.latent_slot_count = int(latent_slot_count)
        self._rrf_engine: Optional[Any] = None

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
        prompt_terms = set(_MEM_TERM_RE.findall(str(prompt or "").lower()))

        active_commitments = [
            {
                "id": str(c.get("id", "")),
                "kind": str(c.get("kind", "preference")),
                "text": str(c.get("text", "")),
            }
            for c in conv_state.get("commitments", [])
            if c.get("active") and c.get("text")
        ][:6]

        # Capture explicit standalone user preference/remember statements missed by narrow frame prefixes
        seen_c_texts = {c["text"].lower() for c in active_commitments}
        for t_idx, turn in enumerate(list(recent_turns)[-12:]):
            u_txt = str(turn.get("user", "") if isinstance(turn, dict) else (turn[0] if isinstance(turn, (list, tuple)) and turn else "")).strip()
            low_u = u_txt.lower()
            if not u_txt or len(u_txt) > 180 or low_u in seen_c_texts:
                continue
            if any(blk in low_u for blk in ("ignore previous", "system:", "[tool]", "quote:")):
                continue
            if any(k in low_u for k in ("remember ", "i prefer ", "we use ", "always ", "keep answers")):
                active_commitments.append({"id": f"mem_{t_idx}", "kind": "preference", "text": u_txt})
                seen_c_texts.add(low_u)
                if len(active_commitments) >= 6:
                    break

        contradictions = [
            {
                "kind": str(item.get("kind", "conflict")),
                "summary": str(item.get("text") or item.get("summary") or "conflicting user constraint"),
            }
            for item in conv_state.get("contradictions", [])
        ][:4]

        open_questions = [
            str(q.get("text", ""))
            for q in conv_state.get("questions", [])
            if q.get("status") == "open" and q.get("text")
        ][:4]

        # OMNIS/Mnemorph importance scoring over prior turns & admitted memories
        scored_memories: List[Dict[str, Any]] = []
        for idx, turn in enumerate(list(recent_turns)[-12:]):
            u_txt = str(turn.get("user", "") if isinstance(turn, dict) else (turn[0] if isinstance(turn, (list, tuple)) and turn else ""))
            a_txt = str(turn.get("assistant", "") if isinstance(turn, dict) else (turn[1] if isinstance(turn, (list, tuple)) and len(turn) > 1 else ""))
            turn_terms = set(_MEM_TERM_RE.findall(u_txt.lower()))
            overlap = len(prompt_terms & turn_terms) / max(1, len(prompt_terms | turn_terms)) if prompt_terms else 0.0
            recency = (idx + 1) / max(1, min(12, len(recent_turns)))
            has_pref = 1.0 if any(k in u_txt.lower() for k in ("prefer", "always", "never", "use ", "remember", "constraint", "project")) else 0.0
            importance = round(0.45 * overlap + 0.30 * recency + 0.25 * has_pref, 4)
            if u_txt:
                scored_memories.append({
                    "memory_class": "procedural" if has_pref > 0 else "episodic",
                    "importance": importance,
                    "user_excerpt": u_txt[:180],
                    "assistant_excerpt": a_txt[:140],
                    "claims_can_authorize_tools": False,
                })
        scored_memories.sort(key=lambda m: -float(m["importance"]))

        style_req = str(conv_state.get("style_request") or "")
        if not style_req:
            for turn in reversed(list(recent_turns)[-12:]):
                u_txt = str(turn.get("user", "") if isinstance(turn, dict) else (turn[0] if isinstance(turn, (list, tuple)) and turn else ""))
                if not u_txt or not any(k in u_txt.lower() for k in ("prefer", "always", "keep ", "be ", "remember")):
                    continue
                if any(blk in u_txt.lower() for blk in ("ignore previous", "system:", "[tool]", "quote:")):
                    continue
                pref = str(cs.style_preference_of(u_txt) or "")
                if pref:
                    style_req = pref
                    break
        self.working_state = {
            "turn_count": len(recent_turns),
            "state_brief": brief,
            "active_memory_count": len([m for m in self.admitted_memories if m["authority"].get("active")]),
            "active_commitments": active_commitments,
            "contradiction_alerts": contradictions,
            "open_questions": open_questions,
            "style_preference": style_req,
            "salient_turns": scored_memories[:3],
        }

        # v102 NexusSearch BM25 + Sparse Cosine Vector RRF + Titans surprise + 7-class taxonomy
        rrf_res: Dict[str, Any] = {
            "hits": [],
            "query_surprise": 0.0,
            "memory_classes_active": [],
            "rrf_k": 60,
            "answer_authority": False,
        }
        try:
            rrf_docs: List[Dict[str, Any]] = []
            for idx, turn in enumerate(list(recent_turns)[-16:]):
                u_txt = str(turn.get("user", "") if isinstance(turn, dict) else (turn[0] if isinstance(turn, (list, tuple)) and turn else "")).strip()
                if u_txt:
                    rrf_docs.append({"id": f"turn_{idx}", "text": u_txt})
            for idx, adm in enumerate(self.admitted_memories[-16:]):
                if adm.get("authority", {}).get("active"):
                    rrf_docs.append(
                        {
                            "id": f"admitted_{idx}",
                            "text": str(adm.get("text") or ""),
                            "memory_class": "long_term",
                        }
                    )
            if rrf_docs:
                if self._rrf_engine is None:
                    import nexus_v102_frontier as v102

                    self._rrf_engine = v102.NexusSearchHybridRRF()
                rrf_res = self._rrf_engine.rank_documents(prompt, rrf_docs, top_k=4)
        except Exception:
            pass
        self.working_state["rrf_memory_hits"] = list(rrf_res.get("hits") or [])
        self.working_state["query_surprise"] = float(rrf_res.get("query_surprise", 0.0))
        self.working_state["memory_classes_active"] = list(rrf_res.get("memory_classes_active") or [])

        latent_summary: Optional[Dict[str, Any]] = None
        if enable_latent_memory:
            if SCFLayeredMemory._LATENT_SUMMARY_CACHE is not None:
                latent_summary = dict(SCFLayeredMemory._LATENT_SUMMARY_CACHE)
            else:
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
                    SCFLayeredMemory._LATENT_SUMMARY_CACHE = dict(latent_summary)
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
