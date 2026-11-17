"""Central Cognitive Fabric runtime orchestrating the 10-step cognitive cycle."""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

SOURCE_DIR = Path(__file__).resolve().parents[2]
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from scf.adaptive import SCFAdaptiveComputeController  # noqa: E402
from scf.collective import SCFCollectiveOrchestrator  # noqa: E402
from scf.config import SCFConfig, SCFFeatureFlags  # noqa: E402
from scf.learning import SCFLearningGate  # noqa: E402
from scf.memory import SCFLayeredMemory  # noqa: E402
from scf.models import SCFModelRegistry  # noqa: E402
from scf.multimodal import SCFMultimodalBridge  # noqa: E402
from scf.observability import SCFExecutionReceipt, SCFObservabilityHub  # noqa: E402
from scf.reasoning import SCFReasoningCoordinator  # noqa: E402
from scf.routing import SCFRouter  # noqa: E402
from scf.verification import SCFVerificationGate  # noqa: E402

__all__ = ["CognitiveFabricRuntime"]


class CognitiveFabricRuntime:
    """Unified, feature-flagged meta-architecture across Supermix model families."""

    def __init__(self, config: Optional[SCFConfig] = None) -> None:
        self.config = config or SCFConfig()
        self.registry = SCFModelRegistry()
        self.router = SCFRouter(registry=self.registry)
        self.collective = SCFCollectiveOrchestrator()
        self.memory = SCFLayeredMemory()
        self.reasoning = SCFReasoningCoordinator()
        self.verifier = SCFVerificationGate()
        self.multimodal = SCFMultimodalBridge()
        self.adaptive = SCFAdaptiveComputeController(
            target_risk_alpha=self.config.conformal_alpha
        )
        self.learning = SCFLearningGate()
        self.observability = SCFObservabilityHub()

    def run_task(
        self,
        prompt: str,
        *,
        flags_override: Optional[SCFFeatureFlags] = None,
        recent_turns: Sequence[Any] = (),
        image_path: Optional[str] = None,
        video_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
        adaptive_thinking: bool = False,
        legacy_fallback_fn: Optional[Callable[[str], str]] = None,
        candidate_generators: Optional[Dict[str, Callable[[str], Dict[str, Any]]]] = None,
        enable_neural_candidates: bool = False,
        neural_checkpoint_path: Optional[str] = None,
        max_neural_tokens: int = 96,
    ) -> Dict[str, Any]:
        """Execute the 10-step cognitive loop, or clean legacy fallback when SCF is off."""

        flags = flags_override or self.config.flags
        started = time.perf_counter()
        was_tracing = self.observability.start_memory_probe()
        task_id = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()[:16]
        stages = []
        experimental_used: List[str] = []
        evidence_links: List[str] = []

        try:
            # 1 & 2. Understand task, complexity, and route specialist/model
            route_info, st_route = self.observability.measure_stage(
                "route",
                self.router.route,
                prompt,
                flags=flags,
            )
            stages.append(st_route)

            # Fallback path when master SCF flag is False (preserves baseline behavior)
            if not flags.SCF_ENABLED:
                legacy_text = (
                    str(legacy_fallback_fn(prompt))
                    if legacy_fallback_fn is not None
                    else ""
                )
                decoding_policy = {
                    "mode": "legacy_baseline",
                    "speculative_allowed": False,
                    "adaptive_thinking": False,
                    "reason": "SCF_ENABLED_false_baseline_fallback",
                }
                peak_kb = self.observability.finish_memory_probe(was_tracing)
                total_ms = round((time.perf_counter() - started) * 1000.0, 3)
                receipt = SCFExecutionReceipt(
                    task_id=task_id,
                    fallback_to_legacy=True,
                    selected_family="LegacyBaseline",
                    decoding_policy=decoding_policy,
                    flags_snapshot=flags.to_dict(),
                    experimental_components_used=[],
                    production_ready_claim=False,
                    verification_passed=None,
                    total_latency_ms=total_ms,
                    peak_traced_kb=peak_kb,
                    stages=stages,
                    evidence_links=[],
                )
                recorded = self.observability.record(receipt)
                return {
                    "scf_active": False,
                    "fallback_to_legacy": True,
                    "final_answer": legacy_text,
                    "routing": route_info,
                    "receipt": recorded,
                }

            # 7. Layered memory & firewall context
            memory_ctx, st_mem = self.observability.measure_stage(
                "memory",
                self.memory.build_context,
                prompt=prompt,
                recent_turns=recent_turns,
                enable_latent_memory=bool(flags.SCF_LATENT_MEMORY_ENABLED),
            )
            stages.append(st_mem)
            if flags.SCF_LATENT_MEMORY_ENABLED:
                experimental_used.append("MiMoMix_LatentStateReasoner")

            # 8. Adaptive compute budget & decode policy enforcement
            budget_info, st_budget = self.observability.measure_stage(
                "adaptive_budget",
                self.adaptive.evaluate_budget,
                prompt=prompt,
                flags=flags,
                adaptive_thinking=adaptive_thinking,
            )
            stages.append(st_budget)

            # Optional Multimodal Bridge (Step 5 specialist context)
            mm_ctx = None
            if flags.SCF_MULTIMODAL_ENABLED or any((image_path, video_path, mesh_path)):
                mm_ctx, st_mm = self.observability.measure_stage(
                    "multimodal",
                    self.multimodal.prepare_multimodal_context,
                    prompt=prompt,
                    image_path=image_path,
                    video_path=video_path,
                    mesh_path=mesh_path,
                )
                stages.append(st_mm)

            # 5 & 6 preliminary deterministic specialist probe
            norm_prompt = route_info["assessment"]["normalised_prompt"]
            if flags.SCF_VERIFICATION_ENABLED:
                verify_prelim, st_ver_pre = self.observability.measure_stage(
                    "verify_preliminary",
                    self.verifier.verify,
                    query=prompt,
                    normalised_query=norm_prompt,
                )
            else:
                verify_prelim, st_ver_pre = self.observability.measure_stage(
                    "verify_preliminary_disabled",
                    lambda: {
                        "verified": False,
                        "verified_answer": None,
                        "verifier_source": "disabled_by_flag",
                        "grounding_audit": {"override_applied": False, "response_text": ""},
                    },
                )
            stages.append(st_ver_pre)

            # 3 & 4. Collective candidate generation & critique (when enabled)
            merged_generators: Dict[str, Callable[[str], Dict[str, Any]]] = dict(
                candidate_generators or {}
            )
            if verify_prelim["verified"] and "NexusMind" not in merged_generators:
                merged_generators["NexusMind"] = lambda _q: {
                    "family": "NexusMind",
                    "answer": str(verify_prelim["verified_answer"]),
                    "confidence": 1.0,
                    "verified": True,
                    "rationale": f"Verified via {verify_prelim['verifier_source']}",
                }
            if enable_neural_candidates and "MiMoMix" not in merged_generators:
                mimo_adapter = self.registry.get_adapter("MiMoMix")

                def _mimo_live_gen(q_text: str) -> Dict[str, Any]:
                    out = mimo_adapter.infer_text(
                        q_text,
                        flags=flags,
                        checkpoint_path=neural_checkpoint_path,
                        max_new_tokens=max_neural_tokens,
                        adaptive_thinking=adaptive_thinking,
                    )
                    return {
                        "family": "MiMoMix",
                        "answer": str(out.get("answer") or ""),
                        "confidence": 0.62 if out.get("generated") else 0.0,
                        "verified": False,
                        "model_conditional": True,
                        "checkpoint": out.get("checkpoint"),
                        "decoding_policy": out.get("policy"),
                        "rationale": (
                            f"Checkpoint-backed candidate ({out.get('policy', {}).get('mode', 'greedy')}, "
                            f"accept_len={out.get('acceptance_length', 1.0)})"
                            if out.get("generated")
                            else f"Skipped: {out.get('reason')}"
                        ),
                    }

                merged_generators["MiMoMix"] = _mimo_live_gen

            collective_res: Optional[Dict[str, Any]] = None
            if flags.SCF_COLLECTIVE_REASONING_ENABLED:
                collective_res, st_coll = self.observability.measure_stage(
                    "collective",
                    self.collective.gather_and_synthesize,
                    prompt,
                    specialist_families=route_info["specialist_candidates"],
                    candidate_generators=merged_generators,
                    shared_context=memory_ctx.get("working_state"),
                    max_candidates=self.config.max_collective_candidates,
                )
                stages.append(st_coll)
                candidate_answer = str(collective_res["best_candidate"].get("answer") or "")
            else:
                gen_fn = merged_generators.get(route_info["selected_family"])
                candidate_answer = str(gen_fn(prompt).get("answer", "")) if gen_fn else ""

            # Reasoning Coordinator (GoT / Problem Decomposition / Causal DAG / Active Inference)
            reasoning_res, st_reason = self.observability.measure_stage(
                "reasoning",
                self.reasoning.reason,
                prompt,
                flags=flags,
                max_steps=self.config.max_reasoning_steps,
                working_state=memory_ctx.get("working_state"),
            )
            stages.append(st_reason)
            if flags.SCF_ACTIVE_INFERENCE_ENABLED:
                experimental_used.append("NexusMind_ActiveInferenceController")
            if route_info.get("hierarchical_moe"):
                experimental_used.append("ChampionNet_HierarchicalMoE_RoutingPolicy")

            # Optional MAP-Elites + SCAMPER/TRIZ Ideation synthesis (when ideation requested)
            submode = str((route_info.get("assessment") or {}).get("submode") or "dialogue")
            ws = memory_ctx.get("working_state") or {}
            active_constraints = [
                str(c.get("text", ""))
                for c in ws.get("active_commitments", [])
                if isinstance(c, dict) and c.get("text")
            ]
            ideation_res: Optional[Dict[str, Any]] = None
            if submode == "ideation":
                import nexus_ideation as nid

                ideation_res = nid.NexusIdeationEngine().brainstorm(
                    prompt,
                    count=4,
                    constraints=active_constraints,
                ).to_dict()

            # 6. Final independent verification pass
            if flags.SCF_VERIFICATION_ENABLED:
                verify_final, st_ver_final = self.observability.measure_stage(
                    "verify_final",
                    self.verifier.verify,
                    query=prompt,
                    candidate_answer=candidate_answer,
                    normalised_query=norm_prompt,
                )
            else:
                verify_final, st_ver_final = self.observability.measure_stage(
                    "verify_final_disabled",
                    lambda: {
                        "verified": False,
                        "verified_answer": None,
                        "verifier_source": "disabled_by_flag",
                        "derivation_steps": [],
                        "explanation": "",
                        "grounding_audit": {"override_applied": False, "response_text": candidate_answer},
                    },
                )
            stages.append(st_ver_final)

            # Build structured, honest Cognitive Synthesis block for callers/UI
            assistant_synthesis: Dict[str, Any]
            if verify_final["verified"] and verify_final["verified_answer"] is not None:
                final_answer = str(verify_final["verified_answer"])
                evidence_links.append(str(verify_final["verifier_source"]))
                deriv_steps = list(verify_final.get("derivation_steps") or [])
                assistant_synthesis = {
                    "mode": "verified_math",
                    "title": f"Verified Solution ({verify_final['verifier_source']})",
                    "summary": str(verify_final.get("explanation") or f"Verified answer: {final_answer}"),
                    "bullets": [
                        f"{s.get('description')}: {s.get('substitution_latex') or s.get('evaluated_value')}"
                        for s in deriv_steps[:4]
                    ],
                    "clarifying_questions": [],
                    "answer_authority": True,
                }
            elif ideation_res is not None:
                concepts = list(ideation_res.get("concepts") or [])
                assistant_synthesis = {
                    "mode": "ideation",
                    "title": "MAP-Elites & SCAMPER/TRIZ Idea Portfolio",
                    "summary": f"Generated {len(concepts)} niche-diverse, topic-grounded concepts with falsification tests.",
                    "bullets": [
                        f"[{c.get('operator')}] {c.get('title')}: {c.get('mechanism')} ({c.get('target_benefit')})"
                        for c in concepts[:4]
                    ],
                    "clarifying_questions": list((reasoning_res.get("problem_decomposition") or {}).get("clarifying_questions") or []),
                    "answer_authority": False,
                }
                final_answer = candidate_answer or ideation_res.get("synthesis_proposal", "")
                evidence_links.append("ideation.map_elites.advisory_only")
            elif submode == "problem_solving":
                prob = dict(reasoning_res.get("problem_decomposition") or {})
                assistant_synthesis = {
                    "mode": "problem_solving",
                    "title": "Structured Problem-Solving & Critic Plan",
                    "summary": f"Decomposed around {', '.join(prob.get('focus_terms') or ['target goal'])} with Crosswire ambiguity probes.",
                    "bullets": list(prob.get("steps") or [])[:4] + list(prob.get("critic_checks") or [])[:1],
                    "clarifying_questions": list(prob.get("clarifying_questions") or []),
                    "answer_authority": False,
                }
                final_answer = candidate_answer or (
                    f"[SCF advisory trace | family={route_info['selected_family']} | "
                    f"domain={route_info['assessment']['domain']}] "
                    f"{reasoning_res['got']['final_output']}"
                )
                evidence_links.append("reasoning.problem_decomposition.advisory_only")
            else:
                prob = dict(reasoning_res.get("problem_decomposition") or {})
                mem_bullets: List[str] = []
                if ws.get("style_preference"):
                    mem_bullets.append(f"Active style preference: {ws['style_preference']}")
                for comm in ws.get("active_commitments", [])[:2]:
                    if isinstance(comm, dict) and comm.get("text"):
                        mem_bullets.append(f"Session constraint ({comm.get('kind', 'note')}): {comm['text']}")
                for alert in ws.get("contradiction_alerts", [])[:1]:
                    if isinstance(alert, dict) and alert.get("summary"):
                        mem_bullets.append(f"Conflict check: {alert['summary']}")
                if not mem_bullets:
                    mem_bullets = list(prob.get("steps") or [])[:2]
                assistant_synthesis = {
                    "mode": "conversation_continuity",
                    "title": "Memory-Aware Conversation Continuity",
                    "summary": (
                        f"Tracked {ws.get('turn_count', 0)} prior turn(s), {len(active_constraints)} active constraint(s), "
                        f"and {len(ws.get('open_questions') or [])} open question(s)."
                    ),
                    "bullets": mem_bullets[:4],
                    "clarifying_questions": list(prob.get("clarifying_questions") or []),
                    "answer_authority": False,
                }
                if candidate_answer:
                    final_answer = str(verify_final["grounding_audit"]["response_text"] or candidate_answer)
                else:
                    final_answer = (
                        f"[SCF advisory trace | family={route_info['selected_family']} | "
                        f"domain={route_info['assessment']['domain']}] "
                        f"{reasoning_res['got']['final_output']}"
                    )
                    evidence_links.append("reasoning.got.advisory_only")

            learning_readiness = self.learning.inspect_readiness()
            peak_kb = self.observability.finish_memory_probe(was_tracing)
            total_ms = round((time.perf_counter() - started) * 1000.0, 3)

            receipt = SCFExecutionReceipt(
                task_id=task_id,
                fallback_to_legacy=False,
                selected_family=route_info["selected_family"],
                decoding_policy=budget_info["decoding_policy"],
                flags_snapshot=flags.to_dict(),
                experimental_components_used=experimental_used,
                production_ready_claim=bool(verify_final["verified"] and not experimental_used),
                verification_passed=bool(verify_final["verified"]),
                total_latency_ms=total_ms,
                peak_traced_kb=peak_kb,
                stages=stages,
                evidence_links=evidence_links,
            )
            recorded = self.observability.record(receipt)
            return {
                "scf_active": True,
                "fallback_to_legacy": False,
                "final_answer": final_answer,
                "verified": bool(verify_final["verified"]),
                "routing": route_info,
                "memory": memory_ctx,
                "adaptive_budget": budget_info,
                "multimodal": mm_ctx,
                "collective": collective_res,
                "reasoning": reasoning_res,
                "ideation": ideation_res,
                "assistant_synthesis": assistant_synthesis,
                "verification": verify_final,
                "learning_gate": {
                    "automatic_promotion_enabled": learning_readiness["automatic_promotion_enabled"],
                },
                "receipt": recorded,
            }
        except Exception:
            self.observability.finish_memory_probe(was_tracing)
            raise
