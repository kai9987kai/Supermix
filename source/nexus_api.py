"""Experimental NexusMind evidence-first API.

The API separates verified closed-world answers from heuristic analysis and
neural telemetry.  Internal routing, consensus, path, and ideation scores are
never presented as calibrated probabilities of correctness.

Service endpoints:
* ``POST /v1/solve`` -- Strict verifier-first solve surface plus audit-only legacy solver details
* ``POST /v1/innovate`` -- Heuristic SCAMPER/TRIZ ideation analysis
* ``POST /v1/chat`` -- Multi-turn conversational chat with persona adaptation and memory
* ``POST /v1/think`` -- Evidence-gated routing across the experimental subsystems
* ``POST /v1/swarm`` -- Bounded heuristic cognitive-swarm deliberation
* ``POST /v1/got`` -- Bounded heuristic graph search
* ``POST /v1/scientific`` -- Strict v71 closed-world deterministic solver
* ``GET /v1/personas`` -- Available conversation personas catalog
* ``GET /v1/telemetry`` -- Diagnostic configuration and synthetic metric probe
* ``POST /v1/feedback`` -- Fail-closed compatibility endpoint (no unverified learning)
* ``GET /v1/models`` -- Model Catalog and Routing Capabilities
* ``GET /health`` -- Health and Readiness Probe
"""

import argparse
import hashlib
import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import grounding_runtime as grounding
import mimomix_observatory as observatory
import nexus_chat as chat
import nexus_epistemics as epistemics
import nexus_got as got
import nexus_swarm as swarm
from nexus_engine import NexusEngine, NexusResult, build_default_engine


__all__ = [
    "ThinkRequest",
    "ThinkResponse",
    "SolveRequest",
    "InnovateRequest",
    "ChatTurnRequest",
    "SwarmRequest",
    "GoTRequest",
    "ScientificRequest",
    "FeedbackRequest",
    "EntropyRequest",
    "NexusApiService",
    "create_app",
    "main",
]


@dataclass
class ThinkRequest:
    messages: List[Dict[str, str]] = field(default_factory=list)
    prompt: Optional[str] = None
    mode: str = "auto"  # "auto" | "fast" | "deep" | "agent" | "swarm" | "got" | "scientific" | "solve" | "innovate" | "chat"
    max_output_tokens: int = 256
    thinking_budget: int = 4
    tools: List[Dict[str, Any]] = field(default_factory=list)
    persona: Optional[str] = None
    session_id: Optional[str] = None
    entropy_source: Optional[str] = "crypto"
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None


@dataclass
class EntropyRequest:
    source: str = "crypto"  # "crypto" | "seeded" | "os_csprng_transform" | "chaotic"
    count: int = 16
    seed: Optional[int] = None
    rule: int = 30
    ca_steps: int = 16
    ca_width: int = 31


@dataclass
class ThinkResponse:
    model: str
    mode_selected: str
    output: str
    confidence: Optional[float]
    latency_ms: float
    speculative_acceptance_rate: Optional[float]
    epistemics: Dict[str, Any] = field(default_factory=dict)
    thought_steps: List[Dict[str, Any]] = field(default_factory=list)
    audit_receipts: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveRequest:
    query: str


@dataclass
class InnovateRequest:
    topic: str
    count: int = 6


@dataclass
class ChatTurnRequest:
    session_id: str
    message: str
    persona: Optional[str] = None


@dataclass
class SwarmRequest:
    query: str
    max_rounds: int = 3
    context: Optional[str] = None


@dataclass
class GoTRequest:
    query: str
    max_depth: int = 4
    beam_width: int = 3


@dataclass
class ScientificRequest:
    query: str


@dataclass
class FeedbackRequest:
    difficulty: float
    epistemic_risk: float
    budget_used: int
    reward: float


_HEURISTIC_MODES = frozenset({"innovate", "chat", "swarm", "got"})
_EXACT_MODES = frozenset({"solve", "scientific"})
_NEURAL_MODES = frozenset({"fast", "deep", "agent"})
_NEXUS_MODES = frozenset({"auto", *_EXACT_MODES, *_HEURISTIC_MODES, *_NEURAL_MODES})
_NEURAL_INPUT_CHAR_LIMIT = 64
_STUDIO_PATH = Path(__file__).resolve().parent.parent / "web_static" / "nexus_studio.html"


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fresh_verified_grounding(
    query: str,
    *,
    require_science_plan: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the sole answer-authority boundary and accept only its strict outputs."""

    try:
        result = grounding.finalize_grounded_response("", query)
    except Exception:
        return None
    if not epistemics.verify_grounded_answer_result(
        result,
        receipt_schema_version=grounding.VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION,
        require_science_plan=require_science_plan,
    ):
        return None
    return dict(result)


def _verified_epistemics(grounded: Mapping[str, Any]) -> Dict[str, Any]:
    receipt = grounded.get("answer_receipt")
    receipt_sha256 = _canonical_sha256(receipt) if isinstance(receipt, Mapping) else ""
    return epistemics.verified_exact_decision(
        reason=f"fresh_grounding_recompute:{grounded.get('reason', 'verified')}",
        claim_scope=(
            "The deterministic result selected for this exact submitted query "
            "within the verifier's bounded grammar."
        ),
        verifier_id="grounding_runtime.finalize_grounded_response",
        verifier_receipt_sha256=receipt_sha256,
        protocol={
            "fresh_recompute": True,
            "grounding_reason": grounded.get("reason", ""),
            "answer_receipt_is_audit_metadata": True,
        },
    ).to_dict()


def _analysis_epistemics(
    *,
    reason: str,
    claim_scope: str,
    evidence_class: str = "deterministic_heuristic",
    internal_score: Optional[float] = None,
    internal_score_name: str = "",
    limitations: Sequence[str] = (),
    protocol: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return epistemics.analysis_only_decision(
        reason=reason,
        claim_scope=claim_scope,
        evidence_class=evidence_class,
        internal_score=internal_score,
        internal_score_name=internal_score_name,
        limitations=limitations,
        protocol=protocol,
    ).to_dict()


def _abstained_epistemics(
    *,
    reason: str,
    evidence_class: str = "no_applicable_verifier",
    limitations: Sequence[str] = (),
) -> Dict[str, Any]:
    return epistemics.abstained_decision(
        reason=reason,
        claim_scope="No answer was admitted for the submitted query.",
        evidence_class=evidence_class,
        limitations=limitations,
    ).to_dict()


def _result_epistemics(result: NexusResult) -> Dict[str, Any]:
    value = getattr(result, "epistemics", None)
    mode = _canonical_mode(getattr(result, "mode_selected", ""))
    if isinstance(value, Mapping) and epistemics.verify_epistemic_receipt(value):
        decision = value.get("decision")
        if decision == "answered":
            # This is only an internal branch signal. The engine receipt is
            # never returned; handle_think independently recomputes the query.
            return dict(value)
        if decision == "analysis_only" and mode in _HEURISTIC_MODES:
            evidence = "template_deliberation" if mode in {"swarm", "got"} else "deterministic_heuristic"
            return _analysis_epistemics(
                reason=f"api_rebuilt_{mode}_analysis_contract",
                claim_scope="Bounded analysis artifact without factual answer authority.",
                evidence_class=evidence,
                limitations=(
                    "The API rebuilt this receipt and did not trust engine-supplied receipt text or protocol fields.",
                ),
                protocol={"mode": mode, "verifier_calls": 0},
            )
        if decision == "abstained":
            return _abstained_epistemics(
                reason="api_rebuilt_engine_abstention",
                evidence_class="unverified_neural" if mode in _NEURAL_MODES else "no_applicable_verifier",
                limitations=(
                    "The API rebuilt this receipt and withheld engine-supplied receipt text and protocol fields.",
                ),
            )

    return _abstained_epistemics(
        reason="missing_or_invalid_engine_epistemic_receipt",
        evidence_class="unverified_neural" if mode not in _EXACT_MODES else "no_applicable_verifier",
        limitations=("The engine output was not admitted by a valid verifier receipt.",),
    )


def _canonical_mode(value: Any, fallback: str = "auto") -> str:
    candidate = str(value or "").strip().lower()
    if candidate in _NEXUS_MODES:
        return candidate
    fallback_clean = str(fallback or "auto").strip().lower()
    return fallback_clean if fallback_clean in _NEXUS_MODES else "auto"


def _model_id_for(mode: str, result_epistemics: Mapping[str, Any]) -> str:
    evidence_class = result_epistemics.get("evidence_class")
    if evidence_class == "verified_exact" or mode in _EXACT_MODES:
        return "nexus-exact-solver"
    if evidence_class in {"deterministic_heuristic", "template_deliberation"} or mode in _HEURISTIC_MODES:
        return "nexus-heuristic-suite"
    return "nexus-experimental-neural-telemetry"


def _analysis_steps(result: NexusResult) -> List[Dict[str, Any]]:
    """Expose heuristic traces while nulling every correctness-confidence slot."""

    rows: List[Dict[str, Any]] = []
    for step in list(getattr(result, "thought_steps", ()) or ()):
        row = step.to_dict() if hasattr(step, "to_dict") else dict(step)
        row["confidence"] = None
        rows.append(row)
    return rows


def _safe_abstention_telemetry(result: NexusResult) -> Dict[str, Any]:
    """Rebuild structural diagnostics; withhold arbitrary engine claims.

    Copying a whole nested mapping from an untrusted/invalid engine result
    would still permit an answer candidate to hide under an otherwise allowed
    top-level key.  Keep only the documented scalar fields and reconstruct the
    two bounded diagnostic mappings field by field.
    """

    raw = getattr(result, "telemetry", None)
    if not isinstance(raw, Mapping):
        return {"engine_telemetry_withheld": True}

    public: Dict[str, Any] = {}
    for key in (
        "q_budget_selected",
        "moe_expert_count",
        "sliding_window",
        "hybrid_ratio",
        "input_limit_characters",
        "declared_tool_count",
        "external_tool_calls_executed",
    ):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 1_000_000:
            public[key] = value

    if raw.get("neural_generator_ready") is False:
        public["neural_generator_ready"] = False
    if raw.get("q_budget_source") in {
        "safe_default",
        "verified_feedback_policy",
        "explicit_request_clamped",
    }:
        public["q_budget_source"] = raw["q_budget_source"]
    if raw.get("q_learning_update") == "skipped_requires_external_verified_feedback":
        public["q_learning_update"] = raw["q_learning_update"]

    synthetic = raw.get("synthetic_observability_probe")
    if isinstance(synthetic, Mapping) and synthetic.get("is_live_quality_evidence") is False:
        safe_synthetic: Dict[str, Any] = {"is_live_quality_evidence": False}
        for key in ("dem_lab_entropy", "rsi_novelty", "rsi_stability"):
            value = synthetic.get(key)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                safe_synthetic[key] = float(value)
        public["synthetic_observability_probe"] = safe_synthetic

    entropy = raw.get("entropy_telemetry")
    if isinstance(entropy, Mapping) and entropy.get("active_source") in {
        "crypto",
        "seeded",
        "os_csprng_transform",
        "chaotic",
    }:
        values = entropy.get("samples_preview")
        safe_values = []
        if isinstance(values, (list, tuple)):
            safe_values = [
                float(value)
                for value in values[:4]
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and 0.0 <= float(value) < 1.0
            ]
        safe_entropy: Dict[str, Any] = {
            "active_source": entropy["active_source"],
            "samples_preview": safe_values,
        }
        mean_value = entropy.get("mean_entropy_value")
        if (
            isinstance(mean_value, (int, float))
            and not isinstance(mean_value, bool)
            and math.isfinite(float(mean_value))
            and 0.0 <= float(mean_value) < 1.0
        ):
            safe_entropy["mean_entropy_value"] = float(mean_value)
        public["entropy_telemetry"] = safe_entropy

    public["engine_output_withheld_by_api_boundary"] = True
    return public


class NexusApiService:
    """Core framework-independent service handler for the NexusMind API."""

    def __init__(self, engine: Optional[NexusEngine] = None):
        self.engine = engine or build_default_engine()
        self._trusted_local_engine = type(self.engine) is NexusEngine
        self._lock = threading.Lock()

    def handle_think(self, req: ThinkRequest) -> ThinkResponse:
        api_started = time.perf_counter()
        query = req.prompt or ""
        if not query and req.messages:
            for m in reversed(req.messages):
                if m.get("role") == "user":
                    query = m.get("content", "")
                    break
            if not query and req.messages:
                query = req.messages[-1].get("content", "")

        if not query:
            query = "Hello NexusMind"

        requested_mode = _canonical_mode(req.mode)
        with self._lock:
            result = self.engine.process(
                query=query,
                mode=requested_mode,
                max_output_tokens=req.max_output_tokens,
                thinking_budget=req.thinking_budget,
                tools=req.tools,
                persona=req.persona,
                session_id=req.session_id,
                entropy_source=req.entropy_source,
            )

        result_epistemics = _result_epistemics(result)
        selected_mode = _canonical_mode(
            getattr(result, "mode_selected", ""),
            fallback=requested_mode,
        )
        decision = str(result_epistemics.get("decision") or "abstained")

        def api_latency_ms() -> float:
            """Measure the whole API path, including admission recomputation."""

            return round((time.perf_counter() - api_started) * 1000.0, 2)

        # A self-hashed engine receipt is not a trust signature. Any claimed
        # answer is independently recomputed at the API boundary from the
        # submitted query, and exact modes never degrade into raw analysis.
        if decision == "answered" or selected_mode in _EXACT_MODES:
            grounded = _fresh_verified_grounding(
                query,
                require_science_plan=selected_mode == "scientific",
            )
            if grounded is not None:
                public_epistemics = _verified_epistemics(grounded)
                answer_receipt = dict(grounded.get("answer_receipt") or {})
                return ThinkResponse(
                    model="nexus-exact-solver",
                    mode_selected=selected_mode,
                    output=str(grounded["text"]),
                    confidence=1.0,
                    latency_ms=api_latency_ms(),
                    speculative_acceptance_rate=None,
                    epistemics=public_epistemics,
                    thought_steps=[
                        {
                            "step_index": 1,
                            "stage": "api_verifier_recompute",
                            "content": "The API independently reran the strict closed-world verifier for this submitted query.",
                            "confidence": 1.0,
                            "telemetry": {
                                "score_kind": "deterministic_in_scope",
                                "verifier_calls": 1,
                            },
                        }
                    ],
                    audit_receipts={"verified_answer_receipt": answer_receipt},
                    telemetry={
                        "answer_admitted": True,
                        "api_fresh_recompute": True,
                        "grounding_reason": grounded.get("reason", ""),
                        "external_tool_calls_executed": 0,
                    },
                )

            result_epistemics = _abstained_epistemics(
                reason="api_fresh_reverification_failed",
                limitations=(
                    "The API independently reran the strict verifier and it did not admit the complete query.",
                    "The engine's candidate, confidence, trace, and receipts were withheld.",
                ),
            )
            decision = "abstained"

        if (
            decision == "analysis_only"
            and selected_mode in _HEURISTIC_MODES
            and self._trusted_local_engine
        ):
            output = str(getattr(result, "final_output", "") or "").strip()
            if not output.lower().startswith(("**analysis", "analysis")):
                output = "Analysis only — not a verified answer.\n\n" + output
            return ThinkResponse(
                model="nexus-heuristic-suite",
                mode_selected=selected_mode,
                output=output,
                confidence=None,
                latency_ms=api_latency_ms(),
                speculative_acceptance_rate=None,
                epistemics=result_epistemics,
                thought_steps=_analysis_steps(result),
                audit_receipts={},
                telemetry={
                    "analysis_artifact_kind": selected_mode,
                    "answer_verified": False,
                    "engine_output_scope": "trusted_local_deterministic_scaffold",
                },
            )

        if decision == "analysis_only":
            result_epistemics = _abstained_epistemics(
                reason="analysis_engine_not_trusted_at_api_boundary",
                evidence_class="no_applicable_verifier",
                limitations=(
                    "Only the built-in NexusEngine may expose deterministic analysis scaffolds through /v1/think.",
                ),
            )

        # Invalid, neural, agent, or otherwise non-authoritative engine results
        # cannot choose their public text or confidence on an abstention path.
        return ThinkResponse(
            model=_model_id_for(selected_mode, result_epistemics),
            mode_selected=selected_mode,
            output=(
                "I can't provide a verified answer for this request. No eligible "
                "fresh verifier admitted the complete query."
            ),
            confidence=None,
            latency_ms=api_latency_ms(),
            speculative_acceptance_rate=None,
            epistemics=result_epistemics,
            thought_steps=[
                {
                    "step_index": 1,
                    "stage": "api_abstention",
                    "content": "The API withheld the engine candidate because it lacked fresh answer authority.",
                    "confidence": None,
                    "telemetry": {"score_kind": "not_scored"},
                }
            ],
            audit_receipts={},
            telemetry=_safe_abstention_telemetry(result),
        )

    def handle_solve(self, req: SolveRequest) -> Dict[str, Any]:
        query = str(req.query or "").strip()
        grounded = _fresh_verified_grounding(query)
        with self._lock:
            legacy = self.engine.solver_engine.solve(query)

        legacy_audit: Dict[str, Any] = {
            "matched": bool(legacy.solved),
            "formula_id": str(legacy.formula_id if legacy.solved else ""),
            "candidate_withheld_unless_strict_gate_passes": True,
            "full_receipt_withheld": True,
            "receipt_is_authority": False,
        }
        if legacy.solved and legacy.receipt is not None:
            legacy_audit["receipt_schema_version"] = legacy.receipt.schema_version

        if grounded is None:
            decision = _abstained_epistemics(
                reason="strict_full_query_verifier_did_not_admit_answer",
                limitations=(
                    "The complete request was unsupported, ambiguous, negated, mixed-scope, or missing required assumptions.",
                    "A legacy formula-pattern match is retained only as audit metadata and its numeric candidate is withheld.",
                ),
            )
            return {
                "status": "abstained",
                "solved": False,
                "answer_authority": False,
                "output": (
                    "No verified answer was admitted. Submit one unambiguous closed-world "
                    "calculation with all required assumptions and units."
                ),
                "confidence": None,
                "epistemics": decision,
                "audit": {"legacy_nexus_solver": legacy_audit},
            }

        reasoning_row = dict(grounded.get("reasoning") or {})
        arithmetic_row = dict(grounded.get("arithmetic") or {})
        answer_row = dict(reasoning_row.get("answer") or {})
        if not str(answer_row.get("display") or "").strip():
            answer_row = {
                "exact": arithmetic_row.get("exact", ""),
                "display": arithmetic_row.get("display", ""),
                "approximation": arithmetic_row.get("approximation", ""),
                "approximate": False,
                "unit": "",
            }
        answer_receipt = dict(grounded.get("answer_receipt") or {})
        receipt = dict(answer_receipt)
        receipt["receipt_is_authority"] = False
        receipt["receipt_sha256"] = _canonical_sha256(receipt)
        method = str(reasoning_row.get("method") or answer_receipt.get("method") or "bounded_exact_arithmetic")
        step_rows = [
            {
                "step_index": index,
                "description": str(step),
                "formula_latex": method,
                "substitution_latex": "fresh deterministic recomputation",
                "evaluated_value": answer_row.get("display", ""),
                "unit": answer_row.get("unit", ""),
            }
            for index, step in enumerate(list(reasoning_row.get("steps") or ()), start=1)
        ]
        if not step_rows:
            step_rows.append(
                {
                    "step_index": 1,
                    "description": "Recompute the bounded arithmetic expression exactly.",
                    "formula_latex": method,
                    "substitution_latex": arithmetic_row.get("expression", ""),
                    "evaluated_value": answer_row.get("display", ""),
                    "unit": answer_row.get("unit", ""),
                }
            )
        return {
            "status": "answered",
            "solved": True,
            "answer_authority": True,
            "output": grounded["text"],
            "domain": "science" if reasoning_row.get("problem_class") == "scientific_scenario" else "arithmetic",
            "scenario": (reasoning_row.get("science_plan") or {}).get("scenario", ""),
            "target": (reasoning_row.get("science_plan") or {}).get("target", ""),
            "formula_id": method,
            "answer": answer_row,
            "answer_value": answer_row.get("exact", ""),
            "display_answer": answer_row.get("display", ""),
            "unit": answer_row.get("unit", ""),
            "steps": step_rows,
            "confidence": 1.0,
            "epistemics": _verified_epistemics(grounded),
            "receipt": receipt,
            "audit": {
                "verified_answer_receipt": answer_receipt,
                "legacy_nexus_solver": legacy_audit,
            },
        }

    def handle_innovate(self, req: InnovateRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.ideation_engine.brainstorm(
                req.topic,
                count=max(1, min(8, int(req.count))),
            )
        payload = res.to_dict()
        internal_score = float(res.receipt.top_composite_score)
        payload.update(
            {
                "status": "analysis_only",
                "answer_authority": False,
                "confidence": None,
                "internal_priority_score": internal_score,
                "score_semantics": "static_fnir_priority_not_measurement_or_correctness",
                "epistemics": _analysis_epistemics(
                    reason="structured_ideation_without_empirical_validation",
                    claim_scope="Concept hypotheses for prioritization and testing.",
                    internal_score=internal_score,
                    internal_score_name="static_fnir_priority",
                    limitations=(
                        "FNIR values are authored heuristic priorities, not measured feasibility, novelty, impact, or robustness.",
                        "Projected mechanisms and benefits require domain review and experiments.",
                    ),
                    protocol={"candidate_count": len(res.concepts), "verifier_calls": 0},
                ),
            }
        )
        return payload

    def handle_chat(self, req: ChatTurnRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.chat_engine.chat(
                session_id=req.session_id,
                user_input=req.message,
                requested_persona=req.persona,
            )
        payload = res.to_dict()
        payload.update(
            {
                "status": "analysis_only",
                "answer_authority": False,
                "confidence": None,
                "epistemics": _analysis_epistemics(
                    reason="persona_template_without_factual_verification",
                    claim_scope="Conversation scaffolding only.",
                    limitations=(
                        "Persona output is not checked by the exact solver or an external evidence tool.",
                    ),
                ),
            }
        )
        return payload

    def handle_personas(self) -> Dict[str, Any]:
        return {
            "personas": [p.to_dict() for p in chat.PERSONA_PROFILES.values()]
        }

    def handle_swarm(self, req: SwarmRequest) -> Dict[str, Any]:
        bounded_engine = swarm.SwarmEngine(
            agents=list(self.engine.swarm_engine.agents),
            max_rounds=req.max_rounds,
            convergence_threshold=self.engine.swarm_engine.convergence_threshold,
        )
        with self._lock:
            res = bounded_engine.deliberate(
                query=req.query,
                external_context=req.context,
            )
        payload = res.to_dict()
        internal_score = float(res.final_confidence)
        payload.update(
            {
                "status": "analysis_only",
                "answer_authority": False,
                "confidence": None,
                "internal_consensus_score": internal_score,
                "score_semantics": "template_agent_agreement_not_correctness",
                "epistemics": _analysis_epistemics(
                    reason="template_swarm_without_grounded_candidate",
                    claim_scope="Structured critique scaffold.",
                    evidence_class="template_deliberation",
                    internal_score=internal_score,
                    internal_score_name="template_agent_consensus",
                    limitations=(
                        "Default agents emit fixed role templates rather than independent factual candidates.",
                        "Agreement among templates is not verification.",
                    ),
                    protocol={"debate_rounds": len(res.rounds), "verifier_calls": 0},
                ),
            }
        )
        return payload

    def handle_got(self, req: GoTRequest) -> Dict[str, Any]:
        bounded_engine = got.GraphOfThoughts(
            max_depth=req.max_depth,
            beam_width=req.beam_width,
            prune_threshold=self.engine.got_engine.prune_threshold,
        )
        with self._lock:
            res = bounded_engine.search(query=req.query)
        payload = res.to_dict()
        internal_score = float(res.receipt.optimal_path_score)
        payload.update(
            {
                "status": "analysis_only",
                "answer_authority": False,
                "confidence": None,
                "internal_path_score": internal_score,
                "score_semantics": "template_position_priority_not_correctness_or_optimality",
                "epistemics": _analysis_epistemics(
                    reason="template_graph_search_without_answer_generator",
                    claim_scope="Search-topology scaffold.",
                    evidence_class="template_deliberation",
                    internal_score=internal_score,
                    internal_score_name="positional_path_priority",
                    limitations=(
                        "Default branches are deterministic placeholders and do not contain generated answers.",
                        "The selected path score is not a correctness or optimality measurement.",
                    ),
                    protocol={
                        "max_depth": bounded_engine.max_depth,
                        "beam_width": bounded_engine.beam_width,
                        "nodes_generated": res.receipt.total_nodes_generated,
                        "verifier_calls": 0,
                    },
                ),
            }
        )
        return payload

    def handle_scientific(self, req: ScientificRequest) -> Dict[str, Any]:
        grounded = _fresh_verified_grounding(req.query, require_science_plan=True)
        if grounded is None:
            return {
                "status": "abstained",
                "reason": "strict_science_verifier_did_not_admit_answer",
                "answer_authority": False,
                "confidence": None,
                "epistemics": _abstained_epistemics(
                    reason="strict_science_verifier_did_not_admit_answer",
                    limitations=(
                        "The query did not pass the allowlisted science-plan, dimensional, domain, and substitution checks.",
                    ),
                ),
            }
        reasoning_row = dict(grounded.get("reasoning") or {})
        answer_receipt = dict(grounded.get("answer_receipt") or {})
        return {
            "status": "answered",
            "answer_authority": True,
            "confidence": 1.0,
            "output": grounded["text"],
            "result": {
                "solved": True,
                "answer": dict(reasoning_row.get("answer") or {}),
                "method": reasoning_row.get("method", ""),
                "steps": list(reasoning_row.get("steps") or ()),
                "science_plan": dict(reasoning_row.get("science_plan") or {}),
            },
            "receipt": answer_receipt,
            "epistemics": _verified_epistemics(grounded),
        }

    def handle_entropy(self, req: EntropyRequest) -> Dict[str, Any]:
        effective_source = self.engine.entropy_engine.normalize_source(req.source)
        with self._lock:
            samples = self.engine.entropy_engine.sample(
                source=effective_source,
                count=req.count,
                seed=req.seed,
            )
            ca_grid = self.engine.entropy_engine.cellular_automata_step(
                rule=req.rule,
                steps=req.ca_steps,
                width=req.ca_width,
            )
        mean_val = sum(samples) / max(1, len(samples))
        var_val = sum((x - mean_val) ** 2 for x in samples) / max(1, len(samples))
        return {
            "source": effective_source,
            "requested_source": req.source,
            "provenance": self.engine.entropy_engine.source_provenance(effective_source),
            "count": len(samples),
            "samples": samples,
            "mean": round(mean_val, 4),
            "variance": round(var_val, 4),
            "rule": req.rule,
            "cellular_automata_grid": ca_grid,
            "status": "computed",
        }

    def handle_signals(self) -> Dict[str, Any]:
        with self._lock:
            q_summary = self.engine.q_policy.get_policy_summary()
            rsi_diag = self.engine.rsi_oscillator.update(0.5)
            entropy_sources = ["crypto", "seeded", "os_csprng_transform", "chaotic"]
            rsi_diag.update(
                {
                    "input_source": "constant_api_probe_0_5",
                    "is_live_reasoning_signal": False,
                }
            )
        return {
            "service": "NexusMind Experimental Signals Diagnostics v80",
            "q_policy": q_summary,
            "q_policy_role": "disconnected_experiment_not_live_routing",
            "rsi_diagnostic": rsi_diag,
            "entropy_sources_available": entropy_sources,
            "hybrid_attention": {
                "sliding_window": self.engine.config.sliding_window,
                "hybrid_ratio": self.engine.config.hybrid_ratio,
                "attention_sinks_enabled": True,
            },
            "sparse_moe": {
                "experts": self.engine.config.n_experts,
                "top_k": self.engine.config.top_k_experts,
                "auxiliary_loss_free": True,
            },
        }

    def handle_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            chsh_dict = observatory.chsh_value(
                {(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.5, (1, 1): -0.5}
            )
            ent = observatory.shannon_entropy([0.25, 0.25, 0.25, 0.25])
            policy_dict = self.engine.q_learner.to_dict()
            q_summary = self.engine.q_policy.get_policy_summary()
            latest_rsi = self.engine.rsi_oscillator.update(0.5)
            latest_rsi.update(
                {
                    "input_source": "constant_api_probe_0_5",
                    "is_live_reasoning_signal": False,
                }
            )
        return {
            "service": "NexusMind Experimental Evidence API v78.1",
            "status": "diagnostic_only",
            "answer_authority": False,
            "synthetic_metric_probe": {
                "chsh_bell_value": round(chsh_dict["s_value"], 4),
                "baseline_entropy": round(ent, 4),
                "input_is_live_model_output": False,
            },
            "moe_experts": self.engine.config.n_experts,
            "sliding_window": self.engine.config.sliding_window,
            "hybrid_ratio": self.engine.config.hybrid_ratio,
            "policy": policy_dict,
            "q_policy_summary": q_summary,
            "q_policy_role": "disconnected_experiment_not_live_routing",
            "rsi_diagnostic": latest_rsi,
        }

    def handle_feedback(self, req: FeedbackRequest) -> Dict[str, Any]:
        return {
            "status": "rejected",
            "policy_updated": False,
            "reason": "unverified_feedback_cannot_update_routing_policy",
            "message": (
                "This compatibility endpoint is fail-closed. Policy updates require a "
                "separate verifier-backed outcome receipt and are not accepted here."
            ),
        }

    def handle_models(self) -> Dict[str, Any]:
        return {
            "models": [
                {
                    "id": "nexus-exact-solver",
                    "description": "Fresh deterministic closed-world arithmetic and allowlisted science verification.",
                    "answer_status": "verified_exact_when_gate_passes",
                    "input_limit": "verifier grammar dependent",
                    "modes": ["solve", "scientific", "auto"],
                },
                {
                    "id": "nexus-heuristic-suite",
                    "description": "Deterministic persona, ideation, swarm-template, and graph-template analysis scaffolds.",
                    "answer_status": "analysis_only",
                    "scores_are_correctness_confidence": False,
                    "modes": ["innovate", "chat", "swarm", "got"],
                },
                {
                    "id": "nexus-experimental-neural-telemetry",
                    "description": "Newly initialized MiMo architecture probe with no loaded text-generation checkpoint.",
                    "answer_status": "abstains",
                    "generator_ready": False,
                    "input_limit_characters": _NEURAL_INPUT_CHAR_LIMIT,
                    "configured_sliding_window_tokens": self.engine.config.sliding_window,
                    "modes": ["fast", "deep", "agent"],
                },
            ],
            "catalog_claim_scope": "runtime capabilities observed in this source tree",
        }


def create_app(service: Optional[NexusApiService] = None):
    """Create FastAPI application if installed, or fallback to lightweight ASGI/WSGI app."""
    svc = service or NexusApiService()

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, Field

        app = FastAPI(
            title="NexusMind Experimental Evidence API",
            description=(
                "Verifier-first closed-world answers plus explicitly bounded heuristic "
                "analysis and neural architecture telemetry."
            ),
            version="78.1.0",
        )

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class PyThinkMessage(BaseModel):
            role: str = "user"
            content: str = ""

        class PyThinkRequest(BaseModel):
            messages: List[PyThinkMessage] = Field(default_factory=list)
            prompt: Optional[str] = None
            mode: str = "auto"
            max_output_tokens: int = 256
            thinking_budget: int = 4
            tools: List[Dict[str, Any]] = Field(default_factory=list)
            persona: Optional[str] = None
            session_id: Optional[str] = None
            entropy_source: Optional[str] = "crypto"

        class PyEntropyRequest(BaseModel):
            source: str = "crypto"
            count: int = 16
            seed: Optional[int] = None
            rule: int = 30
            ca_steps: int = 16
            ca_width: int = 31

        class PySolveRequest(BaseModel):
            query: str

        class PyInnovateRequest(BaseModel):
            topic: str
            count: int = 6

        class PyChatRequest(BaseModel):
            session_id: str
            message: str
            persona: Optional[str] = None

        class PySwarmRequest(BaseModel):
            query: str
            max_rounds: int = 3
            context: Optional[str] = None

        class PyGoTRequest(BaseModel):
            query: str
            max_depth: int = 4
            beam_width: int = 3

        class PyScientificRequest(BaseModel):
            query: str

        class PyFeedbackRequest(BaseModel):
            difficulty: float
            epistemic_risk: float
            budget_used: int
            reward: float

        @app.post("/v1/think")
        async def think_endpoint(req: PyThinkRequest):
            t_req = ThinkRequest(
                messages=[{"role": m.role, "content": m.content} for m in req.messages],
                prompt=req.prompt,
                mode=req.mode,
                max_output_tokens=req.max_output_tokens,
                thinking_budget=req.thinking_budget,
                tools=req.tools,
                persona=req.persona,
                session_id=req.session_id,
                entropy_source=req.entropy_source,
            )
            resp = svc.handle_think(t_req)
            return resp.to_dict()

        @app.post("/v1/entropy")
        async def entropy_endpoint(req: PyEntropyRequest):
            e_req = EntropyRequest(
                source=req.source,
                count=req.count,
                seed=req.seed,
                rule=req.rule,
                ca_steps=req.ca_steps,
                ca_width=req.ca_width,
            )
            return svc.handle_entropy(e_req)

        @app.get("/v1/signals")
        async def signals_endpoint():
            return svc.handle_signals()

        @app.post("/v1/solve")
        async def solve_endpoint(req: PySolveRequest):
            s_req = SolveRequest(query=req.query)
            return svc.handle_solve(s_req)

        @app.post("/v1/innovate")
        async def innovate_endpoint(req: PyInnovateRequest):
            i_req = InnovateRequest(topic=req.topic, count=req.count)
            return svc.handle_innovate(i_req)

        @app.post("/v1/chat")
        async def chat_endpoint(req: PyChatRequest):
            c_req = ChatTurnRequest(session_id=req.session_id, message=req.message, persona=req.persona)
            return svc.handle_chat(c_req)

        @app.get("/v1/personas")
        async def personas_endpoint():
            return svc.handle_personas()

        @app.post("/v1/swarm")
        async def swarm_endpoint(req: PySwarmRequest):
            s_req = SwarmRequest(query=req.query, max_rounds=req.max_rounds, context=req.context)
            return svc.handle_swarm(s_req)

        @app.post("/v1/got")
        async def got_endpoint(req: PyGoTRequest):
            g_req = GoTRequest(query=req.query, max_depth=req.max_depth, beam_width=req.beam_width)
            return svc.handle_got(g_req)

        @app.post("/v1/scientific")
        async def scientific_endpoint(req: PyScientificRequest):
            s_req = ScientificRequest(query=req.query)
            return svc.handle_scientific(s_req)

        @app.get("/v1/telemetry")
        async def telemetry_endpoint():
            return svc.handle_telemetry()

        @app.post("/v1/feedback")
        async def feedback_endpoint(req: PyFeedbackRequest):
            f_req = FeedbackRequest(
                difficulty=req.difficulty,
                epistemic_risk=req.epistemic_risk,
                budget_used=req.budget_used,
                reward=req.reward,
            )
            return svc.handle_feedback(f_req)

        @app.get("/v1/models")
        async def models_endpoint():
            return svc.handle_models()

        @app.get("/health")
        async def health_endpoint():
            return {
                "status": "ok",
                "service": "NexusMind Experimental Evidence API v78.1",
                "answer_policy": epistemics.SELECTIVE_ANSWER_POLICY_VERSION,
            }

        @app.get("/studio", include_in_schema=False)
        async def studio_endpoint():
            if not _STUDIO_PATH.is_file():
                raise HTTPException(status_code=404, detail="NexusMind Studio asset not found")
            return FileResponse(_STUDIO_PATH, media_type="text/html")

        return app

    except ImportError:
        return svc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NexusMind Experimental Evidence API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    import uvicorn
    app = create_app()
    print(f"[*] Starting NexusMind Experimental Evidence API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
