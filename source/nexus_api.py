"""Experimental NexusMind evidence-first API.

The API separates verified closed-world answers from heuristic analysis and
neural telemetry.  Internal routing, consensus, path, and ideation scores are
never presented as calibrated probabilities of correctness.

Service endpoints:
* ``POST /v1/solve`` -- Strict verifier-first solve surface plus audit-only legacy solver details
* ``POST /v1/innovate`` -- Heuristic SCAMPER/TRIZ ideation analysis
* ``POST /v1/chat`` -- Proof-carrying exact turns or persona analysis with memory
* ``POST /v1/think`` -- Evidence-gated routing across the experimental subsystems; ``stream=true`` enables proof-carrying SSE
* ``POST /v1/verify`` -- Fresh renderer revalidation of a proof-carrying answer
* ``POST /v1/swarm`` -- Bounded heuristic cognitive-swarm deliberation
* ``POST /v1/got`` -- Bounded heuristic graph search
* ``POST /v1/scientific`` -- Strict v71 closed-world deterministic solver
* ``GET /v1/personas`` -- Available conversation personas catalog
* ``GET /v1/telemetry`` -- Diagnostic configuration and synthetic metric probe
 * ``POST /v1/feedback`` -- Fail-closed compatibility endpoint (no unverified learning)
 * ``GET /v1/models`` -- Model Catalog and Routing Capabilities
 * ``GET /v1/risk-control`` -- Frozen selective-risk protocol (shadow only)
 * ``POST /v1/risk-control/audit`` -- Deterministic frozen arithmetic audit
 * ``POST /v1/risk-control/evaluate`` -- Evaluate caller-supplied shadow records
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import grounding_runtime as grounding
import mimomix_observatory as observatory
import nexus_chat as chat
import nexus_epistemics as epistemics
import nexus_got as got
import nexus_nonce_ledger as nonce_ledger
import nexus_proof as proof
import nexus_risk_control as risk_control
import nexus_swarm as swarm
from nexus_engine import NexusEngine, NexusResult, build_default_engine


_VERIFY_NONCE_TTL_SECONDS = 15 * 60
_VERIFY_NONCE_CACHE_SIZE = 4096


__all__ = [
    "ThinkRequest",
    "ThinkResponse",
    "SolveRequest",
    "InnovateRequest",
    "ChatTurnRequest",
    "SwarmRequest",
    "GoTRequest",
    "ScientificRequest",
    "VerifyRequest",
    "FeedbackRequest",
    "EntropyRequest",
    "BellRequest",
    "BellResponse",
    "ResonanceRequest",
    "ResonanceResponse",
    "CompareRequest",
    "CompareResponse",
    "QuantumStateRequest",
    "QuantumStateResponse",
    "GliderRequest",
    "GliderResponse",
    "TrajectoryRequest",
    "TrajectoryResponse",
    "SpeculativeTreeRequest",
    "SpeculativeTreeResponse",
    "SpeculativeDraftRequest",
    "SpeculativeDraftResponse",
    "MerminRequest",
    "MerminResponse",
    "ConwayRequest",
    "ConwayResponse",
    "ProofRepairRequest",
    "ProofRepairResponse",
    "CircuitAttributionRequest",
    "CircuitAttributionResponse",
    "ComplexityAnalysisRequest",
    "ComplexityAnalysisResponse",
    "AutoLoopStepRequest",
    "AutoLoopStepResponse",
    "SemanticInvariantsRequest",
    "SemanticInvariantsResponse",
    "ActiveInferenceRequest",
    "ActiveInferenceResponse",
    "ProofVerifyRequest",
    "ProofVerifyResponse",
    "BidirectionalSpeculationRequest",
    "BidirectionalSpeculationResponse",
    "EpistemicTreeSearchRequest",
    "EpistemicTreeSearchResponse",
    "DiffusionThoughtRequest",
    "DiffusionThoughtResponse",
    "ReflexionCorrectionRequest",
    "ReflexionCorrectionResponse",
    "ConformalStoppingRequest",
    "ConformalStoppingResponse",
    "CausalDAGRequest",
    "CausalDAGResponse",
    "NexusApiService",
    "create_app",
    "main",
]


@dataclass
class ThinkRequest:
    messages: List[Dict[str, str]] = field(default_factory=list)
    prompt: Optional[str] = None
    mode: str = "auto"  # "auto" | "adaptive" | "fast" | "deep" | "agent" | "swarm" | "got" | "scientific" | "solve" | "innovate" | "chat"
    max_output_tokens: int = 256
    thinking_budget: int = 4
    tools: List[Dict[str, Any]] = field(default_factory=list)
    persona: Optional[str] = None
    session_id: Optional[str] = None
    entropy_source: Optional[str] = "crypto"
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    request_nonce: str = ""


@dataclass
class EntropyRequest:
    source: str = "crypto"  # "crypto" | "seeded" | "os_csprng_transform" | "chaotic"
    count: int = 16
    seed: Optional[int] = None
    rule: int = 30
    ca_steps: int = 16
    ca_width: int = 31


@dataclass
class BellRequest:
    theta_a: float = 0.0
    theta_a_prime: float = 45.0
    theta_b: float = 22.5
    theta_b_prime: float = 67.5
    shots: int = 1000
    seed: Optional[int] = 42


@dataclass
class BellResponse:
    angles_deg: Dict[str, float]
    shots: int
    quantum_correlations: Dict[str, float]
    classical_correlations: Dict[str, float]
    chsh_s_quantum: float
    chsh_s_classical: float
    classical_bound: float
    tsirelson_bound: float
    violates_classical_bound: bool
    tsirelson_ratio: float
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResonanceRequest:
    query: str = ""


@dataclass
class ResonanceResponse:
    query: str
    archetype_scores: Dict[str, float]
    dominant_archetype: str
    resonance_score: float
    mixture_entropy: float
    coordinates_2d: Tuple[float, float]
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompareRequest:
    query_a: str
    query_b: Optional[str] = None
    mode_a: str = "auto"
    mode_b: str = "deep"
    entropy_source_a: str = "crypto"
    entropy_source_b: str = "seeded"


@dataclass
class CompareResponse:
    query_a: str
    query_b: str
    mode_a: str
    mode_b: str
    output_a: str
    output_b: str
    latency_ms_a: float
    latency_ms_b: float
    latency_delta_pct: float
    token_count_a: int
    token_count_b: int
    latency_class_a: str
    latency_class_b: str
    jensen_shannon_divergence: float
    semantic_distance: float
    rsi_a: float
    rsi_b: float
    summary_verdict: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QuantumStateRequest:
    parameter_p: float = 1.0
    noise_rate: float = 0.0
    channel_type: str = "depolarizing"  # "depolarizing" | "dephasing" | "unitary"


@dataclass
class QuantumStateResponse:
    parameter_p: float
    noise_rate: float
    channel_type: str
    density_matrix: List[List[float]]
    eigenvalues: List[float]
    von_neumann_entropy: float
    purity: float
    concurrence: float
    is_entangled: bool
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GliderRequest:
    glider_type_left: str = "glider_A"
    glider_type_right: str = "glider_C"
    separation: int = 10
    steps: int = 24
    width: int = 40


@dataclass
class GliderResponse:
    rule: int
    grid: List[List[int]]
    steps: int
    width: int
    ether_period: int
    gliders_identified: List[Dict[str, Any]]
    collision_events: List[Dict[str, Any]]
    logic_operation_analog: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryRequest:
    steps: List[str] = field(default_factory=list)


@dataclass
class TrajectoryResponse:
    steps: List[str]
    coordinates_2d: List[Tuple[float, float]]
    step_archetypes: List[str]
    velocities: List[float]
    curvatures: List[float]
    total_path_length: float
    net_cognitive_drift: float
    trajectory_dispersion_entropy: float
    summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SpeculativeTreeRequest:
    query: str
    branching_factor: int = 3
    max_depth: int = 4


@dataclass
class SpeculativeTreeResponse:
    query: str
    nodes: List[Dict[str, Any]]
    optimal_path_node_ids: List[str]
    total_nodes_evaluated: int
    backtracks_count: int
    final_output: str
    prm_mean_score: float
    receipt: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# v85 Request / Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeDraftRequest:
    prompt: str = "Explain quantum decoherence"
    target_acceptance: float = 0.75
    local_entropy: float = 0.5
    steps: int = 4


@dataclass
class SpeculativeDraftResponse:
    prompt: str
    steps_executed: int
    draft_lengths: List[int]
    mean_draft_length: float
    accepted_tokens: int
    rejected_tokens: int
    acceptance_rate: float
    theoretical_speedup: float
    emitted_sequence: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MerminRequest:
    state_type: str = "GHZ"  # "GHZ" | "W" | "separable"


@dataclass
class MerminResponse:
    state_type: str
    mermin_m3: float
    classical_bound: float
    quantum_maximum: float
    violates_classical_bound: bool
    observables: Dict[str, float]
    state_fidelity: float
    summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConwayRequest:
    pattern_name: str = "glider"
    steps: int = 16
    height: int = 24
    width: int = 24


@dataclass
class ConwayResponse:
    pattern_name: str
    grid: List[List[int]]
    steps: int
    height: int
    width: int
    active_cells_trajectory: List[int]
    spatial_block_entropy: float
    morphological_complexity: float
    detected_period: Optional[int]
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProofRepairRequest:
    assertions: List[str] = field(default_factory=lambda: ["A implies B", "B implies C"])


@dataclass
class ProofRepairResponse:
    original_assertions: List[str]
    satisfiable: bool
    detected_contradictions: List[str]
    repaired_assertions: List[str]
    repair_operations_applied: List[str]
    receipt: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitAttributionRequest:
    prompt: str
    target_token: str
    contrast_token: Optional[str] = None
    clean_prompt: Optional[str] = None
    corrupt_prompt: Optional[str] = None
    patch_layer: Optional[int] = None
    patch_head: Optional[int] = None
    test_scratchpad: bool = False
    trace_steps: List[str] = field(default_factory=list)
    next_operation: Optional[str] = None


@dataclass
class CircuitAttributionResponse:
    prompt: str
    target_token: str
    components: List[Dict[str, Any]]
    activation_patch: Optional[Dict[str, Any]] = None
    causal_register: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplexityAnalysisRequest:
    text: str
    compare_text: Optional[str] = None
    window_size: int = 8


@dataclass
class ComplexityAnalysisResponse:
    profile: Dict[str, Any]
    ncd_comparison: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AutoLoopStepRequest:
    query: str
    reward_feedback: Optional[float] = None
    forced_action: Optional[str] = None


@dataclass
class AutoLoopStepResponse:
    iteration: int
    active_query: str
    selected_mode: str
    rsi_value: float
    rsi_regime: str
    reward_awarded: float
    q_value_updated: float
    entropy_sample: float
    complexity_compression_ratio: float
    loop_status: str
    step_receipt: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SemanticInvariantsRequest:
    problem: str
    ground_truth_answer: Optional[str] = None
    task_type: str = "arithmetic"


@dataclass
class SemanticInvariantsResponse:
    canonical_problem: str
    canonical_answer: str
    invariant_paraphrase: str
    operand_reordered: Optional[str]
    distractor_variant: str
    contrast_problem: str
    contrast_expected_answer: str
    invariance_score: float
    contrast_distinction_passed: bool
    all_equivalent_consistent: bool
    stability_classification: str
    variants_evaluated: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActiveInferenceRequest:
    query: str
    current_trace_steps: List[str] = field(default_factory=list)
    local_entropy: float = 0.85
    rsi_volatility: float = 50.0
    verification_confidence: float = 0.80
    has_pending_subgoals: bool = False


@dataclass
class ActiveInferenceResponse:
    query: str
    current_state_summary: str
    local_entropy: float
    rsi_volatility: float
    precision_beta: float
    candidate_actions: List[Dict[str, Any]]
    selected_action: Dict[str, Any]
    epistemic_pragmatic_ratio: float
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProofVerifyRequest:
    problem: str
    trace_steps: List[str] = field(default_factory=list)


@dataclass
class ProofVerifyResponse:
    problem: str
    has_error: bool
    first_error_index: int
    error_category: str
    error_step_text: Optional[str]
    diagnostic_explanation: str
    step_records: List[Dict[str, Any]]
    repaired_trace: List[str]
    verified_final_answer: Optional[str]
    proof_fidelity_score: float
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BidirectionalSpeculationRequest:
    problem: str
    candidate_answer: Optional[str] = None


@dataclass
class BidirectionalSpeculationResponse:
    problem: str
    forward_draft: str
    forward_answer: str
    reverse_draft: str
    reverse_inferred_premise: str
    expected_premise: str
    consistency_score: float
    is_accepted: bool
    rejection_reason: Optional[str]
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpistemicTreeSearchRequest:
    query: str
    max_depth: int = 4
    beam_width: int = 3


@dataclass
class EpistemicTreeSearchResponse:
    query: str
    optimal_trace: List[str]
    verified_answer: Optional[str]
    total_nodes_evaluated: int
    pruned_branches_count: int
    mean_efe: float
    all_nodes: List[Dict[str, Any]]
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffusionThoughtRequest:
    problem: str
    num_timesteps: int = 20
    guidance_scale: float = 3.0
    latent_dim: int = 16
    seed: int = 42


@dataclass
class DiffusionThoughtResponse:
    problem: str
    total_steps: int
    diffusion_trajectory: List[Dict[str, Any]]
    crystallized_thought: str
    crystallization_threshold: float
    is_crystallized: bool
    stability_drift: float
    mean_step_jsd: float
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflexionCorrectionRequest:
    problem: str
    proposed_solution: str
    ground_truth: Optional[str] = None
    max_iterations: int = 3


@dataclass
class ReflexionCorrectionResponse:
    problem: str
    initial_solution: str
    corrected_solution: str
    iterations_used: int
    is_verified_correct: bool
    active_constraints: List[str]
    epistemic_history: List[Dict[str, Any]]
    diagnostic_summary: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConformalStoppingRequest:
    step_entropy: float = 0.4
    rsi_volatility: float = 40.0
    verifier_score: float = 0.85
    step_index: int = 3
    total_budget: int = 10
    target_error_rate: float = 0.05


@dataclass
class ConformalStoppingResponse:
    should_stop: bool
    step_index: int
    total_budget: int
    empirical_nonconformity: float
    calibrated_threshold: float
    target_error_rate: float
    finite_sample_bound: float
    safety_guaranteed: bool
    diagnostic_reason: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CausalDAGRequest:
    scenario: str = "physics_newton"
    treatment_node: str = "force"
    outcome_node: str = "acceleration"
    do_value: float = 10.0
    observed_context: Optional[Dict[str, float]] = None


@dataclass
class CausalDAGResponse:
    scenario: str
    treatment_node: str
    outcome_node: str
    do_value: float
    backdoor_adjustment_set: List[str]
    is_identifiable: bool
    interventional_estimate: float
    counterfactual_estimate: Optional[float]
    causal_graph_nodes: List[str]
    causal_graph_edges: List[List[str]]
    diagnostic_explanation: str
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
    proof_capsule: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SolveRequest:
    query: str
    request_nonce: str = ""


@dataclass
class InnovateRequest:
    topic: str
    count: int = 6


@dataclass
class ChatTurnRequest:
    session_id: str
    message: str
    persona: Optional[str] = None
    request_nonce: str = ""


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
    request_nonce: str = ""


@dataclass
class VerifyRequest:
    query: str
    output: str
    display_answer: str
    surface: str
    proof_capsule: Dict[str, Any] = field(default_factory=dict)
    request_nonce: str = ""


@dataclass
class FeedbackRequest:
    difficulty: float
    epistemic_risk: float
    budget_used: int
    reward: float


_HEURISTIC_MODES = frozenset({"innovate", "chat", "swarm", "got"})
_EXACT_MODES = frozenset({"solve", "scientific"})
_NEURAL_MODES = frozenset({"fast", "deep", "adaptive", "agent"})
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
        runtime_version=grounding.GROUNDING_RUNTIME_VERSION,
        require_science_plan=require_science_plan,
    ):
        return None
    return dict(result)


def _proof_capsule(
    query: str,
    grounded: Mapping[str, Any],
    surface: str,
    request_nonce: str = "",
) -> Optional[Dict[str, Any]]:
    return proof.build_proof_capsule(
        query=query,
        grounded=grounded,
        receipt_schema_version=grounding.VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION,
        runtime_version=grounding.GROUNDING_RUNTIME_VERSION,
        surface=surface,
        request_nonce=request_nonce,
    )


def _verified_epistemics(
    grounded: Mapping[str, Any],
    *,
    query: str,
    surface: str,
    request_nonce: str = "",
) -> Dict[str, Any]:
    if not proof.valid_request_nonce(request_nonce):
        raise ValueError("verified API answers require a valid request nonce")
    receipt = grounded.get("answer_receipt")
    receipt_sha256 = _canonical_sha256(receipt) if isinstance(receipt, Mapping) else ""
    output_text = str(grounded.get("text") or "")
    nonce_sha256 = proof.text_sha256(request_nonce)
    return epistemics.verified_exact_decision(
        reason=f"fresh_grounding_recompute:{grounded.get('reason', 'verified')}",
        claim_scope=(
            "The deterministic result selected for this exact submitted query "
            "within the verifier's bounded grammar."
        ),
        verifier_id="grounding_runtime.finalize_grounded_response",
        request_sha256=proof.text_sha256(query),
        output_sha256=proof.text_sha256(output_text),
        verifier_receipt_sha256=receipt_sha256,
        request_nonce_sha256=nonce_sha256,
        surface=surface,
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

    compute = raw.get("compute_budget_report")
    if (
        isinstance(compute, Mapping)
        and compute.get("mode") == "adaptive"
        and compute.get("policy_evidence")
        == "authored_shadow_heuristic_not_calibrated"
        and compute.get("execution_authorized") is False
        and compute.get("answer_authority") is False
    ):
        cycles = compute.get("applied_max_cycles", compute.get("allocated_cycles"))
        executed = compute.get("executed_mechanisms")
        if (
            isinstance(cycles, int)
            and not isinstance(cycles, bool)
            and 1 <= cycles <= 64
            and isinstance(executed, Mapping)
            and executed.get("applied_max_cycles") == cycles
            and executed.get("adaptive_thinking") is True
        ):
            observed_cycles = executed.get("observed_cycles")
            if observed_cycles is not None and (
                not isinstance(observed_cycles, int)
                or isinstance(observed_cycles, bool)
                or not 1 <= observed_cycles <= cycles
            ):
                observed_cycles = None
            safe_executed: Dict[str, Any] = {
                "requested_cycles": cycles,
                "applied_max_cycles": cycles,
                "observed_cycles": observed_cycles,
                "exit_reason": str(executed.get("exit_reason", "unknown"))[:64],
                "adaptive_thinking": True,
                "differential_attention": executed.get("differential_attention") is True,
                "mixture_of_depths": executed.get("mixture_of_depths") is True,
                "multi_latent_attention": executed.get("multi_latent_attention") is True,
            }
            applied_ratio = executed.get("mod_capacity_ratio")
            if (
                isinstance(applied_ratio, (int, float))
                and not isinstance(applied_ratio, bool)
                and math.isfinite(float(applied_ratio))
                and 0.0 < float(applied_ratio) <= 1.0
            ):
                safe_executed["mod_capacity_ratio"] = float(applied_ratio)
            else:
                safe_executed["mod_capacity_ratio"] = None

            safe_compute: Dict[str, Any] = {
                "mode": "adaptive",
                "allocated_cycles": cycles,
                "shadow_recommended_cycles": (
                    compute.get("shadow_recommended_cycles")
                    if isinstance(compute.get("shadow_recommended_cycles"), int)
                    and not isinstance(compute.get("shadow_recommended_cycles"), bool)
                    and 1 <= compute.get("shadow_recommended_cycles") <= 64
                    else None
                ),
                "applied_max_cycles": cycles,
                "budget_source": str(compute.get("budget_source", "unknown"))[:64],
                "shadow_recommendation_applied": False,
                "policy_evidence": "authored_shadow_heuristic_not_calibrated",
                "execution_authorized": False,
                "answer_authority": False,
                "halting_policy_trained": False,
                "policy_calibrated": False,
                "executed_mechanisms": safe_executed,
                "optional_mechanism_request_applied": (
                    compute.get("optional_mechanism_request_applied") is True
                ),
                "report_scope": (
                    "observed_single_forward_telemetry_not_quality_or_calibration"
                ),
            }
            raw_census = compute.get("module_census")
            safe_census: Dict[str, Any] = {}
            if isinstance(raw_census, Mapping):
                for name in (
                    "differential_attention",
                    "mixture_of_depths",
                    "multi_latent_attention",
                ):
                    row = raw_census.get(name)
                    if isinstance(row, Mapping):
                        safe_census[name] = {
                            "available": row.get("available") is True,
                            "configured": row.get("configured") is True,
                            "executed": row.get("executed") is True,
                            "efficiency_validated": row.get("efficiency_validated") is True,
                        }
            safe_compute["module_census"] = safe_census
            requested_ratio = compute.get("requested_mod_capacity_ratio")
            if (
                isinstance(requested_ratio, (int, float))
                and not isinstance(requested_ratio, bool)
                and math.isfinite(float(requested_ratio))
                and 0.0 < float(requested_ratio) <= 1.0
            ):
                safe_compute["requested_mod_capacity_ratio"] = float(requested_ratio)
            safe_compute["requested_differential_attention"] = (
                compute.get("requested_differential_attention") is True
            )
            public["compute_budget_report"] = safe_compute

    public["engine_output_withheld_by_api_boundary"] = True
    return public


class NexusApiService:
    """Core framework-independent service handler for the NexusMind API."""

    def __init__(
        self,
        engine: Optional[NexusEngine] = None,
        *,
        verification_nonce_store: Optional[nonce_ledger.NonceLedger] = None,
        verification_nonce_db: Optional[str | Path] = None,
    ):
        if verification_nonce_store is not None and verification_nonce_db is not None:
            raise ValueError("choose verification_nonce_store or verification_nonce_db, not both")
        self.engine = engine or build_default_engine()
        self._trusted_local_engine = type(self.engine) is NexusEngine
        self._lock = threading.Lock()
        self._verification_nonce_store = (
            verification_nonce_store
            if verification_nonce_store is not None
            else (
                nonce_ledger.SQLiteNonceLedger(
                    verification_nonce_db,
                    ttl_seconds=_VERIFY_NONCE_TTL_SECONDS,
                    max_entries=_VERIFY_NONCE_CACHE_SIZE,
                )
                if verification_nonce_db is not None
                else nonce_ledger.InMemoryNonceLedger(
                    ttl_seconds=_VERIFY_NONCE_TTL_SECONDS,
                    max_entries=_VERIFY_NONCE_CACHE_SIZE,
                )
            )
        )

    def _ensure_verification_nonce_store(self) -> nonce_ledger.NonceLedger:
        """Lazily support contract-only ``__new__`` test services as well."""

        store = getattr(self, "_verification_nonce_store", None)
        if store is None:
            store = nonce_ledger.InMemoryNonceLedger(
                ttl_seconds=_VERIFY_NONCE_TTL_SECONDS,
                max_entries=_VERIFY_NONCE_CACHE_SIZE,
            )
            self._verification_nonce_store = store
        return store

    def _verification_nonce_seen(self, nonce: str) -> bool:
        """Return whether an eligible nonce was already accepted recently."""

        if not nonce:
            return False
        key = proof.text_sha256(nonce)
        return self._ensure_verification_nonce_store().seen(key)

    def _remember_verification_nonce(self, nonce: str) -> bool:
        """Atomically consume a successful nonce; return false on a race/replay."""

        if not nonce:
            return True
        key = proof.text_sha256(nonce)
        return self._ensure_verification_nonce_store().consume(key)

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
            nonce_valid = proof.valid_request_nonce(req.request_nonce)
            grounded = _fresh_verified_grounding(
                query,
                require_science_plan=selected_mode == "scientific",
            )
            capsule = (
                _proof_capsule(query, grounded, "think", req.request_nonce)
                if grounded is not None
                else None
            )
            if grounded is not None and capsule is not None:
                public_epistemics = _verified_epistemics(
                    grounded,
                    query=query,
                    surface="think",
                    request_nonce=req.request_nonce,
                )
                answer_receipt = dict(grounded.get("answer_receipt") or {})
                return ThinkResponse(
                    model="nexus-exact-solver",
                    mode_selected=selected_mode,
                    output=str(grounded["text"]),
                    confidence=None,
                    latency_ms=api_latency_ms(),
                    speculative_acceptance_rate=None,
                    epistemics=public_epistemics,
                    thought_steps=[
                        {
                            "step_index": 1,
                            "stage": "api_verifier_recompute",
                            "content": "The API independently reran the strict closed-world verifier for this submitted query.",
                            "confidence": None,
                            "telemetry": {
                                "assurance_kind": "deterministic_assurance_not_probability",
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
                        "assurance_kind": "deterministic_assurance_not_probability",
                    },
                    proof_capsule=capsule,
                )

            result_epistemics = _abstained_epistemics(
                reason=(
                    "valid_request_nonce_required"
                    if grounded is not None and not nonce_valid
                    else "api_fresh_reverification_failed"
                ),
                limitations=(
                    (
                        "The strict verifier admitted the query, but public answer authority requires a 16-128 character ASCII request nonce."
                        if grounded is not None and not nonce_valid
                        else "The API independently reran the strict verifier and it did not admit the complete query."
                    ),
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

    def handle_think_stream(self, req: ThinkRequest):
        """Yield real-time JSON event dicts for Server-Sent Events streaming."""
        prompt_text = req.prompt or (req.messages[-1]["content"] if req.messages else "")
        yield {
            "event": "start",
            "stream_contract": "nexus-sse-proof-carrying-v1",
            "mode": req.mode,
            "prompt": prompt_text,
            "timestamp": time.time(),
        }

        resp = self.handle_think(req)

        for step in resp.thought_steps:
            yield {
                "event": "thinking_step",
                "step_index": step.get("step_index", 1),
                "stage": step.get("stage", "ponder"),
                "content": step.get("content", ""),
                "telemetry": step.get("telemetry", {}),
            }

        out_text = resp.output or ""
        chunk_size = max(1, len(out_text) // 5) if len(out_text) > 10 else len(out_text)
        chunks = [out_text[i : i + chunk_size] for i in range(0, len(out_text), chunk_size)]
        if chunks:
            for chunk_index, chunk in enumerate(chunks):
                yield {
                    "event": "token",
                    "delta": chunk,
                    "chunk_index": chunk_index,
                    "chunk_count": len(chunks),
                }

        yield {
            "event": "telemetry",
            "model": resp.model,
            "mode_selected": resp.mode_selected,
            "confidence": resp.confidence,
            "latency_ms": resp.latency_ms,
            "epistemics": resp.epistemics,
            "telemetry": resp.telemetry,
            "audit_receipts": resp.audit_receipts,
            "proof_capsule": resp.proof_capsule,
        }

        yield {
            "event": "done",
            "stream_contract": "nexus-sse-proof-carrying-v1",
            "status": str(resp.epistemics.get("decision") or "abstained"),
            "proof_capsule_sha256": str(resp.proof_capsule.get("capsule_sha256") or ""),
        }

    def handle_solve(self, req: SolveRequest) -> Dict[str, Any]:
        query = str(req.query or "").strip()
        nonce_valid = proof.valid_request_nonce(req.request_nonce)
        grounded = _fresh_verified_grounding(query)
        capsule = (
            _proof_capsule(query, grounded, "solve", req.request_nonce)
            if grounded is not None
            else None
        )
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

        if grounded is None or capsule is None:
            nonce_required = grounded is not None and not nonce_valid
            decision = _abstained_epistemics(
                reason=(
                    "valid_request_nonce_required"
                    if nonce_required
                    else "strict_full_query_verifier_did_not_admit_answer"
                ),
                limitations=(
                    (
                        "The calculation passed grounding, but public answer authority requires a 16-128 character ASCII request nonce."
                        if nonce_required
                        else "The complete request was unsupported, ambiguous, negated, mixed-scope, or missing required assumptions."
                    ),
                    "A legacy formula-pattern match is retained only as audit metadata and its numeric candidate is withheld.",
                ),
            )
            return {
                "status": "abstained",
                "solved": False,
                "answer_authority": False,
                "output": (
                    "No verified answer was admitted. Supply a valid request nonce and "
                    "resubmit this calculation."
                    if nonce_required
                    else "No verified answer was admitted. Submit one unambiguous closed-world "
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
            "confidence": None,
            "assurance_kind": "deterministic_assurance_not_probability",
            "epistemics": _verified_epistemics(
                grounded,
                query=query,
                surface="solve",
                request_nonce=req.request_nonce,
            ),
            "receipt": receipt,
            "proof_capsule": capsule,
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
        query = str(req.message or "").strip()
        nonce_valid = proof.valid_request_nonce(req.request_nonce)
        grounded = _fresh_verified_grounding(query)
        capsule = (
            _proof_capsule(query, grounded, "chat", req.request_nonce)
            if grounded is not None
            else None
        )
        if grounded is not None and not nonce_valid:
            message = (
                "No verified answer was admitted. Public answer authority requires "
                "a valid 16-128 character ASCII request nonce."
            )
            return {
                "status": "abstained",
                "answer_authority": False,
                "confidence": None,
                "reply": message,
                "output": message,
                "conversation_state_updated": False,
                "epistemics": _abstained_epistemics(
                    reason="valid_request_nonce_required",
                    limitations=(
                        "The strict grounder admitted the complete chat turn, but the request lacked an eligible freshness nonce.",
                    ),
                ),
            }
        if grounded is not None and capsule is not None:
            result_row = dict(capsule.get("result") or {})
            return {
                "status": "answered",
                "answer_authority": True,
                "confidence": None,
                "assurance_kind": "deterministic_assurance_not_probability",
                "reply": str(grounded.get("text") or ""),
                "output": str(grounded.get("text") or ""),
                "display_answer": result_row.get("display_answer", ""),
                "unit": result_row.get("unit", ""),
                "intent_detected": "verified_closed_world",
                "persona_used": {
                    "persona_type": "verified_solver",
                    "display_name": "Nexus Verifier",
                },
                "conversation_state_updated": False,
                "thought_steps": [
                    "The strict grounder admitted the complete chat turn.",
                    "The numeric claim capsule must be revalidated before rendering.",
                ],
                "proof_capsule": capsule,
                "epistemics": _verified_epistemics(
                    grounded,
                    query=query,
                    surface="chat",
                    request_nonce=req.request_nonce,
                ),
            }

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
        nonce_valid = proof.valid_request_nonce(req.request_nonce)
        grounded = _fresh_verified_grounding(req.query, require_science_plan=True)
        capsule = (
            _proof_capsule(req.query, grounded, "scientific", req.request_nonce)
            if grounded is not None
            else None
        )
        if grounded is None or capsule is None:
            nonce_required = grounded is not None and not nonce_valid
            return {
                "status": "abstained",
                "reason": (
                    "valid_request_nonce_required"
                    if nonce_required
                    else "strict_science_verifier_did_not_admit_answer"
                ),
                "answer_authority": False,
                "confidence": None,
                "epistemics": _abstained_epistemics(
                    reason=(
                        "valid_request_nonce_required"
                        if nonce_required
                        else "strict_science_verifier_did_not_admit_answer"
                    ),
                    limitations=(
                        (
                            "The science query passed grounding, but public answer authority requires a valid 16-128 character ASCII request nonce."
                            if nonce_required
                            else "The query did not pass the allowlisted science-plan, dimensional, domain, and substitution checks."
                        ),
                    ),
                ),
            }
        reasoning_row = dict(grounded.get("reasoning") or {})
        answer_receipt = dict(grounded.get("answer_receipt") or {})
        return {
            "status": "answered",
            "answer_authority": True,
            "confidence": None,
            "assurance_kind": "deterministic_assurance_not_probability",
            "output": grounded["text"],
            "result": {
                "solved": True,
                "answer": dict(reasoning_row.get("answer") or {}),
                "method": reasoning_row.get("method", ""),
                "steps": list(reasoning_row.get("steps") or ()),
                "science_plan": dict(reasoning_row.get("science_plan") or {}),
            },
            "receipt": answer_receipt,
            "proof_capsule": capsule,
            "epistemics": _verified_epistemics(
                grounded,
                query=req.query,
                surface="scientific",
                request_nonce=req.request_nonce,
            ),
        }

    def handle_verify(self, req: VerifyRequest) -> Dict[str, Any]:
        """Freshly revalidate a renderer capsule without echoing rejected claims."""

        query = str(req.query or "")
        output = str(req.output or "")
        display_answer = str(req.display_answer or "")
        request_nonce = str(req.request_nonce or "")
        if not proof.valid_request_nonce(request_nonce):
            return {
                "status": "rejected",
                "verified": False,
                "reason": "valid_request_nonce_required",
                "confidence": None,
                "assurance_kind": "none",
                "renderer_may_mark_numeric_claims_verified": False,
                "fresh_verifier_calls": 0,
                "capsule_sha256": "",
            }
        if self._verification_nonce_seen(request_nonce):
            return {
                "status": "rejected",
                "verified": False,
                "reason": "request_nonce_replayed",
                "confidence": None,
                "assurance_kind": "none",
                "renderer_may_mark_numeric_claims_verified": False,
                "fresh_verifier_calls": 0,
                "capsule_sha256": "",
            }
        integrity_valid = proof.verify_proof_capsule_integrity(
            req.proof_capsule,
            query=query,
            output_text=output,
            display_answer=display_answer,
            surface=req.surface,
            request_nonce=request_nonce,
        )
        expected: Optional[Dict[str, Any]] = None
        if integrity_valid:
            grounded = _fresh_verified_grounding(
                query,
                require_science_plan=req.surface == "scientific",
            )
            if grounded is not None:
                expected = _proof_capsule(query, grounded, req.surface, request_nonce)
        verified = bool(
            integrity_valid
            and expected is not None
            and dict(req.proof_capsule) == expected
        )
        replay_race = False
        ledger_capacity_exhausted = False
        if verified and request_nonce:
            try:
                replay_race = not self._remember_verification_nonce(request_nonce)
            except nonce_ledger.NonceLedgerCapacityError:
                ledger_capacity_exhausted = True
            if replay_race or ledger_capacity_exhausted:
                verified = False
        return {
            "status": "verified" if verified else "rejected",
            "verified": verified,
            "reason": (
                "fresh_recompute_exact_capsule_match"
                if verified
                else "nonce_ledger_capacity_exhausted"
                if ledger_capacity_exhausted
                else "request_nonce_replayed"
                if replay_race
                else "capsule_or_request_binding_rejected"
            ),
            "confidence": None,
            "assurance_kind": (
                "deterministic_assurance_not_probability" if verified else "none"
            ),
            "renderer_may_mark_numeric_claims_verified": verified,
            "fresh_verifier_calls": 1 if integrity_valid else 0,
            "capsule_sha256": (
                expected.get("capsule_sha256", "") if verified and expected else ""
            ),
        }

    def handle_entropy(self, req: EntropyRequest) -> Dict[str, Any]:
        effective_source = self.engine.entropy_engine.normalize_source(req.source)
        with self._lock:
            samples = self.engine.entropy_engine.sample(
                source=effective_source,
                count=req.count,
                seed=req.seed,
            )
            ca_analysis = self.engine.run_wolfram_analysis(
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
            "complexity_class": ca_analysis.complexity_class,
            "langton_lambda": ca_analysis.langton_lambda,
            "spatial_entropy": ca_analysis.spatial_entropy,
            "active_density_mean": ca_analysis.active_density_mean,
            "transition_table": ca_analysis.transition_table,
            "cellular_automata_grid": ca_analysis.grid,
            "status": "computed",
        }

    def handle_bell(self, req: BellRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_bell_experiment(
                theta_a=req.theta_a,
                theta_a_prime=req.theta_a_prime,
                theta_b=req.theta_b,
                theta_b_prime=req.theta_b_prime,
                shots=req.shots,
                seed=req.seed,
            )
        return res.to_dict()

    def handle_resonance(self, req: ResonanceRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_semantic_resonance(req.query or "")
        return res.to_dict()

    def handle_compare(self, req: CompareRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_compare(
                query_a=req.query_a,
                query_b=req.query_b,
                mode_a=req.mode_a,
                mode_b=req.mode_b,
                entropy_source_a=req.entropy_source_a,
                entropy_source_b=req.entropy_source_b,
            )
        return res.to_dict()

    def handle_quantum_state(self, req: QuantumStateRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_quantum_state_analysis(
                parameter_p=req.parameter_p,
                noise_rate=req.noise_rate,
                channel_type=req.channel_type,
            )
        return res.to_dict()

    def handle_gliders(self, req: GliderRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_glider_simulation(
                glider_type_left=req.glider_type_left,
                glider_type_right=req.glider_type_right,
                separation=req.separation,
                steps=req.steps,
                width=req.width,
            )
        return res.to_dict()

    def handle_trajectory(self, req: TrajectoryRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_cognitive_trajectory(req.steps or ["Initiation"])
        return res.to_dict()

    def handle_speculative_tree(self, req: SpeculativeTreeRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_speculative_tree_search(req.query or "")
        return res.to_dict()

    def handle_circuit_attribution(self, req: CircuitAttributionRequest) -> Dict[str, Any]:
        with self._lock:
            components = [
                c.to_dict()
                for c in self.engine.run_circuit_attribution(
                    req.prompt, req.target_token, req.contrast_token
                )
            ]
            patch_res = None
            if req.clean_prompt and req.corrupt_prompt and req.patch_layer is not None:
                patch_res = self.engine.run_activation_patching(
                    clean_prompt=req.clean_prompt,
                    corrupt_prompt=req.corrupt_prompt,
                    target_token=req.target_token,
                    layer_to_patch=req.patch_layer,
                    head_to_patch=req.patch_head,
                ).to_dict()

            causal_res = None
            if req.test_scratchpad or req.trace_steps:
                causal_res = self.engine.run_causal_register_check(
                    problem=req.prompt,
                    trace_steps=req.trace_steps,
                    next_operation=req.next_operation or "step_result",
                ).to_dict()

        return {
            "prompt": req.prompt,
            "target_token": req.target_token,
            "components": components,
            "activation_patch": patch_res,
            "causal_register": causal_res,
        }

    def handle_complexity_analysis(self, req: ComplexityAnalysisRequest) -> Dict[str, Any]:
        with self._lock:
            profile = self.engine.run_complexity_analysis(req.text).to_dict()
            ncd_res = None
            if req.compare_text:
                ncd_res = self.engine.run_ncd_comparison(req.text, req.compare_text).to_dict()

        return {
            "profile": profile,
            "ncd_comparison": ncd_res,
        }

    def handle_autoloop_step(self, req: AutoLoopStepRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_autoloop_step(
                current_query=req.query,
                reward_feedback=req.reward_feedback,
                forced_action=req.forced_action,
            )
        return res.to_dict()

    def handle_semantic_invariants(self, req: SemanticInvariantsRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_semantic_invariant_eval(
                problem=req.problem,
                ground_truth_answer=req.ground_truth_answer,
                task_type=req.task_type,
            )
        return res.to_dict()

    def handle_active_inference(self, req: ActiveInferenceRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.evaluate_active_inference(
                query=req.query,
                current_trace_steps=req.current_trace_steps,
                local_entropy=req.local_entropy,
                rsi_volatility=req.rsi_volatility,
                verification_confidence=req.verification_confidence,
                has_pending_subgoals=req.has_pending_subgoals,
            )
        return res.to_dict()

    def handle_proof_verify(self, req: ProofVerifyRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.locate_first_error(
                problem=req.problem,
                trace_steps=req.trace_steps,
            )
        return res.to_dict()

    def handle_bidirectional_speculation(self, req: BidirectionalSpeculationRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.verify_bidirectional_speculation(
                problem=req.problem,
                candidate_answer=req.candidate_answer,
            )
        return res.to_dict()

    def handle_epistemic_tree_search(self, req: EpistemicTreeSearchRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.run_epistemic_tree_search(
                query=req.query,
                max_depth=req.max_depth,
                beam_width=req.beam_width,
            )
        return res.to_dict()

    # ------------------------------------------------------------------ #
    #  v90 Frontier DoT handlers                                           #
    # ------------------------------------------------------------------ #

    def handle_diffusion_thought(self, req: DiffusionThoughtRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.denoise_thought_latent(
                problem=req.problem,
                num_timesteps=req.num_timesteps,
                guidance_scale=req.guidance_scale,
                latent_dim=req.latent_dim,
                seed=req.seed,
            )
        d = res.to_dict()
        d.update({"answer_authority": False, "status": "analysis_only"})
        return d

    def handle_reflexion_correction(self, req: ReflexionCorrectionRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.reflexive_self_correct(
                problem=req.problem,
                proposed_solution=req.proposed_solution,
                ground_truth=req.ground_truth,
                max_iterations=req.max_iterations,
            )
        d = res.to_dict()
        d.update({"answer_authority": False, "status": "analysis_only"})
        return d

    def handle_conformal_stopping(self, req: ConformalStoppingRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.evaluate_conformal_stopping(
                step_entropy=req.step_entropy,
                rsi_volatility=req.rsi_volatility,
                verifier_score=req.verifier_score,
                step_index=req.step_index,
                total_budget=req.total_budget,
                target_error_rate=req.target_error_rate,
            )
        return res.to_dict()

    def handle_causal_dag(self, req: CausalDAGRequest) -> Dict[str, Any]:
        with self._lock:
            res = self.engine.evaluate_causal_dag(
                scenario=req.scenario,
                treatment_node=req.treatment_node,
                outcome_node=req.outcome_node,
                do_value=req.do_value,
                observed_context=req.observed_context,
            )
        return res.to_dict()

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
            "service": "NexusMind Frontier Epistemic Diagnostics v88-v89-v90",
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
            "v88_frontier_hybrid": {
                "mechanistic_circuit_prober": True,
                "complexity_analyzer": True,
                "continuous_autoloop_engine": True,
                "semantic_invariant_engine": True,
            },
            "v89_frontier_epistemic": {
                "active_inference_controller": True,
                "proof_first_error_localizer": True,
                "bidirectional_speculation_engine": True,
                "epistemic_tree_search": True,
            },
            "v90_frontier_dot": {
                "diffusion_thought_engine": True,
                "reflexive_correction_engine": True,
                "conformal_stopping_controller": True,
                "causal_dag_engine": True,
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
            "service": "NexusMind Experimental Evidence API v82",
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
                    "modes": ["fast", "deep", "adaptive", "agent"],
                },
            ],
            "catalog_claim_scope": "runtime capabilities observed in this source tree",
        }

    def _risk_runtime_binding_sha256(self) -> str:
        """Bind the shadow receipt to the exact local verifier/runtime sources."""

        files = {
            name: (Path(__file__).resolve().parent / name).read_bytes()
            for name in (
                "nexus_api.py",
                "nexus_engine.py",
                "nexus_proof.py",
                "nexus_independent_checker.py",
                "nexus_nonce_ledger.py",
                "nexus_risk_control.py",
                "grounding_runtime.py",
            )
        }
        rows = [
            {"name": name, "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in sorted(files.items())
        ]
        return hashlib.sha256(risk_control.canonical_json_bytes(rows)).hexdigest()

    def handle_risk_protocol(self) -> Dict[str, Any]:
        """Describe the precommitted risk lab without making a live decision."""

        benchmark = risk_control.build_frozen_arithmetic_benchmark()
        plan = risk_control.build_risk_control_plan(
            regime="bonferroni_grid",
            max_error_rate=0.10,
            alpha=0.05,
            min_accepted=48,
            runtime_binding_sha256=self._risk_runtime_binding_sha256(),
        )
        return {
            "status": "shadow_protocol_ready",
            "scope": "offline_frozen_regression_only",
            "selection_authorized": False,
            "policy_applied": False,
            "answer_authority": False,
            "benchmark": benchmark["manifest"],
            "plan": {
                "schema_version": plan["schema_version"],
                "protocol": plan["protocol"],
                "target": plan["target"],
                "candidate_policies": plan["candidate_policies"],
                "bindings": plan["bindings"],
                "plan_sha256": plan["plan_sha256"],
            },
            "regimes": sorted(risk_control.REGIMES),
            "authority": plan["authority"],
        }

    def handle_risk_evaluate(
        self,
        plan: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Evaluate shadow records and return a non-authoritative receipt."""

        receipt = risk_control.calibrate_selective_risk(plan, records)
        risk_control.validate_risk_control_receipt(receipt, plan=plan, records=records)
        return {
            "status": "shadow_evaluation_complete",
            "scope": "offline_calibration_evidence_only",
            "selection_authorized": False,
            "policy_applied": False,
            "answer_authority": False,
            "receipt": receipt,
        }

    def handle_risk_audit(self) -> Dict[str, Any]:
        """Run the frozen arithmetic/adversarial cohort through the exact gate."""

        benchmark = risk_control.build_frozen_arithmetic_benchmark()
        runtime_binding = self._risk_runtime_binding_sha256()
        plan = risk_control.build_risk_control_plan(
            regime="bonferroni_grid",
            max_error_rate=0.10,
            alpha=0.05,
            min_accepted=48,
            runtime_binding_sha256=runtime_binding,
        )
        records: List[Dict[str, Any]] = []
        for case in benchmark["cases"]:
            audit_nonce = "risk-" + hashlib.sha256(
                str(case["case_id"]).encode("utf-8")
            ).hexdigest()
            result = self.handle_solve(
                SolveRequest(query=str(case["prompt"]), request_nonce=audit_nonce)
            )
            prediction = (
                result.get("display_answer")
                if result.get("solved") is True and result.get("answer_authority") is True
                else "abstain"
            )
            verified = risk_control.evaluate_frozen_answer(case, prediction)
            score = 1.0 if verified["observed_label"] == "answer" else 0.0
            for policy in risk_control.FIXED_CANDIDATE_POLICIES:
                records.append(
                    risk_control.construct_benchmark_shadow_record(
                        case,
                        split="cal",
                        policy_id=policy.policy_id,
                        score=score,
                        prediction=prediction,
                        cost=float(policy.nominal_cost_units),
                    )
                )
        receipt = risk_control.calibrate_selective_risk(plan, records)
        risk_control.validate_risk_control_receipt(receipt, plan=plan, records=records)
        return {
            "status": "shadow_audit_complete",
            "scope": "frozen_synthetic_regression_only",
            "selection_authorized": False,
            "policy_applied": False,
            "answer_authority": False,
            "runtime_binding_sha256": runtime_binding,
            "benchmark": benchmark["manifest"],
            "plan": {
                "schema_version": plan["schema_version"],
                "target": plan["target"],
                "plan_sha256": plan["plan_sha256"],
            },
            "receipt": receipt,
        }


def create_app(service: Optional[NexusApiService] = None):
    """Create FastAPI application if installed, or fallback to lightweight ASGI/WSGI app."""
    svc = service or NexusApiService()

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse, StreamingResponse
        from pydantic import BaseModel, Field

        app = FastAPI(
            title="NexusMind Frontier Epistemic Evidence API v89",
            description=(
                "Verifier-first closed-world answers plus explicitly bounded heuristic "
                "analysis, neural architecture telemetry, mechanistic interpretability, "
                "epistemic active inference, and neuro-symbolic proof verification."
            ),
            version="89.0.0",
        )

        class PyThinkMessage(BaseModel):
            role: str = "user"
            content: str = ""

        class PyThinkRequest(BaseModel):
            messages: List[PyThinkMessage] = Field(default_factory=list)
            prompt: Optional[str] = None
            mode: str = "auto"
            max_output_tokens: int = 256
            thinking_budget: Optional[int] = None
            tools: Optional[List[Dict[str, Any]]] = None
            persona: Optional[str] = None
            session_id: Optional[str] = None
            entropy_source: Optional[str] = None
            request_nonce: str = ""
            stream: bool = False

        class PyActiveInferenceRequest(BaseModel):
            query: str
            current_trace_steps: List[str] = Field(default_factory=list)
            local_entropy: float = 0.85
            rsi_volatility: float = 50.0
            verification_confidence: float = 0.80
            has_pending_subgoals: bool = False

        class PyProofVerifyRequest(BaseModel):
            problem: str
            trace_steps: List[str] = Field(default_factory=list)

        class PyBidirectionalSpeculationRequest(BaseModel):
            problem: str
            candidate_answer: Optional[str] = None

        class PyEpistemicTreeSearchRequest(BaseModel):
            query: str
            max_depth: int = 4
            beam_width: int = 3

        class PyDiffusionThoughtRequest(BaseModel):
            problem: str
            num_timesteps: int = 20
            guidance_scale: float = 3.0
            latent_dim: int = 16
            seed: int = 42

        class PyReflexionCorrectionRequest(BaseModel):
            problem: str
            proposed_solution: str
            ground_truth: Optional[str] = None
            max_iterations: int = 3

        class PyConformalStoppingRequest(BaseModel):
            step_entropy: float = 0.4
            rsi_volatility: float = 40.0
            verifier_score: float = 0.85
            step_index: int = 3
            total_budget: int = 10
            target_error_rate: float = 0.05

        class PyCausalDAGRequest(BaseModel):
            scenario: str = "physics_newton"
            treatment_node: str = "Force"
            outcome_node: str = "Acceleration"
            do_value: float = 10.0
            observed_context: Optional[Dict[str, Any]] = None

        class PyEntropyRequest(BaseModel):
            source: str = "crypto"
            count: int = 16
            seed: Optional[int] = None
            rule: int = 30
            ca_steps: int = 16
            ca_width: int = 31

        class PySolveRequest(BaseModel):
            query: str
            request_nonce: str = ""

        class PyInnovateRequest(BaseModel):
            topic: str
            count: int = 6

        class PyChatRequest(BaseModel):
            session_id: str
            message: str
            persona: Optional[str] = None
            request_nonce: str = ""

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
            request_nonce: str = ""

        class PyVerifyRequest(BaseModel):
            query: str
            output: str
            display_answer: str
            surface: str
            proof_capsule: Dict[str, Any] = Field(default_factory=dict)
            request_nonce: str = ""

        class PyFeedbackRequest(BaseModel):
            difficulty: float
            epistemic_risk: float
            budget_used: int
            reward: float

        class PyRiskEvaluateRequest(BaseModel):
            plan: Dict[str, Any]
            records: List[Dict[str, Any]]

        class PyBellRequest(BaseModel):
            theta_a: float = 0.0
            theta_a_prime: float = 45.0
            theta_b: float = 22.5
            theta_b_prime: float = 67.5
            shots: int = 1000
            seed: Optional[int] = 42

        class PyResonanceRequest(BaseModel):
            query: str = ""

        class PyCompareRequest(BaseModel):
            query_a: str
            query_b: Optional[str] = None
            mode_a: str = "auto"
            mode_b: str = "deep"
            entropy_source_a: str = "crypto"
            entropy_source_b: str = "seeded"

        class PyQuantumStateRequest(BaseModel):
            parameter_p: float = 1.0
            noise_rate: float = 0.0
            channel_type: str = "depolarizing"

        class PyGliderRequest(BaseModel):
            glider_type_left: str = "glider_A"
            glider_type_right: str = "glider_C"
            separation: int = 10
            steps: int = 24
            width: int = 40

        class PyTrajectoryRequest(BaseModel):
            steps: List[str] = Field(default_factory=list)

        class PySpeculativeTreeRequest(BaseModel):
            query: str
            branching_factor: int = 3
            max_depth: int = 4

        class PyCircuitAttributionRequest(BaseModel):
            prompt: str
            target_token: str
            contrast_token: Optional[str] = None
            clean_prompt: Optional[str] = None
            corrupt_prompt: Optional[str] = None
            patch_layer: Optional[int] = None
            patch_head: Optional[int] = None
            test_scratchpad: bool = False
            trace_steps: List[str] = Field(default_factory=list)
            next_operation: Optional[str] = None

        class PyComplexityAnalysisRequest(BaseModel):
            text: str
            compare_text: Optional[str] = None
            window_size: int = 8

        class PyAutoLoopStepRequest(BaseModel):
            query: str
            reward_feedback: Optional[float] = None
            forced_action: Optional[str] = None

        class PySemanticInvariantsRequest(BaseModel):
            problem: str
            ground_truth_answer: Optional[str] = None
            task_type: str = "arithmetic"

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
                request_nonce=req.request_nonce,
                stream=req.stream,
            )
            if req.stream:
                def sse_events():
                    for event in svc.handle_think_stream(t_req):
                        event_name = str(event.get("event") or "message")
                        payload = json.dumps(
                            event,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        )
                        yield f"event: {event_name}\ndata: {payload}\n\n"

                return StreamingResponse(
                    sse_events(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                        "X-Nexus-Stream-Contract": "nexus-sse-proof-carrying-v1",
                    },
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

        @app.post("/v1/quantum/bell")
        async def bell_endpoint(req: PyBellRequest):
            b_req = BellRequest(
                theta_a=req.theta_a,
                theta_a_prime=req.theta_a_prime,
                theta_b=req.theta_b,
                theta_b_prime=req.theta_b_prime,
                shots=req.shots,
                seed=req.seed,
            )
            return svc.handle_bell(b_req)

        @app.post("/v1/resonance")
        async def resonance_endpoint(req: PyResonanceRequest):
            r_req = ResonanceRequest(query=req.query)
            return svc.handle_resonance(r_req)

        @app.post("/v1/compare")
        async def compare_endpoint(req: PyCompareRequest):
            cmp_req = CompareRequest(
                query_a=req.query_a,
                query_b=req.query_b,
                mode_a=req.mode_a,
                mode_b=req.mode_b,
                entropy_source_a=req.entropy_source_a,
                entropy_source_b=req.entropy_source_b,
            )
            return svc.handle_compare(cmp_req)

        @app.post("/v1/quantum/state")
        async def quantum_state_endpoint(req: PyQuantumStateRequest):
            q_req = QuantumStateRequest(
                parameter_p=req.parameter_p,
                noise_rate=req.noise_rate,
                channel_type=req.channel_type,
            )
            return svc.handle_quantum_state(q_req)

        @app.post("/v1/wolfram/gliders")
        async def gliders_endpoint(req: PyGliderRequest):
            g_req = GliderRequest(
                glider_type_left=req.glider_type_left,
                glider_type_right=req.glider_type_right,
                separation=req.separation,
                steps=req.steps,
                width=req.width,
            )
            return svc.handle_gliders(g_req)

        @app.post("/v1/resonance/trajectory")
        async def trajectory_endpoint(req: PyTrajectoryRequest):
            t_req = TrajectoryRequest(steps=req.steps)
            return svc.handle_trajectory(t_req)

        @app.post("/v1/speculative-tree")
        async def speculative_tree_endpoint(req: PySpeculativeTreeRequest):
            st_req = SpeculativeTreeRequest(
                query=req.query,
                branching_factor=req.branching_factor,
                max_depth=req.max_depth,
            )
            return svc.handle_speculative_tree(st_req)

        @app.post("/v1/circuits/attribute")
        async def circuit_attribution_endpoint(req: PyCircuitAttributionRequest):
            c_req = CircuitAttributionRequest(
                prompt=req.prompt,
                target_token=req.target_token,
                contrast_token=req.contrast_token,
                clean_prompt=req.clean_prompt,
                corrupt_prompt=req.corrupt_prompt,
                patch_layer=req.patch_layer,
                patch_head=req.patch_head,
                test_scratchpad=req.test_scratchpad,
                trace_steps=req.trace_steps,
                next_operation=req.next_operation,
            )
            return svc.handle_circuit_attribution(c_req)

        @app.post("/v1/complexity/analyze")
        async def complexity_analysis_endpoint(req: PyComplexityAnalysisRequest):
            c_req = ComplexityAnalysisRequest(
                text=req.text,
                compare_text=req.compare_text,
                window_size=req.window_size,
            )
            return svc.handle_complexity_analysis(c_req)

        @app.post("/v1/autoloop/step")
        async def autoloop_step_endpoint(req: PyAutoLoopStepRequest):
            a_req = AutoLoopStepRequest(
                query=req.query,
                reward_feedback=req.reward_feedback,
                forced_action=req.forced_action,
            )
            return svc.handle_autoloop_step(a_req)

        @app.post("/v1/semantic/invariants")
        async def semantic_invariants_endpoint(req: PySemanticInvariantsRequest):
            s_req = SemanticInvariantsRequest(
                problem=req.problem,
                ground_truth_answer=req.ground_truth_answer,
                task_type=req.task_type,
            )
            return svc.handle_semantic_invariants(s_req)

        @app.post("/v1/active_inference/decide")
        async def active_inference_endpoint(req: PyActiveInferenceRequest):
            a_req = ActiveInferenceRequest(
                query=req.query,
                current_trace_steps=req.current_trace_steps,
                local_entropy=req.local_entropy,
                rsi_volatility=req.rsi_volatility,
                verification_confidence=req.verification_confidence,
                has_pending_subgoals=req.has_pending_subgoals,
            )
            return svc.handle_active_inference(a_req)

        @app.post("/v1/proof/verify_steps")
        async def proof_verify_endpoint(req: PyProofVerifyRequest):
            p_req = ProofVerifyRequest(
                problem=req.problem,
                trace_steps=req.trace_steps,
            )
            return svc.handle_proof_verify(p_req)

        @app.post("/v1/speculative/bidirectional")
        async def bidirectional_speculation_endpoint(req: PyBidirectionalSpeculationRequest):
            b_req = BidirectionalSpeculationRequest(
                problem=req.problem,
                candidate_answer=req.candidate_answer,
            )
            return svc.handle_bidirectional_speculation(b_req)

        @app.post("/v1/mcts/epistemic_search")
        async def epistemic_search_endpoint(req: PyEpistemicTreeSearchRequest):
            e_req = EpistemicTreeSearchRequest(
                query=req.query,
                max_depth=req.max_depth,
                beam_width=req.beam_width,
            )
            return svc.handle_epistemic_tree_search(e_req)

        @app.post("/v1/dot/denoise")
        async def diffusion_thought_endpoint(req: PyDiffusionThoughtRequest):
            d_req = DiffusionThoughtRequest(
                problem=req.problem,
                num_timesteps=req.num_timesteps,
                guidance_scale=req.guidance_scale,
                latent_dim=req.latent_dim,
                seed=req.seed,
            )
            return svc.handle_diffusion_thought(d_req)

        @app.post("/v1/reflexion/correct")
        async def reflexion_correction_endpoint(req: PyReflexionCorrectionRequest):
            r_req = ReflexionCorrectionRequest(
                problem=req.problem,
                proposed_solution=req.proposed_solution,
                ground_truth=req.ground_truth,
                max_iterations=req.max_iterations,
            )
            return svc.handle_reflexion_correction(r_req)

        @app.post("/v1/conformal/evaluate")
        async def conformal_stopping_endpoint(req: PyConformalStoppingRequest):
            c_req = ConformalStoppingRequest(
                step_entropy=req.step_entropy,
                rsi_volatility=req.rsi_volatility,
                verifier_score=req.verifier_score,
                step_index=req.step_index,
                total_budget=req.total_budget,
                target_error_rate=req.target_error_rate,
            )
            return svc.handle_conformal_stopping(c_req)

        @app.post("/v1/causal/dag_query")
        async def causal_dag_endpoint(req: PyCausalDAGRequest):
            cq_req = CausalDAGRequest(
                scenario=req.scenario,
                treatment_node=req.treatment_node,
                outcome_node=req.outcome_node,
                do_value=req.do_value,
                observed_context=req.observed_context,
            )
            return svc.handle_causal_dag(cq_req)

        @app.get("/v1/signals")
        async def signals_endpoint():
            return svc.handle_signals()

        @app.post("/v1/solve")
        async def solve_endpoint(req: PySolveRequest):
            s_req = SolveRequest(query=req.query, request_nonce=req.request_nonce)
            return svc.handle_solve(s_req)

        @app.post("/v1/innovate")
        async def innovate_endpoint(req: PyInnovateRequest):
            i_req = InnovateRequest(topic=req.topic, count=req.count)
            return svc.handle_innovate(i_req)

        @app.post("/v1/chat")
        async def chat_endpoint(req: PyChatRequest):
            c_req = ChatTurnRequest(
                session_id=req.session_id,
                message=req.message,
                persona=req.persona,
                request_nonce=req.request_nonce,
            )
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
            s_req = ScientificRequest(query=req.query, request_nonce=req.request_nonce)
            return svc.handle_scientific(s_req)

        @app.post("/v1/verify")
        async def verify_endpoint(req: PyVerifyRequest):
            v_req = VerifyRequest(
                query=req.query,
                output=req.output,
                display_answer=req.display_answer,
                surface=req.surface,
                proof_capsule=req.proof_capsule,
                request_nonce=req.request_nonce,
            )
            return svc.handle_verify(v_req)

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

        @app.get("/v1/risk-control")
        async def risk_protocol_endpoint():
            return svc.handle_risk_protocol()

        @app.post("/v1/risk-control/audit")
        async def risk_audit_endpoint():
            return svc.handle_risk_audit()

        @app.post("/v1/risk-control/evaluate")
        async def risk_evaluate_endpoint(req: PyRiskEvaluateRequest):
            try:
                return svc.handle_risk_evaluate(req.plan, req.records)
            except risk_control.RiskControlValidationError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        @app.get("/health")
        async def health_endpoint():
            store = svc._ensure_verification_nonce_store()
            return {
                "status": "ok",
                "service": "NexusMind Frontier Epistemic Evidence API v88-v89-v90",
                "answer_policy": epistemics.SELECTIVE_ANSWER_POLICY_VERSION,
                "verification_nonce_backend": (
                    "sqlite_durable"
                    if isinstance(store, nonce_ledger.SQLiteNonceLedger)
                    else "in_memory_process_local"
                ),
                "verification_nonce_ttl_seconds": _VERIFY_NONCE_TTL_SECONDS,
                "verification_nonce_max_entries": _VERIFY_NONCE_CACHE_SIZE,
                "verification_nonce_required": True,
                "independent_witness_required": True,
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
    parser.add_argument(
        "--verification-nonce-db",
        default=None,
        help=(
            "Optional SQLite path for cross-worker/restart nonce replay protection; "
            "only SHA-256 nonce digests are stored"
        ),
    )
    args = parser.parse_args()

    import uvicorn
    app = create_app(NexusApiService(verification_nonce_db=args.verification_nonce_db))
    print(f"[*] Starting NexusMind Experimental Evidence API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
