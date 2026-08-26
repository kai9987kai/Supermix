"""NexusMind experimental evidence-first orchestrator.

The orchestrator keeps several experimental components behind one explicit
answer-admission contract:
1. **Strict closed-world verifier** (`grounding_runtime.py`, `science_plan.py`):
   - freshly recomputed exact arithmetic and allowlisted scientific scenarios
   - full-request gates and bounded deterministic answer authority
   - `nexus_solver.py` remains a broader audit/development pattern library
2. **Creative Ideation & Lateral Innovation** (`nexus_ideation.py`):
   - analysis-only SCAMPER/TRIZ concepts with authored priority scores
3. **Adaptive Conversational Intelligence & Personas** (`nexus_chat.py`):
   - analysis-only personas, multi-turn state and entity tracking
4. **AI-Dem-Lab & Swarm Deliberation** (`nexus_swarm.py`, `nexus_got.py`, `mimomix_observatory.py`):
   - deterministic template swarm and graph scaffolds without answer authority
   - unverified runtime outputs cannot update the Q policy
5. **Xiaomi MiMo Neural Core** (`mimomix_core.py`, `mimomix_decoding.py`):
   - newly initialized telemetry-only architecture probe; no loaded text decoder
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

import mimomix_controller as controller
import mimomix_core as mc
import mimomix_decoding as decoding
import mimomix_observatory as observatory
import mimomix_reasoner as reasoner
import grounding_runtime as grounding
import nexus_chat as chat
import nexus_epistemics as epistemics
import nexus_got as got
import nexus_ideation as ideation
import nexus_solver as solver
import nexus_swarm as swarm
import science_plan as science


__all__ = [
    "ThinkingMode",
    "EntropySource",
    "NexusConfig",
    "NexusThoughtStep",
    "NexusResult",
    "MultiSourceEntropyEngine",
    "RSIMomentumOscillator",
    "QLearningPolicyEngine",
    "NexusEngine",
    "build_default_engine",
]

ThinkingMode = str  # "fast" | "deep" | "agent" | "swarm" | "got" | "scientific" | "solve" | "innovate" | "chat" | "auto"
EntropySource = str  # "crypto" | "seeded" | "os_csprng_transform" | "chaotic"


@dataclass
class NexusConfig:
    """Master configuration for the NexusMind Unified Engine."""

    vocab_size: int = 512
    hidden_size: int = 128
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    n_experts: int = 4
    top_k_experts: int = 2
    sliding_window: int = 128
    hybrid_ratio: int = 5
    max_thinking_budget: int = 6
    swarm_rounds: int = 3
    got_max_depth: int = 4
    got_beam_width: int = 3
    q_learning_enabled: bool = True
    default_entropy_source: EntropySource = "crypto"
    rsi_window: int = 14
    q_alpha: float = 0.20
    q_gamma: float = 0.90
    q_epsilon: float = 0.15


# ---------------------------------------------------------------------------
# AI-Dem-Lab Multi-Source Entropy & State-Space Probing
# ---------------------------------------------------------------------------


class MultiSourceEntropyEngine:
    """Diagnostic sampler with explicit software provenance for every source."""

    def __init__(self, default_seed: int = 424242):
        self.default_seed = default_seed
        self._chaotic_state = 0.61803398875

    @staticmethod
    def normalize_source(source: EntropySource) -> str:
        """Resolve supported names and map the old ``qrng`` label truthfully."""

        value = str(source or "crypto").strip().lower()
        if value in {"qrng", "qrng_simulation", "os_csprng_transform"}:
            return "os_csprng_transform"
        if value in {"crypto", "seeded", "chaotic"}:
            return value
        return "crypto"

    @classmethod
    def source_provenance(cls, source: EntropySource) -> Dict[str, Any]:
        effective = cls.normalize_source(source)
        rows = {
            "crypto": {
                "mechanism": "python_secrets_os_csprng",
                "deterministic": False,
                "quantum_hardware_used": False,
                "security_claim": "delegated_to_python_secrets_and_the_host_os",
            },
            "seeded": {
                "mechanism": "python_random_seeded_prng",
                "deterministic": True,
                "quantum_hardware_used": False,
                "security_claim": "none",
            },
            "os_csprng_transform": {
                "mechanism": "os_urandom_then_sine_numeric_transform",
                "deterministic": False,
                "quantum_hardware_used": False,
                "security_claim": "none_for_the_transformed_stream",
            },
            "chaotic": {
                "mechanism": "software_logistic_map",
                "deterministic": True,
                "quantum_hardware_used": False,
                "security_claim": "none",
            },
        }
        return {"effective_source": effective, **rows[effective]}

    def sample(
        self,
        source: EntropySource = "crypto",
        count: int = 16,
        seed: Optional[int] = None,
    ) -> List[float]:
        """Generate a list of normalized floats in [0.0, 1.0) from the requested entropy backend."""
        count = max(1, min(1024, count))
        src = self.normalize_source(source)

        if src == "seeded":
            import random
            rng = random.Random(seed if seed is not None else self.default_seed)
            return [rng.random() for _ in range(count)]

        elif src == "os_csprng_transform":
            # Software-only transform of OS random bytes. This is not a QRNG
            # and makes no post-transform cryptographic-security claim.
            import os
            raw_bytes = os.urandom(count * 4)
            floats: List[float] = []
            for i in range(count):
                chunk = raw_bytes[i * 4 : (i + 1) * 4]
                val = int.from_bytes(chunk, "big") / (2**32)
                # Apply a bounded numeric sine transform for visualization.
                phase = math.sin(val * 2 * math.pi) * 0.5 + 0.5
                floats.append(round((val * 0.7 + phase * 0.3) % 1.0, 6))
            return floats

        elif src == "chaotic":
            # Non-linear chaotic logistic map: x_{n+1} = r * x_n * (1 - x_n) with r = 3.9999
            floats = []
            x = (seed % 1000) / 1000.0 if seed is not None and seed > 0 else self._chaotic_state
            if x <= 0.0 or x >= 1.0:
                x = 0.54321
            r = 3.9999
            for _ in range(count):
                x = r * x * (1.0 - x)
                floats.append(round(x, 6))
            self._chaotic_state = x
            return floats

        else:  # "crypto" / default
            import secrets
            return [secrets.randbelow(1_000_000) / 1_000_000.0 for _ in range(count)]

    @staticmethod
    def cellular_automata_step(
        rule: int = 30,
        initial_state: Optional[List[int]] = None,
        steps: int = 16,
        width: int = 31,
    ) -> List[List[int]]:
        """Compute Wolfram 1D Elementary Cellular Automata state evolution (Rule 30, 90, 110)."""
        width = max(7, min(127, width))
        steps = max(1, min(64, steps))
        rule = max(0, min(255, rule))

        # Initial state: single central active cell if not provided
        if not initial_state or len(initial_state) != width:
            current = [0] * width
            current[width // 2] = 1
        else:
            current = [int(bool(x)) for x in initial_state[:width]]

        grid: List[List[int]] = [list(current)]

        for _ in range(steps - 1):
            next_state = [0] * width
            for i in range(width):
                left = current[(i - 1) % width]
                center = current[i]
                right = current[(i + 1) % width]
                neighborhood = (left << 2) | (center << 1) | right
                next_state[i] = (rule >> neighborhood) & 1
            grid.append(next_state)
            current = next_state

        return grid


# ---------------------------------------------------------------------------
# AI-Dem-Lab numeric-sequence RSI diagnostic
# ---------------------------------------------------------------------------


class RSIMomentumOscillator:
    """Relative Strength Index over caller-supplied numeric probe sequences.

    The statistic has no inherent connection to reasoning quality, novelty, or
    stability. Callers must attach provenance for the sequence they supply.
    """

    def __init__(self, window: int = 14):
        self.window = max(3, min(60, window))
        self.history: List[float] = []

    def update(self, value: float) -> Dict[str, Any]:
        """Record one numeric observation and compute a descriptive RSI value."""
        self.history.append(float(value))
        if len(self.history) > 100:
            self.history = self.history[-100:]

        if len(self.history) < 2:
            return {
                "rsi": 50.0,
                "volatility": 0.0,
                "regime": "flat_or_unresolved_probe",
                "extreme_momentum_flag": False,
                "history_length": len(self.history),
                "metric_semantics": "numeric_sequence_momentum_not_reasoning_quality",
            }

        diffs = [self.history[i] - self.history[i - 1] for i in range(1, len(self.history))]
        gains = [max(0.0, d) for d in diffs]
        losses = [max(0.0, -d) for d in diffs]

        # Use available window length
        effective_w = min(self.window, len(diffs))
        recent_gains = gains[-effective_w:]
        recent_losses = losses[-effective_w:]

        avg_gain = sum(recent_gains) / max(1, len(recent_gains))
        avg_loss = sum(recent_losses) / max(1, len(recent_losses))

        if avg_loss == 0.0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # Compute standard deviation as volatility
        mean = sum(self.history[-effective_w:]) / effective_w
        variance = sum((x - mean) ** 2 for x in self.history[-effective_w:]) / effective_w
        volatility = math.sqrt(variance)

        if rsi >= 70.0:
            regime = "high_positive_probe_momentum"
        elif rsi <= 30.0:
            regime = "high_negative_probe_momentum"
        else:
            regime = "mixed_probe_momentum"

        return {
            "rsi": round(rsi, 2),
            "volatility": round(volatility, 4),
            "regime": regime,
            "extreme_momentum_flag": rsi >= 75.0 or rsi <= 25.0,
            "history_length": len(self.history),
            "metric_semantics": "numeric_sequence_momentum_not_reasoning_quality",
        }


# ---------------------------------------------------------------------------
# AI-Dem-Lab Adaptive Q-Learning Policy Router
# ---------------------------------------------------------------------------


class QLearningPolicyEngine:
    """Disconnected tabular Q-learning experiment initialized from authored priors."""

    ACTIONS = ["fast", "deep", "agent", "swarm", "got", "solve", "innovate", "chat"]

    def __init__(
        self,
        alpha: float = 0.20,
        gamma: float = 0.90,
        epsilon: float = 0.15,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # State: (difficulty_bin: 0..3, risk_bin: 0..3) -> Action -> Q-value
        self.q_table: Dict[Tuple[int, int], Dict[str, float]] = {}
        self.update_count = 0
        self._init_q_table()

    def _init_q_table(self):
        for d in range(4):
            for r in range(4):
                self.q_table[(d, r)] = {a: 0.5 for a in self.ACTIONS}
                # Author sensible priors
                if d >= 2 and r >= 2:
                    self.q_table[(d, r)]["deep"] = 0.8
                    self.q_table[(d, r)]["swarm"] = 0.75
                elif d == 0:
                    self.q_table[(d, r)]["fast"] = 0.9

    def discretize_state(self, difficulty: float, risk: float) -> Tuple[int, int]:
        d_bin = max(0, min(3, int(difficulty * 4)))
        r_bin = max(0, min(3, int(risk * 4)))
        return (d_bin, r_bin)

    def select_action(
        self,
        difficulty: float,
        risk: float,
        allow_exploration: bool = True,
    ) -> str:
        state = self.discretize_state(difficulty, risk)
        actions_dict = self.q_table.get(state, {a: 0.5 for a in self.ACTIONS})

        import secrets
        if allow_exploration and (secrets.randbelow(100) / 100.0) < self.epsilon:
            keys = list(actions_dict.keys())
            return keys[secrets.randbelow(len(keys))]

        best_action = max(actions_dict.items(), key=lambda kv: kv[1])[0]
        return best_action

    def update(
        self,
        difficulty: float,
        risk: float,
        action: str,
        reward: float,
        next_difficulty: float = 0.0,
        next_risk: float = 0.0,
    ) -> float:
        state = self.discretize_state(difficulty, risk)
        next_state = self.discretize_state(next_difficulty, next_risk)

        if state not in self.q_table:
            self.q_table[state] = {a: 0.5 for a in self.ACTIONS}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.5 for a in self.ACTIONS}

        current_q = self.q_table[state].get(action, 0.5)
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = round(new_q, 4)
        self.update_count += 1
        return new_q

    def get_policy_summary(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "updates_applied": self.update_count,
            "connected_to_nexus_process": False,
            "values_are_calibrated_outcomes": False,
            "initialization": "authored_priors",
            "total_states": len(self.q_table),
            "state_matrix": {
                f"d{s[0]}_r{s[1]}": q_vals for s, q_vals in self.q_table.items()
            },
        }


@dataclass
class NexusThoughtStep:
    """A granular thinking step emitted during reasoning."""

    step_index: int
    stage: str  # "route" | "ponder" | "speculative_draft" | "swarm_debate" | "got_branch" | "science_proof" | "math_derivation" | "ideation" | "persona_chat"
    content: str
    confidence: Optional[float] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NexusResult:
    """Comprehensive output produced by the NexusMind Unified Engine."""

    query: str
    mode_selected: str
    final_output: str
    thought_steps: List[NexusThoughtStep] = field(default_factory=list)
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    speculative_acceptance_rate: Optional[float] = None
    tool_calls_used: int = 0
    audit_receipts: Dict[str, Any] = field(default_factory=dict)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    epistemics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "mode_selected": self.mode_selected,
            "final_output": self.final_output,
            "thought_steps": [s.to_dict() for s in self.thought_steps],
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "speculative_acceptance_rate": self.speculative_acceptance_rate,
            "tool_calls_used": self.tool_calls_used,
            "audit_receipts": self.audit_receipts,
            "telemetry": self.telemetry,
            "epistemics": self.epistemics,
        }


def _sanitize_untrained_probe_telemetry(value: Dict[str, Any]) -> Dict[str, Any]:
    """Keep architecture diagnostics while withholding untrained decision heads.

    The core exposes sigmoid values named ``quality_probability`` and
    ``continue_probability`` for training/controller work. On a newly
    initialized model they are arbitrary head activations, not measured
    quality, calibrated probability, or evidence, so Nexus does not publish
    them on its experimental answer surfaces.
    """

    clean = dict(value or {})
    thinking = dict(clean.get("thinking") or {})
    removed = False
    for key in ("quality_probability", "continue_probability"):
        if key in thinking:
            thinking.pop(key)
            removed = True
    thinking["untrained_decision_head_outputs_withheld"] = removed
    thinking["decision_head_values_are_quality_evidence"] = False
    clean["thinking"] = thinking
    clean["telemetry_scope"] = "untrained_architecture_diagnostics_only"
    return clean


class NexusEngine:
    """Unified hybrid thinking engine executing across all lineages."""

    def __init__(self, config: Optional[NexusConfig] = None):
        self.config = config or NexusConfig()

        # 1. MiMo neural core
        mimo_cfg = mc.MiMoMixConfig(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            n_kv_heads=self.config.n_kv_heads,
            n_routed_experts=self.config.n_experts,
            moe_top_k=self.config.top_k_experts,
            sliding_window=self.config.sliding_window,
            hybrid_ratio=self.config.hybrid_ratio,
        )
        self.model = mc.MiMoMixModel(mimo_cfg)
        self.model.eval()

        # 2. 5-Agent Cognitive Swarm
        self.swarm_engine = swarm.SwarmEngine(max_rounds=self.config.swarm_rounds)

        # 3. Graph-of-Thoughts Reasoner
        self.got_engine = got.GraphOfThoughts(
            max_depth=self.config.got_max_depth,
            beam_width=self.config.got_beam_width,
        )

        # 4. Omni-Science & Math Solver
        self.solver_engine = solver.NexusSolver()

        # 5. Creative Ideation Engine
        self.ideation_engine = ideation.NexusIdeationEngine()

        # 6. Adaptive Persona Chat Engine
        self.chat_engine = chat.NexusChatEngine()

        # 7. Dem-Lab Observatory & Q-learning feedback policy
        self.observatory = observatory.Observatory()
        self.q_learner = observatory.BudgetPolicyLearner()

        # 8. AI-Dem-Lab Multi-Source Entropy, RSI, and Q-Policy Engines
        self.entropy_engine = MultiSourceEntropyEngine()
        self.rsi_oscillator = RSIMomentumOscillator(window=self.config.rsi_window)
        self.q_policy = QLearningPolicyEngine(
            alpha=self.config.q_alpha,
            gamma=self.config.q_gamma,
            epsilon=self.config.q_epsilon,
        )

    def process(
        self,
        query: str,
        mode: ThinkingMode = "auto",
        max_output_tokens: int = 256,
        thinking_budget: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        persona: Optional[str] = None,
        session_id: Optional[str] = None,
        entropy_source: Optional[EntropySource] = None,
    ) -> NexusResult:
        """Process a request through the unified hybrid thinking pipeline."""
        start_time = time.perf_counter()
        query_clean = query.strip()
        tools = tools or []
        entropy_src = self.entropy_engine.normalize_source(
            entropy_source or self.config.default_entropy_source
        )
        steps: List[NexusThoughtStep] = []
        receipts: Dict[str, Any] = {}

        # 1. Verifier-first answer admission.
        #
        # NexusSolver's extended regex handlers are useful corpus/evaluation
        # utilities, but their self-hash does not prove that the full request
        # was consumed.  Only the shipped grounding path may admit an answer:
        # it freshly recomputes with the trusted reasoning/science registry and
        # applies negation, ambiguity, mixed-request, high-stakes, consensus,
        # dimensional and substitution gates.
        if mode in ("auto", "scientific", "solve"):
            try:
                grounded = grounding.finalize_grounded_response("", query_clean)
            except Exception:
                grounded = {}
            if not isinstance(grounded, Mapping):
                grounded = {}
            grounded_valid = epistemics.verify_grounded_answer_result(
                grounded,
                receipt_schema_version=grounding.VERIFIED_ANSWER_RECEIPT_SCHEMA_VERSION,
                require_science_plan=mode == "scientific",
            )
            guard_reason = (
                str(grounded.get("reason")) if grounded_valid else "strict_grounding_gate_not_selected"
            )
            answer_receipt = dict(grounded.get("answer_receipt") or {})
            verified_text = str(grounded.get("text") or "").strip()
            if grounded_valid:
                receipts["verified_answer_receipt"] = answer_receipt
                reasoning_row = dict(grounded.get("reasoning") or {})
                arithmetic_row = dict(grounded.get("arithmetic") or {})
                method = str(
                    reasoning_row.get("method")
                    or answer_receipt.get("method")
                    or "bounded_exact_arithmetic"
                )
                problem_class = str(
                    reasoning_row.get("problem_class")
                    or answer_receipt.get("problem_class")
                    or "arithmetic"
                )
                for step_text in list(reasoning_row.get("steps") or ())[:8]:
                    steps.append(
                        NexusThoughtStep(
                            step_index=len(steps) + 1,
                            stage="verified_reasoning",
                            content=str(step_text),
                            confidence=1.0,
                            telemetry={"method": method, "score_kind": "deterministic_in_scope"},
                        )
                    )
                if not steps and arithmetic_row.get("solved") is True:
                    steps.append(
                        NexusThoughtStep(
                            step_index=1,
                            stage="verified_arithmetic",
                            content="The bounded exact-arithmetic verifier recomputed the requested expression.",
                            confidence=1.0,
                            telemetry={"method": method, "score_kind": "deterministic_in_scope"},
                        )
                    )
                decision = epistemics.verified_exact_decision(
                    reason=guard_reason,
                    claim_scope=f"closed_world:{problem_class}:{method}",
                    verifier_id="grounding_runtime.finalize_grounded_response",
                    protocol={
                        "inference_regime": "deterministic_tool_verification",
                        "candidate_count": 1,
                        "verifier_calls": 1,
                        "generation_calls": 0,
                    },
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return NexusResult(
                    query=query_clean,
                    mode_selected="scientific" if mode == "scientific" else "solve",
                    final_output=verified_text,
                    thought_steps=steps,
                    confidence=decision.correctness_confidence,
                    latency_ms=round(elapsed_ms, 2),
                    speculative_acceptance_rate=None,
                    tool_calls_used=0,
                    audit_receipts=receipts,
                    telemetry={
                        "answer_admitted": True,
                        "guard_reason": guard_reason,
                        "problem_class": problem_class,
                        "method": method,
                        "verifier_calls": 1,
                        "external_tool_calls_executed": 0,
                    },
                    epistemics=decision.to_dict(),
                )

            if mode in ("scientific", "solve"):
                # A legacy match is recorded only as audit metadata.  Its
                # numeric candidate is deliberately not returned because the
                # strict full-request gate has already refused authority.
                legacy_match = self.solver_engine.solve(query_clean)
                legacy_audit: Dict[str, Any] = {
                    "matched": bool(legacy_match.solved),
                    "formula_id": str(legacy_match.formula_id if legacy_match.solved else ""),
                    "answer_withheld": True,
                    "full_receipt_withheld": True,
                    "receipt_is_authority": False,
                }
                if legacy_match.solved and legacy_match.receipt is not None:
                    legacy_audit["receipt_schema_version"] = legacy_match.receipt.schema_version
                receipts["legacy_nexus_solver_audit"] = legacy_audit
                receipts["grounding_gate_audit"] = {
                    "admitted": False,
                    "grounder_result_withheld": True,
                    "receipt_withheld": True,
                    "reason": "strict_grounding_gate_not_selected",
                }
                refusal_reason = "strict_grounding_gate_not_selected"
                decision = epistemics.abstained_decision(
                    reason=refusal_reason,
                    claim_scope="closed_world_problem_not_admitted",
                    limitations=(
                        "The strict verifier did not accept the complete request.",
                        "A legacy formula-pattern match, if present, was withheld because it is not full-query verification.",
                    ),
                    protocol={
                        "inference_regime": "verifier_first_abstention",
                        "candidate_count": int(bool(legacy_match.solved)),
                        "verifier_calls": 1,
                        "generation_calls": 0,
                    },
                )
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return NexusResult(
                    query=query_clean,
                    mode_selected=mode,
                    final_output=(
                        "I can't provide a verified answer for this request. The strict solver "
                        f"rejected it as '{refusal_reason}'. Please provide one unambiguous, "
                        "closed-world problem with all required assumptions and units."
                    ),
                    thought_steps=[
                        NexusThoughtStep(
                            step_index=1,
                            stage="abstention",
                            content="The verifier-first gate withheld an unverified formula match.",
                            telemetry={"reason": refusal_reason},
                        )
                    ],
                    confidence=None,
                    latency_ms=round(elapsed_ms, 2),
                    speculative_acceptance_rate=None,
                    tool_calls_used=0,
                    audit_receipts=receipts,
                    telemetry={
                        "answer_admitted": False,
                        "guard_reason": guard_reason,
                        "legacy_solver_match": bool(legacy_match.solved),
                        "verifier_calls": 1,
                        "external_tool_calls_executed": 0,
                    },
                    epistemics=decision.to_dict(),
                )

        # 2. Creative Ideation & Brainstorming Check
        if mode in ("auto", "innovate") and (
            mode == "innovate"
            or any(w in query_clean.lower() for w in ["brainstorm", "innovate", "new idea", "invent", "scamper", "triz", "novel concept", "what if"])
        ):
            idea_res = self.ideation_engine.brainstorm(query_clean, count=6)
            receipts["ideation_receipt"] = idea_res.receipt.to_dict()
            for c in idea_res.concepts[:4]:
                steps.append(
                    NexusThoughtStep(
                        step_index=len(steps) + 1,
                        stage="ideation",
                        content=f"[{c.operator}] {c.title} (static priority: {c.composite_score:.2f}) — {c.description}",
                        confidence=None,
                        telemetry={
                            "feasibility": c.feasibility,
                            "novelty": c.novelty,
                            "impact": c.impact,
                            "score_kind": "static_heuristic_priority_not_correctness",
                        },
                    )
                )
            decision = epistemics.analysis_only_decision(
                reason="structured_ideation_without_empirical_validation",
                claim_scope="creative_concept_generation",
                evidence_class="deterministic_heuristic",
                internal_score=float(idea_res.receipt.top_composite_score),
                internal_score_name="static_fnir_priority",
                limitations=(
                    "Concept scores are fixed heuristic priorities, not measured feasibility, novelty, impact, robustness, or correctness.",
                    "Claims and projected benefits require independent domain evidence and experiments.",
                ),
                protocol={
                    "inference_regime": "deterministic_template_ideation",
                    "candidate_count": len(idea_res.concepts),
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return NexusResult(
                query=query_clean,
                mode_selected="innovate",
                final_output="**Analysis-only concept sketch - not validated**\n\n" + idea_res.synthesis_proposal,
                thought_steps=steps,
                confidence=None,
                latency_ms=round(elapsed_ms, 2),
                speculative_acceptance_rate=None,
                tool_calls_used=0,
                audit_receipts=receipts,
                telemetry={
                    "concepts_count": len(idea_res.concepts),
                    "pareto_count": len(idea_res.pareto_optimal_concepts),
                    "internal_priority_score": idea_res.receipt.top_composite_score,
                    "score_kind": "static_heuristic_priority_not_correctness",
                    "external_tool_calls_executed": 0,
                },
                epistemics=decision.to_dict(),
            )

        # 3. Adaptive Persona Chat Check
        if mode == "chat":
            sess_id = session_id or f"sess_{hashlib.sha256(query_clean.encode()).hexdigest()[:8]}"
            chat_res = self.chat_engine.chat(sess_id, query_clean, requested_persona=persona, context_override=context)
            for t_step in chat_res.thought_steps:
                steps.append(
                    NexusThoughtStep(
                        step_index=len(steps) + 1,
                        stage="persona_chat",
                        content=t_step,
                        confidence=None,
                        telemetry={"score_kind": "not_scored"},
                    )
                )
            decision = epistemics.analysis_only_decision(
                reason="persona_template_without_factual_generator",
                claim_scope="conversation_scaffolding",
                evidence_class="deterministic_heuristic",
                limitations=(
                    "The persona engine produces deterministic conversation scaffolds, not a trained factual answer.",
                    "Any factual or high-stakes request still requires a verified solver or external evidence.",
                ),
                protocol={
                    "inference_regime": "deterministic_persona_template",
                    "candidate_count": 1,
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return NexusResult(
                query=query_clean,
                mode_selected="chat",
                final_output="**Conversation scaffold - no factual answer authority**\n\n" + chat_res.reply,
                thought_steps=steps,
                confidence=None,
                latency_ms=round(elapsed_ms, 2),
                speculative_acceptance_rate=None,
                tool_calls_used=0,
                audit_receipts={"persona": chat_res.persona_used.to_dict()},
                telemetry={
                    "session_id": sess_id,
                    "intent": chat_res.intent_detected,
                    "external_tool_calls_executed": 0,
                },
                epistemics=decision.to_dict(),
            )

        # 4. Adaptive Mode & Budget Selection (MiMo Router + Dem-Lab Q-Learner)
        req_features = controller.RequestFeatures(
            prompt_tokens=len(query_clean.split()),
            requested_acts=1 if len(query_clean.split()) < 15 else 2,
            tool_calls_available=len(tools),
            has_conflict="conflict" in query_clean.lower() or "versus" in query_clean.lower(),
            needs_evidence=len(tools) > 0 or "evidence" in query_clean.lower(),
            max_output_tokens=max_output_tokens,
        )

        resolved_mode = mode
        if resolved_mode == "auto":
            if len(tools) > 0 or req_features.difficulty() > 0.65:
                resolved_mode = "agent"
            elif "swarm" in query_clean.lower() or "debate" in query_clean.lower():
                resolved_mode = "swarm"
            elif "graph" in query_clean.lower() or "tree" in query_clean.lower():
                resolved_mode = "got"
            elif req_features.difficulty() > 0.35:
                resolved_mode = "deep"
            else:
                resolved_mode = "fast"

        # Q-learning starting budget proposal
        suggested = self.q_learner.suggest(
            difficulty=req_features.difficulty(),
            risk=req_features.epistemic_risk(),
        )
        q_budget = suggested if (suggested is not None and suggested in self.q_learner.budgets) else self.q_learner.budgets[0]
        budget_source = "verified_feedback_policy" if suggested is not None else "safe_default"
        if thinking_budget is not None:
            q_budget = max(1, min(self.config.max_thinking_budget, int(thinking_budget)))
            budget_source = "explicit_request_clamped"

        steps.append(
            NexusThoughtStep(
                step_index=len(steps) + 1,
                stage="route",
                content=f"Nexus Router assigned mode='{resolved_mode}' with bounded probe budget={q_budget} (difficulty={req_features.difficulty():.2f}, risk={req_features.epistemic_risk():.2f})",
                confidence=None,
                telemetry={
                    "difficulty_heuristic": req_features.difficulty(),
                    "q_budget_proposal": q_budget,
                    "q_budget_source": budget_source,
                    "score_kind": "routing_heuristic_not_correctness",
                },
            )
        )

        # 5. Execution by resolved mode.  Every branch must publish an
        # evidence decision; internal route/search scores never become answer
        # confidence.
        component_telemetry: Dict[str, Any] = {}
        if resolved_mode == "swarm":
            swarm_res = self.swarm_engine.deliberate(
                query=query_clean,
                external_context=context,
            )
            receipts["swarm_receipt"] = swarm_res.receipt.to_dict()
            for r in swarm_res.rounds:
                contrib_summary = ", ".join(
                    f"{k}:template_priority={v.confidence:.2f}"
                    for k, v in r.contributions.items()
                )
                steps.append(
                    NexusThoughtStep(
                        step_index=len(steps) + 1,
                        stage="swarm_debate",
                        content=f"Debate Round {r.round_index} [consensus={r.inter_agent_consensus:.2f}] — {contrib_summary}",
                        confidence=None,
                        telemetry={
                            "internal_template_agreement": r.inter_agent_consensus,
                            "score_kind": "template_agent_agreement_not_correctness",
                        },
                    )
                )
            decision = epistemics.analysis_only_decision(
                reason="template_swarm_without_grounded_candidate",
                claim_scope="structured_critique_scaffold",
                evidence_class="template_deliberation",
                internal_score=float(swarm_res.final_confidence),
                internal_score_name="template_agent_consensus",
                limitations=(
                    "Default agents emit fixed role templates rather than independently generated factual candidates.",
                    "Consensus over template scores is not verification and is not a probability of correctness.",
                ),
                protocol={
                    "inference_regime": "template_multi_agent_deliberation",
                    "candidate_count": 0,
                    "debate_rounds": len(swarm_res.rounds),
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            final_output = "**Analysis-only swarm scaffold - not a verified answer**\n\n" + swarm_res.consensus_output
            acceptance_rate = None
            component_telemetry = {
                "internal_consensus_score": swarm_res.final_confidence,
                "score_kind": "template_agent_consensus_not_correctness",
                "rounds": len(swarm_res.rounds),
            }

        elif resolved_mode == "got":
            got_res = self.got_engine.search(query=query_clean)
            receipts["got_receipt"] = got_res.receipt.to_dict()
            for node in got_res.best_path_nodes:
                steps.append(
                    NexusThoughtStep(
                        step_index=len(steps) + 1,
                        stage="got_branch",
                        content=f"[{node.branch_type} depth={node.depth} score={node.score:.2f}]: {node.step_text}",
                        confidence=None,
                        telemetry={
                            "internal_path_priority": node.score,
                            "score_kind": "template_position_priority_not_correctness_or_optimality",
                        },
                    )
                )
            decision = epistemics.analysis_only_decision(
                reason="template_graph_search_without_answer_generator",
                claim_scope="search_topology_scaffold",
                evidence_class="template_deliberation",
                internal_score=float(got_res.receipt.optimal_path_score),
                internal_score_name="positional_path_priority",
                limitations=(
                    "The default graph expands deterministic placeholder text and has no answer generator.",
                    "Its path score ranks template positions; it is not correctness, optimality, or verification.",
                ),
                protocol={
                    "inference_regime": "template_prefix_search",
                    "candidate_count": 0,
                    "nodes_generated": got_res.receipt.total_nodes_generated,
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            final_output = "**Analysis-only graph scaffold - not a verified answer**\n\n" + got_res.final_output
            acceptance_rate = None
            component_telemetry = {
                "internal_path_score": got_res.receipt.optimal_path_score,
                "score_kind": "positional_path_priority_not_correctness",
                "nodes_generated": got_res.receipt.total_nodes_generated,
            }

        elif resolved_mode == "agent":
            # Tool declarations describe availability; they are not executed
            # by NexusEngine and therefore cannot be counted as tool calls or
            # treated as evidence.
            decision = epistemics.abstained_decision(
                reason="no_tool_executor_or_answer_generator",
                claim_scope="agent_request",
                evidence_class="unverified_neural",
                limitations=(
                    "Supplied tool specifications were not executed and produced no receipts.",
                    "The experimental neural core has no trained checkpoint-backed text decoder.",
                ),
                protocol={
                    "inference_regime": "agent_unavailable",
                    "declared_tools": len(tools),
                    "executed_tools": 0,
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            steps.append(
                NexusThoughtStep(
                    step_index=len(steps) + 1,
                    stage="abstention",
                    content="Agent mode withheld an answer because no tool executor or verified generator is attached.",
                    telemetry={"declared_tools": len(tools), "executed_tools": 0},
                )
            )
            final_output = (
                "I can't execute the supplied tools or produce a verified agent answer in this "
                "experimental runtime. Tool declarations are availability metadata only; no "
                "tool was called."
            )
            acceptance_rate = None
            component_telemetry = {
                "declared_tool_count": len(tools),
                "external_tool_calls_executed": 0,
                "neural_generator_ready": False,
            }

        elif resolved_mode == "deep":
            input_ids = torch.tensor([[1] + [min(511, ord(c)) for c in query_clean[:64]]])
            with torch.no_grad():
                out = self.model(
                    input_ids,
                    thinking_cycles=min(self.config.max_thinking_budget, max(2, q_budget)),
                    adaptive_thinking=True,
                )
            probe_telemetry = _sanitize_untrained_probe_telemetry(out.telemetry)
            receipts["mimo_telemetry"] = probe_telemetry
            steps.append(
                NexusThoughtStep(
                    step_index=len(steps) + 1,
                    stage="ponder",
                    content=(
                        "Experimental ACT telemetry probe completed; no decoded answer candidate "
                        "was produced."
                    ),
                    confidence=None,
                    telemetry={**probe_telemetry, "score_kind": "telemetry_only"},
                )
            )
            decision = epistemics.abstained_decision(
                reason="untrained_core_has_no_decoded_answer",
                claim_scope="open_domain_neural_inference",
                evidence_class="unverified_neural",
                limitations=(
                    "The MiMo core is newly initialized, consumes at most 64 character ordinals, and has no loaded answer checkpoint.",
                    "ACT telemetry does not verify a proposition and cannot support a correctness score.",
                ),
                protocol={
                    "inference_regime": "single_trajectory_telemetry_probe",
                    "input_characters_used": min(64, len(query_clean)),
                    "thinking_cycles": min(self.config.max_thinking_budget, max(2, q_budget)),
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            final_output = (
                "I can't provide a verified answer for this request. The experimental deep core "
                "ran a telemetry-only latent probe, but no trained checkpoint-backed text decoder "
                "or applicable verifier is attached."
            )
            acceptance_rate = None
            component_telemetry = {
                "neural_generator_ready": False,
                "input_limit_characters": 64,
                "latent_probe": probe_telemetry,
            }

        else:
            # Fast / Flash telemetry mode.  The neural module is architecture
            # scaffolding, not a trained language generator.
            input_ids = torch.tensor([[1] + [min(511, ord(c)) for c in query_clean[:64]]])
            with torch.no_grad():
                out = self.model(input_ids, thinking_cycles=1)
            probe_telemetry = _sanitize_untrained_probe_telemetry(out.telemetry)
            steps.append(
                NexusThoughtStep(
                    step_index=len(steps) + 1,
                    stage="speculative_draft",
                    content="MiMo-Flash telemetry probe executed; no decoded answer candidate was produced.",
                    confidence=None,
                    telemetry={**probe_telemetry, "score_kind": "telemetry_only"},
                )
            )
            decision = epistemics.abstained_decision(
                reason="untrained_core_has_no_decoded_answer",
                claim_scope="open_domain_neural_inference",
                evidence_class="unverified_neural",
                limitations=(
                    "The MiMo core is newly initialized, consumes at most 64 character ordinals, and has no loaded answer checkpoint.",
                    "The speculative architecture probe produced telemetry, not a text candidate or measured acceptance rate.",
                ),
                protocol={
                    "inference_regime": "single_pass_telemetry_probe",
                    "input_characters_used": min(64, len(query_clean)),
                    "thinking_cycles": 1,
                    "verifier_calls": 0,
                    "generation_calls": 0,
                },
            )
            final_output = (
                "I can't provide a verified answer for this request. The experimental fast core "
                "ran a telemetry-only architecture probe, but no trained checkpoint-backed text "
                "decoder or applicable verifier is attached."
            )
            acceptance_rate = None
            component_telemetry = {
                "neural_generator_ready": False,
                "input_limit_characters": 64,
                "latent_probe": probe_telemetry,
            }

        # 6. Dem-Lab synthetic observability probe & Multi-Source Entropy
        sample_probs = [0.25, 0.25, 0.25, 0.25]
        ent = observatory.shannon_entropy(sample_probs)
        rsi_nov = observatory.novelty_score([1, 2, 3, 4], [[1, 2, 3, 5], [1, 2, 4, 5]]).get("novelty", 0.0)
        rsi_stab = observatory.stability_score([0.8, 0.82, 0.81]).get("stability", 1.0)

        # Synthetic RSI diagnostic. The sine sequence depends only on step
        # count; it is not token entropy, a model-quality signal, or reasoning
        # stability evidence.
        step_entropy_values = [
            0.5 + 0.1 * math.sin(i * 0.8) for i in range(len(steps) + 1)
        ]
        latest_rsi_diag = self.rsi_oscillator.update(
            step_entropy_values[-1] if step_entropy_values else 0.5
        )
        latest_rsi_diag.update(
            {
                "input_source": "synthetic_step_count_sine_probe",
                "is_live_reasoning_signal": False,
            }
        )

        # Sample entropy stream
        entropy_samples = self.entropy_engine.sample(source=entropy_src, count=8)

        # Internal scores and abstentions are not outcome labels. Learning is
        # deliberately skipped until external, verifier-backed feedback is
        # supplied through a separate reviewed path.
        policy_update = "skipped_requires_external_verified_feedback"

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        telemetry = {
            "synthetic_observability_probe": {
                "dem_lab_entropy": round(ent, 4),
                "rsi_novelty": round(rsi_nov, 4),
                "rsi_stability": round(rsi_stab, 4),
                "is_live_quality_evidence": False,
            },
            "entropy_telemetry": {
                "active_source": entropy_src,
                "samples_preview": entropy_samples[:4],
                "mean_entropy_value": round(sum(entropy_samples) / max(1, len(entropy_samples)), 4),
                "provenance": self.entropy_engine.source_provenance(entropy_src),
            },
            "rsi_diagnostic": latest_rsi_diag,
            "q_budget_selected": q_budget,
            "q_budget_source": budget_source,
            "q_learning_update": policy_update,
            "moe_expert_count": self.config.n_experts,
            "sliding_window": self.config.sliding_window,
            "hybrid_ratio": self.config.hybrid_ratio,
            **component_telemetry,
        }

        return NexusResult(
            query=query_clean,
            mode_selected=resolved_mode,
            final_output=final_output,
            thought_steps=steps,
            confidence=decision.correctness_confidence,
            latency_ms=round(elapsed_ms, 2),
            speculative_acceptance_rate=acceptance_rate,
            tool_calls_used=0,
            audit_receipts=receipts,
            telemetry=telemetry,
            epistemics=decision.to_dict(),
        )


def build_default_engine() -> NexusEngine:
    return NexusEngine()
