"""NexusMind Creative Ideation & Lateral Innovation Engine.

Master engine for breakthrough idea generation, lateral problem solving,
and conceptual synthesis:
1. **SCAMPER Transformation Matrix**:
   - Substitute, Combine, Adapt, Modify/Magnify, Put to other uses, Eliminate, Reverse/Rearrange
2. **TRIZ Inventive Principles**:
   - Contradiction resolution, segmentation, asymmetry, dynamic adaptation, self-service feedback
3. **Cross-Domain Analogical Synthesis**:
   - Transposes structural patterns across Biological, Physical/Thermodynamic, Computational, and Game-Theoretic domains
4. **FNIR Multi-Objective Prioritization**:
   - Applies authored Feasibility, Novelty, Impact, and Robustness heuristics
   - Identifies Pareto-optimal frontier candidates and synthesizes unified breakthrough proposals
5. **Cryptographic Ideation Receipt**:
   - Generates deterministic `IdeationReceipt` with concept digests
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


__all__ = [
    "IdeaConcept",
    "IdeationReceipt",
    "IdeationResult",
    "NexusIdeationEngine",
    "generate_innovations",
]


IDEATION_RECEIPT_SCHEMA = "nexus-ideation-receipt-v1"


@dataclass
class IdeaConcept:
    """A distinct innovative concept generated through structured lateral operators."""

    concept_id: str
    title: str
    operator: str  # "SCAMPER:Combine", "TRIZ:Inversion", "Analogical:Biology", etc.
    description: str
    mechanism: str
    target_benefit: str
    feasibility: float  # [0.0, 1.0]
    novelty: float      # [0.0, 1.0]
    impact: float       # [0.0, 1.0]
    robustness: float   # [0.0, 1.0]
    is_pareto_optimal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def composite_score(self) -> float:
        """Weighted harmonic composite score prioritizing balanced high-novelty high-impact."""
        weights = (0.25, 0.30, 0.30, 0.15)
        return (
            weights[0] * self.feasibility
            + weights[1] * self.novelty
            + weights[2] * self.impact
            + weights[3] * self.robustness
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["composite_score"] = round(self.composite_score, 4)
        data["score_semantics"] = "authored_heuristic_priority_not_measurement"
        return data


@dataclass
class IdeationReceipt:
    """Cryptographic audit receipt for an ideation run."""

    schema_version: str = IDEATION_RECEIPT_SCHEMA
    query_digest: str = ""
    total_concepts_generated: int = 0
    pareto_concepts_count: int = 0
    top_concept_id: str = ""
    top_composite_score: float = 0.0
    operators_applied: List[str] = field(default_factory=list)
    receipt_sha256: str = ""
    receipt_is_authority: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IdeationResult:
    """Master result returned by the Ideation Engine."""

    query: str
    core_problem_statement: str
    concepts: List[IdeaConcept]
    pareto_optimal_concepts: List[IdeaConcept]
    synthesis_proposal: str
    receipt: IdeationReceipt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "core_problem_statement": self.core_problem_statement,
            "concepts": [c.to_dict() for c in self.concepts],
            "pareto_optimal_concepts": [c.to_dict() for c in self.pareto_optimal_concepts],
            "synthesis_proposal": self.synthesis_proposal,
            "receipt": self.receipt.to_dict(),
            "status": "analysis_only",
            "answer_authority": False,
            "score_semantics": "authored_fnir_priority_not_measurement_or_correctness",
        }


class NexusIdeationEngine:
    """Master creativity & lateral ideation engine."""

    SCAMPER_PROMPTS = {
        "Substitute": "What components, assumptions, materials, or actors can be replaced with radical alternatives?",
        "Combine": "What unrelated technologies, disciplines, or methodologies can be fused together?",
        "Adapt": "How does nature, biology, physics, or another industry solve an analogous bottleneck?",
        "Modify/Magnify": "What happens if we amplify key parameters by 100x or reduce constraints to zero?",
        "Put to Other Uses": "How can byproduct data, waste compute, or ambient signals be repurposed?",
        "Eliminate": "What traditional intermediary, friction, or legacy step can be deleted entirely?",
        "Reverse/Rearrange": "What if we reverse the causal order, invert dependencies, or turn the system inside out?",
    }

    TRIZ_PRINCIPLES = [
        ("Segmentation / Micro-Modularity", "Deconstruct monolithic architecture into autonomous, self-healing micro-units."),
        ("Asymmetry / Directed Bias", "Break uniform symmetry to concentrate compute or capacity where gradients are steepest."),
        ("Local Quality / Context Specialization", "Tailor execution rules dynamically to the specific locality or data regime."),
        ("Inversion / The Other Way Around", "Instead of bringing queries to the solver, bring active micro-solvers to the data stream."),
        ("Self-Service & Dynamic Feedback", "Enable the system to perform self-speculation, self-correction, and autonomous refinement."),
        ("Equipotentiality / Frictionless Handoff", "Eliminate state translation overhead between cognitive stages."),
    ]

    ANALOGICAL_DOMAINS = [
        ("Biological Immunology", "Adaptive memory cells, multi-tier antigen recognition, somatic hypermutation for rapid response."),
        ("Mycelial Distribution", "Decentralized nutrient routing, self-repairing graph pathways, gradient-following network optimization."),
        ("Thermodynamic Annealing", "Controlled exploration noise cooling down into globally optimal low-entropy states."),
        ("Swarm Ant Stigmergy", "Indirect coordination via persistent environmental pheromone traces rather than central bottlenecking."),
        ("Quantum Superposition Search", "Maintaining multiple speculative hypotheses in parallel until evidence triggers constructive collapse."),
    ]

    def brainstorm(self, topic: str, count: int = 6) -> IdeationResult:
        """Generate structured innovative concepts across SCAMPER, TRIZ, and Analogies."""
        clean_topic = topic.strip()
        concepts: List[IdeaConcept] = []

        # 1. Apply SCAMPER operators
        scamper_ops = [
            ("SCAMPER:Combine", "Hybrid Synthesis Fusion", f"Explore fusing {clean_topic} with bounded feedback loops and distributed review.", "Prototype cross-stream coordination with explicit failure and rollback boundaries.", "Hypothesis: reduce single-point assumptions; measure flexibility, recovery time, and coordination cost.", 0.88, 0.92, 0.94, 0.85),
            ("SCAMPER:Adapt", "Biomimetic Adaptive Scaling", f"Explore an immune-memory analogy for adapting and scaling {clean_topic}.", "Prototype affinity-based routing with bounded reinforcement and decay.", "Hypothesis: improve adaptation to novel conditions; benchmark latency, false routing, and stability.", 0.84, 0.95, 0.90, 0.89),
            ("SCAMPER:Eliminate", "Direct-Path Data Flow", f"Test whether intermediate serialization or buffering can be reduced in {clean_topic}.", "Evaluate zero-copy streaming and bounded lock-free queues where platform semantics permit.", "Hypothesis: reduce latency or memory overhead; validate with profiling and concurrency stress tests.", 0.93, 0.82, 0.88, 0.91),
            ("SCAMPER:Reverse", "Inverted Speculative Execution", f"Explore pre-computing bounded candidates before the full {clean_topic} request is available.", "Use draft trees only when a verifier can reject incorrect or stale candidates.", "Hypothesis: lower perceived latency without degrading verified quality; test acceptance and error rates.", 0.86, 0.93, 0.92, 0.87),
        ]

        for op, title, desc, mech, benefit, f_score, n_score, i_score, r_score in scamper_ops:
            cid = f"scamper_{len(concepts)+1}"
            concepts.append(
                IdeaConcept(
                    concept_id=cid,
                    title=title,
                    operator=op,
                    description=desc,
                    mechanism=mech,
                    target_benefit=benefit,
                    feasibility=f_score,
                    novelty=n_score,
                    impact=i_score,
                    robustness=r_score,
                )
            )

        # 2. Apply TRIZ principles
        triz_ops = [
            ("TRIZ:Segmentation", "Bounded Micro-Kernel Mesh", f"Decompose {clean_topic} into isolated micro-engines with explicit peer-message contracts.", "Prototype localized stopping policies and fail-closed coordination.", "Hypothesis: contain component failures; validate partial-failure behavior and recovery limits.", 0.90, 0.89, 0.93, 0.94),
            ("TRIZ:Self-Service", "Receipt-Gated Recovery Loop", f"Equip {clean_topic} with verifiers that can propose, check, and roll back bounded corrections.", "Track externally verified outcomes before any policy or parameter update.", "Hypothesis: reduce some manual recovery work while retaining review, audit, and override controls.", 0.89, 0.91, 0.95, 0.90),
        ]

        for op, title, desc, mech, benefit, f_score, n_score, i_score, r_score in triz_ops:
            cid = f"triz_{len(concepts)+1}"
            concepts.append(
                IdeaConcept(
                    concept_id=cid,
                    title=title,
                    operator=op,
                    description=desc,
                    mechanism=mech,
                    target_benefit=benefit,
                    feasibility=f_score,
                    novelty=n_score,
                    impact=i_score,
                    robustness=r_score,
                )
            )

        # 3. Apply Cross-Domain Analogical Mappings
        analogy_ops = [
            ("Analogy:MultiBranch", "Multi-Branch Hypothesis Set", f"Maintain several bounded hypotheses for {clean_topic} until discriminating evidence is available.", "Compare candidates under a declared test protocol with pruning based on measured outcomes.", "Hypothesis: reduce premature convergence; compare against a single-candidate baseline at matched compute.", 0.82, 0.96, 0.96, 0.84),
            ("Analogy:MycelialStigmergy", "Decaying Coordination Trace", f"Coordinate distributed tasks in {clean_topic} through inspectable, expiring state traces.", "Use decay, provenance, and bounded reinforcement for routing hints.", "Hypothesis: reduce central coordination pressure; measure bottlenecks, stale-state errors, and recovery.", 0.87, 0.94, 0.91, 0.92),
        ]

        for op, title, desc, mech, benefit, f_score, n_score, i_score, r_score in analogy_ops:
            cid = f"analogy_{len(concepts)+1}"
            concepts.append(
                IdeaConcept(
                    concept_id=cid,
                    title=title,
                    operator=op,
                    description=desc,
                    mechanism=mech,
                    target_benefit=benefit,
                    feasibility=f_score,
                    novelty=n_score,
                    impact=i_score,
                    robustness=r_score,
                )
            )

        # 4. Compute Pareto-Optimal Frontier
        pareto = self._compute_pareto_frontier(concepts)
        for c in concepts:
            c.is_pareto_optimal = c in pareto

        # Sort concepts by composite score descending
        concepts.sort(key=lambda x: x.composite_score, reverse=True)
        pareto.sort(key=lambda x: x.composite_score, reverse=True)

        # 5. Synthesize a testable hybrid hypothesis. The ranking is static and
        # must not be presented as empirical evidence.
        top_concepts = pareto[:3] if len(pareto) >= 3 else concepts[:3]
        synthesis = (
            f"### Testable Hybrid Hypothesis for '{clean_topic}'\n\n"
            "The authored FNIR rankings prioritize three concepts for investigation; they do not validate them: "
            f"**{top_concepts[0].title}** + **{top_concepts[1].title}** + **{top_concepts[2].title}**.\n\n"
            f"1. **Prototype boundary**: {top_concepts[0].mechanism}\n"
            f"2. **Comparison mechanism**: {top_concepts[1].mechanism}\n"
            f"3. **Safety and quality check**: {top_concepts[2].mechanism}\n\n"
            "**Validation required**: define a baseline, matched-compute protocol, failure cases, rollback criteria, "
            "and held-out outcome metrics before treating any projected benefit as established."
        )

        # 6. Emits Cryptographic Ideation Receipt
        receipt = self._build_receipt(clean_topic, concepts, pareto)

        return IdeationResult(
            query=clean_topic,
            core_problem_statement=f"Exploration and innovation across lateral dimensions for: {clean_topic}",
            concepts=concepts[:count],
            pareto_optimal_concepts=pareto,
            synthesis_proposal=synthesis,
            receipt=receipt,
        )

    def _compute_pareto_frontier(self, concepts: List[IdeaConcept]) -> List[IdeaConcept]:
        """Identify non-dominated concepts along (Feasibility, Novelty, Impact, Robustness)."""
        pareto: List[IdeaConcept] = []
        for c1 in concepts:
            dominated = False
            for c2 in concepts:
                if c1 is c2:
                    continue
                # c2 dominates c1 if c2 >= c1 in all dims and strictly greater in at least one
                if (
                    c2.feasibility >= c1.feasibility
                    and c2.novelty >= c1.novelty
                    and c2.impact >= c1.impact
                    and c2.robustness >= c1.robustness
                    and (
                        c2.feasibility > c1.feasibility
                        or c2.novelty > c1.novelty
                        or c2.impact > c1.impact
                        or c2.robustness > c1.robustness
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c1)
        return pareto if pareto else concepts[:2]

    def _build_receipt(
        self,
        query: str,
        concepts: List[IdeaConcept],
        pareto: List[IdeaConcept],
    ) -> IdeationReceipt:
        q_digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        top_c = concepts[0] if concepts else None
        ops = sorted(list(set(c.operator for c in concepts)))

        payload = {
            "schema_version": IDEATION_RECEIPT_SCHEMA,
            "query_digest": q_digest,
            "total_concepts": len(concepts),
            "pareto_concepts_count": len(pareto),
            "top_concept_id": top_c.concept_id if top_c else "",
            "top_composite_score": top_c.composite_score if top_c else 0.0,
            "operators": ops,
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        receipt_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

        return IdeationReceipt(
            schema_version=IDEATION_RECEIPT_SCHEMA,
            query_digest=q_digest,
            total_concepts_generated=len(concepts),
            pareto_concepts_count=len(pareto),
            top_concept_id=top_c.concept_id if top_c else "",
            top_composite_score=round(top_c.composite_score, 4) if top_c else 0.0,
            operators_applied=ops,
            receipt_sha256=receipt_sha256,
        )


_DEFAULT_IDEATION = NexusIdeationEngine()


def generate_innovations(topic: str, count: int = 6) -> IdeationResult:
    """Convenience functional interface for NexusIdeationEngine."""
    return _DEFAULT_IDEATION.brainstorm(topic, count=count)
