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

    @staticmethod
    def _extract_topic_profile(
        topic: str,
        constraints: Optional[Sequence[str]] = None,
        domain_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        lower = topic.lower()
        stop = {
            "the", "and", "for", "with", "into", "that", "this", "from", "how",
            "can", "what", "why", "way", "ways", "idea", "ideas", "brainstorm",
            "make", "better", "improve", "new", "more", "help", "about", "our",
        }
        keywords = [
            w for w in re.findall(r"[a-zA-Z0-9_+-]{3,}", lower)
            if w not in stop
        ]
        # Preserve order without duplicates
        unique_kw = list(dict.fromkeys(keywords))[:8]
        focus_phrase = ", ".join(unique_kw[:4]) if unique_kw else topic.strip()

        detected_domain = domain_hint or "general"
        if detected_domain == "general":
            if any(k in lower for k in ("latency", "memory", "cache", "server", "api", "database", "bug", "race", "queue", "code", "rust", "python", "gpu", "cpu")):
                detected_domain = "software_systems"
            elif any(k in lower for k in ("model", "llm", "neural", "train", "token", "inference", "eval", "dataset", "attention", "moe", "connectome")):
                detected_domain = "ml_ai"
            elif any(k in lower for k in ("math", "equation", "proof", "physics", "energy", "circuit", "chemistry", "bio")):
                detected_domain = "math_science"
            elif any(k in lower for k in ("user", "ui", "chat", "design", "app", "product", "workflow", "onboarding")):
                detected_domain = "product_ux"

        domain_lever_map = {
            "software_systems": ("hot-path serialization / lock contention", "lock-free batching, bounded caches, and replayable state checks", "p99 latency, RSS memory, and race-condition error rate"),
            "ml_ai": ("static full-depth compute per token", "verified early-exit routing, sparse expert gating, and failure-aware distillation", "held-out accuracy, tokens/sec, and calibration drift"),
            "math_science": ("single-pass heuristic estimation", "symbolic invariant decompositions paired with dimensional/boundary assertions", "exact conservation residuals, edge-case coverage, and step fidelity"),
            "product_ux": ("multi-step modal friction", "progressive disclosure, sticky user context, and undoable micro-actions", "task completion time, retention, and accidental-reset rate"),
            "general": ("monolithic linear execution", "modular decoupled stages with early feedback and falsifiable checkpoints", "throughput, defect rate, and iteration speed"),
        }
        bottleneck, lever, metric = domain_lever_map.get(detected_domain, domain_lever_map["general"])
        clean_constraints = [str(c).strip() for c in (constraints or ()) if str(c).strip()][:4]
        return {
            "domain": detected_domain,
            "keywords": unique_kw,
            "focus_phrase": focus_phrase,
            "bottleneck": bottleneck,
            "lever": lever,
            "metric": metric,
            "constraints": clean_constraints,
        }

    @staticmethod
    def _concept_tokens(concept: IdeaConcept) -> set:
        return set(re.findall(r"[a-z0-9_+-]{3,}", f"{concept.title} {concept.description} {concept.mechanism}".lower()))

    @classmethod
    def _select_map_elites_diverse(cls, concepts: List[IdeaConcept], target_count: int) -> List[IdeaConcept]:
        """SONAPSE-ORIGIN MAP-Elites 2D niche selection + NoveltySearch distance pruning."""
        grid: Dict[Tuple[int, int], IdeaConcept] = {}
        f_vals = [float(c.feasibility) for c in concepts]
        n_vals = [float(c.novelty) for c in concepts]
        f_min, f_max = (min(f_vals), max(f_vals)) if f_vals else (0.0, 1.0)
        n_min, n_max = (min(n_vals), max(n_vals)) if n_vals else (0.0, 1.0)
        f_span = max(1e-6, f_max - f_min)
        n_span = max(1e-6, n_max - n_min)
        for c in concepts:
            f_norm = (float(c.feasibility) - f_min) / f_span
            n_norm = (float(c.novelty) - n_min) / n_span
            ix = max(0, min(3, int(f_norm * 3.999)))
            iy = max(0, min(3, int(n_norm * 3.999)))
            c.metadata["map_elites_cell"] = [ix, iy]
            cur = grid.get((ix, iy))
            if cur is None or c.composite_score > cur.composite_score:
                grid[(ix, iy)] = c

        elites = sorted(grid.values(), key=lambda x: x.composite_score, reverse=True)
        selected: List[IdeaConcept] = []
        pool = elites + [c for c in sorted(concepts, key=lambda x: x.composite_score, reverse=True) if c not in elites]

        for _ in range(min(target_count, len(pool))):
            best_cand: Optional[IdeaConcept] = None
            best_rank = -1.0
            for cand in pool:
                if cand in selected:
                    continue
                if not selected:
                    nov_dist = 1.0
                else:
                    t_c = cls._concept_tokens(cand)
                    dists = []
                    for sel in selected:
                        t_s = cls._concept_tokens(sel)
                        union = len(t_c | t_s)
                        dists.append(1.0 - (len(t_c & t_s) / max(1, union)))
                    nov_dist = sum(dists) / max(1, len(dists))
                cand.metadata["novelty_distance"] = round(nov_dist, 4)
                rank = 0.72 * cand.composite_score + 0.28 * nov_dist
                if rank > best_rank:
                    best_rank = rank
                    best_cand = cand
            if best_cand is not None:
                selected.append(best_cand)
        return selected

    def brainstorm(
        self,
        topic: str,
        count: int = 6,
        *,
        constraints: Optional[Sequence[str]] = None,
        domain_hint: Optional[str] = None,
    ) -> IdeationResult:
        """Generate context-grounded concepts across SCAMPER/TRIZ/Analogies with MAP-Elites diversity."""
        clean_topic = topic.strip() or "system improvement"
        prof = self._extract_topic_profile(clean_topic, constraints=constraints, domain_hint=domain_hint)
        focus = prof["focus_phrase"]
        lever = prof["lever"]
        bottleneck = prof["bottleneck"]
        metric = prof["metric"]
        constraint_note = (
            f" Respecting active constraints: {'; '.join(prof['constraints'])}."
            if prof["constraints"]
            else ""
        )

        candidates_spec = [
            (
                "scamper",
                "SCAMPER:Eliminate",
                f"Strip {bottleneck} from {focus}",
                f"Target the dominant friction ({bottleneck}) in {clean_topic} before adding new layers.",
                f"Bypass intermediate transformations using {lever}.{constraint_note}",
                f"Hypothesis: falsification test — benchmark baseline vs bypassed path on {metric}; keep only if p95 improves with zero defect increase.",
                0.94, 0.76, 0.88, 0.93,
            ),
            (
                "scamper",
                "SCAMPER:Combine",
                f"Closed-Loop Critic + Fast Generator for {focus}",
                f"Pair an optimistic fast generator for {clean_topic} with a lightweight invariant checker that intercepts failure modes early.",
                f"Execute speculative/fast draft path alongside deterministic check gates and automatic fallback.{constraint_note}",
                f"Hypothesis: falsification test — measure end-to-end {metric} against unverified single-pass execution under injected edge cases.",
                0.89, 0.87, 0.95, 0.91,
            ),
            (
                "scamper",
                "SCAMPER:Reverse",
                f"Inversion / Failure-First Design for {focus}",
                f"Invert the usual design flow of {clean_topic}: start from hardest failure inputs ({bottleneck}) and constrain the architecture backwards.",
                f"Build a joint-failure suite covering correlated failure modes (FAULTLINE pattern) before scaling capacity.{constraint_note}",
                "Hypothesis: falsification test — test worst-slice robustness on held-out stress conditions compared against average-case tuning.",
                0.85, 0.94, 0.92, 0.90,
            ),
            (
                "scamper",
                "SCAMPER:Adapt",
                f"Surprise-Gated Adaptive Routing in {focus}",
                f"Adapt biological prediction-error / homeostatic gating so {clean_topic} allocates extra compute only on high-uncertainty cases.",
                f"Route routine queries through a fast minimal path and escalate only high-entropy / conflict cases to deep deliberation.{constraint_note}",
                f"Hypothesis: falsification test — verify equal accuracy on hard slice while cutting mean compute/latency on {metric}.",
                0.84, 0.92, 0.93, 0.88,
            ),
            (
                "triz",
                "TRIZ:Segmentation",
                f"Subroutine Library & Modular Boundaries for {focus}",
                f"Break {clean_topic} into independently testable primitives that must be rediscovered across >=2 tasks before promotion (Tessera pattern).",
                f"Separate volatile heuristics from verified reusable kernels with clear rollback seams.{constraint_note}",
                "Hypothesis: falsification test — verify reuse transfers positively on unseen compositions rather than memorizing training fixtures.",
                0.90, 0.85, 0.91, 0.94,
            ),
            (
                "triz",
                "TRIZ:SelfService",
                f"Receipt-Verified Self-Repair Loop for {focus}",
                f"Instrument {clean_topic} so anomalous outputs trigger local containment, root-cause localization, and shadow validation prior to rollout.",
                f"Require paired baseline-vs-candidate receipt improvement before any parameter or policy change promotes.{constraint_note}",
                "Hypothesis: falsification test — confirm automated promotion rejects noisy single-seed blips under paired evaluation.",
                0.87, 0.89, 0.94, 0.92,
            ),
            (
                "analogy",
                "Analogy:ImmuneMemory",
                f"Multi-Tier Episodic & Conflict Memory for {focus}",
                f"Treat edge-case bugs and past solutions in {clean_topic} like tiered immune memory: fast working buffer + surprise-weighted episodic recall + contradiction alerts.",
                f"Index successful interventions and explicitly flag when new assumptions contradict established constraints.{constraint_note}",
                "Hypothesis: falsification test — measure repeat-mistake reduction across multi-step sessions compared to stateless baseline.",
                0.86, 0.91, 0.90, 0.89,
            ),
            (
                "analogy",
                "Analogy:VersionSpace",
                f"Crosswire Hypothesis Preservation under Ambiguity ({focus})",
                f"Instead of committing to one brittle guess on {clean_topic}, keep all surviving hypotheses consistent with observations and probe the sharpest disagreement.",
                f"Run targeted disambiguating questions/checks whenever candidate hypotheses diverge.{constraint_note}",
                "Hypothesis: falsification test — compare wrong-commitment rate on underspecified prompts vs single-best guessing.",
                0.81, 0.96, 0.94, 0.86,
            ),
        ]

        concepts: List[IdeaConcept] = []
        for idx, (prefix, op, title, desc, mech, benefit, f_s, n_s, i_s, r_s) in enumerate(candidates_spec, start=1):
            concepts.append(
                IdeaConcept(
                    concept_id=f"{prefix}_{idx}",
                    title=title,
                    operator=op,
                    description=desc,
                    mechanism=mech,
                    target_benefit=benefit,
                    feasibility=f_s,
                    novelty=n_s,
                    impact=i_s,
                    robustness=r_s,
                    metadata={"domain": prof["domain"], "keywords": prof["keywords"][:4]},
                )
            )

        pareto = self._compute_pareto_frontier(concepts)
        for c in concepts:
            c.is_pareto_optimal = c in pareto

        selected = self._select_map_elites_diverse(concepts, max(1, int(count)))
        pareto.sort(key=lambda x: x.composite_score, reverse=True)
        top_concepts = selected[:3] if len(selected) >= 3 else concepts[:3]
        synthesis = (
            f"### Convergent Action Plan for '{clean_topic}' ({prof['domain']})\n\n"
            "Ranked via MAP-Elites niche coverage (Feasibility x Novelty) + Pareto FNIR trade-offs (advisory heuristic):\n"
            f"- **Quick Win (High Feasibility)**: **{top_concepts[0].title}** — {top_concepts[0].mechanism}\n"
            f"- **Core Lever**: **{top_concepts[1].title}** — {top_concepts[1].mechanism}\n"
            f"- **Breakthrough / Contrarian Angle**: **{top_concepts[2].title}** — {top_concepts[2].mechanism}\n\n"
            f"**Validation required**: track `{metric}` on a paired baseline vs candidate sweep; use {top_concepts[0].target_benefit}"
        )

        receipt = self._build_receipt(clean_topic, selected, pareto)
        return IdeationResult(
            query=clean_topic,
            core_problem_statement=f"Domain-grounded MAP-Elites & SCAMPER/TRIZ exploration for: {clean_topic}",
            concepts=selected,
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
