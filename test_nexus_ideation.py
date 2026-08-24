"""Tests for the NexusIdeationEngine — SCAMPER/TRIZ/FNIR creative innovation engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
for p in (ROOT, SOURCE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import nexus_ideation as ni


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------

def test_generate_innovations_returns_result():
    res = ni.generate_innovations("distributed AI routing", count=4)
    assert isinstance(res, ni.IdeationResult)
    assert res.query == "distributed AI routing"
    assert len(res.concepts) >= 1


def test_concepts_have_required_fields():
    res = ni.generate_innovations("quantum computing optimization")
    for c in res.concepts:
        assert isinstance(c, ni.IdeaConcept)
        assert c.concept_id
        assert c.title
        assert c.operator
        assert c.description
        assert 0.0 <= c.feasibility <= 1.0
        assert 0.0 <= c.novelty <= 1.0
        assert 0.0 <= c.impact <= 1.0
        assert 0.0 <= c.robustness <= 1.0


def test_composite_score_range():
    res = ni.generate_innovations("protein folding simulation")
    for c in res.concepts:
        score = c.composite_score
        assert 0.0 <= score <= 1.0


def test_pareto_frontier_is_non_empty():
    res = ni.generate_innovations("edge computing resource scheduling")
    assert len(res.pareto_optimal_concepts) >= 1


def test_pareto_optimal_concepts_are_not_dominated():
    res = ni.generate_innovations("drug discovery pipeline")
    pareto = res.pareto_optimal_concepts
    for c1 in pareto:
        for c2 in pareto:
            if c1.concept_id == c2.concept_id:
                continue
            # Neither c1 nor c2 should dominate each other
            c1_dominates_c2 = (
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
            )
            # In the Pareto set, by definition, no member should be dominated
            assert not c1_dominates_c2, f"{c2.concept_id} dominates {c1.concept_id} — they shouldn't both be in Pareto frontier"


def test_synthesis_proposal_non_empty():
    res = ni.generate_innovations("neuromorphic chip architecture")
    assert res.synthesis_proposal
    assert len(res.synthesis_proposal) > 50


def test_operators_cover_multiple_families():
    res = ni.generate_innovations("renewable energy storage")
    operators = {c.operator.split(":")[0] for c in res.concepts}
    # Should have at least SCAMPER and TRIZ or Analogy
    assert len(operators) >= 2


def test_receipt_generated():
    res = ni.generate_innovations("climate change modelling")
    assert isinstance(res.receipt, ni.IdeationReceipt)
    assert len(res.receipt.receipt_sha256) == 64
    assert res.receipt.total_concepts_generated >= 1
    assert res.receipt.pareto_concepts_count >= 1


def test_count_parameter_respected():
    res = ni.generate_innovations("robotics swarm control", count=3)
    assert len(res.concepts) <= 3


def test_different_topics_give_different_receipts():
    r1 = ni.generate_innovations("quantum cryptography")
    r2 = ni.generate_innovations("autonomous vehicle perception")
    assert r1.receipt.query_digest != r2.receipt.query_digest


def test_to_dict_serializable():
    import json
    res = ni.generate_innovations("sustainable agriculture AI")
    d = res.to_dict()
    # Should be JSON-serializable
    json_str = json.dumps(d)
    assert len(json_str) > 100


def test_concept_is_pareto_flag_set():
    res = ni.generate_innovations("low-latency inference optimization")
    pareto_ids = {c.concept_id for c in res.pareto_optimal_concepts}
    for c in res.concepts:
        if c.concept_id in pareto_ids:
            assert c.is_pareto_optimal is True


def test_brainstorm_engine_instance():
    engine = ni.NexusIdeationEngine()
    res = engine.brainstorm("memory-efficient transformer attention", count=5)
    assert isinstance(res, ni.IdeationResult)
    assert len(res.concepts) >= 1
