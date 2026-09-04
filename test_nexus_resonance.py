"""Unit tests for Semantic Resonance & Archetype Basin Mapping."""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import pytest
from nexus_engine import SemanticResonanceMapper, SemanticResonanceResult, NexusEngine


def test_semantic_resonance_logos_mapping():
    mapper = SemanticResonanceMapper()
    res = mapper.map_query("Calculate the exact integral and solve the math equation theorem")
    assert isinstance(res, SemanticResonanceResult)
    assert res.dominant_archetype == "logos"
    assert res.archetype_scores["logos"] > res.archetype_scores["mythos"]
    assert 0.0 <= res.resonance_score <= 1.0
    assert res.mixture_entropy > 0.0
    assert len(res.coordinates_2d) == 2


def test_semantic_resonance_mythos_mapping():
    mapper = SemanticResonanceMapper()
    res = mapper.map_query("Imagine a novel fantasy world, brainstorm creative metaphors and innovate")
    assert res.dominant_archetype == "mythos"
    assert res.archetype_scores["mythos"] > res.archetype_scores["logos"]


def test_semantic_resonance_ethos_mapping():
    mapper = SemanticResonanceMapper()
    res = mapper.map_query("Verify the audit receipt, inspect the ledger, and check the evidence safety gate")
    assert res.dominant_archetype == "ethos"


def test_nexus_engine_run_semantic_resonance():
    engine = NexusEngine()
    res = engine.run_semantic_resonance("Tell me how to plan and execute the agent mission")
    assert res.dominant_archetype == "telos"
    data = res.to_dict()
    assert "archetype_scores" in data
    assert "dominant_archetype" in data
