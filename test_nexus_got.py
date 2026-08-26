"""Tests for the NexusMind Graph-of-Thoughts (GoT) Reasoner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_got as got


def test_got_initialization_and_node_creation():
    engine = got.GraphOfThoughts(max_depth=3, beam_width=2)
    root = engine.add_node("Initial hypothesis", depth=0, score=1.0, branch_type="root")

    assert root.node_id == "node_0001"
    assert root.depth == 0
    assert root.score == 1.0
    assert root.branch_type == "root"
    assert len(engine.nodes) == 1


def test_got_tree_expansion_and_search():
    engine = got.GraphOfThoughts(max_depth=3, beam_width=3, prune_threshold=0.4)
    res = engine.search(query="Formulate optimal path for multi-step reasoning")

    assert isinstance(res, got.GoTSearchResult)
    assert len(res.best_path_nodes) >= 2
    assert res.receipt.schema_version == "nexus-got-receipt-v1"
    assert res.receipt.total_nodes_generated > 0
    assert len(res.receipt.query_digest) == 64
    assert len(res.receipt.best_path_digest) == 64
    assert res.receipt.optimal_path_score > 0.0
    assert res.receipt.authority_bits["has_answer_authority"] is False
    assert res.receipt.score_semantics == "template_position_priority_not_correctness_or_optimality"
    assert "no answer was generated or verified" in res.final_output.lower()
    assert res.telemetry["answer_verified"] is False
    assert res.to_dict()["answer_authority"] is False


def test_got_prune_and_merge():
    engine = got.GraphOfThoughts(max_depth=3, beam_width=4, prune_threshold=0.5)
    root = engine.add_node("Root problem", depth=0, score=1.0)
    c1 = engine.add_node("Good path A", parent_id=root.node_id, depth=1, score=0.85)
    c2 = engine.add_node("Good path B", parent_id=root.node_id, depth=1, score=0.80)
    c3 = engine.add_node("Bad path C", parent_id=root.node_id, depth=1, score=0.30)

    pruned = engine.prune_unviable()
    assert pruned == 1
    assert c3.is_pruned is True

    merged = engine.merge_complementary([c1, c2])
    assert merged is not None
    assert merged.branch_type == "merged"
    assert merged.score >= 0.80
