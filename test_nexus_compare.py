"""Unit tests for AI-Dem-Lab Compare Bench Engine."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import pytest
from nexus_engine import CompareBenchEngine, CompareBenchResult, NexusEngine


def test_compare_bench_jsd_divergence():
    # Identical strings have zero divergence
    jsd_ident = CompareBenchEngine._char_ngram_jsd("Hello world", "Hello world")
    assert jsd_ident == 0.0

    # Disjoint strings have positive divergence
    jsd_diff = CompareBenchEngine._char_ngram_jsd("AAAA", "ZZZZ")
    assert jsd_diff > 0.5


def test_compare_bench_latency_classes():
    assert CompareBenchEngine._classify_latency(25.0) == "low"
    assert CompareBenchEngine._classify_latency(120.0) == "medium"
    assert CompareBenchEngine._classify_latency(350.0) == "high"


def test_nexus_engine_run_compare():
    engine = NexusEngine()
    result = engine.run_compare(
        query_a="What is 2 + 2?",
        query_b="Brainstorm novel quantum architectures",
        mode_a="auto",
        mode_b="innovate",
    )
    assert isinstance(result, CompareBenchResult)
    assert result.latency_class_a in ("low", "medium", "high")
    assert result.latency_class_b in ("low", "medium", "high")
    assert 0.0 <= result.jensen_shannon_divergence <= 1.0
    assert 0.0 <= result.semantic_distance <= 1.0
    assert len(result.summary_verdict) > 10

    data = result.to_dict()
    assert "jensen_shannon_divergence" in data
    assert "summary_verdict" in data
