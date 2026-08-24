"""Tests for the NexusMind Unified Engine."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import nexus_engine as engine


def test_engine_initialization():
    eng = engine.build_default_engine()
    assert eng.model is not None
    assert eng.swarm_engine is not None
    assert eng.got_engine is not None
    assert eng.observatory is not None
    assert eng.q_learner is not None


def test_engine_flash_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="What is sliding window attention?", mode="fast")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "fast"
    assert len(res.thought_steps) >= 1
    assert res.confidence > 0.0
    assert "dem_lab_entropy" in res.telemetry


def test_engine_deep_mode_recursive_ponder():
    eng = engine.build_default_engine()
    res = eng.process(query="Deduce hierarchical routing policy", mode="deep")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "deep"
    assert any(s.stage == "ponder" for s in res.thought_steps)


def test_engine_swarm_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="Debate memory authority boundaries", mode="swarm")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "swarm"
    assert "swarm_receipt" in res.audit_receipts
    assert any(s.stage == "swarm_debate" for s in res.thought_steps)


def test_engine_got_mode():
    eng = engine.build_default_engine()
    res = eng.process(query="Explore decision tree for routing", mode="got")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "got"
    assert "got_receipt" in res.audit_receipts


def test_engine_scientific_mode_deterministic_execution():
    eng = engine.build_default_engine()
    q = "Assuming an ideal gas, a sample contains 2 mol, has volume 50 L, and temperature is 300 K. What is its pressure?"
    res = eng.process(query=q, mode="auto")

    assert isinstance(res, engine.NexusResult)
    assert res.mode_selected == "scientific"
    assert res.confidence == 1.0
    assert "scientific_receipt" in res.audit_receipts
    assert "Pa" in str(res.audit_receipts) or "Pa" in res.final_output
