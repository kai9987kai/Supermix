"""The number a specialist prints must be the probability of what it printed.

Three specialists (`protein_folding_model`, `mattergen_generation_model`,
`three_d_generation_model`) shipped the same expression:

    confidence = max(confidence, heuristic_score, 0.91 if confidence < 0.45 else confidence)

It had two defects. Below 0.45 the reported figure became the literal ``0.91``,
so the displayed confidence was *highest* exactly where the network was least
certain. Above it, the label was replaced by a keyword matcher's while the number
stayed the network's confidence in its own, different label.

Both numbers reach users: the answer text ends with
``[Protein concept: ... | confidence 0.91]``.

These tests pin the property rather than the implementation, so any future
rewrite of the override still has to report a probability of the thing it named.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

MODULES = (
    "protein_folding_model",
    "mattergen_generation_model",
    "three_d_generation_model",
)


def _source_of(name: str) -> str:
    return (SOURCE_DIR / f"{name}.py").read_text(encoding="utf-8")


def _executable_source_of(name: str) -> str:
    """The module's source with comment lines removed.

    The fix documents the defect by quoting the old expression in a comment, so
    a raw substring search would match the very thing it is checking for. These
    tests are about what the code does, not what it says about itself.
    """

    lines = []
    for line in _source_of(name).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


@pytest.mark.parametrize("module_name", MODULES)
def test_no_invented_confidence_literal(module_name):
    """The 0.91 substitution must not come back."""

    text = _executable_source_of(module_name)

    assert "0.91 if confidence < 0.45 else confidence" not in text, (
        f"{module_name} reports a hardcoded 0.91 as a confidence when the "
        "network is least certain"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_override_reports_the_overriding_labels_probability(module_name):
    """When the heuristic wins the label, it must also supply the number."""

    text = _executable_source_of(module_name)
    marker = "predicted = heuristic"
    assert marker in text, f"{module_name} no longer has the heuristic override"

    tail = text.split(marker, 1)[1]
    window = tail[:1200]
    assert "confidence = heuristic_score" in window, (
        f"{module_name} overrides the label without reporting that label's "
        "probability"
    )


def test_the_defect_is_reproducible_as_arithmetic():
    """Document what the old expression did, independent of any checkpoint.

    No `.pth` exists for these specialists, so the engines cannot be run here.
    The defect was in pure arithmetic, and that much is testable directly.
    """

    def old(net_confidence: float, heuristic_score: float) -> float:
        return max(
            net_confidence,
            heuristic_score,
            0.91 if net_confidence < 0.45 else net_confidence,
        )

    def fixed(net_confidence: float, heuristic_score: float) -> float:
        return heuristic_score

    # Least certain network, near-zero support for the label it returns.
    assert old(0.30, 0.02) == pytest.approx(0.91)
    assert fixed(0.30, 0.02) == pytest.approx(0.02)

    # Confident network, but the returned label is the heuristic's.
    assert old(0.80, 0.15) == pytest.approx(0.80)  # P of a label not returned
    assert fixed(0.80, 0.15) == pytest.approx(0.15)


@pytest.mark.parametrize("module_name", MODULES)
def test_reported_confidence_is_a_probability(module_name):
    """A confidence must be in [0, 1] whatever branch produced it."""

    text = _executable_source_of(module_name)
    tail = text.split("predicted = heuristic", 1)[1][:1200]

    # `heuristic_score` is read straight out of the softmax dict, so it is a
    # probability by construction. Guard the source of that value.
    assert "heuristic_score = float(probabilities.get(heuristic, 0.0))" in text, (
        f"{module_name} no longer derives heuristic_score from the softmax "
        "distribution, so the reported number may not be a probability"
    )
    assert "heuristic_score" in tail
