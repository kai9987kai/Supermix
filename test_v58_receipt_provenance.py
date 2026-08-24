"""Hold the v58 documents to the standard v58 was written to enforce.

`test_v56_receipt_provenance.py` exists because the v56 documents quoted one
checkpoint's numbers next to another checkpoint's reproduction commands, and
nothing compared prose to provenance. Adding a measurement line while leaving its
own claims unguarded would repeat exactly that.

So every number the v58 document and README state is checked against the receipt
it came from, read out of the JSON rather than hard-coded here. A re-measured
model updates these tests by being re-measured; a doc that drifts from its
receipt fails them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"
V58_DOC = ROOT / "docs" / "V58_GENERALISATION_LADDER.md"
FULL = ROOT / "output" / "v58_full" / "generalisation_results.json"
ABLATION = ROOT / "output" / "v58_ablation" / "generalisation_results.json"
TOKENS = ROOT / "output" / "v58_unseen_sentence_tokens.json"

pytestmark = pytest.mark.skipif(not FULL.exists(), reason="v58 receipts not present")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load(path: Path) -> dict:
    return json.loads(read(path))


@pytest.fixture(scope="module")
def documents() -> str:
    return read(V58_DOC) + "\n" + read(README)


def test_every_tier_perplexity_appears_in_the_documents(documents: str):
    receipt = load(FULL)
    for name, entry in receipt["tiers"].items():
        rendered = f"{entry['perplexity']:.4f}"
        assert rendered in documents, f"{name} perplexity {rendered} is in no document"


def test_every_tier_row_count_appears_in_the_documents(documents: str):
    receipt = load(FULL)
    for name, entry in receipt["tiers"].items():
        assert f"{entry['pairs']:,}" in documents, f"{name} row count is in no document"


def test_the_documents_state_the_ladder_is_monotonic_only_if_it_is():
    """The headline reading. If a re-run inverted a tier, the prose would be
    wrong and this catches it rather than leaving it to a reader."""

    receipt = load(FULL)
    losses = [receipt["tiers"][name]["loss"] for name in
              ("tier1_seen_response", "tier2_unseen_response", "tier3_unseen_sentence")]
    monotonic = losses[0] < losses[1] < losses[2]
    claimed = "monotonic" in read(V58_DOC) or "monotonic" in read(README)
    assert monotonic == claimed, (
        f"documents claim monotonic={claimed} but measured losses are {losses}"
    )


def test_the_vocabulary_coverage_claim_matches_every_tier():
    """Tier 3 is only a composition test if coverage is 1.0. The documents say
    1.0000 in both files; it has to be true of every tier in the receipt."""

    receipt = load(FULL)
    for name, entry in receipt["split"]["tiers"].items():
        assert entry["response_vocabulary_coverage"] == 1.0, name


def test_selection_never_read_a_tier():
    receipt = load(FULL)
    assert receipt["selection"]["selected_on"] == "dev"
    assert receipt["checks"]["selection_never_read_a_tier"] is True


@pytest.mark.skipif(not TOKENS.exists(), reason="token-level receipt not present")
def test_the_token_level_numbers_appear_in_the_documents(documents: str):
    runs = {row["run_name"]: row for row in load(TOKENS)["runs"]}
    full = runs["v58_full"]
    for key in ("seen_sentence_tokens", "unseen_sentence_tokens"):
        rendered = f"{full['measured'][key]['perplexity']:.4f}"
        assert rendered in documents, f"{key} perplexity {rendered} is in no document"
    assert f"{full['unseen_minus_seen_nats']:.4f}"[1:] in documents.replace("+", "")


@pytest.mark.skipif(not TOKENS.exists(), reason="token-level receipt not present")
def test_the_token_level_gap_is_larger_than_the_diluted_tier_gap():
    """The documents' central argument -- that averaging over a tier-3 row hides
    the effect. If a re-run reversed it, the prose would be backwards."""

    tokens = {row["run_name"]: row for row in load(TOKENS)["runs"]}["v58_full"]
    diluted = load(FULL)["gaps"]["total_cost_nats"]
    assert tokens["unseen_minus_seen_nats"] > diluted, (
        f"token-level gap {tokens['unseen_minus_seen_nats']} is not larger than the "
        f"diluted tier gap {diluted}, but the documents say it is"
    )


@pytest.mark.skipif(not ABLATION.exists(), reason="ablation receipt not present")
def test_the_two_ablation_arms_are_actually_matched():
    """The comparison means nothing unless the arms differ in one field."""

    full, ablation = load(FULL), load(ABLATION)
    assert full["hyperparameters"] == ablation["hyperparameters"]
    assert sorted(full["held_out_sentences"]) == sorted(ablation["held_out_sentences"])
    assert full["thinking_core"] is True and ablation["thinking_core"] is False


@pytest.mark.skipif(not ABLATION.exists(), reason="ablation receipt not present")
def test_the_no_measurable_effect_claim_holds_against_the_receipts():
    """Two things the documents assert about the ablation: every tier delta is
    tiny, and the sign is inconsistent. Both are why the claim is 'no measurable
    effect' rather than 'the core helps' or 'the core hurts'."""

    full, ablation = load(FULL), load(ABLATION)
    deltas = [
        ablation["tiers"][name]["loss"] - full["tiers"][name]["loss"]
        for name in full["tiers"]
    ]
    assert all(abs(delta) < 0.01 for delta in deltas), deltas
    assert any(d > 0 for d in deltas) and any(d < 0 for d in deltas), (
        f"documents claim the sign flips across tiers, but deltas are {deltas}"
    )
