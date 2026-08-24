"""Tests for the v59 mechanism causality audit.

The audit's job is to say "this mechanism does nothing". A tool that says that
about everything is worthless, so most of these tests are adversarial: they
corrupt the instrument or plant a mechanism that genuinely matters and require
the audit to notice. In particular :func:`test_live_thinking_core_is_reported_active`
is the positive control -- without it, "inert" would be unfalsifiable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import mechanism_causality as causality  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward  # noqa: E402
from train_mimomix_talk import evaluate as talk_evaluate  # noqa: E402


SEQUENCE_LENGTH = 32


def _tiny_model(seed: int = 59, **overrides) -> MiMoMixModel:
    torch.manual_seed(seed)
    settings = dict(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=64,
        moe_intermediate_size=16,
        n_routed_experts=4,
        moe_top_k=2,
        n_mtp_layers=1,
        native_context=SEQUENCE_LENGTH,
        max_position_embeddings=SEQUENCE_LENGTH,
        rope_scaling="none",
        sliding_window=16,
    )
    settings.update(overrides)
    model = MiMoMixModel(MiMoMixConfig(**settings))
    model.eval()
    return model


def _tiny_batch(vocab_size: int = 64, blocks: int = 6):
    generator = torch.Generator().manual_seed(11)
    inputs = torch.randint(0, vocab_size, (blocks, SEQUENCE_LENGTH), generator=generator)
    labels = inputs.clone()
    labels[:, : SEQUENCE_LENGTH // 4] = -100  # some positions unsupervised, as in real packing
    return inputs, labels


# -- scoring ---------------------------------------------------------------


def test_score_matches_the_repo_s_own_evaluate():
    """The audit must measure the same quantity the trainer publishes.

    If this drifts, every delta is against a private metric and the receipt
    cannot be compared with any tier loss in the repo.
    """

    model = _tiny_model()
    inputs, labels = _tiny_batch()

    mine = causality.score(model, inputs, labels, batch_size=2)
    theirs = talk_evaluate(model, inputs, labels, batch_size=2)

    assert mine.loss == pytest.approx(theirs["loss"], abs=1e-6)


def test_score_is_invariant_to_batch_size_up_to_float_noise():
    model = _tiny_model()
    inputs, labels = _tiny_batch()

    coarse = causality.score(model, inputs, labels, batch_size=6).loss
    fine = causality.score(model, inputs, labels, batch_size=1).loss

    assert coarse == pytest.approx(fine, abs=1e-6)


def test_numerical_floor_is_small_and_non_negative():
    model = _tiny_model()
    inputs, labels = _tiny_batch()

    floor = causality.numerical_noise_floor(model, inputs, labels, batch_sizes=(1, 2, 3))

    assert floor["floor_nats"] >= 0.0
    assert floor["floor_nats"] < 1e-4


def test_score_rejects_a_fully_masked_batch():
    model = _tiny_model()
    inputs, labels = _tiny_batch()
    labels = torch.full_like(labels, -100)

    with pytest.raises(ValueError):
        causality.score(model, inputs, labels, batch_size=2)


# -- interventions restore -------------------------------------------------


@pytest.mark.parametrize("intervention", causality.INTERVENTIONS, ids=lambda i: i.name)
def test_every_intervention_restores_the_model(intervention):
    model = _tiny_model()
    inputs, labels = _tiny_batch()
    if not intervention.available(model):
        pytest.skip(f"{intervention.name} not present on the tiny model")

    before = causality.score(model, inputs, labels, batch_size=2).loss
    with causality.applied(model, intervention):
        pass
    after = causality.score(model, inputs, labels, batch_size=2).loss

    assert after == before


def test_routing_rebuild_is_faithful():
    """The identity rebuild must reproduce the real forward bit-exactly."""

    model = _tiny_model()
    inputs, labels = _tiny_batch()

    baseline = causality.score(model, inputs, labels, batch_size=2)
    with causality.applied(model, causality.IDENTITY):
        rebuilt = causality.score(model, inputs, labels, batch_size=2)

    assert rebuilt.loss == baseline.loss
    assert rebuilt.agreement_with(baseline) == 1.0


def test_self_check_raises_when_the_rebuild_is_unfaithful(monkeypatch):
    """A verifier that cannot fail is not a verifier.

    Corrupt the identity rebuild so it perturbs the output, and require the
    harness to refuse to report rather than attribute the bug to a mechanism.
    """

    model = _tiny_model()
    inputs, labels = _tiny_batch()
    baseline = causality.score(model, inputs, labels, batch_size=2)

    honest = causality._routing_forward

    def sabotaged(mode: str):
        inner = honest(mode)

        def forward(self: SparseMoEFeedForward, x: torch.Tensor) -> torch.Tensor:
            return inner(self, x) * 1.05  # a 5% scaling no real intervention would apply

        return forward

    monkeypatch.setattr(causality, "_routing_forward", sabotaged)
    monkeypatch.setattr(
        causality, "IDENTITY", causality.Intervention(
            name="identity_routing_rebuild",
            description="sabotaged",
            apply=causality._patch_routing("identity"),
            requires=lambda m: True,
        )
    )

    with pytest.raises(AssertionError, match="not faithful"):
        causality.self_check(model, inputs, labels, baseline, batch_size=2)


# -- verdicts --------------------------------------------------------------


def test_live_thinking_core_is_reported_active():
    """Positive control: a thinking core that actually contributes is caught.

    v58's core is inert because its gate never left zero. Open the gate and the
    same audit must report ``active`` -- otherwise 'inert' would be a property
    of the instrument rather than of the mechanism.

    This model is randomly initialised, so its argmax is degenerate and no
    perturbation changes a decision. The open gate still moves held-out loss by
    ~2.4e-03 nats, four times the smallest effect v58 published as a finding,
    and the verdict has to catch it on that axis alone.
    """

    model = _tiny_model()
    inputs, labels = _tiny_batch()
    assert model.thinking_core is not None
    model.thinking_core.residual_scale.data.fill_(0.5)

    report = causality.audit(model, inputs, labels, batch_size=2)
    row = next(r for r in report["mechanisms"] if r["mechanism"] == "thinking_core")

    assert row["verdict"] == causality.ACTIVE
    assert row["abs_delta_nats"] >= causality.RELEVANCE_SCALE


def test_closed_gate_thinking_core_is_reported_inert():
    """The v58 condition: the gate at its initial value contributes nothing."""

    model = _tiny_model()
    inputs, labels = _tiny_batch()
    model.thinking_core.residual_scale.data.zero_()

    report = causality.audit(model, inputs, labels, batch_size=2)
    row = next(r for r in report["mechanisms"] if r["mechanism"] == "thinking_core")

    assert row["verdict"] == causality.INERT
    assert row["decisions_changed"] == 0
    assert row["abs_delta_nats"] == 0.0


def test_routing_is_reported_active_when_experts_actually_differ():
    """Routing can only matter if the experts do.

    On a randomly initialised model the experts are interchangeable, so
    destroying the assignment costs ~8e-05 nats and the audit correctly calls it
    inert. Give each expert a distinct scale and the same audit must flip to
    ``active`` -- the verdict tracks the model, not the mechanism's name.
    """

    model = _tiny_model()
    inputs, labels = _tiny_batch()
    for layer in causality._moe_layers(model):
        for index, expert in enumerate(layer.experts):
            for parameter in expert.parameters():
                parameter.data.mul_(1.0 + 2.0 * index)

    report = causality.audit(model, inputs, labels, batch_size=2)
    row = next(r for r in report["mechanisms"] if r["mechanism"] == "moe_routing_random")

    assert row["verdict"] == causality.ACTIVE


def test_routing_is_inert_when_experts_are_interchangeable():
    """The complementary honest result, on the untouched random model."""

    model = _tiny_model()
    inputs, labels = _tiny_batch()

    report = causality.audit(model, inputs, labels, batch_size=2)
    row = next(r for r in report["mechanisms"] if r["mechanism"] == "moe_routing_random")

    assert row["verdict"] == causality.INERT


def test_audit_raises_if_an_intervention_leaks():
    """Restoration is only credible if it is checked."""

    model = _tiny_model()
    inputs, labels = _tiny_batch()

    def leaky(m):
        m.thinking_core.residual_scale.data.fill_(0.75)
        return lambda: None  # deliberately fails to restore

    leaking = causality.Intervention(
        name="leaky",
        description="does not restore",
        apply=leaky,
        requires=lambda m: True,
    )

    with pytest.raises(AssertionError, match="not restored"):
        causality.audit(model, inputs, labels, batch_size=2, interventions=(leaking,))


def test_absent_mechanism_is_reported_not_silently_skipped():
    model = _tiny_model(use_thinking_core=False)
    inputs, labels = _tiny_batch()

    report = causality.audit(model, inputs, labels, batch_size=2)
    row = next(r for r in report["mechanisms"] if r["mechanism"] == "thinking_core")

    assert row["available"] is False
    assert "reason" in row


def test_receipt_carries_schema_and_non_claims():
    model = _tiny_model()
    inputs, labels = _tiny_batch()

    report = causality.audit(model, inputs, labels, batch_size=2)

    assert report["schema"] == causality.RECEIPT_SCHEMA
    assert len(report["non_claims"]) >= 4
    assert report["self_check"]["ran"] is True
