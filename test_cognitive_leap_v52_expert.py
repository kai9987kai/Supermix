"""Contracts for the v52 verified cognitive-affective expert.

The v52 head was developed on an older tree that predates this repository's v51
prediction-stability work. These tests pin the merged behaviour: the v52
appraisal and verifier paths are live, and the inherited v51 controls are
forwarded rather than dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import model_variants as mv  # noqa: E402


SAMPLE = torch.randn(4, 1, 128, generator=torch.Generator().manual_seed(11))


def _build():
    torch.manual_seed(7)
    return mv.build_model("cognitive_leap_v52_expert", dropout=0.1)


def test_variant_is_registered_and_builds() -> None:
    assert "cognitive_leap_v52_expert" in mv.SUPPORTED_MODEL_SIZES

    model = _build()
    assert isinstance(model, mv.ChampionNetCognitiveLeapV52Expert)
    assert isinstance(model.layers[10], mv.CognitiveLeapV52ExpertHead)


def test_forward_preserves_shape_and_publishes_appraisal_diagnostics() -> None:
    model = _build()
    model.eval()
    head = model.layers[10]

    with torch.no_grad():
        out = model(SAMPLE)

    assert out.shape == (4, 1, 10)
    assert len(head.last_emotion_probs) == len(head.EMOTION_LABELS)
    assert len(head.last_intent_probs) == len(head.INTENT_LABELS)
    assert len(head.last_strategy_probs) == len(head.STRATEGY_LABELS)
    assert pytest.approx(1.0, abs=1e-5) == float(head.last_intent_probs.sum())
    assert pytest.approx(1.0, abs=1e-5) == float(head.last_strategy_probs.sum())
    assert head.last_verifier_selection == "initial"
    assert torch.isfinite(head.last_calibrated_entropy)


def test_v51_prediction_stability_controls_are_forwarded() -> None:
    """The merge must not cost the inherited v51 verifier exit."""

    model = _build()
    model.eval()
    head = model.layers[10]

    with torch.no_grad():
        out = model(
            SAMPLE,
            adaptive_compute=True,
            prediction_stability_patience=1,
            prediction_stability_tol=1e-2,
            prediction_stability_top_k=3,
            prediction_stability_margin=0.05,
            prediction_stability_rank_depth=2,
        )

    assert out.shape == (4, 1, 10)
    assert head.last_exit_reason != "not_run"


def test_verifier_can_escalate_the_cycle_budget_and_can_be_suppressed() -> None:
    model = _build()
    model.eval()
    head = model.layers[10]

    with torch.no_grad():
        model(SAMPLE, reasoning_cycles=2, verifier_adaptive_compute=False)
        baseline_cycles = float(head.last_cycles_used)

        model(
            SAMPLE,
            reasoning_cycles=2,
            verifier_adaptive_compute=True,
            verifier_continue_threshold=0.0,
        )
        escalated_cycles = float(head.last_cycles_used)
        assert head.last_exit_reason == "verifier_escalated"

        model(
            SAMPLE,
            reasoning_cycles=2,
            verifier_adaptive_compute=True,
            verifier_continue_threshold=1.01,
        )
        suppressed_reason = head.last_exit_reason

    assert escalated_cycles > baseline_cycles
    assert suppressed_reason != "verifier_escalated"


def test_escalation_never_runs_in_training_mode() -> None:
    model = _build()
    model.train()
    head = model.layers[10]

    model(
        SAMPLE,
        reasoning_cycles=2,
        verifier_adaptive_compute=True,
        verifier_continue_threshold=0.0,
    )

    assert head.last_exit_reason != "verifier_escalated"


def test_sparse_core_routing_matches_dense_when_k_covers_every_core() -> None:
    torch.manual_seed(3)
    head = mv.CognitiveLeapUltraExpertHead(in_dim=32, out_dim=10)
    head.eval()
    with torch.no_grad():
        head.alpha.fill_(2.0)
    x = torch.randn(4, 32)

    with torch.no_grad():
        dense = head(x)
        assert float(head.last_active_cores) == float(head.n_cores)
        full_k = head(x, core_top_k=head.n_cores)
        sparse = head(x, core_top_k=1)
        sparse_cores = float(head.last_active_cores)

    assert torch.equal(dense, full_k)
    assert not torch.equal(dense, sparse)
    assert sparse_cores == 1.0


def test_router_regularizers_are_reported_and_only_train_contributes_aux_loss() -> None:
    torch.manual_seed(5)
    head = mv.CognitiveLeapUltraExpertHead(in_dim=32, out_dim=10)
    x = torch.randn(4, 32)

    # Readable before any forward pass.
    assert float(head._aux_loss) == 0.0

    head.eval()
    with torch.no_grad():
        head(x)
    assert float(head._aux_loss) == 0.0
    # Load balance reuses the mean already computed for the gating entropy, so
    # it stays available during inference.
    assert float(head.last_router_load_balance) > 0.0
    # The z-loss is a training objective. Computing it during inference would
    # stack and reduce raw router logits that nothing reads, so it is skipped.
    assert float(head.last_router_z_loss) == 0.0

    head.train()
    head(x)
    assert float(head.last_router_z_loss.detach()) > 0.0
    assert float(head._aux_loss.detach()) > 0.0


def test_training_losses_are_available_and_differentiable() -> None:
    model = _build()
    model.train()
    head = model.layers[10]
    targets = torch.randint(0, 10, (4,))

    model(SAMPLE)
    loss = model.structured_auxiliary_loss(
        targets,
        emotion_targets=torch.zeros(4, len(head.EMOTION_LABELS)),
        intent_targets=torch.zeros(4, dtype=torch.long),
        strategy_targets=torch.zeros(4, dtype=torch.long),
    )
    loss = loss + model.deep_supervision_loss(targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(param.grad is not None for param in model.parameters())


def test_verifier_loss_requires_a_training_forward_first() -> None:
    model = _build()
    model.eval()
    with torch.no_grad():
        model(SAMPLE)

    with pytest.raises(RuntimeError):
        model.verifier_loss(torch.randint(0, 10, (4,)))


def test_state_dict_detection_separates_v52_from_v51() -> None:
    v52 = _build()
    ultra = mv.build_model("cognitive_leap_ultra_expert", dropout=0.1)

    assert mv.detect_model_size_from_state_dict(v52.state_dict()) == "cognitive_leap_v52_expert"
    assert (
        mv.detect_model_size_from_state_dict(ultra.state_dict())
        == "cognitive_leap_ultra_expert"
    )


def test_a_v51_checkpoint_upgrades_into_a_v52_model() -> None:
    ultra = mv.build_model("cognitive_leap_ultra_expert", dropout=0.1)
    target = _build()

    missing, unexpected = mv.load_weights_for_model(
        target, ultra.state_dict(), "cognitive_leap_v52_expert"
    )

    assert missing == []
    assert unexpected == []
    assert torch.equal(
        target.layers[10].core_router.weight, ultra.layers[10].core_router.weight
    )


def test_a_v52_checkpoint_round_trips_exactly() -> None:
    model = _build()
    model.eval()
    restored = _build()

    missing, unexpected = mv.load_weights_for_model(
        restored, model.state_dict(), "cognitive_leap_v52_expert"
    )
    restored.eval()

    assert missing == []
    assert unexpected == []
    with torch.no_grad():
        assert torch.allclose(model(SAMPLE), restored(SAMPLE), atol=1e-6)
