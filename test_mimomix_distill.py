"""Tests for MOPD and the domain-RL stage.

The properties that matter: the teacher mixture is a real mixture (normalised,
in probability space), reverse KL is zero exactly when the student matches,
confidence weighting actually picks the right specialist, teachers stay frozen,
and a training step moves the student toward the teachers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_core as mc  # noqa: E402
import mimomix_distill as dl  # noqa: E402


VOCAB = 12


class StubTeacher(nn.Module):
    """Returns fixed logits, so the mixture maths is checkable by hand."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed", logits)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        return self.fixed.view(1, 1, -1).expand(b, t, -1).clone()


class SequenceTeacher(nn.Module):
    """Predict either the causal next token or the already-observed token."""

    def __init__(self, predicts_next: bool):
        super().__init__()
        self.predicts_next = bool(predicts_next)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        targets = torch.roll(input_ids, shifts=-1, dims=1) if self.predicts_next else input_ids
        logits = torch.full((*input_ids.shape, VOCAB), -12.0, device=input_ids.device)
        return logits.scatter(-1, targets.unsqueeze(-1), 12.0)


def build(seed: int = 0, **overrides) -> mc.MiMoMixModel:
    torch.manual_seed(seed)
    base = dict(
        vocab_size=VOCAB,
        hidden_size=32,
        n_layers=3,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        sliding_window=8,
        hybrid_ratio=2,
        native_context=32,
        max_position_embeddings=64,
        n_routed_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        n_mtp_layers=1,
        thinking_cycles=1,
        rope_scaling="none",
    )
    base.update(overrides)
    return mc.MiMoMixModel(mc.MiMoMixConfig(**base))


def one_hot_logits(index: int, sharpness: float = 12.0) -> torch.Tensor:
    logits = torch.zeros(VOCAB)
    logits[index] = sharpness
    return logits


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_validates_its_options():
    with pytest.raises(ValueError):
        dl.MOPDConfig(weighting="telepathy")
    with pytest.raises(ValueError):
        dl.MOPDConfig(temperature=0.0)
    with pytest.raises(ValueError):
        dl.MOPDConfig(min_teacher_weight=1.0)
    with pytest.raises(ValueError):
        dl.MOPDConfig(teacher_top_k=-1)
    with pytest.raises(ValueError):
        dl.MOPDConfig(teacher_probability_floor=0.0)
    with pytest.raises(ValueError):
        dl.MOPDConfig(max_position_kl=0.0)


# ---------------------------------------------------------------------------
# Teacher mixture
# ---------------------------------------------------------------------------


def test_mixture_is_a_normalised_distribution():
    teachers = [torch.randn(2, 4, VOCAB), torch.randn(2, 4, VOCAB)]
    tokens = torch.randint(0, VOCAB, (2, 4))
    mixture, weights = dl.teacher_mixture_log_probs(
        teachers, tokens, dl.MOPDConfig(weighting="uniform")
    )
    assert torch.allclose(mixture.exp().sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5)
    assert weights.shape == (2, 4, 2)


def test_uniform_mixture_of_two_point_masses_splits_the_mass():
    """A mixture in probability space, not a geometric mean of logits."""

    teachers = [
        one_hot_logits(0, 40.0).view(1, 1, -1),
        one_hot_logits(1, 40.0).view(1, 1, -1),
    ]
    tokens = torch.zeros(1, 1, dtype=torch.long)
    mixture, _ = dl.teacher_mixture_log_probs(
        teachers, tokens, dl.MOPDConfig(weighting="uniform", min_teacher_weight=0.0)
    )
    probs = mixture.exp()[0, 0]
    assert probs[0] == pytest.approx(0.5, abs=1e-3)
    assert probs[1] == pytest.approx(0.5, abs=1e-3)
    # A logit average would have produced a tie at ~0.5 each too, so also check
    # a token neither teacher likes stays near zero.
    assert probs[5] < 1e-6


def test_confidence_weighting_backs_the_teacher_that_predicted_the_token():
    teachers = [one_hot_logits(3).view(1, 1, -1), one_hot_logits(7).view(1, 1, -1)]
    tokens = torch.tensor([[3]])
    _, weights = dl.teacher_mixture_log_probs(
        teachers, tokens, dl.MOPDConfig(weighting="confidence", min_teacher_weight=0.0)
    )
    assert weights[0, 0, 0] > 0.99
    assert weights[0, 0, 1] < 0.01


def test_min_teacher_weight_keeps_a_specialist_alive():
    teachers = [one_hot_logits(3).view(1, 1, -1), one_hot_logits(7).view(1, 1, -1)]
    tokens = torch.tensor([[3]])
    _, weights = dl.teacher_mixture_log_probs(
        teachers, tokens, dl.MOPDConfig(weighting="confidence", min_teacher_weight=0.1)
    )
    assert weights[0, 0, 1] >= 0.09
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1), atol=1e-5)


def test_domain_weighting_uses_the_supplied_weights():
    teachers = [one_hot_logits(3).view(1, 1, -1), one_hot_logits(7).view(1, 1, -1)]
    tokens = torch.tensor([[0]])
    mixture, weights = dl.teacher_mixture_log_probs(
        teachers, tokens,
        dl.MOPDConfig(weighting="domain", min_teacher_weight=0.0),
        domain_weights=torch.tensor([0.75, 0.25]),
    )
    assert weights[0, 0, 0] == pytest.approx(0.75, abs=1e-4)
    assert mixture.exp()[0, 0, 3] > mixture.exp()[0, 0, 7]

    with pytest.raises(ValueError):
        dl.teacher_mixture_log_probs(teachers, tokens, dl.MOPDConfig(weighting="domain"))


def test_teacher_top_k_truncation_keeps_only_a_tiny_finite_tail():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0] + [0.0] * (VOCAB - 4)).view(1, 1, -1)
    mixture, _ = dl.teacher_mixture_log_probs(
        [logits], torch.zeros(1, 1, dtype=torch.long), dl.MOPDConfig(teacher_top_k=2)
    )
    probs = mixture.exp()[0, 0]
    assert probs[0] + probs[1] == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < probs[2] < 1e-7


def test_mixture_requires_a_teacher():
    with pytest.raises(ValueError):
        dl.teacher_mixture_log_probs([], torch.zeros(1, 1, dtype=torch.long))


# ---------------------------------------------------------------------------
# Distillation loss
# ---------------------------------------------------------------------------


def test_reverse_kl_is_zero_when_the_student_already_matches():
    logits = torch.randn(2, 5, VOCAB)
    tokens = torch.randint(0, VOCAB, (2, 5))
    result = dl.on_policy_distillation_loss(logits.clone(), [logits.clone()], tokens)
    assert float(result.loss) == pytest.approx(0.0, abs=1e-5)
    assert float(result.dense_reward.max()) == pytest.approx(0.0, abs=1e-5)


def test_reverse_kl_is_positive_and_differentiable():
    student = torch.randn(2, 5, VOCAB, requires_grad=True)
    teacher = torch.randn(2, 5, VOCAB)
    result = dl.on_policy_distillation_loss(student, [teacher], torch.randint(0, VOCAB, (2, 5)))
    assert float(result.loss.detach()) > 0.0
    result.loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0


def test_top_k_reverse_kl_is_finite_when_sample_lies_outside_teacher_support():
    student = torch.randn(1, 2, VOCAB, requires_grad=True)
    teacher = torch.full((1, 2, VOCAB), -20.0)
    teacher[..., 3] = 10.0
    teacher[..., 7] = 9.0
    sampled = torch.zeros(1, 2, dtype=torch.long)  # outside the teacher's top-2

    result = dl.on_policy_distillation_loss(
        student,
        [teacher],
        sampled,
        config=dl.MOPDConfig(teacher_top_k=2),
    )
    assert bool(torch.isfinite(result.loss))
    result.loss.backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())
    assert float(student.grad.abs().sum()) > 0.0


def test_dense_reward_is_per_position():
    student = torch.randn(3, 6, VOCAB)
    teacher = torch.randn(3, 6, VOCAB)
    result = dl.on_policy_distillation_loss(student, [teacher], torch.randint(0, VOCAB, (3, 6)))
    assert result.dense_reward.shape == (3, 6)
    assert bool((result.dense_reward <= 0).all()), "reward is -KL, so never positive"


def test_position_mask_excludes_the_prompt_prefix():
    student = torch.randn(1, 6, VOCAB)
    teacher = student.clone()
    teacher[:, :3] = torch.randn(1, 3, VOCAB) * 10  # garbage in the masked region
    mask = torch.zeros(1, 6, dtype=torch.bool)
    mask[:, 3:] = True
    masked = dl.on_policy_distillation_loss(student, [teacher], torch.zeros(1, 6, dtype=torch.long),
                                            position_mask=mask)
    unmasked = dl.on_policy_distillation_loss(student, [teacher], torch.zeros(1, 6, dtype=torch.long))
    assert float(masked.loss) == pytest.approx(0.0, abs=1e-5)
    assert float(unmasked.loss) > float(masked.loss)


def test_position_kl_is_clipped_and_counted():
    student = one_hot_logits(0, 60.0).view(1, 1, -1)
    teacher = one_hot_logits(1, 60.0).view(1, 1, -1)
    result = dl.on_policy_distillation_loss(
        student, [teacher], torch.zeros(1, 1, dtype=torch.long),
        config=dl.MOPDConfig(max_position_kl=2.0),
    )
    assert float(result.loss) == pytest.approx(2.0, abs=1e-5)
    assert result.clipped_positions == 1


def test_result_is_json_safe():
    import json

    student = torch.randn(1, 3, VOCAB)
    result = dl.on_policy_distillation_loss(student, [torch.randn(1, 3, VOCAB)],
                                            torch.zeros(1, 3, dtype=torch.long))
    json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# GRPO / domain RL stage
# ---------------------------------------------------------------------------


def test_group_advantages_are_centred():
    rewards = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 3.0]])
    centred = dl.group_relative_advantages(rewards, normalise_by_std=False)
    assert torch.allclose(centred.mean(dim=1), torch.zeros(2), atol=1e-6)

    scaled = dl.group_relative_advantages(rewards, normalise_by_std=True)
    assert torch.allclose(scaled.mean(dim=1), torch.zeros(2), atol=1e-5)
    assert float(scaled.std(dim=1)[0]) == pytest.approx(1.0, abs=1e-3)


def test_group_advantages_validate_shape():
    with pytest.raises(ValueError):
        dl.group_relative_advantages(torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError):
        dl.group_relative_advantages(torch.tensor([[1.0]]))


def test_grpo_at_ratio_one_is_the_negative_weighted_advantage():
    log_probs = torch.zeros(2, 3, requires_grad=True)
    old = torch.zeros(2, 3)
    advantages = torch.tensor([1.0, -1.0])
    result = dl.grpo_loss(log_probs, old, advantages)
    assert float(result["loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(result["mean_ratio"]) == pytest.approx(1.0)


def test_grpo_clipping_engages_on_a_large_policy_move():
    log_probs = torch.full((1, 4), 2.0, requires_grad=True)
    old = torch.zeros(1, 4)
    result = dl.grpo_loss(log_probs, old, torch.tensor([1.0]), clip_epsilon=0.2)
    assert float(result["clip_fraction"]) == pytest.approx(1.0)
    assert float(result["loss"]) == pytest.approx(-1.2, abs=1e-4)


def test_grpo_respects_a_token_mask():
    log_probs = torch.zeros(1, 4, requires_grad=True)
    mask = torch.tensor([[True, True, False, False]])
    result = dl.grpo_loss(log_probs, torch.zeros(1, 4), torch.tensor([2.0]), token_mask=mask)
    assert float(result["loss"]) == pytest.approx(-2.0, abs=1e-5)


def test_grpo_validates_shapes():
    with pytest.raises(ValueError):
        dl.grpo_loss(torch.zeros(2, 3), torch.zeros(2, 4), torch.zeros(2))
    with pytest.raises(ValueError):
        dl.grpo_loss(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(3))


# ---------------------------------------------------------------------------
# Rollouts and the trainer
# ---------------------------------------------------------------------------


def test_rollout_is_deterministic_under_a_seeded_generator():
    model = build(0)
    prompt = torch.randint(0, VOCAB, (1, 4))
    first = dl.sample_rollout(model, prompt, max_new_tokens=6,
                              generator=torch.Generator().manual_seed(7))
    second = dl.sample_rollout(model, prompt, max_new_tokens=6,
                               generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)
    assert first.shape == (1, 10)


def test_greedy_rollout_matches_the_reference_decoder():
    import mimomix_decoding as md

    model = build(1)
    model.eval()
    prompt = torch.randint(0, VOCAB, (1, 5))
    rollout = dl.sample_rollout(model, prompt, max_new_tokens=6, temperature=0.0)
    reference = md.greedy_generate(model, prompt, max_new_tokens=6)
    assert torch.equal(rollout[:, 5:], reference.new_tokens)


def test_trainer_requires_a_teacher():
    student = build(0)
    with pytest.raises(ValueError):
        dl.MOPDTrainer(student=student, teachers=[], optimizer=torch.optim.SGD(student.parameters(), lr=0.1))


def test_trainer_freezes_every_teacher():
    student = build(0)
    teacher_model = build(1)
    trainer = dl.MOPDTrainer(
        student=student,
        teachers=[dl.TeacherSpec("t", teacher_model)],
        optimizer=torch.optim.SGD(student.parameters(), lr=0.1),
    )
    assert all(not p.requires_grad for p in teacher_model.parameters())
    assert trainer.teachers[0].model.training is False


def test_a_training_step_moves_the_student_toward_the_teachers():
    torch.manual_seed(0)
    student = build(0)
    teachers = [
        dl.TeacherSpec("math", StubTeacher(one_hot_logits(2)), domain="math"),
        dl.TeacherSpec("code", StubTeacher(one_hot_logits(9)), domain="code"),
    ]
    optimiser = torch.optim.Adam(student.parameters(), lr=5e-3)
    trainer = dl.MOPDTrainer(
        student=student, teachers=teachers, optimizer=optimiser,
        max_new_tokens=4, rollout_temperature=1.0,
    )
    prompt = torch.randint(0, VOCAB, (1, 5))

    losses = []
    for step in range(25):
        report = trainer.step(prompt, generator=torch.Generator().manual_seed(step))
        losses.append(report["mean_kl"])

    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5, f"KL did not fall: {losses}"
    assert report["scored_positions"] == 4
    assert report["teachers"] == ["math", "code"]
    assert report["grad_norm"] >= 0.0


def test_trainer_only_scores_generated_positions():
    student = build(0)
    trainer = dl.MOPDTrainer(
        student=student,
        teachers=[dl.TeacherSpec("t", StubTeacher(one_hot_logits(1)))],
        optimizer=torch.optim.SGD(student.parameters(), lr=1e-3),
        max_new_tokens=3,
    )
    report = trainer.step(torch.randint(0, VOCAB, (1, 6)), generator=torch.Generator().manual_seed(0))
    assert report["sequence_length"] == 9
    assert report["scored_positions"] == 3


def test_trainer_confidence_weights_the_teacher_of_the_causal_next_token(monkeypatch):
    student = build(0, use_moe=False)
    sequence = torch.tensor([[0, 1, 2, 3, 4]])

    def fixed_rollout(model, prompt_ids, **kwargs):
        del model, kwargs
        return sequence.to(prompt_ids.device)

    monkeypatch.setattr(dl, "sample_rollout", fixed_rollout)
    trainer = dl.MOPDTrainer(
        student=student,
        teachers=[
            dl.TeacherSpec("causal-next", SequenceTeacher(predicts_next=True)),
            dl.TeacherSpec("current-token", SequenceTeacher(predicts_next=False)),
        ],
        optimizer=torch.optim.SGD(student.parameters(), lr=1e-3),
        config=dl.MOPDConfig(weighting="confidence", min_teacher_weight=0.0),
        max_new_tokens=2,
    )

    report = trainer.step(sequence[:, :3])
    assert report["scored_positions"] == 2
    assert report["mean_teacher_weights"][0] > 0.99
    assert report["mean_teacher_weights"][1] < 0.01


def test_teacher_spec_accepts_a_mimomix_model():
    teacher = build(2)
    spec = dl.TeacherSpec("peer", teacher)
    logits = spec.logits(torch.randint(0, VOCAB, (1, 4)))
    assert logits.shape == (1, 4, VOCAB)
    assert not logits.requires_grad
