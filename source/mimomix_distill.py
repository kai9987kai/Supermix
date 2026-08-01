"""Post-training for MiMoMix: domain RL, then multi-teacher on-policy distillation.

This implements the two post-training stages MiMo describes after SFT, at a
scale that runs on CPU:

**Stage 2 -- domain-specialised RL.** :func:`group_relative_advantages` and
:func:`grpo_loss` implement the group-relative policy-optimisation shape: sample
a group of rollouts per prompt, score them with a verifiable reward, and use the
group itself as the baseline instead of a learned critic.

**Stage 3 -- MOPD (Multi-Teacher On-Policy Distillation).** Several
domain-specialist teachers are merged back into one student. The key property,
and the reason MiMo highlights it, is that the signal is **dense and
token-level**: the student samples its own trajectory (on-policy, so it learns
on the states it actually visits) and every teacher scores *every position* of
that trajectory. Compare with sequence-level RL, where a whole rollout collapses
to one scalar.

The objective is reverse KL against a per-token teacher mixture:

    L = E_{x ~ student}  sum_i  KL( student(. | x_<i) || teacher_mix(. | x_<i) )

Reverse KL is mode-seeking: the student is punished for putting mass where the
teachers put none, which is what you want when merging specialists -- the
student should commit to each teacher's expertise rather than smear across all
of them. The per-position ``-KL`` is exposed as ``dense_reward`` because that is
literally the token-level reward signal.

Teacher mixing offers three weightings:

``uniform``
    every teacher counts equally; the honest default when you have no domain
    labels.
``confidence``
    weight each teacher at each position by how much probability it assigns to
    the token the student actually produced. Specialists dominate on their own
    domain and abstain elsewhere, without needing labels.
``domain``
    caller-supplied per-sequence weights, when the domain *is* known.

Nothing here claims a trained result. These are correct, tested implementations
of the objectives; a checkpoint has to earn its numbers separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mimomix_core import MiMoMixModel


__all__ = [
    "TeacherSpec",
    "MOPDConfig",
    "MOPDTrainer",
    "teacher_mixture_log_probs",
    "on_policy_distillation_loss",
    "group_relative_advantages",
    "grpo_loss",
    "sample_rollout",
]


LogitFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class TeacherSpec:
    """A domain specialist plus its label.

    ``model`` may be any callable mapping ``(B, T)`` token ids to ``(B, T, V)``
    logits -- a :class:`MiMoMixModel`, a plain ``nn.Module``, or a stub in a
    test. Teachers are always evaluated under ``no_grad``.
    """

    name: str
    model: object
    domain: str = "general"

    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if isinstance(self.model, MiMoMixModel):
                return self.model(input_ids, return_mtp=False, past_length=0).logits
            out = self.model(input_ids)
            if hasattr(out, "logits"):
                return out.logits
            return out


@dataclass
class MOPDConfig:
    temperature: float = 1.0
    weighting: str = "confidence"  # "uniform" | "confidence" | "domain"
    #: keep almost all teacher mass on the top-k tokens; 0 disables truncation.
    #: A tiny tail floor remains because reverse KL is undefined when the
    #: student has non-zero mass outside a teacher's hard-zero support.
    teacher_top_k: int = 0
    #: per-token probability floor used after top-k truncation.  This keeps the
    #: reverse-KL target finite and differentiable without materially changing
    #: the selected support (1e-8 allocates <0.1% tail mass at a 50k vocab).
    teacher_probability_floor: float = 1e-8
    #: floor on any single teacher's mixture weight, so a specialist cannot be
    #: switched off entirely by a confident sibling
    min_teacher_weight: float = 0.05
    #: clamp on the per-position KL, so one pathological position cannot
    #: dominate the batch gradient
    max_position_kl: float = 20.0

    def __post_init__(self) -> None:
        if self.weighting not in {"uniform", "confidence", "domain"}:
            raise ValueError(f"unknown weighting: {self.weighting!r}")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if self.teacher_top_k < 0:
            raise ValueError("teacher_top_k must be non-negative")
        if not 0.0 < self.teacher_probability_floor < 1.0:
            raise ValueError("teacher_probability_floor must lie in (0, 1)")
        if not 0.0 <= self.min_teacher_weight < 1.0:
            raise ValueError("min_teacher_weight must lie in [0, 1)")
        if self.max_position_kl <= 0.0:
            raise ValueError("max_position_kl must be positive")


# ---------------------------------------------------------------------------
# Teacher mixture
# ---------------------------------------------------------------------------


def teacher_mixture_log_probs(
    teacher_logits: Sequence[torch.Tensor],
    student_tokens: torch.Tensor,
    config: Optional[MOPDConfig] = None,
    domain_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine specialists into one per-token target distribution.

    Returns ``(mixture_log_probs, weights)`` shaped ``(B, T, V)`` and
    ``(B, T, n_teachers)``.

    The mixture is formed in probability space (a proper mixture of
    distributions), not by averaging logits -- averaging logits is a geometric
    mean, which silently down-weights any token a single teacher dislikes and
    is not what "combine these experts" means.
    """

    config = config or MOPDConfig()
    if not teacher_logits:
        raise ValueError("at least one teacher is required")
    stacked = torch.stack(
        [logits.float() / config.temperature for logits in teacher_logits], dim=0
    )  # (K, B, T, V)
    n_teachers = stacked.shape[0]
    full_teacher_log_probs = F.log_softmax(stacked, dim=-1)

    if config.teacher_top_k and config.teacher_top_k < stacked.shape[-1]:
        vocab = int(stacked.shape[-1])
        floor = float(config.teacher_probability_floor)
        if floor * vocab >= 1.0:
            raise ValueError(
                "teacher_probability_floor * vocab_size must be below 1"
            )
        top_indices = stacked.topk(config.teacher_top_k, dim=-1).indices
        support = torch.zeros_like(stacked, dtype=torch.bool)
        support.scatter_(-1, top_indices, True)
        truncated = full_teacher_log_probs.exp().masked_fill(~support, 0.0)
        truncated = truncated / truncated.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        # Preserve a tiny, explicit full-vocabulary tail. Hard -inf masking is
        # incompatible with reverse KL: a softmax student has non-zero mass at
        # every token, so the KL becomes infinite (and confidence weighting can
        # become NaN when every picked score is -inf).
        teacher_probs = truncated * (1.0 - floor * vocab) + floor
        teacher_log_probs = teacher_probs.log()
    else:
        teacher_log_probs = full_teacher_log_probs

    if config.weighting == "uniform":
        weights = teacher_log_probs.new_full((n_teachers,) + teacher_log_probs.shape[1:3], 1.0 / n_teachers)
    elif config.weighting == "domain":
        if domain_weights is None:
            raise ValueError("weighting='domain' requires domain_weights")
        dw = domain_weights.float()
        if dw.shape[0] != n_teachers:
            raise ValueError("domain_weights must have one row per teacher")
        while dw.dim() < 3:
            dw = dw.unsqueeze(-1)
        weights = dw.expand(n_teachers, *teacher_log_probs.shape[1:3]).contiguous()
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-9)
    else:  # confidence
        # How much probability does teacher k give the token the student chose?
        # Use the full teacher distributions for this comparison. A sampled
        # token outside every truncated top-k still carries finite evidence
        # about which teacher predicted it best.
        picked = full_teacher_log_probs.gather(
            -1, student_tokens.unsqueeze(0).unsqueeze(-1).expand(n_teachers, -1, -1, 1)
        ).squeeze(-1)  # (K, B, T)
        weights = F.softmax(picked, dim=0)

    if config.min_teacher_weight > 0.0:
        floor = config.min_teacher_weight
        weights = weights.clamp_min(floor)
        weights = weights / weights.sum(dim=0, keepdim=True)

    mixture = torch.logsumexp(teacher_log_probs + weights.unsqueeze(-1).clamp_min(1e-12).log(), dim=0)
    # Renormalise: floating-point logsumexp over a mixture is close but not
    # exactly normalised, and the KL below assumes a real distribution.
    mixture = mixture - torch.logsumexp(mixture, dim=-1, keepdim=True)
    return mixture, weights.permute(1, 2, 0).contiguous()


# ---------------------------------------------------------------------------
# On-policy distillation loss
# ---------------------------------------------------------------------------


@dataclass
class DistillationResult:
    loss: torch.Tensor
    dense_reward: torch.Tensor
    mean_kl: float
    max_position_kl: float
    teacher_weights: torch.Tensor
    clipped_positions: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "loss": float(self.loss.detach()),
            "mean_kl": self.mean_kl,
            "max_position_kl": self.max_position_kl,
            "clipped_positions": self.clipped_positions,
            "mean_teacher_weights": [
                float(v) for v in self.teacher_weights.mean(dim=(0, 1)).tolist()
            ],
        }


def on_policy_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: Sequence[torch.Tensor],
    student_tokens: torch.Tensor,
    config: Optional[MOPDConfig] = None,
    domain_weights: Optional[torch.Tensor] = None,
    position_mask: Optional[torch.Tensor] = None,
) -> DistillationResult:
    """Reverse KL from the student to a per-token teacher mixture.

    ``student_logits``  ``(B, T, V)``, gradient-carrying.
    ``teacher_logits``  list of ``(B, T, V)``, detached.
    ``student_tokens``  ``(B, T)`` -- the sampled tokens predicted by the
                        corresponding logit positions, used only for confidence
                        weighting. For a causal LM this is ``sequence[:, 1:]``
                        aligned with ``logits[:, :-1]``.
    ``position_mask``   optional ``(B, T)`` bool; ``False`` positions (padding,
                        or the prompt prefix) are excluded from the mean.
    """

    config = config or MOPDConfig()
    mixture_log_probs, weights = teacher_mixture_log_probs(
        teacher_logits, student_tokens, config=config, domain_weights=domain_weights
    )
    mixture_log_probs = mixture_log_probs.detach()

    student_log_probs = F.log_softmax(student_logits.float() / config.temperature, dim=-1)
    student_probs = student_log_probs.exp()

    # KL(student || teacher), computed exactly over the vocabulary at every
    # position -- the trajectory is on-policy, the inner expectation is not
    # sampled.
    per_position = (student_probs * (student_log_probs - mixture_log_probs)).sum(dim=-1)

    clipped = int((per_position > config.max_position_kl).sum())
    per_position = per_position.clamp(max=config.max_position_kl)

    if position_mask is not None:
        mask = position_mask.to(per_position.dtype)
        denominator = mask.sum().clamp_min(1.0)
        loss = (per_position * mask).sum() / denominator
        observed = per_position[position_mask]
    else:
        loss = per_position.mean()
        observed = per_position

    return DistillationResult(
        loss=loss,
        dense_reward=(-per_position).detach(),
        mean_kl=float(observed.mean().detach()) if observed.numel() else 0.0,
        max_position_kl=float(observed.max().detach()) if observed.numel() else 0.0,
        teacher_weights=weights.detach(),
        clipped_positions=clipped,
    )


# ---------------------------------------------------------------------------
# Group-relative policy optimisation (domain RL stage)
# ---------------------------------------------------------------------------


def group_relative_advantages(
    rewards: torch.Tensor, normalise_by_std: bool = True, eps: float = 1e-6
) -> torch.Tensor:
    """Centre rewards within each group; optionally scale by the group std.

    ``rewards`` is ``(groups, rollouts)``.

    On ``normalise_by_std``: dividing by the group standard deviation is the
    original GRPO formulation, but it up-weights groups whose rollouts happened
    to agree, which biases the objective toward already-solved prompts. Later
    work (the Dr.GRPO line) drops the division. Both are available; centring is
    always applied because that is what removes the baseline.
    """

    if rewards.dim() != 2:
        raise ValueError(f"expected (groups, rollouts), got {tuple(rewards.shape)}")
    if rewards.shape[1] < 2:
        raise ValueError("group-relative advantages need at least 2 rollouts per group")
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalise_by_std:
        return centred
    return centred / (rewards.std(dim=1, keepdim=True) + eps)


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    token_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Clipped surrogate objective with per-rollout advantages.

    ``log_probs`` / ``old_log_probs``  ``(N, T)`` for the *sampled* tokens.
    ``advantages``                     ``(N,)``, broadcast across the rollout.

    Returns the loss plus the clip fraction, which is the diagnostic that tells
    you whether the trust region is doing anything.
    """

    if log_probs.shape != old_log_probs.shape:
        raise ValueError("log_probs and old_log_probs must have the same shape")
    if advantages.shape[0] != log_probs.shape[0]:
        raise ValueError("one advantage per rollout is required")

    ratio = (log_probs - old_log_probs.detach()).exp()
    advantage = advantages.detach().unsqueeze(-1)
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    surrogate = torch.minimum(unclipped, clipped)

    if token_mask is not None:
        mask = token_mask.to(surrogate.dtype)
        denominator = mask.sum().clamp_min(1.0)
        loss = -(surrogate * mask).sum() / denominator
        clip_fraction = ((unclipped > clipped).to(mask.dtype) * mask).sum() / denominator
    else:
        loss = -surrogate.mean()
        clip_fraction = (unclipped > clipped).float().mean()

    return {
        "loss": loss,
        "clip_fraction": clip_fraction.detach(),
        "mean_ratio": ratio.mean().detach(),
    }


# ---------------------------------------------------------------------------
# Rollouts and the trainer
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_rollout(
    model: MiMoMixModel,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    generator: Optional[torch.Generator] = None,
    thinking_cycles: Optional[int] = None,
) -> torch.Tensor:
    """Sample an on-policy continuation. Deterministic given ``generator``.

    Greedy when ``temperature <= 0``. Uses the model's KV cache, so a rollout
    costs one forward per token rather than one forward per prefix.
    """

    model.eval()
    out = model(prompt_ids, use_cache=True, return_mtp=False, past_length=0,
                thinking_cycles=thinking_cycles)
    past = out.past_key_values
    position = int(prompt_ids.shape[1])
    logits = out.logits[:, -1]
    generated: List[torch.Tensor] = []

    for _ in range(max_new_tokens):
        if temperature <= 0.0:
            token = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits.float() / temperature, dim=-1)
            token = torch.multinomial(probs, num_samples=1, generator=generator)
        generated.append(token)
        step = model(token, past_key_values=past, use_cache=True, return_mtp=False,
                     past_length=position, thinking_cycles=thinking_cycles)
        past = step.past_key_values
        position += 1
        logits = step.logits[:, -1]

    return torch.cat([prompt_ids] + generated, dim=1)


@dataclass
class MOPDTrainer:
    """One-call MOPD step: roll out on-policy, score with teachers, update.

    ``student`` is trained; every teacher stays frozen and is only ever called
    under ``no_grad``.
    """

    student: MiMoMixModel
    teachers: List[TeacherSpec]
    optimizer: torch.optim.Optimizer
    config: MOPDConfig = field(default_factory=MOPDConfig)
    max_new_tokens: int = 8
    rollout_temperature: float = 1.0
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if not self.teachers:
            raise ValueError("MOPD needs at least one teacher")
        for spec in self.teachers:
            model = spec.model
            if isinstance(model, nn.Module):
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)

    def step(
        self,
        prompt_ids: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        domain_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        prompt_length = int(prompt_ids.shape[1])
        sequence = sample_rollout(
            self.student,
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.rollout_temperature,
            generator=generator,
        )

        self.student.train()
        student_out = self.student(sequence, return_mtp=False, past_length=0)
        teacher_logits = [spec.logits(sequence) for spec in self.teachers]

        # Causal alignment is load-bearing here: logits at position i predict
        # sequence[i + 1]. Confidence weighting must therefore inspect the
        # *next* sampled token, not sequence[i] (the already-observed input).
        student_logits = student_out.logits[:, :-1]
        aligned_teacher_logits = [logits[:, :-1] for logits in teacher_logits]
        sampled_next_tokens = sequence[:, 1:]

        # Score only predictions whose targets were generated by the student.
        # The prompt prefix was not sampled on-policy, so distilling on it is
        # off-policy and would reintroduce the mismatch MOPD exists to remove.
        mask = torch.zeros_like(sampled_next_tokens, dtype=torch.bool)
        mask[:, prompt_length - 1 :] = True

        aligned_domain_weights = domain_weights
        if (
            aligned_domain_weights is not None
            and aligned_domain_weights.dim() >= 3
            and aligned_domain_weights.shape[-1] == sequence.shape[1]
        ):
            aligned_domain_weights = aligned_domain_weights[..., :-1]

        result = on_policy_distillation_loss(
            student_logits,
            aligned_teacher_logits,
            sampled_next_tokens,
            config=self.config,
            domain_weights=aligned_domain_weights,
            position_mask=mask,
        )
        total = result.loss
        if student_out.aux_loss is not None:
            total = total + student_out.aux_loss

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.grad_clip)
        self.optimizer.step()
        # Exactly one router-bias update per optimizer step. Firing it per
        # forward would scale the effective gamma by the number of forwards.
        self.student.step_router_bias()

        payload = result.to_dict()
        payload.update(
            {
                "total_loss": float(total.detach()),
                "aux_loss": float(student_out.aux_loss.detach()) if student_out.aux_loss is not None else 0.0,
                "grad_norm": float(grad_norm),
                "scored_positions": int(mask.sum()),
                "sequence_length": int(sequence.shape[1]),
                "teachers": [spec.name for spec in self.teachers],
            }
        )
        return payload
