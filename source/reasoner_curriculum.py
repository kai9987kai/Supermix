"""In-support curriculum sampling for the v56 reasoner.

The v51 chained-modular-arithmetic task is close to non-decomposable. Measured
Bayes-optimal accuracy over 4,000,000 samples:

| information given | Bayes accuracy | Bayes cross-entropy |
| --- | --- | --- |
| nothing (majority class) | 0.1415 | 2.2843 |
| the last operation only | 0.1720 | 2.1240 |
| the last two operations | 0.2013 | 1.9810 |
| the last three operations | 0.2389 | 1.8406 |
| all four operations, no start digit | 0.4105 | 1.3162 |
| the start digit only | 0.1415 | 2.2843 |
| start + first three operations | 0.2521 | 1.9742 |
| **start + all four operations** | **1.0000** | **0.0000** |

Every proper subset of the input is worth almost nothing and the complete input
is worth everything, so a model that has learned three of the four maps gets
essentially no credit for it. That is why architectures from a flat MLP to a
latent state machine all plateau near 0.26 regardless of how much data they are
given: there is no gradient path from a partial solution to the exact one.

## The identity that makes a curriculum possible

`mul by 1` is an identity on the state and the encoding can represent it
(`op_type = 1`, `operand = 1`). A chain whose trailing operations are all
`mul 1` is therefore an *effectively shorter* chain that is still a **valid draw
from the generator's own support** -- the generator can and does produce it. So
the curriculum is a re-weighting of the same input distribution, not a different
task and not a new kind of example. Evaluation stays on the untouched
distribution.

That distinction matters for what a result means. A curriculum built from
modified inputs, extra label fields, or intermediate-state supervision would be
training on information the baseline never had. This one is not: every curriculum
sample is an (input, label) pair the original generator assigns exactly the same
way, and `assert_matches_reference_encoding` pins that against the v51 generator
itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "CurriculumSpec",
    "IDENTITY_OPERAND",
    "IDENTITY_OP_TYPE",
    "N_CLASSES",
    "N_OPS",
    "IN_DIM",
    "apply_chain",
    "assert_matches_reference_encoding",
    "curriculum_stage_sizes",
    "encode_chain",
    "sample_chain",
    "sample_curriculum_pool",
]

IN_DIM = 128
N_CLASSES = 10
N_OPS = 4
#: `mul by 1` -- the one representable identity in this encoding
IDENTITY_OP_TYPE = 1
IDENTITY_OPERAND = 1


def apply_chain(start: torch.Tensor, op_types: torch.Tensor, operands: torch.Tensor) -> torch.Tensor:
    """The generator's own label rule, vectorised.

    ``op_type`` 0 = add, 1 = multiply, 2 = subtract, all mod ``N_CLASSES``.
    """

    accumulator = start.clone()
    for index in range(op_types.shape[1]):
        op = op_types[:, index]
        operand = operands[:, index]
        accumulator = torch.where(
            op == 0,
            (accumulator + operand) % N_CLASSES,
            torch.where(
                op == 1,
                (accumulator * operand) % N_CLASSES,
                (accumulator - operand) % N_CLASSES,
            ),
        )
    return accumulator


def encode_chain(
    start: torch.Tensor,
    op_types: torch.Tensor,
    operands: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    noise: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode integer chains into the v51 128-dim layout and return ``(x, y)``.

    Layout, matching ``benchmark_cognitive_leap_ultra_v51.make_chained_task``:
    dims 0-9 are the start one-hot; block ``k`` begins at ``10 + 24k`` with the
    op-type one-hot at ``base + op`` and the operand one-hot at
    ``base + 3 + operand``. Gaussian noise of scale ``noise`` is added to every
    dimension.
    """

    count = int(start.shape[0])
    x = torch.zeros(count, IN_DIM)
    rows = torch.arange(count)
    x[rows, start] = 1.0
    for index in range(N_OPS):
        base = 10 + index * 24
        x[rows, base + op_types[:, index]] = 1.0
        x[rows, base + 3 + operands[:, index]] = 1.0
    if noise:
        x = x + noise * torch.randn(count, IN_DIM, generator=generator)
    return x, apply_chain(start, op_types, operands)


def sample_chain(
    count: int,
    active_ops: int,
    generator: torch.Generator,
    random_slots: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample integer chains with exactly ``active_ops`` non-identity operations.

    ``active_ops == N_OPS`` is the generator's own distribution. Lower values
    make the remaining slots `mul 1`, which the generator also produces -- just
    rarely (probability ``(1/27)`` per slot).

    ``random_slots`` chooses **which** slots stay active, uniformly per example,
    instead of always keeping a prefix. Measured on the first v56 curriculum
    checkpoint, prefix pinning was a real defect: the last slot only saw genuine
    operations during the final stage, and 83% of that model's remaining errors
    first diverged at exactly that step. Per-slot positional embeddings stop the
    shared operator from covering the gap, so the fix belongs here rather than in
    the model.
    """

    if not 1 <= active_ops <= N_OPS:
        raise ValueError(f"active_ops must be in 1..{N_OPS}, got {active_ops}")
    start = torch.randint(0, N_CLASSES, (count,), generator=generator)
    op_types = torch.randint(0, 3, (count, N_OPS), generator=generator)
    operands = torch.randint(1, N_CLASSES, (count, N_OPS), generator=generator)
    if active_ops < N_OPS:
        if random_slots:
            # keep the `active_ops` highest-scoring slots per row
            scores = torch.rand(count, N_OPS, generator=generator)
            keep = scores.argsort(dim=1, descending=True)[:, :active_ops]
            active = torch.zeros(count, N_OPS, dtype=torch.bool)
            active.scatter_(1, keep, True)
            op_types = torch.where(active, op_types, torch.full_like(op_types, IDENTITY_OP_TYPE))
            operands = torch.where(active, operands, torch.full_like(operands, IDENTITY_OPERAND))
        else:
            op_types[:, active_ops:] = IDENTITY_OP_TYPE
            operands[:, active_ops:] = IDENTITY_OPERAND
    return start, op_types, operands


@dataclass
class CurriculumSpec:
    """A curriculum as an explicit, reproducible list of (active_ops, size)."""

    stages: Tuple[Tuple[int, int], ...]
    seed: int = 51
    noise: float = 0.01
    #: spread the active operations over random slots instead of a fixed prefix
    random_slots: bool = True

    def total_examples(self) -> int:
        return sum(size for _, size in self.stages)

    def to_dict(self) -> Dict[str, object]:
        return {
            "stages": [{"active_ops": int(a), "examples": int(n)} for a, n in self.stages],
            "seed": int(self.seed),
            "noise": float(self.noise),
            "random_slots": bool(self.random_slots),
            "total_examples": int(self.total_examples()),
            "in_support": True,
            "identity_operation": {"op_type": IDENTITY_OP_TYPE, "operand": IDENTITY_OPERAND},
        }


def curriculum_stage_sizes(
    total: int, stages: Sequence[int] = (1, 2, 3, 4), final_weight: float = 2.0
) -> Tuple[Tuple[int, int], ...]:
    """Split ``total`` examples across ``stages``, weighting the last one up.

    The final stage is the real distribution, so it gets ``final_weight`` times
    an earlier stage's share. Sizes are integral and sum exactly to ``total``.
    """

    if total < len(stages):
        raise ValueError("total must be at least one example per stage")
    if final_weight <= 0:
        raise ValueError("final_weight must be > 0")
    weights = [1.0] * (len(stages) - 1) + [float(final_weight)]
    scale = total / sum(weights)
    sizes = [max(1, int(weight * scale)) for weight in weights]
    sizes[-1] += total - sum(sizes)  # absorb rounding into the real-distribution stage
    return tuple((int(stage), int(size)) for stage, size in zip(stages, sizes))


def sample_curriculum_pool(spec: CurriculumSpec) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """Materialise the curriculum as ``[(active_ops, x, y), ...]``, in order."""

    generator = torch.Generator().manual_seed(int(spec.seed))
    pool: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for active_ops, size in spec.stages:
        start, op_types, operands = sample_chain(
            size, active_ops, generator, random_slots=spec.random_slots
        )
        x, y = encode_chain(start, op_types, operands, generator=generator, noise=spec.noise)
        pool.append((int(active_ops), x, y))
    return pool


def assert_matches_reference_encoding(reference_make_chained_task, samples: int = 256) -> None:
    """Pin this module's encoding and label rule against the v51 generator.

    ``reference_make_chained_task`` is
    ``benchmark_cognitive_leap_ultra_v51.make_chained_task``. The two use
    different RNG consumption orders, so the tensors cannot be compared directly.
    What *can* be compared -- and is what matters -- is that decoding the
    reference tensor's one-hot fields and pushing them through this module's
    encoder and label rule reproduces the reference exactly.
    """

    x_reference, y_reference = reference_make_chained_task(samples, seed=97)
    flat = x_reference.squeeze(1)
    start = flat[:, 0:10].argmax(dim=1)
    op_types = torch.stack(
        [flat[:, 10 + 24 * k : 13 + 24 * k].argmax(dim=1) for k in range(N_OPS)], dim=1
    )
    operands = torch.stack(
        [flat[:, 13 + 24 * k : 23 + 24 * k].argmax(dim=1) for k in range(N_OPS)], dim=1
    )

    labels = apply_chain(start, op_types, operands)
    if not torch.equal(labels, y_reference):
        mismatched = int((labels != y_reference).sum())
        raise AssertionError(f"label rule diverges from the reference on {mismatched} samples")

    noiseless, _ = encode_chain(start, op_types, operands, noise=0.0)
    reference_noiseless = (flat > 0.5).float()
    if not torch.equal(noiseless, reference_noiseless):
        differing = int((noiseless != reference_noiseless).sum())
        raise AssertionError(f"encoding diverges from the reference in {differing} positions")
