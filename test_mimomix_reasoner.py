"""Invariants for the v56 latent state-machine reasoner and its curriculum.

A passing suite proves integration, gradient flow, and the specific structural
invariants named below. It does not prove the model is good at anything --
`benchmark_mimomix_reasoner.py` measures that and `run_v56_promotion_gate.py`
decides whether a measurement survives a paired multi-seed test.

Pinned here:

1. No input dimension is silently discarded by any slot layout.
2. A fresh "gated" model is *exactly* position-equivariant: the same operation in
   a different block produces a bit-identical operator.
3. Identity initialisation makes a fresh chain information-preserving instead of
   mixing the state to uniform.
4. Every transition matrix is row-stochastic and every traced state is a
   probability distribution, at every step.
5. The curriculum's encoding and label rule match the v51 generator exactly, and
   every curriculum example lies inside that generator's own support.
6. A checkpoint round-trips bit-exactly, and a foreign checkpoint is refused
   rather than coerced.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))

import mimomix_reasoner as mr  # noqa: E402
import reasoner_curriculum as rc  # noqa: E402
from benchmark_cognitive_leap_ultra_v51 import make_chained_task  # noqa: E402


def small_config(**overrides) -> mr.ReasonerConfig:
    """A deliberately tiny config so the suite stays fast."""

    base = dict(
        hidden_size=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        intermediate_size=64,
        n_routed_experts=4,
        moe_intermediate_size=16,
        operator_hidden=48,
        thinking_latent_dim=16,
        thinking_cycles=2,
        thinking_max_cycles=4,
    )
    base.update(overrides)
    return mr.ReasonerConfig(**base)


# ---------------------------------------------------------------------------
# Slot layouts
# ---------------------------------------------------------------------------


def test_block_layout_covers_every_input_dimension() -> None:
    context, operators = rc_block_slots()
    covered = sorted(
        index for start, stop in list(context) + list(operators) for index in range(start, stop)
    )
    assert covered == list(range(128)), "a slot layout must not drop input dimensions"


def rc_block_slots():
    return mr.block_slots(128, 10, 24, 4)


def test_patch_layout_covers_every_input_dimension() -> None:
    context, operators = mr.patch_slots(128, 8)
    covered = sorted(
        index for start, stop in list(context) + list(operators) for index in range(start, stop)
    )
    assert covered == list(range(128))
    assert len(operators) == 7


def test_block_layout_matches_the_v51_generator_boundaries() -> None:
    _, operators = rc_block_slots()
    assert operators == ((10, 34), (34, 58), (58, 82), (82, 106))


def test_a_layout_that_leaves_no_context_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="no context slot"):
        mr.block_slots(96, 0, 24, 4)


def test_an_oversized_block_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="block layout needs"):
        mr.block_slots(64, 10, 24, 4)


def test_shared_encoder_requires_equal_width_operator_slots() -> None:
    with pytest.raises(ValueError, match="equal-width operator slots"):
        # patches of 128/7 are unequal in the tail, so sharing cannot apply
        mr.LatentStateReasoner(
            small_config(slot_layout="patches", patch_count=7, share_block_encoder=True)
        )


# ---------------------------------------------------------------------------
# Position equivariance
# ---------------------------------------------------------------------------


def _one_block_input(block: int, op_type: int, operand: int, start: int = 3) -> torch.Tensor:
    x = torch.zeros(1, 128)
    x[0, start] = 1.0
    base = 10 + 24 * block
    x[0, base + op_type] = 1.0
    x[0, base + 3 + operand] = 1.0
    return x


@pytest.mark.parametrize("context_mode", ["slot", "gated"])
def test_a_fresh_model_is_position_equivariant(context_mode: str) -> None:
    """The same operation in a different block must give the same operator.

    This is the whole point of sharing the block encoder. The "gated" mode starts
    with a zero gate precisely so that equivariance is the starting point rather
    than something the model has to rediscover.
    """

    torch.manual_seed(0)
    model = mr.LatentStateReasoner(
        small_config(operator_context=context_mode, positional_blocks=False)
    )
    model.eval()
    with torch.no_grad():
        first = model(_one_block_input(0, 1, 7)).operator_log_probs[0]
        second = model(_one_block_input(1, 1, 7)).operator_log_probs[1]
    assert torch.equal(first, second)


def test_trunk_context_mode_is_not_claimed_to_be_equivariant() -> None:
    """`operator_context="trunk"` reads a state attention has mixed.

    Recorded so the trade-off is explicit: the trunk mode is more expressive and
    is *not* equivariant, which is why it is not the default.
    """

    torch.manual_seed(0)
    model = mr.LatentStateReasoner(
        small_config(operator_context="trunk", positional_blocks=False)
    )
    model.eval()
    with torch.no_grad():
        first = model(_one_block_input(0, 1, 7)).operator_log_probs[0]
        second = model(_one_block_input(1, 1, 7)).operator_log_probs[1]
    assert not torch.equal(first, second)


def test_positional_blocks_breaks_equivariance_by_design() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(
        small_config(operator_context="slot", positional_blocks=True)
    )
    model.eval()
    with torch.no_grad():
        first = model(_one_block_input(0, 1, 7)).operator_log_probs[0]
        second = model(_one_block_input(1, 1, 7)).operator_log_probs[1]
    assert not torch.equal(first, second)


# ---------------------------------------------------------------------------
# Identity initialisation and the state machine
# ---------------------------------------------------------------------------


def test_identity_initialisation_preserves_the_initial_state() -> None:
    """A fresh chain must pass its initial state through, not mix it to uniform.

    Without this, a product of four near-uniform stochastic matrices destroys the
    signal and the gradient that has to reach the first operator vanishes.
    """

    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config(identity_gain=8.0))
    model.eval()
    x = torch.randn(4, 128) * 0.01
    with torch.no_grad():
        out = model(x)
    initial = out.state_trace[0].exp()
    final = out.state_trace[-1].exp()
    assert torch.allclose(initial, final, atol=5e-2), (
        "an identity-initialised chain should approximately preserve the state"
    )


def test_zero_identity_gain_mixes_the_state_toward_uniform() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config(identity_gain=0.0))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 128) * 0.01)
    uniform = math.log(float(model.config.n_states))
    final_entropy = float(-(out.state_trace[-1].exp() * out.state_trace[-1]).sum(-1).mean())
    initial_entropy = float(-(out.state_trace[0].exp() * out.state_trace[0]).sum(-1).mean())
    assert final_entropy > initial_entropy
    assert final_entropy <= uniform + 1e-6


def test_every_operator_is_row_stochastic() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(6, 128) * 0.01)
    assert len(out.operator_log_probs) == 4
    for operator in out.operator_log_probs:
        rows = operator.exp().sum(dim=-1)
        assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_every_traced_state_is_a_distribution() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(6, 128) * 0.01)
    assert len(out.state_trace) == 1 + len(out.operator_log_probs)
    for state in out.state_trace:
        mass = state.exp().sum(dim=-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-5)
        assert bool((state.exp() >= 0).all())


def test_the_trace_composition_matches_the_reported_final_state() -> None:
    """Composing the reported operators must reproduce the reported final state.

    If these ever disagree, the interface is showing a reasoning trace the model
    did not actually follow.
    """

    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(3, 128) * 0.01)
    state = out.state_trace[0]
    for operator in out.operator_log_probs:
        state = torch.logsumexp(state.unsqueeze(-1) + operator, dim=1)
    assert torch.allclose(state, out.state_log_probs, atol=1e-6)


# ---------------------------------------------------------------------------
# Training mechanics
# ---------------------------------------------------------------------------


def test_loss_flows_to_the_tokenizer_and_the_operator() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.train()
    out = model(torch.randn(8, 128) * 0.01, labels=torch.randint(0, 10, (8,)))
    out.loss.backward()
    for name in ("tokenizer.operator_proj.0.weight", "operator.net.0.weight", "class_head.weight"):
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and float(parameter.grad.abs().sum()) > 0, name


def test_step_router_bias_reports_every_moe_layer() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.train()
    out = model(torch.randn(8, 128) * 0.01, labels=torch.randint(0, 10, (8,)))
    out.loss.backward()
    from mimomix_core import SparseMoEFeedForward

    expected = sum(1 for m in model.modules() if isinstance(m, SparseMoEFeedForward))
    assert model.step_router_bias() == expected
    assert expected > 0


def test_verifier_loss_requires_a_training_forward() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        model(torch.randn(4, 128) * 0.01)
    with pytest.raises(RuntimeError, match="quality logits"):
        model.verifier_loss(torch.ones(4))


def test_verifier_loss_broadcasts_per_example_correctness() -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    model.train()
    labels = torch.randint(0, 10, (5,))
    out = model(torch.randn(5, 128) * 0.01, labels=labels)
    correctness = (out.logits.argmax(dim=-1) == labels).float()
    loss = model.verifier_loss(correctness)
    assert torch.isfinite(loss) and float(loss.detach()) > 0


def test_a_model_without_a_thinking_core_refuses_verifier_loss() -> None:
    model = mr.LatentStateReasoner(small_config(use_thinking_core=False))
    assert model.thinking_core is None
    with pytest.raises(RuntimeError, match="without a thinking core"):
        model.verifier_loss(torch.ones(2))


def test_operator_entropy_penalty_changes_the_loss_but_not_the_logits() -> None:
    """The crispness prior must be a training signal, not a different model."""

    torch.manual_seed(0)
    plain = mr.LatentStateReasoner(small_config(operator_entropy_weight=0.0))
    torch.manual_seed(0)
    penalised = mr.LatentStateReasoner(small_config(operator_entropy_weight=0.5))
    x = torch.randn(4, 128) * 0.01
    labels = torch.randint(0, 10, (4,))
    plain.eval()
    penalised.eval()
    with torch.no_grad():
        first = plain(x, labels=labels)
        second = penalised(x, labels=labels)
    assert torch.equal(first.logits, second.logits)
    assert float(second.loss) > float(first.loss)


def test_the_v51_input_shape_is_accepted_directly() -> None:
    """The harness hands out (B, 1, 128); consuming it is the whole fairness claim."""

    x, y = make_chained_task(6, seed=97)
    assert x.shape == (6, 1, 128)
    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (6, 10)


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------


def test_curriculum_encoding_matches_the_v51_generator() -> None:
    rc.assert_matches_reference_encoding(make_chained_task, samples=256)


def test_curriculum_tail_operations_are_the_representable_identity() -> None:
    """`mul 1` must be an identity, or the curriculum is a different task."""

    generator = torch.Generator().manual_seed(3)
    start, op_types, operands = rc.sample_chain(
        64, active_ops=1, generator=generator, random_slots=False
    )
    assert bool((op_types[:, 1:] == rc.IDENTITY_OP_TYPE).all())
    assert bool((operands[:, 1:] == rc.IDENTITY_OPERAND).all())
    labels = rc.apply_chain(start, op_types, operands)
    one_step = rc.apply_chain(start, op_types[:, :1], operands[:, :1])
    assert torch.equal(labels, one_step), "the identity tail must not change the label"


def test_identity_operations_anywhere_leave_the_label_alone() -> None:
    """The same invariant under the default, where identities are scattered.

    Dropping every identity slot must give the same label as keeping them, or
    the curriculum is quietly a different task.
    """

    generator = torch.Generator().manual_seed(13)
    start, op_types, operands = rc.sample_chain(
        256, active_ops=2, generator=generator, random_slots=True
    )
    labels = rc.apply_chain(start, op_types, operands)
    for row in range(start.shape[0]):
        keep = [
            index
            for index in range(rc.N_OPS)
            if not (
                int(op_types[row, index]) == rc.IDENTITY_OP_TYPE
                and int(operands[row, index]) == rc.IDENTITY_OPERAND
            )
        ]
        reduced = rc.apply_chain(
            start[row : row + 1],
            op_types[row : row + 1, keep] if keep else op_types[row : row + 1, :0],
            operands[row : row + 1, keep] if keep else operands[row : row + 1, :0],
        )
        assert int(reduced[0]) == int(labels[row])


def test_random_slots_spreads_active_operations_over_every_slot() -> None:
    """Prefix pinning starves the late slots, which was a measured defect.

    With a fixed prefix, slots beyond `active_ops` never see a genuine operation
    until the final stage. On the first v56 curriculum checkpoint 83% of the
    remaining errors first diverged at exactly the last step.
    """

    generator = torch.Generator().manual_seed(1)
    _, prefix_ops, prefix_vals = rc.sample_chain(4000, 2, generator, random_slots=False)
    _, spread_ops, spread_vals = rc.sample_chain(4000, 2, generator, random_slots=True)

    def active_share(op_types, operands):
        active = ~((op_types == rc.IDENTITY_OP_TYPE) & (operands == rc.IDENTITY_OPERAND))
        return [float(active[:, slot].float().mean()) for slot in range(rc.N_OPS)]

    prefix = active_share(prefix_ops, prefix_vals)
    spread = active_share(spread_ops, spread_vals)
    assert prefix[2] == 0.0 and prefix[3] == 0.0, "prefix mode should starve the late slots"
    assert min(spread) > 0.3, "every slot must see genuine operations"
    assert max(spread) - min(spread) < 0.1, "coverage should be even across slots"


def test_random_slots_keeps_the_active_operation_count() -> None:
    """Spreading the slots must not change how many real operations there are."""

    generator = torch.Generator().manual_seed(2)
    for active_ops in (1, 2, 3):
        _, op_types, operands = rc.sample_chain(2000, active_ops, generator, random_slots=True)
        genuine = ~((op_types == rc.IDENTITY_OP_TYPE) & (operands == rc.IDENTITY_OPERAND))
        # a slot can also draw `mul 1` by chance, so this is an upper bound
        assert bool((genuine.sum(dim=1) <= active_ops).all())


def test_random_slots_stays_inside_the_generator_support() -> None:
    generator = torch.Generator().manual_seed(3)
    start, op_types, operands = rc.sample_chain(2000, 2, generator, random_slots=True)
    assert bool(((start >= 0) & (start <= 9)).all())
    assert bool(((op_types >= 0) & (op_types <= 2)).all())
    assert bool(((operands >= 1) & (operands <= 9)).all())


def test_random_slots_is_the_default_and_is_recorded(tmp_path: Path) -> None:
    spec = rc.CurriculumSpec(stages=rc.curriculum_stage_sizes(200))
    assert spec.random_slots is True
    assert spec.to_dict()["random_slots"] is True


def test_curriculum_examples_are_inside_the_generator_support() -> None:
    """Every field must be a value the v51 generator itself can produce."""

    generator = torch.Generator().manual_seed(5)
    for active in (1, 2, 3, 4):
        start, op_types, operands = rc.sample_chain(128, active_ops=active, generator=generator)
        assert bool(((start >= 0) & (start <= 9)).all())
        assert bool(((op_types >= 0) & (op_types <= 2)).all())
        assert bool(((operands >= 1) & (operands <= 9)).all())


def test_full_stage_is_the_untouched_distribution() -> None:
    generator = torch.Generator().manual_seed(7)
    _, op_types, operands = rc.sample_chain(4096, active_ops=rc.N_OPS, generator=generator)
    # all three op types and all nine operands appear in every slot
    for slot in range(rc.N_OPS):
        assert set(op_types[:, slot].tolist()) == {0, 1, 2}
        assert set(operands[:, slot].tolist()) == set(range(1, 10))


def test_stage_sizes_sum_exactly_and_weight_the_real_distribution() -> None:
    stages = rc.curriculum_stage_sizes(10_003, final_weight=2.0)
    assert sum(size for _, size in stages) == 10_003
    assert [stage for stage, _ in stages] == [1, 2, 3, 4]
    assert stages[-1][1] > stages[0][1]


def test_a_curriculum_pool_is_reproducible_from_its_seed() -> None:
    spec = rc.CurriculumSpec(stages=rc.curriculum_stage_sizes(400), seed=11)
    first = rc.sample_curriculum_pool(spec)
    second = rc.sample_curriculum_pool(spec)
    for (stage_a, x_a, y_a), (stage_b, x_b, y_b) in zip(first, second):
        assert stage_a == stage_b
        assert torch.equal(x_a, x_b)
        assert torch.equal(y_a, y_b)


def test_apply_chain_matches_the_generator_on_the_real_test_set() -> None:
    x, y = make_chained_task(512, seed=52)
    flat = x.squeeze(1)
    start = flat[:, 0:10].argmax(dim=1)
    op_types = torch.stack(
        [flat[:, 10 + 24 * k : 13 + 24 * k].argmax(dim=1) for k in range(4)], dim=1
    )
    operands = torch.stack(
        [flat[:, 13 + 24 * k : 23 + 24 * k].argmax(dim=1) for k in range(4)], dim=1
    )
    assert torch.equal(rc.apply_chain(start, op_types, operands), y)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_round_trips_bit_exactly(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config(n_states=12, patch_count=8))
    model.eval()
    x = torch.randn(5, 128) * 0.01
    with torch.no_grad():
        before = model(x).logits
    path = tmp_path / "nested" / "reasoner.pt"
    mr.save_reasoner(model, path, extra={"note": "test"})
    restored, payload = mr.load_reasoner(path)
    with torch.no_grad():
        after = restored(x).logits
    assert torch.equal(before, after)
    assert payload["schema"] == mr.CHECKPOINT_SCHEMA
    assert payload["extra"]["note"] == "test"
    assert restored.config.n_states == 12


def test_a_foreign_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "foreign.pt"
    torch.save({"schema": "something-else", "state_dict": {}}, path)
    with pytest.raises(ValueError, match="not a supermix-v56"):
        mr.load_reasoner(path)


def test_a_bare_state_dict_is_refused(tmp_path: Path) -> None:
    """A state dict alone cannot be reloaded: the shapes depend on the config."""

    torch.manual_seed(0)
    model = mr.LatentStateReasoner(small_config())
    path = tmp_path / "bare.pt"
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match="not a supermix-v56"):
        mr.load_reasoner(path)


def test_parameter_report_separates_idle_experts() -> None:
    model = mr.LatentStateReasoner(small_config())
    report = model.parameter_report()
    assert report["total"] == sum(p.numel() for p in model.parameters())
    assert report["routed_but_idle"] > 0
    assert report["active_per_input"] == report["total"] - report["routed_but_idle"]


def test_telemetry_is_json_safe() -> None:
    import json

    model = mr.LatentStateReasoner(small_config())
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(4, 128) * 0.01)
    json.dumps(out.telemetry)  # must not raise
    assert out.telemetry["n_states"] == model.config.n_states
    assert "thinking" in out.telemetry


def test_an_unknown_operator_context_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown operator_context"):
        small_config(operator_context="magic")


def test_an_unknown_slot_layout_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown slot_layout"):
        small_config(slot_layout="spiral")
