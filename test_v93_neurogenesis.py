"""v93 neurogenesis controller (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md, D6):
one event runs every stage with its quota, exact operations leave the witness
loss unchanged, non-exact ones are measured and logged, AdamW moments are
zeroed on exactly the written slices, apoptosis needs two consecutive weak
readings, the cross-hemisphere quota holds, the state round-trips through a
JSON-safe dict so a crash resume continues the counters, and two identical
models produce identical events.

Everything runs on a tiny model: hidden 32, 2 layers (the second is MoE with
4 experts + 2 spare slots), a synthetic 8-module two-hemisphere graph in 12
slots (4 spare), two read taps, one write site and the thinking bond.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

from mimomix_core import MiMoMixConfig, MiMoMixModel, SparseMoEFeedForward  # noqa: E402
from neurogenesis import (  # noqa: E402
    FLAG_THRESHOLD_NATS,
    GrowthSettings,
    NeurogenesisController,
    build_settings_from_args,
    zero_moments,
)

N = 8          # live modules in the synthetic graph
SPARE = 4      # dead slots after them
CAP = N + SPARE
HIDDEN = 32
VOCAB = 97
SEQ = 12


def _config(**overrides):
    base = dict(
        vocab_size=VOCAB, hidden_size=HIDDEN, n_layers=2, n_heads=4, n_kv_heads=2,
        intermediate_size=48, moe_intermediate_size=16, n_routed_experts=4, moe_spare_experts=2,
        moe_top_k=2, n_mtp_layers=1, sliding_window=8, native_context=32,
        max_position_embeddings=32, thinking_latent_dim=8,
        use_cns_core=True, cns_nodes=CAP, cns_spare_nodes=SPARE, cns_after_layer=1, cns_steps=3,
        cns_read_layers=(0, 1), cns_write_layers=(1,), cns_to_thinking=True,
    )
    base.update(overrides)
    return MiMoMixConfig(**base)


def _graph_npz(path, seed=0, sided=True, density=0.4):
    """An 8-module graph in the layout ``malecns_connectome.py hemispheres``
    writes: modules 0-3 left, 4-7 right (mirror scheme), two afferent
    (sensory) and two efferent (descending) modules, sparse enough that
    every block keeps unconnected pairs for synaptogenesis to open."""

    rng = np.random.default_rng(seed)
    roles = np.array(["sensory", "central", "central", "descending"] * 2, dtype=object)
    sign = np.array([1, -1, 1, 1, 1, -1, 1, 1], dtype=np.int8)
    matrix = rng.random((N, N)) * (rng.random((N, N)) < density)
    np.fill_diagonal(matrix, 0.0)
    for post in range(N):
        if matrix[post].sum() == 0:
            matrix[post, (post + 1) % N] = 0.5
    post, pre = np.nonzero(matrix)
    fraction = matrix[post, pre] / matrix.sum(1)[post]
    arrays = dict(
        module_role=roles, module_sign=sign,
        edge_post=post.astype(np.int64), edge_pre=pre.astype(np.int64),
        edge_fraction=fraction.astype(np.float32),
        rewired_post=post.astype(np.int64), rewired_pre=pre.astype(np.int64),
    )
    if sided:
        arrays["module_side"] = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    np.savez_compressed(path, **arrays)
    return str(path)


def _model(tmp_path, seed=0, sided=True, **overrides):
    """A grafted tiny model whose core is open (gates 0.3) and whose modules
    share a common drive direction, so dev-pass rates co-vary and
    synaptogenesis has positive covariances to rank."""

    torch.manual_seed(seed)
    model = MiMoMixModel(_config(**overrides)).eval()
    core = model.cns_core
    core.load_graph(_graph_npz(tmp_path / f"graph_{seed}_{int(sided)}.npz", seed=seed, sided=sided))
    generator = torch.Generator().manual_seed(seed + 1)
    dead = core.alive == 0
    with torch.no_grad():
        core.gate.fill_(0.3)
        for gate in core.extra_gates:
            gate.fill_(0.3)
        if core.thinking_gate is not None:
            core.thinking_gate.fill_(0.3)
        shared = torch.randn(HIDDEN, generator=generator)
        for tap in [core.read_in, *core.extra_read_in]:
            tap.weight.copy_(0.5 * torch.randn(CAP, HIDDEN, generator=generator) + shared)
            tap.weight[dead] = 0.0
        core.leak_logit.fill_(0.5)
    return model


def _favour_experts(model, slots):
    """Pin the router's selection to ``slots`` in every MoE layer (bias +10
    dominates the softmax score range), so the other experts carry no load."""

    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, SparseMoEFeedForward):
                module.expert_bias.zero_()
                for slot in slots:
                    module.expert_bias[slot] = 10.0


def _witness(seed=7, rows=8):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (rows, SEQ), generator=generator)
    y = x.clone()
    y[:, :3] = -100
    return x, y


def _dev_pass(controller, model, seed=11, batches=3):
    generator = torch.Generator().manual_seed(seed)
    controller.begin_dev_pass()
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x = torch.randint(0, VOCAB, (4, SEQ), generator=generator)
            model(x, return_mtp=False)
    controller.end_dev_pass()


def _controller(model, tmp_path, name="a", **settings):
    return NeurogenesisController(
        model, GrowthSettings(**settings), str(tmp_path / name / "neurogenesis.jsonl"), _witness()
    )


def _moe(model):
    return [(i, layer.mlp) for i, layer in enumerate(model.layers) if isinstance(layer.mlp, SparseMoEFeedForward)]


# ---------------------------------------------------------------------------
# A. one event, every stage
# ---------------------------------------------------------------------------


def test_event_runs_every_stage_with_its_quota(tmp_path):
    model = _model(tmp_path)
    _favour_experts(model, [0, 1])
    core = model.cns_core
    controller = _controller(
        model, tmp_path, grow_every=500, grow_modules=2, grow_edges=4, grow_taps=1,
        grow_experts=True, expert_split_load_fraction=1.5,
    )
    _dev_pass(controller, model)
    assert controller.maybe_grow(499, None) is None
    # the mitosis ranking, recomputed independently from the same statistics
    stats = core.growth_statistics()
    with torch.no_grad():
        strength = core.weight().abs()
        normalised = strength / strength.sum(1, keepdim=True).clamp_min(1e-12)
        out_share = normalised.sum(0) / normalised.sum().clamp_min(1e-12)
        score = stats["mean_rate"] * (1 + out_share)
        score[core.alive == 0] = -1
        score[((core.mask > 0).sum(0) == 0) & (core.out_mask == 0)] = -1  # no output path: never split
    expected_parents = torch.topk(score, 2).indices.tolist()

    record = controller.maybe_grow(500, None)
    assert record["step"] == 500
    assert sorted(parent for parent, _ in record["modules_split"]) == sorted(expected_parents)
    assert sorted(child for _, child in record["modules_split"]) == [N, N + 1]
    assert record["edges_opened"]["count"] == 4 and sum(record["edges_opened"]["by_block"].values()) == 4
    assert record["taps_opened"] == {"in": 1, "out": 1}
    assert record["experts_born"] == [[1, 0, 4]]
    assert record["experts_killed"] == [] and record["experts_on_first_strike"] == 2  # experts 2, 3: strike one
    assert record["modules_killed"] == [] and record["edges_pruned"] == {"grown": 0, "real": 0}
    assert record["alive_modules"] == N + 2 and record["alive_experts_per_layer"] == [5]
    assert record["alive_edges"] == int(core.mask.sum()) and record["noise_seed"] == 93 * 1_000_003 + 500
    assert record["dev_tokens"] == 3 * 4 * SEQ
    assert core.alive[N:N + 2].tolist() == [1, 1] and core.born_step[N:N + 2].tolist() == [500, 500]
    assert int(core.edge_grown.sum()) == 4 and core.tap_grown.sum(0).tolist() == [1, 1]
    # the opened edges are alive, off-diagonal, at the requested logit
    logits = core.edge_logit.detach()
    for post, pre in torch.nonzero(core.edge_grown).tolist():
        assert post != pre and float(logits[post, pre]) == -7.0 and float(core.mask[post, pre]) == 1
    # the newborn expert is the favoured parent plus small noise, routable at once
    moe = _moe(model)[0][1]
    assert moe.alive_count() == 5 and float(moe.expert_bias[4]) == 10.0
    parent_w = moe.experts[0].down_proj.weight.detach()
    assert torch.allclose(moe.experts[4].down_proj.weight, parent_w, atol=0.05 * float(parent_w.std()))
    assert not torch.equal(moe.experts[4].down_proj.weight, parent_w)
    # one JSON line, equal to the record
    lines = open(controller.log_path, encoding="utf-8").read().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == json.loads(json.dumps(record))

    # second event: the two starved experts are killed (strike two), the last
    # two slots fill, and a third event has nothing left to split into
    _dev_pass(controller, model, seed=12)
    second = controller.maybe_grow(1000, None)
    assert second["experts_killed"] == [[1, 2], [1, 3]]
    assert moe.expert_alive[2:4].tolist() == [0, 0] and moe.alive_count() in (3, 4)
    assert len(second["modules_split"]) == 2 and core.free_slots() == []
    _dev_pass(controller, model, seed=13)
    third = controller.maybe_grow(1500, None)
    assert third["modules_split"] == [] and third["alive_modules"] == CAP
    summary = controller.summary()
    assert summary["n_events"] == 3 and len(summary["events"]) == 3
    assert summary["totals"]["modules_split"] == 4 and summary["totals"]["experts_killed"] == 2
    assert summary["totals"]["edges_opened"] == sum(e["edges_opened"] for e in summary["events"])
    assert summary["settings"]["grow_edges"] == 4 and summary["alive_modules"] == CAP
    assert len(open(controller.log_path, encoding="utf-8").read().splitlines()) == 3


def test_dev_pass_flags_and_statistics(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    controller = _controller(model, tmp_path, grow_every=500)
    moes = [m for _, m in _moe(model)]
    assert not core.collect_stats and all(not m.collect_stats for m in moes)
    controller.begin_dev_pass()
    assert core.collect_stats and all(m.collect_stats for m in moes) and core.growth_statistics()["tokens"] == 0
    model.eval()
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (2, SEQ)), return_mtp=False)
    controller.end_dev_pass()
    assert not core.collect_stats and all(not m.collect_stats for m in moes)
    assert core.growth_statistics()["tokens"] == 2 * SEQ and moes[0].growth_statistics()["batches"] == 1
    # the witness measurement neither trains nor pollutes the statistics
    model.train()
    loss = controller.witness_loss()
    assert model.training and loss > 0 and core.growth_statistics()["tokens"] == 2 * SEQ
    model.eval()
    assert controller.witness_loss() == loss and not model.training


# ---------------------------------------------------------------------------
# B. exact and non-exact operations on the witness batch
# ---------------------------------------------------------------------------


def test_exact_operations_leave_the_witness_loss_unchanged(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    with torch.no_grad():
        core.out_mask[:N] = 1.0  # every live module efferent: no efferent tap can open, so the RMS count is fixed
    controller = _controller(model, tmp_path, grow_every=500, grow_edges=3, grow_taps=2)
    _dev_pass(controller, model)
    record = controller.maybe_grow(500, None)
    assert record["taps_opened"] == {"in": 2, "out": 0} and record["edges_opened"]["count"] == 3
    assert record["modules_split"] == [] and record["experts_born"] == []
    assert abs(record["witness_delta"]) < 1e-5 and record["flagged"] is False
    # the afferent taps are exact at birth: zero rows on every read tap
    for module in torch.nonzero(core.tap_grown[:, 0]).flatten().tolist():
        assert float(core.in_mask[module]) == 1
        for tap in [core.read_in, *core.extra_read_in]:
            assert tap.weight[module].abs().sum() == 0


def test_split_delta_is_small_and_logged(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    with torch.no_grad():
        core.read_out.weight.normal_(0, 0.3)
        core.read_out.weight[:, core.alive == 0] = 0.0
    controller = _controller(model, tmp_path, grow_every=500, grow_modules=2)
    _dev_pass(controller, model)
    record = controller.maybe_grow(500, None)
    assert len(record["modules_split"]) == 2 and record["edges_opened"]["count"] == 0
    assert abs(record["witness_delta"]) < 0.05
    assert record["flagged"] == (abs(record["witness_delta"]) > FLAG_THRESHOLD_NATS)
    assert record["witness_loss_after"] == pytest.approx(record["witness_loss_before"] + record["witness_delta"])
    logged = json.loads(open(controller.log_path, encoding="utf-8").read().splitlines()[0])
    for key in (
        "step", "witness_loss_before", "witness_loss_after", "witness_delta", "flagged", "modules_split",
        "edges_opened", "edges_pruned", "taps_opened", "modules_killed", "experts_born", "experts_killed",
        "alive_modules", "alive_edges", "alive_experts_per_layer", "seconds",
    ):
        assert key in logged
    assert logged["flagged"] == record["flagged"] and logged["seconds"] >= 0
    assert controller.summary()["flagged_events"] == ([500] if record["flagged"] else [])


# ---------------------------------------------------------------------------
# C. optimiser moments
# ---------------------------------------------------------------------------


def test_zero_moments_helper():
    param = torch.nn.Parameter(torch.randn(4, 3))
    optimiser = torch.optim.AdamW([param], lr=1e-3)
    assert zero_moments(optimiser, param, None) == 0  # no state yet: nothing to zero, nothing created
    assert param not in optimiser.state
    optimiser.state[param] = {"step": torch.tensor(1.0), "exp_avg": torch.ones(4, 3), "exp_avg_sq": torch.ones(4, 3)}
    assert zero_moments(optimiser, param, (0, torch.tensor([False, True, False, False]))) == 6
    assert optimiser.state[param]["exp_avg"][1].sum() == 0 and optimiser.state[param]["exp_avg"].sum() == 9
    assert zero_moments(optimiser, param, (1, [2])) == 8  # 4 entries x 2 moments
    assert optimiser.state[param]["exp_avg_sq"][:, 2].sum() == 0
    entries = torch.zeros(4, 3, dtype=torch.bool)
    entries[0, 0] = True
    assert zero_moments(optimiser, param, entries) == 2
    assert zero_moments(optimiser, param, None) == 24 and optimiser.state[param]["exp_avg"].sum() == 0
    assert zero_moments(None, param, None) == 0
    with pytest.raises(ValueError):
        zero_moments(optimiser, param, torch.zeros(2, 2, dtype=torch.bool))


def test_moments_are_zeroed_only_on_written_slices(tmp_path):
    model = _model(tmp_path)
    _favour_experts(model, [0, 1])
    core = model.cns_core
    params = list(model.parameters())
    optimiser = torch.optim.AdamW(params, lr=1e-3)
    for p in params:
        optimiser.state[p] = {"step": torch.tensor(3.0), "exp_avg": torch.ones_like(p), "exp_avg_sq": torch.ones_like(p)}
    controller = _controller(
        model, tmp_path, grow_every=500, grow_modules=1, grow_edges=2, grow_taps=1,
        grow_experts=True, expert_split_load_fraction=1.5,
    )
    _dev_pass(controller, model)
    record = controller.maybe_grow(500, optimiser)
    written = controller.last_written
    assert record["moments_zeroed"] > 0 and written["edges"].any() and not written["edges"].all()

    def moments(p):
        return optimiser.state[p]["exp_avg"], optimiser.state[p]["exp_avg_sq"]

    # what the record says was written is in the masks
    (parent, child), = record["modules_split"]
    assert written["edges"][child].all() and written["edges"][:, parent].all() and written["edges"][:, child].all()
    assert written["nodes"][child] and written["in_rows"][child] and written["out_cols"][parent] and written["out_cols"][child]
    assert (written["edges"] | ~core.edge_grown.bool()).all()  # every opened edge
    assert written["experts"] == {1: [4]}
    # the moments are zero exactly there and untouched (still one) everywhere else
    for m in moments(core.edge_logit):
        assert torch.equal(m == 0, written["edges"])
    for p in (core.node_bias, core.leak_logit):
        for m in moments(p):
            assert torch.equal(m == 0, written["nodes"])
    for tap in [core.read_in, *core.extra_read_in]:
        for m in moments(tap.weight):
            assert torch.equal((m == 0).all(1), written["in_rows"]) and torch.equal((m == 1).all(1), ~written["in_rows"])
    for projection in [core.read_out, *core.extra_read_out, core.to_thinking]:
        for m in moments(projection.weight):
            assert torch.equal((m == 0).all(0), written["out_cols"]) and torch.equal((m == 1).all(0), ~written["out_cols"])
    for m in moments(core.gate):
        assert (m == 1).all()
    moe = _moe(model)[0][1]
    for slot in range(moe.n_routed):
        for p in moe.experts[slot].parameters():
            for m in moments(p):
                assert (m == 0).all() if slot == 4 else (m == 1).all()
    for m in moments(moe.gate.weight):
        assert (m[4] == 0).all() and (m[:4] == 1).all() and (m[5] == 1).all()
    for m in moments(model.embed_tokens.weight):
        assert (m == 1).all()
    assert float(optimiser.state[core.edge_logit]["step"]) == 3.0  # the step count is kept
    # a parameter the optimiser never touched (no state) is simply skipped
    fresh = torch.optim.AdamW(params, lr=1e-3)
    _dev_pass(controller, model, seed=12)
    assert controller.maybe_grow(1000, fresh)["moments_zeroed"] == 0


# ---------------------------------------------------------------------------
# D. apoptosis
# ---------------------------------------------------------------------------


def test_edge_apoptosis_needs_two_consecutive_weak_events(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    controller = _controller(model, tmp_path, grow_every=500, grow_edges=2)
    _dev_pass(controller, model)
    first = controller.maybe_grow(500, None)
    assert first["edges_opened"]["count"] == 2
    grown_post, grown_pre = torch.nonzero(core.edge_grown).tolist()[0]
    real_post, real_pre = torch.nonzero((core.mask > 0) & (core.edge_grown == 0)).tolist()[0]
    with torch.no_grad():
        core.edge_logit[grown_post, grown_pre] = -20.0   # strength 2e-9 < 1e-4
        core.edge_logit[real_post, real_pre] = -20.0
    _dev_pass(controller, model, seed=12)
    second = controller.maybe_grow(1000, None)
    assert second["edges_pruned"] == {"grown": 0, "real": 0} and second["edges_on_first_strike"] == 2
    assert float(core.mask[grown_post, grown_pre]) == 1 and float(core.mask[real_post, real_pre]) == 1
    _dev_pass(controller, model, seed=13)
    third = controller.maybe_grow(1500, None)
    assert third["edges_pruned"] == {"grown": 1, "real": 1} and third["edges_on_first_strike"] == 0
    # pruned, returned to the unmasked state, and not re-opened within the same event
    assert float(core.mask[grown_post, grown_pre]) == 0 and float(core.mask[real_post, real_pre]) == 0
    assert int(core.edge_grown[grown_post, grown_pre]) == 0 and float(core.init_fraction[real_post, real_pre]) == 0
    logits = core.edge_logit.detach()
    assert float(logits[grown_post, grown_pre]) == -10 and float(logits[real_post, real_pre]) == -10
    # an edge that recovers between two readings is never pruned (the strike resets)
    other_post, other_pre = torch.nonzero(core.edge_grown).tolist()[0]
    with torch.no_grad():
        core.edge_logit[other_post, other_pre] = -20.0
    _dev_pass(controller, model, seed=14)
    fourth = controller.maybe_grow(2000, None)
    assert fourth["edges_pruned"] == {"grown": 0, "real": 0} and fourth["edges_on_first_strike"] == 1
    with torch.no_grad():
        core.edge_logit[other_post, other_pre] = -7.0
    _dev_pass(controller, model, seed=15)
    fifth = controller.maybe_grow(2500, None)
    assert fifth["edges_pruned"] == {"grown": 0, "real": 0} and fifth["edges_on_first_strike"] == 0
    assert float(core.mask[other_post, other_pre]) == 1
    totals = controller.summary()["totals"]
    assert totals["edges_pruned_grown"] == 1 and totals["edges_pruned_real"] == 1


def test_inert_grown_module_is_killed_and_its_slot_reused(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    controller = _controller(model, tmp_path, grow_every=500, grow_modules=1)
    _dev_pass(controller, model)
    (parent, child), = controller.maybe_grow(500, None)["modules_split"]
    # cut every output path of the child: no outgoing edge, no efferent tap
    with torch.no_grad():
        core.mask[:, child] = 0.0
        core.edge_logit[:, child] = -10.0
        core.edge_grown[:, child] = 0
        core.out_mask[child] = 0.0
        core.read_out.weight[:, child] = 0.0
        core.to_thinking.weight[:, child] = 0.0
    _dev_pass(controller, model, seed=12)
    record = controller.maybe_grow(1000, None)
    assert record["modules_killed"] == [child]
    # the freed slot is the first free one, so this event's split (which runs
    # after apoptosis) reuses it: alive again, re-born at this step
    assert record["modules_split"] and record["modules_split"][0][1] == child
    assert int(core.alive[child]) == 1 and int(core.born_step[child]) == 1000
    assert record["alive_modules"] == N + 1
    # a real module is never killed, even when it is just as inert, and an
    # inert module is never chosen as a mitosis parent (its child would be inert too)
    real = int(torch.nonzero((core.born_step == 0) & (core.alive > 0)).flatten()[0])
    with torch.no_grad():
        core.mask[:, real] = 0.0
        core.out_mask[real] = 0.0
    _dev_pass(controller, model, seed=13)
    record = controller.maybe_grow(1500, None)
    assert record["modules_killed"] == [] and int(core.alive[real]) == 1
    assert all(parent != real for parent, _ in record["modules_split"])


def test_expert_apoptosis_never_drops_below_two_alive(tmp_path):
    model = _model(tmp_path, moe_top_k=1)
    _favour_experts(model, [0])  # top-1 routing: only expert 0 carries load
    moe = _moe(model)[0][1]
    controller = _controller(model, tmp_path, grow_every=500)
    _dev_pass(controller, model)
    first = controller.maybe_grow(500, None)
    assert first["experts_killed"] == [] and first["experts_on_first_strike"] == 3
    _dev_pass(controller, model, seed=12)
    second = controller.maybe_grow(1000, None)
    assert second["experts_killed"] == [[1, 1], [1, 2]]  # weakest first, index breaks the tie; the floor stops the third
    assert moe.alive_count() == 2 and moe.expert_alive.tolist() == [1, 0, 0, 1, 0, 0]
    _dev_pass(controller, model, seed=13)
    third = controller.maybe_grow(1500, None)
    assert third["experts_killed"] == [] and moe.alive_count() == 2
    assert controller.state_dict()["expert_weak"] == [[1, 3, 3]]


# ---------------------------------------------------------------------------
# E. synaptogenesis: the cross-hemisphere quota
# ---------------------------------------------------------------------------


def _eligible_pairs(core, stats):
    alive = core.alive.bool()
    eligible = alive.unsqueeze(1) & alive.unsqueeze(0) & (core.mask == 0) & (stats["rate_cov"] > 0)
    eligible &= ~torch.eye(core.n_nodes, dtype=torch.bool)
    return eligible


def test_cross_quota_is_honoured(tmp_path):
    model = _model(tmp_path, cns_io="all")
    core = model.cns_core
    controller = _controller(model, tmp_path, grow_every=500, grow_edges=4)  # cross_quota 0.5 -> 2 reserved
    _dev_pass(controller, model)
    stats = core.growth_statistics()
    eligible = _eligible_pairs(core, stats)
    side = core.hemisphere
    cross = side.unsqueeze(1) != side.unsqueeze(0)
    assert int((eligible & cross).sum()) >= 2 and int((eligible & ~cross).sum()) >= 2  # precondition
    mask_before = core.mask.clone()
    record = controller.maybe_grow(500, None)
    by_block = record["edges_opened"]["by_block"]
    assert record["edges_opened"]["count"] == 4 and by_block["LR"] + by_block["RL"] >= 2
    opened = (core.mask > 0) & (mask_before == 0)
    assert int(opened.sum()) == 4 and torch.equal(opened, core.edge_grown.bool())
    for post, pre in torch.nonzero(opened).tolist():
        block = ("L" if int(side[post]) == 0 else "R") + ("L" if int(side[pre]) == 0 else "R")
        assert by_block[block] >= 1
    # within each pool the opened edges are the strongest covariances available
    cov = stats["rate_cov"]
    for pool in (cross, ~cross):
        taken, left = opened & pool, eligible & pool & ~opened
        if taken.any() and left.any():
            assert float(cov[taken].min()) >= float(cov[left].max())
    assert record["edges_on_first_strike"] == 0

    # cross_quota >= 1 reserves every slot for the commissure (and never more than the quota)
    model_all = _model(tmp_path, seed=1, cns_io="all")
    controller_all = _controller(model_all, tmp_path, name="all", grow_every=500, grow_edges=2, cross_quota=1.5)
    _dev_pass(controller_all, model_all)
    opened_all = controller_all.maybe_grow(500, None)["edges_opened"]
    by_block = opened_all["by_block"]
    assert opened_all["count"] == 2 and int(model_all.cns_core.edge_grown.sum()) == 2
    assert by_block["LL"] == 0 and by_block["RR"] == 0 and by_block["LR"] + by_block["RL"] == 2

    # a one-sided graph (v91 file, no module_side) has no reservation: everything is LL
    model_one = _model(tmp_path, seed=2, sided=False, cns_io="all")
    assert int(model_one.cns_core.hemisphere.abs().sum()) == 0
    controller_one = _controller(model_one, tmp_path, name="one", grow_every=500, grow_edges=4)
    _dev_pass(controller_one, model_one)
    by_block = controller_one.maybe_grow(500, None)["edges_opened"]["by_block"]
    assert by_block == {"LL": 4, "RR": 0, "LR": 0, "RL": 0}


def test_taps_are_ranked_by_the_statistics(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    controller = _controller(model, tmp_path, grow_every=500, grow_taps=2)
    _dev_pass(controller, model)
    stats = core.growth_statistics()
    alive = core.alive.bool()
    in_score = torch.where(alive & (core.in_mask == 0), stats["resid_cov"], torch.full((CAP,), float("-inf")))
    out_score = torch.where(alive & (core.out_mask == 0), stats["mean_rate"], torch.full((CAP,), float("-inf")))
    expected_in = torch.topk(in_score, 2).indices.tolist()
    expected_out = torch.topk(out_score, 2).indices.tolist()
    record = controller.maybe_grow(500, None)
    assert record["taps_opened"] == {"in": 2, "out": 2}
    assert sorted(torch.nonzero(core.tap_grown[:, 0]).flatten().tolist()) == sorted(expected_in)
    assert sorted(torch.nonzero(core.tap_grown[:, 1]).flatten().tolist()) == sorted(expected_out)
    for module in expected_out:
        assert float(core.out_mask[module]) == 1 and core.read_out.weight[:, module].abs().sum() == 0
        assert core.to_thinking.weight[:, module].abs().sum() == 0


# ---------------------------------------------------------------------------
# F. state, disabled controller, determinism, settings
# ---------------------------------------------------------------------------


def test_state_dict_round_trip_continues_the_counters(tmp_path):
    model = _model(tmp_path)
    core = model.cns_core
    first = _controller(model, tmp_path, name="first", grow_every=500, grow_edges=1)
    _dev_pass(first, model)
    assert first.maybe_grow(500, None)["edges_opened"]["count"] == 1
    post, pre = torch.nonzero(core.edge_grown).tolist()[0]
    with torch.no_grad():
        core.edge_logit[post, pre] = -20.0
    _dev_pass(first, model, seed=12)
    assert first.maybe_grow(1000, None)["edges_pruned"]["grown"] == 0
    state = json.loads(json.dumps(first.state_dict()))  # JSON-safe, as extra['neurogenesis_state'] needs
    assert state["schema"] == "supermix-v93-neurogenesis-state-v1" and state["n_events"] == 2
    assert state["edge_weak"] == [[post, pre, 1]] and state["expert_weak"] == [] and len(state["events"]) == 2
    snapshot = copy.deepcopy(model)

    resumed = _controller(model, tmp_path, name="resumed", grow_every=500, grow_edges=1)
    resumed.load_state_dict(state)
    assert resumed.n_events == 2 and resumed.summary()["n_events"] == 2
    _dev_pass(resumed, model, seed=13)
    record = resumed.maybe_grow(1500, None)
    assert record["edges_pruned"]["grown"] == 1 and float(core.mask[post, pre]) == 0
    assert resumed.summary()["n_events"] == 3 and resumed.summary()["totals"]["edges_pruned_grown"] == 1
    assert resumed.state_dict()["edge_weak"] == []

    # without the restored memory the same event only records the first strike
    fresh = _controller(snapshot, tmp_path, name="fresh", grow_every=500, grow_edges=1)
    _dev_pass(fresh, snapshot, seed=13)
    record = fresh.maybe_grow(1500, None)
    assert record["edges_pruned"]["grown"] == 0 and float(snapshot.cns_core.mask[post, pre]) == 1
    assert record["edges_on_first_strike"] == 1
    # an empty or absent state is a no-op
    fresh.load_state_dict(None)
    fresh.load_state_dict({})
    assert fresh.n_events == 1


def test_no_event_when_disabled_or_off_schedule(tmp_path):
    model = _model(tmp_path)
    off = _controller(model, tmp_path, name="off")
    assert off.settings.grow_every == 0
    _dev_pass(off, model)
    assert off.maybe_grow(500, None) is None and off.maybe_grow(0, None) is None
    assert not os.path.exists(off.log_path)
    summary = off.summary()
    assert summary["n_events"] == 0 and summary["events"] == [] and summary["alive_modules"] is None
    assert summary["totals"]["edges_opened"] == 0
    scheduled = _controller(model, tmp_path, name="on", grow_every=500, grow_edges=1)
    assert scheduled.maybe_grow(250, None) is None and scheduled.maybe_grow(999, None) is None
    assert not os.path.exists(scheduled.log_path)
    assert int(model.cns_core.edge_grown.sum()) == 0


def test_model_without_core_or_spare_experts_records_zero_counts(tmp_path):
    torch.manual_seed(0)
    model = MiMoMixModel(_config(
        use_cns_core=False, moe_spare_experts=0, cns_read_layers=(), cns_write_layers=(), cns_to_thinking=False,
    )).eval()
    assert model.cns_core is None
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    controller = _controller(
        model, tmp_path, grow_every=500, grow_modules=2, grow_edges=4, grow_taps=1, grow_experts=True,
    )
    _dev_pass(controller, model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    record = controller.maybe_grow(500, optimiser)
    assert record["modules_split"] == [] and record["edges_opened"]["count"] == 0
    assert record["taps_opened"] == {"in": 0, "out": 0} and record["edges_pruned"] == {"grown": 0, "real": 0}
    assert record["experts_born"] == [] and record["experts_killed"] == [] and record["modules_killed"] == []
    assert record["alive_modules"] == 0 and record["alive_edges"] == 0 and record["alive_experts_per_layer"] == [4]
    assert record["witness_delta"] == 0.0 and record["flagged"] is False and record["moments_zeroed"] == 0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert controller.summary()["n_events"] == 1
    assert controller.state_dict()["edge_weak"] == []


def test_identical_models_produce_identical_events(tmp_path):
    records, states, controllers = [], [], []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        model = _model(tmp_path / name, seed=3)
        _favour_experts(model, [0, 1])
        controller = _controller(
            model, tmp_path, name=name, grow_every=500, grow_modules=2, grow_edges=4, grow_taps=1,
            grow_experts=True, expert_split_load_fraction=1.5,
        )
        _dev_pass(controller, model)
        record = controller.maybe_grow(500, None)
        # a second event after a second, different dev pass, so the strike
        # counters and the per-event noise seed both enter the comparison
        _dev_pass(controller, model, seed=12)
        second = controller.maybe_grow(1000, None)
        records.append([{k: v for k, v in r.items() if k != "seconds"} for r in (record, second)])
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        controllers.append(controller)
    assert records[0] == records[1]
    assert records[0][0]["experts_born"] == [[1, 0, 4]]
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key

    def strip(state):
        state = json.loads(json.dumps(state))
        for event in state["events"]:
            event.pop("seconds")
        return state

    assert strip(controllers[0].state_dict()) == strip(controllers[1].state_dict())


def test_build_settings_from_args():
    namespace = argparse.Namespace(
        grow_every=1000, grow_modules=3, grow_edges=40, grow_taps=2, grow_experts=True,
        prune_threshold=5e-5, witness_rows=16, seed=7, eval_every=500,
    )
    settings = build_settings_from_args(namespace)
    assert settings == GrowthSettings(
        grow_every=1000, grow_modules=3, grow_edges=40, grow_taps=2, grow_experts=True,
        prune_threshold=5e-5, witness_rows=16,
    )
    assert settings.seed == 93 and settings.cross_quota == 0.5 and settings.edge_logit == -7.0
    assert build_settings_from_args(argparse.Namespace()) == GrowthSettings()
    assert build_settings_from_args(argparse.Namespace(grow_every=None, grow_experts=0)).grow_experts is False
    assert GrowthSettings().grow_every == 0  # the default never grows
