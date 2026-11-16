"""Tests for Supermix v95 ConnectomeCore upgrades, Transactional Neurogenesis, and v93 Solvers."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import mimomix_core as mc
import nexus_solver
import prepare_v94
import prompt_normaliser
from v94_growth_policy import GrowthConfig, GrowthPolicy
from v95_neurogenesis_adapter import (
    CurriculumMixtureScheduler,
    GradientUtilityCollector,
    TransactionalNeurogenesisAdapter,
    evaluate_witness_rows,
)

SHA = "b" * 64


def _tiny_v95_config(**overrides) -> mc.MiMoMixConfig:
    base = dict(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_mtp_layers=0,
        use_cns_core=True,
        cns_nodes=8,
        cns_spare_nodes=2,
        cns_steps=2,
        cns_after_layer=0,
        cns_io="all",
    )
    base.update(overrides)
    return mc.MiMoMixConfig(**base)


def _init_connectome(model: mc.MiMoMixModel) -> None:
    core = model.cns_core
    assert core is not None
    with torch.no_grad():
        core.alive.zero_()
        core.alive[: core.n_nodes - core.spare_nodes] = 1
        core.hemisphere.zero_()
        core.hemisphere[: core.n_nodes // 2] = 0
        core.hemisphere[core.n_nodes // 2 :] = 1
        core.sign.copy_(torch.tensor([1, 1, -1, 1, 1, -1, 1, 1], dtype=core.sign.dtype))
        core.in_mask.fill_(1.0)
        core.out_mask.fill_(1.0)
        core.mask.zero_()
        for post in range(6):
            pre = (post + 1) % 6
            core.mask[post, pre] = 1.0
        core.edge_logit.fill_(-4.0)
        core.edge_logit[core.mask > 0] = 0.0
        core.gate.fill_(0.35)


def test_v95_config_validation_and_default_off_bit_identity():
    with pytest.raises(ValueError, match="cns_slow_leak_init"):
        _tiny_v95_config(cns_slow_leak_init=1.0)
    with pytest.raises(ValueError, match="cns_trunk_dropout"):
        _tiny_v95_config(cns_trunk_dropout=1.0)

    torch.manual_seed(7)
    cfg_a = _tiny_v95_config()
    model_a = mc.MiMoMixModel(cfg_a)
    torch.manual_seed(7)
    cfg_b = _tiny_v95_config(
        cns_dual_soma=False,
        cns_commissural_gate=False,
        cns_glial_homeostasis=False,
        cns_trunk_dropout=0.0,
    )
    model_b = mc.MiMoMixModel(cfg_b)

    assert list(model_a.state_dict().keys()) == list(model_b.state_dict().keys())
    x = torch.randint(0, 64, (2, 6))
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x, return_mtp=False)
        out_b = model_b(x, return_mtp=False)
    assert torch.equal(out_a.logits, out_b.logits)


def test_connectome_dual_soma_commissural_gate_and_glial_homeostasis():
    torch.manual_seed(11)
    cfg = _tiny_v95_config(
        cns_dual_soma=True,
        cns_slow_leak_init=0.1,
        cns_commissural_gate=True,
        cns_glial_homeostasis=True,
        cns_trunk_dropout=0.2,
    )
    model = mc.MiMoMixModel(cfg)
    _init_connectome(model)
    core = model.cns_core
    assert core is not None
    assert core.slow_leak_logit is not None
    assert core.slow_coupling is not None
    assert core.commissural_gate_proj is not None
    assert core.glial_strength is not None
    assert core.glial_activity_ema is not None

    # Dale sign preservation on weight blocks
    w_same_t, w_lr_t, w_rl_t = core._split_commissural_weights(core.weight())
    w_combined = (w_same_t + w_lr_t + w_rl_t).t()
    for pre in range(core.n_nodes):
        s_pre = float(core.sign[pre].item())
        col = w_combined[:, pre]
        if s_pre > 0:
            assert bool((col >= -1e-7).all())
        else:
            assert bool((col <= 1e-7).all())

    # Commisural gate initializes near sigmoid(+4.0) ~= 0.982
    assert float(torch.sigmoid(core.commissural_gate_proj.bias).mean().item()) > 0.97

    # Glial EMA updates under train(), stays fixed under eval()
    ema_before = core.glial_activity_ema.clone()
    x = torch.randint(0, 64, (2, 5))
    y = torch.randint(0, 64, (2, 5))
    model.train()
    out_train = model(x, labels=y, return_mtp=False)
    out_train.loss.backward()
    ema_after_train = core.glial_activity_ema.clone()
    assert not torch.allclose(ema_before, ema_after_train)
    assert core.slow_coupling.grad is not None
    assert core.commissural_gate_proj.weight.grad is not None

    model.eval()
    with torch.no_grad():
        out_eval_1 = model(x, return_mtp=False)
        out_eval_2 = model(x, return_mtp=False)
    assert torch.allclose(core.glial_activity_ema, ema_after_train)
    assert torch.equal(out_eval_1.logits, out_eval_2.logits)

    telem = core.telemetry()
    assert telem["dual_soma"] is True
    assert telem["commissural_gate"] is True
    assert telem["glial_homeostasis"] is True


def test_probe_signed_edge_gradients_and_grow_edges_from_scores():
    torch.manual_seed(17)
    cfg = _tiny_v95_config()
    model = mc.MiMoMixModel(cfg)
    _init_connectome(model)
    core = model.cns_core
    assert core is not None

    # Excitatory pre=1 (sign=+1), negative post_grad at post=4 -> positive utility
    pre_act = torch.ones((1, 8))
    post_grad = torch.zeros((1, 8))
    post_grad[0, 4] = -2.0
    post_grad[0, 5] = 2.0  # matches inhibitory pre=2 (sign=-1) -> +2.0 utility

    scores = core.probe_signed_edge_gradients(pre_act, post_grad)
    assert float(scores[4, 1].item()) == pytest.approx(2.0)
    assert float(scores[5, 2].item()) == pytest.approx(2.0)
    # Dead slots (6, 7) and self-loops must remain zero
    assert float(scores[6, :].sum().item()) == 0.0
    assert float(scores[:, 7].sum().item()) == 0.0
    assert float(scores.diag().sum().item()) == 0.0

    before_edges = int(core.mask.sum().item())
    installed = core.grow_edges_from_scores(scores, count=2, step=12, logit=-6.0, cross_quota=0.5)
    assert len(installed) == 2
    assert int(core.mask.sum().item()) == before_edges + 2
    # Cross-hemisphere quota reserved at least 1 bilateral edge: (4, 1) is R<-L
    assert any(
        int(core.hemisphere[p].item()) != int(core.hemisphere[q].item())
        for p, q in installed
    )


def test_transactional_neurogenesis_adapter_commit_and_rollback():
    torch.manual_seed(23)
    cfg = _tiny_v95_config(cns_dual_soma=True)
    model = mc.MiMoMixModel(cfg)
    _init_connectome(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))
    y = torch.randint(0, 64, (2, 8))

    # Populate optimizer state once so rollback verifies moment restoration
    out0 = model(x, labels=y, return_mtp=False)
    out0.loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

    # Real witness row evaluation helper works cleanly
    real_witness = evaluate_witness_rows(model, [("r0", "arith", x, y)])
    assert len(real_witness) == 1 and real_witness[0]["token_count"] > 0

    # Test GradientUtilityCollector directly
    collector = GradientUtilityCollector(model)
    obs, edge_scores = collector.collect_connectome_utilities([(x, y)], step=1)
    assert len(obs) == 8
    assert tuple(edge_scores.shape) == (8, 8)
    moe_obs = collector.collect_moe_utilities(model)
    assert len(moe_obs) >= 1

    # 1. Accepted transaction (witness shows 0 loss change)
    gcfg = GrowthConfig(
        capacity=8, max_alive=8, min_alive=6, grow_above=1e-9, prune_below=0.0, ema_decay=0.0, maturity_steps=10
    )
    policy = GrowthPolicy(gcfg, SHA)

    def passing_witness(_m):
        return [{"row_id": "w0", "family": "arith", "loss_sum": 1.0, "token_count": 10}]

    adapter = TransactionalNeurogenesisAdapter(
        model,
        policy,
        membership_sha256=SHA,
        total_steps=100,
        optimiser=opt,
        eval_witness_fn=passing_witness,
        edge_quota_per_event=1,
    )
    receipt_ok = adapter.step_neurogenesis(20, [(x, y)])
    assert receipt_ok.witness_accepted is True
    assert receipt_ok.rolled_back is False
    assert adapter.commits == 1
    assert receipt_ok.alive_modules_after == 7

    # 2. Rejected transaction -> exact rollback
    call_counter = {"n": 0}

    def failing_witness(_m):
        call_counter["n"] += 1
        # Second call (after mutation) increases loss beyond max_delta_nats=0.001
        loss_val = 1.0 if call_counter["n"] % 2 == 1 else 2.5
        return [{"row_id": "w0", "family": "arith", "loss_sum": loss_val, "token_count": 10}]

    adapter_rb = TransactionalNeurogenesisAdapter(
        model,
        adapter.policy,
        membership_sha256=SHA,
        total_steps=100,
        optimiser=opt,
        eval_witness_fn=failing_witness,
        edge_quota_per_event=1,
        max_delta_nats=0.001,
    )
    state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    policy_before = adapter_rb.policy.state_dict()
    receipt_rb = adapter_rb.step_neurogenesis(30, [(x, y)])

    assert receipt_rb.witness_accepted is False
    assert receipt_rb.rolled_back is True
    assert "arith" in receipt_rb.regressed_families
    assert adapter_rb.rollbacks == 1
    for key, tensor_before in state_before.items():
        assert torch.equal(model.state_dict()[key], tensor_before), f"mismatch after rollback on {key}"
    assert adapter_rb.policy.state_dict() == policy_before


def test_curriculum_scheduler_and_v95_readiness():
    sched = CurriculumMixtureScheduler(
        {"v89_anchor": 0.9, "v93_code": 0.1},
        {"v89_anchor": 0.4, "v93_code": 0.6},
        warmup_steps=10,
        ramp_steps=40,
        schedule="cosine",
    )
    assert sched.blend_alpha(10) == 0.0
    assert sched.blend_alpha(30) == pytest.approx(0.5)
    assert sched.blend_alpha(50) == 1.0
    w_mid = sched.weights_at(30)
    assert sum(w_mid.values()) == pytest.approx(1.0)
    sched.record_tokens({"v89_anchor": 650, "v93_code": 350})
    rep = sched.quota_report(30)
    assert rep["total_tokens"] == 1000

    ready = prepare_v94.check_v95_readiness()
    assert ready["all_ready"] is True
    assert ready["transactional_adapter_installed"] is True
    assert ready["ecosystem_installed"] is True


def test_prompt_normaliser_and_solver_cover_all_11_v93_tasks():
    engine = nexus_solver.NexusSolver()

    # 1. impulse
    norm_imp = prompt_normaliser.normalise("What is the impulse for force 12 N acting over time 4 s?")
    assert norm_imp is not None and norm_imp.rule == "impulse"
    res_imp = engine.solve(norm_imp.prompt)
    assert res_imp and res_imp.solved and res_imp.display_answer == "48"

    # 2. ohms_current
    norm_ohm = prompt_normaliser.normalise("Given voltage 24 V and resistance 6 ohm, compute the current")
    assert norm_ohm is not None and norm_ohm.rule == "ohms_current"
    res_ohm = engine.solve(norm_ohm.prompt)
    assert res_ohm and res_ohm.solved and res_ohm.display_answer == "4"

    # 3. spring_energy
    norm_spr = prompt_normaliser.normalise("Compute the spring energy for spring constant 200 N/m and extension 0.5 m")
    assert norm_spr is not None and norm_spr.rule == "spring_energy"
    res_spr = engine.solve(norm_spr.prompt)
    assert res_spr and res_spr.solved and res_spr.display_answer == "25"

    # 4. permutations
    norm_perm = prompt_normaliser.normalise("How many permutations of 5 items taken 3 at a time?")
    assert norm_perm is not None and norm_perm.rule == "permutations"
    res_perm = engine.solve(norm_perm.prompt)
    assert res_perm and res_perm.solved and res_perm.display_answer == "60"

    # 5. final_velocity (v = u + a*t)
    norm_vel = prompt_normaliser.normalise("Find the final velocity from initial velocity 5 m/s, acceleration 3 m/s^2 and time 4 s")
    assert norm_vel is not None and norm_vel.rule == "final_velocity"
    res_vel = engine.solve(norm_vel.prompt)
    assert res_vel and res_vel.solved and res_vel.display_answer == "17"

    # 6. code_range_sum
    norm_cr = prompt_normaliser.normalise("What is the sum of range(2, 6)?")
    assert norm_cr is not None and norm_cr.rule == "code_range_sum"
    res_cr = engine.solve(norm_cr.prompt)
    assert res_cr and res_cr.solved and res_cr.display_answer == "14"

    # 7. code_list_count
    norm_cc = prompt_normaliser.normalise("How many times does 3 appear in [1, 3, 2, 3, 3]?")
    assert norm_cc is not None and norm_cc.rule == "code_list_count"
    res_cc = engine.solve(norm_cc.prompt)
    assert res_cc and res_cc.solved and res_cc.display_answer == "3"

    # 8. code_neg_index
    norm_cn = prompt_normaliser.normalise("What is the element at index -2 of [10, 20, 30, 40]?")
    assert norm_cn is not None and norm_cn.rule == "code_neg_index"
    res_cn = engine.solve(norm_cn.prompt)
    assert res_cn and res_cn.solved and res_cn.display_answer == "30"

    # 9-11. Male CNS connectome canonical normalization
    norm_c1 = prompt_normaliser.normalise("male CNS neuron count for cell type KCg")
    assert norm_c1 is not None and norm_c1.rule == "cns_type_count"
    norm_c2 = prompt_normaliser.normalise("male CNS left side neuron count for type L1")
    assert norm_c2 is not None and norm_c2.rule == "cns_side_count"
    norm_c3 = prompt_normaliser.normalise("synapse count from cell type L1 to cell type Tm3 in the male CNS")
    assert norm_c3 is not None and norm_c3.rule == "cns_pair_synapses"
