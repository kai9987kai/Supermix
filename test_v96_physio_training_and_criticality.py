"""v96 physiological criticality, transactional NeurogenesisController, trunk-freeze,
v92 physiological reservoir, and stratified repair corpus tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import build_v96_repair_mix as v96_mix  # noqa: E402
import mimomix_core as mc  # noqa: E402
import neurogenesis as ng  # noqa: E402
import train_mimomix_generalisation as trainer  # noqa: E402
import train_mimomix_talk as talk  # noqa: E402
import v92_temporal_connectome as v92  # noqa: E402


def _synthetic_hemisphere_npz(path: Path, n: int = 8, seed: int = 4) -> str:
    rng = np.random.default_rng(seed)
    roles = np.array(["sensory", "central", "central", "descending"] * 2, dtype=object)
    sign = np.array([1, -1, 1, 1, 1, -1, 1, 1], dtype=np.int8)
    matrix = rng.random((n, n)) * (rng.random((n, n)) < 0.35)
    np.fill_diagonal(matrix, 0.0)
    for post in range(n):
        if matrix[post].sum() == 0:
            matrix[post, (post + 1) % n] = 0.4
    post, pre = np.nonzero(matrix)
    fraction = matrix[post, pre] / matrix.sum(1)[post]
    np.savez_compressed(
        path,
        module_role=roles,
        module_sign=sign,
        edge_post=post.astype(np.int64),
        edge_pre=pre.astype(np.int64),
        edge_fraction=fraction.astype(np.float32),
        rewired_post=post.astype(np.int64),
        rewired_pre=pre.astype(np.int64),
        module_side=np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8),
    )
    return str(path)


def _tiny_cns_model(tmp_path: Path, **overrides) -> mc.MiMoMixModel:
    torch.manual_seed(11)
    base = dict(
        vocab_size=64,
        hidden_size=24,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=36,
        moe_intermediate_size=16,
        n_routed_experts=4,
        moe_spare_experts=2,
        moe_top_k=2,
        n_mtp_layers=1,
        sliding_window=8,
        native_context=32,
        max_position_embeddings=32,
        thinking_latent_dim=8,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=4,
        cns_after_layer=1,
        cns_steps=2,
        cns_read_layers=(0, 1),
        cns_write_layers=(1,),
        cns_to_thinking=True,
        cns_temporal=True,
    )
    base.update(overrides)
    model = mc.MiMoMixModel(mc.MiMoMixConfig(**base))
    model.cns_core.load_graph(_synthetic_hemisphere_npz(tmp_path / "hemi.npz"))
    with torch.no_grad():
        model.cns_core.gate.fill_(0.25)
        for gate in model.cns_core.extra_gates:
            gate.fill_(0.25)
        if model.cns_core.thinking_gate is not None:
            model.cns_core.thinking_gate.fill_(0.25)
        model.cns_core.read_in.weight[:8].normal_(0.0, 0.25)
        model.cns_core.read_out.weight[:, :8].normal_(0.0, 0.25)
    return model


def _dev_pass(controller: ng.NeurogenesisController, model: mc.MiMoMixModel, vocab: int = 64) -> None:
    controller.begin_dev_pass()
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, vocab, (4, 10))
        model(x, return_mtp=False)
    controller.end_dev_pass()


def test_spectral_criticality_homeostasis_and_telemetry(tmp_path: Path):
    model = _tiny_cns_model(tmp_path, cns_homeostatic_radius=0.85)
    core = model.cns_core
    rho_0 = core.estimate_spectral_radius(power_iters=10)
    assert rho_0 > 0.0
    # Inflate edges deliberately and check projection moves smoothly toward target_radius
    with torch.no_grad():
        core.edge_logit.copy_(torch.where(core.mask > 0, torch.full_like(core.edge_logit, 1.5), core.edge_logit))
    rho_high = core.estimate_spectral_radius(power_iters=10)
    assert rho_high > 1.2
    proj1 = core.project_spectral_radius(target_radius=0.85, power_iters=10, max_scale_step=0.25)
    assert proj1["scale"] == pytest.approx(0.75)
    assert proj1["rho_after"] < rho_high
    for _ in range(6):
        proj_k = core.project_spectral_radius(target_radius=0.85, power_iters=12, max_scale_step=0.25)
    assert proj_k["rho_after"] == pytest.approx(0.85, abs=0.05)

    telem = core.telemetry()
    assert telem["homeostatic_radius"] == pytest.approx(0.85)
    assert telem["spectral_radius_est"] == pytest.approx(proj_k["rho_after"])

    # Empty mask edge case returns 0.0 cleanly
    with torch.no_grad():
        core.mask.zero_()
    assert core.estimate_spectral_radius() == 0.0


def test_train_mimomix_talk_wires_v95_and_v96_cns_flags():
    parser = trainer.build_parser()
    default_args = parser.parse_args([])
    default_cfg = talk.build_config(default_args, 256)
    assert default_cfg.cns_dual_soma is False
    assert default_cfg.cns_slow_leak_init == pytest.approx(0.05)
    assert default_cfg.cns_commissural_gate is False
    assert default_cfg.cns_glial_homeostasis is False
    assert default_cfg.cns_trunk_dropout == pytest.approx(0.0)
    assert default_cfg.cns_homeostatic_radius == pytest.approx(0.0)

    args = parser.parse_args([
        "--cns_core",
        "--cns_dual_soma",
        "--cns_slow_leak_init", "0.12",
        "--cns_commissural_gate",
        "--cns_glial_homeostasis",
        "--cns_trunk_dropout", "0.2",
        "--cns_homeostatic_radius", "0.9",
        "--freeze_trunk_steps", "10",
        "--witness_rollback_tolerance", "0.02",
        "--maturity_steps", "50",
        "--max_splits_per_parent", "2",
        "--edge_proposal_mode", "hybrid",
    ])
    cfg = talk.build_config(args, 256)
    assert cfg.cns_dual_soma is True
    assert cfg.cns_slow_leak_init == pytest.approx(0.12)
    assert cfg.cns_commissural_gate is True
    assert cfg.cns_glial_homeostasis is True
    assert cfg.cns_trunk_dropout == pytest.approx(0.2)
    assert cfg.cns_homeostatic_radius == pytest.approx(0.9)

    recorded = talk.recorded_hyperparameters(args)
    assert recorded["cns_dual_soma"] is True
    assert recorded["cns_homeostatic_radius"] == pytest.approx(0.9)
    assert recorded["freeze_trunk_steps"] == 10
    assert recorded["edge_proposal_mode"] == "hybrid"

    trainer.validate_growth_settings(args)
    with pytest.raises(SystemExit):
        trainer.validate_growth_settings(parser.parse_args(["--freeze_trunk_steps", "-1"]))


def test_neurogenesis_controller_rollback_maturity_split_cap_and_hybrid_edges(tmp_path: Path):
    torch.manual_seed(7)
    x = torch.randint(0, 64, (4, 10))
    y = x.clone()
    y[:, :2] = -100

    # 1. Witness rollback restores model, optimizer moments, and controller state atomically
    model_rb = _tiny_cns_model(tmp_path, cns_dual_soma=True)
    opt_rb = torch.optim.AdamW(model_rb.parameters(), lr=1e-3)
    out = model_rb(x, labels=y)
    out.loss.backward()
    opt_rb.step()
    opt_rb.zero_grad(set_to_none=True)

    snap_edge_logit = model_rb.cns_core.edge_logit.detach().clone()
    snap_alive = model_rb.cns_core.alive.detach().clone()
    snap_moment = opt_rb.state[model_rb.cns_core.read_in.weight]["exp_avg"].detach().clone()

    # Negative tolerance guarantees rollback on nonzero positive perturbation or tiny noise; use -1.0 for determinism
    settings_rb = ng.GrowthSettings(
        grow_every=10,
        grow_modules=2,
        grow_edges=4,
        witness_rollback_tolerance=-1.0,
        homeostatic_radius=0.85,
    )
    ctrl_rb = ng.NeurogenesisController(model_rb, settings_rb, str(tmp_path / "rb.jsonl"), (x, y))
    _dev_pass(ctrl_rb, model_rb)
    rec_rb = ctrl_rb.maybe_grow(10, opt_rb)
    assert rec_rb["rolled_back"] is True
    assert rec_rb["moments_zeroed"] == 0
    assert ctrl_rb.rolled_back_events == [10]
    assert torch.equal(model_rb.cns_core.edge_logit, snap_edge_logit)
    assert torch.equal(model_rb.cns_core.alive, snap_alive)
    assert torch.equal(opt_rb.state[model_rb.cns_core.read_in.weight]["exp_avg"], snap_moment)
    assert ctrl_rb._parent_splits == {}

    # 2. Maturity guard + max_splits_per_parent + signed_gradient / hybrid synaptogenesis + gradient restoration
    model = _tiny_cns_model(tmp_path, cns_dual_soma=True, cns_homeostatic_radius=0.88)
    sentinel_grad = torch.full_like(model.embed_tokens.weight, 42.0)
    model.embed_tokens.weight.grad = sentinel_grad.clone()

    settings = ng.GrowthSettings(
        grow_every=10,
        grow_modules=1,
        grow_edges=4,
        maturity_steps=25,
        max_splits_per_parent=1,
        edge_proposal_mode="hybrid",
        witness_rollback_tolerance=1.0,
        homeostatic_radius=0.88,
    )
    ctrl = ng.NeurogenesisController(model, settings, str(tmp_path / "hybrid.jsonl"), (x, y))
    _dev_pass(ctrl, model)
    r1 = ctrl.maybe_grow(10, None)
    assert r1["rolled_back"] is False
    assert r1["edge_proposal_mode"] == "hybrid"
    assert r1["spectral_projection"] is not None
    assert len(r1["modules_split"]) == 1
    first_parent, first_child = r1["modules_split"][0]
    assert r1["edges_opened"]["count"] >= 1
    # Sentinel gradient preserved untouched
    assert torch.equal(model.embed_tokens.weight.grad, sentinel_grad)

    # Make first_child inert; verify maturity_steps=25 prevents premature pruning at step 20, allows pruning at step 40
    with torch.no_grad():
        model.cns_core.mask[:, first_child] = 0.0
        model.cns_core.out_mask[first_child] = 0.0
    _dev_pass(ctrl, model)
    r2 = ctrl.maybe_grow(20, None)
    assert first_child not in r2["modules_killed"]
    # first_parent cannot be split again (max_splits_per_parent=1)
    assert all(p != first_parent for p, _ in r2["modules_split"])

    # Keep first_child inert before step 40 (synaptogenesis at step 20 may have reconnected it)
    with torch.no_grad():
        model.cns_core.mask[:, first_child] = 0.0
        model.cns_core.out_mask[first_child] = 0.0
    _dev_pass(ctrl, model)
    r3 = ctrl.maybe_grow(40, None)
    assert first_child in r3["modules_killed"]

    # Check state_dict round-trip of parent_splits & rolled_back_events
    state = json.loads(json.dumps(ctrl.state_dict()))
    ctrl2 = ng.NeurogenesisController(model, settings, str(tmp_path / "hybrid2.jsonl"), (x, y))
    ctrl2.load_state_dict(state)
    assert ctrl2._parent_splits == ctrl._parent_splits
    assert ctrl2.rolled_back_events == ctrl.rolled_back_events

    # Signed-gradient standalone mode also runs cleanly
    settings_sg = ng.GrowthSettings(grow_every=10, grow_edges=2, edge_proposal_mode="signed_gradient")
    ctrl_sg = ng.NeurogenesisController(model, settings_sg, str(tmp_path / "sg.jsonl"), (x, y))
    _dev_pass(ctrl_sg, model)
    r_sg = ctrl_sg.maybe_grow(50, None)
    assert r_sg["edge_proposal_mode"] == "signed_gradient"


def test_freeze_trunk_steps_grad_filtering_and_trainer_cli(tmp_path: Path):
    model = _tiny_cns_model(tmp_path)
    prefixes = trainer.new_parameter_prefixes(model.config)
    assert "cns_core." in prefixes

    x = torch.randint(0, 64, (2, 10))
    y = x.clone()
    y[:, :2] = -100
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trunk_before = model.embed_tokens.weight.detach().clone()
    cns_before = model.cns_core.read_out.weight.detach().clone()

    out = model(x, labels=y)
    out.loss.backward()
    for name, param in model.named_parameters():
        if not name.startswith(prefixes):
            param.grad = None
    opt.step()

    assert torch.equal(model.embed_tokens.weight, trunk_before)
    assert not torch.equal(model.cns_core.read_out.weight, cns_before)


def test_v92_temporal_connectome_physiological_mode():
    rng = np.random.default_rng(5)
    n = 16
    roles = ["sensory"] * 3 + ["central"] * 10 + ["output"] * 3
    sign = np.where(rng.random(n) < 0.6, 1.0, -1.0)
    matrix = rng.random((n, n)) * (rng.random((n, n)) < 0.3)
    np.fill_diagonal(matrix, 0.0)
    for post_i in range(n):
        if matrix[post_i].sum() == 0:
            matrix[post_i, (post_i + 1) % n] = 0.5
    post, pre = np.nonzero(matrix)
    frac = matrix[post, pre] / matrix.sum(1)[post]

    for temporal in (True, False):
        torch.manual_seed(0)
        core = v92.TemporalConnectomeCore(
            16,
            n,
            inner_steps=2,
            temporal=temporal,
            dual_soma=True,
            commissural_gate=True,
            glial_homeostasis=True,
        )
        core.install(post, pre, frac, sign, roles)
        x = torch.randn(2, 8, 16)
        core.eval()
        with torch.no_grad():
            assert torch.equal(core(x), x)
            core.gate.fill_(0.3)
            core.slow_coupling.fill_(0.2)
            ema_before = core.glial_activity_ema.clone()
            y1 = core(x)
            assert torch.equal(core.glial_activity_ema, ema_before)

        core.train()
        _ = core(x)
        assert not torch.equal(core.glial_activity_ema, ema_before)

        if temporal:
            core.eval()
            x2 = x.clone()
            x2[:, 5] += 2.5
            with torch.no_grad():
                out_a = core(x)
                out_b = core(x2)
            assert torch.allclose(out_a[:, :5], out_b[:, :5])
            assert not torch.allclose(out_a[:, 6:], out_b[:, 6:])
            assert float(y1.abs().sum()) > 0.0

    assert len(v92.PHYSIO_PLAN) == 2
    assert v92.PHYSIO_PLAN[0]["name"] == "T_physio_s1"


def test_build_v96_repair_mix_on_real_schema(tmp_path: Path):
    out_jsonl = tmp_path / "mix.jsonl"
    out_manifest = tmp_path / "manifest.json"
    manifest = v96_mix.build_v96_repair_mix(out_jsonl, out_manifest, seed=96)
    assert manifest["total_rows"] == 3200
    lines = out_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3200
    sample = json.loads(lines[0])
    assert {"user", "assistant", "task"}.issubset(sample.keys())
