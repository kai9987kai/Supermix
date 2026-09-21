"""v91: the male-CNS connectome core and its preprocessing.

Everything here runs on small synthetic graphs, so it needs neither the 1 GB
Janelia download nor a trained checkpoint.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import malecns_connectome as mc  # noqa: E402
from mimomix_core import ConnectomeCore, MiMoMixConfig, MiMoMixModel  # noqa: E402
from train_mimomix_generalisation import split_new_parameter_groups  # noqa: E402
from train_mimomix_talk import parameter_groups  # noqa: E402

N = 24


def _small_config(**overrides):
    base = dict(
        vocab_size=97, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2,
        intermediate_size=48, moe_intermediate_size=16, n_routed_experts=4,
        moe_top_k=2, n_mtp_layers=1, sliding_window=8, native_context=32,
        max_position_embeddings=32, thinking_latent_dim=8,
    )
    base.update(overrides)
    return MiMoMixConfig(**base)


def _graph_npz(tmp_path, seed=0):
    """A random module graph in the exact layout `build_modules` writes."""

    rng = np.random.default_rng(seed)
    roles = np.array(["sensory"] * 4 + ["central"] * 14 + ["descending"] * 3 + ["output"] * 3, dtype=object)
    sign = np.where(rng.random(N) < 0.6, 1, -1).astype(np.int8)
    matrix = rng.random((N, N)) * (rng.random((N, N)) < 0.3)
    np.fill_diagonal(matrix, rng.random(N) * (rng.random(N) < 0.5))
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.01)
    r_post, r_pre, _ = mc.degree_preserving_rewire(post, pre, N, seed=1)
    path = tmp_path / "modules.npz"
    np.savez_compressed(
        path, module_role=roles, module_sign=sign, edge_post=post, edge_pre=pre,
        edge_fraction=fraction.astype(np.float32), rewired_post=r_post, rewired_pre=r_pre,
    )
    return str(path), post, pre


def test_rewire_preserves_degrees_self_loops_and_simplicity():
    rng = np.random.default_rng(3)
    matrix = rng.random((40, 40)) * (rng.random((40, 40)) < 0.2)
    np.fill_diagonal(matrix, 1.0)
    post, pre, _ = mc.threshold_input_fraction(matrix, 0.0001)
    r_post, r_pre, info = mc.degree_preserving_rewire(post, pre, 40, seed=0)
    assert np.array_equal(np.bincount(post, minlength=40), np.bincount(r_post, minlength=40))
    assert np.array_equal(np.bincount(pre, minlength=40), np.bincount(r_pre, minlength=40))
    assert (post == pre).sum() == (r_post == r_pre).sum() == info["self_loops_held"]
    assert len(set(zip(r_post, r_pre))) == len(r_post)  # no multi-edges
    assert info["swaps_accepted"] > 0
    # the wiring actually changed
    assert len(set(zip(post, pre)) & set(zip(r_post, r_pre))) < len(post)


def test_stratified_null_keeps_every_input_property_but_the_source():
    rng = np.random.default_rng(5)
    n = 60
    strata = rng.integers(0, 4, size=n)
    matrix = rng.random((n, n)) * (rng.random((n, n)) < 0.25)
    np.fill_diagonal(matrix, 1.0)
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.001)
    r_post, r_pre, r_fraction, info = mc.stratified_rewire(post, pre, fraction, strata, n, seed=2)
    assert info["swaps_accepted"] > 0
    real = np.zeros((n, n)); real[post, pre] = fraction
    null = np.zeros((n, n)); null[r_post, r_pre] = r_fraction
    # degrees, self-loops, simplicity
    assert np.array_equal((real > 0).sum(0), (null > 0).sum(0))
    assert np.array_equal((real > 0).sum(1), (null > 0).sum(1))
    assert np.array_equal(np.diag(real), np.diag(null))
    assert len(set(zip(r_post, r_pre))) == len(r_post)
    # each module keeps its exact input fractions and the strata they come from
    assert np.allclose(np.sort(real, axis=1).sum(1), np.sort(null, axis=1).sum(1))
    for s in range(4):
        cols = strata == s
        assert np.array_equal((real[:, cols] > 0).sum(1), (null[:, cols] > 0).sum(1))
    assert not np.array_equal(real > 0, null > 0)


def test_mean_hops_walks_the_synaptic_direction():
    # chain 0 -> 1 -> 2: sensory module 0 reaches output module 2 in 2 hops
    roles = np.array(["sensory", "central", "output"])
    post, pre = np.array([1, 2]), np.array([0, 1])
    hops = mc.mean_hops(post, pre, roles, 3, ("sensory",), ("output",))
    assert hops == {"mean": 2.0, "reachable_fraction": 1.0}
    backwards = mc.mean_hops(pre, post, roles, 3, ("sensory",), ("output",))
    assert backwards["reachable_fraction"] == 0.0


def test_load_graph_uses_the_null_fractions(tmp_path):
    path, post, pre = _graph_npz(tmp_path)
    data = dict(np.load(path, allow_pickle=True))
    data["rewired_fraction"] = np.full(len(post), 0.05, dtype=np.float32)
    np.savez_compressed(path, **data)
    core = ConnectomeCore(_small_config(use_cns_core=True, cns_nodes=N))
    core.load_graph(path, "rewired")
    installed = core.init_fraction[core.mask > 0]
    assert torch.allclose(installed, torch.full_like(installed, 0.05))
    core.load_graph(path, "connectome")  # the real wiring ignores the null's fractions
    assert not torch.allclose(core.init_fraction[core.mask > 0], torch.tensor(0.05, dtype=torch.float32))


def test_input_fraction_threshold_keeps_only_strong_inputs():
    matrix = np.array([[0.0, 99.0, 1.0], [5.0, 0.0, 5.0], [0.0, 0.0, 0.0]])
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.05)
    kept = set(zip(post.tolist(), pre.tolist()))
    assert kept == {(0, 1), (1, 0), (1, 2)}  # 1% of row 0 is dropped; empty row 2 keeps nothing
    assert np.allclose(fraction[post == 1], 0.5)


def test_allocate_respects_minimum_and_caps():
    counts = mc._allocate(np.array([100.0, 1.0, 10.0]), 20, np.array([50, 3, 2]))
    assert counts.sum() == 20 and (counts >= 1).all() and (counts <= [50, 3, 2]).all()
    with pytest.raises(ValueError):
        mc._allocate(np.ones(5), 3, np.full(5, 10))


def test_graft_is_function_preserving_at_zero_gate(tmp_path):
    torch.manual_seed(0)
    base = MiMoMixModel(_small_config()).eval()
    grafted = MiMoMixModel(_small_config(use_cns_core=True, cns_nodes=N, cns_after_layer=1)).eval()
    missing, unexpected = grafted.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("cns_core.") for k in missing)
    path, _, _ = _graph_npz(tmp_path)
    grafted.cns_core.load_graph(path)
    x = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        assert torch.equal(base(x).logits, grafted(x).logits)
        grafted.cns_core.gate.fill_(0.5)
        assert not torch.equal(base(x).logits, grafted(x).logits)


def test_load_graph_signs_mask_radius_and_io(tmp_path):
    path, post, pre = _graph_npz(tmp_path)
    core = ConnectomeCore(_small_config(use_cns_core=True, cns_nodes=N))
    info = core.load_graph(path, spectral_radius=0.8)
    w = core.weight().detach()
    assert int((w != 0).sum()) == len(post) == info["edges"]
    # Dale's law: every column carries its presynaptic module's sign
    signs = core.sign.numpy()
    for col in range(N):
        column = w[:, col].numpy()
        nz = column[column != 0]
        assert nz.size == 0 or np.all(np.sign(nz) == signs[col])
    rho = np.abs(np.linalg.eigvals(w.abs().double().numpy())).max()
    assert rho == pytest.approx(0.8, rel=1e-4)
    assert info["input_nodes"] == 4 and info["output_nodes"] == 6
    assert float(core.gate.abs().sum()) == 0.0


def test_rewired_wiring_matches_degrees_and_regime(tmp_path):
    path, _, _ = _graph_npz(tmp_path)
    real = ConnectomeCore(_small_config(use_cns_core=True, cns_nodes=N))
    null = ConnectomeCore(_small_config(use_cns_core=True, cns_nodes=N))
    real.load_graph(path, "connectome")
    null.load_graph(path, "rewired")
    assert torch.equal(real.mask.sum(0), null.mask.sum(0))
    assert torch.equal(real.mask.sum(1), null.mask.sum(1))
    assert not torch.equal(real.mask, null.mask)
    for core in (real, null):
        w = core.weight().detach().abs().double().numpy()
        assert np.abs(np.linalg.eigvals(w)).max() == pytest.approx(0.9, rel=1e-4)


def test_kv_cache_decode_matches_full_forward_with_core_open(tmp_path):
    torch.manual_seed(1)
    model = MiMoMixModel(_small_config(use_cns_core=True, cns_nodes=N, cns_after_layer=0)).eval()
    path, _, _ = _graph_npz(tmp_path)
    model.cns_core.load_graph(path)
    with torch.no_grad():
        model.cns_core.gate.normal_(0, 0.5)
        x = torch.randint(0, 97, (2, 12))
        full = model(x).logits[:, -1]
        cache = model(x[:, :11], use_cache=True).past_key_values
        step = model(x[:, 11:], past_key_values=cache, use_cache=True, past_length=11).logits[:, -1]
    assert torch.allclose(full, step, atol=1e-5)


def test_checkpoint_round_trip_carries_the_wiring(tmp_path):
    path, _, _ = _graph_npz(tmp_path)
    config = _small_config(use_cns_core=True, cns_nodes=N)
    source = MiMoMixModel(config)
    source.cns_core.load_graph(path, "rewired")
    restored = MiMoMixModel(MiMoMixConfig.from_dict(config.to_dict()))
    restored.load_state_dict(source.state_dict())  # strict: every buffer present
    assert torch.equal(source.cns_core.mask, restored.cns_core.mask)
    assert restored.cns_core.telemetry()["edges_installed"] == int(source.cns_core.mask.sum())


def test_parameter_groups_unchanged_without_graft():
    model = MiMoMixModel(_small_config())
    groups = parameter_groups(model, 0.01, "all")
    assert split_new_parameter_groups(model, groups, lr_mult=10.0) is groups


def test_parameter_groups_split_graft(tmp_path):
    model = MiMoMixModel(_small_config(use_cns_core=True, cns_nodes=N))
    groups = split_new_parameter_groups(model, parameter_groups(model, 0.01, "all"), lr_mult=10.0)
    assert len(groups) == 3
    old, new_decayed, new_undecayed = groups
    assert "_lr_mult" not in old and new_decayed["_lr_mult"] == new_undecayed["_lr_mult"] == 10.0
    assert set(new_decayed["_names"]) == {"cns_core.read_in.weight", "cns_core.read_out.weight"}
    assert "cns_core.edge_logit" in new_undecayed["_names"] and new_undecayed["weight_decay"] == 0.0
    total = sum(len(g["params"]) for g in groups)
    assert total == len(list(model.parameters()))


def test_analysis_modes_on_identical_weights(tmp_path):
    import v91_analysis as va

    torch.manual_seed(4)
    base = MiMoMixModel(_small_config()).eval()
    grafted = MiMoMixModel(_small_config(use_cns_core=True, cns_nodes=N, cns_after_layer=1)).eval()
    grafted.load_state_dict(base.state_dict(), strict=False)
    path, _, _ = _graph_npz(tmp_path)
    grafted.cns_core.load_graph(path)
    with torch.no_grad():
        grafted.cns_core.gate.normal_(0, 0.5)
    x = torch.randint(6, 97, (5, 16))
    y = x.clone()
    y[:, :4] = -100
    on, counts = va.per_row_losses(grafted, x, y, batch_size=2)
    with va.CoreMode(grafted, "off"):
        off, _ = va.per_row_losses(grafted, x, y, batch_size=2)
    reference, _ = va.per_row_losses(base, x, y, batch_size=2)
    assert np.allclose(off, reference) and not np.allclose(on, reference)
    assert float(grafted.cns_core.gate.abs().sum()) > 0  # gate restored on exit
    delta = va.mean_core_delta(grafted, x, y, batch_size=2)
    with va.CoreMode(grafted, "mean", delta):
        mean, _ = va.per_row_losses(grafted, x, y, batch_size=2)
    assert mean.shape == on.shape and np.isfinite(mean).all()
    assert (counts == 12).all()  # label positions 4..15 survive the shift
    stats = va.paired(on, counts, off, counts, resamples=200)
    assert stats["rows"] == 5 and len(stats["token_mean_diff_ci95"]) == 2


@pytest.mark.parametrize("bad", [
    dict(cns_after_layer=7), dict(cns_io="nowhere"), dict(cns_wiring="random"), dict(cns_steps=0),
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        _small_config(use_cns_core=True, **bad)
