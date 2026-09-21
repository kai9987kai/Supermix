"""v93: two hemispheres in the male-CNS module graph (design contract D1).

The synthetic tests run on tiny graphs and need neither the 1 GB Janelia
download nor sklearn's spectral step (a precomputed embedding is passed). The
last test opens the real ``datasets/v93_malecns`` files when they exist and
checks the invariants the trainer and the null rely on; it is skipped when
the build has not been run.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import malecns_connectome as mc  # noqa: E402

V93_DIR = os.path.join(os.path.dirname(__file__), "datasets", "v93_malecns")
V91_TYPES = os.path.join(os.path.dirname(__file__), "datasets", "v91_malecns", "malecns_types.npz")


def test_assign_side_rule():
    soma = ["L", "R", None, None, "M", "L", float("nan"), None]
    root = [None, "L", "R", "unknown", None, "R", "M", None]
    out = mc.assign_side(soma, root)
    # somaSide wins; rootSide is the fallback; M is midline; the rest unknown.
    assert out["side"].tolist() == [0, 1, 1, 3, 2, 0, 2, 3]
    assert out["fallback"].tolist() == [False, False, True, False, False, False, False, False]
    assert out["conflict"].tolist() == [False, True, False, False, False, True, False, False]
    assert out["side"].dtype == np.int8


def test_block_codes_follow_pre_then_post_side():
    codes = mc.block_of([0, 1, 0, 1], [0, 1, 1, 0])
    assert codes.tolist() == [0, 1, 2, 3] and codes.dtype == np.int8
    assert mc.BLOCK_NAMES == ("LL", "RR", "LR", "RL")


def _two_sided_graph(seed=7, n=30, density=0.22):
    """A random 2n-module graph in the mirror layout: module i and i + n are homologs."""

    rng = np.random.default_rng(seed)
    roles = np.array(rng.choice(["sensory", "central", "output"], size=n, p=[0.3, 0.5, 0.2]), dtype=object)
    sign = np.where(rng.random(n) < 0.6, 1, -1).astype(np.int8)
    side = np.repeat(np.arange(2, dtype=np.int8), n)
    roles2, sign2 = np.concatenate([roles, roles]), np.concatenate([sign, sign])
    matrix = rng.random((2 * n, 2 * n)) * (rng.random((2 * n, 2 * n)) < density)
    np.fill_diagonal(matrix, rng.random(2 * n) * (rng.random(2 * n) < 0.5))
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.001)
    return post, pre, fraction, side, roles2, sign2


def _input_counts_by_stratum(post, pre, node_stratum, n2):
    dense = np.zeros((n2, n2), dtype=bool)
    dense[post, pre] = True
    return np.stack([dense[:, node_stratum == s].sum(1) for s in range(node_stratum.max() + 1)], axis=1)


def test_edge_strata_null_keeps_blocks_degrees_fractions_and_input_kinds():
    post, pre, fraction, side, roles, sign = _two_sided_graph()
    n2 = len(side)
    node_stratum, keys = mc.hemisphere_strata(side, roles, sign)
    edge_strata = mc.hemisphere_edge_strata(post, pre, side, roles, sign)
    assert edge_strata.shape == post.shape
    assert len(keys) == len({(int(a), str(b), int(c)) for a, b, c in zip(side, roles, sign)})
    r_post, r_pre, r_fraction, info = mc.stratified_rewire(
        post, pre, fraction, node_stratum, n2, seed=2, edge_strata=edge_strata)
    assert info["swaps_accepted"] > 0
    assert info["kind"] == "stratified_by_edge_strata_fraction_carried_with_target"
    real = np.zeros((n2, n2)); real[post, pre] = fraction
    null = np.zeros((n2, n2)); null[r_post, r_pre] = r_fraction
    # everything the v91 null preserves
    assert np.array_equal((real > 0).sum(0), (null > 0).sum(0))
    assert np.array_equal((real > 0).sum(1), (null > 0).sum(1))
    assert np.array_equal(np.diag(real), np.diag(null))
    assert len(set(zip(r_post, r_pre))) == len(r_post)
    assert np.allclose(np.sort(real, axis=1), np.sort(null, axis=1))
    assert np.array_equal(_input_counts_by_stratum(post, pre, node_stratum, n2),
                          _input_counts_by_stratum(r_post, r_pre, node_stratum, n2))
    # and, new in v93: the presynaptic end never moves, every edge keeps its
    # block, so block counts and every module's commissural in-degree hold
    assert np.array_equal(r_pre, pre)
    block, r_block = mc.block_of(side[pre], side[post]), mc.block_of(side[r_pre], side[r_post])
    assert np.array_equal(block, r_block)
    contra, r_contra = block >= 2, r_block >= 2
    assert np.array_equal(np.bincount(post[contra], minlength=n2), np.bincount(r_post[r_contra], minlength=n2))
    assert not np.array_equal(real > 0, null > 0)
    # the side-aware diagnostics agree
    diag = mc.hemisphere_diagnostics(r_post, r_pre, r_fraction, roles, side, reference=(post, pre))
    assert diag["side_block_edge_count_l1_diff"] == 0
    assert diag["modules_with_changed_commissural_input_count"] == 0
    assert {b["edges"] for b in diag["blocks"].values()} == {int((block == c).sum()) for c in range(4)}


def test_side_blind_strata_move_edges_between_blocks():
    """The v91 strata (role x sign) let an L->L edge swap with an L->R edge: the
    reason edge_strata exists. With the same seed the block counts change."""

    post, pre, fraction, side, roles, sign = _two_sided_graph()
    n2 = len(side)
    blind_keys = sorted({(str(r), int(s)) for r, s in zip(roles, sign)})
    blind = np.array([blind_keys.index((str(r), int(s))) for r, s in zip(roles, sign)])
    b_post, b_pre, _, info = mc.stratified_rewire(post, pre, fraction, blind, n2, seed=2)
    assert info["kind"] == "stratified_by_source_role_and_sign_fraction_carried_with_target"
    diag = mc.hemisphere_diagnostics(b_post, b_pre, fraction, roles, side, reference=(post, pre))
    assert diag["side_block_edge_count_l1_diff"] > 0
    assert diag["modules_with_changed_commissural_input_count"] > 0


def test_stratified_rewire_positional_call_is_unchanged():
    """v92 and the v91 tests call stratified_rewire positionally with 5-6 args."""

    post, pre, fraction, side, roles, sign = _two_sided_graph(seed=3)
    n2 = len(side)
    strata = np.array([0 if r == "central" else 1 for r in roles])
    a = mc.stratified_rewire(post, pre, fraction, strata, n2, 10, 4)
    b = mc.stratified_rewire(post, pre, fraction, strata, n2, swaps_per_edge=10, seed=4, edge_strata=None)
    for x, y in zip(a[:3], b[:3]):
        assert np.array_equal(x, y)
    assert a[3] == b[3]
    with pytest.raises(ValueError):
        mc.stratified_rewire(post, pre, fraction, None, n2)


def _tiny_files(tmp_path, seed=0):
    """A synthetic type graph and sided graph in the layouts `build` and
    `build_sided_types` write. Type T29 is the only inhibitory motor type, so it
    is alone in its (role, sign) group and gets a module of its own; it exists
    on the left only, so that module is empty on the right."""

    rng = np.random.default_rng(seed)
    n_types = 30
    type_names = np.array([f"T{i:02d}" for i in range(n_types)], dtype=object)
    superclass = np.array(["cb_sensory"] * 6 + ["cb_intrinsic"] * 18 + ["descending_neuron"] * 3
                          + ["cb_motor"] * 3, dtype=object)
    sign = np.where(rng.random(n_types) < 0.6, 1, -1).astype(np.int8)
    sign[27:30] = [1, 1, -1]
    n_neurons = rng.integers(1, 10, n_types).astype(np.int32)
    common = dict(type_names=type_names, superclass=superclass, cell_class=superclass,
                  nt=np.array(["acetylcholine"] * n_types, dtype=object), sign=sign,
                  sign_is_modulatory=np.zeros(n_types, bool), n_neurons=n_neurons)
    pre_t, post_t = np.nonzero(rng.random((n_types, n_types)) < 0.4)
    types_path = tmp_path / "types.npz"
    np.savez_compressed(types_path, **common, has_flywire=np.zeros(n_types, bool), has_manc=np.zeros(n_types, bool),
                        pre=pre_t.astype(np.int32), post=post_t.astype(np.int32),
                        weight=rng.integers(1, 50, len(pre_t)).astype(np.int64))
    node_type = np.concatenate([np.arange(n_types), np.arange(n_types - 1)]).astype(np.int32)
    node_side = np.concatenate([np.zeros(n_types), np.ones(n_types - 1)]).astype(np.int8)
    nn = len(node_type)
    # Sparse enough (about 3 nodes per module) that the 20-module matrix is not
    # complete, otherwise no degree-preserving swap can ever be accepted.
    pre_s, post_s = np.nonzero(rng.random((nn, nn)) < 0.06)
    sided_path = tmp_path / "sided.npz"
    np.savez_compressed(sided_path, **common, node_type=node_type, node_side=node_side,
                        node_n_neurons=rng.integers(1, 5, nn).astype(np.int32),
                        pre=pre_s.astype(np.int32), post=post_s.astype(np.int32),
                        weight=rng.integers(1, 50, len(pre_s)).astype(np.int64))
    return str(types_path), str(sided_path), rng.standard_normal((n_types, 3))


def _check_hemisphere_arrays(d, min_fraction):
    """Invariants every mirror-scheme npz must satisfy."""

    n2 = len(d["module_sign"])
    n = n2 // 2
    side, homolog = d["module_side"], d["homolog"]
    post, pre, fraction, block = d["edge_post"], d["edge_pre"], d["edge_fraction"].astype(np.float64), d["edge_block"]
    assert side.dtype == np.int8 and block.dtype == np.int8 and homolog.dtype == np.int64
    assert side.tolist() == [0] * n + [1] * n
    assert set(str(r) for r in d["module_role"]) <= set(mc.ROLES)
    assert d["null_kind"].item() == mc.HEMISPHERE_NULL_KIND
    # homologs: an involution across sides with identical role and sign
    assert np.array_equal(homolog[homolog], np.arange(n2))
    assert np.array_equal(side[homolog], 1 - side)
    assert np.array_equal(d["module_role"][homolog], d["module_role"])
    assert np.array_equal(d["module_sign"][homolog], d["module_sign"])
    # mirror scheme: module_of_node = module_of_type[type] + n * side
    expected = d["module_of_type"][d["node_type"]] + n * d["node_side"].astype(np.int64)
    assert np.array_equal(d["module_of_node"], expected)
    assert np.array_equal(d["module_n_types"], np.bincount(d["module_of_node"], minlength=n2))
    # edges: blocks consistent with sides, fractions are shares of TOTAL input
    assert np.array_equal(block, mc.block_of(side[pre], side[post]))
    matrix = d["synapse_matrix"].astype(np.float64)
    row = matrix.sum(1)
    assert np.allclose(fraction, matrix[post, pre] / row[post], rtol=1e-5)
    assert (fraction >= min_fraction * (1 - 1e-6)).all()
    dense = np.zeros((n2, n2)); dense[post, pre] = fraction
    assert dense.sum(1).max() <= 1 + 1e-6
    assert len(set(zip(post.tolist(), pre.tolist()))) == len(post)
    # empty modules keep their slot with zero rows and columns
    empty = d["module_n_types"] == 0
    assert (matrix[empty].sum(1) == 0).all() and (matrix[:, empty].sum(0) == 0).all()
    assert not (np.isin(post, np.flatnonzero(empty)) | np.isin(pre, np.flatnonzero(empty))).any()
    for i in np.flatnonzero(empty):
        assert "<empty>" in str(d["module_label"][i])
    # the null: same pre, same degrees, same self-loops, same blocks, same fraction multiset per row
    r_post, r_pre, r_fraction = d["rewired_post"], d["rewired_pre"], d["rewired_fraction"].astype(np.float64)
    assert np.array_equal(r_pre, pre)
    assert np.array_equal(np.bincount(post, minlength=n2), np.bincount(r_post, minlength=n2))
    assert np.array_equal(post == pre, r_post == r_pre)
    assert np.array_equal(block, mc.block_of(side[r_pre], side[r_post]))
    null = np.zeros((n2, n2)); null[r_post, r_pre] = r_fraction
    assert np.allclose(np.sort(dense, axis=1), np.sort(null, axis=1), atol=1e-6)
    assert len(set(zip(r_post.tolist(), r_pre.tolist()))) == len(r_post)
    assert not np.array_equal(dense > 0, null > 0)


def test_build_hemispheres_mirror_scheme_and_empty_module(tmp_path):
    types_path, sided_path, embedding = _tiny_files(tmp_path)
    output = str(tmp_path / "hemi_10.npz")
    receipt = mc.build_hemispheres(types_path, sided_path, output, 10, 0.01, seed=0, embedding=embedding)
    d = dict(np.load(output, allow_pickle=True))
    _check_hemisphere_arrays(d, 0.01)
    n = receipt["n_per_side"]
    assert n == 10 and receipt["n_modules"] == 2 * n == len(d["module_sign"])
    # T29 is left-only and alone in its module: the right slot is kept, empty, and counted
    t29 = 29
    left_mod = int(d["module_of_type"][t29])
    assert d["module_n_types"][left_mod] == 1 and d["module_n_types"][left_mod + n] == 0
    assert receipt["per_side"]["R"]["empty_modules"] == 1 and receipt["per_side"]["L"]["empty_modules"] == 0
    assert receipt["empty_module_ids"] == [left_mod + n]
    assert d["module_label"][left_mod] == "output:T29@L" and d["module_label"][left_mod + n] == "output:<empty>@R"
    # receipt bookkeeping
    blocks = receipt["blocks"]
    assert sum(b["edges"] for b in blocks.values()) == receipt["edges"] == len(d["edge_post"])
    assert blocks["LR"]["self_loops"] == blocks["RL"]["self_loops"] == 0
    assert receipt["strata"]["node"] == len(set(zip(d["module_side"].tolist(), map(str, d["module_role"]),
                                                     d["module_sign"].tolist())))
    real, null = receipt["diagnostics"]["real"], receipt["diagnostics"]["stratified_null"]
    assert real["rows_over_one"] == null["rows_over_one"] == 0
    assert null["side_block_edge_count_l1_diff"] == 0 and null["role_block_edge_count_l1_diff"] == 0
    assert null["modules_with_changed_commissural_input_count"] == 0
    assert real["abs_spectral_radius"]["joint"] >= max(real["abs_spectral_radius"]["left"],
                                                       real["abs_spectral_radius"]["right"]) - 1e-9
    assert 0 <= real["homolog_edge_symmetry"]["all"] <= 1
    assert os.path.exists(os.path.splitext(output)[0] + ".receipt.json")


def test_connectome_core_loads_a_hemisphere_file(tmp_path):
    """The primary consumer reads the 2N-id file unchanged, both wirings."""

    import torch
    from mimomix_core import ConnectomeCore, MiMoMixConfig

    types_path, sided_path, embedding = _tiny_files(tmp_path)
    output = str(tmp_path / "hemi_10.npz")
    mc.build_hemispheres(types_path, sided_path, output, 10, 0.01, seed=0, embedding=embedding)
    d = np.load(output, allow_pickle=True)
    n2 = len(d["module_sign"])
    config = MiMoMixConfig(
        vocab_size=97, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2, intermediate_size=48,
        moe_intermediate_size=16, n_routed_experts=4, moe_top_k=2, n_mtp_layers=1, sliding_window=8,
        native_context=32, max_position_embeddings=32, thinking_latent_dim=8, use_cns_core=True, cns_nodes=n2,
    )
    real, null = ConnectomeCore(config), ConnectomeCore(config)
    info = real.load_graph(output, "connectome")
    null.load_graph(output, "rewired")
    assert info["nodes"] == n2 and info["edges"] == len(d["edge_post"])
    assert torch.equal(real.mask.sum(0), null.mask.sum(0)) and torch.equal(real.mask.sum(1), null.mask.sum(1))
    assert not torch.equal(real.mask, null.mask)
    assert torch.equal(real.sign[: n2 // 2], real.sign[n2 // 2:])  # homologous signs
    installed = real.init_fraction[real.mask > 0]
    assert float(installed.min()) >= 0.01 * (1 - 1e-6)


@pytest.mark.parametrize("n_per_side", [256, 384])
def test_built_hemisphere_files_if_present(n_per_side):
    path = os.path.join(V93_DIR, f"malecns_hemispheres_{n_per_side}.npz")
    sided_path = os.path.join(V93_DIR, mc.SIDED_TYPES_NAME)
    if not (os.path.exists(path) and os.path.exists(sided_path)):
        pytest.skip("datasets/v93_malecns not built")
    import json

    d = dict(np.load(path, allow_pickle=True))
    sided = dict(np.load(sided_path, allow_pickle=True))
    with open(os.path.splitext(path)[0] + ".receipt.json", encoding="utf-8") as handle:
        receipt = json.load(handle)
    _check_hemisphere_arrays(d, receipt["min_input_fraction"])
    n2 = len(d["module_sign"])
    assert n2 == 2 * n_per_side == receipt["n_modules"]
    assert receipt["edges"] == len(d["edge_post"])
    assert sum(b["edges"] for b in receipt["blocks"].values()) == receipt["edges"]
    assert receipt["diagnostics"]["stratified_null"]["side_block_edge_count_l1_diff"] == 0
    assert receipt["diagnostics"]["stratified_null"]["rows_over_one"] == 0
    assert receipt["per_side"]["L"]["empty_modules"] + receipt["per_side"]["R"]["empty_modules"] == len(
        receipt["empty_module_ids"])
    # the sided type table: same types as v91, type-level metadata copied by name
    assert np.array_equal(d["node_type"], sided["node_type"]) and np.array_equal(d["node_side"], sided["node_side"])
    assert set(sided["node_side"].tolist()) == {0, 1}
    assert int(sided["node_type"].max()) < len(sided["type_names"])
    assert np.array_equal(np.unique(sided["node_type"] * 2 + sided["node_side"]),
                          sided["node_type"] * 2 + sided["node_side"])  # nodes are unique and sorted
    if os.path.exists(V91_TYPES):
        v91 = dict(np.load(V91_TYPES, allow_pickle=True))
        for key in ("type_names", "superclass", "cell_class", "nt", "sign", "sign_is_modulatory", "n_neurons"):
            assert np.array_equal(v91[key], sided[key]), key
    # synapse bookkeeping: the module matrix holds every sided synapse
    assert float(d["synapse_matrix"].astype(np.float64).sum()) == pytest.approx(float(sided["weight"].sum()), rel=1e-6)
    with open(os.path.splitext(sided_path)[0] + ".receipt.json", encoding="utf-8") as handle:
        sided_receipt = json.load(handle)
    assert sided_receipt["synapses_sided"] == int(sided["weight"].sum())
    assert sum(sided_receipt["synapses_by_block"].values()) == sided_receipt["synapses_sided"]
    assert sided_receipt["bodies_by_side"]["L"] + sided_receipt["bodies_by_side"]["R"] == int(
        sided["node_n_neurons"].sum())
    # nodes on each side and neurons per module add up
    assert d["module_n_neurons"].sum() == pytest.approx(float(sided["node_n_neurons"].sum()))
