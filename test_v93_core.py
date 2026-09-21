"""v93 model-side contracts: capacity slots, multi-site taps, temporal core,
MoE expert slots, depth growth (docs/V93_NEUROGENESIS_TWO_HEMISPHERES.md,
D2-D4).

Everything runs on tiny synthetic graphs (8 modules + 2 spare slots, hidden
32, 2-3 layers, 4 experts + 2 spare). The one test that touches a real
checkpoint (v91 strict load) skips when the file is absent.

The backward-compatibility gate has two halves. The half here compares a model
built from a config *without* the v93 fields (``MiMoMixConfig(**old_dict)``)
against one with them spelled out at their defaults. The other half -- bit
identity of every pre-v93 tensor and of the forward logits against the
pre-v93 source snapshot -- was run once at implementation time
(``scratchpad/compare_snapshot.py``) and is reported in the v93 receipt, since
the snapshot is not part of the repository.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import malecns_connectome as mc  # noqa: E402
from mimomix_core import (  # noqa: E402
    ConnectomeCore,
    MiMoMixConfig,
    MiMoMixModel,
    SparseMoEFeedForward,
    pin_layout_for_growth,
)
from mimomix_decoding import assert_greedy_equivalence, trim_past  # noqa: E402

N = 8          # live modules in the synthetic graph
SPARE = 2      # dead slots appended after them
CAP = N + SPARE
V91_CHECKPOINT = os.path.join(os.path.dirname(__file__), "output", "v91_cns_connectome", "v91_cns_connectome.pt")

V93_FIELDS = (
    "cns_spare_nodes", "cns_read_layers", "cns_write_layers", "cns_to_thinking",
    "cns_temporal", "moe_spare_experts", "grow_layers",
)


def _small_config(**overrides):
    base = dict(
        vocab_size=97, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2,
        intermediate_size=48, moe_intermediate_size=16, n_routed_experts=4,
        moe_top_k=2, n_mtp_layers=1, sliding_window=8, native_context=32,
        max_position_embeddings=32, thinking_latent_dim=8,
    )
    base.update(overrides)
    return MiMoMixConfig(**base)


def _core_config(**overrides):
    base = dict(use_cns_core=True, cns_nodes=CAP, cns_spare_nodes=SPARE, cns_after_layer=1, cns_steps=3)
    base.update(overrides)
    return _small_config(**base)


def _graph_npz(tmp_path, seed=0, sided=True, name="modules.npz"):
    """An 8-module graph in the layout `build_modules` / `hemispheres` write.

    Sides are the mirror scheme (modules 0-3 left, 4-7 right); the random
    matrix is dense enough that every block LL/RR/LR/RL carries edges.
    """

    rng = np.random.default_rng(seed)
    roles = np.array(["sensory", "central", "descending", "output"] * 2, dtype=object)
    sign = np.array([1, -1, 1, 1, 1, -1, 1, 1], dtype=np.int8)
    matrix = rng.random((N, N)) * (rng.random((N, N)) < 0.7)
    np.fill_diagonal(matrix, rng.random(N) * (rng.random(N) < 0.5))
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.01)
    r_post, r_pre, _ = mc.degree_preserving_rewire(post, pre, N, seed=1)
    arrays = dict(
        module_role=roles, module_sign=sign, edge_post=post, edge_pre=pre,
        edge_fraction=fraction.astype(np.float32), rewired_post=r_post, rewired_pre=r_pre,
    )
    if sided:
        arrays["module_side"] = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    path = tmp_path / name
    np.savez_compressed(path, **arrays)
    return str(path)


def _open_gates(core: ConnectomeCore, value: float = 0.3, seed: int = 0) -> None:
    """Open every write gate and give the read taps real weights."""

    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        core.gate.fill_(value)
        for gate in core.extra_gates:
            gate.fill_(value)
        if core.thinking_gate is not None:
            core.thinking_gate.fill_(value)
        core.read_in.weight.normal_(0, 0.5, generator=generator)
        for tap in core.extra_read_in:
            tap.weight.normal_(0, 0.5, generator=generator)
        core.leak_logit.fill_(0.5)


def _grafted(tmp_path, seed=0, **overrides):
    torch.manual_seed(seed)
    model = MiMoMixModel(_core_config(**overrides)).eval()
    model.cns_core.load_graph(_graph_npz(tmp_path, seed=seed))
    return model


# ---------------------------------------------------------------------------
# A. backward compatibility
# ---------------------------------------------------------------------------


def test_default_config_matches_a_config_without_the_v93_fields():
    """A config dict from before v93 (no new keys) builds the same model."""

    explicit = _small_config(
        use_cns_core=True, cns_nodes=N, cns_after_layer=1,
        cns_spare_nodes=0, cns_read_layers=(), cns_write_layers=(), cns_to_thinking=False,
        cns_temporal=False, moe_spare_experts=0, grow_layers=0,
    )
    old_dict = {k: v for k, v in explicit.to_dict().items() if k not in V93_FIELDS}
    assert len(old_dict) == len(explicit.to_dict()) - len(V93_FIELDS)
    torch.manual_seed(0)
    old_model = MiMoMixModel(MiMoMixConfig(**old_dict))
    torch.manual_seed(0)
    new_model = MiMoMixModel(explicit)
    old_state, new_state = old_model.state_dict(), new_model.state_dict()
    assert list(old_state) == list(new_state)
    for key in old_state:
        assert old_state[key].shape == new_state[key].shape and torch.equal(old_state[key], new_state[key]), key
    assert [n for n, _ in old_model.named_parameters()] == [n for n, _ in new_model.named_parameters()]
    assert new_model.cns_core.single_site and not new_model.cns_core.extra_gates
    assert new_model.config.cns_read_sites == (1,) and new_model.config.cns_write_sites == (1,)
    x = torch.randint(0, 97, (2, 12))
    old_model.eval(); new_model.eval()
    with torch.no_grad():
        assert torch.equal(old_model(x, past_length=0).logits, new_model(x, past_length=0).logits)
    # v93 buffers exist, persistent, at their inert defaults
    assert "cns_core.alive" in new_state and int(new_state["cns_core.alive"].sum()) == N
    assert int(new_state["cns_core.edge_grown"].sum()) == 0 and int(new_state["cns_core.hemisphere"].abs().sum()) == 0
    assert all(int(new_state[k].sum()) == 4 for k in new_state if k.endswith("expert_alive"))


def test_missing_v93_buffers_are_filled_on_strict_load(tmp_path):
    """A state_dict without the v93 buffers (a v91/v89 checkpoint) loads strictly."""

    source = _small_config(use_cns_core=True, cns_nodes=N, cns_after_layer=1)
    model = MiMoMixModel(source)
    model.cns_core.load_graph(_graph_npz(tmp_path, sided=False))
    state = {k: v.clone() for k, v in model.state_dict().items()}
    dropped = [k for k in state if k.split(".")[-1] in ConnectomeCore._V93_BUFFERS + SparseMoEFeedForward._V93_BUFFERS]
    assert len(dropped) == len(ConnectomeCore._V93_BUFFERS) + 2  # two MoE layers
    for key in dropped:
        del state[key]
    restored = MiMoMixModel(MiMoMixConfig.from_dict(source.to_dict()))
    restored.load_state_dict(state)  # strict
    assert int(restored.cns_core.alive.sum()) == N and int(restored.cns_core.edge_grown.sum()) == 0
    assert all(int(m.mlp.expert_alive.sum()) == 4 for m in restored.layers if m.is_moe)
    assert torch.equal(restored.cns_core.mask, model.cns_core.mask)
    # the caller's dict is untouched
    assert not any(k in state for k in dropped)


@pytest.mark.skipif(not os.path.exists(V91_CHECKPOINT), reason="v91 checkpoint not present")
def test_v91_checkpoint_loads_strictly_and_runs():
    from train_mimomix_talk import load_talk_checkpoint

    model, _, payload = load_talk_checkpoint(V91_CHECKPOINT)
    assert all(k not in payload["config"] for k in V93_FIELDS)
    core = model.cns_core
    assert core.single_site and int(core.alive.sum()) == core.n_nodes == 512
    assert int(core.edge_grown.sum()) == 0 and int(core.hemisphere.abs().sum()) == 0
    with torch.no_grad():
        out = model(torch.tensor([[6, 7]]), past_length=0)
    assert out.logits.shape == (1, 2, model.config.vocab_size) and torch.isfinite(out.logits).all()
    assert out.telemetry["cns_core"]["edges_by_block"]["LL"] == out.telemetry["cns_core"]["edges_installed"]


# ---------------------------------------------------------------------------
# B. config
# ---------------------------------------------------------------------------


def test_config_round_trip_with_tuple_fields():
    config = _core_config(cns_read_layers=[0, 1], cns_write_layers=(2, 1), cns_to_thinking=True)
    assert config.cns_read_layers == (0, 1) and config.cns_write_layers == (1, 2)
    payload = config.to_dict()
    assert payload["cns_read_layers"] == [0, 1] and payload["cns_write_layers"] == [1, 2]
    through_json = json.loads(json.dumps(payload))
    assert MiMoMixConfig(**through_json).to_dict() == payload
    assert MiMoMixConfig.from_dict(through_json).to_dict() == payload
    assert config.cns_run_after_layer == 1 and config.cns_read_sites == (0, 1)


@pytest.mark.parametrize("bad", [
    dict(cns_read_layers=(0, 2), cns_write_layers=(1,)),   # write below the deepest read
    dict(cns_read_layers=(0, 5)),                           # out of range
    dict(cns_spare_nodes=CAP - 1),                          # fewer than 2 live modules
    dict(cns_to_thinking=True, use_thinking_core=False),
    dict(moe_spare_experts=-1),
    dict(grow_layers=1),                                    # not pinned
    dict(grow_layers=3, global_layers=(2,)),                # >= n_layers
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        _core_config(**bad)


# ---------------------------------------------------------------------------
# C. connectome core: graft, taps, temporal, cache
# ---------------------------------------------------------------------------


def test_load_graph_fills_capacity_and_hemisphere(tmp_path):
    core = ConnectomeCore(_core_config())
    info = core.load_graph(_graph_npz(tmp_path), spectral_radius=0.8)
    assert core.alive.tolist() == [1] * N + [0] * SPARE
    assert core.hemisphere.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert core.mask[N:].sum() == 0 and core.mask[:, N:].sum() == 0
    assert core.in_mask[N:].sum() == 0 and core.out_mask[N:].sum() == 0
    assert core.read_in.weight[N:].abs().sum() == 0 and core.read_out.weight[:, N:].abs().sum() == 0
    w = core.weight().detach().abs().double().numpy()[:N, :N]
    assert np.abs(np.linalg.eigvals(w)).max() == pytest.approx(0.8, rel=1e-4)
    blocks = info["edges_by_block_at_load"]
    assert sum(blocks.values()) == info["edges"] and blocks["LR"] + blocks["RL"] > 0
    assert info["hemisphere_nodes"] == [4, 4] and info["spare_nodes"] == SPARE
    with pytest.raises(ValueError, match="live slots"):
        ConnectomeCore(_core_config(cns_spare_nodes=1)).load_graph(_graph_npz(tmp_path))


@pytest.mark.parametrize("temporal", [False, True])
def test_graft_is_function_preserving_with_taps_bond_and_temporal(tmp_path, temporal):
    torch.manual_seed(0)
    base = MiMoMixModel(_small_config()).eval()
    grafted = MiMoMixModel(_core_config(
        cns_read_layers=(0, 1), cns_write_layers=(1, 2), cns_to_thinking=True, cns_temporal=temporal,
    )).eval()
    missing, unexpected = grafted.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("cns_core.") for k in missing)
    assert {"cns_core.extra_read_in.0.weight", "cns_core.extra_read_out.0.weight", "cns_core.extra_gates.0",
            "cns_core.to_thinking.weight", "cns_core.thinking_gate"} <= set(missing)
    grafted.cns_core.load_graph(_graph_npz(tmp_path))
    assert not grafted.cns_core.single_site
    x = torch.randint(0, 97, (2, 16))
    with torch.no_grad():
        assert torch.equal(base(x, past_length=0).logits, grafted(x, past_length=0).logits)
        _open_gates(grafted.cns_core)
        assert not torch.equal(base(x, past_length=0).logits, grafted(x, past_length=0).logits)
    tel = grafted(x, past_length=0).telemetry["cns_core"]
    assert tel["read_layers"] == [0, 1] and tel["write_layers"] == [1, 2] and tel["temporal"] is temporal
    assert tel["extra_gates"][0]["layer"] == 2 and tel["thinking_gate_mean_abs"] == pytest.approx(0.3)
    json.dumps(tel)


def test_thinking_bond_only_enters_the_latent_path(tmp_path):
    """With every write gate closed and only the thinking gate open, the bond
    changes the logits through the recursive core and nothing else."""

    model = _grafted(tmp_path, cns_read_layers=(0, 1), cns_write_layers=(1, 2), cns_to_thinking=True)
    x = torch.randint(0, 97, (2, 10))
    with torch.no_grad():
        model.cns_core.read_in.weight.normal_(0, 0.5)
        before = model(x, past_length=0).logits
        model.cns_core.thinking_gate.fill_(0.5)
        model.thinking_core.residual_scale.fill_(0.0)
        unchanged = model(x, past_length=0).logits
        model.thinking_core.residual_scale.fill_(0.3)
        base_scaled = model(x, past_length=0).logits
        model.cns_core.thinking_gate.zero_()
        without_bond = model(x, past_length=0).logits
    assert torch.equal(before, unchanged)               # residual_scale 0: the core's residual is off
    assert not torch.allclose(base_scaled, without_bond)  # the bond matters once the core's residual is on


@pytest.mark.parametrize("temporal", [False, True])
def test_cached_decode_matches_full_forward_with_gates_open(tmp_path, temporal):
    model = _grafted(tmp_path, seed=1, cns_read_layers=(0, 1), cns_write_layers=(1, 2),
                     cns_to_thinking=True, cns_temporal=temporal)
    _open_gates(model.cns_core, 0.3)
    x = torch.randint(0, 97, (2, 14))
    with torch.no_grad():
        full = model(x, past_length=0)
        prefix = model(x[:, :9], use_cache=True, past_length=0)
        past = prefix.past_key_values
        assert len(past) == len(model.layers) + (1 if temporal else 0)
        rest = model(x[:, 9:], past_key_values=past, use_cache=True, past_length=9)
        # one token at a time as well
        past_single = prefix.past_key_values
        singles = []
        for t in range(9, 14):
            step = model(x[:, t:t + 1], past_key_values=past_single, use_cache=True, past_length=t)
            past_single = step.past_key_values
            singles.append(step.logits)
    assert torch.allclose(rest.logits, full.logits[:, 9:], atol=1e-5)
    assert torch.allclose(torch.cat(singles, 1), full.logits[:, 9:], atol=1e-5)
    if temporal:
        entry = rest.past_key_values[len(model.layers)]
        assert entry[0].shape == (2, 1, CAP) and entry[1].shape == (2, 1, 0)
        assert torch.allclose(entry[0][:, -1], model.cns_core.rates([h for h in _reads(model, x)])[:, -1], atol=1e-5)


def _reads(model, x):
    """Trunk hidden states after each read layer for a full forward (no cache)."""

    positions = torch.arange(x.shape[1])
    cos, sin = model.rotary(positions)
    lcos, lsin = model.rotary_local(positions)
    keys = positions.new_empty((0,))
    hidden = model.embed_tokens(x)
    reads = []
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            c, s = (lcos, lsin) if layer.kind == "swa" else (cos, sin)
            hidden, _ = layer(hidden, c, s, positions, keys, past_kv=None, attention_mask=None, use_cache=False)
            if index in model.cns_core.read_sites:
                reads.append(hidden)
            if index == model.cns_core.read_sites[-1]:
                break
    return reads


def test_temporal_state_entry_rides_the_cache_and_trims(tmp_path):
    model = _grafted(tmp_path, cns_read_layers=(0, 1), cns_write_layers=(1, 2), cns_temporal=True)
    _open_gates(model.cns_core, 0.3)
    x = torch.randint(0, 97, (1, 6))
    with torch.no_grad():
        out = model(x, use_cache=True, cache_slack=2, past_length=0)
        entry = out.past_key_values[-1]
        assert entry[0].shape == (1, 3, CAP)  # 1 + cache_slack states kept
        assert torch.allclose(entry[0], model.cns_core.rates(_reads(model, x))[:, -3:], atol=1e-6)
        trimmed = trim_past(out.past_key_values, 2)
        assert trimmed[-1][0].shape == (1, 1, CAP) and torch.equal(trimmed[-1][0][:, 0], entry[0][:, 0])
        # rolling back two positions then re-feeding them reproduces the full forward
        replay = model(x[:, 4:], past_key_values=trimmed, use_cache=True, past_length=4, cache_slack=2)
        assert torch.allclose(replay.logits, model(x, past_length=0).logits[:, 4:], atol=1e-5)
        # a cache with positions but no state entry is refused, not silently restarted
        with pytest.raises(ValueError, match="state entry"):
            model(x[:, 4:], past_key_values=trimmed[:-1], use_cache=True, past_length=4)
    # the cns entry never inflates the inferred past length
    with torch.no_grad():
        inferred = model(x[:, 4:], past_key_values=trimmed, use_cache=True)
    assert inferred.telemetry["past_length"] == 4


def test_speculative_equals_greedy_for_the_temporal_core(tmp_path):
    model = _grafted(tmp_path, seed=3, cns_read_layers=(0, 1), cns_write_layers=(1, 2),
                     cns_to_thinking=True, cns_temporal=True)
    _open_gates(model.cns_core, 0.3)
    prompt = torch.randint(0, 97, (2, 5))
    report = assert_greedy_equivalence(model, prompt, max_new_tokens=12)
    assert report["tokens"] == 12


def test_temporal_core_is_causal_and_per_token_core_is_not_stateful(tmp_path):
    temporal = _grafted(tmp_path, cns_temporal=True)
    per_token = _grafted(tmp_path)
    for model in (temporal, per_token):
        _open_gates(model.cns_core, 0.3)
    x = torch.randint(0, 97, (1, 12))
    y = x.clone()
    y[0, 6] = (y[0, 6] + 1) % 97
    with torch.no_grad():
        a, b = temporal(x, past_length=0).logits, temporal(y, past_length=0).logits
        assert torch.allclose(a[:, :6], b[:, :6]) and not torch.allclose(a[:, 7:], b[:, 7:])
        reads = _reads(per_token, x)
        rates = per_token.cns_core.rates(reads)
        assert rates.shape == (12, CAP) and rates[:, N:].abs().sum() == 0  # spare slots inert


# ---------------------------------------------------------------------------
# D. growth primitives
# ---------------------------------------------------------------------------


def _rates_and_readout(core, hidden):
    with torch.no_grad():
        rate = core.rates([hidden])
        read = core.read_out(rate * core.out_mask)  # unnormalised efferent read-out
    return rate, read


def test_split_module_conserves_drive_and_readout(tmp_path):
    torch.manual_seed(0)
    core = ConnectomeCore(_core_config(cns_io="all"))
    core.load_graph(_graph_npz(tmp_path))
    _open_gates(core, 0.3)
    with torch.no_grad():
        core.node_bias.normal_(0, 0.3)
        core.read_out.weight.normal_(0, 0.5)
    hidden = torch.randn(3, 5, 32)
    parent = int(core.mask.sum(0).argmax())  # the module with the most outgoing edges
    before_rate, before_read = _rates_and_readout(core, hidden)
    out_strength = (core.weight().detach().abs()[:, parent]).sum()
    child = core.split_module(parent, step=500)
    assert child == N and int(core.alive[child]) == 1 and int(core.born_step[child]) == 500
    assert core.last_event["event"] == "split_module" and core.last_event["child"] == child
    after_rate, after_read = _rates_and_readout(core, hidden)
    keep = [i for i in range(CAP) if i != child]
    assert torch.allclose(after_rate[:, keep], before_rate[:, keep], atol=1e-5)
    assert torch.allclose(after_rate[:, child], after_rate[:, parent], atol=1e-6)
    assert torch.allclose(after_read, before_read, atol=1e-5)
    w = core.weight().detach().abs()
    assert torch.allclose(w[:, parent].sum() + w[:, child].sum(), out_strength, atol=1e-5)
    assert torch.equal(core.mask[child], core.mask[parent]) and torch.equal(core.mask[:, child], core.mask[:, parent])
    assert int(core.sign[child]) == int(core.sign[parent]) and int(core.hemisphere[child]) == int(core.hemisphere[parent])
    assert torch.equal(core.read_in.weight[child], core.read_in.weight[parent])
    assert core.telemetry()["grown_nodes"] == 1 and core.telemetry()["alive_nodes"] == N + 1
    # a second split fills the last slot, a third is refused
    assert core.split_module(parent, step=600) == N + 1
    assert core.split_module(parent, step=700) is None and core.last_event["refused"] == "no free slot"
    assert core.free_slots() == []


def test_grow_edge_open_tap_prune_and_kill(tmp_path):
    core = ConnectomeCore(_core_config())
    core.load_graph(_graph_npz(tmp_path))
    _open_gates(core, 0.3)
    hidden = torch.randn(2, 4, 32)
    unconnected = [(p, q) for p in range(N) for q in range(N) if p != q and float(core.mask[p, q]) == 0]
    post, pre = unconnected[0]
    with torch.no_grad():
        before = core(hidden)
    assert core.grow_edge(post, pre, step=500)
    assert float(core.mask[post, pre]) == 1 and int(core.edge_grown[post, pre]) == 1
    w = core.weight().detach()
    assert w[post, pre] == pytest.approx(float(core.sign[pre]) * 9.1e-4, rel=1e-2)  # softplus(-7), Dale sign
    assert not core.grow_edge(post, pre, step=501)      # exists
    assert not core.grow_edge(post, post, step=501)     # self-loop refused
    assert not core.grow_edge(N, pre, step=501)         # dead endpoint
    assert core.telemetry()["grown_edges"] == 1
    # an afferent tap is exact at birth: zero row, in_mask 1
    non_afferent = int(torch.nonzero(core.in_mask[:N] == 0)[0])
    assert core.open_tap(non_afferent, "in") and not core.open_tap(non_afferent, "in")
    assert float(core.in_mask[non_afferent]) == 1 and core.read_in.weight[non_afferent].abs().sum() == 0
    assert int(core.tap_grown[non_afferent, 0]) == 1
    with torch.no_grad():
        core.ablate_grown = True
        assert torch.equal(core(hidden), before)        # grown edge + tap masked off -> pre-growth output
        core.ablate_grown = False
    non_efferent = int(torch.nonzero(core.out_mask[:N] == 0)[0])
    assert core.open_tap(non_efferent, "out")
    assert float(core.out_mask[non_efferent]) == 1 and core.read_out.weight[:, non_efferent].abs().sum() == 0
    with pytest.raises(ValueError):
        core.open_tap(0, "sideways")
    # pruning: only the grown edge sits below 1e-3
    assert core.prune_edges(1e-3, grown_only=True) == 1 and core.last_event["grown_pruned"] == 1
    assert float(core.mask[post, pre]) == 0 and int(core.edge_grown.sum()) == 0
    real_edges = int(core.mask.sum())
    assert core.prune_edges(1e-9) == 0 and int(core.mask.sum()) == real_edges
    # kill, then the slot is free again and a split can reuse it
    victim = 3
    assert core.kill_module(victim) and not core.kill_module(victim)
    assert int(core.alive[victim]) == 0 and core.mask[victim].sum() == 0 and core.mask[:, victim].sum() == 0
    assert core.read_in.weight[victim].abs().sum() == 0 and core.read_out.weight[:, victim].abs().sum() == 0
    assert core.free_slots() == [victim, N, N + 1]
    assert core.split_module(0, step=900) == victim


def test_ablation_switches(tmp_path):
    model = _grafted(tmp_path, cns_io="all")
    core = model.cns_core
    _open_gates(core, 0.3)
    x = torch.randint(0, 97, (2, 8))
    with torch.no_grad():
        reference = model(x, past_length=0).logits
        core.ablate_cross = True
        w = core.weight()
        side = core.hemisphere
        assert w[(side.unsqueeze(1) != side.unsqueeze(0))].abs().sum() == 0
        assert w[:4, :4].abs().sum() > 0 and w[4:8, 4:8].abs().sum() > 0
        crossed = model(x, past_length=0).logits
        core.ablate_cross = False
        core.ablate_side = 1
        rates = core.rates(_reads(model, x))
        assert rates[:, 4:8].abs().sum() == 0 and rates[:, :4].abs().sum() > 0
        sided = model(x, past_length=0).logits
        core.ablate_side = None
        assert torch.equal(model(x, past_length=0).logits, reference)
    assert not torch.allclose(crossed, reference) and not torch.allclose(sided, reference)
    assert model(x, past_length=0).telemetry["cns_core"].get("ablation") is None
    core.ablate_side = 0
    assert model(x, past_length=0).telemetry["cns_core"]["ablation"] == {"cross": False, "grown": False, "side": 0}


def test_growth_statistics_accumulate_only_in_eval(tmp_path):
    model = _grafted(tmp_path)
    core = model.cns_core
    _open_gates(core, 0.3)
    core.collect_stats = True
    x = torch.randint(0, 97, (2, 6))
    model.train()
    model(x, labels=x)
    assert int(core.stat_tokens) == 0
    model.eval()
    with torch.no_grad():
        model(x, past_length=0)
        model(x, past_length=0)
    stats = core.growth_statistics()
    assert stats["tokens"] == 24
    assert stats["mean_rate"].shape == (CAP,) and stats["rate_cov"].shape == (CAP, CAP)
    assert stats["resid_cov"].shape == (CAP,) and stats["alive"].tolist() == [True] * N + [False] * SPARE
    assert torch.allclose(stats["rate_cov"], stats["rate_cov"].t())
    assert torch.allclose(stats["rate_cov"].diagonal(), stats["rate_var"], atol=1e-5)
    assert (stats["rate_var"] >= -1e-6).all() and stats["mean_rate"][N:].abs().sum() == 0
    core.reset_stats()
    assert core.growth_statistics()["tokens"] == 0


# ---------------------------------------------------------------------------
# E. MoE expert slots
# ---------------------------------------------------------------------------


def _moe_pair():
    """Two MoE layers with the same live weights; the second carries 2 spare slots."""

    torch.manual_seed(0)
    plain = SparseMoEFeedForward(_small_config())
    spare = SparseMoEFeedForward(_small_config(moe_spare_experts=2))
    with torch.no_grad():
        spare.gate.weight[:4].copy_(plain.gate.weight)
        for a, b in zip(plain.experts, spare.experts[:4]):
            for pa, pb in zip(a.parameters(), b.parameters()):
                pb.copy_(pa)
        for pa, pb in zip(plain.shared_expert.parameters(), spare.shared_expert.parameters()):
            pb.copy_(pa)
        plain.expert_bias.copy_(torch.tensor([17.0, 16.5, 17.7, 16.9]))
        spare.expert_bias[:4].copy_(plain.expert_bias)
        spare.expert_bias[4:].fill_(25.0)  # a stale bias must not rescue a dead slot
    return plain, spare


def test_spare_experts_are_never_selected_and_losses_match():
    plain, spare = _moe_pair()
    assert spare.n_routed == 6 and spare.alive_count() == 4 and spare.expert_alive.tolist() == [1, 1, 1, 1, 0, 0]
    x = torch.randn(2, 7, 32)
    for train in (True, False):
        plain.train(train); spare.train(train)
        a, b = plain(x), spare(x)
        assert torch.allclose(a, b, atol=1e-6)
        assert len(spare.last_expert_load) == 6 and spare.last_expert_load[4:].sum() == 0
        assert torch.allclose(spare.last_expert_load[:4], plain.last_expert_load)
        assert torch.allclose(spare.last_router_balance_loss, plain.last_router_balance_loss, atol=1e-6)
        assert torch.allclose(spare.last_router_z_loss, plain.last_router_z_loss, atol=1e-6)
        if train:
            assert torch.allclose(spare.aux_loss(), plain.aux_loss(), atol=1e-7)
    # the bias rule walks the alive slots exactly as before and leaves the dead ones alone
    plain.train(); spare.train()
    plain(x); spare(x)
    plain.update_router_bias(); spare.update_router_bias()
    assert torch.allclose(spare.expert_bias[:4], plain.expert_bias) and spare.expert_bias[4:].tolist() == [25.0, 25.0]


def test_expert_birth_and_death():
    _, moe = _moe_pair()
    moe.eval()
    x = torch.randn(2, 9, 32)
    torch.manual_seed(1)
    child = moe.birth_expert(parent=2, noise=0.01)
    assert child == 4 and moe.alive_count() == 5 and int(moe.expert_alive[4]) == 1
    assert float(moe.expert_bias[4]) == float(moe.expert_bias[2])
    parent_w = moe.experts[2].down_proj.weight
    assert torch.allclose(moe.experts[4].down_proj.weight, parent_w, atol=0.05 * float(parent_w.std()))
    assert not torch.equal(moe.experts[4].down_proj.weight, parent_w)  # noise broke the symmetry
    assert torch.allclose(moe.gate.weight[4], moe.gate.weight[2], atol=0.05 * float(moe.gate.weight[2].std()))
    with torch.no_grad():
        moe(x)
    assert moe.last_expert_load[4] > 0                      # the child is routable at once
    assert moe.birth_expert(parent=4) == 5 and moe.birth_expert(parent=0) is None  # capacity exhausted
    assert moe.birth_expert(parent=0) is None and moe.alive_count() == 6
    assert moe.kill_expert(5) and moe.kill_expert(4) and not moe.kill_expert(4)
    assert moe.alive_count() == 4 and moe.gate.weight[4].abs().sum() == 0 and float(moe.expert_bias[4]) == 0
    assert moe.kill_expert(3) and moe.kill_expert(2) and not moe.kill_expert(1)  # never below top_k alive
    with torch.no_grad():
        moe(x)
    assert moe.last_expert_load[2:].sum() == 0 and moe.last_expert_load[:2].sum() == pytest.approx(2.0)


def test_moe_dev_load_statistics():
    _, moe = _moe_pair()
    moe.collect_stats = True
    x = torch.randn(2, 5, 32)
    moe.train(); moe(x)
    assert int(moe.stat_batches) == 0
    moe.eval()
    with torch.no_grad():
        moe(x); moe(x)
    stats = moe.growth_statistics()
    assert stats["batches"] == 2 and stats["mean_load"].shape == (6,) and stats["n_alive"] == 4
    assert stats["mean_load"].sum() == pytest.approx(2.0, abs=1e-5)  # top-2 -> total load 2
    moe.reset_stats()
    assert moe.growth_statistics()["batches"] == 0


def test_model_with_spare_experts_round_trips_and_reports_loads(tmp_path):
    config = _small_config(moe_spare_experts=2)
    model = MiMoMixModel(config).eval()
    x = torch.randint(0, 97, (1, 6))
    with torch.no_grad():
        out = model(x, past_length=0)
    assert all(len(load) == 6 and sum(load[4:]) == 0 for load in out.telemetry["expert_load"])
    restored = MiMoMixModel(MiMoMixConfig.from_dict(json.loads(json.dumps(config.to_dict()))))
    restored.load_state_dict(model.state_dict())
    assert all(m.mlp.expert_alive.tolist() == [1, 1, 1, 1, 0, 0] for m in restored.layers if m.is_moe)


# ---------------------------------------------------------------------------
# F. depth growth
# ---------------------------------------------------------------------------


def test_pin_layout_for_growth_and_zero_new_blocks_identity():
    torch.manual_seed(0)
    config = _small_config(hybrid_ratio=1, n_layers=3)
    base = MiMoMixModel(config).eval()
    grown_config = pin_layout_for_growth(config, 1)
    assert grown_config.n_layers == 4 and grown_config.grow_layers == 1
    assert grown_config.global_layers == tuple(i for i, k in enumerate(base.layout) if k == "global")
    grown = MiMoMixModel(grown_config).eval()
    assert grown.layout[:3] == base.layout and grown.layout[-1] == "global"
    assert not grown.layers[3].is_moe and grown.layers[2].is_moe
    missing, unexpected = grown.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("layers.3.") for k in missing)
    x = torch.randint(0, 97, (2, 10))
    with torch.no_grad():
        assert not torch.allclose(grown(x, past_length=0).logits, base(x, past_length=0).logits)
        zeroed = grown.zero_new_blocks(3)
        assert zeroed == {"layers": 1, "o_proj": 1, "down_proj": 1}
        assert torch.allclose(grown(x, past_length=0).logits, base(x, past_length=0).logits, atol=1e-6)
    # the new block learns from the first step: its o_proj / down_proj get gradient
    grown.train()
    grown(x, labels=x).loss.backward()
    assert grown.layers[3].self_attn.o_proj.weight.grad.abs().sum() > 0
    assert grown.layers[3].mlp.down_proj.weight.grad.abs().sum() > 0
    # config survives a JSON round trip with the pinned layout
    payload = json.loads(json.dumps(grown_config.to_dict()))
    assert MiMoMixConfig(**payload).global_layers == grown_config.global_layers
    assert pin_layout_for_growth(config, 0).to_dict() == config.to_dict()
