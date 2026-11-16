"""v92: temporal connectome core, frozen-trunk rig, null ensemble, readout.

Small synthetic models only; the real graph npz (342 KB) is read where the
rig needs its 512-module wiring.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))

import malecns_connectome as mc  # noqa: E402
import mimomix_text as text_utils  # noqa: E402
import v92_temporal_connectome as v92  # noqa: E402
from mimomix_core import ConnectomeCore, MiMoMixConfig, MiMoMixModel  # noqa: E402
from train_mimomix_talk import CHECKPOINT_SCHEMA  # noqa: E402

N = 24


def _graph(seed=0):
    rng = np.random.default_rng(seed)
    roles = ["sensory"] * 4 + ["central"] * 14 + ["descending"] * 3 + ["output"] * 3
    sign = np.where(rng.random(N) < 0.6, 1.0, -1.0)
    matrix = rng.random((N, N)) * (rng.random((N, N)) < 0.3)
    post, pre, fraction = mc.threshold_input_fraction(matrix, 0.01)
    return post, pre, fraction, sign, roles


def _core(temporal=True, steps=2, leak=0.2):
    torch.manual_seed(0)
    core = v92.TemporalConnectomeCore(16, N, inner_steps=steps, temporal=temporal, leak_init=leak)
    core.install(*_graph(), signed_radius=0.9)
    return core


def test_install_matches_signed_radius_and_io():
    core = _core()
    w = core.weight().detach().double().numpy()
    assert np.abs(np.linalg.eigvals(w)).max() == pytest.approx(0.9, rel=1e-4)
    assert core.info["input_nodes"] == 4 and core.info["output_nodes"] == 6
    assert core.info["signed_radius"] == pytest.approx(0.9, rel=1e-4)


def test_empty_wiring_is_a_leaky_memory():
    core = v92.TemporalConnectomeCore(16, N)
    post, pre, fraction, sign, roles = _graph()
    info = core.install(post[:0], pre[:0], fraction[:0], sign, roles)
    assert info["edges"] == 0 and float(core.weight().abs().sum()) == 0.0


def test_gate_zero_is_identity():
    core = _core()
    x = torch.randn(2, 9, 16)
    assert torch.equal(core(x), x)


def test_temporal_core_is_causal_and_carries_memory():
    core = _core()
    with torch.no_grad():
        core.gate.fill_(0.3)
        core.read_in.weight.normal_(0, 0.5)
    x = torch.randn(1, 10, 16)
    y = x.clone()
    y[0, 6] += 3.0
    a, b = core(x), core(y)
    assert torch.allclose(a[:, :6], b[:, :6])           # nothing before t=6 changes
    assert not torch.allclose(a[:, 7:], b[:, 7:])        # the future remembers t=6
    per_token = _core(temporal=False, steps=6, leak=0.5)
    with torch.no_grad():
        per_token.gate.fill_(0.3)
        per_token.read_in.weight.normal_(0, 0.5)
    c, d = per_token(x), per_token(y)
    assert torch.allclose(c[:, 7:], d[:, 7:])            # the v91 core has no memory


def test_per_token_mode_reproduces_v91_dynamics():
    config = MiMoMixConfig(vocab_size=50, hidden_size=16, n_layers=2, n_heads=2, n_kv_heads=1,
                           use_cns_core=True, cns_nodes=N, cns_steps=6, cns_after_layer=0)
    v91 = ConnectomeCore(config)
    v92_core = _core(temporal=False, steps=6, leak=0.5)
    with torch.no_grad():
        for name in ("edge_logit", "node_bias", "leak_logit", "gate"):
            getattr(v91, name).copy_(getattr(v92_core, name))
        v91.read_in.weight.copy_(v92_core.read_in.weight)
        v91.read_out.weight.copy_(v92_core.read_out.weight)
        for name in ("mask", "sign", "in_mask", "out_mask"):
            getattr(v91, name).copy_(getattr(v92_core, name))
        v92_core.gate.normal_(0, 0.3)
        v91.gate.copy_(v92_core.gate)
    x = torch.randn(3, 5, 16)
    assert torch.allclose(v91(x), v92_core(x), atol=1e-6)


def test_forward_from_replays_the_model_exactly():
    torch.manual_seed(2)
    config = MiMoMixConfig(vocab_size=97, hidden_size=32, n_layers=5, n_heads=4, n_kv_heads=2,
                           intermediate_size=48, moe_intermediate_size=16, n_routed_experts=4, moe_top_k=2,
                           n_mtp_layers=1, sliding_window=8, native_context=32, max_position_embeddings=32,
                           thinking_latent_dim=8, hybrid_ratio=3)
    model = MiMoMixModel(config).eval()
    x = torch.randint(0, 97, (2, 20))
    with torch.no_grad():
        full = model(x).logits
        replay = v92.forward_from(model, v92.trunk_until(model, x, 2), 3)
    assert torch.allclose(full, replay, atol=1e-6)


def test_wsd_schedule_shape():
    lrs = [v92.wsd_lr(s, 100, 1.0) for s in range(100)]
    assert lrs[0] == pytest.approx(0.2) and lrs[4] == pytest.approx(1.0)
    assert lrs[50] == 1.0 and lrs[99] == pytest.approx(0.05) and max(lrs) == 1.0


def test_plan_is_interleaved_and_complete():
    names = [s["name"] for s in v92.PLAN]
    assert len(names) == len(set(names)) == 16
    assert names[:2] == ["T_real_s1", "T_null_101"]
    assert sum(n.startswith("T_null") for n in names) == 10
    assert sum(n.startswith("T_real") for n in names) == 3


def test_end_to_end_rig_on_a_tiny_model(tmp_path, monkeypatch):
    rows = [(f"what is {a} plus {b}", f"{a} plus {b} is {a + b}") for a in range(12) for b in range(12)]
    tokenizer = text_utils.WordTokenizer.build([t for pair in rows for t in pair], max_vocab=512)
    torch.manual_seed(3)
    config = MiMoMixConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, n_layers=5, n_heads=4, n_kv_heads=2,
                           intermediate_size=48, moe_intermediate_size=16, n_routed_experts=4, moe_top_k=2,
                           n_mtp_layers=1, sliding_window=8, native_context=128, max_position_embeddings=128,
                           thinking_latent_dim=8, hybrid_ratio=3)
    model = MiMoMixModel(config).eval()
    ckpt = tmp_path / "tiny.pt"
    torch.save({"schema": CHECKPOINT_SCHEMA, "config": config.to_dict(), "state_dict": model.state_dict(),
                "tokenizer": tokenizer.to_dict()}, ckpt)
    cache = str(tmp_path / "cache")
    meta = v92.prepare(cache_dir=cache, checkpoint=str(ckpt), train_batches=3, batch_size=4, dev_rows=12,
                       threads=2, rows_override=(rows[:100], rows[100:]))
    assert meta["dev_rows"] == 12 and meta["dtype"] == "float32"
    runs = str(tmp_path / "runs")
    for spec in ({"name": "T_real_s1", "temporal": True, "wiring": "real", "seed": 1},
                 {"name": "T_empty_s1", "temporal": True, "wiring": "empty", "seed": 1},
                 {"name": "P_real_s1", "temporal": False, "wiring": "real", "seed": 1}):
        result = v92.run_one(spec, cache_dir=cache, out_dir=runs, checkpoint=str(ckpt), steps=4, threads=2)
        assert np.isfinite(result["delta_vs_v89"]) and len(result["delta_ci95"]) == 2
        assert result["wiring"]["signed_radius"] == pytest.approx(0.95, rel=1e-3) or spec["wiring"] == "empty"
    report = v92.report(out_dir=runs, output=str(tmp_path / "report.json"))
    assert "H1_branch_helps_frozen_v89" in report and "H3_memory_vs_per_token" in report
    assert "H4_wiring_vs_empty" in report and json.load(open(tmp_path / "report.json"))
