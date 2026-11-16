"""Tests for Supermix v98 predictive-error CNS, AST verifier, SCF CLI, and promotion gates."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def test_cns_predictive_error_default_off_and_zero_init_parity() -> None:
    import mimomix_core as mc

    torch.manual_seed(7)
    base_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        use_cns_core=True,
        cns_nodes=12,
        cns_steps=2,
        cns_after_layer=0,
        cns_predictive_error=False,
    )
    torch.manual_seed(11)
    base_model = mc.MiMoMixModel(base_cfg)
    base_model.eval()

    pred_cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        use_cns_core=True,
        cns_nodes=12,
        cns_steps=2,
        cns_after_layer=0,
        cns_predictive_error=True,
    )
    torch.manual_seed(11)
    pred_model = mc.MiMoMixModel(pred_cfg)
    pred_model.load_state_dict(base_model.state_dict(), strict=False)
    pred_model.eval()

    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        for core in (base_model.cns_core, pred_model.cns_core):
            core.in_mask.fill_(1.0)
            core.out_mask.fill_(1.0)
            core.mask.fill_(1.0)
            core.edge_logit.fill_(0.0)
            core.gate.fill_(0.5)
        y_base = base_model(x).logits
        y_pred = pred_model(x).logits
    assert torch.allclose(y_base, y_pred, atol=1e-6)

    tel = pred_model.telemetry()
    cns_tel = tel["cns_core"]
    assert cns_tel["predictive_error"] is True
    assert cns_tel["pred_error_strength_mean_abs"] == pytest.approx(0.0, abs=1e-7)
    assert cns_tel["pred_error_norm"] == pytest.approx(0.0, abs=1e-7)

    # Non-zero learned strength modulates rates and updates norm telemetry.
    with torch.no_grad():
        pred_model.cns_core.pred_error_strength.fill_(0.8)
        y_active = pred_model(x).logits
    assert not torch.allclose(y_base, y_active, atol=1e-5)
    tel_after = pred_model.telemetry()["cns_core"]
    assert tel_after["pred_error_strength_mean_abs"] > 0.5
    assert tel_after["pred_error_norm"] > 0.0


def test_cns_predictive_error_temporally_causal_and_growth_safe() -> None:
    import mimomix_core as mc

    cfg = mc.MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=64,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=1,
        use_cns_core=True,
        cns_nodes=12,
        cns_spare_nodes=4,
        cns_steps=2,
        cns_after_layer=0,
        cns_temporal=True,
        cns_predictive_error=True,
    )
    model = mc.MiMoMixModel(cfg)
    with torch.no_grad():
        model.cns_core.in_mask.fill_(1.0)
        model.cns_core.out_mask.fill_(1.0)
        model.cns_core.mask.fill_(1.0)
        model.cns_core.edge_logit.fill_(0.0)
        model.cns_core.pred_error_strength.fill_(0.42)
        model.cns_core.gate.fill_(0.5)
    assert int(model.cns_core.alive.sum()) == 8
    child = model.cns_core.split_module(parent=0, step=1)
    assert child is not None
    assert float(model.cns_core.pred_error_strength.detach()[child]) == pytest.approx(0.42, abs=1e-6)
    x = torch.randint(0, 64, (2, 5))
    with torch.no_grad():
        _ = model(x)
    assert model.telemetry()["cns_core"]["predictive_error"] is True


def test_nexus_solver_code_index_bugfix_and_bounded_ast_trace() -> None:
    from nexus_solver import NexusSolver
    import prompt_normaliser as pn

    solver = NexusSolver()

    # Regression for the old regex bug where nums[1] + nums[2] returned 12.
    r_index = solver.solve("nums = [10, 12, 4]\nr = nums[1] + nums[2] Final r?")
    assert r_index.solved and r_index.display_answer == "16"

    # Existing negative index handling remains exact.
    r_neg = solver.solve("nums = [11, 13, 4]\nr = nums[-1] Final r?")
    assert r_neg.solved and r_neg.formula_id == "code.neg_index" and r_neg.display_answer == "4"

    # Conditional, while-loop, and nested-loop AST traces.
    r_cond = solver.solve("a = 30; b = 20\nr = a - b if a > b else b - a Final r?")
    assert r_cond.solved and r_cond.display_answer == "10"

    r_while = solver.solve("x = 0\nwhile x <= 40: x = x + 11 Give the value of x.")
    assert r_while.solved and r_while.display_answer == "44"

    r_nested = solver.solve("c = 8\nfor i in range(2):\n    for j in range(2): c = c + 1 Give c.")
    assert r_nested.solved and r_nested.display_answer == "12"

    # Refusal and step-limit guards must hold.
    r_refuse = solver.solve("Do not calculate: x = 4\nfor i in range(3): x = x + 7")
    assert not r_refuse.solved

    r_inf = solver.solve("x = 0\nwhile x >= 0: x = x + 1 What is x?")
    assert not r_inf.solved

    # Permutation normalization no longer hijacked by "ways" into combination.
    norm_perm = pn.normalise("In how many ways can 2 items be arranged in order from 5?")
    assert norm_perm.rule == "permutations"
    r_perm = solver.solve(norm_perm.prompt)
    assert r_perm.solved and r_perm.display_answer == "20"

    r_comb = solver.solve("In how many ways can 2 items be chosen from 45?")
    assert r_comb.solved and r_comb.display_answer == "990"


@pytest.mark.parametrize("seed", [82, 83, 91])
def test_scf_verification_gate_41_of_41_and_refusal_guards(seed: int) -> None:
    import eval_problem_solving as ep
    import prompt_normaliser as pn
    from scf.verification import SCFVerificationGate

    gate = SCFVerificationGate()
    tasks = ep.generate_novel(len(ep.V89_BENCHMARK_TASKS) + len(ep.V93_NEW_TASKS), seed=seed)
    correct = 0
    for item in tasks:
        norm = pn.normalise(item.prompt)
        res = gate.verify(query=item.prompt, normalised_query=norm.prompt)
        ans = res.get("verified_answer")
        pred = ep.extract_answer(str(ans)) if ans is not None else None
        if res.get("verified") and pred is not None and ep.is_correct(pred, item.answer):
            correct += 1
    assert correct == len(tasks) == 41

    # Strict refusal preservation.
    for blocked in (
        "Check proof: x + 2 = 5",
        "Verify proof derivation: 25% of 232",
        "Do not calculate: What is 20% of 275, then add 46?",
    ):
        res_blocked = gate.verify(query=blocked)
        assert res_blocked["verified"] is False


def test_scf_cli_status_and_verify() -> None:
    from scf.__main__ import main

    buf_status = StringIO()
    with redirect_stdout(buf_status):
        rc_status = main(["status", "--json"])
    assert rc_status == 0
    status_data = json.loads(buf_status.getvalue())
    assert status_data["version"] == "91.0.0-scf"
    assert status_data["connectome_atlas_ready"] is True

    buf_ver = StringIO()
    with redirect_stdout(buf_ver):
        rc_ver = main(["verify", "nums = [10, 12, 4]\nr = nums[1] + nums[2] Final r?", "--json"])
    assert rc_ver == 0
    ver_data = json.loads(buf_ver.getvalue())
    assert ver_data["verified"] is True
    assert ver_data["verified_answer"] == "16"


def test_v98_registry_discovery_preserves_promoted_baseline() -> None:
    from scf.models import MiMoMixAdapter
    from supermix_chat_server import discover_default_models

    ckpts = MiMoMixAdapter().discover_talk_checkpoints()
    if (ROOT / "output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt").is_file():
        assert ckpts[0] == "output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt"
    if (ROOT / "output/v98_scf_predictive/v98_scf_predictive.selected.pt").is_file():
        assert "output/v98_scf_predictive/v98_scf_predictive.selected.pt" in ckpts
        assert ckpts.index("output/v98_scf_predictive/v98_scf_predictive.selected.pt") > 0

    specs = list(discover_default_models(ROOT).keys())
    if "v96_physio_neurogenesis" in specs:
        assert specs[0] == "v96_physio_neurogenesis"
    if (ROOT / "output/v98_scf_predictive/v98_scf_predictive.selected.pt").is_file():
        assert "v98_scf_predictive" in specs
    if (ROOT / "output/v99_scf_evolution/v99_scf_evolution.selected.pt").is_file():
        assert "output/v99_scf_evolution/v99_scf_evolution.selected.pt" in ckpts
        assert "v99_scf_evolution" in specs
        assert specs.index("v99_scf_evolution") > 0

    manifest_path = ROOT / "datasets/v99/v99_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "supermix-v99-evolution-mix-v1"
        assert manifest["total_rows"] >= 7000


def test_chat_server_scf_status_runtime_and_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    import supermix_chat_server as scs

    class _StubModel:
        def telemetry(self):
            return {
                "cns_core": {
                    "alive_nodes": 812,
                    "edges_installed": 19940,
                    "active_fraction": 0.61,
                    "hebbian_trace": True,
                    "hebbian_gain_mean": 1.02,
                    "predictive_error": True,
                    "pred_error_norm": 0.015,
                    "spectral_radius_est": 0.90,
                },
                "thinking": {
                    "cycles_used": 2,
                    "exit_reason": "entropy_halt",
                    "cycle_entropies": [0.41, 0.12],
                },
            }

    loaded = scs.LoadedModel(
        name="v99_scf_evolution",
        checkpoint="stub.pt",
        model=_StubModel(),
        tokenizer=object(),
        extra={"run_name": "v99_scf_evolution"},
        lock=threading.Lock(),
    )
    registry = scs.ModelRegistry({"v99_scf_evolution": "stub.pt"}, max_resident=1)
    monkeypatch.setattr(registry, "acquire", lambda _name: loaded)
    monkeypatch.setattr(scs, "stream_tokens", lambda *args, **kwargs: iter(["16"]))
    monkeypatch.setattr(scs, "stream_speculative_tokens", lambda *args, **kwargs: iter(["16"]))

    app = scs.build_app(registry, max_concurrency=2, history_turns=2)
    client = app.test_client()

    st_resp = client.get("/api/scf_status")
    assert st_resp.status_code == 200
    st_json = st_resp.get_json()
    assert st_json["version"] == "91.0.0-scf"
    assert st_json["connectome_atlas_ready"] is True
    assert "MiMoMix" in st_json["families"]

    bad_resp = client.post(
        "/api/chat",
        json={"message": "2 + 2", "model": "v99_scf_evolution", "scf_runtime": "yes"},
    )
    assert bad_resp.status_code == 400

    chat_resp = client.post(
        "/api/chat",
        json={
            "message": "nums = [10, 12, 4]\nr = nums[1] + nums[2] Final r?",
            "model": "v99_scf_evolution",
            "mode": "speculative",
            "scf_verify": True,
            "scf_runtime": True,
        },
    )
    assert chat_resp.status_code == 200
    raw = chat_resp.get_data(as_text=True)
    done_line = [ln for ln in raw.splitlines() if ln.startswith("data: ")][-1]
    done = json.loads(done_line[len("data: "):])
    assert done["scf_check"]["verified"] is True
    assert done["scf_check"]["verified_answer"] == "16"
    assert done["scf_check"]["proof_capsule"] is not None
    assert done["scf_runtime"]["scf_active"] is True
    assert done["scf_runtime"]["verified"] is True
    assert done["architecture_telemetry"]["cns_core"]["predictive_error"] is True

    bench_resp = client.post(
        "/api/benchmark",
        json={"model": "v99_scf_evolution", "problems": 4, "seed": 82, "scf_verify": True},
    )
    assert bench_resp.status_code == 200
    bench = bench_resp.get_json()
    assert bench["problems"] == 4
    assert bench["scf_hybrid_correct"] == 4
    assert bench["scf_hybrid_accuracy"] == 1.0

    # Verify served inline HTML/JS has zero JS syntax errors.
    import re
    import shutil
    import subprocess
    import tempfile

    page_html = client.get("/").get_data(as_text=True)
    m_js = re.search(r"<script>(.*)</script>", page_html, flags=re.DOTALL)
    assert m_js is not None
    assert "buf.indexOf('\n\n')" not in m_js.group(1)
    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as tf:
            tf.write(m_js.group(1))
            js_path = Path(tf.name)
        try:
            res = subprocess.run([node, "--check", str(js_path)], capture_output=True, text=True)
            assert res.returncode == 0, res.stderr
        finally:
            js_path.unlink(missing_ok=True)
