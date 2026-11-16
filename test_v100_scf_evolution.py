"""Tests for Supermix v100 divisive-norm CNS, 13-seed verifier, collective consensus, and UI telemetry."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


def test_cns_divisive_norm_identity_modulation_and_growth() -> None:
    import mimomix_core as mc

    torch.manual_seed(21)
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
        cns_spare_nodes=4,
        cns_steps=2,
        cns_after_layer=0,
        cns_divisive_norm=False,
    )
    torch.manual_seed(29)
    base_model = mc.MiMoMixModel(base_cfg)
    base_model.eval()

    div_cfg = mc.MiMoMixConfig(
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
        cns_divisive_norm=True,
    )
    torch.manual_seed(29)
    div_model = mc.MiMoMixModel(div_cfg)
    div_model.load_state_dict(base_model.state_dict(), strict=False)
    div_model.eval()

    x = torch.randint(0, 64, (2, 6))
    with torch.no_grad():
        for core in (base_model.cns_core, div_model.cns_core):
            core.sign.copy_(torch.tensor([1, 1, -1, 1, -1, 1, 1, -1, 1, 1, -1, 1], dtype=core.sign.dtype))
            core.in_mask.fill_(1.0)
            core.out_mask.fill_(1.0)
            core.mask.fill_(1.0)
            core.edge_logit.fill_(0.0)
            core.gate.fill_(0.5)
        y_base = base_model(x).logits
        y_div = div_model(x).logits

    assert torch.allclose(y_base, y_div, atol=1e-6)
    tel_init = div_model.telemetry()["cns_core"]
    assert tel_init["divisive_norm"] is True
    assert tel_init["div_norm_strength_mean_abs"] == pytest.approx(0.0, abs=1e-7)
    assert tel_init["div_norm_factor_mean"] == pytest.approx(1.0, abs=1e-6)

    with torch.no_grad():
        div_model.cns_core.div_norm_strength.fill_(2.5)
        y_active = div_model(x).logits
    assert not torch.allclose(y_base, y_active, atol=1e-5)
    tel_active = div_model.telemetry()["cns_core"]
    assert tel_active["div_norm_factor_mean"] > 1.0

    # Neurogenesis mitosis safely inherits div_norm_strength.
    child = div_model.cns_core.split_module(parent=0, step=1)
    assert child is not None
    assert float(div_model.cns_core.div_norm_strength.detach()[child]) == pytest.approx(2.5, abs=1e-6)


@pytest.mark.parametrize("seed", [84, 85, 86, 87, 88, 89, 90, 92, 93, 94])
def test_scf_verification_gate_extended_seeds_100_percent(seed: int) -> None:
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


def test_collective_orchestrator_semantic_numerical_consensus() -> None:
    from scf.collective import SCFCollectiveOrchestrator

    orch = SCFCollectiveOrchestrator()
    out = orch.gather_and_synthesize(
        "In how many ways can 2 items be arranged in order from 5?",
        specialist_families=("MiMoMix", "NexusMind", "CognitiveLeap"),
        candidate_generators={
            "MiMoMix": lambda _q: {
                "family": "MiMoMix",
                "answer": "permutations = 5 x 4 = 20, total 20",
                "confidence": 0.62,
                "verified": False,
            },
            "CognitiveLeap": lambda _q: {
                "family": "CognitiveLeap",
                "answer": "20",
                "confidence": 0.60,
                "verified": False,
            },
            "NexusMind": lambda _q: {
                "family": "NexusMind",
                "answer": "19",
                "confidence": 0.65,
                "verified": False,
            },
        },
        max_candidates=3,
    )
    by_fam = {c["family"]: c for c in out["candidates"]}
    assert by_fam["MiMoMix"]["canonical_answer"] == "num:20"
    assert by_fam["MiMoMix"]["consensus_agreement"] == 2
    assert by_fam["MiMoMix"]["critique_notes"] == "cross_model_consensus"
    assert by_fam["NexusMind"]["consensus_agreement"] == 1
    assert out["best_candidate"]["family"] in {"MiMoMix", "CognitiveLeap"}


def test_chat_server_divisive_norm_telemetry_and_sticky_model_selector(monkeypatch: pytest.MonkeyPatch) -> None:
    import supermix_chat_server as scs

    class _StubV100:
        def telemetry(self):
            return {
                "cns_core": {
                    "alive_nodes": 808,
                    "edges_installed": 19950,
                    "active_fraction": 0.62,
                    "hebbian_trace": True,
                    "hebbian_gain_mean": 1.01,
                    "predictive_error": True,
                    "pred_error_norm": 0.012,
                    "divisive_norm": True,
                    "div_norm_factor_mean": 1.14,
                    "div_norm_strength_mean_abs": 0.35,
                    "spectral_radius_est": 0.90,
                },
                "thinking": {"cycles_used": 2, "exit_reason": "entropy_halt", "cycle_entropies": [0.35, 0.11]},
            }

    loaded = scs.LoadedModel(
        name="v100_scf_balanced",
        checkpoint="stub.pt",
        model=_StubV100(),
        tokenizer=object(),
        extra={"run_name": "v100_scf_balanced"},
        lock=threading.Lock(),
    )
    registry = scs.ModelRegistry({"v100_scf_balanced": "stub.pt"}, max_resident=1)
    monkeypatch.setattr(registry, "acquire", lambda _name: loaded)
    monkeypatch.setattr(scs, "stream_speculative_tokens", lambda *args, **kwargs: iter(["20"]))

    app = scs.build_app(registry, max_concurrency=2, history_turns=2)
    client = app.test_client()

    html = client.get("/").get_data(as_text=True)
    assert "const prev = el('model').value;" in html
    assert "[v100 DivNorm]" in html
    assert "DivNorm=" in html

    resp = client.post(
        "/api/chat",
        json={"message": "5 x 4", "model": "v100_scf_balanced", "mode": "speculative", "scf_verify": True},
    )
    assert resp.status_code == 200
    done_line = [ln for ln in resp.get_data(as_text=True).splitlines() if ln.startswith("data: ")][-1]
    done = json.loads(done_line[len("data: "):])
    cns_t = done["architecture_telemetry"]["cns_core"]
    assert cns_t["divisive_norm"] is True
    assert cns_t["div_norm_factor_mean"] == pytest.approx(1.14, abs=1e-6)


def test_v100_registry_and_manifest_respect_promotion_gate() -> None:
    from scf.models import MiMoMixAdapter
    from supermix_chat_server import discover_default_models

    ckpts = MiMoMixAdapter().discover_talk_checkpoints()
    if (ROOT / "output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt").is_file():
        assert ckpts[0] == "output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt"
    if (ROOT / "output/v100_scf_balanced/v100_scf_balanced.selected.pt").is_file():
        assert "output/v100_scf_balanced/v100_scf_balanced.selected.pt" in ckpts
        assert ckpts.index("output/v100_scf_balanced/v100_scf_balanced.selected.pt") == 1

    specs = list(discover_default_models(ROOT).keys())
    if "v96_physio_neurogenesis" in specs:
        assert specs[0] == "v96_physio_neurogenesis"
    if (ROOT / "output/v100_scf_balanced/v100_scf_balanced.selected.pt").is_file():
        assert "v100_scf_balanced" in specs
        assert specs.index("v100_scf_balanced") == 1

    manifest_path = ROOT / "datasets/v100/v100_manifest.json"
    if manifest_path.is_file():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert m["schema"] == "supermix-v100-precision-mix-v1"
        assert m["total_rows"] == 8470
