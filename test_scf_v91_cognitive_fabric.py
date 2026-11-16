"""Comprehensive verification suite for Supermix v91 Cognitive Fabric (SCF)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from mimomix_core import MiMoMixConfig, MiMoMixModel  # noqa: E402
from scf import (  # noqa: E402
    SCF_INTERNAL_NAME,
    SCF_VERSION,
    CognitiveFabricRuntime,
    SCFConfig,
    SCFFeatureFlags,
    SCFModelRegistry,
    resolve_decoding_policy,
)
from scf.adaptive import SCFAdaptiveComputeController  # noqa: E402
from scf.collective import SCFCollectiveOrchestrator  # noqa: E402
from scf.config import FLAG_NAMES  # noqa: E402
from scf.learning import SCFLearningGate  # noqa: E402
from scf.memory import SCFLayeredMemory  # noqa: E402
from scf.models import (  # noqa: E402
    ChampionNetAdapter,
    CognitiveLeapAdapter,
    MiMoMixAdapter,
    NexusMindAdapter,
    OmniCollectiveAdapter,
    QwenAdapter,
)
from scf.multimodal import SCFMultimodalBridge  # noqa: E402
from scf.observability import SCFObservabilityHub  # noqa: E402
from scf.reasoning import SCFReasoningCoordinator  # noqa: E402
from scf.routing import SCFRouter  # noqa: E402
from scf.verification import SCFVerificationGate  # noqa: E402


def test_phase_zero_artifacts_exist_and_pin_flagship_hashes():
    assert SCF_INTERNAL_NAME == "Cognitive Fabric"
    assert SCF_VERSION == "91.0.0-scf"

    for fname in (
        "SCF91_BASELINE_INVENTORY.json",
        "SCF91_FEATURE_PRESERVATION_MANIFEST.json",
        "SCF91_MODEL_COMPATIBILITY_MATRIX.json",
        "SCF91_REGRESSION_BASELINE.json",
    ):
        p = ROOT / fname
        assert p.is_file(), f"Missing Phase Zero artifact: {fname}"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "schema" in data

    baseline = json.loads((ROOT / "SCF91_REGRESSION_BASELINE.json").read_text(encoding="utf-8"))
    pins = baseline["checkpoint_sha256_pins"]
    assert len(pins) >= 5
    for rel, expected in pins.items():
        check = SCFObservabilityHub.verify_checkpoint_hash(ROOT / rel, expected)
        assert check["exists"] is True and check["verified"] is True, rel


def test_all_12_subpackages_and_6_model_adapters():
    _ = (
        CognitiveFabricRuntime,
        SCFRouter,
        SCFModelRegistry,
        SCFCollectiveOrchestrator,
        SCFLayeredMemory,
        SCFReasoningCoordinator,
        SCFVerificationGate,
        SCFMultimodalBridge,
        SCFAdaptiveComputeController,
        SCFLearningGate,
        SCFObservabilityHub,
        SCFConfig,
    )
    registry = SCFModelRegistry()
    families = registry.list_families()
    assert families == ["ChampionNet", "CognitiveLeap", "QwenLoRA", "OmniCollective", "MiMoMix", "NexusMind"]

    summary = registry.summary()
    for fam in families:
        assert summary[fam]["available_in_source"] is True

    # Honest reporting: OmniCollective has no local .pth in repo root/output by default
    assert isinstance(registry.get_adapter("OmniCollective"), OmniCollectiveAdapter)
    assert isinstance(registry.get_adapter("CognitiveLeap"), CognitiveLeapAdapter)
    assert isinstance(registry.get_adapter("QwenLoRA"), QwenAdapter)
    assert isinstance(registry.get_adapter("MiMoMix"), MiMoMixAdapter)
    assert isinstance(registry.get_adapter("NexusMind"), NexusMindAdapter)

    # Experimental head isolation guard on ChampionNetAdapter
    champ = registry.get_adapter("ChampionNet")
    assert isinstance(champ, ChampionNetAdapter)
    blocked = champ.build(model_size="liquid_spiking", allow_experimental=False)
    assert blocked["built"] is False and blocked["experimental_isolated"] is True
    allowed = champ.build(model_size="liquid_spiking", allow_experimental=True)
    assert allowed["built"] is True and allowed["experimental_isolated"] is True and allowed["production_ready"] is False


def test_feature_flags_toggle_independently_and_fallback_restores_baseline():
    # Default config is completely disabled (safe baseline)
    default_flags = SCFFeatureFlags()
    assert default_flags.enabled_list() == []
    runtime = CognitiveFabricRuntime()
    legacy_out = runtime.run_task(
        "What is 2 + 2?",
        legacy_fallback_fn=lambda q: f"legacy:{q}",
    )
    assert legacy_out["scf_active"] is False
    assert legacy_out["fallback_to_legacy"] is True
    assert legacy_out["final_answer"] == "legacy:What is 2 + 2?"
    assert legacy_out["receipt"]["fallback_to_legacy"] is True

    # Independent env parsing of all 10 flags
    for name in FLAG_NAMES:
        parsed = SCFFeatureFlags.from_env({name: "true"})
        assert getattr(parsed, name) is True
        others = [n for n in FLAG_NAMES if n != name]
        assert all(getattr(parsed, n) is False for n in others)


def test_speculative_decoding_blocked_with_adaptive_thinking_and_greedy_parity():
    flags = SCFFeatureFlags.all_enabled(speculative_decoding=True)
    blocked = resolve_decoding_policy(flags, adaptive_thinking=True, prefer_speculative=True)
    assert blocked["speculative_allowed"] is False
    assert blocked["mode"] == "greedy"
    assert "adaptive_thinking" in blocked["reason"]

    allowed = resolve_decoding_policy(flags, adaptive_thinking=False, prefer_speculative=True)
    assert allowed["speculative_allowed"] is True
    assert allowed["mode"] == "speculative_mtp"

    # CPU greedy vs speculative exact parity check through MiMoMixAdapter
    torch.manual_seed(0)
    cfg = MiMoMixConfig(
        vocab_size=64,
        hidden_size=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        intermediate_size=48,
        use_moe=False,
        n_mtp_layers=1,
        use_thinking_core=False,
        sliding_window=8,
        native_context=32,
        max_position_embeddings=32,
    )
    model = MiMoMixModel(cfg).eval()
    adapter = MiMoMixAdapter()
    spec_out = adapter.generate_tokens(
        model, [1, 5, 9, 13], flags=flags, max_new_tokens=6, adaptive_thinking=False
    )
    greedy_out = adapter.generate_tokens(
        model,
        [1, 5, 9, 13],
        flags=SCFFeatureFlags.all_enabled(speculative_decoding=False),
        max_new_tokens=6,
        adaptive_thinking=False,
    )
    assert spec_out["tokens"] == greedy_out["tokens"]
    assert spec_out["policy"]["speculative_allowed"] is True
    assert greedy_out["policy"]["speculative_allowed"] is False


def test_resilience_missing_checkpoint_and_promotion_gate_enforced(tmp_path: Path):
    qwen = QwenAdapter()
    missing_res = qwen.check_adapter_promotion(tmp_path / "missing_adapter")
    assert missing_res["valid"] is False and missing_res["promoted"] is False

    mm = SCFMultimodalBridge()
    mm_res = mm.prepare_multimodal_context(
        prompt="Inspect missing image",
        image_path=str(tmp_path / "missing.png"),
        video_path=str(tmp_path / "missing.mp4"),
        mesh_path=str(tmp_path / "missing.obj"),
    )
    assert "image_path_missing" in mm_res["warnings"]
    assert "video_path_missing" in mm_res["warnings"]
    assert "mesh_path_missing" in mm_res["warnings"]

    gate = SCFLearningGate()
    assert gate.automatic_promotion_enabled is False
    readiness = gate.inspect_readiness()
    assert readiness["automatic_promotion_enabled"] is False
    promo = gate.evaluate_promotion_request(
        baseline_accuracy=0.90,
        candidate_accuracy=0.95,
        regression_passed=True,
        certified_eval_count=100,
    )
    assert promo["automatic_promotion_enabled"] is False and promo["promoted"] is False
    assert promo["eligible_for_manual_promotion"] is True

    with pytest.raises(ValueError, match="automatic promotion"):
        SCFConfig(allow_automatic_promotion=True)


def test_memory_firewall_and_end_to_end_10_step_cognitive_loop():
    runtime = CognitiveFabricRuntime(
        SCFConfig(flags=SCFFeatureFlags.all_enabled(speculative_decoding=True))
    )
    admitted = runtime.memory.admit_user_memory(
        kind="fact",
        text="User works on Supermix neural connectomes",
        source_turn_id="turn_1",
    )
    assert admitted["claims_can_authorize_tools"] is False

    # 1. Science / Physics problem
    r_sci = runtime.run_task(
        "What is the impulse for force 12 N acting over time 4 s?",
        recent_turns=[{"user": "Hi", "assistant": "Hello"}],
    )
    assert r_sci["scf_active"] is True
    assert r_sci["verified"] is True
    assert r_sci["final_answer"] == "48"
    assert r_sci["receipt"]["total_latency_ms"] >= 0.0
    assert len(r_sci["receipt"]["receipt_sha256"]) == 64

    # 2. Code execution-free tracing problem
    r_code = runtime.run_task("What is the sum of range(2, 6)?")
    assert r_code["verified"] is True
    assert r_code["final_answer"] == "14"

    # 3. Causal / epistemic problem (triggers active inference + causal DAG + experimental isolation label)
    r_causal = runtime.run_task(
        "Please compute causal effect on Acceleration if we do(Force=50) in physics_newton"
    )
    assert r_causal["scf_active"] is True
    assert r_causal["reasoning"]["causal_dag"] is not None
    assert r_causal["reasoning"]["active_inference"] is not None
    assert r_causal["receipt"]["production_ready_claim"] is False
    assert "NexusMind_ActiveInferenceController" in r_causal["receipt"]["experimental_components_used"]

    # 4. Connectome domain routing
    r_cns = runtime.run_task("male CNS neuron count for cell type KCg")
    assert r_cns["routing"]["assessment"]["domain"] == "connectome"
    assert r_cns["routing"]["selected_family"] == "MiMoMix"
