"""Phase Zero Forensic Audit for Supermix v91 Cognitive Fabric (SCF).

Inspects the repository without modifying any existing model files or weights
and writes the four mandatory Phase Zero baseline artifacts:
- SCF91_BASELINE_INVENTORY.json
- SCF91_FEATURE_PRESERVATION_MANIFEST.json
- SCF91_MODEL_COMPATIBILITY_MATRIX.json
- SCF91_REGRESSION_BASELINE.json
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))


FLAGSHIP_HASH_CHECKPOINTS = (
    "output/v80_omni/v80_omni.pt",
    "output/v89_corpus/v89_corpus.pt",
    "output/v91_cns_connectome/v91_cns_connectome.selected.pt",
    "output/v93_neurogenesis/v93_neurogenesis.selected.pt",
    "output/v96_physio_neurogenesis/v96_physio_neurogenesis.pt",
    "output/hf_v56_publish/v56b_randslots_entropy.pt",
    "output/hf_v57_publish/v57_talk_v2.pt",
    "artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth",
    "artifacts/v52_initialization/champion_model_chat_v52_initialized.pth",
    "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth",
)

KEY_ENTRYPOINTS = (
    "source/supermix_multimodel_desktop_app.py",
    "source/supermix_multimodel_web_app.py",
    "source/chat_app.py",
    "source/chat_web_app.py",
    "source/qwen_chat_web_app.py",
    "source/mimomix_talk_web_app.py",
    "source/mimomix_reasoner_web_app.py",
    "source/supermix_chat_server.py",
    "source/nexus_api.py",
    "source/nexus_cli.py",
    "source/train_mimomix_generalisation.py",
    "source/train_supervised.py",
)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _git_info(repo_root: Path) -> Dict[str, Any]:
    def _run(args: Sequence[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {
        "branch": _run(["branch", "--show-current"]) or None,
        "head_commit": _run(["rev-parse", "HEAD"]) or None,
        "tracked_files_count": len(_run(["ls-files"]).splitlines()) if _run(["ls-files"]) else 0,
    }


def _collect_packages(names: Iterable[str]) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in names:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _extract_model_variants_classes(path: Path) -> List[str]:
    if not path.is_file():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]


def _scan_weight_artifacts(repo_root: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    hash_set = {Path(p).as_posix() for p in FLAGSHIP_HASH_CHECKPOINTS}
    for ext in ("*.pt", "*.pth", "*.safetensors"):
        for path in sorted(repo_root.rglob(ext)):
            if any(part in {".git", "venv", ".venv", "__pycache__"} for part in path.parts):
                continue
            rel = path.relative_to(repo_root).as_posix()
            stat = path.stat()
            item: Dict[str, Any] = {
                "path": rel,
                "format": path.suffix.lstrip(".").lower(),
                "size_bytes": int(stat.st_size),
                "size_mb": round(float(stat.st_size) / (1024.0 * 1024.0), 3),
                "sha256": _sha256_file(path) if rel in hash_set else None,
            }
            records.append(item)
    return records


def _inspect_pt_metadata(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive file inspection
        return {"load_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(payload, dict):
        cfg = payload.get("config") or {}
        tok = payload.get("tokenizer") or {}
        sd = payload.get("state_dict") or {}
        extra = payload.get("extra") or {}
        return {
            "schema": payload.get("schema"),
            "top_keys": sorted(str(k) for k in payload.keys()),
            "vocab_size": cfg.get("vocab_size", len(tok.get("tokens", [])) if isinstance(tok, dict) else None),
            "hidden_size": cfg.get("hidden_size"),
            "n_layers": cfg.get("n_layers"),
            "n_heads": cfg.get("n_heads"),
            "n_kv_heads": cfg.get("n_kv_heads"),
            "n_routed_experts": cfg.get("n_routed_experts"),
            "use_cns_core": cfg.get("use_cns_core", False),
            "cns_nodes": cfg.get("cns_nodes"),
            "cns_spare_nodes": cfg.get("cns_spare_nodes"),
            "tensor_count": len(sd) if isinstance(sd, dict) else None,
            "steps": extra.get("steps", extra.get("best_step")) if isinstance(extra, dict) else None,
        }
    return {"payload_type": type(payload).__name__}


def run_phase_zero_audit(repo_root: Path = ROOT) -> Dict[str, Path]:
    """Perform the complete Phase Zero audit and write all four JSON manifests."""

    root = repo_root.resolve()
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    weight_artifacts = _scan_weight_artifacts(root)
    checkpoints_by_path = {row["path"]: row for row in weight_artifacts}

    model_variant_classes = _extract_model_variants_classes(root / "source" / "model_variants.py")
    champion_heads = [
        name for name in model_variant_classes
        if ("ExpertHead" in name or "ClassifierHead" in name or name.startswith("ChampionNet"))
        and "CognitiveLeap" not in name
    ]
    cognitive_leap_classes = [name for name in model_variant_classes if "CognitiveLeap" in name]

    omni_models = sorted(p.relative_to(root).as_posix() for p in (root / "source").glob("omni_collective*.py"))
    qwen_modules = sorted(p.relative_to(root).as_posix() for p in (root / "source").glob("*qwen*.py"))
    mimomix_modules = sorted(p.relative_to(root).as_posix() for p in (root / "source").glob("mimomix*.py"))
    nexus_modules = sorted(p.relative_to(root).as_posix() for p in (root / "source").glob("nexus*.py"))
    test_suites = sorted(p.name for p in root.glob("test_*.py"))
    source_modules = sorted(p.relative_to(root).as_posix() for p in (root / "source").glob("*.py"))
    datasets_jsonl = sorted(
        p.relative_to(root).as_posix()
        for p in (root / "datasets").rglob("*.jsonl")
    ) if (root / "datasets").exists() else []

    from multimodel_catalog import MODEL_SPECS  # type: ignore

    catalog_specs = [
        {
            "key": spec.key,
            "label": spec.label,
            "family": spec.family,
            "kind": spec.kind,
            "capabilities": list(spec.capabilities),
            "preferred_weights": list(spec.preferred_weights),
        }
        for spec in MODEL_SPECS
    ]

    flagship_metadata: Dict[str, Any] = {}
    for rel in FLAGSHIP_HASH_CHECKPOINTS:
        entry = checkpoints_by_path.get(rel)
        if entry is None:
            flagship_metadata[rel] = {"present_on_disk": False}
            continue
        meta = dict(entry)
        meta["present_on_disk"] = True
        if rel.endswith(".pt"):
            meta["inspected"] = _inspect_pt_metadata(root / rel)
        flagship_metadata[rel] = meta

    manifest_path = root / "source" / "studio_runtime_manifest.json"
    studio_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    # 1. SCF91_BASELINE_INVENTORY.json
    baseline_inventory: Dict[str, Any] = {
        "schema": "supermix-scf91-baseline-inventory-v1",
        "generated_at": now_iso,
        "git": _git_info(root),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "packages": _collect_packages((
                "torch", "numpy", "fastapi", "uvicorn", "pydantic", "starlette",
                "httpx", "transformers", "peft", "safetensors", "pytest", "ruff", "Pillow",
            )),
        },
        "entrypoints": [
            {"path": rel, "exists": (root / rel).is_file()} for rel in KEY_ENTRYPOINTS
        ],
        "repository_counts": {
            "source_python_modules": len(source_modules),
            "root_test_suites": len(test_suites),
            "weight_artifacts_total": len(weight_artifacts),
            "dataset_jsonl_files": len(datasets_jsonl),
        },
        "subsystems": {
            "championnet_architectures_and_heads": champion_heads,
            "cognitive_leap_classes": cognitive_leap_classes,
            "omni_collective_modules": omni_models,
            "multimodel_catalog_specs": catalog_specs,
            "qwen_modules": qwen_modules,
            "mimomix_modules": mimomix_modules,
            "nexusmind_modules": nexus_modules,
            "multimodal_modules": [
                m for m in (
                    "source/image_feature_utils.py",
                    "source/image_recognition_model.py",
                    "source/dcgan_image_model.py",
                    "source/video_feature_utils.py",
                    "source/mesh_feature_utils.py",
                    "source/three_d_generation_model.py",
                    "source/protein_folding_model.py",
                    "source/mattergen_generation_model.py",
                    "source/mimomix_multimodal.py",
                )
                if (root / m).is_file()
            ],
            "memory_and_conversation_systems": [
                m for m in (
                    "source/memory_authority.py",
                    "source/multimodel_memory.py",
                    "source/conversation_state.py",
                    "source/conversation_directive.py",
                    "source/chat_memory.py",
                    "source/mimomix_reasoner.py",
                )
                if (root / m).is_file()
            ],
            "verification_and_evaluation_infrastructure": [
                m for m in (
                    "source/nexus_solver.py",
                    "source/science_plan.py",
                    "source/answer_check.py",
                    "source/prompt_normaliser.py",
                    "source/grounding_runtime.py",
                    "source/eval_problem_solving.py",
                    "source/nexus_proof.py",
                    "source/nexus_proof_verification.py",
                    "source/nexus_independent_checker.py",
                    "source/nexus_conformal_stopping.py",
                    "source/qwen_adapter_promotion.py",
                    "source/run_v56_promotion_gate.py",
                    "source/v94_connectome_audit.py",
                )
                if (root / m).is_file()
            ],
        },
        "flagship_checkpoints": flagship_metadata,
        "all_weight_artifacts": weight_artifacts,
        "deployment_interfaces": [
            "FastAPI / Uvicorn NexusMind API (source/nexus_api.py)",
            "Supermix Studio Web App & Desktop Executable (source/supermix_multimodel_web_app.py, source/supermix_multimodel_desktop_app.py)",
            "Supermix Chat Web/Desktop/Server (source/chat_web_app.py, source/supermix_chat_server.py, source/supermix_chat_desktop_app.py)",
            "Qwen Chat Web/Desktop (source/qwen_chat_web_app.py, source/qwen_chat_desktop_app.py)",
            "MiMoMix Talk & Reasoner Web Apps (source/mimomix_talk_web_app.py, source/mimomix_reasoner_web_app.py)",
            "Static Nexus Studio UI (web_static/nexus_studio.html)",
        ],
    }

    # 2. SCF91_FEATURE_PRESERVATION_MANIFEST.json
    feature_preservation: Dict[str, Any] = {
        "schema": "supermix-scf91-feature-preservation-v1",
        "generated_at": now_iso,
        "policy": "additive_zero_deletion_default_off",
        "preserved_families": {
            "ChampionNet": {
                "source": "source/model_variants.py",
                "scf_adapter": "source/scf/models/__init__.py::ChampionNetAdapter",
                "scf_layers": ["source/scf/routing/__init__.py", "source/scf/reasoning/__init__.py"],
                "preserved_capabilities": [
                    "HierarchicalMoEClassifierHead", "DeepExpertClassifierHead", "ExpertChoiceClassifierHead",
                    "SmarterExpertClassifierHead", "ThoughtExpertClassifierHead", "RecursiveThoughtExpertHead",
                    "ReflexiveThoughtExpertHead", "MetaCognitiveExpertHead", "TreeOfThoughtExpertHead",
                    "ConsensusExpertHead", "DeliberativeAlignmentExpertHead", "NeurogenesisExpertHead",
                    "CognitiveSingularityExpertHead", "TranscendentArchitectExpertHead",
                    "OmniversalQuantumExpertHead", "FractalGenesisExpertHead", "LiquidSpikingTensorExpertHead",
                    "ActiveInferenceExpertHead", "TestTimeDifferentialExpertHead", "HolographicStateSpaceExpertHead",
                    "PaperFusionExpertHead", "load_weights_for_model", "detect_model_size_from_state_dict",
                ],
                "notes": "Experimental heads (LiquidSpiking, OmniversalQuantum, FractalGenesis) remain isolated and tagged experimental_isolated.",
            },
            "CognitiveLeap": {
                "source": "source/model_variants.py, source/chat_app.py",
                "scf_adapter": "source/scf/models/__init__.py::CognitiveLeapAdapter",
                "scf_layers": ["source/scf/adaptive/__init__.py"],
                "preserved_capabilities": [
                    "CognitiveLeapExpertHead", "CognitiveLeapUltraExpertHead", "CognitiveLeapV52ExpertHead",
                    "resolve_runtime_compute_cycles", "evaluate_runtime_compute_budgets",
                    "select_auto_runtime_compute_budget", "progressive_auto_compute_forward",
                    "prediction stability margin and rank-depth early stopping guards",
                ],
            },
            "OmniCollective": {
                "source": "source/omni_collective_model.py, source/omni_collective_v41_model.py, source/omni_collective_v42_model.py, source/multimodel_catalog.py, source/score_fusion.py",
                "scf_adapter": "source/scf/models/__init__.py::OmniCollectiveAdapter",
                "scf_layers": ["source/scf/routing/__init__.py", "source/scf/collective/__init__.py"],
                "preserved_capabilities": [
                    "OmniCollective v1-v8, v40-v42, v46-v48 catalog & engine wrappers",
                    "Domain-aware specialist routing (math, coding, protein, materials, 3d, vision, gan)",
                    "Signal calibration & consensus fusion (zmuv, borda, consensus_multiplier)",
                ],
            },
            "QwenLoRA": {
                "source": "source/qwen_supermix_pipeline.py, source/qwen_adapter_promotion.py, source/qwen_chat_web_app.py",
                "scf_adapter": "source/scf/models/__init__.py::QwenAdapter",
                "scf_layers": ["source/scf/learning/__init__.py"],
                "preserved_capabilities": [
                    "Artifact-attested adapter loading",
                    "Explicit promotion gate enforcement (never auto-promoted)",
                ],
            },
            "MiMoMix": {
                "source": "source/mimomix_reasoner.py (v53/v56), source/mimomix_core.py (v57-v96), source/mimomix_decoding.py, source/mimomix_multimodal.py",
                "scf_adapter": "source/scf/models/__init__.py::MiMoMixAdapter",
                "scf_layers": ["source/scf/memory/__init__.py", "source/scf/adaptive/__init__.py", "source/scf/multimodal/__init__.py"],
                "preserved_capabilities": [
                    "v53/v56 LatentStateReasoner slot-transition memory",
                    "Hybrid SWA/global GQA attention + YaRN/NTK RoPE + Differential Attention + MLA + MoD",
                    "Sparse MoE + aux-loss-free bias update + Multi-Token Prediction (MTP)",
                    "Bit-exact greedy == MTP self-speculative decoding (blocked when adaptive_thinking=True)",
                    "Male-CNS bilateral ConnectomeCore + dual_soma, commissural_gate, glial_homeostasis, homeostatic_radius",
                ],
            },
            "NexusMind": {
                "source": "source/nexus_engine.py, source/nexus_*.py",
                "scf_adapter": "source/scf/models/__init__.py::NexusMindAdapter",
                "scf_layers": ["source/scf/reasoning/__init__.py", "source/scf/verification/__init__.py", "source/scf/observability/__init__.py"],
                "preserved_capabilities": [
                    "Graph-of-Thought (nexus_got.py)", "Multi-Agent Swarm (nexus_swarm.py)",
                    "Active Inference & Epistemic Planning (nexus_active_inference.py, nexus_epistemics.py)",
                    "Pearlian Causal DAG (nexus_causal_dag.py)", "Reflexion & Diffusion-of-Thought",
                    "Conformal Stopping + CRC + AURC (nexus_conformal_stopping.py)",
                    "Formal Proof Verification & Deterministic Science/Math Solver (nexus_proof.py, nexus_solver.py, nexus_independent_checker.py)",
                    "v95 Tether/Faultline/Crosswire/Mnemorph forensic diagnostics (answer_authority=False)",
                ],
            },
        },
    }

    # 3. SCF91_MODEL_COMPATIBILITY_MATRIX.json
    compatibility_matrix: Dict[str, Any] = {
        "schema": "supermix-scf91-model-compatibility-v1",
        "generated_at": now_iso,
        "global_invariants": {
            "speculative_with_adaptive_thinking_allowed": False,
            "unverified_weights_claimed_trained": False,
            "silent_tensor_mutation_allowed": False,
        },
        "families": [
            {
                "family": "MiMoMix_v80_to_v96",
                "class": "source/mimomix_core.py::MiMoMixModel",
                "checkpoint_format": ".pt",
                "on_disk_weights": [
                    rel for rel in (
                        "output/v80_omni/v80_omni.pt",
                        "output/v89_corpus/v89_corpus.pt",
                        "output/v91_cns_connectome/v91_cns_connectome.selected.pt",
                        "output/v93_neurogenesis/v93_neurogenesis.selected.pt",
                        "output/v96_physio_neurogenesis/v96_physio_neurogenesis.pt",
                    )
                    if rel in checkpoints_by_path
                ],
                "tensor_contract": "hidden=320 (v89/v91/v93/v96), 5-6 layers, 64+spare MoE top-2, vocab 8679->9415, cns_nodes=832 (64 spare)",
                "decoding_policy": "speculative_generate allowed only when adaptive_thinking=False",
                "maturity_status": "trained_checkpoint_present",
                "production_authority": "experimental_connectome_extension_verified_in_eval",
            },
            {
                "family": "MiMoMix_v53_v56_LatentReasoner",
                "class": "source/mimomix_reasoner.py::LatentStateReasoner",
                "checkpoint_format": ".pt",
                "on_disk_weights": [rel for rel in ("output/hf_v56_publish/v56b_randslots_entropy.pt",) if rel in checkpoints_by_path],
                "tensor_contract": "222 discrete reasoning state slots",
                "decoding_policy": "latent slot transition forward (non-speculative)",
                "maturity_status": "trained_checkpoint_present",
                "production_authority": "promotion_gated",
            },
            {
                "family": "ChampionNet_Legacy_and_CognitiveLeap_v50_v52",
                "class": "source/model_variants.py::build_model",
                "checkpoint_format": ".pth",
                "on_disk_weights": [
                    rel for rel in (
                        "artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth",
                        "artifacts/v52_initialization/champion_model_chat_v52_initialized.pth",
                        "output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth",
                    )
                    if rel in checkpoints_by_path
                ],
                "tensor_contract": "256-dim / expanded feature backbones with detect_model_size_from_state_dict inference",
                "decoding_policy": "progressive_auto_compute_forward with prediction stability guard",
                "maturity_status": "mixed_trained_and_experimental",
                "experimental_subclasses": [
                    "LiquidSpikingTensorExpertHead",
                    "OmniversalQuantumExpertHead",
                    "FractalGenesisExpertHead",
                ],
            },
            {
                "family": "Qwen_LoRA_Adapters",
                "class": "source/qwen_adapter_promotion.py",
                "checkpoint_format": ".safetensors",
                "on_disk_weights": [
                    r["path"] for r in weight_artifacts if r["format"] == "safetensors"
                ],
                "tensor_contract": "PEFT LoRA adapter + bundled Qwen2.5/Qwen base model",
                "decoding_policy": "Attested runtime adapter only after validate_promoted_adapter",
                "maturity_status": "adapter_checkpoints_present_promotion_gated",
            },
            {
                "family": "OmniCollective_Multimodal",
                "class": "source/omni_collective_v42_model.py::OmniCollectiveEngineV42",
                "checkpoint_format": ".pth (optional external bundles in DEFAULT_MODELS_DIR)",
                "on_disk_weights": [],
                "tensor_contract": "Text + image feature vectors with domain routing heads",
                "decoding_policy": "Multi-pass budget routing (short/medium/deep); degrades gracefully if .pth absent",
                "maturity_status": "source_present_external_weights_optional",
            },
            {
                "family": "NexusMind_Symbolic_and_Epistemic",
                "class": "source/nexus_engine.py::NexusEngine",
                "checkpoint_format": "deterministic_python_plus_numpy",
                "on_disk_weights": ["built_in_no_external_weights_required"],
                "tensor_contract": "Structured request/receipt dataclasses + NumPy distributions",
                "decoding_policy": "Independent deterministic verifier + advisory answer_authority=False diagnostics",
                "maturity_status": "production_verified_deterministic_core_plus_experimental_diagnostics",
            },
        ],
    }

    # 4. SCF91_REGRESSION_BASELINE.json
    regression_baseline: Dict[str, Any] = {
        "schema": "supermix-scf91-regression-baseline-v1",
        "generated_at": now_iso,
        "studio_manifest": {
            "schema_version": studio_manifest.get("schema_version"),
            "app_version": studio_manifest.get("app_version"),
            "declared_modules_count": len(studio_manifest.get("modules", [])),
        },
        "checkpoint_sha256_pins": {
            rel: checkpoints_by_path[rel]["sha256"]
            for rel in FLAGSHIP_HASH_CHECKPOINTS
            if rel in checkpoints_by_path and checkpoints_by_path[rel]["sha256"]
        },
        "dark_test_baseline": 30,
        "core_gate_suites": [
            "test_v85_release_contract.py",
            "test_studio_runtime_manifest.py",
            "test_nexus_studio_contract.py",
            "test_mimomix_core.py",
            "test_mimomix_v82.py",
            "test_v91_connectome_core.py",
            "test_v92_temporal_connectome.py",
            "test_v93_core.py",
            "test_v93_neurogenesis.py",
            "test_v93_trainer.py",
            "test_v94_growth_policy.py",
            "test_v94_connectome_audit.py",
            "test_prepare_v94.py",
            "test_v95_neurogenesis_and_connectome.py",
            "test_nexus_v95_ecosystem.py",
            "test_v96_physio_training_and_criticality.py",
        ],
    }

    outputs = {
        "inventory": root / "SCF91_BASELINE_INVENTORY.json",
        "preservation": root / "SCF91_FEATURE_PRESERVATION_MANIFEST.json",
        "compatibility": root / "SCF91_MODEL_COMPATIBILITY_MATRIX.json",
        "regression": root / "SCF91_REGRESSION_BASELINE.json",
    }
    outputs["inventory"].write_text(json.dumps(baseline_inventory, indent=2) + "\n", encoding="utf-8")
    outputs["preservation"].write_text(json.dumps(feature_preservation, indent=2) + "\n", encoding="utf-8")
    outputs["compatibility"].write_text(json.dumps(compatibility_matrix, indent=2) + "\n", encoding="utf-8")
    outputs["regression"].write_text(json.dumps(regression_baseline, indent=2) + "\n", encoding="utf-8")
    return outputs


def main() -> int:
    outputs = run_phase_zero_audit(ROOT)
    print(json.dumps({k: str(v.name) for k, v in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
