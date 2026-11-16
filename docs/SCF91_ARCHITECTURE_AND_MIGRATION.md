# Supermix v91 — Cognitive Fabric (`SCF`) Architecture & Migration Guide

## 1. Overview

**Supermix v91 (`SCF`, internal name Cognitive Fabric)** is a strictly additive, feature-flagged cognitive meta-architecture built on branch `scf-v91-cognitive-fabric`. It coordinates all six historical and frontier Supermix model families—`ChampionNet`, `Cognitive Leap`, `Omni Collective`, `Qwen LoRA`, `MiMoMix (v53/v56–v96)`, and `NexusMind (v71–v95)`—without removing, overwriting, or mutating any existing model weights, checkpoints, datasets, or tensor dimensions.

---

## 2. Package Organization (`source/scf/`)

| Subpackage | Module | Responsibility |
|---|---|---|
| `core/` | `source/scf/core/__init__.py` | `CognitiveFabricRuntime`: 10-step task execution loop and legacy fallback |
| `config/` | `source/scf/config/__init__.py` | `SCFFeatureFlags` (all default `False`), `SCFConfig`, `resolve_decoding_policy()` |
| `models/` | `source/scf/models/__init__.py` | `SCFModelRegistry` and adapters for all 6 model families with isolation guards |
| `routing/` | `source/scf/routing/__init__.py` | `SCFRouter`: complexity estimation, canonical normalization, hierarchical routing |
| `collective/` | `source/scf/collective/__init__.py` | `SCFCollectiveOrchestrator`: multi-candidate generation, critique, score fusion |
| `memory/` | `source/scf/memory/__init__.py` | `SCFLayeredMemory`: working state, `conversation_state`, `memory_authority` firewall, v53/v56 latent slots |
| `reasoning/` | `source/scf/reasoning/__init__.py` | `SCFReasoningCoordinator`: Graph-of-Thought, Causal DAG, Active Inference, Reflexion |
| `verification/` | `source/scf/verification/__init__.py` | `SCFVerificationGate`: `grounding_runtime`, `science_plan`, `nexus_solver`, `nexus_independent_checker`, `nexus_proof` |
| `multimodal/` | `source/scf/multimodal/__init__.py` | `SCFMultimodalBridge`: image, video, 3D mesh, protein, materials, and `use_multimodal` projector |
| `adaptive/` | `source/scf/adaptive/__init__.py` | `SCFAdaptiveComputeController`: Cognitive Leap stability + `ConformalStoppingController` + decode guard |
| `learning/` | `source/scf/learning/__init__.py` | `SCFLearningGate`: promotion-gated validation (`automatic_promotion_enabled=False`) |
| `observability/` | `source/scf/observability/__init__.py` | `SCFObservabilityHub` & `SCFExecutionReceipt`: latency, traced memory, SHA-256 receipts, hash pinning |
| `audit_phase_zero.py` | `source/scf/audit_phase_zero.py` | Generates `SCF91_*` baseline inventory, preservation, compatibility, and regression manifests |

---

## 3. Feature Flags & Safe Defaults

Every SCF flag defaults to `False` in `SCFFeatureFlags` (and can be configured programmatically or via environment variables via `SCFFeatureFlags.from_env()`):

- `SCF_ENABLED` (master switch; when `False`, `CognitiveFabricRuntime.run_task()` delegates cleanly to legacy baseline behavior)
- `SCF_ROUTER_ENABLED`
- `SCF_HIERARCHICAL_MOE_ENABLED`
- `SCF_COLLECTIVE_REASONING_ENABLED`
- `SCF_LATENT_MEMORY_ENABLED`
- `SCF_ADAPTIVE_COMPUTE_ENABLED`
- `SCF_SPECULATIVE_DECODING_ENABLED`
- `SCF_VERIFICATION_ENABLED`
- `SCF_MULTIMODAL_ENABLED`
- `SCF_ACTIVE_INFERENCE_ENABLED`

### Decoding & Experimental Invariants
1. **Speculative + Adaptive Thinking Exclusion**: `resolve_decoding_policy()` refuses `speculative_generate` whenever `adaptive_thinking=True`, falling back to deterministic `greedy_generate` so MTP block-verification equivalence cannot silently break.
2. **Experimental Isolation**: Heads such as `LiquidSpikingTensorExpertHead`, `OmniversalQuantumExpertHead`, and `FractalGenesisExpertHead` require explicit `allow_experimental=True` and force `production_ready_claim=False` on receipts.
3. **Promotion Safety**: `SCFConfig(allow_automatic_promotion=True)` raises `ValueError`, and `SCFLearningGate.automatic_promotion_enabled` is permanently `False`.

---

## 4. Migration & Rollback Instructions

### Enabling SCF Incrementally
1. Verify Phase Zero baseline integrity by confirming all pinned hashes in `SCF91_REGRESSION_BASELINE.json`.
2. Instantiate `CognitiveFabricRuntime(SCFConfig(flags=SCFFeatureFlags(SCF_ENABLED=True, SCF_ROUTER_ENABLED=True, SCF_VERIFICATION_ENABLED=True)))` to enable routing + deterministic verification while keeping experimental layers disabled.
3. Turn on individual flags (`SCF_COLLECTIVE_REASONING_ENABLED`, `SCF_SPECULATIVE_DECODING_ENABLED`, etc.) only after running `pytest -q test_scf_v91_cognitive_fabric.py`.

### Zero-Risk Rollback
1. **Runtime Rollback (Instant, zero code changes)**: Set `SCF_ENABLED=false` (or unset all `SCF_*` environment variables). `CognitiveFabricRuntime` immediately falls back to legacy execution paths.
2. **Git Branch Rollback**: No existing files on `main` or prior branches were overwritten. Switching back cleanly restores pre-SCF state:
   ```bash
   git checkout v85-v86-measurable-architecture
   ```
   Existing checkpoints under `output/`, `artifacts/`, and `datasets/` remain byte-identical to their `SCF91_REGRESSION_BASELINE.json` SHA-256 pins.
