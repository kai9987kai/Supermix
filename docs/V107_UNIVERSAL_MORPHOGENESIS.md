# V107 — Universal Morphogenesis Architecture

> **Supermix v107** · November 2026
> *Additive subsystems only — zero regression, zero removal, birth-identity safe.*

---

## Table of Contents

1. [Overview](#overview)
2. [Universal Fast-Weight Memory (uTTT-MoE / Dense)](#1-universal-fast-weight-memory)
3. [Sonapse Synthetic Neuron Foundry](#2-sonapse-synthetic-neuron-foundry)
4. [Connectome Astrocyte-Gated Multi-Timescale Plasticity](#3-connectome-astrocyte-gated-plasticity)
5. [Retrograde Ephaptic LFP Coupling](#4-retrograde-ephaptic-lfp-coupling)
6. [MARS Variance-Reduced Muon Optimizer](#5-mars-variance-reduced-muon-optimizer)
7. [Outlier-Mixed Int4 KV-Cache Compression](#6-outlier-mixed-int4-kv-cache-compression)
8. [Ten New Exact Solvers](#7-ten-new-exact-solvers)
9. [Five v107 Analytical Engines](#8-five-v107-analytical-engines)
10. [ChampionNet universal_fast_weight_expert Variant](#9-championnet-variant)
11. [Wiring & Integration Map](#10-wiring--integration-map)
12. [Test Coverage](#11-test-coverage)
13. [Research References](#12-research-references)

---

## Overview

v107 adds **seven new additive subsystems** to MiMoMix without touching any
existing logic paths. Every new module is **birth-identity safe**: all gating
parameters are zero-initialised so a checkpoint trained without these modules
produces **bit-exact identical logits** when loaded into a v107 model.

```mermaid
flowchart TD
    subgraph v107["v107 Additive Subsystems"]
        UFWM["Universal Fast-Weight Memory\n(uTTT-MoE / Dense)"]
        SNF["Sonapse Neuron Foundry\n6 Synthetic Families"]
        ASTRO["Connectome Astrocyte Gate\n+ Three-Factor BTSP"]
        RETRO["Retrograde Ephaptic\nLFP Coupling"]
        MARS["MARS Variance-Reduced\nMuon Optimizer"]
        KVC["Outlier-Mixed Int4\nKV-Cache Compression"]
        SOLVERS["10 New Exact Solvers"]
        ENGINES["5 v107 Analytical Engines"]
    end
    UFWM --> |"cross-depth + cross-time"|MiMoMix
    SNF --> |"after_layer insertion"|MiMoMix
    ASTRO --> |"ConnectomeCore methods"|MiMoMix
    RETRO --> |"ConnectomeCore methods"|MiMoMix
    MARS --> |"muon_mars_gamma"|MuonHybridOptimizer
    KVC --> |"quantize/dequantize"|Decoding
    SOLVERS --> |"solve_problem()"|NexusSolver
    ENGINES --> |"v107_universal_morphogenesis"|NexusAPI
    MiMoMix["MiMoMixModel\n(mimomix_core.py)"]
    MuonHybridOptimizer["MuonHybridOptimizer\n(muon_optimizer.py)"]
    Decoding["MiMoMix Decoding\n(mimomix_decoding.py)"]
    NexusSolver["NexusSolver\n(nexus_solver.py)"]
    NexusAPI["NexusAPI\n(nexus_api.py)"]
```

---

## 1. Universal Fast-Weight Memory

**File:** `source/mimomix_core.py` · `class UniversalFastWeightMemory`

### Motivation

Grounded in **Universal Test-Time Training** (Cai et al., Oct 2026,
`arXiv:2610.05484`) and the **Fast Weight Programmers / DeltaNet** lineage.
Standard Transformers have no persistent recurrent state between layers or
between tokens at different time steps. UFWM adds a *single globally shared
fast-weight programmer* that recurs across **two orthogonal axes**:

### Design

| Axis | Description | State Shape |
|------|-------------|-------------|
| **Cross-depth** | Layer-role–conditioned bank pool updated at each layer | `(B, T, E, d, d)` |
| **Cross-time** | Persistent causal bank pool consolidating depth writes | `(B, E, d, d)` |

**Key components:**

- **Bank Router** (`W_router → R^{H → E}`): top-K sparse MoE gating (or dense
  softmax) selects which memory banks receive writes and reads.
- **Decoupled Erase / Write** (`W_erase`, `W_write`): sigmoid-gated delta
  updates with surprise-modulated write gain `tanh(1 + σ_surprise)`.
- **Layer-Role Embeddings** (`e_l ∈ R^{n_layers × H}`): condition the memory
  update differently at each transformer layer.
- **Zero-Init Gates** (`layer_gates[l]`, `temporal_gate`): guarantee birth
  identity; output is `h + gate * δh`.
- **KV-Cache Integration**: temporal state is serialised as a rank-3
  `(B, T', E*d*d)` tail entry in `past_key_values` for bit-exact incremental
  decode and MTP speculative rollback parity.

### Operating Modes

| Mode | Config | Behavior |
|------|--------|----------|
| `moe` (default) | `ufwm_mode="moe"` | Top-K sparse bank routing |
| `dense` | `ufwm_mode="dense"` | Full softmax over all banks |

### CLI Flags

```
--use_ufwm              Enable Universal Fast-Weight Memory
--ufwm_dim   24         State dimension d per bank
--ufwm_banks 4          Number of memory banks E
--ufwm_top_k 2          Top-K for MoE routing
--ufwm_mode  moe        "moe" or "dense"
```

---

## 2. Sonapse Synthetic Neuron Foundry

**File:** `source/sonapse_neuron_foundry.py`

Inspired by `kai9987kai/SONAPSE-MORPHOGENESIS`, `ODDITY-LAB`,
`Two-Soma-Neuron-Lab`, `NeuroPulse-CableLab`, and `Supermix-Aether`.

### Six Synthetic Neuron Families

| # | Family | Key Bio-Mechanism | State Width |
|---|--------|--------------------|-------------|
| 1 | **SolitonAxonNeuron** | Nonlinear dispersion-compensated soliton pulses + refractory trace | `2D` |
| 2 | **SpineApparatusNeuron** | MICrONS EM dual-regime spine switch (nascent Oja-Hebbian ↔ mature error credit) | `3D` |
| 3 | **AstrocyticTripartiteNeuron** | Slow Ca²⁺ wave + gliotransmitter (D-serine/ATP) metaplasticity gate | `2D` |
| 4 | **EphapticFieldNeuron** | Zero-lag extracellular LFP ephaptic coupling for phase synchronization | `2D` |
| 5 | **RetrogradeEndocannabinoidNeuron** | Post-to-pre DSI/DSE retrograde NO/eCB signaling | `2D` |
| 6 | **FastWeightEngramNeuron** | Rank-1 fast-weight associative trace + slow Palimpsest consolidation | `D + D²` |

### Integration Layer

`SonapseNeuronFoundryLayer` wraps `HeterogeneousNeuronPopulation` (runs all 6
families in parallel with learned routing weights + metabolic energy accounting)
and plugs into `MiMoMixModel` via `use_neuron_foundry=True`, inserted after a
configurable layer index (`neuron_foundry_after_layer`).

**Properties:**
- Zero-init birth gate (`gate == 0`)
- Strict causal state recurrence
- Rank-3 `(B, T', state_width)` KV-cache state history
- Bit-exact MTP self-speculative rollback parity

### CLI Flags

```
--use_neuron_foundry          Enable Sonapse Neuron Foundry
--neuron_foundry_dim   48     Per-family hidden dimension
--neuron_foundry_after_layer  Layer index to insert after
```

---

## 3. Connectome Astrocyte-Gated Plasticity

**File:** `source/mimomix_core.py` · `ConnectomeCore` methods

### Three-Factor BTSP Update

`three_factor_btsp_update(child, eligibility, dopamine_signal, instructive)`:
Behavioral Timescale Synaptic Plasticity combining:
- **Eligibility trace** (sustained synaptic tag)
- **Dopamine neuromodulatory signal** (global reward/surprise)
- **Instructive signal** (error-driven credit assignment)

### Astrocyte Gate

When `cns_astrocyte_gate=True`, the connectome adds a differentiable
`astro_strength` parameter per child that modulates synaptic weights through a
slow astrocytic calcium metaplasticity mechanism.

### EWC Synaptic Consolidation

`ewc_synaptic_consolidate(child, fisher_diag, old_params, ewc_lambda)`:
Elastic Weight Consolidation that penalizes drift from consolidated parameters
weighted by the Fisher information diagonal, preventing catastrophic forgetting.

### CLI Flags

```
--cns_astrocyte_gate          Enable astrocyte gating
--cns_retrograde_ephaptic     Enable retrograde ephaptic coupling
```

---

## 4. Retrograde Ephaptic LFP Coupling

**File:** `source/mimomix_core.py` · `ConnectomeCore`

When `cns_retrograde_ephaptic=True`, adds:
- **`retro_strength`**: per-child post-to-pre retrograde coupling weight
- **`ephaptic_strength`**: per-child local field potential influence factor

These enable:
- **Retrograde signaling**: postsynaptic activity modulates presynaptic release
  probability (inspired by endocannabinoid DSI/DSE)
- **Ephaptic coupling**: zero-lag extracellular field-mediated influence between
  neighboring neurons for rapid phase synchronization

---

## 5. MARS Variance-Reduced Muon Optimizer

**File:** `source/muon_optimizer.py`

### MARS γ Preconditioning

When `mars_gamma > 0` (default 0.0, inert), Muon's gradient is preconditioned
with a **variance-reduced** MARS estimate:

```
g_mars = g + γ * (g - g_prev)
```

This reduces gradient variance during training, improving convergence stability
especially in the later stages of training or under distribution shift.

### Telemetry

`optimizer_telemetry()` now reports:
- `mars_gamma` setting
- `mars_active` boolean
- Current `grad_norm` statistics

### CLI Flag

```
--muon_mars_gamma 0.0    MARS variance-reduction coefficient (0 = disabled)
```

---

## 6. Outlier-Mixed Int4 KV-Cache Compression

**File:** `source/mimomix_decoding.py`

### Design

For long-context inference, KV-cache memory becomes the bottleneck. v107 adds
**outlier-aware mixed-precision Int4 quantization** that:

1. **Detects outlier channels** per key/value tensor (channels with values
   exceeding a configurable z-score threshold)
2. **Preserves outliers in FP16** while quantizing remaining channels to Int4
3. **Dequantizes** transparently during attention computation

### Functions

| Function | Purpose |
|----------|---------|
| `quantize_past_kv_mixed_outlier(kv, z_thresh)` | Compress KV-cache to mixed Int4+FP16 |
| `dequantize_past_kv_mixed_outlier(compressed)` | Restore to full precision for attention |
| `kv_cache_outlier_memory_report(compressed)` | Report compression statistics |

### Memory Savings & Kernel Optimizations

Typical savings: **79–80% reduction** (`~5x` compression ratio, `>0.997` cosine
similarity) in KV-cache memory while outperforming uniform Int4 accuracy on heavy
outlier activations. Combined with **precomputed RoPE position caches** and
**zero-copy GQA head broadcasting** in `mimomix_core.py`, attention decode
eliminates redundant trigonometric recomputation and K/V tensor `.repeat_interleave`
allocations.

---

## 7. Ten New Exact Solvers

**File:** `source/nexus_solver.py`

| # | Solver | Formula ID | Domain |
|---|--------|------------|--------|
| 1 | **Nernst Potential** | `nernst_potential` | Neuroscience / Electrochemistry |
| 2 | **Goldman-Hodgkin-Katz** | `ghk_potential` | Membrane Biophysics |
| 3 | **Cable Constants** | `cable_constants` | Cable Theory / Dendrites |
| 4 | **Michaelis-Menten** | `michaelis_menten` | Enzyme Kinetics |
| 5 | **Henderson-Hasselbalch** | `henderson_hasselbalch` | Acid-Base Chemistry |
| 6 | **Shannon-Hartley** | `shannon_hartley` | Information Theory |
| 7 | **M/M/1 Queue** | `mm1_queue` | Queueing Theory |
| 8 | **Markov Stationary** | `markov_stationary` | Stochastic Processes |
| 9 | **Quadratic Form** | `quadratic_form` | Linear Algebra |
| 10 | **Put-Call Parity** | `put_call_parity` | Financial Mathematics |

All solvers:
- Return `SolverResult` with `solved`, `formula_id`, `display_answer`, and
  SHA-256 `receipt`
- Are fail-closed: invalid domains → `solved=False`
- Are deterministic with zero network I/O

---

## 8. Five v107 Analytical Engines

**File:** `source/nexus_v107_frontier.py`

| # | Engine | Purpose |
|---|--------|---------|
| 1 | **UniversalFastWeightProfiler** | Capacity, SIR, parameter-sharing, and KV-cache footprint analysis |
| 2 | **ConnectomePlasticityOrchestrator** | BTSP + spine-apparatus + astrocyte Ca²⁺ metaplasticity simulation |
| 3 | **SyntheticNeuronFoundryAssay** | Dynamical responsiveness, memory-horizon, phase-coherence, ATP budget |
| 4 | **AntipodeFailureDistiller** | Hard-negative clustering + anti-collapse Boltzmann curriculum weights |
| 5 | **SymbiosisContinualRetentionAuditor** | Fisher-weighted EWC drift, proximal consolidation, forgetting risk |

**Contract:** Every engine sets `"answer_authority": False`, validates all
inputs via `ValueError`, and produces deterministic SHA-256 audit receipts.

---

## 9. ChampionNet Variant

**File:** `source/model_variants.py`

`universal_fast_weight_expert` — a ChampionNet configuration with a
`UFWMExpertHead` as the final layer (index 10). The expert head projects through
a lightweight UFWM bank read to specialize generation on fast-weight memory
contents.

**Registration:**
- `model_variants.py`: variant config + `UFWMExpertHead` class
- `scf/models/__init__.py`: `ChampionNetAdapter` updated for UFWM loading

---

## 10. Wiring & Integration Map

```mermaid
flowchart LR
    subgraph CLI["CLI & Training"]
        TRAIN_TALK["train_mimomix_talk.py\n--use_ufwm, --use_neuron_foundry\n--cns_astrocyte_gate\n--muon_mars_gamma"]
        TRAIN_GEN["train_mimomix_generalisation.py\nnew_parameter_prefixes\nmissing key whitelist"]
    end
    subgraph API["API & Studio"]
        NEXUS_API["nexus_api.py\n5 new endpoints\nv107_universal_morphogenesis"]
        SCF["scf/__main__.py\n14 diagnose probes"]
        STUDIO["nexus_studio.html\nSection 10: 5 buttons"]
        CHAT["supermix_chat_server.py\ntelemetry badges"]
        MANIFEST["generate_studio_runtime_manifest.py\nsonapse_neuron_foundry.py"]
    end
    subgraph CI["CI / Quality Gates"]
        WORKFLOW[".github/workflows/\nruntime-quality-gates.yml"]
        TEST85["test_v85_release_contract.py\nCI_COVERAGE_FLOOR = 97"]
        TEST107["test_v107_universal_fast_weight_and_neurons.py\n13 test cases"]
    end
    TRAIN_TALK --> TRAIN_GEN
    NEXUS_API --> SCF
    SCF --> STUDIO
    WORKFLOW --> TEST85
    WORKFLOW --> TEST107
```

**All touchpoints:**

| File | Change |
|------|--------|
| `source/train_mimomix_talk.py` | 11 new argparse flags, `_CONFIG_FROM_ARGS` entries |
| `source/train_mimomix_generalisation.py` | `new_parameter_prefixes`, missing-key whitelist, MARS wiring |
| `source/nexus_api.py` | 5 Pydantic models, 5 FastAPI endpoints, `v107_universal_morphogenesis` signals |
| `source/scf/__main__.py` | 14 diagnose probes, frontier/perf command wiring |
| `source/supermix_chat_server.py` | Architecture dict, telemetry badges, JS model tag |
| `source/generate_studio_runtime_manifest.py` | `sonapse_neuron_foundry.py` in RUNTIME_MODULES |
| `web_static/nexus_studio.html` | Section 10 panel with 5 v107 buttons |
| `.github/workflows/runtime-quality-gates.yml` | v107 compilation + test suite step |
| `test_v85_release_contract.py` | Expected defaults, CI_COVERAGE_FLOOR 96→97 |
| `source/scf/models/__init__.py` | ChampionNetAdapter UFWM variant support |

---

## 11. Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| `test_v107_universal_fast_weight_and_neurons.py` | 13 | ✅ |
| `test_v85_release_contract.py` + `test_studio_runtime_manifest.py` + variant parity | 67 | ✅ |
| Core & Nexus regression suites (`test_nexus_*`, `test_mimomix_*`, etc.) | 500 | ✅ |
| SCF + v104–v106 frontier suites | 100+ | ✅ |

### v107 Test Functions

1. `test_ufwm_birth_identity_causality_grads_and_kv_speculative_parity` (`moe`, `dense`)
2. `test_ufwm_and_foundry_config_validation_fails_closed`
3. `test_all_six_synthetic_neuron_families_and_foundry_layer_parity`
4. `test_champion_net_universal_fast_weight_expert_and_scf_isolation`
5. `test_connectome_astrocyte_retrograde_btsp_and_ewc_consolidation`
6. `test_full_v93_to_v107_megastack_kv_and_speculative_parity`
7. `test_rotary_position_cache_and_zero_copy_gqa_exactness` (`sink_on=True/False`)
8. `test_mars_muon_optimizer_and_outlier_mixed_int4_kv_superior_fidelity`
9. `test_ten_new_v107_exact_solvers_and_scf_verification_gate`
10. `test_five_v107_engines_receipts_and_fail_closed_validation`
11. `test_v107_api_endpoints_cli_trainers_chat_and_studio`

---

## 12. Research References

| Citation | Used In |
|----------|---------|
| Cai et al. (2026), *Universal Test-Time Training*, arXiv:2610.05484 | UFWM |
| Schlag et al. (2021), *Linear Transformers Are Secretly Fast Weight Programmers* | UFWM DeltaNet |
| MICrONS Consortium (2024), *Functional connectomics spanning multiple areas of mouse visual cortex* | SpineApparatusNeuron |
| Araque et al. (2014), *Gliotransmitters travel in time and space* | AstrocyticTripartiteNeuron |
| Anastassiou et al. (2011), *Ephaptic coupling of cortical neurons* | EphapticFieldNeuron |
| Wilson & Nicoll (2001), *Endogenous cannabinoids mediate retrograde signalling* | RetrogradeEndocannabinoidNeuron |
| Kirkpatrick et al. (2017), *Overcoming catastrophic forgetting in neural networks* (EWC) | ConnectomeCore consolidation |
| Bittner et al. (2017), *Behavioral time scale synaptic plasticity* | BTSP three-factor |
| Yuan et al. (2025), *MARS: Unleashing the Power of Variance Reduction for Training Large Models* | Muon MARS γ |
| Hooper et al. (2024), *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization* | Outlier Int4 KV-cache |

---

*Document generated for Supermix v107 — Universal Morphogenesis release.*
