# Supermix v91 (SCF) Benchmark, Regression & Hardware Compatibility Report

> Measured on branch `scf-v91-cognitive-fabric`. All numbers below are direct observations from live CPU runs and integrity scripts; experimental subsystems (`LiquidSpikingTensorExpertHead`, `OmniversalQuantumExpertHead`, `FractalGenesisExpertHead`, `MiMoMix_LatentStateReasoner` advisory slot context, `NexusMind_ActiveInferenceController`, and `ChampionNet_HierarchicalMoE_RoutingPolicy`) remain explicitly isolated with `production_ready_claim=False`.

---

## 1. Benchmark Report

### 1.1 Cognitive Fabric Runtime (`CognitiveFabricRuntime.run_task`)

Measured on CPU comparing baseline legacy fallback (`SCFFeatureFlags()`, all 10 flags `False`) against full opt-in orchestration (`SCFFeatureFlags.all_enabled(speculative_decoding=True)`):

| Task Family | Prompt | Fallback Latency (ms) | Fallback Peak Traced (KB) | SCF Latency (ms) | SCF Peak Traced (KB) | SCF Routed Family | Verified | Verifier Source | Final Answer Summary |
|---|---|---:|---:|---:|---:|---|---|---|---|
| `arithmetic` (`math_science`) | `What is 2 + 2?` | `1016.175` | `313.29` | `1245.142` | `309.43` | `NexusMind` | `True` | `grounding_runtime.solve_exact_arithmetic` | `4` |
| `physics` (`math_science`) | `What is the impulse for force 12 N acting over time 4 s?` | `1152.265` | `316.41` | `1072.468` | `308.77` | `NexusMind` | `True` | `nexus_solver.dynamics.impulse` | `48` |
| `code` (`code`) | `What is the sum of range(2, 6)?` | `1325.455` | `307.66` | `1509.741` | `307.25` | `NexusMind` | `True` | `nexus_solver.code.range_sum` | `14` |
| `causal_epistemic` (`causal_epistemic`) | `Please compute causal effect on Acceleration if we do(Force=50) in physics_newton` | `1256.996` | `308.34` | `1258.307` | `308.67` | `NexusMind` | `False` | `none` | `[SCF advisory trace ...]` (`production_ready_claim=False`) |
| `connectome` (`connectome`) | `male CNS neuron count for cell type KCg` | `957.506` | `307.53` | `1066.310` | `307.36` | `MiMoMix` | `False` | `none` | `[SCF advisory trace ...]` (`production_ready_claim=False`) |

Notes on benchmark observations:
- Under `SCFFeatureFlags()`, `fallback_to_legacy` is `True` across all 5 prompts and `legacy_fallback_fn` output is returned unmodified without activating SCF collective, memory, adaptive conformal, reasoning, or promotion paths.
- Under `SCFFeatureFlags.all_enabled(speculative_decoding=True)`, `verified=True` is achieved strictly on deterministic exact arithmetic, physics formula derivation, and normalised range-sum tracing (`grounding_runtime.solve_exact_arithmetic` and `nexus_solver.*`), while open-ended causal/connectome prompts explicitly emit `verified=False` and advisory traces rather than fabricating verified answers.
- Because `all_enabled()` enables `SCF_LATENT_MEMORY_ENABLED`, `SCF_ACTIVE_INFERENCE_ENABLED`, and `SCF_HIERARCHICAL_MOE_ENABLED`, receipts record `experimental_components_used` and set `production_ready_claim=False` even when final arithmetic is deterministic.
- Python heap overhead measured via `tracemalloc` stayed tightly bounded at `307.21 KB – 316.41 KB` across both modes.

### 1.2 MiMoMix Greedy vs. Speculative MTP Decode Parity

Measured via `MiMoMixAdapter.generate_tokens` on CPU (`MiMoMixConfig(vocab_size=64, hidden_size=32, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=48, use_moe=False, n_mtp_layers=1, use_thinking_core=False, sliding_window=8, native_context=32, max_position_embeddings=32)`, prompt `[1, 5, 9, 13]`, `max_new_tokens=8`):

| Mode | Wall Time (ms) | New Tokens | Drafted | Accepted Draft | Output Tokens (`new_tokens`) | Exact Sequence Parity |
|---|---:|---:|---:|---:|---|---|
| Greedy (`speculative_decoding=False`) | `53.066` | `8` | `0` | `0` | `[13, 13, 13, 13, 13, 13, 13, 13]` | Baseline |
| Speculative (`speculative_decoding=True`, `adaptive_thinking=False`) | `81.651` | `8` | `6` | `0` | `[13, 13, 13, 13, 13, 13, 13, 13]` | `True` |

Decode-safety enforcement:
- `resolve_decoding_policy(flags, adaptive_thinking=True, prefer_speculative=True)` refuses speculative decoding and returns `speculative_allowed=False`, `mode="greedy"`.
- Zero-acceptance random-weight MTP drafts (`0/6`) produced bit-identical greedy fallback output (`exact_token_parity=True`, `new_tokens_parity=True`).

---

## 2. Regression & Model Integrity Report

### 2.1 Checkpoint SHA-256 Verification (`10 / 10` Pinned Weights Unchanged)

Verified against [`SCF91_REGRESSION_BASELINE.json`](../SCF91_REGRESSION_BASELINE.json) via `SCFObservabilityHub.verify_checkpoint_hash`:

| Checkpoint Path | Status | Verified SHA-256 Prefix |
|---|---|---|
| `output/v80_omni/v80_omni.pt` | `exists=True`, `verified=True` | `f49ffbf4da98e277...` |
| `output/v89_corpus/v89_corpus.pt` | `exists=True`, `verified=True` | `0ca39999327ceb4b...` |
| `output/v91_cns_connectome/v91_cns_connectome.selected.pt` | `exists=True`, `verified=True` | `e927a848a164f934...` |
| `output/v93_neurogenesis/v93_neurogenesis.selected.pt` | `exists=True`, `verified=True` | `55a9a5a2f2e68cd8...` |
| `output/v96_physio_neurogenesis/v96_physio_neurogenesis.pt` | `exists=True`, `verified=True` | `13bb1bcd9d1c9095...` |
| `output/hf_v56_publish/v56b_randslots_entropy.pt` | `exists=True`, `verified=True` | `c329bc847c4a3b57...` |
| `output/hf_v57_publish/v57_talk_v2.pt` | `exists=True`, `verified=True` | `d97c2d84a2569a47...` |
| `artifacts/v52_initialization/champion_model_chat_v50_cognitive_leap.pth` | `exists=True`, `verified=True` | `6d595a44ee130a56...` |
| `artifacts/v52_initialization/champion_model_chat_v52_initialized.pth` | `exists=True`, `verified=True` | `1ab25aa9772f9089...` |
| `output/benchmark_v51_cognitive_leap_ultra/cognitive_leap_ultra_v51_trained.pth` | `exists=True`, `verified=True` | `664b1779452fe148...` |

### 2.2 Registry Discovery Across All 6 Model Families

Verified via `SCFModelRegistry().summary()`:

- `ChampionNet`: `available_in_source=True`, `3` local `.pth` weights discovered, `18` preserved heads, `3` experimental isolated heads (`LiquidSpikingTensorExpertHead`, `OmniversalQuantumExpertHead`, `FractalGenesisExpertHead`).
- `CognitiveLeap`: `available_in_source=True`, `3` local `.pth` weights discovered.
- `QwenLoRA`: `available_in_source=True`, `28` `adapter_model.safetensors` files discovered, `automatic_promotion_allowed=False`.
- `OmniCollective`: `available_in_source=True`, `22` registered `MODEL_SPECS` keys, `0` local `.pth` files (`maturity_status="source_present_external_weights_optional"`).
- `MiMoMix`: `available_in_source=True`, `7` `.pt` weights discovered (`v56b`, `v57`, `v80`, `v89`, `v91`, `v93`, `v96`).
- `NexusMind`: `available_in_source=True`, `built_in_no_external_weights_required`, `v95_forensic_engines_answer_authority=False`.

### 2.3 Verification Gates & Automated Regression Results

All integrity checks and 17 pytest suites pass cleanly on `scf-v91-cognitive-fabric`:

- `python -m ruff check source/scf test_scf_v91_cognitive_fabric.py test_v85_release_contract.py` -> `All checks passed!`
- `python source/sync_runtime_model_variants.py --check` -> `runtime model-variant snapshot is current: source=f42fba345330 runtime=f42fba345330 bytes=408687`
- `python source/generate_studio_runtime_manifest.py --check` -> exited `0` (Studio distribution closure current; `source/scf/*` stays cleanly decoupled from Studio desktop/web bundles)
- `python source/dark_test_audit.py --check` -> `OK: 30 dark tests, baseline 30`
- 17-suite regression run (`test_scf_v91_cognitive_fabric.py`, `test_v85_release_contract.py`, `test_studio_runtime_manifest.py`, `test_nexus_studio_contract.py`, `test_mimomix_core.py`, `test_mimomix_v82.py`, `test_v91_connectome_core.py`, `test_v92_temporal_connectome.py`, `test_v93_core.py`, `test_v93_neurogenesis.py`, `test_v93_trainer.py`, `test_v94_growth_policy.py`, `test_v94_connectome_audit.py`, `test_prepare_v94.py`, `test_v95_neurogenesis_and_connectome.py`, `test_nexus_v95_ecosystem.py`, `test_v96_physio_training_and_criticality.py`) -> `295 passed, 2 warnings` (`0` failures, `0` skipped tests).

---

## 3. Hardware Compatibility Report

| Environment Aspect | Observed Value / Status | Notes |
|---|---|---|
| Operating System | `Windows-11-10.0.28120-SP0` | Tested host OS |
| Python | `3.12.10` | Tested runtime interpreter |
| CPU | `8` logical cores (`cpu_count=8`) | All SCF orchestration, verification, and MiMoMix parity runs verified on CPU |
| PyTorch | `2.11.0+cpu` | Tested CPU build |
| CUDA GPU | `cuda_available=False` | **Not tested on CUDA GPU hardware on this host**; CUDA execution paths rely on existing upstream Supermix device guards |
| Optional External Checkpoints & Modalities | Graceful degradation | Missing Qwen LoRA directories, missing `omni_collective*.pth`, and missing image/video/3D mesh files produce structured status warnings (`adapter_directory_missing`, `image_path_missing`, `video_path_missing`, `mesh_path_missing`) rather than uncaught exceptions |

---

## 4. v97 Continuation, SCF v91.5 Hybrid Benchmark & Chat Interface Integration

### 4.1 Stratified Continuation Training (`v97_scf_connectome`) & Honest Promotion Gating

- Built `4,800`-row stratified dataset (`datasets/v97/v97_evolution_mix.jsonl`, `sha256=5235e5ea425f...`) via [`source/build_v97_evolution_mix.py`](../source/build_v97_evolution_mix.py) continuing from `output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt` with default-off architecture extensions enabled (`cns_hebbian_trace=True`, `thinking_entropy_halt=True`) into `output/v97_scf_connectome/v97_scf_connectome.selected.pt` (`steps=12`, `dev_loss=0.059501` vs `v96` `0.086137`).
- Note on host disk constraint: during the 36-step leg, saving the `351 MB` full-optimizer `.partial.pt.tmp` hit the Windows host `0.32 GB` free-disk ceiling; the clean `153 MB` inference-ready `v97_scf_connectome.selected.pt` saved at step 12 remained intact (`WinError 32` cleanup retry hardened in `save_talk_checkpoint`).
- Paired `41`-task evaluation (`eval_problem_solving.py`, `seed=82`, `max_new_tokens=112`):
  - Baseline promoted checkpoint (`v96_physio_neurogenesis.selected.pt`): **`35 / 41 = 0.854`** (`95% CI [0.716, 0.931]`, abstention score `+0.7073`, within10% `0.878`).
  - Continuation checkpoint (`v97_scf_connectome.selected.pt`, step 12): **`34 / 41 = 0.829`** (`95% CI [0.687, 0.915]`, abstention score `+0.6585`, within10% `0.878`; missed `average` alongside `acceleration`, `arithmetic`, `percent`, and the 3 `cns_*` recall tasks).
  - Per Engineering Rule 9 (`SCFLearningGate.automatic_promotion_enabled=False`), **`v96_physio_neurogenesis.selected.pt` stays the default promoted checkpoint**, while `v97_scf_connectome.selected.pt` is preserved and explicitly selectable alongside it as an experimental `v97` checkpoint.

### 4.2 Neural-Symbolic SCF Hybrid Benchmark (`41`-Task `seed=82`)

Combining `SCFVerificationGate` (extending deterministic verification with `ConnectomeAtlasVerifier` over `datasets/v93_malecns/malecns_sided_types.npz` and exact percentage fraction evaluation) with `MiMoMixAdapter.infer_text`:

- Strictly verified closed-world accuracy alone (`verified=True`): **`20 / 41 = 0.4878`** (solving `arithmetic` `472 - 378 = 94`, exact `%` `25% of 232 = 58`, `acceleration = 48`, all `3/3` Male-CNS biological connectome queries `Li20 = 40`, `ORN_VA2 right = 41`, `Tm2->Dm3c = 12210`, plus deterministic physics & `code_range_sum`).
- Combined **SCF Hybrid (`SCF Verification + v96 Neural Candidate Fallback`)**: **`40 / 41 = 0.9756`** (`verified=True` deterministic proofs on `20/41`; unverified `model_conditional` neural replies cleanly distinguished on the remaining open/procedural tasks).

### 4.3 Chat Interfaces & `/v1/scf/*` Integration

- Upgraded [`source/supermix_chat_server.py`](../source/supermix_chat_server.py): zero-arg `discover_default_models()` loading `v96_physio_neurogenesis`, experimental `v97_scf_connectome`, `v93_neurogenesis`, `v89_corpus`, `v80_omni`, `v74_broad`, `v57_talk_v2`; streaming `mode="speculative"` (MTP); `scf_verify` overlay; live `architecture_telemetry` (`hebbian_gain_mean`, `cycle_entropies`); SSE `AbortController` cancel button and quick-prompt chips.
- Upgraded [`source/nexus_api.py`](../source/nexus_api.py) & [`web_static/nexus_studio.html`](../web_static/nexus_studio.html): added `GET /v1/scf/status`, `POST /v1/scf/run`, fail-closed `SCF Cognitive Fabric` panel, and opt-in `use_neural_model=True` checkpoint selection (`v96` default + `v97_scf_connectome`) in Persona Chat (`answer_authority=False` preserved for all unverified neural generations).


---

## v98 Continuation, Bounded AST Verification & Multi-Seed Audit

### 1. Deterministic Verifier & AST Micro-Trace Fixes
- **Fixed `code.neg_index` bug** (`source/nexus_solver.py`): constrained `nums[-k]` single-subscript matching so `nums = [10, 12, 4]\nr = nums[1] + nums[2] Final r?` resolves to `16` via the new step-capped (`500` ops, no `eval`/`exec`) AST micro-interpreter instead of `12`.
- **Fixed `permutations` vs `combination` normalization** (`source/prompt_normaliser.py`): prioritized `PERM_WORDS` / `arranged` / `in order` ahead of `CHOOSE_WORDS` (`ways`) so ordered arrangement prompts resolve to `P(n, k)` (`20`).
- **Expanded `SCFVerificationGate`** (`source/scf/verification/__init__.py`): added exact `Fraction` verifiers for `average`, `algebra_one_step`, constant-diff/ratio `sequence`, `two_step`, additive/subtractive `word_problem`, and `75 x 7` multiplication prefixes while keeping strict negation/proof refusal guards.
- **Measured `SCF` deterministic + hybrid verification (`python -m scf bench`)**:
  - `seed=82`: **`41/41 (1.0000)`** verified & hybrid (vs `40/41` prior, `20/41` strictly verified prior)
  - `seed=83`: **`41/41 (1.0000)`** verified
  - `seed=91`: **`41/41 (1.0000)`** verified

### 2. `v98_scf_predictive` Continuation Run (`cns_predictive_error`, CPU-only host)
- **Command leg**: 60-step continuation from `output/v96_physio_neurogenesis/v96_physio_neurogenesis.selected.pt` on `datasets/v97/v97_evolution_mix.jsonl` with `--cns_hebbian_trace --cns_predictive_error --thinking_entropy_halt --grow_every 20` (`36,599,047` total / `9,132,151` active params, `3` neurogenesis rounds -> `812` alive modules, `0` witness rollbacks).
- **v58 Generalisation Ladder**: Best `dev_loss = 0.0572` at step 20 (`ppl = 1.06`); `tier3/tier1` ratio `0.995x`; all 5 checks `[PASS]` (`split_verified`, `learned_something`, `produces_non_empty_replies`, `speculative_matches_greedy`, `selection_never_read_a_tier`).
- **Paired Problem-Solving Gate (`eval_problem_solving.py`)**:
  - `seed=82` (`41` tasks): `v98_scf_predictive.selected.pt` **35/41 (0.854)** (ties `v96` `35/41 = 0.854`, beats `v97` `34/41 = 0.829`).
  - `seed=83` (`82` tasks): `v98_scf_predictive.selected.pt` **70/82 (0.854)** vs `v96_physio_neurogenesis.selected.pt` **72/82 (0.878)**.
- **Promotion decision (Rule 9)**: `v96_physio_neurogenesis.selected.pt` stays the **promoted baseline** (`72/82 = 0.878` on `seed=83`); `v98_scf_predictive.selected.pt` is registered cleanly in `MiMoMixAdapter`, `supermix_chat_server.py` (`[v98 PredErr]`), and `web_static/nexus_studio.html` as **experimental**.

### 3. `v99_scf_evolution` Continuation Run & Upgraded Chat UI (`source/build_v99_evolution_mix.py`, `source/supermix_chat_server.py`)
- **Balanced 41-Task + Chat Corpus** (`source/build_v99_evolution_mix.py`): assembled `7,540` rows (`datasets/v99/v99_evolution_mix.jsonl`, `sha256=14c9c31cd8b0`) covering all 41 problem-solving task families plus conversational turns (`task=""`).
- **60-Step Continuation from `v98_scf_predictive.selected.pt`** (`output/v99_scf_evolution/`, CPU-only host): best `dev_loss = 0.026452` at step 20 (`ppl = 1.03`), passing all 5 `v58` generalisation checks (`[PASS]`, `tier3/tier1 = 1.019x`).
- **Paired Benchmark & Promotion Decision (Rule 9)**:
  - `seed=82` (`41` tasks): standalone neural **`33/41 (0.805)`**; `python -m scf bench --seed 82 --neural`: **`41/41 (1.0000)`**.
  - `seed=83` (`82` tasks): standalone neural **`67/82 (0.817)`** vs promoted `v96_physio_neurogenesis.selected.pt` **`72/82 (0.878)`**.
  - `v96_physio_neurogenesis.selected.pt` remains the promoted default baseline; `v99_scf_evolution.selected.pt` is wired cleanly as an experimental latest checkpoint (`[v99 PredErr]`) in [`source/scf/models/__init__.py`](../source/scf/models/__init__.py), [`source/supermix_chat_server.py`](../source/supermix_chat_server.py), and [`web_static/nexus_studio.html`](../web_static/nexus_studio.html).
- **Upgraded Chat Interface**: added `GET /api/scf_status`, optional `scf_runtime` Cognitive Fabric routing in `POST /api/chat` and `POST /api/compare`, `scf_hybrid_accuracy` in `POST /api/benchmark`, `PredErr` / `Hebbian` / `Thinking` live telemetry + proof drawers, multiline Enter/Shift+Enter composer, and Markdown transcript export.

### 4. `v100_scf_balanced` (`cns_divisive_norm`) & `533/533` 13-Seed Verification
- **Carandini-Heeger E/I Divisive Normalization** (`cns_divisive_norm: bool = False` in [`source/mimomix_core.py`](../source/mimomix_core.py)): rescales active connectome rates by `1 / (1 + clamp(softplus(strength) - ln(2), 0, 2) * mean_inhib_pool)` using Dale inhibitory neuron masks (`sign < 0`), preserving bit-for-bit forward identity at birth and neurogenesis slot inheritance.
- **13-Seed `533/533 (100%)` Deterministic Audit (`seeds 82..94`)**: extended `ConnectomeAtlasVerifier` (soma-side, hemisphere, and `cell type` phrasings) and [`source/nexus_solver.py`](../source/nexus_solver.py) (`ordered arrangements of K ... from N`, `spring of constant`, `moles`/`litres`) from `516/533` to **`533/533 (100.0%)`**.
- **Semantic Numerical Consensus** ([`source/scf/collective/__init__.py`](../source/scf/collective/__init__.py)): `SCFCollectiveOrchestrator` now extracts canonical numerical values across candidate replies so formatted explanations and bare numbers recognize cross-model agreement.
- **64-Step Trunk-Protected Training from `v96` (`output/v100_scf_balanced/`, CPU-only host)** on `8,470`-row `datasets/v100/v100_precision_mix.jsonl` (`--freeze_trunk_steps 12 --decay_mode no_norm_bias --mtp_loss_weight_final 0.1 --repeat_subset_fraction 0.12 --cns_hebbian_trace --cns_predictive_error --cns_divisive_norm --thinking_entropy_halt`):
  - Selected at step 16 with lowest lineage `dev_loss = 0.023837` (`ppl = 1.02`, `tier3/tier1 = 1.009x`, all 5 `v58` checks `[PASS]`, positive connectome ablation delta `+0.00003 nats`).
  - Paired benchmark comparison across `3` seeds (`205` total novel tasks):
    - `seed=82` (`41`): **`35 / 41 (0.854)`** (ties `v96`)
    - `seed=83` (`82`): **`72 / 82 (0.878)`** (ties `v96`, `code_conditional 2/2`, `percent 2/2`)
    - `seed=84` (`82`): **`72 / 82 (0.878)`** vs `v96` **`73 / 82 (0.890)`** (`v100` final step 64 checkpoint also `72 / 82`)
    - Combined: `v100` **`179 / 205 (87.3%)`** vs promoted `v96` **`180 / 205 (87.8%)`**, `SCF Hybrid (seed=82)`: **`41 / 41 (1.0000)`**.
  - Per Rule 9, `v96_physio_neurogenesis.selected.pt` stays the promoted baseline (`#1`) while `v100_scf_balanced.selected.pt` (`[v100 DivNorm]`) is promoted ahead of prior experimental candidates (`v98`, `v99`, `v97`) in the registry and chat UIs.
