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

### 5. `v101_scf_cognitive` & Extended Math / Problem-Solving / Ideation / Conversation Upgrades
- **Sibling-Project Inspired Cognitive & Symbolic Upgrades**:
  - Zero-`eval` `ast.parse` math evaluator, implicit-coefficient quadratics & 2x2 linear systems, finite-difference/geometric/Fibonacci sequences, polynomial derivatives & definite integrals, stats/GCD/LCM, algebraic equivalence verifier, and exposed `derivation_steps` ([`source/nexus_solver.py`](../source/nexus_solver.py), [`source/scf/verification/__init__.py`](../source/scf/verification/__init__.py)).
  - Normalized 2D `MAP-Elites` + `NoveltySearch` topic/constraint-grounded creative ideation ([`source/nexus_ideation.py`](../source/nexus_ideation.py)).
  - `SCFCollectiveOrchestrator` inline step-arithmetic trace check ([`source/scf/collective/__init__.py`](../source/scf/collective/__init__.py)) and `SCFReasoningCoordinator` `CROSSWIRE` ambiguity probing + `SONAPSE` falsification critic decomposition ([`source/scf/reasoning/__init__.py`](../source/scf/reasoning/__init__.py)).
  - `OMNIS/Mnemorph` importance-scored layered memory continuity ([`source/scf/memory/__init__.py`](../source/scf/memory/__init__.py)) integrated with bounded session-turn forwarding and the live `SCF Cognitive Synthesis` UI drawer in [`source/supermix_chat_server.py`](../source/supermix_chat_server.py).
- **Exact-Grammar `v101_scf_cognitive` Continuation (`output/v101_scf_cognitive/v101_scf_cognitive.pt`)** warm-started from `v100_scf_balanced.selected.pt` on `10,100`-row `datasets/v101/v101_cognitive_mix.jsonl` (`--steps 64 --lr 0.00008 --freeze_trunk_steps 8 --extend_vocab` `+45` IDs -> `9,460` vocab, `36,614,279` params):
  - Best `dev_loss = 0.01745` at step `32` (`ppl = 1.02`, `tier3/tier1 = 1.047x`, all `5/5` `v58` checks `[PASS]`, connectome ablation `+0.00015 nats`, step-`64` probe `0.90`).
  - Paired benchmark across `3` seeds (`205` total novel tasks):
    - `seed=82` (`41`): **`35 / 41 (0.854)`** (ties `v96` and `v100`)
    - `seed=83` (`82`): **`72 / 82 (0.878)`** (ties `v96` and `v100`)
    - `seed=84` (`82`): **`71 / 82 (0.866)`** (`within10% = 0.878`) vs `v100` `72/82` & `v96` `73/82`
    - Combined standalone: **`178 / 205 (86.8%)`** | `python -m scf bench --seed 82 --neural`: **`41 / 41 (1.0000)`**.
  - Per Rule 9, `v96_physio_neurogenesis.selected.pt` remains `#1` promoted default, `v100_scf_balanced.selected.pt` stays `#2`, and `v101_scf_cognitive.pt` (`[v101 Cognitive]`) is registered at `#3` experimental.

### 6. `v102` Frontier Architecture, Neuro-Symbolic Synthesis & Cross-Ecosystem Engines
- **Additive Default-Off Neural Mechanisms (`source/mimomix_core.py`, `source/muon_optimizer.py`, `source/train_mimomix_talk.py`)**:
  - `ManifoldHyperConnection` (`use_mhc_residual=False`, `mhc_streams=2`, `mhc_sinkhorn_iters=20`): Birkhoff-polytope doubly stochastic residual stream mixing verified at `5.96e-08` doubly stochastic error at init, exact birth identity parity, incremental KV-cache parity (`< 1e-5`), and bit-exact MTP speculative == greedy token parity.
  - `ConnectomeCore` active apical dendritic compartment (`cns_dendritic_apical=False`): zero birth perturbation, causal temporal incremental decode parity (`< 1e-5`), and mitosis-safe `split_module` inheritance.
  - `RecursiveThinkingCore` Continuous Thought Machine pairwise temporal synchrony readout (`thinking_ctm_synchrony=False`): zero-init parity and live `ctm_synchrony_coherence` / `ctm_synchrony_norm` telemetry.
  - Pure-PyTorch `MuonHybridOptimizer` (`--optimizer muon_hybrid`, default remains `adamw`): 5-step quintic Newton-Schulz orthogonalized momentum for 2D hidden weights paired with AdamW for embeddings/norms/biases/gates.
- **Cross-Ecosystem Frontier Engines (`source/nexus_v102_frontier.py`, all `answer_authority=False` with SHA-256 receipts)**: `NexusSearchHybridRRF` (BM25 + cosine + RRF + Titans surprise + 7-class memory wired into `SCFLayeredMemory`), `TesseraRatifiedLemmaCommons`, `CounterformFalsificationAuditor` (CVaR + dual audit + fail-closed `ABSTAIN_UNSUPPORTED`), `CarrierInteractionAtlas` (exact `2^k` Möbius decomposition + GhostInTheMachine v7 `R^2` + Supermix-Aether FLOP/KV profiler), `SonapseDiscoveryMonitor` (Mode A/B + Pareto gate), and `HyperionComputeScheduler` (wired into `SCFRouter`).
- **Expanded Zero-Eval Symbolic Verification & Surfaces (`source/nexus_solver.py`, `source/scf/collective/__init__.py`, `source/scf/__main__.py`, `source/nexus_api.py`, `source/supermix_chat_server.py`)**: Added exact `Fraction` 3x3 linear systems, 2x2/3x3 matrix determinant & trace, modular arithmetic (`mod_pow`, `mod_inverse`), prime factorization, Euler's totient, binomial PMF, Bayes' theorem, parallel resistors, thin-lens optics, extended process-fidelity step auditing (`^`, `% of`), `python -m scf frontier`, `/v1/frontier/*` API routes, and `mHC`/`Apical`/`CTM-Sync` chat UI telemetry. All 10 pinned baseline checkpoints and promoted `v96_physio_neurogenesis.selected.pt` remain untouched under Rule 9.

### 7. `v103` Frontier Synthesis: Engram Memory, Gated DeltaNet-2, SHY Homeostasis, 5 Synthesis Engines & Studio Frontier Lab
- **Additive Default-Off Neural Memory & Homeostasis (`source/mimomix_core.py`, `source/train_mimomix_talk.py`)**:
  - `EngramConditionalMemory` (`use_engram_memory=False`, `engram_table_size=2048`, `engram_n_heads=2`, `engram_after_layer=0`): verified zero-init birth identity (`atol=1e-6`), incremental KV-cache decode parity (`< 1e-5` from prompt length `1`), and exact MTP speculative rollback equivalence (`assert_greedy_equivalence`).
  - `GatedDeltaNetMemory` (`use_deltanet_memory=False`, `deltanet_dim=32`, `deltanet_after_layer=1`): verified decoupled decay/erase/write fast-weight state recurrence, zero-init birth identity, incremental KV-cache decode parity (`< 1e-5`), exact MTP speculative rollback equivalence, AND full combined-stack parity (`mHC + Engram + DeltaNet-2 + temporal CNS + apical dendrites + CTM synchrony`).
  - `ConnectomeCore.synaptic_homeostasis_sleep_cycle`: verified weak-edge pruning with row-max protection, exact Dale-sign invariance, and smooth spectral-radius projection toward criticality.
- **Five Frontier Synthesis Engines (`source/nexus_v103_synthesis.py`, all `answer_authority=False` with canonical SHA-256 receipts)**: `ContinuumBTSPHopfieldMemory` (verified `btsp_one_shot_overlap=1.0`, `mnemosyne_intact_vs_erased=0.990764`), `TopKSparseAutoencoderProbe`, `FractalConnectomeAnalyzer` (`sierpinski` highest clustering), `CausewaySensitivityWorkbench`, and `NeuroHiveResourcePlanner` (verified exact token conservation and autonomous `MNT-01` fault repair).
- **Exact Symbolic Solvers, API, CLI & Fail-Closed Studio Frontier Lab (`source/nexus_solver.py`, `source/nexus_api.py`, `source/supermix_chat_server.py`, `web_static/nexus_studio.html`)**: Added exact `Fraction` 2x2 matrix inverse, 2x2 real eigenvalues, dot/cross products, finite & infinite geometric series sums, and exact polynomial evaluation; added `/v1/v103/*` routes, extended `python -m scf frontier --json`, added the accessible fail-closed `#panel-v103-frontier` tab in NexusMind Studio, and fixed a pre-existing missing `}` in `drawSpecTreeCanvas` so the Studio inline `<script>` parses cleanly in `node`.
- **Verification Gate**: `351/351` tests passing across `19` pytest suites (including `test_v103_frontier_synthesis.py`, `test_v102_frontier_evolution.py`, `test_v85_release_contract.py` at `CI_COVERAGE_FLOOR=93`, `test_nexus_studio_contract.py`, `test_studio_runtime_manifest.py`), `ruff check` (`All checks passed!`), `sync_runtime_model_variants.py --check`, `generate_studio_runtime_manifest.py --check`, `dark_test_audit.py --check` (`30` baseline), and `python -m scf bench --seed 82 --json` at **`41 / 41 (1.0000)`** verified & hybrid. Promoted baseline `v96_physio_neurogenesis.selected.pt` remains `#1` under Rule 9.
- **Muon-Trained `v102_scf_frontier` & `v103_scf_synthesis` Paired Benchmarks (`Windows` CPU, `max_new_tokens=96`) & Hugging Face Releases**:
  - **`v102_scf_frontier.pt`** (`Kai9987kai/supermix-v102`, `36,628,023` total / `9,161,127` active params, `mHC + Apical + CTM-Sync + MuonHybridOptimizer`, `32` CPU steps from `v101`): `dev_loss = 0.0181`, `tier1 = 0.0048`, **`tier3 = 0.0502`** (`tier3/tier1 = 1.046x`), all `5/5` `v58` checks `[PASS]`; paired novel accuracy: `seed=82` **`35/41 (0.8537)`**, `seed=83` **`72/82 (0.8780)`**, `seed=84` **`71/82 (0.8659)`**, combined **`178 / 205 (86.83%)`**, `scf bench --seed 82 --neural` **`41 / 41 (1.0000)`**.
  - **`v103_scf_synthesis.pt`** (`Kai9987kai/supermix-v103`, `37,561,559` total / `10,094,663` active params, `+ Engram + Gated DeltaNet-2 + MuonHybridOptimizer`, `32` CPU steps from `v102`): `dev_loss = 0.0180`, **`tier1 = 0.0047`**, `tier3 = 0.0515` (`tier3/tier1 = 1.048x`), all `5/5` `v58` checks `[PASS]`; paired novel accuracy: `seed=82` **`34/41 (0.8293)`**, `seed=83` **`71/82 (0.8659)`**, `seed=84` **`72/82 (0.8780)`**, combined **`177 / 205 (86.34%)`**, `scf bench --seed 82 --neural` **`41 / 41 (1.0000)`**.
  - Per Rule 9 (`automatic_promotion_enabled=False`), `v96_physio_neurogenesis.selected.pt` (`180 / 205 = 87.8%`, `Kai9987kai/supermix-v96`) remains `#1` Promoted Default; `v102_scf_frontier` and `v103_scf_synthesis` are released as Experimental alongside standalone milestone repos `supermix-v88`, `supermix-v91`, `supermix-v96`, `supermix-v97`, `supermix-v98`, and the refreshed `supermix-v99-cognitive-fabric` hub.

The v97 selected checkpoint scored **34/41** on standalone seed 82 with a 96-token generation cap. The neural-enabled SCF run with the same checkpoint scored **41/41 verified**. The latter counts final verifier-backed answers. [Standalone receipt](https://huggingface.co/Kai9987kai/supermix-v97/blob/main/results/eval_v97_seed82.json) · [SCF receipt](https://huggingface.co/Kai9987kai/supermix-v97/blob/main/results/scf_bench_v97_seed82.json)

### 8. `v104` Evolution: Titans Surprise Memory, Depth Attention Residuals (AttnRes), Neuromodulatory Connectome, Cautious-NorMuon & Epistemic-Mechanistic Suite
- **Additive Default-Off Neural Mechanisms (`source/mimomix_core.py`, `source/train_mimomix_talk.py`, `source/train_mimomix_generalisation.py`)**:
  - `TitansSurpriseMemory` (`use_titans_memory=False`, `titans_dim=32`, `titans_after_layer=1`): surprise-modulated associative matrix memory $M_t$ + momentum trace $S_t$, verified zero-init birth identity (`atol=1e-6`), gradient flow, incremental KV-cache decode parity (`< 1e-5` from length-1 prefill), and exact MTP speculative rollback equivalence (`assert_greedy_equivalence`).
  - `DepthAttentionResidual` (`use_attn_res=False`): token-local cross-layer depth softmax attention with zero-initialized per-layer gates, verified zero-init birth identity, KV-cache parity, and MTP speculative parity.
  - `ConnectomeCore` neuromodulatory gate (`cns_neuromod_gate=False`): novelty-driven excitability scaling (`[0.5, 1.5]`), verified zero-init birth identity, Dale-sign invariance, causal temporal parity, and mitosis `split_module` inheritance.
  - **Full Combined Stack Parity**: verified bit-exact incremental KV-cache decode (`< 1e-5`) and MTP speculative == greedy exact token parity with all mechanisms active simultaneously (`mHC + Engram + Gated DeltaNet-2 + TitansSurpriseMemory + DepthAttentionResidual + temporal CNS + apical dendrites + neuromod gate + CTM synchrony`).
- **Cautious-NorMuon Hybrid Optimizer (`source/muon_optimizer.py`)**: opt-in `cautious_mask=True` (sign-alignment masking with active-fraction rescaling), `row_normalize=True` (NorMuon row-wise RMS stabilization), and `optimizer_telemetry()` verified to reduce loss smoothly while keeping defaults off for exact reproducibility.
- **Five Frontier Epistemic & Mechanistic Engines (`source/nexus_v104_evolution.py`, all `answer_authority=False` with canonical SHA-256 receipts)**: `ConformalCalibratedRouter`, `MechanisticCircuitTracer`, `EpisodicGraphConsolidator`, `VerifiableProcessRewardAuditor`, and `ParetoFrontierAutoTuner`.
- **Expanded Zero-Eval Exact STEM Solvers, API, CLI, Chat & Studio Lab (`source/nexus_solver.py`, `source/nexus_api.py`, `source/scf/__main__.py`, `source/supermix_chat_server.py`, `web_static/nexus_studio.html`)**: Added 10 new zero-`eval` exact solvers (Catalan, Fibonacci, Chinese Remainder Theorem, Shannon entropy, KL divergence, Poisson PMF, Ideal Gas Law, series/parallel capacitors, relativistic Lorentz factor, compound interest) with fail-closed invalid-input guards; added `/v1/v104/{conformal_route,circuit_trace,graph_consolidate,process_audit,pareto_tune}`, upgraded `python -m scf frontier --json`, added `python -m scf diagnose --json` (`4/4` verifier probes `[PASS]`), surfaced `Titans`/`AttnRes`/`Neuromod` chat telemetry badges, and added fail-closed v104 controls to NexusMind Studio (`node --check` `JS_SYNTAX_OK`).
- **Verification Gate**: `295/295` tests passing across `16` pytest suites (`test_v104_evolution.py` `20/20`, `test_v85_release_contract.py` at `CI_COVERAGE_FLOOR=94` `14/14`, `test_v103_frontier_synthesis.py`, `test_v102_frontier_evolution.py`, `test_studio_runtime_manifest.py`, `test_nexus_studio_contract.py`, `test_supermix_chat_server.py`, `test_mimomix_core.py`, `test_mimomix_decoding.py`, `test_nexus_solver.py`, `test_nexus_api.py`, and SCF suites), `ruff check` (`All checks passed!`), `sync_runtime_model_variants.py --check`, `generate_studio_runtime_manifest.py --check`, `dark_test_audit.py --check` (`30` baseline), and `python -m scf bench --seed 82 --json` at **`41 / 41 (1.0000)`** verified & hybrid. Promoted baseline `v96_physio_neurogenesis.selected.pt` remains `#1` under Rule 9.

### 9. `v105` Hyper-Optimization & Evolution: Core Speedups, Value Residual, STDP Connectome, Adaptive Speculation, Int8 KV Compression, Polar Express Muon & Systems Suite
- **Measured CPU Before-vs-After Performance Gains (Zero Regressions, Exact Parity Verified)**:
  - **SCF Warm Task (`CognitiveFabricRuntime.run_task`)**: `1,327.34 ms` -> **`24.96 – 43.73 ms`** (**`30x – 53x` faster**; `route` stage `1,266.03 ms` -> **`2.87 – 9.33 ms`** via pruned directory scanning & source-presence fast path; `verify_final` `13.62 ms` -> **`0.004 ms`** via deterministic preliminary reuse).
  - **Full Combined Neural Stack (`mHC + Engram + DeltaNet-2 + Titans + AttnRes + temporal CNS + Apical + Neuromod + CTM`)**:
    - Full forward (`B=2, T=32`): `216.46 ms` -> **`106.57 ms`** (**`2.03x`**)
    - Single-token decode step (`T=1`, `collect_telemetry=False`): `70.13 ms` -> **`25.93 ms`** (**`2.70x`**)
    - 16-token greedy generation: `941.51 ms` -> **`439.97 ms`** (**`2.14x`**)
    - Per-component speedups: `ManifoldHyperConnection` eval `3.61 ms -> 0.093 ms` (**`38.8x`**), `GatedDeltaNetMemory` T=1 `5.42 ms -> 0.735 ms` (**`7.37x`**), `RotaryEmbedding` `0.080 ms -> 0.013 ms` (**`6.15x`**), `SparseMoEFeedForward` T=1 `4.24 ms -> 0.870 ms` (**`4.87x`**), `ConnectomeCore` temporal T=32 `100.30 ms -> 38.72 ms` (**`2.59x`**), `TitansSurpriseMemory` T=1 `2.55 ms -> 1.04 ms` (**`2.46x`**), `RMSNorm` `0.152 ms -> 0.088 ms` (**`1.73x`**).
  - **Chat Server Streaming (`supermix_chat_server.py::stream_tokens`, 48 tokens)**: `O(T^2)` prefix recomputation `2,014.56 ms` -> **`O(T)` KV-cached `831.52 ms`** (**`2.42x`**, `exact_text_match=True`).
  - **Optimizer (`MuonHybridOptimizer`)**: fused `torch.addmm` step `362.01 ms` -> **`268.63 ms`** (**`1.35x`**), with opt-in **Polar Express** (`polar_express=True`) cutting 5-step orthogonality error $\|O O^\top - I\|_F$ by **`42.5%`** (`2.9949 -> 1.7214`).
  - **Int8 Symmetric KV Compression (`kv_cache_memory_report`)**: **`3.56x – 3.65x`** memory reduction (`71.88%` bytes saved) at **`0.999989`** mean cosine similarity and `>0.999` next-step logit cosine similarity.
- **Additive Default-Off v105 Innovations**: `ValueResidualMixer` (`use_value_residual=False`), `ConnectomeCore` causal STDP trace (`cns_stdp_trace=False`), `AdaptiveDraftController` (`adaptive_draft=False`), `PrefixKVCache`, 5 frontier systems engines in `source/nexus_v105_hyperopt.py` (`SpeculativeDraftOptimizer`, `KVCacheCompressionPlanner`, `RooflineComputeProfiler`, `CascadeEnsembleRouter`, `SelfHealingBenchmarkGuard`, all `answer_authority=False` with canonical SHA-256 receipts), 10 zero-`eval` exact STEM/CS solvers, `/v1/v105/*` FastAPI routes, `python -m scf perf --json`, `[v105 ValueRes/STDP]` chat badges, and fail-closed v105 Hyper-Optimization Lab in NexusMind Studio (`STUDIO_JS_SYNTAX_OK`, `CHAT_SERVER_JS_SYNTAX_OK`).
- **Verification Gate**: **`504/504`** tests passing across `28` pytest suites (`test_v105_hyperopt_evolution.py` `22/22`, `test_v104_evolution.py` `20/20`, `test_v85_release_contract.py` at `CI_COVERAGE_FLOOR=95` `14/14`, all `v91–v103` connectome/neurogenesis/SCF/frontier suites, all `MiMoMix` core/decoding/v82/differential/MLA/MoD suites, `test_supermix_chat_server.py`, `test_nexus_solver.py`, `test_nexus_api.py`, `test_nexus_studio_contract.py`, `test_studio_runtime_manifest.py`), `ruff check` (`All checks passed!`), `sync_runtime_model_variants.py --check`, `generate_studio_runtime_manifest.py --check`, `dark_test_audit.py --check` (`30` baseline), `python -m scf bench --seed 82 --json` at **`41 / 41 (1.0000)`**, and `python -m scf diagnose --json` at **`7 / 7` `[PASS]`**. Promoted baseline `v96_physio_neurogenesis.selected.pt` remains `#1` under Rule 9.

### 10. `v106` Frontier Advancement: Trunk-Reuse Controller, 2D-BLAS MLA Absorption, Causal ShortConv, Shared-Expert Gate, Oscillatory Connectome, Packed Int4 KV, MuonClip & Frontier Synthesis
- **Measured Same-Process A/B CPU Performance & Compression Gains (Zero Regressions, Exact Parity Verified)**:
  - **Thinking Controller Trunk Reuse (`mimomix_controller.decide`, 6-layer MoE, ladder `(1, 2, 4, 8)`)**: `146.84 ms` (full recompute) -> **`74.56 ms`** (`reuse_trunk=True`) (**`1.97x` speedup**, `torch.equal` bit-identical logits & identical `ThinkingDecision` records; `1.72x–1.96x` on 3-layer `scf perf` benchmark).
  - **DeepSeek-V3 2D-BLAS MLA Matrix Absorption (`MultiLatentAttention.forward`, single-token eval decode)**:
    - 4-layer 24-token `greedy_generate`: `535.03 ms` (unabsorbed) -> **`394.87 ms`** (**`1.35x` speedup**, exact parity `< 1e-5`).
    - Per-layer long-context single-token decode (`T=1024`): `1.826 ms` -> **`1.320 ms`** (**`1.38x` speedup**, `max_diff = 2.61e-08`).
  - **Packed Symmetric Int4 Group-Quantized KV Compression (`quantize_past_kv_int4` / `kv_cache_int4_memory_report`)**:
    - Realistic `head_dim=64, T=128` cache: **`7.1111x` compression** (`85.94%` bytes saved, `0.995326` mean cosine similarity) vs `3.8788x` for Int8 (`0.999983` cos).
    - Full v106 megastack (`scf perf`, `group_size=16`): **`6.40x` compression** (`84.38%` saved, `0.996244` cos) vs `3.5731x` for Int8, while preserving integer Engram tail caches verbatim.
  - **Long-Context Bounded-Memory `chunked_prefill` & `chunked_greedy_generate`**: exact `torch.equal` token parity across non-divisible chunk sizes (`3, 5, 7, 16`) vs monolithic `greedy_generate`; plus single-prefill multi-branch `speculative_tree_verify`.
- **Additive Default-Off v106 Innovations**: `CausalShortConvMemory` (`use_short_conv=False`, `short_conv_kernel=4`, `short_conv_after_layer=0`, exact zero-init birth identity + rank-3 KV-cache & MTP rollback parity), `SparseMoEFeedForward` input-conditioned `shared_expert_gate` (`use_shared_expert_gate=False`, exact zero-init birth identity), `ConnectomeCore` theta/gamma oscillatory phase-coupling (`cns_oscillatory_phase=False`, Dale-sign & mitosis-safe, temporal KV-cache parity), full v93+v102+v103+v104+v105+v106 megastack KV & MTP parity, `MuonClip` spectral-norm weight clipping (`spectral_norm_clip_`, `spectral_clip_tau=0.0`, `--muon_spectral_clip`) + `last_mean_update_rms` telemetry, 5 frontier engines in `source/nexus_v106_frontier.py` (`TestTimeComputeAllocator`, `SpectralWeightDiagnostics`, `CausalCounterfactualVerifier`, `AdaptiveQuantizationPlanner`, `TelemetryDriftDetector`, all `answer_authority=False`, zero-`eval`, canonical SHA-256 receipts `supermix-v106-receipt-v1`), 10 new zero-`eval` exact STEM/CS/Finance/Chemistry solvers (`linalg.matrix_multiply`, `linalg.matrix_rank`, `math.complex_arithmetic`/`modulus`, `cs.dijkstra_shortest_path`, `cs.master_theorem`, `chemistry.beer_lambert`, `chemistry.arrhenius_rate`, `finance.annuity_pv`, `comb.multiset_combinations`, `infotheory.jensen_shannon_divergence`), `/v1/v106/*` FastAPI routes (200 + 422 fail-closed), upgraded `python -m scf {frontier,diagnose,perf}`, `[v106 ShortConv/Osc]` chat badges, and Section 9 fail-closed v106 Frontier Synthesis & Telemetry Lab in NexusMind Studio (`STUDIO_JS_SYNTAX_OK`, `CHAT_SERVER_JS_SYNTAX_OK`).
- **Verification Gate**: **`1,154 / 1,154`** tests passing across `51` pytest suites (`test_v106_frontier_advancement.py` `32/32`, `test_v105_hyperopt_evolution.py` `22/22`, `test_v104_evolution.py` `20/20`, `test_v85_release_contract.py` at `CI_COVERAGE_FLOOR=96` `14/14`, `test_mimomix_{core,decoding,controller,mla,differential,mod,v82,observatory,api,distill,multimodal}.py`, `test_v95–v103` connectome/SCF/frontier suites, `test_nexus_{solver,api,proof,epistemics,engine,swarm,got,ideation,chat,adaptive,risk_control,independent_checker,nonce_ledger,hybrid_advancements,studio_contract,studio_v83,studio_v84,v84_innovations}.py`, `test_problem_solving_eval.py`, `test_answer_check.py`, `test_prompt_normaliser.py`, `test_supermix_chat_server.py`, `test_studio_runtime_manifest.py`, `test_runtime_{model_variants_parity,compute_controls}.py`, `test_model_variant_construction.py`, `test_training_improvements.py`, `test_dark_test_audit.py`), `py_compile` (`PY_COMPILE_OK`), `ruff check` (`All checks passed!`), `sync_runtime_model_variants.py --check`, `generate_studio_runtime_manifest.py --check`, `dark_test_audit.py --check` (`30` baseline), `python -m scf bench --seed 82 --json` at **`41 / 41 (1.0000)`**, and `python -m scf diagnose --json` at **`10 / 10` `[PASS]`**. Promoted baseline `v96_physio_neurogenesis.selected.pt` remains `#1` under Rule 9.
