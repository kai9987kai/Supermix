# Supermix v95: Unified Frontier Synthesis

## Overview

`v95` unifies frontier algorithmic advances across the `kai9987kai` research ecosystem into `Supermix` with **zero regressions**, full checkpoint backward-compatibility (all new `MiMoMixConfig` physiological options default to `False` / `0.0`), and strict separation between deterministic verifier authority and advisory/diagnostic engines (`answer_authority=False`).

---

## 1. Biological & Structural Connectome Upgrades (`source/mimomix_core.py`)

Five additive, default-off fields on `MiMoMixConfig` govern new `ConnectomeCore` mechanisms:

| Config Field | Default | Research Origin | Mechanism |
|---|---:|---|---|
| `cns_dual_soma` | `False` | ODDITY-LAB / SONAPSE-CONNECTOME | Adds coupled slow-timescale soma state (`slow_leak_logit`, `slow_coupling`) per module in `_per_token_rates` and `_temporal_rates`. |
| `cns_slow_leak_init` | `0.05` | SONAPSE-CONNECTOME | Initial retention leak ratio for the slow compartment (`(0, 1)`). |
| `cns_commissural_gate` | `False` | Qiu et al. (NeurIPS 2025) | Input-dependent nonnegative sigmoid gate (`commissural_gate_proj`, initialized to `weight=0, bias=+4.0`) modulated across LR/RL tracts via `_split_commissural_weights`, strictly preserving Dale's law signs. |
| `cns_glial_homeostasis` | `False` | SONAPSE-CONNECTOME | Tracks running presynaptic activity EMA (`glial_activity_ema`) during training and applies homeostatic multiplicative gain (`glial_strength`) without corrupting eval determinism. |
| `cns_trunk_dropout` | `0.0` | ODDITY-LAB | Counterfactual trunk-bypass dropout in `ConnectomeCore.write` during training (`[0, 1)`) so the trunk cannot bypass open connectome gates. |

Additionally, `ConnectomeCore` exposes first-order synaptogenesis primitives:
- `ConnectomeCore.probe_signed_edge_gradients(...)`: computes Dale-signed utility scores $\text{ReLU}(-\text{sign}[\text{pre}] \cdot \mathbb{E}[\nabla_{\text{post}} r_{\text{pre}}])$ over unconnected live module pairs (or uses `_edge_grad_probe`).
- `ConnectomeCore.grow_edges_from_scores(...)`: ranks eligible candidate pairs by utility while reserving a bilateral cross-hemisphere quota (`cross_quota`).

---

## 2. Transactional Neurogenesis Adapter & Curriculum (`source/v95_neurogenesis_adapter.py`, `source/prepare_v94.py`)

Resolves the deferred `v94` trainer integration items documented in `docs/V94_PREPARATION.md`:

- **`GradientUtilityCollector`**: gathers per-module gradient-energy utilities (`||grad * param||` + `1e-4 * ||grad||`) across node, tap, dual-soma, glial, and synaptic parameters on `growth_calibration` batches, plus Dale-signed edge scores via `ConnectomeCore._edge_grad_probe` and MoE expert utilities.
- **`TransactionalNeurogenesisAdapter`**: snapshots complete model `state_dict`, optimizer state, `GrowthPolicy` state, and CPU RNG state before executing structural births (`split_module`), prunes (`kill_module`), or synaptogenesis (`grow_edges_from_scores`). Commits only if `v94_growth_policy.witness_acceptance` passes across all frozen cohorts (`rolled_back=False`), otherwise atomically restores pre-mutation state (`rolled_back=True`).
- **`CurriculumMixtureScheduler`**: cosine/linear blending scheduler transitioning from anchor families to `v93`/`v95` task distributions with cumulative token-share drift reporting.
- **`prepare_v94.check_v95_readiness()`**: runtime readiness check confirming `v95_neurogenesis_adapter`, `nexus_v95_ecosystem`, and `v94_growth_policy` availability.

---

## 3. Labeled Conformal Risk Control & Risk-Coverage Profiling (`source/nexus_conformal_stopping.py`)

Preserves strict honesty for unlabeled `calibrate(margins)` (`certified_risk_bound=None`) while adding labeled finite-sample control:

- **`calibrate_labeled(margins, losses, *, b_max=1.0, target_risk_alpha=None)`**: computes the smallest monotone stopping threshold $\lambda^*$ satisfying Angelopoulos et al. (ICLR 2024, `arXiv:2208.02814`) Conformal Risk Control on held-out bounded losses $L_i \in [0, b_{\max}]$.
- **`selective_risk_coverage_profile(margins, losses)`**: computes the empirical Risk-Coverage curve, **AURC**, **Oracle AURC**, and **Excess AURC (E-AURC)** (Geifman & El-Yaniv, NeurIPS 2017).

---

## 4. Symbolic Coverage Extension for v93 Tasks (`source/prompt_normaliser.py`, `source/nexus_solver.py`)

- Extended `prompt_normaliser` for all 11 `v93` task templates: `impulse`, `ohms_current`, `spring_energy`, `permutations`, `final_velocity`, `code_range_sum`, `code_list_count`, `code_neg_index`, `cns_type_count`, `cns_side_count`, and `cns_pair_synapses`.
- Extended `NexusSolver` with constant-acceleration final velocity $v = u + a t$ (`_solve_kinematics_extended`) and bounded Python expression evaluation without `eval()` (`_solve_code_expression`).

---

## 5. NexusMind v95 Frontier Ecosystem (`source/nexus_v95_ecosystem.py`, `nexus_engine.py`, `nexus_api.py`, `web_static/nexus_studio.html`)

Four self-contained NumPy/stdlib diagnostic engines (`answer_authority=False`):

1. **`TetherForensicsEngine`** (`POST /v1/tether/geometry`): Fisher–Rao sphere map $\psi(p)=\sqrt{p}$, pairwise geodesic matrix, Riemannian Karcher/Fréchet barycenter & geodesic variance suppression, exact Robustness Sandwich ($0 \le r^{\text{TV}}_{\text{cert}} \le \delta^{\text{break}}_{\text{TV}} \le 1$ plus tangent-Jacobian $r^{\text{logit}}_{\text{cert}}$), coalition Shapley + Order-2 Shapley-Taylor interaction indices, nonnegative Bregman-KL PID, and Spectral-Mahalanobis dark-knowledge spoof detection.
2. **`FaultlineEnsembleEngine`** (`POST /v1/faultline/panel`): Condition-indexed Poisson-binomial majority-vote failure analysis comparing `"accuracy"`, `"pairwise"`, `"pooled"`, and `"joint"` Faultline minimax panel selection, emitting SHA-256 receipts.
3. **`CrosswireInterfaceEngine`** (`POST /v1/crosswire/synthesize`): Black-box Mealy version-space tracking $V(\mathcal{D}) \subseteq H$, maximum-EIG diagnostic probe ranking, and fail-closed all-hypothesis safety shielding (`CERTIFIED_SAFE` vs `ABSTAIN_UNSAFE`).
4. **`MnemorphArchaeologyEngine`** (`POST /v1/mnemorph/assay`): Bayesian posterior inference across `persistent_subspace_trace`, `regenerative_reconstruction`, and `readout_remapping`, expected information gain (EIG) experiment ranking, paired Brier score, and MORPHEUS `Recovery(t)` trajectory metrics.

An interactive **v95 Ecosystem Lab** tab (`#panel-v95-ecosystem`) in `web_static/nexus_studio.html` exposes live controls and SVG visualizations with strict fail-closed backend requirements.

---

## 6. v96 Physiological Connectome Criticality, Transactional Neurogenesis & Real Continuation Training

- **Spectral Criticality Homeostasis (`source/mimomix_core.py`, `test_v85_release_contract.py`)**: Added default-off `cns_homeostatic_radius: float = 0.0` to `MiMoMixConfig` alongside `ConnectomeCore.estimate_spectral_radius()` (deterministic power iteration) and `ConnectomeCore.project_spectral_radius()` (clamped `±max_scale_step` rescaling of active edges toward target singular norm) with `homeostatic_radius` and `spectral_radius_est` telemetry.
- **Trainer CLI Parity (`source/train_mimomix_talk.py`, `source/train_mimomix_generalisation.py`)**: Wired `--cns_dual_soma`, `--cns_slow_leak_init`, `--cns_commissural_gate`, `--cns_glial_homeostasis`, `--cns_trunk_dropout`, and `--cns_homeostatic_radius` into `_CONFIG_FROM_ARGS`, `RECORDED_HYPERPARAMETERS`, and `build_talk_parser()`, plus `--freeze_trunk_steps`, `--witness_rollback_tolerance`, `--maturity_steps`, `--max_splits_per_parent`, and `--edge_proposal_mode` (`covariance | signed_gradient | hybrid`).
- **Transactional & Physiological `NeurogenesisController` (`source/neurogenesis.py`)**: Added atomic pre-event snapshots (model state, AdamW moments, controller counters) with `witness_rollback_tolerance` restoration, newborn `maturity_steps` protection, `max_splits_per_parent` caps, side-effect-free `_probe_signed_edge_scores` (`_edge_grad_probe`) for `signed_gradient` / `hybrid` synaptogenesis, and post-event spectral criticality projection.
- **Frozen-Trunk Reservoir Readout (`source/v92_temporal_connectome.py`, `output/v92_connectome/report.json`)**: Cached 40 train batches / 57 dev rows from `output/v89_corpus/v89_corpus.pt` (`baseline_dev_token_loss = 7.308e-06`) and ran 9 arms for 150 steps (`T_real_s1/s2`, `T_null_101/102/103`, `T_empty_s1`, `P_real_s1`, and physiological `T_physio_s1/s2` with `dual_soma`, `commissural_gate`, `glial_homeostasis`). Every active arm improved over frozen `v89` (`H1_branch_helps_frozen_v89 = true`, `H4_wiring_vs_empty = -1.014e-06`, CI `[-1.491e-06, -6.322e-07]`), while on this already low-loss slice `H2_fly_wiring_beats_nulls` was `false` (`z = 0.245`) and `H5_physio_vs_vanilla_real = +1.095e-07`.
- **Stratified `v96` Repair Continuation (`source/build_v96_repair_mix.py`, `output/v96_physio_neurogenesis/`)**: Built `datasets/v96/v96_repair_mix.jsonl` (3,200 rows across 15 connectome/omni/code/hard-v89 tasks) and ran a 16-step warm-started continuation from `output/v93_neurogenesis/v93_neurogenesis.selected.pt` with `--freeze_trunk_steps 8`, `--cns_dual_soma`, `--cns_commissural_gate`, `--cns_glial_homeostasis`, `--cns_trunk_dropout 0.25`, `--cns_homeostatic_radius 0.90`, `--edge_proposal_mode hybrid`, `--witness_rollback_tolerance 0.02`, `--maturity_steps 8`, `--max_splits_per_parent 1`. Results (`output/v96_physio_neurogenesis/generalisation_results.json`): selected step 8 (`dev_loss = 0.0861`), `tier1_seen_response = 0.0683` (`ppl 1.0707`, 29 rows), `tier3_unseen_sentence = 0.0947` (`ppl 1.0993`, 277 rows, `1.027x` ratio), step-16 accuracy probe `1.00` (12 tasks), `spectral_projection` `rho: 1.530 -> 1.147 -> 0.977`, `4` module splits / `32` hybrid edges opened / `0` rollbacks, and positive connectome ablations (`cns_ablation = +8.4e-05` nats, `gates_off = +0.000232` nats over 64 rows, `right_only = +0.000259`, `left_only = +6.5e-05`, `grown_off = +5.4e-05`).
