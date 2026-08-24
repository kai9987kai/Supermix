# Research-Guided Upgrades Applied

This repo now includes practical upgrades inspired by recent preference-optimization research for chat alignment.

## Papers Used

1. `DPO` (Rafailov et al., 2023):  
   https://arxiv.org/abs/2305.18290
2. `ORPO` (Hong et al., 2024):  
   https://aclanthology.org/2024.emnlp-main.626/
3. `SimPO` (Meng et al., 2024):  
   https://arxiv.org/abs/2405.14734
4. `RE-PO` (Cao et al., 2025):  
   https://arxiv.org/abs/2509.24159
5. `RPO: Reward-aware Preference Optimization: A Unified Mathematical Framework for Model Alignment` (Sun et al., 2025):  
   https://arxiv.org/abs/2502.00203
6. `LMPO: Length-Controlled Margin-Based Preference Optimization without Reference Model` (Li et al., 2025):  
   https://arxiv.org/abs/2502.14643

## What Was Implemented

1. Pairwise preference objective during fine-tuning:
- Added to `finetune_chat.py` as `_preference_loss(...)`.
- Enabled with `--pref_weight` and `--pref_beta`.
- Combines cross-entropy with a preference-style margin objective (chosen class vs sampled negative class).
- Added hard-negative mining via `--hard_negative_ratio` to preferentially train against confusable classes.
- Added objective selection via `--pref_objective`:
  - `sigmoid` (SimPO/ORPO-style pairwise logistic)
  - `repo_relu` (RePO-style max-margin ReLU objective)

2. Robust preference weighting for noisy data:
- Added `--adaptive_pref_weighting` in `finetune_chat.py`.
- Confidence-weighted preference terms approximate robust/noisy-preference handling from recent work.

3. Expectation-style grouped preference estimation:
- Added `--pref_group_size` for multi-negative preference estimation.
- Added `--pref_group_estimator epo` for expectation-style group reduction over sampled negatives.
- This provides a practical grouped-estimation upgrade for noisy/sparse preference settings and aligns with recent RPO-style design guidance on multi-response preference estimation.

4. Class-imbalance mitigation:
- Added `--balanced_sampler` in `finetune_chat.py` using inverse-frequency sampling.

5. Better retrieval-time response selection:
- Updated `chat_app.py` to fuse `--top_labels` predicted buckets instead of a single bucket.
- Updated `chat_pipeline.py` scoring to include query-context similarity, bucket confidence, and diversity penalties.
- Added response sampling control with `--response_temperature`.
- Added stronger response cleanup in `chat_pipeline.py` to remove near-duplicate clauses and filler fragments.

6. Capacity and style upgrades:
- Added an `xlarge` model option in `model_variants.py` with a dual-adapter routed classifier head.
- Added an `xxlarge` model option in `model_variants.py` with tri-branch routed adapters for higher-capacity classification.
- Extended `finetune_chat.py` and `chat_app.py` with `--model_size xlarge` and `--extra_expansion_dim` support.
- Extended `finetune_chat.py` and `chat_app.py` with `--model_size xxlarge` and `--third_expansion_dim` support.
- Added style-aware reranking in `chat_pipeline.py` plus automatic style inference (`balanced`, `creative`, `concise`, `analyst`).
- Added `--style_mode` and `--creativity` controls in `chat_app.py` for more creative, controllable responses.

7. Reliability upgrades for larger runs:
- Added `--grad_accum_steps` in `finetune_chat.py` for stable large-model optimization with bigger effective batch size.
- Added EMA shadow weights via `--ema_decay` with EMA-based evaluation/saving enabled by default (can disable with `--disable_ema_eval`).

8. Dataset scale upgrade:
- Added `build_super_dataset.py` to merge multiple JSONL corpora, apply quality filtering, dedupe, and scale to a larger training set.

## Usage Example

```bash
python finetune_chat.py --data conversation_data.hybrid_v5_clean.jsonl --weights champion_model_chat_large_ft_v5.pth --model_size large --train_all --balanced_sampler --split_mode stratified --pref_weight 0.22 --pref_beta 2.6 --pref_objective sigmoid --pref_group_size 4 --pref_group_estimator epo --hard_negative_ratio 0.78 --adaptive_pref_weighting --pref_warmup_epochs 1.2 --lr_schedule cosine --warmup_steps 180 --early_stop_patience 2 --epochs 2 --batch_size 128 --device cpu --output champion_model_chat_large_ft_v6.pth --meta chat_model_large_meta_v6.json
```

```bash
python chat_app.py --weights champion_model_chat_large_ft_v6.pth --meta chat_model_large_meta_v6.json --device cpu --pool_mode all --top_labels 4 --response_temperature 0.08 --llm_db llm_chat_v5.db --db_top_k 160
```

---

## March 2026: Qwen Supermix Pipeline Upgrades

Applied in `source/qwen_supermix_pipeline.py` and launcher scripts.

### Recent papers reviewed (primary sources)

1. SimPO (NeurIPS 2024):  
   https://arxiv.org/abs/2405.14734
2. Reward-aware Preference Optimization (RPO, 2025):  
   https://arxiv.org/abs/2502.00203
3. Focused-DPO for code error-prone points (ACL Findings 2025):  
   https://arxiv.org/abs/2502.11475
4. IterPref / Target-DPO for iterative code debugging (2025):  
   https://arxiv.org/abs/2503.02783
5. AP2O-Coder adaptive progressive coding preferences (2025):  
   https://arxiv.org/abs/2510.02393
6. TRPA trust-region preference optimization for reasoning stability (2025):  
   https://arxiv.org/abs/2504.04524
7. QLoRA memory-efficient large-model finetuning (NeurIPS 2023):  
   https://arxiv.org/abs/2305.14314
8. DoRA PEFT stability/capacity improvements (ICML 2024):  
   https://arxiv.org/abs/2402.09353

### What changed in this repo

1. Preference mining stability fixes (hang prevention):
- Added mining modes: `auto`, `hybrid`, `dataset`, `generation`.
- `auto` now disables on-the-fly generation on CPU and mines from dataset negatives.
- Added mining progress logs, wall-clock limits, and attempt limits.
- Added generation failure guards so runs continue instead of silently stalling.

2. Coding/problem-solving focus in training:
- Added prompt-aware SFT weighting (`--sft_prompt_skill_boost`).
- Added reasoning source weighting (`--sft_reasoning_boost`).
- Added preference pair weighting boosts for coding/reasoning prompts:
  `--preference_coding_focus_boost`, `--preference_reasoning_focus_boost`.
- Added counterfactual hard negatives for coding/reasoning preference mining:
  `--preference_counterfactual_rejects_per_prompt`.
  This is a practical adaptation inspired by code-focused preference/error-point training papers.

3. Larger-model practicality:
- Added `--device auto|cpu|cuda|mps`.
- Added `--model_dtype auto|float32|float16|bfloat16`.
- Added `--gradient_checkpointing` for memory-constrained larger-model runs.
- Added automatic CPU safety: gradient checkpointing is disabled on CPU to avoid pre-step stalls.
- Added launch profile: `run_train_qwen_supermix_v23_larger_reasoner.ps1`.

4. Training observability:
- Added GUI monitor: `source/training_monitor_gui.py`
  (stage/step/loss/LR/PID/stall detection for `train_*.out.log`).

5. Advanced preference objectives and stability (new):
- Added ORPO-style objective option in Qwen preference stage:
  `--preference_objective orpo`.
- Added progressive preference schedules inspired by adaptive/progressive alignment:
  `--preference_beta_end`, `--preference_margin_end`.
- Added hard-example emphasis during preference training:
  `--preference_hardness_gamma`.
- Added trust-region style anchoring to the pre-preference policy:
  `--preference_reference_anchor_weight`, `--preference_reference_anchor_batch_size`.
  Implementation caches reference chosen/rejected log-probs before preference updates and penalizes large margin drift.

6. New launch profile for smarter alignment:
- Added `source/run_train_qwen_supermix_v24_adaptive_orpo_trust.ps1`.
- Uses ORPO + progressive schedules + trust-region anchoring for stronger reasoning/coding stability.

7. Preference-data selection curation (new, v25):
- Added post-mining pair selection in `source/qwen_supermix_pipeline.py`:
  `_select_preference_pairs(...)` with paper-inspired curation controls.
- Added strategies:
  - `--preference_selection_strategy margin_topk`
  - `--preference_selection_strategy capacity_aware`
- Added controls:
  - `--preference_selection_keep_ratio`
  - `--preference_selection_min_keep`
  - `--preference_selection_max_keep`
  - `--preference_selection_hardness_target`
  - `--preference_selection_hardness_bandwidth`
- Added training telemetry for selection outcomes (kept/mined ratio + difficulty/quality deltas).
- Added launch profile `source/run_train_qwen_supermix_v25_selective_pref.ps1`.

### Additional papers reviewed for v25 selection

1. Less is More: Improving LLM Alignment via Preference Data Selection (2025):  
   https://arxiv.org/abs/2502.14560
2. Principled Data Selection for Alignment: Hidden Risks of Difficult Examples (2025):  
   https://arxiv.org/abs/2502.09650
3. Towards Understanding Valuable Preference Data for LLM Post-Training (2025):  
   https://arxiv.org/abs/2510.13212

### Notes

- These changes are inspired by the above papers and focused on practical reliability and coding/reasoning gains for this codebase.
- They are not full re-implementations of those methods.

## March 2026: Dialogue-Adherence + Creativity Upgrades

Recent papers additionally used as design input:

1. LongPO: Improving Multi-turn Alignment in LLMs via Preference Optimization for Long Dialogue  
   https://arxiv.org/abs/2509.05179
2. ConsistentChat: Benchmarking and Enhancing Consistency for Multi-turn Conversations in LLMs  
   https://arxiv.org/abs/2506.11034
3. CrPO: Creative Writing Improves Reasoning and Coding in Small Language Models  
   https://arxiv.org/abs/2505.15778
4. Temporal Consistency for LLM Reasoning Process Error Identification  
   https://arxiv.org/abs/2501.13210

Repo changes inspired by those papers:

- Preserved recent multi-turn history when JSONL `messages` are converted into SFT pairs, instead of flattening every assistant response into an isolated single-turn prompt.
- Added conversation-adherence scoring to SFT weighting, teacher distillation filtering, and preference mining so follow-up edits and context-dependent turns get rewarded more consistently.
- Added a new preference-pair curation strategy, `innovation_mix`, that favors a balance of difficulty, reasoning structure, creativity, and dialogue continuity.
- Added runtime follow-up-aware reranking plus a light refine pass for requests like “make it shorter,” “go deeper,” and “make it more creative.”
- Added `context_mix_v3`, a smarter runtime context encoder that reads explicit conversation control tags, topic anchors, and prior-answer focus, and auto-upgrades old `context_v2` metadata paths at inference time.

## March 2026: Recovery + Distillation Quality Upgrades

Additional primary sources reviewed:

1. PAD: Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models  
   https://arxiv.org/abs/2502.14272
2. UAPO: Adaptive Preference Optimization with Uncertainty-aware Utility Anchor  
   https://arxiv.org/abs/2509.10515
3. SPHERE: Self-Evolved Preference Optimization for Enhancing Mathematical Reasoning in Small Language Models  
   https://arxiv.org/abs/2503.04813

Repo changes inspired by those papers:

- Added latest-checkpoint resume support in `source/qwen_supermix_pipeline.py` so long CPU runs can restart from the newest saved adapter while preserving SFT/preference step counters and LR schedules.
- Added cached teacher-distillation reuse per output directory via `teacher_distill_pairs.jsonl`, allowing resumed runs to skip regenerating the same teacher pairs.
- Upgraded Supermix teacher distillation from a single deterministic response to a lightweight best-of-N candidate search across a small temperature set, then kept the highest-scoring filtered response.
- Upgraded teacher candidate ranking again to compare each generated answer against the original assistant answer using gain, density, alignment, and compactness, and require a small rank margin before a teacher rewrite is kept.
- Added a CPU safety guard that disables SFT R-Drop automatically on CPU, because the extra forward pass was making training progress look stalled on this machine.

## March 2026: Data Hygiene + Clean Eval Upgrades

Additional primary sources reviewed:

1. PAD: Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models  
   https://arxiv.org/abs/2502.14272
2. UAPO: Adaptive Preference Optimization with Uncertainty-aware Utility Anchor  
   https://arxiv.org/abs/2509.10515
3. Less is More: Improving LLM Alignment via Preference Data Selection  
   https://arxiv.org/abs/2502.14560
4. Towards Understanding Valuable Preference Data for LLM Post-Training  
   https://arxiv.org/abs/2510.13212

Repo changes inspired by those papers:

- Tightened synthetic/template prompt detection around `*-setN` tags, `genre variant`, `debate framing`, and similar prompt-program artifacts so they can be capped or dropped more reliably.
- Added `--sft_drop_synthetic_prompts` to keep templated synthetic prompts out of the SFT stage even when they survive coarse dataset loading.
- Added `--eval_min_quality_score` and `--eval_drop_synthetic_prompts` so `eval_pairs.jsonl` and benchmark inputs are less contaminated by templated prompts and low-signal responses.
- Upgraded teacher distillation with `--supermix_distill_min_gain`, so teacher responses are only kept when they improve on the original assistant answer by a configurable margin instead of merely clearing a fixed quality floor.
- Updated the launcher recipe toward a cleaner v28 profile with stricter synthetic caps, stronger eval curation, and a nonzero preference reference anchor for better stability.

## March 2026: Throughput + Knowledge-Density Upgrades

Additional primary sources reviewed:

1. Fewer Truncations Improve Language Modeling  
   https://arxiv.org/abs/2404.10830
2. T-SHIRT: Token-Selective Data Selection for Efficient and Improved Training of Large Language Models  
   https://arxiv.org/abs/2506.01317
3. SFT-GO: The Key to Supervised Fine-Tuning Over Groups of Interest  
   https://arxiv.org/abs/2507.12856

Repo changes inspired by those papers:

- Added optional length-bucketed SFT batches with `--sft_length_bucketed_batches` and `--sft_length_bucket_window_mult`, reducing padding waste without changing the current LoRA training objective.
- Added matching preference-stage length bucketing with `--preference_length_bucketed_batches` and `--preference_length_bucket_window_mult`, including the reference-logprob caching pass used by DPO/IPO/XPO-style objectives.
- Added a heuristic knowledge-density signal in `source/qwen_supermix_pipeline.py` and exposed `--sft_knowledge_density_boost` so high-information responses get a controllable weight boost during SFT.
- Threaded knowledge density into preference mining and `innovation_mix` selection so dense reasoning/coding targets are favored during post-mining curation instead of only by source-name heuristics.
- Added `--supermix_distill_density_bias` so teacher distillation can prefer denser candidate answers when quality is close, improving the targets before SFT weighting sees them.

Notes:

- These are practical adaptations guided by the above papers, not exact reproductions of their full algorithms.
- The batching change is a safe length-bucketing proxy for more aggressive packing methods, chosen to fit the current weighted SFT/preference implementation without destabilizing resume or checkpoint flows.

## March 2026: Selective SFT + Compact Reasoning Negatives

Additional primary sources reviewed:

1. Less is More: Improving LLM Alignment via Preference Data Selection  
   https://arxiv.org/abs/2502.14560
2. Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples  
   https://arxiv.org/abs/2502.09650
3. Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning: Eliciting Efficient Reasoning in Large Language Models  
   https://arxiv.org/abs/2505.03469
4. QFFT, Question-Free Fine-Tuning for Adaptive Reasoning  
   https://arxiv.org/abs/2506.12860
5. Thinking Preference Optimization  
   https://arxiv.org/abs/2502.13173

Repo changes inspired by those papers:

- Added optional SFT pair selection in `source/qwen_supermix_pipeline.py`:
  - `--sft_selection_strategy none|utility_topk|capacity_aware`
  - `--sft_selection_keep_ratio`
  - `--sft_selection_min_keep`
  - `--sft_selection_max_keep`
  - `--sft_selection_hardness_target`
  - `--sft_selection_hardness_bandwidth`
- The new selector scores filtered SFT pairs by a mix of response quality, knowledge density, reasoning signal, prompt complexity, and compactness, then trims low-value or overly hard examples before tokenization/training.
- This is a practical capacity-matched data-selection pass for the current 0.5B LoRA recipe: it reduces SFT compute while favoring denser, more learnable reasoning examples.
- Added compact reasoning variants inside `_counterfactual_reject_variants(...)`, so coding/reasoning preference mining now produces short-CoT-style rejected answers in addition to the older drop/nudge/flip variants.
- This is an LS-Mixture / QFFT / Thinking-Preference-style adaptation to your existing preference miner: it creates concise-but-weaker reasoning negatives without introducing a separate RL stage.
- Kept the selector opt-in after the first benchmarked candidate regressed; the smoke/full launchers stay on the last accepted baseline and `source/run_autoresearch_smoke.ps1` can pass explicit extra training args for future candidate runs.

Observed effect in the first smoke benchmarked run:

- Training still completed successfully on CPU/auto with the new selector enabled.
- The selector trimmed SFT source pairs from `89` to `75` in the smoke run before augmentation.
- The resulting candidate was benchmarked and correctly marked `discard`, so the workflow remains promotion-safe even when a paper-guided change does not improve quality.
- Example replay command for this rejected candidate:
  `powershell -NoProfile -ExecutionPolicy Bypass -File source\run_autoresearch_smoke.ps1 -RunTag paper_iter_retry -Description "paper-guided sft selection + compact rejects" -ExtraTrainingArgs @("--sft_selection_strategy","capacity_aware","--sft_selection_keep_ratio","0.84","--sft_selection_hardness_target","0.56","--sft_selection_hardness_bandwidth","0.30")`

## March 2026: Token-Budgeted SFT Selection

Additional primary sources reviewed:

1. T-SHIRT: Token-Selective Data Selection for Efficient and Improved Training of Large Language Models  
   https://arxiv.org/abs/2506.01317
2. Long-Short Chain-of-Thought Mixture Supervised Fine-Tuning: Eliciting Efficient Reasoning in Large Language Models  
   https://arxiv.org/abs/2505.03469
3. SFT-GO: Supervised Fine-Tuning with Group Optimization for Large Language Models  
   https://arxiv.org/abs/2506.15021

Repo changes inspired by those papers:

- Extended SFT pair selection with `--sft_selection_budget_mode pairs|tokens`, so `keep_ratio` can target an estimated token budget instead of only a pair-count budget.
- Added `--sft_selection_budget_power` to rank candidates by a softer utility-per-length score when token-budget mode is enabled, reducing the chance that a few long answers consume most of the training budget.
- Added scoped selection controls, `--sft_selection_scope` and `--sft_selection_scope_min_words`, so token-budget trimming can be restricted to verbose synthetic or teacher-generated SFT rows while the rest of the curated set passes through unchanged.
- This is a pragmatic T-SHIRT-style adaptation for the current weighted LoRA pipeline: prefer dense, useful pairs under a token budget without changing the SFT loss or resume/checkpoint flows.
- The new mode stays opt-in through `source/run_autoresearch_smoke.ps1` extra args until it proves itself in benchmarks.

## March 2026: Length-Controlled Preference Margins

Additional primary sources reviewed:

1. LMPO: Length-Controlled Margin-Based Preference Optimization without Reference Model  
   https://arxiv.org/abs/2502.14643
2. Correcting the Mythos of KL-Regularization: Direct Alignment Algorithms with Downsampled KL Divergence Better Mitigate Over-Optimization than RLHF  
   https://arxiv.org/abs/2407.13399

Repo changes inspired by those papers:

- Added `--preference_length_control_strength`, `--preference_length_control_target_ratio`, and `--preference_length_control_max_penalty`.
- These controls add an extra margin penalty inside the preference objective when the chosen response is much longer than the rejected response, instead of relying only on post-hoc sample weighting.
- The implementation is lightweight and works directly in the current IPO/DPO-style training loop, which makes it a practical verbosity-control adaptation for this repo's benchmarked over-generation failures.

## March 2026: Stop-Aware Preference Mining

Repo changes:

- Added `--preference_stop_signal_strength` to apply a prompt-aware brevity bonus/penalty during preference mining for answer-only, yes/no, one-word, one-sentence, and concise prompts.
- Added `--preference_stop_rejects_per_prompt` to inject synthetic overlong rejected variants, creating hard negatives that explicitly model "kept talking after the right answer."
- The implementation stays in the mining stage, so it works on the current CPU-safe smoke path where on-the-fly generation is usually disabled.
- This remains opt-in through `source/run_autoresearch_smoke.ps1` extra args until it produces a benchmark win.

## March 2026: Budgeted Self-Play Preference Mining

Additional primary sources reviewed:

1. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
   https://arxiv.org/abs/2401.01335
2. SPACE: Noise Contrastive Estimation Stabilizes Self-Play Fine-Tuning for Large Language Models
   https://arxiv.org/abs/2512.07175
3. Beyond Scaling Law: A Data-Efficient Distillation Framework for Reasoning
   https://arxiv.org/abs/2508.09883

Repo changes:

- Added `--preference_self_play_budget`, `--preference_self_play_curriculum`, and `--preference_self_play_max_new_tokens`.
- These controls let the current model generate a small number of self-play negatives during preference mining even on CPU, instead of relying only on static dataset negatives when `generation=off`.
- The current implementation uses a small curated budget and reorders those prompts with an `easy_to_hard` curriculum by default. That curriculum choice is an engineering inference from the papers above, not a direct reproduction of any one method.
- This is meant as a lightweight SPIN / SPACE style adaptation for the existing smoke workflow: keep synthetic opponent data bounded, keep the real chosen answers fixed, and make the preference stage see policy-specific mistakes earlier.

## June 2026: v50 Cognitive Leap Expert (Recursive Latent Reasoning)

Primary sources reviewed:

1. Less is More: Recursive Reasoning with Tiny Networks (TRM)
   https://arxiv.org/abs/2510.04871
2. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach
   https://arxiv.org/abs/2502.05171
3. Hierarchical Reasoning Model (ACT halting + deep supervision)
   https://arxiv.org/abs/2506.21734
4. Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute
   https://arxiv.org/abs/2512.11847
5. Recurrent-Depth VLA: Implicit Test-Time Compute Scaling via Latent Iterative Reasoning
   https://arxiv.org/abs/2602.07845

Repo changes:

- Added `CognitiveLeapExpertHead` and `ChampionNetCognitiveLeapExpert` (model size
  `cognitive_leap_expert`) in `source/model_variants.py`.
- The head keeps TRM's dual-latent split: a scratchpad latent `z_L` refined by a tiny
  weight-tied core over inner steps, and an answer latent `z_H` refined once per outer
  cycle. A single shared core replaces the hypernetwork/tree-search machinery of the
  v22 cognitive head, cutting head parameters roughly 7x (10.8M -> 1.5M) while adding
  recursion depth as the new scaling axis.
- Differentiable ACT-style halting distributes probability mass across cycles and the
  output is the halting-weighted mixture of per-cycle decodes, so easy inputs stop
  early. `last_ponder_cost` exposes expected cycles for optional ponder regularization.
- A latent-convergence regularizer (`last_consistency_loss`) penalizes answer-latent
  drift between later cycles, pushing the recursion toward a fixed point so extra
  test-time cycles refine instead of wander.
- Latents are RMS-normalized onto a learned-gain hypersphere each update
  (nGPT-inspired) for recursion stability at depths beyond the training depth.
- `forward(x, reasoning_cycles=N)` scales test-time compute without retraining; the
  wrapper threads the override through to the head.
- Integrated with `build_model`, checkpoint-compatible weight loading (older
  checkpoints load with the new head freshly initialized), and
  `detect_model_size_from_state_dict`.
- Added `test_cognitive_leap_expert.py`: forward/backward shape checks, gradient-flow
  verification for every new component, halting/consistency diagnostics, and a
  test-time compute scaling check (1 vs 8 cycles must differ; fixed depth must be
  deterministic in eval mode).

## June 2026: v50 Cognitive Leap Iteration 2 (Deep Supervision + Adaptive Compute)

Additional primary sources reviewed:

1. Deep Improvement Supervision (DIS) — per-cycle supervision for looped/recursive
   models, up to 18x training-FLOP reduction vs classic stepwise supervision
   https://arxiv.org/abs/2511.16886
2. Answer Convergence as a Signal for Early Stopping in Reasoning
   https://arxiv.org/abs/2506.02536
3. Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning,
   and Test-Time Compute
   https://arxiv.org/abs/2512.11847
4. Learning Dynamic Recursive Depths for Adaptive Computation
   https://arxiv.org/abs/2507.10524

Repo changes to `CognitiveLeapExpertHead` / `ChampionNetCognitiveLeapExpert`:

- Added `deep_supervision_loss(targets)`: per-cycle decodes are cached during
  training-mode forwards and supervised with progressively increasing weights, so
  every recursion cycle learns to improve on the previous one instead of only the
  final halting mixture receiving gradient. This is a practical DIS adaptation for
  the current classification-head setting, not a full reproduction of the diffusion
  target schedule.
- Added cycle conditioning: a learned `cycle_embed` embedding is added to the
  scratchpad latent each cycle so the weight-tied core knows where it is in the
  recursion (identity conditioning, which the TRM-on-ARC analysis found
  load-bearing). Indices clamp at `max_cycles` so deeper test-time unrolls stay valid.
- Added inference-time convergence early-exit: `forward(..., adaptive_compute=True,
  exit_tol=...)` stops cycling once the answer latent's movement drops below
  tolerance or the ACT halting mass is spent, and assigns the leftover halting mass
  to the exit cycle. Training always unrolls fully; `last_cycles_used` reports the
  realized depth for observability.
- Extended `test_cognitive_leap_expert.py`: deep-supervision loss positivity and
  gradient flow (including `cycle_embed`), early exit under loose tolerance, and a
  zero-tolerance guard proving early exit can never trigger spuriously.

## June 2026: v50 Training Integration + Controlled Benchmark

Repo changes:

- Wired the v50 recursive-head objectives into `source/finetune_chat.py`:
  - `--dis_weight` adds the deep-improvement-supervision loss over per-cycle decodes.
  - `--ponder_weight` adds the ACT ponder-cost penalty (encourages early halting).
  - `--latent_consistency_weight` adds the latent fixed-point regularizer.
  - All three are gated on head capability checks, so every other model size is
    unaffected. Verified end-to-end: training a `cognitive_leap_expert` from a fresh
    base checkpoint through the full pipeline (featurization, EMA, cosine schedule,
    checkpoint save) with all three objectives active, plus reload via
    `detect_model_size_from_state_dict`.
- Added `source/benchmark_cognitive_leap_v50.py`: a controlled experiment comparing
  the v50 head against the v22 cognitive expert on a chained-modular-arithmetic task
  (sequential composition — the computation recursion is supposed to buy). Identical
  data, optimizer, batch size, and step budget for both models. Results land in
  `output/benchmark_v50_cognitive_leap/benchmark_results.json` and a tracked copy in
  `artifacts/v50_cognitive_leap_benchmark/`.

First benchmark run (seed 42, 6 epochs, CPU):

| metric | v22 cognitive_expert | v50 cognitive_leap_expert |
|---|---|---|
| parameters | 10,785,999 | 1,685,327 (6.4x fewer) |
| train time | 35.5 s | 24.4 s (1.45x faster) |
| test accuracy | 0.2313 | 0.2427 (+1.1 pt) |
| latency (B=256) | 141 ms | 51-68 ms at 1-8 cycles |

- Test-time scaling behaved as designed: accuracy ticked up monotonically with more
  cycles (0.2420 -> 0.2433 from 1 to 8) and adaptive early-exit matched full-depth
  accuracy while using 4 of 8 cycles.
- Honest caveat: at this tiny budget both models are far from task ceiling; the
  result demonstrates parameter/compute efficiency at equal budget, not a final
  capability claim. Longer runs on real repo datasets are the next step.

## March 2026: Sample-Level Benchmark Traces and Research Board Focus

Repo changes:

- Benchmark runs now save `base_samples.jsonl`, `tuned_samples.jsonl`, and `sample_comparison.jsonl` alongside `benchmark_results.json`.
- `benchmark_results.json` now also includes `artifacts` pointers and a compact `sample_summary.worst_regression` block so tools do not need to rescan the full comparison file to show the top failure.
- The training monitor research board now surfaces the selected run's top regression, prompt preview, and tuned/reference preview, and adds a direct `Open selected samples` action.
- Older aggregate-only benchmark artifacts remain readable; the monitor now explicitly reports when a run needs a rerun to generate detailed sample traces instead of showing a blank state.

## June 2026: Runtime Test-Time Compute Controls

Repo changes:

- Exposed the v50 `CognitiveLeapExpertHead` test-time compute path in the terminal and web chat runtimes:
  - `--reasoning_cycles` / `/cycles` selects extra recursive inference depth.
  - `--adaptive_compute` / `/adaptive` enables convergence early-exit.
  - `--adaptive_exit_tol` / `/exit_tol` controls the early-exit tolerance.
  - `--adaptive_exit_entropy` / `/exit_entropy` controls the entropy-based
    early-exit threshold.
- Added `forward_with_runtime_compute(...)` in `source/chat_app.py` and synced it to `runtime_python/chat_app.py`. The helper introspects the model `forward(...)` signature, so these knobs are applied only to variants that support them and are ignored safely by older checkpoints.
- Runtime requests are capped at `MAX_RUNTIME_REASONING_CYCLES` to prevent
  accidental runaway API/CLI calls while still allowing deeper-than-default
  controlled experiments.
- Updated `source/chat_web_app.py` and `runtime_python/chat_web_app.py` to accept per-request `reasoning_cycles`, `adaptive_compute`, `adaptive_exit_tol`, and `adaptive_exit_entropy`, and to return `compute` diagnostics plus `timing_ms.cycles_used`.
- `compute` diagnostics now include `ponder_cost`, `consistency_loss`, and
  `gating_entropy` when the active model exposes those buffers.
- Added `test_runtime_compute_controls.py` covering supported-model forwarding, legacy-model no-op behavior, `/api/chat` payload forwarding, cycle capping, entropy-threshold forwarding, and diagnostics extraction.
- Added a compute-sweep experiment path in `source/chat_web_app.py` and
  `runtime_python/chat_web_app.py`: `/api/compute_sweep` compares multiple
  reasoning-cycle budgets for the same draft prompt without mutating chat
  history, and both browser UIs expose it through a `Sweep` button. Each row
  reports latency, realized cycles, predicted label, confidence, entropy, and
  compute diagnostics.
- Added `auto_compute` for web chat: the runtime can probe a small cycle ladder,
  choose the earliest budget that meets confidence/entropy targets, then run the
  normal response path at that selected budget.
- Promoted compute-budget evaluation into shared `chat_app.py` helpers and
  exposed terminal `--auto_compute`, `/auto_compute`, and `/auto_targets` so
  browser and terminal runtimes can both use confidence/entropy-based budget
  selection.
- Hardened the packaged runtime web renderer to use DOM/textContent for chat and
  candidate text instead of injecting raw response HTML.

Rationale:

- Recent adaptive test-time compute and latent/recurrent reasoning work argues for controllable inference budgets instead of a single fixed-depth runtime. The v50 architecture already had that mechanism; this change makes it reachable from the actual app surfaces and observable during local experiments.

## July 2026: Adaptive Runtime Routing Feedback

Additional primary sources reviewed:

1. ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving
   https://arxiv.org/abs/2604.00136
2. Adaptive LLM Routing under Budget Constraints
   https://arxiv.org/abs/2508.21141
3. LLM Routing with Dueling Feedback
   https://arxiv.org/html/2510.00841v1
4. SeqRoute: Global Budget-Aware Sequential LLM Routing
   https://arxiv.org/abs/2605.25424

Repo changes:

- Added recency-weighted route-feedback summaries with geometric forgetting so
  recent route quality regressions can override stale positive feedback.
- Added route economics capture for agent-mode usage, including estimated and
  actual model-call, tool-call, cost, and latency units.
- The auto agent router now uses session-scoped feedback to downgrade routes
  with recent adaptive regressions, pace high-cost routes under budget profiles,
  and prefer neighboring routes with stronger quality-cost evidence when enough
  relevant samples exist.
- Added optional target-route session pacing via
  `auto_session_budget_target_routes`, so a fixed session budget can be spread
  across an expected number of future route decisions instead of only reacting
  when the remaining budget is already tight.
- Added a non-mutating route-plan preview path via `preview_route_plan(...)` and
  `/api/route_plan`, allowing the web UI to inspect the selected route,
  estimated cost, feedback adjustment, and session-budget adjustment before any
  model inference or memory writes happen.
- Route-plan previews now include `route_alternatives`, a compact cost estimate
  table for each eligible route mode, so the UI can expose the local routing
  frontier before spending inference calls.
- Added `route_frontier` preview metadata that annotates candidate routes with
  heuristic/adaptive quality estimates, budget-fit status, Pareto-frontier
  labels, stable frontier ranks, and a recommended route for the current budget
  profile without changing the actual execution route.
- Route-frontier budget state now separates remaining-session fit from
  route-horizon pacing fit, exposing `remaining_cost_units`,
  `pacing_cap_cost_units`, and `effective_cap_cost_units` so the UI can
  distinguish hard budget exhaustion from deliberate pacing preservation.
- Candidate frontier rows now include a `budget_blocker` classification
  (`remaining_budget`, `pacing_cap`, or none), and all-over-cap previews use
  separate recommendation reasons for hard budget exhaustion versus route-horizon
  pacing preservation.
- Candidate frontier rows now expose `estimated_quality_cost_score`,
  `quality_cost_source`, and `quality_evidence_status`; complete adaptive
  quality-cost evidence can drive balanced/fast/deep route-frontier ranking,
  while incomplete cost samples, regressions, and non-finite telemetry fall back
  to heuristic cost-adjusted estimates.
- Route-frontier previews now distinguish raw `pareto_frontier` candidates from
  `budget_feasible_pareto_frontier` candidates, so a high-quality route can
  remain visible as an efficient tradeoff without being mistaken for a route that
  fits the current session or pacing budget.
- Merged the missing v41/v42 Omni Collective continuation source from the
  sibling `Supermix_27` checkout into the active `Supermix` tree: v41/v42 engine
  wrappers, blueprint/prep scripts, smoke/frontier train scripts, run launchers,
  and focused tests. Generated build outputs, checkpoints, dataset shards, and
  stale package artifacts were intentionally left out of the merge.
- Hardened route-planning estimates with bounded numeric setting coercion for
  loop budgets, web-search budgets, and web-search result limits so malformed UI
  or API values cannot turn a dry-run route preview into a server error.
- The web app exposes route quality, average cost, latency, feedback controls,
  route health, dry-run route planning, and a distinct Pareto-style trace pill
  for cost-aware adaptive route changes.

Notes:

- This is a compact engineering adaptation of online routing ideas, not a full
  contextual-bandit implementation. It stays prompt/session scoped, blocks
  unrelated recent-feedback fallback from changing new prompts, requires actual
  cost evidence for adaptive quality-cost comparisons, and lets explicit session
  budget pacing make the final route-depth decision. The route-count horizon is a
  lightweight global-budget adaptation inspired by sequential-routing work; it
  does not attempt to solve the full online planning problem.

## July 2026: v51 Cognitive Leap Ultra + Completion Build

Primary sources reviewed:

1. MiMo-V2-Flash Technical Report (efficient MoE, hybrid attention, and MTP)
   https://arxiv.org/abs/2601.02780
2. Scaling Test-time Compute for LLM Agents
   https://arxiv.org/abs/2506.12928
3. Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time
   Scaling
   https://arxiv.org/abs/2505.11730
4. Trust but Verify! A Survey on Verification Design for Test-time Scaling
   https://arxiv.org/abs/2508.16665

Repo changes and verified build path:

- Added the v51 `cognitive_leap_ultra_expert`: a recurrent mixture-of-cores,
  cross-latent attention, deep-supervised refinement head with ACT,
  convergence, and entropy-based inference halting.
- Added a controlled v51 trainer/benchmark and a metadata materializer so a
  trained checkpoint can move directly into the terminal and browser runtimes.
- Kept per-request reasoning budgets, adaptive exits, non-mutating compute
  sweeps, and compute diagnostics available through the canonical runtime and
  its root compatibility entry point.
- Completed a bounded CPU training/build run, reloaded the emitted checkpoint,
  and exercised it through the browser UI and `/api/chat` runtime path.
- Replaced linear legacy scheduler replay in the v8 resume fallback with a
  constant-time cursor restore, preserving the learning-rate position without
  dummy optimizer updates or scheduler-order warnings.

Research boundary:

- The v51 head is a compact adaptation of sparse expert routing and adaptive
  test-time reasoning for this classifier-backed architecture. It is not a
  reproduction of MiMo-V2-Flash's full autoregressive hybrid SWA/global-attention
  backbone or its MTP decoding stack.
- Verifier-guided candidate search and larger-scale autoregressive training are
  useful benchmark-gated follow-ons, not capabilities claimed by this build.

## July 2026: Prediction-Stability Verifier for Local Test-Time Scaling

Additional primary sources reviewed:

1. Adaptive Test-Time Compute Allocation for Reasoning LLMs via Constrained
   Policy Optimization
   https://arxiv.org/abs/2604.14853
2. ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM
   Reasoning
   https://arxiv.org/abs/2606.06915
3. LATTS: Locally Adaptive Test-Time Scaling
   https://arxiv.org/abs/2509.20368
4. Step-level Verifier-guided Hybrid Test-Time Scaling for Large Language Models
   https://arxiv.org/abs/2507.15512

Repo changes:

- Added an inference-only post-head verifier to the v51 recurrent head. At each
  cycle it constructs the exact output that would be returned if reasoning
  stopped at that point, applies softmax only over the caller's verified
  allowed-label scope, and tracks the ordered decision boundary through the
  configured rank depth.
- Adaptive inference can now stop when the ordered top-k decision remains
  unchanged for a configurable patience window, its minimum adjacent margin
  clears the calibrated floor, and confidence drift stays below a configurable
  tolerance. This criterion is separate from latent convergence, low entropy,
  and ACT remaining-mass exits.
- Added `prediction_stability_patience` and `prediction_stability_tol` across
  the shared runtime helper, terminal commands, source web API/UI, packaged
  runtime web API/UI, materialized metadata, diagnostics, and launch profile.
- Compute diagnostics now report `exit_reason`, `prediction_streak`, and
  `prediction_confidence_delta`, making the allocation decision inspectable.
- Restored `context_mix_v4` in the packaged `runtime_python` feature pipeline
  and aligned its optimized imperative matcher with source semantics, so the
  same v51 metadata no longer silently downgrades to `context_mix_v3` outside
  the source launcher.
- Added `source/benchmark_v51_prediction_stability.py`, which compares fixed
  and adaptive policies as request-sized inference rather than hiding local
  exits inside one large batch.

First serving-style checkpoint result (64 held-out requests, CPU):

| policy | accuracy | mean cycles | mean latency |
|---|---:|---:|---:|
| fixed 3 cycles | 0.125 | 3.0 | 84.828 ms |
| stability verifier, max 8 | 0.125 | 2.0 | 72.167 ms |

- Prediction agreement was 100%; every request exited with
  `prediction_stable` at cycle 2.
- This reduced recurrent cycles by 33.3% and measured mean latency by 14.9% on
  that serving slice without changing accuracy. The result is a runtime
  efficiency result on a small synthetic checkpoint, not a general reasoning
  quality claim.

## July 2026: Preference-Aware Routing with Calibrated Evidence

Additional primary sources reviewed:

1. RouteLLM: Learning to Route LLMs from Preference Data (ICLR 2025)
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/5503a7c69d48a2f86fc00b3dc09de686-Abstract-Conference.html
2. Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs (BaRP, 2025)
   https://arxiv.org/abs/2510.07429
3. Learning to Route LLMs from Implicit Cost-Performance Preferences via Meta-Learning (2026)
   https://arxiv.org/abs/2606.06178
4. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing (2026)
   https://arxiv.org/abs/2605.18796
5. Correlation-Aware Contextual Bandits with Surrogate Rewards for LLM Routing (2026)
   https://arxiv.org/abs/2607.09015

Repo changes:

- Replaced binary-only route corrections with a controlled feedback taxonomy:
  `good`, `bad_quality`, `needs_deeper`, `too_costly`, and `too_slow`.
- Split feedback into quality, depth, cost-pressure, and latency-pressure axes.
  Cost or latency complaints therefore no longer teach the router that an
  otherwise useful answer was low quality, while `needs_deeper` can request one
  higher route without poisoning quality estimates.
- Added recency-weighted effective sample sizes and Wilson-shaped heuristic
  evidence bands for observed route quality. These are descriptive recent-signal
  gates, not nominal 90% confidence intervals: with decay `0.6`, Kish effective
  sample size approaches a ceiling of `(1 + 0.6) / (1 - 0.6) = 4` even as raw
  feedback grows. Risk-adjusted quality/cost comparisons use these bands when
  both neighboring modes have established recent evidence.
  Emerging mean evidence can still influence routing after the existing
  minimum-evidence and quality-delta gates; sparse or unrelated fallback
  feedback cannot directly reroute an unmatched prompt.
- Exposed confidence bounds, evidence status, preference direction, cost, and
  latency in route health. The dry-run route planner additionally exposes the
  Pareto frontier and risk-adjusted scores.
- Connected the main multimodel runtime to v51 reasoning-cycle and adaptive-exit
  controls. Auto routing now supplies its prompt-derived compute budget to the
  selected backend, while explicit per-request settings still take precedence.
- Added checkpoint `runtime_defaults` handling with clean reload semantics and
  surfaced compute telemetry (cycles used, exit reason, and prediction drift)
  in the multimodel trace.
- Made absent standalone-CLI compute flags inherit checkpoint metadata instead
  of accidentally overriding it with library defaults; explicit command-line
  values still have highest precedence.
- Layered exact lazy-greedy marginal coverage onto the Qwen SFT and preference
  rarity priors. SFT pair-budget and feasibility-aware token-budget coverage,
  plus preference pair-budget coverage, now use heap upper bounds to reduce
  redundant rescoring while preserving deterministic quality-anchored choices.
- Replaced the packaged model-variant proxy with a self-contained source
  snapshot, added an isolated-import regression test and deterministic sync
  checker, and wired the runtime parity gates into GitHub Actions.

Research boundary:

- This is an inspectable, session-scoped engineering adaptation of
  preference-aware and uncertainty-aware routing. It is not a trained
  generalist router, a Bayesian posterior over model utility, or an online
  contextual-bandit regret guarantee.
- Recency-weighted Wilson-shaped bounds are used as heuristic evidence gates,
  not as nominal coverage or proof that sparse, adaptive user feedback is
  statistically independent. Larger offline replay sets and randomized policy
  evaluation remain benchmark-gated follow-ons.

## July 2026: Honest Route Policy Evidence Lab

Additional primary sources reviewed:

1. Confident Off-Policy Evaluation and Selection through Self-Normalized Importance Weighting (AISTATS 2021)
   https://proceedings.mlr.press/v130/kuzborskij21a.html
2. Anytime-valid Off-policy Inference for Contextual Bandits (ICML 2021)
   https://proceedings.mlr.press/v139/karampatziakis21a.html
3. Improved Offline Contextual Bandits with Second-Order Bounds: Betting and Freezing (COLT 2025)
   https://proceedings.mlr.press/v291/ryu25a.html
4. Oracle-Efficient Pessimism: Offline Policy Optimization in Contextual Bandits (AISTATS 2024)
   https://proceedings.mlr.press/v238/wang24a.html
5. Conservative Contextual Bandits with Interleaving (ICML 2023)
   https://proceedings.mlr.press/v202/takemura23a.html

Repo changes:

- Added immutable shadow threshold profiles (`efficiency`, `balanced`, and
  `quality_first`) plus a pure replay module in `source/route_policy_lab.py`.
- Replay joins usage and feedback only by exact server route ID. It reports
  coverage, candidate agreement, observed approval, cost, and latency only for
  rows where the candidate action matches the action that actually ran.
  Changed actions receive no imputed reward.
- The runtime now returns the same UUID route ID that it writes to usage
  memory. Browser feedback sends only that ID, the controlled feedback intent,
  and an optional note; prompt, mode, model, policy, and economics are recovered
  from the server usage row. Repeated feedback for one route becomes an
  idempotent revision instead of an accidental duplicate.
- Route logs now carry policy/version and feature-schema identifiers, the
  safety-filtered eligible set, exact post-filter action probabilities,
  decision context, and the chosen action's logging propensity. The current
  policy is explicitly marked deterministic with a one-hot vector.
- Added a read-only Route Policy Lab panel and API. Its promotion gate remains
  `shadow_only` and blocks automatic promotion when valid randomized overlap is
  absent. Even propensity-ready rows require a separately validated off-policy
  estimator and review; readiness alone is not an estimate.
- Session JSON writes now replace an on-disk temporary file atomically, reducing
  the chance that an interrupted write destroys the policy evidence ledger.

Research boundary:

- Historical deterministic logs do not identify outcomes for unchosen routes.
  The Policy Lab therefore labels its results associational and does not claim
  inverse-propensity, doubly robust, causal, or high-probability policy value.
- Explicit `needs_deeper`, `too_costly`, and `too_slow` feedback remains useful
  as direct one-step user intent. It is separate from policy-promotion evidence.
- A future opt-in explorer must randomize only between safety-filtered adjacent
  routes, record exact post-filter propensities before execution, retain failed
  routes, and evaluate with session-clustered pessimistic bounds before any
  promotion path can be enabled.

## July 2026: Durable Failure-Aware Route Evidence

Additional primary sources reviewed:

1. Off-Policy Evaluation for Recommendations with Missing-Not-At-Random Rewards (2025 preprint)
   https://arxiv.org/abs/2502.08993
2. Conservative Contextual Bandits: Beyond Linear Representations (ICLR 2025)
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html
3. Off-Policy Evaluation for Ranking Policies under Deterministic Logging Policies (ICLR 2026)
   https://arxiv.org/abs/2603.21485
4. Logging Policy Design for Off-Policy Evaluation (2026 preprint)
   https://arxiv.org/abs/2605.15108
5. Anytime-valid Optimal Policy Identification (2026 preprint)
   https://arxiv.org/abs/2606.17515

Repo changes:

- Added `source/route_policy_ledger.py`, a schema-versioned SQLite WAL ledger
  with short transactions and atomic per-session sequence allocation. Only a
  domain-separated session hash is stored; raw session IDs and prompt text are
  excluded from the durable database.
- Route decisions now commit an `inflight` row after final capability, safety,
  and session-budget filtering but before inference. Successes and exceptions
  transition that row separately, so backend, economics, memory, and
  serialization failures no longer disappear from the evidence base.
- Abrupt process termination intentionally leaves an `inflight` row rather than
  inventing a failure outcome. The Policy Lab exposes completed, failed, and
  in-flight counts separately.
- Explicit feedback is appended as an idempotent, route-ID-keyed revision. A
  missing response remains `unknown`; approval is always shown alongside
  terminal feedback coverage instead of treating silence as neutral or bad.
- The existing JSON store remains a bounded compatibility mirror for adaptive
  route preferences. The durable ledger is the lifecycle source of truth and
  the Policy Lab remains read-only and `shadow_only`.

Research boundary:

- The new ledger fixes survivorship and join integrity, but the current route
  probabilities are still deterministic one-hot vectors. It therefore improves
  traceable associational diagnostics without enabling IPS, SNIPS, doubly
  robust, causal, regret, or anytime-valid policy-superiority claims.
- Exploration remains disabled (`epsilon = 0`). Known post-filter randomized
  probabilities, overlap, versioned bounded outcomes, and a separately modeled
  feedback-observation process are prerequisites for future causal evaluation.
- Runtime reliability, quality, latency, and cost remain separate outcomes. A
  cheap or fast failure can never compensate for a reliability or safety floor.

## July 2026: Durable Support Envelope + Policy Readiness Certificate v2

Additional primary sources reviewed:

1. Anytime-valid Optimal Policy Identification (2026 preprint)
   https://arxiv.org/abs/2606.17515
2. Off-Policy Evaluation for Recommendations with Missing-Not-At-Random Rewards (2025 preprint)
   https://arxiv.org/abs/2502.08993
3. Off-Policy Confidence Sequences (COLT 2025)
   https://proceedings.mlr.press/v291/ryu25a.html
4. Supplementary Outcomes for Off-Policy Evaluation (ICLR 2025)
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html

Design contract:

- A versioned durable support envelope binds each route decision to its final
  post-filter candidate set, normalized logging distribution, chosen
  propensity, policy and feature-schema identities, and canonical support
  fingerprints before execution. It is an audit contract; it does not enable a
  different route-selection policy.
- A read-only readiness certificate reports schema validity, lifecycle and
  feedback-observation coverage, empirical action overlap, propensity floors,
  and diagnostic effective sample size (ESS). Each failed check remains an
  explicit reason code instead of being averaged into a single favorable score.
- The certificate fails closed unless evidence comes from a reconciled durable
  lifecycle window, meets its declared valid-route floor, and reproduces the
  versioned support envelope, chosen propensity, assignment commitment, and
  candidate/distribution fingerprints. The bounded JSON mirror is descriptive
  compatibility data only and can never satisfy lifecycle readiness.
- Explicit feedback now commits to SQLite before the JSON compatibility mirror.
  Content-derived idempotency coalesces only an immediately adjacent identical
  retry; an explicit request ID is matched across the route's full revision
  history. A mirror failure returns an accepted-but-pending reconciliation
  status instead of losing or duplicating the durable acknowledgement.
- The current behavior policy remains deterministic with one-hot propensities.
  Consistent with recent work on deterministic logging, these rows cannot
  identify counterfactual outcomes for unchosen routes and therefore cannot
  produce an off-policy value estimate.
- Missing feedback remains `unknown`. Because reward observation can be
  missing-not-at-random (MNAR), silence is neither a negative label nor evidence
  that observation is independent of route choice; observation coverage is
  reported separately by lifecycle state and action. A scalar observation
  propensity is not enough to clear this gate unless its observation policy and
  outcome definition are both versioned.

Research and deployment boundary:

- The certificate computes no IPS, SNIPS, doubly robust, causal, regret, or
  anytime-valid policy-value estimate. Passing structural checks means only
  that a separately validated evaluator could receive better-formed input.
- Exploration remains disabled (`epsilon = 0`), deployment remains
  `shadow_only`, and automatic promotion remains forbidden. Any future opt-in
  randomized policy requires an independently reviewed safety-filtered
  assignment mechanism, bounded outcome contract, and evaluation protocol.
- Propensity, route-count, session-count, coverage, and ESS thresholds are
  diagnostics for a declared cohort. They expose weak support and unstable
  weights, but no fixed ESS threshold is a universal statistical guarantee or
  proof of policy superiority.

## July 2026: Fail-Closed Route Outcome Contract v1

Additional primary sources reviewed:

1. Off-Policy Evaluation under Nonignorable Missing Data (ICML 2025)
   https://proceedings.mlr.press/v267/wang25dt.html
2. A General Framework for Off-Policy Learning with Partially-Observed Reward
   (ICLR 2025)
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html
3. Clarifying Uncertainty Quantification in Off-Policy Evaluation: Beyond
   Effective Sample Sizes, Towards Confidence Intervals (ICML 2026 DEMO)
   https://openreview.net/pdf?id=FuuLorZ6NQ
4. Beyond the Training Distribution: Evaluating Predictions Under Distribution
   Shift and Selection Bias (2026 preprint)
   https://arxiv.org/abs/2606.14506
5. Anytime-valid Optimal Policy Identification (2026 preprint)
   https://arxiv.org/abs/2606.17515
6. Logging Policy Design for Off-Policy Evaluation (2026 preprint)
   https://arxiv.org/abs/2605.15108
7. Off-Policy Evaluation for Ranking Policies under Deterministic Logging
   Policies (ICLR 2026)
   https://arxiv.org/abs/2603.21485

Implementation contract:

- Readiness now checks the entire fixed-as-of durable population, not only the
  feedback join. Missing or duplicate route IDs, orphan feedback, unevaluable
  usage rows, inconsistent terminal states, or a mismatch between chosen and
  executed route fail closed with explicit population/execution reason codes.
- A versioned, prompt-free decision-record fingerprint binds the immutable
  policy identity, decision context, selected action, eligible set, and support
  projection committed before execution. The snapshot recomputes it rather than
  trusting stored metadata; migrated v1 records remain explicitly legacy.
- Session JSON filenames combine a safe display slug with a full session digest.
  Legacy files migrate only when their embedded session identity matches, while
  atomic replacement, transactional schema migration, and one-transaction
  evidence snapshots prevent collisions or mixed-revision projections.
- Feedback retries without an explicit request ID coalesce only when the newest
  revision has identical content. Explicit request IDs remain globally stable
  across the same route's revision history, so an older retry cannot create a
  new revision and reusing its ID with different content fails closed.
- Route Outcome Contract v1 precommits versioned definitions and observation
  semantics for route success, measured cost, measured latency, and user quality.
  Completion records only outcomes actually available; failure records failure
  and measured elapsed time without fabricating cost, while quality remains
  unknown until explicit quality feedback is observed. Canonical hashes and
  decision-start timing are reverified on replay, and missing, late, posthoc, or
  tampered contracts fail the existing outcome-evidence readiness check.
- Maturity telemetry reports per-outcome precommit/observation coverage and the
  share of legacy posthoc contracts. It is diagnostic evidence hygiene for a
  declared fixed-as-of cohort, not an estimator or promotion score.

Research and deployment boundary:

- Fixed-as-of coverage telemetry is not an observation model. It does not
  identify why quality was observed, correct selective labels or covariate
  shift, or turn missing feedback into a reward; supplementary outcomes require
  their own validated relationship to the target outcome.
- ESS diagnoses weight concentration. It is not confidence-interval width,
  empirical coverage, estimator error, or a universal cross-estimator measure
  of uncertainty.
- The lab still computes no OPE or policy-value estimate, and it enables no
  automatic promotion. Deployment remains `shadow_only`, including when all
  structural checks pass.
- Contracts reconstructed for legacy decisions after their outcomes are known
  are posthoc compatibility records. They cannot satisfy pre-execution contract
  maturity, causal identification, or anytime-valid policy-selection claims.

## July 2026: Bounded-Exposure Adjacent-Route Rehearsal v1

Additional primary sources reviewed:

1. Logging Policy Design for Off-Policy Evaluation (2026 preprint)
   https://arxiv.org/abs/2605.15108
2. Conservative Contextual Bandits: Beyond Linear Representations (ICLR 2025)
   https://proceedings.iclr.cc/paper_files/paper/2025/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html
3. Clarifying Uncertainty Quantification in Off-Policy Evaluation: Beyond
   Effective Sample Sizes, Towards Confidence Intervals (ICML 2026 DEMO)
   https://openreview.net/pdf?id=FuuLorZ6NQ
4. Off-Policy Evaluation under Nonignorable Missing Data (ICML 2025)
   https://proceedings.mlr.press/v267/wang25dt.html
5. Anytime-valid Optimal Policy Identification (2026 preprint)
   https://arxiv.org/abs/2606.17515

Implementation contract:

- The read-only planner consumes the final capability- and budget-filtered route
  candidates plus a strict source contract. That contract binds the source
  policy ID/version, feature schema, support schema, candidate-set hash,
  deterministic distribution hash, and Route Outcome Contract schema into the
  draft charter hash.
- The planner rehearses a fixed incumbent-heavy distribution over the incumbent
  and at most its two nearest feasible neighbors; excluded and non-adjacent
  routes receive no probability. `adjacent` is an ordinal exposure heuristic,
  not a target-aware optimal logging-policy claim.
- A 10% alternate allocation is split across the enrolled neighbors. With two
  neighbors this gives each a 5% planned propensity, matching the Policy Lab's
  current structural probability floor. This is a design diagnostic, not a
  claim that 5% is statistically sufficient.
- The canonical hash also binds the ordered candidate/exclusion projection,
  exact probability vector, repeated-stratum horizon, response-rate and
  target-label scenarios, resource envelope, and fail-closed causal boundaries.
  It contains no prompt or session text and performs no ledger, memory, RNG, or
  inference operation.
- A deterministic nonce-based primitive exists only to inspect replayable draw
  mechanics. Its output is a non-ledger rehearsal receipt and non-ledger support
  proposal. Caller-selected nonces are grindable; no seed was committed before
  assignment, no immutable assignment unit was sealed, and the receipt is not a
  randomization commitment or executed propensity record. The runtime never
  calls this primitive.
- The browser and terminal surfaces expose the same planning semantics. Both
  label the output rehearsal-only, keep execution and evidence writes off, and
  expose expected cost, latency-tier exposure, alternate propensity, expected
  traffic, and a simultaneous label-traffic scenario for a declared response
  rate.

Research and deployment boundary:

- For one alternate the target-label forecast inverts an exact binomial tail.
  For two alternates it inverts the exact joint multinomial probability that
  every alternate reaches the target; the displayed confidence is simultaneous,
  not the weaker marginal confidence for one preselected alternate.
- Expected assignments and the constant-response scenario hypothetically repeat
  the same prompt-specific support stratum. They are not a campaign forecast,
  statistical power, estimator precision, confidence in policy value, or
  evidence that feedback is missing at random. A live campaign would require a
  frozen population/context distribution and an identified observation process.
- ESS remains an overlap and weight-concentration diagnostic. It is not a
  universal uncertainty measure, and the planner computes no ESS, IPS, SNIPS,
  doubly robust estimate, confidence sequence, regret bound, or policy value.
- Reserving most planned mass for the incumbent is an operational exposure cap.
  It is not the high-probability baseline-performance guarantee established by
  conservative contextual-bandit algorithms, whose modeling assumptions are
  not satisfied by the current heuristic route-quality signals.
- Activation is explicitly blocked until a target-policy class and estimand,
  outcome/observation/maturity contract, population scope, preassignment seed
  commitment and unique immutable unit, session carryover/interference strategy,
  resource and stopping rules, and external estimator/review are sealed. The
  cited logging-policy design is one-step; Supermix's stateful session behavior
  is not assumed away.
- Rehearsed probabilities are never written as executed propensities. Existing
  deterministic rows remain a separate `auto-route-v2` cohort, automatic
  promotion remains forbidden, and deployment remains `shadow_only`.

## July 2026: Stateful Route Experiment Preflight v1

Additional primary sources reviewed:

1. Anytime-Valid Off-Policy Inference for Contextual Bandits (Journal of Data
   Science, 2024)
   https://jds.acm.org/files/JDS_Issue3_Paper1.pdf
2. Semiparametric Efficient Inference in Adaptive Experiments (CLeaR 2024)
   https://proceedings.mlr.press/v236/cook24a.html
3. Cluster-Adaptive Network A/B Testing (JMLR 2024)
   https://www.jmlr.org/papers/v25/22-0192.html
4. Data-Driven Switchback Experiments (2024 preprint)
   https://arxiv.org/abs/2406.06768
5. Clustered Switchback Experiments (2023 preprint)
   https://arxiv.org/abs/2312.15574
6. Sequentially-Rerandomized Switchback Experiments (2026 preprint)
   https://arxiv.org/abs/2604.02489
7. Logging Policy Design for Off-Policy Evaluation (2026 preprint)
   https://arxiv.org/abs/2605.15108
8. Off-Policy Evaluation under Nonignorable Missing Data (ICML 2025)
   https://proceedings.mlr.press/v267/wang25dt.html

Implementation contract:

- `route_policy_protocol.py` wraps one or more valid adjacent-route rehearsal
  plans without changing explorer-v1 or invalidating its strict hashes. The
  wrapper is order-invariant across unique support strata and requires one
  common source policy/schema cohort.
- The protocol hash freezes a versioned target-policy threshold class, admitted
  support-stratum hashes, a prompt-free population rule and session-hash cluster
  schema, fixed cluster/route ceilings, an outcome-independent analysis schedule,
  the four Route Outcome Contract hashes, and every stateful design declaration.
- Route-level campaign randomization is screened out in v1. The only design
  screens are sticky session-cluster assignment and clustered switchbacks. The
  latter exposes block and washout declarations; the protocol draft itself does
  not assign either mode.
- Carryover, interference, and temporal variation use closed enums rather than
  free text. Unknowns remain incomplete; declarations remain unvalidated. A
  compatible declaration set can reach only `assumptions_declared_unvalidated`,
  never a causal-design certificate or activation state.
- A seed commitment can be bound as a lowercase digest, but the protocol draft
  never generates or reveals its seed and does not implement HMAC assignment.
  The separate shadow registry described below is the only component that opens
  this later workflow; caller-selected nonces remain prohibited.
- The browser and `route_policy_protocol_cli.py` expose the same canonical
  preflight. The CLI can audit hash integrity and fail-closed boundaries. The
  Windows Studio packaging contract includes this console as
  `SupermixRouteStudy.exe` and binds module/schema hashes in
  `studio_runtime_manifest.json`.

Research and deployment boundary:

- Stateful routing can change later context, memory, and user behavior. A
  one-turn contextual-bandit estimator is therefore not silently extended to a
  session-level causal claim. Sticky clustering reduces within-session treatment
  switching; it does not prove that the cluster captures shared memory or all
  cross-cluster exposure.
- Switchback frequency and washout are bias-variance and identification choices.
  Declaring them does not establish finite carryover, rapid mixing, a valid
  interference graph, or unbiased exposure contrasts.
- Anytime-valid inference needs its own theorem-level conditions, including
  predictable exact propensities, support, bounded outcomes, and a frozen policy
  family/alpha budget. Optional-stopping validity would not itself establish
  causal identification, missing-at-random feedback, or deployment safety.
- Missing ratings remain unknown. The cited MNAR method requires specific
  dropout, state-sufficiency, positivity, response-model, and shadow-variable
  assumptions that sparse voluntary ratings do not automatically satisfy.
- The draft retains all eight activation blockers. It performs no I/O, ledger
  write, assignment, inference, OPE, winner selection, or automatic promotion.
  Independent scientific review and external implementation remain mandatory.

## July 2026: Portable Route Protocol Review Bundle v1

Additional integrity and transparency sources reviewed:

1. An Architecture for Trustworthy and Transparent Digital Supply Chains,
   RFC 9943 (June 2026)
   https://www.rfc-editor.org/rfc/rfc9943.html
2. CBOR Object Signing and Encryption (COSE) Receipts, RFC 9942 (June 2026)
   https://www.rfc-editor.org/rfc/rfc9942.html
3. Certificate Transparency Version 2.0, RFC 9162
   https://www.rfc-editor.org/rfc/rfc9162.pdf
4. Rekor transparency-log architecture
   https://docs.sigstore.dev/logging/overview/
5. Logging Policy Design for Off-Policy Evaluation (2026 preprint)
   https://arxiv.org/abs/2605.15108

Implementation contract:

- The compact protocol audit now checks strict nested schemas and frozen v1
  semantics for source inventories, population declarations, stateful design,
  outcome contracts, stopping rules, randomness, blocker statuses, external
  evaluation, prompt-free guarantees, and causal boundaries. Rehashing a draft
  after enabling outcome-dependent stopping, promotion, validation claims, or
  altered outcome definitions no longer passes.
- Compact drafts intentionally omit complete source plans, so their verifier is
  labeled `structural_without_source_plans`. A digest alone cannot prove that a
  claimed study hash came from the canonical explorer.
- `route-study-review-bundle-v1` carries every canonical prompt-free source plan,
  the complete closed builder option set, and the resulting protocol. Full audit
  validates each explorer plan, canonicalizes ordering, rebuilds the protocol,
  requires exact equality, and then verifies the separate bundle hash. Its
  verification label is `full_source_bound_reconstruction`.
- Runtime and browser bundle endpoints accept only the shared closed protocol
  input schema. Prompt, raw session, and free-text fields fail closed. Browser
  requests are capped at 2 MiB and 100 strata; the core/console contract retains
  the 1,000-stratum ceiling.
- The Studio browser keeps source strata only in ephemeral client memory. It can
  add/remove compatible strata, build/download a bundle, and import either a
  bundle or closed build input for server-side reconstruction. It never pools
  heterogeneous strata without predeclared population weights.
- `SupermixRouteStudy.exe --example-bundle` and `--audit-bundle` expose the same
  contract, and Windows CI reconstructs a frozen two-stratum bundle.

Integrity and deployment boundary:

- Full reconstruction proves internal semantic conformance, not authorship or
  historical existence. RFC 9943, RFC 9942, and Certificate Transparency
  distinguish an artifact digest from a signed receipt backed by a verifiable
  append-only data structure. Supermix has no signature, trusted timestamp,
  external witness, inclusion proof, or consistency proof in this increment.
- A valid bundle still does not validate cluster independence, carryover,
  interference, feedback missingness, support under live execution, or any
  causal estimand. Logging-policy choice can materially change OPE error; an
  internally consistent review artifact cannot repair unsupported actions.
- The portable bundle by itself seals or registers nothing. It generates no
  seed, assigns no cluster, executes no route, writes no evidence, estimates no
  policy value, and leaves all eight activation blockers active. The separate
  shadow registry below can consume only this source-bound artifact; it does not
  turn bundle verification into a transparency receipt or activation approval.

## July 2026: Shadow Whole-Policy Commitment/Reveal Registry v1

Additional primary sources reviewed:

1. JSON Canonicalization Scheme (JCS), RFC 8785
   https://www.rfc-editor.org/rfc/rfc8785.html
2. HMAC-based Extract-and-Expand Key Derivation Function (HKDF), RFC 5869
   https://www.rfc-editor.org/rfc/rfc5869.html
3. An Architecture for Trustworthy and Transparent Digital Supply Chains,
   RFC 9943 (June 2026)
   https://www.rfc-editor.org/rfc/rfc9943.html
4. CBOR Object Signing and Encryption (COSE) Receipts, RFC 9942 (June 2026)
   https://www.rfc-editor.org/rfc/rfc9942.html
5. Analysis of Two-Stage Rollout Designs with Clustering for Causal Inference
   under Network Interference (AISTATS / PMLR 258, 2025)
   https://proceedings.mlr.press/v258/cortez-rodriguez25a.html
6. Randomization Tests in Switchback Experiments (2026 preprint)
   https://arxiv.org/abs/2602.23257

Implementation contract:

- `route_policy_shadow_registry.py` owns a schema-v1 SQLite database separate
  from the executed route-decision ledger. The Studio runtime locates it at
  `memory/route-policy-shadow-registry.sqlite3`; no registry artifact is
  eligible for Policy Lab/OPE input or an executed logging-support row.
- Sealing first performs full source-bound reconstruction of a
  `route-study-review-bundle-v1`. The resulting design binding and assignment
  manifest freeze exactly two 50/50 whole-policy arms:
  `incumbent_source_policy`, bound to the source-policy cohort, and
  `candidate_target_policy`, bound to the target-policy class. Prompt-specific
  `eligible_actions` remain support-stratum metadata and cannot silently become
  cluster-level treatment arms.
- `SupermixRouteShadow.exe seal` obtains 256 bits from the operating-system
  CSPRNG, commits the seed to the design, and writes the seed capsule as a
  separate exclusively-created file before the public package enters SQLite.
  POSIX writes enforce mode `0600`; Windows installs a protected single-user
  DACL on the empty file and verifies it before writing, after `fsync`, and on
  later reads. Any Windows ACL failure deletes the new capsule without writing
  seed bytes. The registry contains no seed material before explicit
  post-closure reveal, and command output never prints the seed. Backup,
  transfer, and independent custody remain operator responsibilities.
- `commit` accepts only the exact canonical `session-hash-v1` digest: 64
  lowercase hexadecimal characters produced by `hash_session_identity`. It
  rejects raw identifiers and alternate spellings without normalization,
  derives a study-scoped HMAC pseudonym internally, and persists neither the
  session hash nor the chosen arm. It appends only the pseudonym and an opaque
  assignment-reveal commitment. Pseudonymity is not anonymity, especially once
  the seed is public and an observer can test candidate session hashes.
- `close` atomically freezes the enrolled commitment count and blocks later
  commitments. `reveal` is rejected until closure, verifies the seed opening,
  and then persists it. Bounded `verify` batches reconstruct each whole-policy
  arm and append a matched or mismatched reveal record. Registry state therefore
  advances from accepting commitments, through closed and seed-revealed, to
  reveal verification complete without affecting inference.
- The assignment algorithm uses RFC 5869 extract-and-expand with separate
  context strings for identity and assignment keys, then HMAC-SHA-256 for the
  study pseudonym and integer basis-point draw. The registry's canonical
  artifact subset is informed by RFC 8785: duplicate keys, non-finite values,
  floating-point registry fields, non-ASCII object keys, and integers outside
  the I-JSON-safe range fail closed. This constrained profile avoids claiming
  general cross-language JCS conformance for arbitrary imported bundle JSON.
- SQLite WAL, `BEGIN IMMEDIATE`, foreign keys, immutable-row triggers, closure
  guards, campaign-order indexes, and a domain-separated event hash chain protect
  normal concurrent local use. Read-only snapshots also audit required schema
  objects plus their exact definition fingerprint, reconstruct every stored
  artifact, and match event artifacts to evidence rows. Reveal verification
  preflights closure and seed artifacts, validates each commitment projection,
  and only reports completion after a passing whole-campaign audit. The executed
  ledger independently rejects shadow/rehearsal flags,
  `ledger_eligible=false`, non-`route-support-v1` envelopes, reserved shadow
  assignment-commitment namespaces, and every non-null commitment outside the
  closed `route-execution-assignment-v1:<sha256>` namespace. For randomized
  execution, schema v4 additionally requires `issue_execution_assignment()` to
  append a nonce-sealed, route/session/policy/context/support-bound record in
  the same ledger before `begin_decision()` can verify and bind it exactly once.
  A namespace-shaped caller string or wrapped shadow hash therefore fails
  closed. This establishes local append-only provenance, not proof that the
  upstream sampler was statistically honest or that a host administrator did
  not replace the ledger.
- All mutations are local-console operations. The Studio browser has only the
  read-only `GET /api/route_shadow_registry/status` endpoint and can refresh
  campaign state, commitment/reveal counts, and chain verification. It has no
  seal, commit, close, seed-reveal, or assignment-verification control.
  CLI `status` uses SQLite read-only mode as well. The server defaults to
  `127.0.0.1`; remote binding is an explicit unauthenticated operator choice.
  Browser responses are `Cache-Control: no-store`, while the server reuses a
  full audit only until the database or non-empty WAL signature changes.

Research and deployment boundary:

- A seed commitment and deterministic reconstruction show that stored reveals
  match the sealed local inputs. They do not show that enrollment was complete
  or unbiased, that the custodian did not inspect or regenerate material before
  publication, or that seed custody was independent. No shadow assignment is a
  live route choice or an executed propensity.
- The 2025 clustering analysis exhibits a bias-variance tradeoff under network
  interference: cutting interference edges and balancing cluster covariates are
  not generally the same objective. Binding a declared session cluster does not
  validate that cluster, identify its interference graph, or license the paper's
  estimator for Supermix.
- The 2026 switchback randomization-test framework requires a known assignment
  mechanism and, for its causal effects, non-anticipation and a finite carryover
  horizon. Supermix v1 implements sticky session-cluster shadow commitments, not
  switchback execution or those tests; diagnostics and verified reveals do not
  establish the required assumptions.
- RFC 9943 and RFC 9942 describe signed statements, verifiable data structures,
  transparency services, and signed receipts. This registry has none of those.
  A host administrator can replace the database or remove its triggers, the
  local clock is untrusted, and the event chain has no external anchor, signer,
  witness, inclusion proof, consistency proof, or anti-equivocation service.
- The registry performs no model inference, live assignment, route execution,
  outcome collection, OPE, causal or policy-value estimation, winner selection,
  activation, or automatic promotion. All existing review and activation
  blockers remain in force after a clean verification.
- The target arm's v2 policy-class manifest binds its source feature schema,
  closed extraction rules, thresholds, action order, tie-breaking, and fallback
  semantics. It still does not bind executable runtime code or a code-artifact
  digest. Canonical session hashes remain private inputs; v1 validates their
  syntax and schema but not external cluster-map membership or independence.
  Those controls remain external prerequisites, not properties implied by a
  verified registry.

## July 2026: v51 decision-fidelity release gate and reference fallback

Recent primary sources reviewed:

1. Understanding and Mitigating Premature Confidence for Better LLM Reasoning
   https://arxiv.org/abs/2605.24396
2. MarginGate: Sparse Margin-Triggered Verification for Batch-Invariant LLM Inference
   https://arxiv.org/abs/2605.30218
3. Stop When Reasoning Converges: Semantic-Preserving Early Exit for Reasoning Models
   https://arxiv.org/abs/2605.17672
4. LESS Is More: Mutual-Stability Sampling for Diffusion Language Models
   https://arxiv.org/abs/2606.16908
5. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing
   https://arxiv.org/abs/2605.18796
6. Conformal Thinking: Risk Control for Reasoning on a Compute Budget
   https://arxiv.org/abs/2602.03814
7. MARS: Margin-Adversarial Risk-controlled Stopping for Parallel LLM Test-time Scaling
   https://arxiv.org/abs/2606.12935

Implementation and release evidence:

- `benchmark_v51_prediction_stability.py` now records top-k Jensen-Shannon
  divergence between consecutive full-prefix output distributions. The shared
  top-k support is chosen from the midpoint distribution and all remaining
  probability is retained in one `other` bucket. This is diagnostic telemetry
  only: it cannot trigger an exit or change a model answer.
- The first strict 4,096-request dual-mode gate rejected the `0.0001` margin:
  release mode observed 5 top-1, 18 ordered-top-3, and 10 top-3-set
  disagreements; isolated mode observed 2, 6, and 4. Exact replay attributed
  13 release top-3 failures to legacy `latent_converged` bypasses and five to
  cycle-2 verifier exits. The largest failing certified decision margin was
  `0.000412636`, so `0.0005` is the smallest round candidate screened here.
- The accepted implementation persists the complete ordered rank tuple, makes
  the post-head verifier the sole authority for early exit, and falls back to
  the exact trained three-cycle ACT mixture with
  `decision_reference_budget` when no strictly earlier decision is certified.
  Bounded disagreement records retain sample index, target, decisions, exit
  reason, cycles, margins, and verifier telemetry without retaining raw inputs.
- The clean v5 CPU gate at commit `81c4dbe7` used eight seeds and 512 requests
  per seed in both isolated-verifier and exact release-runtime modes. Both had
  zero top-1, ordered-top-3, top-3-set, and exact-output disagreements, and all
  per-seed accuracy deltas were zero. Of 4,096 requests, 3,941 exited at cycle 2
  and 155 used the exact cycle-3 reference fallback: mean cycles were 2.0378 and
  cycle reduction was 32.0719%. Release-runtime weighted/median-per-seed latency
  reductions were 4.7567%/3.7062%; isolated reductions were
  7.2346%/5.5732%.
- The frozen 16-prompt response-fidelity gate ran source and a `python -I`
  isolated packaged engine with verified module provenance. It observed zero
  response, ordered-top-five, compute-contract, or packaged-behavior mismatches;
  15 prompts certified at cycle 2 and one used the exact reference fallback.
  The progressive accepted-probe controller separately matched the legacy
  controller exactly on 256 requests while reducing forward evaluations by
  31.25% and weighted latency by 30.095%.

Research and statistical boundary:

- Premature Confidence motivates distrust of early commitment, but its tested
  remedy is training-time confidence shaping. PUMA studies semantic convergence
  in generated reasoning, LESS studies diffusion-language-model token
  commitment, MarginGate studies margin-aware numerical routing, UCCI studies
  calibrated cascades, and MARS analyzes a different switching process. Their
  results motivate this design; none directly proves the Supermix stopping rule.
- `0.0005` is a checkpoint/workload operating point, not a universal margin.
  The zero-error observation over 4,096 requests still has a one-sided 95%
  binomial upper bound of roughly 0.073%. Conformal Thinking is the rationale
  for reporting finite-sample risk rather than turning a held-out zero into a
  universal guarantee.
- JSD and total variation remain shadow diagnostics. In the rejected replay
  they did not uniquely identify every decision-boundary failure, so they are
  not substitutes for ordered decisions, protected adjacent gaps, or exact
  fallback. The release evidence is scoped to this checkpoint, synthetic task,
  seed matrix, CPU configuration, and frozen prompt matrix.

## July 2026: Plan-Evaluate interaction intelligence v1

Recent primary sources reviewed:

1. Think$^{2}$: Grounded Metacognitive Reasoning in Large Language Models
   https://arxiv.org/abs/2602.18806
2. Ask don't tell: Reducing sycophancy in large language models
   https://arxiv.org/abs/2602.23971
3. What If We Allocate Test-Time Compute Adaptively?
   https://arxiv.org/abs/2602.01070
4. Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning
   https://arxiv.org/abs/2605.26849
5. ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM
   Reasoning
   https://arxiv.org/abs/2606.06915
6. Scaling with Confidence: Calibrating Confidence of LLMs for Adaptive Test
   Time Scaling
   https://arxiv.org/abs/2607.01612

Implementation contract:

- `interaction_planner.py` constructs one deterministic
  `supermix-interaction-plan-v1` object from the raw user turn and bounded
  recent-turn context. Studio creates it once and propagates it separately from
  routing, memory, tool, and backend prompt scaffolding. The plan records
  cautious intent scores, appraisal and affect continuity cues, ambiguity,
  epistemic risk, a response strategy, and a response contract made of
  observable capabilities. These labels are routing aids, not a claim to infer
  a user's hidden mental state.
- Candidate scoring adds a bounded plan-alignment signal for empathy,
  actionability, reasoning, calibration, clarification, comparison, requested
  steps, independent assessment, and safety support. Positive alignment is
  gated by the existing semantic scores, while explicit unearned agreement,
  overclaiming, and dismissive phrasing contribute bounded penalties. Passing
  no interaction plan preserves the legacy candidate-score path.
- The response contract is evaluated after selection. Automatic changes are
  intentionally restricted to high-precision cases: immediate crisis
  escalation, immediate urgent-medical escalation, explicit sycophantic
  agreement, and explicit dismissive language. Crisis text that already
  contains escalation guidance can receive a short acknowledgement; non-text
  Studio results are diagnosed but are never rewritten.
- Lower-precision findings remain audit and ranking signals only. In
  particular, missing empathy, missing contract capabilities, unsupported
  certainty, topical continuity, and lexical relevance do not authorize a
  rewrite. Quoted, historical, educational, prevention-policy, process-control,
  and general medical-information contexts are negative controls for the
  immediate-safety heuristics.
- Compact diagnostics expose the plan version, intent, strategy, reasoning
  mode, risk tier, cautious affect cue, uncertainty, factuality and sycophancy
  risk, deliberation summary, response contract, and final guard audit. The
  diagnostics do not include the raw prompt. Studio attaches them to the
  transient agent trace; they do not become route-quality evidence or durable
  route-policy-ledger support.
- The canonical source planner is mirrored in `runtime_python`. Terminal and
  web Champion paths pass the same plan through ranking and finalization; the
  Studio manager creates it once and uses a common post-route finalizer across
  direct, collective, loop, specialist, Champion, and Qwen text routes. The
  standalone Qwen engine and all three static-browser copies implement a
  mirrored Plan-Evaluate boundary and golden negative-control contract, with
  browser diagnostics shown alongside each response.
- Interaction intelligence is enabled by default for normal chat. Controlled
  raw-response evaluation can opt out explicitly with
  `settings.interaction_intelligence=false` in a Studio request, or
  `interaction_enabled=False` in direct Champion or Qwen engine calls. The
  opt-out constructs no plan, applies no interaction reranking or final guard,
  and emits no interaction trace; it exists for fidelity measurement rather
  than as a user-facing safety override.
- `compute_advice.role` is fixed to `shadow_advisory_only`,
  `activation_available` is false, and the suggested reasoning floor is
  bounded. The advice cannot alter a runtime compute request or authorize an
  adaptive exit. The existing checkpoint-bound prediction verifier remains
  the sole decision-exit authority.

Research and evaluation boundary:

- Think$^{2}$ motivates making planning and evaluation explicit, but Supermix
  v1 is a deterministic cue-and-contract layer rather than that paper's
  prompting architecture or learned metacognitive controller. It does not
  demonstrate self-awareness, semantic understanding, or successful
  self-correction.
- Ask don't tell shows that user framing can affect sycophancy and that
  question-form reframing can reduce it in the studied models. Supermix neither
  reproduces that experiment nor rewrites the user's input; it uses a bounded
  agreement-risk signal to prefer independent assessment and blocks only
  explicit unearned agreement after selection.
- Adaptive allocation, uncertainty-aware budgeting, ThinkBooster, and
  confidence-calibrated scaling motivate keeping difficulty, epistemic risk,
  value-of-compute, and confidence visible. Their methods use different models,
  scorers, sampling procedures, training objectives, and benchmarks. They do
  not validate Supermix's lexical heuristics, suggested reasoning floor, or
  response-contract scores, so all compute advice remains non-authoritative.
- The planner is not a truth verifier, medical diagnostic system, crisis
  classifier, calibrated uncertainty estimator, or learned reward model.
  High-precision guards reduce a narrow set of obvious failures but do not
  establish general safety. The negative-control and parity tests validate
  deterministic implementation behavior, not real-world sensitivity,
  specificity, fairness, clinical validity, or improved end-to-end answer
  quality.

## July 2026: Grounded problem solving and verifier-grounded training v1

Recent primary sources reviewed:

1. Uncertainty-Aware Budget Allocation for Adaptive Test-Time Reasoning
   https://arxiv.org/abs/2605.26849
2. ThinkBooster: A Unified Framework for Seamless Test-Time Scaling of LLM
   Reasoning
   https://arxiv.org/abs/2606.06915
3. Search-R1: Training LLMs to Reason and Leverage Search Engines with
   Reinforcement Learning
   https://arxiv.org/abs/2503.09516
4. Sufficient Context: A New Lens on Retrieval Augmented Generation Systems
   https://arxiv.org/abs/2411.06037
5. S2R: Teaching LLMs to Self-verify and Self-correct via Reinforcement
   Learning
   https://arxiv.org/abs/2502.12853
6. TinyV: Reducing False Negatives in Verification
   https://arxiv.org/abs/2505.14625
7. s1: Simple Test-Time Scaling
   https://arxiv.org/abs/2501.19393
8. Correct Answers from Sound Reasoning: Verifiable Process Supervision
   https://arxiv.org/abs/2605.12519

Runtime contract:

- `grounding_runtime.py` is a deterministic, JSON-safe
  `supermix-grounding-v1` layer mirrored exactly in `runtime_python`. It plans
  whether evidence is useful, redacts likely secrets and private paths before
  an external query, ranks bounded evidence with stable `S1` identifiers,
  measures lexical coverage and conflicts, and rejects fabricated citations.
  Its authority block explicitly prevents it from controlling model routes,
  reasoning budgets, adaptive exits, or interaction strategy.
- The response finalizer is deliberately narrow. It changes output only when a
  bounded AST/Fraction arithmetic evaluator safely solves an explicitly
  requested calculation, or when the user explicitly requires an answer based
  only on supplied evidence and that evidence is absent, insufficient, or
  conflicting. Every other grounding finding is audit-only.
- Champion Web now uses the same optional `llm_chat.db` retrieval and
  `chat_memory.db` continuity paths as Champion Terminal. Local database rows
  carry privacy-safe dataset provenance and content hashes; existing databases
  are migrated in place. Source and packaged terminal/database modules remain
  exact mirrors, while the source and packaged web `Engine` knowledge methods
  are AST-parity tested.
- Standalone Qwen receives normalized evidence in a separate system message
  labelled as untrusted reference data. Only supplied `[S#]` identifiers are
  valid. Studio constructs one grounding plan from the raw prompt, reuses web
  tool results as an evidence bundle, audits the final answer before the
  existing interaction finalizer, and renders evidence status and safe source
  links. The offline static browser has a no-eval BigInt rational arithmetic
  mirror and remains byte-identical across its three copies.

Training and evaluation contract:

- `verifiable_reasoning.py` defines `supermix-verifier-v1` with fail-closed,
  non-executable checks for integer, decimal, fraction, normalized exact and
  alias answers, multiple choice, and JSON-field equality. Candidate text is
  never passed to `eval`, a shell, submitted code, the filesystem, or a
  network.
- `build_verifiable_reasoning_curriculum.py` creates deterministic,
  independently seeded train/evaluation JSONL for multi-step arithmetic,
  ratios and probability, sequences, constraint tables, and
  evidence-in-prompt QA. Train and evaluation template identifiers are
  disjoint, every assistant answer is self-verified before inclusion, and the
  manifest records family counts and content hashes.
- The Qwen pipeline now recomputes verifier results instead of trusting cached
  scalar rewards. Verifiably wrong tagged teacher examples are rejected before
  lexical ranking, verified-correct alternatives cannot become preference
  negatives, wrong near-misses are preferred as useful negatives, cached
  tagged distillation rows are revalidated, and evaluation reports overall and
  per-family verified accuracy. Untagged legacy data preserves its previous
  behavior.

Research and evaluation boundary:

- These papers motivate evidence sufficiency, search-aware training, bounded
  test-time scaling, process supervision, and secondary verification. None
  validates this Supermix checkpoint, its lexical evidence coverage, or the
  generated curriculum. The runtime improvement is directly testable; model
  quality can be claimed only after a separate adapter is trained and passes a
  fixed held-out exact-answer/common-benchmark comparison.
- The checked v51 arithmetic classifier is not an open-ended language model,
  and prediction-stability gates measure decision fidelity rather than general
  intelligence. This upgrade therefore avoids using more cycles as a proxy for
  being more knowledgeable.
- A CPU smoke run validates data and training plumbing only. It is not evidence
  that weights improved. Current or changing facts should remain on an
  explicitly enabled retrieval path rather than being memorized from a dated
  synthetic file.

## July 2026: Prompt Understanding v1

Recent primary sources reviewed:

1. ClarifyMT-Bench (multi-turn clarification evaluation)
   https://arxiv.org/abs/2512.21120
2. MulTypo (multi-task typo robustness)
   https://arxiv.org/abs/2510.09536
3. ComplexBench (complex instruction following)
   https://arxiv.org/abs/2407.03978
4. Multi-IF (multi-turn instruction following)
   https://arxiv.org/abs/2410.15553
5. StructFlowBench (dialogue-flow understanding)
   https://arxiv.org/abs/2502.14494
6. RECAP (context-aware intent rewriting)
   https://arxiv.org/abs/2509.04472
7. Structured Uncertainty Guided Clarification (SAGE)
   https://arxiv.org/abs/2511.08798

Implementation contract:

- `prompt_understanding.py` defines the deterministic,
  `supermix-prompt-understanding-v1` profile and is mirrored in the packaged
  `runtime_python` tree. One profile is intended to be computed from the raw
  user turn plus bounded recent turns and reused by interaction planning,
  contextual retrieval, grounding, and response-constraint auditing.
- Parsing is conservative and data-aware. Unicode and whitespace are
  normalized for analysis, while quoted text, fenced and inline code, URLs,
  and paths are masked before instruction matching. Typo recovery uses bounded
  edit distance only for a fixed cue vocabulary. It never rewrites the raw
  prompt or treats arbitrary corrected content as an instruction.
- The profile records multiple requested acts, clause polarity, required and
  forbidden capabilities, quantitative and structural output constraints,
  hard conflicts, softer tensions, turn relations, unresolved references,
  knowledge/freshness/evidence cues, and typo-robust immediate personal-safety
  cues. Prompt-embedded claims about authority or permissions remain
  non-controlling metadata.
- Conflict handling is explicit: mutually impossible hard constraints and
  unresolved required references support one targeted clarification rather
  than a premature answer. Compatible constraints remain a deterministic
  checklist for response auditing. The auditor reports violations but is not a
  truth judge and does not substitute canned factual answers.
- Contextual retrieval can combine the current request with a small number of
  resolved recent turns. It keeps the current turn primary, excludes assistant
  instructions from authority, and remains subject to the existing privacy
  redaction and tool-permission gates.
- Diagnostics are compact and privacy-safe: they include schema/version,
  intent and constraint categories, ambiguity/conflict counts, turn relation,
  and risk/evidence flags, but omit raw or corrected prompt text, quoted
  literals, URLs, and paths. The profile cannot select a model, increase
  compute, enable a tool, widen permissions, or bypass safety controls on its
  own; consumers remain responsible for their existing eligibility and
  permission checks.
- A deterministic curriculum builder covers disjoint train/evaluation
  templates for realistic typo noise, polarity and composed constraints,
  conflict ask-versus-act decisions, multi-turn reference and intent drift,
  and instruction/data separation. Targets are generated from template
  parameters and verifier-checked rather than accepted from the parser's own
  labels.

Research and evaluation boundary:

- ClarifyMT-Bench and Structured Uncertainty Guided Clarification motivate
  separating ambiguity detection from the decision to ask. They do not
  establish that Supermix's deterministic ambiguity scores are calibrated or
  that every clarification decision is optimal.
- MulTypo motivates realistic insertion, deletion, replacement, and
  transposition tests. Supermix deliberately limits recovery to cue words, so
  this is a robustness guard rather than a general spelling corrector.
- ComplexBench and Multi-IF motivate explicit composed-constraint and
  multi-turn checks. StructFlowBench motivates representing dialogue-flow
  relations, while RECAP motivates context-aware retrieval queries. Their
  results use different models, prompts, datasets, and metrics and therefore do
  not validate this implementation.
- The new parser, runtime integrations, deterministic tests, and curriculum
  make the pipeline more capable and testable. Existing Supermix and Qwen model
  weights were not retrained as part of this upgrade. No improvement in trained
  model intelligence, knowledge, benchmark accuracy, or general prompt
  understanding should be claimed until a separate training run and fixed
  held-out evaluation demonstrate it.

---

## July 2026: Deliberate Reasoning v1

Applied in `source/reasoning_engine.py`, mirrored byte-for-byte in
`runtime_python/reasoning_engine.py`, and consumed by `grounding_runtime.py`.

### Motivation

Before this upgrade the only computation Supermix could actually perform at
inference time was an explicit arithmetic expression such as
`Calculate (7 * 9) + 5.`. Anything stated in words — a percentage, a unit
conversion, a rate, an equation — fell through to retrieval-and-rerank, which
selects a plausible-sounding stored response rather than a correct one. That is
the single largest problem-solving gap in the runtime, and it is fixable
deterministically without retraining a checkpoint.

### Papers and ideas used

1. Self-Consistency decoding (Wang et al., 2023):
   https://arxiv.org/abs/2203.11171
2. Program-Aided Language Models (Gao et al., 2023):
   https://arxiv.org/abs/2211.10435
3. Verify step by step / process supervision (Lightman et al., 2023):
   https://arxiv.org/abs/2305.20050
4. Self-Refine and self-verification limits (Madaan et al., 2023):
   https://arxiv.org/abs/2303.17651
5. Large Language Models Cannot Self-Correct Reasoning Yet (Huang et al., 2024):
   https://arxiv.org/abs/2310.01798
6. Adaptive computation and early exit for inference budgets
   (Schuster et al., CALM, 2022): https://arxiv.org/abs/2207.07061
7. Xiaomi MiMo's published direction on hybrid attention, sparse routing, and
   agentic post-training, which motivates the fast/deep budget split rather
   than any specific numeric claim: https://github.com/XiaomiMiMo

### What was implemented

1. Exact rational solving. Every solver computes with `fractions.Fraction`, so
   `10% of 0.1` returns exactly `1/100` and never `0.010000000000000002`. There
   is no `eval`, no network access, and no mutable state.

2. Fifteen solver families with explicit checks: percent of / part-of-whole /
   reverse percent, percent change, ordered percent chains (discount then tax),
   unit conversion across length, mass, volume, time, data, area, speed, and
   temperature, linear equations, speed-distance-time, combined work rates,
   proportions, arithmetic/geometric/quadratic/additive sequences, statistics,
   gcd/lcm/primality/factorization, combinations/permutations/factorials, date
   differences and offsets, simple and compound interest, and sum-and-difference
   word problems.

3. Verification is a precondition for authority, not a report. Each solver
   publishes how its answer was checked and whether that check is independent
   of the path that produced it. Linear equations are re-evaluated by a second
   substitution evaluator that accumulates a numeric total instead of collecting
   symbolic coefficients, so an error in the collector cannot validate itself.
   Unit conversions are checked by exact round trip *and* by a magnitude
   direction test, because a round trip alone cancels an inverted factor.
   Sequence rules must hold for every supplied term, not just the last pair.

4. Cross-solver agreement. In `deep` tier every applicable solver runs and any
   disagreement between solvers of equal verification status marks the attempt
   `conflicting`, which withdraws override authority.

5. Adaptive, bounded compute. A deterministic complexity score over quantity
   count, clause count, unit mentions, and length selects `fast` (stop at the
   first self-verified path) or `deep` (explore all paths, require agreement).
   Solver invocations, literal digit length, sequence and list lengths,
   factorial and combination sizes, date deltas, and result bit width are all
   capped; the adversarial suite holds worst case under 250 ms.

6. One conservative override point. `finalize_grounded_response` replaces a
   retrieved response only when `override_allowed` is true, which requires a
   solved problem, a passing verification, and no conflict. Explicit arithmetic
   keeps its existing dedicated path and takes precedence, and the
   strict-evidence override still outranks both. When the request asks for
   working, the rendered answer includes the recorded steps.

### Research and evaluation boundary

- Self-Consistency motivates sampling several paths and preferring agreement.
  Supermix does not sample a model; it runs disjoint deterministic solvers, so
  agreement here is weaker evidence than a large sampled majority and is used
  only to *withdraw* authority on disagreement, never to manufacture confidence.
- PAL motivates computing rather than narrating an answer. Supermix executes no
  generated code; it dispatches to fixed, bounded solvers.
- Huang et al. is the reason self-verification never grants authority on its
  own. Every check is either an inverse operation, a second independent
  implementation, or a constraint recheck. The one solver whose replay is not
  independent, `percent_chain`, declares `verification.independent: false`.
- CALM motivates budget-aware early exit. The tier here selects how many
  *solvers* run. It is metadata with respect to the model: this layer cannot
  change the reasoning budget, alter routing, or affect the checkpoint-bound
  prediction verifier, and `authority` states this on every result.
- Diagnostics carry class, method, verification, consensus, and budget only —
  never the prompt, the extracted numbers, or the answer.
- This upgrade improves runtime logic, integration, and tests. No model weights
  were retrained. It is not evidence of a smarter trained checkpoint and no
  benchmark improvement should be claimed from it without a separate held-out
  evaluation.

---

## July 2026: Conversation State v1 and the v52 unified merge

### Conversation State v1

`conversation_state.py` adds the first layer in this repo that accumulates
across a session. `analyze_prompt` and `plan_interaction` are both per-turn and
both see a bounded four-turn window, so a constraint the user stated earlier is
simply gone by the time it matters, and an unanswered clarifying question can be
asked again on the next turn.

The layer derives, deterministically and from the turn list each time rather
than by mutating hidden state:

- durable user commitments (identity, tooling, preference, constraint), with a
  later statement about the same subject superseding an earlier one;
- questions the assistant asked, and whether a later user turn answered them,
  including a clarification-loop flag when the same question recurs;
- topic threads and topic shift;
- what the assistant has already delivered, for repetition detection across the
  whole conversation rather than the last two messages;
- user requests the next assistant turn did not engage with;
- contradictions, when a superseding statement flips polarity.

Design boundaries, following the same contract as the interaction, prompt, and
grounding layers:

- Pure and deterministic. The same conversation always produces the same state,
  so any turn sequence is replayable in a test.
- Bounded everywhere: 40 turns, 24 commitments, 8 questions, 6 threads, 24
  delivered signatures, 400 characters per stored span.
- The ranking contribution is clamped to ±0.12, and the parameter defaults to
  `None` with the whole block skipped in that case, so the single-turn ranking
  path remains bit-exact.
- Exactly one behavioural override: a detected clarification loop suppresses a
  third repetition of the same clarifying question. Everything else — repeated
  answers, ignored style requests, unaddressed requests — is audit-only.
- No authority over compute, routing, permissions, tools, or the safety path.
- Diagnostics carry counts, closed-vocabulary category labels, and flags. They
  never carry turn text; `test_conversation_state.py` asserts this directly.

The extraction is deliberately shallow, deterministic pattern matching. It
recognizes stated commitments, not inferred ones, and it will miss commitments
phrased in ways the patterns do not cover. It is a bounded observable-cue layer
in the same spirit as the existing planner, not a learned dialogue-state tracker,
and its recall should not be described as if it were one.

### v52 unified merge

The `cognitive_leap_v52_expert` variant was merged forward from a tree that
predates the v51 prediction-stability work. Its research framing — a separately
trained error-locating verifier rather than model self-correction, difficulty-
proportional test-time compute, sparse conditional execution, and appraisal
heads kept separate from the problem-plan channel — is recorded in
[`docs/V52_UNIFIED_ARCHITECTURE.md`](../docs/V52_UNIFIED_ARCHITECTURE.md), and
the merge decisions in [`docs/V52_MERGE_LOG.md`](../docs/V52_MERGE_LOG.md).

Two boundaries carry over unchanged from that document and are worth restating:

- The named emotion, intent, and strategy outputs are **not** semantic merely
  because the heads exist. They require labelled auxiliary supervision and a
  held-out evaluation before any output is interpreted as an appraisal.
- Sparse top-k core execution is a compute *option*, not a measured speedup.
  Sparse Python dispatch can cost more than it saves for small CPU batches, so
  wall-clock must be benchmarked per deployment. It is off by default: the dense
  path runs unless `core_top_k` is set, and `core_top_k=n_cores` is verified to
  be bit-identical to the dense path.

Fresh and v51-upgraded v52 models start with near-zero residual scales on the
appraisal and verification branches, so an upgraded checkpoint reproduces its
prior behaviour until it is fine-tuned. As with every other upgrade in this
file: no weights were retrained here, so none of this is evidence of a smarter
trained checkpoint.

---

## July 2026: v52 runtime control surface and measured cost

The v52 merge landed the model before the controls that reach it. This entry
records closing that gap, and the measurements taken to check the merge did not
make anything slower or different than it was.

### What was missing

`launch_v52_unified_chat.ps1` passed `--core_top_k 2` to `source/chat_web_app.py`,
whose parser did not define the flag, so the launcher aborted with argparse exit
code 2. More broadly, every v52 capability was unreachable: the sparse router and
the quality/continue verifier were merged model code with no CLI flag, no web
setting, and no telemetry path, and `collect_runtime_compute_metrics` dropped the
seven attributes the merged head emits.

### What was added

- `--core_top_k`, `--verifier_adaptive_compute`, `--verifier_continue_threshold`,
  and `--max_verifier_cycles` in `chat_app.py` and `chat_web_app.py`, threaded
  through `forward_with_runtime_compute`, the compute sweep, and progressive auto
  compute.
- The same four keys in the web app's runtime-compute settings registry, so they
  round-trip through defaults, CLI overrides, and normalisation with the existing
  clamping conventions.
- Seven v52 telemetry fields in `collect_runtime_compute_metrics`:
  `router_load_balance`, `router_z_loss`, `active_cores`, `quality_score`,
  `continue_probability`, `verifier_selection`, `calibrated_entropy`. They stay
  `None` on every pre-v52 variant, which never sets the source attributes.
- `core_routing_mode` in the compute diagnostics, reporting `sparse` only when
  the model actually accepted `core_top_k`, so a control the head ignores is
  never reported as applied.
- `test_v52_runtime_controls.py`, which parses the launcher for the flags it
  passes and asserts the target app's `--help` accepts every one of them. This is
  the contract the shipped defect crossed.

Every control is guarded by `_model_forward_accepts_kwarg`, so pre-v52
checkpoints ignore them entirely.

### Measured cost

Measured on CPU, single thread, median of 12–15 runs. These are this machine's
numbers, not a general claim.

| Path | Before | After | Note |
|---|---|---|---|
| v51 ultra head, dense forward | 26.914 ms | 26.293 ms | bit-identical output |
| Ranking, 60 candidates, no conversation state | 21.146 ms | unchanged | legacy path bit-exact |
| Ranking, 60 candidates, conversation state on | — | +7.6% | was +18.9% before batching |
| v52 forward vs v51 ultra | 32.108 ms | 41.802 ms | +176,668 params, opt-in variant only |
| v52 sparse `core_top_k=2` vs its own dense path | — | 1.63x slower | small CPU batches |

Three findings worth keeping:

1. **Sparse routing is slower here, not faster.** Per-core masking costs more
   than the cores it skips at this batch size. This confirms the donor
   document's own warning and is why the dense path remains the default.
2. **The v51 path was briefly 10% slower after the merge.** The router z-loss
   stacked and reduced raw gate logits on every forward, including inference,
   where nothing reads it. It is now computed only under `self.training`,
   matching how `last_ponder_cost` and `last_consistency_loss` were already
   handled. Load balance stays available in both modes because it reuses the
   mean already computed for the gating entropy.
3. **Conversation-state ranking was 18.9% overhead** because per-candidate
   scoring re-derived the delivered shingle sets, question terms, and thread
   terms for all 60 candidates. `score_candidates_for_conversation` now prepares
   that view once per turn; the batch and single-candidate paths are asserted
   equal.

### Installer upgrade rule

`docs/V52_UNIFIED_ARCHITECTURE.md` states that installer upgrades delete the
prior bundled-model directory so a model removed from a manifest cannot survive
an upgrade. That guarantee was documented but not implemented. The
`[InstallDelete]` rule for `{app}\_internal\bundled_models` is now in
`installer/SupermixStudioDesktop.iss`, matching the PyInstaller `--add-data`
target the build script uses, and the installer contract version moved to
`2026.07.27`.

As with every other entry in this file: no weights were retrained, so none of
this is evidence of a smarter trained checkpoint.

---

## July 2026: Calibrated score fusion, and a negative result

A survey of 2024-2026 retrieval, calibration, and selective-prediction work
produced 24 candidate techniques. This entry records what was built, and — more
usefully — what was measured and rejected.

### The defect, which is real

`rank_response_candidates` fuses about twenty signals with hand-tuned weights.
Roughly half pass through `_normalize_01` (per-batch min-max) and half arrive
raw. Expressing each signal's influence as |weight| x realised spread on a
16-candidate pool:

| Signal | Nominal weight | Actual influence |
|---|---|---|
| `sim_ctx` (raw dot product) | 0.60 | 46.9% |
| `sim_resp` (raw dot product) | 0.25 | 19.5% |
| `bucket_bonus` (raw model score) | 0.18 | 12.8% |
| a `_normalize_01` signal | 0.06 | 12.2% |
| `freq_penalty` (raw `log1p`) | 0.03 | 6.7% |
| `lex_sim` (raw Jaccard) | 0.10 | 1.8% |

`lex_sim` carries three times the nominal weight of `freq_penalty` and a third of
its influence, and a min-max signal at weight 0.06 outweighs it twice over. A
signal's influence is set by its normalisation choice, not its weight. Whatever
the weights were tuned to express, this is not it.

This is motivated by Bruch, Gai and Ingber, "An Analysis of Fusion Functions for
Hybrid Retrieval" (ACM TOIS 2023, arXiv:2210.11934), which argues score
calibration is the first-order step before fusion, and finds min-max
specifically weak because it preserves the underlying distribution's skew.

### What was built

`score_fusion.py` (mirrored to `runtime_python/`), providing four modes:

- `legacy` — the shipped behaviour, **bit-exact**, and the default;
- `gated` — suppress signals whose spread is indistinguishable from noise,
  leaving surviving signals on their original scale;
- `calibrated` — additionally map each signal onto its own percentile rank;
- `consensus` — additionally apply a bounded CombMNZ multiplier rewarding
  candidates that many signals independently support.

One design point is worth stating plainly: **percentile ranking alone is
unsafe.** It is scale-free, so it stretches a signal carrying nothing but
numerical noise onto the same full [0, 1] spread as the signal that decides the
answer. Rank calibration is therefore always paired with a dispersion gate that
zeroes such a signal *before* the transform can amplify it.

### First measurement (synthetic probes, since superseded)

`source/benchmark_ranking_quality.py` scores 31 labelled probes over a 30-item
corpus, reporting top-1, top-3, and MRR with **paired bootstrap confidence
intervals**:

| Mode | top-1 | MRR | Paired delta vs legacy (95% CI) |
|---|---|---|---|
| `legacy` | 61.3% | 0.743 | — |
| `gated` | 61.3% | 0.743 | MRR +0.0000 [+0.0000, +0.0000] |
| `calibrated` | 54.8% | 0.690 | MRR -0.0524 [-0.1421, +0.0325] |
| `consensus` | 54.8% | 0.702 | MRR -0.0405 [-0.1308, +0.0445] |

**No mode differs from legacy beyond sampling noise.** Every interval straddles
zero. Three conclusions, none of them flattering to the original hypothesis:

1. **Rank calibration measured slightly worse, not better.** The premise is
   correct — the weights genuinely do not mean what they say — but the corollary
   is fatal to a drop-in fix: the weights were hand-tuned *against* the
   miscalibrated scales and therefore already compensate for them. Recalibrating
   the signals without re-tuning the weights discards that compensation.
   Calibrated fusion is only a win alongside a weight re-tune, which needs
   labelled data this repository does not have.
2. **`gated` is bit-identical to `legacy` on every probe.** The dispersion gate
   never fires, because no signal in this fixture is noise-only. It is a safety
   net against a degenerate pool, not an improvement.
3. **Vector-PRF was prototyped and rejected.** A Rocchio query-expansion sweep
   (Li et al., TOIS, arXiv:2108.11044) over alpha in [0.5, 0.9] and k in {2,3,5}
   produced a best case of +0.0150 MRR at alpha=0.7, k=3 — but -0.0241 at
   alpha=0.8, k=3. Selecting the winning cell from 23 probes is fitting the
   evaluation, not the problem. It was not merged.

### What is actually delivered

The honest deliverable here is the **measurement**, not the ranking change.
Before this, every weight in `rank_response_candidates` was a judgement call that
nothing measured, and there was no way to tell whether any change to it helped.
The bootstrap intervals are the load-bearing part: the probe set is small enough
that a single query moves top-1 by more than three percentage points, which is
easy to misread as a result.

`legacy` remains the default and is bit-exact however the mode is spelled,
including for unknown values. The other modes exist so that a future weight
re-tune can be evaluated rather than asserted.

The probe set is a hand-written regression fixture, not a validation corpus. A
win on it is evidence that a change did something, not evidence that the system
is smarter. As with every other entry in this file: no weights were retrained,
so none of this is evidence of a smarter trained checkpoint.

### Second measurement, on the real corpus: conclusive rejection

The synthetic result above was inconclusive because the fixture was too weak.
`llm_chat.db` already held the labels: 120,000 rows pairing a `user_text` with
the `response_text` stored for it, of which 57,934 have a `user_text` occurring
exactly once and are therefore unambiguous (query, gold) pairs.

`source/build_ranking_eval_set.py` extracts a reproducible fixture from them.
Two decisions made it useful:

- **Hard negatives.** With randomly drawn distractors the reranker scores 100%
  top-1, because an unrelated response is trivially dissimilar; that measures
  nothing. Drawing distractors from the query's nearest neighbours in context
  space drops legacy to 87.0%, leaving real headroom.
- **Production vector semantics.** A first version featurized the response text
  for both `vec` and `ctx_vec` and reported legacy at 1.0% top-1. That was a bug
  in the harness, not a finding: production scores a candidate's stored
  *context* vector against the live query, and `sim_ctx` is the dominant signal.
  The fixture now carries the query each response was stored under.

On 200 probes with 39 hard negatives each:

| Mode | top-1 | MRR | Paired delta vs legacy (95% CI) |
|---|---|---|---|
| `legacy` | 87.0% | 0.916 | — |
| `gated` | 87.0% | 0.916 | +0.0000 [+0.0000, +0.0000] |
| `calibrated` | 36.0% | 0.546 | **-0.5100 [-0.5800, -0.4400]** |
| `consensus` | 28.5% | 0.474 | **-0.5850 [-0.6600, -0.5100]** |

Rank calibration costs **51 percentage points of top-1 accuracy**, far outside
sampling noise.

### Why, and why no weight re-tune can rescue it

The earlier entry speculated that calibration might win alongside a weight
re-tune. That is now refuted, and the reason is more interesting than the
result.

Percentile ranking is scale-free, which is exactly the property that makes it
destroy this task. When the gold response sits at a context similarity of 0.95
and the distractors sit near 0.1, the *margin* is the information that
identifies the answer. The transform replaces it with uniform rank steps: the
gold becomes 1.00 and the runner-up 0.97, regardless of whether the true gap was
enormous or negligible.

No choice of weights recovers that, because the discarded quantity is no longer
present in any signal. This is an information-destruction problem, not a tuning
problem.

The hand-tuned weights are therefore not naive. They exploit raw magnitudes that
carry real signal, and the weight/influence mismatch documented above is the
price of that — a genuine, measured trade rather than an oversight.

Bruch, Gai and Ingber's calibration argument still holds in its own setting:
fusing heterogeneous rankers over results of *comparable* quality. It does not
transfer to a retrieval task where one candidate dominates by a wide margin.

### What changed as a result

`calibrated` and `consensus` are no longer reachable from the runtime path.
`resolve_fusion_mode` accepts only `RUNTIME_FUSION_MODES` unless a caller passes
`allow_experimental=True`, which only the benchmark does. An unknown mode, a
config typo, or a stale setting degrades to `legacy` rather than to a mode that
halves retrieval quality. They remain in the tree as the evidence trail for why
rank calibration was rejected.

`legacy` remains the default and is bit-exact. `gated` stays selectable and is
bit-identical to `legacy` on every probe measured; it is a safety net for a
degenerate pool, not an improvement.

The lasting deliverable is the pair of tools: a corpus-derived, hard-negative
evaluation set and a benchmark that reports paired bootstrap intervals. Before
them, every weight in `rank_response_candidates` was a judgement call that
nothing measured, and a plausible, well-cited improvement could be adopted
without anyone noticing it cost half the retrieval accuracy.

---

## July 2026: Measuring the reranker, and three rejected changes

`rank_response_candidates` fuses about twenty signals with hand-tuned weights.
Every one of those weights was a judgement call that nothing measured. This entry
records building the measurement, and the three changes it rejected.

### The evaluation

`llm_chat.db` already held the labels: 120,000 rows pairing a query with the
response stored for it, of which 57,934 have a query appearing exactly once and
so carry an unambiguous gold. `source/build_ranking_eval_set.py` turns those into
a fixture, and two decisions determine whether the result means anything.

**Hard negatives are mandatory.** With randomly drawn distractors the reranker
scores 100% top-1, because an unrelated response is trivially dissimilar. That
number cannot move, so it measures nothing. Drawing distractors from the query's
nearest neighbours drops it to 75.8% and leaves headroom a change can move.

**The fixture must mirror production's vector semantics.** Production's dominant
signal is `sim_ctx`, the live query against the *context a candidate was stored
under*. A first version featurized response text for both vectors and measured
legacy at 1.0% top-1 instead of 87.0%. The fixture now carries each candidate's
source query. `test_ranking_eval_harness.py` pins this, because the failure mode
is silent: the benchmark still runs and still prints confident numbers.

**Scoring must credit equivalent answers.** The corpus stores the same answer
many ways: "Okay. Hello. Tell me what you need." beside "Hello. Tell me what you
need. Let me know if you want a deeper walkthrough." Only one is labelled. On a
300-probe dev split, 88 queries missed top-1 and **51 of them (58%) retrieved a
near-duplicate of the gold**. Strict scoring reported 70.7% where
equivalence-aware scoring reports 87.7%. Chasing that 17-point gap would optimise
a labelling artefact and reward reshuffling equivalent responses.

### Rejected: rank calibration of the fusion signals

Percentile-rank normalisation before fusion, motivated by rank-fusion results in
retrieval. Measured on 200 corpus probes with hard negatives:

| mode | top-1 | paired difference vs legacy |
|---|---|---|
| legacy | 93.0% | — |
| calibrated | 69.5% | **-23.5 points** [-29.5, -17.0] |
| consensus | 66.5% | **-26.5 points** [-33.5, -19.5] |

Under strict scoring the gap is wider still, -51 and -58 points.

The mechanism is now clear and worth recording: **rank transforms destroy
margin**. When the gold sits at similarity 0.95 and the distractors at 0.1,
percentile ranking flattens that decisive gap into uniform steps. The hand-tuned
weights are not naive about scale; they exploit magnitudes that carry real
signal. No weight re-tune can recover this, because the information is discarded
before the weights apply.

`calibrated` and `consensus` remain in the tree as the evidence trail but are no
longer runtime-selectable. `resolve_fusion_mode` accepts them only under
`allow_experimental=True`, which the benchmark passes and no runtime path does.
A stale config or a typo degrades to `legacy`, never to a mode that would halve
retrieval quality.

### Rejected: length normalisation

Among all top-1 failures the winning candidate was longer than the gold 78% of
the time, which looked like an obvious length bias. It was not.

The featurizer L2-normalises (`_featurize_text_impl`), so cosine similarity
carries no length advantage, and lexical overlap is Jaccard, which *penalises*
length through the union term. Restricting the measurement to the 34 genuine
failures, the winner was longer in 22 of them: 65% +/- 17pp, **not
distinguishable from chance**. The 78% was produced almost entirely by the
near-duplicate artefact, where the winner is the gold plus boilerplate and is
longer by construction (87% there).

No change was made. The hypothesis was an artefact of measuring on the wrong
subset.

### Rejected: a content-focus query view

Genuine failures skew towards queries padded with conversational filler ("can you
how to speed up python loop thanks? when you can?"), where generic and greeting
responses beat specific ones. `_query_view_texts` already builds multiple query
views, but every one of them *appends* to the full query; none strips the filler.
Adding a parameter-free view of the query's salient terms, tried at three
positions in the view list:

| split | baseline top-1 | with content view |
|---|---|---|
| dev (300) | 88.7% | 86.3%, difference [-0.050, +0.000] |
| held-out test (300) | 87.0% | 87.0%, difference [-0.027, +0.027] |

It did not help on dev and did nothing on test. Not shipped.

### What this leaves

Three interventions measured, three rejected, no ranking change shipped. The
result worth keeping is that the hand-tuned reranker is hard to beat with generic
interventions, and that there is now a harness able to say so with a confidence
interval instead of an opinion.

The remaining headroom is real but small: roughly 11% of queries retrieve a
genuinely wrong answer, concentrated in filler-padded compound requests. Closing
it likely needs the classifier's `bucket_score`, which this fixture deliberately
holds constant, or labelled preference data — not another reweighting.

As with every other entry in this file: no weights were retrained, so none of
this is evidence of a smarter trained checkpoint.

---

## July 2026: Closing the bucket_score gap, and two more rejected changes

The previous entry ended by saying the remaining headroom "likely needs the
classifier's `bucket_score`, which this fixture deliberately holds constant."
That was the right thing to chase, and chasing it produced two results: one
correction to the harness, and one more rejected change.

### The harness was measuring the reranker without its prior

`rank_response_candidates` weights `bucket_score` at 0.14-0.18 as `bucket_bonus`.
Pinning it to a constant makes that term identical across candidates, so it drops
out of the ranking entirely. The fixture was measuring the reranker with one of
its inputs disabled.

Production does not supply a constant. `llm_database.fetch_candidates` hands every
DB candidate

    bucket_score = sigmoid(0.72*sim_ctx + 0.28*sim_resp + 0.04*log1p(count)
                           - 0.03*bm25_penalty + exact_bonus)

The benchmark now replicates that first stage. `bm25_penalty` and `exact_bonus`
are omitted — the fixture has no FTS ranks, and a hard-negative set contains no
exact user-text matches — so this is an approximation of the first stage, not a
reproduction of it. `--constant-bucket-score` restores the old behaviour.

**The correction changed nothing: 93.0% top-1 either way, MRR 0.961 to 0.962.**
The reason is worth recording. That first-stage score is built from the same two
cosine similarities the reranker already weights, so it is very nearly a monotone
function of signals already present. It is redundant, not informative. The
harness is now production-faithful on this path, and the faithfulness turned out
not to matter.

### The classifier prior is genuinely informative

Unlike the first-stage score, the trained classifier carries independent
information. Measured against the v50 checkpoint over its 10 k-means buckets:

| | in-sample (meta exemplars) | out-of-sample (held-out corpus rows) |
|---|---|---|
| bucket top-1 | 78.5% | **80.7%** |
| bucket top-3 | 99.5% | 98.5% |
| chance | 10.0% | 10.0% |
| confidence separation (correct - wrong) | +0.332 | +0.145 |

Out-of-sample labels are nearest-centroid assignments over centroids recovered
from the bucket exemplars — the same rule k-means used, but a proxy for the true
training labels, so treat the absolute number as approximate. The conclusion is
robust regardless: the classifier is far better than chance and its confidence
separates hits from misses.

### Rejected: using the classifier prior for DB candidates

Given a prior that informative, the obvious move is to give DB candidates the
classifier's bucket probability instead of the redundant first-stage score.
Measured with a selection/held-out split:

| prior | dev top-1 | difference vs production |
|---|---|---|
| first-stage (production) | 89.3% | — |
| classifier probability | 84.0% | **-5.3 points** [-8.3, -2.7] |
| 50/50 blend | 87.3% | **-2.0 points** [-4.0, -0.3] |

The blend was best on dev and was then neutral on the held-out split
(87.0% vs 87.0%, difference [-0.017, +0.017]). Nothing was shipped.

The mechanism is the useful part. A 10-way cluster probability assigns **one
score to every candidate in a bucket**. In a hard-negative pool the candidates
are near neighbours by construction, so they cluster together: median 4 distinct
buckets across a 40-candidate pool, and a median 64% of distractors sitting in
the gold's own bucket. The prior therefore hands most of the pool an identical
value, discriminating nothing, while displacing a first-stage score that is at
least fine-grained per candidate.

This is a limit of the cluster hypothesis rather than of this classifier. A
cluster prior is useful for deciding **which buckets to pool candidates from**,
which is exactly what production already uses it for on the model-bucket path.
It cannot rerank *within* a pool drawn from one cluster.

### Running tally

Five interventions measured, five rejected: rank calibration, length
normalisation, a content-focus query view, the classifier prior as a replacement,
and as a blend. No ranking change has survived measurement.

The reranker's hand-tuned weights are more robust than they look, and the harness
is now faithful enough to say so on the DB path. The genuinely unmeasured
remainder is the model-bucket candidate path, where the classifier selects which
buckets to pool from before any reranking happens; this fixture contains only
DB-style candidates and cannot see that decision.

As with every other entry in this file: no weights were retrained, so none of
this is evidence of a smarter trained checkpoint.

---

## July 2026: A literature scan, and a hubness hypothesis that did not survive

A search of recent retrieval work returned 13 techniques with citations. Four of
them were variations on one hypothesis, and that hypothesis was testable before
any of them was implemented.

### The hypothesis

**Hubness** (Radovanovic, Nanopoulos & Ivanovic, *Hubs in Space*, JMLR 11, 2010;
Feldbauer & Flexer, *KAIS* 2019). In high-dimensional spaces, points near the
data centroid acquire disproportionate k-occurrence: they appear in the k
nearest neighbours of many unrelated queries. The published diagnostic is the
skewness of the k-occurrence distribution, with S_k >= 1.4 flagged as
problematic.

This is exactly the predicted signature of the residual failure mode here.
A greeting or generic response sits near the centroid of conversational n-gram
space, and a filler-padded query is dragged toward that same centroid, so the
generic response wins. Four of the scanned techniques are corrections for it:
QB-Norm / Dynamic Inverted Softmax (CVPR 2022), Dual Bank Sinkhorn Normalisation
(arXiv:2508.02538), All-but-the-Top (ICLR 2018), and the MMI anti-prior
(Li et al., NAACL 2016).

Usefully, the scan itself proposed the diagnostic *before* the fixes, and flagged
the right caveat: global hubness need not imply hubness inside the small
candidate set the ranker actually sees.

### The corpus is severely hub-dominated

| | k-occurrence skewness S_10 | max O_10 | mean O_10 |
|---|---|---|---|
| 8,000-row corpus sample | **14.199** | 1,989 | 10.0 |
| the fixture's 4,634-response candidate set | **13.219** | 5,686 | 43.2 |

Both are an order of magnitude past the 1.4 threshold. One response is a top-10
neighbour for 1,989 of 8,000 queries. The phenomenon is unambiguously present.

### It does not explain the failures

Measured over every genuine failure in the fixture, with the full response set
covered so the test has power (n = 14, all with hubness data):

- winning distractor's hubness percentile: **median 0**
- gold's hubness percentile: median 0
- winners above the 90th hubness percentile: **0 of 14**
- winner more hub-like than the gold it displaced: **2 of 14**

The hypothesis predicts winners should be *more* hub-like than the golds they
beat. They are significantly *less*. The failures happen among low-hubness
responses at the bottom of the k-occurrence distribution, and no hub is anywhere
near them.

All four hubness corrections are therefore rejected without implementing any of
them. QB-Norm and DBSN would have been the most tempting, since both apply an
additive per-candidate offset in log space and so avoid the margin destruction
that killed rank calibration — they were the right *shape* of intervention for
this system. They are simply aimed at a problem this evaluation does not have.

### What this does not establish

The hard negatives are drawn by nearest-neighbour search, which plausibly
suppresses hubs in the pool by construction: a tight near-duplicate cluster has
little k-occurrence spread. So this rules hubness out **for the failures this
fixture can see**, not for production, where the first stage pulls from the whole
120k corpus and a global hub with O_10 = 1,989 can reach the candidate set by a
route the fixture never exercises. Testing that needs a fixture built from
first-stage retrieval over the full corpus rather than from sampled negatives.
That is a real gap, and this entry does not close it.

### Running tally

Nine interventions considered, nine rejected. Five were measured end to end
(rank calibration, length normalisation, a content-focus query view, the
classifier prior as replacement and as blend) and four were ruled out by a
diagnostic that cost one offline pass instead of four implementations.

The diagnostic-before-fix pattern has now paid for itself twice: once when the
apparent 78% length bias turned out to be a near-duplicate artefact, and again
here. Both times the intervention would have been built against a cause that was
not there.

As with every other entry in this file: no weights were retrained, so none of
this is evidence of a smarter trained checkpoint.

---

## July 2026: Conversation quality, measured — and the first change that survived

Every measurement in this file so far has been single-turn retrieval. The
conversation layer in `conversation_state.py` was built, wired into ranking, and
never checked. This entry checks it, and the check found the layer inert.

### The harness

`source/benchmark_conversation_quality.py`. It cannot be built from the corpus:
every `context_text` in `llm_chat.db` carries exactly one turn marker, so no
real multi-turn material exists. The 19 cases are therefore **constructed**, and
that is a genuine limit — they are biased towards failures the author could
imagine, and a pass rate here is a contract check, not an estimate of behaviour
in use.

What makes it useful anyway is that trap kinds are *separable*. Each case pairs
a good continuation against one trap that is wrong for a purely conversational
reason, so a change can be shown to fix one behaviour without disturbing another,
which single-turn top-1 cannot express.

### Three of four behaviours were already handled

| trap kind | without state | with state (before the fix) |
|---|---|---|
| repetition | 4/4 | 4/4 |
| re-asked question | 4/4 | 4/4 |
| topic drift | 4/4 | 4/4 |
| **stated style preference** | **1/4** | **1/4** |

Repetition, re-asking, and drift are already covered by the existing
`-0.27 * sim_recent` penalty. The conversation layer added nothing to any of
them, and nothing to the one behaviour that was failing: overall 81.2% with the
layer on and 81.2% with it off.

### The layer was solving the right problem in the wrong place

`conversation_state` detected `style_request = "concise"` correctly. It then
applied it as a bounded score nudge worth at most `0.3 * 0.12 = 0.036`, far too
small to reorder anything against a verbose candidate winning on other signals.

Meanwhile the ranker already had a fully weighted `concise` style mode. The mode
just never arrived: `infer_style_mode` reads the *current turn only*, so for
"how do I list files in python" it returned `analyst`, and the concise signal
never activated. A preference stated two turns earlier was detected, stored, and
then discarded at the moment it mattered.

The fix routes the standing preference into the mode that already exists instead
of adding weight to the score. It introduces no tuned constant — the concise
mode's weights were already there and independently set.

Two supporting gaps were closed with it:

- **Bare imperatives were never recorded.** "be brief" matches none of the cue
  patterns: no "I prefer", no "always", no negation. A `_STYLE_DIRECTIVE_RE`
  now records them, so the most natural phrasing of the preference works.
- **A standing preference must yield to a fresh request.** Without a guard, "be
  brief" on turn one would override "explain that in detail" on turn two, which
  is worse than having no memory at all. `DETAIL_REQUEST_RE` on the current turn
  suppresses the standing preference. Three cases now pin this.

### Result

| | without state | with state |
|---|---|---|
| stated style preference | 1/4 | **4/4** |
| standing yields to fresh request | 3/3 | 3/3 |
| overall (19 cases) | 84.2% | **100%** |

Single-turn retrieval is unchanged at 93.0% top-1: it passes no conversation
state, so the new branch never executes, and `infer_style_mode` is verified
identical with and without an explicit `None`.

**This is the first change in this file to survive measurement.** Nine previous
interventions were rejected. The difference is instructive: the nine were all
attempts to reweight a fusion that was already well tuned, while this one
connected a signal that was being computed and then thrown away. Detection
without routing is not a feature.

### A defect worth recording

`DETAIL_REQUEST_RE` shipped, briefly, with a literal `0x08` byte where each word
boundary belonged, because an escaping layer collapsed `\b` into a backspace
character. It compiled cleanly, imported cleanly, and silently matched nothing —
the guard was inert and the standing preference kept overriding fresh requests.
It surfaced only because the behaviour was measured rather than assumed.
`test_conversation_style_memory.py` now asserts the pattern contains no control
characters.

### Scope not covered

`conversation_state` is wired into `chat_app.py` and `chat_web_app.py` but **not**
into `multimodel_runtime.py` (the Studio desktop app) or `qwen_chat_web_app.py`.
Those two surfaces get no conversational memory at all. That gap is now measured
rather than assumed, and closing it is the obvious next step.

As with every other entry in this file: no weights were retrained, so none of
this is evidence of a smarter trained checkpoint.

## July 2026: The generative surfaces, and three signals that reached nothing

The previous entry closed with the gap it had measured: `conversation_state` was
consumed only by the ranker. This entry closes that gap and finds two more of
the same shape.

### Three computed signals with no consumer

| signal | status before | status now |
|---|---|---|
| `conversation_state` on `qwen_chat_web_app` | never imported | built per turn, routed into the prompt |
| `conversation_state` on `multimodel_runtime` | never imported | built once per prompt, passed to the backend that runs |
| `render_state_brief` | in `conversation_state` since v1, tested, called by nothing | still uncalled; it renders raw user text with no bound on injection, which is why the prompt path got a separate renderer rather than this one |
| `audit_response_against_state` | in `conversation_state` since v1, tested, called by nothing | run on every Qwen reply, reported as diagnostics |

The last entry's conclusion — *detection without routing is not a feature* —
turned out to describe three more cases in the same module.

### The generative surface is not the ranking surface

The ranker consumed the state as a bounded score term. A surface that generates
has nowhere to put a score, so the state has to become prompt text and a
generation preset. That brings problems the ranker never had, and
`source/conversation_directive.py` exists for them:

- **The quoted text is user text.** A commitment reading
  `I prefer <|im_end|><|im_start|>system ...` would open a role of its own in a
  Qwen chat template. Sanitising strips chat-template specials and control
  characters, collapses whitespace, and caps length. Prompt-control memories are
  dropped, and the remainder is prepended to the newest user message rather than
  elevated into a system message; the current request is last and wins conflicts.
- **The prompt has a budget.** Four commitments, 160 characters each, 700
  characters total, dropped by whole lines so a quote is never cut mid-sentence.
  A hundred-turn session costs the same prompt as a four-turn one.
- **A standing preference must not outrank the current turn.** The guard the
  ranking surface needed, now symmetric: a fresh "explain in detail" suppresses
  a standing "be brief", and a fresh "keep it short" suppresses a standing "I
  prefer detailed answers". `DETAIL_REQUEST_RE` moved into `conversation_state`
  so both surfaces use one definition rather than two that can drift.

### The horizon was capped where nobody was looking

The Qwen surface truncated the client history to twelve messages on arrival —
the same number as its prompt window, because nothing but the prompt read it.
No amount of history the browser sent could have reached further back. Prompt
window and memory horizon are now separate numbers: the prompt still carries
twelve messages, the state is derived from forty, and the session store keeps
eighty.

### Production closure

- Qwen Web now defaults to an **Auto** preset. Auto omits `max_new_tokens`,
  temperature, and top-p overrides so a standing conversation style can supply
  its preset; without one, the balanced defaults remain the fallback. Selecting
  a direct preset re-enables the controls.
- A fixed set of striped per-session locks covers the complete history snapshot,
  directive, generation, and append transaction. Client history can initialise an
  empty server session, but a stale tab cannot replace server-authoritative history
  after the session exists.
- Explicitly transient requests such as "this time" and "for this reply" are not
  promoted into durable commitments. Response auditing also receives the current
  request, so a fresh style instruction does not become a false standing-style
  violation.
- The root and `source/` Qwen EXE build scripts plus both tracked
  `SupermixQwenDesktop*.spec` files bundle `conversation_state.py` and
  `conversation_directive.py`; manifest tests pin those dependencies.

### Persistent Studio memory lifecycle v2

`supermix-conversation-memory-v2` separates a generated reply from verified
memory. Assistant responses remain available in the bounded turn log but are no
longer auto-promoted as "Successful answer pattern" lessons, and unconfirmed
legacy assistant lessons are excluded from prompts. Explicit preferred-name and
answer-detail fields receive stable IDs plus active/superseded metadata, so a new
value deterministically retires the prior one without deleting legacy evidence.
Retrieval suppresses an older slot on the same turn that changes it and excludes
zero-overlap memories except the two narrowly global slots. Existing JSON remains
readable and acquires lifecycle metadata in place. Both new and legacy memories
are filtered for prompt-control payloads before retrieval; recalled memories and
prior examples are stripped of chat-role tokens and explicitly labelled as
untrusted historical context below the current request's authority.

### Measured

`source/benchmark_conversation_routing.py`, thirteen constructed cases across
seven kinds, run with the layer off and on:

| case kind | layer off | layer on |
|---|---|---|
| standing style preference | 0/3 | **3/3** |
| standing constraint | 0/2 | **2/2** |
| clarification loop | 0/1 | **1/1** |
| fresh request wins | 2/2 | 2/2 |
| explicit choice wins | 1/1 | 1/1 |
| injection inert | 2/2 | 2/2 |
| no state, no change | 2/2 | 2/2 |
| **overall** | **7/13 (53.8%)** | **13/13 (100%)** |

Mean contract cost 208.3 characters, maximum 418.

### What the harness found

`fresh_request_wins` failed on first run for "keep it short this time" against a
standing "I prefer detailed answers". The guard suppressed the *style line*
correctly, and then the same preference walked back into the prompt as a quoted
commitment two lines later. A standing style is now carried by the style line
only, which is the one place that knows about the guard;
`conversation_state.style_preference_of` is published so the two agree on what
counts as a style statement.

### What this is not

The harness measures what reaches the prompt. It never runs the model. A bounded
memory line reading "keep this reply short" is context for a 0.5B adapter, and
whether the adapter obeys it is a separate question that needs generation
against held-out cases. Nothing here establishes that any reply changed — only
that the signal is now present in the prompt rather than discarded before it.

Ranking is untouched: `DETAIL_REQUEST_RE` moved but its pattern is unchanged,
and the exact-replay, robustness and interaction-regression suites are green.
No weights were retrained.

## July 2026: v53 MiMoMix Hybrid (Attention + MoE + MTP + Verified Recursion)

New, additive line in `source/mimomix_*.py`. It fuses the current Xiaomi MiMo
structural techniques, the Supermix v51/v52 verified-recursion cognition, and the
AI-Dem-Lab research-sandbox concepts into one CPU-runnable stack. No existing
v52 module, checkpoint, manifest, or gate changed.

Full design and boundaries: `docs/V53_MIMOMIX_ARCHITECTURE.md`.

### Papers and sources this draws on

Attention and long context:

- [StreamingLLM](https://arxiv.org/abs/2309.17453) and
  [Why do LLMs attend to the first token?](https://arxiv.org/abs/2504.02732) --
  the attention-sink pathology and the learnable-sink fix
- [YaRN](https://arxiv.org/abs/2309.00071) -- per-frequency-band RoPE
  interpolation plus the `0.1*ln(s)+1` attention temperature
- MiMo-V2-Flash technical report ([arXiv 2601.02780](https://arxiv.org/abs/2601.02780))
  and the MiMo-V2.5 / V2.5-Pro model pages -- the 5:1 and 6:1 SWA/global
  interleave at a 128-token window, and the KV-cache consequence

Sparse mixture of experts:

- [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) -- the
  select-by-bias, weight-by-score rule and the sign update
- ST-MoE router z-loss; DeepSeekMoE shared/fine-grained experts
- [Mixture-of-Depths](https://arxiv.org/abs/2404.02258) -- already cited by v52
  for sparse recurrent-core execution

Multi-token prediction and speculative decoding:

- [Better & Faster LLMs via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
- DeepSeek-V3's sequential MTP modules with shared embedding and head
- MiMo's three-layer MTP reused as the draft model for self-speculative decoding

Adaptive test-time compute:

- [Compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314)
- [REFRAIN](https://arxiv.org/abs/2510.10103) -- the "reason just enough"
  direction, already cited by v52.1
- PonderNet / ACT halting, carried over from the v50-v52 cognitive-leap line

Post-training:

- On-policy distillation (student-sampled trajectories, teacher scores each
  token, reverse KL), generalised by MiMo to Multi-Teacher On-Policy
  Distillation
- GRPO and the Dr.GRPO critique of dividing group advantages by their standard
  deviation

These sources motivate the design. They do not validate this implementation.

### What was implemented

- `mimomix_core.py` -- hybrid SWA/global attention with learnable per-head
  sinks and per-layer cache spans, `none`/NTK/YaRN RoPE extension,
  auxiliary-loss-free top-k MoE with router z-loss and shared experts,
  sequential MTP depths with dense FFNs, and a recursive thinking core carrying
  ACT halting, a ponder cost, trainable verifier temperature, and supervised
  `p(correct)`/`p(continue)` heads
- `mimomix_decoding.py` -- MTP self-speculative decoding with exact
  greedy-equivalence, safe cache rollback under sliding-window trimming,
  EOS-safe commits, bounded output, finished-row handling, and post-prefill
  acceptance-length accounting
- `mimomix_controller.py` -- deterministic difficulty and epistemic-risk
  scoring, fast/deep/agent routing with a safety fast path, a bounded budget
  ladder gated on verifier stand-down plus confidence/entropy targets plus
  ordered top-k cross-budget agreement, accepted-probe reuse, and a paid-in-full
  decision-fidelity audit
- `mimomix_observatory.py` -- exact chi-square survival via regularised
  incomplete gamma, monobit and runs tests, min-entropy, JSD, CHSH as a harness
  self-test, evidence with an optional-stopping penalty, novelty/stability/RSI
  meters, semantic resonance, routing attribution, median/MAD anomalies,
  replicator dynamics, and a tabular budget Q-learner
- `mimomix_distill.py` -- group-relative advantages, clipped GRPO surrogate, and
  MOPD with causal next-token target alignment, a finite top-k probability-space
  teacher mixture, and per-token dense reward
- `mimomix_api.py` -- byte tokenizer, backend registry, `/v1/think` with plan-
  driven routing, and an optional Flask surface
- `web_static/mimomix_lab.html` -- single-file browser observatory reimplementing
  the same algorithms in JavaScript

187 tests across six suites. The load-bearing ones assert properties, not
outputs: that the router bias cannot reach the forward value, that the balancer
recovers a deliberately collapsed router, that speculative decoding is
bit-identical to greedy, that the accepted budget's output is reused rather than
blended, that the verifier can veto but never authorise an early exit, and that
the observatory's statistics match textbook critical points.

The tracked 250-step benchmark reports an MTP acceptance length of **3.917**.
That full benchmark was not rerun after the correctness hardening above; the
current evidence is the passing six-suite CI slice and focused decoding,
distillation, and API tests.

### What this is not

The default backends are randomly initialised. The suites prove integration,
gradient flow, and the named invariants; they prove nothing about quality. The
parameter counts, acceptance lengths, cache ratios, and tool-call accuracies
published for MiMo's checkpoints describe those checkpoints -- v53 implements the
same mechanisms five orders of magnitude smaller, and a mechanism transferring
does not mean a result transfers. No weights were trained, no checkpoint was
promoted, and `cycle_reduction` on an untrained model is correctly negative
because the gates refuse every early exit.

## August 2026: Epistemic Conversation and Deliberate Reasoning v2

This upgrade makes the runtime better at identifying what kind of reasoning a
request needs, enforcing an answer contract appropriate to that task, and using
deterministic verified answers where the supported problem grammar is exact. It
does not claim that a checkpoint learned new knowledge.

### Research basis

- [Self-Discover](https://arxiv.org/abs/2402.03620) motivates selecting a compact,
  task-specific reasoning structure rather than applying one generic chain-of-
  thought prompt to every request.
- [T1](https://arxiv.org/abs/2504.04718) and
  [CRITIC](https://arxiv.org/abs/2305.11738) motivate routing mechanically
  checkable claims through an external verifier instead of trusting unsupported
  intrinsic self-correction.
- [Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) motivates
  separating uncertainty about answer meaning from surface wording; it also
  shows why confidence signals do not catch every systematic error.
- [Conformal factuality](https://arxiv.org/abs/2402.10978) motivates progressive
  backoff toward less specific claims when evidence cannot support a precise
  answer. Its statistical guarantee depends on representative calibration data,
  which this deterministic runtime does not claim to possess.
- [Compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314) motivates
  spending bounded extra work on harder instances instead of using the maximum
  reasoning budget on every prompt.
- [MathCheck](https://arxiv.org/abs/2407.08733) motivates testing mathematical
  reasoning with consistency, metamorphic changes, and invalid-problem cases,
  not answer matching alone.
- [MultiChallenge](https://aclanthology.org/2025.findings-acl.958/) and
  [MT-OSC](https://aclanthology.org/2026.findings-acl.1354/) motivate explicit
  multi-turn request tracking and selective conversation condensation rather
  than replaying an ever-growing transcript.

### Runtime design

- Prompt Understanding v2 extracts polarity-aware mathematical, scientific,
  predictive, causal, investigative, conversational, and multi-part facets after
  quoted/code masking and bounded typo recovery.
- Plan-Evaluate v2 maps those facets to conjunctive response checks. Mentioning
  words such as `evidence`, `units`, or `assumptions` is not sufficient on its
  own to pass a scientific, calculation, or forecast contract.
- Grounding consumes the same privacy-safe facets so science, forecast, and causal
  requests recommend evidence without granting the parser routing, tool, compute,
  or correctness authority.
- Conversation State v2 preserves a bounded missed user request, while Conversation
  Directive v2 resurfaces at most one only after an explicit "you missed this"
  repair cue. Stored text remains filtered as untrusted user-level data and is
  absent from diagnostics.
- Deliberate Reasoning v2 adds geometry, finite probability, formula-based physics,
  a deliberately narrow empirical Bernoulli estimator, and bounded ordered
  quantity transitions with exact same-dimension conversion. The transition
  solver requires one explicit initial state and two to four unambiguous clauses,
  then reverses every operation to reconstruct the initial state before it can
  override an answer. Contradictory directions, competing bases, mixed
  dimensions, negative intermediate states, or incomplete plans abstain. Exact
  supported calculations require bounded consistency checks and full-registry consensus;
  checks are labelled independent only when they actually use an independent
  path. Model-conditional estimates never receive answer-replacement authority.

### Epistemic boundary

The empirical prediction path requires explicit independent, constant-probability
assumptions. Its observed-frequency estimate is labelled model-conditional,
uncalibrated, and not a guarantee. Weather, markets, experiments with distribution
shift, and other open-world forecasts never receive deterministic override
authority. Prediction-stable adaptive exits mean stable model decisions across
budgets, not real-world predictive accuracy.

No model weights were retrained. The changes improve parsing, contracts,
verification, conversation repair, and supported deterministic calculations. They
do not establish broader scientific knowledge, general mathematical intelligence,
or calibrated forecasting outside the tested grammars.

### General Intelligence curriculum v2

The mixed curriculum now includes `quantity_transition_reasoning`: two to four
ordered changes that must mix percentage-of-current-state and fixed deltas.
Training and held-out rows use disjoint domains, templates, wording, and operation
orders. Targets are generated with exact decimal state updates, rechecked by the
versioned verifier, and negative tests reject reordered answers, single-step
plans, one-kind-only plans, non-finite values, 100% decreases, and negative
intermediate states. The downstream repair curriculum is versioned with the same
sixteen-family registry so repair sampling cannot silently discard the new family.

[CryptoX](https://arxiv.org/abs/2502.07813) motivates measuring compositional
reasoning across transformations rather than treating isolated arithmetic as
generalization. [Let's Verify Step by Step](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html)
motivates verifier-grounded supervision. Supermix's current verifier certifies
the exact final outcome and curriculum invariants; it does not claim a learned
process-reward model or proof that every natural-language intermediate sentence
is faithful. No checkpoint was retrained or promoted by this data-only change.

## August 2026: Promotion Evidence v3 and trusted Qwen adapter lifecycle

The general-intelligence curriculum made candidate training reproducible, and
the v2 promotion receipt bound a candidate to its benchmark and evaluator. It
still treated aggregate benchmark metrics as the decision evidence. That is too
weak for a small held-out run: an aggregate gain does not show which exact
questions changed, related template variants are not independent observations,
and a mutable detailed-sample file was not part of the receipt.

### Research basis

- [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) recommends
  comparing models on question-level paired differences and accounting for
  clusters of related questions. V3 records the paired transition table, uses
  an exact binary test, and resamples whole template clusters rather than
  pretending every generated variant is independent.
- [LiveBench](https://arxiv.org/abs/2406.19314) motivates objective,
  automatically verifiable scoring that avoids an LLM judge. Supermix therefore
  replays its deterministic verifier from the trusted held-out row instead of
  accepting a candidate-produced correctness label.
- [MathCheck](https://arxiv.org/abs/2407.08733) motivates evaluating robustness
  across related checklist and perturbation variants. Supermix keeps those
  variants visible through `template_id` and treats the template, not each
  surface variation, as the bootstrap sampling unit.

These papers motivate the evaluation design; they do not validate Supermix's
adapter, verifier, or thresholds.

### Content-bound paired evidence

`supermix-qwen-evaluation-v3` records SHA-256 digests for
`base_samples.jsonl`, `tuned_samples.jsonl`, `sample_comparison.jsonl`, and the
canonical `supermix-qwen-paired-evidence-v1` object. Each detailed row is aligned
by a unique, complete sample index to the selected evaluation artifact. Trusted
`example_id`, `template_id`, split group, family, prompt, reference, and verifier
specification come from that evaluation row. The sample contributes only the
generated prediction and measured generation data.

The gate opens only fixed filenames beneath the benchmark directory, checks
their declared and actual hashes, replays `verify_candidate` against the trusted
metadata, recomputes aggregate metrics and per-family outcomes, and requires the
canonical evidence object to match byte-for-byte. Adapter weights, configuration,
benchmark, curriculum, selected eval, evidence files, and evaluator/verifier code
are checked again for change during the gate. The v3 promotion manifest and gate
record all of those issuance-time hashes. Runtime validation rechecks the
current adapter, configuration, gate, manifest, schemas, and production policy;
revalidating external evidence files requires rerunning the gate. The receipt is
local historical provenance, not a continuously witnessed evidence store.

### Paired decision rule

For every held-out item, v3 records one of four transitions: both correct, both
incorrect, tuned-only correct (a win), or base-only correct (a regression). It
then computes:

- an exact one-sided McNemar/binomial tail over discordant pairs, with the
  alternative that tuned accuracy is greater; and
- a deterministic 95% percentile interval for verified-accuracy delta by
  resampling complete `template_id` clusters. The default configuration is
  5,000 resamples with seed 5203 and R7 linear-interpolated percentiles.

The default gate requires `p <= 0.05`, paired regression rate no greater than
0.02, at least five template clusters, and a bootstrap lower 95% bound strictly
above zero. Those checks are additional to the existing aggregate requirements:
at least 20 verified samples, at least +0.05 verified accuracy, tuned accuracy
of at least 0.20, no family regression, loss ratio at most 1.05, token-F1 delta
at least -0.02, at least one verified item per family, and generation-cap rate
at most 0.05. Missing paired evidence, zero discordant evidence, a non-positive
lower bound, a threshold miss, a family regression, a stale hash, or any replay
mismatch is a blocker. A failed run writes an inspectable failed gate, removes a
stale promotion manifest, and does not update the implicit-adapter pointer.

Production eligibility is no longer caller-defined. Policy
`supermix-qwen-production-promotion-policy-v3` pins the 150-row curriculum
holdout digest, deterministic family-balanced selection, decode limits, exact
5,000-resample/seed-5203 cluster bootstrap, and the minimum statistical floors.
The gate reconstructs the selected set from the bound curriculum, requires all
identity fields plus unique `example_id` values, and strictly derives the
comparison JSONL from the sample artifacts. Custom thresholds, alternate
holdouts, decode settings, or bootstrap seeds remain useful for research, but
`--no-write-pointer`/review mode writes no activating promotion manifest.

The inspected v1-holdout 30-item repair candidate improved five net answers:
six wins, one regression, and 23 ties, for a +0.1667 point estimate. The exact one-sided result
is `p = 0.0625`, its paired regression rate is 0.0333, and the deterministic
template-cluster interval is approximately `[-0.0323, 0.3667]`. All three v3
statistical safeguards therefore block promotion. The correct conclusion is
"encouraging but underpowered," not that the adapter has established a
general-intelligence improvement. Curriculum v2 adds a sixteenth compositional
family and pins a 32-item family-balanced production selection, so this older
receipt cannot be reused for the new holdout.

### Studio activation boundary

Studio no longer equates a readable adapter directory with a trusted implicit
adapter. Before constructing the Qwen engine it classifies and attests the
artifact. A content-valid v3 promotion is eligible; a present but invalid gate
or manifest acts as a revocation, and candidate/general-intelligence namespaces
are not implicitly loadable. Receipt-free compatibility is limited to exact
allowlisted historical adapter and configuration hashes; legacy-looking names
alone are untrusted.

The active-backend status and browser display expose `activation_kind`, adapter
hash, promotion/gate schemas, and `base_revision_status`. A promoted adapter
resolved from a Hugging Face cache must match both the receipt's repository and
its exact `snapshots/<revision>` value. Local model copies without that identity
proof are rejected. Studio uses one promotion-validation snapshot and re-hashes
adapter weights and configuration immediately before and after model loading to
detect a concurrent file swap. The generated Studio runtime manifest binds
`qwen_adapter_promotion.py`, the three v3 schema constants, the pinned production
policy ID, and enforcement guards so packaging drift fails its normal manifest
check. This attestation is a fail-closed local content and
provenance check. It is not a digital signature, secure measurement,
hardware-backed attestation, trusted timestamp, or proof that the recorded
generations were produced by an untampered model process.

## August 2026: Formal Deliberation v3 and oracle-grounded Promotion v4

Two representative failures remained after Deliberate Reasoning v2. A compact
multi-hop rule theory was treated as a numeric prompt with no applicable solver,
and exact arithmetic lost its protected deterministic path when a user appended
an otherwise ordinary request to explain, show, or verify the result. The first
gap prevented structured non-numeric deliberation; the second let presentation
wording change correctness authority. V3 addresses both without granting a
language model, a retrieved answer, or generated prose verifier authority.

### Research basis

- [LogicGame](https://aclanthology.org/2025.findings-acl.77/) motivates explicit,
  automatically verifiable intermediate rule execution and planning tasks.
- [RuleArena](https://aclanthology.org/2025.acl-long.27/) motivates separating
  complex rule-guided reasoning from oracle/tool support and measuring both.
- [Dissecting Logical Reasoning in Language Models](https://aclanthology.org/2025.findings-emnlp.926/)
  shows why final-answer accuracy alone is not enough to establish stepwise
  logical soundness and motivates symbolic supervision.
- [SATBench](https://aclanthology.org/2025.emnlp-main.1716/) motivates solver-
  validated evaluation artifacts for logical puzzles rather than self-judged
  correctness.

These results motivate the architecture and tests. They do not validate
Supermix's runtime, curriculum, thresholds, or model quality.

### Bounded runtime semantics

Deliberate Reasoning v3 adds the exact positive-Horn grammar
`Facts: a, b. Rules: a & b -> c; c -> d. Query: d.`. At runtime it accepts at
most 12 opaque atoms, 16 rules, and three antecedents per rule. Facts, rules,
and antecedents are canonicalized, so semantically irrelevant reordering cannot
change the proof or answer. Natural-language predicates, negation, disjunction,
quantifiers, duplicate clauses, malformed sections, and oversized theories
abstain rather than falling through to a permissive parser.

The primary derivation computes the least Horn model by forward closure and
retains one canonical dependency proof. A separate verifier enumerates every
Boolean interpretation within the same finite atom bound, filters the models
that satisfy all facts and implications, and checks whether the query is true
in every satisfying model. An answer is authoritative only when closure and
finite-model semantics agree. `Not entailed` is explicitly open-world: it means
that at least one satisfying model leaves the query false, not that the query is
false outside the supplied theory. Cycles, unseeded chains, missing conjuncts,
permuted rules, maximum bounds, and injected verifier disagreement have direct
tests.

The explicit-arithmetic parser now recognizes only three bounded suffixes:
`Explain your reasoning`, `Show your work`, and `Verify the result`. It emits a
short exact-arithmetic explanation or verification statement while retaining
the previous high-stakes and strict-evidence ordering. Arbitrary prose,
multiple expressions, injection-like text, code, and quoted expressions remain
ineligible for deterministic replacement.

### Curriculum v3 and verifier v2

`supermix-general-intelligence-curriculum-v3` adds `logical_entailment` as its
seventeenth family. Its training and evaluation splits use disjoint atom
vocabularies, graph topologies, template identities, and surface markers. The
curriculum subset is further capped at 10 atoms, four facts, eight rules, and
three premises. Each row stores canonical
`supermix-logical-entailment-ir-v1` JSON and names the
`exhaustive-positive-horn-models-v1` oracle.

`supermix-verifier-v2` does not trust the stored answer. It parses the prompt's
single final grammar statement, requires exact agreement with the separately
stored canonical IR, independently recomputes entailment by exhaustive model
enumeration, and accepts only the exact candidate text `entailed` or
`not entailed`. Legacy verifier-v1 metadata, duplicate JSON keys, answer or IR
tampering, prompt/IR disagreement, and decorated candidate output fail closed.
The repair curriculum is versioned v3 and requires all 17 families, so repair
sampling cannot silently omit the new capability.

For production seed 6201 with 1,200 training and 150 evaluation rows, the
canonical training digest is
`8d072c364ecb970e70aa2a8e86b2d2ffa9505f429111f5e6a23d87d803fadb39`
and the evaluation digest is
`45a84eb8e95f2a687b8c8ab951e8c687948446f0c23266f4550671f3095c7617`.
The deterministic family-balanced promotion selection contains 34 examples,
two from each family.

### Promotion v4 boundary

The v4 evaluation, gate, promotion-manifest, and production-policy schemas bind
verifier v2, curriculum v3, the new evaluation digest, and the 34-row selected
holdout. The gate now replays the current verifier over every curriculum
reference before it accepts the manifest's `all_targets_verified` claim.
Issuance-time code hashes must contain the exact evaluator, policy, and verifier
file sets; the verifier set includes both `verifiable_reasoning.py` and the new
`logical_entailment.py` oracle. Omitting the oracle, adding an unexpected file,
using a v1 verifier, or presenting a v3 receipt invalidates runtime promotion.
The paired McNemar, clustered-bootstrap, family non-regression, generation-cap,
base-revision, content-hash, and no-write-pointer safeguards from Promotion v3
remain conjunctive.

This is a formal, bounded reasoning and evidence-pipeline improvement. It does
not support unrestricted natural-language logic, first-order logic, negation,
disjunction, quantification, defeasible rules, or world knowledge. No adapter or
checkpoint was trained or promoted by this change; all candidates remain
inactive until they pass the fixed v4 production holdout and statistical gate.

## August 2026: Supermix v54 Verified Probabilistic Scenarios

V54 is an additive deterministic-runtime and release-contract advance. It keeps
the v52 model line, the v53 MiMoMix research modules, Deliberate Reasoning v3,
and Qwen Promotion Evidence v4 intact. It does not introduce a checkpoint,
adapter, route policy, or model-training claim.

### Bounded finite-Bernoulli contract

The reasoning source and compatibility mirror expose
`supermix-finite-bernoulli-scenario-v1` through reasoning engine v4. The parser
consumes one complete request and accepts only:

- 1 to 200 explicitly IID trials, or independent trials with an explicitly
  fixed/constant/same success probability;
- IID fair-coin tosses or flips, whose event probability is exactly `1/2`;
- an exact probability written as a reduced or unreduced fraction, bounded
  decimal, or percent; and
- one `exactly`, `at least`, or `at most` count over successes, heads, or tails.

The grammar is deliberately not a general probability-language parser.
Dependent trials, changing or unknown probabilities, without-replacement
sampling, malformed bounds, certainty requests, late corrections, unrelated
trailing instructions, and high-stakes or open-world predictions abstain. The
canonical IR records only schema, model kind, trial count, event relation,
event count, outcome, exact probability numerator/denominator, and whether the
full query was consumed.

### Exact computation and independent verification

The primary path evaluates the required binomial mass or tail with
`math.comb` and `fractions.Fraction`; it never samples and never uses a float to
decide correctness. A structurally different verifier rebuilds the entire
Bernoulli distribution by repeated convolution. A result is verified only when:

- direct binomial evaluation equals the convolution-derived event mass;
- every reconstructed mass is non-negative;
- all `n + 1` masses sum exactly to one; and
- the event and complementary event sum exactly to one.

The result remains explicitly model-conditional even when it is authoritative
for the supplied mathematical scenario. It does not establish that the stated
probability or independence assumptions hold in the world.

### Defense in depth and release integration

`grounding_runtime.py` reparses the original raw request through the loaded
reasoning engine before accepting the `finite_binomial_event_probability`
method. A stale, replaced, or fabricated reasoning result therefore cannot gain
rewrite authority merely by claiming the method name. Normal exhaustive solver
consensus, strict-evidence precedence, and the source/runtime opt-out controls
remain unchanged.

The Studio distribution contract moves to application version `54.0.0`, binds
the finite-Bernoulli schema from both reasoning mirrors, and declares that the
capability has no open-world authority. The Windows installer uses the same
version. Its build helper resolves the effective output basename before
compilation and hashes that exact newly built installer, preventing a custom v54
name from accidentally hashing a stale default-name artifact.

The checked runtime manifest is regenerated only after production source/runtime
parity is final. Release readiness still requires the manifest and model-snapshot
checks, focused and full tests, PowerShell parsing, live source and packaged
probes, frozen executable inspection, installer upgrade/uninstall verification,
and independently recomputed SHA-256 files. No EXE or installer was produced by
this documentation and release-contract change.

## August 2026: Supermix v55 memory authority and verified receipts

V55 addresses two trust-boundary gaps without changing model weights or granting
new execution authority. First, persistent memory could sanitize recalled text
yet still lose immutable speaker provenance, treat arbitrary direct-user facts
as ordinary context, and reuse generated assistant exemplars. Second, verified
reasoning metadata reached Studio internally but was not expressed as one
privacy-safe cross-surface receipt.

### Memory-poisoning research basis

[AgentPoison (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html),
[MINJA (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42a97bbd9844d2bf68596730af80bcdf-Abstract-Conference.html),
and [PoisonedRAG (USENIX Security 2025)](https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag)
motivate persistent-origin tracking, authority separation, and retrieval-time
attack evaluation. [Task Shield (ACL 2025)](https://aclanthology.org/2025.acl-long.1435/)
and [CaMeL](https://arxiv.org/abs/2503.18813) motivate keeping task and capability
authorization outside untrusted text. [LongMemEval](https://arxiv.org/abs/2410.10813)
motivates preserving benign long-term personalization utility while hardening
recall. These papers inform the design; they do not validate this implementation.

### Implemented boundary

`supermix-conversation-memory-v3` binds newly extracted memory to the
`supermix-memory-authority-v1` schema and
`supermix-memory-authority-firewall-v1` policy. Only direct-user rows can be
eligible. Identity and answer-detail style may personalize responses; projects
and facts remain attributed and unverified outside the shared planner/tool
prompt. Relevance cannot elevate authority. All memory is denied evidence,
grounding, route, compute, tool, permission, safety, and solver authority.
Legacy, assistant, tool, consultant, malformed-role, quoted, fenced, encoded,
prompt-control, and digest-mismatched rows fail closed. Assistant exemplars are
not reinjected. Exact-ID review supports confirmation, quarantine, terminal
revocation, and conflict-safe restoration of quarantined rows through no-store,
loopback-only local Studio controls. Revocation retains an inspectable audit row;
only a fresh direct-user restatement can reissue it.

`supermix-verified-answer-receipt-v1` is built from allowlisted grounding and
reasoning diagnostics. It contains no prompt, answer, expression, proof step, or
evidence text. It reports verification, independent checking, conflicts,
selection/abstention, strict-evidence/high-stakes precedence, and explicit
model-conditional/no-calibration state. Every compute, route, interaction, tool,
permission, safety, and promotion authority field is false.

The Studio manifest moves to `55.0.0`, binds both new contracts, and enforces the
recursive local-import closure of every packaged entry point. This establishes a
stronger deterministic source/package contract, not digital signatures, secure
storage, trusted execution, external fact verification, unrestricted reasoning,
or a newly promoted model.

## August 2026: Supermix v71 verified scientific scenarios

V71 implements a bounded tool-verified scientific-plan path instead of making a
new model-training claim. Read-only probes showed that the existing reasoning
engine handled canonical Newtonian force but rejected fully specified
constant-acceleration and ideal-gas scenarios. The new path covers that measured
gap with one strict, non-executable plan schema and a versioned local registry.

[SciAgent](https://openreview.net/forum?id=N48b6pzMJc),
[T1](https://arxiv.org/abs/2504.04718), and
[neuro-symbolic verifier-feedback work](https://arxiv.org/abs/2505.14479)
motivate tool-first reasoning and deterministic feedback. Work on
[compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314) and
[the limits of resampling with imperfect verifiers](https://openreview.net/forum?id=j8H84v6AZ1)
motivates the bounded one-plan policy: this implementation never samples until
something passes. [SciBench](https://proceedings.mlr.press/v235/wang24z.html)
supports reporting narrow task coverage rather than a universal prompting or
science claim. These papers guide the architecture; their results are not
reproduced here.

`supermix-science-plan-v1` accepts only one explicitly assumed constant-
acceleration or ideal-gas scenario, one supported target, and all labelled
quantities with supported units. It binds prompt spans by digest, normalises in
exact SI arithmetic, and requires registry, plan, binding, dimensional, domain,
and substitution checks. Receipts contain only allowlisted identifiers, hashes,
counts, pass bits, epistemic limits, and false authority flags. Raw text, answers,
and proof traces are excluded.

The same release also distinguishes forecast-shaped language from a bounded
estimate. Irrelevant assumptions no longer satisfy the prediction contract,
domain-local `same success probability` no longer resolves against unrelated
conversation history, and a verified empirical rate may protect the final text
only with explicit model-conditional, non-guarantee, and uncalibrated wording.

The Studio source/runtime contract moves to `71.0.0` and packages the science
module in Studio and Qwen build surfaces. This is not evidence that either
physical assumption holds in the world, an independent empirical validation, a
general formula solver, a high-stakes engineering tool, or a rebuilt/signed
Windows release.
