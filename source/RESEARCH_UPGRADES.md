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

## July 2026: v51 prediction-stability pilot and distribution-drift shadow metric

Recent primary sources reviewed:

1. LESS Is More: Adaptive Early Exit for Diffusion Language Models
   https://arxiv.org/abs/2606.16908
2. Stop When Reasoning Converges: Adaptive Test-Time Scaling with PUMA
   https://arxiv.org/abs/2605.17672
3. LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing
   https://arxiv.org/abs/2601.07206
4. TwinRouterBench: A Systematic Benchmark for Dynamic LLM Routing
   https://arxiv.org/abs/2605.18859

Implementation and pilot evidence:

- `benchmark_v51_prediction_stability.py` now records top-k Jensen-Shannon
  divergence between consecutive full-prefix output distributions. The shared
  top-k support is chosen from the midpoint distribution and all remaining
  probability is retained in one `other` bucket. This is diagnostic telemetry
  only: it cannot trigger an exit or change a model answer.
- A CPU pilot screened five stopping configurations over three unseen seeds and
  32 examples per seed (480 requests total). Patience 2 / tolerance 0.005 kept
  96/96 prediction agreement and zero observed accuracy delta while using 2.135
  mean cycles, a 28.8% reduction from the three-cycle baseline. Patience 1
  changed one prediction and is rejected. Stricter patience/tolerance settings
  increased work without improving observed agreement.
- Latency measurements were not counterbalanced, so they are screening evidence
  rather than a release claim. The next gate is 512 held-out examples over eight
  fresh seeds with zero disagreements, no negative per-seed accuracy delta, at
  least 20% mean cycle reduction, and positive median latency reduction before
  changing any runtime default.
- The research transfer remains provisional. LESS studies diffusion language
  models, PUMA combines semantic convergence with answer verification, and the
  router benchmarks emphasize held-out dynamic evaluation plus strong simple
  baselines. Supermix therefore exposes distribution drift as a shadow metric
  beside its existing output-persistence verifier rather than treating a new
  paper or a small pilot as activation evidence.
