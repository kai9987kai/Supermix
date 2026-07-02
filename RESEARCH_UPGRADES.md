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

## March 2026: Sample-Level Benchmark Traces and Research Board Focus

Repo changes:

- Benchmark runs now save `base_samples.jsonl`, `tuned_samples.jsonl`, and `sample_comparison.jsonl` alongside `benchmark_results.json`.
- `benchmark_results.json` now also includes `artifacts` pointers and a compact `sample_summary.worst_regression` block so tools do not need to rescan the full comparison file to show the top failure.
- The training monitor research board now surfaces the selected run's top regression, prompt preview, and tuned/reference preview, and adds a direct `Open selected samples` action.
- Older aggregate-only benchmark artifacts remain readable; the monitor now explicitly reports when a run needs a rerun to generate detailed sample traces instead of showing a blank state.
