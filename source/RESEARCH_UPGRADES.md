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
