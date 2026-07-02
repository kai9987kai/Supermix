param(
  [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
  [string]$OutputDir = "artifacts\qwen_supermix_enhanced_v26_cognitive_activation",
  [string]$Device = "auto"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"

python -u source\qwen_supermix_pipeline.py `
  --data datasets\conversation_data.quality_anchor_v2.jsonl datasets\conversation_data.coding_knowledge_2026_02_19.jsonl datasets\conversation_data.world_events_2026_02_19.jsonl datasets\conversation_data.supermix_plus_v27_500k.jsonl datasets\conversation_data.mega_reasoning_creative_v25_75582.jsonl datasets\conversation_data.mega_creative_250k_v2.jsonl `
  --base_model "$BaseModel" `
  --output_dir "$OutputDir" `
  --max_records 48000 `
  --max_source_fraction 0.52 `
  --max_synthetic_fraction 0.10 `
  --max_prompt_signature_count 5 `
  --data_log_every_records 2000 `
  --prompt_signature_cap_exempt_sources "conversation_data.quality_anchor_v2.jsonl,conversation_data.mega_reasoning_creative_v25_75582.jsonl" `
  --eval_size 500 `
  --max_length 448 `
  --batch_size 1 `
  --grad_accum_steps 16 `
  --epochs 6 `
  --max_steps 520 `
  --lr 1.0e-5 `
  --sft_lr_schedule cosine `
  --sft_warmup_steps 30 `
  --sft_min_lr_ratio 0.22 `
  --sft_max_grad_norm 0.9 `
  --train_log_every_steps 1 `
  --save_every_steps 20 `
  --weight_decay 0.02 `
  --lora_r 32 `
  --lora_alpha 64 `
  --lora_dropout 0.03 `
  --use_rslora `
  --use_dora `
  --lora_init true `
  --sft_weight_mode quality `
  --sft_min_weight 0.62 `
  --sft_max_weight 1.88 `
  --sft_synthetic_prompt_weight 0.78 `
  --sft_teacher_source_weight 0.92 `
  --sft_quality_anchor_boost 1.14 `
  --sft_coding_boost 1.24 `
  --sft_events_boost 1.08 `
  --sft_reasoning_boost 1.28 `
  --sft_prompt_skill_boost 1.17 `
  --sft_conversation_boost 1.24 `
  --sft_creativity_boost 1.16 `
  --sft_min_quality_score 0.90 `
  --sft_quality_filter_exempt_sources "conversation_data.quality_anchor_v2.jsonl,conversation_data.world_events_2026_02_19.jsonl,conversation_data.mega_reasoning_creative_v25_75582.jsonl" `
  --sft_auto_balance_sources `
  --sft_source_balance_strength 0.66 `
  --sft_source_balance_max_scale 1.95 `
  --preference_objective orpo `
  --preference_steps 150 `
  --preference_pairs 3400 `
  --preference_candidate_count 7 `
  --preference_reject_similarity_min 0.16 `
  --preference_beta 1.9 `
  --preference_beta_end 3.6 `
  --preference_margin 0.18 `
  --preference_margin_end 0.36 `
  --preference_sft_weight 0.32 `
  --preference_length_weight 0.08 `
  --preference_hardness_gamma 1.15 `
  --preference_reference_anchor_weight 0.08 `
  --preference_reference_anchor_batch_size 2 `
  --preference_short_reject_boost 0.75 `
  --preference_long_reject_boost 0.25 `
  --preference_min_chosen_quality 0.92 `
  --preference_min_chosen_words 8 `
  --preference_min_quality_gap 0.05 `
  --preference_allow_template_prompts `
  --preference_max_pairs_per_user 2 `
  --preference_max_pairs_per_source 360 `
  --preference_mining_mode auto `
  --preference_mining_progress_every 30 `
  --preference_mining_max_seconds 4500 `
  --preference_mining_max_attempt_factor 20 `
  --preference_coding_focus_boost 1.30 `
  --preference_reasoning_focus_boost 1.32 `
  --preference_counterfactual_rejects_per_prompt 4 `
  --preference_selection_strategy innovation_mix `
  --preference_selection_keep_ratio 0.66 `
  --preference_selection_min_keep 1600 `
  --preference_selection_max_keep 2400 `
  --preference_selection_hardness_target 0.46 `
  --preference_selection_hardness_bandwidth 0.22 `
  --preference_lr 1.4e-5 `
  --preference_lr_schedule cosine `
  --preference_warmup_steps 18 `
  --preference_min_lr_ratio 0.30 `
  --preference_max_grad_norm 0.9 `
  --preference_max_new_tokens 112 `
  --preference_prompt_max_tokens 352 `
  --supermix_distill_ratio 0.14 `
  --supermix_distill_max 760 `
  --supermix_distill_log_every 40 `
  --supermix_distill_max_seconds 1800 `
  --supermix_distill_min_quality 0.93 `
  --supermix_distill_allow_synthetic_prompts `
  --seed 48 `
  --device "$Device" `
  --model_dtype auto
