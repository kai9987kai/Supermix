[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python source\qwen_supermix_pipeline.py `
  --data datasets\conversation_data.quality_anchor_v2.jsonl datasets\conversation_data.coding_knowledge_2026_02_19.jsonl datasets\conversation_data.world_events_2026_02_19.jsonl datasets\conversation_data.supermix_plus_v27_500k.jsonl `
  --base_model "C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775" `
  --output_dir artifacts\qwen_supermix_enhanced_v10_smarter `
  --init_adapter_dir artifacts\qwen_supermix_enhanced_v8_repair\adapter `
  --max_records 480 `
  --eval_size 40 `
  --max_length 256 `
  --batch_size 1 `
  --grad_accum_steps 4 `
  --epochs 1 `
  --max_steps 6 `
  --lr 3.2e-5 `
  --weight_decay 0.01 `
  --lora_r 24 `
  --lora_alpha 48 `
  --lora_dropout 0.03 `
  --use_rslora `
  --use_dora `
  --lora_init true `
  --preference_objective repo `
  --preference_steps 4 `
  --preference_pairs 72 `
  --preference_candidate_count 5 `
  --preference_beta 2.2 `
  --preference_margin 0.30 `
  --preference_sft_weight 0.22 `
  --preference_length_weight 0.06 `
  --preference_short_reject_boost 0.65 `
  --preference_long_reject_boost 0.20 `
  --preference_lr 3.0e-5 `
  --preference_max_new_tokens 56 `
  --preference_prompt_max_tokens 224 `
  --supermix_distill_ratio 0.15 `
  --supermix_distill_max 80 `
  --seed 42 `
  --device cpu `
  --skip_benchmark
