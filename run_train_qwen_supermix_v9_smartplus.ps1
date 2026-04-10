[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

python source\qwen_supermix_pipeline.py `
  --data datasets\conversation_data.quality_anchor_v2.jsonl datasets\conversation_data.supermix_plus_v27_500k.jsonl `
  --base_model "C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775" `
  --output_dir artifacts\qwen_supermix_enhanced_v9_smartplus `
  --init_adapter_dir artifacts\qwen_supermix_enhanced_v8_repair\adapter `
  --max_records 720 `
  --eval_size 40 `
  --max_length 256 `
  --batch_size 1 `
  --grad_accum_steps 8 `
  --epochs 1 `
  --max_steps 1 `
  --lr 2.8e-5 `
  --weight_decay 0.01 `
  --lora_r 24 `
  --lora_alpha 48 `
  --lora_dropout 0.03 `
  --use_rslora `
  --use_dora `
  --lora_init true `
  --preference_objective repo `
  --preference_steps 1 `
  --preference_pairs 24 `
  --preference_candidate_count 4 `
  --preference_beta 2.2 `
  --preference_margin 0.32 `
  --preference_sft_weight 0.22 `
  --preference_length_weight 0.05 `
  --preference_short_reject_boost 0.60 `
  --preference_long_reject_boost 0.20 `
  --preference_lr 2.8e-5 `
  --preference_max_new_tokens 48 `
  --preference_prompt_max_tokens 192 `
  --supermix_distill_ratio 0.0 `
  --supermix_distill_max 0 `
  --seed 42 `
  --device cpu `
  --skip_benchmark
