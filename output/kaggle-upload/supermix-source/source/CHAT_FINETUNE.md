# Fine-Tune + Chat (Classifier-Backed)

This project model outputs 10 logits, so chat here is retrieval-style:
- the model predicts one of 10 response buckets
- the app returns a learned response candidate from that bucket

## 1) Prepare conversation data

Use JSONL with one record per line.

Flat pair format:

```json
{"user":"How do I fix import error?","assistant":"Activate the correct environment and install the missing package."}
```

Multi-turn format:

```json
{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}
```

## 2) (Optional) Expand to a larger synthetic dataset

If your current dataset is small, expand it first:

```bash
python expand_conversation_data.py --input conversation_data.sample.jsonl --output conversation_data.large.jsonl --target 5000
```

To scale much larger with higher diversity:

```bash
python expand_conversation_data.py --input conversation_data.large.jsonl --output conversation_data.massive.jsonl --target 120000 --creative_prob 0.45 --blend_prob 0.25
```

For a larger merged dataset with quality filtering + dedupe across multiple sources:

```bash
python build_super_dataset.py --inputs conversation_data.hybrid_v6_live_knowledge.jsonl conversation_data.ultra_creative.jsonl conversation_data.massive.jsonl conversation_data.coding_knowledge_2026_02_19.jsonl conversation_data.world_events_2026_02_19.jsonl --output conversation_data.super_v7.jsonl --target 260000 --creative_prob 0.32 --blend_prob 0.22 --seed 42
```

## 3) Build local LLM database

```bash
python llm_database.py build --data conversation_data.massive.jsonl --db llm_chat.db
python llm_database.py stats --db llm_chat.db
```

## 4) Fine-tune

```bash
python finetune_chat.py --data conversation_data.sample.jsonl --device cpu
```

Outputs:
- `champion_model_chat_ft.pth`
- `chat_model_meta.json`

For a larger model variant (adds an adapter branch to the classifier head):

```bash
python finetune_chat.py --data conversation_data.massive.jsonl --weights champion_model_chat_large_ft_v3.pth --model_size large --train_all --balanced_sampler --split_mode stratified --pref_weight 0.22 --pref_beta 2.5 --pref_warmup_epochs 1.0 --hard_negative_ratio 0.7 --label_smoothing 0.05 --adaptive_pref_weighting --lr_schedule cosine --warmup_steps 400 --early_stop_patience 2 --epochs 2 --batch_size 128 --device cpu --output champion_model_chat_large_ft_v4.pth --meta chat_model_large_meta_v4.json
```

For an extra-capacity `xlarge` head (dual adapter branches + routing):

```bash
python finetune_chat.py --data conversation_data.massive.jsonl --weights champion_model_chat_large_ft_v4.pth --model_size xlarge --expansion_dim 896 --extra_expansion_dim 1536 --train_all --balanced_sampler --pref_weight 0.24 --pref_beta 2.8 --hard_negative_ratio 0.75 --adaptive_pref_weighting --epochs 2 --batch_size 128 --device cpu --output champion_model_chat_xlarge_ft_v1.pth --meta chat_model_xlarge_meta_v1.json
```

For maximum-capacity `xxlarge` with reliability defaults (EMA + gradient accumulation):

```bash
python finetune_chat.py --data conversation_data.super_v7.jsonl --weights champion_model_chat_large_ft_v6.pth --model_size xxlarge --expansion_dim 1024 --extra_expansion_dim 2048 --third_expansion_dim 3072 --train_all --balanced_sampler --split_mode stratified --val_split 0.08 --grad_accum_steps 4 --ema_decay 0.999 --pref_weight 0.24 --pref_beta 2.8 --pref_objective sigmoid --pref_group_size 4 --pref_group_estimator epo --hard_negative_ratio 0.78 --adaptive_pref_weighting --pref_warmup_epochs 1.2 --label_smoothing 0.04 --lr 0.00012 --lr_schedule cosine --warmup_steps 260 --early_stop_patience 2 --epochs 2 --batch_size 128 --device cpu --max_candidates_per_bucket 320 --output champion_model_chat_xxlarge_ft_v1.pth --meta chat_model_xxlarge_meta_v1.json
```

Research-driven `v6` recipe (expectation-style grouped preference estimation):

```bash
python finetune_chat.py --data conversation_data.hybrid_v5_clean.jsonl --weights champion_model_chat_large_ft_v5.pth --model_size large --train_all --balanced_sampler --split_mode stratified --val_split 0.08 --pref_weight 0.22 --pref_beta 2.6 --pref_objective sigmoid --pref_group_size 4 --pref_group_estimator epo --hard_negative_ratio 0.78 --adaptive_pref_weighting --pref_warmup_epochs 1.2 --label_smoothing 0.04 --lr 0.00014 --lr_schedule cosine --warmup_steps 180 --early_stop_patience 2 --epochs 2 --batch_size 128 --device cpu --max_candidates_per_bucket 224 --output champion_model_chat_large_ft_v6.pth --meta chat_model_large_meta_v6.json
```

## 5) Run chat app

```bash
python chat_app.py --weights champion_model_chat_ft.pth --meta chat_model_meta.json --device cpu
```

Type `exit` to quit.

For large model weights:

```bash
python chat_app.py --weights champion_model_chat_large_ft_v4.pth --meta chat_model_large_meta_v4.json --device cpu --pool_mode all --response_temperature 0.10 --top_labels 4 --llm_db llm_chat.db --db_top_k 120
```

For xlarge model weights with explicit creativity/style controls:

```bash
python chat_app.py --weights champion_model_chat_xlarge_ft_v1.pth --meta chat_model_xlarge_meta_v1.json --model_size auto --pool_mode all --llm_db llm_chat.db --db_top_k 160 --style_mode auto --creativity 0.35 --response_temperature 0.12
```

For xxlarge model weights:

```bash
python chat_app.py --weights champion_model_chat_xxlarge_ft_v1.pth --meta chat_model_xxlarge_meta_v1.json --model_size auto --pool_mode all --llm_db llm_chat_v6.db --db_top_k 180 --style_mode auto --creativity 0.30 --response_temperature 0.08
```

## Notes
- For better replies, train on a larger and cleaner conversation dataset.
- If you have GPU, pass `--device cuda` for faster training/inference.
- Use `--train_all` in `finetune_chat.py` only when you have enough data and compute.
- `--balanced_sampler` helps when bucket distribution is skewed.
- `--split_mode stratified` usually gives more stable validation tracking on imbalanced buckets.
- `--pref_weight` and `--adaptive_pref_weighting` enable preference-style optimization for stronger class separation.
- `--pref_warmup_epochs` ramps preference pressure in gradually for smoother early training.
- `--hard_negative_ratio` improves preference training by contrasting against confusable classes.
- `--pref_objective repo_relu` enables a RePO-style max-margin preference loss.
- `--pref_group_size` + `--pref_group_estimator epo` enables expectation-style multi-sampling preference estimation.
- `--lr_schedule cosine` with `--warmup_steps` improves optimization stability on long runs.
- `--grad_accum_steps` raises effective batch size for smoother large-model optimization.
- `--ema_decay` keeps a stable EMA shadow model; by default best-checkpoint selection uses EMA weights.
- `--early_stop_patience` stops training when validation loss stalls and keeps the best checkpoint.
- `--pool_mode all` in chat uses classifier probabilities as soft routing while searching all response buckets.
- `--llm_db` adds local database retrieval (FTS + vector rerank) for smarter response selection.
- `--style_mode` supports `auto`, `balanced`, `creative`, `concise`, `analyst` routing at inference time.
- `--creativity` controls how strongly creative rewriting is applied when style resolves to creative.
