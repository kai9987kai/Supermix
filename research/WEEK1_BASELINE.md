# Week 1 Baseline Tracker

Date started: 2026-02-28 (UTC)

## Objective

Establish a reproducible benchmark baseline with quality and efficiency metrics before running new training experiments.

## Tooling Added

- `source/run_research_baseline.py`
  - Reproducible, timestamped benchmark runner.
  - Supports fixed eval set (`--eval_jsonl`) or derived eval split (`--data` + `--seed`).
  - Produces `benchmark_results.json` and optional comparison chart for base vs tuned.
- `source/qwen_supermix_pipeline.py`
  - `evaluate_model(...)` now reports efficiency metrics:
    - `avg_prompt_tokens`
    - `avg_generated_tokens`
    - `total_prompt_tokens`
    - `total_generated_tokens`
    - `eval_seconds`
    - `generated_tokens_per_sec`

## First Executed Run (Smoke)

Command:

```powershell
python source\run_research_baseline.py `
  --base_model "C:\Users\kai99\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct\snapshots\7ae557604adf67be50417f59c2c2f167def9a775" `
  --eval_jsonl artifacts\qwen_supermix_enhanced_v8_repair\eval_pairs.jsonl `
  --adapter_dir artifacts\qwen_supermix_enhanced_v8_repair\adapter `
  --max_eval_samples 8 `
  --max_length 256 `
  --max_new_tokens 32 `
  --output_root artifacts\research_baselines `
  --run_name week1_start_smoke_20260228 `
  --benchmark_type week1_smoke `
  --device cpu `
  --seed 42
```

Artifacts:

- `artifacts/research_baselines/week1_start_smoke_20260228/benchmark_results.json`
- `artifacts/research_baselines/week1_start_smoke_20260228/benchmark_comparison.png`
- `artifacts/research_baselines/week1_start_smoke_20260228/eval_pairs.jsonl`

## Smoke Results Snapshot

- Base:
  - `token_f1`: `0.0792`
  - `char_similarity`: `0.1781`
  - `avg_gen_seconds`: `4.2672`
  - `avg_generated_tokens`: `28.25`
  - `generated_tokens_per_sec`: `5.7566`
- Tuned (`v8_repair` adapter):
  - `token_f1`: `0.7147`
  - `char_similarity`: `0.7912`
  - `avg_gen_seconds`: `3.1759`
  - `avg_generated_tokens`: `20.875`
  - `generated_tokens_per_sec`: `5.5513`
- Delta (`tuned - base`):
  - `token_f1`: `+0.6355`
  - `char_similarity`: `+0.6131`
  - `avg_gen_seconds`: `-1.0913`
  - `avg_generated_tokens`: `-7.375`

## Next Runs (Week 1)

1. Increase eval size to reduce variance:
   - `--max_eval_samples 40`
   - same seed/config for comparability.
2. Repeat with 3 seeds (`42, 43, 44`) and aggregate mean/std.
3. Freeze baseline JSONs and treat them as promotion gates for Week 2 curriculum experiments.

## Eval40 Multi-Seed Results (2026-03-01 UTC)

Executed runs:

- `artifacts/research_baselines/week1_seed42_eval40_20260301/benchmark_results.json`
- `artifacts/research_baselines/week1_seed43_eval40_20260301/benchmark_results.json`
- `artifacts/research_baselines/week1_seed44_eval40_20260301/benchmark_results.json`

Aggregate artifact:

- `artifacts/research_baselines/week1_eval40_aggregate_20260301/summary.json`

Mean delta (`tuned - base`), `n=3`:

- `token_f1`: `+0.3606` (std `0.0000`)
- `char_similarity`: `+0.4234` (std `0.0000`)
- `eval_loss`: `-2.5584` (std `0.0000`)
- `avg_generated_tokens`: `-7.3250` (std `0.0000`)
- `avg_gen_seconds`: `-0.5060` (std `0.7401`)
- `generated_tokens_per_sec`: `-1.2166` (std `0.8889`)

Important note:

- `artifacts/qwen_supermix_enhanced_v8_repair/eval_pairs.jsonl` contains exactly `40` rows.
- With `--max_eval_samples 40`, all seeds evaluate the same fixed sample set, so quality metrics are deterministic in this setup; only latency/throughput varied.

Updated promotion gate recommendation:

1. Keep current quality baseline gates from `summary.json`:
   - Require `token_f1` improvement `>= +0.30`
   - Require `char_similarity` improvement `>= +0.35`
2. Add a larger stochastic eval for variance checks:
   - build an eval split with at least `120` samples from broader datasets, then repeat 3-seed aggregation.
3. Move to Week 2 curriculum experiments only after the larger eval confirms non-regression on quality and acceptable efficiency.

## Eval120 Stochastic Multi-Seed Results (2026-03-01 UTC)

Executed runs:

- `artifacts/research_baselines/week1_seed42_eval120_20260301/benchmark_results.json`
- `artifacts/research_baselines/week1_seed43_eval120_20260301/benchmark_results.json`
- `artifacts/research_baselines/week1_seed44_eval120_20260301/benchmark_results.json`

Aggregate artifact:

- `artifacts/research_baselines/week1_eval120_aggregate_20260301/summary.json`

Configuration:

- Source datasets:
  - `datasets/conversation_data.quality_anchor_v2.jsonl`
  - `datasets/conversation_data.coding_knowledge_2026_02_19.jsonl`
  - `datasets/conversation_data.world_events_2026_02_19.jsonl`
  - `datasets/conversation_data.supermix_plus_v27_500k.jsonl`
- `max_records=2400`, `eval_size=120`, `max_eval_samples=120`
- Quality filter summary per run:
  - `raw=2973 kept=2400 empty=18 filtered=531 deduped=24`

Mean delta (`tuned - base`), `n=3`:

- `token_f1`: `+0.3228` (std `0.0220`)
- `char_similarity`: `+0.2772` (std `0.0151`)
- `eval_loss`: `-2.2485` (std `0.0612`)
- `avg_generated_tokens`: `-3.6556` (std `0.4480`)
- `avg_gen_seconds`: `-0.6386` (std `0.6120`)
- `generated_tokens_per_sec`: `-0.1911` (std `0.3180`)

Observed gate status vs current recommendation:

1. `token_f1` gate (`>= +0.30`): PASS
2. `char_similarity` gate (`>= +0.35`): FAIL on eval120

Interpretation:

- The stronger stochastic eval reduced the apparent margin from the earlier fixed eval40 run.
- Current tuned model still improves reasoning-quality metrics, but not enough on `char_similarity` to clear the stricter gate.

Updated readiness call:

1. Do not promote this adapter as Week 2 champion yet.
2. Proceed to Week 2 curriculum work, but require new checkpoints to beat the eval120 aggregate gate before promotion.
