# Supermix Autoresearch

This repo is much larger than `karpathy/autoresearch`, so the workflow here is narrower on purpose.

## Scope

- Keep the current Qwen/SFT/preference/distillation feature set.
- Use the smoke recipe for fast iteration.
- Use the existing benchmark harness as the only promotion gate.
- Do not edit benchmark rules during experiments.

## In-Scope Files

- `run_train_qwen_supermix_v28_smoke.ps1`
- `source/qwen_supermix_pipeline.py`
- `source/run_research_baseline.py`
- `research/results.tsv`

## Workflow

1. Pick a run tag such as `ar_20260320_230000`.
2. Launch one smoke experiment through `source/run_autoresearch_smoke.ps1`.
3. Let the wrapper:
   - run the smoke recipe
   - benchmark the produced adapter against the base model
   - append a row to `research/results.tsv`
4. Keep a candidate only when the benchmark improves on the current best logged `token_f1_delta` without negative `char_similarity_delta`.
5. Use the full launcher only after a smoke candidate looks promising.

## Constraints

- Keep current model features unless a change is required to make training run.
- Do not remove the training monitor, resume flow, distillation, or preference stages.
- Do not treat an unbenchmarked run as a keep.
- Do not reset unrelated user changes in this repo.
