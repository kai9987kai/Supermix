---
license: other
library_name: pytorch
tags:
- supermix
- custom-model
- pytorch
- reasoning
- frontier
---

# Supermix V33 Frontier

`v33_final` is the best saved common-benchmark text model from the Supermix v33 frontier line in this workspace.

## Files

- `champion_model_chat_v33_frontier_full_final.pth`
- `chat_model_meta_v33_frontier_full_final.json`
- `v33_frontier_training_summary.json`
- `model_frontier_v33.py`
- `model_variants.py`
- `run.py`

## Important Note

This is a custom PyTorch checkpoint, not a standard Transformers `from_pretrained` model.

## Local Sources

- weights and metadata were extracted from `champion_v33_frontier_full_model_20260326.zip`
- architecture code comes from the local Supermix source tree

## Benchmark Context

Saved expanded common-benchmark score in this workspace:

- overall exact: `0.1867`
- ARC-Challenge: `0.20`
- BoolQ: `0.08`
- GSM8K: `0.02`
- HellaSwag: `0.26`
- MMLU: `0.14`
- PIQA: `0.42`

## Usage

This checkpoint needs the included Python source files to construct the model class before loading the `.pth` weights.
