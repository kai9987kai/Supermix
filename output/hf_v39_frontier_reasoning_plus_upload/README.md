---
license: other
library_name: pytorch
tags:
- supermix
- pytorch
- custom-model
- reasoning
- verifier
- frontier
---

# Supermix V39 Frontier Reasoning Plus

This is the public Hugging Face package for the `v39_final` line from the Supermix workspace.

## Important naming note

In the local app and benchmark files, this model is referred to as `v39_final`.

Inside the packaged training artifact, the chosen checkpoint file is:

- `champion_model_chat_v39_frontier_reasoning_plus_stage2.pth`
- `chat_model_meta_v39_frontier_reasoning_plus_stage2.json`

That is because the stage-2 checkpoint was selected as the final chosen artifact for the v39 run.

## Files

- `champion_model_chat_v39_frontier_reasoning_plus_stage2.pth`
- `chat_model_meta_v39_frontier_reasoning_plus_stage2.json`
- `v39_frontier_reasoning_plus_chosen_checkpoint.json`
- `v39_frontier_reasoning_plus_dataset_summary.json`
- `v39_frontier_reasoning_plus_training_summary.json`
- `v39_reasoning_benchmix_summary.json`
- `model_frontier_v39.py`
- `model_variants.py`
- `run.py`

## Important Note

This is a custom PyTorch checkpoint, not a standard Transformers `from_pretrained` model.

## Local Evaluation Context

Saved local recipe holdout score:

- `29 / 528`
- recipe eval accuracy: `0.0549`

There was no completed common-benchmark sweep for this model because the training pod ran out of credit before that pass finished.

## Usage

This checkpoint requires the included Python source files to construct the model class before loading the `.pth` weights.
