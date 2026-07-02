---
license: other
task_categories:
- text-generation
language:
- en
pretty_name: Supermix Conversation Datasets
size_categories:
- 100K<n<1M
---

# Supermix Conversation Datasets

This dataset repo packages the share-ready conversation-training files that were prepared in the Supermix workspace for portable training runs.

## Files

- `conversation_data.coding_knowledge_2026_02_19.jsonl`
- `conversation_data.mega_creative_250k_v2.jsonl`
- `conversation_data.mega_reasoning_creative_v25_75582.jsonl`
- `conversation_data.quality_anchor_v2.jsonl`
- `conversation_data.supermix_plus_v27_500k.jsonl`
- `conversation_data.world_events_2026_02_19.jsonl`

## Format

Each file is JSONL. Rows are conversation/training examples intended for Supermix training pipelines.

## Notes

- This repo is the curated public bundle from `output/kaggle-upload/supermix-datasets`.
- It does not include every intermediate or experimental dataset file from the local workspace.
- Check each file's provenance and intended use before mixing it into downstream training.
