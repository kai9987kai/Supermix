---
license: other
library_name: pytorch
tags:
- supermix
- pytorch
- multimodal
- vision
- math
- custom-model
---

# Supermix Omni Collective V1

`omni_collective_v1` is a compact fused local assistant checkpoint from the Supermix workspace. It combines text routing, math-aware handling, and lightweight image-recognition support in one custom PyTorch model.

## Files

- `omni_collective_v1.pth`
- `omni_collective_v1_meta.json`
- `omni_collective_v1_summary.json`
- `omni_collective_model.py`
- `image_feature_utils.py`
- `image_recognition_model.py`
- `math_equation_model.py`

## Important Note

This is a custom PyTorch checkpoint, not a standard Transformers model.

## Validation Summary

From the bundled training summary:

- intent accuracy: `0.9444`
- response accuracy: `0.7037`
- vision accuracy: `0.10`

## Usage

To load this model, construct the custom `OmniCollectiveNet` / `OmniCollectiveEngine` class from `omni_collective_model.py` and then load `omni_collective_v1.pth` with the provided metadata JSON.
