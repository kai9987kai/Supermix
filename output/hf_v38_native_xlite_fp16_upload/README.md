---
license: other
library_name: pytorch
tags:
- supermix
- pytorch
- custom-model
- image-generation
- native-image
- fp16
---

# Supermix V38 Native Image XLite FP16

`v38_native_xlite_fp16` is the half-precision package of the extra-lite native image checkpoint from the Supermix workspace.

## Files

- `champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth`
- `chat_model_meta_v38_native_image_xlite_single_checkpoint.json`
- `v38_native_image_xlite_training_summary.json`
- `sample_native_image_xlite_water_cycle.png`
- `model_native_image_xlite_v38.py`
- `native_image_infer_v38_xlite.py`
- `model_variants.py`
- `run.py`
- `chat_pipeline.py`

## Important Note

This is a custom PyTorch checkpoint, not a standard Diffusers or Transformers model.

## Summary

This is the smallest packaged native-image line in the workspace that still renders images directly from the model itself.

Saved local context:

- fp16 checkpoint size: about `11.2 MB`
- model family: `v38 native image xlite`
- bundled example image: `sample_native_image_xlite_water_cycle.png`

## Usage

Use `native_image_infer_v38_xlite.py` with the included metadata and weights, for example:

```bash
python native_image_infer_v38_xlite.py \
  --weights champion_model_chat_v38_native_image_xlite_single_checkpoint_fp16.pth \
  --meta chat_model_meta_v38_native_image_xlite_single_checkpoint.json \
  --prompt "simple solar system sketch with sun and several planets" \
  --output out.png \
  --device cpu
```
