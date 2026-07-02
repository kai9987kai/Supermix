---
license: other
task_categories:
- image-classification
language:
- en
pretty_name: Supermix Science Vision Dataset
size_categories:
- n<1K
---

# Supermix Science Vision Dataset

This dataset repo contains the compact science-diagram image set used for the local `Science Vision Micro` model in the Supermix workspace.

## Files

- `images/` with 11 labeled PNG images
- `metadata.csv`
- `metadata.jsonl`
- `science_image_recognition_micro_v1_summary.json`

## Labels

- `cell`
- `magnet`
- `circuit`
- `water_cycle`
- `moon_phases`
- `photosynthesis`
- `solar_system`
- `dna`
- `temperature`
- `states_of_matter`
- `light_dispersion`

## Format

`metadata.csv` and `metadata.jsonl` map each image file to its label, concept, caption, and tags.

## Notes

- This is the local science-vision dataset prepared in `output/science_vision_dataset`.
- The included summary JSON records the training context for the associated `Science Vision Micro` model.
