# Supermix_29

Supermix_29 is the active monorepo for the current Supermix desktop app, the Omni Collective training line, specialist model experiments, benchmark tooling, and local-first packaging flow.

This repo is intentionally a mixed workspace. It contains source code, training scripts, build helpers, generated manifests, research outputs, and packaging metadata for a local-first multimodel app.

## Current status

As of April 7, 2026:

- the latest finished full omni checkpoint in this workspace is `omni_collective_v7`
- the latest interim omni artifact is `omni_collective_v8_preview`
- the full `omni_collective_v8` training run is still in progress locally
- the latest desktop release is `studio-desktop-20260407-curated-core-model-store`
- the desktop installer now uses a curated built-in bundle plus an in-app Hugging Face model store instead of shipping the full legacy model zoo

## What is in this repo

- `source/`
  - active development workspace
  - model definitions, training scripts, dataset builders, benchmark runners, desktop/web app code, release helpers
- `runtime_python/`
  - simpler runtime path for lighter local usage
- `datasets/`
  - local conversation, coding, reasoning, science, and specialist training inputs
- `output/`
  - generated graphs, summaries, manifests, previews, logs, and Hugging Face upload staging
- `installer/`
  - Inno Setup definitions and post-install notes for the desktop app
- `dist/`
  - locally built EXEs and installer outputs
- `web_static/`
  - lightweight browser-only metadata/static bundle

## Main capabilities

- multimodel desktop app with selector, `Auto` routing, collective mode, and agent mode
- local chat, omni, image, vision, math, protein, 3D, and materials specialist model families
- desktop installer with curated built-in models and downloadable legacy/optional models
- benchmark sweeps and local graph generation for the model zoo
- Hugging Face publishing helpers for models, installers, and downloadable model-store artifacts
- training pipelines for omni, frontier, native-image, and specialist lines

## Quick start

### Run the packaged runtime

```bash
python runtime_python/chat_web_app.py
```

Windows launchers:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

### Run the active source web app

```bash
python source/chat_web_app.py
```

### Run the multimodel desktop app from source

```bash
python source/supermix_multimodel_desktop_app.py
```

### Run the training monitor

```bash
python source/training_monitor_gui.py --root .
```

### Open the browser-only static bundle

```text
web_static/index.html
```

## Current desktop release

Latest desktop release:

- Release page:
  - `https://github.com/kai9987kai/Supermix_29/releases/tag/studio-desktop-20260407-curated-core-model-store`
- Portable EXE:
  - `https://github.com/kai9987kai/Supermix_29/releases/download/studio-desktop-20260407-curated-core-model-store/SupermixStudioDesktop.exe`
- Full installer:
  - `https://huggingface.co/datasets/Kai9987kai/supermix-studio-desktop-installer/resolve/main/SupermixStudioDesktopSetup.exe?download=true`
- SHA256:
  - `https://huggingface.co/datasets/Kai9987kai/supermix-studio-desktop-installer/resolve/main/SupermixStudioDesktopReleaseSHA256.txt?download=true`
- Curated bundle manifest:
  - `https://huggingface.co/datasets/Kai9987kai/supermix-studio-desktop-installer/resolve/main/SupermixStudioDesktopCuratedBundleManifest.json?download=true`

Local build outputs:

- `dist/SupermixStudioDesktop/SupermixStudioDesktop.exe`
- `dist/installer/SupermixStudioDesktopSetup.exe`
- `dist/installer/SupermixStudioDesktopReleaseSHA256.txt`

Why the installer is on Hugging Face:

- the full installer is about `2.97 GB`
- GitHub release assets are capped below that size
- the GitHub release hosts the EXE and notes, while Hugging Face hosts the oversized installer

## Curated bundled models

The curated installer build ships these models by default:

- `v40_benchmax`
- `omni_collective_v8_preview`
- `omni_collective_v7`
- `science_vision_micro_v1`
- `v38_native_xlite_fp16`
- `dcgan_v2_in_progress`
- `math_equation_micro_v1`
- `protein_folding_micro_v1`
- `mattergen_micro_v1`
- `three_d_generation_micro_v1`

The live curated bundle manifest is:

- [`output/supermix_studio_bundled_models_manifest.json`](output/supermix_studio_bundled_models_manifest.json)

## Model store

The desktop app now supports an in-app model store backed by your Hugging Face model zoo.

- model zoo:
  - `https://huggingface.co/datasets/Kai9987kai/supermix-model-zoo`
- published downloadable artifacts:
  - `33` model zips
- downloaded models install into the local models directory and automatically become available to:
  - model selection
  - `Auto` routing
  - collective mode
  - agent mode

This lets the installer stay curated while older or optional models remain one-click downloadable.

## Model families in the workspace

The current workspace includes code and/or packaged artifacts for:

- Qwen adapter line
  - `v28`
  - `v30`
- Champion / frontier line
  - `v31`
  - `v32`
  - `v33`
  - `v34`
  - `v35`
  - `v39`
  - `v40_benchmax`
- native-image line
  - `v36`
  - `v37`
  - `v38`
- omni-collective line
  - `v1`
  - `v2`
  - `v3`
  - `v4`
  - `v5`
  - `v6`
  - `v7`
  - `v8_preview`
  - `v8` in progress locally
- specialist lines
  - `math_equation_micro_v1`
  - `science_vision_micro_v1`
  - `protein_folding_micro_v1`
  - `mattergen_micro_v1`
  - `three_d_generation_micro_v1`
  - `dcgan_mnist_model`
  - `dcgan_v2_in_progress`

## Latest finished omni model

The latest finished full omni checkpoint in this workspace is `omni_collective_v7`.

Local artifact:

- `output/supermix_omni_collective_v7_frontier_20260403.zip`

The latest preview checkpoint is:

- `output/supermix_omni_collective_v8_preview_20260407_001155.zip`

The current long-running experiment is `omni_collective_v8`, which is still training locally and is not yet the published stable omni release.

## Hugging Face publishing

Public Hugging Face repos used by this workspace now include:

- model zoo:
  - `Kai9987kai/supermix-model-zoo`
- desktop installer mirror:
  - `Kai9987kai/supermix-studio-desktop-installer`

Previously published model repos from this workspace include:

- `Kai9987kai/supermix-v33-frontier`
- `Kai9987kai/supermix-omni-collective-v1`
- `Kai9987kai/supermix-v38-native-image-xlite-fp16`
- `Kai9987kai/supermix-v39-frontier-reasoning-plus`
- `Kai9987kai/supermix-omni-collective-v2-frontier`
- `Kai9987kai/supermix-math-equation-micro-v1`
- `Kai9987kai/supermix-omni-collective-v4-frontier`
- `Kai9987kai/supermix-omni-collective-v7-frontier`

Public dataset repos include:

- `Kai9987kai/supermix-conversation-datasets`
- `Kai9987kai/supermix-science-vision-dataset`

## Benchmarks

The current local all-model graph bundle is:

- `output/pdf/benchmark_local_all_models_multibench_20260403.pdf`
- `output/benchmark_local_all_models_multibench_20260403.json`
- `output/benchmark_local_all_models_multibench_20260403.csv`

The current preview benchmark bundle for `omni_collective_v8_preview` is:

- `output/benchmark_omni_collective_v8_preview_20260407_001155/benchmark_all_models_common_summary.json`
- `output/benchmark_omni_collective_v8_preview_20260407_001155/benchmark_all_models_common_graph.png`

Representative current results:

- `v40_benchmax`: `0.2433` common benchmark score
- `omni_collective_v8_preview`: `0.2167` on the reduced preview benchmark sweep
- `omni_collective_v7`: `0.1067` on the current 6-benchmark local sweep

Specialist-only models remain in the graph inventory even when the common text benchmark is not the right evaluation fit.

## Training entry points

Representative training and continuation scripts:

- `source/train_omni_collective_v2.py`
- `source/train_omni_collective_v3.py`
- `source/train_omni_collective_v4.py`
- `source/train_omni_collective_v5.py`
- `source/train_omni_collective_v6.py`
- `source/train_omni_collective_v7.py`
- `source/train_omni_collective_v8.py`
- `source/train_math_equation_model.py`
- `source/train_image_recognition_model.py`
- `source/train_protein_folding_model.py`
- `source/train_three_d_generation_model.py`
- `source/train_mattergen_generation_model.py`
- `source/benchmark_all_models_common.py`

If you want the active experimental path, start in `source/`.

## Desktop build entry points

Primary desktop build helpers:

- `source/build_supermix_studio_desktop_exe.ps1`
- `source/build_supermix_studio_desktop_installer.ps1`
- `SupermixStudioDesktop.spec`
- `installer/SupermixStudioDesktop.iss`

The current desktop app now supports:

- curated built-in bundle seeding
- in-app Hugging Face model-store downloads
- richer chat UI with drafts, context bank, compare bench, and dispatch preview
- richer training monitor with recovery posture, rescue guidance, and fleet spotlight summaries

## Notes

- this repo is intentionally not a clean source-only model repo
- large local artifacts, checkpoints, and logs are often kept locally and mirrored selectively to releases or Hugging Face
- the safest way to consume the desktop app is through the curated-core release plus the in-app model store
