# Supermix_29

Supermix_29 is the working monorepo for the current Supermix / ChampionNet / Omni Collective line.

This repository combines:

- local-first chat and multimodal runtime code
- experimental training and continuation pipelines
- desktop EXE and installer packaging
- benchmark tooling and graph generation
- published-model export helpers
- bundled datasets and generated research artifacts

It is intentionally a mixed workspace, not a minimal source-only model repo.

## Current status

As of March 29, 2026:

- the latest finished omni checkpoint in this repo is `omni_collective_v4`
- the latest packaged desktop release is `studio-desktop-20260329-omni-v4-allmodels`
- the installer bundle currently includes `23` zipped model artifacts from the local model-pack directory used by the desktop build
- a `v5` continuation path exists in `source/` and is currently an in-progress local experiment, not a finished released model

## What is in this repo

- `source/`
  - active development workspace
  - training scripts, model definitions, dataset builders, benchmark runners, desktop packaging helpers
- `runtime_python/`
  - packaged local runtime path
  - simpler run path than the full `source/` workspace
- `datasets/`
  - conversation, coding, reasoning, science, and related local training inputs
- `output/`
  - generated artifacts, benchmark graphs, summaries, logs, Hugging Face upload folders
- `installer/`
  - Inno Setup definitions for the desktop app
- `dist/`
  - built EXEs and installer outputs
- `web_static/`
  - lightweight browser-only metadata bundle

## Main capabilities

- multimodel desktop app with model selector, Auto routing, collective mode, and agent mode
- local chat, image-prompt, math, science-image, and omni-fusion model families
- native-image experimental checkpoints
- training pipelines for frontier, omni, lite, and specialist model lines
- benchmark sweeps across common text benchmarks
- export and publishing workflows for GitHub releases and Hugging Face model/dataset repos

<img width="610" height="585" alt="image" src="https://github.com/user-attachments/assets/e9ee2188-9a40-4019-bebe-032b9e38bc4a" />

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

### Run the active source app

```bash
python source/chat_web_app.py
```

### Run the desktop multimodel app from source

```bash
python source/supermix_multimodel_desktop_app.py
```

### Run the browser-only static bundle

Open:

```text
web_static/index.html
```

## Current desktop release

Latest release published from this repo:

- Release page:
  - `https://github.com/kai9987kai/Supermix_29/releases/tag/studio-desktop-20260329-omni-v4-allmodels`
- Installer:
  - `https://github.com/kai9987kai/Supermix_29/releases/download/studio-desktop-20260329-omni-v4-allmodels/SupermixStudioDesktopSetup.exe`
- EXE:
  - `https://github.com/kai9987kai/Supermix_29/releases/download/studio-desktop-20260329-omni-v4-allmodels/SupermixStudioDesktop.exe`

Local build outputs:

- `dist/SupermixStudioDesktop/SupermixStudioDesktop.exe`
- `dist/installer/SupermixStudioDesktopSetup.exe`
- `dist/installer/SupermixStudioDesktopReleaseSHA256.txt`

## Model families in the workspace

The repo contains code and artifacts for several model lines:

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
- native-image line
  - `v36`
  - `v37`
  - `v38`
- omni-collective line
  - `v1`
  - `v2`
  - `v3`
  - `v4`
- specialist lines
  - `math_equation_micro_v1`
  - `science_image_recognition_micro_v1`

## Latest finished omni model

The latest finished omni checkpoint in this repo is `omni_collective_v4`.

Key details from [`output/supermix_omni_collective_v4_frontier_20260329/omni_collective_v4_frontier_summary.json`](output/supermix_omni_collective_v4_frontier_20260329/omni_collective_v4_frontier_summary.json):

- parameter count: `19,032,281`
- stage-1 rows: `8,589`
- stage-2 rows: `9,447`
- final stage-2 weighted validation score: `0.5176`
- final stage-2 validation:
  - intent: `0.8195`
  - response: `0.1402`
  - vision: `0.9020`
  - domain: `0.7765`

Local packaged artifact:

- [`output/supermix_omni_collective_v4_frontier_20260329.zip`](output/supermix_omni_collective_v4_frontier_20260329.zip)

## Hugging Face models

Public model repos already published from this workspace:

- `Kai9987kai/supermix-v33-frontier`
- `Kai9987kai/supermix-omni-collective-v1`
- `Kai9987kai/supermix-v38-native-image-xlite-fp16`
- `Kai9987kai/supermix-v39-frontier-reasoning-plus`
- `Kai9987kai/supermix-omni-collective-v2-frontier`
- `Kai9987kai/supermix-math-equation-micro-v1`
- `Kai9987kai/supermix-omni-collective-v4-frontier`

## Hugging Face datasets

Public dataset repos already published from this workspace:

- `Kai9987kai/supermix-conversation-datasets`
- `Kai9987kai/supermix-science-vision-dataset`

## Benchmarks

The current local multibench comparison bundle is:

- [`output/pdf/benchmark_local_all_models_multibench_20260329.pdf`](output/pdf/benchmark_local_all_models_multibench_20260329.pdf)
- [`output/benchmark_local_all_models_multibench_20260329.json`](output/benchmark_local_all_models_multibench_20260329.json)
- [`output/benchmark_local_all_models_multibench_20260329.csv`](output/benchmark_local_all_models_multibench_20260329.csv)

The current graph covers `20` benchmarked local model entries and keeps specialist-only models labeled separately when the common text suite is not the right evaluation fit.

Representative current common-benchmark leaders from the local graph JSON:

- `v33_final`: `0.1867`
- `v39_final`: `0.1800`
- `omni_collective_v1`: `0.1633`
- `v34_final`: `0.1600`
- `v36_native`: `0.1533`
- `v35_final`: `0.1533`
- `omni_collective_v4`: `0.0900`

## Training entry points

Representative training and continuation scripts:

- `source/train_omni_collective_v2.py`
- `source/train_omni_collective_v3.py`
- `source/train_omni_collective_v4.py`
- `source/train_omni_collective_v5.py`
- `source/train_math_equation_model.py`
- `source/train_image_recognition_model.py`
- `source/build_reasoning_benchmix_v39.py`
- `source/benchmark_all_models_common.py`

If you want the active experimental path, start in `source/`.

## Desktop build entry points

Primary desktop build helpers:

- `source/build_supermix_studio_desktop_exe.ps1`
- `source/build_supermix_studio_desktop_installer.ps1`
- `SupermixStudioDesktop.spec`
- `installer/SupermixStudioDesktop.iss`

The current bundled-model manifest is:

- [`output/supermix_studio_bundled_models_manifest.json`](output/supermix_studio_bundled_models_manifest.json)

## Research and experiment notes

This repo is a living experiment workspace. It contains finished artifacts, release-ready packaging, and in-progress work at the same time.

That means you will see mixed generations such as:

- `v28`
- `v30`
- `v33`
- `v34`
- `v35`
- `v36`
- `v37`
- `v38`
- `v39`
- `omni_collective_v1` through `omni_collective_v5`

That is expected.

## Recommended starting points

If you want to:

- run a packaged local system
  - use `runtime_python/`
- work on the active multimodel app
  - use `source/supermix_multimodel_web_app.py`
  - use `source/supermix_multimodel_desktop_app.py`
- work on training
  - start in `source/`
- inspect the current benchmark outputs
  - use `output/benchmark_local_all_models_multibench_20260329.*`
- build a Windows installer
  - use the PowerShell build scripts in `source/` plus `installer/`

## Platform notes

- Windows is the main desktop packaging target
- the repo includes PyInstaller specs, PowerShell build scripts, and Inno Setup definitions
- some training flows were designed around cloud GPU workflows, but the repo also supports local CPU experimentation

## Security note

Do not commit or publish browser-session dumps, cookies, temporary automation state, or live access tokens.

Relevant policy docs:

- `SECURITY.md`
- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`

## License

See `LICENSE`.
