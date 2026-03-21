# Supermix_29

Supermix_29 is a mixed research and packaging workspace for local AI experiments. It combines:

- Supermix / ChampionNet model research assets
- Qwen-based LoRA training pipelines
- a packaged local Python chat runtime
- a Windows desktop chat app build pipeline
- a browser-only metadata chat for GitHub Pages

This repository is not source-only. It also contains datasets, training artifacts, logs, packaged outputs, and installer files.

## Naming note

The local folder and `origin` remote use `Supermix_27`, while some branches, documents, and artifact names refer to `Supermix_28` or `Supermix_29`. Treat those as experiment-line and snapshot names inside the same evolving project.

## Repository layout

```text
.
|-- source/                Main development workspace
|-- runtime_python/        Packaged local inference runtime
|-- web_static/            Browser-only metadata chat assets
|-- installer/             Inno Setup assets
|-- artifacts/             Training outputs, adapters, checkpoints
|-- datasets/              Training data inputs
|-- dist/                  Built desktop outputs
|-- build/                 Build staging/output
|-- assets/                Branding and packaging assets
|-- ARCHITECTURE.md        Architecture notes
|-- MODEL_CARD_V28.md      Model card for the v28 line
`-- README.md
```

## Main entrypoints

Use `runtime_python/` if you want the fastest way to run the packaged local runtime.

Use `source/` if you want the current development scripts for training, chat app work, desktop packaging, and training-monitor automation.

Use the Colab notebook if you want the same Qwen training pipeline in the cloud with Drive-backed resume and cache paths.

Key files:

- `source/qwen_supermix_pipeline.py`: main Qwen training pipeline
- `source/qwen_chat_web_app.py`: current local web chat app
- `source/qwen_chat_desktop_app.py`: desktop app entrypoint used by PyInstaller
- `source/training_monitor_gui.py`: GUI monitor for active training runs
- `runtime_python/chat_web_app.py`: packaged runtime web app
- `web_static/index.html`: browser-only metadata chat UI
- `output/jupyter-notebook/supermix-colab-current-training.ipynb`: current Google Colab training notebook

## Google Colab

[Open the current training notebook in Colab](https://colab.research.google.com/github/kai9987kai/Supermix_29/blob/main/output/jupyter-notebook/supermix-colab-current-training.ipynb)

Notes:

- The notebook still launches `source/qwen_supermix_pipeline.py`, so the model family, training stages, and architecture stay aligned with the local workflow.
- It defaults to safer Colab free-tier T4 settings and keeps Drive-backed checkpoints, prepared-data caches, and distill caches resumable across runtime resets.
- If Colab gives you a different GPU tier, change `GPU_PROFILE` in the notebook to `auto`, `l4`, `a100`, or `generic`.

## Prerequisites

Runtime dependencies:

```bash
python -m pip install -r runtime_python/requirements_runtime_interface.txt
```

Training and build dependencies:

```bash
python -m pip install -r source/requirements_train_build.txt
```

Typical extras for desktop packaging:

```bash
python -m pip install pywebview pillow pyinstaller
```

Notes:

- Windows is the primary platform for the desktop and training-automation workflows.
- The Qwen pipelines expect a local or cached `Qwen/Qwen2.5-0.5B-Instruct` base model unless you override the default.
- Optional accelerators include CUDA and DirectML, with fallback to CPU.

## Quick start

### Run the packaged local runtime

```bash
python runtime_python/chat_web_app.py
```

Windows launchers are also included:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

This path is the best fit if you want to run the packaged checkpoint and metadata bundle without using the full development workspace.

### Run the current source web app

```bash
python source/qwen_chat_web_app.py
```

Use this path when you want the current development version that works with the latest adapter artifacts and desktop packaging flow.

### Run the browser-only metadata chat

Open `web_static/index.html` in a browser and load:

```text
web_static/chat_model_meta_supermix_v27_500k.browser.json
```

Important limitation: this is metadata-driven browser chat, not full PyTorch inference in the browser.

## Training workflows

The current training scripts are centered on `source/qwen_supermix_pipeline.py` and the v28 recipe family.

### Smoke run

```powershell
powershell -ExecutionPolicy Bypass -File run_train_qwen_supermix_v28_smoke.ps1
```

This writes a short validation run under:

```text
artifacts\qwen_supermix_enhanced_v28_improvements_smoke
```

### Full auto-resume run

The repo includes a Windows launcher that starts the latest full recipe and reattaches to the newest checkpoint when possible:

```bat
launch_train_qwen_supermix_v26_full.bat
```

Equivalent direct command:

```powershell
powershell -ExecutionPolicy Bypass -File source\auto_resume_supermix_training.ps1
```

By default this targets:

```text
artifacts\qwen_supermix_enhanced_v28_clean_eval_robust_ipo
```

and warm-starts from:

```text
artifacts\qwen_supermix_enhanced_v26_full
```

### Training monitor

Start the monitor GUI with:

```bat
source\launch_training_monitor_gui.bat
```

or:

```bash
python source/training_monitor_gui.py --root .
```

The monitor parses run logs, reports stage progress, and surfaces runtime/device details for active training jobs.

### Register auto-resume on boot

```powershell
powershell -ExecutionPolicy Bypass -File source\register_supermix_auto_resume_task.ps1
```

This now creates two Windows hooks when Task Scheduler is available:

- a startup task that resumes the last recorded training run on boot without opening the monitor UI
- a logon task that opens the training monitor and only relaunches training if it is not already running

When scheduled tasks are unavailable, it falls back to an `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` entry, which restores the old login-time behavior.

The boot launcher restores the most recent run from `.last_training_launch.txt`, so it resumes the last recorded output directory and log files instead of always reverting to the default v28 path.

### Disable auto-resume on boot

```powershell
powershell -ExecutionPolicy Bypass -File source\unregister_supermix_auto_resume_task.ps1
```

This removes the startup task, the logon monitor task, the legacy single-task hook, and the `HKCU\Software\Microsoft\Windows\CurrentVersion\Run` fallback entry if present.

## Desktop packaging

Build the desktop application:

```powershell
powershell -ExecutionPolicy Bypass -File build_qwen_chat_desktop_exe.ps1
```

Expected output:

```text
dist\SupermixQwenDesktop\SupermixQwenDesktop.exe
```

The build script generates branding, resolves the latest adapter artifact automatically, stages a desktop bundle, and packages the app with PyInstaller.

Build the installer:

```powershell
powershell -ExecutionPolicy Bypass -File build_qwen_chat_desktop_installer.ps1
```

Expected output:

```text
dist\installer\
```

The installer flow requires Inno Setup 6. If `iscc.exe` is not available, install it with:

```powershell
winget install --id JRSoftware.InnoSetup -e --accept-package-agreements --accept-source-agreements
```

## Tests

This repo includes direct-run smoke and regression tests for the chat app, training pipeline, monitor, and expert variants.

Examples:

```bash
python test_qwen_chat_web_app.py
python test_training_monitor_gui.py
python test_training_resume_automation.py
```

Additional experiment-specific tests live both at the repository root and under `source/`.

## Important docs

- `ARCHITECTURE.md`
- `MODEL_CARD_V28.md`
- `source/CHAT_FINETUNE.md`
- `source/RESEARCH_UPGRADES.md`

## Limitations and repo shape

- This repo contains generated artifacts and logs alongside source code.
- Naming is mixed across `Supermix_27`, `Supermix_28`, and `Supermix_29`.
- The browser build is metadata-only and does not run the full model in-browser.
- Desktop packaging and training automation are Windows-first.
- Some flows assume a locally available base model and local Python environment.

## License

See `LICENSE`.
