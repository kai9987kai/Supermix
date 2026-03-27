
# Supermix_29

Supermix_29 is a local-first AI research, training, packaging, and desktop-delivery workspace for the broader **Supermix / ChampionNet** project line.

It brings together:

- model research and experimental architecture work
- chat fine-tuning and dataset-building pipelines
- packaged local Python inference/runtime code
- static browser metadata chat assets
- Windows desktop build and installer tooling
- benchmark, distillation, and model-export utilities

This repository is intentionally **not source-only**. It also includes runtime assets, model metadata, training manifests, datasets, build outputs, installer files, and research documentation.

---

## What this repo is for

Supermix_29 is best understood as a **full project workspace**, not a single script or a minimal model release.

It supports several parallel use cases:

1. **Run a packaged local chat runtime**
2. **Develop and test the current source-side chat stack**
3. **Build or package desktop variants**
4. **Train or fine-tune chat models from curated manifests**
5. **Export lightweight browser metadata bundles**
6. **Experiment with benchmark, frontier, and native-image model lines**

---

## Core capabilities

- **Local runtime interface**
  - packaged Python runtime under `runtime_python/`
  - web and terminal launchers for local use
  - model metadata and runtime helpers included

- **Research and development workspace**
  - active development files under `source/`
  - chat pipeline, memory, export, benchmarking, distillation, and dataset tooling
  - multiple experimental model lines in one repo

- **Static browser bundle**
  - browser-only metadata chat assets under `web_static/`
  - suitable for lightweight GitHub Pages-style hosting

- **Desktop packaging**
  - PyInstaller spec files in the repo root
  - PowerShell build scripts for desktop EXE generation
  - Inno Setup installer definitions under `installer/`

- **Cloud training support**
  - Kaggle workflow documented in `KAGGLE_GUIDE.md`
  - notebook-oriented training flow kept aligned with the local pipeline

---

## Repository layout

```text
.
├── source/                  Active development workspace
├── runtime_python/          Packaged local runtime
├── web_static/              Browser-only metadata chat bundle
├── installer/               Inno Setup installer definitions
├── artifacts/               Training outputs / adapters / checkpoints
├── assets/                  Branding and packaging assets
├── bundles/                 Packaged bundle assets
├── databases/               Local data storage assets
├── datasets/                Training and evaluation inputs
├── dist/                    Built outputs
├── output/                  Notebook and generated output area
├── research/                Research notes and experiment material
├── ARCHITECTURE.md          Architecture guide
├── KAGGLE_GUIDE.md          Kaggle workflow guide
├── MODEL_CARD_V28.md        Model card for a major earlier line
├── CONTRIBUTING.md          Contribution guide
├── SECURITY.md              Security policy
└── README.md
````

---

## Important directories

### `source/`

This is the main development workspace.

Representative files include:

* `chat_app.py`
* `chat_web_app.py`
* `chat_pipeline.py`
* `chat_memory.py`
* `finetune_chat.py`
* `finetune_chat_manifest_sequential.py`
* `benchmark.py`
* `benchmark_all_models_common.py`
* `build_super_dataset.py`
* `export_browser_chat_meta.py`
* `distill_native_image_lite_v37.py`
* `distill_native_image_xlite_v38.py`
* `model_frontier_v33.py`
* `model_frontier_v35.py`
* `model_frontier_v39.py`
* `model_native_image_lite_v37.py`
* `model_native_image_xlite_v38.py`

Use `source/` when you want the most current development path.

---

### `runtime_python/`

This is the packaged runtime path for local use.

Key files include:

* `chat_app.py`
* `chat_web_app.py`
* `chat_pipeline.py`
* `chat_memory.py`
* `device_utils.py`
* `llm_database.py`
* `model_variants.py`
* `requirements_runtime_interface.txt`
* `run.py`

Use `runtime_python/` when you want the simplest path to run the packaged local runtime without entering the full development workspace.

---

### `web_static/`

This contains the browser-only metadata bundle.

Key files:

* `index.html`
* `chat_model_meta_supermix_v27_500k.browser.json`

Use this path when you want a lightweight browser experience or a GitHub Pages-style static deployment.

---

### `installer/`

Installer definitions for Windows desktop packaging.

Key files:

* `SupermixQwenDesktop.iss`
* `SupermixStudioDesktop.iss`
* `postinstall_notes.txt`
* `postinstall_notes_studio.txt`

---

## Quick start

## 1) Install runtime dependencies

```bash
python -m pip install -r runtime_python/requirements_runtime_interface.txt
```

---

## 2) Run the packaged local web runtime

```bash
python runtime_python/chat_web_app.py
```

Windows launchers are also included:

```bat
runtime_python\launch_chat_web_supermix.bat
runtime_python\launch_chat_terminal_supermix.bat
```

This is the recommended entry path if your goal is to run the packaged local system rather than develop the training stack.

---

## 3) Run the current source-side web app

```bash
python source/chat_web_app.py
```

Use this path when you want the actively developed version of the application and the broader development tooling around it.

---

## 4) Run the browser-only static version

Open:

```text
web_static/index.html
```

Then load:

```text
web_static/chat_model_meta_supermix_v27_500k.browser.json
```

This mode is metadata-driven and intended for lightweight browser delivery.

---

## Desktop build workflow

This repository includes desktop build tooling in both script and spec form.

Relevant files include:

* `SupermixQwenDesktop.spec`
* `SupermixQwenDesktopV26.spec`
* `SupermixStudioDesktop.spec`
* `build_qwen_chat_desktop_exe.ps1`
* `build_qwen_chat_desktop_installer.ps1`

Additional build helpers also exist inside `source/`, including:

* `build_qwen_chat_desktop_exe.ps1`
* `build_qwen_chat_desktop_installer.ps1`
* `build_supermix_studio_desktop_exe.ps1`
* `build_supermix_studio_desktop_installer.ps1`

Installer definitions live under `installer/`.

---

## Training and fine-tuning

The training/fine-tuning side of the project is centered in `source/`.

Important categories in the repo include:

* **fine-tuning scripts**

  * `finetune_chat.py`
  * `finetune_chat_manifest_sequential.py`

* **dataset builders**

  * `build_super_dataset.py`
  * `build_science_knowledge_dataset.py`
  * `build_science_novel_examples_dataset.py`
  * `build_reasoning_benchmix_v39.py`
  * additional domain-specific dataset builders

* **training manifests**

  * multiple `conversation_data.*.json` manifest files
  * curriculum, weighted, multimodal, smoke, and broad-manifest variants

* **distillation / specialist model work**

  * native-image distillation scripts
  * frontier model files
  * image-feature and mesh-feature utilities

This repo is therefore suitable both for **runtime usage** and for **iterating on training data, model variants, and evaluation flows**.

---

## Kaggle workflow

Kaggle support is documented in:

```text
KAGGLE_GUIDE.md
```

The repo also references a notebook-based Kaggle training path under:

```text
output/jupyter-notebook/supermix-kaggle-current-training.ipynb
```

The intent is to keep the cloud notebook workflow aligned with the local training pipeline while avoiding desktop-only components in the cloud environment.

---

## Architecture and model docs

For deeper technical context, start here:

* `ARCHITECTURE.md` — system and architecture notes
* `MODEL_CARD_V28.md` — model-card documentation for an earlier major line
* `KAGGLE_GUIDE.md` — cloud-training workflow
* `CONTRIBUTING.md` — contribution rules
* `SECURITY.md` — security process
* `CODE_OF_CONDUCT.md` — community expectations

---

## Versioning and naming note

This repository reflects an evolving experiment line rather than a perfectly clean single-version snapshot.

You will see references to several names and generations across the project, including:

* `Supermix_27`
* `Supermix_28`
* `Supermix_29`
* `v26`
* `v27`
* `v28`
* `v33`
* `v35`
* `v37`
* `v38`
* `v39`

That is expected.

Interpret the repo as a **living workspace** containing:

* packaged runtime assets from earlier stable lines
* documentation from earlier model generations
* newer research scripts and experimental files
* current snapshot packaging/build assets under the `Supermix_29` repo name

---

## Recommended entry points by goal

### I just want to run it

Use:

```bash
python runtime_python/chat_web_app.py
```

### I want to work on the current source app

Use:

```bash
python source/chat_web_app.py
```

### I want the browser-only version

Open:

```text
web_static/index.html
```

### I want to fine-tune or build datasets

Start in:

```text
source/
```

### I want to build a desktop app

Start with:

* root `.spec` files
* root PowerShell build scripts
* `installer/` `.iss` files

---

## Platform notes

* Windows appears to be the primary packaging target for the desktop workflow.
* The repo includes batch launchers, PowerShell build scripts, PyInstaller specs, and Inno Setup definitions.
* The project layout is mixed-purpose by design: research, runtime, packaging, and artifacts coexist in one repository.

---

## Contributing

Please read:

* `CONTRIBUTING.md`
* `CODE_OF_CONDUCT.md`
* `SECURITY.md`

before opening major pull requests or publishing derivative builds.

---

## License

See:

```text
LICENSE
```

---

## Summary

Supermix_29 is a **hybrid AI workspace** spanning:

* local chat runtime
* active source development
* static browser deployment
* desktop packaging
* dataset construction
* fine-tuning
* benchmarking
* experimental model lines

It is best approached as a **research-and-delivery monorepo** for the Supermix ecosystem.

```

A stronger follow-up would be a second pass that adds badges, a small architecture diagram, and a screenshots section while keeping this version as the clean baseline.
::contentReference[oaicite:2]{index=2}
```

[1]: https://github.com/kai9987kai/Supermix_29/tree/main "GitHub - kai9987kai/Supermix_29: Supermix v29 represents a significant architectural evolution of the ChampionNet series, introducing a Mixture-of-Experts (MoE) classifier head for enhanced retrieval and reasoning capabilities. The project is a structured AI/chat repository designed for high-performance inference, featuring a clear split between source code, runtime assets, and · GitHub"
[2]: https://github.com/kai9987kai/Supermix_29/blob/main/README.md "Supermix_29/README.md at main · kai9987kai/Supermix_29 · GitHub"
