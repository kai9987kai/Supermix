## About the Model (Supermix v28 UltraExpert)

Supermix_27 v28 represents a significant architectural evolution of the ChampionNet series, introducing a Mixture-of-Experts (MoE) classifier head for enhanced retrieval and reasoning capabilities. The project is a structured AI/chat repository designed for high-performance inference, featuring a clear split between source code, runtime assets, and web deployment bundles.

---

## Quick Start

### Run the Chat Web App (Real Neural Inference)
```bash
cd runtime_python
python chat_web_app.py
# Opens on http://localhost:8000
```

### Build the Desktop EXE (Embedded Webview + Auto Server Start)
```powershell
powershell -ExecutionPolicy Bypass -File source\build_qwen_chat_desktop_exe.ps1
dist\SupermixQwenDesktop\SupermixQwenDesktop.exe
```
- Launches the Qwen chat UI in its own desktop window.
- Starts the local Flask chat server automatically when the app opens.
- Bundles the latest adapter found under `artifacts/` into the build output.
- Uses the local Python installation to run `source/qwen_chat_web_app.py` behind the windowed launcher.
- Still expects the Qwen base model snapshot to exist locally unless you pass a different `--base_model` path.

### Fine-Tune a Model Variant
```bash
cd source
python finetune_chat.py \
    --data conversation_data.jsonl \
    --weights ../runtime_python/champion_model_chat_supermix_v27_500k_ft.pth \
    --model_size smarter_expert \
    --epochs 8 \
    --batch_size 32
```

---

## Project Structure

```
Supermix_27/
├── source/                          # Training & development
│   ├── run.py                       # ChampionNet backbone (12 layers, GatedFFN)
│   ├── model_variants.py            # All classifier head & backbone variants
│   ├── finetune_chat.py             # Main training script (DPO/SimPO, EMA, AMP)
│   ├── chat_pipeline.py             # Data loading, feature modes, label assignment
│   ├── device_utils.py              # Cross-platform device resolution
│   ├── benchmark.py                 # Model evaluation benchmarks
│   ├── chat_app.py                  # Terminal-based chat interface
│   ├── chat_web_app.py              # Web-based chat interface (dev)
│   ├── build_*.py                   # Dataset construction scripts
│   └── run_train_qwen_supermix_*.ps1  # Training launch scripts
│
├── runtime_python/                  # Self-contained inference runtime
│   ├── chat_web_app.py              # Production web server
│   ├── champion_model_chat_supermix_v27_500k_ft.pth  # Base checkpoint
│   └── ...                          # Runtime copies of pipeline files
│
├── web_static/                      # GitHub Pages compatible static UI
│   └── index.html                   # Glassmorphism chat interface
│
├── datasets/                        # Training datasets (JSONL)
├── artifacts/                       # Training outputs & checkpoints
├── MODEL_CARD_V28.md                # Model card with benchmarks & specifications
├── ARCHITECTURE.md                  # Technical architecture deep-dive
├── CONTRIBUTING.md                  # Contribution guidelines
├── README.md                        # This file
├── LICENSE                          # MIT License
├── CODE_OF_CONDUCT.md               # Community guidelines
└── SECURITY.md                      # Security policy
```

---

## Model Variants

| Variant | Head Class | Routing Strategy | Key Feature |
|---------|-----------|-----------------|-------------|
| `ultra_expert` | `GatedExpertClassifierHead` | Noisy Top-2 | 6 heterogeneous experts with noise-injected gating |
| `hierarchical_expert` | `HierarchicalMoEClassifierHead` | Two-level hierarchical | 2 domain groups × 4 experts + shared expert |
| `deep_expert` | `DeepExpertClassifierHead` | Top-2 + dynamic bias | Aux-loss-free load balancing |
| `expert_choice` | `ExpertChoiceClassifierHead` | Expert Choice (EC) | Experts pick tokens; guaranteed load balance |
| `smarter_expert` | `SmarterExpertClassifierHead` | Sigma Gating | Independent sigmoids + LoRA adapters |

For detailed architecture information, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Core Architecture

### ChampionNet Backbone
A 12-layer sequential model with `GatedFFN` feature extraction blocks (layers 0–9), a classifier head (layer 10, swapped per variant), and a final output layer (layer 11). Input dimensionality is 256 features, output is 10 classes.

### GatedExpertClassifierHead (v28 Primary)
- **6 Parallel Experts**: Each with a unique activation (SiLU, GELU, Mish, ReLU, SELU, Tanh) and increasing inner dimensions (1024–3584).
- **Noisy Top-K Gating**: Dynamically routes inputs to the top-2 most relevant experts with learnable noise for exploration.
- **Residual Calibration**: A learned linear transformation for logit refinement.

---

## Technical Specifications

| Specification | Value |
|---------------|-------|
| **Base Model** | v27 500k-FT |
| **Precision** | FP16/AMP Mixed Precision supported |
| **Gating** | Noisy Top-2 / Sigma / Expert Choice (per variant) |
| **Experts** | 6–8 (Heterogeneous Activations) |
| **Device Support** | CUDA, NPU, XPU, DML, MPS, CPU (auto-detect) |
| **Optimizer** | AdamW (lr=2e-4, wd=0.01) |
| **LR Schedule** | Cosine annealing with linear warmup |
| **EMA** | Decay 0.999 |

### Preliminary Benchmarks

*Coding Knowledge benchmark (380 samples, 2 epochs fine-tuning)*

| Variant | Accuracy |
|---------|----------|
| v27 Base | 22.63% |
| **v28 UltraExpert** | **19.47%** (Warm-up phase) |

*Note: v28 is currently in a warm-up phase; full convergence is expected to surpass v27 performance.*

---

## User Interface

The repository includes a **Premium Web UI** (`web_static/`) designed for a state-of-the-art experience:
- **Aesthetics**: Glassmorphism design with modern Inter typography.
- **Interaction**: Real-time inference timing and smooth micro-animations.
- **Style Modes**: Auto, Balanced, Creative, Concise, and Analyst.

### Static Web vs Real Model
- **`runtime_python/`**: Real neural inference using the PyTorch checkpoint.
- **`web_static/`**: GitHub Pages compatible static UI using metadata retrieval (does not run the `.pth` model in-browser).

---

## Documentation

| Document | Description |
|----------|-------------|
| [MODEL_CARD_V28.md](MODEL_CARD_V28.md) | Model specifications, benchmarks, and variant details |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Deep-dive into backbone, MoE routing, training pipeline |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines, code style, and testing |
| [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) | Community standards |
| [SECURITY.md](SECURITY.md) | Security vulnerability reporting |

---

## Contact & Links
- **Email**: [kai9987kai@gmail.com](mailto:kai9987kai@gmail.com)
- **GitHub**: [github.com/kai9987kai](https://github.com/kai9987kai)
- **Website**: [kai9987kai.co.uk](https://kai9987kai.co.uk)

## License
MIT License - Copyright (c) 2026 kai9987kai

---

## Qwen Training Additions (March 2026)

### Safer Resume Preference Training
```powershell
powershell -ExecutionPolicy Bypass -File source\run_train_qwen_supermix_v22_fullsmartcreative_resume_pref.ps1
```
- Uses dataset-only preference mining on CPU to avoid `model.generate()` stalls.
- Adds explicit mining progress logs and mining time/attempt guards.
- Adds counterfactual hard negatives for coding/reasoning prompts to sharpen preference learning.

### Larger Smart-Coder/Reasoner Profile
```powershell
powershell -ExecutionPolicy Bypass -File source\run_train_qwen_supermix_v23_larger_reasoner.ps1
```
- Targets larger base models (default `Qwen/Qwen2.5-1.5B-Instruct`).
- Uses stronger coding/reasoning-focused weighting and counterfactual hard negatives.

### Adaptive ORPO + Trust-Region Profile
```powershell
powershell -ExecutionPolicy Bypass -File source\run_train_qwen_supermix_v24_adaptive_orpo_trust.ps1
```
- Uses `--preference_objective orpo` with progressive `beta`/`margin` schedules.
- Adds trust-region style preference anchoring to reduce instability in late preference updates.
- Adds hardness-aware preference weighting to focus training on unresolved/hard pairs.

### Selective Preference Curation Profile (v25)
```powershell
powershell -ExecutionPolicy Bypass -File source\run_train_qwen_supermix_v25_selective_pref.ps1
```
- Adds post-mining preference pair selection inspired by recent preference-data selection papers.
- Uses difficulty-aware `capacity_aware` selection to keep informative pairs while reducing unstable extremes.
- Logs mined-vs-kept pair stats directly in training output for monitoring.

### GUI Training Monitor
```bash
python source/training_monitor_gui.py --root .
```
or:
```bash
source\launch_training_monitor_gui.bat
```
- Live status for all `train_*.out.log` runs: stage, SFT/Pref steps, latest loss/LR, PID, and stall detection.
