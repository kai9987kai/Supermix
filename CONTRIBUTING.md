# Contributing to Supermix

Thank you for contributing to Supermix. This guide covers the active source,
legacy compatibility, test, and packaging contracts.

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **PyTorch 2.0+** (with CUDA support recommended but not required)
- Install dependencies:
  ```bash
  pip install -r source/requirements_train_build.txt    # For training & building
  pip install -r source/requirements_runtime_interface.txt  # For inference only
  ```

### Repository Structure

```
Supermix/
├── source/                     # Training, model definitions, dataset builders
│   ├── run.py                  # ChampionNet backbone definition
│   ├── model_variants.py       # All classifier head variants
│   ├── finetune_chat.py        # Main training script
│   ├── chat_pipeline.py        # Data loading & feature extraction
│   ├── device_utils.py         # Cross-platform device selection
│   ├── benchmark.py            # Evaluation benchmarks
│   └── build_*.py              # Dataset construction scripts
├── runtime_python/             # Legacy self-contained compatibility runtime
├── source/supermix_multimodel_web_app.py     # Active Studio web runtime
├── source/supermix_multimodel_desktop_app.py # Active packaged Studio entrypoint
├── web_static/                 # GitHub Pages static UI
├── MODEL_CARD_V28.md           # Model card with benchmarks
├── ARCHITECTURE.md             # Technical architecture deep-dive
└── README.md                   # Project overview & quick start
```

---

## How to Contribute

### Adding a New MoE Variant

1. **Define the classifier head** in `source/model_variants.py`:
   - Subclass `nn.Module`.
   - Maintain backward-compatible weight keys (`weight`, `bias`) from the base linear layer.
   - Initialize new learnable scales to **0** (for stable warm-start).
   - Implement `reset_parameters()` with appropriate initialization.

2. **Create a backbone wrapper** (e.g., `ChampionNetMyVariant`):
   - Use `ChampionNet()` to get the base layers.
   - Replace `layers[10]` with your new head.
   - Keep `layers[11]` (final normalization) intact.

3. **Register in `build_model()`**: Add your variant name to the dispatch logic.

4. **Register in `finetune_chat.py`**: Add your variant to the `--model_size` choices.

5. **Refresh the standalone runtime snapshot**:

   ```bash
   python source/sync_runtime_model_variants.py
   python source/sync_runtime_model_variants.py --check
   ```

6. **Update documentation**: Add your variant to `MODEL_CARD_V28.md` and `ARCHITECTURE.md`.

### Adding a New Dataset

1. Create `source/build_<dataset_name>_dataset.py` following existing patterns.
2. Output format: JSONL with `{"prompt": "...", "response": "..."}` entries.
3. Add the dataset path to a manifest JSON for multi-shard training.

### Bug Fixes & Improvements

1. Fork the repository.
2. Create a feature branch: `git checkout -b fix/description`.
3. Make your changes with clear commit messages.
4. Ensure all existing functionality still works.
5. Submit a pull request with a description of what changed and why.

---

## Code Style

### Python

- **Docstrings**: Use Google-style docstrings for all public classes and functions.
- **Type hints**: Use type annotations for function signatures.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes.
- **Line length**: 120 characters max.
- **Imports**: Group as stdlib → third-party → local, separated by blank lines.

### Markdown

- Use ATX-style headers (`#`, `##`, `###`).
- Use tables for structured data.
- Use fenced code blocks with language identifiers.
- Wrap lines at 120 characters where practical.

---

## Commit Message Convention

```
<type>(<scope>): <short description>

<body - what and why, not how>
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`

**Examples**:
```
feat(model): add SmarterExpertClassifierHead with Sigma Gating
fix(training): correct preference loss warmup calculation
docs(model_card): fill in v28 benchmark results
```

---

## Testing

The project has a formal pytest suite and focused runtime quality workflow. When contributing:

1. Run the focused tests for the files you changed, then run the full suite:

   ```bash
   python -m pytest -q
   ```

2. For active Studio runtime or route-control changes, verify the checked
   distribution contract and generated model-variant snapshot:

   ```bash
   python source/generate_studio_runtime_manifest.py --check
   python source/sync_runtime_model_variants.py --check
   ```

3. **Smoke test** training-specific changes with a short training session:
   ```bash
   python source/finetune_chat.py --data source/conversation_data.smoke_manifest.json \
       --data_manifest source/conversation_data.smoke_manifest.json \
       --model_size <your_variant> --epochs 1 --batch_size 32
   ```

4. **Verify weight loading** from existing checkpoints to ensure backward compatibility.

5. **Run benchmarks** if applicable:
   ```bash
   python source/benchmark.py
   ```

---

## Contact

- **Maintainer**: kai9987kai
- **Email**: kai9987kai@gmail.com
- **GitHub**: [github.com/kai9987kai](https://github.com/kai9987kai)
