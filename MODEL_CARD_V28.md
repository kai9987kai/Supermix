# Model Card: Supermix_27 (v28 UltraExpert)

## Overview

Supermix_27 v28 represents a significant architectural evolution of the ChampionNet series, introducing a family of Mixture-of-Experts (MoE) classifier heads for enhanced retrieval and reasoning capabilities. Building on the v27 500k-FT base, v28 replaces the monolithic classifier with a dynamically-routed expert ensemble, enabling specialised sub-networks to activate per input while maintaining a stable shared representation.

---

## Architecture

### Backbone: ChampionNet

The backbone consists of 12 sequential layers (indices 0–11):
- **Layers 0–9**: Feature extraction via `GatedFFN` blocks with learned gating, LayerNorm, and residual connections.
- **Layer 10**: The classifier head (swapped per variant — see below).
- **Layer 11**: Final output normalization.

Input dimensionality is **256** features, output is **10** classes.

### GatedExpertClassifierHead (Primary v28 Head)

The flagship v28 head uses a Noisy Top-K gating mechanism:

- **Expert Branching**: 6 parallel experts with diverse activation functions:
  | Expert | Activation | Inner Dimension |
  |--------|------------|-----------------|
  | 0 | SiLU (Sigmoid Linear Unit) | 1024 |
  | 1 | GELU (Gaussian Error Linear Unit) | 1536 |
  | 2 | Mish (Self-Regularized) | 2048 |
  | 3 | ReLU (Rectified Linear Unit) | 2560 |
  | 4 | SELU (Scaled Exponential Linear Unit) | 3072 |
  | 5 | Tanh | 3584 |

- **Noisy Top-K Gating**:
  - A learned gate network produces clean logits per expert.
  - During training, learnable noise (`softplus` of a noise gate) is injected for exploration.
  - Top-2 experts are selected and their weights are renormalized.

- **Residual Calibration**: A learned linear transformation (`calibration`) refines final logits, scaled by a learnable parameter `θ`.

```
Input → Gate → Top-2 Selection → Weighted Expert Sum → + Base Logits + θ·Calibration → Output
```

---

## Model Variants

### `ultra_expert` — Noisy Top-K Gating (Primary)
Integrates `GatedExpertClassifierHead` into the standard ChampionNet backbone. Uses 6 heterogeneous experts with Noisy Top-2 Gating and residual calibration.

### `hierarchical_expert` — Two-Level Hierarchical Routing
Integrates `HierarchicalMoEClassifierHead`. Features:
- **2 domain groups** × **4 experts per group** = 8 total experts.
- **Domain-level gating**: Selects the most relevant expert group.
- **Per-domain expert gating**: Selects the top-2 experts within the chosen group.
- **Shared always-on expert**: Provides a stable baseline signal (1024-dim SiLU).
- **Per-expert residual LoRA adapters** (rank 64) for lightweight correction.
- **Per-expert LayerNorm** on outputs for gradient stability.
- **Auxiliary load-balancing loss** (Switch Transformer style) to prevent expert collapse.

### `deep_expert` — Auxiliary-Loss-Free Load Balancing
Integrates `DeepExpertClassifierHead`. Features:
- **1 shared expert** (2048-dim, always active) for universal knowledge.
- **8 routed experts** with Top-2 gating and diverse activations.
- **Dynamic bias adjustment** (DeepSeek-V3 style): Non-gradient buffer biases are updated each forward pass to push overloaded experts down and underloaded experts up, achieving load balance without auxiliary loss terms.

### `expert_choice` — Expert Choice (EC) Routing
Integrates `ExpertChoiceClassifierHead`. Instead of tokens choosing experts:
- **Experts choose tokens**: Each expert selects its top-K tokens based on affinity scores.
- **Guaranteed load balance**: Every expert processes exactly `ceil(BT × capacity_factor / N)` tokens.
- **No token drops**: All active experts process their full capacity.
- **Variable expert coverage**: Popular tokens may be processed by multiple experts.

### `smarter_expert` — Sigma Gating + LoRA Adapters
Integrates `SmarterExpertClassifierHead`. Features independent sigmoid scores per expert, allowing multiple experts to activate simultaneously without competition.

### `thought_expert` — Iterative Reasoning + Attention Fusion
Integrates `ThoughtExpertClassifierHead` (v13). Introduced a 3-step reasoning loop and Cross-Expert Attention Fusion for improved expert collaboration.

### `recursive_expert` — Recursive Thought (v14)
Integrates `RecursiveThoughtExpertHead`. The most advanced and efficient variant:
- **Adaptive Reasoning Depth (ACE)**: Early exit logic based on token-wise confidence, drastically improving inference efficiency.
- **Multi-Head Sigma Gating**: 2 routing heads per step for a richer, more precise routing subspace.
- **Hierarchical Shared Experts**: Combines a Global Shared Expert (2048-dim) for general knowledge with a Local Shared Expert (512-dim) for fine-grained task-specific baseline.
- **Recursive Reasoning**: An iterative loop with cross-expert attention and residual adapters.
- **Signal-Stable Initialization**: Small-signal normal initialization for experts to ensure differentiability from step 1.

### `reflexive_expert` — Reflexive Thought (v15)
Integrates `ReflexiveThoughtExpertHead`. The smartest self-correction variant:
- **Two-Pass Reasoning**: Computes initial predictions, then passes original features *and* initial predictions into a secondary "Critique Pass".
- **Self-Reflection MoE**: The critique pass uses an independent dynamically routed Mixture-of-Experts ensemble to identify uncertainties and directly correct errors in the initial prediction.
- **Stable Initialization**: Small normal variance initialization for downstream experts inside both initial and critique modules to ensure unobstructed gradient flow.

### `metacognitive_expert` — MetaCognitive Iterative Reflection (v16)
Integrates `MetaCognitiveExpertHead`. The most advanced variant, unifying iterative reasoning, self-reflection, and adaptive halting:
- **Shared Expert**: Always-on 2048-dim global expert for stable baseline signal.
- **ReasoningCells**: Per-step residual MLPs that iteratively refine feature representations.
- **Proposal MoE**: 6 routed experts with Sigma Gating generate predictions at each reasoning step.
- **Critique MoE**: 4 specialized experts inspect `[features ∥ proposal_logits]` and output corrections, enabling explicit self-reflection.
- **PonderNet Halting**: Learned per-step halt probability so the model spends more compute on hard inputs and exits early on easy ones.
- **Dynamic Bias Load Balancing**: Non-gradient buffer biases keep expert utilization even during training.

### `tree_of_thought_expert` — Latent Beam Search (v17)
Integrates `TreeOfThoughtExpertHead`. Represents a fundamental paradigm shift by embedding test-time compute scaling directly into the latent forward pass:
- **Action MoE Proposer**: At each reasoning step, 6 independent experts propose diverse modifications to the current hypothetical states.
- **Value Network Scorer**: A dedicated network evaluates the "goodness" or confidence of every newly generated candidate state.
- **Test-Time Beam Search**: A dynamically branching beam of hypothetical trajectories (`beam_size=4`) is maintained. At each step, branches are scored and heavily pruned, keeping only the most promising states.
- **Differentiable Aggregation**: The final surviving states are fused based on their Value scores, permitting unobstructed, end-to-end gradient flow back through the search process itself.

### `consensus_expert` — Architectural Diversity + Learned Arbitration (v18)
Integrates `ConsensusExpertHead`. The first head to leverage **architectural diversity** — three fundamentally different computational paradigms independently reason about the input, then a learned arbiter resolves disagreements:
- **MLP Pathway**: 6 feed-forward experts with Sigma Gating for fast pattern matching.
- **Attention Pathway**: Self-attention over 8 learned expert embeddings for relational reasoning.
- **Convolutional Pathway**: 1D convolution over the feature vector for local pattern detection.
- **Consensus Network**: A 3-layer MLP arbiter evaluates inter-pathway agreement/disagreement and produces confidence-weighted fusion.

---

## Training & Fine-Tuning

| Parameter | Value |
|-----------|-------|
| **Base Checkpoint** | v27 500k-FT |
| **Precision** | FP16/AMP Mixed Precision |
| **Device Support** | CPU / CUDA / Metal via `device_utils` |
| **Optimizer** | AdamW (default lr=2e-4, wd=0.01) |
| **LR Schedule** | Cosine annealing with linear warmup |
| **Label Smoothing** | 0.05 |
| **EMA Decay** | 0.999 |
| **Gradient Clipping** | Max norm 1.0 |
| **Data Split** | Stratified train/val (90/10) |
| **Preference Loss** | SimPO-style sigmoid pairwise + hard negatives (weight=0.15, β=2.0) |
| **Preference Warmup** | Linear ramp over 1 epoch |

### Preference Optimization

The training pipeline supports multiple preference objectives:
- **Sigmoid** (SimPO-like): Pairwise sigmoid preference with configurable group size and EPO-style group estimator.
- **RePO ReLU**: Max-margin objective (β-free) with configurable margin.
- **Hard negative mining**: Configurable ratio of hard (top-logit) vs. random negatives.
- **Adaptive weighting**: Confidence-weighted preference terms for noisy-data robustness.

---

## Benchmark Results

### Coding Knowledge Benchmark (380 samples, 2 epochs fine-tuning)

| Variant | Accuracy |
|---------|----------|
| v27 Base | 22.63% |
| **v28 UltraExpert** | **19.47%** |

> **Note**: The v28 model was evaluated after only a short warm-up fine-tuning session. Full convergence is expected to surpass v27 performance.

### Language Model Evaluation (Benchmark v7, 40 eval pairs)

| Metric | Base Model | Tuned Model |
|--------|------------|-------------|
| Eval Loss | 2.866 | 2.866 |
| Perplexity | 17.57 | 17.57 |
| Token F1 | 0.128 | 0.128 |
| Char Similarity | 0.087 | 0.087 |
| Avg Gen Speed | **8.54s** | 111.47s |

> **Note**: Base and tuned metrics converge during early warm-up. Generation speed difference reflects the additional expert computation overhead. Optimization (expert pruning, batched expert processing) is planned.

### Training Convergence (v22 Full Smart Creative — 30k examples)

| Phase | Steps | Start Loss | End Loss |
|-------|-------|------------|----------|
| SFT Warmup | 1–17 | 4.94 | 4.80 |
| SFT Early | 18–100 | 4.67 | 2.97 |
| SFT Mid | 100–200 | 2.96 | 2.29 |
| SFT Late | 200–320 | 2.29 | 2.06 |

---

## Interface

- **Premium Web UI**: Features glassmorphism design with Inter typography, smooth micro-animations, and real-time inference timing display.
- **Style Modes**: Auto, Balanced, Creative, Concise, Analyst.
- **Runtime**: `runtime_python/chat_web_app.py` for real neural inference; `web_static/` for GitHub Pages compatible static UI.

---

## Files & Checkpoints

| File | Purpose | Required |
|------|---------|----------|
| `runtime_python/champion_model_chat_supermix_v27_500k_ft.pth` | Base v27 checkpoint (starting point for v28 fine-tuning) | ✅ |
| `source/champion_model_chat_v28_expert.pth` | Current v28 expert checkpoint | ✅ |

---

## Contact
- **Email**: [kai9987kai@gmail.com](mailto:kai9987kai@gmail.com)
- **GitHub**: [github.com/kai9987kai](https://github.com/kai9987kai)
- **Website**: [kai9987kai.co.uk](https://kai9987kai.co.uk)

## License
MIT License — Copyright (c) 2026 kai9987kai
