# Architecture Guide: Supermix_27

This document provides an in-depth technical overview of the Supermix_27 model architecture, training pipeline, and design decisions.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [ChampionNet Backbone](#championnet-backbone)
3. [Classifier Head Evolution](#classifier-head-evolution)
4. [MoE Routing Mechanisms](#moe-routing-mechanisms)
5. [Training Pipeline](#training-pipeline)
6. [Inference Runtime](#inference-runtime)
7. [Key Design Decisions](#key-design-decisions)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Supermix_27 System                       │
├──────────────────┬──────────────────┬───────────────────────┤
│   source/        │  runtime_python/ │  web_static/          │
│   Training &     │  Real Neural     │  Static GitHub        │
│   Fine-tuning    │  Inference       │  Pages UI             │
├──────────────────┴──────────────────┴───────────────────────┤
│                  ChampionNet Backbone                        │
│            (12 layers, 256-dim features, 10 classes)         │
├─────────────────────────────────────────────────────────────┤
│              MoE Classifier Head (Layer 10)                  │
│  GatedExpert │ Hierarchical │ Deep │ ExpertChoice │ Smarter │
└─────────────────────────────────────────────────────────────┘
```

The project is split into three main deployment surfaces:
- **`source/`**: All training scripts, model definitions, dataset builders, and benchmarks.
- **`runtime_python/`**: A self-contained copy of the inference-critical files with the production checkpoint.
- **`web_static/`**: A static HTML/JS interface for GitHub Pages deployment (metadata retrieval only, no `.pth` inference).

---

## ChampionNet Backbone

Defined in [`source/run.py`](source/run.py), the `ChampionNet` is a 12-layer sequential model:

```python
class ChampionNet(nn.Module):
    # layers[0..9]:  GatedFFN feature extraction blocks
    # layers[10]:    Classifier head (swapped per variant)
    # layers[11]:    Final normalization / output projection
```

### GatedFFN Block

Each feature extraction layer is a `GatedFFN` — a feedforward block with learned gating:

```
Input x ──┬── Linear(256 → hidden) → Activation ──┐
           │                                        ├── Element-wise multiply → Linear(hidden → 256)
           └── Linear(256 → hidden) → Sigmoid ─────┘
```

Key properties:
- **Residual connections** preserve gradient flow through the 10-layer stack.
- **LayerNorm** is applied at each stage.
- **Multiple activation functions** (SiLU, GELU) are used across layers for representational diversity.

---

## Classifier Head Evolution

The project has evolved through several classifier head architectures, each building on the last:

| Generation | Head Class | Key Innovation |
|-----------|-----------|----------------|
| v1 | `nn.Linear` (base) | Simple linear classifier |
| v2 | `ExpandedClassifierHead` | Adapter branch with learnable `α` scale |
| v3 | `ExpandedClassifierHeadXL` | Dual adapters + learned router |
| v4 | `ExpandedClassifierHeadXXL` | Tri-branch + second-stage routing |
| v5 | `ExpandedClassifierHeadXXXL` | Quad-branch + third-stage routing |
| v6 | `ExpandedClassifierHeadUltra` | Five branches + domain-expert calibration |
| v7 | `ExpandedClassifierHeadMega` | Six branches + cross-attention fusion + reasoning gate |
| **v8** | **`GatedExpertClassifierHead`** | **MoE with Noisy Top-K Gating** |
| **v9** | **`HierarchicalMoEClassifierHead`** | **Two-level hierarchical routing + shared expert** |
| **v10** | **`DeepExpertClassifierHead`** | **Aux-loss-free dynamic bias load balancing** |
| **v11** | **`ExpertChoiceClassifierHead`** | **Expert Choice routing (experts pick tokens)** |
| **v12** | **`SmarterExpertClassifierHead`** | **Sigma Gating + LoRA adapters** |
| **v13** | **`ThoughtExpertClassifierHead`** | **Iterative reasoning loop + cross-expert attention fusion** |
| **v14** | **`RecursiveThoughtExpertHead`** | **Adaptive Reasoning Depth (ACE) + Multi-Head Sigma Gating** |

> **Note**: All heads preserve backward-compatible weight keys (`weight`, `bias`, `alpha`, etc.) to enable warm-starting from earlier checkpoints.

---

## MoE Routing Mechanisms

### 1. Noisy Top-K Gating (`GatedExpertClassifierHead`)

```
x → Gate(x) → clean_logits
         └──→ NoiseGate(x) → softplus → noise_std
              noise = randn() × noise_std          (training only)
              gate_logits = clean_logits + noise
              weights = softmax(gate_logits)
              top_weights, top_idx = topk(weights, k=2)
              top_weights = normalize(top_weights)
```

**Why noisy gating?** Injecting learnable noise during training encourages exploration of under-used experts and prevents mode collapse where only 1–2 experts receive all traffic.

### 2. Hierarchical Two-Level Routing (`HierarchicalMoEClassifierHead`)

```
Level 1 (Domain):   x → DomainGate(x) → top-1 domain group
Level 2 (Expert):   x → ExpertGate[domain](x) → top-2 experts within group
Shared Expert:      x → SharedUp → SiLU → SharedDown → LayerNorm  (always active)
```

This design is inspired by **DeepSeek-MoE** and **ST-MoE**. The two-level hierarchy reduces the routing search space (2 groups × 4 experts vs. 8 flat experts) and enables domain specialization.

### 3. Auxiliary-Loss-Free Load Balancing (`DeepExpertClassifierHead`)

Traditional MoE uses an auxiliary loss term to prevent expert collapse:
```
L_aux = N × Σ(fraction_dispatched × avg_gate_prob)
```

The `DeepExpertClassifierHead` instead uses **dynamic bias adjustment** (inspired by DeepSeek-V3):
```python
# Non-gradient buffer, updated each forward pass:
expert_load = fraction_of_tokens_each_expert_received  # shape: (N,)
target_load = top_k / n_experts
expert_bias += α_bias × (target_load - expert_load)
```

This avoids interference with the primary training objective.

### 4. Expert Choice Routing (`ExpertChoiceClassifierHead`)

**Reverses the routing direction**: instead of tokens selecting experts, experts select tokens.

```
Affinity = softmax(Gate(x), dim=experts)    # (BT, N)
For each expert i:
    top_tokens = topk(Affinity[:, i], k=capacity)
    output[top_tokens] += Expert_i(x[top_tokens]) × Affinity[top_tokens, i]
```

**Guarantees**:
- Every expert processes exactly `ceil(BT × capacity_factor / N)` tokens.
- No token drops (unlike traditional top-K where overflow tokens are discarded).
- Popular tokens may be processed by multiple experts (variable compute).

### 5. Sigma Gating (`SmarterExpertClassifierHead`)

Replaces competitive softmax routing with **independent sigmoid scores**:

```python
gate_scores = sigmoid(Gate(x) + expert_bias)  # shape: (BT, N)
# Each score ∈ [0, 1] independently — multiple experts can fully activate
routed_out = Σ(expert_i(x) × gate_scores[:, i])
```

**Key difference from softmax**: Experts don't compete. A token can activate all 8 experts at full strength if the gate deems it beneficial — useful for complex queries that benefit from diverse perspectives.

### 6. Multi-Head Recursive Thought (`RecursiveThoughtExpertHead`)

The flagship v14 head introduces **Recursive Thought** with **Adaptive Reasoning Depth (ACE)**:

```python
# Iterative Reasoning Loop (3 steps)
for step in range(3):
    # 1. Refine features
    current_features = ReasoningCell(current_features)
    
    # 2. Multi-Head Sigma Gating
    # Each head (2 per step) provides a different routing perspective
    gate_logits = mean([gate_h(current_features) for gate_h in range(n_heads)])
    gate_scores = sigmoid(gate_logits)
    
    # 3. ACE: Dynamic Early Exit
    exit_prob = sigmoid(exit_gate(current_features))
    
    # 4. Expert Fusion
    step_out = Σ(expert(x) × gate_scores)
    total_routed_out += (1.0 - cumulative_exit_prob) * step_out
    
    # Update exit probability
    cumulative_exit_prob += (1.0 - cumulative_exit_prob) * exit_prob
    
    if cumulative_exit_prob > threshold: break # Inference only
```

**Key Innovations**:
- **Multi-Head Routing**: Using 2 heads per step creates a "routing committee", reducing noise and improving specialization.
- **ACE Efficiency**: Tokens with clear classification paths exit early, saving compute. Complex tokens use the full 3-step recursive capacity.
- **Signal-Stable Init**: Experts use `nn.init.normal_(std=0.01)` instead of zero to ensure non-zero gradient paths from initialization.

---

## Training Pipeline

### Data Flow

```
JSONL datasets → load_conversation_examples() → assign_labels()
    → build_training_tensors(feature_mode) → stratified_split()
    → DataLoader (optional WeightedRandomSampler for class balance)
```

### Feature Modes

| Mode | Description |
|------|-------------|
| `legacy` | Original feature encoding |
| `context_v2` | Context-aware encoding (default) |
| `context_v3`–`v5` | Progressive context improvements |
| `context_mix_v1` | Mixed context features |
| `context_mix_v2_mm` | Multimodal mixed context |

### Loss Function

The total loss combines cross-entropy with preference optimization:

```
L_total = L_CE(label_smoothed) + pref_scale × pref_weight × L_pref + aux_weight × L_aux
```

Where:
- **L_CE**: Standard cross-entropy with configurable label smoothing.
- **L_pref**: Pairwise preference loss (SimPO sigmoid or RePO ReLU margin).
- **L_aux**: MoE auxiliary load-balancing loss (for variants that use it).
- **pref_scale**: Linear warmup from 0 → 1 over configurable epochs.

### Optimization

- **Optimizer**: AdamW with configurable weight decay.
- **LR Schedule**: Cosine annealing with configurable linear warmup.
- **EMA**: Exponential moving average of model weights (decay 0.999) for smoother evaluation.
- **Gradient Clipping**: Max norm clipping (default 1.0) for training stability.
- **Mixed Precision**: AMP GradScaler for CUDA devices.

---

## Inference Runtime

### Local Python Inference

```bash
python runtime_python/chat_web_app.py
# Loads champion_model_chat_supermix_v27_500k_ft.pth
# Starts a Flask/Tornado web server on port 8000
```

### Device Resolution Order

The `device_utils.resolve_device()` function auto-selects the best available device:
```
cuda → npu → xpu → dml → mps → cpu
```

### Static Web Deployment

The `web_static/` directory contains a GitHub Pages-compatible interface that uses precomputed metadata (JSON) for chat retrieval without running the neural model in-browser.

---

## Key Design Decisions

### Why Heterogeneous Expert Activations?

Each expert uses a different activation function (SiLU, GELU, Mish, ReLU, SELU, Tanh, etc.) to encourage **functional diversity**. Research shows that homogeneous experts tend to converge to similar representations, reducing the benefit of the MoE architecture.

### Why Learnable Scale Parameters (α, θ, etc.)?

All new architectural components are gated by a **learnable scalar initialized to 0**. This ensures:
1. **Stable warm-start**: Loading a base checkpoint reproduces the original model's behavior exactly.
2. **Gradual adaptation**: The model learns how much to rely on new components during fine-tuning.

### Why Per-Expert LoRA Adapters?

LoRA adapters (low-rank matrices) add lightweight correction capacity to each expert without dramatically increasing parameter count. This is especially effective for fine-tuning on domain-specific data where each expert may need slight behavioral adjustments.

### Why Dynamic Bias Instead of Auxiliary Loss?

Auxiliary load-balancing losses (like the Switch Transformer loss) interfere with the primary training objective and require careful tuning of the loss weight. Dynamic bias adjustment:
- Operates outside the gradient graph (no interference with backprop).
- Self-corrects in real-time based on actual expert utilization.
- Requires no hyperparameter tuning beyond the bias learning rate.
