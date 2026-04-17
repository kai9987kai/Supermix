# Model Card: Supermix v27 (Champion Model)

## 1) Architecture Summary
The model follows a hybrid architecture combining multi-head linear attention, selective scan state-space models (Mamba), and a highly expanded mixture-of-experts-style classifier head.

### Backbone: `ChampionNet`
The backbone consists of a sequence of specialized layers designed for efficiency and representational power:
1. **Input Projection**: 128 -> 256
2. **MHLA (Multi-Head Linear Attention)**: 8 heads, d_model=256. Uses a low-rank (LoRA) bottleneck for query and key/value projections.
3. **Mamba Selective Scan**: Selective SSM with d_inner=512 and d_state=16. Includes a post-convolution layer.
4. **Sparsity & Pruning**: 
   - `TopKSparse`: Keeps top-32 absolute features.
   - `SaliencyPrune`: Prunes 20% of features based on a learned saliency predictor.
5. **RevRes (Reversible Residuals)**: Splits 256 dimensions into 128/128 for reversible processing.
6. **MoA (Mixture-of-Attention)**: 4 experts operating on 64-dimensional chunks with learned routing.
7. **BitNet Linear**: A normalized linear layer designed for high-capacity representation.

### Head: `SmarterClassifierHead`
An upgraded classifier head with advanced routing and domain calibration:
- **Multi-Branch Adapters**: 4 parallel branches with different activation functions (SiLU, GELU, Mish, ReLU) and expansion dimensions (1024, 2048, 3072, 4096).
- **Domain Calibration**: Learned routing across 4 domain experts (Science, Code, Math, Literature).
- **Reasoning Gate**: A dedicated branch to amplify logit differences for sharper choice selection.

## 2) Metadata & Training
- **Checkpoint**: `champion_model_chat_supermix_v27_500k_ft.pth`
- **Total Parameters**: 3,616,613
- **Feature Dimension**: 128 (input), 256 (inner)
- **Fine-Tuning Data**: `conversation_data.supermix_plus_v27_500k.jsonl` (509,126 examples)
- **Training Objective**: Cross-Entropy + Preference Loss (Sigmoid objective with EPO group estimator)
- **Validation Accuracy**: 73.55% (Label mode: kmeans_10)

## 3) Tokenization & Featurization
The model uses a **token-free, hash-based featurizer** (`chat_pipeline.py`):
- **Hashing**: Blake2b stable hashing of unigrams and bigrams.
- **Domain Hints**: Explicit regex-based bucketing for code, science, math, etc.
- **Context Encoding**: Recency-weighted conversation history encoding (up to 48 lines).

## 4) Usage
- **Primary Runtime**: `runtime_python/chat_web_app.py`
- **Inference Requirements**: CPU-compatible (CUDA/NPU preference supported), ~15MB weights + ~9MB metadata.
