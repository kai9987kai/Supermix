# Supermix v72 — NexusMind Unified Hybrid Thinking Architecture

> **Historical architecture proposal; superseded by the current evidence-first
> contract.** “Production-grade,” context-window, live telemetry, adaptive
> learning, verification, and performance language below describes the original
> design intent, not confirmed current Nexus capability. The default neural path
> has no loaded text-generation checkpoint; swarm and graph outputs are
> deterministic analysis scaffolds; checksums are audit metadata rather than
> factual authority. See
> [`NEXUS_EVIDENCE_FIRST_SELECTIVE_ANSWERING.md`](NEXUS_EVIDENCE_FIRST_SELECTIVE_ANSWERING.md).

## Overview

Supermix v72 ("NexusMind") is the grand synthesis of three previously disparate research lineages into one unified, CPU-runnable, production-grade hybrid thinking system:

| Lineage | Technological Innovations Synthesized |
| :--- | :--- |
| **Xiaomi MiMo** (V2-Flash, V2.5-Pro) | Hybrid Sliding-Window Attention (SWA:GA 5:1/6:1 ratio) with bounded KV-cache reduction (~6–7x), learnable attention sinks per head, auxiliary-loss-free sparse MoE load balancing with router z-loss, Multi-Token Prediction (MTP) self-speculative draft decoding, decoupled dual-base RoPE tables (YaRN/NTK up to 1M context), and Flash vs Pro adaptive router. |
| **Supermix Cognition Stack** (v51–v71) | Weight-tied recursive latent ACT refinement with ponder cost halting, supervised quality & continue verifier, cross-budget ordered top-k agreement with exact output reuse, v56 latent state machine with row-stochastic log-space transition matrices, v70 multi-domain sparse expert routing, and v71 deterministic closed-world scientific scenario solver with exact rational SI arithmetic and cryptographic answer receipts. |
| **AI-Dem-Lab & NexusMind Systems** | 5-Agent Cognitive Swarm (Generator, Critic, Skeptic, Archivist, Anomaly Hunter) with discrete replicator dynamics, Graph-of-Thoughts (GoT) multi-branch speculative search with prune-and-merge graph topology, closed-loop Tabular Q-Learning budget adaptation (`BudgetPolicyLearner`), and the complete Dem-Lab statistical telemetry battery (Shannon/min-entropy, chi-square uniformity p-values, runs & monobit tests, CHSH Bell inequality validation, and RSI momentum meters). |

---

## Architectural Pillars

### 1. Hybrid Attention & Attention Sinks (MiMo Lineage)
Layers alternate between local Sliding Window Attention (SWA, window size $W=128$) and Global Attention (GA) at ratio $r=5$.
$$\text{Memory Reduction} \approx \frac{L \cdot N}{(L / (r + 1)) \cdot N + (L \cdot r / (r + 1)) \cdot W} \approx 6.2\times$$
Each attention head incorporates a learnable sink logit $s_h$, preventing forced probability mass normalisation over the initial token:
$$\alpha_{i,j} = \frac{\exp(q_i k_j^\top / \sqrt{d})}{\exp(s_h) + \sum_{m} \exp(q_i k_m^\top / \sqrt{d})}$$

### 2. Auxiliary-Loss-Free Sparse MoE (MiMo Lineage)
Experts are selected via score plus dynamic bias:
$$\text{selected} = \operatorname{TopK}(s_i + b_i, k=2)$$
Weights are applied using raw scores $s_i$ without gradient distortion, and biases evolve via sign-based load feedback:
$$b_i^{(t+1)} = b_i^{(t)} + \gamma \cdot \operatorname{sign}(\bar{L} - L_i)$$

### 3. 5-Agent Cognitive Swarm (AI-Dem-Lab Lineage)
Five specialized agents deliberate in structured rounds:
- **Generator**: Formulates the initial hypothesis and deductive pathway.
- **Critic**: Analyzes deductive validity, non-contradiction, and formal entailment.
- **Skeptic**: Stress-tests assumptions, counter-examples, and edge failure modes.
- **Archivist**: Enforces grounding against known definitions and prompt constraints.
- **Anomaly Hunter**: Screens for statistical outliers and token probability divergence.

Agent weights evolve across rounds via discrete replicator dynamics:
$$w_i^{(t+1)} = \frac{w_i^{(t)} \left(1 + \eta \left(f_i - \bar{f}\right)\right)}{\sum_j w_j^{(t)} \left(1 + \eta \left(f_j - \bar{f}\right)\right)}$$

### 4. Graph-of-Thoughts (GoT) Search (NexusMind Lineage)
Maintains a directed reasoning graph of thought nodes $(V, E)$:
- **Branch Expansion**: Multi-draft generation along promising directions.
- **Dynamic Pruning**: Nodes with score $S(v) < \tau_{\text{prune}}$ are deactivated.
- **Speculative Merging**: Complementary high-scoring leaves are merged into synthesis nodes:
  $$v_{\text{merged}} = \operatorname{Merge}(v_1, v_2), \quad S(v_{\text{merged}}) = \frac{S(v_1) + S(v_2)}{2} + \delta$$

### 5. Verified Scientific Scenarios (Supermix v71 Lineage)
Resolves closed-world kinematics and ideal-gas physics requests using exact rational SI arithmetic (`Fraction` / `Decimal`), SHA-256 span bindings, and zero runtime `eval`/`exec`.

---

## Module Map

```text
source/
├── mimomix_core.py          MiMo neural core: Hybrid SWA/GA, Attention Sinks, MoE, MTP
├── mimomix_controller.py    Adaptive budget planner with task difficulty & risk floors
├── mimomix_decoding.py      MTP self-speculative draft decoding
├── mimomix_reasoner.py      v56 Latent state machine reasoner with row-stochastic transitions
├── mimomix_observatory.py   Dem-Lab statistical telemetry & Tabular Q-learning
├── science_plan.py          v71 Deterministic closed-world physics & chemistry solver
├── nexus_swarm.py           5-Agent Cognitive Swarm engine with Replicator Dynamics
├── nexus_got.py             Graph-of-Thoughts reasoning engine & tree search
├── nexus_engine.py          Unified NexusMind master orchestrator
├── nexus_api.py             Production FastAPI / Starlette / ASGI service
└── nexus_cli.py             Interactive command-line terminal client

web_static/
└── nexus_studio.html        Single-file reactive web studio with live visualizers
```

---

## API Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/v1/think` | POST | Universal thinking endpoint with Flash vs Pro routing & CoT steps |
| `/v1/swarm` | POST | 5-Agent Cognitive Swarm multi-perspective deliberation |
| `/v1/got` | POST | Graph-of-Thoughts multi-branch tree search & merge |
| `/v1/scientific` | POST | Deterministic closed-world rational SI science solver & receipt |
| `/v1/telemetry` | GET | Real-time Dem-Lab statistics, entropy, RSI momentum, and MoE state |
| `/v1/feedback` | POST | Closed-loop Q-learning reward update for adaptive budget policy |
| `/v1/models` | GET | Model discovery and supported capabilities |
| `/health` | GET | Readiness probe |
