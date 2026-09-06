# Supermix v89 Frontier Epistemic Hybrid (NexusMiMo-Friston Hybrid)

**Release Date:** 2026-09-05  
**Version:** `89.0.0`  
**Milestone:** Epistemic Active Inference, Neuro-Symbolic Proof Verification (First-Error Localization), Bidirectional Speculative Decoding, and Epistemic MCTS.

---

## 1. Executive Summary & Research Motivation

The **Supermix v89 Frontier Epistemic Hybrid** elevates the Supermix architecture from reactive auto-looping into **active, hypothesis-driven cognitive planning**. It unifies three major paradigms at the cutting edge of 2026 machine learning and cognitive science:

1. **Karl Friston's Active Inference & Free Energy Principle**:
   Reasoning agents should not passively accumulate tokens; they must act to minimize **Expected Free Energy (EFE)**, dynamically balancing **Pragmatic Value** (achieving verified target outcomes) against **Epistemic Value** (active exploration that resolves parameter and state uncertainty).
2. **Neuro-Symbolic Proof Verification & First-Error Localization (FEL)**:
   Directly solving the open research challenge documented in `docs/V87_RESEARCH_NOTES.md` ("first-error accounting, explicitly bounded prefix-continuation diagnostics"). Derivations are broken down into atomic steps, audited for premise grounding (eliminating phantom variables), checked against exact rational arithmetic, and repaired via deterministic symbolic surgery.
3. **Xiaomi MiMo Bidirectional Speculative Decoding**:
   Extending Multi-Token Prediction (MTP) speculative drafting into forward-backward consistency verification ($P \to Y$ forward draft, inverted via $Y \to P'$ reverse check). Spurious forward hallucinations that fail algebraic inversion are discarded before context pollution occurs.
4. **Epistemic Monte Carlo Tree Search (NexusEpistemicMCTS)**:
   An autonomous reasoning search tree where node expansion is guided by Friston EFE policies, nodes are pruned when First-Error Localization detects unsound steps, and branches are recovered through symbolic repair.

---

## 2. Mathematical & Theoretical Foundations

### 2.1 Expected Free Energy (EFE) Minimization
For an agent operating on reasoning state $s$ with candidate cognitive actions $a \in \mathcal{A}$ (e.g. `DECOMPOSE_SUBGOAL`, `EXECUTE_DETERMINISTIC_STEP`, `CAUSAL_COUNTERFACTUAL_CHECK`, `EXPAND_SPECULATIVE_DRAFT`, `BACKTRACK_PRUNE`, `HALT_AND_SEAL`):

$$G(a) = \underbrace{D_{\mathrm{KL}}[Q(s'|a) \parallel P(s')]}_{\text{Pragmatic Risk (divergence from goal prior)}} - \underbrace{\mathbb{E}_{Q(s'|a)}[\mathcal{I}(s'; \theta)]}_{\text{Epistemic Information Gain (uncertainty reduction)}}$$

The action selection policy is governed by a precision-weighted Gibbs-Boltzmann distribution:

$$P(a) = \frac{\exp(-\beta \cdot G(a))}{\sum_{a' \in \mathcal{A}} \exp(-\beta \cdot G(a'))}$$

where the dynamic precision parameter $\beta$ is modulated by the Relative Strength Index (RSI) volatility oscillator and local Shannon token entropy $H$:

$$\beta = \frac{1}{T} \cdot \left(1 + 0.8 \cdot \frac{|\text{RSI} - 50|}{50}\right) \cdot \exp(-0.25 \cdot \min(3.0, H))$$

- **High Volatility / Extreme Conviction ($\text{RSI} > 70$ or $< 30$)**: Precision $\beta$ tightens, prioritizing deterministic, low-risk verification.
- **High Entropy ($H > 1.5$)**: Precision softens, encouraging epistemic information-gathering actions.

---

### 2.2 First-Error Localization (FEL)
Given a multi-step chain-of-thought derivation $S = [s_1, s_2, \dots, s_K]$ and initial problem premises $\mathcal{P}$, the localizer tracks active registers:

$$\mathcal{R}_0 = \{ \text{numbers, entities, and constants present in } \mathcal{P} \}$$

For each step $s_i = \langle \text{op}_1, \odot, \text{op}_2 = \text{res} \rangle$:
1. **Register Grounding**: Assert that $\text{op}_1, \text{op}_2 \in \mathcal{R}_{i-1}$. If false, flag `PHANTOM_REGISTER` (hallucinated operand).
2. **Arithmetic Soundness**: Evaluate $\text{expected} = \text{op}_1 \odot \text{op}_2$. If $|\text{res} - \text{expected}| > \epsilon$, flag `ARITHMETIC_ERROR`.
3. **Register Propagation**: $\mathcal{R}_i = \mathcal{R}_{i-1} \cup \{ \text{expected} \}$.

Upon identifying the first erroneous step at index $e$, the system synthesizes a repaired step $s_e^*$ and propagates the corrected register forward to rescue downstream derivations.

---

### 2.3 Bidirectional Speculative Inversion
Given premise $P = \langle A, B \rangle$ and operation $f(A, B) = Y$:
- **Forward Speculative Pass**: Drafts candidate answer $Y_{\mathrm{cand}}$.
- **Reverse Inversion Pass**: Evaluates inverse operator $f^{-1}(Y_{\mathrm{cand}}, B) = A'$.
- **Consistency Metric**:
  $$\rho_{\mathrm{bidir}} = 1.0 - \frac{|A' - A|}{\max(1.0, |A|)}$$

When $\rho_{\mathrm{bidir}} \ge 0.90$, the candidate answer is mathematically confirmed and accepted without deep neural forward invocation.

---

## 3. Architecture & REST Endpoints

### 3.1 REST API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/active_inference/decide` | Evaluates EFE, epistemic information gain, pragmatic risk, and action probabilities |
| `POST` | `/v1/proof/verify_steps` | First-Error Localization (FEL) across scratchpad steps with symbolic branch repair |
| `POST` | `/v1/speculative/bidirectional` | Dual-draft forward-backward speculative consistency verification |
| `POST` | `/v1/mcts/epistemic_search` | Monte Carlo Tree Search guided by EFE policy and First-Error pruning |
| `GET` | `/v1/signals` | Exposes `v89_frontier_epistemic` capability telemetry |
| `GET` | `/health` | Reports version `89.0.0` and proof ledger status |
| `GET` | `/studio` | Unified Web Studio v89 interface |

---

## 4. Unified Web Studio v89

The Unified Web Studio (`web_static/nexus_studio.html`) features four dedicated interactive panels:
1. **Active Inference (EFE)**: Live Free Energy parameter controls (RSI volatility, entropy, confidence), EFE spectrum bar charts, and dynamic precision $\beta$ monitors.
2. **Proof Verifier (FEL)**: Multi-step scratchpad debugger with step-by-step pass/fail badges, phantom variable detection, and symbolic repaired trace generation.
3. **Bidirectional Speculator**: Dual-pass forward draft and reverse equation inversion comparison with consistency score gauges.
4. **Epistemic MCTS**: Search tree visualizer displaying evaluated nodes, pruned branches, mean EFE, and the optimal reasoning path.
