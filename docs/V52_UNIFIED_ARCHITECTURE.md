# Supermix v52 unified architecture

## Merge contract

Supermix v52 is a curated union, not a directory concatenation.

- The advanced `Supermix` working tree is the source of truth for model code,
  v51 runtime compute, `context_mix_v4`, multimodel routing, memory, and UI.
- `Supermix_27` is used only as a reviewed donor. Its committed virtual
  environment, bytecode, `dist/`, `output/`, logs, stale root code mirrors, and
  absolute build paths are excluded.
- `source/` owns model and terminal/web behavior. `runtime_python/` must either
  mirror it or document a deliberate packaged-runtime difference.
- Installer upgrades delete the prior bundled-model directory before installing
  a new manifest, preventing removed models from surviving an upgrade.

## What v52 adds

### Verified recursive reasoning

`cognitive_leap_v52_expert` retains v51's weight-tied latent refinement,
cross-latent attention, ACT halting, deep supervision, and prediction-stability
exit. It adds a supervised `quality_head` with separate `p(correct)` and
`p(continue)` outputs. When explicitly enabled, low-quality requests can receive
a larger recursive budget and the verifier selects between the initial and
escalated proposals.

This follows the practical lesson from
[Tyen et al. (ACL 2024)](https://aclanthology.org/2024.findings-acl.826/):
models often struggle to locate their own mistakes, while a separately trained
classifier can provide useful error-location evidence. It also follows
[compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314), which
allocates inference work by problem difficulty rather than using one budget for
every prompt.

### Sparse recurrent-core execution

V51 calculated every recurrent core even after routing. V52 supports true top-k
execution, top-2 by default, with router load-balance and z-loss diagnostics.
Wall-clock performance must still be benchmarked because sparse Python dispatch
can cost more than it saves for tiny CPU batches. The design is informed by
[Mixture-of-Depths](https://arxiv.org/abs/2404.02258), while remaining a compact
adaptation rather than a reproduction of that Transformer architecture.

### Emotional appraisal without factuality collapse

The model has separate multi-label emotion, user-intent, and response-strategy
heads. Their residual is bounded and separate from the problem-plan channel.
The runtime interaction planner uses the same structure:

1. identify salient emotion/event clues;
2. infer the user's likely goal;
3. choose a response strategy;
4. choose a reasoning/retrieval mode;
5. apply an epistemic guard so acknowledgement does not become agreement with
   an unsupported claim.

This is broader than sentiment analysis and maps to the key-event, mixed-event,
implicit-emotion, and intention tasks in
[EmotionQueen](https://aclanthology.org/2024.findings-acl.128/). Conversation
clues and auxiliary supervision are motivated by
[CoE (ACL 2025)](https://aclanthology.org/2025.acl-long.1148/).

Named model appraisal outputs are not semantic merely because the heads exist.
They require labelled auxiliary training and held-out evaluation.

### Calibration and memory

V52 includes a trainable temperature for the classifier/verifier path and
reports calibrated entropy. If simple temperature scaling is insufficient, an
auxiliary calibrator can be evaluated following
[Thermometer](https://arxiv.org/abs/2403.08819).

External structured memory remains preferred over online weight mutation. The
next memory promotion should add multi-granularity summaries, provenance,
confidence, expiry, contradiction handling, and surprise-gated writes, informed
by [Reflective Memory Management](https://aclanthology.org/2025.acl-long.413/).

## Training

The v52 head can initialize from a v51 checkpoint. `load_weights_for_model`
retains compatible recursive weights and initializes new structured heads.

Recommended starting objective:

```text
L = L_class
  + lambda_deep * L_cycle
  + lambda_verify * L_quality_continue
  + lambda_affect * L_emotion_intent_strategy
  + lambda_ponder * L_ponder
  + lambda_balance * L_router
```

`source/finetune_chat.py` exposes `--verifier_weight` in addition to the existing
deep-supervision, ponder, latent-consistency, and MoE auxiliary weights. A real
affect curriculum must provide multi-label emotion targets plus intent and
strategy targets; the ordinary ten-class response dataset cannot establish
emotional intelligence by itself.

Current implementation boundary: the head-level `structured_auxiliary_loss(...)`
accepts those labelled targets and is gradient-tested, but the general JSONL
loader/collator does not yet carry them into `finetune_chat.py`. The production
trainer currently integrates the verifier objective only. Adding a masked
multi-task dataset schema and an end-to-end labelled-batch test is a required
follow-on before affect training is advertised as a CLI capability.

## Runtime policy

- Default: prompt-derived budget ceiling plus prediction-stability early exit.
- Diagnostic: non-mutating 1/3/8-cycle sweep.
- V52.1 auto mode: plan before inference, set separate task-difficulty and
  epistemic-risk floors, evaluate budgets progressively, require cross-budget
  agreement for harder turns, and reuse the accepted probe output.
- V52 opt-in: verifier escalation, with a hard maximum-cycle cap.
- High uncertainty: clarify, retrieve evidence, disclose uncertainty, or decline
  rather than spending unbounded compute.

### V52.1 metacognition and affect flow

The deterministic interaction planner now uses recent user turns only as a
short-lived, decayed context signal. An anchored follow-up can retain a cautious
``possible_distress`` appraisal, while unrelated topic changes rapidly drop the
carry-over. No inferred emotional state is stored as a user fact. This is a
small runtime adaptation of appraisal-and-reappraisal and conversational affect
flow research: [Third-Person Appraisal Agent](https://aclanthology.org/2025.findings-emnlp.1288/)
and [Appraisal-Theoretic Affect Flow](https://aclanthology.org/2025.conll-1.16/).

Every plan also emits a compositional response contract. The contract preserves
mixed objectives such as emotional acknowledgement plus a substantive debugging
solution, exposes coverage and violations in diagnostics, and allows a relevant
answer to receive a short acknowledgement without deleting its useful content.
Unrelated responses that miss substantive obligations are rejected in favor of
a grounded clarification. Crisis and urgent-health signals take a separate
safety-first fast path: valid emergency guidance is preserved, unrelated
retrieval is replaced with immediate-help guidance, and the controller does not
delay that response by forcing a deep compute budget.

The progressive controller is training-free. It follows the practical
``reason just enough`` direction explored by
[REFRAIN](https://arxiv.org/abs/2510.10103), but uses bounded classifier cycles,
cross-budget label agreement, confidence/entropy targets, and explicit policy
floors rather than reproducing REFRAIN's chain-of-thought bandit.

The verifier toggle composes with progressive auto mode as an advisory stop
signal. The controller consumes the verifier's continue probability rather than
starting a second nested escalation loop; manual compute mode retains the
model-side escalation behavior.

## Promotion gates

A v52 checkpoint is promoted only when it beats a frozen v51/v50 baseline on:

- held-out response-selection accuracy and negative log likelihood;
- ECE, Brier score, selective accuracy, and risk-coverage;
- reasoning/problem-solving suites and adversarial error-localization slices;
- emotion, intention, key-event, mixed-event, and empathetic-response rubrics;
- false-premise, contradiction, sycophancy, and vulnerability tests;
- mean/p95 latency, cycles used, and quality per compute budget;
- terminal, source web, packaged web, and desktop parity.

The architecture and passing smoke tests prove integration and gradient flow;
they do not by themselves prove general cognition or emotional intelligence.
