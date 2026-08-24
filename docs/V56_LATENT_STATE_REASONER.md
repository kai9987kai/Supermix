# Supermix v56 — the latent state reasoner

## What v56 is

V56 is an additive research line. It reuses the v53 MiMoMix components verbatim
and adds one thing v53 does not have: an explicit **latent state machine**
between the trunk and the answer. It is the first line in this repository to
train a MiMoMix-derived model to a checkpoint, persist it, and serve it.

V56 does not change v51, v52, v53, v54 or v55. It adds no model to
`studio_runtime_manifest.json`, no entry to `MODEL_SPECS`, and no branch to
`multimodel_runtime._build_backend`, so no existing checkpoint, manifest, route
decision or promotion decision moves because v56 exists.

| surface | v56 contract |
| --- | --- |
| model | `source/mimomix_reasoner.py` — `ReasonerConfig`, `LatentStateReasoner` |
| curriculum | `source/reasoner_curriculum.py` |
| training and measurement | `source/benchmark_mimomix_reasoner.py` |
| promotion gate | `source/run_v56_promotion_gate.py` |
| web interface | `source/mimomix_reasoner_web_app.py` |
| checkpoint schema | `supermix-v56-reasoner-checkpoint-v1` |
| benchmark receipt schema | `supermix-v56-reasoner-benchmark-v1` |
| gate receipt schema | `supermix-v56-bounded-promotion-v1` |
| source of truth | `source/` |
| compatibility mirror | none — see below |
| tests | `test_mimomix_reasoner.py`, `test_mimomix_reasoner_web_app.py` |

### Deliberate `runtime_python/` difference

The v52 merge contract requires `runtime_python/` to either mirror `source/` or
document a deliberate packaged-runtime difference. This is that documentation:
**v56 is `source/`-only on purpose.** It is a single-task research checkpoint. It
is not a chat model, it has no tokenizer, and it answers exactly one synthetic
question, so mirroring it into the packaged desktop runtime would enlarge the
installer for a model no desktop user can use. Mirroring becomes appropriate only
if a v56 descendant ever serves a product surface.

## The task, and why the previous line could not do it

`benchmark_cognitive_leap_ultra_v51.make_chained_task` builds a 128-dimensional
vector holding a start digit and four `(operation, operand)` blocks, and labels it
with the composed result mod 10. Operations are add, multiply and subtract;
operands are 1–9. Decoding the one-hot fields and re-applying the rule reproduces
the label on 100% of samples, so the input determines the answer exactly. There
are `10 × 27⁴ = 5,314,410` distinct inputs; a 12,000-example training set covers
0.23% of them, and only 4 of the 1,000 canonical test items appear in it. It is a
generalisation test, not a memorisation test.

The recorded v51 result is **0.1710** held-out accuracy. The **majority-class
floor on the same test set is 0.1430**. The previous line beat a constant
predictor by 2.8 points.

### The structural reason

Measured Bayes-optimal accuracy over 4,000,000 samples, by which part of the
input a predictor is allowed to see:

| information given | Bayes accuracy | Bayes cross-entropy |
| --- | --- | --- |
| nothing (majority class) | 0.1415 | 2.2843 |
| the last operation only | 0.1720 | 2.1240 |
| the last two operations | 0.2013 | 1.9810 |
| the last three operations | 0.2389 | 1.8406 |
| all four operations, no start digit | 0.4105 | 1.3162 |
| the start digit only | 0.1415 | 2.2843 |
| start + first three operations | 0.2521 | 1.9742 |
| **start + all four operations** | **1.0000** | **0.0000** |

Every proper subset of the input is worth almost nothing and the complete input
is worth everything. A model that has learned three of the four maps earns
essentially no credit for it, so there is no gradient path from a partial
solution to the exact one. That is the task, not the architecture — and it
explains a plateau that is otherwise mysterious: a flat MLP, a slot-tokenised
transformer, a GRU state recurrence and a latent state machine all stall between
0.24 and 0.27 regardless of architecture, and **raising the training set from
12,000 to 96,000 examples does not move it** (held-out cross-entropy floors at
≈1.51 in every case). The bottleneck is credit assignment, not capacity and not
data.

Note also that 0.1720 — the best number the v51 line ever recorded on the
canonical cohort at any inference setting — is exactly the Bayes-optimal accuracy
for a predictor that sees **only the final operation**.

## The model

### Slot tokenisation

`SlotTokenizer` reads the same flat vector as a short sequence. Two layouts:

* `blocks` uses the generator's block boundaries — four 24-wide operator slots at
  offset 10, with dims 0–9 and 106–127 becoming a single context slot.
* `patches` splits the vector into equal patches and knows nothing about where
  the boundaries actually are.

Both are checked to cover **every** input dimension. Dropping the 22-dimension
unused tail would be a silent change to what the model is given, so the tail is
projected into the context token instead. `LatentStateReasoner.forward` accepts
the harness's `(B, 1, 128)` tensor directly: the v56 model consumes exactly the
tensor the v51 head consumed, with nothing added and nothing recomputed from the
generator.

### Position equivariance

The generator writes every operator block with the same 24-dim layout, so an
operation means the same thing wherever it appears. `share_block_encoder` gives
all operator slots one encoder, which turns `n_blocks × n_operations` maps to
learn into `n_operations`.

That symmetry is easy to lose. The operator head originally read the *post-trunk*
hidden state, and attention mixes neighbouring slots into it, so the same
operation in a different block was no longer the same input.
`ReasonerConfig.operator_context` makes the choice explicit:

* `slot` — the pre-trunk slot embedding only. Exactly equivariant.
* `trunk` — the post-trunk state only. More expressive, not equivariant.
* `gated` (default) — the slot embedding plus a **zero-initialised** scalar gate
  on the trunk state. A fresh model is exactly equivariant and can learn to spend
  context only where context pays.

`test_a_fresh_model_is_position_equivariant` asserts bit-identical operators for
the same operation in different blocks, and
`test_trunk_context_mode_is_not_claimed_to_be_equivariant` records that the trunk
mode is not, so the trade-off cannot be quietly forgotten.

### Row-stochastic transitions, composed in log space

The state is a distribution over `n_states` latent states, not a hidden vector.
Each operator slot emits a `K × K` row-stochastic matrix and the state is
composed with `logsumexp`. A probability-space product underflows and loses the
gradient that has to reach the first operator; log space does not.

The operator head is nonlinear on purpose. A linear map from a concatenation of
one-hot fields is additive in those fields, so it cannot express the
*interaction* between an operation and its operand — and an operator that cannot
represent that interaction cannot represent the task. Measured: with a linear
head the model reaches 0.167, barely above the floor.

### Identity initialisation

A product of four near-uniform stochastic matrices mixes the state to uniform,
which starves the gradient reaching the first operator. Each operator's logits
carry a diagonal bias at init (`identity_gain`), so a fresh chain passes its
initial state through unchanged. This is the same discipline v53 uses when it
zero-initialises the thinking core's residual scale, and
`test_identity_initialisation_preserves_the_initial_state` pins it.

### The crispness prior

The maps being composed are deterministic functions, so a transition row that is
one-hot is a row that has learned a function. `operator_entropy_weight` says that
directly rather than hoping for it. It is a training signal only:
`test_operator_entropy_penalty_changes_the_loss_but_not_the_logits` requires the
penalised and unpenalised models to emit identical logits.

### Reused from v53, unchanged

`MiMoMixBlock` (hybrid SWA/global attention with learnable per-head sinks,
decoupled local/global RoPE tables), `SparseMoEFeedForward` with the
auxiliary-loss-free bias rule, and `RecursiveThinkingCore` with PonderNet/ACT
halting, the trainable verifier temperature, and the supervised
`p(correct)`/`p(continue)` heads. The bias update fires once per optimizer step
via `step_router_bias()`, never inside `forward` — firing per forward makes the
effective step depend on gradient-accumulation depth and the sign rule rings
instead of converging.

### What is deliberately absent

Multi-token prediction and speculative decoding. V56 emits one answer per input,
so there is no next token to draft and no acceptance length to report. They stay
in `mimomix_core`/`mimomix_decoding` for the generative line. `rope_scaling`
defaults to `none` because a five-slot sequence has no context to extend.

## The curriculum

`mul by 1` is an identity on the state and the encoding can represent it
(`op_type = 1`, `operand = 1`). A chain whose trailing operations are all `mul 1`
is therefore an **effectively shorter chain that is still a valid draw from the
generator's own support** — the generator produces it with probability `(1/27)`
per slot. The curriculum is a re-weighting of the same input distribution, not a
different task.

That distinction is the whole fairness argument. A curriculum built from modified
inputs, extra label fields, or intermediate-state supervision would train on
information the baseline never had. This one does not: every curriculum example
is an `(input, label)` pair the original generator assigns exactly the same way.
`reasoner_curriculum.assert_matches_reference_encoding` decodes the reference
generator's own tensors, pushes them back through this module's encoder and label
rule, and requires both the one-hot pattern and the label to match exactly. The
benchmark calls it before every curriculum run.

### Which slots stay active

The first version of the curriculum kept a **prefix** of active operations and
pinned the rest to `mul 1`. That was a real defect, and error analysis of the
first checkpoint found it rather than intuition:

| step | running answer agrees with the true intermediate value |
| --- | --- |
| after op 1 | 0.9631 |
| after op 2 | 0.9933 |
| after op 3 | 0.9833 |
| after op 4 | 0.9355 |

**83% of that model's remaining errors first diverged at the last step**, and
another 12% at the second-to-last. The cause is the recipe, not the model: with a
fixed prefix, slot 4 saw a genuine operation only in the final stage, and
`positional_blocks` gives each slot its own embedding, so the shared operator
could not cover the gap.

`sample_chain(..., random_slots=True)` — now the default — chooses *which* slots
stay active uniformly per example, keeping the number of active operations the
same. Coverage goes from `[0.96, 0.96, 0.00, 0.00]` to `[0.48, 0.48, 0.48, 0.49]`.
`--curriculum_prefix_slots` restores the original behaviour, and the receipt
records which was used.

The invariant that matters survives either way: dropping every identity slot must
leave the label unchanged, which
`test_identity_operations_anywhere_leave_the_label_alone` checks per row.

Evaluation never uses the curriculum. It is always
`make_chained_task(test_size, seed + 1)`, imported verbatim.

**The curriculum is a recipe difference as well as an architecture difference,
and results under it must be reported as such.** `--protocol matched` exists so
the architecture can also be compared on identical data.

## Measured results

Both protocols evaluate on the identical untouched held-out set —
`make_chained_task(1000, seed=52)` — and the baseline row is the recorded v51
result, independently re-run on this machine to a bit-identical receipt.

| model | protocol | training examples | held-out accuracy | NLL | ECE |
| --- | --- | --- | --- | --- | --- |
| majority-class constant | — | — | 0.1430 | — | — |
| v51 `CognitiveLeapUltraExpert` (2,245,715 params) | 12,000 × 4 epochs | 12,000 | 0.1710 | 2.1907 | — |
| **v56 latent state reasoner** (808,626 params) | **matched**, 12,000 × 4 epochs | 12,000 | **0.2410** | 1.5720 | 0.0448 |
| v56, first curriculum (prefix slots) | curriculum | 160,000 | 0.9220 | 0.2021 | 0.0235 |
| **v56, current best** (808,626 params) | **curriculum**, random slots + crispness prior | 160,000 | **0.9740** | **0.0680** | 0.0194 |

### Ablation: what actually moved the number

All four runs use an identical budget to the 0.9220 baseline — 160,000 curriculum
examples, 3 epochs per stage, same seed — so only the named change differs.

| run | slots | operator entropy weight | positional | operator context | accuracy |
| --- | --- | --- | --- | --- | --- |
| first curriculum | prefix | 0 | yes | gated | 0.9220 |
| A | random | 0 | yes | gated | 0.9480 |
| **B (current best)** | **random** | **0.02** | **yes** | **gated** | **0.9740** |
| C | random | 0.02 | **no** | gated | 0.8210 |
| D | random | 0.02 | no | **slot** | 0.6920 |

Two findings, one of each sign:

* **The slot fix and the crispness prior each contribute about +2.6 points and
  they compose.** Fixing slot coverage alone takes 0.9220 to 0.9480; adding the
  operator-entropy penalty takes it to 0.9740. Error falls from 7.8% to 2.6%, a
  67% reduction, and NLL falls by 3x.
* **Removing positional embeddings hurts badly** (0.8210), and removing the trunk
  context from the operator as well hurts more (0.6920). Full position
  equivariance is *not* free here: with the slots randomised the model gains
  little from the symmetry and loses the ability to tell its slots apart. This is
  recorded because the equivariance argument in the design section predicts the
  opposite, and the measurement wins.

### Error analysis before and after

Agreement between the model's running answer and the true intermediate value,
20,000 fresh samples (seed 900001):

| step | first curriculum | current best |
| --- | --- | --- |
| after op 1 | 0.9631 | **0.9923** |
| after op 2 | 0.9933 | **0.9992** |
| after op 3 | 0.9833 | **0.9957** |
| after op 4 | 0.9355 | **0.9758** |

Steps 1 and 2 no longer produce a single first-divergence. Of the errors that
remain, 93.2% first diverge at the final step and 6.8% at the third — the model
gets the whole chain right up to the last operation and then loses it. Mean
confidence separates correct from wrong at 0.9734 versus 0.6504, so what is left
is mostly uncertain rather than confidently wrong.

The matched row changes only the model: identical examples, identical seed,
identical epoch budget, 2.8× fewer parameters. It is +7.0 points of accuracy, and
against the floor it is +9.8 versus the baseline's +2.8 — a 3.5× larger margin
over a constant predictor. The curriculum row changes the training recipe as
well, and clears 0.4105, so that model is composing the whole chain rather than
reading its tail.

Test-time scaling was flat for the matched model (0.2410 at 1, 3 and 8 thinking
cycles, and under adaptive halting). More cycles are not free accuracy here, and
the receipt records that rather than reporting only the best setting.

### Paired promotion gate

`output/v56_promotion_gate.json`, curriculum checkpoint versus the strongest
recorded v51 checkpoint (the 12,000-example candidate, `eval_default` 0.1710), on
20 fresh cohorts of 2,000 samples each — 40,000 paired samples, no reserved seed:

| arm | accuracy | 95% Wilson CI |
| --- | --- | --- |
| v56 current best | **0.9762** | [0.9747, 0.9776] |
| v51 baseline | 0.1718 | [0.1681, 0.1755] |
| majority-class floor | 0.1393 | — |

Delta **+0.8044** (+468% relative). Per-seed: **20 wins, 0 losses, 0 ties**, sign
test p = 1.9e-6. McNemar on the discordant pairs: 32,288 candidate-only versus
112 baseline-only, continuity-corrected chi-square, p below floating-point
resolution. Every gate check passes and the receipt records
`decision: promote_for_this_task` with `production_default_allowed: false`.

The first curriculum checkpoint scored 0.9329 on the same cohorts against the
same baseline; that receipt was overwritten by this one, and both numbers are
recorded here.

### Calibration and the verifier

On the canonical set the **current-best** model (`output/v56b_randslots_entropy`,
the one the gate promoted) has ECE 0.0194 and Brier 0.0357 at 0.9602 mean
confidence. Selective accuracy at 50% coverage is 1.0000 by softmax confidence
and 0.9900 by the verifier.

The first curriculum run is the row to compare it against, because the two differ
only in random slots and the entropy prior: ECE 0.0235, Brier 0.1089, mean
confidence 0.9105, selective accuracy 0.9980 by confidence and 0.9280 by the
verifier. The crispness prior cut Brier by a factor of three, which is the
calibration half of the same change that moved accuracy 0.9220 → 0.9740.

On neither model did the verifier beat plain softmax confidence: both receipts
record `verifier_beats_confidence_at_50pct: false`. A named verifier head is not
evidence of a useful one; the risk-coverage table is.

### Routing

Accumulated over the held-out set the MoE layers reach mean normalised routing
entropy 1.000 with 0 starved experts, expert shares 12–14% across all eight.
Measured on a *single* forward, three experts per layer look idle — that is
arithmetic, not starvation, since top-2 over five slot tokens cannot reach eight
experts. The web interface says which of the two it is showing.

Receipts: `output/v56b_randslots_entropy/benchmark_results.json` (the current-best
row, the promoted checkpoint, and the ablation's run B),
`output/v56_matched_12k_4ep/benchmark_results.json`,
`output/v56_curriculum/benchmark_results.json` (the **first** curriculum run,
0.9220 — not the headline), `output/v56_promotion_gate.json`, and
`output/v56_chat_benchmark.json`. The baseline replication receipt is
`output/v56_baseline_replication_v51_12k_seed51/benchmark_results.json`.

Read the accuracy against the Bayes table above, not against zero. An accuracy
above 0.4105 is the first evidence that a model on this task is using the start
digit at all, because 0.4105 is the ceiling for any predictor that ignores it.

## The chat surface

`source/reasoner_chat.py` plus `/chat` on the web app let a person type an
arithmetic question in English. The division of labour is the whole design, and
the page states it in a banner rather than in a footnote:

* **The parser does the language.** `parse_problem` is a regex and a lookup
  table. Nothing in it is learned. It is not the model.
* **The model does the arithmetic.** It never sees text — it receives the same
  128-dimensional vector `make_chained_task` would have produced.

Three consequences are enforced rather than asserted:

1. **Answers are graded by the generator, not the model.** `correct` comes from
   re-running the arithmetic rule independently, so the model never marks its own
   work.
2. **Chains longer than four operations are repeated model calls.** The input has
   four operator slots. Shorter chains pad with `mul 1` — the representable
   identity the curriculum already uses, so a padded chain is still in
   distribution. Longer ones re-run the model on its own argmax, which compounds
   its errors, so `model_calls` is always reported.
3. **Message content is data.** `_longest_expression` extracts the longest
   well-formed `digit (op digit)+` run and discards everything else, so an
   instruction inside a message contributes nothing a downstream stage can read.

That third property was not free. The first parser mapped "and" to `+`, which
turned surrounding prose into arithmetic; the benchmark's injection case caught
it, and `test_message_content_cannot_steer_the_service` now pins it. The word
list is deliberately conservative — "and", "less" and "take" are absent because
they are ordinary English far more often than they are operators.

### Chat benchmark

`source/benchmark_reasoner_chat.py`, receipt `output/v56_chat_benchmark.json`,
150 samples per row against the current best checkpoint. Parser and model are
measured separately because only one of them is learned, and a single blended
number would hide which failed:

| operations | parse rate | model given parse | end to end | model calls | mean latency | p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.000 | 1.0000 | 1.0000 | 1.0 | 27 ms | 39 ms |
| 2 | 1.000 | 1.0000 | 1.0000 | 1.0 | 28 ms | 38 ms |
| 3 | 1.000 | 0.9933 | 0.9933 | 1.0 | 26 ms | 28 ms |
| 4 | 1.000 | 0.9800 | 0.9800 | 1.0 | 27 ms | 30 ms |
| 8 | 1.000 | 0.9800 | 0.9800 | 2.0 | 52 ms | 55 ms |
| 12 | 1.000 | 0.9267 | 0.9267 | 3.0 | 79 ms | 83 ms |
| 16 | 1.000 | 0.9400 | 0.9400 | 4.0 | 106 ms | 124 ms |

The same table on the previous 0.9220 checkpoint ran 0.9933 / 0.9867 / 0.9733 /
0.9333 / 0.9133 / 0.8667 / 0.8067, so the training fixes carry through the
serving path. Rows at 12 and 16 operations are 150 samples each and differ by
less than their own noise; do not read 16 > 12 as a trend.

Parser robustness: 11 of 11 cases behave as specified, refusing operand 0,
operands above 9, a start above 9, and division, each with the reason. All three
injection payloads answer identically to the benign input.

Read the decay honestly: accuracy above four operations is **not** a longer
model, it is the same model run repeatedly on its own output. Latency scales with
the call count, and these are CPU numbers on one machine with a dev server.

## What this does **not** prove

The suites prove integration, gradient flow, and the structural invariants named
above. The benchmark and the gate prove a difference in held-out accuracy on one
task. None of them prove any of the following, and no number in this repository
should be read as evidence of them:

* **That the model is good at anything else.** V56 answers one synthetic
  arithmetic question. It has no tokenizer, no language capability, and no path
  to one. Its input is a fixed 128-dimensional vector.
* **That the chat surface makes it a language model.** It does not, and the page
  says so before anything else. Every conversational sentence on `/chat` was
  written by hand in `describe_capabilities` or assembled by the parser. None of
  it is generated. A parse rate of 1.000 measures a regex.
* **That the architecture is better in general.** The latent state machine has an
  inductive bias matched to composed maps over a small state set. That is the
  right bias for this task family and says nothing about any other.
* **That the curriculum result and the matched result are the same claim.** The
  curriculum changes the training recipe. Only `--protocol matched` isolates the
  architecture.
* **That beating v51 here means beating v51 anywhere.** The v51 head is a
  general classifier head used across this repository's lines; this task is one of
  its workloads, and it was never trained with a curriculum.
* **That the verifier is semantic.** `p(correct)` is supervised against the
  model's own correctness on this task and calibrated by a temperature trained
  only by `verifier_loss`. It ranks this task's answers. It is not a general
  self-knowledge signal, and the risk-coverage table is the only evidence it
  ranks anything at all.
* **That the reported latency is a product latency.** Measurements are CPU-only,
  batched, on one machine, with no serving stack.
* **That the state trace is an explanation.** It is the model's arithmetic,
  faithfully reported — `test_the_trace_composition_matches_the_reported_final_state`
  pins that the displayed trace composes to the displayed answer. Faithful is not
  the same as interpretable, and a latent state that lines up with the true
  intermediate value on this task may not correspond to anything on another.

## Promotion gates

`source/run_v56_promotion_gate.py` enforces the bar the audit of this task
requires, because a single 1,000-example draw has a standard error of ~1.2 points:

1. **Fresh cohorts.** Seeds 51 and 52 and every seed used by an existing v51 gate
   or receipt are refused outright.
2. **Identical inputs.** Both arms score the same tensor from the same
   `make_chained_task` call. `(seed, size)` jointly determine a cohort, so size is
   held fixed across arms.
3. **A paired test.** McNemar on the discordant pairs — exact binomial where
   tractable, continuity-corrected chi-square beyond, and the method is reported.
4. **Per-seed non-regression.** A declared majority of seeds must favour the
   candidate, with a two-sided sign test reported over them.
5. **Disjoint Wilson intervals** on the pooled accuracy.
6. **A floor check** on every cohort, since the floor is only ~1.3 points below
   the recorded baseline.

Anything beyond this task needs its own evidence: held-out accuracy and NLL
against a frozen baseline on the target workload, ECE / Brier / selective
accuracy / risk-coverage, adversarial slices, mean and p95 latency on the target
hardware, and source/package parity. None of that has been run, because v56 does
not claim it.

## Runtime and packaging integration

V56 adds nothing to the packaged runtime. Specifically it is **not** added to
`source/studio_runtime_manifest.json`, the Qwen or Studio PyInstaller specs, the
desktop build scripts, or `MODEL_SPECS`. It is added to the CI compile allowlist,
the CI ruff allowlist, and the CI pytest list in
`.github/workflows/runtime-quality-gates.yml`.

The web interface binds `127.0.0.1` and refuses to start without a
`supermix-v56-reasoner-checkpoint-v1` file. Only typed, range-checked fields
steer it — there is no prompt, no tool call and no free text that reaches the
model, and `test_free_text_and_unknown_keys_cannot_steer_the_service` fires an
instruction-injection payload and requires bit-identical output.

A trained checkpoint, a passing suite and a running interface are release
evidence for this task only. They are not consequences of this design document.
