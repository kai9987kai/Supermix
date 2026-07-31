# Supermix v53 — MiMoMix hybrid architecture

## What v53 is

V53 is a new, self-contained line in `source/mimomix_*.py`. It fuses three
lineages that until now had nothing to do with each other:

| lineage | what v53 takes |
| --- | --- |
| **Xiaomi MiMo** (V2-Flash, V2.5-Pro) | hybrid SWA/global attention with learnable attention sinks, auxiliary-loss-free sparse MoE, Multi-Token Prediction reused as a self-draft, progressive RoPE context extension, and MOPD post-training |
| **Supermix v51/v52** | weight-tied recursive latent refinement, ACT halting with a ponder cost, the supervised quality/continue verifier, trainable temperature calibration, and progressive decision-safe auto compute |
| **AI-Dem-Lab** | the questions its browser panels ask — entropy, novelty, stability, resonance, evidence, multi-agent competition — re-answered with defensible statistics |

It is **additive**. Nothing in the v52 tree changed. `model_variants.py`,
`multimodel_runtime.py`, the route control plane, and every existing gate are
untouched, so no existing checkpoint, manifest, or promotion decision moves
because v53 exists.

### Deliberate `runtime_python/` difference

The v52 merge contract requires `runtime_python/` to either mirror `source/` or
document a deliberate packaged-runtime difference. This is that documentation:
**v53 is `source/`-only on purpose.** It is an unproven research line with no
trained checkpoint, so mirroring it into the packaged compatibility runtime
would ship an untrained model path into the desktop build and enlarge the
installer for nothing. It is also absent from `studio_runtime_manifest.json` and
from the Studio route control plane for the same reason. Mirroring becomes
appropriate only once a v53 checkpoint clears the promotion gates below.

## Module map

```
source/mimomix_core.py         model: attention, MoE, MTP, thinking core
source/mimomix_decoding.py     MTP self-speculative decoding
source/mimomix_controller.py   fast/deep/agent budget ladder
source/mimomix_observatory.py  Dem-Lab statistics and Q-learning feedback
source/mimomix_distill.py      GRPO domain RL + MOPD multi-teacher distillation
source/mimomix_api.py          /v1/think, /v1/models, /v1/telemetry, /health
source/benchmark_mimomix.py    end-to-end measurement with pass/fail gates
web_static/mimomix_lab.html    single-file browser observatory
```

The dependency graph is acyclic and `mimomix_observatory` imports none of its
siblings — it consumes plain telemetry dicts and tensors, so it can be tested
against hand-written input and can never be entangled with the model.

Tests: `test_mimomix_core.py`, `test_mimomix_decoding.py`,
`test_mimomix_controller.py`, `test_mimomix_observatory.py`,
`test_mimomix_distill.py`, `test_mimomix_api.py`.

---

## The model

### Hybrid attention

Layers alternate between **local sliding-window attention** and **full global
attention** at a configurable ratio. `attention_layout(n_layers, hybrid_ratio)`
makes every `(r+1)`-th layer global, giving the `r:1` interleave MiMo describes:
MiMo-V2-Flash reports 5:1 with a 128-token window, MiMo-V2.5-Pro reports 6:1.

The memory consequence is the point. Only global layers keep an unbounded KV
cache; SWA layers keep `sliding_window` entries. `hybrid_cache_footprint(...)`
computes the ratio for a given layout, and `test_sliding_window_layers_hold_a_smaller_cache`
asserts the caches really are trimmed rather than merely masked. At 36 layers,
5:1, a 128-token window and a 128K sequence, the hybrid layout holds ~6x fewer
cache entries than all-global attention — the same order as the ~7x MiMo reports
for its 6:1 configuration.

Each head also carries a **learnable attention sink**: one extra logit is
concatenated to the score row before the softmax and dropped from the weights
afterwards. A head can therefore place mass in a null slot and emit ~0 rather
than being forced to normalise over real tokens. This is the standard fix for
the sink pathology described in
[StreamingLLM](https://arxiv.org/abs/2309.17453) and analysed in
[*Why do LLMs attend to the first token?*](https://arxiv.org/abs/2504.02732).
`last_sink_mass` is published per layer.

### Progressive context extension

`RotaryEmbedding` implements three explicit policies rather than one implicit
default:

* `none` — plain RoPE.
* `ntk` — NTK-aware base rescaling, `theta' = theta · s^(d/(d-2))`.
* `yarn` — per-band interpolation ([YaRN](https://arxiv.org/abs/2309.00071)):
  high-frequency dimensions are left alone, low-frequency dimensions are
  interpolated by the full factor `s`, and a linear ramp (`beta_fast=32`,
  `beta_slow=1`) blends the band between. YaRN's logit temperature
  `0.1·ln(s) + 1` is applied inside attention.

`context_scale` is `max_position_embeddings / native_context`, mirroring the
32K-native, extended-later schedule (V2-Flash to 256K, V2.5-Pro to 1M).

**Local and global layers get decoupled rotary tables.** A sliding-window layer
can never express a dependency longer than its own window, so a large RoPE base
or a context-extension policy applied to it is at best wasted and at worst
distorts what it *can* see. Gemma 3 and MiMo therefore run two bases; v53 does
the same via `rope_local_theta` (default 10,000, unscaled) and `rope_theta`
(the global base, which carries the extension policy and YaRN's logit
temperature). `rope_scale_local=True` opts local layers back into scaling, and
`rope_local_theta=None` restores a single shared table.

This was missed in the first implementation — a single table was shared by both
kinds — and is the sort of gap that is invisible at test scale and expensive at
long context.

### Sparse MoE with auxiliary-loss-free balancing

`SparseMoEFeedForward` follows the
[auxiliary-loss-free](https://arxiv.org/abs/2408.15664) recipe used by
DeepSeek-V3:

1. score experts from the token;
2. **select** top-k by `score + bias`;
3. **weight** the selected experts by the raw `score`;
4. update `bias_i += gamma · sign(mean_load − load_i)`.

Step 3 is the load-bearing one, and it has its own test: raising the bias on
already-selected experts must leave the output bit-identical
(`test_router_bias_selects_but_never_weights`). The balancer is not allowed to
change what the model computes, only which experts compute it.
`test_bias_rule_rebalances_a_collapsed_router` starts from a deliberately
collapsed router (identical tokens, so every token picks the same pair) and
requires the rule to reach every expert.

**The bias update fires once per optimizer step, not once per forward.**
`forward` only *accumulates* load; `MiMoMixModel.step_router_bias()` applies the
update. This matters more than it looks. The rule is a control loop with step
size `gamma`, so firing it per forward makes the effective step depend on how
many forwards happen per optimizer step — gradient accumulation over `N`
micro-batches multiplies gamma by `N`, and the thinking controller probes the
same request at several budgets. An oversized effective step does not converge
faster; the sign rule overshoots the balance point every update and **rings**,
which reads on a snapshot as *worse* balance.

That is not theoretical: the first implementation updated inside `forward`, and
`benchmark_mimomix.py` measured it. At `gamma=1e-2` roughly one expert per layer
was starved; at `gamma>=5e-2` most were. Moving the update to one-per-optimizer-step
at `gamma=1e-3` took mean normalised routing entropy from 0.826 to 0.970 and
starved experts from 5 to 0 on the same task.

Two small gradient-carrying regularisers remain: a **router z-loss**
(`logsumexp(logits)²`, from ST-MoE) and a light sequence-level balance term.
`n_shared_experts` experts are always applied — the fine-grained segmentation
idea, so routed experts can specialise. The first `n_dense_layers` blocks stay
dense.

`parameter_report()` separates total from per-token-active parameters, which is
the ratio MoE models are actually described by (V2-Flash: 309B/15B;
V2.5-Pro: 1.02T/42B).

### Multi-token prediction

`MultiTokenPredictionModule` implements the sequential DeepSeek-V3 form: depth
`k` combines the previous depth's hidden state with the embedding of the token
`k` positions ahead, projects, and runs one block. Embedding and output head are
shared with the trunk, and the depths use dense FFNs — MiMo describes the MTP
module as lightweight with dense FFNs, and a routed depth would make the draft
path's cost data-dependent, which defeats the purpose.

Training adds a per-depth cross-entropy at `mtp_loss_weight`. Inference reuses
the same modules as the draft model (below).

### Recursive thinking core

`RecursiveThinkingCore` is the sequence-model port of
`CognitiveLeapV52ExpertHead`. It keeps:

* weight-tied latent refinement over up to `thinking_max_cycles` cycles;
* PonderNet/ACT halting — each cycle claims `halt_p` of the mass still in
  flight, unclaimed mass falls to the final residual, so the mixture stays
  convex regardless of which exit fired;
* a ponder cost and a cycle-to-cycle consistency term in the loss;
* trainable temperature scaling on the **verifier's own logits**, trained by
  `verifier_loss` and deliberately *not* by the language objective, so
  calibrating the verifier cannot drift the token distribution;
* supervised `p(correct)` / `p(continue)` heads, following the v52 contract:
  keep thinking exactly when the current answer is wrong.

The residual scale initialises at zero, so a freshly built model reproduces its
trunk output — `test_fresh_thinking_core_is_a_near_identity` pins that.

---

## Decoding

`mimomix_decoding.speculative_generate` runs draft/verify with the MTP depths as
the draft model. For greedy decoding acceptance has an exact form — keep a
drafted token iff it equals the trunk's own argmax — which makes the emitted
sequence **bit-identical** to plain autoregressive greedy decoding.
`assert_greedy_equivalence` checks it, and the suite runs it across seeds,
batch sizes, RoPE policies, thinking budgets, all-SWA layouts, and the
zero-MTP degenerate case.

Two details are load-bearing:

* **Cache rollback.** Rejecting `r` tokens means dropping their KV entries. A
  cache trimmed to exactly `window` has already discarded keys that rollback
  brings back into range, so the decoder requests `cache_slack = draft_length`
  extra entries. Without this, speculation on an SWA layer is silently wrong.
* **Block independence.** Verification scores several positions at once, which
  only equals one-at-a-time decoding if per-position output does not depend on
  block composition. The adaptive thinking core halts on a batch-level
  statistic, so `speculative_generate` **refuses** `adaptive_thinking=True`
  instead of quietly breaking the guarantee.

`DecodeStats.acceptance_length` is tokens committed per trunk forward. Plain
greedy scores exactly 1.0. An untrained model scores near 1.0 because its drafts
are noise — that is correct behaviour, not a bug, which is why
`test_a_learnable_pattern_produces_real_acceptance` trains a small model on a
repeating cycle first and then requires acceptance above 1.5 with strictly fewer
forwards than greedy.

---

## The thinking controller

`mimomix_controller` plans before inference and then climbs a bounded ladder.

**Plan.** `RequestFeatures` produces two independent deterministic scores —
task difficulty and epistemic risk — from observable cues only (token count,
requested acts, declared tools, conflict, evidence need). Mode follows: tools
route to `agent`, difficulty or risk route to `deep`, everything else stays
`fast`. A **safety-critical turn is not pushed to a deep budget**; the v52.1
rule is that urgent-help guidance must not be delayed by forced compute, and
`test_safety_critical_turns_are_not_forced_into_deep_compute` enforces it.

**Ladder.** For each budget in turn, an early exit requires *all* of:

1. the verifier's continue-probability below threshold;
2. calibrated confidence at or above target;
3. `P(rank 1) − P(rank 2)` at or above `decision_margin`;
4. optionally `P(rank k) − P(rank k+1)` at or above `boundary_margin`
   (default 0, i.e. off — see below);
5. the ordered top-k identical to the previous budget's (cross-budget
   agreement).

**Gate 3 measures the right boundary.** The controller originally gated on the
top-k/outside margin, inherited from v51's ten-class classifier. On a generative
next-token distribution that is a trap: a confident model puts ~1.0 on rank 1 and
spreads ~1e-5 across the tail, so the rank-k/rank-k+1 gap collapses toward zero
*exactly when the decision is safest*. Measured on the benchmark task, that
blocked every early exit — cycle reduction was −75% with 100% fidelity, i.e. the
controller paid for a ladder it never used. Gating on the rank-1/rank-2 margin
instead, which is what the emitted token actually depends on, turned the same
16-request audit into +25% cycle reduction with unchanged 100% top-1 and ordered
top-3 fidelity. The tail margin is still reported per probe and can still be
gated by callers for whom the top-k *listing* is a product surface.

The verifier can **veto** an exit but can never authorise one alone
(`test_the_verifier_can_veto_but_never_authorise_an_exit`). The first rung can
never exit while agreement is required, because there is nothing to agree with.

**Reuse, not recompute.** `decide(...)` returns the `MiMoMixOutput` object the
accepted budget actually produced — never a blend, never a patched output.
`test_accepted_output_is_the_probe_not_a_blend` re-runs that budget directly and
requires bit-equality.

**Honest accounting.** `cycle_reduction` is signed. A ladder that exits early
saves cycles; a ladder that runs to exhaustion spends *more* than a fixed
ceiling budget would, and reports a negative number. Forward *count* is
deliberately not reported as a saving — probing at budget 1 and at budget 4 are
not the same unit of work, so a forward ratio would flatter the controller.

`audit_decision_fidelity(...)` is the only basis for a savings claim. It pays
full price: for every request it runs the controller *and* the ceiling budget,
then reports top-1 and ordered-top-k disagreement. The controller alone never
sees the counterfactual it is graded against.

This follows [compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314)
and the "reason just enough" direction of
[REFRAIN](https://arxiv.org/abs/2510.10103), using bounded classifier cycles,
cross-budget agreement, and explicit policy floors rather than a chain-of-thought
bandit.

Worth stating because it cuts against the obvious intuition: stopping early is
not automatically an accuracy tax. More thinking is not monotonically better —
models reach a correct intermediate answer and then talk themselves out of it,
and REFRAIN reports improving GPQA-Diamond accuracy while cutting a third of the
tokens. So a controller that stops early can be a capability intervention, and
conversely a method that only cuts cost should be evaluated on accuracy anyway.

### Latency versus token savings

`cycle_reduction` is a **compute** saving, not automatically a latency saving.
Per-example adaptive depth gives no wall-clock benefit under naive batching,
because the batch runs to the deepest example in it: token counts drop, latency
does not. Realising the saving as latency needs batch size 1, or bucketing
requests by predicted depth. Any latency claim must be measured end to end, not
inferred from cycles.

---

## The observatory

`mimomix_observatory` keeps the questions AI-Dem-Lab's panels ask and replaces
the sandbox heuristics with statistics that can be checked against known values:

| Dem-Lab panel | v53 replacement |
| --- | --- |
| entropy / randomness bench | Shannon, min-entropy, perplexity, JSD, plus chi-square uniformity with an **exact** regularised-incomplete-gamma p-value, monobit, and runs |
| quantum-vs-LLM randomness | the same battery applied to two streams and *reported side by side*, with no verdict |
| Bell locality sandbox | `chsh_value` used as a **self-test of the harness**: classical data scoring above 2 means the statistics code is broken |
| PEAR evidence critique | `sequential_evidence` — a log-likelihood ratio carrying an explicit `ln(looks)` optional-stopping penalty, plus the effect size, because a decisive LR on a trivial effect is still trivial |
| mechanistic explorer | `routing_attribution` — expert load, normalised routing entropy, Herfindahl concentration, starved experts, per-layer sink mass |
| semantic resonance | `semantic_resonance` — cosine geometry with deterministic union-find clustering |
| RSI meters | `novelty_score`, `stability_score`, `recursive_improvement_index` |
| multi-agent ecosystem | `replicator_step` / `run_ecosystem` over controller policies |
| Q-learning feedback | `BudgetPolicyLearner` — tabular Q-learning proposing a *starting* budget per (difficulty, risk) bucket |

Every function is deterministic: no unseeded randomness, no wall-clock input, no
network. The controller consumes some of these numbers, and a non-reproducible
control signal is not a control signal.

The Q-learner's authority is narrow by construction: it proposes a **starting**
budget only. Floors, ceilings, the verifier gate, and the agreement rule all
still apply, so a badly-learned value can waste compute or add a probe — it
cannot authorise an unsafe exit. An unvisited bucket returns `None`, not a guess.

`recursive_improvement_index` is a dashboard aggregate with no units and no
ground truth. It should never gate a promotion decision on its own, and the
docstring says so.

---

## Post-training

`mimomix_distill` implements the two stages that follow SFT.

**Domain-specialised RL.** `group_relative_advantages` centres rewards within a
sampled group so the group itself is the baseline, no critic required.
`normalise_by_std` is exposed with both behaviours documented: dividing by the
group standard deviation is original GRPO but up-weights already-solved prompts,
which the Dr.GRPO line drops. `grpo_loss` is the clipped surrogate and reports
clip fraction, the diagnostic that says whether the trust region is doing
anything.

**MOPD.** The student samples its own trajectory (on-policy, so it learns on the
states it actually visits) and every teacher scores **every position** — the
dense token-level signal MiMo highlights, versus sequence-level RL where a whole
rollout collapses to one scalar. The objective is reverse KL to a per-token
teacher mixture:

```text
L = E_{x ~ student}  sum_i  KL( student(· | x_<i) || teacher_mix(· | x_<i) )
```

Reverse KL is mode-seeking: the student is punished for putting mass where the
teachers put none, which is what merging specialists requires. The mixture is
formed in probability space, not by averaging logits — a logit average is a
geometric mean that silently suppresses any token one teacher dislikes.

Three weightings: `uniform`, `confidence` (weight each teacher by how much
probability it gave the token the student actually emitted, so specialists
dominate their own domain without needing labels), and `domain` (explicit).
`min_teacher_weight` floors any single teacher's share so a confident sibling
cannot switch a specialist off entirely.

Only generated positions are scored. The prompt prefix was not sampled
on-policy, so distilling on it would reintroduce exactly the distribution
mismatch MOPD exists to remove.

---

## The API

`/v1/think` is one endpoint, one router, two backends — the shape sketched in
`readme task.txt`, wired to the real stack:

* routing uses the controller's deterministic plan, so the API and the runtime
  obey the same rules;
* the response reports the accepted budget, the exit reason, every probe, and
  the signed cycle reduction;
* generation runs through MTP speculative decoding and reports a measured
  acceptance length.

**Trust boundary.** Message content is data. Only typed request fields steer the
service — nothing in a prompt can change the mode, the budget, the backend, or
the tool decision. `test_message_content_cannot_change_the_route` fires a direct
instruction-injection payload and requires the routing to match a benign prompt
of identical length. Declared tools are *planned for* but never executed; the
response says so in `warnings`.

The service binds `127.0.0.1`; remote exposure needs an explicit `--host` plus
its own authentication and network controls.

---

## Measured results

`source/benchmark_mimomix.py` trains a 1.2M-parameter model on a synthetic
periodic task and measures the stack end to end. Run it yourself:

```bash
python source/benchmark_mimomix.py --steps 250
```

Observed on CPU (6 layers, 5:1 layout, 8-token window, 8 experts top-2, 3 MTP
depths, 250 steps, ~143s):

| measurement | value |
| --- | --- |
| parameters | 1,205,769 total / 485,469 active per token |
| KV cache vs all-global @ 1M tokens | **6.00x smaller** (83.3% saved) |
| LM loss | 2.947 → 0.000 (uniform baseline 2.773) |
| routing entropy (8 batches accumulated) | **0.970** normalised, 0 starved experts |
| MTP acceptance, untrained | 1.5% rate, 1.067 acceptance length |
| MTP acceptance, trained | **100% rate, 4.000 acceptance length** |
| trunk forwards, trained | **12 vs 47 greedy (−74.5%)** |
| speculative output == greedy | **true, both before and after training** |
| thinking ladder, 16 requests | **+25.0% cycle reduction** |
| decision fidelity | **100% top-1, 100% ordered top-3** |
| exits | 16/16 `cross_budget_agreement` |

Read these as *the mechanisms work*, nothing more. The task is a periodic
sequence chosen precisely because a tiny model can learn it in seconds, which is
what makes the MTP numbers meaningful at all — an untrained draft is noise, and
the untrained row shows exactly that. An acceptance length of 4.000 is the
arithmetic maximum for 3 MTP depths and reflects a model that has memorised a
5-token cycle. It is not a claim about text.

The KV-cache row is arithmetic over the layout, not a memory benchmark, and it
says nothing about the quality cost of restricting most layers to a small
window.

## What this does **not** prove

The suites prove integration, gradient flow, and the specific invariants named
above. They do not prove any of the following, and no number in this repository
should be read as evidence of them:

* **That the model is good at anything.** The default backends are randomly
  initialised. Their text is noise by design; the API says so in its own
  response.
* **That MiMo's published numbers transfer.** 309B/15B, 1.02T/42B, 27T training
  tokens, 3.6 acceptance length, 2.6x decoding speedup, ~7x KV reduction, 97%
  tool-call accuracy — these describe Xiaomi's checkpoints. v53 implements the
  same *mechanisms* at a scale five orders of magnitude smaller. The mechanism
  transferring does not mean the result does.
* **That the hybrid cache saving is free.** `hybrid_cache_footprint` counts
  cache entries at steady state. It says nothing about quality lost by
  restricting most layers to a 128-token window; that is a training question
  this repository has not answered.
* **That the controller saves compute on real traffic.** `cycle_reduction` is
  measured per request against that request's own ceiling. Whether the ladder
  wins on average depends entirely on how often the gates fire, which depends on
  a trained model's calibration. On an untrained model the gates correctly
  refuse every early exit and the ladder is a net *loss*.
* **That the appraisal or verifier outputs are semantic.** Named heads are not
  meaningful because they exist. They require labelled auxiliary supervision and
  held-out evaluation, exactly as the v52 document states for its ancestors.
* **That the observatory measures cognition.** It measures a running system's
  telemetry. Those measurements are only as meaningful as the model producing
  the telemetry.
* **That MOPD improved anything.** The losses are implemented and tested for
  correctness and for reducing KL on a fixed target. No checkpoint has been
  trained, promoted, or evaluated against a frozen baseline.

## Promotion gates

A v53 checkpoint would need, at minimum, the same evidence bar the v52 line
carries: held-out accuracy and NLL against a frozen baseline; ECE, Brier,
selective accuracy and risk-coverage; decision-fidelity audit at the intended
policy with zero top-1 and ordered-top-3 disagreements; measured acceptance
length and end-to-end latency on the target hardware; long-context evaluation at
the claimed extension factor, not just at native length; and source/package
parity. None of that has been run.
