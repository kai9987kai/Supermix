# Supermix v59 — the mechanism causality audit

V58 ran a thinking-core ablation, reported tier deltas between 0.0006 and 0.007
nats, called the result *"no measurable effect on text quality"*, and said in the
same document that this sat below a noise floor which **"has not itself been
quantified"**. It listed quantifying that floor as open work.

The floor was the wrong thing to chase. V59 measures the mechanism instead, and
the mechanism is **inert**:

| | v58_full checkpoint |
| --- | --- |
| Δ held-out loss, thinking core on vs. off | **+8.84e-08 nats** |
| held-out predictions changed | **0 of 12,192** |
| argmax agreement | **1.000000** |
| smallest delta v58 reported as a finding | 5.9e-04 nats |
| ratio | **6,673×** |

V58's two arms differ in baseline held-out loss by 9.5e-04 nats — about **10,700×**
the entire causal contribution of the mechanism they differ by. So the arms were
functionally the same model, and v58's deltas measure run-to-run variance. Its
published conclusion is correct, but not for the reason given, and the
"unquantified noise floor" caveat is answered: **those deltas were the noise.**

## Why the mechanism is inert — and why the obvious answer is wrong

The whole recursive core is multiplied by one scalar,
`RecursiveThinkingCore.residual_scale` (`source/mimomix_core.py`):

```python
scale = self.residual_scale + (1e-4 if self.training else 0.0)
refined = flat + scale * residual_mixture
```

That scalar is initialised to exactly `0.0`, and in `v58_full.pt` it had reached
only **6.410e-04** after 1,000 steps. The obvious hypothesis follows: the gate
multiplies the core's own gradient as well as its output, so starting at zero
starves the mechanism of any path to learn along.

**That hypothesis is false, and this section exists because it was tested.** Two
matched 400-step runs, identical seed and settings, differing only in the gate's
initial value:

| | gate init 0.0 | gate init 0.1 |
| --- | --- | --- |
| final `residual_scale` | **−2.082e-02** | +4.596e-02 |
| thinking-core Δ loss | +5.082e-05 nats | +2.420e-06 nats |
| decisions changed | **35 / 12,192** | 8 / 12,192 |
| verdict | **active** | **active** |

```bash
python source/train_mimomix_talk.py --steps 400 --seed 59 --run_name v59_gate_zero --output_dir output/v59_gate_zero --thinking_residual_init 0.0
python source/train_mimomix_talk.py --steps 400 --seed 59 --run_name v59_gate_warm --output_dir output/v59_gate_warm --thinking_residual_init 0.1
python source/mechanism_causality.py --checkpoint output/v59_gate_zero/v59_gate_zero.pt
```

Starting from exactly zero, the gate reached −2.08e-02 — **32× the magnitude v58
reached in 1,000 steps** — and the core became causally *active*. So a zero
initialisation does not prevent the mechanism from coming alive.

Worse for the hypothesis, warm-starting made the core **less** active, not more:
8 changed decisions against 35, despite ending with a *larger* gate (0.046 vs
−0.021). Gate magnitude alone does not determine causal contribution; the
direction the core writes in matters too, and the warm start evidently landed
somewhere less useful.

**What this leaves.** V58's core is inert — that is measured directly and is not
in doubt. But the cause is not the initialisation. These arms differ from v58 in
width (256 vs 192), depth (6 vs 4 layers) and schedule (400 vs 1,000 steps), so
they refute the initialisation explanation without identifying the real one.
Why v58's specific configuration held the gate at 6.41e-04 is **open**.

Two related findings fell out of the same inspection, both verified in the
trained checkpoint:

* `verifier_loss` is never called by either trainer, so `quality_encoder` and
  `quality_head` (4,482 parameters) receive no gradient. In `v58_full.pt`,
  `quality_head.bias` and `log_temperature` are still bit-exactly `0.0`.
* `self.apply(self._init_weights)` runs *after* the core's own initialisers, so
  the deliberate `nn.init.zeros_(self.quality_head.weight)` is silently replaced
  by `normal_(0, 0.02)`. The checkpoint agrees: that weight is random and
  untrained.

## The instrument

`source/mechanism_causality.py` takes one trained checkpoint, intervenes on one
mechanism, and re-scores the same tokens. Nothing is retrained, so everything
except the mechanism is held bit-identical — which is exactly what a retraining
ablation cannot do, because a retrain moves every weight.

```bash
python source/mechanism_causality.py --checkpoint output/v58_full/v58_full.pt --output output/v59_causality/v58_full_causality.json
```

On `v58_full`, over 8,284 supervised tokens:

| mechanism | Δ nats | decisions changed | verdict |
| --- | --- | --- | --- |
| `moe_routing_inverted` | +5.756e-02 | 1,056 / 12,192 | active |
| `moe_routing_random` | +4.160e-02 | 905 / 12,192 | active |
| `moe_shared_expert` | +4.460e-03 | 595 / 12,192 | active |
| `thinking_core` | +8.841e-08 | 0 / 12,192 | **inert** |
| `mtp_main_path_leak` | +0.000e+00 | 0 / 12,192 | inert (expected) |

The same audit on `v58_ablation` reports `thinking_core` **absent** and routing
active at +3.299e-02 — so the ranking is a property of the architecture, not of
one checkpoint.

**Sparse-MoE routing is the load-bearing mechanism of the v53 stack.** Destroying
the learned assignment costs 470,000× what the thinking core contributes, and the
inverted bound (+5.756e-02) exceeds the random cost as it must. That ordering is
a consistency check on the instrument: an upper bound that came in below the
random case would mean the intervention was wrong.

### Three things make the verdicts falsifiable

1. **The threshold is not chosen.** A mechanism is `active` if it changes at
   least one argmax **or** moves loss by at least 5.9e-04 nats — the smallest
   effect this project has ever published as a finding. Both tests must fail
   before anything is called inert. Neither alone survives: the numerical floor
   is 3.68e-09 nats, tight enough that a dead mechanism can sit 24× above it,
   while on a randomly initialised model a *fully open* thinking core moves loss
   by 2.4e-03 nats and still changes zero decisions.
2. **The instrument self-checks.** Routing interventions run through a
   reimplemented MoE forward. `IDENTITY` runs that same rebuild without changing
   the expert choice and must reproduce the baseline bit-exactly; if it does not,
   the audit raises rather than reporting a transcription bug as a mechanism
   effect. `test_self_check_raises_when_the_rebuild_is_unfaithful` sabotages the
   rebuild by 5% and requires the raise.
3. **Restoration is verified.** After every intervention the baseline is
   re-scored and must match exactly, so a leaked patch voids the run instead of
   contaminating the next verdict.

The audit also reports nulls as results: `mtp_main_path_leak` is a leak test, not
an ablation. It runs the speculative MTP chain during scoring and requires the
main-path logits not to move. They do not, exactly — so MTP is side-effect-free
on the scored path, and every published tier loss is independent of whether the
heads ran.

## The knob, and why it is not a fix

`MiMoMixConfig.thinking_residual_init` (default `0.0`) sets the gate's starting
value, exposed as `--thinking_residual_init`. The default reproduces every
pre-v59 checkpoint exactly — verified: a default-built model's gate is `0.0`, and
`v58_full.pt` still loads with its trained `6.410068e-04`.

It was added to test the initialisation hypothesis, and **the test came back
negative**: warm-starting produced a *less* active core than the zero-init
control. The knob is kept because it is the instrument that produced that
result and makes it reproducible, not because it improves anything. On this
evidence there is no reason to set it above zero, and the default stays at the
value that reproduces v57 and v58.

That is the whole point of having run it. Shipping the knob with the
plausible-sounding rationale and no experiment would have added a tuning
parameter this project has no evidence for.

## What this does not prove

* **That the thinking core is a bad mechanism.** It shows this core, in *these*
  checkpoints, contributes nothing. A 400-step run at a different width and depth
  produced a core that is causally active, so inertness is a property of v58's
  particular training run, not of recursive latent reasoning.
* **That we know why v58's core is inert.** The initialisation hypothesis was
  tested and refuted. The arms that refuted it differ from v58 in three ways at
  once (width, depth, steps), so they isolate nothing further. The cause is open.
* **That the active cores are *useful*.** 35 changed decisions out of 12,192 means
  the mechanism does something, not that it does something good. Neither arm was
  compared against a parameter-matched model without a thinking core, and their
  held-out losses (0.2502 and 0.2481) are worse than v58's 0.2316 — which is
  expected at 400 steps versus 1,000 and is not a comparison either.
* **That intervening equals removing.** Zeroing a gate in a trained model is not
  the same as training without the mechanism; the rest of the model would adapt.
  These are counterfactuals on a fixed checkpoint, which is why they can isolate
  a mechanism, and also why they cannot answer "should it exist".
* **That "active" means important.** The routing numbers say destroying a learned
  assignment costs loss. They do not say the MoE is better than a dense layer of
  equal parameter count — no such comparison was run.
* **That any of this transfers.** One architecture, two checkpoints, one 292-word
  templated corpus, 8,284 scored tokens. Held-out loss is the only quantity
  measured: nothing here evaluates reply quality, latency, or any downstream task.
* **That the noise floor is fully quantified.** The 3.68e-09 nat floor is
  arithmetic order only, and is a *lower* bound on the seed-to-seed floor, which
  is larger and still unmeasured. It is sufficient here only because the thinking
  core's effect is below it on the decision axis as well.

## The same failure, one level up: tests that never run

The thinking core was a mechanism that looked live and did nothing. The test
suite has the same shape of problem, and it is measurable:

| | count |
| --- | --- |
| functions in `test_*.py` that pytest never collects | **30** |
| files affected | 20 |
| files that collect **nothing at all** | **17** |

pytest's default `python_functions = test*` does not match `smoke_test_*`, so
those functions are inert. Seventeen files — almost all of the 2026-06-16 expert
cohort — are test files by name that run zero assertions. The worst single case
is `test_runtime_compute_controls.py`: **8 dark functions in a file CI actually
runs**, so the suite has been reporting green on checks that never execute.

```bash
python source/dark_test_audit.py            # report
python source/dark_test_audit.py --check    # CI gate
```

The gate pins a baseline rather than failing on all 30, because a gate that is
red from the first commit gets ignored. The contract is *no new dark tests*: the
count may fall freely and any rise fails. `test_dark_test_audit.py` plants dark
functions in a temporary directory and requires the detector to find them,
including the mixed case where a collected file hides uncollected checks.

**This does not fix the 30.** It stops the 31st, and makes the debt a number
someone can drive down instead of an invisible property of the suite. Enabling
the existing dark functions is separate work and, on the evidence of a trial
run, some of them fail.

## Retired from v58's open list

> "no seed-to-seed spread has been measured — so 'no measurable effect' means the
> effect is below a noise floor that has not itself been quantified"

Answered, by removing the need for it: the mechanism's causal contribution is
8.84e-08 nats and 0 changed decisions, so no floor estimate is required to
conclude it did nothing. The seed-to-seed floor remains unmeasured and still
matters for any *other* v58 comparison.

v58's remaining gates — corpus diversity beyond 292 word types, rubric or human
evaluation, safety evaluation, latency on target hardware, source/package parity
— are untouched by this work and remain unmet.
