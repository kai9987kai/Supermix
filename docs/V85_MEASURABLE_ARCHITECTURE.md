# v85 — Making the next run measurable, and finding out the ruler had moved

Nothing here was trained. v85 is a release about instruments: four confirmed
architecture bugs, one benchmark that could not see the task it was scoring, and
one measurement that changes how the v79→v80→v85 comparison has to be read.

This is the **training line** (v58–v81). The v82–v84 numbers were taken in
parallel by the NexusMind evidence line; nothing here touches those files.

Every number below was measured on this machine against the shipped
`output/v80_omni/v80_omni.pt`, with the model's weights unchanged.

---

## 1. The benchmark had been asking easier questions

This is the finding that matters most, and it was not what it looked like.

The last commit, `c7041897`, rewrote the three tasks that scored 0.00 —
`kinetic_energy`, `combination` and `arithmetic_series` — so their working stayed
inside what v80 empirically proved learnable. That was the right call. But the
benchmark draws its problems from the same generators, so rewriting the corpus
rewrote the exam.

v80's weights never changed. Only the generator version did:

| task | `a5bd5bf2` (what v80 trained on) | `74642029` (v81) | `c7041897` (current) |
|---|---|---|---|
| `arithmetic_series` | 0.000 | 0.167 | **0.500** |
| `kinetic_energy` | 0.167 | 0.167 | **0.833** |
| `combination` | 0.000 | 0.000 | 0.000 |
| `force` | 0.917 | 0.917 | 1.000 |

n = 12 per cell. The Wilson 95% interval at that size is ±25 points, so a single
cell is weak evidence and only a large move is readable. `kinetic_energy` moves
67 points and `arithmetic_series` 50; both clear it. `force` is the control and
does not move.

`eval_problem_solving.wilson_interval` is now in the receipt, so this stops being
something a reader has to work out. At the sizes this line actually uses:

| n | 95% half-width |
|---|---|
| 3 (per task in a quick sweep) | ±37 points |
| 12 (per era cell above) | ±25 points |
| 30 (per task at n=630) | ±17 points |
| 630 (a full benchmark) | ±4 points |

The per-task rows of every published `problem_solving_n630.json` are n = 30, so
a per-task change under about 17 points has never been distinguishable from
noise in this project's headline reports.

The cause is visible in the prompts:

```
kinetic_energy   a5bd5bf2:  "What is the kinetic energy of a 142 kg body at 74 m/s?"   -> 388,796
                 c7041897:  "mass 10 kg velocity 7 m/s kinetic energy"                 ->     245

arithmetic_series a5bd5bf2: starts at 54, difference 26, first 33 terms                ->  15,510
                  c7041897: starts at 14, difference 5,  first 10 terms                ->     365
```

V81 said this plainly at the time — *"This narrows the task definition, and the
benchmark narrows with it"* — and was right to narrow it. What had not been
measured is **how much the narrowing is worth on its own**. On these two tasks it
is worth most of the gap.

**The consequence for the next run.** If v85 scores `kinetic_energy` at 0.85, that
is not an improvement on v80's published 0.00. v80 already scores 0.83 on those
questions. A v80-versus-v85 comparison is only paired if both are scored against
the *same* generator version, and the published `problem_solving_n630.json` files
were not. `eval_problem_solving.py` now records the seed, the task list, the
generation cap and a generator fingerprint in the receipt so two receipts can be
checked for pairing instead of assumed to be comparable.

**`combination` is the opposite case, and the honest one.** It reads 0.000 in
every era. Asked the new formula-based question, v80 still answers in the format
it memorised:

```
asked:  "In how many ways can 2 items be chosen from 18?"      (truth 153)
target: combinations = n x (n - 1) / 2, 18 - 1 = 17, 10 x 17 = 170, ...
v80:    combinations = n choose k, 18 choose 2 = 164, total 164
```

It is reciting the old template and guessing the value. That task genuinely needs
the retrain; the other two had already partly arrived without one.

---

## 2. The probe could not see the task it was scoring

`train_mimomix_generalisation.py` capped the mid-run accuracy probe at 64 new
tokens. Measured with the v80 tokenizer over the current generators:

| task | median | max | fraction over 64 |
|---|---|---|---|
| `arithmetic_series` | 81 | 84 | **1.00** |
| `combination` | 60 | 65 | 0.22 |
| `kinetic_energy` | 52 | 56 | 0.00 |
| everything else | ≤ 38 | ≤ 38 | 0.00 |

`eval_problem_solving.py`'s own docstring reports different figures (median 92 for `arithmetic_series`, 88 for
`work`) because it measured `datasets/v80/v80_combined.jsonl`, built from the
pre-`c7041897` generators whose operands were larger. Both are correct for their
corpus, and neither transfers to the other — which is section 1's point arriving
again from a different direction.

Every `arithmetic_series` reply is cut off before it can finish. Under
`--select_on accuracy` the run would have selected against a signal that reads
0.00 on that task **whatever the model learned**, and the offline benchmark's
default of 40 tokens truncated more still.

Re-scoring v80 at four caps, same problems, same weights:

| `max_new_tokens` | accuracy | truncated replies |
|---|---|---|
| 40 (the old default) | 0.6349 | 18 |
| 64 (the old probe cap) | 0.6667 | 3 |
| 96 | 0.6667 | 0 |
| 128 | 0.6667 | 0 |

n = 63. The accuracy move is two problems and is inside the noise; the truncation
count is not. One task, `wave_speed`, goes 0.000 → 0.667 on the cap alone.

This is the same mistake as V67 losing the `average` rows and it cost the same
thing: a task reading zero for a reason that has nothing to do with the model.
**The deliverable is the guard, not the number.** The cap is now
`--probe_max_new_tokens` (default 112), the offline default is 96, and the
trainer refuses to start blind:

```
!! PROBE TOKEN BUDGET WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   --probe_max_new_tokens is 64.
   BLIND   arithmetic_series   median   81  p95   84  max   84  -- more than half of
   this task's replies cannot finish inside the cap, so its probe accuracy is
   structurally 0.00 whatever the model learned
   AT RISK combination         median   60  p95   65  max   65
```

Under `--strict` that is a `SystemExit` before the first step rather than a
warning twenty hours later. `build_omni_corpus.py --token_budget_report` writes
the same histogram into the corpus report, so a format change that outgrows the
budget is visible when the corpus is built.

---

## 2b. The memorisation control was measuring a corpus the model never saw

Found by reading the baseline receipt above, which reported a **negative**
memorisation gap: v80 scored 0.360 on "seen" problems and 0.630 on novel ones.
A model doing worse on its own training rows than on unseen ones is not a
result, it is a broken instrument.

`eval_problem_solving.py`'s `--corpus` flag, which supplies the "seen" arm, was
hard-coded to `datasets/v62/english_math_40k.jsonl`. v80 trained on
`datasets/v80/v80_combined.jsonl`. The seen arm was drawing rows from a corpus
the checkpoint had never been trained on, and the difference was being published
as `memorisation_gap`.

That default dates from v62, so **every receipt written after v62 whose run used
a different corpus has a meaningless memorisation gap** — including v74's, v79's
and v80's. The benchmark's own `NON_CLAIMS` block says "a high seen score with a
low novel score is recall, not skill; the gap is the finding". The gap was not
measuring that.

Three changes, all in the direction of failing visibly:

* `--corpus` has **no default**. Omit it and the seen arm is skipped, the reason
  is printed, and `memorisation_gap` is `null`. A missing measurement reads as
  missing; a wrong one reads as a finding.
* Checkpoints now record `corpus_jsonl` in their `extra` block, so a checkpoint
  carries the corpus it trained on.
* When a checkpoint records its corpus and the seen arm was drawn from a
  different file, the gap is **withheld** and the receipt names both files in
  `memorisation_gap_withheld`.

v80 and v74 predate the provenance field, so their `checkpoint_trained_on` reads
`null` and their gaps cannot be recovered — only recomputed, by passing the right
corpus explicitly.

---

## 2c. v80's regression against v74 is real; the cause is probably exposure

Earlier versions of this document listed v80's drop against v74 on their nine
shared tasks as unexplained, and suggested the ruler might account for it. It
does not, and the honest thing is to say so plainly.

**The comparison had never been paired.** Until v85 the benchmark drew every
task's problems from one RNG shared across tasks in turn. v74 was scored with 10
tasks registered and v80 with 21, so the same seed produced different problems
for every shared task. Verified directly: with the v85 per-task RNG, `average`
problems are identical whether drawn from a 9-task or 21-task list; with the
legacy shared RNG they are not. v74's receipt also records no generation cap at
all.

So v85's determinism fix is what makes the real comparison possible. Nine shared
tasks, one seed, **one identical problem list**, both checkpoints, cap 96:

| task | v74 | v80 | delta | *(published, unpaired)* |
|---|---|---|---|---|
| `average` | 0.700 | 0.033 | **−0.667** | *0.589 → 0.033* |
| `algebra_one_step` | 0.900 | 0.433 | **−0.467** | *0.893 → 0.300* |
| `arithmetic` | 0.900 | 0.700 | −0.200 | *0.893 → 0.633* |
| `word_problem` | 0.933 | 0.733 | −0.200 | *0.964 → 0.867* |
| `division` | 1.000 | 0.833 | −0.167 | *1.000 → 0.600* |
| `two_step` | 0.800 | 0.633 | −0.167 | *0.982 → 0.733* |
| `percent` | 0.667 | 0.633 | −0.033 | *0.750 → 0.600* |
| `sequence` | 0.967 | 0.933 | −0.033 | *0.982 → 0.833* |
| `multiplication` | 1.000 | 1.000 | 0.000 | *1.000 → 1.000* |
| **overall** | **0.874** | **0.659** | **−0.215** | *0.894 → 0.622* |

**The paired gap is −0.215, and that is the number.** An earlier version of this
section subtracted it from the published −0.272 and concluded "the ruler accounts
for about six of twenty-seven points". That subtraction is not legitimate: the
published pair comes from two runs on different problems at an unrecorded cap,
and this document's own receipt says they must not be differenced against the
paired figures. The 5.7-point residual is also confounded with the cap change and
with v74's own movement between the two measurements — on the paired list v74
gains 11 points on `average` and loses 18 on `two_step` relative to its published
row — so it cannot be attributed to the RNG sharing.

What survives is stronger for being simpler: **on one identical problem list, v80
is 21.5 points behind v74**, and that is a genuine regression rather than a
measurement artifact. It is concentrated in two tasks.

### Two mechanisms are confounded, and one free test does not separate them

v80 changed two variables at once, and the literature offers a distinct account
for each:

* **Exposure dilution** — per-task exposure fell 1.84× at a fixed step budget.
* **Capacity interference** — twelve tasks were added to a 15M-parameter model.
  The curse-of-multilinguality work (arXiv 2311.09205, >10,000 models up to 45M
  parameters) shows incumbent groups degrading *at constant exposure*, from
  constrained capacity alone. **No sampling weight repairs that one.**

The two make different predictions about the *shape* of the damage: threshold
crossings should give a bimodal per-task profile, capacity a roughly uniform
one. That test is free, and the numbers are already above. Computed over all
nine tasks:

```
Sarle bimodality coefficient  0.285   (> 0.555 would suggest bimodal)
skew  -1.08     kurtosis 3.04
worst two tasks carry 58.6% of the total drop
3 of 9 tasks within 5 points of unchanged
```

**It is skewed, not bimodal**, so the test does not separate the accounts — a
heavy left tail sitting on a broad mild decline is what both running together
would look like. At n = 9 this was never going to be decisive, and saying so is
better than reading a preferred mechanism into it.

Worth recording because it constrains every proposed fix: **every task in v80
holds exactly 40,000 rows**, so cap-K proportional mixing, temperature-scaled
mixing and mT5-style `p ∝ n^α` are all *exact no-ops* on this corpus. They
reduce to the uniform sampling already in place. A dilution fix has to change
the step budget, the batch size, or the per-task sampling weight directly.

Context for why exposure should be first-order at all: this run sees
18,000 × 16 × 128 = 36.9M tokens against 15M parameters, about 2.5 tokens per
parameter, roughly **0.12× Chinchilla-optimal**. Nothing has saturated, so every
task is on the steep part of its curve.

### The exposure arithmetic

v80 is the larger model: 15.3M parameters against v74's 8.6M, 48 routed experts
against 32, so raw capacity did not shrink. What certainly changed is how often
it saw each task.

Both corpora carry **exactly 40,000 rows per arithmetic task**. But v80's corpus
added twelve science tasks, so the same 40,000 rows fell from 8.06% of the
corpus to 4.39%. Both runs then trained for the same **18,000 steps at batch 16**:

| | rows | epochs | sequences seen per arithmetic task |
|---|---|---|---|
| v74 | 496,108 | 0.581 | **~23,212** |
| v80 | 911,478 | 0.316 | **~12,643** |

**v80 saw each arithmetic task 54% as often as v74 did.** The corpus doubled and
the step budget did not follow it. That is a dilution effect at constant compute,
and it is consistent with the shape observed: the tasks that were already hard —
`average` and `algebra_one_step` — collapse, while `multiplication`, which v74
scored 1.000 on, survives at 1.000. Consistent with, not demonstrated by: the
capacity account above predicts a broad decline too, and the bimodality test
cannot tell them apart at this sample size.

It also names three fixes that cost nothing to state and can be measured
separately: scale steps with corpus size, weight sampling per task so a task's
share is a decision rather than a by-product of how many tasks exist, or accept
the trade knowingly and say so in the receipt.

**One fix that looks obvious and is not, plus one that works.** A research pass
recommended `--batch_size 8` on the grounds that per-token cost is flat in batch
size on this machine, so halving the batch would double optimiser updates per
hour for free. Measured properly — all four batch sizes interleaved in one
process, ratio of medians — it is wrong twice over:

| batch | s/step | per-sequence cost | updates/hour | **sequences/hour** |
|---|---|---|---|---|
| 4 | 0.912 | 1.358× | 3,949 | 15,796 (0.74×) |
| 8 | 1.532 | 1.142× | 2,349 | 18,793 (0.88×) |
| **16** (v80) | 2.685 | 1.000× | 1,341 | 21,453 (1.00×) |
| 32 | 4.470 | **0.832×** | 805 | **25,771 (1.20×)** |

Per-sequence cost is not flat: batch 8 costs 14% more per sequence and batch 4
costs 36% more. Larger batches are *more* efficient here, not less.

More importantly, the recommendation optimises the wrong quantity. Exposure —
what the dilution finding above is measured in — is **sequences seen per task**,
not optimiser updates. Batch 8 does buy 1.75× the updates, but it processes 12%
**fewer** sequences per hour, so it makes the exposure problem slightly worse.
The move that helps is the opposite one: **`--batch_size 32` gives 20% more
sequences per hour than v80's 16**, which is 20% more exposure at identical wall
clock and identical corpus.

That is a throughput result, not a quality one. A larger batch is a different
optimisation trajectory and at 0.12× compute-optimal the learning-rate
relationship is not obvious, so it is an arm to run, not a default to change.

There is a connection to section 7, but it is weaker than an earlier version of
this document claimed. **Removing the MTP heads entirely** (`--n_mtp_layers 0`)
is measured at 0.550×, which at equal wall clock is about 1.8× the steps —
roughly the exposure v80 lost. Keeping one depth measured 0.845 and did not
resolve, so it buys back somewhere between nothing and most of it. Either way
this is a lever on the step budget, not a demonstrated fix: no run has tested
whether restoring exposure restores the score.

---

## 3. Four architecture bugs, each confirmed by measurement

All four are in code no trained checkpoint uses, which is why they could be fixed
outright rather than behind a flag.

**The MLA rotary embedding was not a rotation.** `MultiLatentAttention` sliced
`cos[:, :pe_dim]` out of a table built as `cat([freqs, freqs])` over `head_dim/2`
frequencies, so a component was paired with its `rotate_half` partner at a
*different* frequency.

| | sliced table | correctly sized table |
|---|---|---|
| norm drift (a rotation must give 0) | **2.2866** | 4.77e-07 |
| score at offset 3, positions (5,2) | +3.9137 | −0.7673 |
| score at offset 3, positions (15,12) | +7.5882 | −0.7673 |

The score is supposed to depend only on the relative offset. It did not. Fixed by
generalising `RotaryEmbedding` to build its table over a configurable
`rotary_dim`, which also delivers partial RoPE for the hybrid layers.

**MLA broke speculative decoding.** `trim_past` assumed a 4-D `(B,H,T,D)` cache
and sliced dimension 2, which for MLA's 3-D `(B,T,latent)` cache is the latent
dimension. `speculative_generate` raised `RuntimeError`, and `generate_reply`
defaults to `speculative=True`, so the benchmark path was broken for any MLA
model. Now dispatched on `ndim`.

**Mixture-of-Depths trained and decoded differently.** Capacity was
`ceil(seq_len * ratio)` per call, so a cached single-token step had capacity
1 ≥ 1 and selected everything.

| sequence length | tokens selected | skip ratio |
|---|---|---|
| 8 | 4 of 8 | 0.500 |
| 4 | 2 of 4 | 0.500 |
| 1 (cached decode) | 1 of 1 | **0.000** |

Full-forward and incremental-decode logits differed by 8.1e-02 against a 1.2e-07
baseline. Selection was also non-causal: top-k over the whole sequence means
changing token 7 of 8 moved positions 0–3 by 0.519. Fixed with the MoD paper's
causal predictor (`mod_causal_predictor`, arXiv 2404.02258).

**Weight init overwrote the thinking core's deliberate zeros.**
`self.apply(_init_weights)` ran after the submodules and replaced
`RecursiveThinkingCore`'s intended `zeros_(quality_head)` and `std=0.01` with
`normal(0, 0.02)`. Measured after construction: `quality_head.weight.abs().sum()`
was **1.2718** where 0 was intended. Recorded in V59 and still live.

---

## 4. The thinking core is inert on a problem-solving checkpoint too

V59 measured the recursive core inert on v58 — 0 of 12,192 held-out predictions
changed — and was careful to say that inertness is a property of one run, not of
the mechanism. It had never been re-measured on a problem-solving checkpoint.

Sweeping the cycle budget on v80, same 63 problems:

| `thinking_cycles` | 1 | 2 | 3 | 6 |
|---|---|---|---|---|
| correct | 42/63 | 42/63 | 42/63 | 42/63 |

Not close — identical. v80 kept `thinking_residual_init` at 0.0, which gates the
core's own gradient path to zero, so this is what the V59 mechanism predicts.

There is a neat corroboration in the bug-fix above: repairing the `quality_head`
initialisation changes the parameter tensor, and the model's output logits are
still **bit-identical** (delta 0.000e+00). The core's weights genuinely do not
reach the output. `--thinking_residual_init` is now exposed on the trainer, so the
warm-gate arm V59 called for can actually be run.

---

## 5. Two things that turned out fine

**Speculative decoding is exact on a trained checkpoint.** Greedy and MTP
self-speculative decoding produced byte-identical output on 30 of 30 problems.
Previously this had only been checked against random-init models, where a
constant output makes the check vacuous.

**The v79 phrasing decoupling partly worked.** v74 scored 0.894 on its own format
and 0 of 5 typed naturally. v80 answers **5 of 10**:

| typed question | v80 | truth |
|---|---|---|
| "what's the force on a 12 kg object accelerating at 3 m/s^2" | 36 | 36 |
| "how much momentum does a 14 kg trolley moving at 5 m/s have?" | 70 | 70 |
| "Work done pushing with 20 N over 7 metres?" | 140 | 140 |
| "A 9 volt battery drives 3 amps. What's the power?" | 27 | 27 |
| "What is 47 x 6?" | 282 | 282 |
| "A 30 kg mass is pushed with 90 N. How fast does it accelerate?" | 2700 | 3 |
| "If something weighs 25 kg and speeds up at 4 metres per second squared…" | 153 | 100 |
| "what is 47 times 6" | *no answer* | 282 |
| "Find the average of 61, 63, 72 and 61." | *no answer* | 64.25 |
| "Solve for x: x + 29 = 34" | 4 | 5 |

n = 10, so this is an indication, not a rate. The failures are diagnostic rather
than random: 2700 is 30 × 90, so the model picked the wrong operation; `times`
still fails where `x` succeeds, which is the operator brittleness
`prompt_normaliser.py` exists to bridge and which was not applied here; and
`x + 29 = 34 → 4` is the same one-step borrow error that keeps
`algebra_one_step` at 0.30.

### The normaliser had no physics

**Two** of those five failures are physics questions (`force` and
`acceleration`); the other three are `multiplication`, `average` and
`algebra_one_step`. An earlier version of this section said three, which was
wrong. `prompt_normaliser.py` covered only arithmetic, It already rewrites `what is 47 times 6` into
`What is 47 x 6?` and turns a loose average into the corpus form, but it passed
every physics question through untouched.

v85 adds eight science rules, mapping the way a person writes a physics question
onto the terse labelled form `build_omni_corpus` actually generates:

```
"A 30 kg mass is pushed with 90 N. How fast does it accelerate?"
    -> force 90 N mass 30 kg find acceleration

"If something weighs 25 kg and speeds up at 4 metres per second squared, what force is that?"
    -> Given mass 25 kg and acceleration 4 m/s^2, compute the force.
```

The module's existing doctrine is kept, because it is what makes the thing safe:
this is presentation only, it never computes, and it never invents an operand. A
rule fires only when the text names its target **and** every quantity that target
needs, each anchored to its unit. `"What force do you feel in a lift?"` names a
target and no quantities, so it goes through untouched to ordinary conversation.
Units are consumed once and matched most-specific-first, so `m/s^2` cannot be
read as a velocity and a velocity cannot be re-harvested as a distance.

Test count on that module goes from 33 to 51, including five prose cases that
must not fire.

Scored on eighteen hand-written natural questions, same checkpoint, each asked
twice:

| | correct | 95% interval |
|---|---|---|
| as typed | 10/18 = 0.556 | 0.337 – 0.754 |
| normalised | **14/18 = 0.778** | 0.548 – 0.910 |

Sixteen of the eighteen were rewritten. **Four questions were fixed and none was
broken**, three of the four by the new science rules. On four discordant pairs
all in one direction the exact two-sided McNemar test gives **p = 0.125**, so
this is suggestive and not significant; the sample is eighteen questions written
by hand to probe known failure modes, not a random draw of what a user would ask.

The four that remain wrong are the useful part, because they are no longer
phrasing failures:

```
force            "...25 kg and speeds up at 4 metres per second squared..."
average x2       "Find the average of 61, 63, 72 and 61."
algebra_one_step "Solve for x: x + 29 = 34"
```

Three of those four were rewritten correctly into the trained form and the model
still got them wrong; the fourth, `Solve for x: x + 29 = 34`, was already in the
trained form and so was passed through unchanged. Either way none of the four is
a phrasing failure. `average` scores 0.033 and `algebra_one_step` 0.30 on the benchmark, so
what is left after normalisation is exactly the capability gap the benchmark
already reports. That is the outcome the module's docstring promises: it closes
the presentation gap and cannot close a capability one.

---

## 6. What is now reachable that was not

`train_mimomix_talk.py::build_config` was the chokepoint for both trainers.
Counted directly against the pre-v85 snapshot: of `MiMoMixConfig`'s 51 fields,
**15 were reachable by a flag and 36 were not** — including every architecture
switch and every router coefficient. v80 ran `router_score_function='softmax'`
because nothing could pass `sigmoid`.

After v85 the config has 64 fields, **52 reachable and 12 not**, so **29 of the
36 that were unreachable now have a flag**, with defaults that reproduce v80
exactly. An earlier version of this section said 33; that figure was inherited
rather than measured, and 29 is what the parsers actually expose.

The 12 still unreachable are `attention_sink`, `differential_output_norm`,
`final_layer_global`, `mla_global_only`, `mod_causal_predictor`,
`multimodal_input_dim`, `norm_topk_prob`, `tie_word_embeddings`, `use_moe`,
`use_multimodal`, `use_thinking_core` and `vocab_size`. Several are deliberate
(`vocab_size` comes from the tokenizer; `use_thinking_core` is set by `--arm`),
but `mod_causal_predictor` and `differential_output_norm` are the correctness
fixes from section 3 and cannot currently be turned off to A/B them, which is a
gap rather than a decision.

Techniques added behind default-off flags. Each carries someone else's
measurement at someone else's scale; **none has been measured here**:

| flag | technique | source evidence |
|---|---|---|
| `qk_norm` | RMSNorm on per-head q,k before RoPE | SmolLM-360M ablation: final loss 6.334 → 2.496 at LR 1e-3 (arXiv 2512.12167); OLMo 2 (arXiv 2501.00656) |
| `attention_output_gate` | head-wise sigmoid gate on the attention output | 1.7B/400B: PPL 7.499 → 7.404; first-token attention mass 46.7% → 4.8% (arXiv 2505.06708, NeurIPS 2025 oral) |
| `attention_sink_kinds="swa"` | sink on sliding-window layers only | MiMo-V2-Flash 32B, W=128: MMLU 54.9 → 58.3 with sink |
| `rotary_dim` | partial RoPE | MiMo-V2-Flash rotates 64 of 192; Qwen3-Next 64 of 256. Adoption, no isolated ablation |
| `global_layers` | explicit global-layer placement | Jet-Nemotron PostNAS (arXiv 2508.15884) |
| `differential_output_norm`, `differential_noise_ratio` | the reference formulation's sublayer norm; unbalanced signal:noise heads | GDA 0.9B: +0.88% at 3:1, −2.01% at 11:1 |
| `moe_balance_scope="sequence"` | per-sequence balance loss | DeepSeek-V3 §4.5.3 (arXiv 2412.19437) |
| `router_score_function="sigmoid"` | sigmoid affinity + top-k norm | DeepSeek-V3 Table 5: GSM8K 25.4 → 27.8 at 15.7B |
| `--mtp_loss_weight_final` | MTP weight 0.3 → 0.1 | DeepSeek-V3 MTP ablation; MiMo reports no gain from extra pretraining depths |
| `--reverse_digits` | LSB-first digits, inside the tokenizer | NanoGPT 10.6M: 100% on 3-digit addition at ~2.5k samples reversed, never unreversed (arXiv 2307.03381) |
| `--retry_rate` | error-correction rows | iGSM-med: 78% → 94% at rate 0.5 (arXiv 2408.16293) |
| `--balanced_operands` | stratify by digit length and carry count | arXiv 2307.03381 Fig. 3 |
| `--priming_fraction` | a few harder examples to extend the covered range | arXiv 2306.15400 |
| `--repeat_subset_fraction` | two-set repetition | 25M examples repeated ~24× learned 62 GCDs vs 27 with unlimited fresh data (Charton & Kempe 2024) |

---

## 7. Every new flag trains, and what each one weighs

A flag that is default-off and untested in a real optimisation is a trap: the
first time anyone turns it on is twenty hours into a run. So each of the sixteen
arms was given a short real optimisation at the v80 shape -- forward, backward,
gradient clip, optimiser step, router bias update -- and checked for finite loss,
finite gradient norm, and a falling language-model loss.

**All sixteen train. None produced a NaN or a non-finite gradient norm.** That is
the gate; forty steps on a synthetic task says nothing about quality.

Parameter costs are exact and worth having, because they are the part of an arm's
price that does not depend on the machine's mood:

| arm | total params | vs baseline | active per token |
|---|---|---|---|
| baseline (v80) | 15,274,549 | — | 3,917,861 |
| `qk_norm` | 15,274,933 | +384 | 3,918,117 |
| `differential` | 15,274,837 | +288 | 3,918,053 |
| `mod` | 15,276,088 | +1,539 | 3,919,400 |
| `mla` | 15,360,661 | +86,112 | 3,946,565 |
| `attention_output_gate` | 15,669,301 | **+394,752** | 4,181,029 |
| `top_k_4` | 15,274,549 | 0 | **4,360,229** (+11.3%) |
| `no_shared_expert` | 15,053,365 | −221,184 | 3,696,677 |
| `one_mtp_depth` | 14,683,437 | −591,112 | 3,917,861 |
| `no_mtp` | 14,092,325 | **−1,182,224** | 3,917,861 |

The MTP heads are **7.7% of the model** for a draft path whose measured
acceptance length is 2.5 of 3. `top_k_4` is the only arm that raises active
compute without raising parameter count, which is exactly the lever the
optimal-sparsity result points at.

**One verdict in the first sweep was wrong and is worth recording.** `no_mtp` was
reported as "trains but did not learn". It had not failed: with `n_mtp_layers=0`
the model's reported `loss` is the language-model term alone, while every other
arm sums the language-model, auxiliary and multi-token terms
(`mimomix_core.py:2351`). Comparing 9.178 → 9.232 against 11.913 → 11.445 was
comparing two different quantities. The gate now judges learning on `lm_loss`,
which every arm reports identically.

**A second wrong reading, caught the same way.** The first sweep ran the arms
back to back and reported `sink_swa_only` at 1.45x baseline and `rotary_dim_half`
at 1.42x. `rotary_dim_half` has a byte-identical parameter count to the baseline
(15,274,549 exactly) and `sink_swa_only` is 24 parameters *smaller*
(15,274,525), and both do strictly less work, so neither can be 45% slower. The
sweep had measured the machine slowing over twenty-five minutes. Timing on this box moves up to 5x between
identical runs -- the same benchmark read 2.045, then 11.037, then 2.136 s/step
with nothing changed.

Per-flag cost is therefore measured by interleaving each arm with a freshly built
baseline inside one process, over six rounds, and reporting the ratio of medians.
An arm whose round-to-round spread exceeds its own effect is reported as
unresolvable rather than given a number:

| arm | ratio to baseline | spread | reading | implied bound |
|---|---|---|---|---|
| `no_mtp` | **0.550** | 0.208 | cheaper | — |
| `top_k_4` | **1.102** | 0.095 | costs more | — |
| `differential` | **1.211** | 0.147 | costs more | — |
| `one_mtp_depth` | 0.845 | 0.303 | *not resolvable* | 0.54 – 1.15 |
| `mla` | 0.994 | 0.120 | *not resolvable* | 0.87 – 1.11 |
| `mod` | 1.051 | 0.159 | *not resolvable* | 0.89 – 1.21 |

Three of six resolve, and the two that matter most for planning both do. Removing
the multi-token heads halves the step. Doubling the active experts costs ten
percent, which makes `top_k_4` the cheapest capacity on the menu.

The three that do not resolve are still not nothing. An unresolvable arm bounds
its own effect: `mla` and `mod` cannot be costing more than about 10–20% either
way, which is the question anyone enabling them actually has. What cannot be said
is where inside that band they sit. `one_mtp_depth` is the one genuine loss --
its band spans from "halves the step" to "slightly worse than baseline" -- so the
ladder plans around `no_mtp`'s resolved 0.550 rather than guessing at it.

---

## 8. Solver-verified rejection sampling has something to harvest

The repository owns an exact solver that already verifies every generated corpus
row. That makes a rejection-sampling loop possible: sample k answers, keep the
ones the solver confirms, fine-tune on those. Nobody has run it, and finding out
the usual way costs a full training run.

There is a cheaper question that decides it. If the model can already produce a
correct answer *somewhere* in k samples far more often than it does greedily,
that gap is exactly what a rejection-sampling round would harvest. If it cannot,
there is nothing to collect and the run should not be spent.

Measured on v80 over the eight tasks that have room to move, k = 6 at temperature
0.8, top-k 40. Sampling had to be written for this: every decode path in the
repository is argmax.

| task | pass@1 (greedy) | pass@6 (samples only) | union | headroom |
|---|---|---|---|---|
| `acceleration` | 0.000 | 0.667 | 0.667 | **+0.667** |
| `arithmetic` | 0.667 | 1.000 | 1.000 | +0.333 |
| `average` | 0.000 | 0.333 | 0.333 | **+0.333** |
| `two_step` | 0.667 | 1.000 | 1.000 | +0.333 |
| `algebra_one_step` | 0.667 | 0.667 | 0.667 | 0.000 |
| `division` | 0.667 | 0.667 | 0.667 | 0.000 |
| `percent` | 1.000 | 1.000 | 1.000 | 0.000 |
| `power` | 0.333 | **0.000** | 0.333 | 0.000 |
| **overall** | **0.500** | **0.667** | 0.708 | **+0.208** |

**Corrected after review.** An earlier version of this table reported the *union*
column as pass@6, giving an overall 0.708. That folds the greedy decode into the
sampling result, and the greedy decode is the thing a harvest is meant to improve
on rather than part of it. True six-sample pass@6 is **0.667**. The `power` row
is the proof: its union cell reads 0.333 with a per-sample correct rate of
**0.000**, so not one of its eighteen samples was ever right and the whole cell
was the greedy hit. Headroom is union minus greedy either way, so **+0.208 is
unaffected** — but the level was inflated, and the receipt now carries both
columns.

**There is roughly twenty points of headroom, so the run is justified.** The
per-sample correct rate is 0.410, meaning a harvest would keep about two of every
five samples rather than needing hundreds to find one.

The most interesting cell is `average`. It has been the line's worst task since
v70, V73 wrote that *"no format tried so far has fixed it"*, and greedy decoding
still gets it wrong every time here. But one sample in nine is correct. **The
capability is latent and greedy decoding is not reaching it**, which is a
different diagnosis from "the model cannot do averages" and points at a different
fix. `acceleration` reads the same way: nothing greedily, two of three within six
samples.

Read this carefully, because it is easy to over-read:

* n = 3 per task, so a per-task cell carries a ±37 point interval and only the
  overall row (n = 24) is worth much. The per-task column says where to look
  next, not what will happen.
* The overall 0.500 is **not** comparable to the published 0.575 or to the 0.630
  baseline. It covers eight tasks selected before the n=630 baseline existed,
  using the published per-task scores, and they are **not** the eight weakest:
  `combination` (0.000), `arithmetic_series` (0.533), `molarity` (0.533) and
  `word_problem` (0.600) are all weaker and were left out, while `division`
  (0.733), `arithmetic` (0.700), `acceleration` (0.700) and `two_step` (0.633)
  rank 9th to 13th. Read the set as "eight tasks with room", not as a ranking.
* pass@k is an **upper bound** on what rejection sampling can harvest, not a
  prediction of what a fine-tune would reach. Sampling k times at inference is
  also not a decoding strategy here: the solver already knows the answer to
  anything it can parse, so this measures training-data availability, nothing
  about serving.
* Temperature 0.8 and top-k 40 are a first guess, not a tuned choice.

---

## 9. What the next runs should be, and what they cost

At the v80 shape (hidden 256, 4 layers, 8 heads / 2 KV, 48 experts, batch 16 ×
seq 128, fp32, 8 threads) a step costs roughly **2.0–2.5 s**, so 18,000 steps is
about ten to twelve hours of pure step time before probes and checkpoints. The
range is deliberate: this box moves up to 5× between identical runs, and any
single number quoted from it is a number about the machine's mood. Runs are
serial-only and go through `train_supervised.py`, because long runs have twice
segfaulted at a checkpoint boundary.

Three throughput results, all worth recording so nobody spends the afternoon
again:

* **The multi-token heads are 45% of step time** (0.550× measured paired, spread
  0.208), for 7.7% of the parameters and an acceptance length of 2.5 of 3.
  Dropping them entirely is the single largest lever on wall clock here, and both
  DeepSeek-V3 and MiMo report no pretraining gain from extra depths. Dropping to
  *one* depth is a different arm and did not resolve (0.845, spread 0.303), so
  it must not be quoted at 0.550. `top_k_4`,
  by contrast, costs only **1.102×** (spread 0.095) for 11.3% more active
  parameters, which is the cheapest capacity available.
* **bf16 autocast is 25–60× slower than fp32 here**, at 52–123 s/step against
  ~2.0. The `--amp` help text said it had no CPU effect; it has a catastrophic
  one. **The reason is not what an earlier version of this document said.** It
  claimed "this box is ARM64", implying a property of Arm hardware. The hardware
  is ARM64; the software is not. Verified four ways: the running interpreter's
  PE machine word is `0x8664` (AMD64, not `0xAA64`), `sysconfig.get_platform()`
  returns `win-amd64`, `sys.version` reports `MSC v.1943 64 bit (AMD64)`, and
  `torch.__config__.show()` advertises Intel oneAPI MKL "for Intel(R) 64
  architecture applications" with "CPU capability usage: AVX2". Meanwhile
  `platform.machine()` returns `ARM64`.

  **Every FLOP in an 11-hour run is being binary-translated by Windows-on-Arm
  Prism emulation.** What was measured as "bf16 on Arm" is an x86 bf16 fallback
  under emulation; the Snapdragon X Oryon core has native bf16 and i8mm that
  this stack never reaches. Native `win_arm64` PyTorch CPU wheels exist. Whether
  they are faster here is unmeasured and is now the single largest open question
  about this machine — with the caveat that no `win_arm64` build of torch 2.11.0
  exists, so going native is unavoidably also a version change and must be a
  labelled arm rather than a silent default. The current x64 interpreter is the
  environment v80 reproduces in and should not be replaced.
* **`torch.compile` does not work**: no MSVC `cl` on the machine. That is a
  missing install, not a platform limit — there is no `cl.exe`, `clang` or
  `gcc` on this box at all.

Proposed order, most information per CPU-hour first. Each arm states its
acceptance criterion in advance:

1. ~~**The paired rerun.**~~ **Done.** v80 scored on the current generators at
   n = 630, cap 96, per-task RNG, every task:

   > **397/630 = 0.630, 95% interval [0.592, 0.667], 0 truncated replies.**
   > Generator fingerprint `4077062251bc762c9716a730f3818ad2`, seed 65, receipt
   > at `output/v85_measurements/v80_paired_baseline_n630.json`.

   **That is the number every future run must be compared against**, and a
   comparison is only paired when the fingerprint matches.

   The published figure was 0.575, but it must not be differenced against 0.630.
   The receipt's own non-claims forbid it, and for a concrete reason: the two
   runs differ in *both* generator version and generation cap, and section 2
   shows `arithmetic_series` replies are 81–84 tokens, so at the published run's
   40-token cap that task scored 0.000 structurally whatever its generator did.
   `arithmetic_series` reading 0.533 and `kinetic_energy` 0.667 here, against
   0.000 for both in the published receipt, is generator **and** cap together.
   The cap-controlled measurement of the generator change alone is the era table
   in section 1, where the cap is held at 128 across all three eras and
   `arithmetic_series` moves 0.000 → 0.500. `combination` stays at 0.000 and
   `average` at 0.033.
2. **The corpus arm.** Retrain on the fixed generators with the probe cap at 112.
   *Accept:* `combination` above 0.10 — it is 0.000 in every era, so any movement
   is the model. `kinetic_energy` and `arithmetic_series` must be judged against
   line 1, not against the published 0.00.

   **On the MTP depth, and a correction.** An earlier version of this arm
   specified `--n_mtp_layers 1` and credited it with "45% of step time, turning a
   twelve-hour run into about seven". That is wrong, and wrong in exactly the way
   this document's own harness exists to prevent. 45% is `no_mtp`, i.e.
   `n_mtp_layers 0`. The `n_mtp_layers 1` arm measured **0.845 with spread 0.303
   and is recorded as not resolvable**, bounded only to 0.54–1.15. At the point
   estimate the run is about ten hours, not seven, and the honest statement is
   that this machine cannot say. Section 7 said so 100 lines earlier and this arm
   contradicted it.

   So: run one depth if you want the speculative draft path kept, run zero if
   wall clock is the binding constraint and you accept losing the draft, and do
   not plan a schedule around either number until `one_mtp_depth` resolves.
   Dropping the draft path costs benchmark wall clock but not accuracy, since
   greedy and speculative decoding are token-identical (0 mismatches in 30).
3. **`average` and `algebra_one_step`.** These are the two oldest failures, at
   0.033 and 0.30. *Accept:* `average` above 0.33, its v70 high-water mark,
   which V73 could not beat.

   `--average_binary_steps` replaces the one-shot running total with every
   operand shown, which is the fixed-dependency form the citation actually
   tested:

   ```
   off (v80):  sum: 76 then 140 then 202 then 272 then 352, total 352, divide by 5, total 70.4
   on  (v85):  76 + 64 = 140, 140 + 62 = 202, 202 + 70 = 272, 272 + 80 = 352, total 352, divide by 5, total 70.4
   ```

   Note this is **not** what V73 measured. V73 decomposed the accumulator into
   place values and lost ground (24.0% terse against 16.0% decomposed at
   sequence length 128); this shows the operands instead and leaves the
   place-value split off. The distinction is the reason to run the arm rather
   than assume V73 already settled it.

   `--algebra_word_sign` is a **partial** fix and should be reported as one.
   It resolves the sign in English, which removes the double negative that
   appears in 50.7% of v80's rows. But splitting the remainder still produces a
   negative units result in **45.2%** of rows and a degenerate leading step
   (`0 + 10 = 10`) in **27.5%**, measured over 4,000 generated rows. Every row
   passes the generator's own decomposition assert, so the arithmetic is sound;
   the sign reasoning the flag exists to remove has moved rather than gone.
4. **The warm gate.** `--thinking_residual_init 0.1` against 0.0, matched seed.
   *Accept:* any change at all in the cycle sweep above; it is currently
   identical at 1, 2, 3 and 6.
5. **Router.** `sigmoid` scoring plus sequence-scope balance. v80 starved 30 of
   144 expert slots and its biases reached 10.4–11.7 against scores averaging
   1/48. *Accept:* starved slots below 15 without a loss regression.
6. **Solver-verified rejection sampling.** Section 8 measured ~20 points of
   headroom overall and a 0.410 per-sample keep rate, so a harvest is worth
   collecting rather than a gamble. Sample k per prompt, keep what
   `nexus_solver` confirms, fine-tune on those. Target `average` and
   `acceleration` first: both score 0.000 greedily here and both are solvable
   within six samples, which is the clearest evidence in this document that a
   capability is present and greedy decoding is not reaching it.
   *Accept:* `average` above 0.33 — the same bar as arm 3, reached a different
   way, so the two arms are a genuine comparison rather than two attempts at one
   idea.
7. **`qk_norm` with a raised peak LR.** The cited ablation is a stability result,
   so the arm is only meaningful if the LR is raised with it.

---

## 10. What an adversarial review of this release found

Everything above was reviewed by independent agents told to refute rather than
confirm, with one whose only job was to check every number in this document
against its artifact. It was worth buying: **four claims in this document were
wrong**, and they are corrected in place above rather than quietly dropped.

| what was claimed | what is true |
|---|---|
| `--n_mtp_layers 1` saves 45% of step time | 45% is `n_mtp_layers 0`. The one-depth arm measured 0.845 and **did not resolve**. The ladder had contradicted section 7 one hundred lines earlier. |
| pass@6 is 0.708 | That is the union of six samples **and** the greedy decode. True six-sample pass@6 is **0.667**. `power` proves it: union 0.333 with a per-sample rate of 0.000. Headroom of +0.208 is unaffected. |
| "the ruler explains six of twenty-seven points" | A difference of two receipts this project's own non-claims forbid differencing. The paired −0.215 stands on its own; the residual is confounded with the cap change and v74's own re-measurement. |
| "33 previously unreachable config fields" | Measured: **29**. 15 of 51 fields were reachable before, 52 of 64 now, 12 still are not. |

Two smaller ones: `sink_swa_only` is 24 parameters *smaller* than baseline, not
byte-identical to it, and the benchmark's `NON_CLAIMS` asserted "n=30 per task by
default" when `--novel 100` over 21 tasks gives 4 or 5, where the interval is
±33 rather than ±17. Both are fixed.

Two defects in the code, not the document:

* **`--combination_in_envelope` could not be set in the benchmark process.** The
  twelve omni tasks are adapted from `build_omni_corpus.TASKS`, so a corpus built
  with that flag would have been scored against the un-narrowed generator — the
  model tested on problems its corpus never contained. The benchmark now has the
  same flag, and the fingerprint distinguishes the two shapes
  (`d89415c8…` against `0df09b51…`). The nine arithmetic tasks keep their own
  generators, so no corpus format flag reaches them, which is the property that
  makes the ruler stable and is worth stating explicitly.
* **`reverse_digits` breaks streamed decoding.** `decode` reverses digit runs
  over the whole joined string, which is an involution on a complete string but
  not on a growing prefix, so `supermix_chat_server`'s incremental
  `full[len(emitted):]` emits `total 444` for a reply that finishes `total 254`.
  It needs `--digit_tokens` and `--reverse_digits` together, both default off and
  no checkpoint has trained under either, so it is latent rather than live.

The review also refuted five findings, which matters as much: the `--priming_fraction`
overflow claim looked high-severity and turned out to keep 115 of 139 long rows,
defeating its own headline. Findings that could not be reproduced were dropped
rather than softened.

### A second round: three more numbers that measured the wrong thing

Six findings whose verifiers had run out of budget were re-run. **Three
reproduced and three were refuted.** All three that reproduced share a shape:
a number that reads as a measurement of one thing while measuring another.

**`mean_sink_mass` averaged in layers that have no sink.** A sink-less layer
reports a real-looking `0.0`, and the mean was taken over every attention
module rather than the sink-bearing ones. Enabling
`attention_sink_kinds="swa"`, which deliberately removes the sink from global
layers, therefore appeared to collapse sink usage:

| `attention_sink_kinds` | sink-bearing layers | old metric | honest mean |
|---|---|---|---|
| `all` (default) | 8 of 8 | 0.1517 | 0.1517 |
| `swa` | 5 of 8 | **0.0947** | **0.1515** |

Per-layer usage moved by 0.0002. The old metric showed a 37.5% fall that was
entirely the denominator. `mean_sink_mass` is now the mean over sink-bearing
layers, with `mean_sink_mass_all_layers` kept so a pre-v85 figure is still
reconstructable, and the default is byte-identical because there every layer
bears a sink.

**`mod_predictor_agreement` was computed only in training mode.** It says
whether the causal predictor is a usable stand-in for top-k selection, which is
exactly the question at decode time — and under `eval()` it reported the `0.0`
its buffer was initialised with. The auxiliary loss stays training-only; the
diagnostic does not need to, and `topk_mask` is already computed on every call,
so the fix costs one comparison. Under `eval()` it now reports real per-router
agreement (0.31–0.63 on a fresh model).

**`--balanced_operands` was a silent no-op under `--unique`, and the receipt
said otherwise.** Balancing stratifies repeated draws; the uniqueness path
rejects a draw on prompt novelty and never calls it. So
`--unique --balanced_operands` produced a corpus **byte-identical** to
`--unique` alone, while the report recorded `balanced_operands: true`. An A/B of
that flag would have measured an exact null by construction and been written up
as evidence the technique does nothing. The CLI now refuses the pair, and the
library path records what happened rather than what was asked for.

The three refutations are worth as much. `abstention_score` counting a
truncated wrong answer as an abstention is a documented decision at the line the
reviewer pointed at, and changes no published number. `generate_novel`'s
per-task count depending on the task-list length is round-robin semantics stated
in the function's first docstring line, not a determinism regression. And the
missing `ntk` guard on `mla_pe_dim` is real but cannot produce the degenerate
table claimed, because `__post_init__` already rejects the configs that would.

Four regression tests now pin the three real ones, so the release contract
covers four properties rather than three.

---

## Verification

**1,166 tests pass**: 280 across the core, decoding, architecture and release
contract suites; 670 across eval, corpus, answer-checking, the trainer and the
chat server; 216 across Nexus.

`test_v85_release_contract.py` pins the three properties this release rests on,
each a mistake that has actually happened here rather than a hypothetical one:
v80's own recorded command line must still rebuild v80's stored config (44 of 44
fields, 0 differing); the probe budget guard must name `arithmetic_series` as
blind at a 64-token cap and refuse to start under `--strict`; and the CI workflow
must run the suites published claims rest on, with a coverage ratchet that fails
if it drops.

An end-to-end trainer run confirms the guard fires **before the first step** and
that the receipt records what the probe could see. The receipt's
`hyperparameters` block went from 7 keys to 24, plus a `probe_token_budget`
block, so a reader can now tell from a receipt alone whether a 0.00 was the model
or the ruler.

Default behaviour is preserved. Against a snapshot taken before any edit, a
default-config model and a v80-shaped model both produce **bit-identical logits**
(delta 0.000e+00) with a loss delta of 9.5e-07, which is float32 epsilon. The
parameter hash differs only where the `quality_head` initialisation was
deliberately repaired, and that difference does not reach the output — which is
section 4's finding restated.

---

## What is not claimed

* **Nothing here was trained.** No accuracy number moved because of any change in
  v85. The two that moved — `kinetic_energy` 0.167 → 0.833 and
  `arithmetic_series` 0.000 → 0.500 — moved because the question changed, on
  weights that did not. The one genuine gain, the normaliser's 10/18 → 14/18, is
  a **presentation** result: the model answered a question it was trained on the
  form of, and computed nothing it could not compute before.
* **Every research technique above is a hypothesis**, carrying a citation
  measured at 82M–1.7B parameters on natural language. This model is 15M
  parameters on solver-generated arithmetic. None of those results is entitled to
  transfer, and none has been tested here. Sixteen of them are now known to
  *train*, which is not the same as known to help.
* **The measurements are small, and the intervals are in the receipts.** n = 63
  for the cap sweep, 24 for pass@k, 18 for natural phrasing, 12 per era cell,
  3 per task in several places. At n = 30 — the size of every per-task row in
  every published benchmark this project has shipped — the 95% half-width is
  ±17 points. The truncation counts, the cycle-sweep identity, the
  speculative-decoding parity and the parameter tables are exact and do not
  depend on sample size.
* **pass@k is an upper bound, not a forecast.** It says a rejection-sampling
  harvest would find material; it does not say what a fine-tune on that material
  would score. Its 0.500 baseline covers eight selected tasks — not the eight
  weakest; see section 8 — and is not comparable to the published 0.575 or the
  0.630 paired baseline over all twenty-one.
* **Per-flag step costs are ratios, and three of six did not resolve.** This box
  moves up to 5× between identical runs. Where the round-to-round spread exceeded
  the effect the harness reports "not resolvable" rather than a number. Those
  arms are bounded, not measured: `mla` and `mod` are somewhere within about 10-20%
  of baseline either way, and `one_mtp_depth` could be anywhere from halving the
  step to slightly worse than baseline.
* **The four bug fixes are unmeasured as improvements.** They make MLA, MoD and
  the thinking core behave as designed. Whether any of them helps accuracy is
  what a run decides.
* **The v74-versus-v80 regression is now explained, but the explanation is an
  inference.** The paired −0.215 is measured. The dilution mechanism — 54% of the
  per-task exposure at an unchanged step budget — is arithmetic over corpus
  composition and step counts, not a controlled experiment. It predicts the
  observed shape and nothing here rules out a second cause. The experiment that
  would settle it is a v80-shaped run at v74's per-task exposure, which nobody
  has run.
* **Both checkpoints being re-scored does not make the published receipts
  comparable retroactively.** The published columns in section 2c are shown for
  context only; they were measured on different problems at an unrecorded cap and
  must not be differenced against the paired ones.
