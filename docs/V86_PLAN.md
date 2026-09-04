# v86 — what to run next, and why in this order

v85 was a release about instruments. It ended with a paired baseline, a measured
regression, and a list of arms nobody had run. This is that list, narrowed by a
research pass and by four measurements taken since.

Everything below assumes the v85 facts: **v80 scores 0.630 [0.592, 0.667]** at
n=630 on the current generators (fingerprint `4077062251bc762c9716a730f3818ad2`),
runs are serial-only at roughly 2.0–2.7 s/step, and a comparison is only paired
when the fingerprint matches.

---

## 0. Two things to do before spending a single training hour

**A. Find out whether this machine has been emulating x86 all along.** The
hardware is ARM64. The Python and PyTorch are not: the interpreter's PE machine
word is `0x8664`, `sysconfig.get_platform()` is `win-amd64`, and torch links
Intel oneAPI MKL "for Intel 64 architecture applications" with AVX2, while
`platform.machine()` returns `ARM64`. Every FLOP in an 11-hour run is being
binary-translated by Windows-on-Arm Prism.

Native `win_arm64` CPU wheels exist for torch 2.7–2.14. Install an ARM64 Python
**side by side** — the current x64 interpreter is the environment v80 reproduces
in and must not be replaced — and run `--steps 20` under both, back to back in
one command, because this box varies 5× run to run. Adopt only at ≥1.5×.

There is no `win_arm64` build of torch 2.11.0, so going native is unavoidably
also a version change. It must be a labelled arm, never a silent default.

This is cheap, it is the largest unexplored variable on the machine, and if it
pays it pays every subsequent arm.

**B. Break `average` accuracy out by operand count.** ✅ **Done, and it changed
section 3.**

| values | n | accuracy | 95% CI | reached correct sum |
|---|---|---|---|---|
| 4 | 40 | 0.050 | [0.014, 0.165] | 0.050 |
| 5 | 40 | 0.000 | [0.000, 0.088] | 0.000 |
| 6 | 40 | 0.000 | [0.000, 0.088] | 0.025 |

Spread 0.050 against a widest interval of 0.151: **flat**. Chain length is not
the mechanism, because the chain never survives its first step. Decomposing
further:

```
first addition correct           1 / 120  = 0.008
every individual addition step   7 / 479  = 0.015
division correct, given the model's OWN stated total   83 / 120 = 0.692
```

The reply *shape* is right every time and the **division works**. What fails is
the accumulation, and it fails on the very first two-digit addition:

```
Find the average (mean) of these numbers: 48, 98, 18, 82
sum: 48 then 137 then 170 then 240, total 240, divide by 4, total 60.0
             ^^^ 48 + 98 = 146, not 137            ^^^ 240/4 = 60.0, correct
```

**The format never writes an addition as an equation.** It emits running totals
only, so each two-digit sum has to happen in latent space with no scratchpad —
while `arithmetic`, which scores 0.700, writes `500 - 300 = 200, 24 - 5 = 19,
total 219`. V68's rule ("a scratchpad helps only where it decomposes the
operation") applies exactly, and `average` is the task where it was never
applied.

This does not refute the section 3 format arm; it sharpens it and makes it
falsifiable. The research framed it as long chains accumulating error. The
measurement says no addition works at any length, so `AVERAGE_BINARY_STEPS` —
which writes `76 + 64 = 140, 140 + 62 = 202` — should help **equally at 4, 5 and
6 values**. If it helps only at longer chains, this diagnosis is wrong.

It also demotes the quotient–remainder division tail: division is already at
0.692 given the model's own total, so it is not where the loss is.

---

## 1. The corpus arm, paired

Retrain on the fixed generators with `--probe_max_new_tokens 112`.

*Accept:* `combination` above 0.10. It reads 0.000 in every generator era, so any
movement is the model rather than the ruler. `kinetic_energy` and
`arithmetic_series` must be judged against the 0.630 baseline, never against the
published 0.575.

**Consider `--batch_size 32`.** Measured, all four batch sizes interleaved in one
process:

| batch | s/step | per-sequence | sequences/hour |
|---|---|---|---|
| 8 | 1.532 | 1.142× | 18,793 (0.88×) |
| **16** (v80) | 2.685 | 1.000× | 21,453 (1.00×) |
| 32 | 4.470 | **0.832×** | **25,771 (1.20×)** |

20% more exposure per wall-clock hour at the same corpus. It is a different
optimisation trajectory, so it is an arm rather than a new default, and the
learning rate relationship at 0.12× compute-optimal is not obvious.

---

## 2. The dilution arm — and it must hold task count fixed

v85 measured a real −0.215 regression against v74 on identical problems, and the
leading explanation is exposure: each arithmetic task fell from 8.06% to 4.39% of
the corpus at an unchanged 18,000-step budget, so v80 saw each one 54% as often.

**But two mechanisms are confounded.** Adding twelve tasks to a 15M-parameter
model also invites capacity interference, which the curse-of-multilinguality work
(arXiv 2311.09205, >10,000 models to 45M parameters) shows degrades incumbent
groups *at constant exposure*. No sampling weight repairs that one. The free
discriminator — bimodal damage versus uniform — comes out **skewed, not bimodal**
(Sarle 0.285 against a 0.555 threshold), so at nine tasks it does not decide.

So the arm must **hold the task count fixed and change only exposure**. Train the
full 22-task corpus at v74's per-task exposure by upsampling the nine arithmetic
tasks 1.84×, against a control at v80's uniform sampling, same seed, same steps.
If the regression is exposure, it recovers. If it is capacity, it does not.

Two constraints worth knowing before designing the sampler:

* **Every task in v80 holds exactly 40,000 rows**, so cap-K proportional mixing,
  temperature-scaled mixing and mT5-style `p ∝ n^α` are **exact no-ops** on this
  corpus. They reduce to the uniform sampling already in place. The weight has to
  be set directly.
* Upsampling is safe here. At ~0.3 epochs a 1.84× upsample reaches ~0.55 epochs,
  an order of magnitude inside the four-epoch zone Muennighoff et al.
  (arXiv 2305.16264) measure as costing 0.5% validation loss.

*Accept:* the nine shared tasks recover materially toward v74's 0.874, with
`average` and `algebra_one_step` — which carry 58.6% of the total drop — moving
most.

---

## 3. `average`, the oldest failure

v74 scored 0.700 on it (paired). v80 scores 0.033. V70 reached 33.3%; V73 tried a
place-value-decomposed accumulator and got **worse** (24.0% terse against 16.0%
decomposed at sequence length 128).

**Section 0B changed what this arm is for.** The diagnosis is no longer "long
chains accumulate error" — it is that **individual additions succeed 1.5% of the
time and the first one succeeds 0.8%**, while the division succeeds 69% given the
model's own total. The format emits running totals and never writes an addition
as an equation, so every sum happens in latent space. That is a format defect at
every chain length, not a length effect.

**The arm is therefore `AVERAGE_BINARY_STEPS = True`** (implemented, never
trained), which writes both operands of every addition. The quotient–remainder
division tail is **demoted**: division is not where the loss is. Keep it only if
it is free in the token budget.

```
Find the average (mean) of these numbers: 61, 63, 72, 61
61 + 63 = 124, 124 + 72 = 196, 196 + 61 = 257, total 257,
4 x 64 = 256, 257 - 256 = 1, total 64.25
```

Measured with the repo's own token pattern: 62 / 70 / 78 tokens at k = 4/5/6,
maximum 78 — inside the 128 limit with 50 tokens of margin and inside the
96-token reply cap, so turn-aligned packing drops nothing. That margin is the
point, and it is why this beats a tens/units column accumulator (120 tokens at
k=6, 8 tokens of slack) and why it does not repeat V73's mistake of a
decomposition that stops fitting. Every product is two-digit × one-digit, the
shape `multiplication` scores 1.00 on. The reply ends `total <number>`.

*Accept:* `average` above 0.33, its v70 high-water mark, **and** individual
addition-step accuracy above 0.50, which is the quantity the change actually
targets. The sharp prediction that makes this falsifiable: the gain should be
**roughly equal at 4, 5 and 6 values**. If it appears only at longer chains, the
0B diagnosis is wrong and the length account was right after all.

**Protocol point, and the most likely way to waste the run:** pair this against a
v80-format control at identical seed, step count and corpus composition, with the
`average` rows as the only difference. Otherwise it measures dilution again.

The self-consistency check proposed earlier is **deprioritised**. It costs 9x
inference on all 21 tasks, and 0B shows `average` fails on a step the model gets
right 1.5% of the time — majority-voting over samples that are each almost
certainly wrong is not where to spend that.

---

## 4. Solver-verified rejection sampling

v85 measured pass@6 = 0.667 against pass@1 = 0.500, headroom +0.208, per-sample
keep rate 0.410. The repository owns an exact solver that verifies any answer.

Shape of one 11-hour run:

| when | what |
|---|---|
| 0:00–0:10 | batched-sampling timing probe, back to back |
| 0:10–1:40 | harvest: 7 tasks × 300 fresh prompts, k=8, T=0.8, top-k 40 |
| 1:40–1:50 | filter, dedupe, **kill switch** |
| 1:50–10:20 | two paired arms, serial |

Tasks: `average`, `acceleration`, `arithmetic`, `two_step`, `arithmetic_series`,
`molarity`, `word_problem`. Exclude `power` (per-sample rate 0.000) and
`combination` (0.000 everywhere).

Verify with `nexus_solver.solve_problem` on the **canonical** query, obtained by
calling the generators with `keep_canonical=True` — already in the repo, and it
takes verifiability from 41.9% on shipped v80 prompts to 100%.

**The kill switch matters.** Expect ~16,800 samples, ~6,900 answer-correct, and
2,000–4,000 rows after step-checking and deduping to at most two distinct
equation signatures per prompt. If fewer than 1,500 survive, or the keep rate is
under 0.20, stop and do not spend the remaining nine hours.

Both arms continue from the v80 checkpoint at reduced max LR — this is an anneal,
not a restart. **Arm A** original corpus only; **Arm B** harvested rows displacing
original rows of the *same task* at equal count, ~25% of those tasks' slots.
Provenance is the only variable, which is what makes it a test of harvesting
rather than a rediscovery of section 2: extra training concentrated on arithmetic
should help regardless of where the rows came from.

The solver verifies the **answer, not the working**, so step-check before
harvesting or the run trains on right-answer-wrong-method rows.

---

## 5. Lower priority, stated so they are not lost

* **The warm gate.** `--thinking_residual_init 0.1` against 0.0, matched seed.
  *Accept:* any change at all in the cycle sweep, which is currently identical at
  1, 2, 3 and 6.
* **Router.** `sigmoid` scoring plus sequence-scope balance. v80 starved 30 of
  144 expert slots. *Accept:* starved slots below 15 without a loss regression.
* **`qk_norm` with a raised peak LR**, since the cited result is a stability one
  and the arm is meaningless without raising the LR it stabilises.
* **`top_k_4`**, the cheapest capacity on the menu at 1.102× step time for 11.3%
  more active parameters.

---

## What is not claimed here

* **No arm below has been run.** Every acceptance criterion is stated in advance
  precisely because none of them has a result yet.
* **The research is other people's measurements at other people's scales.** The
  closest match found for the phase-transition account is 10.6M parameters on
  synthetic arithmetic (arXiv 2307.03381), which is genuinely close; most of the
  rest is 100M–7B on natural language and is not entitled to transfer.
* **Two research recommendations were checked and one was wrong.** "Batch size is
  free, so `--batch_size 8` doubles updates per hour at no cost" does not survive
  measurement: per-sequence cost rises 14% at batch 8, and because exposure is
  counted in sequences rather than updates, the smaller batch delivers 12% *fewer*
  sequences per hour. The emulation finding did survive, and independently.
* **The dilution mechanism remains an inference.** The −0.215 gap is measured; the
  exposure explanation is arithmetic over corpus composition and step counts, and
  the test that would separate it from capacity interference does not resolve at
  n = 9.
