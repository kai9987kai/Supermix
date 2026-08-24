# Supermix v73 — the training loop can now see accuracy, and the model tied

Two things shipped here: a change to how training is measured, and a run that
applied v71's and v72's findings together.

The process change is the durable one. The model is a tie.

## The process: accuracy during training, not after

Every run in this line cost twelve to seventeen hours and reported whether it
worked only afterwards, because the loop tracked dev loss. That metric has now
failed in both directions:

* v71 finished with a **better** dev loss than v70 and **28 points less** accuracy.
* v72 finished with a **worse** dev loss and worse accuracy.

`--accuracy_every N` measures exact-match accuracy on freshly generated problems
mid-run, and `--select_on accuracy` chooses the checkpoint that answers
correctly rather than the one that fits the corpus. Problems are generated fresh
and never drawn from training, so a memorised answer scores zero.

On this run it tracked the model honestly:

| step | probe accuracy |
| --- | --- |
| 8,000 | 0.15 |
| 12,000 | 0.50 |
| 14,000 | 0.60 |
| 16,000 | 0.55 |

Final measured accuracy was 0.758. The probe was directionally right and
numerically low, which matters -- see the flaws below.

## The run: v71's formats at v72's sequence length

v71 showed decomposition is worth **4x** on `average` at matched length. v72
showed sequence length 160 costs **24 points** against 128. v73 takes both:
decomposed working for 4-5 value averages, terse format for 6-value ones so they
still fit 128, zero rows dropped by the packer.

| task | v70 | v73 | delta | ±95% CI |
| --- | --- | --- | --- | --- |
| `arithmetic` | 91.7% | **100%** | +8.3 | ±0.0 |
| `word_problem` | 100% | **100%** | 0.0 | ±0.0 |
| `algebra_one_step` | 91.7% | 91.7% | 0.0 | ±11.0 |
| `percent` | 58.3% | **62.5%** | +4.2 | ±19.4 |
| `average` | **33.3%** | 25.0% | −8.3 | ±17.3 |
| **overall** | 75.0% | **75.8%** | **+0.8** | **±7.7** |

**This is a tie.** The overall margin is 91 correct against 90, out of 120 -- one
problem, against a 95% confidence interval of ±7.7 points. Every per-task
difference except `arithmetic` sits inside its own interval too.

`arithmetic` reaching 24/24 is the only result that looks real, and it is 2
problems on n=24. It is suggestive, not established.

**`average` did not improve**, despite receiving the decomposition that was worth
4x at seq 160. 25.0% against 33.3% is 6 correct against 8. The honest reading is
that this run cannot tell whether decomposition helps `average` at seq 128,
because the sample is far too small to resolve an effect of that size.

## What the numbers cannot support

The bar set before the run was "beat 75.0%, with `average` above 33.3%". v73 hit
neither in any meaningful sense: it matched 75.0% and `average` fell. Reporting
+0.8 points as an improvement would be exactly the over-claiming this line has
corrected twice already, and the sample size says so explicitly.


## Settled at n=500

The 120-problem comparison above could not separate the two models. A 500-problem
evaluation of each (100 per task, no training, generation only) can:

| task | v70 | v73 | diff | z | significant |
| --- | --- | --- | --- | --- | --- |
| `arithmetic` | 91.0% | **99.0%** | +8.0 | **2.64** | **yes** |
| `word_problem` | 94.0% | 99.0% | +5.0 | 1.94 | no (just under) |
| `algebra_one_step` | 88.0% | 94.0% | +6.0 | 1.49 | no |
| `percent` | 69.0% | 70.0% | +1.0 | 0.15 | no |
| `average` | **24.0%** | 16.0% | −8.0 | −1.42 | no |
| **overall** | 73.2% | 75.6% | **+2.4** | **0.87** | **no** |

**One real result: v73 is better at arithmetic**, 99/100 against 91/100, z=2.64.
`word_problem` sits just under the threshold at 1.94 and points the same way.

**The overall difference is still not significant** (z=0.87), because `average`
regressed by 8 points and cancelled the gains on the other four tasks. So the
earlier "tie" verdict survives at the aggregate level, and the tie is now
explained rather than merely observed: v73 wins on four tasks and loses badly on
the one it was designed to fix.

**Decomposition did not help `average` at sequence length 128.** v71 measured it
as worth 4x at seq 160 (4.2% to 16.7%); here v70's terse format scores 24.0% and
v73's mixed format 16.0%. The direction is reversed, though not significantly.
The cleanest reading is that the seq-160 result was measured against a floor so
low that almost anything looked like an improvement, and that at 128 the terse
format is at least as good. v73's `average` is also a *mixture* -- decomposed for
4-5 values, terse for 6 -- so it is not a clean test of either format.

`average` remains the worst task in both models by a factor of three, and no
format tried so far has fixed it.

## Two flaws in the process change, found by using it

**The probe is too noisy to select on.** At step 8,000 it reported 0.15; a
60-problem evaluation of the step-9,000 partial measured 0.467. Twenty problems
carries roughly ±10 points of sampling error, and `--select_on accuracy` chooses
between checkpoints on exactly that number. It is adequate for *aborting* a run
and inadequate for *selecting* within one.

**It weakened crash safety.** Under `--select_on accuracy` the partial checkpoint
is written only when the accuracy score improves, and accuracy is probed every
2,000 steps -- so this run went 6.3 hours without writing a checkpoint. The v63
protection assumed dev-loss selection, where improvements are frequent. The fix
is to write the partial on dev improvement regardless of which criterion drives
selection; that has not been done.

Both are worth fixing before the next run, and neither is fixed here.

## What to use

`v73_decomposed_short` and `v70_moe` are equivalent within measurement error.
v73 is marginally preferable on the strength of `arithmetic` at 24/24 and one
fewer moving part in its corpus, but anyone choosing on the overall number is
choosing on noise.

## What this does not prove

* **That the decomposition helped.** The two tasks it targeted moved +4.2 and
  −8.3 points, both inside their intervals.
* **That sequence 128 is confirmed better.** v72 established that separately at
  n=120 with a 24-point gap; this run neither adds to nor weakens it.
* **That 120 problems is enough.** It is not, for differences this size. A real
  comparison of v70 against v73 needs several hundred problems per task, which
  is hours of CPU generation and was not run.
