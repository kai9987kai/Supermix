# v75 — Surviving the crash, and v74 finished at 0.894

## The thing that keeps happening

Two long runs on this machine have died with SIGSEGV (exit 139):

| run | died at | of | hours lost |
|---|---|---|---|
| v64 | step 5,500 | 10,000 | 6.8 |
| v74 | step 11,500 | 18,000 | 9.2 |

Both died at an **eval/checkpoint boundary** — a multiple of `--eval_every`,
never between them. For v74 the log's final line and the checkpoint file are
6 ms apart (21:48:50.4357 and 21:48:50.4299), so the process reached the save,
completed it, and died there.

A segfault produces no traceback: the interpreter is gone before any handler
runs. So the cause is inferred, not proven. What is on the record:

* The host has **15.6 GB** of RAM. A run holds the packed corpus
  (471,347 × 128 int64 × 2 tensors ≈ **965 MB**), the model, AdamW moments, and
  at a save a serialisation buffer for all of it.
* The machine was **paging just before the crash**: steps 10,500→11,000 took
  2,866 s (5.7 s/step) against a 2.8 s/step run average, then recovered to
  2.9 s/step for the next 500.

That is consistent with memory pressure at the save. It is not a diagnosis, and
chasing one costs more hours than it saves. **So this work does not try to
prevent the crash. It makes the crash cheap.**

## Three changes

### 1. The recovery checkpoint could be destroyed by the crash it recovers from

`save_talk_checkpoint` called `torch.save(payload, path)`, which **truncates the
destination and writes in place**. The recovery checkpoint is rewritten on every
dev improvement, and the crash happens at exactly that moment. A segfault
partway through the write would have left a corrupt file where the only copy of
9.2 hours of training had been.

It now writes to `<name>.tmp` and `os.replace`s it into position — atomic on
both POSIX and Windows — so the previous checkpoint survives intact until the
new one is completely written. Three tests cover it, including one that writes
partial bytes and then raises, asserting the older checkpoint still loads and
still reports its original step.

This was the highest-severity item and it was latent through every run to date.

### 2. `--start_step`: rejoining the curve instead of restarting it

A crashed leg could already be continued via `--init_from`, but only onto a
*fresh* OneCycle curve — the learning rate would warm up again from the start
while the model was 64% trained.

The blocker was that `source_steps` records **steps completed**, and the guard
compared it against this run's **total** `--steps`. An 11,500-step checkpoint
therefore read as a differently-shaped run than `--steps 18000`, and the
schedule was silently discarded. Those are two different numbers and the
checkpoint now records both (`steps` and `total_steps`).

With `--start_step`, a resumed leg keeps the same schedule and runs only the
tail. For v74 the arithmetic is exact:

```
scheduler last_epoch : 11500   of total_steps 18000
resume runs 6,500 steps  ->  scheduler ends at exactly 18000
learning rate at resume  :  0.00104174, annealing to ~0
```

**A smoke test found a real bug here.** Resuming with `--start_step 10` from a
checkpoint saved at step 20 restored the scheduler at 20, then ran 10 more
steps and hit `Tried to step 21 times. The specified number of total steps is
20`. The restored schedule resumes at the *checkpoint's* step, so `--start_step`
must equal it. That is now checked at startup, where it costs one line, instead
of surfacing hours in.

### 3. A supervisor that restarts, and knows when not to

`source/train_supervised.py` wraps the trainer: on a non-zero exit it reads the
step the recovery checkpoint reached and relaunches from there. A crash now
costs **one eval interval (~25 min) instead of the whole run**.

The two refusals matter more than the restart:

* **It never restarts on exit 0.** A clean exit means the run finished;
  relaunching would train a leg nobody asked for.
* **It never restarts without progress.** If a leg reaches the same step or
  earlier than the one before it, the failure is deterministic rather than a
  random fault, and looping would burn the same hours reproducing it. It stops
  and says so in `supervised_run.json`.

It also refuses to start without `--checkpoint_every_improvement`, since there
would be no step to resume from.

23 tests cover it, including the restart decision driven against a fake trainer.

## The settled result: v74 finished

n=500 novel problems, 0 unparsed, on the completed step-18,000 checkpoint.

**Overall: v74 0.894 (447/500) against v73 0.756 (378/500), z=5.74** — and v74
carries *ten* task types where v73 was measured on five.

| task | v74 final | v74 @11.5k | v73 | verdict |
|---|---|---|---|---|
| division | **1.00** | 0.85 | — | new |
| multiplication | **1.00** | 0.96 | — | new |
| sequence | **0.98** | 0.93 | — | new |
| two_step | **0.98** | 0.67 | — | new |
| word_problem | 0.96 | 0.73 | 0.99 | tie (z=−1.12) |
| algebra_one_step | 0.89 | 0.61 | 0.94 | tie (z=−1.06) |
| arithmetic | 0.89 | 0.55 | 0.99 | **v73 better (z=−2.81)** |
| percent | 0.75 | 0.73 | 0.70 | tie (z=0.67) |
| average | **0.59** | 0.34 | 0.16 | **v74 better (z=5.54)** |

### Read the headline carefully

The 0.894 is real but it is flattered by the four new tasks, which v74 finds
easy (two of them perfect). The like-for-like comparison is the five shared
tasks:

**shared aggregate: v74 0.818 (229/280) vs v73 0.756 (378/500), z=1.99.**

That clears 1.96 by a hair. So on the tasks both models were asked to do, v74
is better — *marginally*, and driven almost entirely by `average`
(0.16 → 0.59). It should not be quoted as a decisive win on the old benchmark.

**One genuine regression survived the anneal:** `arithmetic` 0.99 → 0.89
(z=−2.81). Adding six task types to the corpus cost accuracy on the simplest
one. The other two shared tasks that looked broken at step 11,500 —
`word_problem` 0.73 and `algebra_one_step` 0.61 — recovered to 0.96 and 0.89
and are now statistically indistinguishable from v73.

That recovery is the substantive finding about the mid-run reading below: at
64% trained, three shared tasks looked badly damaged; two of the three were
simply unfinished. **Judging a run at 64% would have produced the wrong
conclusion**, which is precisely why the crash needed to be recoverable rather
than merely survivable.

## What v74 looked like at step 11,500 (superseded, kept for the record)

Measured on the recovered checkpoint, n=500 novel problems, 0 unparsed:

| task | v74 @ 11,500 (n=50) | v73 final (n=100) |
|---|---|---|
| multiplication | **0.96** | — new |
| sequence | **0.93** | — new |
| division | **0.85** | — new |
| percent | 0.73 | 0.70 |
| word_problem | 0.73 | 0.99 |
| two_step | **0.67** | — new |
| algebra_one_step | 0.61 | 0.94 |
| arithmetic | 0.55 | 0.99 |
| average | **0.34** | 0.16 |
| **overall** | **0.708** | **0.756** |

**The four new task types all landed well above zero** — that was the bar, and
they cleared it comfortably. `average`, the task v73 was worst at by a wide
margin, more than doubled (0.16 → 0.34).

**Three shared tasks regressed sharply**: arithmetic 0.99 → 0.55,
algebra_one_step 0.94 → 0.61, word_problem 0.99 → 0.73. Averaged over the five
shared tasks, v74 is at 0.592 against v73's 0.756.

**This reading was superseded by the finished run above.** Two of the three
regressions were unfinished training, not damage. Kept because it is the
evidence that mid-run task accuracy is not a safe basis for judging a run.

Two things about that comparison, both of which cut against reading it as a
verdict:

1. **v74 is 64% trained and v73 is finished.** Dev loss was still falling
   (0.0965 at the crash), and the accuracy probe was still climbing —
   0.04 → 0.22 → 0.61 at steps 3,000 / 6,000 / 9,000, against 0.708 on the
   full n=500 at 11,500. The remaining 6,500 steps are the annealing part of
   the curve, which is where accuracy usually consolidates.
2. **The per-task sample sizes differ.** v74 spreads 500 problems over ten
   tasks (n=50 each, ±14 points at p≈0.5); v73 spread them over five
   (n=100 each, ±10 points). Individual task rows are noisier than they look.

So this was not yet a v74-versus-v73 result. It was a mid-run reading that said
the new capabilities landed and the old ones had not yet come back. The
remaining 6,500 steps decided it, and they brought two of the three back.

## The resumed leg

It completed in **one leg, exit 0, zero restarts** — 17,580 s for 6,500 steps.
The supervisor's journal records the whole thing in
`output/v74_broad/supervised_run.json`.

```
restored     optimiser=True scheduler=True
step 18000/18000  train 0.0511  dev 0.0651  ppl 1.07  acc 0.89  16953s
selected     step 18000 on accuracy (probe 0.89, dev loss 0.0651)
```

The mid-curve resume behaved exactly as the arithmetic predicted: the schedule
restored at 11,500, ran 6,500 steps, and landed on 18,000.

The accuracy probe climbed the whole way:

| step | probe (n=100) |
|---|---|
| 3,000 | 0.04 |
| 6,000 | 0.22 |
| 9,000 | 0.61 |
| 12,000 | 0.70 |
| 15,000 | 0.82 |
| 18,000 | **0.89** |

Generalisation gap is small — tier1 (seen response) 0.0884 against tier3
(unseen sentence) 0.0969, a perplexity ratio of **1.008x**, total cost
+0.0085 nats. All five report checks pass.

### A reporting bug this run exposed

The console printed `selected step 18000 on dev (dev loss 0.0651)` while the
JSON correctly recorded `selected_on: accuracy, best_probe_accuracy: 0.89`.
The line hardcoded "on dev" and its dev loss regardless of `--select_on`,
which states the opposite of what v64 established — dev loss is the thing not
to trust, and under `--select_on accuracy` it is consulted only as a
tie-break. A run chosen on a 0.89 probe that reports only its loss is how a
summary quietly becomes wrong. `describe_selection` now names the actual
criterion and the value it read; seven tests cover it.

### The model recites on dialogue

`best_probe_verbatim_rate` is **1.0**: every one of the five dialogue probe
replies is found verbatim in the training corpus. They read well —

> "Check the traceback first, then we can isolate the failing function."

— and they are reproduction, not composition. This is the predicted
consequence of the corpus measured before v74 started (19.8% of the dialogue
portion is one repeated fragment; its sentences repeat 6.6x on average), and
it is why selection ran on `accuracy` rather than `novelty`. **v74 is a
problem-solving model. Its conversational ability is recall.**

## What is not claimed

* **No cause for the segfault.** Memory pressure is consistent with the
  evidence and unproven. If a supervised run crashes again, the journal will
  say at which step, which is the first real data on whether it is
  step-dependent or time-dependent.
* **No decisive win on the shared benchmark.** Overall 0.894 vs 0.756 is
  significant, but it is inflated by four new tasks v74 finds easy. Like for
  like, the five shared tasks are 0.818 vs 0.756 at z=1.99 -- over the line by
  a hair, and `arithmetic` is genuinely worse (0.99 -> 0.89, z=-2.81).
* **No knowledge or creativity gain.** The corpus cannot deliver it — measured
  earlier: the dialogue portion is 19.8% one repeated fragment and its
  sentences repeat 6.6× on average. Ten problem types requiring composition is
  the reachable part of that request, and it is what was built.
