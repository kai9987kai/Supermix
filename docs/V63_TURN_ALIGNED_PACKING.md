# Supermix v63 — 56% of the training signal had no prompt attached

V62 produced fluent sentences that ignored the question. Asked "hello, how are
you", it answered with an arithmetic sequence. The obvious diagnosis was
undertraining -- it had seen 0.71 epochs -- and the obvious remedy was more
steps. V62 had already run that experiment: 2,000 to 8,000 steps improved every
domain's loss and did not change the behaviour at all.

So the cause was not budget. It was the packing.

## The measurement

`mimomix_text.build_training_tensors` concatenates every turn into one token
stream and cuts it on a fixed `sequence_length` stride. The stride is blind to
turn boundaries, so a block can begin in the middle of a reply with its prompt
sitting in the previous block.

Measured on the v63 corpus at `sequence_length=128`:

| | stream packing (default) | turn-aligned |
| --- | --- | --- |
| blocks | 21,673 | 23,132 |
| supervised tokens | 1,900,569 | 1,085,739 |
| **supervised tokens with no prompt in their block** | **56.0%** | **0.0%** |
| blocks containing no turn start at all | 879 | 0 |

**Over half of every gradient step taught the model to continue a reply without
having seen the question.** That is not a subtle bias; it is direct training for
"emit the corpus's most likely reply regardless of input", which is precisely the
observed failure.

Padding costs 43% of the raw supervised token count, and it is still a net gain,
because the tokens that survive are the ones that can be learned from:

* stream: 1,900,569 supervised x 0.44 conditioned = **~836,000** usable
* turn-aligned: 1,085,739 supervised x 1.00 conditioned = **1,085,739** usable

## Why it stayed invisible for six versions

v57 through v60 trained this way and produced coherent replies. On a corpus built
from 192 sentences the modal reply is usually the right reply, so a model that
ignores its prompt still looks like it is answering. The flaw only becomes
visible once the corpus spans domains where the modal reply is wrong for most
questions -- which is exactly when v62 introduced it and exactly when the
symptom appeared.

That is worth stating plainly: **the templated corpus was hiding a training bug,
not just flattering a perplexity number.**

## The fix

`--turn_aligned_packing` gives every turn its own block, padded to
`sequence_length`. Turns that do not fit are **dropped rather than truncated**: a
truncated reply would teach the model to stop mid-sentence, and a truncated
prompt would reintroduce the conditioning gap the change exists to close.

Default is off, so every result from v57 to v62 reproduces unchanged.

```bash
python source/train_mimomix_generalisation.py --steps 12000 \
  --corpus_jsonl datasets/v63/v63_coherent.jsonl --turn_aligned_packing \
  --checkpoint_every_improvement --min_response_characters 1 --max_vocab 16384 \
  --run_name v63_aligned --output_dir output/v63_aligned
```

`test_v63_training_state.py` asserts the orphan rate is 0.0 under the new packer
**and that it is above zero under the old one**. Without the second assertion the
first could pass against a packer that never orphaned anything, and the fix would
be unfalsifiable.

## Three other defects fixed while getting here

**Optimizer state was not saved.** `--init_from` restored weights only, so AdamW's
moments and the LR schedule restarted cold. V62's continuation watched dev loss
climb from 0.8919 to 1.0036 and spent roughly 1,500 steps recovering ground it
already had. Checkpoints now carry `optimiser_state` and `scheduler_state`. When
`--steps` differs between runs the schedule is deliberately *not* restored --
a OneCycle curve of a different length would resume at the wrong point -- and
only the moments transfer. Restoring both wholesale corrupts the freshly built
schedule and raises `ZeroDivisionError` inside `get_lr`, which is how this was
found.

**Best weights lived only in memory.** They were written once, after the loop. A
crash or a kill at hour eleven left nothing at all, which is the position the
v62 continuation was in for its entire 17.5-hour run.
`--checkpoint_every_improvement` writes `<run_name>.partial.pt` on every dev
improvement. It paid for itself immediately: the packing diagnosis above came
from generating with a partial checkpoint at step 2,000 of a 12,000-step run,
instead of waiting ten hours to see the same failure.

**Runs of ten steps or fewer always crashed.** `OneCycleLR` divides by zero when
`pct_start * total_steps <= 1`, and it does so *in the constructor*, which calls
`step()` to set the initial learning rate. At the default `pct_start=0.1` every
run up to ten steps was affected -- exactly the range a smoke test uses, which is
why this trainer had no cheap smoke test. Short runs now get a flat learning
rate. A 2-step run takes 2 seconds.

**A bf16 crash in the MoE core**, found while preparing the GPU path.
`SparseMoEFeedForward.forward` accumulates expert outputs with `index_add_`,
which requires the source and accumulator to share a dtype exactly. Under
autocast the experts return bf16 into an fp32 buffer, so **every mixed-precision
run would have died on the first step**. Fixed in `mimomix_core.py` and mirrored
in `mechanism_causality.py`; verified not to change fp32 behaviour, since
`mechanism_causality.py` still reproduces its published v59 numbers exactly
(baseline 0.23159635, thinking core +8.841404e-08, routing +4.160336e-02).

## The result: the packing fix worked, and the corpus is the ceiling

The run finished at 12,000 steps with the best numbers of any version in this
repo -- dev loss **0.1230**, tier perplexities 1.0765 / 1.0712 / 1.2205 -- and
the generations are still mostly incoherent.

Register conditioning **did** improve, visibly and as predicted: a storytelling
prompt gets storytelling, an analogy prompt gets an analogy, an arithmetic prompt
gets a numeric format. That is what fixing the orphaned-token defect bought, and
it is real.

But content did not follow, and the reason is not budget, packing, or capacity.
It is the corpus. The single output that looked genuinely good --

> *"The moment hung in the air like a held breath. It was the kind of moment that
> divides life into 'before' and 'after.'"*

-- is not composition. It is recall:

| fragment | rows containing it, of 289,169 |
| --- | --- |
| `The moment hung in the air like a held breath` | **51,022 (17.6%)** |
| `It was the kind of moment that divides life into 'before' and 'after.'` | 10,261 (3.5%) |
| `theater backstage where small delays cascade` | 2,076 |

The corpus contains 800,231 sentence instances built from 179,880 distinct
sentences -- **4.4 repeats each** -- and carries its generator's scaffolding in
plain sight: `[strategic-set2]`, `(real-world-set2 genre variant)`,
`[practical-set9]`. The same fragment appears under two different domain labels.

**This is combinatorial template recombination, and a model that fits it
perfectly reproduces template recombination.** Perplexity 1.14 is the correct
score for memorising a corpus that is 17.6% one sentence; it is not a measure of
language ability, and this is the third time in this repo's history that a low
perplexity has meant memorisation rather than skill -- v57's 1.27 over 192
sentences, v62's 1.26 on its templated portion, and now this.

Notably, generation at step 3,500 (dev 0.2030) was **more** coherent than at
12,000 (dev 0.1230). Driving loss lower drove the model further into the
templates. Lower perplexity was actively worse for output quality, which is the
sharpest available statement that the metric and the goal have come apart.

## What would actually help

Not more steps, not more parameters, not a GPU, and not this corpus. The three
constraints measured across v61 to v63, in the order they bind:

1. **Corpus quality.** Every large corpus on disk is generated by recombining
   fragments. No training procedure extracts composition from data that has none.
2. **Corpus size.** 108.5M tokens total, compute-optimal for ~5.4M parameters --
   which the model already exceeds.
3. **Compute.** Only after the first two, and the one a GPU addresses.

The v63 line should be read as having found and fixed a genuine training defect
that was masked for six versions, and as having established that the remaining
gap is data, not engineering.

## What this does not prove

* **That turn-aligned packing fixes the model.** It fixes a measured defect in
  the training signal. Whether prompt-conditioned replies follow is the question
  the run is answering, and it may not: the model is 6.1M parameters on a corpus
  of ~23M tokens, and there is no guarantee that conditioning is the only thing
  standing between it and coherence.
* **That 56% is the number for every corpus.** It is `sequence_length=128` on the
  v63 blend. A corpus of shorter turns orphans less; longer turns orphan more.
* **That dropping oversized turns is free.** It removes rows entirely, and no
  measurement here says how many or whether they were disproportionately from one
  domain.
* **That the padding cost is optimal.** Packing several short turns per block
  while still respecting boundaries would waste less compute. That is not
  implemented.
