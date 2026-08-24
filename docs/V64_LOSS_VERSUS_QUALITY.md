# Supermix v64 — held-out loss is anti-correlated with generation quality

V64 set out to make the model smarter: a larger vocabulary (32,774 against
16,390), a corpus led by real prose rather than generated dialogue, and explicit
word-to-definition supervision. It also became the version that measured
something more useful than any of that.

**Training it further made it worse at the thing it is for, while the loss
improved.**

## The measurement

Both checkpoints come from the same run, the same corpus and the same split. The
only difference is 4,500 more steps of training. Ten prompts, scored by
`recall_index` against the training corpus:

| | step 5,500 | step ~10,000 |
| --- | --- | --- |
| dev loss | 1.0762 | **0.9910** |
| mean verbatim rate | **0.14** | 0.76 |
| replies judged novel | **5 of 8** | 1 of 6 |
| replies too short to judge (degenerate) | 2 of 10 | 4 of 10 |

Lower held-out loss bought **5.4x more recitation** and **twice the degeneracy**.
The earlier, worse-scoring checkpoint is the better generator, and it is not
close.

This is the second time the pattern has appeared. V63 showed the same shape
qualitatively -- generation at step 3,500 (dev 0.2030) read better than at step
12,000 (dev 0.1230) -- but had no way to quantify it. The recall meter turns that
impression into a number, and the number replicates.

## Why it happens

A language model minimising cross-entropy on a corpus with repeated spans is
rewarded for reproducing those spans. As training continues, the cheapest
remaining loss reduction is to commit harder to the corpus's most frequent
continuations. That is exactly what recitation is.

Perplexity cannot distinguish the two outcomes, because verbatim reproduction of
training text is the *lowest-loss possible behaviour*. The metric is not merely
insensitive to the failure; it actively prefers it.

## What v64 did achieve

The corpus is genuinely better on every axis it targeted:

| | v63 | v64 |
| --- | --- | --- |
| word types | 11,290 (capped 16,390) | **41,871 (capped 32,774)** |
| distinct sentences | 179,880 | **187,351**, from 33% fewer rows |
| mean sentence repetition | 4.4x | **2.1x** |
| most common fragment | 17.6% of rows | **4.4%** |

And the generalisation ladder is finally measuring something hard. On real prose
the unseen-sentence cost is **+1.3801 nats** (tier3/tier1 perplexity ratio
**2.750x**), against +0.1305 for v63 and +0.0955 for v60.

That refines v60's conclusion rather than contradicting it. V60 found v58's
withheld-sentence penalty nearly vanished (+0.0043 nats) on the v29 corpus and
called the original effect an artifact of corpus poverty. Both are true, and the
mechanism is the same: on a templated corpus an "unseen" sentence is a
recombination of seen material, so withholding it costs little; on real prose it
is genuinely novel, and withholding it costs a great deal. **The gap measures how
compositional the corpus is, not how capable the model is.**

## What to serve

**`output/v64_meaning/v64_meaning.partial.pt` -- the step-5,500 checkpoint --
is the better v64 model**, despite the worse number, and it exists only because
`--checkpoint_every_improvement` was added in v63. The run it came from
segfaulted at that step; without mid-run checkpointing there would be no v64 at
all.

For general chat `v60_control_2000` is still preferable: narrower, but reliably
fluent inside its register.

## The rule this establishes

Do not select checkpoints on held-out loss in this regime. Select on a metric
that can see recitation -- the recall meter is one -- or the selection procedure
will reliably choose the most memorised checkpoint available. Every promotion
gate in this repo that reads dev loss alone inherits this flaw, including the one
in `train_mimomix_generalisation.py` that chose step 10,000 over step 5,500.

## What this does not prove

* **That loss and quality are always anti-correlated.** This is one architecture
  on corpora with heavy internal repetition. Early in training loss and quality
  clearly improve together; the inversion appears later, and nothing here locates
  the crossover point.
* **That the step-5,500 checkpoint is good.** It is better. Five of eight scored
  replies were novel, and four of ten prompts still produced degenerate output.
* **That ten prompts settle it.** The effect is large and directionally
  consistent, but the sample is small and hand-chosen, and no confidence interval
  has been computed.
* **That the recall meter measures quality.** It measures provenance. A novel
  reply can be nonsense, and several were -- "clever (adjective) means or." is
  novel and worthless.
* **That the larger vocabulary helped.** It was changed alongside the corpus, so
  nothing here separates the two.
