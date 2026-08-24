# Supermix v72 — sequence length caused v71's collapse, not the scratchpad

V71 decomposed the inner operations of `average` and `percent` and got worse at
everything, including three tasks whose formats were byte-identical to v70's. I
attributed that to scratchpad length -- longer chains accumulating more slips --
and wrote it up as a refinement of v68's rule.

**That attribution was wrong.** v72 is v70's exact corpus at v71's sequence
length, changing one variable, and it reproduces most of the collapse on its own.

## The three-way comparison

| task | v70 (seq 128) | v72 (seq 160) | v71 (seq 160 + decomposed) |
| --- | --- | --- | --- |
| `algebra_one_step` | **91.7%** | 62.5% | 33.3% |
| `arithmetic` | **91.7%** | 58.3% | 62.5% |
| `average` | **33.3%** | 4.2% | 16.7% |
| `percent` | 58.3% | **62.5%** | **62.5%** |
| `word_problem` | **100%** | 66.7% | 58.3% |
| **overall** | **75.0%** | 50.8% | 46.7% |

Attribution of the 28.3-point fall from v70 to v71:

* **24.2 points (86%) from sequence length alone.** v72 changed nothing but 128
  to 160 and lost almost all of it.
* **4.1 points from the decomposed formats.**

## The correction to v71

Two claims in `V71_SCRATCHPAD_LENGTH.md` do not survive this control, and are
corrected here rather than quietly left standing:

**"A scratchpad can be too long" is not supported.** It was the headline, and the
evidence for it was the regression on untouched tasks -- which v72 shows was
sequence length. Chain length may still cost something; this experiment cannot
see an effect that size against a 24-point confound.

**Decomposition actually helped the task it targeted.** At matched sequence
length, `average` goes **4.2% (v72) to 16.7% (v71)** -- the decomposed format is
**four times better**, exactly as v68's rule predicted. v71 read as a failure
only because the sequence-length change buried it. The comparison I made at the
time (v71's 16.7% against v70's 33.3%) was across two variables, and I drew a
conclusion from it anyway.

What survives from v71 is the loss/accuracy inversion: v71's final dev loss
(0.0613) was better than v70's (0.0783) with far worse accuracy. v72 sharpens it
-- its dev loss is 0.1514 against v70's 0.0783, so here worse loss *also* means
worse accuracy. Loss and accuracy are not reliably related in either direction,
which is a stronger and less comfortable statement than "loss prefers
recitation".

## Why sequence length costs this much

Unknown, and worth saying plainly rather than guessing convincingly.

Ruled out: **padding dilution**. Turn-aligned packing at 160 gives *more*
supervised tokens per block than at 128, not fewer -- 39.3 against 32.4, and 628
against 518 per 16-block step. The model sees more signal per step, not less.

Still open, and untested:

* **Sliding-window attention.** `sliding_window=64` with `hybrid_ratio=3` means
  most layers see only 64 tokens locally. At 160 a larger share of each block
  sits outside that window and depends on the sparse global layers.
* **Padding position count.** Every block is padded to length, so a 160-token
  block carries ~35 more padding positions than a 128-token one, and attention
  is spent on them.

Both are testable -- a run at 160 with `sliding_window=96`, or one at 128 with
the v71 formats and shortened averages -- and neither was run.

## The practical rule

**Use the shortest sequence length that fits your turns.** With turn-aligned
packing the cost of headroom is not padding waste, which is what one would
expect and is measurably not the problem; it is something that removed a quarter
of this model's task accuracy for a 25% length increase. That is a large enough
effect to treat sequence length as a tuned hyperparameter rather than a safe
upper bound.

## What to use

**v70 remains the model to use, at 75.0%.** v72 and v71 are both measured
negative results, kept so the attribution can be reproduced.

## What this does not prove

* **That 128 is optimal.** Only 128 and 160 were tried. Shorter might be better;
  nothing here tests it.
* **That the decomposed formats are good.** They help `average` fourfold at
  matched length and cost 4.1 points overall. Whether they win at 128 is exactly
  the run that was not possible, because v71's averages do not fit at 128 --
  81% of six-value rows would be dropped by the packer.
* **That the mechanism is attention.** Two candidates are named above and
  neither was tested. The effect is measured; the cause is not.
