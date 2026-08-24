# Supermix v71 — a scratchpad can be too long, and lower loss proved it

> **CORRECTED BY v72.** The headline claim below is not supported. v72 ran v70's
> exact corpus at v71's sequence length (160 against 128) and reproduced **24.2
> of the 28.3 points** of regression on its own, leaving only 4.1 for the
> decomposed formats. At matched sequence length the decomposition *helped* the
> task it targeted: `average` goes 4.2% (v72) to 16.7% (v71).
>
> What survives is the loss/accuracy inversion. What does not is "a scratchpad
> can be too long" — that was a conclusion drawn across two changed variables.
> See [`V72_SEQUENCE_LENGTH.md`](V72_SEQUENCE_LENGTH.md).


V68 established that a scratchpad helps only where it decomposes the operation.
V70 still had two tasks with one-shot steps -- `average` listed running results
without showing the additions, `percent` guessed the multiply -- and they were
its two worst, at 33.3% and 58.3% against 91.7-100% for the decomposed tasks.

V71 decomposed them. It made the model worse at everything.

## The result

| task | v70 | v71 |
| --- | --- | --- |
| `arithmetic` | **91.7%** | 62.5% |
| `algebra_one_step` | **91.7%** | 33.3% |
| `word_problem` | **100%** | 58.3% |
| `average` | **33.3%** | 16.7% |
| `percent` | 58.3% | **62.5%** |
| **overall** | **75.0%** | **46.7%** |
| final dev loss | 0.0783 | **0.0613** |

**The lower loss is the finding.** V71 fits its corpus better than v70 fits
theirs and performs the task far worse. V64 showed loss preferring recitation
over composition, measured against a recall proxy; this shows loss preferring
*worse task performance* against an objective correctness metric, where a wrong
answer is simply wrong.

The mechanism is not mysterious. Decomposed text is formulaic -- "24 + 11 = 35,
running 135" is highly predictable token by token -- so per-token loss falls.
Task accuracy does not care about per-token loss; it requires **every step in the
chain to be right**.

## Chain length against reliability

Operations per answer, counted as `=` signs in the target:

| task | v70 | v71 |
| --- | --- | --- |
| `average` | 0 (results only) | **5.0** |
| `percent` | 1.0 | 2.2 |
| `addition` | 2.0 | 2.0 (unchanged) |

Treating a correct answer as every operation succeeding, `accuracy = p ** n`
implies a per-operation reliability:

| task | v70 | v71 |
| --- | --- | --- |
| `average` | 0.844 | **0.856** |
| `arithmetic` | **0.958** | 0.791 |

On `average`, **per-operation reliability improved and total accuracy halved**,
because the chain roughly doubled in length. The decomposition did exactly what
it was supposed to -- each individual step got more reliable -- and the answer
still got worse, because there were nearly twice as many steps to survive.

This is visible directly in the generations. The chains execute correctly and
slip once:

    700 - 200 = 500, 37 - 71 = -35, total 465     (37 - 71 is -34; truth 466)
    0 + 65 = 65; 65 + 36 = 101; 1 + 53 = 54; ...  (one step wrong, answer 49.4 not 49.2)

## The rule, refined

V68: *a scratchpad helps only where it decomposes the operation.*

V71 adds the other half: **and only up to the length where accumulated slips
outrun the decomposition's benefit.** A scratchpad has an optimum. Too short and
steps are guessed in one shot (v65, v66). Too long and the chain fails faster
than the extra structure helps. V70 sits nearer that optimum than v71.

## The part I cannot attribute

`arithmetic`, `algebra_one_step` and `word_problem` have **byte-identical
formats** in both corpora, and all three regressed sharply -- arithmetic's
implied per-operation reliability fell from 0.958 to 0.791. Chain length cannot
explain that, because their chains did not change.

Two candidates, neither isolated:

* **Capacity.** The longer `average` and `percent` targets consume more of a
  fixed 8.6M-parameter budget, leaving less for the unchanged tasks. This is the
  v62 and v69 pattern again.
* **Sequence length.** v71 ran at 160 against v70's 128, which changes how much
  falls inside the 64-token sliding-window attention. Supervised tokens per step
  actually *rose* (628 against 518), so this is not a padding-dilution effect.

A matched arm -- v71's formats at sequence length 128, or v70's formats at 160 --
would separate them. Neither was run.

## What to use

**v70 remains the model to use.** v71 is a measured negative result and its
checkpoint is kept only so the comparison can be reproduced.
