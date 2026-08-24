# Supermix v68 — 65% overall, and average fails because its scratchpad shows no working

V67 left `average` at 0% and identified two causes: a coverage bug in the
generator, and error accumulation over the running sum. V68 fixes the first.

## The result

Identical benchmark and scorer, 120 novel problems:

| task | v66 | v67 | v68 |
| --- | --- | --- | --- |
| `word_problem` | 0% *(absent)* | 41.7% | **100%** |
| `arithmetic` | 55% | 41.7% | **91.7%** |
| `algebra_one_step` | 0% *(absent)* | 66.7% | **79.2%** |
| `percent` | 65% | 62.5% | 54.2% |
| `average` | 0% | 0% | **0%** |
| **overall** | 24% | 42.5% | **65.0%** |
| median relative error | 0.179 | 0.004 | **0.000** |

Median relative error of zero means the typical answer is now exactly right.

## Two predictions, both wrong

Stated before the run:

1. *"`average` moves off 0%."* **Wrong.** It did not move at all.
2. *"`arithmetic` stays near 41.7%, so a large move would mean v67's regression
   was noise."* **Wrong.** It went to 91.7%.

The second failure also invalidates the framing. This was described as a
single-variable experiment; it was not. Lengthening the average rows changed how
the corpus packs and shifted the token distribution every task trains on, so the
arithmetic jump is real but **unattributed** -- corpus shift and run-to-run
variance are not separated, and no second arm was run to separate them.

## Why average still fails, precisely

The coverage fix worked. The model now handles six-number problems correctly:

    prompt: 72, 35, 5, 6, 52, 79
    reply:  sum: 72 then 107 then 155 then 202 then 261 then 327,
            total 327, divide by 6, total 54.166667
    truth:  sum  72,     107,    112,    118,    170,    249,
            total 249, divide by 6, total 41.5

It emits the right number of terms, divides by the right count, and computes
`327 / 6 = 54.166667` exactly to six decimals. Counting and division are correct.
The running sum is not: `107 + 5 = 155`.

**The cause is the scratchpad, not the task.** Compare the two formats:

    arithmetic:  900 + 700 = 1600, 87 + 2 = 89, total 1689   <- each sum decomposed
    average:     sum: 72 then 107 then 155 then ...          <- each sum one-shot

`arithmetic` scores 91.7% because every addition is broken into place values.
`average` writes the chain but never shows the working *within* a step, so each
addition is a single-shot guess -- exactly the failure mode v65 had before
scratchpads existed, preserved inside a format that looks like a scratchpad.

A model that does 3-digit addition at 91.7% is failing at 2-digit running sums
purely because of how the steps are written down.

## The rule this suggests

A scratchpad helps only where it decomposes the operation. Listing intermediate
*results* is not the same as showing intermediate *work*, and `average` is the
control that proves it: same model, same corpus, same training, one task whose
steps are decomposed and one whose steps are not, 91.7% against 0%.

The untested fix follows directly -- decompose each running-sum addition the way
`arithmetic` decomposes its operands -- at the cost of much longer sequences.

## What this does not prove

* **That the arithmetic gain came from the average fix.** See above; it is
  unattributed, and claiming otherwise would repeat the error this document
  opens with.
* **That decomposition would fix average.** It is the obvious hypothesis given
  the contrast, and it has not been run.
* **That 65% is problem solving.** Six task types, small operands, four fixed
  phrasings, every problem from a generator.
* **That `percent`'s slip to 54.2% is meaningful.** It is within the range these
  runs move by for reasons this session has repeatedly failed to attribute.
