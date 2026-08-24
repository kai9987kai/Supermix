# Supermix v66 — showing the working took exact arithmetic from 0% to 55%

V65 diagnosed why arithmetic was impossible -- `498` was a single token -- and
fixing that got the model to answer in the right format at roughly the right
magnitude while getting almost every digit wrong. Exact accuracy on novel
addition and subtraction was **0%**.

Right magnitude with wrong digits is the signature of a model guessing an answer
in one step rather than computing it. V66 tests the standard remedy: put the
intermediate work in the target so the model learns a procedure to execute.

## The result

Identical benchmark, identical novel problems, identical scorer:

| task | v65 | v66 |
| --- | --- | --- |
| arithmetic (3-digit + and -) | 0% | **55%** |
| percent | 15% | **65%** |
| average | 0% | 0% |
| **combined, tasks both were trained on** | **5%** (3/60) | **40%** (24/60) |
| median relative error | 0.250 | **0.179** |
| answers within 10% of truth | 23% | **44%** |

`algebra_one_step` and `word_problem` both score 0% and are excluded from the
comparison above: **v66's corpus does not contain them**. Including them would
give a headline of 24%, which would be an honest number for the wrong question --
it would measure the corpus's coverage rather than the effect of the scratchpad.

## The scratchpad

Two-step place-value decomposition, chosen because it is *always* valid:

    524 - 305  ->  500 - 300 = 200, 24 - 5 = 19, total 219
    504 - 309  ->  500 - 300 = 200, 4 - 9 = -5, total 195

The second case is the reason. A column method needs borrows and can produce
negative digits requiring carry handling; splitting into hundreds and remainder
cannot fail, because the two partial results simply add. A generator that were
correct only in easy cases would teach the model to be wrong precisely where
arithmetic is hard.

The corpus was verified rather than trusted: 20,000 rows independently
recomputed, **0 incorrect final answers**, and all 10,121 addition/subtraction
rows checked for `high + low == total`. The generator also asserts the invariant
on every row it emits.

## The failures are procedural, not random

This is the part exact accuracy hides. The model executes the procedure and
slips inside it:

| prompt | reply | truth |
| --- | --- | --- |
| 987 + 702 | `900 + 700 = 1600, 87 + 2 = 90, total 1690` | 1689 |
| What is 12% of 1049? | `1 percent of 1049 = 10.49, times 10, total 104.9` | 125.88 |
| What is 20% of 185? | `1 percent of 185 = 1.85, times 20, total 37.0` | 37.0 |

In the first, the hundreds column is exactly right and `87 + 2` is off by one. In
the second, one percent of 1049 is computed **exactly** and then multiplied by the
wrong number -- it misread the percentage, not the arithmetic. These are
recognisable slips in a method being followed, which is a different failure from
v65's, where there was no method to slip in.

Structure appeared early: at **step 500 of 12,000** the model already emitted the
full decomposition, accumulated running sums, and divided by the right count.
Every number was wrong and every step was in the right place.

## Why average stayed at 0%

It is the longest chain -- a running sum over four to six values, then a division
-- so it has the most places to go wrong, and one wrong partial sum poisons the
total. It also produces non-terminating decimals that the corpus rounds to four
places while the benchmark compares exactly. Both are plausible; neither has been
isolated, and the honest position is that average failed and the cause is untested.

## What this does not prove

* **That the model can do arithmetic.** 55% on 3-digit addition and subtraction
  is not reliability. It is a large change from 0%, not a working calculator.
* **That the scratchpad is the only cause.** v66 also changed corpus composition
  (four task types against twelve) and sequence length (96 against 64). The
  comparison is between two models, not two formats.
* **That showing working is reasoning.** The model imitates a procedure. Nothing
  here shows the steps *constrain* the answer -- and the 1690 case, where correct
  steps produce a wrong total, is evidence they sometimes do not.
* **That it generalises past the generator.** Untrained task types score 0%,
  which is the expected result and also the limit of the claim: this is
  arithmetic in four fixed phrasings with small operands.
* **That accuracy is safe to optimise.** It is recitation-proof, which the loss
  was not, but it is still one narrow benchmark.
