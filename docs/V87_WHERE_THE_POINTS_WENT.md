# v87: where v86's missing points actually went

Measured 5 September 2026 against `output/v86_corpus/v86_corpus.pt`, the
checkpoint that scores 0.779 on the 21-task benchmark. Every number here comes
from a receipt in `output/v87_measurements/`.

v86 leaves 139 of 630 problems wrong. This document locates them. It turns out
they are not spread across the model's competence: **every failing task fails at
one identifiable step, and six of them fail for two reasons that were never
about difficulty at all.**

---

## 1. What the failures are not

The first hypothesis was carrying. Every decomposed format in this corpus
performs one addition it never writes — `multiplication` emits
`70 x 5 = 350, 9 x 5 = 45, total 395` and leaves `350 + 45` silent — and the
partials of a place-value split are usually disjoint, so that silent addition
is assembly rather than arithmetic. Where the partials collide it is real
addition, and the guess was that those are where the points go.

`source/step_audit.py` was written to test this, and it refuted it. Scanning
the corpus for silent additions that need a carry:

| task | v86 accuracy | rows with a carrying silent step |
|---|---|---|
| `force` | 1.000 | 0.174 |
| `multiplication` | 1.000 | 0.172 |
| `voltage` | 1.000 | 0.189 |
| `two_step` | 0.300 | 0.000 |
| `power` | 0.400 | 0.000 |
| `algebra_one_step` | 0.433 | 0.000 |

Three tasks with a carrying silent step in a sixth of their rows score a full
point. Three with none score below 0.45. The hypothesis is dead in the
direction it was proposed and in the reverse direction too.

It is worth recording because the tool built to test it is what found
everything below.

---

## 2. What the failures are

### 2.1 Each written step can determine one place, and no more

`power` writes its division whole: `19152 / 76 = 252`. `division` splits the
quotient by place value: `150 / 3 = 50, 21 / 3 = 7`. Same operation, same
model, same benchmark — 0.400 against 1.000.

Between tasks that is a correlation with a dozen confounds. So
`division_dose_response.py` holds the task, the format, the four prompt
wordings and the model fixed and moves only the size of the operands:

| divisor ↓ / quotient → | 1-digit | 2-digit | 3-digit |
|---|---|---|---|
| 1-digit | 0.725 | 0.375 | 0.125 |
| 2-digit | 0.750 | 0.575 | 0.125 |

The quotient drives it and the divisor does almost nothing — 0.725 against
0.750 at a fixed quotient width. `division_step_true` tracks accuracy to within
a point in five of the six cells, so the division step is the failure, not a
transcription loss after it.

That still conflates two things, because a three-digit quotient in the range
[100, 300] is usually also a three-*significant*-digit quotient.
`significant_digits_sweep.py` separates them, holding the width at three digits:

| quotient | example | accuracy |
|---|---|---|
| 1 digit | 7 | 0.825 |
| 2-digit round | 50 | 0.750 |
| 3-digit round | 200 | 0.525 |
| 3-digit, two places | 250 | 0.275 |
| 3-digit, three places | 174 | 0.075 |

**Seven-fold at constant width.** The cost is per place the model has to
determine, which is exactly the quantity place-value splitting reduces to one.

This also explains the three science divisions' ranking without any further
assumption. Their quotient ranges are `power` 2–300, `molarity` 1–60,
`acceleration` 2–60; their scores are 0.400, 0.667, 0.733.

### 2.2 `percent` is two tasks averaged

`build_scratchpad_math._scratchpad_percent` drew its percentage from
`[5, 10, 20, 25, 50]`. `eval_problem_solving._percent` draws from
`[5, 10, 12, 15, 20, 25]`.

Split v86's percent replies by the percentage asked for:

| percentages | in corpus | v86 |
|---|---|---|
| 5, 10, 20, 25 | yes | **16/17 = 0.941** |
| 12, 15 | no | **0/13 = 0.000** |

`percent` is not a 0.533 task. It is a 0.941 task on what it was taught and a
0.000 task on what it wasn't, and it has been read as difficulty since v79.
Twelve and fifteen appear in a third of the benchmark's percent problems and in
none of the corpus's forty thousand percent rows.

Nothing reported this. The two generators are in different files, neither
imports the other, and their disagreement produces a plausible middling score
rather than an error. `source/coverage_audit.py` now compares them directly and
`test_coverage_audit.py` fails if the gap reopens. Run over all thirty tasks,
percent was the only hole.

### 2.3 The remaining percent failures are the silent sum

Ten of percent's fourteen wrong replies have **every written step true**:

```
What is 15% of 56?
1 percent of 56 = 0.56, times 10 = 5.6, times 5 = 2.8, total 1.8
```

Both parts right; 5.6 + 2.8 produced 1.8. The addition of the two parts was the
one operation the format never wrote. `decompose_product` already writes its
running sums once there are more than two partials — two place values cannot
carry into each other, but 5.6 and 2.8 collide in the tenths.

### 2.4 First-error accounting

Every wrong reply, classified by where it first goes wrong:

| task | wrong | a written step is false | every step true | unreadable |
|---|---|---|---|---|
| `average` | 29 | 13 | **16** | 0 |
| `two_step` | 21 | 21 | 0 | 0 |
| `power` | 18 | 18 | 0 | 0 |
| `algebra_one_step` | 17 | 17 | 0 | 0 |
| `percent` | 14 | 4 | **10** | 0 |
| `arithmetic` | 11 | 7 | 4 | 0 |
| `molarity` | 10 | 10 | 0 | 0 |
| `word_problem` | 9 | 9 | 0 | 0 |
| `acceleration` | 8 | 8 | 0 | 0 |

Nothing is unreadable and nothing is noise. Where the count is in the middle
column the arithmetic is wrong at a known position; where it is in the right
column every written step is true and the error is in an operation the format
does not write. `average` and `percent` are the two tasks with a silent step,
and they are the two tasks in the right column.

`algebra_one_step` deserves its own line. All 17 of its failures state the
correct inverse operation and then evaluate a double negative wrongly:
`6 - -12 = 27`, `3 - -15 = 26`, `19 - -7 = 22`. The plan is never the problem.

---

## 3. What v87 changes

Six format changes, each confined to a task that scored below 0.75, plus the
code family that was built for v86 and never trained.

| change | task | evidence |
|---|---|---|
| `decompose_quotient` | `power`, `molarity`, `acceleration` | §2.1 |
| percentages cover the benchmark | `percent` | §2.2 |
| write the sum of the parts | `percent` | §2.3 |
| `--average_binary_steps` | `average` | §2.4, right column |
| `--algebra_word_sign` | `algebra_one_step` | §2.4, double negatives |
| nine code tasks at 20,000 rows | `code_*` | never trained |

**The eleven tasks at 1.000 are untouched and act as a control.** If they fall,
that is dilution — the mechanism that cost v80 21.5 points — and not any format
here.

### The step budget under-compensates, and this is an error

Steps went from 18,000 to 21,500 to offset the corpus growing 18% **in rows**.
That is the wrong quantity. The format changes also made rows *longer* — `power`
alone went from a 39-token median response to 69 — and training consumes tokens,
not rows.

Measured over 20,000 sampled rows from each corpus with one shared tokenizer:

| | v86 | v87 | ratio |
|---|---|---|---|
| mean tokens per row | 52.0 | 60.1 | 1.156 |
| rows | 976,108 | 1,156,108 | 1.184 |
| **total tokens** | | | **1.369** |
| steps | 18,000 | 21,500 | 1.194 |

Per-token exposure is therefore **0.872 of v86's**, a 13% shortfall. Matching it
exactly would need 24,642 steps, about four hours more.

It is probably not fatal. v86's accuracy curve reads 0.22, 0.40, 0.47, 0.66,
0.78, 0.78 at steps 3,000 to 18,000 — flat over the last 3,000, so it had
converged with headroom. Matching v86's *converged* point rather than its final
step needs 15,000 × 1.369 = 20,535 steps, which 21,500 clears. The margin is
thin and it was not chosen; it is luck, and the next run should budget on tokens.

### Verification of the built corpus

`datasets/v87/v87_combined.jsonl` is 1,156,108 rows: 22 tasks at 40,000, nine
code tasks at 20,000, and the 96,108 language rows copied byte-for-byte out of
the v86 file so the one unchanged component cannot contribute a difference.

Over a 31,000-row sample spanning all 31 tasks:

- **0 rows whose answer the benchmark's extractor cannot read.** A response
  ending in a unit — `total 5 m/s^2` extracts as 2 — has made a task
  unwinnable before.
- **0 disagreements with `answer_check`**, which re-derives each answer from
  the *question* rather than the response, so it is a second implementation
  and not a restatement.
- **0 turns at or over the 128-token block** among task rows; the longest are
  `arithmetic_series` at 107 and `power` at 96. Turn-aligned packing drops an
  over-length turn silently, so this is checked rather than assumed. The only
  rows that reach the block are language rows, which behave exactly as they did
  in v86.
- Every science row verified by `nexus_solver`, every code row by executing the
  snippet (drop rate 0.000).

---

## 4. What is not established

- **No format change here has been trained.** Every number above is v86.
  The predictions are predictions.
- The dose-response is one task on one checkpoint. It says a
  one-significant-digit step is easier than a three-significant-digit one for
  this model; it does not say the rewrite will reach 1.000, and `division`'s
  1.000 is a different task with a smaller range.
- **The `average` fix is partial by construction.** `--average_binary_steps`
  writes the running sum as equations, which addresses the 16 replies whose
  every written step was true. But those equations are themselves multi-place:
  `238 + 58 = 296` determines three places in one step, which §2.1 says is the
  expensive shape. Splitting each addition by place value as well is what v73
  measured and rejected — at sequence length 128 it scored 16.0% against 24.0%,
  because the rows stopped fitting. With binary steps the longest average turn
  is already 100 tokens of 128, so there is no room for both. Expect this task
  to move a long way from 0.033 and not to reach the eleven at 1.000.
- `two_step` (0.300), `word_problem` (0.700) and `arithmetic` (0.633) are
  located but not fixed. All three fail at a written step, and all three still
  contain one-jump operations — `5.66 x 50`, `111 - 43`, `700 - 600` with
  three-digit operands. The same principle applies and no measurement here
  tells us how much of their loss it accounts for. See §6.
- The out-of-distribution probe (`subdivision_probe.json`) found v86 rewriting
  a prompt's numbers into its trained range and answering the rewritten
  problem — `420 / 7` answered as `320 / 7 = 60, total 60`, with the right
  answer above false working. **That does not happen in distribution**: 2 of
  491 correct replies (0.4%) sit above a false step. The fabrication is an
  out-of-distribution collapse, not a general property of the scratchpad.

---

## 5. Research this drew on

- Mirzadeh et al., *GSM-Symbolic* (ICLR 2025) — accuracy over a single draw
  hides variation across instantiations of the same problem. §2.2 is that
  effect at its sharpest: one task, two populations, 0.941 and 0.000.
- *Shattered Compositionality* (arXiv 2601.22510, January 2026) — transformers
  trained on composed arithmetic learn task-specific representations rather
  than reusable components. Consistent with `division` at 1.000 and `power` at
  0.400 for the same operation, and with §2.1's finding that the cost is per
  step written rather than per problem.
- *Arithmetic Pedagogy for Language Models* (arXiv 2606.05106, June 2026) —
  an 86M decoder trained only on next-token prediction over serialised
  execution traces reaches over 80% on held-out arithmetic. The same claim this
  corpus rests on, at five times the parameters.

None of these were followed on authority. §1 records a hypothesis they would
have supported that the measurements refuted.

---

## 6. A lead for v88, labelled as a lead

The three tasks left unfixed fail in a way that looks the same across all of
them. Their wrong replies are almost all off by a power of ten:

```
two_step        139 - 17 = 112     truth 122      +10
word_problem    111 - 43 =  78     truth  68      +10
word_problem     40 + 18 =  68     truth  58      +10
arithmetic       61 - 56 =  15     truth   5      +10
```

The units digit is right and the tens digit is wrong by one. `111 - 43`: the
units column borrows, and the model does not carry the borrow into the tens.
`40 + 18`: nothing carries, and it added one anyway.

Across all 630 replies, every written addition and subtraction, split by
whether the operation actually needs a carry or a borrow:

| | steps | false | of those, off by exactly 10/20/100 |
|---|---|---|---|
| carry-free | 193 | 20 (0.104) | 8 |
| needs a carry | 167 | 31 (**0.186**) | 18 |

95% intervals [0.068, 0.155] and [0.134, 0.251]; Fisher exact two-sided
**p = 0.033**.

**This is post-hoc.** The comparison was chosen after reading the failures, on
the same data that suggested it, and one test at p = 0.033 chosen that way is a
lead and not a result. It also explains only part of the loss: carry-free steps
still fail one time in ten.

Note carefully that this is **not** the hypothesis §1 refuted. That one was
about the *unwritten* sum a decomposition performs, where the partials are
disjoint place values and no carry is possible; it predicted nothing and was
killed. This is about a carry *inside a single written two-digit step*, which
is a different operation. The first being wrong is not evidence for the second.

The designed test is cheap and does not need a training run: generate
`arithmetic` problems whose remainder step does and does not borrow, in equal
numbers, and score them. If it holds, the fix follows the same principle as
§2.1 — split the remainder so no written step spans a carry, `61 - 56` becoming
`61 - 50 = 11, 11 - 6 = 5`. That costs tokens on every arithmetic row, so it
needs the measurement first.

---

## 7. Why the tens digit specifically — and two ways out

§6 found the errors are almost all off by ten, with the units digit right. The
mechanism is documented and it is structural rather than a capacity limit.

Addition carries **right to left**: the tens digit of `111 - 43` cannot be known
until the units column has been resolved. An autoregressive model emits **left
to right**, so it must commit to the tens digit *before* computing the carry
that determines it. Lee et al., *Teaching Arithmetic to Small Transformers*
(arXiv 2307.03381) measure this directly: the plain format is suboptimal while a
reversed output format shows a phase transition to near-perfect addition at
around 2,500 training samples, because reversing makes each output digit depend
only on the two operand digits and the previous carry. They also report that
corrupting the tens place specifically drives error above 0.2 — the same digit
this corpus loses.

This module already cites that paper, for the `AVERAGE_BINARY_STEPS` rationale.
The carry-direction result in it has not been applied.

**Reversing the output is not available here.** Every answer in this corpus is
extracted as the last number of the reply, and every benchmark score since v65
rests on that. Emitting digits least-significant-first would break the
extraction contract and unpair every historical comparison.

**Splitting further is available**, and it is the same move as §2.1: make each
written step determine one place, so no step spans a carry at all.

    61 - 56   ->  61 - 50 = 11, 11 - 6 = 5

This is what `decompose_quotient` does for division and what
`decompose_product` does for multiplication — both of which score 1.000. The
cost is tokens on every arithmetic row, which is why §6 wants the designed
measurement before the change.

### A separate architectural lead, for v89 and not sooner

Abacus embeddings (*Transformers Can Do Arithmetic with the Right Embeddings*,
arXiv 2405.17399) give each digit an embedding encoding its position **within
its number**, which is the information a left-to-right decoder lacks. The
reported result is 99% on 100-digit addition, and — the part that matters
here — combining them with input injection and a **looped** transformer takes
out-of-distribution accuracy from 92.9% to 99.1%.

Supermix already tokenises digits separately, and it already has a recursive
thinking core, which is **currently inert**. That is a suggestive pairing and
nothing more: none of it has been tried here, the reported numbers are on
models trained solely to add, and this project's own history is full of
architecture changes that measured flat. It is recorded so the next person
starts from the paper rather than from scratch, not as a plan.
