# Supermix v65 — arithmetic was unlearnable in principle, and a metric that cannot be recited to

Two requests -- answer what the asker actually wants, and solve problems better --
turned out to share one cause, and it was not a training shortfall.

## `498` was a single token

`mimomix_text.TOKEN_PATTERN` matches `\s*\d+`, so a whole run of digits becomes
one symbol. Measured on a 240,000-row arithmetic corpus, **8,588 of 9,058
distinct tokens (94.8%) were numbers**.

Under that tokenizer, answering `498 - 419` requires a memorised lookup from
token(498) x token(419) to token(79). The model cannot see that 498 contains a 4,
a 9 and an 8, so there is no compositional structure for it to learn.
**Arithmetic was not merely unlearned; it was unrepresentable.**

## The metric that proved it

`source/eval_problem_solving.py` scores whether the answer is *right*, which is
something recitation cannot fake: a remembered answer to a fresh problem is
simply wrong. It reports two populations --

* **seen** -- problems lifted verbatim from the training corpus,
* **novel** -- identical phrasing, operands generated at evaluation time,

-- because aggregate accuracy cannot distinguish a model that learned arithmetic
from one that memorised the corpus, and the gap between them can.

V64 scored **1.7% on both**, with a memorisation gap of **0.0000**. A zero gap is
the tell: it had not even memorised, because across millions of operand pairs
there was nothing memorisable.

## Splitting digits

`DIGIT_TOKEN_PATTERN` replaces `\s*\d+` with `\s*\d`. Roundtrip is preserved --
each digit carries its own leading whitespace -- and the tokenizer records the
setting so a checkpoint cannot be reloaded under the other one and silently
re-segment every number.

| | whole numbers | digit tokens |
| --- | --- | --- |
| vocabulary | 16,390 | **874** |
| held-out coverage | 0.9964 | **1.0000** |
| model parameters | 9,257,385 | **3,132,585** |
| seconds per step | ~4.3 | **0.8** |

The 18x smaller output softmax made training roughly five times faster, so the
fix costs nothing and pays for itself twice.

## What it bought, measured

200 problems, 100 seen and 100 novel, five task types:

| | v64 (whole numbers) | v65 (digit tokens) |
| --- | --- | --- |
| replies containing no number at all | **50 / 200** | **0 / 200** |
| novel exact accuracy | 1.0% | **4.0%** |
| novel median relative error | 0.621 | **0.250** |
| novel answers within 10% of truth | 9.0% | **23.0%** |
| seen answers within 10% of truth | 8.0% | **33.0%** |
| memorisation gap | +0.010 | +0.020 |

Exact accuracy is still near zero. The interesting change is everywhere else: the
model went from *emitting no number a quarter of the time* to *always answering,
in the right format, at roughly the right magnitude*:

| prompt | v65 answer | truth |
| --- | --- | --- |
| Find the average of 65, 84, 93, 10... | 51.5 | 51.333... |
| What is 20% of 1749? | 374.8 | 349.8 |
| Find the average of 61, 63, 72, 61 | 66.25 | 64.25 |

Four-number averages come back with `.25` decimals, which is structurally exactly
right. This is approximate arithmetic, not noise -- a different failure from
v64's, and a much better one.

The near-zero memorisation gap in both directions matters too: whatever v65 is
doing, it is **computing badly rather than reciting**. A model that had memorised
would show a large seen-minus-novel gap, and neither does.

## Relative error, and why exact match was not enough

The benchmark reports median relative error and a within-10% rate alongside exact
accuracy, because exact match cannot separate "computed approximately" from
"produced noise" -- and for a model this size that distinction is the entire
question. Under exact match alone, v64 and v65 look like 1% against 4%. Under
relative error they look like 0.62 against 0.25, which is the real change.

## What this does not prove

* **That the model can do arithmetic.** It cannot. 4% exact on novel problems is
  close to useless, and the honest summary is "right shape, right magnitude,
  wrong digits".
* **That digit tokenisation alone is the fix.** It removes a hard blocker. What
  remains -- exact multi-digit computation -- plausibly needs a scratchpad
  format, reversed digits, or simply far more capacity, none of which was tested.
* **That five task types are problem solving.** They are small-operand arithmetic
  with fixed phrasing. Nothing here measures reasoning, multi-step planning, or
  anything outside the generator.
* **That the improvement is all from tokenisation.** v65 also changed corpus
  (240k arithmetic rows against 30k), sequence length (64 against 128) and model
  size. The comparison is between two models, not two tokenizers.
* **That relative error is a quality measure.** An answer 25% wrong is wrong. It
  measures *how* wrong, which is diagnostic, not a score to optimise.
