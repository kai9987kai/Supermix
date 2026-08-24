# v76 — The benchmark said 0.894 and the chat said nonsense

Both were true. This is what was between them.

## The discrepancy

v74 finished at **0.894** on the n=500 problem-solving benchmark. The first
thing I did with it was serve it in the chat interface and ask it questions the
way a person would type them:

| typed | reply | correct |
|---|---|---|
| what is 47 times 6 | `400. 40 = 100.0, 70 + 47 = 117, total 117` | 282 |
| what is the average of 12 18 and 30 | `100 + 300 = 600, 20 + 28 = 48, total 648` | 20 |
| what is 128 divided by 8 | `100): 11): subtract 7 = 7.0, total 70.0` | 16 |
| what is 15 percent of 240 | `... total 260.0` | 36 |
| sarah has 23 apples and buys 19 more... | `This is one of those questions that rewards deep thinking.` | 42 |

**0 of 5.** A model that scores 0.894 on a benchmark and 0.0 on the same
arithmetic typed by hand is either a broken benchmark or a broken interface.

## It was neither. It was the prompt format.

The corpus writes multiplication as `What is 25 x 7?`. The benchmark generates
its problems in exactly that shape, so it measures the model on its own
distribution. A person types `times`, not `x`.

Rather than assume which part of the format mattered, I varied one feature at
a time against the running model:

| prompt | reply | |
|---|---|---|
| `What is 47 x 6?` | `40 x 6 = 240, 7 x 6 = 42, total 282` | correct |
| `what is 47 x 6?` | same | correct |
| `What is 47 x 6` | same | correct |
| `Quick question: 47 x 6` | same | correct |
| `Please help with this. 47 x 6` | same | correct |
| `What is 47 times 6?` | `400 x 6 = 200, 7 x 6 = 42, total 242` | **wrong** |
| `What is 47 * 6?` | `40 and 0 = 400, 7 x 6 = 42, total 442` | **wrong** |
| `47 x 6` | `subtract 6 from both sides, 44 - 6 = 38, total 38` | **wrong** |

So:

| feature | matters |
|---|---|
| the operator token (`x` vs `times` vs `*`) | **yes** |
| a lead-in phrase being present at all | **yes** |
| capitalisation | no |
| trailing question mark | no |

The last row of the first table is the interesting one. `47 x 6` with no
lead-in was read as **algebra** — "subtract 6 from both sides". The lead-in is
not politeness; it selects the task.

## The fix, and what it is not

`source/prompt_normaliser.py` maps how a person writes an operation onto the
token the model was trained on, then hands it to the model.

Typed naturally, through the server, after the change:

```
what is 47 times 6          -> asked as: What is 47 x 6?
                               40 x 6 = 240, 7 x 6 = 42, total 282
what is 128 divided by 8    -> asked as: Quick question: 128 / 8
                               80 / 8 = 10, 48 / 8 = 6, total 16
what is 20% of 150 then add 12 -> asked as: What is 20% of 150, then add 12?
                               1 percent of 150 = 1.5, times 20 = 30,
                               then 30 + 12 = 42, total 42
832 minus 630               -> asked as: Solve this basic math problem: 832 - 630
                               800 - 600 = 200, 32 - 30 = 2, total 202
```

**7 of 7, from 0 of 5.**

### This is presentation, not capability

It must not be described as making the model smarter. It computes nothing,
never alters a number, and never invents an operand. The evidence that it is
only presentation is that questions the model gets wrong in the training
format stay wrong:

> `What is 15% of 240?` → `1 percent of 240 = 2.4, times 10 = 24.0, times 5 = 12.0, total 26.0`

The decomposition is right (24 + 12 = 36) and the final sum is wrong. That is
the model, and it is why `percent` scores 0.75 rather than 1.00. Normalisation
does not touch it.

### The rewrite is always shown

The reply carries `asked_as` and `normalised_rule`, and the interface renders
them under the answer:

> *asked as: What is 47 x 6?  (multiplication)*

Answering a different question from the one someone typed, without saying so,
would misrepresent the model — it would look like it parsed natural language
when it did not. `--no-normalise` turns the whole thing off.

### What it deliberately leaves alone

Ordinary conversation and word problems go through untouched, verified against
the running server:

```
hello                        rewritten: None
why is my script failing     rewritten: None   -> "Check the traceback first..."
A student has 68 cookies...  rewritten: None   -> 68 + 32 = 100, 100 - 60 = 40, total 40
```

Word problems are already prose in the corpus, so rewriting them would only do
damage. If no rule recognises the shape, the text is passed through unchanged.

33 tests cover the normaliser. Most of them exist to pin what it must *not* do:
never compute an answer, never reorder `A - B`, never sort operands, never
guess a sequence from two terms, never touch conversation. Two real bugs were
caught while writing them — a missing `f` prefix that left `{NUMBER}` as a
literal in the algebra pattern, and `subtract 3 from 10` reversing its operands.

## A second gap the same test found

With normalisation working, the live verifier still said this:

```
what is 47 times 6        check=NOT CHECKED
what is 15 percent of 240 check=WRONG (said 26.0, expected 36.0)
```

`answer_check` re-derives the answer from the question so the interface can say
"wrong, it should be 36" rather than presenting confident arithmetic. It knew
five shapes — addition/subtraction, percent, algebra, average, word problem —
and v74 added four tasks it had never been taught. **The four tasks v74 is best
at (multiplication 1.00, division 1.00, sequence 0.98, two_step 0.98) were the
four nothing could verify.**

Now nine shapes. The ordering carries the real content, because several of them
are ambiguous with each other:

* `_two_step` must precede `_percent` — it *contains* a percent question.
* `_algebra` must precede `_multiplication` — `x` is this corpus's
  multiplication sign *and* its unknown, and only the digit on the left
  separates them.
* `_sequence` must precede `_average` — "7, 17, 27, 37" is a comma-separated
  list of numbers, which is exactly what an average looks like.

Two deliberate refusals: a sequence whose differences are not constant returns
`None` rather than guessing a rule, and division by zero is *not checkable*
rather than an exception. In both cases inventing a verdict would be worse than
admitting there isn't one.

A test asserts that every shape `supported_shapes()` advertises actually
parses — an interface promising to check something it cannot would be a lie in
the same direction as the "on dev" line in v75.

## The honest headline

**v74's 0.894 is a real measurement of the model on its training
distribution.** It was never a measurement of how it handles the way people
write, and before this change the gap between those two things was total.
