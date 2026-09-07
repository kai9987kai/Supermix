# v88: understand the question, not the template

v87 ended level with v86 on maths (0.7677 against 0.7745, McNemar p = 0.81) and
added nine code-tracing tasks at 0.894. It also produced a clear negative result
that this version acts on, and exposed a capability gap that no benchmark in the
project has ever measured.

---

## 1. What v87 proved, and what v88 reverts because of it

`decompose_quotient` and the percent written sum are **off**, both now behind
documented flags rather than deleted, so the negative result stays reproducible.

| task | v86 | v87 | change reverted |
|---|---|---|---|
| `power` | 0.333 | **0.048** | `decompose_quotient` |
| `molarity` | 0.667 | 0.333 | `decompose_quotient` |
| `acceleration` | 0.762 | 0.571 | `decompose_quotient` |
| `percent` | 0.476 | 0.286 | written sum (coverage fix **kept**) |

The rule both violated, stated once so it does not have to be rediscovered:

> **A decomposition helps only when every step it adds is derivable forward from
> what is already written, and inside the model's arithmetic.**

`decompose_quotient` split the **output**, so its partial dividends could only be
obtained by already knowing the answer. The percent sum added a decimal addition
that carries, which this model fails 0.186 of the time against 0.104 for one that
does not.

Everything v87 got right is kept: `algebra_one_step` (+0.476), `average`
(+0.238), the percent coverage fix, and the nine code tasks.

**Prediction for v88, recorded before the run:** `percent` returns to roughly
v86's per-value accuracy of 0.94 on all six benchmark percentages rather than
0.94 on four and 0.00 on two, so ~0.9 against v86's 0.476 — the single largest
expected gain, and it comes from a bug fix rather than a format idea.

---

## 2. The gap no benchmark here has measured

Every benchmark in this project generates its prompts from the same templates the
corpus was built from. All of them therefore measure the model on its own dialect,
and that has hidden a real weakness twice:

- v74 scored **0.894** on its own benchmark and answered **0 of 5** questions
  typed by a person.
- v80 scored **0.556** on hand-typed questions against **0.778** for the same
  questions rewritten into the corpus format — a 22-point gap caused by nothing
  but wording (`output/v85_measurements/natural_phrasing.json`).

`prompt_normaliser.py` exists to paper over this at inference time, and the chat
server runs it by default. It rewrote 16 of those 18 questions.

### `natural_phrasings.py` was dead code

The module carries 112 hand-written phrasings across twelve tasks — casual
register, contractions, missing punctuation, filler openers, long and short units,
number before quantity as well as after:

```
hey can you work out the force for a 42 kg mass at 8 m/s^2
i have a 42 kg mass speeding up at 8 m/s^2, what force is that
force please: 42 kg at 8 m/s^2
```

`build_omni_corpus._pick` takes a `_task` argument that reaches it, and **not one
of the twelve generators passed it**. The flag existed, the module existed, the
wiring did not. Every corpus ever built used the narrow four or five templates.

v88 wires it. Measured on the built rows, this is a genuine single-variable
change:

| property | result |
|---|---|
| answers and worked responses identical | 3996 / 3996 rows |
| prompts that differ | 3703 / 3996 |
| solver verification failures | 0 of 720 |
| `canonical` (what the oracle parses) identical | yes |
| median prompt tokens | **15 → 14** (casual forms are terser) |
| turns reaching the 128 budget | 0 |

It costs nothing and cannot weaken verification, because `canonical` is built
from the parameters rather than the phrasing.

---

## 3. The measurement that makes it mean something

Widening a bank and scoring on that bank measures nothing — memorising fifteen
templates is no harder than memorising five. So
`natural_phrasings.HELD_OUT_PER_TASK` withholds the **last three forms of every
task** from training, and `source/eval_natural_phrasing.py` scores on exactly
those: 36 phrasings across 12 tasks that no corpus has ever contained.

It is paired. Each problem is asked twice with the same operands and the same
answer, once in a trained form and once in a held-out one:

```
trained    Find the acceleration produced by 4500 N on 180 kg.
held out   how quickly does a 180 kg object speed up when pushed with 4500 N
```

so the only variable is the wording, and McNemar's exact test on the discordant
pairs is the right test. **A model that has learned the task rather than the
template scores the same in both columns.** No normaliser runs; needing one is
what this measures.

`test_natural_phrasings.py` pins that no held-out form can reach a corpus. Its
first version passed for the wrong reason — `m/s^2` contains a digit, so the
template and the rendered prompt were normalised differently and the leak check
could never fire. Both sides now go through one helper.

---

## 4. A larger model, sized by measurement

Parameters are the wrong currency for a sparse model: a token routes to `top_k`
experts, so `n_routed_experts` buys capacity at almost no compute while
`hidden_size` and `n_layers` buy it at full price.
`output/v87_measurements/size_cost.py` times the candidates **interleaved in one
process**, because timings on this box are not comparable across processes — the
same benchmark has read 2.045, 11.037 and 2.136 s/step, and v87 lost 3.8 hours to
the CPU clock dropping to 37% while the battery charged.

The step budget will be set on **tokens, not rows**. v87's was scaled by row
count and left per-token exposure at 0.872 of v86's; the control tasks held
anyway, but that was luck.

---

## 5. Intelligence: pre-registering the next format change

The three tasks v87 left unfixed — `arithmetic` (0.714), `word_problem` (0.762),
`two_step` (0.429) — fail in a uniform way. Their errors are almost all off by
exactly ten, with the units digit right:

```
111 - 43 = 78   truth 68     40 + 18 = 68   truth 58     61 - 56 = 15   truth 5
```

Across all 630 v87-era replies, a written `+`/`-` step that needs a carry is
false 0.186 of the time against 0.104 for one that does not (Fisher exact
p = 0.033). The mechanism is documented: carries run right-to-left and an
autoregressive decoder emits left-to-right, so it must commit to the tens digit
before computing the units carry that determines it (Lee et al., arXiv 2307.03381).

The obvious fix is to split so no written step spans a carry —
`61 - 56` becoming `61 - 50 = 11, 11 - 6 = 5`. That satisfies both halves of the
rule in §1: the tens digit is read off the subtrahend, so it is derivable
forward, and neither step crosses a column.

**It is not being built on that reasoning alone.** That is exactly what v87 did.
`output/v87_measurements/carry_probe.py` asks the same subtraction three ways —
whole, split, and a non-borrowing control — and only a result where the split
beats the whole step *and* the control confirms borrowing is the mechanism
justifies the corpus arm. If the control scores no better than the borrowing
case, carrying is not the problem and the split buys nothing.

---

## 6. One consequence to watch

If v88 learns to answer questions as typed, `prompt_normaliser` becomes
unnecessary — and possibly harmful, since it rewrites a form the model now
understands into one it merely also understands. The chat server runs it by
default. The held-out benchmark deliberately runs without it, so the two numbers
together say whether the default should change.
