# Supermix v70 — sparse experts beat both specialists

V69 unified chat and arithmetic in one model and paid 25 points of maths for it:
40.0% against v68's 65.0%. The diagnosis was capacity -- two domains competing
for one dense feed-forward path.

V61 had already tested "more capacity" and found nothing: 3.18x the parameters
moved held-out loss less than the noise floor. But v61 ran on a **single-domain**
templated corpus, and it also measured something else -- that MoE routing
genuinely specialises, costing 3.15x more to destroy at 32 experts than at 8.

The prediction, stated before the run: sparse experts should pay on a corpus with
two genuinely different domains, precisely because there is something to
specialise *into*. That is what happened.

## The result

Same benchmark, same novel problems, same scorer:

| task | v68 (maths only) | v69 (unified, 8 experts) | **v70 (unified, 32 experts)** |
| --- | --- | --- | --- |
| `word_problem` | **100%** | 41.7% | **100%** |
| `arithmetic` | **91.7%** | 41.7% | **91.7%** |
| `algebra_one_step` | 79.2% | 50.0% | **91.7%** |
| `average` | 0% | 4.2% | **33.3%** |
| `percent` | 54.2% | **62.5%** | 58.3% |
| **overall** | 65.0% | 40.0% | **75.0%** |
| median relative error | 0.000 | 0.005 | **0.000** |
| within 10% of truth | 83.3% | 80.0% | **89.2%** |
| holds a chat register | no | yes | **yes** |

**v70 beats the maths specialist by 10 points while also holding a conversation.**
It is better than v69 on four of five tasks and better than v68 on three, tying
on the two v68 had already maxed.

`average` is the headline within the headline. It was 0% in v68, 4.2% in v69, and
is **33.3%** here -- the task whose scratchpad lists results without decomposing
each addition, which v68 identified as the structural failure. Sparse capacity
did not fix the format, but it made the long running sum survivable:

    Find the average (mean) of these numbers: 40, 60, 20, 80
    sum: 40 then 100 then 120 then 200, total 200, divide by 4, total 50.0

Every partial sum correct, division correct.

## The capacity was nearly free

| | v69 | v70 |
| --- | --- | --- |
| total parameters | 4,586,025 | **8,586,345** (1.87x) |
| active per token | 2,802,333 | 2,821,341 (**+0.7%**) |
| measured step cost | -- | **+14%** |

`top_k=2` means most of the new capacity is dark on any given token. This is the
MoE argument working as advertised, on a CPU box.

## Chat is unchanged, and still recitation

| prompt | reply | verdict |
| --- | --- | --- |
| hello | Hello. Tell me what you need and I will do my best to help. | largely_recalled, **1.00** |
| why is my script failing | Check the traceback first, then we can isolate the failing function. | largely_recalled, **1.00** |
| 617 + 288 | `600 + 200 = 800, 17 + 88 = 105, total 905` | **mostly_novel, 0.00** |

Indistinguishable from v60 and v69, and just as memorised. Sparse experts bought
arithmetic, not composition -- the dialogue half is still reproducing training
text, and nothing here changes that.

## What this does not prove

* **That the experts caused it.** Three things changed together: 8 -> 32 experts,
  200k -> 300k maths rows, and 14k -> 16k steps. v68 had 300k maths rows and
  scored 65%, which argues the expert count did real work, but this run cannot
  separate the three. A matched arm at 8 experts was not run.
* **That v61 was wrong.** V61 measured a single-domain corpus and found capacity
  useless there. Both results can hold: capacity is useless without something to
  specialise into, and valuable with it. That is the reading, not a refutation.
* **That the tier ladder means what it usually does.** Tier 3 (0.1079) scores
  *lower* than tier 1 (0.1128), ratio 0.995x. That is not generalisation beating
  recall -- tier 3 is 13,955 of 15,623 rows and dominated by highly structured
  arithmetic, so "unseen sentence" there is easier than dialogue. It is a corpus
  composition artifact and should not be read as a result.
* **That 75% is reliable arithmetic.** It is wrong one time in four, and
  `percent` regressed slightly against v69.
* **That the chat half improved.** It did not. It is the same recitation, now
  next to better arithmetic.

## What to use

**`output/v70_moe/v70_moe.pt` supersedes both v68 and v69.** It is the best
maths model here *and* the only one that also converses. v60 remains equivalent
for pure chat and is smaller if that is all you need.
