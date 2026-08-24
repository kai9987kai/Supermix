# Supermix v69 — one model does both, and is worse at each than the specialist

V69 combined the only two things measured to work at this scale: the dialogue
corpus that made v60 fluent, and the scratchpad arithmetic that took v68 to 65%.
It used every fix from this session -- digit tokens, turn-aligned packing,
per-improvement checkpointing, and `--select_on balanced`.

**The unification worked. The claim "better than all previous in most ways" did
not.**

## The result

| | v68 (maths specialist) | v60 (chat specialist) | v69 (unified) |
| --- | --- | --- | --- |
| maths, overall | **65.0%** | -- | 40.0% |
| arithmetic | **91.7%** | -- | 41.7% |
| word_problem | **100%** | -- | 41.7% |
| algebra_one_step | **79.2%** | -- | 50.0% |
| percent | 54.2% | -- | **62.5%** |
| average | 0% | -- | **4.2%** |
| median relative error | **0.000** | -- | 0.005 |
| holds a chat register | no | yes | **yes** |
| parameters | 2,990,313 | 4,988,073 | 4,586,025 |

v69 answers both kinds of prompt in one model:

    > why is my script failing
      Check the traceback first, then we can isolate the failing function.
    > Solve this basic math problem: 617 + 288
      600 + 200 = 800, 17 + 88 = 105, total 905

The chat replies are indistinguishable from v60's. The arithmetic is correct
here and wrong more often than the specialist's.

## The predicted cost, realised

The risk was stated before the run: v62 showed mixing domains at this scale
splits capacity, and v67 -> v68 measured it directly when going from four task
types to six cost arithmetic 13 points. Adding a second *kind* of task cost far
more than adding a sixth of the same kind:

* arithmetic 91.7% -> 41.7%
* word_problem 100% -> 41.7%
* algebra 79.2% -> 50.0%

Two tasks improved (`percent`, `average`), which is noise-sized against those
losses. The aggregate is 65.0% -> 40.0%.

Nothing here is surprising given v62. The difference this time is that both
ingredients were individually learnable, and it *still* cost 25 points -- so the
constraint is capacity, not the unlearnability of one ingredient.

## What is genuinely better

* **It is the only model here that does both.** Every previous checkpoint is a
  specialist: v60 cannot add, v68 cannot converse.
* **Vocabulary 8,444 with digit tokens** -- smaller than v60's 10,538 despite
  covering arithmetic, because digits collapse 8,588 number symbols into ten.
* **The tier ladder is the tightest measured**: 0.1013 / 0.1522 / 0.1750, ratio
  1.076x, on a corpus that is not purely templated.
* **It was selected on `balanced`**, not dev loss -- the first run in this repo
  chosen by a criterion that can see recitation.

## The recitation result, which is the honest headline

Scored against its own training corpus, the chat replies are **100% verbatim**:

| prompt | verdict |
| --- | --- |
| hello | largely_recalled, 1.00 |
| can you help me write some tests | largely_recalled, 1.00 |
| why is my script failing | largely_recalled, 1.00 |
| Solve this basic math problem: 617 + 288 | **mostly_novel, 0.00** |
| What is 25% of 840? | part_recalled, 0.50 |

The fluent half is recall; the computed half is not. That is the clearest single
demonstration this repo has produced of the distinction the recall meter was
built for -- and it applies equally to v60, whose replies come from the same
corpus and would score the same way. **v60's apparent fluency was always
recitation; v69 makes that visible by sitting next to arithmetic that is not.**

## What to use

* **maths only** -> `v68_average_fix` (65.0%)
* **chat only** -> `v60_control_2000` or v69; they are equivalent, and both recite
* **both in one process** -> `v69_unified_b`, accepting 40% maths

## What this does not prove

* **That unification always costs this much.** One corpus mix, one budget, one
  model size. A larger model or a longer run might close it; neither was tried.
* **That `percent` and `average` genuinely improved.** +8.3 and +4.2 points on 24
  problems each is within the range these runs move by for reasons this session
  has repeatedly failed to attribute.
* **That the chat half is bad.** Reciting a coding-assistant reply is exactly
  what the corpus teaches, and it reads well. It is simply not composition.
* **That "better in most ways" was achievable here.** It was the goal; it was not
  reached, and averaging the wins against a 25-point aggregate loss would be the
  wrong way to report that.
