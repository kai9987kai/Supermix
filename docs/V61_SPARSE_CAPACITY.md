# Supermix v61 — 3.18× the parameters, and what they did not buy

V61 set out to build a more powerful model by scaling the one mechanism v59
identified as load-bearing: sparse MoE routing. It scaled 4,988,073 parameters to
**15,883,701** — 8 routed experts to 32, 4 layers to 6 — for only 1.26× active
compute, because `top_k=2` means most of the new capacity never fires on any
given token.

On held-out loss it bought **nothing measurable**, and the control that shows
this is the reason to trust the rest of the document.

## The comparison is exact, by construction

All three runs share a corpus, a split seed, and therefore an identical
tokenizer, identical withheld sentences and identical tier row counts. Their
losses are comparable rather than merely adjacent:

| run | total params | active | experts | layers | steps | dev loss |
| --- | --- | --- | --- | --- | --- | --- |
| `v60_diverse` | 4,988,073 | 3,204,381 | 8 | 4 | 1,000 | 0.2852 |
| `v61_scaled` | **15,883,701** | 4,036,137 | 32 | 6 | 2,000 | 0.2531 |
| `v60_control_2000` | 4,988,073 | 3,204,381 | 8 | 4 | 2,000 | **0.2512** |

`source/compare_generalisation_runs.py` refuses to print this table unless the
withheld sentences, vocabulary and tier sizes all match, because a table of
losses invites a comparison the numbers may not support.

## The scale-up did not pay off

| tier | v60 (1k steps) | v61 (15.9M, 2k) | control (4.99M, 2k) | v61 − control |
| --- | --- | --- | --- | --- |
| `tier1_seen_response` | 0.2260 | 0.1856 | 0.1870 | −0.0014 |
| `tier2_unseen_response` | 0.2180 | 0.2016 | 0.2058 | −0.0042 |
| `tier3_unseen_sentence` | 0.3135 | 0.2873 | 0.2860 | +0.0013 |
| dev | 0.2852 | 0.2531 | 0.2512 | +0.0019 |

**Every difference between v61 and the control is between 0.0013 and 0.0042 nats,
and the sign is not consistent** — v61 is ahead on two tiers, behind on the third
and on dev. V59 established that exact pattern as the signature of noise, at the
same magnitude, when it showed v58's arm-to-arm deltas were measuring run-to-run
variance rather than a mechanism. The same reading applies here, and it applies
against the conclusion this version was built to reach.

The entire v60 → v61 improvement came from **doubling the schedule**, not from
tripling the parameters. The matched-step check saw it coming: at 1,000 steps
v61's dev loss was **0.3170** against v60's **0.2852** — the larger model was
*worse* per step, which is what happens when 32 experts each receive a quarter of
the tokens 8 would.

## The capacity is real, and it is used — it just does not help

The scale-up was not inert. V59's audit was run on both 2,000-step checkpoints,
so the comparison is matched on schedule:

| intervention | control (8 experts) | v61 (32 experts) | ratio |
| --- | --- | --- | --- |
| `moe_routing_random` | +0.04854 | **+0.15298** | **3.15×** |
| `moe_routing_inverted` | +0.06401 | +0.19201 | 3.00× |
| `moe_shared_expert` | +0.01521 | +0.03583 | 2.36× |

Destroying the learned assignment costs **3.15× more** with 32 experts than with
8. The experts specialised; the routing genuinely carries more of the model's
behaviour. Baseline losses on the same evaluation slice are 0.53584 and 0.53853 —
indistinguishable.

So the model reorganised itself substantially and predicts the same tokens
equally well. **Specialisation is not capability.** On this corpus at this budget
the system is limited by schedule and data, not by capacity, and 10.9M extra
parameters bought internal structure with no external effect.

*(An earlier note in this work put that routing ratio at 6.2×. That compared v61
against the 1,000-step v60 and so folded the schedule into it; 3.15× is the
matched-step figure and supersedes it.)*

## The unseen-sentence gap closes completely

V60 found v58's headline +0.2309 nat withheld-sentence penalty shrank to +0.0043
on a diverse corpus. With 2,000 steps it disappears:

| checkpoint | corpus | steps | unseen − seen |
| --- | --- | --- | --- |
| `v58_full` | 292 word types | 1,000 | **+0.2309** |
| `v60_diverse` | 10,538 types | 1,000 | +0.0043 |
| `v60_control_2000` | 10,538 types | 2,000 | **+0.0002** |
| `v61_scaled` | 10,538 types | 2,000 | **−0.0008** |

Scored on the same 2,394 tier-3 rows, 96,239 seen-sentence tokens against 53,631
withheld-sentence tokens. At 2,000 steps a model predicts sentences it has never
seen exactly as well as the sentences beside them in the same response —
perplexity ratio 1.000 and 0.999.

This is the strongest available statement of v60's finding. V58's central
measurement — that the model "can emit every word of every withheld sentence and
is still markedly worse at them" — does not survive real language and an adequate
schedule.

## What this does not prove

* **That sparse capacity never helps.** It did not help *here*: one corpus, 2,000
  steps, 91,344 rows. A capacity-limited setting would look different, and
  nothing here identifies where that boundary is. The likeliest reading is that
  96k rows cannot fill 15.9M parameters.
* **That the differences are exactly zero.** They are below the resolution of a
  single run. No seed-to-seed spread has been measured for any of these runs, so
  "indistinguishable" means "smaller than an unmeasured noise floor with an
  inconsistent sign", not "identical". This is the same caveat v59 raised against
  v58, and it applies to v61 equally.
* **That 32 is the wrong number of experts.** Only 8 and 32 were trained to
  completion. The measured step-time curve (8.98M total for +4% wall-clock at 32
  experts, 21.7M at 64) says the design space is cheap to explore; it has not been.
* **That the gap-closing result generalises.** It is one corpus family, generated
  by one pipeline. "Unseen sentence" means unseen in this corpus, not unseen in
  English, and 5,280 word types is diverse relative to 292, not relative to a
  language.
* **That any of these models is good.** Dev loss 0.2512 on a 4.99M-parameter model
  over templated-and-synthetic dialogue. Nothing here measures reply quality,
  factuality, safety, or latency, and v58's remaining promotion gates are
  untouched.

## The practical conclusion

For this corpus and budget, **train the small model longer rather than making it
bigger**. `v60_control_2000` matches a model 3.18× its size on every tier while
using 1.26× less active compute per token, and it is the checkpoint to prefer.

The scaling lever that remains untested is data: the sparse capacity is real,
measurably used, and idle for want of rows.
