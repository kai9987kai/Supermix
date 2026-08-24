# Supermix v58 — the generalisation ladder

## What v58 is

V57 trained the v53 `MiMoMixModel` on real dialogue and reported held-out
perplexity **1.27**. In the same document it said that number
*"measures fit to a template distribution, not generalisation to unseen
language"*, and left it there. The caveat was correct and unquantified: it named
a risk without measuring it, so a reader had no way to tell whether the model
had learned to compose language or to recall templates.

V58 measures it. It is additive — `mimomix_core.py`, `mimomix_text.py` and
`train_mimomix_talk.py` are unmodified, and the checkpoint it writes is the same
`supermix-v57-talk-checkpoint-v1` the existing chat interface loads.

| surface | v58 contract |
| --- | --- |
| splits and their verification | `source/mimomix_eval_splits.py` |
| training and measurement | `source/train_mimomix_generalisation.py` |
| model | `mimomix_core.MiMoMixModel` — unchanged from v53 |
| checkpoint schema | `supermix-v57-talk-checkpoint-v1` (unchanged) |
| receipt schema | `supermix-v58-generalisation-benchmark-v1` |
| ablation schema | `supermix-v58-thinking-core-ablation-v1` |
| source of truth | `source/` |
| compatibility mirror | none — research line, same rule as v53, v56 and v57 |
| tests | `test_mimomix_eval_splits.py` |

## The measurement that motivates it

Under the v57 row split, measured rather than assumed:

| property | value |
| --- | --- |
| validation rows | 2,400 |
| **validation responses that appear verbatim in training** | **1,875 (78.1%)** |
| validation `(user, response)` pairs that appear in training | 0 |

And the corpus's real structure, which is why row-splitting cannot fix itself:

| property | value |
| --- | --- |
| rows | 120,000 |
| distinct responses | 37,543 |
| **distinct sentences across all responses** | **192** |
| median sentence frequency | 310 |

A response is a short composition drawn from a 192-sentence inventory — a prefix
(`Recommended path:`, `Short answer:`), a core, and optional trailers. Splitting
by row cannot produce an unseen response in any meaningful sense, because the
parts are all seen thousands of times.

## Three tiers, three names

"Held out" is not one property here, so it gets three names and three numbers.

| tier | definition | what scoring it measures |
| --- | --- | --- |
| `tier1_seen_response` | response string appears verbatim in training | **template recall** |
| `tier2_unseen_response` | response never in training; every sentence in it is | **sentence recombination** |
| `tier3_unseen_sentence` | response contains ≥1 sentence absent from training | **unseen-sentence composition** |

Tier 3 is only answerable by *training* on a split that excludes those
sentences, which is why `build_generalisation_split` builds the training set
rather than partitioning an existing one. Tiers 1 and 2 can be measured
retroactively against any checkpoint whose training rows are known.

### Tier 3 is a composition test, not a vocabulary test

The withheld sentences are chosen so that **every word in them remains in the
training vocabulary** — measured coverage **1.0000**, recorded in the receipt and
pinned by `test_the_real_corpus_gives_tier3_full_vocabulary_coverage`. The model
can express every held-out sentence. The question is only whether it assigns them
probability.

### What tier 3 is not

It is not a test on unseen *language*. A withheld sentence like
`"Recommended path: Yes."` is a novel pairing of a prefix and a core the model
has seen separately. With 192 sentences drawn from a 292-word corpus there is no
way to construct a genuinely unfamiliar sentence from this data. Tier 3 is the
hardest question this corpus can pose, which is a smaller claim than the hardest
question there is.

## The selection bias v57 carried

`train_mimomix_talk.py` keeps the checkpoint with the best **validation** loss and
then reports that same validation set's loss — a minimum over twelve evaluations
of the set being reported. In the published v57 run the loss fell monotonically,
so the minimum was also the final value and the bias cost nothing:

| step | 250 | 500 | 1000 | 1500 | 2000 | 2500 | 3000 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val loss | 0.3272 | 0.2840 | 0.2675 | 0.2513 | 0.2440 | 0.2376 | **0.2351** |

Nothing guaranteed that. V58 carries a separate **`dev`** split for selection;
the three tiers are scored once, after selection is finished, and never steer
training. `selection_never_read_a_tier` is a recorded check in the receipt.

## The split, verified rather than asserted

`verify_split` re-derives every property the tier names promise and raises if one
fails — it runs on the real object inside the trainer, not on a fixture in a
test. Three tests deliberately corrupt a split and require it to raise, because a
verifier that never rejects anything is the same as no verifier.

Duplicate `(user, response)` rows are kept on the same side of the split by
grouping on the pair before shuffling. Splitting rows independently would place
identical copies in both training and test — verbatim memorisation scored as a
held-out result, a smaller version of the leak this whole module measures.

## Results

`output/v58_full/generalisation_results.json`. The **identical** v57 architecture
— 3,076,521 parameters, 1,292,829 active per token, vocabulary 582 — trained for
1,000 steps at batch 16, sequence length 128, 1,550 seconds on CPU. Selected at
step 1,000 on dev (dev loss 0.2403); the tiers were scored once afterwards.

The split: 114,036 training rows, 1,175 dev, 30 sentences withheld (162 of the
192 remain in training). Every tier has response vocabulary coverage **1.0000**.

| tier | rows | tokens | loss | perplexity | bits/token |
| --- | --- | --- | --- | --- | --- |
| `tier1_seen_response` — template recall | 1,883 | 45,405 | 0.2329 | **1.2622** | 0.336 |
| `tier2_unseen_response` — recombination | 468 | 17,218 | 0.2540 | **1.2891** | 0.366 |
| `tier3_unseen_sentence` — composition | 2,438 | 61,874 | 0.3165 | **1.3723** | 0.457 |

| gap | nats |
| --- | --- |
| recombination cost (tier 2 − tier 1) | **+0.0211** |
| unseen-sentence cost (tier 3 − tier 2) | **+0.0626** |
| total (tier 3 − tier 1) | **+0.0836** — a 1.087× perplexity ratio |

Uniform baseline 6.3665. Speculative decoding stayed token-identical to greedy
(acceptance length 2.83), routing entropy 0.9998 normalised with 0 starved
experts, and all five receipt checks pass.

### Reading the number

**The ladder is monotonic.** Every step from recall to recombination to
unseen-sentence composition costs the model something, in the order the tier
names predict. That is the first thing to check about any new split, because a
split whose tiers do not order as their names claim is not measuring what it
says — and this one does.

The total cost is +0.084 nats: perplexity 1.26 → 1.37. Read as a headline, that
says v57's number was well-caveated and not badly flattered.

**Do not stop reading here.** A tier-3 *row* is not a tier-3 *sentence*, and the
next section shows that averaging over the row hides most of the effect. The
1.37 is the honest replacement for v57's 1.27 at the row level; it is not the
answer to what the model does with a sentence it has never seen.

### Where the dilution is, and the sharper number

A tier-3 response is *mostly* seen sentences — only one of its two-to-five
sentences is withheld — so +0.0626 nats is averaged over tokens that are largely
familiar. `source/eval_mimomix_unseen_sentences.py` removes that dilution by
scoring two disjoint token sets **inside the same tier-3 rows**: the tokens that
lie inside a withheld sentence, and every other response token in those rows.
Same prompts, same packing, same forward passes, so the difference isolates the
sentence rather than the row.

`output/v58_unseen_sentence_tokens.json`:

| token set (same 2,438 rows) | tokens | loss | perplexity |
| --- | --- | --- | --- |
| inside a **seen** sentence | 27,064 | 0.1880 | **1.2068** |
| inside a **withheld** sentence | 32,855 | 0.4188 | **1.5202** |
| difference | | **+0.2309 nats** | **1.260×** |

**The dilution was hiding most of the effect.** The diluted tier-3 gap over tier 1
is +0.084 nats; the controlled, token-level gap is **+0.231 nats — 2.8× larger**.
Measured against the familiar tokens sitting in the very same responses,
perplexity on a sentence the model has never seen is **1.52 against 1.21**.

This is the number that answers the question v57 raised. The model can emit every
word of every withheld sentence — coverage is 1.0000 — and it is still
substantially worse at them than at the sentences it was trained on, in the same
rows, under the same prompts. Its fluency is more sentence-recall than
composition, and that is now a measurement rather than a suspicion.

Note the withheld tokens outnumber the seen ones (32,855 vs 27,064) in these
rows: the selection rule caps a sentence by how many *rows* it appears in, not by
its length, and the sentences it admits are disproportionately long composites
like `"Recommended path: Check the traceback first, then we can isolate the
failing function."` The two sets are still disjoint and drawn from the same rows,
which is what the comparison needs, but they are not length-matched.

### The thinking-core ablation

V57's non-claims include: *"That the thinking core contributes anything to text
quality. It is present and trained, and its cycle count is reported. No ablation
has been run against a model without it on this corpus."*

`--arm ablation` runs that ablation. `--no_thinking_core` was already a flag on
the v57 trainer; what was missing was a matched pair. `compare()` refuses to
report unless the two arms agree on steps, batch size, sequence length, learning
rate, both seeds, and the withheld sentence set, so "matched" is checked rather
than asserted.

`output/v58_thinking_core_ablation.json`, `matched: true`. Both arms: same split,
same 30 withheld sentences, same 1,000 steps, same batch, same learning rate,
same seeds. The thinking core is 62,661 parameters, 2.0% of the model.

| tier | full (core on) | ablation (core off) | delta, nats |
| --- | --- | --- | --- |
| `tier1_seen_response` | 0.232862 | 0.233454 | **+0.00059** |
| `tier2_unseen_response` | 0.253957 | 0.254798 | **+0.00084** |
| `tier3_unseen_sentence` | 0.316511 | 0.313340 | **−0.00317** |

And on the token-level measurement, run on both checkpoints:

| | full | ablation |
| --- | --- | --- |
| seen-sentence tokens | 0.1880 | 0.1887 |
| withheld-sentence tokens | 0.4188 | 0.4126 |
| unseen − seen | **+0.2309** | **+0.2238** |

**No measurable effect on text quality on this corpus.** Every delta is between
0.0006 and 0.007 nats — two orders of magnitude below the +0.23 nat effect these
same measurements resolve cleanly — and **the sign is not consistent across
tiers**: the core is marginally ahead on tiers 1 and 2 and marginally behind on
tier 3 and on the token-level gap. Sign inconsistency at that magnitude is the
signature of noise, not of a small effect.

This retires v57's non-claim by answering it, and the answer is negative: on
120,000 rows of templated dialogue at a 1,000-step budget, the recursive thinking
core neither helps nor hurts the text. It does not generalise past that. The
corpus has no reasoning to do, one seed was run per arm, and no seed-to-seed
spread has been measured — so this bounds the effect on *this* task rather than
establishing that the mechanism is inert.

**The ablation arm is also an independent replication.** Two separately
initialised and separately trained models put the withheld-sentence penalty at
+0.2309 and +0.2238 nats. The headline effect does not rest on one training run.

The two arms' wall-clock times in the receipt (1,550s and 2,847s) are **not** a
speed comparison. They ran under different machine load; nothing here measures
the thinking core's cost.

## What this does not prove

* **That the model generalises to language.** See "what tier 3 is not" above.
  Every tier is drawn from one 21MB templated corpus.
* **That the tier-3 number is a property of the architecture.** It is one model,
  one corpus, one training budget, one seed.
* **That the withheld-sentence choice is neutral.** 30 sentences were withheld
  under a deterministic rule; a different seed picks a different 30 and would
  give a different tier-3 number. No spread over split seeds has been measured.
* **That the ablation settles what the thinking core is for.** It bounds the
  effect on *this* task: no measurable difference on templated dialogue with no
  reasoning in it. One seed per arm, one corpus, one budget, and the arms differ
  by 62,661 parameters (2.0%) as well as by the mechanism. The seed-to-seed
  spread of this setup is unmeasured, so "no measurable effect" means the effect
  is below a noise floor that has not itself been quantified — not that it is
  zero. A negative result on a corpus with nothing to reason about says little
  about a corpus that has something.
* **That the thinking core is free.** Nothing here measures its cost. The two
  arms' wall-clock times ran under different machine load and are not comparable.

## Promotion gates

Everything v57's gate list required still applies and is still unmet. V58
retires exactly one item from it — "held-out perplexity on text genuinely
disjoint from training, including unseen response templates" — and only for this
corpus. A v58 descendant intended for a product surface would still need a
corpus with measured diversity beyond 292 word types, rubric or human evaluation
of reply quality, safety evaluation of generated text, latency and throughput on
the target hardware, and source/package parity.
