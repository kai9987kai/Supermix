# Supermix v60 — the ladder on a corpus with real language

V58's promotion-gate list opens with an item it could not retire: *"a corpus with
measured diversity beyond 292 word types."* Every v57/v58 number came from
`databases/llm_chat.db` — 120,000 rows built from **192 distinct sentences** and
**292 word types**. V58 said so plainly and listed "that the model generalises to
language" among the things it does not prove.

V60 retires that gate, and the answer changes the headline.

## The corpus was already on disk

`artifacts/qwen_supermix_enhanced_v29_full_20260320_190817/prepared_train_pairs.jsonl`
(44 MB) was read by no trainer. It is already in the `(user, assistant)` shape the
MiMoMix pipeline consumes:

| | `llm_chat.db` (v57/v58) | v29 pipeline corpus (v60) |
| --- | --- | --- |
| rows | 120,000 | 96,227 |
| word types | **292** | **5,280** |
| distinct assistant sentences | **192** | **39,235** |
| distinct responses | 37,543 | 56,881 |
| sources | 1 template generator | 6 (reasoning, creative, coding, world events, anchors) |

Trained vocabulary rises from **582** to **10,538** types at coverage 1.0000,
still inside `WordTokenizer`'s 16,384 cap.

```bash
python source/train_mimomix_generalisation.py --steps 1000 --run_name v60_diverse \
  --output_dir output/v60_diverse \
  --corpus_jsonl artifacts/qwen_supermix_enhanced_v29_full_20260320_190817/prepared_train_pairs.jsonl
python source/eval_mimomix_unseen_sentences.py --run_dir output/v58_full --run_dir output/v60_diverse
```

## The split machinery had to be fixed first, and the fix is backward-compatible

`verify_split` **raised** on this corpus: `283 tier-2 row(s) contain a sentence
absent from training` — 59% of tier 2. That was the verifier working correctly.
Tier 2 promises "novel response, *every sentence seen*"; carving dev and test out
of the pool can strand a sentence outside training, and on a templated corpus
where sentences recur ~310 times that essentially never happens. On real text it
is the common case.

The remedy was the one `verify_split` named in its own error message — *"they
belong in tier 3"* — applied where tiers are assigned rather than left as a crash.

| corpus | rerouted | tier sizes | `verify_split` |
| --- | --- | --- | --- |
| v29 pipeline (diverse) | 283 | 307 / 194 / 1,083 | now passes |
| `llm_chat.db` (v58) | **0** | 1,883 / 468 / 2,438 | passes |

Zero rows move on v58's corpus and the tier sizes reproduce its published split
exactly, so the change cannot have rewritten the result it is being compared to.

## The ladder, on real language

1,000 steps, same architecture, 91,344 training rows, **640 withheld sentences**
(v58: 30), 4,988,073 parameters:

| tier | rows | loss | perplexity |
| --- | --- | --- | --- |
| `tier1_seen_response` | 1,007 | 0.2260 | 1.2536 |
| `tier2_unseen_response` | 422 | **0.2180** | 1.2436 |
| `tier3_unseen_sentence` | 2,394 | 0.3135 | 1.3682 |

**The ladder is no longer monotonic.** V58's tier-1 → tier-2 step cost +0.0211
nats; here it is **−0.0080** — novel responses are marginally *easier* than
responses seen verbatim. The tier-2 → tier-3 step, meanwhile, grew from +0.0626
to **+0.0955**.

So recombination is free on real text and the whole cost has moved to unseen
sentences. That is a coherent story — a model trained on 37,958 sentences has
learned to arrange language rather than to recall arrangements — but the
−0.0080 step is small, unreplicated, and its sign is opposite to v58's. No
seed-to-seed spread has been measured for either, so treat that step as
"indistinguishable from zero", not as an inversion.

## The headline: v58's sharpest number was an artifact of corpus poverty

V58's strongest claim came from scoring withheld-sentence tokens separately from
seen-sentence tokens *inside the same rows* — same prompts, same packing, same
forward passes. It reported **+0.2309 nats**, called it "2.8× the diluted gap",
and concluded the model "can emit every word of every withheld sentence and is
still markedly worse at them".

Run under one command on both checkpoints:

| token set | v58_full (292 types) | v60_diverse (10,538 types) |
| --- | --- | --- |
| inside a **seen** sentence | 0.1880 (ppl 1.2068) | 0.2945 (ppl 1.3424) |
| inside a **withheld** sentence | 0.4188 (ppl 1.5202) | 0.2988 (ppl 1.3482) |
| **unseen − seen** | **+0.2309** | **+0.0043** |
| perplexity ratio | 1.260× | **1.004×** |

**The effect is 53× smaller on real language.** Withholding a sentence from a
192-sentence corpus removes something the model can only memorise. Withholding
one from a 37,958-sentence corpus removes nothing it cannot rebuild from the
rest of the language.

The row-level and token-level measurements now point in *opposite* directions:
tier-3 rows cost +0.0955 nats, but their withheld sentences cost only +0.0043.
V58's method assumed the row-level gap was diluted evidence of a larger local
penalty; on this corpus the row-level gap is **22× larger** than the local
penalty, so whatever makes tier-3 rows harder is spread across the row rather
than concentrated at the unseen sentence. Tier-3 rows are selected for containing
a rare sentence, which plausibly selects unusual rows throughout — that is a
hypothesis this run does not test.

## What this does not prove

* **That diversity alone caused the collapse of the gap.** Three things changed
  together. The **share of the sentence inventory withheld** is the largest
  confound: v58 withheld 30 of 192 sentences (**15.6%**), v60 withheld 640 of
  37,958 (**1.7%**). A model stripped of a sixth of its language should struggle
  more than one stripped of a sixtieth, independent of diversity. Matching that
  fraction is the experiment that would separate the two, and it has not been run.
* **That the models are comparable.** v60 has 4,988,073 parameters against v58's
  3,076,521, almost entirely because the embedding grew with the vocabulary.
* **That the perplexities are comparable.** They are not, and no conclusion here
  rests on comparing them. Different vocabularies mean perplexity is measured
  over a different unit. Only the *within-model, within-row* gaps are compared,
  which is why the gap is the number reported.
* **That either result is stable.** One seed, one run, one split seed per corpus.
  V58 measured a second arm; v60 has no replicate.
* **That the model is good.** Dev loss 0.2852 at 1,000 steps on a 4.99M-parameter
  model. Nothing here evaluates reply quality, factuality, safety, or latency.
* **That the corpus is clean.** The v29 pipeline corpus was measured for
  diversity, not audited for correctness, licensing, or contamination. 5,280 word
  types is a diversity measurement, not a quality one.

## Retired from v58's gate list

> "a corpus with measured diversity beyond 292 word types"

Retired: 5,280 word types, 39,235 distinct sentences, trained vocabulary 10,538
at coverage 1.0000, with the ladder verified on it.

Still unmet, unchanged: rubric or human evaluation of reply quality, safety
evaluation of generated text, latency and throughput on target hardware, and
source/package parity.
