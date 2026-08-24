# Supermix v62 — breadth bought, coherence not

V61 showed this system is **data-limited, not capacity-limited**: 3.18x the
parameters moved held-out loss by less than the noise floor. V62 acted on that,
replacing the corpus rather than the architecture and holding the model
identical, so any change is attributable to the data.

The corpus work succeeded. The model did not become usable. Both halves are
measured below, and the second is the more useful result.

## The blend

| | v60 corpus | v62 blend |
| --- | --- | --- |
| rows | 96,227 | **239,063** |
| word types | 5,280 | **40,810** |
| domains | 1 | **11, capped for balance** |

Balance is by explicit cap because **row count is not diversity**, and this
repo's own bundle proves it: `supermix_plus_v27_500k` has 509,126 rows and 3,433
word types; `hybrid_v6_live_knowledge` has 182,034 rows and **251** types,
thinner than the 292-type corpus v58 named as its binding constraint. The
vocabulary backbone is `book_extracts_public_domain` at 25,889 types.

```bash
python source/build_english_math_dataset.py --output datasets/v62/english_math_40k.jsonl --target 40000
python source/build_v62_corpus.py
python source/train_mimomix_generalisation.py --steps 2000 --run_name v62_multidomain \
  --output_dir output/v62_multidomain --corpus_jsonl datasets/v62/v62_blend.jsonl \
  --min_response_characters 1 --max_vocab 16384
```

### A filter that deleted three quarters of the maths

The blend first produced 7,950 maths rows against a 30,000 cap. The cause was an
8-character minimum response length inherited from `load_chat_pairs`, where a
short reply meant a truncation artifact. Applied to arithmetic it removed
**73.5%** of rows, because `"79"`, `"9/14"` and `"59.4"` are complete answers.

Fixed in two places -- per-domain thresholds in the blender, and
`--min_response_characters` on the trainer -- and pinned by test. A filter
correct for one domain silently destroyed another, which is the kind of failure
that shows up as a capability gap rather than an error.

## What was bought: vocabulary reachability

A word outside the vocabulary encodes to `<unk>` and **can never be generated at
any quality**, so reachability is a hard ceiling on what a model can say about a
subject. Unlike perplexity it is comparable across models, because it is a
property of the tokenizer rather than the weights.

| domain | v60_control_2000 | v62 |
| --- | --- | --- |
| scripture | 0.6657 | **0.9641** |
| literary_study | 0.7376 | **0.9989** |
| writing | 0.7175 | **0.9654** |
| vocabulary | 0.6525 | **0.8642** |
| logic | 0.8721 | **0.9987** |
| conversation | 0.8803 | **0.9962** |
| maths | 0.8920 | **0.9374** |
| science | 0.7851 | **0.8648** |
| creativity | 0.9856 | **0.9998** |
| **coding** | **1.0000** | **0.6900** |

Nine domains rise substantially. **Coding regresses hard** — capping it at 380
rows cost the one domain v60 was specialised in. That is a real loss, not a
rounding error, and it was the direct consequence of a cap chosen for balance.

Coverage overall is **0.9914, not the 1.0000** every prior version reported. On
real English that figure is unreachable at any sane vocabulary: 32,774 tokens
only reaches 0.9966 and costs 8.42 s/step against 5.20. The standard was met
before because the corpus had 292 word types.

## What was not bought: coherence

2,000 steps covers **0.178 of one epoch** (22,952,960 corpus tokens; one epoch is
11,208 steps). A 6,000-step continuation took **17.5 hours** and reached 0.71
epochs. Dev loss fell 0.8919 → **0.7531**, and every domain improved. The model
still cannot hold a conversation.

| prompt | reply (greedy, 8,000 steps) |
| --- | --- |
| tell me a short story about the sea | `Let me work through this step by step. [analytical-set3 worked solution] Area = 34 x 3 =. Perimeter = 2 x (30 + 3) = 62.` |
| hello, how are you | *(the same worked-solution text)* |
| explain why the sky is blue | `.8: -` *(sampled)* |
| describe a storm at sea | `.` *(sampled)* |
| what is 17 + 25 | `31` / `13` *(sampled; the answer is 42)* |

Greedy decoding collapses onto the single most probable template regardless of
prompt. Sampling occasionally produces good prose -- *"The moment hung in the air
like a held breath"* -- but a five-prompt sweep at the same temperature returned
mostly degenerate output, so that sentence is a lucky draw and not a capability.
Arithmetic has learned the *format* (a bare number) and not the *function*.

## The finding: what it learns tracks whether the text is generated

| domain | word types | loss @2k | loss @8k | perplexity @8k |
| --- | --- | --- | --- | --- |
| creativity | 1,215 | 0.3092 | **0.2328** | **1.26** |
| logic | 3,471 | 0.3275 | **0.2366** | **1.27** |
| conversation | 3,806 | 0.3973 | **0.3314** | **1.39** |
| language | 345 | 2.4947 | 0.9017 | 2.46 |
| literary_study | 811 | 1.5398 | 1.2846 | 3.61 |
| maths | 5,339 | 1.7141 | 1.5179 | 4.56 |
| scripture | 7,912 | 2.9701 | 2.5391 | 12.67 |
| science | 654 | 4.7193 | 2.7962 | 16.38 |
| writing | 25,889 | 3.1893 | 2.9259 | **18.65** |
| vocabulary | 8,112 | 3.3147 | 2.9491 | **19.09** |

The split is near-total and it is **not** a split by domain difficulty. The three
domains the model masters -- creativity, logic, conversation, all at perplexity
under 1.4 -- are the three assembled from generator templates. The domains it
fails -- writing, vocabulary, scripture, at perplexity 12 to 19 -- are the ones
containing real human text.

**At this scale the model learns templates, not language.** That reframes the
whole v57 line: its headline perplexity of 1.27 was not a small model doing well,
it was a small model memorising 192 sentences, and v62 reproduces exactly that
number (1.26) on exactly the templated portion of a far larger corpus while
scoring 18.65 on the literary portion beside it.

## What this does not prove

* **That more training would not fix it.** 0.71 epochs is not converged; dev loss
  was still falling when the run ended. It does show that quadrupling training
  moved writing from 3.19 to 2.93 nats -- a 8% improvement for 17.5 hours -- so
  the remaining distance to coherence is large.
* **That the architecture is at fault.** Nothing here separates model scale,
  training budget and tokenizer from one another. A 6.1M-parameter word-level
  model on 16.4M tokens is far below where language models become coherent, and
  no experiment here identifies which constraint binds first.
* **That the corpus is good.** It was measured for diversity, not audited for
  correctness, licensing or contamination. `maths` means "generated arithmetic";
  a low loss on it would mean predicting those strings, not doing arithmetic --
  and the model demonstrably cannot do the arithmetic.
* **That the domain labels are competences.** They record which file a row came
  from.
* **That reachability implies capability.** It is a ceiling, not a floor. v62 can
  represent 96.5% of literary tokens and still models them at perplexity 18.65.

## What to use

**`v60_control_2000` remains the checkpoint to serve.** It is narrow -- one
register, and it answers anything outside it with a generic line -- but within
that register it is fluent. v62 is broader and coherent nowhere.

The honest trade this version establishes: **at CPU scale you choose breadth or
coherence, and the templated-versus-real split above is where the boundary sits.**
The reusable products are the blend, the per-domain evaluation, the reachability
measure, the two bug fixes, and `--init_from`, which now makes a multi-day
continuation possible without discarding finished compute.
