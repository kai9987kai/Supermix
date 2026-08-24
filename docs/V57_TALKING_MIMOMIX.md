# Supermix v57 — the talking MiMoMix

## What v57 is

V53 built a complete decoder-only language model and never trained it on
language. Its own architecture document says so plainly: the API backends are
randomly initialised and *"their text is noise by design."* V57 is that missing
half — the same `mimomix_core.MiMoMixModel`, unchanged, trained on real dialogue
until it generates text, plus the tokenizer, corpus, checkpoint format and chat
interface needed to serve it.

No file in `mimomix_core.py` was modified. V57 is additive: a tokenizer, a corpus
loader, a training script, and a web app.

| surface | v57 contract |
| --- | --- |
| tokenizer and corpus | `source/mimomix_text.py` |
| training and measurement | `source/train_mimomix_talk.py` |
| chat interface | `source/mimomix_talk_web_app.py` |
| model | `mimomix_core.MiMoMixModel` — unchanged from v53 |
| checkpoint schema | `supermix-v57-talk-checkpoint-v1` |
| receipt schema | `supermix-v57-talk-benchmark-v1` |
| source of truth | `source/` |
| compatibility mirror | none — research line, same rule as v53 and v56 |
| tests | `test_mimomix_text.py`, `test_mimomix_talk_web_app.py` |

## What it is, stated before any number

**A small domain-specific chat model.** It generates fluent short replies in one
register — coding-assistant small talk — and nothing else. It is not
knowledgeable, not instruction-following in any general sense, and not comparable
to a modern assistant. That is a property of the only corpus available locally,
not a claim about the architecture.

## The corpus is the ceiling

`databases/llm_chat.db` holds 120,000 `(user_text, response_text)` pairs, 21M
characters. Measured, not assumed:

| property | value |
| --- | --- |
| pairs | 120,000 (117,600 train / 2,400 validation after the split) |
| characters | 21.0M |
| word tokens | 4.62M |
| **distinct word types** | **292** (582 tokenizer entries: both spacing forms, plus specials) |
| distinct responses | 37,543 of 120,000 |
| distinct user texts | 71,897 of 120,000 |

Two consequences follow directly and are recorded in the receipt:

1. **A word outside the vocabulary can never be generated.** With 582 types in
   the shipped tokenizer — 292 distinct words, admitted in both spacing forms,
   plus the specials — the model's entire expressible language is that list.
   `WordTokenizer.vocabulary_report` reports coverage on held-out text (0.976),
   and the chat interface warns the user when their own message contains words it
   had to replace with `<unk>`.
2. **Held-out perplexity measures fit to a template distribution, not
   generalisation to language.** Rows are split disjointly, but only 37,543 of the
   120,000 responses are distinct, so a validation row's response text may still
   appear in training. The receipt names the metric accordingly. A perplexity near
   1 on this corpus is a statement about the corpus, not about fluency in English.

## The tokenizer

`WordTokenizer` is a whitespace-preserving word tokenizer, not BPE. Every token
carries its own leading whitespace, so `"".join(tokens)` reconstructs the input
byte for byte and decoding needs no spacing heuristics —
`test_encoding_round_trips_exactly` and `test_round_trip_preserves_whitespace_runs`
pin it, including runs of multiple spaces.

Subword tokenisation would buy nothing on data with 292 word types, and
byte-level would multiply sequence length roughly fivefold for the same content.
The choice is a consequence of the measured corpus, not a preference.

## Training

Standard next-token prediction with the full v53 objective: language-model loss,
the multi-token-prediction depth losses, the auxiliary-loss-free MoE bias rule
stepped once per optimizer step, the router z-loss and balance term, and the
thinking core's ponder and consistency costs.

**Prompt tokens are masked out of the loss** with `-100`, which
`cross_entropy` ignores by default, so the model is trained to *produce* replies
rather than to reproduce the user's own words.
`test_prompt_tokens_are_masked_out_of_the_loss` checks that every unmasked label
equals its input and that some labels are masked.

Turns are packed contiguously into fixed-length blocks rather than padded, so no
compute is spent on padding — and, incidentally, the model sees turns following
other turns, which is why the chat interface's bounded history is only mildly out
of distribution rather than wholly so.

## Decoding

Two paths, offered because they answer different questions.

**Greedy** runs `mimomix_decoding.speculative_generate`, which uses the MTP
depths as a draft model. For greedy decoding this is *provably token-identical*
to one-at-a-time generation, so it changes only the cost, never the text —
`test_speculative_decoding_matches_greedy` requires the two to agree. It is
deterministic: the same prompt always gives the same reply. The interface reports
the measured acceptance length and trunk-forward count.

**Sample** applies temperature and nucleus filtering. It is explicitly labelled
`identical_to_plain_greedy: false`. It recomputes the prefix per token rather
than reusing the KV cache — deliberately, because the cache path exists to
support the greedy-equivalence guarantee and reusing it here would risk silently
breaking that guarantee for a feature that does not need it.

## Trust boundary

The prompt is data. It is tokenized and fed to the model; nothing in it selects a
checkpoint, a decoding mode, a temperature, or a token budget. Those come from
typed request fields, and every numeric field is **clamped** rather than trusted.
`test_prompt_text_cannot_change_the_decoding_mode` fires an instruction-carrying
payload and requires the decoding mode to be unchanged.

The service binds `127.0.0.1` and refuses to start without a
`supermix-v57-talk-checkpoint-v1` file. The checkpoint carries its own vocabulary:
weights loaded against the wrong vocabulary would decode to confident nonsense
rather than fail, so the tokenizer travels with the tensors and
`test_checkpoint_carries_the_tokenizer` pins it.

## Measured results

`output/v57_talk_v2/talk_results.json`. 3,000 steps, batch 16, sequence length
128 — about 6.1M tokens, roughly 1.2 passes over the corpus, 4,283 seconds on
CPU.

| measurement | value |
| --- | --- |
| parameters | 3,076,521 total / 1,292,829 active per token |
| vocabulary | 582 types, held-out coverage **1.0000** |
| validation loss | **0.2351** (0.339 bits/token) |
| perplexity | **1.27** (uniform baseline 6.37) |
| MTP acceptance length | **2.429** — 7 trunk forwards versus 17 for greedy |
| speculative output identical to greedy | **true** |
| routing entropy | 1.000 normalised, **0 starved experts** |

Generated replies, greedy, from the receipt:

| prompt | reply |
| --- | --- |
| hello | Hello. Tell me what you need and I will do my best to help. |
| can you help me with tests | Understood. Yes. I can add unit tests and integration tests based on your code. |
| why is my script failing | Check the traceback first, then we can isolate the failing function. |
| what is your name | I am your local coding assistant for this project. |
| write a unit test for login | Okay. I can do that. Share expected behavior and edge cases for login. |

Serving throughput, measured in the browser: 55–75 tokens/second at 218–306 ms
per reply, CPU, dev server.

### The vocabulary bug, and why the perplexity went up

The first checkpoint (`output/v57_talk/`) reported perplexity **1.24** and
coverage 0.9759, and every reply began with a stray `". "`. Both had one cause.

Tokens carry their own leading whitespace, so `"Got"` and `" Got"` are different
strings. The vocabulary was built from `user + " " + assistant` **concatenated**,
which only ever showed a reply's first word with a leading space. The
sentence-initial form was therefore missing, and **17,709 of ~19,600 replies
tokenized with `<unk>` as their first token.**

The lost text was the smaller problem. The larger one is that `<unk>` became a
trivially predictable target at exactly the position after `<assistant>`, so the
reported perplexity was **flattered by the bug**. Fixing it moved perplexity from
1.24 to **1.27** — a worse number that is the honest one.

| | first checkpoint | current |
| --- | --- | --- |
| vocabulary | 340 | 582 |
| most common first token of a reply | `<unk>` (17,709 times) | `Sure`, `Understood`, `Okay`, `Got` |
| held-out coverage | 0.9759 | **1.0000** |
| perplexity | 1.24 (flattered) | **1.27** (honest) |

`WordTokenizer.build` now admits every kept token in both forms, the trainer
counts fields separately rather than concatenating them, and
`test_a_word_is_representable_at_a_sentence_start` is a regression test for it.

## What this does **not** prove

* **That the model knows anything.** It was trained on templated dialogue built
  from 292 distinct words. There is no world knowledge in that corpus to learn, so
  there is none in the model. Fluent phrasing is not evidence of content.
* **That low perplexity means good language.** On a corpus with 37,543 distinct
  responses, a small model can fit the template distribution closely. The number
  describes the corpus at least as much as the model.
* **That this is comparable to any modern language model.** It is roughly three
  million parameters trained for under two hours on a CPU, on 21MB of templated
  text. Nothing here should be read as a comparison to systems trained on
  trillions of tokens.
* **That the architecture is validated as a language model.** V57 shows the v53
  stack trains and generates on one small corpus. Scaling behaviour, long-context
  quality, and the value of the hybrid attention or MoE at any real scale are all
  unmeasured here.
* **That the thinking core contributes anything to text quality.** It is present
  and trained, and its cycle count is reported. No ablation has been run against a
  model without it on this corpus.
* **That the chat interface makes it an assistant.** It answers in the register of
  its corpus. It cannot follow instructions it has never seen phrased.

## Promotion gates

A v57 descendant intended for any product surface would need, at minimum: a
corpus with measured diversity rather than 292 word types; held-out perplexity on
text genuinely disjoint from training, including unseen response templates;
human or rubric evaluation of reply quality, not just perplexity; safety
evaluation of generated text, which this model has had none of; latency and
throughput on the target hardware; and source/package parity. None of that has
been run, because v57 does not claim it.
