# v63 — the GPU run plan

Prepared, not executed. This is what to run, why that and not something bigger,
and what it is honestly expected to produce.

## The binding constraint is data, not compute

Before choosing hardware, the question is whether there is enough text to justify
it. Measured across the whole bundle:

| | |
| --- | --- |
| bundle rows | 1,279,038 |
| bundle characters | 542,673,047 |
| **approximate word tokens** | **~108,500,000** |
| v62 blend actually used | 22,952,960 |
| v62 model size | 6,111,657 parameters |

At the Chinchilla ratio of ~20 tokens per parameter, 108.5M tokens is
compute-optimal for a model of about **5.4M parameters**. **v62 is already
6.1M.** So a bigger model trained on this repo's data is not under-trained
because the GPU is missing; it is data-bound, and scaling parameters without
scaling text reproduces exactly the v61 result — capacity that specialises
internally and changes nothing externally.

**A GPU therefore buys speed, not capability, unless new data comes with it.**
That is the single most important sentence in this plan, and it means the run
below is worth doing while a much larger one is not.

## What speed alone is worth

Measured CPU throughput from the v62 continuation: 16,384,000 tokens in 62,428
seconds ≈ **262 tokens/second**. One epoch of the full bundle is therefore
~115 hours on this box, which is why v62 saw 0.71 epochs of a corpus that was
itself only a fifth of what is available.

A single modern GPU on a model this small is throughput-bound by kernel launches
rather than FLOPs, so the honest range is wide — roughly **50-200x** this CPU
once the batch is sized for the device. That puts one epoch of the full bundle at
**35 minutes to 2.3 hours**, and a genuine multi-epoch run in an afternoon rather
than a fortnight.

## Plan A (recommended): match the model to the data, train to convergence

Use everything on disk, keep the model near compute-optimal, and train until dev
loss stops falling rather than until the clock runs out.

* corpus: full bundle blend, all 15 files, domain-capped as in
  `build_v62_corpus.py` but with the caps raised roughly 5x
* model: 6-20M parameters. The upper end is deliberately past Chinchilla-optimal,
  because the goal is coherence rather than compute efficiency, and small models
  keep improving well past that ratio.
* budget: 3-5 epochs over ~108M tokens = 325-540M training tokens
* batch: 256-512 sequences, not 16. The current default is CPU-shaped and would
  leave a GPU almost idle.

```bash
python source/build_v62_corpus.py --output datasets/v63/v63_full.jsonl   # after raising caps
python source/train_mimomix_generalisation.py \
  --corpus_jsonl datasets/v63/v63_full.jsonl \
  --steps 40000 --batch_size 256 --eval_every 1000 \
  --max_vocab 32768 --min_response_characters 1 \
  --device cuda --amp bf16 --resident_corpus \
  --hidden_size 384 --n_layers 8 --n_heads 6 --n_routed_experts 16 \
  --run_name v63_gpu --output_dir output/v63_gpu
```

40,000 steps x 256 x 128 = **1.31B training tokens**, roughly 12 epochs. At the
conservative 50x figure that is about 9 hours; at 200x, under 3.

**Expected outcome, stated before the run:** coherent, simple English within the
corpus's register. The reference point is TinyStories, where 10-35M-parameter
models produce fluent text when the data is consistent. It will not be
knowledgeable, and it will not do arithmetic it has not seen — v62 learned the
*format* of arithmetic and not the function, and more of the same data will not
change that.

**What would refute it:** if per-domain perplexity on `writing` stays above ~5
after 3 epochs, the constraint is not budget and the model or tokenizer is at
fault instead.

## Plan B: bring new data

Only worth doing if Plan A converges and is still too narrow. 108M tokens is a
small corpus; the jump to genuine breadth needs 1-10B tokens of real text, which
this repo does not contain and which changes the licensing and provenance
question entirely. Nothing here should be pointed at scraped text without that
being decided deliberately.

## What is already ready

The training path is GPU-capable as of this version, and the changes were tested
rather than assumed:

* `device_utils.resolve_device` already supported `cuda`/`xpu`/`mps`/`dml`;
  `--device cuda` needed no work.
* **`--amp {off,bf16,fp16}`** — autocast on the forward pass, with a `GradScaler`
  for fp16 and none for bf16. Default `off`, so every published fp32 result is
  reproduced unchanged.
* **`--resident_corpus`** — keeps the packed corpus on the device instead of
  copying a batch across the bus every step. ~183 MB for the v62 blend, and it
  removes an overhead that dominates at these model sizes.
* **A real bug this found.** `SparseMoEFeedForward.forward` accumulated expert
  outputs into an fp32 buffer with `index_add_`, which requires exactly matching
  dtypes. Under autocast the experts return bf16 and the MoE path **raised on the
  first step**. Fixed in `mimomix_core.py` and mirrored in
  `mechanism_causality.py`. Without this, every mixed-precision run would have
  failed immediately.

Verification that the fix changed nothing in fp32: `mechanism_causality.py`
reproduces its published v59 numbers exactly — baseline 0.23159635, thinking core
+8.841404e-08 with 0 of 12,192 decisions changed, routing +4.160336e-02 — and 106
tests pass.

## What to fix before a long GPU run

* **Save optimizer state with the checkpoint.** `--init_from` restores weights
  only, so a continuation restarts AdamW's moments and the LR schedule. That cost
  v62 roughly 1,500 steps re-warming, visible as dev loss rising from 0.8919 to
  1.0036 before recovering. On a multi-day run this matters much more.
* **Checkpoint periodically.** The trainer holds `best_state` in memory and
  writes once at the end, so a crash 11 hours in loses everything.
* **Re-tune the LR for the larger batch.** Batch 16 to 256 is a 16x change;
  keeping the current learning rate would be a silent mistake.

## What this plan does not claim

* That a GPU makes this model good. It makes the experiment fast enough to run
  properly; the data ceiling above is unchanged by hardware.
* That the 50-200x range is measured. It is not — no GPU was available here, and
  it is an estimate from model size and batch shape. The CPU figure (262
  tokens/second) is measured.
* That coherence is guaranteed at 1.31B tokens. It is the expectation, with the
  refutation condition stated above so the run can fail honestly.
* That `--amp bf16` is validated on a GPU. It is validated as *executing
  correctly* on CPU, which is where the dtype bug surfaced. Numerical behaviour
  on an accelerator is untested.
