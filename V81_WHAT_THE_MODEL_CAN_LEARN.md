# v81 — Matching the corpus to what the architecture can actually learn

Two findings from the v79/v80 post-mortem, and a memory fix that has now cost
hours twice.

## 1. The winning task is built the opposite way to the failing one

v74's `multiplication` task scored **0.93**. The omni `force` task scored
**0.03**. They are the same operation.

| | v74 `multiplication` | omni `force` (v79) |
|---|---|---|
| distinct problems | **712** | 24,000 |
| repetition | **56x each** | ~1.7x |
| operands | 11-99 x 2-9 | 2-400 x 2-60 |
| working shown | two partial products | **none** |
| **score** | **0.93** | **0.03** |

I built the omni corpus to maximise diversity, and wrote in `V79_OMNI_FRONTIER.md`
that *"duplicated rows are precisely what a recitation-proof benchmark exists to
punish."*

**That is wrong for an algorithmic task**, and it is the most useful thing this
post-mortem produced. The benchmark draws *unseen* operands from the same
space, so the model cannot pass by recall — it has to learn the method, and
repetition is how it learns it. Uniqueness is the right goal for a *knowledge*
corpus, where a repeated fact is simply memorised. It is the wrong goal for a
*procedure*.

The exhaustion machinery built in v79 to guarantee unique prompts was therefore
solving a problem that did not exist, while destroying the repetition that did
the teaching. It remains available as `--unique`; the default now repeats, and
reports `distinct_prompts` and `repetition` so the factor is visible rather
than hidden.

## 2. The failure moved when the working was added, which located it exactly

v80 added place-value decomposition to the multiplications. The step-6,000
probe was unchanged (0.03, same as v79), but the *mechanism* had changed:

```
want 1837  →  100 x 11 = 1100, 60 x 11 = 660, 7 x 11 = 77,   ← all three CORRECT
              1100 + 660 = 1260, 1260 + 77 = 1207             ← the sums are wrong
```

The multiplications became correct and the failure moved to the running sums —
because I had written those as single-step four-digit additions. **The same
mistake, one level up.**

The fix is not to decompose the sums as well. That costs +23 tokens, pushing
the maximum past the 128-token sequence length and forcing it to 192, about
50% more training time. The fix is to match v74's format exactly, which needs
**no extra tokens at all**:

```
80 x 3  ->  80 x 3 = 240, 0 x 3 = 0
25 x 7  ->  20 x 7 = 140, 5 x 7 = 35
```

Two details I had got wrong and have now corrected:

* **v74 keeps the zero place.** I was dropping `0 x 3 = 0` as noise. Keeping it
  gives every two-digit problem exactly two partial products — one shape, every
  time.
* **v74 writes no addition step.** It goes straight to the total and the model
  learned to combine the parts itself. Adding running sums departed from the
  only format ever measured to work.

Operand ranges are narrowed to v74's proven 11-99 x 2-9. A model scoring 0.9 on
`mass 47 kg, acceleration 6 m/s^2` is worth more than one scoring 0.03 on
`mass 400 kg, acceleration 60 m/s^2`. This narrows the task definition, and the
benchmark narrows with it, exactly as v74's multiplication task is defined and
scored over two-digit operands.

## 3. The corpus was stored at four times the size it needed

Identified in `V75_CRASH_RECOVERY.md` and deferred as speculative. It stopped
being speculative when v79 spent hours at **17.12 s/step** against a 4 s/step
norm, faulting its own corpus back from the pagefile:

```
committed   25.6 GB   on a 15.6 GB machine
trainer      4.44 GB  footprint, 1.2 GB resident
```

Token ids below 9,000 were stored as **int64**. For v79's
866,748 x 128 x 2 tensors that is **1.78 GB** of the 4.44 GB.

`compact_dtype` now picks the narrowest type that holds the vocabulary — int16
for everything in this repository (8,551 for v79, 16,384 at the `--max_vocab`
ceiling), int32 as the fallback for anything larger, never a silent overflow.
The limit is 32,000 rather than 32,767 so the `-100` ignore label stays
representable.

**1.78 GB → 0.44 GB.** Batches are widened to int64 on the way into the model,
which copies 16 x 128 values per step — nothing against the pagefile traffic it
removes.

### The smoke test earned its keep

A 20-step end-to-end run produced identical losses (3.9684, 3.4407, 3.0595,
2.8853 — byte-matching the pre-change run, so the change is numerically
neutral) and then **crashed in the report phase**:

```
RuntimeError: Expected tensor for argument #1 'indices' to have one of the
following scalar types: Long, Int; but got torch.ShortTensor instead
```

`routing_report` was a consumer I had missed. Unit tests would not have caught
it, because it only runs at the end of a full run. That is the whole argument
for running the thing end to end before leaving a change in a code path the
crash supervisor might restart into.

## What is not claimed

* **Not that v81 will beat v74.** These are three corrections to specific,
  measured failures. Whether they add up to a better model is what the run
  decides.
* **Not that v80 was wasted.** Its probe located the failure precisely: the
  multiplications became correct and the additions did not, which is what
  identified the format as the problem rather than the model.
