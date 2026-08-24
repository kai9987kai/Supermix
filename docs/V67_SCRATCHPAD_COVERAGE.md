# Supermix v67 — the scratchpad format transfers, and average failed for a reason I caused

V66 showed a scratchpad took exact arithmetic from 0% to 55%, and left two task
types at 0% purely because they were absent from its corpus. V67 adds them.

## The result

Identical benchmark, identical scorer, 120 novel problems:

| task | v66 | v67 |
| --- | --- | --- |
| `algebra_one_step` | 0% *(absent from corpus)* | **66.7%** |
| `word_problem` | 0% *(absent from corpus)* | **41.7%** |
| `percent` | 65% | 62.5% |
| `arithmetic` | 55% | 41.7% |
| `average` | 0% | **0%** |
| **overall** | **24%** | **42.5%** |
| median relative error | 0.179 | **0.004** |
| within 10% of truth | 44% | **85%** |

**The format transfers.** Two task types that had never been trained went from
absent to 66.7% and 41.7% by writing their working the same way. Median relative
error fell 45x: the model is now typically within 0.4% of the true answer, and
85% of answers land within 10%.

`arithmetic` regressed from 55% to 41.7% and `percent` slipped slightly. The
corpus grew from four task types to six at a fixed 300,000 rows and a fixed
12,000 steps, so each task got less data and less capacity. That is the expected
cost of breadth at a fixed budget, and it is the same trade v62 paid on prose.

## Why average is still zero, and what I got wrong

I predicted average failed in v66 because the corpus rounded to four decimals
while the benchmark compared to a relative tolerance of the same size, making the
task unwinnable. That fix went in -- six decimals now -- and **average is still
0%**. The hypothesis was wrong, or at least not sufficient.

Generating averages and reading the working shows two real causes:

**1. A coverage bug I introduced.** `_scratchpad_average` emits `rng.choice([4, 5])`
values; the benchmark's `_average` uses `rng.choice([4, 5, 6])`. Every six-number
problem is out of distribution, and the model does the only thing it can:

    prompt: 28, 17, 62, 43, 23, 16          (six numbers)
    reply:  sum: 28 then 63 then 111 then 141 then 158, total 158, divide by 5

It truncates to five terms and divides by five. Roughly a third of the task fails
by construction, and the fault is the generator's, not the model's.

**2. Error accumulation over a long chain.** On four and five value problems the
running sum drifts:

    prompt: 28, 70, 65, 85                  truth 62.0
    reply:  sum: 28 then 98 then 165 then 252, ... total 63.0
    correct chain:   28,     98,     163,    248

Every partial sum must be right for the total to be right, and a single slip
poisons everything after it. A two-step place-value decomposition has one place
to go wrong; a six-term running sum has five. That is why average is structurally
the hardest task here even though it is arithmetically the simplest.

Both causes are now identified rather than suspected. Neither is fixed.

## What this does not prove

* **That the model does arithmetic reliably.** 42.5% overall is a large change
  from 24% and still wrong more often than right.
* **That breadth is free.** It cost `arithmetic` 13 points. Whether more steps or
  more rows would recover that is untested.
* **That fixing the average generator would fix average.** It would remove the
  coverage bug, which is a third of the failures. Error accumulation is the other
  part and would remain.
* **That the rounding fix did nothing.** It corrected a genuine defect -- the
  corpus was stating answers the benchmark scored wrong -- but it did not move
  the number, so its effect is unmeasured rather than zero.
* **That six task types is problem solving.** It is small-operand arithmetic in
  fixed phrasings, and every problem comes from a generator.
