# v78 — The training tracker could not see the training

## What was found

`source/training_monitor_gui.py` — 5,185 lines of Tkinter, with a launcher
batch file — is this project's training tracker. Its parser expects:

```
[train] step=1200 loss=1.83 lr=3e-05
```

`train_mimomix_generalisation.py`, the only trainer used from v58 through v74,
emits:

```
step 12000/18000  train 0.0839  dev 0.0926  ppl 1.10  acc 0.70  2092s
```

Measured against the v74 log: **13 step lines, 0 matched.**

The tracker only understands `qwen_supermix_pipeline.py`, the older LoRA line.
Every generalisation run since v58 has been invisible to it. That is why the
v74 run was tracked by hand — tailing a log, reading the last step line, and
doing arithmetic.

## The mistake that made this worth fixing

Doing it by hand went wrong in one repeatable way: **quoting the fastest recent
interval as the rate.**

Within the single v74 run, observed per-step time ranged **1.98 to 5.73
s/step** — a 2.9x spread, caused by accuracy probes, checkpoint writes, and the
machine paging. An ETA extrapolated from the fastest stretch is wrong by hours,
and it is wrong in the direction that sounds best.

`source/training_tracker.py` is built around not doing that:

* the **working rate** is the *median* of intervals containing no probe, so one
  paging episode cannot drag the estimate that hours are based on
* **probe intervals are identified and priced separately** — a probe generates
  100 replies, so its interval is not a step rate. Remaining probes are costed
  into the estimate using the observed per-probe overhead and the inferred
  cadence
* the ETA is a **range** derived from the observed spread, never a single
  confident number
* when recent pace disagrees materially with the run average, **the
  disagreement is stated rather than resolved silently in either direction**

## It refuses to extrapolate from a dead run

Pointed at the log of the leg that segfaulted:

```
v74  [stalled]
  step        11,500 / 18,000 (63.9%)
  rate        2.76 s/step  (observed 2.09-5.73)
  eta         unknown - no new step for 14h21m
  note        recent pace is 79% slower than the run average
              (4.92 vs 2.76 s/step); the estimate uses the average
  accuracy    3,000:0.04  6,000:0.22  9,000:0.61
```

Reporting "63.9% done, about 5 hours to go" for a process that died fourteen
hours ago is the worst available output, because it reads as normal. The run is
reported as **stalled**, with the age of the silence, and no estimate.

**The `note` line is the finding.** That 79% slowdown is the paging degradation
immediately before the segfault. The tracker surfaces it automatically from
data that was already in the log — the warning was there to be read, and
reading logs by eye did not catch it.

## What a run survived

A log says where a run is. `train_supervised.py`'s journal says what it lived
through, and that appears in no log line because each leg writes its own log:

```
v74_broad  [supervised]  no restarts
  leg 1  from 11,500  4h53m  exit 0 - completed
```

Exit codes are preserved verbatim — 139 is SIGSEGV, and losing it would hide
why a leg ended. A corrupt or unrecognised journal is ignored rather than
raising, because a monitor that crashes on bad input is not a monitor.

## Usage

```bash
python source/training_tracker.py --log_dir logs --output_dir output
python source/training_tracker.py run.log --watch 60
python source/training_tracker.py --log_dir logs --json
```

`--json` exists so this can answer "how far along is it?" to something other
than a human — which the Tkinter GUI structurally cannot do, and which is what
was actually needed every time the question was asked during v74.

36 tests. Most pin the refusals: no ETA from a stalled log, no ETA without
measured intervals, no rate from a resume boundary where the clock restarts, no
verdict from a corrupt journal.

## What was not done

**The Tkinter GUI was not taught this format.** It is 5,185 lines, Windows-only,
and structurally cannot answer a question asked from a terminal or a script.
The headless tracker is the capability that was missing, not a second window.
Bridging the GUI to `parse_log` is a small change if a graphical view is
wanted; it just was not the gap.
