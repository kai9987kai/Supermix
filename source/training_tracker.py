"""Read a generalisation training run and say honestly where it is.

`training_monitor_gui.py` is a 5,185-line Tkinter monitor, and it cannot see
these runs at all. Its parser expects the LoRA pipeline's format::

    [train] step=1200 loss=1.83 lr=3e-05

while `train_mimomix_generalisation.py` -- the only trainer used from v58 to
v74 -- emits::

    step 12000/18000  train 0.0839  dev 0.0926  ppl 1.10  acc 0.70  2092s

Measured against the v74 log: **13 step lines, 0 matched.** So every "how far
along is it?" during the v74 run was answered by hand, by tailing a log and
doing arithmetic. That went wrong repeatedly, in one specific way, which is
what this module is built around.

## The mistake this exists to prevent

Quoting the *fastest recent interval* as the rate.

During v74 the observed per-step time ranged from 1.98 s/step to 5.73 s/step
within a single run -- a 2.9x spread, from accuracy probes, checkpoint writes,
and the machine paging. An ETA extrapolated from the fastest stretch is wrong
by hours, and it is wrong in the direction that sounds best.

So this never reports a single confident number:

* the **working rate** is the median of intervals that contain no probe
* intervals containing an accuracy probe are identified and priced separately,
  because a probe generates 100 replies and inflates its interval
* the ETA is a **range**, derived from the observed interval spread
* when the recent rate disagrees materially with the run average, that
  disagreement is stated rather than silently resolved in either direction

## What it will not do

It will not extrapolate from a log that has stopped moving. A run whose last
line is twenty minutes old is reported as stalled or crashed, with the age
shown -- not as a run that is 60% done. Guessing there is how a dead run gets
reported as progressing.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: `step 12000/18000  train 0.0839  dev 0.0926  ppl 1.10  acc 0.70  2092s`
#: The `acc` group is optional: it appears only on probe evaluations.
STEP_LINE = re.compile(
    r"^\s*step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s+"
    r"train\s+(?P<train>[0-9.eE+-]+)\s+"
    r"dev\s+(?P<dev>[0-9.eE+-]+)\s+"
    r"ppl\s+(?P<ppl>[0-9.eE+-]+)"
    r"(?:\s+acc\s+(?P<acc>[0-9.eE+-]+))?"
    r"\s+(?P<seconds>[0-9.]+)s\s*$"
)
LEG_LINE = re.compile(r"^===\s*leg\s+(?P<leg>\d+)\s+from step\s+(?P<start>\d+)\s*===")
INIT_LINE = re.compile(r"^\s*init_from\s+(?P<path>.+?)\s+\((?P<steps>\d+) prior steps\)")
RESTORED_LINE = re.compile(r"^\s*restored\s+optimiser=(?P<opt>\w+)\s+scheduler=(?P<sched>\w+)")
SELECTED_LINE = re.compile(r"^\s*selected\s+step\s+(?P<step>\d+)\s+on\s+(?P<detail>.+)$")
CHECKPOINT_LINE = re.compile(r"^\s*checkpoint\s+(?P<path>.+?)\s*$")
PARAMS_LINE = re.compile(r"^\s*parameters\s+(?P<total>[\d,]+) total\s*/\s*(?P<active>[\d,]+) active")
ROWS_LINE = re.compile(r"^\s*train\s+(?P<rows>[\d,]+) rows, dev (?P<dev>[\d,]+)")

#: A run whose newest step line is older than this, and which has not finished,
#: is not progressing. Chosen as a multiple of a typical eval interval (500
#: steps at ~3 s/step is ~25 minutes), so a healthy run is never called stalled.
STALL_AFTER_SECONDS = 45 * 60


@dataclass(frozen=True)
class Step:
    step: int
    total: int
    train: float
    dev: float
    ppl: float
    seconds: float
    accuracy: Optional[float] = None

    @property
    def has_probe(self) -> bool:
        return self.accuracy is not None


@dataclass
class Interval:
    """The gap between two consecutive step lines."""

    steps: int
    seconds: float
    #: True when the later line reported an accuracy probe, whose generation
    #: cost lands inside this interval.
    contains_probe: bool

    @property
    def rate(self) -> float:
        """Seconds per step."""

        return self.seconds / self.steps if self.steps else 0.0


@dataclass
class Run:
    name: str
    log_path: Path
    steps: List[Step] = field(default_factory=list)
    legs: List[int] = field(default_factory=list)
    resumed_from: Optional[int] = None
    restored_scheduler: Optional[bool] = None
    selected: Optional[str] = None
    checkpoint: Optional[str] = None
    parameters: Optional[int] = None
    train_rows: Optional[int] = None
    log_age_seconds: float = 0.0

    # -- position ----------------------------------------------------------

    @property
    def latest(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None

    @property
    def current_step(self) -> int:
        return self.latest.step if self.latest else (self.resumed_from or 0)

    @property
    def total_steps(self) -> int:
        return self.latest.total if self.latest else 0

    @property
    def remaining_steps(self) -> int:
        return max(0, self.total_steps - self.current_step)

    @property
    def fraction_done(self) -> float:
        return self.current_step / self.total_steps if self.total_steps else 0.0

    @property
    def finished(self) -> bool:
        return bool(self.latest and self.latest.step >= self.latest.total)

    # -- rate --------------------------------------------------------------

    @property
    def intervals(self) -> List[Interval]:
        out: List[Interval] = []
        for earlier, later in zip(self.steps, self.steps[1:]):
            gap = later.step - earlier.step
            elapsed = later.seconds - earlier.seconds
            if gap <= 0 or elapsed < 0:
                # A resumed leg restarts its clock; that boundary is not an
                # interval and must not be priced as one.
                continue
            out.append(Interval(gap, elapsed, later.has_probe))
        return out

    @property
    def working_rate(self) -> Optional[float]:
        """Median seconds/step over intervals with no probe in them.

        The median rather than the mean: one paging episode (v74 hit 5.73
        s/step for a single interval against a 2.85 norm) should not drag the
        estimate that the next hours are based on.
        """

        clean = [i.rate for i in self.intervals if not i.contains_probe]
        if not clean:
            clean = [i.rate for i in self.intervals]
        return statistics.median(clean) if clean else None

    @property
    def recent_rate(self) -> Optional[float]:
        """Seconds/step over the last few clean intervals."""

        clean = [i.rate for i in self.intervals if not i.contains_probe]
        if not clean:
            return None
        return statistics.median(clean[-3:])

    @property
    def rate_spread(self) -> Optional[tuple]:
        rates = [i.rate for i in self.intervals]
        return (min(rates), max(rates)) if rates else None

    @property
    def probe_overhead(self) -> Optional[float]:
        """Extra seconds an accuracy probe costs, over a clean interval."""

        probe = [i.rate for i in self.intervals if i.contains_probe]
        clean = [i.rate for i in self.intervals if not i.contains_probe]
        if not probe or not clean:
            return None
        per_step = statistics.median(probe) - statistics.median(clean)
        span = statistics.median([i.steps for i in self.intervals if i.contains_probe])
        return max(0.0, per_step * span)

    @property
    def probe_interval(self) -> Optional[int]:
        """How often the accuracy probe runs, inferred from the log."""

        probe_steps = [s.step for s in self.steps if s.has_probe]
        if len(probe_steps) < 2:
            return None
        gaps = {b - a for a, b in zip(probe_steps, probe_steps[1:])}
        return min(gaps) if gaps else None

    # -- estimate ----------------------------------------------------------

    def eta_seconds(self) -> Optional[tuple]:
        """(low, expected, high) seconds remaining, or None if unknowable.

        The range is not decoration. v74's intervals ranged 1.98-5.73 s/step
        inside one run; a single number implies a precision the data does not
        contain.
        """

        if self.finished or not self.remaining_steps:
            return None
        rate = self.working_rate
        if rate is None:
            return None

        expected = self.remaining_steps * rate

        # Price the probes still to come; each generates 100 replies.
        overhead = self.probe_overhead
        cadence = self.probe_interval
        if overhead and cadence:
            remaining_probes = self.remaining_steps // cadence
            expected += remaining_probes * overhead

        spread = self.rate_spread
        if spread:
            low = self.remaining_steps * spread[0]
            high = self.remaining_steps * spread[1]
            # The spread brackets the estimate; it never contradicts it.
            return (min(low, expected), expected, max(high, expected))
        return (expected, expected, expected)

    @property
    def rate_disagreement(self) -> Optional[str]:
        """Say so when recent pace differs materially from the run average.

        This is the specific trap: during v74 the last four intervals ran
        ~2.0 s/step against a 2.85 s/step run average, and quoting the recent
        figure alone would have promised an ETA over an hour early.
        """

        overall, recent = self.working_rate, self.recent_rate
        if overall is None or recent is None or overall <= 0:
            return None
        change = (recent - overall) / overall
        if abs(change) < 0.20:
            return None
        direction = "faster" if change < 0 else "slower"
        return (f"recent pace is {abs(change) * 100:.0f}% {direction} than the run "
                f"average ({recent:.2f} vs {overall:.2f} s/step); "
                "the estimate uses the average")

    # -- state -------------------------------------------------------------

    @property
    def expected_gap_seconds(self) -> Optional[float]:
        """How long the next step line should take, at the observed rate."""

        rate = self.working_rate
        if rate is None or len(self.steps) < 2:
            return None
        spans = [later.step - earlier.step
                 for earlier, later in zip(self.steps, self.steps[1:])
                 if later.step > earlier.step]
        if not spans:
            return None
        return statistics.median(spans) * rate

    @property
    def stall_threshold_seconds(self) -> float:
        """Silence long enough to mean something is wrong, given this run's pace.

        A fixed threshold is wrong, and was wrong in practice within hours of
        being written: 45 minutes assumes ~3 s/step, but v79 degraded to
        17 s/step under memory pressure, where a *healthy* gap between eval
        lines is 2.4 hours. The tracker called a working run stalled.

        So the threshold follows the run: three times the expected gap, floored
        at the fixed value so a fast run still gets a sane minimum.
        """

        expected = self.expected_gap_seconds
        if expected is None:
            return float(STALL_AFTER_SECONDS)
        return max(float(STALL_AFTER_SECONDS), 3.0 * expected)

    @property
    def status(self) -> str:
        if self.finished:
            return "complete"
        if not self.steps:
            return "starting"
        if self.log_age_seconds > self.stall_threshold_seconds:
            return "stalled"
        return "running"

    @property
    def accuracy_trend(self) -> List[tuple]:
        return [(s.step, s.accuracy) for s in self.steps if s.has_probe]

    def to_dict(self) -> Dict[str, Any]:
        eta = self.eta_seconds()
        return {
            "name": self.name,
            "log": str(self.log_path),
            "status": self.status,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "fraction_done": round(self.fraction_done, 4),
            "train_loss": self.latest.train if self.latest else None,
            "dev_loss": self.latest.dev if self.latest else None,
            "working_rate_s_per_step": (round(self.working_rate, 3)
                                        if self.working_rate else None),
            "recent_rate_s_per_step": (round(self.recent_rate, 3)
                                       if self.recent_rate else None),
            "rate_disagreement": self.rate_disagreement,
            "eta_seconds": [round(v) for v in eta] if eta else None,
            "accuracy_trend": self.accuracy_trend,
            "resumed_from": self.resumed_from,
            "restored_scheduler": self.restored_scheduler,
            "selected": self.selected,
            "checkpoint": self.checkpoint,
            "log_age_seconds": round(self.log_age_seconds),
        }


def parse_log(path, now: Optional[float] = None) -> Run:
    """Read one training log into a `Run`."""

    path = Path(path)
    run = Run(name=path.stem, log_path=path)
    text = path.read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        match = STEP_LINE.match(line)
        if match:
            accuracy = match.group("acc")
            run.steps.append(Step(
                step=int(match.group("step")),
                total=int(match.group("total")),
                train=float(match.group("train")),
                dev=float(match.group("dev")),
                ppl=float(match.group("ppl")),
                seconds=float(match.group("seconds")),
                accuracy=float(accuracy) if accuracy is not None else None,
            ))
            continue

        leg = LEG_LINE.match(line)
        if leg:
            run.legs.append(int(leg.group("start")))
            continue

        init = INIT_LINE.match(line)
        if init:
            run.resumed_from = int(init.group("steps"))
            continue

        restored = RESTORED_LINE.match(line)
        if restored:
            run.restored_scheduler = restored.group("sched") == "True"
            continue

        selected = SELECTED_LINE.match(line)
        if selected:
            run.selected = f"step {selected.group('step')} on {selected.group('detail')}"
            continue

        params = PARAMS_LINE.match(line)
        if params:
            run.parameters = int(params.group("total").replace(",", ""))
            continue

        rows = ROWS_LINE.match(line)
        if rows:
            run.train_rows = int(rows.group("rows").replace(",", ""))
            continue

        checkpoint = CHECKPOINT_LINE.match(line)
        if checkpoint and "output" in checkpoint.group("path"):
            run.checkpoint = checkpoint.group("path").strip()

    reference = now if now is not None else time.time()
    try:
        run.log_age_seconds = max(0.0, reference - path.stat().st_mtime)
    except OSError:
        run.log_age_seconds = 0.0
    return run


def read_supervisor_journal(path) -> Optional[Dict[str, Any]]:
    """Summarise a `supervised_run.json` written by `train_supervised.py`.

    A log shows where a run is; the journal shows what it survived. v74
    crashed once with SIGSEGV at step 11,500 and the second leg finished --
    a fact that appears in no log line, because each leg writes its own log.
    """

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("schema") != "supermix-v75-supervised-run-v1":
        return None

    legs = payload.get("legs") or []
    arguments = payload.get("train_args") or []
    run_name = None
    for index, token in enumerate(arguments):
        if token == "--run_name" and index + 1 < len(arguments):
            run_name = arguments[index + 1]
            break

    return {
        "run_name": run_name,
        "path": str(path),
        "restarts": payload.get("restarts", max(0, len(legs) - 1)),
        "legs": [
            {
                "leg": leg.get("leg"),
                "start_step": leg.get("start_step"),
                "reached_step": leg.get("reached_step"),
                "exit_code": leg.get("exit_code"),
                "seconds": leg.get("seconds"),
                "outcome": leg.get("outcome"),
            }
            for leg in legs
        ],
    }


def find_supervisor_journals(directories: Sequence[str]) -> List[Dict[str, Any]]:
    journals = []
    for directory in directories:
        base = Path(directory)
        if not base.is_dir():
            continue
        for candidate in sorted(base.glob("*/supervised_run.json")):
            summary = read_supervisor_journal(candidate)
            if summary:
                journals.append(summary)
    return journals


def render_journal(journal: Dict[str, Any]) -> str:
    name = journal.get("run_name") or Path(journal["path"]).parent.name
    restarts = journal["restarts"]
    header = (f"{name}  [supervised]  "
              f"{'no restarts' if not restarts else f'{restarts} restart(s)'}")
    lines = [header]
    for leg in journal["legs"]:
        detail = f"  leg {leg['leg']}  from {leg['start_step']:,}"
        if leg.get("reached_step") is not None:
            detail += f" reached {leg['reached_step']:,}"
        if leg.get("seconds"):
            detail += f"  {format_duration(leg['seconds'])}"
        # A non-zero exit is the whole point of the journal; name it.
        detail += f"  exit {leg['exit_code']}"
        if leg.get("outcome"):
            detail += f" - {leg['outcome']}"
        lines.append(detail)
    return "\n".join(lines)


def discover_logs(directories: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for directory in directories:
        base = Path(directory)
        if not base.is_dir():
            continue
        found.extend(sorted(base.glob("*.log")))
    return found


def find_runs(directories: Sequence[str]) -> List[Run]:
    """Every log that actually contains generalisation step lines."""

    runs = []
    for path in discover_logs(directories):
        try:
            run = parse_log(path)
        except OSError:
            continue
        if run.steps:  # ignore logs from other tools
            runs.append(run)
    return runs


# -- rendering --------------------------------------------------------------


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes = remainder // 60
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{seconds}s"


def render(run: Run, verbose: bool = False) -> str:
    lines = [f"{run.name}  [{run.status}]"]

    if run.steps:
        lines.append(
            f"  step        {run.current_step:,} / {run.total_steps:,} "
            f"({run.fraction_done * 100:.1f}%)"
        )
        latest = run.latest
        lines.append(f"  loss        train {latest.train:.4f}  dev {latest.dev:.4f}  "
                     f"ppl {latest.ppl:.2f}")

    if run.resumed_from:
        restored = ""
        if run.restored_scheduler is not None:
            restored = ("  schedule restored" if run.restored_scheduler
                        else "  SCHEDULE NOT RESTORED")
        lines.append(f"  resumed     from step {run.resumed_from:,}{restored}")

    rate = run.working_rate
    if rate:
        detail = f"  rate        {rate:.2f} s/step"
        spread = run.rate_spread
        if spread:
            detail += f"  (observed {spread[0]:.2f}-{spread[1]:.2f})"
        lines.append(detail)

    if run.status == "complete":
        lines.append("  eta         finished")
        if run.selected:
            lines.append(f"  selected    {run.selected}")
        if run.checkpoint:
            lines.append(f"  checkpoint  {run.checkpoint}")
    elif run.status == "stalled":
        # Never extrapolate from a log that has stopped moving.
        lines.append(
            f"  eta         unknown - no new step for "
            f"{format_duration(run.log_age_seconds)}"
        )
    else:
        eta = run.eta_seconds()
        if eta:
            low, expected, high = eta
            lines.append(f"  eta         {format_duration(expected)}  "
                         f"(range {format_duration(low)}-{format_duration(high)})")
            finish = time.time() + expected
            lines.append(f"  finishes    ~{time.strftime('%H:%M', time.localtime(finish))}")
        else:
            lines.append("  eta         unknown - not enough measured intervals yet")

    disagreement = run.rate_disagreement
    if disagreement:
        lines.append(f"  note        {disagreement}")

    trend = run.accuracy_trend
    if trend:
        rendered = "  ".join(f"{step:,}:{value:.2f}" for step, value in trend)
        lines.append(f"  accuracy    {rendered}")

    if verbose:
        if run.parameters:
            lines.append(f"  parameters  {run.parameters:,}")
        if run.train_rows:
            lines.append(f"  corpus      {run.train_rows:,} rows")
        overhead = run.probe_overhead
        if overhead:
            lines.append(f"  probe cost  {format_duration(overhead)} each, "
                         f"every {run.probe_interval:,} steps")
        lines.append(f"  log         {run.log_path}")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="*", default=None,
                        help="log files to read; omit to search --log_dir")
    parser.add_argument("--log_dir", action="append", default=[],
                        help="directory to search for *.log; repeatable")
    parser.add_argument("--run", default=None,
                        help="show only the run whose name contains this")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable output")
    parser.add_argument("--verbose", action="store_true",
                        help="include corpus, parameters and probe cost")
    parser.add_argument("--watch", type=float, default=0.0, metavar="SECONDS",
                        help="refresh every SECONDS until interrupted")
    parser.add_argument("--output_dir", action="append", default=[],
                        help=("directory holding run outputs; scanned for "
                              "supervised_run.json crash history. Repeatable"))
    return parser


def collect(args) -> List[Run]:
    if args.logs:
        runs = [parse_log(path) for path in args.logs]
    else:
        directories = args.log_dir or ["."]
        runs = find_runs(directories)
    if args.run:
        runs = [r for r in runs if args.run.lower() in r.name.lower()]
    return runs


def report(runs: List[Run], args) -> str:
    journals = find_supervisor_journals(getattr(args, "output_dir", []) or [])

    if args.json:
        return json.dumps(
            {"runs": [r.to_dict() for r in runs], "supervised": journals}, indent=2
        )

    sections = [render(r, verbose=args.verbose) for r in runs]
    sections += [render_journal(j) for j in journals]
    if not sections:
        return "no generalisation training logs found"
    return "\n\n".join(sections)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.watch > 0:
        try:
            while True:
                print("\033[2J\033[H", end="")
                print(report(collect(args), args), flush=True)
                time.sleep(max(1.0, args.watch))
        except KeyboardInterrupt:
            return 0

    runs = collect(args)
    print(report(runs, args))
    return 0 if (runs or args.output_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
