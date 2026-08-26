"""Tests for reading a generalisation run's progress.

The v74 run was tracked by hand, because `training_monitor_gui.py` parses the
LoRA pipeline's format and matches **0 of 13** step lines in a generalisation
log. Doing it by hand went wrong in one repeatable way: quoting the fastest
recent stretch as the rate. Within v74 the observed per-step time ranged
1.98-5.73 s/step, so that error is worth hours.

Most of these tests pin the refusals -- the cases where the honest answer is
"unknown" and a plausible number would be worse than none.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import training_tracker as tracker  # noqa: E402


REAL_LOG = """v58 generalisation | arm full | thinking core True
  train        471,347 rows, dev 4,866
  parameters   8,575,977 total / 2,810,973 active
  restored     optimiser=True scheduler=True
step 12000/18000  train 0.0839  dev 0.0926  ppl 1.10  acc 0.70  2092s
step 12500/18000  train 0.0877  dev 0.0896  ppl 1.09  3312s
step 13000/18000  train 0.0687  dev 0.0847  ppl 1.09  4761s
step 13500/18000  train 0.0660  dev 0.0800  ppl 1.08  6205s
step 14000/18000  train 0.0714  dev 0.0763  ppl 1.08  7648s
step 15000/18000  train 0.0574  dev 0.0702  ppl 1.07  acc 0.82  10632s
"""


def _write(tmp_path, text, name="run.log"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# -- parsing the format the old monitor cannot read -------------------------


def test_parses_the_generalisation_step_format(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert len(run.steps) == 6
    assert run.current_step == 15000
    assert run.total_steps == 18000


def test_the_old_monitor_regex_would_match_none_of_these():
    """The gap this module exists to close, pinned as a fact."""

    import re

    lora = re.compile(r"\[train\] step=(\d+) loss=")
    step_lines = [l for l in REAL_LOG.splitlines() if l.strip().startswith("step ")]

    assert len(step_lines) == 6
    assert sum(1 for line in step_lines if lora.search(line)) == 0


def test_reads_losses_and_perplexity(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.latest.train == pytest.approx(0.0574)
    assert run.latest.dev == pytest.approx(0.0702)
    assert run.latest.ppl == pytest.approx(1.07)


def test_accuracy_is_optional_and_only_on_probe_lines(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert [s.has_probe for s in run.steps] == [True, False, False, False, False, True]


def test_accuracy_trend_is_extracted(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.accuracy_trend == [(12000, 0.70), (15000, 0.82)]


def test_reads_resume_provenance(tmp_path):
    text = "  init_from    x.pt (11500 prior steps)\n" + REAL_LOG
    run = tracker.parse_log(_write(tmp_path, text))

    assert run.resumed_from == 11500
    assert run.restored_scheduler is True


def test_notices_when_the_schedule_was_not_restored(tmp_path):
    text = REAL_LOG.replace("scheduler=True", "scheduler=False")
    run = tracker.parse_log(_write(tmp_path, text))

    assert run.restored_scheduler is False


def test_reads_corpus_and_parameter_counts(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.parameters == 8575977
    assert run.train_rows == 471347


def test_a_log_from_another_tool_yields_no_steps(tmp_path):
    other = "[train] step=1200 loss=1.83 lr=3e-05\n"

    assert tracker.parse_log(_write(tmp_path, other)).steps == []


# -- rate, and the mistake it exists to prevent -----------------------------


def test_probe_intervals_are_excluded_from_the_working_rate(tmp_path):
    """A probe generates 100 replies; its interval is not a step rate."""

    run = tracker.parse_log(_write(tmp_path, REAL_LOG))
    clean = [i for i in run.intervals if not i.contains_probe]
    probe = [i for i in run.intervals if i.contains_probe]

    assert clean and probe
    import statistics
    assert run.working_rate == pytest.approx(
        statistics.median(i.rate for i in clean), rel=0.02
    )


def test_working_rate_is_not_the_fastest_interval(tmp_path):
    """The exact error made by hand during v74."""

    run = tracker.parse_log(_write(tmp_path, REAL_LOG))
    fastest = min(i.rate for i in run.intervals)

    assert run.working_rate > fastest


def test_rate_spread_is_reported(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))
    low, high = run.rate_spread

    assert low < high


def test_a_single_slow_interval_does_not_dominate(tmp_path):
    """v74 hit 5.73 s/step for one paging episode against a 2.85 norm."""

    text = REAL_LOG + "step 15500/18000  train 0.05  dev 0.07  ppl 1.07  40000s\n"
    run = tracker.parse_log(_write(tmp_path, text))

    assert run.working_rate < 10  # a mean would be dragged far above this


def test_rate_disagreement_is_stated_not_resolved_silently(tmp_path):
    text = REAL_LOG + "".join(
        "step {}/18000  train 0.05  dev 0.06  ppl 1.07  {}s\n".format(
            16000 + i * 500, 10632 + (i + 1) * 500
        )
        for i in range(3)
    )
    run = tracker.parse_log(_write(tmp_path, text))

    note = run.rate_disagreement
    assert note is not None
    assert "faster" in note
    assert "the estimate uses the average" in note


def test_no_disagreement_note_when_pace_is_steady(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.rate_disagreement is None


def test_a_resume_boundary_does_not_create_a_bogus_interval(tmp_path):
    """A resumed leg restarts its clock; that gap is not a step rate."""

    text = REAL_LOG + "step 15500/18000  train 0.05  dev 0.06  ppl 1.07  12s\n"
    run = tracker.parse_log(_write(tmp_path, text))

    assert all(i.seconds >= 0 for i in run.intervals)


# -- the estimate -----------------------------------------------------------


def test_eta_is_a_range_that_brackets_the_estimate(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))
    low, expected, high = run.eta_seconds()

    assert low <= expected <= high


def test_eta_accounts_for_remaining_steps(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.remaining_steps == 3000
    assert run.eta_seconds()[1] > 3000 * 1.0


def test_a_finished_run_has_no_eta(tmp_path):
    text = REAL_LOG + "step 18000/18000  train 0.05  dev 0.065  ppl 1.07  acc 0.89  16953s\n"
    run = tracker.parse_log(_write(tmp_path, text))

    assert run.finished
    assert run.eta_seconds() is None
    assert run.status == "complete"


def test_no_eta_without_measured_intervals(tmp_path):
    one = "step 500/18000  train 1.0  dev 1.0  ppl 3.0  1200s\n"
    run = tracker.parse_log(_write(tmp_path, one))

    assert run.eta_seconds() is None


# -- refusing to extrapolate from a dead run --------------------------------


def test_a_stalled_run_is_not_reported_as_progressing(tmp_path):
    """The v74 crash: the log stopped at 11,500 of 18,000 and never moved.

    Reporting "63.9% done, 5h remaining" for a dead process is the worst
    possible output, because it reads as normal.
    """

    path = _write(tmp_path, REAL_LOG)
    run = tracker.parse_log(path, now=time.time() + 6 * 3600)

    assert run.status == "stalled"


def test_a_stalled_run_renders_without_an_eta(tmp_path):
    path = _write(tmp_path, REAL_LOG)
    run = tracker.parse_log(path, now=time.time() + 6 * 3600)

    rendered = tracker.render(run)
    assert "no new step for" in rendered
    assert "finishes" not in rendered


def test_a_fresh_run_is_running_not_stalled(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.status == "running"


def test_the_stall_threshold_exceeds_a_normal_eval_gap():
    """500 steps at ~3 s/step is ~25 min; a healthy run must never trip it."""

    assert tracker.STALL_AFTER_SECONDS > 25 * 60


SLOW_LOG = """step  1000/18000  train 0.71  dev 0.65  ppl 1.93  4560s
step  1500/18000  train 0.61  dev 0.59  ppl 1.81  6275s
step  2000/18000  train 0.55  dev 0.53  ppl 1.71  9481s
step  2500/18000  train 0.49  dev 0.50  ppl 1.66  18042s
"""


def test_the_stall_threshold_follows_the_run_pace(tmp_path):
    """A fixed threshold called a working run stalled, within hours of being
    written.

    v79 degraded to 17.12 s/step under memory pressure. At that pace a healthy
    gap between eval lines is ~2.4 hours, and the fixed 45-minute threshold
    reported a run that was demonstrably alive -- CPU climbing, 3.6 cores
    busy -- as stalled.
    """

    run = tracker.parse_log(_write(tmp_path, SLOW_LOG))

    assert run.stall_threshold_seconds > tracker.STALL_AFTER_SECONDS
    assert run.expected_gap_seconds > 30 * 60


def test_a_slow_run_silent_for_under_the_adaptive_threshold_is_running(tmp_path):
    path = _write(tmp_path, SLOW_LOG)
    run = tracker.parse_log(path, now=time.time() + 57 * 60)

    assert run.status == "running"


def test_a_slow_run_silent_for_far_too_long_is_still_caught(tmp_path):
    """Adapting the threshold must not disable stall detection."""

    path = _write(tmp_path, SLOW_LOG)
    run = tracker.parse_log(path, now=time.time() + 10 * 3600)

    assert run.status == "stalled"


def test_a_fast_run_keeps_the_floor(tmp_path):
    """A quick run must not get a threshold so tight it trips on noise."""

    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert run.stall_threshold_seconds >= tracker.STALL_AFTER_SECONDS


def test_expected_gap_is_none_without_intervals(tmp_path):
    one = "step 500/18000  train 1.0  dev 1.0  ppl 3.0  1200s\n"
    run = tracker.parse_log(_write(tmp_path, one))

    assert run.expected_gap_seconds is None
    assert run.stall_threshold_seconds == float(tracker.STALL_AFTER_SECONDS)


# -- discovery and output ---------------------------------------------------


def test_discovery_skips_logs_with_no_step_lines(tmp_path):
    _write(tmp_path, "[train] step=1 loss=2\n", "lora.log")
    _write(tmp_path, REAL_LOG, "generalisation.log")

    found = tracker.find_runs([str(tmp_path)])

    assert [r.name for r in found] == ["generalisation"]


def test_json_output_is_machine_readable(tmp_path):
    import json

    run = tracker.parse_log(_write(tmp_path, REAL_LOG))
    payload = json.loads(json.dumps(run.to_dict()))

    assert payload["step"] == 15000
    assert payload["status"] == "running"
    assert payload["accuracy_trend"] == [[12000, 0.70], [15000, 0.82]]


def test_render_includes_the_accuracy_trend(tmp_path):
    run = tracker.parse_log(_write(tmp_path, REAL_LOG))

    assert "0.70" in tracker.render(run)
    assert "0.82" in tracker.render(run)


def test_duration_formatting():
    assert tracker.format_duration(45) == "45s"
    assert tracker.format_duration(600) == "10m"
    assert tracker.format_duration(7 * 3600 + 120) == "7h02m"


# -- what a run survived ----------------------------------------------------
#
# A log says where a run is; the supervisor journal says what it survived.
# v74 crashed with SIGSEGV at step 11,500 and a second leg finished it. That
# fact appears in no log line, because each leg writes its own log.


JOURNAL = {
    "schema": "supermix-v75-supervised-run-v1",
    "train_args": ["--steps", "18000", "--run_name", "v74_broad"],
    "restarts": 1,
    "legs": [
        {"leg": 1, "start_step": 0, "exit_code": 139, "seconds": 33042.0,
         "reached_step": 11500, "outcome": "crashed at ~11500; resuming"},
        {"leg": 2, "start_step": 11500, "exit_code": 0, "seconds": 17580.4,
         "outcome": "completed"},
    ],
}


def _write_journal(tmp_path, payload, run_dir="v74_broad"):
    import json

    directory = tmp_path / run_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "supervised_run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_journal_reports_the_run_name_and_restarts(tmp_path):
    summary = tracker.read_supervisor_journal(_write_journal(tmp_path, JOURNAL))

    assert summary["run_name"] == "v74_broad"
    assert summary["restarts"] == 1
    assert len(summary["legs"]) == 2


def test_journal_preserves_the_crash_exit_code(tmp_path):
    """139 is SIGSEGV; losing it would hide why a leg ended."""

    summary = tracker.read_supervisor_journal(_write_journal(tmp_path, JOURNAL))

    assert summary["legs"][0]["exit_code"] == 139
    assert summary["legs"][1]["exit_code"] == 0


def test_render_names_the_non_zero_exit(tmp_path):
    summary = tracker.read_supervisor_journal(_write_journal(tmp_path, JOURNAL))

    rendered = tracker.render_journal(summary)
    assert "exit 139" in rendered
    assert "1 restart" in rendered


def test_render_says_so_when_nothing_crashed(tmp_path):
    clean = dict(JOURNAL, restarts=0, legs=[JOURNAL["legs"][1]])
    summary = tracker.read_supervisor_journal(_write_journal(tmp_path, clean))

    assert "no restarts" in tracker.render_journal(summary)


def test_journals_are_discovered_under_an_output_directory(tmp_path):
    _write_journal(tmp_path, JOURNAL)

    found = tracker.find_supervisor_journals([str(tmp_path)])

    assert len(found) == 1
    assert found[0]["run_name"] == "v74_broad"


def test_an_unknown_schema_is_ignored(tmp_path):
    path = _write_journal(tmp_path, dict(JOURNAL, schema="something-else"))

    assert tracker.read_supervisor_journal(path) is None


def test_a_corrupt_journal_is_ignored_rather_than_raising(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    path = directory / "supervised_run.json"
    path.write_text("{not json", encoding="utf-8")

    assert tracker.read_supervisor_journal(path) is None
    assert tracker.find_supervisor_journals([str(tmp_path)]) == []


def test_a_missing_journal_is_ignored(tmp_path):
    assert tracker.read_supervisor_journal(tmp_path / "nope.json") is None
