"""Tests for the crash supervisor.

v64 and v74 both died with SIGSEGV at an eval boundary, costing 6.8 and 9.2
hours. The supervisor exists to make the *next* one cost one eval interval.
The behaviours worth pinning are the ones that decide whether it restarts, and
in particular the two cases where it must refuse to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import train_supervised as supervisor  # noqa: E402


BASE = [
    "--steps", "18000",
    "--run_name", "v74_broad",
    "--output_dir", "output/v74_broad",
    "--checkpoint_every_improvement",
]


def test_reads_space_separated_arguments():
    assert supervisor._argument_value(BASE, "--run_name") == "v74_broad"
    assert supervisor._argument_value(BASE, "--steps") == "18000"


def test_reads_equals_separated_arguments():
    assert supervisor._argument_value(["--steps=900"], "--steps") == "900"


def test_missing_argument_is_none():
    assert supervisor._argument_value(BASE, "--init_from") is None


def test_flag_at_end_of_list_does_not_index_past_it():
    """`--run_name` with no value must not raise."""

    assert supervisor._argument_value(["--run_name"], "--run_name") is None


def test_resume_appends_init_from_and_start_step():
    resumed = supervisor.resume_arguments(BASE, Path("ck.pt"), 11500)

    assert resumed[-4:] == ["--init_from", "ck.pt", "--start_step", "11500"]
    assert "--steps" in resumed and "18000" in resumed


def test_resume_replaces_a_previous_resume():
    """A second crash must resume from the newest checkpoint, not the first."""

    once = supervisor.resume_arguments(BASE, Path("a.pt"), 4000)
    twice = supervisor.resume_arguments(once, Path("b.pt"), 9000)

    assert twice.count("--init_from") == 1
    assert twice.count("--start_step") == 1
    assert twice[-4:] == ["--init_from", "b.pt", "--start_step", "9000"]
    assert "a.pt" not in twice


def test_resume_replaces_equals_form_too():
    args = BASE + ["--init_from=a.pt", "--start_step=4000"]

    resumed = supervisor.resume_arguments(args, Path("b.pt"), 9000)

    assert "a.pt" not in " ".join(resumed)
    assert resumed[-4:] == ["--init_from", "b.pt", "--start_step", "9000"]


def test_supervision_refuses_without_a_recovery_checkpoint():
    """No recovery checkpoint means a crash restarts from zero, silently."""

    with pytest.raises(SystemExit, match="checkpoint_every_improvement"):
        supervisor.main(["--", "--steps", "10", "--run_name", "x"])


def test_no_training_arguments_raises():
    with pytest.raises(SystemExit, match="no training arguments"):
        supervisor.main([])


def test_reached_step_reads_the_checkpoint(tmp_path):
    path = tmp_path / "ck.pt"
    torch.save({"extra": {"steps": 11500}}, path)

    assert supervisor.reached_step(path) == 11500


def test_reached_step_survives_a_truncated_checkpoint(tmp_path):
    """A file written while the process died is unusable, not a crash."""

    path = tmp_path / "ck.pt"
    path.write_bytes(b"not a checkpoint")

    assert supervisor.reached_step(path) is None


def test_reached_step_is_none_when_the_step_was_never_recorded(tmp_path):
    path = tmp_path / "ck.pt"
    torch.save({"extra": {}}, path)

    assert supervisor.reached_step(path) is None


def test_recovery_checkpoint_found_when_present(tmp_path):
    (tmp_path / "run.partial.pt").write_bytes(b"x")
    args = ["--output_dir", str(tmp_path), "--run_name", "run"]

    assert supervisor.recovery_checkpoint(args) == tmp_path / "run.partial.pt"


def test_recovery_checkpoint_absent_returns_none(tmp_path):
    args = ["--output_dir", str(tmp_path), "--run_name", "run"]

    assert supervisor.recovery_checkpoint(args) is None


def test_journal_records_every_leg(tmp_path):
    legs = [
        {"leg": 1, "exit_code": 139, "reached_step": 11500},
        {"leg": 2, "exit_code": 0},
    ]
    path = tmp_path / "supervised_run.json"
    supervisor.write_journal(path, legs, BASE)

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["restarts"] == 1
    assert len(payload["legs"]) == 2
    assert payload["schema"] == supervisor.JOURNAL_SCHEMA


# -- the restart decision ---------------------------------------------------
#
# These drive `run()` with a fake trainer. The loop is what burns hours when it
# is wrong, in both directions: not restarting wastes a recoverable run, and
# restarting forever reproduces the same failure until someone notices.


class _FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def _drive(monkeypatch, tmp_path, exit_codes, steps_seen, max_restarts=4):
    """Run the supervisor against a trainer that exits with `exit_codes`."""

    calls = []
    codes = iter(exit_codes)
    reached = iter(steps_seen)
    checkpoint = tmp_path / "run.partial.pt"
    checkpoint.write_bytes(b"x")

    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs.get("env", {}).get("PYTHONUNBUFFERED") == "1"
        return _FakeCompleted(next(codes))

    def fake_reached(_path):
        return next(reached, None)

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(supervisor, "reached_step", fake_reached)

    args = ["--output_dir", str(tmp_path), "--run_name", "run",
            "--steps", "18000", "--checkpoint_every_improvement"]
    code = supervisor.run(max_restarts, args)
    return code, calls


def test_a_clean_run_is_never_restarted(monkeypatch, tmp_path):
    code, calls = _drive(monkeypatch, tmp_path, [0], [])

    assert code == 0
    assert len(calls) == 1


def test_a_crash_resumes_from_the_recovered_step(monkeypatch, tmp_path):
    code, calls = _drive(monkeypatch, tmp_path, [139, 0], [11500])

    assert code == 0
    assert len(calls) == 2
    assert "--start_step" in calls[1]
    assert calls[1][calls[1].index("--start_step") + 1] == "11500"


def test_repeated_crashes_resume_from_each_new_step(monkeypatch, tmp_path):
    code, calls = _drive(monkeypatch, tmp_path, [139, 139, 0], [11500, 14000])

    assert code == 0
    assert calls[1][calls[1].index("--start_step") + 1] == "11500"
    assert calls[2][calls[2].index("--start_step") + 1] == "14000"


def test_a_failure_that_makes_no_progress_stops(monkeypatch, tmp_path):
    """The hour-burning case: crashing at the same step is deterministic."""

    code, calls = _drive(monkeypatch, tmp_path, [139] * 5, [9000, 9000, 9000])

    assert code == 1
    assert len(calls) == 2  # the retry ran once, then the guard stopped it


def test_going_backwards_also_stops(monkeypatch, tmp_path):
    code, calls = _drive(monkeypatch, tmp_path, [139] * 5, [9000, 8500])

    assert code == 1
    assert len(calls) == 2


def test_max_restarts_is_honoured(monkeypatch, tmp_path):
    code, calls = _drive(
        monkeypatch, tmp_path, [139] * 6, [1000, 2000, 3000, 4000, 5000],
        max_restarts=2,
    )

    assert code == 1
    assert len(calls) == 3  # the first leg plus two restarts


def test_an_unreadable_checkpoint_stops_rather_than_guessing(monkeypatch, tmp_path):
    code, calls = _drive(monkeypatch, tmp_path, [139, 139], [None])

    assert code == 1
    assert len(calls) == 1


def test_the_journal_survives_a_giving_up_run(monkeypatch, tmp_path):
    import json

    _drive(monkeypatch, tmp_path, [139] * 5, [9000, 9000])

    payload = json.loads((tmp_path / "supervised_run.json").read_text(encoding="utf-8"))
    assert payload["legs"][-1]["outcome"].startswith("no progress past step 9000")
