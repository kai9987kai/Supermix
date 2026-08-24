"""Tests for the v59 dark-test audit.

The audit exists because a test file can look like coverage and run nothing. A
detector for that failure mode has the same failure mode, so these tests plant
dark functions and require it to notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import dark_test_audit as audit  # noqa: E402


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


def test_detects_a_planted_smoke_test(tmp_path):
    _write(tmp_path, "test_planted.py", "def smoke_test_thing():\n    assert True\n")

    report = audit.audit(tmp_path)

    assert report["dark_functions"] == 1
    assert report["files_collecting_nothing"] == 1


def test_collected_functions_are_not_flagged(tmp_path):
    _write(tmp_path, "test_fine.py", "def test_thing():\n    assert True\n")

    report = audit.audit(tmp_path)

    assert report["dark_functions"] == 0
    assert report["files_with_dark_functions"] == 0


def test_file_with_both_is_flagged_but_not_silent(tmp_path):
    """The dangerous case: a file CI runs that also hides uncollected checks."""

    _write(
        tmp_path,
        "test_mixed.py",
        "def test_real():\n    assert True\n\n\ndef smoke_test_hidden():\n    assert False\n",
    )

    report = audit.audit(tmp_path)

    assert report["dark_functions"] == 1
    assert report["files_collecting_nothing"] == 0
    assert report["files"][0]["collects_nothing"] is False


def test_test_classes_count_as_collection(tmp_path):
    _write(
        tmp_path,
        "test_classy.py",
        "class TestThing:\n    def test_a(self):\n        assert True\n\n\ndef smoke_test_x():\n    pass\n",
    )

    report = audit.audit(tmp_path)

    assert report["files_collecting_nothing"] == 0


def test_nested_functions_are_not_counted(tmp_path):
    """Only top-level functions are collectable, so only they can be dark."""

    _write(
        tmp_path,
        "test_nested.py",
        "def test_outer():\n    def smoke_test_inner():\n        pass\n    smoke_test_inner()\n",
    )

    report = audit.audit(tmp_path)

    assert report["dark_functions"] == 0


def test_unparseable_file_is_skipped_not_crashed(tmp_path):
    _write(tmp_path, "test_broken.py", "def (((\n")

    report = audit.audit(tmp_path)  # must not raise

    assert report["dark_functions"] == 0


def test_check_fails_when_dark_tests_grow():
    report = {"dark_functions": 31}
    verdict = audit.compare(report, {"dark_functions": 30})

    assert verdict["regressed"] is True
    assert verdict["delta"] == 1


def test_check_passes_when_dark_tests_shrink():
    verdict = audit.compare({"dark_functions": 12}, {"dark_functions": 30})

    assert verdict["regressed"] is False
    assert verdict["delta"] == -18


def test_repo_has_not_regressed_against_its_pinned_baseline():
    """The gate itself: no new dark tests may enter the repo."""

    report = audit.audit(audit.REPO_ROOT)
    baseline = audit.load_baseline()
    verdict = audit.compare(report, baseline)

    assert not verdict["regressed"], (
        f"dark tests rose from {verdict['baseline_dark']} to {verdict['current_dark']}. "
        "Rename smoke_test_*/check_* to test_* so pytest collects them."
    )


def test_baseline_file_exists_and_is_pinned():
    baseline = audit.load_baseline()

    assert baseline["dark_functions"] > 0, (
        "the baseline records a known debt; a zero baseline would mean the gate "
        "was pinned before the debt was measured"
    )
