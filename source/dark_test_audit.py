"""v59: find test functions pytest will never run.

`test_runtime_compute_controls.py` is in the CI pytest list and passes. It also
contains eight functions named ``smoke_test_*``, and pytest's default
``python_functions = test*`` never collects them. CI has been reporting green on
a file whose eight checks do not execute.

Repo-wide there are 29 such functions across 19 files. Seventeen of those files
are the 2026-06-16 expert cohort and collect **nothing at all** -- the file is a
test file by name, is counted as coverage by eye, and runs zero assertions.

This is the failure mode the repo already names elsewhere: a verifier that can
never reject. A dark test is worse than a missing one, because a missing test is
visible.

The audit pins a baseline rather than failing outright. Failing on all 29 would
make the gate useless on day one -- it would be red forever and get ignored --
so the contract is *no new dark tests*: the count may fall freely, and any rise
fails. Lowering the baseline is a deliberate edit, which is the point.

    python source/dark_test_audit.py                  # report
    python source/dark_test_audit.py --check          # non-zero if worse than baseline
    python source/dark_test_audit.py --update-baseline
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "source" / "dark_test_baseline.json"

#: pytest's default ``python_functions``. A top-level function is collected only
#: if it starts with this; anything else in a ``test_*.py`` file is inert.
COLLECTED_PREFIX = "test"

#: Prefixes seen in this repo for functions that look like tests but are not
#: collected. Reported specifically because they are the ones that read as
#: coverage.
SUSPICIOUS_PREFIXES = ("smoke_test", "check_", "verify_", "_test")


@dataclass
class FileAudit:
    path: str
    collected: List[str] = field(default_factory=list)
    dark: List[str] = field(default_factory=list)
    has_test_classes: bool = False

    @property
    def collects_nothing(self) -> bool:
        return not self.collected and not self.has_test_classes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "collected": len(self.collected),
            "dark": sorted(self.dark),
            "collects_nothing": self.collects_nothing,
        }


def audit_file(path: Path) -> Optional[FileAudit]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return None

    result = FileAudit(path=path.name)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("Test"):
                result.has_test_classes = True
            continue
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith(COLLECTED_PREFIX):
            result.collected.append(node.name)
        elif node.name.startswith(SUSPICIOUS_PREFIXES):
            result.dark.append(node.name)
    return result


def audit(root: Path = REPO_ROOT) -> Dict[str, Any]:
    audits: List[FileAudit] = []
    for path in sorted(root.glob("test_*.py")):
        found = audit_file(path)
        if found is not None and found.dark:
            audits.append(found)

    dark_total = sum(len(a.dark) for a in audits)
    silent = [a for a in audits if a.collects_nothing]
    return {
        "dark_functions": dark_total,
        "files_with_dark_functions": len(audits),
        "files_collecting_nothing": len(silent),
        "files": [a.to_dict() for a in audits],
        "worst": sorted(
            ({"path": a.path, "dark": len(a.dark), "collects_nothing": a.collects_nothing}
             for a in audits),
            key=lambda row: (-row["dark"], row["path"]),
        )[:5],
    }


def load_baseline(path: Path = BASELINE_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"dark_functions": 0, "files_with_dark_functions": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def compare(report: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    grew = report["dark_functions"] > baseline.get("dark_functions", 0)
    return {
        "baseline_dark": baseline.get("dark_functions", 0),
        "current_dark": report["dark_functions"],
        "delta": report["dark_functions"] - baseline.get("dark_functions", 0),
        "regressed": grew,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if there are more dark tests than the baseline")
    parser.add_argument("--update-baseline", action="store_true",
                        help="pin the current count as the new baseline")
    parser.add_argument("--json", action="store_true", help="emit the raw report")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit()

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"dark test functions      {report['dark_functions']}")
    print(f"files affected           {report['files_with_dark_functions']}")
    print(f"files collecting nothing {report['files_collecting_nothing']}")
    if report["worst"]:
        print("\nworst offenders:")
        for row in report["worst"]:
            note = " (file collects NOTHING)" if row["collects_nothing"] else " (file is otherwise collected)"
            print(f"  {row['dark']:2d}  {row['path']}{note}")

    if args.update_baseline:
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "dark_functions": report["dark_functions"],
                    "files_with_dark_functions": report["files_with_dark_functions"],
                    "note": "no new dark tests; lower this deliberately as they are fixed",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nbaseline pinned at {report['dark_functions']}")
        return 0

    if args.check:
        verdict = compare(report, load_baseline())
        if verdict["regressed"]:
            print(
                f"\nFAIL: dark tests rose from {verdict['baseline_dark']} to "
                f"{verdict['current_dark']}. A function named smoke_test_* or check_* "
                "in a test_*.py file is never collected by pytest -- rename it to "
                "test_* so it runs, or move it out of the test file."
            )
            return 1
        print(f"\nOK: {verdict['current_dark']} dark tests, baseline {verdict['baseline_dark']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
