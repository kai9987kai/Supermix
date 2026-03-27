#!/usr/bin/env python3
from __future__ import annotations

import sys

from build_v33_frontier_dataset import main as build_frontier_main


def _inject_default(flag: str, value: str) -> None:
    if flag in sys.argv:
        return
    sys.argv.insert(1, value)
    sys.argv.insert(1, flag)


if __name__ == "__main__":
    _inject_default("--recipe_tag", "v35_allteachers_collective")
    _inject_default(
        "--summary_note",
        (
            "The v35 all-teachers collective recipe distills every loadable trained text checkpoint on the pod into a larger frontier_collective_expert student, "
            "then refines it on the expanded mixed dataset with automatic best-checkpoint selection."
        ),
    )
    raise SystemExit(build_frontier_main())
