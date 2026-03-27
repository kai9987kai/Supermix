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
    _inject_default("--recipe_tag", "v34_frontier_plus")
    _inject_default(
        "--summary_note",
        (
            "The v34 frontier+ recipe starts from the stronger v33 stage2 checkpoint, expands the real-world dataset mix, "
            "uses more recent paper synthesis rows, applies T-SHIRT-style reference selection, and adds explicit self-repair supervision."
        ),
    )
    raise SystemExit(build_frontier_main())
