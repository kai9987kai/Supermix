"""Run a generalisation training leg and rejoin its curve after a crash.

Two long runs on this machine have died mid-flight with SIGSEGV (exit 139),
both at an eval/checkpoint boundary:

    v64  step  5,500 of 10,000   6.8 hours lost
    v74  step 11,500 of 18,000   9.2 hours lost

Neither left a Python traceback, because a segfault does not raise -- the
interpreter is gone before any handler runs. The host has 15.6 GB of RAM and a
run holds the packed corpus, the model, AdamW moments and, at a save, a
serialisation buffer for all of it; the step timings around v74's crash show
the machine paging (steps 10,500-11,000 ran at 5.7 s/step against a 2.8 s/step
norm, then recovered). That is a plausible cause but not a proven one, and
chasing it further costs more hours than it saves.

So this does not try to prevent the crash. It makes the crash cheap.

`--checkpoint_every_improvement` already writes a recovery checkpoint whenever
dev loss improves, which is most evaluations. `--start_step` lets a new leg
rejoin the *same* OneCycle curve rather than warming up on a fresh one. This
supervisor joins the two: on a non-zero exit it reads the step the recovery
checkpoint reached and relaunches from there, so a segfault costs one eval
interval instead of the whole run.

What it deliberately does not do:

* **Restart on a clean exit.** Exit 0 means the run finished; relaunching would
  train a second leg nobody asked for.
* **Restart without progress.** If a relaunch dies at or before the step the
  previous one reached, the run is not crashing randomly -- it is failing
  deterministically at that point, and looping would burn hours reproducing it.
* **Hide anything.** Every leg's exit code and step range is printed and
  recorded in the journal beside the checkpoint.

Usage -- the training arguments are passed through verbatim after `--`::

    python source/train_supervised.py --max_restarts 4 -- \
        --steps 18000 --run_name v74_broad --output_dir output/v74_broad ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SOURCE_DIR = Path(__file__).resolve().parent
TRAINER = SOURCE_DIR / "train_mimomix_generalisation.py"
JOURNAL_SCHEMA = "supermix-v75-supervised-run-v1"


def _argument_value(train_args: List[str], name: str) -> Optional[str]:
    """Read `--name value` or `--name=value` out of a pass-through list."""

    for index, token in enumerate(train_args):
        if token == name and index + 1 < len(train_args):
            return train_args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def recovery_checkpoint(train_args: List[str]) -> Optional[Path]:
    output_dir = _argument_value(train_args, "--output_dir")
    run_name = _argument_value(train_args, "--run_name")
    if not output_dir or not run_name:
        return None
    candidate = Path(output_dir) / f"{run_name}.partial.pt"
    return candidate if candidate.exists() else None


def reached_step(checkpoint: Path) -> Optional[int]:
    """The step a recovery checkpoint holds, or None if it cannot be read.

    A checkpoint written while the process was dying can be truncated. That is
    a normal outcome here, not an error worth crashing the supervisor over, so
    a failed load returns None and the caller stops rather than guessing.
    """

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - any failure means "unusable"
        print(f"  recovery checkpoint unreadable ({type(error).__name__}: {error})")
        return None
    step = (payload.get("extra") or {}).get("steps")
    return int(step) if step is not None else None


def resume_arguments(train_args: List[str], checkpoint: Path, step: int) -> List[str]:
    """The next leg's arguments: same run, continued from `step`.

    Any `--init_from` or `--start_step` already present is replaced, so a
    supervised run that was itself started as a continuation still resumes from
    its own newest checkpoint rather than the original source.
    """

    stripped: List[str] = []
    skip_next = False
    for token in train_args:
        if skip_next:
            skip_next = False
            continue
        if token in ("--init_from", "--start_step"):
            skip_next = True
            continue
        if token.startswith("--init_from=") or token.startswith("--start_step="):
            continue
        stripped.append(token)
    return stripped + ["--init_from", str(checkpoint), "--start_step", str(step)]


def write_journal(path: Path, legs: List[Dict[str, Any]], train_args: List[str]) -> None:
    payload = {
        "schema": JOURNAL_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_args": train_args,
        "legs": legs,
        "restarts": max(0, len(legs) - 1),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(max_restarts: int, train_args: List[str]) -> int:
    output_dir = _argument_value(train_args, "--output_dir")
    journal = Path(output_dir) / "supervised_run.json" if output_dir else None
    legs: List[Dict[str, Any]] = []
    arguments = list(train_args)
    previous_step = -1

    for leg in range(max_restarts + 1):
        started = time.time()
        start_step = _argument_value(arguments, "--start_step") or "0"
        print(f"\n=== leg {leg + 1} from step {start_step} ===", flush=True)

        # Unbuffered, or a supervised run looks hung. Python block-buffers
        # stdout when it is a pipe or a file, so a trainer that prints one line
        # every ~24 minutes shows nothing for hours and then a burst -- which
        # is indistinguishable from a stall at exactly the moment someone is
        # trying to work out whether it crashed again.
        environment = dict(os.environ)
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("PYTHONIOENCODING", "utf-8")

        completed = subprocess.run(
            [sys.executable, "-u", str(TRAINER), *arguments],
            check=False,
            env=environment,
        )
        elapsed = round(time.time() - started, 1)
        record = {
            "leg": leg + 1,
            "start_step": int(start_step),
            "exit_code": completed.returncode,
            "seconds": elapsed,
        }

        if completed.returncode == 0:
            record["outcome"] = "completed"
            legs.append(record)
            if journal:
                write_journal(journal, legs, train_args)
            print(f"=== finished after {leg + 1} leg(s) ===", flush=True)
            return 0

        checkpoint = recovery_checkpoint(arguments)
        if checkpoint is None:
            record["outcome"] = "no recovery checkpoint; nothing to resume from"
            legs.append(record)
            break

        step = reached_step(checkpoint)
        if step is None:
            record["outcome"] = "recovery checkpoint unreadable"
            legs.append(record)
            break

        record["reached_step"] = step
        if step <= previous_step:
            # The previous leg got no further than this one. A random fault
            # would have advanced; this is reproducible, and retrying it just
            # burns the same hours again.
            record["outcome"] = (
                f"no progress past step {step}; failure looks deterministic, "
                "not a random fault"
            )
            legs.append(record)
            break

        record["outcome"] = f"crashed at ~{step}; resuming"
        legs.append(record)
        if journal:
            write_journal(journal, legs, train_args)
        previous_step = step
        arguments = resume_arguments(train_args, checkpoint, step)
        print(
            f"  exit {completed.returncode} after {elapsed}s; "
            f"resuming from step {step}",
            flush=True,
        )

    if journal:
        write_journal(journal, legs, train_args)
    print(f"=== giving up after {len(legs)} leg(s) ===", flush=True)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max_restarts",
        type=int,
        default=4,
        help="how many times to rejoin the curve after a crash (default 4)",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="arguments passed through to the trainer, after --",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        raise SystemExit("no training arguments given; pass them after --")
    if "--checkpoint_every_improvement" not in train_args:
        raise SystemExit(
            "supervision needs --checkpoint_every_improvement: without a "
            "recovery checkpoint there is no step to resume from, so a crash "
            "would restart the run from zero."
        )
    return run(args.max_restarts, train_args)


if __name__ == "__main__":
    raise SystemExit(main())
