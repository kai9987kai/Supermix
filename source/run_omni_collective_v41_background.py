from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from pathlib import Path
from typing import Iterable, TextIO

import train_omni_collective_v41


class TeeStream(io.TextIOBase):
    def __init__(self, streams: Iterable[TextIO]) -> None:
        super().__init__()
        self._streams = [stream for stream in streams if stream is not None]

    def write(self, s: str) -> int:  # pragma: no cover
        text = str(s)
        for stream in self._streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:  # pragma: no cover
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:  # pragma: no cover
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background launcher for omni_collective_v41 with in-process log teeing.")
    parser.add_argument("--out-log", required=True)
    parser.add_argument("--err-log", required=True)
    parser.add_argument("--worker-pid-file", required=True)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_log).resolve()
    err_path = Path(args.err_log).resolve()
    pid_path = Path(args.worker_pid_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    err_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    out_handle = out_path.open("a", encoding="utf-8", buffering=1, newline="\n")
    err_handle = err_path.open("a", encoding="utf-8", buffering=1, newline="\n")
    pid_path.write_text(str(os.getpid()), encoding="ascii")

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream([original_stdout, out_handle])
    sys.stderr = TeeStream([original_stderr, err_handle])
    try:
        train_args = list(args.train_args)
        if train_args and train_args[0] == "--":
            train_args = train_args[1:]
        sys.argv = ["train_omni_collective_v41.py"] + train_args
        train_omni_collective_v41.main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            out_handle.close()
            err_handle.close()


if __name__ == "__main__":
    main()
