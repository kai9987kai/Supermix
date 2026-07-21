"""Synchronize the self-contained runtime model-variant snapshot.

The packaged ``runtime_python`` surface must not import back into ``source``:
that works in a checkout but fails as soon as the runtime directory is copied or
shipped on its own. This helper keeps the snapshot mechanical and makes drift a
cheap CI/test check.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "source" / "model_variants.py"
RUNTIME_PATH = REPO_ROOT / "runtime_python" / "model_variants.py"


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def snapshot_status() -> tuple[bool, str]:
    source_payload = SOURCE_PATH.read_bytes()
    runtime_payload = RUNTIME_PATH.read_bytes() if RUNTIME_PATH.exists() else b""
    matches = source_payload == runtime_payload
    detail = (
        f"source={_digest(source_payload)[:12]} "
        f"runtime={_digest(runtime_payload)[:12] if runtime_payload else 'missing'} "
        f"bytes={len(source_payload)}"
    )
    return matches, detail


def sync_snapshot() -> str:
    source_payload = SOURCE_PATH.read_bytes()
    RUNTIME_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RUNTIME_PATH.with_suffix(".py.tmp")
    temporary_path.write_bytes(source_payload)
    temporary_path.replace(RUNTIME_PATH)
    return f"synced {RUNTIME_PATH.relative_to(REPO_ROOT)} ({len(source_payload)} bytes)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when runtime_python/model_variants.py differs from the source snapshot.",
    )
    args = parser.parse_args()

    matches, detail = snapshot_status()
    if args.check:
        if matches:
            print(f"runtime model-variant snapshot is current: {detail}")
            return 0
        print(f"runtime model-variant snapshot is stale: {detail}")
        return 1

    if matches:
        print(f"runtime model-variant snapshot already current: {detail}")
        return 0
    print(sync_snapshot())
    matches, detail = snapshot_status()
    if not matches:
        print(f"runtime model-variant snapshot verification failed: {detail}")
        return 1
    print(f"runtime model-variant snapshot verified: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
