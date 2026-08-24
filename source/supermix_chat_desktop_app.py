"""Desktop wrapper around the Supermix chat server with v74 built in.

`supermix_chat_server.py` is a Flask app that expects a checkpoint path on the
command line. That is right for development and wrong for a desktop build,
where there is no command line and the model has to be *inside* the
application. This module is the difference between those two situations and
nothing else: it resolves the bundled checkpoint, starts the same server
in-process, and shows it in a window.

The server itself is unmodified, so anything true of the web interface -- the
prompt normaliser, the independent answer check, bounded concurrency -- is true
here.

## Packaging notes

**The checkpoint travels with the executable.** PyInstaller unpacks bundled
data to `sys._MEIPASS`, so `bundled_checkpoint()` looks there first and falls
back to the repository layout when running from source. A build that silently
found the developer's `output/` directory would work on this machine and fail
on every other one.

**The recall corpus is deliberately not bundled.** It is 101.6 MB against the
checkpoint's 33 MB, and it powers one diagnostic -- the verbatim meter. The
answer check, which is the part that matters for trusting a reply, needs no
corpus and is always on.

**The port is chosen at runtime.** A fixed port fails on the second launch and
on any machine already using it.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

APP_NAME = "Supermix Chat"
MODEL_NAME = "v74"
#: Filename of the checkpoint as it is staged into the bundle.
BUNDLED_CHECKPOINT = "supermix_v74.pt"


def _source_dir() -> Path:
    return Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Where bundled data lives, frozen or not."""

    if is_frozen():
        # PyInstaller unpacks datas here; onedir builds set it to the app dir.
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return _source_dir().parent


def bundled_checkpoint() -> Path:
    """The model shipped with this application.

    Ordered most-specific first. The repository fallbacks exist so the desktop
    app can be run from a source checkout during development; in a frozen build
    the first candidate is the only one that should ever match.
    """

    root = bundle_root()
    candidates = [
        root / "model" / BUNDLED_CHECKPOINT,
        root / BUNDLED_CHECKPOINT,
        root / "output" / "v74_broad" / "v74_broad.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"no bundled checkpoint found. Looked for: "
        + ", ".join(str(c) for c in candidates)
    )


def free_port() -> int:
    """An ephemeral port the OS says is free.

    There is an unavoidable race between closing this socket and the server
    binding it. Binding to port 0 inside the server would remove the race but
    the port would then be unknowable until after startup, which is worse: the
    window has to be pointed somewhere.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def wait_until_ready(url: str, timeout: float = 120.0) -> bool:
    """Poll until the server answers, or give up.

    Loading an 8.6M-parameter checkpoint on a cold CPU takes a few seconds;
    on a slow disk with an antivirus scanning a freshly extracted bundle it
    can take considerably longer, hence the generous ceiling.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def start_server(checkpoint: Path, port: int, threads: int) -> threading.Thread:
    """Run the chat server in this process, on a daemon thread."""

    if str(_source_dir()) not in sys.path:
        sys.path.insert(0, str(_source_dir()))

    import torch

    # A desktop machine is doing other things. Leaving torch to claim every
    # core makes the whole system stutter while a reply generates.
    torch.set_num_threads(max(1, threads))

    import supermix_chat_server as server

    registry = server.ModelRegistry(
        {MODEL_NAME: str(checkpoint)},
        max_resident=1,
        corpora={},  # see module docstring: the recall corpus is not bundled
    )
    app = server.build_app(registry, max_concurrency=1, normalise_prompts=True)

    def serve() -> None:
        # `threaded=True` so the SSE stream and the model-list poll the UI
        # makes after every reply are not serialised behind each other.
        app.run(host="127.0.0.1", port=port, threaded=True,
                debug=False, use_reloader=False)

    thread = threading.Thread(target=serve, name="supermix-chat-server", daemon=True)
    thread.start()
    return thread


def open_window(url: str) -> bool:
    """Show the interface in a native window. False if that is not possible."""

    try:
        import webview
    except ImportError:
        return False
    try:
        webview.create_window(APP_NAME, url, width=1100, height=820,
                              min_size=(680, 560))
        webview.start()
        return True
    except Exception:  # noqa: BLE001 - any windowing failure falls back
        logging.exception("native window failed; falling back to a browser")
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=0,
                        help="port to serve on; 0 picks a free one")
    parser.add_argument("--browser", action="store_true",
                        help="open in the default browser instead of a window")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="torch threads; defaults to half the machine's cores")
    parser.add_argument("--checkpoint", default=None,
                        help="override the bundled model (development use)")
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        checkpoint = Path(args.checkpoint) if args.checkpoint else bundled_checkpoint()
    except FileNotFoundError as error:
        print(f"{APP_NAME}: {error}", file=sys.stderr)
        return 2

    port = args.port or free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"{APP_NAME}: loading {checkpoint.name}")
    start_server(checkpoint, port, args.threads)

    if not wait_until_ready(f"{url}/api/models"):
        print(f"{APP_NAME}: the server did not become ready", file=sys.stderr)
        return 1
    print(f"{APP_NAME}: ready at {url}")

    if args.browser or not open_window(url):
        import webbrowser

        webbrowser.open(url)
        print(f"{APP_NAME}: serving at {url} -- close this window to quit")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
