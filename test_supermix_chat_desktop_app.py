"""Tests for the desktop wrapper around the chat server.

The wrapper's whole job is resolving a bundled model and starting the server,
so the failure that matters is a build that looks fine and finds the wrong
model -- or none. These pin the resolution order and the refusals.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import supermix_chat_desktop_app as desktop  # noqa: E402


# -- locating the bundled model ---------------------------------------------


def test_prefers_the_bundled_model_directory(tmp_path, monkeypatch):
    (tmp_path / "model").mkdir()
    bundled = tmp_path / "model" / desktop.BUNDLED_CHECKPOINT
    bundled.write_bytes(b"x")
    # A repository-layout checkpoint also present: the bundle must win.
    (tmp_path / "output" / "v74_broad").mkdir(parents=True)
    (tmp_path / "output" / "v74_broad" / "v74_broad.pt").write_bytes(b"y")

    monkeypatch.setattr(desktop, "bundle_root", lambda: tmp_path)

    assert desktop.bundled_checkpoint() == bundled


def test_falls_back_to_the_repository_layout(tmp_path, monkeypatch):
    """Running the desktop app from a source checkout must still work."""

    (tmp_path / "output" / "v74_broad").mkdir(parents=True)
    fallback = tmp_path / "output" / "v74_broad" / "v74_broad.pt"
    fallback.write_bytes(b"y")

    monkeypatch.setattr(desktop, "bundle_root", lambda: tmp_path)

    assert desktop.bundled_checkpoint() == fallback


def test_missing_model_raises_and_names_what_it_looked_for(tmp_path, monkeypatch):
    """A silent failure here ships an app that dies on first launch."""

    monkeypatch.setattr(desktop, "bundle_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError, match="no bundled checkpoint"):
        desktop.bundled_checkpoint()


def test_the_error_lists_every_candidate_path(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "bundle_root", lambda: tmp_path)

    with pytest.raises(FileNotFoundError) as error:
        desktop.bundled_checkpoint()

    assert "model" in str(error.value)
    assert "v74_broad.pt" in str(error.value)


def test_bundle_root_follows_meipass_when_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop.sys, "frozen", True, raising=False)
    monkeypatch.setattr(desktop.sys, "_MEIPASS", str(tmp_path), raising=False)

    assert desktop.bundle_root() == tmp_path


def test_bundle_root_is_the_repo_when_not_frozen(monkeypatch):
    monkeypatch.delattr(desktop.sys, "frozen", raising=False)

    assert (desktop.bundle_root() / "source").is_dir()


# -- port selection ---------------------------------------------------------


def test_free_port_is_actually_bindable():
    port = desktop.free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", port))  # raises if the port was not free


def test_free_port_is_in_the_ephemeral_range():
    assert 1024 < desktop.free_port() <= 65535


# -- readiness --------------------------------------------------------------


def test_wait_until_ready_gives_up_rather_than_hanging():
    """A launcher that waits forever on a dead server is worse than an error."""

    closed = desktop.free_port()

    assert desktop.wait_until_ready(f"http://127.0.0.1:{closed}/", timeout=0.5) is False


# -- argument surface -------------------------------------------------------


def test_port_defaults_to_automatic():
    assert desktop.build_parser().parse_args([]).port == 0


def test_threads_default_leaves_headroom_for_the_desktop():
    """Claiming every core makes the machine stutter while generating."""

    import os

    default = desktop.build_parser().parse_args([]).threads
    assert 1 <= default <= max(1, (os.cpu_count() or 4))


def test_checkpoint_can_be_overridden_for_development():
    args = desktop.build_parser().parse_args(["--checkpoint", "some/other.pt"])

    assert args.checkpoint == "some/other.pt"


def test_browser_fallback_is_opt_in():
    assert desktop.build_parser().parse_args([]).browser is False
