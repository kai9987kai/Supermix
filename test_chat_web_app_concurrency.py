"""Concurrent requests must not lose turns or interleave model telemetry.

`Engine.chat` snapshots the session history, runs inference outside the engine
lock, then appends the finished turn. Without per-session serialization two
concurrent requests for the same session both snapshot the same history and the
earlier turn is silently dropped, which also corrupts the conversation state
derived from that history.

Separately, the recursive heads publish their diagnostics as attributes on
themselves (`last_cycles_used`, `last_router_z_loss`, ...). Concurrent forwards
would therefore report each other's metrics, so inference is serialized too.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RUNTIME = ROOT / "runtime_python"


@pytest.fixture(scope="module")
def web():
    sys.path.insert(0, str(SOURCE))
    try:
        spec = importlib.util.spec_from_file_location(
            "concurrency_chat_web_app", SOURCE / "chat_web_app.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["concurrency_chat_web_app"] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(SOURCE))


def _engine(web):
    return web.Engine(
        device="cpu",
        device_info={"device": "cpu"},
        defaults=web._normalize_runtime_compute_defaults(
            web._library_runtime_compute_defaults()
        ),
    )


def test_engine_exposes_both_locks(web) -> None:
    engine = _engine(web)
    assert isinstance(engine.inference_lock, type(threading.Lock()))
    assert engine.session_turn_locks == {}


def test_same_session_gets_one_lock_and_different_sessions_do_not_share(web) -> None:
    engine = _engine(web)
    first = engine._session_turn_lock("s1")
    again = engine._session_turn_lock("s1")
    other = engine._session_turn_lock("s2")

    assert first is again, "a session must reuse its lock or turns are not serialized"
    assert first is not other, "separate sessions must not block each other"


def test_session_lock_table_is_bounded(web) -> None:
    engine = _engine(web)
    for index in range(web.MAX_SESSION_TURN_LOCKS + 64):
        engine._session_turn_lock(f"session-{index}")

    assert len(engine.session_turn_locks) <= web.MAX_SESSION_TURN_LOCKS + 1


def test_a_held_session_lock_is_never_evicted(web) -> None:
    engine = _engine(web)
    held = engine._session_turn_lock("in-flight")
    held.acquire()
    try:
        for index in range(web.MAX_SESSION_TURN_LOCKS + 64):
            engine._session_turn_lock(f"filler-{index}")
        assert engine.session_turn_locks.get("in-flight") is held
    finally:
        held.release()


def test_session_lock_actually_serializes_concurrent_turns(web) -> None:
    """Two threads on one session must not overlap inside the critical region."""

    engine = _engine(web)
    overlaps = []
    inside = threading.Semaphore(0)
    active = 0
    guard = threading.Lock()

    def turn() -> None:
        nonlocal active
        with engine._session_turn_lock("shared"):
            with guard:
                active += 1
                if active > 1:
                    overlaps.append(True)
            inside.release()
            # Hold the region long enough that a racing thread would overlap.
            threading.Event().wait(0.05)
            with guard:
                active -= 1

    threads = [threading.Thread(target=turn) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads), "session lock deadlocked"
    assert not overlaps, "two turns ran concurrently on one session"


def test_inference_is_serialized_in_both_chat_and_sweep(web) -> None:
    """Guard the placement, not just the presence, of the inference lock."""

    source = (SOURCE / "chat_web_app.py").read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Engine"
    )

    for method_name in ("chat", "compute_sweep"):
        method = next(
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        forwards = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr
            in {"forward_with_runtime_compute", "progressive_auto_compute_forward"}
        ]
        assert forwards, f"{method_name} no longer runs a model forward"

        locked_regions = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.With)
            and any(
                "inference_lock" in ast.unparse(item.context_expr) for item in node.items
            )
        ]
        assert locked_regions, f"{method_name} runs inference without the inference lock"

        covered = {
            id(call)
            for region in locked_regions
            for call in ast.walk(region)
            if isinstance(call, ast.Call)
        }
        uncovered = [
            ast.unparse(call.func) for call in forwards if id(call) not in covered
        ]
        assert not uncovered, f"{method_name} has unlocked forwards: {uncovered}"


def test_source_and_packaged_engines_agree_on_concurrency() -> None:
    """The packaged runtime must carry the same fix, not just the source tree."""

    def engine_method(path: Path, name: str) -> str:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        engine = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Engine"
        )
        method = next(
            node
            for node in engine.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        return ast.dump(method, include_attributes=False)

    for name in ("__init__", "_session_turn_lock", "chat", "compute_sweep"):
        assert engine_method(SOURCE / "chat_web_app.py", name) == engine_method(
            RUNTIME / "chat_web_app.py", name
        ), f"Engine.{name} differs between source and packaged runtime"
