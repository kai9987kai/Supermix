"""Tests for the multi-model chat server.

The single-model app had, by grep, zero streaming, queueing or cancellation
primitives: every request took a global lock and held it for the whole
generation, so a second caller waited with no feedback. This server adds model
switching, bounded residency, streaming and admission control, and these tests
pin the parts that fail quietly if they regress -- eviction (memory), refusal
(availability) and spec validation (a typo must not become a 500).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
for candidate in (REPO_ROOT, SOURCE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import supermix_chat_server as server  # noqa: E402


# -- spec parsing -----------------------------------------------------------


def test_model_spec_accepts_name_equals_path(tmp_path):
    ckpt = tmp_path / "m.pt"
    ckpt.write_bytes(b"x")

    assert server.parse_model_spec([f"a={ckpt}"]) == {"a": str(ckpt)}


def test_model_spec_rejects_missing_equals():
    with pytest.raises(ValueError, match="name=path"):
        server.parse_model_spec(["justapath.pt"])


def test_model_spec_rejects_a_checkpoint_that_does_not_exist():
    """A typo must fail at startup, not on the first user request."""

    with pytest.raises(ValueError, match="not found"):
        server.parse_model_spec(["a=/no/such/checkpoint.pt"])


def test_model_spec_requires_at_least_one():
    with pytest.raises(ValueError, match="at least one"):
        server.parse_model_spec([])


# -- registry residency -----------------------------------------------------


class _FakeRegistry(server.ModelRegistry):
    """Registry with checkpoint loading stubbed, so eviction can be tested
    without nine real models in memory."""

    def __init__(self, names, max_resident):
        super().__init__({n: f"/fake/{n}.pt" for n in names}, max_resident=max_resident)
        self.loads = []

    def acquire(self, name):
        if name not in self.spec:
            raise KeyError(f"unknown model {name!r}")
        with self._guard:
            existing = self._resident.get(name)
            if existing is not None:
                self._resident.move_to_end(name)
                return existing
        self.loads.append(name)
        loaded = server.LoadedModel(
            name=name, checkpoint=self.spec[name], model=object(),
            tokenizer=object(), extra={},
        )
        with self._guard:
            self._resident[name] = loaded
            self._resident.move_to_end(name)
            while len(self._resident) > self.max_resident:
                evicted, _ = self._resident.popitem(last=False)
                if evicted == name:
                    self._resident[name] = loaded
                    break
        return loaded


def test_registry_evicts_least_recently_used():
    """Holding every checkpoint at once is how the v64 run met a segfault."""

    registry = _FakeRegistry(["a", "b", "c"], max_resident=2)

    registry.acquire("a")
    registry.acquire("b")
    registry.acquire("c")

    assert set(registry._resident) == {"b", "c"}


def test_registry_keeps_a_model_it_just_loaded():
    """With max_resident=1 the new arrival must survive its own eviction pass."""

    registry = _FakeRegistry(["a", "b"], max_resident=1)

    registry.acquire("a")
    registry.acquire("b")

    assert set(registry._resident) == {"b"}


def test_registry_reuses_a_resident_model():
    registry = _FakeRegistry(["a", "b"], max_resident=2)

    registry.acquire("a")
    registry.acquire("a")

    assert registry.loads == ["a"]


def test_recent_use_protects_a_model_from_eviction():
    registry = _FakeRegistry(["a", "b", "c"], max_resident=2)

    registry.acquire("a")
    registry.acquire("b")
    registry.acquire("a")   # refresh a
    registry.acquire("c")   # should evict b, not a

    assert set(registry._resident) == {"a", "c"}


def test_unknown_model_raises_rather_than_loading():
    registry = _FakeRegistry(["a"], max_resident=2)

    with pytest.raises(KeyError, match="unknown model"):
        registry.acquire("nope")


def test_describe_lists_every_configured_model():
    registry = _FakeRegistry(["a", "b"], max_resident=1)
    registry.acquire("a")

    described = {entry["name"]: entry for entry in registry.describe()}

    assert set(described) == {"a", "b"}
    assert described["a"]["resident"] is True
    assert described["b"]["resident"] is False


# -- admission control ------------------------------------------------------


def test_bounded_semaphore_refuses_past_the_limit():
    """The property the 503 path depends on: non-blocking acquire fails fast."""

    admission = threading.BoundedSemaphore(2)

    assert admission.acquire(blocking=False)
    assert admission.acquire(blocking=False)
    assert not admission.acquire(blocking=False)

    admission.release()
    assert admission.acquire(blocking=False)


def test_per_model_lock_is_not_shared_between_models():
    """Two different models must be able to generate concurrently; the old
    server serialised everything behind one global lock."""

    registry = _FakeRegistry(["a", "b"], max_resident=2)

    first = registry.acquire("a")
    second = registry.acquire("b")

    assert first.lock is not second.lock
