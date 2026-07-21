import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parent / "source"
sys.path.insert(0, str(SOURCE_DIR))

import multimodel_runtime
import supermix_multimodel_web_app
from chat_web_app import Engine, build_app
from route_policy_shadow_registry import RouteShadowAssignmentRegistry


class _MemoryStoreStub:
    def __init__(self, path: Path):
        self.path = path


class _RouteLedgerStub:
    def __init__(self, path: Path):
        self.path = path


def _build_manager(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(multimodel_runtime, "configure_torch_runtime", lambda **_kwargs: None)
    monkeypatch.setattr(
        multimodel_runtime,
        "resolve_device",
        lambda *_args, **_kwargs: (torch.device("cpu"), {"resolved": "cpu"}),
    )
    monkeypatch.setattr(multimodel_runtime, "ConversationMemoryStore", _MemoryStoreStub)
    monkeypatch.setattr(multimodel_runtime, "RoutePolicyLedger", _RouteLedgerStub)
    extraction_root = tmp_path / "tmp" / "ext"
    return multimodel_runtime.UnifiedModelManager(
        records=(),
        extraction_root=extraction_root,
        generated_dir=tmp_path / "tmp" / "gen",
        models_dir=tmp_path / "models",
        common_summary_path=tmp_path / "common.json",
    )


def test_unified_manager_uses_canonical_memory_path_without_creating_registry(
    monkeypatch, tmp_path
):
    manager = _build_manager(monkeypatch, tmp_path)
    expected = (tmp_path / "tmp" / "memory" / "route-policy-shadow-registry.sqlite3").resolve()

    assert manager.route_shadow_registry_path == expected
    assert not expected.exists()
    snapshot = manager.route_shadow_registry_snapshot()

    assert snapshot == {
        "ok": True,
        "available": False,
        "status": "not_initialized",
        "registry_location": "memory/route-policy-shadow-registry.sqlite3",
        "read_only": True,
        "campaign_count": 0,
        "campaigns": [],
        "event_chain": None,
        "execution_enabled": False,
        "activation_available": False,
        "automatic_promotion_allowed": False,
    }
    assert not expected.exists()


def test_unified_manager_snapshot_uses_registry_read_only_mode(monkeypatch, tmp_path):
    manager = _build_manager(monkeypatch, tmp_path)
    RouteShadowAssignmentRegistry(manager.route_shadow_registry_path)
    before = manager.route_shadow_registry_path.read_bytes()

    snapshot = manager.route_shadow_registry_snapshot()
    after = manager.route_shadow_registry_path.read_bytes()

    assert snapshot["ok"] is True
    assert snapshot["available"] is True
    assert snapshot["status"] == "verified"
    assert snapshot["read_only"] is True
    assert snapshot["campaign_count"] == 0
    assert before == after


def test_unified_manager_caches_verified_snapshot_until_durable_state_changes(
    monkeypatch, tmp_path
):
    manager = _build_manager(monkeypatch, tmp_path)
    RouteShadowAssignmentRegistry(manager.route_shadow_registry_path)
    calls = {"count": 0}
    original_snapshot = RouteShadowAssignmentRegistry.snapshot

    def counted_snapshot(registry, *args, **kwargs):
        calls["count"] += 1
        return original_snapshot(registry, *args, **kwargs)

    monkeypatch.setattr(RouteShadowAssignmentRegistry, "snapshot", counted_snapshot)

    first = manager.route_shadow_registry_snapshot()
    first["campaigns"].append({"cache_poison": True})
    second = manager.route_shadow_registry_snapshot()

    assert calls["count"] == 1
    assert second["campaigns"] == []

    registry_stat = manager.route_shadow_registry_path.stat()
    newer = max(time.time_ns(), registry_stat.st_mtime_ns + 1_000_000_000)
    os.utime(manager.route_shadow_registry_path, ns=(newer, newer))
    third = manager.route_shadow_registry_snapshot()

    assert calls["count"] == 2
    assert third["ok"] is True
    assert third["read_only"] is True


class _CanonicalWebManager:
    def route_shadow_registry_snapshot(self):
        return {
            "ok": True,
            "available": False,
            "status": "not_initialized",
            "registry_location": "memory/route-policy-shadow-registry.sqlite3",
            "read_only": True,
            "campaign_count": 0,
            "campaigns": [],
            "event_chain": None,
            "execution_enabled": False,
            "activation_available": False,
            "automatic_promotion_allowed": False,
        }


class _FailingCanonicalWebManager:
    def route_shadow_registry_snapshot(self):
        raise RuntimeError("registry audit failed")


def test_canonical_studio_registry_api_is_get_only_and_ui_is_read_only():
    app = supermix_multimodel_web_app.build_app(_CanonicalWebManager())
    client = app.test_client()

    response = client.get("/api/route_shadow_registry/status")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.get_json()["route_shadow_registry"]["read_only"] is True
    assert client.post("/api/route_shadow_registry/status", json={}).status_code == 405

    html = client.get("/").get_data(as_text=True)
    assert 'id="routeShadowRegistryRefresh"' in html
    assert "/api/route_shadow_registry/status" in html
    assert "Shadow assignment registry - read only" in html
    assert "seal campaign" not in html.lower()
    assert "reveal seed" not in html.lower()


def test_canonical_studio_registry_error_is_not_cached():
    app = supermix_multimodel_web_app.build_app(_FailingCanonicalWebManager())
    response = app.test_client().get("/api/route_shadow_registry/status")

    assert response.status_code == 500
    assert response.headers["Cache-Control"] == "no-store"
    assert response.get_json() == {"ok": False, "error": "registry audit failed"}


def test_canonical_studio_defaults_to_loopback_binding():
    parser = supermix_multimodel_web_app.build_arg_parser()

    assert parser.parse_args([]).host == "127.0.0.1"
    assert parser.parse_args(["--host", "0.0.0.0"]).host == "0.0.0.0"


def test_source_legacy_browser_registry_status_is_get_only(tmp_path):
    registry_path = tmp_path / "missing-shadow.sqlite3"
    engine = Engine(
        torch.device("cpu"),
        {"resolved": "cpu"},
        {"pool_mode": "all", "route_shadow_registry_path": str(registry_path)},
    )
    client = build_app(engine, "weights.pth", "meta.json").test_client()

    response = client.get("/api/route_shadow_registry/status")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.get_json()["route_shadow_registry"]
    assert payload["available"] is False
    assert payload["read_only"] is True
    assert not registry_path.exists()
    assert client.post("/api/route_shadow_registry/status", json={}).status_code == 405
    assert "Shadow registry - read only" in client.get("/").get_data(as_text=True)


def test_packaged_legacy_browser_registry_status_is_get_only(tmp_path):
    runtime_dir = Path("runtime_python").resolve()
    script = f"""
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, {str(runtime_dir)!r})
import chat_web_app

registry_path = Path({str(tmp_path / 'missing-runtime-shadow.sqlite3')!r})
engine = chat_web_app.Engine(
    torch.device('cpu'),
    {{'resolved': 'cpu'}},
    {{'pool_mode': 'all', 'route_shadow_registry_path': str(registry_path)}},
)
client = chat_web_app.build_app(engine, 'weights.pth', 'meta.json').test_client()
response = client.get('/api/route_shadow_registry/status')
print(json.dumps({{
    'status_code': response.status_code,
    'cache_control': response.headers.get('Cache-Control'),
    'payload': response.get_json(),
    'post_status': client.post('/api/route_shadow_registry/status', json={{}}).status_code,
    'created': registry_path.exists(),
    'ui': 'Shadow registry - read only' in client.get('/').get_data(as_text=True),
}}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["status_code"] == 200
    assert payload["cache_control"] == "no-store"
    assert payload["payload"]["route_shadow_registry"]["available"] is False
    assert payload["payload"]["route_shadow_registry"]["read_only"] is True
    assert payload["post_status"] == 405
    assert payload["created"] is False
    assert payload["ui"] is True
