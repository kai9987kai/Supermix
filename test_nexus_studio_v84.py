"""Unit tests for NexusMind Studio v84 HTML and API integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "source"))

import pytest
import nexus_api as api


def test_studio_html_contains_v84_innovations():
    studio_file = Path(__file__).parent / "web_static" / "nexus_studio.html"
    assert studio_file.is_file()
    html = studio_file.read_text(encoding="utf-8")

    # Title check
    assert "NexusMind Studio v84" in html

    # Tab headers
    assert "Quantum Density" in html
    assert "Soliton Collider" in html
    assert "Cognitive Trajectory" in html
    assert "Speculative Tree" in html

    # Panel IDs
    assert 'id="panel-quantum-state"' in html
    assert 'id="panel-gliders"' in html
    assert 'id="panel-trajectory"' in html
    assert 'id="panel-speculative-tree"' in html

    # JavaScript functions and canvas elements
    assert "runQuantumState" in html
    assert "drawQuantumCanvas" in html
    assert "runGliders" in html
    assert "drawGliderCanvas" in html
    assert "runTrajectory" in html
    assert "drawTrajectoryCanvas" in html
    assert "runSpeculativeTree" in html
    assert "drawSpecTreeCanvas" in html
    assert 'id="quantumCanvas"' in html
    assert 'id="gliderCanvas"' in html
    assert 'id="trajectoryCanvas"' in html
    assert 'id="specTreeCanvas"' in html


def test_studio_endpoint_serves_v84():
    import warnings
    from starlette.testclient import TestClient

    app = api.create_app(api.NexusApiService())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(app)

    resp = client.get("/studio")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "NexusMind Studio v84" in resp.text
    assert "panel-quantum-state" in resp.text
    assert "panel-speculative-tree" in resp.text
