"""Unit tests for NexusMind Studio v83 HTML and API integration."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "source"))

import nexus_api as api


def test_studio_html_contains_v83_innovations():
    studio_file = Path(__file__).parent / "web_static" / "nexus_studio.html"
    assert studio_file.is_file()
    html = studio_file.read_text(encoding="utf-8")

    # Later Studio versions must preserve the v83 surfaces without freezing the
    # global page title to an obsolete release number.
    assert "<title>NexusMind Studio v" in html

    # Tab headers
    assert "Compare Bench" in html
    assert "Quantum Bell" in html
    assert "Semantic Resonance" in html

    # Panel IDs
    assert 'id="panel-compare"' in html
    assert 'id="panel-bell"' in html
    assert 'id="panel-resonance"' in html

    # JavaScript functions and canvas elements
    assert "runCompare" in html
    assert "autoLoopStart" in html
    assert "runBellExperiment" in html
    assert "drawBellCanvas" in html
    assert "runResonance" in html
    assert "drawResonanceRadar" in html
    assert 'id="bellCanvas"' in html
    assert 'id="resonanceCanvas"' in html


def test_studio_endpoint_serves_v83(tmp_path):
    import warnings
    from starlette.testclient import TestClient

    app = api.create_app(api.NexusApiService())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = TestClient(app)

    resp = client.get("/studio")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "<title>NexusMind Studio v" in resp.text
    assert "panel-compare" in resp.text
