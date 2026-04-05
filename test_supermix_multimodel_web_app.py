import os
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(), "source"))

from supermix_multimodel_web_app import build_app


class _StubManager:
    def __init__(self, zip_path: Path, summary_path: Path):
        self.records = []
        self.generated_dir = zip_path.parent
        self.uploads_dir = zip_path.parent
        self._payload = {
            "key": "three_d_generation_micro_v1",
            "label": "3D Generation Micro",
            "zip_path": str(zip_path),
            "zip_name": zip_path.name,
            "zip_size_bytes": zip_path.stat().st_size,
            "summary_path": str(summary_path),
            "summary_name": summary_path.name,
            "parameter_count": 35886,
            "train_accuracy": 1.0,
            "val_accuracy": 1.0,
            "concept_count": 14,
            "source_rows": 144,
            "train_rows": 130,
            "val_rows": 14,
            "concept_labels": ["pyramid", "tetrahedron"],
            "sample_predictions": [
                {
                    "prompt": "Create a square pyramid.",
                    "predicted_label": "square pyramid",
                    "confidence": 0.98,
                }
            ],
        }

    def three_d_model_view(self):
        return dict(self._payload)


def test_three_d_model_view_endpoint_and_downloads():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_bytes = b"zip-bytes"
        summary_bytes = b'{"artifact":"supermix_3d_generation_micro_v1_20260403.zip"}'
        zip_path.write_bytes(zip_bytes)
        summary_path.write_bytes(summary_bytes)

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        response = client.get("/api/three_d_model_view")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["ok"] is True
        assert payload["model"]["key"] == "three_d_generation_micro_v1"
        assert payload["model"]["download_zip_url"] == "/download/three_d_model_zip"
        assert payload["model"]["download_summary_url"] == "/download/three_d_model_summary"

        zip_response = client.get("/download/three_d_model_zip")
        assert zip_response.status_code == 200
        assert zip_response.data == zip_bytes
        zip_response.close()

        summary_response = client.get("/download/three_d_model_summary")
        assert summary_response.status_code == 200
        assert summary_response.data == summary_bytes
        summary_response.close()


def test_index_contains_discovery_ui():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'id="modelSearch"' in html
        assert 'id="capabilityFilter"' in html
        assert 'id="quickPickChips"' in html
        assert 'id="discoveryNote"' in html
