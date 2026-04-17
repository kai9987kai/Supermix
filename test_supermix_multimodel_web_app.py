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
        self._store_rows = [
            {
                "file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip",
                "size_bytes": 1647669376,
                "size_mb": 1571.34,
                "family": "fusion",
                "known": True,
                "model_key": "omni_collective_v8_preview",
                "label": "Omni Collective V8 Preview",
                "kind": "omni_collective_v8",
                "capabilities": ["chat", "vision"],
                "note": "Preview snapshot.",
                "benchmark_hint": "Interim preview.",
                "download_url": "https://example.invalid/v8-preview.zip",
                "installed": False,
                "local_path": "",
                "selectable": False,
            },
            {
                "file_name": "dcgan_v2_in_progress.zip",
                "size_bytes": 61069961,
                "size_mb": 58.23,
                "family": "gan",
                "known": True,
                "model_key": "dcgan_v2_in_progress",
                "label": "DCGAN V2 CIFAR",
                "kind": "dcgan_image",
                "capabilities": ["image"],
                "note": "GAN image model.",
                "benchmark_hint": "",
                "download_url": "https://example.invalid/dcgan-v2.zip",
                "installed": True,
                "local_path": str(zip_path),
                "selectable": True,
            },
        ]
        self._jobs = [
            {
                "job_id": "store-1",
                "file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip",
                "status": "downloading",
                "downloaded_bytes": 512,
                "total_bytes": 1024,
                "started_at": "2026-04-07T18:00:00",
                "local_path": "",
                "error": "",
            }
        ]
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

    def model_store_catalog(self, force_refresh: bool = False):
        return {
            "repo_id": "Kai9987kai/supermix-model-zoo",
            "model_count": len(self._store_rows),
            "models": list(self._store_rows),
        }

    def model_store_jobs(self):
        return {"jobs": list(self._jobs)}

    def install_model_store_artifact(self, file_name: str):
        return {
            "job_id": "store-new",
            "file_name": file_name,
            "status": "queued",
            "downloaded_bytes": 0,
            "total_bytes": 2048,
            "started_at": "2026-04-07T18:01:00",
            "local_path": "",
            "error": "",
        }


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
        assert 'id="sessionObjective"' in html
        assert 'id="savedDrafts"' in html
        assert 'id="contextBankList"' in html
        assert 'id="captureLastReplyBtn"' in html
        assert 'id="threadBookmarks"' in html
        assert 'id="compareSummary"' in html
        assert 'id="dispatchPreview"' in html
        assert 'id="modelStoreList"' in html
        assert 'id="refreshStoreBtn"' in html
        assert 'id="appShell"' in html
        assert 'id="composeScroll"' in html
        assert 'id="composeQuickBtn"' in html
        assert 'id="composeMediaBtn"' in html
        assert 'id="composeWorkbenchBtn"' in html
        assert 'id="loopBudget"' in html
        assert '<option value="loop">Loop Agent</option>' in html
        assert '<option value="collective_loop">Collective + Loop</option>' in html
        assert 'id="toggleSidebarBtn"' in html
        assert 'id="toggleThreadDensityBtn"' in html
        assert 'id="responseDeck"' in html
        assert 'id="deliverableTarget"' in html
        assert 'id="successChecks"' in html
        assert 'id="riskBox"' in html
        assert 'id="confidenceMode"' in html
        assert 'id="evidenceMode"' in html
        assert 'id="clarifyMode"' in html
        assert 'id="assumptionMode"' in html
        assert 'id="refinementDeck"' in html
        assert 'id="refineLastReplyBtn"' in html
        assert 'id="challengeLastReplyBtn"' in html


def test_model_store_endpoints():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        zip_path = root / "supermix_3d_generation_micro_v1_20260403.zip"
        summary_path = root / "three_d_generation_micro_v1_summary.json"
        zip_path.write_bytes(b"zip-bytes")
        summary_path.write_bytes(b"{}")

        app = build_app(_StubManager(zip_path, summary_path))
        client = app.test_client()

        store_response = client.get("/api/model_store")
        assert store_response.status_code == 200
        store_payload = store_response.get_json()
        assert store_payload["ok"] is True
        assert store_payload["repo_id"] == "Kai9987kai/supermix-model-zoo"
        assert len(store_payload["models"]) == 2
        assert store_payload["models"][0]["file_name"] == "supermix_omni_collective_v8_preview_20260407_001155.zip"

        jobs_response = client.get("/api/model_store/jobs")
        assert jobs_response.status_code == 200
        jobs_payload = jobs_response.get_json()
        assert jobs_payload["ok"] is True
        assert jobs_payload["jobs"][0]["status"] == "downloading"

        install_response = client.post("/api/model_store/install", json={"file_name": "supermix_omni_collective_v8_preview_20260407_001155.zip"})
        assert install_response.status_code == 200
        install_payload = install_response.get_json()
        assert install_payload["ok"] is True
        assert install_payload["job"]["status"] == "queued"
