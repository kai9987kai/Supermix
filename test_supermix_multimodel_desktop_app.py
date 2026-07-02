import os
import sys

sys.path.append(os.path.join(os.getcwd(), "source"))

import supermix_multimodel_desktop_app as desktop_app


def test_resolve_models_dir_defaults_to_state_directory(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    models_dir = desktop_app.resolve_models_dir("", state_dir)

    assert models_dir == (state_dir / "models").resolve()
    assert models_dir.exists()


def test_hydrate_bundled_models_copies_missing_zips(tmp_path, monkeypatch):
    bundled_dir = tmp_path / "bundled_models"
    bundled_dir.mkdir()
    source_zip = bundled_dir / "sample_model.zip"
    source_zip.write_bytes(b"zip-payload")

    target_dir = tmp_path / "runtime_models"

    monkeypatch.setattr(desktop_app, "resolve_runtime_path", lambda _: bundled_dir)

    result_dir = desktop_app.hydrate_bundled_models(target_dir)

    copied = target_dir / "sample_model.zip"
    assert result_dir == target_dir
    assert copied.exists()
    assert copied.read_bytes() == b"zip-payload"
