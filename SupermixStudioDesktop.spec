# -*- mode: python ; coding: utf-8 -*-
"""Portable PyInstaller template for a prepared Supermix Studio build tree.

The normal build script stages models and a base model under ``build/`` before
invoking PyInstaller.  Environment variables can override those locations for
release builders without embedding a developer machine path in this file.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


repo_root = Path(SPECPATH).resolve()


def data_if_present(path, destination, *, required=False):
    candidate = Path(path).expanduser().resolve()
    if candidate.exists():
        return [(str(candidate), destination)]
    if required:
        raise FileNotFoundError(f"required Studio packaging input is missing: {candidate}")
    return []


models_stage = os.environ.get(
    "SUPERMIX_STUDIO_MODELS_STAGE",
    str(repo_root / "build" / "studio_models_stage"),
)
base_model_stage = os.environ.get(
    "SUPERMIX_STUDIO_BASE_MODEL_STAGE",
    str(repo_root / "build" / "studio_base_model_stage"),
)
benchmark_summary = os.environ.get("SUPERMIX_STUDIO_BENCHMARK_SUMMARY", "")
bundle_manifest = os.environ.get(
    "SUPERMIX_STUDIO_BUNDLE_MANIFEST",
    str(repo_root / "output" / "supermix_studio_bundled_models_manifest.json"),
)
runtime_manifest = repo_root / "source" / "studio_runtime_manifest.json"

datas = [(str(repo_root / "assets"), "assets")]
datas += data_if_present(models_stage, "bundled_models", required=True)
datas += data_if_present(base_model_stage, "bundled_base_model", required=True)
if benchmark_summary:
    datas += data_if_present(benchmark_summary, "output", required=True)
datas += data_if_present(bundle_manifest, "output", required=True)
datas += data_if_present(runtime_manifest, "output", required=True)

binaries = []
hiddenimports = []
for package in (
    "webview",
    "flask",
    "werkzeug",
    "PIL",
    "sympy",
    "mpmath",
    "safetensors",
    "transformers",
    "peft",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    [str(repo_root / "source" / "supermix_multimodel_desktop_app.py")],
    pathex=[str(repo_root / "source")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SupermixStudioDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(repo_root / "assets" / "supermix_qwen_icon.ico")],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="SupermixStudioDesktop",
)
