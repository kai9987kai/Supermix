# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Supermix Chat desktop app, with v74 built in.

Run `build_chat_desktop_exe.ps1` rather than invoking this directly -- it
stages the checkpoint into `build/chat_model_stage/` first, which this spec
then requires. Requiring it (rather than quietly skipping) is deliberate: a
build that produced an executable with no model in it would look successful
and fail on first launch.

Size note: torch is ~448 MB installed and dominates the output. The model
itself is 33 MB. The 101.6 MB recall corpus is **not** bundled -- see the
desktop app's module docstring.
"""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

repo_root = Path(SPECPATH).resolve()
source_dir = repo_root / "source"

model_stage = Path(os.environ.get(
    "SUPERMIX_CHAT_MODEL_STAGE",
    str(repo_root / "build" / "chat_model_stage"),
)).resolve()

if not model_stage.is_dir() or not any(model_stage.glob("*.pt")):
    raise FileNotFoundError(
        f"no checkpoint staged in {model_stage}. Run build_chat_desktop_exe.ps1, "
        "which stages it, or set SUPERMIX_CHAT_MODEL_STAGE."
    )

# The app looks for its model under `model/` inside the bundle.
datas = [(str(model_stage), "model")]

icon_path = repo_root / "assets" / "supermix_qwen_icon.ico"

binaries = []
hiddenimports = [
    # Imported inside functions in the desktop app, so static analysis of the
    # entry script alone is not guaranteed to reach them.
    "supermix_chat_server",
    "mimomix_core",
    "mimomix_text",
    "mimomix_decoding",
    "train_mimomix_talk",
    "prompt_normaliser",
    "answer_check",
    "eval_problem_solving",
    "recall_index",
    "device_utils",
]

for package in ("webview", "flask", "werkzeug", "jinja2", "numpy"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    [str(source_dir / "supermix_chat_desktop_app.py")],
    pathex=[str(source_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Top-level packages nothing in the chat path imports. Each is large and
    # excluding them is worth tens of MB.
    #
    # Only whole third-party packages are listed here, never a submodule of a
    # package we still use. Excluding `torch.distributions` broke the first
    # build outright -- torch's own `__init__` imports it, so the exclusion
    # produced `ImportError: cannot import name 'distributions' from
    # partially initialized module 'torch'` and the app died on launch. If a
    # package is needed at all, let PyInstaller decide which parts of it ship.
    excludes=[
        "transformers",
        "peft",
        "safetensors",
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "notebook",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SupermixChatDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX on torch's DLLs is slow to compress and slow to start, for little
    # gain on already-compressed binaries.
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[str(icon_path)] if icon_path.is_file() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SupermixChatDesktop",
)
