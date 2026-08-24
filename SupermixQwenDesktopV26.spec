# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('source\\\\qwen_chat_web_app.py', 'source'), ('source\\\\qwen_adapter_promotion.py', 'source'), ('source\\\\conversation_state.py', 'source'), ('source\\\\conversation_directive.py', 'source'), ('source\\\\grounding_runtime.py', 'source'), ('source\\\\interaction_planner.py', 'source'), ('source\\\\prompt_understanding.py', 'source'), ('source\\\\reasoning_engine.py', 'source'), ('source\\\\science_plan.py', 'source'), ('runtime_python\\\\prompt_understanding.py', 'runtime_python'), ('runtime_python\\\\reasoning_engine.py', 'runtime_python'), ('runtime_python\\\\science_plan.py', 'runtime_python'), ('assets', 'assets'), ('build\\desktop_bundle_stage', 'bundled_latest_artifact')]
binaries = []
hiddenimports = []
tmp_ret = collect_all('webview')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('bottle')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pythonnet')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('clr_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['source\\qwen_chat_desktop_app.py'],
    pathex=[],
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
    name='SupermixQwenDesktopV26',
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
    icon=['assets\\supermix_qwen_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SupermixQwenDesktopV26',
)
