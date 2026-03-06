param(
  [string]$Name = "SupermixQwenDesktop",
  [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $SkipDependencyInstall) {
  python -m pip install pywebview pillow pyinstaller | Out-Host
}

python "source\generate_desktop_branding.py" | Out-Host

$AdapterDir = python -c "from pathlib import Path; import sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; print(app.find_latest_adapter_dir(Path('.').resolve()))"
if (-not $AdapterDir) {
  throw 'Failed to resolve latest adapter directory.'
}

$IconPath = Join-Path $RepoRoot "assets\supermix_qwen_icon.ico"
if (-not (Test-Path $IconPath)) {
  throw "Expected icon asset at $IconPath"
}

$PyInstallerArgs = @(
  "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onedir",
  "--windowed",
  "--name", $Name,
  "--icon", $IconPath,
  "--collect-all", "webview",
  "--collect-all", "bottle",
  "--collect-all", "pythonnet",
  "--collect-all", "clr_loader",
  "--add-data", "source\\qwen_chat_web_app.py;source",
  "--add-data", "assets;assets",
  "--add-data", "$AdapterDir;bundled_latest_adapter",
  "source\\qwen_chat_desktop_app.py"
)

Write-Host "Building $Name with adapter: $AdapterDir"
python @PyInstallerArgs
if ($LASTEXITCODE -ne 0) {
  throw "PyInstaller build failed."
}

$ExePath = Join-Path $RepoRoot "dist\$Name\$Name.exe"
Write-Host "Build complete: $ExePath"
