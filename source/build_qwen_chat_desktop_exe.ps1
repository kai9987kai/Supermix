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

$AdapterSelectionJson = python -c "from pathlib import Path; import json, sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; adapter = app.find_latest_adapter_dir(Path('.').resolve()); kind = app.adapter_activation_kind(adapter); payload = {'adapter_dir': str(adapter), 'activation': kind} if kind in {'promoted', 'legacy'} else sys.exit('Selected adapter is not eligible for automatic activation.'); print(json.dumps(payload))"
if ($LASTEXITCODE -ne 0 -or -not $AdapterSelectionJson) {
  throw 'Failed to resolve an eligible adapter directory.'
}
$AdapterSelection = $AdapterSelectionJson | ConvertFrom-Json
$AdapterDir = [string]$AdapterSelection.adapter_dir
$AdapterActivation = [string]$AdapterSelection.activation
if (-not $AdapterDir -or $AdapterActivation -notin @("promoted", "legacy")) {
  throw 'Adapter selection returned an invalid activation classification.'
}
$BaseModelDir = python -c "from pathlib import Path; import sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; print(app.resolve_base_model_path(Path(sys.argv[1]), ''))" $AdapterDir
if (-not $BaseModelDir) {
  throw 'Failed to resolve local base model directory.'
}
$ArtifactDir = Split-Path -Parent $AdapterDir
$BundleDir = Join-Path $RepoRoot "build\desktop_bundle_stage"
$BaseModelBundleDir = Join-Path $RepoRoot "build\desktop_base_model_stage"
$PromotionReceiptFiles = @("promotion_manifest.json", "promotion_gate.json")

if ($AdapterActivation -eq "promoted") {
  python -c "from pathlib import Path; import sys; sys.path.insert(0, 'source'); from qwen_adapter_promotion import validate_promoted_adapter; raise SystemExit(0 if validate_promoted_adapter(Path(sys.argv[1])) is not None else 3)" $AdapterDir
  if ($LASTEXITCODE -ne 0) {
    throw "Selected promoted adapter has missing or invalid promotion receipts: $AdapterDir"
  }
  foreach ($FileName in $PromotionReceiptFiles) {
    $SourcePath = Join-Path $ArtifactDir $FileName
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) {
      throw "Selected promoted adapter is missing required receipt: $SourcePath"
    }
  }
}

$IconPath = Join-Path $RepoRoot "assets\supermix_qwen_icon.ico"
if (-not (Test-Path $IconPath)) {
  throw "Expected icon asset at $IconPath"
}

if (Test-Path $BundleDir) {
  Remove-Item -Recurse -Force $BundleDir
}
if (Test-Path $BaseModelBundleDir) {
  Remove-Item -Recurse -Force $BaseModelBundleDir
}
New-Item -ItemType Directory -Path $BundleDir -Force | Out-Null
Copy-Item -Recurse -Force $AdapterDir (Join-Path $BundleDir "adapter")
python "source\materialize_model_dir.py" $BaseModelDir $BaseModelBundleDir | Out-Host
foreach ($FileName in @("benchmark_results.json", "benchmark_comparison.png", "latest_adapter_checkpoint.txt")) {
  $SourcePath = Join-Path $ArtifactDir $FileName
  if (Test-Path $SourcePath) {
    Copy-Item -Force $SourcePath (Join-Path $BundleDir $FileName)
  }
}
$PromotionManifestRelativePath = $null
$PromotionGateRelativePath = $null
if ($AdapterActivation -eq "promoted") {
  foreach ($FileName in $PromotionReceiptFiles) {
    Copy-Item -Force -LiteralPath (Join-Path $ArtifactDir $FileName) (Join-Path $BundleDir $FileName)
  }
  $PromotionManifestRelativePath = "promotion_manifest.json"
  $PromotionGateRelativePath = "promotion_gate.json"
  python -c "from pathlib import Path; import sys; sys.path.insert(0, 'source'); from qwen_adapter_promotion import validate_promoted_adapter; raise SystemExit(0 if validate_promoted_adapter(Path(sys.argv[1])) is not None else 4)" (Join-Path $BundleDir "adapter")
  if ($LASTEXITCODE -ne 0) {
    throw 'Staged promoted adapter failed receipt validation.'
  }
}
$BundleManifest = @{
  artifact_name = Split-Path $ArtifactDir -Leaf
  adapter_relative_path = "adapter"
  base_model_relative_path = "..\bundled_base_model"
  adapter_activation = $AdapterActivation
  promotion_manifest_relative_path = $PromotionManifestRelativePath
  promotion_gate_relative_path = $PromotionGateRelativePath
  created_at_utc = (Get-Date).ToUniversalTime().ToString("o")
}
$BundleManifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 (Join-Path $BundleDir "release_manifest.json")

try {
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
    "--add-data", "source\\qwen_adapter_promotion.py;source",
    "--add-data", "source\\conversation_state.py;source",
    "--add-data", "source\\conversation_directive.py;source",
    "--add-data", "source\\grounding_runtime.py;source",
    "--add-data", "source\\interaction_planner.py;source",
    "--add-data", "source\\prompt_understanding.py;source",
    "--add-data", "source\\reasoning_engine.py;source",
    "--add-data", "source\\science_plan.py;source",
    "--add-data", "runtime_python\\prompt_understanding.py;runtime_python",
    "--add-data", "runtime_python\\reasoning_engine.py;runtime_python",
    "--add-data", "runtime_python\\science_plan.py;runtime_python",
    "--add-data", "assets;assets",
    "--add-data", "$BundleDir;bundled_latest_artifact",
    "--add-data", "$BaseModelBundleDir;bundled_base_model",
    "source\\qwen_chat_desktop_app.py"
  )

  Write-Host "Building $Name with adapter: $AdapterDir"
  Write-Host "Bundled artifact metadata from: $ArtifactDir"
  Write-Host "Bundled base model from: $BaseModelDir"
  python @PyInstallerArgs
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
  }

  $ExePath = Join-Path $RepoRoot "dist\$Name\$Name.exe"
  Write-Host "Build complete: $ExePath"
}
finally {
  if (Test-Path $BundleDir) {
    Remove-Item -Recurse -Force $BundleDir
  }
  if (Test-Path $BaseModelBundleDir) {
    Remove-Item -Recurse -Force $BaseModelBundleDir
  }
}
