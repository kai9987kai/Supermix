param(
  [string]$Name = "SupermixStudioDesktop",
  [string]$ModelsDir = "C:\Users\kai99\Desktop\models",
  [switch]$SkipDependencyInstall
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$PythonExe = Join-Path $RepoRoot ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
  throw "Expected Python environment at $PythonExe"
}

if (-not $SkipDependencyInstall) {
  & $PythonExe -m pip install pywebview pyinstaller pillow sympy | Out-Host
}

python "source\generate_desktop_branding.py" | Out-Host

$BaseModelDir = & $PythonExe -c "import sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; print(app.resolve_local_base_model_path(''))"
if (-not $BaseModelDir) {
  throw "Failed to resolve local Qwen base model directory."
}
$BaseModelDir = $BaseModelDir.Trim()

$ModelsStageDir = Join-Path $RepoRoot "build\studio_models_stage"
$BaseModelStageDir = Join-Path $RepoRoot "build\studio_base_model_stage"
$BundleManifestPath = Join-Path $RepoRoot "output\supermix_studio_bundled_models_manifest.json"

if (Test-Path $ModelsStageDir) { Remove-Item -Recurse -Force $ModelsStageDir }
if (Test-Path $BaseModelStageDir) { Remove-Item -Recurse -Force $BaseModelStageDir }
New-Item -ItemType Directory -Path $ModelsStageDir -Force | Out-Null

$RedundantBundleNames = @(
  "champion_v31_hybrid_plus_refresh_bundle_20260326.zip",
  "champion_v33_frontier_full_bundle_20260326.zip",
  "qwen_supermix_enhanced_v30_anchor_refresh_20260326_experimental_bundle.zip"
)
$ModelZipFiles = Get-ChildItem -Path $ModelsDir -File -Filter *.zip | Where-Object {
  $_.Name -notmatch ' \(\d+\)\.zip$' -and $RedundantBundleNames -notcontains $_.Name
}
if (-not $ModelZipFiles) {
  throw "No model zip files found in $ModelsDir"
}
foreach ($ZipFile in $ModelZipFiles) {
  Copy-Item -Force $ZipFile.FullName (Join-Path $ModelsStageDir $ZipFile.Name)
}
$BundleManifest = [ordered]@{
  generated_at = (Get-Date).ToString("o")
  models_dir = $ModelsDir
  bundled_model_count = @($ModelZipFiles).Count
  bundled_models = @($ModelZipFiles | Sort-Object Name | ForEach-Object {
      [ordered]@{
        name = $_.Name
        size_bytes = $_.Length
      }
    })
}
$BundleManifest | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 $BundleManifestPath

& $PythonExe "source\materialize_model_dir.py" $BaseModelDir $BaseModelStageDir | Out-Host

$IconPath = Join-Path $RepoRoot "assets\supermix_qwen_icon.ico"
$SummaryPath = Get-ChildItem -Path (Join-Path $RepoRoot "output") -Filter "benchmark_all_models_common_plus_summary_*.json" -File -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
if (-not (Test-Path $IconPath)) {
  throw "Expected icon asset at $IconPath"
}
if (-not (Test-Path $SummaryPath)) {
  throw "Expected benchmark summary at $SummaryPath"
}

try {
  $PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", $Name,
    "--icon", $IconPath,
    "--paths", "source",
    "--collect-all", "webview",
    "--collect-all", "flask",
    "--collect-all", "werkzeug",
    "--collect-all", "PIL",
    "--collect-all", "sympy",
    "--collect-all", "mpmath",
    "--collect-all", "safetensors",
    "--collect-all", "transformers",
    "--collect-all", "peft",
    "--add-data", "assets;assets",
    "--add-data", "$ModelsStageDir;bundled_models",
    "--add-data", "$BaseModelStageDir;bundled_base_model",
    "--add-data", "$SummaryPath;output",
    "--add-data", "$BundleManifestPath;output",
    "source\supermix_multimodel_desktop_app.py"
  )

  Write-Host "Building $Name"
  Write-Host "Bundled models from: $ModelsDir"
  Write-Host "Bundled base model from: $BaseModelDir"
  & $PythonExe @PyInstallerArgs
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
  }

  $ExePath = Join-Path $RepoRoot "dist\$Name\$Name.exe"
  Write-Host "Build complete: $ExePath"
}
finally {
  if (Test-Path $ModelsStageDir) { Remove-Item -Recurse -Force $ModelsStageDir }
  if (Test-Path $BaseModelStageDir) { Remove-Item -Recurse -Force $BaseModelStageDir }
}
