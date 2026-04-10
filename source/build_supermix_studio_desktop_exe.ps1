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
$BundledModelKeys = @(
  "v40_benchmax",
  "omni_collective_v41",
  "omni_collective_v8",
  "omni_collective_v7",
  "science_vision_micro_v1",
  "v38_native_xlite_fp16",
  "dcgan_v2_in_progress",
  "math_equation_micro_v1",
  "protein_folding_micro_v1",
  "mattergen_micro_v1",
  "three_d_generation_micro_v1"
)

if (Test-Path $ModelsStageDir) { Remove-Item -Recurse -Force $ModelsStageDir }
if (Test-Path $BaseModelStageDir) { Remove-Item -Recurse -Force $BaseModelStageDir }
New-Item -ItemType Directory -Path $ModelsStageDir -Force | Out-Null

$SelectedBundleJson = & $PythonExe -c @"
import json, sys
from pathlib import Path
sys.path.insert(0, 'source')
from multimodel_catalog import discover_model_records
records = {record.key: record for record in discover_model_records(models_dir=Path(r'''$ModelsDir'''))}
keys = [item for item in r'''$($BundledModelKeys -join "`n")'''.splitlines() if item]
missing = [key for key in keys if key not in records]
if missing:
    raise SystemExit('Missing bundled model keys: ' + ', '.join(missing))
payload = [
    {
        'key': key,
        'label': records[key].label,
        'name': records[key].zip_path.name,
        'path': str(records[key].zip_path),
        'size_bytes': records[key].zip_path.stat().st_size,
    }
    for key in keys
]
print(json.dumps(payload))
"@
$ModelZipFiles = $SelectedBundleJson | ConvertFrom-Json
if (-not $ModelZipFiles) {
  throw "No curated model zip files resolved from $ModelsDir"
}
foreach ($ZipFile in $ModelZipFiles) {
  Copy-Item -Force $ZipFile.path (Join-Path $ModelsStageDir $ZipFile.name)
}
$BundleManifest = [ordered]@{
  generated_at = (Get-Date).ToString("o")
  models_dir = $ModelsDir
  bundle_strategy = "curated_core_plus_model_store"
  bundled_model_count = @($ModelZipFiles).Count
  bundled_model_keys = @($BundledModelKeys)
  bundled_models = @($ModelZipFiles | ForEach-Object {
      [ordered]@{
        key = $_.key
        label = $_.label
        name = $_.name
        size_bytes = $_.size_bytes
      }
    })
  remote_model_store_repo = "Kai9987kai/supermix-model-zoo"
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
