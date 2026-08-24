param(
  [string]$Name = "SupermixStudioDesktop",
  [string]$ModelsDir = "",
  [string]$PythonExe = "",
  [switch]$SkipDependencyInstall,
  [switch]$RuntimeOnly
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

if (-not $RuntimeOnly) {
  if (-not $ModelsDir) {
    $ModelsDir = if ($env:SUPERMIX_MODELS_DIR) {
      $env:SUPERMIX_MODELS_DIR
    } else {
      Join-Path $env:USERPROFILE "Desktop\models"
    }
  }
  $ModelsDir = (Resolve-Path -LiteralPath $ModelsDir -ErrorAction Stop).Path
}

if (-not $PythonExe) {
  $PreferredPython = Join-Path $RepoRoot ".venv-dml\Scripts\python.exe"
  if (Test-Path $PreferredPython) {
    $PythonExe = $PreferredPython
  } else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) {
      throw "No usable Python executable found. Pass -PythonExe explicitly."
    }
    $PythonExe = $PythonCommand.Source
  }
} elseif (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
  throw "Python executable not found at $PythonExe"
}

if (-not $SkipDependencyInstall) {
  & $PythonExe -m pip install pywebview pyinstaller pillow sympy | Out-Host
}

& $PythonExe "source\generate_studio_runtime_manifest.py" --check
if ($LASTEXITCODE -ne 0) {
  throw "The checked Studio runtime manifest is stale. Regenerate and review it before packaging."
}

& $PythonExe "source\generate_desktop_branding.py" | Out-Host

$BaseModelDir = & $PythonExe -c "import sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; print(app.resolve_local_base_model_path(''))"
if (-not $BaseModelDir) {
  throw "Failed to resolve local Qwen base model directory."
}
$BaseModelDir = $BaseModelDir.Trim()

$ModelsStageDir = Join-Path $RepoRoot "build\studio_models_stage"
$BaseModelStageDir = Join-Path $RepoRoot "build\studio_base_model_stage"
$BundleManifestPath = Join-Path $RepoRoot "output\supermix_studio_bundled_models_manifest.json"
$RuntimeManifestPath = Join-Path $RepoRoot "source\studio_runtime_manifest.json"
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

$ModelZipFiles = @()
if (-not $RuntimeOnly) {
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
  $ModelZipFiles = @($SelectedBundleJson | ConvertFrom-Json)
  if (-not $ModelZipFiles) {
    throw "No curated model zip files resolved from $ModelsDir"
  }
  foreach ($ZipFile in $ModelZipFiles) {
    Copy-Item -Force $ZipFile.path (Join-Path $ModelsStageDir $ZipFile.name)
  }
}
$BundleManifest = [ordered]@{
  generated_at = (Get-Date).ToString("o")
  models_dir = if ($RuntimeOnly) { "" } else { $ModelsDir }
  bundle_strategy = if ($RuntimeOnly) { "runtime_only_base_model_plus_model_store" } else { "curated_core_plus_model_store" }
  bundled_model_count = @($ModelZipFiles).Count
  bundled_model_keys = @($ModelZipFiles | ForEach-Object { $_.key })
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
$AssetsDir = Join-Path $RepoRoot "assets"
$SummaryPath = Get-ChildItem -Path (Join-Path $RepoRoot "output") -Filter "benchmark_all_models_common_plus_summary_*.json" -File -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1 -ExpandProperty FullName
if (-not (Test-Path $IconPath)) {
  throw "Expected icon asset at $IconPath"
}
if (-not $RuntimeOnly -and -not (Test-Path $SummaryPath)) {
  throw "Expected benchmark summary at $SummaryPath"
}

try {
  New-Item -ItemType Directory -Path "build\studio_desktop_spec" -Force | Out-Null
  New-Item -ItemType Directory -Path "build\route_study_cli" -Force | Out-Null
  New-Item -ItemType Directory -Path "build\route_shadow_cli" -Force | Out-Null
  $PyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", $Name,
    "--icon", $IconPath,
    "--paths", (Join-Path $RepoRoot "source"),
    "--collect-all", "webview",
    "--collect-all", "flask",
    "--collect-all", "werkzeug",
    "--collect-all", "PIL",
    "--collect-all", "sympy",
    "--collect-all", "mpmath",
    "--collect-all", "safetensors",
    "--collect-all", "transformers",
    "--collect-all", "peft",
    "--add-data", "$AssetsDir;assets",
    "--add-data", "$ModelsStageDir;bundled_models",
    "--add-data", "$BaseModelStageDir;bundled_base_model"
  )
  if ($SummaryPath) {
    $PyInstallerArgs += @("--add-data", "$SummaryPath;output")
  }
  $PyInstallerArgs += @(
    "--add-data", "$(Join-Path $RepoRoot 'source\reasoning_engine.py');.",
    "--add-data", "$(Join-Path $RepoRoot 'source\science_plan.py');.",
    "--add-data", "$BundleManifestPath;output",
    "--add-data", "$RuntimeManifestPath;output",
    "--specpath", "build\studio_desktop_spec",
    (Join-Path $RepoRoot "source\supermix_multimodel_desktop_app.py")
  )

  Write-Host "Building $Name"
  Write-Host $(if ($RuntimeOnly) { "Runtime-only bundle: curated model ZIPs omitted" } else { "Bundled models from: $ModelsDir" })
  Write-Host "Bundled base model from: $BaseModelDir"
  & $PythonExe @PyInstallerArgs
  if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
  }

  $ExePath = Join-Path $RepoRoot "dist\$Name\$Name.exe"
  $RouteStudyCliName = "SupermixRouteStudy"
  $RouteStudyCliArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--console",
    "--name", $RouteStudyCliName,
    "--paths", "source",
    "--distpath", "dist\$Name",
    "--workpath", "build\route_study_cli",
    "--specpath", "build\route_study_cli",
    "source\route_policy_protocol_cli.py"
  )
  Write-Host "Building $RouteStudyCliName prompt-free protocol console"
  & $PythonExe @RouteStudyCliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Route study console build failed."
  }
  $RouteStudyCliPath = Join-Path $RepoRoot "dist\$Name\$RouteStudyCliName.exe"
  if (-not (Test-Path $RouteStudyCliPath)) {
    throw "Expected route study console at $RouteStudyCliPath"
  }
  $RouteShadowCliName = "SupermixRouteShadow"
  $RouteShadowCliArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onefile",
    "--console",
    "--name", $RouteShadowCliName,
    "--paths", "source",
    "--distpath", "dist\$Name",
    "--workpath", "build\route_shadow_cli",
    "--specpath", "build\route_shadow_cli",
    "source\route_policy_shadow_cli.py"
  )
  Write-Host "Building $RouteShadowCliName shadow-only commitment console"
  & $PythonExe @RouteShadowCliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Route shadow console build failed."
  }
  $RouteShadowCliPath = Join-Path $RepoRoot "dist\$Name\$RouteShadowCliName.exe"
  if (-not (Test-Path $RouteShadowCliPath)) {
    throw "Expected route shadow console at $RouteShadowCliPath"
  }
  Write-Host "Build complete: $ExePath"
  Write-Host "Protocol console complete: $RouteStudyCliPath"
  Write-Host "Shadow registry console complete: $RouteShadowCliPath"
}
finally {
  if (Test-Path $ModelsStageDir) { Remove-Item -Recurse -Force $ModelsStageDir }
  if (Test-Path $BaseModelStageDir) { Remove-Item -Recurse -Force $BaseModelStageDir }
}
