<#
.SYNOPSIS
    build_studio_x_installer.ps1 - Supermix Studio X v48 Frontier build script.
    Refreshes benchmark assets, stages the latest bundled models, optionally
    materializes the bundled base model, then builds the PyInstaller app and
    Inno Setup installer.

.USAGE
    cd "c:\Users\kai99\Desktop\New folder (9)\Supermix_27"
    .\build_studio_x_installer.ps1
#>

param(
    [string]$ModelsDir = "C:\Users\kai99\AppData\Local\supermix_studio\models",
    [string]$BaseModelDir = "",
    [string]$PythonExe = "python",
    [string]$InnoSetupExe = "C:\Users\kai99\AppData\Local\Programs\Inno Setup 6\ISCC.exe",
    [switch]$SkipPyinstaller = $false,
    [switch]$SkipInno = $false
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Log { param([string]$msg) Write-Host "[BUILD] $msg" -ForegroundColor Cyan }
function Ok  { param([string]$msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Warn { param([string]$msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Err { param([string]$msg) Write-Host "[ERR]   $msg" -ForegroundColor Red; exit 1 }

function Resolve-FirstExistingFile {
    param([string[]]$Candidates)
    foreach ($Candidate in $Candidates) {
        if ([string]::IsNullOrWhiteSpace($Candidate)) {
            continue
        }
        if (Test-Path $Candidate) {
            return (Resolve-Path $Candidate).Path
        }
    }
    return $null
}

Log "=== Supermix Studio X - v48 Frontier Installer Build ==="
Set-Location $Root

# -- 1. Benchmark assets -------------------------------------------------------
Log "Generating v48 benchmark graph..."
& $PythonExe "source\benchmark_v48.py"
if ($LASTEXITCODE -ne 0) { Err "Benchmark generation failed." }
Ok "Benchmark assets refreshed in output\\v48_benchmark_*"

# -- 2. Stage bundled model assets --------------------------------------------
Log "Staging bundled model assets..."
$stageBase = Join-Path $Root "build\studio_models_stage"
$stageV48 = Join-Path $Root "build\v48_model_stage"
$stageBaseModel = Join-Path $Root "build\studio_base_model_stage"
New-Item -ItemType Directory -Force -Path $stageBase | Out-Null
New-Item -ItemType Directory -Force -Path $stageV48 | Out-Null
New-Item -ItemType Directory -Force -Path $stageBaseModel | Out-Null
Get-ChildItem -Path $stageV48 -File -ErrorAction SilentlyContinue | Remove-Item -Force

$v48WeightsSource = Resolve-FirstExistingFile @(
    (Join-Path $Root "omni_collective_v48_frontier_chat.pth"),
    (Join-Path $Root "omni_collective_v48_frontier.pth"),
    (Join-Path $stageV48 "omni_collective_v48_frontier.pth"),
    (Join-Path $stageV48 "omni_collective_v48_frontier_chat.pth")
)
$metaTemplate = Resolve-FirstExistingFile @(
    (Join-Path $stageBase "omni_collective_v47_frontier_meta.json"),
    (Join-Path $stageBase "omni_collective_v46_frontier_meta.json")
)
$summaryTemplate = Resolve-FirstExistingFile @(
    (Join-Path $stageBase "omni_collective_v47_frontier_summary.json"),
    (Join-Path $stageBase "omni_collective_v46_frontier_summary.json")
)

if (-not $v48WeightsSource) {
    Warn "No v48 weights file found. The installer will ship without the v48 frontier weights."
} elseif (-not $metaTemplate -or -not $summaryTemplate) {
    Err "Could not find a v46/v47 metadata template to stage the v48 bundle."
} else {
    & $PythonExe "source\prepare_v48_release_assets.py" `
        --weights $v48WeightsSource `
        --dest $stageV48 `
        --meta-template $metaTemplate `
        --summary-template $summaryTemplate
    if ($LASTEXITCODE -ne 0) { Err "Failed to stage v48 release assets." }
    Ok "v48 bundle staged in build\\v48_model_stage"
}

$externalArtifacts = @()
if (Test-Path $ModelsDir) {
    $externalArtifacts = @(Get-ChildItem -Path $ModelsDir -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension -in @(".zip", ".pth", ".json") -and
            $_.Name -notmatch "omni_collective_v48_frontier"
        })
    foreach ($Artifact in $externalArtifacts) {
        $Target = Join-Path $stageBase $Artifact.Name
        if ((Test-Path $Target) -and ((Get-Item $Target).Length -eq $Artifact.Length)) {
            continue
        }
        Copy-Item $Artifact.FullName -Destination $Target -Force
    }
    if ($externalArtifacts.Count -gt 0) {
        Ok "Staged $($externalArtifacts.Count) external bundled artifact(s) from $ModelsDir"
    } else {
        Warn "No external .zip/.pth/.json model artifacts found under $ModelsDir"
    }
} else {
    Warn "ModelsDir does not exist: $ModelsDir"
}

# -- 3. Stage bundled base model ----------------------------------------------
Log "Staging bundled base model..."
$resolvedBaseModelDir = ""
if (-not [string]::IsNullOrWhiteSpace($BaseModelDir)) {
    $resolvedBaseModelDir = $BaseModelDir
}
if ([string]::IsNullOrWhiteSpace($resolvedBaseModelDir)) {
    try {
        $resolvedBaseModelDir = (& $PythonExe -c "import sys; sys.path.insert(0, 'source'); import qwen_chat_desktop_app as app; print(app.resolve_local_base_model_path(''))").Trim()
    } catch {
        $resolvedBaseModelDir = ""
    }
}

if (-not [string]::IsNullOrWhiteSpace($resolvedBaseModelDir) -and (Test-Path $resolvedBaseModelDir)) {
    & $PythonExe "source\materialize_model_dir.py" $resolvedBaseModelDir $stageBaseModel
    if ($LASTEXITCODE -ne 0) { Err "Failed to materialize the bundled base model directory." }
    Ok "Bundled base model staged from $resolvedBaseModelDir"
} else {
    Warn "Base model directory could not be resolved. Continuing with an empty bundled_base_model stage."
}

# -- 4. Bundle manifest --------------------------------------------------------
$manifestPath = Join-Path $Root "output\supermix_studio_bundled_models_manifest.json"
$manifest = [ordered]@{
    generated = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
    source_models_dir = $ModelsDir
    bundled_base_model_dir = $resolvedBaseModelDir
    v48_included = (Test-Path (Join-Path $stageV48 "omni_collective_v48_frontier.pth"))
    models = @()
}
@($stageBase, $stageV48) | ForEach-Object {
    Get-ChildItem $_ -File -ErrorAction SilentlyContinue | ForEach-Object {
        $manifest.models += [ordered]@{
            name = $_.Name
            size_bytes = $_.Length
            source_stage = (Split-Path $_.DirectoryName -Leaf)
        }
    }
}
$manifest | ConvertTo-Json -Depth 4 | Out-File $manifestPath -Encoding utf8
Ok "Manifest written to output\\supermix_studio_bundled_models_manifest.json"

# -- 5. PyInstaller ------------------------------------------------------------
if (-not $SkipPyinstaller) {
    Log "Running PyInstaller..."
    & $PythonExe -m PyInstaller SupermixStudioX.spec --noconfirm --clean
    if ($LASTEXITCODE -ne 0) { Err "PyInstaller failed." }
    Ok "EXE built: dist\\SupermixStudioX\\SupermixStudioX.exe"
} else {
    Log "Skipping PyInstaller (--SkipPyinstaller)"
}

# -- 6. Inno Setup -------------------------------------------------------------
if (-not $SkipInno) {
    if (-not (Test-Path $InnoSetupExe)) {
        Warn "Inno Setup not found at $InnoSetupExe - skipping installer compilation."
    } else {
        Log "Compiling installer with Inno Setup..."
        New-Item -ItemType Directory -Force -Path "dist\installer" | Out-Null
        & $InnoSetupExe "installer\SupermixStudioX.iss"
        if ($LASTEXITCODE -ne 0) { Err "Inno Setup compilation failed." }
        $setupFile = Get-ChildItem "dist\installer\SupermixStudioX_V48_Setup.exe" -ErrorAction SilentlyContinue
        if ($setupFile) {
            Ok "Installer ready: $($setupFile.FullName)  ($([math]::Round($setupFile.Length / 1MB, 1)) MB)"
        } else {
            Warn "Inno Setup completed but the expected installer file was not found."
        }
    }
} else {
    Log "Skipping Inno Setup (--SkipInno)"
}

Log "=== Build complete ==="
