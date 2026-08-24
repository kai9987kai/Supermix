<#
.SYNOPSIS
Build Supermix Chat as a Windows desktop application with v74 built in.

.DESCRIPTION
Stages the trained checkpoint, strips the optimiser and scheduler state from
it, and runs PyInstaller against SupermixChatDesktop.spec.

Stripping matters: the training checkpoint is 103.5 MB, of which roughly 69 MB
is AdamW moments and the learning-rate schedule. Those are needed to *resume
training* and are dead weight in an application that only ever runs inference.
The stripped checkpoint is 33 MB and loads identically.

.PARAMETER Checkpoint
Training checkpoint to ship. Defaults to the v74 run output.

.PARAMETER SkipStage
Reuse whatever is already staged in build\chat_model_stage instead of
re-deriving it. Useful when iterating on the spec.

.EXAMPLE
.\build_chat_desktop_exe.ps1

.EXAMPLE
.\build_chat_desktop_exe.ps1 -Checkpoint output\v73_decomposed_short\v73_decomposed_short.pt
#>
[CmdletBinding()]
param(
    [string]$Checkpoint = "output\v74_broad\v74_broad.pt",
    [switch]$SkipStage
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$stageDir = Join-Path $repoRoot "build\chat_model_stage"
$stagedModel = Join-Path $stageDir "supermix_v74.pt"

if (-not $SkipStage) {
    if (-not (Test-Path $Checkpoint)) {
        throw "checkpoint not found: $Checkpoint"
    }

    Write-Host "Staging model from $Checkpoint" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

    # Strip training-only state. Done in Python because the checkpoint is a
    # torch pickle, not something PowerShell can open.
    $strip = @'
import os, sys, torch
source, destination = sys.argv[1], sys.argv[2]
payload = torch.load(source, map_location="cpu", weights_only=False)
slim = {k: v for k, v in payload.items()
        if k not in ("optimiser_state", "scheduler_state")}
torch.save(slim, destination)
before = os.path.getsize(source) / 1e6
after = os.path.getsize(destination) / 1e6
print(f"  {before:.1f} MB -> {after:.1f} MB (training state removed)")
'@
    $stripFile = Join-Path $env:TEMP "supermix_strip_checkpoint.py"
    Set-Content -Path $stripFile -Value $strip -Encoding utf8
    & python $stripFile $Checkpoint $stagedModel
    if ($LASTEXITCODE -ne 0) { throw "failed to stage the checkpoint" }
    Remove-Item $stripFile -ErrorAction SilentlyContinue
}

if (-not (Test-Path $stagedModel)) {
    throw "no staged model at $stagedModel; run without -SkipStage"
}

Write-Host "Running PyInstaller (this takes several minutes)" -ForegroundColor Cyan
& python -m PyInstaller --noconfirm --clean `
    --distpath dist --workpath build\pyi_chat `
    SupermixChatDesktop.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$appDir = Join-Path $repoRoot "dist\SupermixChatDesktop"
$exe = Join-Path $appDir "SupermixChatDesktop.exe"
if (-not (Test-Path $exe)) { throw "build reported success but produced no exe" }

# A build that shipped without the model would launch and then fail, so this
# is checked rather than assumed.
$bundledModel = Get-ChildItem -Path $appDir -Recurse -Filter "supermix_v74.pt" -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $bundledModel) { throw "the built application contains no model" }

$sizeMb = (Get-ChildItem $appDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ""
Write-Host "Built: $exe" -ForegroundColor Green
Write-Host ("  application size : {0:N0} MB" -f $sizeMb)
Write-Host ("  bundled model    : {0:N1} MB" -f ($bundledModel.Length / 1MB))
Write-Host ""
Write-Host "Next: build an installer with one of" -ForegroundColor Cyan
Write-Host "  ISCC.exe installer\SupermixChatDesktop.iss      (needs Inno Setup 6)"
Write-Host "  .\build_chat_desktop_installer.ps1              (no extra tooling)"
