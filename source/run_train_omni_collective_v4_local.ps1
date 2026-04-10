Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "output"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "omni_collective_v4_train_$timestamp.log"
$cmd = @(
    '.venv-dml\Scripts\python.exe',
    'source\train_omni_collective_v4.py'
) + $args
$escaped = $cmd | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }
cmd /c ($escaped -join ' ') 2>&1 | Tee-Object -FilePath $logPath
