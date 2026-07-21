Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "output"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "omni_collective_v42_train_${timestamp}.out.log"
$errLog = Join-Path $logDir "omni_collective_v42_train_${timestamp}.err.log"
$pidPath = Join-Path $logDir "omni_collective_v42_train.pid"
$workerPidPath = Join-Path $logDir "omni_collective_v42_train.worker.pid"

if (Test-Path $pidPath) {
    Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
}
if (Test-Path $workerPidPath) {
    Remove-Item -LiteralPath $workerPidPath -Force -ErrorAction SilentlyContinue
}

$python = Join-Path $repoRoot ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

$pythonArgs = @(
    "-u"
    "source\train_omni_collective_v42.py"
    "--train_frontier"
) + $args

$proc = Start-Process -FilePath $python -ArgumentList $pythonArgs -WorkingDirectory $repoRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru
$launcherPid = $proc.Id
$workerPid = $launcherPid

for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 1
    $child = Get-CimInstance Win32_Process -Filter "ParentProcessId = $launcherPid" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'python*.exe' } |
        Select-Object -First 1
    if ($child -and $child.ProcessId) {
        $workerPid = [int]$child.ProcessId
        break
    }
}

Set-Content -Path $pidPath -Value $launcherPid -Encoding ascii
Set-Content -Path $workerPidPath -Value $workerPid -Encoding ascii

[pscustomobject]@{
    launcher_pid = $launcherPid
    worker_pid = $workerPid
    out_log = $outLog
    err_log = $errLog
    pid_file = $pidPath
    worker_pid_file = $workerPidPath
} | ConvertTo-Json -Depth 4
