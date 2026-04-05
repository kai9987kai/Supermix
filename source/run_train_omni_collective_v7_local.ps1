Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "output"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "omni_collective_v7_train_${timestamp}.out.log"
$errLog = Join-Path $logDir "omni_collective_v7_train_${timestamp}.err.log"
$pidPath = Join-Path $logDir "omni_collective_v7_train.pid"
$workerPidPath = Join-Path $logDir "omni_collective_v7_train.worker.pid"

$python = Join-Path $repoRoot ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}
$activate = Join-Path $repoRoot ".venv-dml\Scripts\activate.bat"
if (-not (Test-Path $activate)) {
    throw "Activation script not found at $activate"
}

$escapedRepo = $repoRoot.Replace('"', '""')
$escapedOut = $outLog.Replace('"', '""')
$escapedErr = $errLog.Replace('"', '""')
$escapedPid = $workerPidPath.Replace('"', '""')
$escapedActivate = $activate.Replace('"', '""')
$escapedArgs = ($args | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' }) -join " "
$bootstrapCommand = "call `"$escapedActivate`" && python -u source\\run_omni_collective_v7_background.py --out-log `"$escapedOut`" --err-log `"$escapedErr`" --worker-pid-file `"$escapedPid`" -- $escapedArgs"
$startArgs = "/d /c start `"`" /b cmd /c `"$bootstrapCommand`""

$proc = Start-Process -FilePath "cmd.exe" -ArgumentList $startArgs -WorkingDirectory $repoRoot -PassThru

$launcherPid = $proc.Id
$workerPid = 0

for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 1
    if (Test-Path $workerPidPath) {
        $rawPid = (Get-Content $workerPidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($rawPid -match '^\d+$') {
            $workerPid = [int]$rawPid
            break
        }
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
