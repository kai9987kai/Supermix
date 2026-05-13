Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$runName = "omni_collective_v46_retention_evo520_plus1h"
$logDir = Join-Path $repoRoot "output\$runName`_train"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "$runName`_${timestamp}.out.log"
$errLog = Join-Path $logDir "$runName`_${timestamp}.err.log"
$pidPath = Join-Path $logDir "$runName.pid"
$workerPidPath = Join-Path $logDir "$runName.worker.pid"

foreach ($path in @($pidPath, $workerPidPath)) {
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
}

function Get-LatestSingleFile {
    param(
        [string]$Path,
        [string]$Filter
    )
    $item = Get-ChildItem -Path $Path -Recurse -File -Filter $Filter -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $item) {
        throw "No file found for filter '$Filter' under $Path"
    }
    return $item.FullName
}

$python = Join-Path $repoRoot ".venv-dml\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python executable not found at $python"
}

$latestBaseDir = Join-Path $repoRoot "output\omni_collective_v46_meta_evo420_plus2h_train"
$latestBaseSummary = Get-LatestSingleFile -Path $latestBaseDir -Filter "omni_collective_v46_meta_evo420_plus2h_frontier_summary.json"
$latestBaseZip = Get-LatestSingleFile -Path $latestBaseDir -Filter "supermix_omni_collective_v46_meta_evo420_plus2h_frontier_*.zip"
$latestV8Summary = Get-LatestSingleFile -Path (Join-Path $repoRoot "output") -Filter "omni_collective_v8_frontier_summary.json"

# Short continuation: preserve the 0.5167 benchmark frontier using success
# retention anchors while repairing the current exact misses.
$stage1Epochs = 1
$stage2Epochs = 1

$pythonArgs = @(
    "-u"
    "source\train_omni_collective_v46.py"
    "--summary"
    ('"{0}"' -f $latestBaseSummary)
    "--base_zip"
    ('"{0}"' -f $latestBaseZip)
    "--cached_v8_summary"
    ('"{0}"' -f $latestV8Summary)
    "--output_root"
    "output\omni_collective_v46_retention_evo520_plus1h_prep"
    "--output_dir"
    "output\omni_collective_v46_retention_evo520_plus1h_train"
    "--family_name"
    "omni_collective_v46_retention_evo520_plus1h"
    "--artifact_prefix"
    "supermix_omni_collective_v46_retention_evo520_plus1h"
    "--seed"
    "1908"
    "--benchmark_limit"
    "64"
    "--teacher_route_limit"
    "72"
    "--verifier_limit"
    "56"
    "--budget_limit"
    "48"
    "--diversity_limit"
    "72"
    "--base_distill_limit"
    "96"
    "--evolution_limit"
    "72"
    "--agentic_evolution_limit"
    "96"
    "--research_evolution_limit"
    "120"
    "--cognitive_evolution_limit"
    "180"
    "--fresh_data_limit"
    "72"
    "--benchmark_failure_replay_limit"
    "180"
    "--base_teacher_model_limit"
    "0"
    "--image_size"
    "96"
    "--batch_size"
    "2"
    "--stage1_epochs"
    "$stage1Epochs"
    "--stage2_epochs"
    "$stage2Epochs"
    "--stage1_lr"
    "0.000018"
    "--stage2_lr"
    "0.000010"
    "--warmup_ratio"
    "0.08"
    "--min_lr_scale"
    "0.035"
    "--device"
    "cpu"
    "--amp"
    "off"
    "--train_frontier"
) + $args

$proc = Start-Process -FilePath $python -ArgumentList $pythonArgs -WorkingDirectory $repoRoot -RedirectStandardOutput $outLog -RedirectStandardError $errLog -PassThru -WindowStyle Hidden

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
    target_hours = "about 1"
    out_log = $outLog
    err_log = $errLog
    pid_file = $pidPath
    worker_pid_file = $workerPidPath
    summary = $latestBaseSummary
    base_zip = $latestBaseZip
    cached_v8_summary = $latestV8Summary
    benchmark_failure_replay_limit = 180
    cognitive_evolution_limit = 180
    research_evolution_limit = 120
    agentic_evolution_limit = 96
} | ConvertTo-Json -Depth 4
