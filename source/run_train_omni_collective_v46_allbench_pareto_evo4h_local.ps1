param(
    [switch]$NoWatchdog
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$runName = "omni_collective_v46_allbench_pareto_evo4h"
$logDir = Join-Path $repoRoot "output\$runName`_train"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "$runName`_${timestamp}.out.log"
$errLog = Join-Path $logDir "$runName`_${timestamp}.err.log"
$pidPath = Join-Path $logDir "$runName.pid"
$workerPidPath = Join-Path $logDir "$runName.worker.pid"
$capPath = Join-Path $logDir "$runName`_${timestamp}.cap.json"
$targetSeconds = 14400

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

$championManifestPath = Join-Path $repoRoot "output\omni_collective_v46_champion.json"
if (-not (Test-Path $championManifestPath)) {
    throw "Champion manifest not found at $championManifestPath"
}
$championManifest = Get-Content -Raw -LiteralPath $championManifestPath | ConvertFrom-Json
$championZip = [string]$championManifest.zip_path
if (-not (Test-Path $championZip)) {
    $championZip = [string]$championManifest.desktop_zip_path
}
if (-not (Test-Path $championZip)) {
    throw "Champion zip not found from manifest."
}

$artifactDir = Split-Path -Parent ([string]$championManifest.meta_path)
$championSummary = Get-LatestSingleFile -Path $artifactDir -Filter "*_summary.json"
$latestV8Summary = Get-LatestSingleFile -Path (Join-Path $repoRoot "output") -Filter "omni_collective_v8_frontier_summary.json"

$env:OMNI_V46_MIN_FREE_GB = "3.0"
$stage1Epochs = 1
$stage2Epochs = 1

$pythonArgs = @(
    "-u"
    "source\train_omni_collective_v46.py"
    "--summary"
    ('"{0}"' -f $championSummary)
    "--base_zip"
    ('"{0}"' -f $championZip)
    "--cached_v8_summary"
    ('"{0}"' -f $latestV8Summary)
    "--output_root"
    "output\omni_collective_v46_allbench_pareto_evo4h_prep"
    "--output_dir"
    "output\omni_collective_v46_allbench_pareto_evo4h_train"
    "--family_name"
    "omni_collective_v46_allbench_pareto_evo4h"
    "--artifact_prefix"
    "supermix_omni_collective_v46_allbench_pareto_evo4h"
    "--seed"
    "2041"
    "--benchmark_limit"
    "200"
    "--teacher_route_limit"
    "120"
    "--verifier_limit"
    "180"
    "--budget_limit"
    "96"
    "--diversity_limit"
    "120"
    "--base_distill_limit"
    "96"
    "--evolution_limit"
    "160"
    "--agentic_evolution_limit"
    "220"
    "--research_evolution_limit"
    "260"
    "--cognitive_evolution_limit"
    "720"
    "--fresh_data_limit"
    "96"
    "--benchmark_failure_replay_limit"
    "2600"
    "--hard_benchmark_limit"
    "900"
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
    "0.0000026"
    "--stage2_lr"
    "0.0000010"
    "--warmup_ratio"
    "0.16"
    "--min_lr_scale"
    "0.02"
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

$watcherPid = $null
if (-not $NoWatchdog) {
    $watcherCommand = @"
Start-Sleep -Seconds $targetSeconds
`$stopped = @()
foreach (`$id in @($workerPid, $launcherPid)) {
    `$p = Get-Process -Id `$id -ErrorAction SilentlyContinue
    if (`$p) {
        Stop-Process -Id `$id -Force -ErrorAction SilentlyContinue
        `$stopped += `$id
    }
}
[pscustomobject]@{
    capped_at = (Get-Date).ToString("o")
    target_seconds = $targetSeconds
    launcher_pid = $launcherPid
    worker_pid = $workerPid
    stopped_pids = `$stopped
    out_log = "$outLog"
    err_log = "$errLog"
} | ConvertTo-Json -Depth 4 | Set-Content -Path "$capPath" -Encoding ascii
"@

    $watcherEncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watcherCommand))
    $watcher = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $watcherEncodedCommand) -PassThru -WindowStyle Hidden
    $watcherPid = $watcher.Id
}

[pscustomobject]@{
    launcher_pid = $launcherPid
    worker_pid = $workerPid
    watchdog_pid = $watcherPid
    target_seconds = if ($NoWatchdog) { $null } else { $targetSeconds }
    target_hours = if ($NoWatchdog) { $null } else { 4 }
    out_log = $outLog
    err_log = $errLog
    cap_marker = if ($NoWatchdog) { $null } else { $capPath }
    pid_file = $pidPath
    worker_pid_file = $workerPidPath
    summary = $championSummary
    base_zip = $championZip
    cached_v8_summary = $latestV8Summary
    benchmark_suite_version = "common_v3_10suite_reasoning"
    strategy = "promoted champion archive first + all-suite Pareto floor + weak-suite lift + anti-regression contrast"
    base_branch = [string]$championManifest.champion_family
    base_score = [double]$championManifest.common_benchmark_score
    benchmark_failure_replay_limit = 2600
    pareto_regression_guard_limit = 1300
    hard_benchmark_limit = 900
    weak_suite_targets = "bbh, commonsenseqa, openbookqa, winogrande, plus non-regression floors for all suites"
} | ConvertTo-Json -Depth 4
