Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "output\omni_collective_v46_research_evo300_train"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "omni_collective_v46_research_evo300_${timestamp}.out.log"
$errLog = Join-Path $logDir "omni_collective_v46_research_evo300_${timestamp}.err.log"
$pidPath = Join-Path $logDir "omni_collective_v46_research_evo300.pid"
$workerPidPath = Join-Path $logDir "omni_collective_v46_research_evo300.worker.pid"

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

$latestBaseDir = Join-Path $repoRoot "output\omni_collective_v46_coevo180_train"
$latestBaseSummary = Get-LatestSingleFile -Path $latestBaseDir -Filter "omni_collective_v46_coevo180_frontier_summary.json"
$latestBaseZip = Get-LatestSingleFile -Path $latestBaseDir -Filter "supermix_omni_collective_v46_coevo180_frontier_*.zip"
$latestV8Summary = Get-LatestSingleFile -Path (Join-Path $repoRoot "output") -Filter "omni_collective_v8_frontier_summary.json"

# This schedule targets roughly five hours on the current CPU setup.
# It adds research_evolution rows from normalized ES, Pareto, hindsight,
# curriculum, anchor-fitness, novelty migration, and behavior-merge supervision.
$stage1Epochs = 2
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
    "output\omni_collective_v46_research_evo300_prep"
    "--output_dir"
    "output\omni_collective_v46_research_evo300_train"
    "--family_name"
    "omni_collective_v46_research_evo300"
    "--artifact_prefix"
    "supermix_omni_collective_v46_research_evo300"
    "--seed"
    "1704"
    "--benchmark_limit"
    "96"
    "--teacher_route_limit"
    "112"
    "--verifier_limit"
    "96"
    "--budget_limit"
    "72"
    "--diversity_limit"
    "144"
    "--base_distill_limit"
    "192"
    "--evolution_limit"
    "128"
    "--agentic_evolution_limit"
    "256"
    "--research_evolution_limit"
    "384"
    "--fresh_data_limit"
    "128"
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
    "0.000034"
    "--stage2_lr"
    "0.000020"
    "--warmup_ratio"
    "0.07"
    "--min_lr_scale"
    "0.035"
    "--device"
    "auto"
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
    target_hours = 5.0
    out_log = $outLog
    err_log = $errLog
    pid_file = $pidPath
    worker_pid_file = $workerPidPath
    summary = $latestBaseSummary
    base_zip = $latestBaseZip
    cached_v8_summary = $latestV8Summary
    research_evolution_limit = 384
} | ConvertTo-Json -Depth 4
