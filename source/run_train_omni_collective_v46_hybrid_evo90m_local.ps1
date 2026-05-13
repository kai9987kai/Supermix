Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$logDir = Join-Path $repoRoot "output\omni_collective_v46_hybrid_evo90m_fit_train"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outLog = Join-Path $logDir "omni_collective_v46_hybrid_evo90m_fit_${timestamp}.out.log"
$errLog = Join-Path $logDir "omni_collective_v46_hybrid_evo90m_fit_${timestamp}.err.log"
$pidPath = Join-Path $logDir "omni_collective_v46_hybrid_evo90m_fit.pid"
$workerPidPath = Join-Path $logDir "omni_collective_v46_hybrid_evo90m_fit.worker.pid"

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

$latestHybridSummary = Get-LatestSingleFile -Path (Join-Path $repoRoot "output\omni_collective_v46_hybrid_evo120_train") -Filter "omni_collective_v46_hybrid_evo120_frontier_summary.json"
$latestHybridZip = Get-LatestSingleFile -Path (Join-Path $repoRoot "output\omni_collective_v46_hybrid_evo120_train") -Filter "supermix_omni_collective_v46_hybrid_evo120_frontier_*.zip"
$latestV8Summary = Get-LatestSingleFile -Path (Join-Path $repoRoot "output") -Filter "omni_collective_v8_frontier_summary.json"

# This schedule targets roughly a 90-minute refinement pass on the current CPU setup.
$stage1Epochs = 1
$stage2Epochs = 1

$pythonArgs = @(
    "-u"
    "source\train_omni_collective_v46.py"
    "--summary"
    ('"{0}"' -f $latestHybridSummary)
    "--base_zip"
    ('"{0}"' -f $latestHybridZip)
    "--cached_v8_summary"
    ('"{0}"' -f $latestV8Summary)
    "--output_root"
    "output\omni_collective_v46_hybrid_evo90m_fit_prep"
    "--output_dir"
    "output\omni_collective_v46_hybrid_evo90m_fit_train"
    "--family_name"
    "omni_collective_v46_hybrid_evo90m_fit"
    "--artifact_prefix"
    "supermix_omni_collective_v46_hybrid_evo90m_fit"
    "--seed"
    "911"
    "--benchmark_limit"
    "72"
    "--teacher_route_limit"
    "88"
    "--verifier_limit"
    "72"
    "--budget_limit"
    "56"
    "--diversity_limit"
    "108"
    "--base_distill_limit"
    "160"
    "--evolution_limit"
    "88"
    "--agentic_evolution_limit"
    "168"
    "--fresh_data_limit"
    "96"
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
    "0.000052"
    "--stage2_lr"
    "0.00003"
    "--warmup_ratio"
    "0.06"
    "--min_lr_scale"
    "0.04"
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
    out_log = $outLog
    err_log = $errLog
    pid_file = $pidPath
    worker_pid_file = $workerPidPath
    summary = $latestHybridSummary
    base_zip = $latestHybridZip
    cached_v8_summary = $latestV8Summary
} | ConvertTo-Json -Depth 4
