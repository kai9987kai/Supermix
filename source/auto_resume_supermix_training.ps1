param(
  [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
  [string]$OutputDir = "artifacts\qwen_supermix_enhanced_v28_clean_eval_robust_ipo",
  [string]$Device = "auto",
  [string]$ResumeWarmStartDir = "artifacts\qwen_supermix_enhanced_v26_full",
  [string]$TrainOutLog = "train_qwen_supermix_v28_clean_eval_robust_ipo.out.log",
  [string]$TrainErrLog = "train_qwen_supermix_v28_clean_eval_robust_ipo.err.log",
  [string]$LaunchStateFile = ".last_training_launch.txt",
  [string]$LauncherPath = "source\run_train_qwen_supermix_v26_full.ps1",
  [switch]$DryRun,
  [switch]$NoMonitor
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

function Resolve-RepoPath([string]$PathText) {
  if ([string]::IsNullOrWhiteSpace($PathText)) {
    return $null
  }
  if ([System.IO.Path]::IsPathRooted($PathText)) {
    return $PathText
  }
  return (Join-Path $repoRoot $PathText)
}

function Read-LaunchState([string]$Path) {
  $state = @{}
  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
    return $state
  }
  foreach ($line in (Get-Content -Path $Path -ErrorAction SilentlyContinue)) {
    $trimmed = [string]$line
    if ([string]::IsNullOrWhiteSpace($trimmed)) {
      continue
    }
    $pair = $trimmed.Split("=", 2)
    if ($pair.Count -ne 2) {
      continue
    }
    $key = [string]$pair[0]
    if ([string]::IsNullOrWhiteSpace($key)) {
      continue
    }
    $state[$key.Trim()] = [string]$pair[1]
  }
  return $state
}

function Write-LaunchState([string]$Path, [hashtable]$State) {
  $dir = Split-Path -Parent $Path
  if ($dir) {
    New-Item -Path $dir -ItemType Directory -Force | Out-Null
  }
  $lines = foreach ($entry in $State.GetEnumerator()) {
    "{0}={1}" -f $entry.Key, $entry.Value
  }
  Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Get-PythonCommand {
  if ($env:SUPERMIX_PYTHON -and (Test-Path $env:SUPERMIX_PYTHON)) {
    return (Resolve-Path $env:SUPERMIX_PYTHON).Path
  }
  foreach ($candidate in @(
    (Join-Path $repoRoot ".venv-dml\Scripts\python.exe"),
    (Join-Path $repoRoot ".venv\Scripts\python.exe")
  )) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }
  $cmd = Get-Command python -ErrorAction SilentlyContinue
  if ($cmd) {
    return $cmd.Source
  }
  return "python"
}

$launchStatePath = Resolve-RepoPath $LaunchStateFile
$launchState = Read-LaunchState -Path $launchStatePath
if ($launchState.Count -gt 0) {
  if (-not $PSBoundParameters.ContainsKey("OutputDir") -and $launchState.ContainsKey("OUTPUT_DIR")) {
    $OutputDir = [string]$launchState["OUTPUT_DIR"]
  }
  if (-not $PSBoundParameters.ContainsKey("TrainOutLog") -and $launchState.ContainsKey("OUT_LOG")) {
    $TrainOutLog = [string]$launchState["OUT_LOG"]
  }
  if (-not $PSBoundParameters.ContainsKey("TrainErrLog") -and $launchState.ContainsKey("ERR_LOG")) {
    $TrainErrLog = [string]$launchState["ERR_LOG"]
  }
  if (-not $PSBoundParameters.ContainsKey("BaseModel") -and $launchState.ContainsKey("BASE_MODEL")) {
    $BaseModel = [string]$launchState["BASE_MODEL"]
  }
  if (-not $PSBoundParameters.ContainsKey("Device") -and $launchState.ContainsKey("DEVICE")) {
    $Device = [string]$launchState["DEVICE"]
  }
  if (-not $PSBoundParameters.ContainsKey("ResumeWarmStartDir") -and $launchState.ContainsKey("RESUME_WARM_START_DIR")) {
    $ResumeWarmStartDir = [string]$launchState["RESUME_WARM_START_DIR"]
  }
  if (-not $PSBoundParameters.ContainsKey("LauncherPath") -and $launchState.ContainsKey("LAUNCHER")) {
    $LauncherPath = [string]$launchState["LAUNCHER"]
  }
  Write-Host "[auto-resume] restored last launch state from $launchStatePath"
}

$resolvedLauncherPath = Resolve-RepoPath $LauncherPath
if (-not $resolvedLauncherPath -or -not (Test-Path $resolvedLauncherPath)) {
  $resolvedLauncherPath = Join-Path $repoRoot "source\run_train_qwen_supermix_v26_full.ps1"
  $LauncherPath = "source\run_train_qwen_supermix_v26_full.ps1"
  Write-Host "[auto-resume] launcher not found in state file; using $LauncherPath"
}

$resolvedState = [ordered]@{
  "OUTPUT_DIR" = $OutputDir
  "OUT_LOG" = $TrainOutLog
  "ERR_LOG" = $TrainErrLog
  "BASE_MODEL" = $BaseModel
  "DEVICE" = $Device
  "RESUME_WARM_START_DIR" = $ResumeWarmStartDir
  "LAUNCHER" = $LauncherPath
  "UPDATED_AT_UTC" = [DateTime]::UtcNow.ToString("o")
}
Write-LaunchState -Path $launchStatePath -State $resolvedState

$outputPattern = [Regex]::Escape($OutputDir)
$outputLeafPattern = [Regex]::Escape((Split-Path $OutputDir -Leaf))
$launcherLeafPattern = [Regex]::Escape((Split-Path $resolvedLauncherPath -Leaf))
$trainingProcs = @(
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {
      $_.CommandLine -and
      $_.Name -match "python|pythonw|powershell|cmd" -and
      ($_.CommandLine -match "qwen_supermix_pipeline.py" -or $_.CommandLine -match $launcherLeafPattern) -and
      ($_.CommandLine -match $outputPattern -or $_.CommandLine -match $outputLeafPattern)
    }
)
if ($trainingProcs.Count -gt 0) {
  $running = ($trainingProcs | Select-Object -ExpandProperty ProcessId) -join ", "
  Write-Host "[auto-resume] training already running for $OutputDir (pid=$running)"
}
else {
  $trainOutPath = Resolve-RepoPath $TrainOutLog
  $trainErrPath = Resolve-RepoPath $TrainErrLog
  $trainCommand = (
    'powershell.exe -NoProfile -ExecutionPolicy Bypass ' +
    '-File "{0}" -BaseModel "{1}" -OutputDir "{2}" -Device "{3}" -ResumeWarmStartDir "{4}" ' +
    '1>>"{5}" 2>>"{6}"'
  ) -f $resolvedLauncherPath, $BaseModel, $OutputDir, $Device, $ResumeWarmStartDir, $trainOutPath, $trainErrPath

  if ($DryRun) {
    Write-Host "[auto-resume] dry-run config: output=$OutputDir out_log=$TrainOutLog err_log=$TrainErrLog launcher=$LauncherPath"
    Write-Host "[auto-resume] dry-run would launch training for $OutputDir"
  }
  else {
    Start-Process -FilePath "cmd.exe" -ArgumentList @("/c", $trainCommand) -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
    Write-Host "[auto-resume] launched training for $OutputDir"
  }
}

if (-not $NoMonitor) {
  $monitorNeedle = [Regex]::Escape("training_monitor_gui.py")
  $monitorProcs = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object {
        $_.CommandLine -and
        $_.Name -match "python|pythonw" -and
        $_.CommandLine -match $monitorNeedle
      }
  )
  if ($monitorProcs.Count -gt 0) {
    $running = ($monitorProcs | Select-Object -ExpandProperty ProcessId) -join ", "
    Write-Host "[auto-resume] monitor already running (pid=$running)"
  }
  else {
    if ($DryRun) {
      Write-Host "[auto-resume] dry-run would launch training monitor"
    }
    else {
      Start-Process -FilePath (Get-PythonCommand) -ArgumentList @("source\training_monitor_gui.py", "--root", ".") -WorkingDirectory $repoRoot -WindowStyle Minimized | Out-Null
      Write-Host "[auto-resume] launched training monitor"
    }
  }
}
