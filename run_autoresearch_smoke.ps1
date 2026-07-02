param(
  [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
  [string]$Device = "auto",
  [string]$ResumeWarmStartDir = "artifacts\qwen_supermix_enhanced_v26_full",
  [string]$SmokeLauncherPath = "run_train_qwen_supermix_v28_smoke.ps1",
  [string]$OutputRoot = "artifacts\autoresearch",
  [string]$BenchmarkRoot = "artifacts\research_baselines",
  [string]$ResultsFile = "research\results.tsv",
  [string]$RunTag = "",
  [string]$Description = "autoresearch smoke",
  [string[]]$ExtraTrainingArgs = @()
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

function Ensure-ResultsFile([string]$Path) {
  $header = "timestamp`tcommit`trun_tag`toutput_dir`tbenchmark_json`ttoken_f1_delta`tchar_similarity_delta`tperplexity_delta`tavg_gen_seconds_delta`tstatus`tdescription"
  if (-not (Test-Path $Path)) {
    $dir = Split-Path -Parent $Path
    if ($dir) {
      New-Item -Path $dir -ItemType Directory -Force | Out-Null
    }
    Set-Content -Path $Path -Value $header -Encoding UTF8
    return
  }
  $firstLine = Get-Content -Path $Path -TotalCount 1 -ErrorAction SilentlyContinue
  if ([string]::IsNullOrWhiteSpace($firstLine)) {
    Set-Content -Path $Path -Value $header -Encoding UTF8
  }
}

function Get-BestLoggedTokenF1Delta([string]$Path) {
  if (-not (Test-Path $Path)) {
    return $null
  }
  $rows = Import-Csv -Path $Path -Delimiter "`t"
  $kept = @(
    $rows |
      Where-Object { $_.status -eq "keep" -and $_.token_f1_delta -match "^-?[0-9]+(\.[0-9]+)?$" } |
      ForEach-Object { [double]$_.token_f1_delta }
  )
  if ($kept.Count -eq 0) {
    return $null
  }
  return ($kept | Measure-Object -Maximum).Maximum
}

function Append-ResultsRow(
  [string]$Path,
  [string]$Commit,
  [string]$RunTagValue,
  [string]$OutputDirValue,
  [string]$BenchmarkJson,
  [double]$TokenF1Delta,
  [double]$CharSimilarityDelta,
  [double]$PerplexityDelta,
  [double]$AvgGenSecondsDelta,
  [string]$Status,
  [string]$DescriptionValue
) {
  $timestamp = [DateTime]::UtcNow.ToString("o")
  $line = @(
    $timestamp,
    $Commit,
    $RunTagValue,
    $OutputDirValue,
    $BenchmarkJson,
    ("{0:F6}" -f $TokenF1Delta),
    ("{0:F6}" -f $CharSimilarityDelta),
    ("{0:F6}" -f $PerplexityDelta),
    ("{0:F6}" -f $AvgGenSecondsDelta),
    $Status,
    ($DescriptionValue -replace "`t", " ")
  ) -join "`t"
  Add-Content -Path $Path -Value $line -Encoding UTF8
}

if ([string]::IsNullOrWhiteSpace($RunTag)) {
  $RunTag = "ar_" + [DateTime]::UtcNow.ToString("yyyyMMdd_HHmmss")
}

$resolvedOutputRoot = Resolve-RepoPath $OutputRoot
$resolvedBenchmarkRoot = Resolve-RepoPath $BenchmarkRoot
$resolvedResultsFile = Resolve-RepoPath $ResultsFile
$resolvedSmokeLauncher = Resolve-RepoPath $SmokeLauncherPath
$resolvedOutputDir = Join-Path $resolvedOutputRoot $RunTag
$resolvedBenchmarkDir = Join-Path $resolvedBenchmarkRoot $RunTag
$resolvedAdapterDir = Join-Path $resolvedOutputDir "adapter"
$resolvedEvalJsonl = Join-Path $resolvedOutputDir "eval_pairs.jsonl"
$resolvedBenchmarkJson = Join-Path $resolvedBenchmarkDir "benchmark_results.json"
$python = Get-PythonCommand

Ensure-ResultsFile -Path $resolvedResultsFile

$commit = "unknown"
try {
  $commit = (git rev-parse --short HEAD).Trim()
} catch {
}

Write-Host "[autoresearch] run_tag=$RunTag"
Write-Host "[autoresearch] output_dir=$resolvedOutputDir"
if ($ExtraTrainingArgs.Count -gt 0) {
  Write-Host "[autoresearch] extra_training_args=$($ExtraTrainingArgs -join ' ')"
}

$launcherArgs = @{
  BaseModel = $BaseModel
  OutputDir = $resolvedOutputDir
  Device = $Device
  ResumeWarmStartDir = $ResumeWarmStartDir
  ExtraArgs = $ExtraTrainingArgs
}

$launcherExitCode = 0
& $resolvedSmokeLauncher @launcherArgs
if ($LASTEXITCODE -is [int]) {
  $launcherExitCode = [int]$LASTEXITCODE
}

if ($launcherExitCode -ne 0 -or -not (Test-Path $resolvedAdapterDir) -or -not (Test-Path $resolvedEvalJsonl)) {
  Append-ResultsRow `
    -Path $resolvedResultsFile `
    -Commit $commit `
    -RunTagValue $RunTag `
    -OutputDirValue $resolvedOutputDir `
    -BenchmarkJson "" `
    -TokenF1Delta 0.0 `
    -CharSimilarityDelta 0.0 `
    -PerplexityDelta 0.0 `
    -AvgGenSecondsDelta 0.0 `
    -Status "crash" `
    -DescriptionValue $Description
  throw "Autoresearch smoke run failed before benchmark."
}

& $python "source\run_research_baseline.py" `
  --base_model $BaseModel `
  --eval_jsonl $resolvedEvalJsonl `
  --adapter_dir $resolvedAdapterDir `
  --max_length 448 `
  --max_new_tokens 112 `
  --output_root $resolvedBenchmarkRoot `
  --run_name $RunTag `
  --benchmark_type "autoresearch_smoke" `
  --device cpu

if ($LASTEXITCODE -ne 0 -or -not (Test-Path $resolvedBenchmarkJson)) {
  Append-ResultsRow `
    -Path $resolvedResultsFile `
    -Commit $commit `
    -RunTagValue $RunTag `
    -OutputDirValue $resolvedOutputDir `
    -BenchmarkJson "" `
    -TokenF1Delta 0.0 `
    -CharSimilarityDelta 0.0 `
    -PerplexityDelta 0.0 `
    -AvgGenSecondsDelta 0.0 `
    -Status "crash" `
    -DescriptionValue "$Description (benchmark failed)"
  throw "Autoresearch benchmark failed."
}

$benchmark = Get-Content -Path $resolvedBenchmarkJson -Raw | ConvertFrom-Json
$delta = $benchmark.delta_tuned_minus_base

$tokenF1Delta = [double]($delta.token_f1)
$charSimilarityDelta = [double]($delta.char_similarity)
$perplexityDelta = [double]($delta.perplexity)
$avgGenSecondsDelta = [double]($delta.avg_gen_seconds)
$bestLogged = Get-BestLoggedTokenF1Delta -Path $resolvedResultsFile
$beatsBase = ($tokenF1Delta -gt 0.0 -and $charSimilarityDelta -ge 0.0)

$status = "discard"
if ($beatsBase) {
  if ($null -eq $bestLogged -or $tokenF1Delta -gt [double]$bestLogged) {
    $status = "keep"
  }
}

Append-ResultsRow `
  -Path $resolvedResultsFile `
  -Commit $commit `
  -RunTagValue $RunTag `
  -OutputDirValue $resolvedOutputDir `
  -BenchmarkJson $resolvedBenchmarkJson `
  -TokenF1Delta $tokenF1Delta `
  -CharSimilarityDelta $charSimilarityDelta `
  -PerplexityDelta $perplexityDelta `
  -AvgGenSecondsDelta $avgGenSecondsDelta `
  -Status $status `
  -DescriptionValue $Description

Write-Host "[autoresearch] status=$status token_f1_delta=$("{0:F6}" -f $tokenF1Delta) char_similarity_delta=$("{0:F6}" -f $charSimilarityDelta)"
Write-Host "[autoresearch] results_tsv=$resolvedResultsFile"
