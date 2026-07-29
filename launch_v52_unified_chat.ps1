param(
    [int]$Port = 8052,
    [string]$HostName = "127.0.0.1",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Weights = Join-Path $Root "artifacts\v52_initialization\champion_model_chat_v52_initialized.pth"
$Meta = Join-Path $Root "chat_model_meta_v50_cognitive_leap.json"
$RunDir = Join-Path $Root "output\v52_initialized_chat"
$Stdout = Join-Path $RunDir "chat_web_stdout.log"
$Stderr = Join-Path $RunDir "chat_web_stderr.log"
$PidFile = Join-Path $RunDir "chat_web.pid"

if (-not (Test-Path -LiteralPath $Weights)) {
    throw "Missing materialized v52 initialization checkpoint: $Weights. Run source\materialize_v52_from_v50.py first."
}
if (-not (Test-Path -LiteralPath $Meta)) {
    throw "Missing conversational metadata: $Meta"
}
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Quote-ProcessArg {
    param([Parameter(Mandatory=$true)][string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

$existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Port $Port is already listening. Reusing http://$HostName`:$Port"
    if (-not $NoOpen) { Start-Process "http://$HostName`:$Port" }
    return
}

$arguments = @(
    "source\chat_web_app.py",
    "--weights", $Weights,
    "--meta", $Meta,
    "--model_size", "cognitive_leap_v52_expert",
    "--autoload",
    "--host", $HostName,
    "--port", [string]$Port,
    "--reasoning_cycles", "auto",
    "--auto_compute",
    "--adaptive_compute",
    "--core_top_k", "2",
    "--prediction_stability_patience", "2",
    "--prediction_stability_tol", "0.005"
) | ForEach-Object { Quote-ProcessArg $_ }

$process = Start-Process -FilePath "python" -ArgumentList $arguments -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
Set-Content -LiteralPath $PidFile -Value $process.Id
Write-Host "Started initialized v52 chat: http://$HostName`:$Port (pid $($process.Id))"
Write-Warning "The imported v50 weights initialize v52, but the new verifier and affect heads are not fine-tuned yet."
if (-not $NoOpen) { Start-Process "http://$HostName`:$Port" }
