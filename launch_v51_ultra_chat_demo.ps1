param(
    [int]$Port = 8017,
    [string]$HostName = "127.0.0.1",
    [string]$ArtifactDir = "output\benchmark_v51_cognitive_leap_ultra_latest",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ArtifactRoot = if ([System.IO.Path]::IsPathRooted($ArtifactDir)) {
    $ArtifactDir
} else {
    Join-Path $Root $ArtifactDir
}
$Weights = Join-Path $ArtifactRoot "cognitive_leap_ultra_v51_trained.pth"
$Metrics = Join-Path $ArtifactRoot "benchmark_results.json"
$Meta = Join-Path $ArtifactRoot "chat_demo_meta.json"
$Stdout = Join-Path $ArtifactRoot "chat_web_stdout.log"
$Stderr = Join-Path $ArtifactRoot "chat_web_stderr.log"
$PidFile = Join-Path $ArtifactRoot "chat_web.pid"

function Quote-ProcessArg {
    param([Parameter(Mandatory=$true)][string]$Value)
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

if (-not (Test-Path -LiteralPath $Weights)) {
    throw "Missing v51 checkpoint: $Weights. Run source\benchmark_cognitive_leap_ultra_v51.py first."
}
if (-not (Test-Path -LiteralPath $Metrics)) {
    throw "Missing v51 benchmark metrics: $Metrics. Run source\benchmark_cognitive_leap_ultra_v51.py first."
}

Push-Location $Root
try {
    python source\materialize_v51_chat_demo.py --weights $Weights --metrics $Metrics --meta $Meta --check

    $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "Port $Port is already listening. Reusing http://$HostName`:$Port"
        if (-not $NoOpen) {
            Start-Process "http://$HostName`:$Port"
        }
        return
    }

    $args = @(
        "source\chat_web_app.py",
        "--weights", $Weights,
        "--meta", $Meta,
        "--model_size", "auto",
        "--autoload",
        "--host", $HostName,
        "--port", [string]$Port,
        "--reasoning_cycles", "3",
        "--adaptive_compute",
        "--adaptive_exit_entropy", "0.2",
        "--prediction_stability_patience", "2",
        "--prediction_stability_tol", "0.005",
        "--prediction_stability_margin", "0.0005",
        "--prediction_stability_rank_depth", "3"
    ) | ForEach-Object { Quote-ProcessArg $_ }
    $process = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $Root -WindowStyle Hidden -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    Set-Content -LiteralPath $PidFile -Value $process.Id
    Write-Host "Started v51 ultra chat demo: http://$HostName`:$Port (pid $($process.Id))"
    if (-not $NoOpen) {
        Start-Process "http://$HostName`:$Port"
    }
}
finally {
    Pop-Location
}
