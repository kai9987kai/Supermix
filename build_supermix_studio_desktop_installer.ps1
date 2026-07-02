param(
  [string]$InstallerScript = "installer\SupermixStudioDesktop.iss",
  [string]$Version = "",
  [string]$SetupBaseName = ""
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$CommonIsccPaths = @(
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
  "C:\Program Files\Inno Setup 6\ISCC.exe",
  "C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
  (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe")
)
$IsccPath = $CommonIsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $IsccPath) {
  throw "Inno Setup compiler not found. Install Inno Setup 6, then rerun this script."
}

$IsccArgs = @()
if ($Version) {
  $IsccArgs += "/DMyAppVersion=$Version"
}
if ($SetupBaseName) {
  $IsccArgs += "/DMySetupBaseName=$SetupBaseName"
}
$IsccArgs += $InstallerScript

& $IsccPath @IsccArgs
if ($LASTEXITCODE -ne 0) {
  throw "Inno Setup build failed."
}

$InstallerDir = Join-Path $RepoRoot "dist\installer"
$ExePath = Join-Path $RepoRoot "dist\SupermixStudioDesktop\SupermixStudioDesktop.exe"
$SetupPath = Join-Path $InstallerDir "SupermixStudioDesktopSetup.exe"
$BundleManifestSource = Join-Path $RepoRoot "output\supermix_studio_bundled_models_manifest.json"
$BundleManifestTarget = Join-Path $InstallerDir "SupermixStudioDesktopCuratedBundleManifest.json"
$HashFilePath = Join-Path $InstallerDir "SupermixStudioDesktopReleaseSHA256.txt"

if (Test-Path $BundleManifestSource) {
  Copy-Item -Force $BundleManifestSource $BundleManifestTarget
}

$HashTargets = @()
if (Test-Path $SetupPath) { $HashTargets += Get-Item $SetupPath }
if (Test-Path $ExePath) { $HashTargets += Get-Item $ExePath }
if (Test-Path $BundleManifestTarget) { $HashTargets += Get-Item $BundleManifestTarget }

if ($HashTargets.Count -gt 0) {
  $HashLines = foreach ($Item in $HashTargets) {
    $Hash = (Get-FileHash -Algorithm SHA256 $Item.FullName).Hash
    "$($Item.Name)  $Hash"
  }
  Set-Content -Encoding ASCII -Path $HashFilePath -Value $HashLines
}
