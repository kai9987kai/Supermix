<#
.SYNOPSIS
Package Supermix Chat for distribution.

.DESCRIPTION
Produces an installer from dist\SupermixChatDesktop.

If Inno Setup 6 is installed, this compiles installer\SupermixChatDesktop.iss
into a single setup .exe -- the better artifact, and the one to prefer.

If it is not installed, this falls back to a zip plus a PowerShell installer
that needs no tooling at all. The fallback is a real installer, not a copy
script: it installs per-user, creates Start Menu and optional desktop
shortcuts, registers in Add/Remove Programs, and writes a matching uninstaller.

.PARAMETER Version
Version stamped into the installer and Add/Remove Programs.

.EXAMPLE
.\build_chat_desktop_installer.ps1
#>
[CmdletBinding()]
param(
    [string]$Version = "74.0.0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$appDir = Join-Path $repoRoot "dist\SupermixChatDesktop"
$outDir = Join-Path $repoRoot "dist\installer"

if (-not (Test-Path (Join-Path $appDir "SupermixChatDesktop.exe"))) {
    throw "no built application at $appDir. Run .\build_chat_desktop_exe.ps1 first."
}
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# -- preferred path: a real setup.exe -------------------------------------
$iscc = $null
foreach ($candidate in @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)) {
    if (Test-Path $candidate) { $iscc = $candidate; break }
}
if (-not $iscc) {
    $onPath = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($onPath) { $iscc = $onPath.Source }
}

if ($iscc) {
    Write-Host "Compiling installer with Inno Setup" -ForegroundColor Cyan
    & $iscc "/DMyAppVersion=$Version" (Join-Path $repoRoot "installer\SupermixChatDesktop.iss")
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
    $setup = Join-Path $outDir "SupermixChatSetup.exe"
    Write-Host ""
    Write-Host "Built: $setup" -ForegroundColor Green
    exit 0
}

Write-Host "Inno Setup 6 not found -- building the no-tooling installer instead." -ForegroundColor Yellow
Write-Host "  (install Inno Setup and re-run for a single setup.exe)" -ForegroundColor Yellow

# -- fallback: zip + PowerShell installer ---------------------------------
$zipPath = Join-Path $outDir "SupermixChat-$Version.zip"
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

$appMb = (Get-ChildItem $appDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host ("Compressing {0:N0} MB (several minutes)" -f $appMb) -ForegroundColor Cyan

# Not Compress-Archive: it is slow on this many files and has historically
# been unreliable near the 2 GB mark, which a ~1.8 GB torch application sits
# right on. ZipFile::CreateFromDirectory is the same .NET machinery without
# the wrapper's limits.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $appDir, $zipPath,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false   # do not nest the directory inside the archive
)

$installer = @'
<#
.SYNOPSIS
Install Supermix Chat for the current user.

.DESCRIPTION
Installs to %LOCALAPPDATA%\Programs\Supermix Chat. No administrator rights are
needed and nothing is written outside your user profile.

.PARAMETER DesktopShortcut
Also create a desktop shortcut.

.PARAMETER Uninstall
Remove a previous installation.
#>
[CmdletBinding()]
param(
    [switch]$DesktopShortcut,
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"

$appName    = "Supermix Chat"
$exeName    = "SupermixChatDesktop.exe"
$version    = "__VERSION__"
$installDir = Join-Path $env:LOCALAPPDATA "Programs\Supermix Chat"
$startMenu  = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\$appName.lnk"
$desktopLnk = Join-Path ([Environment]::GetFolderPath("Desktop")) "$appName.lnk"
$regKey     = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\SupermixChat"

function Remove-Installation {
    foreach ($link in @($startMenu, $desktopLnk)) {
        if (Test-Path $link) { Remove-Item $link -Force }
    }
    if (Test-Path $regKey) { Remove-Item $regKey -Recurse -Force }
    if (Test-Path $installDir) {
        # Refuse rather than delete a directory that is not ours.
        if (-not (Test-Path (Join-Path $installDir $exeName))) {
            throw "$installDir does not look like a $appName installation; not deleting it"
        }
        Remove-Item $installDir -Recurse -Force
    }
    Write-Host "$appName has been removed." -ForegroundColor Green
}

if ($Uninstall) { Remove-Installation; exit 0 }

$zip = Join-Path $PSScriptRoot "SupermixChat-$version.zip"
if (-not (Test-Path $zip)) { throw "cannot find $zip next to this script" }

if (Test-Path (Join-Path $installDir $exeName)) {
    Write-Host "Removing the previous installation" -ForegroundColor Cyan
    Remove-Installation
}

Write-Host "Installing $appName $version to $installDir" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $installDir | Out-Null
Expand-Archive -Path $zip -DestinationPath $installDir -Force

$exePath = Join-Path $installDir $exeName
if (-not (Test-Path $exePath)) { throw "the archive did not contain $exeName" }

$shell = New-Object -ComObject WScript.Shell
foreach ($link in @($startMenu) + $(if ($DesktopShortcut) { @($desktopLnk) } else { @() })) {
    $shortcut = $shell.CreateShortcut($link)
    $shortcut.TargetPath       = $exePath
    $shortcut.WorkingDirectory = $installDir
    $shortcut.IconLocation     = $exePath
    $shortcut.Description      = "$appName - offline arithmetic problem solver"
    $shortcut.Save()
}

# Appear in Settings > Apps, so this can be uninstalled the usual way.
$size = (Get-ChildItem $installDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
New-Item -Path $regKey -Force | Out-Null
Set-ItemProperty -Path $regKey -Name "DisplayName"     -Value $appName
Set-ItemProperty -Path $regKey -Name "DisplayVersion"  -Value $version
Set-ItemProperty -Path $regKey -Name "Publisher"       -Value "Supermix"
Set-ItemProperty -Path $regKey -Name "InstallLocation" -Value $installDir
Set-ItemProperty -Path $regKey -Name "DisplayIcon"     -Value $exePath
Set-ItemProperty -Path $regKey -Name "NoModify"        -Value 1 -Type DWord
Set-ItemProperty -Path $regKey -Name "NoRepair"        -Value 1 -Type DWord
Set-ItemProperty -Path $regKey -Name "EstimatedSize"   -Value ([int]($size / 1KB)) -Type DWord
Set-ItemProperty -Path $regKey -Name "UninstallString" `
    -Value "powershell.exe -ExecutionPolicy Bypass -File `"$installDir\Uninstall-SupermixChat.ps1`""

# Leave an uninstaller behind that does not depend on this script surviving.
$uninstallScript = @"
`$ErrorActionPreference = 'Stop'
`$installDir = '$installDir'
foreach (`$link in @('$startMenu', '$desktopLnk')) {
    if (Test-Path `$link) { Remove-Item `$link -Force }
}
if (Test-Path '$regKey') { Remove-Item '$regKey' -Recurse -Force }
if (Test-Path `$installDir) {
    if (-not (Test-Path (Join-Path `$installDir '$exeName'))) {
        throw "`$installDir does not look like a $appName installation; not deleting it"
    }
    Start-Process powershell -ArgumentList '-NoProfile','-Command',"Start-Sleep 1; Remove-Item -LiteralPath '`$installDir' -Recurse -Force" -WindowStyle Hidden
}
Write-Host '$appName has been removed.'
"@
Set-Content -Path (Join-Path $installDir "Uninstall-SupermixChat.ps1") -Value $uninstallScript -Encoding utf8

Write-Host ""
Write-Host "Installed." -ForegroundColor Green
Write-Host "  Start Menu : $appName"
Write-Host "  Location   : $installDir"
Write-Host "  Uninstall  : Settings > Apps, or run Uninstall-SupermixChat.ps1"
'@

$installer = $installer.Replace("__VERSION__", $Version)
$installerPath = Join-Path $outDir "Install-SupermixChat.ps1"
Set-Content -Path $installerPath -Value $installer -Encoding utf8

Copy-Item (Join-Path $repoRoot "installer\postinstall_notes_chat.txt") `
          (Join-Path $outDir "README.txt") -Force

$zipMb = (Get-Item $zipPath).Length / 1MB
Write-Host ""
Write-Host "Built the no-tooling installer:" -ForegroundColor Green
Write-Host ("  {0}  ({1:N0} MB)" -f $zipPath, $zipMb)
Write-Host "  $installerPath"
Write-Host ""
Write-Host "To install, run from that folder:" -ForegroundColor Cyan
Write-Host "  powershell -ExecutionPolicy Bypass -File .\Install-SupermixChat.ps1 -DesktopShortcut"
