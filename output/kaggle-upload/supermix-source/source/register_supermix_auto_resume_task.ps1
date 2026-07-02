param(
  [string]$TaskName = "SupermixQwenTrainingAutoResume"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$autoResumePath = (Resolve-Path (Join-Path $PSScriptRoot "auto_resume_supermix_training.ps1")).Path
$startupTaskName = "$TaskName-Startup"
$logonTaskName = "$TaskName-Logon"
$startupArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$autoResumePath`" -NoMonitor"
$logonArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$autoResumePath`""
$runCommand = "powershell.exe $logonArgs"
$userId = if ($env:USERDOMAIN) { "$($env:USERDOMAIN)\$($env:USERNAME)" } else { $env:USERNAME }
$registeredStartup = $false
$registeredLogon = $false

function Remove-ScheduledTaskIfPresent([string]$Name) {
  try {
    $task = Get-ScheduledTask -TaskName $Name -ErrorAction Stop
    if ($task) {
      Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction Stop
      Write-Host "[boot] removed existing scheduled task: name=$Name"
    }
  }
  catch {
  }
}

function Remove-RunEntryIfPresent([string]$Name) {
  $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
  try {
    $current = (Get-ItemProperty -Path $runKey -Name $Name -ErrorAction Stop).$Name
    if ($current) {
      Remove-ItemProperty -Path $runKey -Name $Name -ErrorAction Stop
      Write-Host "[boot] removed HKCU Run entry: name=$Name"
    }
  }
  catch {
  }
}

Remove-ScheduledTaskIfPresent -Name $TaskName
Remove-ScheduledTaskIfPresent -Name $startupTaskName
Remove-ScheduledTaskIfPresent -Name $logonTaskName
Remove-RunEntryIfPresent -Name $TaskName

try {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $startupArgs
  $trigger = New-ScheduledTaskTrigger -AtStartup
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
  $principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType S4U -RunLevel Limited

  Register-ScheduledTask `
    -TaskName $startupTaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force `
    -ErrorAction Stop | Out-Null

  $task = Get-ScheduledTask -TaskName $startupTaskName -ErrorAction Stop
  Write-Host "[boot] registered startup task: name=$startupTaskName state=$($task.State)"
  $registeredStartup = $true
}
catch {
  Write-Host "[boot] startup scheduled task unavailable: $($_.Exception.Message)"
}

try {
  $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $logonArgs
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
  $principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Limited

  Register-ScheduledTask `
    -TaskName $logonTaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Force `
    -ErrorAction Stop | Out-Null

  $task = Get-ScheduledTask -TaskName $logonTaskName -ErrorAction Stop
  Write-Host "[boot] registered logon monitor task: name=$logonTaskName state=$($task.State)"
  $registeredLogon = $true
}
catch {
  Write-Host "[boot] logon scheduled task unavailable, falling back to HKCU Run: $($_.Exception.Message)"
}

if (-not $registeredLogon) {
  $runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
  New-Item -Path $runKey -Force | Out-Null
  Set-ItemProperty -Path $runKey -Name $TaskName -Value $runCommand -Type String
  $value = (Get-ItemProperty -Path $runKey -Name $TaskName).$TaskName
  Write-Host "[boot] registered HKCU Run entry for logon fallback: name=$TaskName command=$value"
}

if (-not $registeredStartup) {
  Write-Host "[boot] training boot-resume will fall back to logon until a startup scheduled task can be created."
}
