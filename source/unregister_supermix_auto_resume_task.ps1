param(
  [string]$TaskName = "SupermixQwenTrainingAutoResume"
)

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$removed = $false
$taskNames = @(
  $TaskName,
  "$TaskName-Startup",
  "$TaskName-Logon"
)

foreach ($name in $taskNames) {
  try {
    $task = Get-ScheduledTask -TaskName $name -ErrorAction Stop
    if ($task) {
      Unregister-ScheduledTask -TaskName $name -Confirm:$false -ErrorAction Stop
      Write-Host "[boot] removed scheduled task: name=$name"
      $removed = $true
    }
  }
  catch {
  }
}

$runKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$runValue = $null
try {
  $runValue = (Get-ItemProperty -Path $runKey -Name $TaskName -ErrorAction Stop).$TaskName
}
catch {
}

if ($runValue) {
  Remove-ItemProperty -Path $runKey -Name $TaskName -ErrorAction Stop
  Write-Host "[boot] removed HKCU Run entry: name=$TaskName"
  $removed = $true
}

if (-not $removed) {
  Write-Host "[boot] no auto-resume boot/logon hook found: name=$TaskName"
}
