# Register the exam-station band relay as a scheduled task.
#
# Called by install_station_relay.bat (which handles elevation). Registration
# goes through PowerShell rather than schtasks because several schtasks
# defaults are actively wrong for a machine that must keep working unattended
# — see the settings block below, each of which caused or would have caused a
# real failure.

$ErrorActionPreference = 'Stop'

$task   = 'VivaSenseStationRelay'
$script = Join-Path $PSScriptRoot 'run_station_relay.bat'

if (-not (Test-Path $script)) {
    Write-Host "Cannot find run_station_relay.bat next to this script." -ForegroundColor Red
    exit 1
}

# Warn rather than fail: a station pointed at the wrong backend still installs,
# it just will not authenticate, and that is easier to spot than a silent skip.
$launcher = Get-Content $script -Raw
if ($launcher -match 'STATION_TOKEN=CHANGE_ME') {
    Write-Host "STATION_TOKEN is still CHANGE_ME in run_station_relay.bat." -ForegroundColor Yellow
    Write-Host "The relay will install but every post will be rejected." -ForegroundColor Yellow
}

Write-Host "Registering scheduled task '$task'..."
Unregister-ScheduledTask -TaskName $task -Confirm:$false -ErrorAction SilentlyContinue

$action  = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument ('/c "' + $script + '"')
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERNAME"

# Every one of these overrides a schtasks default that breaks unattended use:
#
#   AllowStartIfOnBatteries / DontStopIfGoingOnBatteries
#     schtasks blocks on battery by default. On an unplugged laptop the task
#     sits in "Queued" and never runs, and if power is pulled mid-viva the
#     relay is killed outright.
#
#   ExecutionTimeLimit 0
#     The default is 72 hours, after which Windows terminates a process that
#     is meant to run indefinitely.
#
#   RestartCount / RestartInterval
#     If the relay dies for a reason its own loop cannot catch — a Bluetooth
#     driver reload, say — Windows brings it back.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

# Interactive, not SYSTEM. Windows binds its BLE stack to a user session: a
# SYSTEM task can usually scan for devices but fails to CONNECT to a GATT
# server. The exam station always has someone logged in — the kiosk is a
# browser and needs a desktop — so this is both safer and sufficient.
$principal = New-ScheduledTaskPrincipal `
    -UserId ("$env:USERDOMAIN\$env:USERNAME") `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal | Out-Null

Start-ScheduledTask -TaskName $task
Start-Sleep -Seconds 8

$t = Get-ScheduledTask -TaskName $task
$i = Get-ScheduledTaskInfo -TaskName $task
Write-Host ''
Write-Host "  state              : $($t.State)"
Write-Host "  runs on battery    : $(-not $t.Settings.DisallowStartIfOnBatteries)"
Write-Host "  time limit         : $(if ($t.Settings.ExecutionTimeLimit -eq 'PT0S') {'none'} else {$t.Settings.ExecutionTimeLimit})"
Write-Host "  last result        : $($i.LastTaskResult)"

$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*station_sidecar*' }
if ($proc) {
    Write-Host "  relay process      : RUNNING" -ForegroundColor Green
} else {
    Write-Host "  relay process      : not up yet - check the console window" -ForegroundColor Yellow
}
