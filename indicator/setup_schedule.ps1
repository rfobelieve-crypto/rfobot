$taskName = "FlowBot_IndicatorUpdate"
$batPath = "C:\Users\rfo\Desktop\flowbot\資金機器人\indicator\auto_update.bat"

$action = New-ScheduledTaskAction -Execute $batPath
# Every 1 hour (bars are 1h)
$trigger = New-ScheduledTaskTrigger -Once -At "00:05" -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

# Remove existing
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "BTC Indicator chart update every 1h -> Telegram"

Write-Host "Scheduled task '$taskName' created. Runs every 1 hour."
Write-Host "To verify: Get-ScheduledTaskInfo -TaskName '$taskName'"
