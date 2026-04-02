$taskName = "FlowBot_IndicatorUpdate"
$batPath = "C:\Users\rfo\Desktop\flowbot\run_indicator.bat"

$action = New-ScheduledTaskAction -Execute $batPath -WorkingDirectory "C:\Users\rfo\Desktop\flowbot"
# Every 1 hour (bars are 1h)
$trigger = New-ScheduledTaskTrigger -Once -At "00:00" -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

# Remove existing
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "BTC Indicator chart update every 1h -> Telegram"

Write-Host "Scheduled task created. Runs every 1 hour."
