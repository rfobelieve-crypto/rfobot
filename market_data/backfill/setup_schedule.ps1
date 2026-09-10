# 2026-09-10: the path below pointed at the pre-rename folder
# (flowbot\資金機器人\...). Running this file as it stood would have
# registered a task aimed at a folder that no longer exists -- exactly
# the 2026-07-05 incident, sitting in a file nobody could see because
# it was gitignored. Fixed, and now tracked.
$taskName = "FlowBot_DailyCollect"
$batPath = "C:\Users\rfo\Desktop\flowbotlow_system\market_data\backfill\daily_collect.bat"

$action = New-ScheduledTaskAction -Execute $batPath
$trigger = New-ScheduledTaskTrigger -Daily -At "04:00"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries

# Remove existing task if any
Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Description "Daily Coinglass data collection + D drive backup"

Write-Host "Scheduled task '$taskName' created. Runs daily at 04:00."
Write-Host "To verify: Get-ScheduledTaskInfo -TaskName '$taskName'"
