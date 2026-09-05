# 註冊出路研究線錄製器的看門狗排程（2026-09-05）
# 用檔案而不是 -Command 字串：路徑穿過 bash → powershell 的轉義層會壞掉
# （mistake.md 2026-08-20 的同族）。
$Name   = 'FlowBot_ExitPathsWatchdog'
$Script = 'C:\Users\rfo\Desktop\flowbot\flow_system\research\ops\exit_paths_watchdog.ps1'

$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
          -Argument ('-NoProfile -ExecutionPolicy Bypass -File "{0}"' -f $Script)
$t1 = New-ScheduledTaskTrigger -AtStartup
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
      -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
$set = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
       -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
       -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $Name -Action $action -Trigger $t1, $t2 -Settings $set -Force | Out-Null
Start-ScheduledTask -TaskName $Name
Start-Sleep -Seconds 20
Get-ScheduledTask -TaskName $Name | Select-Object TaskName, State | Format-Table -AutoSize
(Get-ScheduledTaskInfo -TaskName $Name) | Select-Object LastTaskResult, LastRunTime | Format-Table -AutoSize
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*recorder.py*' } |
  ForEach-Object { 'RUNNING: ' + ($_.CommandLine -split 'flow_system.')[-1] }
