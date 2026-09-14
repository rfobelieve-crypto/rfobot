# register_hmm_watch.ps1 — 把 HMM live 看護註冊成 Windows 排程（2026-09-14）
#
# 為什麼要有這個檔而不是「跑一次指令就算了」：
# **排程的 action 是 grep 不到的**。mistake.md 2026-09-04：套利線搬 repo 時
# 枚舉出 10 個持有舊路徑的地方，第 11 個是看門狗的排程 action，指著一個
# 已經不存在的檔案——它不在任何 grep 結果裡，因為它根本不在 repo 裡。
# 把註冊動作留成檔案，至少讓「這台機器上有這個排程」這件事在 repo 裡看得見。
#
# 為什麼是排程而不是背景迴圈：
# 2026-09-14 晚上，看護跑在一個 shell 背景迴圈裡，被 harness 因記憶體壓力
# **殺了兩次**（而壓力是我自己的量測造成的）。一個會送真單的引擎，它的
# 安全網不可以依賴某個對話 session 活著。
#
# 為什麼要 run_hidden.vbs：
# schtasks / Register-ScheduledTask 建的工作預設 LogonType=Interactive，
# 每次觸發都開一個主控台視窗（每 5 分鐘一次）。改成 S4U 要管理員權限。
# wscript.exe 本身沒有主控台，所以用它當包裝（mistake.md 2026-09-06）。
#
# 驗收判準是**產物**：research\results\hmm_watch.log 長出新行。
# 不是 LastTaskResult——非同步啟動器的退出碼跟被啟動的東西無關。

$ErrorActionPreference = 'Stop'
$ops  = 'C:\Users\rfo\Desktop\flowbot\flow_system\research\ops'
$name = 'FlowBot_HmmWatch'

foreach ($f in @('run_hidden.vbs', 'hmm_watch.bat', 'hmm_watch.py', 'live_hmm.py')) {
    if (-not (Test-Path (Join-Path $ops $f))) {
        throw "$f 不在 $ops —— 先別註冊，會註冊出一條壞路徑"
    }
}

$arg = '//B //Nologo "{0}\run_hidden.vbs" "{0}\hmm_watch.bat"' -f $ops
$act = New-ScheduledTaskAction -Execute 'wscript.exe' -Argument $arg
$trg = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5)
# IgnoreNew：一輪還沒跑完就不要再疊一個上去（看護會呼叫 PowerShell 查行程，
# 偶爾會慢）。ExecutionTimeLimit 10 分鐘：卡住的話自己收掉，不要累積殭屍。
$set = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

Register-ScheduledTask -TaskName $name -Action $act -Trigger $trg -Settings $set -Force | Out-Null

$t = Get-ScheduledTask -TaskName $name
Write-Output ('已註冊 ' + $name)
Write-Output ('  Execute  : ' + $t.Actions[0].Execute)
Write-Output ('  Arguments: ' + $t.Actions[0].Arguments)
Write-Output ('  Repeat   : ' + $t.Triggers[0].Repetition.Interval)
Write-Output ('  State    : ' + $t.State)
Write-Output ''
Write-Output '驗收看產物，不看退出碼：research\results\hmm_watch.log 要長出新行'
