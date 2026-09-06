' run_hidden.vbs — 讓排程的工作在背景跑，不彈視窗（2026-09-06）
'
' 為什麼需要它：schtasks 建立的工作預設是 Interactive，每次觸發都會開一個
' 主控台視窗。FlowBot_ExitPathsWatchdog 與 EntropyArbWatchdog 每 5 分鐘各一次，
' SweepShadow 每小時一次而且會開著跑很久——桌面上一直有視窗跳出來。
'
' 正規解法是把工作的 LogonType 改成 S4U（背景執行），但那需要管理員權限。
' 這個包裝器不需要：wscript.exe 本身沒有主控台，Run 的第二個參數 0 = 隱藏視窗。
'
' 用法（由排程呼叫）：
'   wscript.exe //B //Nologo "…\run_hidden.vbs" "…\某支.ps1"
'   wscript.exe //B //Nologo "…\run_hidden.vbs" "…\某支.bat"
' 依副檔名分派到 powershell 或 cmd。多給的參數會原樣接在後面。

Option Explicit
Dim sh, target, ext, cmd, i, extra

If WScript.Arguments.Count = 0 Then
  WScript.Quit 2
End If

Set sh = CreateObject("WScript.Shell")
target = WScript.Arguments(0)

extra = ""
For i = 1 To WScript.Arguments.Count - 1
  extra = extra & " " & Chr(34) & WScript.Arguments(i) & Chr(34)
Next

ext = LCase(Right(target, 4))

If ext = ".ps1" Then
  cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File " _
        & Chr(34) & target & Chr(34) & extra
ElseIf ext = ".bat" Or ext = ".cmd" Then
  cmd = "cmd.exe /c " & Chr(34) & Chr(34) & target & Chr(34) & extra & Chr(34)
Else
  cmd = Chr(34) & target & Chr(34) & extra
End If

' 0 = 隱藏視窗，True = 等它結束。
' 用 True 不用 False 有兩個理由：(a) 實測 False 版本不會真的啟動子行程（同一支
' 腳本改成 True 立刻可用，2026-09-06 反向證明）；(b) 等待讓排程的
' MultipleInstances=IgnoreNew 生效，長時間的 bat 不會疊起來跑。
Dim rc
rc = sh.Run(cmd, 0, True)
WScript.Quit rc
