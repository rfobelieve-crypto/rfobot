# 出路研究線的錄製器看門狗（2026-09-05）
#
# 判斷「活著」的依據是**產物**不是進程：liq_last.json / lighter_last.json 的
# asof 落後就重啟——一個卡在 WS 讀取上的殭屍進程，工作管理員看起來完全正常
# （mistake.md 2026-08-19：排程 State=Ready 但工作早就死了）。
#
# 掛在 Windows 排程，每 5 分鐘跑一次 + 開機時跑一次。

$ErrorActionPreference = 'Continue'
$Root = 'C:\Users\rfo\Desktop\flowbot\flow_system'
$Log  = Join-Path $Root 'research\exit_paths\logs\watchdog.log'
$Py   = 'python'
$StaleMin = 20

New-Item -ItemType Directory -Force -Path (Split-Path $Log) | Out-Null

function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $m" | Add-Content -Path $Log }

# name = 顯示名, script = 相對路徑, flag = 新鮮度旗標, log = 輸出檔
$Jobs = @(
  @{ name = 'liq';     script = 'research\exit_paths\liq_recorder.py';     flag = 'research\results\liq_last.json';     log = 'research\exit_paths\logs\liq_recorder.log' },
  @{ name = 'lighter'; script = 'research\exit_paths\lighter_recorder.py'; flag = 'research\results\lighter_last.json'; log = 'research\exit_paths\logs\lighter_recorder.log' }
)

foreach ($j in $Jobs) {
  $script = Join-Path $Root $j.script
  if (-not (Test-Path $script)) { continue }

  $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
               Where-Object { $_.CommandLine -like "*$($j.script.Replace('\','\\'))*" -or $_.CommandLine -like "*$(Split-Path $j.script -Leaf)*" })

  $stale = $true
  $flagPath = Join-Path $Root $j.flag
  if (Test-Path $flagPath) {
    try {
      $f = Get-Content $flagPath -Raw | ConvertFrom-Json
      $age = (New-TimeSpan -Start ([datetime]::Parse($f.asof).ToUniversalTime()) -End ([datetime]::UtcNow)).TotalMinutes
      $stale = ($age -gt $StaleMin) -or (-not $f.ok)
      if ($stale) { Say "$($j.name): flag stale/not-ok (age $([math]::Round($age,1))m, ok=$($f.ok))" }
    } catch { Say "$($j.name): flag unreadable: $_" }
  } else { Say "$($j.name): no flag yet" }

  if ($running.Count -gt 0 -and -not $stale) { continue }

  if ($running.Count -gt 0 -and $stale) {
    Say "$($j.name): running but stale -> killing $($running.Count) pid(s)"
    $running | ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }
    Start-Sleep -Seconds 2
  }

  $out = Join-Path $Root $j.log
  Say "$($j.name): starting"
  Start-Process -FilePath $Py -ArgumentList $script -WorkingDirectory $Root `
                -RedirectStandardOutput $out -RedirectStandardError "$out.err" `
                -WindowStyle Hidden
}
