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
# **加進這張表的錄製器必須在啟動時就寫一次 ok=True 的旗標**（liq_recorder:232
# 的 flag(True,"starting") 就是這個慣例）。沒照做的話，這支會在它來得及
# 落盤之前讀到上一輪的舊旗標、判定 stale，然後殺掉一個健康的新行程 ——
# 無限重啟迴圈。hl_tape 2026-09-11 就這樣被殺過一次（落盤週期 300 秒 vs
# 本支 5 分鐘），修法寫在 hl_tape.write_flag 的 docstring 裡。
$Jobs = @(
  @{ name = 'liq';     script = 'research\exit_paths\liq_recorder.py';     flag = 'research\results\liq_last.json';     log = 'research\exit_paths\logs\liq_recorder.log' },
  @{ name = 'lighter'; script = 'research\exit_paths\lighter_recorder.py'; flag = 'research\results\lighter_last.json'; log = 'research\exit_paths\logs\lighter_recorder.log' },
  # 2026-09-11 加入：HL 全市場成交帶。它是常駐 WS，斷線自己會重連，
  # 但行程整個死掉就沒人管 —— 實際發生過（UTC 23:36 死、兩小時後才發現）。
  # 接在這裡而不是另寫一支看門狗：同一個判準（旗標的 asof，不是行程在不在）。
  @{ name = 'hl_tape';  script = 'research\hl\hl_tape.py';  flag = 'research\results\hl_tape_last.json'; log = 'research\hl\logs\hl_tape.log' },
  # 2026-09-11 加入：分鐘級中價與佇列。同樣是常駐 WS、同樣不可回填。
  @{ name = 'hl_mid';   script = 'research\hl\hl_mid.py';   flag = 'research\results\hl_mid_last.json'; log = 'research\hl\logs\hl_mid.log' }
  # **§1.25 的宇宙錄製器刻意不在這張表裡。** 它歸 `../arb/ops/arb_watchdog.ps1`
  # 管（那支 2026-09-11 就加了 'universe' 這一員）。2026-09-11 我一度把它加
  # 進來，因為 grep 這個 repo 的看門狗找不到它 —— 那正是 mistake.md 2026-09-04
  # 的錯：**枚舉的範圍是這台機器，不是這個 repo**。
  #
  # 而且兩支併存比缺一支更糟，因為**判準相反**：arb 那支只看行程在不在、
  # 從不殺；這一支看旗標新鮮度、會殺。同一個行程掛兩個判準不同的看門狗，
  # 其中一支會殺掉另一支剛拉起來的東西。一個錄製器只能有一個看門狗。
)

foreach ($j in $Jobs) {
  $script = Join-Path $Root $j.script
  # 路徑打錯的話，舊版在這裡**靜默** continue —— 而「跳過」跟「健康」在
  # log 上長得一模一樣（這支只在有事時才寫一行）。留一行痕跡，否則一個
  # 打錯的路徑會讓某個錄製器永遠沒有看門狗，而且沒有任何地方看得出來。
  if (-not (Test-Path $script)) { Say "$($j.name): script not found -> $script（跳過）"; continue }

  $running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
               Where-Object { $_.CommandLine -like "*$($j.script.Replace('\','\\'))*" -or $_.CommandLine -like "*$(Split-Path $j.script -Leaf)*" })

  $stale = $true
  $flagPath = Join-Path $Root $j.flag
  if (Test-Path $flagPath) {
    try {
      # **-Encoding UTF8 不是可選的。** 2026-09-11 查出來的：沒有它
      # Get-Content 用系統語系（這台是 cp950）讀檔，而旗標的 `reason` 是
      # Python 寫的 UTF-8 中文 -> 讀壞 -> ConvertFrom-Json 失敗 ->
      # 舊的 catch 什麼都不設、$stale 保持 $true -> **殺掉一個健康的行程**。
      # hl_mid 因此從 13:09 到 18:54 被殺了 70 次（每 5 分鐘一次），
      # 整個下午幾乎沒錄到東西，而它錄的是不可回填的 WS 資料。
      # 同一台機器上 arb 引擎的 logging 早就踩過這個（FileHandler 的
      # encoding 註解），只是方向相反（寫 vs 讀）。
      $f = Get-Content $flagPath -Raw -Encoding UTF8 | ConvertFrom-Json
      $age = (New-TimeSpan -Start ([datetime]::Parse($f.asof).ToUniversalTime()) -End ([datetime]::UtcNow)).TotalMinutes
      $stale = ($age -gt $StaleMin) -or (-not $f.ok)
      if ($stale) { Say "$($j.name): flag stale/not-ok (age $([math]::Round($age,1))m, ok=$($f.ok))" }
    } catch {
      # **讀不懂旗標不是「行程該死」的證據，是「看門狗的儀器壞了」。**
      # 退回用檔案的 mtime 判斷：mtime 新 = 行程還在寫 = 活著，不要殺。
      # （mistake.md 2026-09-11：age_json_flag 用的是 mtime 不是 asof 欄位。）
      $mAge = (New-TimeSpan -Start (Get-Item $flagPath).LastWriteTimeUtc -End ([datetime]::UtcNow)).TotalMinutes
      $stale = ($mAge -gt $StaleMin)
      Say "$($j.name): flag unreadable ($_) -> 退回 mtime age $([math]::Round($mAge,1))m, stale=$stale"
    }
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
