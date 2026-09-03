# arb_watchdog.ps1 — relaunch any member of the §0.75 recording family that
# is not running. Runs every 5 min from the EntropyArbWatchdog scheduled task.
#
# Why (2026-09-03): the whole family died at 21:09 +0800 from one interrupt
# (runner.log ends in ^C) and stayed dead 46 minutes. The 30-second restart
# loop inside each .bat only heals "the python child died"; when the console
# that hosts the loop is taken out, the loop dies with it and nothing
# restarts anything. This is the layer above that loop.
#
# Rules:
#   * presence is judged by PROCESS COMMAND LINE, never by file mtime — a
#     stalled-but-alive process is the recorder's own problem to log, and
#     launching a second copy on top of a live one is the duplicate-scanner
#     bug of 2026-09-03 (two scanners hitting the same public API = the rate
#     limiting we spent a day on). One member, one process, or relaunch.
#   * each member is launched exactly the way the Startup-folder launcher
#     does it (its own minimized console via the same .bat), so the heal
#     path and the boot path are the same path.
#   * every decision is one line in the log, including "all alive" — a
#     watchdog that only writes when it acts looks dead when things are fine
#     (the degradation-guard rule, freshness_board.py registry note).
param([switch]$DryRun)

$Root = 'C:\Users\rfo\Desktop\flowbot\entropy-arb'
$Log  = 'C:\Users\rfo\Desktop\flowbot\flow_system\research\results\arb_watchdog.log'

# member name -> (command-line signature, launcher .bat)
$Members = [ordered]@{
  'SNDK'    = @('--symbol SNDK ', 'run_recorder.bat')
  'NBIS'    = @('--symbol NBIS ', 'run_recorder_NBIS.bat')
  'ANTH'    = @('--symbol ANTH ', 'run_recorder_ANTH.bat')
  'BTC'     = @('--symbol BTC ',  'run_recorder_BTC.bat')
  'ZEC'     = @('--symbol ZEC ',  'run_recorder_ZEC.bat')
  'NEAR'    = @('--symbol NEAR ', 'run_recorder_NEAR.bat')
  'HYPE'    = @('--symbol HYPE ', 'run_recorder_HYPE.bat')
  'scanner' = @('tools\scanner.py', 'run_scanner.bat')
}

$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
         ForEach-Object { [string]$_.CommandLine }
$stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')
$dead = @()
foreach ($name in $Members.Keys) {
  $sig, $bat = $Members[$name]
  $alive = @($procs | Where-Object { $_ -like "*main.py --record-only*$sig*" -or ($name -eq 'scanner' -and $_ -like "*$sig*") }).Count
  if ($alive -ge 1) { continue }
  $dead += $name
  if (-not $DryRun) {
    Start-Process -FilePath (Join-Path $Root $bat) -WorkingDirectory $Root -WindowStyle Minimized
  }
}
$line = if ($dead.Count -eq 0) { "$stamp UTC  all 8 alive" }
        elseif ($DryRun)      { "$stamp UTC  DRY-RUN would relaunch: $($dead -join ',')" }
        else                  { "$stamp UTC  RELAUNCHED: $($dead -join ',')" }
Add-Content -Path $Log -Value $line -Encoding UTF8
Write-Output $line
