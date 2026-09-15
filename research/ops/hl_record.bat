@echo off
REM FlowBot_HLRecord -- since 2026-09-15 this PULLS, it no longer records.
REM
REM The recorder (research/hl/hl_fuel_recorder.py) now runs on Railway (service
REM hl-record, research/hl/hl_record_service.py). Reason: locally it shared the
REM machine IP with the HMM engine's HL leg; HL rate-limits per IP, and all 274
REM HL-side 429s in the MON engine log fell inside this task's :05-:33 window.
REM
REM This task now runs at :45, after the cloud run (:05 UTC minute, ~28 min).
REM 1. hl_record_pull.py pulls new hourly files into research/hl/data and keeps
REM    the remote mtime, so the freshness row for hl_fuel_last.json still turns
REM    red when the CLOUD recorder stops (not only when the pull stops).
REM 2. hl_verify.py stays local: it reads the local WS trade tape on D:, and it
REM    makes one HL request per run.
REM
REM ASCII ONLY, CRLF (cmd.exe skips lines on UTF-8 in LF files; mistake.md 2026-09-13).
REM Proof it ran = new lines in research\results\hl_record.log, never LastTaskResult.

cd /d "C:\Users\rfo\Desktop\flowbot\flow_system"
set LOG=research\results\hl_record.log
set PYTHONIOENCODING=utf-8

echo [%date% %time%] ===== hl_record pull start ===== >> %LOG%
python research/hl/hl_record_pull.py >> %LOG% 2>&1
echo [%date% %time%] verify units >> %LOG%
python research/hl/hl_verify.py >> %LOG% 2>&1

echo [%date% %time%] ===== hl_record pull done rc=%errorlevel% ===== >> %LOG%
exit /b 0
