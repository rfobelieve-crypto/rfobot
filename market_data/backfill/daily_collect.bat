@echo off
REM Daily data collection - Coinglass + research parquets + depth deltas + backup
REM Scheduled 04:00 daily. Logs to research\results\daily_collect.log
REM 2026-08-08 rewrite:
REM  - the 08-01 edit embedded a literal BACKSPACE (0x08) in the step-1.5 path
REM    ("research\backfill" became "research<BS>ackfill"), so that step never ran once
REM  - step 2 (inject_coinglass) read the pre-v7 15m ml_data store (deleted) and
REM    could never succeed - removed
REM  - coinglass_backfill read .env with cp950 and died on a UTF-8 dash since 07-19
REM    (fixed in the .py); all of this hid behind other jobs' ensure_fresh
REM  - failures now LOG and ACCUMULATE; task exits nonzero if ANY step failed
REM    (green-task-over-dead-job lesson: 2026-07-05, 2026-08-01)

cd /d "C:\Users\rfo\Desktop\flowbot\flow_system"
set LOG=research\results\daily_collect.log
set FAILED=0

echo [%date% %time%] ===== daily_collect start ===== >> %LOG%

for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
    if "%%a"=="COINGLASS_API_KEY" set COINGLASS_API_KEY=%%b
)
if "%COINGLASS_API_KEY%"=="" (
    echo [%date% %time%] ERROR: COINGLASS_API_KEY not found in .env >> %LOG%
    exit /b 1
)

echo [%date% %time%] step1 coinglass_backfill >> %LOG%
python -m market_data.backfill.coinglass_backfill --interval 1h >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step2 research backfill_all_parquet >> %LOG%
python research\backfill_all_parquet.py >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step1.6 refresh_fng (weakness-#2 revival) >> %LOG%
python research\refresh_fng.py >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step3 export_depth_deltas >> %LOG%
python -m market_data.backfill.export_depth_deltas >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step4 backup data_collector --light >> %LOG%
python -m market_data.backfill.data_collector --light >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step5 research guards (regression tests) >> %LOG%
python research/run_guards.py >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step6 scheduled task refs >> %LOG%
python research/ops/check_schedules.py >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step7 research data manifest >> %LOG%
python research/ops/data_manifest.py >> %LOG% 2>&1
if errorlevel 1 set FAILED=1

echo [%date% %time%] step8 product live fills >> %LOG%
python research/ops/check_product_fills.py >> %LOG% 2>&1

echo [%date% %time%] ===== done FAILED=%FAILED% ===== >> %LOG%
exit /b %FAILED%
