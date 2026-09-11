@echo off
REM Hyperliquid 鏈上錄製（2026-09-11 起，每小時）。
REM 錄四樣沒有歷史端點的東西：market(OI/funding) / book(L2) /
REM fuel(清算價直方圖) / orders(掛單+觸發單)。可事後補的 candle /
REM fundingHistory / userFills 不在這裡。
REM
REM 判斷它活著的依據是**產物**：research/results/hl_fuel_last.json 的
REM asof 與 ok，不是這支的 exit code（mistake.md 2026-08-26）。
REM 體積：約 0.7MB/輪 -> 17MB/天 -> 0.5GB/月。

cd /d "C:\Users\rfo\Desktop\flowbot\flow_system"
set LOG=research\results\hl_record.log
set PYTHONIOENCODING=utf-8

echo [%date% %time%] ===== hl_record start ===== >> %LOG%
python research/hl/hl_fuel_recorder.py --max-addr 900 >> %LOG% 2>&1
echo [%date% %time%] verify units >> %LOG%
python research/hl/hl_verify.py >> %LOG% 2>&1

echo [%date% %time%] ===== hl_record done rc=%errorlevel% ===== >> %LOG%
exit /b 0
