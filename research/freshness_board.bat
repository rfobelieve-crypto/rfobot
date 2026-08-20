@echo off
cd /d C:\Users\rfo\Desktop\flowbot\flow_system
REM 2026-08-20: unified artifact-freshness board. Runs on its OWN
REM schedule (every 6h) -- it must never ride the hourly train it
REM monitors. Alerts on transitions only; judge by products not panels.
python research\freshness_board.py >> research\results\freshness_board.log 2>&1
