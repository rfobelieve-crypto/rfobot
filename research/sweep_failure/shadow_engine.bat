@echo off
cd /d C:\Users\rfo\Desktop\flowbot\flow_system
python research\sweep_failure\shadow_engine.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-17: weather-station snapshot rides the same hourly cadence (the
REM engine just refreshed the kline caches this step depends on). Failure
REM here must never block the frozen shadow accounting above.
python research\weather_station_publish.py >> research\results\sweep_shadow_run.log 2>&1
