@echo off
cd /d C:\Users\rfo\Desktop\flowbot\flow_system
python research\sweep_failure\shadow_engine.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-17: weather-station snapshot rides the same hourly cadence (the
REM engine just refreshed the kline caches this step depends on). Failure
REM here must never block the frozen shadow accounting above.
python research\weather_station_publish.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-20: publish live raid signals to MySQL for the follow-bot
REM endpoint. The agent used to read the CSV baked into its image, which
REM is only as fresh as the last git push (it was 8 days stale).
python research\raid_signals_publish.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-18 M1: unified ledger mirror (v7_okx_positions -> pf_positions,
REM idempotent upsert, zero live-code change). See TODO §0.5.
python research\pf_mirror.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-18 M3: dry-run intent flow (fresh variant-B fills -> risk engine
REM -> pf_intents with decisions). No orders; decisions are the deliverable.
python research\pf_dry_intents.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-20: v7 veto-clock publish -- the cloud route cannot compute it
REM (needs the local kline cache, not in the image); the site card sat at a
REM build-time snapshot (asof 08-10, trigger 4/60 vs truth 34/60). Same
REM off-cloud-recorder fix family as raid_signals_publish above.
python research\v7_veto_publish.py >> research\results\sweep_shadow_run.log 2>&1
