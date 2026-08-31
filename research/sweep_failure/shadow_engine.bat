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
REM 2026-08-24 (TODO 0.57): publish ARMED levels, not just filled signals.
REM Batch-publishing fills costs 0.1328 R/trade (158% of variant B's edge)
REM -- the consumer must learn a level is armed BEFORE the retest.
python research\raid_pending_publish.py >> research\results\sweep_shadow_run.log 2>&1
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
REM 2026-08-21: cloud-train parity check (weakness-#1 migration) -- compares
REM the local ledger hash vs the cloud recorder's train_parity row. Log-only;
REM 7 consecutive MATCH days unlock the cutover (rule frozen in the script).
python research\train_parity_check.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-26: publish the pre-registration board (open hypotheses +
REM progress + settled verdicts). Progress only -- every verdict keeps
REM its single owning scorer, so nothing here can drift from the number
REM that actually decides. See TODO 0.61.
python research\prereg_publish.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-26 (TODO 0.62): publish CLOSED outcomes. The live feed only
REM carries OPEN signals, so a follower could log 'the slot cap blocked
REM this one' but never find out what it would have done -- and scoring
REM their real fills against research numbers is the asymmetry that made
REM blocked signals look 4.3x better. Same ruler on both arms.
python research\raid_outcomes_publish.py >> research\results\sweep_shadow_run.log 2>&1
REM 2026-08-31: pull jarvis V7Bot executions for the chart overlay
REM (user: charts must mark real entries/exits; v7_okx_positions froze
REM 08-11 at the Bitget migration). Same export token as the mill
REM pipeline (TODO 0.78); env unset = printed skip, never a bat failure.
python research\v7_product_trades_publish.py >> research\results\sweep_shadow_run.log 2>&1
