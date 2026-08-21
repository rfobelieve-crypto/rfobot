# DB Registry — 45 表目錄（自動生成 + 人工註記）

> 2026-08-21 生成（數據工程弱點 #3）。重生成：跑
> `python research/gen_db_registry.py`（SHOW TABLES + git grep 分類
> writer/reader，全自動）。**新表上線必須同步登記**，
> agent 可讀的表另需在 `.claude/rules/agent-boundary.md` 登記——兩處都要。

| 表 | 列數 | 最新資料 | writers | readers |
|---|---|---|---|---|
| `agent_runs` | 49 | 2026-04-20 10:30 | base.py | _summary.py<br>agents.py<br>health.py |
| `agent_user_accounts` | 1 | 2026-07-24 01:18 | queries.py | server.py? |
| `agent_verdicts` | 0 | — | queries.py | — |
| `agent_waitlist_signups` | 2 | 2026-07-27 01:33 | queries.py | — |
| `cancel_eyeball_log` | 27 | 2026-07-26 23:58 | BTC_perp_data.py<br>tv_alert_poller.py | cancel_playbook_watcher.py?<br>queries.py? |
| `cancel_playbook_events` | 2449 | 2026-08-21 13:38 | cancel_playbook_watcher.py | BTC_perp_data.py<br>cancel_flow_analyze.py<br>cancel_flow_interactive.py<br>cancel_path_analysis.py<br>+5 |
| `depth_deltas_1m` | 867632 | 2026-08-21 15:13 | depth_delta_collector.py<br>export_depth_deltas.py | app.py<br>cancel_flip_events.py<br>cancel_flow_analyze.py<br>cancel_flow_interactive.py<br>+19 |
| `depth_events_1s` | 1439609 | 2026-08-21 15:13 | depth_events_1s.py | start_all.py<br>test_depth_events_1s.py |
| `event_feature_snapshots` | 0 | — | BTC_perp_data.py<br>extra_schema.py<br>snapshot_repository.py | oi_schema.py<br>snapshot_query.py<br>snapshot_runner.py?<br>start_all.py? |
| `event_features` | 0 | — | event_builder.py | edge_query.py<br>run_pipeline.py? |
| `event_registry` | 45 | 2026-08-21 08:55 | BTC_perp_data.py<br>snapshot_repository.py | aligner.py<br>feature_assembler.py?<br>oi_schema.py?<br>snapshot_builder.py<br>+2 |
| `feature_bars_15m` | 0 | — | feature_builder.py | event_builder.py<br>run_pipeline.py? |
| `flow_bars_15m` | 0 | — | flow_bars_15m_builder.py | feature_builder.py<br>run_all.py |
| `flow_bars_1m` | 1035435 | — | cleanup.py<br>flow_repository.py<br>import_aggtrades.py | app.py<br>cancel_flow_analyze.py<br>cancel_flow_interactive.py?<br>cancel_lead_ic_tercile.py<br>+21 |
| `funding_rates` | 821250 | — | download_raw.py<br>extra_schema.py<br>funding_backfill.py<br>funding_collector.py | check_pipeline.py<br>feature_assembler.py<br>feature_builder.py<br>feature_builder_v2.py<br>+2 |
| `hybrid_signals` | 0 | — | — | app.py?<br>chart_hybrid.py<br>hybrid_full_simulation.py?<br>hybrid_inference.py<br>+2 |
| `indicator_aggtrades_snapshots` | 3897 | 2026-08-21 15:05 | — | app.py<br>snapshot_collector.py? |
| `indicator_depth_snapshots` | 3904 | 2026-08-21 15:05 | — | app.py<br>liquidity_proxy_features.py<br>snapshot_collector.py?<br>train_with_liq_features.py |
| `indicator_history` | 3289 | 2026-08-21 14:00 | app.py<br>calibration_check.py<br>diagnose_ic.py<br>infra.py<br>+5 | alpha_decay_monitor.py<br>analytics.py<br>chart_interactive.py?<br>chart_renderer.py?<br>+27 |
| `indicator_options_snapshots` | 3909 | 2026-08-21 15:05 | — | app.py?<br>options_positioning_ic.py<br>snapshot_collector.py? |
| `indicator_sentiment_snapshots` | 3902 | 2026-08-21 15:05 | — | app.py?<br>snapshot_collector.py |
| `ldc_swing_positions` | 0 | — | — | chart_hybrid.py<br>hybrid_monitor.py<br>ldc_swing_executor.py<br>paper_trading.py |
| `liquidation_1m` | 57287 | — | extra_schema.py<br>liquidation_collector.py | feature_assembler.py<br>liquidity_proxy_features.py<br>minute_ic_scan.py<br>orderbook_liq_features.py<br>+2 |
| `liquidity_events` | 9 | 2026-08-19 19:26 | BTC_perp_data.py | app.py?<br>cancel_flow_interactive.py<br>cleanup.py?<br>event_builder.py<br>+3 |
| `normalized_trades` | 0 | — | cleanup.py<br>trade_repository.py | app.py?<br>chart_builder.py<br>feature_assembler.py<br>snapshot_builder.py |
| `ohlcv_1m` | 653760 | — | import_klines.py | check_pipeline.py<br>feature_builder_v2.py<br>fetch_1m_for_intrabar.py?<br>minute_ic_scan.py<br>+1 |
| `oi_snapshots` | 508581 | 2026-08-21 15:13 | download_raw.py<br>oi_backfill.py<br>oi_collector.py<br>oi_schema.py | check_pipeline.py?<br>cleanup.py?<br>feature_assembler.py<br>feature_builder.py<br>+3 |
| `orderbook_snapshots_1m` | 589880 | 2026-08-21 15:13 | orderbook_l20_collector.py<br>queries.py | cancel_flip_events.py<br>cancel_flow_analyze.py<br>cancel_flow_interactive.py<br>cancel_lead_ic.py<br>+16 |
| `pf_intents` | 42 | 2026-08-19 01:00 | ledger.py<br>pf_dry_intents.py | — |
| `pf_positions` | 21 | 2026-08-11 10:00 | ledger.py<br>pf_mirror.py | — |
| `raid_signals_live` | 86 | 2026-08-21 15:12 | raid_signals_publish.py | freshness_board.py?<br>server.py |
| `strong_signals` | 0 | — | — | cancel_flow_interactive.py<br>signal_explainer.py<br>signal_tracker.py<br>validate_direction_reg.py? |
| `sweep_outcomes` | 44 | 2026-08-21 08:55 | outcome_tracker.py | BTC_perp_data.py<br>app.py?<br>cleanup.py? |
| `tracked_signals` | 2212 | 2026-08-21 12:00 | — | agents.py<br>alpha_decay_monitor.py<br>analytics.py<br>app.py<br>+40 |
| `tv_alert_events` | 52 | 2026-08-21 08:55 | BTC_perp_data.py<br>tv_alert_poller.py | cancel_flow_interactive.py<br>start_all.py? |
| `v7_okx_approvals` | 7 | 2026-06-02 13:56 | — | — |
| `v7_okx_balance_snapshots` | 1515398 | 2026-08-21 15:14 | — | cleanup.py?<br>freshness_board.py?<br>performance.py<br>pf_dry_intents.py<br>+2 |
| `v7_okx_executor_status` | 1 | 2026-08-21 15:14 | app.py | report.py<br>stability.py<br>state.py |
| `v7_okx_kill_log` | 405 | 2026-08-21 15:14 | app.py | pf_dry_intents.py<br>report.py<br>stability.py<br>state.py? |
| `v7_okx_positions` | 21 | 2026-08-11 10:00 | app.py<br>pf_mirror.py | config.py?<br>empirical_kelly_from_trades.py<br>executor.py?<br>pf_dry_intents.py<br>+8 |
| `v7_okx_positions_archive` | 5 | 2026-06-04 10:00 | — | — |
| `v7_okx_reconciliation_log` | 2381 | 2026-08-21 15:14 | — | state.py |
| `v7_paper_positions_archive` | 17 | 2026-06-04 13:00 | — | empirical_kelly_from_trades.py |
| `v7_veto_clock` | 1 | 2026-08-21 15:01 | v7_veto_publish.py | app.py?<br>freshness_board.py?<br>server.py<br>v7_raid_veto.py? |
| `weather_station` | 1 | 2026-08-21 15:12 | weather_station_publish.py | app.py?<br>cloud_train.py?<br>freshness_board.py?<br>raid_signals_publish.py<br>+2 |

## 人工註記（生成器覆蓋不到的語意）

- **無程式碼引用的表（掃描發現）**：`v7_okx_approvals`（2026-05 首批人工
  確認流程的遺物，approval 已改in-process）、`v7_okx_positions_archive`
  （2026-06-05 paper cohort 移除時的刻意留底）。**兩者皆存檔性質，保留
  不清——但從此有名有姓，不再是「這表還有人用嗎」的懸案**
- **多寫入者高風險表**：`weather_station`/`raid_signals_live`/
  `v7_veto_clock`/`train_parity` 都是單列 UPSERT 快照——班車 cutover
  （§0.56 authority 段）前雲端不得寫前三者，見 cloud_train.py PHASE 規則
- **表的權威寫入方向**：quant 系統寫、agent 只讀（agent-boundary.md
  機器執法）；jarvis 完全不碰此 DB（獨立系統，走 /public HTTP）
