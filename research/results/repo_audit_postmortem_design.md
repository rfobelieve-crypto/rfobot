# Repo Audit — 每週自動化 Post-Mortem Agent Pipeline 設計依據

日期：2026-04-19

---

## Section 1: 數據 Schema

### 1.1 `indicator_history` 表

| 欄位 | 類型 | 說明 |
|------|------|------|
| `dt` | DATETIME PK | 1h bar 時間 |
| `open/high/low/close` | DOUBLE | K 線 |
| `pred_return_4h` | DOUBLE | 預測 TWAP 回報 |
| `pred_direction_code` | DOUBLE | UP=1, DOWN=-1, NEUTRAL=0 |
| `confidence_score` | DOUBLE | 0~100 |
| `strength_code` | DOUBLE | Strong=3, Moderate=2, Weak=1 |
| `bull_bear_power` | DOUBLE | [-1, 1] |
| `regime_code` | DOUBLE | BULL=2, BEAR=-2, CHOPPY=0, WARMUP=-99 |
| `dir_prob_up` | DOUBLE | P(UP) 原始值 |
| `mag_pred` | DOUBLE | |return_4h| |

- **377 行**，範圍 2026-04-03 ~ 2026-04-19（v7 部署後才開始累積）
- Schema 定義在 `indicator/snapshot_collector.py:323-367`
- 每小時寫入最後一行（`save_indicator_history()`）
- 分類欄位存為 numeric code，需映射才能讀

### 1.2 `tracked_signals` 表

| 欄位 | 類型 | 說明 |
|------|------|------|
| `id` | INT AUTO PK | |
| `signal_time` | DATETIME | 信號時間 |
| `direction` | VARCHAR(10) | UP/DOWN |
| `strength` | VARCHAR(10) | 目前只記 Strong |
| `p_up/mag_pred/confidence` | DOUBLE | 模型輸出 |
| `entry_price` | DOUBLE | 進場價 |
| `regime` | VARCHAR(30) | 市場狀態 |
| `exit_price` | DOUBLE | +4h 收盤價 |
| `actual_return_4h` | DOUBLE | TWAP 回報 |
| `correct` | TINYINT | 1=方向正確 |
| `filled` | TINYINT | 1=已回填結果 |
| `shap_top` | TEXT | SHAP JSON |

- **1,822 行**，範圍 2025-11-16 ~ 2026-04-19（含從 `strong_signals` 遷移的歷史數據）
- `record_signal()` 目前 `if strength != "Strong": return`，只記 Strong
- UNIQUE KEY `(signal_time, strength)`
- 結果回填使用 **TWAP**：`mean(close[t+1..t+4]) / entry - 1`

---

## Section 2: 績效分析能力盤點

### 已存在且可直接使用的模組

| 模組 | 指標 | 即時? | 輸出 |
|------|------|-------|------|
| `app.py` `/indicator-perf` | OOS baseline + live Strong WR + regime + SHAP | Yes | Telegram HTML |
| `signal_tracker.py` `get_performance_report()` | Strong 勝率、方向拆解、近 3 天明細 | Yes | Telegram HTML |
| `monitor_icir.py` `run_monitor()` | 方向準確率、tier 拆解、IC、flip rate | Yes | 告警+CSV |
| `dashboard.py` | 24h 分佈、regime 時間線、IC 趨勢、健康檢查 | Yes | HTML 頁面 |
| `strong_signal_perf.py` | Wilson CI、regime 拆解、model-version guard | 手動 | JSON |
| `feature_importance_tracker.py` | 重要性 snapshot/diff/list | 手動 | JSON+CSV |

### `monitor_icir.py` 告警閾值

| 指標 | Critical | Warning |
|------|----------|---------|
| 方向準確率 | < 35% | < 42% |
| Strong 準確率 | — | < 50% |
| IC (100+ bars) | < -0.05 | < 0.0 |
| Flip rate | — | > 40% |
| Neutral rate | — | > 70% |

### 缺口
- `strong_signal_perf.py` 和 `feature_importance_tracker.py` 是手動腳本，未排程
- 沒有自動的 per-regime 績效追蹤（研究腳本存在但未整合到生產端）

---

## Section 3: 特徵 / 模型 Metadata

### 生產模型概覽

| | Direction-Reg | Magnitude |
|---|---|---|
| 類型 | XGBRegressor | XGBRegressor |
| Objective | reg:squarederror | reg:squarederror |
| Target | TWAP path return | |return_4h| |
| 特徵數 | **136** | **72** |
| 訓練樣本 | 3,995 | 3,991 |
| 訓練時間 | 2026-04-15 | 2026-04-16 |
| 數據範圍 | 2025-11 ~ 2026-04-16 | 同上 |
| OOS IC | 0.183 (walk-forward) | 0.340 |

### Direction Top-10 特徵（重要性）

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | cg_oi_close | 0.01609 |
| 2 | hour_cos | 0.01510 |
| 3 | is_trending_bear | 0.01463 |
| 4 | cg_gls_ratio | 0.01390 |
| 5 | cg_oi_agg_close | 0.01353 |
| 6 | cg_oi_close_zscore | 0.01348 |
| 7 | cg_oi_close_pctchg_8h | 0.01323 |
| 8 | vol_kurt_non_bear | 0.01305 |
| 9 | cg_pos_account_divergence | 0.01286 |
| 10 | vol_kurtosis | 0.01283 |

### Magnitude Top-10 特徵（gain）

| Rank | Feature | Gain |
|------|---------|------|
| 1 | hour_cos | 124.98 |
| 2 | cg_oi_close_pctchg_24h | 87.68 |
| 3 | cg_ls_ratio_zscore | 74.39 |
| 4 | weekday_cos | 70.46 |
| 5 | taker_delta_ma_24h | 58.36 |
| 6 | vol_acceleration | 55.07 |
| 7 | cg_oi_close_pctchg_12h | 54.27 |
| 8 | cg_oi_close | 53.94 |
| 9 | taker_delta_std_24h | 53.85 |
| 10 | realized_vol_20b | 53.80 |

### 發現的不一致

| 項目 | CLAUDE.md 記載 | 實際 |
|------|---------------|------|
| Direction 模型類型 | XGBClassifier | **XGBRegressor** |
| Direction 特徵數 | 89 | **136** |
| Magnitude 特徵數 | 87 | **72** |
| `direction_importance.csv` | 應對應 136 特徵 | **只有 98 個（舊模型的）** |
| `is_trending_bull` | 應有貢獻 | **importance = 0.0** |

---

## Section 4: 排程器（Scheduler）

### APScheduler 設定（`app.py:1339-1353`）

| Job | 觸發 | 用途 |
|-----|------|------|
| `update_cycle` | cron `:02` 每小時 | 抓數據 → 預測 → 圖表 → 推送 |
| `_run_watchdog_quick` | cron `:15` 每小時 | 免費規則檢查（10 項閾值） |
| `_run_watchdog_full` | cron 每 4h `:20` | 深度掃描，有異常才呼叫 Claude |

### 加入每週 Post-Mortem 的方式

在 `start_scheduler()` 內（line 1347 後）加一行：

```python
scheduler.add_job(weekly_postmortem, "cron", day_of_week="sun", hour="2", minute="0",
                  misfire_grace_time=3600)
```

`scheduler` 是函數內局部變數，新 job 必須在 `scheduler.start()` 之前加。

---

## Section 5: 通知 / 推送

### 現有通道

| 函數 | 用途 | parse_mode |
|------|------|-----------|
| `_send_telegram_text()` | 文字訊息 | HTML |
| `_send_telegram_photo()` | 圖表 PNG + caption | caption 無 parse_mode |
| `_send_telegram_photo_to()` | 指定 chat_id 發圖 | — |
| `_send_discord_photo()` | Discord webhook | — |

### Agent 通道

- `AGENT_BOT_TOKEN` / `AGENT_CHAT_ID`（獨立 token，fallback 到 `INDICATOR_*`）
- `BaseAgent.send_alert()` 用 agent 專用 token

### 缺口

- **無訊息長度處理**：Telegram 上限 4096 字元，超長訊息會靜默失敗
- **photo caption 無 parse_mode**：HTML 標籤會原文顯示
- **無分段邏輯**：如果 post-mortem 報告很長，需要自己實作 split

---

## Section 6: LLM 整合

### SDK & 設定

- `anthropic` SDK 在 `requirements.indicator.txt`
- 預設模型：`claude-sonnet-4-20250514`
- `AGENT_API_KEY` 環境變數，無 fallback
- `MAX_TOKENS=8192`，`MAX_TURNS=15`

### Agent 架構（`indicator/agents/`）

| 檔案 | 角色 |
|------|------|
| `base.py` | 抽象基底：agentic tool-use loop、Claude API、Telegram 告警 |
| `coordinator.py` | 編排所有 agent、`--context-only` 模式、報告格式化 |
| `watchdog.py` | 成本優化：免費規則先檢查，有異常才呼叫 Claude |
| `data_collector.py` | 數據新鮮度、延遲、錯誤率 |
| `infra.py` | MySQL 健康、排程狀態、磁碟 |
| `feature_guard.py` | 特徵 NaN 率、分佈、追蹤 NaN 來源 |
| `model_eval.py` | 預測歷史、IC、regime/tier 準確率 |
| `signal_tracker.py` | 信號勝率、連勝/連敗、偏差 |
| `meeting.py` | 跨域分析：接收所有 agent 結果、交叉關聯、可調閾值 |
| `repair_actions.py` | 風險分級修復：AUTO 自動執行、SUGGEST 僅推送建議 |
| `config_store.py` | JSON runtime config 覆蓋層 |

### 缺口

- **無 API retry**：單次 `anthropic` API 錯誤直接 abort 整個 agent run
- **無 rate-limit handling**
- **Scheduler 非全域變數**：無法在 runtime 動態新增/移除 job

---

## Section 7: Open Questions（設計決策點）

### 7.1 Post-Mortem 覆蓋範圍

現有 agent 已覆蓋 data/infra/feature/model/signal 五個域。每週 post-mortem 應該：
- (a) **複用現有 agent**（coordinator 已有 `run_all` 邏輯），還是
- (b) **獨立寫一個** weekly agent 做更深的分析（如 walk-forward IC 趨勢、regime 轉換率）？

### 7.2 報告格式

- Telegram 4096 字元上限 → post-mortem 報告可能需要 **多段推送** 或 **改用 photo 截圖**
- 或者只推摘要 + dashboard link？

### 7.3 CLAUDE.md 過期

Direction model 從 Classifier 換成 Regressor、特徵數從 89 變 136、Magnitude 從 87 變 72 — CLAUDE.md 全部過期。**建議 post-mortem pipeline 的第一步就是自動檢查 config vs CLAUDE.md 的一致性**。

### 7.4 `direction_importance.csv` 過期

目前的 importance CSV 是 98 特徵（舊 classifier），不是 136 特徵（當前 regressor）。`feature_importance_tracker.py` 的 snapshot/diff 功能依賴這個 CSV，如果不更新，diff 結果沒意義。

### 7.5 Signal 樣本量

`CURRENT_MODEL_DEPLOY = "2026-04-17"`，至今只有 ~2 天的 live signals。任何統計結論都極不穩定。Post-mortem 需要設定「最小樣本數」閾值（建議 >= 50 筆 filled signals 才出結論）。

### 7.6 LLM 成本控制

現有 watchdog 已用「免費 gatekeeper → 有異常才呼叫 Claude」的模式。每週 post-mortem 如果完整跑一次所有 domain agent + meeting agent，大概是 ~5 次 Claude 呼叫 x 8192 tokens = ~40k tokens/週。需要決定這個成本是否可接受。

### 7.7 歷史比較基準

377 行 `indicator_history` 不足以做有意義的「本週 vs 上週」比較（第一週只有完整的一週數據）。需要等累積到 >= 2 週才能啟用 week-over-week 差異分析。
