# System Architecture — BTC 多空強度預測指標 v7

> Last updated: 2026-04-13

## 系統定位

多空強度預測指標（Market Intelligence Indicator），預測未來 4h BTC 方向、強度、信心。**不是**交易策略或自動下單系統。

## 核心架構：Dual-Model v7

```
Binance REST ─┐
Coinglass v4 ─┼─→ feature_builder_live.py ─→ 130+ features ─→ IndicatorEngine.predict()
Deribit API  ─┘         (12 groups)              │                    │
                                                  │         ┌─────────┴─────────┐
                                                  │    Direction XGB        Magnitude XGB
                                                  │    (29 features)        (76 features)
                                                  │    P(UP) → dir          |return_4h|
                                                  │         └─────────┬─────────┘
                                                  │                   ▼
                                                  │        Confidence = dir_score^0.6 × 80
                                                  │                   + mag_bonus × 20
                                                  │                   ▼
                                                  │    Strong ≥ 80 / Moderate ≥ 65 / Weak
                                                  │                   ▼
                                                  └───→ Chart (4 panels) + Telegram + REST API
```

## 數據層

### Binance REST API (3 endpoints)
- `fapi/v1/klines` — 1h K 線，limit=500
- `fapi/v1/depth` — L20 訂單簿快照
- `fapi/v1/aggTrades` — 聚合成交（大單分析）

### Coinglass API v4 (14 timeseries + snapshots)
- **OI**: oi, oi_agg, oi_coin_margin
- **Funding**: funding_rate
- **Sentiment**: long_short, global_ls, top_ls_position, coinbase_premium, bitfinex_margin
- **Flow**: taker, futures_cvd_agg, spot_cvd_agg
- **Other**: liquidation, liq_agg
- Snapshots: options, etf_flow, fear_greed, etf_aum, futures_netflow, spot_netflow, whale_positions

### Deribit Public API (2 endpoints)
- DVOL 波動率指數
- Options Summary (P/C ratio, IV skew)

## 特徵工程

**130+ 工程特徵，12 個群組**（`feature_builder_live.py`）

| 群組 | 前綴 | 來源 | 範例 |
|------|------|------|------|
| binance_klines | — | Binance | log_return, realized_vol_20b, volume_zscore_20 |
| coinglass_oi | cg_oi_ | Coinglass | cg_oi_delta, cg_oi_close_zscore |
| coinglass_funding | cg_funding_ | Coinglass | cg_funding_close, cg_funding_range |
| coinglass_sentiment | cg_ls_, cg_gls_, cg_cb_ | Coinglass | cg_ls_ratio, cg_cb_premium |
| coinglass_flow | cg_fcvd_, cg_scvd_, cg_taker_ | Coinglass | cg_taker_delta |
| deribit | bvol_ | Deribit | bvol_close, bvol_change_1h |
| depth | depth_ | Binance | depth_imbalance, spread_bps |
| aggtrades | agg_ | Binance | agg_large_delta, agg_large_buy_ratio |
| engineered_alpha | — | 衍生 | impact_asymmetry, post_absorb_breakout |
| direction | — | 衍生 | hour_sin/cos, vol_regime, squeeze_proxy |
| toxicity | tox_ | 衍生 | tox_adverse_ratio, tox_informed_intensity |
| bitfinex | cg_bfx_ | Coinglass | cg_bfx_margin_ratio |

### Direction Model 使用的 29 剪枝特徵（2026-04-12 IC 篩選）
- 25 signal features（|IC| > 0.03, p < 0.05）
- 4 structural features（hour_sin, hour_cos, vol_regime, squeeze_proxy）
- Top 5 by importance: cg_oi_agg_close, cg_oi_close_zscore, vol_acceleration, cg_bfx_margin_ratio, quote_vol_zscore

## 模型

### Direction Model（XGBClassifier）
- 29 剪枝特徵，輸出 P(UP)
- 訓練：walk-forward 77 folds, expanding window
- XGB params: max_depth=4, lr=0.05, n_estimators=300, early_stopping=30
- OOS AUC: 0.5971, last-10-fold AUC: 0.6022

### Magnitude Model（XGBRegressor）
- 76 特徵，輸出 |return_4h| (σ-adjusted)
- σ→return 轉換：mag = mag_sigma × realized_vol_20b

### Regime Detection（trailing-only）
- `vol_24h.expanding(min_periods=72).rank(pct=True)` 計算波動率百分位
- TRENDING_BULL: vol_pct > 0.6 且 ret_24h > 0.5%
- TRENDING_BEAR: vol_pct > 0.6 且 ret_24h < -0.5%
- CHOPPY: 其他
- WARMUP: 前 72 bars（數據不足，佔位符）

## 信號生成

### Direction
- P(UP) > 0.58 → UP
- P(UP) < 0.42 → DOWN
- 其餘 → NEUTRAL

### Confidence（direction-first 公式，2026-04-12 更新）
```
dir_dist = |dir_prob - 0.5|          # 0~0.5
dir_score = (dir_dist / 0.5)^0.6 × 80   # 0~80 pts（主導）
mag_bonus = (mag_percentile / 100) × 20  # 0~20 pts（輔助）
confidence = clip(dir_score + mag_bonus, 0, 100)
```

### Strength Tiers
- Strong ≥ 80, Moderate ≥ 65, Weak < 65

### Gates
- BBP 確認閘門 + Regime 動態死區 + Hysteresis + Cooldown

## 輸出

### 圖表（兩套同步）
1. **靜態圖** (`chart_renderer.py`) — Telegram PNG，4 面板：Confidence / K線+三角形 / Magnitude / BBP
2. **互動圖** (`chart_interactive.py`) — `/ichart` TradingView Lightweight Charts HTML

### Telegram
- 每小時推送圖表 + 預測摘要
- Strong 信號額外文字告警 + SHAP 驅動因子

### REST API（Flask）
- `GET /` — 最新圖表 PNG
- `GET /health` — JSON 健康狀態
- `GET /json` — 最新預測 JSON
- `GET /indicator-status` — 文字報告
- `GET /indicator-perf` — 績效報告（IC, 準確率, regime 拆解）
- `GET /signal-perf` — 信號追蹤績效
- `GET /dashboard` — HTML 診斷面板
- `GET /ichart` — 互動圖表
- `POST /force-update` — 手動觸發更新
- `POST /meeting` — 全面 agent 掃描

### 持久化
- MySQL 8.0（Railway）— indicator_history, tracked_signals
- Parquet — 歷史備份 + raw data
- `.data_cache/` — API fallback 快取

## 監控架構（Watchdog + Agent Swarm）

### 排程
| Job | 時間 | 說明 |
|-----|------|------|
| `update_cycle` | 每小時 :02 | 取數據 → 預測 → 圖表 → Telegram |
| `watchdog quick` | 每小時 :15 | Gatekeeper 規則檢查（$0），異常才啟動 Claude Agent |
| `watchdog full` | 每 4h :20 | Gatekeeper + ModelEval + SignalTracker |

### Gatekeeper 檢查（10 項，純規則，$0）
| 檢查 | 門檻 | 觸發 Agent |
|------|------|-----------|
| data（Binance 新鮮度）| > 120min | DataCollector |
| db（MySQL 延遲/連線）| ping > 500ms 或 conn > 20 | Infra |
| scheduler（排程間隔）| > 130min | Infra |
| features（NaN 率）| > 30% | FeatureGuard |
| model（7d IC）| < -0.05 | ModelEval |
| signals（Strong 勝率）| < 40% | SignalTracker |
| confidence（分數分布）| std < 3 或 Strong > 40% | ModelEval |
| regime（狀態切換）| 卡住 > 48h 或 WARMUP 殘留 | ModelEval |
| feature_drift（特徵漂移）| > 20% 常數 或 |z| > 5 | FeatureGuard |
| dual_model（模型載入）| mode ≠ dual 或 model=None | Infra |

### Agent Swarm（5 個 Agent）
| Agent | 職責 | 自動修復 |
|-------|------|---------|
| DataCollector | 數據源狀態、快取新鮮度 | clear_stale_cache, trigger_update |
| FeatureGuard | NaN 分群、分布、sanity | clear_stale_cache, trigger_update |
| ModelEval | IC 趨勢、方向準確率、regime 拆解 | suggest only |
| SignalTracker | 勝率、streaks、backfill | suggest only |
| Infra | DB、排程、磁碟、環境變數 | kill_idle_connections, trigger_update |

### IC 衰退監控（`monitor_icir.py`）
- 方向準確率、IC decay、flip rate、neutral rate
- 12h cooldown per alert type

## 研究管線

### Parquet 回填（`research/backfill_all_parquet.py`）
- Binance klines + 14 CG endpoints → `market_data/raw_data/*.parquet`
- 自動偵測時間戳格式（秒 vs 毫秒）
- `is_stale()` 檢查**所有** parquet 文件新鮮度
- `ensure_fresh()` 訓練前自動回填

### 訓練管線
- `research/dual_model/shared_data.py` — 載入 parquet → build_live_features
- `research/dual_model/train_direction_model_4h.py` — walk-forward CV
- `research/dual_model/export_production_models.py` — 全量訓練 → 匯出 model_artifacts

## 技術 Stack
- Python 3.11（Railway），本地 Python 3.9
- Pandas + NumPy + SciPy + XGBoost + SHAP
- MySQL 8.0（Railway 託管）
- Flask + APScheduler
- Matplotlib（靜態圖）+ TradingView Lightweight Charts（互動圖）
- Telegram Bot API
- 部署：Railway（git push main 自動部署）

## 部署
- Service 1（main bot）：`Dockerfile`
- Service 2（market data）：`Dockerfile.marketdata`
- Railway env vars 優先於 `.env`
