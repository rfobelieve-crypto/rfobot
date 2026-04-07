# BTC Market Intelligence Indicator — System Architecture v7

> **Last Updated:** 2026-04-04  
> **Author:** rfo  
> **Status:** Production (Railway)

---

## 1. System Overview

Production ML system generating **4-hour directional predictions** for BTC/USDT Perpetual.

- **Input:** 13 API endpoints (Binance + Coinglass + Deribit)
- **Processing:** 130+ engineered features → Dual XGBoost models
- **Output:** Direction / Magnitude / Confidence → 4-panel chart → Telegram

```
Binance (Klines + Depth + aggTrades)
Coinglass (15 timeseries + 9 snapshot)     →  Feature Engineering (130+ cols)
Deribit (DVOL + Options Summary)                      ↓
                                           Dual Model v7
                                           ├── Direction Classifier (89 feat)
                                           └── Magnitude Regressor (78 feat)
                                                      ↓
                                           Signal Generation
                                           (direction, confidence, strength, BBP)
                                                      ↓
                                           4-Panel Chart → Telegram + REST API
```

### Core Output (per 1h bar)

| Field | Type | Description |
|-------|------|-------------|
| `pred_direction` | UP / DOWN / NEUTRAL | 方向預測 |
| `mag_pred` | float ≥ 0 | 預測幅度 \|return_4h\| |
| `pred_return_4h` | float | sign(direction) × magnitude |
| `confidence_score` | 0~100 | 幅度百分位排名 |
| `strength_score` | Strong / Moderate / Weak | 信心等級 |
| `bull_bear_power` | [-1, 1] | 多空力量合成指標 |
| `regime` | CHOPPY / TRENDING_BULL / TRENDING_BEAR | 市場狀態 |

---

## 2. Data Sources

### 2.1 Binance Futures REST API (3 endpoints)

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `/fapi/v1/klines` | 1h OHLCV + taker volume | 每小時, 500 bars |
| `/fapi/v1/depth` | Order book snapshot (L20) | 每小時, 最新 bar |
| `/fapi/v1/aggTrades` | 逐筆交易 (大小單分離) | 每小時, 2h 回看 |

### 2.2 Coinglass API v4 (24 endpoints)

**15 Timeseries Endpoints (原生 1h, 500 bars)**

| Category | Endpoints | Features |
|----------|-----------|----------|
| Open Interest | OI per-exchange, OI aggregated, OI coin-margin | 13 features (delta, accel, share, pctchg) |
| Liquidation | Liq per-exchange, Liq aggregated | 7+ features (surge, cascade, imbalance) |
| Long/Short Ratio | Top trader L/S, Global L/S, Position ratio | 7 features |
| Funding Rate | Funding rate OHLC | 2 features (close, range) |
| Taker Volume | Taker buy/sell | 4 features (delta, ratio) |
| CVD | Futures CVD agg, Spot CVD agg | 4+ features |
| Premium/Margin | Coinbase premium, Bitfinex margin | 4+ features |

**9 Snapshot Endpoints (point-in-time)**

| Endpoint | Data |
|----------|------|
| Options max pain | Max pain price, P/C ratio |
| Options OI | Total OI |
| Options/Futures ratio | Ratio |
| ETF flow | Net flow per ticker (IBIT, FBTC...) |
| Fear & Greed | 0-100 index |
| ETF AUM | Total AUM |
| Futures netflow | Multi-TF (5m~24h) |
| Spot netflow | Multi-TF (5m~24h) |
| HL whale positions | Count, net USD, long% |

### 2.3 Deribit Public API (2 endpoints, 免費)

| Endpoint | Purpose | Features |
|----------|---------|----------|
| Volatility Index | DVOL (BTC 波動率指數) | bvol_open/high/low/close, intra_range, change_1h |
| Book Summary | Options P/C volume, OI, IV | pc_volume_ratio, iv_skew, OTM put/call IV |

### 2.4 Data Quality

- **重試策略:** 指數退避, 最多 3 次 (2s → 4s → 8s)
- **快取回退:** `.data_cache/` parquet 檔案
- **新鮮度監控:** > 3h 未更新觸發告警

---

## 3. Feature Engineering

### 3.1 Feature Groups (130+ total)

| Group | Count | Source | Key Features |
|-------|-------|--------|-------------|
| Kline-derived | ~33 | Binance klines | log_return, realized_vol, taker_delta, return_lag_1~10 |
| Coinglass | ~50+ | CG 15 endpoints | OI delta/accel, liq surge/cascade, L/S ratio, funding |
| Z-scores | ~14 | Computed | All major features zscore over 24h window |
| Cross-features | ~8 | Computed | liq_x_oi, crowding, conviction, cvd_divergence |
| Direction features | ~40 | research/ | imbalance, absorption, momentum, sentiment, OBI |
| BVOL (Deribit) | ~13 | Deribit DVOL | bvol OHLC, intra_range, change, rolling stubs |
| aggTrades | ~11 | Binance aggTrades | agg_large_delta, large_ratio, imbalance_div |
| Volume dynamics | 4 | Computed | vol_acceleration, kurtosis, entropy, squeeze_proxy |
| Time cyclical | 4 | Computed | hour_sin/cos, weekday_sin/cos |
| Order book | 4 | Binance depth | depth_imbalance, spread_bps, bid_ask_ratio |

### 3.2 Direction Feature Groups (research/direction_features.py)

| Group | Features | Description |
|-------|----------|-------------|
| Large Trade Flow | 8~12 | 大單/小單分離, delta zscore, persistence |
| Imbalance Persistence | 8 | imb_1b~8b, 符號持續性, slope |
| Absorption Proxy | 8 | buy/sell absorption, price impact asymmetry |
| Momentum | 8 | ret_1b~5b, wick/body ratio, reversal setup |
| Sentiment | 5~7 | funding extreme, crowding, OI-price confirm |
| Order Book | 4~6 | OBI L1/L5, change, persistence |

### 3.3 Pipeline Flow

```
fetch_binance_klines(500)  ──┐
fetch_coinglass(15 ep.)  ────┤
fetch_binance_depth()  ──────┤──→  build_live_features()  ──→  130+ columns DataFrame
fetch_binance_aggtrades() ───┤       │
fetch_deribit_dvol()  ───────┤       ├── Kline-derived features
options_summary  ────────────┘       ├── merge_asof(Coinglass)
                                     ├── Slope/momentum/divergence
                                     ├── Volume dynamics
                                     ├── Absorption & liquidation enhanced
                                     ├── Depth/aggTrades snapshot (last bar)
                                     ├── BVOL snapshot (last bar)
                                     ├── Time cyclicals
                                     └── Direction feature set (optional)
```

---

## 4. Model Architecture (v7 Dual-Model)

### 4.1 Architecture

```
                    Feature DataFrame (130+ cols)
                           │
              ┌────────────┴────────────┐
              ↓                         ↓
    ┌──────────────────┐      ┌──────────────────┐
    │  Direction Model │      │  Magnitude Model │
    │  XGBClassifier   │      │  XGBRegressor    │
    │  89 features     │      │  78 features     │
    │  Output: P(UP)   │      │  Output: |ret|   │
    └────────┬─────────┘      └────────┬─────────┘
             ↓                         ↓
      P(UP) > 0.60 → UP        mag_pred ≥ 0
      P(UP) < 0.40 → DOWN
      else → NEUTRAL
              ↓                         ↓
              └────────────┬────────────┘
                           ↓
              pred_return_4h = sign(dir) × mag
```

### 4.2 Signal Generation

| Step | Logic |
|------|-------|
| Direction | P(UP) > 0.60 → UP, < 0.40 → DOWN, else NEUTRAL |
| Magnitude | \|return_4h\| prediction, clipped ≥ 0 |
| Confidence | expanding percentile of \|mag\| × direction conviction boost |
| Strength | ≥80 Strong, ≥65 Moderate, else Weak |
| BBP Gate | direction 需與 BBP 同向, 否則降級為 NEUTRAL |
| Regime | CHOPPY: suppress noise (×1.6 deadzone), TRENDING: trust (×0.9) |
| Hysteresis | flip direction 需 1.4x 更強反向信號 |

### 4.3 Fallback Chain

```
1. Dual Model (v7)  ←── 主要 (direction_xgb + magnitude_xgb)
2. Regime Models (v5/v6)  ←── per-target, per-regime XGBRegressor
3. Legacy Model (v2)  ←── 單一 XGBRegressor
```

### 4.4 Regime Detection

| Regime | Condition |
|--------|-----------|
| WARMUP | 前 168 bars (7 days) |
| CHOPPY | vol_pct ≤ 0.6 or \|24h_return\| < 0.005 |
| TRENDING_BULL | vol_pct > 0.6 and 24h_return > 0.005 |
| TRENDING_BEAR | vol_pct > 0.6 and 24h_return < -0.005 |

---

## 5. Chart Output (4-Panel)

| Panel | Height | Content |
|-------|--------|---------|
| 1. Confidence Heatmap | 0.8 | 紫色漸層, 0~100% confidence |
| 2. Candlestick | 8.0 | K 線 + 方向三角形 (Moderate/Strong only) |
| 3. Magnitude | 2.0 | 預測幅度柱狀圖 (UP↑ DOWN↓ NEUTRAL gray) |
| 4. Bull/Bear Power | 2.0 | BBP 柱狀圖 [-1, 1] |

- 顯示最近 200 bars
- 時區: UTC+8
- 格式: PNG (matplotlib Agg)

---

## 6. Delivery & API

### 6.1 REST Endpoints

| Route | Method | Response |
|-------|--------|----------|
| `/` | GET | Latest chart PNG |
| `/health` | GET | System health JSON |
| `/json` | GET | Latest prediction JSON |
| `/force-update` | POST | Trigger immediate update |
| `/indicator-status` | GET | Detailed status + CG health |
| `/db-diag` | GET | MySQL diagnostics |
| `/indicator-perf` | GET | Rolling accuracy metrics |

### 6.2 Telegram

- **Photo:** 4-panel chart + caption (direction, confidence, strength, regime)
- **Inline Keyboard:** 8 quick-action buttons (Chart, Perf, DB, Flow, Status, Events, Sweep, Help)
- **Alerts:** Data freshness warnings, endpoint failure

### 6.3 Persistence

| Storage | Purpose | Format |
|---------|---------|--------|
| MySQL `indicator_history` | 每根 bar 預測結果 | 14+ columns, UPSERT |
| Parquet | 完整歷史 (重啟恢復) | indicator_history.parquet |
| `.data_cache/` | API 回退快取 | per-source parquet |

---

## 7. Infrastructure

| Component | Technology |
|-----------|-----------|
| Runtime | Python 3.11 |
| ML | XGBoost (Classifier + Regressor) |
| Data | Pandas, NumPy, SciPy |
| Web | Flask |
| Database | MySQL 8.0 (Railway) |
| Deploy | Railway (auto-deploy on push to main) |
| Scheduling | APScheduler (1h interval) |
| Chart | Matplotlib (Agg backend) |
| Delivery | Telegram Bot API |

### Threading Model

```
Flask Main Thread (port from $PORT)
    ├── HTTP request handlers
    └── APScheduler daemon thread
        └── update_cycle() every 1h
            ├── Fetch (Binance + CG + Deribit)
            ├── Build features (130+)
            ├── Predict (dual model)
            ├── Backfill mag_pred (if needed)
            ├── Render chart PNG
            ├── Save (parquet + MySQL)
            └── Send Telegram
```

---

## 8. File Structure

```
indicator/
├── app.py                    # Flask app + Telegram + update orchestration
├── auto_update.py            # Standalone scheduler (Windows/manual)
├── data_fetcher.py           # 13 fetch_* functions, retry/cache
├── feature_builder_live.py   # 130+ feature engineering
├── inference.py              # IndicatorEngine v7 (dual-model)
├── chart_renderer.py         # 4-panel chart rendering
├── snapshot_collector.py     # Depth/aggTrades/options → MySQL
└── model_artifacts/
    ├── dual_model/           # v7: direction_xgb + magnitude_xgb
    ├── direction_models/     # v5: standalone direction classifier
    ├── regime_models/        # v5: per-target per-regime models
    └── indicator_history.parquet

research/
├── direction_features.py     # 6 groups, 40+ direction features
├── dual_model/               # Training scripts + evaluation
└── results/                  # Experiment results + metrics
```

---

## 9. Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Spearman IC | > 0.05 | 預測值與實際收益排序相關 |
| ICIR | > 0.3 | IC / std(IC), 穩定性 |
| 方向準確率 | > 58% | direction vs actual |
| Calibration | Monotonic | 預測越強 → 實際收益越高 |
| Confidence 分布 | 均勻 | 不應全集中在 Weak |
