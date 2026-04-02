# BTC Market Intelligence Indicator — System Architecture

> **Multi-Source Order Flow Prediction System for BTC Perpetual Futures**
>
> A production-grade quantitative indicator system that ingests real-time trade flows from multiple exchanges, combines on-chain derivatives metrics from Coinglass, and outputs 4-hour directional predictions with calibrated confidence scoring. Deployed as three independent microservices on Railway with persistent MySQL storage.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Sources & Ingestion Layer](#3-data-sources--ingestion-layer)
4. [Storage Layer](#4-storage-layer)
5. [Feature Engineering Pipeline](#5-feature-engineering-pipeline)
6. [Machine Learning Pipeline](#6-machine-learning-pipeline)
7. [Live Inference Engine](#7-live-inference-engine)
8. [Visualization & Delivery](#8-visualization--delivery)
9. [Research & Experimentation Framework](#9-research--experimentation-framework)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Data Flow Summary](#11-data-flow-summary)
12. [Model Performance & Validation](#12-model-performance--validation)
13. [Technical Stack](#13-technical-stack)
14. [Key Design Decisions](#14-key-design-decisions)

---

## 1. System Overview

### 1.1 Product Definition

This system is a **Market Intelligence Indicator** — it predicts the direction and magnitude of BTC Perpetual Futures price movement over a 4-hour horizon. It is explicitly **not** a trading strategy: there are no entry/exit rules, no position sizing, no TP/SL logic, and no execution layer.

### 1.2 Core Output

Every hour, the system produces:

| Field | Type | Description |
|-------|------|-------------|
| `pred_return_4h` | float | Predicted 4h forward return (e.g., +0.12%) |
| `pred_direction` | enum | `UP` / `DOWN` / `NEUTRAL` |
| `confidence_score` | 0–100 | Calibrated prediction confidence |
| `strength_score` | enum | `Strong` / `Moderate` / `Weak` |
| `bull_bear_power` | -1 to +1 | Multi-source sentiment composite |
| `regime` | string | Current market regime classification |

### 1.3 System Scale

- **Data volume**: ~166 days of 1h bars (3,985 bars), 71 base features
- **Model features**: 40 selected via burn-in importance ranking
- **Prediction horizon**: 4 hours (4 × 1h bars)
- **Update frequency**: Hourly (on the hour)
- **Data sources**: Binance aggTrades WebSocket, OKX WebSocket, Binance REST (klines), Coinglass API v4 (6 endpoints)

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                                │
│                                                                              │
│  ┌────────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ Binance Futures │  │  OKX Futures │  │ Binance REST  │  │ Coinglass v4 │  │
│  │ WebSocket       │  │  WebSocket   │  │ (Klines)      │  │ (6 endpoints)│  │
│  │ aggTrade stream │  │  trades      │  │ 1h OHLCV      │  │ OI/Liq/FR/LS│  │
│  └───────┬────────┘  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘  │
└──────────┼──────────────────┼──────────────────┼──────────────────┼──────────┘
           │                  │                  │                  │
           ▼                  ▼                  │                  │
┌──────────────────────────────────┐             │                  │
│  SERVICE 2: Market Data Service  │             │                  │
│  (Dockerfile.marketdata)         │             │                  │
│                                  │             │                  │
│  ┌────────────┐ ┌──────────────┐ │             │                  │
│  │ Binance    │ │ OKX Adapter  │ │             │                  │
│  │ Adapter    │ │              │ │             │                  │
│  └─────┬──────┘ └──────┬───────┘ │             │                  │
│        │               │         │             │                  │
│        ▼               ▼         │             │                  │
│  ┌──────────────────────────┐    │             │                  │
│  │   Trade Normalizer       │    │             │                  │
│  │   Unified schema:        │    │             │                  │
│  │   canonical_symbol,      │    │             │                  │
│  │   notional_usd,          │    │             │                  │
│  │   taker_side, ts_ms      │    │             │                  │
│  └───────────┬──────────────┘    │             │                  │
│              ▼                   │             │                  │
│  ┌──────────────────────────┐    │             │                  │
│  │   Flow Aggregator        │    │             │                  │
│  │   1-minute bar buckets:  │    │             │                  │
│  │   buy/sell/delta/volume/ │    │             │                  │
│  │   trade_count/CVD        │    │             │                  │
│  └───────────┬──────────────┘    │             │                  │
└──────────────┼───────────────────┘             │                  │
               │                                 │                  │
               ▼                                 │                  │
┌──────────────────────────────────────────────────────────────────────────────┐
│                        PERSISTENT STORAGE (MySQL on Railway)                 │
│                                                                              │
│  flow_bars_1m │ raw_trades │ oi_snapshots │ funding_rates │ sweep_outcomes  │
│  ohlcv_1m     │ features_* │ liquidation_1m│ event_registry│                │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       │
               ┌───────────────────────┼─────────────────────────┐
               │                       │                         │
               ▼                       ▼                         ▼
┌─────────────────────────┐ ┌──────────────────────┐ ┌─────────────────────────┐
│ SERVICE 3: Indicator    │ │ SERVICE 1: Main Bot  │ │ OFFLINE: Research       │
│ (Dockerfile.indicator)  │ │ (Dockerfile)         │ │ Pipeline                │
│                         │ │                      │ │                         │
│ ┌─────────────────────┐ │ │ Telegram Bot         │ │ run_pipeline_v2.py      │
│ │ Data Fetcher        │◄┼─┤ TradingView Webhook  │ │ inject_coinglass.py     │
│ │ Binance REST +      │ │ │ Outcome Tracker      │ │ feature_assembler.py    │
│ │ Coinglass API       │ │ │ Event Registry       │ │                         │
│ └────────┬────────────┘ │ └──────────────────────┘ │ ┌─────────────────────┐ │
│          ▼              │                          │ │ Walk-Forward CV     │ │
│ ┌─────────────────────┐ │                          │ │ Feature Selection   │ │
│ │ Feature Builder     │ │                          │ │ Regime Analysis     │ │
│ │ (71 features)       │ │                          │ │ Abstention Logic    │ │
│ └────────┬────────────┘ │                          │ └─────────────────────┘ │
│          ▼              │                          │           │             │
│ ┌─────────────────────┐ │                          │           ▼             │
│ │ IndicatorEngine     │ │                          │ ┌─────────────────────┐ │
│ │ XGBoost Inference   │ │                          │ │ Model Artifacts     │ │
│ │ Confidence Scoring  │ │                          │ │ xgb_model.json      │ │
│ └────────┬────────────┘ │                          │ │ feature_cols.json   │ │
│          ▼              │                          │ │ training_stats.json │ │
│ ┌─────────────────────┐ │                          │ └─────────────────────┘ │
│ │ Chart Renderer      │ │                          └─────────────────────────┘
│ │ 3-panel PNG         │ │
│ └────────┬────────────┘ │
│          ▼              │
│ ┌─────────────────────┐ │
│ │ Flask API           │ │
│ │ GET /  → PNG chart  │ │
│ │ GET /json → JSON    │ │
│ │ GET /health         │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

---

## 3. Data Sources & Ingestion Layer

### 3.1 Exchange Adapters

The system ingests tick-level trade data from two major derivatives exchanges via persistent WebSocket connections:

**Binance Futures Adapter** (`market_data/adapters/binance_trades.py`)
- **Protocol**: WebSocket (`wss://fstream.binance.com/ws`)
- **Stream**: aggTrade (aggregated trades — multiple fills at same price grouped)
- **Data**: price, quantity (base units), taker side, trade ID, exchange timestamp
- **Contract**: Linear perpetual (BTCUSDT), contract_size = 1 (base units)

**OKX Futures Adapter** (`market_data/adapters/okx_trades.py`)
- **Protocol**: WebSocket (`wss://ws.okx.com:8443/ws/v5/public`)
- **Stream**: Individual trades (non-aggregated)
- **Data**: price, quantity (contracts), side, trade ID, exchange timestamp
- **Contract**: Inverse-style notation, contract_size = 0.01 BTC (for BTC-USDT-SWAP)

### 3.2 Trade Normalization (`market_data/core/trade_normalizer.py`)

All raw trades pass through a unified normalization pipeline:

```
Raw Exchange Trade → normalize() → Unified Trade Schema
```

**Unified Trade Schema**:
```python
{
    "exchange": "binance",               # source exchange
    "raw_symbol": "BTCUSDT",            # exchange-native symbol
    "canonical_symbol": "BTC-USD",       # unified internal symbol
    "instrument_type": "perp",           # always "perp"
    "price": 84250.5,                    # execution price
    "size": 0.5,                         # quantity in native units
    "size_unit": "base",                 # "base" or "contract"
    "taker_side": "buy",                 # aggressor side
    "notional_usd": 42125.25,           # price × size × contract_size
    "trade_id": "123456789",
    "ts_exchange": 1711929600000,        # Unix ms
    "ts_received": 1711929600050,        # Unix ms
    "is_aggregated_trade": True          # Binance aggTrade = True
}
```

**Symbol Mapping** (`market_data/core/symbol_mapper.py`):
- `BTCUSDT` → `BTC-USD` (Binance)
- `BTC-USDT-SWAP` → `BTC-USD` (OKX)
- Canonical format: `{BASE}-USD` (uppercase, hyphen-separated)

### 3.3 Flow Aggregation (`market_data/core/flow_aggregator.py`)

Normalized trades are accumulated into 1-minute flow bars:

```python
class FlowAggregator:
    def add_trade(self, trade: dict)      # Thread-safe accumulation
    def flush(self, now_ms: int) -> list  # Emit completed bars
```

**Per-bar computation**:
| Field | Formula |
|-------|---------|
| `buy_notional_usd` | Σ notional where taker_side = "buy" |
| `sell_notional_usd` | Σ notional where taker_side = "sell" |
| `delta_usd` | buy - sell |
| `volume_usd` | buy + sell |
| `trade_count` | count of trades |
| `cvd_usd` | cvd(prev_bar) + delta(current_bar) |
| `source_count` | distinct exchanges contributing |

- **Bucket key**: (canonical_symbol, minute_start_ms)
- **Flush interval**: ~10 seconds, only completed bars (window_end ≤ now)
- **CVD**: Cumulative — carries state across bars for continuous tracking
- **Thread safety**: `threading.Lock()` protects shared accumulator state

### 3.4 Coinglass API Integration

Six endpoints from Coinglass v4 API provide derivatives market structure data:

| Endpoint | Data | Key Fields |
|----------|------|------------|
| `oi` | Per-exchange Open Interest | Binance OI, share of total |
| `oi_agg` | All-exchange Aggregated OI | Total OI, delta, acceleration |
| `liquidation` | Forced liquidation volumes | Long/short liquidation USD |
| `long_short` | Top trader L/S ratio | Long%, account ratio |
| `global_ls` | Global account L/S ratio | Global long%, divergence |
| `funding` | Funding rate OHLC | Close rate, range |
| `taker` | Taker buy/sell volume | Delta, buy/sell ratio |

- **Native resolution**: 1h (avoids forward-fill artifacts)
- **Historical backfill**: `market_data/backfill/coinglass_backfill.py` (parquet storage)
- **Live fetch**: `indicator/data_fetcher.py` → `fetch_coinglass(interval="1h", limit=500)`

---

## 4. Storage Layer

### 4.1 Database Architecture

```
shared/db.py
├── Connection pooling (queue-based, pool_size=5)
├── Resolution order: ENV vars → .env → config.json
├── Railway internal: mysql.railway.internal
└── Local dev: via .env (external Railway host)
```

**Core principle**: All MySQL access routes through `shared/db.py`. No module creates its own connection logic. Connections are always closed in `finally` blocks.

### 4.2 Table Schema

**Real-time data** (written by Service 2):

| Table | Key | Retention | Purpose |
|-------|-----|-----------|---------|
| `flow_bars_1m` | (canonical_symbol, minute_start_ms) | Permanent | Aggregated 1-min trade bars |
| `raw_trades` | (exchange, trade_id) | 3 days | Individual trades (for debugging) |
| `oi_snapshots` | (symbol, ts) | Permanent | Open interest history |
| `funding_rates` | (symbol, ts) | Permanent | Funding rate snapshots |
| `liquidation_1m` | (symbol, minute_start_ms) | Permanent | Liquidation events |

**Event tracking** (written by Service 1):

| Table | Purpose |
|-------|---------|
| `sweep_outcomes` | Liquidity sweep event tracking with multi-window outcomes |
| `event_registry` | Registered market events (BSL/SSL) |
| `liquidity_events` | Historical event outcomes (15m/1h/4h windows) |

**Research pipeline** (offline):

| Table | Purpose |
|-------|---------|
| `ohlcv_1m` | Imported kline data for feature assembly |
| `features_1h` | Pre-computed hourly features |
| `market_state_bars` | Aggregated market state snapshots |

### 4.3 Offline Storage

Research data is stored as Parquet files for fast offline access:

```
research/ml_data/
├── BTC_USD_1h_enhanced.parquet        # Training data (3985 bars × 71 features)
├── BTC_USD_indicator_4h_v2.parquet    # E1+E2 predictions
├── BTC_USD_indicator_4h_e3e4.parquet  # E3+E4 regime analysis
└── ...

market_data/raw_data/
├── BTC_USD_1h_oi.parquet              # Coinglass OI backfill
├── BTC_USD_1h_liquidation.parquet     # Coinglass liquidation
├── BTC_USD_1h_longshort.parquet       # Coinglass L/S ratio
├── BTC_USD_1h_funding.parquet         # Coinglass funding
├── BTC_USD_1h_taker.parquet           # Coinglass taker
└── ...
```

---

## 5. Feature Engineering Pipeline

### 5.1 Feature Categories

The system computes 71 base features from 4 data sources:

#### A. Price & Volume Features (from Binance 1h Klines)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `log_return` | ln(close / close[-1]) | Base return series |
| `realized_vol_20b` | std(log_return, 20) | 20-bar realized volatility |
| `return_skew` | skew(log_return, 24) | Return distribution asymmetry |
| `return_kurtosis` | kurtosis(log_return, 24) | Tail risk measure |
| `volume_ma_4h` | volume / MA(volume, 4) | Short-term volume anomaly |
| `volume_ma_24h` | volume / MA(volume, 24) | Daily volume anomaly |
| `return_lag_1..10` | log_return[-1..-10] | Autoregressive components |

#### B. Taker Flow Features (from Binance Klines + Coinglass)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `taker_delta_ratio` | (taker_buy_vol - taker_sell_vol) / total_vol | Per-bar buy/sell imbalance |
| `taker_delta_ma_4h` | MA(taker_delta_ratio, 4) | Short-term flow trend |
| `taker_delta_ma_24h` | MA(taker_delta_ratio, 24) | Daily flow trend |
| `taker_delta_std_4h` | std(taker_delta_ratio, 4) | Flow volatility |
| `taker_delta_std_24h` | std(taker_delta_ratio, 24) | Daily flow stability |

#### C. Coinglass Derivatives Features (6 data sources, 30+ features)

**Open Interest**:
- `cg_oi_close`, `cg_oi_delta`, `cg_oi_accel` — Level, change, acceleration
- `cg_oi_agg_delta`, `cg_oi_agg_delta_zscore` — Aggregated OI momentum
- `cg_oi_binance_share` — Binance's share of total OI

**Liquidation**:
- `cg_liq_long`, `cg_liq_short` — Long/short liquidation volumes
- `cg_liq_ratio`, `cg_liq_imbalance` — Liquidation directional skew

**Long/Short Ratios**:
- `cg_ls_ratio`, `cg_ls_long_pct` — Top trader positioning
- `cg_gls_ratio` — Global account ratio
- `cg_ls_divergence` — Top trader vs global divergence

**Funding Rate**:
- `cg_funding_close`, `cg_funding_range` — Current rate and intra-period range
- `cg_funding_close_zscore` — Standardized funding deviation

**Taker Volume**:
- `cg_taker_delta`, `cg_taker_ratio` — Net taker flow
- `cg_taker_delta_zscore` — Standardized taker momentum

#### D. Cross-Feature Derived Signals

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `oi_price_divergence` | oi_delta_zscore - price_return_zscore | OI diverging from price → position buildup |
| `funding_taker_align` | sign(funding) × sign(taker_delta) | Funding and taker agreeing = conviction |
| `cg_crowding` | abs(ls_ratio_z) + abs(funding_z) + abs(oi_z) | Extreme positioning → reversal risk |
| `cg_conviction` | taker_delta_z × oi_delta_z | Flow and OI moving together = genuine flow |

#### E. Temporal Encoding

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `hour_sin` | sin(2π × hour / 24) | Cyclical hour-of-day |
| `hour_cos` | cos(2π × hour / 24) | Cyclical hour-of-day |
| `weekday_sin` | sin(2π × weekday / 7) | Day-of-week seasonality |
| `weekday_cos` | cos(2π × weekday / 7) | Day-of-week seasonality |

### 5.2 Z-Score Normalization

All Coinglass features have corresponding `_zscore` variants:

```python
zscore = (value - rolling_mean_24h) / rolling_std_24h
```

- **Window**: 24 bars (24h for 1h data)
- **Purpose**: Normalize cross-feature magnitude for comparability
- **Clipping**: Z-scores clipped to [-3, 3] before use in Bull/Bear Power

### 5.3 Feature Selection

**Method**: Fixed feature set via burn-in window (E1 design)

```
Burn-in: first 1000 bars (~42 days)
1. Train lightweight XGBoost on burn-in data
2. Rank features by importance
3. Select top 40
4. Lock feature set — never re-select
```

**Top 15 selected features** (by burn-in importance):

| Rank | Feature | Importance | Source |
|------|---------|------------|--------|
| 1 | cg_ls_ratio | 0.055 | Coinglass L/S |
| 2 | hour_cos | 0.035 | Temporal |
| 3 | hour_sin | 0.035 | Temporal |
| 4 | weekday_sin | 0.034 | Temporal |
| 5 | taker_delta_ma_24h | 0.032 | Binance Taker |
| 6 | cg_oi_agg_delta_zscore | 0.031 | Coinglass OI |
| 7 | cg_oi_agg_delta | 0.029 | Coinglass OI |
| 8 | cg_gls_ratio_zscore | 0.028 | Coinglass Global L/S |
| 9 | cg_ls_divergence_zscore | 0.027 | Coinglass Divergence |
| 10 | funding_taker_align | 0.026 | Cross-feature |
| 11 | cg_funding_range | 0.025 | Coinglass Funding |
| 12 | cg_ls_long_pct | 0.024 | Coinglass L/S |
| 13 | cg_taker_delta_zscore | 0.024 | Coinglass Taker |
| 14 | taker_delta_std_24h | 0.024 | Binance Taker |
| 15 | cg_oi_binance_share | 0.024 | Coinglass OI |

---

## 6. Machine Learning Pipeline

### 6.1 Target Variable

```python
y_return_4h = close.shift(-4) / close - 1  # 4-hour forward return
```

- **Horizon**: 4 bars at 1h resolution = 4 hours
- **Type**: Continuous regression target (not classification)
- **Distribution**: mean = -0.038%, std = 1.07% (slightly negative drift)

### 6.2 Walk-Forward Cross-Validation

The system uses strict expanding-window walk-forward validation to prevent information leakage:

```
Time ──────────────────────────────────────────────────────►

│◄── Burn-in (1000 bars) ──►│◄──── OOS Window (2985 bars) ────►│
│     Feature selection      │                                   │
│     (locked after)         │                                   │
│                            │                                   │
│ [Train Fold 1]             │[Test 1]                           │
│ ████████████████████████████│▒▒▒▒▒▒▒                           │
│                            │                                   │
│ [Train Fold 2]                     │[Test 2]                   │
│ ████████████████████████████████████│▒▒▒▒▒▒▒                   │
│                            │                                   │
│ [Train Fold 3]                             │[Test 3]           │
│ ████████████████████████████████████████████│▒▒▒▒▒▒▒           │
│                            │                                   │
│ ... (expanding window, 5 folds total)                          │
```

**Key properties**:
- Training always starts from bar 0 (expanding window)
- No future data leakage: each fold only trains on data before the test period
- Feature selection uses ONLY burn-in data (first 1000 bars)
- OOS predictions are never used for any training decisions

### 6.3 Model Architecture

**XGBoost Regressor** (single model — ensemble with Ridge was tested and rejected due to negative IC):

```python
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 3,
    "reg_alpha": 0.05,        # L1 regularization
    "reg_lambda": 0.5,        # L2 regularization
    "early_stopping_rounds": 40
}
```

**Why XGBoost only**:
- Ridge regression showed negative IC on 1h data (tested and documented)
- LightGBM comparison planned for future (E8 in research blueprint)
- Deep learning (LSTM/TFT) deferred: insufficient data (166 days, need 1+ years)

### 6.4 Model Artifacts

Training exports to `indicator/model_artifacts/`:

| File | Content |
|------|---------|
| `xgb_model.json` | Trained XGBoost model (portable JSON format) |
| `feature_cols.json` | Ordered list of 40 selected feature names |
| `training_stats.json` | Warmup data: last 200 OOS predictions for rolling statistics |

---

## 7. Live Inference Engine

### 7.1 Prediction Flow

```
indicator/app.py (hourly cron at :00)
│
├─ 1. fetch_binance_klines(interval="1h", limit=500)
├─ 2. fetch_coinglass(interval="1h", limit=500)  [6 endpoints]
│
├─ 3. build_live_features(klines, cg_data)
│     └─ 71 features computed identically to research pipeline
│
├─ 4. IndicatorEngine.predict(features)
│     ├─ Feature alignment (forward-fill missing)
│     ├─ XGBoost.predict() → raw_return_4h
│     ├─ Direction: sign(pred) with deadzone (±0.06%)
│     ├─ Confidence: percentile of |pred| in training distribution
│     ├─ Strength: Strong (≥70) / Moderate (≥40) / Weak (<40)
│     └─ Bull/Bear Power: mean of 5 clipped Coinglass z-scores
│
├─ 5. render_chart(indicator_df, last_n=200)
│     └─ 3-panel PNG: confidence heatmap + candlestick + BBP
│
└─ 6. Update shared state (thread-locked)
      ├─ indicator_df (full history)
      ├─ chart_png (latest render)
      └─ last_prediction (JSON)
```

### 7.2 IndicatorEngine Class (`indicator/inference.py`)

```python
class IndicatorEngine:
    """Stateful prediction engine with rolling statistics."""

    def __init__(self):
        self.xgb_model = xgb.XGBRegressor()
        self.xgb_model.load_model("model_artifacts/xgb_model.json")
        self.feature_cols = json.load("model_artifacts/feature_cols.json")
        self.pred_history = deque(maxlen=500)  # warmup from training_stats

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # 1. Align features (forward-fill missing, not zero-fill)
        # 2. XGBoost prediction
        # 3. Direction from raw prediction + deadzone
        # 4. Confidence from training distribution percentile
        # 5. Bull/Bear Power from Coinglass z-scores
        ...
```

**State management**:
- `pred_history`: Deque of last 500 predictions for rolling statistics
- Warmup: Pre-loaded with 200 historical OOS predictions from training
- Thread-safe: Flask routes read from `threading.Lock()`-protected shared state

### 7.3 Direction Logic (v2 — no z-score)

```python
# v1 (deprecated): direction = sign(zscore(pred))  ← z-score destroys signal
# v2 (current):    direction = sign(pred) with |pred| > deadzone

DEADZONE = 0.0006  # 0.06%

direction = "UP"      if pred >  DEADZONE
          = "DOWN"    if pred < -DEADZONE
          = "NEUTRAL" otherwise
```

**Why z-score was removed** (validated in E2):
- Raw pred IC = +0.030, after z-score IC = +0.011 (halved)
- Z-score destroys persistent trend signals (normalizes sustained bearish to 0)
- Z-score creates mean-reversion bias in what should be a momentum signal
- Short rolling window (48 bars) makes normalization unstable

### 7.4 Confidence Scoring (v2)

```python
# v1 (deprecated): confidence = percentile in rolling z-score history
# v2 (current):    confidence = percentile of |pred| in expanding OOS distribution

confidence = percentile_rank(|pred|, all_previous_OOS_predictions) × 100
```

- **Reference**: Expanding OOS distribution (stable, grows over time)
- **Range**: 0–100
- **Interpretation**: "How opinionated is the model compared to its historical baseline?"

### 7.5 Bull/Bear Power

Composite sentiment score from 5 Coinglass derivatives metrics:

```python
components = [
    +cg_oi_delta_zscore / 3,        # OI increase → bullish
    -cg_funding_close_zscore / 3,    # High funding → contrarian bearish
    +cg_taker_delta_zscore / 3,      # Net buying → bullish
    -cg_ls_ratio_zscore / 3,         # Extreme longs → contrarian bearish
    +cg_ls_divergence_zscore / 3,    # Divergence → informed flow
]
bull_bear_power = mean(components).clip(-1, +1)
```

---

## 8. Visualization & Delivery

### 8.1 Chart Renderer (`indicator/chart_renderer.py`)

Three-panel candlestick chart with embedded prediction signals:

```
┌──────────────────────────────────────────────────────────┐
│ Confidence Heatmap (purple gradient, 0–100)              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│   Candlestick Chart (OHLC)                              │
│   + Direction triangles (▲ UP / ▼ DOWN)                  │
│   + Size/color by strength (Strong = larger, darker)     │
│   + Only shown when confidence ≥ 40                      │
│                                                          │
│   ┌─────────────────────────────────┐                   │
│   │ Latest Signal Info Box          │                   │
│   │ Direction: UP ▲                 │                   │
│   │ Pred Return: +0.12%            │                   │
│   │ Confidence: 78                  │                   │
│   │ Regime: TRENDING_BULL          │                   │
│   └─────────────────────────────────┘                   │
│                                                          │
├──────────────────────────────────────────────────────────┤
│ Bull/Bear Power Histogram (green/red bars, ±1 scale)    │
└──────────────────────────────────────────────────────────┘
```

- **Backend**: Matplotlib (Agg backend for headless PNG generation)
- **Output**: PNG bytes (in-memory), cached in shared state
- **Refresh**: Every hour at :00

### 8.2 API Endpoints (Flask)

| Route | Response | Use Case |
|-------|----------|----------|
| `GET /` | PNG image | Browser / Telegram / embed |
| `GET /json` | JSON prediction | Programmatic integration |
| `GET /health` | Status + last update | Monitoring / healthcheck |

### 8.3 Delivery Channels

- **Web dashboard**: Direct browser access to Flask `/` route
- **Telegram**: Hourly push of chart PNG via bot API
- **API**: JSON endpoint for external system integration

### 8.4 Scheduling

| Environment | Scheduler | Interval | Trigger |
|-------------|-----------|----------|---------|
| Railway (indicator) | APScheduler CronTrigger | 1h | minute="0" |
| Local Windows | Task Scheduler | 1h | "00:00" start, 1h repeat |
| Local fallback | auto_update.py loop | 3600s | Sleep-based |

---

## 9. Research & Experimentation Framework

### 9.1 Research Pipeline

```
research/pipeline/run_pipeline_v2.py
│
├─ Step 1: DB Migration (create tables)
├─ Step 2: Import Klines → ohlcv_1m
├─ Step 3: Import aggTrades → flow_bars_1m_ml
├─ Step 4: Feature Assembly (feature_assembler.py)
│   ├─ Query: flow_bars_1m, oi_snapshots, funding_rates, liquidation_1m
│   ├─ Compute: delta, volume, OI, CVD features
│   └─ Write: features_1h table
├─ Step 5: Coinglass Injection (inject_coinglass.py)
│   ├─ Load parquet backfills (6 data sources)
│   ├─ Compute z-scores, cross-features
│   └─ Merge into enhanced parquet
└─ Step 6: Export ML-ready Parquet
    └─ BTC_USD_1h_enhanced.parquet (3985 bars × 71 features)
```

### 9.2 Experiment Tracking

#### E1: Fixed Feature Selection (Completed)
- **Design**: Burn-in first 1000 bars, select top 40 features, lock permanently
- **Result**: ICIR improved from +0.16 to +0.80 (5× stability improvement)

#### E2: Remove Z-Score from Prediction (Completed)
- **Design**: Use raw pred for direction, training-percentile for confidence
- **Result**: IC improved from +0.022 to +0.030, weekly positive IC from 50% to 75%

#### E3: Regime-Sliced Analysis (Completed)
- **Design**: Trailing-only regime detection (expanding percentile, no lookahead)
- **Regime classification**:
  ```python
  ret_24h = close.pct_change(24)
  vol_24h = log_return.rolling(24).std()
  vol_pct = vol_24h.expanding(min_periods=168).rank(pct=True)

  TRENDING_BULL  = (vol_pct > 0.6) & (ret_24h > +0.5%)
  TRENDING_BEAR  = (vol_pct > 0.6) & (ret_24h < -0.5%)
  CHOPPY         = everything else
  ```
- **Result**: All 3 regimes show positive IC — no single drag regime identified
  - TRENDING_BULL: IC = +0.046, dir_acc = 54.0% (n=497)
  - TRENDING_BEAR: IC = +0.054, dir_acc = 48.5% (n=546)
  - CHOPPY: IC = +0.008, dir_acc = 52.1% (n=1942)

#### E4: Selective Abstention (Completed)
- **Design**: Multi-layer abstention to improve signal precision
- **Layer 1 (Regime gate)**: All regimes IC ≥ 0, no filterable regime
- **Layer 2 (Magnitude gate)**: Filter by |pred| percentile — the real winner
- **Results**:

| Strategy | Bars | Retention | IC | Dir Acc |
|----------|------|-----------|-----|---------|
| No filter | 2985 | 100% | +0.030 | 51.9% |
| \|pred\| ≥ 50th pct | 1496 | 50.1% | +0.073 (p=0.005) | 51.9% |
| \|pred\| ≥ 75th pct | 747 | 25.0% | +0.107 (p=0.004) | 54.4% |

**Key finding**: Prediction magnitude is a strong meta-signal — the model's confidence in its own prediction (via |pred| magnitude) is well-calibrated. Emitting signals only when the model is "opinionated" (top 25-50% by magnitude) more than triples IC from +0.030 to +0.107.

### 9.3 Model Audit Methodology

Four-check validation protocol:

1. **OOS Metric Verification**: Walk-forward IC, ICIR, direction accuracy with no future contamination
2. **Leakage Detection**: Feature selection lookahead audit (revealed original IC=+0.071 was inflated → true IC=+0.022)
3. **Stability Analysis**: Per-fold IC variance, weekly IC distribution, monthly breakdown
4. **Attribution**: Feature importance decomposition, per-regime contribution

### 9.4 Performance Progression

| Version | OOS IC | ICIR | Dir Acc | Weekly +IC | Status |
|---------|--------|------|---------|------------|--------|
| v1 (reported) | +0.071 | — | 53.5% | — | Inflated (lookahead) |
| v1 (audited) | +0.022 | +0.16 | 53.5% | 50% | True baseline |
| v2 (E1+E2) | +0.030 | +0.80 | 51.9% | 75% | Fixed features + raw pred |
| v2 + abstention (50th pct) | +0.073 | — | 51.9% | — | Magnitude filter |
| v2 + abstention (75th pct) | +0.107 | — | 54.4% | — | Aggressive filter |

---

## 10. Deployment Architecture

### 10.1 Three-Service Design

```
┌─────────────────────────────────────────────────────────┐
│                     Railway Platform                     │
│                                                         │
│  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │
│  │ Service 1    │  │ Service 2     │  │ Service 3    │  │
│  │ Main Bot     │  │ Market Data   │  │ Indicator    │  │
│  │              │  │               │  │              │  │
│  │ Dockerfile   │  │ Dockerfile.   │  │ Dockerfile.  │  │
│  │              │  │ marketdata    │  │ indicator    │  │
│  │              │  │               │  │              │  │
│  │ • Telegram   │  │ • WS streams  │  │ • Flask API  │  │
│  │ • Webhooks   │  │ • 1m agg      │  │ • XGBoost    │  │
│  │ • Events     │  │ • DB flush    │  │ • Chart PNG  │  │
│  │ • Outcomes   │  │ • Cleanup     │  │ • APScheduler│  │
│  └──────┬───────┘  └──────┬────────┘  └──────┬───────┘  │
│         │                 │                  │          │
│         └─────────────────┼──────────────────┘          │
│                           ▼                             │
│              ┌────────────────────────┐                  │
│              │   MySQL (Railway)      │                  │
│              │   mysql.railway.internal│                  │
│              └────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

### 10.2 Service Independence

- Services share the **MySQL database only** — no cross-service module imports at runtime
- **Exception**: Main bot imports `market_data.query.flow_context` for read-only DB queries
- Each service has its own Dockerfile and requirements file
- Push to `main` triggers auto-deploy on all services

### 10.3 Configuration Management

```
Priority order (highest → lowest):
1. Railway environment variables (production)
2. .env file (local development, gitignored)
3. config.json (legacy fallback, gitignored)
```

Secrets management: `.env` and `config.json` are in `.gitignore`. Only `.env.example` with placeholder values is committed.

---

## 11. Data Flow Summary

### 11.1 Real-Time Path (every hour)

```
Binance REST (1h klines) ──┐
                           ├─► build_live_features() ──► IndicatorEngine
Coinglass API (6 endpoints)┘                              .predict()
                                                              │
                                                              ▼
                                                    ┌─────────────────┐
                                                    │ pred_return_4h  │
                                                    │ pred_direction  │
                                                    │ confidence      │──► render_chart()
                                                    │ strength        │        │
                                                    │ bull_bear_power │        ▼
                                                    │ regime          │    PNG bytes
                                                    └─────────────────┘        │
                                                                               ▼
                                                                    Flask / Telegram
```

### 11.2 Research Path (offline)

```
Parquet backfills ──► inject_coinglass() ──► BTC_USD_1h_enhanced.parquet
                                                      │
                                                      ▼
                                            prediction_indicator_v2.py
                                                      │
                                        ┌─────────────┼─────────────┐
                                        ▼             ▼             ▼
                                   Feature      Walk-Forward     Regime
                                   Selection    Prediction     Analysis
                                   (burn-in)    (5-fold)      (E3+E4)
                                        │             │             │
                                        ▼             ▼             ▼
                                   40 features   OOS IC/ICIR   Abstention
                                   locked        report        thresholds
```

### 11.3 Training → Production Path

```
Research Pipeline                 Production Inference
─────────────────                 ────────────────────
BTC_USD_1h_enhanced.parquet       Binance REST + Coinglass API
        │                                   │
        ▼                                   ▼
  train_export.py                 build_live_features()
        │                           (same logic)
        ▼                                   │
  Walk-forward training                     ▼
        │                         IndicatorEngine.predict()
        ▼                           (loads artifacts)
  Model Artifacts ───────────────────────────┘
  ├─ xgb_model.json
  ├─ feature_cols.json
  └─ training_stats.json
```

**Critical invariant**: `build_live_features()` and the research pipeline's feature computation must produce identical features for the same input data. Any divergence introduces train/serve skew.

---

## 12. Model Performance & Validation

### 12.1 Current Metrics (v2 Baseline)

| Metric | Value | Context |
|--------|-------|---------|
| OOS IC (all bars) | +0.030 | Spearman rank correlation, pred vs actual 4h return |
| OOS IC (top 50% magnitude) | +0.073 (p=0.005) | After magnitude-based abstention |
| OOS IC (top 25% magnitude) | +0.107 (p=0.004) | Aggressive abstention |
| ICIR | +0.80 | IC / std(fold ICs) — signal stability |
| Direction Accuracy (all) | 51.9% | Active signals (non-NEUTRAL) |
| Direction Accuracy (75th pct) | 54.4% | After magnitude filter |
| Weekly Positive IC | 75% (15/20) | Consistency over time |
| Negative IC Folds | 2/5 | Fold 1 and Fold 3 |

### 12.2 Fold-by-Fold Breakdown

| Fold | Period | IC | Dir Acc | n |
|------|--------|-----|---------|---|
| 1 | 11/28 – 12/23 | -0.024 | 47.6% | 597 |
| 2 | 12/23 – 01/16 | +0.169 | 50.3% | 597 |
| 3 | 01/16 – 02/10 | -0.027 | 51.8% | 597 |
| 4 | 02/10 – 03/07 | +0.075 | 55.3% | 597 |
| 5 | 03/07 – 04/01 | +0.116 | 50.6% | 597 |

### 12.3 Regime Analysis

| Regime | % of Bars | IC | Dir Acc | Interpretation |
|--------|-----------|-----|---------|----------------|
| TRENDING_BULL | 15.8% | +0.046 | 54.0% | Moderate signal |
| TRENDING_BEAR | 21.2% | +0.054 | 48.5% | Moderate signal, direction inverted |
| CHOPPY | 58.8% | +0.008 | 52.1% | Near-zero signal (noise) |

### 12.4 Anti-Overfitting Measures

1. **No feature selection lookahead**: Features selected ONCE on burn-in, locked for all folds
2. **Expanding window training**: Each fold trains on all data up to test start
3. **Regularization**: L1 (α=0.05) + L2 (λ=0.5) + early stopping (40 rounds)
4. **Subsampling**: 80% row sampling, 60% column sampling per tree
5. **Feature exclusion list**: Tested features that degrade walk-forward ICIR are permanently excluded

---

## 13. Technical Stack

### 13.1 Core Technologies

| Category | Technology | Version | Purpose |
|----------|-----------|---------|---------|
| Language | Python | 3.11 | All services |
| ML Framework | XGBoost | latest | Gradient boosted tree regression |
| Data Processing | Pandas | latest | Feature engineering, data manipulation |
| Numerical | NumPy, SciPy | latest | Statistics, z-scores, Spearman IC |
| Visualization | Matplotlib | latest | Chart rendering (Agg backend) |
| Web Framework | Flask | latest | API server |
| WSGI Server | Gunicorn | latest | Production HTTP serving |
| Scheduler | APScheduler | latest | Hourly update triggers |
| Database | MySQL (pymysql) | Railway | Persistent storage |
| Data Format | Parquet | latest | Offline data storage |
| Containerization | Docker | latest | Railway deployment |

### 13.2 External APIs

| API | Provider | Purpose | Auth |
|-----|----------|---------|------|
| WebSocket Trade Streams | Binance, OKX | Real-time trade data | API key (Binance) |
| REST Klines | Binance | 1h OHLCV candlesticks | None |
| Derivatives Metrics | Coinglass v4 | OI, liquidation, funding, L/S, taker | Paid API key |
| Bot API | Telegram | Alert delivery, user commands | Bot token |

---

## 14. Key Design Decisions

### 14.1 Indicator, Not Strategy

The system is intentionally scoped as a **prediction indicator** — it predicts direction and confidence but does not generate trading signals. This separation:
- Avoids overfitting to backtest P&L (which introduces survivorship bias)
- Evaluates purely on prediction quality (IC, calibration) — metrics that generalize
- Allows users to incorporate the indicator into their own decision framework

### 14.2 Order Flow + Derivatives Data (No Traditional Technical Indicators)

The feature set deliberately excludes traditional technical indicators (MACD, EMA, RSI, Bollinger Bands). All features derive from:
- **Tick-level order flow**: CVD, delta ratio, taker imbalance
- **Derivatives microstructure**: OI, funding rate, liquidations, L/S ratios
- **Cross-source divergence**: Funding × taker alignment, OI × price divergence
- **Statistical moments**: Realized vol, return skew/kurtosis

**Rationale**: Technical indicators are price-derived and widely known. Order flow and derivatives data represent the actual positioning and flow of institutional participants — alpha that is harder to arbitrage.

### 14.3 1h Native Resolution

The system was migrated from 15-minute to 1-hour bars because:
- Coinglass API provides native 1h data (vs 50% forward-fill at 30m)
- Reduces NaN rate from ~50% to ~0.4%
- Better alignment between feature sources
- 4h horizon (4 bars ahead) remains appropriate at 1h resolution

### 14.4 Burn-In Feature Selection

Fixed feature set via burn-in window (first 1000 bars = 42 days) rather than per-fold selection:
- **Problem**: Per-fold selection created only 16/40 common features across folds → high variance
- **Solution**: Select once, lock forever → ICIR improved from 0.16 to 0.80
- **Trade-off**: Cannot adapt to evolving feature importance (acceptable with 166 days of data)

### 14.5 Magnitude-Based Abstention

The most impactful signal quality improvement comes from selective signaling:
- Filter signals by prediction magnitude (|pred| percentile)
- Top 25% by magnitude: IC triples from +0.030 to +0.107
- This matches the product goal: "indicator for significant moves" not "prediction every bar"

### 14.6 Three-Service Separation

Independent services allow:
- Independent scaling (market data is I/O-bound; indicator is CPU-bound)
- Zero-downtime model updates (swap indicator service only)
- Fault isolation (WebSocket disconnection doesn't affect predictions)
- Independent development cycles

---

## Appendix A: File Structure

```
資金機器人/
├── BTC_perp_data.py                    # Service 1: Telegram bot + event tracking
├── outcome_tracker.py                  # Sweep outcome monitoring
├── shared/
│   └── db.py                           # Database connection pool (single source)
│
├── market_data/                        # Service 2: Real-time data ingestion
│   ├── adapters/
│   │   ├── binance_trades.py           # Binance Futures WebSocket adapter
│   │   ├── okx_trades.py              # OKX Futures WebSocket adapter
│   │   ├── funding_collector.py        # Funding rate collector
│   │   ├── oi_collector.py            # Open interest collector
│   │   └── liquidation_collector.py    # Liquidation event collector
│   ├── core/
│   │   ├── trade_normalizer.py         # Unified trade schema normalization
│   │   ├── flow_aggregator.py          # 1-minute flow bar aggregation
│   │   ├── symbol_mapper.py            # Exchange → canonical symbol mapping
│   │   └── health_monitor.py           # Service health monitoring
│   ├── storage/
│   │   ├── db.py                       # Market data DB operations
│   │   ├── flow_repository.py          # Flow bar CRUD
│   │   └── trade_repository.py         # Trade CRUD
│   ├── query/
│   │   ├── flow_context.py             # Read-only flow queries (used by Service 1)
│   │   └── snapshot_query.py           # Market state snapshot queries
│   ├── features/
│   │   ├── snapshot_builder.py         # Real-time feature snapshot assembly
│   │   ├── snapshot_runner.py          # Scheduled snapshot execution
│   │   └── snapshot_repository.py      # Snapshot persistence
│   ├── tasks/
│   │   ├── start_all.py               # Service 2 entry point
│   │   ├── run_trade_streams.py        # WebSocket stream management
│   │   ├── flush_flow_bars.py          # Periodic bar flush task
│   │   └── cleanup.py                 # Old data cleanup
│   └── backfill/
│       ├── coinglass_backfill.py       # Coinglass historical data fetch
│       ├── data_collector.py           # Coordinated backfill runner
│       ├── import_klines.py            # Kline import to DB
│       └── import_aggtrades.py         # aggTrade import to DB
│
├── indicator/                          # Service 3: ML prediction + API
│   ├── app.py                          # Flask app + APScheduler (entry point)
│   ├── wsgi.py                         # Gunicorn WSGI wrapper
│   ├── data_fetcher.py                 # Binance REST + Coinglass API client
│   ├── feature_builder_live.py         # Live feature computation (71 features)
│   ├── feature_config.py               # Feature list + EXCLUDE set (single source)
│   ├── inference.py                    # IndicatorEngine (XGBoost + confidence)
│   ├── chart_renderer.py              # 3-panel PNG chart generation
│   ├── train_export.py                # Model training + artifact export
│   ├── auto_update.py                 # Standalone update loop (fallback)
│   ├── setup_schedule.ps1             # Windows Task Scheduler setup
│   └── model_artifacts/
│       ├── xgb_model.json             # Trained XGBoost model
│       ├── feature_cols.json          # 40 selected feature names
│       └── training_stats.json        # Warmup data for rolling stats
│
├── research/                           # Offline research & experimentation
│   ├── prediction_indicator_v2.py      # E1+E2: Fixed features + raw pred baseline
│   ├── regime_analysis.py             # E3+E4: Regime detection + abstention
│   ├── model_audit.py                 # 4-check model validation audit
│   ├── ic_scan_extended.py            # Feature candidate evaluation (42 tested)
│   ├── pipeline/
│   │   ├── run_pipeline_v2.py         # Full ML data preparation orchestrator
│   │   ├── inject_coinglass.py        # Coinglass data injection + z-scores
│   │   ├── feature_builder_v2.py      # Research feature computation
│   │   └── export_ml.py              # Parquet export
│   ├── bar_generator/
│   │   ├── runner.py                  # Bar generation daemon
│   │   ├── feature_assembler.py       # Multi-source feature assembly
│   │   ├── aligner.py                # Time alignment utilities
│   │   └── time_bars.py              # Time bar construction
│   ├── features/
│   │   ├── cvd.py                     # CVD feature computation
│   │   ├── delta.py                   # Delta/imbalance features
│   │   ├── volume.py                  # Volume features
│   │   ├── oi.py                      # OI features
│   │   └── statistics.py             # Statistical features
│   ├── viz/
│   │   ├── chart_builder.py           # Research chart utilities
│   │   ├── candlestick.py            # Candlestick plot helpers
│   │   └── cvd_line.py               # CVD visualization
│   ├── ml_data/                       # Parquet data files
│   ├── config/
│   │   └── settings.py               # Research configuration
│   └── RESEARCH_BLUEPRINT_v2.md       # Research plan documentation
│
├── migrations/                         # SQL schema migrations
│   ├── 001_market_data_tables.sql
│   ├── 008_market_state_bars.sql
│   ├── 009_flow_bars_15m.sql
│   ├── 010_feature_pipeline.sql
│   └── 011_ml_tables.sql
│
├── Dockerfile                          # Service 1 (main bot)
├── Dockerfile.marketdata               # Service 2 (market data)
├── Dockerfile.indicator                # Service 3 (indicator)
└── docs/
    └── SYSTEM_ARCHITECTURE.md          # This document
```

---

## Appendix B: Evaluation Metrics Glossary

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Spearman IC** | rank_corr(pred, actual) | Prediction ranking quality (-1 to +1); >+0.03 is actionable |
| **ICIR** | mean(fold_ICs) / std(fold_ICs) | Signal stability; >1.0 is good, >2.0 is excellent |
| **Direction Accuracy** | correct_direction / total_active | % of UP/DOWN predictions matching actual sign |
| **Calibration Monotonicity** | % of adjacent deciles in correct order | Higher pred → higher actual return? |
| **Weekly Positive IC** | weeks_with_IC>0 / total_weeks | Temporal consistency of signal |

---

## Appendix C: Research Experiment Registry

| ID | Name | Status | IC Before | IC After | Key Finding |
|----|------|--------|-----------|----------|-------------|
| E1 | Fixed feature selection | Done | +0.022 | +0.030 | ICIR 0.16→0.80 |
| E2 | Remove z-score | Done | +0.022 | +0.030 | Weekly +IC 50%→75% |
| E3 | Regime analysis | Done | — | — | All regimes IC≥0; CHOPPY is noise |
| E4 | Magnitude abstention | Done | +0.030 | +0.107 | 3.5× IC at 25% retention |
| E5 | Confidence v2 | Planned | — | — | Magnitude + regime composite |
| E6 | OOD detection | Planned | — | — | Feature familiarity gate |
| E7 | Local track record | Planned | — | — | Rolling hit rate dampening |
| E8 | LightGBM comparison | Planned | — | — | Alternative tree algorithm |
| E9 | Horizon scan | Planned | — | — | 1h/2h/4h/8h comparison |

---

*Last updated: 2026-04-02*
*Author: Quantitative Research & Engineering*
