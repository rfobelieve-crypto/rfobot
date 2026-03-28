# Flowbot - BTC Liquidity Sweep Analysis System

## ROLE
Quantitative trading research assistant specialized in:
- Market microstructure & liquidity sweeps (ICT concepts)
- Order flow analysis (delta, CVD, taker flow)
- Short-term crypto perpetual futures trading

Goal: transform raw trading data into validated statistical trading edge.

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│  Railway Service 1: Main Bot (BTC_perp_data.py)         │
│  Flask + OKX WS + Telegram + TradingView webhook        │
│  Writes: liquidity_events, sweep_outcomes                │
└──────────────────────┬──────────────────────────────────┘
                       │ shared/db.py (MySQL)
┌──────────────────────┴──────────────────────────────────┐
│  Railway Service 2: Market Data (market_data/)           │
│  OKX WS + Binance WS → normalize → aggregate            │
│  Writes: normalized_trades, flow_bars_1m                 │
└─────────────────────────────────────────────────────────┘
```

### Service 1: Main Bot
- Entry: `python BTC_perp_data.py`
- Dockerfile: `Dockerfile`
- Responsibilities:
  - TradingView webhook (`/tv`) → receive BSL/SSL sweep events
  - OKX WebSocket → track BTC taker flow per event (15m/1h/4h windows)
  - Outcome tracking (outcome_tracker.py) → ±0.5% first-hit, 15m/1h windows
  - Telegram bot → notifications + commands
  - Query market_data flow_bars_1m for pre-sweep & event-period context

### Service 2: Market Data Layer
- Entry: `python -m market_data.tasks.start_all`
- Dockerfile: `Dockerfile.marketdata`
- Responsibilities:
  - OKX + Binance perpetual trades WebSocket
  - Trade normalization → unified schema
  - 1-minute flow aggregation (delta, volume, CVD)
  - Health monitoring per source

### Shared
- `shared/db.py` — MySQL connection helper (env → .env → config.json fallback)

---

## DEPLOYMENT

- Platform: Railway (auto-deploy on push to `main`)
- Repo: `rfobelieve-crypto/rfobot` on GitHub
- MySQL: Railway internal (`mysql.railway.internal:3306`)
- Local dev: `.env` with external address (`caboose.proxy.rlwy.net:18766`)

---

## MYSQL TABLES

| Table | Owner | Purpose |
|-------|-------|---------|
| `liquidity_events` | Service 1 | v2 sweep events with flow/return/result |
| `sweep_outcomes` | Service 1 | ±0.5% first-hit outcome tracking (15m/1h) |
| `instruments` | Service 2 | Exchange symbol registry |
| `normalized_trades` | Service 2 | Raw normalized trades (debug/replay) |
| `flow_bars_1m` | Service 2 | 1-minute aggregated flow bars |

---

## KEY CONVENTIONS

- Canonical symbols: `BTC-USD`, `ETH-USD`
- flow_bars_1m.exchange_scope = `"all"` (OKX+Binance combined)
- Timestamps: Unix milliseconds (ms) in market_data, Unix seconds (s) in main bot
- OKX contract sizes: BTC=0.01, ETH=0.1
- Binance aggTrade: already in base units (contract_size=1)
- notional_usd = price * size * contract_size

---

## FILE STRUCTURE

```
BTC_perp_data.py          # Main bot entry point
outcome_tracker.py        # Sweep outcome tracker module
shared/
  db.py                   # Shared MySQL connection helper
market_data/
  adapters/
    okx_trades.py         # OKX WS adapter
    binance_trades.py     # Binance WS adapter
  core/
    symbol_mapper.py      # SYMBOL_MAP + CONTRACT_INFO
    trade_normalizer.py   # Raw → unified trade schema
    flow_aggregator.py    # 1m bucket aggregation + CVD
    health_monitor.py     # Per-source health tracking
  storage/
    db.py                 # Delegates to shared/db.py
    trade_repository.py   # Batch insert trades
    flow_repository.py    # Upsert flow bars
  query/
    flow_context.py       # Query flow_bars_1m for sweep context
  tasks/
    start_all.py          # Combined entry point
    run_trade_streams.py  # Start adapters + batch writer
    flush_flow_bars.py    # Periodic flow bar flusher
migrations/
  001_market_data_tables.sql
```

---

## CONSTRAINTS

- Do NOT hallucinate data
- Do NOT assume causation without evidence
- Always prefer statistical validation
- Think like a quant, not a trader
- If unclear, ask for more data
