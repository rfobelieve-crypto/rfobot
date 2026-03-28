# Skill: Ingest Exchange

Ingest real-time trade data from a crypto exchange via WebSocket.

## Trigger
When adding a new exchange data source or modifying an existing adapter.

## Context
- Adapters live in `market_data/adapters/`
- Each adapter connects to one exchange's public WS
- Output: raw trade dicts → `on_trades_callback`
- Normalizer (`core/trade_normalizer.py`) converts to unified schema
- Health monitor tracks latency, reconnects, gaps

## Current Adapters
- `okx_trades.py` — OKX public trades channel (BTC-USDT-SWAP, ETH-USDT-SWAP)
- `binance_trades.py` — Binance fapi aggTrade stream (BTCUSDT, ETHUSDT)

## Key Rules
- Use `websocket-client` library (sync, same as main bot)
- Auto-reconnect with backoff on disconnect
- Never crash on parse errors — log and skip
- Call `health_monitor.on_trade()` per trade for latency tracking
- Do NOT process or aggregate — that's the normalizer/aggregator's job
