# Data Model Rules

## Canonical Symbols
- All internal references use canonical format: `BTC-USD`, `ETH-USD`
- Mapping from exchange raw symbols is in `market_data/core/symbol_mapper.py`
- Binance: `BTCUSDT` → `BTC-USD`, `ETHUSDT` → `ETH-USD`
- OKX: `BTC-USDT-SWAP` → `BTC-USD`, `ETH-USDT-SWAP` → `ETH-USD`

## Unified Trade Schema
All trades must be normalized to:
```
exchange, raw_symbol, canonical_symbol, instrument_type("perp"),
price, size, size_unit, taker_side("buy"/"sell"),
notional_usd, trade_id, ts_exchange(ms), ts_received(ms),
is_aggregated_trade
```

## Notional Calculation
- `notional_usd = price * size * contract_size`
- Binance: contract_size=1 (base units), size_unit="base"
- OKX BTC: contract_size=0.01, size_unit="contract"
- OKX ETH: contract_size=0.1, size_unit="contract"

## Timestamps
- market_data layer: Unix milliseconds (ms) everywhere
- Main bot (BTC_perp_data.py): Unix seconds (s) — legacy, do not change
- When crossing the boundary, convert explicitly

## Flow Bars (flow_bars_1m)
- Bucket key: (canonical_symbol, minute_start_ms)
- exchange_scope = "all" (Phase 1: always combined)
- Fields: buy_notional_usd, sell_notional_usd, delta_usd, volume_usd, trade_count, cvd_usd
- CVD is cumulative: cvd(bar_n) = cvd(bar_n-1) + delta(bar_n)
- Flushed every ~10 seconds, only completed bars (window_end <= now)

## Liquidity Events (liquidity_events)
- Written by main bot on event completion (after 4h observation)
- Contains: entry_price, flow stats per window (15m/1h/4h), forward returns, first-hit, result classification
- liquidity_side: "buy" = BSL, "sell" = SSL

## Sweep Outcomes (sweep_outcomes)
- Written by outcome_tracker on all windows completed
- Independent multi-event tracking (not limited to 1 active event)
- ±0.5% threshold, 15m and 1h observation windows
