# Skill: Compute Flow

Aggregate normalized trades into 1-minute flow bars and provide flow context queries.

## Trigger
When working on flow aggregation logic, flow bar schema, or sweep context queries.

## Context
- Aggregator: `market_data/core/flow_aggregator.py`
- Query: `market_data/query/flow_context.py`
- Storage: `market_data/storage/flow_repository.py` → `flow_bars_1m` table

## Flow Bar Schema
```
canonical_symbol, instrument_type("perp"), exchange_scope("all"),
window_start(ms), window_end(ms),
buy_notional_usd, sell_notional_usd, delta_usd, volume_usd,
trade_count, cvd_usd, source_count, quality_score
```

## Aggregation Rules
- Bucket key: (canonical_symbol, minute_start_ms)
- buy side: buy_notional_usd += notional_usd
- sell side: sell_notional_usd += notional_usd
- delta = buy - sell
- volume = buy + sell
- CVD is cumulative across bars: cvd(n) = cvd(n-1) + delta(n)
- Flush only completed bars (window_end <= now)
- Upsert to DB (ON DUPLICATE KEY UPDATE)

## Query Functions
- `get_pre_sweep_context(symbol, lookback_minutes)` — market state before sweep
- `get_event_flow_context(symbol, event_ts, duration_minutes)` — flow during event
- `format_flow_context(ctx, title)` — Telegram-friendly text output
