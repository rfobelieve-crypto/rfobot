# Debug Flow Data

Query MySQL to check market data pipeline health.

## Quick checks

```sql
-- Trade counts by exchange (last 10 minutes)
SELECT exchange, COUNT(*)
FROM normalized_trades
WHERE ts_exchange > (UNIX_TIMESTAMP() * 1000 - 600000)
GROUP BY exchange;

-- Latest flow bars
SELECT canonical_symbol, window_start, delta_usd, volume_usd, trade_count, cvd_usd
FROM flow_bars_1m
ORDER BY window_start DESC
LIMIT 10;

-- BTC-USD latest 5 bars
SELECT window_start, delta_usd, volume_usd, cvd_usd, trade_count, source_count
FROM flow_bars_1m
WHERE canonical_symbol = 'BTC-USD'
ORDER BY window_start DESC
LIMIT 5;

-- Health: gap detection (bars with 0 trades)
SELECT canonical_symbol, window_start, trade_count
FROM flow_bars_1m
WHERE trade_count = 0
ORDER BY window_start DESC
LIMIT 10;
```

## Python quick test
```python
from market_data.query.flow_context import get_pre_sweep_context, format_flow_context
ctx = get_pre_sweep_context("BTC-USD", lookback_minutes=5)
print(format_flow_context(ctx, title="BTC-USD last 5m"))
```
