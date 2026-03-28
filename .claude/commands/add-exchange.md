# Add a New Exchange Adapter

Steps to add a new exchange to the market data layer.

## 1. Symbol Mapper
Edit `market_data/core/symbol_mapper.py`:
- Add entries to `SYMBOL_MAP["new_exchange"]`
- Add entries to `CONTRACT_INFO["new_exchange"]` with correct contract_size and size_unit

## 2. Create Adapter
Create `market_data/adapters/new_exchange_trades.py`:
- Class with `__init__(self, on_trades_callback)` and `start()`/`stop()`
- WebSocket connection with auto-reconnect loop
- Parse raw trades into dict with fields:
  `exchange, raw_symbol, price, size, taker_side, trade_id, ts_exchange, ts_received, is_aggregated_trade`
- Call `health_monitor.on_trade()` for each trade
- Call `health_monitor.on_reconnect()` on reconnect

## 3. Register in Runner
Edit `market_data/tasks/run_trade_streams.py`:
- Import the new adapter
- Instantiate with `on_raw_trades` callback
- Start in a new daemon thread

## 4. Migration
Add instrument rows to `migrations/` (or INSERT directly):
```sql
INSERT IGNORE INTO instruments (exchange, raw_symbol, canonical_symbol, instrument_type, contract_size, size_unit)
VALUES ('new_exchange', 'BTCUSDT', 'BTC-USD', 'perp', 1.0, 'base');
```

## 5. Test
- Run `python -m market_data.tasks.start_all` locally
- Check logs for `[NewExchange] Connected`
- Query `SELECT exchange, COUNT(*) FROM normalized_trades GROUP BY exchange`
