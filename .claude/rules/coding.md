# Coding Rules

## Language & Runtime
- Python 3.11
- Dependencies: flask, requests, websocket-client, pymysql, gunicorn
- No async frameworks (main bot uses threading)
- No CCXT — direct exchange WebSocket connections

## Error Handling
- Adapters: catch all exceptions in on_message, log and continue
- DB writes: catch and log, never crash the main loop
- Webhook: always return 200 (even on error) to avoid TradingView retries
- Use `logger.exception()` for stack traces

## DB Patterns
- Always use `get_db_conn()` from `shared/db.py`
- Always close connections in `finally` blocks
- Batch inserts for trades (100 trades or 5s interval)
- Upsert (ON DUPLICATE KEY UPDATE) for flow bars
- No ORM — raw pymysql with parameterized queries

## Threading
- Main bot: background daemon threads for WS, watchdogs, cleanup
- Market data: same pattern — daemon threads for adapters, flusher, health
- Use `threading.Lock()` for shared state
- Never block the Flask thread

## Naming
- Files: snake_case
- Canonical symbols: `BTC-USD` format (uppercase, hyphen-separated)
- DB columns: snake_case
- Exchange names: lowercase (`"binance"`, `"okx"`)

## Secrets
- Never commit `.env` or `config.json`
- `.gitignore` protects these
- `.env.example` has placeholder values only
