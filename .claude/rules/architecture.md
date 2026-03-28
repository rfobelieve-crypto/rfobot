# Architecture Rules

## Two-service separation
- Service 1 (main bot) and Service 2 (market data) are independent processes
- They share the same MySQL database but do NOT import each other's modules at runtime
- Exception: main bot imports `market_data.query.flow_context` for read-only DB queries

## Do not break the main flow
- `BTC_perp_data.py` is the production Telegram bot with real users
- Any change to this file must NOT break: Flask routes, TradingView webhook, Telegram commands, OKX WebSocket, outcome_tracker integration
- Test changes locally before pushing

## Shared DB only
- All MySQL access goes through `shared/db.py`
- Do NOT create separate connection logic in any module
- `shared/db.py` supports: env vars → `.env` file → `config.json` fallback

## Market data is additive
- New exchange adapters, new aggregation logic, new query modules go under `market_data/`
- Do NOT modify existing adapter behavior when adding new ones
- Each adapter is responsible only for: WS connection, raw parsing, callback

## Railway deployment
- Push to `main` triggers auto-deploy on both services
- Service 1 uses `Dockerfile`, Service 2 uses `Dockerfile.marketdata`
- Railway env vars take priority over `.env` (which is gitignored)
- Internal MySQL host: `mysql.railway.internal`, external for local dev via `.env`
