# Deploy to Railway

Push current changes to GitHub to trigger Railway auto-deploy.

## Steps
1. Check `git status` for untracked/modified files
2. Verify `.env` is gitignored (never commit secrets)
3. Stage relevant files (not `.env`, not `config.json`, not `__pycache__/`)
4. Commit with descriptive message
5. `git push origin main`
6. Railway auto-deploys both services:
   - Service 1: `Dockerfile` → `BTC_perp_data.py`
   - Service 2: `Dockerfile.marketdata` → `market_data.tasks.start_all`

## Post-deploy check
- Service 1: Telegram `/status` command should respond
- Service 2: Check Railway logs for `[Health] binance: status=healthy`
