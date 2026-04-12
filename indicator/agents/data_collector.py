"""
DataCollector Agent — AI-powered data pipeline health monitor.

This agent acts as a senior SRE monitoring all external data sources.
It feeds raw API responses, timing data, and error details to Claude,
who diagnoses patterns like: correlated failures, silent degradation,
rate limiting, API key issues, and emerging outages.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from indicator.agents.base import BaseAgent

logger = logging.getLogger(__name__)

TZ_TPE = timezone(timedelta(hours=8))


class DataCollectorAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "DataCollector"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior SRE (Site Reliability Engineer) monitoring the data pipeline \
of a BTC prediction indicator system. You have deep experience with crypto \
market data APIs and know their failure modes intimately.

## System Architecture
- Runs on Railway (cloud PaaS), updates every hour at :02
- Predictions require features from 3 data sources combined
- If Binance klines fail → NO predictions possible (complete outage)
- If Coinglass fails → features degrade (NaN-filled), predictions unreliable
- If Deribit fails → minor impact (options features only)

## Data Sources & Known Failure Modes

### Binance REST API (fapi.binance.com)
- **klines**: 1h OHLCV candles. Most critical. Failure = total outage.
  - Common issues: rate limit (418/429), IP ban, maintenance window
  - Normal latency: <1s. If >3s, something is wrong.
  - Should return 500 bars. <10 bars = possible API change.
  - Last bar age should be <120 min (current hour not yet closed)
- **depth**: L20 order book snapshot. Used for liquidity features.
  - Failure impact: depth_imbalance=0, spread features missing
- **aggTrades**: Recent aggregated trades. Used for flow features.
  - Failure impact: large_trade features = 0

### Coinglass API v4 (open-api-v4.coinglass.com)
- 14 timeseries endpoints (OI, funding, liquidation, long/short ratios, etc.)
- 5+ snapshot endpoints (ETF flow, fear/greed, whale positions)
- **API key required** — 401 errors = key missing/expired/rate-limited
- Rate limit: headers may contain X-RateLimit info
- Cache fallback: system keeps parquet cache of last good fetch
  - Cache <4h old = acceptable degradation
  - Cache >24h old = stale, features are misleading
- Endpoints often have different reliability — some may return empty while others work
- If ALL endpoints return 401 → API key issue, not individual endpoint problem

### Deribit Public API (www.deribit.com)
- DVOL (BTC implied volatility index) and options summary
- Public, no auth needed. Rarely fails.
- If both fail → Deribit might be in maintenance

## What to Look For
1. **Correlated failures**: All Coinglass 401 = key issue, not random failures
2. **Silent degradation**: APIs returning data but with stale timestamps
3. **Cache dependency**: If live APIs fail, how old is the cache fallback?
4. **Gap accumulation**: Missing bars in indicator_history compound over time
5. **Timing anomalies**: If data is from 3+ hours ago, the scheduler might be stuck
6. **Weekend/holiday patterns**: Lower volume is normal, but API failures aren't

## Your Response Style
- Think like an oncall engineer at 3am: what's broken, what's the impact, what do I do
- Be specific: "Coinglass OI endpoint returned 401" not "some endpoints failed"
- Quantify impact: "12/14 CG endpoints on cache fallback (2.1h old) = acceptable for now"
- Distinguish between "needs immediate action" vs "monitor and wait"
- Always respond in Traditional Chinese for the summary/report fields"""

    def collect_context(self) -> dict:
        """
        Gather raw data from every source — timing, errors, actual values.
        Feed Claude the numbers, not our interpretation.
        """
        context = {}

        # ── Binance ──
        context["binance"] = self._probe_binance()

        # ── Coinglass ──
        context["coinglass"] = self._probe_coinglass()

        # ── Deribit ──
        context["deribit"] = self._probe_deribit()

        # ── MySQL indicator_history ──
        context["indicator_history"] = self._probe_history()

        # ── Local cache files ──
        context["cache_files"] = self._probe_cache()

        # ── Environment ──
        import os
        context["environment"] = {
            "coinglass_key_set": bool(os.environ.get("COINGLASS_API_KEY")),
            "telegram_configured": bool(os.environ.get("INDICATOR_BOT_TOKEN")),
            "railway_env": bool(os.environ.get("RAILWAY_ENVIRONMENT")),
        }

        return context

    def get_available_actions(self) -> list[dict]:
        return [
            {"name": "ALERT", "description": "Send Telegram alert to operator with diagnosis"},
            {"name": "RETRY_ENDPOINT", "description": "Retry a specific failed endpoint"},
            {"name": "LOG", "description": "Log finding for historical tracking"},
            {"name": "NONE", "description": "Informational only, no action needed"},
        ]

    def execute_action(self, action_name: str, finding: dict, context: dict) -> str:
        if action_name == "RETRY_ENDPOINT":
            endpoint = finding.get("title", "unknown")
            logger.info("[DataCollector] Retrying endpoint: %s", endpoint)
            # Could implement actual retry logic here
            return f"Retry requested: {endpoint}"
        return super().execute_action(action_name, finding, context)

    # ── Probes (raw data collection, no interpretation) ─────────────────

    def _probe_binance(self) -> dict:
        """Probe all Binance endpoints, measure timing."""
        result = {}

        # Klines
        try:
            from indicator.data_fetcher import fetch_binance_klines
            t0 = time.time()
            klines = fetch_binance_klines(limit=10)
            latency = time.time() - t0

            if klines.empty:
                result["klines"] = {"available": False, "latency_s": round(latency, 2)}
            else:
                last_ts = klines.index[-1]
                if hasattr(last_ts, 'to_pydatetime'):
                    last_ts = last_ts.to_pydatetime()
                if last_ts.tzinfo is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)

                result["klines"] = {
                    "available": True,
                    "rows": len(klines),
                    "last_bar_utc": str(last_ts),
                    "age_minutes": round((datetime.now(timezone.utc) - last_ts).total_seconds() / 60, 1),
                    "latest_close": float(klines["close"].iloc[-1]),
                    "latest_volume": float(klines["volume"].iloc[-1]) if "volume" in klines.columns else None,
                    "latency_s": round(latency, 2),
                }
        except Exception as e:
            result["klines"] = {"available": False, "error": str(e)}

        # Depth
        try:
            from indicator.data_fetcher import fetch_binance_depth
            t0 = time.time()
            depth = fetch_binance_depth()
            latency = time.time() - t0

            if depth:
                result["depth"] = {
                    "available": True,
                    "bid_depth_usd": depth.get("bid_depth_usd"),
                    "ask_depth_usd": depth.get("ask_depth_usd"),
                    "depth_imbalance": depth.get("depth_imbalance"),
                    "spread_bps": depth.get("spread_bps"),
                    "latency_s": round(latency, 2),
                }
            else:
                result["depth"] = {"available": False, "latency_s": round(latency, 2)}
        except Exception as e:
            result["depth"] = {"available": False, "error": str(e)}

        # AggTrades
        try:
            from indicator.data_fetcher import fetch_binance_aggtrades
            t0 = time.time()
            agg = fetch_binance_aggtrades(lookback_hours=1)
            latency = time.time() - t0

            if agg:
                result["aggtrades"] = {
                    "available": True,
                    "total_count": agg.get("total_count", 0),
                    "large_buy_usd": agg.get("large_buy_usd", 0),
                    "large_sell_usd": agg.get("large_sell_usd", 0),
                    "large_delta_usd": agg.get("large_delta_usd", 0),
                    "avg_trade_usd": agg.get("avg_trade_usd", 0),
                    "latency_s": round(latency, 2),
                }
            else:
                result["aggtrades"] = {"available": False, "latency_s": round(latency, 2)}
        except Exception as e:
            result["aggtrades"] = {"available": False, "error": str(e)}

        return result

    def _probe_coinglass(self) -> dict:
        """Probe all Coinglass endpoints individually."""
        result = {"timeseries": {}, "snapshots": {}}

        # Timeseries (14 endpoints)
        try:
            from indicator.data_fetcher import fetch_coinglass
            t0 = time.time()
            cg_data = fetch_coinglass(interval="1h", limit=5)
            total_latency = time.time() - t0

            live_count = 0
            cache_count = 0
            fail_count = 0

            for name, df in cg_data.items():
                if df.empty:
                    result["timeseries"][name] = {"status": "empty", "rows": 0}
                    fail_count += 1
                else:
                    # Check if data looks fresh or stale (cache)
                    result["timeseries"][name] = {
                        "status": "ok",
                        "rows": len(df),
                    }
                    # Can't easily distinguish live vs cache here,
                    # but the log messages captured in _probe tell us
                    live_count += 1

            result["timeseries_summary"] = {
                "total_endpoints": len(cg_data),
                "ok": live_count,
                "empty": fail_count,
                "total_latency_s": round(total_latency, 1),
            }
        except Exception as e:
            result["timeseries_summary"] = {"error": str(e)}

        # Snapshot endpoints (each independent)
        from indicator.data_fetcher import (
            fetch_cg_options, fetch_cg_etf_flow, fetch_cg_fear_greed,
            fetch_cg_futures_netflow, fetch_cg_spot_netflow,
        )
        snapshot_fns = {
            "options": fetch_cg_options,
            "etf_flow": fetch_cg_etf_flow,
            "fear_greed": fetch_cg_fear_greed,
            "futures_netflow": fetch_cg_futures_netflow,
            "spot_netflow": fetch_cg_spot_netflow,
        }

        for name, fn in snapshot_fns.items():
            try:
                t0 = time.time()
                data = fn()
                latency = time.time() - t0
                result["snapshots"][name] = {
                    "available": bool(data),
                    "fields": len(data) if data else 0,
                    "latency_s": round(latency, 2),
                    "sample_keys": list(data.keys())[:5] if data else [],
                }
            except Exception as e:
                result["snapshots"][name] = {"available": False, "error": str(e)}

        return result

    def _probe_deribit(self) -> dict:
        """Probe Deribit endpoints."""
        result = {}

        try:
            from indicator.data_fetcher import fetch_deribit_dvol
            t0 = time.time()
            dvol = fetch_deribit_dvol()
            latency = time.time() - t0
            result["dvol"] = {
                "available": bool(dvol),
                "dvol_value": dvol.get("dvol") if dvol else None,
                "dvol_change": dvol.get("dvol_change") if dvol else None,
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            result["dvol"] = {"available": False, "error": str(e)}

        try:
            from indicator.data_fetcher import fetch_deribit_options_summary
            t0 = time.time()
            opts = fetch_deribit_options_summary()
            latency = time.time() - t0
            result["options"] = {
                "available": bool(opts),
                "pc_vol_ratio": opts.get("pc_vol_ratio") if opts else None,
                "pc_oi_ratio": opts.get("pc_oi_ratio") if opts else None,
                "latency_s": round(latency, 2),
            }
        except Exception as e:
            result["options"] = {"available": False, "error": str(e)}

        return result

    def _probe_history(self) -> dict:
        """Check indicator_history in MySQL for gaps and freshness."""
        try:
            from shared.db import get_db_conn
            conn = get_db_conn()
            with conn.cursor() as cur:
                # Basic stats
                cur.execute("""
                    SELECT COUNT(*) as total,
                           MIN(dt) as first_bar,
                           MAX(dt) as last_bar
                    FROM indicator_history
                """)
                stats = cur.fetchone()

                # Recent bars (should be 24 per day)
                cur.execute("""
                    SELECT COUNT(*) as n
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                """)
                recent_24h = cur.fetchone()["n"]

                # 7-day bar count
                cur.execute("""
                    SELECT COUNT(*) as n
                    FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                """)
                bars_7d = cur.fetchone()["n"]

                # Recent gaps: find consecutive bar gaps in last 48h
                cur.execute("""
                    SELECT dt FROM indicator_history
                    WHERE dt >= DATE_SUB(NOW(), INTERVAL 48 HOUR)
                    ORDER BY dt
                """)
                recent_bars = [r["dt"] for r in cur.fetchall()]

                # Find gaps > 1h
                gaps = []
                for i in range(1, len(recent_bars)):
                    diff_h = (recent_bars[i] - recent_bars[i-1]).total_seconds() / 3600
                    if diff_h > 1.5:  # Allow some tolerance
                        gaps.append({
                            "from": str(recent_bars[i-1]),
                            "to": str(recent_bars[i]),
                            "gap_hours": round(diff_h, 1),
                        })

                # Last 5 bars: show pred_return_4h values to detect σ-scale issues
                cur.execute("""
                    SELECT dt, close, pred_return_4h, pred_direction_code,
                           confidence_score, regime_code, mag_pred
                    FROM indicator_history
                    ORDER BY dt DESC LIMIT 5
                """)
                last_5 = []
                for row in cur.fetchall():
                    last_5.append({
                        "dt": str(row["dt"]),
                        "close": float(row["close"]) if row["close"] else None,
                        "pred_return_4h": float(row["pred_return_4h"]) if row["pred_return_4h"] else None,
                        "direction": {1: "UP", -1: "DOWN", 0: "NEUTRAL"}.get(row["pred_direction_code"], "?"),
                        "confidence": float(row["confidence_score"]) if row["confidence_score"] else None,
                        "regime": {2: "BULL", -2: "BEAR", 0: "CHOPPY", -99: "WARMUP"}.get(row["regime_code"], "?"),
                        "mag_pred": float(row["mag_pred"]) if row["mag_pred"] else None,
                    })

            conn.close()

            latest = stats["last_bar"]
            age_min = None
            if latest:
                latest_utc = latest.replace(tzinfo=timezone.utc) if hasattr(latest, 'replace') else latest
                age_min = round((datetime.now(timezone.utc) - latest_utc).total_seconds() / 60, 1)

            return {
                "total_rows": stats["total"],
                "first_bar": str(stats["first_bar"]),
                "last_bar": str(stats["last_bar"]),
                "age_minutes": age_min,
                "bars_24h": recent_24h,
                "expected_24h": 24,
                "bars_7d": bars_7d,
                "expected_7d": 168,
                "recent_gaps": gaps,
                "last_5_predictions": last_5,
            }
        except Exception as e:
            return {"error": str(e)}

    def _probe_cache(self) -> dict:
        """Check local cache files for age and size."""
        cache_dir = Path("indicator/model_artifacts/.data_cache")
        if not cache_dir.exists():
            return {"exists": False}

        files = {}
        for f in sorted(cache_dir.glob("*.parquet")):
            try:
                stat = f.stat()
                age_h = (datetime.now().timestamp() - stat.st_mtime) / 3600
                files[f.stem] = {
                    "size_kb": round(stat.st_size / 1024, 1),
                    "age_hours": round(age_h, 1),
                }
            except Exception:
                files[f.stem] = {"error": "stat failed"}

        return {"exists": True, "files": files}


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = DataCollectorAgent()

    if "--context-only" in sys.argv:
        ctx = agent.collect_context()
        print(_json.dumps(ctx, indent=2, default=str, ensure_ascii=False))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Duration: {result.duration_s:.1f}s")
        if result.actions_taken:
            for a in result.actions_taken:
                print(f"  Action: {a}")
        print(f"\nFull diagnosis:\n{_json.dumps(result.diagnosis, indent=2, ensure_ascii=False)}")
