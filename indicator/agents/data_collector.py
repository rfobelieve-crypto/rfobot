"""
DataCollector Agent — Tool-Use Architecture.

Claude gets tools to probe each data source independently and decides
what to investigate and how deep to go. If Binance looks fine, it moves on.
If Coinglass shows 401s, it drills into which endpoints and checks the cache.
"""
from __future__ import annotations

import json
import logging
import os
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
You are a senior SRE monitoring the data pipeline of a BTC prediction system.

## Your Investigation Protocol
1. Start with the MOST CRITICAL source: Binance klines (without it, system is blind)
2. Check Coinglass (14 timeseries + 5 snapshots) — it powers most features
3. Check Deribit (options data, lower priority)
4. Check indicator_history for gaps and freshness
5. If any source failed, check the cache fallback status
6. Conclude with your diagnosis

## What You Know
- Binance klines failure = total prediction outage
- Coinglass 401 = API key issue (check env var), not endpoint-specific
- Cache fallback: acceptable if <4h old, stale if >24h
- Data gaps in indicator_history = scheduler might be stuck
- pred_return_4h should be ±0.001~0.05. If >0.5 → σ-scale bug
- mag_pred should be <0.05. If >1 → magnitude not converted from vol-adjusted scale
- System runs hourly at :02. If last bar is >2h old → something broke.

## Severity
- critical: Binance klines down, or ALL CG endpoints failed with no cache, or >4h gap
- degraded: >50% CG endpoints on cache, data >2h old, any systematic failure
- healthy: all sources returning data, freshness <90min

## Self-Healing Actions (use repair tools when criteria are met)
- Cache files >12h old + source is alive → `repair_clear_stale_cache` (force fresh fetch)
- Last update >2h old → `repair_trigger_update` (kick the scheduler)
- Always verify after repair: check if the issue is resolved before concluding.
- Send Telegram alert for any repair action taken.

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "check_binance_klines",
                "description": "Fetch Binance 1h klines and return: availability, row count, last bar timestamp, age, latest close price, latency.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_binance_depth",
                "description": "Fetch Binance L20 order book depth: bid/ask depth USD, imbalance, spread.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_binance_aggtrades",
                "description": "Fetch Binance aggregated trades: trade count, large order flow, delta.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_coinglass_timeseries",
                "description": "Fetch all 14 Coinglass timeseries endpoints. Returns per-endpoint status (ok/empty/error), total latency, and summary of failures.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_coinglass_snapshots",
                "description": "Fetch Coinglass snapshot endpoints (options, ETF flow, fear/greed, netflow). Returns per-endpoint availability and sample data.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_deribit",
                "description": "Fetch Deribit DVOL and options summary: implied vol, put/call ratios, latency.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_indicator_history",
                "description": "Query MySQL indicator_history: total rows, last bar age, recent gaps, last 5 predictions (with pred_return_4h and mag_pred values to detect scale bugs).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_cache_status",
                "description": "Check local parquet cache files: which exist, how old, how large. Stale cache >4h means fallback data is unreliable.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_environment",
                "description": "Check if required environment variables are set (API keys, DB config, Telegram).",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        handlers = {
            "check_binance_klines": self._check_binance_klines,
            "check_binance_depth": self._check_binance_depth,
            "check_binance_aggtrades": self._check_binance_aggtrades,
            "check_coinglass_timeseries": self._check_cg_timeseries,
            "check_coinglass_snapshots": self._check_cg_snapshots,
            "check_deribit": self._check_deribit,
            "check_indicator_history": self._check_history,
            "check_cache_status": self._check_cache,
            "check_environment": self._check_env,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        try:
            result = handler()
            return json.dumps(result, default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Tool implementations ────────────────────────────────────────────

    def _check_binance_klines(self) -> dict:
        from indicator.data_fetcher import fetch_binance_klines
        t0 = time.time()
        klines = fetch_binance_klines(limit=10)
        latency = time.time() - t0
        if klines.empty:
            return {"available": False, "latency_s": round(latency, 2)}
        last_ts = klines.index[-1]
        if hasattr(last_ts, 'to_pydatetime'):
            last_ts = last_ts.to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        return {
            "available": True,
            "rows": len(klines),
            "last_bar_utc": str(last_ts),
            "age_minutes": round((datetime.now(timezone.utc) - last_ts).total_seconds() / 60, 1),
            "latest_close": float(klines["close"].iloc[-1]),
            "latest_volume": float(klines["volume"].iloc[-1]) if "volume" in klines.columns else None,
            "latency_s": round(latency, 2),
        }

    def _check_binance_depth(self) -> dict:
        from indicator.data_fetcher import fetch_binance_depth
        t0 = time.time()
        depth = fetch_binance_depth()
        latency = time.time() - t0
        if not depth:
            return {"available": False, "latency_s": round(latency, 2)}
        return {
            "available": True,
            "bid_depth_usd": depth.get("bid_depth_usd"),
            "ask_depth_usd": depth.get("ask_depth_usd"),
            "depth_imbalance": depth.get("depth_imbalance"),
            "spread_bps": depth.get("spread_bps"),
            "latency_s": round(latency, 2),
        }

    def _check_binance_aggtrades(self) -> dict:
        from indicator.data_fetcher import fetch_binance_aggtrades
        t0 = time.time()
        agg = fetch_binance_aggtrades(lookback_hours=1)
        latency = time.time() - t0
        if not agg:
            return {"available": False, "latency_s": round(latency, 2)}
        return {
            "available": True,
            "total_count": agg.get("total_count", 0),
            "large_buy_usd": agg.get("large_buy_usd", 0),
            "large_sell_usd": agg.get("large_sell_usd", 0),
            "large_delta_usd": agg.get("large_delta_usd", 0),
            "avg_trade_usd": agg.get("avg_trade_usd", 0),
            "latency_s": round(latency, 2),
        }

    def _check_cg_timeseries(self) -> dict:
        from indicator.data_fetcher import fetch_coinglass
        t0 = time.time()
        cg_data = fetch_coinglass(interval="1h", limit=5)
        latency = time.time() - t0
        endpoints = {}
        ok = 0
        empty = 0
        for name, df in cg_data.items():
            if df.empty:
                endpoints[name] = "empty"
                empty += 1
            else:
                endpoints[name] = f"ok ({len(df)} rows)"
                ok += 1
        return {
            "total_endpoints": ok + empty,
            "ok": ok,
            "empty": empty,
            "fail_rate_pct": round(empty / (ok + empty) * 100, 1) if (ok + empty) > 0 else 0,
            "total_latency_s": round(latency, 1),
            "endpoints": endpoints,
        }

    def _check_cg_snapshots(self) -> dict:
        from indicator.data_fetcher import (
            fetch_cg_options, fetch_cg_etf_flow, fetch_cg_fear_greed,
            fetch_cg_futures_netflow, fetch_cg_spot_netflow,
        )
        results = {}
        for name, fn in [("options", fetch_cg_options), ("etf_flow", fetch_cg_etf_flow),
                         ("fear_greed", fetch_cg_fear_greed),
                         ("futures_netflow", fetch_cg_futures_netflow),
                         ("spot_netflow", fetch_cg_spot_netflow)]:
            try:
                t0 = time.time()
                data = fn()
                latency = time.time() - t0
                results[name] = {
                    "available": bool(data),
                    "fields": len(data) if data else 0,
                    "sample": {k: v for k, v in list(data.items())[:3]} if data else {},
                    "latency_s": round(latency, 2),
                }
            except Exception as e:
                results[name] = {"available": False, "error": str(e)}
        return results

    def _check_deribit(self) -> dict:
        from indicator.data_fetcher import fetch_deribit_dvol, fetch_deribit_options_summary
        result = {}
        for name, fn in [("dvol", fetch_deribit_dvol), ("options", fetch_deribit_options_summary)]:
            try:
                t0 = time.time()
                data = fn()
                latency = time.time() - t0
                result[name] = {"available": bool(data), "data": data, "latency_s": round(latency, 2)}
            except Exception as e:
                result[name] = {"available": False, "error": str(e)}
        return result

    def _check_history(self) -> dict:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as n, MIN(dt) as first_dt, MAX(dt) as last_dt FROM indicator_history")
            stats = cur.fetchone()
            cur.execute("SELECT COUNT(*) as n FROM indicator_history WHERE dt >= DATE_SUB(NOW(), INTERVAL 24 HOUR)")
            bars_24h = cur.fetchone()["n"]
            # Gaps in last 48h
            cur.execute("SELECT dt FROM indicator_history WHERE dt >= DATE_SUB(NOW(), INTERVAL 48 HOUR) ORDER BY dt")
            bars = [r["dt"] for r in cur.fetchall()]
            gaps = []
            for i in range(1, len(bars)):
                diff_h = (bars[i] - bars[i-1]).total_seconds() / 3600
                if diff_h > 1.5:
                    gaps.append({"from": str(bars[i-1]), "to": str(bars[i]), "hours": round(diff_h, 1)})
            # Last 5 predictions
            cur.execute("""
                SELECT dt, close, pred_return_4h, pred_direction_code, confidence_score, regime_code, mag_pred
                FROM indicator_history ORDER BY dt DESC LIMIT 5
            """)
            last_5 = []
            for r in cur.fetchall():
                last_5.append({
                    "dt": str(r["dt"]),
                    "close": float(r["close"] or 0),
                    "pred_return_4h": float(r["pred_return_4h"] or 0),
                    "direction": {1:"UP", -1:"DOWN", 0:"NEUTRAL"}.get(r["pred_direction_code"], "?"),
                    "confidence": float(r["confidence_score"] or 0),
                    "regime": {2:"BULL", -2:"BEAR", 0:"CHOPPY", -99:"WARMUP"}.get(r["regime_code"], "?"),
                    "mag_pred": float(r["mag_pred"] or 0),
                })
        conn.close()
        latest = stats["last_dt"]
        age_min = None
        if latest:
            latest_utc = latest.replace(tzinfo=timezone.utc) if hasattr(latest, 'replace') else latest
            age_min = round((datetime.now(timezone.utc) - latest_utc).total_seconds() / 60, 1)
        return {
            "total_rows": stats["n"],
            "first_bar": str(stats["first_dt"]),
            "last_bar": str(stats["last_dt"]),
            "age_minutes": age_min,
            "bars_24h": bars_24h, "expected_24h": 24,
            "recent_gaps": gaps,
            "last_5_predictions": last_5,
        }

    def _check_cache(self) -> dict:
        cache_dir = Path("indicator/model_artifacts/.data_cache")
        if not cache_dir.exists():
            return {"exists": False}
        files = {}
        for f in sorted(cache_dir.glob("*.parquet")):
            try:
                stat = f.stat()
                files[f.stem] = {
                    "size_kb": round(stat.st_size / 1024, 1),
                    "age_hours": round((datetime.now().timestamp() - stat.st_mtime) / 3600, 1),
                }
            except Exception:
                files[f.stem] = {"error": "stat failed"}
        return {"exists": True, "files": files}

    def _check_env(self) -> dict:
        checks = {}
        for var in ["COINGLASS_API_KEY", "INDICATOR_BOT_TOKEN", "INDICATOR_CHAT_ID",
                     "AGENT_API_KEY", "MYSQLHOST", "MYSQL_HOST"]:
            checks[var] = bool(os.environ.get(var))
        checks["is_railway"] = bool(os.environ.get("RAILWAY_ENVIRONMENT"))
        return checks


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = DataCollectorAgent()
    if "--context-only" in sys.argv:
        # Quick mode: just run each tool once and print
        for tool in agent.get_tools():
            print(f"\n--- {tool['name']} ---")
            result = agent.execute_tool(tool["name"], {})
            print(result)
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Tools called: {result.tool_calls_made}")
        print(f"Turns: {result.turns_used} | Duration: {result.duration_s:.1f}s")
        if result.diagnosis.get("findings"):
            for f in result.diagnosis["findings"]:
                print(f"  [{f.get('severity')}] {f.get('title')}: {f.get('detail')}")
