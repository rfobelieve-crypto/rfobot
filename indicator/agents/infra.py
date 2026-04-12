"""
Infra Agent — Tool-Use Architecture.

Claude gets tools to probe infrastructure health: MySQL database,
scheduler, app state, disk storage, and environment. It decides
what to investigate based on what it finds.
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


class InfraAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "Infra"

    @property
    def system_prompt(self) -> str:
        return """\
You are a senior infrastructure engineer monitoring a Railway-deployed \
Python application that runs a BTC prediction indicator.

## Investigation Protocol
1. Check database health first (most critical — if DB is down, nothing works)
2. Check scheduler: is it firing on time? Any gaps?
3. Check app state: is the engine loaded? Is indicator_df growing?
4. Check storage: any unbounded growth or stale caches?
5. Check environment: are all required secrets configured?
6. Conclude

## System Architecture
- Platform: Railway (PaaS), auto-deploy on git push to main
- Web server: Gunicorn + Flask, single process
- Scheduler: APScheduler running update_cycle every hour at :02
- Database: MySQL 8.0 on Railway (shared instance)
- Storage: Parquet files for local cache, MySQL for persistent history

## Thresholds (use judgment)
- DB ping <50ms = healthy, 50-200ms = acceptable, >500ms = investigate
- Scheduler gap >2h = critical (predictions are stale)
- Scheduler gap 1-2h = might be a single misfire, check if recovered
- indicator_df should grow ~24 rows/day
- pred_history should have 300+ entries for mag_score warmup
- Missing DB or Telegram env vars = critical
- Missing agent API key = warning (agents can't run but system works)
- Storage: parquet >100MB or cache >24h stale = warning

## Railway-Specific Issues
- After restart, pred_history reloads from training_stats.json
- Memory: if indicator_df grows too large → OOM kill → restart loop
- Environment variables: Railway env vars take priority over .env

## Self-Healing Actions (use repair tools when criteria are met)
- Scheduler gap >2h (last update too old) → `repair_trigger_update`
- active_connections > 20 or connection errors → `repair_kill_idle_connections`
- Always verify: after triggering update, check if indicator_history got a new row.
- Do NOT trigger update if the last update is <90min old (normal cycle).

Respond in Traditional Chinese."""

    def get_tools(self) -> list[dict]:
        return [
            {
                "name": "check_database",
                "description": "Test MySQL connectivity, query latency, table sizes, active connections, and long-running queries.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_scheduler",
                "description": "Check if the hourly scheduler is firing on time by examining recent indicator_history timestamps and gaps.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_app_state",
                "description": "Check in-memory application state: indicator_df size, engine mode, pred_history count, last prediction.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_storage",
                "description": "Check disk usage for model artifacts, data cache, snapshots, and parquet history files.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
            {
                "name": "check_environment",
                "description": "Check if required environment variables are set: DB, API keys, Telegram, Railway.",
                "input_schema": {"type": "object", "properties": {}, "required": []},
            },
        ]

    def execute_tool(self, tool_name: str, tool_input: dict) -> str:
        handlers = {
            "check_database": self._check_database,
            "check_scheduler": self._check_scheduler,
            "check_app_state": self._check_app_state,
            "check_storage": self._check_storage,
            "check_environment": self._check_environment,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            return json.dumps(handler(), default=str, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _check_database(self) -> dict:
        from shared.db import get_db_conn

        t0 = time.time()
        conn = get_db_conn()
        connect_time = time.time() - t0

        with conn.cursor() as cur:
            # Ping latency
            t0 = time.time()
            cur.execute("SELECT 1")
            ping_ms = (time.time() - t0) * 1000

            # Table sizes
            cur.execute("""
                SELECT table_name, table_rows,
                       ROUND(data_length / 1024 / 1024, 2) as data_mb,
                       ROUND(index_length / 1024 / 1024, 2) as index_mb
                FROM information_schema.tables
                WHERE table_schema = DATABASE()
                ORDER BY data_length DESC
            """)
            tables = {}
            for r in cur.fetchall():
                tables[r["table_name"]] = {
                    "rows": int(r["table_rows"] or 0),
                    "data_mb": float(r["data_mb"] or 0),
                    "index_mb": float(r["index_mb"] or 0),
                }

            # Active connections
            cur.execute("SHOW PROCESSLIST")
            processes = cur.fetchall()
            long_queries = []
            for p in processes:
                if p.get("Time") and int(p["Time"]) > 30:
                    long_queries.append({
                        "id": p.get("Id"),
                        "time_s": int(p["Time"]),
                        "state": p.get("State", ""),
                        "info": str(p.get("Info", ""))[:100],
                    })

        conn.close()

        return {
            "connected": True,
            "connect_time_ms": round(connect_time * 1000, 1),
            "ping_ms": round(ping_ms, 1),
            "tables": tables,
            "active_connections": len(processes),
            "long_running_queries": long_queries,
        }

    def _check_scheduler(self) -> dict:
        from shared.db import get_db_conn
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT dt FROM indicator_history
                ORDER BY dt DESC LIMIT 10
            """)
            recent = [r["dt"] for r in cur.fetchall()]
        conn.close()

        if not recent:
            return {"last_update": None, "error": "No bars in indicator_history"}

        latest = recent[0]
        latest_utc = latest.replace(tzinfo=timezone.utc) if hasattr(latest, 'replace') else latest
        age_min = (datetime.now(timezone.utc) - latest_utc).total_seconds() / 60

        gaps = []
        for i in range(1, len(recent)):
            diff_min = (recent[i-1] - recent[i]).total_seconds() / 60
            gaps.append(round(diff_min, 1))

        return {
            "last_update": str(latest),
            "age_minutes": round(age_min, 1),
            "recent_gaps_minutes": gaps,
            "avg_gap_minutes": round(sum(gaps) / len(gaps), 1) if gaps else None,
            "max_gap_minutes": max(gaps) if gaps else None,
            "bars_count": len(recent),
        }

    def _check_app_state(self) -> dict:
        result = {}
        try:
            from indicator.app import _state, _engine, _lock

            with _lock:
                result["last_update"] = _state.get("last_update")
                result["status"] = _state.get("status")

                indicator_df = _state.get("indicator_df")
                if indicator_df is not None and hasattr(indicator_df, '__len__'):
                    result["indicator_df_rows"] = len(indicator_df)
                    result["indicator_df_columns"] = len(indicator_df.columns) if hasattr(indicator_df, 'columns') else 0
                else:
                    result["indicator_df_rows"] = 0

                last_pred = _state.get("last_prediction", {})
                result["last_prediction"] = {
                    "time": last_pred.get("time"),
                    "direction": last_pred.get("direction"),
                    "confidence": last_pred.get("confidence"),
                    "regime": last_pred.get("regime"),
                }

            if _engine:
                result["engine"] = {
                    "mode": _engine.mode,
                    "pred_history_count": len(_engine.pred_history) if hasattr(_engine, 'pred_history') else 0,
                    "has_dual_dir": hasattr(_engine, 'dual_dir_model'),
                    "has_dual_mag": hasattr(_engine, 'dual_mag_model'),
                }
            else:
                result["engine"] = {"loaded": False}

        except ImportError:
            result["note"] = "App not running (import failed — running standalone)"
        except Exception as e:
            result["error"] = str(e)

        return result

    def _check_storage(self) -> dict:
        result = {}

        dirs_to_check = {
            "model_artifacts": Path("indicator/model_artifacts"),
            "data_cache": Path("indicator/model_artifacts/.data_cache"),
            "snapshots": Path("indicator/model_artifacts/.snapshots"),
        }

        for name, path in dirs_to_check.items():
            if not path.exists():
                result[name] = {"exists": False}
                continue

            total_size = 0
            file_count = 0
            largest_file = None
            largest_size = 0

            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        size = f.stat().st_size
                        total_size += size
                        file_count += 1
                        if size > largest_size:
                            largest_size = size
                            largest_file = str(f.name)
                    except Exception:
                        pass

            result[name] = {
                "exists": True,
                "total_mb": round(total_size / 1024 / 1024, 2),
                "file_count": file_count,
                "largest_file": largest_file,
                "largest_mb": round(largest_size / 1024 / 1024, 2),
            }

        # Parquet history file
        hist_path = Path("indicator/model_artifacts/indicator_history.parquet")
        if hist_path.exists():
            result["indicator_history_parquet"] = {
                "size_mb": round(hist_path.stat().st_size / 1024 / 1024, 2),
                "age_hours": round((datetime.now().timestamp() - hist_path.stat().st_mtime) / 3600, 1),
            }

        return result

    def _check_environment(self) -> dict:
        required_vars = {
            "core_db": ["MYSQLHOST", "MYSQL_HOST"],
            "coinglass": ["COINGLASS_API_KEY"],
            "telegram": ["INDICATOR_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"],
            "agent": ["AGENT_API_KEY"],
        }

        result = {}
        for category, var_names in required_vars.items():
            found = any(os.environ.get(v) for v in var_names)
            result[category] = {
                "configured": found,
                "checked_vars": var_names,
            }

        result["railway"] = {
            "is_railway": bool(os.environ.get("RAILWAY_ENVIRONMENT")),
            "environment": os.environ.get("RAILWAY_ENVIRONMENT", "local"),
        }

        return result


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = InfraAgent()
    if "--context-only" in sys.argv:
        for tool in agent.get_tools():
            print(f"\n--- {tool['name']} ---")
            print(agent.execute_tool(tool["name"], {}))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"Tools: {result.tool_calls_made} | Turns: {result.turns_used}")
