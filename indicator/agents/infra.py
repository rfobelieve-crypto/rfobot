"""
Infra Agent — AI-powered infrastructure health monitor.

Monitors: MySQL database, Railway deployment, scheduler health,
disk/memory usage patterns, and system-level anomalies that
domain-specific agents won't catch.
"""
from __future__ import annotations

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

## System Architecture
- **Platform**: Railway (PaaS), auto-deploy on git push to main
- **Web server**: Gunicorn + Flask, single process
- **Scheduler**: APScheduler (BackgroundScheduler) running update_cycle every hour at :02
- **Database**: MySQL 8.0 on Railway (shared instance)
- **Storage**: Parquet files for local cache, MySQL for persistent history

## What You Monitor

### 1. Database Health
- Connection pool status (pool size, available connections)
- Query latency: simple SELECT should be <100ms on Railway internal network
- Table sizes: indicator_history grows ~24 rows/day, tracked_signals ~2-10/day
- Connection errors: "Too many connections" = pool leak or Railway DB overload
- Deadlocks or long-running queries blocking the update cycle

### 2. Scheduler Health
- Is the scheduler actually firing? Check last update timestamp.
- If last update is >2h old → scheduler might be dead (Railway restart, OOM kill)
- Misfire: if update takes >60min, next cron trigger gets skipped (misfire_grace_time=300s)
- Double-fire: APScheduler shouldn't run concurrent update_cycles

### 3. Application State
- In-memory indicator_df: how many bars? Growing normally (~24/day)?
- Engine loaded: dual model should be loaded, not falling back to legacy
- Pred history: should have 300+ entries for mag_score warmup

### 4. Disk & Storage
- Parquet files: indicator_history.parquet shouldn't grow unboundedly
- Cache files: .data_cache/ should be refreshed regularly
- Snapshot files: .snapshots/ accumulate over time

### 5. Railway-Specific Issues
- Railway restarts: after restart, pred_history reloads from training_stats.json
- Memory: Railway free tier has limits. If indicator_df grows too large → OOM
- Environment variables: all required vars must be set (DB, API keys, Telegram)

## Your Judgment
- DB latency <50ms = healthy, 50-200ms = acceptable, >500ms = investigate
- Scheduler gap >2h = critical (predictions are stale, users see old data)
- Missing env vars = critical if they affect core functions
- Storage issues are usually warning-level unless disk is nearly full
- Always check if issues are transient (just restarted) vs persistent

Respond in Traditional Chinese for summary/report."""

    def collect_context(self) -> dict:
        context = {}

        context["database"] = self._probe_database()
        context["scheduler"] = self._probe_scheduler()
        context["app_state"] = self._probe_app_state()
        context["storage"] = self._probe_storage()
        context["environment"] = self._probe_environment()

        return context

    def get_available_actions(self) -> list[dict]:
        return [
            {"name": "ALERT", "description": "Send Telegram alert about infrastructure issue"},
            {"name": "LOG", "description": "Log finding"},
            {"name": "NONE", "description": "No action"},
        ]

    def _probe_database(self) -> dict:
        """Test MySQL connectivity and performance."""
        result = {}

        try:
            from shared.db import get_db_conn

            # Connection test + latency
            t0 = time.time()
            conn = get_db_conn()
            connect_time = time.time() - t0

            with conn.cursor() as cur:
                # Simple ping
                t0 = time.time()
                cur.execute("SELECT 1")
                ping_ms = (time.time() - t0) * 1000

                # Table sizes
                t0 = time.time()
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
                table_query_ms = (time.time() - t0) * 1000

                # Process list (active connections)
                cur.execute("SHOW PROCESSLIST")
                processes = cur.fetchall()

                # Check for long-running queries
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

            result = {
                "connected": True,
                "connect_time_ms": round(connect_time * 1000, 1),
                "ping_ms": round(ping_ms, 1),
                "table_query_ms": round(table_query_ms, 1),
                "tables": tables,
                "active_connections": len(processes),
                "long_running_queries": long_queries,
            }

        except Exception as e:
            result = {"connected": False, "error": str(e)}

        return result

    def _probe_scheduler(self) -> dict:
        """Check if the scheduler is running and firing on time."""
        result = {}

        try:
            from shared.db import get_db_conn
            conn = get_db_conn()
            with conn.cursor() as cur:
                # Check last N update timestamps
                cur.execute("""
                    SELECT dt FROM indicator_history
                    ORDER BY dt DESC LIMIT 10
                """)
                recent = [r["dt"] for r in cur.fetchall()]
            conn.close()

            if recent:
                latest = recent[0]
                latest_utc = latest.replace(tzinfo=timezone.utc) if hasattr(latest, 'replace') else latest
                age_min = (datetime.now(timezone.utc) - latest_utc).total_seconds() / 60

                # Check gaps between consecutive bars
                gaps = []
                for i in range(1, len(recent)):
                    diff_min = (recent[i-1] - recent[i]).total_seconds() / 60
                    gaps.append(round(diff_min, 1))

                result = {
                    "last_update": str(latest),
                    "age_minutes": round(age_min, 1),
                    "recent_gaps_minutes": gaps,
                    "avg_gap_minutes": round(sum(gaps) / len(gaps), 1) if gaps else None,
                    "max_gap_minutes": max(gaps) if gaps else None,
                    "bars_count": len(recent),
                }
            else:
                result = {"last_update": None, "error": "No bars in indicator_history"}

        except Exception as e:
            result = {"error": str(e)}

        return result

    def _probe_app_state(self) -> dict:
        """Check in-memory application state."""
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

    def _probe_storage(self) -> dict:
        """Check disk usage for key directories."""
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

        # Parquet history file specifically
        hist_path = Path("indicator/model_artifacts/indicator_history.parquet")
        if hist_path.exists():
            result["indicator_history_parquet"] = {
                "size_mb": round(hist_path.stat().st_size / 1024 / 1024, 2),
                "age_hours": round((datetime.now().timestamp() - hist_path.stat().st_mtime) / 3600, 1),
            }

        return result

    def _probe_environment(self) -> dict:
        """Check required environment variables."""
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


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json as _json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    agent = InfraAgent()
    if "--context-only" in sys.argv:
        ctx = agent.collect_context()
        print(_json.dumps(ctx, indent=2, default=str, ensure_ascii=False))
    else:
        result = agent.run()
        print(f"\nAgent: {result.agent_name} | Status: {result.status}")
        print(f"Summary: {result.summary}")
        print(f"\nDiagnosis:\n{_json.dumps(result.diagnosis, indent=2, ensure_ascii=False)}")
