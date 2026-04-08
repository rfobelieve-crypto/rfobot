"""
System Health Monitor — centralized health checking for all components.

Checks:
  1. Data freshness: Binance klines, CG endpoints, snapshots
  2. Feature quality: NaN rates per feature group
  3. Prediction health: WARMUP status, pred_history depth
  4. DB connectivity: MySQL read/write
  5. Silent failures: modules that failed without crashing the cycle

Each check returns a status dict with: ok (bool), detail (str), severity (info/warn/critical)
The aggregate health dict powers the dashboard and Telegram alerts.

Usage:
    from indicator.health_monitor import HealthMonitor
    hm = HealthMonitor()
    report = hm.check_all(state, features_df, cg_status)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TZ8 = timezone(timedelta(hours=8))


class HealthMonitor:
    """Centralized health checker for the indicator system."""

    def __init__(
        self,
        max_data_age_min: int = 90,       # API 數據超過 90 分鐘算過時
        nan_warn_threshold: float = 0.3,   # 特徵 NaN 率 > 30% 警告
        nan_critical_threshold: float = 0.7,  # > 70% 嚴重
        min_pred_history: int = 30,        # pred_history 最少 30 bars 才正常
    ):
        self.max_data_age_min = max_data_age_min
        self.nan_warn = nan_warn_threshold
        self.nan_critical = nan_critical_threshold
        self.min_pred_history = min_pred_history

    def check_all(
        self,
        state: dict,
        features_df: pd.DataFrame | None = None,
        cg_status: dict | None = None,
        engine=None,
    ) -> dict:
        """
        Run all health checks.

        Returns:
            dict with keys: overall_status ("healthy"/"degraded"/"critical"),
                           checks (list of check results),
                           summary (text),
                           timestamp
        """
        checks = []

        # 1. Data freshness
        checks.append(self._check_data_freshness(state))

        # 2. CG endpoints
        checks.append(self._check_cg_endpoints(cg_status))

        # 3. Feature NaN rates
        if features_df is not None and not features_df.empty:
            checks.append(self._check_feature_nans(features_df))

        # 4. WARMUP / pred_history
        checks.append(self._check_warmup(state, engine))

        # 5. DB connectivity
        checks.append(self._check_db())

        # 6. Silent failures (from state error log)
        checks.append(self._check_silent_failures(state))

        # Aggregate
        severities = [c["severity"] for c in checks]
        if "critical" in severities:
            overall = "critical"
        elif "warn" in severities:
            overall = "degraded"
        else:
            overall = "healthy"

        # Alert messages (warn + critical only)
        alerts = [c for c in checks if c["severity"] in ("warn", "critical")]

        return {
            "overall_status": overall,
            "checks": checks,
            "alerts": alerts,
            "timestamp": datetime.now(TZ8).strftime("%Y-%m-%d %H:%M UTC+8"),
        }

    # ─── Individual checks ──────────────────────────────────────────

    def _check_data_freshness(self, state: dict) -> dict:
        """檢查最後更新時間是否過舊"""
        last_update = state.get("last_update")
        if not last_update:
            return {"name": "Data Freshness", "ok": False,
                    "severity": "warn", "detail": "No update recorded yet"}

        try:
            if isinstance(last_update, str):
                # ISO format
                last_dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
            else:
                last_dt = last_update

            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)

            age_min = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60

            if age_min > self.max_data_age_min * 2:
                return {"name": "Data Freshness", "ok": False, "severity": "critical",
                        "detail": f"Last update {age_min:.0f}min ago (>{self.max_data_age_min*2}min)"}
            elif age_min > self.max_data_age_min:
                return {"name": "Data Freshness", "ok": False, "severity": "warn",
                        "detail": f"Last update {age_min:.0f}min ago (>{self.max_data_age_min}min)"}
            else:
                return {"name": "Data Freshness", "ok": True, "severity": "info",
                        "detail": f"{age_min:.0f}min ago"}
        except Exception as e:
            return {"name": "Data Freshness", "ok": False, "severity": "warn",
                    "detail": f"Parse error: {e}"}

    def _check_cg_endpoints(self, cg_status: dict | None) -> dict:
        """檢查 Coinglass 各端點是否正常回傳數據"""
        if not cg_status or not isinstance(cg_status, dict):
            return {"name": "Coinglass", "ok": False, "severity": "warn",
                    "detail": "No CG status available"}

        total = len(cg_status)
        failed = [k for k, v in cg_status.items()
                  if isinstance(v, dict) and v.get("empty", True)]
        ok_count = total - len(failed)

        if len(failed) == 0:
            return {"name": "Coinglass", "ok": True, "severity": "info",
                    "detail": f"{ok_count}/{total} OK"}
        elif len(failed) <= 3:
            return {"name": "Coinglass", "ok": True, "severity": "warn",
                    "detail": f"{ok_count}/{total} OK, failed: {', '.join(failed)}"}
        else:
            return {"name": "Coinglass", "ok": False, "severity": "critical",
                    "detail": f"Only {ok_count}/{total} OK, failed: {', '.join(failed[:5])}"}

    def _check_feature_nans(self, features_df: pd.DataFrame) -> dict:
        """檢查特徵的 NaN 率，按群組分類"""
        last_row = features_df.iloc[-1]
        numeric_cols = [c for c in features_df.columns
                        if features_df[c].dtype in ("float64", "float32", "int64")]

        if not numeric_cols:
            return {"name": "Feature Quality", "ok": True, "severity": "info",
                    "detail": "No numeric features"}

        # NaN rate for last row (most important — this is what model sees)
        nan_count = sum(1 for c in numeric_cols if pd.isna(last_row.get(c)))
        nan_rate = nan_count / len(numeric_cols)

        # NaN rate for last 10 rows (trend)
        recent = features_df[numeric_cols].tail(10)
        rolling_nan_rate = recent.isna().mean().mean()

        # Group-level NaN check
        groups = {
            "kline": [c for c in numeric_cols if c.startswith(("log_return", "realized", "return_", "taker_delta", "volume_ma", "vol_"))],
            "coinglass": [c for c in numeric_cols if c.startswith("cg_")],
            "toxicity": [c for c in numeric_cols if c.startswith("tox_")],
            "custom": [c for c in numeric_cols if c.startswith(("impact", "absorption", "fragility", "post_absorb", "flow_trend"))],
        }

        bad_groups = []
        for gname, gcols in groups.items():
            if not gcols:
                continue
            g_nan = sum(1 for c in gcols if pd.isna(last_row.get(c))) / len(gcols)
            if g_nan > self.nan_critical:
                bad_groups.append(f"{gname}({g_nan:.0%})")

        if nan_rate > self.nan_critical:
            return {"name": "Feature Quality", "ok": False, "severity": "critical",
                    "detail": f"NaN rate {nan_rate:.0%} ({nan_count}/{len(numeric_cols)}). Bad groups: {', '.join(bad_groups) or 'spread across'}",
                    "nan_rate": nan_rate, "bad_groups": bad_groups}
        elif nan_rate > self.nan_warn:
            return {"name": "Feature Quality", "ok": False, "severity": "warn",
                    "detail": f"NaN rate {nan_rate:.0%} ({nan_count}/{len(numeric_cols)})",
                    "nan_rate": nan_rate}
        else:
            return {"name": "Feature Quality", "ok": True, "severity": "info",
                    "detail": f"NaN rate {nan_rate:.0%} ({nan_count}/{len(numeric_cols)})",
                    "nan_rate": nan_rate}

    def _check_warmup(self, state: dict, engine=None) -> dict:
        """檢查 pred_history 是否足夠（避免 WARMUP 假信號）"""
        pred = state.get("last_prediction", {})
        regime = pred.get("regime", "") if isinstance(pred, dict) else ""

        # Check pred_history depth from engine
        history_depth = 0
        if engine and hasattr(engine, 'pred_history'):
            history_depth = len(engine.pred_history)
        elif engine and hasattr(engine, '_pred_history'):
            history_depth = len(engine._pred_history)

        if regime == "WARMUP" or (isinstance(regime, str) and "WARMUP" in regime):
            return {"name": "WARMUP Status", "ok": False, "severity": "warn",
                    "detail": f"In WARMUP (history={history_depth} bars, need {self.min_pred_history}). Confidence unreliable.",
                    "history_depth": history_depth}
        elif history_depth > 0 and history_depth < self.min_pred_history:
            return {"name": "WARMUP Status", "ok": False, "severity": "warn",
                    "detail": f"pred_history={history_depth} bars (< {self.min_pred_history}). Exited WARMUP but data thin.",
                    "history_depth": history_depth}
        else:
            return {"name": "WARMUP Status", "ok": True, "severity": "info",
                    "detail": f"OK (history={history_depth} bars)" if history_depth > 0 else "OK",
                    "history_depth": history_depth}

    def _check_db(self) -> dict:
        """檢查 MySQL 連線和數據寫入"""
        try:
            from shared.db import get_db_conn
            conn = get_db_conn()
            with conn.cursor() as cur:
                # Check indicator_history freshness
                cur.execute("SELECT MAX(dt) as last_dt, COUNT(*) as n FROM indicator_history")
                r = cur.fetchone()
                n = int(r["n"] or 0)
                last_dt = r["last_dt"]
            conn.close()

            if n == 0:
                return {"name": "Database", "ok": False, "severity": "warn",
                        "detail": "indicator_history is empty"}

            if last_dt:
                age_min = (datetime.now(timezone.utc) - last_dt.replace(tzinfo=timezone.utc)).total_seconds() / 60
                if age_min > 120:
                    return {"name": "Database", "ok": False, "severity": "warn",
                            "detail": f"Last write {age_min:.0f}min ago, {n} rows"}

            return {"name": "Database", "ok": True, "severity": "info",
                    "detail": f"OK ({n} rows)"}
        except Exception as e:
            return {"name": "Database", "ok": False, "severity": "critical",
                    "detail": f"Connection failed: {str(e)[:60]}"}

    def _check_silent_failures(self, state: dict) -> dict:
        """檢查是否有模組靜默失敗"""
        error = state.get("error")
        if error:
            return {"name": "Silent Failures", "ok": False, "severity": "warn",
                    "detail": str(error)[:100]}
        return {"name": "Silent Failures", "ok": True, "severity": "info",
                "detail": "None detected"}

    # ─── Feature NaN guard (called from update_cycle) ───────────────

    @staticmethod
    def nan_guard(features_df: pd.DataFrame, threshold: float = 0.5) -> tuple[bool, str]:
        """
        Check if features are safe for prediction.
        Call after build_live_features, before predict.

        Returns:
            (safe: bool, message: str)
        """
        if features_df is None or features_df.empty:
            return False, "Features DataFrame is empty"

        last_row = features_df.iloc[-1]
        numeric = [c for c in features_df.columns
                   if features_df[c].dtype in ("float64", "float32", "int64")]

        if not numeric:
            return True, "No numeric features to check"

        nan_count = sum(1 for c in numeric if pd.isna(last_row.get(c)))
        nan_rate = nan_count / len(numeric)

        if nan_rate > threshold:
            return False, f"NaN rate {nan_rate:.0%} ({nan_count}/{len(numeric)}) exceeds {threshold:.0%} threshold"
        return True, f"OK (NaN rate {nan_rate:.0%})"
