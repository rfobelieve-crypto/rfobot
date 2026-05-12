"""
Hybrid v9 + LDC inference engine for production (Path 1, Block 2).

At each hourly K-line close:
  1. Load v9 binary classifiers (long_win / short_win) from
     model_artifacts/dual_model/v9_winrate/
  2. Compute LDC signal via research.ldc_python.compute_ldc_signals
     on the last 2000+ klines (full recompute is acceptable hourly cost
     ~5-10 seconds).
  3. Apply MUST-AGREE rule:
        SHORT trigger: p_short_win >= 0.65 AND ldc_signal == -1
        LONG  trigger: p_long_win  >= 0.65 AND ldc_signal == +1
  4. Record qualifying trigger to MySQL `hybrid_signals` table (independent
     from existing tracked_signals).  Outcome fields filled in later by
     backfill_hybrid_outcomes() once 8h horizon completes.

DESIGN:
  - This is INDEPENDENT of the v7 indicator pipeline.  The existing
    v7 Telegram alerts, /chart, /perf, tracked_signals all keep working.
  - hybrid_signals is its own table, its own paper PnL accounting.
  - Block 3 (Path1-B3) will expose hybrid cohort in /paper-perf.
  - Block 4 will monitor forward window net_bps and auto-pause on regression.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V9_DIR = PROJECT_ROOT / "indicator" / "model_artifacts" / "dual_model" / "v9_winrate"

P_LONG_THRESHOLD = 0.65
P_SHORT_THRESHOLD = 0.65
LDC_MAX_BARS = 2200  # enough for LDC's max_bars_back=2000 + warmup
HORIZON_HOURS = 8
TP_DIST = 0.005
SL_DIST = 0.003


class HybridInferenceEngine:
    """v9 + LDC must-agree hybrid signal generator."""

    def __init__(self) -> None:
        self.long_model: Optional[xgb.XGBClassifier] = None
        self.short_model: Optional[xgb.XGBClassifier] = None
        self.feature_cols: list[str] = []
        self.meta: dict = {}
        self.model_version: str = "unknown"
        self._mtime: float = 0.0
        self.load_models()

    def load_models(self) -> bool:
        """Load (or reload) v9 models from artifacts directory.
        Auto-reload when meta.json mtime changes."""
        meta_path = V9_DIR / "meta.json"
        long_path = V9_DIR / "long_xgb.json"
        short_path = V9_DIR / "short_xgb.json"
        cols_path = V9_DIR / "feature_cols.json"

        if not all(p.exists() for p in (meta_path, long_path, short_path, cols_path)):
            logger.warning("Hybrid inference: v9 artifacts not found in %s", V9_DIR)
            return False

        mtime = meta_path.stat().st_mtime
        if self.long_model is not None and abs(mtime - self._mtime) < 0.001:
            return True  # already loaded

        with open(meta_path) as f:
            self.meta = json.load(f)
        with open(cols_path) as f:
            self.feature_cols = json.load(f)
        self.long_model = xgb.XGBClassifier()
        self.long_model.load_model(str(long_path))
        self.short_model = xgb.XGBClassifier()
        self.short_model.load_model(str(short_path))
        self.model_version = self.meta.get("version", "unknown")
        self._mtime = mtime
        logger.info("Hybrid inference: loaded v9 model %s (CV AUC long=%.3f short=%.3f)",
                    self.model_version,
                    self.meta.get("median_auc_long", -1),
                    self.meta.get("median_auc_short", -1))
        return True

    def predict_v9(self, features: pd.DataFrame) -> tuple[float, float]:
        """Predict (p_long_win, p_short_win) for the LAST row of features."""
        self.load_models()
        if self.long_model is None or self.short_model is None:
            return float("nan"), float("nan")
        missing = [c for c in self.feature_cols if c not in features.columns]
        if missing:
            logger.warning("Hybrid inference: %d feature columns missing: %s",
                            len(missing), missing[:5])
        feat_cols = [c for c in self.feature_cols if c in features.columns]
        x = features[feat_cols].iloc[[-1]].values.astype(np.float32)
        p_long = float(self.long_model.predict_proba(x)[0, 1])
        p_short = float(self.short_model.predict_proba(x)[0, 1])
        return p_long, p_short

    def compute_ldc_latest(self, klines: pd.DataFrame) -> dict:
        """Compute LDC signals on klines tail and return dict for last bar.

        Returns dict: {signal, is_bullish_kernel, is_bearish_kernel,
                       ema_uptrend, ema_downtrend, start_long, start_short, ts}
        """
        try:
            from research.ldc_python import compute_ldc_signals
        except Exception as exc:
            logger.warning("Hybrid inference: cannot import ldc_python: %s", exc)
            return {}

        tail = klines.tail(LDC_MAX_BARS)
        if len(tail) < 250:  # LDC needs 200 bars EMA warmup
            logger.warning("Hybrid inference: not enough klines for LDC (have %d need >=250)",
                            len(tail))
            return {}

        try:
            ldc = compute_ldc_signals(tail)
        except Exception as exc:
            logger.exception("Hybrid inference: LDC compute failed: %s", exc)
            return {}

        last = ldc.iloc[-1]
        return {
            "ts": ldc.index[-1],
            "signal": int(last["signal"]),
            "is_bullish_kernel": bool(last["is_bullish_kernel"]),
            "is_bearish_kernel": bool(last["is_bearish_kernel"]),
            "ema_uptrend": bool(last["ema_uptrend"]),
            "ema_downtrend": bool(last["ema_downtrend"]),
            "start_long": bool(last["start_long_trade"]),
            "start_short": bool(last["start_short_trade"]),
        }

    def maybe_emit_signal(self, features: pd.DataFrame,
                           klines: pd.DataFrame) -> Optional[dict]:
        """Run full hybrid inference for latest bar.

        Returns a dict if a hybrid signal triggers (v9 + LDC must agree),
        None otherwise.  Caller is responsible for persisting to DB.
        """
        if features.empty or klines.empty:
            return None

        p_long, p_short = self.predict_v9(features)
        if not (np.isfinite(p_long) and np.isfinite(p_short)):
            return None

        ldc = self.compute_ldc_latest(klines)
        if not ldc:
            return None

        bar_ts = features.index[-1]
        last_close = float(klines["close"].iloc[-1])

        ldc_sig = ldc["signal"]
        short_trigger = (p_short >= P_SHORT_THRESHOLD) and (ldc_sig == -1)
        long_trigger = (p_long >= P_LONG_THRESHOLD) and (ldc_sig == +1)

        if not (short_trigger or long_trigger):
            return None

        direction = "SHORT" if short_trigger else "LONG"
        return {
            "signal_time": bar_ts,
            "direction": direction,
            "p_long_win": p_long,
            "p_short_win": p_short,
            "ldc_signal": ldc_sig,
            "ldc_kernel_bear": ldc["is_bearish_kernel"],
            "ldc_kernel_bull": ldc["is_bullish_kernel"],
            "ldc_ema_up": ldc["ema_uptrend"],
            "ldc_ema_dn": ldc["ema_downtrend"],
            "entry_price": last_close,
            "model_version": self.model_version,
            "tp_dist": TP_DIST,
            "sl_dist": SL_DIST,
            "horizon_h": HORIZON_HOURS,
        }


# ── DB schema + persistence ──────────────────────────────────────────────────

TABLE = "hybrid_signals"


def _get_conn():
    from shared.db import get_db_conn
    return get_db_conn()


def init_hybrid_signals_table() -> None:
    """Idempotently create hybrid_signals table."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{TABLE}` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `signal_time` DATETIME NOT NULL,
                    `direction` VARCHAR(8) NOT NULL,
                    `p_long_win` DOUBLE NOT NULL,
                    `p_short_win` DOUBLE NOT NULL,
                    `ldc_signal` INT NOT NULL,
                    `ldc_kernel_bear` TINYINT NOT NULL,
                    `ldc_kernel_bull` TINYINT NOT NULL,
                    `ldc_ema_up` TINYINT NOT NULL,
                    `ldc_ema_dn` TINYINT NOT NULL,
                    `entry_price` DOUBLE NOT NULL,
                    `model_version` VARCHAR(48) DEFAULT NULL,
                    `tp_dist` DOUBLE NOT NULL,
                    `sl_dist` DOUBLE NOT NULL,
                    `horizon_h` INT NOT NULL,
                    `exit_price` DOUBLE DEFAULT NULL,
                    `exit_reason` VARCHAR(16) DEFAULT NULL,
                    `bars_held` INT DEFAULT NULL,
                    `gross_pct` DOUBLE DEFAULT NULL,
                    `net_pct_maker` DOUBLE DEFAULT NULL,
                    `win` TINYINT DEFAULT NULL,
                    `paused_at_signal` TINYINT DEFAULT 0,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY `uq_time_dir` (`signal_time`, `direction`),
                    KEY `ix_signal_time` (`signal_time`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


def record_hybrid_signal(sig: dict, paused: bool = False) -> bool:
    """Persist a hybrid signal.  `paused=True` records but flags it as
    "would not have traded" (Block 4 kill switch can set this)."""
    ts = sig["signal_time"]
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc).replace(tzinfo=None)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT IGNORE INTO `{TABLE}`
                  (signal_time, direction, p_long_win, p_short_win,
                   ldc_signal, ldc_kernel_bear, ldc_kernel_bull,
                   ldc_ema_up, ldc_ema_dn,
                   entry_price, model_version, tp_dist, sl_dist, horizon_h,
                   paused_at_signal)
                VALUES
                  (%s, %s, %s, %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s)
            """, (
                ts.strftime("%Y-%m-%d %H:%M:%S"), sig["direction"],
                sig["p_long_win"], sig["p_short_win"],
                sig["ldc_signal"], int(sig["ldc_kernel_bear"]),
                int(sig["ldc_kernel_bull"]), int(sig["ldc_ema_up"]),
                int(sig["ldc_ema_dn"]),
                sig["entry_price"], sig.get("model_version", "unknown"),
                sig["tp_dist"], sig["sl_dist"], sig["horizon_h"],
                int(paused),
            ))
            inserted = cur.rowcount > 0
        conn.commit()
    finally:
        conn.close()
    if inserted:
        logger.info("Hybrid signal recorded: %s %s @ %.2f (P_long=%.3f P_short=%.3f LDC=%d)%s",
                     ts, sig["direction"], sig["entry_price"],
                     sig["p_long_win"], sig["p_short_win"], sig["ldc_signal"],
                     " [PAUSED]" if paused else "")
    return inserted


def backfill_hybrid_outcomes(klines: pd.DataFrame, maker_cost: float = 0.0005) -> int:
    """For each open hybrid_signal (exit_price IS NULL) whose horizon has
    elapsed, find first TP/SL hit in next H bars and fill outcome columns.

    Returns number of rows updated.
    """
    from research.paper_trading_tpsl import _find_exit_for_signal
    if klines.empty:
        return 0
    klines_idx = klines.index
    if klines.index.tz is not None:
        klines = klines.copy()
        klines.index = klines.index.tz_convert("UTC").tz_localize(None)
        klines_idx = klines.index

    conn = _get_conn()
    updated = 0
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT id, signal_time, direction, entry_price, tp_dist, sl_dist, horizon_h
                FROM `{TABLE}`
                WHERE exit_price IS NULL
                ORDER BY signal_time ASC
            """)
            rows = cur.fetchall()

        for row in rows:
            sig_ts = pd.Timestamp(row["signal_time"])
            horizon = int(row["horizon_h"])
            future_mask = klines_idx > sig_ts
            future = klines.loc[future_mask]
            if len(future) < horizon:
                continue  # not enough future bars yet

            entry = float(row["entry_price"])
            sl_dist = float(row["sl_dist"])
            tp_dist = float(row["tp_dist"])
            rr = tp_dist / sl_dist
            dir_for_exit = "UP" if row["direction"] == "LONG" else "DOWN"
            exit_price, bars_held, reason = _find_exit_for_signal(
                entry_price=entry, direction=dir_for_exit,
                sl_dist=sl_dist, rr=rr,
                bars=future, timeout_bars=horizon,
            )
            if row["direction"] == "LONG":
                gross_pct = exit_price / entry - 1.0
            else:
                gross_pct = -(exit_price / entry - 1.0)
            net_pct = gross_pct - maker_cost
            win = int(gross_pct > 0)

            with conn.cursor() as cur:
                cur.execute(f"""
                    UPDATE `{TABLE}`
                    SET exit_price=%s, exit_reason=%s, bars_held=%s,
                        gross_pct=%s, net_pct_maker=%s, win=%s
                    WHERE id=%s
                """, (
                    float(exit_price), str(reason), int(bars_held),
                    float(gross_pct), float(net_pct), win, row["id"],
                ))
            updated += 1
        conn.commit()
    finally:
        conn.close()

    if updated:
        logger.info("Backfilled %d hybrid_signals outcomes", updated)
    return updated


# Module-level singleton (lazy)
_engine: Optional[HybridInferenceEngine] = None


def get_engine() -> HybridInferenceEngine:
    global _engine
    if _engine is None:
        _engine = HybridInferenceEngine()
    return _engine


def run_hybrid_cycle(features: pd.DataFrame, klines: pd.DataFrame,
                       paused: bool = False) -> Optional[dict]:
    """Public entry for app.py update_cycle integration.
    Returns the emitted signal dict (or None if no trigger)."""
    try:
        init_hybrid_signals_table()
    except Exception as exc:
        logger.warning("hybrid_signals table init failed: %s", exc)
        return None

    engine = get_engine()
    sig = engine.maybe_emit_signal(features, klines)
    if sig is None:
        return None
    try:
        record_hybrid_signal(sig, paused=paused)
    except Exception as exc:
        logger.exception("record_hybrid_signal failed: %s", exc)
    try:
        backfill_hybrid_outcomes(klines)
    except Exception as exc:
        logger.warning("backfill_hybrid_outcomes failed: %s", exc)
    return sig
