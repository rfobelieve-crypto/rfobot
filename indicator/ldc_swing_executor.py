"""
LDC Swing Executor — production runner for jdehorty LDC swing-trade strategy.

Replaces the previous v9+LDC must-agree hybrid (indicator/hybrid_inference.py)
after the user chose Path 2 (jdehorty original) on 2026-05-12.

Strategy (LDC swing with dynamic exits, min hold 4h):
    Entry:
        start_long_trade   -> open LONG  (close at signal-bar's close)
        start_short_trade  -> open SHORT
    Exit (whichever fires first AFTER min_hold_bars=4 from entry):
        is_bearish_cross   -> close LONG  (yhat1 crosses below yhat2)
        is_bullish_cross   -> close SHORT (yhat1 crosses above yhat2)
        opposite start_*_trade -> close current AND open opposite (jdehorty
                                  flip behaviour, also subject to min_hold)
    No TP/SL.  Maker cost assumed (5 bps round-trip).

Stateful — at most one position open at a time, persisted to
`ldc_swing_positions` MySQL table.

Sizing (Stage 1 paper):
    Capital = $1000
    Notional per trade = $1000 (full deployment)
    Leverage = 3x (margin per trade = $333)
    No partial / pyramid for now.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Strategy config ──────────────────────────────────────────────────────────
MIN_HOLD_BARS = 4          # bars after entry before any exit can fire
MAKER_COST = 0.0005         # round-trip 5 bps
LDC_MAX_BARS = 2200         # tail for LDC computation
NOTIONAL_USD = 1000.0       # paper-mode notional
LEVERAGE = 3.0              # user spec 2026-05-12
INITIAL_CAPITAL_USD = 1000.0

# LDC parameters — tuned for BTC 1h via grid search 2026-05-13.
# Best risk-adjusted (cum/|MDD| = 2.36) on 5.5mo backtest:
#   cum +29.4% (1x) / +88.3% (3x), MDD -12.5% (1x) / -37.4% (3x)
#   n=120 trades, WR 42.5%, PF 1.36
# Tuned: kernel_h 8 -> 10, ema_filter ON -> OFF.
# Kept default: k=8, kernel_r=8, vol+regime filters ON, regime_thr=-0.1,
#               3 features (rsi 14/1, wt 10/11, cci 20/1).
LDC_PARAMS = dict(
    neighbors_count=8,
    max_bars_back=2000,
    feature_spec=(("rsi", 14, 1), ("wt", 10, 11), ("cci", 20, 1)),
    use_volatility_filter=True,
    use_regime_filter=True,
    regime_threshold=-0.1,
    use_ema_filter=False,       # CHANGED 2026-05-13 (was True, period=200)
    ema_period=200,             # ignored when use_ema_filter=False
    kernel_h=10,                # CHANGED 2026-05-13 (was 8)
    kernel_r=8.0,
    kernel_x=25,
    kernel_lag=2,
)

# v9 order-flow filter at entry time.  Enabled 2026-05-13 per user request.
# Threshold 0.55 is permissive — skips only entries where v9 has NO bias
# towards the LDC-proposed direction.  Production v9 model covers every
# bar (vs walk-forward OOS which was sparse), so the filter sample limit
# from earlier backtest does not apply at runtime.
V9_FILTER_ENABLED = True
V9_LONG_THRESHOLD = 0.55
V9_SHORT_THRESHOLD = 0.55


# ── DB ───────────────────────────────────────────────────────────────────────

TABLE = "ldc_swing_positions"


def _get_conn():
    from shared.db import get_db_conn
    return get_db_conn()


def init_table() -> None:
    """Idempotent CREATE TABLE."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS `{TABLE}` (
                    `id` INT AUTO_INCREMENT PRIMARY KEY,
                    `entry_time` DATETIME NOT NULL,
                    `direction` VARCHAR(8) NOT NULL,
                    `entry_price` DOUBLE NOT NULL,
                    `status` VARCHAR(8) NOT NULL DEFAULT 'OPEN',
                    `exit_time` DATETIME DEFAULT NULL,
                    `exit_price` DOUBLE DEFAULT NULL,
                    `exit_reason` VARCHAR(16) DEFAULT NULL,
                    `bars_held` INT DEFAULT NULL,
                    `gross_pct` DOUBLE DEFAULT NULL,
                    `net_pct_maker` DOUBLE DEFAULT NULL,
                    `win` TINYINT DEFAULT NULL,
                    `notional_usd` DOUBLE NOT NULL,
                    `leverage` DOUBLE NOT NULL,
                    `paused_at_signal` TINYINT NOT NULL DEFAULT 0,
                    `model_version` VARCHAR(48) DEFAULT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY `uq_entry_time` (`entry_time`),
                    KEY `ix_status` (`status`),
                    KEY `ix_entry_time` (`entry_time`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


def get_open_position() -> Optional[dict]:
    """Return dict of open position or None."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"""
                    SELECT id, entry_time, direction, entry_price,
                           notional_usd, leverage, paused_at_signal
                    FROM `{TABLE}`
                    WHERE status = 'OPEN'
                    ORDER BY entry_time DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return None
                raise
    finally:
        conn.close()
    if not row:
        return None
    row["entry_time"] = pd.Timestamp(row["entry_time"])
    return row


def insert_open(entry_time: datetime, direction: str, entry_price: float,
                paused: bool = False) -> Optional[int]:
    """Insert a new OPEN position. Returns row id or None if duplicate."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT IGNORE INTO `{TABLE}`
                  (entry_time, direction, entry_price, status,
                   notional_usd, leverage, paused_at_signal)
                VALUES (%s, %s, %s, 'OPEN', %s, %s, %s)
            """, (
                entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                direction, float(entry_price),
                NOTIONAL_USD, LEVERAGE, int(paused),
            ))
            inserted = cur.rowcount > 0
            new_id = cur.lastrowid if inserted else None
        conn.commit()
    finally:
        conn.close()
    return new_id


def close_position(position_id: int, exit_time: datetime, exit_price: float,
                    exit_reason: str, direction: str, entry_price: float,
                    bars_held: int, cost: float = MAKER_COST) -> dict:
    """Close an open position; fill outcome columns."""
    if direction == "LONG":
        gross_pct = exit_price / entry_price - 1.0
    else:
        gross_pct = -(exit_price / entry_price - 1.0)
    net_pct = gross_pct - cost
    win = int(gross_pct > 0)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE `{TABLE}` SET
                    status='CLOSED',
                    exit_time=%s, exit_price=%s, exit_reason=%s,
                    bars_held=%s, gross_pct=%s, net_pct_maker=%s, win=%s
                WHERE id=%s
            """, (
                exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                float(exit_price), exit_reason, int(bars_held),
                float(gross_pct), float(net_pct), win,
                int(position_id),
            ))
        conn.commit()
    finally:
        conn.close()
    return {
        "id": position_id, "direction": direction,
        "entry_price": entry_price, "exit_price": exit_price,
        "gross_pct": gross_pct, "net_pct": net_pct,
        "win": win, "bars_held": bars_held, "reason": exit_reason,
    }


# ── LDC computation (cached) ─────────────────────────────────────────────────

# ── v9 order-flow filter ─────────────────────────────────────────────────────

_v9_engine = None


def _get_v9_engine():
    """Lazy-load the hybrid_inference engine which already wraps v9.
    Returns None if v9 model artifacts not present."""
    global _v9_engine
    if _v9_engine is not None:
        return _v9_engine
    try:
        from indicator.hybrid_inference import HybridInferenceEngine
        engine = HybridInferenceEngine()
        if engine.long_model is None:
            logger.warning("ldc_swing: v9 artifacts missing — filter disabled")
            return None
        _v9_engine = engine
        return _v9_engine
    except Exception as exc:
        logger.warning("ldc_swing: failed to load v9 engine: %s", exc)
        return None


def evaluate_v9_for_direction(features: Optional[pd.DataFrame],
                                 direction: str) -> dict:
    """Return dict with p_long, p_short, allow.
    allow=False means v9 vetoes the proposed entry.
    If features missing or v9 unavailable, returns allow=True (no filter)."""
    if not V9_FILTER_ENABLED or features is None or features.empty:
        return {"p_long": None, "p_short": None, "allow": True,
                "reason": "v9_disabled_or_no_features"}
    engine = _get_v9_engine()
    if engine is None:
        return {"p_long": None, "p_short": None, "allow": True,
                "reason": "v9_engine_unavailable"}
    try:
        p_long, p_short = engine.predict_v9(features)
    except Exception as exc:
        logger.warning("v9 predict failed: %s", exc)
        return {"p_long": None, "p_short": None, "allow": True,
                "reason": f"v9_predict_failed: {exc}"}

    if direction == "LONG":
        allow = p_long >= V9_LONG_THRESHOLD
        reason = (f"v9_pass P_long={p_long:.3f}>={V9_LONG_THRESHOLD}"
                   if allow else f"v9_veto P_long={p_long:.3f}<{V9_LONG_THRESHOLD}")
    else:  # SHORT
        allow = p_short >= V9_SHORT_THRESHOLD
        reason = (f"v9_pass P_short={p_short:.3f}>={V9_SHORT_THRESHOLD}"
                   if allow else f"v9_veto P_short={p_short:.3f}<{V9_SHORT_THRESHOLD}")
    return {"p_long": p_long, "p_short": p_short, "allow": allow, "reason": reason}


def compute_ldc_tail(klines: pd.DataFrame) -> pd.DataFrame:
    """Compute LDC signal frame on tail of klines (full recompute, ~5s)."""
    try:
        from research.ldc_python import compute_ldc_signals
    except Exception as exc:
        logger.warning("ldc_swing: cannot import ldc_python: %s", exc)
        return pd.DataFrame()

    tail = klines.tail(LDC_MAX_BARS)
    if len(tail) < 250:
        logger.warning("ldc_swing: not enough klines (have %d need >=250)",
                        len(tail))
        return pd.DataFrame()
    try:
        return compute_ldc_signals(tail, verbose=False, **LDC_PARAMS)
    except Exception as exc:
        logger.exception("ldc_swing: LDC compute failed: %s", exc)
        return pd.DataFrame()


# ── Cycle orchestrator ───────────────────────────────────────────────────────

def run_cycle(klines: pd.DataFrame,
               features: Optional[pd.DataFrame] = None,
               paused: bool = False) -> dict:
    """Run one LDC swing inference cycle.

    features : optional features DataFrame for v9 order-flow filter.
        If provided and V9_FILTER_ENABLED, entries are gated by v9 P(side).
        If None, no v9 filtering occurs (LDC pure mode).

    Returns dict describing what happened:
      {action: "open"|"close"|"reverse"|"hold"|"none"|"v9_veto", ...}
    """
    try:
        init_table()
    except Exception as exc:
        logger.warning("ldc_swing_positions table init failed: %s", exc)
        return {"action": "none", "error": "table init failed"}

    if klines.empty:
        return {"action": "none", "error": "no klines"}

    ldc = compute_ldc_tail(klines)
    if ldc.empty:
        return {"action": "none", "error": "ldc compute failed"}

    bar = ldc.iloc[-1]
    bar_ts = ldc.index[-1]
    if bar_ts.tzinfo is not None:
        bar_ts = bar_ts.tz_convert("UTC").tz_localize(None)
    last_close = float(klines["close"].iloc[-1])

    start_long = bool(bar.get("start_long_trade", False))
    start_short = bool(bar.get("start_short_trade", False))
    is_bullish_cross = bool(bar.get("is_bullish_cross", False))
    is_bearish_cross = bool(bar.get("is_bearish_cross", False))

    pos = get_open_position()

    # ─── Case A: have open position ──
    if pos:
        entry_ts = pos["entry_time"]
        if entry_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_convert("UTC").tz_localize(None)
        bars_held = max(0, int((bar_ts - entry_ts).total_seconds() / 3600))
        past_min_hold = bars_held >= MIN_HOLD_BARS
        direction = pos["direction"]

        exit_reason = None
        if past_min_hold:
            if direction == "LONG":
                if start_short:
                    exit_reason = "reverse"
                elif is_bearish_cross:
                    exit_reason = "dynamic"
            else:  # SHORT
                if start_long:
                    exit_reason = "reverse"
                elif is_bullish_cross:
                    exit_reason = "dynamic"

        if exit_reason is None:
            return {"action": "hold", "position_id": pos["id"],
                    "direction": direction, "bars_held": bars_held,
                    "past_min_hold": past_min_hold}

        # Close
        closed = close_position(
            position_id=pos["id"], exit_time=bar_ts, exit_price=last_close,
            exit_reason=exit_reason, direction=direction,
            entry_price=float(pos["entry_price"]), bars_held=bars_held,
        )
        logger.info("LDC swing CLOSE id=%d %s @ %.2f gross=%+.3f%% net=%+.3f%% reason=%s",
                     pos["id"], direction, last_close,
                     closed["gross_pct"] * 100, closed["net_pct"] * 100, exit_reason)

        # If reverse, immediately open opposite — also gated by v9
        if exit_reason == "reverse":
            new_dir = "SHORT" if start_short else "LONG"
            v9 = evaluate_v9_for_direction(features, new_dir)
            if not v9["allow"]:
                logger.info("LDC swing REVERSE %s vetoed by v9: %s",
                             new_dir, v9["reason"])
                return {"action": "close", "closed": closed,
                        "skipped_reverse": new_dir, "v9": v9}
            new_id = insert_open(bar_ts, new_dir, last_close, paused=paused)
            if new_id:
                logger.info("LDC swing REVERSE OPEN id=%d %s @ %.2f paused=%s v9=%s",
                             new_id, new_dir, last_close, paused, v9["reason"])
                return {"action": "reverse", "closed": closed,
                        "opened_id": new_id, "opened_direction": new_dir,
                        "entry_price": last_close, "paused": paused, "v9": v9}
        return {"action": "close", "closed": closed}

    # ─── Case B: no open position — check for entry ──
    if not (start_long or start_short):
        return {"action": "none"}

    new_dir = "LONG" if start_long else "SHORT"
    v9 = evaluate_v9_for_direction(features, new_dir)
    if not v9["allow"]:
        logger.info("LDC swing %s entry vetoed by v9 at %s: %s",
                     new_dir, bar_ts, v9["reason"])
        return {"action": "v9_veto", "skipped_direction": new_dir,
                "v9": v9, "bar_ts": str(bar_ts), "entry_price": last_close}

    new_id = insert_open(bar_ts, new_dir, last_close, paused=paused)
    if not new_id:
        return {"action": "none", "note": "dup entry timestamp"}
    logger.info("LDC swing OPEN id=%d %s @ %.2f paused=%s v9=%s",
                 new_id, new_dir, last_close, paused, v9["reason"])
    return {"action": "open", "opened_id": new_id,
            "opened_direction": new_dir, "entry_price": last_close,
            "paused": paused, "v9": v9}


# ── Status helper ────────────────────────────────────────────────────────────

def get_status_summary() -> dict:
    """Snapshot for monitoring/dashboard."""
    try:
        init_table()
    except Exception:
        return {"error": "table not available"}

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM `{TABLE}`")
            total = int(cur.fetchone()["n"])
            cur.execute(f"SELECT COUNT(*) AS n FROM `{TABLE}` WHERE status='OPEN'")
            n_open = int(cur.fetchone()["n"])
            cur.execute(f"""
                SELECT id, entry_time, direction, entry_price,
                       notional_usd, leverage
                FROM `{TABLE}` WHERE status='OPEN'
                ORDER BY entry_time DESC LIMIT 1
            """)
            open_row = cur.fetchone()
    finally:
        conn.close()
    return {
        "total_trades": total,
        "n_open": n_open,
        "open_position": open_row,
        "notional_usd": NOTIONAL_USD,
        "leverage": LEVERAGE,
        "min_hold_bars": MIN_HOLD_BARS,
    }
