"""
V7 Paper Executor — Stage 1 paper-trading runner for the v7.1 signal-exit
strategy ("V7": pure signal-decay exit + 3xATR trailing stop).

Runs ALONGSIDE ldc_swing_executor (LDC is untouched) as a parallel paper
cohort, so the two strategies can be compared on forward data before any
graduation decision.

Strategy (validated in research/v71_v7_sizing_1x.py):
    Entry:
        v7.1 decodes a Strong/Moderate signal (direction UP or DOWN)
        -> open LONG (UP) / SHORT (DOWN) at the signal bar's close.
    Exit (whichever fires first, checked each closed bar):
        1. 3xATR(14) trailing stop  (intrabar; gap-through filled at bar open)
        2. 72h time cap
        3. opposite v7.1 signal     (decoded direction flips to the opposite)
    No TP.  Taker cost 0.08% round-trip.

Position sizing (fixed-fractional, deployable):
    - 2% of CURRENT equity risked per trade (risk = initial 3xATR stop).
    - notional = min( 1x * equity , 0.02 * equity / stop_pct ).  1x leverage cap.
    - Account compounds; starts at 1000 USDT.

Stateful — at most one position open at a time, persisted to the
`v7_paper_positions` MySQL table.  The trailing-stop state (running extreme +
current stop) is persisted on the open row and updated every cycle.

NOTE on backtest vs live: the research backtest used WF-OOS fold predictions
and entered at next-bar open; live uses the PRODUCTION model's decoded signal
and enters at the signal bar's close (same 1-bar convention as LDC). Paper
results will therefore not exactly match the backtest — that is the point of
Stage 1 forward validation.
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Strategy / sizing config ─────────────────────────────────────────────────
ATR_PERIOD = 14
TRAIL_MULT = 3.0            # 3 x ATR(14) trailing stop
TIME_CAP_HOURS = 72         # safety cap on holding period
TAKER_COST = 0.0008         # round-trip 8 bps
INITIAL_CAPITAL_USD = 1000.0
RISK_FRAC = 0.02            # 2% of equity risked per trade
MAX_LEVERAGE = 1.0          # leverage cap (notional <= 1x equity)

TABLE = "v7_paper_positions"


# ── DB ───────────────────────────────────────────────────────────────────────

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
                    `entry_tier` VARCHAR(12) DEFAULT NULL,
                    `status` VARCHAR(8) NOT NULL DEFAULT 'OPEN',
                    `atr_at_entry` DOUBLE NOT NULL,
                    `stop_dist` DOUBLE NOT NULL,
                    `trail_extreme` DOUBLE NOT NULL,
                    `current_stop` DOUBLE NOT NULL,
                    `size_frac` DOUBLE NOT NULL,
                    `notional_usd` DOUBLE NOT NULL,
                    `equity_before` DOUBLE NOT NULL,
                    `exit_time` DATETIME DEFAULT NULL,
                    `exit_price` DOUBLE DEFAULT NULL,
                    `exit_reason` VARCHAR(16) DEFAULT NULL,
                    `bars_held` INT DEFAULT NULL,
                    `gross_pct` DOUBLE DEFAULT NULL,
                    `net_pct` DOUBLE DEFAULT NULL,
                    `equity_ret_pct` DOUBLE DEFAULT NULL,
                    `equity_after` DOUBLE DEFAULT NULL,
                    `win` TINYINT DEFAULT NULL,
                    `paused_at_signal` TINYINT NOT NULL DEFAULT 0,
                    `model_version` VARCHAR(48) DEFAULT NULL,
                    `created_at` DATETIME DEFAULT CURRENT_TIMESTAMP,
                    `updated_at` DATETIME DEFAULT CURRENT_TIMESTAMP
                        ON UPDATE CURRENT_TIMESTAMP,
                    UNIQUE KEY `uq_entry_time` (`entry_time`),
                    KEY `ix_status` (`status`),
                    KEY `ix_entry_time` (`entry_time`)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
        conn.commit()
    finally:
        conn.close()


def get_open_position() -> Optional[dict]:
    """Return dict of the open position, or None."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"""
                    SELECT id, entry_time, direction, entry_price, entry_tier,
                           atr_at_entry, stop_dist, trail_extreme, current_stop,
                           size_frac, notional_usd, equity_before
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


def get_current_equity() -> float:
    """Latest equity = equity_after of the most recent CLOSED trade, else seed."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"""
                    SELECT equity_after FROM `{TABLE}`
                    WHERE status = 'CLOSED' AND equity_after IS NOT NULL
                    ORDER BY exit_time DESC, id DESC
                    LIMIT 1
                """)
                row = cur.fetchone()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return INITIAL_CAPITAL_USD
                raise
    finally:
        conn.close()
    if not row or row.get("equity_after") is None:
        return INITIAL_CAPITAL_USD
    return float(row["equity_after"])


def insert_open(entry_time: datetime, direction: str, entry_price: float,
                entry_tier: str, atr_at_entry: float, stop_dist: float,
                current_stop: float, size_frac: float, notional_usd: float,
                equity_before: float, paused: bool = False,
                model_version: Optional[str] = None) -> Optional[int]:
    """Insert a new OPEN position. Returns row id, or None if duplicate ts."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT IGNORE INTO `{TABLE}`
                  (entry_time, direction, entry_price, entry_tier, status,
                   atr_at_entry, stop_dist, trail_extreme, current_stop,
                   size_frac, notional_usd, equity_before,
                   paused_at_signal, model_version)
                VALUES (%s, %s, %s, %s, 'OPEN',
                        %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                direction, float(entry_price), entry_tier,
                float(atr_at_entry), float(stop_dist),
                float(entry_price),               # trail_extreme starts at entry
                float(current_stop), float(size_frac), float(notional_usd),
                float(equity_before), int(paused), model_version,
            ))
            inserted = cur.rowcount > 0
            new_id = cur.lastrowid if inserted else None
        conn.commit()
    finally:
        conn.close()
    return new_id


def update_trail(position_id: int, trail_extreme: float,
                 current_stop: float) -> None:
    """Persist the ratcheted trailing-stop state on an open position."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE `{TABLE}`
                SET trail_extreme=%s, current_stop=%s
                WHERE id=%s AND status='OPEN'
            """, (float(trail_extreme), float(current_stop), int(position_id)))
        conn.commit()
    finally:
        conn.close()


def close_position(pos: dict, exit_time: datetime, exit_price: float,
                   exit_reason: str, bars_held: int) -> dict:
    """Close an open position; fill outcome + compounded equity columns."""
    direction = pos["direction"]
    entry_price = float(pos["entry_price"])
    size_frac = float(pos["size_frac"])
    equity_before = float(pos["equity_before"])

    if direction == "LONG":
        gross_pct = exit_price / entry_price - 1.0
    else:
        gross_pct = -(exit_price / entry_price - 1.0)
    net_pct = gross_pct - TAKER_COST
    equity_ret = size_frac * net_pct                 # 1x cap: size_frac <= 1
    equity_after = max(equity_before * (1.0 + equity_ret), 0.0)
    win = int(gross_pct > 0)

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE `{TABLE}` SET
                    status='CLOSED',
                    exit_time=%s, exit_price=%s, exit_reason=%s, bars_held=%s,
                    gross_pct=%s, net_pct=%s, equity_ret_pct=%s,
                    equity_after=%s, win=%s
                WHERE id=%s
            """, (
                exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                float(exit_price), exit_reason, int(bars_held),
                float(gross_pct), float(net_pct), float(equity_ret * 100.0),
                float(equity_after), win, int(pos["id"]),
            ))
        conn.commit()
    finally:
        conn.close()
    return {
        "id": pos["id"], "direction": direction,
        "entry_price": entry_price, "exit_price": exit_price,
        "gross_pct": gross_pct, "net_pct": net_pct,
        "equity_ret_pct": equity_ret * 100.0, "equity_after": equity_after,
        "win": win, "bars_held": bars_held, "reason": exit_reason,
    }


def fetch_positions_for_chart(start_dt: datetime, end_dt: datetime) -> list:
    """Positions whose holding window intersects [start_dt, end_dt].

    For chart overlay — includes closed trades and the currently-open one.
    Times are naive UTC. Returns [] if the table does not exist yet.
    """
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"""
                    SELECT entry_time, direction, entry_price, entry_tier,
                           status, exit_time, exit_price, exit_reason, win
                    FROM `{TABLE}`
                    WHERE entry_time <= %s
                      AND (exit_time IS NULL OR exit_time >= %s)
                    ORDER BY entry_time ASC
                """, (end_dt.strftime("%Y-%m-%d %H:%M:%S"),
                      start_dt.strftime("%Y-%m-%d %H:%M:%S")))
                rows = cur.fetchall()
            except Exception as exc:
                if "doesn't exist" in str(exc).lower():
                    return []
                raise
    finally:
        conn.close()
    return list(rows or [])


# ── ATR ──────────────────────────────────────────────────────────────────────

def _atr_wilder(klines: pd.DataFrame, period: int = ATR_PERIOD) -> float:
    """Wilder ATR of the latest bar (matches Pine ta.atr / RMA of true range)."""
    high = klines["high"].astype(float)
    low = klines["low"].astype(float)
    close = klines["close"].astype(float)
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()],
                   axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    val = atr.iloc[-1]
    return float(val) if np.isfinite(val) else 0.0


# ── Cycle orchestrator ───────────────────────────────────────────────────────

def run_cycle(klines: pd.DataFrame, signal_direction: str,
              signal_strength: str = "", paused: bool = False,
              model_version: Optional[str] = None) -> dict:
    """Run one V7 paper cycle on the latest closed bar.

    signal_direction : v7.1 decoded direction for the latest bar
                       ("UP" / "DOWN" / "NEUTRAL").
    signal_strength  : "Strong" / "Moderate" / "Weak" (stored for reporting;
                       not gated on — direction != NEUTRAL is the signal).

    Returns a dict: {action: "open"|"close"|"hold"|"none", ...}.
    """
    try:
        init_table()
    except Exception as exc:
        logger.warning("v7_paper_positions table init failed: %s", exc)
        return {"action": "none", "error": "table init failed"}

    if klines is None or klines.empty or len(klines) < ATR_PERIOD + 2:
        return {"action": "none", "error": "insufficient klines"}

    bar_ts = klines.index[-1]
    if getattr(bar_ts, "tzinfo", None) is not None:
        bar_ts = bar_ts.tz_convert("UTC").tz_localize(None)
    bar_open = float(klines["open"].iloc[-1])
    bar_high = float(klines["high"].iloc[-1])
    bar_low = float(klines["low"].iloc[-1])
    last_close = float(klines["close"].iloc[-1])

    direction_up = signal_direction == "UP"
    direction_dn = signal_direction == "DOWN"

    pos = get_open_position()

    # ─── Case A: open position — manage exit / trailing stop ──
    if pos:
        side = pos["direction"]                       # LONG / SHORT
        entry_ts = pos["entry_time"]
        if getattr(entry_ts, "tzinfo", None) is not None:
            entry_ts = entry_ts.tz_convert("UTC").tz_localize(None)
        bars_held = max(0, int((bar_ts - entry_ts).total_seconds() / 3600))
        stop_dist = float(pos["stop_dist"])
        # stop level for THIS bar is computed from the extreme observed
        # through the PREVIOUS bar (persisted) — no look-ahead.
        prev_extreme = float(pos["trail_extreme"])
        if side == "LONG":
            cur_stop = prev_extreme - stop_dist
        else:
            cur_stop = prev_extreme + stop_dist

        exit_price = exit_reason = None
        # 1) trailing stop — intrabar; gap-through filled at bar open
        if side == "LONG" and bar_low <= cur_stop:
            exit_price = min(cur_stop, bar_open)
            exit_reason = "trail_stop"
        elif side == "SHORT" and bar_high >= cur_stop:
            exit_price = max(cur_stop, bar_open)
            exit_reason = "trail_stop"
        # 2) time cap
        if exit_reason is None and bars_held >= TIME_CAP_HOURS:
            exit_price, exit_reason = last_close, "time_cap"
        # 3) opposite v7.1 signal
        if exit_reason is None:
            opp = (side == "LONG" and direction_dn) or \
                  (side == "SHORT" and direction_up)
            if opp:
                exit_price, exit_reason = last_close, "opp_signal"

        if exit_reason is None:
            # hold — ratchet the trailing extreme with this bar
            if side == "LONG":
                new_extreme = max(prev_extreme, bar_high)
                new_stop = new_extreme - stop_dist
            else:
                new_extreme = min(prev_extreme, bar_low)
                new_stop = new_extreme + stop_dist
            if new_extreme != prev_extreme:
                update_trail(pos["id"], new_extreme, new_stop)
            return {"action": "hold", "position_id": pos["id"],
                    "direction": side, "bars_held": bars_held,
                    "current_stop": new_stop}

        closed = close_position(pos, bar_ts, exit_price, exit_reason, bars_held)
        logger.info("V7 paper CLOSE id=%d %s @ %.2f net=%+.3f%% "
                    "eq_ret=%+.3f%% equity=%.2f reason=%s",
                    pos["id"], side, exit_price, closed["net_pct"] * 100,
                    closed["equity_ret_pct"], closed["equity_after"],
                    exit_reason)
        return {"action": "close", "closed": closed}

    # ─── Case B: flat — check for entry ──
    if not (direction_up or direction_dn):
        return {"action": "none"}

    atr = _atr_wilder(klines, ATR_PERIOD)
    if atr <= 0 or not np.isfinite(atr):
        return {"action": "none", "error": "atr unavailable"}

    side = "LONG" if direction_up else "SHORT"
    stop_dist = TRAIL_MULT * atr
    stop_pct = stop_dist / last_close
    size_frac = min(MAX_LEVERAGE, RISK_FRAC / stop_pct) if stop_pct > 0 else MAX_LEVERAGE
    equity = get_current_equity()
    notional = size_frac * equity
    current_stop = (last_close - stop_dist if side == "LONG"
                    else last_close + stop_dist)
    tier = signal_strength if signal_strength in ("Strong", "Moderate") else "Moderate"

    new_id = insert_open(
        entry_time=bar_ts, direction=side, entry_price=last_close,
        entry_tier=tier, atr_at_entry=atr, stop_dist=stop_dist,
        current_stop=current_stop, size_frac=size_frac, notional_usd=notional,
        equity_before=equity, paused=paused, model_version=model_version)
    if not new_id:
        return {"action": "none", "note": "dup entry timestamp"}
    logger.info("V7 paper OPEN id=%d %s @ %.2f tier=%s size=%.0f%% "
                "notional=%.0f equity=%.2f stop=%.2f paused=%s",
                new_id, side, last_close, tier, size_frac * 100,
                notional, equity, current_stop, paused)
    return {"action": "open", "opened_id": new_id, "opened_direction": side,
            "entry_price": last_close, "entry_tier": tier,
            "size_frac": size_frac, "notional_usd": notional,
            "current_stop": current_stop, "paused": paused}


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
    finally:
        conn.close()
    return {
        "total_trades": total,
        "n_open": n_open,
        "equity": get_current_equity(),
        "initial_capital": INITIAL_CAPITAL_USD,
        "risk_frac": RISK_FRAC,
        "max_leverage": MAX_LEVERAGE,
        "trail_mult": TRAIL_MULT,
        "time_cap_hours": TIME_CAP_HOURS,
    }
