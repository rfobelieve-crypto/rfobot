"""OKX Stage 3 performance report — powers /okx-perf endpoint.

Pulls live state from:
  - v7_okx_balance_snapshots (current equity)
  - v7_okx_positions (trade history)
  - v7_okx_executor_status (state machine)
  - v7_okx_kill_log (alerts)

Outputs an HTML-friendly text block suitable for both the Flask endpoint
and Telegram /okx-perf command.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

from shared.db import get_db_conn

logger = logging.getLogger(__name__)


# ── Sharpe helpers ────────────────────────────────────────────────────


def per_trade_sharpe(net_pcts: list[float]) -> Optional[float]:
    """Mean / std of per-trade net_pct.  None if n < 2 or std == 0.

    Per-trade (not annualised) so it's robust to low trade frequency.
    """
    n = len(net_pcts)
    if n < 2:
        return None
    mean = sum(net_pcts) / n
    var = sum((x - mean) ** 2 for x in net_pcts) / (n - 1)
    std = math.sqrt(var)
    if std == 0:
        return None
    return mean / std


def annualised_sharpe(net_pcts: list[float],
                       trades_per_year: float) -> Optional[float]:
    """Per-trade Sharpe × sqrt(trades_per_year).

    trades_per_year is observed (not assumed): if cohort is 14 days
    with 10 trades, that's 10 × (365/14) ≈ 261 trades/year basis.
    """
    base = per_trade_sharpe(net_pcts)
    if base is None or trades_per_year <= 0:
        return None
    return base * math.sqrt(trades_per_year)


# ── DB pulls ──────────────────────────────────────────────────────────


def _get_latest_balance() -> Optional[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT ts, total_eq_usd, available_usd, source
                    FROM v7_okx_balance_snapshots
                    ORDER BY ts DESC LIMIT 1
                """)
                return cur.fetchone()
            except Exception as e:
                if "doesn't exist" in str(e).lower():
                    return None
                raise
    finally:
        conn.close()


def _get_executor_state() -> Optional[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT status, last_changed_at, reason, trigger_id "
                    "FROM v7_okx_executor_status WHERE id=1")
                return cur.fetchone()
            except Exception as e:
                if "doesn't exist" in str(e).lower():
                    return None
                raise
    finally:
        conn.close()


def _get_closed_trades() -> list[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT id, entry_time, exit_time, direction, entry_tier,
                           entry_price, exit_price, exit_reason,
                           net_pct, gross_pct, equity_ret_pct, equity_after,
                           notional_usd, size_contracts
                    FROM v7_okx_positions
                    WHERE status IN ('CLOSED', 'DEMOTED')
                    ORDER BY entry_time
                """)
                return list(cur.fetchall())
            except Exception as e:
                if "doesn't exist" in str(e).lower():
                    return []
                raise
    finally:
        conn.close()


def _get_open_position() -> Optional[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT id, entry_time, direction, entry_tier,
                           entry_price, current_stop, atr_at_entry,
                           size_contracts, notional_usd, equity_before
                    FROM v7_okx_positions
                    WHERE status='OPEN' ORDER BY entry_time DESC LIMIT 1
                """)
                return cur.fetchone()
            except Exception as e:
                if "doesn't exist" in str(e).lower():
                    return None
                raise
    finally:
        conn.close()


def _get_recent_kill_log(days: int = 7) -> list[dict]:
    conn = get_db_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute("""
                    SELECT ts, trigger_id, severity, context
                    FROM v7_okx_kill_log
                    WHERE ts >= DATE_SUB(NOW(), INTERVAL %s DAY)
                    ORDER BY ts DESC
                """, (int(days),))
                return list(cur.fetchall())
            except Exception as e:
                if "doesn't exist" in str(e).lower():
                    return []
                raise
    finally:
        conn.close()


# ── Composition ───────────────────────────────────────────────────────


INITIAL_CAPITAL_USD = 155.0   # Stage 3 baseline (2026-06-01)


def compute_okx_summary() -> dict:
    """Aggregate stats for the OKX live cohort.

    Returns a dict suitable for both JSON and templated rendering.
    """
    balance = _get_latest_balance()
    state = _get_executor_state()
    closed = _get_closed_trades()
    open_pos = _get_open_position()
    kills = _get_recent_kill_log(days=7)

    # Trade stats
    n = len(closed)
    wins = sum(1 for t in closed if (t.get("gross_pct") or 0) > 0)
    avg_bps = (sum((t.get("net_pct") or 0) for t in closed) / n * 10000
               if n else 0.0)
    cum_pct = (sum((t.get("net_pct") or 0) for t in closed) * 100
               if n else 0.0)
    cum_equity_pct = (sum((t.get("equity_ret_pct") or 0) for t in closed)
                      if n else 0.0)

    # Sharpe — per-trade + naive annualised by observed cadence
    net_pcts = [float(t.get("net_pct") or 0) for t in closed]
    pt_sharpe = per_trade_sharpe(net_pcts)
    ann_sharpe = None
    if n >= 2 and closed:
        first = closed[0].get("entry_time")
        last = closed[-1].get("exit_time") or closed[-1].get("entry_time")
        if first and last:
            days = max(1.0, (last - first).total_seconds() / 86400)
            trades_per_year = n * (365.0 / days)
            ann_sharpe = annualised_sharpe(net_pcts, trades_per_year)

    current_eq = (float(balance["total_eq_usd"])
                  if balance and balance.get("total_eq_usd") is not None
                  else None)
    eq_pct_from_initial = (
        ((current_eq - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD * 100)
        if current_eq is not None else None
    )

    return {
        "current_equity_usd": current_eq,
        "available_usd": (float(balance["available_usd"])
                          if balance and balance.get("available_usd") is not None
                          else None),
        "balance_age_sec": (
            (datetime.utcnow() - balance["ts"]).total_seconds()
            if balance and balance.get("ts") else None),
        "initial_capital_usd": INITIAL_CAPITAL_USD,
        "eq_pct_from_initial": eq_pct_from_initial,
        "executor_status": (state.get("status") if state else "UNKNOWN"),
        "executor_reason": (state.get("reason") if state else None),
        "executor_changed_at": (state.get("last_changed_at") if state else None),
        "n_closed": n,
        "wins": wins,
        "win_rate_pct": (wins / n * 100 if n else 0.0),
        "avg_net_bps": avg_bps,
        "cum_net_pct": cum_pct,
        "cum_equity_pct": cum_equity_pct,
        "sharpe_per_trade": pt_sharpe,
        "sharpe_annualised": ann_sharpe,
        "open_position": open_pos,
        "recent_trades": closed[-5:],
        "kill_log_7d": kills,
    }


def format_okx_report(summary: dict) -> str:
    """Telegram-friendly HTML report."""
    lines: list[str] = []
    lines.append("<b>OKX LIVE Stage 3 ($100 + 10x)</b>")
    lines.append("")

    # Account
    eq = summary.get("current_equity_usd")
    if eq is not None:
        delta = summary.get("eq_pct_from_initial") or 0.0
        age = summary.get("balance_age_sec") or 0
        lines.append(
            f"💰 Equity: ${eq:.2f}  "
            f"({delta:+.2f}% from ${summary['initial_capital_usd']:.0f})")
        lines.append(f"   Available: ${summary.get('available_usd', 0):.2f}  "
                     f"(updated {age:.0f}s ago)")
    else:
        lines.append("💰 Equity: <i>no balance snapshot yet</i>")
    lines.append("")

    # Executor
    status = summary.get("executor_status") or "UNKNOWN"
    icon = ("🟢" if status == "ACTIVE"
            else "🟡" if status == "HALTED"
            else "🔴")
    lines.append(f"{icon} Executor: <code>{status}</code>")
    if summary.get("executor_reason"):
        lines.append(f"   {summary['executor_reason']}")
    lines.append("")

    # Trade stats
    n = summary.get("n_closed", 0)
    if n == 0:
        lines.append("📊 <i>No closed trades yet</i>")
    else:
        lines.append(
            f"📊 Trades: {n}  WR: {summary['wins']}/{n} = "
            f"{summary['win_rate_pct']:.0f}%")
        lines.append(
            f"   Avg net: {summary['avg_net_bps']:+.1f} bps  "
            f"Cum net: {summary['cum_net_pct']:+.2f}%")
        lines.append(
            f"   Cum equity Δ: {summary['cum_equity_pct']:+.2f}%")
        pt = summary.get("sharpe_per_trade")
        ann = summary.get("sharpe_annualised")
        if pt is not None:
            sharpe_line = f"   Sharpe/trade: {pt:.2f}"
            if ann is not None:
                sharpe_line += f"  (annualised: {ann:.2f})"
            lines.append(sharpe_line)
    lines.append("")

    # Open position
    op = summary.get("open_position")
    if op:
        lines.append(
            f"🔓 Open #{op['id']}: {op['direction']} {op.get('entry_tier','')}")
        lines.append(
            f"   entry ${op['entry_price']:.1f}  "
            f"stop ${op['current_stop']:.1f}  "
            f"size {op['size_contracts']} contracts")
        if op.get('entry_time'):
            held_h = ((datetime.utcnow() - op['entry_time']).total_seconds()
                      / 3600)
            lines.append(f"   held {held_h:.1f}h")
    else:
        lines.append("🔓 <i>No open position</i>")
    lines.append("")

    # Recent trades
    rt = summary.get("recent_trades") or []
    if rt:
        lines.append("📋 Recent trades:")
        for t in rt[-5:]:
            bps = (t.get("net_pct") or 0) * 10000
            sign = "+" if bps >= 0 else ""
            lines.append(
                f"   #{t['id']} {t.get('entry_time','')} {t['direction']} "
                f"→ {t.get('exit_reason','')} {sign}{bps:.0f} bps")
        lines.append("")

    # Kill triggers
    kills = summary.get("kill_log_7d") or []
    if kills:
        lines.append(f"⚠️ Kill log (7d, {len(kills)} entries):")
        for k in kills[:3]:
            lines.append(f"   {k['ts']} {k['trigger_id']} {k['severity']}")
    return "\n".join(lines)


def get_okx_report() -> str:
    """Public entry: compute + format."""
    try:
        summary = compute_okx_summary()
        return format_okx_report(summary)
    except Exception:
        logger.exception("get_okx_report_failed")
        return "❌ OKX report query failed"
